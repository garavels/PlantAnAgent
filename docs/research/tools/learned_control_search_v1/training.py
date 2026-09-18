"""Cross-entropy search over14hybrid actor biases using exact weekly reward.

This is derivative-free policy optimization, not PPO. PPO checkpoints are only
the shared serialization/inference container. No reward or physical rule changes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import copy
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from docs.research.tools.learned_control_v1 import environment, imitation
from docs.research.tools.learned_control_hybrid_v1 import training as hybrid
from docs.research.tools.learned_control_hybrid_v1.policy import HybridActorCriticPolicy, distribution_identity
from docs.research.tools.wholeplant_ppo_v1 import training as common

SCHEMA = "learned-control-cem-bias-search-v1"
CONDITIONS = ("ordinary", "pv6", "pv18", "pv36", "meth24", "meth48")
WEIGHTS = {"ordinary": .5, "pv6": .1, "pv18": .1, "pv36": .1, "meth24": .1, "meth48": .1}
file_record, publish_json, sha, now = common.file_record, common.publish_json, common.sha, common.now
_WORKER = None


def default_recipe(**changes):
    recipe = {"parent_checkpoint": None, "training_dataset": None, "search_seed": 3901, "optimizer_seed": 3901,
        "generations": 4, "population": 16, "antithetic_pairs": 7, "elite_count": 4,
        "smoothing": .5, "initial_std": 1., "minimum_std": .15,
        "gate_parameter_scale": 2., "mean_scale_multiplier": .03,
        "mean_scale_floor": .001, "mean_scale_cap": .05,
        "training_seed_pairs": [[50_300_000 + 2 * i, 50_300_001 + 2 * i] for i in range(4)],
        "validation_seeds": list(range(50_310_000, 50_310_004)), "condition_weights": dict(WEIGHTS),
        "workers": 4, "deadline_utc": "2026-09-18T02:00:00Z", "extra_source_paths": []}
    unknown = set(changes) - set(recipe)
    if unknown:
        raise ValueError(f"Unknown search recipe fields:{sorted(unknown)}")
    recipe.update(changes)
    if "search_seed" in changes and "optimizer_seed" not in changes:
        recipe["optimizer_seed"] = changes["search_seed"]
    return recipe


def _validate(recipe):
    if set(recipe) != set(default_recipe()):
        raise ValueError("Search recipe fields differ")
    common._deadline(recipe["deadline_utc"])
    if recipe["optimizer_seed"] != recipe["search_seed"]:
        raise ValueError("Optimizer seed is the evaluator-compatible alias of the actual CEM search seed")
    if not recipe["parent_checkpoint"] or not recipe["training_dataset"]:
        raise ValueError("Parent checkpoint and authenticated training action labels are required")
    if recipe["population"] != 2 * recipe["antithetic_pairs"] + 2 or recipe["antithetic_pairs"] < 1:
        raise ValueError("Population must contain antithetic pairs,mean,and incumbent")
    if not 1 <= recipe["elite_count"] <= recipe["population"] or not 0 < recipe["smoothing"] <= 1:
        raise ValueError("Invalid elite update settings")
    if not 0 < recipe["minimum_std"] <= recipe["initial_std"] or recipe["gate_parameter_scale"] <= 0:
        raise ValueError("Invalid normalized search spread")
    if not 0 < recipe["mean_scale_floor"] <= recipe["mean_scale_cap"] or recipe["mean_scale_multiplier"] <= 0:
        raise ValueError("Invalid ON-mean parameter scales")
    if recipe["condition_weights"] != WEIGHTS:
        raise ValueError("Preserve the registered0.5ordinary+0.1perstress objective")
    pairs = recipe["training_seed_pairs"]
    if recipe["generations"] < 1 or len(pairs) != recipe["generations"] or any(len(pair) != 2 for pair in pairs):
        raise ValueError("Each generation requires exactly two prospectively chosen seeds")
    seeds = [seed for pair in pairs for seed in pair] + recipe["validation_seeds"]
    if not recipe["validation_seeds"] or len(set(seeds)) != len(seeds):
        raise ValueError("Search and reporting-validation seeds must be nonempty and disjoint")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or not environment.TRAIN_BASE <= seed < environment.TRAIN_STOP for seed in seeds):
        raise ValueError("Every search and reporting-validation seed must be training-only")
    if recipe["workers"] < 1:
        raise ValueError("At least one local worker is required")


def _pins(recipe):
    paths = {Path(row["path"]).resolve() for row in environment.source_records()}
    paths.update(Path(hybrid.__file__).parent.glob("*.py"))
    paths.update((Path(common.__file__), Path(imitation.__file__), Path(imitation.__file__).with_name("training.py")))
    paths.update(Path(__file__).parent.glob("*.py"))
    paths.update(Path(path).resolve() for path in recipe["extra_source_paths"])
    paths.add(ROOT / "uv.lock")
    return [file_record(path) for path in sorted(paths)]


def derive_parameter_scales(actions, recipe):
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all() or np.any(abs(actions) > 1):
        raise ValueError("Expected finite originalBoxtraining-action labels")
    counts, conditional_mean, scales = [], [], []
    for coordinate in range(7):
        selected = actions[:, coordinate] > -1.
        counts.append(int(selected.sum()))
        mean = float(np.mean(actions[selected, coordinate] + 1)) if selected.any() else 0.
        conditional_mean.append(mean)
        scales.append(float(np.clip(recipe["mean_scale_multiplier"] * mean, recipe["mean_scale_floor"], recipe["mean_scale_cap"])))
    return {"gate_scales": [recipe["gate_parameter_scale"]] * 7, "on_mean_scales": scales,
        "all_scales": [recipe["gate_parameter_scale"]] * 7 + scales,
        "teacher_on_counts": counts, "teacher_mean_action_plus_one_given_on": conditional_mean,
        "derivation": "Authenticated training teacher labels only; ON means action>-1. Scale is recipe.mean_scale_multiplier * conditional mean(action+1), with declared scale floor/cap. These are search coordinates, not action or parameter bounds."}


def unchanged_tensor_hash(model):
    tensors = []
    for name, tensor in model.policy.state_dict().items():
        tensors.append((name, tensor[14:] if name == "action_net.bias" else tensor))
    return common._tensor_hash(tensors)


def freeze_plan(root, recipe):
    recipe = copy.deepcopy(recipe)
    _validate(recipe)
    parent = Path(recipe["parent_checkpoint"]).resolve()
    dataset = Path(recipe["training_dataset"]).resolve()
    imitation.require_dataset(dataset)
    with np.load(dataset, allow_pickle=False) as data:
        scales = derive_parameter_scales(data["actions"], recipe)
    policy = hybrid.load_checkpoint(parent)
    try:
        if type(policy.model.policy) is not HybridActorCriticPolicy or policy.model.policy.action_net.bias.shape != (21,):
            raise ValueError("Expected the registered seven-gate/seven-mean/seven-scale hybrid head")
        base_bias = policy.model.policy.action_net.bias.detach().cpu().tolist()
        preserved_hash = unchanged_tensor_hash(policy.model)
        parent_hash = common.policy_parameter_sha256(policy.model)
        optimizer_hash = common.optimizer_sha256(policy.model)
        parent_seed = int(policy.model.seed)
    finally:
        policy.close()
    metadata = json.loads(dataset.with_suffix(".json").read_text())
    dependencies = [file_record(parent.with_suffix(suffix)) for suffix in (".zip", ".json", ".vecnormalize.pkl")]
    dependencies += [file_record(dataset), file_record(dataset.with_suffix(".json")), file_record(Path(metadata["plan"]["path"]))]
    expected_search = recipe["generations"] * recipe["population"] * 2 * len(CONDITIONS)
    expected_validation = 2 * len(recipe["validation_seeds"]) * len(CONDITIONS)
    plan = {"schema": SCHEMA, "created_at": now(), "recipe": recipe,
        "optimizer_family": "Cross-entropy method parameter-space policy search; no gradient updates or PPO optimization",
        "environment_identity": environment.environment_identity(), "policy_distribution": distribution_identity(),
        "source_files": _pins(recipe), "dependencies": dependencies, "runtime": common.runtime_record(),
        "parameterization": scales, "parent_full_bias": base_bias,
        "unchanged_tensor_sha256": preserved_hash, "parent_policy_sha256": parent_hash,
        "parent_optimizer_sha256": optimizer_hash,
        "parent_training_seed": parent_seed,
        "search_parameters": "Absolute normalized14Doffsets from parent action_net.bias[0:14]; bias[14:21]and every other tensor are unchanged",
        "objective": "0.5*mean actual weekly scalar economic reward on ordinary +0.1*mean on each existing stress condition; no other reward",
        "comparison": "All candidates share the same 12 weeks within a generation; incumbent is reevaluated. No ranking compares scores from different generations.",
        "selection": "Highest current-generation objective; ties prefer retained incumbent, then candidate index. The registered elite_count updates mean/std with smoothing. No incomplete panel can rank.",
        "validation": "Parent and final incumbent on fixed fresh training seeds; reporting only,never checkpoint selection",
        "deadline": "Workers stop at the next physical decision and parent cancels unscheduled cases; incomplete panels do not rank or produce a completed-final label",
        "expected_rollouts": {"search": expected_search, "validation": expected_validation, "total": expected_search + expected_validation,
            "decisions": (expected_search + expected_validation) * 672, "physical_minutes": (expected_search + expected_validation) * 10080}}
    publish_json(Path(root) / "plan.json", plan)
    return plan


def require_plan(root):
    plan = json.loads((Path(root) / "plan.json").read_text())
    _validate(plan["recipe"])
    if plan["schema"] != SCHEMA or plan["source_files"] != _pins(plan["recipe"]) or plan["environment_identity"] != environment.environment_identity() or plan["runtime"] != common.runtime_record():
        raise ValueError("Frozen CEM source,environment,or runtime changed")
    for row in plan["dependencies"]:
        if file_record(Path(row["path"])) != row:
            raise ValueError("Frozen CEM dependency changed")
    return plan


def generation_candidates(mean, std, incumbent, rng, recipe):
    noise = rng.normal(size=(recipe["antithetic_pairs"], 14))
    candidates = []
    for index, draw in enumerate(noise):
        for sign in (1, -1):
            candidates.append({"index": len(candidates), "role": "antithetic", "pair": index, "sign": sign,
                "parameters": (mean + sign * std * draw).tolist()})
    candidates.append({"index": len(candidates), "role": "mean", "parameters": np.asarray(mean).tolist()})
    candidates.append({"index": len(candidates), "role": "incumbent", "parameters": np.asarray(incumbent).tolist()})
    return candidates


def panel_cases(seeds):
    return [{"condition": condition, "seed": int(seed)} for seed in seeds for condition in CONDITIONS]


def weighted_objective(rows, cases, weights=WEIGHTS):
    expected = {(case["condition"], case["seed"]) for case in cases}
    actual = [(row["condition"], row["seed"]) for row in rows]
    if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != expected or any(row["status"] != "completed" for row in rows):
        raise ValueError("Only a complete matched case panel can receive a score")
    means = {condition: float(np.mean([r["raw_reward_eur"] for r in rows if r["condition"] == condition])) for condition in CONDITIONS}
    return {"weighted_reward_eur": float(sum(weights[c] * means[c] for c in CONDITIONS)), "condition_mean_reward_eur": means,
        "mean_unmet_kg": float(np.mean([r["metrics"]["unmet_kg"] for r in rows])),
        "weeks_with_total_unmet_below_1e_minus_6_kg": sum(r["metrics"]["unmet_kg"] < 1e-6 for r in rows), "weeks": len(rows)}


def update_distribution(candidates, scores, mean, std, recipe):
    ranked = sorted(candidates, key=lambda c: (-scores[c["index"]]["weighted_reward_eur"], c["role"] != "incumbent", c["index"]))
    elites = np.asarray([row["parameters"] for row in ranked[:recipe["elite_count"]]], dtype=np.float64)
    alpha = recipe["smoothing"]
    next_mean = (1 - alpha) * mean + alpha * elites.mean(axis=0)
    next_std = np.maximum(recipe["minimum_std"], (1 - alpha) * std + alpha * elites.std(axis=0, ddof=0))
    return next_mean, next_std, np.asarray(ranked[0]["parameters"], dtype=np.float64), [c["index"] for c in ranked[:recipe["elite_count"]]], ranked[0]["index"]


def apply_parameters(model, parent_bias, parameters, scales):
    values = np.asarray(parameters, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    if values.shape != (14,) or scales.shape != (14,) or not np.isfinite(values).all() or not np.isfinite(scales).all():
        raise ValueError("Candidate coordinates and scales must be finite14-vectors")
    bias = torch.as_tensor(parent_bias, dtype=model.policy.action_net.bias.dtype)
    with torch.no_grad():
        model.policy.action_net.bias.copy_(bias)
        model.policy.action_net.bias[:14].add_(torch.as_tensor(values * scales, dtype=bias.dtype))


def _worker_initializer(plan):
    global _WORKER
    common.configure_torch()
    policy = hybrid.load_checkpoint(Path(plan["recipe"]["parent_checkpoint"]))
    _WORKER = {"plan": plan, "policy": policy, "deadline": common._deadline(plan["recipe"]["deadline_utc"])}


def _evaluate_case(task):
    if _WORKER is None:
        raise RuntimeError("CEM worker is not initialized")
    plan, policy, deadline = (_WORKER[key] for key in ("plan", "policy", "deadline"))
    started, env, decisions, reward_sum, info = time.monotonic(), None, 0, 0., {}
    row = {key: task[key] for key in ("phase", "generation", "candidate", "condition", "seed")}
    row.update(parameters=task["parameters"], worker_pid=os.getpid(), started_at=now())
    try:
        if datetime.now(timezone.utc) >= deadline:
            row.update(status="deadline_not_started", decisions=0, raw_reward_eur=None)
            return row
        apply_parameters(policy.model, plan["parent_full_bias"], task["parameters"], plan["parameterization"]["all_scales"])
        if unchanged_tensor_hash(policy.model) != plan["unchanged_tensor_sha256"]:
            raise ValueError("A nonsearched policy tensor changed")
        env = environment.FixedEnv(condition=task["condition"], split="train", record_minutes=False)
        observation, _ = env.reset(seed=task["seed"])
        done = False
        while not done:
            if datetime.now(timezone.utc) >= deadline:
                row["status"] = "deadline_incomplete"
                break
            action = policy.predict(observation, deterministic=True)[0]
            if action.shape != (7,) or not np.isfinite(action).all() or np.any(abs(action) > 1):
                raise ValueError("Candidate actor output violates the original public interface")
            observation, reward, terminated, truncated, info = env.step(action)
            decisions += 1
            reward_sum += float(reward)
            done = terminated or truncated
        if done:
            if decisions != 672 or info["metrics"]["physical_steps"] != 10080:
                raise ValueError("Candidate did not execute the unchanged full seven-day mission")
            if not np.isclose(reward_sum, info["economic_metrics"]["net_operating_value_eur"], rtol=0, atol=1e-6):
                raise ValueError("Scalar objective and reported economics disagree")
            row["status"] = "completed"
        row.update(decisions=decisions, physical_steps=decisions * 15, raw_reward_eur=reward_sum,
            economic_metrics=info.get("economic_metrics"), metrics=info.get("metrics"))
        return row
    except BaseException as exc:
        row.update(status="failed", decisions=decisions, physical_steps=decisions * 15,
            raw_reward_eur=reward_sum, error_type=type(exc).__name__, error=str(exc))
        return row
    finally:
        if env is not None:
            env.close()
        row["elapsed_s"], row["finished_at"] = time.monotonic() - started, now()


def _task_path(root, task):
    return root / "cases" / task["phase"] / f"generation-{task['generation']:03d}" / f"candidate-{task['candidate']:03d}" / f"{task['condition']}--{task['seed']}.json"


def execute_panel(executor, tasks, root, recipe):
    """Bound in-flight work; preserve every completed,failed,and unstarted task."""
    deadline = common._deadline(recipe["deadline_utc"]).timestamp()
    pending, result, next_index, submission_error = {}, [], 0, None
    while next_index < len(tasks) or pending:
        while next_index < len(tasks) and len(pending) < recipe["workers"] and time.time() < deadline and submission_error is None:
            task = tasks[next_index]
            try:
                pending[executor.submit(_evaluate_case, task)] = task
            except BaseException as exc:
                submission_error = {"error_type": type(exc).__name__, "error": str(exc)}
                break
            next_index += 1
        if not pending:
            break
        completed, _ = wait(pending, timeout=.5, return_when=FIRST_COMPLETED)
        for future in completed:
            task = pending.pop(future)
            try:
                row = future.result()
            except BaseException as exc:
                row = {**task, "status": "worker_failed", "decisions": None, "physical_steps": None,
                    "error_type": type(exc).__name__, "error": str(exc), "finished_at": now()}
            publish_json(_task_path(root, task), row)
            result.append(row)
        if time.time() > deadline + 5 and pending:
            for future, task in pending.items():
                future.cancel()
                row = {**task, "status": "hard_deadline_terminated", "decisions": None,
                    "physical_steps": None, "finished_at": now(), "error": "Worker did not return within5seconds of the decision-level deadline;partialtransitioncount unavailable"}
                publish_json(_task_path(root, task), row)
                result.append(row)
            for process in list((getattr(executor, "_processes", None) or {}).values()):
                if process.is_alive():
                    process.terminate()
            pending.clear()
            break
    for task in tasks[next_index:]:
        row = {**task, "status": "submission_failed_not_started" if submission_error else "deadline_not_started",
            "decisions": 0, "physical_steps": 0, "finished_at": now(), **(submission_error or {})}
        publish_json(_task_path(root, task), row)
        result.append(row)
    return result


def save_checkpoint(policy, path, *, plan, parameters, kind, ledger, generation):
    from stable_baselines3 import PPO
    path = Path(path)
    if any(path.with_suffix(suffix).exists() for suffix in (".zip", ".json", ".vecnormalize.pkl")):
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = policy.model
    apply_parameters(model, plan["parent_full_bias"], parameters, plan["parameterization"]["all_scales"])
    if unchanged_tensor_hash(model) != plan["unchanged_tensor_sha256"] or common.optimizer_sha256(model) != plan["parent_optimizer_sha256"]:
        raise ValueError("Search changed a nonsearched tensor or gradient optimizer state")
    model.seed = plan["recipe"]["search_seed"]
    model.research_identity, model.research_plan_sha256 = plan["environment_identity"], sha(plan)
    model.research_search = {"schema": SCHEMA, "parameters": list(parameters), "generation": generation,
        "algorithm": "CEM", "gradient_updates_this_run": 0, "ledger": ledger}
    temporary = path.with_name(path.stem + ".writing.zip")
    model.save(temporary)
    temporary.rename(path)
    normalizer_path = path.with_suffix(".vecnormalize.pkl")
    with normalizer_path.open("xb") as stream:
        stream.write(Path(plan["recipe"]["parent_checkpoint"]).with_suffix(".vecnormalize.pkl").read_bytes())
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        reopened = PPO.load(path, device="cpu")
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
    hashes = {"policy_tensor_sha256": common.policy_parameter_sha256(model),
        "actor_tensor_sha256": common.actor_parameter_sha256(model),
        "actor_mean_tensor_sha256": common.actor_mean_parameter_sha256(model),
        "optimizer_state_sha256": common.optimizer_sha256(model)}
    if common.policy_parameter_sha256(reopened) != hashes["policy_tensor_sha256"] or common.optimizer_sha256(reopened) != hashes["optimizer_state_sha256"] or reopened.research_plan_sha256 != sha(plan):
        raise ValueError("CEM checkpoint did not reopen exactly in the sharedPPOcontainer")
    probe = np.random.default_rng(913).normal(size=(32, 149)).astype(np.float32)
    if not np.array_equal(model.predict(probe, deterministic=True)[0], reopened.predict(probe, deterministic=True)[0]):
        raise ValueError("Saved CEM actor predictions changed")
    record = {"schema": SCHEMA, "kind": kind, "created_at": now(), **hashes,
        "optimizer_family": "CEM parameter-space search", "container_format": "SB3PPOserialization for shared actor inference only",
        "optimizer_algorithm": "cross_entropy_search", "parent_training_seed": plan["parent_training_seed"],
        "gradient_updates_this_run": 0, "ppo_updates_this_run": 0,
        "optimizer_seed": model.seed, "num_timesteps": model.num_timesteps,
        "ppo_update_count_inherited_from_parent": model._n_updates,
        "timestep_field_semantics": "Inherited parent gradient-training counter; actual search/validation effort is recorded separately insearch_ledger",
        "search_ledger": ledger, "generation": generation, "parameters": list(parameters),
        "plan_sha256": sha(plan), "environment_identity": plan["environment_identity"],
        "policy_distribution": distribution_identity(), "checkpoint": file_record(path), "normalizer": file_record(normalizer_path),
        "unchanged_tensor_sha256": unchanged_tensor_hash(model), "normalizer_bytes_identical_to_parent": True,
        "checkpoint_reopen_verified": True, "deterministic_prediction_reopen_verified": True,
        "gaussian_log_std": None}
    publish_json(path.with_suffix(".json"), record)
    checked = common.load_checkpoint(path)
    checked.close()
    return record


def effort_ledger(rows):
    known = [r for r in rows if r.get("decisions") is not None]
    return {"scheduled_rollouts": len(rows), "completed_rollouts": sum(r["status"] == "completed" for r in rows),
        "started_rollouts": sum((r.get("decisions") or 0) > 0 for r in rows),
        "physical_decisions": sum(r.get("decisions", 0) for r in known),
        "physical_minutes": sum(r.get("physical_steps", 0) for r in known),
        "unknown_partial_rollouts": sum(r.get("decisions") is None for r in rows),
        "gradient_updates": 0, "ppo_updates": 0}


def train(root):
    root = Path(root).resolve()
    plan = require_plan(root)
    recipe = plan["recipe"]
    if datetime.now(timezone.utc) >= common._deadline(recipe["deadline_utc"]):
        raise ValueError("Search deadline already passed")
    common.configure_torch()
    out = root / "training"
    out.mkdir(exist_ok=False)
    publish_json(out / "attempt.json", {"started_at": now(), "pid": os.getpid(), "plan_sha256": sha(plan)})
    rng = np.random.default_rng(recipe["search_seed"])
    mean, std, incumbent = np.zeros(14), np.full(14, recipe["initial_std"]), np.zeros(14)
    all_rows, generation_results, started = [], [], time.monotonic()
    policy = hybrid.load_checkpoint(Path(recipe["parent_checkpoint"]))
    executor = ProcessPoolExecutor(max_workers=recipe["workers"], mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_worker_initializer, initargs=(plan,))
    try:
        for generation in range(recipe["generations"]):
            require_plan(root)
            rng_before = copy.deepcopy(rng.bit_generator.state)
            candidates = generation_candidates(mean, std, incumbent, rng, recipe)
            cases = panel_cases(recipe["training_seed_pairs"][generation])
            prospective = {"generation": generation, "registered_at": now(), "rng_before": rng_before,
                "mean_before": mean.tolist(), "std_before": std.tolist(), "incumbent_before": incumbent.tolist(),
                "candidates": candidates, "cases": cases, "plan_sha256": sha(plan)}
            publish_json(out / f"generation-{generation:03d}-candidates.json", prospective)
            tasks = [{"phase": "search", "generation": generation, "candidate": candidate["index"],
                      "parameters": candidate["parameters"], **case} for candidate in candidates for case in cases]
            rows = execute_panel(executor, tasks, root, recipe)
            all_rows.extend(rows)
            if any(row["status"] != "completed" for row in rows):
                publish_json(out / f"generation-{generation:03d}-incomplete.json", {"generation": generation,
                    "status_counts": {status: sum(r["status"] == status for r in rows) for status in {r["status"] for r in rows}},
                    "ranking_performed": False, "ledger": effort_ledger(rows)})
                break
            scores = {c["index"]: weighted_objective([r for r in rows if r["candidate"] == c["index"]], cases, recipe["condition_weights"]) for c in candidates}
            mean, std, incumbent, elites, winner = update_distribution(candidates, scores, mean, std, recipe)
            state = {"generation": generation, "completed_at": now(), "scores": scores,
                "elite_candidate_indices": elites, "winner": winner, "incumbent": incumbent.tolist(),
                "incumbent_current_panel_score_eur": scores[winner]["weighted_reward_eur"],
                "mean_after": mean.tolist(), "std_after": std.tolist(), "rng_after": copy.deepcopy(rng.bit_generator.state),
                "ledger": effort_ledger(all_rows), "elapsed_s": time.monotonic() - started}
            state["remaining_expected_s_at_observed_rate"] = (plan["expected_rollouts"]["total"] - len(all_rows)) * state["elapsed_s"] / len(all_rows)
            checkpoint = save_checkpoint(policy, out / "checkpoints" / f"generation-{generation:03d}.zip",
                plan=plan, parameters=incumbent, kind="generation-incumbent", ledger=effort_ledger(all_rows), generation=generation)
            state["checkpoint"] = checkpoint["checkpoint"]
            publish_json(out / f"generation-{generation:03d}-result.json", state)
            generation_results.append(state)
            print(json.dumps({"event": "generation_complete", "generation": generation, "winner": winner,
                "weighted_reward_eur": scores[winner]["weighted_reward_eur"], "incumbent_reevaluated_eur": scores[recipe["population"] - 1]["weighted_reward_eur"],
                "elapsed_s": state["elapsed_s"], "remaining_expected_s": state["remaining_expected_s_at_observed_rate"], "ledger": state["ledger"]}), flush=True)
        search_complete = len(generation_results) == recipe["generations"]
        validation = None
        if search_complete:
            cases = panel_cases(recipe["validation_seeds"])
            validation_tasks = [{"phase": "validation", "generation": recipe["generations"], "candidate": index,
                "parameters": parameters.tolist(), **case} for index, parameters in ((0, np.zeros(14)), (1, incumbent)) for case in cases]
            publish_json(out / "validation-plan.json", {"registered_before_execution": now(), "selection_allowed": False,
                "cases": cases, "parent_parameters": [0.] * 14, "endpoint_parameters": incumbent.tolist()})
            rows = execute_panel(executor, validation_tasks, root, recipe)
            all_rows.extend(rows)
            if all(row["status"] == "completed" for row in rows):
                validation = {"parent": weighted_objective([r for r in rows if r["candidate"] == 0], cases),
                    "endpoint": weighted_objective([r for r in rows if r["candidate"] == 1], cases),
                    "selection_allowed": False, "cases_per_policy": len(cases)}
                publish_json(out / "validation-result.json", validation)
        complete = search_complete and validation is not None
        require_plan(root)
        endpoint = save_checkpoint(policy, out / ("final-policy.zip" if complete else "incomplete-policy.zip"),
            plan=plan, parameters=incumbent, kind="final" if complete else "incomplete", ledger=effort_ledger(all_rows), generation=len(generation_results) - 1)
        result = {"schema": SCHEMA, "status": "completed" if complete else "incomplete",
            "completed_at": now(), "elapsed_s": time.monotonic() - started, "completed_generations": len(generation_results),
            "search_ledger": effort_ledger([r for r in all_rows if r["phase"] == "search"]),
            "validation_ledger": effort_ledger([r for r in all_rows if r["phase"] == "validation"]),
            "total_ledger": effort_ledger(all_rows), "expected_rollouts": plan["expected_rollouts"],
            "final_parameters": incumbent.tolist(), "validation": validation, "endpoint": endpoint,
            "optimizer_family": "CEM parameter-space policy search", "gradient_updates": 0, "ppo_updates": 0}
        if complete and result["total_ledger"]["completed_rollouts"] != plan["expected_rollouts"]["total"]:
            raise ValueError("Completed CEM effort ledger differs from prospectively registered work")
        publish_json(out / "result.json", result)
        return result
    except BaseException as exc:
        publish_json(out / "failure.json", {"at": now(), "type": type(exc).__name__, "error": str(exc),
            "elapsed_s": time.monotonic() - started, "ledger": effort_ledger(all_rows)})
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        policy.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "train", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_plan(args.root, json.loads(args.recipe.read_text()))
    elif args.command == "train":
        result = train(args.root)
    else:
        result = {"verified": True, "plan_sha256": sha(require_plan(args.root))}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Exploratory development-only initial-actor / midpoint-normalizer ablation.

This creates a combined inference fixture, never a canonical checkpoint. It
loads both original checkpoints through the frozen verifier, then discards the
midpoint model before any environment reset. No policy weights are saved.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCHEMA = "free-ppo-v1-exploratory-normalization-ablation"
OPTIMIZERS = (101, 211, 307)
SCENARIOS = tuple(range(10100000, 10100008))
MIDPOINT = 393216


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def normalizer_identity(normalizer):
    return {"mean": normalizer.obs_rms.mean.tolist(), "variance": normalizer.obs_rms.var.tolist(),
            "count": float(normalizer.obs_rms.count), "clip_obs": float(normalizer.clip_obs),
            "epsilon": float(normalizer.epsilon), "training": bool(normalizer.training),
            "norm_obs": bool(normalizer.norm_obs), "norm_reward": bool(normalizer.norm_reward)}


def panel(path, optimizer, steps, freeze_sha):
    value = json.loads(path.read_text())
    expected_label = f"free-ppo-v1-{optimizer}-step-{steps}"
    if (value["schema"] != "free-ppo-v1-evaluation-panel" or value["controller"] != expected_label or
        value["split"] != "development" or value["requested_seeds"] != list(SCENARIOS) or
        value["evaluation_freeze_sha256"] != freeze_sha or value["source_files_unchanged_during_panel"] is not True or
        sorted(r["seed"] for r in value["runs"]) != list(SCENARIOS)):
        raise ValueError("Canonical development comparison identity differs")
    for row in value["runs"]:
        if (row["checkpoint_timesteps"] != steps or row["optimizer_seed"] != optimizer or
            row["controller"] != expected_label or row["split"] != "development" or
            row["evaluation_freeze_sha256"] != freeze_sha or not row["execution_complete"]):
            raise ValueError("Canonical comparison contains a wrong or incomplete case")
    return {r["seed"]: r for r in value["runs"]}


def metric(row, name):
    if name in ("execution_complete", "every_delivery_passed", "delivery_and_physics_passed"):
        return float(row[name])
    for group in ("physical_metrics", "economic_metrics"):
        if name in row[group]:
            return row[group][name]
    raise KeyError(name)


def run_optimizer(campaign_string, output_string, optimizer, deadline):
    import numpy as np
    from docs.research.tools.free_ppo_v1 import environment
    from docs.research.tools.free_ppo_v1 import train
    from docs.research.tools.free_ppo_v1 import evaluate
    import torch

    train.configure_torch()
    campaign, output = Path(campaign_string), Path(output_string) / f"seed-{optimizer}"
    output.mkdir(exist_ok=False)
    freeze = evaluate.require_freeze(campaign / "evaluation-freeze.json")
    gate = campaign / "gate-development.json"
    source_before = train.source_records() + [pin(evaluate.__file__), pin(__file__), pin(gate)]
    if time.time() >= deadline:
        raise TimeoutError("Exploratory diagnostic deadline reached before loading")
    initial_path = campaign / "training" / f"seed-{optimizer}" / "initial-policy.zip"
    midpoint_path = campaign / "training" / f"seed-{optimizer}" / "checkpoints" / f"step-{MIDPOINT}.zip"
    initial_panel_path = campaign / "evaluations/learning-curves" / f"free-ppo-v1-{optimizer}-step-0" / "summary.json"
    midpoint_panel_path = campaign / "evaluations/learning-curves" / f"free-ppo-v1-{optimizer}-step-{MIDPOINT}" / "summary.json"
    initial_cases = panel(initial_panel_path, optimizer, 0, freeze["sha256"])
    midpoint_cases = panel(midpoint_panel_path, optimizer, MIDPOINT, freeze["sha256"])
    plan_path = campaign / "training/plan.json"
    initial = train.load_checkpoint(initial_path, plan_path)
    midpoint = None
    started = time.monotonic()
    try:
        midpoint = train.load_checkpoint(midpoint_path, plan_path)
        if initial.num_timesteps != 0 or midpoint.num_timesteps != MIDPOINT or initial.seed != optimizer or midpoint.seed != optimizer:
            raise ValueError("Unexpected loaded actor/normalizer checkpoint identity")
        initial_actor = train.actor_parameter_sha256(initial.model)
        initial_mean = train.actor_mean_parameter_sha256(initial.model)
        initial_policy = train.policy_parameter_sha256(initial.model)
        midpoint_actor = train.actor_parameter_sha256(midpoint.model)
        if initial_actor == midpoint_actor:
            raise ValueError("Midpoint actor is unexpectedly unchanged")
        midpoint_norm = normalizer_identity(midpoint.normalizer)
        if midpoint_norm["training"] or not midpoint_norm["norm_obs"] or midpoint_norm["norm_reward"]:
            raise ValueError("Midpoint normalization is not frozen observation-only inference")
        norm_sha = sha(midpoint_norm)
        initial_norm_sha = sha(normalizer_identity(initial.normalizer))
        # Retain only the verified midpoint statistics. No midpoint actor remains
        # in this process during any rollout; all inference below uses initial.model.
        del midpoint.model
        initial_meta = json.loads(initial_path.with_suffix(".json").read_text())
        midpoint_meta = json.loads(midpoint_path.with_suffix(".json").read_text())
        for seed in SCENARIOS:
            if (initial_cases[seed]["checkpoint"]["checkpoint"]["sha256"] != initial_meta["checkpoint"]["sha256"] or
                midpoint_cases[seed]["checkpoint"]["checkpoint"]["sha256"] != midpoint_meta["checkpoint"]["sha256"] or
                midpoint_cases[seed]["checkpoint"]["normalizer"]["sha256"] != midpoint_meta["normalizer"]["sha256"]):
                raise ValueError("Canonical panels do not pin the loaded checkpoints")
        identity = {"schema": SCHEMA, "exploratory": True, "split": "development", "optimizer_seed": optimizer,
                    "actor_checkpoint": pin(initial_path), "normalizer_checkpoint": pin(midpoint_path),
                    "normalizer": pin(midpoint_path.with_suffix(".vecnormalize.pkl")),
                    "initial_actor_tensor_sha256": initial_actor, "initial_actor_mean_tensor_sha256": initial_mean,
                    "initial_policy_tensor_sha256": initial_policy, "unused_midpoint_actor_tensor_sha256": midpoint_actor,
                    "midpoint_model_discarded_before_any_reset": True, "normalizer_statistics_sha256": norm_sha,
                    "original_initial_normalizer_statistics_sha256": initial_norm_sha,
                    "normalizer_observation_count": midpoint_norm["count"],
                    "canonical_panels": {"initial": pin(initial_panel_path), "midpoint": pin(midpoint_panel_path)},
                    "evaluation_freeze_sha256": freeze["sha256"], "source_files": source_before,
                    "interpretation": "Original step-0 actor combined with same-seed midpoint observation statistics. Not a canonical checkpoint, endpoint, or prespecified confirmatory comparison."}
        publish(output / "identity.json", identity)
        cases = []
        for seed in SCENARIOS:
            if time.time() >= deadline:
                raise TimeoutError("Exploratory diagnostic deadline reached")
            env = environment.DirectPlantEnv(split="development", gate_path=gate, record_minutes=True)
            case_started = time.monotonic()
            try:
                obs, _ = env.reset(seed=seed)
                opening = env.simulator.state.to_dict()
                audit = evaluate.MinuteAudit(env)
                audit.add(opening)
                digest = hashlib.sha256()
                reward_sum = 0.0
                first_witness = None
                changed_normalized_inputs = 0
                changed_initial_actions = 0
                terminated = truncated = False
                for decision in range(672):
                    if time.time() >= deadline:
                        raise TimeoutError("Exploratory diagnostic deadline reached during a case")
                    if not np.array_equal(obs, env.current_policy_observation):
                        raise ValueError("Inference did not receive the actual public observation")
                    normalized = midpoint.normalize_observation(obs.copy())
                    manual = np.clip((np.asarray(obs, dtype=np.float32) - midpoint.normalizer.obs_rms.mean)
                                     / np.sqrt(midpoint.normalizer.obs_rms.var + midpoint.normalizer.epsilon),
                                     -midpoint.normalizer.clip_obs, midpoint.normalizer.clip_obs).astype(np.float32)
                    if not np.array_equal(normalized, manual):
                        raise ValueError("Normalization differs from the verified midpoint statistics")
                    action, _ = initial.model.predict(normalized.copy(), deterministic=True)
                    tensor, _ = initial.model.policy.obs_to_tensor(normalized)
                    with torch.no_grad():
                        raw_mean = initial.model.policy.get_distribution(tensor).get_actions(deterministic=True).cpu().numpy()[0]
                    if not np.array_equal(action, np.clip(raw_mean, initial.model.action_space.low, initial.model.action_space.high)):
                        raise ValueError("Actor witness differs from executed action")
                    original_normalized = initial.normalize_observation(obs.copy())
                    original_action, _ = initial.model.predict(original_normalized, deterministic=True)
                    changed_normalized_inputs += int(not np.array_equal(normalized, original_normalized))
                    changed_initial_actions += int(not np.array_equal(action, original_action))
                    witness = {"decision": decision, "observation": obs.tolist(), "normalized_observation": normalized.tolist(),
                               "raw_initial_actor_mean": raw_mean.tolist(), "action": np.asarray(action).tolist()}
                    if first_witness is None:
                        first_witness = witness
                    digest.update(canonical(witness))
                    obs, reward, terminated, truncated, _ = env.step(action)
                    reward_sum += reward
                    if len(env.last_minute_truth_states) != 16:
                        raise ValueError("Missing minute audit states")
                    for state in env.last_minute_truth_states[1:]:
                        audit.add(state)
                    if terminated or truncated:
                        break
                physical, economic = env.metrics.to_dict(), env.economic_metrics.to_dict()
                closing = env.simulator.state.to_dict()
                manifest = env.audit_manifest()
                minute_audit = audit.result()
                econ_audit = evaluate.economic_audit(env, opening, closing, reward_sum)
                complete = bool(terminated and not truncated and env.decisions_taken == 672 and env.physics_steps_taken == 10080 and audit.count == 10081)
                settlements = env.simulator.delivery_audit()["settlements"]
                deliveries = bool(complete and len(settlements) == 28 and all(s["unmet_kg"] <= 1e-9 for s in settlements))
                for reference in (initial_cases[seed], midpoint_cases[seed]):
                    if (reference["exogenous_episode_sha256"] != manifest["beta_exogenous_episode_sha256"] or
                        reference["manifest_sha256"] != manifest["manifest_sha256"] or reference["initial_state"] != opening):
                        raise ValueError("Ablation and canonical cases do not share exogenous forcing/opening stocks")
                actor_unchanged = train.actor_parameter_sha256(initial.model) == initial_actor and train.actor_mean_parameter_sha256(initial.model) == initial_mean and train.policy_parameter_sha256(initial.model) == initial_policy
                norm_unchanged = sha(normalizer_identity(midpoint.normalizer)) == norm_sha and sha(normalizer_identity(initial.normalizer)) == initial_norm_sha
                if not (complete and minute_audit["all_physical_invariants_passed"] and econ_audit["passed"] and actor_unchanged and norm_unchanged):
                    raise ValueError("Complete execution, accounting or frozen-inference verification failed")
                row = {"schema": SCHEMA + "-case", "split": "development", "optimizer_seed": optimizer, "seed": seed,
                       "execution_complete": complete, "every_delivery_passed": deliveries,
                       "delivery_and_physics_passed": deliveries and minute_audit["all_physical_invariants_passed"],
                       "physical_metrics": physical, "economic_metrics": economic, "minute_audit": minute_audit,
                       "economic_audit": econ_audit, "initial_state": opening, "final_state": closing,
                       "exogenous_episode_sha256": manifest["beta_exogenous_episode_sha256"], "manifest_sha256": manifest["manifest_sha256"],
                       "inference": {"verified_actual_observation_decisions": env.decisions_taken,
                                     "midpoint_normalizer_manually_verified_decisions": env.decisions_taken,
                                     "changed_inputs_vs_original_normalizer": changed_normalized_inputs,
                                     "changed_actions_vs_original_normalizer_on_same_states": changed_initial_actions,
                                     "initial_actor_unchanged": actor_unchanged, "normalizers_unchanged": norm_unchanged,
                                     "midpoint_actor_used": False, "witness_sha256": digest.hexdigest(), "first_decision_witness": first_witness},
                       "wall_time_s": time.monotonic() - case_started}
                publish(output / f"case-{seed}.json", row)
                cases.append(row)
            finally:
                env.close()
        if train.source_records() + [pin(evaluate.__file__), pin(__file__), pin(gate)] != source_before:
            raise ValueError("Frozen runtime or ablation source changed during execution")
        measurements = ("every_delivery_passed", "delivery_and_physics_passed", "delivered_kg", "unmet_kg", "net_operating_value_eur", "electrolyser_forced_stops", "battery_equivalent_full_cycles")
        comparison = {}
        for name in measurements:
            a = [metric(initial_cases[s], name) for s in SCENARIOS]
            b = [metric(c, name) for c in cases]
            c = [metric(midpoint_cases[s], name) for s in SCENARIOS]
            comparison[name] = {"canonical_initial_mean": statistics.mean(a), "initial_actor_midpoint_normalizer_mean": statistics.mean(b),
                                "canonical_midpoint_mean": statistics.mean(c), "ablation_minus_initial_mean": statistics.mean(y-x for x,y in zip(a,b)),
                                "midpoint_minus_ablation_mean": statistics.mean(y-x for x,y in zip(b,c)),
                                "paired_scenario_values": [{"seed": s, "initial": x, "ablation": y, "midpoint": z} for s,x,y,z in zip(SCENARIOS,a,b,c)]}
        result = {"schema": SCHEMA + "-panel", "exploratory": True, "split": "development", "optimizer_seed": optimizer,
                  "scenarios": list(SCENARIOS), "identity": pin(output / "identity.json"),
                  "all_cases_completed_and_audited": True, "all_initial_actor_hashes_unchanged": True,
                  "all_normalizer_statistics_unchanged": True, "midpoint_actor_used": False,
                  "comparison": comparison, "cases": [pin(output / f"case-{s}.json") for s in SCENARIOS],
                  "source_files_unchanged": True, "wall_time_s": time.monotonic() - started}
        publish(output / "comparison.json", result)
        return result
    except BaseException as exc:
        publish(output / "failure.json", {"schema": SCHEMA + "-failure", "optimizer_seed": optimizer,
                                         "error": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        initial.close()
        if midpoint is not None:
            midpoint.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        raise ValueError("Deadline must include timezone")
    seconds = deadline.timestamp() - time.time()
    if not 0 < seconds <= 600:
        raise ValueError("Deadline must be within ten minutes")
    plan = {"schema": SCHEMA + "-attempt", "created_at": datetime.now(timezone.utc).isoformat(),
            "deadline_utc": deadline.isoformat(), "workers": 2, "optimizer_seeds": list(OPTIMIZERS),
            "scenarios": list(SCENARIOS), "actor_steps": 0, "normalizer_steps": MIDPOINT,
            "exploratory_development_only": True, "canonical_checkpoints_saved": False,
            "helper": pin(__file__), "campaign": str(args.campaign.resolve())}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    publish(args.output / "attempt.json", plan)
    results = []
    try:
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [executor.submit(run_optimizer, str(args.campaign.resolve()), str(args.output.resolve()), optimizer, deadline.timestamp()) for optimizer in OPTIMIZERS]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps({"optimizer_seed": result["optimizer_seed"], "complete": True,
                                  "delivery_success_rate": result["comparison"]["every_delivery_passed"]}), flush=True)
        results.sort(key=lambda r: r["optimizer_seed"])
        conclusion = {"schema": SCHEMA + "-comparison", "exploratory": True, "split": "development", "all24cases_completed": True,
                      "optimizer_seeds": list(OPTIMIZERS), "environmental_scenarios": 8, "panels": results,
                      "interpretation": "Post-hoc initial-actor/midpoint-normalizer ablation on eight already-observed development scenarios. It tests whether normalization alone reproduces midpoint behavior for these fixed random actors. It is not a canonical policy, a held-out result, an additive attribution of causal effects, or an algorithm-wide generalization claim."}
        publish(args.output / "comparison.json", conclusion)
    except BaseException as exc:
        publish(args.output / "failure.json", {"schema": SCHEMA + "-failure", "error": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()

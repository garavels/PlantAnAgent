"""Paired deterministic evaluation with immutable panels and full inference witnesses.

Development panels may be small. Final panels require all new final seeds and
conditions, and the environment's separately frozen selection gate. No outcome
is dropped on failure; an incomplete case invalidates the panel's claim.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np

from docs.research.tools.wholeplant_ppo_v1 import evaluate as old_audit, training as old_training
from docs.research.tools.wholeplant_ppo_v1.baselines import BaselineConfig, BaselinePolicy
from docs.research.tools.wholeplant_ppo_v1.portable import canonical, digest, file_pin
from docs.research.tools.tail_stress_v1.audit import MinuteAudit as TailMinuteAudit
from docs.research.tools.tail_stress_v1.environment import condition_spec

ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "learned-control-evaluation-v1"
CONDITIONS = ("ordinary", "control", "pv6", "pv18", "pv36", "meth24", "meth48")
DEVELOPMENT_SEEDS = tuple(range(51_000_000, 51_000_016))
FINAL_SEEDS = tuple(range(52_000_000, 52_000_032))
KINDS = ("baseline", "legacy-ppo", "learned")
CONTROLLER_IDENTITY_FIELDS = ("label", "kind", "baseline", "config", "checkpoint", "group", "recipe", "optimizer_seed")


def publish(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _controller_spec(controller):
    result = deepcopy(controller)
    if result.get("kind") not in KINDS:
        raise ValueError("Unknown controller kind")
    label = result.get("label", "")
    if not label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in label):
        raise ValueError("Controller label must be a safe unique directory component")
    if result["kind"] == "baseline":
        if result.get("baseline") not in ("daylight", "storage", "storage-persistence"):
            raise ValueError("Unknown baseline")
        # Config is an override of the actual unchanged plant constants, not an
        # alternative environment configuration.
        BaselineConfig(**result.get("config", {}))
    else:
        path = Path(result["checkpoint"]).resolve()
        result["checkpoint"] = str(path)
        result["checkpoint_files"] = {"checkpoint": file_pin(path),
            "metadata": file_pin(path.with_suffix(".json")),
            "normalizer": file_pin(path.with_suffix(".vecnormalize.pkl"))}
    return result


def _validate_grid(split, seeds, conditions):
    if split not in ("train", "development", "final"):
        raise ValueError("Unknown evaluation split")
    if not seeds or len(set(seeds)) != len(seeds) or not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Evaluation seeds and conditions must be nonempty and unique")
    if any(c not in CONDITIONS for c in conditions):
        raise ValueError("Unknown evaluation condition")
    allowed = range(50_000_000, 51_000_000) if split == "train" else (DEVELOPMENT_SEEDS if split == "development" else FINAL_SEEDS)
    if any(isinstance(s, bool) or not isinstance(s, int) or s not in allowed for s in seeds):
        raise ValueError("Evaluation seed outside declared split")
    if split == "final" and (tuple(seeds) != FINAL_SEEDS or tuple(conditions) != CONDITIONS):
        raise ValueError("Final evaluation requires the complete 32-seed seven-condition grid")


def _validate_selection(gate_path, controllers, seeds, conditions, primary):
    """Bind final evaluation policies and comparator to the pre-outcome choice."""
    gate_path = Path(gate_path).resolve()
    gate = json.loads(gate_path.read_text())
    if gate.get("schema") != "learned-control-v1-evaluation-gate" or gate.get("selected_recipe_frozen") is not True:
        raise ValueError("Final selection gate is not frozen")
    pin = gate["selected_recipe"]
    path = Path(pin["path"])
    if not path.is_absolute():
        path = gate_path.parent / path
    if file_pin(path)["sha256"] != pin["sha256"]:
        raise ValueError("Selected recipe bytes differ from gate")
    selected = json.loads(path.read_text())
    if selected.get("seeds") != list(seeds) or selected.get("conditions") != list(conditions):
        raise ValueError("Final grid differs from selected recipe")
    if selected.get("primary") != primary:
        raise ValueError("Final primary comparison differs from selected recipe")
    identity = lambda c: {k: c[k] for k in CONTROLLER_IDENTITY_FIELDS if k in c}
    actual = {c["label"]: c for c in controllers if c["kind"] == "learned"}
    chosen = {c["label"]: c for c in selected["controllers"]}
    if set(actual) != set(chosen) or any(identity(actual[k]) != identity(chosen[k]) for k in chosen):
        raise ValueError("Final learned policies differ from selected recipe")
    for label, chosen_policy in chosen.items():
        if chosen_policy.get("checkpoint_sha256") not in (None, actual[label]["checkpoint_files"]["checkpoint"]["sha256"]):
            raise ValueError("Selected checkpoint hash differs")
    champion = selected["baseline_champion"]
    matches = [c for c in controllers if c["label"] == champion["label"]]
    if len(matches) != 1 or identity(matches[0]) != identity(champion):
        raise ValueError("Final baseline champion differs from selected recipe")
    if (not primary or primary.get("recipe", primary.get("group")) != selected["recipe_id"]
            or primary.get("reference") != champion["label"]):
        raise ValueError("Primary recipe and champion are not the frozen selection")
    return file_pin(path)


def freeze_panel(output, controllers, seeds, conditions=CONDITIONS, *, split="development", gate_path=None,
                 source_paths=(), deadline_utc=None, primary=None, max_decisions=None):
    seeds, conditions = list(seeds), list(conditions)
    _validate_grid(split, seeds, conditions)
    if max_decisions is not None and (split != "train" or not 1 <= max_decisions < 672):
        raise ValueError("Truncated engineering evaluations require a training seed and 1..671 decisions")
    specs = [_controller_spec(c) for c in controllers]
    labels = [c["label"] for c in specs]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Controller labels must be nonempty and unique")
    gate_pin = selection_pin = None
    if split == "final":
        if gate_path is None:
            raise ValueError("Final selection gate is required")
        gate_pin = file_pin(gate_path)
        selection_pin = _validate_selection(gate_path, specs, seeds, conditions, primary)
    old_freeze = ROOT / "docs/research/results/wholeplant_ppo_20260917/held-out-freeze.json"
    original = old_audit.require_freeze(old_freeze)
    paths = {Path(p["path"]) for p in original["sources"]}
    paths.update((Path(__file__), Path(__file__).with_name("analyze.py"),
                  ROOT / "docs/research/tools/tail_stress_v1/environment.py",
                  ROOT / "docs/research/tools/tail_stress_v1/audit.py"))
    paths.update(p for p in Path(__file__).parent.glob("*.py"))
    paths.update(Path(p) for p in source_paths)
    spec = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(), "split": split,
        "seeds": seeds, "conditions": conditions, "controllers": {c["label"]: c for c in specs},
        "cases": [{"seed": s, "condition": c, "controller": p} for s in seeds for c in conditions for p in labels],
        "expected_cases": len(seeds)*len(conditions)*len(labels), "attempts_per_case": 1,
        "gate": gate_pin, "selected_recipe": selection_pin, "deadline_utc": deadline_utc, "max_decisions": max_decisions,
        "episode_hours": 168, "sources": [file_pin(p) for p in sorted(paths)],
        "original_evaluation": file_pin(old_freeze), "primary": primary,
        "trace_rule": "Every raw input and deterministic action retained; every physical minute audited; complete minute states for first seed of each controller-condition panel",
        "interpretation": "Prescribed synthetic condition distribution, not naturally occurring disruption frequencies"}
    spec["sha256"] = digest(spec)
    publish(output, spec)
    return spec


def require_freeze(path):
    spec = json.loads(Path(path).read_text())
    if spec.get("schema") != SCHEMA or spec.get("sha256") != digest({k: v for k, v in spec.items() if k != "sha256"}):
        raise ValueError("Evaluation freeze digest differs")
    _validate_grid(spec["split"], spec["seeds"], spec["conditions"])
    pins = [spec["original_evaluation"], *spec["sources"]]
    if spec.get("gate"):
        pins.append(spec["gate"])
        pins.append(spec["selected_recipe"])
    for controller in spec["controllers"].values():
        pins.extend(controller.get("checkpoint_files", {}).values())
    for record in pins:
        if file_pin(record["path"]) != record:
            raise ValueError("Frozen source, checkpoint, or gate changed: " + record["path"])
    if spec["split"] == "final":
        _validate_selection(spec["gate"]["path"], list(spec["controllers"].values()), spec["seeds"], spec["conditions"], spec["primary"])
    return spec


def load_controller(spec, env):
    if spec["kind"] == "baseline":
        config = BaselineConfig.from_plant_config(env.config, shared_bus=env.shared_bus,
                                                downstream_dynamics=env.downstream_dynamics)
        config = replace(config, **spec.get("config", {}))
        return BaselinePolicy(spec["baseline"], config)
    if spec["kind"] == "legacy-ppo":
        return old_training.load_checkpoint(Path(spec["checkpoint"]))
    from docs.research.tools.learned_control_v1 import training
    return training.load_checkpoint(Path(spec["checkpoint"]))


def _event(condition):
    if condition == "ordinary":
        return {"is_intervention": False, "onset_h": None, "end_h": None}
    timing = condition_spec(condition)
    return {"is_intervention": condition != "control",
            "onset_h": timing["onset_h"] if timing["onset_h"] is not None else 48.0,
            "end_h": timing["end_h"] if timing["end_h"] is not None else 48.0}


def run_case(freeze_path, seed, condition, controller, directory):
    from docs.research.tools.learned_control_v1.environment import FixedEnv
    spec = require_freeze(freeze_path)
    if {"seed": seed, "condition": condition, "controller": controller} not in spec["cases"]:
        raise ValueError("Case not prescribed in frozen evaluation grid")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = datetime.fromisoformat(spec["deadline_utc"]) if spec.get("deadline_utc") else None
    env = policy = auditor = operating = None
    initial = final = failure = None
    states, commands = [], []
    event = _event(condition)
    reward_sum = 0.0
    terminated = truncated = False
    witness = hashlib.sha256()
    retain = seed == spec["seeds"][0]
    forcing = base_source = manifest_hash = None
    cs = spec["controllers"][controller]
    try:
        if deadline and datetime.now(timezone.utc) >= deadline:
            raise TimeoutError("Evaluation deadline reached")
        env = FixedEnv(condition=condition, split=spec["split"],
                       gate_path=spec["gate"]["path"] if spec.get("gate") else None, record_minutes=True)
        obs, _ = env.reset(seed=seed)
        initial = env.simulator.state.to_dict()
        if retain:
            states.append(initial)
        forcing = digest([float(v) for v in env.frozen_source.expected_solar_kw])
        manifest_hash = digest(env.audit_manifest())
        if hasattr(env, "base_source_manifest"):
            base_source = env.base_source_manifest().get("base_paired_source_sha256")
        auditor = old_audit.MinuteAudit(env) if condition == "ordinary" else TailMinuteAudit(env)
        operating = old_audit.OperatingAudit(env)
        auditor.add(initial)
        operating.add(initial)
        policy = load_controller(cs, env)
        for index in range(spec.get("max_decisions") or 672):
            if deadline and datetime.now(timezone.utc) >= deadline:
                raise TimeoutError("Evaluation deadline reached")
            if obs.shape != (149,) or not np.array_equal(obs, env.current_policy_observation):
                raise ValueError("Actual policy input differs from the public observation")
            action, _ = policy.predict(obs.copy(), deterministic=True)
            action = np.asarray(action)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError("Controller did not return seven finite actions")
            evidence = {"observation": obs.tolist(), "observation_sha256": digest(obs.tolist()), "action": action.tolist()}
            if cs["kind"] == "baseline":
                checked, details = policy.inference(obs.copy())
                if not np.array_equal(action, checked):
                    raise ValueError("Baseline inference witness differs")
                evidence["baseline_inference"] = details
            else:
                details = policy.deterministic_details(obs.copy())
                if not np.array_equal(action, details["clipped_action"]):
                    raise ValueError("Learned policy deterministic witness differs")
                evidence.update(normalized_observation=np.asarray(details["normalized_input"]).tolist(),
                                raw_mean=np.asarray(details["unclipped_mean"]).tolist())
            witness.update(canonical(evidence))
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            trace = env.last_minute_truth_states
            if trace is None or len(trace) != 16:
                raise ValueError("Missing physical minute trace")
            old_audit.audit_intervention(env, action, info, trace)
            for state in trace[1:]:
                auditor.add(state)
                operating.add(state)
                if retain:
                    states.append(state.to_dict())
            commands.append({"decision_index": index, "evidence": evidence, "action": action.tolist(),
                             "reward_eur": float(reward)})
            for key, boundary in (("pre_event_state", event["onset_h"]), ("post_event_state", event["end_h"])):
                if boundary is not None and env.physics_steps_taken == round(boundary*60):
                    event[key] = deepcopy(env.simulator.state.to_dict())
            if terminated or truncated:
                break
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    try:
        final = env.simulator.state.to_dict() if initial is not None else None
        delivery = env.simulator.delivery_audit() if final else None
        if delivery and event["onset_h"] is not None:
            settlements = delivery["settlements"]
            onset, end = event["onset_h"]*60, event["end_h"]*60
            event.update(pre_event_unmet_kg=sum(d["unmet_kg"] for d in settlements if d["due_step"] <= onset),
                event_unmet_kg=sum(d["unmet_kg"] for d in settlements if onset < d["due_step"] <= end),
                post_event_unmet_kg=sum(d["unmet_kg"] for d in settlements if d["due_step"] > end))
        complete = bool(failure is None and terminated and not truncated and env.decisions_taken == 672 and auditor.count == 10081)
        try:
            require_freeze(freeze_path)
            unchanged = True
        except Exception as exc:
            unchanged = complete = False
            failure = failure or {"type": type(exc).__name__, "message": str(exc)}
        row = {"schema": SCHEMA + "-case", "seed": seed, "condition": condition, "controller": controller,
            "controller_kind": cs["kind"], "recipe": cs.get("recipe"), "optimizer_seed": cs.get("optimizer_seed"),
            "split": spec["split"], "freeze_sha256": spec["sha256"], "execution_complete": complete,
            "engineering_prefix": spec.get("max_decisions") is not None, "failure": failure, "sources_unchanged": unchanged,
            "decisions": len(commands), "physical_states": auditor.count if auditor else 0,
            "initial_state": initial, "final_state": final, "delivery": delivery,
            "economic_metrics": env.economic_metrics.to_dict() if final else None,
            "physical_metrics": env.metrics.to_dict() if final else None,
            "operating_metrics": operating.totals if operating else None,
            "minute_audit": auditor.result() if auditor else {"all_physical_invariants_passed": False},
            "economic_audit": old_audit.economic_audit(env, initial, final, reward_sum, operating) if final else {"passed": False},
            "every_delivery_passed": bool(complete and len(delivery["settlements"]) == 28 and all(d["unmet_kg"] <= 1e-9 for d in delivery["settlements"])),
            "event_metrics": event, "base_paired_source_sha256": base_source,
            "physical_solar_forcing_sha256": forcing, "environment_manifest_sha256": manifest_hash,
            "input_action_witness_sha256": witness.hexdigest(), "retained_minute_truth_sha256": digest(states) if states else None,
            "wall_time_s": time.monotonic()-started}
        recording = {"schema": SCHEMA + "-recording", "summary": deepcopy(row), "decision_trace": commands,
                     "minute_truth_states": states, "environment_manifest": env.audit_manifest() if retain and final else None}
        recording_path = directory / "recording.json.gz"
        with recording_path.open("xb") as stream:
            stream.write(gzip.compress(canonical(recording), compresslevel=1, mtime=0))
        row["recording"] = file_pin(recording_path)
        publish(directory / "result.json", row)
        return row
    finally:
        if policy is not None and hasattr(policy, "close"):
            policy.close()
        if env is not None:
            env.close()


def _worker(args):
    return run_case(**args)


def campaign(output, workers=3):
    output = Path(output).resolve()
    spec = require_freeze(output / "freeze.json")
    rows = []
    tasks = [{"freeze_path": output / "freeze.json", **c,
              "directory": output / "cases" / f"{c['condition']}--{c['controller']}--{c['seed']}"} for c in spec["cases"]]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=old_training.configure_torch) as pool:
        pending = {pool.submit(_worker, args): args for args in tasks}
        for future in as_completed(pending):
            row = future.result()
            rows.append(row)
            print(json.dumps({"completed": len(rows), "total": len(tasks), "seed": row["seed"], "condition": row["condition"],
                "controller": row["controller"], "complete": row["execution_complete"],
                "audit": row["minute_audit"]["all_physical_invariants_passed"],
                "unmet_kg": (row.get("economic_metrics") or {}).get("unmet_kg"), "failure": row["failure"]}), flush=True)
    rows.sort(key=lambda r: (r["seed"], r["condition"], r["controller"]))
    valid = all(r["execution_complete"] and r["sources_unchanged"] and r["minute_audit"]["all_physical_invariants_passed"]
                and r["economic_audit"]["passed"] for r in rows)
    pairing = all(len({r["physical_solar_forcing_sha256"] for r in rows if r["seed"] == s and r["condition"] == c}) == 1
                  and len({digest(r["initial_state"]) for r in rows if r["seed"] == s and r["condition"] == c}) == 1
                  for s in spec["seeds"] for c in spec["conditions"])
    summary = {"schema": SCHEMA + "-summary", "completed_at": datetime.now(timezone.utc).isoformat(),
               "freeze": file_pin(output / "freeze.json"), "case_count": len(rows),
               "all_executions_and_audits_passed": valid, "paired_forcing_and_openings_equal": pairing, "rows": rows}
    publish(output / "summary.json", summary)
    if not (valid and pairing) and spec.get("max_decisions") is None:
        raise RuntimeError("Incomplete or invalid evaluation evidence")
    return summary


def replay_recording(recording_path, freeze_path, directory):
    """Recompute a retained case and compare states, actual inference, and audits."""
    with gzip.open(recording_path, "rt") as stream:
        expected = json.load(stream)
    row = expected["summary"]
    if not expected.get("minute_truth_states"):
        raise ValueError("Replay requires retained minute states")
    actual = run_case(freeze_path, row["seed"], row["condition"], row["controller"], directory)
    for field in ("initial_state", "final_state", "economic_metrics", "physical_metrics", "operating_metrics",
                  "minute_audit", "economic_audit", "delivery", "event_metrics", "input_action_witness_sha256", "retained_minute_truth_sha256"):
        if canonical(actual[field]) != canonical(row[field]):
            raise ValueError("Replay differs: " + field)
    with gzip.open(actual["recording"]["path"], "rt") as stream:
        replay = json.load(stream)
    for field in ("decision_trace", "minute_truth_states"):
        if canonical(expected[field]) != canonical(replay[field]):
            raise ValueError("Replay differs: " + field)
    return {"passed": True, "seed": row["seed"], "condition": row["condition"], "controller": row["controller"],
            "decisions": row["decisions"], "physical_states": row["physical_states"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controllers", type=Path)
    parser.add_argument("--split", choices=("train", "development", "final"), default="development")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--deadline-utc")
    parser.add_argument("--max-decisions", type=int)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.command == "freeze":
        if args.controllers is None:
            parser.error("freeze requires --controllers")
        seeds = args.seeds or (FINAL_SEEDS if args.split == "final" else DEVELOPMENT_SEEDS)
        spec = freeze_panel(args.output / "freeze.json", json.loads(args.controllers.read_text()), seeds, args.conditions,
            split=args.split, gate_path=args.gate, deadline_utc=args.deadline_utc, max_decisions=args.max_decisions,
            primary=json.loads(args.primary.read_text()) if args.primary else None)
        print(json.dumps({"freeze_sha256": spec["sha256"], "expected_cases": spec["expected_cases"]}))
    else:
        campaign(args.output, args.workers)


if __name__ == "__main__":
    main()

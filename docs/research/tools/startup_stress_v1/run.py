"""One prospectively specified startup diagnostic; no training or case selection."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from docs.research.tools.wholeplant_ppo_v1 import evaluate as audit_tools
from docs.research.tools.wholeplant_ppo_v1 import training
from docs.research.tools.wholeplant_ppo_v1.baselines import baseline
from docs.research.tools.wholeplant_ppo_v1.physics import WholePlantAction, action_to_vector
from docs.research.tools.wholeplant_ppo_v1.portable import file_pin, digest

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT / "docs/research/results/wholeplant_ppo_20260917"
CONTROLLERS = ("feasibility-reference", "daylight", "storage", "ppo-101", "ppo-211", "ppo-307")
CONDITIONS = ("solar-loss", "clear-sky")
SEED = 42_000_000
SCHEMA = "startup-solar-loss-diagnostic-v1"


def publish(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def freeze(output):
    original_path = OLD / "held-out-freeze.json"
    original = audit_tools.require_freeze(original_path)
    paths = {Path(p["path"]) for p in original["sources"]}
    paths.update(Path(__file__).parent / name for name in ("__init__.py", "environment.py", "run.py"))
    paths.add(ROOT / "docs/research/STARTUP_STRESS_V1_PROTOCOL.md")
    paths.add(ROOT / "tests/test_startup_stress_environment.py")
    now = datetime.now(timezone.utc)
    controllers = {name: original["controllers"][name] for name in CONTROLLERS if name != "feasibility-reference"}
    controllers["feasibility-reference"] = {
        "kind": "engineered-reference", "not_an_autonomy_comparator": True,
        "first_24_decisions": "four90kW meth-only, two150kW meth-only, eighteen all-off",
        "remaining_decisions": "unchanged storage baseline",
    }
    spec = {"schema": SCHEMA, "created_at": now.isoformat(),
        "deadline_utc": (now + timedelta(minutes=30)).isoformat(),
        "seed": SEED, "independent_scenarios": 1, "conditions": list(CONDITIONS),
        "controller_order": list(CONTROLLERS), "controllers": controllers,
        "cases": [{"condition": c, "controller": p} for c in CONDITIONS for p in CONTROLLERS],
        "expected_cases": 12, "attempts_per_case": 1, "episode_hours": 168,
        "primary_delivery_step": 360, "original_evaluation": file_pin(original_path),
        "sources": [file_pin(p) for p in sorted(paths)],
        "witness_expected": {"produced_ch4_kg": 61.11111111111111,
            "actual_battery_discharge_kwh": 165, "battery_soc_at_6h": 0.228515625,
            "initial_ch4_kg": 120, "requested_first_delivery_kg": 180},
        "interpretation": "Constructed capability diagnostic; not a population tail-reliability estimate"}
    spec["sha256"] = digest(spec)
    publish(output, spec)
    return spec


def require_freeze(path):
    spec = json.loads(Path(path).read_text())
    if spec["schema"] != SCHEMA or spec["sha256"] != digest({k: v for k, v in spec.items() if k != "sha256"}):
        raise ValueError("Diagnostic freeze digest differs")
    for record in [spec["original_evaluation"], *spec["sources"]]:
        if file_pin(record["path"]) != record:
            raise ValueError(f"Frozen source changed: {record['path']}")
    for controller in spec["controllers"].values():
        for record in controller.get("checkpoint", {}).values():
            if file_pin(record["path"]) != record:
                raise ValueError("Frozen checkpoint, normalizer or sidecar changed")
    return spec


def witness_action(index):
    support = 90.0 if index < 4 else 150.0 if index < 6 else 0.0
    return action_to_vector(WholePlantAction(0, 0, 0, 0, 0, 0, support))


def run_case(freeze_path, condition, controller, directory):
    from docs.research.tools.startup_stress_v1.environment import StartupStressEnv

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    spec = require_freeze(freeze_path)
    if {"condition": condition, "controller": controller} not in spec["cases"]:
        raise ValueError("Case not prescribed")
    deadline = datetime.fromisoformat(spec["deadline_utc"])
    started = time.monotonic()
    env = policy = fallback = auditor = operating = None
    initial = first = failure = forcing_sha256 = None
    commands, states = [], []
    reward_sum, terminated, truncated = 0.0, False, False
    evidence_hash = hashlib.sha256()
    try:
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError("Diagnostic deadline reached before setup")
        env = StartupStressEnv(condition=condition, record_minutes=True)
        obs, _ = env.reset(seed=SEED)
        initial = env.simulator.state.to_dict()
        forcing_sha256 = digest([float(value) for value in env.frozen_source.expected_solar_kw])
        states.append(initial)
        auditor, operating = audit_tools.MinuteAudit(env), audit_tools.OperatingAudit(env)
        auditor.add(initial)
        operating.add(initial)
        controller_spec = spec["controllers"][controller]
        if controller_spec["kind"] == "ppo":
            policy = training.load_checkpoint(Path(controller_spec["checkpoint"]["checkpoint"]["path"]))
            if policy.num_timesteps != 786432 or policy.research_identity != controller_spec["checkpoint_environment_identity"]:
                raise ValueError("Wrong frozen trained policy")
        else:
            policy = baseline("storage" if controller == "feasibility-reference" else controller, env)
        for index in range(672):
            if datetime.now(timezone.utc) >= deadline:
                raise TimeoutError("Diagnostic local deadline reached")
            if obs.shape != (149,) or not np.array_equal(obs, env.current_policy_observation):
                raise ValueError("Input differs from actual emitted observation")
            evidence = {"observation": obs.tolist(), "observation_sha256": digest(obs.tolist())}
            if controller == "feasibility-reference" and index < 24:
                action = witness_action(index)
                evidence["rule"] = "prospectively specified physical feasibility schedule"
            else:
                action, _ = policy.predict(obs.copy(), deterministic=True)
                if controller_spec["kind"] == "ppo":
                    details = policy.deterministic_details(obs.copy())
                    if not np.array_equal(action, details["clipped_action"]):
                        raise ValueError("PPO mean/inference witness differs")
                    evidence.update(normalized_observation=details["normalized_input"].tolist(),
                                    raw_mean=details["unclipped_mean"].tolist())
                else:
                    checked, details = policy.inference(obs.copy())
                    if not np.array_equal(action, checked):
                        raise ValueError("Baseline inference witness differs")
                    evidence["baseline_inference"] = details
            action = np.asarray(action)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError("Invalid seven-action command")
            evidence["action"] = action.tolist()
            evidence_hash.update(json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            trace = env.last_minute_truth_states
            if trace is None or len(trace) != 16:
                raise ValueError("Missing physical minute trace")
            audit_tools.audit_intervention(env, action, info, trace)
            for state in trace[1:]:
                auditor.add(state)
                operating.add(state)
                states.append(state.to_dict())
            commands.append({"decision_index": index, "evidence": evidence,
                "action": action.tolist(), "reward_eur": float(reward),
                "direct_control": info["direct_control"], "battery_intervention": info["battery_intervention"]})
            if env.physics_steps_taken == 360:
                settlements = env.simulator.delivery_audit()["settlements"]
                if len(settlements) != 1 or settlements[0]["due_step"] != 360:
                    raise ValueError("First delivery boundary differs")
                at_six = env.simulator.state.to_dict()
                first = deepcopy({"delivery": settlements[0], "state": at_six,
                    "economic_metrics": env.economic_metrics.to_dict(),
                    "operating_metrics": dict(operating.totals),
                    "minute_audit": auditor.result(),
                    "economic_audit": audit_tools.economic_audit(env, initial, at_six, reward_sum, operating),
                    "first_delivery_passed": settlements[0]["unmet_kg"] <= 1e-9})
            if terminated or truncated:
                break
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    try:
        complete = bool(failure is None and env is not None and terminated and not truncated
            and env.decisions_taken == 672 and env.physics_steps_taken == 10080 and len(states) == 10081)
        final = env.simulator.state.to_dict() if env is not None and initial is not None else None
        delivery = env.simulator.delivery_audit() if final is not None else None
        economic = env.economic_metrics.to_dict() if final is not None else None
        eaudit = audit_tools.economic_audit(env, initial, final, reward_sum, operating) if final is not None else {"passed": False}
        paudit = auditor.result() if auditor else {"all_physical_invariants_passed": False}
        try:
            require_freeze(freeze_path)
            sources_unchanged = True
        except Exception as exc:
            sources_unchanged = False
            failure = failure or {"type": type(exc).__name__, "message": str(exc)}
            complete = False
        row = {"schema": SCHEMA + "-case", "controller": controller, "condition": condition,
            "seed": SEED, "freeze_sha256": spec["sha256"], "failure": failure,
            "execution_complete": complete, "sources_unchanged": sources_unchanged,
            "decisions": len(commands), "physical_states": len(states),
            "first_six_hours": first, "initial_state": initial, "final_state": final,
            "delivery": delivery, "economic_metrics": economic,
            "physical_metrics": env.metrics.to_dict() if final is not None else None,
            "operating_metrics": operating.totals if operating else None,
            "minute_audit": paudit, "economic_audit": eaudit,
            "every_delivery_passed": bool(complete and len(delivery["settlements"]) == 28 and all(d["unmet_kg"] <= 1e-9 for d in delivery["settlements"])),
            "first_delivery_passed": bool(first and first["first_delivery_passed"]),
            "input_action_witness_sha256": evidence_hash.hexdigest(),
            "physical_solar_forcing_sha256": forcing_sha256,
            "minute_states_sha256": digest(states),
            "environment_manifest": env.audit_manifest() if final is not None else None,
            "wall_time_s": time.monotonic()-started}
        record = {"schema": SCHEMA + "-recording", "summary": row, "decision_trace": commands, "minute_truth_states": states}
        recording_path = directory / "recording.json.gz"
        with recording_path.open("xb") as stream:
            stream.write(gzip.compress(json.dumps(record, separators=(",", ":"), allow_nan=False).encode(), mtime=0))
        row["recording"] = file_pin(recording_path)
        publish(directory / "result.json", row)
        return row
    finally:
        if policy is not None and hasattr(policy, "close"):
            policy.close()
        if env is not None:
            env.close()


def campaign(output):
    output = Path(output).resolve()
    spec = require_freeze(output / "freeze.json")
    training.configure_torch()
    rows = []
    for case in spec["cases"]:
        print(json.dumps({"event": "case_started", **case}), flush=True)
        row = run_case(output / "freeze.json", **case,
                       directory=output / "cases" / (case["condition"] + "--" + case["controller"]))
        rows.append(row)
        print(json.dumps({"event": "case_completed", **case, "complete": row["execution_complete"],
            "first_delivered_kg": row["first_six_hours"]["delivery"]["delivered_kg"] if row["first_six_hours"] else None,
            "week_unmet_kg": row["economic_metrics"]["unmet_kg"] if row["economic_metrics"] else None,
            "failure": row["failure"]}), flush=True)
    checks = all(r["execution_complete"] and r["sources_unchanged"] and r["minute_audit"]["all_physical_invariants_passed"]
                 and r["economic_audit"]["passed"] and r["first_six_hours"]["economic_audit"]["passed"] for r in rows)
    starts_equal = all(len({digest(r["initial_state"]) for r in rows if r["condition"] == condition}) == 1 for condition in CONDITIONS)
    forcing_equal = all(len({r["physical_solar_forcing_sha256"] for r in rows if r["condition"] == condition}) == 1 for condition in CONDITIONS)
    # Physical initial PV differs across conditions; all stock/equipment starts must match.
    opening_fields = ("battery_energy_kwh", "h2_level_kg", "co2_level_kg", "ch4_level_kg", "dac_readiness_kwh", "methanator_readiness_kwh")
    openings_equal = all(r["initial_state"] is not None and all(r["initial_state"][k] == rows[0]["initial_state"][k] for k in opening_fields) for r in rows)
    sunlight_zero = all(r["first_six_hours"] and r["first_six_hours"]["operating_metrics"]["solar_available_kwh"] == 0 for r in rows if r["condition"] == "solar-loss")
    day = next(r for r in rows if (r["condition"], r["controller"]) == ("solar-loss", "daylight"))
    witness = next(r for r in rows if (r["condition"], r["controller"]) == ("solar-loss", "feasibility-reference"))
    mechanism_valid = bool(day["first_six_hours"] and abs(day["first_six_hours"]["delivery"]["delivered_kg"] - 120) < 1e-7
        and witness["first_delivery_passed"] and abs(witness["first_six_hours"]["operating_metrics"]["battery_discharge_kwh"] - 165) < 1e-5)
    summary = {"schema": SCHEMA + "-summary", "completed_at": datetime.now(timezone.utc).isoformat(),
        "freeze": file_pin(output / "freeze.json"), "case_count": len(rows),
        "all_executions_and_audits_passed": checks, "identical_initial_state_within_condition": starts_equal,
        "identical_solar_forcing_within_condition": forcing_equal,
        "paired_opening_stocks_equal": openings_equal, "zero_solar_first_six_hours_verified": sunlight_zero,
        "daylight_bound_and_feasible_witness_verified": mechanism_valid, "rows": rows}
    publish(output / "summary.json", summary)
    if not (checks and starts_equal and openings_equal and forcing_equal and sunlight_zero and mechanism_valid):
        raise RuntimeError("Diagnostic retained incomplete or audit-failed cases; inspect before interpreting")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("freeze", "run"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "freeze":
        print(json.dumps(freeze(args.output / "freeze.json")), flush=True)
    else:
        campaign(args.output)


if __name__ == "__main__":
    main()

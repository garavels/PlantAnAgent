"""Independent supplementary evaluator; unchanged main-study physical/economic audits.

The case loop is derived from frozen v2 but has its own explicit persistence
factory, transformation witnesses, freeze schema and late-bound main closure.
No imports are monkeypatched and no baseline or physical model is changed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import statistics
import sys
import time
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from docs.research.tools.free_ppo_v1.analyze import canonical_sha, file_pin, publish, RESOURCES
from docs.research.tools.free_ppo_v1.evaluate import MinuteAudit, economic_audit
from docs.research.tools.comparison_ppo_v2 import evaluate as main_e
from docs.research.tools.comparison_ppo_v2.evaluate import OperatingAudit, audit_intervention, write_record, environment
from docs.research.tools.comparison_ppo_v2.baselines import BaselineConfig
from docs.research.tools.forecast_ablation_v1.policy import PersistencePolicy, verify_witness
from docs.research.tools.forecast_ablation_v1.freeze import SCHEMA, sources, require_freeze, admit, main_admission


def empty_case(**kwargs):
    row = main_e.empty_case(**kwargs)
    row["schema"] = SCHEMA + "-case"
    return row


def evaluate_case(*, freeze_path, gate_path=None, controller, condition, split, seed, output, retain_minutes=False, max_decisions=None, deadline_utc=None, main_completion_path=None):
    import numpy as np
    freeze = require_freeze(freeze_path)
    panel = admit(freeze, controller=controller, condition=condition, split=split, seed=seed, gate_path=gate_path, max_decisions=max_decisions, main_completion_path=main_completion_path)
    if retain_minutes and seed != panel["seeds"][0]:
        raise ValueError("Full trace selection was fixed to the first scenario")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    deadline = datetime.fromisoformat((deadline_utc or freeze["hard_stop_utc"]).replace("Z", "+00:00"))
    if deadline and deadline.tzinfo is None:
        raise ValueError("Deadline must include timezone")
    deadline = min(deadline, datetime.fromisoformat(freeze["hard_stop_utc"]))
    spec = freeze["controllers"][controller]
    env = model = audit = operating = None
    failure = None
    commands, states, latencies = [], [], []
    reward_sum = 0.0
    terminated = truncated = False
    clipped_means = 0
    input_digest = hashlib.sha256()
    try:
        env = environment().ComparisonEnv(**freeze["conditions"][condition], split=split,
            gate_path=gate_path or (freeze["gate"]["path"] if freeze["gate"] else None), record_minutes=True)
        obs, _ = env.reset(seed=seed)
        initial = env.simulator.state.to_dict()
        audit, operating = MinuteAudit(env), OperatingAudit(env)
        audit.add(initial); operating.add(initial)
        if retain_minutes:
            states.append(initial)
        model = PersistencePolicy(BaselineConfig.from_plant_config(env.config))
        for index in range(max_decisions or 672):
            if deadline and datetime.now(timezone.utc) >= deadline:
                raise TimeoutError("Campaign UTC deadline reached; partial attempt retained")
            if not np.array_equal(obs, env.current_policy_observation):
                raise ValueError("Policy input differs from actual current observation")
            t = time.perf_counter()
            action, evidence = model.inference(obs.copy())
            verify_witness(evidence, model)
            latencies.append(time.perf_counter() - t)
            input_digest.update(json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            trace = env.last_minute_truth_states
            if trace is None or len(trace) != 16:
                raise ValueError("Missing exact 15-minute physical trace")
            audit_intervention(env, action, info, trace)
            for state in trace[1:]:
                audit.add(state); operating.add(state)
                if retain_minutes:
                    states.append(state.to_dict())
            commands.append({"decision_index": index, "action": np.asarray(action).tolist(), "evidence": evidence,
                             "reward_eur": reward, "direct_control": info["direct_control"], "battery_intervention": info.get("battery_intervention"),
                             "state": {k: getattr(env.simulator.state, k) for k in ("battery_energy_kwh", "battery_soc", "h2_level_kg", "co2_level_kg", "ch4_level_kg", "solar_available_kw", "electrolyser_power_kw", "dac_power_kw", "methanator_power_kw")}})
            if terminated or truncated:
                break
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    try:
        if env is None or audit is None:
            row = empty_case(seed=seed, controller=controller, condition=condition, split=split, freeze=freeze,
                             failure=failure or {"type": "SetupFailure"}, elapsed=time.monotonic() - started)
            row.pop("recording")
            return {**row, "recording": write_record(output, {"schema": SCHEMA + "-recording", "summary": row, "decision_trace": commands, "minute_truth_states": states or None})}
        final = env.simulator.state.to_dict()
        physical, economic = env.metrics.to_dict(), env.economic_metrics.to_dict()
        delivery = env.simulator.delivery_audit()
        cycles = [s.to_dict() for s in env.cycle_settlements]
        complete = failure is None and terminated and not truncated and env.decisions_taken == 672 and env.physics_steps_taken == 10080 and audit.count == 10081
        audit_result = audit.result()
        delivery_pass = complete and len(delivery["settlements"]) == 28 and all(r["unmet_kg"] <= 1e-9 for r in delivery["settlements"])
        physical_pass = complete and audit_result["all_physical_invariants_passed"]
        manifest = env.audit_manifest()
        paired_source = env.paired_source_manifest()
        row = {"schema": SCHEMA + "-case", "seed": seed, "controller": controller, "condition": condition, "split": split,
            "variant": freeze["conditions"][condition], "controller_kind": spec["kind"], "training_family": spec.get("training_family"),
            "optimizer_seed": spec.get("optimizer_seed"), "checkpoint_timesteps": spec.get("checkpoint_timesteps"), "checkpoint": spec.get("checkpoint"),
            "evaluation_freeze_sha256": freeze["sha256"], "execution_complete": bool(complete), "every_delivery_passed": bool(delivery_pass),
            "physical_invariants_passed": bool(physical_pass), "delivery_and_physics_passed": bool(delivery_pass and physical_pass),
            "failure": failure, "decisions": env.decisions_taken, "physical_steps": env.physics_steps_taken, "wall_time_s": time.monotonic() - started,
            "delivery_settlements": len(delivery["settlements"]), "maximum_delivery_unmet_kg": max((r["unmet_kg"] for r in delivery["settlements"]), default=0),
            "daily_returns_met": sum(c["feasible"] for c in cycles),
            "stock_deficits": {r: {"observed_daily_deficits": [c["deficits"][r] for c in cycles], "sum": sum(c["deficits"][r] for c in cycles), "max": max((c["deficits"][r] for c in cycles), default=0)} for r in RESOURCES},
            "physical_metrics": physical, "economic_metrics": economic, "operating_metrics": operating.totals, "minute_audit": audit_result,
            "economic_audit": economic_audit(env, initial, final, reward_sum), "initial_state": initial, "final_state": final,
            "paired_source_sha256": paired_source["paired_source_sha256"], "paired_source_manifest": paired_source,
            "exogenous_episode_sha256": manifest["beta_exogenous_episode_sha256"], "manifest_sha256": manifest["manifest_sha256"],
            "controller_latency_s": {"mean": statistics.mean(latencies) if latencies else None, "max": max(latencies, default=0), "calls": len(latencies)},
            "inference": {"verified_decisions": len(commands), "actual_observation_inputs": True, "transformation_verified_decisions": len(commands), "raw_mean_clipped_decisions": clipped_means, "witness_sha256": input_digest.hexdigest()},
            "baseline_metadata": model.metadata() if model is not None else None,
            "main_campaign_admission": main_admission(freeze, gate_path, main_completion_path) if split == "held_out" else None,
            "retained_minute_truth_sha256": canonical_sha(states) if states else None}
        record = {"schema": SCHEMA + "-recording", "summary": row, "decision_trace": commands, "cycle_settlements": cycles, "delivery": delivery,
                  "manifest": manifest, "minute_truth_states": states or None,
                  "trace_selection": "First prescribed scenario" if retain_minutes else "All minute states audited and streaming-hashed; full raw minute trace omitted"}
        return {**row, "recording": write_record(output, record)}
    finally:
        if env is not None:
            env.close()
        if model is not None and hasattr(model, "close"):
            model.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--gate", type=Path)
    p.add_argument("--main-completion", type=Path)
    p.add_argument("--controller", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--split", choices=("train", "held_out"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--retain-minutes", action="store_true")
    p.add_argument("--max-decisions", type=int)
    p.add_argument("--deadline-utc")
    args = p.parse_args()
    def interrupted(signum, frame):
        raise TimeoutError(f"Parent watchdog signal {signum}; partial attempt retained")
    signal.signal(signal.SIGTERM, interrupted)
    result = evaluate_case(freeze_path=args.freeze, gate_path=args.gate, controller=args.controller, condition=args.condition,
        split=args.split, seed=args.seed, output=args.output, retain_minutes=args.retain_minutes, max_decisions=args.max_decisions, deadline_utc=args.deadline_utc, main_completion_path=args.main_completion)
    publish(args.result, result)
    print(json.dumps({"controller": args.controller, "condition": args.condition, "seed": args.seed, "complete": result["execution_complete"], "failure": result["failure"]}), flush=True)


if __name__ == "__main__":
    main()

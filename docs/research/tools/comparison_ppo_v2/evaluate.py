"""Prospectively frozen, one-attempt comparisons in the same seven-day plant.

Imported minute/economic auditors are unchanged from the completed v1 study.
Old policies are transfer evaluations; no observation normalization is refitted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import signal
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.research.tools.free_ppo_v1.analyze import canonical_sha, file_pin, publish, RESOURCES
from docs.research.tools.free_ppo_v1.evaluate import MinuteAudit, economic_audit, checkpoint_record

SCHEMA = "comparison-ppo-v2-evaluation"


def environment():
    from docs.research.tools.comparison_ppo_v2 import environment as module
    return module


def sources():
    paths = {Path(row["path"]) for row in environment().source_records()}
    paths.update(Path(__file__).parent / name for name in ("evaluate.py", "analyze.py", "run_panels.py", "baselines.py", "training.py"))
    paths.update(Path(__file__).parent.parent / "free_ppo_v1" / name for name in ("evaluate.py", "analyze.py", "train.py"))
    protocol = ROOT / "docs/research/COMPARISON_PPO_V2_PROTOCOL.md"
    if protocol.exists():
        paths.add(protocol)
    return [file_pin(path) for path in sorted(paths)]


def freeze_evaluation(output, *, controllers, conditions, panels, gate_path=None, description=None):
    """Pin all evaluated identities and panel membership before running any case."""
    controller_map = {}
    for original in controllers:
        item = dict(original)
        label = item["label"]
        if label in controller_map or item["kind"] not in ("ppo", "baseline"):
            raise ValueError("Duplicate or invalid controller")
        if item["kind"] == "ppo":
            item["checkpoint"] = checkpoint_record(item["checkpoint"])
            metadata = json.loads(Path(item["checkpoint"]["metadata"]["path"]).read_text())
            item["checkpoint_timesteps"] = metadata["num_timesteps"]
            item["checkpoint_environment_identity"] = metadata["environment_identity"]
        elif item.get("baseline") not in ("daylight", "storage"):
            raise ValueError("Unknown baseline")
        controller_map[label] = item
    condition_map = {}
    for original in conditions:
        item = dict(original)
        label = item.pop("label")
        if label in condition_map or item.get("delivery_rate_kg_h") not in (25, 30) or item.get("opening_battery") not in ("sampled", "floor") or item.get("battery_mode") not in ("enabled", "disabled"):
            raise ValueError("Duplicate or invalid condition")
        condition_map[label] = item
    selections = []
    seen = set()
    for original in panels:
        item = dict(original)
        key = item["split"], item["controller"], item["condition"]
        if key in seen or key[1] not in controller_map or key[2] not in condition_map or key[0] not in ("train", "development", "held_out"):
            raise ValueError("Duplicate or invalid panel")
        seen.add(key)
        expected = list(environment().DEVELOPMENT_SEEDS if key[0] == "development" else environment().HELD_OUT_SEEDS) if key[0] != "train" else item.get("seeds")
        if item.get("seeds", expected) != expected or not expected or len(set(expected)) != len(expected):
            raise ValueError("Evaluation panels must contain the complete prescribed split")
        item["seeds"] = expected
        selections.append(item)
    if not selections:
        raise ValueError("No panels selected")
    payload = {"schema": SCHEMA + "-freeze", "created_at": datetime.now(timezone.utc).isoformat(),
               "description": description, "controllers": controller_map, "conditions": condition_map, "panels": selections,
               "environment_identity_sha256": environment().environment_identity_sha256(),
               "source_identity_sha256": environment().source_identity_sha256(), "sources": sources(),
               "gate": file_pin(gate_path) if gate_path else None, "attempts_per_case": 1,
               "full_trace_rule": "First prescribed scenario per controller-condition panel; every minute audited in every case",
               "bootstrap": {"seed": 20260917, "replicates": 10000, "unit": "paired environmental scenario, conditional on fixed policies"}}
    payload["sha256"] = canonical_sha(payload)
    publish(output, payload)
    return payload


def require_freeze(path):
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != SCHEMA + "-freeze" or payload.get("sha256") != canonical_sha({k: v for k, v in payload.items() if k != "sha256"}):
        raise ValueError("Comparison freeze identity differs")
    if payload["sources"] != sources() or payload["environment_identity_sha256"] != environment().environment_identity_sha256() or payload["source_identity_sha256"] != environment().source_identity_sha256():
        raise ValueError("Frozen comparison source changed")
    for controller in payload["controllers"].values():
        if controller["kind"] == "ppo":
            for pin in controller["checkpoint"].values():
                if file_pin(pin["path"]) != pin:
                    raise ValueError("Frozen policy, metadata, or normalizer changed")
    if payload["gate"] and file_pin(payload["gate"]["path"]) != payload["gate"]:
        raise ValueError("Frozen evaluation gate changed")
    return payload


def admit(freeze, *, controller, condition, split, seed, gate_path=None, max_decisions=None):
    matching = [p for p in freeze["panels"] if (p["controller"], p["condition"], p["split"]) == (controller, condition, split)]
    if len(matching) != 1 or seed not in matching[0]["seeds"]:
        raise ValueError("Case is outside the frozen panel")
    if max_decisions is not None and (split != "train" or not 1 <= max_decisions <= 672):
        raise ValueError("Short engineering fixtures require training seeds")
    if freeze["gate"] and file_pin(gate_path or freeze["gate"]["path"]) != freeze["gate"]:
        raise ValueError("Case gate differs from frozen gate")
    if split != "train":
        environment()._admit(split, Path(gate_path or freeze["gate"]["path"]))
    return matching[0]


class OperatingAudit:
    """Descriptive integrated flows: actual delivered powers, never requests."""
    def __init__(self, env):
        self.dt_h = env.config.dt_h
        self.previous = None
        self.totals = {name: 0.0 for name in ("solar_available_kwh", "battery_charge_kwh", "battery_discharge_kwh", "electrolyser_kwh", "dac_kwh", "methanator_kwh", "zero_solar_electrolyser_kwh", "zero_solar_ch4_produced_kg", "solar_present_ch4_produced_kg", "zero_solar_battery_discharge_kwh", "electrolyser_on_minutes", "dac_on_minutes", "methanator_on_minutes")}

    def add(self, state):
        s = state.to_dict() if hasattr(state, "to_dict") else state
        if self.previous is not None:
            for label, field in (("solar_available_kwh", "solar_available_kw"), ("battery_charge_kwh", "solar_to_battery_kw"), ("battery_discharge_kwh", "battery_to_electrolyser_kw"), ("electrolyser_kwh", "electrolyser_power_kw"), ("dac_kwh", "dac_power_kw"), ("methanator_kwh", "methanator_power_kw")):
                self.totals[label] += s[field] * self.dt_h
            methane = s["ch4_produced_total_kg"] - self.previous["ch4_produced_total_kg"]
            if s["solar_available_kw"] <= 1e-8:
                self.totals["zero_solar_electrolyser_kwh"] += s["electrolyser_power_kw"] * self.dt_h
                self.totals["zero_solar_battery_discharge_kwh"] += s["battery_to_electrolyser_kw"] * self.dt_h
                self.totals["zero_solar_ch4_produced_kg"] += methane
            else:
                self.totals["solar_present_ch4_produced_kg"] += methane
            for unit in ("electrolyser", "dac", "methanator"):
                self.totals[unit + "_on_minutes"] += (s[unit + "_power_kw"] > 1e-8) * self.dt_h * 60
        self.previous = s


def empty_case(*, seed, controller, condition, split, freeze, failure, elapsed=0):
    return {"schema": SCHEMA + "-case", "seed": seed, "controller": controller, "condition": condition, "variant": freeze["conditions"][condition], "split": split,
            "evaluation_freeze_sha256": freeze["sha256"], "execution_complete": False, "every_delivery_passed": False,
            "physical_invariants_passed": False, "delivery_and_physics_passed": False, "failure": failure,
            "decisions": 0, "physical_steps": 0, "wall_time_s": elapsed, "delivery_settlements": 0, "maximum_delivery_unmet_kg": None,
            "daily_returns_met": 0, "stock_deficits": {r: {"observed_daily_deficits": [], "sum": 0, "max": 0} for r in RESOURCES},
            "physical_metrics": {}, "economic_metrics": {}, "operating_metrics": {}, "minute_audit": {"all_physical_invariants_passed": False, "states_checked": 0},
            "economic_audit": {"passed": False}, "recording": None}


def write_record(output, record):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(gzip.compress(json.dumps(record, separators=(",", ":"), allow_nan=False).encode(), mtime=0))
    return file_pin(output)


def audit_intervention(env, action, info, trace):
    requested, _ = environment().decode_action(action, max_discharge_kw=env.config.battery.max_discharge_kw)
    request = requested.to_dict()
    expected = dict(request)
    if env.battery_mode == "disabled":
        expected.update(battery_share=0.0, battery_support_kw=0.0)
    witness = info["battery_intervention"]
    if witness["mode"] != env.battery_mode or witness["requested_action"] != request or witness["applied_action"] != expected or info["requested"]["action"] != request or info["direct_control"]["intervention_applied_action"] != expected:
        raise ValueError("Requested/applied experimental intervention evidence differs")
    if len(env.last_direct_minute_actions or ()) != 15:
        raise ValueError("Missing minute action witnesses")
    for minute in env.last_direct_minute_actions:
        if minute["requested_action"] != request or minute["intervention_applied_action"] != expected:
            raise ValueError("Minute-level experimental action differs")
    if env.battery_mode == "disabled" and any(abs(s.solar_to_battery_kw) > 1e-8 or abs(s.battery_to_electrolyser_kw) > 1e-8 for s in trace[1:]):
        raise ValueError("Disabled battery produced nonzero physical charge/discharge")


def evaluate_case(*, freeze_path, gate_path=None, controller, condition, split, seed, output, retain_minutes=False, max_decisions=None, deadline_utc=None):
    import numpy as np
    freeze = require_freeze(freeze_path)
    panel = admit(freeze, controller=controller, condition=condition, split=split, seed=seed, gate_path=gate_path, max_decisions=max_decisions)
    if retain_minutes and seed != panel["seeds"][0]:
        raise ValueError("Full trace selection was fixed to the first scenario")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    deadline = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00")) if deadline_utc else None
    if deadline and deadline.tzinfo is None:
        raise ValueError("Deadline must include timezone")
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
        if spec["kind"] == "ppo":
            from docs.research.tools.free_ppo_v1 import train
            train.configure_torch()
            model = train.load_checkpoint(Path(spec["checkpoint"]["checkpoint"]["path"]))
            if model.research_identity != spec["checkpoint_environment_identity"] or model.num_timesteps != spec["checkpoint_timesteps"]:
                raise ValueError("Loaded policy differs from frozen identity")
        else:
            from docs.research.tools.comparison_ppo_v2.baselines import BaselinePolicy, BaselineConfig
            model = BaselinePolicy(spec["baseline"], config=BaselineConfig.from_plant_config(env.config))
        for index in range(max_decisions or 672):
            if deadline and datetime.now(timezone.utc) >= deadline:
                raise TimeoutError("Campaign UTC deadline reached; partial attempt retained")
            if not np.array_equal(obs, env.current_policy_observation):
                raise ValueError("Policy input differs from actual current observation")
            t = time.perf_counter()
            action, _ = model.predict(obs.copy(), deterministic=True)
            evidence = {"observation_sha256": canonical_sha(obs.tolist()), "action": np.asarray(action).tolist()}
            if spec["kind"] == "ppo":
                details = model.deterministic_details(obs.copy())
                if not np.array_equal(action, details["clipped_action"]):
                    raise ValueError("Recorded inference differs from prediction")
                mean = np.asarray(details["unclipped_mean"])
                clipped_means += int(np.any(mean != action))
                evidence.update(normalized_observation_sha256=canonical_sha(np.asarray(details["normalized_input"]).tolist()), raw_mean=mean.tolist())
                if retain_minutes:
                    evidence["normalized_observation"] = np.asarray(details["normalized_input"]).tolist()
            if retain_minutes:
                evidence["observation"] = obs.tolist()
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
            "inference": {"verified_decisions": len(commands), "actual_observation_inputs": True, "raw_mean_clipped_decisions": clipped_means, "witness_sha256": input_digest.hexdigest()},
            "baseline_metadata": model.metadata() if spec["kind"] == "baseline" and model is not None else None,
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
    p.add_argument("--controller", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--split", choices=("train", "development", "held_out"), required=True)
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
        split=args.split, seed=args.seed, output=args.output, retain_minutes=args.retain_minutes, max_decisions=args.max_decisions, deadline_utc=args.deadline_utc)
    publish(args.result, result)
    print(json.dumps({"controller": args.controller, "condition": args.condition, "seed": args.seed, "complete": result["execution_complete"], "failure": result["failure"]}), flush=True)


if __name__ == "__main__":
    main()

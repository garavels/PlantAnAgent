"""Read-only integrity checks and exact replays of already completed tail cases.

Run from the repository root or an extracted package in a fresh interpreter.
Original absolute file paths remain provenance; matching relative members are
resolved against this module's root and authenticated before use.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from docs.research.tools.wholeplant_ppo_v1 import evaluate, training
from docs.research.tools.wholeplant_ppo_v1.baselines import baseline
from docs.research.tools.wholeplant_ppo_v1.portable import canonical, digest, file_pin, file_sha
from docs.research.tools.tail_stress_v1.environment import TailStressEnv, SEEDS, CONDITIONS, condition_spec
from docs.research.tools.tail_stress_v1.audit import MinuteAudit

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS = ROOT / "docs/research/results/tail_stress_20260918"
CONTROLLERS = ("daylight", "storage", "ppo-101", "ppo-211", "ppo-307")


def same(actual, expected, label):
    if canonical(actual) != canonical(expected):
        raise ValueError(f"Verification differs: {label}")


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


class Evidence:
    def __init__(self, results):
        self.results = Path(results).resolve()
        self.freeze_path = self.results / "freeze.json"
        self.spec = json.loads(self.freeze_path.read_text())
        suffix = "/docs/research/tools/tail_stress_v1/environment.py"
        originals = [r["path"][:-len(suffix)] for r in self.spec["sources"] if r["path"].endswith(suffix)]
        if len(originals) != 1:
            raise ValueError("No unique original source root")
        self.original_root = Path(originals[0])
        self.verify_sources()

    def resolve(self, pin):
        original = Path(pin["path"])
        relative = original.relative_to(self.original_root)
        path = ROOT / relative
        if ".." in relative.parts or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError("Evidence member escapes the relocated root")
        if path.stat().st_size != pin["bytes"] or file_sha(path) != pin["sha256"]:
            raise ValueError("Pinned evidence changed: " + str(relative))
        return path

    def verify_sources(self):
        same(self.spec["sha256"], digest({k: v for k, v in self.spec.items() if k != "sha256"}), "freeze digest")
        for record in [self.spec["original_evaluation"], *self.spec["sources"]]:
            self.resolve(record)
        for controller in self.spec["controllers"].values():
            for record in controller.get("checkpoint", {}).values():
                self.resolve(record)

    def read(self, seed, condition, controller):
        path = self.results / "cases" / f"{condition}--{controller}--{seed}" / "result.json"
        row = json.loads(path.read_text())
        same([row[k] for k in ("seed", "condition", "controller")], [seed, condition, controller], "case identity")
        same(row["freeze_sha256"], self.spec["sha256"], "case freeze")
        with gzip.open(self.resolve(row["recording"]), "rt") as stream:
            recording = json.load(stream)
        same(recording["summary"], {k: v for k, v in row.items() if k != "recording"}, "recording and result summary")
        return row, recording


def replay_one(evidence, condition, controller, *, deadline):
    seed = SEEDS[0]
    row, saved = evidence.read(seed, condition, controller)
    if len(saved["minute_truth_states"]) != 10081 or len(saved["decision_trace"]) != 672:
        raise ValueError("Prescribed replay requires a complete retained physical trace")
    cs = evidence.spec["controllers"][controller]
    env = policy = None
    started = time.monotonic()
    try:
        env = TailStressEnv(condition, record_minutes=True)
        obs, _ = env.reset(seed=seed)
        initial = env.simulator.state.to_dict()
        same(initial, saved["minute_truth_states"][0], "initial physical state")
        audit, operating = MinuteAudit(env), evaluate.OperatingAudit(env)
        audit.add(initial); operating.add(initial)
        if cs["kind"] == "ppo":
            policy = training.load_checkpoint(evidence.resolve(cs["checkpoint"]["checkpoint"]))
            same(policy.research_identity, cs["checkpoint_environment_identity"], "policy research identity")
            same(policy.num_timesteps, 786432, "policy training endpoint")
        else:
            policy = baseline(controller, env)
        witness = hashlib.sha256()
        reward_sum = 0.0
        states_checked = 1
        event = {}
        timing = condition_spec(condition)
        onset = timing["onset_h"] if timing["onset_h"] is not None else 48.0
        end = timing["end_h"] if timing["end_h"] is not None else 48.0
        for index, expected in enumerate(saved["decision_trace"]):
            if time.monotonic() >= deadline:
                raise TimeoutError("Replay budget exhausted")
            if obs.shape != (149,) or not np.array_equal(obs, env.current_policy_observation):
                raise ValueError("Replay input is not the actual issued149-vector")
            action, _ = policy.predict(obs.copy(), deterministic=True)
            details = {"observation": obs.tolist(), "observation_sha256": digest(obs.tolist())}
            if cs["kind"] == "ppo":
                inference = policy.deterministic_details(obs.copy())
                if not np.array_equal(action, inference["clipped_action"]):
                    raise ValueError("PPO replay action is not its clipped deterministic mean")
                details.update(normalized_observation=inference["normalized_input"].tolist(),
                               raw_mean=inference["unclipped_mean"].tolist())
            else:
                checked, inference = policy.inference(obs.copy())
                if not np.array_equal(action, checked):
                    raise ValueError("Baseline replay witness differs")
                details["baseline_inference"] = inference
            details["action"] = np.asarray(action).tolist()
            same(details, expected["evidence"], f"decision{index} full inference witness")
            witness.update(canonical(details))
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            same({"decision_index": index, "evidence": details, "action": np.asarray(action).tolist(),
                  "reward_eur": float(reward)}, expected, f"decision{index} action and reward")
            trace = env.last_minute_truth_states
            if trace is None or len(trace) != 16:
                raise ValueError("Replay missing minute interval")
            evaluate.audit_intervention(env, action, info, trace)
            for state in trace[1:]:
                audit.add(state); operating.add(state)
                same(state.to_dict(), saved["minute_truth_states"][states_checked], f"minute{states_checked}")
                states_checked += 1
            for name, hour in (("pre_event_state", onset), ("post_event_state", end)):
                if env.physics_steps_taken == round(hour * 60):
                    event[name] = env.simulator.state.to_dict()
            if terminated != (index == 671) or truncated:
                raise ValueError("Replay episode boundary differs")
        final = env.simulator.state.to_dict()
        comparisons = {
            "initial_state": initial, "final_state": final, "delivery": env.simulator.delivery_audit(),
            "economic_metrics": env.economic_metrics.to_dict(), "physical_metrics": env.metrics.to_dict(),
            "operating_metrics": operating.totals, "minute_audit": audit.result(),
            "economic_audit": evaluate.economic_audit(env, initial, final, reward_sum, operating),
            "base_paired_source_sha256": env.base_source_manifest()["base_paired_source_sha256"],
            "physical_solar_forcing_sha256": digest([float(v) for v in env.frozen_source.expected_solar_kw]),
            "input_action_witness_sha256": witness.hexdigest(),
        }
        for key, value in comparisons.items():
            same(value, row[key], key)
        for key, value in event.items():
            same(value, row["event_metrics"][key], key)
        return {"schema": "tail-stress-exact-replay-v1", "verified": True,
            "scientific_outcome_attempt": False, "seed": seed, "condition": condition, "controller": controller,
            "decisions": 672, "physical_states": states_checked, "all149inputs_and7actions_exact": True,
            "normalizer_and_actor_witness_exact": cs["kind"] == "ppo",
            "all_interval_rewards_exact": True, "every_physical_state_exact": True,
            "final_ledgers_and_audits_exact": True, "event_states_exact": True,
            "physical_stream_sha256": audit.result()["stream_sha256"],
            "input_action_witness_sha256": witness.hexdigest(), "recording": row["recording"],
            "freeze_sha256": evidence.spec["sha256"], "wall_time_s": time.monotonic() - started,
            "completed_at": datetime.now(timezone.utc).isoformat(), "verifier": file_pin(__file__)}
    finally:
        if policy is not None and hasattr(policy, "close"):
            policy.close()
        if env is not None:
            env.close()


def replay_panel(evidence, output, *, max_seconds):
    training.configure_torch()
    deadline = time.monotonic() + max_seconds
    records = []
    for condition in CONDITIONS:
        for controller in CONTROLLERS:
            receipt = Path(output) / f"replay-{condition}--{controller}.json"
            result = replay_one(evidence, condition, controller, deadline=deadline)
            publish(receipt, result)
            records.append(file_pin(receipt))
            print(json.dumps({"replayed": len(records), "condition": condition, "controller": controller,
                              "wall_time_s": result["wall_time_s"]}), flush=True)
    evidence.verify_sources()
    publish(Path(output) / "replay-summary.json", {"schema": "tail-stress-replay-panel-v1", "verified": True,
        "case_count": 30, "physical_states": 30*10081, "decisions": 30*672,
        "freeze_sha256": evidence.spec["sha256"], "receipts": records,
        "completed_at": datetime.now(timezone.utc).isoformat(), "verifier": file_pin(__file__)})


def verify_all(evidence, output):
    """Inspect every saved witness and compare all stress prefixes with controls."""
    summary = json.loads((evidence.results / "summary.json").read_text())
    if summary["case_count"] != 480 or not summary["all_executions_and_audits_passed"]:
        raise ValueError("The canonical panel is incomplete or unaudited")
    index = {(r["seed"], r["condition"], r["controller"]): r for r in summary["rows"]}
    expected = {(s, c, p) for s in SEEDS for c in CONDITIONS for p in CONTROLLERS}
    if set(index) != expected or len(summary["rows"]) != 480:
        raise ValueError("Summary case membership differs")
    count = prefix_count = full_state_prefix_count = decisions = 0
    for seed in SEEDS:
        base_hashes, opening_hashes = set(), set()
        for controller in CONTROLLERS:
            control, control_record = evidence.read(seed, "control", controller)
            control_commands = control_record["decision_trace"]
            for condition in CONDITIONS:
                row, recording = ((control, control_record) if condition == "control"
                                  else evidence.read(seed, condition, controller))
                same(row, index[(seed, condition, controller)], "summary row")
                if not (row["execution_complete"] and row["sources_unchanged"] and row["failure"] is None
                        and row["minute_audit"]["all_physical_invariants_passed"]
                        and row["economic_audit"]["passed"]):
                    raise ValueError("Unaudited case")
                if row["decisions"] != 672 or row["physical_states"] != 10081 or row["minute_audit"]["states_checked"] != 10081:
                    raise ValueError("Case geometry differs")
                trace = recording["decision_trace"]
                if len(trace) != 672:
                    raise ValueError("Missing recorded decisions")
                witness = hashlib.sha256(); total_reward = 0.0
                for i, command in enumerate(trace):
                    same(command["decision_index"], i, "decision order")
                    details = command["evidence"]
                    obs, action = np.asarray(details["observation"]), np.asarray(details["action"])
                    if obs.shape != (149,) or action.shape != (7,) or not np.isfinite(obs).all() or not np.isfinite(action).all():
                        raise ValueError("Invalid stored observation/action")
                    same(details["observation_sha256"], digest(details["observation"]), "input digest")
                    same(details["action"], command["action"], "stored action copies")
                    if controller.startswith("ppo-"):
                        normalized, mean = np.asarray(details["normalized_observation"]), np.asarray(details["raw_mean"])
                        if normalized.shape != (149,) or mean.shape != (7,) or not np.isfinite(normalized).all() or not np.isfinite(mean).all():
                            raise ValueError("Invalid stored PPO inference")
                        same(np.clip(mean, -1, 1).tolist(), action.tolist(), "actor clipping")
                    else:
                        same(details["baseline_inference"]["actual_observation"], details["observation"], "baseline actual input")
                    witness.update(canonical(details)); total_reward += command["reward_eur"]
                same(witness.hexdigest(), row["input_action_witness_sha256"], "complete stored witness stream")
                if abs(total_reward-row["economic_metrics"]["net_operating_value_eur"]) > 1e-6:
                    raise ValueError("Stored rewards do not sum to economic total")
                settlements = row["delivery"]["settlements"]
                if len(settlements) != 28 or any(d["due_step"] != (i+1)*360 or d["requested_kg"] != 180 for i,d in enumerate(settlements)):
                    raise ValueError("Delivery schedule differs")
                same(row["every_delivery_passed"], all(d["unmet_kg"] <= 1e-9 for d in settlements), "delivery success label")
                if abs(sum(d["unmet_kg"] for d in settlements)-row["economic_metrics"]["unmet_kg"]) > 1e-6:
                    raise ValueError("Unmet kilograms do not reconcile")
                physical = recording["minute_truth_states"]
                if len(physical) != (10081 if seed == SEEDS[0] else 0):
                    raise ValueError("Prescribed full-trace retention differs")
                if physical:
                    state_hash = hashlib.sha256()
                    for i, state in enumerate(physical):
                        same(state["step"], i, "physical state order")
                        raw = canonical(state); state_hash.update(len(raw).to_bytes(8, "big")); state_hash.update(raw)
                    same(state_hash.hexdigest(), row["minute_audit"]["stream_sha256"], "retained physical stream hash")
                    same(physical[0], row["initial_state"], "first retained state")
                    same(physical[-1], row["final_state"], "last retained state")
                if condition != "control":
                    onset = int(condition_spec(condition)["onset_h"]*4)
                    same(trace[:onset], control_commands[:onset], "endogenous pre-event decisions and rewards")
                    same(trace[onset]["evidence"], control_commands[onset]["evidence"], "no early warning at onset action")
                    if physical:
                        same(physical[:onset*15+1], control_record["minute_truth_states"][:onset*15+1], "endogenous physical prefix")
                        full_state_prefix_count += 1
                    prefix_count += 1
                base_hashes.add(row["base_paired_source_sha256"])
                opening_hashes.add(digest(row["initial_state"]))
                count += 1; decisions += len(trace)
        if len(base_hashes) != 1 or len(opening_hashes) != 1:
            raise ValueError("Underlying sources or openings differ across paired controllers/conditions")
        for condition in CONDITIONS:
            if len({index[(seed, condition, p)]["physical_solar_forcing_sha256"] for p in CONTROLLERS}) != 1:
                raise ValueError("Physical forcing differs across controllers")
        print(json.dumps({"verified_seed": seed, "cases_checked": count}), flush=True)
    evidence.verify_sources()
    report = {"schema": "tail-stress-postflight-review-v1", "verified": True,
        "completed_at": datetime.now(timezone.utc).isoformat(), "cases_checked": count,
        "decisions_checked": decisions, "source_and_checkpoint_pins_unchanged": True,
        "recording_pins_and_witness_hashes_valid": True, "physical_and_economic_audits_passed": True,
        "all_delivery_ledgers_reconciled": True, "paired_sources_openings_and_forcing_equal": True,
        "stress_control_input_action_prefixes_exact": prefix_count,
        "first_seed_stress_control_physical_prefixes_exact": full_state_prefix_count,
        "prefix_scope": "Every stress cell has exact actual input/action and reward prefixes; full hidden physical prefixes are available only for the first prescribed seed",
        "freeze": file_pin(evidence.freeze_path), "summary": file_pin(evidence.results/"summary.json"),
        "verifier": file_pin(__file__)}
    publish(Path(output)/"review.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("replay", "all"))
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-seconds", type=float, default=300)
    args = parser.parse_args()
    evidence = Evidence(args.results)
    output = args.output or args.results/"postflight"
    if args.command == "replay":
        replay_panel(evidence, output, max_seconds=args.max_seconds)
    else:
        print(json.dumps(verify_all(evidence, output), indent=2))


if __name__ == "__main__":
    main()

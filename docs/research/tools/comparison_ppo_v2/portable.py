"""Verify relative bundle contents and replay a prescribed comparison trajectory.

Original absolute paths are preserved as provenance. No original-host source or
freeze lookup is performed. Exact equality is tested under the installed replay
runtime, not promised across hardware or dependency versions.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCHEMA = "comparison-ppo-v2-portable-bundle"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def member_path(relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Invalid bundle-relative path")
    result = ROOT / relative
    if result.is_symlink() or not result.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Bundle member escaped its root")
    return result


def verify_bundle():
    manifest = json.loads((ROOT / "bundle-manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unknown comparison bundle schema")
    if hashlib.sha256(canonical({k: v for k, v in manifest.items() if k != "sha256"})).hexdigest() != manifest["sha256"]:
        raise ValueError("Bundle manifest checksum differs")
    seen = set()
    for row in manifest["files"]:
        if row["path"] in seen:
            raise ValueError("Duplicate bundle-relative path")
        seen.add(row["path"])
        path = member_path(row["path"])
        if not path.is_file() or path.stat().st_size != row["bytes"] or file_sha(path) != row["sha256"]:
            raise ValueError(f"Bundle content changed: {row['path']}")
    for item in manifest["prescribed_replays"]:
        for key in ("recording", "freeze", "gate", "checkpoint"):
            if item.get(key) is not None and item[key] not in seen:
                raise ValueError(f"Replay references an unlisted member: {key}")
    for source_set in manifest["verified_scientific_source_sets"]:
        if source_set["record"] not in seen:
            raise ValueError("A scientific source set references an unlisted record")
        record = json.loads(member_path(source_set["record"]).read_text())
        pins = record["sources" if source_set["kind"] == "evaluation_freeze" else "source_files"]
        if len(pins) != source_set["files"]:
            raise ValueError("Scientific source count differs")
        for pin in pins:
            recorded_pin_path(pin, manifest)
        if source_set["kind"] == "evaluation_freeze":
            for controller in record["controllers"].values():
                if controller["kind"] == "ppo":
                    for pin in controller["checkpoint"].values():
                        recorded_pin_path(pin, manifest)
    supplement = manifest.get("forecast_ablation", {"status": "not_declared"})
    for key in ("freeze", "analysis", "completion", "deadline_receipt"):
        if supplement.get(key) is not None and supplement[key] not in seen:
            raise ValueError(f"Supplement references an unlisted member: {key}")
    if supplement["status"] == "completed":
        analysis = json.loads(member_path(supplement["analysis"]).read_text())
        completion = json.loads(member_path(supplement["completion"]).read_text())
        if (len(analysis["rows"]) != 32 or supplement["rows"] != 32
                or manifest["main_analysis_rows"] != 1160
                or completion.get("schema") != "forecast-ablation-finish-completion"
                or completion.get("status") != "completed" or completion.get("rows") != 32
                or recorded_pin_path(completion["analysis"], manifest) != member_path(supplement["analysis"])
                or recorded_pin_path(analysis["reference_main_analysis"], manifest) != member_path(manifest["analysis"])):
            raise ValueError("Supplementary portable completion or main reference differs")
    return manifest


def recorded_pin_path(pin, manifest):
    """Resolve original provenance through the verified relative content map."""
    original = Path(pin["path"])
    root = Path(manifest["original_root"])
    if not original.is_absolute() or not original.is_relative_to(root):
        raise ValueError("Recorded scientific pin is outside the original bundle root")
    relative = original.relative_to(root).as_posix()
    members = {item["path"]: item for item in manifest["files"]}
    member = members.get(relative)
    if member is None or any(member[key] != pin[key] for key in ("bytes", "sha256")):
        raise ValueError("Recorded scientific pin differs from portable content")
    return member_path(relative)


def verify_supplement_admission(freeze, row, manifest, gate):
    """Recheck the original execution admission without original-host lookups."""
    admission = row["main_campaign_admission"]
    local = {name: recorded_pin_path(pin, manifest) for name, pin in admission.items()}
    if local["gate"] != gate:
        raise ValueError("Supplement references another main held-out gate")
    gate_record = json.loads(gate.read_text())
    completion = json.loads(local["main_completion"].read_text())
    contract = freeze["late_gate_contract"]
    if (completion.get("schema") != contract["completion_schema"]
            or completion.get("status") != "completed" or completion.get("outcomes") != 1160
            or gate_record.get("schema") != contract["schema"]
            or gate_record.get("endpoints_frozen") is not True
            or datetime.fromisoformat(gate_record["created_at"]) <= datetime.fromisoformat(freeze["created_at"])
            or completion["endpoints"] != gate_record["endpoints"]
            or admission["endpoints"] != gate_record["endpoints"]
            or admission["training_closure"] != gate_record["training_closure"]
            or admission["main_analysis"] != completion["analysis"]):
        raise ValueError("Supplement admission differs from the frozen main campaign closure")
    stages = completion.get("stages", [])
    if (not {"development-learning", "held-out-primary", "held-out-battery", "analysis"}.issubset({s.get("stage") for s in stages})
            or any(stage.get("exit_code") != 0 for stage in stages)):
        raise ValueError("Supplement was admitted by an incomplete main campaign")


def replay(controller="ppo25-101", condition="d30-sampled", *, split="held_out"):
    import numpy as np
    from docs.research.tools.comparison_ppo_v2 import environment, evaluate
    from docs.research.tools.free_ppo_v1 import train

    manifest = verify_bundle()
    matches = [r for r in manifest["prescribed_replays"]
               if (r["controller"], r["condition"], r["split"]) == (controller, condition, split)]
    if len(matches) != 1:
        raise ValueError("No unique prescribed full trace for this controller/condition/split")
    prescribed = matches[0]
    if split == "train" and not manifest["engineering_fixture"]:
        raise ValueError("Training fixtures must use an explicitly labeled engineering bundle")
    recorded = json.loads(gzip.decompress(member_path(prescribed["recording"]).read_bytes()))
    row = recorded["summary"]
    if (row["controller"], row["condition"], row["split"], row["seed"]) != (
            controller, condition, split, prescribed["scenario_seed"]):
        raise ValueError("Prescribed trajectory identity differs")
    if (not row["execution_complete"] or len(recorded["decision_trace"]) != 672
            or len(recorded["minute_truth_states"] or []) != 10081):
        raise ValueError("Replay requires a complete retained seven-day trajectory")
    freeze = json.loads(member_path(prescribed["freeze"]).read_text())
    if hashlib.sha256(canonical({k: v for k, v in freeze.items() if k != "sha256"})).hexdigest() != freeze["sha256"]:
        raise ValueError("Retained evaluation freeze checksum differs")
    if freeze["sha256"] != row["evaluation_freeze_sha256"]:
        raise ValueError("Trajectory belongs to another evaluation freeze")
    for pin in freeze["sources"]:
        recorded_pin_path(pin, manifest)
    supplementary = freeze.get("schema") == "forecast-ablation-v1-freeze"
    if not supplementary and freeze.get("schema") != evaluate.SCHEMA + "-freeze":
        raise ValueError("Unknown replay evaluation family")
    spec = freeze["controllers"][controller]
    if row["variant"] != freeze["conditions"][condition] or row["variant"] != prescribed["variant"]:
        raise ValueError("Replay condition differs from its frozen intervention")
    panel = [p for p in freeze["panels"] if (p["controller"], p["condition"], p["split"]) == (controller, condition, split)]
    if len(panel) != 1 or row["seed"] != panel[0]["seeds"][0]:
        raise ValueError("Full trace must be the first prospectively prescribed scenario")
    train.configure_torch()
    model = None
    gate = member_path(prescribed["gate"]) if prescribed.get("gate") else None
    if supplementary and split == "held_out":
        verify_supplement_admission(freeze, row, manifest, gate)
    env = environment.ComparisonEnv(**row["variant"], split=split, gate_path=gate, record_minutes=True)
    states, evidence_digest = [], hashlib.sha256()
    actions_exact = rewards_exact = inputs_exact = intervention_exact = True
    reward_sum = 0.0
    terminated = truncated = False
    try:
        if spec["kind"] == "ppo":
            model = train.load_checkpoint(member_path(prescribed["checkpoint"]))
            # A PPO25 checkpoint remains a PPO25 checkpoint when evaluated on
            # another demand or initial condition; no false identity equivalence.
            if model.research_identity != spec["checkpoint_environment_identity"] or model.num_timesteps != spec["checkpoint_timesteps"]:
                raise ValueError("Loaded policy differs from its own frozen training identity")
        else:
            from docs.research.tools.comparison_ppo_v2.baselines import BaselinePolicy, BaselineConfig
            if supplementary:
                from docs.research.tools.forecast_ablation_v1.policy import PersistencePolicy
                if spec["baseline"] != "daylight-persistence":
                    raise ValueError("Unknown supplementary policy")
                model = PersistencePolicy(config=BaselineConfig.from_plant_config(env.config))
            else:
                model = BaselinePolicy(spec["baseline"], config=BaselineConfig.from_plant_config(env.config))
            if model.metadata() != row["baseline_metadata"]:
                raise ValueError("Baseline constants or declared inputs changed")
        observation, _ = env.reset(seed=row["seed"])
        if env.paired_source_manifest() != row["paired_source_manifest"]:
            raise ValueError("Weather, faults or original inventories differ after relocation")
        states.append(env.simulator.state.to_dict())
        for decision in recorded["decision_trace"]:
            if supplementary:
                action, evidence = model.inference(observation.copy())
                predicted, _ = model.predict(observation.copy(), deterministic=True)
                inputs_exact &= bool(np.array_equal(action, predicted))
            else:
                action, _ = model.predict(observation.copy(), deterministic=True)
                evidence = {"observation_sha256": hashlib.sha256(canonical(observation.tolist())).hexdigest(),
                            "action": np.asarray(action).tolist()}
            if spec["kind"] == "ppo":
                details = model.deterministic_details(observation.copy())
                evidence.update(normalized_observation_sha256=hashlib.sha256(canonical(np.asarray(details["normalized_input"]).tolist())).hexdigest(),
                                raw_mean=np.asarray(details["unclipped_mean"]).tolist(),
                                normalized_observation=np.asarray(details["normalized_input"]).tolist())
                inputs_exact &= bool(np.array_equal(action, details["clipped_action"]))
            evidence["observation"] = observation.tolist()
            inputs_exact &= evidence == decision["evidence"]
            inputs_exact &= bool(np.array_equal(observation, env.current_policy_observation))
            evidence_digest.update(canonical(evidence))
            actions_exact &= np.asarray(action).tolist() == decision["action"]
            observation, reward, terminated, truncated, info = env.step(action)
            evaluate.audit_intervention(env, action, info, env.last_minute_truth_states)
            intervention_exact &= info["battery_intervention"] == decision["battery_intervention"]
            intervention_exact &= info["direct_control"] == decision["direct_control"]
            rewards_exact &= reward == decision["reward_eur"]
            reward_sum += reward
            states.extend(state.to_dict() for state in env.last_minute_truth_states[1:])
        physical_digest = hashlib.sha256(canonical(states)).hexdigest()
        physical_exact = physical_digest == row["retained_minute_truth_sha256"] == prescribed["trace_sha256"]
        physical_exact &= states == recorded["minute_truth_states"]
        inference_exact = bool(inputs_exact and evidence_digest.hexdigest() == row["inference"]["witness_sha256"])
        complete = bool(terminated and not truncated and env.decisions_taken == 672 and len(states) == 10081)
        checks = {"actions_exact": bool(actions_exact), "interval_rewards_exact": bool(rewards_exact),
                  "inference_witness_exact": inference_exact, "requested_and_applied_intervention_exact": bool(intervention_exact),
                  "physical_states_exact": bool(physical_exact), "complete_seven_day_replay": complete,
                  "economic_ledger_exact": env.economic_metrics.to_dict() == row["economic_metrics"]}
        if not all(checks.values()):
            raise ValueError(f"Portable replay diverged: {checks}")
        packages = {name: importlib.metadata.version(name) for name in manifest["recorded_runtime"]["packages"]}
        return {"schema": "comparison-ppo-v2-portable-replay", "controller": controller,
                "evaluation_family": freeze["schema"], "supplementary": supplementary,
                "training_family": spec.get("training_family"), "condition": condition, "split": split,
                "scenario_seed": row["seed"], "variant": row["variant"], **checks,
                "decisions": env.decisions_taken, "physical_states": len(states),
                "physical_trace_sha256": physical_digest, "inference_witness_sha256": evidence_digest.hexdigest(),
                "economic_reward_eur": reward_sum, "bundle_manifest_sha256": manifest["sha256"],
                "engineering_fixture": manifest["engineering_fixture"],
                "runtime": {"python": sys.version, "platform": platform.platform(), "packages": packages},
                "dependency_versions_match_recorded": packages == manifest["recorded_runtime"]["packages"],
                "interpretation": "Exact replay verified for this execution; no cross-hardware or cross-version bitwise guarantee."}
    finally:
        env.close()
        if model is not None and hasattr(model, "close"):
            model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "replay"))
    parser.add_argument("--controller", default="ppo25-101")
    parser.add_argument("--condition", default="d30-sampled")
    parser.add_argument("--split", choices=("train", "development", "held_out"), default="held_out")
    args = parser.parse_args()
    if args.command == "verify":
        manifest = verify_bundle()
        result = {"verified": True, "files": len(manifest["files"]), "manifest_sha256": manifest["sha256"],
                  "engineering_fixture": manifest["engineering_fixture"],
                  "forecast_ablation": manifest.get("forecast_ablation", {"status": "not_declared"}),
                  "prescribed_replays": [{k: r[k] for k in ("controller", "condition", "split", "scenario_seed")} for r in manifest["prescribed_replays"]]}
    else:
        result = replay(args.controller, args.condition, split=args.split)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

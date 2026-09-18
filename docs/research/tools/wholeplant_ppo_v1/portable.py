"""Verify relocated whole-plant evidence and replay a prescribed retained case.

Absolute paths in scientific records remain provenance. A verified manifest maps
those paths to immutable relative members; the records themselves are not edited.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCHEMA = "wholeplant-ppo-v1-portable-bundle"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def file_pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha(path)}


def member_path(relative, root=ROOT):
    relative, root = Path(relative), Path(root).resolve()
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Bundle member must be a relative path inside its root")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Bundle member escaped its root")
    return path


def rebind(record, manifest, root=ROOT):
    """Authenticate an original file pin and return its relocated path."""
    matches = [r for r in manifest["files"] if r.get("original_path") == record["path"]]
    if len(matches) != 1:
        raise ValueError(f"Original record has no unique bundled member: {record['path']}")
    found = matches[0]
    if any(found[k] != record[k] for k in ("bytes", "sha256")):
        raise ValueError("Original file pin differs from its bundled content")
    return member_path(found["path"], root)


def verify_bundle(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / "bundle-manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("sha256") != digest({k: v for k, v in manifest.items() if k != "sha256"}):
        raise ValueError("Invalid whole-plant bundle manifest")
    seen, originals = set(), set()
    for row in manifest["files"]:
        path = member_path(row["path"], root)
        if row["path"] in seen or (row.get("original_path") and row["original_path"] in originals):
            raise ValueError("Duplicate relative or original bundle path")
        seen.add(row["path"])
        if row.get("original_path"):
            originals.add(row["original_path"])
        if path.stat().st_size != row["bytes"] or file_sha(path) != row["sha256"]:
            raise ValueError(f"Bundle bytes changed: {row['path']}")
    for pin in manifest["scientific_bindings"]:
        rebind(pin, manifest, root)
    for key in ("release", "analysis", "held_out_freeze", "training_plan"):
        if manifest[key] not in seen:
            raise ValueError(f"Required member was not hashed: {key}")
    return manifest


def _same(actual, expected, label):
    if canonical(actual) != canonical(expected):
        raise ValueError(f"Replay differs: {label}")


def replay_recording(recording, spec, *, checkpoint=None, gate_path=None, engineering=False):
    """Replay one retained record. Engineering calls admit only training seeds."""
    import numpy as np
    from docs.research.tools.wholeplant_ppo_v1 import environment, evaluate, training
    from docs.research.tools.wholeplant_ppo_v1.baselines import baseline

    with gzip.open(recording, "rt") as stream:
        recorded = json.load(stream)
    row = recorded["summary"]
    if engineering:
        if row["split"] != "train" or not environment.TRAIN_BASE <= row["seed"] < environment.TRAIN_STOP:
            raise ValueError("Engineering replay is restricted to retained training cases")
    elif row["split"] != "held_out" or row["seed"] != environment.HELD_OUT_SEEDS[0]:
        raise ValueError("Portable scientific replay requires the prescribed first held-out scenario")
    commands = recorded["decision_trace"]
    retained_states = recorded.get("minute_truth_states") or []
    if not commands or len(retained_states) != 1 + 15 * len(commands):
        raise ValueError("Replay requires the complete retained minute trace")
    if not engineering and (not row["execution_complete"] or len(commands) != 672):
        raise ValueError("Prescribed scientific replay did not complete the week")
    if row["controller"] != spec["label"] or row["controller_kind"] != spec["kind"]:
        raise ValueError("Recording controller differs from selected specification")
    _same(digest(retained_states), row["retained_minute_truth_sha256"], "retained trace hash")
    env = model = None
    try:
        env = environment.WholePlantEnv(arm=row["condition"], split=row["split"], gate_path=gate_path, record_minutes=True)
        observation, _ = env.reset(seed=row["seed"])
        initial = env.simulator.state.to_dict()
        states = [initial]
        audit, operating = evaluate.MinuteAudit(env), evaluate.OperatingAudit(env)
        audit.add(initial); operating.add(initial)
        if spec["kind"] == "ppo":
            training.configure_torch()
            model = training.load_checkpoint(checkpoint)
            if (model.research_identity != environment.environment_identity()
                    or model.num_timesteps != row["checkpoint_timesteps"]
                    or int(model.seed) != row["optimizer_seed"]):
                raise ValueError("Replay policy differs from retained endpoint")
        else:
            model = baseline(spec["baseline"], env)
            _same(model.metadata(), row["baseline_metadata"], "baseline metadata")
        reward_sum, witness = 0.0, hashlib.sha256()
        terminated = truncated = False
        for index, expected in enumerate(commands):
            if (np.asarray(observation).shape != (149,)
                    or not np.array_equal(observation, env.current_policy_observation)):
                raise ValueError("Replay input is not the actual public 149-vector")
            action, _ = model.predict(observation.copy(), deterministic=True)
            if np.asarray(action).shape != (7,) or not np.isfinite(action).all():
                raise ValueError("Replay policy action is not seven finite values")
            evidence = {"observation_sha256": digest(observation.tolist()),
                        "observation": observation.tolist(), "action": np.asarray(action).tolist()}
            if spec["kind"] == "baseline":
                verified, details = model.inference(observation.copy())
                if not np.array_equal(action, verified) or not np.array_equal(observation, details["actual_observation"]):
                    raise ValueError("Baseline input/action witness differs")
                evidence.update(baseline_inference=details,
                                controller_input_sha256=digest(np.asarray(details["controller_input"]).tolist()))
            else:
                details = model.deterministic_details(observation.copy())
                if not np.array_equal(action, details["clipped_action"]):
                    raise ValueError("PPO action differs from clipped actor mean")
                evidence.update(normalized_observation_sha256=digest(np.asarray(details["normalized_input"]).tolist()),
                                raw_mean=np.asarray(details["unclipped_mean"]).tolist(),
                                normalized_observation=np.asarray(details["normalized_input"]).tolist())
            _same(evidence, expected["evidence"], f"decision {index} actual inference evidence")
            witness.update(canonical(evidence))
            observation, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            trace = env.last_minute_truth_states
            if trace is None or len(trace) != 16:
                raise ValueError("Missing 15-minute physical/action witness")
            evaluate.audit_intervention(env, action, info, trace)
            for state in trace[1:]:
                audit.add(state); operating.add(state); states.append(state.to_dict())
            actual = {"decision_index": index, "action": np.asarray(action).tolist(), "evidence": evidence,
                      "reward_eur": reward, "direct_control": info["direct_control"],
                      "battery_intervention": info.get("battery_intervention"),
                      "state": {k: getattr(env.simulator.state, k) for k in expected["state"]}}
            _same(actual, expected, f"decision {index} requested/applied actions, reward and state")
            if (terminated or truncated) and index != len(commands)-1:
                raise ValueError("Replay terminated before its retained endpoint")
        _same(witness.hexdigest(), row["inference"]["witness_sha256"], "inference stream hash")
        _same(states, retained_states, "every retained minute state")
        _same(digest(states), row["retained_minute_truth_sha256"], "replayed minute trace hash")
        final = env.simulator.state.to_dict()
        comparisons = {
            "initial_state": initial, "final_state": final,
            "physical_metrics": env.metrics.to_dict(), "economic_metrics": env.economic_metrics.to_dict(),
            "operating_metrics": operating.totals, "minute_audit": audit.result(),
            "economic_audit": evaluate.economic_audit(env, initial, final, reward_sum, operating),
            "paired_source_manifest": env.paired_source_manifest(),
        }
        for name, value in comparisons.items():
            _same(value, row[name], name)
        _same(env.simulator.delivery_audit(), recorded["delivery"], "delivery ledger")
        _same([s.to_dict() for s in env.cycle_settlements], recorded["cycle_settlements"], "cycle settlements")
        complete = bool(terminated and not truncated and env.decisions_taken == 672 and env.physics_steps_taken == 10080 and audit.count == 10081)
        if not engineering and not complete:
            raise ValueError("Scientific replay did not complete all 10080 physical steps")
        if complete != row["execution_complete"]:
            raise ValueError("Replay completion differs from recorded case")
        return {"schema": "wholeplant-ppo-v1-portable-replay", "engineering_fixture": engineering,
                "canonical_scientific_outcome": False, "controller": row["controller"], "arm": row["condition"],
                "scenario_seed": row["seed"], "decisions": env.decisions_taken, "physical_states": len(states),
                "complete_week": complete, "actual_inputs_and_inference_exact": True,
                "requested_and_applied_actions_exact": True, "interval_rewards_exact": True,
                "physical_states_exact": True, "final_accounting_exact": True,
                "economic_reward_eur": reward_sum, "inference_witness_sha256": witness.hexdigest(),
                "physical_trace_sha256": digest(states), "recording": file_pin(recording)}
    finally:
        if model is not None and hasattr(model, "close"):
            model.close()
        if env is not None:
            env.close()


def replay(controller="storage", arm="shared-dynamic", root=ROOT):
    if Path(root).resolve() != ROOT.resolve():
        raise ValueError("Invoke portable replay from the extracted module in a fresh Python process; root cannot redirect already imported scientific code")
    manifest = verify_bundle(root)
    selected = manifest["prescribed_replays"][controller + "/" + arm]
    if not selected["available"]:
        raise ValueError(selected["unavailable_reason"])
    freeze = json.loads(member_path(manifest["held_out_freeze"], root).read_text())
    _same(freeze["sha256"], digest({k: v for k, v in freeze.items() if k != "sha256"}), "evaluation freeze digest")
    from docs.research.tools.wholeplant_ppo_v1 import environment
    if environment.ROOT.resolve() != ROOT.resolve():
        raise ValueError("Replay imported science from another source tree")
    _same(freeze["source_identity_sha256"], environment.source_identity_sha256(), "relocated source identity")
    _same(freeze["environment_identity_sha256"], environment.environment_identity_sha256(), "environment identity")
    for record in freeze["sources"]:
        rebind(record, manifest, root)
    panels = [p for p in freeze["panels"] if (p["split"], p["controller"], p["condition"]) == ("held_out", controller, arm)]
    if len(panels) != 1 or selected["scenario_seed"] != panels[0]["seeds"][0]:
        raise ValueError("Replay selection differs from first prescribed case")
    spec = freeze["controllers"][controller]
    checkpoint = None
    if spec["kind"] == "ppo":
        for record in spec["checkpoint"].values():
            rebind(record, manifest, root)
        checkpoint = rebind(spec["checkpoint"]["checkpoint"], manifest, root)
    recording = rebind(selected["recording"], manifest, root)
    with gzip.open(recording, "rt") as stream:
        row = json.load(stream)["summary"]
    _same([row["split"], row["controller"], row["condition"], row["seed"]],
          ["held_out", controller, arm, selected["scenario_seed"]], "prescribed case identity")
    _same(row["evaluation_freeze_sha256"], freeze["sha256"], "recording evaluation freeze")
    _same(row["variant"], freeze["conditions"][arm], "recording physical arm")
    if spec["kind"] == "ppo":
        _same(row["checkpoint"], spec["checkpoint"], "recording endpoint pins")
    result = replay_recording(recording, spec, checkpoint=checkpoint,
                              gate_path=rebind(freeze["gate"], manifest, root))
    _same(result["physical_trace_sha256"], selected["trace_sha256"], "prescribed trace hash")
    result["bundle_sha256"] = manifest["sha256"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    play = sub.add_parser("replay")
    play.add_argument("--controller", default="storage")
    play.add_argument("--arm", default="shared-dynamic")
    fixture = sub.add_parser("engineering-fixture")
    fixture.add_argument("--recording", type=Path, required=True)
    fixture.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        manifest = verify_bundle()
        result = {"verified": True, "files": len(manifest["files"]), "manifest_sha256": manifest["sha256"]}
    elif args.command == "replay":
        result = replay(args.controller, args.arm)
    else:
        with gzip.open(args.recording, "rt") as stream:
            row = json.load(stream)["summary"]
        spec = {"label": row["controller"], "kind": row["controller_kind"], "baseline": row["controller"]}
        result = replay_recording(args.recording, spec, checkpoint=args.checkpoint, engineering=True)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

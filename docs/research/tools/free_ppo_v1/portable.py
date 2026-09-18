"""Verify a relocated research bundle and replay its prescribed policy trajectory.

Original absolute paths remain provenance in the records. Integrity is checked
against content hashes and bundle-relative names before inference is permitted.
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
RESULTS = ROOT / "docs/research/results/free_ppo_20260917"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle():
    path = ROOT / "bundle-manifest.json"
    manifest = json.loads(path.read_text())
    expected = manifest["sha256"]
    payload = {k: v for k, v in manifest.items() if k != "sha256"}
    if hashlib.sha256(canonical(payload)).hexdigest() != expected:
        raise ValueError("Bundle manifest checksum differs")
    seen = set()
    for row in manifest["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or row["path"] in seen:
            raise ValueError("Invalid or duplicate bundle-relative path")
        seen.add(row["path"])
        target = ROOT / relative
        if target.is_symlink() or not target.resolve().is_relative_to(ROOT):
            raise ValueError("Bundle member escaped its root")
        if target.stat().st_size != row["bytes"] or file_sha(target) != row["sha256"]:
            raise ValueError(f"Bundle content changed: {relative}")
    return manifest


def replay(optimizer_seed=101):
    import numpy as np
    from docs.research.tools.free_ppo_v1 import environment, train

    manifest = verify_bundle()
    prescribed = manifest["prescribed_replays"][str(optimizer_seed)]
    with gzip.open(ROOT / prescribed["recording"], "rt") as stream:
        recorded = json.load(stream)
    row = recorded["summary"]
    if not row["execution_complete"] or not recorded["minute_truth_states"]:
        raise ValueError("Prescribed replay must be a complete retained physical trace")
    gate = ROOT / manifest["held_out_gate"]
    train.configure_torch()
    model = train.load_checkpoint(ROOT / prescribed["checkpoint"])
    if model.research_identity != environment.environment_identity():
        raise ValueError("Checkpoint and physical environment differ")
    env = environment.DirectPlantEnv(split="held_out", gate_path=gate, record_minutes=True)
    reward_sum = 0.0
    states = []
    actions_exact = rewards_exact = evidence_exact = True
    inference_digest = hashlib.sha256()
    terminated = truncated = False
    try:
        observation, _ = env.reset(seed=row["seed"])
        states.append(env.simulator.state.to_dict())
        for decision in recorded["decision_trace"]:
            details = model.deterministic_details(observation.copy())
            action, _ = model.predict(observation, deterministic=True)
            evidence = {
                "observation_sha256": hashlib.sha256(canonical(observation.tolist())).hexdigest(),
                "normalized_observation_sha256": hashlib.sha256(canonical(np.asarray(details["normalized_input"]).tolist())).hexdigest(),
                "raw_mean": np.asarray(details["unclipped_mean"]).tolist(),
                "action": np.asarray(action).tolist(),
                "observation": observation.tolist(),
                "normalized_observation": np.asarray(details["normalized_input"]).tolist(),
            }
            evidence_exact &= evidence == decision["evidence"]
            evidence_exact &= np.array_equal(observation, env.current_policy_observation)
            evidence_exact &= np.array_equal(action, details["clipped_action"])
            inference_digest.update(canonical(evidence))
            actions_exact &= np.array_equal(action, np.asarray(decision["action"], dtype=action.dtype))
            observation, reward, terminated, truncated, _ = env.step(action)
            rewards_exact &= reward == decision["reward_eur"]
            reward_sum += reward
            states.extend(state.to_dict() for state in env.last_minute_truth_states[1:])
        digest = hashlib.sha256(canonical(states)).hexdigest()
        physical_exact = digest == row["retained_minute_truth_sha256"] == prescribed["trace_sha256"]
        inference_exact = bool(evidence_exact and inference_digest.hexdigest() == row["inference"]["witness_sha256"])
        complete = bool(terminated and not truncated and env.decisions_taken == 672)
        if not (actions_exact and rewards_exact and physical_exact and inference_exact and complete):
            raise ValueError(f"Replay diverged: actions={actions_exact}, rewards={rewards_exact}, physical={physical_exact}, inference={inference_exact}, complete={complete}")
        return {"schema": "free-ppo-v1-portable-replay", "optimizer_seed": optimizer_seed,
                "scenario_seed": row["seed"], "decisions": env.decisions_taken,
                "physical_states": len(states), "actions_exact": bool(actions_exact),
                "interval_rewards_exact": bool(rewards_exact), "physical_states_exact": physical_exact,
                "inference_witness_exact": inference_exact, "inference_witness_sha256": inference_digest.hexdigest(),
                "physical_trace_sha256": digest, "economic_reward_eur": reward_sum,
                "bundle_sha256": manifest["sha256"]}
    finally:
        env.close()
        model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "replay"))
    parser.add_argument("--optimizer-seed", type=int, default=101)
    args = parser.parse_args()
    result = verify_bundle() if args.command == "verify" else replay(args.optimizer_seed)
    if args.command == "verify":
        result = {"verified": True, "files": len(result["files"]), "manifest_sha256": result["sha256"]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

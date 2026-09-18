"""Prove three saved initial PPO policies share each exact coverage trajectory.

This verifier does not train or select weights. At each boundary it evaluates
all three saved initial actors before advancing one common physical simulation.
Held-out access requires the joint endpoint and comparison freezes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from docs.research.tools.coverage_ppo_v1 import (
    CoveragePPOController,
    OPTIMIZER_SEEDS,
    file_record,
    load_initial,
    make_coverage_prior,
    require_frozen_endpoints,
)
from docs.research.tools.evaluate_feasible25_v1 import (
    TRAINING_ROOT,
    require_evaluation_freeze,
)
from docs.research.tools.feasible_benchmark_v1 import (
    BenchmarkEnv,
    load_benchmark_contract,
)
from docs.research.tools.feasible_mission_replay import canonical_sha, run_controller
from plant_backend.rl.v2.training.custody_io import (
    mkdir_exclusive,
    publish_bytes,
    publish_json,
    read_regular,
)
from plant_backend.rl.v2.training.custody_json import canonical_bytes

SCHEMA = "saved-initial-coverage-equivalence-v1"
COMPARISON_FIELDS = (
    "minute_truth_sha256",
    "physical_metrics",
    "economic_metrics",
    "cycle_settlements",
    "delivery",
    "minute_checks",
    "execution_complete",
    "hard_safety_passed",
    "every_delivery_passed",
    "all_daily_returns_passed",
    "mission_passed",
    "decisions",
    "physical_steps",
    "terminated",
    "truncated",
)


def physical_command(command):
    """Compare every field that controls execution; provenance is separate."""
    value = command.to_dict() if hasattr(command, "to_dict") else command
    return {
        key: value[key]
        for key in (
            "schema",
            "decision_index",
            "observation_sha256",
            "hold_duration_s",
            "targets",
            "envelope",
        )
    }


def update_digest(digest, value):
    data = canonical_bytes(value)
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)


def unclipped_actor_action(model, vector):
    """Read the deterministic actor output before SB3's Box clipping."""
    tensor, _ = model.policy.obs_to_tensor(vector.copy())
    with torch.no_grad():
        action = model.policy._predict(tensor, deterministic=True)
    value = action.detach().cpu().numpy().reshape(4)
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite initial actor action")
    return value


class SharedInitialController:
    id = "daily-coverage-offgrid-v1"

    def __init__(self, env, models, reference):
        self.env = env
        self.coverage = make_coverage_prior()
        self.adapters = {
            seed: CoveragePPOController(env, model) for seed, model in models.items()
        }
        self.reference = reference
        self.records = []
        self.vector = None
        self.input_hash = hashlib.sha256(b"actual-public-135-stream-v1\0")
        self.policy_hashes = {
            seed: hashlib.sha256(b"saved-initial-139-action-command-stream-v1\0")
            for seed in models
        }

    def reset(self):
        if self.records:
            raise ValueError("Initial witness controller is single-use")
        self.coverage.reset()
        for adapter in self.adapters.values():
            adapter.reset()

    def set_observation(self, vector):
        if (
            not isinstance(vector, np.ndarray)
            or vector.dtype != np.float32
            or vector.shape != (135,)
            or not np.isfinite(vector).all()
        ):
            raise ValueError("Expected actual finite float32[135]")
        self.vector = vector.copy()
        for adapter in self.adapters.values():
            adapter.set_observation(vector.copy())

    def compute_command(self, observation, **kwargs):
        if self.vector is None:
            raise ValueError("Current returned observation was not provided")
        index = self.env.decisions_taken
        nominal = self.coverage.compute_command(observation, **kwargs)
        fields = physical_command(nominal.command)
        saved = self.reference["commands"][index]
        if (
            fields != physical_command(saved["command"])
            or nominal.invalid != saved["controller_invalid"]
        ):
            raise ValueError(f"Coverage physical command differs at decision {index}")
        public_input = {"decision_index": index, "actual135": self.vector.tolist()}
        update_digest(self.input_hash, public_input)
        record = {**public_input, "physical_command": fields, "policies": []}
        for seed, adapter in self.adapters.items():
            decision = adapter.compute_command(observation, **kwargs)
            evidence = decision.evidence
            vector = np.asarray(evidence["policy_observation"], dtype=np.float32)
            if vector.shape != (139,) or not np.array_equal(vector[:135], self.vector):
                raise ValueError("Policy did not receive the actual current vector")
            raw = np.asarray(evidence["raw_residual"])
            bounded = np.asarray(evidence["bounded_residual"])
            unclipped = unclipped_actor_action(adapter.model, vector)
            if not all(
                action.shape == (4,) and np.all(action == 0.0)
                for action in (raw, bounded, unclipped)
            ):
                raise ValueError(f"Initial actor {seed} was not exactly zero")
            if (
                physical_command(decision.command) != fields
                or decision.invalid != nominal.invalid
            ):
                raise ValueError(f"Initial actor {seed} changed physical execution")
            item = {
                "optimizer_seed": seed,
                "decision_index": index,
                "actual139": vector.tolist(),
                "adapter_input_action": raw.tolist(),
                "bounded_action": bounded.tolist(),
                "unclipped_deterministic_actor_action": unclipped.tolist(),
                "physical_command_sha256": canonical_sha(fields),
                "initial_command_sha256": decision.command.sha256,
                "coverage_command_sha256": nominal.command.sha256,
                "controller_invalid": decision.invalid,
            }
            update_digest(self.policy_hashes[seed], item)
            record["policies"].append(item)
        self.records.append(record)
        self.vector = None
        return nominal


def verify(*, split, seeds, coverage_dir, output, training_root=TRAINING_ROOT):
    contract = load_benchmark_contract()
    if split not in {"development", "held_out"} or any(
        seed not in contract["splits"][split] for seed in seeds
    ):
        raise ValueError("Scenario outside declared split")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Expected distinct nonempty scenario list")
    training_root, coverage_dir, output = (
        Path(path).resolve() for path in (training_root, coverage_dir, output)
    )
    endpoints = require_frozen_endpoints(training_root)
    freeze = require_evaluation_freeze() if split == "held_out" else None
    if freeze is not None and freeze["endpoints"] != endpoints:
        raise ValueError("Initial policy endpoint set differs from evaluation freeze")
    sources = [file_record(Path(__file__).resolve())]
    models = {seed: load_initial(training_root, seed) for seed in OPTIMIZER_SEEDS}
    initial_pins = [
        {
            "optimizer_seed": seed,
            "initial_checkpoint": file_record(
                training_root / f"seed-{seed}/initial-policy.pt"
            ),
            "training_result": file_record(training_root / f"seed-{seed}/result.json"),
            "policy_identity": model.research_identity,
        }
        for seed, model in models.items()
    ]
    mkdir_exclusive(output)
    report = {
        "schema": SCHEMA,
        "split": split,
        "benchmark_contract_sha256": contract["contract_sha256"],
        "endpoints_sha256": canonical_sha(endpoints),
        "evaluation_freeze_sha256": freeze["sha256"] if freeze else None,
        "initial_policies": initial_pins,
        "source_files": sources,
        "provenance_policy": "Issuer, source evidence and command hashes may differ; all physical command fields and controller-invalid flags must match exactly.",
        "claim": "One common physical replay validates all three saved initial policies after exact command equivalence at every decision.",
        "runs": [],
    }
    publish_json(output / "plan.json", {**report, "requested_seeds": seeds})
    for seed in seeds:
        started = time.monotonic()
        reference_path = coverage_dir / f"{seed}.json.gz"
        reference_pin = file_record(reference_path)
        reference = json.loads(gzip.decompress(read_regular(reference_path)))
        if (
            reference["seed"] != seed
            or reference["split"] != split
            or reference["controller"] != SharedInitialController.id
            or reference["benchmark_contract_sha256"] != contract["contract_sha256"]
            or not reference["execution_complete"]
            or len(reference["commands"]) != 672
            or reference["minute_truth_sha256"]
            != canonical_sha(reference["minute_truth_states"])
        ):
            raise ValueError("Coverage reference identity or full minute trace differs")
        if (
            freeze is not None
            and reference.get("evaluation_freeze_sha256") != freeze["sha256"]
        ):
            raise ValueError("Held-out coverage reference belongs to another freeze")
        env = BenchmarkEnv(split=split)
        try:
            env.reset(seed=seed)
            shared = SharedInitialController(env, models, reference)
            replay = run_controller(env, shared, retain_minutes=False)
            if not replay["execution_complete"] or len(shared.records) != 672:
                raise ValueError(f"Shared initial replay failed: {replay['failure']}")
            mismatches = [
                field
                for field in COMPARISON_FIELDS
                if replay[field] != reference[field]
            ]
            if mismatches:
                raise ValueError(f"Shared physical replay differs: {mismatches}")
            if file_record(reference_path) != reference_pin:
                raise ValueError("Coverage reference changed during verification")
            row = {
                "seed": seed,
                "reference": reference_pin,
                "verified": True,
                "physical_decisions": len(shared.records),
                "saved_initial_decisions": len(shared.records) * len(models),
                "all_initial_actions_exactly_zero": True,
                "all_physical_commands_exactly_equal": True,
                "actual135_stream_sha256": shared.input_hash.hexdigest(),
                "policy_streams": [
                    {
                        "optimizer_seed": optimizer_seed,
                        "decisions": len(shared.records),
                        "actual139_action_command_stream_sha256": digest.hexdigest(),
                    }
                    for optimizer_seed, digest in shared.policy_hashes.items()
                ],
                "exact_comparison_fields": list(COMPARISON_FIELDS),
                "minute_truth_sha256": replay["minute_truth_sha256"],
                "mission_passed": replay["mission_passed"],
                "seconds": time.monotonic() - started,
            }
            witness_path = output / f"{seed}-initial-witness.json.gz"
            publish_bytes(
                witness_path,
                gzip.compress(
                    canonical_bytes({**row, "decisions": shared.records}), mtime=0
                ),
            )
            row["decision_witness"] = file_record(witness_path)
            publish_json(output / f"{seed}-summary.json", row)
            report["runs"].append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
        finally:
            env.close()
    if sources != [file_record(Path(__file__).resolve())]:
        raise ValueError("Verifier source changed during evaluation")
    if endpoints != require_frozen_endpoints(training_root):
        raise ValueError("Training endpoint freeze changed")
    if split == "held_out":
        require_evaluation_freeze()
    publish_json(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("development", "held_out"))
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--coverage-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    verify(**vars(args))


if __name__ == "__main__":
    main()

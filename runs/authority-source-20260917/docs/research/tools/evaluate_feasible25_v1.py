"""Evaluate frozen coverage/PPO policies with the common exact-minute auditor.

Held-out access requires all three PPO endpoints and the full comparison source
set to be frozen first. The original beta/historical recordings are untouched.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import time

from docs.research.tools.feasible_benchmark_v1 import (
    BenchmarkEnv,
    ROOT,
    load_benchmark_contract,
)
from docs.research.tools.feasible_mission_replay import (
    canonical_sha,
    file_pin,
    run_controller,
)
from docs.research.tools.offgrid_coverage_v1 import OffgridCoverageController

BASE = ROOT / "docs/research/results/feasible25_offgrid_20260915"
TRAINING_ROOT = BASE / "coverage-ppo-v1"
FREEZE_PATH = BASE / "evaluation-freeze.json"


def instrumented_ppo_controller(env, model):
    """Retain SB3's deterministic output before its normal Box clipping."""
    from dataclasses import replace
    import numpy as np
    import torch
    from docs.research.tools.coverage_ppo_v1 import (
        CoveragePPOController,
        interface_identity,
    )

    # SB3's JSON checkpoint metadata changes tuple containers into lists. Admit
    # only byte-equivalent canonical JSON, then restore the in-memory shape
    # expected by the frozen controller's strict Python equality check.
    expected_identity = interface_identity(env)
    canonical = lambda value: json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if canonical(model.research_identity) != canonical(expected_identity):
        raise ValueError("Loaded policy has a different canonical task identity")
    serialized_identity_sha256 = canonical_sha(model.research_identity)
    model.research_identity = expected_identity

    class InstrumentedController(CoveragePPOController):
        def compute_command(self, observation, **kwargs):
            decision = super().compute_command(observation, **kwargs)
            vector = np.asarray(
                decision.evidence["policy_observation"], dtype=np.float32
            )
            tensor, _ = self.model.policy.obs_to_tensor(vector)
            if self.model.policy.squash_output:
                raise ValueError(
                    "Unexpected squashed policy changes action interpretation"
                )
            with torch.no_grad():
                mean = (
                    self.model.policy.get_distribution(tensor)
                    .get_actions(deterministic=True)
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
            expected = np.clip(
                mean, self.model.action_space.low, self.model.action_space.high
            )
            actual = np.asarray(decision.evidence["raw_residual"], dtype=np.float32)
            if not np.array_equal(expected, actual):
                raise ValueError(
                    "Recorded adapter input differs from deterministic policy inference"
                )
            return replace(
                decision,
                evidence={
                    **decision.evidence,
                    "inference_witness": {
                        "unclipped_deterministic_action": mean.tolist(),
                        "sb3_clipped_action": expected.tolist(),
                        "upstream_action_clipped": bool(np.any(mean != expected)),
                        "adapter_input_verified": True,
                        "canonical_task_identity_sha256": serialized_identity_sha256,
                        "checkpoint_metadata_container_normalization": "Canonical JSON equality verified before restoring tuple/list containers; weights unchanged.",
                    },
                },
            )

    return InstrumentedController(env, model)


def comparison_sources():
    return [
        file_pin(ROOT / path)
        for path in (
            "docs/research/tools/feasible_benchmark_v1.py",
            "docs/research/tools/feasible_mission_controller.py",
            "docs/research/tools/feasible_mission_replay.py",
            "docs/research/tools/offgrid_coverage_v1.py",
            "docs/research/tools/coverage_ppo_v1.py",
            "docs/research/tools/verify_initial_coverage_v1.py",
            "docs/research/tools/evaluate_feasible25_v1.py",
            "docs/research/tools/run_mission_mpc_v1.py",
            "docs/research/tools/evaluate_mission_mpc_heldout_v1.py",
            "docs/research/tools/mission_mpc_recovery_v1.py",
            "docs/research/tools/run_mission_mpc_recovery_v1.py",
            "backend/src/plant_backend/mission_mpc.py",
        )
    ]


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def freeze_evaluation():
    from docs.research.tools.coverage_ppo_v1 import require_frozen_endpoints

    endpoints = require_frozen_endpoints(TRAINING_ROOT)
    payload = {
        "schema": "feasible25-comparison-freeze-v1",
        "benchmark_contract_sha256": load_benchmark_contract()["contract_sha256"],
        "created_at_unix_s": time.time(),
        "sources": comparison_sources(),
        "endpoints": endpoints,
        "held_out_seeds": load_benchmark_contract()["splits"]["held_out"],
        "selection": "All three fixed PPO endpoints, selected coverage prior, frozen hard-constraint MPC. No tuning after opening held-out scenarios.",
    }
    payload["sha256"] = canonical_sha(payload)
    publish(FREEZE_PATH, payload)
    return payload


def require_evaluation_freeze():
    from docs.research.tools.coverage_ppo_v1 import require_frozen_endpoints

    payload = json.loads(FREEZE_PATH.read_text())
    if payload["sha256"] != canonical_sha(
        {k: v for k, v in payload.items() if k != "sha256"}
    ):
        raise ValueError("Evaluation freeze checksum differs")
    if payload["sources"] != comparison_sources():
        raise ValueError("Comparison source changed after held-out freeze")
    if payload["endpoints"] != require_frozen_endpoints(TRAINING_ROOT):
        raise ValueError("Comparison endpoints changed after held-out freeze")
    return payload


def evaluate(*, kind, seeds, split, output, optimizer_seed=None):
    contract = load_benchmark_contract()
    if split not in {"development", "held_out"} or any(
        s not in contract["splits"][split] for s in seeds
    ):
        raise ValueError("Evaluation seeds are outside selected split")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate evaluation seeds")
    freeze = require_evaluation_freeze() if split == "held_out" else None
    source_pins = comparison_sources()
    model = None
    if kind == "ppo":
        import torch
        from docs.research.tools.coverage_ppo_v1 import load_final

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        model = load_final(TRAINING_ROOT, optimizer_seed)
        label = f"coverage-ppo-{optimizer_seed}"
    elif kind == "coverage":
        if optimizer_seed is not None:
            raise ValueError("Coverage has no optimizer seed")
        label = "daily-coverage-offgrid-v1"
    else:
        raise ValueError("Unknown evaluation controller")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "feasible25-evaluation-panel-v1",
        "controller": label,
        "split": split,
        "requested_seeds": seeds,
        "benchmark_contract_sha256": contract["contract_sha256"],
        "evaluation_freeze_sha256": freeze["sha256"] if freeze else None,
        "source_files": source_pins,
        "started_at_unix_s": time.time(),
        "runs": [],
    }
    for seed in seeds:
        started = time.monotonic()
        env = BenchmarkEnv(split=split)
        try:
            env.reset(seed=seed)
            if kind == "ppo":
                controller = instrumented_ppo_controller(env, model)
            else:
                controller = OffgridCoverageController()
            result = run_controller(env, controller, controller_id=label)
            result.update(
                schema="feasible25-evaluation-recording-v1",
                split=split,
                benchmark_contract_sha256=contract["contract_sha256"],
                evaluation_freeze_sha256=report["evaluation_freeze_sha256"],
                controller_configuration=controller.to_dict(),
                electricity_model=env.electricity_model.to_dict(),
                additional_source_files=source_pins,
                change="New25kg/h off-grid task; original physical draw and all mandatory service, daily inventory and safety requirements retained.",
                wall_time_s=time.monotonic() - started,
            )
            if source_pins != comparison_sources():
                raise ValueError("Comparison source changed during evaluation")
            path = output / f"{seed}.json.gz"
            with path.open("xb") as stream:
                stream.write(
                    gzip.compress(
                        json.dumps(
                            result, separators=(",", ":"), allow_nan=False
                        ).encode(),
                        mtime=0,
                    )
                )
            row = {
                k: v
                for k, v in result.items()
                if k
                not in {
                    "commands",
                    "decision_frames",
                    "minute_truth_states",
                    "manifest",
                }
            }
            row["recording"] = file_pin(path)
            report["runs"].append(row)
            # Only this process owns its output directory; summary is a progress view.
            (output / "summary.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    {
                        "controller": label,
                        "seed": seed,
                        "complete": row["execution_complete"],
                        "mission": row["mission_passed"],
                        "safe": row["hard_safety_passed"],
                        "daily_returns": sum(
                            x["feasible"] for x in row["cycle_settlements"]
                        ),
                        "unmet_kg": row["physical_metrics"]["unmet_kg"],
                        "seconds": row["wall_time_s"],
                    }
                ),
                flush=True,
            )
        finally:
            env.close()
    if split == "held_out":
        require_evaluation_freeze()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--controller", choices=("coverage", "ppo"))
    parser.add_argument("--optimizer-seed", type=int, choices=(101, 211, 307))
    parser.add_argument(
        "--split", choices=("development", "held_out"), default="development"
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.freeze:
        print(json.dumps(freeze_evaluation(), indent=2))
        return
    if (
        args.controller is None
        or args.output is None
        or (args.controller == "ppo" and args.optimizer_seed is None)
    ):
        parser.error(
            "Evaluation requires controller/output and an optimizer seed for PPO"
        )
    seeds = args.seeds or load_benchmark_contract()["splits"][args.split]
    evaluate(
        kind=args.controller,
        seeds=seeds,
        split=args.split,
        output=args.output,
        optimizer_seed=args.optimizer_seed,
    )


if __name__ == "__main__":
    main()

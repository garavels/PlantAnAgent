"""Publish closed PPO custody; optionally record the prospectively fixed policy.

Training export resets no plant. Rehearsal is a separate, explicitly authorized,
single fixture attempt; neither command performs training or study selection.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import time
from typing import TYPE_CHECKING, Any

from . import demo_recordings, ppo_evidence
from .record_demo import RehearsalTimeout, WALL_LIMIT_SECONDS
from .rl.v2.training import custody_io as io
from .rl.v2.training.config import BETA_PI_LAGRANGIAN_ALGORITHM_ID
from .v2_demo import FIXTURE_SEED, StepRequest, V2DemoSession

if TYPE_CHECKING:
    from .rl.policies import LoadedPolicy
    from .rl.v2.study.training_evidence import TrainingAdmission


def _projection(admission: TrainingAdmission) -> dict[str, Any]:
    """Pure presentation projection, not a substitute for admission."""
    slots = []
    for slot, cell in zip(admission.slots, admission.report["cells"], strict=True):
        rows = []
        transitions = 0
        for row in slot.rollouts or ():
            transitions += row["transition_count"]
            rows.append(
                {
                    "index": row["rollout_index"],
                    "transitions": transitions,
                    "objective_sum": row["reward_return"],
                    "completed_episodes": row["completed_episode_count"],
                    "complete_cycles": row["complete_cycle_count"],
                    "raw_cost": list(row["raw_cost"]),
                    "training_cost": list(row["training_cost"]),
                    "mean_episode_cost": row["mean_completed_episode_cost"],
                    "multipliers": list(row["multipliers"]),
                    "cost_value_losses": list(row["cost_value_losses"]),
                }
            )
        checkpoint = cell["checkpoint"]
        slots.append(
            {
                "algorithm": slot.algorithm,
                "optimizer_seed": slot.optimizer_seed,
                "training_completed": slot.training_completed,
                "checkpoint_admitted": slot.checkpoint_admitted,
                "failure": slot.failure,
                "checkpoint_evidence_sha256": (
                    checkpoint["checkpoint_evidence_sha256"] if checkpoint else None
                ),
                "rollouts": rows,
            }
        )
    payload = {
        "schema_version": ppo_evidence.SCHEMA,
        "source_commit": admission.identity.source["authorization_commit"],
        "training_report_sha256": admission.report["sha256"],
        "runtime_provenance_sha256": admission.identity.runtime["sha256"],
        "campaign_status": admission.report["campaign_status"],
        "custody_admitted_at_export": True,
        "evaluation_performed": False,
        "slots": slots,
    }
    return ppo_evidence.validate({**payload, "sha256": ppo_evidence.digest(payload)})


def export_training(report: Path) -> dict[str, Any]:
    from .rl.v2.study.training_evidence import admit_training

    admission = admit_training(report)
    result = _projection(admission)
    admission.recheck()
    raw = json.dumps(result, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > ppo_evidence.MAX_BYTES:
        raise ValueError("training export exceeds its public byte cap")
    ppo_evidence.PUBLIC_ROOT.mkdir(parents=True, exist_ok=True)
    io.publish_bytes(ppo_evidence.PUBLIC_ROOT / "training.json", raw)
    return result


def _presentation_policy(
    admission: TrainingAdmission,
) -> tuple[LoadedPolicy, dict[str, Any]]:
    from .rl.policies import load_policy
    from .rl.v2.beta_env import PlantCyclicMarketEnvV2Beta
    from .rl.v2.policy_setpoint_controller import policy_contract_for_environment
    from .rl.v2.training.runtime_provenance import BetaTrainingRuntimeProvenanceV1

    slot = admission.slots[0]
    if (slot.algorithm, slot.optimizer_seed) != (BETA_PI_LAGRANGIAN_ALGORITHM_ID, 101):
        raise ValueError("presentation policy identity differs")
    if not slot.checkpoint_admitted:
        raise ValueError(
            "fixed presentation checkpoint is unavailable; no replacement allowed"
        )
    checkpoint = admission.report["cells"][0]["checkpoint"]
    environment = PlantCyclicMarketEnvV2Beta()
    try:
        loaded = load_policy(
            checkpoint["path"],
            contract=policy_contract_for_environment(environment),
            environment=environment,
            expected_beta_runtime_provenance=BetaTrainingRuntimeProvenanceV1.from_dict(
                admission.identity.runtime
            ),
            require_registered_evaluation_eligibility=True,
        )
    finally:
        environment.close()
    admission.recheck()
    return loaded, {
        "algorithm": slot.algorithm,
        "optimizer_seed": slot.optimizer_seed,
        "checkpoint_evidence_sha256": checkpoint["checkpoint_evidence_sha256"],
        "metadata_sha256": checkpoint["metadata_sha256"],
        "runtime_provenance_sha256": checkpoint["runtime_provenance_sha256"],
        "training_report_sha256": admission.report["sha256"],
        "selection": "prospective-pi-seed-101; descriptive fixture, not a winner",
    }


def record_rehearsal(report: Path) -> dict[str, Any]:
    from .rl.v2.study.training_evidence import admit_training

    admission = admit_training(report)
    loaded, identity = _presentation_policy(admission)
    root = demo_recordings.RECORDING_ROOT
    root.mkdir(parents=True, exist_ok=True)
    marker = {
        "schema": "plant-ppo-presentation-attempt-v1",
        "fixture_seed": FIXTURE_SEED,
        "horizon_hours": 168,
        "wall_limit_seconds": WALL_LIMIT_SECONDS,
        "policy": identity,
        "source_commit": admission.identity.source["authorization_commit"],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "claimed",
    }
    # Consume the attempt before any reset. Never replace it, including on failure.
    io.publish_json(root / "PPO.attempt.json", marker)
    started = time.monotonic()
    demo = V2DemoSession()

    def timeout(*_: object) -> None:
        raise RehearsalTimeout("fixed PPO rehearsal wall limit reached")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(WALL_LIMIT_SECONDS)
    outcome: dict[str, Any] = {**marker, "status": "failed", "error": None}
    record: dict[str, Any] | None = None
    try:
        try:
            snapshot = demo.reset_ppo(loaded, identity)
            while snapshot["status"] == "ready":
                snapshot = demo.step(
                    StepRequest(
                        expected_run_id=snapshot["run_id"],
                        expected_revision=snapshot["revision"],
                        decisions=4,
                    )
                )
        except RehearsalTimeout:
            snapshot = demo.recording()["snapshot"]
            snapshot.update(
                status="failed",
                error="PPO rehearsal stopped (wall_limit). Last completed boundary retained; a partial interval may have executed.",
            )
        finally:
            signal.alarm(0)
        record = demo_recordings.validate(
            demo_recordings.seal_recording(snapshot, demo.recording()["frames"])
        )
        admission.recheck()
        io.publish_bytes(
            root / f"{snapshot['run_id']}.json",
            json.dumps(record, separators=(",", ":"), allow_nan=False).encode(),
        )
        outcome.update(
            demo_recordings.summarize(record, "saved"), error=snapshot["error"]
        )
    except Exception as exc:
        # A failed post-run custody recheck cannot publish a trustworthy replay.
        outcome.update(status="failed", error=type(exc).__name__)
        if record is not None:
            io.publish_bytes(
                root / "PPO.rejected-recording.json",
                json.dumps(record, separators=(",", ":"), allow_nan=False).encode(),
            )
            outcome["rejected_recording_sha256"] = record["sha256"]
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        if demo._environment is not None:
            demo._environment.close()
        outcome["wall_seconds"] = time.monotonic() - started
        io.publish_json(root / "PPO.outcome.json", outcome)
    return outcome


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("training", "development", "heldout", "rehearsal")
    )
    parser.add_argument(
        "report", type=Path, help="Closed production training report.json"
    )
    parser.add_argument(
        "--assert-authorized",
        action="store_true",
        help="Required for the separate fixed rehearsal",
    )
    args = parser.parse_args(argv)
    if args.command == "rehearsal" and not args.assert_authorized:
        parser.error(
            "rehearsal requires explicit operator authorization and --assert-authorized"
        )
    # Fresh offline process retains the original training admission host contract.
    import torch

    torch.set_num_threads(8)
    torch.set_num_interop_threads(12)
    if args.command == "training":
        result = export_training(args.report)
    elif args.command == "rehearsal":
        result = record_rehearsal(args.report)
    else:
        from .ppo_evaluation import export_stage

        if args.report.name != "report.json":
            parser.error("stage export takes the original training report.json path")
        result = export_stage(args.report.parent, args.command)
    print(json.dumps({k: v for k, v in result.items() if k != "slots"}, indent=2))


if __name__ == "__main__":
    main()

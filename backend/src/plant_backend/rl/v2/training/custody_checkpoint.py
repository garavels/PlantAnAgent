"""Registered production admission without resetting any environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ... import checkpoints
from ...policies import load_policy
from ..policy_setpoint_controller import policy_contract_for_environment
from .beta_checkpoint import parse_beta_checkpoint_evidence
from .config import BETA_REGISTERED_RUN_KIND, ControllerNeutralTrainingConfigV1
from .custody_io import file_record, inventory, read_regular
from .custody_json import canonical_bytes, decode
from .ppo import BETA_CHECKPOINT_EXTRA_KEY, make_env
from .registered_contract import require_task_template
from .runtime_provenance import BetaTrainingRuntimeProvenanceV1


class CheckpointAdmissionFailure(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def admit_checkpoint(
    config: ControllerNeutralTrainingConfigV1,
    runtime: BetaTrainingRuntimeProvenanceV1,
) -> dict[str, Any]:
    """Reopen exact artifacts with the non-injected registered policy loader."""
    config.validate()
    require_task_template(config)
    if config.beta_rl is None or config.beta_rl.run_kind != BETA_REGISTERED_RUN_KIND:
        raise ValueError("checkpoint admission requires registered configuration")
    root = Path(config.output_dir)
    checkpoint = root / "checkpoint"
    before = inventory(root)
    paths = {item["path"]: item["node_type"] for item in before["entries"]}
    if "checkpoint" not in paths:
        raise CheckpointAdmissionFailure("checkpoint_missing")
    expected_paths = {
        "checkpoint": "directory",
        "checkpoint/metadata.json": "regular_file",
        "checkpoint/policy.zip": "regular_file",
        "checkpoint/training_config.json": "regular_file",
    }
    if paths != expected_paths or before["forbidden_count"]:
        raise CheckpointAdmissionFailure("artifact_inventory_invalid")
    try:
        config_raw = read_regular(checkpoint / "training_config.json")
        decode(config_raw)
        decode(read_regular(checkpoint / "metadata.json"))
        metadata = checkpoints.load_metadata(checkpoint, check=False)
        if set(metadata.extra) != {BETA_CHECKPOINT_EXTRA_KEY}:
            raise ValueError("checkpoint evidence fields differ")
        evidence = parse_beta_checkpoint_evidence(
            metadata.extra[BETA_CHECKPOINT_EXTRA_KEY]
        )
    except (ValueError, TypeError, KeyError, checkpoints.CheckpointError) as reason:
        raise CheckpointAdmissionFailure("checkpoint_malformed") from reason
    try:
        expected_config = canonical_bytes(config.to_dict())
        if (
            config_raw != expected_config
            or canonical_bytes(metadata.training_config) != expected_config
        ):
            raise ValueError("checkpoint configuration differs from its cell")
        if (
            metadata.algorithm != config.beta_rl.algorithm
            or metadata.seed != config.seed
            or metadata.total_timesteps != 1_048_576
            or evidence.runtime_provenance != runtime
            or evidence.run_kind != BETA_REGISTERED_RUN_KIND
            or evidence.optimizer_seed != config.seed
            or evidence.algorithm_id != config.beta_rl.algorithm
            or evidence.total_timesteps != 1_048_576
            or evidence.resumed_from_parent
            or evidence.resume_parent is not None
            or not evidence.exact_endpoint_admitted
            or not evidence.eligible_for_registered_evaluation
        ):
            raise ValueError("checkpoint identity differs from registered custody")
        environment = make_env(config)
        try:
            loaded = load_policy(
                checkpoint,
                contract=policy_contract_for_environment(environment),
                environment=environment,
                expected_beta_runtime_provenance=runtime,
                require_registered_evaluation_eligibility=True,
            )
        finally:
            environment.close()
        if loaded.metadata != metadata:
            raise ValueError("checkpoint metadata changed during admission")
        rollouts = evidence.constraint_state["payload"]["rollout_ledgers"]
        allocator = evidence.allocator_snapshot["ledger"]["entries"]
        optimizer_steps = sum(item["optimizer_steps"] for item in rollouts)
        if len(rollouts) != 128 or len(allocator) != 1564 or optimizer_steps != 40960:
            raise ValueError("registered checkpoint endpoint counts differ")
        files = [
            file_record(checkpoint / name)
            for name in ("metadata.json", "policy.zip", "training_config.json")
        ]
        if inventory(root) != before:
            raise ValueError("checkpoint artifacts changed during admission")
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        RuntimeError,
        checkpoints.CheckpointError,
    ) as reason:
        raise CheckpointAdmissionFailure("checkpoint_admission_failed") from reason
    return {
        "path": str(checkpoint),
        "metadata_sha256": files[0]["sha256"],
        "checkpoint_evidence_sha256": evidence.sha256,
        "runtime_provenance_sha256": runtime.sha256,
        "environment_contract_sha256": metadata.environment_contract_digest,
        "checkpoint_contract_sha256": metadata.checkpoint_contract_digest,
        "files": files,
        "logical_bytes": sum(item["logical_bytes"] for item in files),
        "allocated_bytes": sum(item["allocated_bytes"] for item in files),
        "rollout_entry_count": 128,
        "allocator_entry_count": 1564,
        "optimizer_steps": 40960,
        "registered_evaluation_eligible": True,
    }

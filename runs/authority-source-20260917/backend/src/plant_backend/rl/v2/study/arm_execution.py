"""One environment, one reset, one controller; enclosing custody owns permission."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ... import checkpoints
from ...policies import load_policy
from .. import beta_controller_benchmark as benchmark, beta_spaces
from ..mpc_setpoint_controller import ServiceMPCSetpointControllerV1
from ..setpoint_controllers import LegacyHeuristicSetpointControllerV1
from ..policy_setpoint_controller import (
    NormalizedSetpointPolicyControllerV1,
    policy_contract_for_environment,
)
from ..training import custody_io as io
from ..training.config import (
    BETA_REGISTERED_RUN_KIND,
    ControllerNeutralTrainingConfigV1,
)
from ..training.custody_authorization import repository_root
from ..training.custody_json import authenticate, canonical_bytes, decode
from ..training.registered_contract import require_task_template
from ..training.runtime_provenance import BetaTrainingRuntimeProvenanceV1
from . import (
    archive,
    arm_evidence as arms,
    compressed_evidence as compressed,
    latency_evidence as timing,
)
from .contract import PROTOCOL_ID, PROTOCOL_SHA256, analysis_contract
from .latency import LatencyRecorder


@dataclass(frozen=True)
class ArmExecution:
    run: dict[str, Any]
    checkpoint: dict[str, Any] | None
    manifest_identity: dict[str, str] | None
    audit_manifest: dict[str, Any] | None
    source: dict[str, Any]
    host_before: dict[str, Any]
    host_after: dict[str, Any]
    measurement: dict[str, Any]


def _execute_arm(
    config: ControllerNeutralTrainingConfigV1,
    *,
    arm: str,
    seed: int,
    checkpoint: Path | None,
    expected_training_runtime: BetaTrainingRuntimeProvenanceV1 | None = None,
) -> ArmExecution:
    """Private engine: fixture/registered callers must admit scope and custody first."""
    config.validate()
    if (
        arm not in benchmark.BETA_CONTROLLER_ARM_ORDER
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError("study arm/seed identity differs")
    if (arm == "PPO") != (checkpoint is not None):
        raise ValueError("only PPO requires a study checkpoint")
    registered = (
        config.beta_rl is not None
        and config.beta_rl.run_kind == BETA_REGISTERED_RUN_KIND
    )
    if registered:
        require_task_template(config)
        if expected_training_runtime is None:
            raise ValueError("registered arm requires admitted training runtime")
    elif expected_training_runtime is not None:
        raise ValueError("fixture arm cannot carry registered runtime admission")
    source = benchmark._source_provenance(require_clean=registered)
    host_before = timing.validate_latency_host(timing.capture_latency_host())
    recorder = LatencyRecorder(arm)
    settings = arms._settings(config)
    environment = benchmark._new_beta_environment(config)
    identity = None
    manifest = None
    policy_descriptor = None
    try:
        loaded = None
        if checkpoint is not None:
            loaded = load_policy(
                checkpoint,
                contract=policy_contract_for_environment(environment),
                environment=environment,
                expected_beta_runtime_provenance=expected_training_runtime,
                require_registered_evaluation_eligibility=registered,
            )
            if canonical_bytes(loaded.metadata.training_config) != canonical_bytes(
                config.to_dict()
            ):
                raise ValueError(
                    "study checkpoint task differs from the declared actor task"
                )
            if registered and (
                loaded.metadata.format != checkpoints.FORMAT_SB3_PPO
                or not loaded.metadata.trained
            ):
                raise ValueError("registered study requires a trained PPO checkpoint")
            policy_descriptor = benchmark._checkpoint_identity(loaded)
        try:
            if arm == "PPO":
                assert loaded is not None
                loaded = recorder.warm_policy(loaded)
                controller: Any = NormalizedSetpointPolicyControllerV1.from_environment(
                    loaded, environment
                )
            elif arm == "H":
                controller = LegacyHeuristicSetpointControllerV1()
            elif arm == "M24":
                controller = ServiceMPCSetpointControllerV1(settings)
            else:
                controller = benchmark._real_beta_economic_controller(config)
            controller.reset()
            if not isinstance(
                controller.to_dict(), dict
            ) or not controller.to_dict().get("id"):
                raise ValueError("study controller descriptor is malformed")
            if environment.decisions_taken or environment.physics_steps_taken:
                raise ValueError("study controller advanced the plant before reset")
            vector, info = environment.reset(seed=seed)
            if info.get("environment_id") != beta_spaces.ENVIRONMENT_ID:
                raise ValueError("study reset returned a different environment")
            if environment.decisions_taken or environment.physics_steps_taken:
                raise ValueError("study reset advanced physical time")
            identity = benchmark._beta_manifest_identity(environment)
            manifest = environment.audit_manifest()
            if (
                policy_descriptor
                and identity["checkpoint_contract_sha256"]
                != policy_descriptor["checkpoint_contract_sha256"]
            ):
                raise ValueError(
                    "study reset differs from its admitted policy contract"
                )
        except Exception as reason:
            identity = None
            manifest = None
            run = benchmark._preflight_failed_cell(seed, reason)["runs"][arm]
        else:
            if arm == "M24":
                run = benchmark._service_mpc_result(
                    environment=environment,
                    controller=controller,
                    settings=settings,
                    manifest_identity=identity,
                    initial_vector=vector,
                    study_timing=recorder,
                )
            else:
                run = benchmark._generic_arm_result(
                    arm=arm,
                    environment=environment,
                    controller=cast(benchmark._CommandController, controller),
                    manifest_identity=identity,
                    initial_vector=vector,
                    study_timing=recorder,
                )
    finally:
        environment.close()
    host_after = timing.validate_latency_host(timing.capture_latency_host())
    if canonical_bytes(host_after) != canonical_bytes(host_before):
        raise ValueError("study host changed during single-arm execution")
    if canonical_bytes(source) != canonical_bytes(
        benchmark._source_provenance(require_clean=registered)
    ):
        raise ValueError("study source changed during single-arm execution")
    return ArmExecution(
        arms._standalone_result(run),
        policy_descriptor,
        identity,
        manifest,
        source,
        host_before,
        host_after,
        recorder.to_dict(),
    )


def execute_synthetic_arm(
    config: ControllerNeutralTrainingConfigV1,
    *,
    arm: str,
    seed: int,
    directory: Path | str,
    limits: compressed.EvidenceLimits,
    checkpoint: Path | str | None = None,
) -> dict[str, Any]:
    """Fixture-only producer; consume a fresh directory before any plant/reset."""
    arms._fixture_seed(seed)
    config = arms._fixture_config(config.to_dict())
    arms._actor(arm, config)
    if (arm == "PPO") != (checkpoint is not None):
        raise ValueError("only PPO requires a study checkpoint")
    if checkpoint is not None:
        metadata = decode(io.read_regular(checkpoints.metadata_path(checkpoint)))
        declared = arms._fixture_config(metadata.get("training_config"))
        if canonical_bytes(declared.to_dict()) != canonical_bytes(config.to_dict()):
            raise ValueError("fixture checkpoint task differs before execution")
    limits.to_dict()
    root = Path(directory)
    io.require_new_root(root, repository_root(), 0)
    io.mkdir_exclusive(root)
    executed = _execute_arm(
        config,
        arm=arm,
        seed=seed,
        checkpoint=None if checkpoint is None else Path(checkpoint),
    )
    envelope = authenticate(
        {
            "schema": arms.SCHEMA,
            "protocol": {"id": PROTOCOL_ID, "raw_sha256": PROTOCOL_SHA256},
            "analysis_contract_sha256": analysis_contract()["sha256"],
            "stage": "fixture",
            "seed": seed,
            "actor": arms._actor(arm, config),
            "training_task": config.to_dict(),
            "checkpoint": executed.checkpoint,
            "source_provenance": executed.source,
            "service_mpc_settings": arms._settings(config).to_dict(),
            "manifest_identity": executed.manifest_identity,
            "audit_manifest": executed.audit_manifest,
            "run": executed.run,
            "claim_boundary": arms.CLAIM_BOUNDARY,
            "result_claim": None,
        }
    )
    arm_path = root / compressed.ARM_FILENAME
    archive.publish_archive(arm_path, envelope, **limits.arm())
    timing_document = compressed.synthetic_timing_archive(
        arm_path,
        executed.measurement,
        executed.host_before,
        executed.host_after,
        limits=limits,
    )
    archive.publish_archive(
        root / compressed.TIMING_FILENAME, timing_document, **limits.timing()
    )
    return compressed.verify_synthetic_execution(root, limits=limits)

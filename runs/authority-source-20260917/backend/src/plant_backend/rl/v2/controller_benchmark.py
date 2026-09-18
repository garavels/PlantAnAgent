"""Controller-neutral four-arm benchmark on one checkpoint-bound economic task.

The benchmark is intentionally stricter than the historical pairwise evaluator:
every seed is prepared on four independent environments, all immutable manifest
identities are compared before the first physical step, and an incompatible or
unavailable arm prevents every arm in that cell from moving.  Once admitted, arm
failures are retained and never replaced by another controller.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
from importlib import import_module
import json
import math
from numbers import Integral, Real
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from .. import checkpoints
from ..policies import LoadedPolicy, load_policy
from . import economic_spaces
from .economic_env import PlantEconomicSetpointEnvV2
from .mpc_execution import public_observation_sha256
from .mpc_schema import (
    DEFAULT_SERVICE_MPC_SETPOINT_SETTINGS_V1,
    ServiceMPCSetpointSettingsV1,
    domain_sha256,
)
from .mpc_setpoint_controller import ServiceMPCSetpointControllerV1
from .mpc_setpoint_decision import MPCSetpointDecisionEvidenceV1
from .mpc_setpoint_execution import MPCSetpointIntervalExecutionV1
from .mpc_setpoint_run import (
    build_mpc_setpoint_run_evidence,
    verify_mpc_setpoint_run_digests,
)
from .policy_evaluation import load_training_task
from .policy_setpoint_controller import (
    NORMALIZED_POLICY_SETPOINT_CONTROLLER_ID,
    NormalizedSetpointPolicyControllerV1,
    policy_contract_for_environment,
)
from .setpoint_controllers import LegacyHeuristicSetpointControllerV1
from .setpoint_execution import ProcessSetpointIntervalExecutionV1
from .training.config import ControllerNeutralTrainingConfigV1
from .training.ppo import make_env


CONTROLLER_NEUTRAL_BENCHMARK_FORMAT = "controller-neutral-benchmark-v1"
CONTROLLER_NEUTRAL_BENCHMARK_CELL_FORMAT = "controller-neutral-benchmark-cell-v1"
CONTROLLER_NEUTRAL_BENCHMARK_ARM_FORMAT = "controller-neutral-benchmark-arm-v1"
CONTROLLER_NEUTRAL_BENCHMARK_MAX_SEEDS = 32

HEURISTIC_ARM = "H"
SERVICE_MPC_ARM = "M24"
ECONOMIC_MPC_ARM = "E"
PPO_ARM = "PPO"
CONTROLLER_ARM_ORDER = (HEURISTIC_ARM, SERVICE_MPC_ARM, ECONOMIC_MPC_ARM, PPO_ARM)

CONTROLLER_NEUTRAL_BENCHMARK_CLAIM_BOUNDARY = (
    "descriptive controller-neutral development benchmark on checkpoint-bound "
    "simulation episodes; no statistical-superiority, site-performance, savings, "
    "commission, or commercial claim"
)


class _CommandController(Protocol):
    id: str

    def reset(self) -> None: ...

    def compute_command(self, observation: object, **kwargs: object) -> object: ...

    def to_dict(self) -> dict[str, Any]: ...


EconomicControllerFactory = Callable[
    [ControllerNeutralTrainingConfigV1], _CommandController
]


@dataclass(slots=True)
class _PreparedCell:
    seed: int
    manifest_identity: dict[str, str]
    environments: dict[str, PlantEconomicSetpointEnvV2]
    controllers: dict[str, object]
    policy_observations: dict[str, NDArray[np.generic]]

    def close(self) -> None:
        for environment in self.environments.values():
            _close_environment(environment)


def _close_environment(environment: object) -> None:
    """Call Gym's dynamically typed close hook behind one audited boundary."""

    cast(Any, environment).close()


def _validated_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    values: list[int] = []
    for index, seed in enumerate(seeds):
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError(f"seed {index} must be a non-negative integer")
        values.append(int(seed))
    if not values:
        raise ValueError("controller benchmark requires at least one seed")
    if len(values) > CONTROLLER_NEUTRAL_BENCHMARK_MAX_SEEDS:
        raise ValueError(
            "controller benchmark accepts at most "
            f"{CONTROLLER_NEUTRAL_BENCHMARK_MAX_SEEDS} seeds per invocation"
        )
    if len(set(values)) != len(values):
        raise ValueError("controller benchmark seeds must be unique")
    return tuple(values)


def _repository_root() -> Path:
    root = Path(__file__).resolve().parents[5]
    if not (root / "pyproject.toml").is_file() or not (root / "uv.lock").is_file():
        raise RuntimeError("controller benchmark must run from a source checkout")
    return root


def _runtime_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


def _source_provenance(*, require_clean: bool) -> dict[str, Any]:
    """Bind a report to its Git source, lock file, and numerical runtimes."""

    repository = _repository_root()

    def git(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ("git", *arguments),
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
                timeout=10.0,
            )
        except (OSError, subprocess.SubprocessError) as reason:
            raise RuntimeError(
                f"cannot identify benchmark source: {reason}"
            ) from reason
        return completed.stdout.strip()

    revision = git("rev-parse", "--verify", "HEAD")
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise RuntimeError("benchmark source revision is not a full Git object id")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    clean = not status
    if require_clean and not clean:
        raise RuntimeError(
            "a qualified controller benchmark requires a clean committed worktree"
        )
    lock_bytes = (repository / "uv.lock").read_bytes()
    return {
        "git_revision": revision,
        "worktree_clean": clean,
        "worktree_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "dependency_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": _runtime_version("numpy"),
            "scipy": _runtime_version("scipy"),
            "stable_baselines3": _runtime_version("stable-baselines3"),
            "gymnasium": _runtime_version("gymnasium"),
        },
    }


def _manifest_identity(environment: PlantEconomicSetpointEnvV2) -> dict[str, str]:
    manifest = environment.audit_manifest()
    return {
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "physical_exogenous_episode_sha256": str(manifest["exogenous_episode_sha256"]),
        "economic_episode_sha256": str(manifest["economic_episode_sha256"]),
        "checkpoint_contract_sha256": environment.checkpoint_contract_digest,
        "economic_assumptions_sha256": environment.economic_assumptions.sha256,
    }


def _vector_sha256(vector: NDArray[np.generic]) -> str:
    return str(
        domain_sha256(
            "rl-policy-observation-vector-v1",
            [float(value) for value in np.asarray(vector).reshape(-1)],
        )
    )


def _failure(phase: str, reason: BaseException) -> dict[str, Any]:
    message = " ".join(str(reason).split()) or "(no exception message)"
    result: dict[str, Any] = {
        "phase": phase,
        "exception_type": type(reason).__name__,
        "message": message[:500],
    }
    stages = getattr(reason, "stages", None)
    if isinstance(stages, (list, tuple)):
        serialized_stages: list[dict[str, Any]] = []
        for stage in stages:
            serializer = getattr(stage, "to_dict", None)
            if not callable(serializer):
                continue
            try:
                serialized = serializer()
            except Exception:
                continue
            if isinstance(serialized, Mapping):
                serialized_stages.append(dict(serialized))
        if serialized_stages:
            result["solver_stages"] = serialized_stages
    return result


def _timing_free(value: object) -> object:
    """Detach scientific evidence from evaluation and solver wall-clock timing.

    Raw evidence, checkpoint locations, and all timings remain in the report. Exact
    ``sha256`` fields are removed here because some existing evidence schemas hash
    their elapsed fields; content identities are supplied separately to the digest.
    """

    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if (
                key == "sha256"
                or key
                in {
                    "started_at_utc",
                    "completed_at_utc",
                    "decision_latency",
                    "solver_stage_latency",
                    "dispatcher_latency",
                    "event_sha256",
                    "evidence_graph_sha256",
                    "path",
                }
                or key.endswith("_elapsed_ns")
                or key == "elapsed_ns"
            ):
                continue
            result[key] = _timing_free(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_timing_free(item) for item in value]
    return value


def _arm_deterministic_body(run: Mapping[str, Any]) -> dict[str, Any]:
    body = cast(dict[str, Any], _timing_free(dict(run)))
    body.pop("deterministic_evidence_sha256", None)
    return body


def _attach_arm_digest(run: dict[str, Any]) -> dict[str, Any]:
    run["deterministic_evidence_sha256"] = domain_sha256(
        "controller-neutral-benchmark-arm-evidence-v1",
        _arm_deterministic_body(run),
    )
    return run


def _service_settings(
    config: ControllerNeutralTrainingConfigV1,
    supplied: ServiceMPCSetpointSettingsV1 | None,
) -> ServiceMPCSetpointSettingsV1:
    if supplied is None:
        return ServiceMPCSetpointSettingsV1(
            task_horizon_h=config.episode.horizon_h,
            planning_horizon_h=min(24.0, config.episode.horizon_h),
            interval_s=config.episode.decision_interval_s,
            terminal_battery_soc_target=(
                DEFAULT_SERVICE_MPC_SETPOINT_SETTINGS_V1.terminal_battery_soc_target
            ),
        )
    if not isinstance(supplied, ServiceMPCSetpointSettingsV1):
        raise TypeError("service_settings must be ServiceMPCSetpointSettingsV1")
    if not math.isclose(
        supplied.task_horizon_h,
        config.episode.horizon_h,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("service MPC task horizon differs from checkpoint task")
    if not math.isclose(
        supplied.interval_s,
        config.episode.decision_interval_s,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("service MPC cadence differs from checkpoint task")
    if not math.isclose(
        supplied.planning_horizon_h,
        min(24.0, config.episode.horizon_h),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("service MPC arm must use the rolling 24-hour horizon")
    return supplied


def _real_economic_controller(
    config: ControllerNeutralTrainingConfigV1,
) -> _CommandController:
    """Load the production tariff-aware controller without a substitute path."""

    try:
        controller_module = import_module(
            ".economic_mpc_controller", package=__package__
        )
        schema_module = import_module(".economic_mpc_schema", package=__package__)
        controller_type = getattr(
            controller_module, "TariffAwareEconomicMPCControllerV1"
        )
        settings_type = getattr(schema_module, "EconomicMPCSettingsV1")
    except (AttributeError, ImportError) as reason:
        raise RuntimeError(
            "the real tariff-aware economic MPC implementation is unavailable"
        ) from reason
    settings = settings_type(
        task_horizon_h=config.episode.horizon_h,
        planning_horizon_h=min(24.0, config.episode.horizon_h),
        interval_s=config.episode.decision_interval_s,
    )
    controller = controller_type(
        settings=settings,
        assumptions=config.economic_assumptions,
    )
    if not callable(getattr(controller, "compute_command", None)):
        raise TypeError("tariff-aware economic MPC has no compute_command method")
    return cast(_CommandController, controller)


def _assert_economic_task(config: ControllerNeutralTrainingConfigV1) -> None:
    config.validate()
    if config.environment_id != economic_spaces.ENVIRONMENT_ID:
        raise ValueError(
            "the four-arm benchmark requires the checkpoint-bound economic V2 task"
        )


def _checkpoint_identity(loaded: LoadedPolicy) -> dict[str, Any]:
    persisted = checkpoints.load_metadata(loaded.path, check=False)
    if persisted != loaded.metadata:
        raise checkpoints.CorruptCheckpointError(
            f"checkpoint metadata changed after loading ({loaded.path})"
        )
    checked = checkpoints.load_metadata(
        loaded.path,
        contract=checkpoints.PolicyEnvironmentContract(
            spec_version=persisted.spec_version,
            observation_fields=persisted.observation_fields,
            action_fields=persisted.action_fields,
            environment_id=persisted.environment_id,
            environment_contract_digest=persisted.environment_contract_digest,
            checkpoint_contract_digest=persisted.checkpoint_contract_digest,
        ),
    )
    if checked != persisted:
        raise checkpoints.CorruptCheckpointError(
            f"checkpoint changed during benchmark admission ({loaded.path})"
        )
    try:
        metadata_file_sha256 = hashlib.sha256(
            checkpoints.metadata_path(loaded.path).read_bytes()
        ).hexdigest()
    except OSError as reason:
        raise checkpoints.CorruptCheckpointError(
            f"cannot bind checkpoint metadata ({loaded.path}): {reason}"
        ) from reason
    return {
        "path": str(Path(loaded.path).expanduser().resolve()),
        "format": persisted.format,
        "algorithm": persisted.algorithm,
        "trained": persisted.trained,
        "total_timesteps": persisted.total_timesteps,
        "metadata_file_sha256": metadata_file_sha256,
        "policy_artifact_sha256": persisted.policy_artifact_sha256,
        "checkpoint_contract_sha256": persisted.checkpoint_contract_digest,
    }


def _readmit_policy(
    loaded: LoadedPolicy,
    config: ControllerNeutralTrainingConfigV1,
    *,
    require_trained_policy: bool,
) -> tuple[LoadedPolicy, dict[str, Any]]:
    if not isinstance(loaded, LoadedPolicy):
        raise TypeError("loaded must be LoadedPolicy")
    _assert_economic_task(config)
    environment = make_env(config)
    try:
        if not isinstance(environment, PlantEconomicSetpointEnvV2):
            raise RuntimeError("economic task factory returned another environment")
        contract = policy_contract_for_environment(environment)
        loaded.metadata.check_compatible(path=loaded.path, contract=contract)
        readmitted = load_policy(
            loaded.path,
            contract=contract,
            environment=environment,
        )
    finally:
        _close_environment(environment)
    if readmitted.metadata != loaded.metadata:
        raise checkpoints.CorruptCheckpointError(
            f"checkpoint changed during policy readmission ({loaded.path})"
        )
    if require_trained_policy and (
        readmitted.metadata.format != checkpoints.FORMAT_SB3_PPO
        or not readmitted.metadata.trained
    ):
        raise checkpoints.IncompatibleCheckpointError(
            "the production benchmark requires a trained SB3 PPO checkpoint; "
            "fixed policies are integration fixtures only"
        )
    return readmitted, _checkpoint_identity(readmitted)


def load_benchmark_policy(
    checkpoint: Path | str,
) -> tuple[LoadedPolicy, ControllerNeutralTrainingConfigV1]:
    """Recover and fully admit a real PPO checkpoint's exact economic task."""

    config = load_training_task(checkpoint)
    _assert_economic_task(config)
    environment = make_env(config)
    try:
        loaded = load_policy(
            checkpoint,
            contract=policy_contract_for_environment(environment),
            environment=environment,
        )
    finally:
        _close_environment(environment)
    if (
        loaded.metadata.format != checkpoints.FORMAT_SB3_PPO
        or not loaded.metadata.trained
    ):
        raise checkpoints.IncompatibleCheckpointError(
            "the production benchmark requires a trained SB3 PPO checkpoint"
        )
    return loaded, config


def _new_economic_environment(
    config: ControllerNeutralTrainingConfigV1,
) -> PlantEconomicSetpointEnvV2:
    environment = make_env(config)
    if not isinstance(environment, PlantEconomicSetpointEnvV2):
        _close_environment(environment)
        raise RuntimeError(
            "benchmark task factory did not create an economic environment"
        )
    return environment


def _prepare_cell(
    loaded: LoadedPolicy,
    config: ControllerNeutralTrainingConfigV1,
    *,
    seed: int,
    service_settings: ServiceMPCSetpointSettingsV1,
    economic_controller_factory: EconomicControllerFactory,
) -> _PreparedCell:
    environments: dict[str, PlantEconomicSetpointEnvV2] = {}
    observations: dict[str, NDArray[np.generic]] = {}
    try:
        for arm in CONTROLLER_ARM_ORDER:
            environment = _new_economic_environment(config)
            observation, _ = environment.reset(seed=seed)
            if environment.decisions_taken != 0 or environment.physics_steps_taken != 0:
                raise RuntimeError(f"{arm} environment moved during reset preflight")
            environments[arm] = environment
            observations[arm] = observation.copy()

        identities = {arm: _manifest_identity(env) for arm, env in environments.items()}
        expected = identities[CONTROLLER_ARM_ORDER[0]]
        mismatched = [
            arm for arm, identity in identities.items() if identity != expected
        ]
        if mismatched:
            raise RuntimeError(
                "preflight manifest identity differs for arm(s): "
                + ", ".join(mismatched)
            )
        if expected["checkpoint_contract_sha256"] != (
            loaded.metadata.checkpoint_contract_digest
        ):
            raise checkpoints.IncompatibleCheckpointError(
                "prepared cell task identity differs from the PPO checkpoint"
            )

        heuristic = LegacyHeuristicSetpointControllerV1()
        service = ServiceMPCSetpointControllerV1(service_settings)
        economic = economic_controller_factory(config)
        policy = NormalizedSetpointPolicyControllerV1.from_environment(
            loaded, environments[PPO_ARM]
        )
        controllers: dict[str, object] = {
            HEURISTIC_ARM: heuristic,
            SERVICE_MPC_ARM: service,
            ECONOMIC_MPC_ARM: economic,
            PPO_ARM: policy,
        }
        for arm, controller in controllers.items():
            reset = getattr(controller, "reset", None)
            descriptor = getattr(controller, "to_dict", None)
            if not callable(reset) or not callable(descriptor):
                raise TypeError(f"{arm} controller does not satisfy the benchmark API")
            reset()
            described = descriptor()
            if not isinstance(described, Mapping) or not described.get("id"):
                raise TypeError(f"{arm} controller descriptor is malformed")
            if (
                environments[arm].decisions_taken
                or environments[arm].physics_steps_taken
            ):
                raise RuntimeError(f"{arm} controller moved the plant during preflight")
        return _PreparedCell(
            seed=seed,
            manifest_identity=dict(expected),
            environments=environments,
            controllers=controllers,
            policy_observations=observations,
        )
    except Exception:
        for environment in environments.values():
            _close_environment(environment)
        raise


def _economic_interval(info: Mapping[str, Any]) -> dict[str, Any]:
    interval = info.get("economic_interval")
    reward = info.get("reward")
    if not isinstance(interval, Mapping) or not isinstance(reward, Mapping):
        raise RuntimeError("economic interval evidence is malformed")
    return {"signals": dict(interval), "reward": dict(reward)}


def _generic_arm_result(
    *,
    arm: str,
    environment: PlantEconomicSetpointEnvV2,
    controller: _CommandController,
    manifest_identity: Mapping[str, str],
    initial_vector: NDArray[np.generic],
) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    rewards: list[float] = []
    controller_elapsed_ns: list[int] = []
    dispatcher_elapsed_ns: list[list[int]] = []
    observation_vector = initial_vector.copy()
    terminated = truncated = False
    failure: dict[str, Any] | None = None
    phase = "controller_compute"
    try:
        while not terminated and not truncated:
            if arm == HEURISTIC_ARM:
                source: object = environment.controller_observation
            else:
                source = environment.economic_controller_observation
            started_ns = time.perf_counter_ns()
            decision = controller.compute_command(
                source,
                constraints=environment.constraints,
                config=environment.config,
                commitment_profile=environment.episode_config.electrolyser_commitment,
            )
            controller_elapsed_ns.append(time.perf_counter_ns() - started_ns)
            decision_dict = getattr(decision, "to_dict", None)
            command = getattr(decision, "command", None)
            if not callable(decision_dict) or command is None:
                raise TypeError(
                    f"{arm} controller returned malformed decision evidence"
                )
            serialized = decision_dict()
            if not isinstance(serialized, Mapping):
                raise TypeError(f"{arm} decision serialization is malformed")
            if arm == PPO_ARM:
                evidence = getattr(decision, "evidence", None)
                if getattr(
                    evidence, "policy_observation_sha256", None
                ) != _vector_sha256(observation_vector):
                    raise RuntimeError(
                        "PPO controller encoding differs from environment observation"
                    )
            decisions.append(dict(serialized))
            raw_invalid = getattr(decision, "invalid", False)
            if not isinstance(raw_invalid, bool):
                raise TypeError(f"{arm} decision invalid flag is not bool")
            phase = "environment_execution"
            observation_vector, reward, terminated, truncated, info = (
                environment.step_process_setpoints(
                    command,
                    controller_invalid=raw_invalid,
                )
            )
            execution = environment.last_process_setpoint_execution
            if not isinstance(execution, ProcessSetpointIntervalExecutionV1):
                raise RuntimeError(f"{arm} omitted typed per-minute execution evidence")
            elapsed = info.get("dispatcher_elapsed_ns")
            if not isinstance(elapsed, list) or any(
                isinstance(value, bool) or not isinstance(value, Integral) or value < 0
                for value in elapsed
            ):
                raise RuntimeError(f"{arm} dispatcher timing evidence is malformed")
            executions.append(execution.to_dict())
            dispatcher_elapsed_ns.append([int(value) for value in elapsed])
            rewards.append(float(reward))
            intervals.append(_economic_interval(info))
            phase = "controller_compute"
    except Exception as reason:
        failure = _failure(f"{arm.lower()}_{phase}", reason)
    if failure is None and (not terminated or truncated):
        failure = _failure(
            f"{arm.lower()}_completion_validation",
            RuntimeError("episode did not reach its declared terminal boundary"),
        )
    status = "complete" if failure is None else "failed"
    run = {
        "format": CONTROLLER_NEUTRAL_BENCHMARK_ARM_FORMAT,
        "arm": arm,
        "status": status,
        "controller": controller.to_dict(),
        "manifest_identity": dict(manifest_identity),
        "completed_decisions": environment.decisions_taken,
        "physical_steps": environment.physics_steps_taken,
        "terminated": terminated,
        "truncated": truncated,
        "reward_trace_eur": rewards,
        "reward_trace_sum_eur": float(math.fsum(rewards)),
        "physical_metrics": environment.metrics.to_dict(),
        "economic_metrics": environment.economic_metrics.to_dict(),
        "evidence": {
            "decisions": decisions,
            "intervals": intervals,
            "executions": executions,
            "decision_count": len(decisions),
            "execution_count": len(executions),
            "physical_minute_count": sum(
                int(execution["application_count"]) for execution in executions
            ),
        },
        "controller_elapsed_ns": controller_elapsed_ns,
        "dispatcher_elapsed_ns": dispatcher_elapsed_ns,
        "failure": failure,
    }
    return _attach_arm_digest(run)


def _service_mpc_result(
    *,
    environment: PlantEconomicSetpointEnvV2,
    controller: ServiceMPCSetpointControllerV1,
    settings: ServiceMPCSetpointSettingsV1,
    manifest_identity: Mapping[str, str],
) -> dict[str, Any]:
    rewards: list[float] = []
    intervals: list[dict[str, Any]] = []
    dispatcher_elapsed: list[tuple[int, ...]] = []
    terminated = truncated = False
    failure: dict[str, Any] | None = None
    phase = "controller_compute"
    try:
        while not terminated and not truncated:
            decision = controller.compute_setpoints(
                environment.controller_observation,
                constraints=environment.constraints,
                config=environment.config,
                commitment_profile=environment.episode_config.electrolyser_commitment,
            )
            phase = "environment_execution"
            _, reward, terminated, truncated, info = environment.step_setpoints(
                decision.setpoints
            )
            execution = environment.last_setpoint_execution
            if not isinstance(execution, MPCSetpointIntervalExecutionV1):
                raise RuntimeError("M24 omitted typed per-minute execution evidence")
            elapsed = info.get("dispatcher_elapsed_ns")
            if not isinstance(elapsed, list) or any(
                isinstance(value, bool) or not isinstance(value, Integral) or value < 0
                for value in elapsed
            ):
                raise RuntimeError("M24 dispatcher timing evidence is malformed")
            phase = "transition_reconciliation"
            controller.observe_transition(
                environment.controller_observation,
                execution=execution,
            )
            dispatcher_elapsed.append(tuple(int(value) for value in elapsed))
            rewards.append(float(reward))
            intervals.append(_economic_interval(info))
            phase = "controller_compute"
    except Exception as reason:
        failure = _failure(f"m24_{phase}", reason)
        if controller.has_pending_transition:
            execution = environment.last_setpoint_execution
            failure["unreconciled_execution"] = (
                None if execution is None else execution.to_dict()
            )

    closed_decisions = sum(
        isinstance(event, MPCSetpointDecisionEvidenceV1) and event.closed
        for event in controller.decision_events
    )
    run_succeeded = bool(
        failure is None
        and terminated
        and not truncated
        and not controller.has_pending_transition
        and closed_decisions == settings.task_interval_count
    )
    if not run_succeeded and failure is None:
        failure = _failure(
            "m24_completion_validation",
            RuntimeError("episode did not reach its declared terminal boundary"),
        )
    evidence_failure: dict[str, Any] | None = None
    try:
        run_evidence = build_mpc_setpoint_run_evidence(
            controller.decision_events,
            tuple(dispatcher_elapsed),
            settings=settings,
            completed_decisions=closed_decisions,
            run_succeeded=run_succeeded,
            pending_transition=controller.has_pending_transition,
            runner_failure_phase=(
                None
                if run_succeeded
                else str(cast(Mapping[str, Any], failure)["phase"])
            ),
            final_observation_sha256=public_observation_sha256(
                environment.controller_observation
            ),
        )
        verify_mpc_setpoint_run_digests(run_evidence)
    except Exception as reason:
        run_evidence = {
            "events": [event.to_dict() for event in controller.decision_events],
            "dispatcher_elapsed_ns": [list(group) for group in dispatcher_elapsed],
        }
        evidence_failure = _failure("m24_evidence_validation", reason)
        if failure is None:
            failure = dict(evidence_failure)
        run_succeeded = False
    run = {
        "format": CONTROLLER_NEUTRAL_BENCHMARK_ARM_FORMAT,
        "arm": SERVICE_MPC_ARM,
        "status": "complete" if run_succeeded else "failed",
        "controller": controller.to_dict(),
        "manifest_identity": dict(manifest_identity),
        "completed_decisions": environment.decisions_taken,
        "closed_decisions": closed_decisions,
        "physical_steps": environment.physics_steps_taken,
        "terminated": terminated,
        "truncated": truncated,
        "reward_trace_eur": rewards,
        "reward_trace_sum_eur": float(math.fsum(rewards)),
        "physical_metrics": environment.metrics.to_dict(),
        "economic_metrics": environment.economic_metrics.to_dict(),
        "evidence": {
            "intervals": intervals,
            "run": run_evidence,
        },
        "failure": failure,
        "evidence_validation_failure": evidence_failure,
    }
    return _attach_arm_digest(run)


def _preflight_failed_cell(seed: int, reason: BaseException) -> dict[str, Any]:
    failure = _failure("cell_preflight", reason)
    runs = {
        arm: _attach_arm_digest(
            {
                "format": CONTROLLER_NEUTRAL_BENCHMARK_ARM_FORMAT,
                "arm": arm,
                "status": "not_run_preflight_failed",
                "controller": None,
                "manifest_identity": None,
                "completed_decisions": 0,
                "physical_steps": 0,
                "terminated": False,
                "truncated": False,
                "reward_trace_eur": [],
                "reward_trace_sum_eur": 0.0,
                "physical_metrics": None,
                "economic_metrics": None,
                "evidence": {"decisions": [], "executions": []},
                "failure": dict(failure),
            }
        )
        for arm in CONTROLLER_ARM_ORDER
    }
    cell = {
        "format": CONTROLLER_NEUTRAL_BENCHMARK_CELL_FORMAT,
        "seed": seed,
        "status": "preflight_failed",
        "paired_complete": False,
        "manifest_identity": None,
        "arm_order": list(CONTROLLER_ARM_ORDER),
        "runs": runs,
        "failure": failure,
    }
    cell["deterministic_evidence_sha256"] = domain_sha256(
        "controller-neutral-benchmark-cell-evidence-v1",
        _timing_free(cell),
    )
    return cell


def execute_controller_benchmark_cell(
    loaded: LoadedPolicy,
    config: ControllerNeutralTrainingConfigV1,
    *,
    seed: int,
    service_settings: ServiceMPCSetpointSettingsV1 | None = None,
    economic_controller_factory: EconomicControllerFactory | None = None,
) -> dict[str, Any]:
    """Execute one four-arm paired cell after fail-closed identity preflight."""

    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    settings = _service_settings(config, service_settings)
    factory = economic_controller_factory or _real_economic_controller
    try:
        prepared = _prepare_cell(
            loaded,
            config,
            seed=int(seed),
            service_settings=settings,
            economic_controller_factory=factory,
        )
    except Exception as reason:
        return _preflight_failed_cell(int(seed), reason)

    try:
        runs: dict[str, dict[str, Any]] = {}
        # PPO runs first so the already-admitted checkpoint has the shortest possible
        # admission/use interval.  Presentation order remains the frozen arm order.
        runs[PPO_ARM] = _generic_arm_result(
            arm=PPO_ARM,
            environment=prepared.environments[PPO_ARM],
            controller=cast(_CommandController, prepared.controllers[PPO_ARM]),
            manifest_identity=prepared.manifest_identity,
            initial_vector=prepared.policy_observations[PPO_ARM],
        )
        runs[HEURISTIC_ARM] = _generic_arm_result(
            arm=HEURISTIC_ARM,
            environment=prepared.environments[HEURISTIC_ARM],
            controller=cast(_CommandController, prepared.controllers[HEURISTIC_ARM]),
            manifest_identity=prepared.manifest_identity,
            initial_vector=prepared.policy_observations[HEURISTIC_ARM],
        )
        runs[SERVICE_MPC_ARM] = _service_mpc_result(
            environment=prepared.environments[SERVICE_MPC_ARM],
            controller=cast(
                ServiceMPCSetpointControllerV1,
                prepared.controllers[SERVICE_MPC_ARM],
            ),
            settings=settings,
            manifest_identity=prepared.manifest_identity,
        )
        runs[ECONOMIC_MPC_ARM] = _generic_arm_result(
            arm=ECONOMIC_MPC_ARM,
            environment=prepared.environments[ECONOMIC_MPC_ARM],
            controller=cast(_CommandController, prepared.controllers[ECONOMIC_MPC_ARM]),
            manifest_identity=prepared.manifest_identity,
            initial_vector=prepared.policy_observations[ECONOMIC_MPC_ARM],
        )
        ordered_runs = {arm: runs[arm] for arm in CONTROLLER_ARM_ORDER}
        paired_complete = all(run["status"] == "complete" for run in runs.values())
        cell = {
            "format": CONTROLLER_NEUTRAL_BENCHMARK_CELL_FORMAT,
            "seed": int(seed),
            "status": "complete" if paired_complete else "arm_failed",
            "paired_complete": paired_complete,
            "manifest_identity": dict(prepared.manifest_identity),
            "arm_order": list(CONTROLLER_ARM_ORDER),
            "runs": ordered_runs,
            "failure": None,
        }
        cell["deterministic_evidence_sha256"] = domain_sha256(
            "controller-neutral-benchmark-cell-evidence-v1",
            _timing_free(cell),
        )
        return cell
    finally:
        prepared.close()


def _number(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"complete arm has invalid numeric metric {key}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"complete arm has non-finite metric {key}")
    return number


def aggregate_complete_paired_cells(
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize only complete four-way pairs while exposing every failed arm."""

    paired = [cell for cell in cells if cell.get("paired_complete") is True]
    incomplete: list[dict[str, Any]] = []
    for cell in cells:
        if cell.get("paired_complete") is True:
            continue
        raw_runs = cell.get("runs")
        runs = raw_runs if isinstance(raw_runs, Mapping) else {}
        incomplete.append(
            {
                "seed": cell.get("seed"),
                "cell_status": cell.get("status"),
                "arm_status": {
                    arm: (
                        runs[arm].get("status")
                        if isinstance(runs.get(arm), Mapping)
                        else "missing"
                    )
                    for arm in CONTROLLER_ARM_ORDER
                },
                "arm_failures": {
                    arm: (
                        runs[arm].get("failure")
                        if isinstance(runs.get(arm), Mapping)
                        else {"phase": "aggregation", "message": "arm record missing"}
                    )
                    for arm in CONTROLLER_ARM_ORDER
                    if not isinstance(runs.get(arm), Mapping)
                    or runs[arm].get("status") != "complete"
                },
            }
        )

    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm in CONTROLLER_ARM_ORDER:
        rewards: list[float] = []
        delivered: list[float] = []
        unmet: list[float] = []
        values: list[float] = []
        for cell in paired:
            run = cast(Mapping[str, Any], cast(Mapping[str, Any], cell["runs"])[arm])
            physical = cast(Mapping[str, Any], run["physical_metrics"])
            economic = cast(Mapping[str, Any], run["economic_metrics"])
            rewards.append(_number(run, "reward_trace_sum_eur"))
            delivered.append(_number(physical, "delivered_kg"))
            unmet.append(_number(physical, "unmet_kg"))
            values.append(_number(economic, "net_operating_value_eur"))
        count = len(rewards)
        arm_summaries[arm] = {
            "paired_cell_count": count,
            "reward_mean_eur": None if not count else float(math.fsum(rewards) / count),
            "net_operating_value_mean_eur": (
                None if not count else float(math.fsum(values) / count)
            ),
            "delivered_mean_kg": (
                None if not count else float(math.fsum(delivered) / count)
            ),
            "unmet_mean_kg": None if not count else float(math.fsum(unmet) / count),
        }

    return {
        "aggregation_policy": "complete four-arm paired cells only",
        "registered_cell_count": len(cells),
        "complete_paired_cell_count": len(paired),
        "excluded_cell_count": len(cells) - len(paired),
        "complete_paired_seeds": [int(cell["seed"]) for cell in paired],
        "incomplete_cells": incomplete,
        "arms": arm_summaries,
    }


def evaluate_controller_benchmark(
    loaded: LoadedPolicy,
    config: ControllerNeutralTrainingConfigV1,
    *,
    seeds: Sequence[int],
    service_settings: ServiceMPCSetpointSettingsV1 | None = None,
    economic_controller_factory: EconomicControllerFactory | None = None,
    require_trained_policy: bool = True,
) -> dict[str, Any]:
    """Run a bounded four-controller comparison without hiding incomplete pairs."""

    seed_values = _validated_seeds(seeds)
    canonical_settings = _service_settings(config, None)
    settings = _service_settings(config, service_settings)
    if require_trained_policy and settings != canonical_settings:
        raise ValueError(
            "a qualified benchmark requires the canonical rolling M24 settings"
        )
    if require_trained_policy and economic_controller_factory is not None:
        raise ValueError(
            "a qualified benchmark cannot replace the production economic MPC arm"
        )
    readmitted, checkpoint_identity = _readmit_policy(
        loaded,
        config,
        require_trained_policy=require_trained_policy,
    )
    source_provenance = _source_provenance(require_clean=require_trained_policy)
    cells = [
        execute_controller_benchmark_cell(
            readmitted,
            config,
            seed=seed,
            service_settings=settings,
            economic_controller_factory=economic_controller_factory,
        )
        for seed in seed_values
    ]
    body: dict[str, Any] = {
        "format": CONTROLLER_NEUTRAL_BENCHMARK_FORMAT,
        "claim_boundary": CONTROLLER_NEUTRAL_BENCHMARK_CLAIM_BOUNDARY,
        "policy_qualification": (
            "trained-sb3-ppo"
            if readmitted.metadata.trained
            else "integration-fixture-not-a-learned-controller"
        ),
        "benchmark_qualification": (
            "canonical-four-arm-development-benchmark"
            if require_trained_policy
            else "unqualified-integration-fixture-or-custom-arm"
        ),
        "training_task": config.to_dict(),
        "checkpoint": checkpoint_identity,
        "source_provenance": source_provenance,
        "seeds": list(seed_values),
        "arm_order": list(CONTROLLER_ARM_ORDER),
        "service_mpc_settings": settings.to_dict(),
        "cells": cells,
        "aggregate": aggregate_complete_paired_cells(cells),
    }
    body["deterministic_evidence_sha256"] = domain_sha256(
        "controller-neutral-benchmark-evidence-v1",
        {
            "format": body["format"],
            "claim_boundary": body["claim_boundary"],
            "policy_qualification": body["policy_qualification"],
            "benchmark_qualification": body["benchmark_qualification"],
            "training_task": body["training_task"],
            "checkpoint": _timing_free(body["checkpoint"]),
            "source_provenance": body["source_provenance"],
            "seeds": body["seeds"],
            "arm_order": body["arm_order"],
            "service_mpc_settings": body["service_mpc_settings"],
            "cell_evidence": [cell["deterministic_evidence_sha256"] for cell in cells],
            "aggregate": body["aggregate"],
        },
    )
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare H, rolling M24 service MPC, tariff-aware economic MPC, and a "
            "trained PPO on identical checkpoint-bound economic V2 episodes."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--json", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.seed_start < 0 or arguments.count <= 0:
            raise ValueError(
                "seed-start must be non-negative and count must be positive"
            )
        if arguments.count > CONTROLLER_NEUTRAL_BENCHMARK_MAX_SEEDS:
            raise ValueError(
                f"count cannot exceed {CONTROLLER_NEUTRAL_BENCHMARK_MAX_SEEDS}"
            )
        if not arguments.json.is_absolute():
            raise ValueError("benchmark output must be an absolute new path")
        resolved_output = arguments.json.resolve()
        repository = _repository_root()
        if resolved_output == repository or repository in resolved_output.parents:
            raise ValueError("benchmark output must be outside the source repository")
        if arguments.json.exists():
            raise ValueError(
                f"benchmark output already exists at {arguments.json}; choose a new path"
            )
        loaded, config = load_benchmark_policy(arguments.checkpoint)
        result = evaluate_controller_benchmark(
            loaded,
            config,
            seeds=range(arguments.seed_start, arguments.seed_start + arguments.count),
        )
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        with arguments.json.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
            handle.write("\n")
    except (
        checkpoints.CheckpointError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as reason:
        print(f"controller benchmark refused: {reason}", file=sys.stderr)
        return 2

    summary = {
        "format": result["format"],
        "output": str(arguments.json),
        "seeds": result["seeds"],
        "complete_paired_cells": result["aggregate"]["complete_paired_cell_count"],
        "excluded_cells": result["aggregate"]["excluded_cell_count"],
        "deterministic_evidence_sha256": result["deterministic_evidence_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["aggregate"]["excluded_cell_count"] == 0 else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CONTROLLER_ARM_ORDER",
    "CONTROLLER_NEUTRAL_BENCHMARK_CLAIM_BOUNDARY",
    "CONTROLLER_NEUTRAL_BENCHMARK_FORMAT",
    "CONTROLLER_NEUTRAL_BENCHMARK_MAX_SEEDS",
    "EconomicControllerFactory",
    "aggregate_complete_paired_cells",
    "evaluate_controller_benchmark",
    "execute_controller_benchmark_cell",
    "load_benchmark_policy",
    "main",
]

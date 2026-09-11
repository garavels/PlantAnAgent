"""Fail-closed four-arm benchmark for the cyclic stochastic-market beta.

This runner is deliberately separate from the alpha controller benchmark.  A
cell is admitted only after four independent beta environments have reset to the
same immutable episode identity and the PPO checkpoint has been re-admitted to
that exact task.  Controller failures, safety failures, and cycle violations are
retained as results; none is replaced by a fallback controller.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from importlib import import_module
from importlib import metadata as importlib_metadata
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
import plant_sim

from .. import checkpoints
from ..policies import LoadedPolicy, load_policy
from . import beta_spaces
from .beta_control import CyclicMarketControllerObservationV1
from .beta_env import PlantCyclicMarketEnvV2Beta
from .beta_mpc_schema import beta_cycle_observation_sha256
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
    NormalizedSetpointPolicyControllerV1,
    policy_contract_for_environment,
)
from .setpoint_command import process_setpoint_boundary_sha256
from .setpoint_controllers import LegacyHeuristicSetpointControllerV1
from .study.legacy_admission import require_legacy_evaluation_scope
from .setpoint_execution import ProcessSetpointIntervalExecutionV1
from .training.config import (
    BETA_REGISTERED_RUN_KIND,
    ControllerNeutralTrainingConfigV1,
)
from .training.ppo import make_env


BETA_CONTROLLER_BENCHMARK_FORMAT = "cyclic-market-controller-benchmark-v1"
BETA_CONTROLLER_BENCHMARK_CELL_FORMAT = "cyclic-market-controller-benchmark-cell-v1"
BETA_CONTROLLER_BENCHMARK_ARM_FORMAT = "cyclic-market-controller-benchmark-arm-v1"
BETA_BOUNDARY_LEDGER_FORMAT = "cyclic-market-boundary-ledger-v1"
BETA_CONTROLLER_BENCHMARK_MAX_SEEDS = 32
BETA_SERVICE_COMPARISON_TOLERANCE_KG = 1e-6

HEURISTIC_ARM = "H"
SERVICE_MPC_ARM = "M24"
BETA_ECONOMIC_MPC_ARM = "E-beta"
PPO_ARM = "PPO"
BETA_CONTROLLER_ARM_ORDER = (
    HEURISTIC_ARM,
    SERVICE_MPC_ARM,
    BETA_ECONOMIC_MPC_ARM,
    PPO_ARM,
)

BETA_CONTROLLER_BENCHMARK_CLAIM_BOUNDARY = (
    "descriptive controller-neutral development benchmark on checkpoint-bound "
    "cyclic stochastic-market simulation episodes; no statistical-superiority, "
    "site-performance, savings, commission, or commercial claim"
)

BETA_CONTROLLER_INFORMATION_PROFILES: dict[str, dict[str, Any]] = {
    HEURISTIC_ARM: {
        "profile": "physical-reactive-ablation-v1",
        "input": "ControllerObservationV2 physical and public weather state",
        "market_forecast": False,
        "cycle_margins": False,
        "private_information": False,
    },
    SERVICE_MPC_ARM: {
        "profile": "physical-service-forecast-ablation-v1",
        "input": "ControllerObservationV2 physical state and public solar forecast",
        "market_forecast": False,
        "cycle_margins": False,
        "private_information": False,
    },
    BETA_ECONOMIC_MPC_ARM: {
        "profile": "full-public-beta-model-based-v1",
        "input": "CyclicMarketControllerObservationV1",
        "market_forecast": True,
        "cycle_margins": True,
        "private_information": False,
    },
    PPO_ARM: {
        "profile": "full-public-beta-learned-v1",
        "input": "checkpoint-declared 135-field beta observation vector",
        "market_forecast": True,
        "cycle_margins": True,
        "private_information": False,
    },
}

BETA_EXECUTION_DOORS: dict[str, str] = {
    HEURISTIC_ARM: "controller-neutral process command -> guard -> simulator",
    SERVICE_MPC_ARM: (
        "rich four-target MPC setpoints -> shared dispatcher/guard/simulator "
        "authority -> typed transition reconciliation"
    ),
    BETA_ECONOMIC_MPC_ARM: ("controller-neutral process command -> guard -> simulator"),
    PPO_ARM: "controller-neutral process command -> guard -> simulator",
}

_CONTROLLER_HARD_SAFETY_COUNT_METRICS = (
    "out_of_range_decisions",
    "electrolyser_forced_stops",
)
_BATTERY_SAFETY_TOLERANCE = 1e-9


class _CommandController(Protocol):
    id: str

    def reset(self) -> None: ...

    def compute_command(self, observation: object, **kwargs: object) -> object: ...

    def to_dict(self) -> dict[str, Any]: ...


BetaEconomicControllerFactory = Callable[
    [ControllerNeutralTrainingConfigV1], _CommandController
]


@dataclass(slots=True)
class _PreparedBetaCell:
    seed: int
    manifest_identity: dict[str, str]
    audit_manifest: dict[str, Any]
    environments: dict[str, PlantCyclicMarketEnvV2Beta]
    controllers: dict[str, object]
    policy_observations: dict[str, NDArray[np.generic]]

    def close(self) -> None:
        for environment in self.environments.values():
            _close_environment(environment)


def _close_environment(environment: object) -> None:
    cast(Any, environment).close()


def _validated_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    values: list[int] = []
    for index, seed in enumerate(seeds):
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError(f"seed {index} must be a non-negative integer")
        values.append(int(seed))
    if not values:
        raise ValueError("beta controller benchmark requires at least one seed")
    if len(values) > BETA_CONTROLLER_BENCHMARK_MAX_SEEDS:
        raise ValueError(
            "beta controller benchmark accepts at most "
            f"{BETA_CONTROLLER_BENCHMARK_MAX_SEEDS} seeds per invocation"
        )
    if len(set(values)) != len(values):
        raise ValueError("beta controller benchmark seeds must be unique")
    return tuple(values)


def _repository_root() -> Path:
    root = Path(__file__).resolve().parents[5]
    if not (root / "pyproject.toml").is_file() or not (root / "uv.lock").is_file():
        raise RuntimeError("beta controller benchmark must run from a source checkout")
    return root


def _runtime_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


def _source_provenance(*, require_clean: bool) -> dict[str, Any]:
    repository = _repository_root()
    raw_plant_sim_path = getattr(plant_sim, "__file__", None)
    if not isinstance(raw_plant_sim_path, str) or not raw_plant_sim_path:
        raise RuntimeError("cannot identify the imported plant_sim implementation")
    plant_sim_path = Path(raw_plant_sim_path).resolve()
    if not plant_sim_path.is_file():
        raise RuntimeError("imported plant_sim source path is not a file")
    expected_simulator_root = (repository / "simulator").resolve()
    if require_clean and expected_simulator_root not in plant_sim_path.parents:
        raise RuntimeError(
            "qualified beta benchmark must use plant_sim from this source checkout"
        )

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
        raise RuntimeError("beta benchmark source revision is not a full Git object id")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    clean = not status
    if require_clean and not clean:
        raise RuntimeError(
            "a qualified beta controller benchmark requires a clean committed worktree"
        )
    return {
        "git_revision": revision,
        "worktree_clean": clean,
        "worktree_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "dependency_lock_sha256": hashlib.sha256(
            (repository / "uv.lock").read_bytes()
        ).hexdigest(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": _runtime_version("numpy"),
            "scipy": _runtime_version("scipy"),
            "stable_baselines3": _runtime_version("stable-baselines3"),
            "gymnasium": _runtime_version("gymnasium"),
            "plant_sim": {
                "version": str(plant_sim.__version__),
                "path": str(plant_sim_path),
                "module_sha256": hashlib.sha256(
                    plant_sim_path.read_bytes()
                ).hexdigest(),
            },
        },
    }


def _require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"beta manifest omitted {name}")
    return cast(Mapping[str, Any], value)


def _require_digest(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"beta manifest {name} is not a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as reason:
        raise RuntimeError(f"beta manifest {name} is not a SHA-256 digest") from reason
    return value


def _beta_manifest_identity(
    environment: PlantCyclicMarketEnvV2Beta,
) -> dict[str, str]:
    manifest = environment.audit_manifest()
    beta_exogenous = _require_mapping("beta_exogenous", manifest.get("beta_exogenous"))
    disturbances = _require_mapping("disturbances", manifest.get("disturbances"))
    identity = {
        "manifest_sha256": manifest.get("manifest_sha256"),
        "alpha_physical_exogenous_episode_sha256": manifest.get(
            "alpha_physical_exogenous_episode_sha256"
        ),
        "beta_exogenous_episode_sha256": manifest.get("beta_exogenous_episode_sha256"),
        "market_episode_sha256": beta_exogenous.get("market_episode_sha256"),
        "disturbance_episode_sha256": beta_exogenous.get("disturbance_episode_sha256"),
        "disturbance_schedule_sha256": disturbances.get("schedule_sha256"),
        "cycle_task_sha256": beta_exogenous.get("cycle_task_sha256"),
        "checkpoint_contract_sha256": environment.checkpoint_contract_digest,
        "economic_assumptions_sha256": environment.economic_assumptions.sha256,
        "beta_task_sha256": environment.beta_task.sha256,
    }
    return {name: _require_digest(name, value) for name, value in identity.items()}


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
    """Remove wall-clock/path fields from deterministic scientific identities."""

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


def _attach_arm_digest(run: dict[str, Any]) -> dict[str, Any]:
    body = cast(dict[str, Any], _timing_free(dict(run)))
    body.pop("deterministic_evidence_sha256", None)
    run["deterministic_evidence_sha256"] = domain_sha256(
        "cyclic-market-controller-benchmark-arm-evidence-v1",
        body,
    )
    return run


def _attach_cell_digest(cell: dict[str, Any]) -> dict[str, Any]:
    body = cast(dict[str, Any], _timing_free(dict(cell)))
    body.pop("deterministic_evidence_sha256", None)
    cell["deterministic_evidence_sha256"] = domain_sha256(
        "cyclic-market-controller-benchmark-cell-evidence-v1",
        body,
    )
    return cell


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
    for name, actual, expected in (
        ("task horizon", supplied.task_horizon_h, config.episode.horizon_h),
        ("cadence", supplied.interval_s, config.episode.decision_interval_s),
        (
            "planning horizon",
            supplied.planning_horizon_h,
            min(24.0, config.episode.horizon_h),
        ),
    ):
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"service MPC {name} differs from the beta task")
    return supplied


def _real_beta_economic_controller(
    config: ControllerNeutralTrainingConfigV1,
) -> _CommandController:
    """Resolve only the production cyclic-market economic MPC implementation."""

    if config.beta_task is None:  # pragma: no cover - task validation closes this
        raise RuntimeError("validated beta task omitted beta_task")
    try:
        controller_module = import_module(".beta_mpc_controller", package=__package__)
        schema_module = import_module(".beta_mpc_schema", package=__package__)
        controller_type = getattr(
            controller_module, "CyclicMarketEconomicMPCControllerV1"
        )
        settings_type = getattr(schema_module, "BetaMPCSettingsV1")
    except (AttributeError, ImportError) as reason:
        raise RuntimeError(
            "the production cyclic-market economic MPC implementation is unavailable"
        ) from reason
    settings = settings_type(
        task_horizon_h=config.episode.horizon_h,
        planning_horizon_h=min(24.0, config.episode.horizon_h),
        interval_s=config.episode.decision_interval_s,
    )
    controller = controller_type(
        settings=settings,
        assumptions=config.economic_assumptions,
        beta_task=config.beta_task,
    )
    if not callable(getattr(controller, "compute_command", None)):
        raise TypeError("cyclic-market economic MPC has no compute_command method")
    return cast(_CommandController, controller)


def _assert_beta_task(config: ControllerNeutralTrainingConfigV1) -> None:
    config.validate()
    if config.environment_id != beta_spaces.ENVIRONMENT_ID:
        raise ValueError(
            "the beta controller benchmark requires the exact cyclic-market beta task"
        )
    if config.beta_task is None:  # pragma: no cover - config.validate closes this
        raise RuntimeError("validated cyclic-market beta task omitted beta_task")


def _checkpoint_identity(loaded: LoadedPolicy) -> dict[str, Any]:
    persisted = checkpoints.load_metadata(loaded.path, check=False)
    if persisted != loaded.metadata:
        raise checkpoints.CorruptCheckpointError(
            f"checkpoint metadata changed after loading ({loaded.path})"
        )
    contract = checkpoints.PolicyEnvironmentContract(
        spec_version=persisted.spec_version,
        observation_fields=persisted.observation_fields,
        action_fields=persisted.action_fields,
        environment_id=persisted.environment_id,
        environment_contract_digest=persisted.environment_contract_digest,
        checkpoint_contract_digest=persisted.checkpoint_contract_digest,
    )
    checked = checkpoints.load_metadata(loaded.path, contract=contract)
    if checked != persisted:
        raise checkpoints.CorruptCheckpointError(
            f"checkpoint changed during beta benchmark admission ({loaded.path})"
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
        "plant_sim_version": persisted.plant_sim_version,
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
    _assert_beta_task(config)
    require_legacy_evaluation_scope(
        config.environment_id, training_config=config.to_dict()
    )
    require_legacy_evaluation_scope(
        loaded.metadata.environment_id, training_config=loaded.metadata.training_config
    )
    if not loaded.metadata.training_config:
        raise checkpoints.CorruptCheckpointError(
            "beta benchmark checkpoint has no controller-neutral training_config"
        )
    try:
        checkpoint_config = ControllerNeutralTrainingConfigV1.from_dict(
            loaded.metadata.training_config
        )
    except (TypeError, ValueError) as reason:
        if isinstance(reason, checkpoints.CheckpointError):
            raise
        raise checkpoints.CorruptCheckpointError(
            f"beta benchmark checkpoint has invalid training_config: {reason}"
        ) from reason
    if checkpoint_config != config:
        raise checkpoints.IncompatibleCheckpointError(
            "benchmark training task differs from checkpoint training_config"
        )
    if loaded.metadata.trained and loaded.metadata.seed != config.seed:
        raise checkpoints.CorruptCheckpointError(
            "trained checkpoint seed differs from checkpoint training_config"
        )
    if require_trained_policy and (
        config.beta_rl is None or config.beta_rl.run_kind != BETA_REGISTERED_RUN_KIND
    ):
        raise checkpoints.IncompatibleCheckpointError(
            "a qualified beta benchmark requires a registered constrained-RL run"
        )
    if loaded.metadata.plant_sim_version != str(plant_sim.__version__):
        raise checkpoints.IncompatibleCheckpointError(
            "checkpoint plant_sim version differs from the benchmark runtime"
        )
    environment = _new_beta_environment(config)
    try:
        contract = policy_contract_for_environment(environment)
        loaded.metadata.check_compatible(path=loaded.path, contract=contract)
        readmitted = load_policy(
            loaded.path,
            contract=contract,
            environment=environment,
            require_registered_evaluation_eligibility=require_trained_policy,
        )
    finally:
        _close_environment(environment)
    if readmitted.metadata != loaded.metadata:
        raise checkpoints.CorruptCheckpointError(
            f"checkpoint changed during policy readmission ({loaded.path})"
        )
    readmitted = LoadedPolicy(
        policy=readmitted.policy,
        metadata=readmitted.metadata,
        path=Path(readmitted.path).expanduser().resolve(),
    )
    if require_trained_policy and (
        readmitted.metadata.format != checkpoints.FORMAT_SB3_PPO
        or not readmitted.metadata.trained
    ):
        raise checkpoints.IncompatibleCheckpointError(
            "the qualified beta benchmark requires a trained SB3 PPO checkpoint; "
            "fixed policies are integration fixtures only"
        )
    return readmitted, _checkpoint_identity(readmitted)


def load_beta_benchmark_policy(
    checkpoint: Path | str,
) -> tuple[LoadedPolicy, ControllerNeutralTrainingConfigV1]:
    """Recover and fully admit a trained PPO checkpoint's exact beta task."""

    config = load_training_task(checkpoint)
    _assert_beta_task(config)
    require_legacy_evaluation_scope(
        config.environment_id, training_config=config.to_dict()
    )
    environment = _new_beta_environment(config)
    try:
        loaded = load_policy(
            checkpoint,
            contract=policy_contract_for_environment(environment),
            environment=environment,
            require_registered_evaluation_eligibility=True,
        )
    finally:
        _close_environment(environment)
    if (
        loaded.metadata.format != checkpoints.FORMAT_SB3_PPO
        or not loaded.metadata.trained
    ):
        raise checkpoints.IncompatibleCheckpointError(
            "the qualified beta benchmark requires a trained SB3 PPO checkpoint"
        )
    return loaded, config


def _new_beta_environment(
    config: ControllerNeutralTrainingConfigV1,
) -> PlantCyclicMarketEnvV2Beta:
    environment = make_env(config)
    if type(environment) is not PlantCyclicMarketEnvV2Beta:
        _close_environment(environment)
        raise RuntimeError(
            "beta benchmark task factory did not create the exact beta environment"
        )
    return environment


def _prepare_cell(
    loaded: LoadedPolicy,
    config: ControllerNeutralTrainingConfigV1,
    *,
    seed: int,
    service_settings: ServiceMPCSetpointSettingsV1,
    economic_controller_factory: BetaEconomicControllerFactory,
) -> _PreparedBetaCell:
    environments: dict[str, PlantCyclicMarketEnvV2Beta] = {}
    observations: dict[str, NDArray[np.generic]] = {}
    try:
        for arm in BETA_CONTROLLER_ARM_ORDER:
            environment = _new_beta_environment(config)
            observation, info = environment.reset(seed=seed)
            if info.get("environment_id") != beta_spaces.ENVIRONMENT_ID:
                raise RuntimeError(f"{arm} reset returned another environment identity")
            if environment.decisions_taken != 0 or environment.physics_steps_taken != 0:
                raise RuntimeError(f"{arm} environment moved during reset preflight")
            environments[arm] = environment
            observations[arm] = observation.copy()

        identities = {
            arm: _beta_manifest_identity(environment)
            for arm, environment in environments.items()
        }
        expected = identities[HEURISTIC_ARM]
        mismatched = [
            arm for arm, identity in identities.items() if identity != expected
        ]
        if mismatched:
            raise RuntimeError(
                "preflight beta manifest identity differs for arm(s): "
                + ", ".join(mismatched)
            )
        vector_mismatches = [
            arm
            for arm, vector in observations.items()
            if not np.array_equal(vector, observations[HEURISTIC_ARM])
        ]
        if vector_mismatches:
            raise RuntimeError(
                "preflight initial beta observation differs for arm(s): "
                + ", ".join(vector_mismatches)
            )
        if expected["checkpoint_contract_sha256"] != (
            loaded.metadata.checkpoint_contract_digest
        ):
            raise checkpoints.IncompatibleCheckpointError(
                "prepared beta cell task identity differs from the PPO checkpoint"
            )
        loaded.metadata.check_compatible(
            path=loaded.path,
            contract=policy_contract_for_environment(environments[PPO_ARM]),
        )

        controllers: dict[str, object] = {
            HEURISTIC_ARM: LegacyHeuristicSetpointControllerV1(),
            SERVICE_MPC_ARM: ServiceMPCSetpointControllerV1(service_settings),
            BETA_ECONOMIC_MPC_ARM: economic_controller_factory(config),
            PPO_ARM: NormalizedSetpointPolicyControllerV1.from_environment(
                loaded, environments[PPO_ARM]
            ),
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
            environment = environments[arm]
            if environment.decisions_taken or environment.physics_steps_taken:
                raise RuntimeError(f"{arm} controller moved the plant during preflight")

        return _PreparedBetaCell(
            seed=seed,
            manifest_identity=dict(expected),
            audit_manifest=environments[HEURISTIC_ARM].audit_manifest(),
            environments=environments,
            controllers=controllers,
            policy_observations=observations,
        )
    except Exception:
        for environment in environments.values():
            _close_environment(environment)
        raise


def _boundary_record(
    environment: PlantCyclicMarketEnvV2Beta,
    vector: NDArray[np.generic] | None,
    *,
    kind: str,
    previous_solar_kw: float | None = None,
) -> dict[str, Any]:
    observation = environment.cyclic_market_controller_observation
    if not isinstance(observation, CyclicMarketControllerObservationV1):
        raise RuntimeError("beta environment omitted its typed controller observation")
    current_vector = np.asarray(environment._observe(), dtype=np.float32)
    current_solar_kw = float(observation.plant.state.boundary.solar_available_kw)
    prior_solar_kw = (
        current_solar_kw if previous_solar_kw is None else float(previous_solar_kw)
    )
    solar_trend_index = beta_spaces.OBSERVATION_NAMES.index("solar_trend")
    current_vector[solar_trend_index] = np.float32(
        np.clip(
            (current_solar_kw - prior_solar_kw) / environment.config.solar.peak_kw,
            -1.0,
            1.0,
        )
    )
    if vector is not None and not np.array_equal(
        current_vector,
        np.asarray(vector, dtype=np.float32),
    ):
        raise RuntimeError(
            "returned beta vector differs from the current typed environment boundary"
        )
    flattened = [float(value) for value in current_vector.reshape(-1)]
    if len(flattened) != beta_spaces.OBSERVATION_SIZE:
        raise RuntimeError("beta boundary vector has the wrong size")
    return {
        "format": "cyclic-market-controller-boundary-v1",
        "kind": kind,
        "decision_index": environment.decisions_taken,
        "physics_step": environment.physics_steps_taken,
        "terminal": environment.decisions_taken == environment.episode_decisions,
        "typed_observation": observation.to_dict(),
        "typed_observation_sha256": observation.sha256,
        "physical_observation_sha256": public_observation_sha256(observation.plant),
        "market_issue_sha256": observation.market.sha256,
        "cycle_observation_sha256": beta_cycle_observation_sha256(observation.cycle),
        "normalized_observation": flattened,
        "normalized_observation_sha256": _vector_sha256(current_vector),
    }


def _economic_interval(info: Mapping[str, Any]) -> dict[str, Any]:
    interval = info.get("economic_interval")
    reward = info.get("reward")
    cycle = info.get("cycle_constraint")
    if (
        not isinstance(interval, Mapping)
        or not isinstance(reward, Mapping)
        or not isinstance(cycle, Mapping)
    ):
        raise RuntimeError("beta economic/cycle interval evidence is malformed")
    return {
        "signals": dict(interval),
        "reward": dict(reward),
        "cycle_constraint": dict(cycle),
    }


def _integer_metric(metrics: Mapping[str, Any], key: str) -> int:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise RuntimeError(f"physical metric {key} is not a non-negative integer")
    return int(value)


def _declared_dac_outage_step_indices(
    environment: PlantCyclicMarketEnvV2Beta,
) -> frozenset[int]:
    """Return private exogenous outage steps intersecting executed physics."""

    executed_steps = environment.physics_steps_taken
    dt_s = float(environment.config.dt_s)
    outage_steps: set[int] = set()
    for fault in environment.disturbance_episode.faults:
        if (
            fault.type != "dac_outage"
            or fault.start_s is None
            or fault.duration_s is None
        ):
            raise RuntimeError("beta disturbance schedule is not a finite DAC outage")
        start_step = int(round(float(fault.start_s) / dt_s))
        duration_steps = int(round(float(fault.duration_s) / dt_s))
        if not math.isclose(
            float(fault.start_s), start_step * dt_s, rel_tol=0.0, abs_tol=1e-9
        ) or not math.isclose(
            float(fault.duration_s),
            duration_steps * dt_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError("beta disturbance schedule left the physics lattice")
        outage_steps.update(
            range(
                max(0, start_step),
                min(executed_steps, start_step + duration_steps),
            )
        )
    return frozenset(outage_steps)


def _hard_safety_classification(
    environment: PlantCyclicMarketEnvV2Beta,
    physical_metrics: Mapping[str, Any],
    minute_boundaries: Sequence[Mapping[str, Any]],
) -> tuple[bool, dict[str, int | float | bool]]:
    """Reconstruct safety and separate only isolated declared DAC unavailability."""

    executed_steps = environment.physics_steps_taken
    if len(minute_boundaries) != executed_steps + 1:
        raise RuntimeError(
            "minute safety evidence does not span every executed physical step"
        )
    allowed_soc = float(environment.config.battery.min_soc)
    minimum_soc = 1.0
    observed_shutdown = 0
    unexpected_shutdown = 0
    declared_indices = _declared_dac_outage_step_indices(environment)
    realized_declared_indices: set[int] = set()
    for expected_step, boundary in enumerate(minute_boundaries):
        step = boundary.get("step")
        if isinstance(step, bool) or not isinstance(step, Integral):
            raise RuntimeError("minute safety boundary step is not an integer")
        if int(step) != expected_step:
            raise RuntimeError("minute safety boundaries are not contiguous")
        soc = _metric_number(boundary, "battery_soc")
        if not 0.0 <= soc <= 1.0:
            raise RuntimeError("minute battery SOC lies outside [0, 1]")
        minimum_soc = min(minimum_soc, soc)
        if expected_step == 0:
            continue
        status = boundary.get("plant_status")
        if status not in {"normal", "degraded", "critical", "offline"}:
            raise RuntimeError("minute plant status is invalid")
        critical = status in {"critical", "offline"}
        observed_shutdown += int(critical)
        executed_index = expected_step - 1
        declared = executed_index in declared_indices
        health = boundary.get("health")
        if not isinstance(health, Mapping):
            raise RuntimeError("minute component-health evidence is malformed")
        offline: dict[str, bool] = {}
        for component, raw_health in health.items():
            if not isinstance(component, str) or not isinstance(raw_health, Mapping):
                raise RuntimeError("minute component-health evidence is malformed")
            raw_offline = raw_health.get("offline")
            if not isinstance(raw_offline, bool):
                raise RuntimeError("minute component offline flag is not boolean")
            offline[component] = raw_offline
        if "dac" not in offline:
            raise RuntimeError("minute component-health evidence omits DAC")
        isolated_declared_dac = bool(
            declared
            and critical
            and offline["dac"]
            and not any(
                value for component, value in offline.items() if component != "dac"
            )
            and soc > allowed_soc + _BATTERY_SAFETY_TOLERANCE
        )
        if declared and critical and offline["dac"]:
            realized_declared_indices.add(executed_index)
        if critical and not isolated_declared_dac:
            unexpected_shutdown += 1
    if realized_declared_indices != declared_indices:
        raise RuntimeError(
            "declared DAC outage is not exactly expressed in minute health evidence"
        )
    declared_outage = len(declared_indices)
    if observed_shutdown != _integer_metric(physical_metrics, "shutdown_steps"):
        raise RuntimeError("physical shutdown metric differs from minute evidence")
    reported_minimum_soc = _metric_number(physical_metrics, "min_battery_soc")
    if not math.isclose(
        minimum_soc,
        reported_minimum_soc,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("physical minimum SOC differs from minute evidence")
    counts = {
        key: _integer_metric(physical_metrics, key)
        for key in _CONTROLLER_HARD_SAFETY_COUNT_METRICS
    }
    battery_reserve_respected = bool(
        minimum_soc > allowed_soc + _BATTERY_SAFETY_TOLERANCE
    )
    evidence: dict[str, int | float | bool] = {
        "observed_critical_or_offline_steps": observed_shutdown,
        "declared_dac_outage_steps": declared_outage,
        "unexpected_critical_or_offline_steps": unexpected_shutdown,
        **counts,
        "minimum_battery_soc": minimum_soc,
        "minimum_allowed_battery_soc": allowed_soc,
        "battery_reserve_respected": battery_reserve_respected,
    }
    passed = bool(
        unexpected_shutdown == 0
        and all(value == 0 for value in counts.values())
        and battery_reserve_respected
    )
    return passed, evidence


def _classify_run(
    *,
    environment: PlantCyclicMarketEnvV2Beta,
    failure: Mapping[str, Any] | None,
    terminated: bool,
    truncated: bool,
    minute_boundaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_decisions = environment.episode_decisions
    execution_complete = bool(
        failure is None
        and terminated
        and not truncated
        and environment.decisions_taken == expected_decisions
        and environment.physics_steps_taken == environment.episode_physics_steps
    )
    expected_cycle_count = (
        expected_decisions
        // environment.beta_task.cycle.cycle_decisions(environment.episode_config)
    )
    settlements = environment.cycle_settlements
    settlement_boundaries = [item.boundary_decision for item in settlements]
    expected_boundaries = [
        environment.beta_task.cycle.cycle_decisions(environment.episode_config)
        * (index + 1)
        for index in range(expected_cycle_count)
    ]
    cycle_feasible = bool(
        execution_complete
        and len(settlements) == expected_cycle_count
        and settlement_boundaries == expected_boundaries
        and all(item.feasible for item in settlements)
    )
    physical_metrics = environment.metrics.to_dict()
    hard_safety_passed, safety_values = _hard_safety_classification(
        environment,
        physical_metrics,
        minute_boundaries,
    )
    if not execution_complete:
        status = "execution_failed"
    elif not cycle_feasible:
        status = "complete_cycle_infeasible"
    elif not hard_safety_passed:
        status = "complete_hard_safety_failed"
    else:
        status = "complete_feasible_safe"
    exclusions: list[str] = []
    if not execution_complete:
        exclusions.append("execution_incomplete")
    if not cycle_feasible:
        exclusions.append("cycle_infeasible_or_unsettled")
    if not hard_safety_passed:
        exclusions.append("hard_safety_failed")
    return {
        "status": status,
        "execution_complete": execution_complete,
        "cycle_feasible": cycle_feasible,
        "hard_safety_passed": hard_safety_passed,
        "value_comparison_candidate": bool(
            execution_complete and cycle_feasible and hard_safety_passed
        ),
        # Eligibility is a paired-cell property and is filled only after the
        # service-comparability gate has inspected all four completed arms.
        "value_comparison_eligible": False,
        "exclusion_reasons": exclusions,
        "expected_decisions": expected_decisions,
        "expected_cycle_count": expected_cycle_count,
        "hard_safety_metrics": safety_values,
        "cycle_settlements": [item.to_dict() for item in settlements],
        "cumulative_cycle_constraint_cost": float(
            math.fsum(item.total_constraint_cost for item in settlements)
        ),
    }


def _final_boundary(
    environment: PlantCyclicMarketEnvV2Beta,
    *,
    previous_solar_kw: float | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        return (
            _boundary_record(
                environment,
                None,
                kind="final",
                previous_solar_kw=previous_solar_kw,
            ),
            None,
        )
    except Exception as reason:
        return None, _failure("final_boundary_capture", reason)


def _validate_generic_decision_binding(
    *,
    arm: str,
    decision: object,
    boundary: Mapping[str, Any],
) -> None:
    """Reject stale or relabelled controller evidence before plant movement."""

    command = getattr(decision, "command", None)
    if command is None:
        raise TypeError(f"{arm} decision omitted its process command")
    if getattr(command, "decision_index", None) != boundary["decision_index"]:
        raise RuntimeError(f"{arm} command decision index differs from the boundary")
    if getattr(command, "observation_sha256", None) != (
        boundary["physical_observation_sha256"]
    ):
        raise RuntimeError(f"{arm} command is bound to another physical observation")

    evidence = getattr(decision, "evidence", None)
    if arm == BETA_ECONOMIC_MPC_ARM:
        if evidence is None:
            raise TypeError("E-beta decision omitted its typed evidence")
        expected = {
            "beta_observation_sha256": boundary["typed_observation_sha256"],
            "physical_observation_sha256": boundary["physical_observation_sha256"],
            "market_issue_sha256": boundary["market_issue_sha256"],
            "cycle_observation_sha256": boundary["cycle_observation_sha256"],
        }
        for field, digest in expected.items():
            if getattr(evidence, field, None) != digest:
                raise RuntimeError(
                    f"E-beta {field} differs from the captured beta boundary"
                )
    elif arm == PPO_ARM:
        if evidence is None:
            raise TypeError("PPO decision omitted its typed evidence")
        if getattr(evidence, "policy_observation_sha256", None) != (
            boundary["normalized_observation_sha256"]
        ):
            raise RuntimeError(
                "PPO controller encoding differs from beta environment observation"
            )
        if getattr(evidence, "economic_observation_sha256", None) != (
            boundary["typed_observation_sha256"]
        ):
            raise RuntimeError(
                "PPO decision is not bound to the typed beta observation"
            )


def _generic_arm_result(
    *,
    arm: str,
    environment: PlantCyclicMarketEnvV2Beta,
    controller: _CommandController,
    manifest_identity: Mapping[str, str],
    initial_vector: NDArray[np.generic],
) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    minute_boundaries: list[dict[str, Any]] = [environment.telemetry.to_dict()]
    rewards: list[float] = []
    decision_boundaries: list[dict[str, Any]] = []
    controller_elapsed_ns: list[int] = []
    dispatcher_elapsed_ns: list[list[int]] = []
    observation_vector = initial_vector.copy()
    previous_solar_kw: float | None = None
    terminated = truncated = False
    failure: dict[str, Any] | None = None
    phase = "boundary_capture"
    try:
        while not terminated and not truncated:
            boundary = _boundary_record(
                environment,
                observation_vector,
                kind="decision",
                previous_solar_kw=previous_solar_kw,
            )
            decision_boundaries.append(boundary)
            typed = environment.cyclic_market_controller_observation
            decision_solar_kw = float(typed.plant.state.boundary.solar_available_kw)
            source: object = typed.plant if arm == HEURISTIC_ARM else typed
            phase = "controller_compute"
            started_ns = time.perf_counter_ns()
            decision = controller.compute_command(
                source,
                constraints=environment.constraints,
                config=environment.config,
                commitment_profile=environment.episode_config.electrolyser_commitment,
            )
            controller_elapsed_ns.append(time.perf_counter_ns() - started_ns)
            serializer = getattr(decision, "to_dict", None)
            command = getattr(decision, "command", None)
            if not callable(serializer) or command is None:
                raise TypeError(
                    f"{arm} controller returned malformed decision evidence"
                )
            serialized = serializer()
            if not isinstance(serialized, Mapping):
                raise TypeError(f"{arm} decision serialization is malformed")
            _validate_generic_decision_binding(
                arm=arm,
                decision=decision,
                boundary=boundary,
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
            previous_solar_kw = decision_solar_kw
            execution = environment.last_process_setpoint_execution
            if not isinstance(execution, ProcessSetpointIntervalExecutionV1):
                raise RuntimeError(f"{arm} omitted typed per-minute execution evidence")
            elapsed = info.get("dispatcher_elapsed_ns")
            if not isinstance(elapsed, list) or any(
                isinstance(value, bool) or not isinstance(value, Integral) or value < 0
                for value in elapsed
            ):
                raise RuntimeError(f"{arm} dispatcher timing evidence is malformed")
            serialized_execution = execution.to_dict()
            serialized_elapsed = [int(value) for value in elapsed]
            interval = _economic_interval(info)
            reward_value = float(reward)
            if not math.isfinite(reward_value):
                raise RuntimeError(f"{arm} reward is non-finite")
            truth_states = environment.last_minute_truth_states
            if (
                truth_states is None
                or len(truth_states) != len(execution.applications) + 1
            ):
                raise RuntimeError(f"{arm} omitted its complete minute truth trace")
            if truth_states[0].to_dict() != minute_boundaries[-1]:
                raise RuntimeError(f"{arm} minute truth-state boundary chain is broken")
            for application, before_state, after_state in zip(
                execution.applications,
                truth_states[:-1],
                truth_states[1:],
                strict=True,
            ):
                if application.boundary_sha256 != process_setpoint_boundary_sha256(
                    before_state
                ) or application.next_boundary_sha256 != process_setpoint_boundary_sha256(
                    after_state
                ):
                    raise RuntimeError(
                        f"{arm} minute truth trace differs from execution evidence"
                    )
                minute_boundaries.append(after_state.to_dict())
            # Publish one completed interval atomically. If movement succeeded but
            # any required evidence is malformed, the integrity check below aborts
            # the whole artifact instead of emitting an unverifiable partial arm.
            executions.append(serialized_execution)
            dispatcher_elapsed_ns.append(serialized_elapsed)
            rewards.append(reward_value)
            intervals.append(interval)
            phase = "boundary_capture"
    except Exception as reason:
        retained_minutes = sum(
            int(execution["application_count"]) for execution in executions
        )
        if (
            environment.decisions_taken != len(intervals)
            or environment.physics_steps_taken != retained_minutes
        ):
            raise RuntimeError(
                f"{arm} moved without a complete verifiable interval bundle"
            ) from reason
        failure = _failure(f"{arm.lower()}_{phase}", reason)

    final_boundary, final_boundary_failure = _final_boundary(
        environment,
        previous_solar_kw=previous_solar_kw,
    )
    if final_boundary_failure is not None:
        if failure is None:
            failure = final_boundary_failure
        else:
            failure["final_boundary_capture_failure"] = final_boundary_failure
    if failure is None and (not terminated or truncated):
        failure = _failure(
            f"{arm.lower()}_completion_validation",
            RuntimeError("episode did not reach its declared terminal boundary"),
        )
    classification = _classify_run(
        environment=environment,
        failure=failure,
        terminated=terminated,
        truncated=truncated,
        minute_boundaries=minute_boundaries,
    )
    run = {
        "format": BETA_CONTROLLER_BENCHMARK_ARM_FORMAT,
        "arm": arm,
        **classification,
        "information_profile": dict(BETA_CONTROLLER_INFORMATION_PROFILES[arm]),
        "execution_door": BETA_EXECUTION_DOORS[arm],
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
            "boundary_ledger": {
                "format": BETA_BOUNDARY_LEDGER_FORMAT,
                "decision_boundaries": decision_boundaries,
                "final_boundary": final_boundary,
            },
            "decisions": decisions,
            "intervals": intervals,
            "executions": executions,
            "decision_count": len(decisions),
            "execution_count": len(executions),
            "physical_minute_count": sum(
                int(execution["application_count"]) for execution in executions
            ),
            "minute_truth_states": minute_boundaries,
        },
        "controller_elapsed_ns": controller_elapsed_ns,
        "dispatcher_elapsed_ns": dispatcher_elapsed_ns,
        "failure": failure,
    }
    return _attach_arm_digest(run)


def _service_mpc_result(
    *,
    environment: PlantCyclicMarketEnvV2Beta,
    controller: ServiceMPCSetpointControllerV1,
    settings: ServiceMPCSetpointSettingsV1,
    manifest_identity: Mapping[str, str],
    initial_vector: NDArray[np.generic],
) -> dict[str, Any]:
    rewards: list[float] = []
    intervals: list[dict[str, Any]] = []
    dispatcher_elapsed: list[tuple[int, ...]] = []
    decision_boundaries: list[dict[str, Any]] = []
    minute_boundaries: list[dict[str, Any]] = [environment.telemetry.to_dict()]
    observation_vector = initial_vector.copy()
    previous_solar_kw: float | None = None
    terminated = truncated = False
    failure: dict[str, Any] | None = None
    phase = "boundary_capture"
    try:
        while not terminated and not truncated:
            boundary = _boundary_record(
                environment,
                observation_vector,
                kind="decision",
                previous_solar_kw=previous_solar_kw,
            )
            decision_boundaries.append(boundary)
            decision_solar_kw = float(
                environment.controller_observation.state.boundary.solar_available_kw
            )
            phase = "controller_compute"
            decision = controller.compute_setpoints(
                environment.controller_observation,
                constraints=environment.constraints,
                config=environment.config,
                commitment_profile=environment.episode_config.electrolyser_commitment,
            )
            phase = "environment_execution"
            observation_vector, reward, terminated, truncated, info = (
                environment.step_setpoints(decision.setpoints)
            )
            previous_solar_kw = decision_solar_kw
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
            interval = _economic_interval(info)
            reward_value = float(reward)
            if not math.isfinite(reward_value):
                raise RuntimeError("M24 reward is non-finite")
            controller.observe_transition(
                environment.controller_observation,
                execution=execution,
            )
            truth_states = environment.last_minute_truth_states
            if (
                truth_states is None
                or len(truth_states) != len(execution.applications) + 1
            ):
                raise RuntimeError("M24 omitted its complete minute truth trace")
            if truth_states[0].to_dict() != minute_boundaries[-1]:
                raise RuntimeError("M24 minute truth-state boundary chain is broken")
            for application, before_state, after_state in zip(
                execution.applications,
                truth_states[:-1],
                truth_states[1:],
                strict=True,
            ):
                if (
                    application.boundary_record != before_state.to_dict()
                    or application.next_boundary_record != after_state.to_dict()
                ):
                    raise RuntimeError(
                        "M24 minute truth trace differs from execution evidence"
                    )
                minute_boundaries.append(after_state.to_dict())
            # The controller reconciliation and the complete interval evidence are
            # one publishable unit. A post-movement failure cannot become a report.
            dispatcher_elapsed.append(tuple(int(value) for value in elapsed))
            rewards.append(reward_value)
            intervals.append(interval)
            phase = "boundary_capture"
    except Exception as reason:
        retained_minutes = sum(len(group) for group in dispatcher_elapsed)
        if (
            environment.decisions_taken != len(intervals)
            or environment.physics_steps_taken != retained_minutes
        ):
            raise RuntimeError(
                "M24 moved without a complete reconciled interval bundle"
            ) from reason
        failure = _failure(f"m24_{phase}", reason)
        if controller.has_pending_transition:
            execution = environment.last_setpoint_execution
            failure["unreconciled_execution"] = (
                None if execution is None else execution.to_dict()
            )

    final_boundary, final_boundary_failure = _final_boundary(
        environment,
        previous_solar_kw=previous_solar_kw,
    )
    if final_boundary_failure is not None:
        if failure is None:
            failure = final_boundary_failure
        else:
            failure["final_boundary_capture_failure"] = final_boundary_failure
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

    classification = _classify_run(
        environment=environment,
        failure=failure,
        terminated=terminated,
        truncated=truncated,
        minute_boundaries=minute_boundaries,
    )
    run = {
        "format": BETA_CONTROLLER_BENCHMARK_ARM_FORMAT,
        "arm": SERVICE_MPC_ARM,
        **classification,
        "information_profile": dict(
            BETA_CONTROLLER_INFORMATION_PROFILES[SERVICE_MPC_ARM]
        ),
        "execution_door": BETA_EXECUTION_DOORS[SERVICE_MPC_ARM],
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
            "boundary_ledger": {
                "format": BETA_BOUNDARY_LEDGER_FORMAT,
                "decision_boundaries": decision_boundaries,
                "final_boundary": final_boundary,
            },
            "intervals": intervals,
            "run": run_evidence,
            "minute_truth_states": minute_boundaries,
        },
        "failure": failure,
        "evidence_validation_failure": evidence_failure,
    }
    return _attach_arm_digest(run)


def _preflight_failed_cell(seed: int, reason: BaseException) -> dict[str, Any]:
    failure = _failure("cell_preflight", reason)
    runs: dict[str, dict[str, Any]] = {}
    for arm in BETA_CONTROLLER_ARM_ORDER:
        runs[arm] = _attach_arm_digest(
            {
                "format": BETA_CONTROLLER_BENCHMARK_ARM_FORMAT,
                "arm": arm,
                "status": "not_run_preflight_failed",
                "execution_complete": False,
                "cycle_feasible": False,
                "hard_safety_passed": False,
                "value_comparison_candidate": False,
                "value_comparison_eligible": False,
                "exclusion_reasons": ["cell_preflight_failed"],
                "information_profile": dict(BETA_CONTROLLER_INFORMATION_PROFILES[arm]),
                "execution_door": BETA_EXECUTION_DOORS[arm],
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
                "cycle_settlements": [],
                "evidence": {
                    "boundary_ledger": {
                        "format": BETA_BOUNDARY_LEDGER_FORMAT,
                        "decision_boundaries": [],
                        "final_boundary": None,
                    },
                    "decisions": [],
                    "executions": [],
                    "minute_truth_states": [],
                },
                "failure": dict(failure),
            }
        )
    return _attach_cell_digest(
        {
            "format": BETA_CONTROLLER_BENCHMARK_CELL_FORMAT,
            "seed": seed,
            "status": "preflight_failed",
            "execution_complete": False,
            "cycle_feasible": False,
            "hard_safety_passed": False,
            "service_comparable": False,
            "value_comparison_eligible": False,
            "exclusion_reasons": ["cell_preflight_failed"],
            "manifest_identity": None,
            "audit_manifest": None,
            "arm_order": list(BETA_CONTROLLER_ARM_ORDER),
            "runs": runs,
            "failure": failure,
        }
    )


def _metric_number(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"complete beta arm has invalid numeric metric {key}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"complete beta arm has non-finite metric {key}")
    return number


def _cell_classification(runs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    execution_complete = all(
        runs[arm].get("execution_complete") is True for arm in BETA_CONTROLLER_ARM_ORDER
    )
    cycle_feasible = all(
        runs[arm].get("cycle_feasible") is True for arm in BETA_CONTROLLER_ARM_ORDER
    )
    hard_safety_passed = all(
        runs[arm].get("hard_safety_passed") is True for arm in BETA_CONTROLLER_ARM_ORDER
    )
    service_comparable = False
    service_comparison: dict[str, Any] = {
        "tolerance_kg": BETA_SERVICE_COMPARISON_TOLERANCE_KG,
        "delivered_spread_kg": None,
        "unmet_spread_kg": None,
    }
    if execution_complete:
        delivered = []
        unmet = []
        requested = []
        for arm in BETA_CONTROLLER_ARM_ORDER:
            physical = _require_mapping(
                f"{arm} physical_metrics", runs[arm].get("physical_metrics")
            )
            delivered.append(_metric_number(physical, "delivered_kg"))
            unmet.append(_metric_number(physical, "unmet_kg"))
            requested.append(_metric_number(physical, "requested_kg"))
        delivered_spread = max(delivered) - min(delivered)
        unmet_spread = max(unmet) - min(unmet)
        requested_spread = max(requested) - min(requested)
        service_comparable = bool(
            delivered_spread <= BETA_SERVICE_COMPARISON_TOLERANCE_KG
            and unmet_spread <= BETA_SERVICE_COMPARISON_TOLERANCE_KG
            and requested_spread <= BETA_SERVICE_COMPARISON_TOLERANCE_KG
        )
        service_comparison.update(
            {
                "delivered_spread_kg": delivered_spread,
                "unmet_spread_kg": unmet_spread,
                "requested_spread_kg": requested_spread,
            }
        )
    eligible = bool(
        execution_complete
        and cycle_feasible
        and hard_safety_passed
        and service_comparable
    )
    exclusions: list[str] = []
    for arm in BETA_CONTROLLER_ARM_ORDER:
        if runs[arm].get("execution_complete") is not True:
            exclusions.append(f"arm_execution_incomplete:{arm}")
        if runs[arm].get("cycle_feasible") is not True:
            exclusions.append(f"arm_cycle_infeasible_or_unsettled:{arm}")
        if runs[arm].get("hard_safety_passed") is not True:
            exclusions.append(f"arm_hard_safety_failed:{arm}")
    if not service_comparable:
        exclusions.append("service_not_comparable")
    return {
        "status": "value_comparison_eligible" if eligible else "ineligible",
        "execution_complete": execution_complete,
        "cycle_feasible": cycle_feasible,
        "hard_safety_passed": hard_safety_passed,
        "service_comparable": service_comparable,
        "service_comparison": service_comparison,
        "value_comparison_eligible": eligible,
        "exclusion_reasons": exclusions,
    }


def execute_beta_controller_benchmark_cell(
    loaded: LoadedPolicy,
    config: ControllerNeutralTrainingConfigV1,
    *,
    seed: int,
    service_settings: ServiceMPCSetpointSettingsV1 | None = None,
    economic_controller_factory: BetaEconomicControllerFactory | None = None,
) -> dict[str, Any]:
    """Execute one paired beta cell after a zero-movement fail-closed preflight."""

    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    _assert_beta_task(config)
    require_legacy_evaluation_scope(
        config.environment_id, training_config=config.to_dict(), seeds=(int(seed),)
    )
    if isinstance(loaded, LoadedPolicy):
        require_legacy_evaluation_scope(
            loaded.metadata.environment_id,
            training_config=loaded.metadata.training_config,
            seeds=(int(seed),),
        )
    settings = _service_settings(config, service_settings)
    factory = economic_controller_factory or _real_beta_economic_controller
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
        # Use the admitted policy promptly; output ordering remains frozen below.
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
            initial_vector=prepared.policy_observations[SERVICE_MPC_ARM],
        )
        runs[BETA_ECONOMIC_MPC_ARM] = _generic_arm_result(
            arm=BETA_ECONOMIC_MPC_ARM,
            environment=prepared.environments[BETA_ECONOMIC_MPC_ARM],
            controller=cast(
                _CommandController, prepared.controllers[BETA_ECONOMIC_MPC_ARM]
            ),
            manifest_identity=prepared.manifest_identity,
            initial_vector=prepared.policy_observations[BETA_ECONOMIC_MPC_ARM],
        )
        ordered_runs: dict[str, Mapping[str, Any]] = {
            arm: runs[arm] for arm in BETA_CONTROLLER_ARM_ORDER
        }
        classification = _cell_classification(ordered_runs)
        for run in runs.values():
            run["value_comparison_eligible"] = bool(
                classification["value_comparison_eligible"]
                and run.get("value_comparison_candidate") is True
            )
            if not classification["value_comparison_eligible"]:
                exclusions = cast(list[str], run["exclusion_reasons"])
                if "paired_cell_ineligible" not in exclusions:
                    exclusions.append("paired_cell_ineligible")
            _attach_arm_digest(run)
        return _attach_cell_digest(
            {
                "format": BETA_CONTROLLER_BENCHMARK_CELL_FORMAT,
                "seed": int(seed),
                **classification,
                "manifest_identity": dict(prepared.manifest_identity),
                "audit_manifest": prepared.audit_manifest,
                "arm_order": list(BETA_CONTROLLER_ARM_ORDER),
                "runs": dict(ordered_runs),
                "failure": None,
            }
        )
    finally:
        prepared.close()


def aggregate_value_eligible_cells(
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate value only across eligible complete four-way paired cells."""

    eligible = [cell for cell in cells if cell.get("value_comparison_eligible") is True]
    excluded: list[dict[str, Any]] = []
    for cell in cells:
        if cell.get("value_comparison_eligible") is True:
            continue
        raw_reasons = cell.get("exclusion_reasons")
        reasons = (
            [str(reason) for reason in raw_reasons]
            if isinstance(raw_reasons, list)
            else ["missing_eligibility_evidence"]
        )
        excluded.append(
            {
                "seed": cell.get("seed"),
                "cell_status": cell.get("status"),
                "exclusion_reasons": reasons,
            }
        )

    arms: dict[str, dict[str, Any]] = {}
    for arm in BETA_CONTROLLER_ARM_ORDER:
        all_status: dict[str, int] = {}
        execution_count = cycle_count = safety_count = 0
        for cell in cells:
            raw_runs = cell.get("runs")
            runs = raw_runs if isinstance(raw_runs, Mapping) else {}
            run = runs.get(arm)
            if not isinstance(run, Mapping):
                status = "missing"
                all_status[status] = all_status.get(status, 0) + 1
                continue
            status = str(run.get("status", "missing"))
            all_status[status] = all_status.get(status, 0) + 1
            execution_count += int(run.get("execution_complete") is True)
            cycle_count += int(run.get("cycle_feasible") is True)
            safety_count += int(run.get("hard_safety_passed") is True)

        rewards: list[float] = []
        values: list[float] = []
        delivered: list[float] = []
        unmet: list[float] = []
        for cell in eligible:
            run = cast(Mapping[str, Any], cast(Mapping[str, Any], cell["runs"])[arm])
            physical = cast(Mapping[str, Any], run["physical_metrics"])
            economic = cast(Mapping[str, Any], run["economic_metrics"])
            rewards.append(_metric_number(run, "reward_trace_sum_eur"))
            values.append(_metric_number(economic, "net_operating_value_eur"))
            delivered.append(_metric_number(physical, "delivered_kg"))
            unmet.append(_metric_number(physical, "unmet_kg"))
        count = len(rewards)
        arms[arm] = {
            "registered_cell_count": len(cells),
            "execution_complete_count": execution_count,
            "cycle_feasible_count": cycle_count,
            "hard_safety_passed_count": safety_count,
            "status_counts": all_status,
            "value_eligible_cell_count": count,
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
        "aggregation_policy": (
            "value metrics use only complete four-arm cells that pass cycle, hard "
            "safety, and service-comparability gates; no fallback or partial pooling"
        ),
        "registered_cell_count": len(cells),
        "preflight_failed_cell_count": sum(
            cell.get("status") == "preflight_failed" for cell in cells
        ),
        "execution_complete_cell_count": sum(
            cell.get("execution_complete") is True for cell in cells
        ),
        "cycle_feasible_cell_count": sum(
            cell.get("cycle_feasible") is True for cell in cells
        ),
        "hard_safety_passed_cell_count": sum(
            cell.get("hard_safety_passed") is True for cell in cells
        ),
        "service_comparable_cell_count": sum(
            cell.get("service_comparable") is True for cell in cells
        ),
        "value_eligible_cell_count": len(eligible),
        "excluded_cell_count": len(excluded),
        "value_eligible_seeds": [int(cell["seed"]) for cell in eligible],
        "excluded_cells": excluded,
        "arms": arms,
    }


def evaluate_beta_controller_benchmark(
    loaded: LoadedPolicy,
    config: ControllerNeutralTrainingConfigV1,
    *,
    seeds: Sequence[int],
    service_settings: ServiceMPCSetpointSettingsV1 | None = None,
    economic_controller_factory: BetaEconomicControllerFactory | None = None,
    require_trained_policy: bool = True,
) -> dict[str, Any]:
    """Run the bounded production beta comparison without hiding failed arms."""

    seed_values = _validated_seeds(seeds)
    _assert_beta_task(config)
    require_legacy_evaluation_scope(
        config.environment_id, training_config=config.to_dict(), seeds=seed_values
    )
    canonical_settings = _service_settings(config, None)
    settings = _service_settings(config, service_settings)
    if require_trained_policy and settings != canonical_settings:
        raise ValueError(
            "a qualified beta benchmark requires the canonical rolling M24 settings"
        )
    if require_trained_policy and economic_controller_factory is not None:
        raise ValueError(
            "a qualified beta benchmark cannot replace the production E-beta arm"
        )
    readmitted, checkpoint_identity = _readmit_policy(
        loaded,
        config,
        require_trained_policy=require_trained_policy,
    )
    provenance = _source_provenance(require_clean=require_trained_policy)
    cells = [
        execute_beta_controller_benchmark_cell(
            readmitted,
            config,
            seed=seed,
            service_settings=settings,
            economic_controller_factory=economic_controller_factory,
        )
        for seed in seed_values
    ]
    completed_provenance = _source_provenance(require_clean=require_trained_policy)
    if completed_provenance != provenance:
        raise RuntimeError(
            "benchmark source or numerical runtime changed during execution"
        )
    body: dict[str, Any] = {
        "format": BETA_CONTROLLER_BENCHMARK_FORMAT,
        "claim_boundary": BETA_CONTROLLER_BENCHMARK_CLAIM_BOUNDARY,
        "policy_qualification": (
            "trained-sb3-ppo"
            if readmitted.metadata.trained
            else "integration-fixture-not-a-learned-controller"
        ),
        "benchmark_qualification": (
            "canonical-four-arm-beta-development-benchmark"
            if require_trained_policy
            else "unqualified-integration-fixture-or-custom-arm"
        ),
        "training_task": config.to_dict(),
        "checkpoint": checkpoint_identity,
        "source_provenance": provenance,
        "seeds": list(seed_values),
        "arm_order": list(BETA_CONTROLLER_ARM_ORDER),
        "information_profiles": {
            arm: dict(BETA_CONTROLLER_INFORMATION_PROFILES[arm])
            for arm in BETA_CONTROLLER_ARM_ORDER
        },
        "execution_doors": dict(BETA_EXECUTION_DOORS),
        "service_mpc_settings": settings.to_dict(),
        "cells": cells,
        "aggregate": aggregate_value_eligible_cells(cells),
    }
    body["deterministic_evidence_sha256"] = domain_sha256(
        "cyclic-market-controller-benchmark-evidence-v1",
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
            "information_profiles": body["information_profiles"],
            "execution_doors": body["execution_doors"],
            "service_mpc_settings": body["service_mpc_settings"],
            "cell_evidence": [cell["deterministic_evidence_sha256"] for cell in cells],
            "aggregate": body["aggregate"],
        },
    )
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare H, M24, cyclic-market economic MPC, and a trained PPO on "
            "identical checkpoint-bound beta episodes."
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
        if arguments.count > BETA_CONTROLLER_BENCHMARK_MAX_SEEDS:
            raise ValueError(
                f"count cannot exceed {BETA_CONTROLLER_BENCHMARK_MAX_SEEDS}"
            )
        if not arguments.json.is_absolute():
            raise ValueError("beta benchmark output must be an absolute new path")
        resolved_output = arguments.json.resolve()
        repository = _repository_root()
        if resolved_output == repository or repository in resolved_output.parents:
            raise ValueError("beta benchmark output must be outside the repository")
        if arguments.json.exists():
            raise ValueError(
                f"beta benchmark output exists at {arguments.json}; choose a new path"
            )
        loaded, config = load_beta_benchmark_policy(arguments.checkpoint)
        result = evaluate_beta_controller_benchmark(
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
        print(f"beta controller benchmark refused: {reason}", file=sys.stderr)
        return 2

    summary = {
        "format": result["format"],
        "output": str(arguments.json),
        "seeds": result["seeds"],
        "value_eligible_cells": result["aggregate"]["value_eligible_cell_count"],
        "excluded_cells": result["aggregate"]["excluded_cell_count"],
        "deterministic_evidence_sha256": result["deterministic_evidence_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["aggregate"]["excluded_cell_count"] == 0 else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BETA_CONTROLLER_ARM_ORDER",
    "BETA_CONTROLLER_BENCHMARK_CLAIM_BOUNDARY",
    "BETA_CONTROLLER_BENCHMARK_FORMAT",
    "BETA_CONTROLLER_BENCHMARK_MAX_SEEDS",
    "BETA_CONTROLLER_INFORMATION_PROFILES",
    "BETA_ECONOMIC_MPC_ARM",
    "BetaEconomicControllerFactory",
    "aggregate_value_eligible_cells",
    "evaluate_beta_controller_benchmark",
    "execute_beta_controller_benchmark_cell",
    "load_beta_benchmark_policy",
    "main",
]

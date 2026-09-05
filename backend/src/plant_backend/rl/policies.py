"""Loading a checkpoint into something callable, with no room for a fake.

    loaded = load_policy("checkpoints/ppo-demo")
    action = loaded.policy.act(observation)

`Policy` is one method wide on purpose: inference is all the running plant needs
from a learned controller, so the backend never imports a training library to
drive the plant, and a test can pass a two-line stub where a network would be.

The training library is imported *inside* the loader, not at module scope. That
keeps Stable-Baselines3 and Torch an optional install: the backend, the tests and
the simulator run without them, and asking for an `sb3-ppo` checkpoint without
them installed produces a message naming the extra to install rather than an
ImportError from three frames deep.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from . import checkpoints
from .checkpoints import CheckpointError, CheckpointMetadata

if TYPE_CHECKING:
    from .v2.training.runtime_provenance import BetaTrainingRuntimeProvenanceV1


@runtime_checkable
class Policy(Protocol):
    """Observation in, action out. Deterministic: the plant is not exploring."""

    def act(self, observation: NDArray[np.generic]) -> NDArray[np.float32]: ...


@dataclass(frozen=True)
class LoadedPolicy:
    """A callable policy plus the metadata that vouched for it."""

    policy: Policy
    metadata: CheckpointMetadata
    path: Path

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "metadata": self.metadata.to_dict()}


class FixedActionPolicy:
    """Returns the same action every step. A stand-in, never a trained policy."""

    def __init__(self, action: NDArray[np.generic] | list[float]) -> None:
        self._action = cast(
            NDArray[np.float32],
            np.asarray(action, dtype=np.float32).reshape(-1),
        )

    def act(self, observation: NDArray[np.generic]) -> NDArray[np.float32]:
        del observation
        return cast(NDArray[np.float32], self._action.copy())


class SB3Policy:
    """Deterministic inference from a Stable-Baselines3 model."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def act(self, observation: NDArray[np.generic]) -> NDArray[np.float32]:
        action, _ = self._model.predict(observation, deterministic=True)
        return cast(
            NDArray[np.float32],
            np.asarray(action, dtype=np.float32).reshape(-1),
        )


def _capture_metadata_snapshot(
    directory: Path,
    *,
    contract: checkpoints.PolicyEnvironmentContract | None,
) -> tuple[CheckpointMetadata, bytes]:
    """Parse and bind stable metadata bytes plus their referenced artifact."""

    metadata = checkpoints.load_metadata(directory, contract=contract)
    try:
        snapshot = checkpoints.metadata_path(directory).read_bytes()
    except OSError as reason:
        raise checkpoints.CorruptCheckpointError(
            f"cannot bind checkpoint metadata at {directory}: {reason}"
        ) from reason
    reopened = checkpoints.load_metadata(directory, contract=contract)
    try:
        reopened_bytes = checkpoints.metadata_path(directory).read_bytes()
    except OSError as reason:
        raise checkpoints.CorruptCheckpointError(
            f"cannot re-bind checkpoint metadata at {directory}: {reason}"
        ) from reason
    if reopened != metadata or reopened_bytes != snapshot:
        raise checkpoints.CorruptCheckpointError(
            f"checkpoint metadata changed during admission ({directory})"
        )
    return reopened, snapshot


def _require_metadata_snapshot_unchanged(
    directory: Path,
    *,
    contract: checkpoints.PolicyEnvironmentContract | None,
    expected_metadata: CheckpointMetadata,
    expected_bytes: bytes,
) -> None:
    reopened, reopened_bytes = _capture_metadata_snapshot(
        directory,
        contract=contract,
    )
    if reopened != expected_metadata or reopened_bytes != expected_bytes:
        raise checkpoints.CorruptCheckpointError(
            f"checkpoint metadata changed while loading ({directory})"
        )


def load_policy(
    path: Path | str,
    *,
    contract: checkpoints.PolicyEnvironmentContract | None = None,
    environment: object | None = None,
    expected_beta_runtime_provenance: BetaTrainingRuntimeProvenanceV1 | None = None,
    require_registered_evaluation_eligibility: bool = False,
) -> LoadedPolicy:
    """Load a checkpoint for inference, or raise `CheckpointError` saying why not.

    Supplying the admitted environment additionally checks the concrete Gym
    spaces embedded in a real model artifact.  V2 callers always provide it;
    legacy callers retain their historical metadata-only admission behavior.
    """
    directory = Path(path)
    if not isinstance(require_registered_evaluation_eligibility, bool):
        raise TypeError("require_registered_evaluation_eligibility must be a boolean")
    if environment is not None and contract is None:
        raise ValueError("an admitted environment requires an explicit policy contract")
    metadata, admitted_metadata_bytes = _capture_metadata_snapshot(
        directory,
        contract=contract,
    )
    from .v2.training.config import (
        BETA_CORE_ALGORITHM_IDS,
        ControllerNeutralTrainingConfigV1,
    )

    if metadata.format == checkpoints.FORMAT_FIXED:
        if (
            metadata.algorithm in BETA_CORE_ALGORITHM_IDS
            or metadata.training_config.get("beta_rl") is not None
        ):
            raise checkpoints.CorruptCheckpointError(
                "a fixed policy cannot claim constrained beta training provenance"
            )
        if expected_beta_runtime_provenance is not None:
            raise ValueError("a fixed policy cannot receive beta runtime provenance")
        if require_registered_evaluation_eligibility:
            raise checkpoints.IncompatibleCheckpointError(
                "a fixed policy is not eligible for registered beta evaluation"
            )
        action = metadata.extra.get("action")
        if not isinstance(action, list) or len(action) != metadata.action_size:
            raise checkpoints.CorruptCheckpointError(
                f"{directory}: a fixed checkpoint needs an 'action' of "
                f"{metadata.action_size} numbers in its metadata"
            )
        _require_metadata_snapshot_unchanged(
            directory,
            contract=contract,
            expected_metadata=metadata,
            expected_bytes=admitted_metadata_bytes,
        )
        return LoadedPolicy(FixedActionPolicy(action), metadata, directory)

    artefact = checkpoints.policy_path(directory)

    # A beta constrained-RL checkpoint has stronger semantics than its shared
    # SB3 container format.  Parse and externally bind metadata before
    # deserialisation, validate the artifact's embedded state after load, and
    # only then admit resume lineage or registered-evaluation eligibility.
    configured_beta = metadata.training_config.get("beta_rl") is not None
    declared_beta = metadata.algorithm in BETA_CORE_ALGORITHM_IDS
    if configured_beta != declared_beta:
        raise checkpoints.CorruptCheckpointError(
            "beta algorithm and beta_rl training provenance must be present together"
        )
    if configured_beta:
        try:
            training_config = ControllerNeutralTrainingConfigV1.from_dict(
                metadata.training_config
            )
        except (TypeError, ValueError) as reason:
            raise checkpoints.CorruptCheckpointError(
                f"{directory} has invalid beta training provenance: {reason}"
            ) from reason
        if training_config.beta_rl is None:  # pragma: no cover - raw gate narrows this
            raise checkpoints.CorruptCheckpointError(
                "beta checkpoint lost its beta_rl training provenance"
            )
        if environment is None or contract is None:
            raise checkpoints.IncompatibleCheckpointError(
                "beta constrained-RL loading requires its admitted environment"
            )
        from .v2.training.beta_checkpoint import (
            admit_beta_checkpoint_evidence,
            parse_beta_checkpoint_evidence,
        )
        from .v2.training.constrained_ppo import (
            ConstrainedPPO,
            training_dependencies_available,
        )
        from .v2.training.ppo import (
            BETA_CHECKPOINT_EXTRA_KEY,
            admit_resume,
            expected_beta_checkpoint_timesteps,
            policy_contract,
            verify_beta_allocator_evidence,
            verify_beta_model_configuration,
        )
        from .v2.training.runtime_provenance import (
            capture as capture_runtime_provenance,
        )
        from .v2.training.runtime_provenance import (
            verify_unchanged as verify_runtime_unchanged,
        )

        run = training_config.beta_rl
        metadata.check_compatible(
            path=directory,
            contract=policy_contract(training_config),
        )
        if metadata.algorithm != run.algorithm:
            raise checkpoints.CorruptCheckpointError(
                "beta checkpoint algorithm differs from its training contract"
            )
        if metadata.seed != training_config.seed:
            raise checkpoints.CorruptCheckpointError(
                "beta checkpoint seed differs from its training contract"
            )
        if set(metadata.extra) != {BETA_CHECKPOINT_EXTRA_KEY}:
            raise checkpoints.CorruptCheckpointError(
                "beta checkpoint metadata has unknown or missing evidence fields"
            )
        if not training_dependencies_available():
            raise CheckpointError(
                "constrained PPO loading requires the optional training dependencies "
                "(`uv sync --group train`)"
            )
        try:
            parsed = parse_beta_checkpoint_evidence(
                metadata.extra[BETA_CHECKPOINT_EXTRA_KEY]
            )
            current_runtime = capture_runtime_provenance(
                require_clean=run.run_kind == "registered"
            )
            if expected_beta_runtime_provenance is None:
                runtime = current_runtime
            else:
                verify_runtime_unchanged(
                    expected_beta_runtime_provenance,
                    current_runtime,
                )
                runtime = expected_beta_runtime_provenance
            parent_identity = (
                admit_resume(training_config, runtime_provenance=runtime)
                if training_config.resume_from is not None
                else None
            )
            expected_parent = (
                None
                if parent_identity is None
                else parent_identity.beta_parent_identity()
            )
            expected_total_timesteps = expected_beta_checkpoint_timesteps(
                training_config,
                parent_total_timesteps=(
                    0 if parent_identity is None else parent_identity.total_timesteps
                ),
            )
            if metadata.total_timesteps != expected_total_timesteps:
                raise ValueError(
                    "beta checkpoint cumulative timestep cut is not explained by "
                    "its training config and parent"
                )
            for label, actual, expected in (
                ("artifact", parsed.artifact_sha256, metadata.policy_artifact_sha256),
                ("algorithm", parsed.algorithm_id, run.algorithm),
                ("run kind", parsed.run_kind, run.run_kind),
                ("optimizer seed", parsed.optimizer_seed, training_config.seed),
                ("timesteps", parsed.total_timesteps, metadata.total_timesteps),
                ("runtime/source identity", parsed.runtime_provenance, runtime),
                ("resume parent", parsed.resume_parent, expected_parent),
            ):
                if actual != expected:
                    raise ValueError(
                        f"beta checkpoint {label} differs from external custody"
                    )
            verify_beta_allocator_evidence(
                parsed,
                training_config,
                path=directory,
            )
            beta_model = ConstrainedPPO.load(artefact, device="cpu")
            observation_space = getattr(environment, "observation_space", None)
            action_space = getattr(environment, "action_space", None)
            if observation_space is None or action_space is None:
                raise ValueError(
                    "the admitted environment must expose observation_space and action_space"
                )
            checkpoints.verify_policy_model_spaces(
                beta_model,
                observation_space=observation_space,
                action_space=action_space,
                path=directory,
            )
            verify_beta_model_configuration(
                beta_model,
                training_config,
                path=directory,
            )
            artifact_constraint_state = beta_model.embedded_constraint_state
            if artifact_constraint_state != parsed.constraint_state:
                raise ValueError(
                    "artifact-loaded constraint state differs from checkpoint evidence"
                )
            artifact_attestation = beta_model.loaded_constraint_state_attestation()
            parent_attestation = None
            if parent_identity is not None:
                parent_model = ConstrainedPPO.load(
                    checkpoints.policy_path(parent_identity.path),
                    device="cpu",
                )
                parent_attestation = parent_model.loaded_constraint_state_attestation()
            admit_beta_checkpoint_evidence(
                parsed.to_dict(),
                expected_artifact_sha256=metadata.policy_artifact_sha256,
                expected_algorithm_id=run.algorithm,
                expected_run_kind=run.run_kind,
                expected_optimizer_seed=training_config.seed,
                expected_resume_parent=expected_parent,
                expected_runtime_provenance=runtime,
                expected_total_timesteps=metadata.total_timesteps,
                expected_artifact_attestation=artifact_attestation,
                expected_resume_parent_evidence=(
                    None
                    if parent_identity is None or parent_identity.beta_evidence is None
                    else parent_identity.beta_evidence.to_dict()
                ),
                expected_parent_artifact_attestation=parent_attestation,
                require_registered_evaluation_eligibility=(
                    require_registered_evaluation_eligibility
                ),
            )
            runtime_after = capture_runtime_provenance(
                require_clean=run.run_kind == "registered"
            )
            verify_runtime_unchanged(runtime, runtime_after)
            _require_metadata_snapshot_unchanged(
                directory,
                contract=contract,
                expected_metadata=metadata,
                expected_bytes=admitted_metadata_bytes,
            )
        except checkpoints.CheckpointError:
            raise
        except (
            ImportError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as reason:
            raise checkpoints.CorruptCheckpointError(
                f"cannot admit beta constrained-RL checkpoint {directory}: {reason}"
            ) from reason
        return LoadedPolicy(SB3Policy(beta_model), metadata, directory)

    if require_registered_evaluation_eligibility:
        raise checkpoints.IncompatibleCheckpointError(
            "checkpoint is not a registered constrained-RL beta policy"
        )
    if expected_beta_runtime_provenance is not None:
        raise ValueError("a non-beta checkpoint cannot receive beta runtime provenance")
    try:
        from stable_baselines3 import PPO
    except ImportError as reason:  # pragma: no cover - depends on the optional extra
        raise CheckpointError(
            f"{directory} is a {metadata.format} checkpoint, but stable-baselines3 is not "
            "installed: install the optional training dependencies "
            "(`uv sync --group train`) to load it"
        ) from reason
    try:
        model = PPO.load(artefact, device="cpu")
    except Exception as reason:  # pragma: no cover - depends on the optional extra
        raise checkpoints.CorruptCheckpointError(
            f"cannot load {artefact}: {reason}"
        ) from reason
    if getattr(model, "algorithm_id", None) in BETA_CORE_ALGORITHM_IDS or any(
        getattr(model, name, None) is not None
        for name in (
            "_embedded_constraint_state_schema",
            "_embedded_constraint_state_payload_json",
            "_embedded_constraint_state_sha256",
        )
    ):
        raise checkpoints.CorruptCheckpointError(
            "beta constrained artifact was downgraded to generic PPO metadata"
        )
    if environment is not None:
        observation_space = getattr(environment, "observation_space", None)
        action_space = getattr(environment, "action_space", None)
        if observation_space is None or action_space is None:
            raise ValueError(
                "the admitted environment must expose observation_space and action_space"
            )
        checkpoints.verify_policy_model_spaces(
            model,
            observation_space=observation_space,
            action_space=action_space,
            path=directory,
        )
    _require_metadata_snapshot_unchanged(
        directory,
        contract=contract,
        expected_metadata=metadata,
        expected_bytes=admitted_metadata_bytes,
    )
    return LoadedPolicy(SB3Policy(model), metadata, directory)


__all__ = ["FixedActionPolicy", "LoadedPolicy", "Policy", "SB3Policy", "load_policy"]

"""Stable-Baselines3 wiring for controller-neutral Environment V2 policies."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import math
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Mapping, cast

from ... import checkpoints
from .. import beta_spaces, economic_spaces, setpoint_spaces
from ..beta_env import PlantCyclicMarketEnvV2Beta
from ..economic_env import PlantEconomicSetpointEnvV2
from ..policy_setpoint_controller import (
    SetpointPolicyEnvironmentV1,
    policy_contract_for_environment,
)
from ..setpoint_env import PlantSetpointEnvV2
from .beta_checkpoint import (
    BetaRLCheckpointEvidenceV1,
    admit_beta_checkpoint_evidence,
    build_beta_checkpoint_evidence,
    build_beta_resume_parent_identity,
    parse_beta_checkpoint_evidence,
)
from .config import (
    BETA_CORE_ALGORITHM_IDS,
    BETA_FIXTURE_RUN_KIND,
    BETA_REGISTERED_RUN_KIND,
    ControllerNeutralTrainingConfigV1,
)
from .episode_allocator import (
    EpisodeAllocatorModeV1,
    EpisodeSeedAllocatorV1,
    EpisodeSeedSplitV1,
    find_seed_allocator,
)
from .runtime_provenance import (
    BetaTrainingRuntimeProvenanceV1,
    capture as capture_runtime_provenance,
    verify_unchanged as verify_runtime_unchanged,
)


CONFIG_FILE = "training_config.json"
RESUME_PARENT_SCHEMA = "controller-neutral-rl-resume-parent-v1"
BETA_CHECKPOINT_EXTRA_KEY = "beta_rl_checkpoint_evidence"


def _reject_beta_artifact_downgrade(model: object) -> None:
    """Refuse a constrained core presented through the legacy PPO door."""

    if getattr(model, "algorithm_id", None) in BETA_CORE_ALGORITHM_IDS or any(
        getattr(model, name, None) is not None
        for name in (
            "_embedded_constraint_state_schema",
            "_embedded_constraint_state_payload_json",
            "_embedded_constraint_state_sha256",
        )
    ):
        raise checkpoints.CorruptCheckpointError(
            "beta constrained artifact cannot be relabelled as generic PPO"
        )


@dataclass(frozen=True, slots=True)
class ResumeCheckpointIdentityV1:
    """Immutable identity of the checkpoint from which one run resumed."""

    path: str
    metadata_sha256: str
    policy_artifact_sha256: str
    checkpoint_contract_digest: str
    total_timesteps: int
    beta_evidence: BetaRLCheckpointEvidenceV1 | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RESUME_PARENT_SCHEMA,
            "path": self.path,
            "metadata_sha256": self.metadata_sha256,
            "policy_artifact_sha256": self.policy_artifact_sha256,
            "checkpoint_contract_digest": self.checkpoint_contract_digest,
            "total_timesteps": self.total_timesteps,
        }

    def beta_parent_identity(self) -> dict[str, Any]:
        """Return the portable parent identity embedded in a beta child."""

        evidence = self.beta_evidence
        if evidence is None:
            raise ValueError("a legacy PPO checkpoint has no beta parent identity")
        identity: dict[str, Any] = build_beta_resume_parent_identity(
            metadata_sha256=self.metadata_sha256,
            policy_artifact_sha256=self.policy_artifact_sha256,
            checkpoint_contract_digest=self.checkpoint_contract_digest,
            checkpoint_evidence_sha256=evidence.sha256,
            allocator_snapshot_sha256=evidence.allocator_snapshot["sha256"],
            constraint_state_sha256=evidence.constraint_state["sha256"],
            total_timesteps=self.total_timesteps,
        )
        return identity


def make_env(
    config: ControllerNeutralTrainingConfigV1,
    *,
    index: int = 0,
) -> SetpointPolicyEnvironmentV1:
    """Construct one independent task; seeding occurs at reset, not construction."""

    del index
    config.validate()
    if config.environment_id == beta_spaces.ENVIRONMENT_ID:
        if config.beta_task is None:  # pragma: no cover - config.validate closes this
            raise RuntimeError("validated beta training task omitted beta_task")
        return PlantCyclicMarketEnvV2Beta(
            config=config.plant,
            episode=config.episode,
            constraints=config.constraints,
            weights=config.diagnostic_reward,
            economic_assumptions=config.economic_assumptions,
            beta_task=config.beta_task,
        )
    if config.environment_id == economic_spaces.ENVIRONMENT_ID:
        return PlantEconomicSetpointEnvV2(
            config=config.plant,
            episode=config.episode,
            constraints=config.constraints,
            weights=config.diagnostic_reward,
            economic_assumptions=config.economic_assumptions,
        )
    if config.environment_id == setpoint_spaces.ENVIRONMENT_ID:
        return PlantSetpointEnvV2(
            config=config.plant,
            episode=config.episode,
            constraints=config.constraints,
            weights=config.diagnostic_reward,
        )
    raise ValueError("unsupported controller-neutral training environment")


def policy_contract(
    config: ControllerNeutralTrainingConfigV1,
) -> checkpoints.PolicyEnvironmentContract:
    environment = make_env(config)
    try:
        return cast(
            checkpoints.PolicyEnvironmentContract,
            policy_contract_for_environment(environment),
        )
    finally:
        environment.close()


def build_episode_allocator(
    config: ControllerNeutralTrainingConfigV1,
) -> EpisodeSeedAllocatorV1:
    """Construct fresh episode custody for an admitted beta-RL run.

    Restored fixture custody is deliberately not accepted here: it must first be
    admitted against checkpoint metadata and is then supplied explicitly to
    :func:`build_vec_env`.
    """

    config.validate()
    run = config.beta_rl
    if run is None:
        raise ValueError("episode custody is available only for a beta RL run")
    if config.resume_from is not None:
        raise ValueError(
            "a resumed beta RL run requires checkpoint-admitted episode custody"
        )
    if run.run_kind == BETA_REGISTERED_RUN_KIND:
        return EpisodeSeedAllocatorV1.registered()
    if run.run_kind != BETA_FIXTURE_RUN_KIND:  # pragma: no cover - config validates
        raise ValueError("beta RL run kind is not supported")
    return EpisodeSeedAllocatorV1.fixture(
        base_seed=run.episode_seed_base,
        episode_decisions=config.episode.decision_steps(config.plant),
    )


def _validate_episode_allocator(
    config: ControllerNeutralTrainingConfigV1,
    allocator: EpisodeSeedAllocatorV1,
) -> None:
    run = config.beta_rl
    if run is None:
        raise ValueError("an episode allocator cannot be attached to a legacy PPO run")
    expected_mode = (
        EpisodeAllocatorModeV1.REGISTERED
        if run.run_kind == BETA_REGISTERED_RUN_KIND
        else EpisodeAllocatorModeV1.FIXTURE
    )
    expected_split = (
        EpisodeSeedSplitV1.TRAINING
        if run.run_kind == BETA_REGISTERED_RUN_KIND
        else EpisodeSeedSplitV1.FIXTURE
    )
    expected_decisions = config.episode.decision_steps(config.plant)
    if allocator.mode is not expected_mode or allocator.split is not expected_split:
        raise ValueError("episode allocator mode or seed split differs from the run")
    if allocator.base_seed != run.episode_seed_base:
        raise ValueError("episode allocator base seed differs from the run")
    if allocator.episode_decisions != expected_decisions:
        raise ValueError("episode allocator horizon differs from the beta task")


def _require_pristine_episode_allocator(
    allocator: EpisodeSeedAllocatorV1,
) -> None:
    snapshot = allocator.snapshot()
    if allocator.ledger or allocator.decisions_consumed_by_environment != (0, 0, 0, 0):
        raise ValueError("a fresh beta run requires pristine episode custody")
    if any(
        cursor.current_episode_ordinal is not None
        or cursor.decision_offset != 0
        or cursor.next_episode_ordinal != 0
        or not cursor.awaiting_reset
        for cursor in snapshot.cursors
    ):
        raise ValueError("a fresh beta allocator contains a non-pristine cursor")


def build_vec_env(
    config: ControllerNeutralTrainingConfigV1,
    *,
    episode_allocator: EpisodeSeedAllocatorV1 | None = None,
) -> Any:
    """Create synchronous environments with the run's exact seed semantics."""

    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    config.validate()
    if config.beta_rl is None:
        if episode_allocator is not None:
            raise ValueError("legacy PPO cannot receive a beta episode allocator")
    else:
        if episode_allocator is None:
            episode_allocator = build_episode_allocator(config)
        _validate_episode_allocator(config, episode_allocator)
        if config.resume_from is None:
            _require_pristine_episode_allocator(episode_allocator)

    def factory(index: int) -> Callable[[], Any]:
        def build() -> Any:
            raw_environment: Any = make_env(config, index=index)
            if episode_allocator is not None:
                return Monitor(
                    episode_allocator.wrap(
                        raw_environment,
                        environment_index=index,
                    )
                )
            environment: Any = Monitor(raw_environment)
            environment.reset(seed=config.seed + index)
            return environment

        return build

    return DummyVecEnv([factory(index) for index in range(config.n_envs)])


def _resume_ppo_differences(
    parent: ControllerNeutralTrainingConfigV1,
    requested: ControllerNeutralTrainingConfigV1,
) -> tuple[str, ...]:
    """Return settings a loaded PPO model would silently keep from its parent."""

    # ``device`` is the sole supported resume override because PPO.load receives
    # and applies it explicitly.  Architecture and optimisation settings remain
    # embedded in the loaded model and therefore must not be relabelled.
    return tuple(
        item.name
        for item in fields(requested.ppo)
        if item.name != "device"
        and getattr(parent.ppo, item.name) != getattr(requested.ppo, item.name)
    )


def _beta_resume_config_differences(
    parent: ControllerNeutralTrainingConfigV1,
    requested: ControllerNeutralTrainingConfigV1,
) -> tuple[str, ...]:
    """Return beta campaign fields that continuation is forbidden to relabel."""

    mutable_run_fields = {"output_dir", "resume_from", "total_timesteps", "notes"}
    return tuple(
        item.name
        for item in fields(requested)
        if item.name not in mutable_run_fields
        and getattr(parent, item.name) != getattr(requested, item.name)
    )


def _constrained_ppo_kwargs(
    config: ControllerNeutralTrainingConfigV1,
) -> dict[str, Any]:
    settings = config.ppo
    return {
        "policy": settings.policy,
        "learning_rate": settings.learning_rate,
        "n_steps": settings.n_steps,
        "batch_size": settings.batch_size,
        "n_epochs": settings.n_epochs,
        "gamma": settings.gamma,
        "gae_lambda": settings.gae_lambda,
        "clip_range": settings.clip_range,
        "ent_coef": settings.ent_coef,
        "vf_coef": settings.vf_coef,
        "max_grad_norm": settings.max_grad_norm,
        "device": settings.device,
    }


def expected_beta_checkpoint_timesteps(
    config: ControllerNeutralTrainingConfigV1,
    *,
    parent_total_timesteps: int = 0,
) -> int:
    """Return the exact cumulative SB3 rollout cut implied by a beta run."""

    if config.beta_rl is None:
        raise ValueError("beta checkpoint timing requires a beta RL run")
    if (
        isinstance(parent_total_timesteps, bool)
        or not isinstance(parent_total_timesteps, Integral)
        or parent_total_timesteps < 0
    ):
        raise ValueError("parent beta checkpoint timesteps must be non-negative")
    rollout_size = config.ppo.n_steps * config.n_envs
    requested_updates = math.ceil(config.total_timesteps / rollout_size)
    return int(parent_total_timesteps) + requested_updates * rollout_size


def verify_beta_model_configuration(
    model: object,
    config: ControllerNeutralTrainingConfigV1,
    *,
    path: Path | str,
) -> None:
    """Bind a live constrained learner to its externally declared run config."""

    from .constrained_ppo import ConstrainedPPO, ConstrainedPPOSettingsV1

    run = config.beta_rl
    if run is None:
        raise ValueError("beta model verification requires a beta RL run")
    if type(model) is not ConstrainedPPO:
        raise checkpoints.CorruptCheckpointError(
            f"{path}: beta artifact is not the exact ConstrainedPPO core"
        )
    expected_settings = ConstrainedPPOSettingsV1.for_algorithm(
        run.algorithm,
        optimizer_seed=config.seed,
    )
    if model.constraint_settings != expected_settings:
        raise checkpoints.CorruptCheckpointError(
            f"{path}: constrained settings differ from the training config"
        )
    if model.algorithm_id != run.algorithm:
        raise checkpoints.CorruptCheckpointError(
            f"{path}: constrained algorithm differs from the training config"
        )
    if config.ppo.policy != "MlpPolicy":
        raise checkpoints.CorruptCheckpointError(
            f"{path}: beta training config does not declare the frozen MlpPolicy"
        )
    live = model.ppo_configuration
    expected = {
        "learning_rate": config.ppo.learning_rate,
        "n_steps": config.ppo.n_steps,
        "batch_size": config.ppo.batch_size,
        "n_epochs": config.ppo.n_epochs,
        "n_envs": config.n_envs,
        "gamma": config.ppo.gamma,
        "gae_lambda": config.ppo.gae_lambda,
        "clip_range": config.ppo.clip_range,
        "ent_coef": config.ppo.ent_coef,
        "vf_coef": config.ppo.vf_coef,
        "max_grad_norm": config.ppo.max_grad_norm,
        "seed": config.seed,
    }
    differences = tuple(
        name
        for name, expected_value in expected.items()
        if live.get(name) != expected_value
    )
    if differences:
        raise checkpoints.CorruptCheckpointError(
            f"{path}: live PPO configuration differs from training config for "
            + ", ".join(differences)
        )


def verify_beta_allocator_evidence(
    evidence: BetaRLCheckpointEvidenceV1,
    config: ControllerNeutralTrainingConfigV1,
    *,
    path: Path | str,
) -> None:
    """Bind parsed seed-custody evidence to the declared beta task."""

    run = config.beta_rl
    if run is None:
        raise ValueError("beta allocator verification requires a beta RL run")
    identity = evidence.allocator_snapshot.get("allocator_identity")
    if not isinstance(identity, Mapping):
        raise checkpoints.CorruptCheckpointError(
            f"{path}: beta allocator evidence omitted its identity"
        )
    expected_mode = (
        EpisodeAllocatorModeV1.REGISTERED.value
        if run.run_kind == BETA_REGISTERED_RUN_KIND
        else EpisodeAllocatorModeV1.FIXTURE.value
    )
    expected_split = (
        EpisodeSeedSplitV1.TRAINING.value
        if run.run_kind == BETA_REGISTERED_RUN_KIND
        else EpisodeSeedSplitV1.FIXTURE.value
    )
    expected = {
        "mode": expected_mode,
        "split": expected_split,
        "base_seed": run.episode_seed_base,
        "vector_environment_count": config.n_envs,
        "episode_decisions": config.episode.decision_steps(config.plant),
    }
    differences = tuple(
        name
        for name, expected_value in expected.items()
        if identity.get(name) != expected_value
    )
    if differences:
        raise checkpoints.CorruptCheckpointError(
            f"{path}: allocator evidence differs from training config for "
            + ", ".join(differences)
        )


def _verify_beta_model_spaces(
    model: object,
    config: ControllerNeutralTrainingConfigV1,
    *,
    path: Path | str,
) -> None:
    declared_environment = make_env(config)
    try:
        checkpoints.verify_policy_model_spaces(
            model,
            observation_space=declared_environment.observation_space,
            action_space=declared_environment.action_space,
            path=path,
        )
    finally:
        declared_environment.close()


def _load_beta_artifact_evidence(
    directory: Path,
    config: ControllerNeutralTrainingConfigV1,
) -> tuple[dict[str, Any], Any]:
    """Deserialize an outer-admitted beta artifact and validate its live state."""

    from .constrained_ppo import ConstrainedPPO, training_dependencies_available

    if not training_dependencies_available():
        raise checkpoints.CheckpointError(
            "constrained PPO loading requires the optional training dependencies "
            "(`uv sync --group train`)"
        )

    try:
        model = ConstrainedPPO.load(
            checkpoints.policy_path(directory),
            device=config.ppo.device,
        )
        verify_beta_model_configuration(model, config, path=directory)
        _verify_beta_model_spaces(model, config, path=directory)
        constraint_state: dict[str, Any] = model.embedded_constraint_state
        attestation = model.loaded_constraint_state_attestation()
        if attestation.constraint_state != constraint_state:
            raise ValueError("artifact attestation differs from embedded state")
        return constraint_state, attestation
    except checkpoints.CheckpointError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as reason:
        raise checkpoints.CorruptCheckpointError(
            f"cannot validate constrained PPO artifact at {directory}: {reason}"
        ) from reason


def _admit_resume(
    config: ControllerNeutralTrainingConfigV1,
    *,
    runtime_provenance: BetaTrainingRuntimeProvenanceV1 | None,
    visited: frozenset[Path],
) -> ResumeCheckpointIdentityV1 | None:
    """Recursive implementation retaining external beta lineage custody."""

    config.validate()
    if config.resume_from is None:
        return None
    directory = Path(config.resume_from).expanduser().resolve()
    if directory in visited:
        raise checkpoints.CorruptCheckpointError(
            f"beta resume lineage contains a path cycle at {directory}"
        )
    metadata = checkpoints.load_metadata(
        directory,
        contract=policy_contract(config),
    )
    try:
        admitted_metadata_bytes = checkpoints.metadata_path(directory).read_bytes()
    except OSError as reason:
        raise checkpoints.CorruptCheckpointError(
            f"cannot bind resume metadata at {directory}: {reason}"
        ) from reason
    if metadata.format != checkpoints.FORMAT_SB3_PPO:
        raise checkpoints.IncompatibleCheckpointError(
            f"cannot resume from a {metadata.format!r} checkpoint ({directory}): "
            f"expected {checkpoints.FORMAT_SB3_PPO}"
        )
    if not metadata.training_config:
        raise checkpoints.CorruptCheckpointError(
            f"resume checkpoint {directory} has no controller-neutral training task"
        )
    try:
        parent_config = ControllerNeutralTrainingConfigV1.from_dict(
            metadata.training_config
        )
        parent_contract = policy_contract(parent_config)
        metadata.check_compatible(path=directory, contract=parent_contract)
    except (TypeError, ValueError) as reason:
        if isinstance(reason, checkpoints.CheckpointError):
            raise
        raise checkpoints.CorruptCheckpointError(
            f"resume checkpoint {directory} has invalid training provenance: {reason}"
        ) from reason
    if metadata.seed != parent_config.seed:
        raise checkpoints.CorruptCheckpointError(
            f"resume checkpoint {directory} seed disagrees with its training task"
        )

    if config.beta_rl is None:
        if parent_config.beta_rl is not None:
            raise checkpoints.IncompatibleCheckpointError(
                "legacy PPO cannot resume a beta constrained-RL checkpoint"
            )
        differences = _resume_ppo_differences(parent_config, config)
        if differences:
            raise checkpoints.IncompatibleCheckpointError(
                "resume PPO settings differ from the loaded model for "
                f"{', '.join(differences)}; start a fresh run or preserve the parent "
                "settings (device is the only supported resume override)"
            )
        beta_evidence = None
    else:
        if runtime_provenance is None:  # pragma: no cover - public wrapper supplies it
            raise RuntimeError("beta resume admission omitted runtime provenance")
        if parent_config.beta_rl is None:
            raise checkpoints.IncompatibleCheckpointError(
                "beta constrained RL cannot resume a historical PPO checkpoint"
            )
        differences = _beta_resume_config_differences(parent_config, config)
        if differences:
            raise checkpoints.IncompatibleCheckpointError(
                "beta resume campaign identity differs for " + ", ".join(differences)
            )
        run = config.beta_rl
        if metadata.algorithm != run.algorithm:
            raise checkpoints.CorruptCheckpointError(
                "beta checkpoint algorithm differs from its training contract"
            )
        if set(metadata.extra) != {BETA_CHECKPOINT_EXTRA_KEY}:
            raise checkpoints.CorruptCheckpointError(
                "beta checkpoint metadata has unknown or missing evidence fields"
            )
        try:
            parsed = parse_beta_checkpoint_evidence(
                metadata.extra[BETA_CHECKPOINT_EXTRA_KEY]
            )
            ancestor = (
                _admit_resume(
                    parent_config,
                    runtime_provenance=runtime_provenance,
                    visited=visited | {directory},
                )
                if parent_config.resume_from is not None
                else None
            )
            expected_parent = (
                None if ancestor is None else ancestor.beta_parent_identity()
            )
            expected_total_timesteps = expected_beta_checkpoint_timesteps(
                parent_config,
                parent_total_timesteps=(
                    0 if ancestor is None else ancestor.total_timesteps
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
                ("optimizer seed", parsed.optimizer_seed, config.seed),
                ("timesteps", parsed.total_timesteps, metadata.total_timesteps),
                (
                    "runtime/source identity",
                    parsed.runtime_provenance,
                    runtime_provenance,
                ),
                ("resume parent", parsed.resume_parent, expected_parent),
            ):
                if actual != expected:
                    raise ValueError(
                        f"beta checkpoint {label} differs from external custody"
                    )

            verify_beta_allocator_evidence(
                parsed,
                parent_config,
                path=directory,
            )
            artifact_constraint_state, artifact_attestation = (
                _load_beta_artifact_evidence(
                    directory,
                    parent_config,
                )
            )
            if artifact_constraint_state != parsed.constraint_state:
                raise ValueError(
                    "artifact-loaded constraint state differs from checkpoint evidence"
                )
            ancestor_artifact = (
                None
                if ancestor is None
                else _load_beta_artifact_evidence(
                    Path(ancestor.path),
                    ControllerNeutralTrainingConfigV1.from_dict(
                        checkpoints.load_metadata(
                            ancestor.path,
                            check=False,
                        ).training_config
                    ),
                )
            )
            beta_evidence = admit_beta_checkpoint_evidence(
                parsed.to_dict(),
                expected_artifact_sha256=metadata.policy_artifact_sha256,
                expected_algorithm_id=run.algorithm,
                expected_run_kind=run.run_kind,
                expected_optimizer_seed=config.seed,
                expected_resume_parent=expected_parent,
                expected_runtime_provenance=runtime_provenance,
                expected_total_timesteps=metadata.total_timesteps,
                expected_artifact_attestation=artifact_attestation,
                expected_resume_parent_evidence=(
                    None
                    if ancestor is None or ancestor.beta_evidence is None
                    else ancestor.beta_evidence.to_dict()
                ),
                expected_parent_artifact_attestation=(
                    None if ancestor_artifact is None else ancestor_artifact[1]
                ),
            )
        except checkpoints.CheckpointError:
            raise
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as reason:
            raise checkpoints.CorruptCheckpointError(
                f"resume checkpoint {directory} has invalid beta evidence: {reason}"
            ) from reason

    try:
        reopened_metadata = checkpoints.load_metadata(
            directory,
            contract=parent_contract,
        )
        reopened_metadata_bytes = checkpoints.metadata_path(directory).read_bytes()
    except OSError as reason:
        raise checkpoints.CorruptCheckpointError(
            f"cannot re-bind resume metadata at {directory}: {reason}"
        ) from reason
    if (
        reopened_metadata != metadata
        or reopened_metadata_bytes != admitted_metadata_bytes
    ):
        raise checkpoints.CorruptCheckpointError(
            f"resume checkpoint metadata changed during admission ({directory})"
        )
    metadata_sha256 = hashlib.sha256(admitted_metadata_bytes).hexdigest()
    return ResumeCheckpointIdentityV1(
        path=str(directory),
        metadata_sha256=metadata_sha256,
        policy_artifact_sha256=metadata.policy_artifact_sha256,
        checkpoint_contract_digest=metadata.checkpoint_contract_digest,
        total_timesteps=metadata.total_timesteps,
        beta_evidence=beta_evidence,
    )


def admit_resume(
    config: ControllerNeutralTrainingConfigV1,
    *,
    runtime_provenance: BetaTrainingRuntimeProvenanceV1 | None = None,
) -> ResumeCheckpointIdentityV1 | None:
    """Validate a resume source and freeze its byte/task/hyperparameter identity."""

    config.validate()
    if config.beta_rl is not None:
        if runtime_provenance is None:
            runtime_provenance = capture_runtime_provenance(require_clean=False)
    elif runtime_provenance is not None:
        raise ValueError("legacy PPO resume cannot receive beta runtime provenance")
    identity = _admit_resume(
        config,
        runtime_provenance=runtime_provenance,
        visited=frozenset(),
    )
    if config.beta_rl is not None:
        if runtime_provenance is None:  # pragma: no cover - normalized above
            raise RuntimeError("beta resume admission lost runtime provenance")
        verify_runtime_unchanged(
            runtime_provenance,
            capture_runtime_provenance(
                require_clean=(config.beta_rl.run_kind == BETA_REGISTERED_RUN_KIND)
            ),
        )
    return identity


def _verify_resume_identity(identity: ResumeCheckpointIdentityV1) -> None:
    """Close the admission/use race before deserialising a resume parent."""

    directory = Path(identity.path)
    try:
        metadata_digest = hashlib.sha256(
            checkpoints.metadata_path(directory).read_bytes()
        ).hexdigest()
        artifact_digest = hashlib.sha256(
            checkpoints.policy_path(directory).read_bytes()
        ).hexdigest()
    except OSError as reason:
        raise checkpoints.CorruptCheckpointError(
            f"cannot reopen admitted resume checkpoint {directory}: {reason}"
        ) from reason
    if metadata_digest != identity.metadata_sha256:
        raise checkpoints.CorruptCheckpointError(
            f"resume checkpoint metadata changed after admission ({directory})"
        )
    if artifact_digest != identity.policy_artifact_sha256:
        raise checkpoints.CorruptCheckpointError(
            f"resume policy artifact changed after admission ({directory})"
        )
    metadata = checkpoints.load_metadata(directory, check=False)
    if metadata.policy_artifact_sha256 != identity.policy_artifact_sha256:
        raise checkpoints.CorruptCheckpointError(
            f"resume parent artifact identity disagrees with metadata ({directory})"
        )
    if metadata.checkpoint_contract_digest != identity.checkpoint_contract_digest:
        raise checkpoints.CorruptCheckpointError(
            f"resume parent task identity disagrees with metadata ({directory})"
        )
    if metadata.total_timesteps != identity.total_timesteps:
        raise checkpoints.CorruptCheckpointError(
            f"resume parent timestep count disagrees with metadata ({directory})"
        )


def restore_beta_episode_allocator(
    config: ControllerNeutralTrainingConfigV1,
    identity: ResumeCheckpointIdentityV1,
) -> EpisodeSeedAllocatorV1:
    """Restore exact fixture custody from a fully admitted beta parent."""

    config.validate()
    if config.beta_rl is None or config.resume_from is None:
        raise ValueError("beta allocator restoration requires a resumed beta run")
    if config.beta_rl.run_kind != BETA_FIXTURE_RUN_KIND:
        raise ValueError("only fixture beta training can restore episode custody")
    if str(Path(config.resume_from).expanduser().resolve()) != identity.path:
        raise checkpoints.CorruptCheckpointError(
            "allocator parent path differs from admitted resume identity"
        )
    evidence = identity.beta_evidence
    if evidence is None:
        raise checkpoints.CorruptCheckpointError(
            "admitted beta parent omitted checkpoint evidence"
        )
    fixture_resume = evidence.fixture_resume
    if fixture_resume is None:
        raise checkpoints.CorruptCheckpointError(
            "admitted beta parent omitted fixture resume state"
        )
    try:
        allocator = EpisodeSeedAllocatorV1.restore_fixture(
            fixture_resume,
            expected_parent_checkpoint_sha256=identity.policy_artifact_sha256,
            expected_resume_sha256=str(fixture_resume["sha256"]),
        )
        _validate_episode_allocator(config, allocator)
        if any(
            cursor.current_episode_ordinal is None
            or cursor.awaiting_reset
            or cursor.decision_offset != 0
            for cursor in allocator.cursors
        ):
            raise ValueError(
                "only an exact episode-boundary fixture checkpoint can reconstruct "
                "physical resume state"
            )
        if allocator.snapshot().to_dict() != evidence.allocator_snapshot:
            raise ValueError(
                "restored allocator differs from checkpoint evidence snapshot"
            )
    except (KeyError, TypeError, ValueError) as reason:
        raise checkpoints.CorruptCheckpointError(
            f"cannot restore beta episode custody from {identity.path}: {reason}"
        ) from reason
    return allocator


def build_model(
    config: ControllerNeutralTrainingConfigV1,
    env: Any,
    *,
    resume_parent: ResumeCheckpointIdentityV1 | None = None,
    runtime_provenance: BetaTrainingRuntimeProvenanceV1 | None = None,
    runtime_observation: Any | None = None,
) -> Any:
    """Construct a fresh PPO model or resume only an exactly compatible task."""

    settings = config.ppo
    beta_run = config.beta_rl
    if beta_run is not None:
        from .constrained_ppo import (
            ConstrainedPPO,
            ConstrainedPPOSettingsV1,
            build_constrained_model,
        )

        constrained_settings = ConstrainedPPOSettingsV1.for_algorithm(
            beta_run.algorithm,
            optimizer_seed=config.seed,
        )
        allocator = find_seed_allocator(env)
        _validate_episode_allocator(config, allocator)
        declared_environment = make_env(config)
        try:
            checkpoints.verify_policy_model_spaces(
                env,
                observation_space=declared_environment.observation_space,
                action_space=declared_environment.action_space,
                path=config.output_dir,
            )
        finally:
            declared_environment.close()
        if config.resume_from:
            if runtime_provenance is None:
                runtime_provenance = capture_runtime_provenance(require_clean=False)
            admitted = admit_resume(
                config,
                runtime_provenance=runtime_provenance,
            )
            if admitted is None:  # pragma: no cover - guarded by config.resume_from
                raise RuntimeError("resume admission did not produce a parent identity")
            if resume_parent is not None and resume_parent != admitted:
                raise checkpoints.CorruptCheckpointError(
                    "supplied resume identity differs from independent admission"
                )
            _verify_resume_identity(admitted)
            evidence = admitted.beta_evidence
            if evidence is None:
                raise checkpoints.CorruptCheckpointError(
                    "admitted beta parent omitted checkpoint evidence"
                )
            if runtime_observation is None:
                raise ValueError("beta resume requires the restored vector observation")
            beta_model = ConstrainedPPO.load(
                checkpoints.policy_path(admitted.path),
                env=env,
                device=settings.device,
                force_reset=False,
            )
            checkpoints.verify_policy_model_spaces(
                beta_model,
                observation_space=env.observation_space,
                action_space=env.action_space,
                path=admitted.path,
            )
            verify_beta_model_configuration(beta_model, config, path=admitted.path)
            loaded_attestation = beta_model.loaded_constraint_state_attestation()
            if loaded_attestation.artifact_sha256 != admitted.policy_artifact_sha256:
                raise checkpoints.CorruptCheckpointError(
                    "loaded beta resume artifact differs from its admitted identity"
                )
            if beta_model.embedded_constraint_state != evidence.constraint_state:
                raise checkpoints.CorruptCheckpointError(
                    "loaded constrained state differs from checkpoint evidence"
                )
            snapshot = allocator.snapshot().to_dict()
            if snapshot != evidence.allocator_snapshot:
                raise checkpoints.CorruptCheckpointError(
                    "live restored allocator differs from checkpoint evidence"
                )
            pin = evidence.constraint_state["payload"]["external_episode_state_pin"]
            if not isinstance(pin, Mapping) or not isinstance(pin.get("sha256"), str):
                raise checkpoints.CorruptCheckpointError(
                    "checkpoint constraint state omitted its allocator pin"
                )
            beta_model.admit_external_episode_state(
                snapshot,
                decision_indices=tuple(
                    cursor.decision_offset for cursor in allocator.cursors
                ),
                runtime_observation=runtime_observation,
                sha256=pin["sha256"],
            )
            return beta_model
        if resume_parent is not None:
            raise ValueError("a fresh beta model cannot receive a resume parent")
        if runtime_observation is not None:
            raise ValueError("a fresh beta model cannot receive a resume observation")
        _require_pristine_episode_allocator(allocator)
        beta_model = build_constrained_model(
            env,
            settings=constrained_settings,
            ppo_kwargs=_constrained_ppo_kwargs(config),
            verbose=1,
        )
        verify_beta_model_configuration(beta_model, config, path=config.output_dir)
        return beta_model

    if runtime_provenance is not None or runtime_observation is not None:
        raise ValueError("legacy PPO cannot receive beta runtime state")
    from stable_baselines3 import PPO

    if config.resume_from:
        admitted = admit_resume(config)
        if admitted is None:  # pragma: no cover - guarded by config.resume_from
            raise RuntimeError("resume admission did not produce a parent identity")
        if resume_parent is not None and resume_parent != admitted:
            raise checkpoints.CorruptCheckpointError(
                "supplied resume identity differs from independent admission"
            )
        _verify_resume_identity(admitted)
        model = PPO.load(
            checkpoints.policy_path(admitted.path),
            env=env,
            device=settings.device,
            seed=config.seed,
        )
        _reject_beta_artifact_downgrade(model)
        checkpoints.verify_policy_model_spaces(
            model,
            observation_space=env.observation_space,
            action_space=env.action_space,
            path=admitted.path,
        )
        return model
    return PPO(
        settings.policy,
        env,
        learning_rate=settings.learning_rate,
        n_steps=settings.n_steps,
        batch_size=settings.batch_size,
        n_epochs=settings.n_epochs,
        gamma=settings.gamma,
        gae_lambda=settings.gae_lambda,
        clip_range=settings.clip_range,
        ent_coef=settings.ent_coef,
        vf_coef=settings.vf_coef,
        max_grad_norm=settings.max_grad_norm,
        seed=config.seed,
        device=settings.device,
        verbose=1,
    )


def save_checkpoint(
    model: Any,
    config: ControllerNeutralTrainingConfigV1,
    path: Path | str,
    *,
    total_timesteps: int,
    resume_parent: ResumeCheckpointIdentityV1 | None = None,
    runtime_provenance: BetaTrainingRuntimeProvenanceV1 | None = None,
) -> Path:
    """Save policy bytes plus exact interface, task, artifact, and run identity."""

    config.validate()
    if (
        isinstance(total_timesteps, bool)
        or not isinstance(total_timesteps, Integral)
        or total_timesteps < 0
    ):
        raise ValueError("checkpoint total_timesteps must be a non-negative integer")
    if config.beta_rl is not None:
        from .constrained_ppo import ConstrainedPPO

        if total_timesteps <= 0:
            raise ValueError("beta checkpoint total_timesteps must be positive")
        if type(model) is not ConstrainedPPO:
            raise TypeError("beta checkpoint requires the exact ConstrainedPPO core")
        if int(model.num_timesteps) != int(total_timesteps):
            raise ValueError(
                "beta checkpoint timesteps differ from the live constrained model"
            )
        run = config.beta_rl
        current_runtime = capture_runtime_provenance(
            require_clean=run.run_kind == BETA_REGISTERED_RUN_KIND
        )
        if runtime_provenance is None:
            raise ValueError(
                "beta checkpoint saving requires pre-training runtime provenance"
            )
        verify_runtime_unchanged(runtime_provenance, current_runtime)
        if config.resume_from is None:
            if resume_parent is not None:
                raise ValueError("a fresh beta run cannot record a resume parent")
        else:
            if resume_parent is None:
                raise ValueError(
                    "a resumed beta run must record its admitted parent checkpoint"
                )
            admitted = admit_resume(
                config,
                runtime_provenance=runtime_provenance,
            )
            if admitted is None:  # pragma: no cover - config guards this
                raise RuntimeError("resume admission did not produce a parent identity")
            if resume_parent != admitted:
                raise checkpoints.CorruptCheckpointError(
                    "supplied resume identity differs from independent admission"
                )
            _verify_resume_identity(admitted)

        expected_total_timesteps = expected_beta_checkpoint_timesteps(
            config,
            parent_total_timesteps=(
                0 if resume_parent is None else resume_parent.total_timesteps
            ),
        )
        if int(total_timesteps) != expected_total_timesteps:
            raise ValueError(
                "beta checkpoint cumulative timestep cut is not explained by its "
                "training config and admitted parent"
            )
        verify_beta_model_configuration(model, config, path=path)
        _verify_beta_model_spaces(model, config, path=path)

        vector_environment = model.get_env()
        if vector_environment is None:
            raise ValueError("beta checkpoint model has no attached vector environment")
        allocator = find_seed_allocator(vector_environment)
        _validate_episode_allocator(config, allocator)
        snapshot = allocator.snapshot().to_dict()
        expected_per_environment = int(total_timesteps) // config.n_envs
        if int(total_timesteps) % config.n_envs or (
            allocator.decisions_consumed_by_environment
            != (expected_per_environment,) * config.n_envs
        ):
            raise ValueError(
                "beta checkpoint allocator progress differs from model timesteps"
            )
        cursor_offsets = tuple(cursor.decision_offset for cursor in allocator.cursors)
        if model.episode_decision_indices != cursor_offsets:
            raise ValueError(
                "beta constraint accumulators differ from allocator decision cursors"
            )
        model.pin_external_episode_state(snapshot)
        constraint_state = cast(dict[str, Any], model.constraint_provenance())

        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=False)
        artifact = checkpoints.policy_path(directory)
        model.save(artifact)
        if not artifact.is_file():
            raise RuntimeError(
                f"constrained PPO did not write its expected artifact at {artifact}"
            )
        if model.embedded_constraint_state != constraint_state:
            raise RuntimeError(
                "saved constrained state differs from pre-save optimizer evidence"
            )
        artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
        reopened = ConstrainedPPO.load(artifact, device=config.ppo.device)
        verify_beta_model_configuration(reopened, config, path=directory)
        _verify_beta_model_spaces(reopened, config, path=directory)
        reopened_attestation = reopened.loaded_constraint_state_attestation()
        if (
            reopened_attestation.artifact_sha256 != artifact_sha256
            or reopened_attestation.num_timesteps != int(total_timesteps)
            or reopened_attestation.constraint_state != constraint_state
        ):
            raise RuntimeError(
                "artifact attestation differs from saved optimizer state"
            )
        verify_runtime_unchanged(
            runtime_provenance,
            capture_runtime_provenance(
                require_clean=run.run_kind == BETA_REGISTERED_RUN_KIND
            ),
        )
        artifact_attestation = (
            reopened_attestation if run.run_kind == BETA_REGISTERED_RUN_KIND else None
        )
        fixture_resume = (
            allocator.export_fixture_resume(
                parent_checkpoint_sha256=artifact_sha256,
            )
            if run.run_kind == BETA_FIXTURE_RUN_KIND
            else None
        )
        evidence = build_beta_checkpoint_evidence(
            algorithm_id=run.algorithm,
            run_kind=run.run_kind,
            optimizer_seed=config.seed,
            total_timesteps=int(total_timesteps),
            artifact_sha256=artifact_sha256,
            runtime_provenance=runtime_provenance,
            allocator_snapshot=snapshot,
            constraint_state=constraint_state,
            fixture_resume=fixture_resume,
            resume_parent=(
                None if resume_parent is None else resume_parent.beta_parent_identity()
            ),
            resumed_from_parent=resume_parent is not None,
            exact_endpoint_admitted=(run.run_kind == BETA_REGISTERED_RUN_KIND),
            eligible_for_registered_evaluation=(
                run.run_kind == BETA_REGISTERED_RUN_KIND
            ),
            artifact_attestation=artifact_attestation,
        )
        verify_beta_allocator_evidence(evidence, config, path=directory)
        contract = policy_contract(config)
        checkpoints.save_metadata(
            directory,
            checkpoints.CheckpointMetadata(
                format=checkpoints.FORMAT_SB3_PPO,
                algorithm=run.algorithm,
                spec_version=contract.spec_version,
                observation_fields=contract.observation_fields,
                action_fields=contract.action_fields,
                environment_id=contract.environment_id,
                environment_contract_digest=contract.environment_contract_digest,
                checkpoint_contract_digest=contract.checkpoint_contract_digest,
                policy_artifact_sha256=artifact_sha256,
                seed=config.seed,
                total_timesteps=int(total_timesteps),
                training_config=config.to_dict(),
                extra={BETA_CHECKPOINT_EXTRA_KEY: evidence.to_dict()},
                notes=config.notes,
            ),
        )
        config.write(directory / CONFIG_FILE)
        return directory

    if runtime_provenance is not None:
        raise ValueError("legacy PPO checkpoint cannot receive beta runtime provenance")
    _reject_beta_artifact_downgrade(model)
    if config.resume_from is None and resume_parent is not None:
        raise ValueError("a fresh run cannot record a resume parent")
    if config.resume_from is not None:
        if resume_parent is None:
            raise ValueError("a resumed run must record its admitted parent checkpoint")
        admitted = admit_resume(config)
        if admitted is None:  # pragma: no cover - guarded by config.resume_from
            raise RuntimeError("resume admission did not produce a parent identity")
        if resume_parent != admitted:
            raise checkpoints.CorruptCheckpointError(
                "supplied resume identity differs from independent admission"
            )
        _verify_resume_identity(admitted)
    declared_environment = make_env(config)
    try:
        checkpoints.verify_policy_model_spaces(
            model,
            observation_space=declared_environment.observation_space,
            action_space=declared_environment.action_space,
            path=path,
        )
    finally:
        declared_environment.close()
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=False)
    artifact = checkpoints.policy_path(directory)
    model.save(artifact)
    if not artifact.is_file():
        raise RuntimeError(f"PPO did not write its expected artifact at {artifact}")
    contract = policy_contract(config)
    checkpoints.save_metadata(
        directory,
        checkpoints.CheckpointMetadata(
            format=checkpoints.FORMAT_SB3_PPO,
            algorithm="PPO (stable-baselines3)",
            spec_version=contract.spec_version,
            observation_fields=contract.observation_fields,
            action_fields=contract.action_fields,
            environment_id=contract.environment_id,
            environment_contract_digest=contract.environment_contract_digest,
            checkpoint_contract_digest=contract.checkpoint_contract_digest,
            policy_artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            seed=config.seed,
            total_timesteps=int(total_timesteps),
            training_config=config.to_dict(),
            extra=(
                {"resume_parent": resume_parent.to_dict()}
                if resume_parent is not None
                else {}
            ),
            notes=config.notes,
        ),
    )
    config.write(directory / CONFIG_FILE)
    return directory


def plan(config: ControllerNeutralTrainingConfigV1) -> dict[str, Any]:
    """Identify the exact task without importing a trainer or taking a gradient.

    Legacy plans retain their historical single reset.  A beta-RL fixture plan
    resets all four allocator-wrapped environments only inside the fixture seed
    domain.  A registered plan is deliberately outcome-blind and performs no
    reset in the protected training domain.
    """

    config.validate()
    environments = [make_env(config, index=0)]
    wrapped_environments: list[Any] = []
    allocator_snapshot: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    reset_info: dict[str, Any] | None = None
    first_episode_seed: int | None = None
    environment = environments[0]
    try:
        contract = policy_contract_for_environment(environment)
        if config.beta_rl is None:
            observation, reset_info = environment.reset(seed=config.seed)
            observation_shape = list(observation.shape)
            manifest = environment.audit_manifest()
        elif config.beta_rl.run_kind == BETA_FIXTURE_RUN_KIND:
            allocator = build_episode_allocator(config)
            environments.extend(
                make_env(config, index=index) for index in range(1, config.n_envs)
            )
            wrapped_environments = [
                allocator.wrap(item, environment_index=index)
                for index, item in enumerate(environments)
            ]
            reset_results = [item.reset() for item in wrapped_environments]
            observation_shape = list(reset_results[0][0].shape)
            reset_info = reset_results[0][1]
            manifest = environment.audit_manifest()
            allocator_snapshot = allocator.snapshot().to_dict()
            first_episode_seed = allocator.ledger[0].episode_seed
        else:
            observation_shape = list(environment.observation_space.shape or ())

        if manifest is None:
            exogenous_episode_sha256 = None
            physical_exogenous_episode_sha256 = None
            first_episode_manifest_sha256 = None
        elif isinstance(environment, PlantCyclicMarketEnvV2Beta):
            exogenous_episode_sha256 = str(manifest["beta_exogenous_episode_sha256"])
            physical_exogenous_episode_sha256 = str(
                manifest["alpha_physical_exogenous_episode_sha256"]
            )
            first_episode_manifest_sha256 = manifest["manifest_sha256"]
        else:
            exogenous_episode_sha256 = str(manifest["exogenous_episode_sha256"])
            physical_exogenous_episode_sha256 = exogenous_episode_sha256
            first_episode_manifest_sha256 = manifest["manifest_sha256"]

        steps_per_episode = environment.episode_decisions
        rollout_size = config.ppo.n_steps * config.n_envs
        minimum_updates = math.ceil(config.total_timesteps / rollout_size)
        minimum_collected_timesteps = minimum_updates * rollout_size
        return {
            "environment_id": config.environment_id,
            "environment_contract_digest": contract.environment_contract_digest,
            "checkpoint_contract_digest": contract.checkpoint_contract_digest,
            "seed": config.seed,
            "optimizer_seed": config.seed if config.beta_rl is not None else None,
            "first_episode_seed": first_episode_seed,
            "total_timesteps": config.total_timesteps,
            "minimum_collected_timesteps": minimum_collected_timesteps,
            "n_envs": config.n_envs,
            "rollout_transitions_per_update": rollout_size,
            "minimum_updates": minimum_updates,
            "observation_shape": observation_shape,
            "action_shape": list(environment.action_space.shape or ()),
            "steps_per_episode": steps_per_episode,
            "physics_steps_per_episode": environment.episode_physics_steps,
            "minimum_episode_equivalents_per_env": (
                minimum_collected_timesteps / max(1, config.n_envs) / steps_per_episode
            ),
            "minimum_simulated_days_total": (
                minimum_collected_timesteps
                * config.episode.decision_interval_s
                / 86_400.0
            ),
            "minimum_simulated_days_per_env": (
                minimum_collected_timesteps
                * config.episode.decision_interval_s
                / max(1, config.n_envs)
                / 86_400.0
            ),
            "environment_reset": manifest is not None,
            "first_episode_manifest_sha256": first_episode_manifest_sha256,
            "first_exogenous_episode_sha256": exogenous_episode_sha256,
            "first_physical_exogenous_episode_sha256": (
                physical_exogenous_episode_sha256
            ),
            "first_reset_checkpoint_contract_digest": (
                None if reset_info is None else reset_info["checkpoint_contract_digest"]
            ),
            "episode_allocator": allocator_snapshot,
            "output_dir": config.output_dir,
            "resume_from": config.resume_from,
            "training_library_imported": False,
        }
    finally:
        if wrapped_environments:
            for item in wrapped_environments:
                item.close()
        else:
            environment.close()


__all__ = [
    "BETA_CHECKPOINT_EXTRA_KEY",
    "CONFIG_FILE",
    "RESUME_PARENT_SCHEMA",
    "ResumeCheckpointIdentityV1",
    "admit_resume",
    "build_episode_allocator",
    "build_model",
    "build_vec_env",
    "expected_beta_checkpoint_timesteps",
    "make_env",
    "plan",
    "policy_contract",
    "restore_beta_episode_allocator",
    "save_checkpoint",
    "verify_beta_allocator_evidence",
    "verify_beta_model_configuration",
]

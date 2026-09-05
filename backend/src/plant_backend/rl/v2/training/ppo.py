"""Stable-Baselines3 wiring for controller-neutral Environment V2 policies."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import math
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, cast

from ... import checkpoints
from .. import economic_spaces, setpoint_spaces
from ..economic_env import PlantEconomicSetpointEnvV2
from ..policy_setpoint_controller import (
    SetpointPolicyEnvironmentV1,
    policy_contract_for_environment,
)
from ..setpoint_env import PlantSetpointEnvV2
from .config import ControllerNeutralTrainingConfigV1


CONFIG_FILE = "training_config.json"
RESUME_PARENT_SCHEMA = "controller-neutral-rl-resume-parent-v1"


@dataclass(frozen=True, slots=True)
class ResumeCheckpointIdentityV1:
    """Immutable identity of the checkpoint from which one run resumed."""

    path: str
    metadata_sha256: str
    policy_artifact_sha256: str
    checkpoint_contract_digest: str
    total_timesteps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RESUME_PARENT_SCHEMA,
            "path": self.path,
            "metadata_sha256": self.metadata_sha256,
            "policy_artifact_sha256": self.policy_artifact_sha256,
            "checkpoint_contract_digest": self.checkpoint_contract_digest,
            "total_timesteps": self.total_timesteps,
        }


def make_env(
    config: ControllerNeutralTrainingConfigV1,
    *,
    index: int = 0,
) -> SetpointPolicyEnvironmentV1:
    """Construct one independent task; seeding occurs at reset, not construction."""

    del index
    config.validate()
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


def build_vec_env(config: ControllerNeutralTrainingConfigV1) -> Any:
    """Create independently seeded synchronous environments for reproducible PPO."""

    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    def factory(index: int) -> Callable[[], Any]:
        def build() -> Any:
            environment: Any = Monitor(make_env(config, index=index))
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


def admit_resume(
    config: ControllerNeutralTrainingConfigV1,
) -> ResumeCheckpointIdentityV1 | None:
    """Validate a resume source and freeze its byte/task/hyperparameter identity."""

    config.validate()
    if config.resume_from is None:
        return None
    directory = Path(config.resume_from).expanduser().resolve()
    metadata = checkpoints.load_metadata(
        directory,
        contract=policy_contract(config),
    )
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
    differences = _resume_ppo_differences(parent_config, config)
    if differences:
        raise checkpoints.IncompatibleCheckpointError(
            "resume PPO settings differ from the loaded model for "
            f"{', '.join(differences)}; start a fresh run or preserve the parent "
            "settings (device is the only supported resume override)"
        )
    try:
        metadata_sha256 = hashlib.sha256(
            checkpoints.metadata_path(directory).read_bytes()
        ).hexdigest()
    except OSError as reason:
        raise checkpoints.CorruptCheckpointError(
            f"cannot bind resume metadata at {directory}: {reason}"
        ) from reason
    return ResumeCheckpointIdentityV1(
        path=str(directory),
        metadata_sha256=metadata_sha256,
        policy_artifact_sha256=metadata.policy_artifact_sha256,
        checkpoint_contract_digest=metadata.checkpoint_contract_digest,
        total_timesteps=metadata.total_timesteps,
    )


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


def build_model(
    config: ControllerNeutralTrainingConfigV1,
    env: Any,
    *,
    resume_parent: ResumeCheckpointIdentityV1 | None = None,
) -> Any:
    """Construct a fresh PPO model or resume only an exactly compatible task."""

    from stable_baselines3 import PPO

    settings = config.ppo
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
) -> Path:
    """Save policy bytes plus exact interface, task, artifact, and run identity."""

    config.validate()
    if (
        isinstance(total_timesteps, bool)
        or not isinstance(total_timesteps, Integral)
        or total_timesteps < 0
    ):
        raise ValueError("checkpoint total_timesteps must be a non-negative integer")
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
    """Build and reset the exact task, but import no trainer and take no gradient."""

    environment = make_env(config)
    observation, info = environment.reset(seed=config.seed)
    contract = policy_contract_for_environment(environment)
    steps_per_episode = environment.episode_decisions
    rollout_size = config.ppo.n_steps * config.n_envs
    minimum_updates = math.ceil(config.total_timesteps / rollout_size)
    minimum_collected_timesteps = minimum_updates * rollout_size
    return {
        "environment_id": config.environment_id,
        "environment_contract_digest": contract.environment_contract_digest,
        "checkpoint_contract_digest": contract.checkpoint_contract_digest,
        "seed": config.seed,
        "total_timesteps": config.total_timesteps,
        "minimum_collected_timesteps": minimum_collected_timesteps,
        "n_envs": config.n_envs,
        "rollout_transitions_per_update": rollout_size,
        "minimum_updates": minimum_updates,
        "observation_shape": list(observation.shape),
        "action_shape": list(environment.action_space.shape or ()),
        "steps_per_episode": steps_per_episode,
        "physics_steps_per_episode": environment.episode_physics_steps,
        "minimum_episode_equivalents_per_env": (
            minimum_collected_timesteps / max(1, config.n_envs) / steps_per_episode
        ),
        "minimum_simulated_days_total": (
            minimum_collected_timesteps * config.episode.decision_interval_s / 86_400.0
        ),
        "minimum_simulated_days_per_env": (
            minimum_collected_timesteps
            * config.episode.decision_interval_s
            / max(1, config.n_envs)
            / 86_400.0
        ),
        "first_episode_manifest_sha256": environment.audit_manifest()[
            "manifest_sha256"
        ],
        "first_exogenous_episode_sha256": environment.audit_manifest()[
            "exogenous_episode_sha256"
        ],
        "first_reset_checkpoint_contract_digest": info["checkpoint_contract_digest"],
        "output_dir": config.output_dir,
        "resume_from": config.resume_from,
        "training_library_imported": False,
    }


__all__ = [
    "CONFIG_FILE",
    "RESUME_PARENT_SCHEMA",
    "ResumeCheckpointIdentityV1",
    "admit_resume",
    "build_model",
    "build_vec_env",
    "make_env",
    "plan",
    "policy_contract",
    "save_checkpoint",
]

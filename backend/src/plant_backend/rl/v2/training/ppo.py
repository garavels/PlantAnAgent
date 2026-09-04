"""Stable-Baselines3 wiring for controller-neutral Environment V2 policies."""

from __future__ import annotations

import hashlib
import math
from numbers import Integral
from pathlib import Path
from typing import Any, Callable

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
    return policy_contract_for_environment(make_env(config))


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


def build_model(config: ControllerNeutralTrainingConfigV1, env: Any) -> Any:
    """Construct a fresh PPO model or resume only an exactly compatible task."""

    from stable_baselines3 import PPO

    settings = config.ppo
    if config.resume_from:
        directory = Path(config.resume_from)
        metadata = checkpoints.load_metadata(
            directory,
            contract=policy_contract(config),
        )
        if metadata.format != checkpoints.FORMAT_SB3_PPO:
            raise checkpoints.IncompatibleCheckpointError(
                f"cannot resume from a {metadata.format!r} checkpoint ({directory}): "
                f"expected {checkpoints.FORMAT_SB3_PPO}"
            )
        return PPO.load(
            checkpoints.policy_path(directory),
            env=env,
            device=settings.device,
            seed=config.seed,
        )
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
) -> Path:
    """Save policy bytes plus exact interface, task, artifact, and run identity."""

    config.validate()
    if (
        isinstance(total_timesteps, bool)
        or not isinstance(total_timesteps, Integral)
        or total_timesteps < 0
    ):
        raise ValueError("checkpoint total_timesteps must be a non-negative integer")
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
    "build_model",
    "build_vec_env",
    "make_env",
    "plan",
    "policy_contract",
    "save_checkpoint",
]

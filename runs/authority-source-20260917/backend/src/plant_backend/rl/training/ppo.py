"""PPO on the plant: env construction, model construction, checkpoint writing.

PPO because the action is five continuous shares, it is the most boring choice
available, and a maintained implementation of a boring algorithm is worth more
here than a bespoke one: the interesting part of this project is the plant, not
the optimiser. Stable-Baselines3 provides it; this file is only the wiring, so
swapping in SAC or TD3 later means another `build_model`, not another env.

Nothing in here trains anything by itself and nothing imports Torch at module
scope -- the imports are inside the functions that need them, so the backend and
the test suite stay installable without the training extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...controllers.base import Constraints
from .. import checkpoints
from ..env import PlantEnv
from .config import TrainingConfig

#: Written next to the checkpoint so a run can be repeated from its output alone.
CONFIG_FILE = "training_config.json"


def make_env(config: TrainingConfig, *, index: int = 0) -> PlantEnv:
    """One plant, seeded at `config.seed + index` when it is first reset."""
    return PlantEnv(
        episode=config.episode,
        constraints=Constraints(),
        weights=config.reward,
    )


def build_vec_env(config: TrainingConfig) -> Any:
    """`config.n_envs` plants in one vectorised env, each with its own seed.

    Vectorised for throughput only: the plants are independent, and each one draws
    its episode from `config.seed + index`, so a run is reproducible for a given
    `n_envs` and identical episode distribution.
    """
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    def factory(index: int):
        def build() -> Any:
            env = Monitor(make_env(config, index=index))
            env.reset(seed=config.seed + index)
            return env

        return build

    return DummyVecEnv([factory(index) for index in range(config.n_envs)])


def build_model(config: TrainingConfig, env: Any) -> Any:
    """A fresh PPO model, or the resumed one when `resume_from` is set."""
    from stable_baselines3 import PPO

    ppo = config.ppo
    if config.resume_from:
        directory = Path(config.resume_from)
        metadata = checkpoints.load_metadata(directory)
        if metadata.format != checkpoints.FORMAT_SB3_PPO:
            raise checkpoints.IncompatibleCheckpointError(
                f"cannot resume from a {metadata.format!r} checkpoint ({directory}): "
                f"expected {checkpoints.FORMAT_SB3_PPO}"
            )
        model = PPO.load(
            checkpoints.policy_path(directory),
            env=env,
            device=ppo.device,
            seed=config.seed,
        )
        return model
    return PPO(
        ppo.policy,
        env,
        learning_rate=ppo.learning_rate,
        n_steps=ppo.n_steps,
        batch_size=ppo.batch_size,
        n_epochs=ppo.n_epochs,
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
        clip_range=ppo.clip_range,
        ent_coef=ppo.ent_coef,
        vf_coef=ppo.vf_coef,
        max_grad_norm=ppo.max_grad_norm,
        seed=config.seed,
        device=ppo.device,
        verbose=1,
    )


def save_checkpoint(
    model: Any,
    config: TrainingConfig,
    path: Path | str,
    *,
    total_timesteps: int,
) -> Path:
    """Write `policy.zip` + `metadata.json` + the config that produced them."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(checkpoints.policy_path(directory))
    checkpoints.save_metadata(
        directory,
        checkpoints.CheckpointMetadata(
            format=checkpoints.FORMAT_SB3_PPO,
            algorithm="PPO (stable-baselines3)",
            seed=config.seed,
            total_timesteps=total_timesteps,
            training_config=config.to_dict(),
            notes=config.notes,
        ),
    )
    config.write(directory / CONFIG_FILE)
    return directory


def plan(config: TrainingConfig) -> dict[str, Any]:
    """What this config would do, without doing any of it.

    Used by `--dry-run` and by the tests: it constructs one plant, resets it and
    reports the shapes and the episode arithmetic, which is the part of a training
    run that can be wrong without a GPU being involved.
    """
    env = make_env(config)
    observation, info = env.reset(seed=config.seed)
    steps_per_episode = env.episode_steps
    return {
        "seed": config.seed,
        "total_timesteps": config.total_timesteps,
        "n_envs": config.n_envs,
        "observation_shape": list(observation.shape),
        "action_shape": list(env.action_space.shape or ()),
        "steps_per_episode": steps_per_episode,
        "episodes_per_env": config.total_timesteps / max(1, config.n_envs) / steps_per_episode,
        "simulated_days": config.total_timesteps * env.config.dt_h / 24.0,
        "first_episode": info["episode"]["scenario"]["id"],
        "output_dir": config.output_dir,
        "resume_from": config.resume_from,
    }


__all__ = ["CONFIG_FILE", "build_model", "build_vec_env", "make_env", "plan", "save_checkpoint"]

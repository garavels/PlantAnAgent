"""Everything a training run is, as data: seeds, episodes, reward, PPO settings.

One file describes one run. It is JSON rather than code so a run can be recorded
next to its checkpoint (`CheckpointMetadata.training_config`) and repeated later
without a diff being involved, and so the entry point has nothing to configure
beyond "which config, which output directory".

The PPO numbers are **library defaults with a longer rollout**, not tuned values:
nothing has been trained yet, so there is no evidence to tune against. The one
deliberate choice is `n_steps`: the plant's timestep is 60 s, so 2048 steps is a
day and a half of simulated operation per update -- long enough that a rollout
contains a night, a morning ramp and a full tank cycle, which is the shortest
window in which the trade-off this reward describes is even visible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from ..episodes import EpisodeConfig
from ..reward import RewardWeights


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters handed straight to Stable-Baselines3's PPO."""

    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    #: Steps per environment per update. 2048 * 60 s ~ 34 h of plant operation.
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    #: CPU is the right default: the network is small and the env is the bottleneck.
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    """A complete, repeatable training run."""

    #: Seeds every stochastic part of the run: env resets, action sampling, init.
    seed: int = 0
    total_timesteps: int = 1_000_000
    #: Parallel copies of the plant. Each gets `seed + index`.
    n_envs: int = 4
    #: Timesteps between intermediate checkpoints. 0 saves only at the end.
    checkpoint_every: int = 100_000
    output_dir: str = "runs/ppo"
    #: Resume from this checkpoint directory instead of starting fresh.
    resume_from: str | None = None
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    reward: RewardWeights = field(default_factory=RewardWeights)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    #: Free-text provenance, copied into the checkpoint metadata.
    notes: str = ""

    def validate(self) -> "TrainingConfig":
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.n_envs <= 0:
            raise ValueError("n_envs must be positive")
        self.episode.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "total_timesteps": self.total_timesteps,
            "n_envs": self.n_envs,
            "checkpoint_every": self.checkpoint_every,
            "output_dir": self.output_dir,
            "resume_from": self.resume_from,
            "episode": self.episode.to_dict(),
            "reward": self.reward.to_dict(),
            "ppo": self.ppo.to_dict(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "TrainingConfig":
        if not raw:
            return cls()
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown training settings: {', '.join(sorted(unknown))}")
        updates: dict[str, Any] = {}
        for key, value in raw.items():
            if key == "episode":
                updates[key] = EpisodeConfig.from_dict(value)
            elif key == "reward":
                updates[key] = RewardWeights.from_dict(value)
            elif key == "ppo":
                updates[key] = _ppo_from_dict(value)
            elif key == "resume_from":
                updates[key] = None if value is None else str(value)
            elif key in ("output_dir", "notes"):
                updates[key] = str(value)
            else:
                updates[key] = int(value)
        return replace(cls(), **updates).validate()

    @classmethod
    def from_file(cls, path: Path | str) -> "TrainingConfig":
        target = Path(path)
        if not target.is_file():
            raise ValueError(f"no training config at {target}")
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as reason:
            raise ValueError(f"cannot read {target}: {reason}") from reason
        return cls.from_dict(raw)

    def write(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target


def _ppo_from_dict(values: dict[str, Any] | None) -> PPOConfig:
    if not values:
        return PPOConfig()
    known = {f.name for f in fields(PPOConfig)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown PPO settings: {', '.join(sorted(unknown))}")
    typed: dict[str, Any] = {}
    for key, value in values.items():
        if key in ("policy", "device"):
            typed[key] = str(value)
        elif key in ("n_steps", "batch_size", "n_epochs"):
            typed[key] = int(value)
        else:
            typed[key] = float(value)
    return replace(PPOConfig(), **typed)


__all__ = ["PPOConfig", "TrainingConfig"]

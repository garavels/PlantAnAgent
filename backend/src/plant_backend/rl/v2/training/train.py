"""CLI for training a checkpoint-bound policy on the four-target V2 task."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

from ... import checkpoints
from .. import beta_spaces
from ..beta_config import CyclicMarketTaskConfigV1
from . import ppo
from .config import (
    BETA_REGISTERED_RUN_KIND,
    SUPPORTED_CONTROLLER_NEUTRAL_ENVIRONMENTS,
    ControllerNeutralTrainingConfigV1,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plant_backend.rl.v2.training",
        description="Train PPO through the controller-neutral four-setpoint door.",
    )
    parser.add_argument("--config", type=Path, help="strict JSON training contract")
    parser.add_argument("--output", type=Path, help="override the output directory")
    parser.add_argument("--seed", type=int, help="override the run seed")
    parser.add_argument("--timesteps", type=int, help="override learning timesteps")
    parser.add_argument(
        "--resume", type=Path, help="resume an exact-compatible checkpoint"
    )
    parser.add_argument(
        "--environment",
        choices=SUPPORTED_CONTROLLER_NEUTRAL_ENVIRONMENTS,
        help="override the four-target task",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="construct, reset, and identify the task without importing SB3 or training",
    )
    return parser


def resolve(arguments: argparse.Namespace) -> ControllerNeutralTrainingConfigV1:
    config = (
        ControllerNeutralTrainingConfigV1.from_file(arguments.config)
        if arguments.config
        else ControllerNeutralTrainingConfigV1()
    )
    if (
        config.beta_rl is not None
        and config.beta_rl.run_kind == BETA_REGISTERED_RUN_KIND
    ):
        forbidden_overrides = tuple(
            name
            for name, value in (
                ("seed", arguments.seed),
                ("timesteps", arguments.timesteps),
                ("resume", arguments.resume),
                ("environment", arguments.environment),
            )
            if value is not None
        )
        if forbidden_overrides:
            raise ValueError(
                "registered beta RL permits only an output override; refused "
                + ", ".join(forbidden_overrides)
            )
    if (
        config.beta_rl is not None
        and arguments.environment is not None
        and arguments.environment != config.environment_id
    ):
        raise ValueError(
            "a beta RL training contract cannot be relabelled through an "
            "environment override"
        )
    updates: dict[str, Any] = {}
    if arguments.output is not None:
        updates["output_dir"] = str(arguments.output)
    if arguments.seed is not None:
        updates["seed"] = arguments.seed
    if arguments.timesteps is not None:
        updates["total_timesteps"] = arguments.timesteps
    if arguments.resume is not None:
        updates["resume_from"] = str(arguments.resume)
    if arguments.environment is not None:
        updates["environment_id"] = arguments.environment
        if arguments.environment == beta_spaces.ENVIRONMENT_ID:
            if config.beta_task is None:
                updates["beta_task"] = CyclicMarketTaskConfigV1()
            if arguments.config is None and arguments.output is None:
                updates["output_dir"] = "runs/v2-cyclic-market-beta-ppo"
        elif config.beta_task is not None:
            updates["beta_task"] = None
    return replace(config, **updates).validate()


def _checkpoint_callback(
    config: ControllerNeutralTrainingConfigV1,
    output: Path,
    resume_parent: ppo.ResumeCheckpointIdentityV1 | None,
) -> Any:
    from stable_baselines3.common.callbacks import BaseCallback

    class AuditableCheckpointCallback(BaseCallback):  # type: ignore[misc]
        """Write complete checkpoints; raw SB3 zip files are never called checkpoints."""

        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.next_checkpoint: int | None = None

        def _on_training_start(self) -> None:
            if config.checkpoint_every > 0:
                self.next_checkpoint = (
                    int(self.model.num_timesteps) + config.checkpoint_every
                )

        def _on_step(self) -> bool:
            if self.next_checkpoint is None:
                return True
            if self.num_timesteps < self.next_checkpoint:
                return True
            ppo.save_checkpoint(
                self.model,
                config,
                output / "intermediate" / f"step-{self.num_timesteps:012d}",
                total_timesteps=self.num_timesteps,
                resume_parent=resume_parent,
            )
            while self.next_checkpoint <= self.num_timesteps:
                self.next_checkpoint += config.checkpoint_every
            return True

    return AuditableCheckpointCallback()


def train(config: ControllerNeutralTrainingConfigV1) -> Path:
    """Run the explicitly requested long optimization and save a complete checkpoint."""

    output = Path(config.output_dir)
    if output.exists():
        raise ValueError(
            f"training output already exists at {output}; choose a new run directory"
        )
    resume_parent = ppo.admit_resume(config)
    environment = ppo.build_vec_env(config)
    try:
        model = ppo.build_model(
            config,
            environment,
            resume_parent=resume_parent,
        )
        # Dependency, environment, and resume admission all complete before the
        # run directory is claimed. A missing optional trainer must not leave an
        # empty path that looks like a failed experiment.
        output.mkdir(parents=True, exist_ok=False)
        callback = _checkpoint_callback(config, output, resume_parent)
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callback,
            reset_num_timesteps=config.resume_from is None,
        )
        return ppo.save_checkpoint(
            model,
            config,
            output / "checkpoint",
            total_timesteps=int(model.num_timesteps),
            resume_parent=resume_parent,
        )
    finally:
        environment.close()


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        config = resolve(arguments)
    except (TypeError, ValueError) as reason:
        print(f"invalid controller-neutral RL configuration: {reason}", file=sys.stderr)
        return 2

    if arguments.dry_run:
        print(json.dumps({"dry_run": True, "plan": ppo.plan(config)}, indent=2))
        print("nothing was trained", file=sys.stderr)
        return 0

    try:
        checkpoint = train(config)
    except ImportError:
        print(
            "stable-baselines3 is not installed: run `uv sync --group train` first",
            file=sys.stderr,
        )
        return 2
    except (checkpoints.CheckpointError, OSError, RuntimeError, ValueError) as reason:
        print(f"controller-neutral training failed: {reason}", file=sys.stderr)
        return 2
    print(f"checkpoint written to {checkpoint}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main", "resolve", "train"]

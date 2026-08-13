"""The training entry point. Nothing has been trained with it yet.

    uv run python -m plant_backend.rl.training.train --dry-run
    uv run python -m plant_backend.rl.training.train --config <file> --output runs/ppo-1

It talks to the simulator and to Stable-Baselines3, and to nothing else: no
FastAPI, no SSE, no frontend. That is what makes it runnable anywhere a Python
process and this repository fit -- a laptop, a CI box, or a rented GPU -- without
the application server being part of the deployment.

`--dry-run` builds a plant, resets it and prints the arithmetic of the run
(observation shape, steps per episode, simulated days) without constructing a
network or taking a gradient step. It is the check that the wiring is right, and
it is what the test suite runs; the real command is deliberately left for a
separate, deliberate session.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import ppo
from .config import TrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plant_backend.rl.training.train",
        description="Train a PPO controller for the plant. Nothing is trained by --dry-run.",
    )
    parser.add_argument("--config", type=Path, help="JSON training config; defaults apply if omitted")
    parser.add_argument("--output", type=Path, help="output directory (overrides the config)")
    parser.add_argument("--seed", type=int, help="override the config's seed")
    parser.add_argument("--timesteps", type=int, help="override total_timesteps")
    parser.add_argument("--resume", type=Path, help="resume from this checkpoint directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build the environment and print the plan without training",
    )
    return parser


def resolve(args: argparse.Namespace) -> TrainingConfig:
    """Config file first, command line second: flags override what the file said."""
    config = TrainingConfig.from_file(args.config) if args.config else TrainingConfig()
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.timesteps is not None:
        overrides["total_timesteps"] = args.timesteps
    if args.output is not None:
        overrides["output_dir"] = str(args.output)
    if args.resume is not None:
        overrides["resume_from"] = str(args.resume)
    if overrides:
        config = TrainingConfig.from_dict({**config.to_dict(), **overrides})
    return config.validate()


def train(config: TrainingConfig) -> Path:
    """Run PPO for `config.total_timesteps` and save the checkpoint. Long."""
    from stable_baselines3.common.callbacks import CheckpointCallback

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    env = ppo.build_vec_env(config)
    model = ppo.build_model(config, env)
    callbacks = []
    if config.checkpoint_every > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, config.checkpoint_every // config.n_envs),
                save_path=str(output / "intermediate"),
                name_prefix="ppo-plant",
            )
        )
    model.learn(total_timesteps=config.total_timesteps, callback=callbacks or None)
    return ppo.save_checkpoint(
        model, config, output / "checkpoint", total_timesteps=config.total_timesteps
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = resolve(args)
    except ValueError as reason:
        print(f"invalid configuration: {reason}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": ppo.plan(config)}, indent=2))
        print("nothing was trained: drop --dry-run to run the real thing", file=sys.stderr)
        return 0

    try:
        checkpoint = train(config)
    except ImportError:
        print(
            "stable-baselines3 is not installed: run `uv sync --group train` first",
            file=sys.stderr,
        )
        return 2
    print(f"checkpoint written to {checkpoint}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())

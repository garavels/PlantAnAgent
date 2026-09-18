"""`python -m plant_backend.rl.evaluation` -- the evaluation command.

    uv run python -m plant_backend.rl.evaluation --seeds 1 2 3 --hours 24
    uv run python -m plant_backend.rl.evaluation --checkpoint runs/ppo-1/checkpoint --json out.json

Prints the summary table and, with `--json`, writes the full machine-readable
result. Without a checkpoint it evaluates the heuristic baseline only and says so.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..checkpoints import CheckpointError
from ..episodes import EpisodeConfig
from .compare import compare, write


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plant_backend.rl.evaluation",
        description="Compare the heuristic controller with an RL checkpoint, deterministically.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="episode seeds; both controllers see every seed (default: 1 2 3)",
    )
    parser.add_argument("--hours", type=float, default=24.0, help="episode horizon in simulated hours")
    parser.add_argument(
        "--scenarios", nargs="+", help="scenario ids to draw episodes from (default: the RL pool)"
    )
    parser.add_argument("--checkpoint", type=Path, help="RL checkpoint directory to evaluate")
    parser.add_argument("--json", type=Path, help="write the full result here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings: dict[str, object] = {"horizon_h": args.hours}
    if args.scenarios:
        settings["scenarios"] = args.scenarios
    try:
        episode = EpisodeConfig.from_dict({**EpisodeConfig().to_dict(), **settings})
        result = compare(seeds=args.seeds, checkpoint=args.checkpoint, episode=episode)
    except CheckpointError as reason:
        print(f"checkpoint unusable: {reason}", file=sys.stderr)
        return 2
    except ValueError as reason:
        print(f"invalid evaluation: {reason}", file=sys.stderr)
        return 2

    print(result.report())
    if args.json:
        target = write(result, args.json)
        print(f"\nwritten to {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())

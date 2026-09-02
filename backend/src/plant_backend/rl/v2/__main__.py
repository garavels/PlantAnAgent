"""Run the auditable delivery-alpha baseline calibration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import EpisodeConfigV2
from .evaluation import evaluate, write


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m plant_backend.rl.v2",
        description="Evaluate transparent baselines on Environment V2 delivery alpha.",
    )
    parser.add_argument("--seed-start", type=int, default=0, help="first episode seed")
    parser.add_argument("--count", type=int, default=20, help="number of consecutive seeds")
    parser.add_argument("--hours", type=float, default=168.0, help="episode horizon in hours")
    parser.add_argument(
        "--delivery-rate",
        type=float,
        default=55.0,
        help="contracted average methane delivery in kg/h",
    )
    parser.add_argument("--json", type=Path, help="write complete evidence to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.seed_start < 0:
            raise ValueError("seed-start must be non-negative")
        if args.count <= 0:
            raise ValueError("count must be positive")
        episode = EpisodeConfigV2(
            horizon_h=args.hours,
            delivery_rate_kg_h=args.delivery_rate,
        )
        result = evaluate(
            seeds=range(args.seed_start, args.seed_start + args.count),
            episode=episode,
        )
    except ValueError as reason:
        print(f"invalid V2 baseline evaluation: {reason}", file=sys.stderr)
        return 2

    print(result.report())
    if args.json:
        target = write(result, args.json)
        print(f"\nwritten to {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = ["build_parser", "main"]

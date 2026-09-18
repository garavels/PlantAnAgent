"""CLI for the causal commitment-alpha controller development comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import EpisodeConfigV2
from .controllers import (
    ClimatologyControllerV2,
    IssuedForecastControllerV2,
    RiskAdjustedIssuedForecastControllerV2,
)
from .evaluation import LegacyHeuristicControllerV2, evaluate_controllers, write

DEVELOPMENT_SEEDS = frozenset({0, 1, 2})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m plant_backend.rl.v2.compare",
        description=(
            "Compare transparent causal controllers on commitment-alpha V2. "
            "Outputs are development evidence, never held-out claims."
        ),
    )
    parser.add_argument("--seed-start", type=int, default=0, help="first episode seed")
    parser.add_argument("--count", type=int, default=3, help="number of consecutive seeds")
    parser.add_argument(
        "--hours",
        type=float,
        default=72.0,
        help="episode horizon in hours",
    )
    parser.add_argument(
        "--delivery-rate",
        type=float,
        default=35.0,
        help="contracted average methane delivery in kg/h",
    )
    parser.add_argument(
        "--allow-dirty-development",
        action="store_true",
        help=(
            "allow a content-identified but dirty worktree; never use for prospective "
            "or claim-bearing evidence"
        ),
    )
    parser.add_argument(
        "--acknowledge-seed-contamination",
        action="store_true",
        help=(
            "explicitly acknowledge that seeds outside the frozen development set "
            "will no longer be eligible as unseen holdouts"
        ),
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
        seeds = list(range(args.seed_start, args.seed_start + args.count))
        undeclared = sorted(set(seeds) - DEVELOPMENT_SEEDS)
        if undeclared and not args.acknowledge_seed_contamination:
            raise ValueError(
                "seeds outside the declared development set {0, 1, 2} require "
                "--acknowledge-seed-contamination; requested "
                + ", ".join(str(seed) for seed in undeclared)
            )
        episode = EpisodeConfigV2(
            horizon_h=args.hours,
            delivery_rate_kg_h=args.delivery_rate,
        )
        result = evaluate_controllers(
            seeds=seeds,
            controllers=(
                LegacyHeuristicControllerV2(),
                ClimatologyControllerV2(),
                IssuedForecastControllerV2(),
                RiskAdjustedIssuedForecastControllerV2(),
            ),
            episode=episode,
            require_clean_source=not args.allow_dirty_development,
        )
    except (TypeError, ValueError) as reason:
        print(f"invalid V2 controller comparison: {reason}", file=sys.stderr)
        return 2
    except RuntimeError as reason:
        print(f"V2 controller comparison failed closed: {reason}", file=sys.stderr)
        return 1

    try:
        print(result.report())
        if args.json:
            target = write(result, args.json)
            print(f"\nwritten to {target}")
    except (TypeError, ValueError) as reason:
        print(f"invalid V2 controller evidence: {reason}", file=sys.stderr)
        return 1
    if not result.execution_summary()["all_planned_arms_succeeded"]:
        print(
            "controller comparison is ineligible: at least one planned arm failed",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = ["DEVELOPMENT_SEEDS", "build_parser", "main"]

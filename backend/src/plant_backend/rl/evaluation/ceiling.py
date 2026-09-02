"""Diagnose whether Environment V1 is capped by its methane tank.

This module does not change the environment.  It rolls the heuristic and an
optional checkpoint through fresh :class:`PlantEnv` instances on identical
episode seeds, then records the physical quantity that the normal comparison
cannot expose: how much methane storage was available when the episode began.

If many runs finish close to a full tank and, among those runs, starting methane
is strongly negatively correlated with methane produced, controller returns are
probably being compressed by a storage ceiling.  That is evidence about the
benchmark, not proof that either controller is optimal.

Run it directly with::

    python -m plant_backend.rl.evaluation.ceiling --seed-start 1 --count 20
    python -m plant_backend.rl.evaluation.ceiling --checkpoint runs/ppo/checkpoint \
        --hours 48 --json out/ceiling.json

The JSON contains every rollout and population mean/std/min/max summaries, so a
diagnosis can be audited rather than inferred from a single headline number.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from plant_sim import PlantConfig, PlantState

from ...controllers.base import Constraints, Controller
from ...controllers.heuristic import HeuristicController
from ..checkpoints import CheckpointError
from ..env import PlantEnv
from ..episodes import EpisodeConfig
from ..metrics import EpisodeMetrics
from ..policies import LoadedPolicy, Policy, load_policy
from ..reward import RewardWeights

DEFAULT_TERMINAL_FILL_THRESHOLD = 0.96


@dataclass(frozen=True)
class Distribution:
    """Population summary for one scalar across all evaluated seeds."""

    mean: float
    std: float
    minimum: float
    maximum: float

    @classmethod
    def of(cls, values: Sequence[float]) -> "Distribution":
        if not values:
            raise ValueError("cannot summarise an empty sequence")
        numbers = [float(value) for value in values]
        return cls(
            mean=statistics.fmean(numbers),
            std=statistics.pstdev(numbers),
            minimum=min(numbers),
            maximum=max(numbers),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
        }


@dataclass
class CeilingRun:
    """One controller on one episode, including methane-tank boundary data."""

    controller: str
    seed: int
    scenario: str
    ch4_capacity_kg: float
    initial_ch4_kg: float
    final_ch4_kg: float
    theoretical_ch4_headroom_kg: float
    terminal_ch4_headroom_kg: float
    produced_ch4_kg: float
    terminal_fill: float
    reward_total: float
    metrics: EpisodeMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "controller": self.controller,
            "seed": self.seed,
            "scenario": self.scenario,
            "ch4_capacity_kg": self.ch4_capacity_kg,
            "initial_ch4_kg": self.initial_ch4_kg,
            "final_ch4_kg": self.final_ch4_kg,
            "theoretical_ch4_headroom_kg": self.theoretical_ch4_headroom_kg,
            "terminal_ch4_headroom_kg": self.terminal_ch4_headroom_kg,
            "produced_ch4_kg": self.produced_ch4_kg,
            "terminal_fill": self.terminal_fill,
            "reward_total": self.reward_total,
            "metrics": self.metrics.to_dict(),
        }


DIAGNOSTIC_FIELDS = (
    "ch4_capacity_kg",
    "initial_ch4_kg",
    "final_ch4_kg",
    "theoretical_ch4_headroom_kg",
    "terminal_ch4_headroom_kg",
    "produced_ch4_kg",
    "terminal_fill",
    "reward_total",
)


@dataclass
class CeilingDiagnostic:
    """All paired rollouts plus an explicitly scoped ceiling diagnosis."""

    seeds: list[int]
    horizon_h: float
    terminal_fill_threshold: float
    smoke_test: bool
    baseline_only: bool
    runs: list[CeilingRun] = field(default_factory=list)
    checkpoint: dict[str, Any] | None = None
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def controllers(self) -> list[str]:
        seen: list[str] = []
        for run in self.runs:
            if run.controller not in seen:
                seen.append(run.controller)
        return seen

    @property
    def evidence_scope(self) -> str:
        if self.baseline_only:
            return "baseline_only"
        return "smoke_test" if self.smoke_test else "trained_checkpoint_comparison"

    @property
    def disclaimer(self) -> str:
        if self.baseline_only:
            return (
                "Baseline-only ceiling diagnostic: no RL checkpoint was supplied, so this "
                "does not establish heuristic/RL parity."
            )
        if self.smoke_test:
            return (
                "Smoke test: the checkpoint metadata does not establish a trained policy; "
                "its row verifies evaluation wiring, not RL performance."
            )
        return (
            "The checkpoint metadata identifies a trained policy. Tank saturation can "
            "explain compressed returns, but this diagnostic does not prove optimality."
        )

    def summary(self) -> dict[str, Any]:
        return {
            controller: _summarise_controller(
                [run for run in self.runs if run.controller == controller],
                terminal_fill_threshold=self.terminal_fill_threshold,
            )
            for controller in self.controllers
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "diagnostic": "environment_v1_methane_tank_ceiling",
            "evidence_scope": self.evidence_scope,
            "smoke_test": self.smoke_test,
            "baseline_only": self.baseline_only,
            "disclaimer": self.disclaimer,
            "seeds": self.seeds,
            "horizon_h": self.horizon_h,
            "terminal_fill_threshold": self.terminal_fill_threshold,
            "statistics": {"standard_deviation": "population (ddof=0)"},
            "checkpoint": self.checkpoint,
            "labels": self.labels,
            "summary": self.summary(),
            "runs": [run.to_dict() for run in self.runs],
        }

    def report(self) -> str:
        """Human-readable evidence, including ranges rather than only averages."""
        lines = [
            "Environment V1 methane-tank ceiling diagnostic",
            (
                f"{len(self.seeds)} seed(s) x {self.horizon_h:g} h per controller; "
                f"ceiling criterion: terminal fill >= {self.terminal_fill_threshold:.1%}"
            ),
            f"Evidence scope: {self.evidence_scope}",
            self.disclaimer,
        ]
        for controller, summary in self.summary().items():
            label = self.labels.get(controller, controller)
            diagnostic = summary["diagnostics"]
            metrics = summary["metrics"]
            ceiling = summary["ceiling_limited"]
            correlation = ceiling["pearson_initial_vs_produced_ch4"]
            correlation_text = "n/a" if correlation is None else f"{correlation:+.3f}"
            lines += [
                "",
                f"{label} ({summary['runs']} run(s))",
                "  initial CH4:       " + _format_distribution(diagnostic["initial_ch4_kg"], "kg"),
                "  theoretical room: "
                + _format_distribution(diagnostic["theoretical_ch4_headroom_kg"], "kg"),
                "  produced CH4:      " + _format_distribution(diagnostic["produced_ch4_kg"], "kg"),
                "  final CH4:         " + _format_distribution(diagnostic["final_ch4_kg"], "kg"),
                "  terminal fill:     " + _format_distribution(diagnostic["terminal_fill"], "%"),
                "  reward:            " + _format_distribution(diagnostic["reward_total"]),
                (
                    "  ceiling-limited:  "
                    f"{ceiling['count']}/{summary['runs']} ({ceiling['fraction']:.1%}); "
                    "Pearson r(initial, produced)="
                    f"{correlation_text} on {ceiling['correlation_runs']} run(s)"
                ),
                (
                    "  operating metrics: "
                    f"solar use {metrics['solar_utilisation']['mean']:.1%}, "
                    f"curtailed {metrics['curtailed_kwh']['mean']:.1f} kWh, "
                    f"minimum SOC {metrics['min_battery_soc']['mean']:.1%}, "
                    f"idle {metrics['idle_fraction']['mean']:.1%}, "
                    f"corrections {metrics['corrections']['mean']:.1f}"
                ),
            ]
        lines += [
            "",
            (
                "Interpretation: a high ceiling-limited fraction together with a Pearson "
                "correlation near -1 supports the storage-cap hypothesis. It does not show "
                "that either controller is optimal."
            ),
        ]
        return "\n".join(lines)


def _format_distribution(values: dict[str, float], unit: str = "") -> str:
    if unit == "%":
        return (
            f"{values['mean']:.1%} +/- {values['std']:.1%} "
            f"[{values['min']:.1%}, {values['max']:.1%}]"
        )
    suffix = f" {unit}" if unit else ""
    return (
        f"{values['mean']:.2f} +/- {values['std']:.2f} "
        f"[{values['min']:.2f}, {values['max']:.2f}]{suffix}"
    )


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson's r, or ``None`` where fewer/constant values make it undefined."""
    if len(xs) != len(ys):
        raise ValueError("Pearson inputs must have equal length")
    if len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    centred_x = [float(value) - mean_x for value in xs]
    centred_y = [float(value) - mean_y for value in ys]
    variance_x = math.fsum(value * value for value in centred_x)
    variance_y = math.fsum(value * value for value in centred_y)
    denominator = math.sqrt(variance_x * variance_y)
    if denominator == 0.0:
        return None
    coefficient = math.fsum(
        left * right for left, right in zip(centred_x, centred_y)
    ) / denominator
    return min(1.0, max(-1.0, coefficient))


def _numeric_metrics(runs: Sequence[CeilingRun]) -> dict[str, dict[str, float]]:
    records = [run.metrics.to_dict() for run in runs]
    names = sorted(
        {
            name
            for record in records
            for name, value in record.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    return {
        name: Distribution.of([float(record[name]) for record in records]).to_dict()
        for name in names
    }


def _reward_components(runs: Sequence[CeilingRun]) -> dict[str, dict[str, float]]:
    records = [run.metrics.reward_components for run in runs]
    names = sorted({name for record in records for name in record})
    return {
        name: Distribution.of([float(record.get(name, 0.0)) for record in records]).to_dict()
        for name in names
    }


def _summarise_controller(
    runs: Sequence[CeilingRun], *, terminal_fill_threshold: float
) -> dict[str, Any]:
    if not runs:
        raise ValueError("cannot summarise a controller without runs")
    limited = [run for run in runs if run.terminal_fill >= terminal_fill_threshold]
    correlation = _pearson(
        [run.initial_ch4_kg for run in limited],
        [run.produced_ch4_kg for run in limited],
    )
    return {
        "runs": len(runs),
        "diagnostics": {
            name: Distribution.of([float(getattr(run, name)) for run in runs]).to_dict()
            for name in DIAGNOSTIC_FIELDS
        },
        "metrics": _numeric_metrics(runs),
        "reward_components": _reward_components(runs),
        "ceiling_limited": {
            "terminal_fill_threshold": terminal_fill_threshold,
            "count": len(limited),
            "fraction": len(limited) / len(runs),
            "correlation_runs": len(limited),
            "pearson_initial_vs_produced_ch4": correlation,
        },
    }


def _make_run(
    env: PlantEnv,
    *,
    controller: str,
    seed: int,
    initial: PlantState,
) -> CeilingRun:
    final = env.simulator.state
    capacity = float(initial.ch4_capacity_kg)
    produced = float(final.ch4_produced_total_kg - initial.ch4_produced_total_kg)
    return CeilingRun(
        controller=controller,
        seed=seed,
        scenario=env.simulator.scenario or "unknown",
        ch4_capacity_kg=capacity,
        initial_ch4_kg=float(initial.ch4_level_kg),
        final_ch4_kg=float(final.ch4_level_kg),
        theoretical_ch4_headroom_kg=max(0.0, capacity - initial.ch4_level_kg),
        terminal_ch4_headroom_kg=max(0.0, capacity - final.ch4_level_kg),
        produced_ch4_kg=produced,
        terminal_fill=float(final.ch4_fill),
        reward_total=float(env.metrics.reward_total),
        metrics=env.metrics,
    )


def run_controller(env: PlantEnv, controller: Controller, *, seed: int) -> CeilingRun:
    """Roll out a controller through PlantEnv and retain true boundary states."""
    env.reset(seed=seed)
    initial = env.simulator.state
    controller.reset()
    finished = False
    while not finished:
        decision = controller.compute_action(
            env.telemetry, constraints=env.constraints, config=env.config
        )
        _, _, terminated, truncated, _ = env.step_action(decision.action)
        finished = terminated or truncated
    return _make_run(env, controller=controller.id, seed=seed, initial=initial)


def run_policy(env: PlantEnv, policy: Policy, *, seed: int) -> CeilingRun:
    """Roll out deterministic policy inference through the same PlantEnv path."""
    observation, _ = env.reset(seed=seed)
    initial = env.simulator.state
    finished = False
    while not finished:
        observation, _, terminated, truncated, _ = env.step(policy.act(observation))
        finished = terminated or truncated
    return _make_run(env, controller="rl", seed=seed, initial=initial)


def diagnose(
    *,
    seeds: Sequence[int],
    checkpoint: Path | str | None = None,
    episode: EpisodeConfig | None = None,
    plant: PlantConfig | None = None,
    constraints: Constraints | None = None,
    weights: RewardWeights | None = None,
    terminal_fill_threshold: float = DEFAULT_TERMINAL_FILL_THRESHOLD,
) -> CeilingDiagnostic:
    """Evaluate the V1 ceiling on identical seeds without mutating the environment."""
    chosen_seeds = [int(seed) for seed in seeds]
    if not chosen_seeds:
        raise ValueError("at least one seed is required")
    if not math.isfinite(terminal_fill_threshold) or not 0.0 <= terminal_fill_threshold <= 1.0:
        raise ValueError("terminal fill threshold must be between 0 and 1")

    episode = (episode or EpisodeConfig()).validate()
    plant = plant or PlantConfig()
    constraints = constraints or Constraints()
    weights = weights or RewardWeights()
    loaded: LoadedPolicy | None = load_policy(checkpoint) if checkpoint else None

    labels = {"heuristic": "Heuristic"}
    if loaded is not None:
        labels["rl"] = (
            "RL policy"
            if loaded.metadata.trained
            else f"RL mock/untrained ({loaded.metadata.algorithm})"
        )
    result = CeilingDiagnostic(
        seeds=chosen_seeds,
        horizon_h=episode.horizon_h,
        terminal_fill_threshold=float(terminal_fill_threshold),
        smoke_test=loaded is None or not loaded.metadata.trained,
        baseline_only=loaded is None,
        checkpoint=None if loaded is None else loaded.to_dict(),
        labels=labels,
    )

    def make_env() -> PlantEnv:
        return PlantEnv(
            config=plant,
            episode=episode,
            constraints=constraints,
            weights=weights,
        )

    for seed in chosen_seeds:
        result.runs.append(run_controller(make_env(), HeuristicController(), seed=seed))
        if loaded is not None:
            result.runs.append(run_policy(make_env(), loaded.policy, seed=seed))
    return result


def write(result: CeilingDiagnostic, path: Path | str) -> Path:
    """Write deterministic, machine-readable evidence for later comparison."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m plant_backend.rl.evaluation.ceiling",
        description="Diagnose methane-tank saturation in Environment V1.",
    )
    parser.add_argument("--seed-start", type=int, default=1, help="first episode seed (default: 1)")
    parser.add_argument("--count", type=int, default=20, help="number of consecutive seeds")
    parser.add_argument("--hours", type=float, default=48.0, help="episode horizon in hours")
    parser.add_argument("--checkpoint", type=Path, help="optional RL checkpoint directory")
    parser.add_argument("--json", type=Path, help="write complete evidence to this JSON file")
    parser.add_argument(
        "--terminal-fill-threshold",
        type=float,
        default=DEFAULT_TERMINAL_FILL_THRESHOLD,
        help="terminal CH4 fill counted as ceiling-limited (default: 0.96)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.count <= 0:
            raise ValueError("count must be positive")
        seeds = list(range(args.seed_start, args.seed_start + args.count))
        result = diagnose(
            seeds=seeds,
            checkpoint=args.checkpoint,
            episode=EpisodeConfig(horizon_h=args.hours),
            terminal_fill_threshold=args.terminal_fill_threshold,
        )
    except CheckpointError as reason:
        print(f"checkpoint unusable: {reason}", file=sys.stderr)
        return 2
    except ValueError as reason:
        print(f"invalid ceiling diagnostic: {reason}", file=sys.stderr)
        return 2

    print(result.report())
    if args.json:
        target = write(result, args.json)
        print(f"\nwritten to {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "DEFAULT_TERMINAL_FILL_THRESHOLD",
    "CeilingDiagnostic",
    "CeilingRun",
    "Distribution",
    "build_parser",
    "diagnose",
    "main",
    "run_controller",
    "run_policy",
    "write",
]

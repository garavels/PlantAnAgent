"""Heuristic against policy, on the same plant, the same seeds, the same days.

    uv run python -m plant_backend.rl.evaluation --seeds 1 2 3
    uv run python -m plant_backend.rl.evaluation --checkpoint runs/ppo-1/checkpoint

A comparison is only worth reading if the two controllers were given the same
problem, so both are rolled out through `PlantEnv` on the same seed list: the seed
draws the scenario, the starting inventories and the weather, and each controller
gets its own freshly reset plant from that same draw. Both go through
`safety.validate()`, both are scored by the same reward and measured by the same
`EpisodeMetrics`.

What this cannot do is report a real result, because **no policy has been trained
in this repository yet**. Without `--checkpoint` it evaluates the heuristic
baseline alone. With a `fixed` (mock) checkpoint it runs both sides and marks the
output `smoke_test: true`, because a constant action is a wiring check, not a
contender. Only a checkpoint that declares real training makes the comparison a
comparison, and the flag in the output is what says which of the three happened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from plant_sim import PlantConfig

from ...controllers.base import Constraints, Controller
from ...controllers.heuristic import HeuristicController
from .. import metrics as metrics_module
from ..env import PlantEnv
from ..episodes import EpisodeConfig
from ..policies import LoadedPolicy, Policy, load_policy
from ..reward import RewardWeights


@dataclass
class RunResult:
    """One controller on one seed."""

    controller: str
    seed: int
    scenario: str
    metrics: metrics_module.EpisodeMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "controller": self.controller,
            "seed": self.seed,
            "scenario": self.scenario,
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class Comparison:
    """Every run, aggregated per controller, plus what the numbers do and do not mean."""

    #: True when neither side is a trained policy: a wiring check, not a result.
    smoke_test: bool
    seeds: list[int]
    horizon_h: float
    runs: list[RunResult] = field(default_factory=list)
    checkpoint: dict[str, Any] | None = None
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def controllers(self) -> list[str]:
        seen: list[str] = []
        for run in self.runs:
            if run.controller not in seen:
                seen.append(run.controller)
        return seen

    def summary(self) -> dict[str, Any]:
        return {
            name: metrics_module.aggregate(
                [run.metrics for run in self.runs if run.controller == name]
            )
            for name in self.controllers
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "smoke_test": self.smoke_test,
            "disclaimer": (
                "Smoke test: no trained policy is involved, so these numbers compare "
                "wiring, not controllers."
                if self.smoke_test
                else "Comparison of the heuristic baseline against a trained checkpoint."
            ),
            "seeds": self.seeds,
            "horizon_h": self.horizon_h,
            "checkpoint": self.checkpoint,
            "labels": self.labels,
            "summary": self.summary(),
            "runs": [run.to_dict() for run in self.runs],
        }

    def report(self) -> str:
        """The human-readable version: one line of headline metrics per controller."""
        summary = self.summary()
        header = (
            f"{len(self.seeds)} seed(s) x {self.horizon_h:.0f} h per controller"
            f"{'  [SMOKE TEST -- no trained policy]' if self.smoke_test else ''}"
        )
        columns = (
            ("CH4 kg", "ch4_kg", "{:.1f}"),
            ("solar use", "solar_utilisation", "{:.1%}"),
            ("curtailed kWh", "curtailed_kwh", "{:.0f}"),
            ("min SOC", "min_battery_soc", "{:.1%}"),
            ("idle", "idle_fraction", "{:.1%}"),
            ("corrections", "corrections", "{:.1f}"),
            ("cycles", "cycles", "{:.1f}"),
            ("reward", "reward_total", "{:.1f}"),
        )
        lines = [header, ""]
        lines.append("controller".ljust(14) + "".join(name.rjust(15) for name, _, _ in columns))
        for controller in self.controllers:
            values = summary[controller]
            label = self.labels.get(controller, controller)
            lines.append(
                label[:13].ljust(14)
                + "".join(shape.format(values[key]).rjust(15) for _, key, shape in columns)
            )
        if self.smoke_test:
            lines += [
                "",
                "No policy has been trained yet: the RL row above is either absent or a",
                "constant-action mock. Performance numbers follow the first real training run.",
            ]
        return "\n".join(lines)


def _make_env(
    *,
    plant: PlantConfig,
    episode: EpisodeConfig,
    constraints: Constraints,
    weights: RewardWeights,
) -> PlantEnv:
    return PlantEnv(
        config=plant, episode=episode, constraints=constraints, weights=weights
    )


def run_controller(env: PlantEnv, controller: Controller, *, seed: int) -> RunResult:
    """Roll out a `Controller` for one episode through the env's own step path."""
    env.reset(seed=seed)
    controller.reset()
    truncated = False
    while not truncated:
        decision = controller.compute_action(
            env.telemetry, constraints=env.constraints, config=env.config
        )
        _, _, terminated, truncated, _ = env.step_action(decision.action)
        truncated = truncated or terminated
    return RunResult(
        controller=controller.id,
        seed=seed,
        scenario=env.simulator.scenario or "unknown",
        metrics=env.metrics,
    )


def run_policy(env: PlantEnv, policy: Policy, *, seed: int) -> RunResult:
    """Roll out a policy for one episode. Deterministic inference, no exploration."""
    observation, _ = env.reset(seed=seed)
    truncated = False
    while not truncated:
        observation, _, terminated, truncated, _ = env.step(policy.act(observation))
        truncated = truncated or terminated
    return RunResult(
        controller="rl",
        seed=seed,
        scenario=env.simulator.scenario or "unknown",
        metrics=env.metrics,
    )


def compare(
    *,
    seeds: list[int],
    checkpoint: Path | str | None = None,
    episode: EpisodeConfig | None = None,
    plant: PlantConfig | None = None,
    constraints: Constraints | None = None,
    weights: RewardWeights | None = None,
) -> Comparison:
    """Evaluate the heuristic, and the checkpoint's policy when one is supplied."""
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
            else f"RL mock ({loaded.metadata.algorithm})"
        )

    result = Comparison(
        smoke_test=loaded is None or not loaded.metadata.trained,
        seeds=list(seeds),
        horizon_h=episode.horizon_h,
        checkpoint=None if loaded is None else loaded.to_dict(),
        labels=labels,
    )
    for seed in seeds:
        env = _make_env(plant=plant, episode=episode, constraints=constraints, weights=weights)
        result.runs.append(run_controller(env, HeuristicController(), seed=seed))
        if loaded is not None:
            policy_env = _make_env(
                plant=plant, episode=episode, constraints=constraints, weights=weights
            )
            result.runs.append(run_policy(policy_env, loaded.policy, seed=seed))
    return result


def write(result: Comparison, path: Path | str) -> Path:
    """Machine-readable results, for a later run to be read against."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return target


__all__ = [
    "Comparison",
    "RunResult",
    "compare",
    "run_controller",
    "run_policy",
    "write",
]

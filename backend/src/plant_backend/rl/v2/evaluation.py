"""Paired baseline evaluation for the delivery-only Environment V2 alpha.

This module deliberately evaluates *only* transparent, non-learning baselines.
It establishes that the task and its accounting can distinguish behavior before
we spend compute training a policy.  A result from this module is not evidence
that RL works, or that any controller is commercially superior.
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from plant_sim import Action, PlantConfig

from ...controllers.base import Constraints
from ...controllers.heuristic import HeuristicController
from . import spaces
from .config import EpisodeConfigV2
from .contract import describe_checkpoint_contract
from .env import PlantEnvV2
from .metrics import EpisodeMetricsV2
from .reward import RewardWeightsV2

BASELINE_ONLY_DISCLAIMER = (
    "BASELINE ONLY: no trained RL policy, MPC, or perfect-foresight oracle is "
    "included. These results validate task discrimination and evaluation wiring; "
    "they do not establish RL performance, optimality, or commercial superiority."
)


@dataclass(frozen=True)
class Baseline:
    """One transparent controller arm in the alpha calibration tournament."""

    id: str
    label: str
    action: Action | None = None


BASELINES: tuple[Baseline, ...] = (
    Baseline("heuristic", "Shipped heuristic"),
    Baseline("idle", "Idle", Action.idle()),
    Baseline(
        "balanced",
        "Fixed balanced mix",
        Action(
            battery_share=0.20,
            electrolyser_share=0.50,
            dac_share=0.20,
            methanator_share=0.10,
        ),
    ),
    Baseline(
        "production_mix",
        "Fixed production mix",
        Action(
            battery_share=0.05,
            electrolyser_share=0.65,
            dac_share=0.20,
            methanator_share=0.10,
        ),
    ),
)


@dataclass(frozen=True)
class Distribution:
    """Population summary for one scalar metric."""

    count: int
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
            count=len(numbers),
            mean=statistics.fmean(numbers),
            std=statistics.pstdev(numbers),
            minimum=min(numbers),
            maximum=max(numbers),
        )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
        }


@dataclass(frozen=True)
class BaselineRun:
    """One baseline on one episode, tied to its immutable replay manifest."""

    controller: str
    seed: int
    manifest_sha256: str
    metrics: EpisodeMetricsV2

    def to_dict(self) -> dict[str, Any]:
        return {
            "controller": self.controller,
            "seed": self.seed,
            "manifest_sha256": self.manifest_sha256,
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class BaselineEvaluation:
    """Paired baseline runs, their distributions, and offline provenance."""

    seeds: list[int]
    episode_config: EpisodeConfigV2
    provenance: dict[str, Any]
    runs: list[BaselineRun] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def controllers(self) -> list[str]:
        seen: list[str] = []
        for run in self.runs:
            if run.controller not in seen:
                seen.append(run.controller)
        return seen

    def pairing(self) -> dict[str, Any]:
        manifest_hashes: dict[int, set[str]] = {seed: set() for seed in self.seeds}
        arms_by_seed: dict[int, set[str]] = {seed: set() for seed in self.seeds}
        for run in self.runs:
            manifest_hashes.setdefault(run.seed, set()).add(run.manifest_sha256)
            arms_by_seed.setdefault(run.seed, set()).add(run.controller)
        expected = set(self.labels)
        return {
            "common_random_numbers": True,
            "manifest_match_by_seed": all(
                len(manifest_hashes.get(seed, set())) == 1 for seed in self.seeds
            ),
            "all_arms_present_by_seed": all(
                arms_by_seed.get(seed, set()) == expected for seed in self.seeds
            ),
            "manifest_sha256_by_seed": {
                str(seed): sorted(manifest_hashes.get(seed, set()))
                for seed in self.seeds
            },
        }

    def summary(self) -> dict[str, Any]:
        return {
            controller: _summarise(
                [run.metrics for run in self.runs if run.controller == controller]
            )
            for controller in self.controllers
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "study": "environment_v2_delivery_alpha_paired_baselines",
            "evidence_scope": "baseline_only",
            "disclaimer": BASELINE_ONLY_DISCLAIMER,
            "statistics": {"standard_deviation": "population (ddof=0)"},
            "seeds": list(self.seeds),
            "episode_config": self.episode_config.to_dict(),
            "labels": dict(self.labels),
            "pairing": self.pairing(),
            "provenance": self.provenance,
            "summary": self.summary(),
            "runs": [run.to_dict() for run in self.runs],
        }

    def report(self) -> str:
        summary = self.summary()
        columns = (
            ("fill", "delivery_fill_rate", "{:.1%}"),
            ("unmet kg", "unmet_kg", "{:.1f}"),
            ("service", "commitment_service_level", "{:.1%}"),
            ("CH4 kg", "ch4_produced_kg", "{:.1f}"),
            ("curtail kWh", "curtailed_kwh", "{:.0f}"),
            ("min SOC", "min_battery_soc", "{:.1%}"),
            ("guard", "guard_interventions", "{:.0f}"),
            ("guard kWh", "guard_intervention_kwh", "{:.0f}"),
            ("reward", "reward_total", "{:.2f}"),
        )
        lines = [
            "Environment V2 delivery-alpha paired baseline evaluation",
            (
                f"{len(self.seeds)} seed(s) x {self.episode_config.horizon_h:g} h; "
                f"contract {self.episode_config.delivery_rate_kg_h:g} kg/h"
            ),
            BASELINE_ONLY_DISCLAIMER,
            "",
            "controller".ljust(22)
            + "".join(label.rjust(14) for label, _, _ in columns),
        ]
        for controller in self.controllers:
            metrics = summary[controller]["metrics"]
            lines.append(
                self.labels.get(controller, controller)[:21].ljust(22)
                + "".join(
                    formatter.format(metrics[name]["mean"]).rjust(14)
                    for _, name, formatter in columns
                )
            )
        pairing = self.pairing()
        lines += [
            "",
            (
                "Pairing audit: "
                f"manifest hashes {'match' if pairing['manifest_match_by_seed'] else 'DO NOT MATCH'}; "
                f"all arms {'present' if pairing['all_arms_present_by_seed'] else 'MISSING'}"
            ),
        ]
        return "\n".join(lines)


def _summarise(metrics: Sequence[EpisodeMetricsV2]) -> dict[str, Any]:
    if not metrics:
        raise ValueError("cannot summarise an empty controller arm")
    records = [item.to_dict() for item in metrics]
    numeric_names = sorted(
        {
            name
            for record in records
            for name, value in record.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    reward_names = sorted(
        {
            name
            for record in records
            for name in record.get("reward_components", {})
        }
    )
    return {
        "runs": len(records),
        "metrics": {
            name: Distribution.of([float(record[name]) for record in records]).to_dict()
            for name in numeric_names
        },
        "reward_components": {
            name: Distribution.of(
                [float(record.get("reward_components", {}).get(name, 0.0)) for record in records]
            ).to_dict()
            for name in reward_names
        },
    }


def _repository_root(start: Path | None = None) -> Path | None:
    candidate = (start or Path(__file__)).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for parent in (candidate, *candidate.parents):
        if (parent / ".git").exists() and (parent / "uv.lock").is_file():
            return parent
    return None


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def collect_provenance(
    *,
    episode: EpisodeConfigV2,
    plant: PlantConfig,
    constraints: Constraints,
    weights: RewardWeightsV2,
    root: Path | None = None,
) -> dict[str, Any]:
    """Collect source and lock identity without requiring network access."""
    repository = root or _repository_root()
    revision = None
    dirty: bool | None = None
    lock: dict[str, str | None] = {"path": None, "sha256": None}
    if repository is not None:
        revision = _git(repository, "rev-parse", "HEAD")
        status = _git(repository, "status", "--porcelain", "--untracked-files=normal")
        dirty = None if status is None else bool(status)
        lock_path = repository / "uv.lock"
        if lock_path.is_file():
            lock = {
                "path": str(lock_path.relative_to(repository)),
                "sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
            }
    return {
        "source": {
            "repository_root": None if repository is None else str(repository),
            "git_revision": revision,
            "git_dirty": dirty,
        },
        "dependency_lock": lock,
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "environment": spaces.describe(),
        "checkpoint_contract": describe_checkpoint_contract(
            plant=plant,
            episode=episode,
            constraints=constraints,
            weights=weights,
        ),
        "episode_config": episode.to_dict(),
        "plant_config": plant.to_dict(),
        "constraints": constraints.to_dict(),
        "reward_weights": weights.to_dict(),
    }


def run_baseline(
    env: PlantEnvV2,
    baseline: Baseline,
    *,
    seed: int,
) -> BaselineRun:
    """Roll one transparent arm through the same guarded environment path."""
    env.reset(seed=seed)
    manifest_sha256 = str(env.audit_manifest()["manifest_sha256"])
    heuristic = HeuristicController() if baseline.action is None else None
    if heuristic is not None:
        heuristic.reset()
    finished = False
    while not finished:
        if heuristic is None:
            assert baseline.action is not None
            action = baseline.action
        else:
            action = heuristic.compute_action(
                env.telemetry,
                constraints=env.constraints,
                config=env.config,
            ).action
        _, _, terminated, truncated, _ = env.step_action(action)
        finished = terminated or truncated
    return BaselineRun(
        controller=baseline.id,
        seed=int(seed),
        manifest_sha256=manifest_sha256,
        metrics=env.metrics,
    )


def evaluate(
    *,
    seeds: Sequence[int],
    episode: EpisodeConfigV2 | None = None,
    plant: PlantConfig | None = None,
    constraints: Constraints | None = None,
    weights: RewardWeightsV2 | None = None,
    baselines: Sequence[Baseline] = BASELINES,
) -> BaselineEvaluation:
    """Run paired common-random-number baselines and retain full audit identity."""
    chosen_seeds = [int(seed) for seed in seeds]
    if not chosen_seeds:
        raise ValueError("at least one seed is required")
    if any(seed < 0 for seed in chosen_seeds):
        raise ValueError("seeds must be non-negative")
    if len(set(chosen_seeds)) != len(chosen_seeds):
        raise ValueError("seeds must be unique")
    chosen_baselines = tuple(baselines)
    if not chosen_baselines:
        raise ValueError("at least one baseline is required")
    ids = [baseline.id for baseline in chosen_baselines]
    if len(set(ids)) != len(ids):
        raise ValueError("baseline ids must be unique")

    plant = plant or PlantConfig()
    episode = (episode or EpisodeConfigV2()).validate(plant)
    constraints = constraints or Constraints()
    weights = weights or RewardWeightsV2()
    result = BaselineEvaluation(
        seeds=chosen_seeds,
        episode_config=episode,
        provenance=collect_provenance(
            episode=episode,
            plant=plant,
            constraints=constraints,
            weights=weights,
        ),
        labels={baseline.id: baseline.label for baseline in chosen_baselines},
    )
    for seed in chosen_seeds:
        for baseline in chosen_baselines:
            env = PlantEnvV2(
                config=plant,
                episode=episode,
                constraints=constraints,
                weights=weights,
            )
            result.runs.append(run_baseline(env, baseline, seed=seed))

    pairing = result.pairing()
    if not pairing["manifest_match_by_seed"] or not pairing["all_arms_present_by_seed"]:
        raise RuntimeError("paired baseline audit failed")
    return result


def write(result: BaselineEvaluation, path: Path | str) -> Path:
    """Write complete machine-readable evidence for later comparison."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return target


__all__ = [
    "BASELINES",
    "BASELINE_ONLY_DISCLAIMER",
    "Baseline",
    "BaselineEvaluation",
    "BaselineRun",
    "Distribution",
    "collect_provenance",
    "evaluate",
    "run_baseline",
    "write",
]

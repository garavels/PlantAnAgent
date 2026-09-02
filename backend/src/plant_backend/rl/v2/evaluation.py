"""Paired transparent-controller evaluation for Environment V2.

The frozen baseline study and newer typed causal comparators share the same
common-random-number, provenance, metrics, and guarded-execution machinery.
Neither development result is evidence that RL works, that a controller is
optimal, or that this prototype is commercially superior.
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import subprocess
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Literal, Sequence

from plant_sim import Action, ElectrolyserCommitmentProfile, PlantConfig

from ...controllers.base import Constraints, ControlDecision
from ...controllers.heuristic import HeuristicController
from . import spaces
from .config import EpisodeConfigV2
from .control import ControllerObservationV2, ControllerV2
from .contract import describe_checkpoint_contract
from .env import PlantEnvV2
from .metrics import EpisodeMetricsV2
from .reward import RewardWeightsV2

BASELINE_ONLY_DISCLAIMER = (
    "BASELINE ONLY: no forecast-aware controller, trained RL policy, MPC, or "
    "perfect-information planner is included, and these legacy/open-loop diagnostic "
    "arms do not share the full V2 information set. Results validate commitment-task "
    "discrimination and evaluation wiring; they do not establish controller "
    "optimality, RL performance, or commercial superiority."
)
LEGACY_INFORMATION_SET = (
    "latest one-minute V2 telemetry plus safety constraints; no preceding-interval "
    "aggregate, delivery context, or issued forecast"
)
OPEN_LOOP_INFORMATION_SET = "constant action; observes no environment information"
RESULT_SCHEMA_VERSION = 3
CAUSAL_RESULT_SCHEMA_VERSION = 4
CAUSAL_COMPARATOR_DISCLAIMER = (
    "DEVELOPMENT COMPARISON ONLY: transparent causal controllers are evaluated on "
    "development seeds selected during task calibration. Results are not held out, "
    "do not establish optimality, and are not commercial-performance evidence."
)
CAUSAL_INFORMATION_FAIRNESS = (
    "Controllers share one detached causal source and identical actuation/safety paths; "
    "forecast access differs only where declared. The structured source is richer than "
    "the current 57-field RL encoding, so this is not yet an equal-representation RL "
    "comparison."
)


@dataclass(frozen=True)
class Baseline:
    """One transparent controller arm in the alpha calibration tournament."""

    id: str
    label: str
    kind: Literal["legacy_heuristic", "fixed_action"]
    action: Action | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.label:
            raise ValueError("baseline id and label must not be empty")
        if self.kind not in ("legacy_heuristic", "fixed_action"):
            raise ValueError(f"unsupported baseline kind: {self.kind!r}")
        if self.kind == "legacy_heuristic" and self.action is not None:
            raise ValueError("a legacy_heuristic baseline cannot define a fixed action")
        if self.kind == "fixed_action" and self.action is None:
            raise ValueError("a fixed_action baseline requires an action")

    @property
    def information_set(self) -> str:
        return (
            LEGACY_INFORMATION_SET
            if self.kind == "legacy_heuristic"
            else OPEN_LOOP_INFORMATION_SET
        )


BASELINES: tuple[Baseline, ...] = (
    Baseline(
        "heuristic",
        "Legacy forecast-blind heuristic",
        "legacy_heuristic",
    ),
    Baseline("idle", "Idle", "fixed_action", Action.idle()),
    Baseline(
        "balanced",
        "Fixed balanced mix",
        "fixed_action",
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
        "fixed_action",
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
    exogenous_episode_sha256: str
    metrics: EpisodeMetricsV2

    def to_dict(self) -> dict[str, Any]:
        return {
            "controller": self.controller,
            "seed": self.seed,
            "manifest_sha256": self.manifest_sha256,
            "exogenous_episode_sha256": self.exogenous_episode_sha256,
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class BaselineEvaluation:
    """Paired controller runs, their distributions, and offline provenance.

    The historical name is retained for compatibility with the frozen baseline
    evidence.  Configurable study metadata also lets the same audited container
    represent later transparent-controller comparisons without mislabelling
    them as baseline-only evidence.
    """

    seeds: list[int]
    episode_config: EpisodeConfigV2
    provenance: dict[str, Any]
    runs: list[BaselineRun] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    information_sets: dict[str, str] = field(default_factory=dict)
    controller_specs: dict[str, dict[str, object]] = field(default_factory=dict)
    study: str = "environment_v2_commitment_alpha_paired_baselines"
    evidence_scope: str = "baseline_only"
    disclaimer: str = BASELINE_ONLY_DISCLAIMER
    title: str = "Environment V2 commitment-alpha paired baseline evaluation"
    information_fairness: str = (
        "Diagnostic only; equal-information comparators remain open."
    )
    result_schema_version: int = RESULT_SCHEMA_VERSION

    @property
    def controllers(self) -> list[str]:
        seen: list[str] = []
        for run in self.runs:
            if run.controller not in seen:
                seen.append(run.controller)
        return seen

    def pairing(self) -> dict[str, Any]:
        manifest_hashes: dict[int, set[str]] = {seed: set() for seed in self.seeds}
        exogenous_hashes: dict[int, set[str]] = {seed: set() for seed in self.seeds}
        arms_by_seed: dict[int, set[str]] = {seed: set() for seed in self.seeds}
        for run in self.runs:
            manifest_hashes.setdefault(run.seed, set()).add(run.manifest_sha256)
            exogenous_hashes.setdefault(run.seed, set()).add(
                run.exogenous_episode_sha256
            )
            arms_by_seed.setdefault(run.seed, set()).add(run.controller)
        expected = set(self.labels)
        full_manifest_match = all(
            len(manifest_hashes.get(seed, set())) == 1 for seed in self.seeds
        )
        return {
            "common_random_numbers": True,
            "exogenous_episode_match_by_seed": all(
                len(exogenous_hashes.get(seed, set())) == 1 for seed in self.seeds
            ),
            # Kept as an explicit, stronger identity for arms which share the
            # complete environment configuration. Commitment on/off ablations
            # legitimately have different full manifests while retaining the
            # same exogenous episode identity.
            "full_manifest_match_by_seed": full_manifest_match,
            "manifest_match_by_seed": full_manifest_match,
            "all_arms_present_by_seed": all(
                arms_by_seed.get(seed, set()) == expected for seed in self.seeds
            ),
            "exogenous_episode_sha256_by_seed": {
                str(seed): sorted(exogenous_hashes.get(seed, set()))
                for seed in self.seeds
            },
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
        payload = {
            "schema_version": self.result_schema_version,
            "study": self.study,
            "evidence_scope": self.evidence_scope,
            "disclaimer": self.disclaimer,
            "statistics": {"standard_deviation": "population (ddof=0)"},
            "seeds": list(self.seeds),
            "episode_config": self.episode_config.to_dict(),
            "labels": dict(self.labels),
            "information_sets": dict(self.information_sets),
            "pairing": self.pairing(),
            "provenance": self.provenance,
            "summary": self.summary(),
            "runs": [run.to_dict() for run in self.runs],
        }
        if self.controller_specs:
            payload["controller_specs"] = deepcopy(self.controller_specs)
        return payload

    def report(self) -> str:
        summary = self.summary()
        columns = [
            ("fill", "delivery_fill_rate", "{:.1%}"),
            ("unmet kg", "unmet_kg", "{:.1f}"),
            ("service", "commitment_service_level", "{:.1%}"),
            ("CH4 kg", "ch4_produced_kg", "{:.1f}"),
            ("curtail kWh", "curtailed_kwh", "{:.0f}"),
            ("min SOC", "min_battery_soc", "{:.1%}"),
            ("guard", "guard_interventions", "{:.0f}"),
            ("guard kWh", "guard_intervention_kwh", "{:.0f}"),
            ("dispatch gap", "dispatch_gap_kwh", "{:.0f}"),
            ("reward", "reward_total", "{:.2f}"),
        ]
        if any(run.metrics.electrolyser_commitment_enabled for run in self.runs):
            columns.extend(
                (
                    ("cold starts", "electrolyser_cold_starts", "{:.1f}"),
                    ("forced stops", "electrolyser_forced_stops", "{:.1f}"),
                    ("start kWh", "electrolyser_startup_energy_kwh", "{:.0f}"),
                    (
                        "nonprod kWh",
                        "electrolyser_nonproductive_energy_kwh",
                        "{:.0f}",
                    ),
                    (
                        "ramp hits",
                        "electrolyser_ramp_limited_intervals",
                        "{:.1f}",
                    ),
                )
            )
        lines = [
            self.title,
            (
                f"{len(self.seeds)} seed(s) x {self.episode_config.horizon_h:g} h; "
                f"contract {self.episode_config.delivery_rate_kg_h:g} kg/h"
            ),
            self.disclaimer,
            f"Information fairness: {self.information_fairness}",
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
                "exogenous episode hashes "
                f"{'match' if pairing['exogenous_episode_match_by_seed'] else 'DO NOT MATCH'}; "
                "full manifest hashes "
                f"{'match' if pairing['full_manifest_match_by_seed'] else 'DO NOT MATCH'}; "
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


def _validated_seeds(seeds: Sequence[int]) -> list[int]:
    raw_seeds = list(seeds)
    if not raw_seeds:
        raise ValueError("at least one seed is required")
    if any(
        isinstance(seed, bool) or not isinstance(seed, Integral)
        for seed in raw_seeds
    ):
        raise ValueError("seeds must be non-negative integers")
    chosen = [int(seed) for seed in raw_seeds]
    if any(seed < 0 for seed in chosen):
        raise ValueError("seeds must be non-negative integers")
    if len(set(chosen)) != len(chosen):
        raise ValueError("seeds must be unique")
    return chosen


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
    manifest = env.audit_manifest()
    manifest_sha256 = str(manifest["manifest_sha256"])
    exogenous_episode_sha256 = str(manifest["exogenous_episode_sha256"])
    heuristic = HeuristicController() if baseline.kind == "legacy_heuristic" else None
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
        exogenous_episode_sha256=exogenous_episode_sha256,
        metrics=env.metrics,
    )


class LegacyHeuristicControllerV2:
    """Declared V2 adapter preserving the legacy heuristic information set."""

    id = "legacy_heuristic"
    label = "Legacy forecast-blind heuristic"
    description = (
        "V1 operator heuristic adapted without behavior changes; it receives only "
        "the latest one-minute boundary telemetry and active safety constraints."
    )
    uses_forecast = False
    information_set = LEGACY_INFORMATION_SET

    def __init__(self) -> None:
        self._controller = HeuristicController()

    def reset(self) -> None:
        self._controller.reset()

    def compute_action(
        self,
        observation: ControllerObservationV2,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: ElectrolyserCommitmentProfile | None = None,
    ) -> ControlDecision:
        # The commitment profile is deliberately ignored: changing the adapter
        # must not improve or otherwise alter the historical controller arm.
        del commitment_profile
        return self._controller.compute_action(
            observation.state.boundary,
            constraints=constraints,
            config=config,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "uses_forecast": self.uses_forecast,
            "information_set": self.information_set,
            "adapter": "legacy_v1_boundary_telemetry",
        }


def run_controller(
    env: PlantEnvV2,
    controller: ControllerV2,
    *,
    seed: int,
) -> BaselineRun:
    """Run one typed causal controller without exposing its environment."""
    if not isinstance(controller.uses_forecast, bool):
        raise TypeError("controller uses_forecast must be a bool")
    env.reset(seed=seed)
    manifest = env.audit_manifest()
    controller.reset()
    finished = False
    while not finished:
        observation = env.controller_observation
        if not controller.uses_forecast:
            observation = observation.without_forecast()
        decision = controller.compute_action(
            observation,
            constraints=env.constraints,
            config=env.config,
            commitment_profile=env.episode_config.electrolyser_commitment,
        )
        if not isinstance(decision, ControlDecision):
            raise TypeError("V2 controller must return a ControlDecision")
        _, _, terminated, truncated, _ = env.step_action(decision.action)
        finished = terminated or truncated
    return BaselineRun(
        controller=controller.id,
        seed=int(seed),
        manifest_sha256=str(manifest["manifest_sha256"]),
        exogenous_episode_sha256=str(manifest["exogenous_episode_sha256"]),
        metrics=env.metrics,
    )


def evaluate_controllers(
    *,
    seeds: Sequence[int],
    controllers: Sequence[ControllerV2],
    episode: EpisodeConfigV2 | None = None,
    plant: PlantConfig | None = None,
    constraints: Constraints | None = None,
    weights: RewardWeightsV2 | None = None,
) -> BaselineEvaluation:
    """Paired development comparison over the typed causal controller seam."""
    chosen_seeds = _validated_seeds(seeds)

    chosen_controllers = tuple(controllers)
    if not chosen_controllers:
        raise ValueError("at least one controller is required")
    ids: list[str] = []
    specs: dict[str, dict[str, object]] = {}
    for controller in chosen_controllers:
        controller_id = getattr(controller, "id", None)
        label = getattr(controller, "label", None)
        description = getattr(controller, "description", None)
        information_set = getattr(controller, "information_set", None)
        uses_forecast = getattr(controller, "uses_forecast", None)
        if not isinstance(controller_id, str) or not controller_id:
            raise ValueError("controller id must be a non-empty string")
        if not isinstance(label, str) or not label:
            raise ValueError(f"controller {controller_id!r} label must not be empty")
        if not isinstance(description, str) or not description:
            raise ValueError(
                f"controller {controller_id!r} description must not be empty"
            )
        if not isinstance(information_set, str) or not information_set:
            raise ValueError(
                f"controller {controller_id!r} information_set must not be empty"
            )
        if not isinstance(uses_forecast, bool):
            raise TypeError(
                f"controller {controller_id!r} uses_forecast must be a bool"
            )
        if any(
            not callable(getattr(controller, name, None))
            for name in ("compute_action", "reset", "to_dict")
        ):
            raise TypeError(f"controller {controller_id!r} does not implement ControllerV2")
        descriptor = controller.to_dict()
        if not isinstance(descriptor, dict):
            raise TypeError(f"controller {controller_id!r} descriptor must be a mapping")
        try:
            json.dumps(descriptor, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"controller {controller_id!r} descriptor must be strict JSON"
            ) from exc
        if descriptor.get("id") != controller_id:
            raise ValueError(
                f"controller {controller_id!r} descriptor id does not match"
            )
        declared_fields = {
            "label": label,
            "description": description,
            "uses_forecast": uses_forecast,
            "information_set": information_set,
        }
        for name, declared in declared_fields.items():
            if descriptor.get(name) != declared:
                raise ValueError(
                    f"controller {controller_id!r} descriptor {name} does not match"
                )
        ids.append(controller_id)
        specs[controller_id] = deepcopy(descriptor)
    if len(set(ids)) != len(ids):
        raise ValueError("controller ids must be unique")

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
        labels={controller.id: controller.label for controller in chosen_controllers},
        information_sets={
            controller.id: controller.information_set
            for controller in chosen_controllers
        },
        controller_specs=specs,
        study="environment_v2_commitment_alpha_causal_comparators",
        evidence_scope="development_comparator",
        disclaimer=CAUSAL_COMPARATOR_DISCLAIMER,
        title="Environment V2 commitment-alpha causal controller comparison",
        information_fairness=CAUSAL_INFORMATION_FAIRNESS,
        result_schema_version=CAUSAL_RESULT_SCHEMA_VERSION,
    )
    for seed in chosen_seeds:
        for prototype in chosen_controllers:
            env = PlantEnvV2(
                config=plant,
                episode=episode,
                constraints=constraints,
                weights=weights,
            )
            result.runs.append(
                run_controller(env, deepcopy(prototype), seed=seed)
            )

    pairing = result.pairing()
    if (
        not pairing["exogenous_episode_match_by_seed"]
        or not pairing["full_manifest_match_by_seed"]
        or not pairing["all_arms_present_by_seed"]
    ):
        raise RuntimeError("paired causal-controller audit failed")
    return result


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
    chosen_seeds = _validated_seeds(seeds)
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
        information_sets={
            baseline.id: baseline.information_set for baseline in chosen_baselines
        },
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
    if (
        not pairing["exogenous_episode_match_by_seed"]
        or not pairing["full_manifest_match_by_seed"]
        or not pairing["all_arms_present_by_seed"]
    ):
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
    "CAUSAL_COMPARATOR_DISCLAIMER",
    "CAUSAL_INFORMATION_FAIRNESS",
    "CAUSAL_RESULT_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "Baseline",
    "BaselineEvaluation",
    "BaselineRun",
    "Distribution",
    "LegacyHeuristicControllerV2",
    "collect_provenance",
    "evaluate",
    "evaluate_controllers",
    "run_baseline",
    "run_controller",
    "write",
]

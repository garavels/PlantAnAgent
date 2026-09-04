"""Paired transparent-controller evaluation for Environment V2.

The frozen baseline study and newer typed causal comparators share the same
common-random-number, provenance, metrics, and guarded-execution machinery.
Neither development result is evidence that RL works, that a controller is
optimal, or that this prototype is commercially superior.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import stat
import statistics
import subprocess
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Sequence, TypeGuard

from plant_sim import Action, ElectrolyserCommitmentProfile, PlantConfig

from ...controllers.base import Constraints, ControlDecision
from ...controllers.heuristic import HeuristicController
from . import spaces
from .config import EpisodeConfigV2
from .control import ControllerObservationV2, ControllerV2
from .contract import describe_checkpoint_contract
from .env import PlantEnvV2
from .metrics import EpisodeMetricsV2
from .mpc_execution import (
    MPCDecisionApplicationV1,
    MPCDecisionEvidenceV1,
    MPCDecisionFailureV2,
    SERVICE_MPC_RUN_EVIDENCE_FORMAT,
    build_mpc_run_evidence,
    capture_mpc_application,
)
from .mpc_schema import SERVICE_MPC_CONTROLLER_ID
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
RESULT_SCHEMA_VERSION = 4
CAUSAL_RESULT_SCHEMA_VERSION = 7
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

# A deliberately small, predeclared scorecard.  Every raw delta is treatment
# minus control.  ``benefit_delta`` flips the sign only for lower-is-better
# metrics so that positive always favours the treatment.  We do not assign a
# benefit direction to ambiguous inventory or production-volume diagnostics.
PAIRED_CONTRAST_METRICS: tuple[tuple[str, str, str], ...] = (
    ("delivery_fill_rate", "higher_is_better", "delivery fill rate"),
    (
        "commitment_service_level",
        "higher_is_better",
        "commitment service level",
    ),
    (
        "worst_commitment_fill_rate",
        "higher_is_better",
        "worst commitment fill rate",
    ),
    ("delivered_kg", "higher_is_better", "delivered methane (kg)"),
    ("unmet_kg", "lower_is_better", "unmet methane (kg)"),
    (
        "terminal_h2_kg",
        "diagnostic",
        "terminal hydrogen inventory (kg; no automatic benefit sign)",
    ),
    (
        "terminal_co2_kg",
        "diagnostic",
        "terminal carbon-dioxide inventory (kg; no automatic benefit sign)",
    ),
    (
        "terminal_ch4_kg",
        "diagnostic",
        "terminal methane inventory (kg; no automatic benefit sign)",
    ),
    (
        "terminal_battery_soc",
        "diagnostic",
        "terminal battery state of charge (no automatic benefit sign)",
    ),
    (
        "terminal_battery_energy_kwh",
        "diagnostic",
        "terminal battery energy (kWh; no automatic benefit sign)",
    ),
    ("curtailed_kwh", "lower_is_better", "curtailed energy (kWh)"),
    (
        "accepted_process_energy_per_delivered_kg",
        "lower_is_better",
        "accepted process energy per delivered methane (kWh/kg)",
    ),
    (
        "battery_discharged_kwh",
        "diagnostic",
        "battery discharge (kWh; no automatic benefit sign)",
    ),
    (
        "battery_throughput_kwh",
        "diagnostic",
        "battery bus-side throughput (kWh; no automatic benefit sign)",
    ),
    (
        "dispatch_gap_kwh",
        "diagnostic",
        "plant routing/refusal/overlap distance (kWh; no automatic benefit sign)",
    ),
    (
        "request_application_gap_kwh",
        "diagnostic",
        "request-to-plant distance (kWh; no automatic benefit sign)",
    ),
    ("shutdown_steps", "lower_is_better", "shutdown physics steps"),
    ("guard_interventions", "lower_is_better", "guard interventions"),
    (
        "guard_intervention_kwh",
        "lower_is_better",
        "request-to-guard correction distance (kWh)",
    ),
    (
        "corrected_physics_steps",
        "lower_is_better",
        "corrected physics steps",
    ),
    (
        "out_of_range_decisions",
        "lower_is_better",
        "out-of-range decisions",
    ),
    (
        "electrolyser_cold_starts",
        "lower_is_better",
        "electrolyser cold starts",
    ),
    (
        "electrolyser_aborted_starts",
        "lower_is_better",
        "electrolyser aborted starts",
    ),
    (
        "electrolyser_forced_stops",
        "lower_is_better",
        "electrolyser forced stops",
    ),
    (
        "electrolyser_nonproductive_energy_kwh",
        "lower_is_better",
        "electrolyser nonproductive energy (kWh)",
    ),
)

_REPORT_CONTRAST_METRICS = {
    "delivery_fill_rate",
    "commitment_service_level",
    "worst_commitment_fill_rate",
    "unmet_kg",
    "terminal_h2_kg",
    "terminal_co2_kg",
    "terminal_ch4_kg",
    "terminal_battery_soc",
    "terminal_battery_energy_kwh",
    "accepted_process_energy_per_delivered_kg",
    "dispatch_gap_kwh",
    "request_application_gap_kwh",
    "guard_intervention_kwh",
    "electrolyser_forced_stops",
}

_TRACKED_DIFF_ARGS = (
    "diff",
    "--binary",
    "--full-index",
    "--no-color",
    "--no-ext-diff",
    "--no-textconv",
    "--no-renames",
    "--diff-algorithm=myers",
    "--src-prefix=a/",
    "--dst-prefix=b/",
    "HEAD",
    "--",
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
class ControllerFailureV2:
    """Sanitised, serialisable failure from one attempted controller run."""

    phase: str
    exception_type: str
    message: str
    completed_decisions: int
    elapsed_hours: float
    partial_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "exception_type": self.exception_type,
            "message": self.message,
            "completed_decisions": self.completed_decisions,
            "elapsed_hours": self.elapsed_hours,
            "partial_metrics": deepcopy(self.partial_metrics),
        }


@dataclass(frozen=True)
class RequestedActionV2:
    """One detached controller request before the shared guard and plant path."""

    decision_index: int
    battery_share: float
    electrolyser_share: float
    dac_share: float
    methanator_share: float
    battery_support_kw: float
    controller_invalid: bool

    @classmethod
    def from_action(
        cls,
        decision_index: int,
        action: Action,
        *,
        controller_invalid: bool = False,
    ) -> "RequestedActionV2":
        if isinstance(decision_index, bool) or not isinstance(
            decision_index,
            Integral,
        ):
            raise TypeError("requested-action decision index must be an integer")
        if decision_index < 0:
            raise ValueError("requested-action decision index must be non-negative")
        if not isinstance(action, Action):
            raise TypeError("controller decision action must be an Action")
        if not isinstance(controller_invalid, bool):
            raise TypeError("controller decision invalid flag must be a bool")
        values = (
            action.battery_share,
            action.electrolyser_share,
            action.dac_share,
            action.methanator_share,
            action.battery_support_kw,
        )
        if not all(_finite_numeric(value) for value in values):
            raise ValueError("requested controller action must contain finite numbers")
        return cls(
            decision_index=int(decision_index),
            battery_share=float(action.battery_share),
            electrolyser_share=float(action.electrolyser_share),
            dac_share=float(action.dac_share),
            methanator_share=float(action.methanator_share),
            battery_support_kw=float(action.battery_support_kw),
            controller_invalid=controller_invalid,
        )

    @property
    def share_vector(self) -> tuple[float, float, float, float]:
        return (
            self.battery_share,
            self.electrolyser_share,
            self.dac_share,
            self.methanator_share,
        )

    def to_dict(self) -> dict[str, bool | int | float]:
        return {
            "decision_index": self.decision_index,
            "battery_share": self.battery_share,
            "electrolyser_share": self.electrolyser_share,
            "dac_share": self.dac_share,
            "methanator_share": self.methanator_share,
            "battery_support_kw": self.battery_support_kw,
            "controller_invalid": self.controller_invalid,
        }


def _action_trace_payload(
    trace: Sequence[RequestedActionV2],
) -> dict[str, Any]:
    rows = [item.to_dict() for item in trace]
    encoded = json.dumps(
        rows,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "format": "controller-requested-action-v1",
        "count": len(rows),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "actions": rows,
    }


@dataclass(frozen=True)
class PairedContrastV2:
    """One predeclared paired treatment-minus-control comparison."""

    treatment: str
    control: str

    def __post_init__(self) -> None:
        if not self.treatment or not self.control:
            raise ValueError("contrast controller ids must not be empty")
        if self.treatment == self.control:
            raise ValueError("contrast treatment and control must differ")

    @property
    def id(self) -> str:
        return f"{self.treatment}_minus_{self.control}"

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "treatment": self.treatment,
            "control": self.control,
        }


@dataclass(frozen=True)
class BaselineRun:
    """One attempted arm, tied to its immutable replay manifest."""

    controller: str
    seed: int
    manifest_sha256: str
    exogenous_episode_sha256: str
    metrics: EpisodeMetricsV2 | None
    failure: ControllerFailureV2 | None = None
    requested_actions: tuple[RequestedActionV2, ...] | None = None
    controller_evidence: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.controller, str) or not self.controller:
            raise ValueError("run controller id must not be empty")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, Integral)
            or self.seed < 0
        ):
            raise ValueError("run seed must be a non-negative integer")
        for name, digest in (
            ("manifest_sha256", self.manifest_sha256),
            ("exogenous_episode_sha256", self.exogenous_episode_sha256),
        ):
            try:
                valid_digest = isinstance(digest, str) and len(digest) == 64
                if valid_digest:
                    int(digest, 16)
            except ValueError:
                valid_digest = False
            if not valid_digest:
                raise ValueError(f"{name} must be a SHA-256 digest")
        if (self.metrics is None) == (self.failure is None):
            raise ValueError("a run must contain exactly one of metrics or failure")
        if self.requested_actions is not None:
            if any(
                not isinstance(item, RequestedActionV2)
                for item in self.requested_actions
            ):
                raise TypeError("requested_actions must contain RequestedActionV2 rows")
            indices = [item.decision_index for item in self.requested_actions]
            if indices != list(range(len(indices))):
                raise ValueError(
                    "requested-action decision indices must be contiguous from zero"
                )
            if (
                self.metrics is not None
                and len(self.requested_actions) != self.metrics.decisions
            ):
                raise ValueError(
                    "successful requested-action trace length must equal metric decisions"
                )
        if self.controller_evidence is not None:
            if not isinstance(self.controller_evidence, dict):
                raise TypeError("controller_evidence must be a mapping or None")
            try:
                json.dumps(
                    self.controller_evidence,
                    allow_nan=False,
                    sort_keys=True,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("controller_evidence must be strict JSON") from exc
            if self.controller == SERVICE_MPC_CONTROLLER_ID and (
                self.controller_evidence.get("format")
                != SERVICE_MPC_RUN_EVIDENCE_FORMAT
            ):
                raise ValueError("service-MPC run evidence format is invalid")

    @property
    def successful(self) -> bool:
        return self.metrics is not None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "controller": self.controller,
            "seed": self.seed,
            "manifest_sha256": self.manifest_sha256,
            "exogenous_episode_sha256": self.exogenous_episode_sha256,
        }
        if self.metrics is not None:
            # Preserve the frozen successful-baseline serialisation exactly.
            payload["metrics"] = self.metrics.to_dict()
        else:
            assert self.failure is not None
            payload.update(
                {
                    "status": "failed",
                    "metrics": None,
                    "failure": self.failure.to_dict(),
                }
            )
        if self.requested_actions is not None:
            payload["requested_action_trace"] = _action_trace_payload(
                self.requested_actions
            )
        if self.controller_evidence is not None:
            payload["controller_evidence"] = deepcopy(self.controller_evidence)
        return payload


@dataclass
class ControllerEvaluationV2:
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
    contrasts: tuple[PairedContrastV2, ...] = ()

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
        successful_by_seed: dict[int, set[str]] = {seed: set() for seed in self.seeds}
        run_counts: dict[tuple[int, str], int] = {}
        for run in self.runs:
            manifest_hashes.setdefault(run.seed, set()).add(run.manifest_sha256)
            exogenous_hashes.setdefault(run.seed, set()).add(
                run.exogenous_episode_sha256
            )
            arms_by_seed.setdefault(run.seed, set()).add(run.controller)
            run_counts[(run.seed, run.controller)] = (
                run_counts.get((run.seed, run.controller), 0) + 1
            )
            if run.successful:
                successful_by_seed.setdefault(run.seed, set()).add(run.controller)
        expected = set(self.labels)
        declared_seeds = set(self.seeds)
        full_manifest_match = all(
            len(manifest_hashes.get(seed, set())) == 1 for seed in self.seeds
        )
        exogenous_match = all(
            len(exogenous_hashes.get(seed, set())) == 1 for seed in self.seeds
        )
        all_arms_present = all(
            arms_by_seed.get(seed, set()) == expected for seed in self.seeds
        )
        exactly_one = all(
            run_counts.get((seed, controller), 0) == 1
            for seed in self.seeds
            for controller in expected
        )
        only_declared_runs = all(
            run.seed in declared_seeds and run.controller in expected
            for run in self.runs
        )
        payload = {
            "common_random_numbers": bool(
                exogenous_match
                and all_arms_present
                and exactly_one
                and only_declared_runs
            ),
            "exogenous_episode_match_by_seed": exogenous_match,
            # Kept as an explicit, stronger identity for arms which share the
            # complete environment configuration. Commitment on/off ablations
            # legitimately have different full manifests while retaining the
            # same exogenous episode identity.
            "full_manifest_match_by_seed": full_manifest_match,
            "manifest_match_by_seed": full_manifest_match,
            "all_arms_present_by_seed": all_arms_present,
            "exactly_one_run_per_arm_by_seed": exactly_one,
            "only_declared_seed_controller_runs": only_declared_runs,
            "all_arms_successful_by_seed": all(
                successful_by_seed.get(seed, set()) == expected for seed in self.seeds
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
        return payload

    def summary(self) -> dict[str, Any]:
        return {
            controller: _summarise(
                [
                    run.metrics
                    for run in self.runs
                    if run.controller == controller and run.metrics is not None
                ]
            )
            for controller in self.labels
        }

    def execution_summary(self) -> dict[str, Any]:
        """Report every attempted arm, including failures excluded from metrics."""
        by_controller: dict[str, dict[str, int]] = {}
        for controller in self.labels:
            attempted = [run for run in self.runs if run.controller == controller]
            succeeded = sum(run.successful for run in attempted)
            by_controller[controller] = {
                "planned_runs": len(self.seeds),
                "attempted_runs": len(attempted),
                "successful_runs": succeeded,
                "failed_runs": len(attempted) - succeeded,
            }
        pairing = self.pairing()
        return {
            "by_controller": by_controller,
            "all_planned_arms_succeeded": bool(
                pairing["all_arms_present_by_seed"]
                and pairing["exactly_one_run_per_arm_by_seed"]
                and pairing["only_declared_seed_controller_runs"]
                and pairing["all_arms_successful_by_seed"]
            ),
            "failed_runs": [run.to_dict() for run in self.runs if not run.successful],
        }

    def contrast_results(self) -> list[dict[str, Any]]:
        """Calculate only the paired contrasts declared before execution."""
        return [_paired_contrast(self, contrast) for contrast in self.contrasts]

    def to_dict(self) -> dict[str, Any]:
        execution = self.execution_summary()
        payload: dict[str, Any] = {
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
            "summary_scope": (
                "finite/defined values from all planned runs; each metric count is explicit"
                if execution["all_planned_arms_succeeded"]
                else "finite/defined values from successful runs only; aggregate "
                "comparison ineligible because planned execution is incomplete"
            ),
            "summary": self.summary(),
            "runs": [run.to_dict() for run in self.runs],
        }
        if self.controller_specs:
            payload["controller_specs"] = deepcopy(self.controller_specs)
            payload["execution"] = execution
        if self.contrasts:
            payload["statistics"].update(
                {
                    "paired_delta": "exact seed-aligned treatment minus control",
                    "median": "empirical sample median",
                    "lower_tail": (
                        "empirical mean of the lowest 20% benefit deltas, rounded "
                        "up to at least one seed"
                    ),
                    "missing_pairs": "reported unavailable and never imputed",
                }
            )
            payload["contrasts"] = self.contrast_results()
        return payload

    def report(self) -> str:
        summary = self.summary()
        execution = self.execution_summary()
        columns = [
            ("success", None, "{}"),
            ("fill", "delivery_fill_rate", "{:.1%}"),
            ("unmet kg", "unmet_kg", "{:.1f}"),
            ("service", "commitment_service_level", "{:.1%}"),
            ("CH4 kg", "ch4_produced_kg", "{:.1f}"),
            ("terminal H2", "terminal_h2_kg", "{:.1f}"),
            ("terminal CO2", "terminal_co2_kg", "{:.1f}"),
            ("terminal CH4", "terminal_ch4_kg", "{:.1f}"),
            ("terminal SOC", "terminal_battery_soc", "{:.1%}"),
            (
                "process kWh/kg",
                "accepted_process_energy_per_delivered_kg",
                "{:.1f}",
            ),
            ("curtail kWh", "curtailed_kwh", "{:.0f}"),
            ("min SOC", "min_battery_soc", "{:.1%}"),
            ("guard", "guard_interventions", "{:.0f}"),
            ("guard kWh", "guard_intervention_kwh", "{:.0f}"),
            ("plant route Δ", "dispatch_gap_kwh", "{:.0f}"),
            ("reward", "reward_total", "{:.2f}"),
        ]
        if any(
            run.metrics is not None and run.metrics.electrolyser_commitment_enabled
            for run in self.runs
        ):
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
            (
                "Controller-run completion: all planned controller runs succeeded."
                if execution["all_planned_arms_succeeded"]
                else "Controller-run completion: INCOMPLETE; at least one planned "
                "controller run failed or is missing. Aggregate effects are ineligible."
            ),
            "",
            "controller".ljust(22)
            + "".join(label.rjust(14) for label, _, _ in columns),
        ]
        for controller in self.labels:
            metrics = summary[controller]["metrics"]
            controller_execution = execution["by_controller"][controller]
            complete = (
                controller_execution["attempted_runs"]
                == controller_execution["planned_runs"]
                == controller_execution["successful_runs"]
            )
            lines.append(
                self.labels.get(controller, controller)[:21].ljust(22)
                + "".join(
                    (
                        (
                            f"{controller_execution['successful_runs']}/"
                            f"{controller_execution['planned_runs']}"
                        )
                        if name is None
                        else (
                            formatter.format(metrics[name]["mean"])
                            if (
                                complete
                                and name in metrics
                                and metrics[name]["count"]
                                == controller_execution["planned_runs"]
                            )
                            else "n/a"
                        )
                    ).rjust(14)
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
                f"all arms {'present' if pairing['all_arms_present_by_seed'] else 'MISSING'}; "
                "one run per arm "
                f"{'confirmed' if pairing['exactly_one_run_per_arm_by_seed'] else 'VIOLATED'}"
            ),
            (
                "Distance semantics: request/guard-to-plant gaps are routing, "
                "equipment-refusal, and overlap-cancellation distances; they are "
                "not control-authority failures by themselves."
            ),
            (
                "Endowment semantics: delivery and process-energy intensity must be "
                "interpreted jointly with terminal H2, CO2, CH4, and battery balances; "
                "inventory liquidation is not free performance."
            ),
        ]
        failed = [run for run in self.runs if not run.successful]
        if failed:
            lines += ["", "Failed controller runs (retained, never imputed):"]
            for run in failed:
                assert run.failure is not None
                lines.append(
                    f"- {run.controller}, seed {run.seed}, {run.failure.phase}: "
                    f"{run.failure.exception_type}: {run.failure.message}"
                )
        contrast_results = self.contrast_results()
        if contrast_results:
            lines += [
                "",
                "Predeclared paired contrasts (benefit signs only where direction is declared):",
            ]
            for contrast in contrast_results:
                lines.append(
                    f"{contrast['id']}: {contrast['treatment']} minus "
                    f"{contrast['control']}"
                )
                if contrast["effect_summary_status"] != "valid":
                    lines.append(
                        "  aggregate effect: INVALID because planned paired "
                        "execution is incomplete or identity-mismatched; per-seed "
                        "observations are retained below"
                    )
                mechanism = contrast["requested_action_mechanism"]
                lines.append(
                    "  requested-action treatment: "
                    f"{mechanism['treatment_expression_status']}"
                    + (
                        f"; {mechanism['unequal_requested_decisions']}/"
                        f"{mechanism['decisions_compared']} matched decisions differ"
                        if mechanism["status"] == "available"
                        else "; action-trace comparison incomplete"
                    )
                )
                for name, result in contrast["metrics"].items():
                    if name not in _REPORT_CONTRAST_METRICS:
                        continue
                    raw = result.get("raw_treatment_minus_control_summary")
                    if result["direction"] == "diagnostic":
                        if raw is None:
                            lines.append(
                                f"  {name} [diagnostic]: aggregate unavailable "
                                f"({result['available_pairs']}/{len(self.seeds)} "
                                "paired seeds observed)"
                            )
                        else:
                            lines.append(
                                f"  {name} [diagnostic, no benefit sign]: "
                                f"raw mean {raw['mean']:.6g}; median "
                                f"{raw['median']:.6g} "
                                f"({raw['count']}/{len(self.seeds)} paired seeds)"
                            )
                        seed_values = [
                            (
                                f"{item['seed']}:raw="
                                f"{item['raw_treatment_minus_control']:.6g}"
                                if item["status"] == "available"
                                else f"{item['seed']}:unavailable"
                            )
                            for item in result["by_seed"]
                        ]
                        lines.append("    seed deltas: " + "; ".join(seed_values))
                        continue
                    benefit = result.get("benefit_delta_summary")
                    if benefit is None:
                        lines.append(
                            f"  {name}: aggregate unavailable "
                            f"({result['available_pairs']}/{len(self.seeds)} "
                            "paired seeds observed)"
                        )
                        continue
                    assert raw is not None
                    lines.append(
                        f"  {name} [{result['direction']}]: "
                        f"raw mean {raw['mean']:.6g}; "
                        f"benefit mean {benefit['mean']:.6g}; "
                        f"median {benefit['median']:.6g}; "
                        f"worst {benefit['worst']:.6g}; "
                        f"lower-tail mean {benefit['lower_tail_mean']:.6g} "
                        f"({benefit['count']}/{len(self.seeds)} paired seeds)"
                    )
                    seed_values = []
                    for item in result["by_seed"]:
                        if item["status"] == "available":
                            seed_values.append(
                                f"{item['seed']}:raw={item['raw_treatment_minus_control']:.6g},"
                                f"benefit={item['benefit_delta']:.6g}"
                            )
                        else:
                            seed_values.append(f"{item['seed']}:unavailable")
                    lines.append("    seed deltas: " + "; ".join(seed_values))
        return "\n".join(lines)


@dataclass
class CausalControllerEvaluationV2(ControllerEvaluationV2):
    """Nominal type preventing causal evidence from masquerading as a baseline."""


# Compatibility alias for the original baseline-only public API.
BaselineEvaluation = ControllerEvaluationV2


def _finite_numeric(value: object) -> TypeGuard[Real]:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _summarise(metrics: Sequence[EpisodeMetricsV2]) -> dict[str, Any]:
    records = [item.to_dict() for item in metrics]
    numeric_names = sorted(
        {
            name
            for record in records
            for name, value in record.items()
            if _finite_numeric(value)
        }
    )
    reward_names = sorted(
        {
            name
            for record in records
            for name, value in record.get("reward_components", {}).items()
            if _finite_numeric(value)
        }
    )
    return {
        "runs": len(records),
        "metrics": {
            name: Distribution.of(
                [
                    float(record[name])
                    for record in records
                    if _finite_numeric(record.get(name))
                ]
            ).to_dict()
            for name in numeric_names
        },
        "reward_components": {
            name: Distribution.of(
                [
                    float(value)
                    for record in records
                    if _finite_numeric(
                        value := record.get("reward_components", {}).get(name)
                    )
                ]
            ).to_dict()
            for name in reward_names
        },
    }


def _delta_summary(
    values_by_seed: Sequence[tuple[int, float]],
    *,
    benefit_oriented: bool,
) -> dict[str, Any] | None:
    """Summarise empirical paired deltas without pretending asymptotic inference."""
    if not values_by_seed:
        return None
    values = [value for _, value in values_by_seed]
    payload: dict[str, Any] = {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }
    if benefit_oriented:
        ordered = sorted(values_by_seed, key=lambda item: (item[1], item[0]))
        tail_count = max(1, math.ceil(0.20 * len(ordered)))
        tail = ordered[:tail_count]
        worst = ordered[0][1]
        payload.update(
            {
                "worst": worst,
                "worst_seeds": [seed for seed, value in ordered if value == worst],
                "best": ordered[-1][1],
                "lower_tail_fraction": 0.20,
                "lower_tail_count": tail_count,
                "lower_tail_mean": statistics.fmean(value for _, value in tail),
                "lower_tail_seeds": [seed for seed, _ in tail],
            }
        )
    return payload


def _requested_action_contrast(
    seeds: Sequence[int],
    aligned: dict[int, tuple[BaselineRun, BaselineRun] | None],
) -> dict[str, Any]:
    """Audit whether a declared treatment changed realized controller requests."""
    by_seed: list[dict[str, Any]] = []
    all_share_l1: list[float] = []
    all_support_abs_kw: list[float] = []
    unequal_total = 0
    compared_total = 0
    invalid_disagreement_total = 0
    control_invalid_total = 0
    treatment_invalid_total = 0
    for seed in seeds:
        pair = aligned[seed]
        if pair is None:
            by_seed.append(
                {
                    "seed": seed,
                    "status": "unavailable",
                    "reasons": ["paired_execution_unavailable"],
                }
            )
            continue
        control_run, treatment_run = pair
        control_trace = control_run.requested_actions
        treatment_trace = treatment_run.requested_actions
        reasons: list[str] = []
        if control_trace is None:
            reasons.append("control_action_trace_missing")
        if treatment_trace is None:
            reasons.append("treatment_action_trace_missing")
        if reasons:
            by_seed.append({"seed": seed, "status": "unavailable", "reasons": reasons})
            continue
        assert control_trace is not None
        assert treatment_trace is not None
        if len(control_trace) != len(treatment_trace):
            by_seed.append(
                {
                    "seed": seed,
                    "status": "unavailable",
                    "reasons": ["action_trace_length_mismatch"],
                    "control_decisions": len(control_trace),
                    "treatment_decisions": len(treatment_trace),
                }
            )
            continue
        if any(
            control.decision_index != treatment.decision_index
            for control, treatment in zip(
                control_trace,
                treatment_trace,
                strict=True,
            )
        ):
            by_seed.append(
                {
                    "seed": seed,
                    "status": "unavailable",
                    "reasons": ["action_trace_decision_index_mismatch"],
                }
            )
            continue

        share_l1 = [
            sum(
                abs(treatment_value - control_value)
                for treatment_value, control_value in zip(
                    treatment.share_vector,
                    control.share_vector,
                    strict=True,
                )
            )
            for control, treatment in zip(
                control_trace,
                treatment_trace,
                strict=True,
            )
        ]
        support_abs_kw = [
            abs(treatment.battery_support_kw - control.battery_support_kw)
            for control, treatment in zip(
                control_trace,
                treatment_trace,
                strict=True,
            )
        ]
        invalid_disagreements = sum(
            control.controller_invalid != treatment.controller_invalid
            for control, treatment in zip(
                control_trace,
                treatment_trace,
                strict=True,
            )
        )
        unequal = sum(
            shares > 1e-12
            or support > 1e-9
            or control.controller_invalid != treatment.controller_invalid
            for shares, support, control, treatment in zip(
                share_l1,
                support_abs_kw,
                control_trace,
                treatment_trace,
                strict=True,
            )
        )
        compared = len(control_trace)
        unequal_total += unequal
        compared_total += compared
        invalid_disagreement_total += invalid_disagreements
        control_invalid_total += sum(item.controller_invalid for item in control_trace)
        treatment_invalid_total += sum(
            item.controller_invalid for item in treatment_trace
        )
        all_share_l1.extend(share_l1)
        all_support_abs_kw.extend(support_abs_kw)
        by_seed.append(
            {
                "seed": seed,
                "status": "available",
                "decisions_compared": compared,
                "unequal_requested_decisions": unequal,
                "unequal_requested_decision_fraction": (
                    unequal / compared if compared else 0.0
                ),
                "controller_invalid_disagreement_decisions": invalid_disagreements,
                "control_invalid_decisions": sum(
                    item.controller_invalid for item in control_trace
                ),
                "treatment_invalid_decisions": sum(
                    item.controller_invalid for item in treatment_trace
                ),
                "share_l1_sum": sum(share_l1),
                "share_l1_max": max(share_l1, default=0.0),
                "battery_support_abs_delta_kw_sum": sum(support_abs_kw),
                "battery_support_abs_delta_kw_max": max(support_abs_kw, default=0.0),
                "control_trace_sha256": _action_trace_payload(control_trace)["sha256"],
                "treatment_trace_sha256": _action_trace_payload(treatment_trace)[
                    "sha256"
                ],
            }
        )

    available_pairs = sum(item["status"] == "available" for item in by_seed)
    complete = available_pairs == len(seeds)
    if not complete:
        expression_status = "unavailable"
    elif unequal_total > 0:
        expression_status = "expressed"
    else:
        expression_status = "not_expressed"
    return {
        "status": "available" if complete else "incomplete",
        "treatment_expression_status": expression_status,
        "definition": (
            "matched pre-guard requested actions; solar-share L1 and battery-support "
            "kW differences are reported separately because their units differ; "
            "controller-side invalid/clipping flags are retained"
        ),
        "expected_pairs": len(seeds),
        "available_pairs": available_pairs,
        "decisions_compared": compared_total,
        "unequal_requested_decisions": unequal_total if complete else None,
        "unequal_requested_decision_fraction": (
            unequal_total / compared_total if complete and compared_total else None
        ),
        "controller_invalid_disagreement_decisions": (
            invalid_disagreement_total if complete else None
        ),
        "control_invalid_decisions": control_invalid_total if complete else None,
        "treatment_invalid_decisions": (treatment_invalid_total if complete else None),
        "share_l1_sum": sum(all_share_l1) if complete else None,
        "share_l1_max": max(all_share_l1, default=0.0) if complete else None,
        "battery_support_abs_delta_kw_sum": (
            sum(all_support_abs_kw) if complete else None
        ),
        "battery_support_abs_delta_kw_max": (
            max(all_support_abs_kw, default=0.0) if complete else None
        ),
        "by_seed": by_seed,
    }


def _paired_contrast(
    evaluation: ControllerEvaluationV2,
    contrast: PairedContrastV2,
) -> dict[str, Any]:
    """Build one seed-aligned contrast with explicit unavailable pairs."""
    known_controllers = set(evaluation.labels)
    if contrast.treatment not in known_controllers:
        raise ValueError(
            f"contrast treatment {contrast.treatment!r} is not a declared controller"
        )
    if contrast.control not in known_controllers:
        raise ValueError(
            f"contrast control {contrast.control!r} is not a declared controller"
        )

    indexed: dict[tuple[str, int], list[BaselineRun]] = {}
    known_metric_names: set[str] = set()
    records: dict[int, dict[str, Any]] = {}
    for run in evaluation.runs:
        indexed.setdefault((run.controller, run.seed), []).append(run)
        if run.metrics is not None:
            record = run.metrics.to_dict()
            records[id(run)] = record
            known_metric_names.update(record)

    paired_execution: list[dict[str, Any]] = []
    aligned: dict[int, tuple[BaselineRun, BaselineRun] | None] = {}
    for seed in evaluation.seeds:
        controls = indexed.get((contrast.control, seed), [])
        treatments = indexed.get((contrast.treatment, seed), [])
        entry: dict[str, Any] = {"seed": seed}
        reasons: list[str] = []
        if len(controls) != 1:
            reasons.append(
                "control_run_missing" if not controls else "control_run_duplicated"
            )
        if len(treatments) != 1:
            reasons.append(
                "treatment_run_missing"
                if not treatments
                else "treatment_run_duplicated"
            )
        if not reasons:
            control_run = controls[0]
            treatment_run = treatments[0]
            if (
                control_run.exogenous_episode_sha256
                != treatment_run.exogenous_episode_sha256
            ):
                reasons.append("exogenous_episode_mismatch")
            if control_run.manifest_sha256 != treatment_run.manifest_sha256:
                reasons.append("full_manifest_mismatch")
            if not control_run.successful:
                reasons.append("control_failed")
                assert control_run.failure is not None
                entry["control_failure_phase"] = control_run.failure.phase
            if not treatment_run.successful:
                reasons.append("treatment_failed")
                assert treatment_run.failure is not None
                entry["treatment_failure_phase"] = treatment_run.failure.phase
            if not reasons:
                aligned[seed] = (control_run, treatment_run)
        if reasons:
            aligned[seed] = None
            entry.update({"status": "unavailable", "reasons": reasons})
        else:
            entry["status"] = "available"
        paired_execution.append(entry)

    local_execution_complete = all(
        entry["status"] == "available" for entry in paired_execution
    )
    study_pairing = evaluation.pairing()
    study_execution_complete = bool(
        study_pairing["exogenous_episode_match_by_seed"]
        and study_pairing["full_manifest_match_by_seed"]
        and study_pairing["all_arms_present_by_seed"]
        and study_pairing["exactly_one_run_per_arm_by_seed"]
        and study_pairing["only_declared_seed_controller_runs"]
        and study_pairing["all_arms_successful_by_seed"]
    )
    execution_complete = local_execution_complete and study_execution_complete

    metric_results: dict[str, Any] = {}
    for metric_name, direction, interpretation in PAIRED_CONTRAST_METRICS:
        if metric_name not in known_metric_names:
            continue
        by_seed: list[dict[str, Any]] = []
        raw_values: list[tuple[int, float]] = []
        benefit_values: list[tuple[int, float]] = []
        for seed in evaluation.seeds:
            pair = aligned[seed]
            if pair is None:
                execution = next(
                    item for item in paired_execution if item["seed"] == seed
                )
                by_seed.append(
                    {
                        "seed": seed,
                        "status": "unavailable",
                        "reasons": list(execution["reasons"]),
                    }
                )
                continue
            control_run, treatment_run = pair
            control_value = records[id(control_run)].get(metric_name)
            treatment_value = records[id(treatment_run)].get(metric_name)
            unavailable: list[str] = []
            if not _finite_numeric(control_value):
                unavailable.append("control_metric_unavailable")
            if not _finite_numeric(treatment_value):
                unavailable.append("treatment_metric_unavailable")
            if unavailable:
                by_seed.append(
                    {
                        "seed": seed,
                        "status": "unavailable",
                        "reasons": unavailable,
                    }
                )
                continue
            assert _finite_numeric(control_value)
            assert _finite_numeric(treatment_value)
            control_number = float(control_value)
            treatment_number = float(treatment_value)
            raw_delta = treatment_number - control_number
            if raw_delta == 0.0:
                raw_delta = 0.0
            raw_values.append((seed, raw_delta))
            seed_result = {
                "seed": seed,
                "status": "available",
                "control": control_number,
                "treatment": treatment_number,
                "raw_treatment_minus_control": raw_delta,
            }
            if direction != "diagnostic":
                benefit_delta = (
                    raw_delta if direction == "higher_is_better" else -raw_delta
                )
                if benefit_delta == 0.0:
                    benefit_delta = 0.0
                benefit_values.append((seed, benefit_delta))
                seed_result["benefit_delta"] = benefit_delta
            by_seed.append(seed_result)
        all_metric_pairs_defined = len(raw_values) == len(evaluation.seeds)
        aggregate_eligible = execution_complete and all_metric_pairs_defined
        metric_results[metric_name] = {
            "direction": direction,
            "interpretation": interpretation,
            "expected_pairs": len(evaluation.seeds),
            "available_pairs": len(raw_values),
            "unavailable_pairs": len(evaluation.seeds) - len(raw_values),
            "all_metric_pairs_defined": all_metric_pairs_defined,
            "by_seed": by_seed,
            "raw_treatment_minus_control_summary": _delta_summary(
                raw_values if aggregate_eligible else (),
                benefit_oriented=False,
            ),
            "benefit_delta_summary": _delta_summary(
                benefit_values if aggregate_eligible else (),
                benefit_oriented=True,
            ),
        }

    return {
        **contrast.to_dict(),
        "raw_delta_definition": "treatment minus control",
        "benefit_delta_definition": (
            "raw delta for higher-is-better metrics; negative raw delta for "
            "lower-is-better metrics; positive favours treatment; diagnostic "
            "metrics have no benefit delta"
        ),
        "seed_alignment": "exact seed equality; unavailable pairs are never imputed",
        "effect_summary_status": (
            "valid"
            if execution_complete
            else "invalid_incomplete_or_mismatched_execution"
        ),
        "local_pair_execution_complete": local_execution_complete,
        "whole_study_execution_complete": study_execution_complete,
        "aggregate_summary_eligibility": (
            "all planned study arms succeeded with matching seed and manifest "
            "identities; one failed or missing arm invalidates every aggregate"
        ),
        "paired_seed_execution": paired_execution,
        "requested_action_mechanism": _requested_action_contrast(
            evaluation.seeds,
            aligned,
        ),
        "metrics": metric_results,
    }


def _validated_seeds(seeds: Sequence[int]) -> list[int]:
    raw_seeds = list(seeds)
    if not raw_seeds:
        raise ValueError("at least one seed is required")
    if any(
        isinstance(seed, bool) or not isinstance(seed, Integral) for seed in raw_seeds
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


def _git_bytes(root: Path, *args: str) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
        )
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def _untracked_inventory(repository: Path) -> list[dict[str, Any]] | None:
    """Hash untracked files and symlink payloads without retaining contents."""
    listed = _git_bytes(
        repository,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    if listed is None:
        return None
    inventory: list[dict[str, Any]] = []
    for raw_path in sorted(item for item in listed.split(b"\0") if item):
        relative = os.fsdecode(raw_path)
        path = repository / relative
        entry: dict[str, Any] = {"path": relative}
        try:
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                content = os.fsencode(os.readlink(path))
                entry["kind"] = "symlink"
            elif stat.S_ISREG(mode):
                content = path.read_bytes()
                entry["kind"] = "file"
            else:
                entry.update(
                    {
                        "kind": "unsupported",
                        "content_sha256": None,
                        "bytes": None,
                    }
                )
                inventory.append(entry)
                continue
        except OSError as exc:
            entry.update(
                {
                    "kind": "unreadable",
                    "content_sha256": None,
                    "bytes": None,
                    "error_type": type(exc).__name__,
                }
            )
            inventory.append(entry)
            continue
        entry.update(
            {
                "content_sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        )
        inventory.append(entry)
    return inventory


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
    tracked_diff_sha256: str | None = None
    tracked_diff_bytes: int | None = None
    untracked_files: list[dict[str, Any]] | None = None
    worktree_identity_sha256: str | None = None
    source_identity_complete = False
    lock: dict[str, str | None] = {"path": None, "sha256": None}
    if repository is not None:
        revision = _git(repository, "rev-parse", "HEAD")
        status = _git(repository, "status", "--porcelain", "--untracked-files=normal")
        dirty = None if status is None else bool(status)
        tracked_diff = _git_bytes(repository, *_TRACKED_DIFF_ARGS)
        if tracked_diff is not None:
            tracked_diff_sha256 = hashlib.sha256(tracked_diff).hexdigest()
            tracked_diff_bytes = len(tracked_diff)
        untracked_files = _untracked_inventory(repository)
        source_identity_complete = (
            revision is not None
            and dirty is not None
            and tracked_diff_sha256 is not None
            and tracked_diff_bytes is not None
            and untracked_files is not None
            and all(
                entry.get("content_sha256") is not None for entry in untracked_files
            )
        )
        if source_identity_complete:
            identity = {
                "format": "git-worktree-content-v1",
                "git_revision": revision,
                "tracked_binary_diff_sha256": tracked_diff_sha256,
                "untracked_files": untracked_files,
            }
            canonical_identity = json.dumps(
                identity,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            worktree_identity_sha256 = hashlib.sha256(canonical_identity).hexdigest()
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
            "tracked_binary_diff_sha256": tracked_diff_sha256,
            "tracked_binary_diff_bytes": tracked_diff_bytes,
            "tracked_binary_diff_git_args": list(_TRACKED_DIFF_ARGS),
            "untracked_files": untracked_files,
            "worktree_identity_format": "git-worktree-content-v1",
            "worktree_identity_sha256": worktree_identity_sha256,
            "identity_complete": source_identity_complete,
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


def _execution_source_audit(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    require_clean_source: bool,
) -> dict[str, Any]:
    """Compare complete content identities at both execution boundaries."""
    before_complete = before.get("identity_complete") is True
    after_complete = after.get("identity_complete") is True
    identity_complete = before_complete and after_complete
    before_identity = before.get("worktree_identity_sha256")
    after_identity = after.get("worktree_identity_sha256")
    unchanged = bool(
        identity_complete
        and isinstance(before_identity, str)
        and before_identity == after_identity
    )
    clean_start = before.get("git_dirty") is False
    clean_end = after.get("git_dirty") is False
    valid = bool(
        identity_complete
        and unchanged
        and (not require_clean_source or (clean_start and clean_end))
    )
    return {
        "status": "valid" if valid else "invalid",
        "require_clean_source": require_clean_source,
        "identity_complete_at_start": before_complete,
        "identity_complete_at_end": after_complete,
        "worktree_identity_unchanged": unchanged,
        "clean_at_start": clean_start,
        "clean_at_end": clean_end,
        "start_worktree_identity_sha256": before_identity,
        "end_worktree_identity_sha256": after_identity,
        "end_source": deepcopy(after),
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


def _validated_controller_descriptor(controller: object) -> dict[str, object]:
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
        raise ValueError(f"controller {controller_id!r} description must not be empty")
    if not isinstance(information_set, str) or not information_set:
        raise ValueError(
            f"controller {controller_id!r} information_set must not be empty"
        )
    if not isinstance(uses_forecast, bool):
        raise TypeError(f"controller {controller_id!r} uses_forecast must be a bool")
    if any(
        not callable(getattr(controller, name, None))
        for name in ("compute_action", "reset", "to_dict")
    ):
        raise TypeError(f"controller {controller_id!r} does not implement ControllerV2")
    descriptor = controller.to_dict()  # type: ignore[attr-defined]
    if not isinstance(descriptor, dict):
        raise TypeError(f"controller {controller_id!r} descriptor must be a mapping")
    try:
        json.dumps(descriptor, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"controller {controller_id!r} descriptor must be strict JSON"
        ) from exc
    if descriptor.get("id") != controller_id:
        raise ValueError(f"controller {controller_id!r} descriptor id does not match")
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
    return deepcopy(descriptor)


def _safe_failure_message(exception: Exception) -> str:
    collapsed = " ".join(str(exception).splitlines()).strip()
    return (collapsed or "(no exception message)")[:500]


def _controller_run_evidence(
    controller: object | None,
    applications: Sequence[MPCDecisionApplicationV1],
    *,
    completed_decisions: int,
    run_succeeded: bool,
    failure_phase: str | None,
) -> dict[str, Any] | None:
    if (
        controller is None
        or getattr(controller, "id", None) != SERVICE_MPC_CONTROLLER_ID
    ):
        return None
    events = getattr(controller, "decision_events", None)
    pending = getattr(controller, "has_pending_transition", None)
    if not isinstance(events, tuple) or any(
        not isinstance(event, (MPCDecisionEvidenceV1, MPCDecisionFailureV2))
        for event in events
    ):
        raise TypeError("service-MPC controller did not expose typed decision events")
    if not isinstance(pending, bool):
        raise TypeError(
            "service-MPC controller did not expose pending-transition state"
        )
    return build_mpc_run_evidence(
        events,
        tuple(applications),
        completed_decisions=completed_decisions,
        run_succeeded=run_succeeded,
        pending_transition=pending,
        runner_failure_phase=failure_phase,
    )


def _failed_controller_run(
    env: PlantEnvV2,
    *,
    controller_id: str,
    seed: int,
    manifest: dict[str, Any],
    phase: str,
    exception: Exception,
    requested_actions: Sequence[RequestedActionV2] = (),
    controller: object | None = None,
    applications: Sequence[MPCDecisionApplicationV1] = (),
) -> BaselineRun:
    partial_metrics = env.metrics.to_dict()
    try:
        controller_evidence = _controller_run_evidence(
            controller,
            applications,
            completed_decisions=env.metrics.decisions,
            run_succeeded=False,
            failure_phase=phase,
        )
    except Exception as evidence_exception:
        exception = RuntimeError(
            f"{type(exception).__name__}: {_safe_failure_message(exception)}; "
            "controller evidence collection also failed: "
            f"{type(evidence_exception).__name__}: "
            f"{_safe_failure_message(evidence_exception)}"
        )
        phase = "controller_evidence_collection"
        controller_evidence = None
    return BaselineRun(
        controller=controller_id,
        seed=int(seed),
        manifest_sha256=str(manifest["manifest_sha256"]),
        exogenous_episode_sha256=str(manifest["exogenous_episode_sha256"]),
        metrics=None,
        failure=ControllerFailureV2(
            phase=phase,
            exception_type=type(exception).__name__,
            message=_safe_failure_message(exception),
            completed_decisions=env.metrics.decisions,
            elapsed_hours=env.metrics.hours,
            partial_metrics=partial_metrics,
        ),
        requested_actions=tuple(requested_actions),
        controller_evidence=controller_evidence,
    )


def run_controller(
    env: PlantEnvV2,
    controller: ControllerV2,
    *,
    seed: int,
    expected_descriptor: dict[str, object] | None = None,
) -> BaselineRun:
    """Run one typed causal controller, retaining controller-side failures."""
    expected = (
        _validated_controller_descriptor(controller)
        if expected_descriptor is None
        else deepcopy(expected_descriptor)
    )
    controller_id = expected.get("id")
    if not isinstance(controller_id, str) or not controller_id:
        raise ValueError("expected controller descriptor id must not be empty")
    env.reset(seed=seed)
    manifest = env.audit_manifest()
    applications: list[MPCDecisionApplicationV1] = []
    try:
        controller.reset()
    except Exception as exc:
        return _failed_controller_run(
            env,
            controller_id=controller_id,
            seed=seed,
            manifest=manifest,
            phase="controller_reset",
            exception=exc,
            controller=controller,
            applications=applications,
        )
    try:
        executed_descriptor = _validated_controller_descriptor(controller)
        if executed_descriptor != expected:
            raise ValueError(
                "executed controller descriptor differs from its predeclared prototype"
            )
    except Exception as exc:
        return _failed_controller_run(
            env,
            controller_id=controller_id,
            seed=seed,
            manifest=manifest,
            phase="descriptor_integrity_after_reset",
            exception=exc,
            controller=controller,
            applications=applications,
        )
    uses_forecast = expected.get("uses_forecast")
    if not isinstance(uses_forecast, bool):
        raise TypeError("expected controller uses_forecast must be a bool")
    finished = False
    requested_actions: list[RequestedActionV2] = []
    while not finished:
        observation = env.controller_observation
        if not uses_forecast:
            observation = observation.without_forecast()
        try:
            decision = controller.compute_action(
                observation,
                constraints=env.constraints,
                config=env.config,
                commitment_profile=env.episode_config.electrolyser_commitment,
            )
        except Exception as exc:
            return _failed_controller_run(
                env,
                controller_id=controller_id,
                seed=seed,
                manifest=manifest,
                phase="controller_compute_action",
                exception=exc,
                requested_actions=requested_actions,
                controller=controller,
                applications=applications,
            )
        try:
            if _validated_controller_descriptor(controller) != expected:
                raise ValueError(
                    "controller descriptor changed while computing an action"
                )
        except Exception as exc:
            return _failed_controller_run(
                env,
                controller_id=controller_id,
                seed=seed,
                manifest=manifest,
                phase="descriptor_integrity_after_decision",
                exception=exc,
                requested_actions=requested_actions,
                controller=controller,
                applications=applications,
            )
        if not isinstance(decision, ControlDecision):
            return _failed_controller_run(
                env,
                controller_id=controller_id,
                seed=seed,
                manifest=manifest,
                phase="controller_output_validation",
                exception=TypeError("V2 controller must return a ControlDecision"),
                requested_actions=requested_actions,
                controller=controller,
                applications=applications,
            )
        try:
            requested_actions.append(
                RequestedActionV2.from_action(
                    env.decisions_taken,
                    decision.action,
                    controller_invalid=decision.invalid,
                )
            )
        except Exception as exc:
            return _failed_controller_run(
                env,
                controller_id=controller_id,
                seed=seed,
                manifest=manifest,
                phase="controller_output_validation",
                exception=exc,
                requested_actions=requested_actions,
                controller=controller,
                applications=applications,
            )
        try:
            _, _, terminated, truncated, info = env.step_action(
                decision.action,
                controller_invalid=decision.invalid,
            )
        except Exception as exc:
            return _failed_controller_run(
                env,
                controller_id=controller_id,
                seed=seed,
                manifest=manifest,
                phase="controller_action_application",
                exception=exc,
                requested_actions=requested_actions,
                controller=controller,
                applications=applications,
            )
        if controller_id == SERVICE_MPC_CONTROLLER_ID:
            try:
                observe_transition = getattr(controller, "observe_transition", None)
                if not callable(observe_transition):
                    raise TypeError(
                        "service-MPC controller does not expose observe_transition"
                    )
                observe_transition(env.controller_observation)
            except Exception as exc:
                return _failed_controller_run(
                    env,
                    controller_id=controller_id,
                    seed=seed,
                    manifest=manifest,
                    phase="controller_transition_observation",
                    exception=exc,
                    requested_actions=requested_actions,
                    controller=controller,
                    applications=applications,
                )
            try:
                events = getattr(controller, "decision_events", None)
                decision_index = env.metrics.decisions - 1
                if not isinstance(events, tuple) or len(events) <= decision_index:
                    raise ValueError(
                        "service-MPC controller evidence is missing the applied decision"
                    )
                event = events[decision_index]
                if not isinstance(event, MPCDecisionEvidenceV1):
                    raise TypeError(
                        "service-MPC applied decision is not successful evidence"
                    )
                if event.decision_index != decision_index:
                    raise ValueError(
                        "service-MPC evidence index differs from the applied decision"
                    )
                applications.append(capture_mpc_application(event, info))
            except Exception as exc:
                return _failed_controller_run(
                    env,
                    controller_id=controller_id,
                    seed=seed,
                    manifest=manifest,
                    phase="controller_evidence_capture",
                    exception=exc,
                    requested_actions=requested_actions,
                    controller=controller,
                    applications=applications,
                )
            try:
                if _validated_controller_descriptor(controller) != expected:
                    raise ValueError(
                        "controller descriptor changed while observing a transition"
                    )
            except Exception as exc:
                return _failed_controller_run(
                    env,
                    controller_id=controller_id,
                    seed=seed,
                    manifest=manifest,
                    phase="descriptor_integrity_after_transition",
                    exception=exc,
                    requested_actions=requested_actions,
                    controller=controller,
                    applications=applications,
                )
        finished = terminated or truncated
    try:
        controller_evidence = _controller_run_evidence(
            controller,
            applications,
            completed_decisions=env.metrics.decisions,
            run_succeeded=True,
            failure_phase=None,
        )
    except Exception as exc:
        return _failed_controller_run(
            env,
            controller_id=controller_id,
            seed=seed,
            manifest=manifest,
            phase="controller_evidence_collection",
            exception=exc,
            requested_actions=requested_actions,
        )
    return BaselineRun(
        controller=controller_id,
        seed=int(seed),
        manifest_sha256=str(manifest["manifest_sha256"]),
        exogenous_episode_sha256=str(manifest["exogenous_episode_sha256"]),
        metrics=env.metrics,
        requested_actions=tuple(requested_actions),
        controller_evidence=controller_evidence,
    )


def evaluate_controllers(
    *,
    seeds: Sequence[int],
    controllers: Sequence[ControllerV2],
    episode: EpisodeConfigV2 | None = None,
    plant: PlantConfig | None = None,
    constraints: Constraints | None = None,
    weights: RewardWeightsV2 | None = None,
    require_clean_source: bool = False,
) -> CausalControllerEvaluationV2:
    """Paired development comparison over the typed causal controller seam."""
    if not isinstance(require_clean_source, bool):
        raise TypeError("require_clean_source must be a bool")
    chosen_seeds = _validated_seeds(seeds)

    chosen_controllers = tuple(controllers)
    if not chosen_controllers:
        raise ValueError("at least one controller is required")
    ids: list[str] = []
    specs: dict[str, dict[str, object]] = {}
    for controller in chosen_controllers:
        descriptor = _validated_controller_descriptor(controller)
        controller_id = str(descriptor["id"])
        ids.append(controller_id)
        specs[controller_id] = deepcopy(descriptor)
    if len(set(ids)) != len(ids):
        raise ValueError("controller ids must be unique")

    plant = plant or PlantConfig()
    episode = (episode or EpisodeConfigV2()).validate(plant)
    constraints = constraints or Constraints()
    weights = weights or RewardWeightsV2()
    provenance = collect_provenance(
        episode=episode,
        plant=plant,
        constraints=constraints,
        weights=weights,
    )
    source_at_start = provenance["source"]
    if require_clean_source and (
        source_at_start.get("identity_complete") is not True
        or source_at_start.get("git_dirty") is not False
    ):
        raise ValueError(
            "clean-source evaluation requires a complete identity and clean git worktree"
        )
    result = CausalControllerEvaluationV2(
        seeds=chosen_seeds,
        episode_config=episode,
        provenance=provenance,
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
        contrasts=tuple(
            PairedContrastV2(treatment=ids[index], control=ids[index - 1])
            for index in range(1, len(ids))
        ),
    )
    for seed in chosen_seeds:
        for prototype in chosen_controllers:
            env = PlantEnvV2(
                config=plant,
                episode=episode,
                constraints=constraints,
                weights=weights,
            )
            prototype_id = str(getattr(prototype, "id"))
            try:
                executed = deepcopy(prototype)
                if executed is prototype:
                    raise ValueError("controller copy returned the prototype object")
            except Exception as exc:
                env.reset(seed=seed)
                result.runs.append(
                    _failed_controller_run(
                        env,
                        controller_id=prototype_id,
                        seed=seed,
                        manifest=env.audit_manifest(),
                        phase="controller_copy",
                        exception=exc,
                    )
                )
                continue
            result.runs.append(
                run_controller(
                    env,
                    executed,
                    seed=seed,
                    expected_descriptor=specs[prototype_id],
                )
            )

    pairing = result.pairing()
    if (
        not pairing["exogenous_episode_match_by_seed"]
        or not pairing["full_manifest_match_by_seed"]
        or not pairing["all_arms_present_by_seed"]
        or not pairing["exactly_one_run_per_arm_by_seed"]
        or not pairing["only_declared_seed_controller_runs"]
    ):
        raise RuntimeError("paired causal-controller audit failed")
    source_at_end = collect_provenance(
        episode=episode,
        plant=plant,
        constraints=constraints,
        weights=weights,
    )["source"]
    source_audit = _execution_source_audit(
        source_at_start,
        source_at_end,
        require_clean_source=require_clean_source,
    )
    result.provenance["execution_source_audit"] = source_audit
    if not (
        source_audit["identity_complete_at_start"]
        and source_audit["identity_complete_at_end"]
    ):
        raise RuntimeError("controller evaluation source identity is incomplete")
    if not source_audit["worktree_identity_unchanged"]:
        raise RuntimeError("controller evaluation source changed during execution")
    if require_clean_source and not (
        source_audit["clean_at_start"] and source_audit["clean_at_end"]
    ):
        raise RuntimeError("controller evaluation source was not clean throughout")
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


def _validate_causal_evidence_for_write(result: ControllerEvaluationV2) -> None:
    """Reject mutable causal artifacts whose pairing/audit identity no longer holds."""
    if not isinstance(result, CausalControllerEvaluationV2):
        return
    if not isinstance(result.seeds, list):
        raise ValueError("causal evidence seeds must remain an ordered list")
    validated_seeds = _validated_seeds(result.seeds)
    run_seed_order: list[int] = []
    for run in result.runs:
        if run.seed not in run_seed_order:
            run_seed_order.append(run.seed)
    if validated_seeds != run_seed_order:
        raise ValueError(
            "causal evidence seed order differs from the executed run order"
        )
    if result.result_schema_version != CAUSAL_RESULT_SCHEMA_VERSION:
        raise ValueError("causal evidence schema version was mutated")
    controller_ids = list(result.controller_specs)
    if not controller_ids:
        raise ValueError("causal evidence must retain declared controller specs")
    if (
        list(result.labels) != controller_ids
        or list(result.information_sets) != controller_ids
    ):
        raise ValueError(
            "causal evidence controller labels, information sets, and specs disagree"
        )
    first_seed_order = [
        run.controller for run in result.runs if run.seed == validated_seeds[0]
    ]
    if first_seed_order != controller_ids:
        raise ValueError(
            "causal evidence controller order differs from the executed run order"
        )
    for controller_id, spec in result.controller_specs.items():
        if (
            spec.get("id") != controller_id
            or spec.get("label") != result.labels[controller_id]
            or spec.get("information_set") != result.information_sets[controller_id]
        ):
            raise ValueError(
                f"causal evidence metadata disagrees for controller {controller_id!r}"
            )
    expected_contrasts = tuple(
        PairedContrastV2(
            treatment=controller_ids[index],
            control=controller_ids[index - 1],
        )
        for index in range(1, len(controller_ids))
    )
    if result.contrasts != expected_contrasts:
        raise ValueError("causal evidence predeclared adjacent contrasts were mutated")
    pairing = result.pairing()
    required_pairing = (
        "exogenous_episode_match_by_seed",
        "full_manifest_match_by_seed",
        "all_arms_present_by_seed",
        "exactly_one_run_per_arm_by_seed",
        "only_declared_seed_controller_runs",
    )
    failed = [name for name in required_pairing if pairing.get(name) is not True]
    if failed:
        raise ValueError(
            "causal evidence pairing is invalid: " + ", ".join(sorted(failed))
        )
    missing_traces = [
        f"{run.controller}@{run.seed}"
        for run in result.runs
        if run.requested_actions is None
    ]
    if missing_traces:
        raise ValueError(
            "causal evidence is missing requested-action traces: "
            + ", ".join(missing_traces)
        )
    source_audit = result.provenance.get("execution_source_audit")
    if not isinstance(source_audit, dict) or source_audit.get("status") != "valid":
        raise ValueError("causal evidence has no valid start/end source-identity audit")
    # Recompute every contrast so post-evaluation mutations cannot bypass the
    # per-pair manifest checks or strict action-trace alignment.
    result.contrast_results()


def write(result: BaselineEvaluation, path: Path | str) -> Path:
    """Write complete machine-readable evidence for later comparison."""
    _validate_causal_evidence_for_write(result)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result.to_dict(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
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
    "ControllerFailureV2",
    "CausalControllerEvaluationV2",
    "Distribution",
    "ControllerEvaluationV2",
    "LegacyHeuristicControllerV2",
    "PAIRED_CONTRAST_METRICS",
    "PairedContrastV2",
    "RequestedActionV2",
    "collect_provenance",
    "evaluate",
    "evaluate_controllers",
    "run_baseline",
    "run_controller",
    "write",
]

"""Descriptive operational scorecards; physical admission is a separate layer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from typing import Any

import numpy as np

from ..beta_config import CYCLE_FEASIBILITY_EPSILON, RESOURCE_NAMES
from ..economics import EconomicEpisodeMetricsV1
from ..metrics import EpisodeMetricsV2
from ..training.config import ControllerNeutralTrainingConfigV1
from ..training.custody_json import authenticate, canonical_bytes
from .contract import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    COMPONENT_NAMES,
    EPISODE_COUNT,
    PREFIX,
    analysis_contract,
)
from .statistics import binomial, finite_real

CONTRACT_SCHEMA = PREFIX + "operational-contract-v1"
EPISODE_SCHEMA = PREFIX + "operational-episode-v1"
SUMMARY_SCHEMA = PREFIX + "operational-summary-v1"
REGIMES = ("calm", "volatile", "scarcity")
CAPTURE_THRESHOLD_KG_H = 1e-9
CYCLE_SIGNALS = (
    "delivered_kg",
    "unmet_kg",
    "accepted_solar_kwh",
    "battery_throughput_kwh",
    "battery_equivalent_full_cycles",
    "electrolyser_equivalent_operating_h",
)


def _numeric_leaves(raw: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result = {}
    for name, value in raw.items():
        key = prefix + name
        if isinstance(value, Mapping):
            result.update(_numeric_leaves(value, key + "."))
        elif type(value) in (int, float) or value is None:
            result[key] = value
    return result


def _metric_template() -> dict[str, Any]:
    # Metric classes establish the reporting field set without a plant or reset.
    economic = EconomicEpisodeMetricsV1(0.0, 0.0, 1.0).to_dict()
    economic["components_eur"] = {name: 0.0 for name in COMPONENT_NAMES}
    return _numeric_leaves(
        {"physical_metrics": EpisodeMetricsV2().to_dict(), "economic_metrics": economic}
    )


def operational_contract() -> dict[str, Any]:
    return authenticate(
        {
            "schema": CONTRACT_SCHEMA,
            "analysis_contract_sha256": analysis_contract()["sha256"],
            "resources": list(RESOURCE_NAMES),
            "resource_units": {
                r: "kWh" if r == "battery" else "kg" for r in RESOURCE_NAMES
            },
            "episode_metric_fields": list(_metric_template()),
            "feasibility_epsilon": CYCLE_FEASIBILITY_EPSILON,
            "safety_scope": "complete-episode safety separate from observed-prefix safety; preflight has no observed safety trial",
            "empty_cycle": "null economic totals and signals when no interval was observed",
            "feasibility_denominators": [
                "all 32 episode slots at each cycle index",
                "observed settlements at that index",
            ],
            "distribution": {
                "quantiles": [0, 0.25, 0.5, 0.75, 0.95, 1],
                "method": "linear",
                "raw_values": "retain identities and missingness",
            },
            "exposure_rates": "count/observed exposure; null at zero; no independent-trial interval",
            "strata": {
                "market": list(REGIMES),
                "outage": ["outage", "no_outage"],
                "cluster": "episode seed",
                "statistic": "pooled mean of observed complete units",
                "bootstrap": "same frozen 32-episode draws across metrics",
                "empty_resample": "retain undefined count; conditional interval only",
                "selection_role": "none",
            },
            "recovery": {
                "threshold_kg_h": CAPTURE_THRESHOLD_KG_H,
                "duration": "outage end to productive interval start; includes intentional idling",
                "censor": "next outage or observed execution end",
                "unobserved": "no imputation",
            },
            "training": "all five slots; complete admitted rollout trajectories; saturation/observed rollouts",
            "claim_boundary": "descriptive arithmetic; no source, checkpoint or stage admission",
        }
    )


def _count(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def exposure_rate(count: object, total: object) -> dict[str, Any]:
    n, d = _count(count, "count"), _count(total, "exposure")
    if n > d:
        raise ValueError("event count exceeds exposure")
    return {"count": n, "exposure": d, "rate": n / d if d else None}


def distribution(values: Sequence[object]) -> dict[str, Any]:
    numbers = [finite_real(value) for value in values]
    result: dict[str, Any] = {
        "count": len(numbers),
        "values": numbers,
        "mean": None,
        "minimum": None,
        "p25": None,
        "median": None,
        "p75": None,
        "p95": None,
        "maximum": None,
    }
    if numbers:
        with np.errstate(over="ignore", invalid="ignore"):
            result["mean"] = finite_real(np.mean(numbers))
            result.update(
                zip(
                    ("minimum", "p25", "median", "p75", "p95", "maximum"),
                    [
                        finite_real(v)
                        for v in np.quantile(
                            numbers, [0, 0.25, 0.5, 0.75, 0.95, 1], method="linear"
                        )
                    ],
                    strict=True,
                )
            )
    return result


def outage_recovery(
    schedule: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
    *,
    physics_dt_s: float,
    horizon_s: float,
) -> list[dict[str, Any]]:
    """Use an already admitted schedule and contiguous post-minute truth ledger."""
    dt = finite_real(physics_dt_s)
    horizon = finite_real(horizon_s)
    if dt <= 0 or horizon <= 0 or not states:
        raise ValueError(
            "outage projection requires an initial state and positive geometry"
        )
    observed_end = (len(states) - 1) * dt
    if observed_end > horizon or any(
        state["step"] != index for index, state in enumerate(states)
    ):
        raise ValueError("outage truth ledger is not a contiguous task prefix")
    result = []
    previous_end = 0.0
    for index, fault in enumerate(schedule):
        start, duration = finite_real(fault["start_s"]), finite_real(
            fault["duration_s"]
        )
        end = start + duration
        if (
            start < previous_end
            or duration <= 0
            or end > horizon
            or start % dt
            or duration % dt
        ):
            raise ValueError(
                "outage schedule differs from ordered finite minute intervals"
            )
        previous_end = end
        next_start = (
            finite_real(schedule[index + 1]["start_s"])
            if index + 1 < len(schedule)
            else horizon
        )
        censor = min(observed_end, next_start)
        productive = next(
            (
                step * dt
                for step in range(int(end / dt), int(censor / dt))
                if finite_real(states[step + 1]["co2_capture_kg_h"])
                > CAPTURE_THRESHOLD_KG_H
            ),
            None,
        )
        if observed_end <= start:
            status = "not_reached"
        elif observed_end < end:
            status = "outage_incomplete"
        elif productive is not None:
            status = "resumed"
        else:
            status = "censored_no_capture"
        result.append(
            {
                "outage_index": index,
                "start_s": start,
                "end_s": end,
                "scheduled_duration_s": duration,
                "encountered_duration_s": max(0.0, min(observed_end, end) - start),
                "status": status,
                "first_productive_interval_start_s": productive,
                "recovery_delay_s": None if productive is None else productive - end,
                "followup_end_s": max(end, censor) if observed_end >= end else None,
                "censor_reason": (
                    (
                        "next_outage"
                        if next_start <= observed_end and next_start < horizon
                        else "task_end" if observed_end == horizon else "execution_end"
                    )
                    if status == "censored_no_capture"
                    else None
                ),
            }
        )
    return result


def episode_scorecard(
    *,
    seed: int,
    actor: Mapping[str, Any],
    config: ControllerNeutralTrainingConfigV1,
    manifest: Mapping[str, Any] | None,
    run: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a fully admitted arm. This function itself supplies no admission."""
    _count(seed, "episode seed")
    if config.beta_task is None:
        raise ValueError("operational scorecards require a beta task")
    expected = config.episode.decision_steps(config.plant)
    cycle_decisions = config.beta_task.cycle.cycle_decisions(config.episode)
    cycle_count = expected // cycle_decisions
    cycle_s = config.beta_task.cycle.cycle_h * 3600
    preflight = run["status"] == "not_run_preflight_failed"
    evidence = run["evidence"]
    physical, economic = deepcopy(run["physical_metrics"]), deepcopy(
        run["economic_metrics"]
    )
    safety = deepcopy(run.get("hard_safety_metrics"))
    completed = _count(run["completed_decisions"], "completed decisions")
    minutes = _count(run["physical_steps"], "physical steps")
    attempted = len(evidence["boundary_ledger"]["decision_boundaries"])
    if completed > expected or completed > attempted or attempted > completed + 1:
        raise ValueError("operational decision coverage differs")
    if preflight != (manifest is None):
        raise ValueError("operational manifest availability differs from preflight")
    if not preflight and safety is None:
        raise ValueError("operational safety evidence is missing")
    schedule = None if manifest is None else manifest["disturbances"]["schedule"]
    regime = None if manifest is None else manifest["market"]["selected_regime"]
    if regime is not None and regime not in REGIMES:
        raise ValueError("unknown market stratum")
    intervals = evidence.get("intervals", [])
    settlements = run["cycle_settlements"]
    if len(intervals) != completed or len(settlements) != completed // cycle_decisions:
        raise ValueError("operational cycle/interval coverage differs")
    cycles = []
    for index in range(cycle_count):
        settlement = deepcopy(settlements[index]) if index < len(settlements) else None
        rows = intervals[index * cycle_decisions : (index + 1) * cycle_decisions]
        full = len(rows) == cycle_decisions
        components = {
            name: (
                math.fsum(row["reward"]["components_eur"][name] for row in rows)
                if rows
                else None
            )
            for name in COMPONENT_NAMES
        }
        resources = (
            None
            if settlement is None
            else {
                name: {
                    "target": settlement["targets"][name],
                    "actual": settlement["actual"][name],
                    "deficit": settlement["deficits"][name],
                    "tolerance": settlement["tolerances"][name],
                    "feasible": settlement["deficits"][name]
                    <= settlement["tolerances"][name] + CYCLE_FEASIBILITY_EPSILON,
                    "unit": "kWh" if name == "battery" else "kg",
                }
                for name in RESOURCE_NAMES
            }
        )
        cycles.append(
            {
                "cycle_index": index + 1,
                "complete": full,
                "observed_decisions": len(rows),
                "expected_decisions": cycle_decisions,
                "outage_stratum": (
                    None
                    if schedule is None
                    else (
                        "outage"
                        if any(
                            index * cycle_s <= fault["start_s"] < (index + 1) * cycle_s
                            for fault in schedule
                        )
                        else "no_outage"
                    )
                ),
                "settlement": settlement,
                "resources": resources,
                "net_operating_value_eur": (
                    math.fsum(row["reward"]["total_eur"] for row in rows)
                    if rows
                    else None
                ),
                "components_eur": components,
                "signals": {
                    name: (
                        math.fsum(row["reward"]["signals"][name] for row in rows)
                        if rows
                        else None
                    )
                    for name in CYCLE_SIGNALS
                },
            }
        )
    rates = (
        None
        if preflight
        else {
            "guard_correction": exposure_rate(
                physical["corrected_physics_steps"], minutes
            ),
            "unexpected_critical_or_offline": exposure_rate(
                (safety or {})["unexpected_critical_or_offline_steps"], minutes
            ),
            "forced_stop": exposure_rate(
                physical["electrolyser_forced_stops"], minutes
            ),
            "invalid_command": exposure_rate(
                physical["out_of_range_decisions"], attempted
            ),
        }
    )
    outages = (
        None
        if schedule is None
        else outage_recovery(
            schedule,
            evidence["minute_truth_states"],
            physics_dt_s=config.plant.dt_s,
            horizon_s=config.episode.horizon_h * 3600,
        )
    )
    return authenticate(
        {
            "schema": EPISODE_SCHEMA,
            "operational_contract_sha256": operational_contract()["sha256"],
            "seed": seed,
            "actor": dict(actor),
            "status": run["status"],
            "failure": deepcopy(run["failure"]),
            "execution_complete": run["execution_complete"],
            "hard_safety_passed": run["hard_safety_passed"],
            "cycle_feasible": run["cycle_feasible"],
            "expected_decisions": expected,
            "completed_decisions": completed,
            "attempted_decisions": attempted,
            "expected_physical_steps": int(
                config.episode.horizon_h * 3600 / config.plant.dt_s
            ),
            "physical_steps": minutes,
            "cycles": cycles,
            "market_regime": regime,
            "physical_metrics": physical,
            "economic_metrics": economic,
            "hard_safety_metrics": safety,
            "exposure_rates": rates,
            "outages": outages,
            "outage_count": None if outages is None else len(outages),
            "scheduled_outage_duration_s": (
                None
                if outages is None
                else math.fsum(row["scheduled_duration_s"] for row in outages)
            ),
            "encountered_outage_duration_s": (
                None
                if outages is None
                else math.fsum(row["encountered_duration_s"] for row in outages)
            ),
            "source_evidence_verified": False,
            "result_claim": None,
        }
    )


def clustered_summary(values: Sequence[Sequence[object]], draws: Any) -> dict[str, Any]:
    """Pooled unit mean, retaining the episode as bootstrap cluster."""
    if len(values) != EPISODE_COUNT:
        raise ValueError("strata retain exactly 32 episode clusters")
    numbers = [[finite_real(v) for v in row] for row in values]
    counts = np.array([len(row) for row in numbers], dtype=np.int64)
    totals = np.array([math.fsum(row) for row in numbers], dtype=np.float64)
    sampled_counts = counts[draws].sum(axis=1)
    defined = sampled_counts > 0
    sampled_totals = totals[draws].sum(axis=1)
    samples = sampled_totals[defined] / sampled_counts[defined]
    interval = (
        [finite_real(v) for v in np.quantile(samples, [0.025, 0.975], method="linear")]
        if len(samples)
        else None
    )
    return {
        "cluster_values": numbers,
        "observed_cluster_count": int(np.count_nonzero(counts)),
        "unit_distribution": distribution([v for row in numbers for v in row]),
        "bootstrap_defined_replicates": int(np.count_nonzero(defined)),
        "bootstrap_undefined_replicates": int(len(draws) - np.count_nonzero(defined)),
        "conditional_mean_interval_95": interval,
    }


def policy_scorecard(
    episode_seeds: Sequence[int],
    records: Sequence[Mapping[str, Any] | None],
    *,
    cycle_count: int = 7,
) -> dict[str, Any]:
    """All-slot descriptive arithmetic, independent of paired-value eligibility."""
    if (
        len(episode_seeds) != EPISODE_COUNT
        or len(records) != EPISODE_COUNT
        or len(set(episode_seeds)) != EPISODE_COUNT
    ):
        raise ValueError(
            "operational policy summary requires 32 unique ordered episode slots"
        )
    for seed in episode_seeds:
        _count(seed, "episode seed")
    if _count(cycle_count, "cycle count") == 0:
        raise ValueError("cycle count must be positive")
    rows = [None if row is None else deepcopy(dict(row)) for row in records]
    actor = None
    for seed, row in zip(episode_seeds, rows, strict=True):
        if row is None:
            continue
        if (
            row["schema"] != EPISODE_SCHEMA
            or row["operational_contract_sha256"] != operational_contract()["sha256"]
            or row["seed"] != seed
            or row["source_evidence_verified"] is not False
            or row["result_claim"] is not None
            or canonical_bytes(row) != canonical_bytes(authenticate(row))
        ):
            raise ValueError("operational episode identity or digest differs")
        if actor is not None and row["actor"] != actor:
            raise ValueError("operational policy mixes actors")
        actor = row["actor"]
        if len(row["cycles"]) != cycle_count or any(
            c["cycle_index"] != i + 1 for i, c in enumerate(row["cycles"])
        ):
            raise ValueError("operational cycle registry differs")
    observed = [row for row in rows if row is not None]
    flattened = [
        (
            None
            if row is None
            else _numeric_leaves(
                {
                    "physical_metrics": row["physical_metrics"] or {},
                    "economic_metrics": row["economic_metrics"] or {},
                }
            )
        )
        for row in rows
    ]
    for row, metric_values in zip(rows, flattened, strict=True):
        if (
            row is not None
            and row["economic_metrics"] is not None
            and metric_values is not None
        ):
            # The verified initial economic ledger has an empty additive map;
            # its observed zero prefix is distinct from a preflight's null ledger.
            for component in COMPONENT_NAMES:
                metric_values.setdefault(
                    "economic_metrics.components_eur." + component, 0.0
                )
    metric_columns: dict[str, Any] = {}
    for key in _metric_template():
        prefix_values = [
            None if value is None else value.get(key) for value in flattened
        ]
        complete_values = [
            value if row is not None and row["execution_complete"] else None
            for row, value in zip(rows, prefix_values, strict=True)
        ]
        metric_columns[key] = {
            "observed_prefix": {
                "values_by_episode": prefix_values,
                "distribution": distribution(
                    [v for v in prefix_values if v is not None]
                ),
                "missing_count": sum(v is None for v in prefix_values),
            },
            "complete_episode": {
                "values_by_episode": complete_values,
                "distribution": distribution(
                    [v for v in complete_values if v is not None]
                ),
                "missing_count": sum(v is None for v in complete_values),
            },
        }
    resources = {}
    for resource in RESOURCE_NAMES:
        per_cycle = []
        identified = []
        for index in range(cycle_count):
            values = [
                (
                    None
                    if row is None or row["cycles"][index]["resources"] is None
                    else row["cycles"][index]["resources"][resource]
                )
                for row in rows
            ]
            present = [v for v in values if v is not None]
            for seed, value in zip(episode_seeds, values, strict=True):
                if value is not None:
                    identified.append(
                        {
                            "seed": seed,
                            "cycle_index": index + 1,
                            "deficit": finite_real(value["deficit"]),
                        }
                    )
            successes = sum(v["feasible"] is True for v in present)
            per_cycle.append(
                {
                    "cycle_index": index + 1,
                    "values": values,
                    "verified_feasible_over_registered": binomial(
                        successes, EPISODE_COUNT
                    ),
                    "feasible_over_observed": binomial(successes, len(present)),
                    "missing_settlements": EPISODE_COUNT - len(present),
                    "deficits": distribution([v["deficit"] for v in present]),
                }
            )
        resources[resource] = {
            "unit": "kWh" if resource == "battery" else "kg",
            "per_cycle": per_cycle,
            "identified_deficits": identified,
            "observed_deficits": distribution([v["deficit"] for v in identified]),
            "expected_settlements": EPISODE_COUNT * cycle_count,
            "missing_settlements": EPISODE_COUNT * cycle_count - len(identified),
        }
    draws = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, EPISODE_COUNT, size=(BOOTSTRAP_REPLICATES, EPISODE_COUNT)
    )
    strata: dict[str, Any] = {}
    for kind, names in (("outage", ("outage", "no_outage")), ("market", REGIMES)):
        strata[kind] = {}
        for name in names:
            units = (
                [
                    (
                        [c for c in row["cycles"] if c["outage_stratum"] == name]
                        if row is not None
                        else []
                    )
                    for row in rows
                ]
                if kind == "outage"
                else [
                    [row] if row is not None and row["market_regime"] == name else []
                    for row in rows
                ]
            )
            complete_units = (
                [[unit for unit in cluster if unit["complete"]] for cluster in units]
                if kind == "outage"
                else [
                    [unit for unit in cluster if unit["execution_complete"]]
                    for cluster in units
                ]
            )
            value_rows = [
                [
                    (
                        unit["net_operating_value_eur"]
                        if kind == "outage"
                        else unit["economic_metrics"]["net_operating_value_eur"]
                    )
                    for unit in cluster
                ]
                for cluster in complete_units
            ]
            metrics = {"net_operating_value_eur": clustered_summary(value_rows, draws)}
            for component in COMPONENT_NAMES:
                metrics[component] = clustered_summary(
                    [
                        [
                            (
                                unit["components_eur"][component]
                                if kind == "outage"
                                else unit["economic_metrics"]["components_eur"][
                                    component
                                ]
                            )
                            for unit in cluster
                        ]
                        for cluster in complete_units
                    ],
                    draws,
                )
            if kind == "outage":
                for signal in CYCLE_SIGNALS:
                    metrics[signal] = clustered_summary(
                        [
                            [unit["signals"][signal] for unit in cluster]
                            for cluster in complete_units
                        ],
                        draws,
                    )
                for resource in RESOURCE_NAMES:
                    for metric in ("deficit", "feasible"):
                        metrics[f"{resource}_{metric}"] = clustered_summary(
                            [
                                [
                                    float(unit["resources"][resource][metric])
                                    for unit in cluster
                                ]
                                for cluster in complete_units
                            ],
                            draws,
                        )
            else:
                for key, column in metric_columns.items():
                    metrics[key] = clustered_summary(
                        [
                            (
                                [value]
                                if row is not None
                                and row["market_regime"] == name
                                and value is not None
                                else []
                            )
                            for row, value in zip(
                                rows,
                                column["complete_episode"]["values_by_episode"],
                                strict=True,
                            )
                        ],
                        draws,
                    )
            strata[kind][name] = {
                "known_unit_count": sum(map(len, units)),
                "complete_unit_count": sum(map(len, complete_units)),
                "incomplete_unit_count": sum(map(len, units))
                - sum(map(len, complete_units)),
                "episode_seeds": [
                    seed
                    for seed, cluster in zip(episode_seeds, units, strict=True)
                    if cluster
                ],
                "metrics": metrics,
            }
    exposure = {}
    for rate in (
        "guard_correction",
        "unexpected_critical_or_offline",
        "forced_stop",
        "invalid_command",
    ):
        rate_values = [
            row["exposure_rates"][rate]
            for row in observed
            if row["exposure_rates"] is not None
        ]
        exposure[rate] = exposure_rate(
            sum(v["count"] for v in rate_values),
            sum(v["exposure"] for v in rate_values),
        )
    identified_outages = [
        {"seed": seed, **outage}
        for seed, row in zip(episode_seeds, rows, strict=True)
        if row is not None and row["outages"] is not None
        for outage in row["outages"]
    ]
    safety_observed = [
        row for row in observed if row["hard_safety_metrics"] is not None
    ]
    return authenticate(
        {
            "schema": SUMMARY_SCHEMA,
            "operational_contract_sha256": operational_contract()["sha256"],
            "episode_seeds": list(episode_seeds),
            "actor": actor,
            "episode_records": rows,
            "missing_episode_seeds": [
                seed
                for seed, row in zip(episode_seeds, rows, strict=True)
                if row is None
            ],
            "arm_evidence_available": binomial(len(observed), EPISODE_COUNT),
            "execution_complete": binomial(
                sum(row["execution_complete"] is True for row in observed),
                EPISODE_COUNT,
            ),
            "hard_safety_complete_episode": binomial(
                sum(
                    row["hard_safety_passed"] is True
                    and row["execution_complete"] is True
                    for row in observed
                ),
                EPISODE_COUNT,
            ),
            "hard_safety_observed_prefix": binomial(
                sum(row["hard_safety_passed"] is True for row in safety_observed),
                len(safety_observed),
            ),
            "all_cycle_returns_verified": binomial(
                sum(row["cycle_feasible"] is True for row in observed), EPISODE_COUNT
            ),
            "resources": resources,
            "episode_metric_distributions": metric_columns,
            "outage_recovery": {
                "identified_outages": identified_outages,
                "known_manifest_episode_count": sum(
                    row["outages"] is not None for row in observed
                ),
                "missing_manifest_episode_count": EPISODE_COUNT
                - sum(row["outages"] is not None for row in observed),
                "status_counts": {
                    status: sum(row["status"] == status for row in identified_outages)
                    for status in (
                        "not_reached",
                        "outage_incomplete",
                        "resumed",
                        "censored_no_capture",
                    )
                },
                "observed_recovery_delay_s": distribution(
                    [
                        row["recovery_delay_s"]
                        for row in identified_outages
                        if row["recovery_delay_s"] is not None
                    ]
                ),
                "scheduled_duration_s": distribution(
                    [row["scheduled_duration_s"] for row in identified_outages]
                ),
                "encountered_duration_s": distribution(
                    [row["encountered_duration_s"] for row in identified_outages]
                ),
            },
            "exposure_rates": exposure,
            "strata": strata,
            "source_evidence_verified": False,
            "result_claim": None,
        }
    )

"""Frozen admission, aggregation, and gate report for the service-MPC replication.

The registered cell runner deliberately writes each large causal artifact outside Git.
This module admits exactly the six declared artifacts, binds them to the already-frozen
future-information planner cells, recomputes the predeclared utility gates, and publishes
one compact non-overwriting report.  It contains no controller or model tuning surface.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any

from plant_sim import PlantConfig

from ...controllers.base import Constraints
from .config import EpisodeConfigV2
from .controllers import IssuedForecastControllerV2
from .evaluation import (
    CAUSAL_RESULT_SCHEMA_VERSION,
    LegacyHeuristicControllerV2,
    collect_provenance,
)
from .metrics import EpisodeMetricsV2
from .mpc_controller import ServiceMPCControllerV1
from .mpc_execution import (
    SERVICE_MPC_ACTION_DOMAIN,
    SERVICE_MPC_ACTION_SCHEMA,
    SERVICE_MPC_APPLICATION_DOMAIN,
    SERVICE_MPC_DECISION_DOMAIN,
    SERVICE_MPC_FAILURE_DOMAIN,
    SERVICE_MPC_FAILURE_SCHEMA,
    SERVICE_MPC_GUARDED_ACTION_DOMAIN,
    SERVICE_MPC_INTERVAL_POWER_DOMAIN,
    SERVICE_MPC_REQUEST_DOMAIN,
    SERVICE_MPC_RESIDUAL_DOMAIN,
    SERVICE_MPC_RUN_EVIDENCE_DOMAIN,
    SERVICE_MPC_RUN_EVIDENCE_FORMAT,
)
from .mpc_experiment import (
    REGISTERED_MPC_CELLS,
    REGISTERED_MPC_CELL_EXECUTION_FORMAT,
    REGISTERED_MPC_CONTROLLER_IDS,
    REGISTERED_MPC_DECISIONS,
    REGISTERED_MPC_HORIZON_H,
    REGISTERED_MPC_MATRIX_CONTRACT,
    REGISTERED_MPC_MATRIX_CONTRACT_SHA256,
    find_repository_root,
    validate_artifact_destination,
)
from .mpc_schema import (
    MODEL_BALANCE_TOLERANCE,
    SERVICE_MPC_CONTROLLER_ID,
    SERVICE_MPC_SOLVER,
    canonical_json,
    domain_sha256,
    require_sha256,
)
from .reward import RewardWeightsV2


SERVICE_MPC_REPORT_SCHEMA_VERSION = 2
SERVICE_MPC_REPORT_FORMAT = "causal-service-mpc-engineering-replication-report-v2"
SERVICE_MPC_REPORT_DOMAIN = SERVICE_MPC_REPORT_FORMAT
FROZEN_PLANNER_RELATIVE_PATH = Path(
    "docs/research/results/v2_feasible_planner_matrix_564002c.json"
)
FROZEN_PLANNER_RAW_SHA256 = (
    "05bd4b66bf71c01847aad369fe3abbea99fe2efe38a5c15be49b131cb551065e"
)
FROZEN_PLANNER_REPORT_SHA256 = (
    "33f7873b0fba1623d816109900f8a5f6a41929924f6dcf99bff5dedd856b1d64"
)
PLANNER_GAIN_TOLERANCE_KG = 1e-9
MPC_IMPROVEMENT_TOLERANCE_KG = 1e-9
RATE_42_MINIMUM_IMPROVED_CELLS = 2
RATE_42_MINIMUM_MEAN_CAUSAL_CAPTURE = 0.20
RATE_42_MINIMUM_COMMITMENTS_MET = 26
RATE_35_MAXIMUM_MEAN_FILL_LOSS = 0.01
MINIMUM_TERMINAL_BATTERY_SOC = 0.25
MAXIMUM_COLD_STARTS = 4
MAXIMUM_DECISION_P99_NS = 2_000_000_000
MAXIMUM_DECISION_NS = 5_000_000_000

_REPORT_METRICS = (
    "delivered_kg",
    "delivery_fill_rate",
    "unmet_kg",
    "commitments_met",
    "commitments_settled",
    "commitment_service_level",
    "worst_commitment_fill_rate",
    "ch4_produced_kg",
    "terminal_h2_kg",
    "terminal_co2_kg",
    "terminal_ch4_kg",
    "terminal_battery_soc",
    "terminal_battery_energy_kwh",
    "min_battery_soc",
    "solar_kwh",
    "curtailed_kwh",
    "battery_discharged_kwh",
    "battery_throughput_kwh",
    "electrolyser_applied_energy_kwh",
    "dac_applied_energy_kwh",
    "methanator_applied_energy_kwh",
    "accepted_process_energy_kwh",
    "accepted_process_energy_per_delivered_kg",
    "shutdown_steps",
    "guard_interventions",
    "guard_intervention_kwh",
    "dispatch_gap_kwh",
    "request_application_gap_kwh",
    "corrected_physics_steps",
    "out_of_range_decisions",
    "electrolyser_cold_starts",
    "electrolyser_hot_starts",
    "electrolyser_aborted_starts",
    "electrolyser_forced_stops",
    "electrolyser_controlled_stops",
    "electrolyser_startup_energy_kwh",
    "electrolyser_standby_energy_kwh",
    "electrolyser_nonproductive_energy_kwh",
)


@dataclass(frozen=True, slots=True)
class _TraceAudit:
    sha256: str
    count: int
    invalid_count: int
    actions: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _RunAudit:
    controller: str
    successful: bool
    manifest_sha256: str
    exogenous_episode_sha256: str
    metrics: dict[str, Any] | None
    failure: dict[str, Any] | None
    trace: _TraceAudit
    evidence: dict[str, Any] | None
    decision_latencies_ns: tuple[int, ...] = ()
    stage_latencies_ns: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class _CellAudit:
    rate: float
    seed: int
    path: Path
    raw_sha256: str
    byte_count: int
    started_at: datetime
    completed_at: datetime
    source: dict[str, Any]
    dependency_lock: dict[str, Any]
    runtime: dict[str, Any]
    planner: dict[str, Any]
    runs: tuple[_RunAudit, ...]
    requested_action_mechanisms: tuple[dict[str, Any], ...]

    @property
    def all_arms_succeeded(self) -> bool:
        return all(run.successful for run in self.runs)

    def run(self, controller: str) -> _RunAudit:
        matches = tuple(run for run in self.runs if run.controller == controller)
        if len(matches) != 1:  # pragma: no cover - protected by admission
            raise ValueError("admitted cell lost its unique controller arm")
        return matches[0]


@dataclass(frozen=True, slots=True)
class _SuccessEventAudit:
    event: dict[str, Any]
    sha256: str
    action: dict[str, Any]
    residual: dict[str, Any] | None
    decision_latency_ns: int
    stage_latencies_ns: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _ApplicationAudit:
    guard_interventions: int
    corrected_physics_steps: int
    out_of_range: bool
    guard_changed_request: bool
    plant_refused_or_rerouted_power: bool
    guard_gap_kwh: float
    dispatch_gap_kwh: float
    request_gap_kwh: float


@dataclass(frozen=True, slots=True)
class PublishedServiceMPCReportV1:
    """One durable compact matrix report and its exact byte identity."""

    report: dict[str, Any]
    path: Path
    file_sha256: str


def _mapping(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a JSON object")
    return dict(value)


def _sequence(name: str, value: object) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    return value


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}")
    return number


def _utc_instant(name: str, value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be an RFC 3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} must be an RFC 3339 UTC timestamp") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"{name} must be an RFC 3339 UTC timestamp")
    return parsed


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_json_bytes(name: str, raw: bytes) -> dict[str, Any]:
    try:
        decoded = raw.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{name} is not strict unique-key UTF-8 JSON") from exc
    result = _mapping(name, value)
    expected = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if raw != expected:
        raise ValueError(f"{name} is not in the registered canonical file form")
    return result


def _strict_json_path(name: str, path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    return _strict_json_bytes(name, raw), raw


def _expected_controller_specs() -> dict[str, dict[str, object]]:
    controllers = (
        LegacyHeuristicControllerV2(),
        IssuedForecastControllerV2(),
        ServiceMPCControllerV1(),
    )
    return {controller.id: controller.to_dict() for controller in controllers}


def _validate_trace(value: object, *, name: str) -> _TraceAudit:
    trace = _mapping(name, value)
    if set(trace) != {"format", "count", "sha256", "actions"}:
        raise ValueError(f"{name} has an unexpected shape")
    if trace["format"] != "controller-requested-action-v1":
        raise ValueError(f"{name} has the wrong format")
    count = _integer(f"{name} count", trace["count"])
    actions = _sequence(f"{name} actions", trace["actions"])
    if len(actions) != count:
        raise ValueError(f"{name} count does not match its action rows")
    invalid = 0
    rows: list[dict[str, Any]] = []
    fields_expected = {
        "decision_index",
        "battery_share",
        "electrolyser_share",
        "dac_share",
        "methanator_share",
        "battery_support_kw",
        "controller_invalid",
    }
    for index, raw_action in enumerate(actions):
        action = _mapping(f"{name} action {index}", raw_action)
        if set(action) != fields_expected:
            raise ValueError(f"{name} action {index} has an unexpected shape")
        if _integer(f"{name} action index", action["decision_index"]) != index:
            raise ValueError(f"{name} action indices are not contiguous")
        for field_name in fields_expected - {"decision_index", "controller_invalid"}:
            _finite(f"{name} action {field_name}", action[field_name])
        if not isinstance(action["controller_invalid"], bool):
            raise TypeError(f"{name} action invalid flag must be a bool")
        invalid += int(action["controller_invalid"])
        rows.append(action)
    encoded = json.dumps(
        rows,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = require_sha256(f"{name} sha256", trace["sha256"])
    if hashlib.sha256(encoded).hexdigest() != digest:
        raise ValueError(f"{name} digest is invalid")
    return _TraceAudit(digest, count, invalid, tuple(rows))


def _validate_metrics(value: object, *, name: str) -> dict[str, Any]:
    metrics = _mapping(name, value)
    expected_keys = set(EpisodeMetricsV2().to_dict())
    if set(metrics) != expected_keys:
        raise ValueError(f"{name} does not match the current metric schema")
    constructor_names = {field.name for field in fields(EpisodeMetricsV2)}
    rebuilt = EpisodeMetricsV2(
        **{
            field_name: deepcopy(metrics[field_name])
            for field_name in constructor_names
        }
    ).to_dict()
    if canonical_json(rebuilt) != canonical_json(metrics):
        raise ValueError(f"{name} derived fields are internally inconsistent")
    for field_name in _REPORT_METRICS:
        item = metrics[field_name]
        if item is not None:
            _finite(f"{name} {field_name}", item, minimum=0.0)
    reward = _mapping(f"{name} reward components", metrics["reward_components"])
    for key, item in reward.items():
        _finite(f"{name} reward component {key}", item)
    return metrics


def _digest_body(name: str, domain: str, value: object) -> dict[str, Any]:
    item = _mapping(name, value)
    body = deepcopy(item)
    observed = require_sha256(f"{name} sha256", body.pop("sha256", None))
    if domain_sha256(domain, body) != observed:
        raise ValueError(f"{name} digest is invalid")
    return item


def _same_json(left: object, right: object) -> bool:
    """Compare serialized contracts without Python tuple/list representation noise."""

    return bool(canonical_json(left) == canonical_json(right))


def _nearest_rank_latency(values: Sequence[int]) -> dict[str, int | float | str]:
    if not values:
        return {
            "count": 0,
            "mean_ns": 0.0,
            "minimum_ns": 0,
            "p99_ns": 0,
            "maximum_ns": 0,
            "p99_method": "empirical nearest rank",
        }
    ordered = sorted(values)
    rank = max(1, math.ceil(0.99 * len(ordered)))
    return {
        "count": len(ordered),
        "mean_ns": math.fsum(ordered) / len(ordered),
        "minimum_ns": ordered[0],
        "p99_ns": ordered[rank - 1],
        "maximum_ns": ordered[-1],
        "p99_method": "empirical nearest rank",
    }


def _validate_latency_summary(
    name: str,
    value: object,
    *,
    expected: dict[str, int | float | str],
) -> None:
    summary = _mapping(name, value)
    if set(summary) != set(expected):
        raise ValueError(f"{name} shape is invalid")
    for key in ("count", "minimum_ns", "p99_ns", "maximum_ns"):
        _integer(f"{name} {key}", summary.get(key))
    _finite(f"{name} mean_ns", summary.get("mean_ns"), minimum=0.0)
    if summary.get("p99_method") != "empirical nearest rank":
        raise ValueError(f"{name} method is invalid")
    if summary != expected:
        raise ValueError(f"{name} is invalid")


def _validate_stage_record(
    value: object,
    *,
    expected_name: str,
    expected_success: bool | None,
    name: str,
) -> tuple[int, bool]:
    stage = _mapping(name, value)
    if stage.get("stage") != expected_name:
        raise ValueError(f"{name} is out of order")
    success = stage.get("success")
    if not isinstance(success, bool):
        raise TypeError(f"{name} success must be bool")
    if expected_success is not None and success is not expected_success:
        raise ValueError(f"{name} has invalid success status")
    status = stage.get("status")
    if isinstance(status, bool) or not isinstance(status, Integral):
        raise TypeError(f"{name} status must be an integer")
    if stage.get("solver") != SERVICE_MPC_SOLVER:
        raise ValueError(f"{name} solver identity is invalid")
    for key in ("message", "scipy_version", "highs_version"):
        if not isinstance(stage.get(key), str) or not stage[key]:
            raise ValueError(f"{name} {key} is missing")
    if len(stage["message"]) > 500:
        raise ValueError(f"{name} message exceeds the evidence bound")
    if success:
        if status != 0:
            raise ValueError(f"{name} is not an evidenced optimal solve")
        for key in ("objective_value", "mip_dual_bound"):
            _finite(f"{name} {key}", stage.get(key))
        _finite(f"{name} mip_gap", stage.get("mip_gap"), minimum=0.0)
        _integer(f"{name} mip_node_count", stage.get("mip_node_count"))
        require_sha256(f"{name} solution_sha256", stage.get("solution_sha256"))
    else:
        if stage.get("solution_sha256") is not None:
            raise ValueError(f"{name} failed solve publishes a solution digest")
        for key in ("objective_value", "mip_dual_bound", "mip_gap"):
            if stage.get(key) is not None:
                _finite(
                    f"{name} {key}",
                    stage[key],
                    minimum=0.0 if key == "mip_gap" else None,
                )
        if stage.get("mip_node_count") is not None:
            _integer(f"{name} mip_node_count", stage["mip_node_count"])
    return _integer(f"{name} elapsed_ns", stage.get("elapsed_ns")), success


def _validate_stage(value: object, *, expected_name: str, name: str) -> int:
    elapsed_ns, _ = _validate_stage_record(
        value,
        expected_name=expected_name,
        expected_success=True,
        name=name,
    )
    return elapsed_ns


def _validate_success_event(
    value: object,
    *,
    index: int,
    settings_sha256: str,
    trace_action: Mapping[str, Any] | None,
) -> _SuccessEventAudit:
    event = _mapping(f"service-MPC event {index}", value)
    if (
        event.get("schema") != "service-mpc-decision-evidence-v1"
        or event.get("controller_id") != SERVICE_MPC_CONTROLLER_ID
        or event.get("settings_sha256") != settings_sha256
    ):
        raise ValueError(f"service-MPC event {index} identity is invalid")
    if (
        _integer(
            f"service-MPC event {index} decision_index",
            event.get("decision_index"),
        )
        != index
    ):
        raise ValueError(f"service-MPC event {index} identity is invalid")
    expected_horizon = min(96, REGISTERED_MPC_DECISIONS - index)
    if (
        expected_horizon <= 0
        or _integer(
            f"service-MPC event {index} horizon_intervals",
            event.get("horizon_intervals"),
            minimum=1,
        )
        != expected_horizon
    ):
        raise ValueError(f"service-MPC event {index} horizon is invalid")
    stages = _sequence(f"service-MPC event {index} stages", event.get("stages"))
    if len(stages) != 2:
        raise ValueError(f"service-MPC event {index} lacks two solver stages")
    stage_latencies = (
        _validate_stage(
            stages[0],
            expected_name="primary_delivery",
            name=f"service-MPC event {index} primary stage",
        ),
        _validate_stage(
            stages[1],
            expected_name="secondary_tiebreak",
            name=f"service-MPC event {index} secondary stage",
        ),
    )
    decision_latency = _integer(
        f"service-MPC event {index} decision latency",
        event.get("decision_elapsed_ns"),
    )
    if decision_latency < sum(stage_latencies):
        raise ValueError(f"service-MPC event {index} latency is impossible")

    event_identity = deepcopy(event)
    event_digest = require_sha256(
        f"service-MPC event {index} sha256", event_identity.pop("sha256", None)
    )
    event_identity.pop("decision_elapsed_ns", None)
    identity_stages = _sequence(
        f"service-MPC event {index} identity stages",
        event_identity.pop("stages", None),
    )
    event_identity["stage_solution_sha256"] = [
        _mapping("solver stage", stage)["solution_sha256"] for stage in identity_stages
    ]
    if domain_sha256(SERVICE_MPC_DECISION_DOMAIN, event_identity) != event_digest:
        raise ValueError(f"service-MPC event {index} digest is invalid")

    action = _digest_body(
        f"service-MPC event {index} action",
        SERVICE_MPC_ACTION_DOMAIN,
        event.get("action"),
    )
    if action.get("schema") != SERVICE_MPC_ACTION_SCHEMA:
        raise ValueError(f"service-MPC event {index} action schema is invalid")
    action_values = _mapping(
        f"service-MPC event {index} action values", action.get("action")
    )
    action_fields = {
        "battery_share",
        "electrolyser_share",
        "dac_share",
        "methanator_share",
        "battery_support_kw",
    }
    if set(action_values) != action_fields:
        raise ValueError(f"service-MPC event {index} action shape is invalid")
    shares = tuple(
        _finite(
            f"service-MPC event {index} action {key}",
            action_values[key],
            minimum=0.0,
        )
        for key in (
            "battery_share",
            "electrolyser_share",
            "dac_share",
            "methanator_share",
        )
    )
    if any(value > 1.0 for value in shares) or math.fsum(shares) > 1.0:
        raise ValueError(f"service-MPC event {index} action simplex is invalid")
    _finite(
        f"service-MPC event {index} action battery_support_kw",
        action_values["battery_support_kw"],
        minimum=0.0,
    )
    bridge = _mapping(
        f"service-MPC event {index} actuation bridge",
        action.get("actuation_bridge"),
    )
    bridge_fields = {
        "planned_battery_support_kw",
        "battery_support_authority_kw",
        "battery_support_limit_kw",
        "electrolyser_target_kw",
        "authority_rule",
    }
    if set(bridge) != bridge_fields:
        raise ValueError(f"service-MPC event {index} actuation bridge is invalid")
    planned_support = _finite(
        f"service-MPC event {index} planned battery support",
        bridge["planned_battery_support_kw"],
        minimum=0.0,
    )
    authority = _finite(
        f"service-MPC event {index} battery support authority",
        bridge["battery_support_authority_kw"],
        minimum=0.0,
    )
    support_limit = _finite(
        f"service-MPC event {index} battery support limit",
        bridge["battery_support_limit_kw"],
        minimum=0.0,
    )
    target = _finite(
        f"service-MPC event {index} electrolyser target",
        bridge["electrolyser_target_kw"],
        minimum=0.0,
    )
    expected_authority = 0.0 if target == 0.0 else min(target, support_limit)
    if (
        bridge["authority_rule"] != "zero when electrolyser target is zero; otherwise "
        "min(electrolyser target, active support limit)"
        or authority != action_values["battery_support_kw"]
        or not math.isclose(
            authority,
            expected_authority,
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        or planned_support > support_limit + 1e-7
        or planned_support > target + 1e-7
    ):
        raise ValueError(f"service-MPC event {index} actuation bridge is invalid")
    first_step = _mapping(
        f"service-MPC event {index} first step", event.get("first_step")
    )
    first_planned_support = _finite(
        f"service-MPC event {index} first-step battery support",
        first_step.get("battery_support_kw"),
        minimum=0.0,
    )
    first_target = _finite(
        f"service-MPC event {index} first-step startup power",
        first_step.get("electrolyser_startup_kw"),
        minimum=0.0,
    ) + _finite(
        f"service-MPC event {index} first-step productive power",
        first_step.get("electrolyser_productive_kw"),
        minimum=0.0,
    )
    if not math.isclose(
        planned_support,
        first_planned_support,
        rel_tol=0.0,
        abs_tol=1e-7,
    ) or not math.isclose(
        target,
        first_target,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise ValueError(
            f"service-MPC event {index} actuation bridge is not plan-bound"
        )
    if trace_action is not None:
        for key in action_fields:
            if action_values.get(key) != trace_action[key]:
                raise ValueError(
                    f"service-MPC event {index} differs from its requested trace"
                )
    raw_residual = event.get("residual")
    residual = None
    if raw_residual is not None:
        residual = _digest_body(
            f"service-MPC event {index} residual",
            SERVICE_MPC_RESIDUAL_DOMAIN,
            raw_residual,
        )
        if (
            _integer(
                f"service-MPC event {index} residual decision_index",
                residual.get("decision_index"),
            )
            != index
        ):
            raise ValueError(f"service-MPC event {index} residual index is invalid")
    return _SuccessEventAudit(
        event=event,
        sha256=event_digest,
        action=action,
        residual=residual,
        decision_latency_ns=decision_latency,
        stage_latencies_ns=stage_latencies,
    )


def _validate_failure_event(
    value: object,
    *,
    index: int,
) -> tuple[dict[str, Any], tuple[int, ...]]:
    name = f"service-MPC failure event {index}"
    event = _mapping(name, value)
    if (
        event.get("schema") != SERVICE_MPC_FAILURE_SCHEMA
        or event.get("controller_id") != SERVICE_MPC_CONTROLLER_ID
    ):
        raise ValueError(f"{name} identity is invalid")
    if _integer(f"{name} decision_index", event.get("decision_index")) != index:
        raise ValueError(f"{name} identity is invalid")
    phase = event.get("phase")
    valid_phases = {
        "transition_reconciliation",
        "input_validation",
        "forecast_projection",
        "model_construction",
        "solver",
        "action_realization",
        "evidence_construction",
    }
    if phase not in valid_phases:
        raise ValueError(f"{name} phase is invalid")
    for key in ("exception_type", "message"):
        if not isinstance(event.get(key), str) or not event[key]:
            raise ValueError(f"{name} {key} is missing")
    if len(event["message"]) > 500:
        raise ValueError(f"{name} message exceeds the evidence bound")
    _integer(f"{name} elapsed_ns", event.get("elapsed_ns"))
    for key in (
        "observation_sha256",
        "public_forecast_sha256",
        "projection_sha256",
        "model_sha256",
        "plan_sha256",
    ):
        if event.get(key) is not None:
            require_sha256(f"{name} {key}", event[key])
    stages = _sequence(f"{name} stages", event.get("stages"))
    if len(stages) > 2:
        raise ValueError(f"{name} contains too many solver stages")
    expected_names = ("primary_delivery", "secondary_tiebreak")
    latencies: list[int] = []
    successes: list[bool] = []
    for stage_index, raw_stage in enumerate(stages):
        elapsed_ns, success = _validate_stage_record(
            raw_stage,
            expected_name=expected_names[stage_index],
            expected_success=None,
            name=f"{name} stage {stage_index}",
        )
        latencies.append(elapsed_ns)
        successes.append(success)
    if any(not success for success in successes[:-1]):
        raise ValueError(f"{name} failed before its final attempted stage")
    if stages and phase not in {
        "solver",
        "action_realization",
        "evidence_construction",
    }:
        raise ValueError(f"{name} phase cannot contain solver stages")
    if (
        event["exception_type"] == "ServiceMPCSolveError"
        and successes
        and successes[-1]
    ):
        raise ValueError(f"{name} solve failure ends with a successful stage")
    if phase in {"action_realization", "evidence_construction"} and any(
        not success for success in successes
    ):
        raise ValueError(f"{name} post-solve failure contains a failed stage")

    identity = deepcopy(event)
    observed = require_sha256(f"{name} sha256", identity.pop("sha256", None))
    identity.pop("elapsed_ns", None)
    if domain_sha256(SERVICE_MPC_FAILURE_DOMAIN, identity) != observed:
        raise ValueError(f"{name} digest is invalid")
    return event, tuple(latencies)


def _validate_application(
    value: object,
    *,
    index: int,
    event: _SuccessEventAudit,
) -> _ApplicationAudit:
    application = _digest_body(
        f"service-MPC application {index}",
        SERVICE_MPC_APPLICATION_DOMAIN,
        value,
    )
    residual = event.residual
    if residual is None:
        raise ValueError(f"service-MPC application {index} has no public residual")
    if (
        application.get("schema") != "service-mpc-decision-application-v1"
        or application.get("controller_id") != SERVICE_MPC_CONTROLLER_ID
        or application.get("decision_evidence_sha256") != event.sha256
        or application.get("action_sha256") != event.action["sha256"]
        or application.get("residual_sha256") != residual["sha256"]
    ):
        raise ValueError(f"service-MPC application {index} binding is invalid")
    if (
        _integer(
            f"service-MPC application {index} decision_index",
            application.get("decision_index"),
        )
        != index
    ):
        raise ValueError(f"service-MPC application {index} binding is invalid")
    for key, domain in (
        ("request", SERVICE_MPC_REQUEST_DOMAIN),
        ("guarded_action", SERVICE_MPC_GUARDED_ACTION_DOMAIN),
        ("interval_power", SERVICE_MPC_INTERVAL_POWER_DOMAIN),
    ):
        payload = _mapping(f"service-MPC application {index} {key}", application[key])
        if application.get(f"{key}_sha256") != domain_sha256(domain, payload):
            raise ValueError(f"service-MPC application {index} {key} digest is invalid")
    intervention_count = _integer(
        f"service-MPC application {index} interventions",
        application.get("guard_interventions"),
    )
    adjustments = _sequence(
        f"service-MPC application {index} adjustments",
        application.get("adjustments"),
    )
    if len(adjustments) != intervention_count or any(
        not isinstance(item, str) or not item for item in adjustments
    ):
        raise ValueError(f"service-MPC application {index} adjustments are invalid")
    corrected = _integer(
        f"service-MPC application {index} corrected steps",
        application.get("corrected_physics_steps"),
    )
    for key in (
        "out_of_range",
        "guard_changed_request",
        "plant_refused_or_rerouted_power",
    ):
        if not isinstance(application.get(key), bool):
            raise TypeError(f"service-MPC application {index} {key} must be bool")
    guard_gap = _finite(
        f"service-MPC application {index} guard gap",
        application.get("guard_intervention_integral_l1_kwh"),
        minimum=0.0,
    )
    dispatch_gap = _finite(
        f"service-MPC application {index} dispatch gap",
        application.get("dispatch_gap_integral_l1_kwh"),
        minimum=0.0,
    )
    request_gap = _finite(
        f"service-MPC application {index} request gap",
        application.get("request_application_integral_l1_kwh"),
        minimum=0.0,
    )
    expected_guard_change = bool(
        application["out_of_range"]
        or corrected > 0
        or guard_gap > MODEL_BALANCE_TOLERANCE
    )
    if application["guard_changed_request"] != expected_guard_change:
        raise ValueError(f"service-MPC application {index} guard flag is invalid")
    expected_refusal = dispatch_gap > MODEL_BALANCE_TOLERANCE
    if application["plant_refused_or_rerouted_power"] != expected_refusal:
        raise ValueError(f"service-MPC application {index} refusal flag is invalid")
    return _ApplicationAudit(
        guard_interventions=intervention_count,
        corrected_physics_steps=corrected,
        out_of_range=application["out_of_range"],
        guard_changed_request=application["guard_changed_request"],
        plant_refused_or_rerouted_power=application["plant_refused_or_rerouted_power"],
        guard_gap_kwh=guard_gap,
        dispatch_gap_kwh=dispatch_gap,
        request_gap_kwh=request_gap,
    )


def _validate_mpc_run_evidence(
    value: object,
    *,
    trace: _TraceAudit,
    expected_status: str,
    completed_decisions: int,
    runner_failure_phase: str | None,
) -> tuple[dict[str, Any], tuple[int, ...], tuple[int, ...]]:
    evidence = _digest_body(
        "service-MPC run evidence",
        SERVICE_MPC_RUN_EVIDENCE_DOMAIN,
        value,
    )
    expected_scalars: dict[str, object] = {
        "format": SERVICE_MPC_RUN_EVIDENCE_FORMAT,
        "controller_id": SERVICE_MPC_CONTROLLER_ID,
        "status": expected_status,
        "runner_failure_phase": runner_failure_phase,
    }
    for key, expected in expected_scalars.items():
        if evidence.get(key) != expected:
            raise ValueError(f"service-MPC run evidence has invalid {key}")
    if (
        _integer(
            "service-MPC run evidence completed_decisions",
            evidence.get("completed_decisions"),
        )
        != completed_decisions
    ):
        raise ValueError("service-MPC run evidence has invalid completed_decisions")
    if (
        _integer(
            "service-MPC run evidence fallback_action_count",
            evidence.get("fallback_action_count"),
        )
        != 0
    ):
        raise ValueError("service-MPC run evidence has invalid fallback_action_count")
    if trace.count != completed_decisions:
        raise ValueError("service-MPC trace and completed decisions disagree")
    pending = evidence.get("pending_transition")
    if not isinstance(pending, bool):
        raise TypeError("service-MPC pending_transition must be bool")

    events = _sequence("service-MPC events", evidence.get("events"))
    applications = _sequence("service-MPC applications", evidence.get("applications"))
    raw_successes: list[object] = []
    raw_failures: list[object] = []
    for raw_event in events:
        schema = _mapping("service-MPC event", raw_event).get("schema")
        if schema == "service-mpc-decision-evidence-v1" and not raw_failures:
            raw_successes.append(raw_event)
        elif schema == SERVICE_MPC_FAILURE_SCHEMA and not raw_failures:
            raw_failures.append(raw_event)
        else:
            raise ValueError(
                "service-MPC failure event must terminate the event sequence"
            )
    if len(raw_failures) > 1:
        raise ValueError("service-MPC run contains multiple failure events")

    successes_count = len(raw_successes)
    failures_count = len(raw_failures)
    applications_count = len(applications)
    expected_counts = {
        "event_count": len(events),
        "successful_decision_count": successes_count,
        "failed_decision_count": failures_count,
        "application_count": applications_count,
    }
    for key, expected in expected_counts.items():
        if _integer(f"service-MPC run evidence {key}", evidence.get(key)) != expected:
            raise ValueError(f"service-MPC run evidence has invalid {key}")
    if successes_count - completed_decisions not in (0, 1):
        raise ValueError("service-MPC successes and completed decisions disagree")
    if completed_decisions - applications_count not in (0, 1):
        raise ValueError("service-MPC applications and completed decisions disagree")
    if applications_count > successes_count:
        raise ValueError("service-MPC applications exceed successful decisions")
    if expected_status == "complete":
        if (
            runner_failure_phase is not None
            or failures_count
            or pending
            or successes_count != completed_decisions
            or applications_count != completed_decisions
        ):
            raise ValueError("complete service-MPC run evidence is incomplete")
    elif expected_status == "failed":
        if not isinstance(runner_failure_phase, str) or not runner_failure_phase:
            raise ValueError("failed service-MPC run lacks a runner failure phase")
    else:  # pragma: no cover - closed internal call surface
        raise ValueError("service-MPC evidence status is unsupported")

    expected_settings = _mapping(
        "service-MPC settings descriptor",
        _expected_controller_specs()[SERVICE_MPC_CONTROLLER_ID]["settings"],
    )
    settings_sha256 = require_sha256(
        "service-MPC settings descriptor sha256",
        expected_settings.get("sha256"),
    )
    audited_successes = tuple(
        _validate_success_event(
            raw_event,
            index=index,
            settings_sha256=settings_sha256,
            trace_action=trace.actions[index] if index < trace.count else None,
        )
        for index, raw_event in enumerate(raw_successes)
    )
    failure_stage_latencies: tuple[int, ...] = ()
    if raw_failures:
        _, failure_stage_latencies = _validate_failure_event(
            raw_failures[0],
            index=successes_count,
        )
    unreconciled = tuple(item.residual is None for item in audited_successes)
    if pending != bool(audited_successes and unreconciled[-1]):
        raise ValueError("service-MPC pending state and final residual disagree")
    if any(unreconciled[:-1]):
        raise ValueError("only the final service-MPC success may be unreconciled")
    all_reconciled = all(not item for item in unreconciled)
    if evidence.get("all_successful_decisions_reconciled") is not all_reconciled:
        raise ValueError("service-MPC reconciliation summary is invalid")

    audited_applications = tuple(
        _validate_application(
            raw_application,
            index=index,
            event=audited_successes[index],
        )
        for index, raw_application in enumerate(applications)
    )
    decision_latencies = tuple(item.decision_latency_ns for item in audited_successes)
    stage_latencies = (
        tuple(
            latency for item in audited_successes for latency in item.stage_latencies_ns
        )
        + failure_stage_latencies
    )
    expected_aggregates: dict[str, int | float] = {
        "guard_intervention_count": sum(
            item.guard_interventions for item in audited_applications
        ),
        "corrected_physics_steps": sum(
            item.corrected_physics_steps for item in audited_applications
        ),
        "out_of_range_decisions": sum(
            item.out_of_range for item in audited_applications
        ),
        "guard_changed_decisions": sum(
            item.guard_changed_request for item in audited_applications
        ),
        "plant_refusal_or_reroute_decisions": sum(
            item.plant_refused_or_rerouted_power for item in audited_applications
        ),
        "guard_intervention_integral_l1_kwh": math.fsum(
            item.guard_gap_kwh for item in audited_applications
        ),
        "dispatch_gap_integral_l1_kwh": math.fsum(
            item.dispatch_gap_kwh for item in audited_applications
        ),
        "request_application_integral_l1_kwh": math.fsum(
            item.request_gap_kwh for item in audited_applications
        ),
    }
    integer_aggregates = {
        "guard_intervention_count",
        "corrected_physics_steps",
        "out_of_range_decisions",
        "guard_changed_decisions",
        "plant_refusal_or_reroute_decisions",
    }
    for key, expected in expected_aggregates.items():
        observed: int | float
        if key in integer_aggregates:
            observed = _integer(f"service-MPC run evidence {key}", evidence.get(key))
        else:
            observed = _finite(
                f"service-MPC run evidence {key}",
                evidence.get(key),
                minimum=0.0,
            )
        if observed != expected:
            raise ValueError(f"service-MPC run evidence aggregate {key} is invalid")
    _validate_latency_summary(
        "service-MPC decision latency summary",
        evidence.get("decision_latency"),
        expected=_nearest_rank_latency(decision_latencies),
    )
    _validate_latency_summary(
        "service-MPC stage latency summary",
        evidence.get("solver_stage_latency"),
        expected=_nearest_rank_latency(stage_latencies),
    )

    compact = {
        key: deepcopy(evidence[key])
        for key in evidence
        if key not in {"events", "applications"}
    }
    compact["event_ledger_sha256"] = hashlib.sha256(
        canonical_json(events).encode("utf-8")
    ).hexdigest()
    compact["application_ledger_sha256"] = hashlib.sha256(
        canonical_json(applications).encode("utf-8")
    ).hexdigest()
    return compact, decision_latencies, stage_latencies


def _validate_mpc_evidence(
    value: object,
    *,
    trace: _TraceAudit,
) -> tuple[dict[str, Any], tuple[int, ...], tuple[int, ...]]:
    return _validate_mpc_run_evidence(
        value,
        trace=trace,
        expected_status="complete",
        completed_decisions=REGISTERED_MPC_DECISIONS,
        runner_failure_phase=None,
    )


def _validate_failed_run_evidence(
    value: object,
    *,
    trace: _TraceAudit,
    completed_decisions: int,
    runner_failure_phase: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    compact, _, _ = _validate_mpc_run_evidence(
        value,
        trace=trace,
        expected_status="failed",
        completed_decisions=completed_decisions,
        runner_failure_phase=runner_failure_phase,
    )
    return compact


def _validate_run(
    value: object,
    *,
    controller: str,
    seed: int,
) -> _RunAudit:
    run = _mapping(f"{controller} run", value)
    if run.get("controller") != controller or run.get("seed") != seed:
        raise ValueError(f"{controller} run identity is invalid")
    manifest = require_sha256(f"{controller} manifest", run.get("manifest_sha256"))
    exogenous = require_sha256(
        f"{controller} exogenous episode", run.get("exogenous_episode_sha256")
    )
    trace = _validate_trace(
        run.get("requested_action_trace"),
        name=f"{controller} requested-action trace",
    )
    raw_metrics = run.get("metrics")
    if raw_metrics is not None:
        metrics = _validate_metrics(raw_metrics, name=f"{controller} metrics")
        if trace.count != metrics["decisions"]:
            raise ValueError(f"{controller} successful trace length is invalid")
        if run.get("status") is not None or run.get("failure") is not None:
            raise ValueError(f"{controller} successful run contains failure fields")
        evidence: dict[str, Any] | None = None
        decision_latencies: tuple[int, ...] = ()
        stage_latencies: tuple[int, ...] = ()
        if controller == SERVICE_MPC_CONTROLLER_ID:
            evidence, decision_latencies, stage_latencies = _validate_mpc_evidence(
                run.get("controller_evidence"),
                trace=trace,
            )
        elif "controller_evidence" in run:
            raise ValueError(f"{controller} unexpectedly contains controller evidence")
        return _RunAudit(
            controller=controller,
            successful=True,
            manifest_sha256=manifest,
            exogenous_episode_sha256=exogenous,
            metrics=metrics,
            failure=None,
            trace=trace,
            evidence=evidence,
            decision_latencies_ns=decision_latencies,
            stage_latencies_ns=stage_latencies,
        )

    if run.get("status") != "failed":
        raise ValueError(f"{controller} unsuccessful run is not labelled failed")
    failure = _mapping(f"{controller} failure", run.get("failure"))
    completed = _integer(
        f"{controller} completed decisions", failure.get("completed_decisions")
    )
    _finite(f"{controller} elapsed hours", failure.get("elapsed_hours"), minimum=0.0)
    for key in ("phase", "exception_type", "message"):
        if not isinstance(failure.get(key), str) or not failure[key]:
            raise ValueError(f"{controller} failure {key} is missing")
    _mapping(f"{controller} partial metrics", failure.get("partial_metrics"))
    if trace.count != completed:
        raise ValueError(f"{controller} failed trace length is invalid")
    evidence = None
    if controller == SERVICE_MPC_CONTROLLER_ID:
        evidence = _validate_failed_run_evidence(
            run.get("controller_evidence"),
            trace=trace,
            completed_decisions=completed,
            runner_failure_phase=failure["phase"],
        )
    elif "controller_evidence" in run:
        raise ValueError(f"{controller} unexpectedly contains controller evidence")
    return _RunAudit(
        controller=controller,
        successful=False,
        manifest_sha256=manifest,
        exogenous_episode_sha256=exogenous,
        metrics=None,
        failure=failure,
        trace=trace,
        evidence=evidence,
    )


def _validate_source_provenance(
    provenance: dict[str, Any],
    *,
    repository_root: Path,
    rate: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], datetime, datetime]:
    source = _mapping("cell source", provenance.get("source"))
    if (
        source.get("identity_complete") is not True
        or source.get("git_dirty") is not False
        or source.get("untracked_files") != []
        or source.get("tracked_binary_diff_bytes") != 0
    ):
        raise ValueError("cell source is not a complete clean worktree")
    require_sha256("cell worktree identity", source.get("worktree_identity_sha256"))
    require_sha256("cell tracked diff", source.get("tracked_binary_diff_sha256"))
    revision = source.get("git_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("cell source revision is invalid")
    reported_root = source.get("repository_root")
    if not isinstance(reported_root, str) or Path(reported_root).resolve() != (
        repository_root
    ):
        raise ValueError("cell source repository is invalid")

    audit = _mapping(
        "cell execution source audit", provenance.get("execution_source_audit")
    )
    if audit.get("status") != "valid":
        raise ValueError("cell execution source audit is invalid")
    for key in (
        "identity_complete_at_start",
        "identity_complete_at_end",
        "worktree_identity_unchanged",
        "clean_at_start",
        "clean_at_end",
        "require_clean_source",
    ):
        if audit.get(key) is not True:
            raise ValueError(f"cell execution source audit failed {key}")
    worktree_identity = source["worktree_identity_sha256"]
    if (
        audit.get("start_worktree_identity_sha256") != worktree_identity
        or audit.get("end_worktree_identity_sha256") != worktree_identity
    ):
        raise ValueError("cell source audit identities disagree")

    dependency_lock = _mapping(
        "cell dependency lock", provenance.get("dependency_lock")
    )
    if dependency_lock.get("path") != "uv.lock":
        raise ValueError("cell dependency lock path is invalid")
    require_sha256("cell dependency lock", dependency_lock.get("sha256"))
    runtime = _mapping("cell runtime", provenance.get("runtime"))

    metadata = _mapping(
        "registered cell execution", provenance.get("registered_service_mpc_cell")
    )
    expected_index = REGISTERED_MPC_CELLS.index((rate, seed))
    expected_cell = {"delivery_rate_kg_h": rate, "seed": seed}
    if (
        metadata.get("format") != REGISTERED_MPC_CELL_EXECUTION_FORMAT
        or metadata.get("matrix_contract_sha256")
        != REGISTERED_MPC_MATRIX_CONTRACT_SHA256
        or metadata.get("sequence_index") != expected_index
        or metadata.get("cell") != expected_cell
        or set(metadata)
        != {
            "format",
            "matrix_contract_sha256",
            "sequence_index",
            "cell",
            "started_at_utc",
            "evaluation_completed_at_utc",
        }
    ):
        raise ValueError("registered cell execution metadata is invalid")
    started = _utc_instant("cell start", metadata.get("started_at_utc"))
    completed = _utc_instant(
        "cell evaluation completion", metadata.get("evaluation_completed_at_utc")
    )
    if completed < started:
        raise ValueError("cell completion precedes its start")
    return source, dependency_lock, runtime, started, completed


def _validate_contrasts(
    document: dict[str, Any],
    *,
    seed: int,
    runs: tuple[_RunAudit, ...],
) -> tuple[dict[str, Any], ...]:
    contrasts = _sequence("cell contrasts", document.get("contrasts"))
    expected = (
        (IssuedForecastControllerV2.id, LegacyHeuristicControllerV2.id),
        (SERVICE_MPC_CONTROLLER_ID, IssuedForecastControllerV2.id),
    )
    if len(contrasts) != len(expected):
        raise ValueError("cell contrast count is invalid")
    by_controller = {run.controller: run for run in runs}
    all_successful = all(run.successful for run in runs)
    mechanisms: list[dict[str, Any]] = []
    for index, ((treatment, control), raw_contrast) in enumerate(
        zip(expected, contrasts, strict=True)
    ):
        contrast = _mapping(f"cell contrast {index}", raw_contrast)
        if (
            contrast.get("id") != f"{treatment}_minus_{control}"
            or contrast.get("treatment") != treatment
            or contrast.get("control") != control
        ):
            raise ValueError(f"cell contrast {index} identity is invalid")
        expected_status = (
            "valid" if all_successful else "invalid_incomplete_or_mismatched_execution"
        )
        if contrast.get("effect_summary_status") != expected_status:
            raise ValueError(f"cell contrast {index} status is invalid")
        if contrast.get("whole_study_execution_complete") is not all_successful:
            raise ValueError(f"cell contrast {index} completeness is invalid")
        pairs = _sequence(
            f"cell contrast {index} paired execution",
            contrast.get("paired_seed_execution"),
        )
        if (
            len(pairs) != 1
            or _mapping("paired execution", pairs[0]).get("seed") != seed
        ):
            raise ValueError(f"cell contrast {index} seed alignment is invalid")
        mechanism = _mapping(
            f"cell contrast {index} requested-action mechanism",
            contrast.get("requested_action_mechanism"),
        )
        mechanisms.append(deepcopy(mechanism))
        if all_successful:
            metric_table = _mapping(
                f"cell contrast {index} metrics", contrast.get("metrics")
            )
            delivered = _mapping(
                f"cell contrast {index} delivered metric",
                metric_table.get("delivered_kg"),
            )
            rows = _sequence(
                f"cell contrast {index} delivered rows", delivered.get("by_seed")
            )
            if len(rows) != 1:
                raise ValueError(f"cell contrast {index} delivered row is missing")
            row = _mapping("delivered contrast row", rows[0])
            control_metrics = by_controller[control].metrics
            treatment_metrics = by_controller[treatment].metrics
            assert control_metrics is not None and treatment_metrics is not None
            expected_delta = (
                treatment_metrics["delivered_kg"] - control_metrics["delivered_kg"]
            )
            if (
                row.get("seed") != seed
                or row.get("control") != control_metrics["delivered_kg"]
                or row.get("treatment") != treatment_metrics["delivered_kg"]
                or row.get("raw_treatment_minus_control") != expected_delta
            ):
                raise ValueError(f"cell contrast {index} delivered effect is invalid")
    return tuple(mechanisms)


def _validate_a0_replay(
    run: _RunAudit,
    *,
    planner: dict[str, Any],
) -> None:
    if not run.successful:
        return
    assert run.metrics is not None
    current = deepcopy(run.metrics)
    throughput = current.pop("battery_throughput_kwh")
    planner_a0 = _mapping("frozen planner A0", planner.get("a0"))
    planner_metrics_payload = _mapping(
        "frozen planner A0 metrics", planner_a0.get("metrics")
    )
    if planner_metrics_payload.get("domain") != "planner-metrics-v1":
        raise ValueError("frozen planner A0 metric domain is invalid")
    planner_metrics = _mapping(
        "frozen planner A0 metric value", planner_metrics_payload.get("value")
    )
    if canonical_json(current) != canonical_json(planner_metrics):
        raise ValueError("current A0 replay differs from the frozen planner A0")
    if throughput != planner_a0.get("battery_throughput_kwh"):
        raise ValueError("current A0 battery throughput differs from the frozen replay")


def _admit_cell_document(
    document: dict[str, Any],
    *,
    path: Path,
    raw: bytes,
    repository_root: Path,
    rate: float,
    seed: int,
    planner: dict[str, Any],
) -> _CellAudit:
    if document.get("schema_version") != CAUSAL_RESULT_SCHEMA_VERSION:
        raise ValueError("cell causal schema version is invalid")
    if (
        document.get("study") != "environment_v2_commitment_alpha_causal_comparators"
        or document.get("evidence_scope") != "development_comparator"
        or document.get("seeds") != [seed]
    ):
        raise ValueError("cell study scope is invalid")
    plant = PlantConfig()
    episode = EpisodeConfigV2(
        horizon_h=REGISTERED_MPC_HORIZON_H,
        delivery_rate_kg_h=rate,
    ).validate(plant)
    if not _same_json(document.get("episode_config"), episode.to_dict()):
        raise ValueError("cell episode is outside the frozen matrix")
    expected_specs = _expected_controller_specs()
    if not _same_json(document.get("controller_specs"), expected_specs):
        raise ValueError("cell controller descriptors are invalid")
    if document.get("labels") != {
        controller: expected_specs[controller]["label"]
        for controller in REGISTERED_MPC_CONTROLLER_IDS
    }:
        raise ValueError("cell controller labels are invalid")
    if document.get("information_sets") != {
        controller: expected_specs[controller]["information_set"]
        for controller in REGISTERED_MPC_CONTROLLER_IDS
    }:
        raise ValueError("cell controller information sets are invalid")

    raw_runs = _sequence("cell runs", document.get("runs"))
    if len(raw_runs) != len(REGISTERED_MPC_CONTROLLER_IDS):
        raise ValueError("cell does not contain exactly three arms")
    runs = tuple(
        _validate_run(value, controller=controller, seed=seed)
        for value, controller in zip(
            raw_runs, REGISTERED_MPC_CONTROLLER_IDS, strict=True
        )
    )
    manifests = {run.manifest_sha256 for run in runs}
    exogenous = {run.exogenous_episode_sha256 for run in runs}
    planner_identities = _mapping(
        "frozen planner identities", planner.get("identities")
    )
    if (
        len(manifests) != 1
        or len(exogenous) != 1
        or next(iter(manifests)) != planner_identities.get("manifest_sha256")
        or next(iter(exogenous)) != planner_identities.get("exogenous_episode_sha256")
    ):
        raise ValueError("cell identities do not match the frozen planner cell")
    pairing = _mapping("cell pairing", document.get("pairing"))
    for key in (
        "common_random_numbers",
        "exogenous_episode_match_by_seed",
        "full_manifest_match_by_seed",
        "all_arms_present_by_seed",
        "exactly_one_run_per_arm_by_seed",
        "only_declared_seed_controller_runs",
    ):
        if pairing.get(key) is not True:
            raise ValueError(f"cell pairing failed {key}")
    if pairing.get("all_arms_successful_by_seed") is not all(
        run.successful for run in runs
    ):
        raise ValueError("cell pairing success status is inconsistent")

    provenance = _mapping("cell provenance", document.get("provenance"))
    if not _same_json(provenance.get("plant_config"), plant.to_dict()):
        raise ValueError("cell plant configuration is invalid")
    if not _same_json(provenance.get("episode_config"), episode.to_dict()):
        raise ValueError("cell provenance episode is invalid")
    if not _same_json(provenance.get("constraints"), Constraints().to_dict()):
        raise ValueError("cell safety constraints are invalid")
    if not _same_json(provenance.get("reward_weights"), RewardWeightsV2().to_dict()):
        raise ValueError("cell reward weights are invalid")
    source, lock, runtime, started, completed = _validate_source_provenance(
        provenance,
        repository_root=repository_root,
        rate=rate,
        seed=seed,
    )
    expected_lock_sha = hashlib.sha256(
        (repository_root / "uv.lock").read_bytes()
    ).hexdigest()
    if lock.get("sha256") != expected_lock_sha:
        raise ValueError("cell dependency lock is stale")

    execution = _mapping("cell execution summary", document.get("execution"))
    if execution.get("all_planned_arms_succeeded") is not all(
        run.successful for run in runs
    ):
        raise ValueError("cell execution summary is inconsistent")
    mechanisms = _validate_contrasts(document, seed=seed, runs=runs)
    _validate_a0_replay(runs[0], planner=planner)
    canonical_json(document)
    return _CellAudit(
        rate=rate,
        seed=seed,
        path=path,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        byte_count=len(raw),
        started_at=started,
        completed_at=completed,
        source=source,
        dependency_lock=lock,
        runtime=runtime,
        planner=planner,
        runs=runs,
        requested_action_mechanisms=mechanisms,
    )


def load_frozen_planner_cells(
    repository_root: Path,
) -> dict[tuple[float, int], dict[str, Any]]:
    """Load only the exact committed planner reference named by the protocol."""

    root = Path(repository_root).resolve(strict=True)
    path = root / FROZEN_PLANNER_RELATIVE_PATH
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FROZEN_PLANNER_RAW_SHA256:
        raise ValueError("frozen planner report raw identity changed")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("frozen planner report is not strict JSON") from exc
    report = _mapping("frozen planner report", value)
    if (
        report.get("schema_version") != 1
        or report.get("study")
        != "v2_feasible_future_information_planner_development_matrix"
        or report.get("status") != "complete_evidence_ready"
        or report.get("report_sha256") != FROZEN_PLANNER_REPORT_SHA256
    ):
        raise ValueError("frozen planner report identity is invalid")
    matrix = _mapping("frozen planner matrix", report.get("matrix"))
    expected_order = [
        {"delivery_rate_kg_h": rate, "seed": seed}
        for rate, seed in REGISTERED_MPC_CELLS
    ]
    if (
        matrix.get("order") != expected_order
        or matrix.get("horizon_h") != REGISTERED_MPC_HORIZON_H
    ):
        raise ValueError("frozen planner matrix differs from the service-MPC matrix")
    rows = _sequence("frozen planner cells", report.get("cells"))
    if len(rows) != len(REGISTERED_MPC_CELLS):
        raise ValueError("frozen planner matrix does not contain six cells")
    result: dict[tuple[float, int], dict[str, Any]] = {}
    for expected, raw_row in zip(REGISTERED_MPC_CELLS, rows, strict=True):
        row = _mapping("frozen planner cell", raw_row)
        key = (
            _finite("frozen planner rate", row.get("delivery_rate_kg_h")),
            _integer("frozen planner seed", row.get("seed")),
        )
        if key != expected or key in result:
            raise ValueError("frozen planner cell order or uniqueness is invalid")
        identities = _mapping("frozen planner identities", row.get("identities"))
        require_sha256("frozen planner manifest", identities.get("manifest_sha256"))
        require_sha256(
            "frozen planner exogenous episode",
            identities.get("exogenous_episode_sha256"),
        )
        values = _mapping("frozen planner values", row.get("values"))
        for name in (
            "requested_kg",
            "a0_delivered_kg",
            "best_known_feasible_delivered_kg",
            "dual_certified_relaxation_ceiling_kg",
        ):
            _finite(f"frozen planner {name}", values.get(name), minimum=0.0)
        result[key] = row
    return result


def _external_cell_path(
    value: Path | str,
    *,
    repository_root: Path,
) -> Path:
    requested = Path(value)
    if not requested.is_absolute():
        raise ValueError("registered matrix cell paths must be absolute")
    if requested.is_symlink():
        raise ValueError("registered matrix cell paths must not be symlinks")
    path = requested.resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".json":
        raise ValueError("registered matrix cells must be existing JSON files")
    try:
        path.relative_to(repository_root)
    except ValueError:
        return path
    raise ValueError("registered matrix cells must remain outside the repository")


def admit_registered_service_mpc_matrix(
    cell_paths: Sequence[Path | str],
    *,
    repository_root: Path | None = None,
) -> tuple[_CellAudit, ...]:
    """Admit exactly six external artifacts without computing subset statistics."""

    root = find_repository_root(repository_root).resolve(strict=True)
    if len(cell_paths) != len(REGISTERED_MPC_CELLS):
        raise ValueError(
            "registered service-MPC matrix requires exactly six cell paths"
        )
    paths = tuple(
        _external_cell_path(value, repository_root=root) for value in cell_paths
    )
    if len(set(paths)) != len(paths):
        raise ValueError("registered service-MPC matrix cell paths must be unique")
    planner = load_frozen_planner_cells(root)
    cells: list[_CellAudit] = []
    for path, (rate, seed) in zip(paths, REGISTERED_MPC_CELLS, strict=True):
        document, raw = _strict_json_path(
            f"registered service-MPC cell {rate:g}/{seed}", path
        )
        cells.append(
            _admit_cell_document(
                document,
                path=path,
                raw=raw,
                repository_root=root,
                rate=rate,
                seed=seed,
                planner=planner[(rate, seed)],
            )
        )
    first = cells[0]
    previous_completed = first.started_at
    for expected_index, cell in enumerate(cells):
        if (cell.rate, cell.seed) != REGISTERED_MPC_CELLS[expected_index]:
            raise ValueError("registered service-MPC matrix order changed")
        if cell.started_at < previous_completed:
            raise ValueError(
                "registered service-MPC cells overlapped or ran out of order"
            )
        previous_completed = cell.completed_at
        if canonical_json(cell.source) != canonical_json(first.source):
            raise ValueError(
                "registered service-MPC cells used different source revisions"
            )
        if cell.dependency_lock != first.dependency_lock:
            raise ValueError(
                "registered service-MPC cells used different dependency locks"
            )
        if cell.runtime != first.runtime:
            raise ValueError("registered service-MPC cells used different runtimes")
    return tuple(cells)


def _distribution(values: Sequence[float]) -> dict[str, int | float]:
    if not values:
        raise ValueError("cannot summarize an empty distribution")
    numbers = [float(value) for value in values]
    return {
        "count": len(numbers),
        "mean": statistics.fmean(numbers),
        "population_std": statistics.pstdev(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


def _terminal_ch4_equivalent(metrics: Mapping[str, Any]) -> float:
    methanator = PlantConfig().methanator
    result = float(metrics["terminal_ch4_kg"]) + min(
        float(metrics["terminal_h2_kg"]) / methanator.h2_kg_per_kg_ch4,
        float(metrics["terminal_co2_kg"]) / methanator.co2_kg_per_kg_ch4,
    )
    return float(result)


def _compact_run(run: _RunAudit) -> dict[str, Any]:
    result: dict[str, Any] = {
        "controller": run.controller,
        "status": "complete" if run.successful else "failed",
        "manifest_sha256": run.manifest_sha256,
        "exogenous_episode_sha256": run.exogenous_episode_sha256,
        "requested_action_trace": {
            "count": run.trace.count,
            "sha256": run.trace.sha256,
            "controller_invalid_count": run.trace.invalid_count,
        },
    }
    if run.successful:
        assert run.metrics is not None
        result["metrics"] = deepcopy(run.metrics)
        result["terminal_ch4_equivalent_kg"] = _terminal_ch4_equivalent(run.metrics)
    else:
        result["failure"] = deepcopy(run.failure)
    if run.evidence is not None:
        result["controller_evidence"] = deepcopy(run.evidence)
    return result


def _compact_cell(cell: _CellAudit) -> dict[str, Any]:
    planner_values = _mapping("frozen planner values", cell.planner["values"])
    result: dict[str, Any] = {
        "delivery_rate_kg_h": cell.rate,
        "seed": cell.seed,
        "artifact": {
            "filename": cell.path.name,
            "bytes": cell.byte_count,
            "raw_file_sha256": cell.raw_sha256,
        },
        "execution": {
            "started_at_utc": cell.started_at.isoformat().replace("+00:00", "Z"),
            "evaluation_completed_at_utc": cell.completed_at.isoformat().replace(
                "+00:00", "Z"
            ),
            "all_arms_succeeded": cell.all_arms_succeeded,
        },
        "identities": {
            "manifest_sha256": cell.runs[0].manifest_sha256,
            "exogenous_episode_sha256": cell.runs[0].exogenous_episode_sha256,
        },
        "frozen_planner_reference": {
            key: planner_values[key]
            for key in (
                "requested_kg",
                "a0_delivered_kg",
                "best_known_feasible_delivered_kg",
                "dual_certified_relaxation_ceiling_kg",
            )
        },
        "runs": [_compact_run(run) for run in cell.runs],
        "requested_action_mechanisms": [
            deepcopy(value) for value in cell.requested_action_mechanisms
        ],
    }
    if cell.all_arms_succeeded:
        a0 = cell.run(LegacyHeuristicControllerV2.id).metrics
        a2 = cell.run(IssuedForecastControllerV2.id).metrics
        mpc = cell.run(SERVICE_MPC_CONTROLLER_ID).metrics
        assert a0 is not None and a2 is not None and mpc is not None
        gap = float(planner_values["best_known_feasible_delivered_kg"]) - float(
            planner_values["a0_delivered_kg"]
        )
        result["effects"] = {
            "a2_minus_a0_delivered_kg": a2["delivered_kg"] - a0["delivered_kg"],
            "mpc_minus_a2_delivered_kg": mpc["delivered_kg"] - a2["delivered_kg"],
            "mpc_minus_a0_delivered_kg": mpc["delivered_kg"] - a0["delivered_kg"],
            "eligible_planner_gap_kg": (
                gap if gap > PLANNER_GAIN_TOLERANCE_KG else None
            ),
            "causal_capture": (
                (mpc["delivered_kg"] - a0["delivered_kg"]) / gap
                if gap > PLANNER_GAIN_TOLERANCE_KG
                else None
            ),
        }
    else:
        result["effects"] = None
    return result


def _metric_summary(cells: Sequence[_CellAudit]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for rate in (35.0, 42.0):
        rate_cells = [cell for cell in cells if cell.rate == rate]
        controllers: dict[str, Any] = {}
        for controller in REGISTERED_MPC_CONTROLLER_IDS:
            metrics = [cell.run(controller).metrics for cell in rate_cells]
            assert all(value is not None for value in metrics)
            complete_metrics = [value for value in metrics if value is not None]
            summaries: dict[str, Any] = {}
            for name in _REPORT_METRICS:
                values = [
                    float(metric[name])
                    for metric in complete_metrics
                    if metric[name] is not None
                ]
                summaries[name] = {
                    "defined_count": len(values),
                    "distribution": None if not values else _distribution(values),
                }
            equivalents = [
                _terminal_ch4_equivalent(metric) for metric in complete_metrics
            ]
            summaries["terminal_ch4_equivalent_kg"] = {
                "defined_count": len(equivalents),
                "distribution": _distribution(equivalents),
            }
            controllers[controller] = summaries
        result[f"{rate:g}"] = controllers
    return result


def _effect_summary(cells: Sequence[_CellAudit]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for rate in (35.0, 42.0):
        rows = [_compact_cell(cell)["effects"] for cell in cells if cell.rate == rate]
        assert all(isinstance(row, Mapping) for row in rows)
        effects = [dict(row) for row in rows if isinstance(row, Mapping)]
        names = (
            "a2_minus_a0_delivered_kg",
            "mpc_minus_a2_delivered_kg",
            "mpc_minus_a0_delivered_kg",
        )
        rate_result: dict[str, Any] = {
            name: _distribution([float(row[name]) for row in effects]) for name in names
        }
        captures = [
            float(row["causal_capture"])
            for row in effects
            if row["causal_capture"] is not None
        ]
        rate_result["causal_capture"] = {
            "eligible_cells": len(captures),
            "distribution": None if not captures else _distribution(captures),
        }
        result[f"{rate:g}"] = rate_result
    return result


def _gate(
    identifier: str,
    criterion: str,
    status: str,
    observed: object,
) -> dict[str, Any]:
    if status not in {"pass", "fail", "not_evaluated"}:
        raise ValueError("development gate status is invalid")
    return {
        "id": identifier,
        "criterion": criterion,
        "status": status,
        "observed": observed,
    }


def _development_gates(cells: Sequence[_CellAudit]) -> dict[str, Any]:
    all_arms_succeeded = all(cell.all_arms_succeeded for cell in cells)
    mpc_runs = [cell.run(SERVICE_MPC_CONTROLLER_ID) for cell in cells]
    successful_mpc_runs = [run for run in mpc_runs if run.successful]
    completed_decisions = sum(
        len(run.decision_latencies_ns) for run in successful_mpc_runs
    )
    successful_stages = sum(len(run.stage_latencies_ns) for run in successful_mpc_runs)
    fallback_actions = sum(
        int(run.evidence.get("fallback_action_count", 0))
        for run in mpc_runs
        if run.evidence is not None
    )
    execution_observed = {
        "cell_artifacts": len(cells),
        "planned_cell_artifacts": len(REGISTERED_MPC_CELLS),
        "successful_arms": sum(
            int(run.successful) for cell in cells for run in cell.runs
        ),
        "planned_arms": len(cells) * len(REGISTERED_MPC_CONTROLLER_IDS),
        "successful_mpc_rollouts": len(successful_mpc_runs),
        "planned_mpc_rollouts": len(cells),
        "completed_mpc_decisions": completed_decisions,
        "expected_mpc_decisions": len(cells) * REGISTERED_MPC_DECISIONS,
        "successful_solver_stages": successful_stages,
        "expected_solver_stages": len(cells) * REGISTERED_MPC_DECISIONS * 2,
        "fallback_actions": fallback_actions,
        "surviving_subset_aggregation": False,
    }
    gate_one_passed = bool(
        all_arms_succeeded
        and len(successful_mpc_runs) == len(cells)
        and completed_decisions == len(cells) * REGISTERED_MPC_DECISIONS
        and successful_stages == len(cells) * REGISTERED_MPC_DECISIONS * 2
        and fallback_actions == 0
    )
    gates = [
        _gate(
            "G1",
            "All six paired cells and all 1,728 MPC decisions / 3,456 solver "
            "stages are evidence-valid, with zero fallback and no subset aggregate.",
            "pass" if gate_one_passed else "fail",
            execution_observed,
        )
    ]
    if not gate_one_passed:
        reason = {
            "reason": "whole_matrix_execution_incomplete",
            "policy": "No outcome gate or aggregate is computed from surviving arms.",
        }
        for identifier, criterion in (
            (
                "G2",
                "At 42 kg/h, MPC improves over A0 on at least two seeds and mean "
                "causal capture is at least 20%.",
            ),
            (
                "G3",
                "MPC meets at least A0's frozen 26 of 36 commitments at 42 kg/h.",
            ),
            (
                "G4",
                "At 35 kg/h, mean fill loses at most one percentage point and neither "
                "A0-saturated seed loses a complete commitment.",
            ),
            (
                "G5",
                "Every MPC rollout has zero reserve breach, shutdown step, invalid "
                "action, and forced electrolyser stop.",
            ),
            (
                "G6",
                "Every MPC rollout ends at or above 25% battery SOC and has at most "
                "four cold starts.",
            ),
            (
                "G7",
                "Pooled and per-rollout decision latency p99 is at most 2 s and the "
                "maximum is at most 5 s.",
            ),
        ):
            gates.append(_gate(identifier, criterion, "not_evaluated", reason))
        return {
            "status": "failed",
            "all_predeclared_gates_passed": False,
            "gates": gates,
        }

    rate_42 = [cell for cell in cells if cell.rate == 42.0]
    rate_42_rows: list[dict[str, float | int]] = []
    for cell in rate_42:
        a0 = cell.run(LegacyHeuristicControllerV2.id).metrics
        mpc = cell.run(SERVICE_MPC_CONTROLLER_ID).metrics
        assert a0 is not None and mpc is not None
        planner_values = _mapping("frozen planner values", cell.planner["values"])
        planner_gap = float(planner_values["best_known_feasible_delivered_kg"]) - float(
            planner_values["a0_delivered_kg"]
        )
        if planner_gap <= PLANNER_GAIN_TOLERANCE_KG:
            raise ValueError("rate-42 planner gap unexpectedly became ineligible")
        gain = float(mpc["delivered_kg"]) - float(a0["delivered_kg"])
        rate_42_rows.append(
            {
                "seed": cell.seed,
                "mpc_minus_a0_delivered_kg": gain,
                "causal_capture": gain / planner_gap,
            }
        )
    improved = sum(
        float(row["mpc_minus_a0_delivered_kg"]) > MPC_IMPROVEMENT_TOLERANCE_KG
        for row in rate_42_rows
    )
    captures = [float(row["causal_capture"]) for row in rate_42_rows]
    mean_capture = statistics.fmean(captures)
    gate_two_passed = bool(
        improved >= RATE_42_MINIMUM_IMPROVED_CELLS
        and mean_capture >= RATE_42_MINIMUM_MEAN_CAUSAL_CAPTURE
    )
    gates.append(
        _gate(
            "G2",
            "At 42 kg/h, MPC improves over A0 by more than 1e-9 kg on at least "
            "two seeds and mean causal capture is at least 20%.",
            "pass" if gate_two_passed else "fail",
            {
                "improved_cells": improved,
                "required_improved_cells": RATE_42_MINIMUM_IMPROVED_CELLS,
                "mean_causal_capture": mean_capture,
                "minimum_mean_causal_capture": (RATE_42_MINIMUM_MEAN_CAUSAL_CAPTURE),
                "by_seed": rate_42_rows,
            },
        )
    )

    mpc_rate_42_service = 0
    for cell in rate_42:
        metrics = cell.run(SERVICE_MPC_CONTROLLER_ID).metrics
        assert metrics is not None
        mpc_rate_42_service += int(metrics["commitments_met"])
    gate_three_passed = mpc_rate_42_service >= RATE_42_MINIMUM_COMMITMENTS_MET
    gates.append(
        _gate(
            "G3",
            "MPC meets at least A0's frozen 26 of 36 commitments at 42 kg/h.",
            "pass" if gate_three_passed else "fail",
            {
                "mpc_commitments_met": mpc_rate_42_service,
                "commitments_settled": 36,
                "minimum_commitments_met": RATE_42_MINIMUM_COMMITMENTS_MET,
            },
        )
    )

    rate_35 = [cell for cell in cells if cell.rate == 35.0]
    a0_fill: list[float] = []
    mpc_fill: list[float] = []
    for cell in rate_35:
        a0_metrics = cell.run(LegacyHeuristicControllerV2.id).metrics
        mpc_metrics = cell.run(SERVICE_MPC_CONTROLLER_ID).metrics
        assert a0_metrics is not None and mpc_metrics is not None
        a0_fill.append(float(a0_metrics["delivery_fill_rate"]))
        mpc_fill.append(float(mpc_metrics["delivery_fill_rate"]))
    saturated_rows: list[dict[str, int | bool]] = []
    for cell in rate_35:
        planner_a0 = _mapping("frozen planner A0", cell.planner["a0"])
        score = _mapping("frozen planner A0 score", planner_a0["final_score"])
        if score["service_numerator"] == score["service_denominator"]:
            mpc = cell.run(SERVICE_MPC_CONTROLLER_ID).metrics
            assert mpc is not None
            retained = int(mpc["commitments_met"]) >= int(score["service_numerator"])
            saturated_rows.append({"seed": cell.seed, "retained": retained})
    fill_loss = statistics.fmean(a0_fill) - statistics.fmean(mpc_fill)
    gate_four_passed = bool(
        fill_loss <= RATE_35_MAXIMUM_MEAN_FILL_LOSS + 1e-12
        and all(bool(row["retained"]) for row in saturated_rows)
    )
    gates.append(
        _gate(
            "G4",
            "At 35 kg/h, mean fill loses at most one percentage point and neither "
            "A0-saturated seed loses a complete commitment.",
            "pass" if gate_four_passed else "fail",
            {
                "a0_mean_fill": statistics.fmean(a0_fill),
                "mpc_mean_fill": statistics.fmean(mpc_fill),
                "mean_fill_loss": fill_loss,
                "maximum_mean_fill_loss": RATE_35_MAXIMUM_MEAN_FILL_LOSS,
                "a0_saturated_seeds": saturated_rows,
            },
        )
    )

    reserve_soc = Constraints().reserve_soc(PlantConfig())
    safety_rows: list[dict[str, Any]] = []
    for cell in cells:
        run = cell.run(SERVICE_MPC_CONTROLLER_ID)
        assert run.metrics is not None
        reward = _mapping("MPC reward components", run.metrics["reward_components"])
        row = {
            "delivery_rate_kg_h": cell.rate,
            "seed": cell.seed,
            "reserve_reward_component": reward.get("reserve_breach"),
            "min_battery_soc": run.metrics["min_battery_soc"],
            "shutdown_steps": run.metrics["shutdown_steps"],
            "out_of_range_decisions": run.metrics["out_of_range_decisions"],
            "controller_invalid_actions": run.trace.invalid_count,
            "electrolyser_forced_stops": run.metrics["electrolyser_forced_stops"],
        }
        row["passed"] = bool(
            row["reserve_reward_component"] == 0.0
            and float(row["min_battery_soc"]) + 1e-12 >= reserve_soc
            and row["shutdown_steps"] == 0
            and row["out_of_range_decisions"] == 0
            and row["controller_invalid_actions"] == 0
            and row["electrolyser_forced_stops"] == 0
        )
        safety_rows.append(row)
    gate_five_passed = all(bool(row["passed"]) for row in safety_rows)
    gates.append(
        _gate(
            "G5",
            "Every MPC rollout has zero reserve breach, shutdown step, invalid "
            "action, and forced electrolyser stop.",
            "pass" if gate_five_passed else "fail",
            {"active_reserve_soc": reserve_soc, "by_cell": safety_rows},
        )
    )

    terminal_rows: list[dict[str, Any]] = []
    for cell in cells:
        metrics = cell.run(SERVICE_MPC_CONTROLLER_ID).metrics
        assert metrics is not None
        row = {
            "delivery_rate_kg_h": cell.rate,
            "seed": cell.seed,
            "terminal_battery_soc": metrics["terminal_battery_soc"],
            "electrolyser_cold_starts": metrics["electrolyser_cold_starts"],
        }
        row["passed"] = bool(
            float(row["terminal_battery_soc"]) + 1e-12 >= MINIMUM_TERMINAL_BATTERY_SOC
            and int(row["electrolyser_cold_starts"]) <= MAXIMUM_COLD_STARTS
        )
        terminal_rows.append(row)
    gate_six_passed = all(bool(row["passed"]) for row in terminal_rows)
    gates.append(
        _gate(
            "G6",
            "Every MPC rollout ends at or above 25% battery SOC and has at most "
            "four cold starts.",
            "pass" if gate_six_passed else "fail",
            {
                "minimum_terminal_battery_soc": MINIMUM_TERMINAL_BATTERY_SOC,
                "maximum_cold_starts": MAXIMUM_COLD_STARTS,
                "by_cell": terminal_rows,
            },
        )
    )

    latency_rows: list[dict[str, Any]] = []
    pooled_latencies: list[int] = []
    for cell in cells:
        latencies = cell.run(SERVICE_MPC_CONTROLLER_ID).decision_latencies_ns
        pooled_latencies.extend(latencies)
        summary = _nearest_rank_latency(latencies)
        passed = bool(
            int(summary["p99_ns"]) <= MAXIMUM_DECISION_P99_NS
            and int(summary["maximum_ns"]) <= MAXIMUM_DECISION_NS
        )
        latency_rows.append(
            {
                "delivery_rate_kg_h": cell.rate,
                "seed": cell.seed,
                **summary,
                "passed": passed,
            }
        )
    pooled = _nearest_rank_latency(pooled_latencies)
    pooled_passed = bool(
        int(pooled["p99_ns"]) <= MAXIMUM_DECISION_P99_NS
        and int(pooled["maximum_ns"]) <= MAXIMUM_DECISION_NS
    )
    gate_seven_passed = pooled_passed and all(
        bool(row["passed"]) for row in latency_rows
    )
    gates.append(
        _gate(
            "G7",
            "Pooled and per-rollout decision latency p99 is at most 2 s and the "
            "maximum is at most 5 s (empirical nearest-rank p99).",
            "pass" if gate_seven_passed else "fail",
            {
                "maximum_p99_ns": MAXIMUM_DECISION_P99_NS,
                "maximum_single_decision_ns": MAXIMUM_DECISION_NS,
                "pooled": pooled,
                "by_cell": latency_rows,
            },
        )
    )
    passed = all(gate["status"] == "pass" for gate in gates)
    return {
        "status": "passed" if passed else "failed",
        "all_predeclared_gates_passed": passed,
        "gates": gates,
    }


def _current_report_provenance(repository_root: Path) -> dict[str, Any]:
    plant = PlantConfig()
    return _mapping(
        "current report provenance",
        collect_provenance(
            episode=EpisodeConfigV2(
                horizon_h=REGISTERED_MPC_HORIZON_H,
                delivery_rate_kg_h=35.0,
            ).validate(plant),
            plant=plant,
            constraints=Constraints(),
            weights=RewardWeightsV2(),
            root=repository_root,
        ),
    )


def _validate_aggregation_provenance(
    value: object,
    *,
    repository_root: Path,
    cells: Sequence[_CellAudit],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    provenance = _mapping("aggregation provenance", value)
    source = _mapping("aggregation source", provenance.get("source"))
    lock = _mapping("aggregation dependency lock", provenance.get("dependency_lock"))
    runtime = _mapping("aggregation runtime", provenance.get("runtime"))
    if (
        source.get("identity_complete") is not True
        or source.get("git_dirty") is not False
        or source.get("untracked_files") != []
        or source.get("tracked_binary_diff_bytes") != 0
    ):
        raise ValueError("aggregation requires a complete clean source revision")
    if source.get("repository_root") != str(repository_root):
        raise ValueError("aggregation source repository is invalid")
    if canonical_json(source) != canonical_json(cells[0].source):
        raise ValueError("aggregation source differs from cell execution source")
    if lock != cells[0].dependency_lock or runtime != cells[0].runtime:
        raise ValueError("aggregation runtime or dependency lock differs from cells")
    return source, lock, runtime


def build_service_mpc_report(
    cells: Sequence[_CellAudit],
    *,
    repository_root: Path,
    aggregation_provenance: object,
) -> dict[str, Any]:
    """Build the frozen report from a complete admitted matrix."""

    if tuple((cell.rate, cell.seed) for cell in cells) != REGISTERED_MPC_CELLS:
        raise ValueError("service-MPC report requires the complete ordered matrix")
    root = Path(repository_root).resolve(strict=True)
    source, lock, runtime = _validate_aggregation_provenance(
        aggregation_provenance,
        repository_root=root,
        cells=cells,
    )
    all_arms_succeeded = all(cell.all_arms_succeeded for cell in cells)
    development_gate = _development_gates(cells)
    body: dict[str, Any] = {
        "schema_version": SERVICE_MPC_REPORT_SCHEMA_VERSION,
        "format": SERVICE_MPC_REPORT_FORMAT,
        "study": "causal_service_mpc_engineering_replication_matrix",
        "status": (
            "complete_engineering_replication_gate_passed"
            if development_gate["all_predeclared_gates_passed"]
            else "complete_engineering_replication_gate_failed"
        ),
        "evidence_scope": "already_inspected_synthetic_development_cells",
        "disclaimer": (
            "Engineering evidence on six development-contaminated synthetic cells; not "
            "held-out performance, field validation, a financial-return claim, an RL "
            "comparison, or authorization for autonomous plant deployment."
        ),
        "matrix": deepcopy(REGISTERED_MPC_MATRIX_CONTRACT),
        "statistics": {
            "population_standard_deviation": "ddof=0; descriptive only",
            "causal_capture": "(M - A0) / (F - A0) only when F - A0 > 1e-9 kg",
            "mpc_improvement": "M - A0 > 1e-9 kg",
            "latency_p99": "empirical nearest rank",
            "surviving_subset_policy": "forbidden",
            "aggregate_status": (
                "whole_matrix" if all_arms_succeeded else "ineligible"
            ),
        },
        "provenance": {
            "aggregation_source": source,
            "dependency_lock": lock,
            "runtime": runtime,
            "report_builder_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "frozen_planner_report": {
                "path": str(FROZEN_PLANNER_RELATIVE_PATH),
                "raw_file_sha256": FROZEN_PLANNER_RAW_SHA256,
                "report_sha256": FROZEN_PLANNER_REPORT_SHA256,
            },
            "cell_artifact_custody_at_reporting": [str(cell.path) for cell in cells],
        },
        "execution_integrity": {
            "complete_ordered_cell_set": True,
            "nonoverlapping_declared_execution_order": True,
            "common_clean_source_revision": source["git_revision"],
            "common_worktree_identity_sha256": source["worktree_identity_sha256"],
            "common_dependency_lock_sha256": lock["sha256"],
            "all_arms_succeeded": all_arms_succeeded,
            "no_surviving_subset_aggregate": True,
        },
        "development_gate": development_gate,
        "aggregate": (
            {
                "metrics_by_rate_and_controller": _metric_summary(cells),
                "effects_by_rate": _effect_summary(cells),
            }
            if all_arms_succeeded
            else None
        ),
        "cells": [_compact_cell(cell) for cell in cells],
    }
    report = {
        **body,
        "report_sha256": domain_sha256(SERVICE_MPC_REPORT_DOMAIN, body),
    }
    canonical_json(report)
    return report


def _atomic_write_report(report: dict[str, Any], target: Path) -> str:
    raw = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        reopened = _strict_json_bytes(
            "temporary service-MPC report", temporary.read_bytes()
        )
        observed = require_sha256(
            "service-MPC report sha256", reopened.pop("report_sha256", None)
        )
        if domain_sha256(SERVICE_MPC_REPORT_DOMAIN, reopened) != observed:
            raise ValueError("temporary service-MPC report digest is invalid")
        os.link(temporary, target)
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        published = target.read_bytes()
        if published != raw:
            raise ValueError("published service-MPC report bytes changed")
        return hashlib.sha256(published).hexdigest()
    finally:
        temporary.unlink(missing_ok=True)


def publish_registered_service_mpc_report(
    *,
    cell_paths: Sequence[Path | str],
    destination: Path | str,
    repository_root: Path | None = None,
) -> PublishedServiceMPCReportV1:
    """Admit, aggregate, and durably publish the complete registered matrix."""

    root = find_repository_root(repository_root).resolve(strict=True)
    target = validate_artifact_destination(destination, repository_root=root)
    before = _current_report_provenance(root)
    cells = admit_registered_service_mpc_matrix(cell_paths, repository_root=root)
    after_admission = _current_report_provenance(root)
    if before["source"] != after_admission["source"]:
        raise RuntimeError("source changed while admitting service-MPC cells")
    report = build_service_mpc_report(
        cells,
        repository_root=root,
        aggregation_provenance=after_admission,
    )
    file_sha256 = _atomic_write_report(report, target)
    after_publication = _current_report_provenance(root)
    if before["source"] != after_publication["source"]:
        raise RuntimeError("source changed while publishing the service-MPC report")
    return PublishedServiceMPCReportV1(report, target, file_sha256)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m plant_backend.rl.v2.mpc_report",
        description=(
            "Aggregate exactly six preregistered causal service-MPC cell artifacts."
        ),
    )
    parser.add_argument(
        "--cell",
        action="append",
        type=Path,
        required=True,
        help="external cell JSON; repeat exactly six times in 35/0..42/2 order",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        published = publish_registered_service_mpc_report(
            cell_paths=arguments.cell,
            destination=arguments.output,
        )
    except (TypeError, ValueError) as exc:
        message = " ".join(str(exc).split()) or type(exc).__name__
        print(
            f"invalid registered service-MPC matrix: {message[:500]}", file=sys.stderr
        )
        return 2
    except Exception as exc:
        message = " ".join(str(exc).split()) or type(exc).__name__
        print(f"service-MPC report failed: {message[:500]}", file=sys.stderr)
        return 1

    gate = _mapping("published development gate", published.report["development_gate"])
    print(f"written to {published.path}")
    print(f"artifact sha256: {published.file_sha256}")
    print(f"report sha256: {published.report['report_sha256']}")
    for row in _sequence("published gates", gate["gates"]):
        item = _mapping("published gate", row)
        print(f"{item['id']}: {item['status']}")
    return 0 if gate["all_predeclared_gates_passed"] is True else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "FROZEN_PLANNER_RAW_SHA256",
    "FROZEN_PLANNER_RELATIVE_PATH",
    "FROZEN_PLANNER_REPORT_SHA256",
    "PublishedServiceMPCReportV1",
    "SERVICE_MPC_REPORT_FORMAT",
    "SERVICE_MPC_REPORT_SCHEMA_VERSION",
    "admit_registered_service_mpc_matrix",
    "build_parser",
    "build_service_mpc_report",
    "load_frozen_planner_cells",
    "main",
    "publish_registered_service_mpc_report",
]

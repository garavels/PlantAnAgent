"""One-shot development probe for the frozen service-MPC actuation hypothesis.

The probe is intentionally narrower than the registered six-cell runner.  It executes
only rate 35 / seed 0, retains the complete standard causal-evaluation ledger, adds the
predeclared bridge audit and gates, and publishes one non-overwriting external artifact.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from plant_sim import PlantConfig

from ...controllers.base import Constraints
from .config import EpisodeConfigV2
from .evaluation import (
    CAUSAL_RESULT_SCHEMA_VERSION,
    CausalControllerEvaluationV2,
    BaselineRun,
    collect_provenance,
    evaluate_controllers,
)
from .mpc_controller import ServiceMPCControllerV1
from .mpc_execution import (
    SERVICE_MPC_ACTION_DOMAIN,
    SERVICE_MPC_ACTION_SCHEMA,
    SERVICE_MPC_APPLICATION_DOMAIN,
    SERVICE_MPC_DECISION_DOMAIN,
    SERVICE_MPC_GUARDED_ACTION_DOMAIN,
    SERVICE_MPC_INTERVAL_POWER_DOMAIN,
    SERVICE_MPC_REQUEST_DOMAIN,
    SERVICE_MPC_RESIDUAL_DOMAIN,
    SERVICE_MPC_RUN_EVIDENCE_DOMAIN,
    SERVICE_MPC_RUN_EVIDENCE_FORMAT,
)
from .mpc_experiment import find_repository_root, validate_artifact_destination
from .mpc_schema import (
    DEFAULT_SERVICE_MPC_SETTINGS_V1,
    SERVICE_MPC_CONTROLLER_ID,
    canonical_json,
    domain_sha256,
)
from .reward import RewardWeightsV2


BRIDGE_PROBE_FORMAT = "service-mpc-actuation-bridge-mechanism-probe-v1"
BRIDGE_PROBE_DOMAIN = BRIDGE_PROBE_FORMAT
BRIDGE_PROTOCOL_PATH = "docs/research/CAUSAL_SERVICE_MPC_ACTUATION_BRIDGE_PROTOCOL.md"
BRIDGE_PROTOCOL_SHA256 = (
    "653b118a5d0003e0cb5a37a8076a0f9d765c8eaa7db9a9b759616bf501f981c0"
)
BRIDGE_DELIVERY_RATE_KG_H = 35.0
BRIDGE_SEED = 0
BRIDGE_HORIZON_H = 72.0
BRIDGE_DECISIONS = 288
BRIDGE_SOLVER_STAGES = 576
BRIDGE_PRIOR_DELIVERED_KG = 647.7138003263635
BRIDGE_PRIOR_FORCED_STOPS = 41
BRIDGE_PRIOR_ABORTED_STARTS = 41
BRIDGE_EXPECTED_MANIFEST_SHA256 = (
    "1e036f69c0514ad34a307b2d213fab3c781561ab6ad1efd199ed8313eca8dae4"
)
BRIDGE_EXPECTED_EXOGENOUS_SHA256 = (
    "2e0b7fd35b85c2a0b9d220c32151a00f96adbd7e24e610bb3b17740345903368"
)
BRIDGE_CLAIM_BOUNDARY = (
    "single previously inspected development cell; mechanism diagnosis only; "
    "not replication, holdout, controller-selection, or commercial evidence"
)

BRIDGE_PROBE_CONTRACT = {
    "format": BRIDGE_PROBE_FORMAT,
    "protocol": {
        "path": BRIDGE_PROTOCOL_PATH,
        "raw_file_sha256": BRIDGE_PROTOCOL_SHA256,
    },
    "claim_boundary": BRIDGE_CLAIM_BOUNDARY,
    "prior_result": {
        "source_revision": "3796eb5c935b736f45d245f50175e96b2e776179",
        "report_sha256": (
            "5a5acf5f46ea0363bfee90925b1346d33b11ffc0766fb214ee55d8baaee2df13"
        ),
        "cell_raw_sha256": (
            "80f5d27af14d4d7633e194ed49307b31de96bd2c164485fcba51762d99333a13"
        ),
        "delivered_kg": BRIDGE_PRIOR_DELIVERED_KG,
        "forced_stops": BRIDGE_PRIOR_FORCED_STOPS,
        "aborted_starts": BRIDGE_PRIOR_ABORTED_STARTS,
    },
    "cell": {
        "delivery_rate_kg_h": BRIDGE_DELIVERY_RATE_KG_H,
        "seed": BRIDGE_SEED,
        "horizon_h": BRIDGE_HORIZON_H,
        "decision_interval_s": 900.0,
        "expected_decisions": BRIDGE_DECISIONS,
        "expected_solver_stages": BRIDGE_SOLVER_STAGES,
        "manifest_sha256": BRIDGE_EXPECTED_MANIFEST_SHA256,
        "exogenous_episode_sha256": BRIDGE_EXPECTED_EXOGENOUS_SHA256,
    },
    "controller": {
        "id": SERVICE_MPC_CONTROLLER_ID,
        "settings_sha256": DEFAULT_SERVICE_MPC_SETTINGS_V1.sha256,
        "action_realization_schema": SERVICE_MPC_ACTION_SCHEMA,
    },
    "permitted_change": (
        "first-action realization separates planned battery discharge from bounded "
        "contingent support authority; model, plan, forecast, objective, safety, and "
        "plant are unchanged"
    ),
    "gates": {
        "B1": {
            "decisions": BRIDGE_DECISIONS,
            "solver_stages": BRIDGE_SOLVER_STAGES,
            "fallback_actions": 0,
        },
        "B2": {
            "minimum_operating_time_s_exclusive": 0.0,
            "maximum_forced_stops": 4,
            "maximum_aborted_starts": 4,
        },
        "B3": {
            "minimum_soc": 0.15,
            "minimum_terminal_soc": 0.25,
            "maximum_shutdown_steps": 0,
            "maximum_out_of_range_decisions": 0,
        },
        "B4": {"minimum_delivered_kg_exclusive": BRIDGE_PRIOR_DELIVERED_KG},
        "B5": {"maximum_accounting_violations": 0},
        "B6": {
            "maximum_p99_ns": 2_000_000_000,
            "maximum_single_decision_ns": 5_000_000_000,
        },
    },
    "execution_unit": "one fresh process; no same-revision retry",
}
BRIDGE_PROBE_CONTRACT_SHA256 = domain_sha256(
    "service-mpc-actuation-bridge-mechanism-probe-contract-v1",
    BRIDGE_PROBE_CONTRACT,
)

Evaluator = Callable[..., CausalControllerEvaluationV2]


@dataclass(frozen=True, slots=True)
class PublishedBridgeProbeV1:
    """One externally published mechanism-probe artifact."""

    path: Path
    raw_file_sha256: str
    report_sha256: str
    status: str
    gates: tuple[dict[str, Any], ...]
    elapsed_ns: int

    def __post_init__(self) -> None:
        path = Path(self.path)
        if not path.is_absolute():
            raise ValueError("published bridge-probe path must be absolute")
        object.__setattr__(self, "path", path.resolve(strict=False))
        _require_sha256("raw_file_sha256", self.raw_file_sha256)
        _require_sha256("report_sha256", self.report_sha256)
        if self.status not in {
            "execution_failed",
            "complete_all_gates_passed",
            "complete_gate_failed",
        }:
            raise ValueError("published bridge-probe status is invalid")
        if (
            not isinstance(self.gates, tuple)
            or len(self.gates) != 6
            or [gate.get("id") for gate in self.gates]
            != [f"B{index}" for index in range(1, 7)]
        ):
            raise ValueError("published bridge-probe gates are invalid")
        if (
            isinstance(self.elapsed_ns, bool)
            or not isinstance(self.elapsed_ns, Integral)
            or self.elapsed_ns < 0
        ):
            raise ValueError("elapsed_ns must be a non-negative integer")


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _sequence(name: str, value: object) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    return value


def _finite(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}")
    return number


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 digest") from exc
    return value


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _validate_protocol(root: Path) -> None:
    protocol = root / BRIDGE_PROTOCOL_PATH
    if not protocol.is_file():
        raise ValueError("frozen actuation-bridge protocol is missing")
    observed = hashlib.sha256(protocol.read_bytes()).hexdigest()
    if observed != BRIDGE_PROTOCOL_SHA256:
        raise ValueError("frozen actuation-bridge protocol digest changed")


def _validate_source(result: CausalControllerEvaluationV2, root: Path) -> None:
    provenance = _mapping("probe provenance", result.provenance)
    source = _mapping("probe source", provenance.get("source"))
    audit = _mapping("probe source audit", provenance.get("execution_source_audit"))
    if (
        source.get("identity_complete") is not True
        or source.get("git_dirty") is not False
    ):
        raise ValueError("probe did not start from a complete clean source")
    for name in (
        "identity_complete_at_start",
        "identity_complete_at_end",
        "worktree_identity_unchanged",
        "clean_at_start",
        "clean_at_end",
        "require_clean_source",
    ):
        if audit.get(name) is not True:
            raise ValueError(f"probe source audit failed: {name}")
    if audit.get("status") != "valid":
        raise ValueError("probe source audit status is invalid")
    start_identity = _require_sha256(
        "probe start identity", audit.get("start_worktree_identity_sha256")
    )
    end_identity = _require_sha256(
        "probe end identity", audit.get("end_worktree_identity_sha256")
    )
    if (
        start_identity != end_identity
        or source.get("worktree_identity_sha256") != start_identity
    ):
        raise ValueError("probe source identities disagree")
    reported_root = source.get("repository_root")
    if not isinstance(reported_root, str) or Path(reported_root).resolve() != root:
        raise ValueError("probe reports a different repository root")
    lock = _mapping("probe dependency lock", provenance.get("dependency_lock"))
    if (
        lock.get("path") != "uv.lock"
        or lock.get("sha256")
        != hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    ):
        raise ValueError("probe dependency-lock identity is invalid")


def _validate_result(
    result: CausalControllerEvaluationV2,
    *,
    root: Path,
) -> tuple[BaselineRun, Mapping[str, Any]]:
    if not isinstance(result, CausalControllerEvaluationV2):
        raise TypeError("probe evaluator must return CausalControllerEvaluationV2")
    expected_episode = EpisodeConfigV2(
        horizon_h=BRIDGE_HORIZON_H,
        delivery_rate_kg_h=BRIDGE_DELIVERY_RATE_KG_H,
    ).validate(PlantConfig())
    if result.seeds != [BRIDGE_SEED] or result.episode_config != expected_episode:
        raise ValueError("probe result escaped the frozen episode cell")
    if result.result_schema_version != CAUSAL_RESULT_SCHEMA_VERSION:
        raise ValueError("probe result uses the wrong causal schema")
    expected_descriptor = ServiceMPCControllerV1().to_dict()
    if result.controller_specs != {SERVICE_MPC_CONTROLLER_ID: expected_descriptor}:
        raise ValueError("probe controller descriptor is invalid")
    if result.contrasts:
        raise ValueError(
            "single-controller mechanism probe must not publish a contrast"
        )
    if len(result.runs) != 1:
        raise ValueError("probe must contain exactly one controller run")
    run = result.runs[0]
    if run.controller != SERVICE_MPC_CONTROLLER_ID or run.seed != BRIDGE_SEED:
        raise ValueError("probe run identity is invalid")
    if run.manifest_sha256 != BRIDGE_EXPECTED_MANIFEST_SHA256:
        raise ValueError("probe manifest differs from the frozen comparator cell")
    if run.exogenous_episode_sha256 != BRIDGE_EXPECTED_EXOGENOUS_SHA256:
        raise ValueError(
            "probe exogenous episode differs from the frozen comparator cell"
        )
    if run.requested_actions is None:
        raise ValueError("probe must retain its requested-action trace")
    evidence = _mapping("probe controller evidence", run.controller_evidence)
    if evidence.get("format") != SERVICE_MPC_RUN_EVIDENCE_FORMAT:
        raise ValueError("probe controller evidence format is invalid")
    if evidence.get("fallback_action_count") != 0:
        raise ValueError("probe used a hidden fallback")
    _validate_source(result, root)
    canonical_json(result.to_dict())
    return run, evidence


def _bridge_audit(
    run: BaselineRun,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    events = _sequence("probe events", evidence.get("events"))
    applications = _sequence("probe applications", evidence.get("applications"))
    requested = () if run.requested_actions is None else run.requested_actions
    comparable = min(len(events), len(applications), len(requested))
    violations: list[str] = []
    evidence_body = dict(evidence)
    evidence_sha = evidence_body.pop("sha256", None)
    if (
        not isinstance(evidence_sha, str)
        or domain_sha256(SERVICE_MPC_RUN_EVIDENCE_DOMAIN, evidence_body) != evidence_sha
    ):
        violations.append("run evidence digest")
    planned_support_kwh = 0.0
    authority_capacity_kwh = 0.0
    guarded_support_kwh = 0.0
    applied_support_kwh = 0.0
    active_target_decisions = 0
    authority_above_expected_decisions = 0

    for index in range(comparable):
        event = _mapping(f"probe event {index}", events[index])
        application = _mapping(f"probe application {index}", applications[index])
        event_identity = dict(event)
        event_sha = event_identity.pop("sha256", None)
        event_identity.pop("decision_elapsed_ns", None)
        event_stages = _sequence(
            "probe event stages", event_identity.pop("stages", None)
        )
        event_identity["stage_solution_sha256"] = [
            _mapping("probe event stage", stage).get("solution_sha256")
            for stage in event_stages
        ]
        if (
            not isinstance(event_sha, str)
            or domain_sha256(SERVICE_MPC_DECISION_DOMAIN, event_identity) != event_sha
        ):
            violations.append(f"decision {index}: decision digest")
        action_record = _mapping(f"probe event {index} action", event.get("action"))
        action_sha = _require_sha256(
            f"probe event {index} action sha256", action_record.get("sha256")
        )
        action_body = dict(action_record)
        del action_body["sha256"]
        if (
            action_body.get("schema") != SERVICE_MPC_ACTION_SCHEMA
            or domain_sha256(SERVICE_MPC_ACTION_DOMAIN, action_body) != action_sha
        ):
            violations.append(f"decision {index}: action schema or digest")
        action = _mapping(
            f"probe event {index} low-level action", action_body.get("action")
        )
        bridge = _mapping(
            f"probe event {index} actuation bridge",
            action_body.get("actuation_bridge"),
        )
        first = _mapping(f"probe event {index} first step", event.get("first_step"))
        residual_record = _mapping(
            f"probe event {index} residual", event.get("residual")
        )
        residual_body = dict(residual_record)
        residual_sha = residual_body.pop("sha256", None)
        if (
            not isinstance(residual_sha, str)
            or domain_sha256(SERVICE_MPC_RESIDUAL_DOMAIN, residual_body) != residual_sha
        ):
            violations.append(f"decision {index}: residual digest")
        planned = _finite(
            f"probe event {index} planned support",
            bridge.get("planned_battery_support_kw"),
            minimum=0.0,
        )
        authority = _finite(
            f"probe event {index} support authority",
            bridge.get("battery_support_authority_kw"),
            minimum=0.0,
        )
        support_limit = _finite(
            f"probe event {index} support limit",
            bridge.get("battery_support_limit_kw"),
            minimum=0.0,
        )
        target = _finite(
            f"probe event {index} electrolyser target",
            bridge.get("electrolyser_target_kw"),
            minimum=0.0,
        )
        expected_authority = 0.0 if target == 0.0 else min(target, support_limit)
        first_planned = _finite(
            f"probe event {index} first planned support",
            first.get("battery_support_kw"),
            minimum=0.0,
        )
        first_target = _finite(
            f"probe event {index} first startup power",
            first.get("electrolyser_startup_kw"),
            minimum=0.0,
        ) + _finite(
            f"probe event {index} first productive power",
            first.get("electrolyser_productive_kw"),
            minimum=0.0,
        )
        request = _mapping(
            f"probe application {index} request", application.get("request")
        )
        guarded_action = _mapping(
            f"probe application {index} guarded action",
            application.get("guarded_action"),
        )
        request_action = _mapping(
            f"probe application {index} request action", request.get("action")
        )
        interval = _mapping(
            f"probe application {index} interval power",
            application.get("interval_power"),
        )
        application_body = dict(application)
        application_sha = application_body.pop("sha256", None)
        if (
            application_body.get("schema") != "service-mpc-decision-application-v1"
            or not isinstance(application_sha, str)
            or domain_sha256(SERVICE_MPC_APPLICATION_DOMAIN, application_body)
            != application_sha
        ):
            violations.append(f"decision {index}: application schema or digest")
        for name, payload, domain in (
            ("request", request, SERVICE_MPC_REQUEST_DOMAIN),
            ("guarded_action", guarded_action, SERVICE_MPC_GUARDED_ACTION_DOMAIN),
            ("interval_power", interval, SERVICE_MPC_INTERVAL_POWER_DOMAIN),
        ):
            if application.get(f"{name}_sha256") != domain_sha256(domain, payload):
                violations.append(f"decision {index}: {name} digest")
        hours = _finite(
            f"probe application {index} hours", interval.get("hours"), minimum=0.0
        )
        guarded_kwh = _mapping(
            f"probe application {index} guarded kWh", interval.get("guarded_kwh")
        )
        applied_kwh = _mapping(
            f"probe application {index} applied kWh", interval.get("applied_kwh")
        )
        guarded_support = _finite(
            f"probe application {index} guarded support kWh",
            guarded_kwh.get("battery_support"),
            minimum=0.0,
        )
        applied_support = _finite(
            f"probe application {index} applied support kWh",
            applied_kwh.get("battery_support"),
            minimum=0.0,
        )
        requested_authority = _finite(
            f"probe application {index} requested support authority",
            request_action.get("battery_support_kw"),
            minimum=0.0,
        )
        trace_authority = float(requested[index].battery_support_kw)
        tolerance = 1e-7 + 1e-7 * max(1.0, authority * hours)
        if not (
            application.get("decision_evidence_sha256") == event_sha
            and application.get("action_sha256") == action_sha
            and application.get("residual_sha256") == residual_sha
        ):
            violations.append(f"decision {index}: decision/application binding")
        if not math.isclose(planned, first_planned, rel_tol=0.0, abs_tol=1e-7):
            violations.append(f"decision {index}: planned-support/plan binding")
        if not math.isclose(target, first_target, rel_tol=0.0, abs_tol=1e-7):
            violations.append(f"decision {index}: target/plan binding")
        if not math.isclose(authority, expected_authority, rel_tol=0.0, abs_tol=1e-7):
            violations.append(f"decision {index}: authority rule")
        if planned > authority + 1e-7:
            violations.append(f"decision {index}: planned support exceeds authority")
        if not (
            requested_authority == authority
            and trace_authority == authority
            and action.get("battery_support_kw") == authority
        ):
            violations.append(f"decision {index}: requested authority binding")
        if guarded_support > authority * hours + tolerance:
            violations.append(f"decision {index}: guarded support exceeds authority")
        if applied_support > guarded_support + tolerance:
            violations.append(
                f"decision {index}: applied support exceeds guarded support"
            )

        planned_support_kwh += planned * hours
        authority_capacity_kwh += authority * hours
        guarded_support_kwh += guarded_support
        applied_support_kwh += applied_support
        active_target_decisions += int(target > 0.0)
        authority_above_expected_decisions += int(authority > planned + 1e-9)

    return {
        "events": len(events),
        "applications": len(applications),
        "requested_actions": len(requested),
        "comparable_decisions": comparable,
        "active_target_decisions": active_target_decisions,
        "authority_above_expected_decisions": authority_above_expected_decisions,
        "planned_support_kwh": planned_support_kwh,
        "authority_capacity_kwh": authority_capacity_kwh,
        "guarded_support_kwh": guarded_support_kwh,
        "applied_support_kwh": applied_support_kwh,
        "accounting_violation_count": len(violations),
        "accounting_violations": violations[:20],
        "all_accounting_violations_retained": len(violations) <= 20,
    }


def _gate(
    gate_id: str,
    criterion: str,
    passed: bool | None,
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "id": gate_id,
        "criterion": criterion,
        "status": "not_evaluated" if passed is None else ("pass" if passed else "fail"),
        "observed": dict(observed),
    }


def _build_gates(
    run: BaselineRun,
    evidence: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    metrics = run.metrics
    stage_count = 0
    valid_stage_pairs = True
    for raw_event in _sequence("probe events", evidence.get("events")):
        event = _mapping("probe event", raw_event)
        if event.get("schema") == "service-mpc-decision-evidence-v1":
            stages = _sequence("probe stages", event.get("stages"))
            stage_names = [
                _mapping("probe solver stage", stage).get("stage") for stage in stages
            ]
            pair_is_valid = stage_names == [
                "primary_delivery",
                "secondary_tiebreak",
            ] and all(
                _mapping("probe solver stage", stage).get("success") is True
                for stage in stages
            )
            valid_stage_pairs = valid_stage_pairs and pair_is_valid
            stage_count += sum(
                _mapping("probe solver stage", stage).get("success") is True
                for stage in stages
            )
    execution_passed = bool(
        metrics is not None
        and metrics.decisions == BRIDGE_DECISIONS
        and evidence.get("status") == "complete"
        and evidence.get("completed_decisions") == BRIDGE_DECISIONS
        and evidence.get("successful_decision_count") == BRIDGE_DECISIONS
        and evidence.get("failed_decision_count") == 0
        and evidence.get("application_count") == BRIDGE_DECISIONS
        and evidence.get("all_successful_decisions_reconciled") is True
        and evidence.get("pending_transition") is False
        and evidence.get("fallback_action_count") == 0
        and valid_stage_pairs
        and stage_count == BRIDGE_SOLVER_STAGES
    )
    gates: list[dict[str, Any]] = [
        _gate(
            "B1",
            "288 decisions, 576 successful stages, all transitions reconciled, zero fallback.",
            execution_passed,
            {
                "run_successful": run.successful,
                "completed_decisions": evidence.get("completed_decisions"),
                "successful_solver_stages": stage_count,
                "fallback_actions": evidence.get("fallback_action_count"),
            },
        )
    ]
    if not execution_passed or metrics is None:
        for gate_id, criterion in (
            (
                "B2",
                "At least one productive interval and at most four forced/aborted starts.",
            ),
            ("B3", "Reserve, terminal SOC, validity, and shutdown limits."),
            ("B4", "Delivery strictly exceeds the frozen 647.7138003263635 kg result."),
            (
                "B5",
                "Planned, authorized, guarded, and applied support accounting closes.",
            ),
            ("B6", "Decision p99 at most 2 s and maximum at most 5 s."),
        ):
            gates.append(_gate(gate_id, criterion, None, {}))
        return tuple(gates)

    operating_time_s = float(metrics.electrolyser_operating_time_s)
    gates.append(
        _gate(
            "B2",
            "At least one productive interval and at most four forced/aborted starts.",
            operating_time_s > 0.0
            and metrics.electrolyser_forced_stops <= 4
            and metrics.electrolyser_aborted_starts <= 4,
            {
                "electrolyser_operating_time_s": operating_time_s,
                "electrolyser_cold_starts": metrics.electrolyser_cold_starts,
                "electrolyser_forced_stops": metrics.electrolyser_forced_stops,
                "electrolyser_aborted_starts": metrics.electrolyser_aborted_starts,
                "prior_forced_stops": BRIDGE_PRIOR_FORCED_STOPS,
                "prior_aborted_starts": BRIDGE_PRIOR_ABORTED_STARTS,
            },
        )
    )
    reserve_component = _finite(
        "reserve reward component",
        metrics.reward_components.get("reserve_breach", 0.0),
    )
    gates.append(
        _gate(
            "B3",
            "No reserve breach, invalid action, or shutdown; terminal SOC at least 25%.",
            reserve_component == 0.0
            and metrics.min_battery_soc + 1e-12 >= 0.15
            and metrics.terminal_battery_soc + 1e-12 >= 0.25
            and metrics.shutdown_steps == 0
            and metrics.out_of_range_decisions == 0,
            {
                "reserve_reward_component": reserve_component,
                "min_battery_soc": metrics.min_battery_soc,
                "terminal_battery_soc": metrics.terminal_battery_soc,
                "shutdown_steps": metrics.shutdown_steps,
                "out_of_range_decisions": metrics.out_of_range_decisions,
            },
        )
    )
    gates.append(
        _gate(
            "B4",
            "Delivery strictly exceeds the frozen 647.7138003263635 kg result.",
            metrics.delivered_kg > BRIDGE_PRIOR_DELIVERED_KG,
            {
                "delivered_kg": metrics.delivered_kg,
                "prior_delivered_kg": BRIDGE_PRIOR_DELIVERED_KG,
                "delta_kg": metrics.delivered_kg - BRIDGE_PRIOR_DELIVERED_KG,
            },
        )
    )
    accounting_violations = _integer(
        "accounting_violation_count", audit.get("accounting_violation_count")
    )
    gates.append(
        _gate(
            "B5",
            "All 288 decisions bind planned, authorized, guarded, and applied support.",
            audit.get("events") == BRIDGE_DECISIONS
            and audit.get("applications") == BRIDGE_DECISIONS
            and audit.get("requested_actions") == BRIDGE_DECISIONS
            and audit.get("comparable_decisions") == BRIDGE_DECISIONS
            and accounting_violations == 0,
            dict(audit),
        )
    )
    latency = _mapping("probe decision latency", evidence.get("decision_latency"))
    p99_ns = _integer("probe decision p99", latency.get("p99_ns"))
    maximum_ns = _integer("probe maximum decision latency", latency.get("maximum_ns"))
    gates.append(
        _gate(
            "B6",
            "Decision p99 at most 2 s and maximum at most 5 s.",
            p99_ns <= 2_000_000_000 and maximum_ns <= 5_000_000_000,
            {"p99_ns": p99_ns, "maximum_ns": maximum_ns},
        )
    )
    return tuple(gates)


def _assert_current_source_matches(
    result: CausalControllerEvaluationV2,
    *,
    root: Path,
    episode: EpisodeConfigV2,
    plant: PlantConfig,
    constraints: Constraints,
    weights: RewardWeightsV2,
) -> None:
    current = collect_provenance(
        episode=episode,
        plant=plant,
        constraints=constraints,
        weights=weights,
        root=root,
    )
    current_source = _mapping("current source", current.get("source"))
    recorded_source = _mapping("recorded source", result.provenance.get("source"))
    recorded_audit = _mapping(
        "recorded source audit", result.provenance.get("execution_source_audit")
    )
    if current_source.get("identity_complete") is not True:
        raise RuntimeError("current source identity is incomplete")
    if current_source.get("git_dirty") is not False:
        raise RuntimeError("current source is dirty")
    if current_source.get("git_revision") != recorded_source.get("git_revision"):
        raise RuntimeError("source revision changed after bridge evaluation")
    if current_source.get("worktree_identity_sha256") != recorded_audit.get(
        "end_worktree_identity_sha256"
    ):
        raise RuntimeError("source content changed after bridge evaluation")
    if current.get("dependency_lock") != result.provenance.get("dependency_lock"):
        raise RuntimeError("dependency lock changed after bridge evaluation")


def _atomic_write(payload: Mapping[str, Any], target: Path) -> str:
    raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        reopened = temporary.read_bytes()
        if reopened != raw or canonical_json(json.loads(reopened)) != canonical_json(
            payload
        ):
            raise ValueError("bridge probe changed during serialization")
        os.link(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        published = target.read_bytes()
        if published != raw:
            raise ValueError("published bridge probe bytes changed")
        return hashlib.sha256(published).hexdigest()
    finally:
        temporary.unlink(missing_ok=True)


def run_bridge_probe(
    *,
    destination: Path | str,
    repository_root: Path | None = None,
    evaluator: Evaluator = evaluate_controllers,
) -> PublishedBridgeProbeV1:
    """Run and publish the one frozen actuation-bridge development cell."""

    started_ns = time.perf_counter_ns()
    root = find_repository_root(repository_root)
    _validate_protocol(root)
    target = validate_artifact_destination(destination, repository_root=root)
    plant = PlantConfig()
    episode = EpisodeConfigV2(
        horizon_h=BRIDGE_HORIZON_H,
        delivery_rate_kg_h=BRIDGE_DELIVERY_RATE_KG_H,
    ).validate(plant)
    constraints = Constraints()
    weights = RewardWeightsV2()
    started_at_utc = _utc_now()
    result = evaluator(
        seeds=[BRIDGE_SEED],
        controllers=(ServiceMPCControllerV1(),),
        episode=episode,
        plant=plant,
        constraints=constraints,
        weights=weights,
        require_clean_source=True,
    )
    completed_at_utc = _utc_now()
    result.study = "service_mpc_actuation_bridge_mechanism_probe"
    result.evidence_scope = "contaminated_development_mechanism_probe"
    result.disclaimer = BRIDGE_CLAIM_BOUNDARY
    result.title = "Service-MPC commitment-aware actuation bridge mechanism probe"
    result.information_fairness = (
        "Same public controller information and exact exogenous episode as the frozen "
        "3796eb5 cell; only first-action support authority changes."
    )
    result.provenance["service_mpc_actuation_bridge_probe"] = {
        "format": BRIDGE_PROBE_FORMAT,
        "contract_sha256": BRIDGE_PROBE_CONTRACT_SHA256,
        "started_at_utc": started_at_utc,
        "evaluation_completed_at_utc": completed_at_utc,
    }
    run, evidence = _validate_result(result, root=root)
    audit = _bridge_audit(run, evidence)
    gates = _build_gates(run, evidence, audit)
    _assert_current_source_matches(
        result,
        root=root,
        episode=episode,
        plant=plant,
        constraints=constraints,
        weights=weights,
    )
    status = (
        "execution_failed"
        if gates[0]["status"] != "pass"
        else (
            "complete_all_gates_passed"
            if all(gate["status"] == "pass" for gate in gates)
            else "complete_gate_failed"
        )
    )
    body: dict[str, Any] = {
        "format": BRIDGE_PROBE_FORMAT,
        "schema_version": 1,
        "status": status,
        "contract": BRIDGE_PROBE_CONTRACT,
        "contract_sha256": BRIDGE_PROBE_CONTRACT_SHA256,
        "execution": {
            "started_at_utc": started_at_utc,
            "evaluation_completed_at_utc": completed_at_utc,
        },
        "bridge_audit": audit,
        "gates": list(gates),
        "evaluation": result.to_dict(),
    }
    report_sha256 = domain_sha256(BRIDGE_PROBE_DOMAIN, body)
    payload = {**body, "report_sha256": report_sha256}
    canonical_json(payload)
    raw_file_sha256 = _atomic_write(payload, target)
    _assert_current_source_matches(
        result,
        root=root,
        episode=episode,
        plant=plant,
        constraints=constraints,
        weights=weights,
    )
    return PublishedBridgeProbeV1(
        path=target,
        raw_file_sha256=raw_file_sha256,
        report_sha256=report_sha256,
        status=status,
        gates=gates,
        elapsed_ns=time.perf_counter_ns() - started_ns,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m plant_backend.rl.v2.mpc_bridge_probe",
        description="Run the single frozen service-MPC actuation-bridge mechanism probe.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        published = run_bridge_probe(destination=arguments.output)
    except (TypeError, ValueError) as exc:
        message = " ".join(str(exc).split()) or type(exc).__name__
        print(f"invalid actuation-bridge probe: {message[:500]}", file=sys.stderr)
        return 2
    except Exception as exc:
        message = " ".join(str(exc).split()) or type(exc).__name__
        print(f"actuation-bridge probe failed: {message[:500]}", file=sys.stderr)
        return 1

    print(f"status: {published.status}")
    for gate in published.gates:
        print(f"{gate['id']}: {gate['status']}")
    print(f"written to {published.path}")
    print(f"artifact sha256: {published.raw_file_sha256}")
    print(f"report sha256: {published.report_sha256}")
    print(f"elapsed seconds: {published.elapsed_ns / 1_000_000_000:.6f}")
    return 0 if published.status == "complete_all_gates_passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BRIDGE_CLAIM_BOUNDARY",
    "BRIDGE_PROBE_CONTRACT",
    "BRIDGE_PROBE_CONTRACT_SHA256",
    "BRIDGE_PROBE_FORMAT",
    "PublishedBridgeProbeV1",
    "run_bridge_probe",
]

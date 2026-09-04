"""Outcome-free tests for the frozen service-MPC actuation-bridge probe."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any

import pytest
from plant_sim import Action

from plant_backend.rl.v2.evaluation import BaselineRun, RequestedActionV2
from plant_backend.rl.v2.metrics import EpisodeMetricsV2
from plant_backend.rl.v2.mpc_bridge_probe import (
    BRIDGE_DECISIONS,
    BRIDGE_PRIOR_DELIVERED_KG,
    BRIDGE_PROBE_CONTRACT,
    BRIDGE_PROBE_CONTRACT_SHA256,
    BRIDGE_PROTOCOL_PATH,
    BRIDGE_PROTOCOL_SHA256,
    PublishedBridgeProbeV1,
    _bridge_audit,
    _build_gates,
)
from plant_backend.rl.v2.mpc_execution import (
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
from plant_backend.rl.v2.mpc_schema import (
    SERVICE_MPC_CONTROLLER_ID,
    domain_sha256,
)


_MANIFEST_SHA = "a" * 64
_EXOGENOUS_SHA = "b" * 64


def _run(
    *,
    metrics: EpisodeMetricsV2,
    requests: tuple[RequestedActionV2, ...],
    evidence: dict[str, Any],
) -> BaselineRun:
    return BaselineRun(
        controller=SERVICE_MPC_CONTROLLER_ID,
        seed=0,
        manifest_sha256=_MANIFEST_SHA,
        exogenous_episode_sha256=_EXOGENOUS_SHA,
        metrics=metrics,
        requested_actions=requests,
        controller_evidence=evidence,
    )


def _audit_fixture() -> tuple[BaselineRun, dict[str, Any]]:
    action = Action(
        battery_share=0.0,
        electrolyser_share=1.0,
        dac_share=0.0,
        methanator_share=0.0,
        battery_support_kw=600.0,
    )
    action_body = {
        "schema": SERVICE_MPC_ACTION_SCHEMA,
        "plan_sha256": "c" * 64,
        "forecast_solar_kw": 500.0,
        "action": action.to_dict(),
        "simplex_sum": 1.0,
        "realization": "synthetic outcome-free bridge fixture",
        "actuation_bridge": {
            "planned_battery_support_kw": 100.0,
            "battery_support_authority_kw": 600.0,
            "battery_support_limit_kw": 2_500.0,
            "electrolyser_target_kw": 600.0,
            "authority_rule": (
                "zero when electrolyser target is zero; otherwise "
                "min(electrolyser target, active support limit)"
            ),
        },
    }
    action_record = {
        **action_body,
        "sha256": domain_sha256(SERVICE_MPC_ACTION_DOMAIN, action_body),
    }
    residual_body = {"decision_index": 0}
    residual = {
        **residual_body,
        "sha256": domain_sha256(SERVICE_MPC_RESIDUAL_DOMAIN, residual_body),
    }
    stages = [
        {
            "stage": "primary_delivery",
            "success": True,
            "solution_sha256": "d" * 64,
        },
        {
            "stage": "secondary_tiebreak",
            "success": True,
            "solution_sha256": "e" * 64,
        },
    ]
    event_body = {
        "schema": "service-mpc-decision-evidence-v1",
        "action": action_record,
        "first_step": {
            "battery_support_kw": 100.0,
            "electrolyser_startup_kw": 600.0,
            "electrolyser_productive_kw": 0.0,
        },
        "stages": stages,
        "decision_elapsed_ns": 1,
        "residual": residual,
    }
    event_identity = deepcopy(event_body)
    event_identity.pop("decision_elapsed_ns")
    identity_stages = event_identity.pop("stages")
    event_identity["stage_solution_sha256"] = [
        stage["solution_sha256"] for stage in identity_stages
    ]
    event = {
        **event_body,
        "sha256": domain_sha256(SERVICE_MPC_DECISION_DOMAIN, event_identity),
    }
    request = {"action": action.to_dict()}
    guarded_action = action.to_dict()
    interval_power = {
        "hours": 0.25,
        "guarded_kwh": {"battery_support": 150.0},
        "applied_kwh": {"battery_support": 40.0},
    }
    application_body = {
        "schema": "service-mpc-decision-application-v1",
        "controller_id": SERVICE_MPC_CONTROLLER_ID,
        "decision_index": 0,
        "decision_evidence_sha256": event["sha256"],
        "action_sha256": action_record["sha256"],
        "residual_sha256": residual["sha256"],
        "request": request,
        "request_sha256": domain_sha256(SERVICE_MPC_REQUEST_DOMAIN, request),
        "guarded_action": guarded_action,
        "guarded_action_sha256": domain_sha256(
            SERVICE_MPC_GUARDED_ACTION_DOMAIN, guarded_action
        ),
        "interval_power": interval_power,
        "interval_power_sha256": domain_sha256(
            SERVICE_MPC_INTERVAL_POWER_DOMAIN, interval_power
        ),
    }
    application = {
        **application_body,
        "sha256": domain_sha256(SERVICE_MPC_APPLICATION_DOMAIN, application_body),
    }
    evidence_body = {
        "format": SERVICE_MPC_RUN_EVIDENCE_FORMAT,
        "events": [event],
        "applications": [application],
    }
    evidence = {
        **evidence_body,
        "sha256": domain_sha256(SERVICE_MPC_RUN_EVIDENCE_DOMAIN, evidence_body),
    }
    metrics = EpisodeMetricsV2(decisions=1)
    requests = (RequestedActionV2.from_action(0, action),)
    return _run(metrics=metrics, requests=requests, evidence=evidence), evidence


def _gate_fixture() -> tuple[BaselineRun, dict[str, Any], dict[str, Any]]:
    stage_pair = [
        {"stage": "primary_delivery", "success": True},
        {"stage": "secondary_tiebreak", "success": True},
    ]
    events = [
        {
            "schema": "service-mpc-decision-evidence-v1",
            "stages": deepcopy(stage_pair),
        }
        for _ in range(BRIDGE_DECISIONS)
    ]
    evidence = {
        "format": SERVICE_MPC_RUN_EVIDENCE_FORMAT,
        "status": "complete",
        "completed_decisions": BRIDGE_DECISIONS,
        "successful_decision_count": BRIDGE_DECISIONS,
        "failed_decision_count": 0,
        "application_count": BRIDGE_DECISIONS,
        "all_successful_decisions_reconciled": True,
        "pending_transition": False,
        "fallback_action_count": 0,
        "decision_latency": {
            "p99_ns": 1_000_000_000,
            "maximum_ns": 2_000_000_000,
        },
        "events": events,
        "applications": [{} for _ in range(BRIDGE_DECISIONS)],
    }
    metrics = EpisodeMetricsV2(
        decisions=BRIDGE_DECISIONS,
        reward_components={"reserve_breach": 0.0},
        delivered_kg=BRIDGE_PRIOR_DELIVERED_KG + 1.0,
        min_battery_soc=0.15,
        terminal_battery_soc=0.25,
        electrolyser_operating_time_s=900.0,
    )
    requests = tuple(
        RequestedActionV2.from_action(index, Action.idle())
        for index in range(BRIDGE_DECISIONS)
    )
    audit = {
        "events": BRIDGE_DECISIONS,
        "applications": BRIDGE_DECISIONS,
        "requested_actions": BRIDGE_DECISIONS,
        "comparable_decisions": BRIDGE_DECISIONS,
        "accounting_violation_count": 0,
    }
    return _run(metrics=metrics, requests=requests, evidence=evidence), evidence, audit


def test_contract_binds_the_frozen_protocol_and_versioned_action_schema() -> None:
    protocol = Path(BRIDGE_PROTOCOL_PATH)

    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == BRIDGE_PROTOCOL_SHA256
    assert BRIDGE_PROBE_CONTRACT["protocol"]["raw_file_sha256"] == (
        BRIDGE_PROTOCOL_SHA256
    )
    assert BRIDGE_PROBE_CONTRACT["controller"]["action_realization_schema"] == (
        SERVICE_MPC_ACTION_SCHEMA
    )
    assert BRIDGE_PROBE_CONTRACT_SHA256 == domain_sha256(
        "service-mpc-actuation-bridge-mechanism-probe-contract-v1",
        BRIDGE_PROBE_CONTRACT,
    )


def test_bridge_audit_closes_expected_authority_and_applied_energy() -> None:
    run, evidence = _audit_fixture()

    audit = _bridge_audit(run, evidence)

    assert audit["accounting_violation_count"] == 0
    assert audit["planned_support_kwh"] == pytest.approx(25.0)
    assert audit["authority_capacity_kwh"] == pytest.approx(150.0)
    assert audit["guarded_support_kwh"] == pytest.approx(150.0)
    assert audit["applied_support_kwh"] == pytest.approx(40.0)
    assert audit["authority_above_expected_decisions"] == 1


@pytest.mark.parametrize(
    ("mutation", "violation"),
    (
        ("legacy_schema", "action schema or digest"),
        ("wrong_authority", "authority rule"),
        ("unfunded_plan", "planned support exceeds authority"),
        ("over_application", "applied support exceeds guarded support"),
    ),
)
def test_bridge_audit_fails_closed_on_semantic_or_accounting_drift(
    mutation: str,
    violation: str,
) -> None:
    run, original = _audit_fixture()
    evidence = deepcopy(original)
    event = evidence["events"][0]
    bridge = event["action"]["actuation_bridge"]
    if mutation == "legacy_schema":
        event["action"]["schema"] = "service-mpc-requested-action-v1"
    elif mutation == "wrong_authority":
        bridge["battery_support_authority_kw"] = 100.0
    elif mutation == "unfunded_plan":
        bridge["planned_battery_support_kw"] = 700.0
        event["first_step"]["battery_support_kw"] = 700.0
    elif mutation == "over_application":
        evidence["applications"][0]["interval_power"]["applied_kwh"][
            "battery_support"
        ] = 151.0
    else:  # pragma: no cover - closed parameter table
        raise AssertionError(mutation)

    audit = _bridge_audit(run, evidence)

    assert audit["accounting_violation_count"] >= 1
    assert any(violation in item for item in audit["accounting_violations"])


def test_all_probe_gates_pass_only_for_the_predeclared_complete_case() -> None:
    run, evidence, audit = _gate_fixture()

    gates = _build_gates(run, evidence, audit)

    assert [gate["id"] for gate in gates] == [f"B{index}" for index in range(1, 7)]
    assert [gate["status"] for gate in gates] == ["pass"] * 6


def test_failed_execution_suppresses_all_downstream_gate_decisions() -> None:
    run, original, audit = _gate_fixture()
    evidence = deepcopy(original)
    evidence["events"][0]["stages"][1]["success"] = False

    gates = _build_gates(run, evidence, audit)

    assert gates[0]["status"] == "fail"
    assert [gate["status"] for gate in gates[1:]] == ["not_evaluated"] * 5


def test_mechanism_and_accounting_gates_fail_independently() -> None:
    run, evidence, original_audit = _gate_fixture()
    assert run.metrics is not None
    run.metrics.electrolyser_forced_stops = 5
    audit = {**original_audit, "applications": BRIDGE_DECISIONS - 1}

    gates = _build_gates(run, evidence, audit)

    assert gates[0]["status"] == "pass"
    assert gates[1]["status"] == "fail"
    assert gates[4]["status"] == "fail"
    assert gates[2]["status"] == gates[3]["status"] == gates[5]["status"] == "pass"


def test_published_probe_record_rejects_invalid_identity_or_gate_set(
    tmp_path: Path,
) -> None:
    gates = tuple(
        {"id": f"B{index}", "status": "pass", "criterion": "fixture", "observed": {}}
        for index in range(1, 7)
    )
    record = PublishedBridgeProbeV1(
        path=(tmp_path / "probe.json").resolve(),
        raw_file_sha256="d" * 64,
        report_sha256="e" * 64,
        status="complete_all_gates_passed",
        gates=gates,
        elapsed_ns=1,
    )

    assert record.path.is_absolute()
    with pytest.raises(ValueError, match="gates"):
        PublishedBridgeProbeV1(
            path=record.path,
            raw_file_sha256=record.raw_file_sha256,
            report_sha256=record.report_sha256,
            status=record.status,
            gates=gates[:-1],
            elapsed_ns=record.elapsed_ns,
        )

"""Deep validation and aggregation for absolute-setpoint MPC run evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

from .mpc_execution import MPCFirstIntervalResidualV1
from .mpc_schema import (
    MODEL_BALANCE_TOLERANCE,
    SERVICE_MPC_SETPOINT_CONTROLLER_ID,
    ServiceMPCSetpointSettingsV1,
    canonical_json,
    domain_sha256,
    require_sha256,
)
from .mpc_setpoint_decision import (
    MPCSetpointDecisionEventV1,
    MPCSetpointDecisionEvidenceV1,
    MPCSetpointDecisionFailureV1,
)
from .mpc_setpoint_execution import MPCSetpointIntervalExecutionV1

SERVICE_MPC_SETPOINT_RUN_EVIDENCE_FORMAT = (
    "service-mpc-setpoint-controller-run-evidence-v1"
)
SERVICE_MPC_SETPOINT_RUN_EVIDENCE_DOMAIN = SERVICE_MPC_SETPOINT_RUN_EVIDENCE_FORMAT
SERVICE_MPC_SETPOINT_EVIDENCE_GRAPH_DOMAIN = "service-mpc-setpoint-evidence-graph-v1"
SETPOINT_TRACKING_TOLERANCE_KW = 1e-7

_ENERGY_GROUPS = ("planned", "target", "requested", "guarded", "applied")


def _index(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
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


def _mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return value


def _latency_group(name: str, values: object) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be an immutable tuple")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} values must be non-negative integers")
        result.append(int(value))
    return tuple(result)


def _latency_summary(values: tuple[int, ...]) -> dict[str, int | float | str]:
    if not values:
        return {
            "count": 0,
            "mean_ns": 0.0,
            "minimum_ns": 0,
            "p99_ns": 0,
            "maximum_ns": 0,
            "p99_method": "empirical nearest rank",
        }
    ordered = tuple(sorted(values))
    rank = max(1, math.ceil(0.99 * len(ordered)))
    return {
        "count": len(ordered),
        "mean_ns": math.fsum(ordered) / len(ordered),
        "minimum_ns": ordered[0],
        "p99_ns": ordered[rank - 1],
        "maximum_ns": ordered[-1],
        "p99_method": "empirical nearest rank",
    }


def _aggregate_maps(
    name: str,
    values: tuple[Mapping[str, object], ...],
) -> dict[str, float]:
    keys = tuple(sorted({key for value in values for key in value}))
    return {
        key: float(
            math.fsum(_finite(f"{name}.{key}", value.get(key, 0.0)) for value in values)
        )
        for key in keys
    }


def _close(left: float, right: float) -> bool:
    scale = max(1.0, abs(left), abs(right))
    tolerance = MODEL_BALANCE_TOLERANCE + MODEL_BALANCE_TOLERANCE * scale
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _validate_execution_residual(
    execution: MPCSetpointIntervalExecutionV1,
    residual: MPCFirstIntervalResidualV1,
) -> None:
    if residual.decision_index != execution.setpoints.decision_index:
        raise ValueError("setpoint execution and residual decision indices differ")
    if not _close(residual.hours, execution.hours):
        raise ValueError("setpoint execution and public residual durations differ")

    body = execution.body()
    energy = _mapping("interval energy", body.get("energy_kwh"))
    applied = _mapping("interval applied energy", energy.get("applied"))
    hours = execution.hours
    observed = {
        "solar": _finite("applied solar", applied.get("solar_available")) / hours,
        "battery": (
            _finite("applied battery charge", applied.get("solar_to_battery"))
            - _finite("applied battery support", applied.get("battery_support"))
        )
        / hours,
        "electrolyser": _finite(
            "applied electrolyser", applied.get("electrolyser_total")
        )
        / hours,
        "dac": _finite("applied DAC", applied.get("dac")) / hours,
        "methanator": _finite("applied methanator", applied.get("methanator")) / hours,
        "curtailed": _finite(
            "applied solar curtailment", applied.get("solar_curtailed")
        )
        / hours,
    }
    residual_observed = {
        "solar": residual.observed_solar_mean_kw,
        "battery": residual.observed_battery_net_kw,
        "electrolyser": residual.observed_electrolyser_kw,
        "dac": residual.observed_dac_kw,
        "methanator": residual.observed_methanator_kw,
        "curtailed": residual.observed_curtailed_kw,
    }
    for channel, value in observed.items():
        if not _close(value, residual_observed[channel]):
            raise ValueError(
                f"setpoint execution and public residual {channel} powers differ"
            )

    first = execution.applications[0].boundary_record
    terminal = execution.applications[-1].next_boundary_record
    terminal_fields = {
        "battery": ("battery_energy_kwh", residual.observed_battery_end_kwh),
        "H2": ("h2_level_kg", residual.observed_h2_end_kg),
        "CO2": ("co2_level_kg", residual.observed_co2_end_kg),
        "CH4": ("ch4_level_kg", residual.observed_ch4_end_kg),
    }
    for label, (field, expected) in terminal_fields.items():
        observed_value = _finite(f"terminal {field}", terminal.get(field))
        if not _close(observed_value, expected):
            raise ValueError(
                f"setpoint terminal boundary and residual {label} state differ"
            )

    cumulative_fields = {
        "H2": ("h2_produced_total_kg", residual.observed_h2_kg),
        "CO2": ("co2_captured_total_kg", residual.observed_co2_kg),
        "CH4": ("ch4_produced_total_kg", residual.observed_ch4_kg),
    }
    for label, (field, expected) in cumulative_fields.items():
        observed_delta = _finite(f"terminal {field}", terminal.get(field)) - _finite(
            f"initial {field}", first.get(field)
        )
        if not _close(observed_delta, expected):
            raise ValueError(
                f"setpoint boundary ledger and residual {label} production differ"
            )


def _raw_identity_lists(
    events: tuple[MPCSetpointDecisionEventV1, ...],
) -> tuple[list[str], list[str], list[str]]:
    event_sha = [event.sha256 for event in events]
    executions = [
        event.execution
        for event in events
        if isinstance(event, MPCSetpointDecisionEvidenceV1)
        and event.execution is not None
    ]
    execution_sha = [execution.sha256 for execution in executions]
    application_sha = [
        application.sha256
        for execution in executions
        for application in execution.applications
    ]
    return event_sha, execution_sha, application_sha


def _graph_body(payload: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "format",
        "controller_id",
        "settings_sha256",
        "status",
        "expected_decisions",
        "completed_decisions",
        "runner_failure_phase",
        "pending_transition",
        "final_observation_sha256",
        "counts",
        "energy_kwh",
        "target_tracking",
        "target_deficit_attribution_kwh",
        "event_sha256",
        "execution_sha256",
        "application_sha256",
    )
    return {key: payload[key] for key in keys}


def build_mpc_setpoint_run_evidence(
    events: tuple[MPCSetpointDecisionEventV1, ...],
    dispatcher_elapsed_ns: tuple[tuple[int, ...], ...],
    *,
    settings: ServiceMPCSetpointSettingsV1,
    completed_decisions: int,
    run_succeeded: bool,
    pending_transition: bool,
    runner_failure_phase: str | None,
    final_observation_sha256: str,
) -> dict[str, Any]:
    """Deeply validate and summarize a complete or stopped setpoint-MPC run."""

    if not isinstance(settings, ServiceMPCSetpointSettingsV1):
        raise TypeError("settings must be ServiceMPCSetpointSettingsV1")
    if not isinstance(events, tuple) or any(
        not isinstance(
            event,
            (MPCSetpointDecisionEvidenceV1, MPCSetpointDecisionFailureV1),
        )
        for event in events
    ):
        raise TypeError("setpoint-MPC events must be an immutable typed tuple")
    if not isinstance(dispatcher_elapsed_ns, tuple):
        raise TypeError("dispatcher latencies must be an immutable tuple of tuples")
    dispatcher_groups = tuple(
        _latency_group(f"dispatcher latency group {index}", group)
        for index, group in enumerate(dispatcher_elapsed_ns)
    )
    completed = _index("completed_decisions", completed_decisions)
    if not isinstance(run_succeeded, bool) or not isinstance(pending_transition, bool):
        raise TypeError("setpoint-MPC run status flags must be bools")
    if runner_failure_phase is not None and (
        not isinstance(runner_failure_phase, str) or not runner_failure_phase
    ):
        raise ValueError("runner_failure_phase must be non-empty or None")
    if run_succeeded != (runner_failure_phase is None):
        raise ValueError("run success and runner failure phase disagree")
    final_sha = require_sha256("final_observation_sha256", final_observation_sha256)

    successes = tuple(
        event for event in events if isinstance(event, MPCSetpointDecisionEvidenceV1)
    )
    failures = tuple(
        event for event in events if isinstance(event, MPCSetpointDecisionFailureV1)
    )
    if events != (*successes, *failures):
        raise ValueError("setpoint-MPC failure event must terminate the event sequence")
    if tuple(event.decision_index for event in successes) != tuple(
        range(len(successes))
    ):
        raise ValueError("successful setpoint-MPC decisions are not contiguous")
    if len(failures) > 1:
        raise ValueError("a stopped setpoint-MPC run cannot contain multiple failures")
    if failures and failures[0].decision_index != len(successes):
        raise ValueError("setpoint-MPC failure is not after successful decisions")
    if any(event.settings_sha256 != settings.sha256 for event in events):
        raise ValueError("setpoint-MPC event settings identity is inconsistent")

    closed_flags = tuple(event.closed for event in successes)
    closed_count = sum(closed_flags)
    if (
        closed_flags
        != (True,) * closed_count + (False,) * (len(successes) - closed_count)
        or len(successes) - closed_count > 1
    ):
        raise ValueError("only the final setpoint-MPC success may remain open")
    expected_pending = bool(successes and not successes[-1].closed)
    if pending_transition != expected_pending:
        raise ValueError("pending state and final setpoint decision disagree")
    if completed != closed_count:
        raise ValueError("completed decisions must equal closed physical executions")
    if len(dispatcher_groups) != closed_count:
        raise ValueError("dispatcher latency groups do not match closed executions")

    expected_decisions = settings.task_interval_count
    for event in successes:
        expected_horizon = min(
            settings.maximum_planning_intervals,
            expected_decisions - event.decision_index,
        )
        if event.horizon_intervals != expected_horizon:
            raise ValueError("setpoint-MPC decision horizon is inconsistent")
        if not math.isclose(
            event.setpoints.hold_duration_s,
            settings.interval_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("setpoint-MPC command hold differs from its settings")

    if run_succeeded:
        if failures or pending_transition:
            raise ValueError("successful setpoint-MPC run contains unfinished evidence")
        if not (
            len(successes) == completed == expected_decisions
            and all(event.closed for event in successes)
        ):
            raise ValueError("successful setpoint-MPC run evidence is incomplete")
    elif completed > expected_decisions or len(successes) > expected_decisions:
        raise ValueError("failed setpoint-MPC run exceeds the declared task")

    for index, event in enumerate(successes):
        if index > 0:
            prior = successes[index - 1]
            assert prior.residual is not None
            if prior.residual.next_observation_sha256 != event.observation_sha256:
                raise ValueError("setpoint-MPC public observation chain is broken")
        if not event.closed:
            continue
        assert event.execution is not None
        assert event.residual is not None
        execution = event.execution
        timings = dispatcher_groups[index]
        if len(timings) != len(execution.applications):
            raise ValueError(
                "dispatcher latency count differs from physical minute evidence"
            )
        _validate_execution_residual(execution, event.residual)
        if index > 0:
            prior_execution = successes[index - 1].execution
            assert prior_execution is not None
            prior_application = prior_execution.applications[-1]
            application = execution.applications[0]
            if (
                prior_application.next_boundary_json != application.boundary_json
                or prior_application.next_boundary_sha256 != application.boundary_sha256
                or prior_application.commitment_after != application.commitment_before
                or prior_application.commitment_after_sha256
                != application.commitment_before_sha256
            ):
                raise ValueError("setpoint-MPC physical execution chain is broken")

    if successes:
        final_event = successes[-1]
        expected_final_sha = (
            final_event.residual.next_observation_sha256
            if final_event.residual is not None
            else final_event.observation_sha256
        )
        if final_sha != expected_final_sha:
            raise ValueError("final public observation does not close the event chain")
    elif failures and failures[0].observation_sha256 is not None:
        if failures[0].observation_sha256 != final_sha:
            raise ValueError("failure and final public observation identities differ")
    if failures and failures[0].observation_sha256 is not None:
        if failures[0].observation_sha256 != final_sha:
            raise ValueError("failure observation is not the final public boundary")

    closed_successes = successes[:closed_count]
    executions = tuple(
        event.execution for event in closed_successes if event.execution is not None
    )
    energy_sources: dict[str, list[Mapping[str, object]]] = {
        group: [] for group in _ENERGY_GROUPS
    }
    error_sources: list[Mapping[str, object]] = []
    applications = tuple(
        application
        for execution in executions
        for application in execution.applications
    )
    for execution in executions:
        energy = _mapping("interval energy", execution.body().get("energy_kwh"))
        for group in _ENERGY_GROUPS:
            energy_sources[group].append(
                _mapping(f"interval {group} energy", energy.get(group))
            )
        error_sources.append(
            _mapping(
                "interval target-error energy",
                energy.get("electrolyser_target_error"),
            )
        )
    aggregate_energy = {
        group: _aggregate_maps(group, tuple(energy_sources[group]))
        for group in _ENERGY_GROUPS
    }
    aggregate_energy["electrolyser_target_error"] = _aggregate_maps(
        "electrolyser_target_error", tuple(error_sources)
    )

    requested_errors: list[float] = []
    guarded_errors: list[float] = []
    applied_errors: list[float] = []
    resource_gaps: list[float] = []
    guard_gaps: list[float] = []
    plant_gaps: list[float] = []
    attribution_errors: list[float] = []
    requested_overshoots = 0
    reachable_violations = 0
    simultaneous_charge_support = 0
    resource_shortfalls = 0
    guard_reductions = 0
    plant_refusals = 0
    target_deficits = 0
    unattributed_deficits = 0
    support_authority_violations = 0
    for application in applications:
        dispatch = application.dispatch
        target = dispatch.electrolyser_target_kw
        requested = dispatch.electrolyser_offered_kw
        guarded = application.guarded_electrolyser_kw
        applied = application.applied_electrolyser_kw
        requested_error = requested - target
        guarded_error = guarded - target
        applied_error = applied - target
        requested_errors.append(requested_error)
        guarded_errors.append(guarded_error)
        applied_errors.append(applied_error)
        requested_overshoots += int(requested_error > SETPOINT_TRACKING_TOLERANCE_KW)
        reachable_violations += int(
            dispatch.electrolyser_target_reachable
            and abs(requested_error) > SETPOINT_TRACKING_TOLERANCE_KW
        )
        simultaneous_charge_support += int(
            dispatch.solar_to_battery_kw > 0.0 and dispatch.battery_support_kw > 0.0
        )

        resource_gap = max(0.0, target - requested)
        guard_gap = max(0.0, requested - guarded)
        plant_gap = max(0.0, guarded - applied)
        total_deficit = max(0.0, target - applied)
        attribution_error = total_deficit - resource_gap - guard_gap - plant_gap
        resource_gaps.append(resource_gap)
        guard_gaps.append(guard_gap)
        plant_gaps.append(plant_gap)
        attribution_errors.append(attribution_error)
        resource_shortfalls += int(resource_gap > SETPOINT_TRACKING_TOLERANCE_KW)
        guard_reductions += int(guard_gap > SETPOINT_TRACKING_TOLERANCE_KW)
        plant_refusals += int(plant_gap > SETPOINT_TRACKING_TOLERANCE_KW)
        target_deficits += int(total_deficit > SETPOINT_TRACKING_TOLERANCE_KW)
        unattributed_deficits += int(
            total_deficit > SETPOINT_TRACKING_TOLERANCE_KW
            and abs(attribution_error) > SETPOINT_TRACKING_TOLERANCE_KW
        )
        guarded_solar_kw = (
            application.guarded_action.electrolyser_share
            * application.applied_solar_available_kw
        )
        guarded_support_shortfall = max(0.0, target - guarded_solar_kw)
        support_authority_violations += int(
            application.applied_battery_support_kw
            > guarded_support_shortfall + SETPOINT_TRACKING_TOLERANCE_KW
        )

    dt_h_values = tuple(execution.physics_step_s / 3_600.0 for execution in executions)
    if dt_h_values and any(
        not math.isclose(value, dt_h_values[0], rel_tol=0.0, abs_tol=1e-12)
        for value in dt_h_values[1:]
    ):
        raise ValueError("setpoint-MPC physics timestep changed within the run")
    minute_h = 0.0 if not dt_h_values else dt_h_values[0]
    attribution = {
        "resource_shortfall": float(math.fsum(resource_gaps) * minute_h),
        "guard_reduction": float(math.fsum(guard_gaps) * minute_h),
        "plant_refusal_or_reroute": float(math.fsum(plant_gaps) * minute_h),
        "closure_error": float(math.fsum(attribution_errors) * minute_h),
        "maximum_absolute_closure_error_kw": float(
            max((abs(value) for value in attribution_errors), default=0.0)
        ),
    }
    if unattributed_deficits:
        raise ValueError("setpoint target deficit attribution does not close")

    successful_stage_count = sum(
        stage.success for event in successes for stage in event.stages
    ) + sum(stage.success for failure in failures for stage in failure.stages)
    attempted_stage_count = sum(len(event.stages) for event in events)
    counts = {
        "event_count": len(events),
        "successful_decision_count": len(successes),
        "failed_decision_count": len(failures),
        "closed_decision_count": closed_count,
        "interval_execution_count": len(executions),
        "physical_minute_count": len(applications),
        "attempted_solver_stage_count": attempted_stage_count,
        "successful_solver_stage_count": successful_stage_count,
        "fallback_action_count": 0,
        "requested_target_overshoot_minutes": requested_overshoots,
        "reachable_tracking_violation_minutes": reachable_violations,
        "simultaneous_charge_support_request_minutes": simultaneous_charge_support,
        "resource_shortfall_minutes": resource_shortfalls,
        "guard_target_reduction_minutes": guard_reductions,
        "plant_target_refusal_minutes": plant_refusals,
        "target_deficit_minutes": target_deficits,
        "unattributed_target_deficit_minutes": unattributed_deficits,
        "guard_changed_request_minutes": sum(
            application.guard_changed_request for application in applications
        ),
        "plant_refused_or_rerouted_minutes": sum(
            application.plant_refused_or_rerouted_power for application in applications
        ),
        "applied_support_authority_violation_minutes": (support_authority_violations),
        "accounting_violation_count": 0,
        "chain_violation_count": 0,
    }
    target_tracking = {
        "tolerance_kw": SETPOINT_TRACKING_TOLERANCE_KW,
        "requested_max_overshoot_kw": float(
            max((max(0.0, value) for value in requested_errors), default=0.0)
        ),
        "requested_max_undershoot_kw": float(
            max((max(0.0, -value) for value in requested_errors), default=0.0)
        ),
        "guarded_max_overshoot_kw": float(
            max((max(0.0, value) for value in guarded_errors), default=0.0)
        ),
        "guarded_max_undershoot_kw": float(
            max((max(0.0, -value) for value in guarded_errors), default=0.0)
        ),
        "applied_max_overshoot_kw": float(
            max((max(0.0, value) for value in applied_errors), default=0.0)
        ),
        "applied_max_undershoot_kw": float(
            max((max(0.0, -value) for value in applied_errors), default=0.0)
        ),
    }

    decision_latencies = tuple(event.decision_elapsed_ns for event in successes)
    stage_latencies = tuple(
        stage.elapsed_ns for event in events for stage in event.stages
    )
    dispatcher_latencies = tuple(
        value for group in dispatcher_groups for value in group
    )
    event_sha, execution_sha, application_sha = _raw_identity_lists(events)
    payload: dict[str, Any] = {
        "format": SERVICE_MPC_SETPOINT_RUN_EVIDENCE_FORMAT,
        "controller_id": SERVICE_MPC_SETPOINT_CONTROLLER_ID,
        "settings_sha256": settings.sha256,
        "status": "complete" if run_succeeded else "failed",
        "expected_decisions": expected_decisions,
        "completed_decisions": completed,
        "runner_failure_phase": runner_failure_phase,
        "pending_transition": pending_transition,
        "final_observation_sha256": final_sha,
        "counts": counts,
        "energy_kwh": aggregate_energy,
        "target_tracking": target_tracking,
        "target_deficit_attribution_kwh": attribution,
        "decision_latency": _latency_summary(decision_latencies),
        "solver_stage_latency": _latency_summary(stage_latencies),
        "dispatcher_latency": _latency_summary(dispatcher_latencies),
        "event_sha256": event_sha,
        "execution_sha256": execution_sha,
        "application_sha256": application_sha,
        "events": [event.to_dict() for event in events],
        "dispatcher_elapsed_ns": [list(group) for group in dispatcher_groups],
    }
    payload["evidence_graph_sha256"] = domain_sha256(
        SERVICE_MPC_SETPOINT_EVIDENCE_GRAPH_DOMAIN,
        _graph_body(payload),
    )
    payload["sha256"] = domain_sha256(
        SERVICE_MPC_SETPOINT_RUN_EVIDENCE_DOMAIN,
        payload,
    )
    canonical_json(payload)
    return payload


def verify_mpc_setpoint_run_digests(payload: Mapping[str, object]) -> None:
    """Verify the outer and deterministic-graph digests of a detached run payload."""

    evidence = _mapping("setpoint-MPC run evidence", payload)
    supplied_sha = require_sha256("run evidence sha256", evidence.get("sha256"))
    body = dict(evidence)
    del body["sha256"]
    if domain_sha256(SERVICE_MPC_SETPOINT_RUN_EVIDENCE_DOMAIN, body) != supplied_sha:
        raise ValueError("setpoint-MPC run evidence digest is invalid")
    supplied_graph_sha = require_sha256(
        "evidence_graph_sha256", evidence.get("evidence_graph_sha256")
    )
    graph_source = dict(evidence)
    del graph_source["sha256"]
    del graph_source["evidence_graph_sha256"]
    if (
        domain_sha256(
            SERVICE_MPC_SETPOINT_EVIDENCE_GRAPH_DOMAIN,
            _graph_body(graph_source),
        )
        != supplied_graph_sha
    ):
        raise ValueError("setpoint-MPC evidence graph digest is invalid")
    canonical_json(dict(evidence))


__all__ = [
    "SERVICE_MPC_SETPOINT_EVIDENCE_GRAPH_DOMAIN",
    "SERVICE_MPC_SETPOINT_RUN_EVIDENCE_DOMAIN",
    "SERVICE_MPC_SETPOINT_RUN_EVIDENCE_FORMAT",
    "SETPOINT_TRACKING_TOLERANCE_KW",
    "build_mpc_setpoint_run_evidence",
    "verify_mpc_setpoint_run_digests",
]

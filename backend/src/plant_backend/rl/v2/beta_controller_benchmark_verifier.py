"""Detached semantic verification for cyclic-market controller benchmarks.

The benchmark runner deliberately publishes enough public evidence to verify a
result without reopening a simulator or executing a controller.  This module is
that admission boundary.  It does not trust report-owned status flags, hashes,
or aggregates: typed public boundaries and generic command/execution records are
rebuilt, M24 is handed to its existing strict typed codec, cycle and economic
ledgers are recomputed, and deterministic arm/cell/report identities are derived
again with the runner's published timing-free rule.

The file loader additionally rejects duplicate object keys, non-finite JSON
numbers, symlinks, non-regular files, non-UTF-8 input, and non-canonical output.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import fields
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import stat
import sys
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from plant_sim import (
    Action,
    ComponentHealth,
    ElectrolyserCommitmentSnapshot,
    ElectrolyserMode,
    PlantState,
)
from plant_sim.state import CRITICAL, DEGRADED, NORMAL, OFFLINE

from ...safety.validation import BatterySupportAvailability
from .. import checkpoints
from . import beta_spaces
from .beta_config import (
    CycleReferenceV1,
    CycleSettlementV1,
    RESOURCE_NAMES,
    canonical_sha256,
)
from .beta_control import (
    BETA_CONTROLLER_OBSERVATION_SCHEMA,
    CYCLE_OBSERVATION_SCHEMA,
    CycleObservationV1,
    CyclicMarketControllerObservationV1,
)
from .beta_disturbances import generate_disturbance_episode
from .beta_env import BETA_ROOT_STREAM_NAMES, describe_beta_checkpoint_contract
from .beta_mpc_model import build_beta_mpc_model, derive_beta_mpc_cycle_return
from .beta_controller_benchmark import (
    BETA_BOUNDARY_LEDGER_FORMAT,
    BETA_CONTROLLER_ARM_ORDER,
    BETA_CONTROLLER_BENCHMARK_ARM_FORMAT,
    BETA_CONTROLLER_BENCHMARK_CELL_FORMAT,
    BETA_CONTROLLER_BENCHMARK_CLAIM_BOUNDARY,
    BETA_CONTROLLER_BENCHMARK_FORMAT,
    BETA_CONTROLLER_BENCHMARK_MAX_SEEDS,
    BETA_CONTROLLER_INFORMATION_PROFILES,
    BETA_ECONOMIC_MPC_ARM,
    BETA_EXECUTION_DOORS,
    BETA_SERVICE_COMPARISON_TOLERANCE_KG,
    HEURISTIC_ARM,
    PPO_ARM,
    SERVICE_MPC_ARM,
)
from .beta_mpc_controller import CyclicMarketEconomicMPCControllerV1
from .beta_mpc_schema import (
    BETA_MPC_CONTROLLER_ID,
    BETA_MPC_CYCLE_OBSERVATION_DOMAIN,
    BETA_MPC_CYCLE_RETURN_DOMAIN,
    BETA_MPC_DECISION_DOMAIN,
    BETA_MPC_DECISION_SCHEMA,
    BETA_MPC_PLAN_DOMAIN,
    BETA_MPC_SOLUTION_DOMAIN,
    BetaMPCPlanV1,
    BetaMPCCycleReturnV1,
    BetaMPCResourceReturnV1,
    BetaMPCSettingsV1,
    project_beta_mpc_prices,
)
from .boundary import ControlStateV2, IntervalSummaryV2
from .control import ControllerObservationV2, DeliveryObservation
from .economic_mpc_schema import EconomicMPCStageSolveV1
from .economic_mpc_model import economic_delivery_lock_row
from .economics import ECONOMIC_OBJECTIVE_VERSION, ch4_equivalent_inventory_kg
from .full_day_horizon_codec import (
    decode_mpc_setpoint_run,
    verify_mpc_setpoint_run_semantics,
)
from .market import (
    FORECAST_HORIZON_H,
    PUBLIC_PRICE_FORECAST_SCHEMA,
    PriceForecastBinV1,
    PublicPriceForecastIssueV1,
)
from .metrics import EpisodeMetricsV2
from .mpc_execution import SERVICE_MPC_OBSERVATION_DOMAIN
from .mpc_forecast import project_issued_mean_solar
from .mpc_schema import (
    MODEL_BALANCE_TOLERANCE,
    MPCDeliveryTargetV1,
    MPCPlanStepV1,
    ServiceMPCSetpointSettingsV1,
    canonical_json,
    domain_sha256,
    require_sha256,
)
from .mpc_setpoint_controller import ServiceMPCSetpointControllerV1
from .mpc_setpoint_decision import MPCSetpointDecisionEvidenceV1
from .observation import ObservationEncoderV2
from .policy_setpoint_controller import (
    NORMALIZED_POLICY_SETPOINT_CONTROLLER_ID,
    NORMALIZED_POLICY_SETPOINT_EVIDENCE_DOMAIN,
    NORMALIZED_POLICY_SETPOINT_EVIDENCE_SCHEMA,
    NormalizedSetpointPolicyControllerV1,
    PolicySetpointDecisionEvidenceV1,
)
from .reward import OBJECTIVE_VERSION
from .setpoint_command import (
    PROCESS_SETPOINT_BOUNDARY_DOMAIN,
    ProcessSetpointCommandV1,
    ProcessSetpointEnvelopeV1,
    ProcessSetpointMinuteDispatchV1,
    resolve_process_setpoint_envelope,
)
from .setpoint_control import (
    DecodedProcessSetpointActionV1,
    ProcessSetpointActionAdapterV1,
    ProcessSetpointAllocationV1,
    ProcessSetpointTargetsV1,
)
from .setpoint_controllers import (
    LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID,
    LEGACY_HEURISTIC_SETPOINT_EVIDENCE_DOMAIN,
    LEGACY_HEURISTIC_SETPOINT_EVIDENCE_SCHEMA,
    LegacyHeuristicSetpointControllerV1,
    LegacyHeuristicSetpointEvidenceV1,
)
from .setpoint_execution import (
    ProcessSetpointIntervalExecutionV1,
    ProcessSetpointMinuteApplicationV1,
)
from .training.config import ControllerNeutralTrainingConfigV1
from .weather import ForecastWindowV2, PublicForecastIssueV2


_REPORT_DOMAIN = "cyclic-market-controller-benchmark-evidence-v1"
_CELL_DOMAIN = "cyclic-market-controller-benchmark-cell-evidence-v1"
_ARM_DOMAIN = "cyclic-market-controller-benchmark-arm-evidence-v1"
_VECTOR_DOMAIN = "rl-policy-observation-vector-v1"
_CHECKPOINT_METADATA_DOMAIN = "rl-checkpoint-metadata-v1"
_POLICY_CONTRACT_DOMAIN = "rl-policy-environment-contract-v1"

_ECONOMIC_SIGNAL_FIELDS = frozenset(
    {
        "opportunity_price_eur_per_mwh",
        "effective_opportunity_price_eur_per_mwh",
        "accepted_solar_kwh",
        "delivered_kg",
        "unmet_kg",
        "battery_throughput_kwh",
        "battery_equivalent_full_cycles",
        "electrolyser_equivalent_operating_h",
        "methane_endowment_deficit_before_kg",
        "methane_endowment_deficit_after_kg",
        "battery_endowment_deficit_before_kwh",
        "battery_endowment_deficit_after_kwh",
        "solar_opportunity_cost_eur",
        "battery_throughput_reserve_eur",
        "electrolyser_life_reserve_eur",
    }
)
_ECONOMIC_COMPONENT_FIELDS = frozenset(
    {
        "delivered_service_value",
        "unmet_delivery_penalty",
        "solar_opportunity_cost",
        "battery_throughput_reserve",
        "electrolyser_life_reserve",
        "methane_endowment_liability_change",
        "battery_endowment_liability_change",
    }
)

_REPORT_FIELDS = frozenset(
    {
        "format",
        "claim_boundary",
        "policy_qualification",
        "benchmark_qualification",
        "training_task",
        "checkpoint",
        "source_provenance",
        "seeds",
        "arm_order",
        "information_profiles",
        "execution_doors",
        "service_mpc_settings",
        "cells",
        "aggregate",
        "deterministic_evidence_sha256",
    }
)
_NORMAL_CELL_FIELDS = frozenset(
    {
        "format",
        "seed",
        "status",
        "execution_complete",
        "cycle_feasible",
        "hard_safety_passed",
        "service_comparable",
        "service_comparison",
        "value_comparison_eligible",
        "exclusion_reasons",
        "manifest_identity",
        "audit_manifest",
        "arm_order",
        "runs",
        "failure",
        "deterministic_evidence_sha256",
    }
)
_PREFLIGHT_CELL_FIELDS = _NORMAL_CELL_FIELDS - {"service_comparison"}
_CLASSIFICATION_FIELDS = frozenset(
    {
        "status",
        "execution_complete",
        "cycle_feasible",
        "hard_safety_passed",
        "value_comparison_candidate",
        "value_comparison_eligible",
        "exclusion_reasons",
        "expected_decisions",
        "expected_cycle_count",
        "hard_safety_metrics",
        "cycle_settlements",
        "cumulative_cycle_constraint_cost",
    }
)
_COMMON_ARM_FIELDS = frozenset(
    {
        "format",
        "arm",
        "information_profile",
        "execution_door",
        "controller",
        "manifest_identity",
        "completed_decisions",
        "physical_steps",
        "terminated",
        "truncated",
        "reward_trace_eur",
        "reward_trace_sum_eur",
        "physical_metrics",
        "economic_metrics",
        "evidence",
        "failure",
        "deterministic_evidence_sha256",
    }
)
_GENERIC_ARM_FIELDS = (
    _COMMON_ARM_FIELDS
    | _CLASSIFICATION_FIELDS
    | frozenset({"controller_elapsed_ns", "dispatcher_elapsed_ns"})
)
_M24_ARM_FIELDS = (
    _COMMON_ARM_FIELDS
    | _CLASSIFICATION_FIELDS
    | frozenset({"closed_decisions", "evidence_validation_failure"})
)
_PREFLIGHT_ARM_FIELDS = frozenset(
    {
        "format",
        "arm",
        "status",
        "execution_complete",
        "cycle_feasible",
        "hard_safety_passed",
        "value_comparison_candidate",
        "value_comparison_eligible",
        "exclusion_reasons",
        "information_profile",
        "execution_door",
        "controller",
        "manifest_identity",
        "completed_decisions",
        "physical_steps",
        "terminated",
        "truncated",
        "reward_trace_eur",
        "reward_trace_sum_eur",
        "physical_metrics",
        "economic_metrics",
        "cycle_settlements",
        "evidence",
        "failure",
        "deterministic_evidence_sha256",
    }
)
_BOUNDARY_FIELDS = frozenset(
    {
        "format",
        "kind",
        "decision_index",
        "physics_step",
        "terminal",
        "typed_observation",
        "typed_observation_sha256",
        "physical_observation_sha256",
        "market_issue_sha256",
        "cycle_observation_sha256",
        "normalized_observation",
        "normalized_observation_sha256",
    }
)
_MANIFEST_IDENTITY_FIELDS = frozenset(
    {
        "manifest_sha256",
        "alpha_physical_exogenous_episode_sha256",
        "beta_exogenous_episode_sha256",
        "market_episode_sha256",
        "disturbance_episode_sha256",
        "disturbance_schedule_sha256",
        "cycle_task_sha256",
        "checkpoint_contract_sha256",
        "economic_assumptions_sha256",
        "beta_task_sha256",
    }
)
_CONTROLLER_HARD_SAFETY_COUNT_METRICS = (
    "out_of_range_decisions",
    "electrolyser_forced_stops",
)
_HARD_SAFETY_FIELDS = frozenset(
    {
        "observed_critical_or_offline_steps",
        "declared_dac_outage_steps",
        "unexpected_critical_or_offline_steps",
        *_CONTROLLER_HARD_SAFETY_COUNT_METRICS,
        "minimum_battery_soc",
        "minimum_allowed_battery_soc",
        "battery_reserve_respected",
    }
)
_BATTERY_SAFETY_TOLERANCE = 1e-9
_EMPTY_FEEDSTOCK_KG = 0.05
_PLANT_HEALTH_UNITS = frozenset(
    {
        "solar",
        "battery",
        "electrolyser",
        "h2_tank",
        "dac",
        "co2_tank",
        "methanator",
        "ch4_tank",
    }
)
_PLANT_STATUS_UNITS = frozenset(
    {"solar", "battery", "electrolyser", "dac", "methanator"}
)
_RESOURCE_STATE_FIELDS = {
    "battery": "battery_energy_kwh",
    "h2": "h2_level_kg",
    "co2": "co2_level_kg",
    "ch4": "ch4_level_kg",
}
_RESOURCE_MARGIN_FIELDS = {
    "battery": "battery_margin_kwh",
    "h2": "h2_margin_kg",
    "co2": "co2_margin_kg",
    "ch4": "ch4_margin_kg",
}
_RESOURCE_TOLERANCE_FIELDS = {
    "battery": "battery_tolerance_kwh",
    "h2": "h2_tolerance_kg",
    "co2": "co2_tolerance_kg",
    "ch4": "ch4_tolerance_kg",
}
_RESOURCE_RETURN_FIELDS = {
    "battery": "battery_return_fraction",
    "h2": "h2_return_fraction",
    "co2": "co2_return_fraction",
    "ch4": "ch4_return_fraction",
}
_RESOURCE_UNITS = {"battery": "kWh", "h2": "kg", "co2": "kg", "ch4": "kg"}


def _mapping(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return dict(value)


def _array(name: str, value: object) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    return value


def _exact_keys(name: str, value: Mapping[str, Any], expected: frozenset[str]) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"{name} fields differ; missing={missing}, unknown={unknown}")


def _exact(name: str, actual: object, expected: object) -> None:
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError(f"{name} differs from independent reconstruction")


def _boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _number(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}")
    return result


def _text(name: str, value: object, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    if not allow_empty and not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _digest(name: str, value: object) -> str:
    return require_sha256(name, value)


def _finite_tree(name: str, value: object) -> None:
    """Reject non-JSON and non-finite scalar values in in-memory reports."""

    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, Integral):
        return
    if isinstance(value, Real):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{name} contains a non-string object key")
        for key, item in value.items():
            _finite_tree(f"{name}.{key}", item)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_tree(f"{name}[{index}]", item)
        return
    raise TypeError(f"{name} contains a non-JSON value {type(value).__name__}")


def _timing_free(value: object) -> object:
    """Independent implementation of the runner's published identity projection."""

    if isinstance(value, Mapping):
        detached: dict[str, object] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if (
                key == "sha256"
                or key
                in {
                    "started_at_utc",
                    "completed_at_utc",
                    "decision_latency",
                    "solver_stage_latency",
                    "dispatcher_latency",
                    "event_sha256",
                    "evidence_graph_sha256",
                    "path",
                }
                or key.endswith("_elapsed_ns")
                or key == "elapsed_ns"
            ):
                continue
            detached[key] = _timing_free(item)
        return detached
    if isinstance(value, (list, tuple)):
        return [_timing_free(item) for item in value]
    return value


def _verify_domain_record(name: str, raw: object, domain: str) -> dict[str, Any]:
    payload = _mapping(name, raw)
    supplied = _digest(f"{name} sha256", payload.get("sha256"))
    body = dict(payload)
    del body["sha256"]
    if domain_sha256(domain, body) != supplied:
        raise ValueError(f"{name} digest differs")
    return payload


def _decode_action(raw: object, *, name: str) -> Action:
    payload = _mapping(name, raw)
    _exact_keys(
        name,
        payload,
        frozenset(
            {
                "battery_share",
                "electrolyser_share",
                "dac_share",
                "methanator_share",
                "battery_support_kw",
            }
        ),
    )
    action = Action(
        **{key: _number(f"{name}.{key}", value) for key, value in payload.items()}
    )
    if Action.sanitised(action) != action:
        raise ValueError(f"{name} is not a sanitized action")
    _exact(name, payload, action.to_dict())
    return action


def _decode_targets(raw: object, *, name: str) -> ProcessSetpointTargetsV1:
    payload = _mapping(name, raw)
    values = _mapping(f"{name}.setpoints_kw", payload.get("setpoints_kw"))
    _exact_keys(
        f"{name}.setpoints_kw",
        values,
        frozenset({"electrolyser", "dac", "methanator", "battery_charge_ceiling"}),
    )
    target = ProcessSetpointTargetsV1(
        electrolyser_target_kw=_number(f"{name}.electrolyser", values["electrolyser"]),
        dac_target_kw=_number(f"{name}.dac", values["dac"]),
        methanator_target_kw=_number(f"{name}.methanator", values["methanator"]),
        battery_charge_ceiling_kw=_number(
            f"{name}.battery_charge_ceiling", values["battery_charge_ceiling"]
        ),
    )
    _exact(name, payload, target.to_dict())
    return target


def _decode_envelope(raw: object, *, name: str) -> ProcessSetpointEnvelopeV1:
    payload = _mapping(name, raw)
    limits = _mapping(f"{name}.limits_kw", payload.get("limits_kw"))
    _exact_keys(
        f"{name}.limits_kw",
        limits,
        frozenset(
            {"electrolyser", "dac", "methanator", "battery_charge", "battery_support"}
        ),
    )
    envelope = ProcessSetpointEnvelopeV1(
        electrolyser_limit_kw=_number(f"{name}.electrolyser", limits["electrolyser"]),
        dac_limit_kw=_number(f"{name}.dac", limits["dac"]),
        methanator_limit_kw=_number(f"{name}.methanator", limits["methanator"]),
        battery_charge_limit_kw=_number(
            f"{name}.battery_charge", limits["battery_charge"]
        ),
        battery_support_limit_kw=_number(
            f"{name}.battery_support", limits["battery_support"]
        ),
        battery_reserve_soc=_number(
            f"{name}.battery_reserve_soc", payload.get("battery_reserve_soc")
        ),
    )
    _exact(name, payload, envelope.to_dict())
    return envelope


def _decode_command(raw: object, *, name: str) -> ProcessSetpointCommandV1:
    payload = _mapping(name, raw)
    issuer = _mapping(f"{name}.issuer", payload.get("issuer"))
    _exact_keys(
        f"{name}.issuer", issuer, frozenset({"id", "source_schema", "source_sha256"})
    )
    command = ProcessSetpointCommandV1(
        decision_index=_integer(
            f"{name}.decision_index", payload.get("decision_index")
        ),
        observation_sha256=_digest(
            f"{name}.observation_sha256", payload.get("observation_sha256")
        ),
        hold_duration_s=_number(
            f"{name}.hold_duration_s", payload.get("hold_duration_s")
        ),
        issuer_id=_text(f"{name}.issuer.id", issuer.get("id")),
        source_schema=_text(
            f"{name}.issuer.source_schema", issuer.get("source_schema")
        ),
        source_sha256=_digest(
            f"{name}.issuer.source_sha256", issuer.get("source_sha256")
        ),
        targets=_decode_targets(payload.get("targets"), name=f"{name}.targets"),
        envelope=_decode_envelope(payload.get("envelope"), name=f"{name}.envelope"),
    )
    _exact(name, payload, command.to_dict())
    return command


def _decode_support(raw: object, *, name: str) -> BatterySupportAvailability:
    payload = _mapping(name, raw)
    expected = {item.name for item in fields(BatterySupportAvailability)}
    _exact_keys(name, payload, frozenset(expected))
    values: dict[str, object] = {}
    for key, value in payload.items():
        values[key] = (
            _boolean(f"{name}.{key}", value)
            if key in {"battery_available", "electrolyser_available"}
            else _number(f"{name}.{key}", value)
        )
    support = BatterySupportAvailability(**values)  # type: ignore[arg-type]
    _exact(name, payload, support.to_dict())
    return support


def _decode_allocation(
    raw: object,
    *,
    name: str,
    expected_targets: ProcessSetpointTargetsV1,
) -> ProcessSetpointAllocationV1:
    payload = _mapping(name, raw)
    request = _mapping(f"{name}.request_kw", payload.get("request_kw"))
    _exact_keys(
        f"{name}.request_kw",
        request,
        frozenset(
            {
                "solar_to_battery",
                "solar_to_electrolyser",
                "solar_to_dac",
                "solar_to_methanator",
                "battery_support",
                "unallocated_solar",
                "electrolyser_total",
            }
        ),
    )
    allocation = ProcessSetpointAllocationV1(
        targets=_decode_targets(payload.get("targets"), name=f"{name}.targets"),
        solar_available_kw=_number(
            f"{name}.solar_available_kw", payload.get("solar_available_kw")
        ),
        support_availability=_decode_support(
            payload.get("support_availability"), name=f"{name}.support_availability"
        ),
        routes_kw=(
            _number(f"{name}.solar_to_battery", request["solar_to_battery"]),
            _number(f"{name}.solar_to_electrolyser", request["solar_to_electrolyser"]),
            _number(f"{name}.solar_to_dac", request["solar_to_dac"]),
            _number(f"{name}.solar_to_methanator", request["solar_to_methanator"]),
        ),
        support_kw=_number(f"{name}.battery_support", request["battery_support"]),
        unallocated_kw=_number(
            f"{name}.unallocated_solar", request["unallocated_solar"]
        ),
        target_reachable=_boolean(
            f"{name}.electrolyser_target_reachable",
            payload.get("electrolyser_target_reachable"),
        ),
        action=_decode_action(payload.get("action"), name=f"{name}.action"),
    )
    if allocation.targets != expected_targets:
        raise ValueError(f"{name} belongs to another command target")
    _exact(name, payload, allocation.to_dict())
    return allocation


def _decode_dispatch(
    raw: object,
    *,
    name: str,
    command: ProcessSetpointCommandV1,
) -> ProcessSetpointMinuteDispatchV1:
    payload = _mapping(name, raw)
    boundary = _mapping(f"{name}.boundary", payload.get("boundary"))
    _exact_keys(f"{name}.boundary", boundary, frozenset({"sha256", "step"}))
    dispatch = ProcessSetpointMinuteDispatchV1(
        command=command,
        boundary_sha256=_digest(f"{name}.boundary.sha256", boundary.get("sha256")),
        boundary_step=_integer(f"{name}.boundary.step", boundary.get("step")),
        allocation=_decode_allocation(
            payload.get("allocation"),
            name=f"{name}.allocation",
            expected_targets=command.targets,
        ),
    )
    _exact(name, payload, dispatch.to_dict())
    return dispatch


def _decode_application(
    raw: object,
    *,
    name: str,
    command: ProcessSetpointCommandV1,
) -> ProcessSetpointMinuteApplicationV1:
    payload = _mapping(name, raw)
    guard = _mapping(f"{name}.guard", payload.get("guard"))
    next_boundary = _mapping(f"{name}.next_boundary", payload.get("next_boundary"))
    plant = _mapping(f"{name}.plant", payload.get("plant"))
    applied = _mapping(f"{name}.plant.applied_kw", plant.get("applied_kw"))
    attribution = _mapping(
        f"{name}.plant.battery_charge_attribution",
        plant.get("battery_charge_attribution"),
    )
    adjustments = tuple(
        _text(f"{name}.guard.adjustments[{index}]", item)
        for index, item in enumerate(
            _array(f"{name}.guard.adjustments", guard.get("adjustments"))
        )
    )
    application = ProcessSetpointMinuteApplicationV1(
        ordinal=_integer(f"{name}.ordinal", payload.get("ordinal")),
        dispatch=_decode_dispatch(
            payload.get("dispatch"), name=f"{name}.dispatch", command=command
        ),
        guarded_action=_decode_action(guard.get("action"), name=f"{name}.guard.action"),
        adjustments=adjustments,
        next_boundary_sha256=_digest(
            f"{name}.next_boundary.sha256", next_boundary.get("sha256")
        ),
        next_boundary_step=_integer(
            f"{name}.next_boundary.step", next_boundary.get("step")
        ),
        applied_action=_decode_action(plant.get("action"), name=f"{name}.plant.action"),
        applied_solar_available_kw=_number(
            f"{name}.applied.solar_available", applied.get("solar_available")
        ),
        applied_solar_to_battery_kw=_number(
            f"{name}.applied.solar_to_battery", applied.get("solar_to_battery")
        ),
        applied_solar_to_electrolyser_kw=_number(
            f"{name}.applied.solar_to_electrolyser",
            applied.get("solar_to_electrolyser"),
        ),
        applied_solar_to_dac_kw=_number(
            f"{name}.applied.solar_to_dac", applied.get("solar_to_dac")
        ),
        applied_solar_to_methanator_kw=_number(
            f"{name}.applied.solar_to_methanator",
            applied.get("solar_to_methanator"),
        ),
        applied_battery_support_kw=_number(
            f"{name}.applied.battery_support", applied.get("battery_support")
        ),
        applied_electrolyser_kw=_number(
            f"{name}.applied.electrolyser", applied.get("electrolyser")
        ),
        applied_dac_kw=_number(f"{name}.applied.dac", applied.get("dac")),
        applied_methanator_kw=_number(
            f"{name}.applied.methanator", applied.get("methanator")
        ),
        applied_solar_curtailed_kw=_number(
            f"{name}.applied.solar_curtailed", applied.get("solar_curtailed")
        ),
    )
    # The typed record recomputes the guard flag and battery attribution too.
    _boolean(f"{name}.guard.changed_request", guard.get("changed_request"))
    for key, value in attribution.items():
        _number(f"{name}.battery_charge_attribution.{key}", value)
    _exact(name, payload, application.to_dict())
    return application


def _decode_execution(
    raw: object,
    *,
    name: str,
    command: ProcessSetpointCommandV1,
) -> ProcessSetpointIntervalExecutionV1:
    payload = _mapping(name, raw)
    encoded_command = _decode_command(payload.get("command"), name=f"{name}.command")
    if encoded_command != command:
        raise ValueError(f"{name} embeds another decision command")
    applications = tuple(
        _decode_application(
            item,
            name=f"{name}.applications[{index}]",
            command=command,
        )
        for index, item in enumerate(
            _array(f"{name}.applications", payload.get("applications"))
        )
    )
    execution = ProcessSetpointIntervalExecutionV1(
        command=command,
        physics_step_s=_number(f"{name}.physics_step_s", payload.get("physics_step_s")),
        applications=applications,
    )
    _exact(name, payload, execution.to_dict())
    return execution


def _decode_plant_state(raw: object, *, name: str) -> PlantState:
    payload = _mapping(name, raw)
    field_names = {item.name for item in fields(PlantState)}
    expected = field_names | {"time_h", "time_of_day_h", "day", "clock"}
    _exact_keys(name, payload, frozenset(expected))
    values = {key: payload[key] for key in field_names}
    values["step"] = _integer(f"{name}.step", values["step"])
    values["time_s"] = _number(f"{name}.time_s", values["time_s"])
    values["battery_soc"] = _number(f"{name}.battery_soc", values["battery_soc"])
    values["battery_energy_kwh"] = _number(
        f"{name}.battery_energy_kwh", values["battery_energy_kwh"]
    )
    values["battery_capacity_kwh"] = _number(
        f"{name}.battery_capacity_kwh", values["battery_capacity_kwh"]
    )
    if not 0.0 <= values["battery_soc"] <= 1.0:
        raise ValueError(f"{name}.battery_soc lies outside [0, 1]")
    values["plant_status"] = _text(f"{name}.plant_status", values["plant_status"])
    if values["plant_status"] not in {"normal", "degraded", CRITICAL, OFFLINE}:
        raise ValueError(f"{name}.plant_status is unsupported")
    values["action"] = _decode_action(values["action"], name=f"{name}.action")
    health_raw = _mapping(f"{name}.health", values["health"])
    health: dict[str, ComponentHealth] = {}
    health_fields = frozenset(item.name for item in fields(ComponentHealth))
    for unit, item in health_raw.items():
        record = _mapping(f"{name}.health.{unit}", item)
        _exact_keys(f"{name}.health.{unit}", record, health_fields)
        health[unit] = ComponentHealth(
            status=_text(f"{name}.health.{unit}.status", record["status"]),
            degradation=_number(
                f"{name}.health.{unit}.degradation", record["degradation"]
            ),
            offline=_boolean(f"{name}.health.{unit}.offline", record["offline"]),
        )
    values["health"] = health
    state = PlantState(**values)
    _exact(name, payload, state.to_dict())
    return state


def _decode_interval_summary(raw: object, *, name: str) -> IntervalSummaryV2:
    payload = _mapping(name, raw)
    base_names = {item.name for item in fields(IntervalSummaryV2)}
    summary = IntervalSummaryV2(**{key: payload[key] for key in base_names})
    _exact(name, payload, summary.to_dict())
    return summary


def _decode_commitment(
    raw: object, *, name: str
) -> ElectrolyserCommitmentSnapshot | None:
    if raw is None:
        return None
    payload = _mapping(name, raw)
    field_names = {item.name for item in fields(ElectrolyserCommitmentSnapshot)}
    _exact_keys(name, payload, frozenset(field_names | {"committed"}))
    values = {key: payload[key] for key in field_names}
    values["mode"] = ElectrolyserMode(_text(f"{name}.mode", values["mode"]))
    mode_times = _mapping(f"{name}.mode_time_s", values["mode_time_s"])
    if set(mode_times) != {mode.value for mode in ElectrolyserMode}:
        raise ValueError(f"{name}.mode_time_s must contain every commitment mode")
    values["mode_time_s"] = tuple(
        (mode, _number(f"{name}.mode_time_s.{mode.value}", mode_times[mode.value]))
        for mode in ElectrolyserMode
    )
    snapshot = ElectrolyserCommitmentSnapshot(**values)
    _exact(name, payload, snapshot.to_dict())
    return snapshot


def _decode_weather(raw: object, *, name: str) -> PublicForecastIssueV2:
    payload = _mapping(name, raw)
    windows = tuple(
        ForecastWindowV2(
            name=_text(f"{name}.windows[{index}].name", item.get("name")),
            start_offset_s_from_issue=_number(
                f"{name}.windows[{index}].start", item.get("start_offset_s_from_issue")
            ),
            end_offset_s_from_issue=_number(
                f"{name}.windows[{index}].end", item.get("end_offset_s_from_issue")
            ),
            mean_peak_power_fraction=_number(
                f"{name}.windows[{index}].mean", item.get("mean_peak_power_fraction")
            ),
            p10_peak_power_fraction=_number(
                f"{name}.windows[{index}].p10", item.get("p10_peak_power_fraction")
            ),
        )
        for index, item in enumerate(
            _mapping(f"{name}.windows[{index}]", raw_item)
            for index, raw_item in enumerate(
                _array(f"{name}.windows", payload.get("windows"))
            )
        )
    )
    issue = PublicForecastIssueV2(
        issued_at_physics_boundary_step=_integer(
            f"{name}.issued_at_physics_boundary_step",
            payload.get("issued_at_physics_boundary_step"),
        ),
        issued_at_offset_s=_number(
            f"{name}.issued_at_offset_s", payload.get("issued_at_offset_s")
        ),
        windows=windows,
    )
    _exact(name, payload, issue.to_dict())
    return issue


def _decode_physical_observation(raw: object, *, name: str) -> ControllerObservationV2:
    payload = _mapping(name, raw)
    _exact_keys(
        name,
        payload,
        frozenset({"state", "delivery", "forecast", "progress", "decision_interval_s"}),
    )
    state_raw = _mapping(f"{name}.state", payload["state"])
    _exact_keys(
        f"{name}.state",
        state_raw,
        frozenset({"boundary", "previous_interval", "electrolyser_commitment"}),
    )
    state = ControlStateV2(
        boundary=_decode_plant_state(
            state_raw["boundary"], name=f"{name}.state.boundary"
        ),
        previous_interval=_decode_interval_summary(
            state_raw["previous_interval"], name=f"{name}.state.previous_interval"
        ),
        electrolyser_commitment=_decode_commitment(
            state_raw["electrolyser_commitment"],
            name=f"{name}.state.electrolyser_commitment",
        ),
    )
    delivery_raw = _mapping(f"{name}.delivery", payload["delivery"])
    delivery = DeliveryObservation(
        steps_to_next=(
            None
            if delivery_raw.get("steps_to_next") is None
            else _integer(
                f"{name}.delivery.steps_to_next", delivery_raw["steps_to_next"]
            )
        ),
        delivery_period_steps=_integer(
            f"{name}.delivery.delivery_period_steps",
            delivery_raw.get("delivery_period_steps"),
            minimum=1,
        ),
        next_requested_kg=_number(
            f"{name}.delivery.next_requested_kg", delivery_raw.get("next_requested_kg")
        ),
        cumulative_requested_kg=_number(
            f"{name}.delivery.cumulative_requested_kg",
            delivery_raw.get("cumulative_requested_kg"),
        ),
        cumulative_delivered_kg=_number(
            f"{name}.delivery.cumulative_delivered_kg",
            delivery_raw.get("cumulative_delivered_kg"),
        ),
    )
    forecast_raw = payload["forecast"]
    if forecast_raw is None:
        raise ValueError(f"{name} beta forecast cannot be null")
    observation = ControllerObservationV2(
        state=state,
        delivery=delivery,
        forecast=_decode_weather(forecast_raw, name=f"{name}.forecast"),
        progress=_number(f"{name}.progress", payload["progress"]),
        decision_interval_s=_number(
            f"{name}.decision_interval_s", payload["decision_interval_s"]
        ),
    )
    _exact(name, payload, observation.to_dict())
    return observation


def _decode_market(raw: object, *, name: str) -> PublicPriceForecastIssueV1:
    payload = _mapping(name, raw)
    bins: list[PriceForecastBinV1] = []
    for index, raw_bin in enumerate(_array(f"{name}.bins", payload.get("bins"))):
        item = _mapping(f"{name}.bins[{index}]", raw_bin)
        forecast_bin = PriceForecastBinV1(
            hour_index=_integer(
                f"{name}.bins[{index}].hour_index", item.get("hour_index")
            ),
            mean_eur_per_mwh=_number(
                f"{name}.bins[{index}].mean", item.get("mean_eur_per_mwh")
            ),
            p10_eur_per_mwh=_number(
                f"{name}.bins[{index}].p10", item.get("p10_eur_per_mwh")
            ),
            p90_eur_per_mwh=_number(
                f"{name}.bins[{index}].p90", item.get("p90_eur_per_mwh")
            ),
        )
        _exact(f"{name}.bins[{index}]", item, forecast_bin.to_dict())
        bins.append(forecast_bin)
    issue = PublicPriceForecastIssueV1(
        issued_at_decision=_integer(
            f"{name}.issued_at_decision", payload.get("issued_at_decision")
        ),
        issued_at_offset_s=_number(
            f"{name}.issued_at_offset_s", payload.get("issued_at_offset_s")
        ),
        current_eur_per_mwh=_number(
            f"{name}.current_eur_per_mwh", payload.get("current_eur_per_mwh")
        ),
        bins=tuple(bins),
        market_id=_text(f"{name}.market_id", payload.get("market_id")),
    )
    if payload.get("schema") != PUBLIC_PRICE_FORECAST_SCHEMA:
        raise ValueError(f"{name} schema differs")
    _exact(name, payload, issue.to_dict())
    return issue


def _decode_cycle(raw: object, *, name: str) -> CycleObservationV1:
    payload = _mapping(name, raw)
    if payload.get("schema") != CYCLE_OBSERVATION_SCHEMA:
        raise ValueError(f"{name} schema differs")
    cycle = CycleObservationV1(
        cycle_index=_integer(
            f"{name}.cycle_index", payload.get("cycle_index"), minimum=1
        ),
        decision_in_cycle=_integer(
            f"{name}.decision_in_cycle", payload.get("decision_in_cycle")
        ),
        cycle_decisions=_integer(
            f"{name}.cycle_decisions", payload.get("cycle_decisions"), minimum=1
        ),
        progress=_number(f"{name}.progress", payload.get("progress")),
        battery_margin_kwh=_number(
            f"{name}.battery_margin_kwh", payload.get("battery_margin_kwh")
        ),
        h2_margin_kg=_number(f"{name}.h2_margin_kg", payload.get("h2_margin_kg")),
        co2_margin_kg=_number(f"{name}.co2_margin_kg", payload.get("co2_margin_kg")),
        ch4_margin_kg=_number(f"{name}.ch4_margin_kg", payload.get("ch4_margin_kg")),
    )
    _exact(name, payload, cycle.to_dict())
    return cycle


def _decode_beta_observation(
    raw: object,
    *,
    name: str,
) -> CyclicMarketControllerObservationV1:
    payload = _mapping(name, raw)
    _exact_keys(
        name,
        payload,
        frozenset(
            {
                "schema",
                "plant",
                "market",
                "cycle",
                "economic_assumptions_sha256",
                "beta_task_sha256",
            }
        ),
    )
    if payload.get("schema") != BETA_CONTROLLER_OBSERVATION_SCHEMA:
        raise ValueError(f"{name} schema differs")
    observation = CyclicMarketControllerObservationV1(
        plant=_decode_physical_observation(payload["plant"], name=f"{name}.plant"),
        market=_decode_market(payload["market"], name=f"{name}.market"),
        cycle=_decode_cycle(payload["cycle"], name=f"{name}.cycle"),
        economic_assumptions_sha256=_digest(
            f"{name}.economic_assumptions_sha256",
            payload.get("economic_assumptions_sha256"),
        ),
        beta_task_sha256=_digest(
            f"{name}.beta_task_sha256", payload.get("beta_task_sha256")
        ),
    )
    _exact(name, payload, observation.to_dict())
    return observation


def _verify_boundary(
    raw: object,
    *,
    name: str,
    config: ControllerNeutralTrainingConfigV1,
    encoder: ObservationEncoderV2,
    expected_kind: str,
    expected_decision: int,
    expected_physics_step: int,
    terminal: bool,
    previous_solar_kw: float | None,
) -> tuple[dict[str, Any], CyclicMarketControllerObservationV1, np.ndarray]:
    payload = _mapping(name, raw)
    _exact_keys(name, payload, _BOUNDARY_FIELDS)
    if payload["format"] != "cyclic-market-controller-boundary-v1":
        raise ValueError(f"{name} format differs")
    if payload["kind"] != expected_kind:
        raise ValueError(f"{name} kind differs")
    if (
        _integer(f"{name}.decision_index", payload["decision_index"])
        != expected_decision
    ):
        raise ValueError(f"{name} decision order differs")
    if (
        _integer(f"{name}.physics_step", payload["physics_step"])
        != expected_physics_step
    ):
        raise ValueError(f"{name} physics boundary differs")
    if _boolean(f"{name}.terminal", payload["terminal"]) != terminal:
        raise ValueError(f"{name} terminal flag differs")
    observation = _decode_beta_observation(
        payload["typed_observation"], name=f"{name}.typed_observation"
    )
    if payload["typed_observation_sha256"] != observation.sha256:
        raise ValueError(f"{name} typed observation digest differs")
    physical_sha = domain_sha256(
        SERVICE_MPC_OBSERVATION_DOMAIN, observation.plant.to_dict()
    )
    if payload["physical_observation_sha256"] != physical_sha:
        raise ValueError(f"{name} physical observation digest differs")
    if payload["market_issue_sha256"] != observation.market.sha256:
        raise ValueError(f"{name} market issue digest differs")
    cycle_sha = domain_sha256(
        BETA_MPC_CYCLE_OBSERVATION_DOMAIN, observation.cycle.to_dict()
    )
    if payload["cycle_observation_sha256"] != cycle_sha:
        raise ValueError(f"{name} cycle observation digest differs")

    decision_interval_s = config.episode.decision_interval_s
    if not math.isclose(
        observation.plant.decision_interval_s,
        decision_interval_s,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{name} physical cadence differs from the task")
    if observation.plant.state.boundary.step != expected_physics_step:
        raise ValueError(f"{name} plant-state step differs from its ledger position")
    state = observation.plant.state.boundary
    expected_time_s = (
        config.episode.start_hour * 3_600.0 + expected_physics_step * config.plant.dt_s
    )
    if not math.isclose(state.time_s, expected_time_s, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{name} plant-state clock differs from its physics grid")
    episode_decisions = config.episode.decision_steps(config.plant)
    if not math.isclose(
        observation.plant.progress,
        min(1.0, expected_decision / episode_decisions),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{name} task progress differs from the decision grid")
    forecast = observation.plant.forecast
    if forecast is None:  # pragma: no cover - typed decoder requires a forecast
        raise ValueError(f"{name} omitted its public weather issue")
    stride = config.episode.decision_stride(config.plant)
    expected_forecast_step = (
        max(0, expected_physics_step - stride) if terminal else expected_physics_step
    )
    if (
        forecast.issued_at_physics_boundary_step != expected_forecast_step
        or not math.isclose(
            forecast.issued_at_offset_s,
            expected_forecast_step * config.plant.dt_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError(f"{name} public weather issue is stale or off-grid")
    capacity_expectations = {
        "h2_capacity_kg": config.plant.h2_tank.capacity_kg,
        "co2_capacity_kg": config.plant.co2_tank.capacity_kg,
        "ch4_capacity_kg": config.plant.ch4_tank.capacity_kg,
    }
    for field, expected in capacity_expectations.items():
        if not math.isclose(
            float(getattr(state, field)), float(expected), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"{name} static {field} differs from plant config")
    if not 0.0 < state.battery_capacity_kwh <= config.plant.battery.capacity_kwh + 1e-9:
        raise ValueError(f"{name} battery capacity lies outside its plant design")
    for level, capacity, fraction in (
        (state.battery_energy_kwh, state.battery_capacity_kwh, state.battery_soc),
        (state.h2_level_kg, state.h2_capacity_kg, state.h2_fill),
        (state.co2_level_kg, state.co2_capacity_kg, state.co2_fill),
        (state.ch4_level_kg, state.ch4_capacity_kg, state.ch4_fill),
    ):
        expected_fraction = level / capacity if capacity > 0.0 else 0.0
        if not math.isclose(fraction, expected_fraction, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{name} public inventory fraction is inconsistent")
    if observation.economic_assumptions_sha256 != config.economic_assumptions.sha256:
        raise ValueError(f"{name} economic assumptions identity differs")
    if config.beta_task is None:  # pragma: no cover - report config validation closes
        raise RuntimeError("validated beta report omitted beta task")
    if observation.beta_task_sha256 != config.beta_task.sha256:
        raise ValueError(f"{name} beta task identity differs")

    cycle_count = config.beta_task.cycle.cycle_decisions(config.episode)
    if terminal:
        expected_cycle_index = max(1, math.ceil(expected_decision / cycle_count))
        expected_in_cycle = cycle_count
        expected_market_decision = max(0, expected_decision - 1)
    else:
        expected_cycle_index = expected_decision // cycle_count + 1
        expected_in_cycle = expected_decision % cycle_count
        expected_market_decision = expected_decision
    if (
        observation.cycle.cycle_index != expected_cycle_index
        or observation.cycle.cycle_decisions != cycle_count
        or observation.cycle.decision_in_cycle != expected_in_cycle
    ):
        raise ValueError(f"{name} cycle counters differ from the decision grid")
    if observation.market.issued_at_decision != expected_market_decision:
        raise ValueError(f"{name} market issue is stale or from another decision")
    if not math.isclose(
        observation.market.issued_at_offset_s,
        expected_market_decision * decision_interval_s,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{name} market issue offset differs from the decision grid")

    vector_raw = _array(
        f"{name}.normalized_observation", payload["normalized_observation"]
    )
    if len(vector_raw) != beta_spaces.OBSERVATION_SIZE:
        raise ValueError(f"{name} normalized observation has the wrong size")
    vector = cast(
        NDArray[np.float32],
        np.asarray(
            [
                _number(f"{name}.normalized_observation[{i}]", value)
                for i, value in enumerate(vector_raw)
            ],
            dtype=np.float32,
        ),
    )
    expected_vector = np.concatenate(
        (
            encoder.encode_controller_observation(
                observation.plant, constraints=config.constraints
            ),
            np.asarray(observation.market.normalized(), dtype=np.float32),
            np.asarray(observation.cycle.normalized(config.plant), dtype=np.float32),
        )
    ).astype(np.float32, copy=False)
    current_solar_kw = float(observation.plant.state.boundary.solar_available_kw)
    prior_solar_kw = (
        current_solar_kw if previous_solar_kw is None else previous_solar_kw
    )
    solar_trend_index = beta_spaces.OBSERVATION_NAMES.index("solar_trend")
    expected_vector[solar_trend_index] = np.float32(
        np.clip(
            (current_solar_kw - prior_solar_kw) / config.plant.solar.peak_kw,
            -1.0,
            1.0,
        )
    )
    if not np.array_equal(vector, expected_vector):
        raise ValueError(f"{name} normalized vector differs from typed public evidence")
    expected_vector_sha = domain_sha256(
        _VECTOR_DOMAIN, cast(list[float], vector.tolist())
    )
    if payload["normalized_observation_sha256"] != expected_vector_sha:
        raise ValueError(f"{name} normalized vector digest differs")
    return payload, observation, vector


def _parse_service_settings(raw: object) -> ServiceMPCSetpointSettingsV1:
    payload = _mapping("service MPC settings", raw)
    solver = _mapping("service MPC solver options", payload.get("solver_options"))
    settings = ServiceMPCSetpointSettingsV1(
        task_horizon_h=_number("service task horizon", payload.get("task_horizon_h")),
        planning_horizon_h=_number(
            "service planning horizon", payload.get("planning_horizon_h")
        ),
        interval_s=_number("service interval", payload.get("interval_s")),
        terminal_battery_soc_target=_number(
            "service terminal battery target",
            payload.get("terminal_battery_soc_target"),
        ),
        delivery_lock_tolerance_kg=_number(
            "service delivery tolerance", payload.get("delivery_lock_tolerance_kg")
        ),
        energy_reconstruction_tolerance=_number(
            "service reconstruction tolerance",
            payload.get("energy_reconstruction_tolerance"),
        ),
        solver_presolve=_boolean("service presolve", solver.get("presolve")),
        solver_mip_rel_gap=_number("service MIP gap", solver.get("mip_rel_gap")),
        solver_mip_feasibility_tolerance=_number(
            "service MIP feasibility tolerance", solver.get("mip_feasibility_tolerance")
        ),
        solver_time_limit_s=_number(
            "service solver time limit", solver.get("time_limit")
        ),
        minimum_fill_weight=_number(
            "service minimum fill weight",
            _mapping("service weights", payload.get("secondary_objective_weights")).get(
                "minimum_commitment_fill"
            ),
        ),
        terminal_ch4_equivalent_weight=_number(
            "service terminal methane weight",
            _mapping("service weights", payload.get("secondary_objective_weights")).get(
                "terminal_ch4_equivalent_inventory"
            ),
        ),
        terminal_battery_weight=_number(
            "service terminal battery weight",
            _mapping("service weights", payload.get("secondary_objective_weights")).get(
                "terminal_battery_energy"
            ),
        ),
        cold_start_weight=_number(
            "service cold-start weight",
            _mapping("service weights", payload.get("secondary_objective_weights")).get(
                "cold_starts"
            ),
        ),
        curtailment_weight=_number(
            "service curtailment weight",
            _mapping("service weights", payload.get("secondary_objective_weights")).get(
                "curtailed_energy"
            ),
        ),
        battery_throughput_weight=_number(
            "service throughput weight",
            _mapping("service weights", payload.get("secondary_objective_weights")).get(
                "battery_throughput"
            ),
        ),
    )
    _exact("service MPC settings", payload, settings.to_dict())
    return settings


def _parse_beta_settings(raw: object) -> BetaMPCSettingsV1:
    payload = _mapping("beta MPC settings", raw)
    solver = _mapping("beta MPC solver options", payload.get("solver_options"))
    settings = BetaMPCSettingsV1(
        task_horizon_h=_number("beta MPC task horizon", payload.get("task_horizon_h")),
        planning_horizon_h=_number(
            "beta MPC planning horizon", payload.get("planning_horizon_h")
        ),
        interval_s=_number("beta MPC interval", payload.get("interval_s")),
        terminal_battery_soc_target=_number(
            "beta MPC terminal battery target",
            payload.get("terminal_battery_soc_target"),
        ),
        delivery_lock_tolerance_kg=_number(
            "beta MPC delivery tolerance", payload.get("delivery_lock_tolerance_kg")
        ),
        energy_reconstruction_tolerance=_number(
            "beta MPC reconstruction tolerance",
            payload.get("energy_reconstruction_tolerance"),
        ),
        solver_presolve=_boolean("beta MPC presolve", solver.get("presolve")),
        solver_mip_rel_gap=_number("beta MPC MIP gap", solver.get("mip_rel_gap")),
        solver_mip_feasibility_tolerance=_number(
            "beta MPC MIP feasibility tolerance",
            solver.get("mip_feasibility_tolerance"),
        ),
        solver_time_limit_s=_number(
            "beta MPC solver time limit", solver.get("time_limit")
        ),
    )
    _exact("beta MPC settings", payload, settings.to_dict())
    return settings


def _verify_manifest_identity(
    raw: object,
    *,
    name: str,
    config: ControllerNeutralTrainingConfigV1,
) -> dict[str, str]:
    payload = _mapping(name, raw)
    _exact_keys(name, payload, _MANIFEST_IDENTITY_FIELDS)
    result = {key: _digest(f"{name}.{key}", value) for key, value in payload.items()}
    if config.beta_task is None:  # pragma: no cover
        raise RuntimeError("validated beta report omitted beta task")
    if result["beta_task_sha256"] != config.beta_task.sha256:
        raise ValueError(f"{name} beta task differs from training task")
    if result["cycle_task_sha256"] != config.beta_task.sha256:
        raise ValueError(f"{name} cycle task differs from training task")
    if result["economic_assumptions_sha256"] != config.economic_assumptions.sha256:
        raise ValueError(f"{name} economic assumptions differ from training task")
    return result


def _verify_audit_manifest(
    raw: object,
    *,
    identity: Mapping[str, str],
    config: ControllerNeutralTrainingConfigV1,
    seed: int,
) -> dict[str, Any]:
    manifest = _mapping("beta audit manifest", raw)
    _exact_keys(
        "beta audit manifest",
        manifest,
        frozenset(
            {
                "manifest_sha256",
                "alpha_physical_exogenous_episode_sha256",
                "beta_exogenous_episode_sha256",
                "manifest_schema_version",
                "environment",
                "checkpoint_contract",
                "objective_version",
                "diagnostic_legacy_objective_version",
                "episode_config",
                "beta_task",
                "economic_assumptions",
                "plant_config",
                "constraints",
                "diagnostic_legacy_reward_weights",
                "episode",
                "market",
                "disturbances",
                "cycle_reference",
                "beta_exogenous",
                "provenance_gaps",
                "delivery_accounting",
                "economic_accounting",
                "cycle_accounting",
            }
        ),
    )
    immutable = {
        key: value
        for key, value in manifest.items()
        if key
        not in {
            "manifest_sha256",
            "alpha_physical_exogenous_episode_sha256",
            "beta_exogenous_episode_sha256",
            "delivery_accounting",
            "economic_accounting",
            "cycle_accounting",
        }
    }
    if canonical_sha256(immutable) != manifest["manifest_sha256"]:
        raise ValueError("beta audit manifest digest differs")
    beta_exogenous = _mapping("beta exogenous manifest", manifest["beta_exogenous"])
    if canonical_sha256(beta_exogenous) != manifest["beta_exogenous_episode_sha256"]:
        raise ValueError("beta exogenous episode digest differs")
    disturbances = _mapping("beta disturbance manifest", manifest["disturbances"])
    checkpoint = _mapping("beta checkpoint contract", manifest["checkpoint_contract"])
    if config.beta_task is None:  # pragma: no cover
        raise RuntimeError("validated beta report omitted beta task")
    expected_checkpoint = describe_beta_checkpoint_contract(
        plant=config.plant,
        episode=config.episode,
        constraints=config.constraints,
        weights=config.diagnostic_reward,
        assumptions=config.economic_assumptions,
        beta_task=config.beta_task,
    )
    declared_task = {
        "environment": beta_spaces.describe(),
        "checkpoint_contract": expected_checkpoint,
        "episode_config": config.episode.to_dict(),
        "plant_config": config.plant.to_dict(),
        "constraints": config.constraints.to_dict(),
        "diagnostic_legacy_reward_weights": config.diagnostic_reward.to_dict(),
        "economic_assumptions": config.economic_assumptions.to_dict(),
        "beta_task": config.beta_task.to_dict(),
    }
    for key, expected in declared_task.items():
        _exact(f"beta manifest {key}", manifest[key], expected)
    cycle_decisions = config.beta_task.cycle.cycle_decisions(config.episode)
    cycle_count = config.episode.decision_steps(config.plant) // cycle_decisions
    root_streams = np.random.SeedSequence(seed).spawn(len(BETA_ROOT_STREAM_NAMES))
    regenerated_disturbances = generate_disturbance_episode(
        config.beta_task.disturbances,
        seed=root_streams[4],
        cycle_h=config.beta_task.cycle.cycle_h,
        cycle_count=cycle_count,
        physics_dt_s=config.plant.dt_s,
    )
    _exact(
        "beta manifest regenerated disturbance episode",
        disturbances,
        regenerated_disturbances.private_audit_manifest(),
    )
    if (
        manifest["objective_version"] != ECONOMIC_OBJECTIVE_VERSION
        or manifest["diagnostic_legacy_objective_version"] != OBJECTIVE_VERSION
        or manifest["manifest_schema_version"] != 1
        or manifest["provenance_gaps"]
        != [
            "source revision and dependency-lock digest must be added by the experiment runner"
        ]
    ):
        raise ValueError("beta manifest static contract fields differ")
    expected_identity = {
        "manifest_sha256": manifest["manifest_sha256"],
        "alpha_physical_exogenous_episode_sha256": manifest[
            "alpha_physical_exogenous_episode_sha256"
        ],
        "beta_exogenous_episode_sha256": manifest["beta_exogenous_episode_sha256"],
        "market_episode_sha256": beta_exogenous.get("market_episode_sha256"),
        "disturbance_episode_sha256": beta_exogenous.get("disturbance_episode_sha256"),
        "disturbance_schedule_sha256": disturbances.get("schedule_sha256"),
        "cycle_task_sha256": beta_exogenous.get("cycle_task_sha256"),
        "checkpoint_contract_sha256": checkpoint.get("checkpoint_contract_digest"),
        "economic_assumptions_sha256": canonical_sha256(
            manifest["economic_assumptions"]
        ),
        "beta_task_sha256": canonical_sha256(manifest["beta_task"]),
    }
    _exact("beta manifest identity", dict(identity), expected_identity)
    return manifest


def _verify_checkpoint_and_policy_descriptor(
    checkpoint_raw: object,
    *,
    config: ControllerNeutralTrainingConfigV1,
) -> tuple[dict[str, Any], str, str]:
    if config.beta_task is None:  # pragma: no cover - caller validates beta task
        raise RuntimeError("validated beta report omitted beta task")
    expected_contract = describe_beta_checkpoint_contract(
        plant=config.plant,
        episode=config.episode,
        constraints=config.constraints,
        weights=config.diagnostic_reward,
        assumptions=config.economic_assumptions,
        beta_task=config.beta_task,
    )
    checkpoint = _mapping("benchmark checkpoint", checkpoint_raw)
    _exact_keys(
        "benchmark checkpoint",
        checkpoint,
        frozenset(
            {
                "path",
                "format",
                "algorithm",
                "trained",
                "plant_sim_version",
                "total_timesteps",
                "metadata_file_sha256",
                "policy_artifact_sha256",
                "checkpoint_contract_sha256",
            }
        ),
    )
    _text("checkpoint path", checkpoint["path"])
    _text("checkpoint format", checkpoint["format"])
    _text("checkpoint algorithm", checkpoint["algorithm"])
    _boolean("checkpoint trained", checkpoint["trained"])
    _text("checkpoint plant_sim_version", checkpoint["plant_sim_version"])
    _integer("checkpoint total timesteps", checkpoint["total_timesteps"])
    for key in ("metadata_file_sha256", "checkpoint_contract_sha256"):
        _digest(f"checkpoint {key}", checkpoint[key])
    if checkpoint["format"] == checkpoints.FORMAT_FIXED:
        if (
            checkpoint["trained"] is not False
            or checkpoint["policy_artifact_sha256"] != ""
        ):
            raise ValueError("fixed checkpoint claims trained policy bytes")
    else:
        _digest(
            "checkpoint policy_artifact_sha256", checkpoint["policy_artifact_sha256"]
        )
    if (
        checkpoint["checkpoint_contract_sha256"]
        != expected_contract["checkpoint_contract_digest"]
    ):
        raise ValueError("checkpoint contract digest differs from the training task")
    return (
        checkpoint,
        str(checkpoint["metadata_file_sha256"]),
        str(checkpoint["checkpoint_contract_sha256"]),
    )


def _verify_policy_controller(
    raw: object,
    *,
    config: ControllerNeutralTrainingConfigV1,
    checkpoint: Mapping[str, Any],
) -> tuple[str, str]:
    descriptor = _mapping("PPO controller descriptor", raw)
    _exact_keys(
        "PPO controller descriptor",
        descriptor,
        frozenset(
            {
                "id",
                "label",
                "description",
                "uses_forecast",
                "information_set",
                "checkpoint",
                "policy_contract",
                "constraints",
                "checkpoint_metadata_sha256",
                "policy_contract_sha256",
                "economic_assumptions_sha256",
                "tariff_id",
                "beta_task_sha256",
                "inference",
                "failure_policy",
            }
        ),
    )
    expected_declarations = {
        "label": NormalizedSetpointPolicyControllerV1.label,
        "description": NormalizedSetpointPolicyControllerV1.description,
        "uses_forecast": NormalizedSetpointPolicyControllerV1.uses_forecast,
        "information_set": NormalizedSetpointPolicyControllerV1.information_set,
        "failure_policy": "raise; no controller-internal fallback",
    }
    _exact(
        "PPO controller declarations",
        {key: descriptor[key] for key in expected_declarations},
        expected_declarations,
    )
    if descriptor.get("id") != NORMALIZED_POLICY_SETPOINT_CONTROLLER_ID:
        raise ValueError("PPO controller identity differs")
    loaded = _mapping("PPO loaded checkpoint", descriptor.get("checkpoint"))
    _exact_keys("PPO loaded checkpoint", loaded, frozenset({"path", "metadata"}))
    metadata_raw = _mapping("PPO checkpoint metadata", loaded.get("metadata"))
    metadata = checkpoints.CheckpointMetadata.from_dict(metadata_raw)
    _exact("PPO checkpoint metadata", metadata_raw, metadata.to_dict())
    if canonical_json(metadata.training_config) != canonical_json(config.to_dict()):
        raise ValueError("PPO checkpoint training task differs from report")
    if metadata.trained and metadata.seed != config.seed:
        raise ValueError("trained PPO checkpoint seed differs from its training task")
    if (
        metadata.format != checkpoint["format"]
        or metadata.algorithm != checkpoint["algorithm"]
        or metadata.trained != checkpoint["trained"]
        or metadata.plant_sim_version != checkpoint["plant_sim_version"]
        or metadata.total_timesteps != checkpoint["total_timesteps"]
        or metadata.policy_artifact_sha256 != checkpoint["policy_artifact_sha256"]
        or metadata.checkpoint_contract_digest
        != checkpoint["checkpoint_contract_sha256"]
    ):
        raise ValueError("PPO descriptor and admitted checkpoint identity differ")
    loaded_path = _text("PPO checkpoint path", loaded.get("path"))
    admitted_path = str(checkpoint["path"])
    if not Path(loaded_path).is_absolute() or not Path(admitted_path).is_absolute():
        raise ValueError("PPO checkpoint paths must be absolute")
    if loaded_path != admitted_path:
        raise ValueError("PPO descriptor and admitted checkpoint paths differ")
    contract = _mapping("PPO policy contract", descriptor.get("policy_contract"))
    expected_contract = checkpoints.PolicyEnvironmentContract(
        spec_version=beta_spaces.SPEC_VERSION,
        observation_fields=beta_spaces.OBSERVATION_NAMES,
        action_fields=beta_spaces.ACTION_NAMES,
        environment_id=beta_spaces.ENVIRONMENT_ID,
        environment_contract_digest=beta_spaces.ENVIRONMENT_CONTRACT_DIGEST,
        checkpoint_contract_digest=str(checkpoint["checkpoint_contract_sha256"]),
    ).to_dict()
    _exact("PPO policy contract", contract, expected_contract)
    metadata_sha = domain_sha256(_CHECKPOINT_METADATA_DOMAIN, metadata.to_dict())
    contract_sha = domain_sha256(_POLICY_CONTRACT_DOMAIN, contract)
    if descriptor.get("checkpoint_metadata_sha256") != metadata_sha:
        raise ValueError("PPO checkpoint metadata digest differs")
    persisted_metadata = (
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if (
        hashlib.sha256(persisted_metadata).hexdigest()
        != checkpoint["metadata_file_sha256"]
    ):
        raise ValueError("PPO persisted metadata file digest differs")
    if descriptor.get("policy_contract_sha256") != contract_sha:
        raise ValueError("PPO policy contract digest differs")
    if descriptor.get("constraints") != config.constraints.to_dict():
        raise ValueError("PPO controller constraints differ from training task")
    if (
        descriptor.get("economic_assumptions_sha256")
        != config.economic_assumptions.sha256
    ):
        raise ValueError("PPO economic assumptions identity differs")
    if config.beta_task is None:  # pragma: no cover
        raise RuntimeError("validated beta report omitted beta task")
    if descriptor.get("beta_task_sha256") != config.beta_task.sha256:
        raise ValueError("PPO beta task identity differs")
    if descriptor.get("tariff_id") is not None:
        raise ValueError("PPO beta controller unexpectedly declares an alpha tariff")
    if descriptor.get("inference") != "deterministic":
        raise ValueError("PPO benchmark inference is not deterministic")
    return metadata_sha, contract_sha


def _verify_source_provenance(raw: object) -> dict[str, Any]:
    payload = _mapping("source provenance", raw)
    _exact_keys(
        "source provenance",
        payload,
        frozenset(
            {
                "git_revision",
                "worktree_clean",
                "worktree_status_sha256",
                "dependency_lock_sha256",
                "runtime",
            }
        ),
    )
    revision = _text("source revision", payload["git_revision"])
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("source revision is not a full lowercase Git object id")
    clean = _boolean("source worktree clean", payload["worktree_clean"])
    status_sha = _digest("source worktree status", payload["worktree_status_sha256"])
    if clean and status_sha != hashlib.sha256(b"").hexdigest():
        raise ValueError("clean source provenance has a non-empty worktree digest")
    _digest("source dependency lock", payload["dependency_lock_sha256"])
    runtime = _mapping("source runtime", payload["runtime"])
    _exact_keys(
        "source runtime",
        runtime,
        frozenset(
            {"python", "numpy", "scipy", "stable_baselines3", "gymnasium", "plant_sim"}
        ),
    )
    for key, value in runtime.items():
        if key == "plant_sim":
            continue
        _text(f"source runtime {key}", value)
    plant_sim = _mapping("source runtime plant_sim", runtime["plant_sim"])
    _exact_keys(
        "source runtime plant_sim",
        plant_sim,
        frozenset({"version", "path", "module_sha256"}),
    )
    _text("source runtime plant_sim version", plant_sim["version"])
    plant_sim_path = Path(_text("source runtime plant_sim path", plant_sim["path"]))
    if not plant_sim_path.is_absolute():
        raise ValueError("source runtime plant_sim path is not absolute")
    _digest("source runtime plant_sim module", plant_sim["module_sha256"])
    return payload


def _verify_failure(raw: object, *, name: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    payload = _mapping(name, raw)
    for key in ("phase", "exception_type", "message"):
        _text(f"{name}.{key}", payload.get(key))
    allowed = {
        "phase",
        "exception_type",
        "message",
        "solver_stages",
        "final_boundary_capture_failure",
        "unreconciled_execution",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {sorted(unknown)}")
    if "solver_stages" in payload:
        _array(f"{name}.solver_stages", payload["solver_stages"])
    if "final_boundary_capture_failure" in payload:
        _verify_failure(
            payload["final_boundary_capture_failure"],
            name=f"{name}.final_boundary_capture_failure",
        )
    return payload


def _verify_settlement(
    raw: object,
    *,
    name: str,
    cycle_index: int,
    boundary_decision: int,
    boundary_observation: CyclicMarketControllerObservationV1,
    targets: Mapping[str, float],
    config: ControllerNeutralTrainingConfigV1,
) -> dict[str, Any]:
    if config.beta_task is None:  # pragma: no cover
        raise RuntimeError("validated beta report omitted beta task")
    payload = _mapping(name, raw)
    actual = {
        resource: float(
            getattr(
                boundary_observation.plant.state.boundary,
                _RESOURCE_STATE_FIELDS[resource],
            )
        )
        for resource in RESOURCE_NAMES
    }
    tolerances = {
        resource: float(
            getattr(config.beta_task.cycle, _RESOURCE_TOLERANCE_FIELDS[resource])
        )
        for resource in RESOURCE_NAMES
    }
    deficits = {
        resource: max(0.0, float(targets[resource]) - actual[resource])
        for resource in RESOURCE_NAMES
    }
    settlement = CycleSettlementV1(
        cycle_index=cycle_index,
        boundary_decision=boundary_decision,
        targets=dict(targets),
        actual=actual,
        deficits=deficits,
        tolerances=tolerances,
        feasible=all(
            deficits[resource] <= tolerances[resource] + 1e-12
            for resource in RESOURCE_NAMES
        ),
    )
    _exact(name, payload, settlement.to_dict())
    return payload


def _verify_economic_intervals(
    raw: object,
    *,
    name: str,
    boundaries: Sequence[CyclicMarketControllerObservationV1],
    reward_trace: Sequence[float],
    settlements: Sequence[Mapping[str, Any]],
    economic_metrics: Mapping[str, Any],
    physical_metrics: Mapping[str, Any],
    config: ControllerNeutralTrainingConfigV1,
) -> None:
    intervals = _array(name, raw)
    if len(intervals) != len(reward_trace):
        raise ValueError(f"{name} cardinality differs from reward trace")
    seen_settlements: list[Mapping[str, Any]] = []
    cumulative_cost = 0.0
    aggregate_components: dict[str, list[float]] = {}
    aggregate_signals: dict[str, list[float]] = {
        key: []
        for key in (
            "accepted_solar_kwh",
            "delivered_kg",
            "unmet_kg",
            "battery_throughput_kwh",
            "electrolyser_equivalent_operating_h",
        )
    }
    assumptions = config.economic_assumptions
    initial_state = boundaries[0].plant.state.boundary
    initial_ch4_equivalent = ch4_equivalent_inventory_kg(initial_state, config.plant)
    initial_battery_energy = float(initial_state.battery_energy_kwh)
    battery_capacity = float(initial_state.battery_capacity_kwh)
    reserve_per_h = (
        config.plant.electrolyser.rated_kw
        * assumptions.electrolyser_capital_eur_per_kw
        / assumptions.electrolyser_life_h
    )
    for index, raw_interval in enumerate(intervals):
        interval = _mapping(f"{name}[{index}]", raw_interval)
        _exact_keys(
            f"{name}[{index}]",
            interval,
            frozenset({"signals", "reward", "cycle_constraint"}),
        )
        before = boundaries[index]
        after = boundaries[index + 1]
        signals = _mapping(f"{name}[{index}].signals", interval["signals"])
        reward = _mapping(f"{name}[{index}].reward", interval["reward"])
        cycle = _mapping(
            f"{name}[{index}].cycle_constraint", interval["cycle_constraint"]
        )
        _exact_keys(
            f"{name}[{index}].signals",
            signals,
            frozenset((*_ECONOMIC_SIGNAL_FIELDS, "tariff")),
        )
        if signals.get("tariff") != before.market.to_dict():
            raise ValueError(f"{name}[{index}] price issue is not the decision issue")
        if not math.isclose(
            _number(
                f"{name}[{index}] opportunity price",
                signals.get("opportunity_price_eur_per_mwh"),
            ),
            before.market.current_eur_per_mwh,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{name}[{index}] realized price differs from public issue"
            )
        reward_signals = _mapping(
            f"{name}[{index}].reward.signals", reward.get("signals")
        )
        _exact_keys(
            f"{name}[{index}].reward.signals",
            reward_signals,
            _ECONOMIC_SIGNAL_FIELDS,
        )
        if {
            key: value for key, value in signals.items() if key != "tariff"
        } != reward_signals:
            raise ValueError(f"{name}[{index}] reward and interval signals differ")
        _exact_keys(
            f"{name}[{index}].reward",
            reward,
            frozenset(
                {"objective_version", "unit", "total_eur", "components_eur", "signals"}
            ),
        )
        if (
            reward.get("objective_version") != ECONOMIC_OBJECTIVE_VERSION
            or reward.get("unit") != "EUR per decision interval"
        ):
            raise ValueError(f"{name}[{index}] economic objective version differs")

        numeric_signals = {
            key: _number(f"{name}[{index}].signal.{key}", reward_signals[key])
            for key in _ECONOMIC_SIGNAL_FIELDS
        }
        for key in aggregate_signals:
            if numeric_signals[key] < 0.0:
                raise ValueError(f"{name}[{index}] {key} must be non-negative")
            aggregate_signals[key].append(numeric_signals[key])
        before_state = before.plant.state.boundary
        after_state = after.plant.state.boundary
        methane_before = ch4_equivalent_inventory_kg(before_state, config.plant)
        methane_after = ch4_equivalent_inventory_kg(after_state, config.plant)
        methane_deficit_before = max(0.0, initial_ch4_equivalent - methane_before)
        methane_deficit_after = max(0.0, initial_ch4_equivalent - methane_after)
        battery_deficit_before = max(
            0.0, initial_battery_energy - before_state.battery_energy_kwh
        )
        battery_deficit_after = max(
            0.0, initial_battery_energy - after_state.battery_energy_kwh
        )
        effective_price = max(0.0, before.market.current_eur_per_mwh)
        signal_expectations = {
            "opportunity_price_eur_per_mwh": before.market.current_eur_per_mwh,
            "effective_opportunity_price_eur_per_mwh": effective_price,
            "battery_equivalent_full_cycles": numeric_signals["battery_throughput_kwh"]
            / (2.0 * battery_capacity),
            "methane_endowment_deficit_before_kg": methane_deficit_before,
            "methane_endowment_deficit_after_kg": methane_deficit_after,
            "battery_endowment_deficit_before_kwh": battery_deficit_before,
            "battery_endowment_deficit_after_kwh": battery_deficit_after,
            "solar_opportunity_cost_eur": numeric_signals["accepted_solar_kwh"]
            * effective_price
            / 1_000.0,
            "battery_throughput_reserve_eur": numeric_signals["battery_throughput_kwh"]
            * assumptions.battery_throughput_eur_per_kwh,
            "electrolyser_life_reserve_eur": numeric_signals[
                "electrolyser_equivalent_operating_h"
            ]
            * reserve_per_h,
        }
        for key, expected in signal_expectations.items():
            if not math.isclose(
                numeric_signals[key], expected, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(f"{name}[{index}] economic signal {key} differs")

        components = _mapping(
            f"{name}[{index}].reward.components_eur", reward.get("components_eur")
        )
        _exact_keys(
            f"{name}[{index}].reward.components_eur",
            components,
            _ECONOMIC_COMPONENT_FIELDS,
        )
        component_values = {
            key: _number(f"{name}[{index}].component.{key}", value)
            for key, value in components.items()
        }
        expected_components = {
            "delivered_service_value": numeric_signals["delivered_kg"]
            * assumptions.delivered_service_value_eur_per_kg,
            "unmet_delivery_penalty": -numeric_signals["unmet_kg"]
            * assumptions.unmet_delivery_penalty_eur_per_kg,
            "solar_opportunity_cost": -signal_expectations[
                "solar_opportunity_cost_eur"
            ],
            "battery_throughput_reserve": -signal_expectations[
                "battery_throughput_reserve_eur"
            ],
            "electrolyser_life_reserve": -signal_expectations[
                "electrolyser_life_reserve_eur"
            ],
            "methane_endowment_liability_change": -(
                methane_deficit_after - methane_deficit_before
            )
            * assumptions.methane_endowment_value_eur_per_kg_ch4_equivalent,
            "battery_endowment_liability_change": -(
                battery_deficit_after - battery_deficit_before
            )
            * assumptions.battery_endowment_value_eur_per_kwh,
        }
        for key, expected in expected_components.items():
            if not math.isclose(
                component_values[key], expected, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(f"{name}[{index}] economic component {key} differs")
        component_total = math.fsum(component_values.values())
        total = _number(f"{name}[{index}].reward.total_eur", reward.get("total_eur"))
        if not math.isclose(total, component_total, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"{name}[{index}] economic component ledger does not close"
            )
        if not math.isclose(total, reward_trace[index], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{name}[{index}] reward trace differs")
        for key, value in component_values.items():
            aggregate_components.setdefault(key, []).append(value)

        _exact_keys(
            f"{name}[{index}].cycle_constraint",
            cycle,
            frozenset(
                {
                    "scalar_reward_contains_constraint_penalty",
                    "current",
                    "settled_this_step",
                    "settled_cycle_count",
                    "all_settled_cycles_feasible",
                    "cumulative_constraint_cost",
                }
            ),
        )
        if cycle["scalar_reward_contains_constraint_penalty"] is not False:
            raise ValueError(f"{name}[{index}] hides cycle cost in scalar reward")
        if cycle["current"] != after.cycle.to_dict():
            raise ValueError(f"{name}[{index}] cycle state differs from next boundary")
        settled = cycle["settled_this_step"]
        if settled is not None:
            settled_mapping = _mapping(f"{name}[{index}].settlement", settled)
            seen_settlements.append(settled_mapping)
            cumulative_cost = math.fsum(
                (
                    cumulative_cost,
                    _number(
                        f"{name}[{index}].settlement.total_constraint_cost",
                        settled_mapping.get("total_constraint_cost"),
                    ),
                )
            )
        if _integer(
            f"{name}[{index}].settled_cycle_count", cycle["settled_cycle_count"]
        ) != len(seen_settlements):
            raise ValueError(f"{name}[{index}] settled-cycle count differs")
        if _boolean(
            f"{name}[{index}].all_settled_cycles_feasible",
            cycle["all_settled_cycles_feasible"],
        ) != all(item.get("feasible") is True for item in seen_settlements):
            raise ValueError(f"{name}[{index}] cumulative feasibility differs")
        if not math.isclose(
            _number(
                f"{name}[{index}].cumulative_constraint_cost",
                cycle["cumulative_constraint_cost"],
            ),
            cumulative_cost,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{name}[{index}] cumulative cycle cost differs")
    if list(seen_settlements) != [dict(item) for item in settlements]:
        raise ValueError(f"{name} settlement events differ from arm classification")
    if intervals:
        terminal_state = boundaries[-1].plant.state.boundary
        terminal_ch4_equivalent = ch4_equivalent_inventory_kg(
            terminal_state, config.plant
        )
        _exact_keys(
            "economic metrics",
            economic_metrics,
            frozenset(
                {
                    "objective_version",
                    "unit",
                    "decisions",
                    "net_operating_value_eur",
                    "components_eur",
                    "accepted_solar_kwh",
                    "delivered_kg",
                    "unmet_kg",
                    "battery_throughput_kwh",
                    "battery_equivalent_full_cycles",
                    "electrolyser_equivalent_operating_h",
                    "initial_ch4_equivalent_kg",
                    "terminal_ch4_equivalent_kg",
                    "terminal_ch4_equivalent_deficit_kg",
                    "initial_battery_energy_kwh",
                    "terminal_battery_energy_kwh",
                    "terminal_battery_deficit_kwh",
                }
            ),
        )
        if (
            economic_metrics.get("objective_version") != ECONOMIC_OBJECTIVE_VERSION
            or economic_metrics.get("unit") != "EUR"
            or economic_metrics.get("decisions") != len(intervals)
        ):
            raise ValueError("economic metric identity/cardinality differs")
        if not math.isclose(
            _number(
                "economic metric reward total",
                economic_metrics.get("net_operating_value_eur"),
            ),
            math.fsum(reward_trace),
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError("economic metric value differs from reward trace")
        expected_components = {
            key: float(math.fsum(values))
            for key, values in aggregate_components.items()
        }
        actual_components = _mapping(
            "economic metric components", economic_metrics.get("components_eur")
        )
        for key in set(expected_components) | set(actual_components):
            if not math.isclose(
                expected_components.get(key, 0.0),
                _number(
                    f"economic metric component {key}", actual_components.get(key, 0.0)
                ),
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise ValueError("economic metric components differ from intervals")
        expected_economic_scalars = {
            key: math.fsum(values) for key, values in aggregate_signals.items()
        }
        expected_economic_scalars.update(
            {
                "battery_equivalent_full_cycles": expected_economic_scalars[
                    "battery_throughput_kwh"
                ]
                / (2.0 * battery_capacity),
                "initial_ch4_equivalent_kg": initial_ch4_equivalent,
                "terminal_ch4_equivalent_kg": terminal_ch4_equivalent,
                "terminal_ch4_equivalent_deficit_kg": max(
                    0.0, initial_ch4_equivalent - terminal_ch4_equivalent
                ),
                "initial_battery_energy_kwh": initial_battery_energy,
                "terminal_battery_energy_kwh": terminal_state.battery_energy_kwh,
                "terminal_battery_deficit_kwh": max(
                    0.0,
                    initial_battery_energy - terminal_state.battery_energy_kwh,
                ),
            }
        )
        for key, expected in expected_economic_scalars.items():
            if not math.isclose(
                _number(f"economic metric {key}", economic_metrics.get(key)),
                expected,
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise ValueError(f"economic metric {key} differs from intervals")

        _exact_keys(
            "physical metrics",
            physical_metrics,
            frozenset(EpisodeMetricsV2().to_dict()),
        )
        physical_expected = {
            "hours": len(intervals) * config.episode.decision_interval_s / 3_600.0,
            "requested_kg": expected_economic_scalars["delivered_kg"]
            + expected_economic_scalars["unmet_kg"],
            "delivered_kg": expected_economic_scalars["delivered_kg"],
            "unmet_kg": expected_economic_scalars["unmet_kg"],
            "initial_h2_kg": initial_state.h2_level_kg,
            "initial_co2_kg": initial_state.co2_level_kg,
            "initial_ch4_kg": initial_state.ch4_level_kg,
            "initial_battery_soc": initial_state.battery_soc,
            "initial_battery_energy_kwh": initial_battery_energy,
            "terminal_h2_kg": terminal_state.h2_level_kg,
            "terminal_co2_kg": terminal_state.co2_level_kg,
            "terminal_ch4_kg": terminal_state.ch4_level_kg,
            "terminal_battery_soc": terminal_state.battery_soc,
            "terminal_battery_energy_kwh": terminal_state.battery_energy_kwh,
            "h2_inventory_change_kg": terminal_state.h2_level_kg
            - initial_state.h2_level_kg,
            "co2_inventory_change_kg": terminal_state.co2_level_kg
            - initial_state.co2_level_kg,
            "ch4_inventory_change_kg": terminal_state.ch4_level_kg
            - initial_state.ch4_level_kg,
            "battery_energy_change_kwh": terminal_state.battery_energy_kwh
            - initial_battery_energy,
            "battery_throughput_kwh": expected_economic_scalars[
                "battery_throughput_kwh"
            ],
            "accepted_process_energy_kwh": math.fsum(
                _number(f"physical metric {key}", physical_metrics.get(key))
                for key in (
                    "electrolyser_applied_energy_kwh",
                    "dac_applied_energy_kwh",
                    "methanator_applied_energy_kwh",
                )
            ),
        }
        for key, expected in physical_expected.items():
            if not math.isclose(
                _number(f"physical metric {key}", physical_metrics.get(key)),
                expected,
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise ValueError(f"physical metric {key} differs from public evidence")
        accepted_solar = _number(
            "physical metric solar_kwh", physical_metrics.get("solar_kwh")
        ) - _number(
            "physical metric curtailed_kwh", physical_metrics.get("curtailed_kwh")
        )
        if not math.isclose(
            accepted_solar,
            expected_economic_scalars["accepted_solar_kwh"],
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError("physical and economic accepted-solar totals differ")
        expected_equivalent_h = (
            _number(
                "physical operating time",
                physical_metrics.get("electrolyser_operating_time_s"),
            )
            / 3_600.0
            + _number(
                "physical low-load time",
                physical_metrics.get("electrolyser_low_load_time_s"),
            )
            / 3_600.0
            * (assumptions.electrolyser_low_load_wear_multiplier - 1.0)
            + _integer(
                "physical cold starts",
                physical_metrics.get("electrolyser_cold_starts"),
            )
            * assumptions.cold_start_equivalent_operating_h
            + _integer(
                "physical hot starts",
                physical_metrics.get("electrolyser_hot_starts"),
            )
            * assumptions.hot_start_equivalent_operating_h
            + _integer(
                "physical aborted starts",
                physical_metrics.get("electrolyser_aborted_starts"),
            )
            * assumptions.aborted_start_equivalent_operating_h
            + _integer(
                "physical forced stops",
                physical_metrics.get("electrolyser_forced_stops"),
            )
            * assumptions.forced_stop_equivalent_operating_h
        )
        if not math.isclose(
            expected_equivalent_h,
            expected_economic_scalars["electrolyser_equivalent_operating_h"],
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError(
                "physical and economic electrolyser exposure totals differ"
            )
        delivery = boundaries[-1].plant.delivery
        if not math.isclose(
            delivery.cumulative_requested_kg,
            expected_economic_scalars["delivered_kg"]
            + expected_economic_scalars["unmet_kg"],
            rel_tol=0.0,
            abs_tol=1e-8,
        ) or not math.isclose(
            delivery.cumulative_delivered_kg,
            expected_economic_scalars["delivered_kg"],
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError(
                "final public delivery ledger differs from interval totals"
            )


def _cycle_targets_from_manifest(
    manifest: Mapping[str, Any],
    config: ControllerNeutralTrainingConfigV1,
    *,
    initial_observation: CyclicMarketControllerObservationV1,
) -> dict[str, float]:
    if config.beta_task is None:  # pragma: no cover
        raise RuntimeError("validated beta report omitted beta task")
    reference = _mapping("cycle opening reference", manifest.get("cycle_reference"))
    expected_reference = CycleReferenceV1.from_state(
        initial_observation.plant.state.boundary
    )
    _exact("cycle opening reference", reference, expected_reference.to_dict())
    source_fields = {
        "battery": "battery_energy_kwh",
        "h2": "h2_kg",
        "co2": "co2_kg",
        "ch4": "ch4_kg",
    }
    return {
        resource: _number(
            f"cycle opening reference {resource}",
            reference.get(source_fields[resource]),
        )
        * float(getattr(config.beta_task.cycle, _RESOURCE_RETURN_FIELDS[resource]))
        for resource in RESOURCE_NAMES
    }


def _decode_mpc_delivery(raw: object, *, name: str) -> MPCDeliveryTargetV1:
    payload = _mapping(name, raw)
    delivery = MPCDeliveryTargetV1(
        ordinal=_integer(f"{name}.ordinal", payload.get("ordinal")),
        due_boundary=_integer(
            f"{name}.due_boundary", payload.get("due_boundary"), minimum=1
        ),
        due_offset_s=_number(
            f"{name}.due_offset_s", payload.get("due_offset_s"), minimum=0.0
        ),
        requested_kg=_number(
            f"{name}.requested_kg", payload.get("requested_kg"), minimum=0.0
        ),
        source=_text(f"{name}.source", payload.get("source")),
    )
    _exact(name, payload, delivery.to_dict())
    return delivery


def _decode_mpc_plan_step(raw: object, *, name: str) -> MPCPlanStepV1:
    payload = _mapping(name, raw)
    boolean_fields = {"cold_start", "warmup", "running", "battery_charging"}
    values: dict[str, Any] = {}
    for field in fields(MPCPlanStepV1):
        if field.name not in payload:
            raise ValueError(f"{name} is missing {field.name!r}")
        if field.name == "index":
            values[field.name] = _integer(f"{name}.{field.name}", payload[field.name])
        elif field.name in boolean_fields:
            values[field.name] = _boolean(f"{name}.{field.name}", payload[field.name])
        else:
            values[field.name] = _number(
                f"{name}.{field.name}", payload[field.name], minimum=0.0
            )
    step = MPCPlanStepV1(**values)
    _exact(name, payload, step.to_dict())
    return step


def _beta_plan_solution_witness(
    *,
    model: Any,
    plan: BetaMPCPlanV1,
    config: ControllerNeutralTrainingConfigV1,
    selected_solution: Sequence[float],
) -> tuple[float, ...]:
    """Bind every projected plan field to the exact retained stage-two vector."""

    values = tuple(float(value) for value in selected_solution)
    if len(values) != model.variable_count:
        raise ValueError("E-beta selected solution cardinality differs from its model")

    def require_group(group_name: str, group_values: Sequence[float | bool]) -> None:
        indices = model.group(group_name)
        if len(indices) != len(group_values):
            raise ValueError(f"E-beta plan group {group_name} cardinality differs")
        for index, expected in zip(indices, group_values, strict=True):
            actual = values[index]
            if not math.isclose(
                actual,
                float(expected),
                rel_tol=0.0,
                abs_tol=MODEL_BALANCE_TOLERANCE,
            ):
                raise ValueError(
                    f"E-beta plan group {group_name} differs from selected solution"
                )

    direct_groups = (
        "solar_to_battery_kw",
        "solar_to_electrolyser_kw",
        "solar_to_dac_kw",
        "solar_to_methanator_kw",
        "solar_curtailed_kw",
        "battery_support_kw",
        "electrolyser_startup_kw",
        "electrolyser_productive_kw",
        "h2_produced_kg",
        "co2_captured_kg",
        "ch4_produced_kg",
        "delivered_kg",
        "cold_start",
        "warmup",
        "running",
        "battery_charging",
    )
    for group_name in direct_groups:
        require_group(group_name, [getattr(step, group_name) for step in plan.steps])

    state = model.physical_model.metadata()["initial_state"]
    if not isinstance(state, Mapping):  # pragma: no cover - model validates metadata
        raise TypeError("E-beta model initial state is malformed")
    for group_name, initial_key, step_field in (
        ("battery_energy_kwh", "battery_energy_kwh", "battery_energy_end_kwh"),
        ("h2_inventory_kg", "h2_kg", "h2_end_kg"),
        ("co2_inventory_kg", "co2_kg", "co2_end_kg"),
        ("ch4_inventory_kg", "ch4_kg", "ch4_end_kg"),
    ):
        require_group(
            group_name,
            [
                float(state[initial_key]),
                *[getattr(step, step_field) for step in plan.steps],
            ],
        )

    on = [float(step.warmup or step.running) for step in plan.steps]
    require_group("on", on)
    constants = model.physical_model.metadata()["constants"]
    if not isinstance(constants, Mapping):  # pragma: no cover
        raise TypeError("E-beta model constants are malformed")
    prior_on = float(constants["initial_on"])
    stops: list[float] = []
    for index, step in enumerate(plan.steps):
        stop = prior_on + float(step.cold_start) - on[index]
        if not math.isclose(stop, round(stop), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("E-beta plan implies a non-binary stop transition")
        stops.append(float(round(stop)))
        prior_on = on[index]
    require_group("stop", stops)

    if plan.deliveries:
        maximum_minimum_fill = min(
            plan.steps[item.due_boundary - 1].delivered_kg / item.requested_kg
            for item in plan.deliveries
        )
    else:
        maximum_minimum_fill = 1.0
    minimum_fill = values[model.group("minimum_commitment_fill")[0]]
    if minimum_fill > maximum_minimum_fill + MODEL_BALANCE_TOLERANCE:
        raise ValueError("E-beta selected minimum-fill witness exceeds delivery")
    terminal = plan.steps[-1]
    feedstock = values[model.group("terminal_feedstock_ch4_equivalent_kg")[0]]
    expected_ch4_deficit = max(
        0.0,
        model.economic_model.initial_ch4_equivalent_kg
        - terminal.ch4_end_kg
        - feedstock,
    )
    expected_battery_deficit = max(
        0.0,
        model.economic_model.initial_battery_energy_kwh
        - terminal.battery_energy_end_kwh,
    )
    selected_ch4_deficit = values[model.group("terminal_ch4_equivalent_deficit_kg")[0]]
    selected_battery_deficit = values[model.group("terminal_battery_deficit_kwh")[0]]
    if (
        not math.isclose(
            selected_ch4_deficit,
            expected_ch4_deficit,
            rel_tol=0.0,
            abs_tol=MODEL_BALANCE_TOLERANCE,
        )
        or not math.isclose(
            selected_battery_deficit,
            expected_battery_deficit,
            rel_tol=0.0,
            abs_tol=MODEL_BALANCE_TOLERANCE,
        )
        or not math.isclose(
            plan.terminal_ch4_equivalent_deficit_kg,
            selected_ch4_deficit,
            rel_tol=0.0,
            abs_tol=MODEL_BALANCE_TOLERANCE,
        )
        or not math.isclose(
            plan.terminal_battery_deficit_kwh,
            selected_battery_deficit,
            rel_tol=0.0,
            abs_tol=MODEL_BALANCE_TOLERANCE,
        )
    ):
        raise ValueError("E-beta plan terminal endowment hinges are not exact")
    return values


def _verify_beta_plan(
    raw: object,
    *,
    boundary: CyclicMarketControllerObservationV1,
    settings: BetaMPCSettingsV1,
    config: ControllerNeutralTrainingConfigV1,
    initial_ch4_equivalent_kg: float,
    initial_battery_energy_kwh: float,
    expected_scipy_version: str,
    highs_versions: set[str],
) -> tuple[dict[str, Any], ProcessSetpointTargetsV1]:
    if config.beta_task is None:  # pragma: no cover
        raise RuntimeError("validated beta report omitted beta task")
    plan = _mapping("E-beta plan", raw)
    price_projection_typed = project_beta_mpc_prices(
        boundary.market,
        interval_s=settings.interval_s,
        horizon_intervals=boundary.cycle.cycle_decisions
        - boundary.cycle.decision_in_cycle,
    )
    price_projection = price_projection_typed.to_dict()
    _exact("E-beta price projection", plan.get("price_projection"), price_projection)
    cycle_resources: list[BetaMPCResourceReturnV1] = []
    physical = boundary.plant.state.boundary
    cycle_config = config.beta_task.cycle
    for resource in RESOURCE_NAMES:
        current = float(getattr(physical, _RESOURCE_STATE_FIELDS[resource]))
        margin = float(getattr(boundary.cycle, _RESOURCE_MARGIN_FIELDS[resource]))
        target = current - margin
        tolerance = float(getattr(cycle_config, _RESOURCE_TOLERANCE_FIELDS[resource]))
        cycle_resources.append(
            BetaMPCResourceReturnV1(
                resource=resource,
                unit=_RESOURCE_UNITS[resource],
                current=current,
                signed_margin=margin,
                target=target,
                tolerance=tolerance,
                required_minimum=max(0.0, target - tolerance),
            )
        )
    cycle_return = BetaMPCCycleReturnV1(
        cycle_index=boundary.cycle.cycle_index,
        decision_in_cycle=boundary.cycle.decision_in_cycle,
        cycle_decisions=boundary.cycle.cycle_decisions,
        issued_at_decision=boundary.market.issued_at_decision,
        next_boundary_decision=(
            boundary.market.issued_at_decision
            + boundary.cycle.cycle_decisions
            - boundary.cycle.decision_in_cycle
        ),
        remaining_intervals=(
            boundary.cycle.cycle_decisions - boundary.cycle.decision_in_cycle
        ),
        resources=cast(Any, tuple(cycle_resources)),
    )
    derived_cycle_return = derive_beta_mpc_cycle_return(
        boundary,
        settings=settings,
        beta_task=config.beta_task,
        config=config.plant,
    )
    if cycle_return != derived_cycle_return:
        raise ValueError("E-beta cycle-return derivations disagree")
    cycle_return_dict = cycle_return.to_dict()
    _exact("E-beta cycle return", plan.get("cycle_return"), cycle_return_dict)
    if plan.get("market_issue_sha256") != boundary.market.sha256:
        raise ValueError("E-beta plan market issue differs from boundary")
    if plan.get("cycle_observation_sha256") != domain_sha256(
        BETA_MPC_CYCLE_OBSERVATION_DOMAIN, boundary.cycle.to_dict()
    ):
        raise ValueError("E-beta plan cycle observation differs from boundary")
    if plan.get("settings_sha256") != settings.sha256:
        raise ValueError("E-beta plan settings differ")
    if plan.get("assumptions_sha256") != config.economic_assumptions.sha256:
        raise ValueError("E-beta plan assumptions differ")
    if plan.get("beta_task_sha256") != config.beta_task.sha256:
        raise ValueError("E-beta plan task differs")
    horizon = _integer("E-beta plan horizon", plan.get("horizon_intervals"), minimum=1)
    if horizon != cycle_return_dict["remaining_intervals"]:
        raise ValueError("E-beta plan does not end at the next cycle boundary")
    forecast = boundary.plant.forecast
    if forecast is None:  # pragma: no cover - beta observation decoder requires it
        raise ValueError("E-beta boundary omitted its public weather issue")
    weather_projection = project_issued_mean_solar(
        forecast,
        boundary_time_s=boundary.plant.state.boundary.time_s,
        config=config.plant,
        interval_s=settings.interval_s,
        energy_tolerance=settings.energy_reconstruction_tolerance,
    )
    if plan.get("weather_projection_sha256") != weather_projection.sha256:
        raise ValueError("E-beta weather projection differs from public forecast")
    model = build_beta_mpc_model(
        boundary,
        weather_projection,
        price_projection_typed,
        settings=settings,
        assumptions=config.economic_assumptions,
        beta_task=config.beta_task,
        constraints=config.constraints,
        config=config.plant,
        commitment_profile=config.episode.electrolyser_commitment,
        initial_ch4_equivalent_kg=initial_ch4_equivalent_kg,
        initial_battery_energy_kwh=initial_battery_energy_kwh,
    )
    if plan.get("model_sha256") != model.sha256:
        raise ValueError("E-beta model digest differs from public inputs")
    if plan.get("forecast_sha256") != weather_projection.sha256:
        raise ValueError("E-beta plan forecast digest differs")
    _exact(
        "E-beta deliveries",
        plan.get("deliveries"),
        [item.to_dict() for item in model.deliveries],
    )
    steps = _array("E-beta plan steps", plan.get("steps"))
    if len(steps) != horizon:
        raise ValueError("E-beta plan step cardinality differs")
    decoded_steps: list[MPCPlanStepV1] = []
    for index, raw_step in enumerate(steps):
        decoded_step = _decode_mpc_plan_step(raw_step, name=f"E-beta plan step {index}")
        if decoded_step.index != index:
            raise ValueError("E-beta plan steps are reordered")
        if not math.isclose(
            decoded_step.forecast_solar_kw,
            weather_projection.available_kw[index],
            rel_tol=0.0,
            abs_tol=MODEL_BALANCE_TOLERANCE,
        ):
            raise ValueError("E-beta plan solar path differs from public projection")
        decoded_steps.append(decoded_step)
    first = decoded_steps[0]
    targets = ProcessSetpointTargetsV1(
        electrolyser_target_kw=(
            first.electrolyser_startup_kw + first.electrolyser_productive_kw
        ),
        dac_target_kw=first.solar_to_dac_kw,
        methanator_target_kw=first.solar_to_methanator_kw,
        battery_charge_ceiling_kw=first.solar_to_battery_kw,
    )
    terminal = decoded_steps[-1]
    terminal_fields = {
        "battery": "battery_energy_end_kwh",
        "h2": "h2_end_kg",
        "co2": "co2_end_kg",
        "ch4": "ch4_end_kg",
    }
    for resource in RESOURCE_NAMES:
        required = cycle_return.resource(resource).required_minimum
        actual = float(getattr(terminal, terminal_fields[resource]))
        if actual < required - MODEL_BALANCE_TOLERANCE:
            raise ValueError(f"E-beta plan violates hard {resource} return")
    stages = _array("E-beta plan stages", plan.get("stages"))
    if len(stages) != 2:
        raise ValueError("E-beta plan requires two solver stages")
    expected_stage_names = ("service_maximization", "economic_cost_minimization")
    decoded_stages: list[EconomicMPCStageSolveV1] = []
    for index, raw_stage in enumerate(stages):
        stage = _mapping(f"E-beta stage {index}", raw_stage)
        decoded = EconomicMPCStageSolveV1(
            stage=_text(f"E-beta stage {index}.stage", stage.get("stage")),
            success=_boolean(f"E-beta stage {index}.success", stage.get("success")),
            status=_integer(f"E-beta stage {index}.status", stage.get("status")),
            message=_text(f"E-beta stage {index}.message", stage.get("message")),
            elapsed_ns=_integer(
                f"E-beta stage {index}.elapsed_ns", stage.get("elapsed_ns")
            ),
            objective_value=(
                None
                if stage.get("objective_value") is None
                else _number(
                    f"E-beta stage {index}.objective_value",
                    stage.get("objective_value"),
                )
            ),
            mip_gap=(
                None
                if stage.get("mip_gap") is None
                else _number(f"E-beta stage {index}.mip_gap", stage.get("mip_gap"))
            ),
            mip_node_count=(
                None
                if stage.get("mip_node_count") is None
                else _integer(
                    f"E-beta stage {index}.mip_node_count",
                    stage.get("mip_node_count"),
                )
            ),
            solution_sha256=(
                None
                if stage.get("solution_sha256") is None
                else _digest(
                    f"E-beta stage {index}.solution_sha256",
                    stage.get("solution_sha256"),
                )
            ),
            scipy_version=_text(
                f"E-beta stage {index}.scipy_version", stage.get("scipy_version")
            ),
            highs_version=_text(
                f"E-beta stage {index}.highs_version", stage.get("highs_version")
            ),
        )
        if decoded.stage != expected_stage_names[index] or not decoded.success:
            raise ValueError(
                "E-beta stages are not ordered successful lexicographic solves"
            )
        if decoded.scipy_version != expected_scipy_version:
            raise ValueError("E-beta stage SciPy version differs from provenance")
        highs_versions.add(decoded.highs_version)
        _exact(f"E-beta stage {index}", stage, decoded.to_dict())
        decoded_stages.append(decoded)
    delivered = math.fsum(
        _number(
            f"E-beta step {index} delivered", _mapping("step", item).get("delivered_kg")
        )
        for index, item in enumerate(steps)
    )
    if not math.isclose(
        delivered,
        _number("E-beta selected delivery", plan.get("selected_delivered_kg")),
        rel_tol=0.0,
        abs_tol=MODEL_BALANCE_TOLERANCE,
    ):
        raise ValueError("E-beta selected delivery differs from its steps")
    components = _mapping(
        "E-beta predicted cost components", plan.get("predicted_cost_components_eur")
    )
    predicted = math.fsum(
        _number(f"E-beta predicted component {key}", value, minimum=0.0)
        for key, value in components.items()
    )
    if not math.isclose(
        predicted,
        _number(
            "E-beta predicted economic cost", plan.get("predicted_economic_cost_eur")
        ),
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError("E-beta predicted cost ledger does not close")
    decoded_deliveries = tuple(
        _decode_mpc_delivery(item, name=f"E-beta delivery {index}")
        for index, item in enumerate(
            _array("E-beta deliveries", plan.get("deliveries"))
        )
    )
    selected_solution = tuple(
        _number(f"E-beta selected solution[{index}]", value)
        for index, value in enumerate(
            _array(
                "E-beta selected solution vector",
                plan.get("selected_solution_vector"),
            )
        )
    )
    if len(selected_solution) != model.variable_count:
        raise ValueError("E-beta selected solution cardinality differs from its model")
    typed_plan = BetaMPCPlanV1(
        model_sha256=_digest("E-beta plan model", plan.get("model_sha256")),
        forecast_sha256=_digest("E-beta plan forecast", plan.get("forecast_sha256")),
        settings_sha256=_digest("E-beta plan settings", plan.get("settings_sha256")),
        assumptions_sha256=_digest(
            "E-beta plan assumptions", plan.get("assumptions_sha256")
        ),
        horizon_intervals=horizon,
        deliveries=decoded_deliveries,
        stages=cast(Any, tuple(decoded_stages)),
        steps=tuple(decoded_steps),
        primary_delivered_optimum_kg=_number(
            "E-beta primary delivery",
            plan.get("primary_delivered_optimum_kg"),
            minimum=0.0,
        ),
        selected_delivered_kg=_number(
            "E-beta selected delivery", plan.get("selected_delivered_kg"), minimum=0.0
        ),
        delivery_lock_tolerance_kg=_number(
            "E-beta delivery lock", plan.get("delivery_lock_tolerance_kg"), minimum=0.0
        ),
        predicted_cost_components_eur=tuple(
            (
                _text("E-beta predicted component name", key),
                _number(f"E-beta predicted component {key}", value, minimum=0.0),
            )
            for key, value in components.items()
        ),
        terminal_ch4_equivalent_deficit_kg=_number(
            "E-beta terminal methane deficit",
            plan.get("terminal_ch4_equivalent_deficit_kg"),
            minimum=0.0,
        ),
        terminal_battery_deficit_kwh=_number(
            "E-beta terminal battery deficit",
            plan.get("terminal_battery_deficit_kwh"),
            minimum=0.0,
        ),
        price_projection=price_projection_typed,
        market_issue_sha256=boundary.market.sha256,
        cycle_observation_sha256=domain_sha256(
            BETA_MPC_CYCLE_OBSERVATION_DOMAIN, boundary.cycle.to_dict()
        ),
        beta_task_sha256=config.beta_task.sha256,
        cycle_return=cycle_return,
        selected_solution_vector=selected_solution,
    )
    _exact("E-beta typed plan", plan, typed_plan.to_dict())
    witness = _beta_plan_solution_witness(
        model=model,
        plan=typed_plan,
        config=config,
        selected_solution=selected_solution,
    )
    lock = economic_delivery_lock_row(
        model,
        delivered_optimum_kg=typed_plan.primary_delivered_optimum_kg,
        tolerance_kg=typed_plan.delivery_lock_tolerance_kg,
    )
    expected_solution_sha256 = domain_sha256(
        BETA_MPC_SOLUTION_DOMAIN,
        {
            "model_sha256": model.sha256,
            "stage": "economic_cost_minimization",
            "extra_rows": [lock.to_dict()],
            "values": list(witness),
        },
    )
    if decoded_stages[1].solution_sha256 != expected_solution_sha256:
        raise ValueError(
            "E-beta stage-two solution digest differs from retained vector"
        )
    audit = model.audit_solution(
        witness,
        tolerance=2.0 * settings.solver_mip_feasibility_tolerance,
        extra_rows=(lock,),
    )
    if not audit.valid:
        raise ValueError(
            "E-beta selected plan fails its reconstructed model audit: "
            f"{audit.worst_row or 'bounds/integrality'}"
        )
    expected_costs = dict(model.predicted_cost_components(witness))
    if set(expected_costs) != set(components):
        raise ValueError("E-beta predicted cost component schema differs")
    for key, expected in expected_costs.items():
        if not math.isclose(
            _number(f"E-beta predicted component {key}", components[key]),
            expected,
            rel_tol=0.0,
            abs_tol=max(1e-7, 1e-9 * max(1.0, abs(expected))),
        ):
            raise ValueError(f"E-beta predicted component {key} differs from model")
    expected_objective = math.fsum(
        coefficient * value
        for coefficient, value in zip(model.economic_objective, witness, strict=True)
    )
    if not math.isclose(
        cast(float, decoded_stages[1].objective_value),
        expected_objective,
        rel_tol=0.0,
        abs_tol=max(1e-7, 1e-9 * max(1.0, abs(expected_objective))),
    ):
        raise ValueError("E-beta stage-two objective differs from selected plan")
    return plan, targets


def _verify_generic_decision(
    raw: object,
    *,
    arm: str,
    boundary_payload: Mapping[str, Any],
    boundary: CyclicMarketControllerObservationV1,
    config: ControllerNeutralTrainingConfigV1,
    checkpoint: Mapping[str, Any],
    beta_settings: BetaMPCSettingsV1 | None,
    policy_metadata_sha: str | None,
    policy_contract_sha: str | None,
    initial_ch4_equivalent_kg: float,
    initial_battery_energy_kwh: float,
    expected_scipy_version: str,
    highs_versions: set[str],
) -> ProcessSetpointCommandV1:
    name = f"{arm} decision {boundary_payload['decision_index']}"
    payload = _mapping(name, raw)
    expected_fields = (
        frozenset({"evidence", "command"})
        if arm == BETA_ECONOMIC_MPC_ARM
        else frozenset({"evidence", "command", "invalid"})
    )
    _exact_keys(name, payload, expected_fields)
    evidence = _mapping(f"{name}.evidence", payload["evidence"])
    command = _decode_command(payload["command"], name=f"{name}.command")
    index = int(boundary_payload["decision_index"])
    if command.decision_index != index:
        raise ValueError(f"{name} command decision index differs")
    if command.observation_sha256 != boundary_payload["physical_observation_sha256"]:
        raise ValueError(f"{name} command physical observation differs")
    if not math.isclose(
        command.hold_duration_s,
        config.episode.decision_interval_s,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{name} command hold differs from task cadence")
    expected_envelope = resolve_process_setpoint_envelope(
        constraints=config.constraints, config=config.plant
    )
    if command.envelope != expected_envelope:
        raise ValueError(f"{name} command envelope differs from task authority")

    if arm == HEURISTIC_ARM:
        if command.issuer_id != LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID:
            raise ValueError(f"{name} command issuer differs")
        source = _mapping(f"{name}.source_decision", evidence.get("source_decision"))
        heuristic_evidence = LegacyHeuristicSetpointEvidenceV1(
            decision_index=_integer(
                f"{name}.evidence.decision_index", evidence.get("decision_index")
            ),
            observation_sha256=_digest(
                f"{name}.evidence.observation_sha256",
                evidence.get("observation_sha256"),
            ),
            boundary_solar_kw=_number(
                f"{name}.evidence.boundary_solar_kw", evidence.get("boundary_solar_kw")
            ),
            source_action=_decode_action(
                source.get("action"), name=f"{name}.source_action"
            ),
            source_notes=tuple(
                _text(f"{name}.source_notes[{i}]", item)
                for i, item in enumerate(
                    _array(f"{name}.source_notes", source.get("notes"))
                )
            ),
            source_invalid=_boolean(f"{name}.source_invalid", source.get("invalid")),
            requested_targets=_decode_targets(
                evidence.get("requested_targets"), name=f"{name}.requested_targets"
            ),
            issued_targets=_decode_targets(
                evidence.get("issued_targets"), name=f"{name}.issued_targets"
            ),
            battery_charge_limited=_boolean(
                f"{name}.battery_charge_limited", evidence.get("battery_charge_limited")
            ),
        )
        _exact(f"{name}.evidence", evidence, heuristic_evidence.to_dict())
        if (
            heuristic_evidence.observation_sha256
            != boundary_payload["physical_observation_sha256"]
        ):
            raise ValueError(f"{name} heuristic evidence observation differs")
        if (
            heuristic_evidence.boundary_solar_kw
            != boundary.plant.state.boundary.solar_available_kw
        ):
            raise ValueError(f"{name} heuristic solar differs from boundary")
        if payload["invalid"] != heuristic_evidence.source_invalid:
            raise ValueError(f"{name} invalid flag differs")
        expected_source_schema = LEGACY_HEURISTIC_SETPOINT_EVIDENCE_SCHEMA
        expected_source_sha = heuristic_evidence.sha256
        expected_targets = heuristic_evidence.issued_targets
    elif arm == PPO_ARM:
        decoded_raw = _mapping(f"{name}.decoded_action", evidence.get("decoded_action"))
        raw_vector = _array(
            f"{name}.raw_policy_vector", decoded_raw.get("raw_policy_vector")
        )
        decoded = ProcessSetpointActionAdapterV1().decode(
            [
                _number(f"{name}.raw_policy_vector[{i}]", value)
                for i, value in enumerate(raw_vector)
            ],
            constraints=config.constraints,
            config=config.plant,
        )
        _exact(f"{name}.decoded_action", decoded_raw, decoded.to_dict())
        if policy_metadata_sha is None or policy_contract_sha is None:
            raise RuntimeError("PPO descriptor was not admitted before its decisions")
        policy_evidence = PolicySetpointDecisionEvidenceV1(
            decision_index=_integer(
                f"{name}.evidence.decision_index", evidence.get("decision_index")
            ),
            observation_sha256=_digest(
                f"{name}.evidence.observation_sha256",
                evidence.get("observation_sha256"),
            ),
            policy_observation_sha256=_digest(
                f"{name}.evidence.policy_observation_sha256",
                evidence.get("policy_observation_sha256"),
            ),
            economic_observation_sha256=_digest(
                f"{name}.evidence.economic_observation_sha256",
                evidence.get("economic_observation_sha256"),
            ),
            checkpoint_metadata_sha256=_digest(
                f"{name}.evidence.checkpoint_metadata_sha256",
                evidence.get("checkpoint_metadata_sha256"),
            ),
            policy_contract_sha256=_digest(
                f"{name}.evidence.policy_contract_sha256",
                evidence.get("policy_contract_sha256"),
            ),
            decoded_action=decoded,
        )
        _exact(f"{name}.evidence", evidence, policy_evidence.to_dict())
        if (
            policy_evidence.observation_sha256
            != boundary_payload["physical_observation_sha256"]
            or policy_evidence.policy_observation_sha256
            != boundary_payload["normalized_observation_sha256"]
            or policy_evidence.economic_observation_sha256
            != boundary_payload["typed_observation_sha256"]
            or policy_evidence.checkpoint_metadata_sha256 != policy_metadata_sha
            or policy_evidence.policy_contract_sha256 != policy_contract_sha
        ):
            raise ValueError(
                f"{name} PPO evidence chain differs from boundary/checkpoint"
            )
        if payload["invalid"] != decoded.clipped:
            raise ValueError(f"{name} invalid flag differs")
        expected_source_schema = NORMALIZED_POLICY_SETPOINT_EVIDENCE_SCHEMA
        expected_source_sha = policy_evidence.sha256
        expected_targets = decoded.targets
    else:
        if beta_settings is None:
            raise RuntimeError("E-beta settings were not admitted before decisions")
        plan, expected_targets = _verify_beta_plan(
            evidence.get("plan"),
            boundary=boundary,
            settings=beta_settings,
            config=config,
            initial_ch4_equivalent_kg=initial_ch4_equivalent_kg,
            initial_battery_energy_kwh=initial_battery_energy_kwh,
            expected_scipy_version=expected_scipy_version,
            highs_versions=highs_versions,
        )
        if evidence.get("schema") != BETA_MPC_DECISION_SCHEMA:
            raise ValueError(f"{name} evidence schema differs")
        expected_links = {
            "controller_id": BETA_MPC_CONTROLLER_ID,
            "decision_index": index,
            "beta_observation_sha256": boundary_payload["typed_observation_sha256"],
            "physical_observation_sha256": boundary_payload[
                "physical_observation_sha256"
            ],
            "market_issue_sha256": boundary_payload["market_issue_sha256"],
            "cycle_observation_sha256": boundary_payload["cycle_observation_sha256"],
            "settings_sha256": beta_settings.sha256,
            "assumptions_sha256": config.economic_assumptions.sha256,
            "beta_task_sha256": config.beta_task.sha256 if config.beta_task else None,
            "model_sha256": plan.get("model_sha256"),
            "plan_sha256": plan.get("sha256"),
            "price_projection_sha256": _mapping(
                "E-beta price projection", plan["price_projection"]
            ).get("sha256"),
            "cycle_return_sha256": _mapping(
                "E-beta cycle return", plan["cycle_return"]
            ).get("sha256"),
            "market_id": boundary.market.market_id,
        }
        for key, expected in expected_links.items():
            if evidence.get(key) != expected:
                raise ValueError(f"{name} E-beta {key} differs")
        if evidence.get("weather_projection_sha256") != plan.get(
            "weather_projection_sha256"
        ):
            raise ValueError(f"{name} E-beta weather projection differs")
        evidence_targets = _decode_targets(
            evidence.get("targets"), name=f"{name}.evidence.targets"
        )
        if evidence_targets != expected_targets:
            raise ValueError(f"{name} E-beta targets differ from first plan step")
        supplied = _digest(f"{name}.evidence.sha256", evidence.get("sha256"))
        body = dict(evidence)
        del body["sha256"]
        body["plan"] = {"sha256": evidence.get("plan_sha256")}
        if domain_sha256(BETA_MPC_DECISION_DOMAIN, body) != supplied:
            raise ValueError(f"{name} E-beta decision digest differs")
        expected_source_schema = BETA_MPC_DECISION_SCHEMA
        expected_source_sha = supplied
    if (
        command.source_schema != expected_source_schema
        or command.source_sha256 != expected_source_sha
        or command.targets != expected_targets
    ):
        raise ValueError(f"{name} evidence-to-command chain differs")
    return command


def _derived_plant_status(
    state: PlantState,
    *,
    config: ControllerNeutralTrainingConfigV1,
) -> str:
    """Mirror the simulator's public, state-derived plant-status semantics."""

    process_units_offline = all(
        not online
        for online in (
            state.electrolyser_online,
            state.dac_online,
            state.methanator_online,
        )
    )
    battery_reserve_kwh = max(
        0.0,
        state.battery_energy_kwh
        - float(config.plant.battery.min_soc) * state.battery_capacity_kwh,
    )
    if (
        process_units_offline
        and state.solar_available_kw <= 0.0
        and battery_reserve_kwh <= 0.0
    ):
        return OFFLINE
    if any(
        state.health[unit].offline for unit in _PLANT_STATUS_UNITS
    ) or state.battery_soc <= float(config.plant.battery.min_soc):
        return CRITICAL
    if any(state.health[unit].degradation > 0.0 for unit in _PLANT_STATUS_UNITS):
        return DEGRADED
    feedstock_starved = bool(
        not state.methanator_online
        and state.action.methanator_share > 0.0
        and (
            state.h2_level_kg < _EMPTY_FEEDSTOCK_KG
            or state.co2_level_kg < _EMPTY_FEEDSTOCK_KG
        )
    )
    return DEGRADED if feedstock_starved else NORMAL


def _decode_minute_truth_states(
    raw: object,
    *,
    name: str,
    config: ControllerNeutralTrainingConfigV1,
    physical_steps: int,
) -> list[PlantState]:
    """Decode the exhaustive run-wide truth ledger and its local invariants."""

    records = _array(name, raw)
    if len(records) != physical_steps + 1:
        raise ValueError(f"{name} does not cover every executed physics step")
    states: list[PlantState] = []
    for expected_step, record in enumerate(records):
        state = _decode_plant_state(record, name=f"{name}[{expected_step}]")
        if state.step != expected_step:
            raise ValueError(f"{name} is not contiguous from episode step zero")
        expected_time_s = (
            config.episode.start_hour * 3_600.0 + expected_step * config.plant.dt_s
        )
        if not math.isclose(state.time_s, expected_time_s, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{name}[{expected_step}] is off the physics clock")
        if frozenset(state.health) != _PLANT_HEALTH_UNITS:
            raise ValueError(f"{name}[{expected_step}] has incomplete component health")
        for unit, health in state.health.items():
            if not 0.0 <= health.degradation <= 1.0:
                raise ValueError(
                    f"{name}[{expected_step}].health.{unit}.degradation "
                    "lies outside [0, 1]"
                )
            expected_health_status = (
                "offline"
                if health.offline
                else "degraded" if health.degradation > 0.0 else "nominal"
            )
            if health.status != expected_health_status:
                raise ValueError(
                    f"{name}[{expected_step}].health.{unit}.status differs "
                    "from component condition"
                )
        if state.battery_capacity_kwh <= 0.0 or not math.isclose(
            state.battery_soc,
            state.battery_energy_kwh / state.battery_capacity_kwh,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{name}[{expected_step}] battery state is inconsistent")
        if state.plant_status != _derived_plant_status(state, config=config):
            raise ValueError(
                f"{name}[{expected_step}].plant_status differs from plant condition"
            )
        states.append(state)
    return states


def _forced_stops_from_public_commitment(
    initial: CyclicMarketControllerObservationV1,
    final: CyclicMarketControllerObservationV1,
) -> int:
    before = initial.plant.state.electrolyser_commitment
    after = final.plant.state.electrolyser_commitment
    if before is None and after is None:
        return 0
    if before is None or after is None:
        raise ValueError("public commitment evidence changed enabledness")
    delta = after.forced_stops - before.forced_stops
    if delta < 0:
        raise ValueError("public forced-stop counter moved backwards")
    return int(delta)


def _declared_dac_outage_lattice(
    manifest: Mapping[str, Any],
    *,
    config: ControllerNeutralTrainingConfigV1,
    executed_steps: int,
) -> frozenset[int]:
    """Rebuild the executed portion of the private, half-open outage schedule."""

    disturbances = _mapping("beta disturbance manifest", manifest.get("disturbances"))
    manifest_dt_s = _number(
        "beta disturbance physics_dt_s", disturbances.get("physics_dt_s")
    )
    dt_s = float(config.plant.dt_s)
    if not math.isclose(manifest_dt_s, dt_s, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("beta disturbance schedule uses another physics lattice")
    schedule = _array("beta disturbance schedule", disturbances.get("schedule"))
    if _integer("beta disturbance fault_count", disturbances.get("fault_count")) != len(
        schedule
    ):
        raise ValueError("beta disturbance fault count differs from its schedule")

    declared_steps: set[int] = set()
    previous_start_step = -1
    occupied_cycles: set[int] = set()
    if config.beta_task is None:  # pragma: no cover - report config closes this
        raise RuntimeError("validated beta report omitted beta task")
    cycle_steps = int(round(config.beta_task.cycle.cycle_h * 3_600.0 / dt_s))
    cycle_count = config.episode.decision_steps(
        config.plant
    ) // config.beta_task.cycle.cycle_decisions(config.episode)
    for index, raw_fault in enumerate(schedule):
        fault = _mapping(f"beta disturbance schedule[{index}]", raw_fault)
        _exact_keys(
            f"beta disturbance schedule[{index}]",
            fault,
            frozenset({"type", "severity", "start_s", "duration_s", "channel"}),
        )
        if fault["type"] != "dac_outage":
            raise ValueError("beta disturbance schedule contains a non-DAC outage")
        if (
            not math.isclose(
                _number(
                    f"beta disturbance schedule[{index}].severity", fault["severity"]
                ),
                1.0,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or fault["channel"] is not None
        ):
            raise ValueError("beta DAC outage has noncanonical severity or channel")
        start_s = _number(
            f"beta disturbance schedule[{index}].start_s", fault["start_s"]
        )
        duration_s = _number(
            f"beta disturbance schedule[{index}].duration_s", fault["duration_s"]
        )
        start_step = int(round(start_s / dt_s))
        duration_steps = int(round(duration_s / dt_s))
        if (
            start_s < 0.0
            or duration_steps < 1
            or not math.isclose(start_s, start_step * dt_s, rel_tol=0.0, abs_tol=1e-9)
            or not math.isclose(
                duration_s,
                duration_steps * dt_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("beta disturbance schedule left the physics lattice")
        if start_step <= previous_start_step:
            raise ValueError("beta disturbance schedule is not strictly ordered")
        previous_start_step = start_step
        cycle_index = start_step // cycle_steps
        if cycle_index >= cycle_count or cycle_index in occupied_cycles:
            raise ValueError("beta disturbance schedule violates cycle ownership")
        occupied_cycles.add(cycle_index)
        if start_step + duration_steps > (cycle_index + 1) * cycle_steps - 1:
            raise ValueError("beta disturbance schedule crosses its cycle boundary")
        declared_steps.update(
            range(
                max(0, start_step),
                min(executed_steps, start_step + duration_steps),
            )
        )
    return frozenset(declared_steps)


def _verify_arm_classification(
    run: Mapping[str, Any],
    *,
    config: ControllerNeutralTrainingConfigV1,
    manifest: Mapping[str, Any],
    expected_decisions: int,
    expected_cycle_count: int,
    settlements: Sequence[Mapping[str, Any]],
    paired_eligible: bool,
    minute_states: Sequence[PlantState],
    derived_out_of_range: int,
    derived_forced_stops: int,
) -> dict[str, Any]:
    completed = _integer("arm completed decisions", run.get("completed_decisions"))
    physical_steps = _integer("arm physical steps", run.get("physical_steps"))
    terminated = _boolean("arm terminated", run.get("terminated"))
    truncated = _boolean("arm truncated", run.get("truncated"))
    failure = _verify_failure(run.get("failure"), name="arm failure")
    execution_complete = bool(
        failure is None
        and terminated
        and not truncated
        and completed == expected_decisions
        and physical_steps == config.episode.physics_steps(config.plant)
    )
    if config.beta_task is None:  # pragma: no cover - report config closes this
        raise RuntimeError("validated beta report omitted beta task")
    cycle_feasible = bool(
        execution_complete
        and len(settlements) == expected_cycle_count
        and [item.get("boundary_decision") for item in settlements]
        == [
            config.beta_task.cycle.cycle_decisions(config.episode) * (index + 1)
            for index in range(expected_cycle_count)
        ]
        and all(item.get("feasible") is True for item in settlements)
    )
    physical = _mapping("arm physical metrics", run.get("physical_metrics"))
    if _integer("physical metric decisions", physical.get("decisions")) != completed:
        raise ValueError("arm and physical-metric decision counts differ")
    if (
        _integer("physical metric steps", physical.get("physical_steps"))
        != physical_steps
    ):
        raise ValueError("arm and physical-metric physics counts differ")
    if len(minute_states) != physical_steps + 1:
        raise ValueError("minute boundary ledger does not cover executed physics")
    if not minute_states or minute_states[0].step != 0:
        raise ValueError("minute boundary ledger does not start at episode step zero")
    outage_lattice = _declared_dac_outage_lattice(
        manifest,
        config=config,
        executed_steps=physical_steps,
    )
    minimum_allowed_soc = float(config.plant.battery.min_soc)
    minimum_soc = min(state.battery_soc for state in minute_states)
    observed_shutdown = 0
    unexpected_shutdown = 0
    for before, after in zip(minute_states[:-1], minute_states[1:], strict=True):
        if after.step != before.step + 1:
            raise ValueError("minute safety ledger has a discontinuous physics step")
        critical_or_offline = after.plant_status in {CRITICAL, OFFLINE}
        observed_shutdown += int(critical_or_offline)
        declared_dac_outage = before.step in outage_lattice
        dac_health = after.health.get("dac")
        if dac_health is None:
            raise ValueError("minute safety ledger omitted DAC health")
        non_dac_offline = any(
            health.offline for unit, health in after.health.items() if unit != "dac"
        )
        battery_above_reserve = bool(
            after.battery_soc > minimum_allowed_soc + _BATTERY_SAFETY_TOLERANCE
        )
        if declared_dac_outage and (not dac_health.offline or not critical_or_offline):
            raise ValueError(
                "declared DAC outage is not reflected in minute truth evidence"
            )
        exogenous_dac_only = bool(
            declared_dac_outage
            and dac_health.offline
            and not non_dac_offline
            and battery_above_reserve
        )
        if critical_or_offline and not exogenous_dac_only:
            unexpected_shutdown += 1

    reported_shutdown = _integer(
        "physical metric shutdown_steps", physical.get("shutdown_steps")
    )
    if reported_shutdown != observed_shutdown:
        raise ValueError("physical shutdown_steps differs from minute truth evidence")
    reported_minimum_soc = _number(
        "physical metric min_battery_soc", physical.get("min_battery_soc")
    )
    if not math.isclose(reported_minimum_soc, minimum_soc, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("physical min_battery_soc differs from minute truth evidence")
    controller_counts = {
        "out_of_range_decisions": _integer(
            "derived out-of-range decisions", derived_out_of_range
        ),
        "electrolyser_forced_stops": _integer(
            "derived electrolyser forced stops", derived_forced_stops
        ),
    }
    for key, expected_count in controller_counts.items():
        if _integer(f"physical metric {key}", physical.get(key)) != expected_count:
            raise ValueError(f"physical metric {key} differs from retained evidence")
    battery_reserve_respected = bool(
        minimum_soc > minimum_allowed_soc + _BATTERY_SAFETY_TOLERANCE
    )
    safety: dict[str, int | float | bool] = {
        "observed_critical_or_offline_steps": observed_shutdown,
        "declared_dac_outage_steps": len(outage_lattice),
        "unexpected_critical_or_offline_steps": unexpected_shutdown,
        **controller_counts,
        "minimum_battery_soc": minimum_soc,
        "minimum_allowed_battery_soc": minimum_allowed_soc,
        "battery_reserve_respected": battery_reserve_respected,
    }
    if set(safety) != _HARD_SAFETY_FIELDS:  # pragma: no cover - local invariant
        raise RuntimeError("hard-safety verifier schema is inconsistent")
    hard_safety = bool(
        unexpected_shutdown == 0
        and all(value == 0 for value in controller_counts.values())
        and battery_reserve_respected
    )
    if not execution_complete:
        status = "execution_failed"
    elif not cycle_feasible:
        status = "complete_cycle_infeasible"
    elif not hard_safety:
        status = "complete_hard_safety_failed"
    else:
        status = "complete_feasible_safe"
    exclusions: list[str] = []
    if not execution_complete:
        exclusions.append("execution_incomplete")
    if not cycle_feasible:
        exclusions.append("cycle_infeasible_or_unsettled")
    if not hard_safety:
        exclusions.append("hard_safety_failed")
    candidate = execution_complete and cycle_feasible and hard_safety
    eligible = bool(paired_eligible and candidate)
    if not paired_eligible:
        exclusions.append("paired_cell_ineligible")
    expected = {
        "status": status,
        "execution_complete": execution_complete,
        "cycle_feasible": cycle_feasible,
        "hard_safety_passed": hard_safety,
        "value_comparison_candidate": candidate,
        "value_comparison_eligible": eligible,
        "exclusion_reasons": exclusions,
        "expected_decisions": expected_decisions,
        "expected_cycle_count": expected_cycle_count,
        "hard_safety_metrics": safety,
        "cycle_settlements": [dict(item) for item in settlements],
        "cumulative_cycle_constraint_cost": float(
            math.fsum(
                _number("settlement constraint cost", item.get("total_constraint_cost"))
                for item in settlements
            )
        ),
    }
    actual = {key: run.get(key) for key in _CLASSIFICATION_FIELDS}
    _exact("arm classification", actual, expected)
    return expected


def _verify_arm_digest(run: Mapping[str, Any]) -> None:
    supplied = _digest(
        "arm deterministic evidence", run.get("deterministic_evidence_sha256")
    )
    body = cast(dict[str, Any], _timing_free(dict(run)))
    body.pop("deterministic_evidence_sha256", None)
    if domain_sha256(_ARM_DOMAIN, body) != supplied:
        raise ValueError("arm deterministic evidence digest differs")


def _verify_preflight_arm(raw: object, *, arm: str) -> dict[str, Any]:
    run = _mapping(f"preflight {arm} arm", raw)
    _exact_keys(f"preflight {arm} arm", run, _PREFLIGHT_ARM_FIELDS)
    fixed = {
        "format": BETA_CONTROLLER_BENCHMARK_ARM_FORMAT,
        "arm": arm,
        "status": "not_run_preflight_failed",
        "execution_complete": False,
        "cycle_feasible": False,
        "hard_safety_passed": False,
        "value_comparison_candidate": False,
        "value_comparison_eligible": False,
        "exclusion_reasons": ["cell_preflight_failed"],
        "information_profile": BETA_CONTROLLER_INFORMATION_PROFILES[arm],
        "execution_door": BETA_EXECUTION_DOORS[arm],
        "controller": None,
        "manifest_identity": None,
        "completed_decisions": 0,
        "physical_steps": 0,
        "terminated": False,
        "truncated": False,
        "reward_trace_eur": [],
        "reward_trace_sum_eur": 0.0,
        "physical_metrics": None,
        "economic_metrics": None,
        "cycle_settlements": [],
    }
    for key, expected in fixed.items():
        if run.get(key) != expected:
            raise ValueError(f"preflight {arm} arm {key} differs")
    evidence = _mapping(f"preflight {arm} evidence", run["evidence"])
    ledger = _mapping(
        f"preflight {arm} boundary ledger", evidence.get("boundary_ledger")
    )
    if ledger != {
        "format": BETA_BOUNDARY_LEDGER_FORMAT,
        "decision_boundaries": [],
        "final_boundary": None,
    }:
        raise ValueError(f"preflight {arm} boundary ledger differs")
    if evidence.get("decisions") != [] or evidence.get("executions") != []:
        raise ValueError(f"preflight {arm} retained movement evidence")
    if evidence.get("minute_truth_states") != []:
        raise ValueError(f"preflight {arm} retained minute truth evidence")
    _verify_failure(run["failure"], name=f"preflight {arm} failure")
    _verify_arm_digest(run)
    return run


def _verify_normal_arm(
    raw: object,
    *,
    arm: str,
    config: ControllerNeutralTrainingConfigV1,
    settings: ServiceMPCSetpointSettingsV1,
    manifest_identity: Mapping[str, str],
    manifest: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    paired_eligible: bool,
    expected_scipy_version: str,
    highs_versions: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run = _mapping(f"{arm} arm", raw)
    _exact_keys(
        f"{arm} arm",
        run,
        _M24_ARM_FIELDS if arm == SERVICE_MPC_ARM else _GENERIC_ARM_FIELDS,
    )
    if run["format"] != BETA_CONTROLLER_BENCHMARK_ARM_FORMAT or run["arm"] != arm:
        raise ValueError(f"{arm} arm format/label differs")
    if run["information_profile"] != BETA_CONTROLLER_INFORMATION_PROFILES[arm]:
        raise ValueError(f"{arm} information profile differs")
    if run["execution_door"] != BETA_EXECUTION_DOORS[arm]:
        raise ValueError(f"{arm} execution door differs")
    if run["manifest_identity"] != dict(manifest_identity):
        raise ValueError(f"{arm} manifest identity differs from cell")

    completed = _integer(f"{arm} completed decisions", run["completed_decisions"])
    physical_steps = _integer(f"{arm} physical steps", run["physical_steps"])
    ledger = _mapping(
        f"{arm} boundary ledger",
        _mapping(f"{arm} evidence", run["evidence"]).get("boundary_ledger"),
    )
    _exact_keys(
        f"{arm} boundary ledger",
        ledger,
        frozenset({"format", "decision_boundaries", "final_boundary"}),
    )
    if ledger["format"] != BETA_BOUNDARY_LEDGER_FORMAT:
        raise ValueError(f"{arm} boundary ledger format differs")
    decision_raw = _array(f"{arm} decision boundaries", ledger["decision_boundaries"])
    encoder = ObservationEncoderV2(config.plant, config.episode.electrolyser_commitment)
    encoder.reset()
    boundary_payloads: list[dict[str, Any]] = []
    observations: list[CyclicMarketControllerObservationV1] = []
    stride = config.episode.decision_stride(config.plant)
    previous_solar_kw: float | None = None
    for index, item in enumerate(decision_raw):
        payload, observation, _ = _verify_boundary(
            item,
            name=f"{arm} decision boundary {index}",
            config=config,
            encoder=encoder,
            expected_kind="decision",
            expected_decision=index,
            expected_physics_step=index * stride,
            terminal=False,
            previous_solar_kw=previous_solar_kw,
        )
        boundary_payloads.append(payload)
        observations.append(observation)
        # The runner advances its solar-trend memory only after a complete
        # environment interval.  An extra boundary at ``index == completed``
        # is an attempted decision whose controller failed before movement, so
        # it must not replace the last successfully consumed solar sample.
        if index < completed:
            previous_solar_kw = float(
                observation.plant.state.boundary.solar_available_kw
            )
    final_raw = ledger["final_boundary"]
    if final_raw is None:
        raise ValueError(f"{arm} normal cell omitted final boundary")
    final_payload, final_observation, _ = _verify_boundary(
        final_raw,
        name=f"{arm} final boundary",
        config=config,
        encoder=encoder,
        expected_kind="final",
        expected_decision=completed,
        expected_physics_step=physical_steps,
        terminal=completed == config.episode.decision_steps(config.plant),
        previous_solar_kw=previous_solar_kw,
    )
    all_observations = [*observations, final_observation]
    if len(decision_raw) not in {completed, completed + 1}:
        raise ValueError(f"{arm} boundary cardinality differs from attempted decisions")
    if len(decision_raw) == completed + 1:
        expected_final = dict(boundary_payloads[-1])
        expected_final["kind"] = "final"
        _exact(
            f"{arm} duplicate final boundary",
            final_payload,
            expected_final,
        )

    rewards = [
        _number(f"{arm} reward[{index}]", value)
        for index, value in enumerate(
            _array(f"{arm} reward trace", run["reward_trace_eur"])
        )
    ]
    if len(rewards) != completed:
        raise ValueError(
            f"{arm} reward trace cardinality differs from completed decisions"
        )
    if not math.isclose(
        _number(f"{arm} reward sum", run["reward_trace_sum_eur"]),
        math.fsum(rewards),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{arm} reward trace sum differs")
    economic_metrics = _mapping(f"{arm} economic metrics", run["economic_metrics"])
    physical_metrics = _mapping(f"{arm} physical metrics", run["physical_metrics"])
    if (
        _integer(f"{arm} economic decisions", economic_metrics.get("decisions"))
        != completed
    ):
        raise ValueError(f"{arm} economic metric decision count differs")

    if not observations:
        raise ValueError(f"{arm} normal cell omitted its initial public boundary")
    targets = _cycle_targets_from_manifest(
        manifest, config, initial_observation=observations[0]
    )
    margin_fields = {
        "battery": "battery_margin_kwh",
        "h2": "h2_margin_kg",
        "co2": "co2_margin_kg",
        "ch4": "ch4_margin_kg",
    }
    for boundary_index, observation in enumerate(all_observations):
        state = observation.plant.state.boundary
        for resource in RESOURCE_NAMES:
            expected_margin = (
                float(getattr(state, _RESOURCE_STATE_FIELDS[resource]))
                - targets[resource]
            )
            if not math.isclose(
                float(getattr(observation.cycle, margin_fields[resource])),
                expected_margin,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"{arm} boundary {boundary_index} {resource} cycle margin differs"
                )
    raw_settlements = _array(f"{arm} cycle settlements", run["cycle_settlements"])
    settlements: list[dict[str, Any]] = []
    observation_by_decision = {
        int(payload["decision_index"]): observation
        for payload, observation in zip(boundary_payloads, observations, strict=True)
    }
    observation_by_decision[completed] = final_observation
    cycle_decisions = (
        config.beta_task.cycle.cycle_decisions(config.episode)
        if config.beta_task
        else 0
    )
    for index, item in enumerate(raw_settlements, start=1):
        boundary_decision = cycle_decisions * index
        settlement_observation = observation_by_decision.get(boundary_decision)
        if settlement_observation is None:
            raise ValueError(f"{arm} settlement lacks its public boundary")
        settlements.append(
            _verify_settlement(
                item,
                name=f"{arm} cycle settlement {index}",
                cycle_index=index,
                boundary_decision=boundary_decision,
                boundary_observation=settlement_observation,
                targets=targets,
                config=config,
            )
        )

    evidence = _mapping(f"{arm} evidence", run["evidence"])
    _verify_economic_intervals(
        evidence.get("intervals"),
        name=f"{arm} economic intervals",
        boundaries=all_observations[: completed + 1],
        reward_trace=rewards,
        settlements=settlements,
        economic_metrics=economic_metrics,
        physical_metrics=physical_metrics,
        config=config,
    )

    policy_metadata_sha: str | None = None
    policy_contract_sha: str | None = None
    beta_settings: BetaMPCSettingsV1 | None = None
    if arm == HEURISTIC_ARM:
        _exact(
            "H controller descriptor",
            run["controller"],
            LegacyHeuristicSetpointControllerV1().to_dict(),
        )
    elif arm == SERVICE_MPC_ARM:
        _exact(
            "M24 controller descriptor",
            run["controller"],
            ServiceMPCSetpointControllerV1(settings).to_dict(),
        )
    elif arm == PPO_ARM:
        policy_metadata_sha, policy_contract_sha = _verify_policy_controller(
            run["controller"], config=config, checkpoint=checkpoint
        )
    else:
        descriptor = _mapping("E-beta controller descriptor", run["controller"])
        beta_settings = _parse_beta_settings(descriptor.get("settings"))
        canonical_beta_settings = BetaMPCSettingsV1(
            task_horizon_h=config.episode.horizon_h,
            planning_horizon_h=min(24.0, config.episode.horizon_h),
            interval_s=config.episode.decision_interval_s,
        )
        if beta_settings != canonical_beta_settings:
            raise ValueError("E-beta controller changed canonical production settings")
        expected = CyclicMarketEconomicMPCControllerV1(
            settings=beta_settings,
            assumptions=config.economic_assumptions,
            beta_task=config.beta_task,
        ).to_dict()
        _exact("E-beta controller descriptor", descriptor, expected)

    minute_states = _decode_minute_truth_states(
        evidence.get("minute_truth_states"),
        name=f"{arm} minute truth states",
        config=config,
        physical_steps=physical_steps,
    )
    _exact(
        f"{arm} initial public/truth boundary",
        minute_states[0].to_dict(),
        observations[0].plant.state.boundary.to_dict(),
    )
    for index, observation in enumerate(observations):
        physics_step = index * stride
        if physics_step > physical_steps:
            raise ValueError(f"{arm} public boundary lies beyond retained truth")
        _exact(
            f"{arm} decision boundary {index} public/truth state",
            observation.plant.state.boundary.to_dict(),
            minute_states[physics_step].to_dict(),
        )
    _exact(
        f"{arm} final public/truth boundary",
        final_observation.plant.state.boundary.to_dict(),
        minute_states[-1].to_dict(),
    )
    derived_out_of_range = 0
    derived_forced_stops = _forced_stops_from_public_commitment(
        observations[0], final_observation
    )
    if arm == SERVICE_MPC_ARM:
        run_evidence = verify_mpc_setpoint_run_semantics(
            evidence.get("run"), settings=settings
        )
        typed_events, _ = decode_mpc_setpoint_run(run_evidence, settings=settings)
        if _integer("M24 closed decisions", run["closed_decisions"]) != completed:
            raise ValueError("M24 closed/completed decision counts differ")
        if run_evidence["completed_decisions"] != run["closed_decisions"]:
            raise ValueError("M24 run and arm completion counts differ")
        if (
            run_evidence["final_observation_sha256"]
            != final_payload["physical_observation_sha256"]
        ):
            raise ValueError("M24 run does not close on beta final boundary")
        events = _array("M24 events", run_evidence["events"])
        for event_index, raw_event in enumerate(events):
            event = _mapping(f"M24 event {event_index}", raw_event)
            for stage_index, raw_stage in enumerate(
                _array(f"M24 event {event_index} stages", event.get("stages"))
            ):
                stage = _mapping(
                    f"M24 event {event_index} stage {stage_index}", raw_stage
                )
                if stage.get("scipy_version") != expected_scipy_version:
                    raise ValueError("M24 stage SciPy version differs from provenance")
                highs_versions.add(
                    _text("M24 stage HiGHS version", stage.get("highs_version"))
                )
        successful_events = [
            event
            for event in events
            if _mapping("M24 event", event).get("schema")
            == "service-mpc-setpoint-decision-evidence-v1"
        ]
        closed_events = [
            event
            for event in typed_events
            if isinstance(event, MPCSetpointDecisionEvidenceV1)
            and event.execution is not None
        ]
        if len(closed_events) != completed:
            raise ValueError(
                "M24 closed execution count differs from completed decisions"
            )
        commitment_cursor = observations[0].plant.state.electrolyser_commitment
        truth_offset = 0
        for index, typed_event in enumerate(closed_events):
            execution = typed_event.execution
            if execution is None:  # pragma: no cover - filtered above
                raise RuntimeError("closed M24 event omitted execution")
            for application in execution.applications:
                if application.commitment_before != commitment_cursor:
                    raise ValueError(
                        "M24 minute commitment chain differs from public evidence"
                    )
                commitment_cursor = application.commitment_after
                if truth_offset + 1 >= len(minute_states):
                    raise ValueError("M24 execution exceeds retained minute truth")
                _exact(
                    f"M24 execution {index} minute {application.ordinal} boundary",
                    application.boundary_record,
                    minute_states[truth_offset].to_dict(),
                )
                _exact(
                    f"M24 execution {index} minute {application.ordinal} next boundary",
                    application.next_boundary_record,
                    minute_states[truth_offset + 1].to_dict(),
                )
                truth_offset += 1
            _exact(
                f"M24 execution {index} terminal public boundary",
                minute_states[truth_offset].to_dict(),
                all_observations[index + 1].plant.state.boundary.to_dict(),
            )
        if truth_offset != physical_steps:
            raise ValueError(
                "M24 execution ledger does not cover retained minute truth"
            )
        if commitment_cursor != final_observation.plant.state.electrolyser_commitment:
            raise ValueError("M24 terminal commitment differs from public evidence")
        for index, event in enumerate(successful_events):
            event_map = _mapping(f"M24 event {index}", event)
            if (
                event_map.get("observation_sha256")
                != boundary_payloads[index]["physical_observation_sha256"]
            ):
                raise ValueError("M24 event and beta boundary observations differ")
        _verify_failure(
            run.get("evidence_validation_failure"),
            name="M24 evidence validation failure",
        )
    else:
        decisions = _array(f"{arm} decisions", evidence.get("decisions"))
        executions = _array(f"{arm} executions", evidence.get("executions"))
        if _integer(
            f"{arm} evidence decision count", evidence.get("decision_count")
        ) != len(decisions):
            raise ValueError(f"{arm} evidence decision count differs")
        if _integer(
            f"{arm} evidence execution count", evidence.get("execution_count")
        ) != len(executions):
            raise ValueError(f"{arm} evidence execution count differs")
        if len(decisions) != len(executions) or len(executions) != completed:
            raise ValueError(f"{arm} decision/execution cardinality differs")
        derived_out_of_range = (
            0
            if arm == BETA_ECONOMIC_MPC_ARM
            else sum(
                _boolean(
                    f"{arm} decision {index} invalid",
                    _mapping(f"{arm} decision {index}", decision).get("invalid"),
                )
                for index, decision in enumerate(decisions)
            )
        )
        controller_elapsed = _array(
            f"{arm} controller elapsed", run.get("controller_elapsed_ns")
        )
        dispatcher_elapsed = _array(
            f"{arm} dispatcher elapsed", run.get("dispatcher_elapsed_ns")
        )
        if len(controller_elapsed) != len(decisions) or len(dispatcher_elapsed) != len(
            executions
        ):
            raise ValueError(f"{arm} timing cardinality differs")
        commands: list[ProcessSetpointCommandV1] = []
        decoded_executions: list[ProcessSetpointIntervalExecutionV1] = []
        initial_state = observations[0].plant.state.boundary
        initial_ch4_equivalent_kg = ch4_equivalent_inventory_kg(
            initial_state, config.plant
        )
        initial_battery_energy_kwh = float(initial_state.battery_energy_kwh)
        for index, decision in enumerate(decisions):
            commands.append(
                _verify_generic_decision(
                    decision,
                    arm=arm,
                    boundary_payload=boundary_payloads[index],
                    boundary=observations[index],
                    config=config,
                    checkpoint=checkpoint,
                    beta_settings=beta_settings,
                    policy_metadata_sha=policy_metadata_sha,
                    policy_contract_sha=policy_contract_sha,
                    initial_ch4_equivalent_kg=initial_ch4_equivalent_kg,
                    initial_battery_energy_kwh=initial_battery_energy_kwh,
                    expected_scipy_version=expected_scipy_version,
                    highs_versions=highs_versions,
                )
            )
            _integer(f"{arm} controller elapsed {index}", controller_elapsed[index])
        truth_offset = 0
        for index, (execution_raw, command) in enumerate(
            zip(executions, commands, strict=True)
        ):
            decoded_execution = _decode_execution(
                execution_raw, name=f"{arm} execution {index}", command=command
            )
            decoded_executions.append(decoded_execution)
            for minute_application in decoded_execution.applications:
                if truth_offset + 1 >= len(minute_states):
                    raise ValueError(f"{arm} execution exceeds retained minute truth")
                expected_boundary_sha256 = domain_sha256(
                    PROCESS_SETPOINT_BOUNDARY_DOMAIN,
                    minute_states[truth_offset].to_dict(),
                )
                expected_next_boundary_sha256 = domain_sha256(
                    PROCESS_SETPOINT_BOUNDARY_DOMAIN,
                    minute_states[truth_offset + 1].to_dict(),
                )
                if minute_application.boundary_sha256 != expected_boundary_sha256:
                    raise ValueError(
                        f"{arm} execution {index} minute boundary differs from truth"
                    )
                if (
                    minute_application.next_boundary_sha256
                    != expected_next_boundary_sha256
                ):
                    raise ValueError(
                        f"{arm} execution {index} next minute boundary differs from truth"
                    )
                truth_offset += 1
            _exact(
                f"{arm} execution {index} terminal public boundary",
                minute_states[truth_offset].to_dict(),
                all_observations[index + 1].plant.state.boundary.to_dict(),
            )
            expected_start_sha = domain_sha256(
                PROCESS_SETPOINT_BOUNDARY_DOMAIN,
                observations[index].plant.state.boundary.to_dict(),
            )
            expected_end_sha = domain_sha256(
                PROCESS_SETPOINT_BOUNDARY_DOMAIN,
                all_observations[index + 1].plant.state.boundary.to_dict(),
            )
            if decoded_execution.first_boundary_sha256 != expected_start_sha:
                raise ValueError(
                    f"{arm} execution {index} starts from another boundary"
                )
            if decoded_execution.final_boundary_sha256 != expected_end_sha:
                raise ValueError(f"{arm} execution {index} ends at another boundary")
            timings = _array(
                f"{arm} dispatcher elapsed {index}", dispatcher_elapsed[index]
            )
            if len(timings) != len(decoded_execution.applications):
                raise ValueError(f"{arm} dispatcher timing count differs")
            for timing in timings:
                _integer(f"{arm} dispatcher latency", timing)
        minute_count = sum(len(item.applications) for item in decoded_executions)
        if (
            _integer(
                f"{arm} physical minute count", evidence.get("physical_minute_count")
            )
            != minute_count
        ):
            raise ValueError(f"{arm} physical minute evidence count differs")
        if minute_count != physical_steps:
            raise ValueError(f"{arm} execution ledger does not cover physical steps")
        if truth_offset != physical_steps:
            raise ValueError(f"{arm} execution ledger does not cover retained truth")

    expected_decisions = config.episode.decision_steps(config.plant)
    expected_cycle_count = expected_decisions // cycle_decisions
    _verify_arm_classification(
        run,
        config=config,
        manifest=manifest,
        expected_decisions=expected_decisions,
        expected_cycle_count=expected_cycle_count,
        settlements=settlements,
        paired_eligible=paired_eligible,
        minute_states=minute_states,
        derived_out_of_range=derived_out_of_range,
        derived_forced_stops=derived_forced_stops,
    )
    _verify_arm_digest(run)
    return run, boundary_payloads


def _metric_number(mapping: Mapping[str, Any], key: str) -> float:
    return _number(f"metric {key}", mapping.get(key))


def _rebuild_cell_classification(
    runs: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    execution_complete = all(
        runs[arm].get("execution_complete") is True for arm in BETA_CONTROLLER_ARM_ORDER
    )
    cycle_feasible = all(
        runs[arm].get("cycle_feasible") is True for arm in BETA_CONTROLLER_ARM_ORDER
    )
    hard_safety = all(
        runs[arm].get("hard_safety_passed") is True for arm in BETA_CONTROLLER_ARM_ORDER
    )
    comparison: dict[str, Any] = {
        "tolerance_kg": BETA_SERVICE_COMPARISON_TOLERANCE_KG,
        "delivered_spread_kg": None,
        "unmet_spread_kg": None,
    }
    service_comparable = False
    if execution_complete:
        delivered: list[float] = []
        unmet: list[float] = []
        requested: list[float] = []
        for arm in BETA_CONTROLLER_ARM_ORDER:
            physical = _mapping(
                f"{arm} physical metrics", runs[arm]["physical_metrics"]
            )
            delivered.append(_metric_number(physical, "delivered_kg"))
            unmet.append(_metric_number(physical, "unmet_kg"))
            requested.append(_metric_number(physical, "requested_kg"))
        delivered_spread = max(delivered) - min(delivered)
        unmet_spread = max(unmet) - min(unmet)
        requested_spread = max(requested) - min(requested)
        service_comparable = bool(
            delivered_spread <= BETA_SERVICE_COMPARISON_TOLERANCE_KG
            and unmet_spread <= BETA_SERVICE_COMPARISON_TOLERANCE_KG
            and requested_spread <= BETA_SERVICE_COMPARISON_TOLERANCE_KG
        )
        comparison.update(
            {
                "delivered_spread_kg": delivered_spread,
                "unmet_spread_kg": unmet_spread,
                "requested_spread_kg": requested_spread,
            }
        )
    eligible = (
        execution_complete and cycle_feasible and hard_safety and service_comparable
    )
    exclusions: list[str] = []
    for arm in BETA_CONTROLLER_ARM_ORDER:
        if runs[arm].get("execution_complete") is not True:
            exclusions.append(f"arm_execution_incomplete:{arm}")
        if runs[arm].get("cycle_feasible") is not True:
            exclusions.append(f"arm_cycle_infeasible_or_unsettled:{arm}")
        if runs[arm].get("hard_safety_passed") is not True:
            exclusions.append(f"arm_hard_safety_failed:{arm}")
    if not service_comparable:
        exclusions.append("service_not_comparable")
    return {
        "status": "value_comparison_eligible" if eligible else "ineligible",
        "execution_complete": execution_complete,
        "cycle_feasible": cycle_feasible,
        "hard_safety_passed": hard_safety,
        "service_comparable": service_comparable,
        "service_comparison": comparison,
        "value_comparison_eligible": eligible,
        "exclusion_reasons": exclusions,
    }


def _rebuild_aggregate(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [cell for cell in cells if cell.get("value_comparison_eligible") is True]
    excluded = [
        {
            "seed": cell.get("seed"),
            "cell_status": cell.get("status"),
            "exclusion_reasons": list(
                _array("cell exclusion reasons", cell.get("exclusion_reasons"))
            ),
        }
        for cell in cells
        if cell.get("value_comparison_eligible") is not True
    ]
    arms: dict[str, dict[str, Any]] = {}
    for arm in BETA_CONTROLLER_ARM_ORDER:
        statuses: dict[str, int] = {}
        execution_count = cycle_count = safety_count = 0
        for cell in cells:
            run = _mapping("cell runs", cell.get("runs")).get(arm)
            if not isinstance(run, Mapping):
                status = "missing"
            else:
                status = str(run.get("status", "missing"))
                execution_count += int(run.get("execution_complete") is True)
                cycle_count += int(run.get("cycle_feasible") is True)
                safety_count += int(run.get("hard_safety_passed") is True)
            statuses[status] = statuses.get(status, 0) + 1
        rewards: list[float] = []
        values: list[float] = []
        delivered: list[float] = []
        unmet: list[float] = []
        for cell in eligible:
            run = _mapping("eligible cell runs", cell["runs"])[arm]
            run_map = _mapping(f"eligible {arm} run", run)
            physical = _mapping(
                f"eligible {arm} physical metrics", run_map["physical_metrics"]
            )
            economic = _mapping(
                f"eligible {arm} economic metrics", run_map["economic_metrics"]
            )
            rewards.append(_metric_number(run_map, "reward_trace_sum_eur"))
            values.append(_metric_number(economic, "net_operating_value_eur"))
            delivered.append(_metric_number(physical, "delivered_kg"))
            unmet.append(_metric_number(physical, "unmet_kg"))
        count = len(rewards)
        arms[arm] = {
            "registered_cell_count": len(cells),
            "execution_complete_count": execution_count,
            "cycle_feasible_count": cycle_count,
            "hard_safety_passed_count": safety_count,
            "status_counts": statuses,
            "value_eligible_cell_count": count,
            "reward_mean_eur": None if not count else float(math.fsum(rewards) / count),
            "net_operating_value_mean_eur": (
                None if not count else float(math.fsum(values) / count)
            ),
            "delivered_mean_kg": (
                None if not count else float(math.fsum(delivered) / count)
            ),
            "unmet_mean_kg": None if not count else float(math.fsum(unmet) / count),
        }
    return {
        "aggregation_policy": (
            "value metrics use only complete four-arm cells that pass cycle, hard "
            "safety, and service-comparability gates; no fallback or partial pooling"
        ),
        "registered_cell_count": len(cells),
        "preflight_failed_cell_count": sum(
            cell.get("status") == "preflight_failed" for cell in cells
        ),
        "execution_complete_cell_count": sum(
            cell.get("execution_complete") is True for cell in cells
        ),
        "cycle_feasible_cell_count": sum(
            cell.get("cycle_feasible") is True for cell in cells
        ),
        "hard_safety_passed_cell_count": sum(
            cell.get("hard_safety_passed") is True for cell in cells
        ),
        "service_comparable_cell_count": sum(
            cell.get("service_comparable") is True for cell in cells
        ),
        "value_eligible_cell_count": len(eligible),
        "excluded_cell_count": len(excluded),
        "value_eligible_seeds": [int(cell["seed"]) for cell in eligible],
        "excluded_cells": excluded,
        "arms": arms,
    }


def _verify_cell_digest(cell: Mapping[str, Any]) -> None:
    supplied = _digest(
        "cell deterministic evidence", cell.get("deterministic_evidence_sha256")
    )
    body = cast(dict[str, Any], _timing_free(dict(cell)))
    body.pop("deterministic_evidence_sha256", None)
    if domain_sha256(_CELL_DOMAIN, body) != supplied:
        raise ValueError("cell deterministic evidence digest differs")


def _verify_preflight_cell(raw: object, *, seed: int) -> dict[str, Any]:
    cell = _mapping(f"preflight cell seed {seed}", raw)
    _exact_keys(f"preflight cell seed {seed}", cell, _PREFLIGHT_CELL_FIELDS)
    fixed = {
        "format": BETA_CONTROLLER_BENCHMARK_CELL_FORMAT,
        "seed": seed,
        "status": "preflight_failed",
        "execution_complete": False,
        "cycle_feasible": False,
        "hard_safety_passed": False,
        "service_comparable": False,
        "value_comparison_eligible": False,
        "exclusion_reasons": ["cell_preflight_failed"],
        "manifest_identity": None,
        "audit_manifest": None,
        "arm_order": list(BETA_CONTROLLER_ARM_ORDER),
    }
    for key, expected in fixed.items():
        if cell.get(key) != expected:
            raise ValueError(f"preflight cell {key} differs")
    runs = _mapping("preflight cell runs", cell["runs"])
    if set(runs) != set(BETA_CONTROLLER_ARM_ORDER):
        raise ValueError("preflight cell arms are missing or unknown")
    for arm in BETA_CONTROLLER_ARM_ORDER:
        _verify_preflight_arm(runs[arm], arm=arm)
    failure = _verify_failure(cell["failure"], name="preflight cell failure")
    for arm in BETA_CONTROLLER_ARM_ORDER:
        if (
            _mapping(f"preflight {arm} failure", _mapping("run", runs[arm])["failure"])
            != failure
        ):
            raise ValueError(
                "preflight arms do not retain the common preflight failure"
            )
    _verify_cell_digest(cell)
    return cell


def _verify_normal_cell(
    raw: object,
    *,
    seed: int,
    config: ControllerNeutralTrainingConfigV1,
    settings: ServiceMPCSetpointSettingsV1,
    checkpoint: Mapping[str, Any],
    expected_scipy_version: str,
    highs_versions: set[str],
) -> dict[str, Any]:
    cell = _mapping(f"beta cell seed {seed}", raw)
    _exact_keys(f"beta cell seed {seed}", cell, _NORMAL_CELL_FIELDS)
    if cell["format"] != BETA_CONTROLLER_BENCHMARK_CELL_FORMAT or cell["seed"] != seed:
        raise ValueError("beta cell format/seed differs")
    if cell["arm_order"] != list(BETA_CONTROLLER_ARM_ORDER):
        raise ValueError("beta cell arm order differs")
    if cell["failure"] is not None:
        raise ValueError("non-preflight beta cell records a cell failure")
    identity = _verify_manifest_identity(
        cell["manifest_identity"], name="cell manifest identity", config=config
    )
    if (
        identity["checkpoint_contract_sha256"]
        != checkpoint["checkpoint_contract_sha256"]
    ):
        raise ValueError("cell task and checkpoint contract identities differ")
    manifest = _verify_audit_manifest(
        cell["audit_manifest"], identity=identity, config=config, seed=seed
    )
    runs = _mapping("beta cell runs", cell["runs"])
    if set(runs) != set(BETA_CONTROLLER_ARM_ORDER):
        raise ValueError("beta cell arms are missing or unknown")

    # Classification is calculated once before per-arm eligible flags are checked.
    provisional = _rebuild_cell_classification(
        {arm: _mapping(f"{arm} run", runs[arm]) for arm in BETA_CONTROLLER_ARM_ORDER}
    )
    expected_cell = provisional
    actual_cell = {key: cell.get(key) for key in expected_cell}
    _exact("beta cell classification", actual_cell, expected_cell)

    boundary_ledgers: dict[str, list[dict[str, Any]]] = {}
    for arm in BETA_CONTROLLER_ARM_ORDER:
        _, boundaries = _verify_normal_arm(
            runs[arm],
            arm=arm,
            config=config,
            settings=settings,
            manifest_identity=identity,
            manifest=manifest,
            checkpoint=checkpoint,
            paired_eligible=bool(provisional["value_comparison_eligible"]),
            expected_scipy_version=expected_scipy_version,
            highs_versions=highs_versions,
        )
        boundary_ledgers[arm] = boundaries
    first = boundary_ledgers[HEURISTIC_ARM]
    if not first:
        raise ValueError("normal beta cell has no initial boundary")
    for arm in BETA_CONTROLLER_ARM_ORDER[1:]:
        other = boundary_ledgers[arm]
        if not other or other[0] != first[0]:
            raise ValueError("beta arms did not start from the same admitted boundary")
    maximum_common = min(len(items) for items in boundary_ledgers.values())
    for decision in range(maximum_common):
        market_issues = {
            canonical_json(items[decision]["typed_observation"]["market"])
            for items in boundary_ledgers.values()
        }
        if len(market_issues) != 1:
            raise ValueError("beta arms observed different public market issues")
        weather_issues = {
            canonical_json(items[decision]["typed_observation"]["plant"]["forecast"])
            for items in boundary_ledgers.values()
        }
        if len(weather_issues) != 1:
            raise ValueError("beta arms observed different public weather issues")
        delivery_schedules = {
            canonical_json(
                {
                    key: items[decision]["typed_observation"]["plant"]["delivery"][key]
                    for key in (
                        "steps_to_next",
                        "delivery_period_steps",
                        "next_requested_kg",
                        "cumulative_requested_kg",
                    )
                }
            )
            for items in boundary_ledgers.values()
        }
        if len(delivery_schedules) != 1:
            raise ValueError("beta arms observed different public delivery schedules")
    _verify_cell_digest(cell)
    return cell


def verify_beta_controller_benchmark(
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently admit one in-memory beta benchmark report.

    The returned mapping is a compact verifier result, not a trusted projection
    copied from the report.
    """

    report = _mapping("beta benchmark report", raw)
    _finite_tree("beta benchmark report", report)
    _exact_keys("beta benchmark report", report, _REPORT_FIELDS)
    if report["format"] != BETA_CONTROLLER_BENCHMARK_FORMAT:
        raise ValueError("beta benchmark format differs")
    if report["claim_boundary"] != BETA_CONTROLLER_BENCHMARK_CLAIM_BOUNDARY:
        raise ValueError("beta benchmark claim boundary differs")
    config = ControllerNeutralTrainingConfigV1.from_dict(
        _mapping("beta training task", report["training_task"])
    )
    _exact("beta training task", report["training_task"], config.to_dict())
    if config.environment_id != beta_spaces.ENVIRONMENT_ID or config.beta_task is None:
        raise ValueError("benchmark training task is not the exact cyclic-market beta")
    settings = _parse_service_settings(report["service_mpc_settings"])
    if (
        not math.isclose(settings.task_horizon_h, config.episode.horizon_h)
        or not math.isclose(
            settings.planning_horizon_h, min(24.0, config.episode.horizon_h)
        )
        or not math.isclose(settings.interval_s, config.episode.decision_interval_s)
    ):
        raise ValueError("service MPC settings differ from the beta task")
    checkpoint, _, checkpoint_contract_sha = _verify_checkpoint_and_policy_descriptor(
        report["checkpoint"], config=config
    )
    provenance = _verify_source_provenance(report["source_provenance"])
    if checkpoint["plant_sim_version"] != provenance["runtime"]["plant_sim"]["version"]:
        raise ValueError("checkpoint and benchmark runtime plant_sim versions differ")
    trained = bool(checkpoint["trained"])
    expected_policy_qualification = (
        "trained-sb3-ppo" if trained else "integration-fixture-not-a-learned-controller"
    )
    if report["policy_qualification"] != expected_policy_qualification:
        raise ValueError("policy qualification differs from checkpoint evidence")
    benchmark_qualification = report["benchmark_qualification"]
    if benchmark_qualification not in {
        "canonical-four-arm-beta-development-benchmark",
        "unqualified-integration-fixture-or-custom-arm",
    }:
        raise ValueError("benchmark qualification is unsupported")
    if benchmark_qualification == "canonical-four-arm-beta-development-benchmark":
        if not trained or checkpoint["format"] != checkpoints.FORMAT_SB3_PPO:
            raise ValueError("qualified benchmark does not use a trained SB3 PPO")
        if provenance["worktree_clean"] is not True:
            raise ValueError("qualified benchmark source was not clean")
        qualified_plant_sim_path = Path(str(provenance["runtime"]["plant_sim"]["path"]))
        if "simulator" not in qualified_plant_sim_path.parts:
            raise ValueError(
                "qualified benchmark plant_sim was not loaded from simulator source"
            )
        canonical_settings = ServiceMPCSetpointSettingsV1(
            task_horizon_h=config.episode.horizon_h,
            planning_horizon_h=min(24.0, config.episode.horizon_h),
            interval_s=config.episode.decision_interval_s,
            terminal_battery_soc_target=(
                ServiceMPCSetpointSettingsV1().terminal_battery_soc_target
            ),
        )
        if settings != canonical_settings:
            raise ValueError("qualified benchmark changed canonical M24 settings")
    seeds = [
        _integer(f"benchmark seed {index}", value)
        for index, value in enumerate(_array("benchmark seeds", report["seeds"]))
    ]
    if not seeds or len(seeds) > BETA_CONTROLLER_BENCHMARK_MAX_SEEDS:
        raise ValueError("benchmark seed cardinality is unsupported")
    if len(set(seeds)) != len(seeds):
        raise ValueError("benchmark seeds are not unique")
    if report["arm_order"] != list(BETA_CONTROLLER_ARM_ORDER):
        raise ValueError("benchmark arm order differs")
    expected_profiles = {
        arm: dict(BETA_CONTROLLER_INFORMATION_PROFILES[arm])
        for arm in BETA_CONTROLLER_ARM_ORDER
    }
    if report["information_profiles"] != expected_profiles:
        raise ValueError("benchmark information profiles differ")
    if report["execution_doors"] != dict(BETA_EXECUTION_DOORS):
        raise ValueError("benchmark execution doors differ")

    cells_raw = _array("benchmark cells", report["cells"])
    if len(cells_raw) != len(seeds):
        raise ValueError("benchmark cell cardinality differs from seeds")
    cells: list[dict[str, Any]] = []
    highs_versions: set[str] = set()
    for seed, raw_cell in zip(seeds, cells_raw, strict=True):
        cell_map = _mapping(f"cell seed {seed}", raw_cell)
        if cell_map.get("seed") != seed:
            raise ValueError("benchmark cells are reordered")
        if cell_map.get("status") == "preflight_failed":
            cells.append(_verify_preflight_cell(cell_map, seed=seed))
        else:
            cells.append(
                _verify_normal_cell(
                    cell_map,
                    seed=seed,
                    config=config,
                    settings=settings,
                    checkpoint=checkpoint,
                    expected_scipy_version=str(provenance["runtime"]["scipy"]),
                    highs_versions=highs_versions,
                )
            )
    if len(highs_versions) > 1:
        raise ValueError("benchmark solver stages report inconsistent HiGHS versions")
    rebuilt_aggregate = _rebuild_aggregate(cells)
    _exact("beta benchmark aggregate", report["aggregate"], rebuilt_aggregate)
    expected_report_body = {
        "format": report["format"],
        "claim_boundary": report["claim_boundary"],
        "policy_qualification": report["policy_qualification"],
        "benchmark_qualification": report["benchmark_qualification"],
        "training_task": report["training_task"],
        "checkpoint": _timing_free(checkpoint),
        "source_provenance": report["source_provenance"],
        "seeds": report["seeds"],
        "arm_order": report["arm_order"],
        "information_profiles": report["information_profiles"],
        "execution_doors": report["execution_doors"],
        "service_mpc_settings": report["service_mpc_settings"],
        "cell_evidence": [cell["deterministic_evidence_sha256"] for cell in cells],
        "aggregate": rebuilt_aggregate,
    }
    expected_sha = domain_sha256(_REPORT_DOMAIN, expected_report_body)
    if report["deterministic_evidence_sha256"] != expected_sha:
        raise ValueError("beta benchmark deterministic evidence digest differs")
    return {
        "format": BETA_CONTROLLER_BENCHMARK_FORMAT,
        "verification": "passed",
        "deterministic_evidence_sha256": expected_sha,
        "checkpoint_contract_sha256": checkpoint_contract_sha,
        "cell_count": len(cells),
        "value_eligible_cell_count": rebuilt_aggregate["value_eligible_cell_count"],
        "excluded_cell_count": rebuilt_aggregate["excluded_cell_count"],
    }


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _finite_json_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite JSON number {value!r} is forbidden")
    return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _read_regular_file(path: Path) -> bytes:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("beta benchmark JSON must be a regular nonsymlink file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)

        def identity(item: os.stat_result) -> tuple[int, int, int, int, int]:
            return (
                item.st_dev,
                item.st_ino,
                item.st_mode,
                item.st_size,
                item.st_mtime_ns,
            )

        if identity(opened) != identity(before):
            raise RuntimeError("beta benchmark JSON changed while opening")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if identity(after) != identity(before):
        raise RuntimeError("beta benchmark JSON changed while reading")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise RuntimeError("beta benchmark JSON size changed while reading")
    return raw


def verify_beta_controller_benchmark_file(path: Path | str) -> dict[str, Any]:
    """Strictly reopen and semantically verify one runner-produced JSON file."""

    target = Path(path)
    raw = _read_regular_file(target)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as reason:
        raise ValueError("beta benchmark JSON is not UTF-8") from reason
    decoded = json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_json_constant,
        parse_float=_finite_json_float,
    )
    report = _mapping("beta benchmark JSON", decoded)
    expected = (
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if raw != expected:
        raise ValueError(
            "beta benchmark JSON is not canonical sorted two-space JSON with one trailing newline"
        )
    result = verify_beta_controller_benchmark(report)
    return {
        **result,
        "path": str(target.resolve()),
        "raw_file_sha256": hashlib.sha256(raw).hexdigest(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Verify one report from the command line without executing the benchmark."""

    parser = argparse.ArgumentParser(
        prog="python -m plant_backend.rl.v2.beta_controller_benchmark_verifier",
        description="Strictly verify a cyclic-market beta controller benchmark JSON.",
    )
    parser.add_argument("report", type=Path, help="canonical benchmark JSON path")
    arguments = parser.parse_args(None if argv is None else list(argv))
    try:
        result = verify_beta_controller_benchmark_file(arguments.report)
    except (OSError, RuntimeError, TypeError, ValueError) as reason:
        print(f"beta benchmark refused: {reason}", file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


__all__ = [
    "main",
    "verify_beta_controller_benchmark",
    "verify_beta_controller_benchmark_file",
]


if __name__ == "__main__":  # pragma: no cover - exercised through CLI integration
    raise SystemExit(main())

"""Typed action and public transition evidence for causal service MPC."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, ClassVar

from plant_sim import Action

from .control import ControllerObservationV2, close_unit_simplex
from .mpc_schema import (
    MODEL_BALANCE_TOLERANCE,
    MPCDeliveryTargetV1,
    MPCPlanStepV1,
    MPCPlanV1,
    MPCStageSolveV1,
    SERVICE_MPC_CONTROLLER_ID,
    ServiceMPCPlanError,
    domain_sha256,
    require_sha256,
)

SERVICE_MPC_ACTION_DOMAIN = "service-mpc-requested-action-v1"
SERVICE_MPC_DECISION_DOMAIN = "service-mpc-decision-evidence-v1"
SERVICE_MPC_FAILURE_DOMAIN = "service-mpc-decision-failure-v1"
SERVICE_MPC_FORECAST_ISSUE_DOMAIN = "service-mpc-public-forecast-issue-v1"
SERVICE_MPC_OBSERVATION_DOMAIN = "service-mpc-public-observation-v1"
SERVICE_MPC_RESIDUAL_DOMAIN = "service-mpc-first-interval-residual-v1"


def _finite(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum:g}")
    return number


def _index(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def public_observation_sha256(observation: ControllerObservationV2) -> str:
    if not isinstance(observation, ControllerObservationV2):
        raise TypeError("observation must be ControllerObservationV2")
    return domain_sha256(SERVICE_MPC_OBSERVATION_DOMAIN, observation.to_dict())


def public_forecast_issue_sha256(observation: ControllerObservationV2) -> str:
    if not isinstance(observation, ControllerObservationV2):
        raise TypeError("observation must be ControllerObservationV2")
    if observation.forecast is None:
        raise ValueError("service MPC requires a public forecast issue")
    return domain_sha256(
        SERVICE_MPC_FORECAST_ISSUE_DOMAIN,
        observation.forecast.to_dict(),
    )


@dataclass(frozen=True, slots=True)
class MPCActionRealizationV1:
    """First planned route encoded into the existing five-field action seam."""

    plan_sha256: str
    forecast_solar_kw: float
    battery_share: float
    electrolyser_share: float
    dac_share: float
    methanator_share: float
    battery_support_kw: float

    def __post_init__(self) -> None:
        require_sha256("plan_sha256", self.plan_sha256)
        _finite("forecast_solar_kw", self.forecast_solar_kw, minimum=0.0)
        shares = tuple(
            _finite(name, getattr(self, name), minimum=0.0, maximum=1.0)
            for name in (
                "battery_share",
                "electrolyser_share",
                "dac_share",
                "methanator_share",
            )
        )
        _finite("battery_support_kw", self.battery_support_kw, minimum=0.0)
        if sum(shares) > 1.0:
            raise ValueError(
                "service-MPC requested action oversubscribes the solar bus"
            )
        if self.forecast_solar_kw == 0.0 and any(value != 0.0 for value in shares):
            raise ValueError("a dark MPC interval must use zero solar shares")

    @property
    def action(self) -> Action:
        return Action(
            battery_share=self.battery_share,
            electrolyser_share=self.electrolyser_share,
            dac_share=self.dac_share,
            methanator_share=self.methanator_share,
            battery_support_kw=self.battery_support_kw,
        )

    def body(self) -> dict[str, Any]:
        return {
            "plan_sha256": self.plan_sha256,
            "forecast_solar_kw": float(self.forecast_solar_kw),
            "action": self.action.to_dict(),
            "simplex_sum": float(
                math.fsum(
                    (
                        self.battery_share,
                        self.electrolyser_share,
                        self.dac_share,
                        self.methanator_share,
                    )
                )
            ),
            "realization": (
                "first planned solar-route kW divided by first reconstructed "
                "interval PV; binary64 simplex residue closed downward"
            ),
        }

    @property
    def sha256(self) -> str:
        return domain_sha256(SERVICE_MPC_ACTION_DOMAIN, self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


def realize_first_mpc_action(plan: MPCPlanV1) -> MPCActionRealizationV1:
    """Encode only the first valid plan interval without sanitizer repair."""

    if not isinstance(plan, MPCPlanV1):
        raise TypeError("plan must be MPCPlanV1")
    first = plan.steps[0]
    denominator = first.forecast_solar_kw
    routes = (
        first.solar_to_battery_kw,
        first.solar_to_electrolyser_kw,
        first.solar_to_dac_kw,
        first.solar_to_methanator_kw,
    )
    if denominator == 0.0:
        if any(value != 0.0 for value in routes):
            raise ServiceMPCPlanError(
                "service-MPC dark first interval contains a nonzero solar route"
            )
        shares = (0.0, 0.0, 0.0, 0.0)
    else:
        shares = close_unit_simplex(
            (
                routes[0] / denominator,
                routes[1] / denominator,
                routes[2] / denominator,
                routes[3] / denominator,
            )
        )
        for name, planned_kw, share in zip(
            ("battery", "electrolyser", "DAC", "methanator"),
            routes,
            shares,
            strict=True,
        ):
            if not math.isclose(
                share * denominator,
                planned_kw,
                rel_tol=0.0,
                abs_tol=MODEL_BALANCE_TOLERANCE,
            ):
                raise ServiceMPCPlanError(
                    f"service-MPC {name} share does not realize its first route"
                )
    realization = MPCActionRealizationV1(
        plan_sha256=plan.sha256,
        forecast_solar_kw=denominator,
        battery_share=shares[0],
        electrolyser_share=shares[1],
        dac_share=shares[2],
        methanator_share=shares[3],
        battery_support_kw=first.battery_support_kw,
    )
    if Action.sanitised(realization.action) != realization.action:
        raise ServiceMPCPlanError(
            "service-MPC first action would require sanitizer correction"
        )
    return realization


@dataclass(frozen=True, slots=True)
class MPCFirstIntervalResidualV1:
    """Plan-minus-plant reconciliation from the next public boundary only."""

    decision_index: int
    next_observation_sha256: str
    hours: float
    planned_forecast_solar_kw: float
    observed_solar_mean_kw: float
    solar_forecast_error_kw: float
    planned_battery_net_kw: float
    observed_battery_net_kw: float
    battery_net_error_kw: float
    planned_electrolyser_kw: float
    observed_electrolyser_kw: float
    electrolyser_error_kw: float
    planned_dac_kw: float
    observed_dac_kw: float
    dac_error_kw: float
    planned_methanator_kw: float
    observed_methanator_kw: float
    methanator_error_kw: float
    planned_curtailed_kw: float
    observed_curtailed_kw: float
    curtailed_error_kw: float
    planned_h2_kg: float
    observed_h2_kg: float
    h2_error_kg: float
    planned_co2_kg: float
    observed_co2_kg: float
    co2_error_kg: float
    planned_ch4_kg: float
    observed_ch4_kg: float
    ch4_error_kg: float
    planned_delivered_kg: float
    observed_delivered_kg: float
    delivered_error_kg: float
    planned_battery_end_kwh: float
    observed_battery_end_kwh: float
    battery_end_error_kwh: float
    planned_h2_end_kg: float
    observed_h2_end_kg: float
    h2_end_error_kg: float
    planned_co2_end_kg: float
    observed_co2_end_kg: float
    co2_end_error_kg: float
    planned_ch4_end_kg: float
    observed_ch4_end_kg: float
    ch4_end_error_kg: float
    process_power_l1_error_kwh: float

    _PAIRS: ClassVar[tuple[tuple[str, str, str], ...]] = (
        (
            "planned_forecast_solar_kw",
            "observed_solar_mean_kw",
            "solar_forecast_error_kw",
        ),
        ("planned_battery_net_kw", "observed_battery_net_kw", "battery_net_error_kw"),
        (
            "planned_electrolyser_kw",
            "observed_electrolyser_kw",
            "electrolyser_error_kw",
        ),
        ("planned_dac_kw", "observed_dac_kw", "dac_error_kw"),
        ("planned_methanator_kw", "observed_methanator_kw", "methanator_error_kw"),
        ("planned_curtailed_kw", "observed_curtailed_kw", "curtailed_error_kw"),
        ("planned_h2_kg", "observed_h2_kg", "h2_error_kg"),
        ("planned_co2_kg", "observed_co2_kg", "co2_error_kg"),
        ("planned_ch4_kg", "observed_ch4_kg", "ch4_error_kg"),
        ("planned_delivered_kg", "observed_delivered_kg", "delivered_error_kg"),
        (
            "planned_battery_end_kwh",
            "observed_battery_end_kwh",
            "battery_end_error_kwh",
        ),
        ("planned_h2_end_kg", "observed_h2_end_kg", "h2_end_error_kg"),
        ("planned_co2_end_kg", "observed_co2_end_kg", "co2_end_error_kg"),
        ("planned_ch4_end_kg", "observed_ch4_end_kg", "ch4_end_error_kg"),
    )

    def __post_init__(self) -> None:
        _index("residual decision_index", self.decision_index)
        require_sha256("next_observation_sha256", self.next_observation_sha256)
        hours = _finite("residual hours", self.hours, minimum=0.0)
        if hours <= 0.0:
            raise ValueError("residual hours must be positive")
        for planned_name, observed_name, error_name in self._PAIRS:
            planned = _finite(planned_name, getattr(self, planned_name))
            observed = _finite(observed_name, getattr(self, observed_name))
            error = _finite(error_name, getattr(self, error_name))
            if not math.isclose(
                error,
                observed - planned,
                rel_tol=0.0,
                abs_tol=MODEL_BALANCE_TOLERANCE,
            ):
                raise ValueError(f"{error_name} does not equal observed minus planned")
            if planned_name != "planned_battery_net_kw" and planned < 0.0:
                raise ValueError(f"{planned_name} must be non-negative")
            if observed_name != "observed_battery_net_kw" and observed < 0.0:
                raise ValueError(f"{observed_name} must be non-negative")
        power_l1 = _finite(
            "process_power_l1_error_kwh",
            self.process_power_l1_error_kwh,
            minimum=0.0,
        )
        expected_l1 = hours * math.fsum(
            abs(getattr(self, name))
            for name in (
                "battery_net_error_kw",
                "electrolyser_error_kw",
                "dac_error_kw",
                "methanator_error_kw",
                "curtailed_error_kw",
            )
        )
        if not math.isclose(
            power_l1,
            expected_l1,
            rel_tol=0.0,
            abs_tol=MODEL_BALANCE_TOLERANCE,
        ):
            raise ValueError("process-power L1 residual is inconsistent")

    def body(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return domain_sha256(SERVICE_MPC_RESIDUAL_DOMAIN, self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


def reconcile_first_interval(
    *,
    decision_index: int,
    before: ControllerObservationV2,
    after: ControllerObservationV2,
    planned: MPCPlanStepV1,
) -> MPCFirstIntervalResidualV1:
    """Compare the first plan interval with the next detached public observation."""

    index = _index("decision_index", decision_index)
    if not isinstance(before, ControllerObservationV2) or not isinstance(
        after, ControllerObservationV2
    ):
        raise TypeError("residual boundaries must be ControllerObservationV2 records")
    if not isinstance(planned, MPCPlanStepV1) or planned.index != 0:
        raise TypeError("residual planned step must be MPC plan interval zero")
    interval = after.state.previous_interval
    expected_steps = after.state.boundary.step - before.state.boundary.step
    if expected_steps <= 0 or interval.physical_steps != expected_steps:
        raise ValueError(
            "public residual interval has an unexpected physics-step count"
        )
    expected_h = before.decision_interval_s / 3_600.0
    if not math.isclose(interval.hours, expected_h, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("public residual interval has an unexpected duration")
    delivered = (
        after.delivery.cumulative_delivered_kg - before.delivery.cumulative_delivered_kg
    )
    if delivered < -MODEL_BALANCE_TOLERANCE:
        raise ValueError("public cumulative delivery moved backwards")
    delivered = max(0.0, delivered)
    observed_curtailed_kw = interval.solar_curtailed_kwh / interval.hours
    values = {
        "forecast": (planned.forecast_solar_kw, interval.solar_available_mean_kw),
        "battery": (
            planned.solar_to_battery_kw - planned.battery_support_kw,
            interval.battery_power_mean_kw,
        ),
        "electrolyser": (
            planned.electrolyser_startup_kw + planned.electrolyser_productive_kw,
            interval.electrolyser_power_mean_kw,
        ),
        "dac": (planned.solar_to_dac_kw, interval.dac_power_mean_kw),
        "methanator": (
            planned.solar_to_methanator_kw,
            interval.methanator_power_mean_kw,
        ),
        "curtailed": (planned.solar_curtailed_kw, observed_curtailed_kw),
        "h2": (planned.h2_produced_kg, interval.h2_produced_kg),
        "co2": (planned.co2_captured_kg, interval.co2_captured_kg),
        "ch4": (planned.ch4_produced_kg, interval.ch4_produced_kg),
        "delivered": (planned.delivered_kg, delivered),
        "battery_end": (
            planned.battery_energy_end_kwh,
            after.state.boundary.battery_energy_kwh,
        ),
        "h2_end": (planned.h2_end_kg, after.state.boundary.h2_level_kg),
        "co2_end": (planned.co2_end_kg, after.state.boundary.co2_level_kg),
        "ch4_end": (planned.ch4_end_kg, after.state.boundary.ch4_level_kg),
    }
    errors = {
        name: observed - expected for name, (expected, observed) in values.items()
    }
    process_l1 = interval.hours * math.fsum(
        abs(errors[name])
        for name in ("battery", "electrolyser", "dac", "methanator", "curtailed")
    )
    return MPCFirstIntervalResidualV1(
        decision_index=index,
        next_observation_sha256=public_observation_sha256(after),
        hours=interval.hours,
        planned_forecast_solar_kw=values["forecast"][0],
        observed_solar_mean_kw=values["forecast"][1],
        solar_forecast_error_kw=errors["forecast"],
        planned_battery_net_kw=values["battery"][0],
        observed_battery_net_kw=values["battery"][1],
        battery_net_error_kw=errors["battery"],
        planned_electrolyser_kw=values["electrolyser"][0],
        observed_electrolyser_kw=values["electrolyser"][1],
        electrolyser_error_kw=errors["electrolyser"],
        planned_dac_kw=values["dac"][0],
        observed_dac_kw=values["dac"][1],
        dac_error_kw=errors["dac"],
        planned_methanator_kw=values["methanator"][0],
        observed_methanator_kw=values["methanator"][1],
        methanator_error_kw=errors["methanator"],
        planned_curtailed_kw=values["curtailed"][0],
        observed_curtailed_kw=values["curtailed"][1],
        curtailed_error_kw=errors["curtailed"],
        planned_h2_kg=values["h2"][0],
        observed_h2_kg=values["h2"][1],
        h2_error_kg=errors["h2"],
        planned_co2_kg=values["co2"][0],
        observed_co2_kg=values["co2"][1],
        co2_error_kg=errors["co2"],
        planned_ch4_kg=values["ch4"][0],
        observed_ch4_kg=values["ch4"][1],
        ch4_error_kg=errors["ch4"],
        planned_delivered_kg=values["delivered"][0],
        observed_delivered_kg=values["delivered"][1],
        delivered_error_kg=errors["delivered"],
        planned_battery_end_kwh=values["battery_end"][0],
        observed_battery_end_kwh=values["battery_end"][1],
        battery_end_error_kwh=errors["battery_end"],
        planned_h2_end_kg=values["h2_end"][0],
        observed_h2_end_kg=values["h2_end"][1],
        h2_end_error_kg=errors["h2_end"],
        planned_co2_end_kg=values["co2_end"][0],
        observed_co2_end_kg=values["co2_end"][1],
        co2_end_error_kg=errors["co2_end"],
        planned_ch4_end_kg=values["ch4_end"][0],
        observed_ch4_end_kg=values["ch4_end"][1],
        ch4_end_error_kg=errors["ch4_end"],
        process_power_l1_error_kwh=process_l1,
    )


@dataclass(frozen=True, slots=True)
class MPCDecisionEvidenceV1:
    """One successful causal decision, later closed by a public residual."""

    decision_index: int
    observation_sha256: str
    public_forecast_sha256: str
    projection_sha256: str
    settings_sha256: str
    model_sha256: str
    model_variable_count: int
    model_binary_variable_count: int
    model_row_count: int
    model_nonzero_count: int
    horizon_intervals: int
    deliveries: tuple[MPCDeliveryTargetV1, ...]
    stages: tuple[MPCStageSolveV1, MPCStageSolveV1]
    plan_sha256: str
    primary_delivered_optimum_kg: float
    selected_delivered_kg: float
    minimum_commitment_fill: float
    terminal_ch4_equivalent_kg: float
    secondary_utility: float
    first_step: MPCPlanStepV1
    action: MPCActionRealizationV1
    notes: tuple[str, ...]
    decision_elapsed_ns: int
    residual: MPCFirstIntervalResidualV1 | None = None

    def __post_init__(self) -> None:
        index = _index("decision_index", self.decision_index)
        for name in (
            "observation_sha256",
            "public_forecast_sha256",
            "projection_sha256",
            "settings_sha256",
            "model_sha256",
            "plan_sha256",
        ):
            require_sha256(name, getattr(self, name))
        for name in (
            "model_variable_count",
            "model_binary_variable_count",
            "model_row_count",
            "model_nonzero_count",
            "horizon_intervals",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.model_binary_variable_count > self.model_variable_count:
            raise ValueError("binary variable count exceeds total model variables")
        if not isinstance(self.deliveries, tuple) or any(
            not isinstance(item, MPCDeliveryTargetV1) for item in self.deliveries
        ):
            raise TypeError("decision deliveries must be an immutable typed tuple")
        if not isinstance(self.stages, tuple) or len(self.stages) != 2:
            raise ValueError("decision evidence requires both solver stages")
        if tuple(stage.stage for stage in self.stages) != (
            "primary_delivery",
            "secondary_tiebreak",
        ) or not all(stage.success for stage in self.stages):
            raise ValueError("decision evidence requires ordered successful stages")
        for name in (
            "primary_delivered_optimum_kg",
            "selected_delivered_kg",
            "terminal_ch4_equivalent_kg",
        ):
            _finite(name, getattr(self, name), minimum=0.0)
        _finite(
            "minimum_commitment_fill",
            self.minimum_commitment_fill,
            minimum=0.0,
            maximum=1.0,
        )
        _finite("secondary_utility", self.secondary_utility)
        if not isinstance(self.first_step, MPCPlanStepV1) or self.first_step.index != 0:
            raise ValueError("decision evidence requires plan interval zero")
        if not isinstance(self.action, MPCActionRealizationV1):
            raise TypeError("decision action must be MPCActionRealizationV1")
        if self.action.plan_sha256 != self.plan_sha256:
            raise ValueError("decision action and plan identities differ")
        if not math.isclose(
            self.action.forecast_solar_kw,
            self.first_step.forecast_solar_kw,
            rel_tol=0.0,
            abs_tol=MODEL_BALANCE_TOLERANCE,
        ):
            raise ValueError("decision action and first-step forecast differ")
        for name, share, planned_kw in zip(
            ("battery", "electrolyser", "DAC", "methanator"),
            (
                self.action.battery_share,
                self.action.electrolyser_share,
                self.action.dac_share,
                self.action.methanator_share,
            ),
            (
                self.first_step.solar_to_battery_kw,
                self.first_step.solar_to_electrolyser_kw,
                self.first_step.solar_to_dac_kw,
                self.first_step.solar_to_methanator_kw,
            ),
            strict=True,
        ):
            if not math.isclose(
                share * self.action.forecast_solar_kw,
                planned_kw,
                rel_tol=0.0,
                abs_tol=MODEL_BALANCE_TOLERANCE,
            ):
                raise ValueError(
                    f"decision {name} share does not match the first-step route"
                )
        if (
            not isinstance(self.notes, tuple)
            or not self.notes
            or any(not isinstance(note, str) or not note for note in self.notes)
        ):
            raise ValueError("decision notes must be a non-empty immutable tuple")
        if (
            isinstance(self.decision_elapsed_ns, bool)
            or not isinstance(self.decision_elapsed_ns, Integral)
            or self.decision_elapsed_ns < 0
        ):
            raise ValueError("decision_elapsed_ns must be non-negative")
        if self.decision_elapsed_ns < sum(stage.elapsed_ns for stage in self.stages):
            raise ValueError("decision latency cannot be below summed solver latency")
        if self.residual is not None and (
            not isinstance(self.residual, MPCFirstIntervalResidualV1)
            or self.residual.decision_index != index
        ):
            raise ValueError(
                "decision residual is absent or belongs to another decision"
            )

    def body(self) -> dict[str, Any]:
        return {
            "schema": "service-mpc-decision-evidence-v1",
            "controller_id": SERVICE_MPC_CONTROLLER_ID,
            "decision_index": int(self.decision_index),
            "observation_sha256": self.observation_sha256,
            "public_forecast_sha256": self.public_forecast_sha256,
            "projection_sha256": self.projection_sha256,
            "settings_sha256": self.settings_sha256,
            "model": {
                "sha256": self.model_sha256,
                "variables": int(self.model_variable_count),
                "binary_variables": int(self.model_binary_variable_count),
                "rows": int(self.model_row_count),
                "nonzeros": int(self.model_nonzero_count),
            },
            "horizon_intervals": int(self.horizon_intervals),
            "deliveries": [item.to_dict() for item in self.deliveries],
            "stages": [stage.to_dict() for stage in self.stages],
            "plan": {
                "sha256": self.plan_sha256,
                "primary_delivered_optimum_kg": float(
                    self.primary_delivered_optimum_kg
                ),
                "selected_delivered_kg": float(self.selected_delivered_kg),
                "minimum_commitment_fill": float(self.minimum_commitment_fill),
                "terminal_ch4_equivalent_kg": float(self.terminal_ch4_equivalent_kg),
                "secondary_utility": float(self.secondary_utility),
            },
            "first_step": self.first_step.to_dict(),
            "action": self.action.to_dict(),
            "notes": list(self.notes),
            "decision_elapsed_ns": int(self.decision_elapsed_ns),
            "residual": None if self.residual is None else self.residual.to_dict(),
        }

    def identity_body(self) -> dict[str, Any]:
        body = self.body()
        del body["decision_elapsed_ns"]
        del body["stages"]
        body["stage_solution_sha256"] = [stage.solution_sha256 for stage in self.stages]
        return body

    @property
    def sha256(self) -> str:
        return domain_sha256(SERVICE_MPC_DECISION_DOMAIN, self.identity_body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class MPCDecisionFailureV1:
    """One failed decision attempt retained without manufacturing an action."""

    decision_index: int
    phase: str
    exception_type: str
    message: str
    elapsed_ns: int
    observation_sha256: str | None = None
    public_forecast_sha256: str | None = None
    projection_sha256: str | None = None
    model_sha256: str | None = None
    plan_sha256: str | None = None
    stage_evidence: MPCStageSolveV1 | None = None

    def __post_init__(self) -> None:
        _index("failure decision_index", self.decision_index)
        if self.phase not in {
            "transition_reconciliation",
            "input_validation",
            "forecast_projection",
            "model_construction",
            "solver",
            "action_realization",
            "evidence_construction",
        }:
            raise ValueError("service-MPC failure phase is invalid")
        for name in ("exception_type", "message"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"failure {name} must be non-empty")
        if len(self.message) > 500:
            raise ValueError("failure message exceeds the evidence bound")
        if (
            isinstance(self.elapsed_ns, bool)
            or not isinstance(self.elapsed_ns, Integral)
            or self.elapsed_ns < 0
        ):
            raise ValueError("failure elapsed_ns must be non-negative")
        for name in (
            "observation_sha256",
            "public_forecast_sha256",
            "projection_sha256",
            "model_sha256",
            "plan_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                require_sha256(name, value)
        if self.stage_evidence is not None and (
            not isinstance(self.stage_evidence, MPCStageSolveV1)
            or self.stage_evidence.success
        ):
            raise ValueError("failure stage evidence must be an unsuccessful stage")

    def body(self) -> dict[str, Any]:
        return {
            "schema": "service-mpc-decision-failure-v1",
            "controller_id": SERVICE_MPC_CONTROLLER_ID,
            "decision_index": int(self.decision_index),
            "phase": self.phase,
            "exception_type": self.exception_type,
            "message": self.message,
            "elapsed_ns": int(self.elapsed_ns),
            "observation_sha256": self.observation_sha256,
            "public_forecast_sha256": self.public_forecast_sha256,
            "projection_sha256": self.projection_sha256,
            "model_sha256": self.model_sha256,
            "plan_sha256": self.plan_sha256,
            "stage_evidence": (
                None if self.stage_evidence is None else self.stage_evidence.to_dict()
            ),
        }

    @property
    def sha256(self) -> str:
        identity = self.body()
        del identity["elapsed_ns"]
        return domain_sha256(SERVICE_MPC_FAILURE_DOMAIN, identity)

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


MPCDecisionEventV1 = MPCDecisionEvidenceV1 | MPCDecisionFailureV1


__all__ = [
    "MPCActionRealizationV1",
    "MPCDecisionEventV1",
    "MPCDecisionEvidenceV1",
    "MPCDecisionFailureV1",
    "MPCFirstIntervalResidualV1",
    "public_forecast_issue_sha256",
    "public_observation_sha256",
    "realize_first_mpc_action",
    "reconcile_first_interval",
]

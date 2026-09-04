"""Absolute service-MPC intent and deterministic one-minute dispatch.

The optimizer speaks only at the 15-minute controller boundary.  This module turns
its first planned interval into an immutable absolute-power command, then realizes
that held intent against one contemporaneous solar bus at a time.  It does not
forecast, optimize, inspect private episode state, or bypass the independent guard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, ClassVar

from plant_sim import Action, PlantConfig, PlantState

from ...controllers.base import Constraints
from ...safety.validation import (
    BatterySupportAvailability,
    battery_support_availability,
)
from .control import close_unit_simplex
from .mpc_forecast import MPCForecastTrajectoryV1
from .mpc_schema import (
    MODEL_BALANCE_TOLERANCE,
    MPCPlanV1,
    SERVICE_MPC_SETPOINT_CONTROLLER_ID,
    ServiceMPCPlanError,
    domain_sha256,
    require_sha256,
)

SERVICE_MPC_PROCESS_SETPOINT_SCHEMA = "service-mpc-process-setpoints-v1"
SERVICE_MPC_PROCESS_SETPOINT_DOMAIN = SERVICE_MPC_PROCESS_SETPOINT_SCHEMA
SERVICE_MPC_SETPOINT_ENVELOPE_DOMAIN = "service-mpc-setpoint-envelope-v1"
SERVICE_MPC_FIRST_INTERVAL_DOMAIN = "service-mpc-projected-first-interval-v1"
SERVICE_MPC_DISPATCH_BOUNDARY_DOMAIN = "service-mpc-minute-boundary-v1"
SERVICE_MPC_MINUTE_DISPATCH_SCHEMA = "service-mpc-minute-dispatch-v1"
SERVICE_MPC_MINUTE_DISPATCH_DOMAIN = SERVICE_MPC_MINUTE_DISPATCH_SCHEMA
RESERVE_AWARE_DISPATCH_POLICY = "reserve-aware-proportional-unsupported-loads-v1"


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


def _balanced(error: float, *scale_values: float) -> bool:
    scale = max(1.0, *(abs(value) for value in scale_values))
    tolerance = MODEL_BALANCE_TOLERANCE + MODEL_BALANCE_TOLERANCE * scale
    return abs(error) <= tolerance


@dataclass(frozen=True, slots=True)
class MPCSetpointEnvelopeV1:
    """Resolved issue-time equipment and safety envelope for one command."""

    electrolyser_limit_kw: float
    dac_limit_kw: float
    methanator_limit_kw: float
    battery_charge_limit_kw: float
    battery_support_limit_kw: float
    battery_reserve_soc: float

    def __post_init__(self) -> None:
        for name in (
            "electrolyser_limit_kw",
            "dac_limit_kw",
            "methanator_limit_kw",
            "battery_charge_limit_kw",
            "battery_support_limit_kw",
        ):
            _finite(name, getattr(self, name), minimum=0.0)
        _finite(
            "battery_reserve_soc",
            self.battery_reserve_soc,
            minimum=0.0,
            maximum=1.0,
        )

    def body(self) -> dict[str, float]:
        return {
            "electrolyser_limit_kw": float(self.electrolyser_limit_kw),
            "dac_limit_kw": float(self.dac_limit_kw),
            "methanator_limit_kw": float(self.methanator_limit_kw),
            "battery_charge_limit_kw": float(self.battery_charge_limit_kw),
            "battery_support_limit_kw": float(self.battery_support_limit_kw),
            "battery_reserve_soc": float(self.battery_reserve_soc),
        }

    @property
    def sha256(self) -> str:
        return domain_sha256(SERVICE_MPC_SETPOINT_ENVELOPE_DOMAIN, self.body())

    def to_dict(self) -> dict[str, float | str]:
        return {**self.body(), "sha256": self.sha256}


def resolve_setpoint_envelope(
    *,
    constraints: Constraints,
    config: PlantConfig,
) -> MPCSetpointEnvelopeV1:
    if not isinstance(constraints, Constraints):
        raise TypeError("constraints must be Constraints")
    if not isinstance(config, PlantConfig):
        raise TypeError("config must be PlantConfig")
    return MPCSetpointEnvelopeV1(
        electrolyser_limit_kw=constraints.electrolyser_limit_kw(config),
        dac_limit_kw=constraints.dac_limit_kw(config),
        methanator_limit_kw=constraints.methanator_limit_kw(config),
        battery_charge_limit_kw=config.battery.max_charge_kw,
        battery_support_limit_kw=constraints.support_limit_kw(config),
        battery_reserve_soc=constraints.reserve_soc(config),
    )


def projected_first_interval_sha256(projection: MPCForecastTrajectoryV1) -> str:
    if not isinstance(projection, MPCForecastTrajectoryV1):
        raise TypeError("projection must be MPCForecastTrajectoryV1")
    return domain_sha256(
        SERVICE_MPC_FIRST_INTERVAL_DOMAIN,
        {
            "projection_sha256": projection.sha256,
            "interval_index": 0,
            "interval_s": float(projection.interval_s),
            "available_kw": float(projection.available_kw[0]),
            "band_name": projection.interval_band_names[0],
        },
    )


@dataclass(frozen=True, slots=True)
class MPCProcessSetpointsV1:
    """One hash-bound 15-minute absolute process intent from service MPC."""

    decision_index: int
    observation_sha256: str
    public_forecast_sha256: str
    projection_sha256: str
    first_interval_sha256: str
    settings_sha256: str
    model_sha256: str
    plan_sha256: str
    hold_duration_s: float
    electrolyser_target_kw: float
    dac_target_kw: float
    methanator_target_kw: float
    battery_charge_ceiling_kw: float
    expected_battery_support_kw: float
    planned_forecast_solar_kw: float
    planned_solar_to_battery_kw: float
    planned_solar_to_electrolyser_kw: float
    planned_solar_to_dac_kw: float
    planned_solar_to_methanator_kw: float
    planned_solar_curtailed_kw: float
    envelope: MPCSetpointEnvelopeV1
    dispatch_policy_id: str = RESERVE_AWARE_DISPATCH_POLICY

    schema: ClassVar[str] = SERVICE_MPC_PROCESS_SETPOINT_SCHEMA
    controller_id: ClassVar[str] = SERVICE_MPC_SETPOINT_CONTROLLER_ID

    def __post_init__(self) -> None:
        _index("setpoint decision_index", self.decision_index)
        for name in (
            "observation_sha256",
            "public_forecast_sha256",
            "projection_sha256",
            "first_interval_sha256",
            "settings_sha256",
            "model_sha256",
            "plan_sha256",
        ):
            require_sha256(name, getattr(self, name))
        duration = _finite("hold_duration_s", self.hold_duration_s, minimum=0.0)
        if duration <= 0.0:
            raise ValueError("hold_duration_s must be positive")
        for name in (
            "electrolyser_target_kw",
            "dac_target_kw",
            "methanator_target_kw",
            "battery_charge_ceiling_kw",
            "expected_battery_support_kw",
            "planned_forecast_solar_kw",
            "planned_solar_to_battery_kw",
            "planned_solar_to_electrolyser_kw",
            "planned_solar_to_dac_kw",
            "planned_solar_to_methanator_kw",
            "planned_solar_curtailed_kw",
        ):
            _finite(name, getattr(self, name), minimum=0.0)
        if not isinstance(self.envelope, MPCSetpointEnvelopeV1):
            raise TypeError("setpoint envelope must be MPCSetpointEnvelopeV1")
        if self.dispatch_policy_id != RESERVE_AWARE_DISPATCH_POLICY:
            raise ValueError("setpoint dispatch policy is unsupported")

        caps = (
            (
                "electrolyser",
                self.electrolyser_target_kw,
                self.envelope.electrolyser_limit_kw,
            ),
            ("DAC", self.dac_target_kw, self.envelope.dac_limit_kw),
            (
                "methanator",
                self.methanator_target_kw,
                self.envelope.methanator_limit_kw,
            ),
            (
                "battery charge",
                self.battery_charge_ceiling_kw,
                self.envelope.battery_charge_limit_kw,
            ),
        )
        for name, target, limit in caps:
            if target > limit + MODEL_BALANCE_TOLERANCE:
                raise ValueError(f"{name} setpoint exceeds its active limit")
        if (
            self.expected_battery_support_kw
            > self.envelope.battery_support_limit_kw + MODEL_BALANCE_TOLERANCE
        ):
            raise ValueError("expected battery support exceeds its active limit")
        if (
            self.expected_battery_support_kw
            > self.electrolyser_target_kw + MODEL_BALANCE_TOLERANCE
        ):
            raise ValueError("expected battery support exceeds the electrolyser target")

        solar_routed = math.fsum(
            (
                self.planned_solar_to_battery_kw,
                self.planned_solar_to_electrolyser_kw,
                self.planned_solar_to_dac_kw,
                self.planned_solar_to_methanator_kw,
                self.planned_solar_curtailed_kw,
            )
        )
        if not _balanced(
            solar_routed - self.planned_forecast_solar_kw,
            solar_routed,
            self.planned_forecast_solar_kw,
        ):
            raise ValueError("setpoint planned solar evidence does not balance")
        bindings = (
            (
                "electrolyser",
                self.electrolyser_target_kw,
                self.planned_solar_to_electrolyser_kw
                + self.expected_battery_support_kw,
            ),
            ("DAC", self.dac_target_kw, self.planned_solar_to_dac_kw),
            (
                "methanator",
                self.methanator_target_kw,
                self.planned_solar_to_methanator_kw,
            ),
            (
                "battery charge",
                self.battery_charge_ceiling_kw,
                self.planned_solar_to_battery_kw,
            ),
        )
        for name, target, planned in bindings:
            if not _balanced(target - planned, target, planned):
                raise ValueError(f"{name} setpoint does not match first-plan evidence")

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "controller_id": self.controller_id,
            "decision_index": int(self.decision_index),
            "observation_sha256": self.observation_sha256,
            "public_forecast_sha256": self.public_forecast_sha256,
            "projection_sha256": self.projection_sha256,
            "first_interval_sha256": self.first_interval_sha256,
            "settings_sha256": self.settings_sha256,
            "model_sha256": self.model_sha256,
            "plan_sha256": self.plan_sha256,
            "hold_duration_s": float(self.hold_duration_s),
            "setpoints_kw": {
                "electrolyser": float(self.electrolyser_target_kw),
                "dac": float(self.dac_target_kw),
                "methanator": float(self.methanator_target_kw),
                "battery_charge_ceiling": float(self.battery_charge_ceiling_kw),
            },
            "expected_battery_support_kw": float(self.expected_battery_support_kw),
            "planned_first_interval_kw": {
                "forecast_solar": float(self.planned_forecast_solar_kw),
                "solar_to_battery": float(self.planned_solar_to_battery_kw),
                "solar_to_electrolyser": float(self.planned_solar_to_electrolyser_kw),
                "solar_to_dac": float(self.planned_solar_to_dac_kw),
                "solar_to_methanator": float(self.planned_solar_to_methanator_kw),
                "solar_curtailed": float(self.planned_solar_curtailed_kw),
            },
            "envelope": self.envelope.to_dict(),
            "dispatch_policy_id": self.dispatch_policy_id,
        }

    @property
    def sha256(self) -> str:
        return domain_sha256(SERVICE_MPC_PROCESS_SETPOINT_DOMAIN, self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


def realize_first_mpc_setpoints(
    plan: MPCPlanV1,
    projection: MPCForecastTrajectoryV1,
    *,
    decision_index: int,
    observation_sha256: str,
    public_forecast_sha256: str,
    model_sha256: str,
    constraints: Constraints,
    config: PlantConfig,
) -> MPCProcessSetpointsV1:
    """Bind the first selected interval to an absolute-process command."""

    if not isinstance(plan, MPCPlanV1):
        raise TypeError("plan must be MPCPlanV1")
    if not isinstance(projection, MPCForecastTrajectoryV1):
        raise TypeError("projection must be MPCForecastTrajectoryV1")
    index = _index("decision_index", decision_index)
    require_sha256("observation_sha256", observation_sha256)
    forecast_sha = require_sha256("public_forecast_sha256", public_forecast_sha256)
    model_sha = require_sha256("model_sha256", model_sha256)
    if plan.model_sha256 != model_sha:
        raise ServiceMPCPlanError("setpoint model and selected plan identities differ")
    if plan.forecast_sha256 != projection.sha256:
        raise ServiceMPCPlanError(
            "setpoint forecast projection and selected plan identities differ"
        )
    if projection.source_issue_sha256 != forecast_sha:
        raise ServiceMPCPlanError(
            "setpoint public issue and forecast projection identities differ"
        )
    if any(stage.solution_selection is None for stage in plan.stages):
        raise ServiceMPCPlanError(
            "setpoint plan lacks feasibility-selection stage evidence"
        )
    first = plan.steps[0]
    if not math.isclose(
        first.forecast_solar_kw,
        projection.available_kw[0],
        rel_tol=0.0,
        abs_tol=MODEL_BALANCE_TOLERANCE,
    ):
        raise ServiceMPCPlanError(
            "setpoint first-plan solar differs from the projected interval"
        )
    return MPCProcessSetpointsV1(
        decision_index=index,
        observation_sha256=observation_sha256,
        public_forecast_sha256=forecast_sha,
        projection_sha256=projection.sha256,
        first_interval_sha256=projected_first_interval_sha256(projection),
        settings_sha256=plan.settings_sha256,
        model_sha256=model_sha,
        plan_sha256=plan.sha256,
        hold_duration_s=projection.interval_s,
        electrolyser_target_kw=(
            first.electrolyser_startup_kw + first.electrolyser_productive_kw
        ),
        dac_target_kw=first.solar_to_dac_kw,
        methanator_target_kw=first.solar_to_methanator_kw,
        battery_charge_ceiling_kw=first.solar_to_battery_kw,
        expected_battery_support_kw=first.battery_support_kw,
        planned_forecast_solar_kw=first.forecast_solar_kw,
        planned_solar_to_battery_kw=first.solar_to_battery_kw,
        planned_solar_to_electrolyser_kw=first.solar_to_electrolyser_kw,
        planned_solar_to_dac_kw=first.solar_to_dac_kw,
        planned_solar_to_methanator_kw=first.solar_to_methanator_kw,
        planned_solar_curtailed_kw=first.solar_curtailed_kw,
        envelope=resolve_setpoint_envelope(constraints=constraints, config=config),
    )


def minute_boundary_sha256(boundary: PlantState) -> str:
    if not isinstance(boundary, PlantState):
        raise TypeError("boundary must be PlantState")
    return domain_sha256(SERVICE_MPC_DISPATCH_BOUNDARY_DOMAIN, boundary.to_dict())


@dataclass(frozen=True, slots=True)
class _DispatchAllocation:
    routes_kw: tuple[float, float, float, float]
    support_kw: float
    unallocated_kw: float
    target_reachable: bool
    action: Action


def _proportional_unsupported_routes(
    remaining_solar_kw: float,
    dac_target_kw: float,
    methanator_target_kw: float,
) -> tuple[float, float]:
    total_target = dac_target_kw + methanator_target_kw
    if total_target <= 0.0 or remaining_solar_kw <= 0.0:
        return 0.0, 0.0
    factor = min(1.0, remaining_solar_kw / total_target)
    dac_kw = dac_target_kw * factor
    methanator_kw = methanator_target_kw * factor
    routed = dac_kw + methanator_kw
    if routed > remaining_solar_kw:
        closure = math.nextafter(remaining_solar_kw / routed, 0.0)
        dac_kw *= closure
        methanator_kw *= closure
    return dac_kw, methanator_kw


def _allocate_dispatch(
    *,
    solar_kw: float,
    electrolyser_target_kw: float,
    dac_target_kw: float,
    methanator_target_kw: float,
    battery_charge_ceiling_kw: float,
    support: BatterySupportAvailability,
) -> _DispatchAllocation:
    reserved_electrolyser_kw = min(
        solar_kw,
        max(0.0, electrolyser_target_kw - support.available_kw),
    )
    remaining_kw = max(0.0, solar_kw - reserved_electrolyser_kw)
    dac_kw, methanator_kw = _proportional_unsupported_routes(
        remaining_kw,
        dac_target_kw,
        methanator_target_kw,
    )
    remaining_kw = max(0.0, remaining_kw - dac_kw - methanator_kw)
    extra_electrolyser_kw = min(
        remaining_kw,
        max(0.0, electrolyser_target_kw - reserved_electrolyser_kw),
    )
    electrolyser_solar_kw = reserved_electrolyser_kw + extra_electrolyser_kw
    remaining_kw = max(0.0, remaining_kw - extra_electrolyser_kw)
    support_kw = min(
        support.available_kw,
        max(0.0, electrolyser_target_kw - electrolyser_solar_kw),
    )
    battery_kw = (
        min(battery_charge_ceiling_kw, remaining_kw) if support_kw == 0.0 else 0.0
    )

    if solar_kw > 0.0:
        shares = close_unit_simplex(
            (
                battery_kw / solar_kw,
                electrolyser_solar_kw / solar_kw,
                dac_kw / solar_kw,
                methanator_kw / solar_kw,
            )
        )
    else:
        shares = (0.0, 0.0, 0.0, 0.0)
    action = Action(
        battery_share=shares[0],
        electrolyser_share=shares[1],
        dac_share=shares[2],
        methanator_share=shares[3],
        battery_support_kw=support_kw,
    )
    if Action.sanitised(action) != action:
        raise RuntimeError("setpoint dispatcher did not close the action simplex")
    routes = (
        shares[0] * solar_kw,
        shares[1] * solar_kw,
        shares[2] * solar_kw,
        shares[3] * solar_kw,
    )
    unallocated_kw = max(0.0, solar_kw - math.fsum(routes))
    target_reachable = bool(
        electrolyser_target_kw == 0.0
        or (
            support.electrolyser_available
            and electrolyser_target_kw
            <= support.electrolyser_headroom_kw + MODEL_BALANCE_TOLERANCE
            and solar_kw + support.available_kw
            >= electrolyser_target_kw - MODEL_BALANCE_TOLERANCE
        )
    )
    return _DispatchAllocation(
        routes_kw=routes,
        support_kw=support_kw,
        unallocated_kw=unallocated_kw,
        target_reachable=target_reachable,
        action=action,
    )


@dataclass(frozen=True, slots=True)
class MPCMinuteDispatchV1:
    """Self-auditing request produced for one contemporaneous physical minute."""

    setpoint_sha256: str
    decision_index: int
    boundary_sha256: str
    boundary_step: int
    solar_available_kw: float
    electrolyser_target_kw: float
    dac_target_kw: float
    methanator_target_kw: float
    battery_charge_ceiling_kw: float
    support_availability: BatterySupportAvailability
    solar_to_battery_kw: float
    solar_to_electrolyser_kw: float
    solar_to_dac_kw: float
    solar_to_methanator_kw: float
    battery_support_kw: float
    unallocated_solar_kw: float
    electrolyser_target_reachable: bool
    action: Action

    schema: ClassVar[str] = SERVICE_MPC_MINUTE_DISPATCH_SCHEMA
    policy_id: ClassVar[str] = RESERVE_AWARE_DISPATCH_POLICY

    def __post_init__(self) -> None:
        require_sha256("setpoint_sha256", self.setpoint_sha256)
        require_sha256("boundary_sha256", self.boundary_sha256)
        _index("dispatch decision_index", self.decision_index)
        _index("dispatch boundary_step", self.boundary_step)
        for name in (
            "solar_available_kw",
            "electrolyser_target_kw",
            "dac_target_kw",
            "methanator_target_kw",
            "battery_charge_ceiling_kw",
            "solar_to_battery_kw",
            "solar_to_electrolyser_kw",
            "solar_to_dac_kw",
            "solar_to_methanator_kw",
            "battery_support_kw",
            "unallocated_solar_kw",
        ):
            _finite(name, getattr(self, name), minimum=0.0)
        if not isinstance(self.support_availability, BatterySupportAvailability):
            raise TypeError("support_availability must be typed safety evidence")
        if not isinstance(self.electrolyser_target_reachable, bool):
            raise TypeError("electrolyser_target_reachable must be a bool")
        if not isinstance(self.action, Action):
            raise TypeError("dispatch action must be Action")
        if Action.sanitised(self.action) != self.action:
            raise ValueError("minute dispatch action is not a sanitizer fixed point")

        expected = _allocate_dispatch(
            solar_kw=self.solar_available_kw,
            electrolyser_target_kw=self.electrolyser_target_kw,
            dac_target_kw=self.dac_target_kw,
            methanator_target_kw=self.methanator_target_kw,
            battery_charge_ceiling_kw=self.battery_charge_ceiling_kw,
            support=self.support_availability,
        )
        if (
            (
                self.solar_to_battery_kw,
                self.solar_to_electrolyser_kw,
                self.solar_to_dac_kw,
                self.solar_to_methanator_kw,
            )
            != expected.routes_kw
            or self.battery_support_kw != expected.support_kw
            or self.unallocated_solar_kw != expected.unallocated_kw
            or self.electrolyser_target_reachable != expected.target_reachable
            or self.action != expected.action
        ):
            raise ValueError("minute dispatch does not match the declared policy")

        routes = (
            self.solar_to_battery_kw,
            self.solar_to_electrolyser_kw,
            self.solar_to_dac_kw,
            self.solar_to_methanator_kw,
        )
        for name, share, routed in zip(
            ("battery", "electrolyser", "DAC", "methanator"),
            (
                self.action.battery_share,
                self.action.electrolyser_share,
                self.action.dac_share,
                self.action.methanator_share,
            ),
            routes,
            strict=True,
        ):
            if not math.isclose(
                share * self.solar_available_kw,
                routed,
                rel_tol=0.0,
                abs_tol=MODEL_BALANCE_TOLERANCE,
            ):
                raise ValueError(f"minute {name} route disagrees with its action share")
        routed_solar = math.fsum((*routes, self.unallocated_solar_kw))
        if not _balanced(
            routed_solar - self.solar_available_kw,
            routed_solar,
            self.solar_available_kw,
        ):
            raise ValueError("minute dispatch solar bus does not balance")
        if not math.isclose(
            self.action.battery_support_kw,
            self.battery_support_kw,
            rel_tol=0.0,
            abs_tol=MODEL_BALANCE_TOLERANCE,
        ):
            raise ValueError("minute support disagrees with its action")
        if (
            self.solar_to_electrolyser_kw + self.battery_support_kw
            > self.electrolyser_target_kw + MODEL_BALANCE_TOLERANCE
        ):
            raise ValueError("minute dispatch over-offers the electrolyser target")
        if self.electrolyser_target_reachable and not math.isclose(
            self.solar_to_electrolyser_kw + self.battery_support_kw,
            self.electrolyser_target_kw,
            rel_tol=0.0,
            abs_tol=MODEL_BALANCE_TOLERANCE,
        ):
            raise ValueError("reachable electrolyser target is not realized")
        if self.solar_to_dac_kw > self.dac_target_kw + MODEL_BALANCE_TOLERANCE:
            raise ValueError("minute DAC route exceeds its target")
        if (
            self.solar_to_methanator_kw
            > self.methanator_target_kw + MODEL_BALANCE_TOLERANCE
        ):
            raise ValueError("minute methanator route exceeds its target")
        if (
            self.solar_to_battery_kw
            > self.battery_charge_ceiling_kw + MODEL_BALANCE_TOLERANCE
        ):
            raise ValueError("minute battery route exceeds its charge ceiling")
        if (
            self.battery_support_kw
            > self.support_availability.available_kw + MODEL_BALANCE_TOLERANCE
        ):
            raise ValueError("minute support exceeds reserve-aware availability")
        if self.solar_to_battery_kw > 0.0 and self.battery_support_kw > 0.0:
            raise ValueError("minute dispatch cannot charge and support simultaneously")

    @property
    def electrolyser_offered_kw(self) -> float:
        return self.solar_to_electrolyser_kw + self.battery_support_kw

    @property
    def electrolyser_shortfall_kw(self) -> float:
        return max(0.0, self.electrolyser_target_kw - self.electrolyser_offered_kw)

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "setpoint_sha256": self.setpoint_sha256,
            "decision_index": int(self.decision_index),
            "boundary_sha256": self.boundary_sha256,
            "boundary_step": int(self.boundary_step),
            "solar_available_kw": float(self.solar_available_kw),
            "targets_kw": {
                "electrolyser": float(self.electrolyser_target_kw),
                "dac": float(self.dac_target_kw),
                "methanator": float(self.methanator_target_kw),
                "battery_charge_ceiling": float(self.battery_charge_ceiling_kw),
            },
            "support_availability": self.support_availability.to_dict(),
            "request_kw": {
                "solar_to_battery": float(self.solar_to_battery_kw),
                "solar_to_electrolyser": float(self.solar_to_electrolyser_kw),
                "solar_to_dac": float(self.solar_to_dac_kw),
                "solar_to_methanator": float(self.solar_to_methanator_kw),
                "battery_support": float(self.battery_support_kw),
                "unallocated_solar": float(self.unallocated_solar_kw),
                "electrolyser_total": float(self.electrolyser_offered_kw),
                "electrolyser_shortfall": float(self.electrolyser_shortfall_kw),
            },
            "electrolyser_target_reachable": self.electrolyser_target_reachable,
            "action": self.action.to_dict(),
        }

    @property
    def sha256(self) -> str:
        return domain_sha256(SERVICE_MPC_MINUTE_DISPATCH_DOMAIN, self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


def dispatch_mpc_setpoints(
    setpoints: MPCProcessSetpointsV1,
    boundary: PlantState,
    *,
    solar_available_kw: float,
    constraints: Constraints,
    config: PlantConfig,
) -> MPCMinuteDispatchV1:
    """Realize one held command against exactly one contemporaneous solar bus."""

    if not isinstance(setpoints, MPCProcessSetpointsV1):
        raise TypeError("setpoints must be MPCProcessSetpointsV1")
    if not isinstance(boundary, PlantState):
        raise TypeError("boundary must be PlantState")
    solar_kw = _finite("solar_available_kw", solar_available_kw, minimum=0.0)
    active_envelope = resolve_setpoint_envelope(
        constraints=constraints,
        config=config,
    )
    if active_envelope.sha256 != setpoints.envelope.sha256:
        raise ValueError("active setpoint envelope changed during the held command")
    support = battery_support_availability(
        boundary,
        constraints=constraints,
        config=config,
        electrolyser_headroom_kw=active_envelope.electrolyser_limit_kw,
    )

    allocation = _allocate_dispatch(
        solar_kw=solar_kw,
        electrolyser_target_kw=setpoints.electrolyser_target_kw,
        dac_target_kw=setpoints.dac_target_kw,
        methanator_target_kw=setpoints.methanator_target_kw,
        battery_charge_ceiling_kw=setpoints.battery_charge_ceiling_kw,
        support=support,
    )
    routes = allocation.routes_kw
    return MPCMinuteDispatchV1(
        setpoint_sha256=setpoints.sha256,
        decision_index=setpoints.decision_index,
        boundary_sha256=minute_boundary_sha256(boundary),
        boundary_step=boundary.step,
        solar_available_kw=solar_kw,
        electrolyser_target_kw=setpoints.electrolyser_target_kw,
        dac_target_kw=setpoints.dac_target_kw,
        methanator_target_kw=setpoints.methanator_target_kw,
        battery_charge_ceiling_kw=setpoints.battery_charge_ceiling_kw,
        support_availability=support,
        solar_to_battery_kw=routes[0],
        solar_to_electrolyser_kw=routes[1],
        solar_to_dac_kw=routes[2],
        solar_to_methanator_kw=routes[3],
        battery_support_kw=allocation.support_kw,
        unallocated_solar_kw=allocation.unallocated_kw,
        electrolyser_target_reachable=allocation.target_reachable,
        action=allocation.action,
    )


__all__ = [
    "MPCMinuteDispatchV1",
    "MPCProcessSetpointsV1",
    "MPCSetpointEnvelopeV1",
    "RESERVE_AWARE_DISPATCH_POLICY",
    "SERVICE_MPC_FIRST_INTERVAL_DOMAIN",
    "SERVICE_MPC_DISPATCH_BOUNDARY_DOMAIN",
    "SERVICE_MPC_MINUTE_DISPATCH_DOMAIN",
    "SERVICE_MPC_MINUTE_DISPATCH_SCHEMA",
    "SERVICE_MPC_PROCESS_SETPOINT_DOMAIN",
    "SERVICE_MPC_PROCESS_SETPOINT_SCHEMA",
    "SERVICE_MPC_SETPOINT_ENVELOPE_DOMAIN",
    "dispatch_mpc_setpoints",
    "minute_boundary_sha256",
    "projected_first_interval_sha256",
    "realize_first_mpc_setpoints",
    "resolve_setpoint_envelope",
]

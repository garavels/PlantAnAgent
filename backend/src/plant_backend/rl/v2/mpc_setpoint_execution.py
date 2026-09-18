"""Fail-closed minute and interval evidence for absolute MPC setpoints."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, ClassVar

from plant_sim import (
    Action,
    ElectrolyserCommitmentSnapshot,
    PlantState,
)

from ...safety.validation import ValidatedAction
from .mpc_schema import (
    MODEL_BALANCE_TOLERANCE,
    canonical_json,
    domain_sha256,
    require_sha256,
)
from .mpc_setpoints import (
    MPCMinuteDispatchV1,
    MPCProcessSetpointsV1,
    SERVICE_MPC_DISPATCH_BOUNDARY_DOMAIN,
    minute_boundary_sha256,
)

SERVICE_MPC_COMMITMENT_BOUNDARY_DOMAIN = "service-mpc-minute-commitment-v1"
SERVICE_MPC_MINUTE_APPLICATION_SCHEMA = "service-mpc-minute-application-v1"
SERVICE_MPC_MINUTE_APPLICATION_DOMAIN = SERVICE_MPC_MINUTE_APPLICATION_SCHEMA
SERVICE_MPC_SETPOINT_INTERVAL_SCHEMA = "service-mpc-setpoint-interval-execution-v1"
SERVICE_MPC_SETPOINT_INTERVAL_DOMAIN = SERVICE_MPC_SETPOINT_INTERVAL_SCHEMA

_COMMITMENT_MODES = frozenset(
    {
        "disabled",
        "off",
        "cold_starting",
        "running",
        "hot_standby",
        "hot_starting",
        "cooldown",
    }
)


def _finite(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}")
    return number


def _index(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _close(left: float, right: float) -> bool:
    scale = max(1.0, abs(left), abs(right))
    tolerance = MODEL_BALANCE_TOLERANCE + MODEL_BALANCE_TOLERANCE * scale
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _canonical_object(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be canonical JSON text")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is invalid JSON") from exc
    if not isinstance(decoded, dict) or canonical_json(decoded) != value:
        raise ValueError(f"{name} must be a canonical JSON object")
    return decoded


def _boundary_record_sha256(record: dict[str, Any]) -> str:
    return str(domain_sha256(SERVICE_MPC_DISPATCH_BOUNDARY_DOMAIN, record))


def commitment_boundary_sha256(
    snapshot: ElectrolyserCommitmentSnapshot | None,
) -> str:
    if snapshot is not None and not isinstance(
        snapshot, ElectrolyserCommitmentSnapshot
    ):
        raise TypeError("commitment snapshot must be typed or None")
    body = None if snapshot is None else snapshot.to_dict()
    return str(domain_sha256(SERVICE_MPC_COMMITMENT_BOUNDARY_DOMAIN, body))


@dataclass(frozen=True, slots=True)
class MPCMinuteApplicationV1:
    """One dispatch -> guard -> plant -> next-boundary causal evidence link."""

    ordinal: int
    dispatch: MPCMinuteDispatchV1
    guarded_action: Action
    adjustments: tuple[str, ...]
    boundary_json: str
    next_boundary_json: str
    commitment_before: ElectrolyserCommitmentSnapshot | None
    commitment_after: ElectrolyserCommitmentSnapshot | None
    h2_headroom_before_kg: float
    co2_headroom_before_kg: float
    ch4_headroom_before_kg: float
    battery_charge_headroom_before_kwh: float
    applied_solar_available_kw: float
    applied_solar_to_battery_kw: float
    applied_solar_to_electrolyser_kw: float
    applied_solar_to_dac_kw: float
    applied_solar_to_methanator_kw: float
    applied_battery_support_kw: float
    applied_electrolyser_kw: float
    applied_dac_kw: float
    applied_methanator_kw: float
    applied_solar_curtailed_kw: float

    schema: ClassVar[str] = SERVICE_MPC_MINUTE_APPLICATION_SCHEMA

    def __post_init__(self) -> None:
        _index("minute application ordinal", self.ordinal)
        if not isinstance(self.dispatch, MPCMinuteDispatchV1):
            raise TypeError("minute application requires typed dispatch evidence")
        if not isinstance(self.guarded_action, Action):
            raise TypeError("guarded_action must be Action")
        if Action.sanitised(self.guarded_action) != self.guarded_action:
            raise ValueError("guarded minute action is not a sanitizer fixed point")
        if not isinstance(self.adjustments, tuple) or any(
            not isinstance(item, str) or not item for item in self.adjustments
        ):
            raise ValueError("minute adjustments must be an immutable text tuple")
        boundary = _canonical_object("boundary_json", self.boundary_json)
        next_boundary = _canonical_object("next_boundary_json", self.next_boundary_json)
        boundary_step = _index("boundary step", boundary.get("step"))
        next_step = _index("next boundary step", next_boundary.get("step"))
        if (
            boundary_step != self.dispatch.boundary_step
            or _boundary_record_sha256(boundary) != self.dispatch.boundary_sha256
        ):
            raise ValueError("minute application boundary differs from its dispatch")
        if next_step != self.dispatch.boundary_step + 1:
            raise ValueError("minute application boundary steps are not contiguous")
        for name in ("commitment_before", "commitment_after"):
            snapshot = getattr(self, name)
            if snapshot is not None and not isinstance(
                snapshot, ElectrolyserCommitmentSnapshot
            ):
                raise TypeError(f"{name} must be a typed commitment snapshot or None")
            if _commitment_mode(snapshot) not in _COMMITMENT_MODES:
                raise ValueError(f"{name} mode is invalid")
            commitment_boundary_sha256(snapshot)
        for name in (
            "h2_headroom_before_kg",
            "co2_headroom_before_kg",
            "ch4_headroom_before_kg",
            "battery_charge_headroom_before_kwh",
            "applied_solar_available_kw",
            "applied_solar_to_battery_kw",
            "applied_solar_to_electrolyser_kw",
            "applied_solar_to_dac_kw",
            "applied_solar_to_methanator_kw",
            "applied_battery_support_kw",
            "applied_electrolyser_kw",
            "applied_dac_kw",
            "applied_methanator_kw",
            "applied_solar_curtailed_kw",
        ):
            _finite(name, getattr(self, name), minimum=0.0)

        if not _close(
            self.applied_solar_available_kw,
            self.dispatch.solar_available_kw,
        ):
            raise ValueError("applied solar differs from the dispatch bus")
        solar_total = math.fsum(
            (
                self.applied_solar_to_battery_kw,
                self.applied_solar_to_electrolyser_kw,
                self.applied_solar_to_dac_kw,
                self.applied_solar_to_methanator_kw,
                self.applied_solar_curtailed_kw,
            )
        )
        if not _close(solar_total, self.applied_solar_available_kw):
            raise ValueError("minute applied solar bus does not balance")
        if not _close(
            self.applied_electrolyser_kw,
            self.applied_solar_to_electrolyser_kw + self.applied_battery_support_kw,
        ):
            raise ValueError("minute applied electrolyser bus does not balance")

        guarded_solar_electrolyser_kw = (
            self.guarded_action.electrolyser_share * self.applied_solar_available_kw
        )
        guarded_electrolyser_kw = (
            guarded_solar_electrolyser_kw + self.guarded_action.battery_support_kw
        )
        if (
            self.guarded_action.electrolyser_share
            > self.dispatch.action.electrolyser_share + MODEL_BALANCE_TOLERANCE
            or self.guarded_action.dac_share
            > self.dispatch.action.dac_share + MODEL_BALANCE_TOLERANCE
            or self.guarded_action.methanator_share
            > self.dispatch.action.methanator_share + MODEL_BALANCE_TOLERANCE
            or self.guarded_action.battery_support_kw
            > self.dispatch.action.battery_support_kw + MODEL_BALANCE_TOLERANCE
        ):
            raise ValueError("guard increased process authority")
        if (
            guarded_electrolyser_kw
            > self.dispatch.electrolyser_target_kw + MODEL_BALANCE_TOLERANCE
        ):
            raise ValueError("guarded offer exceeds the electrolyser target")
        if (
            self.guarded_action.battery_support_kw
            > max(
                0.0,
                self.dispatch.electrolyser_target_kw - guarded_solar_electrolyser_kw,
            )
            + MODEL_BALANCE_TOLERANCE
        ):
            raise ValueError("guarded support exceeds the guarded target shortfall")
        if (
            self.applied_battery_support_kw
            > self.guarded_action.battery_support_kw + MODEL_BALANCE_TOLERANCE
        ):
            raise ValueError("applied support exceeds guarded support")
        if (
            self.applied_electrolyser_kw
            > guarded_electrolyser_kw + MODEL_BALANCE_TOLERANCE
        ):
            raise ValueError("plant electrolyser draw exceeds guarded authority")
        if (
            self.applied_electrolyser_kw
            > self.dispatch.electrolyser_target_kw + MODEL_BALANCE_TOLERANCE
        ):
            raise ValueError("plant electrolyser draw exceeds its absolute target")
        if (
            self.applied_dac_kw
            > self.guarded_action.dac_share * self.applied_solar_available_kw
            + MODEL_BALANCE_TOLERANCE
            or self.applied_dac_kw
            > self.dispatch.dac_target_kw + MODEL_BALANCE_TOLERANCE
        ):
            raise ValueError("plant DAC draw exceeds guarded target authority")
        if (
            self.applied_methanator_kw
            > self.guarded_action.methanator_share * self.applied_solar_available_kw
            + MODEL_BALANCE_TOLERANCE
            or self.applied_methanator_kw
            > self.dispatch.methanator_target_kw + MODEL_BALANCE_TOLERANCE
        ):
            raise ValueError("plant methanator draw exceeds guarded target authority")

    @property
    def boundary_record(self) -> dict[str, Any]:
        return _canonical_object("boundary_json", self.boundary_json)

    @property
    def boundary_sha256(self) -> str:
        return _boundary_record_sha256(self.boundary_record)

    @property
    def boundary_step(self) -> int:
        return _index("boundary step", self.boundary_record.get("step"))

    @property
    def next_boundary_record(self) -> dict[str, Any]:
        return _canonical_object("next_boundary_json", self.next_boundary_json)

    @property
    def next_boundary_sha256(self) -> str:
        return _boundary_record_sha256(self.next_boundary_record)

    @property
    def next_boundary_step(self) -> int:
        return _index("next boundary step", self.next_boundary_record.get("step"))

    @property
    def commitment_before_sha256(self) -> str:
        return commitment_boundary_sha256(self.commitment_before)

    @property
    def commitment_after_sha256(self) -> str:
        return commitment_boundary_sha256(self.commitment_after)

    @property
    def commitment_mode_before(self) -> str:
        return _commitment_mode(self.commitment_before)

    @property
    def commitment_mode_after(self) -> str:
        return _commitment_mode(self.commitment_after)

    @property
    def guarded_electrolyser_kw(self) -> float:
        return float(
            self.guarded_action.electrolyser_share * self.applied_solar_available_kw
            + self.guarded_action.battery_support_kw
        )

    @property
    def guard_changed_request(self) -> bool:
        return self.guarded_action != self.dispatch.action or bool(self.adjustments)

    @property
    def plant_refused_or_rerouted_power(self) -> bool:
        guarded = (
            self.guarded_action.battery_share * self.applied_solar_available_kw,
            self.guarded_electrolyser_kw,
            self.guarded_action.dac_share * self.applied_solar_available_kw,
            self.guarded_action.methanator_share * self.applied_solar_available_kw,
        )
        applied = (
            self.applied_solar_to_battery_kw,
            self.applied_electrolyser_kw,
            self.applied_dac_kw,
            self.applied_methanator_kw,
        )
        return any(
            not math.isclose(
                expected,
                observed,
                rel_tol=0.0,
                abs_tol=MODEL_BALANCE_TOLERANCE,
            )
            for expected, observed in zip(guarded, applied, strict=True)
        )

    def body(self) -> dict[str, Any]:
        target = self.dispatch.electrolyser_target_kw
        guarded_solar_battery_kw = (
            self.guarded_action.battery_share * self.applied_solar_available_kw
        )
        guarded_solar_electrolyser_kw = (
            self.guarded_action.electrolyser_share * self.applied_solar_available_kw
        )
        guarded_solar_dac_kw = (
            self.guarded_action.dac_share * self.applied_solar_available_kw
        )
        guarded_solar_methanator_kw = (
            self.guarded_action.methanator_share * self.applied_solar_available_kw
        )
        return {
            "schema": self.schema,
            "ordinal": int(self.ordinal),
            "boundary": {
                "record": self.boundary_record,
                "sha256": self.boundary_sha256,
                "step": int(self.boundary_step),
            },
            "dispatch": self.dispatch.to_dict(),
            "guard": {
                "action": self.guarded_action.to_dict(),
                "adjustments": list(self.adjustments),
                "changed_request": self.guard_changed_request,
                "electrolyser_offered_kw": float(self.guarded_electrolyser_kw),
                "electrolyser_error_kw": float(self.guarded_electrolyser_kw - target),
                "offered_kw": {
                    "solar_to_battery": float(guarded_solar_battery_kw),
                    "solar_to_electrolyser": float(guarded_solar_electrolyser_kw),
                    "solar_to_dac": float(guarded_solar_dac_kw),
                    "solar_to_methanator": float(guarded_solar_methanator_kw),
                    "battery_support": float(self.guarded_action.battery_support_kw),
                    "electrolyser_total": float(self.guarded_electrolyser_kw),
                },
            },
            "next_boundary": {
                "record": self.next_boundary_record,
                "sha256": self.next_boundary_sha256,
                "step": int(self.next_boundary_step),
            },
            "commitment": {
                "before": {
                    "record": (
                        None
                        if self.commitment_before is None
                        else self.commitment_before.to_dict()
                    ),
                    "sha256": self.commitment_before_sha256,
                    "mode": self.commitment_mode_before,
                },
                "after": {
                    "record": (
                        None
                        if self.commitment_after is None
                        else self.commitment_after.to_dict()
                    ),
                    "sha256": self.commitment_after_sha256,
                    "mode": self.commitment_mode_after,
                },
            },
            "storage_headroom_before": {
                "h2_kg": float(self.h2_headroom_before_kg),
                "co2_kg": float(self.co2_headroom_before_kg),
                "ch4_kg": float(self.ch4_headroom_before_kg),
                "battery_charge_kwh": float(self.battery_charge_headroom_before_kwh),
            },
            "applied_kw": {
                "solar_available": float(self.applied_solar_available_kw),
                "solar_to_battery": float(self.applied_solar_to_battery_kw),
                "solar_to_electrolyser": float(self.applied_solar_to_electrolyser_kw),
                "solar_to_dac": float(self.applied_solar_to_dac_kw),
                "solar_to_methanator": float(self.applied_solar_to_methanator_kw),
                "battery_support": float(self.applied_battery_support_kw),
                "electrolyser": float(self.applied_electrolyser_kw),
                "dac": float(self.applied_dac_kw),
                "methanator": float(self.applied_methanator_kw),
                "solar_curtailed": float(self.applied_solar_curtailed_kw),
                "electrolyser_error": float(self.applied_electrolyser_kw - target),
            },
            "plant_gap_kw": {
                "solar_to_battery": float(
                    self.applied_solar_to_battery_kw - guarded_solar_battery_kw
                ),
                "solar_to_electrolyser": float(
                    self.applied_solar_to_electrolyser_kw
                    - guarded_solar_electrolyser_kw
                ),
                "solar_to_dac": float(self.applied_dac_kw - guarded_solar_dac_kw),
                "solar_to_methanator": float(
                    self.applied_methanator_kw - guarded_solar_methanator_kw
                ),
                "battery_support": float(
                    self.applied_battery_support_kw
                    - self.guarded_action.battery_support_kw
                ),
                "electrolyser_total": float(
                    self.applied_electrolyser_kw - self.guarded_electrolyser_kw
                ),
            },
            "plant_refused_or_rerouted_power": self.plant_refused_or_rerouted_power,
        }

    @property
    def sha256(self) -> str:
        return str(domain_sha256(SERVICE_MPC_MINUTE_APPLICATION_DOMAIN, self.body()))

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


def _commitment_mode(snapshot: ElectrolyserCommitmentSnapshot | None) -> str:
    return "disabled" if snapshot is None else str(snapshot.mode.value)


def record_minute_application(
    *,
    ordinal: int,
    before: PlantState,
    dispatch: MPCMinuteDispatchV1,
    validated: ValidatedAction,
    after: PlantState,
    next_boundary: PlantState,
    commitment_before: ElectrolyserCommitmentSnapshot | None,
    commitment_after: ElectrolyserCommitmentSnapshot | None,
) -> MPCMinuteApplicationV1:
    """Construct one application link from independently owned runtime values."""

    if not isinstance(before, PlantState) or not isinstance(after, PlantState):
        raise TypeError("minute application plant boundaries must be PlantState")
    if not isinstance(next_boundary, PlantState):
        raise TypeError("minute application next boundary must be PlantState")
    if not isinstance(dispatch, MPCMinuteDispatchV1):
        raise TypeError("dispatch must be MPCMinuteDispatchV1")
    if not isinstance(validated, ValidatedAction):
        raise TypeError("validated must be ValidatedAction")
    if minute_boundary_sha256(before) != dispatch.boundary_sha256:
        raise ValueError("minute dispatch does not belong to its before boundary")
    if before.step != dispatch.boundary_step:
        raise ValueError("minute dispatch step does not match its before boundary")
    if after.step != next_boundary.step or after.step != before.step + 1:
        raise ValueError("minute application after boundaries are not contiguous")
    if after.action != validated.action:
        raise ValueError("simulator action differs from the guarded action")
    return MPCMinuteApplicationV1(
        ordinal=_index("ordinal", ordinal),
        dispatch=dispatch,
        guarded_action=validated.action,
        adjustments=validated.adjustments,
        boundary_json=canonical_json(before.to_dict()),
        next_boundary_json=canonical_json(next_boundary.to_dict()),
        commitment_before=commitment_before,
        commitment_after=commitment_after,
        h2_headroom_before_kg=max(0.0, before.h2_capacity_kg - before.h2_level_kg),
        co2_headroom_before_kg=max(0.0, before.co2_capacity_kg - before.co2_level_kg),
        ch4_headroom_before_kg=max(0.0, before.ch4_capacity_kg - before.ch4_level_kg),
        battery_charge_headroom_before_kwh=max(
            0.0, before.battery_capacity_kwh - before.battery_energy_kwh
        ),
        applied_solar_available_kw=after.solar_available_kw,
        applied_solar_to_battery_kw=after.solar_to_battery_kw,
        applied_solar_to_electrolyser_kw=after.solar_to_electrolyser_kw,
        applied_solar_to_dac_kw=after.solar_to_dac_kw,
        applied_solar_to_methanator_kw=after.solar_to_methanator_kw,
        applied_battery_support_kw=after.battery_to_electrolyser_kw,
        applied_electrolyser_kw=after.electrolyser_power_kw,
        applied_dac_kw=after.dac_power_kw,
        applied_methanator_kw=after.methanator_power_kw,
        applied_solar_curtailed_kw=after.solar_curtailed_kw,
    )


@dataclass(frozen=True, slots=True)
class MPCSetpointIntervalExecutionV1:
    """Complete ordered physical realization of one held setpoint command."""

    setpoints: MPCProcessSetpointsV1
    physics_step_s: float
    applications: tuple[MPCMinuteApplicationV1, ...]

    schema: ClassVar[str] = SERVICE_MPC_SETPOINT_INTERVAL_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.setpoints, MPCProcessSetpointsV1):
            raise TypeError("interval execution requires typed setpoints")
        step_s = _finite("physics_step_s", self.physics_step_s, minimum=0.0)
        if step_s <= 0.0:
            raise ValueError("physics_step_s must be positive")
        expected_count = self.setpoints.hold_duration_s / step_s
        if not math.isclose(
            expected_count,
            round(expected_count),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("setpoint hold does not contain whole physics steps")
        if not isinstance(self.applications, tuple) or len(self.applications) != int(
            round(expected_count)
        ):
            raise ValueError("interval applications do not cover the setpoint hold")
        if any(
            not isinstance(item, MPCMinuteApplicationV1) or item.ordinal != index
            for index, item in enumerate(self.applications)
        ):
            raise ValueError("interval applications are not typed and contiguous")
        command_sha = self.setpoints.sha256
        for index, application in enumerate(self.applications):
            dispatch = application.dispatch
            if (
                dispatch.setpoint_sha256 != command_sha
                or dispatch.decision_index != self.setpoints.decision_index
            ):
                raise ValueError("minute application belongs to another setpoint")
            if index > 0:
                prior = self.applications[index - 1]
                if (
                    dispatch.boundary_step != prior.next_boundary_step
                    or dispatch.boundary_sha256 != prior.next_boundary_sha256
                    or application.boundary_json != prior.next_boundary_json
                    or application.commitment_before_sha256
                    != prior.commitment_after_sha256
                    or application.commitment_before != prior.commitment_after
                ):
                    raise ValueError("minute application chain is broken")

    @property
    def hours(self) -> float:
        return len(self.applications) * self.physics_step_s / 3_600.0

    def _energy(self, values: tuple[float, ...]) -> float:
        return math.fsum(values) * self.physics_step_s / 3_600.0

    def body(self) -> dict[str, Any]:
        applications = self.applications
        hold_h = self.hours
        setpoints = self.setpoints

        planned_kwh = {
            "solar_available": setpoints.planned_forecast_solar_kw * hold_h,
            "solar_to_battery": setpoints.planned_solar_to_battery_kw * hold_h,
            "solar_to_electrolyser": (
                setpoints.planned_solar_to_electrolyser_kw * hold_h
            ),
            "solar_to_dac": setpoints.planned_solar_to_dac_kw * hold_h,
            "solar_to_methanator": (setpoints.planned_solar_to_methanator_kw * hold_h),
            "battery_support": setpoints.expected_battery_support_kw * hold_h,
            "solar_curtailed": setpoints.planned_solar_curtailed_kw * hold_h,
            "electrolyser_total": setpoints.electrolyser_target_kw * hold_h,
        }
        target_kwh = {
            "battery_charge_ceiling": setpoints.battery_charge_ceiling_kw * hold_h,
            "electrolyser": setpoints.electrolyser_target_kw * hold_h,
            "dac": setpoints.dac_target_kw * hold_h,
            "methanator": setpoints.methanator_target_kw * hold_h,
        }
        requested_kwh = {
            "solar_available": self._energy(
                tuple(item.dispatch.solar_available_kw for item in applications)
            ),
            "solar_to_battery": self._energy(
                tuple(item.dispatch.solar_to_battery_kw for item in applications)
            ),
            "solar_to_electrolyser": self._energy(
                tuple(item.dispatch.solar_to_electrolyser_kw for item in applications)
            ),
            "solar_to_dac": self._energy(
                tuple(item.dispatch.solar_to_dac_kw for item in applications)
            ),
            "solar_to_methanator": self._energy(
                tuple(item.dispatch.solar_to_methanator_kw for item in applications)
            ),
            "battery_support": self._energy(
                tuple(item.dispatch.battery_support_kw for item in applications)
            ),
            "unallocated_solar": self._energy(
                tuple(item.dispatch.unallocated_solar_kw for item in applications)
            ),
            "electrolyser_total": self._energy(
                tuple(item.dispatch.electrolyser_offered_kw for item in applications)
            ),
        }
        guarded_kwh = {
            "solar_available": requested_kwh["solar_available"],
            "solar_to_battery": self._energy(
                tuple(
                    item.guarded_action.battery_share * item.applied_solar_available_kw
                    for item in applications
                )
            ),
            "solar_to_electrolyser": self._energy(
                tuple(
                    item.guarded_action.electrolyser_share
                    * item.applied_solar_available_kw
                    for item in applications
                )
            ),
            "solar_to_dac": self._energy(
                tuple(
                    item.guarded_action.dac_share * item.applied_solar_available_kw
                    for item in applications
                )
            ),
            "solar_to_methanator": self._energy(
                tuple(
                    item.guarded_action.methanator_share
                    * item.applied_solar_available_kw
                    for item in applications
                )
            ),
            "battery_support": self._energy(
                tuple(item.guarded_action.battery_support_kw for item in applications)
            ),
            "electrolyser_total": self._energy(
                tuple(item.guarded_electrolyser_kw for item in applications)
            ),
        }
        guarded_kwh["unallocated_solar"] = max(
            0.0,
            guarded_kwh["solar_available"]
            - math.fsum(
                guarded_kwh[name]
                for name in (
                    "solar_to_battery",
                    "solar_to_electrolyser",
                    "solar_to_dac",
                    "solar_to_methanator",
                )
            ),
        )
        applied_kwh = {
            "solar_available": self._energy(
                tuple(item.applied_solar_available_kw for item in applications)
            ),
            "solar_to_battery": self._energy(
                tuple(item.applied_solar_to_battery_kw for item in applications)
            ),
            "solar_to_electrolyser": self._energy(
                tuple(item.applied_solar_to_electrolyser_kw for item in applications)
            ),
            "solar_to_dac": self._energy(
                tuple(item.applied_solar_to_dac_kw for item in applications)
            ),
            "solar_to_methanator": self._energy(
                tuple(item.applied_solar_to_methanator_kw for item in applications)
            ),
            "battery_support": self._energy(
                tuple(item.applied_battery_support_kw for item in applications)
            ),
            "electrolyser_total": self._energy(
                tuple(item.applied_electrolyser_kw for item in applications)
            ),
            "dac": self._energy(tuple(item.applied_dac_kw for item in applications)),
            "methanator": self._energy(
                tuple(item.applied_methanator_kw for item in applications)
            ),
            "solar_curtailed": self._energy(
                tuple(item.applied_solar_curtailed_kw for item in applications)
            ),
        }
        applied_errors = tuple(
            item.applied_electrolyser_kw - item.dispatch.electrolyser_target_kw
            for item in applications
        )
        requested_errors = tuple(
            item.dispatch.electrolyser_offered_kw - item.dispatch.electrolyser_target_kw
            for item in applications
        )
        guarded_errors = tuple(
            item.guarded_electrolyser_kw - item.dispatch.electrolyser_target_kw
            for item in applications
        )
        return {
            "schema": self.schema,
            "setpoints": self.setpoints.to_dict(),
            "physics_step_s": float(self.physics_step_s),
            "physical_steps": len(applications),
            "hours": float(self.hours),
            "applications": [item.to_dict() for item in applications],
            "application_sha256": [item.sha256 for item in applications],
            "energy_kwh": {
                "planned": {name: float(value) for name, value in planned_kwh.items()},
                "target": {name: float(value) for name, value in target_kwh.items()},
                "requested": {
                    name: float(value) for name, value in requested_kwh.items()
                },
                "guarded": {name: float(value) for name, value in guarded_kwh.items()},
                "applied": {name: float(value) for name, value in applied_kwh.items()},
                "electrolyser_target_error": {
                    "requested_overshoot": float(
                        self._energy(
                            tuple(max(0.0, value) for value in requested_errors)
                        )
                    ),
                    "requested_undershoot": float(
                        self._energy(
                            tuple(max(0.0, -value) for value in requested_errors)
                        )
                    ),
                    "guarded_overshoot": float(
                        self._energy(tuple(max(0.0, value) for value in guarded_errors))
                    ),
                    "guarded_undershoot": float(
                        self._energy(
                            tuple(max(0.0, -value) for value in guarded_errors)
                        )
                    ),
                    "applied_overshoot": float(
                        self._energy(tuple(max(0.0, value) for value in applied_errors))
                    ),
                    "applied_undershoot": float(
                        self._energy(
                            tuple(max(0.0, -value) for value in applied_errors)
                        )
                    ),
                },
            },
            "instantaneous_electrolyser_target_error_kw": {
                "requested_max_overshoot": float(max(0.0, max(requested_errors))),
                "requested_max_undershoot": float(max(0.0, -min(requested_errors))),
                "guarded_max_overshoot": float(max(0.0, max(guarded_errors))),
                "guarded_max_undershoot": float(max(0.0, -min(guarded_errors))),
                "applied_max_overshoot": float(max(0.0, max(applied_errors))),
                "applied_max_undershoot": float(max(0.0, -min(applied_errors))),
            },
            "counts": {
                "target_reachable": sum(
                    item.dispatch.electrolyser_target_reachable for item in applications
                ),
                "guard_changed_request": sum(
                    item.guard_changed_request for item in applications
                ),
                "plant_refused_or_rerouted_power": sum(
                    item.plant_refused_or_rerouted_power for item in applications
                ),
            },
            "terminal": {
                "boundary_sha256": applications[-1].next_boundary_sha256,
                "boundary_step": int(applications[-1].next_boundary_step),
                "commitment_sha256": applications[-1].commitment_after_sha256,
                "commitment_mode": applications[-1].commitment_mode_after,
            },
        }

    @property
    def sha256(self) -> str:
        return str(domain_sha256(SERVICE_MPC_SETPOINT_INTERVAL_DOMAIN, self.body()))

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


__all__ = [
    "MPCMinuteApplicationV1",
    "MPCSetpointIntervalExecutionV1",
    "SERVICE_MPC_COMMITMENT_BOUNDARY_DOMAIN",
    "SERVICE_MPC_MINUTE_APPLICATION_DOMAIN",
    "SERVICE_MPC_MINUTE_APPLICATION_SCHEMA",
    "SERVICE_MPC_SETPOINT_INTERVAL_DOMAIN",
    "SERVICE_MPC_SETPOINT_INTERVAL_SCHEMA",
    "commitment_boundary_sha256",
    "record_minute_application",
]

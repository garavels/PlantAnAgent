"""Typed dispatch, guard, and plant evidence for common setpoint commands."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, ClassVar

from plant_sim import Action, PlantState

from ...safety.validation import ValidatedAction
from .mpc_schema import MODEL_BALANCE_TOLERANCE, domain_sha256, require_sha256
from .setpoint_command import (
    ProcessSetpointCommandV1,
    ProcessSetpointMinuteDispatchV1,
    process_setpoint_boundary_sha256,
)


PROCESS_SETPOINT_MINUTE_APPLICATION_SCHEMA = "process-setpoint-minute-application-v1"
PROCESS_SETPOINT_MINUTE_APPLICATION_DOMAIN = PROCESS_SETPOINT_MINUTE_APPLICATION_SCHEMA
PROCESS_SETPOINT_INTERVAL_EXECUTION_SCHEMA = "process-setpoint-interval-execution-v1"
PROCESS_SETPOINT_INTERVAL_EXECUTION_DOMAIN = PROCESS_SETPOINT_INTERVAL_EXECUTION_SCHEMA


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


@dataclass(frozen=True, slots=True)
class ProcessSetpointMinuteApplicationV1:
    """One common dispatch crossing the guard and physical plant."""

    ordinal: int
    dispatch: ProcessSetpointMinuteDispatchV1
    guarded_action: Action
    adjustments: tuple[str, ...]
    next_boundary_sha256: str
    next_boundary_step: int
    applied_action: Action
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

    schema: ClassVar[str] = PROCESS_SETPOINT_MINUTE_APPLICATION_SCHEMA

    def __post_init__(self) -> None:
        _index("application ordinal", self.ordinal)
        if not isinstance(self.dispatch, ProcessSetpointMinuteDispatchV1):
            raise TypeError("application dispatch must be typed common dispatch")
        for name in ("guarded_action", "applied_action"):
            action = getattr(self, name)
            if not isinstance(action, Action) or Action.sanitised(action) != action:
                raise ValueError(f"{name} must be a sanitized Action")
        if self.applied_action != self.guarded_action:
            raise ValueError("plant action differs from the guarded action")
        if not isinstance(self.adjustments, tuple) or any(
            not isinstance(item, str) or not item for item in self.adjustments
        ):
            raise ValueError("adjustments must be an immutable text tuple")
        require_sha256("next_boundary_sha256", self.next_boundary_sha256)
        next_step = _index("next_boundary_step", self.next_boundary_step)
        if next_step != self.dispatch.boundary_step + 1:
            raise ValueError("application boundaries are not contiguous")
        for name in (
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
            self.dispatch.allocation.solar_available_kw,
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
            raise ValueError("applied solar bus does not balance")
        if not _close(
            self.applied_electrolyser_kw,
            self.applied_solar_to_electrolyser_kw + self.applied_battery_support_kw,
        ):
            raise ValueError("applied electrolyser bus does not balance")

        solar = self.applied_solar_available_kw
        guarded = self.guarded_action
        requested = self.dispatch.action
        if (
            guarded.electrolyser_share
            > requested.electrolyser_share + MODEL_BALANCE_TOLERANCE
            or guarded.dac_share > requested.dac_share + MODEL_BALANCE_TOLERANCE
            or guarded.methanator_share
            > requested.methanator_share + MODEL_BALANCE_TOLERANCE
            or guarded.battery_support_kw
            > requested.battery_support_kw + MODEL_BALANCE_TOLERANCE
        ):
            raise ValueError("guard increased common process authority")
        guarded_solar_electrolyser_kw = guarded.electrolyser_share * solar
        guarded_electrolyser_kw = (
            guarded_solar_electrolyser_kw + guarded.battery_support_kw
        )
        targets = self.dispatch.targets
        if (
            guarded_electrolyser_kw
            > targets.electrolyser_target_kw + MODEL_BALANCE_TOLERANCE
        ):
            raise ValueError("guarded offer exceeds the electrolyser target")
        if (
            guarded.battery_support_kw
            > max(
                0.0,
                targets.electrolyser_target_kw - guarded_solar_electrolyser_kw,
            )
            + MODEL_BALANCE_TOLERANCE
        ):
            raise ValueError("guarded support exceeds the target shortfall")
        if (
            self.applied_battery_support_kw
            > guarded.battery_support_kw + MODEL_BALANCE_TOLERANCE
            or self.applied_electrolyser_kw
            > guarded_electrolyser_kw + MODEL_BALANCE_TOLERANCE
            or self.applied_electrolyser_kw
            > targets.electrolyser_target_kw + MODEL_BALANCE_TOLERANCE
            or self.applied_dac_kw > guarded.dac_share * solar + MODEL_BALANCE_TOLERANCE
            or self.applied_dac_kw > targets.dac_target_kw + MODEL_BALANCE_TOLERANCE
            or self.applied_methanator_kw
            > guarded.methanator_share * solar + MODEL_BALANCE_TOLERANCE
            or self.applied_methanator_kw
            > targets.methanator_target_kw + MODEL_BALANCE_TOLERANCE
        ):
            raise ValueError("plant process draw exceeds guarded target authority")

    @property
    def boundary_sha256(self) -> str:
        return str(self.dispatch.boundary_sha256)

    @property
    def boundary_step(self) -> int:
        return int(self.dispatch.boundary_step)

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "ordinal": int(self.ordinal),
            "dispatch": self.dispatch.to_dict(),
            "guard": {
                "action": self.guarded_action.to_dict(),
                "adjustments": list(self.adjustments),
                "changed_request": (
                    self.guarded_action != self.dispatch.action
                    or bool(self.adjustments)
                ),
            },
            "next_boundary": {
                "sha256": self.next_boundary_sha256,
                "step": int(self.next_boundary_step),
            },
            "plant": {
                "action": self.applied_action.to_dict(),
                "applied_kw": {
                    "solar_available": float(self.applied_solar_available_kw),
                    "solar_to_battery": float(self.applied_solar_to_battery_kw),
                    "solar_to_electrolyser": float(
                        self.applied_solar_to_electrolyser_kw
                    ),
                    "solar_to_dac": float(self.applied_solar_to_dac_kw),
                    "solar_to_methanator": float(self.applied_solar_to_methanator_kw),
                    "battery_support": float(self.applied_battery_support_kw),
                    "electrolyser": float(self.applied_electrolyser_kw),
                    "dac": float(self.applied_dac_kw),
                    "methanator": float(self.applied_methanator_kw),
                    "solar_curtailed": float(self.applied_solar_curtailed_kw),
                },
                "battery_charge_attribution": {
                    "direct_dispatch_kw": float(self.dispatch.routes_kw[0]),
                    "direct_dispatch_ceiling_kw": float(
                        self.dispatch.targets.battery_charge_ceiling_kw
                    ),
                    "applied_including_recovered_spill_kw": float(
                        self.applied_solar_to_battery_kw
                    ),
                    "recovered_or_guard_redirect_kw": float(
                        max(
                            0.0,
                            self.applied_solar_to_battery_kw
                            - self.dispatch.routes_kw[0],
                        )
                    ),
                },
            },
        }

    @property
    def sha256(self) -> str:
        return str(
            domain_sha256(
                PROCESS_SETPOINT_MINUTE_APPLICATION_DOMAIN,
                self.body(),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


def record_process_setpoint_application(
    *,
    ordinal: int,
    before: PlantState,
    dispatch: ProcessSetpointMinuteDispatchV1,
    validated: ValidatedAction,
    after: PlantState,
    next_boundary: PlantState,
) -> ProcessSetpointMinuteApplicationV1:
    """Construct one common dispatch-to-plant evidence link."""

    if not all(isinstance(item, PlantState) for item in (before, after, next_boundary)):
        raise TypeError("application boundaries must be PlantState")
    if not isinstance(dispatch, ProcessSetpointMinuteDispatchV1):
        raise TypeError("dispatch must be ProcessSetpointMinuteDispatchV1")
    if not isinstance(validated, ValidatedAction):
        raise TypeError("validated must be ValidatedAction")
    if process_setpoint_boundary_sha256(before) != dispatch.boundary_sha256:
        raise ValueError("dispatch belongs to another before boundary")
    if before.step != dispatch.boundary_step:
        raise ValueError("dispatch step differs from its before boundary")
    if after.step != before.step + 1 or next_boundary.step != after.step:
        raise ValueError("application after boundaries are not contiguous")
    if after.action != validated.action:
        raise ValueError("simulator action differs from the guarded action")
    return ProcessSetpointMinuteApplicationV1(
        ordinal=_index("ordinal", ordinal),
        dispatch=dispatch,
        guarded_action=validated.action,
        adjustments=tuple(validated.adjustments),
        next_boundary_sha256=process_setpoint_boundary_sha256(next_boundary),
        next_boundary_step=next_boundary.step,
        applied_action=after.action,
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
class ProcessSetpointIntervalExecutionV1:
    """Complete minute ledger for one held common command."""

    command: ProcessSetpointCommandV1
    physics_step_s: float
    applications: tuple[ProcessSetpointMinuteApplicationV1, ...]

    schema: ClassVar[str] = PROCESS_SETPOINT_INTERVAL_EXECUTION_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.command, ProcessSetpointCommandV1):
            raise TypeError("execution command must be ProcessSetpointCommandV1")
        step_s = _finite("physics_step_s", self.physics_step_s, minimum=0.0)
        if step_s <= 0.0:
            raise ValueError("physics_step_s must be positive")
        if not isinstance(self.applications, tuple) or not self.applications:
            raise ValueError("execution applications must be a non-empty tuple")
        expected_count = self.command.hold_duration_s / step_s
        if not math.isclose(
            expected_count,
            round(expected_count),
            rel_tol=0.0,
            abs_tol=1e-9,
        ) or len(self.applications) != int(round(expected_count)):
            raise ValueError(
                "application count does not cover the command hold duration"
            )
        for index, application in enumerate(self.applications):
            if not isinstance(application, ProcessSetpointMinuteApplicationV1):
                raise TypeError("execution applications must be typed minute evidence")
            if application.ordinal != index:
                raise ValueError("application ordinals are not contiguous")
            if application.dispatch.command != self.command:
                raise ValueError("application belongs to another common command")
            if index > 0:
                previous = self.applications[index - 1]
                if (
                    previous.next_boundary_step != application.boundary_step
                    or previous.next_boundary_sha256 != application.boundary_sha256
                ):
                    raise ValueError("application physical boundary chain is broken")

    @property
    def first_boundary_sha256(self) -> str:
        return self.applications[0].boundary_sha256

    @property
    def final_boundary_sha256(self) -> str:
        return self.applications[-1].next_boundary_sha256

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "command": self.command.to_dict(),
            "physics_step_s": float(self.physics_step_s),
            "application_count": len(self.applications),
            "first_boundary_sha256": self.first_boundary_sha256,
            "final_boundary_sha256": self.final_boundary_sha256,
            "applications": [item.to_dict() for item in self.applications],
        }

    @property
    def sha256(self) -> str:
        return str(
            domain_sha256(
                PROCESS_SETPOINT_INTERVAL_EXECUTION_DOMAIN,
                self.body(),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


__all__ = [
    "PROCESS_SETPOINT_INTERVAL_EXECUTION_DOMAIN",
    "PROCESS_SETPOINT_INTERVAL_EXECUTION_SCHEMA",
    "PROCESS_SETPOINT_MINUTE_APPLICATION_DOMAIN",
    "PROCESS_SETPOINT_MINUTE_APPLICATION_SCHEMA",
    "ProcessSetpointIntervalExecutionV1",
    "ProcessSetpointMinuteApplicationV1",
    "record_process_setpoint_application",
]

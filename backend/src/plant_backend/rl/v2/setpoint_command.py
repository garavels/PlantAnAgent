"""Observation-bound controller-neutral setpoint commands and minute dispatch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, ClassVar

from plant_sim import Action, PlantConfig, PlantState

from ...controllers.base import Constraints
from ...safety.validation import (
    BatterySupportAvailability,
    battery_support_availability,
)
from .mpc_schema import domain_sha256, require_sha256
from .setpoint_control import (
    PROCESS_SETPOINT_TOLERANCE_KW,
    ProcessSetpointAllocationV1,
    ProcessSetpointTargetsV1,
    allocate_process_setpoint_targets,
)

if TYPE_CHECKING:
    from .mpc_setpoints import MPCProcessSetpointsV1


PROCESS_SETPOINT_ENVELOPE_SCHEMA = "process-setpoint-envelope-v1"
PROCESS_SETPOINT_ENVELOPE_DOMAIN = PROCESS_SETPOINT_ENVELOPE_SCHEMA
PROCESS_SETPOINT_COMMAND_SCHEMA = "process-setpoint-command-v1"
PROCESS_SETPOINT_COMMAND_DOMAIN = PROCESS_SETPOINT_COMMAND_SCHEMA
PROCESS_SETPOINT_BOUNDARY_DOMAIN = "process-setpoint-minute-boundary-v1"
PROCESS_SETPOINT_MINUTE_DISPATCH_SCHEMA = "process-setpoint-minute-dispatch-v1"
PROCESS_SETPOINT_MINUTE_DISPATCH_DOMAIN = PROCESS_SETPOINT_MINUTE_DISPATCH_SCHEMA


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


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class ProcessSetpointEnvelopeV1:
    """Resolved actuator and reserve limits bound into one common command."""

    electrolyser_limit_kw: float
    dac_limit_kw: float
    methanator_limit_kw: float
    battery_charge_limit_kw: float
    battery_support_limit_kw: float
    battery_reserve_soc: float

    schema: ClassVar[str] = PROCESS_SETPOINT_ENVELOPE_SCHEMA

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

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "limits_kw": {
                "electrolyser": float(self.electrolyser_limit_kw),
                "dac": float(self.dac_limit_kw),
                "methanator": float(self.methanator_limit_kw),
                "battery_charge": float(self.battery_charge_limit_kw),
                "battery_support": float(self.battery_support_limit_kw),
            },
            "battery_reserve_soc": float(self.battery_reserve_soc),
        }

    @property
    def sha256(self) -> str:
        return domain_sha256(PROCESS_SETPOINT_ENVELOPE_DOMAIN, self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


def resolve_process_setpoint_envelope(
    *,
    constraints: Constraints,
    config: PlantConfig,
) -> ProcessSetpointEnvelopeV1:
    """Resolve the common command authority from current operator limits."""

    if not isinstance(constraints, Constraints):
        raise TypeError("constraints must be Constraints")
    if not isinstance(config, PlantConfig):
        raise TypeError("config must be PlantConfig")
    return ProcessSetpointEnvelopeV1(
        electrolyser_limit_kw=constraints.electrolyser_limit_kw(config),
        dac_limit_kw=constraints.dac_limit_kw(config),
        methanator_limit_kw=constraints.methanator_limit_kw(config),
        battery_charge_limit_kw=config.battery.max_charge_kw,
        battery_support_limit_kw=constraints.support_limit_kw(config),
        battery_reserve_soc=constraints.reserve_soc(config),
    )


@dataclass(frozen=True, slots=True)
class ProcessSetpointCommandV1:
    """Controller-neutral absolute targets bound to one public decision boundary."""

    decision_index: int
    observation_sha256: str
    hold_duration_s: float
    issuer_id: str
    source_schema: str
    source_sha256: str
    targets: ProcessSetpointTargetsV1
    envelope: ProcessSetpointEnvelopeV1

    schema: ClassVar[str] = PROCESS_SETPOINT_COMMAND_SCHEMA

    def __post_init__(self) -> None:
        _index("decision_index", self.decision_index)
        require_sha256("observation_sha256", self.observation_sha256)
        _finite("hold_duration_s", self.hold_duration_s, minimum=0.0)
        if self.hold_duration_s <= 0.0:
            raise ValueError("hold_duration_s must be positive")
        _text("issuer_id", self.issuer_id)
        _text("source_schema", self.source_schema)
        require_sha256("source_sha256", self.source_sha256)
        if not isinstance(self.targets, ProcessSetpointTargetsV1):
            raise TypeError("command targets must be ProcessSetpointTargetsV1")
        if not isinstance(self.envelope, ProcessSetpointEnvelopeV1):
            raise TypeError("command envelope must be ProcessSetpointEnvelopeV1")
        limits = (
            (
                "electrolyser",
                self.targets.electrolyser_target_kw,
                self.envelope.electrolyser_limit_kw,
            ),
            ("DAC", self.targets.dac_target_kw, self.envelope.dac_limit_kw),
            (
                "methanator",
                self.targets.methanator_target_kw,
                self.envelope.methanator_limit_kw,
            ),
            (
                "battery charge",
                self.targets.battery_charge_ceiling_kw,
                self.envelope.battery_charge_limit_kw,
            ),
        )
        for name, target, limit in limits:
            if target > limit + PROCESS_SETPOINT_TOLERANCE_KW:
                raise ValueError(f"{name} target exceeds the active command envelope")

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "decision_index": int(self.decision_index),
            "observation_sha256": self.observation_sha256,
            "hold_duration_s": float(self.hold_duration_s),
            "issuer": {
                "id": self.issuer_id,
                "source_schema": self.source_schema,
                "source_sha256": self.source_sha256,
            },
            "targets": self.targets.to_dict(),
            "envelope": self.envelope.to_dict(),
        }

    @property
    def sha256(self) -> str:
        return domain_sha256(PROCESS_SETPOINT_COMMAND_DOMAIN, self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


def issue_process_setpoint_command(
    targets: ProcessSetpointTargetsV1,
    *,
    decision_index: int,
    observation_sha256: str,
    hold_duration_s: float,
    issuer_id: str,
    source_schema: str,
    source_sha256: str,
    constraints: Constraints,
    config: PlantConfig,
) -> ProcessSetpointCommandV1:
    """Bind controller evidence and current authority to absolute targets."""

    return ProcessSetpointCommandV1(
        decision_index=decision_index,
        observation_sha256=observation_sha256,
        hold_duration_s=hold_duration_s,
        issuer_id=issuer_id,
        source_schema=source_schema,
        source_sha256=source_sha256,
        targets=targets,
        envelope=resolve_process_setpoint_envelope(
            constraints=constraints,
            config=config,
        ),
    )


def adapt_mpc_process_setpoints(
    setpoints: MPCProcessSetpointsV1,
) -> ProcessSetpointCommandV1:
    """Project typed MPC setpoints into the common door without losing provenance."""

    from .mpc_setpoints import MPCProcessSetpointsV1

    if not isinstance(setpoints, MPCProcessSetpointsV1):
        raise TypeError("setpoints must be MPCProcessSetpointsV1")
    return ProcessSetpointCommandV1(
        decision_index=setpoints.decision_index,
        observation_sha256=setpoints.observation_sha256,
        hold_duration_s=setpoints.hold_duration_s,
        issuer_id=setpoints.controller_id,
        source_schema=setpoints.schema,
        source_sha256=setpoints.sha256,
        targets=ProcessSetpointTargetsV1(
            electrolyser_target_kw=setpoints.electrolyser_target_kw,
            dac_target_kw=setpoints.dac_target_kw,
            methanator_target_kw=setpoints.methanator_target_kw,
            battery_charge_ceiling_kw=setpoints.battery_charge_ceiling_kw,
        ),
        envelope=ProcessSetpointEnvelopeV1(
            electrolyser_limit_kw=setpoints.envelope.electrolyser_limit_kw,
            dac_limit_kw=setpoints.envelope.dac_limit_kw,
            methanator_limit_kw=setpoints.envelope.methanator_limit_kw,
            battery_charge_limit_kw=setpoints.envelope.battery_charge_limit_kw,
            battery_support_limit_kw=setpoints.envelope.battery_support_limit_kw,
            battery_reserve_soc=setpoints.envelope.battery_reserve_soc,
        ),
    )


def process_setpoint_boundary_sha256(boundary: PlantState) -> str:
    if not isinstance(boundary, PlantState):
        raise TypeError("boundary must be PlantState")
    return domain_sha256(PROCESS_SETPOINT_BOUNDARY_DOMAIN, boundary.to_dict())


@dataclass(frozen=True, slots=True)
class ProcessSetpointMinuteDispatchV1:
    """One common command allocated against one current physical minute."""

    command: ProcessSetpointCommandV1
    boundary_sha256: str
    boundary_step: int
    allocation: ProcessSetpointAllocationV1

    schema: ClassVar[str] = PROCESS_SETPOINT_MINUTE_DISPATCH_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.command, ProcessSetpointCommandV1):
            raise TypeError("dispatch command must be ProcessSetpointCommandV1")
        require_sha256("boundary_sha256", self.boundary_sha256)
        _index("boundary_step", self.boundary_step)
        if not isinstance(self.allocation, ProcessSetpointAllocationV1):
            raise TypeError("dispatch allocation must be ProcessSetpointAllocationV1")
        if self.allocation.targets != self.command.targets:
            raise ValueError("dispatch allocation belongs to another command")

    @property
    def command_sha256(self) -> str:
        return self.command.sha256

    @property
    def decision_index(self) -> int:
        return self.command.decision_index

    @property
    def action(self) -> Action:
        return self.allocation.action

    @property
    def targets(self) -> ProcessSetpointTargetsV1:
        return self.allocation.targets

    @property
    def support_availability(self) -> BatterySupportAvailability:
        return self.allocation.support_availability

    @property
    def routes_kw(self) -> tuple[float, float, float, float]:
        return self.allocation.routes_kw

    @property
    def support_kw(self) -> float:
        return self.allocation.support_kw

    @property
    def target_reachable(self) -> bool:
        return self.allocation.target_reachable

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "command_sha256": self.command_sha256,
            "decision_index": int(self.decision_index),
            "boundary": {
                "sha256": self.boundary_sha256,
                "step": int(self.boundary_step),
            },
            "allocation": self.allocation.to_dict(),
        }

    @property
    def sha256(self) -> str:
        return domain_sha256(PROCESS_SETPOINT_MINUTE_DISPATCH_DOMAIN, self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


def dispatch_process_setpoint_command(
    command: ProcessSetpointCommandV1,
    boundary: PlantState,
    *,
    solar_available_kw: float,
    constraints: Constraints,
    config: PlantConfig,
) -> ProcessSetpointMinuteDispatchV1:
    """Realize one held common command using only the current physical boundary."""

    if not isinstance(command, ProcessSetpointCommandV1):
        raise TypeError("command must be ProcessSetpointCommandV1")
    if not isinstance(boundary, PlantState):
        raise TypeError("boundary must be PlantState")
    solar_kw = _finite("solar_available_kw", solar_available_kw, minimum=0.0)
    active_envelope = resolve_process_setpoint_envelope(
        constraints=constraints,
        config=config,
    )
    if active_envelope != command.envelope:
        raise ValueError("active process setpoint envelope changed")
    support = battery_support_availability(
        boundary,
        constraints=constraints,
        config=config,
        electrolyser_headroom_kw=active_envelope.electrolyser_limit_kw,
    )
    allocation = allocate_process_setpoint_targets(
        command.targets,
        solar_available_kw=solar_kw,
        support_availability=support,
    )
    return ProcessSetpointMinuteDispatchV1(
        command=command,
        boundary_sha256=process_setpoint_boundary_sha256(boundary),
        boundary_step=boundary.step,
        allocation=allocation,
    )


__all__ = [
    "PROCESS_SETPOINT_BOUNDARY_DOMAIN",
    "PROCESS_SETPOINT_COMMAND_DOMAIN",
    "PROCESS_SETPOINT_COMMAND_SCHEMA",
    "PROCESS_SETPOINT_ENVELOPE_DOMAIN",
    "PROCESS_SETPOINT_ENVELOPE_SCHEMA",
    "PROCESS_SETPOINT_MINUTE_DISPATCH_DOMAIN",
    "PROCESS_SETPOINT_MINUTE_DISPATCH_SCHEMA",
    "ProcessSetpointCommandV1",
    "ProcessSetpointEnvelopeV1",
    "ProcessSetpointMinuteDispatchV1",
    "adapt_mpc_process_setpoints",
    "dispatch_process_setpoint_command",
    "issue_process_setpoint_command",
    "process_setpoint_boundary_sha256",
    "resolve_process_setpoint_envelope",
]

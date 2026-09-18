"""Controller-neutral absolute targets, policy adapter, and dispatch kernel."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from plant_sim import Action, PlantConfig

from ...controllers.base import Constraints
from ...safety.validation import BatterySupportAvailability
from .control import close_unit_simplex
from .mpc_schema import domain_sha256


PROCESS_SETPOINT_TARGET_SCHEMA = "process-setpoint-targets-v1"
PROCESS_SETPOINT_TARGET_DOMAIN = PROCESS_SETPOINT_TARGET_SCHEMA
PROCESS_SETPOINT_ALLOCATION_SCHEMA = "process-setpoint-allocation-v1"
PROCESS_SETPOINT_ALLOCATION_DOMAIN = PROCESS_SETPOINT_ALLOCATION_SCHEMA
PROCESS_SETPOINT_POLICY_ACTION_SCHEMA = "normalized-process-setpoint-action-v1"
PROCESS_SETPOINT_POLICY_ACTION_DOMAIN = PROCESS_SETPOINT_POLICY_ACTION_SCHEMA
PROCESS_SETPOINT_ACTION_NAMES = (
    "electrolyser_target",
    "dac_target",
    "methanator_target",
    "battery_charge_ceiling",
)
PROCESS_SETPOINT_ACTION_SIZE = len(PROCESS_SETPOINT_ACTION_NAMES)
PROCESS_SETPOINT_TOLERANCE_KW = 1e-7


def _finite(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}")
    return number


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)


@dataclass(frozen=True, slots=True)
class ProcessSetpointTargetsV1:
    """Four controller-neutral absolute power targets in kW."""

    electrolyser_target_kw: float
    dac_target_kw: float
    methanator_target_kw: float
    battery_charge_ceiling_kw: float

    schema: ClassVar[str] = PROCESS_SETPOINT_TARGET_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "electrolyser_target_kw",
            "dac_target_kw",
            "methanator_target_kw",
            "battery_charge_ceiling_kw",
        ):
            _finite(name, getattr(self, name), minimum=0.0)

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "setpoints_kw": {
                "electrolyser": float(self.electrolyser_target_kw),
                "dac": float(self.dac_target_kw),
                "methanator": float(self.methanator_target_kw),
                "battery_charge_ceiling": float(self.battery_charge_ceiling_kw),
            },
        }

    @property
    def sha256(self) -> str:
        return domain_sha256(PROCESS_SETPOINT_TARGET_DOMAIN, self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class DecodedProcessSetpointActionV1:
    """Auditable conversion of one normalized learned-policy action."""

    raw_policy_vector: tuple[float, ...]
    bounded_policy_vector: tuple[float, ...]
    targets: ProcessSetpointTargetsV1
    clipped: bool

    schema: ClassVar[str] = PROCESS_SETPOINT_POLICY_ACTION_SCHEMA

    def __post_init__(self) -> None:
        for name, values in (
            ("raw_policy_vector", self.raw_policy_vector),
            ("bounded_policy_vector", self.bounded_policy_vector),
        ):
            if (
                not isinstance(values, tuple)
                or len(values) != PROCESS_SETPOINT_ACTION_SIZE
            ):
                raise ValueError(f"{name} must contain four values")
            for value in values:
                _finite(name, value)
        if any(not -1.0 <= value <= 1.0 for value in self.bounded_policy_vector):
            raise ValueError("bounded policy vector must lie within [-1, 1]")
        expected_clipped = self.raw_policy_vector != self.bounded_policy_vector
        if not isinstance(self.clipped, bool) or self.clipped != expected_clipped:
            raise ValueError("clipped flag disagrees with the policy vectors")
        if not isinstance(self.targets, ProcessSetpointTargetsV1):
            raise TypeError("decoded action targets must be ProcessSetpointTargetsV1")

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "action_names": list(PROCESS_SETPOINT_ACTION_NAMES),
            "raw_policy_vector": list(self.raw_policy_vector),
            "bounded_policy_vector": list(self.bounded_policy_vector),
            "clipped": self.clipped,
            "targets": self.targets.to_dict(),
        }

    @property
    def sha256(self) -> str:
        return domain_sha256(PROCESS_SETPOINT_POLICY_ACTION_DOMAIN, self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


class ProcessSetpointActionAdapterV1:
    """Map a four-dimensional [-1, 1] policy action to active kW limits."""

    names = PROCESS_SETPOINT_ACTION_NAMES
    size = PROCESS_SETPOINT_ACTION_SIZE

    @staticmethod
    def low() -> NDArray[np.float32]:
        return np.full(PROCESS_SETPOINT_ACTION_SIZE, -1.0, dtype=np.float32)

    @staticmethod
    def high() -> NDArray[np.float32]:
        return np.full(PROCESS_SETPOINT_ACTION_SIZE, 1.0, dtype=np.float32)

    def decode(
        self,
        action: NDArray[np.generic] | list[float] | tuple[float, ...],
        *,
        constraints: Constraints,
        config: PlantConfig,
    ) -> DecodedProcessSetpointActionV1:
        if not isinstance(constraints, Constraints):
            raise TypeError("constraints must be Constraints")
        if not isinstance(config, PlantConfig):
            raise TypeError("config must be PlantConfig")
        untyped = np.asarray(action, dtype=object)
        if untyped.shape != (PROCESS_SETPOINT_ACTION_SIZE,):
            raise ValueError("setpoint action must contain exactly four entries")
        raw_array = np.asarray(
            tuple(_finite("setpoint action entry", value) for value in untyped),
            dtype=np.float64,
        )
        bounded_array = np.clip(raw_array, -1.0, 1.0)
        raw = tuple(float(value) for value in raw_array)
        bounded = tuple(float(value) for value in bounded_array)
        limits = (
            constraints.electrolyser_limit_kw(config),
            constraints.dac_limit_kw(config),
            constraints.methanator_limit_kw(config),
            config.battery.max_charge_kw,
        )
        decoded = tuple(
            (value + 1.0) * 0.5 * limit
            for value, limit in zip(bounded, limits, strict=True)
        )
        return DecodedProcessSetpointActionV1(
            raw_policy_vector=raw,
            bounded_policy_vector=bounded,
            targets=ProcessSetpointTargetsV1(*decoded),
            clipped=raw != bounded,
        )


@dataclass(frozen=True, slots=True)
class ProcessSetpointAllocationV1:
    """One target allocation against a contemporaneous solar/support boundary."""

    targets: ProcessSetpointTargetsV1
    solar_available_kw: float
    support_availability: BatterySupportAvailability
    routes_kw: tuple[float, float, float, float]
    support_kw: float
    unallocated_kw: float
    target_reachable: bool
    action: Action

    schema: ClassVar[str] = PROCESS_SETPOINT_ALLOCATION_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.targets, ProcessSetpointTargetsV1):
            raise TypeError("allocation targets must be ProcessSetpointTargetsV1")
        solar = _finite("solar_available_kw", self.solar_available_kw, minimum=0.0)
        if not isinstance(self.support_availability, BatterySupportAvailability):
            raise TypeError("support_availability must be BatterySupportAvailability")
        if not isinstance(self.routes_kw, tuple) or len(self.routes_kw) != 4:
            raise ValueError("routes_kw must contain four values")
        routes = tuple(
            _finite("route kW", value, minimum=0.0) for value in self.routes_kw
        )
        support_kw = _finite("support_kw", self.support_kw, minimum=0.0)
        unallocated = _finite("unallocated_kw", self.unallocated_kw, minimum=0.0)
        if not isinstance(self.target_reachable, bool):
            raise TypeError("target_reachable must be a bool")
        if (
            not isinstance(self.action, Action)
            or Action.sanitised(self.action) != self.action
        ):
            raise ValueError("allocation action must be a sanitized Action")
        if math.fsum(routes) > solar + PROCESS_SETPOINT_TOLERANCE_KW:
            raise ValueError("allocation routes exceed available solar")
        if not _close(unallocated, max(0.0, solar - math.fsum(routes))):
            raise ValueError("unallocated solar does not close")
        if (
            support_kw
            > self.support_availability.available_kw + PROCESS_SETPOINT_TOLERANCE_KW
        ):
            raise ValueError("allocation support exceeds availability")
        if routes[0] > 0.0 and support_kw > 0.0:
            raise ValueError("allocation cannot charge and support simultaneously")
        if solar > 0.0:
            action_routes = (
                self.action.battery_share * solar,
                self.action.electrolyser_share * solar,
                self.action.dac_share * solar,
                self.action.methanator_share * solar,
            )
        else:
            action_routes = (0.0, 0.0, 0.0, 0.0)
        if any(
            not _close(left, right)
            for left, right in zip(routes, action_routes, strict=True)
        ):
            raise ValueError("allocation routes disagree with the ordinary action")
        if not _close(support_kw, self.action.battery_support_kw):
            raise ValueError("allocation support disagrees with the ordinary action")
        if routes[2] > self.targets.dac_target_kw + PROCESS_SETPOINT_TOLERANCE_KW:
            raise ValueError("DAC allocation exceeds its target")
        if (
            routes[3]
            > self.targets.methanator_target_kw + PROCESS_SETPOINT_TOLERANCE_KW
        ):
            raise ValueError("methanator allocation exceeds its target")
        if (
            routes[0]
            > self.targets.battery_charge_ceiling_kw + PROCESS_SETPOINT_TOLERANCE_KW
        ):
            raise ValueError("battery allocation exceeds its charge ceiling")
        offered = routes[1] + support_kw
        if (
            offered
            > self.targets.electrolyser_target_kw + PROCESS_SETPOINT_TOLERANCE_KW
        ):
            raise ValueError("electrolyser allocation exceeds its target")
        if (
            self.target_reachable
            and abs(offered - self.targets.electrolyser_target_kw)
            > PROCESS_SETPOINT_TOLERANCE_KW
        ):
            raise ValueError("reachable electrolyser target is not met")

    @property
    def electrolyser_offered_kw(self) -> float:
        return self.routes_kw[1] + self.support_kw

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "targets": self.targets.to_dict(),
            "solar_available_kw": float(self.solar_available_kw),
            "support_availability": self.support_availability.to_dict(),
            "request_kw": {
                "solar_to_battery": float(self.routes_kw[0]),
                "solar_to_electrolyser": float(self.routes_kw[1]),
                "solar_to_dac": float(self.routes_kw[2]),
                "solar_to_methanator": float(self.routes_kw[3]),
                "battery_support": float(self.support_kw),
                "unallocated_solar": float(self.unallocated_kw),
                "electrolyser_total": float(self.electrolyser_offered_kw),
            },
            "electrolyser_target_reachable": self.target_reachable,
            "action": self.action.to_dict(),
        }

    @property
    def sha256(self) -> str:
        return domain_sha256(PROCESS_SETPOINT_ALLOCATION_DOMAIN, self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


def _proportional_routes(
    remaining_kw: float,
    dac_kw: float,
    methanator_kw: float,
) -> tuple[float, float]:
    total = dac_kw + methanator_kw
    if total <= 0.0 or remaining_kw <= 0.0:
        return 0.0, 0.0
    factor = min(1.0, remaining_kw / total)
    allocated_dac = dac_kw * factor
    allocated_methanator = methanator_kw * factor
    routed = allocated_dac + allocated_methanator
    if routed > remaining_kw:
        closure = math.nextafter(remaining_kw / routed, 0.0)
        allocated_dac *= closure
        allocated_methanator *= closure
    return allocated_dac, allocated_methanator


def allocate_process_setpoint_targets(
    targets: ProcessSetpointTargetsV1,
    *,
    solar_available_kw: float,
    support_availability: BatterySupportAvailability,
) -> ProcessSetpointAllocationV1:
    """Apply the shared reserve-aware target-allocation rule for one minute."""

    if not isinstance(targets, ProcessSetpointTargetsV1):
        raise TypeError("targets must be ProcessSetpointTargetsV1")
    solar_kw = _finite("solar_available_kw", solar_available_kw, minimum=0.0)
    if not isinstance(support_availability, BatterySupportAvailability):
        raise TypeError("support_availability must be BatterySupportAvailability")

    electrolyser_target_kw = targets.electrolyser_target_kw
    reserved_electrolyser_kw = min(
        solar_kw,
        max(0.0, electrolyser_target_kw - support_availability.available_kw),
    )
    remaining_kw = max(0.0, solar_kw - reserved_electrolyser_kw)
    dac_kw, methanator_kw = _proportional_routes(
        remaining_kw,
        targets.dac_target_kw,
        targets.methanator_target_kw,
    )
    remaining_kw = max(0.0, remaining_kw - dac_kw - methanator_kw)
    extra_electrolyser_kw = min(
        remaining_kw,
        max(0.0, electrolyser_target_kw - reserved_electrolyser_kw),
    )
    electrolyser_solar_kw = reserved_electrolyser_kw + extra_electrolyser_kw
    remaining_kw = max(0.0, remaining_kw - extra_electrolyser_kw)
    support_kw = min(
        support_availability.available_kw,
        max(0.0, electrolyser_target_kw - electrolyser_solar_kw),
    )
    battery_kw = (
        min(targets.battery_charge_ceiling_kw, remaining_kw)
        if support_kw == 0.0
        else 0.0
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
    routes = (
        shares[0] * solar_kw,
        shares[1] * solar_kw,
        shares[2] * solar_kw,
        shares[3] * solar_kw,
    )
    target_reachable = bool(
        electrolyser_target_kw == 0.0
        or (
            support_availability.electrolyser_available
            and electrolyser_target_kw
            <= support_availability.electrolyser_headroom_kw
            + PROCESS_SETPOINT_TOLERANCE_KW
            and solar_kw + support_availability.available_kw
            >= electrolyser_target_kw - PROCESS_SETPOINT_TOLERANCE_KW
        )
    )
    return ProcessSetpointAllocationV1(
        targets=targets,
        solar_available_kw=solar_kw,
        support_availability=support_availability,
        routes_kw=routes,
        support_kw=support_kw,
        unallocated_kw=max(0.0, solar_kw - math.fsum(routes)),
        target_reachable=target_reachable,
        action=action,
    )


__all__ = [
    "PROCESS_SETPOINT_ACTION_NAMES",
    "PROCESS_SETPOINT_ACTION_SIZE",
    "PROCESS_SETPOINT_ALLOCATION_DOMAIN",
    "PROCESS_SETPOINT_ALLOCATION_SCHEMA",
    "PROCESS_SETPOINT_POLICY_ACTION_DOMAIN",
    "PROCESS_SETPOINT_POLICY_ACTION_SCHEMA",
    "PROCESS_SETPOINT_TARGET_DOMAIN",
    "PROCESS_SETPOINT_TARGET_SCHEMA",
    "DecodedProcessSetpointActionV1",
    "ProcessSetpointActionAdapterV1",
    "ProcessSetpointAllocationV1",
    "ProcessSetpointTargetsV1",
    "allocate_process_setpoint_targets",
]

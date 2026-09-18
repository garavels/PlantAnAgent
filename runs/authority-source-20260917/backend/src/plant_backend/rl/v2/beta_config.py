"""Immutable cyclic-task contracts for Environment V2 beta.

The physical episode, market process, and disturbance generator remain separate
objects.  This module defines only the cycle reference and settlement semantics
that bind them into one checkpointable task.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Mapping

from plant_sim import PlantConfig, PlantState

from .beta_disturbances import DISTURBANCE_GENERATOR_VERSION, DisturbanceConfigV1
from .config import EpisodeConfigV2
from .economics import EconomicAssumptionsV1
from .market import MARKET_GENERATOR_VERSION, MarketConfigV1


CYCLIC_MARKET_TASK_SCHEMA = "cyclic-market-task-v1"
CYCLIC_RETURN_SCHEMA = "cyclic-resource-return-v1"
CYCLE_REFERENCE_SCHEMA = "cycle-opening-reference-v1"
CYCLE_SETTLEMENT_SCHEMA = "cycle-resource-settlement-v1"
RESOURCE_NAMES = ("battery", "h2", "co2", "ch4")
CYCLE_FEASIBILITY_EPSILON = 1e-12


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _aligned_count(name: str, numerator_s: float, denominator_s: float) -> int:
    count = int(round(numerator_s / denominator_s))
    if count <= 0 or not math.isclose(
        numerator_s,
        count * denominator_s,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{name} must align to {denominator_s:g} s")
    return count


@dataclass(frozen=True, slots=True)
class CyclicReturnConfigV1:
    """Non-ratcheting inventory return condition settled every civil day."""

    cycle_h: float = 24.0
    battery_return_fraction: float = 1.0
    h2_return_fraction: float = 1.0
    co2_return_fraction: float = 1.0
    ch4_return_fraction: float = 1.0
    battery_tolerance_kwh: float = 1e-6
    h2_tolerance_kg: float = 1e-6
    co2_tolerance_kg: float = 1e-6
    ch4_tolerance_kg: float = 1e-6

    def validate(
        self,
        *,
        episode: EpisodeConfigV2,
        plant: PlantConfig,
    ) -> "CyclicReturnConfigV1":
        cycle_h = _finite("cycle_h", self.cycle_h, minimum=1e-12)
        if not math.isclose(cycle_h, 24.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("cyclic-market beta requires an exact 24 h cycle")
        for name in (
            "battery_return_fraction",
            "h2_return_fraction",
            "co2_return_fraction",
            "ch4_return_fraction",
        ):
            _finite(name, getattr(self, name), minimum=0.0, maximum=1.0)
        for name in (
            "battery_tolerance_kwh",
            "h2_tolerance_kg",
            "co2_tolerance_kg",
            "ch4_tolerance_kg",
        ):
            _finite(name, getattr(self, name), minimum=0.0)
        cycle_s = cycle_h * 3_600.0
        cycle_decisions = _aligned_count(
            "cycle_h",
            cycle_s,
            episode.decision_interval_s,
        )
        horizon_decisions = episode.decision_steps(plant)
        if horizon_decisions % cycle_decisions:
            raise ValueError(
                "episode horizon must contain a whole number of 24 h cycles"
            )
        cycle_physics = _aligned_count("cycle_h", cycle_s, plant.dt_s)
        delivery_physics = episode.delivery_stride(plant)
        if cycle_physics % delivery_physics:
            raise ValueError("a 24 h cycle must contain whole delivery periods")
        return self

    def cycle_decisions(self, episode: EpisodeConfigV2) -> int:
        return _aligned_count(
            "cycle_h",
            self.cycle_h * 3_600.0,
            episode.decision_interval_s,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"schema": CYCLIC_RETURN_SCHEMA, **asdict(self)}


@dataclass(frozen=True, slots=True)
class CyclicMarketTaskConfigV1:
    """Complete independently ablatable beta task extension."""

    cycle: CyclicReturnConfigV1 = field(default_factory=CyclicReturnConfigV1)
    market: MarketConfigV1 = field(default_factory=MarketConfigV1)
    disturbances: DisturbanceConfigV1 = field(default_factory=DisturbanceConfigV1)

    def validate(
        self,
        *,
        episode: EpisodeConfigV2,
        plant: PlantConfig,
        assumptions: EconomicAssumptionsV1,
    ) -> "CyclicMarketTaskConfigV1":
        if not isinstance(self.cycle, CyclicReturnConfigV1):
            raise TypeError("cycle must be CyclicReturnConfigV1")
        if not isinstance(self.market, MarketConfigV1):
            raise TypeError("market must be MarketConfigV1")
        if not isinstance(self.disturbances, DisturbanceConfigV1):
            raise TypeError("disturbances must be DisturbanceConfigV1")
        episode.validate(plant)
        assumptions.validate()
        self.cycle.validate(episode=episode, plant=plant)
        self.market.validate()
        cycle_decisions = self.cycle.cycle_decisions(episode)
        self.disturbances.validate_geometry(
            cycle_h=self.cycle.cycle_h,
            cycle_count=episode.decision_steps(plant) // cycle_decisions,
            physics_dt_s=plant.dt_s,
        )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CYCLIC_MARKET_TASK_SCHEMA,
            "cycle": self.cycle.to_dict(),
            "market": {
                "generator_version": MARKET_GENERATOR_VERSION,
                "config": self.market.to_dict(),
            },
            "disturbances": {
                "generator_version": DISTURBANCE_GENERATOR_VERSION,
                "config": self.disturbances.to_dict(),
            },
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class CycleReferenceV1:
    """Authoritative opening resources used at every cycle settlement."""

    battery_energy_kwh: float
    h2_kg: float
    co2_kg: float
    ch4_kg: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _finite(name, value, minimum=0.0)

    @classmethod
    def from_state(cls, state: PlantState) -> "CycleReferenceV1":
        if not isinstance(state, PlantState):
            raise TypeError("cycle reference requires PlantState")
        return cls(
            battery_energy_kwh=state.battery_energy_kwh,
            h2_kg=state.h2_level_kg,
            co2_kg=state.co2_level_kg,
            ch4_kg=state.ch4_level_kg,
        )

    def targets(self, config: CyclicReturnConfigV1) -> dict[str, float]:
        if not isinstance(config, CyclicReturnConfigV1):
            raise TypeError("cycle config must be CyclicReturnConfigV1")
        return {
            "battery": self.battery_energy_kwh * config.battery_return_fraction,
            "h2": self.h2_kg * config.h2_return_fraction,
            "co2": self.co2_kg * config.co2_return_fraction,
            "ch4": self.ch4_kg * config.ch4_return_fraction,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"schema": CYCLE_REFERENCE_SCHEMA, **asdict(self)}


@dataclass(frozen=True, slots=True)
class CycleSettlementV1:
    """One exact boundary check against the immutable opening reference."""

    cycle_index: int
    boundary_decision: int
    targets: Mapping[str, float]
    actual: Mapping[str, float]
    deficits: Mapping[str, float]
    tolerances: Mapping[str, float]
    feasible: bool

    def __post_init__(self) -> None:
        for name in ("cycle_index", "boundary_decision"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for field_name in ("targets", "actual", "deficits", "tolerances"):
            values = getattr(self, field_name)
            if not isinstance(values, Mapping) or tuple(sorted(values)) != tuple(
                sorted(RESOURCE_NAMES)
            ):
                raise ValueError(f"{field_name} must cover all cyclic resources")
            detached: dict[str, float] = {}
            for resource, value in values.items():
                detached[str(resource)] = _finite(
                    f"{field_name}.{resource}", value, minimum=0.0
                )
            object.__setattr__(self, field_name, MappingProxyType(detached))
        expected_deficits = {
            name: max(0.0, float(self.targets[name]) - float(self.actual[name]))
            for name in RESOURCE_NAMES
        }
        for name in RESOURCE_NAMES:
            if not math.isclose(
                float(self.deficits[name]),
                expected_deficits[name],
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("cycle deficit does not match target and actual")
        expected_feasible = all(
            expected_deficits[name]
            <= float(self.tolerances[name]) + CYCLE_FEASIBILITY_EPSILON
            for name in RESOURCE_NAMES
        )
        if not isinstance(self.feasible, bool) or self.feasible != expected_feasible:
            raise ValueError("cycle feasibility does not match its resource deficits")

    @classmethod
    def evaluate(
        cls,
        *,
        cycle_index: int,
        boundary_decision: int,
        state: PlantState,
        reference: CycleReferenceV1,
        config: CyclicReturnConfigV1,
    ) -> "CycleSettlementV1":
        if not isinstance(state, PlantState):
            raise TypeError("cycle settlement requires PlantState")
        targets = reference.targets(config)
        actual = {
            "battery": state.battery_energy_kwh,
            "h2": state.h2_level_kg,
            "co2": state.co2_level_kg,
            "ch4": state.ch4_level_kg,
        }
        deficits = {
            name: max(0.0, targets[name] - actual[name]) for name in RESOURCE_NAMES
        }
        tolerances = {
            "battery": config.battery_tolerance_kwh,
            "h2": config.h2_tolerance_kg,
            "co2": config.co2_tolerance_kg,
            "ch4": config.ch4_tolerance_kg,
        }
        feasible = all(
            deficits[name] <= tolerances[name] + CYCLE_FEASIBILITY_EPSILON
            for name in RESOURCE_NAMES
        )
        return cls(
            cycle_index=cycle_index,
            boundary_decision=boundary_decision,
            targets=targets,
            actual=actual,
            deficits=deficits,
            tolerances=tolerances,
            feasible=feasible,
        )

    @property
    def total_constraint_cost(self) -> float:
        """Dimensionless zero-at-feasibility violation used for vector reporting."""

        return math.fsum(
            max(0.0, float(self.deficits[name]) - float(self.tolerances[name]))
            / max(1.0, float(self.targets[name]))
            for name in RESOURCE_NAMES
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CYCLE_SETTLEMENT_SCHEMA,
            "cycle_index": int(self.cycle_index),
            "boundary_decision": int(self.boundary_decision),
            "targets": {name: float(self.targets[name]) for name in RESOURCE_NAMES},
            "actual": {name: float(self.actual[name]) for name in RESOURCE_NAMES},
            "deficits": {name: float(self.deficits[name]) for name in RESOURCE_NAMES},
            "tolerances": {
                name: float(self.tolerances[name]) for name in RESOURCE_NAMES
            },
            "feasible": self.feasible,
            "total_constraint_cost": self.total_constraint_cost,
        }


__all__ = [
    "CYCLE_FEASIBILITY_EPSILON",
    "CYCLE_REFERENCE_SCHEMA",
    "CYCLE_SETTLEMENT_SCHEMA",
    "CYCLIC_MARKET_TASK_SCHEMA",
    "CYCLIC_RETURN_SCHEMA",
    "RESOURCE_NAMES",
    "CycleReferenceV1",
    "CycleSettlementV1",
    "CyclicMarketTaskConfigV1",
    "CyclicReturnConfigV1",
    "canonical_sha256",
]

"""Controller-neutral causal observation for the cyclic market beta."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any

from plant_sim import PlantConfig

from .control import ControllerObservationV2
from .market import PublicPriceForecastIssueV1


BETA_CONTROLLER_OBSERVATION_SCHEMA = "cyclic-market-controller-observation-v1"
CYCLE_OBSERVATION_SCHEMA = "cyclic-return-observation-v1"


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 digest") from exc
    return value


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class CycleObservationV1:
    """Current progress and signed margins to the non-ratcheting cycle target."""

    cycle_index: int
    decision_in_cycle: int
    cycle_decisions: int
    progress: float
    battery_margin_kwh: float
    h2_margin_kg: float
    co2_margin_kg: float
    ch4_margin_kg: float

    def __post_init__(self) -> None:
        for name in ("cycle_index", "cycle_decisions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.decision_in_cycle, bool)
            or not isinstance(self.decision_in_cycle, Integral)
            or not 0 <= self.decision_in_cycle <= self.cycle_decisions
        ):
            raise ValueError("decision_in_cycle must lie within the current cycle")
        progress = _finite("cycle progress", self.progress)
        if not 0.0 <= progress <= 1.0:
            raise ValueError("cycle progress must lie within [0, 1]")
        expected = self.decision_in_cycle / self.cycle_decisions
        if not math.isclose(progress, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("cycle progress disagrees with its decision counters")
        for name in (
            "battery_margin_kwh",
            "h2_margin_kg",
            "co2_margin_kg",
            "ch4_margin_kg",
        ):
            _finite(name, getattr(self, name))

    def normalized(self, config: PlantConfig) -> tuple[float, ...]:
        if not isinstance(config, PlantConfig):
            raise TypeError("cycle normalization requires PlantConfig")

        def scale(value: float, capacity: float) -> float:
            if not math.isfinite(capacity) or capacity <= 0.0:
                raise ValueError("cycle resource capacity must be positive and finite")
            return max(-1.0, min(1.0, value / capacity))

        return (
            float(self.progress),
            scale(self.battery_margin_kwh, config.battery.capacity_kwh),
            scale(self.h2_margin_kg, config.h2_tank.capacity_kg),
            scale(self.co2_margin_kg, config.co2_tank.capacity_kg),
            scale(self.ch4_margin_kg, config.ch4_tank.capacity_kg),
        )

    def to_dict(self) -> dict[str, int | float | str]:
        return {"schema": CYCLE_OBSERVATION_SCHEMA, **asdict(self)}


@dataclass(frozen=True, slots=True)
class CyclicMarketControllerObservationV1:
    """Exact typed source shared by beta policies and optimizers."""

    plant: ControllerObservationV2
    market: PublicPriceForecastIssueV1
    cycle: CycleObservationV1
    economic_assumptions_sha256: str
    beta_task_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.plant, ControllerObservationV2):
            raise TypeError("plant must be ControllerObservationV2")
        if not isinstance(self.market, PublicPriceForecastIssueV1):
            raise TypeError("market must be PublicPriceForecastIssueV1")
        if not isinstance(self.cycle, CycleObservationV1):
            raise TypeError("cycle must be CycleObservationV1")
        _require_sha256("economic_assumptions_sha256", self.economic_assumptions_sha256)
        _require_sha256("beta_task_sha256", self.beta_task_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BETA_CONTROLLER_OBSERVATION_SCHEMA,
            "plant": self.plant.to_dict(),
            "market": self.market.to_dict(),
            "cycle": self.cycle.to_dict(),
            "economic_assumptions_sha256": self.economic_assumptions_sha256,
            "beta_task_sha256": self.beta_task_sha256,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "BETA_CONTROLLER_OBSERVATION_SCHEMA",
    "CYCLE_OBSERVATION_SCHEMA",
    "CycleObservationV1",
    "CyclicMarketControllerObservationV1",
]

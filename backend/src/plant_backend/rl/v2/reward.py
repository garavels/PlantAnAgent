"""Delivery-service objective for the first Environment V2 alpha."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from plant_sim import PlantConfig

from ...controllers.base import Constraints

OBJECTIVE_VERSION = 1


@dataclass(frozen=True)
class RewardWeightsV2:
    """Provisional scalarization; benchmark conclusions use physical metrics."""

    delivery: float = 1.0
    terminal_inventory: float = 0.10
    shortfall: float = 4.0
    curtailment: float = 0.10
    battery_drain: float = 0.02
    reserve_breach: float = 4.0
    shutdown: float = 0.50
    correction: float = 0.05

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class IntervalOutcome:
    """Physical/accounting deltas over one controller decision interval."""

    hours: float
    physical_steps: int
    requested_kg: float
    delivered_kg: float
    unmet_kg: float
    produced_kg: float
    solar_kwh: float
    curtailed_kwh: float
    battery_discharged_kwh: float
    min_battery_soc: float
    shutdown_steps: int
    corrections: int
    out_of_range: bool
    terminal_ch4_kg: float
    reference_delivery_kg: float


@dataclass(frozen=True)
class RewardBreakdownV2:
    total: float
    components: dict[str, float] = field(default_factory=dict)
    signals: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_version": OBJECTIVE_VERSION,
            "total": self.total,
            "components": dict(self.components),
            "signals": dict(self.signals),
        }


def compute(
    outcome: IntervalOutcome,
    *,
    config: PlantConfig,
    constraints: Constraints,
    terminal: bool,
    weights: RewardWeightsV2 | None = None,
) -> RewardBreakdownV2:
    """Score delivered service; raw production is diagnostic and never rewarded."""
    weights = weights or RewardWeightsV2()
    reference = max(outcome.reference_delivery_kg, 1e-12)
    discharge_reference = max(
        config.battery.max_discharge_kw * max(outcome.hours, 0.0), 1e-12
    )
    physical_steps = max(1, outcome.physical_steps)
    reserve_deficit = max(0.0, constraints.reserve_soc(config) - outcome.min_battery_soc)
    signals = {
        "delivery": max(0.0, outcome.delivered_kg) / reference,
        "terminal_inventory": (
            min(1.0, max(0.0, outcome.terminal_ch4_kg) / reference) if terminal else 0.0
        ),
        "shortfall": max(0.0, outcome.unmet_kg) / reference,
        "curtailment": (
            max(0.0, outcome.curtailed_kwh) / outcome.solar_kwh
            if outcome.solar_kwh > 0.0
            else 0.0
        ),
        "battery_drain": max(0.0, outcome.battery_discharged_kwh) / discharge_reference,
        "reserve_breach": reserve_deficit,
        "shutdown": max(0, outcome.shutdown_steps) / physical_steps,
        "correction": min(
            1.0,
            (max(0, outcome.corrections) + (1 if outcome.out_of_range else 0))
            / (3.0 * physical_steps),
        ),
        # Visible for diagnosis, deliberately absent from ``components``.
        "raw_production_kg": max(0.0, outcome.produced_kg),
    }
    components = {
        "delivery": weights.delivery * signals["delivery"],
        "terminal_inventory": weights.terminal_inventory * signals["terminal_inventory"],
        "shortfall": -weights.shortfall * signals["shortfall"],
        "curtailment": -weights.curtailment * signals["curtailment"],
        "battery_drain": -weights.battery_drain * signals["battery_drain"],
        "reserve_breach": -weights.reserve_breach * signals["reserve_breach"],
        "shutdown": -weights.shutdown * signals["shutdown"],
        "correction": -weights.correction * signals["correction"],
    }
    return RewardBreakdownV2(
        total=float(sum(components.values())),
        components=components,
        signals=signals,
    )


__all__ = [
    "OBJECTIVE_VERSION",
    "IntervalOutcome",
    "RewardBreakdownV2",
    "RewardWeightsV2",
    "compute",
]

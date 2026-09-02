"""Delivery-service objective for the first Environment V2 alpha."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from typing import Any

from plant_sim import PlantConfig

from ...controllers.base import Constraints
from .boundary import CommitmentExposureV2

OBJECTIVE_VERSION = 3


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
    #: Provisional opportunity-cost proxy, not a calibrated electricity price.
    nonproductive_energy: float = 0.05
    #: Provisional operational-event proxy, not an equipment-life model.
    forced_stop: float = 0.10

    def validate(self) -> "RewardWeightsV2":
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"reward weight {name} must be finite and non-negative")
        return self

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
    guard_intervention_kwh: float
    dispatch_gap_kwh: float
    request_application_gap_kwh: float
    out_of_range: bool
    terminal_ch4_kg: float
    reference_delivery_kg: float
    reference_period_h: float
    commitment: CommitmentExposureV2 = CommitmentExposureV2()

    def __post_init__(self) -> None:
        positive_fields = ("hours", "reference_delivery_kg", "reference_period_h")
        nonnegative_fields = (
            "requested_kg",
            "delivered_kg",
            "unmet_kg",
            "produced_kg",
            "solar_kwh",
            "curtailed_kwh",
            "battery_discharged_kwh",
            "guard_intervention_kwh",
            "dispatch_gap_kwh",
            "request_application_gap_kwh",
            "terminal_ch4_kg",
        )
        for name in (*positive_fields, *nonnegative_fields, "min_battery_soc"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"interval outcome {name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"interval outcome {name} must be finite")
        for name in positive_fields:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"interval outcome {name} must be positive")
        for name in nonnegative_fields:
            if getattr(self, name) < 0.0:
                raise ValueError(f"interval outcome {name} must be non-negative")
        if not 0.0 <= self.min_battery_soc <= 1.0:
            raise ValueError("interval outcome min_battery_soc must lie within [0, 1]")
        for name in ("physical_steps", "shutdown_steps", "corrections"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"interval outcome {name} must be an integer")
            if value < 0:
                raise ValueError(f"interval outcome {name} must be non-negative")
        if self.physical_steps == 0:
            raise ValueError("interval outcome physical_steps must be positive")
        if not isinstance(self.out_of_range, bool):
            raise TypeError("interval outcome out_of_range must be a bool")
        if not isinstance(self.commitment, CommitmentExposureV2):
            raise TypeError("interval outcome commitment must be CommitmentExposureV2")


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
    weights = (weights or RewardWeightsV2()).validate()
    reference = max(outcome.reference_delivery_kg, 1e-12)
    reference_period_h = max(outcome.reference_period_h, 1e-12)
    solar_energy_reference = max(
        config.solar.peak_kw * reference_period_h,
        1e-12,
    )
    discharge_reference = max(
        config.battery.max_discharge_kw * reference_period_h,
        1e-12,
    )
    electrolyser_energy_reference = max(
        config.electrolyser.rated_kw * reference_period_h,
        1e-12,
    )
    reference_physical_steps = max(
        1,
        int(round(reference_period_h / config.dt_h)),
    )
    reserve_deficit = max(0.0, constraints.reserve_soc(config) - outcome.min_battery_soc)
    signals = {
        "delivery": max(0.0, outcome.delivered_kg) / reference,
        "terminal_inventory": (
            min(1.0, max(0.0, outcome.terminal_ch4_kg) / reference) if terminal else 0.0
        ),
        "shortfall": max(0.0, outcome.unmet_kg) / reference,
        "curtailment": max(0.0, outcome.curtailed_kwh) / solar_energy_reference,
        "battery_drain": max(0.0, outcome.battery_discharged_kwh) / discharge_reference,
        "reserve_breach": (
            reserve_deficit * max(0.0, outcome.hours) / reference_period_h
        ),
        "shutdown": max(0, outcome.shutdown_steps) / reference_physical_steps,
        "correction": (
            (max(0, outcome.corrections) + (1 if outcome.out_of_range else 0))
            / (3.0 * reference_physical_steps)
        ),
        "nonproductive_energy": (
            max(0.0, outcome.commitment.nonproductive_energy_kwh)
            / electrolyser_energy_reference
        ),
        "forced_stop": float(max(0, outcome.commitment.forced_stops)),
        # Visible for diagnosis, deliberately absent from ``components``.
        "raw_production_kg": max(0.0, outcome.produced_kg),
        "cold_starts": float(max(0, outcome.commitment.cold_starts)),
        "hot_starts": float(max(0, outcome.commitment.hot_starts)),
        "aborted_starts": float(max(0, outcome.commitment.aborted_starts)),
        "controlled_stops": float(max(0, outcome.commitment.controlled_stops)),
        "startup_energy_kwh": max(0.0, outcome.commitment.startup_energy_kwh),
        "standby_energy_kwh": max(0.0, outcome.commitment.standby_energy_kwh),
        "operating_time_s": max(0.0, outcome.commitment.operating_time_s),
        "low_load_time_s": max(0.0, outcome.commitment.low_load_time_s),
        "ramp_limited_intervals": float(
            max(0, outcome.commitment.ramp_limited_intervals)
        ),
        "absolute_load_movement_kw": max(
            0.0,
            outcome.commitment.absolute_load_movement_kw,
        ),
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
        "nonproductive_energy": (
            -weights.nonproductive_energy * signals["nonproductive_energy"]
        ),
        "forced_stop": -weights.forced_stop * signals["forced_stop"],
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

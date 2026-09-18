"""Transparent economic and equipment-exposure objective for the V2 research task.

The numbers in this module are benchmark assumptions, not a vendor-calibrated
business case.  Physical safety remains in the independent guard.  This layer
only values service, electricity, replenishable starting endowments, and two
auditable life-consumption proxies already exposed by the simulator.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from numbers import Real
from typing import Any, Mapping

from plant_sim import PlantConfig, PlantState

from .boundary import CommitmentExposureV2
from .control import ControllerObservationV2

ECONOMIC_OBJECTIVE_VERSION = 1
TARIFF_SCHEMA = "synthetic-day-ahead-tariff-v1"
ECONOMIC_ASSUMPTIONS_SCHEMA = "economic-wear-assumptions-v1"
ECONOMIC_CONTROLLER_OBSERVATION_SCHEMA = "economic-controller-observation-v1"
MAX_ABS_TARIFF_EUR_PER_MWH = 300.0

# A deliberately stressful, repeating day-ahead shape.  It represents neither
# a historical day nor a price forecast.  Midday low/negative values and the
# evening peak make otherwise tied dispatch trajectories economically distinct.
DEFAULT_HOURLY_TARIFF_EUR_PER_MWH = (
    70.0,
    65.0,
    60.0,
    55.0,
    55.0,
    65.0,
    90.0,
    115.0,
    105.0,
    75.0,
    45.0,
    15.0,
    -10.0,
    -5.0,
    20.0,
    45.0,
    80.0,
    125.0,
    150.0,
    120.0,
    100.0,
    90.0,
    80.0,
    75.0,
)


def _real(
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
        raise ValueError(f"{name} must be at least {minimum:g}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be at most {maximum:g}")
    return number


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class TariffObservationV1:
    """Causal price information exposed at one decision boundary."""

    current_eur_per_mwh: float
    next_1h_mean_eur_per_mwh: float
    next_6h_min_eur_per_mwh: float
    next_6h_mean_eur_per_mwh: float
    next_6h_max_eur_per_mwh: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _real(
                name,
                value,
                minimum=-MAX_ABS_TARIFF_EUR_PER_MWH,
                maximum=MAX_ABS_TARIFF_EUR_PER_MWH,
            )

    def normalized(self) -> tuple[float, ...]:
        return tuple(
            float(value) / MAX_ABS_TARIFF_EUR_PER_MWH for value in asdict(self).values()
        )

    def to_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}


@dataclass(frozen=True)
class EconomicControllerObservationV1:
    """Detached information source shared by structured and vector controllers."""

    plant: ControllerObservationV2
    tariff: TariffObservationV1
    tariff_id: str
    economic_assumptions_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.plant, ControllerObservationV2):
            raise TypeError("plant must be ControllerObservationV2")
        if not isinstance(self.tariff, TariffObservationV1):
            raise TypeError("tariff must be TariffObservationV1")
        if not isinstance(self.tariff_id, str) or not self.tariff_id.strip():
            raise ValueError("tariff_id must be a non-empty string")
        digest = self.economic_assumptions_sha256
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("economic assumptions digest must be SHA-256")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError("economic assumptions digest must be SHA-256") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ECONOMIC_CONTROLLER_OBSERVATION_SCHEMA,
            "plant": self.plant.to_dict(),
            "tariff": self.tariff.to_dict(),
            "tariff_id": self.tariff_id,
            "economic_assumptions_sha256": self.economic_assumptions_sha256,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class DayAheadTariffV1:
    """A public, deterministic, hourly tariff repeated every civil day."""

    id: str = "synthetic-eu-flexibility-stress-v1"
    hourly_eur_per_mwh: tuple[float, ...] = DEFAULT_HOURLY_TARIFF_EUR_PER_MWH

    def validate(self) -> "DayAheadTariffV1":
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("tariff id must be a non-empty string")
        if not isinstance(self.hourly_eur_per_mwh, tuple):
            raise TypeError("hourly tariff must be an immutable tuple")
        if len(self.hourly_eur_per_mwh) != 24:
            raise ValueError("hourly tariff must contain exactly 24 prices")
        for hour, price in enumerate(self.hourly_eur_per_mwh):
            _real(
                f"tariff hour {hour}",
                price,
                minimum=-MAX_ABS_TARIFF_EUR_PER_MWH,
                maximum=MAX_ABS_TARIFF_EUR_PER_MWH,
            )
        return self

    def price_at(self, time_h: float) -> float:
        self.validate()
        time_h = _real("tariff time_h", time_h, minimum=0.0)
        hour = int(math.floor(time_h % 24.0))
        return float(self.hourly_eur_per_mwh[hour])

    def window_summary(
        self, time_h: float, duration_h: float
    ) -> tuple[float, float, float]:
        """Return exact piecewise-constant minimum, mean, and maximum."""

        self.validate()
        cursor = _real("tariff time_h", time_h, minimum=0.0)
        duration_h = _real("tariff duration_h", duration_h, minimum=1e-12)
        end = cursor + duration_h
        weighted_sum = 0.0
        prices: list[float] = []
        while cursor < end - 1e-12:
            next_hour = math.floor(cursor) + 1.0
            segment_end = min(end, next_hour)
            if segment_end <= cursor:
                segment_end = min(end, cursor + 1.0)
            price = self.price_at(cursor)
            width = segment_end - cursor
            weighted_sum += price * width
            prices.append(price)
            cursor = segment_end
        if not prices:
            raise RuntimeError("tariff window did not contain a positive segment")
        return min(prices), weighted_sum / duration_h, max(prices)

    def observe(self, time_h: float) -> TariffObservationV1:
        _, next_1h_mean, _ = self.window_summary(time_h, 1.0)
        next_6h_min, next_6h_mean, next_6h_max = self.window_summary(time_h, 6.0)
        return TariffObservationV1(
            current_eur_per_mwh=self.price_at(time_h),
            next_1h_mean_eur_per_mwh=next_1h_mean,
            next_6h_min_eur_per_mwh=next_6h_min,
            next_6h_mean_eur_per_mwh=next_6h_mean,
            next_6h_max_eur_per_mwh=next_6h_max,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TARIFF_SCHEMA,
            "id": self.id,
            "hourly_eur_per_mwh": [float(value) for value in self.hourly_eur_per_mwh],
            "negative_price_semantics": (
                "renewable energy may be curtailed, so its opportunity price is floored "
                "at zero in the objective"
            ),
        }


@dataclass(frozen=True)
class EconomicAssumptionsV1:
    """Explicit alpha assumptions; every value is checkpoint-bound and ablatable."""

    tariff: DayAheadTariffV1 = field(default_factory=DayAheadTariffV1)
    delivered_service_value_eur_per_kg: float = 4.0
    unmet_delivery_penalty_eur_per_kg: float = 20.0
    methane_endowment_value_eur_per_kg_ch4_equivalent: float = 4.0
    battery_endowment_value_eur_per_kwh: float = 0.15
    battery_replacement_eur_per_kwh: float = 300.0
    battery_full_cycle_life: float = 2_640.0
    electrolyser_capital_eur_per_kw: float = 480.0
    electrolyser_life_h: float = 80_000.0
    electrolyser_low_load_wear_multiplier: float = 1.5
    cold_start_equivalent_operating_h: float = 2.0
    hot_start_equivalent_operating_h: float = 0.25
    aborted_start_equivalent_operating_h: float = 2.0
    forced_stop_equivalent_operating_h: float = 4.0

    def validate(self) -> "EconomicAssumptionsV1":
        if not isinstance(self.tariff, DayAheadTariffV1):
            raise TypeError("tariff must be DayAheadTariffV1")
        self.tariff.validate()
        strictly_positive = {
            "unmet_delivery_penalty_eur_per_kg",
            "methane_endowment_value_eur_per_kg_ch4_equivalent",
            "battery_endowment_value_eur_per_kwh",
            "battery_replacement_eur_per_kwh",
            "battery_full_cycle_life",
            "electrolyser_capital_eur_per_kw",
            "electrolyser_life_h",
            "electrolyser_low_load_wear_multiplier",
        }
        for name, value in asdict(self).items():
            if name == "tariff":
                continue
            minimum = 1e-12 if name in strictly_positive else 0.0
            _real(name, value, minimum=minimum)
        if self.unmet_delivery_penalty_eur_per_kg <= (
            self.delivered_service_value_eur_per_kg
        ):
            raise ValueError(
                "unmet delivery penalty must exceed delivered service value"
            )
        if self.electrolyser_low_load_wear_multiplier < 1.0:
            raise ValueError("low-load wear multiplier must be at least one")
        return self

    @property
    def battery_throughput_eur_per_kwh(self) -> float:
        """Linear replacement reserve per bus-side charge/discharge kWh."""

        return self.battery_replacement_eur_per_kwh / (
            2.0 * self.battery_full_cycle_life
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = {
            name: float(value)
            for name, value in asdict(self).items()
            if name != "tariff"
        }
        return {
            "schema": ECONOMIC_ASSUMPTIONS_SCHEMA,
            "calibration_status": (
                "literature-anchored benchmark assumptions; not a vendor quote, "
                "site tariff, or bankable financial model"
            ),
            "tariff": self.tariff.to_dict(),
            **payload,
            "battery_throughput_eur_per_kwh": self.battery_throughput_eur_per_kwh,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


def ch4_equivalent_inventory_kg(state: PlantState, config: PlantConfig) -> float:
    """Saleable methane plus immediately stoichiometric H2/CO2 feedstock."""

    if not isinstance(state, PlantState):
        raise TypeError("state must be PlantState")
    h2_ratio = _real(
        "methanator H2 ratio",
        config.methanator.h2_kg_per_kg_ch4,
        minimum=1e-12,
    )
    co2_ratio = _real(
        "methanator CO2 ratio",
        config.methanator.co2_kg_per_kg_ch4,
        minimum=1e-12,
    )
    ch4 = _real("CH4 inventory", state.ch4_level_kg, minimum=0.0)
    h2 = _real("H2 inventory", state.h2_level_kg, minimum=0.0)
    co2 = _real("CO2 inventory", state.co2_level_kg, minimum=0.0)
    return ch4 + min(h2 / h2_ratio, co2 / co2_ratio)


@dataclass(frozen=True)
class EconomicIntervalOutcomeV1:
    """Physical and endowment deltas needed by the economic objective."""

    opportunity_price_eur_per_mwh: float
    accepted_solar_kwh: float
    delivered_kg: float
    unmet_kg: float
    battery_throughput_kwh: float
    initial_ch4_equivalent_kg: float
    ch4_equivalent_before_kg: float
    ch4_equivalent_after_kg: float
    initial_battery_energy_kwh: float
    battery_energy_before_kwh: float
    battery_energy_after_kwh: float
    commitment: CommitmentExposureV2

    def __post_init__(self) -> None:
        _real(
            "opportunity price",
            self.opportunity_price_eur_per_mwh,
            minimum=-MAX_ABS_TARIFF_EUR_PER_MWH,
            maximum=MAX_ABS_TARIFF_EUR_PER_MWH,
        )
        for name, value in asdict(self).items():
            if name in {"opportunity_price_eur_per_mwh", "commitment"}:
                continue
            _real(name, value, minimum=0.0)
        if not isinstance(self.commitment, CommitmentExposureV2):
            raise TypeError("commitment must be CommitmentExposureV2")


@dataclass(frozen=True)
class EconomicRewardBreakdownV1:
    total_eur: float
    components_eur: dict[str, float]
    signals: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_version": ECONOMIC_OBJECTIVE_VERSION,
            "unit": "EUR per decision interval",
            "total_eur": self.total_eur,
            "components_eur": dict(self.components_eur),
            "signals": dict(self.signals),
        }


def compute_economic_reward(
    outcome: EconomicIntervalOutcomeV1,
    *,
    config: PlantConfig,
    assumptions: EconomicAssumptionsV1,
) -> EconomicRewardBreakdownV1:
    """Compute auditable operating value; hard safety is never monetized here."""

    assumptions.validate()
    rated_kw = _real(
        "electrolyser rated power", config.electrolyser.rated_kw, minimum=1e-12
    )
    battery_capacity_kwh = _real(
        "battery capacity", config.battery.capacity_kwh, minimum=1e-12
    )
    exposure = outcome.commitment
    operating_h = exposure.operating_time_s / 3_600.0
    low_load_h = exposure.low_load_time_s / 3_600.0
    equivalent_operating_h = (
        operating_h
        + low_load_h * (assumptions.electrolyser_low_load_wear_multiplier - 1.0)
        + exposure.cold_starts * assumptions.cold_start_equivalent_operating_h
        + exposure.hot_starts * assumptions.hot_start_equivalent_operating_h
        + exposure.aborted_starts * assumptions.aborted_start_equivalent_operating_h
        + exposure.forced_stops * assumptions.forced_stop_equivalent_operating_h
    )
    electrolyser_reserve_eur_per_h = (
        rated_kw
        * assumptions.electrolyser_capital_eur_per_kw
        / assumptions.electrolyser_life_h
    )
    effective_opportunity_price = max(0.0, outcome.opportunity_price_eur_per_mwh)
    solar_opportunity_cost_eur = (
        outcome.accepted_solar_kwh * effective_opportunity_price / 1_000.0
    )
    battery_throughput_reserve_eur = (
        outcome.battery_throughput_kwh * assumptions.battery_throughput_eur_per_kwh
    )
    electrolyser_life_reserve_eur = (
        equivalent_operating_h * electrolyser_reserve_eur_per_h
    )

    methane_deficit_before = max(
        0.0,
        outcome.initial_ch4_equivalent_kg - outcome.ch4_equivalent_before_kg,
    )
    methane_deficit_after = max(
        0.0,
        outcome.initial_ch4_equivalent_kg - outcome.ch4_equivalent_after_kg,
    )
    battery_deficit_before = max(
        0.0,
        outcome.initial_battery_energy_kwh - outcome.battery_energy_before_kwh,
    )
    battery_deficit_after = max(
        0.0,
        outcome.initial_battery_energy_kwh - outcome.battery_energy_after_kwh,
    )
    methane_liability_change_eur = (
        methane_deficit_after - methane_deficit_before
    ) * assumptions.methane_endowment_value_eur_per_kg_ch4_equivalent
    battery_liability_change_eur = (
        battery_deficit_after - battery_deficit_before
    ) * assumptions.battery_endowment_value_eur_per_kwh

    components = {
        "delivered_service_value": (
            outcome.delivered_kg * assumptions.delivered_service_value_eur_per_kg
        ),
        "unmet_delivery_penalty": -(
            outcome.unmet_kg * assumptions.unmet_delivery_penalty_eur_per_kg
        ),
        "solar_opportunity_cost": -solar_opportunity_cost_eur,
        "battery_throughput_reserve": -battery_throughput_reserve_eur,
        "electrolyser_life_reserve": -electrolyser_life_reserve_eur,
        "methane_endowment_liability_change": -methane_liability_change_eur,
        "battery_endowment_liability_change": -battery_liability_change_eur,
    }
    signals = {
        "opportunity_price_eur_per_mwh": outcome.opportunity_price_eur_per_mwh,
        "effective_opportunity_price_eur_per_mwh": effective_opportunity_price,
        "accepted_solar_kwh": outcome.accepted_solar_kwh,
        "delivered_kg": outcome.delivered_kg,
        "unmet_kg": outcome.unmet_kg,
        "battery_throughput_kwh": outcome.battery_throughput_kwh,
        "battery_equivalent_full_cycles": outcome.battery_throughput_kwh
        / (2.0 * battery_capacity_kwh),
        "electrolyser_equivalent_operating_h": equivalent_operating_h,
        "methane_endowment_deficit_before_kg": methane_deficit_before,
        "methane_endowment_deficit_after_kg": methane_deficit_after,
        "battery_endowment_deficit_before_kwh": battery_deficit_before,
        "battery_endowment_deficit_after_kwh": battery_deficit_after,
        "solar_opportunity_cost_eur": solar_opportunity_cost_eur,
        "battery_throughput_reserve_eur": battery_throughput_reserve_eur,
        "electrolyser_life_reserve_eur": electrolyser_life_reserve_eur,
    }
    total = float(sum(components.values()))
    if not math.isfinite(total):
        raise ValueError("economic reward must be finite")
    return EconomicRewardBreakdownV1(
        total_eur=total,
        components_eur=components,
        signals=signals,
    )


@dataclass
class EconomicEpisodeMetricsV1:
    """Additive economic ledger kept separate from historical V2 metrics."""

    initial_ch4_equivalent_kg: float
    initial_battery_energy_kwh: float
    battery_capacity_kwh: float
    decisions: int = 0
    reward_total_eur: float = 0.0
    components_eur: dict[str, float] = field(default_factory=dict)
    accepted_solar_kwh: float = 0.0
    delivered_kg: float = 0.0
    unmet_kg: float = 0.0
    battery_throughput_kwh: float = 0.0
    electrolyser_equivalent_operating_h: float = 0.0
    terminal_ch4_equivalent_kg: float = 0.0
    terminal_battery_energy_kwh: float = 0.0

    def __post_init__(self) -> None:
        for name in ("initial_ch4_equivalent_kg", "initial_battery_energy_kwh"):
            _real(name, getattr(self, name), minimum=0.0)
        _real("battery_capacity_kwh", self.battery_capacity_kwh, minimum=1e-12)
        self.terminal_ch4_equivalent_kg = self.initial_ch4_equivalent_kg
        self.terminal_battery_energy_kwh = self.initial_battery_energy_kwh

    def record(
        self,
        outcome: EconomicIntervalOutcomeV1,
        breakdown: EconomicRewardBreakdownV1,
    ) -> None:
        self.decisions += 1
        self.reward_total_eur += breakdown.total_eur
        for name, value in breakdown.components_eur.items():
            self.components_eur[name] = self.components_eur.get(name, 0.0) + value
        self.accepted_solar_kwh += outcome.accepted_solar_kwh
        self.delivered_kg += outcome.delivered_kg
        self.unmet_kg += outcome.unmet_kg
        self.battery_throughput_kwh += outcome.battery_throughput_kwh
        self.electrolyser_equivalent_operating_h += breakdown.signals[
            "electrolyser_equivalent_operating_h"
        ]
        self.terminal_ch4_equivalent_kg = outcome.ch4_equivalent_after_kg
        self.terminal_battery_energy_kwh = outcome.battery_energy_after_kwh

    @property
    def battery_equivalent_full_cycles(self) -> float:
        return self.battery_throughput_kwh / (2.0 * self.battery_capacity_kwh)

    @property
    def terminal_ch4_equivalent_deficit_kg(self) -> float:
        return max(
            0.0,
            self.initial_ch4_equivalent_kg - self.terminal_ch4_equivalent_kg,
        )

    @property
    def terminal_battery_deficit_kwh(self) -> float:
        return max(
            0.0,
            self.initial_battery_energy_kwh - self.terminal_battery_energy_kwh,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_version": ECONOMIC_OBJECTIVE_VERSION,
            "unit": "EUR",
            "decisions": self.decisions,
            "net_operating_value_eur": self.reward_total_eur,
            "components_eur": dict(self.components_eur),
            "accepted_solar_kwh": self.accepted_solar_kwh,
            "delivered_kg": self.delivered_kg,
            "unmet_kg": self.unmet_kg,
            "battery_throughput_kwh": self.battery_throughput_kwh,
            "battery_equivalent_full_cycles": self.battery_equivalent_full_cycles,
            "electrolyser_equivalent_operating_h": (
                self.electrolyser_equivalent_operating_h
            ),
            "initial_ch4_equivalent_kg": self.initial_ch4_equivalent_kg,
            "terminal_ch4_equivalent_kg": self.terminal_ch4_equivalent_kg,
            "terminal_ch4_equivalent_deficit_kg": (
                self.terminal_ch4_equivalent_deficit_kg
            ),
            "initial_battery_energy_kwh": self.initial_battery_energy_kwh,
            "terminal_battery_energy_kwh": self.terminal_battery_energy_kwh,
            "terminal_battery_deficit_kwh": self.terminal_battery_deficit_kwh,
        }


__all__ = [
    "DEFAULT_HOURLY_TARIFF_EUR_PER_MWH",
    "ECONOMIC_ASSUMPTIONS_SCHEMA",
    "ECONOMIC_CONTROLLER_OBSERVATION_SCHEMA",
    "ECONOMIC_OBJECTIVE_VERSION",
    "MAX_ABS_TARIFF_EUR_PER_MWH",
    "TARIFF_SCHEMA",
    "DayAheadTariffV1",
    "EconomicAssumptionsV1",
    "EconomicControllerObservationV1",
    "EconomicEpisodeMetricsV1",
    "EconomicIntervalOutcomeV1",
    "EconomicRewardBreakdownV1",
    "TariffObservationV1",
    "ch4_equivalent_inventory_kg",
    "compute_economic_reward",
]

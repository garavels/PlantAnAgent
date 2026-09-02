"""Explicit decision-boundary telemetry for multi-rate Environment V2."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from numbers import Real
from typing import Any

from plant_sim import PlantState


@dataclass(frozen=True)
class IntervalSummaryV2:
    """Flows integrated over the physical interval preceding a decision."""

    physical_steps: int = 0
    hours: float = 0.0
    solar_kwh: float = 0.0
    solar_potential_kwh: float = 0.0
    solar_curtailed_kwh: float = 0.0
    battery_net_kwh: float = 0.0
    electrolyser_kwh: float = 0.0
    dac_kwh: float = 0.0
    methanator_kwh: float = 0.0
    h2_produced_kg: float = 0.0
    co2_captured_kg: float = 0.0
    ch4_produced_kg: float = 0.0
    electrolyser_load_fraction_h: float = 0.0
    dac_load_fraction_h: float = 0.0
    methanator_load_fraction_h: float = 0.0
    electrolyser_online_h: float = 0.0
    dac_online_h: float = 0.0
    methanator_online_h: float = 0.0

    def _mean_power(self, energy_kwh: float) -> float:
        return energy_kwh / self.hours if self.hours > 0.0 else 0.0

    def _mean_rate(self, mass_kg: float) -> float:
        return mass_kg / self.hours if self.hours > 0.0 else 0.0

    def _online_fraction(self, online_h: float) -> float:
        return online_h / self.hours if self.hours > 0.0 else 0.0

    @property
    def solar_available_mean_kw(self) -> float:
        return self._mean_power(self.solar_kwh)

    @property
    def solar_potential_mean_kw(self) -> float:
        return self._mean_power(self.solar_potential_kwh)

    @property
    def solar_output_factor(self) -> float:
        if self.solar_potential_kwh <= 0.0:
            return 1.0
        return self.solar_kwh / self.solar_potential_kwh

    @property
    def battery_power_mean_kw(self) -> float:
        return self._mean_power(self.battery_net_kwh)

    @property
    def electrolyser_power_mean_kw(self) -> float:
        return self._mean_power(self.electrolyser_kwh)

    @property
    def dac_power_mean_kw(self) -> float:
        return self._mean_power(self.dac_kwh)

    @property
    def methanator_power_mean_kw(self) -> float:
        return self._mean_power(self.methanator_kwh)

    @property
    def h2_production_mean_kg_h(self) -> float:
        return self._mean_rate(self.h2_produced_kg)

    @property
    def co2_capture_mean_kg_h(self) -> float:
        return self._mean_rate(self.co2_captured_kg)

    @property
    def ch4_production_mean_kg_h(self) -> float:
        return self._mean_rate(self.ch4_produced_kg)

    @property
    def electrolyser_load_mean(self) -> float:
        return self._online_fraction(self.electrolyser_load_fraction_h)

    @property
    def dac_load_mean(self) -> float:
        return self._online_fraction(self.dac_load_fraction_h)

    @property
    def methanator_load_mean(self) -> float:
        return self._online_fraction(self.methanator_load_fraction_h)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "solar_available_mean_kw": self.solar_available_mean_kw,
            "solar_potential_mean_kw": self.solar_potential_mean_kw,
            "solar_output_factor": self.solar_output_factor,
            "solar_curtailed_mean_kw": self._mean_power(self.solar_curtailed_kwh),
            "battery_power_mean_kw": self.battery_power_mean_kw,
            "electrolyser_power_mean_kw": self.electrolyser_power_mean_kw,
            "dac_power_mean_kw": self.dac_power_mean_kw,
            "methanator_power_mean_kw": self.methanator_power_mean_kw,
            "h2_production_mean_kg_h": self.h2_production_mean_kg_h,
            "co2_capture_mean_kg_h": self.co2_capture_mean_kg_h,
            "ch4_production_mean_kg_h": self.ch4_production_mean_kg_h,
            "electrolyser_load_mean": self.electrolyser_load_mean,
            "dac_load_mean": self.dac_load_mean,
            "methanator_load_mean": self.methanator_load_mean,
            "electrolyser_online_fraction": self._online_fraction(
                self.electrolyser_online_h
            ),
            "dac_online_fraction": self._online_fraction(self.dac_online_h),
            "methanator_online_fraction": self._online_fraction(
                self.methanator_online_h
            ),
        }


@dataclass
class IntervalTelemetryAccumulator:
    """Mutable integrator used only while one policy decision is executing."""

    physical_steps: int = 0
    hours: float = 0.0
    solar_kwh: float = 0.0
    solar_potential_kwh: float = 0.0
    solar_curtailed_kwh: float = 0.0
    battery_net_kwh: float = 0.0
    electrolyser_kwh: float = 0.0
    dac_kwh: float = 0.0
    methanator_kwh: float = 0.0
    h2_produced_kg: float = 0.0
    co2_captured_kg: float = 0.0
    ch4_produced_kg: float = 0.0
    electrolyser_load_fraction_h: float = 0.0
    dac_load_fraction_h: float = 0.0
    methanator_load_fraction_h: float = 0.0
    electrolyser_online_h: float = 0.0
    dac_online_h: float = 0.0
    methanator_online_h: float = 0.0

    def record(self, state: PlantState, *, dt_h: float) -> None:
        if isinstance(dt_h, bool) or not isinstance(dt_h, Real):
            raise TypeError("dt_h must be a real number")
        step_h = float(dt_h)
        if not math.isfinite(step_h) or step_h <= 0.0:
            raise ValueError("dt_h must be positive and finite")
        self.physical_steps += 1
        self.hours += step_h
        self.solar_kwh += state.solar_available_kw * step_h
        self.solar_potential_kwh += state.solar_potential_kw * step_h
        self.solar_curtailed_kwh += state.solar_curtailed_kw * step_h
        self.battery_net_kwh += state.battery_power_kw * step_h
        self.electrolyser_kwh += state.electrolyser_power_kw * step_h
        self.dac_kwh += state.dac_power_kw * step_h
        self.methanator_kwh += state.methanator_power_kw * step_h
        self.h2_produced_kg += state.h2_production_kg_h * step_h
        self.co2_captured_kg += state.co2_capture_kg_h * step_h
        self.ch4_produced_kg += state.ch4_production_kg_h * step_h
        self.electrolyser_load_fraction_h += state.electrolyser_load_frac * step_h
        self.dac_load_fraction_h += state.dac_load_frac * step_h
        self.methanator_load_fraction_h += state.methanator_load_frac * step_h
        self.electrolyser_online_h += float(state.electrolyser_online) * step_h
        self.dac_online_h += float(state.dac_online) * step_h
        self.methanator_online_h += float(state.methanator_online) * step_h

    def summary(self) -> IntervalSummaryV2:
        return IntervalSummaryV2(**asdict(self))


@dataclass(frozen=True)
class ControlStateV2:
    """Boundary inventories/health plus explicit preceding-interval flows."""

    boundary: PlantState
    previous_interval: IntervalSummaryV2 = IntervalSummaryV2()

    def encoder_telemetry(self) -> PlantState:
        """V1-shaped view whose rate/load fields have declared 15-minute meaning."""
        interval = self.previous_interval
        if interval.physical_steps == 0:
            return self.boundary
        return replace(
            self.boundary,
            battery_power_kw=interval.battery_power_mean_kw,
            electrolyser_power_kw=interval.electrolyser_power_mean_kw,
            electrolyser_load_frac=interval.electrolyser_load_mean,
            h2_production_kg_h=interval.h2_production_mean_kg_h,
            dac_power_kw=interval.dac_power_mean_kw,
            dac_load_frac=interval.dac_load_mean,
            co2_capture_kg_h=interval.co2_capture_mean_kg_h,
            methanator_power_kw=interval.methanator_power_mean_kw,
            methanator_load_frac=interval.methanator_load_mean,
            ch4_production_kg_h=interval.ch4_production_mean_kg_h,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary.to_dict(),
            "previous_interval": self.previous_interval.to_dict(),
        }


__all__ = [
    "ControlStateV2",
    "IntervalSummaryV2",
    "IntervalTelemetryAccumulator",
]

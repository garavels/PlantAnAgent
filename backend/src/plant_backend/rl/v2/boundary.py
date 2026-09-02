"""Current inventories plus explicit completed-interval telemetry for V2."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from typing import Any

from plant_sim import ElectrolyserCommitmentSnapshot, ElectrolyserMode, PlantState


COMMITMENT_MODE_NAMES = (
    "legacy",
    *(mode.value for mode in ElectrolyserMode),
)


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
            return 0.0
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
class CommitmentExposureV2:
    """Exact cumulative-counter deltas over one controller decision."""

    enabled: bool = False
    cold_starts: int = 0
    hot_starts: int = 0
    aborted_starts: int = 0
    forced_stops: int = 0
    controlled_stops: int = 0
    startup_energy_kwh: float = 0.0
    standby_energy_kwh: float = 0.0
    operating_time_s: float = 0.0
    low_load_time_s: float = 0.0
    ramp_limited_intervals: int = 0
    absolute_load_movement_kw: float = 0.0
    mode_time_s: tuple[tuple[str, float], ...] = tuple(
        (name, 0.0) for name in COMMITMENT_MODE_NAMES
    )

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("commitment exposure enabled must be a bool")
        integer_fields = (
            "cold_starts",
            "hot_starts",
            "aborted_starts",
            "forced_stops",
            "controlled_stops",
            "ramp_limited_intervals",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"commitment exposure {name} must be an integer")
            if value < 0:
                raise ValueError(f"commitment exposure {name} must be non-negative")
        float_fields = (
            "startup_energy_kwh",
            "standby_energy_kwh",
            "operating_time_s",
            "low_load_time_s",
            "absolute_load_movement_kw",
        )
        for name in float_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"commitment exposure {name} must be a real number")
            if not math.isfinite(float(value)) or value < 0.0:
                raise ValueError(
                    f"commitment exposure {name} must be finite and non-negative"
                )

        if not isinstance(self.mode_time_s, tuple):
            raise TypeError("commitment exposure mode_time_s must be an immutable tuple")
        names: list[str] = []
        for item in self.mode_time_s:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("each commitment mode-time item must be a pair")
            name, seconds = item
            names.append(name)
            if isinstance(seconds, bool) or not isinstance(seconds, Real):
                raise TypeError("commitment mode time must be a real number")
            if not math.isfinite(float(seconds)) or seconds < 0.0:
                raise ValueError("commitment mode time must be finite and non-negative")
        if tuple(names) != COMMITMENT_MODE_NAMES:
            raise ValueError("commitment mode-time fields must use the canonical order")
        mode_times = dict(self.mode_time_s)
        if self.enabled and mode_times["legacy"] > 0.0:
            raise ValueError("enabled commitment exposure cannot contain legacy mode time")
        if not self.enabled and any(
            getattr(self, name) != 0 for name in (*integer_fields, *float_fields)
        ):
            raise ValueError("disabled commitment exposure cannot contain equipment counters")
        if not self.enabled and any(
            mode_times[name] > 0.0 for name in COMMITMENT_MODE_NAMES if name != "legacy"
        ):
            raise ValueError("disabled commitment exposure can contain only legacy mode time")

    @classmethod
    def between(
        cls,
        before: ElectrolyserCommitmentSnapshot | None,
        after: ElectrolyserCommitmentSnapshot | None,
        *,
        interval_s: float,
    ) -> "CommitmentExposureV2":
        if isinstance(interval_s, bool) or not isinstance(interval_s, Real):
            raise TypeError("interval_s must be a real number")
        if not math.isfinite(float(interval_s)) or interval_s < 0.0:
            raise ValueError("interval_s must be finite and non-negative")
        if before is None and after is None:
            return cls(
                mode_time_s=tuple(
                    (name, interval_s if name == "legacy" else 0.0)
                    for name in COMMITMENT_MODE_NAMES
                )
            )
        if before is None or after is None:
            raise ValueError("commitment cannot be enabled or disabled inside an interval")
        provenance_fields = (
            "profile_id",
            "technology",
            "calibration_basis",
            "dynamics_version",
            "profile_sha256",
        )
        if any(
            getattr(before, name) != getattr(after, name)
            for name in provenance_fields
        ):
            raise ValueError("commitment profile cannot change inside an interval")

        def integer_delta(name: str) -> int:
            before_value = getattr(before, name)
            after_value = getattr(after, name)
            if any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in (before_value, after_value)
            ):
                raise TypeError(f"commitment counter {name} must remain integral")
            delta = int(after_value) - int(before_value)
            if delta < 0:
                raise ValueError(f"commitment counter {name} moved backwards")
            return delta

        def float_delta(name: str) -> float:
            before_value = float(getattr(before, name))
            after_value = float(getattr(after, name))
            if not (math.isfinite(before_value) and math.isfinite(after_value)):
                raise ValueError(f"commitment counter {name} must remain finite")
            delta = after_value - before_value
            if delta < -1e-9:
                raise ValueError(f"commitment counter {name} moved backwards")
            return max(0.0, delta)

        if any(
            not isinstance(mode, ElectrolyserMode)
            for snapshot in (before, after)
            for mode, _ in snapshot.mode_time_s
        ):
            raise TypeError("commitment snapshot mode-time keys must be ElectrolyserMode")
        before_modes = {mode.value: seconds for mode, seconds in before.mode_time_s}
        after_modes = {mode.value: seconds for mode, seconds in after.mode_time_s}
        expected_modes = set(COMMITMENT_MODE_NAMES) - {"legacy"}
        if (
            len(before.mode_time_s) != len(expected_modes)
            or len(after.mode_time_s) != len(expected_modes)
            or set(before_modes) != expected_modes
            or set(after_modes) != expected_modes
        ):
            raise ValueError("commitment mode-time counters are incomplete or duplicated")
        mode_deltas: dict[str, float] = {}
        for name in COMMITMENT_MODE_NAMES:
            if name == "legacy":
                continue
            before_value = float(before_modes[name])
            after_value = float(after_modes[name])
            if not (math.isfinite(before_value) and math.isfinite(after_value)):
                raise ValueError("commitment mode-time counters must remain finite")
            delta = after_value - before_value
            if delta < -1e-9:
                raise ValueError(
                    f"commitment mode-time counter {name} moved backwards"
                )
            mode_deltas[name] = max(0.0, delta)
        if not math.isclose(
            sum(mode_deltas.values()),
            interval_s,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError("commitment mode-time deltas do not cover the interval")
        return cls(
            enabled=True,
            cold_starts=integer_delta("cold_starts"),
            hot_starts=integer_delta("hot_starts"),
            aborted_starts=integer_delta("aborted_starts"),
            forced_stops=integer_delta("forced_stops"),
            controlled_stops=integer_delta("controlled_stops"),
            startup_energy_kwh=float_delta("startup_energy_kwh"),
            standby_energy_kwh=float_delta("standby_energy_kwh"),
            operating_time_s=float_delta("operating_time_s"),
            low_load_time_s=float_delta("low_load_time_s"),
            ramp_limited_intervals=integer_delta("ramp_limited_intervals"),
            absolute_load_movement_kw=float_delta("absolute_load_movement_kw"),
            mode_time_s=tuple(
                (name, 0.0 if name == "legacy" else mode_deltas[name])
                for name in COMMITMENT_MODE_NAMES
            ),
        )

    @property
    def nonproductive_energy_kwh(self) -> float:
        return self.startup_energy_kwh + self.standby_energy_kwh

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mode_time_s"] = dict(self.mode_time_s)
        payload["nonproductive_energy_kwh"] = self.nonproductive_energy_kwh
        return payload


@dataclass(frozen=True)
class ControlStateV2:
    """Current inventories plus final-sample and aggregate interval telemetry."""

    boundary: PlantState
    previous_interval: IntervalSummaryV2 = IntervalSummaryV2()
    electrolyser_commitment: ElectrolyserCommitmentSnapshot | None = None

    def encoder_telemetry(self) -> PlantState:
        """V1-shaped view whose rate/load fields have declared interval meaning."""
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
            "electrolyser_commitment": (
                None
                if self.electrolyser_commitment is None
                else self.electrolyser_commitment.to_dict()
            ),
        }


__all__ = [
    "COMMITMENT_MODE_NAMES",
    "CommitmentExposureV2",
    "ControlStateV2",
    "IntervalSummaryV2",
    "IntervalTelemetryAccumulator",
]

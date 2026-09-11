"""Causal deterministic solar reconstruction for the service-MPC controller.

The V2 forecast exposes four aggregate available-PV bands.  An optimizer needs a
trajectory, but it must not invent access to the private weather tape.  This module
uses only one :class:`PublicForecastIssueV2` and public clear-sky plant geometry.  It
scales the analytic clear-sky shape independently inside every band so that darkness
is preserved and the issued band energy is reconstructed to numerical tolerance.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Any, Sequence

from plant_sim import PlantConfig

from .weather import FORECAST_WINDOWS_H, PublicForecastIssueV2

SECONDS_PER_HOUR = 3_600.0
HOURS_PER_DAY = 24.0
FORECAST_PROJECTION_VERSION = "issued-mean-clear-sky-scaled-v2"
P10_FORECAST_PROJECTION_VERSION = "issued-band-p10-clear-sky-scaled-v1"
FORECAST_TRAJECTORY_DOMAIN = "service-mpc-forecast-trajectory-v1"
PUBLIC_FORECAST_ISSUE_DOMAIN = "service-mpc-public-forecast-issue-v1"
DEFAULT_ENERGY_TOLERANCE = 1e-12


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _domain_sha256(domain: str, value: object) -> str:
    preimage = f"{domain}\0{_canonical_json(value)}".encode("utf-8")
    return hashlib.sha256(preimage).hexdigest()


def public_forecast_issue_sha256(forecast: PublicForecastIssueV2) -> str:
    """Bind a projection to the complete detached public issuance."""

    if not isinstance(forecast, PublicForecastIssueV2):
        raise TypeError("forecast must be PublicForecastIssueV2")
    return _domain_sha256(PUBLIC_FORECAST_ISSUE_DOMAIN, forecast.to_dict())


def _finite_real(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}")
    return number


def _positive_aligned_steps(name: str, seconds: float, interval_s: float) -> int:
    value = _finite_real(name, seconds, minimum=0.0)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    steps = int(round(value / interval_s))
    if steps <= 0 or not math.isclose(
        value,
        steps * interval_s,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{name} must align to interval_s={interval_s:g}")
    return steps


def clear_sky_equivalent_hours(
    *,
    boundary_time_s: float,
    start_offset_s: float,
    end_offset_s: float,
    sunrise_h: float,
    sunset_h: float,
) -> float:
    """Integrate the unit-peak public half-sine over one offset interval."""

    boundary_s = _finite_real("boundary_time_s", boundary_time_s, minimum=0.0)
    start_s = _finite_real("start_offset_s", start_offset_s, minimum=0.0)
    end_s = _finite_real("end_offset_s", end_offset_s, minimum=0.0)
    sunrise = _finite_real("sunrise_h", sunrise_h, minimum=0.0)
    sunset = _finite_real("sunset_h", sunset_h, minimum=0.0)
    if not 0.0 <= sunrise < sunset <= HOURS_PER_DAY:
        raise ValueError(
            "clear-sky geometry must satisfy 0 <= sunrise_h < sunset_h <= 24"
        )
    if end_s <= start_s:
        raise ValueError("clear-sky integration interval must be positive")

    daylight_h = sunset - sunrise
    daily_equivalent_h = 2.0 * daylight_h / math.pi

    def primitive(absolute_h: float) -> float:
        day = math.floor(absolute_h / HOURS_PER_DAY)
        hour = absolute_h - day * HOURS_PER_DAY
        if hour <= sunrise:
            partial_h = 0.0
        elif hour >= sunset:
            partial_h = daily_equivalent_h
        else:
            phase = math.pi * (hour - sunrise) / daylight_h
            partial_h = daylight_h / math.pi * (1.0 - math.cos(phase))
        return day * daily_equivalent_h + partial_h

    absolute_start_h = (boundary_s + start_s) / SECONDS_PER_HOUR
    absolute_end_h = (boundary_s + end_s) / SECONDS_PER_HOUR
    duration_h = (end_s - start_s) / SECONDS_PER_HOUR
    equivalent_h = primitive(absolute_end_h) - primitive(absolute_start_h)
    return min(duration_h, max(0.0, equivalent_h))


@dataclass(frozen=True, slots=True)
class MPCForecastBandV1:
    """Audit record for one reconstructed public forecast band."""

    name: str
    start_offset_s: float
    end_offset_s: float
    first_interval: int
    interval_count: int
    issued_mean_peak_fraction: float
    clear_sky_mean_peak_fraction: float
    resource_scale: float
    reconstructed_mean_peak_fraction: float
    absolute_mean_error: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("forecast band name must be non-empty")
        start = _finite_real("start_offset_s", self.start_offset_s, minimum=0.0)
        end = _finite_real("end_offset_s", self.end_offset_s, minimum=0.0)
        if end <= start:
            raise ValueError("forecast band offsets must be ordered")
        for name in ("first_interval", "interval_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        if self.first_interval < 0 or self.interval_count <= 0:
            raise ValueError("forecast band interval bounds are invalid")
        for name in (
            "issued_mean_peak_fraction",
            "clear_sky_mean_peak_fraction",
            "resource_scale",
            "reconstructed_mean_peak_fraction",
            "absolute_mean_error",
        ):
            number = _finite_real(name, getattr(self, name), minimum=0.0)
            if name != "absolute_mean_error" and number > 1.0 + 1e-12:
                raise ValueError(f"{name} must not exceed one")

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "name": self.name,
            "start_offset_s": float(self.start_offset_s),
            "end_offset_s": float(self.end_offset_s),
            "first_interval": int(self.first_interval),
            "interval_count": int(self.interval_count),
            "issued_mean_peak_fraction": float(self.issued_mean_peak_fraction),
            "clear_sky_mean_peak_fraction": float(self.clear_sky_mean_peak_fraction),
            "resource_scale": float(self.resource_scale),
            "reconstructed_mean_peak_fraction": float(
                self.reconstructed_mean_peak_fraction
            ),
            "absolute_mean_error": float(self.absolute_mean_error),
        }


@dataclass(frozen=True, slots=True)
class MPCForecastTrajectoryV1:
    """Immutable 24-hour public-forecast projection at controller resolution."""

    projection_version: str
    source_issue_sha256: str
    issued_at_physics_boundary_step: int
    issued_at_offset_s: float
    boundary_time_s: float
    interval_s: float
    peak_kw: float
    clear_sky_mean_peak_fractions: tuple[float, ...]
    available_peak_fractions: tuple[float, ...]
    available_kw: tuple[float, ...]
    interval_band_names: tuple[str, ...]
    bands: tuple[MPCForecastBandV1, ...]

    def __post_init__(self) -> None:
        if self.projection_version not in (
            FORECAST_PROJECTION_VERSION,
            P10_FORECAST_PROJECTION_VERSION,
        ):
            raise ValueError("forecast projection version is unsupported")
        if (
            not isinstance(self.source_issue_sha256, str)
            or len(self.source_issue_sha256) != 64
        ):
            raise ValueError("forecast source issue must be a SHA-256 digest")
        try:
            int(self.source_issue_sha256, 16)
        except ValueError as exc:
            raise ValueError("forecast source issue must be a SHA-256 digest") from exc
        step = self.issued_at_physics_boundary_step
        if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
            raise ValueError("forecast issue boundary step must be non-negative")
        _finite_real("issued_at_offset_s", self.issued_at_offset_s, minimum=0.0)
        _finite_real("boundary_time_s", self.boundary_time_s, minimum=0.0)
        interval_s = _finite_real("interval_s", self.interval_s, minimum=0.0)
        peak_kw = _finite_real("peak_kw", self.peak_kw, minimum=0.0)
        if interval_s <= 0.0 or peak_kw <= 0.0:
            raise ValueError("forecast interval and peak power must be positive")

        sequence_fields = (
            "clear_sky_mean_peak_fractions",
            "available_peak_fractions",
            "available_kw",
            "interval_band_names",
            "bands",
        )
        if any(not isinstance(getattr(self, name), tuple) for name in sequence_fields):
            raise TypeError("forecast trajectory sequences must be immutable tuples")
        lengths = {
            len(self.clear_sky_mean_peak_fractions),
            len(self.available_peak_fractions),
            len(self.available_kw),
            len(self.interval_band_names),
        }
        if len(lengths) != 1 or not self.available_kw:
            raise ValueError("forecast interval arrays must be equal and non-empty")
        if not math.isclose(
            len(self.available_kw) * interval_s,
            HOURS_PER_DAY * SECONDS_PER_HOUR,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("forecast trajectory must cover exactly 24 hours")
        for index, (clear, available, power) in enumerate(
            zip(
                self.clear_sky_mean_peak_fractions,
                self.available_peak_fractions,
                self.available_kw,
                strict=True,
            )
        ):
            clear_value = _finite_real(
                f"clear_sky_mean_peak_fractions[{index}]", clear, minimum=0.0
            )
            available_value = _finite_real(
                f"available_peak_fractions[{index}]", available, minimum=0.0
            )
            power_value = _finite_real(f"available_kw[{index}]", power, minimum=0.0)
            if clear_value > 1.0 + 1e-12 or available_value > 1.0 + 1e-12:
                raise ValueError("forecast fractions must not exceed one")
            if available_value > clear_value + 1e-12:
                raise ValueError("available forecast cannot exceed clear-sky geometry")
            if not math.isclose(
                power_value,
                available_value * peak_kw,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("available power and peak fraction disagree")
        if tuple(band.name for band in self.bands) != tuple(
            f"{start:g}-{end:g}h" for start, end in FORECAST_WINDOWS_H
        ):
            raise ValueError("forecast bands must use the canonical order")
        cursor = 0
        for band in self.bands:
            if not isinstance(band, MPCForecastBandV1):
                raise TypeError("forecast bands must be MPCForecastBandV1 records")
            stop = cursor + band.interval_count
            if band.first_interval != cursor:
                raise ValueError("forecast band interval ranges must be contiguous")
            if not (
                math.isclose(
                    band.start_offset_s,
                    cursor * interval_s,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                and math.isclose(
                    band.end_offset_s,
                    stop * interval_s,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError("forecast band offsets disagree with interval ranges")
            clear_values = self.clear_sky_mean_peak_fractions[cursor:stop]
            available_values = self.available_peak_fractions[cursor:stop]
            observed_clear_mean = math.fsum(clear_values) / band.interval_count
            observed_available_mean = math.fsum(available_values) / band.interval_count
            if not math.isclose(
                observed_clear_mean,
                band.clear_sky_mean_peak_fraction,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    "forecast band clear-sky mean is not bound to intervals"
                )
            if not math.isclose(
                observed_available_mean,
                band.reconstructed_mean_peak_fraction,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    "forecast band reconstructed mean is not bound to intervals"
                )
            if not math.isclose(
                abs(
                    band.reconstructed_mean_peak_fraction
                    - band.issued_mean_peak_fraction
                ),
                band.absolute_mean_error,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError("forecast band reconstruction error is inconsistent")
            for clear_value, available_value in zip(
                clear_values, available_values, strict=True
            ):
                if not math.isclose(
                    available_value,
                    clear_value * band.resource_scale,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ):
                    raise ValueError("forecast band scale is not bound to intervals")
            cursor = stop
        if cursor != len(self.available_kw):
            raise ValueError("forecast bands do not cover the complete trajectory")
        expected_names = tuple(
            band.name for band in self.bands for _ in range(band.interval_count)
        )
        if self.interval_band_names != expected_names:
            raise ValueError("interval-to-band mapping is not canonical")

    @property
    def interval_count(self) -> int:
        return len(self.available_kw)

    def prefix(self, interval_count: int) -> tuple[float, ...]:
        """Return a causal prefix without renormalizing a partial forecast band."""

        if (
            isinstance(interval_count, bool)
            or not isinstance(interval_count, Integral)
            or not 1 <= interval_count <= self.interval_count
        ):
            raise ValueError("prefix interval_count is outside the forecast horizon")
        return self.available_kw[: int(interval_count)]

    def body(self) -> dict[str, Any]:
        return {
            "projection_version": self.projection_version,
            "source_issue_sha256": self.source_issue_sha256,
            "issued_at_physics_boundary_step": int(
                self.issued_at_physics_boundary_step
            ),
            "issued_at_offset_s": float(self.issued_at_offset_s),
            "boundary_time_s": float(self.boundary_time_s),
            "interval_s": float(self.interval_s),
            "peak_kw": float(self.peak_kw),
            "clear_sky_mean_peak_fractions": list(self.clear_sky_mean_peak_fractions),
            "available_peak_fractions": list(self.available_peak_fractions),
            "available_kw": list(self.available_kw),
            "interval_band_names": list(self.interval_band_names),
            "bands": [band.to_dict() for band in self.bands],
        }

    @property
    def sha256(self) -> str:
        return _domain_sha256(FORECAST_TRAJECTORY_DOMAIN, self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


def _validate_window_grid(
    forecast: PublicForecastIssueV2,
    *,
    interval_s: float,
) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    for expected, window in zip(FORECAST_WINDOWS_H, forecast.windows, strict=True):
        expected_start_s = expected[0] * SECONDS_PER_HOUR
        expected_end_s = expected[1] * SECONDS_PER_HOUR
        if not (
            math.isclose(
                window.start_offset_s_from_issue,
                expected_start_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                window.end_offset_s_from_issue,
                expected_end_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("public forecast window offsets are not canonical")
        first = int(round(expected_start_s / interval_s))
        stop = _positive_aligned_steps(
            f"forecast window {window.name} end",
            expected_end_s,
            interval_s,
        )
        if not math.isclose(
            first * interval_s,
            expected_start_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"forecast window {window.name} start must align to interval_s"
            )
        ranges.append((first, stop))
    return tuple(ranges)


def project_issued_mean_solar(
    forecast: PublicForecastIssueV2,
    *,
    boundary_time_s: float,
    config: PlantConfig,
    interval_s: float,
    energy_tolerance: float = DEFAULT_ENERGY_TOLERANCE,
) -> MPCForecastTrajectoryV1:
    """Reconstruct a coherent deterministic path from one public issuance."""

    if not isinstance(forecast, PublicForecastIssueV2):
        raise TypeError("forecast must be a PublicForecastIssueV2")
    if not isinstance(config, PlantConfig):
        raise TypeError("config must be a PlantConfig")
    boundary_s = _finite_real("boundary_time_s", boundary_time_s, minimum=0.0)
    step_s = _finite_real("interval_s", interval_s, minimum=0.0)
    tolerance = _finite_real("energy_tolerance", energy_tolerance, minimum=0.0)
    if step_s <= 0.0:
        raise ValueError("interval_s must be positive")
    if tolerance <= 0.0:
        raise ValueError("energy_tolerance must be positive")
    peak_kw = _finite_real("config.solar.peak_kw", config.solar.peak_kw, minimum=0.0)
    if peak_kw <= 0.0:
        raise ValueError("config.solar.peak_kw must be positive")

    total_intervals = _positive_aligned_steps(
        "public forecast horizon",
        HOURS_PER_DAY * SECONDS_PER_HOUR,
        step_s,
    )
    ranges = _validate_window_grid(forecast, interval_s=step_s)
    clear_sky: list[float] = []
    for index in range(total_intervals):
        start_s = index * step_s
        end_s = (index + 1) * step_s
        equivalent_h = clear_sky_equivalent_hours(
            boundary_time_s=boundary_s,
            start_offset_s=start_s,
            end_offset_s=end_s,
            sunrise_h=config.solar.sunrise_h,
            sunset_h=config.solar.sunset_h,
        )
        clear_sky.append(equivalent_h / (step_s / SECONDS_PER_HOUR))

    available: list[float] = [0.0] * total_intervals
    band_names: list[str] = [""] * total_intervals
    audits: list[MPCForecastBandV1] = []
    for window, (first, stop) in zip(forecast.windows, ranges, strict=True):
        clear_values = clear_sky[first:stop]
        if not clear_values:
            raise ValueError("forecast band contains no optimization intervals")
        clear_mean = math.fsum(clear_values) / len(clear_values)
        issued_mean = _finite_real(
            f"forecast window {window.name} issued mean",
            window.mean_peak_power_fraction,
            minimum=0.0,
        )
        if issued_mean > 1.0:
            raise ValueError("issued forecast mean must not exceed one")
        if clear_mean <= tolerance:
            if issued_mean > tolerance:
                raise ValueError(
                    f"forecast window {window.name} has power in public darkness"
                )
            scale = 0.0
        else:
            scale = issued_mean / clear_mean
            if scale > 1.0 + tolerance:
                raise ValueError(
                    f"forecast window {window.name} exceeds clear-sky energy"
                )
            scale = min(1.0, max(0.0, scale))
        reconstructed_values = [value * scale for value in clear_values]
        reconstructed_mean = math.fsum(reconstructed_values) / len(reconstructed_values)
        error = abs(reconstructed_mean - issued_mean)
        if error > tolerance:
            raise RuntimeError(
                f"forecast window {window.name} energy reconstruction failed"
            )
        available[first:stop] = reconstructed_values
        band_names[first:stop] = [window.name] * (stop - first)
        audits.append(
            MPCForecastBandV1(
                name=window.name,
                start_offset_s=window.start_offset_s_from_issue,
                end_offset_s=window.end_offset_s_from_issue,
                first_interval=first,
                interval_count=stop - first,
                issued_mean_peak_fraction=issued_mean,
                clear_sky_mean_peak_fraction=clear_mean,
                resource_scale=scale,
                reconstructed_mean_peak_fraction=reconstructed_mean,
                absolute_mean_error=error,
            )
        )

    if any(not name for name in band_names):
        raise RuntimeError("forecast windows do not cover the complete horizon")
    power = tuple(value * peak_kw for value in available)
    return MPCForecastTrajectoryV1(
        projection_version=FORECAST_PROJECTION_VERSION,
        source_issue_sha256=public_forecast_issue_sha256(forecast),
        issued_at_physics_boundary_step=int(forecast.issued_at_physics_boundary_step),
        issued_at_offset_s=float(forecast.issued_at_offset_s),
        boundary_time_s=boundary_s,
        interval_s=step_s,
        peak_kw=peak_kw,
        clear_sky_mean_peak_fractions=tuple(clear_sky),
        available_peak_fractions=tuple(available),
        available_kw=power,
        interval_band_names=tuple(band_names),
        bands=tuple(audits),
    )


def project_issued_p10_solar(
    forecast: PublicForecastIssueV2,
    *,
    boundary_time_s: float,
    config: PlantConfig,
    interval_s: float,
    energy_tolerance: float = DEFAULT_ENERGY_TOLERANCE,
) -> MPCForecastTrajectoryV1:
    """Reconstruct band-p10 energy; no interval/joint coverage is implied.

    Reuse the exact energy integration, but retain the original full issue digest
    and a distinct statistic version. The temporary means are band averages of
    the chosen lower-energy scenario, never a replacement public issuance.
    """
    if not isinstance(forecast, PublicForecastIssueV2):
        raise TypeError("forecast must be a PublicForecastIssueV2")
    lower_issue = replace(
        forecast,
        windows=tuple(
            replace(window, mean_peak_power_fraction=window.p10_peak_power_fraction)
            for window in forecast.windows
        ),
    )
    lower = project_issued_mean_solar(
        lower_issue,
        boundary_time_s=boundary_time_s,
        config=config,
        interval_s=interval_s,
        energy_tolerance=energy_tolerance,
    )
    return replace(
        lower,
        projection_version=P10_FORECAST_PROJECTION_VERSION,
        source_issue_sha256=public_forecast_issue_sha256(forecast),
    )


def trajectory_sha256_from_public_inputs(
    forecast: PublicForecastIssueV2,
    *,
    boundary_time_s: float,
    config: PlantConfig,
    interval_s: float,
) -> str:
    """Small convenience used by information-isolation tests and evidence."""

    return project_issued_mean_solar(
        forecast,
        boundary_time_s=boundary_time_s,
        config=config,
        interval_s=interval_s,
    ).sha256


def mean_peak_fraction(values_kw: Sequence[float], *, peak_kw: float) -> float:
    """Strict helper for independent reconstruction checks."""

    peak = _finite_real("peak_kw", peak_kw, minimum=0.0)
    if peak <= 0.0:
        raise ValueError("peak_kw must be positive")
    values = tuple(values_kw)
    if not values:
        raise ValueError("values_kw must not be empty")
    checked = tuple(
        _finite_real(f"values_kw[{index}]", value, minimum=0.0)
        for index, value in enumerate(values)
    )
    return math.fsum(checked) / len(checked) / peak


__all__ = [
    "DEFAULT_ENERGY_TOLERANCE",
    "FORECAST_PROJECTION_VERSION",
    "FORECAST_TRAJECTORY_DOMAIN",
    "PUBLIC_FORECAST_ISSUE_DOMAIN",
    "MPCForecastBandV1",
    "MPCForecastTrajectoryV1",
    "clear_sky_equivalent_hours",
    "mean_peak_fraction",
    "project_issued_mean_solar",
    "public_forecast_issue_sha256",
    "trajectory_sha256_from_public_inputs",
]

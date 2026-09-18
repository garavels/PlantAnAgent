"""Operating-only issued-mean projection on the physical left-sample envelope.

Weather issuance averages ``SolarArray.clear_sky_kw`` at the start of each
physical interval. This separately identified projection uses that same public
geometry, never the private weather tape. The historical service/beta mean and
p10 analytic projections remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from plant_sim import PlantConfig
from plant_sim.components.solar import SolarArray
from plant_sim.config import SolarConfig

from .economic_mpc_model import EconomicMPCModelV1
from .mpc_forecast import (
    DEFAULT_ENERGY_TOLERANCE,
    HOURS_PER_DAY,
    SECONDS_PER_HOUR,
    MPCForecastBandV1,
    MPCForecastTrajectoryV1,
    _domain_sha256,
    _finite_real,
    _positive_aligned_steps,
    _validate_window_grid,
    public_forecast_issue_sha256,
)
from .mpc_model import MPCSparseRowV1
from .weather import PublicForecastIssueV2

OPERATING_FORECAST_PROJECTION_VERSION = "issued-mean-physics-left-samples-v1"
OPERATING_FORECAST_TRAJECTORY_DOMAIN = "operating-mpc-sampled-forecast-trajectory-v1"
SAMPLE_RULE = "SolarArray.clear_sky_kw at each physical interval start"


def _sampled_envelope(
    *,
    boundary_time_s: float,
    interval_s: float,
    physics_dt_s: float,
    peak_kw: float,
    sunrise_h: float,
    sunset_h: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Public means and minima, using the producer's exact point samples."""
    boundary = _finite_real("boundary_time_s", boundary_time_s, minimum=0.0)
    step = _finite_real("interval_s", interval_s, minimum=0.0)
    dt = _finite_real("physics_dt_s", physics_dt_s, minimum=0.0)
    peak = _finite_real("peak_kw", peak_kw, minimum=0.0)
    sunrise = _finite_real("sunrise_h", sunrise_h, minimum=0.0)
    sunset = _finite_real("sunset_h", sunset_h, minimum=0.0)
    if min(step, dt, peak) <= 0.0:
        raise ValueError(
            "forecast interval, physics timestep and peak must be positive"
        )
    if not 0.0 <= sunrise < sunset <= HOURS_PER_DAY:
        raise ValueError(
            "clear-sky geometry must satisfy 0 <= sunrise_h < sunset_h <= 24"
        )
    stride = _positive_aligned_steps("interval_s", step, dt)
    count = _positive_aligned_steps(
        "public forecast horizon", HOURS_PER_DAY * SECONDS_PER_HOUR, step
    )
    solar = SolarArray(SolarConfig(peak_kw=peak, sunrise_h=sunrise, sunset_h=sunset))
    samples = tuple(
        solar.clear_sky_kw(boundary + index * dt) / peak
        for index in range(count * stride)
    )
    groups = tuple(
        samples[start : start + stride] for start in range(0, len(samples), stride)
    )
    return (
        tuple(math.fsum(group) / stride for group in groups),
        tuple(min(group) for group in groups),
    )


@dataclass(frozen=True, slots=True)
class OperatingMPCForecastTrajectoryV1(MPCForecastTrajectoryV1):
    """Legacy model-compatible trajectory with independently bound sampling."""

    physics_dt_s: float
    sunrise_h: float
    sunset_h: float
    clear_sky_min_peak_fractions: tuple[float, ...]

    def _validate_projection_version(self) -> None:
        if self.projection_version != OPERATING_FORECAST_PROJECTION_VERSION:
            raise ValueError("operating forecast projection version is unsupported")

    def __post_init__(self) -> None:
        MPCForecastTrajectoryV1.__post_init__(self)
        if not isinstance(self.clear_sky_min_peak_fractions, tuple):
            raise TypeError("sampled minima must be an immutable tuple")
        for index, value in enumerate(self.clear_sky_min_peak_fractions):
            _finite_real(f"clear_sky_min_peak_fractions[{index}]", value, minimum=0.0)
        means, minima = _sampled_envelope(
            boundary_time_s=self.boundary_time_s,
            interval_s=self.interval_s,
            physics_dt_s=self.physics_dt_s,
            peak_kw=self.peak_kw,
            sunrise_h=self.sunrise_h,
            sunset_h=self.sunset_h,
        )
        if means != self.clear_sky_mean_peak_fractions:
            raise ValueError("clear-sky means differ from the bound physics samples")
        if minima != self.clear_sky_min_peak_fractions:
            raise ValueError("clear-sky minima differ from the bound physics samples")

    def body(self) -> dict[str, Any]:
        return {
            **MPCForecastTrajectoryV1.body(self),
            "sampling": {
                "rule": SAMPLE_RULE,
                "physics_dt_s": float(self.physics_dt_s),
                "sunrise_h": float(self.sunrise_h),
                "sunset_h": float(self.sunset_h),
            },
            "clear_sky_min_peak_fractions": list(self.clear_sky_min_peak_fractions),
        }

    @property
    def sha256(self) -> str:
        return _domain_sha256(OPERATING_FORECAST_TRAJECTORY_DOMAIN, self.body())


def project_operating_issued_mean_solar(
    forecast: PublicForecastIssueV2,
    *,
    boundary_time_s: float,
    config: PlantConfig,
    interval_s: float,
    energy_tolerance: float = DEFAULT_ENERGY_TOLERANCE,
) -> OperatingMPCForecastTrajectoryV1:
    """Preserve complete public band energies and darkness on the physical grid."""
    if not isinstance(forecast, PublicForecastIssueV2):
        raise TypeError("forecast must be a PublicForecastIssueV2")
    if not isinstance(config, PlantConfig):
        raise TypeError("config must be a PlantConfig")
    tolerance = _finite_real("energy_tolerance", energy_tolerance, minimum=0.0)
    if tolerance <= 0.0:
        raise ValueError("energy_tolerance must be positive")
    clear_sky, minima = _sampled_envelope(
        boundary_time_s=boundary_time_s,
        interval_s=interval_s,
        physics_dt_s=config.dt_s,
        peak_kw=config.solar.peak_kw,
        sunrise_h=config.solar.sunrise_h,
        sunset_h=config.solar.sunset_h,
    )
    ranges = _validate_window_grid(forecast, interval_s=interval_s)
    available = [0.0] * len(clear_sky)
    names = [""] * len(clear_sky)
    audits = []
    for window, (first, stop) in zip(forecast.windows, ranges, strict=True):
        values = clear_sky[first:stop]
        if not values:
            raise ValueError("forecast band contains no optimization intervals")
        clear_mean = math.fsum(values) / len(values)
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
        reconstructed = [value * scale for value in values]
        reconstructed_mean = math.fsum(reconstructed) / len(reconstructed)
        error = abs(reconstructed_mean - issued_mean)
        if error > tolerance:
            raise RuntimeError(
                f"forecast window {window.name} energy reconstruction failed"
            )
        available[first:stop] = reconstructed
        names[first:stop] = [window.name] * (stop - first)
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
    return OperatingMPCForecastTrajectoryV1(
        projection_version=OPERATING_FORECAST_PROJECTION_VERSION,
        source_issue_sha256=public_forecast_issue_sha256(forecast),
        issued_at_physics_boundary_step=int(forecast.issued_at_physics_boundary_step),
        issued_at_offset_s=float(forecast.issued_at_offset_s),
        boundary_time_s=float(boundary_time_s),
        interval_s=float(interval_s),
        peak_kw=float(config.solar.peak_kw),
        clear_sky_mean_peak_fractions=clear_sky,
        available_peak_fractions=tuple(available),
        available_kw=tuple(value * config.solar.peak_kw for value in available),
        interval_band_names=tuple(names),
        bands=tuple(audits),
        physics_dt_s=float(config.dt_s),
        sunrise_h=float(config.solar.sunrise_h),
        sunset_h=float(config.solar.sunset_h),
        clear_sky_min_peak_fractions=minima,
    )


def operating_constant_power_rows(
    model: EconomicMPCModelV1,
    weather: OperatingMPCForecastTrajectoryV1,
    config: PlantConfig,
) -> tuple[tuple[MPCSparseRowV1, ...], tuple[float, ...]]:
    """Bound held solar targets by the same sampled envelope as band energy."""
    if not isinstance(weather, OperatingMPCForecastTrajectoryV1):
        raise TypeError(
            "operating firm power requires the sampled operating trajectory"
        )
    if not isinstance(config, PlantConfig):
        raise TypeError("config must be a PlantConfig")
    if (
        weather.physics_dt_s != config.dt_s
        or weather.peak_kw != config.solar.peak_kw
        or weather.sunrise_h != config.solar.sunrise_h
        or weather.sunset_h != config.solar.sunset_h
    ):
        raise ValueError("firm-power config differs from the bound sampled geometry")
    if not 1 <= model.horizon_intervals <= weather.interval_count:
        raise ValueError("firm-power horizon exceeds the sampled trajectory")
    groups = tuple(
        model.group(name)
        for name in (
            "solar_to_battery_kw",
            "solar_to_electrolyser_kw",
            "solar_to_dac_kw",
            "solar_to_methanator_kw",
        )
    )
    rows = []
    powers = []
    for index in range(model.horizon_intervals):
        mean = weather.clear_sky_mean_peak_fractions[index]
        firm = (
            0.0
            if mean <= 1e-12
            else min(
                weather.available_kw[index],
                weather.available_kw[index]
                * weather.clear_sky_min_peak_fractions[index]
                / mean,
            )
        )
        powers.append(firm)
        rows.append(
            MPCSparseRowV1(
                name=f"beta_constant_solar_power[{index}]",
                lower=None,
                upper=firm,
                coefficients=tuple(sorted((group[index], 1.0) for group in groups)),
            )
        )
    return tuple(rows), tuple(powers)

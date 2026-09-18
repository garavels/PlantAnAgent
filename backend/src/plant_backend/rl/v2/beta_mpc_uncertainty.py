"""Public-only conservative solar and spendable early resource buffers."""

from __future__ import annotations

import math
from typing import Any

from plant_sim import PlantConfig

from .beta_mpc_schema import (
    BETA_MPC_UNCERTAINTY_POLICY,
    BetaMPCCycleReturnV1,
    BetaMPCSettingsV1,
)
from .economic_mpc_model import EconomicMPCModelV1
from .mpc_forecast import (
    MPCForecastTrajectoryV1,
    project_issued_mean_solar,
    project_issued_p10_solar,
)
from .mpc_model import MPCSparseRowV1
from .weather import PublicForecastIssueV2


def constant_power_rows(
    model: EconomicMPCModelV1,
    weather: MPCForecastTrajectoryV1,
    config: PlantConfig,
) -> tuple[tuple[MPCSparseRowV1, ...], tuple[float, ...]]:
    """Bound held targets by sampled daylight power, not average interval energy."""
    stride = int(round(weather.interval_s / config.dt_s))
    if stride < 1 or not math.isclose(
        stride * config.dt_s, weather.interval_s, abs_tol=1e-9
    ):
        raise ValueError("constant-power projection requires aligned physics samples")
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
    available = []
    for index in range(model.horizon_intervals):
        fractions = []
        for minute in range(stride):
            hour = (
                (
                    weather.boundary_time_s
                    + index * weather.interval_s
                    + minute * config.dt_s
                )
                / 3600.0
            ) % 24.0
            phase = (hour - config.solar.sunrise_h) / (
                config.solar.sunset_h - config.solar.sunrise_h
            )
            fractions.append(math.sin(math.pi * phase) if 0.0 < phase < 1.0 else 0.0)
        average = weather.clear_sky_mean_peak_fractions[index]
        firm = (
            0.0
            if average <= 1e-12
            else min(
                weather.available_kw[index],
                weather.available_kw[index] * min(fractions) / average,
            )
        )
        available.append(firm)
        rows.append(
            MPCSparseRowV1(
                name=f"beta_constant_solar_power[{index}]",
                lower=None,
                upper=firm,
                coefficients=tuple(sorted((group[index], 1.0) for group in groups)),
            )
        )
    return tuple(rows), tuple(available)


def project_beta_mpc_solar(
    forecast: PublicForecastIssueV2,
    *,
    boundary_time_s: float,
    config: PlantConfig,
    settings: BetaMPCSettingsV1,
) -> MPCForecastTrajectoryV1:
    project = (
        project_issued_p10_solar
        if settings.uncertainty_policy == BETA_MPC_UNCERTAINTY_POLICY
        else project_issued_mean_solar
    )
    return project(
        forecast,
        boundary_time_s=boundary_time_s,
        config=config,
        interval_s=settings.interval_s,
        energy_tolerance=settings.energy_reconstruction_tolerance,
    )


def early_reserve_rows(
    model: EconomicMPCModelV1,
    cycle: BetaMPCCycleReturnV1,
    weather: MPCForecastTrajectoryV1,
    config: PlantConfig,
) -> tuple[tuple[MPCSparseRowV1, ...], dict[str, Any]]:
    """Keep original task minima separate from additional planning reserves.

    After the advance restoration deadline, a buffer cannot exceed the currently
    observed surplus. Forecast error can spend that buffer without creating a
    demand to recharge it in darkness. This has no persistent ratcheting state.
    """
    horizon = cycle.remaining_intervals
    daylight = [
        index + 1
        for index, value in enumerate(weather.clear_sky_mean_peak_fractions[:horizon])
        if value > 1e-12
    ]
    last_daylight_boundary = max(daylight, default=0)
    lead = int(math.ceil(2.0 * 3600.0 / weather.interval_s))
    deadline = max(0, last_daylight_boundary - lead)
    specifications = (
        ("battery", "battery_energy_kwh", config.battery.capacity_kwh),
        ("h2", "h2_inventory_kg", config.h2_tank.capacity_kg),
        ("co2", "co2_inventory_kg", config.co2_tank.capacity_kg),
        ("ch4", "ch4_inventory_kg", config.ch4_tank.capacity_kg),
    )
    rows: list[MPCSparseRowV1] = []
    resources = []
    for name, group_name, capacity in specifications:
        requirement = cycle.resource(name)
        requested = min(
            0.01 * capacity, max(0.0, capacity - requirement.required_minimum)
        )
        retained = (
            requested
            if deadline > 0
            else min(
                requested, max(0.0, requirement.current - requirement.required_minimum)
            )
        )
        floor = requirement.required_minimum + retained
        for boundary in range(max(1, deadline), horizon + 1):
            rows.append(
                MPCSparseRowV1(
                    name=f"beta_early_reserve_{name}[{boundary}]",
                    lower=floor,
                    upper=None,
                    coefficients=((model.group(group_name)[boundary], 1.0),),
                )
            )
        resources.append(
            {
                "resource": name,
                "capacity": capacity,
                "required_cycle_minimum": requirement.required_minimum,
                "requested_buffer": requested,
                "retained_buffer": retained,
                "planning_floor": floor,
            }
        )
    return tuple(rows), {
        "policy": BETA_MPC_UNCERTAINTY_POLICY,
        "restoration_lead_h": 2.0,
        "buffer_capacity_fraction": 0.01,
        "last_daylight_boundary": last_daylight_boundary,
        "restoration_boundary": deadline,
        "after_restoration_deadline": deadline == 0,
        "resources": resources,
        "guarantee": "conservative planning policy, not a joint weather guarantee",
    }

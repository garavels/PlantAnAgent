"""Hand-solvable physical and commitment fixtures for the service-MPC MILP."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
from plant_sim import ElectrolyserMode, PlantConfig

from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.control import ControllerObservationV2, DeliveryObservation
from plant_backend.rl.v2.env import PlantEnvV2
from plant_backend.rl.v2.mpc_forecast import (
    MPCForecastTrajectoryV1,
    clear_sky_equivalent_hours,
    project_issued_mean_solar,
)
from plant_backend.rl.v2.mpc_model import (
    ServiceMPCModelV1,
    build_service_mpc_model,
)
from plant_backend.rl.v2.mpc_schema import (
    DEFAULT_SERVICE_MPC_SETTINGS_V1,
    MPCPlanV1,
)
from plant_backend.rl.v2.mpc_solver import solve_service_mpc_model


SETTINGS = DEFAULT_SERVICE_MPC_SETTINGS_V1
INTERVAL_S = SETTINGS.interval_s


def _scaled_public_forecast(
    observation: ControllerObservationV2,
    config: PlantConfig,
    *,
    resource_scale: float,
) -> tuple[ControllerObservationV2, MPCForecastTrajectoryV1]:
    issue = observation.forecast
    assert issue is not None
    windows = []
    for window in issue.windows:
        equivalent_h = clear_sky_equivalent_hours(
            boundary_time_s=observation.state.boundary.time_s,
            start_offset_s=window.start_offset_s_from_issue,
            end_offset_s=window.end_offset_s_from_issue,
            sunrise_h=config.solar.sunrise_h,
            sunset_h=config.solar.sunset_h,
        )
        duration_h = (
            window.end_offset_s_from_issue - window.start_offset_s_from_issue
        ) / 3_600.0
        mean_fraction = equivalent_h / duration_h * resource_scale
        windows.append(
            replace(
                window,
                mean_peak_power_fraction=mean_fraction,
                p10_peak_power_fraction=mean_fraction,
            )
        )
    scaled_issue = replace(issue, windows=tuple(windows))
    public = replace(observation, forecast=scaled_issue)
    trajectory = project_issued_mean_solar(
        scaled_issue,
        boundary_time_s=public.state.boundary.time_s,
        config=config,
        interval_s=INTERVAL_S,
    )
    return public, trajectory


def _case_observation(
    config: PlantConfig,
    *,
    horizon_intervals: int,
    requested_kg: float,
    resource_scale: float,
    battery_soc: float,
    h2_kg: float,
    co2_kg: float,
    ch4_kg: float,
    mode: ElectrolyserMode = ElectrolyserMode.OFF,
    min_up_remaining_s: float = 0.0,
    min_down_remaining_s: float = 0.0,
) -> tuple[PlantEnvV2, ControllerObservationV2, MPCForecastTrajectoryV1]:
    episode = EpisodeConfigV2(
        horizon_h=72.0,
        start_hour=12.0,
        delivery_period_h=0.25,
        delivery_rate_kg_h=1.0,
        ch4_coverage_h=(0.0, 0.0),
    )
    env = PlantEnvV2(config=config, episode=episode)
    env.reset(seed=0)
    observation, forecast = _scaled_public_forecast(
        env.controller_observation,
        config,
        resource_scale=resource_scale,
    )
    boundary = observation.state.boundary
    snapshot = observation.state.electrolyser_commitment
    assert snapshot is not None
    commitment_profile = episode.electrolyser_commitment
    assert commitment_profile is not None
    capacity = config.battery.capacity_kwh
    stable_power = (
        commitment_profile.min_stable_load_frac * config.electrolyser.rated_kw
    )
    updated_boundary = replace(
        boundary,
        battery_soc=battery_soc,
        battery_energy_kwh=battery_soc * capacity,
        battery_capacity_kwh=capacity,
        h2_level_kg=h2_kg,
        h2_capacity_kg=config.h2_tank.capacity_kg,
        h2_fill=h2_kg / config.h2_tank.capacity_kg,
        co2_level_kg=co2_kg,
        co2_capacity_kg=config.co2_tank.capacity_kg,
        co2_fill=co2_kg / config.co2_tank.capacity_kg,
        ch4_level_kg=ch4_kg,
        ch4_capacity_kg=config.ch4_tank.capacity_kg,
        ch4_fill=ch4_kg / config.ch4_tank.capacity_kg,
    )
    updated_snapshot = replace(
        snapshot,
        mode=mode,
        mode_elapsed_s=0.0,
        startup_remaining_s=0.0,
        cooldown_remaining_s=0.0,
        min_up_remaining_s=min_up_remaining_s,
        min_down_remaining_s=min_down_remaining_s,
        shortfall_elapsed_s=0.0,
        previous_power_kw=(stable_power if mode is ElectrolyserMode.RUNNING else 0.0),
    )
    public = replace(
        observation,
        state=replace(
            observation.state,
            boundary=updated_boundary,
            electrolyser_commitment=updated_snapshot,
        ),
        delivery=DeliveryObservation(
            steps_to_next=(
                None
                if requested_kg == 0.0
                else horizon_intervals * round(INTERVAL_S / config.dt_s)
            ),
            delivery_period_steps=round(24.0 * 3_600.0 / config.dt_s),
            next_requested_kg=requested_kg,
            cumulative_requested_kg=0.0,
            cumulative_delivered_kg=0.0,
        ),
    )
    return env, public, forecast


def _solve_case(
    config: PlantConfig,
    *,
    horizon_intervals: int,
    requested_kg: float,
    resource_scale: float,
    battery_soc: float,
    h2_kg: float,
    co2_kg: float,
    ch4_kg: float,
    mode: ElectrolyserMode = ElectrolyserMode.OFF,
    min_up_remaining_s: float = 0.0,
    min_down_remaining_s: float = 0.0,
) -> tuple[ServiceMPCModelV1, MPCPlanV1]:
    pytest.importorskip("scipy")
    env, observation, forecast = _case_observation(
        config,
        horizon_intervals=horizon_intervals,
        requested_kg=requested_kg,
        resource_scale=resource_scale,
        battery_soc=battery_soc,
        h2_kg=h2_kg,
        co2_kg=co2_kg,
        ch4_kg=ch4_kg,
        mode=mode,
        min_up_remaining_s=min_up_remaining_s,
        min_down_remaining_s=min_down_remaining_s,
    )
    model = build_service_mpc_model(
        observation,
        forecast,
        settings=SETTINGS,
        constraints=env.constraints,
        config=config,
        commitment_profile=env.episode_config.electrolyser_commitment,
        horizon_intervals=horizon_intervals,
    )
    return model, solve_service_mpc_model(model, settings=SETTINGS)


def test_zero_resource_and_no_usable_battery_has_exact_zero_service() -> None:
    config = PlantConfig()
    reserve_soc = 0.15
    _, plan = _solve_case(
        config,
        horizon_intervals=1,
        requested_kg=10.0,
        resource_scale=0.0,
        battery_soc=reserve_soc,
        h2_kg=0.0,
        co2_kg=0.0,
        ch4_kg=0.0,
    )

    step = plan.steps[0]
    assert plan.primary_delivered_optimum_kg == 0.0
    assert plan.selected_delivered_kg == 0.0
    assert plan.minimum_commitment_fill == 0.0
    assert step.forecast_solar_kw == 0.0
    assert step.delivered_kg == 0.0
    assert step.h2_produced_kg == 0.0
    assert step.co2_captured_kg == 0.0
    assert step.ch4_produced_kg == 0.0
    assert step.battery_support_kw == 0.0
    assert step.battery_energy_end_kwh == pytest.approx(
        reserve_soc * config.battery.capacity_kwh
    )


def test_abundant_solar_converts_exact_stoichiometric_endowment_before_delivery() -> (
    None
):
    base = PlantConfig()
    config = replace(
        base,
        h2_tank=replace(base.h2_tank, capacity_kg=5.0),
        co2_tank=replace(base.co2_tank, capacity_kg=27.5),
        ch4_tank=replace(base.ch4_tank, capacity_kg=10.0),
    )
    _, plan = _solve_case(
        config,
        horizon_intervals=1,
        requested_kg=10.0,
        resource_scale=1.0,
        battery_soc=0.25,
        h2_kg=5.0,
        co2_kg=27.5,
        ch4_kg=0.0,
    )

    step = plan.steps[0]
    assert plan.primary_delivered_optimum_kg == pytest.approx(10.0)
    assert plan.selected_delivered_kg == pytest.approx(10.0)
    assert plan.minimum_commitment_fill == pytest.approx(1.0)
    assert step.h2_produced_kg == pytest.approx(0.0, abs=1e-8)
    assert step.co2_captured_kg == pytest.approx(0.0, abs=1e-8)
    assert step.ch4_produced_kg == pytest.approx(10.0)
    assert step.delivered_kg == pytest.approx(10.0)
    assert step.solar_to_methanator_kw == pytest.approx(
        10.0 * config.methanator.specific_energy_kwh_per_kg / 0.25
    )
    assert step.h2_end_kg == pytest.approx(0.0, abs=1e-8)
    assert step.co2_end_kg == pytest.approx(0.0, abs=1e-8)
    assert step.ch4_end_kg == pytest.approx(0.0, abs=1e-8)


def test_full_methane_tank_blocks_same_interval_production_before_settlement() -> None:
    base = PlantConfig()
    config = replace(
        base,
        h2_tank=replace(base.h2_tank, capacity_kg=5.0),
        co2_tank=replace(base.co2_tank, capacity_kg=27.5),
        ch4_tank=replace(base.ch4_tank, capacity_kg=5.0),
    )
    _, plan = _solve_case(
        config,
        horizon_intervals=1,
        requested_kg=10.0,
        resource_scale=1.0,
        battery_soc=0.25,
        h2_kg=5.0,
        co2_kg=27.5,
        ch4_kg=5.0,
    )

    step = plan.steps[0]
    assert plan.primary_delivered_optimum_kg == pytest.approx(5.0)
    assert step.delivered_kg == pytest.approx(5.0)
    assert step.ch4_produced_kg == pytest.approx(0.0, abs=1e-8)
    assert step.ch4_end_kg == pytest.approx(0.0, abs=1e-8)


def test_battery_endowment_improves_service_but_cannot_cross_terminal_floor() -> None:
    base = PlantConfig()
    config = replace(
        base,
        solar=replace(base.solar, peak_kw=500.0),
        battery=replace(
            base.battery,
            capacity_kwh=1_000.0,
            initial_soc=0.50,
        ),
    )
    _, low_plan = _solve_case(
        config,
        horizon_intervals=1,
        requested_kg=100.0,
        resource_scale=1.0,
        battery_soc=0.25,
        h2_kg=0.0,
        co2_kg=1_000.0,
        ch4_kg=0.0,
        mode=ElectrolyserMode.RUNNING,
    )
    _, high_plan = _solve_case(
        config,
        horizon_intervals=1,
        requested_kg=100.0,
        resource_scale=1.0,
        battery_soc=0.50,
        h2_kg=0.0,
        co2_kg=1_000.0,
        ch4_kg=0.0,
        mode=ElectrolyserMode.RUNNING,
    )

    low = low_plan.steps[0]
    high = high_plan.steps[0]
    assert low.battery_support_kw == pytest.approx(0.0, abs=1e-8)
    assert high.battery_support_kw > 0.0
    assert high_plan.selected_delivered_kg > low_plan.selected_delivered_kg
    assert high.battery_energy_end_kwh == pytest.approx(250.0, abs=1e-3)
    assert high.battery_charging is False
    assert high.solar_to_battery_kw == pytest.approx(0.0, abs=1e-8)


def test_battery_only_endowment_produces_hydrogen_and_closes_energy_balance() -> None:
    base = PlantConfig()
    config = replace(
        base,
        h2_tank=replace(base.h2_tank, capacity_kg=6.0),
        co2_tank=replace(base.co2_tank, capacity_kg=27.5),
        ch4_tank=replace(base.ch4_tank, capacity_kg=10.0),
    )
    model, plan = _solve_case(
        config,
        horizon_intervals=1,
        requested_kg=0.0,
        resource_scale=0.0,
        battery_soc=0.50,
        h2_kg=0.0,
        co2_kg=27.5,
        ch4_kg=0.0,
        mode=ElectrolyserMode.RUNNING,
        min_up_remaining_s=INTERVAL_S,
    )

    step = plan.steps[0]
    assert step.forecast_solar_kw == 0.0
    assert step.solar_to_battery_kw == 0.0
    assert step.solar_to_electrolyser_kw == 0.0
    assert step.solar_to_dac_kw == 0.0
    assert step.solar_to_methanator_kw == 0.0
    assert step.battery_support_kw > 0.0
    assert step.running is True
    conservative_specific_energy = model.metadata()["constants"][
        "conservative_electrolyser_specific_energy_kwh_per_kg"
    ]
    expected_h2_kg = (
        step.battery_support_kw * (INTERVAL_S / 3_600.0)
    ) / conservative_specific_energy
    assert step.h2_produced_kg == pytest.approx(expected_h2_kg)
    assert step.h2_end_kg == pytest.approx(expected_h2_kg)
    expected_end_kwh = 0.50 * config.battery.capacity_kwh - (
        step.battery_support_kw
        * (INTERVAL_S / 3_600.0)
        / config.battery.discharge_efficiency
    )
    assert step.battery_energy_end_kwh == pytest.approx(expected_end_kwh)
    assert step.battery_energy_end_kwh >= 0.25 * config.battery.capacity_kwh


@pytest.mark.parametrize(
    ("minimum_down_s", "expected_start", "horizon"),
    (
        (0.0, 0, 8),
        (1_800.0, 2, 10),
    ),
)
def test_cold_start_delay_energy_and_minimum_times_are_hand_solved(
    minimum_down_s: float,
    expected_start: int,
    horizon: int,
) -> None:
    config = PlantConfig()
    model, plan = _solve_case(
        config,
        horizon_intervals=horizon,
        requested_kg=100.0,
        resource_scale=1.0,
        battery_soc=0.25,
        h2_kg=0.0,
        co2_kg=1_000.0,
        ch4_kg=0.0,
        mode=ElectrolyserMode.OFF,
        min_down_remaining_s=minimum_down_s,
    )

    start_indices = tuple(
        index for index, step in enumerate(plan.steps) if step.cold_start
    )
    warmup_indices = tuple(
        index for index, step in enumerate(plan.steps) if step.warmup
    )
    running_indices = tuple(
        index for index, step in enumerate(plan.steps) if step.running
    )
    assert start_indices == (expected_start,)
    assert warmup_indices == tuple(range(expected_start, expected_start + 4))
    assert running_indices == tuple(range(expected_start + 4, expected_start + 8))
    start_power_kw = model.metadata()["constants"]["electrolyser_start_power_kw"]
    startup_energy_kwh = math.fsum(
        step.electrolyser_startup_kw * 0.25 for step in plan.steps
    )
    assert startup_energy_kwh == pytest.approx(start_power_kw * 1.0)
    assert all(
        step.h2_produced_kg == pytest.approx(0.0, abs=1e-8)
        for step in plan.steps[expected_start : expected_start + 4]
    )

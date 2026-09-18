"""Structural audits for the tariff-aware economic MPC model."""

from __future__ import annotations

from dataclasses import replace

import pytest

from plant_backend.rl.v2.beta_mpc_schema import BetaMPCSettingsV1
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.economic_env import PlantEconomicSetpointEnvV2
from plant_backend.rl.v2.economic_mpc_model import build_economic_mpc_model
from plant_backend.rl.v2.economic_mpc_schema import (
    EconomicMPCInputError,
    EconomicMPCSettingsV1,
)
from plant_backend.rl.v2.mpc_forecast import project_issued_mean_solar


SETTINGS = EconomicMPCSettingsV1(task_horizon_h=1.0, planning_horizon_h=1.0)


def _inputs():  # type: ignore[no-untyped-def]
    episode = EpisodeConfigV2(
        horizon_h=1.0,
        delivery_period_h=1.0,
        delivery_rate_kg_h=35.0,
        start_hour=18.0,
    )
    env = PlantEconomicSetpointEnvV2(episode=episode)
    env.reset(seed=0)
    observation = env.economic_controller_observation
    assert observation.plant.forecast is not None
    projection = project_issued_mean_solar(
        observation.plant.forecast,
        boundary_time_s=observation.plant.state.boundary.time_s,
        config=env.config,
        interval_s=SETTINGS.interval_s,
    )
    return env, observation, projection


def _build():  # type: ignore[no-untyped-def]
    env, observation, projection = _inputs()
    boundary = observation.plant.state.boundary
    assumptions = env.economic_assumptions
    h2_ratio = env.config.methanator.h2_kg_per_kg_ch4
    co2_ratio = env.config.methanator.co2_kg_per_kg_ch4
    initial_ch4 = boundary.ch4_level_kg + min(
        boundary.h2_level_kg / h2_ratio,
        boundary.co2_level_kg / co2_ratio,
    )
    return env, build_economic_mpc_model(
        observation,
        projection,
        settings=SETTINGS,
        assumptions=assumptions,
        constraints=env.constraints,
        config=env.config,
        commitment_profile=env.episode_config.electrolyser_commitment,
        horizon_intervals=4,
        initial_ch4_equivalent_kg=initial_ch4,
        initial_battery_energy_kwh=boundary.battery_energy_kwh,
    )


def test_model_appends_exact_non_rewarding_endowment_hinges() -> None:
    _, model = _build()

    assert model.variable_count == model.physical_model.variable_count + 2
    assert model.row_count == model.physical_model.row_count + 2
    methane = model.group("terminal_ch4_equivalent_deficit_kg")[0]
    battery = model.group("terminal_battery_deficit_kwh")[0]
    assert model.variable_lower[methane] == model.variable_lower[battery] == 0.0
    assert model.variable_upper[methane] == model.initial_ch4_equivalent_kg
    assert model.variable_upper[battery] == model.initial_battery_energy_kwh
    assert model.economic_objective[methane] > 0.0
    assert model.economic_objective[battery] > 0.0
    assert {row.name for row in model.hinge_rows} == {
        "terminal_ch4_endowment_hinge",
        "terminal_battery_endowment_hinge",
    }


def test_objective_prices_accepted_solar_and_bus_side_throughput_exactly() -> None:
    env, model = _build()
    dt_h = SETTINGS.interval_s / 3_600.0
    price_term = dt_h * 150.0 / 1_000.0
    throughput_term = dt_h * env.economic_assumptions.battery_throughput_eur_per_kwh

    assert model.interval_prices_eur_per_mwh[0] == 150.0
    assert model.economic_objective[
        model.group("solar_to_electrolyser_kw")[0]
    ] == pytest.approx(price_term)
    assert model.economic_objective[
        model.group("solar_to_battery_kw")[0]
    ] == pytest.approx(price_term + throughput_term)
    assert model.economic_objective[
        model.group("battery_support_kw")[0]
    ] == pytest.approx(throughput_term)
    assert model.economic_objective[model.group("solar_curtailed_kw")[0]] == 0.0


def test_builder_recomputes_summary_from_full_declared_tariff() -> None:
    env, observation, projection = _inputs()
    corrupted = replace(
        observation,
        tariff=replace(
            observation.tariff,
            next_6h_mean_eur_per_mwh=(
                observation.tariff.next_6h_mean_eur_per_mwh + 1.0
            ),
        ),
    )
    boundary = observation.plant.state.boundary

    with pytest.raises(EconomicMPCInputError, match="full declared"):
        build_economic_mpc_model(
            corrupted,
            projection,
            settings=SETTINGS,
            assumptions=env.economic_assumptions,
            constraints=env.constraints,
            config=env.config,
            commitment_profile=env.episode_config.electrolyser_commitment,
            horizon_intervals=4,
            initial_ch4_equivalent_kg=boundary.ch4_level_kg,
            initial_battery_energy_kwh=boundary.battery_energy_kwh,
        )


def test_alpha_builder_rejects_beta_settings_identity() -> None:
    env, observation, projection = _inputs()
    boundary = observation.plant.state.boundary

    with pytest.raises(TypeError, match="settings must be EconomicMPCSettingsV1"):
        build_economic_mpc_model(
            observation,
            projection,
            settings=BetaMPCSettingsV1(),
            assumptions=env.economic_assumptions,
            constraints=env.constraints,
            config=env.config,
            commitment_profile=env.episode_config.electrolyser_commitment,
            horizon_intervals=4,
            initial_ch4_equivalent_kg=boundary.ch4_level_kg,
            initial_battery_energy_kwh=boundary.battery_energy_kwh,
        )

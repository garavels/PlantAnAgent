"""Numerical evidence tests for the two-stage economic MPC solve."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.economic_env import PlantEconomicSetpointEnvV2
from plant_backend.rl.v2.economic_mpc_model import build_economic_mpc_model
from plant_backend.rl.v2.economic_mpc_schema import EconomicMPCSettingsV1
from plant_backend.rl.v2.economic_mpc_solver import solve_economic_mpc_model
from plant_backend.rl.v2.economics import ch4_equivalent_inventory_kg
from plant_backend.rl.v2.mpc_forecast import project_issued_mean_solar


def test_solver_preserves_service_then_reconstructs_every_euro() -> None:
    pytest.importorskip("scipy")
    settings = EconomicMPCSettingsV1(
        task_horizon_h=1.0,
        planning_horizon_h=1.0,
    )
    env = PlantEconomicSetpointEnvV2(
        episode=EpisodeConfigV2(
            horizon_h=1.0,
            delivery_period_h=1.0,
            delivery_rate_kg_h=35.0,
        )
    )
    env.reset(seed=0)
    observation = env.economic_controller_observation
    assert observation.plant.forecast is not None
    projection = project_issued_mean_solar(
        observation.plant.forecast,
        boundary_time_s=observation.plant.state.boundary.time_s,
        config=env.config,
        interval_s=settings.interval_s,
    )
    boundary = observation.plant.state.boundary
    model = build_economic_mpc_model(
        observation,
        projection,
        settings=settings,
        assumptions=env.economic_assumptions,
        constraints=env.constraints,
        config=env.config,
        commitment_profile=env.episode_config.electrolyser_commitment,
        horizon_intervals=4,
        initial_ch4_equivalent_kg=ch4_equivalent_inventory_kg(boundary, env.config),
        initial_battery_energy_kwh=boundary.battery_energy_kwh,
    )

    plan = solve_economic_mpc_model(model, settings=settings)

    assert tuple(stage.stage for stage in plan.stages) == (
        "service_maximization",
        "economic_cost_minimization",
    )
    assert all(stage.success and stage.solution_sha256 for stage in plan.stages)
    assert plan.selected_delivered_kg >= (
        plan.primary_delivered_optimum_kg - settings.delivery_lock_tolerance_kg - 1e-7
    )
    assert plan.predicted_economic_cost_eur == pytest.approx(
        plan.stages[1].objective_value,
        abs=1e-7,
    )
    assert math.isclose(
        plan.predicted_economic_cost_eur,
        math.fsum(dict(plan.predicted_cost_components_eur).values()),
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    assert len(plan.sha256) == 64

    with pytest.raises(ValueError, match="selected delivery"):
        replace(plan, selected_delivered_kg=plan.selected_delivered_kg + 1.0)
    changed_components = list(plan.predicted_cost_components_eur)
    name, value = changed_components[0]
    changed_components[0] = (name, value + 1.0)
    with pytest.raises(ValueError, match="cost ledger"):
        replace(plan, predicted_cost_components_eur=tuple(changed_components))

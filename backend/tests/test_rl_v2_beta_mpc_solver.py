"""Numerical and fail-closed checks for the hard-return beta MPC solver."""

from __future__ import annotations

from dataclasses import replace

import pytest

from plant_backend.rl.v2.beta_config import CyclicMarketTaskConfigV1
from plant_backend.rl.v2.beta_disturbances import DisturbanceConfigV1
from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta
from plant_backend.rl.v2.beta_mpc_model import BetaMPCModelV1, build_beta_mpc_model
from plant_backend.rl.v2.beta_mpc_schema import (
    BETA_MPC_SOLUTION_DOMAIN,
    BetaMPCSettingsV1,
    BetaMPCSolveError,
    project_beta_mpc_prices,
)
from plant_backend.rl.v2.beta_mpc_solver import solve_beta_mpc_model
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.economic_mpc_model import economic_delivery_lock_row
from plant_backend.rl.v2.economic_mpc_schema import EconomicMPCPlanError
from plant_backend.rl.v2.economic_mpc_solver import (
    _economic_mpc_plan_values_from_solve_result,
    economic_mpc_plan_from_solve_result,
    solve_economic_mpc_stages,
)
from plant_backend.rl.v2.economics import ch4_equivalent_inventory_kg
from plant_backend.rl.v2.market import MarketConfigV1
from plant_backend.rl.v2.mpc_forecast import project_issued_mean_solar
from plant_backend.rl.v2.mpc_schema import domain_sha256


def _model() -> tuple[PlantCyclicMarketEnvV2Beta, BetaMPCSettingsV1, BetaMPCModelV1]:
    episode = EpisodeConfigV2(
        horizon_h=24.0,
        decision_interval_s=3_600.0,
        start_hour=6.0,
        delivery_period_h=24.0,
        delivery_rate_kg_h=1.0,
        battery_soc=(0.5, 0.5),
        h2_fill=(0.1, 0.1),
        co2_fill=(0.1, 0.1),
        ch4_coverage_h=(1.0, 1.0),
    )
    task = CyclicMarketTaskConfigV1(
        market=MarketConfigV1(forecast_ensemble_members=10),
        disturbances=DisturbanceConfigV1(
            enabled=False,
            per_cycle_probability=0.0,
        ),
    )
    environment = PlantCyclicMarketEnvV2Beta(episode=episode, beta_task=task)
    environment.reset(seed=1702)
    settings = BetaMPCSettingsV1(
        task_horizon_h=24.0,
        planning_horizon_h=24.0,
        interval_s=3_600.0,
        solver_time_limit_s=10.0,
    )
    observation = environment.cyclic_market_controller_observation
    assert observation.plant.forecast is not None
    weather = project_issued_mean_solar(
        observation.plant.forecast,
        boundary_time_s=observation.plant.state.boundary.time_s,
        config=environment.config,
        interval_s=settings.interval_s,
    )
    price = project_beta_mpc_prices(
        observation.market,
        interval_s=settings.interval_s,
        horizon_intervals=24,
    )
    boundary = observation.plant.state.boundary
    model = build_beta_mpc_model(
        observation,
        weather,
        price,
        settings=settings,
        assumptions=environment.economic_assumptions,
        beta_task=environment.beta_task,
        constraints=environment.constraints,
        config=environment.config,
        commitment_profile=environment.episode_config.electrolyser_commitment,
        initial_ch4_equivalent_kg=ch4_equivalent_inventory_kg(
            boundary,
            environment.config,
        ),
        initial_battery_energy_kwh=boundary.battery_energy_kwh,
    )
    return environment, settings, model


def test_beta_solver_publishes_beta_domain_and_feasible_terminal_returns() -> None:
    pytest.importorskip("scipy")
    environment, settings, model = _model()
    try:
        plan = solve_beta_mpc_model(model, settings=settings)
    finally:
        environment.close()

    assert plan.model_sha256 == model.sha256
    terminal = plan.steps[-1]
    terminal_values = {
        "battery": terminal.battery_energy_end_kwh,
        "h2": terminal.h2_end_kg,
        "co2": terminal.co2_end_kg,
        "ch4": terminal.ch4_end_kg,
    }
    for requirement in plan.cycle_return.resources:
        assert terminal_values[requirement.resource] >= (
            requirement.required_minimum - 1e-6
        )
    assert all(stage.success for stage in plan.stages)
    assert len(plan.selected_solution_vector) == model.variable_count
    lock = economic_delivery_lock_row(
        model,
        delivered_optimum_kg=plan.primary_delivered_optimum_kg,
        tolerance_kg=plan.delivery_lock_tolerance_kg,
    )
    assert model.audit_solution(
        plan.selected_solution_vector,
        tolerance=2.0 * settings.solver_mip_feasibility_tolerance,
        extra_rows=(lock,),
    ).valid
    assert plan.stages[1].solution_sha256 == domain_sha256(
        BETA_MPC_SOLUTION_DOMAIN,
        {
            "model_sha256": model.sha256,
            "stage": "economic_cost_minimization",
            "extra_rows": [lock.to_dict()],
            "values": list(plan.selected_solution_vector),
        },
    )
    assert BETA_MPC_SOLUTION_DOMAIN == "cyclic-market-beta-mpc-solution-vector-v1"


def test_cycle_rows_are_hard_during_primary_service_stage() -> None:
    pytest.importorskip("scipy")
    environment, settings, model = _model()
    try:
        battery = model.cycle_return.resources[0]
        terminal_index = model.group("battery_energy_kwh")[-1]
        impossible = model.variable_upper[terminal_index] + 1.0
        impossible_battery = replace(
            battery,
            current=impossible,
            signed_margin=0.0,
            target=impossible,
            tolerance=0.0,
            required_minimum=impossible,
        )
        cycle_return = replace(
            model.cycle_return,
            resources=(
                impossible_battery,
                model.cycle_return.resources[1],
                model.cycle_return.resources[2],
                model.cycle_return.resources[3],
            ),
        )
        rows = (
            replace(model.return_rows[0], lower=impossible),
            model.return_rows[1],
            model.return_rows[2],
            model.return_rows[3],
        )
        infeasible = replace(model, cycle_return=cycle_return, return_rows=rows)

        with pytest.raises(BetaMPCSolveError) as captured:
            solve_beta_mpc_model(infeasible, settings=settings)
    finally:
        environment.close()

    assert len(captured.value.stages) == 1
    assert captured.value.stages[0].stage == "service_maximization"
    assert captured.value.stages[0].success is False


def test_shared_solver_result_is_reaudited_before_plan_extraction() -> None:
    pytest.importorskip("scipy")
    environment, settings, model = _model()
    try:
        solved = solve_economic_mpc_stages(
            model,
            settings=settings,
            solution_domain=BETA_MPC_SOLUTION_DOMAIN,
        )
        with pytest.raises(TypeError, match="model must be EconomicMPCModelV1"):
            economic_mpc_plan_from_solve_result(
                model,
                settings=settings,
                solved=solved,
            )
        wrong_model = replace(solved, model_sha256="0" * 64)
        with pytest.raises(EconomicMPCPlanError, match="model identity"):
            _economic_mpc_plan_values_from_solve_result(
                model,
                settings=settings,
                solved=wrong_model,
                expected_solution_domain=BETA_MPC_SOLUTION_DOMAIN,
            )

        altered_primary_stage = replace(
            solved.stages[0],
            solution_sha256="0" * 64,
        )
        altered_digest = replace(
            solved,
            stages=(altered_primary_stage, solved.stages[1]),
        )
        with pytest.raises(EconomicMPCPlanError, match="solution digest"):
            _economic_mpc_plan_values_from_solve_result(
                model,
                settings=settings,
                solved=altered_digest,
                expected_solution_domain=BETA_MPC_SOLUTION_DOMAIN,
            )
    finally:
        environment.close()

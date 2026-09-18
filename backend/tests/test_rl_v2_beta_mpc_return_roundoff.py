"""Saved fixture boundary regression; no environment or optimizer execution."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from plant_backend.rl.v2 import economic_mpc_solver as economic_solver
from plant_backend.rl.v2.beta_controller_benchmark_verifier import (
    _decode_beta_observation,
)
from plant_backend.rl.v2.beta_mpc_model import BetaMPCModelV1, build_beta_mpc_model
from plant_backend.rl.v2.beta_mpc_schema import (
    BETA_MPC_MODEL_VERSION,
    BetaMPCSettingsV1,
    BetaMPCSolveError,
    project_beta_mpc_prices,
)
from plant_backend.rl.v2.beta_mpc_solver import solve_beta_mpc_model
from plant_backend.rl.v2.mpc_forecast import project_issued_mean_solar
from plant_backend.rl.v2.training.config import ControllerNeutralTrainingConfigV1


@pytest.fixture
def retained_model() -> tuple[dict[str, Any], BetaMPCSettingsV1, BetaMPCModelV1]:
    fixture = json.loads(
        (
            Path(__file__).parent / "data/beta_mpc_terminal_return_roundoff.json"
        ).read_text()
    )
    assert fixture["fixture_seed"] == 9_100_001
    config = ControllerNeutralTrainingConfigV1.from_dict(fixture["task"])
    assert config.beta_rl is None and config.beta_task is not None
    observation = _decode_beta_observation(fixture["boundary"], name="saved fixture")
    assert observation.cycle.decision_in_cycle == 52
    settings = BetaMPCSettingsV1(**fixture["settings_parameters"])
    assert observation.plant.forecast is not None
    weather = project_issued_mean_solar(
        observation.plant.forecast,
        boundary_time_s=observation.plant.state.boundary.time_s,
        config=config.plant,
        interval_s=settings.interval_s,
        energy_tolerance=settings.energy_reconstruction_tolerance,
    )
    prices = project_beta_mpc_prices(
        observation.market, interval_s=settings.interval_s, horizon_intervals=44
    )
    model = build_beta_mpc_model(
        observation,
        weather,
        prices,
        settings=settings,
        assumptions=config.economic_assumptions,
        beta_task=config.beta_task,
        constraints=config.constraints,
        config=config.plant,
        commitment_profile=config.episode.electrolyser_commitment,
        initial_ch4_equivalent_kg=fixture["initial_ch4_equivalent_kg"],
        initial_battery_energy_kwh=fixture["initial_battery_energy_kwh"],
    )
    return fixture, settings, model


def _fixed_solver(
    monkeypatch: pytest.MonkeyPatch,
    fixture: dict[str, Any],
    model: BetaMPCModelV1,
    *,
    economic_battery: float | None = None,
) -> list[str]:
    runtime = economic_solver._load_runtime()
    calls: list[str] = []

    def fixed_milp(**kwargs: Any) -> SimpleNamespace:
        assert len(calls) < 2, "no retry or fallback is permitted"
        name = "primary_solution" if not calls else "economic_solution"
        calls.append(name)
        values = np.array(fixture[name], dtype=np.float64)
        if name == "economic_solution" and economic_battery is not None:
            values[model.group("battery_energy_kwh")[-1]] = economic_battery
        np.testing.assert_array_equal(kwargs["bounds"].lb, model.variable_lower)
        return SimpleNamespace(
            x=values,
            fun=math.fsum(float(c) * float(v) for c, v in zip(kwargs["c"], values)),
            status=0,
            success=True,
            message="fixed saved-vector response; no optimization executed",
            mip_gap=0.0,
            mip_node_count=0,
        )

    monkeypatch.setattr(
        economic_solver, "_load_runtime", lambda: replace(runtime, milp=fixed_milp)
    )
    return calls


def test_saved_two_ulp_deficit_is_projected_before_strict_plan_validation(
    retained_model: tuple[dict[str, Any], BetaMPCSettingsV1, BetaMPCModelV1],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, settings, model = retained_model
    index = model.group("battery_energy_kwh")[-1]
    required = model.cycle_return.resource("battery").required_minimum
    assert required - fixture["economic_solution"][index] == 2 * math.ulp(required)
    assert model.sha256 != fixture["original_model_sha256"]
    assert BETA_MPC_MODEL_VERSION == "cyclic-return-economic-mpc-v2"
    calls = _fixed_solver(monkeypatch, fixture, model)

    plan = solve_beta_mpc_model(model, settings=settings)

    assert calls == ["primary_solution", "economic_solution"]
    assert plan.selected_solution_vector[index] == required
    assert plan.steps[-1].battery_energy_end_kwh == required
    assert (
        plan.stages[1].solution_sha256 != fixture["original_stage_solution_sha256"][1]
    )
    assert model.audit_solution(plan.selected_solution_vector).valid
    for row in model.return_rows:
        assert plan.selected_solution_vector[row.coefficients[0][0]] >= row.lower


def test_feasible_interior_battery_is_not_snapped_to_return_boundary(
    retained_model: tuple[dict[str, Any], BetaMPCSettingsV1, BetaMPCModelV1],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, settings, model = retained_model
    required = model.cycle_return.resource("battery").required_minimum
    interior = math.nextafter(required, math.inf)
    calls = _fixed_solver(monkeypatch, fixture, model, economic_battery=interior)
    plan = solve_beta_mpc_model(model, settings=settings)
    assert plan.steps[-1].battery_energy_end_kwh == interior
    assert len(calls) == 2


def test_material_return_deficit_still_fails_without_fallback(
    retained_model: tuple[dict[str, Any], BetaMPCSettingsV1, BetaMPCModelV1],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, settings, model = retained_model
    required = model.cycle_return.resource("battery").required_minimum
    calls = _fixed_solver(monkeypatch, fixture, model, economic_battery=required - 1.0)
    with pytest.raises(BetaMPCSolveError, match="raw solver solution failed"):
        solve_beta_mpc_model(model, settings=settings)
    assert len(calls) == 2


def test_all_four_return_bounds_are_redundant_and_other_bounds_are_unchanged(
    retained_model: tuple[dict[str, Any], BetaMPCSettingsV1, BetaMPCModelV1],
) -> None:
    _, _, model = retained_model
    terminal_indices = {row.coefficients[0][0] for row in model.return_rows}
    for row in model.return_rows:
        index = row.coefficients[0][0]
        assert model.variable_lower[index] == max(
            model.economic_model.variable_lower[index], row.lower
        )
    for index, lower in enumerate(model.variable_lower):
        if index not in terminal_indices:
            assert lower == model.economic_model.variable_lower[index]
    assert model.variable_upper == model.economic_model.variable_upper
    assert model.row_count == model.economic_model.row_count + 4


def test_lower_cycle_target_does_not_weaken_physical_battery_reserve(
    retained_model: tuple[dict[str, Any], BetaMPCSettingsV1, BetaMPCModelV1],
) -> None:
    _, _, model = retained_model
    battery = model.cycle_return.resource("battery")
    weaker = replace(
        battery,
        target=0.0,
        signed_margin=battery.current,
        tolerance=0.0,
        required_minimum=0.0,
    )
    changed = replace(
        model,
        cycle_return=replace(
            model.cycle_return, resources=(weaker, *model.cycle_return.resources[1:])
        ),
        return_rows=(replace(model.return_rows[0], lower=0.0), *model.return_rows[1:]),
    )
    terminal = changed.group("battery_energy_kwh")[-1]
    assert (
        changed.variable_lower[terminal]
        == model.economic_model.variable_lower[terminal]
    )

"""Public uncertainty policy regressions; fixture domains only."""

from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from plant_backend.rl.v2.beta_config import CyclicMarketTaskConfigV1
from plant_backend.rl.v2.beta_disturbances import DisturbanceConfigV1
from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta
from plant_backend.rl.v2.beta_mpc_controller import CyclicMarketEconomicMPCControllerV1
from plant_backend.rl.v2.beta_mpc_model import build_beta_mpc_model
from plant_backend.rl.v2.beta_mpc_schema import (
    BETA_MPC_UNCERTAINTY_POLICY,
    BetaMPCInputError,
    BetaMPCSettingsV1,
    project_beta_mpc_prices,
)
from plant_backend.rl.v2.beta_mpc_uncertainty import (
    early_reserve_rows,
    project_beta_mpc_solar,
)
from plant_backend.rl.v2.beta_controller_benchmark_verifier import _parse_beta_settings
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.economics import ch4_equivalent_inventory_kg
from plant_backend.rl.v2.mpc_forecast import (
    project_issued_mean_solar,
    public_forecast_issue_sha256,
)


@pytest.fixture
def case():
    env = PlantCyclicMarketEnvV2Beta(
        episode=EpisodeConfigV2(
            horizon_h=24.0,
            decision_interval_s=900.0,
            start_hour=6.0,
            delivery_period_h=24.0,
            delivery_rate_kg_h=1.0,
            battery_soc=(0.6, 0.6),
            h2_fill=(0.3, 0.3),
            co2_fill=(0.3, 0.3),
            ch4_coverage_h=(1.0, 1.0),
        ),
        beta_task=CyclicMarketTaskConfigV1(
            disturbances=DisturbanceConfigV1(enabled=False, per_cycle_probability=0.0)
        ),
    )
    env.reset(seed=9_100_010)
    settings = BetaMPCSettingsV1(
        task_horizon_h=24.0,
        interval_s=900.0,
        uncertainty_policy=BETA_MPC_UNCERTAINTY_POLICY,
    )
    try:
        yield env, settings
    finally:
        env.close()


def build(case, *, weather=None, settings=None):
    env, default = case
    settings = default if settings is None else settings
    obs = env.cyclic_market_controller_observation
    weather = weather or project_beta_mpc_solar(
        obs.plant.forecast,
        boundary_time_s=obs.plant.state.boundary.time_s,
        config=env.config,
        settings=settings,
    )
    model = build_beta_mpc_model(
        obs,
        weather,
        project_beta_mpc_prices(obs.market, interval_s=900.0, horizon_intervals=96),
        settings=settings,
        assumptions=env.economic_assumptions,
        beta_task=env.beta_task,
        constraints=env.constraints,
        config=env.config,
        commitment_profile=env.episode_config.electrolyser_commitment,
        initial_ch4_equivalent_kg=ch4_equivalent_inventory_kg(
            obs.plant.state.boundary, env.config
        ),
        initial_battery_energy_kwh=obs.plant.state.boundary.battery_energy_kwh,
    )
    return model, weather


def test_p10_reconstructs_lower_band_energy_and_binds_original_issue(case):
    env, settings = case
    issue = env.cyclic_market_controller_observation.plant.forecast
    model, lower = build(case)
    mean = project_issued_mean_solar(
        issue, boundary_time_s=21600.0, config=env.config, interval_s=900.0
    )
    assert (
        lower.source_issue_sha256
        == mean.source_issue_sha256
        == public_forecast_issue_sha256(issue)
    )
    assert lower.sha256 != mean.sha256
    assert all(
        lo <= hi + 1e-10 for lo, hi in zip(lower.available_kw, mean.available_kw)
    )
    for window, band in zip(issue.windows, lower.bands):
        assert band.reconstructed_mean_peak_fraction == pytest.approx(
            window.p10_peak_power_fraction, abs=1e-12
        )
    assert all(
        value == 0
        for value, clear in zip(lower.available_kw, lower.clear_sky_mean_peak_fractions)
        if clear == 0
    )
    with pytest.raises(BetaMPCInputError, match="weather projection"):
        build(case, weather=mean)


def test_four_original_minima_unchanged_and_early_buffers_are_capacity_based(case):
    model, _ = build(case)
    legacy, _ = build(
        case, settings=replace(case[1], uncertainty_policy="issued-mean-v1")
    )
    assert model.return_rows == legacy.return_rows
    assert model.cycle_return == legacy.cycle_return
    assert not legacy.reserve_rows and not legacy.power_rows
    audit = model.metadata()["uncertainty_policy"]
    assert audit["restoration_boundary"] == 46  # 17:30, two hours before sunset
    assert [r["requested_buffer"] for r in audit["resources"]] == [
        80.0,
        20.0,
        60.0,
        30.0,
    ]
    assert all(
        row.name.endswith(f"[{i}]")
        for row, i in zip(model.reserve_rows[:51], range(46, 97))
    )
    assert len(model.reserve_rows) == 4 * 51
    for row in model.reserve_rows:
        assert model.variable_lower[row.coefficients[0][0]] >= row.lower
    assert not any("slack" in name for name in model.variable_names)


def test_sunset_constant_power_is_below_average_and_has_no_future_truth(case):
    model, weather = build(case)
    last = 53  # 19:15 to 19:30
    firm = model.metadata()["uncertainty_policy"]["constant_solar_power_kw"][last]
    clear_last_minute = math.sin(math.pi * ((19.5 - 1 / 60) - 6.0) / 13.5)
    expected = (
        weather.available_kw[last]
        * clear_last_minute
        / weather.clear_sky_mean_peak_fractions[last]
    )
    assert firm == pytest.approx(expected)
    assert 0 < firm < weather.available_kw[last] / 5
    assert model.power_rows[last].upper == firm
    assert all(row.upper == 0 for row in model.power_rows[54:])
    assert model.power_rows[0].upper == 0  # sample exactly at sunrise


def test_spent_buffer_is_not_demanded_back_after_deadline(case):
    model, weather = build(case)
    # Reconstruct a public 19:30 boundary; zero remaining daylight, no plant run.
    env, settings = case
    issue = env.cyclic_market_controller_observation.plant.forecast
    windows = tuple(
        replace(w, mean_peak_power_fraction=0.0, p10_peak_power_fraction=0.0)
        for w in issue.windows
    )
    dark = project_beta_mpc_solar(
        replace(issue, windows=windows),
        boundary_time_s=19.5 * 3600,
        config=env.config,
        settings=settings,
    )
    resources = tuple(
        replace(
            r,
            current=r.required_minimum + 0.25,
            signed_margin=r.required_minimum + 0.25 - r.target,
        )
        for r in model.cycle_return.resources
    )
    cycle = replace(
        model.cycle_return,
        decision_in_cycle=54,
        issued_at_decision=54,
        remaining_intervals=42,
        resources=resources,
    )
    rows, audit = early_reserve_rows(model.economic_model, cycle, dark, env.config)
    assert audit["restoration_boundary"] == 0
    assert all(
        r["retained_buffer"] == pytest.approx(0.25, abs=1e-12)
        for r in audit["resources"]
    )
    assert len(rows) == 4 * 42
    assert all(
        r["planning_floor"] >= r["required_cycle_minimum"] for r in audit["resources"]
    )


def test_power_constraints_remain_hash_bound_without_optional_reserve_rows(case):
    model, _ = build(case)
    unbuffered = replace(model, reserve_rows=())
    changed = replace(
        unbuffered,
        power_rows=(
            replace(unbuffered.power_rows[0], upper=1.0),
            *unbuffered.power_rows[1:],
        ),
    )
    assert changed.rows != unbuffered.rows
    assert changed.sha256 != unbuffered.sha256


def test_uncertainty_descriptor_roundtrip_and_tampering(case):
    settings = case[1]
    assert _parse_beta_settings(settings.to_dict()) == settings
    assert settings.to_dict()["forecast_statistic"] == "issued_band_p10"
    assert (
        settings.to_dict()["forecast_projection_version"]
        == "issued-band-p10-clear-sky-scaled-v1"
    )
    changed = settings.to_dict()
    changed["resource_buffer_capacity_fraction"] = 0.0
    with pytest.raises(ValueError):
        _parse_beta_settings(changed)
    with pytest.raises(ValueError, match="uncertainty policy"):
        replace(settings, uncertainty_policy="automatic-fallback")
    assert (
        "uncertainty_policy"
        not in replace(settings, uncertainty_policy="issued-mean-v1").to_dict()
    )


def test_real_solver_obeys_all_new_rows_and_exposes_reserve_evidence(case):
    env, settings = case
    controller = CyclicMarketEconomicMPCControllerV1(
        settings=settings, assumptions=env.economic_assumptions, beta_task=env.beta_task
    )
    decision = controller.compute_command(
        env.cyclic_market_controller_observation,
        constraints=env.constraints,
        config=env.config,
        commitment_profile=env.episode_config.electrolyser_commitment,
    )
    model, _ = build(case)
    assert model.audit_solution(decision.evidence.plan.selected_solution_vector).valid
    assert (
        decision.evidence.plan.to_dict()["uncertainty_policy"]
        == model.metadata()["uncertainty_policy"]
    )
    assert "band-p10" in controller.to_dict()["description"]


def test_retained_deficit_is_dispatch_mismatch_despite_more_available_solar():
    evidence = json.loads(
        (Path(__file__).parent / "data/beta_mpc_forecast_mismatch.json").read_text()
    )
    states = evidence["minute_records"]
    dt_h = (states[-1]["time_s"] - states[0]["time_s"]) / 3600 / 15
    available = states[-1]["energy_produced_kwh"] - states[0]["energy_produced_kwh"]
    curtailed = states[-1]["energy_curtailed_kwh"] - states[0]["energy_curtailed_kwh"]
    applied_charge = math.fsum(s["solar_to_battery_kw"] * dt_h for s in states[1:])
    actual_change = states[-1]["battery_energy_kwh"] - states[0]["battery_energy_kwh"]
    assert available > evidence["forecast_available_solar_kwh"]
    assert available - curtailed == pytest.approx(16.756857240052227)
    assert actual_change == pytest.approx(applied_charge * 0.96, abs=1e-9)
    assert evidence["planned_battery_change_kwh"] - actual_change == pytest.approx(
        1.16889234221253
    )
    assert states[1]["solar_curtailed_kw"] > 0
    assert states[-1]["solar_to_battery_kw"] < 1e-10


def test_mandatory_delivery_disjunction_matches_physical_minimum(case):
    model, _ = build(case)
    physical = model.physical_model
    request = physical.deliveries[0].requested_kg
    boundary = physical.deliveries[0].due_boundary
    delivered_index = physical.group("delivered_kg")[boundary - 1]
    inventory_index = physical.group("ch4_inventory_kg")[boundary]
    stockout_index = physical.group("mandatory_delivery_stockout")[0]
    rows = [row for row in physical.rows if row.name.startswith("mandatory_delivery_")]
    assert len(rows) == 2
    capacity = case[0].config.ch4_tank.capacity_kg
    # Exhaust both binary choices. This oracle does not depend on objective weights.
    for available in (0.0, request / 2, request, request + 50):
        for delivery in {
            0.0,
            request / 4,
            request / 2,
            request,
            min(available, request),
        }:
            accepted = False
            for stockout in (0.0, 1.0):
                values = {
                    delivered_index: delivery,
                    inventory_index: available - delivery,
                    stockout_index: stockout,
                }
                feasible = (
                    0 <= delivery <= request and 0 <= available - delivery <= capacity
                )
                for row in rows:
                    value = sum(
                        coefficient * values[index]
                        for index, coefficient in row.coefficients
                    )
                    feasible &= row.lower is None or value >= row.lower
                    feasible &= row.upper is None or value <= row.upper
                accepted |= feasible
            assert accepted == (delivery == min(available, request))
    legacy, _ = build(
        case, settings=replace(case[1], uncertainty_policy="issued-mean-v1")
    )
    assert not any(row.name.startswith("mandatory_delivery_") for row in legacy.rows)

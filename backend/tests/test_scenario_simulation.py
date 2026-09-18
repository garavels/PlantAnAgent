"""Authored scenarios reach real weather, inventories, faults and contracts."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from plant_backend.site_simulation import (
    SiteOperationalMPC,
    create_site_simulation,
    run_site_simulation,
    scenario_solar_scale,
)


@pytest.fixture
def scenario():
    return {
        "version": "plant-scenario-v1", "name": "One-day stress test",
        "description": "Low stocks, a capture outage and two deliveries.",
        "duration_days": 1, "seed": 42, "solar_kw": 5000, "battery_kwh": 8000,
        "initial": {"battery_soc": 0.3, "h2_kg": 0, "co2_kg": 0, "ch4_kg": 100},
        "sunlight": [],
        "faults": [{"type": "dac_outage", "severity": 1, "start_hour": 0,
                    "duration_hours": 1, "channel": None}],
        "deliveries": [{"due_hour": 0.25, "kg": 40}, {"due_hour": 1, "kg": 110}],
        "economics": {"delivered_service_value_eur_per_kg": 6,
                      "unmet_delivery_penalty_eur_per_kg": 40},
        "notes": [], "unsupported": [],
    }


@pytest.fixture
def forecast():
    start = datetime(2026, 9, 15, tzinfo=timezone.utc)
    return {
        "id": "scenario-test-48-hours", "start_time": start.isoformat(),
        "site": {"name": "Test site", "latitude": 45, "longitude": 13},
        "source": {"provider": "test", "forecast_issued_at": start.isoformat()},
        "hours": [
            {"time": (start + timedelta(hours=hour)).isoformat(), "shortwave_w_m2": 800}
            for hour in range(48)
        ],
    }


def test_authored_stocks_horizon_seed_and_reset_are_reproducible(forecast, scenario):
    scenario["solar_kw"] = 7000
    scenario["battery_kwh"] = 12000
    environment, _ = create_site_simulation(forecast, controller="H", scenario=scenario)
    try:
        assert environment.config.solar.peak_kw == 7000
        assert environment.config.battery.capacity_kwh == 12000
        assert environment.episode_decisions == 96
        assert environment.telemetry.battery_soc == 0.3
        assert environment.telemetry.h2_level_kg == 0
        assert environment.telemetry.co2_level_kg == 0
        assert environment.telemetry.ch4_level_kg == 100
        assert environment._episode.seed == 42
        assert environment._episode.scenario.faults[0].type == "dac_outage"
        assert environment.telemetry.health["dac"].offline
        manifest = environment._episode.audit_manifest()
        scenario["initial"]["ch4_kg"] = 999  # caller edits cannot mutate replay
        environment.reset()
        assert environment._episode.audit_manifest() == manifest
        assert environment.telemetry.ch4_level_kg == 100
    finally:
        environment.close()


def test_known_and_unexpected_sunlight_have_distinct_forecast_semantics(forecast, scenario):
    scenario["sunlight"] = [
        {"start_hour": 0, "end_hour": 6, "scale": 0.5, "forecast_known": True},
        {"start_hour": 6, "end_hour": 12, "scale": 0.25, "forecast_known": False},
    ]
    environment, _ = create_site_simulation(forecast, controller="H", scenario=scenario)
    try:
        assert scenario_solar_scale(scenario, 0, forecast_only=True) == 0.5
        assert scenario_solar_scale(scenario, 6 * 3600, forecast_only=True) == 1
        assert scenario_solar_scale(scenario, 6 * 3600) == 0.25
        assert scenario_solar_scale(scenario, 12 * 3600) == 1
        assert environment.profile.at(0) == pytest.approx(0.68 * 0.5)
        assert environment.profile.at(6 * 3600) == pytest.approx(0.68)
        assert environment._episode.weather.tape.at_boundary(0) == 0.5
        assert environment._episode.weather.tape.at_boundary(6 * 60) == 0.5
        assert environment._episode.weather.tape.at_boundary(6 * 60 + 1) == 0.25
        assert environment.telemetry.solar_available_kw == pytest.approx(1700)
        # The final decision still has a complete 24-hour projection.
        assert len(environment.profile.intervals(23.75 * 3600)) == 96
    finally:
        environment.close()


def test_sunlight_windows_cover_full_physical_hours_without_early_transition(forecast, scenario):
    scenario["sunlight"] = [
        {"start_hour": 0, "end_hour": 1, "scale": 0.5, "forecast_known": True},
        {"start_hour": 1, "end_hour": 2, "scale": 0.25, "forecast_known": False},
    ]
    environment, policy = create_site_simulation(forecast, controller="H", scenario=scenario)
    try:
        for index in range(8):
            decision = policy.compute_command(
                environment.cyclic_market_controller_observation.plant,
                constraints=environment.constraints, config=environment.config,
                commitment_profile=environment.episode_config.electrolyser_commitment,
            )
            environment.step_process_setpoints(decision.command)
            expected_kw = 1700 if index < 4 else 850
            assert all(
                state.solar_available_kw == pytest.approx(expected_kw)
                for state in environment.last_minute_truth_states[1:]
            )
            if index == 3:
                assert environment.telemetry.energy_produced_kwh == pytest.approx(1700)
        assert environment.telemetry.energy_produced_kwh == pytest.approx(2550)
    finally:
        environment.close()


def test_custom_contracts_settle_physically_and_economics_use_authored_values(forecast, scenario):
    for hour in forecast["hours"]:
        hour["shortwave_w_m2"] = 0
    result = run_site_simulation(forecast, controller="H", scenario=scenario)
    assert result["horizon_hours"] == 24
    assert len(result["history"]) == 25
    assert result["metrics"]["delivery_requested_kg"] == 150
    assert result["metrics"]["delivery_delivered_kg"] == 100
    assert result["metrics"]["methane_produced_kg"] == 0
    assert result["history"][1]["fault_active"]
    assert not result["history"][2]["fault_active"]
    assert result["economic_assumptions"]["delivered_service_value_eur_per_kg"] == 6
    assert result["economic_assumptions"]["unmet_delivery_penalty_eur_per_kg"] == 40
    terms = result["economic_accounting"]["components_eur"]
    assert terms["delivered_service_value"] == pytest.approx(600)
    assert terms["unmet_delivery_penalty"] == pytest.approx(-2000)


def test_mpc_receives_exact_irregular_deliveries_and_task_horizon(forecast, scenario):
    pytest.importorskip("scipy.optimize")
    scenario["faults"] = []
    scenario["deliveries"] = [
        {"due_hour": 4, "kg": 40}, {"due_hour": 8.25, "kg": 110},
        {"due_hour": 24, "kg": 20},
    ]
    environment, policy = create_site_simulation(forecast, controller="MPC", scenario=scenario)
    try:
        assert policy.settings.task_interval_count == 96
        model = policy.build_model(
            environment.cyclic_market_controller_observation,
            constraints=environment.constraints, config=environment.config,
            commitment_profile=environment.episode_config.electrolyser_commitment,
        )
        assert [(item.due_boundary, item.requested_kg) for item in model.deliveries] == [
            (16, 40), (33, 110), (96, 20),
        ]
        assert all(item.source == "scenario_schedule" for item in model.deliveries)
    finally:
        environment.close()


def test_empty_deliveries_do_not_create_implicit_mpc_contracts(forecast, scenario):
    scenario["faults"] = []
    scenario["deliveries"] = []
    environment, _ = create_site_simulation(forecast, controller="H", scenario=scenario)
    try:
        policy = SiteOperationalMPC(environment)
        model = policy.build_model(
            environment.cyclic_market_controller_observation,
            constraints=environment.constraints, config=environment.config,
            commitment_profile=environment.episode_config.electrolyser_commitment,
        )
        assert environment._episode.commitments == ()
        assert model.deliveries == ()
    finally:
        environment.close()


def test_delivery_larger_than_storage_is_allowed_and_settles_as_unmet(forecast, scenario):
    scenario["deliveries"] = [{"due_hour": 0.25, "kg": 100000}]
    environment, policy = create_site_simulation(forecast, controller="H", scenario=scenario)
    try:
        decision = policy.compute_command(
            environment.cyclic_market_controller_observation.plant,
            constraints=environment.constraints, config=environment.config,
            commitment_profile=environment.episode_config.electrolyser_commitment,
        )
        environment.step_process_setpoints(decision.command)
        assert environment.delivery_observation.cumulative_requested_kg == 100000
        assert environment.delivery_observation.cumulative_delivered_kg <= 3000
        assert environment.economic_metrics.unmet_kg >= 97000
    finally:
        environment.close()


def test_non_dac_fault_uses_explicit_guarded_fallback(forecast, scenario):
    scenario["faults"][0]["type"] = "methanator_outage"
    environment, _ = create_site_simulation(forecast, controller="H", scenario=scenario)
    try:
        policy = SiteOperationalMPC(environment)
        decision = policy.compute_command(
            environment.cyclic_market_controller_observation,
            constraints=environment.constraints, config=environment.config,
            commitment_profile=environment.episode_config.electrolyser_commitment,
        )
        assert decision.evidence.plan is None
        assert decision.evidence.operation()["solver_failure"]["reason"] == "unsupported_observed_state"
        environment.step_process_setpoints(decision.command)
        assert environment.physics_steps_taken == 15
        assert environment.telemetry.ch4_produced_total_kg == 0
    finally:
        environment.close()


@pytest.mark.parametrize("field,value", [("duration_days", 0), ("duration_days", 1.5), ("seed", 1.5)])
def test_invalid_authored_horizon_and_seed_are_rejected(forecast, scenario, field, value):
    invalid = deepcopy(scenario)
    invalid[field] = value
    with pytest.raises(ValueError):
        create_site_simulation(forecast, controller="H", scenario=invalid)


def test_forecast_must_cover_run_plus_full_controller_lookahead(forecast, scenario):
    forecast["hours"].pop()
    with pytest.raises(ValueError, match="48 hourly intervals"):
        create_site_simulation(forecast, controller="H", scenario=scenario)

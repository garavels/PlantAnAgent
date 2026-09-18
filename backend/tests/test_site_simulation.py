"""Real physics and profile-boundary checks for the isolated site projection."""

from datetime import datetime, timedelta, timezone

import pytest

from plant_sim import SolarArray
from plant_sim.faults import FaultSpec
from plant_backend.site_simulation import (
    SiteEnvironment,
    SiteOperationalMPC,
    SitePVProfile,
    _trajectory,
    run_site_simulation,
)


@pytest.fixture
def forecast():
    # Deliberately cross midnight in UTC; radiation defines daylight itself.
    start = datetime(2026, 9, 14, 23, tzinfo=timezone.utc)
    return {
        "id": "test-contiguous-site-forecast-v1",
        "start_time": start.isoformat(),
        "site": {
            "name": "Test coast",
            "latitude": 45.0,
            "longitude": 13.0,
            "elevation_m": 24.0,
            "timezone": "Europe/Rome",
        },
        "source": {
            "provider": "test fixture",
            "forecast_issued_at": None,
            "retrieved_at": start.isoformat(),
            "kind": "forecast",
        },
        "hours": [
            {
                "time": (start + timedelta(hours=index)).isoformat(),
                "shortwave_w_m2": 800.0 if 8 <= index % 24 < 18 else 0.0,
                "temperature_c": 20.0,
                "humidity_pct": 60.0,
                "wind_m_s": 2.0,
            }
            for index in range(264)
        ],
    }


def test_profile_is_exact_hourly_input_with_clipping_and_no_fabricated_geometry(
    forecast,
):
    forecast["hours"][0]["shortwave_w_m2"] = 1500.0
    profile = SitePVProfile.from_forecast(forecast)
    assert profile.at(0) == 1.0
    assert profile.at(3599) == 1.0
    assert profile.at(3600) == 0.0
    assert profile.at(8 * 3600) == pytest.approx(0.68)
    assert profile.start_time.hour == 23
    # A site instance must never change the frozen default solar implementation.
    assert SolarArray().clear_sky_kw(0) == 0.0
    forecast["hours"][0]["shortwave_w_m2"] = 0.0
    assert profile.at(0) == 1.0  # Imported data detached from later caller edits.


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "naive", "null", "negative"]
)
def test_profile_rejects_gaps_misalignment_or_missing_radiation(forecast, change):
    if change == "missing":
        forecast["hours"].pop()
    elif change == "duplicate":
        forecast["hours"][5]["time"] = forecast["hours"][4]["time"]
    elif change == "naive":
        forecast["start_time"] = "2026-09-14T23:00:00"
    elif change == "null":
        forecast["hours"][5]["shortwave_w_m2"] = None
    else:
        forecast["hours"][5]["shortwave_w_m2"] = -1.0
    with pytest.raises(ValueError):
        SitePVProfile.from_forecast(forecast)


def test_stress_changes_physics_only_and_static_forecast_provenance_is_preserved(
    forecast,
):
    profile = SitePVProfile.from_forecast(forecast)
    nominal = SiteEnvironment(
        profile, solar_kw=5000, battery_kwh=8000, solar_scale=1, inject_fault=True
    )
    stressed = SiteEnvironment(
        profile, solar_kw=5000, battery_kwh=8000, solar_scale=0.7, inject_fault=True
    )
    try:
        nominal.reset(seed=20260915)
        stressed.reset(seed=20260915)
        assert nominal.current_forecast.to_dict() == stressed.current_forecast.to_dict()
        assert nominal.current_forecast.to_dict()["provider_forecast_issued_at"] is None
        assert "not a calibrated" in nominal.current_forecast.to_dict()["uncertainty"]
        assert nominal._episode.scenario.initial == stressed._episode.scenario.initial
        assert nominal._episode.scenario.faults == stressed._episode.scenario.faults
        assert nominal._episode.weather.tape.at_boundary(100) == 1
        assert stressed._episode.weather.tape.at_boundary(100) == 0.7
        assert nominal.simulator.solar.clear_sky_kw(31 * 3600) == pytest.approx(3400)
        projection = _trajectory(
            profile, nominal.current_forecast, nominal.config, nominal.telemetry.time_s
        )
        assert projection.available_kw[8 * 4] == pytest.approx(3400)
        assert projection.available_kw[7 * 4] == 0
        assert sum(projection.available_kw) / 4 == pytest.approx(34000)
    finally:
        nominal.close()
        stressed.close()


def test_ten_day_energy_conservation_hourly_aggregation_and_actual_fault_response(
    forecast,
):
    result = run_site_simulation(forecast, controller="H", solar_scale=0.7)
    assert result["horizon_hours"] == 240
    assert len(result["history"]) == 241
    assert result["history"][-1]["elapsed_hours"] == 240
    expected_kwh = 10 * 10 * 5000 * 0.85 * 0.8 * 0.7
    assert result["metrics"]["solar_energy_kwh"] == pytest.approx(
        expected_kwh, abs=1e-6
    )
    rows = result["history"][1:]
    assert sum(row["solar_kw"] for row in rows) == pytest.approx(expected_kwh, abs=1e-6)
    assert sum(row["methane_kg_h"] for row in rows) == pytest.approx(
        result["metrics"]["methane_produced_kg"]
    )
    assert sum(row["curtailed_kw"] for row in rows) == pytest.approx(
        result["metrics"]["curtailed_energy_kwh"]
    )
    assert result["metrics"]["utilisation"]["dac"] == pytest.approx(
        sum(row["dac_kw"] for row in rows) / (1200 * 240)
    )
    active = [row for row in rows if row["fault_active"]]
    assert [row["elapsed_hours"] for row in active] == [109, 110, 111, 112, 113, 114]
    assert all(row["dac_kw"] == 0 for row in active)
    assert not result["history"][108]["fault_active"]
    assert not result["history"][115]["fault_active"]
    assert result["metrics"]["delivery_requested_kg"] == 6000
    assert result["metrics"]["curtailed_energy_kwh"] <= expected_kwh
    assert all(0 <= row["battery_soc"] <= 1 for row in rows)


def test_mpc_executes_profile_plan_rows_through_common_minute_dispatcher(forecast):
    pytest.importorskip("scipy.optimize")
    environment = SiteEnvironment(
        SitePVProfile.from_forecast(forecast),
        solar_kw=5000,
        battery_kwh=8000,
        solar_scale=1,
        inject_fault=False,
    )
    try:
        environment.reset(seed=20260915)
        controller = SiteOperationalMPC(environment)
        for index in range(2):
            decision = controller.compute_command(
                environment.cyclic_market_controller_observation,
                constraints=environment.constraints,
                config=environment.config,
                commitment_profile=environment.episode_config.electrolyser_commitment,
            )
            assert decision.command.decision_index == index
            environment.step_process_setpoints(decision.command)
            assert len(environment.last_process_setpoint_execution.applications) == 15
        assert controller.solve_count == 1
        assert controller.to_dict()["replanning_interval_hours"] == 3
        assert environment.physics_steps_taken == 30
        # An injected outage becomes visible only after the next physical step.
        environment.simulator.inject_fault(
            FaultSpec(type="dac_outage", duration_s=3600)
        )
        for _ in range(2):
            decision = controller.compute_command(
                environment.cyclic_market_controller_observation,
                constraints=environment.constraints,
                config=environment.config,
                commitment_profile=environment.episode_config.electrolyser_commitment,
            )
            environment.step_process_setpoints(decision.command)
        assert controller.solve_count == 2
        offline_model = controller.build_model(
            environment.cyclic_market_controller_observation,
            constraints=environment.constraints,
            config=environment.config,
            commitment_profile=environment.episode_config.electrolyser_commitment,
        )
        # Every future DAC allocation is constrained to zero, not just the
        # first interval. The fault's actual repair schedule is never an input.
        individually_zero = {
            row.coefficients[0][0]
            for row in offline_model.physical_model.rows
            if row.upper == 0
            and len(row.coefficients) == 1
            and row.coefficients[0][1] == 1
        }
        assert set(offline_model.group("solar_to_dac_kw")) <= individually_zero
    finally:
        environment.close()

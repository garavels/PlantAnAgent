"""Authored scenarios run, restart and replay through the actual shared workspace."""

from copy import deepcopy

import pytest

from plant_backend import demo_recordings, site_weather, v2_demo
from plant_backend.scenario_authoring import defaults


def setup_scenario():
    spec = defaults()
    spec.update(
        name="One-day delivery test",
        duration_days=1,
        sunlight=[
            {"start_hour": 0, "end_hour": 1, "scale": 0.3, "forecast_known": True},
            {"start_hour": 1, "end_hour": 2, "scale": 0.2, "forecast_known": False},
        ],
        deliveries=[{"due_hour": 0.25, "kg": 25.0}, {"due_hour": 0.75, "kg": 40.0}],
    )
    forecast = site_weather.load_example("trieste")
    for hour in forecast["hours"]:
        hour["shortwave_w_m2"] = 600.0
    session = v2_demo.V2DemoSession()
    initial = session.activate_site(forecast, controller_id="H", scenario=spec)
    return session, initial, spec


def advance(session, count=4):
    current = session.snapshot()
    return session.step(v2_demo.StepRequest(
        expected_run_id=current["run_id"], expected_revision=current["revision"], decisions=count,
    ))


def test_shared_scenario_projects_real_initial_stocks_weather_and_custom_deadlines():
    session, initial, spec = setup_scenario()
    assert initial["environment"]["horizon_hours"] == 24
    assert initial["frame"]["clock"]["decision_count"] == 96
    assert initial["site_parameters"]["scenario"] == spec
    assert "2 deliveries totaling 65 kg" in initial["events"][0]["detail"]
    after = advance(session)
    assert after["status"] == "ready", after["error"]
    assert after["frame"]["delivery"]["delivered_kg"] == pytest.approx(65)
    assert [entry["status"] for entry in after["frame"]["delivery"]["commitments"]] == ["met", "met"]
    weather = after["frame"]["weather"]
    assert weather["site_sample"]["forecast_solar_kw"] == pytest.approx(765)
    assert weather["available_kw"] == pytest.approx(765)
    assert weather["site_sample"]["shortwave_w_m2"] == 600
    surprise = advance(session, 1)["frame"]["weather"]
    assert surprise["available_kw"] == pytest.approx(510)
    assert surprise["site_sample"]["forecast_solar_kw"] == pytest.approx(2550)
    assert surprise["scenario_forecast_scale"] == 1
    assert surprise["stress_factor"] == 0.2


def test_restart_retains_scenario_and_recordings_validate_variable_horizons():
    session, initial, spec = setup_scenario()
    advance(session)
    record = session.recording()
    assert demo_recordings.validate(record) == record
    assert demo_recordings.summarize(record, "live")["title"] == spec["name"]
    restarted = session.reset("H")
    assert restarted["run_id"] != initial["run_id"]
    assert restarted["frame"] == initial["frame"]
    assert restarted["site_parameters"]["scenario"] == spec
    for change in ("duration", "weather", "delivery"):
        altered = deepcopy(record)
        if change == "duration":
            altered["snapshot"]["environment"]["horizon_hours"] = 240
        elif change == "weather":
            altered["frames"][1]["weather"]["site_sample"]["forecast_solar_kw"] = 2550
        else:
            altered["frames"][1]["delivery"]["commitments"][0]["requested_kg"] = 99
        altered["sha256"] = demo_recordings.digest({k: v for k, v in altered.items() if k != "sha256"})
        with pytest.raises(ValueError):
            demo_recordings.validate(altered)


def test_scenario_completes_at_its_requested_horizon():
    session, _, _ = setup_scenario()
    for _ in range(24):
        state = advance(session)
    assert state["status"] == "complete", state["error"]
    assert state["frame"]["clock"]["elapsed_hours"] == 24
    assert len(state["frame"]["cycle_settlements"]) == 1
    assert len(session.recording()["frames"]) == 97
    demo_recordings.validate(session.recording())

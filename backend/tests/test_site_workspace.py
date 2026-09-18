"""A selected site drives the shared plant, weather, planning and replay state."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from plant_backend import demo_recordings, site_api, site_weather, v2_demo
from plant_backend.site_simulation import create_site_simulation


@pytest.fixture
def forecast():
    start = datetime(2026, 9, 14, 23, tzinfo=timezone.utc)
    return {
        "id": "a" * 64,
        "start_time": start.isoformat(),
        "site": {
            "name": "Test coast",
            "latitude": 45,
            "longitude": 13,
            "elevation_m": 12,
            "timezone": "Europe/Rome",
        },
        "source": {
            "provider": "test fixture",
            "forecast_issued_at": "2026-09-14T12:00:00Z",
            "retrieved_at": "2026-09-14T13:00:00Z",
            "kind": "forecast",
        },
        "hours": [
            {
                "time": (start + timedelta(hours=i)).isoformat(),
                "shortwave_w_m2": 400 if i % 2 == 0 else 800,
                "temperature_c": 10 + i % 24,
                "humidity_pct": 65,
                "wind_m_s": 3,
                "cloud_cover_pct": 0 if i % 2 == 0 else 80,
                "cloud_cover_low_pct": 0 if i % 2 == 0 else None,
                "cloud_cover_mid_pct": 0 if i % 2 == 0 else 30,
                "cloud_cover_high_pct": 0 if i % 2 == 0 else 50,
            }
            for i in range(264)
        ],
    }


@pytest.fixture
def session():
    return v2_demo.V2DemoSession()


@pytest.fixture
def client(monkeypatch, session, forecast):
    app = FastAPI()
    app.include_router(site_api.router)
    app.include_router(v2_demo.router)
    monkeypatch.setattr(v2_demo, "session", session)
    monkeypatch.setattr(
        site_weather, "get_forecast", lambda identifier: deepcopy(forecast)
    )
    with TestClient(app) as value:
        yield value


def step(session, count=1):
    state = session.snapshot()
    return session.step(
        v2_demo.StepRequest(
            expected_run_id=state["run_id"],
            expected_revision=state["revision"],
            decisions=count,
        )
    )


def test_activate_and_step_use_real_site_physics_in_the_shared_session(
    client, session, forecast
):
    response = client.post(
        "/api/sites/activate",
        json={
            "forecast_id": forecast["id"],
            "controller": "H",
            "solar_scale": 0.7,
            "solar_kw": 5000,
            "battery_kwh": 10000,
            "inject_fault": False,
        },
    )
    assert response.status_code == 200, response.text
    initial = response.json()
    assert initial["mode"] == "site_forecast"
    assert initial["site_forecast"] == forecast
    assert initial["frame"]["clock"]["decision_count"] == 960
    assert initial["environment"]["horizon_hours"] == 240
    assert initial["environment"]["fixture_seed"] is None
    assert initial["frame"]["state"]["solar_available_kw"] == pytest.approx(1190)
    assert "Ten days" in initial["events"][0]["detail"]
    after = step(session, 4)
    assert client.get("/api/v2/demo").json() == after
    assert after["frame"]["state"]["step"] == 60
    assert after["frame"]["state"]["energy_produced_kwh"] == pytest.approx(1190)
    assert after["frame"]["weather"]["clear_sky_kw"] == pytest.approx(1700)
    assert after["frame"]["weather"]["resource_factor"] is None
    assert after["frame"]["weather"]["attenuation"] is None
    sample = after["frame"]["weather"]["site_sample"]
    assert sample["interval_start"] == forecast["hours"][0]["time"]
    assert sample["time"] == "2026-09-15T00:00:00Z"
    assert sample["temperature_c"] == 10
    assert sample["conditions_time"] == "2026-09-15T00:00:00Z"
    assert sample["cloud_cover_pct"] == 0
    assert sample["cloud_cover_low_pct"] == 0
    assert after["frame"]["forecasts"]["forecast_issued_at"] == "2026-09-14T12:00:00Z"
    assert after["frame"]["forecasts"]["issue_age_hours"] == 12
    assert after["frame"]["clock"]["local_time"] == "2026-09-15T02:00:00+02:00"
    later = step(session)
    assert later["frame"]["weather"]["site_sample"]["temperature_c"] == 11
    assert later["frame"]["weather"]["site_sample"]["conditions_time"] == "2026-09-15T01:00:00Z"
    assert later["frame"]["weather"]["site_sample"]["time"] == "2026-09-15T00:15:00Z"
    assert later["frame"]["weather"]["site_sample"]["cloud_cover_pct"] == 80
    assert later["frame"]["weather"]["site_sample"]["cloud_cover_low_pct"] is None
    assert later["frame"]["weather"]["site_sample"]["cloud_cover_mid_pct"] == 30
    assert later["frame"]["weather"]["site_sample"]["cloud_cover_high_pct"] == 50
    assert later["frame"]["state"]["solar_available_kw"] == pytest.approx(2380)
    assert later["history"][-1]["clear_sky_kw"] == pytest.approx(3400)
    assert later["frame"]["decision"]["input_forecasts"] == after["frame"]["forecasts"]


def test_site_switch_reset_and_deactivate_preserve_replay_and_default_fixture(
    client, session, forecast
):
    default = session.snapshot()
    activated = client.post(
        "/api/sites/activate", json={"forecast_id": forecast["id"], "controller": "H"}
    ).json()
    environment, policy = create_site_simulation(forecast, controller="H")
    try:
        assert activated["frame"]["state"] == environment.telemetry.to_dict()
    finally:
        environment.close()
    advanced = step(session)
    assert advanced["run_id"] != default["run_id"]
    reset = client.post("/api/v2/demo/reset", json={"controller": "H"}).json()
    assert reset["mode"] == "site_forecast"
    assert reset["site_forecast"] == forecast
    assert reset["frame"]["state"] == activated["frame"]["state"]
    assert reset["run_id"] != advanced["run_id"]
    restored = client.post("/api/sites/deactivate").json()
    assert restored["mode"] == "interactive_fixture"
    assert "site_forecast" not in restored
    assert restored["frame"]["state"] == default["frame"]["state"]
    assert restored["environment"]["fixture_seed"] == 9000034
    assert session.archives()[-1]["schema"] == "plant-site-recording-v1"


def test_site_electricity_is_zero_trade_in_the_actual_ledger(forecast, session):
    session.activate_site(forecast, controller_id="H")
    after = step(session, 4)
    assert after["frame"]["forecasts"]["price_current_eur_mwh"] == 0
    assert all(
        item["mean"] == item["p10"] == item["p90"] == 0
        for item in after["frame"]["forecasts"]["prices"]
    )
    entries = {
        item["id"]: item["value_eur"] for item in after["frame"]["economics"]["entries"]
    }
    assert entries["solar_opportunity_cost"] == 0
    assert after["frame"]["assumptions"]["tariff"]["hourly_eur_per_mwh"] == [0] * 24
    assert after["frame"]["electricity"]["mode"] == "dedicated_pv_no_export"


def test_site_exports_retain_source_timing_and_validate_independently(
    client, session, forecast
):
    session.activate_site(forecast, controller_id="H")
    step(session, 4)
    record = client.get("/api/v2/demo/recording").json()
    assert record["schema"] == "plant-site-recording-v1"
    assert demo_recordings.validate(record) == record
    assert record["snapshot"]["site_forecast"] == forecast
    assert "not measured" in record["evidence_level"]
    changed = deepcopy(record)
    changed["frames"][1]["weather"]["site_sample"]["shortwave_w_m2"] = 123
    changed["sha256"] = demo_recordings.digest(
        {k: v for k, v in changed.items() if k != "sha256"}
    )
    with pytest.raises(ValueError, match="weather context"):
        demo_recordings.validate(changed)
    changed = deepcopy(record)
    changed["schema"] = "plant-public-recording-v1"
    changed["sha256"] = demo_recordings.digest(
        {k: v for k, v in changed.items() if k != "sha256"}
    )
    with pytest.raises(ValueError, match="disposable"):
        demo_recordings.validate(changed)


def test_site_activation_rejects_bad_inputs_without_replacing_existing_run(
    client, session, forecast, monkeypatch
):
    initial = session.snapshot()
    for field, value in (
        ("controller", "PPO"),
        ("solar_kw", 0),
        ("compare_baseline", True),
    ):
        response = client.post(
            "/api/sites/activate", json={"forecast_id": forecast["id"], field: value}
        )
        assert response.status_code == 422
        assert session.snapshot() == initial
    monkeypatch.setattr(v2_demo, "_has_scipy", lambda: False)
    response = client.post(
        "/api/sites/activate", json={"forecast_id": forecast["id"], "controller": "MPC"}
    )
    assert response.status_code == 503
    assert session._env.decisions_taken == 0
    assert session.snapshot()["run_id"] == initial["run_id"]


@pytest.mark.parametrize("field,value", [
    ("cloud_cover_pct", 99),
    ("cloud_cover_low_pct", None),
    ("cloud_cover_mid_pct", 1),
    ("cloud_cover_high_pct", 50),
    ("conditions_time", "2026-09-15T00:15:00Z"),
])
def test_site_recordings_reject_changed_cloud_or_sample_time_even_with_new_checksum(
    client, session, forecast, field, value
):
    session.activate_site(forecast, controller_id="H")
    step(session, 2)
    record = client.get("/api/v2/demo/recording").json()
    assert demo_recordings.validate(record) == record
    record["frames"][1]["weather"]["site_sample"][field] = value
    record["sha256"] = demo_recordings.digest(
        {key: item for key, item in record.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="weather context"):
        demo_recordings.validate(record)


def test_legacy_site_weather_remains_unavailable_and_old_recordings_still_validate(
    client, session, forecast
):
    for hour in forecast["hours"]:
        for key in site_weather.CLOUD_COVER_FIELDS:
            del hour[key]
    initial = session.activate_site(forecast, controller_id="H")
    assert all(
        initial["frame"]["weather"]["site_sample"][key] is None
        for key in site_weather.CLOUD_COVER_FIELDS
    )
    step(session)
    record = client.get("/api/v2/demo/recording").json()
    for frame in [*record["frames"], record["snapshot"]["frame"]]:
        sample = frame["weather"]["site_sample"]
        sample.pop("conditions_time", None)
        for key in site_weather.CLOUD_COVER_FIELDS:
            sample.pop(key, None)
    record["sha256"] = demo_recordings.digest(
        {key: item for key, item in record.items() if key != "sha256"}
    )
    assert demo_recordings.validate(record) == record


def test_cached_site_mpc_keeps_original_plan_time_and_full_operation_fields(
    forecast, session
):
    pytest.importorskip("scipy.optimize")
    session.activate_site(forecast, controller_id="MPC", inject_fault=False)
    first = step(session)
    assert first["status"] == "ready", first.get("error")
    later = step(session)
    assert later["status"] == "ready", later.get("error")
    operation = later["frame"]["decision"]["operation"]
    assert operation["planned_at_hours"] == 0
    assert operation["executing_at_hours"] == 0.25
    assert {
        "resources",
        "deliveries",
        "solver_failure",
        "horizon_hours",
    } <= operation.keys()
    if later["frame"]["plan"] is not None:
        assert later["frame"]["plan"]["issued_at_hours"] == 0
    assert (
        later["frame"]["forecasts"]["forecast_issued_at"]
        == forecast["source"]["forecast_issued_at"]
    )

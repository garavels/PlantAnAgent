"""Site HTTP validation and bounded worker lifecycle, without network or solvers."""

from __future__ import annotations

from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from plant_backend import site_api

FORECAST_ID = "a" * 64


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch):
    manager = site_api.SiteJobs()
    monkeypatch.setattr(site_api, "jobs", manager)
    data = {"id": FORECAST_ID, "site": {"name": "Trieste"}, "hours": []}

    def read(identity: str) -> dict[str, Any]:
        if identity != FORECAST_ID:
            raise site_api.site_weather.ForecastUnavailableError("Unknown forecast; load it again.")
        return data

    monkeypatch.setattr(site_api.site_weather, "get_forecast", read)
    monkeypatch.setattr(site_api.site_weather, "fetch_forecast", lambda **kwargs: {**data, "request": kwargs})
    app = FastAPI()
    app.include_router(site_api.router)
    with TestClient(app) as client:
        yield client, manager
    manager.close()


def test_forecast_bounds_and_unknown_fields(api) -> None:
    client, _ = api
    assert client.post("/api/sites/forecast", json={"latitude": 45.65, "longitude": 13.77}).status_code == 200
    for change in ({"latitude": 20}, {"longitude": 90}, {"latitude": True}, {"latitude": "45"}, {"url": "https://example.com"}):
        payload = {"latitude": 45.65, "longitude": 13.77, **change}
        assert client.post("/api/sites/forecast", json=payload).status_code == 422


def test_run_rejects_invalid_capacity_and_missing_forecast(api) -> None:
    client, _ = api
    for change in ({"solar_kw": 0}, {"battery_kwh": -1}, {"solar_kw": True}, {"controller": "PPO"}, {"solar_scale": 1.1}, {"path": "/tmp/data"}):
        assert client.post("/api/sites/runs", json={"forecast_id": FORECAST_ID, **change}).status_code == 422
    assert client.post("/api/sites/runs", json={"forecast_id": "b" * 64}).status_code == 404
    assert client.get(f'/api/sites/forecasts/{"b" * 64}/download').status_code == 404


def test_job_completion_download_and_copy_isolation(api, monkeypatch: pytest.MonkeyPatch) -> None:
    client, manager = api
    monkeypatch.setattr(site_api, "_compute", lambda request, data, progress: {"forecast_id": data["id"], "methane_kg": 42})
    response = client.post("/api/sites/runs", json={"forecast_id": FORECAST_ID})
    assert response.status_code == 202
    identity = response.json()["id"]
    manager._executor.submit(lambda: None).result(timeout=5)
    done = client.get(f"/api/sites/runs/{identity}").json()
    assert done["status"] == "complete"
    assert done["progress"] == 1.0
    done["result"]["methane_kg"] = -1
    assert manager.get(identity)["result"]["methane_kg"] == 42
    exported = client.get(f"/api/sites/runs/{identity}/download")
    assert exported.status_code == 200
    assert exported.json()["forecast_id"] == FORECAST_ID
    assert "attachment" in exported.headers["content-disposition"]
    raw = client.get(f"/api/sites/forecasts/{FORECAST_ID}/download")
    assert raw.status_code == 200
    assert raw.json()["id"] == FORECAST_ID


def test_worker_failure_is_visible_and_never_falls_back(api, monkeypatch: pytest.MonkeyPatch) -> None:
    client, manager = api
    calls = []

    def fail(request, data, progress):
        calls.append(request.controller)
        raise RuntimeError("internal solver path should not be shown")

    monkeypatch.setattr(site_api, "_compute", fail)
    identity = client.post("/api/sites/runs", json={"forecast_id": FORECAST_ID}).json()["id"]
    manager._executor.submit(lambda: None).result(timeout=5)
    result = client.get(f"/api/sites/runs/{identity}").json()
    assert result["status"] == "failed"
    assert result["result"] is None
    assert calls == ["MPC"]
    assert "internal solver path" not in result["error"]
    assert client.get(f"/api/sites/runs/{identity}/download").status_code == 409


def test_at_most_two_jobs_pending(api, monkeypatch: pytest.MonkeyPatch) -> None:
    client, manager = api
    started, release = Event(), Event()

    def blocking(request, data, progress):
        started.set()
        assert release.wait(5)
        return {}

    monkeypatch.setattr(site_api, "_compute", blocking)
    try:
        first = client.post("/api/sites/runs", json={"forecast_id": FORECAST_ID})
        assert first.status_code == 202
        assert started.wait(5)
        assert client.post("/api/sites/runs", json={"forecast_id": FORECAST_ID}).status_code == 202
        assert client.post("/api/sites/runs", json={"forecast_id": FORECAST_ID}).status_code == 429
    finally:
        release.set()
        manager._executor.submit(lambda: None).result(timeout=5)


def test_provider_outage_is_actionable(api, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = api

    def unavailable(**kwargs):
        raise site_api.site_weather.ForecastUnavailableError("Weather service unavailable; choose a saved example.")

    monkeypatch.setattr(site_api.site_weather, "fetch_forecast", unavailable)
    response = client.post("/api/sites/forecast", json={"latitude": 45, "longitude": 13})
    assert response.status_code == 503
    assert "saved example" in response.json()["detail"]


def test_comparison_preserves_site_capacity_and_fault_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    progress = []

    def simulate(data, **options):
        callback = options.pop("progress_callback")
        calls.append((data, options))
        callback(1.0)
        return {"controller": options["controller"]}

    monkeypatch.setitem(__import__("sys").modules, "plant_backend.site_simulation", SimpleNamespace(run_site_simulation=simulate))
    data = {"id": FORECAST_ID}
    request = site_api.SiteRunRequest(
        forecast_id=FORECAST_ID, solar_kw=7000, battery_kwh=12000,
        solar_scale=0.7, inject_fault=True, controller="MPC",
    )
    result = site_api._compute(request, data, progress.append)
    assert result["controller"] == "MPC"
    assert result["baseline"]["controller"] == "H"
    assert [call[0] for call in calls] == [data, data]
    assert {k: v for k, v in calls[0][1].items() if k != "controller"} == {
        k: v for k, v in calls[1][1].items() if k != "controller"
    }
    assert progress == [0.8, 1.0]
    calls.clear()
    request.controller = "H"
    assert site_api._compute(request, data, progress.append)["baseline"] is None
    assert len(calls) == 1

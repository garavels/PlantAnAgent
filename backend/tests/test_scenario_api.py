"""Scenario previews freeze reviewed inputs and never alter the active twin."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from plant_backend import scenario_api as api_module
from plant_backend import scenario_authoring as authoring


@pytest.fixture
def api(monkeypatch):
    import plant_backend

    forecast = {
        "id": "a" * 64,
        "site": {"name": "Trieste, Italy", "latitude": 45.6495, "longitude": 13.7768},
        "start_time": "2026-09-01T00:00:00Z",
        "source": {"kind": "cached_forecast", "retrieved_at": "2026-08-31T12:00:00Z"},
        "hours": [{"shortwave_w_m2": 500.0}] * 264,
    }
    calls = {"weather": [], "activated": []}

    def example(identity):
        calls["weather"].append(identity)
        return forecast

    def live(**options):
        calls["weather"].append(options)
        return forecast

    def activate(weather, **options):
        calls["activated"].append((weather, options))
        return {"run_id": "scenario-run", "status": "ready"}

    monkeypatch.setattr(api_module, "previews", api_module.ScenarioPreviews())
    monkeypatch.setattr(api_module.site_weather, "load_example", example)
    monkeypatch.setattr(api_module.site_weather, "fetch_forecast", live)
    monkeypatch.setattr(plant_backend, "v2_demo", SimpleNamespace(session=SimpleNamespace(activate_site=activate)), raising=False)
    app = FastAPI()
    app.include_router(api_module.router)
    with TestClient(app) as client:
        yield client, forecast, calls


def preview(client, **changes):
    return client.post("/api/scenario-studio/preview", json={"scenario": {**authoring.defaults(), **changes}})


def test_capabilities_work_without_credentials_and_only_expose_configuration(api, monkeypatch) -> None:
    client, _, calls = api
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-only-do-not-expose")
    monkeypatch.delenv("PLANT_SCENARIO_MODEL", raising=False)
    monkeypatch.setenv("PLANT_AUTHOR_MODEL", "legacy-direct-openai-model")
    response = client.get("/api/scenario-studio/capabilities")
    assert response.status_code == 200
    body = response.json()
    assert body["llm_available"] is False
    assert body["model"] == "openai/gpt-4.1-mini"
    assert "openai-only-do-not-expose" not in str(body)
    assert body["defaults"]["duration_days"] == 7
    assert len(body["examples"]) == 3
    assert calls == {"weather": [], "activated": []}
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-do-not-expose")
    body = client.get("/api/scenario-studio/capabilities").json()
    assert body["llm_available"] is True
    assert "router-do-not-expose" not in str(body)
    assert "openai-only-do-not-expose" not in str(body)
    monkeypatch.setenv("PLANT_SCENARIO_MODEL", "provider/scenario-model")
    body = client.get("/api/scenario-studio/capabilities").json()
    assert body["model"] == "provider/scenario-model"


def test_missing_key_generation_is_actionable_but_manual_preview_still_works(api, monkeypatch) -> None:
    client, _, calls = api
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-only-must-not-enable-studio")
    response = client.post("/api/scenario-studio/generate", json={"prompt": "A week"})
    assert response.status_code == 503
    assert "OPENROUTER_API_KEY" in response.json()["detail"]
    assert "openai-only-must-not-enable-studio" not in response.json()["detail"]
    assert "manually" in response.json()["detail"]
    assert preview(client).status_code == 200
    assert calls["activated"] == []


def test_generate_returns_draft_and_unsupported_explanations_without_weather_or_activation(api, monkeypatch) -> None:
    client, _, calls = api

    async def generate(prompt):
        return authoring.build({**authoring.defaults(), "unsupported": ["Temperature damage is not modeled."], "notes": ["Assumed seven days."]})

    monkeypatch.setattr(authoring, "author", generate)
    response = client.post("/api/scenario-studio/generate", json={"prompt": "Make it cold"})
    assert response.status_code == 200
    body = response.json()
    assert body["scenario"]["unsupported"]
    assert "Assumed seven days." in body["warnings"]
    assert any("Unsupported request" in warning for warning in body["warnings"])
    assert calls == {"weather": [], "activated": []}


def test_provider_unavailable_generation_is_reported_as_503_without_mutation(api, monkeypatch) -> None:
    client, _, calls = api

    async def generate(prompt):
        raise authoring.AuthoringUnavailableError(
            "OpenRouter could not generate a scenario. Try again or edit the scenario manually."
        )

    monkeypatch.setattr(authoring, "author", generate)
    response = client.post("/api/scenario-studio/generate", json={"prompt": "A week"})
    assert response.status_code == 503
    assert "OpenRouter" in response.json()["detail"]
    assert "manually" in response.json()["detail"]
    assert calls == {"weather": [], "activated": []}


@pytest.mark.parametrize("payload", [{"prompt": ""}, {"prompt": " "}, {"prompt": True}, {"prompt": "A week", "model": "arbitrary"}])
def test_generation_rejects_empty_or_unexpected_fields(api, payload) -> None:
    client, _, _ = api
    assert client.post("/api/scenario-studio/generate", json=payload).status_code == 422


def test_preview_is_nonmutating_and_reports_real_frozen_dates_and_totals(api) -> None:
    client, _, calls = api
    response = preview(client)
    assert response.status_code == 200
    result = response.json()
    assert result["summary"] == {
        "duration_hours": 168, "start_time": "2026-09-01T00:00:00Z",
        "end_time": "2026-09-08T00:00:00Z", "total_delivery_kg": 4200.0, "delivery_count": 28,
    }
    assert "hours" not in result["forecast"]
    assert calls == {"weather": ["trieste"], "activated": []}


def test_activation_uses_only_frozen_reviewed_inputs_even_after_weather_changes(api) -> None:
    client, forecast, calls = api
    result = preview(client).json()
    forecast["hours"][0]["shortwave_w_m2"] = 1
    forecast["source"]["retrieved_at"] = "2030-01-01T00:00:00Z"
    result["scenario"]["initial"]["h2_kg"] = 999
    activation = client.post("/api/scenario-studio/activate", json={"preview_id": result["preview_id"], "controller": "H"})
    assert activation.status_code == 200
    assert activation.json()["run_id"] == "scenario-run"
    weather, options = calls["activated"][0]
    assert weather["hours"][0]["shortwave_w_m2"] == 500
    assert weather["source"]["retrieved_at"] == "2026-08-31T12:00:00Z"
    assert options["scenario"]["initial"]["h2_kg"] == 300
    assert options["solar_kw"] == 5000
    assert options["battery_kwh"] == 8000
    assert options["controller_id"] == "H"
    assert options["inject_fault"] is False
    assert options["solar_scale"] == 1
    assert calls["weather"] == ["trieste"]


def test_preview_refuses_unsupported_or_invalid_scenarios_before_fetching(api) -> None:
    client, _, calls = api
    assert preview(client, unsupported=["Grid electricity trading is unavailable."]).status_code == 422
    assert preview(client, duration_days=11).status_code == 422
    assert preview(client, solar_kw=True).status_code == 422
    assert preview(client, arbitrary="untrusted").status_code == 422
    assert calls == {"weather": [], "activated": []}


def test_live_preview_uses_validated_coordinates_and_clear_provider_failure(api, monkeypatch) -> None:
    client, _, calls = api
    live = {"name": "Another site", "latitude": 48.0, "longitude": 12.0, "source": "live", "example_id": None}
    assert preview(client, site=live).status_code == 200
    assert calls["weather"] == [{"name": "Another site", "latitude": 48.0, "longitude": 12.0}]

    def unavailable(**options):
        raise api_module.site_weather.ForecastUnavailableError("Weather unavailable; choose a saved example.")

    monkeypatch.setattr(api_module.site_weather, "fetch_forecast", unavailable)
    response = preview(client, site=live)
    assert response.status_code == 503
    assert "saved example" in response.json()["detail"]
    assert calls["activated"] == []


def test_insufficient_weather_is_rejected_before_preview_or_activation(api) -> None:
    client, forecast, calls = api
    forecast["hours"] = forecast["hours"][:168]
    assert preview(client).status_code == 422
    assert calls["activated"] == []


def test_activation_rejects_unknown_id_tampering_and_unsupported_controller(api) -> None:
    client, _, calls = api
    assert client.post("/api/scenario-studio/activate", json={"preview_id": "a" * 32, "controller": "H"}).status_code == 404
    identity = preview(client).json()["preview_id"]
    for extra in ({"controller": "PPO"}, {"scenario": authoring.defaults()}, {"forecast_id": "b" * 64}):
        response = client.post("/api/scenario-studio/activate", json={"preview_id": identity, "controller": "H", **extra})
        assert response.status_code == 422
    assert calls["activated"] == []


def test_preview_store_is_bounded_and_copy_isolated(api) -> None:
    client, _, _ = api
    first = preview(client).json()["preview_id"]
    copied = api_module.previews.get(first)
    copied["scenario"]["solar_kw"] = 999
    assert api_module.previews.get(first)["scenario"]["solar_kw"] == 5000
    for _ in range(16):
        assert preview(client).status_code == 200
    response = client.post("/api/scenario-studio/activate", json={"preview_id": first, "controller": "H"})
    assert response.status_code == 404
    assert "Preview" in response.json()["detail"]

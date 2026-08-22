"""Natural language -> scenario: the validation, and the path a draft then takes.

The model itself is not exercised here (that needs credentials and is not
deterministic); what is exercised is everything the backend does *around* it,
which is where the guarantees live: only supported faults, only supported
ranges, and a saved draft that runs down the existing scenario pipeline.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from plant_sim import SENSOR_CHANNELS
from plant_sim.faults import CATALOGUE as FAULT_CATALOGUE
from plant_sim.scenarios import SCENARIOS

from plant_backend import app as app_module
from plant_backend import authoring
from plant_backend.agent import RuleEngine, SupervisorAgent
from plant_backend.orchestration import PlantRuntime


@pytest.fixture
def client() -> Iterator[TestClient]:
    app_module.runtime = PlantRuntime()
    app_module.runtime.supervisor = SupervisorAgent(app_module.runtime, RuleEngine())
    app_module.clock.runtime = app_module.runtime
    with TestClient(app_module.app) as test_client:
        yield test_client


def draft(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": "Solar collapse",
        "description": "Generation halves at noon while the DAC degrades.",
        "start_hour": 6.0,
        "initial": {"battery_soc": 0.5, "h2_kg": None, "co2_kg": None, "ch4_kg": None},
        "faults": [
            {
                "type": "solar_drop",
                "severity": 0.5,
                "start_hour": 2.0,
                "duration_hours": 3.0,
                "channel": None,
            },
        ],
        "notes": [],
    }
    payload.update(overrides)
    return payload


# --- the schema the model is held to ---------------------------------------
def test_schema_only_offers_what_the_simulator_supports() -> None:
    fault = authoring.schema()["properties"]["faults"]["items"]
    assert fault["properties"]["type"]["enum"] == list(FAULT_CATALOGUE)
    assert fault["additionalProperties"] is False
    channel = fault["properties"]["channel"]["anyOf"][0]
    assert channel["enum"] == list(SENSOR_CHANNELS)
    assert "code" not in authoring.schema()["properties"]


def test_instructions_carry_the_catalogue_and_the_tank_sizes() -> None:
    text = authoring.instructions()
    for fault_type in FAULT_CATALOGUE:
        assert fault_type in text
    assert "6000" in text.replace(" ", "") or "6,000" in text


# --- validation -------------------------------------------------------------
def test_a_valid_draft_becomes_a_scenario() -> None:
    scenario, warnings = authoring.build(draft())
    assert warnings == []
    assert scenario.id not in SCENARIOS
    assert scenario.start_hour == 6.0
    assert scenario.initial.battery_soc == 0.5
    (fault,) = scenario.faults
    assert (fault.type, fault.severity) == ("solar_drop", 0.5)
    assert (fault.start_s, fault.duration_s) == (7200.0, 10800.0)


def test_unsupported_faults_are_refused_with_the_supported_list() -> None:
    payload = draft(faults=[{"type": "turbine_trip", "severity": 0.5, "start_hour": 1.0}])
    with pytest.raises(authoring.ScenarioValidationError) as refusal:
        authoring.build(payload)
    assert "solar_drop" in str(refusal.value)


def test_out_of_range_numbers_are_clamped_and_reported() -> None:
    payload = draft(
        start_hour=40.0,
        initial={"battery_soc": 4.0, "h2_kg": -10.0, "co2_kg": 10_000.0, "ch4_kg": None},
        faults=[
            {
                "type": "electrolyser_degradation",
                "severity": 2.0,
                "start_hour": 100.0,
                "duration_hours": 1.0,
                "channel": "h2_level_kg",
            },
        ],
    )
    scenario, warnings = authoring.build(payload)
    assert scenario.start_hour == 24.0
    assert scenario.initial.battery_soc == 1.0
    assert scenario.initial.h2_kg == 0.0
    assert scenario.initial.co2_kg == 6_000.0
    (fault,) = scenario.faults
    assert fault.start_s == authoring.MAX_START_H * 3600.0
    assert fault.severity <= 0.95
    # a channel on a process fault is meaningless, and is dropped
    assert fault.channel is None
    assert len(warnings) >= 4


def test_sensor_faults_keep_a_measured_channel() -> None:
    payload = draft(
        faults=[
            {
                "type": "sensor_drift",
                "severity": 0.2,
                "start_hour": 1.0,
                "duration_hours": None,
                "channel": "reactor_pressure",
            },
        ],
    )
    scenario, warnings = authoring.build(payload)
    (fault,) = scenario.faults
    assert fault.channel in SENSOR_CHANNELS
    assert any("reactor_pressure" in warning for warning in warnings)


def test_too_many_faults_are_trimmed() -> None:
    one = {
        "type": "dac_degradation",
        "severity": 0.3,
        "start_hour": 1.0,
        "duration_hours": None,
        "channel": None,
    }
    scenario, warnings = authoring.build(draft(faults=[dict(one)] * (authoring.MAX_FAULTS + 3)))
    assert len(scenario.faults) == authoring.MAX_FAULTS
    assert any("kept the first" in warning for warning in warnings)


def test_model_notes_reach_the_operator() -> None:
    _, warnings = authoring.build(draft(notes=["mapped 'turbine trip' to methanator_outage"]))
    assert warnings == ["mapped 'turbine trip' to methanator_outage"]


def test_generated_ids_never_collide() -> None:
    first, _ = authoring.build(draft())
    second, _ = authoring.build(draft(), taken=(first.id,))
    assert first.id != second.id


# --- the API ----------------------------------------------------------------
def test_a_saved_draft_runs_like_a_predefined_scenario(client: TestClient) -> None:
    saved = client.post("/api/scenarios/save", json={"scenario": draft()}).json()
    scenario_id = saved["scenario_id"]
    assert scenario_id in {entry["id"] for entry in saved["scenarios"]}
    assert [entry for entry in saved["scenarios"] if entry["saved"]][0]["id"] == scenario_id

    payload = client.post("/api/scenarios", json={"scenario": scenario_id}).json()
    assert payload["scenario"] == scenario_id
    assert payload["faults"], "the saved scenario should have scheduled its faults"

    catalogue = client.get("/api/scenarios").json()
    assert scenario_id in {entry["id"] for entry in catalogue["scenarios"]}


def test_an_invalid_draft_is_refused_by_the_api(client: TestClient) -> None:
    payload = draft(faults=[{"type": "sabotage", "severity": 1.0, "start_hour": 0.0}])
    response = client.post("/api/scenarios/save", json={"scenario": payload})
    assert response.status_code == 422
    assert "sabotage" in response.json()["detail"]


def test_generation_without_credentials_says_so(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(authoring.API_KEY_ENV, raising=False)
    response = client.post("/api/scenarios/generate", json={"prompt": "break the electrolyser"})
    assert response.status_code == 422
    assert authoring.API_KEY_ENV in response.json()["detail"]


def test_an_empty_prompt_never_reaches_the_model(client: TestClient) -> None:
    assert client.post("/api/scenarios/generate", json={"prompt": ""}).status_code == 422

"""The frontend-facing contract: everything the control room needs, and no more."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from plant_backend import app as app_module
from plant_backend.orchestration import PlantRuntime


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A fresh plant per test; the module-level runtime is swapped out."""
    app_module.runtime = PlantRuntime()
    app_module.clock.runtime = app_module.runtime
    with TestClient(app_module.app) as test_client:
        yield test_client


def state_of(client: TestClient) -> dict[str, object]:
    return client.get("/api/state").json()["state"]


def test_state_is_what_the_simulator_reports(client: TestClient) -> None:
    payload = client.get("/api/state").json()
    assert payload["state"] == app_module.runtime.session.state
    assert payload["frames"][-1] == payload["state"]
    assert payload["replace"] is True
    assert payload["config"]["dt_s"] > 0
    assert payload["control"]["mode"] == "auto"
    assert payload["control"]["controller"]["id"] == "heuristic"
    assert "plant_status" in payload["state"]


def test_play_pause_and_step_run_through_the_backend(client: TestClient) -> None:
    assert client.post("/api/simulation/play").json()["control"]["playing"] is True
    assert client.post("/api/simulation/pause").json()["control"]["playing"] is False

    before = state_of(client)["step"]
    payload = client.post("/api/simulation/step", json={"repeat": 5}).json()
    assert len(payload["frames"]) == 5
    assert payload["state"]["step"] == before + 5  # type: ignore[operator]


def test_speed_is_owned_by_the_backend(client: TestClient) -> None:
    assert client.post("/api/simulation/speed", json={"speed": 16}).json()["control"]["speed"] == 16
    assert client.post("/api/simulation/speed", json={"speed": 0}).status_code == 422


def test_reset_returns_the_plant_and_the_loop_to_the_start(client: TestClient) -> None:
    client.post("/api/simulation/play")
    client.post("/api/simulation/step", json={"repeat": 30})
    client.post("/api/faults", json={"type": "dac_outage"})

    payload = client.post("/api/simulation/reset").json()

    assert payload["state"]["step"] == 0
    assert payload["control"]["playing"] is False
    assert payload["faults"] == []
    assert len(payload["frames"]) == 1


def test_scenarios_load_through_the_backend(client: TestClient) -> None:
    catalogue = client.get("/api/scenarios").json()
    assert {"clear_day", "dac_failure"} <= {item["id"] for item in catalogue["scenarios"]}

    payload = client.post("/api/scenarios", json={"scenario": "dac_failure"}).json()
    assert payload["scenario"] == "dac_failure"
    assert payload["state"]["step"] == 0
    assert payload["faults"], "the scenario should have scheduled its faults"
    assert client.post("/api/scenarios", json={"scenario": "nope"}).status_code == 404


def test_faults_inject_and_clear_through_the_backend(client: TestClient) -> None:
    payload = client.post("/api/faults", json={"type": "electrolyser_degradation", "severity": 0.6}).json()
    fault_id = payload["fault"]["id"]
    assert payload["faults"][0]["type"] == "electrolyser_degradation"

    client.post("/api/simulation/step", json={"repeat": 1})
    assert client.delete(f"/api/faults/{fault_id}").json()["faults"] == []
    assert client.delete("/api/faults/ghost").status_code == 404

    client.post("/api/faults", json={"type": "dac_outage"})
    assert client.delete("/api/faults").json()["faults"] == []
    assert client.post("/api/faults", json={"type": "not_a_fault"}).status_code == 400


def test_events_come_from_the_simulator(client: TestClient) -> None:
    client.post("/api/faults", json={"type": "methanator_outage"})
    client.post("/api/simulation/step", json={"repeat": 5})
    events = client.get("/api/events").json()["events"]
    assert any(event["level"] != "info" for event in events)
    assert {"time_s", "level", "message", "clock"} <= set(events[0])


def test_control_mode_and_constraints_are_configured_through_the_backend(client: TestClient) -> None:
    assert client.get("/api/controller").json()["mode"] == "auto"

    payload = client.post("/api/controller/config", json={"mode": "manual"}).json()
    assert payload["control"]["mode"] == "manual"

    payload = client.post(
        "/api/controller/config",
        json={"constraints": {"electrolyser_max_kw": 500.0, "battery_min_soc": 0.25}},
    ).json()
    assert payload["control"]["constraints"]["electrolyser_max_kw"] == 500.0
    assert payload["control"]["constraints"]["battery_min_soc"] == 0.25

    assert client.post("/api/controller/config", json={"mode": "vibes"}).status_code == 400
    assert client.post("/api/controller/config", json={"constraints": {"battery_min_soc": 3}}).status_code == 400
    assert client.post("/api/controller/config", json={"controller": "rl"}).status_code == 400


def test_manual_actions_are_validated_before_the_simulator_sees_them(client: TestClient) -> None:
    client.post("/api/simulation/step", json={"repeat": 400})  # daylight
    client.post("/api/controller/config", json={"mode": "manual"})
    client.post("/api/controller/config", json={"constraints": {"electrolyser_max_kw": 250.0}})

    payload = client.post(
        "/api/simulation/step",
        json={"action": {"battery_share": 0.0, "electrolyser_share": 1.0}, "repeat": 1},
    ).json()

    assert payload["state"]["electrolyser_power_kw"] <= 250.0 + 1e-6
    assert payload["control"]["adjustments"], "the clamp should be reported to the operator"


def test_every_change_is_pushed_to_the_stream(client: TestClient) -> None:
    """What a subscriber receives is exactly what the caller was answered with.

    (The SSE route itself is exercised by hand -- Starlette's TestClient cannot
    read an endless response.)
    """
    queue = app_module.broadcast.subscribe()
    try:
        answer = client.post("/api/simulation/step", json={"repeat": 2}).json()
        pushed = queue.get_nowait()
        assert pushed == answer
        assert queue.empty()

        client.post("/api/faults", json={"type": "solar_drop"})
        assert queue.get_nowait()["faults"]
    finally:
        app_module.broadcast.unsubscribe(queue)


# --- V6: the supervisory agent -------------------------------------------
def test_agent_status_is_exposed_for_the_control_room(client: TestClient) -> None:
    payload = client.get("/api/agent").json()

    assert payload["status"] == "monitoring"
    assert payload["incident"] is None
    assert payload["runs"] == 0
    assert payload["trace"] == []


def test_investigate_runs_the_supervisor_on_demand(client: TestClient) -> None:
    client.post("/api/simulation/step", json={"repeat": 420})  # daylight
    client.post("/api/faults", json={"type": "electrolyser_degradation", "severity": 0.5})
    client.post("/api/simulation/step", json={"repeat": 30})

    payload = client.post("/api/agent/investigate").json()

    incident = payload["incident"]
    assert incident["trigger"] == "manual"
    assert incident["category"] == "electrolyser_degradation"
    assert payload["agent"]["runs"] == 1
    assert payload["control"]["constraints"]["electrolyser_max_kw"] is not None

    trace = client.get("/api/agent/trace", params={"run": incident["run"]}).json()
    assert [entry["kind"] for entry in trace["trace"]][0] == "trigger"
    assert trace["incidents"][-1]["id"] == incident["id"]


def test_the_agent_can_be_switched_off(client: TestClient) -> None:
    assert client.post("/api/agent/config", json={"enabled": False}).json()["agent"]["status"] == (
        "disabled"
    )
    assert client.post("/api/agent/config", json={"enabled": True}).json()["agent"]["status"] == (
        "monitoring"
    )

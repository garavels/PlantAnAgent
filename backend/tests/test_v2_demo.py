"""Public fixture boundary, execution fidelity and isolation for V2 visibility."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from plant_backend import v2_demo
from plant_backend.orchestration import PlantRuntime


@pytest.fixture
def demo() -> v2_demo.V2DemoSession:
    return v2_demo.V2DemoSession()


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch, demo: v2_demo.V2DemoSession
) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(v2_demo.router)
    monkeypatch.setattr(v2_demo, "session", demo)
    with TestClient(app) as connection:
        yield connection


def _step(snapshot: dict[str, Any], decisions: int = 1) -> v2_demo.StepRequest:
    return v2_demo.StepRequest(
        expected_run_id=snapshot["run_id"],
        expected_revision=snapshot["revision"],
        decisions=decisions,
    )


def _keys(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def test_session_creation_and_rejected_requests_do_not_sample_a_task(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, demo: v2_demo.V2DemoSession
) -> None:
    def forbidden() -> None:
        raise AssertionError("invalid request constructed a beta task")

    monkeypatch.setattr(v2_demo, "PlantCyclicMarketEnvV2Beta", forbidden)
    assert demo._environment is None
    assert client.get("/api/v2/demo?seed=1000000").status_code == 422
    assert (
        client.post("/api/v2/demo/reset", json={"controller": "PPO"}).status_code == 422
    )
    for forbidden_field in ("seed", "config", "checkpoint", "path", "horizon_h"):
        response = client.post(
            "/api/v2/demo/reset", json={"controller": "H", forbidden_field: "forbidden"}
        )
        assert response.status_code == 422
    for decisions in (0, 5, -1, True, 1.5, "1"):
        response = client.post(
            "/api/v2/demo/step",
            json={
                "expected_run_id": "missing",
                "expected_revision": 0,
                "decisions": decisions,
            },
        )
        assert response.status_code == 422
    assert (
        client.post(
            "/api/v2/demo/step",
            json={
                "expected_run_id": "missing",
                "expected_revision": 0,
                "decisions": 1,
                "seed": 1_000_000,
            },
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v2/demo/step",
            json={
                "expected_run_id": "missing",
                "expected_revision": True,
                "decisions": 1,
            },
        ).status_code
        == 422
    )
    assert demo._environment is None


def test_fixed_public_snapshot_is_lazy_and_never_reads_audit_manifests(
    demo: v2_demo.V2DemoSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("public demo read an audit manifest")

    monkeypatch.setattr(v2_demo.PlantCyclicMarketEnvV2Beta, "audit_manifest", forbidden)
    snapshot = demo.snapshot()
    assert snapshot["environment"]["fixture_seed"] == 9_000_034
    assert snapshot["environment"]["observation_count"] == 135
    assert snapshot["environment"]["action_count"] == 4
    assert snapshot["environment"]["horizon_hours"] == 168
    assert snapshot["frame"]["clock"]["decision_count"] == 672
    assert snapshot["controller"]["id"] == "H"
    assert snapshot["frame"]["plan"] is None
    assert snapshot["frame"]["decision"] is None
    assert len(snapshot["history"]) == 1
    assert demo.snapshot() == snapshot
    assert demo._env.decisions_taken == 0
    assert len(snapshot["frame"]["forecasts"]["prices"]) == 24
    public_keys = set(_keys(snapshot))
    assert not public_keys.intersection(
        {
            "seed",
            "audit_manifest",
            "market_episode",
            "disturbance_episode",
            "faults",
            "random_streams",
            "spawn_key",
            "minute_truth_states",
            "regime",
            "source_path",
            "checkpoint",
            "solver_vector",
            "applications",
            "stage_two_solution",
        }
    )
    assert {"mean_kw", "p10_kw", "offset_hours", "end_offset_hours"} == set(
        snapshot["frame"]["forecasts"]["solar"][0]
    )
    json.dumps(snapshot, allow_nan=False)


def test_four_decisions_use_real_minute_dispatch_and_cumulative_economics(
    demo: v2_demo.V2DemoSession,
) -> None:
    before = demo.snapshot()
    after = demo.step(_step(before, 4))
    assert after["status"] == "ready"
    assert after["revision"] == 1
    assert after["frame"]["clock"]["decision_index"] == 4
    assert after["frame"]["clock"]["elapsed_hours"] == 1
    assert after["frame"]["state"]["step"] == 60
    assert demo._env.simulator.step_index == 60
    assert len(after["history"]) == 5
    execution = demo._env.last_process_setpoint_execution
    assert execution is not None
    assert len(execution.applications) == 15
    decision = after["frame"]["decision"]
    assert decision["execution_digest"] == execution.sha256
    assert decision["command_digest"] == execution.command.sha256
    target = next(item for item in decision["targets"] if item["id"] == "electrolyser")
    assert target["requested_kw"] == execution.command.targets.electrolyser_target_kw
    assert target["applied_mean_kw"] == pytest.approx(
        sum(item.applied_electrolyser_kw for item in execution.applications) / 15
    )
    assert decision["guard_adjustments"] == sum(
        len(item.adjustments) for item in execution.applications
    )
    economics = after["frame"]["economics"]
    assert sum(item["value_eur"] for item in economics["entries"]) == pytest.approx(
        economics["net_value_eur"]
    )
    assert after["frame"]["forecasts"]["issued_at_hours"] == 1
    assert after["frame"]["plan"] is None
    assert "recovered spill" in " ".join(decision["notes"])


def test_stale_step_and_reset_cannot_move_the_new_run(client: TestClient) -> None:
    original = client.get("/api/v2/demo").json()
    body = _step(original).model_dump()
    advanced = client.post("/api/v2/demo/step", json=body)
    assert advanced.status_code == 200
    assert client.post("/api/v2/demo/step", json=body).status_code == 409
    reset = client.post("/api/v2/demo/reset", json={"controller": "H"}).json()
    assert reset["run_id"] != original["run_id"]
    assert reset["revision"] == 0
    assert reset["frame"]["state"] == original["frame"]["state"]
    assert len(reset["history"]) == 1
    assert (
        client.post(
            "/api/v2/demo/step", json=_step(advanced.json()).model_dump()
        ).status_code
        == 409
    )
    assert client.get("/api/v2/demo").json() == reset


def test_concurrent_duplicate_step_is_applied_once(demo: v2_demo.V2DemoSession) -> None:
    request = _step(demo.snapshot())

    def attempt() -> int:
        try:
            demo.step(request)
            return 200
        except HTTPException as exc:
            return exc.status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: attempt(), range(2)))
    assert sorted(outcomes) == [200, 409]
    assert demo.snapshot()["frame"]["clock"]["decision_index"] == 1


def test_unavailable_optional_controller_preserves_existing_run(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = client.get("/api/v2/demo").json()
    monkeypatch.setattr(v2_demo, "_has_scipy", lambda: False)
    assert (
        client.post("/api/v2/demo/reset", json={"controller": "MPC"}).status_code == 503
    )
    after = client.get("/api/v2/demo").json()
    assert after["run_id"] == before["run_id"]
    assert after["controller"]["id"] == "H"
    assert not after["available_controllers"][1]["available"]
    assert after["frame"] == before["frame"]


def test_unexpected_mpc_error_stops_without_fallback_or_private_exception_text(
    demo: v2_demo.V2DemoSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []

    class FailedEconomicController:
        id = "test-failed-controller"

        def reset(self) -> None:
            pass

        def compute_command(self, observation: Any, **kwargs: Any) -> None:
            calls.append(observation)
            raise RuntimeError("secret private solver report /protected/path")

    monkeypatch.setattr(v2_demo, "_has_scipy", lambda: True)
    monkeypatch.setattr(
        v2_demo, "_controller", lambda *args: FailedEconomicController()
    )
    before = demo.reset("MPC")
    after = demo.step(_step(before, 4))
    assert len(calls) == 1
    assert hasattr(calls[0], "market") and hasattr(calls[0], "cycle")
    assert after["status"] == "failed"
    assert after["controller"]["id"] == "MPC"
    assert "RuntimeError" in after["error"]
    assert "protected" not in json.dumps(after)
    assert after["frame"]["state"] == before["frame"]["state"]
    assert after["frame"]["decision"] is None
    with pytest.raises(HTTPException) as stopped:
        demo.step(_step(after))
    assert stopped.value.status_code == 409


def test_optional_operating_mpc_projects_its_actual_plan(
    demo: v2_demo.V2DemoSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("scipy")
    before = demo.reset("MPC")
    after = demo.step(_step(before))
    assert after["status"] == "ready", after["error"]
    assert after["controller"]["id"] == "MPC"
    assert after["frame"]["clock"]["decision_index"] == 1
    assert after["frame"]["state"]["step"] == 15
    plan = after["frame"]["plan"]
    assert plan["issued_at_hours"] == 0
    assert len(plan["points"]) == 96
    assert [item["offset_hours"] for item in plan["points"]] == [
        index / 4 for index in range(96)
    ]
    projected_first = plan["points"][0]
    retained_first = demo._controller.decision_events[-1].plan.steps[0]
    assert projected_first["electrolyser_kw"] == pytest.approx(
        retained_first.electrolyser_startup_kw
        + retained_first.electrolyser_productive_kw
    )
    assert projected_first["battery_charge_kw"] == retained_first.solar_to_battery_kw
    targets = {
        row["id"]: row["requested_kw"] for row in after["frame"]["decision"]["targets"]
    }
    assert targets["electrolyser"] == projected_first["electrolyser_kw"]
    assert targets["dac"] == projected_first["dac_kw"]
    assert targets["methanator"] == projected_first["methanator_kw"]
    assert targets["battery"] == projected_first["battery_charge_kw"]


def test_fixture_actions_do_not_mutate_v1_and_snapshots_are_detached(
    demo: v2_demo.V2DemoSession,
) -> None:
    legacy = PlantRuntime()
    before = legacy.payload()
    snapshot = demo.snapshot()
    stepped = demo.step(_step(snapshot))
    stepped["frame"]["state"]["health"].clear()
    stepped["history"].clear()
    assert legacy.payload() == before
    fresh = demo.snapshot()
    assert fresh["frame"]["state"]["health"]
    assert len(fresh["history"]) == 2


def test_partial_minute_failure_keeps_last_committed_boundary_usable(
    demo: v2_demo.V2DemoSession, monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    before = demo.snapshot()
    original_step = demo._env.simulator.step
    calls = 0

    def fail_after_physical_movement(action: Any) -> Any:
        nonlocal calls
        result = original_step(action)
        calls += 1
        if calls == 17:
            raise RuntimeError("partial physical interval")
        return result

    monkeypatch.setattr(demo._env.simulator, "step", fail_after_physical_movement)
    failed = demo.step(_step(before, 4))
    assert calls == 17
    assert demo._env.simulator.step_index == 17
    assert failed["status"] == "failed"
    assert failed["frame"]["state"]["step"] == 15
    assert failed["frame"]["clock"]["decision_index"] == 1
    assert len(failed["history"]) == 2
    assert failed["revision"] == 1
    assert "partial interval may have executed" in failed["error"]

    def unreadable() -> None:
        raise AssertionError("failed snapshot re-read damaged environment")

    monkeypatch.setattr(demo, "_frame", unreadable)
    assert client.get("/api/v2/demo").json() == failed
    assert client.get("/api/v2/demo").json() == failed
    with pytest.raises(HTTPException) as stopped:
        demo.step(_step(failed))
    assert stopped.value.status_code == 409
    assert calls == 17


def test_projection_failure_after_step_keeps_prior_completed_snapshot(
    demo: v2_demo.V2DemoSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = demo.snapshot()
    original_frame = demo._frame
    reads = 0

    def broken_projection() -> dict[str, Any]:
        nonlocal reads
        reads += 1
        if demo._env.decisions_taken >= 2:
            raise ValueError("unserializable controller state")
        return original_frame()

    monkeypatch.setattr(demo, "_frame", broken_projection)
    failed = demo.step(_step(before, 4))
    assert demo._env.decisions_taken == 2
    assert demo._env.simulator.step_index == 30
    assert failed["status"] == "failed"
    assert failed["frame"]["clock"]["decision_index"] == 1
    assert failed["frame"]["state"]["step"] == 15
    assert len(failed["history"]) == 2
    assert reads == 2
    assert demo.snapshot() == failed
    assert reads == 2


def test_single_fixture_day_settles_real_deliveries_and_resource_cycle(
    demo: v2_demo.V2DemoSession,
) -> None:
    """One fixed-fixture day checks exact 6h and 24h presentation boundaries."""
    snapshot = demo.snapshot()
    initial_resources = snapshot["frame"]["resources"]
    for _ in range(24):
        snapshot = demo.step(_step(snapshot, 4))
        assert snapshot["status"] == "ready"
    frame = snapshot["frame"]
    assert frame["clock"]["elapsed_hours"] == 24
    assert frame["clock"]["cycle_index"] == 2
    assert frame["clock"]["hours_to_cycle_end"] == 24
    settled = [
        row for row in frame["delivery"]["commitments"] if row["status"] != "pending"
    ]
    assert [row["due_hours"] for row in settled] == [6, 12, 18, 24]
    assert sum(row["delivered_kg"] for row in settled) == pytest.approx(
        frame["delivery"]["delivered_kg"]
    )
    assert sum(row["unmet_kg"] for row in settled) == pytest.approx(
        frame["delivery"]["unmet_kg"]
    )
    assert frame["delivery"]["requested_kg"] == 960
    assert len(frame["cycle_settlements"]) == 1
    assert frame["cycle_settlements"][0]["cycle_index"] == 1
    for initial, current in zip(initial_resources, frame["resources"], strict=True):
        assert initial["required"] == pytest.approx(current["required"])
        assert current["current"] - current["required"] == pytest.approx(
            current["margin"]
        )
    assert len(snapshot["history"]) == 97
    assert demo._history.maxlen == 673


def test_default_operating_mpc_reports_recovery_and_continues(demo):
    pytest.importorskip("scipy")
    before = demo.reset("MPC")
    after = demo.step(_step(before))
    assert after["status"] == "ready"
    assert after["error"] is None
    operation = after["frame"]["decision"]["operation"]
    assert operation["mode"] == "recovery"
    assert operation["policy_id"] == "operational-economic-mpc-v1"
    assert operation["forecast_statistic"] == "issued_mean"
    assert any(r["projected_deficit"] > 1 for r in operation["resources"])
    assert operation["fallback_controller"] is None
    assert after["frame"]["clock"]["decision_index"] == 1
    assert after["frame"]["operating_policy"] == "inventory-targets-v1"


def test_research_ebeta_is_not_an_interactive_product_option(client):
    before = client.get("/api/v2/demo").json()
    assert {c["id"] for c in before["available_controllers"]} == {"H", "MPC"}
    assert (
        client.post("/api/v2/demo/reset", json={"controller": "E-beta"}).status_code
        == 422
    )
    assert client.get("/api/v2/demo").json() == before

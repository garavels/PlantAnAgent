"""Read-only replay, public weather attribution and retained decision causality."""

from copy import deepcopy
import json

import pytest
from fastapi import HTTPException

from plant_backend import demo_recordings
from plant_backend.demo_projection import boundary_events, weather_view
from plant_backend.v2_demo import StepRequest, V2DemoSession


def step(demo, count=1):
    snap = demo.snapshot()
    return demo.step(
        StepRequest(
            expected_run_id=snap["run_id"],
            expected_revision=snap["revision"],
            decisions=count,
        )
    )


@pytest.mark.parametrize(
    "potential,available,health,status,factor",
    [
        (0, 0, "nominal", "night", None),
        (0.5, 0.2, "nominal", "night", None),
        (1000, 600, "nominal", "observed", 0.6),
        (1000, 600, "degraded", "unattributed", None),
        (1000, 0, "offline", "unattributed", None),
    ],
)
def test_clouds_never_reconstruct_night_or_equipment_losses(
    potential, available, health, status, factor
):
    view = weather_view(
        {
            "solar_potential_kw": potential,
            "solar_available_kw": available,
            "solar_curtailed_kw": 300,
            "health": {"solar": {"status": health}},
        },
        5,
    )
    assert view["status"] == status
    assert view["resource_factor"] == factor
    assert view["attenuation"] == (None if factor is None else 1 - factor)
    assert view["curtailed_kw"] == 300
    assert view["sample_end_hours"] == 5


def test_original_issue_frames_and_archives_are_detached_and_read_only():
    demo = V2DemoSession()
    initial = demo.snapshot()
    result = step(demo, 2)
    frame = demo.frame(initial["run_id"], 1)
    assert frame["decision"]["input_forecasts"] == initial["frame"]["forecasts"]
    assert frame["forecasts"]["issued_at_hours"] == 0.25
    assert len(frame["decision"]["minutes"]) == 15
    assert frame["decision"]["input_state"] == initial["frame"]["state"]
    frame["state"].clear()
    assert demo.frame(initial["run_id"], 1)["state"]
    record = demo_recordings.validate(demo.recording())
    assert len(record["frames"]) == 3
    assert demo.snapshot() == result
    demo.reset("H")
    with pytest.raises(HTTPException) as stale:
        demo.frame(initial["run_id"], 1)
    assert stale.value.status_code == 409
    archive = demo.archives()[0]
    assert archive == record
    archive["frames"].clear()
    assert len(demo.archives()[0]["frames"]) == 3


def test_failed_projection_exports_only_last_complete_boundary(monkeypatch):
    demo = V2DemoSession()
    demo.snapshot()
    original = demo._frame

    def fail():
        if demo._env.decisions_taken == 2:
            raise ValueError("projection failure")
        return original()

    monkeypatch.setattr(demo, "_frame", fail)
    result = step(demo, 4)
    assert result["status"] == "failed"
    record = demo_recordings.validate(demo.recording())
    assert len(record["frames"]) == 2
    assert record["snapshot"]["error"] == result["error"]
    assert demo._env.decisions_taken == 2


def test_recording_integrity_and_fixed_local_identity(monkeypatch, tmp_path):
    demo = V2DemoSession()
    initial = demo.snapshot()
    record = demo.recording()
    monkeypatch.setattr(demo_recordings, "RECORDING_ROOT", tmp_path)
    path = tmp_path / f"{initial['run_id']}.json"
    path.write_text(json.dumps(record))
    assert demo_recordings.read_recording(initial["run_id"]) == record
    changed = deepcopy(record)
    changed["snapshot"]["controller"]["id"] = "PPO"
    with pytest.raises(ValueError, match="checksum"):
        demo_recordings.validate(changed)
    changed = demo_recordings.seal_recording(initial, [])
    with pytest.raises(ValueError, match="boundaries"):
        demo_recordings.validate(changed)
    path.write_text("{")
    assert demo_recordings.catalog()[0]["available"] is False
    with pytest.raises(HTTPException):
        demo_recordings.read_recording("../outside")
    wrong = tmp_path / ("0" * 36 + ".json")
    wrong.write_text(json.dumps(record))
    with pytest.raises(HTTPException):
        demo_recordings.read_recording(wrong.stem)


def test_event_changes_refer_only_to_observed_boundary():
    demo = V2DemoSession()
    first = demo.snapshot()["frame"]
    later = deepcopy(first)
    later["clock"].update(decision_index=1, elapsed_hours=0.25)
    later["state"]["health"]["dac"]["status"] = "offline"
    events = boundary_events(first, later)
    assert len(events) == 1
    assert events[0]["kind"] == "warning"
    assert events[0]["unit"] == "dac"
    assert events[0]["at_hours"] == 0.25
    assert "recovery time" in events[0]["detail"]


def test_daily_inventory_labels_preserve_historical_research_meaning():
    initial = V2DemoSession().snapshot()["frame"]
    settled = deepcopy(initial)
    settled["clock"].update(decision_index=96, elapsed_hours=24)
    settled["cycle_settlements"] = [{"cycle_index": 1, "feasible": False}]
    events = boundary_events(initial, settled)
    assert (
        next(e for e in events if e["kind"] == "cycle")["title"]
        == "Day 1 inventory deficit"
    )
    del settled["operating_policy"]
    events = boundary_events(initial, settled)
    assert (
        next(e for e in events if e["kind"] == "cycle")["title"]
        == "Day 1 resource returns failed"
    )

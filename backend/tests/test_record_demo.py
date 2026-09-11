"""A rehearsal deadline must not become a recoverable controller failure."""

import json
import sys

from plant_backend import demo_recordings, operational_mpc, record_demo


def test_wall_limit_bypasses_solver_fallback_and_retains_completed_prefix(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(record_demo, "RECORDING_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["record_demo", "MPC"])
    handlers = []
    monkeypatch.setattr(
        record_demo.signal, "signal", lambda _, handler: handlers.append(handler)
    )
    monkeypatch.setattr(record_demo.signal, "alarm", lambda _: None)

    def timed_out(*args, **kwargs):
        try:
            handlers[0]()
        except Exception as exc:
            raise AssertionError(
                "deadline was caught by an ordinary failure handler"
            ) from exc

    monkeypatch.setattr(operational_mpc, "solve_economic_mpc_model", timed_out)
    record_demo.main()
    manifest = json.loads((tmp_path / "MPC.attempt.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["hours"] == 0
    assert manifest["fallback_decisions"] == 0
    assert "wall_limit" in manifest["error"]
    record = demo_recordings.validate(
        json.loads((tmp_path / f"{manifest['id']}.json").read_text())
    )
    assert len(record["frames"]) == 1
    assert record["snapshot"]["error"] == manifest["error"]

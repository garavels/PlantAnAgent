"""Fixture adapter and recording integration, never registered policy evidence."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from plant_backend import demo_recordings, publish_ppo, v2_demo
from plant_backend.rl.v2 import beta_spaces
from plant_backend.rl.v2.beta_config import CyclicMarketTaskConfigV1
from plant_backend.rl.v2.policy_setpoint_controller import (
    NormalizedSetpointPolicyControllerV1,
)
from plant_backend.rl.v2.training.config import ControllerNeutralTrainingConfigV1
from test_rl_v2_beta_controller_benchmark_verifier import _loaded, _config


@pytest.fixture
def loaded(tmp_path):
    # Fixed untrained policy is deliberately used only at this projection seam.
    config = ControllerNeutralTrainingConfigV1(
        environment_id=beta_spaces.ENVIRONMENT_ID, beta_task=CyclicMarketTaskConfigV1()
    )
    return _loaded(tmp_path, config)


def test_policy_commands_cross_the_shared_dispatcher_and_public_projection(loaded):
    demo = v2_demo.V2DemoSession()
    snapshot = demo.reset_ppo(loaded, {"fixture_only": "untrained adapter test"})
    try:
        assert snapshot["controller"]["id"] == "PPO"
        assert snapshot["controller"]["uses_forecast"] is True
        assert [c["id"] for c in snapshot["available_controllers"]] == ["H", "MPC"]
        for index in range(1, 3):
            snapshot = demo.step(
                v2_demo.StepRequest(
                    expected_run_id=snapshot["run_id"],
                    expected_revision=snapshot["revision"],
                    decisions=1,
                )
            )
            assert snapshot["status"] == "ready"
            frame = snapshot["frame"]
            assert frame["clock"]["decision_index"] == index
            assert "operating_policy" not in frame
            assert frame["plan"] is None
            assert frame["decision"]["operation"] is None
            assert len(frame["decision"]["minutes"]) == 15
            assert len(frame["decision"]["targets"]) == 4
            assert demo._env.last_process_setpoint_execution is not None
        record = demo_recordings.validate(demo.recording())
        assert len(record["frames"]) == 3
        assert demo_recordings.summarize(record, "fixture")["controller"] == "PPO"
    finally:
        demo._env.close()


def test_incompatible_policy_rejected_before_fixture_reset(tmp_path, monkeypatch):
    loaded = _loaded(tmp_path, _config())

    def forbidden(*args, **kwargs):
        pytest.fail("incompatible policy reset a fixture")

    monkeypatch.setattr(v2_demo.PlantCyclicMarketEnvV2Beta, "reset", forbidden)
    with pytest.raises(ValueError):
        v2_demo.V2DemoSession().reset_ppo(loaded, {"fixture_only": "wrong task"})


@pytest.mark.parametrize("stop", ["policy_error", "wall_limit", "custody_mutation"])
def test_single_attempt_retains_failure_without_fallback_or_retry(
    loaded, monkeypatch, tmp_path, stop
):
    from plant_backend.rl.v2.study import training_evidence

    root = tmp_path / "recordings"
    monkeypatch.setattr(demo_recordings, "RECORDING_ROOT", root)
    monkeypatch.setattr(
        demo_recordings, "RESEARCH_RECORDING_ROOT", tmp_path / "research"
    )
    identity = {"fixture_only": "synthetic admitted boundary, not real custody"}

    def recheck():
        if stop == "custody_mutation":
            raise ValueError("fixture mutation")

    admission = SimpleNamespace(
        identity=SimpleNamespace(source={"authorization_commit": "a" * 40}),
        recheck=recheck,
    )
    monkeypatch.setattr(training_evidence, "admit_training", lambda _: admission)
    monkeypatch.setattr(
        publish_ppo, "_presentation_policy", lambda _: (loaded, identity)
    )
    original = NormalizedSetpointPolicyControllerV1.compute_command
    count = 0

    def fail(self, *args, **kwargs):
        nonlocal count
        count += 1
        if count > 1:
            if stop == "wall_limit":
                raise publish_ppo.RehearsalTimeout()
            raise RuntimeError("fixture policy exception")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(NormalizedSetpointPolicyControllerV1, "compute_command", fail)
    if stop == "custody_mutation":
        with pytest.raises(ValueError, match="mutation"):
            publish_ppo.record_rehearsal(tmp_path / "report.json")
        assert not demo_recordings.catalog()
        assert (root / "PPO.rejected-recording.json").is_file()
    else:
        result = publish_ppo.record_rehearsal(tmp_path / "report.json")
        assert result["status"] == "failed"
        assert result["hours"] == 0.25
        assert result["fallback_decisions"] == 0
        record = demo_recordings.read_recording(result["id"])
        assert len(record["frames"]) == 2
    before = (root / "PPO.attempt.json").read_bytes()
    assert json.loads((root / "PPO.outcome.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError):
        publish_ppo.record_rehearsal(tmp_path / "report.json")
    assert (root / "PPO.attempt.json").read_bytes() == before
    assert count == 2


def test_attempt_markers_do_not_hide_a_saved_ppo_recording(
    loaded, monkeypatch, tmp_path
):
    monkeypatch.setattr(demo_recordings, "RECORDING_ROOT", tmp_path)
    monkeypatch.setattr(
        demo_recordings, "RESEARCH_RECORDING_ROOT", tmp_path / "research"
    )
    demo = v2_demo.V2DemoSession()
    snapshot = demo.reset_ppo(loaded, {"fixture_only": "untrained"})
    try:
        identifier = "ffffffff-ffff-ffff-ffff-ffffffffffff"
        snapshot["run_id"] = identifier
        record = demo_recordings.seal_recording(snapshot, demo.recording()["frames"])
        (tmp_path / f"{identifier}.json").write_text(json.dumps(record))
        for index in range(12):
            (tmp_path / f"PPO.{index}.attempt.json").write_text("{}")
        assert [row["id"] for row in demo_recordings.catalog()] == [identifier]
    finally:
        demo._env.close()

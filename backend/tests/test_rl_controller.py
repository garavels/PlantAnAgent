"""V7: checkpoints, the RL controller, and what the backend does without one.

The behaviour under test is mostly refusal: no checkpoint means no RL controller,
a bad checkpoint means a clear error, and a policy that throws means the heuristic
takes the plant back. Nothing here trains anything, and the only checkpoints
involved are `fixed` mocks -- which the code is required to label as such.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from plant_sim import Action, PlantConfig, PlantSimulator

from plant_backend.app import app, runtime
from plant_backend.controllers import Constraints, HeuristicController
from plant_backend.controllers.rl import RLController, RLControllerSlot
from plant_backend.orchestration import PlantRuntime
from plant_backend.rl import checkpoints, spaces
from plant_backend.rl.evaluation import compare, write
from plant_backend.rl.episodes import EpisodeConfig
from plant_backend.rl.policies import LoadedPolicy, load_policy
from plant_backend.rl.training.config import TrainingConfig
from plant_backend.rl.training.train import main as train_main
from plant_backend.rl.training.ppo import plan

BUSY_ACTION = [-0.6, 0.0, -0.4, -0.6, 0.0]
SHORT = EpisodeConfig(horizon_h=1.0)


@pytest.fixture
def mock_checkpoint(tmp_path: Path) -> Path:
    return checkpoints.write_fixed_checkpoint(tmp_path / "mock", BUSY_ACTION)


@pytest.fixture(autouse=True)
def _clean_runtime_rl():
    """The app-level runtime is a module singleton; leave it as we found it."""
    yield
    runtime.unload_rl_checkpoint()
    runtime.controller_id = HeuristicController.id
    runtime.set_mode("auto")


# --- checkpoint metadata --------------------------------------------------
def test_a_mock_checkpoint_never_claims_to_be_trained(mock_checkpoint: Path) -> None:
    metadata = checkpoints.load_metadata(mock_checkpoint)
    assert metadata.format == checkpoints.FORMAT_FIXED
    assert not metadata.trained
    assert "not a trained policy" in metadata.notes.lower()
    assert metadata.observation_fields == spaces.OBSERVATION_NAMES
    assert metadata.environment_id == spaces.ENVIRONMENT_ID
    assert metadata.environment_contract_digest == spaces.ENVIRONMENT_CONTRACT_DIGEST


def test_new_checkpoint_metadata_serialises_the_environment_contract(
    mock_checkpoint: Path,
) -> None:
    raw = json.loads(checkpoints.metadata_path(mock_checkpoint).read_text(encoding="utf-8"))
    assert raw["environment_id"] == "plant-env-v1"
    assert raw["environment_contract_digest"] == spaces.ENVIRONMENT_CONTRACT_DIGEST


def test_missing_checkpoint_is_an_error_not_a_random_policy(tmp_path: Path) -> None:
    with pytest.raises(checkpoints.MissingCheckpointError, match="no checkpoint at"):
        load_policy(tmp_path / "nothing-here")


def test_a_directory_without_metadata_is_not_a_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(checkpoints.CorruptCheckpointError, match="metadata.json"):
        load_policy(tmp_path / "empty")


def test_unreadable_metadata_is_reported_as_corrupt(tmp_path: Path) -> None:
    directory = tmp_path / "broken"
    directory.mkdir()
    checkpoints.metadata_path(directory).write_text("{not json", encoding="utf-8")
    with pytest.raises(checkpoints.CorruptCheckpointError, match="cannot read"):
        load_policy(directory)


def test_a_checkpoint_from_another_observation_space_is_refused(mock_checkpoint: Path) -> None:
    """The failure names the field that moved, so the fix is obvious."""
    target = checkpoints.metadata_path(mock_checkpoint)
    raw = json.loads(target.read_text(encoding="utf-8"))
    raw["observation_fields"] = ["battery_soc", *raw["observation_fields"][1:]]
    target.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(checkpoints.IncompatibleCheckpointError, match="observation fields differ"):
        load_policy(mock_checkpoint)


def test_a_checkpoint_of_the_wrong_size_is_refused(mock_checkpoint: Path) -> None:
    target = checkpoints.metadata_path(mock_checkpoint)
    raw = json.loads(target.read_text(encoding="utf-8"))
    raw["observation_fields"] = raw["observation_fields"][:-1]
    target.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(checkpoints.IncompatibleCheckpointError, match="observation fields"):
        load_policy(mock_checkpoint)


def test_an_older_spec_version_is_refused(mock_checkpoint: Path) -> None:
    target = checkpoints.metadata_path(mock_checkpoint)
    raw = json.loads(target.read_text(encoding="utf-8"))
    raw["spec_version"] = spaces.SPEC_VERSION - 1
    target.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(checkpoints.IncompatibleCheckpointError, match="spec version"):
        load_policy(mock_checkpoint)


def test_a_checkpoint_from_another_environment_is_refused(mock_checkpoint: Path) -> None:
    target = checkpoints.metadata_path(mock_checkpoint)
    raw = json.loads(target.read_text(encoding="utf-8"))
    raw["environment_id"] = "plant-env-v2"
    target.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(checkpoints.IncompatibleCheckpointError, match="checkpoint environment"):
        load_policy(mock_checkpoint)


def test_a_normalisation_only_change_is_refused_by_digest(mock_checkpoint: Path) -> None:
    changed = spaces.contract_payload()
    observation_fields = changed["observation_fields"]
    assert isinstance(observation_fields, list)
    first_field = observation_fields[0]
    assert isinstance(first_field, dict)
    first_field["normalisation"] = "sin(2*pi*h/12)"
    changed_digest = spaces.contract_digest(changed)
    assert changed_digest != spaces.ENVIRONMENT_CONTRACT_DIGEST

    target = checkpoints.metadata_path(mock_checkpoint)
    raw = json.loads(target.read_text(encoding="utf-8"))
    raw["environment_contract_digest"] = changed_digest
    target.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(checkpoints.IncompatibleCheckpointError, match="contract digest"):
        load_policy(mock_checkpoint)


def test_legacy_v1_metadata_without_environment_fields_still_loads(
    mock_checkpoint: Path,
) -> None:
    target = checkpoints.metadata_path(mock_checkpoint)
    raw = json.loads(target.read_text(encoding="utf-8"))
    raw.pop("environment_id")
    raw.pop("environment_contract_digest")
    target.write_text(json.dumps(raw), encoding="utf-8")

    metadata = checkpoints.load_metadata(mock_checkpoint)
    assert metadata.environment_id == spaces.ENVIRONMENT_ID
    assert metadata.environment_contract_digest == ""


def test_an_unknown_format_is_refused(mock_checkpoint: Path) -> None:
    target = checkpoints.metadata_path(mock_checkpoint)
    raw = json.loads(target.read_text(encoding="utf-8"))
    raw["format"] = "pickle-of-hope"
    target.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(checkpoints.CorruptCheckpointError, match="unsupported checkpoint format"):
        load_policy(mock_checkpoint)


# --- the slot -------------------------------------------------------------
def test_an_empty_slot_says_why_rl_is_unavailable() -> None:
    slot = RLControllerSlot()
    assert not slot.available
    assert slot.controller is None
    assert slot.status.state == "unavailable"
    assert checkpoints.CHECKPOINT_ENV in slot.status.reason
    assert slot.descriptor()["available"] is False


def test_a_failed_load_leaves_the_slot_empty_with_the_reason(tmp_path: Path) -> None:
    slot = RLControllerSlot()
    with pytest.raises(checkpoints.CheckpointError):
        slot.load(tmp_path / "absent")
    assert slot.controller is None
    assert slot.status.state == "failed"
    assert "no checkpoint at" in slot.status.reason


def test_a_failed_load_does_not_disturb_the_loaded_policy(
    mock_checkpoint: Path, tmp_path: Path
) -> None:
    """A bad path says nothing about the policy that already loaded cleanly."""
    slot = RLControllerSlot()
    controller = slot.load(mock_checkpoint)
    with pytest.raises(checkpoints.CheckpointError):
        slot.load(tmp_path / "absent")
    assert slot.controller is controller
    assert slot.available and slot.status.state == "ready"


def test_loading_a_mock_marks_it_as_untrained(mock_checkpoint: Path) -> None:
    slot = RLControllerSlot()
    slot.load(mock_checkpoint)
    assert slot.available
    assert "not a trained policy" in slot.status.reason
    assert slot.status.metadata is not None and not slot.status.metadata["trained"]


def test_autoload_reads_the_environment(mock_checkpoint: Path, monkeypatch) -> None:
    monkeypatch.setenv(checkpoints.CHECKPOINT_ENV, str(mock_checkpoint))
    slot = RLControllerSlot()
    assert slot.autoload().available


def test_autoload_survives_a_bad_path(monkeypatch, tmp_path: Path) -> None:
    """Start-up must not die because a checkpoint moved."""
    monkeypatch.setenv(checkpoints.CHECKPOINT_ENV, str(tmp_path / "gone"))
    status = RLControllerSlot().autoload()
    assert status.state == "failed" and not status.available


# --- the controller -------------------------------------------------------
def test_the_controller_produces_valid_actions(mock_checkpoint: Path) -> None:
    controller = RLController(load_policy(mock_checkpoint))
    config = PlantConfig()
    simulator = PlantSimulator(config)
    simulator.run(int(10 * 3600 / config.dt_s), Action.idle())
    decision = controller.compute_action(
        simulator.get_telemetry(), constraints=Constraints(), config=config
    )
    assert decision.action == Action.sanitised(decision.action)
    assert any("Mock checkpoint" in note for note in decision.notes)
    assert controller.failure is None


def test_inference_failure_is_recorded_and_idles_the_plant() -> None:
    class Broken:
        def act(self, observation: np.ndarray) -> np.ndarray:
            raise RuntimeError("no weights")

    metadata = checkpoints.CheckpointMetadata(format=checkpoints.FORMAT_FIXED, algorithm="broken")
    controller = RLController(LoadedPolicy(Broken(), metadata, Path("/nowhere")))
    simulator = PlantSimulator()
    decision = controller.compute_action(
        simulator.get_telemetry(), constraints=Constraints(), config=simulator.config
    )
    assert decision.action == Action.idle()
    assert controller.failure is not None and "no weights" in controller.failure


# --- the runtime ----------------------------------------------------------
def test_the_heuristic_is_the_default_without_a_checkpoint() -> None:
    plant = PlantRuntime()
    assert plant.controller_id == HeuristicController.id
    # V8 added the safe controller; RL still needs a checkpoint to exist at all
    assert list(plant.controllers) == [HeuristicController.id, "safe"]
    status = plant.control_status()
    assert status["rl"]["available"] is False
    # RL is still offered, so the reason is visible instead of the option missing
    assert [entry["id"] for entry in status["controllers"]] == ["heuristic", "safe", "rl"]


def test_selecting_rl_without_a_checkpoint_fails_with_the_reason() -> None:
    plant = PlantRuntime()
    with pytest.raises(ValueError, match="RL controller unavailable"):
        plant.set_controller("rl")
    assert plant.controller_id == HeuristicController.id


def test_loading_a_checkpoint_does_not_switch_controller(mock_checkpoint: Path) -> None:
    plant = PlantRuntime()
    plant.load_rl_checkpoint(str(mock_checkpoint))
    assert plant.controller_id == HeuristicController.id
    assert plant.set_controller("rl") == "rl"


def test_rl_actions_still_pass_the_guard(mock_checkpoint: Path) -> None:
    """The policy asks for everything; the plant only ever gets a valid action."""
    plant = PlantRuntime()
    plant.load_rl_checkpoint(str(mock_checkpoint))
    plant.set_controller("rl")
    plant.advance(20)
    assert plant.applied_action == Action.sanitised(plant.applied_action)
    assert plant.applied_action.battery_support_kw <= plant.constraints.support_limit_kw(
        plant.config
    )


def test_unloading_hands_the_plant_back_to_the_heuristic(mock_checkpoint: Path) -> None:
    plant = PlantRuntime()
    plant.load_rl_checkpoint(str(mock_checkpoint))
    plant.set_controller("rl")
    status = plant.unload_rl_checkpoint()
    assert plant.controller_id == HeuristicController.id
    assert status["available"] is False
    plant.advance(2)  # and it keeps running


def test_a_bad_checkpoint_never_strands_a_driving_policy(
    mock_checkpoint: Path, tmp_path: Path
) -> None:
    """The status and the driver may never disagree about whether RL can drive."""
    plant = PlantRuntime()
    plant.load_rl_checkpoint(str(mock_checkpoint))
    plant.set_controller("rl")
    with pytest.raises(ValueError, match="no checkpoint at"):
        plant.load_rl_checkpoint(str(tmp_path / "absent"))
    assert plant.controller_id == "rl"
    assert plant.rl.status.state == "ready"
    assert plant.control_status()["rl"]["available"] is True
    plant.advance(2)


def test_a_broken_policy_reverts_the_runtime_to_the_heuristic(mock_checkpoint: Path) -> None:
    class Broken:
        def act(self, observation: np.ndarray) -> np.ndarray:
            raise RuntimeError("kernel panic")

    plant = PlantRuntime()
    plant.load_rl_checkpoint(str(mock_checkpoint))
    plant.set_controller("rl")
    plant.controllers["rl"].loaded = LoadedPolicy(  # type: ignore[union-attr]
        Broken(), checkpoints.load_metadata(mock_checkpoint, check=False), mock_checkpoint
    )
    plant.advance(1)
    assert plant.controller_id == HeuristicController.id
    assert plant.rl.status.state == "failed"
    assert any("reverted to heuristic" in note for note in plant.notes)
    assert "rl" not in plant.controllers


def test_manual_mode_is_untouched_by_the_rl_slot(mock_checkpoint: Path) -> None:
    plant = PlantRuntime()
    plant.load_rl_checkpoint(str(mock_checkpoint))
    plant.set_controller("rl")
    plant.set_mode("manual")
    plant.set_manual_action({"electrolyser_share": 0.5, "battery_support_kw": 0.0})
    plant.advance(1)
    assert plant.notes == ("Manual control: operator allocation",)
    assert plant.applied_action.electrolyser_share == pytest.approx(0.5)


# --- the API --------------------------------------------------------------
def test_api_reports_rl_as_unavailable_by_default() -> None:
    client = TestClient(app)
    status = client.get("/api/controller").json()
    assert status["rl"]["available"] is False
    assert status["controller"]["id"] == "heuristic"


def test_api_refuses_rl_selection_and_bad_checkpoints(tmp_path: Path) -> None:
    client = TestClient(app)
    refused = client.post("/api/controller/config", json={"controller": "rl"})
    assert refused.status_code == 400
    assert "unavailable" in refused.json()["detail"]
    missing = client.post("/api/controller/rl", json={"checkpoint": str(tmp_path / "nope")})
    assert missing.status_code == 400
    assert "no checkpoint at" in missing.json()["detail"]
    assert client.get("/api/controller").json()["controller"]["id"] == "heuristic"


def test_api_loads_selects_and_unloads_a_checkpoint(mock_checkpoint: Path) -> None:
    client = TestClient(app)
    loaded = client.post("/api/controller/rl", json={"checkpoint": str(mock_checkpoint)})
    assert loaded.status_code == 200
    assert loaded.json()["control"]["rl"]["available"] is True
    assert client.post("/api/controller/config", json={"controller": "rl"}).status_code == 200
    driving = client.post("/api/simulation/step", json={"repeat": 3}).json()["control"]
    assert driving["controller"]["id"] == "rl"
    assert any("Mock checkpoint" in note for note in driving["notes"])
    unloaded = client.post("/api/controller/rl", json={"checkpoint": None}).json()["control"]
    assert unloaded["controller"]["id"] == "heuristic"
    assert unloaded["rl"]["available"] is False


# --- evaluation -----------------------------------------------------------
def test_evaluation_of_the_baseline_alone_is_labelled_a_smoke_test() -> None:
    result = compare(seeds=[1], episode=SHORT)
    assert result.smoke_test
    assert result.controllers == ["heuristic"]
    summary = result.summary()["heuristic"]
    assert summary["runs"] == 1 and summary["ch4_kg"] > 0.0
    assert "no trained policy" in result.to_dict()["disclaimer"].lower()


def test_evaluation_gives_both_controllers_the_same_episode(mock_checkpoint: Path) -> None:
    result = compare(seeds=[7], checkpoint=mock_checkpoint, episode=SHORT)
    assert result.smoke_test  # a mock is not a contender
    scenarios = {run.controller: run.scenario for run in result.runs}
    assert scenarios["heuristic"] == scenarios["rl"]
    assert {run.controller for run in result.runs} == {"heuristic", "rl"}
    for run in result.runs:
        assert run.metrics.steps == SHORT.steps(PlantConfig())


def test_evaluation_is_reproducible(mock_checkpoint: Path) -> None:
    first = compare(seeds=[3], checkpoint=mock_checkpoint, episode=SHORT)
    second = compare(seeds=[3], checkpoint=mock_checkpoint, episode=SHORT)
    assert first.summary() == second.summary()


def test_evaluation_writes_machine_readable_results(tmp_path: Path) -> None:
    result = compare(seeds=[1], episode=SHORT)
    target = write(result, tmp_path / "eval.json")
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["smoke_test"] is True
    assert payload["summary"]["heuristic"]["runs"] == 1
    assert "SMOKE TEST" in result.report()


# --- training wiring ------------------------------------------------------
def test_the_training_plan_describes_the_run_without_training() -> None:
    config = TrainingConfig(seed=1, total_timesteps=1000, episode=SHORT)
    described = plan(config)
    assert described["observation_shape"] == [spaces.OBSERVATION_SIZE]
    assert described["action_shape"] == [spaces.ACTION_SIZE]
    assert described["steps_per_episode"] == SHORT.steps(PlantConfig())


def test_dry_run_trains_nothing_and_exits_cleanly(capsys) -> None:
    assert train_main(["--dry-run", "--timesteps", "1000", "--seed", "2"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["dry_run"] is True
    assert printed["plan"]["seed"] == 2


def test_training_config_round_trips_through_a_file(tmp_path: Path) -> None:
    config = TrainingConfig(seed=9, total_timesteps=5000, notes="wiring check")
    target = config.write(tmp_path / "train.json")
    assert TrainingConfig.from_file(target).to_dict() == config.to_dict()


def test_training_config_rejects_unknown_settings() -> None:
    with pytest.raises(ValueError, match="unknown training settings"):
        TrainingConfig.from_dict({"learning_rate": 1e-3})
    with pytest.raises(ValueError, match="unknown PPO settings"):
        TrainingConfig.from_dict({"ppo": {"lr": 1e-3}})

"""Fail-closed evaluator preflight and command-line publication tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from plant_backend.rl import checkpoints
from plant_backend.rl.v2 import economic_spaces, policy_evaluation, setpoint_spaces
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.policy_setpoint_controller import (
    policy_contract_for_environment,
)
from plant_backend.rl.v2.training.config import ControllerNeutralTrainingConfigV1
from plant_backend.rl.v2.training.ppo import make_env


def _config(environment_id: str) -> ControllerNeutralTrainingConfigV1:
    return ControllerNeutralTrainingConfigV1(
        environment_id=environment_id,
        total_timesteps=4,
        n_envs=1,
        checkpoint_every=0,
        episode=EpisodeConfigV2(
            horizon_h=0.25,
            delivery_period_h=0.25,
            delivery_rate_kg_h=1.0,
            battery_soc=(0.5, 0.5),
            h2_fill=(0.0, 0.0),
            co2_fill=(0.0, 0.0),
            ch4_coverage_h=(0.0, 0.0),
        ),
    ).validate()


def _checkpoint(
    tmp_path: Path,
    config: ControllerNeutralTrainingConfigV1,
) -> Path:
    environment = make_env(config)
    try:
        contract = policy_contract_for_environment(environment)
    finally:
        environment.close()
    path = checkpoints.write_fixed_checkpoint(
        tmp_path / f"checkpoint-{config.environment_id}",
        [-1.0, -1.0, -1.0, -1.0],
        contract=contract,
    )
    metadata = checkpoints.load_metadata(path, check=False)
    checkpoints.save_metadata(
        path,
        replace(metadata, training_config=config.to_dict()),
    )
    return path


@pytest.mark.parametrize(
    ("checkpoint_environment", "selected_environment"),
    [
        (economic_spaces.ENVIRONMENT_ID, setpoint_spaces.ENVIRONMENT_ID),
        (setpoint_spaces.ENVIRONMENT_ID, economic_spaces.ENVIRONMENT_ID),
    ],
)
def test_wrong_task_is_refused_before_either_arm_and_matching_task_recovers(
    checkpoint_environment: str,
    selected_environment: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_config = _config(checkpoint_environment)
    selected_config = _config(selected_environment)
    path = _checkpoint(tmp_path, checkpoint_config)
    loaded = policy_evaluation.load_policy_for_task(path, checkpoint_config)
    arm_calls: list[str] = []

    def moved(*_: object, **__: object) -> dict[str, object]:
        arm_calls.append("moved")
        raise AssertionError("an evaluation arm moved before task admission")

    with monkeypatch.context() as patch:
        patch.setattr(policy_evaluation, "execute_heuristic_episode", moved)
        patch.setattr(policy_evaluation, "execute_policy_episode", moved)
        with pytest.raises(checkpoints.IncompatibleCheckpointError):
            policy_evaluation.compare_policy_to_heuristic(
                loaded,
                selected_config,
                seeds=[0],
            )

    assert arm_calls == []
    recovered = policy_evaluation.compare_policy_to_heuristic(
        loaded,
        checkpoint_config,
        seeds=[0],
    )
    assert all(run["status"] == "complete" for run in recovered["runs"])
    observed_task_digests = {
        run["manifest_identity"]["checkpoint_contract_sha256"]
        for run in recovered["runs"]
    }
    assert observed_task_digests == {loaded.metadata.checkpoint_contract_digest}


@pytest.mark.parametrize(
    "environment_id",
    [economic_spaces.ENVIRONMENT_ID, setpoint_spaces.ENVIRONMENT_ID],
)
def test_cli_recovers_stored_task_and_exclusively_writes_json(
    environment_id: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _checkpoint(tmp_path, _config(environment_id))
    target = tmp_path / f"evaluation-{environment_id}.json"

    assert (
        policy_evaluation.main(
            ["--checkpoint", str(path), "--seed-start", "3", "--json", str(target)]
        )
        == 0
    )

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["training_task"]["environment_id"] == environment_id
    assert payload["seeds"] == [3]
    assert all(run["status"] == "complete" for run in payload["runs"])
    assert json.loads(capsys.readouterr().out)["format"] == payload["format"]


def test_cli_refuses_to_overwrite_an_existing_json_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _checkpoint(tmp_path, _config(economic_spaces.ENVIRONMENT_ID))
    target = tmp_path / "existing.json"
    target.write_text("operator-owned\n", encoding="utf-8")

    assert (
        policy_evaluation.main(["--checkpoint", str(path), "--json", str(target)]) == 2
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "output already exists" in captured.err
    assert target.read_text(encoding="utf-8") == "operator-owned\n"


def test_cli_catches_runtime_refusal_without_claiming_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _checkpoint(tmp_path, _config(setpoint_spaces.ENVIRONMENT_ID))
    target = tmp_path / "must-not-exist.json"

    def refuse(*_: object, **__: object) -> dict[str, object]:
        raise RuntimeError("paired preflight refused deliberately")

    monkeypatch.setattr(policy_evaluation, "compare_policy_to_heuristic", refuse)

    assert (
        policy_evaluation.main(["--checkpoint", str(path), "--json", str(target)]) == 2
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "evaluation refused" in captured.err
    assert "preflight refused deliberately" in captured.err
    assert not target.exists()

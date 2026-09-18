"""Outcome-free cross-task checks for the controller-neutral RL integration."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from plant_backend.rl import checkpoints
from plant_backend.rl.policies import load_policy
from plant_backend.rl.v2 import economic_spaces, setpoint_spaces
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.economic_env import PlantEconomicSetpointEnvV2
from plant_backend.rl.v2.policy_evaluation import (
    compare_policy_to_heuristic,
    load_policy_for_task,
)
from plant_backend.rl.v2.policy_setpoint_controller import (
    NORMALIZED_POLICY_SETPOINT_CONTROLLER_ID,
    NormalizedSetpointPolicyControllerV1,
    PolicyObservationV1,
    SetpointPolicyEnvironmentV1,
    policy_contract_for_environment,
)
from plant_backend.rl.v2.training.config import ControllerNeutralTrainingConfigV1
from plant_backend.rl.v2.training.ppo import build_model, make_env, save_checkpoint


TASK_MATRIX = (
    (setpoint_spaces.ENVIRONMENT_ID, setpoint_spaces.OBSERVATION_SIZE),
    (economic_spaces.ENVIRONMENT_ID, economic_spaces.OBSERVATION_SIZE),
)
FIXED_ACTION = np.asarray((-0.75, 0.25, 0.5, -0.5), dtype=np.float32)


def _config(environment_id: str) -> ControllerNeutralTrainingConfigV1:
    return ControllerNeutralTrainingConfigV1(
        environment_id=environment_id,
        episode=EpisodeConfigV2(
            horizon_h=0.25,
            delivery_period_h=0.25,
            delivery_rate_kg_h=1.0,
            battery_soc=(0.5, 0.5),
            h2_fill=(0.0, 0.0),
            co2_fill=(0.0, 0.0),
            ch4_coverage_h=(0.0, 0.0),
        ),
        total_timesteps=4,
        n_envs=1,
        checkpoint_every=0,
    ).validate()


def _fixed_checkpoint(
    tmp_path: Path,
    config: ControllerNeutralTrainingConfigV1,
) -> Path:
    environment = make_env(config)
    try:
        return checkpoints.write_fixed_checkpoint(
            tmp_path / f"fixed-{config.environment_id}",
            FIXED_ACTION,
            contract=policy_contract_for_environment(environment),
        )
    finally:
        environment.close()


def _controller_source(
    environment: SetpointPolicyEnvironmentV1,
) -> PolicyObservationV1:
    if isinstance(environment, PlantEconomicSetpointEnvV2):
        return environment.economic_controller_observation
    return environment.controller_observation


def _deterministic_step_info(info: dict[str, Any]) -> dict[str, Any]:
    """Remove only command provenance and wall-clock fields from a step record."""

    excluded = {
        "requested",
        "process_setpoint_execution",
        "source_decision",
        "dispatcher_elapsed_ns",
        "dispatcher_call_count",
    }
    return {key: value for key, value in info.items() if key not in excluded}


@pytest.mark.parametrize(("environment_id", "observation_size"), TASK_MATRIX)
def test_fixed_checkpoint_gym_and_common_command_have_exact_task_parity(
    tmp_path: Path,
    environment_id: str,
    observation_size: int,
) -> None:
    """Both policy doors must produce the same state and deterministic accounting."""

    config = _config(environment_id)
    gym_environment = make_env(config)
    command_environment = make_env(config)
    gym_observation, gym_reset_info = gym_environment.reset(seed=904)
    command_observation, command_reset_info = command_environment.reset(seed=904)
    loaded = load_policy(
        _fixed_checkpoint(tmp_path, config),
        contract=policy_contract_for_environment(command_environment),
    )
    controller = NormalizedSetpointPolicyControllerV1.from_environment(
        loaded,
        command_environment,
    )
    controller.reset()
    decision = controller.compute_command(
        _controller_source(command_environment),
        constraints=command_environment.constraints,
        config=command_environment.config,
        commitment_profile=(command_environment.episode_config.electrolyser_commitment),
    )

    gym_result = gym_environment.step(FIXED_ACTION.copy())
    command_result = command_environment.step_process_setpoints(decision.command)

    assert gym_observation.shape == command_observation.shape == (observation_size,)
    assert gym_environment.action_space.shape == (4,)
    np.testing.assert_array_equal(gym_observation, command_observation)
    assert gym_reset_info == command_reset_info
    np.testing.assert_array_equal(gym_result[0], command_result[0])
    assert gym_result[1:4] == command_result[1:4]
    assert _deterministic_step_info(gym_result[4]) == _deterministic_step_info(
        command_result[4]
    )
    assert gym_result[4]["source_decision"]["targets"] == (
        decision.evidence.decoded_action.targets.to_dict()
    )
    assert gym_environment.telemetry == command_environment.telemetry
    assert gym_environment.control_state == command_environment.control_state
    assert gym_environment.metrics.to_dict() == command_environment.metrics.to_dict()
    assert gym_environment.audit_manifest() == command_environment.audit_manifest()
    if isinstance(gym_environment, PlantEconomicSetpointEnvV2):
        assert isinstance(command_environment, PlantEconomicSetpointEnvV2)
        assert gym_environment.economic_metrics.to_dict() == (
            command_environment.economic_metrics.to_dict()
        )


@pytest.mark.parametrize(("environment_id", "observation_size"), TASK_MATRIX)
def test_fixed_checkpoint_pairs_with_heuristic_on_each_exact_task(
    tmp_path: Path,
    environment_id: str,
    observation_size: int,
) -> None:
    """The evaluator pairs task identity without making a performance claim."""

    config = _config(environment_id)
    checkpoint = _fixed_checkpoint(tmp_path, config)
    loaded = load_policy_for_task(checkpoint, config)

    result = compare_policy_to_heuristic(loaded, config, seeds=[905])

    assert result["smoke_test"] is True
    assert result["claim_boundary"] == "integration smoke; no learned-controller claim"
    assert result["training_task"]["environment_id"] == environment_id
    assert result["checkpoint"]["metadata"]["observation_size"] == observation_size
    assert result["checkpoint"]["metadata"]["action_size"] == 4
    assert result["controller_order"] == [
        "legacy-heuristic-setpoint-authority-v1",
        NORMALIZED_POLICY_SETPOINT_CONTROLLER_ID,
    ]
    heuristic, policy = result["runs"]
    assert heuristic["manifest_identity"] == policy["manifest_identity"]
    assert heuristic["status"] == policy["status"] == "complete"
    assert heuristic["physical_steps"] == policy["physical_steps"] == 15
    assert heuristic["completed_decisions"] == policy["completed_decisions"] == 1
    assert (heuristic["economic_metrics"] is None) == (
        environment_id == setpoint_spaces.ENVIRONMENT_ID
    )
    assert (policy["economic_metrics"] is None) == (
        environment_id == setpoint_spaces.ENVIRONMENT_ID
    )


@pytest.mark.parametrize(
    ("checkpoint_environment_id", "requested_environment_id"),
    (
        (setpoint_spaces.ENVIRONMENT_ID, economic_spaces.ENVIRONMENT_ID),
        (economic_spaces.ENVIRONMENT_ID, setpoint_spaces.ENVIRONMENT_ID),
    ),
)
def test_checkpoint_loading_rejects_physical_economic_task_substitution(
    tmp_path: Path,
    checkpoint_environment_id: str,
    requested_environment_id: str,
) -> None:
    source = _config(checkpoint_environment_id)
    requested = _config(requested_environment_id)
    checkpoint = _fixed_checkpoint(tmp_path, source)

    with pytest.raises(
        checkpoints.IncompatibleCheckpointError,
        match="observation fields|environment",
    ):
        load_policy_for_task(checkpoint, requested)


@pytest.mark.parametrize(
    ("checkpoint_environment_id", "resume_environment_id"),
    (
        (setpoint_spaces.ENVIRONMENT_ID, economic_spaces.ENVIRONMENT_ID),
        (economic_spaces.ENVIRONMENT_ID, setpoint_spaces.ENVIRONMENT_ID),
    ),
)
def test_resume_rejects_wrong_task_before_loading_sb3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_environment_id: str,
    resume_environment_id: str,
) -> None:
    """A fake artifact proves task admission happens before any trainer loading."""

    source = _config(checkpoint_environment_id)
    source_environment = make_env(source)

    class FakeSavedModel:
        observation_space = source_environment.observation_space
        action_space = source_environment.action_space

        def save(self, path: Path) -> None:
            path.write_bytes(b"not an SB3 model; admission must reject it first")

    checkpoint = save_checkpoint(
        FakeSavedModel(),
        source,
        tmp_path / f"resume-source-{checkpoint_environment_id}",
        total_timesteps=4,
    )
    source_environment.close()
    resume = replace(
        _config(resume_environment_id),
        resume_from=str(checkpoint),
    )

    class ForbiddenPPO:
        @classmethod
        def load(cls, *_: object, **__: object) -> object:
            raise AssertionError("PPO.load ran before checkpoint task admission")

    fake_sb3 = ModuleType("stable_baselines3")
    fake_sb3.PPO = ForbiddenPPO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stable_baselines3", fake_sb3)

    with pytest.raises(
        checkpoints.IncompatibleCheckpointError,
        match="observation fields|environment",
    ):
        build_model(resume, env=object())

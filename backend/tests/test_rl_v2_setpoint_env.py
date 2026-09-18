"""Opt-in Gym and physical-evidence tests for common setpoint authority."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from numpy.typing import NDArray

from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.contract import (
    CheckpointContractError,
    require_compatible_checkpoint,
)
from plant_backend.rl.v2.env import PlantEnvV2
from plant_backend.rl.v2.mpc_execution import public_observation_sha256
from plant_backend.rl.v2.setpoint_command import (
    ProcessSetpointCommandV1,
    issue_process_setpoint_command,
)
from plant_backend.rl.v2.setpoint_control import ProcessSetpointActionAdapterV1
from plant_backend.rl.v2.setpoint_env import PlantSetpointEnvV2
from plant_backend.rl.v2 import setpoint_spaces


TINY = EpisodeConfigV2(horizon_h=0.25, delivery_period_h=0.25)


def _zero_command(env: PlantEnvV2) -> ProcessSetpointCommandV1:
    observation = env.controller_observation
    decoded = ProcessSetpointActionAdapterV1().decode(
        np.zeros(4, dtype=np.float32),
        constraints=env.constraints,
        config=env.config,
    )
    return issue_process_setpoint_command(
        decoded.targets,
        decision_index=env.decisions_taken,
        observation_sha256=public_observation_sha256(observation),
        hold_duration_s=observation.decision_interval_s,
        issuer_id="test-normalized-policy-v1",
        source_schema=decoded.schema,
        source_sha256=decoded.sha256,
        constraints=env.constraints,
        config=env.config,
    )


def test_setpoint_env_reuses_observations_but_has_distinct_action_and_identity() -> (
    None
):
    legacy = PlantEnvV2(episode=TINY)
    setpoint = PlantSetpointEnvV2(episode=TINY)
    legacy_observation, legacy_info = legacy.reset(seed=17)
    setpoint_observation, setpoint_info = setpoint.reset(seed=17)

    np.testing.assert_array_equal(setpoint_observation, legacy_observation)
    assert setpoint_observation.shape == (57,)
    assert legacy.action_space.shape == (5,)
    assert setpoint.action_space.shape == (4,)
    assert setpoint_info["environment_id"] == setpoint_spaces.ENVIRONMENT_ID
    assert setpoint_info["environment_id"] != legacy_info["environment_id"]
    assert setpoint.checkpoint_contract_digest != legacy.checkpoint_contract_digest
    assert (
        setpoint.audit_manifest()["exogenous_episode_sha256"]
        == legacy.audit_manifest()["exogenous_episode_sha256"]
    )

    with pytest.raises(CheckpointContractError, match="environment"):
        require_compatible_checkpoint(
            legacy.checkpoint_contract,
            expected=setpoint.checkpoint_contract,
        )
    with pytest.raises(CheckpointContractError, match="environment"):
        require_compatible_checkpoint(
            setpoint.checkpoint_contract,
            expected=legacy.checkpoint_contract,
        )


def test_setpoint_env_passes_the_gymnasium_contract_checker() -> None:
    check_env(PlantSetpointEnvV2(episode=TINY), skip_render_check=True)


def test_normalized_action_is_bound_to_one_complete_common_execution() -> None:
    env = PlantSetpointEnvV2(episode=TINY)
    env.reset(seed=3)
    before_sha256 = public_observation_sha256(env.controller_observation)

    observation, _, terminated, truncated, info = env.step(
        np.zeros(4, dtype=np.float32)
    )

    assert observation.shape == (57,)
    assert terminated is True
    assert truncated is False
    assert info["environment_id"] == setpoint_spaces.ENVIRONMENT_ID
    source = info["source_decision"]
    command = info["requested"]
    execution = info["process_setpoint_execution"]
    assert source["raw_policy_vector"] == [0.0, 0.0, 0.0, 0.0]
    assert source["targets"]["setpoints_kw"] == {
        "electrolyser": 1_500.0,
        "dac": 600.0,
        "methanator": 90.0,
        "battery_charge_ceiling": 1_250.0,
    }
    assert command["observation_sha256"] == before_sha256
    assert command["issuer"]["source_sha256"] == source["sha256"]
    assert execution["command"]["sha256"] == command["sha256"]
    assert execution["application_count"] == 15
    assert len(execution["applications"]) == 15
    assert info["dispatcher_call_count"] == 15
    assert "dispatcher_elapsed_ns" not in info
    assert env.last_process_setpoint_execution is not None
    assert env.last_process_setpoint_execution.sha256 == execution["sha256"]
    assert env.last_setpoint_execution is None
    truth_states = env.last_minute_truth_states
    assert truth_states is not None
    assert len(truth_states) == 16
    assert [state.step for state in truth_states] == list(range(16))
    assert (
        min(state.battery_soc for state in truth_states)
        == info["metrics"]["min_battery_soc"]
    )
    assert "minute_truth_states" not in info

    attributions = [
        item["plant"]["battery_charge_attribution"]
        for item in execution["applications"]
    ]
    assert all(
        item["direct_dispatch_kw"] <= item["direct_dispatch_ceiling_kw"]
        for item in attributions
    )


@pytest.mark.parametrize(
    ("action", "error", "message"),
    [
        (np.zeros(3, dtype=np.float32), ValueError, "exactly four"),
        (
            np.asarray([0.0, True, 0.0, 0.0], dtype=object),
            TypeError,
            "real number",
        ),
        (
            np.asarray([0.0, np.nan, 0.0, 0.0], dtype=np.float64),
            ValueError,
            "finite",
        ),
    ],
)
def test_invalid_policy_action_fails_before_the_plant_moves(
    action: NDArray[np.generic],
    error: type[Exception],
    message: str,
) -> None:
    env = PlantSetpointEnvV2(episode=TINY)
    env.reset(seed=5)
    before = env.telemetry

    with pytest.raises(error, match=message):
        env.step(action)

    assert env.telemetry == before
    assert env.decisions_taken == 0
    assert env.physics_steps_taken == 0


def test_clipped_policy_action_remains_visible_in_health_metrics() -> None:
    env = PlantSetpointEnvV2(episode=TINY)
    env.reset(seed=7)

    _, _, _, _, info = env.step(np.asarray([2.0, -2.0, 0.0, 0.0], dtype=np.float32))

    assert info["source_decision"]["clipped"] is True
    assert info["guard"]["out_of_range"] is True
    assert info["metrics"]["out_of_range_decisions"] == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"decision_index": 1}, "decision index"),
        ({"observation_sha256": "f" * 64}, "another observation"),
        ({"hold_duration_s": 60.0}, "hold differs"),
    ],
)
def test_common_command_identity_fails_closed_before_the_plant_moves(
    mutation: dict[str, object],
    message: str,
) -> None:
    env = PlantEnvV2(episode=TINY)
    env.reset(seed=11)
    command = replace(_zero_command(env), **mutation)
    before = env.telemetry

    with pytest.raises(ValueError, match=message):
        env.step_process_setpoints(command)

    assert env.telemetry == before
    assert env.decisions_taken == 0


def test_common_command_rejects_a_stale_envelope_before_the_plant_moves() -> None:
    env = PlantEnvV2(episode=TINY)
    env.reset(seed=13)
    command = _zero_command(env)
    stale = replace(
        command,
        envelope=replace(command.envelope, battery_reserve_soc=0.25),
    )
    before = env.telemetry

    with pytest.raises(ValueError, match="envelope is stale"):
        env.step_process_setpoints(stale)

    assert env.telemetry == before
    assert env.decisions_taken == 0


def test_interval_evidence_rejects_missing_or_broken_minute_chains() -> None:
    env = PlantSetpointEnvV2(episode=TINY)
    env.reset(seed=19)
    env.step(np.zeros(4, dtype=np.float32))
    execution = env.last_process_setpoint_execution
    assert execution is not None

    with pytest.raises(ValueError, match="hold duration"):
        replace(execution, applications=execution.applications[:-1])

    second = execution.applications[1]
    broken_dispatch = replace(second.dispatch, boundary_sha256="f" * 64)
    broken_second = replace(second, dispatch=broken_dispatch)
    with pytest.raises(ValueError, match="boundary chain"):
        replace(
            execution,
            applications=(
                execution.applications[0],
                broken_second,
                *execution.applications[2:],
            ),
        )

"""Delivery-only Environment V2 contract, isolation, accounting, and reward tests."""

from __future__ import annotations

import json

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from plant_sim import Action, PlantConfig

from plant_backend.rl import PlantEnv as PlantEnvV1
from plant_backend.rl import spaces as v1_spaces
from plant_backend.rl.v2 import EpisodeConfigV2, PlantEnvV2, RewardWeightsV2
from plant_backend.rl.v2 import spaces as v2_spaces
from plant_backend.rl.v2.config import sample_episode
from plant_backend.rl.v2.contract import (
    CheckpointContractError,
    require_compatible_checkpoint,
)


SHORT = EpisodeConfigV2(horizon_h=2.0, delivery_period_h=1.0)


def test_v1_and_v2_are_explicit_coexisting_contracts() -> None:
    v1 = PlantEnvV1()
    v2 = PlantEnvV2(episode=SHORT)
    v1_observation, _ = v1.reset(seed=3)
    v2_observation, _ = v2.reset(seed=3)

    assert v1_spaces.ENVIRONMENT_ID == "plant-env-v1"
    assert v2_spaces.ENVIRONMENT_ID == "plant-env-v2-delivery-alpha"
    assert v1_observation.shape == (30,)
    assert v2_observation.shape == (34,)
    assert v1_spaces.ENVIRONMENT_CONTRACT_DIGEST != v2_spaces.ENVIRONMENT_CONTRACT_DIGEST
    assert v1.observation_space.contains(v1_observation)
    assert v2.observation_space.contains(v2_observation)


def test_v2_passes_the_gymnasium_contract_checker() -> None:
    check_env(PlantEnvV2(episode=SHORT), skip_render_check=True)


def test_v2_descriptor_digest_is_complete_and_deterministic() -> None:
    described = v2_spaces.describe()
    assert described["environment_contract_digest"] == v2_spaces.contract_digest()
    assert len(v2_spaces.ENVIRONMENT_CONTRACT_DIGEST) == 64
    int(v2_spaces.ENVIRONMENT_CONTRACT_DIGEST, 16)
    observation_fields = described["observation_fields"]
    assert isinstance(observation_fields, list)
    assert len(observation_fields) == v2_spaces.OBSERVATION_SIZE
    assert len(set(v2_spaces.OBSERVATION_NAMES)) == v2_spaces.OBSERVATION_SIZE
    assert described["digest_scope"] == "observation/action interface only"


def test_full_checkpoint_contract_changes_with_timing_and_objective() -> None:
    base = PlantEnvV2(episode=SHORT)
    slower = PlantEnvV2(
        episode=EpisodeConfigV2(
            horizon_h=2.0,
            decision_interval_s=1_800.0,
            delivery_period_h=1.0,
        )
    )
    reweighted = PlantEnvV2(
        episode=SHORT,
        weights=RewardWeightsV2(shortfall=5.0),
    )

    assert base.checkpoint_contract_digest != slower.checkpoint_contract_digest
    assert base.checkpoint_contract_digest != reweighted.checkpoint_contract_digest
    assert (
        base.checkpoint_contract["interface_contract_digest"]
        == slower.checkpoint_contract["interface_contract_digest"]
    )


def test_v2_checkpoint_compatibility_requires_the_full_exact_digest() -> None:
    env = PlantEnvV2(episode=SHORT)
    expected = env.checkpoint_contract
    require_compatible_checkpoint(
        {
            "environment_id": v2_spaces.ENVIRONMENT_ID,
            "checkpoint_contract_digest": env.checkpoint_contract_digest,
        },
        expected=expected,
    )

    with pytest.raises(CheckpointContractError, match="missing"):
        require_compatible_checkpoint(
            {"environment_id": v2_spaces.ENVIRONMENT_ID},
            expected=expected,
        )
    with pytest.raises(CheckpointContractError, match="mismatch"):
        require_compatible_checkpoint(
            {
                "environment_id": v2_spaces.ENVIRONMENT_ID,
                "checkpoint_contract_digest": "0" * 64,
            },
            expected=expected,
        )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (EpisodeConfigV2(decision_interval_s=100.0), "physics timestep"),
        (EpisodeConfigV2(horizon_h=1.1), "controller decisions"),
        (EpisodeConfigV2(delivery_period_h=1.1), "controller decisions"),
        (EpisodeConfigV2(horizon_h=1.0, delivery_period_h=2.0), "at least one"),
        (EpisodeConfigV2(delivery_rate_kg_h=0.0), "delivery_rate"),
        (EpisodeConfigV2(battery_soc=(0.2, 1.2)), "battery_soc"),
        (
            EpisodeConfigV2(
                horizon_h=1.0,
                delivery_period_h=1.0,
                delivery_rate_kg_h=4_000.0,
                ch4_coverage_h=(0.0, 0.0),
            ),
            "commitment exceeds",
        ),
    ],
)
def test_v2_episode_configuration_rejects_misaligned_or_invalid_values(
    config: EpisodeConfigV2,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate(PlantConfig())


def test_named_random_streams_keep_inventory_draws_stable() -> None:
    plant = PlantConfig()
    first = sample_episode(SHORT, plant, seed=42)
    changed_contract = sample_episode(
        EpisodeConfigV2(horizon_h=2.0, delivery_period_h=1.0, delivery_rate_kg_h=50.0),
        plant,
        seed=42,
    )
    assert tuple(first.stream_spawn_keys) == ("inventories", "weather", "contracts")
    assert first.scenario.initial.battery_soc == changed_contract.scenario.initial.battery_soc
    assert first.scenario.initial.h2_kg == changed_contract.scenario.initial.h2_kg
    assert first.scenario.initial.co2_kg == changed_contract.scenario.initial.co2_kg
    assert first.commitments[0].requested_kg != changed_contract.commitments[0].requested_kg


@pytest.mark.parametrize("value", [-0.01, float("nan"), float("inf")])
def test_v2_rejects_invalid_reward_weights(value: float) -> None:
    with pytest.raises(ValueError, match="shortfall.*finite and non-negative"):
        PlantEnvV2(episode=SHORT, weights=RewardWeightsV2(shortfall=value))


def test_public_info_contains_current_contract_context_but_no_future_tape() -> None:
    env = PlantEnvV2(episode=SHORT)
    _, info = env.reset(seed=17)
    serialized = json.dumps(info, sort_keys=True).lower()

    assert info["next_delivery"]["steps_until_due"] == 60
    assert "seed" not in serialized
    assert "scenario" not in serialized
    assert "fault" not in serialized
    assert "schedule" not in serialized
    assert "commitments" not in serialized
    assert info["checkpoint_contract_digest"] == env.checkpoint_contract_digest


def test_a_decision_holds_one_action_for_fifteen_physics_steps() -> None:
    env = PlantEnvV2(episode=SHORT)
    env.reset(seed=2)
    _, _, terminated, truncated, info = env.step_action(Action.idle())

    assert not terminated and not truncated
    assert env.simulator.step_index == 15
    assert env.physics_steps_taken == 15
    assert env.decisions_taken == 1
    assert info["metrics"]["physical_steps"] == 15
    assert info["metrics"]["decisions"] == 1
    assert info["metrics"]["hours"] == pytest.approx(0.25)


def test_vector_and_controller_doors_share_the_same_transition_path() -> None:
    vector_env = PlantEnvV2(episode=SHORT)
    controller_env = PlantEnvV2(episode=SHORT)
    vector_env.reset(seed=9)
    controller_env.reset(seed=9)
    action = Action(
        battery_share=0.25,
        electrolyser_share=0.5,
        dac_share=0.125,
        methanator_share=0.125,
        battery_support_kw=625.0,
    )
    vector = vector_env.adapter.encode(action, constraints=vector_env.constraints)

    vector_result = vector_env.step(vector)
    controller_result = controller_env.step_action(action)
    assert vector_result[0] == pytest.approx(controller_result[0], abs=1e-7)
    assert vector_result[1] == pytest.approx(controller_result[1], abs=1e-12)
    assert vector_result[2:4] == controller_result[2:4]
    assert vector_result[4]["reward"] == controller_result[4]["reward"]
    assert vector_result[4]["metrics"] == controller_result[4]["metrics"]


def test_raw_production_before_a_deadline_has_no_positive_reward_component() -> None:
    env = PlantEnvV2(episode=SHORT)
    env.reset(seed=5)
    vector = np.asarray([-1.0, -1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    _, reward, _, truncated, info = env.step(vector)

    assert not truncated
    assert info["reward"]["signals"]["raw_production_kg"] > 0.0
    assert info["reward"]["components"]["delivery"] == 0.0
    assert info["reward"]["components"]["terminal_inventory"] == 0.0
    assert reward <= 0.0
    assert sum(info["reward"]["components"].values()) == pytest.approx(reward)


def test_delivery_shortfall_drives_service_metrics_and_terminal_ordering() -> None:
    episode = EpisodeConfigV2(
        horizon_h=1.0,
        delivery_period_h=1.0,
        ch4_coverage_h=(0.0, 0.0),
    )
    env = PlantEnvV2(episode=episode)
    env.reset(seed=1)
    while True:
        _, reward, terminated, truncated, info = env.step_action(Action.idle())
        assert not terminated
        if truncated:
            break

    metrics = info["metrics"]
    assert info["delivery"]["commitments_settled"] == 1
    assert metrics["commitments_settled"] == 1
    assert metrics["commitments_met"] == 0
    assert metrics["requested_kg"] == pytest.approx(55.0)
    assert metrics["delivered_kg"] == 0.0
    assert metrics["unmet_kg"] == pytest.approx(55.0)
    assert metrics["delivery_fill_rate"] == 0.0
    assert metrics["commitment_service_level"] == 0.0
    assert info["reward"]["components"]["shortfall"] == pytest.approx(-4.0)
    # Deadline settlement happens before terminal inventory valuation.
    assert info["reward"]["components"]["terminal_inventory"] == 0.0
    assert reward < 0.0


def test_observations_remain_bounded_through_delivery_boundaries() -> None:
    env = PlantEnvV2(episode=SHORT)
    observation, _ = env.reset(seed=8)
    assert env.observation_space.contains(observation)
    truncated = False
    while not truncated:
        observation, _, _, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(observation)
        assert np.all(np.isfinite(observation))


def test_same_seed_and_actions_replay_exactly() -> None:
    actions = [
        np.linspace(-1.0, 1.0, v2_spaces.ACTION_SIZE, dtype=np.float32),
        np.zeros(v2_spaces.ACTION_SIZE, dtype=np.float32),
    ] * 4

    def rollout() -> tuple[list[list[float]], list[float], dict[str, object], str]:
        env = PlantEnvV2(episode=SHORT)
        observation, _ = env.reset(seed=101)
        observations = [observation.tolist()]
        rewards: list[float] = []
        for action in actions:
            observation, reward, _, _, _ = env.step(action)
            observations.append(observation.tolist())
            rewards.append(reward)
        manifest = env.audit_manifest()
        return observations, rewards, env.metrics.to_dict(), manifest["manifest_sha256"]

    assert rollout() == rollout()


def test_private_audit_manifest_contains_replay_data_and_detached_accounting() -> None:
    env = PlantEnvV2(episode=SHORT)
    env.reset(seed=23)
    before = env.audit_manifest()
    assert before["episode"]["seed"] == 23
    assert len(before["episode"]["commitments"]) == 2
    assert before["delivery_accounting"]["pending_count"] == 2
    assert before["delivery_accounting"]["frozen"] is True
    digest = before["manifest_sha256"]
    assert (
        before["checkpoint_contract"]["checkpoint_contract_digest"]
        == env.checkpoint_contract_digest
    )

    for _ in range(4):
        env.step_action(Action.idle())
    after = env.audit_manifest()
    assert after["manifest_sha256"] == digest
    assert after["delivery_accounting"]["pending_count"] < 2


def test_v2_delivery_schedule_cannot_be_replaced_mid_episode() -> None:
    env = PlantEnvV2(episode=SHORT)
    env.reset(seed=31)
    env.step_action(Action.idle())
    before = env.simulator.delivery_audit()

    with pytest.raises(RuntimeError, match="frozen"):
        env.simulator.set_delivery_schedule([])

    assert env.simulator.delivery_audit() == before

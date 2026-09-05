"""Focused software evidence for the beta constrained-PPO core."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Callable, cast

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest

from plant_backend.rl.v2.beta_constraint_transition import (
    CycleConstraintTransitionV1,
)
from plant_backend.rl.v2.training.constrained_ppo import (
    BETA_DECISIONS_PER_CYCLE,
    BETA_DECISIONS_PER_EPISODE,
    BETA_EPISODE_CYCLE_COUNT,
    CONSTRAINT_STATE_ENVELOPE_SCHEMA,
    CONSTRAINT_STATE_SCHEMA,
    I_LAGRANGIAN_ALGORITHM_ID,
    PI_LAGRANGIAN_ALGORITHM_ID,
    PROTOCOL_ID,
    RESOURCE_ORDER,
    SCALAR_PPO_ALGORITHM_ID,
    SUPPORTED_ALGORITHM_IDS,
    TRANSITION_INFO_KEY,
    ValidatedConstraintStateAttestationV1,
    CompletedEpisodeCostV1,
    ConstrainedActorCriticPolicyV1,
    ConstrainedPPO,
    ConstrainedPPOSettingsV1,
    EpisodeCostAccumulatorV1,
    FourHeadCostCriticV1,
    PILagrangeControllerV1,
    PILagrangeStateV1,
    RolloutLedgerV1,
    actor_reward_parameter_sha256,
    combine_advantages,
    compute_vector_gae,
    extract_constraint_transition,
    normalize_advantage,
    training_dependencies_available,
)


FIXTURE_SEED = 9_000_101


def _resign_constraint_payload(data: dict[str, Any], payload: dict[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    data["_embedded_constraint_state_payload_json"] = encoded
    data["_embedded_constraint_state_sha256"] = hashlib.sha256(
        CONSTRAINT_STATE_ENVELOPE_SCHEMA.encode("ascii")
        + b"\x00"
        + encoded.encode("utf-8")
    ).hexdigest()


def _transition(
    decision_index: int,
    *,
    raw_cost: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    cycle_index: int | None = None,
    episode_cycle_count: int = BETA_EPISODE_CYCLE_COUNT,
) -> dict[str, Any]:
    settled = cycle_index is not None
    record = CycleConstraintTransitionV1(
        decision_index=decision_index,
        cycle_settled=settled,
        cycle_index=cycle_index,
        episode_cycle_count=episode_cycle_count,
        resource_order=RESOURCE_ORDER,
        raw_cost=raw_cost,
        training_cost=(
            raw_cost[0] / episode_cycle_count,
            raw_cost[1] / episode_cycle_count,
            raw_cost[2] / episode_cycle_count,
            raw_cost[3] / episode_cycle_count,
        ),
    )
    return record.to_dict()


def test_frozen_settings_round_trip_all_three_algorithm_families() -> None:
    for algorithm_id in SUPPORTED_ALGORITHM_IDS:
        settings = ConstrainedPPOSettingsV1.for_algorithm(
            algorithm_id,
            optimizer_seed=101,
        )

        assert ConstrainedPPOSettingsV1.from_dict(settings.to_dict()) == settings
        assert settings.to_dict()["protocol_id"] == PROTOCOL_ID
        assert settings.cost_network_seed == 10_000_101
        assert settings.resource_order == ("battery", "h2", "co2", "ch4")

    pi = ConstrainedPPOSettingsV1.for_algorithm(
        PI_LAGRANGIAN_ALGORITHM_ID,
        optimizer_seed=101,
    )
    integral = ConstrainedPPOSettingsV1.for_algorithm(
        I_LAGRANGIAN_ALGORITHM_ID,
        optimizer_seed=101,
    )
    scalar = ConstrainedPPOSettingsV1.for_algorithm(
        SCALAR_PPO_ALGORITHM_ID,
        optimizer_seed=101,
    )
    assert (pi.proportional_gain, pi.integral_gain) == (0.25, 0.01)
    assert (integral.proportional_gain, integral.integral_gain) == (0.0, 0.01)
    assert scalar.uses_constraints is False
    assert scalar.initial_integral == (0.0, 0.0, 0.0, 0.0)


def test_settings_fail_closed_on_cost_or_algorithm_substitution() -> None:
    settings = ConstrainedPPOSettingsV1.for_algorithm(
        PI_LAGRANGIAN_ALGORITHM_ID,
        optimizer_seed=101,
    )

    with pytest.raises(ValueError, match="cost_gamma"):
        replace(settings, cost_gamma=0.99).validate()
    with pytest.raises(ValueError, match="resource_order"):
        replace(
            settings,
            resource_order=("h2", "battery", "co2", "ch4"),
        ).validate()
    raw = settings.to_dict()
    raw["cost_network_seed"] += 1
    with pytest.raises(ValueError, match="cost_network_seed"):
        ConstrainedPPOSettingsV1.from_dict(raw)


def test_constraint_state_attestation_rejects_ordinary_construction() -> None:
    with pytest.raises(RuntimeError, match="issued only by ConstrainedPPO.load"):
        ValidatedConstraintStateAttestationV1(
            artifact_sha256="a" * 64,
            algorithm_id=PI_LAGRANGIAN_ALGORITHM_ID,
            num_timesteps=0,
            constraint_state={},
            checkpoint_observation_sha256="b" * 64,
        )


def test_vector_gae_distinguishes_rollout_cut_from_true_terminal() -> None:
    signals = np.zeros((3, 1, 4), dtype=np.float64)
    signals[1, 0, 0] = 1.0
    values = np.zeros_like(signals)
    values[:, 0, 0] = [0.5, 0.4, 0.3]
    starts = np.asarray([[1.0], [0.0], [0.0]])
    last_values = np.asarray([[0.2, 0.0, 0.0, 0.0]])

    cut_advantage, cut_return = compute_vector_gae(
        signals,
        values,
        starts,
        last_values,
        np.asarray([False]),
        gamma=1.0,
        gae_lambda=1.0,
    )
    terminal_advantage, terminal_return = compute_vector_gae(
        signals,
        values,
        starts,
        last_values,
        np.asarray([True]),
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert cut_return[:, 0, 0] == pytest.approx([1.2, 1.2, 0.2])
    assert cut_advantage[:, 0, 0] == pytest.approx([0.7, 0.8, -0.1])
    assert terminal_return[:, 0, 0] == pytest.approx([1.0, 1.0, 0.0])
    assert terminal_advantage[:, 0, 0] == pytest.approx([0.5, 0.6, -0.3])
    assert np.count_nonzero(cut_return[:, :, 1:]) == 0


def test_vector_gae_stops_at_an_internal_episode_boundary() -> None:
    signals = np.zeros((4, 1, 4), dtype=np.float64)
    signals[0, 0, 0] = 2.0
    signals[3, 0, 0] = 7.0
    values = np.zeros_like(signals)
    starts = np.asarray([[1.0], [0.0], [1.0], [0.0]])

    _, returns = compute_vector_gae(
        signals,
        values,
        starts,
        np.zeros((1, 4)),
        np.asarray([True]),
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert returns[:, 0, 0] == pytest.approx([2.0, 0.0, 7.0, 7.0])


def test_advantages_are_independently_sample_normalized_and_rescaled() -> None:
    reward = np.asarray([1.0, 2.0, 3.0])
    costs = np.column_stack(
        (
            np.asarray([3.0, 2.0, 1.0]),
            np.asarray([4.0, 4.0, 4.0]),
            np.asarray([1.0, 3.0, 2.0]),
            np.asarray([0.0, 0.0, 0.0]),
        )
    )
    multipliers = (2.0, 100.0, 0.5, 0.0)

    combined = combine_advantages(reward, costs, multipliers)
    expected = (
        normalize_advantage(reward)
        - 2.0 * normalize_advantage(costs[:, 0])
        - 100.0 * np.zeros(3)
        - 0.5 * normalize_advantage(costs[:, 2])
    ) / 103.5

    assert combined == pytest.approx(expected)
    assert np.array_equal(normalize_advantage([9.0, 9.0]), np.zeros(2))
    assert np.array_equal(
        combine_advantages(reward, costs, multipliers, constrained=False),
        normalize_advantage(reward),
    )


def test_pi_and_integral_updates_are_channel_independent() -> None:
    pi = PILagrangeControllerV1(
        ConstrainedPPOSettingsV1.for_algorithm(
            PI_LAGRANGIAN_ALGORITHM_ID,
            optimizer_seed=101,
        )
    )
    initial = pi.state
    assert pi.update([]) is initial

    state = pi.update([(2.0, 0.0, 0.0, 0.0), (4.0, 0.0, 0.0, 0.0)])
    assert state.proportional == pytest.approx((0.15, 0.0, 0.0, 0.0))
    assert state.integral == pytest.approx((1.03, 1.0, 1.0, 1.0))
    assert state.multipliers == pytest.approx((1.0675, 1.0, 1.0, 1.0))
    assert state.completed_update_count == 1

    integral = PILagrangeControllerV1(
        ConstrainedPPOSettingsV1.for_algorithm(
            I_LAGRANGIAN_ALGORITHM_ID,
            optimizer_seed=101,
        )
    )
    integral_state = integral.update([(2.0, 0.0, 0.0, 0.0)])
    assert integral_state.proportional == pytest.approx((0.1, 0.0, 0.0, 0.0))
    assert integral_state.integral == pytest.approx((1.02, 1.0, 1.0, 1.0))
    assert integral_state.multipliers == integral_state.integral

    scalar_settings = ConstrainedPPOSettingsV1.for_algorithm(
        SCALAR_PPO_ALGORITHM_ID,
        optimizer_seed=101,
    )
    with pytest.raises(ValueError, match="scalar PPO"):
        PILagrangeControllerV1(
            scalar_settings,
            PILagrangeStateV1(
                proportional=(0.0, 0.0, 0.0, 0.0),
                integral=(0.0, 0.0, 0.0, 0.0),
                multipliers=(0.0, 0.0, 0.0, 0.0),
                completed_update_count=1,
            ),
        )


def test_transition_extraction_rejects_tampering_and_cumulative_substitution() -> None:
    transition = _transition(
        BETA_DECISIONS_PER_CYCLE,
        raw_cost=(0.2, 0.4, 0.0, 0.8),
        cycle_index=1,
    )
    info = {
        TRANSITION_INFO_KEY: transition,
        "cycle_constraint": {"cumulative_constraint_cost": 999_999.0},
    }

    extracted = extract_constraint_transition(info)

    assert extracted.raw_cost == (0.2, 0.4, 0.0, 0.8)
    assert extracted.training_cost == pytest.approx(
        (0.2 / 7.0, 0.4 / 7.0, 0.0, 0.8 / 7.0)
    )
    tampered = json.loads(json.dumps(transition))
    tampered["training_cost"][0] = 99.0
    with pytest.raises(ValueError, match="training cost"):
        extract_constraint_transition({TRANSITION_INFO_KEY: tampered})

    substituted = _transition(1)
    substituted["episode_cycle_count"] = 2
    with pytest.raises(ValueError, match="exactly seven"):
        extract_constraint_transition({TRANSITION_INFO_KEY: substituted})

    wrong_boundary = _transition(BETA_DECISIONS_PER_CYCLE, cycle_index=2)
    with pytest.raises(ValueError, match="does not match cycle_index"):
        extract_constraint_transition({TRANSITION_INFO_KEY: wrong_boundary})
    outside_episode = _transition(
        BETA_DECISIONS_PER_EPISODE + BETA_DECISIONS_PER_CYCLE,
        cycle_index=BETA_EPISODE_CYCLE_COUNT + 1,
    )
    with pytest.raises(ValueError, match="episode range"):
        extract_constraint_transition({TRANSITION_INFO_KEY: outside_episode})


def test_episode_accumulator_spans_rollouts_and_emits_terminal_once() -> None:
    accumulator = EpisodeCostAccumulatorV1(1)
    for decision in range(1, BETA_DECISIONS_PER_CYCLE):
        transition = extract_constraint_transition(
            {TRANSITION_INFO_KEY: _transition(decision)}
        )
        assert accumulator.observe([transition], np.asarray([False])) == ()

    before_settlement = accumulator.to_dict()
    forged_pre_settlement = json.loads(json.dumps(before_settlement))
    forged_pre_settlement["raw_cost"][0][0] = 0.7
    forged_pre_settlement["training_cost"][0][0] = 0.1
    with pytest.raises(ValueError, match="pre-settlement"):
        EpisodeCostAccumulatorV1.from_dict(forged_pre_settlement)
    missing = extract_constraint_transition(
        {TRANSITION_INFO_KEY: _transition(BETA_DECISIONS_PER_CYCLE)}
    )
    with pytest.raises(RuntimeError, match="settlement presence"):
        EpisodeCostAccumulatorV1.from_dict(before_settlement).observe(
            [missing], np.asarray([False])
        )
    with pytest.raises(ValueError, match="does not match cycle_index"):
        extract_constraint_transition(
            {
                TRANSITION_INFO_KEY: _transition(
                    BETA_DECISIONS_PER_CYCLE,
                    cycle_index=2,
                )
            }
        )

    first_settlement = extract_constraint_transition(
        {
            TRANSITION_INFO_KEY: _transition(
                BETA_DECISIONS_PER_CYCLE,
                raw_cost=(0.4, 0.0, 0.0, 0.0),
                cycle_index=1,
            )
        }
    )
    assert accumulator.observe([first_settlement], np.asarray([False])) == ()
    restored = EpisodeCostAccumulatorV1.from_dict(accumulator.to_dict())
    completed: tuple[CompletedEpisodeCostV1, ...] = ()
    for decision in range(BETA_DECISIONS_PER_CYCLE + 1, 673):
        is_boundary = decision % BETA_DECISIONS_PER_CYCLE == 0
        cycle_index = decision // BETA_DECISIONS_PER_CYCLE if is_boundary else None
        raw_cost = (
            (0.0, 0.6, 0.0, 0.0)
            if decision == BETA_DECISIONS_PER_EPISODE
            else (0.0, 0.0, 0.0, 0.0)
        )
        transition = extract_constraint_transition(
            {
                TRANSITION_INFO_KEY: _transition(
                    decision,
                    raw_cost=raw_cost,
                    cycle_index=cycle_index,
                )
            }
        )
        completed = restored.observe(
            [transition],
            np.asarray([decision == BETA_DECISIONS_PER_EPISODE]),
        )

    assert completed == (
        CompletedEpisodeCostV1(
            environment_index=0,
            raw_cost=(0.4, 0.6, 0.0, 0.0),
            training_cost=(0.4 / 7.0, 0.6 / 7.0, 0.0, 0.0),
            settled_cycle_count=7,
            expected_cycle_count=7,
        ),
    )
    with pytest.raises(RuntimeError, match="discontinuity"):
        restored.observe([transition], np.asarray([False]))


def test_rollout_ledger_is_hash_chained_and_tamper_evident() -> None:
    controller = PILagrangeControllerV1(
        ConstrainedPPOSettingsV1.for_algorithm(
            PI_LAGRANGIAN_ALGORITHM_ID,
            optimizer_seed=101,
        )
    )
    state = controller.update([(0.1, 0.0, 0.0, 0.0)])
    first = RolloutLedgerV1.build(
        algorithm_id=PI_LAGRANGIAN_ALGORITHM_ID,
        rollout_index=1,
        transition_count=8,
        completed_episode_count=1,
        complete_cycle_count=2,
        reward_return=3.0,
        raw_cost=(0.2, 0.0, 0.0, 0.0),
        training_cost=(0.1, 0.0, 0.0, 0.0),
        mean_completed_episode_cost=(0.1, 0.0, 0.0, 0.0),
        cost_value_losses=(0.3, 0.2, 0.1, 0.0),
        multiplier_state=state,
        multiplier_min=0.0,
        multiplier_max=100.0,
        approximate_kl=0.01,
        entropy_loss=-1.0,
        clip_fraction=0.25,
        optimizer_steps=2,
        elapsed_compute_s=0.1,
    )
    first.verify(expected_previous_sha256=None)
    second = RolloutLedgerV1.build(
        algorithm_id=PI_LAGRANGIAN_ALGORITHM_ID,
        rollout_index=2,
        transition_count=8,
        completed_episode_count=0,
        complete_cycle_count=0,
        reward_return=1.0,
        raw_cost=(0.0, 0.0, 0.0, 0.0),
        training_cost=(0.0, 0.0, 0.0, 0.0),
        mean_completed_episode_cost=None,
        cost_value_losses=(0.3, 0.2, 0.1, 0.0),
        multiplier_state=state,
        multiplier_min=0.0,
        multiplier_max=100.0,
        approximate_kl=0.01,
        entropy_loss=-1.0,
        clip_fraction=0.0,
        optimizer_steps=2,
        elapsed_compute_s=0.1,
        previous_sha256=first.sha256,
    )
    second.verify(expected_previous_sha256=first.sha256)
    assert RolloutLedgerV1.from_dict(first.to_dict()) == first

    tampered = first.to_dict()
    tampered["reward_return"] = 4.0
    with pytest.raises(ValueError, match="digest"):
        RolloutLedgerV1.from_dict(tampered)
    with pytest.raises(ValueError, match="zero discontinuities"):
        RolloutLedgerV1.build(
            algorithm_id=PI_LAGRANGIAN_ALGORITHM_ID,
            rollout_index=2,
            transition_count=8,
            completed_episode_count=0,
            complete_cycle_count=0,
            reward_return=1.0,
            raw_cost=(0.0, 0.0, 0.0, 0.0),
            training_cost=(0.0, 0.0, 0.0, 0.0),
            mean_completed_episode_cost=None,
            cost_value_losses=(0.0, 0.0, 0.0, 0.0),
            multiplier_state=state,
            multiplier_min=0.0,
            multiplier_max=100.0,
            approximate_kl=0.0,
            entropy_loss=0.0,
            clip_fraction=0.0,
            optimizer_steps=1,
            elapsed_compute_s=0.1,
            rollout_discontinuities=1,
            previous_sha256=first.sha256,
        )


class _FixtureCostEnv(gym.Env[np.ndarray, np.ndarray]):
    """Deterministic seven-day task with exact 96-decision settlements."""

    observation_space = spaces.Box(-2.0, 2.0, shape=(3,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def __init__(self, *, settlement_raw_cost: float) -> None:
        self.settlement_raw_cost = settlement_raw_cost
        self.decision = 0
        self.reset_seeds: list[int] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        if seed is None:
            if not self.reset_seeds:
                raise AssertionError("first reset requires an allocator fixture seed")
            seed = self.reset_seeds[-1] + 1
        if not 9_000_000 <= seed <= 9_999_999:
            raise AssertionError(f"non-fixture reset seed: {seed}")
        super().reset(seed=seed)
        self.reset_seeds.append(seed)
        self.decision = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.decision += 1
        settled = self.decision % BETA_DECISIONS_PER_CYCLE == 0
        raw_cost = (
            (self.settlement_raw_cost, 0.0, 0.0, 0.0)
            if settled
            else (0.0, 0.0, 0.0, 0.0)
        )
        observation = np.asarray(
            [
                self.decision / BETA_DECISIONS_PER_EPISODE,
                float(action[0]),
                float(action[1]),
            ],
            dtype=np.float32,
        )
        return (
            observation,
            float(0.25 * action[0] - 0.1 * action[1]),
            self.decision == BETA_DECISIONS_PER_EPISODE,
            False,
            {
                TRANSITION_INFO_KEY: _transition(
                    self.decision,
                    raw_cost=raw_cost,
                    cycle_index=(
                        self.decision // BETA_DECISIONS_PER_CYCLE if settled else None
                    ),
                )
            },
        )


class _AdversarialFixtureCostEnv(_FixtureCostEnv):
    """One-transition fixture for finite-value and truncation fail-closed tests."""

    def __init__(self, *, failure: str) -> None:
        super().__init__(settlement_raw_cost=0.7)
        self.failure = failure

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        if self.failure == "initial-observation":
            observation[0] = np.nan
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        if self.failure == "next-observation":
            observation[0] = np.nan
        elif self.failure == "reward":
            reward = float("nan")
        elif self.failure == "truncation":
            truncated = True
        return observation, reward, terminated, truncated, info


def _task_factory(task: _FixtureCostEnv) -> Callable[[], _FixtureCostEnv]:
    def build() -> _FixtureCostEnv:
        return task

    return build


def _close_model_env(model: ConstrainedPPO) -> None:
    environment = model.get_env()
    assert environment is not None
    environment.close()


def _model(
    algorithm_id: str,
    *,
    settlement_raw_cost: float = 0.7,
) -> tuple[ConstrainedPPO, _FixtureCostEnv]:
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("torch")
    from stable_baselines3.common.vec_env import DummyVecEnv

    task = _FixtureCostEnv(settlement_raw_cost=settlement_raw_cost)
    environment = DummyVecEnv([_task_factory(task)])
    settings = ConstrainedPPOSettingsV1.for_algorithm(
        algorithm_id,
        optimizer_seed=101,
    )
    model = ConstrainedPPO(
        "MlpPolicy",
        environment,
        n_steps=BETA_DECISIONS_PER_CYCLE,
        batch_size=BETA_DECISIONS_PER_CYCLE,
        n_epochs=1,
        constraint_settings=settings,
        seed=101,
        verbose=0,
    )
    environment.seed(FIXTURE_SEED)
    return model, task


def _model_for_task(
    task: _FixtureCostEnv,
    *,
    algorithm_id: str = SCALAR_PPO_ALGORITHM_ID,
) -> ConstrainedPPO:
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("torch")
    from stable_baselines3.common.vec_env import DummyVecEnv

    environment = DummyVecEnv([_task_factory(task)])
    model = ConstrainedPPO(
        "MlpPolicy",
        environment,
        n_steps=BETA_DECISIONS_PER_CYCLE,
        batch_size=BETA_DECISIONS_PER_CYCLE,
        n_epochs=1,
        constraint_settings=ConstrainedPPOSettingsV1.for_algorithm(
            algorithm_id,
            optimizer_seed=101,
        ),
        seed=101,
        verbose=0,
    )
    environment.seed(FIXTURE_SEED)
    return model


def _four_environment_model() -> tuple[ConstrainedPPO, list[_FixtureCostEnv]]:
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("torch")
    from stable_baselines3.common.vec_env import DummyVecEnv

    tasks = [
        _FixtureCostEnv(settlement_raw_cost=0.7 + 0.1 * index) for index in range(4)
    ]
    environment = DummyVecEnv([_task_factory(task) for task in tasks])
    settings = ConstrainedPPOSettingsV1.for_algorithm(
        PI_LAGRANGIAN_ALGORITHM_ID,
        optimizer_seed=101,
    )
    model = ConstrainedPPO(
        "MlpPolicy",
        environment,
        n_steps=BETA_DECISIONS_PER_CYCLE,
        batch_size=4 * BETA_DECISIONS_PER_CYCLE,
        n_epochs=1,
        constraint_settings=settings,
        seed=101,
        verbose=0,
    )
    environment.seed(FIXTURE_SEED)
    return model, tasks


def _assert_nested_equal(left: Any, right: Any) -> None:
    """Assert exact equality for Torch/optimizer state trees."""

    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover - guarded by caller skip
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
        return
    assert left == right


@pytest.mark.skipif(
    not training_dependencies_available(), reason="optional train dependencies absent"
)
def test_cost_network_rng_isolation_preserves_actor_and_reward_initialization() -> None:
    digests = []
    for algorithm_id in SUPPORTED_ALGORITHM_IDS:
        model, _ = _model(algorithm_id)
        digests.append(actor_reward_parameter_sha256(model))
        _close_model_env(model)

    assert len(set(digests)) == 1


@pytest.mark.skipif(
    not training_dependencies_available(), reason="optional train dependencies absent"
)
def test_each_cost_target_has_disjoint_gradient_and_update_ownership() -> None:
    import torch

    torch.manual_seed(17)
    critic = FourHeadCostCriticV1(3)
    inputs = torch.tensor(
        [[0.25, -0.5, 0.75], [-0.1, 0.2, 0.3]],
        dtype=torch.float32,
    )
    before = {
        name: value.detach().clone() for name, value in critic.state_dict().items()
    }
    optimizer = torch.optim.SGD(critic.parameters(), lr=0.05)
    target = torch.tensor([2.0, -1.0], dtype=torch.float32)
    loss = torch.nn.functional.mse_loss(critic(inputs)[:, 0], target)
    optimizer.zero_grad()
    loss.backward()

    for index, head in enumerate(critic.heads):
        gradients = [parameter.grad for parameter in head.parameters()]
        if index == 0:
            assert any(
                gradient is not None and torch.count_nonzero(gradient) > 0
                for gradient in gradients
            )
        else:
            assert all(
                gradient is None or torch.count_nonzero(gradient) == 0
                for gradient in gradients
            )
    optimizer.step()

    after = critic.state_dict()
    assert any(
        not torch.equal(before[name], after[name])
        for name in before
        if name.startswith("heads.0.")
    )
    assert all(
        torch.equal(before[name], after[name])
        for name in before
        if not name.startswith("heads.0.")
    )


@pytest.mark.skipif(
    not training_dependencies_available(), reason="optional train dependencies absent"
)
def test_live_actor_module_graph_rejects_hidden_activation_substitution(
    tmp_path: Path,
) -> None:
    import torch

    model, task = _model(SCALAR_PPO_ALGORITHM_ID)
    # Metadata still says Tanh and parameter names/shapes remain unchanged.
    model.policy.mlp_extractor.policy_net[1] = torch.nn.ReLU()

    with pytest.raises(ValueError, match="actor hidden network module topology"):
        model.learn(total_timesteps=BETA_DECISIONS_PER_CYCLE)
    assert model.num_timesteps == 0
    assert not task.reset_seeds
    model.pin_external_episode_state(
        {"allocator_snapshot": "activation-substitution", "decision_indices": [0]}
    )
    artifact = tmp_path / "activation-substitution-must-not-exist.zip"
    with pytest.raises(ValueError, match="actor hidden network module topology"):
        model.save(artifact)
    assert not artifact.exists()
    _close_model_env(model)


@pytest.mark.skipif(
    not training_dependencies_available(), reason="optional train dependencies absent"
)
def test_scalar_family_is_parameter_and_optimizer_equivalent_to_stock_sb3() -> None:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.buffers import RolloutBuffer
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.vec_env import DummyVecEnv

    stock_task = _FixtureCostEnv(settlement_raw_cost=0.7)
    stock_env = DummyVecEnv([lambda: stock_task])
    stock = PPO(
        "MlpPolicy",
        stock_env,
        n_steps=BETA_DECISIONS_PER_CYCLE,
        batch_size=BETA_DECISIONS_PER_CYCLE,
        n_epochs=1,
        gamma=0.995,
        policy_kwargs={
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "activation_fn": torch.nn.Tanh,
        },
        seed=101,
        device="cpu",
        verbose=0,
    )
    stock_env.seed(FIXTURE_SEED)
    stock_initial = actor_reward_parameter_sha256(stock)
    stock.learn(total_timesteps=BETA_DECISIONS_PER_CYCLE)

    scalar, scalar_task = _model(SCALAR_PPO_ALGORITHM_ID)
    assert type(scalar.policy) is ActorCriticPolicy
    assert type(scalar.rollout_buffer) is RolloutBuffer
    assert actor_reward_parameter_sha256(scalar) == stock_initial
    scalar.learn(total_timesteps=BETA_DECISIONS_PER_CYCLE)

    _assert_nested_equal(stock.policy.state_dict(), scalar.policy.state_dict())
    _assert_nested_equal(
        stock.policy.optimizer.state_dict(),
        scalar.policy.optimizer.state_dict(),
    )
    assert stock_task.reset_seeds == [FIXTURE_SEED]
    assert scalar_task.reset_seeds == [FIXTURE_SEED]
    stock_env.close()
    _close_model_env(scalar)


@pytest.mark.skipif(
    not training_dependencies_available(), reason="optional train dependencies absent"
)
def test_scalar_ppo_gradient_is_invariant_to_changed_cost_evidence() -> None:
    without_cost, task_without = _model(
        SCALAR_PPO_ALGORITHM_ID,
        settlement_raw_cost=0.0,
    )
    without_cost.learn(total_timesteps=BETA_DECISIONS_PER_CYCLE)
    digest_without_cost = actor_reward_parameter_sha256(without_cost)

    # Constructing the matched arm after the first run resets the SB3/Torch RNG
    # to the same optimizer seed immediately before its rollout.
    with_cost, task_with = _model(
        SCALAR_PPO_ALGORITHM_ID,
        settlement_raw_cost=0.7,
    )
    with_cost.learn(total_timesteps=BETA_DECISIONS_PER_CYCLE)

    assert digest_without_cost == actor_reward_parameter_sha256(with_cost)
    assert with_cost.rollout_ledgers[0].training_cost[0] == pytest.approx(0.1)
    assert without_cost.rollout_ledgers[0].training_cost[0] == 0.0
    assert task_without.reset_seeds == [FIXTURE_SEED]
    assert task_with.reset_seeds == [FIXTURE_SEED]
    _close_model_env(without_cost)
    _close_model_env(with_cost)


@pytest.mark.skipif(
    not training_dependencies_available(), reason="optional train dependencies absent"
)
def test_pi_model_updates_multiplier_only_after_full_seven_cycle_episode(
    tmp_path: Path,
) -> None:
    from stable_baselines3.common.save_util import (
        load_from_zip_file,
        save_to_zip_file,
    )

    model, _ = _model(PI_LAGRANGIAN_ALGORITHM_ID, settlement_raw_cost=0.7)
    model.learn(total_timesteps=BETA_DECISIONS_PER_EPISODE)

    assert len(model.rollout_ledgers) == BETA_EPISODE_CYCLE_COUNT
    assert [item.complete_cycle_count for item in model.rollout_ledgers] == [1] * 7
    assert [item.completed_episode_count for item in model.rollout_ledgers] == [
        0,
        0,
        0,
        0,
        0,
        0,
        1,
    ]
    assert [
        item.multiplier_completed_update_count for item in model.rollout_ledgers
    ] == [
        0,
        0,
        0,
        0,
        0,
        0,
        1,
    ]
    assert model.multiplier_state.proportional == pytest.approx((0.035, 0.0, 0.0, 0.0))
    assert model.multiplier_state.integral == pytest.approx((1.007, 1.0, 1.0, 1.0))
    assert model.multiplier_state.multipliers == pytest.approx((1.01575, 1.0, 1.0, 1.0))
    assert model.episode_decision_indices == (0,)
    complete_artifact = tmp_path / "complete-episode.zip"
    with pytest.raises(RuntimeError, match="requires pinned external state"):
        model.save(complete_artifact)
    pin = model.pin_external_episode_state(
        {
            "allocator_snapshot": "next-fixture-episode",
            "decision_indices": [0],
        }
    )
    model.save(complete_artifact)
    restored = ConstrainedPPO.load(complete_artifact)
    assert restored.multiplier_state == model.multiplier_state
    assert restored.constraint_state_dict() == model.constraint_state_dict()
    assert restored.external_episode_state_pin == pin
    assert restored.requires_external_episode_state_admission is True
    with pytest.raises(RuntimeError, match="external-state admission"):
        restored.learn(
            total_timesteps=BETA_DECISIONS_PER_CYCLE,
            reset_num_timesteps=False,
        )

    data, parameters, variables = load_from_zip_file(complete_artifact, device="cpu")
    assert data is not None and parameters is not None
    data["_external_episode_state_pin"] = None
    payload = json.loads(data["_embedded_constraint_state_payload_json"])
    payload["external_episode_state_pin"] = None
    _resign_constraint_payload(data, payload)
    pinless_artifact = tmp_path / "pinless-inference-only.zip"
    save_to_zip_file(
        pinless_artifact,
        data=data,
        params=parameters,
        pytorch_variables=variables,
    )
    inference_only = ConstrainedPPO.load(pinless_artifact)
    inference_only.predict(np.zeros(3, dtype=np.float32), deterministic=True)
    assert inference_only.requires_external_episode_state_admission is True
    with pytest.raises(RuntimeError, match="has no pinned external"):
        inference_only.admit_external_episode_state(
            {},
            decision_indices=[0],
            runtime_observation=np.zeros((1, 3), dtype=np.float32),
        )
    with pytest.raises(RuntimeError, match="external-state admission"):
        inference_only.learn(
            total_timesteps=BETA_DECISIONS_PER_CYCLE,
            reset_num_timesteps=False,
        )
    _close_model_env(model)


@pytest.mark.skipif(
    not training_dependencies_available(), reason="optional train dependencies absent"
)
def test_callback_abort_is_permanent_unledgered_evidence_and_forbids_save(
    tmp_path: Path,
) -> None:
    from stable_baselines3.common.callbacks import BaseCallback

    class StopImmediately(BaseCallback):
        def _on_step(self) -> bool:
            return False

    model, task = _model(SCALAR_PPO_ALGORITHM_ID)
    model.learn(
        total_timesteps=BETA_DECISIONS_PER_CYCLE,
        callback=StopImmediately(),
    )

    assert model.num_timesteps == 1
    assert model.rollout_ledgers == ()
    assert model.aborted_rollout is not None
    assert model.aborted_rollout["reason"] == "callback-returned-false"
    assert model.aborted_rollout["transition_count"] == 1
    with pytest.raises(RuntimeError, match="aborted rollout"):
        model.save(tmp_path / "must-not-exist.zip")
    assert not (tmp_path / "must-not-exist.zip").exists()
    assert task.reset_seeds == [FIXTURE_SEED]
    _close_model_env(model)


@pytest.mark.skipif(
    not training_dependencies_available(), reason="optional train dependencies absent"
)
def test_optimizer_exception_is_permanent_and_cannot_be_retried_or_saved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model, _ = _model(SCALAR_PPO_ALGORITHM_ID)
    original_step = model.policy.optimizer.step

    def step_then_fail(closure: Any = None) -> Any:
        original_step(closure)
        raise KeyboardInterrupt("injected interrupt after Adam mutation")

    monkeypatch.setattr(model.policy.optimizer, "step", step_then_fail)
    with pytest.raises(KeyboardInterrupt, match="after Adam mutation"):
        model.learn(total_timesteps=BETA_DECISIONS_PER_CYCLE)

    assert model.num_timesteps == BETA_DECISIONS_PER_CYCLE
    assert model.rollout_ledgers == ()
    assert model.aborted_rollout is not None
    assert model.aborted_rollout["reason"] == "optimizer-exception"
    assert model.aborted_rollout["transition_count"] == BETA_DECISIONS_PER_CYCLE
    with pytest.raises(RuntimeError, match="permanently blocks training"):
        model.train()
    with pytest.raises(RuntimeError, match="permanently blocks training"):
        model.learn(
            total_timesteps=BETA_DECISIONS_PER_CYCLE,
            reset_num_timesteps=False,
        )
    with pytest.raises(RuntimeError, match="aborted rollout"):
        model.pin_external_episode_state({"allocator_snapshot": "unsafe"})
    artifact = tmp_path / "optimizer-abort-must-not-exist.zip"
    with pytest.raises(RuntimeError, match="aborted rollout"):
        model.save(artifact)
    assert not artifact.exists()
    _close_model_env(model)


@pytest.mark.skipif(
    not training_dependencies_available(), reason="optional train dependencies absent"
)
@pytest.mark.parametrize(
    "failure, expected_message",
    [
        ("initial-observation", "rollout observation"),
        ("next-observation", "next rollout observation"),
        ("reward", "rollout rewards"),
        ("truncation", "cannot use time-limit truncation"),
    ],
)
def test_collection_rejects_nonfinite_environment_data_and_truncation(
    failure: str,
    expected_message: str,
) -> None:
    model = _model_for_task(_AdversarialFixtureCostEnv(failure=failure))

    with pytest.raises((ValueError, RuntimeError), match=expected_message):
        model.learn(total_timesteps=BETA_DECISIONS_PER_CYCLE)

    if failure == "initial-observation":
        assert model.aborted_rollout is None
    else:
        assert model.aborted_rollout is not None
        assert model.aborted_rollout["transition_count"] == 1
    _close_model_env(model)


@pytest.mark.skipif(
    not training_dependencies_available(), reason="optional train dependencies absent"
)
@pytest.mark.parametrize(
    "field, expected_message",
    [
        ("actions", "policy actions"),
        ("values", "policy values"),
        ("log_prob", "policy log probabilities"),
    ],
)
def test_collection_rejects_nonfinite_stock_policy_outputs(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    expected_message: str,
) -> None:
    import torch

    model, _ = _model(SCALAR_PPO_ALGORITHM_ID)
    original_forward = model.policy.forward

    def nonfinite_forward(
        observation: Any,
        deterministic: bool = False,
    ) -> tuple[Any, Any, Any]:
        actions, values, log_prob = original_forward(
            observation,
            deterministic=deterministic,
        )
        outputs = [actions.clone(), values.clone(), log_prob.clone()]
        index = {"actions": 0, "values": 1, "log_prob": 2}[field]
        outputs[index].reshape(-1)[0] = torch.nan
        return outputs[0], outputs[1], outputs[2]

    monkeypatch.setattr(model.policy, "forward", nonfinite_forward)
    with pytest.raises(ValueError, match=expected_message):
        model.learn(total_timesteps=BETA_DECISIONS_PER_CYCLE)
    assert model.num_timesteps == 0
    assert model.aborted_rollout is None
    _close_model_env(model)


@pytest.mark.skipif(
    not training_dependencies_available(), reason="optional train dependencies absent"
)
def test_collection_rejects_nonfinite_cost_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    model, _ = _model(PI_LAGRANGIAN_ALGORITHM_ID)
    cost_policy = cast(ConstrainedActorCriticPolicyV1, model.policy)
    original = cost_policy.predict_cost_values

    def nonfinite_cost_values(observation: Any) -> Any:
        values = original(observation).clone()
        values.reshape(-1)[0] = torch.nan
        return values

    monkeypatch.setattr(cost_policy, "predict_cost_values", nonfinite_cost_values)
    with pytest.raises(ValueError, match="policy cost values"):
        model.learn(total_timesteps=BETA_DECISIONS_PER_CYCLE)
    assert model.num_timesteps == 0
    assert model.aborted_rollout is None
    _close_model_env(model)


@pytest.mark.skipif(
    not training_dependencies_available(), reason="optional train dependencies absent"
)
def test_partial_resume_requires_exact_external_state_and_continues_cursor(
    tmp_path: Path,
) -> None:
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv

    model, _ = _model(SCALAR_PPO_ALGORITHM_ID)
    model.learn(total_timesteps=BETA_DECISIONS_PER_CYCLE)
    artifact = tmp_path / "partial-scalar.zip"
    with pytest.raises(RuntimeError, match="pinned external state"):
        model.save(artifact)
    external = {
        "allocator_snapshot": "fixture-resume-state",
        "decision_indices": [BETA_DECISIONS_PER_CYCLE],
    }
    pin = model.pin_external_episode_state(external)
    model.save(artifact)
    saved_envelope = model.embedded_constraint_state

    # The uninterrupted arm starts from exactly the checkpoint cut. Saving is
    # required to be RNG-transparent, so it is also the exact comparison arm.
    model.learn(
        total_timesteps=BETA_DECISIONS_PER_CYCLE,
        reset_num_timesteps=False,
    )
    uninterrupted_policy = {
        name: tensor.detach().clone()
        for name, tensor in model.policy.state_dict().items()
    }
    uninterrupted_optimizer = copy.deepcopy(model.policy.optimizer.state_dict())
    uninterrupted_update_count = model._n_updates

    continued_task = _FixtureCostEnv(settlement_raw_cost=0.7)
    continued_task.decision = BETA_DECISIONS_PER_CYCLE
    continued_env = DummyVecEnv([lambda: continued_task])
    python_rng_before_load = random.getstate()
    numpy_rng_before_load = cast(
        tuple[str, np.ndarray[Any, Any], int, int, float],
        np.random.get_state(),
    )
    torch_rng_before_load = torch.random.get_rng_state().clone()
    restored = ConstrainedPPO.load(
        artifact,
        env=continued_env,
        force_reset=False,
    )
    assert random.getstate() == python_rng_before_load
    numpy_rng_after_load = cast(
        tuple[str, np.ndarray[Any, Any], int, int, float],
        np.random.get_state(),
    )
    assert numpy_rng_after_load[0] == numpy_rng_before_load[0]
    assert np.array_equal(numpy_rng_after_load[1], numpy_rng_before_load[1])
    assert numpy_rng_after_load[2:] == numpy_rng_before_load[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_rng_before_load)
    embedded_envelope = restored.embedded_constraint_state
    assert embedded_envelope == saved_envelope
    attestation = restored.loaded_constraint_state_attestation()
    assert type(attestation) is ValidatedConstraintStateAttestationV1
    assert (
        attestation.artifact_sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
    )
    assert attestation.algorithm_id == SCALAR_PPO_ALGORITHM_ID
    assert attestation.num_timesteps == BETA_DECISIONS_PER_CYCLE
    assert attestation.constraint_state == saved_envelope
    assert attestation.constraint_state_sha256 == saved_envelope["sha256"]
    assert random.getstate() == python_rng_before_load
    numpy_rng_after_envelope = cast(
        tuple[str, np.ndarray[Any, Any], int, int, float],
        np.random.get_state(),
    )
    assert numpy_rng_after_envelope[0] == numpy_rng_before_load[0]
    assert np.array_equal(numpy_rng_after_envelope[1], numpy_rng_before_load[1])
    assert numpy_rng_after_envelope[2:] == numpy_rng_before_load[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_rng_before_load)

    embedded_envelope["payload"]["num_timesteps"] = 0
    assert restored.embedded_constraint_state == saved_envelope
    substituted_metadata = copy.deepcopy(saved_envelope)
    substituted_metadata["payload"]["num_timesteps"] += 1
    assert restored.embedded_constraint_state != substituted_metadata
    assert restored.external_episode_state_pin == pin
    assert restored.requires_external_episode_state_admission is True
    checkpoint_observation = restored.checkpoint_observation
    with pytest.raises(RuntimeError, match="external-state admission"):
        restored.learn(
            total_timesteps=BETA_DECISIONS_PER_CYCLE,
            reset_num_timesteps=False,
        )
    with pytest.raises(ValueError, match="differs from checkpoint pin"):
        restored.admit_external_episode_state(
            {**external, "allocator_snapshot": "substituted"},
            decision_indices=[BETA_DECISIONS_PER_CYCLE],
            runtime_observation=checkpoint_observation,
        )
    with pytest.raises(ValueError, match="decision cursors"):
        restored.admit_external_episode_state(
            external,
            decision_indices=[BETA_DECISIONS_PER_CYCLE - 1],
            runtime_observation=checkpoint_observation,
        )

    substituted_observation = checkpoint_observation.copy()
    substituted_observation[0, 0] += np.float32(0.01)
    with pytest.raises(ValueError, match="observation differs"):
        restored.admit_external_episode_state(
            external,
            decision_indices=[BETA_DECISIONS_PER_CYCLE],
            runtime_observation=substituted_observation,
        )

    restored.admit_external_episode_state(
        external,
        decision_indices=[BETA_DECISIONS_PER_CYCLE],
        runtime_observation=checkpoint_observation,
    )
    with pytest.raises(RuntimeError, match="first resumed rollout"):
        restored.save(tmp_path / "unadvanced-resume.zip")
    restored.learn(
        total_timesteps=BETA_DECISIONS_PER_CYCLE,
        reset_num_timesteps=False,
    )

    assert restored.num_timesteps == 2 * BETA_DECISIONS_PER_CYCLE
    assert restored.episode_decision_indices == (2 * BETA_DECISIONS_PER_CYCLE,)
    assert len(restored.rollout_ledgers) == 2
    assert restored.rollout_ledgers[-1].complete_cycle_count == 1
    child_state = restored.constraint_state_dict()
    receipt = child_state["payload"]["resume_continuation_receipt"]
    assert receipt["parent_artifact_sha256"] == attestation.artifact_sha256
    assert (
        receipt["parent_constraint_state_sha256"] == attestation.constraint_state_sha256
    )
    assert (
        receipt["parent_checkpoint_observation_sha256"]
        == attestation.checkpoint_observation_sha256
    )
    assert (
        receipt["parent_policy_parameter_sha256"]
        == saved_envelope["payload"]["policy_parameter_sha256"]
    )
    assert (
        receipt["parent_optimizer_state_sha256"]
        == saved_envelope["payload"]["optimizer_state_sha256"]
    )
    assert receipt["first_resumed_rollout_index"] == 2
    assert receipt["first_resumed_rollout_sha256"] == restored.rollout_ledgers[1].sha256

    tampered_receipt_state = copy.deepcopy(child_state)
    tampered_receipt_state["payload"]["resume_continuation_receipt"][
        "first_resumed_rollout_sha256"
    ] = ("f" * 64)
    receipt_body = dict(
        tampered_receipt_state["payload"]["resume_continuation_receipt"]
    )
    receipt_body.pop("sha256")
    encoded_receipt = json.dumps(
        receipt_body,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    tampered_receipt_state["payload"]["resume_continuation_receipt"]["sha256"] = (
        hashlib.sha256(
            b"constrained-ppo-resume-continuation-receipt-v1\x00" + encoded_receipt
        ).hexdigest()
    )
    encoded_payload = json.dumps(
        tampered_receipt_state["payload"],
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    tampered_receipt_state["sha256"] = hashlib.sha256(
        CONSTRAINT_STATE_ENVELOPE_SCHEMA.encode("ascii") + b"\x00" + encoded_payload
    ).hexdigest()
    with pytest.raises(ValueError, match="receipt first rollout differs"):
        restored.load_constraint_state_dict(tampered_receipt_state)

    child_artifact = tmp_path / "resumed-child.zip"
    restored.pin_external_episode_state(
        {
            "allocator_snapshot": "fixture-resume-state-child",
            "decision_indices": [2 * BETA_DECISIONS_PER_CYCLE],
        }
    )
    saved_child_state = restored.constraint_state_dict()
    restored.save(child_artifact)
    reloaded_child = ConstrainedPPO.load(child_artifact, env=continued_env)
    assert reloaded_child.embedded_constraint_state == saved_child_state
    assert (
        reloaded_child.loaded_constraint_state_attestation().constraint_state
        == saved_child_state
    )
    assert continued_task.decision == 2 * BETA_DECISIONS_PER_CYCLE
    assert continued_task.reset_seeds == []
    assert restored._n_updates == uninterrupted_update_count
    assert all(
        torch.equal(tensor, restored.policy.state_dict()[name])
        for name, tensor in uninterrupted_policy.items()
    )
    _assert_nested_equal(
        uninterrupted_optimizer,
        restored.policy.optimizer.state_dict(),
    )
    assert [
        (
            ledger.reward_return,
            ledger.raw_cost,
            ledger.training_cost,
            ledger.complete_cycle_count,
            ledger.completed_episode_count,
            ledger.optimizer_steps,
        )
        for ledger in restored.rollout_ledgers
    ] == [
        (
            ledger.reward_return,
            ledger.raw_cost,
            ledger.training_cost,
            ledger.complete_cycle_count,
            ledger.completed_episode_count,
            ledger.optimizer_steps,
        )
        for ledger in model.rollout_ledgers
    ]
    _close_model_env(model)
    _close_model_env(reloaded_child)
    continued_env.close()


@pytest.mark.skipif(
    not training_dependencies_available(), reason="optional train dependencies absent"
)
def test_artifact_load_rejects_valid_range_pickle_and_envelope_substitutions(
    tmp_path: Path,
) -> None:
    import torch
    from stable_baselines3.common.save_util import (
        load_from_zip_file,
        save_to_zip_file,
    )

    model, _ = _model(PI_LAGRANGIAN_ALGORITHM_ID)
    model.learn(total_timesteps=BETA_DECISIONS_PER_CYCLE)
    model.pin_external_episode_state(
        {"allocator_snapshot": "tamper-fixture", "decision_indices": [96]}
    )
    artifact = tmp_path / "authentic.zip"
    model.save(artifact)

    data, parameters, variables = load_from_zip_file(artifact, device="cpu")
    assert data is not None and parameters is not None
    data["_multiplier_controller"].state = PILagrangeStateV1(
        proportional=(0.0, 0.0, 0.0, 0.0),
        integral=(1.1, 1.0, 1.0, 1.0),
        multipliers=(1.1, 1.0, 1.0, 1.0),
        completed_update_count=0,
    )
    tampered_pickle = tmp_path / "tampered-pickle.zip"
    save_to_zip_file(
        tampered_pickle,
        data=data,
        params=parameters,
        pytorch_variables=variables,
    )
    with pytest.raises(ValueError, match="embedded multiplier state"):
        ConstrainedPPO.load(tampered_pickle)

    data, parameters, variables = load_from_zip_file(artifact, device="cpu")
    assert data is not None and parameters is not None
    data["_n_updates"] = 0
    tampered_count = tmp_path / "tampered-count.zip"
    save_to_zip_file(
        tampered_count,
        data=data,
        params=parameters,
        pytorch_variables=variables,
    )
    with pytest.raises(ValueError, match="embedded PPO update count"):
        ConstrainedPPO.load(tampered_count)

    data, parameters, variables = load_from_zip_file(artifact, device="cpu")
    assert data is not None and parameters is not None
    payload = json.loads(data["_embedded_constraint_state_payload_json"])
    payload["multiplier_state"]["integral"][0] = 1.1
    payload["multiplier_state"]["multipliers"][0] = 1.1
    _resign_constraint_payload(data, payload)
    tampered_envelope = tmp_path / "tampered-envelope.zip"
    save_to_zip_file(
        tampered_envelope,
        data=data,
        params=parameters,
        pytorch_variables=variables,
    )
    with pytest.raises(ValueError, match="final multiplier state"):
        ConstrainedPPO.load(tampered_envelope)

    data, parameters, variables = load_from_zip_file(artifact, device="cpu")
    assert data is not None and parameters is not None
    data["gamma"] = 0.9
    tampered_ppo = tmp_path / "tampered-ppo-configuration.zip"
    save_to_zip_file(
        tampered_ppo,
        data=data,
        params=parameters,
        pytorch_variables=variables,
    )
    with pytest.raises(ValueError, match="PPO configuration"):
        ConstrainedPPO.load(tampered_ppo)

    data, parameters, variables = load_from_zip_file(artifact, device="cpu")
    assert data is not None and parameters is not None
    data["policy_kwargs"] = dict(data["policy_kwargs"])
    data["policy_kwargs"]["activation_fn"] = torch.nn.ReLU
    tampered_architecture = tmp_path / "tampered-policy-architecture.zip"
    save_to_zip_file(
        tampered_architecture,
        data=data,
        params=parameters,
        pytorch_variables=variables,
    )
    with pytest.raises(ValueError, match="policy activation"):
        ConstrainedPPO.load(tampered_architecture)

    data, parameters, variables = load_from_zip_file(artifact, device="cpu")
    assert data is not None and parameters is not None
    parameter_name = next(iter(parameters["policy"]))
    parameters["policy"][parameter_name] = (
        parameters["policy"][parameter_name].detach().clone()
    )
    parameters["policy"][parameter_name].reshape(-1)[0] += 0.25
    tampered_policy = tmp_path / "tampered-policy-parameter.zip"
    save_to_zip_file(
        tampered_policy,
        data=data,
        params=parameters,
        pytorch_variables=variables,
    )
    with pytest.raises(ValueError, match="policy parameters"):
        ConstrainedPPO.load(tampered_policy)

    data, parameters, variables = load_from_zip_file(artifact, device="cpu")
    assert data is not None and parameters is not None
    optimizer_archive = cast(dict[str, Any], parameters["policy.optimizer"])
    optimizer_states = cast(dict[Any, dict[str, Any]], optimizer_archive["state"])
    first_optimizer_state = next(iter(optimizer_states.values()))
    first_optimizer_state["exp_avg"] = first_optimizer_state["exp_avg"].clone()
    first_optimizer_state["exp_avg"].reshape(-1)[0] += 0.25
    tampered_optimizer = tmp_path / "tampered-optimizer-state.zip"
    save_to_zip_file(
        tampered_optimizer,
        data=data,
        params=parameters,
        pytorch_variables=variables,
    )
    with pytest.raises(ValueError, match="optimizer state"):
        ConstrainedPPO.load(tampered_optimizer)

    # Co-tamper the pickled live accumulator and its freshly re-signed JSON
    # mirror.  Cross-checking the mirrors alone would accept this; cumulative
    # ledger reconciliation must still reject the invented cost.
    data, parameters, variables = load_from_zip_file(artifact, device="cpu")
    assert data is not None and parameters is not None
    data["_episode_cost_accumulator"].raw_cost[0, 0] += 0.7
    data["_episode_cost_accumulator"].training_cost[0, 0] += 0.1
    payload = json.loads(data["_embedded_constraint_state_payload_json"])
    payload["episode_cost_accumulator"]["raw_cost"][0][0] += 0.7
    payload["episode_cost_accumulator"]["training_cost"][0][0] += 0.1
    _resign_constraint_payload(data, payload)
    cotampered_cost = tmp_path / "co-tampered-cost-state.zip"
    save_to_zip_file(
        cotampered_cost,
        data=data,
        params=parameters,
        pytorch_variables=variables,
    )
    with pytest.raises(ValueError, match="cumulative ledger"):
        ConstrainedPPO.load(cotampered_cost)

    # Co-tamper every serialized Adam counter and the embedded optimizer digest.
    # The tensor hash then agrees, but no rollout ledger declares the extra step.
    for state in model.policy.optimizer.state.values():
        state["step"].add_(1)
    substituted_optimizer_digest = model.optimizer_state_sha256
    data, parameters, variables = load_from_zip_file(artifact, device="cpu")
    assert data is not None and parameters is not None
    optimizer_archive = cast(dict[str, Any], parameters["policy.optimizer"])
    serialized_states = cast(dict[Any, dict[str, Any]], optimizer_archive["state"])
    for state in serialized_states.values():
        state["step"] = state["step"].clone()
        state["step"].add_(1)
    payload = json.loads(data["_embedded_constraint_state_payload_json"])
    payload["optimizer_state_sha256"] = substituted_optimizer_digest
    _resign_constraint_payload(data, payload)
    unledgered_optimizer_step = tmp_path / "unledgered-optimizer-step.zip"
    save_to_zip_file(
        unledgered_optimizer_step,
        data=data,
        params=parameters,
        pytorch_variables=variables,
    )
    with pytest.raises(ValueError, match="live optimizer step counter"):
        ConstrainedPPO.load(unledgered_optimizer_step)
    _close_model_env(model)


@pytest.mark.skipif(
    not training_dependencies_available(), reason="optional train dependencies absent"
)
def test_constrained_ppo_genuinely_learns_saves_and_loads_for_inference(
    tmp_path: Path,
) -> None:
    from stable_baselines3 import PPO

    model, tasks = _four_environment_model()
    cost_policy = cast(ConstrainedActorCriticPolicyV1, model.policy)
    initial_actor = actor_reward_parameter_sha256(model)
    initial_cost = {
        name: tensor.detach().cpu().clone()
        for name, tensor in cost_policy.cost_critic.state_dict().items()
    }

    model.learn(total_timesteps=4 * BETA_DECISIONS_PER_CYCLE)

    assert actor_reward_parameter_sha256(model) != initial_actor
    assert any(
        not np.array_equal(
            initial_cost[name].numpy(),
            tensor.detach().cpu().numpy(),
        )
        for name, tensor in cost_policy.cost_critic.state_dict().items()
    )
    assert len(model.rollout_ledgers) == 1
    assert model.rollout_ledgers[0].transition_count == (4 * BETA_DECISIONS_PER_CYCLE)
    assert model.rollout_ledgers[0].complete_cycle_count == 4
    assert model.rollout_ledgers[0].optimizer_steps == 1
    assert [task.reset_seeds for task in tasks] == [
        [FIXTURE_SEED + index] for index in range(4)
    ]

    artifact = tmp_path / "constrained-policy.zip"
    model.pin_external_episode_state(
        {
            "fixture": "four-env-safe-cut",
            "decision_indices": [BETA_DECISIONS_PER_CYCLE] * 4,
        }
    )
    model.save(artifact)
    restored = ConstrainedPPO.load(artifact, env=model.get_env())
    state = model.constraint_state_dict()
    assert state["schema"] == CONSTRAINT_STATE_ENVELOPE_SCHEMA
    assert state["payload"]["schema"] == CONSTRAINT_STATE_SCHEMA
    assert restored.constraint_state_dict() == state

    observation = np.asarray([0.25, 0.0, 0.0], dtype=np.float32)
    expected_action, _ = restored.predict(observation, deterministic=True)
    ordinary_ppo = PPO.load(artifact, env=model.get_env())
    ordinary_action, _ = ordinary_ppo.predict(observation, deterministic=True)
    assert ordinary_action == pytest.approx(expected_action)

    tampered = json.loads(json.dumps(state))
    tampered["payload"]["settings"]["algorithm_id"] = I_LAGRANGIAN_ALGORITHM_ID
    with pytest.raises(ValueError):
        restored.load_constraint_state_dict(tampered)
    _close_model_env(model)

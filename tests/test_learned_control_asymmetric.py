"""Privileged critic isolation, context alignment, and exact actor transfer."""
from datetime import datetime, timedelta, timezone
from types import MethodType

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from docs.research.tools.learned_control_asymmetric_v1.policy import (
    AsymmetricHybridPolicy, CONDITIONS, CONTEXT_DIM, encode_conditions,
)
from docs.research.tools.learned_control_asymmetric_v1.algorithm import AsymmetricPPO, AsymmetricRolloutBuffer
from docs.research.tools.learned_control_asymmetric_v1 import training as t
from docs.research.tools.learned_control_hybrid_v1 import training as hybrid
from docs.research.tools.wholeplant_ppo_v1 import training as common


class ToyConditionEnv(gym.Env):
    def __init__(self, truncated=False):
        self.observation_space = gym.spaces.Box(-10, 10, shape=(149,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(7,), dtype=np.float32)
        self.truncated = truncated
        self.episodes = -1
        self.training_scenario = None

    def observation(self):
        obs = np.zeros(149, dtype=np.float32)
        obs[0], obs[1] = CONDITIONS.index(self.training_scenario["condition"]), self.steps
        return obs

    def reset(self, *, seed=None, options=None):
        self.episodes += 1
        self.steps = 0
        self.training_scenario = {"condition": ("ordinary", "pv6", "meth24")[self.episodes % 3], "seed": self.episodes}
        return self.observation(), {"training_scenario": dict(self.training_scenario)}

    def step(self, action):
        self.steps += 1
        end = self.steps == 2
        return self.observation(), 1., end and not self.truncated, end and self.truncated, {"training_scenario": dict(self.training_scenario)}


def make_toy(truncated=False, mode="condition"):
    t.configure_torch()
    return AsymmetricPPO(AsymmetricHybridPolicy, DummyVecEnv([lambda: ToyConditionEnv(truncated)]),
        n_steps=4, batch_size=2, n_epochs=1, seed=718, learning_rate=.0001, gamma=1, gae_lambda=1,
        policy_kwargs={"net_arch": {"pi": [16, 16], "vf": [16, 16]}, "critic_context_hidden": 8,
                       "critic_context_mode": mode})


def test_actor_actions_and_logprob_bitwise_invariant_to_private_context():
    model = make_toy()
    try:
        obs = torch.as_tensor(np.random.default_rng(12).normal(size=(7, 149)).astype(np.float32))
        context = torch.eye(CONTEXT_DIM)
        other = torch.flip(context, dims=(0,))
        # Nonzero correction proves the value can actually use the privilege.
        with torch.no_grad():
            model.policy.private_value_head[-1].weight.fill_(.2)
            torch.manual_seed(812)
            actions, values, scores = model.policy.forward_with_context(obs, context)
            torch.manual_seed(812)
            actions_other, values_other, scores_other = model.policy.forward_with_context(obs, other)
            _, evaluated, _ = model.policy.evaluate_actions_with_context(obs, actions, context)
            _, evaluated_other, _ = model.policy.evaluate_actions_with_context(obs, actions, other)
        assert torch.equal(actions, actions_other)
        assert torch.equal(scores, scores_other)
        assert torch.equal(evaluated, evaluated_other)
        assert not torch.equal(values, values_other)
        np.testing.assert_array_equal(obs.numpy(), np.random.default_rng(12).normal(size=(7, 149)).astype(np.float32))
    finally:
        model.env.close()


def test_value_only_update_cannot_change_actor_parameters():
    model = make_toy()
    try:
        before = common.actor_parameter_sha256(model)
        obs = torch.zeros(CONTEXT_DIM, 149)
        target = torch.arange(CONTEXT_DIM, dtype=torch.float32).unsqueeze(-1)
        for _ in range(3):
            model.policy.optimizer.zero_grad()
            loss = (model.policy.predict_values_with_context(obs, torch.eye(CONTEXT_DIM)) - target).square().mean()
            loss.backward()
            assert all(parameter.grad is None for parameter in model.policy.mlp_extractor.policy_net.parameters())
            assert all(parameter.grad is None for parameter in model.policy.action_net.parameters())
            model.policy.optimizer.step()
        assert common.actor_parameter_sha256(model) == before
    finally:
        model.env.close()


def test_constant_context_control_has_no_condition_information_even_with_nonzero_head():
    model = make_toy(mode="constant")
    try:
        obs = torch.as_tensor(np.random.default_rng(16).normal(size=(7, 149)).astype(np.float32))
        with torch.no_grad():
            model.policy.private_value_head[-1].weight.fill_(.2)
            one = model.policy.predict_values_with_context(obs, torch.eye(CONTEXT_DIM))
            two = model.policy.predict_values_with_context(obs, torch.flip(torch.eye(CONTEXT_DIM), dims=(0,)))
        assert torch.equal(one, two)
    finally:
        model.env.close()


def test_public_and_private_modes_have_identical_seeded_architecture_and_initial_tensors():
    public, private = make_toy(mode="constant"), make_toy(mode="condition")
    try:
        assert common.policy_parameter_sha256(public) == common.policy_parameter_sha256(private)
        assert common.optimizer_sha256(public) == common.optimizer_sha256(private)
        assert set(public.policy.state_dict()) == set(private.policy.state_dict())
    finally:
        public.env.close()
        private.env.close()


def test_rollout_context_and_observation_indices_stay_aligned_in_shuffled_minibatches():
    observations = gym.spaces.Box(-100, 100, shape=(149,), dtype=np.float32)
    actions = gym.spaces.Box(-1, 1, shape=(7,), dtype=np.float32)
    buffer = AsymmetricRolloutBuffer(3, observations, actions, n_envs=2, gamma=1, gae_lambda=1)
    for step in range(3):
        obs = np.zeros((2, 149), dtype=np.float32)
        obs[:, 0] = [step, 10 + step]
        context = encode_conditions([CONDITIONS[step], CONDITIONS[step + 1]])
        buffer.add(obs, np.zeros((2, 7), dtype=np.float32), np.ones(2), np.array([step == 0] * 2),
            torch.zeros(2), torch.zeros(2), critic_context=context)
    buffer.compute_returns_and_advantage(torch.zeros(2), np.zeros(2, dtype=bool))
    for _ in range(2):
        seen = []
        for batch in buffer.get(2):
            ids = batch.observations[:, 0].numpy().astype(int)
            np.testing.assert_array_equal(batch.critic_context.argmax(dim=-1).numpy(), ids % 10 + ids // 10)
            assert batch.observations.shape[-1] == 149
            seen.extend(ids.tolist())
        assert sorted(seen) == [0, 1, 2, 10, 11, 12]


@pytest.mark.parametrize("truncated,expected_rewards", [(False, [1., 1., 1., 1.]), (True, [1., 11., 1., 13.])])
def test_autoreset_and_terminal_bootstrap_use_correct_condition(truncated, expected_rewards):
    model = make_toy(truncated)
    try:
        def fixed_values(policy, obs, context):
            return (10 + context.argmax(dim=-1)).float().unsqueeze(-1)
        model.policy.predict_values_with_context = MethodType(fixed_values, model.policy)
        _, callback = model._setup_learn(4, callback=None, reset_num_timesteps=True, tb_log_name="engineering", progress_bar=False)
        assert model.collect_rollouts(model.env, callback, model.rollout_buffer, 4)
        np.testing.assert_array_equal(model.rollout_buffer.critic_contexts[:, 0].argmax(axis=-1), [0, 0, 2, 2])
        np.testing.assert_array_equal(model.rollout_buffer.values[:, 0], [10, 10, 12, 12])
        np.testing.assert_allclose(model.rollout_buffer.rewards[:, 0], expected_rewards)
        # The finalfreshresetcondition ismeth24(index5), but no such private
        # value leaks backward across either ended episode.
        expected_returns = [2., 1., 2., 1.] if not truncated else [12., 11., 14., 13.]
        np.testing.assert_allclose(model.rollout_buffer.returns[:, 0], expected_returns)
    finally:
        model.env.close()


def tiny_recipe(module, **changes):
    recipe = module.default_recipe(total_timesteps=64, environments=2, n_steps=16, checkpoint_steps=32,
        freeze_observation_statistics=False,
        deadline_utc=(datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat())
    recipe["ppo"].update(batch_size=16, n_epochs=1, target_kl=None)
    recipe.update(changes)
    return recipe


def test_actual_environment_transfer_training_and_existing_evaluator_reopen(tmp_path):
    from docs.research.tools.learned_control_v1.environment import FixedEnv
    parent_root = tmp_path / "hybrid"
    hybrid.freeze_plan(parent_root, tiny_recipe(hybrid))
    parent_result = hybrid.train(parent_root)
    parent_checkpoint = parent_root / "training/final-policy.zip"
    recipe = tiny_recipe(t, initialization="continue", initial_checkpoint=str(parent_checkpoint),
                         freeze_observation_statistics=True)
    child_root = tmp_path / "asymmetric"
    plan = t.freeze_plan(child_root, recipe)
    model, vector, wrappers = t.make_learner(plan)
    source = hybrid.load_checkpoint(parent_checkpoint)
    try:
        assert model.research_transfer["actor_tensors_exact"]
        assert model.research_transfer["existing_optimizer_named_states_exact"]
        assert common.actor_parameter_sha256(model) == parent_result["endpoint"]["actor_tensor_sha256"]
        for attribute in ("mean", "var", "count"):
            np.testing.assert_array_equal(getattr(vector.obs_rms, attribute), getattr(source.normalizer.obs_rms, attribute))
        obs = torch.zeros(7, 149)
        with torch.no_grad():
            assert torch.equal(source.model.policy.predict_values(obs), model.policy.predict_values_with_context(obs, torch.eye(CONTEXT_DIM)))
    finally:
        source.close()
        vector.close()
    result = t.train(child_root)
    assert result["status"] == "completed"
    assert result["num_timesteps"] == 128
    assert result["endpoint"]["normalizer_observation_count"] == parent_result["endpoint"]["normalizer_observation_count"]
    checkpoint = child_root / "training/final-policy.zip"
    loaded, evaluator_loaded = t.load_checkpoint(checkpoint), common.load_checkpoint(checkpoint)
    env = FixedEnv(condition="ordinary", split="train")
    try:
        obs, _ = env.reset(seed=50_060_001)
        np.testing.assert_array_equal(loaded.predict(obs, deterministic=True)[0], evaluator_loaded.predict(obs, deterministic=True)[0])
        assert loaded.model.observation_space.shape == (149,)
        assert loaded.model.action_space.shape == (7,)
        assert common.optimizer_sha256(loaded.model) == result["endpoint"]["optimizer_state_sha256"]
    finally:
        loaded.close()
        evaluator_loaded.close()
        env.close()


def test_invalid_private_condition_rejected():
    with pytest.raises(ValueError):
        encode_conditions(["invented"])
    model = make_toy()
    try:
        with pytest.raises(ValueError):
            model.policy.predict_values_with_context(torch.zeros(1, 149), torch.zeros(1, CONTEXT_DIM))
    finally:
        model.env.close()

"""Explicit context-aligned PPO collection and optimization.

Collector and optimization structure follow installed Stable-Baselines3PPO;
the only learning-path addition is private context for value estimation.
"""
from __future__ import annotations

from typing import NamedTuple
import numpy as np
import torch
from torch.nn import functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import explained_variance, obs_as_tensor

from docs.research.tools.learned_control_asymmetric_v1.policy import CONTEXT_DIM, CONDITIONS, encode_conditions


class AsymmetricSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    critic_context: torch.Tensor


class AsymmetricRolloutBuffer(RolloutBuffer):
    def reset(self):
        super().reset()
        self.critic_contexts = np.zeros((self.buffer_size, self.n_envs, CONTEXT_DIM), dtype=np.float32)

    def add(self, *args, critic_context, **kwargs):
        context = np.asarray(critic_context, dtype=np.float32)
        if context.shape != (self.n_envs, CONTEXT_DIM) or not np.all((context == 0) | (context == 1)) or not np.all(context.sum(axis=-1) == 1):
            raise ValueError("Rollout context must contain one current condition per environment")
        index = self.pos
        super().add(*args, **kwargs)
        self.critic_contexts[index] = context

    def get(self, batch_size=None):
        if not self.generator_ready:
            self.critic_contexts = self.swap_and_flatten(self.critic_contexts)
        yield from super().get(batch_size)

    def _get_samples(self, batch_inds, env=None):
        public = super()._get_samples(batch_inds, env=env)
        return AsymmetricSamples(*public, self.to_torch(self.critic_contexts[batch_inds]))


def current_context(env):
    records = env.get_attr("training_scenario")
    if any(not isinstance(row, dict) or "condition" not in row for row in records):
        raise ValueError("Current private training scenario metadata is missing")
    return encode_conditions([row["condition"] for row in records])


class AsymmetricPPO(PPO):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("rollout_buffer_class", AsymmetricRolloutBuffer)
        super().__init__(*args, **kwargs)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        assert self._last_obs is not None
        if self.use_sde or not isinstance(self.action_space, spaces.Box):
            raise ValueError("This candidate supports the existing hybridBoxactor withoutSDE")
        if not isinstance(rollout_buffer, AsymmetricRolloutBuffer):
            raise TypeError("Asymmetric PPO requires a context-carrying rollout buffer")
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()
        while n_steps < n_rollout_steps:
            # Capture this before step(), since DummyVecEnv may autoreset.
            critic_context = current_context(env)
            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy.forward_with_context(
                    obs_tensor, obs_as_tensor(critic_context, self.device))
            actions = actions.cpu().numpy()
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            if not np.array_equal(actions, clipped_actions):
                raise ValueError("Hybrid actor emitted an action outside the originalBox")
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1
            for index, done in enumerate(dones):
                step_scenario = infos[index].get("training_scenario")
                if step_scenario is None or step_scenario["condition"] != CONDITIONS[int(critic_context[index].argmax())]:
                    raise ValueError("Step metadata and pre-step critic condition disagree")
                if done and infos[index].get("terminal_observation") is not None and infos[index].get("TimeLimit.truncated", False):
                    terminal_obs = self.policy.obs_to_tensor(infos[index]["terminal_observation"])[0]
                    # The terminal state belongs to the outgoing condition.
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values_with_context(
                            terminal_obs, obs_as_tensor(critic_context[index:index + 1], self.device))[0]
                    rewards[index] += self.gamma * terminal_value
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs,
                               critic_context=critic_context)
            self._last_obs, self._last_episode_starts = new_obs, dones
        with torch.no_grad():
            # new_obs is already the fresh reset state for any ended episode.
            values = self.policy.predict_values_with_context(obs_as_tensor(new_obs, self.device),
                obs_as_tensor(current_context(env), self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []
        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                values, log_prob, entropy = self.policy.evaluate_actions_with_context(
                    rollout_data.observations, actions, rollout_data.critic_context)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss = -torch.min(advantages * ratio,
                    advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(torch.mean((torch.abs(ratio - 1) > clip_range).float()).item())
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                entropy_loss = -torch.mean(-log_prob) if entropy is None else -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break
        explained = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        self.logger.record("train/private_condition_critic", 1)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

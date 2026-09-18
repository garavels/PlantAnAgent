"""Read-only, pre-optimization diagnostics for adaptive research PPO rollouts.

These observations describe one deterministic diagnostic minibatch, not the
shuffled minibatches or gradients eventually used by SGD. They do not select a
checkpoint, change the learner, or provide evidence of operational performance.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from .constrained_ppo import ConstrainedPPO


DIAGNOSTIC_SCHEMA = "plant-research-ppo-rollout-diagnostics-v1"
DIAGNOSTIC_SAMPLE_LIMIT = 256


def _summary(values: Any) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("diagnostic data must be non-empty and finite")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std_population": float(array.std()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean_absolute": float(np.abs(array).mean()),
        "p10": float(np.quantile(array, 0.1)),
        "median": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
    }


def _explained_variance(predictions: Any, targets: Any) -> float | None:
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    if predicted.shape != target.shape or not target.size:
        raise ValueError("explained variance requires aligned, non-empty arrays")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("explained variance requires finite data")
    variance = float(np.var(target))
    if variance == 0.0:
        return None
    return float(1.0 - np.var(target - predicted) / variance)


def _flat(
    buffer: Any, name: str, width: int | None = None
) -> np.ndarray[Any, np.dtype[Any]]:
    """Copy the SB3 env-major order without changing the rollout buffer."""

    array = np.asarray(getattr(buffer, name))
    expected: tuple[int, ...] = (buffer.buffer_size, buffer.n_envs)
    if width is not None:
        expected += (width,)
    if array.shape != expected or not np.isfinite(array).all():
        raise ValueError(f"unaligned or non-finite diagnostic buffer field {name}")
    suffix = () if width is None else (width,)
    return cast(
        np.ndarray[Any, np.dtype[Any]],
        array.swapaxes(0, 1).reshape((-1, *suffix)).copy(),
    )


def capture_rollout_diagnostics(model: ConstrainedPPO) -> dict[str, Any]:
    """Inspect a just-collected rollout from ``BaseCallback._on_rollout_end``.

    The rollout must be full and not yet flattened by ``buffer.get``. Reward
    quantities retain exactly the buffer's training units (including any outer
    reward scaling). Raw-unit cost advantages are the cost GAE multiplied by
    seven, scaling both its targets and value estimates consistently. This is
    distinct from the per-minibatch normalization used by the actor.

    No optimizer step, backward accumulation, sampling, mode change, forward
    hook, distribution-cache update, or mutable buffer iterator is performed.
    Unsupported architectures fail before any forward pass rather than risk
    dropout RNG consumption or batch-normalization state mutation.
    """

    import torch as th
    from gymnasium import spaces
    from stable_baselines3.common.distributions import DiagGaussianDistribution
    from stable_baselines3.common.torch_layers import FlattenExtractor

    from .constrained_ppo import (
        BETA_EPISODE_CYCLE_COUNT,
        RESOURCE_ORDER,
        _torch_normalize_advantage,
    )

    buffer = model.rollout_buffer
    policy: Any = model.policy
    if not buffer.full or buffer.generator_ready:
        raise ValueError("diagnostics require a full, unflattened pre-update rollout")
    if (
        not isinstance(model.action_space, spaces.Box)
        or len(model.action_space.shape) != 1
        or not isinstance(model.observation_space, spaces.Box)
        or len(model.observation_space.shape) != 1
        or type(policy.features_extractor) is not FlattenExtractor
        or type(policy.action_dist) is not DiagGaussianDistribution
        or policy.squash_output
    ):
        raise ValueError("diagnostics require the frozen flat-Box Gaussian policy")
    for module in policy.modules():
        if not tuple(module.children()) and type(module) not in (
            th.nn.Linear,
            th.nn.Tanh,
            th.nn.Flatten,
        ):
            raise ValueError("diagnostics require stateless Linear/Tanh/Flatten leaves")
        if (
            tuple(module.buffers(recurse=False))
            or module._forward_hooks
            or module._forward_pre_hooks
            or module._backward_hooks
            or module._backward_pre_hooks
        ):
            raise ValueError("diagnostics refuse mutable buffers or module hooks")
    if any(parameter._backward_hooks for parameter in policy.parameters()):
        raise ValueError("diagnostics refuse parameter gradient hooks")
    if tuple(policy.features_extractor.parameters()):
        raise ValueError("diagnostics require an unparameterized feature extractor")

    observation_width = model.observation_space.shape[0]
    action_width = model.action_space.shape[0]
    observations = _flat(buffer, "observations", observation_width)
    actions = _flat(buffer, "actions", action_width)
    old_values = _flat(buffer, "values")
    old_log_prob = _flat(buffer, "log_probs")
    rewards = _flat(buffer, "rewards")
    advantages = _flat(buffer, "advantages")
    returns = _flat(buffer, "returns")
    sample_count = min(len(advantages), DIAGNOSTIC_SAMPLE_LIMIT)
    cost_enabled = bool(model.constraint_settings.uses_constraints)
    sample: dict[str, Any] = {
        "count": sample_count,
        "order": "first-256-env-major-then-time",
        "actual_shuffled_sgd_minibatch": False,
        "loss_stage": "pre-optimization-after-rollout-multiplier-update",
    }
    result: dict[str, Any] = {
        "schema": DIAGNOSTIC_SCHEMA,
        "num_timesteps": int(model.num_timesteps),
        "rollout_transition_count": len(advantages),
        "algorithm_id": model.algorithm_id,
        "sample": sample,
        "reward_units": "rollout-buffer-training-units",
        "full_rollout": {
            "reward": _summary(rewards),
            "reward_return": _summary(returns),
            "reward_advantage": _summary(advantages),
            "reward_value": _summary(old_values),
            "reward_explained_variance": _explained_variance(old_values, returns),
            "costs": None,
        },
        "gradient_attribution": None,
    }

    # Detached copies isolate the retained buffer from all Torch views.
    def tensor(values: Any) -> Any:
        return th.tensor(values, dtype=th.float32, device=model.device)

    def numpy(values: Any) -> np.ndarray[Any, np.dtype[Any]]:
        return cast(np.ndarray[Any, np.dtype[Any]], values.detach().cpu().numpy())

    def gradients(loss: Any, parameters: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(
            th.zeros_like(parameter) if gradient is None else gradient
            for parameter, gradient in zip(
                parameters,
                th.autograd.grad(
                    loss, parameters, retain_graph=True, allow_unused=True
                ),
                strict=True,
            )
        )

    def norm(values: tuple[Any, ...]) -> float:
        return float(
            th.sqrt(sum(value.detach().double().square().sum() for value in values))
            .cpu()
            .item()
        )

    def cosine(left: tuple[Any, ...], right: tuple[Any, ...]) -> float | None:
        denominator = norm(left) * norm(right)
        if denominator == 0.0:
            return None
        dot = sum(
            a.detach().double().mul(b.detach().double()).sum()
            for a, b in zip(left, right, strict=True)
        )
        return float(np.clip(float(dot.cpu().item()) / denominator, -1.0, 1.0))

    obs = tensor(observations[:sample_count])
    taken_actions = tensor(actions[:sample_count])
    old_logs = tensor(old_log_prob[:sample_count])
    target = tensor(returns[:sample_count])
    raw_reward_advantage = tensor(advantages[:sample_count])
    reward_advantage = (
        _torch_normalize_advantage(raw_reward_advantage)
        if cost_enabled or (model.normalize_advantage and sample_count > 1)
        else raw_reward_advantage
    )
    parameters = tuple(policy.named_parameters())
    actor_parameters = tuple(
        parameter
        for name, parameter in parameters
        if name == "log_std"
        or name.startswith(("mlp_extractor.policy_net.", "action_net."))
    )
    reward_parameters = tuple(
        parameter
        for name, parameter in parameters
        if name.startswith(("mlp_extractor.value_net.", "value_net."))
    )
    common_parameters = tuple(
        parameter
        for name, parameter in parameters
        if not name.startswith("cost_critic.")
    )
    if {id(p) for p in common_parameters} != {
        id(p) for p in (*actor_parameters, *reward_parameters)
    }:
        raise ValueError(
            "diagnostics require disjoint actor and reward critic networks"
        )

    with th.enable_grad():
        # Avoid evaluate_actions/get_distribution: SB3 stores their newly created
        # distribution on policy.action_dist. Direct MLP/Normal evaluation is pure.
        features = policy.extract_features(obs)
        latent_actor = policy.mlp_extractor.policy_net(features)
        latent_value = features
        saturation = []
        for layer_index, layer in enumerate(policy.mlp_extractor.value_net):
            latent_value = layer(latent_value)
            if type(layer) is th.nn.Tanh:
                output = numpy(latent_value)
                saturation.append(
                    {
                        "layer_index": layer_index,
                        "fraction_abs_output_ge_0_95": float(
                            (np.abs(output) >= 0.95).mean()
                        ),
                        "mean_tanh_derivative": float(
                            cast(np.ndarray[Any, np.dtype[Any]], 1.0 - output**2).mean()
                        ),
                    }
                )
        predicted = policy.value_net(latent_value).flatten()
        mean = policy.action_net(latent_actor)
        std = th.ones_like(mean) * policy.log_std.exp()
        gaussian: Any = th.distributions.Normal(mean, std)
        log_prob = gaussian.log_prob(taken_actions).sum(dim=1)
        ratio = th.exp(log_prob - old_logs)
        entropy_loss = -gaussian.entropy().sum(dim=1).mean()
        low = tensor(model.action_space.low)
        high = tensor(model.action_space.high)
        clipping = gaussian.cdf(low) + (1.0 - gaussian.cdf(high))
        sample.update(
            {
                "normalized_reward_advantage": _summary(numpy(reward_advantage)),
                "probability_ratio": _summary(numpy(ratio)),
                "reward_critic_tanh": saturation,
                "gaussian": {
                    "mean_by_action": [
                        _summary(numpy(mean[:, i])) for i in range(action_width)
                    ],
                    "std_by_action": [
                        _summary(numpy(std[:, i])) for i in range(action_width)
                    ],
                    "expected_clipped_fraction_by_action": numpy(
                        clipping.mean(dim=0)
                    ).tolist(),
                    "expected_clipped_fraction_all_action_coordinates": float(
                        clipping.mean().detach().cpu()
                    ),
                    "expected_any_action_clipped_fraction": float(
                        (1.0 - (1.0 - clipping).prod(dim=1)).mean().detach().cpu()
                    ),
                    "observed_clipped_fraction_all_action_coordinates": float(
                        ((taken_actions < low) | (taken_actions > high))
                        .float()
                        .mean()
                        .cpu()
                    ),
                    "meaning": "Gaussian probability outside policy action-space bounds, before environment action clipping",
                },
                "costs": None,
            }
        )
        multipliers = tensor(model.multiplier_state.multipliers)
        denominator = 1.0 + multipliers.sum() if cost_enabled else tensor(1.0)
        actor_advantage = reward_advantage
        cost_surrogate = None
        cost_heads: list[dict[str, Any]] | None = None
        channel_gradients: list[tuple[Any, ...]] = []
        if cost_enabled:
            costs = _flat(buffer, "training_costs", 4)
            raw_costs = _flat(buffer, "raw_costs", 4)
            cost_advantages = _flat(buffer, "cost_advantages", 4)
            cost_returns = _flat(buffer, "cost_returns", 4)
            cost_values = _flat(buffer, "cost_values", 4)
            normalized_costs = th.stack(
                [
                    _torch_normalize_advantage(
                        tensor(cost_advantages[:sample_count, i])
                    )
                    for i in range(4)
                ],
                dim=1,
            )
            result["full_rollout"]["costs"] = [
                {
                    "resource": resource,
                    "raw_cost": _summary(raw_costs[:, i]),
                    "training_cost": _summary(costs[:, i]),
                    "training_cost_advantage": _summary(cost_advantages[:, i]),
                    "raw_unit_cost_advantage": _summary(
                        cost_advantages[:, i].astype(np.float64)
                        * BETA_EPISODE_CYCLE_COUNT
                    ),
                    "training_cost_return": _summary(cost_returns[:, i]),
                    "cost_explained_variance": _explained_variance(
                        cost_values[:, i], cost_returns[:, i]
                    ),
                }
                for i, resource in enumerate(RESOURCE_ORDER)
            ]
            sample["costs"] = [
                {
                    "resource": resource,
                    "normalized_cost_advantage": _summary(
                        numpy(normalized_costs[:, i])
                    ),
                    "multiplier": float(multipliers[i].detach().cpu()),
                }
                for i, resource in enumerate(RESOURCE_ORDER)
            ]
            actor_advantage = (
                reward_advantage - (normalized_costs * multipliers).sum(dim=1)
            ) / denominator
            cost_surrogate = (
                (normalized_costs * multipliers).sum(dim=1) * ratio
            ).mean() / denominator
            channel_gradients = [
                gradients((normalized_costs[:, i] * ratio).mean(), actor_parameters)
                for i in range(4)
            ]
            predicted_cost = policy.cost_values(obs)
            cost_heads = []
            for i, resource in enumerate(RESOURCE_ORDER):
                cost_loss = th.nn.functional.mse_loss(
                    predicted_cost[:, i], tensor(cost_returns[:sample_count, i])
                )
                # Production takes the mean over four independent cost heads.
                weighted_cost_loss = (
                    model.constraint_settings.cost_value_loss_coefficient
                    * cost_loss
                    / 4.0
                )
                head_norm = norm(
                    gradients(
                        weighted_cost_loss,
                        tuple(policy.cost_critic.heads[i].parameters()),
                    )
                )
                cost_heads.append(
                    {
                        "resource": resource,
                        "weighted_loss": float(weighted_cost_loss.detach().cpu()),
                        "preclip_gradient_norm": head_norm,
                        "implied_clip_factor": min(
                            1.0, float(model.max_grad_norm) / (head_norm + 1e-6)
                        ),
                    }
                )

        clip_setting = model.clip_range
        clip = float(
            clip_setting(model._current_progress_remaining)
            if callable(clip_setting)
            else clip_setting
        )
        policy_loss = -th.minimum(
            actor_advantage * ratio,
            actor_advantage * ratio.clamp(1.0 - clip, 1.0 + clip),
        ).mean()
        if model.clip_range_vf is not None:
            value_clip_setting = model.clip_range_vf
            value_clip = float(
                value_clip_setting(model._current_progress_remaining)
                if callable(value_clip_setting)
                else value_clip_setting
            )
            previous = tensor(old_values[:sample_count])
            value_prediction = previous + (predicted - previous).clamp(
                -value_clip, value_clip
            )
        else:
            value_prediction = predicted
        value_loss = th.nn.functional.mse_loss(value_prediction, target)
        common_loss = (
            policy_loss + model.ent_coef * entropy_loss + model.vf_coef * value_loss
        )
        reward_surrogate = -(reward_advantage * ratio).mean() / denominator
        reward_gradient = gradients(reward_surrogate, actor_parameters)
        cost_gradient = (
            None
            if cost_surrogate is None
            else gradients(cost_surrogate, actor_parameters)
        )
        common_norm = norm(gradients(common_loss, common_parameters))
        result["gradient_attribution"] = {
            "units": "loss gradients before optimizer and gradient clipping",
            "component_definition": "unclipped reward descent and cost-penalty descent surrogates; cosine below zero means conflict",
            "reward_actor_gradient_norm": norm(reward_gradient),
            "weighted_cost_actor_gradient_norm": (
                None if cost_gradient is None else norm(cost_gradient)
            ),
            "reward_vs_weighted_cost_gradient_cosine": (
                None
                if cost_gradient is None
                else cosine(reward_gradient, cost_gradient)
            ),
            "cost_actor_components": (
                None
                if not cost_enabled
                else [
                    {
                        "resource": resource,
                        "unweighted_cost_gradient_norm": norm(channel_gradients[i]),
                        "reward_vs_unweighted_cost_gradient_cosine": cosine(
                            reward_gradient, channel_gradients[i]
                        ),
                    }
                    for i, resource in enumerate(RESOURCE_ORDER)
                ]
            ),
            "actor_total_preclip_gradient_norm": norm(
                gradients(common_loss, actor_parameters)
            ),
            "reward_critic_preclip_gradient_norm": norm(
                gradients(common_loss, reward_parameters)
            ),
            "actor_and_reward_common_preclip_gradient_norm": common_norm,
            "implied_common_clip_factor": min(
                1.0, float(model.max_grad_norm) / (common_norm + 1e-6)
            ),
            "max_grad_norm": float(model.max_grad_norm),
            "reward_value_mse": float(value_loss.detach().cpu()),
            "weighted_reward_value_loss": float(
                (model.vf_coef * value_loss).detach().cpu()
            ),
            "policy_loss": float(policy_loss.detach().cpu()),
            "entropy_loss": float(entropy_loss.detach().cpu()),
            "cost_heads": cost_heads,
        }
    # Undefined variances/cosines are explicit nulls. Any other non-finite
    # numerical failure rejects the diagnostic rather than fabricating a score.
    json.dumps(result, allow_nan=False)
    return result

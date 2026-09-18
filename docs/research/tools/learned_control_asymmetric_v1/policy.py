"""A private value baseline attached to an unchanged public-observation actor."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from docs.research.tools.learned_control_hybrid_v1.policy import HybridActorCriticPolicy

CONDITIONS = ("ordinary", "control", "pv6", "pv18", "pv36", "meth24", "meth48")
CONTEXT_DIM = len(CONDITIONS)


def critic_identity():
    return {"schema": "learned-control-asymmetric-critic-v1", "conditions": list(CONDITIONS),
        "public_actor_observation_shape": [149], "public_actor_action_shape": [7],
        "critic_privilege": "Inconditionmode only:a training-only onehot identifier of the exogenous episode condition,including its already-defined disturbance family",
        "critic_context_modes": {"condition": "Use the actual exogenous condition for value estimation only",
                                 "constant": "Architecture-matched public control:replace every critic context with ordinaryonehot"},
        "active_mode_record": "plan.critic_context_mode and checkpoint.private_critic_conditioning_mode",
        "critic_architecture": "Existing public value estimate plus zero-initialized correction from public critic latent and private condition onehot",
        "actor_condition_access": False, "seed_or_future_weather_access": False,
        "condition_source": "MixtureEnv.training_scenario before each decision; terminal bootstrap uses the outgoing episode condition",
        "unbiased_baseline_argument": "Given public actor input o and exogenous condition c, actions follow pi(a|o). Therefore E_a[grad(log pi(a|o))*b(o,c)|o,c]=0 for any action-independent baseline. Approximate GAE and clipped PPO retain their usual approximation bias.",
        "deployment": "The evaluator calls the same public actor distribution; private condition is neither requested nor available"}


def encode_conditions(conditions):
    values = np.zeros((len(conditions), CONTEXT_DIM), dtype=np.float32)
    for index, condition in enumerate(conditions):
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown private critic condition:{condition}")
        values[index, CONDITIONS.index(condition)] = 1.
    return values


class AsymmetricHybridPolicy(HybridActorCriticPolicy):
    def __init__(self, *args, critic_context_hidden=64, critic_context_mode="condition", **kwargs):
        if critic_context_mode not in ("condition", "constant"):
            raise ValueError("Critic context mode must beconditionorconstant")
        self.critic_context_hidden = critic_context_hidden
        self.critic_context_mode = critic_context_mode
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.private_value_head = nn.Sequential(
            nn.Linear(self.mlp_extractor.latent_dim_vf + CONTEXT_DIM, self.critic_context_hidden),
            nn.Tanh(), nn.Linear(self.critic_context_hidden, 1))
        nn.init.orthogonal_(self.private_value_head[0].weight)
        nn.init.zeros_(self.private_value_head[0].bias)
        # An initialized BC checkpoint retains its exact old values initially.
        nn.init.zeros_(self.private_value_head[-1].weight)
        nn.init.zeros_(self.private_value_head[-1].bias)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data["critic_context_hidden"] = self.critic_context_hidden
        data["critic_context_mode"] = self.critic_context_mode
        return data

    def predict_values_with_context(self, obs, critic_context):
        if critic_context.ndim != 2 or critic_context.shape != (obs.shape[0], CONTEXT_DIM):
            raise ValueError("Private critic context shape does not match the public observation batch")
        if not torch.isfinite(critic_context).all() or not torch.all((critic_context == 0) | (critic_context == 1)) or not torch.all(critic_context.sum(dim=-1) == 1):
            raise ValueError("Private critic context must be an explicit condition onehot")
        if self.critic_context_mode == "constant":
            critic_context = torch.zeros_like(critic_context)
            critic_context[:, 0] = 1.
        features = self.extract_features(obs)
        vf_features = features if self.share_features_extractor else features[1]
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        public_value = self.value_net(latent_vf)
        private_value = self.private_value_head(torch.cat((latent_vf, critic_context.to(latent_vf)), dim=-1))
        return public_value + private_value

    def forward_with_context(self, obs, critic_context, deterministic=False):
        # Only the critic sees context. The actor call is literally inherited.
        actions, _, log_prob = super().forward(obs, deterministic=deterministic)
        values = self.predict_values_with_context(obs, critic_context)
        return actions, values, log_prob

    def evaluate_actions_with_context(self, obs, actions, critic_context):
        distribution = self.get_distribution(obs)
        return self.predict_values_with_context(obs, critic_context), distribution.log_prob(actions), distribution.entropy()


__all__ = ["AsymmetricHybridPolicy", "CONDITIONS", "CONTEXT_DIM", "encode_conditions", "critic_identity"]

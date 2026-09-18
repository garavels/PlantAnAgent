"""Learned OFF atoms plus censored Gaussian continuous actions.

There are no plant-specific branches. Every coordinate receives a learned gate,
continuous mean and standard deviation. The probability of a clipped endpoint
is its *marginal probability mass*, not a Gaussian density at that endpoint.
"""
from __future__ import annotations

from functools import partial
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy

SCHEMA = "learned-off-censored-gaussian-v1"
# PPO stores Box actions as float32. A latent within half an ulp of an endpoint
# rounds to that endpoint, so that probability belongs to its atom, too.
ENDPOINT_HALF_ULP = 2.0 ** -25
LOWER_CENSOR = -1.0 + ENDPOINT_HALF_ULP
UPPER_CENSOR = 1.0 - ENDPOINT_HALF_ULP


def distribution_identity():
    return {"schema": SCHEMA, "coordinates": 7,
        "learned_parameters_per_coordinate": ["off_gate_logit", "on_mean", "on_scale_logit"],
        "on_distribution": "Normal(mean,state-dependent_scale), censored into existing[-1,1]Box",
        "off_mass": "q+(1-q)*Phi((lower_censor-mean)/scale)",
        "maximum_mass": "(1-q)*(1-Phi((upper_censor-mean)/scale))",
        "interior_density": "(1-q)*NormalPDF(action;mean,scale)",
        "endpoint_half_ulp": ENDPOINT_HALF_ULP,
        "endpoint_numerics": "Float32 endpoint-rounding intervals are included in the endpoint masses",
        "deterministic_rule": "Choose learned OFF gate mode atq>=0.5; otherwise clip learned continuous mean intoBox",
        "legacy_inference_field_semantics": "Existing replay fields unclipped_mean/raw_mean contain the hierarchical deterministic action for this hybrid actor,not the continuousONmean",
        "entropy": "Marginal Shannon endpoint terms plus interior differential entropy",
        "plant_specific_inference_logic": False, "teacher_at_inference": False,
        "physical_action_support": "All seven originalBox coordinates,including exact-1 and+1"}


class OffCensoredGaussianDistribution(Distribution):
    def __init__(self, action_dim, *, min_on_std=.005, max_on_std=.5):
        super().__init__()
        if not 0 < min_on_std < max_on_std:
            raise ValueError("ON scale limits must be strictly positive and ordered")
        self.action_dim = action_dim
        self.min_on_std, self.max_on_std = min_on_std, max_on_std

    def proba_distribution_net(self, latent_dim):
        return nn.Linear(latent_dim, self.action_dim * 3)

    def proba_distribution(self, parameters):
        if parameters.shape[-1] != self.action_dim * 3:
            raise ValueError("Expected gate,mean,and scale parameters for every coordinate")
        # Float64 CDFs/entropy keep extreme-tail endpoint scores differentiable.
        self.parameters = parameters
        self.gate_logits, self.on_mean, raw_scale = parameters.double().split(self.action_dim, dim=-1)
        self.on_std = self.min_on_std + (self.max_on_std - self.min_on_std) * torch.sigmoid(raw_scale)
        self.log_off_gate, self.log_on_gate = F.logsigmoid(self.gate_logits), F.logsigmoid(-self.gate_logits)
        self.distribution = torch.distributions.Normal(self.on_mean, self.on_std)
        return self

    def _components(self):
        a = (LOWER_CENSOR - self.on_mean) / self.on_std
        b = (UPPER_CENSOR - self.on_mean) / self.on_std
        log_left = torch.logaddexp(self.log_off_gate, self.log_on_gate + torch.special.log_ndtr(a))
        log_right = self.log_on_gate + torch.special.log_ndtr(-b)
        # Use survival differences when both limits are positive to avoid
        # subtracting two probabilities rounded to one.
        interior = torch.where(a > 0,
            torch.exp(torch.special.log_ndtr(-a)) - torch.exp(torch.special.log_ndtr(-b)),
            torch.exp(torch.special.log_ndtr(b)) - torch.exp(torch.special.log_ndtr(a)))
        return a, b, log_left, log_right, interior.clamp_min(0.)

    def coordinate_masses(self):
        _, _, left, right, interior = self._components()
        return {"off": torch.exp(left), "maximum": torch.exp(right),
                "interior": torch.exp(self.log_on_gate) * interior}

    def log_prob(self, actions):
        values = actions.double()
        _, _, left, right, _ = self._components()
        continuous = self.log_on_gate + self.distribution.log_prob(values)
        logp = torch.where(values == -1., left, torch.where(values == 1., right, continuous))
        valid = ((values > LOWER_CENSOR) & (values < UPPER_CENSOR)) | (values == -1.) | (values == 1.)
        logp = torch.where(valid, logp, torch.full_like(logp, -torch.inf))
        return logp.sum(dim=-1).to(self.parameters.dtype)

    def entropy(self):
        a, b, log_left, log_right, interior = self._components()
        left, right, on = torch.exp(log_left), torch.exp(log_right), torch.exp(self.log_on_gate)
        phi_a = torch.exp(-.5 * a.square()) / math.sqrt(2 * math.pi)
        phi_b = torch.exp(-.5 * b.square()) / math.sqrt(2 * math.pi)
        normal_integral = interior * (torch.log(self.on_std) + .5 * math.log(2 * math.pi))
        normal_integral += .5 * (interior + a * phi_a - b * phi_b)
        entropy = -left * log_left - right * log_right - on * interior * self.log_on_gate + on * normal_integral
        return entropy.sum(dim=-1).to(self.parameters.dtype)

    def sample(self):
        off = torch.rand_like(self.gate_logits) < torch.sigmoid(self.gate_logits)
        latent = self.distribution.sample()
        action = torch.where(latent <= LOWER_CENSOR, -torch.ones_like(latent),
                             torch.where(latent >= UPPER_CENSOR, torch.ones_like(latent), latent))
        return torch.where(off, -torch.ones_like(action), action).to(self.parameters.dtype)

    def mode(self):
        # This is the registered hierarchical deterministic convention, not a
        # comparison between an atom's probability and a continuous density.
        action = torch.where(self.on_mean <= LOWER_CENSOR, -torch.ones_like(self.on_mean),
                             torch.where(self.on_mean >= UPPER_CENSOR, torch.ones_like(self.on_mean), self.on_mean))
        return torch.where(self.gate_logits >= 0, -torch.ones_like(action), action).to(self.parameters.dtype)

    def actions_from_params(self, parameters, deterministic=False):
        return self.proba_distribution(parameters).get_actions(deterministic=deterministic)

    def log_prob_from_params(self, parameters):
        self.proba_distribution(parameters)
        actions = self.get_actions()
        return actions, self.log_prob(actions)


class HybridActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, min_on_std=.005, max_on_std=.5,
                 initial_on_std=.05, initial_off_probability=.5, **kwargs):
        if not min_on_std < initial_on_std < max_on_std:
            raise ValueError("Initial ON scale must lie strictly inside its parameterization")
        if not 0 < initial_off_probability < 1:
            raise ValueError("Initial gate probability must lie inside(0,1)")
        self.min_on_std, self.max_on_std = min_on_std, max_on_std
        self.initial_on_std, self.initial_off_probability = initial_on_std, initial_off_probability
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        self._build_mlp_extractor()
        dimension = int(np.prod(self.action_space.shape))
        if dimension != 7 or not np.all(self.action_space.low == -1) or not np.all(self.action_space.high == 1):
            raise ValueError("Hybrid policy requires the unchanged seven-dimensional[-1,1]Box")
        self.action_dist = OffCensoredGaussianDistribution(dimension,
            min_on_std=self.min_on_std, max_on_std=self.max_on_std)
        self.action_net = self.action_dist.proba_distribution_net(self.mlp_extractor.latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        if self.ortho_init:
            gains = {self.features_extractor: np.sqrt(2), self.mlp_extractor: np.sqrt(2),
                     self.action_net: .01, self.value_net: 1.}
            if not self.share_features_extractor:
                del gains[self.features_extractor]
                gains[self.pi_features_extractor] = np.sqrt(2)
                gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in gains.items():
                module.apply(partial(self.init_weights, gain=gain))
        with torch.no_grad():
            gate = math.log(self.initial_off_probability / (1 - self.initial_off_probability))
            scale_fraction = (self.initial_on_std - self.min_on_std) / (self.max_on_std - self.min_on_std)
            self.action_net.bias[:dimension].fill_(gate)
            self.action_net.bias[dimension:2 * dimension].zero_()
            self.action_net.bias[2 * dimension:].fill_(math.log(scale_fraction / (1 - scale_fraction)))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi):
        return self.action_dist.proba_distribution(self.action_net(latent_pi))

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(min_on_std=self.min_on_std, max_on_std=self.max_on_std,
                    initial_on_std=self.initial_on_std, initial_off_probability=self.initial_off_probability)
        return data


__all__ = ["HybridActorCriticPolicy", "OffCensoredGaussianDistribution", "distribution_identity"]

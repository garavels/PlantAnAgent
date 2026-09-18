"""The horizon-conditioned predictor: where the plant will be in Δt, in latent space.

    h_hat(t, t+Δt] = g_phi(h_t, Δt)

`g_phi` is the piece that makes HEPA horizon-conditioned rather than one-step: the
*same* weights answer "in one hour" and "in a day", because the horizon enters as an
input instead of as a separate head. That matters for a plant, where the interesting
question is not what the next sample looks like but whether the tanks survive the
night.

The horizon is embedded on a log scale -- Δt spans 12 to 288 samples (1 h to 24 h)
and is sampled log-uniformly during pretraining, so log(Δt) is the axis the model
should see linearly -- and expanded into sinusoidal features before the MLP, so that
nearby horizons start out similar without being forced to be identical.

Shapes are broadcast on purpose: `(B, D)` with `(B, M)` horizons yields `(B, M, D)`,
one predicted representation per horizon, which is exactly what the survival head
needs to evaluate a whole horizon grid in one pass.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import HepaConfig

#: Sinusoidal features used to expand log(Δt) before the embedding MLP.
HORIZON_FOURIER_FEATURES = 8


class HorizonEmbedding(nn.Module):
    """Δt in samples -> `(..., D)`. Log-scaled, Fourier-expanded, then learned."""

    def __init__(self, config: HepaConfig) -> None:
        super().__init__()
        self.max_horizon_steps = config.max_horizon_steps
        frequencies = 2.0 ** torch.arange(HORIZON_FOURIER_FEATURES, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.project = nn.Sequential(
            nn.Linear(1 + 2 * HORIZON_FOURIER_FEATURES, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

    def forward(self, horizon_steps: Tensor) -> Tensor:
        steps = horizon_steps.to(dtype=self.frequencies.dtype).clamp(min=1.0)
        # 0 at Δt = 1, 1 at the deepest horizon: the scale the model is trained over.
        scaled = torch.log(steps) / math.log(float(max(self.max_horizon_steps, 2)))
        angles = scaled.unsqueeze(-1) * self.frequencies * math.pi
        features = torch.cat(
            [scaled.unsqueeze(-1), torch.sin(angles), torch.cos(angles)], dim=-1
        )
        return self.project(features)


class HorizonPredictor(nn.Module):
    """`g_phi`: `(h_t, Δt)` -> the representation the encoder should produce at t+Δt."""

    def __init__(self, config: HepaConfig) -> None:
        super().__init__()
        self.config = config
        self.horizon = HorizonEmbedding(config)
        hidden = config.d_model * config.predictor_multiplier
        self.net = nn.Sequential(
            nn.LayerNorm(config.d_model * 2),
            nn.Linear(config.d_model * 2, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.d_model),
        )

    def forward(self, context_latent: Tensor, horizon_steps: Tensor) -> Tensor:
        """`(B, D)` and `(B,)` -> `(B, D)`; `(B, D)` and `(B, M)` -> `(B, M, D)`."""
        horizons = torch.as_tensor(horizon_steps, device=context_latent.device)
        if horizons.dim() == 1:
            embedded = self.horizon(horizons)
            joined = torch.cat([context_latent, embedded], dim=-1)
            return self.net(joined)
        if horizons.dim() != 2:
            raise ValueError("horizon_steps must be (B,) or (B, M)")
        embedded = self.horizon(horizons)
        expanded = context_latent.unsqueeze(1).expand(-1, embedded.shape[1], -1)
        joined = torch.cat([expanded, embedded], dim=-1)
        return self.net(joined)


__all__ = ["HORIZON_FOURIER_FEATURES", "HorizonEmbedding", "HorizonPredictor"]

"""The causal context encoder: telemetry window -> latent representation.

    x (B, T, S)  ->  instance norm  ->  patches (B, N, P*S)  ->  linear  ->  +positions
                 ->  causal Transformer  ->  tokens (B, N, D)  ->  h_t = tokens[:, -1]

This is `f_theta` from the paper: a causal Transformer over non-overlapping patches
of the series, with per-context instance normalisation and sinusoidal positional
encodings. `h_t` is the representation of everything up to and including sample `t`,
which is the last token of a causal stack -- so no future sample can reach it, by
construction rather than by convention. `test_models.py` checks that: perturbing a
sample after `t` must not move `h_t`.

The same module, with the same weights, produces the *target* representation from a
future interval (`pool` over its tokens with attention). Weight sharing is what
makes the pretraining objective self-consistent: the predictor is asked to land on
a point the encoder itself would produce.

One deviation, deliberate: instance-normalisation statistics for a future interval
are taken over the whole recorded interval (K samples) rather than over the first
Δt of it, because a batch mixes horizons and the alternative is a per-sample
renormalisation for every drawn Δt. It affects targets only -- never a model input,
never inference.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import HepaConfig

#: Guard for the per-context standard deviation. A constant channel inside one
#: window is normal (a tank that did not move), and dividing it by its own noise
#: would turn nothing into a large input.
EPS = 1e-5


def instance_normalise(x: Tensor) -> Tensor:
    """Per context, per feature: subtract the window mean, divide by its spread.

    `x` is `(B, T, S)`; statistics are taken over `T`. This removes the slow
    offset (a tank sitting at 80% versus 20%) and leaves the shape of the window,
    which is what a horizon-conditioned predictor can actually generalise over.
    """
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True, unbiased=False)
    return (x - mean) / (std + EPS)


def sinusoidal_positions(length: int, width: int) -> Tensor:
    """`(length, width)` sinusoidal encodings, as in the original Transformer."""
    if width % 2 != 0:
        raise ValueError(f"d_model must be even for sinusoidal encodings, got {width}")
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    index = torch.arange(0, width, 2, dtype=torch.float32)
    frequency = torch.exp(-math.log(10_000.0) * index / width)
    encoding = torch.zeros(length, width)
    encoding[:, 0::2] = torch.sin(position * frequency)
    encoding[:, 1::2] = torch.cos(position * frequency)
    return encoding


class AttentionPooling(nn.Module):
    """Learned single-query attention over tokens -> one vector.

    Used to turn a future interval's token sequence into the target representation.
    A mask hides tokens beyond the drawn horizon, so one encoded future block serves
    every horizon in the batch.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(d_model))
        self.scale = d_model**-0.5
        nn.init.normal_(self.query, std=0.02)

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> Tensor:
        """`tokens` `(B, N, D)`, `mask` `(B, N)` with True for tokens to keep."""
        scores = torch.einsum("bnd,d->bn", tokens, self.query) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bn,bnd->bd", weights, tokens)


class CausalPatchEncoder(nn.Module):
    """`f_theta`: a causal Transformer over patch tokens of a telemetry window."""

    def __init__(self, config: HepaConfig) -> None:
        super().__init__()
        self.config = config
        self.patch = nn.Linear(config.patch_size * config.input_dim, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * config.ff_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Nested tensors are a padding optimisation for variable-length batches. Every
        # window here has the same length, and it is incompatible with `norm_first`.
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.pool = AttentionPooling(config.d_model)
        self.register_buffer(
            "positions",
            sinusoidal_positions(self._max_tokens(config), config.d_model),
            persistent=False,
        )

    @staticmethod
    def _max_tokens(config: HepaConfig) -> int:
        longest = max(config.context_len, config.max_horizon_steps)
        return -(-longest // config.patch_size) + 1

    def tokenise(self, x: Tensor) -> Tensor:
        """`(B, T, S)` -> `(B, N, D)` embedded patch tokens, N = T // P."""
        batch, total, features = x.shape
        patch_size = self.config.patch_size
        if features != self.config.input_dim:
            raise ValueError(
                f"expected {self.config.input_dim} features per sample, got {features}"
            )
        if total < patch_size or total % patch_size != 0:
            raise ValueError(
                f"a window of {total} samples is not a whole number of {patch_size}-sample patches"
            )
        normalised = instance_normalise(x)
        patches = normalised.reshape(batch, total // patch_size, patch_size * features)
        tokens = self.patch(patches)
        return tokens + self.positions[: tokens.shape[1]].unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        """`(B, T, S)` -> `(B, N, D)` causal token representations."""
        tokens = self.tokenise(x)
        length = tokens.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(
            length, device=tokens.device, dtype=tokens.dtype
        )
        return self.norm(self.transformer(tokens, mask=mask, is_causal=True))

    def encode_context(self, x: Tensor) -> Tensor:
        """`h_t`: `(B, T, S)` -> `(B, D)`, the last causal token. The whole input."""
        return self.forward(x)[:, -1]

    def encode_future(self, future: Tensor, horizon_steps: Tensor | None = None) -> Tensor:
        """The target representation of a future interval: `(B, K, S)` -> `(B, D)`.

        `horizon_steps` `(B,)` selects how much of the interval counts: a token is
        kept when the interval it covers ends at or before Δt, and the first token
        is always kept so no target is empty.
        """
        tokens = self.forward(future)
        mask = None
        if horizon_steps is not None:
            ends = (
                torch.arange(1, tokens.shape[1] + 1, device=tokens.device)
                * self.config.patch_size
            )
            mask = ends.unsqueeze(0) <= horizon_steps.reshape(-1, 1)
            mask[:, 0] = True
        return self.pool(tokens, mask)


__all__ = [
    "EPS",
    "AttentionPooling",
    "CausalPatchEncoder",
    "instance_normalise",
    "sinusoidal_positions",
]

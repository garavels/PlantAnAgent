"""The event head: hazards per step, survival across a horizon.

The head does not predict "probability of an event within 6 h" directly. It predicts
a *hazard* -- the chance the event first happens in this step, given it has not yet --
and the horizon probability is the survival complement from the paper:

    p(t, Δt) = 1 - prod_{j=1..Δt} (1 - lambda_j(t))

Two properties come free from that form, and both matter when an operator reads the
numbers: the probability is monotone in Δt (a longer window cannot be safer), and the
per-horizon outputs are mutually consistent instead of five independent classifiers
that occasionally disagree.

Hazards are produced at *patch resolution*: one hazard per event per anchor block of
P samples, held constant inside the block. That is a deliberate deviation from a
per-step hazard, and it is about cost -- a per-step hazard for K = 288 needs 288
evaluations of `g_phi` per sample, an anchored one needs 18, and the resulting
probability is still an exact survival product over 288 steps:

    log S(Δt) = sum_m n_m(Δt) * log(1 - lambda_m),  n_m(Δt) = clamp(Δt - m*P, 0, P)

so a horizon that ends mid-block is counted by the number of steps it actually covers.
Everything is computed in log space (`softplus`, `expm1`) because a product of 288
near-one factors is exactly where a naive implementation loses its gradient.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import HepaConfig


def anchor_steps(config: HepaConfig, device: torch.device | None = None) -> Tensor:
    """`(M,)` the horizon each anchor block ends at, in samples: P, 2P, ..."""
    return (
        torch.arange(1, config.horizon_anchors + 1, device=device, dtype=torch.float32)
        * config.patch_size
    )


def anchor_coverage(config: HepaConfig, horizon_steps: Tensor) -> Tensor:
    """How many steps of each anchor block fall inside each horizon.

    `horizon_steps` `(..., H)` -> `(..., H, M)`, values in `[0, P]`.
    """
    starts = (
        torch.arange(config.horizon_anchors, device=horizon_steps.device, dtype=torch.float32)
        * config.patch_size
    )
    steps = horizon_steps.to(dtype=torch.float32).unsqueeze(-1)
    return (steps - starts).clamp(min=0.0, max=float(config.patch_size))


class SurvivalEventHead(nn.Module):
    """Predicted latents -> per-anchor hazards -> per-horizon event probabilities."""

    def __init__(self, config: HepaConfig) -> None:
        super().__init__()
        self.config = config
        self.norm = nn.LayerNorm(config.d_model)
        self.hazard = nn.Linear(config.d_model, config.event_count)
        # Start pessimistic-free: a large negative bias means "nothing is about to
        # happen", which is the right prior for events that are rare per step.
        nn.init.zeros_(self.hazard.weight)
        nn.init.constant_(self.hazard.bias, -5.0)

    def hazard_logits(self, latents: Tensor) -> Tensor:
        """`(B, M, D)` predicted latents -> `(B, M, E)` hazard logits."""
        if latents.dim() != 3:
            raise ValueError("latents must be (B, M, D), one row per anchor block")
        return self.hazard(self.norm(latents))

    def log_survival(self, latents: Tensor, horizon_steps: Tensor) -> Tensor:
        """`log(1 - p)` for every event and horizon: `(B, E, H)`.

        `horizon_steps` is `(H,)` (shared) or `(B, H)` (per sample), in samples.
        """
        logits = self.hazard_logits(latents)
        horizons = torch.as_tensor(horizon_steps, device=logits.device, dtype=torch.float32)
        if horizons.dim() == 1:
            horizons = horizons.unsqueeze(0).expand(logits.shape[0], -1)
        coverage = anchor_coverage(self.config, horizons)  # (B, H, M)
        # log(1 - sigmoid(z)) = -softplus(z): stable, and never touches log(0).
        log_survival_per_step = -F.softplus(logits)  # (B, M, E)
        return torch.einsum("bhm,bme->beh", coverage, log_survival_per_step)

    def probabilities(self, latents: Tensor, horizon_steps: Tensor) -> Tensor:
        """`P(event within Δt)` for every event and horizon: `(B, E, H)` in [0, 1]."""
        return -torch.expm1(self.log_survival(latents, horizon_steps))

    def logit_probabilities(self, latents: Tensor, horizon_steps: Tensor) -> Tensor:
        """The same probabilities as logits, for a numerically safe BCE.

        `logit(p) = log(1 - S) - log(S)` with `log1p(-exp(log S))` for the first term.
        """
        log_survival = self.log_survival(latents, horizon_steps)
        # log(-expm1(x)) for x < 0, clamped away from exactly 0 where p -> 0.
        log_event = torch.log(-torch.expm1(log_survival.clamp(max=-1e-7)))
        return log_event - log_survival


__all__ = ["SurvivalEventHead", "anchor_coverage", "anchor_steps"]

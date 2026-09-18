"""The stage-1 objective: predict the future representation, keep it isotropic.

    loss = (1 - alpha) * L1(h_hat, target) + alpha * SIGReg(h_hat)      alpha = 0.1

The L1 term is the prediction. The SIGReg term is what stops the trivial solution:
without it, a predictor and a weight-shared target encoder can agree perfectly by
mapping every window to the same point, which is why JEPA-style methods normally need
a stop-gradient or an EMA teacher. The paper follows the regularised route instead --
a distributional penalty that pushes the latent distribution towards an isotropic
Gaussian, which cannot be satisfied by a collapsed representation, so the target
encoder can keep its gradients and there is no teacher to tune.

SIGReg here is the sketched Cramér-von Mises form: project the batch of latents onto
R random unit directions and compare each projection's empirical CDF against the
standard normal one. An isotropic Gaussian is exactly the distribution whose every
one-dimensional projection is standard normal, so testing sketched projections is a
cheap, differentiable stand-in for testing the joint distribution. Directions are
redrawn every step, so no single set of projections can be gamed.

Horizons are drawn log-uniformly, as the paper does, so short horizons -- which the
model sees most often at serving time -- are not drowned out by the long tail. They
are drawn in *patch units*, a deviation the target path forces: an interval's target
representation is pooled over whole patch tokens, so a Δt that ends mid-patch has no
distinct target. It costs resolution in the pretraining horizon only; fine-tuning and
inference use exact sample horizons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

#: Random projections per SIGReg evaluation. Enough to be a meaningful sketch of a
#: 256-dimensional latent, cheap enough to run every step.
DEFAULT_PROJECTIONS = 64

#: The paper's SIGReg weight.
DEFAULT_SIGREG_WEIGHT = 0.1

_SQRT_2 = math.sqrt(2.0)


def sample_horizons(
    batch_size: int,
    *,
    max_steps: int,
    patch_size: int,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """`(B,)` horizons in samples, log-uniform over the whole horizon range."""
    anchors = max(1, max_steps // patch_size)
    uniform = torch.rand(batch_size, generator=generator, device=device)
    drawn = torch.exp(uniform * math.log(float(anchors)))
    steps = drawn.round().clamp(min=1.0, max=float(anchors)) * float(patch_size)
    return steps.clamp(max=float(max_steps))


def standard_normal_cdf(values: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(values / _SQRT_2))


def sigreg(
    latents: Tensor,
    *,
    projections: int = DEFAULT_PROJECTIONS,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sketched isotropic-Gaussian regularisation of `(B, D)` latents. Scalar >= 0.

    Cramer-von Mises distance between each random projection's empirical CDF and
    the standard normal CDF, averaged over projections:

        W^2 = 1/(12 B) + sum_i (Phi(z_i) - (2i - 1) / (2 B))^2

    Zero only when every sketched projection looks standard normal, which a
    collapsed (constant) representation never does.
    """
    if latents.dim() != 2:
        raise ValueError("SIGReg expects (B, D) latents")
    batch, width = latents.shape
    if batch < 2:
        # A single sample has no distribution to regularise; returning zero keeps a
        # degenerate batch from producing a meaningless gradient.
        return latents.new_zeros(())
    directions = torch.randn(
        width, projections, generator=generator, device=latents.device, dtype=latents.dtype
    )
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
    projected, _ = torch.sort(latents @ directions, dim=0)  # (B, R)
    ranks = torch.arange(
        1, batch + 1, device=latents.device, dtype=latents.dtype
    ).unsqueeze(1)
    expected = (2.0 * ranks - 1.0) / (2.0 * batch)
    statistic = ((standard_normal_cdf(projected) - expected) ** 2).sum(dim=0) + 1.0 / (
        12.0 * batch
    )
    return statistic.mean()


@dataclass(frozen=True)
class PretrainLoss:
    """The loss and its parts, so a log line can say which term is moving."""

    total: Tensor
    prediction: Tensor
    regularisation: Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "prediction": float(self.prediction.detach()),
            "sigreg": float(self.regularisation.detach()),
        }


def pretrain_loss(
    predicted: Tensor,
    target: Tensor,
    *,
    sigreg_weight: float = DEFAULT_SIGREG_WEIGHT,
    projections: int = DEFAULT_PROJECTIONS,
    generator: torch.Generator | None = None,
) -> PretrainLoss:
    """`(1 - alpha) * L1 + alpha * SIGReg`, on `(B, D)` predicted and target latents."""
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted {tuple(predicted.shape)} and target {tuple(target.shape)} must match"
        )
    prediction = F.l1_loss(predicted, target)
    regularisation = sigreg(predicted, projections=projections, generator=generator)
    total = (1.0 - sigreg_weight) * prediction + sigreg_weight * regularisation
    return PretrainLoss(total=total, prediction=prediction, regularisation=regularisation)


__all__ = [
    "DEFAULT_PROJECTIONS",
    "DEFAULT_SIGREG_WEIGHT",
    "PretrainLoss",
    "pretrain_loss",
    "sample_horizons",
    "sigreg",
    "standard_normal_cdf",
]

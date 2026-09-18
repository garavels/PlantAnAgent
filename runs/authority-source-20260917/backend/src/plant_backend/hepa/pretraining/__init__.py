"""Stage 1: self-supervised pretraining of the representation.

    objective   the loss: L1 on latents + SIGReg, and log-uniform horizon sampling
    train       the loop, the checkpoint, and the CLI

Importing this package imports torch (`uv sync --group hepa`).
"""

from .objective import PretrainLoss, pretrain_loss, sample_horizons, sigreg
from .train import PretrainConfig, PretrainRun, plan, pretrain

__all__ = [
    "PretrainConfig",
    "PretrainLoss",
    "PretrainRun",
    "plan",
    "pretrain",
    "pretrain_loss",
    "sample_horizons",
    "sigreg",
]

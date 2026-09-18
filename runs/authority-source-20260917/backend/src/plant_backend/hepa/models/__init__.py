"""The HEPA network. Importing this package imports torch.

    config      shapes and hyperparameters (`HepaConfig`)
    encoder     the causal patch Transformer, `f_theta`
    predictor   the horizon-conditioned predictor, `g_phi`
    heads       the survival event head
    hepa        the three of them assembled (`Hepa`)

Install torch with `uv sync --group hepa`; the rest of the HEPA package works
without it.
"""

from .config import HepaConfig
from .encoder import AttentionPooling, CausalPatchEncoder, instance_normalise
from .heads import SurvivalEventHead, anchor_coverage, anchor_steps
from .hepa import Hepa
from .predictor import HorizonEmbedding, HorizonPredictor

__all__ = [
    "AttentionPooling",
    "CausalPatchEncoder",
    "Hepa",
    "HepaConfig",
    "HorizonEmbedding",
    "HorizonPredictor",
    "SurvivalEventHead",
    "anchor_coverage",
    "anchor_steps",
    "instance_normalise",
]

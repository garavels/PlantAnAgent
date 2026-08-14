"""The model's shape, in one dataclass, serialisable into every checkpoint.

Defaults follow the paper's reported hyperparameters (`d_model` 256, 2 encoder
layers, 4 heads, patch size 16) with the plant's own numbers where the paper's do
not apply: the input dimension is the telemetry schema's width, the event count is
the number of centrally defined events, and the horizon range is this plant's
deepest horizon in HEPA samples (288 = 24 h at 300 s).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..events import EVENT_COUNT
from ..horizons import HorizonGrid
from ..telemetry import FEATURE_COUNT


@dataclass(frozen=True)
class HepaConfig:
    """Architecture and shapes. Nothing here depends on how it is trained."""

    #: Telemetry features per sample (S).
    input_dim: int = FEATURE_COUNT
    #: Latent width (D). The paper's `d_model`.
    d_model: int = 256
    #: Causal Transformer depth and width.
    num_layers: int = 2
    num_heads: int = 4
    ff_multiplier: int = 4
    dropout: float = 0.1
    #: Non-overlapping patch size (P) in samples. The paper's default.
    patch_size: int = 16
    #: Context length (C) in samples; must be a whole number of patches.
    context_len: int = 144
    #: Deepest horizon (K) in samples the predictor is conditioned on.
    max_horizon_steps: int = 288
    #: Events the head predicts (E), in `EVENT_NAMES` order.
    event_count: int = EVENT_COUNT
    #: Hidden width of the horizon-conditioned predictor MLP, as a multiple of D.
    predictor_multiplier: int = 2

    def __post_init__(self) -> None:
        if self.context_len % self.patch_size != 0:
            raise ValueError(
                f"context_len={self.context_len} must be a whole number of patches of "
                f"{self.patch_size} samples"
            )
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by num_heads={self.num_heads}"
            )
        for name in ("input_dim", "d_model", "num_layers", "num_heads", "patch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_horizon_steps < self.patch_size:
            raise ValueError("max_horizon_steps must cover at least one patch")

    @property
    def context_patches(self) -> int:
        """Tokens the encoder sees for one context."""
        return self.context_len // self.patch_size

    @property
    def horizon_anchors(self) -> int:
        """Patch-resolution hazard blocks covering the deepest horizon.

        The survival head evaluates the predictor once per anchor rather than once
        per sample: 18 evaluations instead of 288 for the same monotone
        probability, with hazards held constant inside a patch. See `heads.py`.
        """
        return -(-self.max_horizon_steps // self.patch_size)

    @classmethod
    def for_grid(cls, grid: HorizonGrid, **overrides: Any) -> "HepaConfig":
        """A config whose horizon range matches the horizons a dataset labels."""
        return cls(max_horizon_steps=grid.max_steps, **overrides)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ff_multiplier": self.ff_multiplier,
            "dropout": self.dropout,
            "patch_size": self.patch_size,
            "context_len": self.context_len,
            "max_horizon_steps": self.max_horizon_steps,
            "event_count": self.event_count,
            "predictor_multiplier": self.predictor_multiplier,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "HepaConfig":
        if not raw:
            return cls()
        base = cls()
        known = set(base.to_dict())
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown model settings: {', '.join(sorted(unknown))}")
        return cls(
            input_dim=int(raw.get("input_dim", base.input_dim)),
            d_model=int(raw.get("d_model", base.d_model)),
            num_layers=int(raw.get("num_layers", base.num_layers)),
            num_heads=int(raw.get("num_heads", base.num_heads)),
            ff_multiplier=int(raw.get("ff_multiplier", base.ff_multiplier)),
            dropout=float(raw.get("dropout", base.dropout)),
            patch_size=int(raw.get("patch_size", base.patch_size)),
            context_len=int(raw.get("context_len", base.context_len)),
            max_horizon_steps=int(raw.get("max_horizon_steps", base.max_horizon_steps)),
            event_count=int(raw.get("event_count", base.event_count)),
            predictor_multiplier=int(
                raw.get("predictor_multiplier", base.predictor_multiplier)
            ),
        )


__all__ = ["HepaConfig"]

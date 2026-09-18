"""HEPA, assembled: encoder, horizon-conditioned predictor, survival event head.

Stage 1 (self-supervised) uses the first two:

    context (B, C, S) --f_theta--> h_t (B, D) --g_phi(.,Δt)--> h_hat (B, D)
    future  (B, K, S) --f_theta--> pooled target (B, D)                 <- compared against

Stage 2 (event fine-tuning) keeps `f_theta` frozen and reuses the same `g_phi`:

    context --f_theta(frozen)--> h_t --g_phi(., anchors)--> (B, M, D) --head--> P(event | Δt)

Nothing about the plant is baked in beyond `HepaConfig`: the input width comes from
the telemetry schema and the event count from the event rules, so adding a sensor or
an event is a config change and a recollection, not a model rewrite.

The model never sees a future sample outside the pretraining target path. At serving
time `event_probabilities` takes a context and nothing else -- there is no argument
through which the future could arrive.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import torch
from torch import Tensor, nn

from .config import HepaConfig
from .encoder import CausalPatchEncoder
from .heads import SurvivalEventHead, anchor_steps
from .predictor import HorizonPredictor


class Hepa(nn.Module):
    """The whole architecture. One `state_dict`, both stages."""

    def __init__(self, config: HepaConfig | None = None) -> None:
        super().__init__()
        self.config = config or HepaConfig()
        self.encoder = CausalPatchEncoder(self.config)
        self.predictor = HorizonPredictor(self.config)
        self.event_head = SurvivalEventHead(self.config)

    # --- representations ---------------------------------------------------
    def encode(self, context: Tensor) -> Tensor:
        """`h_t`: `(B, C, S)` -> `(B, D)`."""
        return self.encoder.encode_context(context)

    def predict_latent(self, context_latent: Tensor, horizon_steps: Tensor) -> Tensor:
        """`g_phi(h_t, Δt)`: `(B, D)` + `(B,)` -> `(B, D)`, or `(B, M)` -> `(B, M, D)`."""
        return self.predictor(context_latent, horizon_steps)

    def target_latent(self, future: Tensor, horizon_steps: Tensor) -> Tensor:
        """The representation the predictor is aiming at: `(B, K, S)` -> `(B, D)`."""
        return self.encoder.encode_future(future, horizon_steps)

    def forward(self, context: Tensor, future: Tensor, horizon_steps: Tensor) -> tuple[Tensor, Tensor]:
        """The stage-1 pair: `(predicted latent, target latent)`, both `(B, D)`."""
        context_latent = self.encode(context)
        predicted = self.predict_latent(context_latent, horizon_steps)
        target = self.target_latent(future, horizon_steps)
        return predicted, target

    # --- event prediction --------------------------------------------------
    def anchor_horizons(self, device: torch.device | None = None) -> Tensor:
        """`(M,)` horizons the hazard blocks are evaluated at."""
        return anchor_steps(self.config, device)

    def hazard_latents(self, context: Tensor) -> Tensor:
        """`(B, C, S)` -> `(B, M, D)`: one predicted latent per hazard anchor."""
        context_latent = self.encode(context)
        anchors = self.anchor_horizons(context.device).unsqueeze(0).expand(
            context.shape[0], -1
        )
        return self.predict_latent(context_latent, anchors)

    def event_probabilities(self, context: Tensor, horizon_steps: Sequence[int] | Tensor) -> Tensor:
        """`P(event within Δt)`: `(B, C, S)` -> `(B, E, H)`, monotone in Δt."""
        horizons = torch.as_tensor(
            list(horizon_steps) if isinstance(horizon_steps, Sequence) else horizon_steps,
            device=context.device,
            dtype=torch.float32,
        )
        return self.event_head.probabilities(self.hazard_latents(context), horizons)

    def event_logits(self, context: Tensor, horizon_steps: Sequence[int] | Tensor) -> Tensor:
        """The same quantity as logits: what the fine-tuning BCE consumes."""
        horizons = torch.as_tensor(
            list(horizon_steps) if isinstance(horizon_steps, Sequence) else horizon_steps,
            device=context.device,
            dtype=torch.float32,
        )
        return self.event_head.logit_probabilities(self.hazard_latents(context), horizons)

    # --- stage handling ----------------------------------------------------
    def freeze_encoder(self) -> None:
        """Stage 2, as the paper specifies: the representation stops moving.

        `requires_grad = False` alone is not enough -- the encoder must also stay in
        eval mode so dropout does not keep perturbing a representation that is
        supposed to be fixed. `train()` is overridden below to keep that true.
        """
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()

    def unfreeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)

    @property
    def encoder_frozen(self) -> bool:
        return not any(parameter.requires_grad for parameter in self.encoder.parameters())

    def train(self, mode: bool = True) -> "Hepa":
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def finetune_parameters(self) -> Iterator[nn.Parameter]:
        """The parameters stage 2 optimises: predictor and event head."""
        yield from self.predictor.parameters()
        yield from self.event_head.parameters()

    def parameter_counts(self) -> dict[str, int]:
        return {
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "predictor": sum(p.numel() for p in self.predictor.parameters()),
            "event_head": sum(p.numel() for p in self.event_head.parameters()),
        }


__all__ = ["Hepa"]

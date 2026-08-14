"""The stage-2 objective: positive-weighted BCE against cumulative event labels.

Three details do the work here, and all three exist because these events are rare:

*Positive weighting.* The paper weights positives by `N_neg / N_pos`. A 1 h
`co2_starvation` label is positive in well under one percent of windows, and an
unweighted BCE is minimised by predicting "no" forever -- with excellent accuracy and
no value. The weight is measured per event *and* per horizon, since a 24 h label is
far commoner than a 1 h one for the same event.

*Masking.* Labels near the end of an episode are unknown rather than negative
(`data/labels.py`), so they are dropped from the loss instead of being taught as
"nothing happened".

*Logit space.* The model's probability comes from a survival product; taking its log
and feeding `binary_cross_entropy_with_logits` keeps the whole path in log space,
where a probability of 1e-7 is still a usable gradient rather than a `log(0)`.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

#: Cap on `N_neg / N_pos`. Without it, an event with a single positive in the split
#: gets a weight in the thousands and its gradient drowns out every other event.
MAX_POSITIVE_WEIGHT = 100.0


def positive_weights(
    labels: Tensor, valid: Tensor, *, maximum: float = MAX_POSITIVE_WEIGHT
) -> Tensor:
    """`N_neg / N_pos` per event and horizon, from `(N, E, H)` labels and `(N, H)` masks."""
    mask = valid.unsqueeze(1).to(labels.dtype)
    positives = (labels * mask).sum(dim=0)
    counted = mask.expand_as(labels).sum(dim=0)
    negatives = counted - positives
    weights = torch.where(
        positives > 0, negatives / positives.clamp(min=1.0), torch.ones_like(positives)
    )
    return weights.clamp(min=1.0, max=maximum)


def event_loss(
    logits: Tensor,
    labels: Tensor,
    valid: Tensor,
    *,
    weights: Tensor | None = None,
) -> Tensor:
    """Masked, positive-weighted BCE. `logits`/`labels` `(B, E, H)`, `valid` `(B, H)`."""
    if logits.shape != labels.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)} and labels {tuple(labels.shape)} must match"
        )
    mask = valid.unsqueeze(1).to(logits.dtype).expand_as(logits)
    per_element = F.binary_cross_entropy_with_logits(
        logits, labels.to(logits.dtype), reduction="none"
    )
    if weights is not None:
        # Weight positives only, exactly as `pos_weight` would, but per (event, horizon).
        per_element = per_element * (1.0 + (weights.unsqueeze(0) - 1.0) * labels)
    total = (per_element * mask).sum()
    counted = mask.sum().clamp(min=1.0)
    return total / counted


__all__ = ["MAX_POSITIVE_WEIGHT", "event_loss", "positive_weights"]

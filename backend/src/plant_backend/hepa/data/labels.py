"""Horizon labels: "did this event happen in the next Δt", per step, per event.

    trajectory
        -> observation time t          (the model sees samples <= t)
        -> future interval (t, t+Δt]   (the label reads samples > t)
        -> label(event, horizon)

Two rules make the difference between a real predictor and a leaking one, and both
are enforced here:

*Strictly future.* The interval is half-open on the left: sample `t` itself is
part of the input, never part of the label. An event that is already happening at
`t` is not a prediction, and a label that included it would let the model score
well by reporting the present.

*Never truncate.* A label for a 24 h horizon needs 24 h of recorded future. Where
the episode ends first, the label is not "no event" -- it is unknown, and the
returned mask marks it so, so the loss can ignore it. Silently treating the tail
of every episode as negative is a reliable way to publish an optimistic AUPRC.

The label is *cumulative* ("any time within Δt"), matching the survival form the
model predicts: `p(t, Δt) = 1 - prod_j (1 - lambda_j)` is by construction the
probability of at least one occurrence in the interval, and is monotone in Δt --
so the labels must be monotone in Δt too, which cumulative labels are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class HorizonLabels:
    """Labels for a whole trajectory.

    Shapes:
        labels (T, E, H) float32  1.0 if the event occurs in (t, t+Δt_h]
        valid  (T, H)    bool     False where the episode ends inside the horizon
    """

    labels: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        if self.labels.ndim != 3 or self.valid.ndim != 2:
            raise ValueError("labels must be (T, E, H) and valid must be (T, H)")
        if self.labels.shape[0] != self.valid.shape[0] or (
            self.labels.shape[2] != self.valid.shape[1]
        ):
            raise ValueError("labels and valid disagree about T or H")

    @property
    def event_count(self) -> int:
        return int(self.labels.shape[1])

    def positive_rate(self) -> np.ndarray:
        """`(E, H)` share of positives among *valid* labels. For loss weighting."""
        mask = self.valid[:, None, :]
        counted = mask.sum(axis=0)
        positives = (self.labels * mask).sum(axis=0)
        return np.divide(
            positives, counted, out=np.zeros_like(positives), where=counted > 0
        )


def horizon_labels(occurrences: np.ndarray, steps: Sequence[int]) -> HorizonLabels:
    """Per-step cumulative labels for each horizon, from `(T, E)` occurrences.

    `steps` are horizons in *samples* (`HorizonGrid.steps`), not hours.
    """
    events = np.asarray(occurrences, dtype=bool)
    if events.ndim != 2:
        raise ValueError("occurrences must be (T, E)")
    horizons = [int(step) for step in steps]
    if not horizons or any(step < 1 for step in horizons):
        raise ValueError("horizons must be one or more positive step counts")

    total, event_count = events.shape
    # cumulative[i] = number of occurrences in samples [0, i), so the count inside
    # a half-open interval (t, t+K] is one subtraction and no Python loop over t.
    cumulative = np.concatenate(
        [np.zeros((1, event_count), dtype=np.int64), np.cumsum(events, axis=0, dtype=np.int64)]
    )
    starts = np.arange(total) + 1

    labels = np.zeros((total, event_count, len(horizons)), dtype=np.float32)
    valid = np.zeros((total, len(horizons)), dtype=bool)
    for index, step in enumerate(horizons):
        ends = np.minimum(starts + step, total)
        counts = cumulative[ends] - cumulative[np.minimum(starts, total)]
        labels[:, :, index] = (counts > 0).astype(np.float32)
        valid[:, index] = (starts + step) <= total
    return HorizonLabels(labels=labels, valid=valid)


def observable_range(total: int, context_len: int, max_steps: int) -> np.ndarray:
    """The observation times a full sample can be built at.

    A window needs `context_len` samples up to and including `t`, and `max_steps`
    samples after it -- the latter both for the deepest label and for the target
    representation the pretraining loss compares against.
    """
    if context_len < 1 or max_steps < 1:
        raise ValueError("context_len and max_steps must be positive")
    first = context_len - 1
    last = total - max_steps - 1
    if last < first:
        return np.zeros(0, dtype=np.int64)
    return np.arange(first, last + 1, dtype=np.int64)


__all__ = ["HorizonLabels", "horizon_labels", "observable_range"]

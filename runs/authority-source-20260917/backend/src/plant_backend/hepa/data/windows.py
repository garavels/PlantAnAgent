"""Windowing: trajectories -> the fixed-shape samples both training stages consume.

One window, observed at sample `t` of one episode:

    context  (C, S)      samples t-C+1 .. t          the only thing the model reads
    future   (K, S)      samples t+1 .. t+K          targets only, never an input
    labels   (E, H)      cumulative event labels     derived from `future`'s events
    valid    (H,)        which horizons are complete

`future` earns its place because HEPA's pretraining target is a *representation of
the future*, produced by the same weight-shared encoder: the loss needs the future
samples even though the model must never be given them at prediction time. Keeping
them in a separate field, with the shapes above, is what makes that separation
auditable -- `context` is the entire model input, in both stages and at serving
time (`inference.py` builds a context and nothing else).

Windows are enumerated once into an index of `(episode, t)` pairs, so a Dataset is
a cheap lookup and a shuffle is a permutation of integers rather than of arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..events import EVENT_NAMES
from ..horizons import HorizonGrid
from ..telemetry import FEATURE_NAMES, NormalisationStats
from .labels import horizon_labels, observable_range
from .trajectories import Trajectory

#: Twelve hours of context at the default five-minute sample period, which is nine
#: patches of 16 samples for the encoder: long enough to contain a full solar ramp
#: and an overnight discharge, short enough to stay cheap. Divisibility by the
#: patch size is not decoration -- the encoder refuses a context it cannot patch.
DEFAULT_CONTEXT_LEN = 144


@dataclass(frozen=True)
class WindowSpec:
    """How a trajectory is cut up: context length and the horizons to label."""

    context_len: int = DEFAULT_CONTEXT_LEN
    horizons: HorizonGrid = HorizonGrid.of_hours()

    def __post_init__(self) -> None:
        if self.context_len < 1:
            raise ValueError("context_len must be positive")

    @property
    def max_steps(self) -> int:
        return self.horizons.max_steps

    @property
    def min_samples(self) -> int:
        """Shortest episode that yields a single window."""
        return self.context_len + self.max_steps

    def to_dict(self) -> dict[str, object]:
        return {"context_len": self.context_len, "horizons": self.horizons.to_dict()}


@dataclass(frozen=True)
class Window:
    """One training sample. See the module docstring for the shapes."""

    context: np.ndarray
    future: np.ndarray
    labels: np.ndarray
    valid: np.ndarray
    episode_id: str
    observed_at: int


class WindowIndex:
    """Every valid `(episode, t)` window across a set of trajectories."""

    def __init__(
        self,
        trajectories: Sequence[Trajectory],
        spec: WindowSpec,
        *,
        normalisation: NormalisationStats | None = None,
    ) -> None:
        self.spec = spec
        self.trajectories = list(trajectories)
        self.normalisation = normalisation or NormalisationStats.identity()
        self._labels = [
            horizon_labels(trajectory.events, spec.horizons.steps)
            for trajectory in self.trajectories
        ]
        pairs: list[tuple[int, int]] = []
        for index, trajectory in enumerate(self.trajectories):
            for observed_at in observable_range(
                len(trajectory), spec.context_len, spec.max_steps
            ):
                pairs.append((index, int(observed_at)))
        self.pairs: tuple[tuple[int, int], ...] = tuple(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def window(self, position: int) -> Window:
        episode, observed_at = self.pairs[position]
        trajectory = self.trajectories[episode]
        spec = self.spec
        context = trajectory.features[observed_at - spec.context_len + 1 : observed_at + 1]
        future = trajectory.features[observed_at + 1 : observed_at + 1 + spec.max_steps]
        labels = self._labels[episode]
        return Window(
            context=self.normalisation.apply(context),
            future=self.normalisation.apply(future),
            labels=labels.labels[observed_at],
            valid=labels.valid[observed_at],
            episode_id=trajectory.metadata.episode_id,
            observed_at=observed_at,
        )

    def positive_rate(self) -> np.ndarray:
        """`(E, H)` positive share over the indexed windows. For BCE weighting."""
        if not self.pairs:
            shape = (len(EVENT_NAMES), len(self.spec.horizons.steps))
            return np.zeros(shape, dtype=np.float32)
        stacked = np.stack(
            [self._labels[episode].labels[at] for episode, at in self.pairs]
        )
        return stacked.mean(axis=0).astype(np.float32)

    def fit_normalisation(self, names: Sequence[str] | None = None) -> NormalisationStats:
        """Statistics over the *context* samples of the indexed windows only.

        Contexts, not whole trajectories: the statistics must describe what the
        model is actually fed, and must not read any sample that only ever appears
        in a future interval.
        """
        feature_names = tuple(names) if names else FEATURE_NAMES
        blocks = []
        for position in range(len(self)):
            episode, observed_at = self.pairs[position]
            trajectory = self.trajectories[episode]
            blocks.append(
                trajectory.features[observed_at - self.spec.context_len + 1 : observed_at + 1]
            )
        return NormalisationStats.fit(blocks, feature_names)


__all__ = ["DEFAULT_CONTEXT_LEN", "Window", "WindowIndex", "WindowSpec"]

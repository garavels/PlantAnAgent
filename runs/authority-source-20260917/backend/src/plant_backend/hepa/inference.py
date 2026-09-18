"""Serving: telemetry history in, `P(event within Δt)` out.

    predictor = EventPredictor.load("checkpoints/hepa/stage2")
    predictions = predictor.predict(telemetry_history=history, horizons=["1h", "6h"])
    # {'battery_reserve_violation': {'1h': 0.02, '6h': 0.31}, 'co2_starvation': {...}}

The structure is the stable part of this module: a mapping from event name to a mapping
from horizon label to probability. Callers -- the API, a notebook, a future supervisor --
depend on that shape and on nothing else; the model behind it can be retrained, resized
or replaced.

Three things this class refuses to do, all of them for the same reason (a wrong number
that looks right is worse than an error):

* it will not answer from a checkpoint whose feature schema, event rules or horizons
  disagree with this build -- `checkpoints.py` raises instead;
* it will not answer from fewer samples than the context it was trained on, and it will
  not pad a short history with zeros;
* it will not answer a horizon the checkpoint was not fine-tuned for, even though the
  survival head could produce a number for any Δt.

The history it takes is a sequence of `PlantState` *telemetry* at the checkpoint's
sample period, oldest first. Nothing in the call signature can carry a future sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from plant_sim import PlantConfig, PlantState

from . import checkpoints, evaluation
from .checkpoints import CheckpointMetadata
from .data.trajectories import TrajectoryDataset
from .data.windows import WindowIndex, WindowSpec
from .telemetry import FeatureEncoder

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .models import Hepa


class InferenceError(ValueError):
    """The request could not be answered as asked. The message says what was wrong."""


@dataclass(frozen=True)
class Prediction:
    """One answer: the probabilities, plus what produced them."""

    probabilities: dict[str, dict[str, float]]
    horizons: tuple[str, ...]
    observed_at_s: float | None
    checkpoint: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "probabilities": self.probabilities,
            "horizons": list(self.horizons),
            "observed_at_s": self.observed_at_s,
            "checkpoint": self.checkpoint,
        }


class EventPredictor:
    """A loaded stage-2 checkpoint, ready to answer about a telemetry history."""

    def __init__(
        self,
        model: "Hepa",
        metadata: CheckpointMetadata,
        *,
        plant: PlantConfig | None = None,
    ) -> None:
        self.model = model
        self.metadata = metadata
        # The feature schema scales against the plant's *design* (rated power, tank
        # capacity), so the timestep plays no part here; a re-sized plant does.
        self.encoder = FeatureEncoder(plant or PlantConfig())
        self.model.eval()

    @classmethod
    def load(cls, path: Path | str, *, plant: PlantConfig | None = None) -> "EventPredictor":
        """Load a checkpoint, refusing anything that is not a fine-tuned model."""
        model, metadata = checkpoints.load_model(path)
        if metadata.stage != checkpoints.STAGE_FINETUNE:
            raise InferenceError(
                f"{path} is a {metadata.stage} checkpoint: only a "
                f"{checkpoints.STAGE_FINETUNE} checkpoint predicts events"
            )
        return cls(model, metadata, plant=plant)

    # --- properties --------------------------------------------------------
    @property
    def event_names(self) -> tuple[str, ...]:
        return self.metadata.event_names

    @property
    def horizon_labels(self) -> tuple[str, ...]:
        return self.metadata.horizons.labels

    @property
    def context_len(self) -> int:
        return self.metadata.context_len

    def describe(self) -> dict[str, Any]:
        """What this predictor is, for a status endpoint or a log line."""
        return {
            "stage": self.metadata.stage,
            "trained": self.metadata.trained,
            "events": list(self.event_names),
            "horizons": list(self.horizon_labels),
            "context_len": self.context_len,
            "sample_period_s": self.metadata.horizons.sample_period_s,
            "created_at": self.metadata.created_at,
            "pretrained_from": self.metadata.pretrained_from,
            "notes": self.metadata.notes,
        }

    # --- prediction --------------------------------------------------------
    def predict(
        self,
        telemetry_history: Sequence[PlantState],
        horizons: Sequence[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """`P(event within Δt)` per event and horizon label. The stable interface."""
        return self.predict_detailed(telemetry_history, horizons).probabilities

    def predict_detailed(
        self,
        telemetry_history: Sequence[PlantState],
        horizons: Sequence[str] | None = None,
    ) -> Prediction:
        """The same answer with provenance attached."""
        features = self.encoder.encode_many(telemetry_history)
        labels = self._resolve_horizons(horizons)
        probabilities = self.predict_features(features, labels)
        observed_at = (
            float(telemetry_history[-1].time_s) if len(telemetry_history) else None
        )
        return Prediction(
            probabilities={
                event: {label: float(probabilities[index, position]) for position, label in enumerate(labels)}
                for index, event in enumerate(self.event_names)
            },
            horizons=tuple(labels),
            observed_at_s=observed_at,
            checkpoint=self.describe(),
        )

    def predict_features(
        self, features: np.ndarray, horizons: Sequence[str] | None = None
    ) -> np.ndarray:
        """`(T, S)` scaled telemetry -> `(E, H)` probabilities. No normalisation applied yet."""
        import torch

        labels = self._resolve_horizons(horizons)
        history = np.asarray(features, dtype=np.float32)
        if history.ndim != 2:
            raise InferenceError("telemetry history must be (T, S)")
        if history.shape[0] < self.context_len:
            raise InferenceError(
                f"need {self.context_len} telemetry samples at "
                f"{self.metadata.horizons.sample_period_s:g} s spacing, got {history.shape[0]}"
            )
        if history.shape[1] != len(self.metadata.feature_names):
            raise InferenceError(
                f"expected {len(self.metadata.feature_names)} features per sample, "
                f"got {history.shape[1]}"
            )
        context = self.metadata.normalisation.apply(history[-self.context_len :])
        steps = [self.metadata.horizons.steps_for(label) for label in labels]
        with torch.no_grad():
            batch = torch.from_numpy(context).unsqueeze(0)
            probabilities = self.model.event_probabilities(batch, steps)
        return probabilities[0].cpu().numpy()

    def _resolve_horizons(self, horizons: Sequence[str] | None) -> tuple[str, ...]:
        if horizons is None:
            return self.horizon_labels
        requested = tuple(str(label) for label in horizons)
        if not requested:
            raise InferenceError("at least one horizon must be requested")
        unknown = [label for label in requested if label not in self.horizon_labels]
        if unknown:
            raise InferenceError(
                f"horizon(s) {', '.join(unknown)} were not fine-tuned for; this checkpoint "
                f"answers {', '.join(self.horizon_labels)}"
            )
        return requested

    # --- offline evaluation -------------------------------------------------
    def evaluate_split(
        self,
        dataset_root: Path | str,
        *,
        split: str = "test",
        batch_size: int = 64,
        threshold: float = evaluation.DEFAULT_THRESHOLD,
    ) -> dict[str, Any]:
        """Score every window of one split. The only place metrics are produced."""
        import torch

        dataset = TrajectoryDataset.load(dataset_root)
        grid = self.metadata.horizons
        spec = WindowSpec(context_len=self.context_len, horizons=grid)
        index = WindowIndex(
            dataset.load_split(split), spec, normalisation=self.metadata.normalisation
        )
        if len(index) == 0:
            raise InferenceError(f"no windows in the {split} split of {dataset_root}")

        contexts: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        valid: list[np.ndarray] = []
        for position in range(len(index)):
            window = index.window(position)
            contexts.append(window.context)
            labels.append(window.labels)
            valid.append(window.valid)

        steps = list(grid.steps)
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(contexts), batch_size):
                batch = torch.from_numpy(np.stack(contexts[start : start + batch_size]))
                predictions.append(
                    self.model.event_probabilities(batch, steps).cpu().numpy()
                )

        report = evaluation.evaluate(
            np.concatenate(predictions),
            np.stack(labels),
            np.stack(valid),
            event_names=self.event_names,
            horizon_labels=grid.labels,
            threshold=threshold,
        )
        return {
            "split": split,
            "windows": len(index),
            "dataset": str(dataset.root),
            "checkpoint": self.describe(),
            **report,
        }


def status(path: Path | str | None) -> dict[str, Any]:
    """Whether a usable checkpoint is present, without importing torch.

    Shaped for `GET /api/hepa/status`: it never raises, and it says *why* when the
    answer is no, because "no model" and "a model this build cannot read" are
    different problems for whoever has to fix it.
    """
    if path is None:
        return {
            "available": False,
            "reason": (
                f"no checkpoint configured; set {checkpoints.CHECKPOINT_ENV} to a "
                "fine-tuned HEPA checkpoint directory"
            ),
        }
    try:
        metadata = checkpoints.load_metadata(path)
    except checkpoints.HepaCheckpointError as reason:
        return {"available": False, "path": str(path), "reason": str(reason)}
    available = metadata.stage == checkpoints.STAGE_FINETUNE
    return {
        "available": available,
        "path": str(path),
        "stage": metadata.stage,
        "trained": metadata.trained,
        "events": list(metadata.event_names),
        "horizons": list(metadata.horizons.labels),
        "context_len": metadata.context_len,
        "sample_period_s": metadata.horizons.sample_period_s,
        "created_at": metadata.created_at,
        "metrics": metadata.metrics,
        "notes": metadata.notes,
        **(
            {}
            if available
            else {
                "reason": (
                    f"checkpoint at {path} is stage {metadata.stage}; event prediction "
                    f"needs a {checkpoints.STAGE_FINETUNE} checkpoint"
                )
            }
        ),
    }


__all__ = ["EventPredictor", "InferenceError", "Prediction", "status"]

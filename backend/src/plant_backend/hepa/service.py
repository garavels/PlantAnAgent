"""The HEPA inference service: the only door between the plant and the model.

    Agent -> HEPA tool -> this service -> a fine-tuned checkpoint

Everything above this module speaks in semantic event probabilities:

    forecast = CheckpointForecaster().forecast(frames, ["1h", "6h", "12h"])
    forecast.status           # "active" | "unavailable" | "error"
    forecast.probabilities    # {'co2_starvation': {'1h': 0.01, '6h': 0.52}, ...}

Nothing above it sees a tensor, a normalisation table, a context length or a
checkpoint path -- the supervisor asks "what may happen?", not "which weights
answered?".

The service never raises. A supervisor whose forecast provider can throw is a
supervisor that can take the plant down, so the failure modes are values:

    unavailable   no checkpoint configured, torch not installed, a checkpoint
                  this build cannot trust, or too little history to answer from
    error         anything unexpected, recorded with its type

and in both cases `probabilities` is empty. There is no fallback number: a
fabricated probability is worse than no probability, because someone would act
on it.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Sequence

from . import checkpoints, telemetry
from .inference import InferenceError, status as checkpoint_status

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .inference import EventPredictor

log = logging.getLogger(__name__)

#: What a forecast can be. `active` is the only one that carries numbers.
ACTIVE = "active"
UNAVAILABLE = "unavailable"
ERROR = "error"

#: Where the numbers came from. Anything that is not a checkpoint says so.
SOURCE_CHECKPOINT = "checkpoint"


class ForecastUnavailable(RuntimeError):
    """No forecast can be served, and the message says what is missing."""


@dataclass(frozen=True)
class Forecast:
    """One answer from HEPA, or one honest explanation of its absence."""

    status: str
    probabilities: dict[str, dict[str, float]] = field(default_factory=dict)
    horizons: tuple[str, ...] = ()
    observed_at_s: float | None = None
    #: `checkpoint`, or `fixture:<name>` for the deterministic test doubles.
    source: str = SOURCE_CHECKPOINT
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.status == ACTIVE and bool(self.probabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "probabilities": {
                event: dict(per_horizon) for event, per_horizon in self.probabilities.items()
            },
            "horizons": list(self.horizons),
            "observed_at_s": self.observed_at_s,
            "source": self.source,
            "reason": self.reason,
        }

    @classmethod
    def unavailable(cls, reason: str, *, source: str = SOURCE_CHECKPOINT) -> "Forecast":
        return cls(status=UNAVAILABLE, reason=reason, source=source)

    @classmethod
    def failed(cls, reason: str, *, source: str = SOURCE_CHECKPOINT) -> "Forecast":
        return cls(status=ERROR, reason=reason, source=source)


class Forecaster(Protocol):
    """Anything that can answer "what may happen?" from a telemetry history."""

    def forecast(
        self,
        frames: Sequence[dict[str, Any]],
        horizons: Sequence[str] | None = None,
    ) -> Forecast: ...

    def status(self) -> dict[str, Any]: ...


def checkpoint_path() -> str | None:
    """The configured checkpoint directory, or None. No local path is hardcoded."""
    value = os.environ.get(checkpoints.CHECKPOINT_ENV, "").strip()
    return value or None


# --- the loaded model, once per process -----------------------------------
_predictor: EventPredictor | None = None
_predictor_path: str | None = None


def reset_cache() -> None:
    """Drop the loaded model. For tests, and for pointing at a new checkpoint."""
    global _predictor, _predictor_path
    _predictor = None
    _predictor_path = None


def load_predictor(path: str) -> EventPredictor:
    """The cached `EventPredictor` for `path`.

    Raises `ForecastUnavailable` -- never a torch import error, never a
    checkpoint-format error -- so every caller has exactly one failure to handle.
    """
    global _predictor, _predictor_path
    if _predictor is not None and _predictor_path == path:
        return _predictor
    try:
        from .inference import EventPredictor  # noqa: PLC0415 - torch is optional
    except ModuleNotFoundError as reason:  # torch, or one of its dependencies
        raise ForecastUnavailable(
            f"HEPA inference needs the optional model dependencies ({reason.name}): "
            "install them with `uv sync --group hepa`"
        ) from reason
    try:
        _predictor = EventPredictor.load(path)
    except ModuleNotFoundError as reason:
        raise ForecastUnavailable(
            f"HEPA inference needs the optional model dependencies ({reason.name}): "
            "install them with `uv sync --group hepa`"
        ) from reason
    except (checkpoints.HepaCheckpointError, InferenceError) as reason:
        raise ForecastUnavailable(str(reason)) from reason
    _predictor_path = path
    return _predictor


def resample(frames: Sequence[dict[str, Any]], sample_period_s: float) -> list[dict[str, Any]]:
    """The live history at the model's sample period, anchored on the newest frame.

    The plant runs at a 60 s timestep and the model is trained on 300 s samples, so
    handing the raw history over would compress a 12 h context into 2.4 h of plant.
    The spacing is measured from the frames themselves rather than assumed, and the
    walk runs backwards from `now`, because the sample that matters most is the last
    one.
    """
    if len(frames) < 2:
        return list(frames)
    spacing = float(frames[-1].get("time_s", 0.0)) - float(frames[-2].get("time_s", 0.0))
    if spacing <= 0.0:
        return list(frames)
    stride = max(1, int(round(sample_period_s / spacing)))
    return list(reversed(list(frames)[::-1][::stride]))


class CheckpointForecaster:
    """HEPA behind a checkpoint, with every failure turned into a `Forecast`.

    The checkpoint is found through the environment (`PLANT_HEPA_CHECKPOINT`),
    loaded on the first forecast and cached afterwards, so a backend with no
    checkpoint configured costs nothing and behaves exactly as it did before.
    """

    def forecast(
        self,
        frames: Sequence[dict[str, Any]],
        horizons: Sequence[str] | None = None,
    ) -> Forecast:
        path = checkpoint_path()
        if path is None:
            return Forecast.unavailable(
                f"no HEPA checkpoint configured; set {checkpoints.CHECKPOINT_ENV} to a "
                "fine-tuned checkpoint directory"
            )
        if not Path(path).exists():
            return Forecast.unavailable(
                f"no HEPA checkpoint at {path} ({checkpoints.CHECKPOINT_ENV})"
            )
        try:
            predictor = load_predictor(path)
        except ForecastUnavailable as reason:
            return Forecast.unavailable(str(reason))
        try:
            history = [
                telemetry.state_from_frame(frame)
                for frame in resample(frames, predictor.metadata.horizons.sample_period_s)
            ]
            prediction = predictor.predict_detailed(history, horizons)
        except (InferenceError, TypeError, ValueError) as reason:
            return Forecast.unavailable(str(reason))
        except Exception as reason:  # noqa: BLE001 - the plant outranks the forecast
            log.warning("HEPA inference failed: %s", reason, exc_info=True)
            return Forecast.failed(f"{type(reason).__name__}: {reason}")
        return Forecast(
            status=ACTIVE,
            probabilities=prediction.probabilities,
            horizons=prediction.horizons,
            observed_at_s=prediction.observed_at_s,
        )

    def status(self) -> dict[str, Any]:
        """Whether a forecast could be served, without loading anything."""
        return checkpoint_status(checkpoint_path())


__all__ = [
    "ACTIVE",
    "CheckpointForecaster",
    "ERROR",
    "Forecast",
    "ForecastUnavailable",
    "Forecaster",
    "UNAVAILABLE",
    "checkpoint_path",
    "load_predictor",
    "resample",
    "reset_cache",
]

"""The HTTP surface of HEPA: two read-only routes.

    GET  /api/hepa/status     is a model loaded, for which events and horizons
    POST /api/hepa/predict    P(event within Δt) for the live plant, or a given history

Both routes are read-only, and both are *one* way of asking. Since V10 the supervisory
agent asks the same question through `hepa.service`, which these routes now share, so a
prediction the control room reads and a prediction the supervisor acted on come from the
same loaded checkpoint and the same resampled history. Neither route touches the
controller or the action guard: HEPA predicts, the supervisor decides.

Everything degrades on purpose:

* torch not installed -> `503` naming `uv sync --group hepa`, and `/status` still
  answers, because status is metadata and metadata needs no tensors;
* no checkpoint configured or the path is empty -> `503` with the environment variable
  to set;
* a checkpoint this build cannot trust (different feature schema, event rules or
  horizons) -> `503` with the mismatch, never a number;
* too little history, or a horizon the model was not fine-tuned for -> `400`.

The model is loaded on the first prediction and cached, so importing the app costs
nothing and a backend with no HEPA checkpoint starts exactly as it did in V7.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import checkpoints, events as event_rules, horizons as horizon_grid, service, telemetry
from .inference import EventPredictor, InferenceError, status

router = APIRouter(prefix="/api/hepa", tags=["hepa"])

#: Returns the live plant's telemetry history, oldest first, as serialised frames.
#: Set by `app.py`; without it `/predict` needs an explicit history in the request.
_history_provider: Callable[[], Sequence[dict[str, Any]]] | None = None

#: The checkpoint lookup, the model cache and the resampling all live in `service`,
#: so these routes and the supervisor cannot disagree about any of the three.
checkpoint_path = service.checkpoint_path
reset_cache = service.reset_cache


def configure(history: Callable[[], Sequence[dict[str, Any]]] | None) -> None:
    """Tell the routes where the live telemetry history comes from."""
    global _history_provider
    _history_provider = history


def _load_predictor(path: str) -> EventPredictor:
    try:
        return service.load_predictor(path)
    except service.ForecastUnavailable as reason:
        raise HTTPException(status_code=503, detail=str(reason)) from reason


class PredictRequest(BaseModel):
    """Which horizons to answer for, and optionally which history to answer about."""

    horizons: list[str] | None = Field(
        default=None,
        description="Horizon labels such as 1h, 6h. Omit for every horizon the model has.",
    )
    telemetry: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Telemetry frames, oldest first, as /api/state returns them. Omit to use "
            "the running plant's own history."
        ),
    )


@router.get("/status")
def get_status() -> dict[str, Any]:
    """Whether a prediction can be served, and what this build's contract is.

    Answers with `available: false` and a reason rather than an error when there is
    no usable model: a control room asking "is the forecast up" should get an answer,
    not a stack trace.
    """
    return {
        "checkpoint": status(checkpoint_path()),
        "schema": {
            "telemetry": telemetry.describe(),
            "events": event_rules.describe(),
            "default_horizons": horizon_grid.HorizonGrid.of_hours().to_dict(),
        },
    }


@router.post("/predict")
def post_predict(request: PredictRequest | None = None) -> dict[str, Any]:
    """`P(event within Δt)` per event and horizon, for one point in time."""
    body = request or PredictRequest()
    path = checkpoint_path()
    if path is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"no HEPA checkpoint configured: set {checkpoints.CHECKPOINT_ENV} to a "
                "fine-tuned checkpoint directory and restart the backend"
            ),
        )
    if not Path(path).exists():
        raise HTTPException(
            status_code=503, detail=f"no HEPA checkpoint at {path} ({checkpoints.CHECKPOINT_ENV})"
        )

    predictor = _load_predictor(path)
    if body.telemetry is not None:
        frames = list(body.telemetry)
    else:
        frames = service.resample(
            _live_history(), predictor.metadata.horizons.sample_period_s
        )
    try:
        history = [telemetry.state_from_frame(frame) for frame in frames]
    except (TypeError, ValueError) as reason:
        raise HTTPException(
            status_code=400, detail=f"unreadable telemetry frame: {reason}"
        ) from reason
    try:
        prediction = predictor.predict_detailed(history, body.horizons)
    except InferenceError as reason:
        raise HTTPException(status_code=400, detail=str(reason)) from reason
    return prediction.to_dict()


def _live_history() -> list[dict[str, Any]]:
    if _history_provider is None:
        raise HTTPException(
            status_code=400,
            detail="no live plant history available: send telemetry frames in the request",
        )
    return list(_history_provider())


__all__ = ["PredictRequest", "checkpoint_path", "configure", "reset_cache", "router"]

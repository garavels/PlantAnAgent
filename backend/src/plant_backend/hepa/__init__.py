"""HEPA (V9): horizon-conditioned event prediction for the plant.

    telemetry      the feature schema the model sees (observed telemetry only)
    events         the event rules, defined once, used by every stage
    horizons       hours <-> sample steps
    data/          trajectories, labels, windows, splits, the collector
    models/        the causal encoder, the horizon predictor, the survival head
    pretraining/   stage 1: self-supervised representation learning
    finetuning/    stage 2: event fine-tuning on a frozen representation
    inference      serving: telemetry history -> P(event within Δt)
    evaluation     AUROC/AUPRC/Brier and calibration, per event and per horizon
    checkpoints    weights plus the metadata needed to refuse them

The pipeline, in order:

    simulator -> collect -> pretrain (stage 1) -> finetune (stage 2) -> predict

Torch is an optional dependency (`uv sync --group hepa`). This module imports none of
it: the schema, the event rules, the horizon grid, the labels, the dataset format, the
metrics and the checkpoint *metadata* are all torch-free, so the backend starts and
`GET /api/hepa/status` answers on a machine that has never installed it. `models`,
`pretraining`, `finetuning` and `EventPredictor` are the parts that need torch, and
they are imported where they are used.

Nothing here is wired into the controllers or the agent: HEPA answers when asked.
"""

from . import checkpoints, evaluation, events, horizons, telemetry

__all__ = ["checkpoints", "evaluation", "events", "horizons", "telemetry"]

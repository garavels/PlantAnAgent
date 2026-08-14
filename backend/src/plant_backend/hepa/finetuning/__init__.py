"""Stage 2: event fine-tuning on a frozen HEPA representation.

    objective   masked, positive-weighted BCE against cumulative event labels
    train       the loop, the checkpoint, and the CLI

Importing this package imports torch (`uv sync --group hepa`).
"""

from .objective import event_loss, positive_weights
from .train import FinetuneConfig, FinetuneRun, finetune, load_pretrained, plan

__all__ = [
    "FinetuneConfig",
    "FinetuneRun",
    "event_loss",
    "finetune",
    "load_pretrained",
    "plan",
    "positive_weights",
]

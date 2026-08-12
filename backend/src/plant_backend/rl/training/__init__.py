"""Training: the part that runs offline, without the application server.

    uv run python -m plant_backend.rl.training.train --dry-run

`config.py` is what a run *is*, `ppo.py` is how it is built, `train.py` is the
entry point. Nothing here is imported by the backend at runtime: the plant loads
checkpoints (`rl.policies`), it does not train.
"""

from .config import PPOConfig, TrainingConfig

__all__ = ["PPOConfig", "TrainingConfig"]

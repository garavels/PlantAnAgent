"""Seeding, in one place, so two runs of the same config are the same run.

Both training stages, the collector and the evaluation code draw randomness, and a
number that cannot be reproduced is a number that cannot be debugged. `seed_all`
seeds Python, NumPy and (if it is installed) torch, and returns a torch generator
for the places that should not share the global stream -- SIGReg's projections and
horizon sampling, which are drawn every step and would otherwise make an otherwise
identical rerun diverge after the first shuffle.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np


def seed_all(seed: int) -> Any:
    """Seed every generator this package uses. Returns a torch `Generator` or None."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
    except ModuleNotFoundError:
        return None
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


__all__ = ["seed_all"]

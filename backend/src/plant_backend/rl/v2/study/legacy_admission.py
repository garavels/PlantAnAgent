"""Keep historical evaluation entry points outside registered study custody.

This is a rejection boundary, never an authorization mechanism. Registered
execution belongs to the separate stage custodian and its fixed private runner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

from .. import beta_spaces
from ..training.config import BETA_REGISTERED_RUN_KIND
from ..training.episode_allocator import (
    DEVELOPMENT_SEED_MIN,
    DEVELOPMENT_SEED_MAX,
    HELDOUT_SEED_MIN,
    HELDOUT_SEED_MAX,
    TRAINING_SEED_MIN,
    TRAINING_SEED_MAX,
)

MESSAGE = "registered beta evaluation requires admitted study-stage custody; historical evaluation routes cannot open study seeds or registered checkpoints"


def require_legacy_evaluation_scope(
    environment_id: str,
    *,
    training_config: Mapping[str, object] | None = None,
    seeds: Sequence[int] = (),
) -> None:
    """Reject before policy admission, environment construction or any reset."""
    declared = None if training_config is None else training_config.get("beta_rl")
    if (
        isinstance(declared, Mapping)
        and declared.get("run_kind") == BETA_REGISTERED_RUN_KIND
    ):
        raise ValueError(MESSAGE)
    beta = environment_id == beta_spaces.ENVIRONMENT_ID or (
        training_config is not None
        and training_config.get("environment_id") == beta_spaces.ENVIRONMENT_ID
    )
    if not beta:
        return
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("beta evaluation seed must be a non-negative integer")
        if any(
            lower <= seed <= upper
            for lower, upper in (
                (TRAINING_SEED_MIN, TRAINING_SEED_MAX),
                (DEVELOPMENT_SEED_MIN, DEVELOPMENT_SEED_MAX),
                (HELDOUT_SEED_MIN, HELDOUT_SEED_MAX),
            )
        ):
            raise ValueError(MESSAGE)

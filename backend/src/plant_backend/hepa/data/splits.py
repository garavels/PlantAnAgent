"""Train / validation / test, assigned per episode and reproducibly.

Windows overlap: two samples five minutes apart share almost all of their context,
so splitting *windows* would put near-duplicates of training data into the test
set and report a score that means nothing. Episodes are the unit that does not
overlap, so they are the unit that is split.

    assign(("episode_0000", "episode_0001", ...), SplitFractions(), seed=0)
    # {'episode_0000': 'train', 'episode_0001': 'test', ...}

Same ids and same seed give the same assignment, so a dataset can be extended and
re-split without silently reshuffling what was already published.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class SplitFractions:
    """How the episodes are divided. Must sum to 1."""

    train: float = 0.7
    val: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split fractions must sum to 1, got {total:g}")
        if min(self.train, self.val, self.test) < 0.0:
            raise ValueError("split fractions must not be negative")

    def to_dict(self) -> dict[str, float]:
        return {"train": self.train, "val": self.val, "test": self.test}

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "SplitFractions":
        if not raw:
            return cls()
        return cls(
            train=float(raw.get("train", cls.train)),
            val=float(raw.get("val", cls.val)),
            test=float(raw.get("test", cls.test)),
        )


def assign(
    episode_ids: Sequence[str],
    fractions: SplitFractions | None = None,
    *,
    seed: int = 0,
) -> dict[str, str]:
    """Assign every episode id to exactly one split.

    With very few episodes the fractions cannot all be honoured; validation and
    test each get at least one episode as soon as there are three, because an
    empty validation split turns early stopping into a silent no-op.
    """
    shares = fractions or SplitFractions()
    ids = list(episode_ids)
    if not ids:
        return {}
    order = np.random.default_rng(seed).permutation(len(ids))
    shuffled = [ids[index] for index in order]

    total = len(shuffled)
    val_count = int(round(shares.val * total))
    test_count = int(round(shares.test * total))
    if total >= 3:
        val_count = max(1, val_count)
        test_count = max(1, test_count)
    train_count = max(total - val_count - test_count, 0 if total < 3 else 1)
    # Trim the tail splits if rounding overcommitted.
    while train_count + val_count + test_count > total:
        if test_count > (1 if total >= 3 else 0):
            test_count -= 1
        elif val_count > (1 if total >= 3 else 0):
            val_count -= 1
        else:
            train_count -= 1

    assignment: dict[str, str] = {}
    for index, episode_id in enumerate(shuffled):
        if index < train_count:
            assignment[episode_id] = "train"
        elif index < train_count + val_count:
            assignment[episode_id] = "val"
        else:
            assignment[episode_id] = "test"
    return {episode_id: assignment[episode_id] for episode_id in ids}


__all__ = ["SPLIT_NAMES", "SplitFractions", "assign"]

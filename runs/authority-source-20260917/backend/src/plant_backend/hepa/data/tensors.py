"""`torch.utils.data` on top of the NumPy windowing. The only torch in `data/`.

    index = WindowIndex(dataset.load_split("train"), spec, normalisation=stats)
    loader = make_loader(WindowDataset(index), batch_size=64, shuffle=True, seed=0)
    for batch in loader:
        batch["context"]  # (B, C, S)
        batch["future"]   # (B, K, S)  -- targets only
        batch["labels"]   # (B, E, H)
        batch["valid"]    # (B, H)

Batches are dictionaries rather than tuples so a training loop cannot mix up
`future` and `context` by position -- the one mistake in this pipeline that would
still train, still converge, and be worthless.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from .windows import WindowIndex


class WindowDataset(Dataset[dict[str, torch.Tensor]]):
    """A `WindowIndex` as a Dataset. Copies one window per item, no more."""

    def __init__(self, index: WindowIndex) -> None:
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        window = self.index.window(position)
        return {
            "context": torch.from_numpy(window.context.copy()),
            "future": torch.from_numpy(window.future.copy()),
            "labels": torch.from_numpy(window.labels.copy()),
            "valid": torch.from_numpy(window.valid.copy()),
        }


def make_loader(
    dataset: Dataset[dict[str, torch.Tensor]],
    *,
    batch_size: int = 64,
    shuffle: bool = False,
    seed: int | None = None,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader[dict[str, torch.Tensor]]:
    """A loader whose shuffle is reproducible: same seed, same order."""
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(0 if seed is None else int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        generator=generator,
    )


__all__ = ["WindowDataset", "make_loader"]

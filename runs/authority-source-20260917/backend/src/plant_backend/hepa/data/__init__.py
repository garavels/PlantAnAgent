"""HEPA data: collection, dataset format, horizon labels, windowing, splits.

    collect      run the simulator and write trajectories  (`python -m ...data.collect`)
    trajectories the on-disk dataset format and its manifest
    labels       "did the event happen in the next Δt", without leaking the future
    windows      trajectories -> fixed-shape training samples
    splits       train / val / test, per episode
    torch        `torch.utils.data` wrappers (imported only where torch is wanted)

Everything except `torch` is NumPy-only, so labels, windows and splits can be
tested without installing a deep-learning stack.
"""

from .labels import HorizonLabels, horizon_labels, observable_range
from .splits import SplitFractions
from .trajectories import (
    DatasetError,
    DatasetManifest,
    Trajectory,
    TrajectoryDataset,
    TrajectoryMetadata,
    write_dataset,
)
from .windows import Window, WindowIndex, WindowSpec

__all__ = [
    "DatasetError",
    "DatasetManifest",
    "HorizonLabels",
    "SplitFractions",
    "Trajectory",
    "TrajectoryDataset",
    "TrajectoryMetadata",
    "Window",
    "WindowIndex",
    "WindowSpec",
    "horizon_labels",
    "observable_range",
    "write_dataset",
]

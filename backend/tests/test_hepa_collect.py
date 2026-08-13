"""V9: the collector, on trajectories small enough to run in a test.

A handful of episodes of a few simulated hours each. What is under test is the
plumbing between the simulator and the dataset -- resampling, event accumulation,
split assignment, training-only statistics -- not the plant and not the model. No
pretraining or fine-tuning happens here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plant_backend.hepa import events, telemetry
from plant_backend.hepa.data import splits as split_rules
from plant_backend.hepa.data.collect import CollectionConfig, build_controller, collect
from plant_backend.hepa.data.trajectories import TrajectoryDataset
from plant_backend.hepa.horizons import HorizonGrid
from plant_backend.rl.episodes import EpisodeConfig

#: One hour and two hours ahead at the default sample period: 12 and 24 samples.
TINY_HORIZONS = HorizonGrid.of_hours((1.0, 2.0))

TINY = CollectionConfig(
    episodes=3,
    seed=7,
    episode=EpisodeConfig(horizon_h=6.0, scenarios=("clear_day", "cloud_cover")),
    horizons=TINY_HORIZONS,
    split=split_rules.SplitFractions(train=1 / 3, val=1 / 3, test=1 / 3),
)


@pytest.fixture(scope="module")
def collected(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("collected")
    collect(TINY, root)
    return root


def test_an_unknown_controller_is_refused() -> None:
    assert build_controller("heuristic") is not None
    with pytest.raises(ValueError):
        build_controller("does_not_exist")


def test_collection_writes_a_loadable_dataset(collected: Path) -> None:
    dataset = TrajectoryDataset.load(collected)
    assert sum(dataset.counts().values()) == TINY.episodes
    assert dataset.manifest.horizons == TINY_HORIZONS
    assert dataset.manifest.thresholds == TINY.thresholds
    assert dataset.manifest.collection_config["episodes"] == TINY.episodes


def test_trajectories_are_resampled_and_labelled(collected: Path) -> None:
    dataset = TrajectoryDataset.load(collected)
    expected = int(TINY.episode.horizon_h * 3600.0 / TINY_HORIZONS.sample_period_s)
    for trajectory in dataset.trajectories():
        assert len(trajectory) == expected
        assert trajectory.features.shape == (expected, telemetry.FEATURE_COUNT)
        assert trajectory.events.shape == (expected, events.EVENT_COUNT)
        assert np.all(np.isfinite(trajectory.features))
        # samples are one sample period apart, and time only moves forwards
        spacing = np.diff(trajectory.time_s)
        assert np.allclose(spacing, TINY_HORIZONS.sample_period_s)
        assert trajectory.metadata.sample_period_s == TINY_HORIZONS.sample_period_s
        assert trajectory.metadata.dt_s == TINY.dt_s
        assert trajectory.metadata.split in ("train", "val", "test")


def test_collection_is_reproducible_from_the_seed(tmp_path: Path) -> None:
    again = collect(TINY, tmp_path / "again")
    first = TrajectoryDataset.load(tmp_path / "again").load_split("train")
    twice = again.load_split("train")
    for one, other in zip(first, twice):
        assert np.array_equal(one.features, other.features)
        assert np.array_equal(one.events, other.events)


def test_normalisation_statistics_come_from_the_training_split_only(collected: Path) -> None:
    dataset = TrajectoryDataset.load(collected)
    stats = dataset.manifest.normalisation
    train = np.concatenate([t.features for t in dataset.trajectories("train")])
    assert stats.names == telemetry.FEATURE_NAMES
    assert np.allclose(stats.mean, train.mean(axis=0), atol=1e-5)

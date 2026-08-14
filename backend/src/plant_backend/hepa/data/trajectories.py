"""The dataset on disk: one trajectory per episode, plus a manifest that explains it.

    datasets/demo/
        manifest.json          schema, event rules, horizons, splits, statistics
        episode_0000.npz       features (T, S) float32, events (T, E) bool, time_s (T,)
        episode_0001.npz
        ...

A trajectory is a *resampled* episode: telemetry is recorded every
`sample_period_s` (see `horizons.py`), already encoded through the feature schema,
with the per-step event occurrences stored next to it. Encoding at collection time
rather than at training time is on purpose -- the raw `PlantState` stream is far
larger, and re-deriving events later is exactly how two modules end up disagreeing
about what "starvation" meant.

The manifest is the compatibility contract: which feature schema version, which
event rules and thresholds, which sample period, which episode belongs to which
split, and the training-split normalisation statistics. A model trained on a
dataset copies these into its checkpoint, so a mismatch surfaces as a refusal
rather than as quietly wrong probabilities.

Splits are assigned per *episode*, never per window: two windows from the same
episode overlap in time, so splitting windows would put near-copies of a training
sample into the test set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .. import events as event_rules
from .. import telemetry as telemetry_schema
from ..horizons import HorizonGrid

MANIFEST_FILE = "manifest.json"
EPISODE_GLOB = "episode_*.npz"

#: Splits, in the order the manifest lists them.
SPLITS: tuple[str, ...] = ("train", "val", "test")


class DatasetError(ValueError):
    """A dataset could not be used as it is. The message says which part."""


@dataclass(frozen=True)
class TrajectoryMetadata:
    """Where one trajectory came from, in enough detail to reproduce it."""

    episode_id: str
    seed: int
    scenario: str
    sample_period_s: float
    dt_s: float
    controller: str = "heuristic"
    hours: float = 0.0
    faults: tuple[str, ...] = ()
    split: str = "train"

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "scenario": self.scenario,
            "sample_period_s": self.sample_period_s,
            "dt_s": self.dt_s,
            "controller": self.controller,
            "hours": self.hours,
            "faults": list(self.faults),
            "split": self.split,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrajectoryMetadata":
        return cls(
            episode_id=str(raw["episode_id"]),
            seed=int(raw.get("seed", 0)),
            scenario=str(raw.get("scenario", "unknown")),
            sample_period_s=float(raw.get("sample_period_s", 0.0)),
            dt_s=float(raw.get("dt_s", 0.0)),
            controller=str(raw.get("controller", "heuristic")),
            hours=float(raw.get("hours", 0.0)),
            faults=tuple(str(name) for name in raw.get("faults", ())),
            split=str(raw.get("split", "train")),
        )


@dataclass(frozen=True)
class Trajectory:
    """One episode as arrays: what was observed, and what happened, per sample.

    Shapes, and they are checked:
        features  (T, S) float32   scaled telemetry, S = len(FEATURE_NAMES)
        events    (T, E) bool      per-sample event occurrence, E = len(EVENT_NAMES)
        time_s    (T,)   float64   simulated time of each sample
    """

    features: np.ndarray
    events: np.ndarray
    time_s: np.ndarray
    metadata: TrajectoryMetadata

    def __post_init__(self) -> None:
        if self.features.ndim != 2 or self.events.ndim != 2 or self.time_s.ndim != 1:
            raise DatasetError("a trajectory is (T, S) features, (T, E) events, (T,) time")
        if not (len(self.features) == len(self.events) == len(self.time_s)):
            raise DatasetError(
                "features, events and time_s must have the same number of samples: "
                f"{len(self.features)}, {len(self.events)}, {len(self.time_s)}"
            )

    def __len__(self) -> int:
        return int(self.features.shape[0])

    @property
    def feature_count(self) -> int:
        return int(self.features.shape[1])

    @property
    def event_count(self) -> int:
        return int(self.events.shape[1])

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            features=self.features.astype(np.float32),
            events=self.events.astype(np.bool_),
            time_s=self.time_s.astype(np.float64),
            metadata=np.asarray(json.dumps(self.metadata.to_dict())),
        )
        return target

    @classmethod
    def load(cls, path: Path | str) -> "Trajectory":
        target = Path(path)
        if not target.is_file():
            raise DatasetError(f"no trajectory at {target}")
        with np.load(target, allow_pickle=False) as payload:
            metadata = TrajectoryMetadata.from_dict(json.loads(str(payload["metadata"])))
            return cls(
                features=np.asarray(payload["features"], dtype=np.float32),
                events=np.asarray(payload["events"], dtype=bool),
                time_s=np.asarray(payload["time_s"], dtype=np.float64),
                metadata=metadata,
            )


@dataclass(frozen=True)
class DatasetManifest:
    """What this dataset is, and what a model trained on it must agree with."""

    feature_names: tuple[str, ...] = telemetry_schema.FEATURE_NAMES
    event_names: tuple[str, ...] = event_rules.EVENT_NAMES
    schema_version: int = telemetry_schema.SCHEMA_VERSION
    event_spec_version: int = event_rules.EVENT_SPEC_VERSION
    thresholds: event_rules.EventThresholds = field(default_factory=event_rules.EventThresholds)
    horizons: HorizonGrid = field(default_factory=HorizonGrid.of_hours)
    dt_s: float = 60.0
    episodes: tuple[TrajectoryMetadata, ...] = ()
    normalisation: telemetry_schema.NormalisationStats = field(
        default_factory=telemetry_schema.NormalisationStats.identity
    )
    collection_config: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_period_s(self) -> float:
        return self.horizons.sample_period_s

    def files_for(self, split: str) -> tuple[str, ...]:
        return tuple(
            f"{episode.episode_id}.npz" for episode in self.episodes if episode.split == split
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "event_names": list(self.event_names),
            "schema_version": self.schema_version,
            "event_spec_version": self.event_spec_version,
            "thresholds": self.thresholds.to_dict(),
            "horizons": self.horizons.to_dict(),
            "dt_s": self.dt_s,
            "episodes": [episode.to_dict() for episode in self.episodes],
            "normalisation": self.normalisation.to_dict(),
            "collection_config": self.collection_config,
            "counts": {split: len(self.files_for(split)) for split in SPLITS},
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DatasetManifest":
        return cls(
            feature_names=tuple(str(name) for name in raw["feature_names"]),
            event_names=tuple(str(name) for name in raw["event_names"]),
            schema_version=int(raw.get("schema_version", 0)),
            event_spec_version=int(raw.get("event_spec_version", 0)),
            thresholds=event_rules.EventThresholds.from_dict(raw.get("thresholds")),
            horizons=HorizonGrid.from_dict(raw.get("horizons")),
            dt_s=float(raw.get("dt_s", 60.0)),
            episodes=tuple(
                TrajectoryMetadata.from_dict(item) for item in raw.get("episodes", ())
            ),
            normalisation=telemetry_schema.NormalisationStats.from_dict(
                raw.get("normalisation")
            ),
            collection_config=dict(raw.get("collection_config") or {}),
        )

    def check_current(self) -> None:
        """Raise unless this dataset was built by the schema this build defines."""
        if self.schema_version != telemetry_schema.SCHEMA_VERSION:
            raise DatasetError(
                f"dataset feature schema version {self.schema_version} does not match this "
                f"build's {telemetry_schema.SCHEMA_VERSION}: recollect the dataset"
            )
        if self.event_spec_version != event_rules.EVENT_SPEC_VERSION:
            raise DatasetError(
                f"dataset event rules version {self.event_spec_version} does not match this "
                f"build's {event_rules.EVENT_SPEC_VERSION}: relabel the dataset"
            )
        if self.feature_names != telemetry_schema.FEATURE_NAMES:
            raise DatasetError("dataset feature names differ from this build's schema")
        if self.event_names != event_rules.EVENT_NAMES:
            raise DatasetError("dataset event names differ from this build's rules")


class TrajectoryDataset:
    """A dataset directory: the manifest plus lazy access to its trajectories."""

    def __init__(self, root: Path | str, manifest: DatasetManifest) -> None:
        self.root = Path(root)
        self.manifest = manifest

    @classmethod
    def load(cls, root: Path | str, *, check: bool = True) -> "TrajectoryDataset":
        directory = Path(root)
        target = directory / MANIFEST_FILE
        if not target.is_file():
            raise DatasetError(f"{directory} is not a HEPA dataset: expected {MANIFEST_FILE}")
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as reason:
            raise DatasetError(f"cannot read {target}: {reason}") from reason
        manifest = DatasetManifest.from_dict(raw)
        if check:
            manifest.check_current()
        return cls(directory, manifest)

    def save_manifest(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / MANIFEST_FILE
        target.write_text(
            json.dumps(self.manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

    def trajectories(self, split: str | None = None) -> Iterator[Trajectory]:
        """Stream the trajectories of one split (or all of them), in order."""
        for episode in self.manifest.episodes:
            if split is not None and episode.split != split:
                continue
            yield Trajectory.load(self.root / f"{episode.episode_id}.npz")

    def load_split(self, split: str) -> list[Trajectory]:
        return list(self.trajectories(split))

    def counts(self) -> dict[str, int]:
        return {name: len(self.manifest.files_for(name)) for name in SPLITS}


def write_dataset(
    root: Path | str,
    trajectories: Sequence[Trajectory],
    manifest: DatasetManifest,
) -> TrajectoryDataset:
    """Write trajectories plus manifest, keeping the two consistent by construction."""
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    for trajectory in trajectories:
        trajectory.save(directory / f"{trajectory.metadata.episode_id}.npz")
    dataset = TrajectoryDataset(
        directory,
        DatasetManifest(
            feature_names=manifest.feature_names,
            event_names=manifest.event_names,
            schema_version=manifest.schema_version,
            event_spec_version=manifest.event_spec_version,
            thresholds=manifest.thresholds,
            horizons=manifest.horizons,
            dt_s=manifest.dt_s,
            episodes=tuple(trajectory.metadata for trajectory in trajectories),
            normalisation=manifest.normalisation,
            collection_config=manifest.collection_config,
        ),
    )
    dataset.save_manifest()
    return dataset


__all__ = [
    "EPISODE_GLOB",
    "MANIFEST_FILE",
    "SPLITS",
    "DatasetError",
    "DatasetManifest",
    "Trajectory",
    "TrajectoryDataset",
    "TrajectoryMetadata",
    "write_dataset",
]

"""Collecting a dataset: run the simulator, record telemetry, write trajectories.

    python -m plant_backend.hepa.data.collect --config backend/src/plant_backend/hepa/configs/collect.json --out datasets/demo

Nothing here is HEPA-specific machinery: it is the existing plant, driven by the
existing heuristic controller through the existing controller seam, observed
through `get_telemetry()`. The collector adds three things:

*Resampling.* Telemetry is recorded once every `sample_period_s` rather than every
60 s timestep (see `horizons.py` for why). An event that starts and ends inside one
sample interval would otherwise vanish, so occurrence is accumulated across every
simulator step of the interval: a sample is labelled with an event if the event was
true at *any* step it covers. Features are the telemetry at the end of the interval,
which is what a controller reading at that moment would see.

*Reproducibility.* Episodes are drawn with `rl.episodes.sample`, reusing the "an
episode is a sampled scenario" machinery from V7 rather than inventing a second way
to randomise a starting condition. Episode `i` is drawn from `seed + i`, so any
single episode can be recollected on its own.

*Split assignment and statistics.* Splits are per episode (`splits.py`) and the
normalisation statistics in the manifest are measured on the training split only.

Scale note: at the default 72 h per episode and 300 s sampling, one episode is 864
samples. The default config is small on purpose; a real run is a matter of choosing
`episodes` and letting it run on whatever machine has the time.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from plant_sim import PlantConfig, PlantSimulator

from ...controllers import Constraints, Controller, HeuristicController
from ...rl import episodes as rl_episodes
from .. import events as event_rules
from ..horizons import DEFAULT_SAMPLE_PERIOD_S, HorizonGrid
from ..telemetry import FEATURE_NAMES, FeatureEncoder, NormalisationStats
from . import splits as split_rules
from .trajectories import DatasetManifest, Trajectory, TrajectoryDataset, TrajectoryMetadata, write_dataset

#: Every catalogued scenario: unlike RL training, event prediction wants the days
#: where something goes wrong, otherwise most labels are negative.
DEFAULT_SCENARIOS: tuple[str, ...] = (
    "clear_day",
    "cloud_cover",
    "electrolyser_degradation",
    "dac_failure",
    "sensor_drift",
    "multiple_problems",
)

#: How the collector is invoked from Python or from the CLI, with defaults that
#: run in seconds rather than hours. A real dataset overrides `episodes`.
@dataclass(frozen=True)
class CollectionConfig:
    """Everything that decides what a dataset contains."""

    episodes: int = 8
    seed: int = 0
    #: Episode distribution: horizon, scenario pool, starting inventories,
    #: disturbances. Faults come from the scenarios themselves.
    episode: rl_episodes.EpisodeConfig = rl_episodes.EpisodeConfig(
        horizon_h=72.0, scenarios=DEFAULT_SCENARIOS
    )
    horizons: HorizonGrid = HorizonGrid.of_hours()
    controller: str = "heuristic"
    #: Operating envelope the controller runs inside, as `Constraints` fields.
    constraints: dict[str, Any] = field(default_factory=dict)
    thresholds: event_rules.EventThresholds = field(default_factory=event_rules.EventThresholds)
    split: split_rules.SplitFractions = field(default_factory=split_rules.SplitFractions)
    #: Simulator timestep. Must divide the sample period evenly.
    dt_s: float = 60.0

    def plant(self) -> PlantConfig:
        return PlantConfig(dt_s=self.dt_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "seed": self.seed,
            "episode": self.episode.to_dict(),
            "horizons": self.horizons.to_dict(),
            "controller": self.controller,
            "constraints": dict(self.constraints),
            "thresholds": self.thresholds.to_dict(),
            "split": self.split.to_dict(),
            "dt_s": self.dt_s,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "CollectionConfig":
        if not raw:
            return cls()
        known = set(cls().to_dict())
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown collection settings: {', '.join(sorted(unknown))}")
        base = cls()
        return replace(
            base,
            episodes=int(raw.get("episodes", base.episodes)),
            seed=int(raw.get("seed", base.seed)),
            episode=(
                rl_episodes.EpisodeConfig.from_dict(raw["episode"])
                if "episode" in raw
                else base.episode
            ),
            horizons=HorizonGrid.from_dict(raw.get("horizons")) if "horizons" in raw else base.horizons,
            controller=str(raw.get("controller", base.controller)),
            constraints=dict(raw.get("constraints") or {}),
            thresholds=event_rules.EventThresholds.from_dict(raw.get("thresholds")),
            split=split_rules.SplitFractions.from_dict(raw.get("split")),
            dt_s=float(raw.get("dt_s", base.dt_s)),
        )

    @classmethod
    def load(cls, path: Path | str) -> "CollectionConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)


#: Controllers the collector may drive the plant with. The RL controller needs a
#: policy artefact, so it is not reachable from a config string here.
CONTROLLERS: dict[str, Callable[[], Controller]] = {"heuristic": HeuristicController}


def build_controller(name: str) -> Controller:
    try:
        factory = CONTROLLERS[name]
    except KeyError as reason:
        raise ValueError(
            f"unknown controller {name!r} (available: {', '.join(sorted(CONTROLLERS))})"
        ) from reason
    return factory()


def collect_episode(
    config: CollectionConfig,
    *,
    index: int,
    split: str = "train",
) -> Trajectory:
    """Roll out one episode and return it as a resampled, labelled trajectory."""
    plant = config.plant()
    stride = config.horizons.stride(plant.dt_s)
    seed = config.seed + index
    episode = rl_episodes.sample(
        config.episode.validate(), plant, np.random.default_rng(seed), seed=seed
    )

    simulator = PlantSimulator(plant)
    simulator.load_scenario(episode.scenario)
    controller = build_controller(config.controller)
    controller.reset()
    constraints = Constraints().merged(config.constraints)
    encoder = FeatureEncoder(plant)

    features: list[np.ndarray] = []
    occurrences: list[np.ndarray] = []
    times: list[float] = []
    pending = np.zeros(event_rules.EVENT_COUNT, dtype=bool)

    telemetry = simulator.get_telemetry()
    for step in range(episode.steps):
        decision = controller.compute_action(telemetry, constraints=constraints, config=plant)
        simulator.step(decision.action)
        telemetry = simulator.get_telemetry()
        pending |= np.fromiter(
            event_rules.occurring(telemetry, config.thresholds).values(),
            dtype=bool,
            count=event_rules.EVENT_COUNT,
        )
        if (step + 1) % stride == 0:
            features.append(encoder.encode(telemetry))
            occurrences.append(pending.copy())
            times.append(telemetry.time_s)
            pending[:] = False

    return Trajectory(
        features=(
            np.stack(features).astype(np.float32)
            if features
            else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        ),
        events=(
            np.stack(occurrences).astype(bool)
            if occurrences
            else np.zeros((0, event_rules.EVENT_COUNT), dtype=bool)
        ),
        time_s=np.asarray(times, dtype=np.float64),
        metadata=TrajectoryMetadata(
            episode_id=f"episode_{index:04d}",
            seed=seed,
            scenario=episode.scenario.id,
            sample_period_s=config.horizons.sample_period_s,
            dt_s=plant.dt_s,
            controller=config.controller,
            hours=config.episode.horizon_h,
            faults=tuple(spec.type for spec in episode.scenario.faults),
            split=split,
        ),
    )


def collect(
    config: CollectionConfig,
    root: Path | str,
    *,
    on_episode: Callable[[int, Trajectory], None] | None = None,
) -> TrajectoryDataset:
    """Collect every episode of `config` into a dataset directory."""
    if config.episodes < 1:
        raise ValueError("a dataset needs at least one episode")
    episode_ids = [f"episode_{index:04d}" for index in range(config.episodes)]
    assignment = split_rules.assign(episode_ids, config.split, seed=config.seed)

    trajectories: list[Trajectory] = []
    for index, episode_id in enumerate(episode_ids):
        trajectory = collect_episode(config, index=index, split=assignment[episode_id])
        trajectories.append(trajectory)
        if on_episode is not None:
            on_episode(index, trajectory)

    normalisation = _fit_normalisation(trajectories)
    manifest = DatasetManifest(
        thresholds=config.thresholds,
        horizons=config.horizons,
        dt_s=config.dt_s,
        normalisation=normalisation,
        collection_config=config.to_dict(),
    )
    return write_dataset(root, trajectories, manifest)


def _fit_normalisation(trajectories: Sequence[Trajectory]) -> NormalisationStats:
    """Training-split statistics only: the other splits must stay unseen."""
    blocks = [
        trajectory.features
        for trajectory in trajectories
        if trajectory.metadata.split == "train" and len(trajectory) > 0
    ]
    if not blocks:
        return NormalisationStats.identity()
    return NormalisationStats.fit(blocks, FEATURE_NAMES)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m plant_backend.hepa.data.collect",
        description="Collect a HEPA telemetry dataset from the plant simulator.",
    )
    parser.add_argument("--config", type=Path, help="JSON collection config")
    parser.add_argument("--out", type=Path, required=True, help="dataset directory to write")
    parser.add_argument("--episodes", type=int, help="override the episode count")
    parser.add_argument("--hours", type=float, help="override the simulated hours per episode")
    parser.add_argument("--seed", type=int, help="override the base seed")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be collected and write nothing",
    )
    args = parser.parse_args(argv)

    config = CollectionConfig.load(args.config) if args.config else CollectionConfig()
    if args.episodes is not None:
        config = replace(config, episodes=args.episodes)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.hours is not None:
        config = replace(config, episode=replace(config.episode, horizon_h=args.hours))

    stride = config.horizons.stride(config.plant().dt_s)
    samples = config.episode.steps(config.plant()) // stride
    print(f"episodes           {config.episodes}")
    print(f"hours per episode  {config.episode.horizon_h:g}")
    print(f"sample period      {config.horizons.sample_period_s:g} s ({stride} timesteps)")
    print(f"samples per episode{samples:>6}")
    print(f"scenarios          {', '.join(config.episode.scenarios)}")
    print(f"output             {args.out}")
    if args.dry_run:
        print("dry run: nothing collected")
        return 0

    dataset = collect(
        config,
        args.out,
        on_episode=lambda index, trajectory: print(
            f"  {trajectory.metadata.episode_id}  {trajectory.metadata.scenario:<24}"
            f"{len(trajectory):>5} samples  {trajectory.metadata.split}"
        ),
    )
    counts = dataset.counts()
    print(f"wrote {sum(counts.values())} episodes to {dataset.root}: {counts}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "CONTROLLERS",
    "DEFAULT_SAMPLE_PERIOD_S",
    "CollectionConfig",
    "build_controller",
    "collect",
    "collect_episode",
    "main",
]

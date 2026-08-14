"""V9: the parts of HEPA that decide what a number *means* -- and need no torch.

Telemetry schema, event rules, horizon conversion, future-only labels, the dataset
on disk, and the metrics. These are the contracts a checkpoint is validated against,
so they are tested without the optional model dependencies installed: a wrong
feature order or a leaking label must fail here, on every CI run, not only on a
machine with `uv sync --group hepa`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from plant_sim import Action, PlantConfig, PlantSimulator, PlantState

from plant_backend.hepa import checkpoints, events, horizons, telemetry
from plant_backend.hepa.data import labels as label_rules
from plant_backend.hepa.data.trajectories import (
    DatasetError,
    DatasetManifest,
    Trajectory,
    TrajectoryDataset,
    TrajectoryMetadata,
    write_dataset,
)
from plant_backend.hepa.data.windows import WindowIndex, WindowSpec
from plant_backend.hepa import evaluation


# --- telemetry schema -----------------------------------------------------
def test_feature_contract_is_self_consistent() -> None:
    assert telemetry.FEATURE_COUNT == len(telemetry.FEATURE_NAMES) == len(telemetry.FEATURES)
    assert len(set(telemetry.FEATURE_NAMES)) == telemetry.FEATURE_COUNT
    for feature in telemetry.FEATURES:
        assert feature.source and feature.unit and feature.formula
    described = telemetry.describe()
    assert described["schema_version"] == telemetry.SCHEMA_VERSION
    assert len(described["features"]) == telemetry.FEATURE_COUNT


def test_encoding_a_state_is_finite_and_ordered() -> None:
    simulator = PlantSimulator()
    state = simulator.reset()
    encoder = telemetry.FeatureEncoder()
    vector = encoder.encode(state)
    assert vector.shape == (telemetry.FEATURE_COUNT,)
    assert vector.dtype == np.float32
    assert np.all(np.isfinite(vector))
    # cyclic time features live on the unit circle
    index = telemetry.FEATURE_NAMES.index("time_of_day_sin")
    cosine = telemetry.FEATURE_NAMES.index("time_of_day_cos")
    assert abs(float(vector[index]) ** 2 + float(vector[cosine]) ** 2 - 1.0) < 1e-5


def test_encoding_uses_only_observed_telemetry() -> None:
    """Whatever `get_telemetry()` hides must not reach the feature vector."""
    simulator = PlantSimulator()
    simulator.reset()
    for _ in range(20):
        simulator.step(Action.all_to_electrolyser())
    encoder = telemetry.FeatureEncoder()
    observed = encoder.encode(simulator.get_telemetry())
    assert observed.shape == (telemetry.FEATURE_COUNT,)
    assert np.all(np.isfinite(observed))


def test_a_serialised_frame_round_trips_into_a_state() -> None:
    simulator = PlantSimulator()
    state = simulator.reset()
    for _ in range(5):
        state = simulator.step(Action.all_to_electrolyser())
    frame = state.to_dict()
    rebuilt = telemetry.state_from_frame(frame)
    encoder = telemetry.FeatureEncoder()
    assert np.allclose(encoder.encode(rebuilt), encoder.encode(state))


def test_normalisation_is_invertible_in_the_statistics_it_recorded() -> None:
    block = np.random.default_rng(0).normal(size=(64, telemetry.FEATURE_COUNT)).astype(np.float32)
    stats = telemetry.NormalisationStats.fit([block], telemetry.FEATURE_NAMES)
    scaled = stats.apply(block)
    assert scaled.shape == block.shape
    assert abs(float(scaled.mean())) < 1e-4
    restored = telemetry.NormalisationStats.from_dict(stats.to_dict())
    assert np.allclose(restored.apply(block), scaled)


# --- event rules ----------------------------------------------------------
def _state(**overrides: object) -> PlantState:
    return PlantState(**overrides)  # type: ignore[arg-type]


def test_battery_reserve_violation_fires_at_and_below_the_threshold() -> None:
    thresholds = events.EventThresholds(reserve_soc=0.15)
    assert events.occurring(_state(battery_soc=0.15), thresholds)["battery_reserve_violation"]
    assert events.occurring(_state(battery_soc=0.10), thresholds)["battery_reserve_violation"]
    assert not events.occurring(_state(battery_soc=0.16), thresholds)["battery_reserve_violation"]


def test_starvation_needs_demand_an_offline_unit_and_an_empty_tank() -> None:
    thresholds = events.EventThresholds(empty_kg=1.0)
    demanded = Action(methanator_share=0.5)
    starved = _state(action=demanded, methanator_online=False, h2_level_kg=0.0, co2_level_kg=5.0)
    occurring = events.occurring(starved, thresholds)
    assert occurring["h2_starvation"]
    assert not occurring["co2_starvation"]

    # the same empty tank with no demand is not an event: nothing was denied
    idle = _state(action=Action.idle(), methanator_online=False, h2_level_kg=0.0)
    assert not events.occurring(idle, thresholds)["h2_starvation"]

    # and a running methanator is not starving whatever the level says
    running = _state(action=demanded, methanator_online=True, h2_level_kg=0.0)
    assert not events.occurring(running, thresholds)["h2_starvation"]

    co2 = _state(action=demanded, methanator_online=False, h2_level_kg=9.0, co2_level_kg=0.2)
    assert events.occurring(co2, thresholds)["co2_starvation"]


def test_forced_shutdown_and_critical_state() -> None:
    thresholds = events.EventThresholds()
    denied = _state(action=Action(electrolyser_share=0.4), electrolyser_online=False)
    assert events.occurring(denied, thresholds)["forced_production_shutdown"]
    assert not events.occurring(_state(), thresholds)["forced_production_shutdown"]
    assert events.occurring(_state(plant_status="critical"), thresholds)["critical_plant_state"]
    assert events.occurring(_state(plant_status="offline"), thresholds)["critical_plant_state"]
    assert not events.occurring(_state(plant_status="degraded"), thresholds)["critical_plant_state"]


def test_occurrence_matrix_shape_and_documentation() -> None:
    states = [_state(battery_soc=0.9), _state(battery_soc=0.0)]
    matrix = events.occurrence_matrix(states)
    assert matrix.shape == (2, events.EVENT_COUNT)
    assert matrix.dtype == bool
    assert not matrix[0, 0] and matrix[1, 0]
    described = events.describe()
    assert {item["name"] for item in described["events"]} == set(events.EVENT_NAMES)
    for item in described["events"]:
        assert item["rule"] and item["summary"]


# --- horizons -------------------------------------------------------------
def test_default_horizons_convert_to_the_documented_step_counts() -> None:
    grid = horizons.HorizonGrid.of_hours()
    assert grid.labels == ("1h", "3h", "6h", "12h", "24h")
    assert grid.steps == (12, 36, 72, 144, 288)
    assert grid.max_steps == 288
    assert grid.stride(60.0) == 5
    assert grid.steps_for("6h") == 72
    assert grid.steps_for(6.0) == 72
    with pytest.raises(KeyError):
        grid.steps_for("7h")
    assert horizons.HorizonGrid.from_dict(grid.to_dict()) == grid


def test_a_sample_period_that_does_not_divide_the_timestep_is_refused() -> None:
    with pytest.raises(ValueError):
        horizons.HorizonGrid.of_hours((1.0,), sample_period_s=90.0).stride(60.0)


# --- labels ---------------------------------------------------------------
def test_labels_read_the_future_only() -> None:
    """The event at t is input, not label; the one at t+1 is the label."""
    occurrences = np.zeros((6, 1), dtype=bool)
    occurrences[2, 0] = True
    result = label_rules.horizon_labels(occurrences, [1])
    # t=1 sees the occurrence at t=2 in (1, 2]; t=2 does not see its own
    assert result.labels[1, 0, 0] == 1.0
    assert result.labels[2, 0, 0] == 0.0
    assert result.labels[0, 0, 0] == 0.0


def test_labels_are_cumulative_and_monotone_in_the_horizon() -> None:
    occurrences = np.zeros((20, 2), dtype=bool)
    occurrences[9, 0] = True
    result = label_rules.horizon_labels(occurrences, [1, 4, 8])
    row = result.labels[3, 0]
    assert list(row) == [0.0, 0.0, 1.0]
    assert np.all(np.diff(result.labels, axis=2) >= 0.0)


def test_a_truncated_horizon_is_unknown_rather_than_negative() -> None:
    occurrences = np.zeros((10, 1), dtype=bool)
    result = label_rules.horizon_labels(occurrences, [4])
    assert result.valid[5, 0]  # (5, 9] fits
    assert not result.valid[6, 0]  # (6, 10] does not
    assert not result.valid[-1, 0]


def test_observable_range_leaves_room_for_context_and_future() -> None:
    positions = label_rules.observable_range(total=100, context_len=10, max_steps=20)
    assert positions[0] == 9
    assert positions[-1] == 79
    assert len(label_rules.observable_range(total=20, context_len=10, max_steps=20)) == 0


# --- dataset and windows --------------------------------------------------
def _trajectory(episode_id: str, samples: int, *, split: str = "train", seed: int = 0) -> Trajectory:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(samples, telemetry.FEATURE_COUNT)).astype(np.float32)
    occurrences = np.zeros((samples, events.EVENT_COUNT), dtype=bool)
    occurrences[samples // 2 :, 0] = True
    return Trajectory(
        features=features,
        events=occurrences,
        time_s=np.arange(samples, dtype=np.float64) * 300.0,
        metadata=TrajectoryMetadata(
            episode_id=episode_id,
            seed=seed,
            scenario="clear_day",
            sample_period_s=300.0,
            dt_s=60.0,
            hours=samples * 300.0 / 3600.0,
            split=split,
        ),
    )


TINY = horizons.HorizonGrid.of_hours((1.0, 2.0), sample_period_s=300.0)  # 12, 24 steps


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    write_dataset(
        root,
        [
            _trajectory("episode_0000", 80, split="train", seed=1),
            _trajectory("episode_0001", 80, split="val", seed=2),
            _trajectory("episode_0002", 80, split="test", seed=3),
        ],
        DatasetManifest(horizons=TINY),
    )
    return root


def test_a_dataset_round_trips_through_disk(dataset_root: Path) -> None:
    dataset = TrajectoryDataset.load(dataset_root)
    assert dataset.counts() == {"train": 1, "val": 1, "test": 1}
    assert dataset.manifest.horizons == TINY
    assert dataset.manifest.feature_names == telemetry.FEATURE_NAMES
    loaded = dataset.load_split("train")[0]
    assert len(loaded) == 80
    assert loaded.feature_count == telemetry.FEATURE_COUNT
    assert loaded.event_count == events.EVENT_COUNT


def test_a_dataset_from_another_schema_is_refused(dataset_root: Path) -> None:
    manifest = dataset_root / "manifest.json"
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["schema_version"] = telemetry.SCHEMA_VERSION + 1
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DatasetError):
        TrajectoryDataset.load(dataset_root)


def test_windows_split_context_from_future_without_overlap(dataset_root: Path) -> None:
    dataset = TrajectoryDataset.load(dataset_root)
    spec = WindowSpec(context_len=16, horizons=TINY)
    index = WindowIndex(dataset.load_split("train"), spec)
    assert len(index) == 80 - 16 - 24 + 1
    window = index.window(0)
    assert window.context.shape == (16, telemetry.FEATURE_COUNT)
    assert window.future.shape == (24, telemetry.FEATURE_COUNT)
    assert window.labels.shape == (events.EVENT_COUNT, 2)
    assert window.valid.shape == (2,)
    assert bool(window.valid.all())

    trajectory = dataset.load_split("train")[0]
    at = window.observed_at
    assert np.array_equal(window.context[-1], trajectory.features[at])
    assert np.array_equal(window.future[0], trajectory.features[at + 1])
    # no sample appears in both halves
    assert not np.any(np.all(window.future[:, None, :] == window.context[None, :, :], axis=-1))


def test_normalisation_is_fitted_on_context_samples_only(dataset_root: Path) -> None:
    dataset = TrajectoryDataset.load(dataset_root)
    spec = WindowSpec(context_len=16, horizons=TINY)
    index = WindowIndex(dataset.load_split("train"), spec)
    stats = index.fit_normalisation()
    normalised = WindowIndex(dataset.load_split("train"), spec, normalisation=stats)
    assert normalised.window(0).context.shape == (16, telemetry.FEATURE_COUNT)
    rates = index.positive_rate()
    assert rates.shape == (events.EVENT_COUNT, 2)
    assert np.all((rates >= 0.0) & (rates <= 1.0))


# --- metrics --------------------------------------------------------------
def test_auroc_and_auprc_on_a_known_ranking() -> None:
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert evaluation.auroc(perfect, labels) == pytest.approx(1.0)
    assert evaluation.auprc(perfect, labels) == pytest.approx(1.0)
    inverted = np.array([0.9, 0.8, 0.2, 0.1])
    assert evaluation.auroc(inverted, labels) == pytest.approx(0.0)
    tied = np.full(4, 0.5)
    assert evaluation.auroc(tied, labels) == pytest.approx(0.5)


def test_brier_and_threshold_metrics_on_a_known_confusion_matrix() -> None:
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    scores = np.array([0.9, 0.4, 0.6, 0.1])
    assert evaluation.brier(scores, labels) == pytest.approx(
        (0.01 + 0.36 + 0.36 + 0.01) / 4.0
    )
    metrics = evaluation.threshold_metrics(scores, labels, threshold=0.5)
    assert (metrics.true_positives, metrics.false_positives) == (1, 1)
    assert (metrics.false_negatives, metrics.true_negatives) == (1, 1)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)


def test_a_report_covers_every_event_and_horizon_and_ignores_masked_windows() -> None:
    rng = np.random.default_rng(0)
    predictions = rng.uniform(size=(32, 2, 2)).astype(np.float32)
    labels = (rng.uniform(size=(32, 2, 2)) > 0.5).astype(np.float32)
    valid = np.ones((32, 2), dtype=bool)
    valid[-8:, 1] = False
    report = evaluation.evaluate(
        predictions,
        labels,
        valid,
        event_names=("a", "b"),
        horizon_labels=("1h", "2h"),
    )
    assert set(report["events"]) == {"a", "b"}
    assert set(report["events"]["a"]) == {"1h", "2h"}
    assert report["events"]["a"]["1h"]["count"] == 32
    assert report["events"]["a"]["2h"]["count"] == 24
    for horizon in report["events"]["a"].values():
        assert 0.0 <= horizon["auroc"] <= 1.0
        assert 0.0 <= horizon["brier"] <= 1.0
        assert horizon["calibration"]["bins"]
    assert evaluation.summarise(report)


# --- checkpoint metadata, without torch ------------------------------------
def test_checkpoint_metadata_round_trips_and_refuses_a_foreign_schema(tmp_path: Path) -> None:
    metadata = checkpoints.CheckpointMetadata(
        stage=checkpoints.STAGE_FINETUNE, horizons=TINY, context_len=16
    )
    checkpoints.save_metadata(tmp_path, metadata)
    loaded = checkpoints.load_metadata(tmp_path)
    assert loaded.stage == checkpoints.STAGE_FINETUNE
    assert loaded.horizons == TINY
    assert loaded.context_len == 16
    assert not loaded.trained
    assert loaded.created_at

    raw = json.loads((tmp_path / checkpoints.METADATA_FILE).read_text(encoding="utf-8"))
    raw["feature_names"] = list(telemetry.FEATURE_NAMES)[:-1]
    (tmp_path / checkpoints.METADATA_FILE).write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(checkpoints.IncompatibleCheckpointError):
        checkpoints.load_metadata(tmp_path)


def test_a_missing_or_unreadable_checkpoint_names_the_problem(tmp_path: Path) -> None:
    with pytest.raises(checkpoints.MissingCheckpointError):
        checkpoints.load_metadata(tmp_path / "absent")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(checkpoints.CorruptCheckpointError):
        checkpoints.load_metadata(empty)


def test_the_plant_config_the_encoder_scales_against_is_explicit() -> None:
    """A re-sized plant changes the meaning of the features, so it is a parameter."""
    encoder = telemetry.FeatureEncoder(PlantConfig())
    state = PlantSimulator().reset()
    assert encoder.encode(state).shape == (telemetry.FEATURE_COUNT,)

"""V9: the two stages, the checkpoint between them, and the surfaces that read it.

A whole HEPA pipeline runs here -- stage 1, a checkpoint, stage 2, another
checkpoint, inference, evaluation, the HTTP routes -- on a synthetic dataset of a
few hundred samples with a model of a few thousand parameters. It finishes in
seconds on a CPU and it proves nothing about accuracy: the point is that the
stages fit together, that the contracts they record are enforced, and that a
prediction cannot come out of a checkpoint this build cannot trust. Real
pretraining and fine-tuning are hardware-dependent work and are not run here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest

from plant_backend.hepa import checkpoints, events, telemetry
from plant_backend.hepa.data.trajectories import (
    DatasetManifest,
    Trajectory,
    TrajectoryMetadata,
    write_dataset,
)
from plant_backend.hepa.horizons import HorizonGrid

torch = pytest.importorskip("torch", reason="HEPA training needs `uv sync --group hepa`")

from plant_backend.hepa.data.windows import WindowIndex, WindowSpec  # noqa: E402
from plant_backend.hepa.finetuning.train import (  # noqa: E402
    FinetuneConfig,
    finetune,
    load_pretrained,
)
from plant_backend.hepa.finetuning.train import plan as finetune_plan  # noqa: E402
from plant_backend.hepa.inference import EventPredictor, InferenceError, status  # noqa: E402
from plant_backend.hepa.pretraining.train import PretrainConfig, pretrain  # noqa: E402
from plant_backend.hepa.pretraining.train import plan as pretrain_plan  # noqa: E402

#: One horizon, one hour ahead: 12 samples at the default sample period.
HORIZONS = HorizonGrid.of_hours((1.0,))
CONTEXT_LEN = 16
#: A model small enough that a handful of steps is a few seconds of CPU.
TINY_MODEL: dict[str, Any] = {
    "d_model": 32,
    "num_layers": 1,
    "num_heads": 2,
    "patch_size": 4,
    "dropout": 0.0,
}

PRETRAIN = PretrainConfig(
    epochs=2,
    batch_size=8,
    context_len=CONTEXT_LEN,
    max_steps_per_epoch=3,
    projections=8,
    model=TINY_MODEL,
)
FINETUNE = FinetuneConfig(
    epochs=2,
    batch_size=8,
    context_len=CONTEXT_LEN,
    max_steps_per_epoch=3,
    model=TINY_MODEL,
)


def _trajectory(index: int, split: str, *, samples: int = 96) -> Trajectory:
    """A synthetic episode: smooth telemetry with events on a fixed rhythm."""
    generator = np.random.default_rng(index)
    time = np.arange(samples, dtype=np.float64)
    ramp = np.sin(time / 8.0)[:, None] * np.linspace(0.2, 1.0, telemetry.FEATURE_COUNT)[None, :]
    features = (ramp + 0.05 * generator.normal(size=(samples, telemetry.FEATURE_COUNT))).astype(
        np.float32
    )
    occurrence = np.zeros((samples, events.EVENT_COUNT), dtype=bool)
    occurrence[(index + 3) :: 7, 0] = True  # frequent: the head has positives to learn from
    occurrence[(index + 5) :: 31, 1] = True  # rare
    return Trajectory(
        features=features,
        events=occurrence,
        time_s=time * HORIZONS.sample_period_s,
        metadata=TrajectoryMetadata(
            episode_id=f"episode_{index:04d}",
            seed=index,
            scenario="synthetic",
            sample_period_s=HORIZONS.sample_period_s,
            dt_s=60.0,
            hours=samples * HORIZONS.sample_period_s / 3600.0,
            split=split,
        ),
    )


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("synthetic-dataset")
    splits = ["train", "train", "train", "train", "val", "test"]
    write_dataset(
        root,
        [_trajectory(index, split) for index, split in enumerate(splits)],
        DatasetManifest(horizons=HORIZONS),
    )
    return root


@pytest.fixture(scope="module")
def stage1(dataset: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("stage1")
    pretrain(PRETRAIN, dataset, out)
    return out


@pytest.fixture(scope="module")
def stage2(dataset: Path, stage1: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("stage2")
    finetune(FINETUNE, dataset, stage1, out)
    return out


# --- windows --------------------------------------------------------------
def test_windows_leave_room_for_the_longest_horizon(dataset: Path) -> None:
    from plant_backend.hepa.data.trajectories import TrajectoryDataset

    loaded = TrajectoryDataset.load(dataset)
    spec = WindowSpec(context_len=CONTEXT_LEN, horizons=HORIZONS)
    index = WindowIndex(loaded.load_split("train"), spec, normalisation=loaded.manifest.normalisation)
    assert len(index) > 0
    window = index.window(0)
    assert window.context.shape == (CONTEXT_LEN, telemetry.FEATURE_COUNT)
    assert window.labels.shape == (events.EVENT_COUNT, len(HORIZONS.labels))
    assert window.valid.shape == (len(HORIZONS.labels),)
    assert index.positive_rate().shape == (events.EVENT_COUNT, len(HORIZONS.labels))


# --- stage 1 --------------------------------------------------------------
def test_the_pretraining_plan_reports_what_a_real_run_would_do(dataset: Path) -> None:
    reported = pretrain_plan(PRETRAIN, dataset)
    assert reported["windows"]["train"] > 0
    assert reported["config"]["context_len"] == CONTEXT_LEN
    assert reported["context_len"] == CONTEXT_LEN
    assert reported["max_horizon_steps"] == HORIZONS.max_steps
    assert reported["batches_per_epoch"] >= 1


def test_stage_one_writes_a_loadable_pretrained_checkpoint(stage1: Path) -> None:
    metadata = checkpoints.load_metadata(stage1)
    assert metadata.stage == checkpoints.STAGE_PRETRAIN
    assert metadata.feature_names == telemetry.FEATURE_NAMES
    assert metadata.event_names == events.EVENT_NAMES
    assert metadata.context_len == CONTEXT_LEN
    assert metadata.horizons == HORIZONS
    assert metadata.trained
    model, _ = checkpoints.load_model(stage1)
    assert model.config.d_model == TINY_MODEL["d_model"]


def test_pretraining_is_deterministic_given_the_seed(dataset: Path) -> None:
    first = pretrain(PRETRAIN, dataset)
    second = pretrain(PRETRAIN, dataset)
    assert first.history == second.history
    for one, other in zip(first.model.state_dict().values(), second.model.state_dict().values()):
        assert torch.equal(one, other)


def test_pretraining_refuses_a_dataset_with_no_usable_windows(tmp_path: Path) -> None:
    root = tmp_path / "too-short"
    write_dataset(
        root,
        [_trajectory(0, "train", samples=CONTEXT_LEN)],
        DatasetManifest(horizons=HORIZONS),
    )
    with pytest.raises(ValueError, match="no training windows"):
        pretrain(PRETRAIN, root)


# --- stage 2 --------------------------------------------------------------
def test_stage_two_starts_from_the_pretrained_encoder(dataset: Path, stage1: Path) -> None:
    spec = WindowSpec(context_len=CONTEXT_LEN, horizons=HORIZONS)
    loaded, provenance = load_pretrained(stage1, FINETUNE, spec)
    pretrained, _ = checkpoints.load_model(stage1)
    assert provenance == str(stage1)
    for one, other in zip(
        loaded.encoder.state_dict().values(), pretrained.encoder.state_dict().values()
    ):
        assert torch.equal(one, other)

    scratch, no_provenance = load_pretrained(None, FINETUNE, spec)
    assert no_provenance == ""
    assert scratch.config.context_len == CONTEXT_LEN


def test_stage_two_refuses_a_checkpoint_from_the_wrong_stage(dataset: Path, stage2: Path) -> None:
    spec = WindowSpec(context_len=CONTEXT_LEN, horizons=HORIZONS)
    with pytest.raises(checkpoints.HepaCheckpointError):
        load_pretrained(stage2, FINETUNE, spec)


def test_stage_two_refuses_a_context_it_was_not_pretrained_for(stage1: Path) -> None:
    spec = WindowSpec(context_len=CONTEXT_LEN * 2, horizons=HORIZONS)
    with pytest.raises(checkpoints.HepaCheckpointError):
        load_pretrained(stage1, FinetuneConfig(context_len=CONTEXT_LEN * 2), spec)


def test_stage_two_freezes_the_encoder_it_was_given(dataset: Path, stage1: Path) -> None:
    before, _ = checkpoints.load_model(stage1)
    run = finetune(FINETUNE, dataset, stage1)
    assert run.model.encoder_frozen
    for one, other in zip(
        before.encoder.state_dict().values(), run.model.encoder.state_dict().values()
    ):
        assert torch.equal(one, other)
    assert run.history
    assert run.validation_report["events"]


def test_stage_two_writes_a_checkpoint_that_records_its_provenance(
    stage1: Path, stage2: Path
) -> None:
    metadata = checkpoints.load_metadata(stage2)
    assert metadata.stage == checkpoints.STAGE_FINETUNE
    assert metadata.pretrained_from == str(stage1)
    assert metadata.horizons == HORIZONS
    assert metadata.thresholds == events.EventThresholds()
    assert metadata.trained


def test_the_fine_tuning_plan_reports_the_windows_and_positives(
    dataset: Path, stage1: Path
) -> None:
    reported = finetune_plan(FINETUNE, dataset)
    assert reported["windows"]["train"] > 0
    assert reported["config"]["freeze_encoder"] is True
    assert reported["horizons"]["labels"] == list(HORIZONS.labels)


# --- checkpoint contracts -------------------------------------------------
def test_a_checkpoint_recorded_against_another_schema_is_refused(stage2: Path, tmp_path: Path) -> None:
    raw = json.loads(checkpoints.metadata_path(stage2).read_text(encoding="utf-8"))
    raw["schema_version"] = raw["schema_version"] + 1
    other = tmp_path / "wrong-schema"
    other.mkdir()
    checkpoints.metadata_path(other).write_text(json.dumps(raw), encoding="utf-8")
    (other / checkpoints.WEIGHTS_FILE).write_bytes(
        checkpoints.weights_path(stage2).read_bytes()
    )
    with pytest.raises(checkpoints.IncompatibleCheckpointError):
        checkpoints.load_model(other)
    # ... and the same checkpoint loads when the caller explicitly skips the check
    assert checkpoints.load_metadata(other, check=False).schema_version != telemetry.SCHEMA_VERSION


def test_weights_that_do_not_fit_the_recorded_shape_are_refused(stage2: Path, tmp_path: Path) -> None:
    other = tmp_path / "wrong-weights"
    other.mkdir()
    raw = json.loads(checkpoints.metadata_path(stage2).read_text(encoding="utf-8"))
    raw["model"]["d_model"] = int(raw["model"]["d_model"]) * 2
    checkpoints.metadata_path(other).write_text(json.dumps(raw), encoding="utf-8")
    (other / checkpoints.WEIGHTS_FILE).write_bytes(
        checkpoints.weights_path(stage2).read_bytes()
    )
    with pytest.raises(checkpoints.IncompatibleCheckpointError):
        checkpoints.load_model(other)


# --- inference ------------------------------------------------------------
def test_a_prediction_names_every_event_and_horizon(stage2: Path) -> None:
    predictor = EventPredictor.load(stage2)
    history = _trajectory(99, "test").features[:CONTEXT_LEN]
    probabilities = predictor.predict_features(history)
    assert probabilities.shape == (events.EVENT_COUNT, len(HORIZONS.labels))
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert predictor.event_names == events.EVENT_NAMES
    assert predictor.horizon_labels == HORIZONS.labels
    assert predictor.describe()["trained"] is True


def test_inference_refuses_a_short_history_and_an_untrained_horizon(stage2: Path) -> None:
    predictor = EventPredictor.load(stage2)
    features = _trajectory(1, "test").features
    with pytest.raises(InferenceError, match="need 16 telemetry samples"):
        predictor.predict_features(features[: CONTEXT_LEN - 1])
    with pytest.raises(InferenceError, match="not fine-tuned for"):
        predictor.predict_features(features[:CONTEXT_LEN], ["24h"])
    with pytest.raises(InferenceError):
        predictor.predict_features(features[:CONTEXT_LEN], [])


def test_inference_refuses_a_stage_one_checkpoint(stage1: Path) -> None:
    with pytest.raises(InferenceError, match="only a stage2 checkpoint"):
        EventPredictor.load(stage1)


def test_status_answers_for_a_missing_and_for_a_present_checkpoint(stage2: Path) -> None:
    assert status(None)["available"] is False
    missing = status("/definitely/not/a/checkpoint")
    assert missing["available"] is False and missing["reason"]
    present = status(stage2)
    assert present["available"] is True
    assert present["events"] == list(events.EVENT_NAMES)
    assert present["horizons"] == list(HORIZONS.labels)


def test_evaluation_scores_every_event_and_horizon(dataset: Path, stage2: Path) -> None:
    predictor = EventPredictor.load(stage2)
    report = predictor.evaluate_split(dataset, split="test", batch_size=8)
    assert report["split"] == "test"
    assert report["windows"] > 0
    for name in events.EVENT_NAMES:
        per_horizon = report["events"][name]
        assert set(per_horizon) == set(HORIZONS.labels)
        for metrics in per_horizon.values():
            assert 0.0 <= metrics["brier"] <= 1.0
            # an event with no positives in the split has no ranking to score: nan,
            # never a made-up 0.5
            assert np.isnan(metrics["auroc"]) or 0.0 <= metrics["auroc"] <= 1.0
            for metric in ("precision", "recall", "f1"):
                assert np.isnan(metrics[metric]) or 0.0 <= metrics[metric] <= 1.0
            # windows whose horizon runs past the episode are excluded, not counted
            # as negatives
            assert 0 < metrics["count"] <= report["windows"]


# --- HTTP routes ----------------------------------------------------------
@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    from plant_backend import app as backend_app
    from plant_backend.hepa import api as hepa_api

    monkeypatch.delenv(checkpoints.CHECKPOINT_ENV, raising=False)
    hepa_api.reset_cache()
    with TestClient(backend_app.app) as test_client:
        yield test_client
    hepa_api.reset_cache()


def test_status_reports_the_contract_with_no_checkpoint_configured(client: Any) -> None:
    body = client.get("/api/hepa/status").json()
    assert body["checkpoint"]["available"] is False
    assert len(body["schema"]["telemetry"]["features"]) == telemetry.FEATURE_COUNT
    assert len(body["schema"]["events"]["events"]) == events.EVENT_COUNT


def test_predict_without_a_checkpoint_is_unavailable_not_a_guess(client: Any) -> None:
    response = client.post("/api/hepa/predict", json={})
    assert response.status_code == 503
    assert checkpoints.CHECKPOINT_ENV in response.json()["detail"]


def test_predict_answers_from_a_posted_history(
    client: Any, stage2: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plant_sim.simulator import PlantSimulator

    monkeypatch.setenv(checkpoints.CHECKPOINT_ENV, str(stage2))
    simulator = PlantSimulator()
    frames = [simulator.get_telemetry().to_dict()]
    while len(frames) < CONTEXT_LEN:
        frames.append(simulator.step().to_dict())

    response = client.post("/api/hepa/predict", json={"telemetry": frames})
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body["probabilities"]) == set(events.EVENT_NAMES)
    assert body["horizons"] == list(HORIZONS.labels)
    for per_horizon in body["probabilities"].values():
        for probability in per_horizon.values():
            assert 0.0 <= probability <= 1.0
    assert body["checkpoint"]["stage"] == checkpoints.STAGE_FINETUNE


def test_predict_rejects_a_short_history_and_an_unknown_horizon(
    client: Any, stage2: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plant_sim.simulator import PlantSimulator

    monkeypatch.setenv(checkpoints.CHECKPOINT_ENV, str(stage2))
    frame = PlantSimulator().get_telemetry().to_dict()
    short = client.post("/api/hepa/predict", json={"telemetry": [frame]})
    assert short.status_code == 400
    unknown = client.post(
        "/api/hepa/predict", json={"telemetry": [frame] * CONTEXT_LEN, "horizons": ["24h"]}
    )
    assert unknown.status_code == 400
    broken = client.post("/api/hepa/predict", json={"telemetry": [{"nonsense": 1}]})
    assert broken.status_code == 400


def test_predict_with_a_checkpoint_that_is_not_there_is_unavailable(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(checkpoints.CHECKPOINT_ENV, "/definitely/not/a/checkpoint")
    response = client.post("/api/hepa/predict", json={})
    assert response.status_code == 503

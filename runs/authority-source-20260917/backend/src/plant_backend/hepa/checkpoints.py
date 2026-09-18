"""Checkpoints: weights plus everything needed to refuse them.

    checkpoints/hepa/stage1/
        metadata.json      schema, events, horizons, statistics, provenance
        weights.pt         the `state_dict`

A HEPA checkpoint is only meaningful together with the *meaning* of its inputs and
outputs, so the metadata records the feature schema (names, in order, and its
version), the event rules (names, thresholds, and their version), the horizon grid,
the context length, the normalisation statistics and the model shape. Loading checks
all of it and fails with a message naming the mismatch, because a model served with a
shifted feature vector or a redefined event does not crash -- it produces confident,
wrong probabilities.

Reading metadata deliberately does not import torch: `GET /api/hepa/status` must be
able to say "a stage-2 checkpoint exists, trained for these horizons" on a machine
where the training extra was never installed. Only `load_model` needs torch.

Stage 1 and stage 2 are separate checkpoints. A stage-2 checkpoint carries the id of
the stage-1 checkpoint it was fine-tuned from, so a served prediction can be traced
back to the representation it rests on. No trained checkpoint is committed to this
repository; `write_untrained_checkpoint` exists for wiring tests and marks itself as
untrained everywhere it surfaces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import plant_sim

from . import events as event_rules
from . import telemetry as telemetry_schema
from .horizons import HorizonGrid

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps torch out of import time
    from .models import Hepa

#: A self-supervised representation: encoder + predictor, no event head trained.
STAGE_PRETRAIN = "stage1"
#: An event predictor: frozen encoder, fine-tuned predictor and survival head.
STAGE_FINETUNE = "stage2"
STAGES = (STAGE_PRETRAIN, STAGE_FINETUNE)

METADATA_FILE = "metadata.json"
WEIGHTS_FILE = "weights.pt"

#: Environment variable the backend reads at start-up for a checkpoint to serve.
CHECKPOINT_ENV = "PLANT_HEPA_CHECKPOINT"


class HepaCheckpointError(ValueError):
    """A checkpoint could not be used. The message says why, in operator terms."""


class MissingCheckpointError(HepaCheckpointError):
    """Nothing at the given path."""


class CorruptCheckpointError(HepaCheckpointError):
    """The metadata is missing, unreadable, or not describing a HEPA checkpoint."""


class IncompatibleCheckpointError(HepaCheckpointError):
    """Trained against a different feature schema, event definition or model shape."""


@dataclass(frozen=True)
class CheckpointMetadata:
    """Everything needed to decide whether this model may answer about this plant."""

    stage: str
    model: dict[str, Any] = field(default_factory=dict)
    feature_names: tuple[str, ...] = telemetry_schema.FEATURE_NAMES
    schema_version: int = telemetry_schema.SCHEMA_VERSION
    event_names: tuple[str, ...] = event_rules.EVENT_NAMES
    event_spec_version: int = event_rules.EVENT_SPEC_VERSION
    thresholds: event_rules.EventThresholds = field(default_factory=event_rules.EventThresholds)
    horizons: HorizonGrid = field(default_factory=HorizonGrid.of_hours)
    context_len: int = 144
    normalisation: telemetry_schema.NormalisationStats = field(
        default_factory=telemetry_schema.NormalisationStats.identity
    )
    plant_sim_version: str = plant_sim.__version__
    created_at: str = ""
    #: Provenance. `epochs` and `steps` stay 0 for an untrained artefact.
    seed: int | None = None
    epochs: int = 0
    steps: int = 0
    training_config: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    #: Validation metrics recorded by the training run. Never invented elsewhere.
    metrics: dict[str, Any] = field(default_factory=dict)
    #: For stage 2: the stage-1 checkpoint this was fine-tuned from.
    pretrained_from: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise CorruptCheckpointError(
                f"unknown stage {self.stage!r}: expected one of {', '.join(STAGES)}"
            )

    @property
    def trained(self) -> bool:
        """False for a wiring artefact: nothing may present it as a trained model."""
        return self.steps > 0

    @property
    def horizon_labels(self) -> tuple[str, ...]:
        return self.horizons.labels

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "model": dict(self.model),
            "feature_names": list(self.feature_names),
            "schema_version": self.schema_version,
            "event_names": list(self.event_names),
            "event_spec_version": self.event_spec_version,
            "thresholds": self.thresholds.to_dict(),
            "horizons": self.horizons.to_dict(),
            "context_len": self.context_len,
            "normalisation": self.normalisation.to_dict(),
            "plant_sim_version": self.plant_sim_version,
            "created_at": self.created_at,
            "seed": self.seed,
            "epochs": self.epochs,
            "steps": self.steps,
            "training_config": dict(self.training_config),
            "dataset": dict(self.dataset),
            "metrics": dict(self.metrics),
            "pretrained_from": self.pretrained_from,
            "notes": self.notes,
            "trained": self.trained,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, path: Path | None = None) -> "CheckpointMetadata":
        where = f" in {path}" if path else ""
        try:
            stage = str(raw["stage"])
        except (KeyError, TypeError) as reason:
            raise CorruptCheckpointError(
                f"checkpoint metadata has no stage{where}"
            ) from reason
        if stage not in STAGES:
            raise CorruptCheckpointError(
                f"unsupported checkpoint stage {stage!r}{where}: expected {', '.join(STAGES)}"
            )
        return cls(
            stage=stage,
            model=dict(raw.get("model") or {}),
            feature_names=tuple(str(name) for name in raw.get("feature_names", ())),
            schema_version=int(raw.get("schema_version", 0)),
            event_names=tuple(str(name) for name in raw.get("event_names", ())),
            event_spec_version=int(raw.get("event_spec_version", 0)),
            thresholds=event_rules.EventThresholds.from_dict(raw.get("thresholds")),
            horizons=HorizonGrid.from_dict(raw.get("horizons")),
            context_len=int(raw.get("context_len", 0)),
            normalisation=telemetry_schema.NormalisationStats.from_dict(
                raw.get("normalisation")
            ),
            plant_sim_version=str(raw.get("plant_sim_version", "unknown")),
            created_at=str(raw.get("created_at", "")),
            seed=None if raw.get("seed") is None else int(raw["seed"]),
            epochs=int(raw.get("epochs", 0)),
            steps=int(raw.get("steps", 0)),
            training_config=dict(raw.get("training_config") or {}),
            dataset=dict(raw.get("dataset") or {}),
            metrics=dict(raw.get("metrics") or {}),
            pretrained_from=str(raw.get("pretrained_from", "")),
            notes=str(raw.get("notes", "")),
        )

    def check_compatible(self, *, path: Path | None = None) -> None:
        """Raise unless this checkpoint speaks this build's schema and event rules."""
        where = f" ({path})" if path else ""
        if self.schema_version != telemetry_schema.SCHEMA_VERSION:
            raise IncompatibleCheckpointError(
                f"checkpoint telemetry schema version {self.schema_version} does not match "
                f"this build's {telemetry_schema.SCHEMA_VERSION}{where}: retrain"
            )
        if self.event_spec_version != event_rules.EVENT_SPEC_VERSION:
            raise IncompatibleCheckpointError(
                f"checkpoint event rules version {self.event_spec_version} does not match "
                f"this build's {event_rules.EVENT_SPEC_VERSION}{where}: relabel and retrain"
            )
        if self.feature_names != telemetry_schema.FEATURE_NAMES:
            raise IncompatibleCheckpointError(
                f"checkpoint features differ from this plant's telemetry schema{where}: "
                f"{_first_difference(self.feature_names, telemetry_schema.FEATURE_NAMES)}"
            )
        if self.event_names != event_rules.EVENT_NAMES:
            raise IncompatibleCheckpointError(
                f"checkpoint events differ from this build's event rules{where}: "
                f"{_first_difference(self.event_names, event_rules.EVENT_NAMES)}"
            )
        model_input = int(self.model.get("input_dim", telemetry_schema.FEATURE_COUNT))
        if model_input != telemetry_schema.FEATURE_COUNT:
            raise IncompatibleCheckpointError(
                f"checkpoint model expects {model_input} input features, this plant has "
                f"{telemetry_schema.FEATURE_COUNT}{where}"
            )


def _first_difference(recorded: tuple[str, ...], current: tuple[str, ...]) -> str:
    if len(recorded) != len(current):
        return f"{len(recorded)} recorded vs {len(current)} in this build"
    for index, (was, now) in enumerate(zip(recorded, current)):
        if was != now:
            return f"position {index}: {was!r} -> {now!r}"
    return "no difference"


def metadata_path(path: Path | str) -> Path:
    return Path(path) / METADATA_FILE


def weights_path(path: Path | str) -> Path:
    return Path(path) / WEIGHTS_FILE


def save_metadata(path: Path | str, metadata: CheckpointMetadata) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    payload = metadata.to_dict()
    if not payload.get("created_at"):
        payload["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    target = metadata_path(directory)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_metadata(path: Path | str, *, check: bool = True) -> CheckpointMetadata:
    """Read, and by default verify, a checkpoint's metadata. No torch involved."""
    directory = Path(path)
    if not directory.exists():
        raise MissingCheckpointError(f"no HEPA checkpoint at {directory}")
    target = metadata_path(directory)
    if not target.is_file():
        raise CorruptCheckpointError(
            f"{directory} is not a HEPA checkpoint: expected {METADATA_FILE} inside it"
        )
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as reason:
        raise CorruptCheckpointError(f"cannot read {target}: {reason}") from reason
    if not isinstance(raw, dict):
        raise CorruptCheckpointError(f"{target} does not contain a metadata object")
    metadata = CheckpointMetadata.from_dict(raw, path=directory)
    if check:
        metadata.check_compatible(path=directory)
    return metadata


def save(path: Path | str, model: "Hepa", metadata: CheckpointMetadata) -> Path:
    """Write weights and metadata, with the model's own shape recorded from it."""
    import torch

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path(directory))
    save_metadata(directory, replace(metadata, model=model.config.to_dict()))
    return directory


def load_model(
    path: Path | str, *, check: bool = True, map_location: str = "cpu"
) -> tuple["Hepa", CheckpointMetadata]:
    """Rebuild the model this checkpoint describes and load its weights into it."""
    import torch

    from .models import Hepa, HepaConfig

    directory = Path(path)
    metadata = load_metadata(directory, check=check)
    target = weights_path(directory)
    if not target.is_file():
        raise CorruptCheckpointError(f"{directory} has no {WEIGHTS_FILE}")
    model = Hepa(HepaConfig.from_dict(metadata.model))
    try:
        state = torch.load(target, map_location=map_location, weights_only=True)
    except Exception as reason:  # torch raises several unrelated types here
        raise CorruptCheckpointError(f"cannot read {target}: {reason}") from reason
    try:
        model.load_state_dict(state)
    except RuntimeError as reason:
        raise IncompatibleCheckpointError(
            f"weights in {target} do not fit the recorded model shape: {reason}"
        ) from reason
    model.eval()
    return model, metadata


def write_untrained_checkpoint(
    path: Path | str,
    *,
    stage: str = STAGE_FINETUNE,
    horizons: HorizonGrid | None = None,
    context_len: int = 144,
    notes: str = "Untrained weights. For wiring checks only -- never a prediction.",
) -> CheckpointMetadata:
    """A structurally valid checkpoint holding freshly initialised weights.

    Used by tests and by anyone verifying that the API, the loader and the
    inference path fit together. `trained` is False, and every surface that
    reports on a checkpoint reports that.
    """
    from .models import Hepa, HepaConfig

    grid = horizons or HorizonGrid.of_hours()
    config = HepaConfig(context_len=context_len, max_horizon_steps=grid.max_steps)
    metadata = CheckpointMetadata(
        stage=stage,
        model=config.to_dict(),
        horizons=grid,
        context_len=context_len,
        notes=notes,
    )
    save(path, Hepa(config), metadata)
    return metadata


__all__ = [
    "CHECKPOINT_ENV",
    "METADATA_FILE",
    "STAGES",
    "STAGE_FINETUNE",
    "STAGE_PRETRAIN",
    "WEIGHTS_FILE",
    "CheckpointMetadata",
    "CorruptCheckpointError",
    "HepaCheckpointError",
    "IncompatibleCheckpointError",
    "MissingCheckpointError",
    "load_metadata",
    "load_model",
    "metadata_path",
    "save",
    "save_metadata",
    "weights_path",
    "write_untrained_checkpoint",
]

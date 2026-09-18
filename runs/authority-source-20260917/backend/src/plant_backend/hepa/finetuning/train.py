"""Stage 2: event fine-tuning on top of a frozen representation.

    python -m plant_backend.hepa.finetuning.train --config .../configs/finetune.json \
        --dataset datasets/demo --pretrained checkpoints/hepa/stage1 \
        --out checkpoints/hepa/stage2

What one step does:

    context --f_theta (frozen)--> h_t --g_phi(., anchors)--> hazards --survival--> p(event, Δt)
    loss = positive-weighted BCE(p, cumulative labels), masked where the future is unknown

The encoder is frozen exactly as the paper's downstream protocol specifies: the
representation is what pretraining produced, and stage 2 only learns how to read it. It
is also held in eval mode, so dropout stops perturbing a representation that is meant to
be fixed (`Hepa.freeze_encoder`). `freeze_encoder = false` exists in the config for
someone who wants to measure the difference, not as a default.

The predictor `g_phi` is *not* frozen: it is the horizon-conditioned part, and the
hazards are read off its output at each anchor. Fine-tuning it is what turns "where will
the plant be" into "how likely is this failure by then".

Validation reports the loss and per-event, per-horizon AUROC/AUPRC on the validation
split (`evaluation.py`), and the best epoch by validation loss is what gets saved.

Nothing here has been run at scale in this repository -- `--dry-run` prints the plan, and
the tests take a couple of steps on a tiny synthetic dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .. import evaluation
from ..checkpoints import (
    STAGE_FINETUNE,
    STAGE_PRETRAIN,
    CheckpointMetadata,
    HepaCheckpointError,
    load_model,
    save,
)
from ..data.tensors import WindowDataset, make_loader
from ..data.trajectories import TrajectoryDataset
from ..data.windows import DEFAULT_CONTEXT_LEN, WindowIndex, WindowSpec
from ..models import Hepa, HepaConfig
from ..reproducibility import seed_all
from .objective import event_loss, positive_weights


@dataclass(frozen=True)
class FinetuneConfig:
    """Stage-2 hyperparameters. Defaults are the paper's downstream protocol."""

    epochs: int = 50
    patience: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    context_len: int = DEFAULT_CONTEXT_LEN
    seed: int = 0
    device: str = "cpu"
    #: The paper's protocol. Turning this off is an experiment, not a default.
    freeze_encoder: bool = True
    max_steps_per_epoch: int = 0
    model: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "context_len": self.context_len,
            "seed": self.seed,
            "device": self.device,
            "freeze_encoder": self.freeze_encoder,
            "max_steps_per_epoch": self.max_steps_per_epoch,
            "model": dict(self.model),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "FinetuneConfig":
        if not raw:
            return cls()
        base = cls()
        unknown = set(raw) - set(base.to_dict())
        if unknown:
            raise ValueError(f"unknown fine-tuning settings: {', '.join(sorted(unknown))}")
        return replace(
            base,
            epochs=int(raw.get("epochs", base.epochs)),
            patience=int(raw.get("patience", base.patience)),
            batch_size=int(raw.get("batch_size", base.batch_size)),
            learning_rate=float(raw.get("learning_rate", base.learning_rate)),
            weight_decay=float(raw.get("weight_decay", base.weight_decay)),
            context_len=int(raw.get("context_len", base.context_len)),
            seed=int(raw.get("seed", base.seed)),
            device=str(raw.get("device", base.device)),
            freeze_encoder=bool(raw.get("freeze_encoder", base.freeze_encoder)),
            max_steps_per_epoch=int(raw.get("max_steps_per_epoch", base.max_steps_per_epoch)),
            model=dict(raw.get("model") or {}),
        )

    @classmethod
    def load(cls, path: Path | str) -> "FinetuneConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class FinetuneRun:
    """What a finished run produced."""

    model: Hepa
    metadata: CheckpointMetadata
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    best_loss: float = math.inf
    validation_report: dict[str, Any] = field(default_factory=dict)


def load_pretrained(
    path: Path | str | None, config: FinetuneConfig, spec: WindowSpec
) -> tuple[Hepa, str]:
    """The stage-1 model to fine-tune, or a fresh one when no checkpoint is given.

    Starting from scratch is allowed (it is the ablation the paper reports against),
    but it is recorded in the checkpoint so nothing can later claim a pretrained
    representation it never had.
    """
    if path is None:
        overrides = dict(config.model)
        overrides.pop("context_len", None)
        overrides.pop("max_horizon_steps", None)
        model = Hepa(
            HepaConfig(
                context_len=spec.context_len, max_horizon_steps=spec.max_steps, **overrides
            )
        )
        return model, ""
    model, metadata = load_model(path)
    if metadata.stage != STAGE_PRETRAIN:
        raise HepaCheckpointError(
            f"{path} is a {metadata.stage} checkpoint; stage 2 starts from a "
            f"{STAGE_PRETRAIN} checkpoint"
        )
    if metadata.context_len != spec.context_len:
        raise HepaCheckpointError(
            f"{path} was pretrained with a context of {metadata.context_len} samples, "
            f"fine-tuning asks for {spec.context_len}"
        )
    if model.config.max_horizon_steps != spec.max_steps:
        raise HepaCheckpointError(
            f"{path} was pretrained for horizons up to {model.config.max_horizon_steps} "
            f"samples, this dataset labels up to {spec.max_steps}"
        )
    return model, str(path)


def finetune(
    config: FinetuneConfig,
    dataset_root: Path | str,
    pretrained: Path | str | None = None,
    out: Path | str | None = None,
    *,
    on_epoch: Callable[[dict[str, float]], None] | None = None,
) -> FinetuneRun:
    """Run stage 2 and, if `out` is given, save the best checkpoint into it."""
    import torch

    seed_all(config.seed)
    device = torch.device(config.device)

    dataset = TrajectoryDataset.load(dataset_root)
    grid = dataset.manifest.horizons
    spec = WindowSpec(context_len=config.context_len, horizons=grid)
    normalisation = dataset.manifest.normalisation
    train_index = WindowIndex(dataset.load_split("train"), spec, normalisation=normalisation)
    val_index = WindowIndex(dataset.load_split("val"), spec, normalisation=normalisation)
    if len(train_index) == 0:
        raise ValueError(
            f"no training windows in {dataset_root}: episodes need at least "
            f"{spec.min_samples} samples"
        )

    model, pretrained_from = load_pretrained(pretrained, config, spec)
    model = model.to(device)
    if config.freeze_encoder:
        model.freeze_encoder()
    optimiser = torch.optim.AdamW(
        list(model.finetune_parameters()) if config.freeze_encoder else list(model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    steps = list(grid.steps)
    weights = torch.from_numpy(_positive_weights(train_index)).to(device)
    train_loader = make_loader(
        WindowDataset(train_index),
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    val_loader = make_loader(WindowDataset(val_index), batch_size=config.batch_size)

    run = FinetuneRun(
        model=model,
        metadata=_metadata(config, spec, dataset, model, pretrained_from),
    )
    best_state: dict[str, Any] | None = None
    stale = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        for batch in train_loader:
            if config.max_steps_per_epoch and batches >= config.max_steps_per_epoch:
                break
            logits = model.event_logits(batch["context"].to(device), steps)
            loss = event_loss(
                logits,
                batch["labels"].to(device),
                batch["valid"].to(device),
                weights=weights,
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
            batches += 1

        record = {
            "epoch": float(epoch),
            "steps": float(batches),
            "loss": total / max(batches, 1),
            "val_loss": _validation_loss(model, val_loader, steps, weights, device),
        }
        run.history.append(record)
        if on_epoch is not None:
            on_epoch(record)

        score = record["val_loss"] if math.isfinite(record["val_loss"]) else record["loss"]
        if score < run.best_loss:
            run.best_loss = score
            run.best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if config.patience and stale >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if len(val_index):
        run.validation_report = _report(model, val_index, steps, grid, device, config.batch_size)
    run.metadata = replace(
        run.metadata,
        epochs=len(run.history),
        steps=int(sum(record["steps"] for record in run.history)),
        metrics={
            "best_epoch": run.best_epoch,
            "best_val_loss": run.best_loss,
            "validation": _headline(run.validation_report),
        },
    )
    if out is not None:
        save(out, model, run.metadata)
    return run


def _positive_weights(index: WindowIndex) -> np.ndarray:
    """`N_neg / N_pos` per event and horizon, measured on the training windows."""
    import torch

    labels = []
    valid = []
    for position in range(len(index)):
        window = index.window(position)
        labels.append(window.labels)
        valid.append(window.valid)
    weights = positive_weights(
        torch.from_numpy(np.stack(labels)), torch.from_numpy(np.stack(valid))
    )
    return weights.numpy().astype(np.float32)


def _validation_loss(
    model: Hepa, loader: Any, steps: Sequence[int], weights: Any, device: Any
) -> float:
    import torch

    if len(loader.dataset) == 0:
        return math.inf
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            logits = model.event_logits(batch["context"].to(device), list(steps))
            total += float(
                event_loss(
                    logits,
                    batch["labels"].to(device),
                    batch["valid"].to(device),
                    weights=weights,
                )
            )
            batches += 1
    return total / max(batches, 1)


def _report(
    model: Hepa,
    index: WindowIndex,
    steps: Sequence[int],
    grid: Any,
    device: Any,
    batch_size: int,
) -> dict[str, Any]:
    """Per-event, per-horizon metrics on the validation windows."""
    import torch

    from ..events import EVENT_NAMES

    contexts = []
    labels = []
    valid = []
    for position in range(len(index)):
        window = index.window(position)
        contexts.append(window.context)
        labels.append(window.labels)
        valid.append(window.valid)

    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(contexts), batch_size):
            batch = torch.from_numpy(np.stack(contexts[start : start + batch_size])).to(device)
            predictions.append(model.event_probabilities(batch, list(steps)).cpu().numpy())
    return evaluation.evaluate(
        np.concatenate(predictions),
        np.stack(labels),
        np.stack(valid),
        event_names=EVENT_NAMES,
        horizon_labels=grid.labels,
    )


def _headline(report: dict[str, Any]) -> dict[str, Any]:
    """AUROC/AUPRC per event and horizon, without the calibration tables."""
    if not report:
        return {}
    return {
        event: {
            horizon: {
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "base_rate": metrics["base_rate"],
            }
            for horizon, metrics in horizons.items()
        }
        for event, horizons in report["events"].items()
    }


def _metadata(
    config: FinetuneConfig,
    spec: WindowSpec,
    dataset: TrajectoryDataset,
    model: Hepa,
    pretrained_from: str,
) -> CheckpointMetadata:
    manifest = dataset.manifest
    return CheckpointMetadata(
        stage=STAGE_FINETUNE,
        model=model.config.to_dict(),
        thresholds=manifest.thresholds,
        horizons=manifest.horizons,
        context_len=spec.context_len,
        normalisation=manifest.normalisation,
        seed=config.seed,
        training_config=config.to_dict(),
        dataset={
            "root": str(dataset.root),
            "episodes": len(manifest.episodes),
            "counts": dataset.counts(),
        },
        pretrained_from=pretrained_from,
        notes=(
            "Event predictor."
            if pretrained_from
            else "Event predictor trained without a pretrained representation."
        ),
    )


def plan(config: FinetuneConfig, dataset_root: Path | str) -> dict[str, Any]:
    """What a run would do. No torch, nothing built."""
    dataset = TrajectoryDataset.load(dataset_root)
    spec = WindowSpec(context_len=config.context_len, horizons=dataset.manifest.horizons)
    windows = {
        split: len(
            WindowIndex(
                dataset.load_split(split), spec, normalisation=dataset.manifest.normalisation
            )
        )
        for split in ("train", "val", "test")
    }
    return {
        "dataset": str(dataset.root),
        "episodes": dataset.counts(),
        "windows": windows,
        "horizons": dataset.manifest.horizons.to_dict(),
        "batches_per_epoch": -(-windows["train"] // max(config.batch_size, 1)),
        "config": config.to_dict(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m plant_backend.hepa.finetuning.train",
        description="Stage 2: fine-tune a HEPA representation into an event predictor.",
    )
    parser.add_argument("--config", type=Path, help="JSON fine-tuning config")
    parser.add_argument("--dataset", type=Path, required=True, help="dataset directory")
    parser.add_argument("--pretrained", type=Path, help="stage-1 checkpoint to start from")
    parser.add_argument("--out", type=Path, help="checkpoint directory to write")
    parser.add_argument("--epochs", type=int, help="override the epoch count")
    parser.add_argument("--seed", type=int, help="override the seed")
    parser.add_argument("--device", help="override the device (cpu, cuda, ...)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and train nothing")
    args = parser.parse_args(argv)

    config = FinetuneConfig.load(args.config) if args.config else FinetuneConfig()
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.device:
        config = replace(config, device=args.device)

    if args.dry_run:
        print(json.dumps(plan(config, args.dataset), indent=2))
        print("dry run: nothing trained")
        return 0

    run = finetune(
        config,
        args.dataset,
        args.pretrained,
        args.out,
        on_epoch=lambda record: print(
            f"epoch {int(record['epoch']):>4}  loss {record['loss']:.4f}  "
            f"val {record['val_loss']:.4f}"
        ),
    )
    print(f"best epoch {run.best_epoch} at validation loss {run.best_loss:.4f}")
    if run.validation_report:
        print(evaluation.summarise(run.validation_report))
    if args.out:
        print(f"checkpoint written to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "FinetuneConfig",
    "FinetuneRun",
    "finetune",
    "load_pretrained",
    "main",
    "plan",
]

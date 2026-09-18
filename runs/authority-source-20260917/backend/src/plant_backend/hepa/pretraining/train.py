"""Stage 1: self-supervised pretraining, end to end.

    python -m plant_backend.hepa.pretraining.train --config .../configs/pretrain.json \
        --dataset datasets/demo --out checkpoints/hepa/stage1

What one step does:

    context, future, Δt ~ LogUniform
        -> h_t = f_theta(context)
        -> h_hat = g_phi(h_t, Δt)
        -> target = pool(f_theta(future))    masked to Δt
        -> loss = (1 - alpha) L1(h_hat, target) + alpha SIGReg(h_hat)

No sensor value is regressed anywhere in this loop. That is the point of a JEPA-style
objective: predicting raw telemetry 24 h ahead spends the whole model on the parts of
the signal nobody can predict (exact cloud shapes), while predicting a *representation*
lets it keep only what is predictable and useful downstream.

Everything is deterministic given `seed`: the loader's shuffle, the drawn horizons and
SIGReg's projections all come from seeded generators, and validation uses its own fixed
generator so the reported number moves only when the model does.

Nothing here has been run at scale in this repository. `--dry-run` prints the plan --
window counts, batches, parameter counts -- without touching the optimiser, and the
test suite runs a couple of optimisation steps on a tiny synthetic dataset. A real run
belongs on a machine with a GPU.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

from ..checkpoints import STAGE_PRETRAIN, CheckpointMetadata, save
from ..data.tensors import WindowDataset, make_loader
from ..data.trajectories import TrajectoryDataset
from ..data.windows import DEFAULT_CONTEXT_LEN, WindowIndex, WindowSpec
from ..models import Hepa, HepaConfig
from ..reproducibility import seed_all
from .objective import DEFAULT_PROJECTIONS, DEFAULT_SIGREG_WEIGHT, pretrain_loss, sample_horizons


@dataclass(frozen=True)
class PretrainConfig:
    """Stage-1 hyperparameters. Defaults are the paper's where the paper has one."""

    epochs: int = 100
    patience: int = 10
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    sigreg_weight: float = DEFAULT_SIGREG_WEIGHT
    projections: int = DEFAULT_PROJECTIONS
    context_len: int = DEFAULT_CONTEXT_LEN
    seed: int = 0
    device: str = "cpu"
    #: Cap on optimisation steps per epoch. 0 means "the whole split". Used by the
    #: smoke tests, and by anyone who wants a short run without editing the data.
    max_steps_per_epoch: int = 0
    #: Stop the target encoder's gradients. The paper does not, and SIGReg is what
    #: makes that safe; the switch exists to make the choice explicit and testable.
    detach_target: bool = False
    model: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "sigreg_weight": self.sigreg_weight,
            "projections": self.projections,
            "context_len": self.context_len,
            "seed": self.seed,
            "device": self.device,
            "max_steps_per_epoch": self.max_steps_per_epoch,
            "detach_target": self.detach_target,
            "model": dict(self.model),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "PretrainConfig":
        if not raw:
            return cls()
        base = cls()
        unknown = set(raw) - set(base.to_dict())
        if unknown:
            raise ValueError(f"unknown pretraining settings: {', '.join(sorted(unknown))}")
        return replace(
            base,
            epochs=int(raw.get("epochs", base.epochs)),
            patience=int(raw.get("patience", base.patience)),
            batch_size=int(raw.get("batch_size", base.batch_size)),
            learning_rate=float(raw.get("learning_rate", base.learning_rate)),
            weight_decay=float(raw.get("weight_decay", base.weight_decay)),
            sigreg_weight=float(raw.get("sigreg_weight", base.sigreg_weight)),
            projections=int(raw.get("projections", base.projections)),
            context_len=int(raw.get("context_len", base.context_len)),
            seed=int(raw.get("seed", base.seed)),
            device=str(raw.get("device", base.device)),
            max_steps_per_epoch=int(raw.get("max_steps_per_epoch", base.max_steps_per_epoch)),
            detach_target=bool(raw.get("detach_target", base.detach_target)),
            model=dict(raw.get("model") or {}),
        )

    @classmethod
    def load(cls, path: Path | str) -> "PretrainConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class PretrainRun:
    """What a finished (or dry) run produced."""

    model: Hepa
    metadata: CheckpointMetadata
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    best_loss: float = math.inf

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "history": self.history,
            "parameters": self.model.parameter_counts(),
        }


def build_model(config: PretrainConfig, spec: WindowSpec) -> Hepa:
    """A model whose shapes agree with the windows it will be fed."""
    overrides = dict(config.model)
    overrides.pop("context_len", None)
    overrides.pop("max_horizon_steps", None)
    return Hepa(
        HepaConfig(
            context_len=spec.context_len,
            max_horizon_steps=spec.max_steps,
            **overrides,
        )
    )


def pretrain(
    config: PretrainConfig,
    dataset_root: Path | str,
    out: Path | str | None = None,
    *,
    on_epoch: Callable[[dict[str, float]], None] | None = None,
) -> PretrainRun:
    """Run stage 1 and, if `out` is given, save the best checkpoint into it."""
    import torch

    generator = seed_all(config.seed)
    device = torch.device(config.device)

    dataset = TrajectoryDataset.load(dataset_root)
    spec = WindowSpec(context_len=config.context_len, horizons=dataset.manifest.horizons)
    normalisation = dataset.manifest.normalisation
    train_index = WindowIndex(dataset.load_split("train"), spec, normalisation=normalisation)
    val_index = WindowIndex(dataset.load_split("val"), spec, normalisation=normalisation)
    if len(train_index) == 0:
        raise ValueError(
            f"no training windows in {dataset_root}: episodes need at least "
            f"{spec.min_samples} samples for a context of {spec.context_len} and a "
            f"horizon of {spec.max_steps}"
        )

    model = build_model(config, spec).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_loader = make_loader(
        WindowDataset(train_index),
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    val_loader = make_loader(WindowDataset(val_index), batch_size=config.batch_size)

    run = PretrainRun(model=model, metadata=_metadata(config, spec, dataset, model))
    best_state: dict[str, Any] | None = None
    stale = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "prediction": 0.0, "sigreg": 0.0}
        steps = 0
        for batch in train_loader:
            if config.max_steps_per_epoch and steps >= config.max_steps_per_epoch:
                break
            loss = _batch_loss(model, batch, config, device, generator)
            optimiser.zero_grad(set_to_none=True)
            loss.total.backward()
            optimiser.step()
            for key, value in loss.to_dict().items():
                totals[key] += value
            steps += 1

        record = {"epoch": float(epoch), "steps": float(steps)}
        record.update({key: value / max(steps, 1) for key, value in totals.items()})
        record["val_loss"] = _validate(model, val_loader, config, device)
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
    run.metadata = replace(
        run.metadata,
        epochs=len(run.history),
        steps=int(sum(record["steps"] for record in run.history)),
        metrics={"best_epoch": run.best_epoch, "best_loss": run.best_loss},
    )
    if out is not None:
        save(out, model, run.metadata)
    return run


def _batch_loss(
    model: Hepa, batch: dict[str, Any], config: PretrainConfig, device: Any, generator: Any
) -> Any:
    context = batch["context"].to(device)
    future = batch["future"].to(device)
    horizons = sample_horizons(
        context.shape[0],
        max_steps=model.config.max_horizon_steps,
        patch_size=model.config.patch_size,
        generator=generator,
        device=None,
    ).to(device)
    context_latent = model.encode(context)
    predicted = model.predict_latent(context_latent, horizons)
    target = model.target_latent(future, horizons)
    if config.detach_target:
        target = target.detach()
    return pretrain_loss(
        predicted,
        target,
        sigreg_weight=config.sigreg_weight,
        projections=config.projections,
        generator=generator,
    )


def _validate(model: Hepa, loader: Any, config: PretrainConfig, device: Any) -> float:
    """Mean loss over the validation split, with horizons drawn from a fixed seed."""
    import torch

    if len(loader.dataset) == 0:
        return math.inf
    model.eval()
    generator = torch.Generator()
    generator.manual_seed(config.seed + 1)
    total = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            total += float(_batch_loss(model, batch, config, device, generator).total)
            batches += 1
    return total / max(batches, 1)


def _metadata(
    config: PretrainConfig, spec: WindowSpec, dataset: TrajectoryDataset, model: Hepa
) -> CheckpointMetadata:
    manifest = dataset.manifest
    return CheckpointMetadata(
        stage=STAGE_PRETRAIN,
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
        notes="Self-supervised representation. Not an event predictor on its own.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m plant_backend.hepa.pretraining.train",
        description="Stage 1: self-supervised HEPA pretraining on a collected dataset.",
    )
    parser.add_argument("--config", type=Path, help="JSON pretraining config")
    parser.add_argument("--dataset", type=Path, required=True, help="dataset directory")
    parser.add_argument("--out", type=Path, help="checkpoint directory to write")
    parser.add_argument("--epochs", type=int, help="override the epoch count")
    parser.add_argument("--seed", type=int, help="override the seed")
    parser.add_argument("--device", help="override the device (cpu, cuda, ...)")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and train nothing"
    )
    args = parser.parse_args(argv)

    config = PretrainConfig.load(args.config) if args.config else PretrainConfig()
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

    run = pretrain(
        config,
        args.dataset,
        args.out,
        on_epoch=lambda record: print(
            f"epoch {int(record['epoch']):>4}  loss {record['loss']:.4f}  "
            f"(L1 {record['prediction']:.4f}, sigreg {record['sigreg']:.4f})  "
            f"val {record['val_loss']:.4f}"
        ),
    )
    print(json.dumps(run.to_dict()["parameters"], indent=2))
    print(f"best epoch {run.best_epoch} at loss {run.best_loss:.4f}")
    if args.out:
        print(f"checkpoint written to {args.out}")
    return 0


def plan(config: PretrainConfig, dataset_root: Path | str) -> dict[str, Any]:
    """What a run would do, without importing torch or building anything."""
    dataset = TrajectoryDataset.load(dataset_root)
    spec = WindowSpec(context_len=config.context_len, horizons=dataset.manifest.horizons)
    windows = {
        split: len(
            WindowIndex(dataset.load_split(split), spec, normalisation=dataset.manifest.normalisation)
        )
        for split in ("train", "val", "test")
    }
    return {
        "dataset": str(dataset.root),
        "episodes": dataset.counts(),
        "windows": windows,
        "context_len": spec.context_len,
        "max_horizon_steps": spec.max_steps,
        "batches_per_epoch": -(-windows["train"] // max(config.batch_size, 1)),
        "config": config.to_dict(),
    }


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = ["PretrainConfig", "PretrainRun", "build_model", "main", "plan", "pretrain"]

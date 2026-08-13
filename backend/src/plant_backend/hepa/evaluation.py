"""Evaluation: the metrics that mean something for rare, horizon-conditioned events.

    python -m plant_backend.hepa.evaluation --checkpoint checkpoints/hepa/stage2 \
        --dataset datasets/demo --split test

Metrics are computed per event *and* per horizon, never pooled. Pooling across
horizons would let an easy 24 h label carry a useless 1 h one, and the 1 h number is
the one an operator would act on.

What is reported, and why each is here:

* **AUROC** -- ranking quality, insensitive to the base rate. Reads well but flatters
  a rare-event model: with 99.5% negatives, a nearly useless score can still look good.
* **AUPRC** -- the honest headline for rare events. Its floor is the base rate, so the
  base rate is reported next to it, always.
* **Precision / recall / F1** at a threshold, because an alarm is eventually a decision.
* **Brier score** and a reliability table -- a probability an operator is meant to read
  has to be calibrated, not merely ordered. A survival head is monotone in Δt but that
  says nothing about whether "0.3" happens three times in ten.

Everything is NumPy: no metrics dependency to install, and the arithmetic is small
enough to test against hand-computed arrays (`test_evaluation.py`).

No numbers from any real training run appear anywhere in this repository. Nothing here
has been run beyond the synthetic fixtures in the tests.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

#: Default alarm threshold for the threshold-dependent metrics.
DEFAULT_THRESHOLD = 0.5
#: Reliability-table resolution.
DEFAULT_BINS = 10


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve, by rank sum. NaN when one class is missing."""
    score = np.asarray(scores, dtype=np.float64).ravel()
    label = np.asarray(labels, dtype=np.float64).ravel() > 0.5
    positives = int(label.sum())
    negatives = int(label.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(score)
    ranks[order] = np.arange(1, score.size + 1, dtype=np.float64)
    # Average ranks inside ties, or every tie would count as a win for one side.
    sorted_scores = score[order]
    start = 0
    for index in range(1, sorted_scores.size + 1):
        if index == sorted_scores.size or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    rank_sum = ranks[label].sum()
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision: the step-wise area under the precision-recall curve."""
    score = np.asarray(scores, dtype=np.float64).ravel()
    label = np.asarray(labels, dtype=np.float64).ravel() > 0.5
    positives = int(label.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    hits = label[order].astype(np.float64)
    true_positives = np.cumsum(hits)
    precision = true_positives / np.arange(1, hits.size + 1, dtype=np.float64)
    return float((precision * hits).sum() / positives)


def brier(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error of the probabilities. Lower is better; 0.25 is a coin."""
    score = np.asarray(scores, dtype=np.float64).ravel()
    label = np.asarray(labels, dtype=np.float64).ravel()
    if score.size == 0:
        return float("nan")
    return float(np.mean((score - label) ** 2))


@dataclass(frozen=True)
class ThresholdMetrics:
    """Confusion-matrix metrics at one alarm threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "counts": {
                "tp": self.true_positives,
                "fp": self.false_positives,
                "fn": self.false_negatives,
                "tn": self.true_negatives,
            },
        }


def threshold_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> ThresholdMetrics:
    predicted = np.asarray(scores, dtype=np.float64).ravel() >= threshold
    actual = np.asarray(labels, dtype=np.float64).ravel() > 0.5
    true_positives = int(np.sum(predicted & actual))
    false_positives = int(np.sum(predicted & ~actual))
    false_negatives = int(np.sum(~predicted & actual))
    true_negatives = int(np.sum(~predicted & ~actual))
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else float("nan")
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else float("nan")
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and precision + recall > 0.0
        else float("nan")
    )
    return ThresholdMetrics(
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
    )


def reliability(
    scores: np.ndarray, labels: np.ndarray, bins: int = DEFAULT_BINS
) -> dict[str, Any]:
    """Predicted versus observed frequency per probability bin, plus the ECE."""
    score = np.asarray(scores, dtype=np.float64).ravel()
    label = np.asarray(labels, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    error = 0.0
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        inside = (score >= low) & (score < high if index < bins - 1 else score <= high)
        count = int(inside.sum())
        if count == 0:
            rows.append({"bin": [low, high], "count": 0, "predicted": None, "observed": None})
            continue
        predicted = float(score[inside].mean())
        observed = float(label[inside].mean())
        rows.append(
            {"bin": [low, high], "count": count, "predicted": predicted, "observed": observed}
        )
        error += count / score.size * abs(predicted - observed)
    return {"bins": rows, "expected_calibration_error": error}


def metrics_for(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    bins: int = DEFAULT_BINS,
) -> dict[str, Any]:
    """Every metric for one `(event, horizon)` pair."""
    label = np.asarray(labels, dtype=np.float64).ravel()
    return {
        "count": int(label.size),
        "positives": int((label > 0.5).sum()),
        "base_rate": float(label.mean()) if label.size else float("nan"),
        "auroc": auroc(scores, labels),
        "auprc": auprc(scores, labels),
        "brier": brier(scores, labels),
        **threshold_metrics(scores, labels, threshold).to_dict(),
        "calibration": reliability(scores, labels, bins),
    }


def evaluate(
    probabilities: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    *,
    event_names: Sequence[str],
    horizon_labels: Sequence[str],
    threshold: float = DEFAULT_THRESHOLD,
    bins: int = DEFAULT_BINS,
) -> dict[str, Any]:
    """Per-event, per-horizon metrics from `(N, E, H)` predictions and labels.

    `valid` is `(N, H)`: windows whose horizon runs past the end of the episode are
    excluded rather than counted as negatives.
    """
    probability = np.asarray(probabilities, dtype=np.float64)
    label = np.asarray(labels, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if probability.shape != label.shape:
        raise ValueError("probabilities and labels must have the same shape")
    if probability.ndim != 3:
        raise ValueError("expected (N, E, H) probabilities")
    if mask.shape != (probability.shape[0], probability.shape[2]):
        raise ValueError("valid must be (N, H)")
    if len(event_names) != probability.shape[1] or len(horizon_labels) != probability.shape[2]:
        raise ValueError("event_names and horizon_labels must match the array shape")

    report: dict[str, Any] = {"events": {}, "threshold": threshold}
    for event_index, event in enumerate(event_names):
        per_horizon: dict[str, Any] = {}
        for horizon_index, horizon in enumerate(horizon_labels):
            keep = mask[:, horizon_index]
            per_horizon[horizon] = metrics_for(
                probability[keep, event_index, horizon_index],
                label[keep, event_index, horizon_index],
                threshold=threshold,
                bins=bins,
            )
        report["events"][event] = per_horizon
    return report


def summarise(report: dict[str, Any]) -> str:
    """A compact table of the headline numbers, for a terminal or a log."""
    lines = [f"{'event':<28}{'horizon':>8}{'base':>8}{'AUROC':>8}{'AUPRC':>8}{'Brier':>8}"]
    for event, horizons in report["events"].items():
        for horizon, metrics in horizons.items():
            lines.append(
                f"{event:<28}{horizon:>8}{metrics['base_rate']:>8.3f}"
                f"{metrics['auroc']:>8.3f}{metrics['auprc']:>8.3f}{metrics['brier']:>8.4f}"
            )
    return "\n".join(lines)


@dataclass(frozen=True)
class EvalConfig:
    """What to score and how. Everything a config file can set."""

    split: str = "test"
    batch_size: int = 64
    threshold: float = DEFAULT_THRESHOLD
    bins: int = DEFAULT_BINS

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "batch_size": self.batch_size,
            "threshold": self.threshold,
            "bins": self.bins,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "EvalConfig":
        if not raw:
            return cls()
        base = cls()
        unknown = set(raw) - set(base.to_dict())
        if unknown:
            raise ValueError(f"unknown evaluation settings: {', '.join(sorted(unknown))}")
        split = str(raw.get("split", base.split))
        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown split {split!r}: expected train, val or test")
        return cls(
            split=split,
            batch_size=int(raw.get("batch_size", base.batch_size)),
            threshold=float(raw.get("threshold", base.threshold)),
            bins=int(raw.get("bins", base.bins)),
        )

    @classmethod
    def load(cls, path: Path | str) -> "EvalConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate a stage-2 checkpoint on one split of a collected dataset."""
    parser = argparse.ArgumentParser(
        prog="python -m plant_backend.hepa.evaluation",
        description="Evaluate a fine-tuned HEPA checkpoint per event and per horizon.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="stage-2 checkpoint")
    parser.add_argument("--dataset", type=Path, required=True, help="dataset directory")
    parser.add_argument("--config", type=Path, help="JSON evaluation config")
    parser.add_argument("--split", choices=("train", "val", "test"), help="override the split")
    parser.add_argument("--batch-size", type=int, help="override the batch size")
    parser.add_argument("--threshold", type=float, help="override the alarm threshold")
    parser.add_argument("--out", type=Path, help="write the full report as JSON here")
    args = parser.parse_args(argv)

    config = EvalConfig.load(args.config) if args.config else EvalConfig()
    if args.split:
        config = replace(config, split=args.split)
    if args.batch_size is not None:
        config = replace(config, batch_size=args.batch_size)
    if args.threshold is not None:
        config = replace(config, threshold=args.threshold)

    from .inference import EventPredictor  # imports torch

    predictor = EventPredictor.load(args.checkpoint)
    report = predictor.evaluate_split(
        args.dataset,
        split=config.split,
        batch_size=config.batch_size,
        threshold=config.threshold,
    )
    print(summarise(report))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"report written to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BINS",
    "DEFAULT_THRESHOLD",
    "EvalConfig",
    "ThresholdMetrics",
    "auprc",
    "auroc",
    "brier",
    "evaluate",
    "main",
    "metrics_for",
    "reliability",
    "summarise",
    "threshold_metrics",
]

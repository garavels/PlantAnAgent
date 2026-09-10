"""Exact paired statistics and clustered uncertainty without environment access."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from statistics import NormalDist
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .contract import (
    BETA_OPTIMIZER_SEEDS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EPISODE_COUNT,
    PREFIX,
    PROFILE_THRESHOLDS_EUR,
    analysis_contract,
)

Vector = NDArray[np.float64]
NullableFamily = Mapping[int, Sequence[object] | None]


def finite_real(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("study values must be finite real numbers")
    try:
        result = float(value)
    except OverflowError as reason:
        raise ValueError(
            "study values must be representable as finite float64"
        ) from reason
    if not math.isfinite(result):
        raise ValueError("study values must be finite real numbers")
    return result


def complete_vector(values: Sequence[object]) -> Vector:
    if isinstance(values, (str, bytes, bytearray, Mapping)) or not isinstance(
        values, (Sequence, np.ndarray)
    ):
        raise ValueError("paired values must be an ordered numeric sequence")
    if len(values) != EPISODE_COUNT:
        raise ValueError("a paired policy vector must contain exactly 32 observations")
    return np.array([finite_real(value) for value in values], dtype=np.float64)


def binomial(successes: int, total: int) -> dict[str, Any]:
    if (
        type(total) is not int
        or type(successes) is not int
        or not 0 <= successes <= total
    ):
        raise ValueError("binomial counts must be exact non-negative integers")
    if total == 0:
        return {"successes": 0, "total": 0, "rate": None, "wilson_95": None}
    p = successes / total
    z = NormalDist().inv_cdf(0.975)
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return {
        "successes": successes,
        "total": total,
        "rate": p,
        "wilson_95": [max(0.0, centre - half), min(1.0, centre + half)],
    }


def _metrics(samples: Vector) -> dict[str, Vector]:
    """The last axis is always one full 32-observation vector, including repeats."""
    if samples.shape[-1] != EPISODE_COUNT:
        raise ValueError("statistics require the frozen episode cardinality")
    ordered = np.sort(samples, axis=-1)
    with np.errstate(over="ignore", invalid="ignore"):
        result = {
            "mean": np.mean(samples, axis=-1),
            "sample_std": np.std(samples, axis=-1, ddof=1),
            "median": np.median(ordered, axis=-1),
            "iqm": np.mean(ordered[..., 8:24], axis=-1),
            "cvar20": np.mean(ordered[..., :7], axis=-1),
            "minimum": ordered[..., 0],
            "maximum": ordered[..., -1],
            "positive_fraction": np.mean(samples > 0, axis=-1),
        }
    if any(not np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError("computed study statistic is non-finite")
    return result


def point_statistics(values: Sequence[object]) -> dict[str, float]:
    return {
        key: float(value) for key, value in _metrics(complete_vector(values)).items()
    }


def _interval(samples: Vector) -> list[float]:
    if samples.shape != (BOOTSTRAP_REPLICATES,) or not np.all(np.isfinite(samples)):
        raise ValueError("bootstrap samples differ from the finite frozen contract")
    return [
        float(value) for value in np.quantile(samples, [0.025, 0.975], method="linear")
    ]


def _rng() -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))


def policy_summary(values: Sequence[object]) -> dict[str, Any]:
    vector = complete_vector(values)
    statistics = point_statistics(values)
    indices = _rng().integers(
        0, EPISODE_COUNT, size=(BOOTSTRAP_REPLICATES, EPISODE_COUNT)
    )
    sampled = vector[indices]
    bootstrapped = _metrics(sampled)
    positive = int(np.count_nonzero(vector > 0))
    negative = int(np.count_nonzero(vector < 0))
    profiles = []
    for threshold in PROFILE_THRESHOLDS_EUR:
        profiles.append(
            {
                "threshold_eur": threshold,
                **binomial(int(np.count_nonzero(vector > threshold)), EPISODE_COUNT),
                "bootstrap_95": _interval(np.mean(sampled > threshold, axis=-1)),
            }
        )
    return {
        "schema": PREFIX + "paired-statistics-v1",
        "analysis_contract_sha256": analysis_contract()["sha256"],
        "unit": "paired delta EUR",
        "sample_count": EPISODE_COUNT,
        "statistics": statistics,
        "sign_counts": {
            "positive": positive,
            "zero": EPISODE_COUNT - positive - negative,
            "negative": negative,
        },
        "probability_of_improvement": binomial(positive, EPISODE_COUNT),
        "bootstrap_95": {
            key: _interval(sample) for key, sample in bootstrapped.items()
        },
        "performance_profile": profiles,
    }


def family_vectors(values: NullableFamily) -> dict[int, Vector | None]:
    if any(type(seed) is not int for seed in values) or set(values) != set(
        BETA_OPTIMIZER_SEEDS
    ):
        raise ValueError("a family must retain all five exact optimizer identifiers")
    result: dict[int, Vector | None] = {}
    for seed in BETA_OPTIMIZER_SEEDS:
        vector = values[seed]
        result[seed] = None if vector is None else complete_vector(vector)
    return result


def family_summary(values: NullableFamily) -> dict[str, Any]:
    vectors = family_vectors(values)
    seeds = [seed for seed, value in vectors.items() if value is not None]
    result: dict[str, Any] = {
        "schema": PREFIX + "family-statistics-v1",
        "analysis_contract_sha256": analysis_contract()["sha256"],
        "unit": "paired delta EUR",
        "estimand": "equal-policy mean of 32-episode statistics, conditional on complete policies",
        "optimizer_count": 5,
        "contributing_optimizer_seeds": seeds,
        "missing_optimizer_seeds": [
            seed for seed in BETA_OPTIMIZER_SEEDS if seed not in seeds
        ],
        "policy_admission": binomial(len(seeds), 5),
        "statistics": None,
        "observed_pair_extrema": None,
        "pair_sign_counts": None,
        "bootstrap_95": None,
        "performance_profile": None,
    }
    if not seeds:
        return result
    matrix = np.stack([vector for vector in vectors.values() if vector is not None])
    generator = _rng()
    optimizer_indices = generator.integers(
        0, len(seeds), size=(BOOTSTRAP_REPLICATES, len(seeds))
    )
    episode_indices = generator.integers(
        0, EPISODE_COUNT, size=(BOOTSTRAP_REPLICATES, EPISODE_COUNT)
    )
    sampled = matrix[optimizer_indices[:, :, None], episode_indices[:, None, :]]
    metrics = _metrics(matrix)
    bootstrap_metrics = _metrics(sampled)
    result["statistics"] = {
        key: finite_real(np.mean(value)) for key, value in metrics.items()
    }
    result["observed_pair_extrema"] = {
        "minimum": float(np.min(matrix)),
        "maximum": float(np.max(matrix)),
    }
    result["pair_sign_counts"] = {
        "positive": int(np.count_nonzero(matrix > 0)),
        "zero": int(np.count_nonzero(matrix == 0)),
        "negative": int(np.count_nonzero(matrix < 0)),
        "interpretation": "descriptive paired cells; not independent binomial trials",
    }
    result["bootstrap_95"] = {
        key: _interval(np.mean(value, axis=1))
        for key, value in bootstrap_metrics.items()
    }
    result["performance_profile"] = [
        {
            "threshold_eur": threshold,
            "rate": float(np.mean(matrix > threshold)),
            "bootstrap_95": _interval(np.mean(sampled > threshold, axis=(1, 2))),
        }
        for threshold in PROFILE_THRESHOLDS_EUR
    ]
    return result


def matched_family_summary(
    policy: NullableFamily, scalar: NullableFamily
) -> dict[str, Any]:
    left, right = family_vectors(policy), family_vectors(scalar)
    differences: dict[int, Sequence[object] | None] = {}
    for seed in BETA_OPTIMIZER_SEEDS:
        p, s = left[seed], right[seed]
        if p is None or s is None:
            differences[seed] = None
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                differences[seed] = (p - s).tolist()
    return {
        "schema": PREFIX + "matched-family-statistics-v1",
        "orientation": "policy minus scalar PPO; identical optimizer and episode IDs",
        "statistics": family_summary(differences),
    }

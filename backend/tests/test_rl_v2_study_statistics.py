"""Independent numeric oracles; no environment or protected data is accessed."""

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from study_test_helpers import forbid_execution  # noqa: F401
from plant_backend.rl.v2.study.contract import (
    BETA_OPTIMIZER_SEEDS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    PROFILE_THRESHOLDS_EUR,
    analysis_contract,
)
from plant_backend.rl.v2.study.statistics import (
    binomial,
    family_summary,
    matched_family_summary,
    point_statistics,
    policy_summary,
)


def test_analysis_contract_matches_committed_upstream_and_frozen_choices() -> None:
    contract = analysis_contract()
    assert (
        contract["sha256"]
        == "94c4ce4b665ab31228b124f9fe056c98fb37ab5b349a12f785c2368b4d095d72"
    )
    protocol = (
        Path(__file__).resolve().parents[2]
        / "docs/research/CONSTRAINED_RL_BETA_PROTOCOL.md"
    )
    assert (
        hashlib.sha256(protocol.read_bytes()).hexdigest()
        == contract["protocol"]["raw_sha256"]
    )
    assert contract["optimizer_seeds"] == [101, 211, 307, 401, 503]
    assert len(contract["family_order"]) == 3
    assert contract["bootstrap"]["replicates"] == 10000
    contract["profile"]["thresholds"].append(123)
    assert analysis_contract()["profile"]["thresholds"] == list(PROFILE_THRESHOLDS_EUR)


def test_closed_form_32_pair_statistics_and_exact_zero() -> None:
    values = list(range(32))
    result = policy_summary(values)
    assert result["sample_count"] == 32
    assert result["statistics"] == {
        "mean": 15.5,
        "sample_std": math.sqrt(88),
        "median": 15.5,
        "iqm": 15.5,
        "cvar20": 3.0,
        "minimum": 0.0,
        "maximum": 31.0,
        "positive_fraction": 31 / 32,
    }
    assert result["sign_counts"] == {"negative": 0, "zero": 1, "positive": 31}
    assert result["probability_of_improvement"]["successes"] == 31
    profile_zero = next(
        p for p in result["performance_profile"] if p["threshold_eur"] == 0
    )
    assert profile_zero["rate"] == 31 / 32
    assert values == list(range(32))


def test_iqm_preserves_duplicate_sample_positions_and_cvar_rounds_up() -> None:
    values = [-10.0] * 7 + [0.0] * 9 + [2.0] * 8 + [100.0] * 8
    result = point_statistics(values)
    assert result["iqm"] == 1.0
    assert result["cvar20"] == -10.0
    assert result["median"] == 1.0


@pytest.mark.parametrize(
    "bad", [None, True, np.bool_(False), "1", float("nan"), float("inf"), -float("inf")]
)
def test_missing_or_nonfinite_pair_never_enters_an_aggregate(bad: object) -> None:
    with pytest.raises(ValueError):
        policy_summary([1.0] * 31 + [bad])


@pytest.mark.parametrize("count", [0, 1, 7, 16, 31, 33, 160])
def test_policy_aggregation_cannot_score_a_surviving_subset(count: int) -> None:
    with pytest.raises(ValueError, match="32"):
        policy_summary([1.0] * count)


@pytest.mark.parametrize(
    "values",
    ["1" * 32, b"1" * 32, {i: 999 for i in range(32)}, [10**400] * 32, [1e308] * 32],
)
def test_non_sequences_and_float64_overflow_cannot_be_scored(values: object) -> None:
    with pytest.raises(ValueError):
        policy_summary(values)  # type: ignore[arg-type]


def _linear_interval(values: list[float]) -> list[float]:
    ordered = sorted(values)
    result = []
    for quantile in (0.025, 0.975):
        position = (len(ordered) - 1) * quantile
        low = math.floor(position)
        fraction = position - low
        result.append(ordered[low] * (1 - fraction) + ordered[low + 1] * fraction)
    return result


def test_bootstrap_iqm_matches_independent_sort_fsum_and_linear_endpoint_oracle() -> (
    None
):
    values = np.array([-100.0] * 8 + list(range(-8, 8)) + [100.0] * 8)
    generator = np.random.Generator(np.random.PCG64(20260905))
    draws = generator.integers(0, 32, size=(10000, 32))
    iqms = [math.fsum(sorted(values[row].tolist())[8:24]) / 16 for row in draws]
    expected = _linear_interval(iqms)
    result = policy_summary(values.tolist())
    assert result["bootstrap_95"]["iqm"] == pytest.approx(expected, abs=1e-12)
    assert result == policy_summary(values.tolist())


def test_wilson_known_cases_include_empty_and_exact_endpoints() -> None:
    assert binomial(0, 0) == {
        "successes": 0,
        "total": 0,
        "rate": None,
        "wilson_95": None,
    }
    assert binomial(16, 32)["wilson_95"] == pytest.approx(
        [0.336308828693274, 0.663691171306726]
    )
    assert binomial(0, 32)["wilson_95"][0] == 0
    assert binomial(32, 32)["wilson_95"][1] == 1
    assert binomial(0, 32)["wilson_95"][1] == pytest.approx(0.1071791982550706)


@pytest.mark.parametrize(
    "counts", [(True, 32), (1, False), (-1, 32), (33, 32), (1.0, 32), (1, 0)]
)
def test_binomial_counts_cannot_be_coerced(counts: tuple[object, object]) -> None:
    with pytest.raises(ValueError):
        binomial(*counts)  # type: ignore[arg-type]


def test_family_missingness_keeps_the_denominator_of_five() -> None:
    values = {seed: None for seed in BETA_OPTIMIZER_SEEDS}
    empty = family_summary(values)
    assert empty["optimizer_count"] == 5
    assert empty["policy_admission"]["total"] == 5
    assert empty["statistics"] is None and empty["bootstrap_95"] is None
    one = family_summary({**values, 307: [2.0] * 32})
    assert one["contributing_optimizer_seeds"] == [307]
    assert one["policy_admission"]["rate"] == 0.2
    assert one["missing_optimizer_seeds"] == [101, 211, 401, 503]
    assert one["statistics"]["iqm"] == 2
    assert one["bootstrap_95"]["iqm"] == [2.0, 2.0]
    assert all("wilson_95" not in profile for profile in one["performance_profile"])


def test_hierarchical_resampling_keeps_the_common_episode_cluster() -> None:
    values = list(range(32))
    result = family_summary({seed: values for seed in BETA_OPTIMIZER_SEEDS})
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    generator.integers(0, 5, size=(BOOTSTRAP_REPLICATES, 5))
    episodes = generator.integers(0, 32, size=(BOOTSTRAP_REPLICATES, 32))
    iqms = [math.fsum(sorted(row.tolist())[8:24]) / 16 for row in episodes]
    assert result["bootstrap_95"]["iqm"] == pytest.approx(_linear_interval(iqms))
    assert result["statistics"]["iqm"] == 15.5
    assert result["pair_sign_counts"]["zero"] == 5


def test_optimizer_resampling_retains_between_policy_variation() -> None:
    constants = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    result = family_summary(
        {
            seed: [float(c)] * 32
            for seed, c in zip(BETA_OPTIMIZER_SEEDS, constants, strict=True)
        }
    )
    indices = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, 5, size=(10000, 5)
    )
    means = [math.fsum(constants[row]) / 5 for row in indices]
    assert result["statistics"]["iqm"] == 20.0
    assert result["bootstrap_95"]["iqm"] == pytest.approx(_linear_interval(means))
    assert result["bootstrap_95"]["iqm"][0] < 20 < result["bootstrap_95"]["iqm"][1]


def test_scalar_family_contrast_matches_identical_optimizer_ids_only() -> None:
    scalar = {seed: [float(seed)] * 32 for seed in BETA_OPTIMIZER_SEEDS}
    policy = {seed: [float(seed) + 1] * 32 for seed in BETA_OPTIMIZER_SEEDS}
    result = matched_family_summary(policy, scalar)["statistics"]
    assert result["statistics"]["iqm"] == 1
    assert result["bootstrap_95"]["iqm"] == [1, 1]
    missing = matched_family_summary({**policy, 211: None}, scalar)["statistics"]
    assert missing["missing_optimizer_seeds"] == [211]
    assert missing["policy_admission"]["total"] == 5


@pytest.mark.parametrize(
    "values",
    [{}, {101: [1] * 32}, {True: [1] * 32, 211: None, 307: None, 401: None, 503: None}],
)
def test_family_identifiers_cannot_be_omitted_or_replaced(
    values: dict[object, object]
) -> None:
    with pytest.raises(ValueError):
        family_summary(values)  # type: ignore[arg-type]

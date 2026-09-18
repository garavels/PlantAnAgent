"""Every frozen tie-break and missingness boundary uses explicit synthetic values."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

from study_test_helpers import forbid_execution  # noqa: F401
from plant_backend.rl.v2.study.contract import (
    BETA_CORE_ALGORITHM_IDS,
    BETA_OPTIMIZER_SEEDS,
)
from plant_backend.rl.v2.study.selection import (
    DevelopmentPolicy,
    select_family,
    verify_selection_result,
)


def _policy(
    values: list[float | None], *, trained: bool = True, verified: bool = True
) -> DevelopmentPolicy:
    return DevelopmentPolicy(trained, verified, tuple(values))


def _family() -> dict[int, DevelopmentPolicy]:
    return {
        seed: _policy([None] * 32, trained=False, verified=False)
        for seed in BETA_OPTIMIZER_SEEDS
    }


@pytest.mark.parametrize("family", BETA_CORE_ALGORITHM_IDS)
def test_all_five_failed_policies_remain_visible_with_no_champion(family: str) -> None:
    result = select_family(family, _family())
    assert result["champion_optimizer_seed"] is None
    assert [row["optimizer_seed"] for row in result["policies"]] == [
        101,
        211,
        307,
        401,
        503,
    ]
    assert result["development_admission"]["total"] == 5
    assert result["development_admission"]["successes"] == 0
    assert all(row["selection_statistics"] is None for row in result["policies"])
    assert verify_selection_result(result) == {
        "verification": "selection_reconstructed",
        "source_evidence_verified": False,
        "result_claim": None,
    }


@pytest.mark.parametrize(
    "component,lower,higher",
    [
        ("iqm", [0.0] * 32, [1.0] * 32),
        ("cvar20", [-2.0] * 7 + [1.0] * 25, [-1.0] * 7 + [1.0] * 25),
        (
            "median",
            [-1.0] * 8 + [2.0] * 16 + [10.0] * 8,
            [-1.0] * 8 + [0.0] * 7 + [3.0] * 8 + [8.0] + [10.0] * 8,
        ),
        (
            "positive_fraction",
            [-2.0] * 7 + [0.0] + [2.0] * 16 + [10.0] * 8,
            [-2.0] * 7 + [1.0] + [2.0] * 16 + [10.0] * 8,
        ),
    ],
)
def test_each_lexicographic_component_precedes_optimizer_seed(
    component: str,
    lower: list[float],
    higher: list[float],
) -> None:
    policies = {**_family(), 101: _policy(lower), 503: _policy(higher)}
    result = select_family(BETA_CORE_ALGORITHM_IDS[0], policies)
    first, last = result["policies"][0], result["policies"][-1]
    key_order = ["iqm", "cvar20", "median", "positive_fraction"]
    position = key_order.index(component)
    for earlier in key_order[:position]:
        assert (
            first["selection_statistics"][earlier]
            == last["selection_statistics"][earlier]
        )
    assert (
        first["selection_statistics"][component]
        < last["selection_statistics"][component]
    )
    assert result["champion_optimizer_seed"] == 503
    assert sum(row["selected"] for row in result["policies"]) == 1


def test_lower_optimizer_seed_breaks_an_exact_four_statistic_tie() -> None:
    policies = {seed: _policy([0.0] * 32) for seed in reversed(BETA_OPTIMIZER_SEEDS)}
    result = select_family(BETA_CORE_ALGORITHM_IDS[0], policies)
    assert result["champion_optimizer_seed"] == 101
    assert result["development_admission"]["rate"] == 1
    assert result["policies"][0]["selection_statistics"]["positive_fraction"] == 0


@pytest.mark.parametrize("defect", ["one_missing", "training_failed", "unverified"])
def test_incomplete_or_unadmitted_policy_cannot_win_with_large_observed_values(
    defect: str,
) -> None:
    dominant = _policy([1_000_000.0] * 32)
    if defect == "one_missing":
        dominant = replace(dominant, paired_deltas_eur=(None,) + (1_000_000.0,) * 31)
    elif defect == "training_failed":
        dominant = replace(dominant, training_admitted=False)
    else:
        dominant = replace(dominant, development_report_verified=False)
    result = select_family(
        BETA_CORE_ALGORITHM_IDS[0],
        {**_family(), 101: dominant, 211: _policy([-1.0] * 32)},
    )
    excluded = result["policies"][0]
    assert result["champion_optimizer_seed"] == 211
    assert (
        excluded["selection_statistics"] is None and excluded["selection_key"] is None
    )
    assert excluded["development_admissible"] is False
    assert result["development_admission"]["successes"] == 1
    assert result["development_admission"]["total"] == 5


@pytest.mark.parametrize(
    "part",
    [
        "champion",
        "score",
        "admission",
        "selected",
        "reorder",
        "extra_field",
        "missing_flag",
        "integer_flag",
        "family",
    ],
)
def test_selection_reconstruction_rejects_result_tampering(part: str) -> None:
    raw = deepcopy(
        select_family(
            BETA_CORE_ALGORITHM_IDS[0],
            {seed: _policy([float(seed)] * 32) for seed in BETA_OPTIMIZER_SEEDS},
        )
    )
    if part == "champion":
        raw["champion_optimizer_seed"] = 101
    elif part == "score":
        raw["policies"][0]["selection_statistics"]["iqm"] += 10
    elif part == "admission":
        raw["development_admission"]["successes"] = 4
    elif part == "selected":
        raw["policies"][0]["selected"] = True
    elif part == "reorder":
        raw["policies"] = list(reversed(raw["policies"]))
    elif part == "extra_field":
        raw["latency_selection_bonus"] = 999
    elif part == "missing_flag":
        del raw["policies"][0]["training_admitted"]
    elif part == "integer_flag":
        raw["policies"][0]["training_admitted"] = 1
    else:
        raw["family"] = None
    with pytest.raises(ValueError):
        verify_selection_result(raw)


@pytest.mark.parametrize(
    "values", [[0.0] * 31, [0.0] * 33, [float("nan")] * 32, [True] * 32]
)
def test_selection_input_must_have_exact_finite_pair_slots(values: list[Any]) -> None:
    with pytest.raises(ValueError):
        _policy(values)


def test_unknown_family_or_optimizer_matrix_cannot_be_substituted() -> None:
    with pytest.raises(ValueError):
        select_family("replacement-family", _family())
    with pytest.raises(ValueError):
        select_family(BETA_CORE_ALGORITHM_IDS[0], {101: _policy([0.0] * 32)})

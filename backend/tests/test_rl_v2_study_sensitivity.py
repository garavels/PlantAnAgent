"""Explicit additive arithmetic and corruption checks; controllers cannot run."""

from copy import deepcopy
import math
from typing import Any

import pytest

from study_test_helpers import forbid_execution  # noqa: F401
from plant_backend.rl.v2.study.contract import COMPONENT_NAMES
from plant_backend.rl.v2.study.sensitivity import (
    rescore_pair,
    verify_sensitivity_result,
)
from plant_backend.rl.v2.training.custody_json import authenticate


def _ledger(**values: float) -> dict[str, float]:
    return {name: values.get(name, 0.0) for name in COMPONENT_NAMES}


def test_sensitivity_retains_the_existing_component_contract() -> None:
    from plant_backend.rl.v2.economic_robustness import ECONOMIC_COMPONENT_NAMES

    assert COMPONENT_NAMES == ECONOMIC_COMPONENT_NAMES


def test_every_setting_is_independently_recomputed_as_policy_minus_e_beta() -> None:
    policy = _ledger(
        delivered_service_value=100,
        unmet_delivery_penalty=-20,
        solar_opportunity_cost=-7,
        battery_throughput_reserve=-3,
        electrolyser_life_reserve=-4,
        methane_endowment_liability_change=-10,
        battery_endowment_liability_change=-2,
    )
    reference = _ledger(
        delivered_service_value=90,
        unmet_delivery_penalty=-25,
        solar_opportunity_cost=-5,
        battery_throughput_reserve=-1,
        electrolyser_life_reserve=-2,
        methane_endowment_liability_change=-4,
        battery_endowment_liability_change=-1,
    )
    result = rescore_pair(policy, reference)
    assert result["orientation"] == "policy minus E-beta"
    assert len(result["settings"]) == 16
    for setting in result["settings"]:
        expected = math.fsum(
            policy[name] * setting["weights"][name] for name in COMPONENT_NAMES
        ) - math.fsum(
            reference[name] * setting["weights"][name] for name in COMPONENT_NAMES
        )
        assert setting["delta_eur"] == pytest.approx(expected, abs=1e-12)
    settings = {row["name"]: row for row in result["settings"]}
    assert settings["default"]["delta_eur"] == 2
    assert settings["methane_endowment.0"]["delta_eur"] == 8
    assert settings["methane_endowment.0"]["weights"]["unmet_delivery_penalty"] == 1
    assert settings["declared_wear.0"]["delta_eur"] == 6
    assert settings["solar_opportunity.0"]["delta_eur"] == 4
    assert settings["service_plus_solar_only"]["delta_eur"] == 13
    assert verify_sensitivity_result(result) == {
        "verification": "sensitivity_reconstructed",
        "source_evidence_verified": False,
        "result_claim": None,
    }


def test_reversing_the_pair_reverses_every_delta_and_keeps_all_rank_changes() -> None:
    policy = _ledger(delivered_service_value=2, methane_endowment_liability_change=-3)
    other = _ledger()
    forward = rescore_pair(policy, other)
    backward = rescore_pair(other, policy)
    for f, b in zip(forward["settings"], backward["settings"], strict=True):
        assert f["delta_eur"] == -b["delta_eur"]
    assert "methane_endowment.0" in forward["ranking_reversals_from_default"]
    assert (
        forward["ranking_reversals_from_default"]
        == backward["ranking_reversals_from_default"]
    )


def test_exact_zero_and_tiny_nonzero_are_never_hidden_by_a_sign_tolerance() -> None:
    result = rescore_pair(
        _ledger(delivered_service_value=1, methane_endowment_liability_change=-1),
        _ledger(),
    )
    assert result["settings"][0]["higher_arm"] == "tie"
    assert "methane_endowment.0" in result["ranking_reversals_from_default"]
    assert "methane_endowment.2" in result["ranking_reversals_from_default"]
    tiny = rescore_pair(_ledger(delivered_service_value=1e-12), _ledger())
    assert tiny["settings"][0]["higher_arm"] == "policy_higher"


@pytest.mark.parametrize(
    "part",
    [
        "orientation",
        "coefficient",
        "delta",
        "higher_arm",
        "reversals",
        "unknown",
        "missing_setting",
        "input_extra",
        "schema",
        "contract",
    ],
)
def test_rehashed_sensitivity_result_cannot_change_the_fixed_arithmetic(
    part: str,
) -> None:
    raw = deepcopy(rescore_pair(_ledger(delivered_service_value=10), _ledger()))
    if part == "orientation":
        raw["orientation"] = "E-beta minus policy"
    elif part == "coefficient":
        raw["settings"][1]["weights"]["unmet_delivery_penalty"] = 0
    elif part == "delta":
        raw["settings"][0]["delta_eur"] += 1
    elif part == "higher_arm":
        raw["settings"][0]["higher_arm"] = "E-beta_higher"
    elif part == "reversals":
        raw["ranking_reversals_from_default"] = ["default"]
    elif part == "unknown":
        raw["fitted_coefficient"] = 1
    elif part == "missing_setting":
        raw["settings"].pop()
    elif part == "input_extra":
        raw["inputs"]["unlisted"] = {}
    elif part == "schema":
        raw["schema"] = "M-minus-H"
    else:
        raw["analysis_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        verify_sensitivity_result(authenticate(raw))


@pytest.mark.parametrize("bad", [None, True, "1", float("nan"), float("inf")])
def test_sensitivity_rejects_non_numeric_or_nonfinite_components(bad: Any) -> None:
    with pytest.raises(ValueError):
        rescore_pair(_ledger(delivered_service_value=bad), _ledger())


def test_sensitivity_requires_all_components_and_rejects_arithmetic_overflow() -> None:
    missing = _ledger()
    del missing["unmet_delivery_penalty"]
    with pytest.raises(ValueError):
        rescore_pair(missing, _ledger())
    with pytest.raises(ValueError):
        rescore_pair({**_ledger(), "extra": 0}, _ledger())
    with pytest.raises(ValueError):
        rescore_pair(
            _ledger(delivered_service_value=1e308),
            _ledger(delivered_service_value=-1e308),
        )

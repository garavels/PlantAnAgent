"""Fabricated report fixtures; no live outcomes or environment resets."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from plant_backend.rl.v2.metrics import EpisodeMetricsV2

spec = importlib.util.spec_from_file_location(
    "adaptive_report", str(Path(__file__).with_name("plant-adaptive-report.py"))
)
assert spec is not None and spec.loader is not None
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def packet():
    physical = EpisodeMetricsV2(
        requested_kg=6720.0,
        delivered_kg=6000.0,
        unmet_kg=720.0,
        commitments_settled=28,
        commitments_met=24,
    ).to_dict()
    summary = {
        "execution_complete": True,
        "physical_metrics": physical,
        "economic_metrics": {
            "net_operating_value_eur": -10000.0,
            "components_eur": {"delivered_service": 2000.0, "operating": -12000.0},
        },
        "hard_safety_passed": True,
        "all_cycles_feasible": True,
        "resource_order": list(reporter.RESOURCES),
        "cycle_deficit_vectors": [[0.0] * 4 for _ in range(7)],
        "fallback_decisions": 0,
        "recovery_decisions": 0,
    }
    actors = {
        name: {
            "slots": [
                {
                    "seed": seed,
                    "summary": deepcopy(summary),
                    "eligible_for_descriptive_comparison": True,
                    "failures": [],
                }
                for seed in reporter.SEEDS
            ]
        }
        for name in (*reporter.ARMS, *reporter.COMPARATORS)
    }
    training = {arm: fake_training_data() for arm in reporter.ARMS}
    return {
        "integrity_verified": True,
        "batch_closed": True,
        "actors": actors,
        "training": training,
    }


def test_exact_paired_signs_and_no_false_positive_claim():
    p = packet()
    for slot in p["actors"]["A-pi-raw"]["slots"]:
        slot["summary"]["economic_metrics"]["net_operating_value_eur"] += 2000
        slot["summary"]["physical_metrics"]["delivered_kg"] -= 100
        slot["summary"]["physical_metrics"]["commitments_met"] -= 1
    p["actors"]["A-pi-raw"]["slots"][0]["summary"]["cycle_deficit_vectors"][2][1] = 3.0
    result = reporter.make_report(p)
    pair = next(
        x
        for x in result["operational_comparisons"]
        if x["candidate"] == "A-pi-raw" and x["comparator"] == "OperatingMPC"
    )
    assert pair["paired_differences"]["net_operating_value_eur"]["mean"] == 2000
    assert pair["paired_differences"]["delivered_kg"]["minimum"] == -100
    assert pair["paired_differences"]["commitments_met"]["minimum"] == -1
    assert len(pair["daily_resource_differences"]["all_56_boundaries"]) == 56
    assert pair["daily_resource_differences"]["each_resource"]["h2"]["maximum"] == 3
    assert pair["daily_resource_differences"]["each_resource"]["co2"]["maximum"] == 0
    assert pair["confirmation_or_material_benefit_claim"] is False


def test_stopped_baseline_has_no_subset_average_or_numeric_win():
    p = packet()
    slot = p["actors"]["E-beta"]["slots"][4]
    slot["eligible_for_descriptive_comparison"] = False
    slot["summary"]["execution_complete"] = False
    slot["failures"] = ["opening-infeasibility"]
    result = reporter.make_report(p)
    baseline = result["actors"]["E-beta"]
    assert baseline["aggregate"] is None
    assert baseline["hard_safety_pass_count"] == 7
    assert baseline["all_daily_returns_pass_count"] == 7
    assert baseline["slots"][4]["failures"] == ["opening-infeasibility"]
    for pair in result["operational_comparisons"]:
        if pair["comparator"] == "E-beta":
            assert pair["paired_differences"] is None
            assert pair["unavailable_seeds"] == [reporter.SEEDS[4]]
    assert "unavailable" in reporter.markdown(result)


@pytest.mark.parametrize(
    "change",
    [
        "unclosed",
        "unaudited",
        "missing_actor",
        "missing_seed",
        "resource_order",
        "nonfinite",
    ],
)
def test_incomplete_or_inconsistent_packet_refused(change):
    p = packet()
    if change == "unclosed":
        p["batch_closed"] = False
    elif change == "unaudited":
        p["integrity_verified"] = False
    elif change == "missing_actor":
        del p["actors"]["H"]
    elif change == "missing_seed":
        p["actors"]["H"]["slots"].pop()
    elif change == "resource_order":
        p["actors"]["H"]["slots"][0]["summary"]["resource_order"].reverse()
    else:
        p["actors"]["H"]["slots"][0]["summary"]["economic_metrics"][
            "net_operating_value_eur"
        ] = float("nan")
    with pytest.raises(ValueError):
        reporter.make_report(p)


def fake_training_data():
    gradient = {
        "reward_critic_preclip_gradient_norm": 10.0,
        "actor_total_preclip_gradient_norm": 2.0,
        "implied_common_clip_factor": 0.05,
        "reward_value_mse": 4.0,
        "weighted_cost_actor_gradient_norm": None,
        "reward_actor_gradient_norm": 2.0,
        "reward_vs_weighted_cost_gradient_cosine": None,
        "cost_actor_components": None,
    }
    sample = {
        "reward_critic_tanh": [],
        "costs": None,
        "gaussian": {
            "mean_by_action": [{"mean": 0.0}] * 4,
            "std_by_action": [{"mean": 1.0}] * 4,
            "expected_clipped_fraction_all_action_coordinates": 0.317,
            "observed_clipped_fraction_all_action_coordinates": 0.3,
        },
    }
    diagnostics = [
        {
            "num_timesteps": n,
            "gradient_attribution": deepcopy(gradient),
            "sample": deepcopy(sample),
            "full_rollout": {"reward_explained_variance": None, "costs": None},
        }
        for n in range(8192, 131073, 8192)
    ]
    data = {
        "closed_successfully": True,
        "result": {"arm": {"reward_scale": 0.001}},
        "diagnostics": diagnostics,
    }
    return data


def test_scaled_mse_converted_only_to_units_and_undefined_costs_remain_null():
    data = fake_training_data()
    result = reporter.training_report("D-scalar-scaled", data)
    first = result["diagnostic_rows"][0]
    assert first["critic_mse_eur_squared"] == 4_000_000
    assert first["diagnostic_actor_norm_after_common_clip"] == 0.1
    assert first["costs"] is None and first["reward_explained_variance"] is None
    data["diagnostics"].pop()
    with pytest.raises(ValueError, match="sixteen"):
        reporter.training_report("D-scalar-scaled", data)


def test_empty_or_bool_metrics_do_not_become_zero_or_one():
    with pytest.raises(ValueError):
        reporter.describe([])
    with pytest.raises(ValueError):
        reporter.numeric(True)


def test_failed_training_cannot_supply_an_eligible_comparison():
    p = packet()
    p["training"]["A-pi-raw"] = {
        "closed_successfully": False,
        "result": None,
        "diagnostics": [],
        "failures": ["timeout"],
    }
    with pytest.raises(ValueError, match="failed training closure"):
        reporter.make_report(p)
    for actor in ("A-pi-raw", "initial"):
        for slot in p["actors"][actor]["slots"]:
            slot.update(
                summary=None,
                eligible_for_descriptive_comparison=False,
                failures=["training unavailable"],
            )
    result = reporter.make_report(p)
    assert result["actors"]["A-pi-raw"]["aggregate"] is None
    assert len(result["actors"]["A-pi-raw"]["slots"]) == 8
    assert result["training"]["A-pi-raw"]["failure_evidence"]["failures"] == ["timeout"]

"""Prevent selection, pseudoreplication, and optimistic missing-outcome treatment."""
from copy import deepcopy

import pytest

from docs.research.tools.tail_stress_v1 import analyze as a


def make_row(seed, condition, controller, unmet=0.0, contribution=1000.0):
    return {"seed": seed, "condition": condition, "controller": controller,
            "execution_complete": True, "sources_unchanged": True, "failure": None,
            "every_delivery_passed": unmet == 0,
            "economic_metrics": {"unmet_kg": unmet, "net_operating_value_eur": contribution},
            "delivery": {"settlements": [{"unmet_kg": unmet}] + [{"unmet_kg": 0.0} for _ in range(27)]},
            "minute_audit": {"all_physical_invariants_passed": True},
            "economic_audit": {"passed": True}, "physical_metrics": {"guard_interventions": 0}}


def grid():
    return [make_row(s, c, p) for s in a.SEEDS for c in a.CONDITIONS for p in a.CONTROLLERS]


def test_exact_grid_retains_each_policy_and_rejects_selection():
    rows = grid()
    assert len(a.validate_rows(rows)) == 480
    with pytest.raises(ValueError, match="Incomplete prescribed grid"):
        a.validate_rows(rows[:-1])
    with pytest.raises(ValueError, match="Duplicate prescribed case"):
        a.validate_rows(rows + [rows[0]])
    altered = deepcopy(rows)
    altered[0]["seed"] = 42_000_000
    with pytest.raises(ValueError, match="Unprescribed case"):
        a.validate_rows(altered)


def test_cluster_bootstrap_preserves_seed_dependence():
    single, repeated = {}, {}
    for i, seed in enumerate(a.SEEDS):
        left = make_row(seed, "pv6", "daylight", unmet=0)
        right = make_row(seed, "pv6", "ppo-101", unmet=100 if i % 2 else 0)
        single[(seed, "pv6")] = (left, right)
        for condition in a.STRESS_CONDITIONS:
            repeated[(seed, condition)] = (left, right)
    one = a.paired_cluster_estimate(single, "unmet_kg")
    five = a.paired_cluster_estimate(repeated, "unmet_kg")
    assert one["interval"] == five["interval"]
    assert one["mean"] == five["mean"] == 50
    assert five["n_paired_cells"] == 80
    assert five["n_base_scenarios"] == 16


def test_primary_adjustment_and_criterion_do_not_select_best_seed():
    rows = grid()
    for row in rows:
        if row["condition"] != "control" and row["controller"] == "daylight":
            row.update(make_row(row["seed"], row["condition"], row["controller"], unmet=60))
    result = a.analyze_rows(rows)
    assert len(result["groups"]) == 30
    assert len(result["primary_claims"]) == 6
    assert len(result["pooled_stress_groups"]) == 5
    assert {p["treatment"] for p in result["primary_claims"]} == {"ppo-101", "ppo-211", "ppo-307"}
    for claim in result["primary_claims"]:
        criterion = claim["primary_criterion"]
        assert criterion["adjusted_success_difference"]["confidence"] == pytest.approx(1 - .05 / 6)
        assert criterion["met"] == (claim["reference"] == "daylight")
        assert claim["rescues"] == (80 if claim["reference"] == "daylight" else 0)


def test_a_single_regression_blocks_primary_claim_even_with_many_rescues():
    rows = grid()
    for row in rows:
        if row["condition"] != "control" and row["controller"] == "daylight":
            row.update(make_row(row["seed"], row["condition"], row["controller"], unmet=60))
    for row in rows:
        if row["seed"] == a.SEEDS[0] and row["condition"] == "pv6":
            if row["controller"] == "daylight":
                row.update(make_row(row["seed"], row["condition"], row["controller"]))
            elif row["controller"] == "ppo-101":
                row.update(make_row(row["seed"], row["condition"], row["controller"], unmet=1))
    index = a.validate_rows(rows)
    claim = a.paired_comparison(index, reference="daylight", treatment="ppo-101", conditions=a.STRESS_CONDITIONS, label="pooled-stress")
    assert claim["rescues"] == 79 and claim["regressions"] == 1
    assert claim["primary_criterion"]["no_greater_mean_unmet_kg"]
    assert not claim["primary_criterion"]["met"]


def test_execution_failure_is_retained_and_invalidates_full_panel_estimate():
    rows = grid()
    target = next(r for r in rows if (r["condition"], r["controller"]) == ("pv6", "ppo-101"))
    target.update(execution_complete=False, failure={"type": "test failure"})
    index = a.validate_rows(rows)
    claim = a.paired_comparison(index, reference="daylight", treatment="ppo-101", conditions=a.STRESS_CONDITIONS, label="pooled-stress")
    assert claim["n_paired_cells"] == 80
    assert not claim["all_comparisons_complete_and_audited"]
    assert claim["metrics"]["unmet_kg"]["mean"] is None
    assert not claim["primary_criterion"]["met"]


def test_audit_failure_cannot_pass_claim():
    rows = grid()
    for row in rows:
        if row["condition"] != "control" and row["controller"] == "daylight":
            row.update(make_row(row["seed"], row["condition"], row["controller"], unmet=60))
    next(r for r in rows if (r["condition"], r["controller"]) == ("pv6", "ppo-101"))["economic_audit"]["passed"] = False
    claim = a.paired_comparison(a.validate_rows(rows), reference="daylight", treatment="ppo-101", conditions=a.STRESS_CONDITIONS, label="pooled-stress")
    assert claim["rescues"] == 80
    assert not claim["primary_criterion"]["met"]


def test_quarter_loss_and_pairing_checks():
    assert a.worst_quarter([0]*12 + [10, 20, 30, 40])["mean_unmet_kg"] == 25
    assert a.worst_quarter([0, None])["mean_unmet_kg"] is None
    rows = grid()
    for row in rows:
        row["physical_solar_forcing_sha256"] = f"{row['seed']}-{row['condition']}"
    rows[0]["physical_solar_forcing_sha256"] = "different forcing"
    with pytest.raises(ValueError, match="Paired controller forcing differs"):
        a.validate_rows(rows)


def test_delivery_classification_and_unmet_reconcile():
    rows = grid()
    rows[0]["every_delivery_passed"] = False
    with pytest.raises(ValueError, match="Delivery classification differs"):
        a.validate_rows(rows)
    rows = grid()
    rows[0]["economic_metrics"]["unmet_kg"] = 1
    with pytest.raises(ValueError, match="Unmet kilograms do not reconcile"):
        a.validate_rows(rows)

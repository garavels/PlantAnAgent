"""Guard held-out admission, clustered inference, and auditable fixed physics."""
from copy import deepcopy
import gzip
import json

import pytest

from docs.research.tools.learned_control_v1 import analyze as a, evaluate as e


def fixture():
    controllers = {"storage": {"label": "storage", "kind": "baseline", "baseline": "storage"}}
    controllers.update({f"learned-{s}": {"label": f"learned-{s}", "kind": "learned", "group": "selected", "optimizer_seed": s}
                        for s in (101, 211, 307)})
    spec = {"sha256": "frozen", "split": "development", "seeds": list(e.DEVELOPMENT_SEEDS[:4]),
            "conditions": list(e.CONDITIONS), "controllers": controllers,
            "primary": {"recipe": "selected", "reference": "storage"}}
    spec["cases"] = [{"seed": s, "condition": c, "controller": p} for s in spec["seeds"] for c in spec["conditions"] for p in controllers]
    rows = []
    for case in spec["cases"]:
        baseline = case["controller"] == "storage"
        unmet = (100.0 if baseline else 40.0) if case["condition"] in a.STRESS else 0.0
        rows.append({**case, "freeze_sha256": "frozen", "execution_complete": True, "sources_unchanged": True, "failure": None,
            "every_delivery_passed": unmet == 0, "economic_metrics": {"unmet_kg": unmet, "net_operating_value_eur": 1000.0 if baseline else 2000.0},
            "delivery": {"settlements": [{"unmet_kg": unmet}] + [{"unmet_kg": 0.0} for _ in range(27)]},
            "minute_audit": {"all_physical_invariants_passed": True}, "economic_audit": {"passed": True},
            "physical_solar_forcing_sha256": f"{case['seed']}-{case['condition']}", "initial_state": {"seed": case["seed"]}})
    return spec, rows


def test_final_grid_cannot_be_shortened_or_reuse_development():
    e._validate_grid("final", list(e.FINAL_SEEDS), list(e.CONDITIONS))
    for seeds, conditions in ((e.FINAL_SEEDS[:2], e.CONDITIONS), (e.FINAL_SEEDS, e.CONDITIONS[1:]), (e.DEVELOPMENT_SEEDS, e.CONDITIONS)):
        with pytest.raises(ValueError):
            e._validate_grid("final", list(seeds), list(conditions))
    with pytest.raises(ValueError, match="Truncated engineering"):
        e.freeze_panel("unused", [], e.DEVELOPMENT_SEEDS[:1], max_decisions=10)
    with pytest.raises(ValueError, match="gate"):
        e.freeze_panel("unused", [{"label": "storage", "kind": "baseline", "baseline": "storage"}], e.FINAL_SEEDS, split="final")


def test_exact_case_grid_rejects_missing_duplicate_foreign_and_nonpaired():
    spec, rows = fixture()
    assert len(a.validate_rows(rows, spec)) == 112
    with pytest.raises(ValueError, match="Incomplete"):
        a.validate_rows(rows[:-1], spec)
    with pytest.raises(ValueError, match="Duplicate"):
        a.validate_rows(rows + [rows[0]], spec)
    changed = deepcopy(rows)
    changed[0]["seed"] = 123
    with pytest.raises(ValueError, match="Unprescribed"):
        a.validate_rows(changed, spec)
    changed = deepcopy(rows)
    changed[0]["initial_state"] = {"bad": 1}
    with pytest.raises(ValueError, match="Paired"):
        a.validate_rows(changed, spec)


def test_joint_bar_uses_three_policies_and_declared_mixture():
    spec, rows = fixture()
    result = a.analyze_rows(rows, spec)
    claim = result["primary_claim"]
    assert claim["strong_headline_bar_met"]
    assert claim["consistent_across_optimizer_seeds"]
    assert claim["relative_stress_unmet_reduction"] == pytest.approx(.6)
    estimate = claim["comparisons"]["mixture"]["metrics"]["net_operating_value_eur"]
    assert estimate["confidence"] == .975
    assert estimate["replicates"] == 10_000
    assert estimate["mean"] == pytest.approx(1000)
    assert len(result["groups"]["selected"]) == 3
    assert len(claim["individual_policies"]) == 3


def test_bad_optimizer_seed_is_not_selected_away():
    spec, rows = fixture()
    for row in rows:
        if row["controller"] == "learned-307":
            row["economic_metrics"]["net_operating_value_eur"] = -1000
    result = a.analyze_rows(rows, spec)
    assert not result["primary_claim"]["strong_headline_bar_met"]
    assert not result["primary_claim"]["consistent_across_optimizer_seeds"]
    assert result["primary_claim"]["comparisons"]["mixture"]["metrics"]["net_operating_value_eur"]["mean"] == pytest.approx(0)


def test_failures_remain_in_denominator_and_block_headline():
    for failure in ("execution", "audit"):
        spec, rows = fixture()
        row = next(r for r in rows if r["controller"] == "learned-101" and r["condition"] == "pv6")
        if failure == "execution":
            row.update(execution_complete=False, failure={"type": "test"})
        else:
            row["economic_audit"]["passed"] = False
        result = a.analyze_rows(rows, spec)
        assert not result["all_cases_audited"]
        assert not result["primary_claim"]["strong_headline_bar_met"]
        assert result["primary_claim"]["comparisons"]["stress"]["metrics"]["unmet_kg"]["mean"] is None


def test_cluster_unit_is_seed_not_repeated_conditions():
    spec, rows = fixture()
    for row in rows:
        if row["controller"] != "storage":
            row["economic_metrics"]["net_operating_value_eur"] = 1000 + (row["seed"] % 4)*100
    index = a.validate_rows(rows, spec)
    single = a.comparison(index, spec["seeds"], ["learned-101"], ["storage"], {"pv6": 1})
    repeated = a.comparison(index, spec["seeds"], ["learned-101"], ["storage"], {c: .2 for c in a.STRESS})
    assert single["metrics"]["net_operating_value_eur"]["interval"] == repeated["metrics"]["net_operating_value_eur"]["interval"]
    assert repeated["metrics"]["net_operating_value_eur"]["n_base_scenarios"] == 4


def test_final_selection_binds_recipe_checkpoint_champion_and_primary(tmp_path):
    policy = {"label": "chosen", "kind": "learned", "checkpoint": "/fixed/policy.zip", "group": "recipe", "optimizer_seed": 101}
    champion = {"label": "storage", "kind": "baseline", "baseline": "storage", "config": {"methane_buffer_h": 72.0}}
    primary = {"recipe": "recipe", "reference": "storage"}
    selected = {"recipe_id": "recipe", "controllers": [policy], "baseline_champion": champion,
                "seeds": list(e.FINAL_SEEDS), "conditions": list(e.CONDITIONS), "primary": primary}
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(json.dumps(selected))
    gate = {"schema": "learned-control-v1-evaluation-gate", "selected_recipe_frozen": True,
            "selected_recipe": {"path": selected_path.name, "sha256": e.file_pin(selected_path)["sha256"]}}
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate))
    actual = [{**policy, "checkpoint_files": {"checkpoint": {"sha256": "hash"}}}, champion]
    assert e._validate_selection(gate_path, actual, e.FINAL_SEEDS, e.CONDITIONS, primary)["sha256"] == gate["selected_recipe"]["sha256"]
    changed = deepcopy(actual)
    changed[0]["checkpoint"] = "/another/policy.zip"
    with pytest.raises(ValueError, match="learned policies"):
        e._validate_selection(gate_path, changed, e.FINAL_SEEDS, e.CONDITIONS, primary)
    changed = deepcopy(actual)
    changed[1]["config"]["methane_buffer_h"] = 48.0
    with pytest.raises(ValueError, match="baseline champion"):
        e._validate_selection(gate_path, changed, e.FINAL_SEEDS, e.CONDITIONS, primary)
    with pytest.raises(ValueError, match="primary comparison"):
        e._validate_selection(gate_path, actual, e.FINAL_SEEDS, e.CONDITIONS, {"recipe": "recipe", "reference": "other"})


@pytest.mark.parametrize("condition", ["ordinary", "meth48"])
def test_full_week_baseline_uses_correct_fault_auditor_and_retains_inputs(tmp_path, monkeypatch, condition):
    seed = 50_000_000
    spec = {"sha256": "engineering", "split": "train", "seeds": [seed], "gate": None,
            "max_decisions": None, "controllers": {"storage": {"kind": "baseline", "baseline": "storage"}},
            "cases": [{"seed": seed, "condition": condition, "controller": "storage"}]}
    monkeypatch.setattr(e, "require_freeze", lambda path: spec)
    row = e.run_case(tmp_path / "unused-freeze.json", seed, condition, "storage", tmp_path / "case")
    assert row["failure"] is None
    assert row["execution_complete"]
    assert row["minute_audit"]["all_physical_invariants_passed"]
    assert row["economic_audit"]["passed"]
    assert row["physical_states"] == 10081
    with gzip.open(row["recording"]["path"], "rt") as stream:
        record = json.load(stream)
    assert len(record["decision_trace"]) == 672
    assert len(record["minute_truth_states"]) == 10081
    assert len(record["decision_trace"][0]["evidence"]["observation"]) == 149
    assert len(record["decision_trace"][0]["evidence"]["action"]) == 7

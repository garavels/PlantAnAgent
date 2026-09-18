"""Report integrity fixtures only; no final environment is ever constructed."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path

import pytest

from docs.research.tools.learned_control_report_v1 import claims
from docs.research.tools.learned_control_v1 import analyze, evaluate, tune_baselines
from docs.research.tools.wholeplant_ppo_v1.portable import digest, file_pin


def fixture_panel(*, seeds=(51_000_000, 51_000_001), policies=("a",), grouped=False):
    controllers = {"storage": {"label": "storage", "kind": "baseline", "baseline": "storage", "config": {}}}
    controllers.update({p: {"label": p, "kind": "learned", **({"group": "recipe"} if grouped else {})} for p in policies})
    cases = [{"seed": s, "condition": c, "controller": p}
             for s in seeds for c in evaluate.CONDITIONS for p in controllers]
    spec = {"split": "development", "seeds": list(seeds), "conditions": list(evaluate.CONDITIONS),
            "controllers": controllers, "cases": cases, "sha256": "fixture", "primary": None}
    rows = []
    for case in cases:
        rows.append({**case, "execution_complete": True, "sources_unchanged": True, "failure": None,
            "minute_audit": {"all_physical_invariants_passed": True}, "economic_audit": {"passed": True},
            "every_delivery_passed": True,
            "economic_metrics": {"unmet_kg": 0., "net_operating_value_eur": 10000.,
                                 "components_eur": {key: 0. for key in claims.COMPONENTS}},
            "final_state": {key: .5 if key == "battery_soc" else 100. for key in claims.INVENTORIES},
            "initial_state": {"battery_soc": .5}, "physical_solar_forcing_sha256": "paired",
            "delivery": {"settlements": [{"unmet_kg": 0.} for _ in range(28)]}})
    return rows, spec


def set_unmet(row, amount):
    row["economic_metrics"]["unmet_kg"] = float(amount)
    row["economic_metrics"]["net_operating_value_eur"] = 10000. - 24 * amount
    row["delivery"]["settlements"][0]["unmet_kg"] = float(amount)
    row["every_delivery_passed"] = amount <= 1e-9


def report(rows, spec):
    analysis = analyze.analyze_rows(rows, spec)
    return claims.build_claims(rows, spec, analysis, integrity={"required": False, "verified": None})


@pytest.mark.parametrize("audited,aggregate,consistent", list(product((False, True), repeat=3)))
def test_joint_claim_requires_all_three_flags(audited, aggregate, consistent):
    analysis = {"split": "final", "all_cases_audited": audited, "primary_claim": {
        "strong_headline_bar_met": aggregate, "consistent_across_optimizer_seeds": consistent}}
    result = claims.headline_decision(analysis, {"verified": True})
    assert result["final_joint_superiority_supported"] is (audited and aggregate and consistent)
    development = claims.headline_decision({**analysis, "split": "development"}, {"verified": None})
    assert development["final_joint_superiority_supported"] is False


def test_final_claim_requires_verified_concrete_selection():
    with pytest.raises(ValueError, match="provenance"):
        claims.headline_decision({"split": "final", "all_cases_audited": True, "primary_claim": None}, {"verified": False})


def test_worst_quarter_uses_ceiling_and_keeps_stress_cells_equal_weighted():
    rows, spec = fixture_panel()
    stressed = [r for r in rows if r["controller"] == "a" and r["condition"] in analyze.STRESS]
    for amount, row in enumerate(stressed, 1):
        set_unmet(row, amount)
    result = report(rows, spec)
    stress = next(r for r in result["summaries"] if r["population"] == "a" and r["panel"] == "stress")
    assert stress["expected_cells"] == 10
    assert stress["worst_quarter_cell_count"] == 3
    assert stress["worst_quarter_unmet_kg"] == 9.
    assert stress["mean_unmet_kg"] == pytest.approx(5.5)
    mixture = next(r for r in result["summaries"] if r["population"] == "a" and r["panel"] == "mixture")
    assert mixture["mean_unmet_kg"] == pytest.approx(2.75)
    assert mixture["worst_quarter_unmet_kg"] is None


def test_rescues_regressions_and_failed_case_are_not_complete_case_deleted():
    rows, spec = fixture_panel()
    one = lambda s, c, p: next(r for r in rows if (r["seed"], r["condition"], r["controller"]) == (s, c, p))
    set_unmet(one(spec["seeds"][0], "pv6", "storage"), 10)
    set_unmet(one(spec["seeds"][1], "pv6", "a"), 20)
    result = report(rows, spec)
    comparison = next(r for r in result["delivery_comparisons"] if r["population"] == "a" and r["panel"] == "stress")
    assert (comparison["rescues"], comparison["regressions"], comparison["expected_pairs"]) == (1, 1, 10)
    failed = one(spec["seeds"][0], "pv18", "a")
    failed.update(execution_complete=False, failure={"type": "fixture"})
    result = report(rows, spec)
    comparison = next(r for r in result["delivery_comparisons"] if r["population"] == "a" and r["panel"] == "stress")
    assert comparison["complete"] is False and comparison["rescues"] is None and comparison["regressions"] is None
    assert comparison["expected_pairs"] == 10 and comparison["audited_pairs"] == 9
    stress = next(r for r in result["summaries"] if r["population"] == "a" and r["panel"] == "stress")
    assert stress["mean_unmet_kg"] is None and stress["worst_quarter_unmet_kg"] is None
    assert all(v is None for v in stress["mean_components_eur"].values())
    assert all(v is None for v in stress["mean_ending_inventory"].values())


def test_recipe_tail_is_mean_of_policy_tails_not_tail_of_an_ensemble():
    rows, spec = fixture_panel(seeds=(51_000_000,), policies=("a", "b"), grouped=True)
    for policy, amounts in (("a", (0, 0, 0, 0, 100)), ("b", (60, 60, 0, 0, 0))):
        for condition, amount in zip(analyze.STRESS, amounts):
            set_unmet(next(r for r in rows if r["controller"] == policy and r["condition"] == condition), amount)
    result = report(rows, spec)
    recipe = next(r for r in result["summaries"] if r["population"] == "recipe:recipe" and r["panel"] == "stress")
    assert recipe["worst_quarter_unmet_kg"] == 55.
    assert recipe["population_kind"] == "mean_of_fixed_policies_not_ensemble"


def test_tampered_analysis_and_missing_prescribed_case_are_rejected():
    rows, spec = fixture_panel()
    analysis = analyze.analyze_rows(rows, spec)
    analysis["all_cases_audited"] = False
    with pytest.raises(ValueError, match="reproduce"):
        claims.build_claims(rows, spec, analysis, integrity={})
    with pytest.raises(ValueError, match="Incomplete"):
        report(rows[:-1], spec)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return file_pin(path)


def final_metadata_fixture(tmp_path):
    # Metadata-only final admission fixture: no final environment or outcomes.
    rows, dev_spec = fixture_panel(seeds=(51_000_000,), policies=())
    dev_freeze = write_json(tmp_path / "development/freeze.json", dev_spec)
    summary = {"rows": rows, "freeze": dev_freeze}
    summary_pin = write_json(tmp_path / "development/summary.json", summary)
    champion = dev_spec["controllers"]["storage"]
    selection_pin = write_json(tmp_path / "development/selection-ranking.json", {
        "summary": summary_pin, "ranking": tune_baselines.ranking(summary), "champion": champion})
    controllers = {"storage": deepcopy(champion)}
    attempts = {}
    for seed in (901, 902, 903):
        label = f"learned-{seed}"
        root = tmp_path / "attempts" / label
        plan = {"recipe": {"optimizer_seed": seed}, "environment_identity": {"fixture": True}}
        attempts[label] = write_json(root / "plan.json", plan)
        path = root / "training/policy.zip"
        path.parent.mkdir(parents=True)
        path.write_bytes(f"metadata-only-checkpoint-{seed}".encode())
        checkpoint = file_pin(path)
        metadata = write_json(path.with_suffix(".json"), {"optimizer_seed": seed, "plan_sha256": digest(plan),
            "environment_identity": plan["environment_identity"], "checkpoint": checkpoint})
        controllers[label] = {"label": label, "kind": "learned", "group": "chosen", "optimizer_seed": seed,
            "checkpoint": str(path), "checkpoint_files": {"checkpoint": checkpoint, "metadata": metadata}}
    selected = {"recipe_id": "chosen", "seeds": [], "conditions": [],
        "primary": {"recipe": "chosen", "reference": "storage"},
        "controllers": [deepcopy(c) for c in controllers.values() if c["kind"] == "learned"],
        "baseline_champion": champion, "development_baseline_selection": selection_pin,
        "training_attempts": attempts}
    spec = {"split": "final", "seeds": [], "conditions": [], "controllers": controllers, "primary": selected["primary"]}
    def persist():
        spec["selected_recipe"] = write_json(tmp_path / "selected.json", selected)
        spec["gate"] = write_json(tmp_path / "gate.json", {"schema": "learned-control-v1-evaluation-gate",
            "selected_recipe_frozen": True, "selected_recipe": spec["selected_recipe"]})
    persist()
    return spec, selected, persist


def test_final_sidecar_seeds_distinct_attempts_and_development_champion_verify(tmp_path):
    spec, selected, persist = final_metadata_fixture(tmp_path)
    result = claims.verify_final_selection(spec)
    assert result["verified"] is True and len(result["training_attempts"]) == 3
    label = "learned-902"
    pin = spec["controllers"][label]["checkpoint_files"]["metadata"]
    metadata = json.loads(Path(pin["path"]).read_text())
    metadata["optimizer_seed"] = 901
    new_pin = write_json(Path(pin["path"]), metadata)
    spec["controllers"][label]["checkpoint_files"]["metadata"] = new_pin
    with pytest.raises(ValueError, match="seed or checkpoint"):
        claims.verify_final_selection(spec)


def test_unpinned_attempt_or_changed_champion_cannot_support_final_claim(tmp_path):
    spec, selected, persist = final_metadata_fixture(tmp_path)
    label = "learned-903"
    del selected["training_attempts"][label]
    persist()
    with pytest.raises(ValueError, match="training_attempts"):
        claims.verify_final_selection(spec)
    spec, selected, persist = final_metadata_fixture(tmp_path / "second")
    development_pin = selected["development_baseline_selection"]
    development = json.loads(Path(development_pin["path"]).read_text())
    development["champion"] = {**development["champion"], "config": {"battery_target_soc": .85}}
    selected["development_baseline_selection"] = write_json(Path(development_pin["path"]), development)
    persist()
    with pytest.raises(ValueError, match="champion differs"):
        claims.verify_final_selection(spec)


def test_existing_development_panel_end_to_end_writes_only_requested_report(tmp_path):
    panel = claims.ROOT / "docs/research/results/learned_control_20260918/development/bc-initial-fidelity"
    if not (panel / "analysis.json").exists():
        pytest.skip("Completed development engineering fixture is unavailable")
    destination = tmp_path / "claims.json"
    result = claims.run(panel, destination)
    assert destination.is_file()
    assert result["headline"]["final_joint_superiority_supported"] is False
    assert result["selection_integrity"]["required"] is False
    assert result["provenance"]["analysis"] == file_pin(panel / "analysis.json")
    assert len([r for r in result["summaries"] if r["panel"] == "stress"]) == 3
    with pytest.raises(FileExistsError):
        claims.run(panel, destination)

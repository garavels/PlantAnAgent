"""Versioned report decisions and descriptive statistics over frozen evaluations.

The original analyzer remains unchanged. This report combines its aggregate
and optimizer-seed checks, verifies final selection provenance, and reports
supplementary outcomes without dropping failed cases or inventing an ensemble.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from numbers import Real
from pathlib import Path
import statistics

from docs.research.tools.learned_control_v1 import analyze, evaluate
from docs.research.tools.wholeplant_ppo_v1.portable import digest, file_pin

ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "learned-control-report-claims-v1"
COMPONENTS = ("delivered_service_value", "unmet_delivery_penalty", "solar_opportunity_cost",
              "battery_throughput_reserve", "electrolyser_life_reserve",
              "methane_endowment_liability_change", "battery_endowment_liability_change")
INVENTORIES = ("battery_soc", "battery_energy_kwh", "h2_level_kg", "co2_level_kg", "ch4_level_kg")


def definitions():
    return {
        "schema": SCHEMA,
        "strict_joint_claim": "all_cases_audited AND aggregate strong_headline_bar_met AND consistent_across_optimizer_seeds; final selection integrity must also verify",
        "unchanged_primary_analysis": "Original paired environmental-seed bootstrap and endpoint thresholds are reused exactly, without changing the frozen analyzer",
        "stress_conditions": list(analyze.STRESS), "mixture_weights": dict(analyze.MIXTURE),
        "delivery_rescue": "Treatment meets all28 deliveries and matched reference fails at least one; paired by environment seed and condition",
        "delivery_regression": "Treatment fails at least one delivery and matched reference meets all28",
        "worst_quarter_stress_unmet": "For each fixed policy, sort all N equally weighted stress seed-condition cells by unmet kg descending and average the largest ceil(0.25*N); descriptive, no independence or inferential claim",
        "worst_quarter_condition_unmet": "The same rule within one condition across its environmental seeds",
        "worst_quarter_mixture": "Not reported for the unequally weighted ordinary/stress mixture",
        "cost_components": "Mean signed EUR components of the unchanged economic ledger; service value is positive and charges normally negative",
        "ending_inventories": "Mean physical state at the final seven-day boundary, after its delivery settlement",
        "recipe_summary": "Arithmetic mean of the fixed policies' descriptive statistics, including their individual worst-quarter means; not an ensemble or a new deployed controller",
        "failed_cases": "Every prescribed case remains in its denominator. Any failed, unaudited, missing or nonfinite constituent makes the affected aggregate unavailable; no complete-case deletion",
        "uncertainty": "Supplementary tails and rescue counts are descriptive; their cells are not asserted independent. Primary bootstrap uncertainty remains conditional on the trained policies",
    }


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _identity(controller):
    return {key: controller[key] for key in evaluate.CONTROLLER_IDENTITY_FIELDS if key in controller}


def _verified_json(pin):
    if not isinstance(pin, dict) or not {"path", "bytes", "sha256"}.issubset(pin):
        raise ValueError("A concrete path/bytes/SHA256 pin is required")
    actual = file_pin(pin["path"])
    if any(actual[key] != pin[key] for key in ("path", "bytes", "sha256")):
        raise ValueError("Pinned evidence changed: " + str(pin["path"]))
    return json.loads(Path(pin["path"]).read_text())


def verify_final_selection(spec):
    """Authenticate concrete attempts and the development-selected comparator."""
    if spec["split"] != "final":
        return {"required": False, "verified": None, "reason": "Development panel; final selection remains closed"}
    selected = _verified_json(spec["selected_recipe"])
    evaluate._validate_selection(spec["gate"]["path"], list(spec["controllers"].values()),
                                 spec["seeds"], spec["conditions"], spec["primary"])
    members = [c for c in spec["controllers"].values()
               if c.get("group", c.get("recipe")) == selected["recipe_id"]]
    if len(members) != 3 or any(c["kind"] != "learned" for c in members):
        raise ValueError("Final recipe must contain exactly three learned policies")
    attempts = selected.get("training_attempts")
    if not isinstance(attempts, dict) or set(attempts) != {c["label"] for c in members}:
        raise ValueError("Final selection must pin all three training_attempts by controller label")
    records = []
    for controller in members:
        metadata_pin = controller["checkpoint_files"]["metadata"]
        metadata = _verified_json(metadata_pin)
        plan_pin = attempts[controller["label"]]
        plan = _verified_json(plan_pin)
        seed = controller.get("optimizer_seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Optimizer seed must be an actual integer")
        if (metadata.get("optimizer_seed") != seed or plan.get("recipe", {}).get("optimizer_seed") != seed
                or metadata.get("plan_sha256") != digest(plan)):
            raise ValueError("Controller seed or checkpoint does not match its pinned training attempt")
        if metadata.get("environment_identity") != plan.get("environment_identity"):
            raise ValueError("Checkpoint and training environment identities differ")
        checkpoint_pin = controller["checkpoint_files"]["checkpoint"]
        if any(metadata.get("checkpoint", {}).get(k) != checkpoint_pin[k] for k in ("bytes", "sha256")):
            raise ValueError("Checkpoint sidecar does not identify the selected checkpoint")
        records.append({"controller": controller["label"], "optimizer_seed": seed,
                        "training_attempt": plan_pin, "checkpoint_metadata": metadata_pin,
                        "checkpoint": checkpoint_pin, "plan_content_sha256": digest(plan)})
    for name, values in {
        "optimizer seeds": [r["optimizer_seed"] for r in records],
        "training attempt files": [r["training_attempt"]["path"] for r in records],
        "training attempt digests": [r["plan_content_sha256"] for r in records],
        "checkpoint digests": [r["checkpoint"]["sha256"] for r in records],
    }.items():
        if len(set(values)) != 3:
            raise ValueError("Final policies do not have three distinct " + name)
    development_pin = selected.get("development_baseline_selection")
    development = _verified_json(development_pin)
    champion = selected["baseline_champion"]
    if champion.get("kind") != "baseline" or _identity(champion) != _identity(development["champion"]):
        raise ValueError("Final champion differs from the pinned development selection")
    development_summary = _verified_json(development["summary"])
    development_spec = _verified_json(development_summary["freeze"])
    if development_spec["split"] != "development":
        raise ValueError("Comparator selection must use development data")
    analyze.validate_rows(development_summary["rows"], development_spec)
    from docs.research.tools.learned_control_v1.tune_baselines import ranking
    actual_ranking = ranking(development_summary)
    if (digest(actual_ranking) != digest(development["ranking"])
            or actual_ranking[0]["controller"] != champion["label"]
            or _identity(development_spec["controllers"][champion["label"]]) != _identity(champion)):
        raise ValueError("Baseline champion is not the recorded development ranking winner")
    return {"required": True, "verified": True, "selected_recipe": spec["selected_recipe"],
            "training_attempts": records, "development_baseline_selection": development_pin,
            "development_summary": development["summary"], "development_freeze": development_summary["freeze"],
            "baseline_champion": champion,
            "scope": "Distinct declared seeds and authenticated attempt plans; not a claim of universal retraining reliability"}


def headline_decision(analysis, integrity):
    primary = analysis.get("primary_claim")
    checks = {"all_prescribed_cases_audited": bool(analysis.get("all_cases_audited")),
              "aggregate_bar_met": bool(primary and primary.get("strong_headline_bar_met")),
              "optimizer_seed_directions_consistent": bool(primary and primary.get("consistent_across_optimizer_seeds"))}
    strict = all(checks.values())
    final = analysis.get("split") == "final"
    if final and integrity.get("verified") is not True:
        raise ValueError("Final headline requires verified selection provenance")
    return {"checks": checks, "strict_joint_bar_met": strict,
            "final_joint_superiority_supported": bool(final and strict),
            "status": "no_primary_comparison" if primary is None else ("met" if strict else "not_met"),
            "not_met": [key for key, value in checks.items() if not value],
            "original_aggregate_flag": None if primary is None else primary.get("strong_headline_bar_met"),
            "original_consistency_flag": None if primary is None else primary.get("consistent_across_optimizer_seeds"),
            "note": "Development results never establish the fresh-final headline; the original analyzer's two flags remain visible"}


def _policy_summary(index, spec, controller, panel, weights):
    rows = [index[(seed, condition, controller)] for seed in spec["seeds"] for condition in weights]
    complete = all(analyze.audited(r) for r in rows)
    def value(row, field):
        if field in ("unmet_kg", "net_operating_value_eur"):
            return row.get("economic_metrics", {}).get(field)
        if field == "all_delivery_success_rate":
            return float(row["every_delivery_passed"]) if "every_delivery_passed" in row else None
        prefix, key = field.split(":", 1)
        return ((row.get("economic_metrics") or {}).get("components_eur", {}).get(key) if prefix == "component"
                else (row.get("final_state") or {}).get(key))
    fields = ("unmet_kg", "net_operating_value_eur", "all_delivery_success_rate",
              *("component:" + k for k in COMPONENTS), *("inventory:" + k for k in INVENTORIES))
    complete = complete and all(_finite(value(row, field)) for row in rows for field in fields)
    def mean(field):
        return (sum(weights[row["condition"]] * value(row, field) for row in rows)
                / (len(spec["seeds"]) * sum(weights.values()))) if complete else None
    worst_defined = panel == "stress" or len(weights) == 1
    count = math.ceil(.25 * len(rows)) if worst_defined else None
    worst = statistics.mean(sorted((value(r, "unmet_kg") for r in rows), reverse=True)[:count]) if complete and worst_defined else None
    return {"population": controller, "population_kind": "fixed_policy", "members": [controller],
            "panel": panel, "weights": weights, "expected_cells": len(rows),
            "audited_cells": sum(analyze.audited(r) for r in rows), "complete": complete,
            "mean_unmet_kg": mean("unmet_kg"), "mean_contribution_eur": mean("net_operating_value_eur"),
            "all_delivery_success_rate": mean("all_delivery_success_rate"),
            "worst_quarter_unmet_kg": worst, "worst_quarter_cell_count": count,
            "mean_components_eur": {k: mean("component:" + k) for k in COMPONENTS},
            "mean_ending_inventory": {k: mean("inventory:" + k) for k in INVENTORIES}}


def _rescue_summary(index, spec, controller, reference, panel, conditions):
    pairs = [(seed, condition, index[(seed, condition, controller)], index[(seed, condition, reference)])
             for seed in spec["seeds"] for condition in conditions]
    invalid = [{"seed": s, "condition": c} for s, c, a, b in pairs if not (analyze.audited(a) and analyze.audited(b))]
    complete = not invalid
    rescued = [{"seed": s, "condition": c} for s, c, a, b in pairs if a["every_delivery_passed"] and not b["every_delivery_passed"]] if complete else None
    regressed = [{"seed": s, "condition": c} for s, c, a, b in pairs if b["every_delivery_passed"] and not a["every_delivery_passed"]] if complete else None
    return {"population": controller, "population_kind": "fixed_policy", "members": [controller],
            "reference": reference, "panel": panel, "expected_pairs": len(pairs),
            "audited_pairs": len(pairs) - len(invalid), "complete": complete,
            "rescued_cells": rescued, "regressed_cells": regressed,
            "rescues": len(rescued) if complete else None, "regressions": len(regressed) if complete else None,
            "both_succeed": sum(a["every_delivery_passed"] and b["every_delivery_passed"] for _, _, a, b in pairs) if complete else None,
            "both_fail": sum(not a["every_delivery_passed"] and not b["every_delivery_passed"] for _, _, a, b in pairs) if complete else None,
            "invalid_cells": invalid}


def build_claims(rows, spec, analysis, *, integrity):
    index = analyze.validate_rows(rows, spec)
    expected = analyze.analyze_rows(rows, spec)
    if any(digest(analysis.get(k)) != digest(v) for k, v in expected.items()):
        raise ValueError("Supplied analysis does not reproduce from the frozen rows")
    panels = analyze.panel_weights(spec["conditions"])
    summaries = [_policy_summary(index, spec, c, panel, weights) for c in spec["controllers"] for panel, weights in panels.items()]
    comparisons = [_rescue_summary(index, spec, c, b, panel, tuple(weights))
                   for c in spec["controllers"] for b, bs in spec["controllers"].items() if b != c and bs["kind"] == "baseline"
                   for panel, weights in panels.items() if panel != "mixture"]
    for group, members in analysis["groups"].items():
        for panel in panels:
            constituents = [next(r for r in summaries if r["population"] == m and r["panel"] == panel) for m in members]
            complete = all(r["complete"] for r in constituents)
            average = lambda key: statistics.mean(r[key] for r in constituents) if complete and all(r[key] is not None for r in constituents) else None
            summaries.append({"population": "recipe:" + group, "population_kind": "mean_of_fixed_policies_not_ensemble",
                "members": members, "panel": panel, "weights": panels[panel], "complete": complete,
                "expected_cells": sum(r["expected_cells"] for r in constituents),
                "audited_cells": sum(r["audited_cells"] for r in constituents),
                **{key: average(key) for key in ("mean_unmet_kg", "mean_contribution_eur", "all_delivery_success_rate", "worst_quarter_unmet_kg")},
                "worst_quarter_cell_count_per_policy": constituents[0]["worst_quarter_cell_count"],
                **{field: {k: statistics.mean(r[field][k] for r in constituents) if complete else None for k in keys}
                   for field, keys in (("mean_components_eur", COMPONENTS), ("mean_ending_inventory", INVENTORIES))}})
        for b, bs in spec["controllers"].items():
            if bs["kind"] != "baseline" or b in members:
                continue
            for panel in (p for p in panels if p != "mixture"):
                constituents = [next(r for r in comparisons if r["population"] == m and r["reference"] == b and r["panel"] == panel) for m in members]
                complete = all(r["complete"] for r in constituents)
                comparisons.append({"population": "recipe:" + group, "population_kind": "mean_of_fixed_policies_not_ensemble",
                    "members": members, "reference": b, "panel": panel, "complete": complete,
                    "expected_pairs_per_policy": constituents[0]["expected_pairs"],
                    **{"mean_" + key: statistics.mean(r[key] for r in constituents) if complete else None
                       for key in ("rescues", "regressions", "both_succeed", "both_fail")}})
    return {"schema": SCHEMA, "freeze_sha256": spec["sha256"], "split": spec["split"],
            "definitions": definitions(), "definitions_sha256": digest(definitions()),
            "selection_integrity": integrity, "headline": headline_decision(analysis, integrity),
            "summaries": summaries, "delivery_comparisons": comparisons}


def run(panel, destination=None):
    panel = Path(panel).resolve()
    spec = evaluate.require_freeze(panel / "freeze.json")
    summary = json.loads((panel / "summary.json").read_text())
    if file_pin(panel / "freeze.json") != summary["freeze"]:
        raise ValueError("Summary is not bound to this panel freeze")
    analysis_path = panel / "analysis.json"
    analysis = json.loads(analysis_path.read_text())
    if file_pin(panel / "summary.json") != analysis["summary"]:
        raise ValueError("Analysis is not bound to the completed summary")
    result = build_claims(summary["rows"], spec, analysis, integrity=verify_final_selection(spec))
    result.update(created_at=datetime.now(timezone.utc).isoformat(), provenance={
        "protocol": file_pin(ROOT / "docs/research/LEARNED_CONTROL_V1_PROTOCOL.md"),
        "panel_freeze": file_pin(panel / "freeze.json"), "summary": file_pin(panel / "summary.json"),
        "analysis": file_pin(analysis_path), "report_source": file_pin(Path(__file__)),
        "unchanged_analyzer_source": file_pin(Path(analyze.__file__))})
    destination = panel / "claims.json" if destination is None else Path(destination)
    evaluate.publish(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", "--output", dest="panel", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    result = run(args.panel, args.destination)
    print(json.dumps({"schema": SCHEMA, "headline": result["headline"]}, indent=2))


if __name__ == "__main__":
    main()

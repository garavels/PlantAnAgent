"""Prospective fixed-environment recipe comparison without selecting optimizer seeds.

Recipe means average all selected policies within each environment seed and
condition. Bootstrap resampling moves all conditions and policies for a base
seed together; its uncertainty is conditional on these trained policies.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

import numpy as np

from docs.research.tools.wholeplant_ppo_v1.portable import digest, file_pin

SCHEMA = "learned-control-analysis-v1"
STRESS = ("pv6", "pv18", "pv36", "meth24", "meth48")
MIXTURE = {"ordinary": 0.5, **{condition: 0.1 for condition in STRESS}}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 91_818
PRIMARY_CONFIDENCE = 0.975
METRICS = ("net_operating_value_eur", "unmet_kg", "all_delivery_success", "missed_deliveries")


def audited(row):
    return bool(row.get("execution_complete") and row.get("sources_unchanged") and row.get("failure") is None
                and row.get("minute_audit", {}).get("all_physical_invariants_passed")
                and row.get("economic_audit", {}).get("passed"))


def metric(row, name):
    if not audited(row):
        return None
    if name == "all_delivery_success":
        return float(row["every_delivery_passed"])
    if name == "missed_deliveries":
        return float(sum(d["unmet_kg"] > 1e-9 for d in row["delivery"]["settlements"]))
    value = row["economic_metrics"].get(name)
    return float(value) if isinstance(value, (float, int)) and math.isfinite(value) else None


def validate_rows(rows, spec):
    expected = {(c["seed"], c["condition"], c["controller"]) for c in spec["cases"]}
    index = {}
    for row in rows:
        key = (row["seed"], row["condition"], row["controller"])
        if key not in expected:
            raise ValueError("Unprescribed evaluation case")
        if key in index:
            raise ValueError("Duplicate evaluation case")
        if row.get("freeze_sha256") not in (None, spec["sha256"]):
            raise ValueError("Case belongs to another frozen panel")
        if row.get("execution_complete"):
            settlements = row["delivery"]["settlements"]
            if len(settlements) != 28:
                raise ValueError("Complete week must contain 28 delivery settlements")
            if bool(row["every_delivery_passed"]) != all(d["unmet_kg"] <= 1e-9 for d in settlements):
                raise ValueError("Delivery classification differs from settlements")
            if abs(sum(d["unmet_kg"] for d in settlements)-row["economic_metrics"]["unmet_kg"]) > 1e-6:
                raise ValueError("Unmet kilograms do not reconcile")
        index[key] = row
    if set(index) != expected:
        raise ValueError("Incomplete prescribed evaluation grid")
    for seed in spec["seeds"]:
        for condition in spec["conditions"]:
            cell = [r for (s, c, _), r in index.items() if (s, c) == (seed, condition)]
            for field in ("physical_solar_forcing_sha256", "initial_state"):
                values = [digest(r[field]) for r in cell if r.get(field) is not None]
                if len(set(values)) > 1:
                    raise ValueError("Paired forcing or opening state differs")
    return index


def describe(values):
    complete = bool(values) and all(v is not None and math.isfinite(v) for v in values)
    return {"n": len(values), "n_observed": sum(v is not None for v in values), "complete": complete,
            "mean": statistics.mean(values) if complete else None,
            "min": min(values) if complete else None, "max": max(values) if complete else None,
            "values": values}


def bootstrap(values, *, confidence=0.95):
    result = {"n_base_scenarios": len(values), "confidence": confidence, "replicates": BOOTSTRAP_REPLICATES,
              "bootstrap_seed": BOOTSTRAP_SEED, "unit": "base environment seed; conditions and all recipe policies remain together",
              "complete": bool(values) and all(v is not None and math.isfinite(v) for v in values),
              "mean": None, "interval": [None, None], "per_seed_differences": values}
    if result["complete"]:
        data = np.asarray(values, dtype=float)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        means = data[rng.integers(0, len(data), size=(BOOTSTRAP_REPLICATES, len(data)))].mean(axis=1)
        tail = (1-confidence)/2
        result.update(mean=float(data.mean()), interval=np.quantile(means, [tail, 1-tail]).tolist())
    return result


def _value(index, seed, condition, members, name):
    values = [metric(index[(seed, condition, member)], name) for member in members]
    return statistics.mean(values) if all(v is not None for v in values) else None


def _weighted(index, seed, members, name, weights):
    values = [(weight, _value(index, seed, condition, members, name)) for condition, weight in weights.items()]
    return sum(w*v for w, v in values) / sum(weights.values()) if all(v is not None for _, v in values) else None


def panel_weights(conditions):
    result = {c: {c: 1.0} for c in conditions}
    if all(c in conditions for c in STRESS):
        result["stress"] = {c: 1/len(STRESS) for c in STRESS}
    if all(c in conditions for c in MIXTURE):
        result["mixture"] = MIXTURE
    return result


def comparison(index, seeds, treatment, reference, weights, *, confidence=0.95):
    estimates = {}
    for name in METRICS:
        left = [_weighted(index, s, reference, name, weights) for s in seeds]
        right = [_weighted(index, s, treatment, name, weights) for s in seeds]
        differences = [r-l if r is not None and l is not None else None for l, r in zip(left, right)]
        estimates[name] = {**bootstrap(differences, confidence=confidence),
                           "reference_mean": describe(left)["mean"], "treatment_mean": describe(right)["mean"]}
    return {"treatment_members": treatment, "reference_members": reference, "weights": weights,
            "direction": "treatment minus reference", "metrics": estimates}


def primary_claim(index, spec, groups, primary):
    recipe = primary.get("recipe", primary.get("group"))
    reference = primary.get("reference")
    if recipe not in groups or reference not in spec["controllers"]:
        raise ValueError("Primary recipe or baseline champion is missing from the frozen grid")
    members = groups[recipe]
    optimizer_seeds = [spec["controllers"][member].get("optimizer_seed") for member in members]
    three_independent = len(members) == 3 and None not in optimizer_seeds and len(set(optimizer_seeds)) == 3
    if not all(c in spec["conditions"] for c in MIXTURE):
        raise ValueError("Primary claim requires ordinary plus every prescribed stress condition")
    comparisons = {panel: comparison(index, spec["seeds"], members, [reference], weights, confidence=PRIMARY_CONFIDENCE)
                   for panel, weights in {"mixture": MIXTURE, "stress": {c: .2 for c in STRESS}, "ordinary": {"ordinary": 1.0}}.items()}
    mixture = comparisons["mixture"]["metrics"]["net_operating_value_eur"]
    unmet = comparisons["stress"]["metrics"]["unmet_kg"]
    success = comparisons["stress"]["metrics"]["all_delivery_success"]
    ordinary = comparisons["ordinary"]["metrics"]["net_operating_value_eur"]
    complete = all(v["complete"] for v in (mixture, unmet, success, ordinary))
    reference_unmet = unmet["reference_mean"]
    relative_reduction = ((reference_unmet-unmet["treatment_mean"])/reference_unmet
                          if complete and reference_unmet > 0 else None)
    checks = {
        "three_independent_optimizer_seeds": three_independent,
        "all_cases_complete_and_audited": complete,
        "mixture_contribution_gain_at_least_500_eur": bool(complete and mixture["mean"] >= 500),
        "mixture_contribution_975ci_positive": bool(complete and mixture["interval"][0] > 0),
        "stress_unmet_reduction_at_least_20_percent": bool(relative_reduction is not None and relative_reduction >= .20),
        "stress_unmet_delta_975ci_negative": bool(complete and unmet["interval"][1] < 0),
        "stress_all_delivery_success_not_lower": bool(complete and success["mean"] >= 0),
        "ordinary_contribution_loss_at_most_500_eur": bool(complete and ordinary["mean"] >= -500),
    }
    individuals = []
    for member in members:
        value = comparison(index, spec["seeds"], [member], [reference], MIXTURE)["metrics"]["net_operating_value_eur"]
        loss = comparison(index, spec["seeds"], [member], [reference], {c: .2 for c in STRESS})["metrics"]["unmet_kg"]
        individuals.append({"controller": member, "optimizer_seed": spec["controllers"][member].get("optimizer_seed"),
            "mixture_contribution_delta": value["mean"], "stress_unmet_delta": loss["mean"],
            "both_directions_improved": bool(value["complete"] and loss["complete"] and value["mean"] > 0 and loss["mean"] < 0)})
    return {"recipe": recipe, "reference": reference, "members": members, "checks": checks,
            "strong_headline_bar_met": all(checks.values()),
            "consistent_across_optimizer_seeds": all(i["both_directions_improved"] for i in individuals) and three_independent,
            "relative_stress_unmet_reduction": relative_reduction, "comparisons": comparisons, "individual_policies": individuals,
            "multiplicity": "Two inferential endpoints, Bonferroni 97.5% two-sided bootstrap intervals",
            "limitation": "Uncertainty is over fresh environment seeds conditional on these policies and fixed known disruption timings; three optimizer seeds do not establish universal training reliability"}


def analyze_rows(rows, spec):
    index = validate_rows(rows, spec)
    weights = panel_weights(spec["conditions"])
    groups = {}
    for label, controller in spec["controllers"].items():
        group = controller.get("group", controller.get("recipe"))
        if group:
            groups.setdefault(group, []).append(label)
    populations = {label: [label] for label in spec["controllers"]}
    populations.update({"recipe:"+group: members for group, members in groups.items()})
    summaries = []
    for label, members in populations.items():
        for panel, panel_weight in weights.items():
            summaries.append({"population": label, "members": members, "panel": panel, "weights": panel_weight,
                "n_base_scenarios": len(spec["seeds"]), "n_policy_condition_cells": len(spec["seeds"])*len(members)*len(panel_weight),
                "metrics": {name: describe([_weighted(index, s, members, name, panel_weight) for s in spec["seeds"]]) for name in METRICS}})
    reference_labels = [label for label, c in spec["controllers"].items() if c["kind"] == "baseline"]
    paired = []
    for label, members in populations.items():
        for reference in reference_labels:
            if members == [reference]:
                continue
            for panel, panel_weight in weights.items():
                paired.append({"treatment": label, "reference": reference, "panel": panel,
                    **comparison(index, spec["seeds"], members, [reference], panel_weight)})
    primary = primary_claim(index, spec, groups, spec["primary"]) if spec.get("primary") else None
    return {"schema": SCHEMA, "freeze_sha256": spec["sha256"], "split": spec["split"], "case_count": len(rows),
            "all_cases_audited": all(audited(r) for r in rows), "groups": groups, "summaries": summaries,
            "paired_comparisons": paired, "primary_claim": primary,
            "distribution_caveat": "Mixture weights are a declared experimental objective, not estimated natural disruption probabilities"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from docs.research.tools.learned_control_v1.evaluate import publish, require_freeze
    spec = require_freeze(args.output / "freeze.json")
    summary = json.loads((args.output / "summary.json").read_text())
    result = analyze_rows(summary["rows"], spec)
    result.update(created_at=datetime.now(timezone.utc).isoformat(), summary=file_pin(args.output / "summary.json"))
    publish(args.output / "analysis.json", result)
    print(json.dumps({"case_count": result["case_count"], "all_cases_audited": result["all_cases_audited"],
                      "primary_claim": result["primary_claim"]}, indent=2))


if __name__ == "__main__":
    main()

"""Prospective paired stress analysis; synthetic severities have no event frequency.

The independent resampling unit is the base environment seed. All five stress
conditions for that seed travel together, including when a controller fails.
No policy or condition is selected according to the observed outcome.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

import numpy as np

SEEDS = tuple(range(43_000_000, 43_000_016))
CONDITIONS = ("control", "pv6", "pv18", "pv36", "meth24", "meth48")
CONTROLLERS = ("daylight", "storage", "ppo-101", "ppo-211", "ppo-307")
STRESS_CONDITIONS = CONDITIONS[1:]
FAMILIES = {"solar-loss": ("pv6", "pv18", "pv36"), "methanator-outage": ("meth24", "meth48")}
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 91_708
PRIMARY_COMPARISONS = 6
ADJUSTED_CONFIDENCE = 1 - 0.05 / PRIMARY_COMPARISONS
STOCK_FIELDS = ("battery_energy_kwh", "battery_soc", "h2_level_kg", "co2_level_kg", "ch4_level_kg", "dac_readiness_kwh", "methanator_readiness_kwh")
SCHEMA = "wholeplant-tail-stress-analysis-v1"


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def row_audited(row):
    return bool(row.get("execution_complete") and row.get("sources_unchanged")
                and row.get("failure") is None
                and row.get("minute_audit", {}).get("all_physical_invariants_passed")
                and row.get("economic_audit", {}).get("passed"))


def metric(row, name):
    if not row.get("execution_complete"):
        return None
    if name == "all_delivery_success":
        return float(row["every_delivery_passed"])
    if name == "missed_deliveries":
        settlements = row.get("delivery", {}).get("settlements")
        return sum(d["unmet_kg"] > 1e-9 for d in settlements) if settlements is not None else None
    value = row.get("economic_metrics", {}).get(name)
    return float(value) if finite_number(value) else None


def describe(values):
    """Retain missing attempts in the denominator; never estimate a complete panel from a subset."""
    known = [float(v) for v in values if finite_number(v)]
    complete = len(known) == len(values) and bool(values)
    return {"n": len(values), "n_observed": len(known), "complete": complete,
            "mean": statistics.mean(known) if complete else None,
            "min": min(known) if complete else None, "max": max(known) if complete else None,
            "observed_values": values}


def worst_quarter(values):
    if not values or any(not finite_number(v) for v in values):
        return {"n": len(values), "count": math.ceil(len(values) / 4), "mean_unmet_kg": None}
    count = math.ceil(len(values) / 4)
    return {"n": len(values), "count": count,
            "mean_unmet_kg": statistics.mean(sorted(values, reverse=True)[:count]),
            "interpretation": "Descriptive mean of the worst quarter of these prescribed cells, not population CVaR"}


def paired_cluster_estimate(paired, metric_name, *, confidence=0.95):
    """Bootstrap the mean seed-level difference, preserving all conditions per seed."""
    per_seed = {}
    paired_values = []
    for seed in SEEDS:
        differences = []
        for (cell_seed, condition), (left, right) in sorted(paired.items()):
            if cell_seed != seed:
                continue
            lvalue, rvalue = metric(left, metric_name), metric(right, metric_name)
            delta = rvalue - lvalue if lvalue is not None and rvalue is not None else None
            differences.append(delta)
            paired_values.append({"seed": seed, "condition": condition, "delta": delta})
        per_seed[seed] = (statistics.mean(differences)
                          if differences and all(v is not None for v in differences) else None)
    complete = all(v is not None for v in per_seed.values())
    estimate = {"metric": metric_name, "direction": "treatment minus reference",
                "n_base_scenarios": len(SEEDS), "n_paired_cells": len(paired),
                "bootstrap_unit": "Base seed, retaining every selected condition for that seed",
                "confidence": confidence, "replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED, "complete": complete,
                "mean": None, "interval": [None, None],
                "per_seed_mean_differences": [{"seed": s, "delta": v} for s, v in per_seed.items()],
                "paired_values": paired_values}
    if complete:
        data = np.asarray(list(per_seed.values()), dtype=float)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        sampled_means = data[rng.integers(0, len(data), size=(BOOTSTRAP_REPLICATES, len(data)))].mean(axis=1)
        tail = (1 - confidence) / 2
        estimate.update(mean=float(data.mean()), interval=np.quantile(sampled_means, [tail, 1-tail]).tolist())
    return estimate


def summarize_group(rows, *, label, controller):
    metrics = {name: describe([metric(r, name) for r in rows]) for name in (
        "unmet_kg", "net_operating_value_eur", "missed_deliveries")}
    event_fields = sorted({k for r in rows for k, v in (r.get("event_metrics") or {}).items() if finite_number(v)})
    events = {name: describe([(r.get("event_metrics") or {}).get(name) for r in rows]) for name in event_fields}
    pre_event = {}
    for name in STOCK_FIELDS:
        values = [(r.get("event_metrics") or {}).get("pre_event_state", {}).get(name) for r in rows]
        if any(v is not None for v in values):
            pre_event[name] = describe(values)
    return {"panel": label, "controller": controller, "n_cells": len(rows),
            "n_base_scenarios": len({r["seed"] for r in rows}),
            "complete_cases": sum(bool(r.get("execution_complete")) for r in rows),
            "audited_cases": sum(row_audited(r) for r in rows),
            "all_delivery_successes": sum(bool(r.get("execution_complete") and r.get("every_delivery_passed")) for r in rows),
            "delivery_failure_cells": [{"seed": r["seed"], "condition": r["condition"]} for r in rows if not r.get("every_delivery_passed")],
            "metrics": metrics, "worst_quarter": worst_quarter([metric(r, "unmet_kg") for r in rows]),
            "event_metrics": events, "pre_event_stocks": pre_event}


def paired_comparison(index, *, reference, treatment, conditions, label):
    paired = {(seed, c): (index[(seed, c, reference)], index[(seed, c, treatment)])
              for seed in SEEDS for c in conditions}
    rescues, regressions = [], []
    for (seed, condition), (left, right) in paired.items():
        # Execution failures are retained but never converted to delivery evidence.
        if not left.get("execution_complete") or not right.get("execution_complete"):
            continue
        cell = {"seed": seed, "condition": condition}
        if not left["every_delivery_passed"] and right["every_delivery_passed"]:
            rescues.append(cell)
        if left["every_delivery_passed"] and not right["every_delivery_passed"]:
            regressions.append(cell)
    estimates = {name: paired_cluster_estimate(paired, name) for name in (
        "all_delivery_success", "unmet_kg", "net_operating_value_eur", "missed_deliveries")}
    result = {"panel": label, "reference": reference, "treatment": treatment,
              "conditions": list(conditions), "n_paired_cells": len(paired), "n_base_scenarios": len(SEEDS),
              "all_comparisons_complete_and_audited": all(row_audited(r) for pair in paired.values() for r in pair),
              "rescued_cells": rescues, "regressed_cells": regressions,
              "rescues": len(rescues), "regressions": len(regressions), "metrics": estimates}
    if label == "pooled-stress" and treatment.startswith("ppo-"):
        adjusted = paired_cluster_estimate(paired, "all_delivery_success", confidence=ADJUSTED_CONFIDENCE)
        no_greater_unmet = estimates["unmet_kg"]["mean"] is not None and estimates["unmet_kg"]["mean"] <= 1e-9
        positive_interval = adjusted["interval"][0] is not None and adjusted["interval"][0] > 0
        result["primary_criterion"] = {
            "rescues_at_least_eight": len(rescues) >= 8, "zero_regressions": not regressions,
            "no_greater_mean_unmet_kg": no_greater_unmet,
            "positive_adjusted_success_difference_interval": positive_interval,
            "adjusted_success_difference": adjusted,
            "bonferroni_primary_comparisons": PRIMARY_COMPARISONS,
            "definition": "At least eight rescued stress cells, zero regressions, no greater mean unmet kg, and positive Bonferroni-adjusted seed-cluster bootstrap interval lower bound; all 80 comparisons complete and audited",
            "met": bool(result["all_comparisons_complete_and_audited"] and len(rescues) >= 8
                        and not regressions and no_greater_unmet and positive_interval)}
    return result


def validate_rows(rows):
    expected = {(s, c, p) for s in SEEDS for c in CONDITIONS for p in CONTROLLERS}
    index = {}
    for row in rows:
        key = row["seed"], row["condition"], row["controller"]
        if key in index:
            raise ValueError(f"Duplicate prescribed case: {key}")
        if key not in expected:
            raise ValueError(f"Unprescribed case: {key}")
        index[key] = row
        if row.get("execution_complete"):
            economic = row.get("economic_metrics") or {}
            if not finite_number(economic.get("unmet_kg")) or not finite_number(economic.get("net_operating_value_eur")):
                raise ValueError(f"Completed case lacks finite economic outcomes: {key}")
            settlements = (row.get("delivery") or {}).get("settlements")
            if settlements is not None:
                if len(settlements) != 28:
                    raise ValueError(f"Completed case lacks 28 delivery settlements: {key}")
                passed = all(d["unmet_kg"] <= 1e-9 for d in settlements)
                if passed != row.get("every_delivery_passed"):
                    raise ValueError(f"Delivery classification differs: {key}")
                if abs(sum(d["unmet_kg"] for d in settlements) - economic["unmet_kg"]) > 1e-6:
                    raise ValueError(f"Unmet kilograms do not reconcile: {key}")
            if bool(row.get("every_delivery_passed")) != (economic["unmet_kg"] <= 28e-9):
                raise ValueError(f"Week success contradicts unmet kilograms: {key}")
    if set(index) != expected:
        raise ValueError(f"Incomplete prescribed grid: expected {len(expected)}, got {len(index)}")
    # Pairing provenance is optional for synthetic unit fixtures but enforced when provided.
    for seed in SEEDS:
        for condition in CONDITIONS:
            panel = [index[(seed, condition, p)] for p in CONTROLLERS]
            for field in ("physical_solar_forcing_sha256", "paired_source_sha256"):
                values = [r.get(field) for r in panel]
                if any(v is not None for v in values) and (any(v is None for v in values) or len(set(values)) != 1):
                    raise ValueError(f"Paired controller forcing differs: {seed}/{condition}/{field}")
    return index


def analyze_rows(rows):
    index = validate_rows(rows)
    groups = [summarize_group([index[(s, c, p)] for s in SEEDS], label=c, controller=p)
              for c in CONDITIONS for p in CONTROLLERS]
    pooled = [summarize_group([index[(s, c, p)] for s in SEEDS for c in STRESS_CONDITIONS], label="pooled-stress", controller=p)
              for p in CONTROLLERS]
    families = [summarize_group([index[(s, c, p)] for s in SEEDS for c in conditions], label=family, controller=p)
                for family, conditions in FAMILIES.items() for p in CONTROLLERS]
    pairs = [("daylight", p) for p in CONTROLLERS[1:]] + [("storage", p) for p in CONTROLLERS[2:]]
    panels = [(c, (c,)) for c in CONDITIONS] + list(FAMILIES.items()) + [("pooled-stress", STRESS_CONDITIONS)]
    comparisons = [paired_comparison(index, reference=left, treatment=right, conditions=conditions, label=label)
                   for label, conditions in panels for left, right in pairs]
    primary = [comparison for comparison in comparisons if "primary_criterion" in comparison]
    guard_counts = [(r.get("physical_metrics") or {}).get("guard_interventions", 0) for r in rows]
    return {"schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
            "design": {"seeds": list(SEEDS), "controllers": list(CONTROLLERS), "conditions": list(CONDITIONS),
                       "n_base_scenarios": len(SEEDS), "n_rollouts": len(rows), "episode_hours": 168,
                       "stress_cells_per_controller": len(SEEDS) * len(STRESS_CONDITIONS),
                       "bootstrap_replicates": BOOTSTRAP_REPLICATES, "primary_confidence": ADJUSTED_CONFIDENCE,
                       "primary_comparison_count": PRIMARY_COMPARISONS,
                       "weighting": "Equal weight to the five prospectively prescribed stress conditions; all conditions clustered by base seed",
                       "interpretation_limit": "Synthetic challenge suite with no estimated real-world event frequency or population tail probability; all three frozen PPO policies reported separately"},
            "audits": {"complete_rollouts": sum(bool(r.get("execution_complete")) for r in rows),
                       "audited_rollouts": sum(row_audited(r) for r in rows),
                       "all_rollouts_complete_and_audited": all(row_audited(r) for r in rows),
                       "guard_interventions": sum(guard_counts),
                       "execution_failures": [{"seed": r["seed"], "condition": r["condition"], "controller": r["controller"], "failure": r.get("failure")} for r in rows if not r.get("execution_complete")],
                       "summed_case_wall_time_s": sum(r.get("wall_time_s", 0) for r in rows)},
            "groups": groups, "pooled_stress_groups": pooled, "family_groups": families,
            "comparisons": comparisons, "primary_claims": primary,
            "any_primary_claim_met": any(p["primary_criterion"]["met"] for p in primary)}


def report_markdown(analysis):
    lines = ["# Frozen-policy stress experiment", "", analysis["design"]["interpretation_limit"], "",
             "Six conditions share 16 base scenarios; five fixed controllers give 480 seven-day rollouts. Every controller manages its own inventories before the disruption. The control condition measures ordinary performance on these same base scenarios.", "",
             "## Condition results", "", "| Condition | Controller | Every delivery met | Mean unmet kg | Worst-quarter mean unmet kg | Mean modeled €/week |", "|---|---|---:|---:|---:|---:|"]
    def number(value, digits=1):
        return f"{value:,.{digits}f}" if value is not None else "unavailable"
    for group in analysis["groups"]:
        lines.append(f"| {group['panel']} | {group['controller']} | {group['all_delivery_successes']}/{group['n_cells']} | {number(group['metrics']['unmet_kg']['mean'])} | {number(group['worst_quarter']['mean_unmet_kg'])} | {number(group['metrics']['net_operating_value_eur']['mean'], 0)} |")
    lines += ["", "Worst-quarter loss is descriptive: the largest four unmet-kilogram outcomes among 16 scenarios in each row. It is not an estimate of real-world population CVaR.", "", "## Prespecified broad reliability criteria", "",
              "Each fixed PPO is compared separately with daylight and storage across all 80 stress cells. Passing requires at least eight rescues, zero regressions, no greater mean unmet kilograms, and a positive adjusted lower confidence bound for the all-delivery-success difference. All comparisons must complete and pass physical/economic audits.", "",
              "Intervals resample the 16 base seeds (all five conditions together), with 2,000 bootstrap replicates. The primary success intervals use Bonferroni adjustment across all six PPO/comparator comparisons (99.1667% two-sided intervals). They do not treat 80 cells as independent. Small-sample bootstrap uncertainty remains.", "",
              "| PPO | Reference | Rescues | Regressions | Success difference, pp [adjusted interval] | Mean unmet difference, kg | Criterion met |", "|---|---|---:|---:|---|---:|---|"]
    for p in analysis["primary_claims"]:
        estimate = p["primary_criterion"]["adjusted_success_difference"]
        vals = [estimate["mean"], *estimate["interval"]]
        interval = "unavailable" if any(v is None for v in vals) else f"{100*vals[0]:.1f} [{100*vals[1]:.1f}, {100*vals[2]:.1f}]"
        lines.append(f"| {p['treatment']} | {p['reference']} | {p['rescues']} | {p['regressions']} | {interval} | {number(p['metrics']['unmet_kg']['mean'])} | {'yes' if p['primary_criterion']['met'] else 'no'} |")
    lines += ["", "## Ordinary-operation contribution tradeoff", "", "| PPO | Reference | Mean contribution difference, €/week |", "|---|---|---:|"]
    for p in analysis["comparisons"]:
        if p["panel"] == "control" and p["treatment"].startswith("ppo-"):
            lines.append(f"| {p['treatment']} | {p['reference']} | {number(p['metrics']['net_operating_value_eur']['mean'], 0)} |")
    audits = analysis["audits"]
    lines += ["", "The contribution difference is treatment minus reference. No break-even disruption frequency is inferred because this suite assigns severities by design rather than sampling their real-world occurrence probabilities.", "", "## Audit and scope", "",
              f"Completed rollouts: {audits['complete_rollouts']}/480. Physical/economic audits passed: {audits['audited_rollouts']}/480. Added policy-guard interventions recorded: {audits['guard_interventions']}.", "",
              "All prescribed cases and all three policies are retained, including failed deliveries. No policy was retrained, selected, or altered from these stress outcomes. A successful comparison would apply to the specified synthetic suite and frozen policy, not establish global optimality or deployed reliability.", ""]
    return "\n".join(lines)


def write_analysis(rows, output, *, provenance=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = analyze_rows(rows)
    result["provenance"] = provenance or {}
    for name, content in (("analysis.json", json.dumps(result, indent=2, allow_nan=False) + "\n"),
                          ("REPORT.md", report_markdown(result))):
        with (output / name).open("x") as stream:
            stream.write(content)
    with (output / "outcomes.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("seed", "condition", "controller", "execution_complete", "every_delivery_passed", "unmet_kg", "net_operating_value_eur", "missed_deliveries"))
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["seed"], r["condition"], r["controller"])):
            writer.writerow({**{k: row.get(k) for k in ("seed", "condition", "controller", "execution_complete", "every_delivery_passed")},
                             **{k: metric(row, k) for k in ("unmet_kg", "net_operating_value_eur", "missed_deliveries")}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from docs.research.tools.tail_stress_v1.run import require_freeze
    from docs.research.tools.wholeplant_ppo_v1.portable import file_pin
    freeze = require_freeze(args.results / "freeze.json")
    paths = sorted((args.results / "cases").rglob("result.json"))
    rows = [json.loads(p.read_text()) for p in paths]
    if any(r.get("freeze_sha256") != freeze["sha256"] for r in rows):
        raise ValueError("Case and prospective freeze differ")
    result = write_analysis(rows, args.output or args.results / "analysis",
                            provenance={"freeze": file_pin(args.results / "freeze.json"),
                                        "case_results": [file_pin(p) for p in paths]})
    print(json.dumps({"cases": result["design"]["n_rollouts"], "all_audited": result["audits"]["all_rollouts_complete_and_audited"],
                      "primary_claims_met": sum(p["primary_criterion"]["met"] for p in result["primary_claims"])}))


if __name__ == "__main__":
    main()

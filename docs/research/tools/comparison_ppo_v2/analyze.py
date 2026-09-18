"""Paired, all-attempt analysis; seven-day operating value is not plant profit."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from docs.research.tools.free_ppo_v1 import analyze as v1
from docs.research.tools.comparison_ppo_v2.evaluate import SCHEMA, require_freeze

STOCKS = ("battery_energy_kwh", "battery_soc", "h2_level_kg", "co2_level_kg", "ch4_level_kg")
OPERATING = ("solar_available_kwh", "battery_charge_kwh", "battery_discharge_kwh", "electrolyser_kwh", "dac_kwh", "methanator_kwh", "zero_solar_electrolyser_kwh", "zero_solar_ch4_produced_kg", "solar_present_ch4_produced_kg", "zero_solar_battery_discharge_kwh", "electrolyser_on_minutes", "dac_on_minutes", "methanator_on_minutes")


def all_metrics():
    return (*v1.all_metrics(), *(f"initial:{k}" for k in STOCKS), *(f"final:{k}" for k in STOCKS), *(f"operating:{k}" for k in OPERATING))


def outcome_value(row, metric):
    if metric.startswith(("initial:", "final:", "operating:")):
        if not row["execution_complete"]:
            return None
        kind, field = metric.split(":", 1)
        return row.get(kind + ("_metrics" if kind == "operating" else "_state"), {}).get(field)
    return v1.outcome_value(row, metric)


def summarize_group(rows):
    passed = [r for r in rows if r["delivery_and_physics_passed"]]
    return {"split": rows[0]["split"], "controller": rows[0]["controller"], "condition": rows[0]["condition"], "variant": rows[0]["variant"],
        "n_scenarios": len(rows), "complete_weeks": sum(r["execution_complete"] for r in rows),
        "delivery_successes": sum(r["every_delivery_passed"] for r in rows), "physical_invariant_successes": sum(r["physical_invariants_passed"] for r in rows),
        "economic_audit_successes": sum(r["economic_audit"]["passed"] for r in rows), "delivery_and_physics_successes": len(passed),
        "failure_seeds": [r["seed"] for r in rows if not r["delivery_and_physics_passed"]],
        "metrics": {key: v1.estimates([outcome_value(r, key) for r in rows], [outcome_value(r, key) for r in passed]) for key in all_metrics()}}


def assert_paired(reference, treatment):
    if set(reference) != set(treatment):
        raise ValueError("Paired panels require identical scenario sets; no outcome intersection")
    for seed in reference:
        left, right = reference[seed], treatment[seed]
        if left.get("paired_source_sha256") and right.get("paired_source_sha256") and left["paired_source_sha256"] != right["paired_source_sha256"]:
            raise ValueError("Paired scenario weather, disturbances, or opening gas stocks differ")


def paired_comparison(reference, treatment, *, label, split, condition=None):
    assert_paired(reference, treatment)
    values = {}
    for metric in all_metrics():
        differences, conditional = [], []
        for seed in sorted(reference):
            left, right = outcome_value(reference[seed], metric), outcome_value(treatment[seed], metric)
            delta = right - left if left is not None and right is not None else None
            differences.append({"seed": seed, "delta": delta})
            if reference[seed]["delivery_and_physics_passed"] and treatment[seed]["delivery_and_physics_passed"]:
                conditional.append(delta)
        values[metric] = {**v1.estimates([r["delta"] for r in differences], conditional), "paired_values": differences}
    first = next(iter(reference))
    return {"split": split, "condition": condition, "comparison": label,
        "reference": reference[first]["controller"], "treatment": treatment[first]["controller"],
        "reference_condition": reference[first]["condition"], "treatment_condition": treatment[first]["condition"],
        "n_scenarios": len(reference), "both_complete": sum(reference[s]["execution_complete"] and treatment[s]["execution_complete"] for s in reference),
        "rescued_cases": [s for s in reference if not reference[s]["every_delivery_passed"] and treatment[s]["every_delivery_passed"]],
        "regressed_cases": [s for s in reference if reference[s]["every_delivery_passed"] and not treatment[s]["every_delivery_passed"]], "metrics": values}


def aggregate_comparison(by_controller, treatment_labels, reference_label, *, split, condition):
    reference = by_controller[reference_label]
    treatments = [by_controller[label] for label in treatment_labels]
    for treatment in treatments:
        assert_paired(reference, treatment)
    metrics = {}
    for metric in all_metrics():
        differences, conditional = [], []
        for seed in sorted(reference):
            left = outcome_value(reference[seed], metric)
            right = [outcome_value(t[seed], metric) for t in treatments]
            delta = statistics.mean(r - left for r in right) if left is not None and all(r is not None for r in right) else None
            differences.append({"seed": seed, "delta": delta})
            if reference[seed]["delivery_and_physics_passed"] and all(t[seed]["delivery_and_physics_passed"] for t in treatments):
                conditional.append(delta)
        metrics[metric] = {**v1.estimates([v["delta"] for v in differences], conditional), "paired_values": differences}
    return {"split": split, "condition": condition, "comparison": f"Mean of {','.join(treatment_labels)} vs {reference_label}",
        "reference": reference_label, "treatment_controllers": treatment_labels, "n_scenarios": len(reference),
        "bootstrap_unit": "Environmental scenario after averaging the fixed optimizer-policy differences", "metrics": metrics}


def validate_row(row):
    v1.validate_row(row)
    if row["execution_complete"]:
        if row["minute_audit"]["states_checked"] != 10081 or row["inference"]["verified_decisions"] != 672 or row["inference"]["actual_observation_inputs"] is not True:
            raise ValueError("Completed case lacks full minute or actual-inference evidence")
        if row["paired_source_sha256"] != v1.canonical_sha({k: v for k, v in row["paired_source_manifest"].items() if k != "paired_source_sha256"}):
            raise ValueError("Pairing manifest hash differs")
        delivered, unmet = row["economic_metrics"]["delivered_kg"], row["economic_metrics"]["unmet_kg"]
        if abs(delivered + unmet - row["variant"]["delivery_rate_kg_h"] * 168) > 1e-6:
            raise ValueError("Delivery demand differs from selected condition")
        if row["variant"]["battery_mode"] == "disabled" and (abs(row["operating_metrics"]["battery_charge_kwh"]) > 1e-8 or abs(row["operating_metrics"]["battery_discharge_kwh"]) > 1e-8):
            raise ValueError("Battery-disabled diagnostic contains actual battery use")


def collect(roots, *, require_complete_panels=True):
    if isinstance(roots, (str, Path)):
        roots = [roots]
    rows, summaries, freezes, freeze_pins, seen, pairings, within_condition = [], [], {}, {}, set(), {}, {}
    paths = sorted({p.resolve() for root in roots for p in Path(root).rglob("summary.json")})
    for path in paths:
        panel = json.loads(path.read_text())
        if panel.get("schema") != SCHEMA + "-panel" or panel["split"] == "train":
            continue
        freeze_pin = panel["evaluation_freeze"]
        if v1.file_pin(freeze_pin["path"]) != freeze_pin:
            raise ValueError("Panel freeze file changed")
        if freeze_pin["sha256"] not in freezes:
            freezes[freeze_pin["sha256"]] = require_freeze(freeze_pin["path"])
            freeze_pins[freeze_pin["sha256"]] = freeze_pin
        freeze = freezes[freeze_pin["sha256"]]
        if panel["evaluation_freeze_sha256"] != freeze["sha256"] or panel["source_files_unchanged_during_panel"] is not True:
            raise ValueError("Panel source or freeze differs")
        selected = [p for p in freeze["panels"] if (p["split"], p["condition"], p["controller"]) == (panel["split"], panel["condition"], panel["controller"])]
        if len(selected) != 1 or panel["requested_seeds"] != selected[0]["seeds"] or sorted(r["seed"] for r in panel["runs"]) != sorted(panel["requested_seeds"]):
            raise ValueError("Panel does not retain every prescribed outcome")
        summaries.append(v1.file_pin(path))
        for row in panel["runs"]:
            key = row["split"], row["condition"], row["controller"], row["seed"]
            if key in seen:
                raise ValueError(f"Duplicate outcome: {key}")
            seen.add(key)
            if key[:3] != (panel["split"], panel["condition"], panel["controller"]) or row["evaluation_freeze_sha256"] != freeze["sha256"]:
                raise ValueError("Case identity differs from panel")
            validate_row(row)
            pin = row.get("recording")
            if pin:
                if v1.file_pin(pin["path"]) != pin:
                    raise ValueError("Recording bytes changed")
                with gzip.open(pin["path"], "rt") as stream:
                    record = json.load(stream)
                if record["summary"] != {k: v for k, v in row.items() if k != "recording"}:
                    raise ValueError("Recording and compact summary differ")
                if record.get("minute_truth_states") and v1.canonical_sha(record["minute_truth_states"]) != row["retained_minute_truth_sha256"]:
                    raise ValueError("Prescribed full trace differs")
            if row.get("paired_source_sha256"):
                scenario = row["split"], row["seed"]
                if scenario in pairings and pairings[scenario] != row["paired_source_sha256"]:
                    raise ValueError("Conditions changed paired weather, faults, or gas stocks")
                pairings[scenario] = row["paired_source_sha256"]
                identity = row["exogenous_episode_sha256"], row["manifest_sha256"]
                exact_scenario = row["split"], row["condition"], row["seed"]
                if exact_scenario in within_condition and within_condition[exact_scenario] != identity:
                    raise ValueError("Controllers within a condition faced different cases")
                within_condition[exact_scenario] = identity
            rows.append(row)
    groups_by = defaultdict(list)
    for row in rows:
        groups_by[row["split"], row["condition"], row["controller"]].append(row)
    if require_complete_panels:
        for freeze in freezes.values():
            for panel in freeze["panels"]:
                if panel["split"] != "train" and sorted(r["seed"] for r in groups_by[panel["split"], panel["condition"], panel["controller"]]) != panel["seeds"]:
                    raise ValueError("A frozen panel has no complete set of retained outcomes")
    groups = [summarize_group(sorted(items, key=lambda r: r["seed"])) for _, items in sorted(groups_by.items()) if items]
    paired, aggregate, conditions = [], [], []
    split_conditions = sorted({(r["split"], r["condition"]) for r in rows})
    for split, condition in split_conditions:
        by = {label: {r["seed"]: r for r in items} for (sp, cond, label), items in groups_by.items() if (sp, cond) == (split, condition) and items}
        for baseline in ("daylight", "storage"):
            if baseline not in by:
                continue
            for treatment in sorted(by):
                if treatment == baseline or (treatment == "daylight" and baseline == "storage"):
                    continue
                paired.append(paired_comparison(by[baseline], by[treatment], label=f"{treatment} vs {baseline}", split=split, condition=condition))
            for prefix in ("ppo25-", "ppo30-"):
                labels = [prefix + str(seed) for seed in (101, 211, 307)]
                if all(label in by for label in labels):
                    aggregate.append(aggregate_comparison(by, labels, baseline, split=split, condition=condition))
        for seed in (101, 211, 307):
            old, new = f"ppo25-{seed}", f"ppo30-{seed}"
            if old in by and new in by:
                paired.append(paired_comparison(by[old], by[new], label=f"{new} vs {old}", split=split, condition=condition))
    pairs = (("d25-sampled", "d25-floor"), ("d30-sampled", "d30-floor"), ("d25-sampled", "d30-sampled"), ("d25-floor", "d30-floor"), ("d25-sampled", "d25-sampled-no-battery"))
    for split, label in sorted({(r["split"], r["controller"]) for r in rows}):
        for reference, treatment in pairs:
            left, right = groups_by.get((split, reference, label)), groups_by.get((split, treatment, label))
            if left and right:
                result = paired_comparison({r["seed"]: r for r in left}, {r["seed"]: r for r in right},
                    label=f"{label}: {treatment} vs {reference}", split=split)
                result["interpretation"] = "Frozen-policy dependence under channel removal; not an estimate of the best batteryless controller" if treatment.endswith("no-battery") else "Paired condition sensitivity; demand and opening reserves are explicit interventions"
                conditions.append(result)
    return {"schema": SCHEMA + "-analysis", "summaries": summaries,
        "freezes": [{"sha256": f["sha256"], "description": f["description"], "file": freeze_pins[key]} for key, f in freezes.items()],
        "analysis_source": v1.file_pin(__file__), "rows": rows, "groups": groups,
        "paired_comparisons": paired, "aggregate_comparisons": aggregate, "condition_comparisons": conditions,
        "bootstrap": {"seed": v1.BOOTSTRAP_SEED, "replicates": v1.BOOTSTRAP_REPLICATES,
            "unit": "Paired environmental scenario after averaging fixed policies where specified", "interval": "95% percentile, descriptive, not multiplicity adjusted"},
        "interpretation": "Every prescribed success and failure retained. Partial monetary outcomes are not imputed as weekly value. Seven days in a simplified digital twin; contribution is not calibrated profit. Intervals condition on fixed controllers; repeated policies do not increase environmental sample size."}


def write_csv(result, output):
    fields = ("split", "condition", "controller", "seed", *all_metrics())
    with Path(output).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({**{k: row[k] for k in fields[:4]}, **{key: outcome_value(row, key) for key in all_metrics()}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-pending-panels", action="store_true")
    args = p.parse_args()
    result = collect(args.root, require_complete_panels=not args.allow_pending_panels)
    args.output.mkdir(parents=True, exist_ok=False)
    v1.publish(args.output / "analysis.json", result)
    write_csv(result, args.output / "outcomes.csv")
    print(json.dumps({"outcomes": len(result["rows"]), "groups": len(result["groups"]), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()

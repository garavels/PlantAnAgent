"""Paired, all-attempt analysis; seven-day operating value is not plant profit."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from docs.research.tools.free_ppo_v1 import analyze as v1
from docs.research.tools.wholeplant_ppo_v1.evaluate import SCHEMA, require_freeze, ARMS, PROCESSES

STOCKS = ("battery_energy_kwh", "battery_soc", "h2_level_kg", "co2_level_kg", "ch4_level_kg", "dac_readiness_kwh", "methanator_readiness_kwh")
OPERATING = ("solar_available_kwh", "battery_charge_kwh", "battery_discharge_kwh",
    "zero_solar_ch4_produced_kg", "solar_present_ch4_produced_kg", "zero_solar_battery_discharge_kwh",
    "zero_solar_h2_produced_kg", "zero_solar_co2_captured_kg",
    *(u+s for u in PROCESSES for s in ("_kwh", "_on_minutes", "_battery_kwh")),
    *("zero_solar_"+u+"_kwh" for u in PROCESSES),
    *(u+s for u in ("dac", "methanator") for s in ("_startup_kwh", "_standby_kwh", "_productive_kwh", "_heat_loss_kwh", "_ready_minutes", "_starts", "_cold_starts")))



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
    return {"split": rows[0]["split"], "controller": rows[0]["controller"], "condition": rows[0]["condition"], "variant": rows[0]["variant"], "arm": rows[0]["condition"],
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


def verify_inference_witnesses(commands, row):
    """Check retained raw inputs, interventions and inference digests, no rollouts."""
    import numpy as np
    digest = hashlib.sha256()
    for index, command in enumerate(commands):
        evidence = command["evidence"]
        raw = np.asarray(evidence["observation"])
        action = np.asarray(command["action"])
        if command["decision_index"] != index or raw.shape != (149,) or action.shape != (7,) or not np.isfinite(raw).all() or not np.isfinite(action).all():
            raise ValueError("Malformed actual-input/action witness")
        if v1.canonical_sha(raw.tolist()) != evidence["observation_sha256"] or not np.array_equal(action, evidence["action"]):
            raise ValueError("Raw policy input/action witness changed")
        if row["controller_kind"] == "ppo":
            normalized = np.asarray(evidence["normalized_observation"])
            mean = np.asarray(evidence["raw_mean"])
            if normalized.shape != (149,) or mean.shape != (7,) or not np.isfinite(normalized).all() or not np.isfinite(mean).all():
                raise ValueError("Invalid normalized policy inference witness")
            if v1.canonical_sha(normalized.tolist()) != evidence["normalized_observation_sha256"] or not np.array_equal(np.clip(mean,-1,1), action):
                raise ValueError("Normalized input or clipped policy mean changed")
        else:
            baseline = evidence["baseline_inference"]
            transformed = np.asarray(baseline["controller_input"])
            if not np.array_equal(raw, baseline["actual_observation"]) or transformed.shape != (149,) or v1.canonical_sha(transformed.tolist()) != evidence["controller_input_sha256"]:
                raise ValueError("Baseline raw/transformed input evidence changed")
            expected = raw.copy()
            if row["controller"] == "storage-persistence":
                metadata = row["baseline_metadata"]
                indices = metadata["consumed_forecast_indices"]
                if not indices or len(indices) != len(set(indices)) or any(not 0 <= i < 149 for i in indices):
                    raise ValueError("Invalid registered forecast coordinates")
                expected[indices] = raw[metadata["current_solar_index"]]
            if not np.array_equal(transformed, expected):
                raise ValueError("Baseline transformation differs from registered persistence intervention")
        digest.update(json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    if digest.hexdigest() != row["inference"]["witness_sha256"]:
        raise ValueError("Actual-input inference stream hash differs")


def factorial_interaction(arms, *, controller, split):
    reference = arms["restricted-instant"]
    for cases in arms.values():
        assert_paired(reference, cases)
    metrics = {}
    weights = {"restricted-instant": 1, "shared-instant": -1, "restricted-dynamic": -1, "shared-dynamic": 1}
    for metric in all_metrics():
        differences, conditional = [], []
        for seed in sorted(reference):
            values = {arm: outcome_value(arms[arm][seed], metric) for arm in ARMS}
            delta = sum(weights[arm]*value for arm, value in values.items()) if all(v is not None for v in values.values()) else None
            differences.append({"seed": seed, "delta": delta})
            if all(arms[arm][seed]["delivery_and_physics_passed"] for arm in ARMS):
                conditional.append(delta)
        metrics[metric] = {**v1.estimates([v["delta"] for v in differences], conditional), "paired_values": differences}
    return {"controller": controller, "split": split, "n_scenarios": len(reference),
        "comparison": "(shared-dynamic minus restricted-dynamic) minus (shared-instant minus restricted-instant)",
        "weights": weights, "metrics": metrics,
        "interpretation": "Interaction of routing and readiness under a fixed baseline algorithm; not reoptimized architectural value"}


def forecast_result(forecast, persistence):
    comparison = paired_comparison(persistence, forecast, label="storage forecast minus storage persistence", split="held_out", condition="shared-dynamic")
    estimate = comparison["metrics"]["net_operating_value_eur"]["all_case"]
    complete = all(r["execution_complete"] and r["physical_invariants_passed"] and r["economic_audit"]["passed"] for r in [*forecast.values(), *persistence.values()])
    rescue_count, regression_count = len(comparison["rescued_cases"]), len(comparison["regressed_cases"])
    no_lower = sum(r["every_delivery_passed"] for r in forecast.values()) >= sum(r["every_delivery_passed"] for r in persistence.values())
    reliability = complete and rescue_count >= 4 and regression_count == 0
    contribution = bool(complete and no_lower and estimate["mean"] is not None and estimate["mean"] >= 250 and estimate["ci95"][0] > 0)
    comparison["criterion"] = {"all_comparisons_complete_and_audited": complete,
        "forecast_rescues": rescue_count, "forecast_regressions": regression_count,
        "no_lower_delivery_count": no_lower, "reliability_bar_met": reliability,
        "contribution_bar_met": contribution, "practical_evidence_criterion_met": reliability or contribution,
        "definition": "At least four rescued weeks with no regression OR at least EUR250/week with positive paired95% lower bound and no lower delivery count; all comparisons audited"}
    comparison["interpretation"] = "Incremental forecast information value within the fixed storage rule, not universal planning or PPO necessity; same 32 shared environmental scenarios"
    return comparison


def validate_row(row):
    v1.validate_row(row)
    if row["execution_complete"]:
        if row["minute_audit"]["states_checked"] != 10081 or row["inference"]["verified_decisions"] != 672 or row["inference"]["actual_observation_inputs"] is not True:
            raise ValueError("Completed case lacks full minute or actual-inference evidence")
        if row["paired_source_sha256"] != v1.canonical_sha({k: v for k, v in row["paired_source_manifest"].items() if k != "paired_source_sha256"}):
            raise ValueError("Pairing manifest hash differs")
        if row["economic_audit"].get("all_destinations_independently_reconciled") is not True:
            raise ValueError("Whole-plant battery wear was not reconciled from independent minute flows")
        delivered, unmet = row["economic_metrics"]["delivered_kg"], row["economic_metrics"]["unmet_kg"]
        if abs(delivered + unmet - row["variant"]["delivery_rate_kg_h"] * 168) > 1e-6:
            raise ValueError("Delivery demand differs from selected condition")
        if row["condition"].startswith("restricted") and any(abs(row["operating_metrics"][u+"_battery_kwh"]) > 1e-8 for u in ("dac", "methanator")):
            raise ValueError("Restricted architectural arm used downstream battery energy")
        discharge = sum(row["operating_metrics"][u+"_battery_kwh"] for u in PROCESSES)
        if abs(discharge-row["operating_metrics"]["battery_discharge_kwh"]) > 1e-6:
            raise ValueError("Whole-plant discharge does not sum destinations")
        if row["condition"].endswith("instant") and any(abs(row["operating_metrics"][u+s]) > 1e-8 for u in ("dac", "methanator") for s in ("_startup_kwh", "_standby_kwh", "_heat_loss_kwh")):
            raise ValueError("Instantaneous arm contains readiness energy")


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
                if row["execution_complete"]:
                    if len(record["decision_trace"]) != 672:
                        raise ValueError("Complete case lacks decision witnesses")
                    if row["seed"] == selected[0]["seeds"][0] and len(record.get("minute_truth_states") or []) != 10081:
                        raise ValueError("First prescribed case lacks its complete minute trace")
                    verify_inference_witnesses(record["decision_trace"], row)
            if row["execution_complete"] and not pin:
                raise ValueError("Complete case has no retained recording")
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
    if not rows:
        raise ValueError("No recognized development or held-out panels found")
    groups_by = defaultdict(list)
    for row in rows:
        groups_by[row["split"], row["condition"], row["controller"]].append(row)
    if require_complete_panels:
        for freeze in freezes.values():
            for panel in freeze["panels"]:
                if panel["split"] != "train" and sorted(r["seed"] for r in groups_by[panel["split"], panel["condition"], panel["controller"]]) != panel["seeds"]:
                    raise ValueError("A frozen panel has no complete set of retained outcomes")
    groups = [summarize_group(sorted(items, key=lambda r: r["seed"])) for _, items in sorted(groups_by.items()) if items]
    paired, aggregate, conditions, factorial, learning = [], [], [], [], []
    split_conditions = sorted({(r["split"], r["condition"]) for r in rows})
    for split, condition in split_conditions:
        by = {label: {r["seed"]: r for r in items} for (sp, cond, label), items in groups_by.items() if (sp, cond) == (split, condition) and items}
        for baseline in ("daylight", "storage"):
            if baseline not in by:
                continue
            for treatment in sorted(by):
                if treatment == baseline or (treatment == "daylight" and baseline == "storage"):
                    continue
                paired.append(paired_comparison(by[baseline], by[treatment], label=f"{treatment} minus {baseline}", split=split, condition=condition))
            labels = [label for label, cases in by.items() if next(iter(cases.values())).get("controller_kind") == "ppo" and next(iter(cases.values())).get("phase") == "final"]
            if sorted(next(iter(by[label].values()))["optimizer_seed"] for label in labels) == [101, 211, 307]:
                aggregate.append(aggregate_comparison(by, sorted(labels), baseline, split=split, condition=condition))
        if split == "development":
            for optimizer_seed in (101, 211, 307):
                phases = {next(iter(cases.values())).get("phase"): label for label, cases in by.items() if next(iter(cases.values())).get("controller_kind") == "ppo" and next(iter(cases.values())).get("optimizer_seed") == optimizer_seed}
                if "initial" in phases and "final" in phases:
                    comparison = paired_comparison(by[phases["initial"]], by[phases["final"]], label=f"Optimizer {optimizer_seed}: final minus initial", split=split, condition=condition)
                    comparison["interpretation"] = "Development-only learning check; actor and normalization change jointly; not independent held-out evidence"
                    learning.append(comparison)
    arm_pairs = (("restricted-instant", "shared-instant", "shared_bus_without_readiness"),
                 ("restricted-dynamic", "shared-dynamic", "shared_bus_with_readiness"),
                 ("restricted-instant", "restricted-dynamic", "readiness_without_shared_bus"),
                 ("shared-instant", "shared-dynamic", "readiness_with_shared_bus"))
    for split in sorted({r["split"] for r in rows}):
        for label in ("daylight", "storage"):
            arms = {arm: {r["seed"]: r for r in groups_by.get((split, arm, label), [])} for arm in ARMS}
            for reference, treatment, effect in arm_pairs:
                if arms[reference] and arms[treatment]:
                    result = paired_comparison(arms[reference], arms[treatment], label=f"{label}: {treatment} minus {reference}", split=split)
                    result.update(effect=effect, interpretation="Physical-arm effect under the same fixed baseline algorithm; not independently reoptimized architectural value")
                    conditions.append(result)
            if all(arms.values()):
                factorial.append(factorial_interaction(arms, controller=label, split=split))
    forecast = None
    fore = groups_by.get(("held_out", "shared-dynamic", "storage"))
    pers = groups_by.get(("held_out", "shared-dynamic", "storage-persistence"))
    if fore and pers:
        forecast = forecast_result({r["seed"]: r for r in fore}, {r["seed"]: r for r in pers})
    return {"schema": SCHEMA + "-analysis", "summaries": summaries,
        "freezes": [{"sha256": f["sha256"], "description": f["description"], "file": freeze_pins[key]} for key, f in freezes.items()],
        "analysis_source": v1.file_pin(__file__), "rows": rows, "groups": groups,
        "paired_comparisons": paired, "aggregate_comparisons": aggregate, "condition_comparisons": conditions,
        "factorial_contrasts": factorial, "learning_comparisons": learning, "forecast_comparison": forecast,
        "bootstrap": {"seed": v1.BOOTSTRAP_SEED, "replicates": v1.BOOTSTRAP_REPLICATES,
            "unit": "Paired environmental scenario after averaging fixed policies where specified", "interval": "95% percentile, descriptive, not multiplicity adjusted"},
        "interpretation": "Every prescribed success and failure retained. Partial monetary outcomes are not imputed as weekly value. Four assumed physical arms, with fresh PPO trained only on shared-dynamic. Baseline contrasts condition on the fixed algorithms. Seven days in a simplified digital twin; contribution is not calibrated profit. Intervals condition on fixed controllers; repeated policies do not increase environmental sample size."}


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

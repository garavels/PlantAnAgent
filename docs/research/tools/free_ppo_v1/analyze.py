"""Read-only, paired-scenario analysis of direct-control PPO outcomes.

Daily inventory returns and forced stops are descriptive, never success gates.
Partial execution is retained as failure for full-week flags; partial money is
never substituted for a complete week. Intervals condition on fixed policies.
"""
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

SCHEMA = "free-ppo-v1-evaluation"
BASELINE = "coverage-direct"
UNTRAINED = "untrained-101"
RESOURCES = ("battery", "h2", "co2", "ch4")
BOOTSTRAP_SEED = 20260917
BOOTSTRAP_REPLICATES = 10000
BINARY = ("execution_complete", "every_delivery_passed", "physical_invariants_passed", "delivery_and_physics_passed")
PHYSICAL = ("delivered_kg", "unmet_kg", "ch4_produced_kg", "delivery_fill_rate", "curtailed_kwh", "battery_throughput_kwh", "electrolyser_cold_starts", "electrolyser_hot_starts", "electrolyser_aborted_starts", "electrolyser_forced_stops", "electrolyser_controlled_stops", "electrolyser_operating_time_s", "guard_interventions", "guard_intervention_kwh", "request_application_gap_kwh", "min_battery_soc", "h2_inventory_change_kg", "co2_inventory_change_kg", "ch4_inventory_change_kg", "battery_energy_change_kwh")
ECONOMIC = ("net_operating_value_eur", "battery_equivalent_full_cycles", "electrolyser_equivalent_operating_h", "terminal_ch4_equivalent_deficit_kg", "terminal_battery_deficit_kwh")
COMPONENTS = ("delivered_service_value", "unmet_delivery_penalty", "solar_opportunity_cost", "battery_throughput_reserve", "electrolyser_life_reserve", "methane_endowment_liability_change", "battery_endowment_liability_change")


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def bootstrap_mean(values, *, seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES):
    import numpy as np
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not np.isfinite(data).all():
        raise ValueError("Expected finite one-dimensional scenario measurements")
    if not len(data):
        return {"n_scenarios": 0, "mean": None, "ci95": None}
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), size=(replicates, len(data)))].mean(axis=1)
    return {"n_scenarios": len(data), "mean": float(data.mean()), "ci95": [float(x) for x in np.quantile(means, [.025, .975])]}


def all_metrics():
    return (*BINARY, *PHYSICAL, *ECONOMIC, *(f"component:{k}" for k in COMPONENTS), *(f"stock:{resource}:{stat}" for resource in RESOURCES for stat in ("sum", "max")), "daily_returns_met", "controller_latency_s")


def outcome_value(row, metric):
    if metric in BINARY:
        return float(row[metric])
    if not row["execution_complete"]:
        return None
    if metric.startswith("component:"):
        return row["economic_metrics"]["components_eur"].get(metric.split(":", 1)[1])
    if metric.startswith("stock:"):
        _, resource, statistic = metric.split(":")
        return row["stock_deficits"][resource][statistic]
    if metric == "daily_returns_met":
        return row["daily_returns_met"]
    if metric == "controller_latency_s":
        return row.get("controller_latency_s", {}).get("mean")
    for field in ("physical_metrics", "economic_metrics"):
        if metric in row.get(field, {}):
            return float(row[field][metric])
    return None


def estimates(values, conditional):
    observed = [v for v in values if v is not None]
    return {"all_case": bootstrap_mean(observed) if len(observed) == len(values) else {"n_scenarios": len(values), "mean": None, "ci95": None, "reason": "Missing complete-week measurements; partial values not imputed"},
            "complete_observed": bootstrap_mean(observed),
            "delivery_and_physics_passed_only": bootstrap_mean([v for v in conditional if v is not None]),
            "missing_cases": len(values) - len(observed)}


def summarize_group(rows):
    passed = [r for r in rows if r["delivery_and_physics_passed"]]
    return {"split": rows[0]["split"], "controller": rows[0]["controller"], "n_scenarios": len(rows),
            "complete_weeks": sum(r["execution_complete"] for r in rows),
            "delivery_successes": sum(r["every_delivery_passed"] for r in rows),
            "physical_invariant_successes": sum(r["physical_invariants_passed"] for r in rows),
            "delivery_and_physics_successes": len(passed),
            "failure_seeds": [r["seed"] for r in rows if not r["delivery_and_physics_passed"]],
            "metrics": {metric: estimates([outcome_value(r, metric) for r in rows], [outcome_value(r, metric) for r in passed]) for metric in all_metrics()}}


def paired_comparison(reference, treatment, *, label, split):
    if set(reference) != set(treatment):
        raise ValueError("Paired panels must have identical scenario identities")
    seeds = sorted(reference)
    paired = [{"seed": seed, "both_complete": reference[seed]["execution_complete"] and treatment[seed]["execution_complete"],
               "reference_passed": reference[seed]["delivery_and_physics_passed"], "treatment_passed": treatment[seed]["delivery_and_physics_passed"]} for seed in seeds]
    metrics = {}
    for metric in all_metrics():
        values, conditional = [], []
        for pair in paired:
            seed = pair["seed"]
            a, b = outcome_value(reference[seed], metric), outcome_value(treatment[seed], metric)
            delta = b - a if a is not None and b is not None else None
            values.append({"seed": seed, "delta": delta})
            if pair["reference_passed"] and pair["treatment_passed"]:
                conditional.append(delta)
        metrics[metric] = {**estimates([r["delta"] for r in values], conditional), "paired_values": values}
    return {"split": split, "comparison": label, "n_scenarios": len(seeds), "both_complete": sum(r["both_complete"] for r in paired),
            "both_delivery_and_physics_passed": sum(r["reference_passed"] and r["treatment_passed"] for r in paired),
            "rescued_cases": [r["seed"] for r in paired if not r["reference_passed"] and r["treatment_passed"]],
            "regressed_cases": [r["seed"] for r in paired if r["reference_passed"] and not r["treatment_passed"]], "metrics": metrics, "paired": paired}


def aggregate_comparison(by_controller, treatment_labels, reference_label, split):
    """Cross optimizer realizations within a scenario, never pool them as n=96."""
    reference = by_controller[reference_label]
    treatments = [by_controller[name] for name in treatment_labels]
    if any(set(t) != set(reference) for t in treatments):
        raise ValueError("Optimizer panels do not share the same scenarios")
    metrics = {}
    for metric in all_metrics():
        values, conditional = [], []
        for seed in sorted(reference):
            a = outcome_value(reference[seed], metric)
            bs = [outcome_value(t[seed], metric) for t in treatments]
            delta = statistics.mean(b - a for b in bs) if a is not None and all(b is not None for b in bs) else None
            values.append({"seed": seed, "delta": delta})
            if reference[seed]["delivery_and_physics_passed"] and all(t[seed]["delivery_and_physics_passed"] for t in treatments):
                conditional.append(delta)
        metrics[metric] = {**estimates([r["delta"] for r in values], conditional), "paired_values": values}
    return {"split": split, "comparison": f"trained mean vs {reference_label}", "reference": reference_label, "treatment_controllers": treatment_labels,
            "n_scenarios": len(reference), "bootstrap_unit": "environmental scenario after averaging fixed optimizer differences", "metrics": metrics}


def validate_row(row):
    complete = row["execution_complete"]
    if complete and (row["decisions"] != 672 or row["physical_steps"] != 10080):
        raise ValueError("Claimed complete week has wrong duration")
    delivery = complete and row["delivery_settlements"] == 28 and row.get("maximum_delivery_unmet_kg", float("inf")) <= 1e-9
    if row["every_delivery_passed"] != delivery:
        raise ValueError("Delivery result disagrees with recorded settlements")
    physical = complete and row["minute_audit"]["all_physical_invariants_passed"]
    if row["physical_invariants_passed"] != physical or row["delivery_and_physics_passed"] != (delivery and physical):
        raise ValueError("Qualification improperly classified")
    e = row.get("economic_metrics", {})
    if e and abs(e["net_operating_value_eur"] - sum(e["components_eur"].values())) > 1e-6:
        raise ValueError("Economic components do not sum")
    if complete and not row["economic_audit"]["passed"]:
        raise ValueError("Complete economic accounting failed independent audit")


def collect(root, *, freeze_path, require_complete_panel=False):
    freeze = json.loads(Path(freeze_path).read_text())
    if freeze["sha256"] != canonical_sha({k:v for k,v in freeze.items() if k != "sha256"}):
        raise ValueError("Evaluation freeze digest differs")
    if not any(Path(pin["path"]).resolve()==Path(__file__).resolve() for pin in freeze["sources"]):
        raise ValueError("Analysis rules were not frozen")
    for pin in freeze["sources"]:
        if file_pin(pin["path"])!=pin:raise ValueError("A frozen comparison source changed")
    rows, summary_pins, seen, identities = [], [], set(), {}
    for path in sorted(Path(root).rglob("summary.json")):
        panel = json.loads(path.read_text())
        if panel.get("schema") != SCHEMA + "-panel" or panel["split"] == "train":
            continue
        if panel.get("evaluation_freeze_sha256") != freeze["sha256"] or panel.get("source_files_unchanged_during_panel") is not True:
            raise ValueError(f"Panel source/freeze mismatch: {path}")
        if sorted(r["seed"] for r in panel["runs"]) != sorted(panel["requested_seeds"]):
            raise ValueError(f"Panel omits requested outcomes: {path}")
        summary_pins.append(file_pin(path))
        for row in panel["runs"]:
            key = row["split"], row["controller"], row["seed"]
            if key in seen:
                raise ValueError(f"Duplicate outcome: {key}")
            seen.add(key)
            validate_row(row)
            pin = row.get("recording")
            if pin:
                if file_pin(pin["path"]) != pin:
                    raise ValueError("Recording bytes changed")
                with gzip.open(pin["path"], "rt") as stream:
                    record = json.load(stream)
                if record["summary"] != {k:v for k,v in row.items() if k != "recording"}:
                    raise ValueError("Recording and summary differ")
                if record.get("minute_truth_states") and canonical_sha(record["minute_truth_states"]) != row["retained_minute_truth_sha256"]:
                    raise ValueError("Selected minute trace hash differs")
            identity = (row.get("exogenous_episode_sha256"),row.get("manifest_sha256"))
            if identity[0]:
                scenario = row["split"], row["seed"]
                if scenario in identities and identities[scenario] != identity:
                    raise ValueError("Controllers faced different exogenous realizations")
                identities[scenario] = identity
            rows.append(row)
    groups_by = defaultdict(list)
    for row in rows:
        groups_by[row["split"], row["controller"]].append(row)
    selection = freeze["selection"]
    if require_complete_panel:
        for label in selection["primary_controllers"]:
            actual = sorted(r["seed"] for r in groups_by["held_out",label])
            if actual != selection["held_out_seeds"]:
                raise ValueError(f"Missing full primary panel: {label}")
    groups = [summarize_group(sorted(items, key=lambda r:r["seed"])) for _,items in sorted(groups_by.items()) if items]
    paired, aggregated = [], []
    trained = selection["trained_controllers"]
    for split in ("development", "held_out"):
        by = {label:{r["seed"]:r for r in items} for (sp,label),items in groups_by.items() if sp == split and items}
        for reference in (BASELINE, UNTRAINED):
            if reference not in by:
                continue
            for label in trained + ([UNTRAINED] if reference == BASELINE else []):
                if label in by:
                    paired.append(paired_comparison(by[reference],by[label],label=f"{label} vs {reference}",split=split))
            if all(label in by for label in trained):
                aggregated.append(aggregate_comparison(by,trained,reference,split))
    curves = []
    for label in trained:
        points = []
        for g in groups:
            if g["split"] == "development" and (g["controller"] == label or g["controller"].startswith(label + "-step-")):
                sample = groups_by["development",g["controller"]][0]
                # A launch/load failure has no observed checkpoint metadata, but
                # still belongs to its prospectively requested curve point.
                transitions = sample["checkpoint_timesteps"]
                if transitions is None:
                    transitions = int(g["controller"].rsplit("-step-", 1)[1]) if "-step-" in g["controller"] else selection["final_timesteps"]
                points.append({"transitions": transitions, "controller":g["controller"], "n_scenarios":g["n_scenarios"], "metrics":{k:g["metrics"][k]["all_case"] for k in ("every_delivery_passed","net_operating_value_eur","unmet_kg","electrolyser_forced_stops","terminal_ch4_equivalent_deficit_kg")}})
        if points:
            curves.append({"controller":label,"points":sorted(points,key=lambda p:p["transitions"])})
    return {"schema":SCHEMA+"-analysis", "evaluation_freeze":file_pin(freeze_path), "evaluation_freeze_sha256":freeze["sha256"],
            "analysis_source":file_pin(__file__), "summaries":summary_pins, "rows":rows, "groups":groups, "paired_comparisons":paired, "aggregate_comparisons":aggregated, "learning_curves":curves,
            "bootstrap":{"unit":"paired environmental scenario", "seed":BOOTSTRAP_SEED,"replicates":BOOTSTRAP_REPLICATES,"interval":"95% percentile, descriptive, not multiplicity adjusted"},
            "interpretation":"Daily stock returns and forced stops are descriptive, not success gates. Monetary results cover only complete weeks. Paired intervals condition on fixed trained policies and omit retraining uncertainty; optimizer replications do not increase environmental n."}


def plot_results(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    groups=[g for g in result["groups"] if g["split"]=="held_out"]
    groups.sort(key=lambda g:(g["controller"] != BASELINE,g["controller"] != UNTRAINED,g["controller"]))
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout="constrained")
    for ax,metric,title in zip(axes,("delivered_kg","net_operating_value_eur","electrolyser_forced_stops"),("Delivered methane (kg/week)","Modeled contribution (EUR/week)","Forced electrolyser stops/week")):
        for i,g in enumerate(groups):
            item=g["metrics"][metric]["all_case"]
            if item["mean"] is not None:
                lo,hi=item["ci95"]; mean=item["mean"]
                ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt="o",color="#226B73",capsize=3)
            else: ax.text(.02,i,"Incomplete measurements",transform=ax.get_yaxis_transform(),fontsize=8)
        ax.set_yticks(range(len(groups)),[g["controller"] for g in groups] if ax is axes[0] else [])
        ax.invert_yaxis(); ax.set_title(title,fontsize=10);ax.grid(axis="x",alpha=.2)
    fig.savefig(output/"held-out-outcomes.png",dpi=180);fig.savefig(output/"held-out-outcomes.pdf");plt.close(fig)
    if result["learning_curves"]:
        fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout="constrained")
        for curve in result["learning_curves"]:
            for ax,key,title in zip(axes,("net_operating_value_eur","unmet_kg"),("Development contribution (EUR/week)","Development unmet methane (kg/week)")):
                points=[p for p in curve["points"] if p["metrics"][key]["mean"] is not None]
                ax.plot([p["transitions"] for p in points],[p["metrics"][key]["mean"] for p in points],marker="o",label=curve["controller"])
                ax.set_title(title,fontsize=10);ax.set_xlabel("Training transitions");ax.grid(alpha=.2)
        axes[0].legend(fontsize=7);fig.savefig(output/"development-learning-curves.png",dpi=180);fig.savefig(output/"development-learning-curves.pdf");plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--freeze",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--require-complete-panel",action="store_true")
    parser.add_argument("--plots",action="store_true")
    args=parser.parse_args()
    result=collect(args.root,freeze_path=args.freeze,require_complete_panel=args.require_complete_panel)
    args.output.mkdir(parents=True,exist_ok=True);publish(args.output/"analysis.json",result)
    with (args.output/"results.csv").open("x",newline="") as stream:
        names=("split","controller","seed",*all_metrics());writer=csv.DictWriter(stream,fieldnames=names);writer.writeheader()
        for row in result["rows"]:
            writer.writerow({**{k:row[k] for k in ("split","controller","seed")},**{k:outcome_value(row,k) for k in all_metrics()}})
    if args.plots:plot_results(result,args.output/"figures")
    print(json.dumps({"analysis":str(args.output/"analysis.json"),"outcomes":len(result["rows"]),"groups":len(result["groups"])}))


if __name__ == "__main__":
    main()

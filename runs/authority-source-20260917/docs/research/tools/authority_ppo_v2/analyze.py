"""Hash-verified paired scenario analysis; no environment resets or policy changes.

The independent environmental unit is a scenario (32 held out). Optimizer seeds
are reported separately and averaged *within* each scenario for arm summaries.
All-case means remain unavailable when any full-week measurement is missing.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Keep reporting-only imports independent of the simulation/training runtime.
SCHEMA = "authority-ppo-v2-evaluation"
ARMS = ("narrow", "broad")
OPTIMIZER_SEEDS = (101, 211, 307)
RESOURCES = ("battery", "h2", "co2", "ch4")
COVERAGE_LABEL = "daily-coverage-offgrid-v1"
MPC_LABEL = "daily-return-mission-mpc-v1"


def evaluator():
    from docs.research.tools.authority_ppo_v2 import evaluate
    return evaluate


def benchmark():
    return evaluator().benchmark()


def controller_label(*args):
    return evaluator().controller_label(*args)


def summarize_record(*args):
    return evaluator().summarize_record(*args)


def publish(*args, **kwargs):
    return evaluator().publish(*args, **kwargs)


def canonical_sha(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_pin(path):
    import hashlib
    raw = Path(path).read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


BOOTSTRAP_SEED = 20260917
BOOTSTRAP_REPLICATES = 10000
BINARY_METRICS = ("mission_passed", "execution_complete", "every_delivery_passed", "all_daily_returns_passed", "hard_safety_passed")
PHYSICAL_METRICS = ("unmet_kg", "delivered_kg", "curtailed_kwh", "battery_throughput_kwh", "electrolyser_cold_starts", "electrolyser_hot_starts", "electrolyser_forced_stops", "electrolyser_controlled_stops", "electrolyser_absolute_load_movement_kw", "guard_interventions", "request_application_gap_kwh")
ECONOMIC_METRICS = ("net_operating_value_eur", "battery_equivalent_full_cycles", "electrolyser_equivalent_operating_h")


def bootstrap_mean(values: list[float], *, seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES) -> dict:
    """Percentile interval; one input value per paired environmental scenario."""
    import numpy as np
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not np.isfinite(data).all():
        raise ValueError("Bootstrap requires finite one-dimensional scenario values")
    if not len(data):
        return {"n_scenarios": 0, "mean": None, "ci95": None}
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), size=(replicates, len(data)))].mean(axis=1)
    return {"n_scenarios": len(data), "mean": float(data.mean()), "ci95": [float(v) for v in np.quantile(means, [0.025, 0.975])]}


def outcome_value(row: dict, metric: str) -> float | None:
    if metric in BINARY_METRICS:
        return float(row[metric])
    if not row["execution_complete"]:
        return None
    if metric.startswith("stock_"):
        _, resource, statistic = metric.split("_")
        return float(row["stock_deficits"][resource][statistic])
    if metric == "controller_latency_s":
        return row.get("controller_latency_s", {}).get("mean")
    for field in ("physical_metrics", "economic_metrics"):
        if metric in row.get(field, {}):
            return float(row[field][metric])
    return None


def all_metrics() -> tuple[str, ...]:
    return (*BINARY_METRICS, *PHYSICAL_METRICS, *ECONOMIC_METRICS,
            *(f"stock_{resource}_{stat}" for resource in RESOURCES for stat in ("sum", "max")), "controller_latency_s")


def summarize_group(rows: list[dict]) -> dict:
    complete = [row for row in rows if row["execution_complete"]]
    passed = [row for row in rows if row["mission_passed"]]
    metrics = {}
    for metric in all_metrics():
        values = [outcome_value(row, metric) for row in rows]
        observed = [value for value in values if value is not None]
        conditional = [outcome_value(row, metric) for row in passed]
        conditional = [value for value in conditional if value is not None]
        metrics[metric] = {
            "all_case": bootstrap_mean(observed) if len(observed) == len(rows) else {"n_scenarios": len(rows), "mean": None, "ci95": None, "reason": "Missing full-week measurements; not imputed"},
            "complete_observed": bootstrap_mean(observed),
            "mission_passed_only": bootstrap_mean(conditional),
            "missing_cases": len(rows) - len(observed),
        }
    return {"split": rows[0]["split"], "controller": rows[0]["controller"], "n_scenarios": len(rows),
            "complete_weeks": len(complete), "full_missions": len(passed),
            "failure_seeds": [row["seed"] for row in rows if not row["mission_passed"]],
            "metrics": metrics}


def paired_comparison(reference: dict[int, dict], treatment: dict[int, dict], *, label: str, split: str) -> dict:
    seeds = sorted(set(reference) & set(treatment))
    paired = [{"seed": seed, "reference_passed": reference[seed]["mission_passed"],
               "treatment_passed": treatment[seed]["mission_passed"],
               "both_complete": reference[seed]["execution_complete"] and treatment[seed]["execution_complete"],
               "both_passed": reference[seed]["mission_passed"] and treatment[seed]["mission_passed"]} for seed in seeds]
    metrics = {}
    for metric in all_metrics():
        values = []
        for row in paired:
            a, b = outcome_value(reference[row["seed"]], metric), outcome_value(treatment[row["seed"]], metric)
            values.append({**row, "delta": b - a if a is not None and b is not None else None})
        observed = [row["delta"] for row in values if row["delta"] is not None]
        conditional = [row["delta"] for row in values if row["both_passed"] and row["delta"] is not None]
        metrics[metric] = {"all_case": bootstrap_mean(observed) if len(observed) == len(seeds) else {"n_scenarios": len(seeds), "mean": None, "ci95": None, "reason": "Incomplete pair; no full-week value imputed"},
                           "complete_observed_pairs": bootstrap_mean(observed), "both_missions_passed": bootstrap_mean(conditional),
                           "paired_values": [{"seed": row["seed"], "delta": row["delta"]} for row in values]}
    return {"split": split, "comparison": label, "n_scenarios": len(seeds),
            "both_complete": sum(row["both_complete"] for row in paired), "both_passed": sum(row["both_passed"] for row in paired),
            "rescued_missions": [row["seed"] for row in paired if not row["reference_passed"] and row["treatment_passed"]],
            "regressed_missions": [row["seed"] for row in paired if row["reference_passed"] and not row["treatment_passed"]],
            "metrics": metrics, "paired": paired}


def arm_comparison(by_controller: dict[str, dict[int, dict]], arm: str, reference_arm: str | None, split: str) -> dict:
    """Average the three optimizer deltas within a scenario before resampling."""
    treatment = [by_controller.get(controller_label("ppo", arm, seed), {}) for seed in OPTIMIZER_SEEDS]
    reference = ([by_controller.get(controller_label("ppo", reference_arm, seed), {}) for seed in OPTIMIZER_SEEDS]
                 if reference_arm else [by_controller.get(COVERAGE_LABEL, {})] * len(OPTIMIZER_SEEDS))
    common = sorted(set.intersection(*(set(items) for items in treatment + reference)))
    metrics = {}
    for metric in all_metrics():
        data, missing, mutual_passed = [], [], []
        for scenario in common:
            pairs = [(outcome_value(a[scenario], metric), outcome_value(b[scenario], metric)) for a, b in zip(reference, treatment)]
            if any(a is None or b is None for a, b in pairs):
                missing.append(scenario)
            else:
                value = statistics.mean(b - a for a, b in pairs)
                data.append({"seed": scenario, "delta": value})
                if all(a[scenario]["mission_passed"] and b[scenario]["mission_passed"] for a, b in zip(reference, treatment)):
                    mutual_passed.append(value)
        result = bootstrap_mean([row["delta"] for row in data])
        metrics[metric] = {"all_case": result if not missing else {"n_scenarios": len(common), "mean": None, "ci95": None, "reason": "At least one optimizer run lacks a full week"},
                           "complete_observed_scenarios": result,
                           "mutual_all_optimizer_seeds_pass": bootstrap_mean(mutual_passed),
                           "missing_scenarios": missing, "paired_values": data}
    return {"split": split, "comparison": f"{arm} mean across optimizer seeds vs {reference_arm or 'coverage'}", "arm": arm,
            "reference_arm": reference_arm, "n_scenarios": len(common), "optimizer_seeds": list(OPTIMIZER_SEEDS),
            "bootstrap_unit": "scenario after averaging three paired optimizer differences", "metrics": metrics}


def learning_curves(groups: list[dict]) -> list[dict]:
    """Prospective diagnostic endpoints, separate from final policy selection."""
    by_name = {group["controller"]: group for group in groups if group["split"] == "development"}
    curves = []
    for arm in ARMS:
        for optimizer in OPTIMIZER_SEEDS:
            label = controller_label("ppo", arm, optimizer)
            points = []
            for transitions in (65536, 131072, 196608, 262144):
                group = by_name.get(label if transitions == 262144 else f"{label}-step-{transitions}")
                if group:
                    points.append({"transitions": transitions, "n_scenarios": group["n_scenarios"],
                        "full_missions": group["full_missions"], "complete_weeks": group["complete_weeks"],
                        "metrics": {name: group["metrics"][name]["all_case"] for name in
                                    ("mission_passed", "net_operating_value_eur", "stock_ch4_max", "battery_equivalent_full_cycles", "electrolyser_equivalent_operating_h")}})
            if points:
                curves.append({"arm": arm, "optimizer_seed": optimizer, "points": points,
                    "interpretation": "Same eight development scenarios at prospectively fixed budgets; diagnostic only, no checkpoint selection."})
    return curves


def validate_record(record: dict, expected_row: dict, contract: dict) -> dict:
    if record.get("schema") != SCHEMA + "-recording":
        raise ValueError("Foreign recording schema")
    if record["benchmark_contract_sha256"] != contract["contract_sha256"]:
        raise ValueError("Recording belongs to another benchmark")
    if record["seed"] not in contract["splits"][record["split"]]:
        raise ValueError("Scenario belongs to another split")
    if record["split"] == "held_out" and not record.get("evaluation_freeze_sha256"):
        raise ValueError("Held-out recording has no freeze")
    for key in ("controller", "seed", "split", "evaluation_freeze_sha256"):
        if record[key] != expected_row[key]:
            raise ValueError(f"Published outcome identity differs: {key}")
    full = record["execution_complete"]
    if full and (record["decisions"] != 672 or record["physical_steps"] != 10080 or len(record["minute_truth_states"]) != 10081):
        raise ValueError("Claimed complete execution has an incomplete trajectory")
    if canonical_sha(record["minute_truth_states"]) != record["minute_truth_sha256"]:
        raise ValueError("Minute trajectory hash differs")
    daily = full and len(record["cycle_settlements"]) == 7 and all(row["feasible"] for row in record["cycle_settlements"])
    deliveries = full and len(record["delivery"]["settlements"]) == 28 and all(row["unmet_kg"] <= 1e-9 for row in record["delivery"]["settlements"])
    if record["all_daily_returns_passed"] != daily or record["every_delivery_passed"] != deliveries:
        raise ValueError("Delivery or daily-return full-week classification differs")
    if record["mission_passed"] != (full and record["hard_safety_passed"] and daily and deliveries):
        raise ValueError("Mission classification differs")
    if not full and record["hard_safety_passed"]:
        raise ValueError("Partial execution was granted a full-week safety pass")
    e = record["economic_metrics"]
    if abs(e["net_operating_value_eur"] - sum(e["components_eur"].values())) > 1e-7:
        raise ValueError("Economic components do not sum")
    if e["components_eur"].get("solar_opportunity_cost", 0) != 0:
        raise ValueError("Unexpected off-grid solar charge")
    actual = summarize_record(record)
    for key, value in actual.items():
        if expected_row.get(key) != value:
            raise ValueError(f"Compact summary differs from recording: {key}")
    if record["controller"].startswith("authority-ppo-v2-"):
        if actual["inference"]["verified_decisions"] != record["decisions"] or not actual["inference"]["all_inputs_verified"]:
            raise ValueError("PPO decisions lack complete actual-observation inference witnesses")
    return actual


def trace_excerpt(record: dict) -> dict:
    """Decision-resolution artifact for predefined replay cases and fault analysis."""
    fields = ("battery_energy_kwh", "battery_soc", "h2_level_kg", "co2_level_kg", "ch4_level_kg", "solar_available_kw", "electrolyser_power_kw", "dac_power_kw", "methanator_power_kw")
    frames = [{"decision_index": frame["decision_index"], **{key: frame["state"][key] for key in fields}} for frame in record["decision_frames"]]
    return {"seed": record["seed"], "controller": record["controller"], "split": record["split"], "frames": frames,
            "cycle_settlements": record["cycle_settlements"], "faults": record["manifest"]["disturbances"]["schedule"],
            "mission_passed": record["mission_passed"], "failure": record["failure"]}


def collect(root: Path, *, require_complete_panel=False) -> dict:
    contract = benchmark().load_benchmark_contract()
    rows, summaries, traces = [], [], []
    seen, scenario_identity = set(), {}
    freeze_ids = set()
    for path in sorted(root.rglob("summary.json")):
        report = json.loads(path.read_text())
        if report.get("schema") != SCHEMA + "-panel" or report.get("split") == "train":
            continue
        if report.get("source_files_unchanged_during_panel") is not True or report.get("validity_failure"):
            raise ValueError(f"Panel is unfinished or source changed: {path}")
        if report["benchmark_contract_sha256"] != contract["contract_sha256"]:
            raise ValueError(f"Panel task differs: {path}")
        if sorted(row["seed"] for row in report["runs"]) != sorted(report["requested_seeds"]):
            raise ValueError(f"Panel omits requested outcomes: {path}")
        summaries.append(file_pin(path))
        if report["split"] == "held_out":
            freeze_ids.add(report["evaluation_freeze_sha256"])
        for item in report["runs"]:
            key = (item["split"], item["controller"], item["seed"])
            if key in seen:
                raise ValueError(f"Duplicate scenario/controller outcome: {key}")
            seen.add(key)
            row = dict(item)
            if item.get("recording"):
                pin = item["recording"]
                recording_path = Path(pin["path"])
                if file_pin(recording_path) != pin:
                    raise ValueError(f"Recording hash differs: {recording_path}")
                with gzip.open(recording_path, "rt") as stream:
                    record = json.load(stream)
                validate_record(record, row, contract)
                identity = (row["manifest_sha256"], row["beta_exogenous_episode_sha256"])
                scenario = (row["split"], row["seed"])
                if scenario in scenario_identity and scenario_identity[scenario] != identity:
                    raise ValueError(f"Controllers saw different scenario realizations: {scenario}")
                scenario_identity[scenario] = identity
                if row["split"] == "held_out" and row["seed"] in contract["splits"]["held_out"][:2] and row["controller"] in (COVERAGE_LABEL, controller_label("ppo", "narrow", 101), controller_label("ppo", "broad", 101)):
                    traces.append(trace_excerpt(record))
            elif any(row.get(field) for field in BINARY_METRICS):
                raise ValueError("Unrecorded runner failure cannot receive a pass")
            rows.append(row)
    if len(freeze_ids) > 1 or None in freeze_ids:
        raise ValueError("Held-out results come from different or absent freezes")
    if require_complete_panel:
        for split in ("development", "held_out"):
            expected = set(contract["splits"][split])
            for label in (COVERAGE_LABEL, *(controller_label("ppo", arm, opt) for arm in ARMS for opt in OPTIMIZER_SEEDS)):
                observed = {row["seed"] for row in rows if row["split"] == split and row["controller"] == label}
                if observed != expected:
                    raise ValueError(f"Incomplete required panel: {split}/{label}")
        expected_mpc = set(contract["splits"]["held_out"][:8])
        observed_mpc = {row["seed"] for row in rows if row["split"] == "held_out" and row["controller"] == MPC_LABEL}
        if observed_mpc != expected_mpc:
            raise ValueError("Incomplete strict MPC supplementary panel")
    return {"rows": rows, "summaries": summaries, "traces": traces, "benchmark_contract_sha256": contract["contract_sha256"], "evaluation_freeze_sha256": next(iter(freeze_ids), None)}


def analyze(root: Path, *, require_complete_panel=False) -> dict:
    collected = collect(root, require_complete_panel=require_complete_panel)
    rows = collected["rows"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["controller"])].append(row)
    groups = [summarize_group(sorted(items, key=lambda row: row["seed"])) for key, items in sorted(grouped.items())]
    paired, arms = [], []
    for split in ("development", "held_out"):
        by_controller = {label: {row["seed"]: row for row in items} for (panel_split, label), items in grouped.items() if panel_split == split}
        coverage = by_controller.get(COVERAGE_LABEL, {})
        for label, treatment in sorted(by_controller.items()):
            if label != COVERAGE_LABEL and coverage:
                paired.append(paired_comparison(coverage, treatment, label=f"{label} vs coverage", split=split))
        for opt in OPTIMIZER_SEEDS:
            narrow = by_controller.get(controller_label("ppo", "narrow", opt), {})
            broad = by_controller.get(controller_label("ppo", "broad", opt), {})
            if narrow and broad:
                paired.append(paired_comparison(narrow, broad, label=f"broad vs narrow optimizer {opt}", split=split))
        if all(controller_label("ppo", arm, opt) in by_controller for arm in ARMS for opt in OPTIMIZER_SEEDS) and coverage:
            arms.extend(arm_comparison(by_controller, arm, None, split) for arm in ARMS)
            arms.append(arm_comparison(by_controller, "broad", "narrow", split))
    return {"schema": SCHEMA + "-analysis", **collected, "groups": groups, "paired_comparisons": paired, "arm_comparisons": arms,
            "learning_curves": learning_curves(groups),
            "analysis_source": file_pin(Path(__file__).resolve()),
            "bootstrap": {"unit": "environmental scenario", "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "interval": "95% percentile; descriptive, not multiplicity adjusted"},
            "interpretation": "32 held-out environmental replications, with three optimizer seeds nested within each arm. Partial executions are failures for full-week outcomes. Economic comparisons include an all-case estimate only when every case completed; conditional success comparisons are explicitly labeled and cannot repair mission failures. Contribution excludes uncalibrated solar capital and fixed O&M; it is not full business profit.",
            "mission_failures": [{"split": row["split"], "controller": row["controller"], "seed": row["seed"], "failure": row["failure"], "execution_complete": row["execution_complete"], "stock_deficits": row.get("stock_deficits"), "unmet_kg_observed": row.get("physical_metrics", {}).get("unmet_kg")} for row in rows if not row["mission_passed"]]}


def export_csv(result: dict, output: Path):
    columns = ("split", "controller", "n_scenarios", "complete_weeks", "full_missions", *all_metrics())
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for group in result["groups"]:
            writer.writerow({**{key: group[key] for key in columns[:5]}, **{metric: group["metrics"][metric]["all_case"]["mean"] for metric in all_metrics()}})


def plot(result: dict, output: Path) -> list[Path]:
    """Imports matplotlib lazily; can run in a separate reporting environment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180})
    output.mkdir(parents=True, exist_ok=True)
    groups = [group for group in result["groups"] if group["split"] == "held_out" and group["controller"] != MPC_LABEL]
    labels = ["Coverage" if g["controller"] == COVERAGE_LABEL else g["controller"].replace("authority-ppo-v2-", "").replace("-", " · ") for g in groups]
    colors = ["#55645a" if g["controller"] == COVERAGE_LABEL else ("#217d94" if "narrow" in g["controller"] else "#a06432") for g in groups]
    paths = []
    if groups:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), layout="constrained")
        x = np.arange(len(groups))
        rates = [g["full_missions"] / g["n_scenarios"] for g in groups]
        axes[0].bar(x, rates, color=colors)
        axes[0].set_ylim(0, 1.09)
        axes[0].set_ylabel("Complete mission fraction")
        for i, group in enumerate(groups):
            axes[0].text(i, rates[i]+0.025, f'{group["full_missions"]}/{group["n_scenarios"]}', ha="center", fontsize=9)
        axes[0].set_title("Reliability across the same 32 weeks")
        deltas = {item["comparison"].replace(" vs coverage", ""): item["metrics"]["net_operating_value_eur"]["all_case"] for item in result["paired_comparisons"] if item["split"] == "held_out"}
        for i, group in enumerate(groups):
            item = {"mean": 0, "ci95": [0, 0]} if group["controller"] == COVERAGE_LABEL else deltas.get(group["controller"])
            if item and item["mean"] is not None:
                lo, hi = item["ci95"]
                axes[1].errorbar(i, item["mean"], yerr=[[item["mean"]-lo], [hi-item["mean"]]], fmt="o", color=colors[i], capsize=4)
        axes[1].axhline(0, color="#999999", linewidth=0.8)
        axes[1].set_ylabel("Paired change in contribution (€ / week)")
        axes[1].set_title("All-case economics; 95% scenario bootstrap")
        for axis in axes:
            axis.set_xticks(x, labels, rotation=50, ha="right")
        path = output / "held-out-reliability-economics.png"
        fig.savefig(path, bbox_inches="tight"); plt.close(fig); paths.append(path)
        fig, axis = plt.subplots(figsize=(9, 4.2), layout="constrained")
        matrix = []
        allseeds = sorted({row["seed"] for row in result["rows"] if row["split"] == "held_out" and row["controller"] == COVERAGE_LABEL})
        lookup = {(row["controller"], row["seed"]): row for row in result["rows"] if row["split"] == "held_out"}
        for group in groups:
            matrix.append([int(lookup[(group["controller"], seed)]["mission_passed"]) for seed in allseeds])
        from matplotlib.colors import ListedColormap
        axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap=ListedColormap(["#c87b5a", "#d3e6dc"]))
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xticks(range(len(allseeds)), [str(seed)[-2:] for seed in allseeds], fontsize=8)
        axis.set_xlabel("Held-out scenario suffix (seed 99600xx)")
        axis.set_title("Complete mission by scenario — green passes, rust fails")
        path = output / "held-out-mission-matrix.png"
        fig.savefig(path, bbox_inches="tight"); plt.close(fig); paths.append(path)
    traces = [trace for trace in result["traces"] if trace["seed"] == 9960000]
    if traces:
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, layout="constrained")
        fields = [("ch4_level_kg", "Methane stock (kg)"), ("battery_energy_kwh", "Battery energy (kWh)"), ("dac_power_kw", "DAC power (kW)")]
        for trace in traces:
            label = "Coverage" if trace["controller"] == COVERAGE_LABEL else trace["controller"].replace("authority-ppo-v2-", "")
            for axis, (field, title) in zip(axes, fields):
                axis.plot([frame["decision_index"] / 4 for frame in trace["frames"]], [frame[field] for frame in trace["frames"]], label=label, linewidth=1.1)
                axis.set_ylabel(title)
            for fault in trace["faults"] if trace == traces[0] else []:
                for axis in axes:
                    axis.axvspan(fault["start_s"] / 3600, (fault["start_s"] + fault["duration_s"]) / 3600, color="#d6ac6b", alpha=.2)
        axes[0].legend(ncol=3, frameon=False)
        axes[0].set_title("Prospectively selected scenario 9960000; optimizer seed 101")
        axes[-1].set_xlabel("Elapsed simulation hours; shading marks declared DAC outages")
        path = output / "preselected-policy-trajectories.png"
        fig.savefig(path, bbox_inches="tight"); plt.close(fig); paths.append(path)
    curves = result.get("learning_curves", [])
    if any(len(curve["points"]) > 1 for curve in curves):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        styles = {101: "-", 211: "--", 307: ":"}
        for curve in curves:
            points = curve["points"]
            color = "#217d94" if curve["arm"] == "narrow" else "#a06432"
            label = f'{curve["arm"]} · {curve["optimizer_seed"]}'
            for axis, metric in zip(axes, ("mission_passed", "net_operating_value_eur")):
                usable = [point for point in points if point["metrics"][metric]["mean"] is not None]
                axis.plot([point["transitions"] / 1024 for point in usable], [point["metrics"][metric]["mean"] for point in usable],
                          marker="o", color=color, linestyle=styles[curve["optimizer_seed"]], label=label)
                axis.set_xlabel("Training transitions (× 1024)")
                axis.set_xticks([64, 128, 192, 256])
        axes[0].set_ylim(0, 1.03)
        axes[0].set_ylabel("Complete mission fraction on eight development weeks")
        axes[1].set_ylabel("Mean full-week contribution (€)")
        axes[1].legend(frameon=False, fontsize=8, ncol=2)
        fig.suptitle("Fixed-budget development diagnostics; final selection unchanged")
        path = output / "development-learning-curves.png"
        fig.savefig(path, bbox_inches="tight"); plt.close(fig); paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete-panel", action="store_true")
    parser.add_argument("--plot-from", type=Path, help="Generate figures from existing compact analysis JSON without simulator or training imports")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot_from:
        result = json.loads(args.plot_from.read_text())
        print(json.dumps({"figures": [str(path) for path in plot(result, args.output)]}))
        return
    if args.root is None:
        parser.error("Analysis requires --root")
    result = analyze(args.root, require_complete_panel=args.require_complete_panel)
    publish(args.output / "analysis.json", result, exclusive=False)
    export_csv(result, args.output / "results.csv")
    if args.plots:
        plot(result, args.output)
    print(json.dumps({"outcomes": len(result["rows"]), "groups": len(result["groups"]), "analysis": str(args.output / "analysis.json")}))


if __name__ == "__main__":
    main()

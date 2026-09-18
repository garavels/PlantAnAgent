"""Data-bound tables, figures and two-column note for the comparison study."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[4]
CONDITIONS = ("d25-sampled", "d25-floor", "d30-sampled", "d30-floor")
CONTROLLERS = ("daylight", "storage", "ppo25-101", "ppo25-211", "ppo25-307", "ppo30-101", "ppo30-211", "ppo30-307")
LABELS = {"daylight": "Daylight", "storage": "Storage", **{f"ppo{rate}-{seed}": f"PPO {rate}/{seed}" for rate in (25, 30) for seed in (101, 211, 307)}}
LABELS["daylight-persistence"] = "Persistence*"


def estimate(group, metric):
    return group["metrics"][metric]["all_case"]


def number(value, digits=0):
    return "unavailable" if value is None else f"{value:,.{digits}f}"


def prepare(analysis_path: Path, decision_path: Path, output: Path, forecast_analysis_path: Path | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis = json.loads(analysis_path.read_text())
    decision = json.loads(decision_path.read_text())
    learning = []
    rows = [r for r in analysis["rows"] if r["split"] == "held_out"]
    groups = {(g["controller"], g["condition"]): g for g in analysis["groups"] if g["split"] == "held_out"}
    supplement = None
    if forecast_analysis_path is not None:
        supplement = json.loads(forecast_analysis_path.read_text())
        if (supplement.get("supplementary") is not True
                or supplement["reference_main_analysis"]["sha256"] != hashlib.sha256(analysis_path.read_bytes()).hexdigest()
                or len(supplement["groups"]) != 1):
            raise ValueError("Supplementary result does not match the completed main analysis")
        added = supplement["rows"]
        if (sorted(r["seed"] for r in added) != list(range(20200000, 20200032))
                or any((r["split"], r["controller"], r["condition"]) != ("held_out", "daylight-persistence", "d30-sampled") for r in added)):
            raise ValueError("Supplement must retain the complete prescribed persistence panel")
        group = supplement["groups"][0]
        if (group["controller"], group["condition"], group["n_scenarios"]) != ("daylight-persistence", "d30-sampled", 32):
            raise ValueError("Supplementary group differs from its prescribed panel")
        rows.extend(added)
        groups["daylight-persistence", "d30-sampled"] = group
    required = {(c, k) for c in CONTROLLERS[:5] for k in CONDITIONS}
    if decision["train_new_policies"]:
        required.update((c, k) for c in CONTROLLERS[5:] for k in CONDITIONS[2:])
    for key in required:
        group = groups.get(key)
        cases = [r for r in rows if (r["controller"], r["condition"]) == key]
        if group is None or group["n_scenarios"] != 32 or sorted(r["seed"] for r in cases) != list(range(20200000, 20200032)):
            raise ValueError(f"Missing or duplicated prescribed outcomes: {key}")
        if group["delivery_successes"] != sum(r["every_delivery_passed"] for r in cases):
            raise ValueError("Group counts disagree with retained outcomes")
    ordered = [c for c in CONTROLLERS if any((c, k) in groups for k in CONDITIONS)]
    if supplement:
        ordered.insert(1, "daylight-persistence")
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    table = []
    for controller in ordered:
        counts = [f"{groups[controller, condition]['delivery_successes']}/32" if (controller, condition) in groups else "--" for condition in CONDITIONS]
        table.append(f"{LABELS[controller]} & " + " & ".join(counts) + r"\\")
    baseline = {c: {k: groups[c, k]["delivery_successes"] for k in CONDITIONS} for c in ("daylight", "storage")}
    old_30 = [groups[c, "d30-sampled"]["delivery_successes"] for c in CONTROLLERS[2:5]]
    abstract = (f"At 30\\,kg/h with sampled opening charge, daylight and storage control meet every delivery in "
                f"{baseline['daylight']['d30-sampled']} and {baseline['storage']['d30-sampled']} of 32 weeks, respectively.")
    if decision["train_new_policies"]:
        training = json.loads((decision_path.parent / "training/plan.json").read_text())
        results = [json.loads((decision_path.parent / f"training/seed-{seed}/result.json").read_text()) for seed in (101, 211, 307)]
        if any(r["status"] != "completed" or r["num_timesteps"] != training["total_timesteps"] for r in results):
            raise ValueError("Cannot present incomplete new training as final")
        for seed in (101, 211, 307):
            paired = []
            for label in (f"ppo30-initial-{seed}", f"ppo30-{seed}"):
                cases = {r["seed"]: r for r in analysis["rows"] if
                         (r["split"], r["controller"], r["condition"]) == ("development", label, "d30-sampled")}
                if sorted(cases) != list(range(20100000, 20100008)):
                    raise ValueError(f"Missing prescribed development learning cases for {label}")
                paired.append(cases)
            initial, final = paired
            deltas = [final[s]["economic_metrics"]["net_operating_value_eur"] - initial[s]["economic_metrics"]["net_operating_value_eur"]
                      for s in initial] if all(r["execution_complete"] for cases in paired for r in cases.values()) else []
            learning.append({"optimizer_seed": seed, "environmental_scenarios": 8,
                "initial_delivery_successes": sum(r["every_delivery_passed"] for r in initial.values()),
                "final_delivery_successes": sum(r["every_delivery_passed"] for r in final.values()),
                "final_minus_initial_eur_mean": statistics.mean(deltas) if deltas else None})
        train_text = (f"The declared criterion triggered three fresh runs at 30\\,kg/h, each receiving {training['total_timesteps']:,} transitions "
                      "from random weights with sampled opening charge.")
        new_30 = [groups[c, "d30-sampled"]["delivery_successes"] for c in CONTROLLERS[5:]]
        abstract += f" Fresh 30\\,kg/h policies succeed in {min(new_30)}--{max(new_30)} weeks."
        if min(new_30) == max(new_30):
            abstract = abstract.replace(f"{min(new_30)}--{max(new_30)} weeks", f"{new_30[0]} weeks each")
        if all(estimate(groups['daylight', 'd30-sampled'], 'net_operating_value_eur')['mean'] >
               estimate(groups[c, 'd30-sampled'], 'net_operating_value_eur')['mean'] for c in CONTROLLERS[5:]):
            abstract += " Daylight control nevertheless earns higher mean modeled contribution than every fresh policy."
    else:
        train_text = "The declared decision rule did not trigger further production training; the original policies are retained unchanged."
    supplement_methods = ""
    if supplement:
        abstract += (" A matched persistence comparison meets the declared threshold for practical forecast value."
                     if supplement['criterion']['practical_evidence_bar_met'] else
                     " A matched persistence comparison does not establish the declared practical forecast benefit.")
        supplement_methods = ("A supplementary test, declared after development but before held-out access, replaces the daylight controller's three forecast inputs with current-solar persistence. "
            "All inventory and commitment rules remain fixed. It reuses the same 32 higher-demand, sampled-charge scenarios to test incremental forecast value, without adding independent weather evidence. "
            "Its practical evidence criterion is at least four forecast-rescued delivery weeks with no regressions, or no lower delivery-success count and at least EUR\\,250/week advantage with a positive paired 95\\% lower bound. Forecast means and lower-tail estimates are ablated together.")
    macros = {"AbstractFinding": abstract, "TrainingDecision": train_text, "ResultTableRows": "\n".join(table),
              "SupplementMethods": supplement_methods}
    (output / "results.tex").write_text("\n".join(f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in macros.items()) + "\n")
    primary_rows = [r for r in rows if (r["controller"], r["condition"]) in required]
    complete = sum(r["execution_complete"] for r in primary_rows)
    physical = sum(r["physical_invariants_passed"] for r in primary_rows)
    findings = [f"Table~\\ref{{tab:results}} retains every prescribed primary outcome. Of {len(primary_rows)} controller-condition evaluations, {complete} complete the week and {physical} pass all minute-level physical checks."]
    if learning:
        initial_counts = ", ".join(str(item["initial_delivery_successes"]) for item in learning)
        final_counts = ", ".join(str(item["final_delivery_successes"]) for item in learning)
        findings.append(f"In the development-only learning check, initial policies meet every delivery in {initial_counts} of eight weeks; final policies do so in {final_counts}, respectively (optimizer seeds 101, 211, 307). These are not additional held-out observations.")
    for condition, phrase in (("d25-sampled", "the original demand and sampled charge"), ("d30-sampled", "the higher demand and sampled charge")):
        values = [f"{LABELS[c]}: EUR\\,{number(estimate(groups[c, condition], 'net_operating_value_eur')['mean'])}" for c in ("daylight", "storage")]
        findings.append(f"At {phrase}, baseline mean weekly contributions are " + "; ".join(values) + ".")
    if supplement:
        criterion = supplement["criterion"]
        forecast_stats = criterion["forecast_minus_persistence_eur"]
        ci = forecast_stats["ci95"]
        interval_text = "unavailable" if ci is None else f"[{number(ci[0])}, {number(ci[1])}]"
        counts = groups["daylight-persistence", "d30-sampled"]["delivery_successes"]
        conclusion = "met" if criterion["practical_evidence_bar_met"] else "not met"
        findings.append(f"The supplementary persistence controller succeeds in {counts}/32 weeks. Forecast minus persistence contribution is EUR\\,{number(forecast_stats['mean'])}/week (paired 95\\% interval {interval_text}). The declared practical evidence criterion is {conclusion}.")
    (output / "findings.tex").write_text("\n\n".join(findings) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.35, 2.25), layout="constrained")
    for i, controller in enumerate(ordered):
        stats = estimate(groups[controller, "d30-sampled"], "net_operating_value_eur")
        if stats["mean"] is None or stats["ci95"] is None:
            ax.text(0, i, "Incomplete: mean unavailable", fontsize=6)
            continue
        mean, (lo, hi) = stats["mean"], stats["ci95"]
        color = "#A3793C" if controller == "daylight-persistence" else "#667580" if controller in ("daylight", "storage") else "#176D72" if controller.startswith("ppo25") else "#5E6798"
        ax.plot([lo / 1000, hi / 1000], [i, i], color=color, linewidth=1.6)
        ax.scatter([mean / 1000], [i], color=color, s=15, zorder=3)
    ax.set_yticks(range(len(ordered)), [LABELS[c] for c in ordered])
    ax.invert_yaxis()
    ax.set_xlabel("Contribution (thousand EUR/week)")
    ax.axvline(0, color="#999999", linewidth=.6)
    ax.grid(axis="x", alpha=.2)
    fig.savefig(output / "figures/contribution-comparison.pdf")
    fig.savefig(output / "figures/contribution-comparison.png", dpi=200)
    plt.close(fig)
    shutil.copyfile(Path(__file__).with_name("paper-template.tex"), output / "plantanagent-comparison-ppo.tex")

    md = ["# When does learned control improve on daylight production?", "", "Filippo Gombac and Vittorio Garavelli", "",
          "All episodes remain seven days. The 32 held-out weather/fault scenarios are shared across controllers and conditions.", "",
          "| Controller | 25, sampled | 25, floor | 30, sampled | 30, floor |", "|---|---:|---:|---:|---:|"]
    for controller in ordered:
        md.append("| " + LABELS[controller] + " | " + " | ".join(f"{groups[controller, k]['delivery_successes']}/32" if (controller, k) in groups else "Not prescribed" for k in CONDITIONS) + " |")
    if supplement:
        md += ["", "*Persistence is a supplementary comparator declared after development and before held-out access. It replaces three forecast inputs with current sunlight while preserving the daylight controller's stock and commitment rules. It shares the same 32 environmental scenarios.", "",
               "Its predeclared practical criterion is either at least four forecast-rescued delivery weeks with no delivery regressions, or no lower delivery-success count and at least EUR 250/week forecast advantage with the lower paired 95% interval above zero.", "",
               f"Forecast rescues: {criterion['forecast_rescues']}; regressions: {criterion['forecast_regressions']}. Forecast minus persistence: EUR {number(forecast_stats['mean'])}/week, paired 95% interval {interval_text}. Evidence criterion: {conclusion}.", "",
               "This isolates the bundled forecast inputs' contribution within one fixed controller. It does not isolate forecast means from uncertainty estimates, establish the best forecastless controller, or prove universal planning or RL superiority."]
    md += ["", f"{complete}/{len(primary_rows)} primary evaluations completed; {physical}/{len(primary_rows)} passed all minute-level physical checks.", "",
           "The original PPO25 policies are transferred to higher demand without changing their weights or normalizers. Every PPO floor-opening evaluation, including PPO30, changes the initial-charge distribution used in training: all policies were trained with sampled 25–65% opening charge. Floor sensitivity therefore mixes changed initial energy with policy transfer to a different starting state; it is not an isolated proof of a physical energy requirement. The daylight baseline does not use the battery; the storage baseline can use it. Both use the same public observation and direct 15-minute action interface as PPO.", "",
           "Disabling both battery control channels tests policy dependence on the interface, not the isolated value of stored electricity. Charge/support netting can route solar directly to electrolysis; the intervention removes that route too. It does not test the optimal batteryless controller. The low-charge opening experiment measures initial-energy dependence, not indefinite sustainability. Individual gas stocks matter even if methane-equivalent inventory is preserved.", "",
           "The plant remains synthetic: only electrolysis has detailed startup/standby/shutdown dynamics. Carbon capture and methanation are memoryless; battery electricity reaches only electrolysis. Wear and inventory prices are assumptions. Monetary contributions exclude fixed costs and are not calibrated profit.", ""]
    if learning:
        md += ["## Development-only learning check", "",
               "Initial random policies and their final trained versions use their respective frozen observation normalizers. All three optimizer seeds are compared on the same eight development scenarios; these cases are not fresh test evidence. Positive contribution differences favor the trained policy.", "",
               "| Optimizer seed | Initial delivery weeks | Final delivery weeks | Paired mean gain, EUR/week |",
               "|---|---:|---:|---:|"]
        for item in learning:
            md.append(f"| {item['optimizer_seed']} | {item['initial_delivery_successes']}/8 | {item['final_delivery_successes']}/8 | {number(item['final_minus_initial_eur_mean'])} |")
    md += ["## Paired contribution differences", "",
           "Positive differences favor the three-policy PPO mean over the named baseline. Each interval resamples environmental scenarios after averaging the three fixed policies within a scenario. It does not treat optimizer seeds as independent weather samples or estimate uncertainty from new training runs.", "",
           "| Condition | Policies | Baseline | Mean difference, EUR/week | Paired 95% interval |",
           "|---|---|---|---:|---:|"]
    for comparison in analysis["aggregate_comparisons"]:
        if comparison["split"] != "held_out":
            continue
        stats = comparison["metrics"]["net_operating_value_eur"]["all_case"]
        interval = stats["ci95"]
        ci_text = "unavailable" if interval is None else f"[{number(interval[0])}, {number(interval[1])}]"
        md.append(f"| {comparison['condition']} | {', '.join(comparison['treatment_controllers'])} | {comparison['reference']} | {number(stats['mean'])} | {ci_text} |")
    md += ["", "## Every controller and condition", "",
           "All prescribed held-out cases are included. Contribution is the modeled objective after all penalties. Energy values are measured actual flows, not requested control values.", "",
           "| Condition | Controller | Full weeks | All deliveries | Accounting audits | EUR/week | Battery in/out, kWh | Zero-solar electrolysis, kWh |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for (controller, condition), group in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        mean = lambda metric: number(estimate(group, metric)["mean"])
        md.append(f"| {condition} | {controller} | {group['complete_weeks']}/32 | {group['delivery_successes']}/32 | {group['economic_audit_successes']}/32 | {mean('net_operating_value_eur')} | {mean('operating:battery_charge_kwh')} / {mean('operating:battery_discharge_kwh')} | {mean('operating:zero_solar_electrolyser_kwh')} |")
    md += ["", "## Individual inventory changes", "",
           "Mean ending minus opening stocks. A negative number means depletion over the seven-day episode. A positive methane change can coexist with hydrogen or CO2 depletion. These measurements do not establish indefinitely repeatable operation.", "",
           "| Condition | Controller | Battery, kWh | Hydrogen, kg | CO2, kg | Methane, kg |",
           "|---|---|---:|---:|---:|---:|"]
    for (controller, condition), group in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        differences = []
        for stock in ("battery_energy_kwh", "h2_level_kg", "co2_level_kg", "ch4_level_kg"):
            start, end = (estimate(group, f"{when}:{stock}")["mean"] for when in ("initial", "final"))
            differences.append(number(end - start, 1) if start is not None and end is not None else "unavailable")
        md.append(f"| {condition} | {controller} | " + " | ".join(differences) + " |")
    md += ["", "## Assumptions and the evidence standard", "",
           "This is an assumed controller benchmark. Its purpose is to establish feasible operation, learning, differences between competent controllers, and the incremental value of information. A demonstration need not reproduce a particular commercial plant, and overnight operation is not an objective in itself.", "",
           "The [bounded plausibility review](../posthoc-design-provenance/plausibility-review.md) compares conversion magnitudes with primary literature and distinguishes the model's assumptions from measurements. The numerical defaults were not originally derived from those sources; the comparison is retrospective. The [nominal sizing calculations](../posthoc-design-provenance/nominal-sizing.json) show how component capacities, stoichiometry, buffers and delivery demand relate.", "",
           "At 30 kg/h, the assumed contract requires 720 kg methane per day. Under the model's stoichiometry this requires 360 kg hydrogen and 1,980 kg carbon dioxide, with 23.544 MWh/day nominal conversion input before startup, part-load and storage losses. These are dimensional balances, not a realizable schedule or evidence of thermal dynamics. Paired controller experiments supply the separate evidence about which decisions improve performance."]
    md.append("")
    (output / "REPORT.md").write_text("\n".join(md))
    facts = {"analysis_sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
             "training_decision_sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest(),
             "primary_outcomes": len(primary_rows), "complete_primary_outcomes": complete,
             "primary_physical_passes": physical, "controller_order": ordered,
             "development_learning": learning,
             "successes": {c: {k: groups[c, k]["delivery_successes"] if (c, k) in groups else None for k in CONDITIONS} for c in ordered}}
    if supplement:
        facts["forecast_analysis_sha256"] = hashlib.sha256(forecast_analysis_path.read_bytes()).hexdigest()
        facts["forecast_criterion"] = supplement["criterion"]
    (output / "report-data.json").write_text(json.dumps(facts, indent=2) + "\n")
    return facts


def compile_pdf(output: Path):
    if not (output / "interpretation.tex").exists():
        raise ValueError("Review and write interpretation.tex against actual results before compiling")
    build = output / ".build"
    build.mkdir(exist_ok=True)
    for _ in range(2):
        run = subprocess.run(["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
            f"-output-directory={build}", "plantanagent-comparison-ppo.tex"], cwd=output, capture_output=True, text=True,
            env={**os.environ, "SOURCE_DATE_EPOCH": "1789603200", "FORCE_SOURCE_DATE": "1", "TZ": "UTC"})
        if run.returncode:
            raise RuntimeError(run.stdout + run.stderr)
    log = (build / "plantanagent-comparison-ppo.log").read_text()
    issues = [line for line in log.splitlines() if any(word in line for word in ("Overfull", "undefined", "multiply defined"))]
    match = re.search(r"Output written on .*?\((\d+) pages?[,\s]", log.replace("\n", ""))
    if issues or not match or int(match.group(1)) != 2:
        raise ValueError(f"Require two clean pages; pages={match.group(1) if match else None}; issues={issues}")
    source = build / "plantanagent-comparison-ppo.pdf"
    destination = ROOT / "output/pdf/plantanagent-comparison-ppo.pdf"
    shutil.copyfile(source, output / source.name)
    shutil.copyfile(source, destination)
    return {"pdf": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "pages": 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--training-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--forecast-analysis", type=Path)
    args = parser.parse_args()
    result = prepare(args.analysis.resolve(), args.training_decision.resolve(), args.output.resolve(),
                     args.forecast_analysis.resolve() if args.forecast_analysis else None)
    if args.compile:
        result.update(compile_pdf(args.output.resolve()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

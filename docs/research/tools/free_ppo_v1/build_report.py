"""Build data-bound tables and a two-column LaTeX research note after evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import gzip
from pathlib import Path
import shutil
import subprocess
import os
import re

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "docs/research/results/free_ppo_20260917"
LABELS = {"coverage-direct": "Translated prior", "untrained-101": "Untrained policy",
          "free-ppo-v1-101": "PPO / 101", "free-ppo-v1-211": "PPO / 211", "free-ppo-v1-307": "PPO / 307"}


def estimate(group, key):
    result = group["metrics"][key]["all_case"]
    if result["mean"] is None:
        raise ValueError(f"Missing full-week measurement: {key}")
    return result


def f(value, digits=1):
    return f"{value:,.{digits}f}"


def prepare(analysis_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis = json.loads(analysis_path.read_text())
    plan = json.loads((RESULTS / "training/plan.json").read_text())
    groups = {g["controller"]: g for g in analysis["groups"] if g["split"] == "held_out"}
    if set(groups) != set(LABELS) or any(g["n_scenarios"] != 32 for g in groups.values()):
        raise ValueError("Report requires all five complete prescribed panels")
    results = [json.loads((RESULTS / f"training/seed-{s}/result.json").read_text()) for s in plan["optimizer_seeds"]]
    if any(r["status"] != "completed" or r["num_timesteps"] != plan["total_timesteps"] for r in results):
        raise ValueError("Report cannot call incomplete training full-budget")
    contrasts = {c["reference"]: c for c in analysis["aggregate_comparisons"] if c["split"] == "held_out"}
    trained = [groups[f"free-ppo-v1-{s}"] for s in plan["optimizer_seeds"]]
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    rows = []
    for label, name in LABELS.items():
        g = groups[label]
        rows.append(f"{name} & {g['delivery_successes']}/32 & {f(estimate(g,'delivered_kg')['mean'],0)} & {f(estimate(g,'net_operating_value_eur')['mean'],0)}\\\\")
    counts = [g["delivery_successes"] for g in trained]
    if min(counts) == max(counts):
        successes = f"{counts[0]} of 32"
    else:
        successes = f"{min(counts)}--{max(counts)} of 32"
    abstract = f"The trained policies meet every delivery in {successes} test weeks; the untrained policy does so in {groups['untrained-101']['delivery_successes']} of 32. This establishes feasibility for the modeled seven-day task; longer-term sustainability remains untested."
    macros = {
        "TrainingSteps": f(plan["total_timesteps"], 0),
        "CompleteTrainingWeeks": str(results[0]["completed_episodes"]),
        "AbstractFinding": abstract,
        "ResultTableRows": "\n".join(rows),
    }
    (output / "results.tex").write_text("\n".join(f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in macros.items()) + "\n")
    completed = sum(g["complete_weeks"] for g in groups.values())
    physical = sum(g["physical_invariant_successes"] for g in groups.values())
    text = [f"Across the five controllers, {completed}/160 evaluations complete the seven-day episode and {physical}/160 pass the minute-level physical checks. Table~\\ref{{tab:results}} reports all final policies, including failed commitments."]
    for reference, name in (("untrained-101", "the fixed untrained policy"), ("coverage-direct", "the prior controller translated to direct actions")):
        c = contrasts[reference]["metrics"]["net_operating_value_eur"]["all_case"]
        lo, hi = c["ci95"]
        text.append(f"Averaging the three trained policies within each scenario, modeled contribution differs from {name} by EUR\\,{f(c['mean'])}/week (95\\% interval $[{f(lo)}, {f(hi)}]$).")
    stops = [estimate(g, "electrolyser_forced_stops")["mean"] for g in trained]
    liabilities = [-(estimate(g, "component:methane_endowment_liability_change")["mean"] + estimate(g, "component:battery_endowment_liability_change")["mean"]) for g in trained]
    liability_text = f"approximately EUR\\,{f(sum(liabilities) / len(liabilities), 0)}" if round(min(liabilities), 1) == round(max(liabilities), 1) else f"EUR\\,{f(min(liabilities))}--{f(max(liabilities))}"
    text.append(f"Trained policies average {f(min(stops))}--{f(max(stops))} forced electrolyser stops/week and {liability_text} in terminal depletion liabilities. Daily stock restoration and zero forced stops were not required for delivery success.")
    (output / "findings.tex").write_text("\n\n".join(text) + "\n")
    curves = analysis["learning_curves"]
    expected_curves = {f"free-ppo-v1-{s}" for s in plan["optimizer_seeds"]}
    if {c["controller"] for c in curves} != expected_curves or len(curves) != 3:
        raise ValueError("Research note requires all three prescribed development learning curves")
    for curve in curves:
        if [p["transitions"] for p in curve["points"]] != [0, 65536, 393216, plan["total_timesteps"]] or any(p["n_scenarios"] != 8 for p in curve["points"]):
            raise ValueError("Each learning curve requires all four fixed checkpoints on eight development scenarios")
        if any(p["metrics"]["net_operating_value_eur"]["mean"] is None for p in curve["points"]):
            raise ValueError("Learning curve contains missing complete-week contribution; do not impute failed evaluations")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.35, 2.15), layout="constrained")
    colors = {101: "#176D72", 211: "#D48534", 307: "#5E6798"}
    markers = {101: "o", 211: "s", 307: "^"}
    for curve in curves:
        points = curve["points"]
        seed = int(curve["controller"].split("-")[-1])
        ax.plot([p["transitions"] / 1000 for p in points], [p["metrics"]["net_operating_value_eur"]["mean"] / 1000 for p in points],
                label=f"PPO {seed}", color=colors[seed], marker=markers[seed], markersize=3, linewidth=1.5)
    ax.set_xlabel("Training decisions (thousands)")
    ax.set_ylabel("Contribution (thousand EUR/week)")
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="best", columnspacing=.8)
    fig.savefig(output / "figures/learning.pdf")
    fig.savefig(output / "figures/learning.png", dpi=200)
    plt.close(fig)
    # Selected prospectively: final optimizer101, first held-out scenario.
    selected = next(row for row in analysis["rows"] if row["split"] == "held_out" and row["controller"] == "free-ppo-v1-101" and row["seed"] == 10200000)
    recording_path = Path(selected["recording"]["path"])
    if hashlib.sha256(recording_path.read_bytes()).hexdigest() != selected["recording"]["sha256"]:
        raise ValueError("Prescribed trajectory recording changed")
    with gzip.open(recording_path, "rt") as stream:
        recording = json.load(stream)
    states = recording["minute_truth_states"]
    if len(states) != 10081:
        raise ValueError("Prescribed trajectory is not a complete minute trace")
    days = [s["step"] / 1440 for s in states]
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 2.7), sharex=True, layout="constrained")
    axes[0].plot(days, [s["solar_available_kw"] / 1000 for s in states], color="#D48534", linewidth=.8, label="PV")
    axes[0].plot(days, [s["electrolyser_power_kw"] / 1000 for s in states], color="#176D72", linewidth=.8, label="Electrolysis")
    axes[0].set_ylabel("Power (MW)")
    axes[0].legend(frameon=False, ncol=2, fontsize=6, loc="upper right")
    axes[1].plot(days, [s["battery_soc"] * 100 for s in states], color="#5E6798", linewidth=.8)
    axes[1].axhline(5, color="#777777", linestyle=":", linewidth=.6)
    axes[1].set_ylabel("Battery (%)")
    axes[1].set_ylim(0, 102)
    axes[2].plot(days, [s["ch4_level_kg"] for s in states], color="#176D72", linewidth=.8)
    axes[2].set_ylabel("Methane (kg)")
    axes[2].set_xlabel("Days since episode start")
    axes[2].set_xlim(0, 7)
    for ax in axes:
        ax.grid(alpha=.15)
        ax.tick_params(labelsize=7)
    fig.savefig(output / "figures/operating-week.pdf")
    fig.savefig(output / "figures/operating-week.png", dpi=200)
    plt.close(fig)
    shutil.copyfile(Path(__file__).with_name("paper-template.tex"), output / "plantanagent-direct-ppo.tex")
    readable = ["# Learning plant operation from scratch", "", "Filippo Gombac and Vittorio Garavelli", "", abstract.replace("--", " to "), "",
                f"Each of the three independent policies completed {plan['total_timesteps']:,} transitions from random weights. All final endpoints are reported; intermediate development scores did not select a winner.", "",
                "| Controller | All deliveries | Methane delivered (kg/week) | Contribution (EUR/week) | Forced stops/week |",
                "|---|---:|---:|---:|---:|"]
    for label, name in LABELS.items():
        g = groups[label]
        readable.append(f"| {name} | {g['delivery_successes']}/32 | {f(estimate(g,'delivered_kg')['mean'])} | {f(estimate(g,'net_operating_value_eur')['mean'])} | {f(estimate(g,'electrolyser_forced_stops')['mean'])} |")
    readable += ["", "The policy controls power directly, without the handcrafted controller, a correction envelope, a daily-stock constraint learner, the former 15% operator reserve, or automatic spill charging. Equipment dynamics, physical bounds and the modeled 5% battery floor remain. Delivery, missed commitments, wear and terminal depletion define the scalar reward.", "",
                 "The translated rule-based reference uses the same direct-action runtime. Holding solar fractions for 15 minutes cannot maintain constant startup power as sunlight changes. Its result does not establish the performance of the original controller or a strong adapted rule-based controller. This experiment changes the task from the earlier hybrid study; its success counts are not interchangeable with the old daily-stock/zero-forced-stop qualification.", "",
                 "The trained policies end with a mean battery deficit of approximately 3.27 MWh, while none ends with a material-equivalent deficit. The prescribed trajectory draws the battery down early and later recharges small amounts. Depletion alone cannot distinguish reliance on opening energy from operation at a lower reserve; continuous multiweek testing is needed. These seven-day results do not establish indefinite sustainability.", "",
                 "Paired uncertainty uses 32 environmental scenarios, averaging the three trained policies within each scenario. It conditions on these trained policies and omits broader retraining uncertainty. All monetary values use synthetic assumptions and exclude solar capital recovery and fixed operating costs.", ""]
    (output / "REPORT.md").write_text("\n".join(readable))
    facts = {"analysis_sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(), "training_timesteps_each": plan["total_timesteps"],
             "completed_evaluations": completed, "physical_passes": physical,
             "controllers": {label: {"all_deliveries": g["delivery_successes"], "mean_delivered_kg": estimate(g, "delivered_kg")["mean"],
                                      "mean_contribution_eur": estimate(g, "net_operating_value_eur")["mean"],
                                      "mean_forced_stops": estimate(g, "electrolyser_forced_stops")["mean"]} for label, g in groups.items()},
             "contrasts": contrasts}
    (output / "report-data.json").write_text(json.dumps(facts, indent=2) + "\n")
    return facts


def compile_pdf(output, destination):
    if not (output / "interpretation.tex").exists():
        raise ValueError("Write and review interpretation.tex against the actual results before compiling")
    build = output / ".build"
    build.mkdir(exist_ok=True)
    for _ in range(2):
        result = subprocess.run(["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
                                 f"-output-directory={build}", "plantanagent-direct-ppo.tex"], cwd=output,
                                env={**os.environ, "SOURCE_DATE_EPOCH": "1789603200", "FORCE_SOURCE_DATE": "1", "TZ": "UTC"}, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
    log = (build / "plantanagent-direct-ppo.log").read_text()
    issues = [line for line in log.splitlines() if "Overfull" in line or "undefined" in line or "multiply defined" in line]
    # TeX wraps long output paths and even the word "pages" across log lines.
    match = re.search(r"Output written on .*?\((\d+) pages?[,\s]", log.replace("\n", ""))
    if issues or not match or int(match.group(1)) != 2:
        raise ValueError(f"Paper requires two clean pages; pages={match.group(1) if match else None}, issues={issues}")
    source = build / "plantanagent-direct-ppo.pdf"
    shutil.copyfile(source, output / source.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return {"pdf": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "pages": 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    facts = prepare(args.analysis.resolve(), args.output.resolve())
    result = {"prepared": str(args.output), "training_timesteps_each": facts["training_timesteps_each"]}
    if args.compile:
        result.update(compile_pdf(args.output.resolve(), ROOT / "output/pdf/plantanagent-direct-ppo.pdf"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

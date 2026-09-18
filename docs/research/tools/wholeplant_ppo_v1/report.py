"""Build a data-bound report from the complete, frozen whole-plant campaign."""
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
ARMS = ("restricted-instant", "shared-instant", "restricted-dynamic", "shared-dynamic")
ARM_LABELS = dict(zip(ARMS, ("R--I", "S--I", "R--D", "S--D")))
PRIMARY = ("daylight", "storage", "ppo-101", "ppo-211", "ppo-307", "storage-persistence")
LABELS = {"daylight": "Daylight", "storage": "Storage", "storage-persistence": "Persistence",
          **{f"ppo-{seed}": f"PPO {seed}" for seed in (101, 211, 307)}}
VALUE = "net_operating_value_eur"


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def number(value, digits=0):
    return "unavailable" if value is None else f"{value:,.{digits}f}"


def estimate(group, metric=VALUE):
    return group["metrics"][metric]["all_case"]


def interval(stats, digits=0):
    return "unavailable" if stats["ci95"] is None else f"[{number(stats['ci95'][0], digits)}, {number(stats['ci95'][1], digits)}]"


def validate(analysis, results):
    expected = {(c, a) for c in ("daylight", "storage") for a in ARMS}
    expected |= {(c, "shared-dynamic") for c in PRIMARY[2:]}
    held = [r for r in analysis["rows"] if r["split"] == "held_out"]
    if len(analysis["rows"]) != 496 or len(held) != 384:
        raise ValueError("Final report requires all 496 prescribed cases")
    for controller, arm in expected:
        cases = [r for r in held if (r["controller"], r["condition"]) == (controller, arm)]
        if sorted(r["seed"] for r in cases) != list(range(30200000, 30200032)):
            raise ValueError(f"Missing or duplicated held-out panel: {controller}/{arm}")
    completion = read(results / "supervision/completion.json")
    if completion["status"] != "completed" or completion["analysis"] != pin(results / "analysis/analysis.json"):
        raise ValueError("Analysis lacks matching campaign completion")
    for row in analysis["rows"]:
        if not (row["execution_complete"] and row["physical_invariants_passed"] and row["economic_audit"]["passed"]):
            raise ValueError("Report needs an explicit incomplete/audit-failure treatment before publication")
        if row["physical_metrics"]["guard_interventions"] != 0:
            raise ValueError("A reported guard intervention needs review against the direct-control claim")
    for seed in (101, 211, 307):
        result = read(results / f"training/seed-{seed}/result.json")
        if (result["status"] != "completed" or not result["actor_mean_parameters_changed"]
                or result["num_timesteps"] != completion["training_transitions_per_seed"]):
            raise ValueError("Training endpoint does not match completion")
    return held, completion


def prepare(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results, output = Path(results).resolve(), Path(output).resolve()
    analysis_path = results / "analysis/analysis.json"
    data = read(analysis_path)
    held, completion = validate(data, results)
    groups = {(g["controller"], g["condition"]): g for g in data["groups"] if g["split"] == "held_out"}
    mean = lambda c, a, m=VALUE: estimate(groups[c, a], m)["mean"]
    successes = lambda c, a="shared-dynamic": groups[c, a]["delivery_successes"]
    aggregates = {c["reference"]: c for c in data["aggregate_comparisons"] if c["split"] == "held_out"}
    forecast = data["forecast_comparison"]
    criterion = forecast["criterion"]
    fstats = estimate(forecast)
    steps = completion["training_transitions_per_seed"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)

    counts = [successes(c) for c in PRIMARY[2:5]]
    range_text = str(counts[0]) if min(counts) == max(counts) else f"{min(counts)}--{max(counts)}"
    abstract = (f"With both changes enabled, daylight and storage control meet every delivery in "
                f"{successes('daylight')}/32 and {successes('storage')}/32 weeks; PPO succeeds in {range_text}/32. "
                f"Mean PPO contribution relative to daylight is EUR\\,{number(estimate(aggregates['daylight'])['mean'])}/week "
                f"(paired 95\\% interval {interval(estimate(aggregates['daylight']))}).")
    train_text = (f"Three fresh policies (optimizer seeds 101, 211, 307) each receive {steps:,} transitions from random weights "
                  "in the shared--dynamic arm. A discarded three-learner speed pilot sets this common budget using throughput "
                  "and the remaining four-hour local allowance, without selecting policies by score.")
    architecture_rows = []
    for arm in ARMS:
        values = [f"{successes(c, arm)}/32; {number(mean(c, arm))}" for c in PRIMARY[:2]]
        architecture_rows.append(f"{ARM_LABELS[arm]} & " + " & ".join(values) + r"\\")
    macros = {"AbstractFinding": abstract, "TrainingText": train_text, "ArchitectureRows": "\n".join(architecture_rows)}
    (output / "results.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")

    learning = []
    for seed in (101, 211, 307):
        comparison = next(c for c in data["learning_comparisons"] if c["treatment"] == f"ppo-{seed}")
        initial, final = [], []
        for label, target in ((f"ppo-initial-{seed}", initial), (f"ppo-{seed}", final)):
            target.extend(r for r in data["rows"] if r["split"] == "development" and r["controller"] == label)
            if sorted(r["seed"] for r in target) != list(range(30100000, 30100008)):
                raise ValueError("Missing development-only learning panel")
        learning.append({"optimizer_seed": seed,
            "initial_successes": sum(r["every_delivery_passed"] for r in initial),
            "final_successes": sum(r["every_delivery_passed"] for r in final),
            "contribution_gain": estimate(comparison)})
    count_list = ", ".join(str(x["initial_successes"]) for x in learning)
    final_list = ", ".join(str(x["final_successes"]) for x in learning)
    findings = ["All 496 prescribed cases complete and pass minute-level physical and independent economic audits. "
                f"On the eight development scenarios, initial policies meet every delivery in {count_list} weeks, "
                f"versus {final_list} after training (seeds 101, 211, 307). These are learning checks, not additional held-out evidence.",
                f"The shared--dynamic PPO policies meet every delivery in {', '.join(str(x) for x in counts)} of 32 held-out weeks. "
                f"Their mean contribution relative to storage is EUR\\,{number(estimate(aggregates['storage'])['mean'])}/week "
                f"(paired 95\\% interval {interval(estimate(aggregates['storage']))}).",
                f"Forecast minus persistence contribution is EUR\\,{number(fstats['mean'])}/week "
                f"(paired 95\\% interval {interval(fstats)}), with {criterion['forecast_rescues']} rescued and "
                f"{criterion['forecast_regressions']} regressed delivery weeks. The declared practical forecast criterion is "
                f"{'met' if criterion['practical_evidence_criterion_met'] else 'not met'}."]
    (output / "findings.tex").write_text("\n\n".join(findings) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                        "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.35, 1.95), layout="constrained")
    for i, c in enumerate(PRIMARY):
        stats = estimate(groups[c, "shared-dynamic"])
        mu, (lo, hi) = stats["mean"], stats["ci95"]
        color = "#226B73" if c.startswith("ppo") else "#9B793D" if c == "storage-persistence" else "#64737E"
        ax.plot([lo / 1000, hi / 1000], [i, i], color=color, linewidth=1.5)
        ax.scatter([mu / 1000], [i], s=17, color=color, zorder=3)
    ax.set_yticks(range(len(PRIMARY)), [LABELS[c] for c in PRIMARY])
    ax.invert_yaxis()
    ax.set_xlabel("Contribution (thousand EUR/week)")
    ax.grid(axis="x", alpha=.2)
    fig.savefig(output / "figures/contribution-comparison.pdf")
    fig.savefig(output / "figures/contribution-comparison.png", dpi=220)
    plt.close(fig)
    progress_files = [results / f"training/seed-{seed}/progress.jsonl" for seed in (101, 211, 307)]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.7), layout="constrained")
    for seed, color, style, path in zip((101, 211, 307), ("#226B73", "#906837", "#646CB1"), ("-", "--", ":"), progress_files):
        progress = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not progress or progress[-1]["num_timesteps"] != steps or any(p["optimizer_seed"] != seed for p in progress):
            raise ValueError("Training curve lacks the prescribed complete optimizer trajectory")
        if any(a["num_timesteps"] >= b["num_timesteps"] for a, b in zip(progress, progress[1:])):
            raise ValueError("Training curve has nonmonotone or duplicate progress records")
        x = [p["num_timesteps"] / 1000 for p in progress]
        for ax, metric, scale in zip(axes, ("recent_mean_delivered_fraction", "recent_mean_raw_reward_eur", "recent_mean_forced_stops"), (100, .001, 1)):
            ax.plot(x, [p[metric] * scale for p in progress], color=color, linestyle=style, linewidth=1.3, label=str(seed))
    for ax, title, unit in zip(axes, ("Delivered demand", "Training objective", "Electrolyser forced stops"), ("Percent", "Thousand EUR / episode", "Count / episode")):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Training transitions (thousands)")
        ax.set_ylabel(unit)
        ax.grid(alpha=.2)
    axes[0].set_ylim(0, 102)
    axes[0].legend(title="Optimizer seed", fontsize=7, title_fontsize=7, frameon=False)
    fig.savefig(output / "figures/training-curves.pdf")
    fig.savefig(output / "figures/training-curves.png", dpi=220)
    plt.close(fig)
    shutil.copyfile(Path(__file__).with_name("paper-template.tex"), output / "plantanagent-wholeplant-ppo.tex")

    md = ["# Giving the controller a whole-plant storage choice", "", "Filippo Gombac and Vittorio Garavelli", "",
          "The seven-day delivery mission is unchanged. The battery can now supply every process; downstream readiness consumes energy and persists between decisions. "
          "All numerical readiness coefficients are explicit benchmark assumptions. The experiment asks whether these extra choices improve control, rather than requiring overnight production.", "",
          f"All 496 prescribed cases completed and passed physical and economic accounting: 112 development and 384 held-out evaluations. "
          f"The latter share 32 environmental scenarios; repeated controllers do not add independent weather evidence. Each of three fresh PPO policies received {steps:,} transitions.", "",
          "## Architecture comparison", "", "Every-delivery weeks and mean modeled contribution. Same fixed baseline algorithms, not a separate optimization for each architecture.", "",
          "| Battery access | Downstream readiness | Daylight delivery weeks | Daylight EUR/week | Storage delivery weeks | Storage EUR/week |",
          "|---|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        access, dynamics = arm.split("-")
        md.append(f"| {access} | {dynamics} | {successes('daylight', arm)}/32 | {number(mean('daylight', arm))} | {successes('storage', arm)}/32 | {number(mean('storage', arm))} |")
    md += ["", "## Shared battery and downstream readiness", "",
           "| Controller | Every delivery | Mean EUR/week | Mean battery discharge, kWh/week | Mean zero-solar methane, kg/week |",
           "|---|---:|---:|---:|---:|"]
    for c in PRIMARY:
        md.append(f"| {LABELS[c]} | {successes(c)}/32 | {number(mean(c, 'shared-dynamic'))} | "
                  f"{number(mean(c, 'shared-dynamic', 'operating:battery_discharge_kwh'))} | "
                  f"{number(mean(c, 'shared-dynamic', 'operating:zero_solar_ch4_produced_kg'), 1)} |")
    md += ["", "## Learning check", "", "Eight shared development cases; normalization and actor change jointly. Initial policies were random, not demonstrations.", "",
           "| Optimizer seed | Initial every-delivery weeks | Final every-delivery weeks | Mean contribution gain, EUR/week | Paired 95% interval |",
           "|---|---:|---:|---:|---:|"]
    for item in learning:
        md.append(f"| {item['optimizer_seed']} | {item['initial_successes']}/8 | {item['final_successes']}/8 | "
                  f"{number(item['contribution_gain']['mean'])} | {interval(item['contribution_gain'])} |")
    md += ["", "![Training curves for every optimizer seed](figures/training-curves.png)", "",
           "Training curves show the logged mean over the most recent up to 16 completed training episodes. "
           "Actions include exploration and scenarios change during training; these curves are not fixed-scenario deterministic test performance. "
           "No seed, checkpoint or smoothing window was selected by its results."]
    md += ["", "## Paired effects", "", "For direct comparisons, positive values favor treatment. A positive factorial interaction means the shared-bus effect is more favorable with readiness than without it; it does not by itself rank the four arms. Intervals are descriptive scenario bootstraps, without multiplicity correction or retraining uncertainty.", "",
           "| Comparison | Mean contribution difference, EUR/week | Paired 95% interval |",
           "|---|---:|---:|"]
    comparisons = [c for field in ("aggregate_comparisons", "condition_comparisons", "factorial_contrasts") for c in data[field] if c["split"] == "held_out"]
    for c in comparisons:
        label = (c.get("controller", "") + ": " if "controller" in c else "") + c["comparison"]
        stats = estimate(c)
        precision = 3 if 0 < abs(stats["mean"]) < 1 else 0
        md.append(f"| {label} | {number(stats['mean'], precision)} | {interval(stats, precision)} |")
    md += ["", "## Forecast intervention", "",
           "The storage-persistence controller replaces the same three consumed forecast coordinates with current solar, preserving all other rules. "
           "The practical criterion was fixed before outcomes: at least four forecast-rescued weeks and no regressions, or at least EUR 250/week with a positive paired 95% lower bound and no lower delivery count.", "",
           f"Forecast contribution difference: EUR {number(fstats['mean'])}/week, paired 95% interval {interval(fstats)}. "
           f"Rescued/regressed weeks: {criterion['forecast_rescues']}/{criterion['forecast_regressions']}. "
           f"Criterion {'met' if criterion['practical_evidence_criterion_met'] else 'not met'}. "
           "This estimates incremental forecast information within this fixed controller; it does not isolate the best forecastless strategy or PPO's use of its forecasts.", "",
           "## Actual process operation", "",
           "All energy is measured actual flow. Zero solar means no available solar during that physical minute, which can include daytime PV outages as well as night. Startup includes preparation losses; standby is metered separately from productive input.", "",
           "| Arm | Controller | Battery to E / DAC / meth, kWh | Zero-solar H2 / CO2 / methane, kg | DAC starts / cold starts | Meth starts / cold starts |",
           "|---|---|---:|---:|---:|---:|"]
    for (c, arm), g in sorted(groups.items(), key=lambda x: (ARMS.index(x[0][1]), x[0][0])):
        m = lambda key: estimate(g, "operating:" + key)["mean"]
        discharge = " / ".join(number(m(u + "_battery_kwh")) for u in ("electrolyser", "dac", "methanator"))
        gas = " / ".join(number(m(k), 1) for k in ("zero_solar_h2_produced_kg", "zero_solar_co2_captured_kg", "zero_solar_ch4_produced_kg"))
        md.append(f"| {arm} | {LABELS[c]} | {discharge} | {gas} | {number(m('dac_starts'), 1)} / {number(m('dac_cold_starts'), 1)} | {number(m('methanator_starts'), 1)} / {number(m('methanator_cold_starts'), 1)} |")
    md += ["", "## Readiness energy", "", "Mean metered energy per week and ending readiness, both in kWh. All dynamic units begin cold. Ending readiness receives no economic salvage credit.", "",
           "| Arm | Controller | Process | Startup | Standby | Productive | Readiness loss | Ending readiness |",
           "|---|---|---|---:|---:|---:|---:|---:|"]
    for (c, arm), g in sorted(groups.items(), key=lambda x: (ARMS.index(x[0][1]), x[0][0])):
        if arm.endswith("instant"):
            continue
        for unit in ("dac", "methanator"):
            energies = [estimate(g, f"operating:{unit}_{suffix}_kwh")["mean"] for suffix in ("startup", "standby", "productive", "heat_loss")]
            energies.append(estimate(g, f"final:{unit}_readiness_kwh")["mean"])
            md.append(f"| {arm} | {LABELS[c]} | {unit} | " + " | ".join(number(x, 1) for x in energies) + " |")
    components = ("delivered_service_value", "unmet_delivery_penalty", "solar_opportunity_cost", "battery_throughput_reserve", "electrolyser_life_reserve", "methane_endowment_liability_change", "battery_endowment_liability_change")
    md += ["", "## Contribution components", "", "Mean EUR/week. The ledger stores signed contributions: positive values add to the objective, negative values subtract. These are modeled service value and costs, not calibrated plant profit.", "",
           "| Arm | Controller | Delivered value | Unmet penalty | Solar | Battery wear | Electrolyser wear | Gas depletion | Battery depletion | Net |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (c, arm), g in sorted(groups.items(), key=lambda x: (ARMS.index(x[0][1]), x[0][0])):
        values = [estimate(g, "component:" + component)["mean"] for component in components]
        values.append(estimate(g)["mean"])
        md.append(f"| {arm} | {LABELS[c]} | " + " | ".join(number(x, 1) for x in values) + " |")
    md += ["", "## Individual stock changes", "", "Mean final minus initial. Gas conversion can preserve equivalent value while depleting an individual feedstock. These finite-week results do not establish indefinite sustainability.", "",
           "| Arm | Controller | Battery, kWh | H2, kg | CO2, kg | Methane, kg |",
           "|---|---|---:|---:|---:|---:|"]
    for (c, arm), g in sorted(groups.items(), key=lambda x: (ARMS.index(x[0][1]), x[0][0])):
        changes = [estimate(g, "final:" + k)["mean"] - estimate(g, "initial:" + k)["mean"]
                   for k in ("battery_energy_kwh", "h2_level_kg", "co2_level_kg", "ch4_level_kg")]
        md.append(f"| {arm} | {LABELS[c]} | " + " | ".join(number(x, 1) for x in changes) + " |")
    md += ["", "## Assumptions and limits", "",
           "DAC is an aggregate readiness proxy, not a literal common bed temperature. Methanation heat recovery, downstream wear, detailed bed cycling and a calibrated site are absent. "
           "Cold startup and standby are paid using the same electrical budget as production. Full-rate yield is preserved, with a fixed readiness overhead raising part-load costs. "
           "Battery availability is checked before netting simultaneous charge/support requests; direct solar allocation remains available at the floor. "
           "Cold initial readiness and no terminal readiness credit are finite-horizon conventions. The economic ledger excludes fixed costs and is not commercial profit.", "",
           "The current experiment retrains PPO only on the full shared-dynamic model. Architectural contrasts use fixed baseline rules, so they do not establish optimized architectural value. "
           "Every optimizer seed and prescribed failure is retained. More internal states or overnight activity alone do not prove a difficult planning benchmark.", ""]
    (output / "REPORT.md").write_text("\n".join(md))
    facts = {"analysis": pin(analysis_path), "completion": pin(results / "supervision/completion.json"),
             "cases":496, "development_cases":112, "held_out_cases":384, "held_out_environmental_scenarios":32,
             "guard_interventions_across_all_cases": sum(row["physical_metrics"]["guard_interventions"] for row in data["rows"]),
             "training_transitions_per_seed": steps, "training_progress_sources": [pin(path) for path in progress_files], "development_learning": learning,
             "forecast_criterion": criterion, "forecast_contribution": fstats,
             "held_out_groups": list(groups.values()), "paired_effects": comparisons}
    (output / "report-data.json").write_text(json.dumps(facts, indent=2) + "\n")
    (output / "README.md").write_text("# Editable whole-plant experiment paper\n\n"
        "Main source: `plantanagent-wholeplant-ppo.tex`. Build twice using `pdflatex -no-shell-escape plantanagent-wholeplant-ppo.tex`.\n\n"
        "The generated results are in `results.tex` and `findings.tex`; reviewed interpretation is in `interpretation.tex`. "
        "`report-data.json` binds figures and numbers to the complete experiment. "
        "`REPORT.md` provides expanded results and stock/operating tables. Keep those files beside the `figures` directory.\n\n"
        "Required TeX packages include newtx, geometry, microtype, booktabs, caption, fancyhdr, titlesec and hyperref. "
        "The standalone source archive includes a content-hash manifest and has been extracted and recompiled during validation.\n")
    return {k:v for k,v in facts.items() if k not in ("held_out_groups", "paired_effects")}


def compile_pdf(output):
    output = Path(output).resolve()
    if not (output / "interpretation.tex").exists():
        raise ValueError("Write reviewed interpretation from the actual completed results first")
    build = output / ".build"
    build.mkdir(exist_ok=True)
    for _ in range(2):
        run = subprocess.run(["/Library/TeX/texbin/pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
            f"-output-directory={build}", "plantanagent-wholeplant-ppo.tex"], cwd=output, capture_output=True, text=True, timeout=60,
            env={**os.environ, "SOURCE_DATE_EPOCH": "1789603200", "FORCE_SOURCE_DATE": "1", "TZ": "UTC"})
        if run.returncode:
            raise RuntimeError(run.stdout + run.stderr)
    log = (build / "plantanagent-wholeplant-ppo.log").read_text()
    issues = [line for line in log.splitlines() if any(word in line for word in ("Overfull", "undefined", "multiply defined"))]
    match = re.search(r"Output written on .*?\((\d+) pages?[,\s]", log.replace("\n", ""))
    if issues or not match or int(match.group(1)) != 2:
        raise ValueError(f"Require two clean pages; pages={match.group(1) if match else None}; issues={issues}")
    source = build / "plantanagent-wholeplant-ppo.pdf"
    destination = ROOT / "output/pdf" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output / source.name)
    shutil.copyfile(source, destination)
    return {"pdf": pin(destination), "pages":2}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()
    facts = prepare(args.results, args.output)
    if args.compile:
        facts.update(compile_pdf(args.output))
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()

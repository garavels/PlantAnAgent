"""Render scientific figures without creating or evaluating a policy.

Inputs are one completed learned_control_v1 panel and explicit recipe, reference
and example-policy labels. Every underlying case must remain present and audited;
delivery failures remain in all means. PNG is the default. PDF export is optional
for a later paper build, under the paper author's artifact workflow.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import statistics
import textwrap

from docs.research.tools.learned_control_report_v2 import inspector

SCHEMA = "learned-control-scientific-figures-v2"
STRESS = ("pv6", "pv18", "pv36", "meth24", "meth48")
TRACE_CONDITIONS = ("pv36", "meth48")
TRACE_FIELDS = ("ch4_level_kg", "battery_soc", "electrolyser_power_kw", "dac_power_kw", "methanator_power_kw", "solar_available_kw")
CONDITION_LABELS = {"ordinary": "Ordinary", "control": "Healthy\ncontrol", "pv6": "Solar loss\n6 h", "pv18": "Solar loss\n18 h", "pv36": "Solar loss\n36 h", "meth24": "Methanator loss\n24 h", "meth48": "Methanator loss\n48 h"}
COLORS = ("#17684a", "#43535f", "#356396", "#b77a22", "#82558b", "#38929a", "#c36c48")


def publish(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def select_series(payload, recipe, reference, imitation=None):
    recipe = recipe.removeprefix("recipe:")
    if recipe not in payload["groups"]:
        raise ValueError("Requested recipe is not a frozen analysis group")
    if reference not in payload["controllers"] or payload["controller_metadata"][reference]["kind"] != "baseline":
        raise ValueError("Reference must be a baseline in the frozen panel")
    members = payload["groups"][recipe]
    if not members or any(payload["controller_metadata"][m]["kind"] != "learned" for m in members):
        raise ValueError("The plotted recipe must consist of learned policies")
    series = [{"population": "recipe:"+recipe, "role": "recipe mean", "members": members},
              {"population": reference, "role": "reference", "members": [reference]}]
    for label in payload["controllers"]:
        meta = payload["controller_metadata"][label]
        if meta["kind"] == "baseline" and not meta.get("config") and label != reference:
            series.append({"population": label, "role": "original baseline", "members": [label]})
    if imitation:
        if imitation not in payload["controllers"]:
            raise ValueError("Explicit imitation reference is absent from the frozen panel")
        if not any(s["population"] == imitation for s in series):
            series.append({"population": imitation, "role": "imitation reference (explicit)", "members": [imitation]})
    return recipe, series


def prepare_data(payload, *, panel, recipe, reference, policy, imitation=None):
    """Derive plotting arrays from independently validated source summaries."""
    if not payload["all_cases_audited"]:
        raise ValueError("Figures require all prescribed cases complete and audited; failed executions are not omitted")
    recipe, series = select_series(payload, recipe, reference, imitation)
    members = payload["groups"][recipe]
    if policy not in members:
        raise ValueError("Example policy must be explicitly named and belong to the chosen recipe")
    summaries = {(s["population"], s["panel"]): s for s in payload["summaries"]}
    if not all(c in payload["conditions"] for c in ("ordinary", *STRESS)):
        raise ValueError("Paired scatter requires ordinary and all five prescribed stress conditions")
    conditions = []
    for condition in payload["conditions"]:
        bars = []
        for s in series:
            entry = summaries[(s["population"], condition)]
            bars.append({**s, "mean_unmet_kg": entry["metrics"]["unmet_kg"]["mean"],
                         "n_base_scenarios": entry["n_base_scenarios"],
                         "n_policy_condition_cells": entry["n_policy_condition_cells"]})
        points = [{"policy": member, "optimizer_seed": payload["controller_metadata"][member].get("optimizer_seed"),
                   "mean_unmet_kg": summaries[(member, condition)]["metrics"]["unmet_kg"]["mean"]} for member in members]
        if abs(statistics.mean(p["mean_unmet_kg"] for p in points)-bars[0]["mean_unmet_kg"]) > 1e-7:
            raise ValueError("Individual-policy dots do not average to the recipe bar")
        conditions.append({"condition": condition, "bars": bars, "individual_policy_means": points})
    comparisons = {(c["treatment"], c["reference"], c["panel"]): c for c in payload["comparisons"]}
    stress = comparisons[("recipe:"+recipe, reference, "stress")]
    mixture = comparisons[("recipe:"+recipe, reference, "mixture")]
    if stress["weights"] != {c: .2 for c in STRESS} or mixture["weights"] != {"ordinary": .5, **{c: .1 for c in STRESS}}:
        raise ValueError("Scatter weights differ from the exact declared analyzer weights")
    x = stress["metrics"]["unmet_kg"]["per_seed_differences"]
    y = mixture["metrics"]["net_operating_value_eur"]["per_seed_differences"]
    if len(x) != len(payload["seeds"]) or len(y) != len(payload["seeds"]) or any(v is None for v in [*x, *y]):
        raise ValueError("Paired scatter would omit an environment seed")
    points = [{"environment_seed": seed, "stress_unmet_difference_kg": xv, "mixture_contribution_difference_eur": yv}
              for seed, xv, yv in zip(payload["seeds"], x, y)]
    metadata = {member: payload["controller_metadata"][member] for member in dict.fromkeys([*members, reference, *[s["population"] for s in series if not s["population"].startswith("recipe:")]])}
    efforts = [metadata[name].get("training_effort", {}) for name in members]
    return {"schema": SCHEMA+"-data", "panel": str(Path(panel).resolve()), "split": payload["split"],
            "scenario_count": payload["scenario_count"], "seeds": payload["seeds"], "case_count": payload["case_count"],
            "all_cases_audited": True, "recipe": recipe, "recipe_members": members, "reference": reference,
            "example_policy": policy, "imitation_reference": imitation, "series": series,
            "controller_metadata": metadata, "recipe_training_effort": efforts,
            "conditions": conditions, "scatter": {"direction": "recipe mean minus reference for the same environment seed",
                "x_metric": "stress unmet methane, kg/week", "y_metric": "declared-mixture contribution, EUR/week",
                "stress_weights": stress["weights"], "mixture_weights": mixture["weights"],
                "points": points, "mean_x": statistics.mean(x), "mean_y": statistics.mean(y)},
            "traces": [], "provenance": payload["provenance"],
            "interpretation": "All fixed recipe policies are averaged, not combined into an ensemble; prescribed weights are not natural event probabilities; example environment is the first frozen seed, never selected by outcome"}


def collect(panel, *, recipe, reference, policy, imitation=None):
    panel = Path(panel).resolve()
    paths = {name: panel/(name+".json") for name in ("freeze", "summary", "analysis")}
    spec, summary, analysis = [json.loads(paths[name].read_text()) for name in ("freeze", "summary", "analysis")]
    inspector.verify_pin(summary["freeze"], paths["freeze"])
    if analysis.get("summary"):
        inspector.verify_pin(analysis["summary"], paths["summary"])
    for key in ("gate", "selected_recipe"):
        if spec.get(key):
            inspector.verify_pin(spec[key])
    rows = inspector.validate_panel(spec, summary, analysis)
    case_pins = []
    for seed, condition, controller in rows:
        case_path = panel/"cases"/f"{condition}--{controller}--{seed}"/"result.json"
        if json.loads(case_path.read_text()) != rows[(seed, condition, controller)]:
            raise ValueError("Case result differs from the complete source summary")
        case_pins.append(inspector.file_pin(case_path))
    payload = {"groups": analysis["groups"], "controllers": list(spec["controllers"]),
        "controller_metadata": {k: inspector.controller_metadata(v) for k,v in spec["controllers"].items()},
        "summaries": analysis["summaries"], "comparisons": analysis["paired_comparisons"],
        "all_cases_audited": analysis["all_cases_audited"], "conditions": spec["conditions"],
        "scenario_count": len(spec["seeds"]), "seeds": spec["seeds"], "case_count": analysis["case_count"],
        "split": spec["split"], "provenance": {**{name: inspector.file_pin(path) for name,path in paths.items()},
            "case_results": case_pins, "selection_gate": spec.get("gate"), "selected_recipe": spec.get("selected_recipe"),
            "analysis_summary_pin_present": bool(analysis.get("summary")),
            "unlinked_analysis_validation": "When the optional analysis summary pin is absent, every summary, paired estimate and bootstrap interval is independently reconciled with all retained case rows"}}
    data = prepare_data(payload, panel=panel, recipe=recipe, reference=reference, policy=policy, imitation=imitation)
    first_seed = payload["seeds"][0]
    for condition in TRACE_CONDITIONS:
        if condition not in payload["conditions"]:
            raise ValueError("Prescribed PV36/meth48 example condition is absent")
        traces = []
        for controller in (policy, reference):
            selected = rows[(first_seed, condition, controller)]
            path = panel / "cases" / f"{condition}--{controller}--{first_seed}" / "recording.json.gz"
            pin = inspector.verify_pin(selected["recording"], path)
            with gzip.open(path, "rt") as stream:
                recording = json.load(stream)
            inspector.trace_payload(recording, selected, spec["controllers"][controller])
            states = recording["minute_truth_states"]
            if len(states) != 10081 or inspector.digest(states) != recording["summary"]["retained_minute_truth_sha256"]:
                raise ValueError("Example does not have its original complete minute trace")
            traces.append({"controller": controller, "environment_seed": first_seed,
                "elapsed_h": [state["step"]/60 for state in states],
                "methane_kg": [state["ch4_level_kg"] for state in states],
                "battery_soc_percent": [100*state["battery_soc"] for state in states],
                "total_process_power_mw": [(state["electrolyser_power_kw"]+state["dac_power_kw"]+state["methanator_power_kw"])/1000 for state in states],
                "solar_available_mw": [state["solar_available_kw"]/1000 for state in states],
                "delivery": recording["summary"]["delivery"]["settlements"], "recording": pin})
        event = rows[(first_seed, condition, policy)]["event_metrics"]
        if event["onset_h"] != rows[(first_seed, condition, reference)]["event_metrics"]["onset_h"] or event["end_h"] != rows[(first_seed, condition, reference)]["event_metrics"]["end_h"]:
            raise ValueError("Compared example controllers faced different event intervals")
        data["traces"].append({"condition": condition, "environment_seed": first_seed,
                               "event": {k: event[k] for k in ("is_intervention", "onset_h", "end_h")},
                               "controllers": traces})
    return data


def context_label(data):
    text = f"{data['split'].upper()} · {data['scenario_count']} environment seeds · {len(data['recipe_members'])} fixed policies"
    if data["split"] != "final":
        text += " · not final evidence"
    efforts = data["recipe_training_effort"]
    descriptions = list(dict.fromkeys(e.get("compact", "Training method not recorded") for e in efforts))
    if len(descriptions) == 1:
        text += "\n"+descriptions[0]+(" per policy" if len(efforts)>1 else "")
    else:
        algorithms = list(dict.fromkeys(e.get("algorithm", "unrecorded") for e in efforts))
        text += "\n"+" / ".join(algorithms)+"; per-policy training effort is in figure-data.json"
    return "\n".join(textwrap.fill(line, width=118) for line in text.splitlines())


def figure_title(fig, title, data):
    fig.suptitle(title, x=.08, y=.98, ha="left", fontsize=15, fontweight="bold")
    fig.text(.08, .91, context_label(data), ha="left", va="top", fontsize=9,
             color="#955914" if data["split"] != "final" else "#17684a")


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c5d0c7")
    ax.tick_params(colors="#40564e", labelsize=9)
    ax.grid(axis="y", color="#e6ebe6", linewidth=.7, zorder=0)
    ax.set_axisbelow(True)


def condition_figure(data, plt):
    import numpy as np
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(10.2, 5.7))
    fig.subplots_adjust(left=.09, right=.98, bottom=.24, top=.68)
    figure_title(fig, "Unmet methane by operating condition", data)
    n = len(data["series"])
    width = .78/n
    positions = np.arange(len(data["conditions"]))
    handles, labels = [], []
    for j, series in enumerate(data["series"]):
        values = [c["bars"][j]["mean_unmet_kg"] for c in data["conditions"]]
        offset = (j-(n-1)/2)*width
        bars = ax.bar(positions+offset, values, width*.9, color=COLORS[j%len(COLORS)], zorder=3)
        reference_kind = "Configured reference" if data["controller_metadata"][data["reference"]].get("config") else "Original reference"
        role = "Recipe mean" if j == 0 else reference_kind if series["role"] == "reference" else series["role"].capitalize()
        label = series["population"].removeprefix("recipe:")
        handles.append(bars)
        labels.append(role+"\n"+textwrap.fill(label, 29))
    jitter = np.linspace(-width*.22, width*.22, len(data["recipe_members"]))
    for i, condition in enumerate(data["conditions"]):
        ax.scatter([i-(n-1)/2*width+j for j in jitter],
                   [p["mean_unmet_kg"] for p in condition["individual_policy_means"]],
                   s=22, marker="o", facecolors="white", edgecolors="#172c24", linewidths=.9, zorder=5, clip_on=False)
    handles.append(Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="#172c24", markersize=5))
    labels.append("Individual fixed-policy means")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, .815), ncol=min(3, len(handles)), frameon=False, fontsize=8.5)
    ax.set_xticks(positions, [CONDITION_LABELS[c["condition"]] for c in data["conditions"]])
    ax.set_ylabel("Mean weekly unmet methane (kg)", fontsize=10)
    ax.set_ylim(bottom=0)
    tidy(ax)
    fig.text(.09, .075, "Bars average all retained environment seeds. Open dots show separate fixed policies; they are not an ensemble.\nOrdinary includes random equipment faults; healthy control does not. No delivery failure is omitted.", fontsize=8.5, va="bottom", color="#40564e")
    return fig


def scatter_figure(data, plt):
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(8.7, 6.6))
    fig.subplots_adjust(left=.15, right=.97, bottom=.25, top=.75)
    figure_title(fig, "Paired changes within each environment", data)
    fig.text(.15, .79, "Recipe: "+data["recipe"]+"\nReference: "+data["reference"], fontsize=8.2, va="bottom", color="#40564e")
    points = data["scatter"]["points"]
    xs = [p["stress_unmet_difference_kg"] for p in points]
    ys = [p["mixture_contribution_difference_eur"] for p in points]
    def limits(values):
        lo, hi = min(0, *values), max(0, *values)
        pad = max((hi-lo)*.14, 1)
        return lo-pad, hi+pad
    xmin, xmax = limits(xs)
    ymin, ymax = limits(ys)
    ax.set(xlim=(xmin,xmax), ylim=(ymin,ymax))
    ax.add_patch(Rectangle((xmin, 0), -xmin, ymax, color="#e6f1e6", zorder=0))
    ax.axvline(0, color="#84948b", linestyle="--", linewidth=.8)
    ax.axhline(0, color="#84948b", linestyle="--", linewidth=.8)
    ax.scatter(xs, ys, s=45, color=COLORS[0], edgecolors="white", linewidths=.8, zorder=3,
               label=f"One point per environment seed (n={len(points)})")
    ax.scatter([data["scatter"]["mean_x"]], [data["scatter"]["mean_y"]], s=74, marker="D", color="#243831", edgecolors="white", linewidths=.8, zorder=4, label="Mean across environment seeds")
    if len(points) <= 8:
        for p in points:
            ax.annotate(str(p["environment_seed"]), (p["stress_unmet_difference_kg"], p["mixture_contribution_difference_eur"]), xytext=(5,6), textcoords="offset points", fontsize=7.5, color="#40564e")
    ax.set_xlabel("Stress unmet methane difference (kg/week)\nRecipe mean − reference", fontsize=10)
    ax.set_ylabel("Mixture contribution difference (€/week)\nRecipe mean − reference", fontsize=10)
    tidy(ax)
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.text(.15, .055, "Upper-left indicates less unmet methane and higher contribution.\nStress: 20% per stress condition. Mixture: 50% ordinary + 10% per stress condition.\nAll recipe policies are averaged within each environment; these are declared weights, not event frequencies.", fontsize=8.2, va="bottom", color="#40564e")
    return fig


def trace_figure(data, trace, plt):
    fig, axes = plt.subplots(3, 1, figsize=(9.8, 7.0), sharex=True)
    fig.subplots_adjust(left=.10, right=.98, bottom=.14, top=.73, hspace=.17)
    title = "Prescribed solar-loss example: 36 hours" if trace["condition"] == "pv36" else "Prescribed methanator-outage example: 48 hours"
    figure_title(fig, title, data)
    event = trace["event"]
    handles = []
    for j, controller in enumerate(trace["controllers"]):
        color, style = COLORS[j], "-" if j == 0 else "--"
        label = ("Policy: " if j == 0 else "Reference: ")+controller["controller"]
        for ax, field in zip(axes, ("methane_kg", "battery_soc_percent", "total_process_power_mw")):
            line, = ax.plot(controller["elapsed_h"], controller[field], color=color, linestyle=style, linewidth=1.1, label=label)
            if ax is axes[0]:
                handles.append(line)
    for ax, label in zip(axes, ("Methane stock\n(kg)", "Battery charge\n(%)", "Total process power\n(MW)")):
        if event["is_intervention"]:
            ax.axvspan(event["onset_h"], event["end_h"], color="#f4debd", alpha=.72, linewidth=0, zorder=0)
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(bottom=0)
        tidy(ax)
    axes[1].set_ylim(0,100)
    axes[-1].set(xlim=(0,168), xticks=range(0,169,24), xlabel="Elapsed hours from episode start (D1 06:00)")
    fig.legend(handles, [h.get_label() for h in handles], loc="upper left", bbox_to_anchor=(.10,.825), ncol=1, frameon=False, fontsize=8.5)
    fig.text(.10, .03, f"First frozen environment seed: {trace['environment_seed']} (environment seed not selected by outcome). Policy explicitly named in build arguments.\nFull one-minute traces; shaded interval is the declared disruption. Process power sums electrolysis, capture and methanation.", fontsize=8.2, va="bottom", color="#40564e")
    return fig


def build(panel, output, *, recipe, reference, policy, imitation=None, formats=("png",)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    formats = tuple(dict.fromkeys(formats))
    if not formats or any(fmt not in ("png", "pdf") for fmt in formats):
        raise ValueError("Only PNG and optional PDF exports are supported")
    data = collect(panel, recipe=recipe, reference=reference, policy=policy, imitation=imitation)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    publish(output/"figure-data.json", data)
    freeze = {"schema": SCHEMA+"-freeze", "created_at": datetime.now(timezone.utc).isoformat(),
              "arguments": {"panel": str(Path(panel).resolve()), "recipe": recipe, "reference": reference,
                            "policy": policy, "imitation": imitation, "formats": formats},
              "figure_data": inspector.file_pin(output/"figure-data.json"), "input_provenance": data["provenance"],
              "source_files": [inspector.file_pin(Path(__file__).with_name("__init__.py")), inspector.file_pin(__file__), inspector.file_pin(inspector.__file__)],
              "libraries": {"matplotlib": matplotlib.__version__, "numpy": np.__version__}}
    publish(output/"figure-freeze.json", freeze)
    paths = []
    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.labelcolor": "#243831", "text.color": "#243831", "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42}
    with plt.rc_context(style):
        figures = [("condition-unmet", condition_figure(data, plt)), ("paired-environment-changes", scatter_figure(data, plt))]
        figures.extend(("trace-"+trace["condition"], trace_figure(data, trace, plt)) for trace in data["traces"])
        for name, figure in figures:
            for fmt in formats:
                path = output/(name+"."+fmt)
                figure.savefig(path, dpi=190)
                paths.append(inspector.file_pin(path))
            plt.close(figure)
    # Input/model evidence is immutable during the rendering step as well.
    inspector.verify_pin(freeze["figure_data"])
    for pin in (data["provenance"]["freeze"], data["provenance"]["summary"], data["provenance"]["analysis"]):
        inspector.verify_pin(pin)
    result = {"schema": SCHEMA+"-manifest", "completed_at": datetime.now(timezone.utc).isoformat(),
              "split": data["split"], "case_count": data["case_count"], "scenario_count": data["scenario_count"],
              "recipe_members": data["recipe_members"], "example_seed": data["seeds"][0],
              "figure_freeze": inspector.file_pin(output/"figure-freeze.json"), "files": paths,
              "all_cases_complete_and_audited": True, "failed_deliveries_omitted": False,
              "policy_evaluations_executed": 0}
    publish(output/"figure-manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing output is never overwritten")
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--imitation", help="Optional explicitly identified imitation reference already in the panel")
    parser.add_argument("--format", choices=("png", "pdf"), action="append", dest="formats")
    args = parser.parse_args()
    result = build(args.panel, args.output, recipe=args.recipe, reference=args.reference, policy=args.policy,
                   imitation=args.imitation, formats=args.formats or ("png",))
    print(json.dumps({"output": str(args.output), "split": result["split"], "cases": result["case_count"],
                      "scenarios": result["scenario_count"], "figures": [f["path"] for f in result["files"]]}))


if __name__ == "__main__":
    main()

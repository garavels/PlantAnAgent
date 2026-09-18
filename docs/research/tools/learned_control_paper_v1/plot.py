"""Reformat the frozen scientific figure export for a two-column paper.

No simulation, statistical estimation, scenario selection, or data smoothing.
The input must be the authenticated data file of a completed v2 figure build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def build(figure_directory, output):
    figure_directory, output = Path(figure_directory).resolve(), Path(output).resolve()
    manifest_path = figure_directory / "figure-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    data_path = figure_directory / "figure-data.json"
    if manifest.get("schema") != "learned-control-scientific-figures-v2-manifest":
        raise ValueError("Expected the completed v2 scientific figure manifest")
    freeze_path = figure_directory / "figure-freeze.json"
    if pin(freeze_path) != manifest["figure_freeze"]:
        raise ValueError("Scientific figure freeze does not match its manifest")
    freeze = json.loads(freeze_path.read_text())
    if pin(data_path) != freeze["figure_data"]:
        raise ValueError("Scientific figure data do not match their authenticated manifest")
    if manifest.get("all_cases_complete_and_audited") is not True:
        raise ValueError("Incomplete or unaudited panel")
    data = json.loads(data_path.read_text())
    if data.get("schema") != "learned-control-scientific-figures-v2-data" or data.get("all_cases_audited") is not True:
        raise ValueError("Expected complete audited v2 figure data")
    if not data["seeds"] or data["example_policy"] not in data["recipe_members"]:
        raise ValueError("Missing registered example policy or scenario")
    if output.exists():
        raise FileExistsError("Paper figure output must be a new directory")
    output.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.linewidth": .5, "lines.linewidth": .85,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 220,
        "path.simplify": False})
    files, plotted = [], {}
    # Same preselected first-seed examples as the source builder, both retained.
    for condition in ("pv36", "meth48"):
        cases = [t for t in data["traces"] if t["condition"] == condition]
        if len(cases) != 1 or cases[0]["environment_seed"] != data["seeds"][0]:
            raise ValueError("Trace must be the unique prescribed first-seed example")
        trace = cases[0]
        by_label = {t["controller"]: t for t in trace["controllers"]}
        reference_metadata = data["controller_metadata"][data["reference"]]
        reference_name = "Tuned rule" if reference_metadata.get("config") else "Rule baseline"
        series = [(data["example_policy"], "Learned", "#157154", "-"),
                  (data["reference"], reference_name, "#455765", "--")]
        fig, axes = plt.subplots(3, 1, figsize=(3.28, 2.55), sharex=True)
        fields = [("methane_kg", "CH₄ (kg)"), ("battery_soc_percent", "Battery (%)"),
                  ("total_process_power_mw", "Power (MW)")]
        for label, display, color, style in series:
            row = by_label[label]
            x = np.asarray(row["elapsed_h"], dtype=float)
            if row["environment_seed"] != data["seeds"][0] or len(x) != 10081 or x[0] != 0 or x[-1] != 168:
                raise ValueError("Expected the complete unsmoothed one-minute seven-day trace")
            for ax, (field, ylabel) in zip(axes, fields):
                y = np.asarray(row[field], dtype=float)
                if y.shape != x.shape or not np.isfinite(y).all():
                    raise ValueError("Invalid trace coordinates")
                ax.plot(x, y, color=color, linestyle=style, label=display)
                ax.set_ylabel(ylabel)
        for ax in axes:
            ax.axvspan(trace["event"]["onset_h"], trace["event"]["end_h"], color="#f3d9aa", alpha=.6, zorder=-2)
            ax.grid(axis="y", linewidth=.35, color="#d5dfd9", zorder=-1)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=2, width=.5)
            ax.set_xlim(0, 168)
            ax.set_ylim(bottom=0)
        axes[1].set_ylim(0, 100)
        axes[-1].set_xticks(range(0, 169, 24))
        axes[-1].set_xlabel("Elapsed hours (start: 06:00)")
        axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False,
                       borderaxespad=0, handlelength=2.1, columnspacing=1.3)
        fig.subplots_adjust(left=.20, right=.965, bottom=.14, top=.91, hspace=.16)
        for extension in ("pdf", "png"):
            path = output / f"paper-trace-{condition}.{extension}"
            fig.savefig(path)
            files.append(pin(path))
        plt.close(fig)
        plotted[condition] = {"environment_seed": trace["environment_seed"],
            "policy": data["example_policy"], "reference": data["reference"],
            "event": trace["event"], "fields": [f for f, _ in fields],
            "samples_per_series": 10081, "full_horizon_h": [0, 168], "smoothing": None}
    receipt = {"schema": "learned-control-compact-paper-figures-v1", "split": data["split"],
        "source": pin(Path(__file__)), "input_manifest": pin(manifest_path),
        "input_freeze": pin(freeze_path), "input_data": pin(data_path), "files": files, "plotted": plotted,
        "caption_requirements": "Identify one prescribed first-environment-seed trajectory and the explicitly selected fixed policy, not a recipe-mean trajectory. Shading marks the declared outage. Power is the sum of electrolysis, capture and methanation power, excluding battery charging. State whether panel is development or final. These examples do not establish aggregate superiority."}
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.figures, args.output), indent=2))

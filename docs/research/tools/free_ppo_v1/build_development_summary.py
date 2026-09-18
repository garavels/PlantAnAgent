"""Create a clearly interim, data-bound view of the three completed checkpoints."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs/research/results/free_ppo_20260917/development-inspection"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = OUT / "analysis/analysis.json"
    analysis = json.loads(source.read_text())
    if any(g["split"] != "development" for g in analysis["groups"]):
        raise ValueError("This interim view admits development results only")
    groups = {g["controller"]: g for g in analysis["groups"]}
    seeds, steps = (101, 211, 307), (0, 65536, 393216)
    for seed in seeds:
        for step in steps:
            g = groups[f"free-ppo-v1-{seed}-step-{step}"]
            if g["n_scenarios"] != 8 or g["complete_weeks"] != 8:
                raise ValueError("Missing development cases")

    def metric(g, key):
        value = g["metrics"][key]["all_case"]["mean"]
        if value is None:
            raise ValueError(f"Missing complete-week metric: {key}")
        return value

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), layout="constrained")
    colors = {101: "#176D72", 211: "#D48534", 307: "#5E6798"}
    for seed in seeds:
        points = [groups[f"free-ppo-v1-{seed}-step-{step}"] for step in steps]
        axes[0].plot([s / 1000 for s in steps], [metric(g, "net_operating_value_eur") / 1000 for g in points],
                     marker="o", linewidth=1.8, markersize=5, color=colors[seed], label=f"Policy {seed}")
        axes[1].plot([s / 1000 for s in steps], [g["delivery_successes"] for g in points],
                     marker="o", linewidth=1.8, markersize=5, color=colors[seed], label=f"Policy {seed}")
    axes[0].set_ylabel("Contribution (thousand EUR/week)")
    axes[0].set_title("Learning improves the modeled objective", fontsize=11)
    axes[1].set_ylabel("Weeks meeting every delivery (out of 8)")
    axes[1].set_title("Delivery performance on development weeks", fontsize=11)
    axes[1].set_ylim(-.3, 9.5)
    axes[1].set_yticks([0, 2, 4, 6, 8])
    for ax in axes:
        ax.axvspan(393.216, 786.432, color="#777777", alpha=.06)
        ax.axvline(786.432, color="#888888", linestyle="--", linewidth=.8)
        ax.text(.77, .78, "Final checkpoint\nnot evaluated yet", ha="center", va="center", transform=ax.transAxes, color="#666666", fontsize=9)
        ax.set_xlim(-15, 825)
        ax.set_xticks([0, 65.536, 393.216, 786.432], ["0", "66", "393", "786"])
        ax.set_xlabel("Training decisions (thousands)")
        ax.grid(alpha=.17)
    axes[0].legend(frameon=False, ncol=3, fontsize=8, loc="lower right")
    fig.suptitle("Interim results · the same eight development scenarios · final test set unopened", fontsize=11)
    fig.savefig(OUT / "learning-so-far.png", dpi=180)
    fig.savefig(OUT / "learning-so-far.pdf")
    plt.close(fig)

    lines = ["# Results available so far", "", "**Interim development results.** Training continues to its fixed 786,432-decision endpoint. These are the same eight development weeks at three saved checkpoints; the 32 final test scenarios remain unopened.", "",
             "## Did the policies learn?", "", "| Policy | Initial: every delivery | Early: 65,536 decisions | Halfway: 393,216 decisions | Halfway contribution (EUR/week) |", "|---|---:|---:|---:|---:|"]
    for seed in seeds:
        ps = [groups[f"free-ppo-v1-{seed}-step-{s}"] for s in steps]
        lines.append(f"| {seed} | {ps[0]['delivery_successes']}/8 | {ps[1]['delivery_successes']}/8 | {ps[2]['delivery_successes']}/8 | {metric(ps[2], 'net_operating_value_eur'):,.0f} |")
    lines += ["", "Every-delivery success means meeting all 28 commitments, totaling 4,200 kg of methane in one week. Contribution is the synthetic reward in euros after missed deliveries, modeled wear and depletion costs; it is not calibrated plant profit.", "", "![Actual learning curves](learning-so-far.png)", "",
              "## What remains unresolved?", "", "The halfway policies end with about 3.74 MWh less battery energy than they started with, on average. All three have zero mean terminal material deficit in methane-equivalent terms on these development cases. They meet deliveries while drawing down starting electrical reserves. In the prospectively selected first development scenario, all three batteries also recharge and discharge on later days after first approaching the 5% floor. Ending below the initial level does not by itself prove an unsustainable strategy: continuous multiweek tests are needed to determine whether the opening reserve is necessary or the policy can sustain a lower reserve.", "",
              "More training has not produced monotonically higher economic contribution. All three midpoint contributions are slightly below their early checkpoint values. Every final policy will still be tested and reported; these diagnostics do not select a preferred checkpoint.", "",
              "The translated prior controller meets every delivery in 0/8 cases. Its once-per-15-minute solar-share interface disrupts startup when sunlight changes, so this does not establish superiority over the original controller with its native allocator.", "",
              "An exploratory input-scaling check retained 23/24 cases before the laptop suspension caused one timeout. Random actors with learned midpoint normalization still achieved only 6/23 complete-delivery weeks; learned midpoint actors achieved 23/23 on those same retained scenarios. This supports a role for learned network weights, with the missing case and exploratory design explicitly retained in the diagnostic evidence.", "",
              "## Inspect the evidence", "", "- [Interactive development inspector](inspector.html): select a policy and scrub through the first development week's power, requested controls, stocks and deliveries.", "- [Machine-readable case table](analysis/results.csv)", "- [Complete interim analysis](analysis/analysis.json)", "- [Normalization diagnostic](../preflight/normalization-ablation/README.md)", "- [First-principles guide and editing map](../../../FREE_CONTROL_GUIDE.md)", "",
              f"This snapshot binds analysis SHA-256 `{hashlib.sha256(source.read_bytes()).hexdigest()}`. It includes {len(analysis['rows'])} canonical development evaluations; the optional normalization diagnostic is separate.", ""]
    (OUT / "RESULTS_SO_FAR.md").write_text("\n".join(lines))
    print(json.dumps({"summary": str(OUT / "RESULTS_SO_FAR.md"), "figure": str(OUT / "learning-so-far.png")}))


if __name__ == "__main__":
    main()

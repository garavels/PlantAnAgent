"""Render descriptive startup diagnostic outcomes, without population inference."""
import argparse
import gzip
import json
from pathlib import Path


LABELS = {"daylight": "Daylight", "storage": "Storage", "ppo-101": "PPO 101",
          "ppo-211": "PPO 211", "ppo-307": "PPO 307", "feasibility-reference": "Feasibility reference"}
ORDER = tuple(LABELS)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    args = p.parse_args()
    root = args.results.resolve()
    data = json.loads((root / "summary.json").read_text())
    if not data["all_executions_and_audits_passed"] or not data["daylight_bound_and_feasible_witness_verified"]:
        raise ValueError("Interpret retained failures before making a final report")
    rows = {(r["condition"], r["controller"]): r for r in data["rows"]}
    if len(rows) != 12:
        raise ValueError("Expected all 12 prescribed rollouts")
    lines = ["# Six-hour loss of solar at startup", "",
        "Evaluation only: all three previously trained PPO policies are frozen. No policy was retrained or selected on this test. "
        "This is one deliberately constructed diagnostic and one matched clear-sky condition, not 12 independent environmental samples.", "",
        "The plant starts at 06:00 with 120 kg methane, 100 kg hydrogen, 600 kg CO2 and 25% battery charge. "
        "It owes 180 kg methane at noon. In the stress condition, no solar electricity is available before that deadline. "
        "Equipment is healthy, other faults are removed, and all cases retain the seven-day horizon and original economics. "
        "The original causal ensemble forecast service is retained; it has no access to scheduled restoration time.", "",
        "## First delivery: the primary outcome", "",
        "The same initial stocks are used for every controller. The feasibility reference is an engineered witness, not a learned or competitive autonomous controller.", "",
        "| Controller | Solar-loss delivered / 180 kg | Solar-loss unmet, kg | Clear-sky delivered / 180 kg | Stress methane made, kg | Stress battery discharge, kWh | Stress battery at 6h |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for c in ORDER:
        stress, clear = (rows[k, c]["first_six_hours"] for k in ("solar-loss", "clear-sky"))
        lines.append(f"| {LABELS[c]} | {stress['delivery']['delivered_kg']:.2f} | {stress['delivery']['unmet_kg']:.2f} | "
            f"{clear['delivery']['delivered_kg']:.2f} | {stress['state']['ch4_produced_total_kg']:.2f} | "
            f"{stress['operating_metrics']['battery_discharge_kwh']:.2f} | {100*stress['state']['battery_soc']:.2f}% |")
    lines += ["", "![First six hours with no solar](first-six-hours-solar-loss.png)", "",
        "## Where the battery electricity went", "",
        "Actual accepted energy during the six-hour solar-loss interval. Requested support that equipment refuses is not charged as delivered energy.", "",
        "| Controller | Electrolysis, kWh | Carbon capture, kWh | Methanation, kWh | Methanator preparation, kWh | Methanator productive input, kWh |",
        "|---|---:|---:|---:|---:|---:|"]
    for c in ORDER:
        op = rows["solar-loss", c]["first_six_hours"]["operating_metrics"]
        values = [op[k] for k in ("electrolyser_battery_kwh", "dac_battery_kwh", "methanator_battery_kwh", "methanator_startup_kwh", "methanator_productive_kwh")]
        lines.append(f"| {LABELS[c]} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    lines += ["", "## Seven-day continuation: a secondary outcome", "",
        "All controllers continue for 168 hours, with 28 deliveries. Successful recovery of the first delivery does not establish success across the week. "
        "The reference switches to the frozen storage heuristic after the first six hours; its full-week result is not attributed to autonomous emergency planning.", "",
        "| Controller | Condition | Deliveries met / 28 | Total unmet, kg | Modeled contribution, EUR/week |",
        "|---|---|---:|---:|---:|"]
    for c in ORDER:
        for condition in ("solar-loss", "clear-sky"):
            r = rows[condition, c]
            passed = sum(x["unmet_kg"] <= 1e-9 for x in r["delivery"]["settlements"])
            lines.append(f"| {LABELS[c]} | {condition} | {passed}/28 | {r['economic_metrics']['unmet_kg']:.2f} | {r['economic_metrics']['net_operating_value_eur']:,.2f} |")
    lines += ["", "![Matched clear-sky first six hours](first-six-hours-clear-sky.png)", "",
        "## Verification and scope", "",
        "All 12 full-week executions pass the independent physical and economic auditors. Every minute state, "
        "raw policy observation, normalized PPO input, raw actor mean and requested action is retained. "
        "Initial states and realized solar forcing match across controllers within each condition; opening resource stocks match across both conditions.", "",
        "The witness is fixed before evaluation: one hour of battery-to-methanator power at 90 kW, then 30 minutes at 150 kW, "
        "then all-off until delivery. It uses approximately 165 kWh, makes 61.111 kg methane and leaves a 1.111 kg margin. "
        "The daylight controller's 120 kg delivery bound follows from zero electricity and non-increasing methane without production.", "",
        "This test can demonstrate or refute recovery in this particular constructed state. It cannot estimate rare-event frequency, "
        "population reliability, optimality, or whether normal-operation costs purchase general tail protection. "
        "The supplied opening charge and feedstocks are identical for every controller; a later full-week disruption study would also test "
        "whether controllers preserve those resources before an unexpected event.", "",
        "Reproduce with the frozen whole-plant source/runtime and checkpoints from the preceding study, plus the files in "
        "`docs/research/tools/startup_stress_v1/`. `freeze.json` pins every scientific dependency. "
        "Do not overwrite completed cases; a new execution must use a fresh result directory and freeze. "
        "`summary.json` and `cases/<condition>--<controller>/recording.json.gz` contain the results and complete traces.", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"daylight": "#a57722", "storage": "#254c65", "ppo-101": "#137f6e", "ppo-211": "#b14e5a", "ppo-307": "#7664ae", "feasibility-reference": "#303030"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for condition in ("solar-loss", "clear-sky"):
        fig, axes = plt.subplots(3, 1, figsize=(10.5, 8), sharex=True, layout="constrained")
        for c in ORDER:
            record = json.loads(gzip.decompress(Path(rows[condition, c]["recording"]["path"]).read_bytes()))
            states = record["minute_truth_states"][:361]
            x = [s["step"]/60 for s in states]
            ys = ([120+s["ch4_produced_total_kg"] for s in states],
                  [100*s["battery_soc"] for s in states],
                  [s["methanator_power_kw"] for s in states])
            for ax, y in zip(axes, ys):
                ax.plot(x, y, label=LABELS[c], color=colors[c], linewidth=1.7,
                    linestyle="--" if c == "feasibility-reference" else "-", alpha=.95)
        axes[0].axhline(180, color="#777777", linestyle=":", linewidth=1.1)
        axes[0].set_ylabel("Opening methane +\nproduction (kg)")
        axes[0].set_title("First delivery after six hours without solar" if condition == "solar-loss" else "Matched clear-sky startup")
        axes[0].legend(ncol=3, fontsize=9, loc="upper left")
        axes[1].set_ylabel("Battery charge (%)")
        axes[2].set_ylabel("Actual methanator\npower (kW)")
        axes[2].set_xlabel("Hours since 06:00 start — first 180 kg delivery due at hour 6")
        for ax in axes:
            ax.grid(alpha=.2)
            ax.set_xlim(0, 6)
        fig.savefig(root / f"first-six-hours-{condition}.png", dpi=170)
        fig.savefig(root / f"first-six-hours-{condition}.svg")
        plt.close(fig)
    print(json.dumps({"report": str(root / "REPORT.md"), "case_count": len(rows)}))


if __name__ == "__main__":
    main()

"""Build a two-page commission report only from completed, pinned study results.

Reporting-only runtime: stdlib, reportlab, matplotlib and pypdf. No environment,
learner or evaluation imports; this command cannot open a new scenario.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "docs/research/results/authority_ppo_20260917"
DEFAULT_DRAFT = RESULTS / "WRITEUP_DRAFT.md"
DIAGNOSIS = RESULTS / "diagnosis"
SCHEMA = "authority-ppo-v2-evaluation-analysis"
OPTS = (101, 211, 307)
ARMS = ("narrow", "broad")
COVERAGE = "daily-coverage-offgrid-v1"
MPC = "daily-return-mission-mpc-v1"
LABELS = (COVERAGE, *(f"authority-ppo-v2-{a}-{s}" for a in ARMS for s in OPTS))
PASSES = ("execution_complete", "mission_passed", "every_delivery_passed", "all_daily_returns_passed", "hard_safety_passed")


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def training_plan_digest(plan: dict) -> str:
    # The training custody format is sorted, pretty JSON with a trailing newline;
    # evaluation freeze identities use the compact canonical() format instead.
    encoded = (json.dumps(plan, allow_nan=False, ensure_ascii=True, indent=2,
                          separators=(",", ": "), sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pin(path: Path) -> dict:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def check_pin(record: dict) -> Path:
    path = Path(record["path"])
    actual = pin(path)
    if actual["bytes"] != record["bytes"] or actual["sha256"] != record["sha256"]:
        raise ValueError(f"Pinned artifact changed: {path}")
    return path


def finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"Expected a finite numeric {name}")
    return float(value)


def label(value: str) -> str:
    if value == COVERAGE:
        return "Coverage"
    if value == MPC:
        return "Strict MPC"
    return value.replace("authority-ppo-v2-", "").replace("-", " / ").title()


def validate_inputs(analysis_path: Path, training_root: Path, diagnosis: Path, evaluation_freeze_path: Path) -> dict:
    """Reject absent panels, changed artifacts or uncompleted training endpoints.

    Failed evaluation attempts are valid observations. Requiring panel completeness
    never requires that a policy completed or qualified every simulated week.
    """
    analysis = load(analysis_path)
    if analysis.get("schema") != SCHEMA or not analysis.get("evaluation_freeze_sha256"):
        raise ValueError("Missing completed analysis or held-out evaluation freeze")
    if len(analysis["evaluation_freeze_sha256"]) != 64:
        raise ValueError("Malformed held-out evaluation freeze digest")
    for record in analysis["summaries"]:
        check_pin(record)
    check_pin(analysis["analysis_source"])
    seen = {}
    for row in analysis["rows"]:
        key = (row["split"], row["controller"], row["seed"])
        if key in seen:
            raise ValueError(f"Duplicate outcome: {key}")
        if not all(isinstance(row.get(k), bool) for k in PASSES):
            raise ValueError(f"Missing binary qualifications: {key}")
        if row["mission_passed"] != all(row[k] for k in PASSES if k != "mission_passed"):
            raise ValueError(f"Inconsistent mission qualification: {key}")
        seen[key] = row
    for split, seeds in (("development", range(9950000, 9950008)), ("held_out", range(9960000, 9960032))):
        for controller in LABELS:
            actual = {r["seed"] for r in analysis["rows"] if r["split"] == split and r["controller"] == controller}
            if actual != set(seeds):
                raise ValueError(f"Missing or extra required outcomes: {split}/{controller}")
    mpc_rows = [r for r in analysis["rows"] if r["split"] == "held_out" and r["controller"] == MPC]
    if {r["seed"] for r in mpc_rows} != set(range(9960000, 9960008)):
        raise ValueError("The eight-case supplementary MPC panel is missing")
    groups = {}
    for controller in (*LABELS, MPC):
        found = [g for g in analysis["groups"] if g["split"] == "held_out" and g["controller"] == controller]
        if len(found) != 1:
            raise ValueError(f"Missing or duplicated aggregate: {controller}")
        group = found[0]
        rows = [r for r in analysis["rows"] if r["split"] == "held_out" and r["controller"] == controller]
        if group["n_scenarios"] != len(rows) or group["full_missions"] != sum(r["mission_passed"] for r in rows) or group["complete_weeks"] != sum(r["execution_complete"] for r in rows):
            raise ValueError(f"Aggregate counts differ from retained rows: {controller}")
        for metric in PASSES:
            estimate = group["metrics"][metric]["all_case"]
            expected = sum(r[metric] for r in rows) / len(rows)
            if estimate["n_scenarios"] != len(rows) or abs(finite(estimate["mean"], metric) - expected) > 1e-12:
                raise ValueError(f"Qualification aggregate differs: {controller}/{metric}")
        contribution = group["metrics"]["net_operating_value_eur"]["complete_observed"]
        if contribution["n_scenarios"] != group["complete_weeks"]:
            raise ValueError(f"Contribution and completed-week denominators differ: {controller}")
        if group["complete_weeks"]:
            finite(contribution["mean"], "completed-week operating contribution")
        groups[controller] = group
    plan = load(training_root / "plan.json")
    endpoints = load(training_root / "endpoints-frozen.json")
    evaluation_freeze = load(evaluation_freeze_path)
    freeze_digest = hashlib.sha256(canonical({k: v for k, v in evaluation_freeze.items() if k != "sha256"})).hexdigest()
    if (evaluation_freeze.get("sha256") != freeze_digest or analysis["evaluation_freeze_sha256"] != freeze_digest
        or canonical(evaluation_freeze.get("endpoints")) != canonical(endpoints)
        or analysis["benchmark_contract_sha256"] != plan["benchmark_contract_sha256"]
        or analysis["benchmark_contract_sha256"] != evaluation_freeze["benchmark_contract_sha256"]):
        raise ValueError("Analysis, evaluation freeze and trained endpoints are not the same study")
    if plan["total_transitions_per_policy"] != 262144 or plan["optimizer_seeds"] != list(OPTS) or plan["arms"] != list(ARMS):
        raise ValueError("Training plan differs from the described comparison")
    if endpoints["plan_sha256"] != training_plan_digest(plan):
        raise ValueError("Frozen endpoints belong to a different training plan")
    expected = {(a, s) for a in ARMS for s in OPTS}
    if {(r["arm"], r["optimizer_seed"]) for r in endpoints["endpoints"]} != expected or len(endpoints["endpoints"]) != 6:
        raise ValueError("All six fixed final endpoints are required")
    training_results = []
    for endpoint in endpoints["endpoints"]:
        result = load(check_pin(endpoint["result"]))
        check_pin(endpoint["checkpoint"])
        if (result["arm"], result["optimizer_seed"]) != (endpoint["arm"], endpoint["optimizer_seed"]):
            raise ValueError("Endpoint identity differs")
        if result["final_checkpoint"] != endpoint["checkpoint"]:
            raise ValueError("Training result and frozen checkpoint differ")
        if result["num_timesteps"] != 262144 or not result.get("checkpoint_reopen_verified") or not result.get("actor_mean_parameters_changed"):
            raise ValueError("A completed, reopenable trained actor endpoint is missing")
        training_results.append(result)
    diagnostic = load(diagnosis / "selected-full-weeks.json")
    if {r["seed"] for r in diagnostic["rows"]} != {9930001, 9930011} or len(diagnostic["rows"]) != 2:
        raise ValueError("Expected both previously exposed diagnostic cases")
    for row in diagnostic["rows"]:
        if not all(row.get(k) for k in ("execution_complete", "mission_passed", "all_daily_returns_passed", "every_delivery_passed", "hard_safety_passed")):
            raise ValueError("Constructive diagnostic gate no longer passes")
        check_pin(row["recording"])
    plot_sources = load(diagnosis / "day1-plot-data-sources.json")
    check_pin(plot_sources["csv"])
    mechanism_path = diagnosis / "held-out-mechanism-comparison.json"
    mechanism = load(mechanism_path)
    if (mechanism.get("evaluation_freeze_sha256") != freeze_digest
        or mechanism.get("rescued_cases") != [9960020, 9960023]
        or mechanism.get("regressed_cases") != []):
        raise ValueError("Post hoc mechanism note differs from the described frozen comparison")
    for record in mechanism["sources"]:
        check_pin({**record, "path": str(ROOT / record["path"])})
    return {"analysis": analysis, "groups": groups, "training": training_results, "diagnostic": diagnostic,
            "analysis_pin": pin(analysis_path), "training_plan_pin": pin(training_root / "plan.json"),
            "endpoint_pin": pin(training_root / "endpoints-frozen.json"), "diagnosis_pin": pin(diagnosis / "selected-full-weeks.json"),
            "evaluation_freeze_pin": pin(evaluation_freeze_path), "mechanism": mechanism,
            "mechanism_pin": pin(mechanism_path)}


def estimate_text(estimate: dict, *, percent=False) -> str:
    if estimate.get("mean") is None:
        return "unavailable because at least one full-week measurement is missing"
    factor, unit = (100., " percentage points") if percent else (1., "")
    mean = finite(estimate["mean"], "estimate") * factor
    low, high = [finite(v, "interval") * factor for v in estimate["ci95"]]
    return f"{mean:+.2f}{unit} (95% interval {low:+.2f} to {high:+.2f})"


def arm_result(analysis: dict, arm="broad", reference="narrow") -> dict:
    items = [r for r in analysis["arm_comparisons"] if r["split"] == "held_out" and r["arm"] == arm and r["reference_arm"] == reference]
    if len(items) != 1 or items[0]["n_scenarios"] != 32:
        raise ValueError("Required paired arm comparison is absent")
    return items[0]


def report_text(data: dict) -> dict:
    analysis, groups = data["analysis"], data["groups"]
    counts = {arm: [groups[f"authority-ppo-v2-{arm}-{opt}"]["full_missions"] for opt in OPTS] for arm in ARMS}
    paired = arm_result(analysis)["metrics"]["mission_passed"]["all_case"]
    result = (f"Coverage qualified {groups[COVERAGE]['full_missions']}/32 weeks. The three narrow policies qualified "
              f"{min(counts['narrow'])}-{max(counts['narrow'])}/32; broad policies qualified {min(counts['broad'])}-{max(counts['broad'])}/32. "
              f"The broad-minus-narrow mean difference was {estimate_text(paired, percent=True)}.")
    lo, hi = paired["ci95"]
    if lo > 0:
        interpretation = "The broader interface improved qualification on this panel; the result does not establish algorithm-wide or real-site superiority."
    elif hi < 0:
        interpretation = "The broader interface reduced qualification on this panel despite its greater authority."
    else:
        interpretation = "This panel does not establish a qualification advantage for the broader interface."
    incomplete = sum(32 - groups[c]["complete_weeks"] for c in LABELS)
    economics = ("Table contribution averages only completed weeks; incomplete execution is retained as a failed mission and never valued as a complete week. "
                 + (f"There were {incomplete} incomplete primary-controller runs. " if incomplete else "All primary-controller weeks completed. ")
                 + "A better economic or wear outcome cannot repair a failed delivery, return or safety condition.")
    mpc = groups[MPC]
    supplementary = f"Strict MPC qualified {mpc['full_missions']}/8 prespecified cases and completed {mpc['complete_weeks']}/8 under a 300-second wall-clock ceiling per simulated week. All failures remain in its denominator. This supplementary comparison does not establish physical infeasibility."
    mpc_rows = [row for row in analysis["rows"] if row["split"] == "held_out" and row["controller"] == MPC]
    messages = [(row.get("failure") or {}).get("message", "") for row in mpc_rows]
    time_limits = sum("CaseTimeLimitExceeded" in message for message in messages)
    infeasible = sum("infeasible" in message.lower() and "CaseTimeLimitExceeded" not in message for message in messages)
    subset = {row["seed"] for row in mpc_rows}
    coverage_subset = sum(row["mission_passed"] for row in analysis["rows"]
                          if row["split"] == "held_out" and row["controller"] == COVERAGE and row["seed"] in subset)
    supplementary += (f" {time_limits} cases reached the time ceiling; {infeasible} stopped on planner infeasibility errors. "
                      f"Coverage qualified {coverage_subset}/8 on the same subset. "
                      "The whole-week cap measures research throughput, not the 15-minute control deadline.")
    shared = next(row for row in data["mechanism"]["cases"] if row["seed"] == 9960013)
    before = -shared["cycle"]["coverage"]["closing_margins"]["co2"]
    after = -shared["cycle"]["broad-101"]["closing_margins"]["co2"]
    mechanism = ("Post hoc traces for broad seed 101 show two repaired methane returns through changed production and stored gas use, "
                 f"but its shared CO2 deficit worsened from {before:.1f} to {after:.1f} kg. These cases expose coupled operating tradeoffs, not uniform improvement or an isolated causal mechanism.")
    return {"result": result, "interpretation": interpretation, "economics": economics, "mpc": supplementary, "mechanism": mechanism}


INTRO = ("Renewable methane production couples solar availability to battery dispatch, electrolysis, direct air capture (DAC), gas storage and methanation. "
         "The control problem is to deliver useful output while replenishing the resources that made it possible. We test whether more authority helps a learned correction to an existing coverage controller.")
METHOD = ("The synthetic off-grid plant has 5 MW peak PV, an 8 MWh battery, a 3 MW electrolyser, 1.2 MW DAC and a 180 kW methanator. "
          "The battery can support only the electrolyser; DAC and methanation depend on contemporaneous solar power. "
          "A week has 10,080 one-minute physical steps and 672 fifteen-minute commands. Public observations contain current plant state and issued causal weather forecasts; stochastic DAC outages remain active. "
          "The mission is 150 kg methane every six hours (4,200 kg/week), plus daily restoration of opening battery, H2, CO2 and CH4 stocks. "
          "The shared dispatcher enforces power, stock and battery-reserve constraints; equipment retains startup, minimum-load and commitment dynamics.")
LEARNING = ("Six fresh PI-Lagrangian PPO policies use 139 inputs, four outputs and two 64-unit tanh layers. Each receives 262,144 transitions over four environments; optimizer seeds 101, 211 and 307 are paired across arms. "
            "Narrow control preserves the original +/-10% multiplicative mapping. Broad control requests +/-30% of installed power ratings before explicit startup/commitment projection and can activate a prior-zero target. "
            "Both deterministic initial policies exactly reproduce coverage; Gaussian action-space standard deviation starts at 0.2. The prior, observations, reward, scenarios and learner settings match, while effective physical exploration differs.")
EVALUATION = ("All six fixed final checkpoints were frozen before 32 fresh test scenarios were opened; eight separate development scenarios are also retained. Primary panel membership and final-checkpoint selection were fixed before outcomes. "
              "Mission qualification requires full execution, all 28 deliveries, all seven four-stock daily returns and exact-minute safety. All failures remain in denominators. "
              "Uncertainty resamples paired scenarios. Arm comparisons average three optimizer-seed differences within each scenario; per-policy intervals use that policy's paired cases. Intervals condition on these fixed endpoints. There are 32 environmental replications, not 192.")
DIAGNOSTIC_TEXT = ("Two previously exposed coverage failures were missing 54.223 and 37.454 kg methane at the first daily return despite surplus other stocks. "
                   "A bounded diagnostic tested 12 coverage settings and 12 timed schedules on each case (48 first-day runs). The first timed schedule tested, a 10:00 start with a concentrated 3 MW stack request, passed both cases; "
                   "continuing with unchanged coverage from day 2 passed both full weeks, all 28 deliveries and all seven returns, with zero forced stops. "
                   "First-day stack energy fell by 1,258 and 767 kWh while methane production reached 600 kg in each case. This establishes feasibility of these two cases, not generalization or an RL gain.")
LIMITATIONS = ("The diagnostic schedule was selected on exposed cases and uses the full actuator envelope; it may not be representable by the +/-30% residual interface. One diagnostic case touches the unchanged 15% operator reserve. "
               "The simulator and wear model are synthetic and not vendor-calibrated. Dedicated PV has no purchase/export settlement; solar capital recovery and fixed O&M are unestimated and excluded. "
               "Product service value is EUR 4/kg and missed-delivery penalty EUR 20/kg. Operating contribution includes benchmark inventory accounting and wear reserves; it is not full business profit. Constraint penalties and projected actions do not guarantee feasibility.")
FUTURE = ("Next work should test forecast-aware production timing, explicit survival of daily-return obligations, and robustness to storage and conversion-model uncertainty. "
          "A from-scratch controller, larger algorithm search and calibrated site economics require a separate study. The present comparison measures complete hybrid interfaces within a twelve-hour research budget.")


def markdown(data: dict | None = None) -> str:
    lines = ["# Learning to operate an intermittent methane plant", "", "SOTA Commission II | 17 September 2026", ""]
    if data is None:
        lines += ["**DRAFT: training and fresh-test results are pending. This is not a completed-results report. The PDF builder refuses missing panels or endpoints.**", ""]
    lines += [INTRO, "", "## Experiment", "", METHOD, "", LEARNING, "", EVALUATION, "", "## Fresh-test results", ""]
    if data is None:
        lines += ["Pending: all six fixed final policies plus coverage on all 32 test scenarios; the separate eight-case MPC panel must also be present.", "",
                  "| Controller | Complete /32 | Mission /32 | Delivery /32 | Returns /32 | Safety /32 | Mean contribution, EUR/completed week |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        lines += [f"| {label(c)} | Pending | Pending | Pending | Pending | Pending | Pending |" for c in LABELS]
        lines += ["", "Pending: paired broad-minus-narrow qualification difference, its scenario-bootstrap interval, per-seed paired wear and contribution, and retained failure explanations."]
    else:
        text = report_text(data)
        lines += [text["result"], "", text["interpretation"], "",
                  "| Controller | Complete /32 | Mission /32 | Delivery /32 | Returns /32 | Safety /32 | Mean contribution, EUR/completed week |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for c in LABELS:
            g = data["groups"][c]
            values = [round(g["metrics"][m]["all_case"]["mean"] * 32) for m in PASSES]
            value = g["metrics"]["net_operating_value_eur"]["complete_observed"]["mean"]
            money = "Unavailable" if value is None else f"{value:,.2f}"
            lines.append(f"| {label(c)} | {' | '.join(map(str, values))} | {money} |")
        lines += ["", text["economics"], "", text["mpc"], "", text["mechanism"], "",
                  "[Post hoc case selection, balances and exact traces](diagnosis/held-out-mechanism-comparison.md)."]
    lines += ["", "## Constructive diagnosis of known failures", "", DIAGNOSTIC_TEXT, "",
              "[Exact trajectories, attempt counts and unchanged-gate checks](diagnosis/README.md). These two cases are excluded from fresh-test evidence.", "",
              "## Interpretation and next work", "", LIMITATIONS, "", FUTURE, "",
              "## Evidence", "", "- [Frozen protocol](../../AUTHORITY_PPO_V2_PROTOCOL.md).",
              "- [Versioned benchmark](../../benchmarks/authority_ppo_v2.json).",
              "- [Diagnostic source data](diagnosis/day1-plot-data.csv).",
              "- [Commission brief](https://sotaletters.substack.com/p/sota-commission-ii-intermittent-abundance).",
              "- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347), [residual reinforcement learning](https://arxiv.org/abs/1812.03201), [PID Lagrangian methods](https://proceedings.mlr.press/v119/stooke20a.html). The learner here uses a PI variant; no guarantee from these papers is claimed."]
    if data:
        lines += [f"- Analysis SHA-256: `{data['analysis_pin']['sha256']}`.", f"- Endpoint-freeze SHA-256: `{data['endpoint_pin']['sha256']}`."]
    return "\n".join(lines) + "\n"


def figures(data: dict, diagnosis: Path, directory: Path) -> tuple[Path, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.3, "axes.spines.top": False, "axes.spines.right": False})
    directory.mkdir(parents=True, exist_ok=True)
    comparisons = {r["comparison"].replace(" vs coverage", ""): r for r in data["analysis"]["paired_comparisons"] if r["split"] == "held_out"}
    figure, axes = plt.subplots(1, 2, figsize=(7.26, 2.17), sharey=True, layout="constrained")
    treatments = LABELS[1:]
    for index, controller in enumerate(treatments):
        color = "#217d94" if "narrow" in controller else "#a06432"
        for axis, metric, title in zip(axes, ("electrolyser_equivalent_operating_h", "battery_equivalent_full_cycles"), ("Stack equivalent operating hours", "Battery equivalent full cycles")):
            estimate = comparisons[controller]["metrics"][metric]["all_case"]
            if estimate["mean"] is None:
                axis.text(.98, index, "unavailable", ha="right", va="center", transform=axis.get_yaxis_transform(), color=color)
            else:
                mean = estimate["mean"]
                low, high = estimate["ci95"]
                axis.errorbar(mean, index, xerr=[[max(0, mean - low)], [max(0, high - mean)]], fmt="o", ms=4, capsize=2.5, color=color)
            axis.set_title(title, fontsize=9)
    for axis in axes:
        axis.axvline(0, color="#7c868a", lw=.7)
        axis.grid(axis="x", alpha=.15)
        axis.set_xlabel("Paired change per week vs coverage")
    axes[0].set_yticks(range(6), [label(c) for c in treatments])
    axes[0].invert_yaxis()
    first = directory / "writeup-paired-wear.png"
    figure.savefig(first, dpi=220); plt.close(figure)
    with (diagnosis / "day1-plot-data.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    figure, axes = plt.subplots(1, 2, figsize=(7.26, 1.95), sharey=True, layout="constrained")
    for column, seed in enumerate((9930001, 9930011)):
        for controller, color in (("Coverage", "#858d92"), ("Diagnostic batch", "#15806c")):
            selected = [r for r in rows if int(r["seed"]) == seed and r["controller"] == controller]
            x = [float(r["elapsed_h"]) + 6 for r in selected]
            y = [float(r["ch4_margin_kg"]) for r in selected]
            axes[column].plot(x, y, color=color, lw=1.1, label=controller)
            axes[column].scatter(x[-1], y[-1], color=color, s=12, zorder=4)
        axis = axes[column]
        axis.set_title(f"Known case {seed}", fontsize=9)
        axis.set_xlim(6, 30.4); axis.grid(alpha=.15); axis.axhline(0, color="#7c868a", lw=.6, ls="--")
        axis.set_xticks([6, 12, 18, 24, 30], ["06:00", "12:00", "18:00", "00:00", "06:00"])
        axis.set_xlabel("First-day clock time")
    axes[0].set_ylabel("CH4 above opening (kg)")
    axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    second = directory / "writeup-diagnostic.png"
    figure.savefig(second, dpi=220); plt.close(figure)
    return first, second


def build_pdf(data: dict, output: Path, diagnosis: Path, *, overwrite=False) -> dict:
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.lib.utils import ImageReader
    from pypdf import PdfReader
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite a report without --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    first, second = figures(data, diagnosis, output.parent / "writeup-figures")
    temporary = output.with_suffix(".part.pdf")
    width, height = A4
    margin, usable = 36., width - 72.
    styles = {
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, leading=12.2, textColor=HexColor("#253942")),
        "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8.1, leading=10.2, textColor=HexColor("#42555d")),
        "heading": ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=HexColor("#17675d")),
    }
    canvas = Canvas(str(temporary), pagesize=A4, pageCompression=1)
    canvas.setTitle("Learning to operate an intermittent methane plant")
    canvas.setSubject("SOTA Commission II: six trained hybrid PPO policies and paired exact-simulator evaluation")

    def paragraph(text, y, style="body", *, x=margin, available=usable):
        item = Paragraph(text, styles[style])
        _, h = item.wrap(available, height)
        if y - h < 42:
            raise ValueError(f"Two-page layout overflow while placing: {text[:80]}")
        item.drawOn(canvas, x, y - h)
        return y - h - 6

    def image(path, y, target_width=usable):
        reader = ImageReader(str(path)); iw, ih = reader.getSize()
        h = target_width * ih / iw
        if y - h < 42:
            raise ValueError("Two-page layout overflow while placing figure")
        canvas.drawImage(reader, margin, y - h, width=target_width, height=h, preserveAspectRatio=True, mask="auto")
        return y - h - 4

    def footer(page):
        canvas.setStrokeColor(HexColor("#d7dfdf")); canvas.line(margin, 32, width - margin, 32)
        canvas.setFont("Helvetica", 7.5); canvas.setFillColor(HexColor("#596a70"))
        canvas.drawString(margin, 21, "Synthetic benchmark | Fixed final checkpoints | Paired environmental scenarios")
        canvas.drawRightString(width - margin, 21, f"{page} / 2")

    text = report_text(data)
    try:
        y = height - 36
        canvas.setFillColor(HexColor("#61757a")); canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(margin, y, "SOTA COMMISSION II  |  17 SEPTEMBER 2026")
        y -= 26
        canvas.setFillColor(HexColor("#153b40")); canvas.setFont("Helvetica-Bold", 20)
        canvas.drawString(margin, y, "Learning to operate an intermittent plant")
        y -= 17
        y = paragraph(INTRO, y)
        y = paragraph("Matched hybrid-controller experiment", y, "heading")
        col = (usable - 16) / 2
        left = paragraph(METHOD, y, x=margin, available=col)
        right = paragraph(LEARNING, y, x=margin + col + 16, available=col)
        y = min(left, right)
        y = paragraph("Fresh-test results: every outcome retained", y, "heading")
        headers = ["Controller", "Complete\n/32", "Mission\n/32", "Delivery\n/32", "Returns\n/32", "Safety\n/32", "Contribution*\nEUR/week"]
        table_rows = [headers]
        for controller in LABELS:
            group = data["groups"][controller]
            counts = [str(round(group["metrics"][m]["all_case"]["mean"] * 32)) for m in PASSES]
            money = group["metrics"]["net_operating_value_eur"]["complete_observed"]["mean"]
            table_rows.append([label(controller), *counts, "N/A" if money is None else f"{money:,.2f}"])
        widths = [96, 52, 52, 52, 52, 49, usable - 353]
        table = Table(table_rows, colWidths=widths, rowHeights=[29] + [18] * 7)
        table.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                                  ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("LEADING", (0, 0), (-1, 0), 10),
                                  ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e5eeeb")), ("TEXTCOLOR", (0, 0), (-1, -1), HexColor("#253942")),
                                  ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                                  ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f6f8f7")]),
                                  ("LINEBELOW", (0, 0), (-1, 0), .4, HexColor("#b6c9c4")), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        _, th = table.wrap(usable, height)
        table.drawOn(canvas, margin, y - th); y -= th + 6
        y = paragraph("*Mean over completed weeks only; the Complete column gives its denominator. Delivery and Returns count whole weeks meeting every obligation. Safety requires complete exact-minute checks. Contribution excludes unestimated solar capital and fixed O&amp;M.", y, "small")
        y = paragraph(escape(text["result"] + " " + text["interpretation"]), y)
        y = image(first, y)
        y = paragraph("Figure 1. Modeled wear changes against coverage; negative is less wear. Bars are 95% paired-scenario bootstrap intervals. An all-case comparison is unavailable if any required week is incomplete. These are synthetic wear proxies, not calibrated lifetime estimates.", y, "small")
        footer(1); canvas.showPage()
        y = height - 38
        y = paragraph("What the experiment can establish", y, "heading")
        y = paragraph(EVALUATION, y)
        y = paragraph(escape(text["economics"] + " " + text["mpc"]), y)
        y = paragraph(escape(text["mechanism"]), y)
        y = paragraph("Known failures have a constructive feasible schedule", y, "heading")
        y = paragraph(DIAGNOSTIC_TEXT, y)
        y = image(second, y)
        y = paragraph("Figure 2. Exposed diagnostic cases, excluded from fresh-test evidence. Gray: coverage; green: selected day-1 batch. The daily methane return follows the final delivery. Both diagnostic full-week replays pass the unchanged mission.", y, "small")
        y = paragraph("Limits and next work", y, "heading")
        y = paragraph(LIMITATIONS + " " + FUTURE, y)
        y = paragraph("Evidence: <link href='https://github.com/garavels/PlantAnAgent' color='#17675d'>PlantAnAgent repository</link>. The accompanying portable research package contains the executed source, frozen authority_ppo_v2 benchmark/protocol, six checkpoints, selected exact-minute recordings and complete paired results. "
                      + f"Analysis digest: {data['analysis_pin']['sha256'][:16]}...; endpoint-freeze digest: {data['endpoint_pin']['sha256'][:16]}...", y, "small")
        y = paragraph("Method references: <link href='https://arxiv.org/abs/1707.06347' color='#17675d'>PPO</link>; <link href='https://arxiv.org/abs/1812.03201' color='#17675d'>residual reinforcement learning</link>; <link href='https://proceedings.mlr.press/v119/stooke20a.html' color='#17675d'>PID Lagrangian methods</link>. The learner uses a PI variant. Contribution: a reproducible coupled plant testbed and empirical action-interface comparison.", y, "small")
        footer(2); canvas.save()
        reader = PdfReader(temporary)
        if len(reader.pages) != 2:
            raise ValueError("Commission report must have exactly two pages")
        extracted = "\n".join(p.extract_text() or "" for p in reader.pages)
        if "Pending" in extracted or "DRAFT:" in extracted or any(label(c) not in extracted for c in LABELS):
            raise ValueError("PDF omitted required controllers or contains draft placeholders")
        temporary.replace(output)
        return {"schema": "authority-ppo-v2-writeup", "created_at": datetime.now(timezone.utc).isoformat(), "pages": 2,
                "pdf": pin(output), "analysis": data["analysis_pin"], "endpoints": data["endpoint_pin"],
                "diagnosis": data["diagnosis_pin"], "mechanism": data["mechanism_pin"], "evaluation_freeze": data["evaluation_freeze_pin"],
                "builder": pin(Path(__file__)), "figures": [pin(first), pin(second)],
                "visual_qa": "Required: render both PDF pages and inspect before delivery; structural checks do not replace visual QA."}
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--evaluation-freeze", type=Path)
    parser.add_argument("--diagnosis", type=Path, default=DIAGNOSIS)
    parser.add_argument("--output", type=Path, default=RESULTS / "commission-writeup.pdf")
    parser.add_argument("--markdown", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--draft-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.draft_only:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown())
        print(json.dumps({"draft": str(args.markdown), "pdf_created": False, "status": "Results pending; final PDF prohibited until every required panel and endpoint is present."}))
        return
    if args.analysis is None or args.training_root is None or args.evaluation_freeze is None:
        parser.error("A final report requires --analysis, --training-root and --evaluation-freeze; use --draft-only before results exist")
    data = validate_inputs(args.analysis.resolve(), args.training_root.resolve(), args.diagnosis.resolve(), args.evaluation_freeze.resolve())
    report_text(data)
    if args.validate_only:
        print(json.dumps({"valid": True, "pdf_created": False, "primary_outcomes": 224, "development_outcomes": 56, "supplementary_mpc_outcomes": 8}))
        return
    result = build_pdf(data, args.output.resolve(), args.diagnosis.resolve(), overwrite=args.overwrite)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown(data))
    result["markdown"] = pin(args.markdown)
    args.output.with_suffix(".build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Unfrozen reporting helper: completed evidence -> final commission paper.

Does not evaluate controllers or change any scientific source. A layout-only
mode creates explicitly pending macros and never copies a final PDF.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[4]
RESULT = ROOT / "docs/research/results/tail_stress_20260918"
ORDINARY = ROOT / "docs/research/results/wholeplant_ppo_20260917/analysis/analysis.json"
STARTUP = ROOT / "docs/research/results/startup_stress_20260917/summary.json"
CONTROLLERS = ("daylight", "storage", "ppo-101", "ppo-211", "ppo-307")
LABELS = {"daylight": "Daylight", "storage": "Storage", "ppo-101": "PPO 101", "ppo-211": "PPO 211", "ppo-307": "PPO 307"}


def identity(path):
    payload = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def load(path):
    return json.loads(path.read_text())


def money(value):
    return f"{value:,.0f}"


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def validate(tail, ordinary):
    if tail["schema"] != "wholeplant-tail-stress-analysis-v1":
        raise ValueError("Unexpected tail analysis schema")
    if tail["design"]["n_rollouts"] != 480 or tail["design"]["n_base_scenarios"] != 16:
        raise ValueError("Tail analysis has wrong prescribed dimensions")
    if not tail["audits"]["all_rollouts_complete_and_audited"] or tail["audits"]["complete_rollouts"] != 480 or tail["audits"]["audited_rollouts"] != 480:
        raise ValueError("Final paper requires all 480 cases complete and audited")
    if tail["audits"]["guard_interventions"] != 0:
        raise ValueError("Guard intervention contradicts draft method statement")
    primary = tail["primary_claims"]
    if len(primary) != 6 or any(not p["all_comparisons_complete_and_audited"] for p in primary):
        raise ValueError("Incomplete primary paired comparisons")
    for p in primary:
        est = p["primary_criterion"]["adjusted_success_difference"]
        if abs(est["confidence"] - (1 - .05 / 6)) > 1e-12 or est["n_base_scenarios"] != 16 or est["n_paired_cells"] != 80:
            raise ValueError("Primary inference differs from declared design")
    groups = {g["controller"]: g for g in ordinary["groups"] if g["split"] == "held_out" and g["condition"] == "shared-dynamic" and g["controller"] in CONTROLLERS}
    if set(groups) != set(CONTROLLERS):
        raise ValueError("Ordinary comparison lacks a declared controller")
    for g in groups.values():
        if any(g[k] != 32 for k in ("n_scenarios", "complete_weeks", "physical_invariant_successes", "economic_audit_successes")):
            raise ValueError("Incomplete or unaudited ordinary comparison")
    if len(ordinary["rows"]) != 496 or any(not (r["execution_complete"] and r["physical_invariants_passed"] and r["economic_audit"]["passed"]) for r in ordinary["rows"]):
        raise ValueError("Ordinary study is not the complete audited 496-case record")
    return groups


def make_figure(primary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "pdf.fonttype": 42})
    order = sorted(primary, key=lambda p: (p["reference"] != "daylight", p["treatment"]))
    fig, ax = plt.subplots(figsize=(3.35, 1.95))
    labels = []
    for i, p in enumerate(order):
        est = p["primary_criterion"]["adjusted_success_difference"]
        mean, lo, hi = [100 * v for v in (est["mean"], *est["interval"])]
        color = "#226B73" if p["reference"] == "daylight" else "#B77236"
        ax.plot([lo, hi], [i, i], color=color, linewidth=1.8)
        ax.scatter([mean], [i], c=color, s=22, zorder=3)
        labels.append(f"{p['treatment'][-3:]} - {LABELS[p['reference']]}")
    ax.set_yticks(range(6), labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#77838A", linewidth=.8, linestyle="--")
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Success difference\n(percentage points)")
    ax.set_ylim(5.65, -.65)
    fig.tight_layout(pad=.45)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=200)
    plt.close(fig)


def generate(args):
    out = args.output.resolve()
    if args.layout:
        write(out / "results.tex", r"\newcommand{\AbstractResults}{\textbf{RESULTS PENDING: layout draft only.}}" + "\n" +
              r"\newcommand{\DeliveryRows}{Daylight & pending & pending & pending & pending\\Storage & pending & pending & pending & pending\\PPO 101 & pending & pending & pending & pending\\PPO 211 & pending & pending & pending & pending\\PPO 307 & pending & pending & pending & pending\\}" + "\n" +
              r"\newcommand{\TailFigureCaption}{Pending final data; this is a layout-only draft.}" + "\n")
        write(out / "findings.tex", "\\textbf{Pending complete, audited tail analysis.} The final results will distinguish ordinary delivery success, the constructed startup diagnostic, and all prespecified stress conditions. No incomplete result is interpreted here.\n\n" +
              "The final text will report each optimizer seed, all failures, paired primary intervals, and the prespecified reliability criterion. The available ordinary and startup results remain preserved in their original packages.\n")
        write(out / "interpretation.tex", "\\textbf{Pending evidence review.} The final conclusion will state the actual reach and limits of learned autonomy, including whether any policy meets the prespecified comparison criterion. It will not infer a real-world interruption frequency or justify an insurance premium from a synthetic challenge suite.\n")
        return
    tail = load(args.analysis)
    ordinary = load(args.ordinary)
    startup = load(args.startup)
    ordinary_groups = validate(tail, ordinary)
    if startup["case_count"] != 12 or not startup["all_executions_and_audits_passed"]:
        raise ValueError("Startup diagnostic is incomplete or unaudited")
    startup_stress = {r["controller"]: r["first_six_hours"] for r in startup["rows"] if r["condition"] == "solar-loss"}
    startup_211_unmet = startup_stress["ppo-211"]["delivery"]["unmet_kg"]
    startup_307_unmet = startup_stress["ppo-307"]["delivery"]["unmet_kg"]
    startup_101_energy = startup_stress["ppo-101"]["operating_metrics"]["battery_discharge_kwh"]
    startup_storage_energy = startup_stress["storage"]["operating_metrics"]["battery_discharge_kwh"]
    stress_groups = {g["controller"]: g for g in tail["pooled_stress_groups"]}
    control_groups = {g["controller"]: g for g in tail["groups"] if g["panel"] == "control"}
    if set(stress_groups) != set(CONTROLLERS) or any(g["n_cells"] != 80 for g in stress_groups.values()):
        raise ValueError("Missing pooled stress controller")
    rows = []
    for controller in CONTROLLERS:
        o, s = ordinary_groups[controller], stress_groups[controller]
        rows.append(f"{LABELS[controller]} & {o['delivery_successes']}/32 & {money(o['metrics']['net_operating_value_eur']['all_case']['mean'])} & {s['all_delivery_successes']}/80 & {money(s['metrics']['net_operating_value_eur']['mean'])}" + r"\\")
    met = [p for p in tail["primary_claims"] if p["primary_criterion"]["met"]]
    criterion_sentence = (f"{len(met)} of six prespecified PPO--baseline reliability comparisons pass the joint criterion." if met
                          else "No PPO policy meets the prespecified broad-reliability criterion against either baseline.")
    control_sentence = (f"On the new 16-case control without injected faults, Daylight and Storage succeed in {control_groups['daylight']['all_delivery_successes']} and {control_groups['storage']['all_delivery_successes']} weeks; "
                        f"PPO succeeds in {control_groups['ppo-101']['all_delivery_successes']}, {control_groups['ppo-211']['all_delivery_successes']} and {control_groups['ppo-307']['all_delivery_successes']}. ")
    p307_day = next(p for p in tail["primary_claims"] if p["reference"] == "daylight" and p["treatment"] == "ppo-307")
    p307_store = next(p for p in tail["primary_claims"] if p["reference"] == "storage" and p["treatment"] == "ppo-307")
    p307_unmet = p307_day["metrics"]["unmet_kg"]["mean"]
    nuance = (f"PPO 307 rescues {p307_day['rescues']} weeks but regresses in {p307_day['regressions']} versus Daylight, "
              f"while lowering mean unmet mass by {-p307_unmet:.1f}\\,kg. Against Storage it rescues {p307_store['rescues']} and regresses in {p307_store['regressions']}. ")
    abstract = ("After training, the policies complete every delivery in 16--31 of 32 ordinary held-out weeks, compared with 32 for both rules. "
                + ("The later 480-case stress test identifies a qualified reliability advantage in the specified challenge suite." if met
                   else "A later 480-case stress test does not establish the prespecified PPO reliability advantage."))
    macros = (r"\newcommand{\AbstractResults}{" + abstract + "}\n" +
              r"\newcommand{\DeliveryRows}{" + "\n".join(rows) + "}\n" +
              r"\newcommand{\TailFigureCaption}{" + "Pooled stress success differences for each fixed PPO versus each rule. Bars are 99.1667\\% intervals resampling the 16 base scenarios with all five conditions retained; zero denotes no advantage. The primary criterion also requires rescues, no regressions and no larger unmet mass." + "}\n")
    write(out / "results.tex", macros)
    aggregate = next(p for p in ordinary["aggregate_comparisons"] if p["split"] == "held_out" and p["reference"] == "daylight")
    delta = aggregate["metrics"]["net_operating_value_eur"]["all_case"]
    findings = ("The ordinary study completes 496 development and held-out cases with physical and economic audits. Daylight meets every delivery in all four physical arms. On eight development scenarios, initial policies achieve 0/8 complete-delivery weeks, improving to 8/8, 4/8 and 8/8 after training. On ordinary held-out scenarios, PPO achieves 27/32, 16/32 and 31/32, respectively (Table~\\ref{tab:deliveries}). "
                f"Its three-policy mean contribution trails daylight by EUR\\,{money(-delta['mean'])}/week (paired 95\\% interval for PPO minus daylight [{money(delta['ci95'][0])}, {money(delta['ci95'][1])}]).\n\n"
                f"In the constructed startup case, a six-hour solar loss precedes the first delivery. Opening methane covers only 120 of 180\\,kg owed. Storage and PPO 101 recover that delivery; PPO 211 and 307 miss {startup_211_unmet:.0f} and {startup_307_unmet:.0f}\\,kg. PPO 101 uses {startup_101_energy:,.0f}\\,kWh of battery output versus Storage's {startup_storage_energy:,.0f}\\,kWh. One successful recovery demonstrates a capability, not general reliability.\n\n"
                f"All {tail['audits']['complete_rollouts']} tail rollouts complete and pass physical/economic audits. {criterion_sentence} "
                + nuance + control_sentence + "Figure~\\ref{fig:tails} reports all six adjusted comparisons.\n")
    write(out / "findings.tex", findings)
    if met:
        interpretation = ("The tail result is conditional on the frozen policies, initial-stock distribution and five designed disruptions. It demonstrates the stated criterion only for the named policy--baseline contrasts; it is not an algorithm-wide or deployed reliability guarantee. The ordinary-operation deficit remains part of the same tradeoff. Without real event frequencies, the experiment cannot establish whether that deficit is an economically worthwhile insurance premium.\n")
    else:
        interpretation = ("PPO learns useful coordinated actions and can recover a particular battery-dependent delivery, but does not meet the declared broad-reliability criterion. These frozen policies were not retrained for the imposed disruptions; this result does not establish that PPO cannot learn better tail control. Unknown event frequencies prevent an expected insurance valuation.\n")
    storage_stress = next(p for p in tail["comparisons"] if p["panel"] == "pooled-stress" and p["reference"] == "daylight" and p["treatment"] == "storage")
    storage_control = next(p for p in tail["comparisons"] if p["panel"] == "control" and p["reference"] == "daylight" and p["treatment"] == "storage")
    buffering = (f"Storage shows a useful buffering tradeoff: it rescues {storage_stress['rescues']} Daylight failures with no regressions, "
                 f"sacrificing EUR\\,{money(-storage_control['metrics']['net_operating_value_eur']['mean'])}/week in matched controls and gaining EUR\\,{money(storage_stress['metrics']['net_operating_value_eur']['mean'])}/week in the designed stress mix. This is a heuristic result, not a PPO gain.\n\n")
    interpretation = buffering + interpretation
    write(out / "interpretation.tex", interpretation)
    make_figure(tail["primary_claims"], out / "figures/tail-contribution.pdf")
    data = {"schema": "plantanagent-final-commission-report-v1", "generated_at": datetime.now(timezone.utc).isoformat(),
            "inputs": [identity(args.analysis), identity(args.ordinary), identity(args.startup)],
            "controllers": list(CONTROLLERS), "ordinary_groups": ordinary_groups, "stress_groups": stress_groups,
            "primary_claims": tail["primary_claims"], "audits": tail["audits"], "any_primary_claim_met": bool(met),
            "scope": "Three frozen policies; no selection, no new training, no inferred population disruption frequency."}
    write(out / "report-data.json", json.dumps(data, indent=2) + "\n")


def compile_paper(args):
    out = args.output.resolve()
    build = out / (".layout" if args.layout else ".build")
    build.mkdir(parents=True, exist_ok=True)
    name = "plantanagent-commission-final"
    for _ in range(2):
        proc = subprocess.run(["/Library/TeX/texbin/pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={build}", name + ".tex"], cwd=out, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(proc.stdout[-10000:] + proc.stderr)
    log = (build / (name + ".log")).read_text(errors="replace")
    pages = int(re.search(r"Output written on .*?\((\d+) pages?", log, re.S).group(1))
    issues = [line for line in log.splitlines() if "Overfull" in line or "undefined" in line]
    print(json.dumps({"pages": pages, "issues": issues, "pdf": str(build / (name + ".pdf")), "layout_only": args.layout}))
    if args.layout:
        return
    if pages != 2 or issues:
        raise ValueError("Final paper must have exactly two pages and no overflow/unresolved references")
    for dest in (out / (name + ".pdf"), ROOT / "output/pdf" / (name + ".pdf")):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(build / (name + ".pdf"), dest)
    receipt = {"generated_at": datetime.now(timezone.utc).isoformat(), "pages": pages,
               "paper": identity(out / (name + ".pdf")), "source": identity(out / (name + ".tex")),
               "report_data": identity(out / "report-data.json"), "reporting_source": identity(Path(__file__).resolve()),
               "visual_review_required": True}
    write(out / "build-receipt.json", json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, default=RESULT / "analysis/analysis.json")
    parser.add_argument("--ordinary", type=Path, default=ORDINARY)
    parser.add_argument("--startup", type=Path, default=STARTUP)
    parser.add_argument("--output", type=Path, default=RESULT / "writeup")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--layout", action="store_true", help="Pending-results layout only; never delivers final PDF")
    args = parser.parse_args()
    args.analysis = args.analysis.resolve()
    args.ordinary = args.ordinary.resolve()
    args.startup = args.startup.resolve()
    generate(args)
    if args.compile:
        compile_paper(args)


if __name__ == "__main__":
    main()

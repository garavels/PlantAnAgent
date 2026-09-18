"""Collect the fixed-panel feasibility and frozen-policy transfer results."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "docs/research/results/feasible_mission_20260915"
SEEDS = set(range(9700300, 9700308)) | {9000034}
CONTROLLERS = {
    "coverage": "daily-coverage-feasibility-v2",
    "mpc": "operational-economic-mpc-v1",
    "ppo": "frozen-scalar-ppo-gamma1000-mission-transfer-v1",
}


def collect(base=BASE, *, require_complete=True):
    coverage = json.loads((base / "verified/coverage-25-summary.json").read_text())
    groups = {
        "coverage": {x["seed"]: x for x in coverage["runs"]},
        "mpc": {},
        "ppo": {},
    }
    sources = [base / "verified/coverage-25-summary.json"]
    for path in sorted(base.rglob("*-25-summary.json")):
        payload = json.loads(path.read_text())
        kind = payload.get("controller_kind")
        if kind not in {"mpc", "ppo"}:
            continue
        sources.append(path)
        for run in payload["runs"]:
            seed = run["seed"]
            old = groups[kind].get(seed)
            if (
                old is not None
                and old["recording"]["sha256"] != run["recording"]["sha256"]
            ):
                raise ValueError(f"Conflicting comparator records for {kind} {seed}")
            groups[kind][seed] = run
    manifests = {}
    for kind, runs in groups.items():
        if require_complete and set(runs) != SEEDS:
            raise ValueError(f"Incomplete {kind} panel: {sorted(runs)}")
        for seed, run in runs.items():
            if (
                run["delivery_rate_kg_h"] != 25
                or run["source"] != groups["coverage"][seed]["source"]
            ):
                raise ValueError("Comparison changed source scenario or delivery rate")
            if not run["execution_complete"]:
                raise ValueError(f"Comparator did not complete its week: {kind} {seed}")
            path = Path(run["recording"]["path"])
            if not path.is_absolute():
                path = ROOT / path
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != run["recording"]["sha256"]:
                raise ValueError(f"Recording hash changed: {path}")
            recording = json.loads(gzip.decompress(raw))
            if recording["controller"] != CONTROLLERS[kind]:
                raise ValueError(f"Unexpected controller identity: {kind} {seed}")
            omitted = {"commands", "decision_frames", "minute_truth_states", "manifest"}
            for key, value in recording.items():
                if key not in omitted and (key not in run or run[key] != value):
                    raise ValueError(
                        f"Summary differs from verified recording: {kind} {seed} {key}"
                    )
            manifest = recording["manifest"]
            identity = {
                key: manifest[key]
                for key in ("manifest_sha256", "beta_exogenous_episode_sha256")
            }
            if kind == "coverage":
                manifests[seed] = identity
            elif identity != manifests[seed]:
                raise ValueError(
                    f"Comparator task or exogenous identity differs: {kind} {seed}"
                )
    result = {
        "schema": "fixed-panel-feasible-mission-comparison-v1",
        "mission_id": coverage["mission_id"],
        "complete": all(set(x) == SEEDS for x in groups.values()),
        "same_scenario_manifest_verified": manifests,
        "interpretation": "Same 25kg/h mission and original exogenous cases. Coverage controller is development-tuned; PPO uses frozen weights trained at40kg/h and is an exploratory mission-transfer evaluation.",
        "sources": [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sources
        ],
        "groups": {},
    }
    for kind, runs in groups.items():
        research = [x for s, x in sorted(runs.items()) if s != 9000034]
        rows = []
        for seed, run in sorted(runs.items()):
            physical = run["physical_metrics"]
            rows.append(
                {
                    "seed": seed,
                    "mission_passed": run["mission_passed"],
                    "hard_safety_passed": run["hard_safety_passed"],
                    "deliveries_met": physical["commitments_met"],
                    "delivered_kg": physical["delivered_kg"],
                    "unmet_kg": physical["unmet_kg"],
                    "daily_returns_met": sum(
                        x["feasible"] for x in run["cycle_settlements"]
                    ),
                    "forced_stops": physical["electrolyser_forced_stops"],
                    "modeled_net_value_eur": run["economic_metrics"][
                        "net_operating_value_eur"
                    ],
                    "recording": run["recording"],
                }
            )
        result["groups"][kind] = {
            "runs": rows,
            "research": {
                "cases": len(research),
                "mission_pass_count": sum(x["mission_passed"] for x in research),
                "hard_safety_pass_count": sum(
                    x["hard_safety_passed"] for x in research
                ),
                "every_delivery_pass_count": sum(
                    x["every_delivery_passed"] for x in research
                ),
                "all_daily_returns_pass_count": sum(
                    x["all_daily_returns_passed"] for x in research
                ),
                "mean_delivered_kg": (
                    statistics.mean(
                        x["physical_metrics"]["delivered_kg"] for x in research
                    )
                    if research
                    else None
                ),
                "mean_unmet_kg": (
                    statistics.mean(x["physical_metrics"]["unmet_kg"] for x in research)
                    if research
                    else None
                ),
                "mean_modeled_net_value_eur": (
                    statistics.mean(
                        x["economic_metrics"]["net_operating_value_eur"]
                        for x in research
                    )
                    if research
                    else None
                ),
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    if args.allow_partial and args.write_report:
        parser.error("--write-report cannot be combined with --allow-partial")
    result = collect(require_complete=not args.allow_partial)
    path = BASE / (
        "comparison-partial.json" if args.allow_partial else "comparison.json"
    )
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.write_report:
        if not result["complete"]:
            raise ValueError("Only a complete comparison may be written to the report")
        write_report_section(result)
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "groups": {k: v["research"] for k, v in result["groups"].items()},
            },
            indent=2,
        )
    )


def write_report_section(result):
    labels = {
        "coverage": "Daily coverage controller",
        "mpc": "Operating MPC",
        "ppo": "Frozen scalar PPO · gamma 1.000",
    }
    lines = [
        "<!--FEASIBLE_COMPARISON_BEGIN-->",
        "## MPC and frozen PPO on the feasible mission",
        "",
        "All three controllers completed all nine exact-simulator weeks at 25 kg/h. Every comparator's complete task manifest and beta exogenous identity matches the baseline for that scenario. The PPO weights are the admitted, frozen `B-scalar-gamma1000` endpoint from batch 009, trained at 40 kg/h. Its results here are exploratory transfer performance; no training was performed.",
        "",
        "### Eight research scenarios",
        "",
        "| Controller | Full missions | Passing daily returns | Safe weeks | Mean unmet kg/week | Mean modeled value/week |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind, group in result["groups"].items():
        s = group["research"]
        returns = sum(
            x["daily_returns_met"] for x in group["runs"] if x["seed"] != 9000034
        )
        lines.append(
            f"| {labels[kind]} | {s['mission_pass_count']} / 8 | {returns} / 56 | {s['hard_safety_pass_count']} / 8 | {s['mean_unmet_kg']:.6f} | €{s['mean_modeled_net_value_eur']:,.2f} |"
        )
    lines += [
        "",
        "Both comparators fail inventory restoration and have material delivery shortfalls in some cases. Strict delivery counts also include tiny numerical shortages; the per-case mass table below distinguishes their scale. Higher modeled values from schedules that fail mandatory requirements are not evidence of a better successful policy. The daily-coverage controller is a feasible reference, not an economic optimum; some of its end-of-day surplus is uncredited by the ledger.",
        "",
        "### Public demo scenario 9000034",
        "",
        "| Controller | Full mission | Daily returns | Safety | Unmet kg | Modeled value/week |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    for kind, group in result["groups"].items():
        x = next(x for x in group["runs"] if x["seed"] == 9000034)
        lines.append(
            f"| {labels[kind]} | {'Pass' if x['mission_passed'] else 'Fail'} | {x['daily_returns_met']} / 7 | {'Pass' if x['hard_safety_passed'] else 'Fail'} | {x['unmet_kg']:.6f} | €{x['modeled_net_value_eur']:,.2f} |"
        )
    lines += [
        "",
        "### Delivery shortfalls by case",
        "",
        "The daily-coverage controller meets all 28 deliveries with zero unmet mass in every case. Comparator delivery counts use the original strict tolerance of 1e−9 kg; one milligram is 0.000001 kg. A missed delivery with a numerical residue is different in scale from the kilogram shortages shown here, but neither changes the independent inventory-return failures.",
        "",
        "| Scenario | MPC deliveries | MPC unmet kg | PPO deliveries | PPO unmet kg |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    by_seed = {
        kind: {x["seed"]: x for x in result["groups"][kind]["runs"]}
        for kind in ("mpc", "ppo")
    }
    for seed in sorted(SEEDS, key=lambda seed: (seed == 9000034, seed)):
        mpc, ppo = by_seed["mpc"][seed], by_seed["ppo"][seed]
        label = f"{'Demo' if seed == 9000034 else 'Research'} {seed}"
        lines.append(
            f"| {label} | {mpc['deliveries_met']} / 28 | {mpc['unmet_kg']:.6f} | {ppo['deliveries_met']} / 28 | {ppo['unmet_kg']:.6f} |"
        )
    lines += [
        "",
        "### Interpretation and reproducibility",
        "",
        "The demonstrated baseline removes physical infeasibility as an explanation for these particular restoration failures at 25 kg/h. Operating MPC measures daily restoration without enforcing it as a hard optimization constraint; the tested scalar PPO uses the economic reward and is also being transferred to a new mission. These results do not establish that PPO could not learn a successful policy when trained on a feasible task with the full requirements represented.",
        "",
        "The saved Operating MPC and PPO schedules for research seed 9700300 were additionally replayed without their controllers; all 10,081 minute states matched the original executions. The baseline has this full replay check for all nine scenarios.",
        "",
        "Comparator runs used unchanged solver settings, including fixed wall-clock solve limits, and ran concurrently on the local machine. System load can affect solver deadlines, fallback selection and small performance differences. Treat the scores as observed development results, not deterministic timing measurements, held-out qualification or a statistical ranking.",
        "",
        f"Detailed per-case results and recording/source hashes: [comparison.json]({BASE / 'comparison.json'}). Checkpoint admission and numerical-source provenance: [PPO execution provenance]({BASE / 'comparators/ppo-execution-provenance.json'}).",
        "",
        "The PPO reproduction additionally requires the original `PlantAnAgentResearch/adaptive-rl-batch-009/training/B-scalar-gamma1000` files. Use fresh output directories:",
        "",
        "```sh",
        ".venv/bin/python docs/research/tools/run_feasible_mission.py --controller mpc --rate 25 --output runs/feasible-mission-mpc-reproduction",
        ".venv/bin/python docs/research/tools/run_feasible_mission.py --controller ppo --rate 25 --output runs/feasible-mission-ppo-reproduction",
        "```",
        "<!--FEASIBLE_COMPARISON_END-->",
    ]
    report = ROOT / "docs/research/results/FEASIBLE_MISSION_20260915.md"
    text = report.read_text()
    section = "\n".join(lines)
    pattern = r"<!--FEASIBLE_COMPARISON_BEGIN-->[\s\S]*?<!--FEASIBLE_COMPARISON_END-->"
    if re.search(pattern, text):
        text = re.sub(pattern, lambda _: section, text)
    else:
        text += "\n" + section + "\n"
    report.write_text(text)


if __name__ == "__main__":
    main()

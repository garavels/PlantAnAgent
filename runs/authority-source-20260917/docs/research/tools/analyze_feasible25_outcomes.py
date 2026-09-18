"""Summarize paired outcomes without changing any evaluated policy or scenario."""

from __future__ import annotations

import gzip
import json
import statistics
from pathlib import Path

from docs.research.tools.feasible_mission_replay import file_pin
from docs.research.tools.summarize_feasible25_v1 import BASE, collect


def analyze():
    inspection = collect()
    rows = inspection["rows"]
    comparisons = []
    for split in ("development", "held_out"):
        reference = {
            row["seed"]: row
            for row in rows
            if row["split"] == split
            and row["controller"] == "daily-coverage-offgrid-v1"
        }
        for optimizer_seed in (101, 211, 307):
            controller = f"coverage-ppo-{optimizer_seed}"
            selected = [
                row
                for row in rows
                if row["split"] == split and row["controller"] == controller
            ]
            paired = []
            for row in selected:
                baseline = reference[row["seed"]]
                complete_pair = (
                    row["execution_complete"] and baseline["execution_complete"]
                )
                paired.append(
                    {
                        "seed": row["seed"],
                        "both_complete": complete_pair,
                        "baseline_passed": baseline["mission_passed"],
                        "ppo_passed": row["mission_passed"],
                        "delta_contribution_eur": (
                            row["modeled_value_eur"] - baseline["modeled_value_eur"]
                            if complete_pair
                            else None
                        ),
                        "delta_components_eur": (
                            {
                                key: value - baseline["components_eur"].get(key, 0)
                                for key, value in row["components_eur"].items()
                            }
                            if complete_pair
                            else None
                        ),
                    }
                )
            complete = [x for x in paired if x["both_complete"]]
            both_passed = [
                x for x in complete if x["baseline_passed"] and x["ppo_passed"]
            ]

            def mean_delta(items):
                return (
                    statistics.mean(x["delta_contribution_eur"] for x in items)
                    if items
                    else None
                )

            comparisons.append(
                {
                    "split": split,
                    "controller": controller,
                    "paired_cases": len(paired),
                    "complete_pairs": len(complete),
                    "both_passed": len(both_passed),
                    "regressed_missions": [
                        x["seed"]
                        for x in paired
                        if x["baseline_passed"] and not x["ppo_passed"]
                    ],
                    "rescued_missions": [
                        x["seed"]
                        for x in paired
                        if not x["baseline_passed"] and x["ppo_passed"]
                    ],
                    "mean_delta_eur_complete_pairs": mean_delta(complete),
                    "mean_delta_eur_both_passed": mean_delta(both_passed),
                    "paired": paired,
                }
            )
    failures = []
    inference = []
    for row in rows:
        missed_returns = [x for x in row["cycle_settlements"] if not x["feasible"]]
        if not row["mission_passed"]:
            failures.append(
                {
                    key: row[key]
                    for key in (
                        "split",
                        "seed",
                        "controller",
                        "execution_complete",
                        "unmet_kg",
                        "failure",
                    )
                }
                | {"missed_returns": missed_returns}
            )
        if row["controller"].startswith("coverage-ppo-"):
            path = row["recording"]["path"]
            with gzip.open(path, "rt") as stream:
                record = json.load(stream)
            evidence = [x["evidence"] for x in record["commands"]]
            witnesses = [x["inference_witness"] for x in evidence]
            if not all(w["adapter_input_verified"] for w in witnesses):
                raise ValueError(
                    "An evaluated PPO action lacks a valid inference witness"
                )
            changed = sum(
                x["nominal_command"]["targets"] != x["mapped_targets"] for x in evidence
            )
            inference.append(
                {
                    "split": row["split"],
                    "seed": row["seed"],
                    "controller": row["controller"],
                    "decisions": len(evidence),
                    "commands_different_from_current_coverage": changed,
                    "exact_zero_residual_decisions": sum(
                        x["exact_zero_command"] for x in evidence
                    ),
                    "upstream_clipped_decisions": sum(
                        w["upstream_action_clipped"] for w in witnesses
                    ),
                    "max_absolute_unclipped_action": max(
                        (
                            abs(v)
                            for w in witnesses
                            for v in w["unclipped_deterministic_action"]
                        ),
                        default=0,
                    ),
                    "mean_absolute_unclipped_action_by_actuator": (
                        [
                            statistics.mean(
                                abs(w["unclipped_deterministic_action"][i])
                                for w in witnesses
                            )
                            for i in range(4)
                        ]
                        if witnesses
                        else None
                    ),
                    "physical_guard_interventions": record["physical_metrics"][
                        "guard_interventions"
                    ],
                    "recording": row["recording"],
                }
            )
    return {
        "schema": "feasible25-paired-outcomes-v1",
        "benchmark_contract_sha256": inspection["benchmark_contract_sha256"],
        "evaluation_freeze": file_pin(BASE / "evaluation-freeze.json"),
        "analysis_source": file_pin(Path(__file__).resolve()),
        "groups": inspection["groups"],
        "paired_comparisons": comparisons,
        "mission_failures": failures,
        "ppo_inference": inference,
        "interpretation": "Comparisons use the same scenario within each split. Positive contribution does not repair a mission failure. Partial execution money is excluded from full-week means. Three optimizer runs share scenarios and are not independent environmental replications.",
    }


if __name__ == "__main__":
    result = analyze()
    (BASE / "paired-outcomes.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result["groups"], indent=2))

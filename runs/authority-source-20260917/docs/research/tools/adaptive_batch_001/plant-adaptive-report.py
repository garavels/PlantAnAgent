"""All-slot descriptive analysis of an independently audited, closed PPO batch.

No environment, policy or training execution occurs here. This is development
analysis, with no policy selection, materiality threshold or confirmation claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

ARMS = (
    "A-pi-raw",
    "B-pi-scaled",
    "C-scalar-raw",
    "D-scalar-scaled",
    "E-pi-scaled-lowstd",
)
COMPARATORS = ("initial", "H", "OperatingMPC", "M24", "E-beta")
SEEDS = tuple(range(9_700_000, 9_700_008))
RESOURCES = ("battery", "h2", "co2", "ch4")
PHYSICAL = (
    "delivery_fill_rate",
    "delivered_kg",
    "requested_kg",
    "unmet_kg",
    "worst_commitment_fill_rate",
    "commitments_settled",
    "commitments_met",
    "ch4_produced_kg",
    "battery_throughput_kwh",
    "electrolyser_cold_starts",
    "electrolyser_hot_starts",
    "electrolyser_nonproductive_energy_kwh",
    "electrolyser_forced_stops",
    "guard_interventions",
    "terminal_h2_kg",
    "terminal_co2_kg",
    "terminal_ch4_kg",
    "terminal_battery_soc",
)


def numeric(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
    ):
        raise ValueError("expected a finite numeric metric")
    return float(value)


def describe(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("an empty subset cannot supply an aggregate")
    values = [numeric(v) for v in values]
    ordered = sorted(values)
    # Exact empirical order-statistic interpolation, explicitly descriptive only.
    position = 0.1 * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "minimum": min(values),
        "maximum": max(values),
        "empirical_p10": ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo]),
        "positive_count": sum(v > 0 for v in values),
        "negative_count": sum(v < 0 for v in values),
        "values": values,
    }


def metrics(summary: dict[str, Any]) -> dict[str, float]:
    result = {key: numeric(summary["physical_metrics"][key]) for key in PHYSICAL}
    result["net_operating_value_eur"] = numeric(
        summary["economic_metrics"]["net_operating_value_eur"]
    )
    for key, value in summary["economic_metrics"]["components_eur"].items():
        result["economic_component_" + key] = numeric(value)
    for key in ("fallback_decisions", "recovery_decisions"):
        result[key] = numeric(summary[key])
    return result


def complete_slots(actor: dict[str, Any]) -> bool:
    return all(s["eligible_for_descriptive_comparison"] is True for s in actor["slots"])


def actor_report(name: str, data: dict[str, Any]) -> dict[str, Any]:
    slots = data["slots"]
    if tuple(s["seed"] for s in slots) != SEEDS:
        raise ValueError("actor must retain the complete, ordered eight-seed panel")
    for slot in slots:
        summary = slot["summary"]
        if slot["eligible_for_descriptive_comparison"] is True:
            if (
                not isinstance(summary, dict)
                or summary["execution_complete"] is not True
            ):
                raise ValueError(
                    "comparison eligibility contradicts execution evidence"
                )
            if tuple(summary["resource_order"]) != RESOURCES:
                raise ValueError("resource vector order differs")
            if len(summary["cycle_deficit_vectors"]) != 7 or any(
                len(v) != 4 for v in summary["cycle_deficit_vectors"]
            ):
                raise ValueError(
                    "complete operation requires seven four-resource settlements"
                )
            for vector in summary["cycle_deficit_vectors"]:
                for value in vector:
                    numeric(value)
    eligible = complete_slots(data)
    result: dict[str, Any] = {
        "actor": name,
        "fixed_scenario_count": len(SEEDS),
        "execution_eligible_count": sum(
            s["eligible_for_descriptive_comparison"] is True for s in slots
        ),
        "hard_safety_pass_count": sum(
            s["eligible_for_descriptive_comparison"] is True
            and s["summary"] is not None
            and s["summary"]["hard_safety_passed"] is True
            for s in slots
        ),
        "all_daily_returns_pass_count": sum(
            s["eligible_for_descriptive_comparison"] is True
            and s["summary"] is not None
            and s["summary"]["all_cycles_feasible"] is True
            for s in slots
        ),
        "aggregate_available": eligible,
        "aggregate": None,
        "slots": slots,
    }
    if eligible:
        rows = [metrics(s["summary"]) for s in slots]
        if any(row.keys() != rows[0].keys() for row in rows):
            raise ValueError("economic component schema differs between complete cases")
        result["aggregate"] = {
            key: describe([row[key] for row in rows]) for key in rows[0]
        }
    return result


def paired_report(
    candidate: str, comparator: str, actors: dict[str, Any]
) -> dict[str, Any]:
    left, right = actors[candidate], actors[comparator]
    result: dict[str, Any] = {
        "candidate": candidate,
        "comparator": comparator,
        "all_eight_pairs_available": left["aggregate_available"]
        and right["aggregate_available"],
        "direction": "candidate minus comparator; higher EUR/service is better; lower wear/deficit is better",
        "paired_differences": None,
        "daily_resource_differences": None,
        "candidate_all_cases_hard_safety_passed": left["hard_safety_pass_count"]
        == len(SEEDS),
        "candidate_all_cases_daily_returns_passed": left["all_daily_returns_pass_count"]
        == len(SEEDS),
        "confirmation_or_material_benefit_claim": False,
    }
    if not result["all_eight_pairs_available"]:
        result["unavailable_seeds"] = [
            a["seed"]
            for a, b in zip(left["slots"], right["slots"], strict=True)
            if not a["eligible_for_descriptive_comparison"]
            or not b["eligible_for_descriptive_comparison"]
        ]
        return result
    differences: list[dict[str, float]] = []
    daily: list[dict[str, Any]] = []
    for a, b in zip(left["slots"], right["slots"], strict=True):
        if a["seed"] != b["seed"]:
            raise ValueError("paired seeds differ")
        am, bm = metrics(a["summary"]), metrics(b["summary"])
        if am.keys() != bm.keys():
            raise ValueError("paired metric schema differs")
        differences.append({key: am[key] - bm[key] for key in am})
        for index, (av, bv) in enumerate(
            zip(
                a["summary"]["cycle_deficit_vectors"],
                b["summary"]["cycle_deficit_vectors"],
                strict=True,
            )
        ):
            daily.append(
                {
                    "seed": a["seed"],
                    "cycle": index + 1,
                    "candidate_minus_comparator": {
                        resource: numeric(x) - numeric(y)
                        for resource, x, y in zip(RESOURCES, av, bv, strict=True)
                    },
                }
            )
    result["paired_differences"] = {
        key: describe([d[key] for d in differences]) for key in differences[0]
    }
    result["daily_resource_differences"] = {
        "all_56_boundaries": daily,
        "each_resource": {
            resource: describe(
                [row["candidate_minus_comparator"][resource] for row in daily]
            )
            for resource in RESOURCES
        },
        "interpretation": "Raw individual differences are retained; no weighted sum or post-outcome noninferiority threshold is applied.",
    }
    return result


def training_report(arm: str, data: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "arm": arm,
        "closed_successfully": data["closed_successfully"],
        "diagnostic_rows": [],
        "interpretation": "pre-update diagnostic sample, not actual shuffled SGD or Adam updates",
    }
    if not data["closed_successfully"]:
        result["failure_evidence"] = {
            k: v for k, v in data.items() if k not in ("diagnostics", "result")
        }
        return result
    scale = numeric(data["result"]["arm"]["reward_scale"])
    if scale <= 0:
        raise ValueError("reward scale must be positive")
    records = data["diagnostics"]
    if len(records) != 16 or [d["num_timesteps"] for d in records] != list(
        range(8192, 131073, 8192)
    ):
        raise ValueError(
            "completed training must retain all sixteen fixed diagnostic rollouts"
        )
    for record in records:
        gradient = record["gradient_attribution"]
        sample = record["sample"]
        gaussian = sample["gaussian"]
        result["diagnostic_rows"].append(
            {
                "num_timesteps": record["num_timesteps"],
                "reward_scale": scale,
                "reward_explained_variance": record["full_rollout"][
                    "reward_explained_variance"
                ],
                "reward_critic_tanh": sample["reward_critic_tanh"],
                "reward_critic_preclip_gradient_norm": gradient[
                    "reward_critic_preclip_gradient_norm"
                ],
                "actor_total_preclip_gradient_norm": gradient[
                    "actor_total_preclip_gradient_norm"
                ],
                "implied_common_clip_factor": gradient["implied_common_clip_factor"],
                "diagnostic_actor_norm_after_common_clip": numeric(
                    gradient["actor_total_preclip_gradient_norm"]
                )
                * numeric(gradient["implied_common_clip_factor"]),
                "critic_mse_optimizer_units_squared": gradient["reward_value_mse"],
                "critic_mse_eur_squared": numeric(gradient["reward_value_mse"])
                / scale**2,
                "weighted_cost_actor_gradient_norm": gradient[
                    "weighted_cost_actor_gradient_norm"
                ],
                "reward_actor_gradient_norm": gradient["reward_actor_gradient_norm"],
                "reward_vs_weighted_cost_gradient_cosine": gradient[
                    "reward_vs_weighted_cost_gradient_cosine"
                ],
                "cost_actor_components": gradient["cost_actor_components"],
                "costs": record["full_rollout"]["costs"],
                "multipliers_and_normalized_cost_advantages": sample["costs"],
                "action_means": [r["mean"] for r in gaussian["mean_by_action"]],
                "action_std": [r["mean"] for r in gaussian["std_by_action"]],
                "expected_action_space_clipping": gaussian[
                    "expected_clipped_fraction_all_action_coordinates"
                ],
                "observed_action_space_clipping": gaussian[
                    "observed_clipped_fraction_all_action_coordinates"
                ],
            }
        )
    return result


def make_report(packet: dict[str, Any]) -> dict[str, Any]:
    if (
        packet.get("integrity_verified") is not True
        or packet.get("batch_closed") is not True
    ):
        raise ValueError(
            "analysis requires a fully closed and integrity-verified batch"
        )
    if set(packet["actors"]) != set((*ARMS, *COMPARATORS)) or set(
        packet["training"]
    ) != set(ARMS):
        raise ValueError("analysis requires all fixed actor and training slots")
    actors = {
        name: actor_report(name, packet["actors"][name])
        for name in (*COMPARATORS, *ARMS)
    }
    for name in (*ARMS, "initial"):
        source_arm = ARMS[0] if name == "initial" else name
        if (
            actors[name]["execution_eligible_count"]
            and packet["training"][source_arm]["closed_successfully"] is not True
        ):
            raise ValueError("eligible policy contradicts failed training closure")
    pairs = [(arm, comparator) for arm in ARMS for comparator in COMPARATORS]
    mechanism_pairs = [
        ("B-pi-scaled", "A-pi-raw"),
        ("D-scalar-scaled", "C-scalar-raw"),
        ("A-pi-raw", "C-scalar-raw"),
        ("B-pi-scaled", "D-scalar-scaled"),
        ("E-pi-scaled-lowstd", "B-pi-scaled"),
    ]
    report = {
        "schema": "plant-adaptive-development-analysis-v1",
        "audited_packet_sha256": hashlib.sha256(
            json.dumps(
                packet, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest(),
        "fixed_development_seeds": list(SEEDS),
        "actors": actors,
        "training": {
            arm: training_report(arm, packet["training"][arm]) for arm in ARMS
        },
        "operational_comparisons": [paired_report(a, b, actors) for a, b in pairs],
        "mechanism_comparisons": [
            paired_report(a, b, actors) for a, b in mechanism_pairs
        ],
        "limits": [
            "One optimizer seed and adaptive development scenarios; no generalization or algorithm-level claim.",
            "Failed/stopped scenarios stay visible; no complete-case subset supplies a performance mean.",
            "No checkpoint selection, practical effect threshold, p-value or confirmation claim is produced.",
            "An economic gain is not sufficient if service, safety or any daily resource deficit worsens.",
            "Critic MSE conversion restores units only; visited states and bootstrapped targets can differ.",
            "Pre-update gradient clipping diagnostics are not actual shuffled minibatch or Adam parameter updates.",
            "Scalar cost diagnostics are unavailable, never zero resource violations.",
            "Physical replay does not independently authenticate reported controller failures.",
        ],
    }
    json.dumps(report, allow_nan=False)
    return report


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Adaptive PPO development batch 001",
        "",
        "All fixed scenarios and failed actors are retained. This is descriptive development evidence.",
        "",
        "| Actor | Complete/replayed | Safety passes | All daily returns | Mean delivery | Mean EUR/week |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for actor in report["actors"].values():
        aggregate = actor["aggregate"]
        delivery = (
            "unavailable"
            if aggregate is None
            else f"{100*aggregate['delivery_fill_rate']['mean']:.2f}%"
        )
        eur = (
            "unavailable"
            if aggregate is None
            else f"{aggregate['net_operating_value_eur']['mean']:,.2f}"
        )
        lines.append(
            f"| {actor['actor']} | {actor['execution_eligible_count']}/8 | {actor['hard_safety_pass_count']}/8 | {actor['all_daily_returns_pass_count']}/8 | {delivery} | {eur} |"
        )
    lines.extend(
        [
            "",
            "## Paired operating comparisons",
            "",
            "| PPO variant | Comparator | Mean EUR difference | Worst EUR difference | Minimum delivered kg difference |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for pair in report["operational_comparisons"]:
        differences = pair["paired_differences"]
        if differences is None:
            values = "unavailable | unavailable | unavailable"
        else:
            values = f"{differences['net_operating_value_eur']['mean']:,.2f} | {differences['net_operating_value_eur']['minimum']:,.2f} | {differences['delivered_kg']['minimum']:,.6f}"
        lines.append(f"| {pair['candidate']} | {pair['comparator']} | {values} |")
    lines.extend(
        [
            "",
            "## Daily resource differences",
            "",
            "Maximum candidate-minus-comparator deficit over all 56 daily boundaries; positive values identify a worsened resource deficit.",
            "",
            "| PPO variant | Comparator | Battery kWh | H2 kg | CO2 kg | CH4 kg |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for pair in report["operational_comparisons"]:
        daily = pair["daily_resource_differences"]
        values = (
            "unavailable | unavailable | unavailable | unavailable"
            if daily is None
            else " | ".join(
                f"{daily['each_resource'][resource]['maximum']:,.6f}"
                for resource in RESOURCES
            )
        )
        lines.append(f"| {pair['candidate']} | {pair['comparator']} | {values} |")
    lines.extend(
        [
            "",
            "## Evidence limits",
            "",
            *("- " + line for line in report["limits"]),
            "",
            "The JSON report retains every scenario, economic contribution and all 56 paired daily resource boundaries.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audited_packet", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    packet = json.loads(args.audited_packet.read_text())
    result = make_report(packet)
    args.output_directory.mkdir(exist_ok=False)
    (args.output_directory / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (args.output_directory / "analysis.md").write_text(markdown(result))


if __name__ == "__main__":
    main()

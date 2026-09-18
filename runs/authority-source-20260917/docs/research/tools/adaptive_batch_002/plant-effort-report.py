"""All-slot descriptive analysis of an independently audited, closed PPO batch.

No environment, policy or training execution occurs here. This is development
analysis, with no policy selection, materiality threshold or confirmation claim.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

AUDIT_SCHEMA = "plant-effort-closed-batch-audit-v1"
REPORT_SCHEMA = "plant-effort-development-analysis-v1"
PINNED_HEAD = "a442a4a9f5b028568d36bbbddf21a34c3609a7cc"
ARMS = ("A-absolute-pi", "B-absolute-scalar", "C-residual-pi", "D-residual-scalar")
BASELINES = ("H", "OperatingMPC", "M24", "E-beta")
INITIAL_SOURCES = {"initial-absolute": ARMS[0], "initial-residual": ARMS[2]}
COMPARATORS = (*BASELINES, *INITIAL_SOURCES)
OWN_INITIAL = {
    arm: "initial-residual" if "residual" in arm else "initial-absolute" for arm in ARMS
}
METHOD_PAIRS = (
    (ARMS[1], ARMS[0]),
    (ARMS[3], ARMS[2]),
    (ARMS[2], ARMS[0]),
    (ARMS[3], ARMS[1]),
)
SEEDS = tuple(range(9_700_100, 9_700_108))
REWARD_SCALE = 0.001
INITIAL_STD = 0.2
ROLLOUT_TRANSITIONS = 8192
ROLLOUT_COUNT = 16
ADAPTED_REPORTER_SHA256 = (
    "680208dbd02d4fdff95fe23e1b5c00e9dde513969235bb53d2f63566f5b073a0"
)

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
    "electrolyser_aborted_starts",
    "electrolyser_startup_energy_kwh",
    "electrolyser_standby_energy_kwh",
    "electrolyser_low_load_time_s",
    "min_battery_soc",
    "out_of_range_decisions",
    "request_application_gap_kwh",
    "guard_intervention_kwh",
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


def _optional_numeric(value: Any) -> float | None:
    return None if value is None else numeric(value)


def _multiply(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return numeric(left) * numeric(right)


def _scale(value: Any, scale: float) -> float | None:
    return None if value is None else numeric(value) / scale


def daily_grid(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every scheduled daily slot remains visible, including an absent prefix."""
    result = []
    for slot in slots:
        summary = slot["summary"]
        vectors = [] if summary is None else summary["cycle_deficit_vectors"]
        if len(vectors) > 7:
            raise ValueError("more than seven daily settlements")
        for index in range(7):
            vector = vectors[index] if index < len(vectors) else None
            if vector is not None and len(vector) != len(RESOURCES):
                raise ValueError("daily vector must retain four resources")
            result.append(
                {
                    "seed": slot["seed"],
                    "cycle": index + 1,
                    "eligible_complete_episode": slot[
                        "eligible_for_descriptive_comparison"
                    ],
                    "recorded": vector is not None,
                    "deficits": (
                        None
                        if vector is None
                        else {
                            resource: numeric(value)
                            for resource, value in zip(RESOURCES, vector, strict=True)
                        }
                    ),
                }
            )
    return result


def training_report(arm: str, data: dict[str, Any]) -> dict[str, Any]:
    """Keep every raw diagnostic and ledger; derive curves without selection."""
    if data["closed_successfully"] not in (True, False) or not isinstance(
        data["closed_successfully"], bool
    ):
        raise ValueError("training closure must be a boolean")
    success = data["closed_successfully"]
    records, ledgers = data["diagnostics"], data["rollouts"]
    if len(records) > ROLLOUT_COUNT or len(ledgers) > ROLLOUT_COUNT:
        raise ValueError("training cannot exceed sixteen fixed rollouts")
    if [d["num_timesteps"] for d in records] != [
        ROLLOUT_TRANSITIONS * i for i in range(1, len(records) + 1)
    ]:
        raise ValueError("diagnostic sequence must be the complete retained prefix")
    if [d["rollout_index"] for d in ledgers] != list(range(1, len(ledgers) + 1)):
        raise ValueError("ledger sequence must be the complete retained prefix")
    if any(d["transition_count"] != ROLLOUT_TRANSITIONS for d in ledgers):
        raise ValueError("ledger transition count differs from the fixed design")
    if len(ledgers) > len(records):
        raise ValueError(
            "a published optimizer ledger must retain its pre-update diagnostic"
        )
    if success:
        if len(records) != ROLLOUT_COUNT or len(ledgers) != ROLLOUT_COUNT:
            raise ValueError(
                "completed training must retain all sixteen diagnostic and optimizer rollouts"
            )
        result_arm = data["result"]["arm"]
        if (
            result_arm["name"] != arm
            or numeric(result_arm["reward_scale"]) != REWARD_SCALE
            or numeric(result_arm["initial_std"]) != INITIAL_STD
            or result_arm["residual"] is not ("residual" in arm)
        ):
            raise ValueError(
                "training arm or fixed reward/exploration/interface design differs"
            )
    elif data["result"] is not None:
        raise ValueError("failed training must not supply an admitted final result")
    report: dict[str, Any] = {
        "arm": arm,
        "closed_successfully": success,
        "raw_training_evidence": deepcopy(data),
        "reward_scale": REWARD_SCALE,
        "initial_gaussian_std": INITIAL_STD,
        "policy_observation_size": 139 if "residual" in arm else 135,
        "raw_diagnostic_count": len(records),
        "optimizer_ledger_count": len(ledgers),
        "diagnostic_rows": [],
        "optimizer_rows": [],
        "failure_evidence": deepcopy(data.get("failures", [])),
        "units": {
            "optimizer_rewards": "raw interval EUR multiplied by 0.001 before rollout-buffer float32 storage",
            "rollout_reward_eur_equivalent": "unit-rescaled optimizer reward sum; not a replacement for the physical raw-EUR ledger",
            "critic_mse_optimizer_units_squared": "squared scaled-EUR training targets",
            "critic_mse_eur_squared": "MSE divided by 0.001 squared; unit conversion only, with potentially different visited states and bootstrapped targets",
            "evaluation_rewards": "unmodified raw interval EUR",
            "resource_costs": "unmodified daily physical/normalized constraint signals",
        },
        "clipping_definitions": {
            "gaussian_action_bounds": "Gaussian action-coordinate probability or observed sample outside [-1, 1], before environment action clipping; pre-update fixed diagnostic sample",
            "ppo_ratio_clip_fraction": "optimizer minibatch fraction whose policy probability ratio leaves the PPO ratio-clip band; this is not action saturation",
            "common_gradient_clip": "pre-update diagnostic implied common actor/reward-critic gradient clipping; not actual shuffled SGD gradients or Adam parameter updates",
        },
        "scalar_cost_semantics": "Raw ledger constraint totals remain physical evidence. Scalar cost-critic diagnostic fields remain null; ledger cost-value/multiplier placeholders do not establish learned cost critics or zero physical deficits.",
    }
    scalar = arm.endswith("-scalar")
    for record in records:
        sample = record["sample"]
        gaussian = sample["gaussian"]
        gradient = record["gradient_attribution"]
        full = record["full_rollout"]
        if len(gaussian["mean_by_action"]) != 4 or len(gaussian["std_by_action"]) != 4:
            raise ValueError("Gaussian diagnostics must retain four action coordinates")
        report["diagnostic_rows"].append(
            {
                "num_timesteps": record["num_timesteps"],
                "reward_scale": REWARD_SCALE,
                "full_rollout_training_statistics": deepcopy(full),
                "reward_explained_variance": _optional_numeric(
                    full["reward_explained_variance"]
                ),
                "reward_critic_tanh": deepcopy(sample["reward_critic_tanh"]),
                "reward_critic_preclip_gradient_norm": _optional_numeric(
                    gradient["reward_critic_preclip_gradient_norm"]
                ),
                "actor_total_preclip_gradient_norm": _optional_numeric(
                    gradient["actor_total_preclip_gradient_norm"]
                ),
                "implied_common_clip_factor": _optional_numeric(
                    gradient["implied_common_clip_factor"]
                ),
                "diagnostic_actor_norm_after_common_clip": _multiply(
                    gradient["actor_total_preclip_gradient_norm"],
                    gradient["implied_common_clip_factor"],
                ),
                "critic_mse_optimizer_units_squared": _optional_numeric(
                    gradient["reward_value_mse"]
                ),
                "critic_mse_eur_squared": _scale(
                    gradient["reward_value_mse"], REWARD_SCALE**2
                ),
                "weighted_cost_actor_gradient_norm": _optional_numeric(
                    gradient["weighted_cost_actor_gradient_norm"]
                ),
                "reward_actor_gradient_norm": _optional_numeric(
                    gradient["reward_actor_gradient_norm"]
                ),
                "reward_vs_weighted_cost_gradient_cosine": _optional_numeric(
                    gradient["reward_vs_weighted_cost_gradient_cosine"]
                ),
                "cost_actor_components": deepcopy(gradient["cost_actor_components"]),
                "costs": deepcopy(full["costs"]),
                "multipliers_and_normalized_cost_advantages": deepcopy(sample["costs"]),
                "action_means": [
                    numeric(v["mean"]) for v in gaussian["mean_by_action"]
                ],
                "action_std": [numeric(v["mean"]) for v in gaussian["std_by_action"]],
                "expected_action_space_clipping": numeric(
                    gaussian["expected_clipped_fraction_all_action_coordinates"]
                ),
                "observed_action_space_clipping": numeric(
                    gaussian["observed_clipped_fraction_all_action_coordinates"]
                ),
                "ppo_probability_ratio_diagnostic": deepcopy(
                    sample.get("probability_ratio")
                ),
            }
        )
    for ledger in ledgers:
        report["optimizer_rows"].append(
            {
                "rollout_index": ledger["rollout_index"],
                "num_timesteps": ledger["rollout_index"] * ROLLOUT_TRANSITIONS,
                "transition_count": ledger["transition_count"],
                "completed_episode_count": ledger["completed_episode_count"],
                "complete_cycle_count": ledger["complete_cycle_count"],
                "optimizer_reward_sum": numeric(ledger["reward_return"]),
                "optimizer_reward_sum_eur_equivalent": numeric(ledger["reward_return"])
                / REWARD_SCALE,
                "raw_cost": deepcopy(ledger["raw_cost"]),
                "training_cost": deepcopy(ledger["training_cost"]),
                "mean_completed_episode_cost": deepcopy(
                    ledger["mean_completed_episode_cost"]
                ),
                "learner_cost_value_losses": (
                    None if scalar else deepcopy(ledger["cost_value_losses"])
                ),
                "learner_multipliers": (
                    None if scalar else deepcopy(ledger["multipliers"])
                ),
                "approximate_kl": numeric(ledger["approximate_kl"]),
                "entropy_loss": numeric(ledger["entropy_loss"]),
                "ppo_ratio_clip_fraction": numeric(ledger["clip_fraction"]),
                "optimizer_steps": ledger["optimizer_steps"],
                "elapsed_compute_s": numeric(ledger["elapsed_compute_s"]),
                "ledger_sha256": ledger["sha256"],
            }
        )
    return report


def validate_equivalence(
    equivalence: dict[str, Any], actors: dict[str, Any]
) -> dict[str, Any]:
    """Validate and surface the auditor's full physical projection comparison."""
    slots = equivalence["slots"]
    if tuple(slot["seed"] for slot in slots) != SEEDS:
        raise ValueError("equivalence audit must retain all eight ordered pairs")
    available_equal = []
    for slot, h, initial in zip(
        slots, actors["H"]["slots"], actors["initial-residual"]["slots"], strict=True
    ):
        available = (
            h["eligible_for_descriptive_comparison"] is True
            and initial["eligible_for_descriptive_comparison"] is True
        )
        if slot["available"] is not available:
            raise ValueError(
                "equivalence pair availability contradicts audited episodes"
            )
        if available:
            checks = slot["comparisons"]
            if (
                not isinstance(checks, dict)
                or not checks
                or any(not isinstance(v, bool) for v in checks.values())
            ):
                raise ValueError(
                    "available equivalence requires explicit physical-field checks"
                )
            equal = all(checks.values())
            if slot["equal"] is not equal:
                raise ValueError("equivalence equality contradicts individual checks")
            if set(slot["archive_records"]) != {"H", "initial-residual"}:
                raise ValueError("equivalence audit lacks paired archive provenance")
            available_equal.append(equal)
        elif slot["equal"] is not None:
            raise ValueError("unavailable physical equivalence must remain null")
    all_available = len(available_equal) == len(SEEDS)
    if (
        equivalence["all_eight_pairs_available"] is not all_available
        or equivalence["all_available_pairs_equal"]
        is not (all(available_equal) if available_equal else None)
        or equivalence["all_eight_pairs_equal"]
        is not (all(available_equal) if all_available else None)
    ):
        raise ValueError("equivalence aggregate contradicts full paired slots")
    return deepcopy(equivalence)


def make_report(packet: dict[str, Any]) -> dict[str, Any]:
    # The auditor schema guarantees it opens outcomes only after verifying a
    # normal closed report. A partial/in-progress/aborted audit is not admissible.
    if (
        packet.get("schema") != AUDIT_SCHEMA
        or packet.get("integrity_verified") is not True
        or packet.get("batch_closed") is not True
        or packet.get("registered_evaluation_eligible") is not False
    ):
        raise ValueError(
            "analysis requires the normally closed, integrity-verified effort-batch audit schema"
        )
    if packet.get("source", {}).get("git_head") != PINNED_HEAD:
        raise ValueError("audited source differs from frozen a442a4a")
    if set(packet["actors"]) != set((*COMPARATORS, *ARMS)) or set(
        packet["training"]
    ) != set(ARMS):
        raise ValueError("analysis requires all fixed actor and training slots")
    packet = deepcopy(packet)
    actors = {
        name: actor_report(name, packet["actors"][name])
        for name in (*COMPARATORS, *ARMS)
    }
    for name, data in actors.items():
        if any(
            not isinstance(slot["eligible_for_descriptive_comparison"], bool)
            for slot in data["slots"]
        ):
            raise ValueError("comparison eligibility must be a boolean")
        data["all_56_daily_resource_slots"] = daily_grid(data["slots"])
    for name in (*ARMS, *INITIAL_SOURCES):
        source_arm = INITIAL_SOURCES.get(name, name)
        if (
            actors[name]["execution_eligible_count"]
            and packet["training"][source_arm]["closed_successfully"] is not True
        ):
            raise ValueError("eligible policy contradicts failed training closure")
    equivalence = validate_equivalence(
        packet["initial_residual_heuristic_equivalence"], actors
    )
    pairs = [
        (arm, comparator)
        for arm in ARMS
        for comparator in (OWN_INITIAL[arm], *BASELINES)
    ]
    result = {
        "schema": REPORT_SCHEMA,
        "audited_packet_sha256": hashlib.sha256(
            json.dumps(
                packet, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest(),
        "reporter_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "adapted_batch001_reporter_sha256": ADAPTED_REPORTER_SHA256,
        "fixed_development_seeds": list(SEEDS),
        "optimizer_seed": 101,
        "registered_evaluation_eligible": False,
        "selected_arm": None,
        "operational_benefit_claim": False,
        "confirmation_or_materiality_threshold": None,
        "audit_provenance": {
            key: deepcopy(packet.get(key))
            for key in (
                "schema",
                "root",
                "source",
                "plan",
                "report",
                "auditor_source",
                "artifact_records",
                "integrity",
                "initial_tensor_comparison",
            )
        },
        "actors": actors,
        "initial_residual_heuristic_equivalence": equivalence,
        "training": {
            arm: training_report(arm, packet["training"][arm]) for arm in ARMS
        },
        "operational_comparisons": [paired_report(a, b, actors) for a, b in pairs],
        "method_comparisons": [paired_report(a, b, actors) for a, b in METHOD_PAIRS],
        "method_contrast_scope": {
            "B-A and D-C": "matched scalar-minus-PI comparisons within each representation",
            "C-A and D-B": "residual-minus-absolute method-package contrasts; input features, representation and initialization change together",
            "initialization_comparators": "initial-absolute is from A; initial-residual is from C; failed source training leaves its comparator unavailable",
        },
        "limits": [
            "One optimizer seed 101 and eight adaptive development cases. No held-out, site-generalization, commercial or algorithm-level claim.",
            "No arm is selected or declared an operational winner. No materiality or confirmation threshold is chosen after outcomes.",
            "Full-panel aggregates and paired differences require all eight complete and replayed episodes. Failed/stopped/null slots remain visible and cannot produce subset wins.",
            "Economic differences must be read with per-case service, strict safety including forced stops, and every daily four-resource deficit; a favorable mean alone is insufficient.",
            "The initial residual-versus-H check is the auditor's per-case full physical projection equivalence, not equality of an average performance number.",
            "Absolute and residual Gaussian coordinates have different operational meanings. Residual H features are causal derived inputs, and the mapping restricts startup/stop authority without guaranteeing feasibility or safety.",
            "All sixteen diagnostic and optimizer ledger rows are retained for completed training; failed training retains every available prefix and no replacement arm or initialization.",
            "Gaussian action-bound clipping, PPO probability-ratio clipping and common gradient clipping are distinct quantities. Diagnostic gradients are not actual shuffled SGD gradients or Adam parameter updates.",
            "All optimizer rewards use 0.001 times EUR. Unit-restored rewards/MSE do not undo float32 rounding, visited-state differences or different bootstrapped targets; evaluated economics stay raw EUR.",
            "Scalar cost-critic diagnostics remain null, not zero physical violations. Raw constraint ledger costs are retained for every learner.",
            "Physical replay does not itself prove the checkpoint generated saved actions or authenticate a comparator exception; process/checkpoint/source custody is separate.",
        ],
    }
    json.dumps(result, allow_nan=False)
    return result


def _cell(value: Any, *, digits: int = 6) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{numeric(value):,.{digits}f}"
    text = (
        json.dumps(value, ensure_ascii=False, allow_nan=False)
        if isinstance(value, (dict, list))
        else str(value)
    )
    return text.replace("|", "\\|").replace("\n", " ")


def _pair_table(lines: list[str], pairs: list[dict[str, Any]], title: str) -> None:
    lines.extend(
        [
            "",
            f"## {title}",
            "",
            "Candidate minus comparator. All eight complete/replayed pairs are required; positive EUR/service and negative wear/deficit differences have different meanings.",
            "",
            "| Candidate | Comparator | Mean EUR difference | Worst EUR difference | Minimum delivered kg difference |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for pair in pairs:
        d = pair["paired_differences"]
        values = (
            "unavailable | unavailable | unavailable"
            if d is None
            else " | ".join(
                _cell(v)
                for v in (
                    d["net_operating_value_eur"]["mean"],
                    d["net_operating_value_eur"]["minimum"],
                    d["delivered_kg"]["minimum"],
                )
            )
        )
        lines.append(f"| {pair['candidate']} | {pair['comparator']} | {values} |")
    lines.extend(
        [
            "",
            "All 56 individual paired daily vectors remain in JSON. This table shows the maximum candidate-minus-comparator deficit for each resource; a positive entry identifies at least one worsened daily deficit.",
            "",
            "| Candidate | Comparator | Battery kWh | H2 kg | CO2 kg | CH4 kg |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for pair in pairs:
        daily = pair["daily_resource_differences"]
        values = (
            "unavailable | unavailable | unavailable | unavailable"
            if daily is None
            else " | ".join(
                _cell(daily["each_resource"][resource]["maximum"])
                for resource in RESOURCES
            )
        )
        lines.append(f"| {pair['candidate']} | {pair['comparator']} | {values} |")


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Adaptive PPO development batch 002",
        "",
        "One optimizer seed, four fixed methods and eight adaptive development cases. All failed/null slots remain visible. No operational winner, confirmation threshold or benefit claim is assigned.",
        "",
        "## Initial residual versus H",
        "",
    ]
    equivalence = report["initial_residual_heuristic_equivalence"]
    lines.extend(
        [
            f"All eight physical pairs available: {_cell(equivalence['all_eight_pairs_available'])}. All eight pairs equal: {_cell(equivalence['all_eight_pairs_equal'])}.",
            "",
            "| Seed | Available | Full physical projection equal | Reason/checks |",
            "|---|---|---|---|",
        ]
    )
    for slot in equivalence["slots"]:
        lines.append(
            f"| {slot['seed']} | {_cell(slot['available'])} | {_cell(slot['equal'])} | {_cell({'reason': slot.get('reason'), 'comparisons': slot.get('comparisons')})} |"
        )
    lines.extend(
        [
            "",
            "## Complete-panel outcomes",
            "",
            "| Actor | Complete/replayed | Strict safety passes | All daily returns | Mean delivery | Mean EUR/week |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for actor in report["actors"].values():
        aggregate = actor["aggregate"]
        delivery = (
            "unavailable"
            if aggregate is None
            else f"{100 * aggregate['delivery_fill_rate']['mean']:.2f}%"
        )
        eur = (
            "unavailable"
            if aggregate is None
            else _cell(aggregate["net_operating_value_eur"]["mean"], digits=2)
        )
        lines.append(
            f"| {actor['actor']} | {actor['execution_eligible_count']}/8 | {actor['hard_safety_pass_count']}/8 | {actor['all_daily_returns_pass_count']}/8 | {delivery} | {eur} |"
        )
    _pair_table(
        lines,
        report["method_comparisons"],
        "Prespecified method contrasts B-A, D-C, C-A, D-B",
    )
    _pair_table(
        lines,
        report["operational_comparisons"],
        "Each final versus its own initialization and required baselines",
    )
    lines.extend(
        [
            "",
            "## Every scenario",
            "",
            "Recorded EUR/service can be a stopped physical prefix in an ineligible row. Such rows never enter means or paired comparisons. Safety here retains the recorded classifier result; a prefix result cannot count as a complete-episode pass.",
            "",
            "| Actor | Seed | Eligible | Recorded fill | Recorded EUR | Strict safety | Forced stops | Min SOC | Allowed SOC | Daily returns | Failures |",
            "|---|---:|---|---:|---:|---|---:|---:|---:|---|---|",
        ]
    )
    for actor in report["actors"].values():
        for slot in actor["slots"]:
            s = slot["summary"] or {}
            p, e = s.get("physical_metrics", {}), s.get("economic_metrics", {})
            failures = {
                "slot": slot.get("failures"),
                "controller": s.get("failure"),
                "safety_classifier": s.get("hard_safety_classifier_failure"),
            }
            values = [
                actor["actor"],
                slot["seed"],
                slot["eligible_for_descriptive_comparison"],
                p.get("delivery_fill_rate"),
                e.get("net_operating_value_eur"),
                s.get("hard_safety_passed"),
                p.get("electrolyser_forced_stops"),
                p.get("min_battery_soc"),
                s.get("hard_safety_metrics", {}).get("minimum_allowed_battery_soc"),
                s.get("all_cycles_feasible"),
                failures,
            ]
            lines.append(
                "| "
                + " | ".join(
                    _cell(v, digits=6) if i not in (0, 1) else str(v)
                    for i, v in enumerate(values)
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Training curves",
            "",
            "Every completed cell retains 16 pre-update diagnostic rows and 16 optimizer ledgers. Reward is EUR × 0.001; the equivalent-EUR column only rescales rollout-buffer rewards. Gaussian clipping is action-coordinate clipping outside [-1, 1]; PPO ratio clipping is the optimizer's probability-ratio statistic. Missing cost-critic or unfinished optimizer data stays null.",
        ]
    )
    for arm in ARMS:
        t = report["training"][arm]
        lines.extend(
            [
                "",
                f"### {arm}",
                "",
                f"Closed successfully: {_cell(t['closed_successfully'])}. Failures: {_cell(t['failure_evidence'])}.",
                "",
                "| Rollout | Steps | Optimizer reward sum | EUR equivalent | Reward EV | Critic MSE scaled² | Critic MSE EUR² | Expected Gaussian clip | Observed Gaussian clip | PPO ratio clip | KL | Action means | Action std |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        optimizer = {r["num_timesteps"]: r for r in t["optimizer_rows"]}
        for index, d in enumerate(t["diagnostic_rows"], start=1):
            o = optimizer.get(d["num_timesteps"], {})
            values = [
                index,
                d["num_timesteps"],
                o.get("optimizer_reward_sum"),
                o.get("optimizer_reward_sum_eur_equivalent"),
                d["reward_explained_variance"],
                d["critic_mse_optimizer_units_squared"],
                d["critic_mse_eur_squared"],
                d["expected_action_space_clipping"],
                d["observed_action_space_clipping"],
                o.get("ppo_ratio_clip_fraction"),
                o.get("approximate_kl"),
                d["action_means"],
                d["action_std"],
            ]
            lines.append(
                "| "
                + " | ".join(
                    str(v) if i < 2 else _cell(v) for i, v in enumerate(values)
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Every daily resource vector",
            "",
            "Every actor has 56 scheduled daily slots below. Null means no retained settlement at that boundary; it never means zero deficit. Ineligible-prefix settlements remain visible and are excluded from paired aggregates.",
        ]
    )
    for actor in report["actors"].values():
        lines.extend(
            [
                "",
                f"### {actor['actor']}",
                "",
                "| Seed | Cycle | Eligible episode | Battery kWh | H2 kg | CO2 kg | CH4 kg |",
                "|---:|---:|---|---:|---:|---:|---:|",
            ]
        )
        for row in actor["all_56_daily_resource_slots"]:
            d = row["deficits"] or {}
            lines.append(
                f"| {row['seed']} | {row['cycle']} | {_cell(row['eligible_complete_episode'])} | "
                + " | ".join(_cell(d.get(resource)) for resource in RESOURCES)
                + " |"
            )
    lines.extend(
        [
            "",
            "## Evidence limits",
            "",
            *("- " + line for line in report["limits"]),
            "",
            "JSON additionally retains every raw slot, EUR component, training diagnostic/ledger/receipt, failure, archive binding, and individual paired daily difference.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audited_packet", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    # Only an auditor-produced packet is opened; never traverse the batch root.
    raw = args.audited_packet.read_bytes()
    packet = json.loads(raw)
    result = make_report(packet)
    result["audited_packet_file"] = {
        "path": str(args.audited_packet.resolve()),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    args.output_directory.mkdir(exist_ok=False)
    (args.output_directory / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (args.output_directory / "analysis.md").write_text(markdown(result))


if __name__ == "__main__":
    main()

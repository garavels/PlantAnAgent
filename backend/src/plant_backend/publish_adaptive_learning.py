"""Offline, deterministic projection of the already closed adaptive PPO record.

This command reads pinned reports, never a model, simulator or mutable run folder.
HTTP serves only the resulting portable package data through adaptive_learning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from plant_backend.adaptive_learning import (
    ACTOR_IDS,
    POLICY_IDS,
    PUBLIC_PATH,
    RESOURCES,
    SCHEMA,
    SEEDS,
    digest,
    validate,
)

REPORT_PINS = {
    "008": "9b4e86afe39148a9ad5f9877c8a2ddcdb5e135bbb104ba6043d51a76da56934c",
    "009": "285f53d1ae6f3a07a4b6c657071711b338c9ef206fa9a0575fc122f34ccb7f31",
}
AUDIT_PINS = {
    "008": "22aff1c0aa3b7c4a3b9bf78d6ec9a7825c14b88534dbe77537021bd66f6d9536",
    "009": "9fc2e4b6fb74f4252079f94903b34831875281cf62a621e10cec0dd024f69b31",
}
# These summaries describe the closed historical record, not comparable new arms.
HISTORY = (
    (
        "pilot",
        "Initial pilot",
        "completed",
        "ppo_pilot_912d9e6_diagnosis.json",
        "a24d7fb188f16bb1da4083e03a70a44bf9db68aa0e12e1cc02d11338053ff4c0",
        "One PI-Lagrangian policy completed training. It delivered less value than the initial policy and heuristic on all four development scenarios.",
        "No detached replay or strict safety qualification; a different four-scenario panel.",
    ),
    (
        "001",
        "Batch 001",
        "completed",
        "adaptive_rl_batch_001_results.json",
        "5fc9dd6a54602b9ff5357cd98646da003f1c2b0b88fcfa8df012421736760a9b",
        "Five policies completed 131,072 transitions each. Every PPO arm failed the strict safety and daily resource-return conditions in all eight scenarios.",
        "Earlier development panel; no policy qualified.",
    ),
    (
        "002",
        "Batch 002",
        "completed",
        "adaptive_rl_batch_002_results.json",
        "17ba1d16cf37adb1b79c7e8e552d0aec519f71ebe1773d3c13e8dc1bd99e2031",
        "Four policies completed 131,072 transitions each. Heuristic residuals produced modest gains over the heuristic, with forced stops and delivery/resource regressions.",
        "No demonstrated Operating MPC benefit; earlier development panel.",
    ),
    (
        "003",
        "Batch 003",
        "completed",
        "adaptive_rl_batch_003_results.json",
        "0d91e53afd4e704667850b517ce081af32fa9f83448a257e92901636dfca982e",
        "One MPC-residual scalar policy completed 32,768 transitions. Mean value improved by EUR 104.98 versus MPC; individual delivery and resource regressions prevented qualification.",
        "Eight earlier development scenarios; M24 and E-beta stopped before completing their weeks.",
    ),
    (
        "004",
        "Batch 004",
        "aborted",
        "adaptive_rl_batch_004_results.json",
        "43da491388c802597866703dc1030b0dacee30429e9c7de90a0b91f20dd9b017",
        "Two 65,536-transition runs were attempted. A forecast-interface failure stopped training before the first PPO update; no trained endpoints or evaluation.",
        "Planned transitions are not completed training.",
    ),
    (
        "005",
        "Batch 005",
        "incomplete",
        "adaptive_rl_batch_005_evidence_index.json",
        "a5737dbd8bd5b87216d71671142915c2c6514a16690c49898b275a5bc5462893",
        "Two training attempts were interrupted with storage loss. Three ledgers survived per arm; no final checkpoints or evaluation were admitted.",
        "Exact final transitions and optimizer updates are unknown.",
    ),
    (
        "006",
        "Batch 006",
        "aborted",
        "adaptive_rl_batch_006_evidence_index.json",
        "a8533ef871ec48da3395032b19cbe073ac164768c7a9852905afba2aa011b63c",
        "Both interrupted runs preserved 24,576 transitions and 960 updates. Host hibernation prevented completed endpoints and evaluation.",
        "Subsequent partial progress is unknown; neither run is a completed policy.",
    ),
    (
        "007",
        "Batch 007",
        "aborted",
        "ADAPTIVE_RL_BATCH_007_RESULTS.md",
        "81505c89ef6468aeee121f4bc5afad9a2e5f6aa1d34050fafe4a819f5d509eb3",
        "Scalar and PI policies each completed 65,536 transitions and 2,560 updates. A power-guard false positive aborted replay while on AC power; the campaign's performance was not admitted.",
        "A separate allocated-byte metadata mismatch also prevented admission. The two exact endpoints were later admitted and evaluated in Batch 008.",
    ),
    (
        "008",
        "Batch 008",
        "completed",
        "adaptive_rl_batch_008_results.json",
        "bfb21fa42522b9e3416bb5f6e861af1fea53d07c0365efaaa19da7ec19c9e00a",
        "The two Batch 007 endpoints were admitted without new training. Both passed hard safety in eight scenarios but returned lower mean value than Operating MPC, with delivery/resource regressions.",
        "Development evidence only; 40 complete weeks and 16 stopped prefixes across seven actors.",
    ),
    (
        "009",
        "Batch 009",
        "completed",
        "adaptive_rl_batch_009_results.json",
        "1b6df673ca1ac3df03f16a11d01f454251b75f8f9b5a1efed92f27de9058d217",
        "Two scalar policies completed 16,384 transitions each. Mean value improved by EUR 143.87 and EUR 145.48 versus Operating MPC, with individual delivery/resource regressions.",
        "Eight reused development scenarios and seven reused Batch 008 actors; neither policy qualified or was promoted.",
    ),
)
LABELS = {
    "H": "Heuristic",
    "OperatingMPC": "Operating MPC",
    "M24": "M24",
    "E-beta": "E-beta",
    "initial-mpc-residual": "Initial residual",
    "A-mpc-residual-scalar": "Scalar PPO · 65k",
    "B-mpc-residual-pi": "PI-Lagrangian PPO · 65k",
    "A-scalar-gamma0995": "Scalar PPO · gamma 0.995",
    "B-scalar-gamma1000": "Scalar PPO · gamma 1.000",
}


def read_pinned(path: Path, sha256: str, identifier: str) -> tuple[Any, dict[str, Any]]:
    """Reject changed evidence before using even one of its numerical outcomes."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f"source digest mismatch: {identifier}")
    artifact = {"id": identifier, "sha256": sha256, "bytes": len(raw)}
    return (json.loads(raw) if path.suffix == ".json" else None), artifact


def project_rollouts(training: dict[str, Any], *, older: bool) -> list[dict[str, Any]]:
    diagnostics = {
        row["num_timesteps"]: row
        for row in training["diagnostic_rows" if older else "diagnostics"]
    }
    result = []
    transitions = 0
    for row in training["optimizer_rows" if older else "rollouts"]:
        transitions += row["transition_count"]
        diagnostic = diagnostics.get(transitions)
        if diagnostic is None:
            mean_reward = explained = value_loss = None
        elif older:
            mean_reward = diagnostic["full_rollout_training_statistics"]["reward"][
                "mean"
            ]
            explained = diagnostic["reward_explained_variance"]
            value_loss = diagnostic["critic_mse_optimizer_units_squared"]
        else:
            mean_reward = diagnostic["full_rollout"]["reward"]["mean"]
            explained = diagnostic["full_rollout"]["reward_explained_variance"]
            value_loss = diagnostic["gradient_attribution"]["reward_value_mse"]
        result.append(
            {
                "index": row["rollout_index"],
                "transitions": transitions,
                "reward_sum": row["optimizer_reward_sum" if older else "reward_return"],
                "completed_episodes": row["completed_episode_count"],
                "complete_cycles": row["complete_cycle_count"],
                "training_mean_reward": mean_reward,
                "explained_variance": explained,
                "value_loss": value_loss,
                "approx_kl": row["approximate_kl"],
                "clip_fraction": row[
                    "ppo_ratio_clip_fraction" if older else "clip_fraction"
                ],
                "mean_episode_cost": row["mean_completed_episode_cost"],
                "multipliers": row["learner_multipliers"] if older else None,
            }
        )
    return result


def project_failures(failures: list[Any]) -> list[str]:
    result = []
    for failure in failures:
        if isinstance(failure, str):
            result.append(failure)
        elif set(failure) == {"kind", "detail"} and set(failure["detail"]) == {
            "type",
            "phase",
            "message",
        }:
            detail = failure["detail"]
            result.append(
                f"{failure['kind']} / {detail['phase']} / {detail['type']}: {detail['message']}"
            )
        else:
            result.append(
                json.dumps(failure, sort_keys=True, ensure_ascii=True, allow_nan=False)
            )
    return result


def project_actor(identifier: str, actor: dict[str, Any]) -> dict[str, Any]:
    cases = []
    for slot in actor["slots"]:
        seed, summary = slot["seed"], slot["summary"]
        complete = slot["eligible_for_descriptive_comparison"]
        metrics = summary["economic_metrics"]
        mpc_modes = None
        if identifier in ("OperatingMPC", "initial-mpc-residual", *POLICY_IDS):
            recovery, fallback = (
                summary["recovery_decisions"],
                summary["fallback_decisions"],
            )
            mpc_modes = {
                "scheduled": summary["decisions"] - recovery - fallback,
                "recovery": recovery,
                "fallback": fallback,
            }
        cases.append(
            {
                "seed": seed,
                "status": "completed" if complete else "stopped",
                "decisions": summary["decisions"],
                "failures": project_failures(slot["failures"]),
                "net_operating_value_eur": (
                    metrics["net_operating_value_eur"] if complete else None
                ),
                "delivered_kg": metrics["delivered_kg"] if complete else None,
                "battery_throughput_kwh": (
                    metrics["battery_throughput_kwh"] if complete else None
                ),
                "hard_safety_passed": (
                    summary["hard_safety_passed"] if complete else None
                ),
                "prefix_hard_safety_passed": summary["hard_safety_passed"],
                "all_daily_returns_passed": (
                    summary["all_cycles_feasible"] if complete else None
                ),
                "mpc_modes": mpc_modes,
                "deliveries": [
                    {
                        key: row[key]
                        for key in ("ordinal", "delivered_kg", "requested_kg")
                    }
                    for row in actor["all_224_delivery_slots"]
                    if row["seed"] == seed and row["recorded"]
                ],
                "daily_resources": [
                    {"cycle": row["cycle"], "deficits": row["deficits"]}
                    for row in actor["all_56_daily_resource_slots"]
                    if row["seed"] == seed and row["recorded"]
                ],
            }
        )
    kind = "ppo" if identifier in POLICY_IDS else "mpc"
    if identifier == "H":
        kind = "heuristic"
    elif identifier == "initial-mpc-residual":
        kind = "initial"
    return {
        "id": identifier,
        "label": LABELS[identifier],
        "kind": kind,
        "evidence_origin": actor["evidence_origin"],
        "cases": cases,
    }


def project_comparison(pair: dict[str, Any], batch: str) -> dict[str, Any]:
    cases = []
    for slot in pair["all_eight_pair_slots"]:
        seed, available = slot["seed"], slot["available"]
        metrics = slot.get("metric_differences") or {}
        deliveries = (pair["delivery_differences"] or {}).get("all_224_commitments", [])
        resources = (pair["daily_resource_differences"] or {}).get(
            "all_56_boundaries", []
        )
        cases.append(
            {
                "seed": seed,
                "available": available,
                "candidate_failures": project_failures(slot["candidate_failures"]),
                "comparator_failures": project_failures(slot["comparator_failures"]),
                **{
                    f"delta_{key}": metrics[key] if available else None
                    for key in (
                        "net_operating_value_eur",
                        "delivered_kg",
                        "battery_throughput_kwh",
                    )
                },
                "delivery_regressions": [
                    {
                        "ordinal": row["ordinal"],
                        "difference_kg": row["delivered_kg_difference"],
                    }
                    for row in deliveries
                    if available
                    and row["seed"] == seed
                    and row["delivered_kg_difference"] < 0
                ],
                "resource_regressions": [
                    {
                        "cycle": row["cycle"],
                        "resource": resource,
                        "difference": row["candidate_minus_comparator"][resource],
                    }
                    for row in resources
                    for resource in RESOURCES
                    if available
                    and row["seed"] == seed
                    and row["candidate_minus_comparator"][resource] > 0
                ],
            }
        )
    return {
        "candidate": pair["candidate"],
        "comparator": pair["comparator"],
        "evidence_batch": batch,
        "cases": cases,
    }


def export(research_root: Path, results_root: Path) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    report_artifacts: dict[str, Any] = {}
    artifacts = []
    for batch in ("008", "009"):
        directory = research_root / f"adaptive-rl-batch-{batch}-analysis"
        report, artifact = read_pinned(
            directory / "report/analysis.json",
            REPORT_PINS[batch],
            f"batch{batch}/analysis.json",
        )
        reports[batch], report_artifacts[batch] = report, artifact
        artifacts.append(artifact)
        audit, artifact = read_pinned(
            directory / "audit.json", AUDIT_PINS[batch], f"batch{batch}/audit.json"
        )
        if audit["integrity_verified"] is not True or audit["batch_closed"] is not True:
            raise ValueError("source campaign has not closed with verified integrity")
        artifacts.append(artifact)
    history = []
    compact: dict[str, Any] = {}
    for identifier, label, status, filename, sha256, summary, limitation in HISTORY:
        data, artifact = read_pinned(
            results_root / filename, sha256, f"history/{filename}"
        )
        compact[identifier] = data
        artifacts.append(artifact)
        history.append(
            {
                "id": identifier,
                "label": label,
                "status": status,
                "summary": summary,
                "policy_ids": list(
                    POLICY_IDS[:2]
                    if identifier in ("007", "008")
                    else POLICY_IDS[2:] if identifier == "009" else ()
                ),
                "source": artifact,
                "limitations": [limitation],
            }
        )
    validation_path = results_root / "adaptive_rl_batch_009_validation.json"
    # The compact validation itself binds the corrected publication and preserves
    # the independent closed-record verification; source reports remain unchanged.
    verification = json.loads(validation_path.read_bytes())
    if (
        verification["passed"] is not True
        or verification["closed_evidence_verification"]["passed"] is not True
    ):
        raise ValueError("independent publication verification has not passed")
    # Pin this separately so a changed boolean cannot assert successful verification.
    _, validation_artifact = read_pinned(
        validation_path, VALIDATION_SHA256, "batch009/publication-validation.json"
    )
    artifacts.append(validation_artifact)
    latest = compact["009"]
    policies = []
    for identifier in POLICY_IDS:
        older = identifier in POLICY_IDS[:2]
        batch = "008" if older else "009"
        training = reports[batch]["training"][identifier]
        if training["closed_successfully"] is not True:
            raise ValueError("incomplete training endpoint")
        checkpoint = (
            next(
                record
                for record in latest["historical_reference"]["older_final_checkpoints"]
                if Path(record["path"]).parent.name == identifier
            )
            if older
            else latest["training"][identifier]["final_checkpoint"]
        )
        checkpoint_artifact = {
            "id": f"batch{'007' if older else '009'}/{identifier}/final-policy.zip",
            "sha256": checkpoint["sha256"],
            "bytes": checkpoint["bytes"],
        }
        artifacts.append(checkpoint_artifact)
        policies.append(
            {
                "id": identifier,
                "label": LABELS[identifier],
                "campaign_id": "007" if older else "009",
                "algorithm": (
                    "PI-Lagrangian-PPO"
                    if identifier == "B-mpc-residual-pi"
                    else "Scalar-PPO"
                ),
                "optimizer_seed": 101,
                "training_status": "completed",
                "evaluation_status": "evaluated",
                "operational_status": "experimental",
                "transitions": training[
                    "fixed_transition_budget" if older else "total_timesteps"
                ],
                "minibatch_updates": training[
                    "fixed_optimizer_step_budget" if older else "optimizer_steps"
                ],
                "reward_gamma": 0.995 if older else training["reward_gamma"],
                "gae_lambda": 0.95,
                "reward_scale": training["reward_scale"],
                "checkpoint": checkpoint_artifact,
                "training_source": report_artifacts[batch],
                "rollouts": project_rollouts(training, older=older),
                "limitations": [
                    "Experimental MPC-residual policy: 139 inputs and four residual actions; requires the causal MPC prior.",
                    "One optimizer seed and eight reused development scenarios; no held-out confirmation or operational promotion.",
                    "Recorded rollout statistics and pre-update diagnostics do not establish convergence or fixed-scenario improvement during training.",
                ],
            }
        )
    payload = {
        "schema_version": SCHEMA,
        "source": {
            key: reports["009"]["source"][key] for key in ("git_head", "git_tree")
        },
        "provenance": {
            "exported_from_closed_evidence": True,
            "independent_verification_passed": True,
            "development_only": True,
            "artifacts": artifacts,
        },
        "task": {
            "hours": 168,
            "decisions": 672,
            "deliveries": 28,
            "requested_methane_kg": 6720,
            "development_seeds": SEEDS,
        },
        "policies": policies,
        "actors": [
            project_actor(identifier, reports["009"]["actors"][identifier])
            for identifier in ACTOR_IDS
        ],
        "comparisons": [
            project_comparison(pair, batch)
            for batch in ("008", "009")
            for pair in reports[batch]["operational_comparisons"]
        ],
        "history": history,
        "registered_study": {
            "status": "pending",
            "families": 3,
            "optimizer_seeds": 5,
            "policies": 15,
            "protected_seeds_opened": False,
        },
        "limitations": [
            "Development evidence only. No registered-study selection, held-out confirmation, commercial savings or site-generalization claim.",
            "All four evaluated trained policies pass hard safety in eight completed weeks, but fail daily resource-return conditions and worsen individual deliveries or resources.",
            "Recovery uses an accepted optimized MPC plan. Only Fallback uses the heuristic; zero Scheduled decisions does not mean zero optimization.",
            "Stopped M24 and E-beta prefixes have no complete-week value or safety result. Observed prefix safety is retained separately.",
            "Batch 009 reuses seven Batch 008 actor records. Earlier history uses different scenarios and is not pooled into matched comparisons.",
            "Reward curves are scaled immediate training reward, not evaluation profit. Critic diagnostics are recorded before each update; no interpolation creates additional evidence.",
        ],
    }
    payload["sha256"] = digest(payload)
    return validate(payload)


# The corrected publication at revision90f9b43 is the authoritative validation.
VALIDATION_SHA256 = "19a5a21ca3842d779a510295745967692513d93e9e0c30efb8fa971beccd969a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PUBLIC_PATH)
    args = parser.parse_args()
    evidence = export(args.research_root, args.results_root)
    raw = (
        json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    if args.output.is_symlink():
        raise ValueError("output must not be a symlink")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A concurrent reader sees either complete version, never a partial rewrite.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            temporary.replace(args.output)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Published {len(raw.encode())} bytes; evidence SHA256 {evidence['sha256']}")


if __name__ == "__main__":
    main()

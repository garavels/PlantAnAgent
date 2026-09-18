"""Build the compact retained report for the frozen feasible-planner matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any

from plant_backend.rl.v2.planner_schema import (
    EPISODE_CONFIG_SHA256_BY_RATE,
    canonical_json,
    canonical_payload,
    canonical_payload_from_dict,
    canonical_payload_preimage,
    planner_artifact_from_dict,
)


CELLS = ((35, 0), (35, 1), (35, 2), (42, 0), (42, 1), (42, 2))
ENDOWMENT_DOMAIN = "planner-endowment-diagnostic-artifact-v1"
REPORT_DOMAIN = "planner-development-report-v1"
REPAIRED_REVISION = "564002cd9a49761b7eb78a84424d5325e5b240c6"
FAILED_REVISION = "0059626313866d7a2bda2a44f64926a757a9f5d6"
ENDOWMENT_DRIVER_SHA256 = (
    "c98068b8deb380d04fe2e8817c79aa7313494062341068bb7e60886bd09eb3ea"
)
FAILED_ENDOWMENT_DRIVER_SHA256 = (
    "41ee77760fef6fdb40628fcdb09084b87d8638ca771a0a224d04819e9c2c6159"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _domain_sha256(domain: str, value: object) -> str:
    return hashlib.sha256(
        canonical_payload_preimage(domain, canonical_json(value))
    ).hexdigest()


def _payload_value(value: object) -> object:
    payload = canonical_payload_from_dict(value)
    return payload.value()


def _source_value(value: object) -> dict[str, object]:
    parsed = _payload_value(value)
    if not isinstance(parsed, dict):
        raise TypeError("source snapshot is not an object")
    return parsed


def _require_clean_revision(value: object, revision: str) -> dict[str, object]:
    source = _source_value(value)
    if source.get("git_revision") != revision:
        raise ValueError("artifact source revision is not the declared freeze")
    if source.get("git_dirty") is not False:
        raise ValueError("artifact source is dirty")
    if source.get("untracked_files") != []:
        raise ValueError("artifact source has untracked files")
    if source.get("identity_complete") is not True:
        raise ValueError("artifact source identity is incomplete")
    return source


def _compact_payload(value: object) -> dict[str, object]:
    payload = canonical_payload_from_dict(value)
    return {
        "domain": payload.domain,
        "sha256": payload.sha256,
        "value": payload.value(),
    }


def _compact_replay(
    replay: dict[str, Any], *, include_actions: bool
) -> dict[str, object]:
    metrics = _compact_payload(replay["metrics"])
    terminal_truth = _compact_payload(replay["terminal_truth"])
    terminal_telemetry = _compact_payload(replay["terminal_telemetry"])
    delivery_audit = _compact_payload(replay["delivery_audit"])
    result: dict[str, object] = {
        "role": replay["role"],
        "manifest_sha256": replay["manifest_sha256"],
        "exogenous_episode_sha256": replay["exogenous_episode_sha256"],
        "checkpoint_contract_digest": replay["checkpoint_contract_digest"],
        "decision_count": replay["decision_count"],
        "physics_steps": replay["physics_steps"],
        "terminated": replay["terminated"],
        "truncated": replay["truncated"],
        "battery_throughput_kwh": replay["battery_throughput_kwh"],
        "replay_elapsed_ns": replay["replay_elapsed_ns"],
        "final_score": replay["final_score"],
        "metrics": metrics,
        "terminal_truth": terminal_truth,
        "terminal_telemetry": terminal_telemetry,
        "delivery_audit": delivery_audit,
        "request_trace_sha256": replay["request_trace"]["sha256"],
        "source_trace_sha256": replay["source_trace"]["sha256"],
        "action_record_trace_sha256": replay["action_record_trace"]["sha256"],
        "generation_outcome_sha256": replay["generation_outcome"]["sha256"],
        "replay_outcome_sha256": replay["replay_outcome"]["sha256"],
        "replay_binding_sha256": replay["replay_binding"]["sha256"],
    }
    if include_actions:
        result["selected_action_records"] = replay["action_records"]
    return result


def _validate_endowment(
    path: Path,
    *,
    rate: int,
    seed: int,
    planner_source: dict[str, object],
    lock_sha256: str,
    runtime: object,
    core: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    raw: dict[str, Any] = json.loads(path.read_text("utf-8"))
    artifact_sha256 = raw.pop("artifact_sha256")
    if _domain_sha256(ENDOWMENT_DOMAIN, raw) != artifact_sha256:
        raise ValueError("endowment artifact digest mismatch")
    if raw["delivery_rate_kg_h"] != float(rate) or raw["seed"] != seed:
        raise ValueError("endowment artifact cell mismatch")
    if raw["driver_sha256"] != ENDOWMENT_DRIVER_SHA256:
        raise ValueError("endowment driver identity mismatch")
    start = _require_clean_revision(raw["start_source"], REPAIRED_REVISION)
    end = _require_clean_revision(raw["end_source"], REPAIRED_REVISION)
    if start != end or start != planner_source:
        raise ValueError("endowment and planner source identities differ")
    if raw["start_lock_sha256"] != raw["end_lock_sha256"]:
        raise ValueError("endowment dependency lock changed during cell")
    if raw["start_lock_sha256"] != lock_sha256:
        raise ValueError("endowment and planner dependency locks differ")
    runtime_payload = canonical_payload_from_dict(raw["runtime_versions"])
    planner_runtime = canonical_payload_from_dict(runtime)
    if runtime_payload != planner_runtime:
        raise ValueError("endowment and planner runtime identities differ")

    result: dict[str, Any] = raw["result"]
    deterministic = dict(result)
    deterministic.pop("elapsed_s")
    result_payload = canonical_payload("planner-relaxation-result-v1", deterministic)
    if result_payload.sha256 != raw["deterministic_result_sha256"]:
        raise ValueError("endowment result digest mismatch")
    required = {
        "success": True,
        "status": 0,
        "failure_code": None,
        "canonical_episode_verified": True,
        "preserve_initial_endowment": True,
        "is_task_upper_bound": False,
        "checkpoint_contract_digest": core["checkpoint_contract_digest"],
        "exogenous_episode_sha256": core["exogenous_episode_sha256"],
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise ValueError(f"endowment result {key} is not cross-bound")
    if not math.isclose(
        result["requested_kg"],
        core["comparison"]["upper_bound_kg"],
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("endowment requested mass is not cross-bound")
    witness = result.get("witness")
    if not isinstance(witness, dict):
        raise ValueError("endowment result has no witness")
    witness_sha256 = canonical_payload("planner-relaxation-witness-v1", witness).sha256
    return raw, witness_sha256


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty distribution")
    return {
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, object]:
    def values(name: str) -> list[float]:
        return [float(row["values"][name]) for row in rows]

    fractions = [
        float(row["values"]["fraction_closed"])
        for row in rows
        if row["values"]["fraction_closed"] is not None
    ]
    requested = values("requested_kg")
    endowment = values("endowment_preserving_relaxed_delivery_kg")
    a0_scores = [row["a0"]["final_score"] for row in rows]
    feasible_scores = [row["best_known_feasible"]["final_score"] for row in rows]
    return {
        "cells": len(rows),
        "delivery_kg": {
            name: _distribution(values(name))
            for name in (
                "a0_delivered_kg",
                "best_known_feasible_delivered_kg",
                "dual_certified_relaxation_ceiling_kg",
                "endowment_preserving_relaxed_delivery_kg",
                "exact_planning_gain_kg",
                "residual_relaxed_gap_kg",
            )
        },
        "fraction_closed": (
            None
            if not fractions
            else {"eligible_cells": len(fractions), **_distribution(fractions)}
        ),
        "endowment_preserving_fill": _distribution(
            [value / total for value, total in zip(endowment, requested, strict=True)]
        ),
        "positive_exact_planning_gain_cells": sum(
            row["values"]["exact_planning_gain_kg"] > 0.0 for row in rows
        ),
        "feasible_reaches_relaxed_ceiling_cells": sum(
            row["values"]["residual_relaxed_gap_kg"] == 0.0 for row in rows
        ),
        "endowment_preserving_full_service_cells": sum(
            math.isclose(value, total, rel_tol=0.0, abs_tol=1e-9)
            for value, total in zip(endowment, requested, strict=True)
        ),
        "a0_commitment_service": {
            "met": sum(score["service_numerator"] for score in a0_scores),
            "settled": sum(score["service_denominator"] for score in a0_scores),
        },
        "feasible_commitment_service": {
            "met": sum(score["service_numerator"] for score in feasible_scores),
            "settled": sum(score["service_denominator"] for score in feasible_scores),
        },
        "search_elapsed_s": _distribution(
            [row["search"]["elapsed_ns"] / 1_000_000_000 for row in rows]
        ),
        "search_peak_rss_bytes": _distribution(
            [float(row["search"]["peak_rss_bytes"]) for row in rows]
        ),
    }


def _planner_cell(
    *,
    planner_dir: Path,
    rate: int,
    seed: int,
) -> dict[str, Any]:
    planner_path = planner_dir / f"rate{rate}-seed{seed}.json"
    raw: dict[str, Any] = json.loads(planner_path.read_text("utf-8"))
    artifact = planner_artifact_from_dict(raw)
    if not artifact.success or not artifact.evidence_ready:
        raise ValueError("registered planner artifact is not evidence-ready")
    if artifact.runner_failure is not None or artifact.core_result is None:
        raise ValueError("registered planner artifact has no successful core")
    core: dict[str, Any] = raw["core_result"]
    expected_episode_sha = dict(EPISODE_CONFIG_SHA256_BY_RATE)[float(rate)]
    if (
        core["seed"] != seed
        or core["settings"]["episode_config_sha256"] != expected_episode_sha
    ):
        raise ValueError("registered planner artifact cell mismatch")
    start = _require_clean_revision(raw["start_source"], REPAIRED_REVISION)
    end = _require_clean_revision(raw["end_source"], REPAIRED_REVISION)
    if start != end:
        raise ValueError("planner source changed during cell")
    if raw["start_lock_sha256"] != raw["end_lock_sha256"]:
        raise ValueError("planner dependency lock changed during cell")

    endowment_path = planner_dir / f"endowment-rate{rate}-seed{seed}.json"
    endowment, endowment_witness_sha256 = _validate_endowment(
        endowment_path,
        rate=rate,
        seed=seed,
        planner_source=start,
        lock_sha256=raw["start_lock_sha256"],
        runtime=raw["runtime_versions"],
        core=core,
    )
    task_relaxation = _payload_value(core["relaxation_result"])
    if not isinstance(task_relaxation, dict):
        raise TypeError("task relaxation is not an object")
    comparison = core["comparison"]
    endowment_result = endowment["result"]
    values = {
        "requested_kg": task_relaxation["requested_kg"],
        "a0_delivered_kg": comparison["a0_delivered_kg"],
        "best_known_feasible_delivered_kg": comparison["selected_delivered_kg"],
        "dual_certified_relaxation_ceiling_kg": comparison["upper_bound_kg"],
        "endowment_preserving_relaxed_delivery_kg": endowment_result[
            "dual_certified_delivery_upper_bound_kg"
        ],
        "exact_planning_gain_kg": comparison["gain_kg"],
        "residual_relaxed_gap_kg": comparison["residual_gap_kg"],
        "fraction_closed": comparison["fraction_closed"],
        "bracket_status": comparison["bracket_status"],
    }
    selected = core["searched_finalist"]
    if (
        core["selected_role"] != "searched_finalist"
        or comparison["selected_role"] != "searched_finalist"
    ):
        raise ValueError(
            "report builder only admits the observed searched-finalist selection"
        )
    search = dict(core["search"])
    counters = _compact_payload(search.pop("deterministic_counters"))
    search["deterministic_counters"] = counters
    return {
        "delivery_rate_kg_h": float(rate),
        "seed": seed,
        "values": values,
        "identities": {
            "manifest_sha256": core["manifest_sha256"],
            "exogenous_episode_sha256": core["exogenous_episode_sha256"],
            "checkpoint_contract_digest": core["checkpoint_contract_digest"],
            "settings_sha256": core["settings"]["settings_sha256"],
            "action_library_sha256": core["settings"]["action_library"]["sha256"],
            "a0_controller_sha256": core["settings"]["a0_controller"]["sha256"],
            "task_relaxation_model_sha256": core["relaxation_model_sha256"],
            "task_relaxation_result_sha256": core["relaxation_result_sha256"],
            "task_relaxation_witness_sha256": core["relaxation_witness"]["sha256"],
            "relaxation_guidance_sha256": core["settings"][
                "relaxation_guidance_sha256"
            ],
            "deterministic_planner_result_sha256": core["deterministic_result_sha256"],
            "selected_action_record_trace_sha256": selected["action_record_trace"][
                "sha256"
            ],
            "endowment_relaxation_model_sha256": endowment_result["model_sha256"],
            "endowment_relaxation_result_sha256": endowment[
                "deterministic_result_sha256"
            ],
            "endowment_relaxation_witness_sha256": endowment_witness_sha256,
        },
        "planner_artifact": {
            "filename": planner_path.name,
            "bytes": planner_path.stat().st_size,
            "raw_file_sha256": _sha256(planner_path),
            "artifact_sha256": raw["artifact_sha256"],
            "core_result_sha256": raw["core_result_sha256"],
            "evidence_ready": raw["evidence_ready"],
            "start_source_sha256": raw["start_source"]["sha256"],
            "end_source_sha256": raw["end_source"]["sha256"],
            "start_lock_sha256": raw["start_lock_sha256"],
            "end_lock_sha256": raw["end_lock_sha256"],
            "runtime_versions_sha256": raw["runtime_versions"]["sha256"],
            "source_boundary_claim": raw["source_boundary_claim"],
        },
        "endowment_artifact": {
            "filename": endowment_path.name,
            "bytes": endowment_path.stat().st_size,
            "raw_file_sha256": _sha256(endowment_path),
            "artifact_sha256": json.loads(endowment_path.read_text("utf-8"))[
                "artifact_sha256"
            ],
            "deterministic_result_sha256": endowment["deterministic_result_sha256"],
            "start_source_sha256": endowment["start_source"]["sha256"],
            "end_source_sha256": endowment["end_source"]["sha256"],
            "start_lock_sha256": endowment["start_lock_sha256"],
            "end_lock_sha256": endowment["end_lock_sha256"],
            "runtime_versions_sha256": endowment["runtime_versions"]["sha256"],
        },
        "a0": _compact_replay(core["a0_incumbent"], include_actions=False),
        "best_known_feasible": _compact_replay(selected, include_actions=True),
        "comparison": comparison,
        "task_relaxation": {
            key: task_relaxation[key]
            for key in (
                "success",
                "status",
                "message",
                "solver",
                "solver_version",
                "solve_settings",
                "model_sha256",
                "checkpoint_contract_digest",
                "exogenous_episode_sha256",
                "canonical_episode_verified",
                "preserve_initial_endowment",
                "is_task_upper_bound",
                "requested_kg",
                "numerical_primal_delivery_kg",
                "dual_certified_delivery_upper_bound_kg",
                "relaxed_fill_ceiling",
                "primal_dual_gap_kg",
                "iterations",
                "marginal_projection_within_tolerance",
                "certificate",
                "certificate_scope",
                "simulator_embedding_status",
            )
        },
        "endowment_preserving_relaxation": {
            key: endowment_result[key]
            for key in (
                "success",
                "status",
                "message",
                "solver",
                "solver_version",
                "solve_settings",
                "model_sha256",
                "checkpoint_contract_digest",
                "exogenous_episode_sha256",
                "canonical_episode_verified",
                "preserve_initial_endowment",
                "is_task_upper_bound",
                "requested_kg",
                "numerical_primal_delivery_kg",
                "dual_certified_delivery_upper_bound_kg",
                "relaxed_fill_ceiling",
                "primal_dual_gap_kg",
                "iterations",
                "elapsed_s",
                "marginal_projection_within_tolerance",
                "certificate",
                "certificate_scope",
                "simulator_embedding_status",
            )
        },
        "search": search,
    }


def _failed_invocation(path: Path) -> dict[str, object]:
    raw: dict[str, Any] = json.loads(path.read_text("utf-8"))
    artifact = planner_artifact_from_dict(raw)
    if artifact.success or artifact.evidence_ready or artifact.core_result is None:
        raise ValueError("original failed artifact no longer has its declared state")
    core = raw["core_result"]
    start = _require_clean_revision(raw["start_source"], FAILED_REVISION)
    end = _require_clean_revision(raw["end_source"], FAILED_REVISION)
    if start != end:
        raise ValueError("failed invocation source changed during cell")
    return {
        "status": "retained_failed_invocation_before_repair",
        "cell": {"delivery_rate_kg_h": 35.0, "seed": 0},
        "source_revision": FAILED_REVISION,
        "artifact_sha256": raw["artifact_sha256"],
        "raw_file_sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "runner_failure": raw["runner_failure"],
        "core_failure": core["failure"],
        "search": core["search"],
        "note": (
            "This artifact is not part of the repaired six-cell result and is never "
            "relabelled, overwritten, or omitted from the research history."
        ),
    }


def _write_atomic(path: Path, value: object) -> None:
    serialized = (
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner-dir", type=Path, required=True)
    parser.add_argument("--failed-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().exists():
        raise FileExistsError(f"refusing to overwrite {args.output.resolve()}")
    builder_sha256 = _sha256(Path(__file__))
    rows = [
        _planner_cell(planner_dir=args.planner_dir.resolve(), rate=rate, seed=seed)
        for rate, seed in CELLS
    ]
    first_planner = json.loads(
        (args.planner_dir.resolve() / "rate35-seed0.json").read_text("utf-8")
    )
    source = _require_clean_revision(first_planner["start_source"], REPAIRED_REVISION)
    lock_sha256 = first_planner["start_lock_sha256"]
    runtime = _compact_payload(first_planner["runtime_versions"])
    source_sha256 = first_planner["start_source"]["sha256"]
    runtime_sha256 = first_planner["runtime_versions"]["sha256"]
    for row in rows:
        if (
            row["planner_artifact"]["core_result_sha256"]
            != row["identities"]["deterministic_planner_result_sha256"]
        ):
            raise ValueError("planner result identity mismatch")
        for artifact_name in ("planner_artifact", "endowment_artifact"):
            retained_artifact = row[artifact_name]
            if (
                retained_artifact["start_source_sha256"] != source_sha256
                or retained_artifact["end_source_sha256"] != source_sha256
                or retained_artifact["start_lock_sha256"] != lock_sha256
                or retained_artifact["end_lock_sha256"] != lock_sha256
                or retained_artifact["runtime_versions_sha256"] != runtime_sha256
            ):
                raise ValueError("matrix provenance identities are not common")
    body: dict[str, object] = {
        "schema_version": 1,
        "study": "v2_feasible_future_information_planner_development_matrix",
        "status": "complete_evidence_ready",
        "evidence_scope": "privileged_offline_development",
        "disclaimer": (
            "Six already-inspected synthetic development cells. The planner and both "
            "relaxations use perfect future information and are non-deployable. Results "
            "are engineering evidence only: not held-out performance, field evidence, "
            "financial return, RL superiority, global optimality, or site readiness."
        ),
        "matrix": {
            "order": [
                {"delivery_rate_kg_h": float(rate), "seed": seed}
                for rate, seed in CELLS
            ],
            "horizon_h": 72.0,
            "decision_interval_s": 900.0,
            "physics_interval_s": 60.0,
            "statistics": "population standard deviation (ddof=0); descriptive only",
        },
        "provenance": {
            "repaired_execution_revision": REPAIRED_REVISION,
            "source_snapshot": source,
            "dependency_lock_sha256": lock_sha256,
            "runtime_versions": runtime,
            "report_builder_sha256": builder_sha256,
            "endowment_driver_sha256": ENDOWMENT_DRIVER_SHA256,
            "artifact_custody_at_reporting": str(args.planner_dir.resolve()),
        },
        "instrumentation_incidents": [
            {
                "phase": "endowment_artifact_assembly",
                "affected_cell": {"delivery_rate_kg_h": 35.0, "seed": 0},
                "driver_sha256": FAILED_ENDOWMENT_DRIVER_SHA256,
                "failure": (
                    "The first endowment solve returned to the external wrapper, which "
                    "then rejected its unregistered custom CanonicalPayloadV1 domain. "
                    "No artifact or numeric result was published; the remaining five "
                    "solves were not started."
                ),
                "remedy": (
                    "The wrapper alone was changed to hash its custom domain through the "
                    "public preimage rule. A new driver identity was declared, and all six "
                    "diagnostics were executed without selecting on the unseen value."
                ),
            }
        ],
        "original_failed_invocation": _failed_invocation(
            args.failed_artifact.resolve()
        ),
        "common_identities": {
            "protocol_anchor_commit": first_planner["core_result"]["settings"][
                "protocol_anchor_commit"
            ],
            "protocol_blob_sha256": first_planner["core_result"]["settings"][
                "protocol_blob_sha256"
            ],
            "plant_config_sha256": first_planner["core_result"]["settings"][
                "plant_config_sha256"
            ],
            "constraints_sha256": first_planner["core_result"]["settings"][
                "constraints_sha256"
            ],
            "reward_weights_sha256": first_planner["core_result"]["settings"][
                "reward_weights_sha256"
            ],
            "action_library": _compact_payload(
                first_planner["core_result"]["settings"]["action_library"]
            ),
            "a0_controller": _compact_payload(
                first_planner["core_result"]["settings"]["a0_controller"]
            ),
            "planner_kind": first_planner["core_result"]["planner_kind"],
            "optimality_claim": first_planner["core_result"]["optimality_claim"],
            "privileged_information": first_planner["core_result"][
                "privileged_information"
            ],
        },
        "summary_by_rate": {
            str(rate): _summarize(
                [row for row in rows if row["delivery_rate_kg_h"] == float(rate)]
            )
            for rate in (35, 42)
        },
        "cells": rows,
    }
    report_sha256 = _domain_sha256(REPORT_DOMAIN, body)
    report = {**body, "report_sha256": report_sha256}
    _write_atomic(args.output.resolve(), report)
    reopened = json.loads(args.output.resolve().read_text("utf-8"))
    observed = reopened.pop("report_sha256")
    if observed != _domain_sha256(REPORT_DOMAIN, reopened):
        raise RuntimeError("retained report did not revalidate after publication")
    print(
        f"report={args.output.resolve()}; sha256={report_sha256}; "
        f"raw_sha256={_sha256(args.output.resolve())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

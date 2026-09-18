"""One-shot runner for the frozen absolute-setpoint MPC mechanism cell.

This module deliberately exposes only an external output path.  The episode,
controller, seed, protocol identities, gates, and no-retry claim boundary are
fixed before the first full setpoint trajectory is opened.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from plant_sim import PlantConfig

from ...controllers.base import Constraints
from .config import EpisodeConfigV2
from .env import PlantEnvV2
from .evaluation import collect_provenance
from .mpc_execution import public_observation_sha256
from .mpc_experiment import find_repository_root, validate_artifact_destination
from .mpc_schema import (
    DEFAULT_SERVICE_MPC_SETPOINT_SETTINGS_V1,
    SERVICE_MPC_SETPOINT_CONTROLLER_ID,
    ServiceMPCSetpointSettingsV1,
    canonical_json,
    domain_sha256,
    require_sha256,
)
from .mpc_setpoint_controller import ServiceMPCSetpointControllerV1
from .mpc_setpoint_decision import MPCSetpointDecisionEvidenceV1
from .mpc_setpoint_execution import MPCSetpointIntervalExecutionV1
from .mpc_setpoint_run import (
    SETPOINT_TRACKING_TOLERANCE_KW,
    build_mpc_setpoint_run_evidence,
    verify_mpc_setpoint_run_digests,
)
from .reward import RewardWeightsV2


SETPOINT_MECHANISM_FORMAT = "service-mpc-setpoint-mechanism-cell-v1"
SETPOINT_MECHANISM_DOMAIN = SETPOINT_MECHANISM_FORMAT
SETPOINT_EPISODE_EXECUTION_FORMAT = "service-mpc-setpoint-episode-execution-v1"
SETPOINT_RUNNER_FAILURE_FORMAT = "service-mpc-setpoint-runner-failure-v1"

SETPOINT_PROTOCOL_PATH = (
    "docs/research/CAUSAL_SERVICE_MPC_SETPOINT_DISPATCH_PROTOCOL.md"
)
SETPOINT_PROTOCOL_SHA256 = (
    "21b12eaa245b21d20d9458b9e3d0b8d163c682ef85ad9d5579fde392647e8782"
)
SETPOINT_CLARIFICATION_PATH = (
    "docs/research/CAUSAL_SERVICE_MPC_SETPOINT_DISPATCH_CLARIFICATION_001.md"
)
SETPOINT_CLARIFICATION_SHA256 = (
    "5aecfc8e87b87deeb8fb03a4f226854f1b6245ec762209325f88c9c08aaea349"
)
SETPOINT_LOCK_SHA256 = (
    "080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd"
)

SETPOINT_DELIVERY_RATE_KG_H = 35.0
SETPOINT_SEED = 0
SETPOINT_HORIZON_H = 72.0
SETPOINT_DECISIONS = 288
SETPOINT_SOLVER_STAGES = 576
SETPOINT_PHYSICAL_MINUTES = 4_320
SETPOINT_PRIOR_DELIVERED_KG = 647.7138003263635
SETPOINT_EXPECTED_MANIFEST_SHA256 = (
    "1e036f69c0514ad34a307b2d213fab3c781561ab6ad1efd199ed8313eca8dae4"
)
SETPOINT_EXPECTED_EXOGENOUS_SHA256 = (
    "2e0b7fd35b85c2a0b9d220c32151a00f96adbd7e24e610bb3b17740345903368"
)
SETPOINT_CLAIM_BOUNDARY = (
    "single previously inspected rate-35/seed-0 development mechanism cell; "
    "not a retry, holdout, controller-superiority, commission, or commercial claim"
)

SETPOINT_MECHANISM_CONTRACT = {
    "format": SETPOINT_MECHANISM_FORMAT,
    "protocol": {
        "path": SETPOINT_PROTOCOL_PATH,
        "raw_file_sha256": SETPOINT_PROTOCOL_SHA256,
    },
    "clarification": {
        "path": SETPOINT_CLARIFICATION_PATH,
        "raw_file_sha256": SETPOINT_CLARIFICATION_SHA256,
        "scope": "implementation gate 9 cross-version structural equivalence only",
    },
    "dependency_lock": {"path": "uv.lock", "sha256": SETPOINT_LOCK_SHA256},
    "claim_boundary": SETPOINT_CLAIM_BOUNDARY,
    "cell": {
        "delivery_rate_kg_h": SETPOINT_DELIVERY_RATE_KG_H,
        "seed": SETPOINT_SEED,
        "horizon_h": SETPOINT_HORIZON_H,
        "decision_interval_s": 900.0,
        "expected_decisions": SETPOINT_DECISIONS,
        "expected_solver_stages": SETPOINT_SOLVER_STAGES,
        "expected_physical_minutes": SETPOINT_PHYSICAL_MINUTES,
        "manifest_sha256": SETPOINT_EXPECTED_MANIFEST_SHA256,
        "exogenous_episode_sha256": SETPOINT_EXPECTED_EXOGENOUS_SHA256,
    },
    "controller": {
        "id": SERVICE_MPC_SETPOINT_CONTROLLER_ID,
        "settings_sha256": DEFAULT_SERVICE_MPC_SETPOINT_SETTINGS_V1.sha256,
    },
    "gates": {
        "S1": {
            "successful_decisions": SETPOINT_DECISIONS,
            "successful_solver_stages": SETPOINT_SOLVER_STAGES,
            "physical_minute_dispatches": SETPOINT_PHYSICAL_MINUTES,
            "fallback_actions": 0,
        },
        "S2": {
            "maximum_preguard_target_overshoot_kw": (SETPOINT_TRACKING_TOLERANCE_KW),
            "simultaneous_charge_support_requests": 0,
        },
        "S3": {
            "maximum_reachable_tracking_error_kw": (SETPOINT_TRACKING_TOLERANCE_KW),
            "unattributed_target_deficits": 0,
        },
        "S4": {
            "minimum_productive_time_s_exclusive": 0.0,
            "maximum_aborted_starts": 4,
            "maximum_forced_stops": 4,
        },
        "S5": {
            "minimum_soc": 0.15,
            "minimum_terminal_soc": 0.25,
            "maximum_shutdown_steps": 0,
            "maximum_out_of_range_decisions": 0,
        },
        "S6": {
            "maximum_accounting_chain_or_authority_violations": 0,
            "maximum_target_decomposition_error_kw": (SETPOINT_TRACKING_TOLERANCE_KW),
        },
        "S7": {
            "minimum_delivered_kg_exclusive": SETPOINT_PRIOR_DELIVERED_KG,
            "interpretation": "direction diagnostic only",
        },
        "S8": {
            "controller_p99_ns": 2_000_000_000,
            "controller_maximum_ns": 5_000_000_000,
            "dispatcher_p99_ns": 5_000_000,
            "dispatcher_maximum_ns": 25_000_000,
        },
    },
    "execution_unit": "one fresh process; no same-revision retry",
}
SETPOINT_MECHANISM_CONTRACT_SHA256 = domain_sha256(
    "service-mpc-setpoint-mechanism-contract-v1",
    SETPOINT_MECHANISM_CONTRACT,
)

Executor = Callable[..., dict[str, Any]]
ProvenanceCollector = Callable[..., dict[str, Any]]


@dataclass(frozen=True, slots=True)
class PublishedSetpointMechanismV1:
    """Identity and gate result for one externally published mechanism cell."""

    path: Path
    raw_file_sha256: str
    report_sha256: str
    status: str
    gates: tuple[dict[str, Any], ...]
    elapsed_ns: int

    def __post_init__(self) -> None:
        path = Path(self.path)
        if not path.is_absolute():
            raise ValueError("published setpoint mechanism path must be absolute")
        object.__setattr__(self, "path", path.resolve(strict=False))
        require_sha256("raw_file_sha256", self.raw_file_sha256)
        require_sha256("report_sha256", self.report_sha256)
        if self.status not in {
            "execution_failed",
            "complete_all_gates_passed",
            "complete_gate_failed",
        }:
            raise ValueError("published setpoint mechanism status is invalid")
        if (
            not isinstance(self.gates, tuple)
            or len(self.gates) != 8
            or [gate.get("id") for gate in self.gates]
            != [f"S{index}" for index in range(1, 9)]
        ):
            raise ValueError("published setpoint mechanism gates are invalid")
        if (
            isinstance(self.elapsed_ns, bool)
            or not isinstance(self.elapsed_ns, Integral)
            or self.elapsed_ns < 0
        ):
            raise ValueError("elapsed_ns must be a non-negative integer")


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return value


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _failure_record(
    *,
    phase: str,
    exception: Exception,
    decision_index: int,
    physical_steps: int,
    unreconciled_execution: MPCSetpointIntervalExecutionV1 | None,
) -> dict[str, Any]:
    message = " ".join(str(exception).split()) or "(no exception message)"
    return {
        "format": SETPOINT_RUNNER_FAILURE_FORMAT,
        "phase": phase,
        "exception_type": type(exception).__name__,
        "message": message[:500],
        "decision_index": decision_index,
        "physical_steps": physical_steps,
        "unreconciled_execution": (
            None if unreconciled_execution is None else unreconciled_execution.to_dict()
        ),
    }


def execute_setpoint_episode(
    *,
    plant: PlantConfig,
    episode: EpisodeConfigV2,
    constraints: Constraints,
    weights: RewardWeightsV2,
    settings: ServiceMPCSetpointSettingsV1,
    seed: int,
) -> dict[str, Any]:
    """Execute one typed setpoint episode without publication or source checks.

    Tests may call this with a short horizon.  The public runner below always
    supplies the frozen 72-hour settings and cell.
    """

    if not isinstance(plant, PlantConfig):
        raise TypeError("plant must be PlantConfig")
    if not isinstance(episode, EpisodeConfigV2):
        raise TypeError("episode must be EpisodeConfigV2")
    if not isinstance(constraints, Constraints):
        raise TypeError("constraints must be Constraints")
    if not isinstance(weights, RewardWeightsV2):
        raise TypeError("weights must be RewardWeightsV2")
    if not isinstance(settings, ServiceMPCSetpointSettingsV1):
        raise TypeError("settings must be ServiceMPCSetpointSettingsV1")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    validated_episode = episode.validate(plant)
    if not math.isclose(
        validated_episode.horizon_h,
        settings.task_horizon_h,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("episode and setpoint-controller task horizons differ")
    if not math.isclose(
        validated_episode.decision_interval_s,
        settings.interval_s,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("episode and setpoint-controller decision cadences differ")
    if validated_episode.decision_steps(plant) != settings.task_interval_count:
        raise ValueError("episode and setpoint-controller decision counts differ")

    started_at_utc = _utc_now()
    env = PlantEnvV2(
        config=plant,
        episode=validated_episode,
        constraints=constraints,
        weights=weights,
    )
    env.reset(seed=int(seed))
    initial_manifest = env.audit_manifest()
    controller = ServiceMPCSetpointControllerV1(settings)
    dispatcher_elapsed: list[tuple[int, ...]] = []
    failure: dict[str, Any] | None = None
    terminated = False
    truncated = False

    while not (terminated or truncated):
        decision_index = env.decisions_taken
        phase = "controller_compute"
        try:
            decision = controller.compute_setpoints(
                env.controller_observation,
                constraints=constraints,
                config=plant,
                commitment_profile=validated_episode.electrolyser_commitment,
            )
            phase = "environment_setpoint_execution"
            _, _, terminated, truncated, info = env.step_setpoints(decision.setpoints)
            execution = env.last_setpoint_execution
            if not isinstance(execution, MPCSetpointIntervalExecutionV1):
                raise RuntimeError("environment omitted typed setpoint execution")
            raw_timings = info.get("dispatcher_elapsed_ns")
            if not isinstance(raw_timings, list) or any(
                isinstance(value, bool) or not isinstance(value, Integral) or value < 0
                for value in raw_timings
            ):
                raise ValueError("environment dispatcher timing record is invalid")
            timings = tuple(int(value) for value in raw_timings)
            phase = "transition_reconciliation"
            controller.observe_transition(
                env.controller_observation,
                execution=execution,
            )
            dispatcher_elapsed.append(timings)
            if truncated and not terminated:
                raise RuntimeError("setpoint mechanism episode was truncated")
        except Exception as exc:
            failure = _failure_record(
                phase=phase,
                exception=exc,
                decision_index=decision_index,
                physical_steps=env.physics_steps_taken,
                unreconciled_execution=(
                    env.last_setpoint_execution
                    if controller.has_pending_transition
                    else None
                ),
            )
            break

    events = controller.decision_events
    closed_decisions = sum(
        isinstance(event, MPCSetpointDecisionEvidenceV1) and event.closed
        for event in events
    )
    run_succeeded = bool(
        failure is None
        and terminated
        and not truncated
        and closed_decisions == settings.task_interval_count
        and not controller.has_pending_transition
    )
    if not run_succeeded and failure is None:
        failure = {
            "format": SETPOINT_RUNNER_FAILURE_FORMAT,
            "phase": "completion_validation",
            "exception_type": "IncompleteSetpointEpisode",
            "message": "episode did not reach its declared terminal boundary",
            "decision_index": env.decisions_taken,
            "physical_steps": env.physics_steps_taken,
            "unreconciled_execution": None,
        }

    final_observation_sha = public_observation_sha256(env.controller_observation)
    run_evidence: dict[str, Any] | None
    evidence_failure: dict[str, Any] | None = None
    try:
        run_evidence = build_mpc_setpoint_run_evidence(
            events,
            tuple(dispatcher_elapsed),
            settings=settings,
            completed_decisions=closed_decisions,
            run_succeeded=run_succeeded,
            pending_transition=controller.has_pending_transition,
            runner_failure_phase=(
                None
                if run_succeeded
                else str(_mapping("runner failure", failure)["phase"])
            ),
            final_observation_sha256=final_observation_sha,
        )
    except Exception as exc:
        run_evidence = None
        evidence_failure = _failure_record(
            phase="run_evidence_validation",
            exception=exc,
            decision_index=env.decisions_taken,
            physical_steps=env.physics_steps_taken,
            unreconciled_execution=env.last_setpoint_execution,
        )
        if failure is None:
            failure = evidence_failure

    terminal_manifest = env.audit_manifest()
    result = {
        "format": SETPOINT_EPISODE_EXECUTION_FORMAT,
        "started_at_utc": started_at_utc,
        "completed_at_utc": _utc_now(),
        "seed": int(seed),
        "manifest": initial_manifest,
        "terminal_delivery_accounting": terminal_manifest["delivery_accounting"],
        "controller": controller.to_dict(),
        "metrics": env.metrics.to_dict(),
        "run_evidence": run_evidence,
        "failure": failure,
        "evidence_validation_failure": evidence_failure,
        "unvalidated_controller_events": (
            None if run_evidence is not None else [event.to_dict() for event in events]
        ),
        "unvalidated_dispatcher_elapsed_ns": (
            None
            if run_evidence is not None
            else [list(group) for group in dispatcher_elapsed]
        ),
    }
    canonical_json(result)
    return result


def _validate_protocols(root: Path) -> None:
    for label, relative, expected in (
        ("setpoint protocol", SETPOINT_PROTOCOL_PATH, SETPOINT_PROTOCOL_SHA256),
        (
            "setpoint clarification",
            SETPOINT_CLARIFICATION_PATH,
            SETPOINT_CLARIFICATION_SHA256,
        ),
        ("dependency lock", "uv.lock", SETPOINT_LOCK_SHA256),
    ):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"frozen {label} is missing")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen {label} digest changed")


def _validate_source_start(provenance: Mapping[str, Any], root: Path) -> None:
    source = _mapping("source provenance", provenance.get("source"))
    lock = _mapping("dependency-lock provenance", provenance.get("dependency_lock"))
    if source.get("identity_complete") is not True:
        raise RuntimeError("source identity is incomplete")
    if source.get("git_dirty") is not False:
        raise RuntimeError("setpoint mechanism requires a clean source revision")
    if source.get("repository_root") != str(root):
        raise RuntimeError("source provenance reports another repository root")
    revision = source.get("git_revision")
    if not isinstance(revision, str) or not revision:
        raise RuntimeError("source provenance omits the Git revision")
    require_sha256("source worktree identity", source.get("worktree_identity_sha256"))
    if lock != {"path": "uv.lock", "sha256": SETPOINT_LOCK_SHA256}:
        raise RuntimeError("source dependency lock is not the frozen lock")


def _source_audit(
    start: Mapping[str, Any],
    end: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    _validate_source_start(start, root)
    _validate_source_start(end, root)
    start_source = _mapping("start source", start.get("source"))
    end_source = _mapping("end source", end.get("source"))
    audit = {
        "format": "setpoint-mechanism-source-audit-v1",
        "status": "valid",
        "clean_at_start": start_source.get("git_dirty") is False,
        "clean_at_end": end_source.get("git_dirty") is False,
        "identity_complete_at_start": start_source.get("identity_complete") is True,
        "identity_complete_at_end": end_source.get("identity_complete") is True,
        "revision_unchanged": (
            start_source.get("git_revision") == end_source.get("git_revision")
        ),
        "worktree_identity_unchanged": (
            start_source.get("worktree_identity_sha256")
            == end_source.get("worktree_identity_sha256")
        ),
        "dependency_lock_unchanged": (
            start.get("dependency_lock") == end.get("dependency_lock")
        ),
        "git_revision": start_source.get("git_revision"),
        "start_worktree_identity_sha256": start_source.get("worktree_identity_sha256"),
        "end_worktree_identity_sha256": end_source.get("worktree_identity_sha256"),
    }
    required = (
        "clean_at_start",
        "clean_at_end",
        "identity_complete_at_start",
        "identity_complete_at_end",
        "revision_unchanged",
        "worktree_identity_unchanged",
        "dependency_lock_unchanged",
    )
    if any(audit[name] is not True for name in required):
        audit["status"] = "invalid"
        raise RuntimeError("source or dependency identity changed during execution")
    return audit


def _validate_execution(
    execution: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    if execution.get("format") != SETPOINT_EPISODE_EXECUTION_FORMAT:
        raise ValueError("setpoint executor returned the wrong format")
    if execution.get("seed") != SETPOINT_SEED:
        raise ValueError("setpoint executor escaped the frozen seed")
    if execution.get("controller") != ServiceMPCSetpointControllerV1().to_dict():
        raise ValueError("setpoint executor used another controller or settings")
    manifest = _mapping("setpoint execution manifest", execution.get("manifest"))
    if manifest.get("manifest_sha256") != SETPOINT_EXPECTED_MANIFEST_SHA256:
        raise ValueError("setpoint execution manifest differs from the frozen cell")
    if manifest.get("exogenous_episode_sha256") != SETPOINT_EXPECTED_EXOGENOUS_SHA256:
        raise ValueError(
            "setpoint execution exogenous episode differs from the frozen cell"
        )
    metrics = _mapping("setpoint execution metrics", execution.get("metrics"))
    evidence_raw = execution.get("run_evidence")
    failure_raw = execution.get("failure")
    if failure_raw is None:
        failure = None
    else:
        failure = _mapping("setpoint runner failure", failure_raw)
        if failure.get("format") != SETPOINT_RUNNER_FAILURE_FORMAT:
            raise ValueError("setpoint runner failure format is invalid")
        if not isinstance(failure.get("phase"), str) or not failure.get("phase"):
            raise ValueError("setpoint runner failure phase is invalid")
    evidence: Mapping[str, Any] | None
    if evidence_raw is None:
        evidence = None
        if failure is None:
            raise ValueError("setpoint execution omitted both evidence and failure")
    else:
        evidence = _mapping("setpoint run evidence", evidence_raw)
        verify_mpc_setpoint_run_digests(evidence)
        if evidence.get("controller_id") != SERVICE_MPC_SETPOINT_CONTROLLER_ID:
            raise ValueError("setpoint evidence controller identity is invalid")
        if evidence.get("settings_sha256") != (
            DEFAULT_SERVICE_MPC_SETPOINT_SETTINGS_V1.sha256
        ):
            raise ValueError("setpoint evidence settings identity is invalid")
        if evidence.get("status") == "complete" and failure is not None:
            raise ValueError("complete setpoint evidence contains a runner failure")
        if evidence.get("status") == "failed" and failure is None:
            raise ValueError("failed setpoint evidence omitted its runner failure")
    canonical_json(dict(execution))
    return metrics, evidence


def _gate(
    gate_id: str,
    criterion: str,
    passed: bool | None,
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "id": gate_id,
        "criterion": criterion,
        "status": "not_evaluated" if passed is None else ("pass" if passed else "fail"),
        "observed": dict(observed),
    }


def _build_gates(
    metrics: Mapping[str, Any],
    evidence: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], ...]:
    counts = (
        {} if evidence is None else _mapping("setpoint counts", evidence.get("counts"))
    )
    execution_passed = bool(
        evidence is not None
        and evidence.get("status") == "complete"
        and evidence.get("expected_decisions") == SETPOINT_DECISIONS
        and evidence.get("completed_decisions") == SETPOINT_DECISIONS
        and counts.get("successful_decision_count") == SETPOINT_DECISIONS
        and counts.get("failed_decision_count") == 0
        and counts.get("closed_decision_count") == SETPOINT_DECISIONS
        and counts.get("interval_execution_count") == SETPOINT_DECISIONS
        and counts.get("physical_minute_count") == SETPOINT_PHYSICAL_MINUTES
        and counts.get("attempted_solver_stage_count") == SETPOINT_SOLVER_STAGES
        and counts.get("successful_solver_stage_count") == SETPOINT_SOLVER_STAGES
        and counts.get("fallback_action_count") == 0
        and evidence.get("pending_transition") is False
        and len(evidence.get("event_sha256", ())) == SETPOINT_DECISIONS
        and len(evidence.get("execution_sha256", ())) == SETPOINT_DECISIONS
        and len(evidence.get("application_sha256", ())) == SETPOINT_PHYSICAL_MINUTES
        and metrics.get("decisions") == SETPOINT_DECISIONS
        and metrics.get("physical_steps") == SETPOINT_PHYSICAL_MINUTES
    )
    gates: list[dict[str, Any]] = [
        _gate(
            "S1",
            "288 closed decisions, 576 successful stages, 4,320 reconciled minute dispatches, zero fallback.",
            execution_passed,
            {
                "evidence_status": None if evidence is None else evidence.get("status"),
                "completed_decisions": (
                    None if evidence is None else evidence.get("completed_decisions")
                ),
                "successful_solver_stages": counts.get("successful_solver_stage_count"),
                "physical_minutes": counts.get("physical_minute_count"),
                "fallback_actions": counts.get("fallback_action_count"),
            },
        )
    ]
    if not execution_passed or evidence is None:
        for gate_id, criterion in (
            ("S2", "No pre-guard target overshoot or simultaneous charge/support."),
            ("S3", "Exact reachable tracking and complete deficit attribution."),
            ("S4", "Productive operation with at most four aborted and forced starts."),
            ("S5", "Reserve, SOC, validity, and shutdown safety limits."),
            ("S6", "Digest, chain, accounting, decomposition, and authority closure."),
            ("S7", "Delivery strictly exceeds the frozen 647.7138003263635 kg result."),
            ("S8", "Controller and dispatcher latency limits."),
        ):
            gates.append(_gate(gate_id, criterion, None, {}))
        return tuple(gates)

    tracking = _mapping("target tracking", evidence.get("target_tracking"))
    requested_overshoot = _finite(
        "requested maximum overshoot", tracking.get("requested_max_overshoot_kw")
    )
    gates.append(
        _gate(
            "S2",
            "No pre-guard target overshoot above 1e-7 kW or simultaneous charge/support.",
            counts.get("requested_target_overshoot_minutes") == 0
            and counts.get("simultaneous_charge_support_request_minutes") == 0
            and requested_overshoot <= SETPOINT_TRACKING_TOLERANCE_KW,
            {
                "requested_target_overshoot_minutes": counts.get(
                    "requested_target_overshoot_minutes"
                ),
                "requested_max_overshoot_kw": requested_overshoot,
                "simultaneous_charge_support_request_minutes": counts.get(
                    "simultaneous_charge_support_request_minutes"
                ),
            },
        )
    )
    attribution = _mapping(
        "target deficit attribution", evidence.get("target_deficit_attribution_kwh")
    )
    maximum_closure_error = abs(
        _finite(
            "maximum target-decomposition error",
            attribution.get("maximum_absolute_closure_error_kw"),
        )
    )
    gates.append(
        _gate(
            "S3",
            "Reachable target error is at most 1e-7 kW and every deficit is attributed.",
            counts.get("reachable_tracking_violation_minutes") == 0
            and counts.get("unattributed_target_deficit_minutes") == 0
            and maximum_closure_error <= SETPOINT_TRACKING_TOLERANCE_KW,
            {
                "reachable_tracking_violation_minutes": counts.get(
                    "reachable_tracking_violation_minutes"
                ),
                "unattributed_target_deficit_minutes": counts.get(
                    "unattributed_target_deficit_minutes"
                ),
                "maximum_absolute_closure_error_kw": maximum_closure_error,
            },
        )
    )
    operating_time_s = _finite(
        "electrolyser operating time", metrics.get("electrolyser_operating_time_s")
    )
    aborted_starts = _integer(
        "electrolyser aborted starts", metrics.get("electrolyser_aborted_starts")
    )
    forced_stops = _integer(
        "electrolyser forced stops", metrics.get("electrolyser_forced_stops")
    )
    gates.append(
        _gate(
            "S4",
            "Productive operation is positive with at most four aborted starts and forced stops.",
            operating_time_s > 0.0 and aborted_starts <= 4 and forced_stops <= 4,
            {
                "electrolyser_operating_time_s": operating_time_s,
                "electrolyser_aborted_starts": aborted_starts,
                "electrolyser_forced_stops": forced_stops,
            },
        )
    )
    reward_components = _mapping("reward components", metrics.get("reward_components"))
    reserve_component = _finite(
        "reserve reward component", reward_components.get("reserve_breach", 0.0)
    )
    min_soc = _finite("minimum battery SOC", metrics.get("min_battery_soc"))
    terminal_soc = _finite("terminal battery SOC", metrics.get("terminal_battery_soc"))
    shutdown_steps = _integer("shutdown steps", metrics.get("shutdown_steps"))
    out_of_range = _integer(
        "out-of-range decisions", metrics.get("out_of_range_decisions")
    )
    gates.append(
        _gate(
            "S5",
            "No reserve breach, invalid decision, or shutdown; minimum SOC 15% and terminal SOC 25%.",
            reserve_component == 0.0
            and min_soc + 1e-12 >= 0.15
            and terminal_soc + 1e-12 >= 0.25
            and shutdown_steps == 0
            and out_of_range == 0,
            {
                "reserve_reward_component": reserve_component,
                "min_battery_soc": min_soc,
                "terminal_battery_soc": terminal_soc,
                "shutdown_steps": shutdown_steps,
                "out_of_range_decisions": out_of_range,
            },
        )
    )
    violation_fields = (
        "accounting_violation_count",
        "chain_violation_count",
        "unattributed_target_deficit_minutes",
        "applied_support_authority_violation_minutes",
    )
    violations = {
        field: _integer(field, counts.get(field)) for field in violation_fields
    }
    gates.append(
        _gate(
            "S6",
            "Zero digest, chain, accounting, decomposition, and support-authority violations.",
            all(value == 0 for value in violations.values())
            and maximum_closure_error <= SETPOINT_TRACKING_TOLERANCE_KW,
            {**violations, "maximum_absolute_closure_error_kw": maximum_closure_error},
        )
    )
    delivered_kg = _finite("delivered methane", metrics.get("delivered_kg"))
    gates.append(
        _gate(
            "S7",
            "Delivery strictly exceeds the frozen 647.7138003263635 kg controller result; diagnostic only.",
            delivered_kg > SETPOINT_PRIOR_DELIVERED_KG,
            {
                "delivered_kg": delivered_kg,
                "prior_delivered_kg": SETPOINT_PRIOR_DELIVERED_KG,
                "delta_kg": delivered_kg - SETPOINT_PRIOR_DELIVERED_KG,
            },
        )
    )
    controller_latency = _mapping(
        "controller latency", evidence.get("decision_latency")
    )
    dispatcher_latency = _mapping(
        "dispatcher latency", evidence.get("dispatcher_latency")
    )
    controller_p99 = _integer(
        "controller p99 latency", controller_latency.get("p99_ns")
    )
    controller_maximum = _integer(
        "controller maximum latency", controller_latency.get("maximum_ns")
    )
    dispatcher_p99 = _integer(
        "dispatcher p99 latency", dispatcher_latency.get("p99_ns")
    )
    dispatcher_maximum = _integer(
        "dispatcher maximum latency", dispatcher_latency.get("maximum_ns")
    )
    gates.append(
        _gate(
            "S8",
            "Controller p99/max <= 2/5 s and dispatcher p99/max <= 5/25 ms.",
            controller_p99 <= 2_000_000_000
            and controller_maximum <= 5_000_000_000
            and dispatcher_p99 <= 5_000_000
            and dispatcher_maximum <= 25_000_000,
            {
                "controller_p99_ns": controller_p99,
                "controller_maximum_ns": controller_maximum,
                "dispatcher_p99_ns": dispatcher_p99,
                "dispatcher_maximum_ns": dispatcher_maximum,
            },
        )
    )
    return tuple(gates)


def _status(gates: tuple[dict[str, Any], ...]) -> str:
    if gates[0]["status"] != "pass":
        return "execution_failed"
    if all(gate["status"] == "pass" for gate in gates):
        return "complete_all_gates_passed"
    return "complete_gate_failed"


def verify_setpoint_mechanism_artifact(payload: Mapping[str, Any]) -> None:
    """Deeply verify a detached published payload and recompute all gates."""

    artifact = _mapping("setpoint mechanism artifact", payload)
    if artifact.get("format") != SETPOINT_MECHANISM_FORMAT:
        raise ValueError("setpoint mechanism artifact format is invalid")
    if artifact.get("contract") != SETPOINT_MECHANISM_CONTRACT:
        raise ValueError("setpoint mechanism contract changed")
    if artifact.get("contract_sha256") != SETPOINT_MECHANISM_CONTRACT_SHA256:
        raise ValueError("setpoint mechanism contract digest is invalid")
    supplied_report_sha = require_sha256(
        "setpoint mechanism report sha256", artifact.get("report_sha256")
    )
    body = dict(artifact)
    del body["report_sha256"]
    if domain_sha256(SETPOINT_MECHANISM_DOMAIN, body) != supplied_report_sha:
        raise ValueError("setpoint mechanism report digest is invalid")
    execution = _mapping("setpoint mechanism execution", artifact.get("execution"))
    metrics, evidence = _validate_execution(execution)
    expected_gates = _build_gates(metrics, evidence)
    raw_gates = artifact.get("gates")
    if not isinstance(raw_gates, list) or raw_gates != list(expected_gates):
        raise ValueError("setpoint mechanism gates are invalid")
    if artifact.get("status") != _status(expected_gates):
        raise ValueError("setpoint mechanism status is invalid")
    source = _mapping("setpoint mechanism source", artifact.get("source"))
    audit = _mapping("setpoint mechanism source audit", source.get("audit"))
    if audit.get("status") != "valid":
        raise ValueError("setpoint mechanism source audit is invalid")
    for field in (
        "clean_at_start",
        "clean_at_end",
        "identity_complete_at_start",
        "identity_complete_at_end",
        "revision_unchanged",
        "worktree_identity_unchanged",
        "dependency_lock_unchanged",
    ):
        if audit.get(field) is not True:
            raise ValueError(f"setpoint mechanism source audit failed: {field}")
    start = _mapping("setpoint mechanism start provenance", source.get("start"))
    end = _mapping("setpoint mechanism end provenance", source.get("end"))
    start_source = _mapping("setpoint mechanism start source", start.get("source"))
    end_source = _mapping("setpoint mechanism end source", end.get("source"))
    if not (
        start_source.get("git_revision")
        == end_source.get("git_revision")
        == audit.get("git_revision")
        and start_source.get("worktree_identity_sha256")
        == end_source.get("worktree_identity_sha256")
        == audit.get("start_worktree_identity_sha256")
        == audit.get("end_worktree_identity_sha256")
        and start.get("dependency_lock")
        == end.get("dependency_lock")
        == {"path": "uv.lock", "sha256": SETPOINT_LOCK_SHA256}
    ):
        raise ValueError("setpoint mechanism source audit identities disagree")
    canonical_json(dict(artifact))


def _atomic_write(payload: Mapping[str, Any], target: Path) -> str:
    raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        reopened = temporary.read_bytes()
        if reopened != raw or canonical_json(json.loads(reopened)) != canonical_json(
            payload
        ):
            raise ValueError("setpoint mechanism changed during serialization")
        os.link(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        published = target.read_bytes()
        if published != raw:
            raise ValueError("published setpoint mechanism bytes changed")
        return hashlib.sha256(published).hexdigest()
    finally:
        temporary.unlink(missing_ok=True)


def run_setpoint_mechanism(
    *,
    destination: Path | str,
    repository_root: Path | None = None,
    executor: Executor = execute_setpoint_episode,
    provenance_collector: ProvenanceCollector = collect_provenance,
) -> PublishedSetpointMechanismV1:
    """Run and publish the single frozen absolute-setpoint mechanism cell."""

    started_ns = time.perf_counter_ns()
    root = find_repository_root(repository_root)
    _validate_protocols(root)
    target = validate_artifact_destination(destination, repository_root=root)
    plant = PlantConfig()
    episode = EpisodeConfigV2(
        horizon_h=SETPOINT_HORIZON_H,
        delivery_rate_kg_h=SETPOINT_DELIVERY_RATE_KG_H,
    ).validate(plant)
    constraints = Constraints()
    weights = RewardWeightsV2()
    settings = DEFAULT_SERVICE_MPC_SETPOINT_SETTINGS_V1
    start_provenance = provenance_collector(
        episode=episode,
        plant=plant,
        constraints=constraints,
        weights=weights,
        root=root,
    )
    _validate_source_start(start_provenance, root)
    runner_started_at_utc = _utc_now()
    execution = executor(
        plant=plant,
        episode=episode,
        constraints=constraints,
        weights=weights,
        settings=settings,
        seed=SETPOINT_SEED,
    )
    runner_completed_at_utc = _utc_now()
    execution_mapping = _mapping("setpoint mechanism execution", execution)
    metrics, evidence = _validate_execution(execution_mapping)
    gates = _build_gates(metrics, evidence)
    status = _status(gates)
    end_provenance = provenance_collector(
        episode=episode,
        plant=plant,
        constraints=constraints,
        weights=weights,
        root=root,
    )
    source_audit = _source_audit(start_provenance, end_provenance, root=root)
    body: dict[str, Any] = {
        "format": SETPOINT_MECHANISM_FORMAT,
        "schema_version": 1,
        "status": status,
        "claim_boundary": SETPOINT_CLAIM_BOUNDARY,
        "contract": SETPOINT_MECHANISM_CONTRACT,
        "contract_sha256": SETPOINT_MECHANISM_CONTRACT_SHA256,
        "runner": {
            "started_at_utc": runner_started_at_utc,
            "completed_at_utc": runner_completed_at_utc,
        },
        "source": {
            "start": start_provenance,
            "end": end_provenance,
            "audit": source_audit,
        },
        "gates": list(gates),
        "execution": dict(execution_mapping),
    }
    report_sha256 = domain_sha256(SETPOINT_MECHANISM_DOMAIN, body)
    payload = {**body, "report_sha256": report_sha256}
    verify_setpoint_mechanism_artifact(payload)
    raw_file_sha256 = _atomic_write(payload, target)
    final_provenance = provenance_collector(
        episode=episode,
        plant=plant,
        constraints=constraints,
        weights=weights,
        root=root,
    )
    _source_audit(end_provenance, final_provenance, root=root)
    return PublishedSetpointMechanismV1(
        path=target,
        raw_file_sha256=raw_file_sha256,
        report_sha256=report_sha256,
        status=status,
        gates=gates,
        elapsed_ns=time.perf_counter_ns() - started_ns,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m plant_backend.rl.v2.mpc_setpoint_experiment",
        description="Run the single frozen absolute-setpoint MPC mechanism cell.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        published = run_setpoint_mechanism(destination=arguments.output)
    except (TypeError, ValueError) as exc:
        message = " ".join(str(exc).split()) or type(exc).__name__
        print(f"invalid setpoint mechanism run: {message[:500]}", file=sys.stderr)
        return 2
    except Exception as exc:
        message = " ".join(str(exc).split()) or type(exc).__name__
        print(f"setpoint mechanism run failed: {message[:500]}", file=sys.stderr)
        return 1

    print(f"status: {published.status}")
    for gate in published.gates:
        print(f"{gate['id']}: {gate['status']}")
    print(f"written to {published.path}")
    print(f"artifact sha256: {published.raw_file_sha256}")
    print(f"report sha256: {published.report_sha256}")
    print(f"elapsed seconds: {published.elapsed_ns / 1_000_000_000:.6f}")
    return 0 if published.status == "complete_all_gates_passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SETPOINT_MECHANISM_CONTRACT",
    "SETPOINT_MECHANISM_CONTRACT_SHA256",
    "SETPOINT_MECHANISM_FORMAT",
    "PublishedSetpointMechanismV1",
    "execute_setpoint_episode",
    "run_setpoint_mechanism",
    "verify_setpoint_mechanism_artifact",
]

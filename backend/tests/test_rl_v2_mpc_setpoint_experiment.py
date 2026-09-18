"""Admission tests for the frozen one-shot absolute-setpoint mechanism runner."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from plant_sim import PlantConfig

from plant_backend.controllers.base import Constraints
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.mpc_schema import ServiceMPCSetpointSettingsV1
from plant_backend.rl.v2.mpc_setpoint_controller import (
    ServiceMPCSetpointControllerV1,
)
from plant_backend.rl.v2.mpc_setpoint_experiment import (
    SETPOINT_EPISODE_EXECUTION_FORMAT,
    SETPOINT_EXPECTED_EXOGENOUS_SHA256,
    SETPOINT_EXPECTED_MANIFEST_SHA256,
    SETPOINT_LOCK_SHA256,
    SETPOINT_MECHANISM_CONTRACT_SHA256,
    SETPOINT_PHYSICAL_MINUTES,
    SETPOINT_PRIOR_DELIVERED_KG,
    SETPOINT_PROTOCOL_SHA256,
    SETPOINT_RUNNER_FAILURE_FORMAT,
    SETPOINT_SEED,
    _build_gates,
    _validate_protocols,
    build_parser,
    execute_setpoint_episode,
    run_setpoint_mechanism,
    verify_setpoint_mechanism_artifact,
)
from plant_backend.rl.v2.reward import RewardWeightsV2


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SHORT_EPISODE = EpisodeConfigV2(
    horizon_h=1.0,
    delivery_period_h=1.0,
    delivery_rate_kg_h=35.0,
)
SHORT_SETTINGS = ServiceMPCSetpointSettingsV1(
    task_horizon_h=1.0,
    planning_horizon_h=1.0,
)


@pytest.fixture(scope="module")
def short_execution() -> dict[str, object]:
    pytest.importorskip("scipy")
    return execute_setpoint_episode(
        plant=PlantConfig(),
        episode=SHORT_EPISODE,
        constraints=Constraints(),
        weights=RewardWeightsV2(),
        settings=SHORT_SETTINGS,
        seed=0,
    )


def _clean_provenance(**kwargs: object) -> dict[str, object]:
    root = Path(kwargs["root"]).resolve()
    return {
        "source": {
            "repository_root": str(root),
            "git_revision": "a" * 40,
            "git_dirty": False,
            "worktree_identity_sha256": "b" * 64,
            "identity_complete": True,
        },
        "dependency_lock": {"path": "uv.lock", "sha256": SETPOINT_LOCK_SHA256},
        "runtime": {"test": True},
    }


def _failed_frozen_execution(**_: object) -> dict[str, object]:
    return {
        "format": SETPOINT_EPISODE_EXECUTION_FORMAT,
        "started_at_utc": "2026-09-04T00:00:00.000000Z",
        "completed_at_utc": "2026-09-04T00:00:01.000000Z",
        "seed": SETPOINT_SEED,
        "manifest": {
            "manifest_sha256": SETPOINT_EXPECTED_MANIFEST_SHA256,
            "exogenous_episode_sha256": SETPOINT_EXPECTED_EXOGENOUS_SHA256,
        },
        "terminal_delivery_accounting": {},
        "controller": ServiceMPCSetpointControllerV1().to_dict(),
        "metrics": {"decisions": 0, "physical_steps": 0},
        "run_evidence": None,
        "failure": {
            "format": SETPOINT_RUNNER_FAILURE_FORMAT,
            "phase": "controller_compute",
            "exception_type": "SyntheticFailure",
            "message": "outcome-free publication fixture",
            "decision_index": 0,
            "physical_steps": 0,
            "unreconciled_execution": None,
        },
        "evidence_validation_failure": None,
        "unvalidated_controller_events": [],
        "unvalidated_dispatcher_elapsed_ns": [],
    }


def _passing_gate_inputs() -> tuple[dict[str, object], dict[str, object]]:
    counts = {
        "successful_decision_count": 288,
        "failed_decision_count": 0,
        "closed_decision_count": 288,
        "interval_execution_count": 288,
        "physical_minute_count": SETPOINT_PHYSICAL_MINUTES,
        "attempted_solver_stage_count": 576,
        "successful_solver_stage_count": 576,
        "fallback_action_count": 0,
        "requested_target_overshoot_minutes": 0,
        "reachable_tracking_violation_minutes": 0,
        "simultaneous_charge_support_request_minutes": 0,
        "unattributed_target_deficit_minutes": 0,
        "accounting_violation_count": 0,
        "chain_violation_count": 0,
        "applied_support_authority_violation_minutes": 0,
    }
    evidence = {
        "status": "complete",
        "expected_decisions": 288,
        "completed_decisions": 288,
        "pending_transition": False,
        "counts": counts,
        "event_sha256": ["1" * 64] * 288,
        "execution_sha256": ["2" * 64] * 288,
        "application_sha256": ["3" * 64] * SETPOINT_PHYSICAL_MINUTES,
        "target_tracking": {"requested_max_overshoot_kw": 0.0},
        "target_deficit_attribution_kwh": {"maximum_absolute_closure_error_kw": 0.0},
        "decision_latency": {"p99_ns": 1_000, "maximum_ns": 2_000},
        "dispatcher_latency": {"p99_ns": 100, "maximum_ns": 200},
    }
    metrics = {
        "decisions": 288,
        "physical_steps": SETPOINT_PHYSICAL_MINUTES,
        "electrolyser_operating_time_s": 60.0,
        "electrolyser_aborted_starts": 0,
        "electrolyser_forced_stops": 0,
        "reward_components": {"reserve_breach": 0.0},
        "min_battery_soc": 0.25,
        "terminal_battery_soc": 0.30,
        "shutdown_steps": 0,
        "out_of_range_decisions": 0,
        "delivered_kg": SETPOINT_PRIOR_DELIVERED_KG + 1.0,
    }
    return metrics, evidence


def test_frozen_protocol_and_lock_identities_are_current() -> None:
    _validate_protocols(REPOSITORY_ROOT)
    protocol = REPOSITORY_ROOT / (
        "docs/research/CAUSAL_SERVICE_MPC_SETPOINT_DISPATCH_PROTOCOL.md"
    )
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == SETPOINT_PROTOCOL_SHA256
    assert SETPOINT_MECHANISM_CONTRACT_SHA256 == (
        "421561b2ee75650024c11fdf55d806012275b44838a33f6b3714635fa689e304"
    )


def test_short_executor_closes_every_decision_and_minute(
    short_execution: dict[str, object],
) -> None:
    assert short_execution["failure"] is None
    evidence = short_execution["run_evidence"]
    assert isinstance(evidence, dict)
    assert evidence["status"] == "complete"
    assert evidence["completed_decisions"] == 4
    assert evidence["counts"]["successful_solver_stage_count"] == 8
    assert evidence["counts"]["physical_minute_count"] == 60
    assert evidence["counts"]["fallback_action_count"] == 0
    assert short_execution["metrics"]["decisions"] == 4
    assert short_execution["metrics"]["physical_steps"] == 60


def test_executor_rejects_episode_controller_horizon_mismatch() -> None:
    with pytest.raises(ValueError, match="task horizons differ"):
        execute_setpoint_episode(
            plant=PlantConfig(),
            episode=SHORT_EPISODE,
            constraints=Constraints(),
            weights=RewardWeightsV2(),
            settings=ServiceMPCSetpointSettingsV1(
                task_horizon_h=2.0,
                planning_horizon_h=1.0,
            ),
            seed=0,
        )


def test_s1_failure_suppresses_all_outcome_gates() -> None:
    gates = _build_gates({"decisions": 0, "physical_steps": 0}, None)
    assert gates[0]["status"] == "fail"
    assert [gate["status"] for gate in gates[1:]] == ["not_evaluated"] * 7


def test_gate_builder_implements_all_frozen_s1_to_s8_thresholds() -> None:
    metrics, evidence = _passing_gate_inputs()
    gates = _build_gates(metrics, evidence)
    assert [gate["status"] for gate in gates] == ["pass"] * 8

    failed_metrics = dict(metrics)
    failed_metrics["delivered_kg"] = SETPOINT_PRIOR_DELIVERED_KG
    failed = _build_gates(failed_metrics, evidence)
    assert failed[6]["id"] == "S7"
    assert failed[6]["status"] == "fail"
    assert all(gate["status"] == "pass" for gate in (*failed[:6], failed[7]))


def test_runner_publishes_one_external_failure_artifact_fail_closed(
    tmp_path: Path,
) -> None:
    target = tmp_path / "setpoint-failure.json"
    published = run_setpoint_mechanism(
        destination=target,
        repository_root=REPOSITORY_ROOT,
        executor=_failed_frozen_execution,
        provenance_collector=_clean_provenance,
    )

    assert published.status == "execution_failed"
    assert published.gates[0]["status"] == "fail"
    assert [gate["status"] for gate in published.gates[1:]] == ["not_evaluated"] * 7
    raw = target.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == published.raw_file_sha256
    payload = json.loads(raw)
    verify_setpoint_mechanism_artifact(payload)
    assert payload["report_sha256"] == published.report_sha256

    with pytest.raises(ValueError, match="already exists"):
        run_setpoint_mechanism(
            destination=target,
            repository_root=REPOSITORY_ROOT,
            executor=_failed_frozen_execution,
            provenance_collector=_clean_provenance,
        )


def test_runner_refuses_dirty_source_before_opening_outcome(tmp_path: Path) -> None:
    called = False

    def executor(**_: object) -> dict[str, object]:
        nonlocal called
        called = True
        return _failed_frozen_execution()

    def dirty_provenance(**kwargs: object) -> dict[str, object]:
        result = _clean_provenance(**kwargs)
        result["source"]["git_dirty"] = True
        return result

    with pytest.raises(RuntimeError, match="clean source"):
        run_setpoint_mechanism(
            destination=tmp_path / "must-not-exist.json",
            repository_root=REPOSITORY_ROOT,
            executor=executor,
            provenance_collector=dirty_provenance,
        )
    assert called is False


def test_artifact_report_digest_rejects_substitution(tmp_path: Path) -> None:
    target = tmp_path / "setpoint-failure.json"
    run_setpoint_mechanism(
        destination=target,
        repository_root=REPOSITORY_ROOT,
        executor=_failed_frozen_execution,
        provenance_collector=_clean_provenance,
    )
    payload = json.loads(target.read_bytes())
    substituted = deepcopy(payload)
    substituted["execution"]["failure"]["message"] = "substituted"
    with pytest.raises(ValueError, match="report digest"):
        verify_setpoint_mechanism_artifact(substituted)


def test_public_cli_exposes_no_seed_rate_or_tuning_surface(tmp_path: Path) -> None:
    parser = build_parser()
    arguments = parser.parse_args(["--output", str(tmp_path / "out.json")])
    assert arguments.output == tmp_path / "out.json"
    with pytest.raises(SystemExit):
        parser.parse_args(["--output", str(tmp_path / "out.json"), "--seed", "1"])

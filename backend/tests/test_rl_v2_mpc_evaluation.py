"""Evaluator integration and fail-closed evidence tests for service MPC."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from plant_sim import Action

from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.env import PlantEnvV2
from plant_backend.rl.v2.evaluation import (
    LegacyHeuristicControllerV2,
    evaluate_controllers,
    run_controller,
)
from plant_backend.rl.v2.mpc_controller import ServiceMPCControllerV1
from plant_backend.rl.v2.mpc_execution import (
    MPCDecisionFailureV2,
    build_mpc_run_evidence,
)
from plant_backend.rl.v2.mpc_schema import (
    MPCStageSolveV1,
    SERVICE_MPC_CONTROLLER_ID,
    ServiceMPCSettingsV1,
    ServiceMPCSolveError,
)
import plant_backend.rl.v2.mpc_controller as mpc_controller_module


SHORT_EPISODE = EpisodeConfigV2(
    horizon_h=1.0,
    delivery_period_h=1.0,
    delivery_rate_kg_h=35.0,
)
SHORT_SETTINGS = ServiceMPCSettingsV1(
    task_horizon_h=1.0,
    planning_horizon_h=1.0,
)


def _failed_stage() -> MPCStageSolveV1:
    return MPCStageSolveV1(
        stage="primary_delivery",
        success=False,
        status=1,
        message="deliberate evaluator timeout fixture",
        elapsed_ns=123,
        objective_value=None,
        mip_dual_bound=None,
        mip_gap=None,
        mip_node_count=None,
        solution_sha256=None,
        scipy_version="test-scipy",
        highs_version="test-highs",
    )


def _install_solver_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    failed_stage = _failed_stage()

    def fail_solver(*_: object, **__: object):
        raise ServiceMPCSolveError(
            "deliberate evaluator timeout fixture",
            stage_evidence=failed_stage,
        )

    monkeypatch.setattr(
        mpc_controller_module,
        "solve_service_mpc_model",
        fail_solver,
    )


def test_runner_binds_every_mpc_decision_to_public_and_exact_execution() -> None:
    pytest.importorskip("scipy")
    controller = ServiceMPCControllerV1(SHORT_SETTINGS)
    run = run_controller(
        PlantEnvV2(episode=SHORT_EPISODE),
        controller,
        seed=0,
    )

    assert run.successful
    assert run.metrics is not None
    assert run.metrics.decisions == 4
    assert run.requested_actions is not None
    assert len(run.requested_actions) == 4
    evidence = run.controller_evidence
    assert evidence is not None
    assert evidence["format"] == "service-mpc-controller-run-evidence-v2"
    assert evidence["controller_id"] == SERVICE_MPC_CONTROLLER_ID
    assert evidence["status"] == "complete"
    assert evidence["completed_decisions"] == 4
    assert evidence["runner_failure_phase"] is None
    assert evidence["pending_transition"] is False
    assert evidence["event_count"] == 4
    assert evidence["successful_decision_count"] == 4
    assert evidence["failed_decision_count"] == 0
    assert evidence["application_count"] == 4
    assert evidence["fallback_action_count"] == 0
    assert evidence["all_successful_decisions_reconciled"] is True
    assert evidence["decision_latency"]["count"] == 4
    assert evidence["solver_stage_latency"]["count"] == 8
    assert evidence["guard_intervention_count"] == run.metrics.guard_interventions
    assert evidence["corrected_physics_steps"] == run.metrics.corrected_physics_steps
    assert evidence["out_of_range_decisions"] == run.metrics.out_of_range_decisions

    events = evidence["events"]
    applications = evidence["applications"]
    assert isinstance(events, list)
    assert isinstance(applications, list)
    for index, (event, application) in enumerate(
        zip(events, applications, strict=True)
    ):
        assert event["decision_index"] == application["decision_index"] == index
        assert event["residual"] is not None
        assert application["decision_evidence_sha256"] == event["sha256"]
        assert application["action_sha256"] == event["action"]["sha256"]
        assert application["residual_sha256"] == event["residual"]["sha256"]
        assert application["request"]["action"] == event["action"]["action"]
        assert len(application["request_sha256"]) == 64
        assert len(application["guarded_action_sha256"]) == 64
        assert len(application["interval_power_sha256"]) == 64
    assert sum(
        application["interval_power"]["guard_intervention_l1_kwh"]
        for application in applications
    ) == pytest.approx(run.metrics.guard_intervention_kwh)
    assert sum(
        application["interval_power"]["dispatch_gap_l1_kwh"]
        for application in applications
    ) == pytest.approx(run.metrics.dispatch_gap_kwh)
    assert sum(
        application["interval_power"]["request_application_l1_kwh"]
        for application in applications
    ) == pytest.approx(run.metrics.request_application_gap_kwh)
    with pytest.raises(ValueError, match="applications"):
        build_mpc_run_evidence(
            controller.decision_events,
            (),
            completed_decisions=4,
            run_succeeded=True,
            pending_transition=False,
            runner_failure_phase=None,
        )
    json.dumps(run.to_dict(), allow_nan=False, sort_keys=True)


def test_runner_retains_solver_failure_and_never_manufactures_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_solver_failure(monkeypatch)
    controller = ServiceMPCControllerV1(SHORT_SETTINGS)

    run = run_controller(
        PlantEnvV2(episode=SHORT_EPISODE),
        controller,
        seed=0,
    )

    assert not run.successful
    assert run.failure is not None
    assert run.failure.phase == "controller_compute_action"
    assert run.failure.completed_decisions == 0
    assert run.requested_actions == ()
    evidence = run.controller_evidence
    assert evidence is not None
    assert evidence["status"] == "failed"
    assert evidence["completed_decisions"] == 0
    assert evidence["runner_failure_phase"] == "controller_compute_action"
    assert evidence["pending_transition"] is False
    assert evidence["event_count"] == 1
    assert evidence["successful_decision_count"] == 0
    assert evidence["failed_decision_count"] == 1
    assert evidence["application_count"] == 0
    assert evidence["fallback_action_count"] == 0
    event = controller.decision_events[0]
    assert isinstance(event, MPCDecisionFailureV2)
    assert event.stages == (_failed_stage(),)
    assert event.stage_evidence == _failed_stage()
    assert evidence["events"][0]["stages"] == [_failed_stage().to_dict()]
    json.dumps(run.to_dict(), allow_nan=False, sort_keys=True)


def test_runner_fails_closed_when_environment_request_evidence_is_tampered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("scipy")
    env = PlantEnvV2(episode=SHORT_EPISODE)
    original_step = env.step_action

    def tampered_step(action: Action, *, controller_invalid: bool = False):
        observation, reward, terminated, truncated, info = original_step(
            action,
            controller_invalid=controller_invalid,
        )
        changed = deepcopy(info)
        changed["requested"]["action"]["battery_share"] = 0.123
        return observation, reward, terminated, truncated, changed

    monkeypatch.setattr(env, "step_action", tampered_step)
    run = run_controller(
        env,
        ServiceMPCControllerV1(SHORT_SETTINGS),
        seed=0,
    )

    assert not run.successful
    assert run.failure is not None
    assert run.failure.phase == "controller_evidence_capture"
    assert run.failure.completed_decisions == 1
    evidence = run.controller_evidence
    assert evidence is not None
    assert evidence["status"] == "failed"
    assert evidence["successful_decision_count"] == 1
    assert evidence["failed_decision_count"] == 0
    assert evidence["application_count"] == 0
    assert evidence["all_successful_decisions_reconciled"] is True
    assert evidence["fallback_action_count"] == 0


def test_failed_mpc_arm_invalidates_the_whole_paired_effect_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_solver_failure(monkeypatch)

    result = evaluate_controllers(
        seeds=[0],
        controllers=(
            LegacyHeuristicControllerV2(),
            ServiceMPCControllerV1(SHORT_SETTINGS),
        ),
        episode=SHORT_EPISODE,
    )

    assert len(result.runs) == 2
    assert result.pairing()["all_arms_present_by_seed"] is True
    assert result.pairing()["all_arms_successful_by_seed"] is False
    assert result.execution_summary()["all_planned_arms_succeeded"] is False
    contrast = result.contrast_results()[0]
    assert (
        contrast["effect_summary_status"]
        == "invalid_incomplete_or_mismatched_execution"
    )
    assert all(
        metric["raw_treatment_minus_control_summary"] is None
        and metric["benefit_delta_summary"] is None
        for metric in contrast["metrics"].values()
    )
    failed = next(
        run for run in result.runs if run.controller == SERVICE_MPC_CONTROLLER_ID
    )
    assert failed.controller_evidence is not None
    assert failed.controller_evidence["fallback_action_count"] == 0

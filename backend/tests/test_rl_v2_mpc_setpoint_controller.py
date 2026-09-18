"""Causal planning, setpoint output, transition closure, and failure tests."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import NoReturn

import pytest

from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.control import ControllerV2
from plant_backend.rl.v2.env import PlantEnvV2
from plant_backend.rl.v2.mpc_controller import ServiceMPCControllerV1
from plant_backend.rl.v2.mpc_execution import (
    MPCDecisionEvidenceV1,
    public_observation_sha256,
)
from plant_backend.rl.v2.mpc_schema import (
    MPCStageSolveV1,
    ServiceMPCSetpointSettingsV1,
    ServiceMPCSettingsV1,
    ServiceMPCSolveError,
)
from plant_backend.rl.v2.mpc_setpoint_controller import (
    SERVICE_MPC_SETPOINT_RUN_EVIDENCE_FORMAT,
    ServiceMPCSetpointControllerV1,
    SetpointControllerV2,
)
from plant_backend.rl.v2.mpc_setpoint_decision import (
    MPCSetpointDecisionEvidenceV1,
    MPCSetpointDecisionFailureV1,
    MPCSetpointDecisionV1,
)
from plant_backend.rl.v2.mpc_setpoint_execution import (
    MPCSetpointIntervalExecutionV1,
)
import plant_backend.rl.v2.mpc_setpoint_controller as controller_module


SHORT_EPISODE = EpisodeConfigV2(
    horizon_h=1.0,
    delivery_period_h=1.0,
    delivery_rate_kg_h=35.0,
)
SHORT_SETTINGS = ServiceMPCSetpointSettingsV1(
    task_horizon_h=1.0,
    planning_horizon_h=1.0,
)


def _reset_pair(
    *,
    seed: int = 0,
) -> tuple[PlantEnvV2, ServiceMPCSetpointControllerV1]:
    pytest.importorskip("scipy")
    env = PlantEnvV2(episode=SHORT_EPISODE)
    env.reset(seed=seed)
    return env, ServiceMPCSetpointControllerV1(SHORT_SETTINGS)


def _compute(
    env: PlantEnvV2,
    controller: ServiceMPCSetpointControllerV1,
) -> MPCSetpointDecisionV1:
    return controller.compute_setpoints(
        env.controller_observation,
        constraints=env.constraints,
        config=env.config,
        commitment_profile=env.episode_config.electrolyser_commitment,
    )


def _apply_and_close(
    env: PlantEnvV2,
    controller: ServiceMPCSetpointControllerV1,
    decision: MPCSetpointDecisionV1,
) -> dict[str, object]:
    _, _, _, _, info = env.step_setpoints(decision.setpoints)
    execution = env.last_setpoint_execution
    assert isinstance(execution, MPCSetpointIntervalExecutionV1)
    controller.observe_transition(
        env.controller_observation,
        execution=execution,
    )
    return info


def test_descriptor_declares_distinct_setpoint_contract_and_no_fallback() -> None:
    controller = ServiceMPCSetpointControllerV1(SHORT_SETTINGS)
    descriptor = controller.to_dict()

    assert isinstance(controller, SetpointControllerV2)
    assert not isinstance(controller, ControllerV2)
    assert descriptor["id"] == "service-mpc-setpoint-dispatch-v1"
    assert descriptor["settings"] == SHORT_SETTINGS.to_dict()
    assert descriptor["output_schema"] == "service-mpc-process-setpoints-v1"
    assert descriptor["minute_application_schema"] == (
        "service-mpc-minute-application-v1"
    )
    assert descriptor["interval_execution_schema"] == (
        "service-mpc-setpoint-interval-execution-v1"
    )
    assert descriptor["run_evidence_format"] == (
        SERVICE_MPC_SETPOINT_RUN_EVIDENCE_FORMAT
    )
    assert "no controller-internal fallback" in str(descriptor["failure_policy"])
    assert "expected-plan evidence" in str(descriptor["support_semantics"])
    assert "private weather truth" in str(descriptor["information_set"])
    json.dumps(descriptor, allow_nan=False, sort_keys=True)


def test_controller_rejects_historical_settings_identity() -> None:
    with pytest.raises(TypeError, match="ServiceMPCSetpointSettingsV1"):
        ServiceMPCSetpointControllerV1(ServiceMPCSettingsV1())  # type: ignore[arg-type]


def test_first_decision_is_hash_bound_and_closes_with_physical_execution() -> None:
    env, controller = _reset_pair()
    descriptor = controller.to_dict()
    decision = _compute(env, controller)
    event_before = controller.decision_events[0]

    assert isinstance(decision, MPCSetpointDecisionV1)
    assert isinstance(event_before, MPCSetpointDecisionEvidenceV1)
    assert controller.has_pending_transition
    assert event_before.setpoints == decision.setpoints
    assert event_before.settings_sha256 == SHORT_SETTINGS.sha256
    assert event_before.setpoints.plan_sha256 == event_before.plan_sha256
    assert event_before.setpoints.model_sha256 == event_before.model_sha256
    assert event_before.setpoints.projection_sha256 == event_before.projection_sha256
    assert all(stage.solution_selection is not None for stage in event_before.stages)
    assert event_before.execution is None
    assert event_before.residual is None
    open_sha = event_before.sha256

    info = _apply_and_close(env, controller, decision)
    event_after = controller.decision_events[0]

    assert controller.to_dict() == descriptor
    assert isinstance(event_after, MPCSetpointDecisionEvidenceV1)
    assert event_after.closed
    assert not controller.has_pending_transition
    assert event_after.execution == env.last_setpoint_execution
    assert event_after.execution is not None
    assert event_after.execution.to_dict() == info["setpoint_execution"]
    assert event_after.residual is not None
    assert event_after.residual.next_observation_sha256 == public_observation_sha256(
        env.controller_observation
    )
    assert event_after.sha256 != open_sha
    json.dumps(event_after.to_dict(), allow_nan=False, sort_keys=True)


def test_identical_public_input_produces_identical_plan_command_and_identity() -> None:
    env, first = _reset_pair()
    second = ServiceMPCSetpointControllerV1(SHORT_SETTINGS)
    observation = env.controller_observation

    first_decision = first.compute_setpoints(
        observation,
        constraints=env.constraints,
        config=env.config,
        commitment_profile=env.episode_config.electrolyser_commitment,
    )
    second_decision = second.compute_setpoints(
        observation,
        constraints=env.constraints,
        config=env.config,
        commitment_profile=env.episode_config.electrolyser_commitment,
    )
    first_event = first.decision_events[0]
    second_event = second.decision_events[0]
    assert isinstance(first_event, MPCSetpointDecisionEvidenceV1)
    assert isinstance(second_event, MPCSetpointDecisionEvidenceV1)

    assert first_decision == second_decision
    assert first_event.model_sha256 == second_event.model_sha256
    assert first_event.plan_sha256 == second_event.plan_sha256
    assert first_event.setpoints.sha256 == second_event.setpoints.sha256
    assert first_event.sha256 == second_event.sha256
    assert first_event.decision_elapsed_ns > 0
    assert second_event.decision_elapsed_ns > 0


def test_private_episode_identity_cannot_change_same_public_setpoint() -> None:
    left_env, left = _reset_pair(seed=0)
    right_env, right = _reset_pair(seed=97)
    assert left_env._episode is not None
    assert right_env._episode is not None
    assert (
        left_env._episode.exogenous_episode_sha256
        != right_env._episode.exogenous_episode_sha256
    )
    shared = left_env.controller_observation

    left_decision = left.compute_setpoints(
        shared,
        constraints=left_env.constraints,
        config=left_env.config,
        commitment_profile=left_env.episode_config.electrolyser_commitment,
    )
    right_decision = right.compute_setpoints(
        shared,
        constraints=right_env.constraints,
        config=right_env.config,
        commitment_profile=right_env.episode_config.electrolyser_commitment,
    )

    assert left_decision.setpoints == right_decision.setpoints
    assert left.decision_events[0].sha256 == right.decision_events[0].sha256


def test_new_identity_preserves_initial_frozen_plan_quantities() -> None:
    """Expose rather than hide the protocol gate-9 versioning contradiction."""

    pytest.importorskip("scipy")
    env = PlantEnvV2(episode=EpisodeConfigV2(horizon_h=72.0, delivery_rate_kg_h=35.0))
    env.reset(seed=0)
    observation = env.controller_observation
    historical = ServiceMPCControllerV1()
    setpoint = ServiceMPCSetpointControllerV1()

    historical.compute_action(
        observation,
        constraints=env.constraints,
        config=env.config,
        commitment_profile=env.episode_config.electrolyser_commitment,
    )
    setpoint.compute_setpoints(
        observation,
        constraints=env.constraints,
        config=env.config,
        commitment_profile=env.episode_config.electrolyser_commitment,
    )
    historical_event = historical.decision_events[0]
    setpoint_event = setpoint.decision_events[0]
    assert isinstance(historical_event, MPCDecisionEvidenceV1)
    assert isinstance(setpoint_event, MPCSetpointDecisionEvidenceV1)

    assert setpoint_event.observation_sha256 == (
        "d1d2b3de1cf6314d5cfe857c3f49f7fe48744929853439fa77ebd07b0531ec1c"
    )
    assert historical_event.plan_sha256 == (
        "782220137621a4578abac42ebd36800b028d22f6aea25f86e2c8dbe93b6d08de"
    )
    assert setpoint_event.model_sha256 == (
        "9fc997f369fc7d83d50b5ade834e4ef63bbb1f720a4aab47923a1d997821d120"
    )
    assert setpoint_event.plan_sha256 == (
        "a26f020ef6e55b4a225f796d940d7329f235db16731e07ce5310362d83b8dbcc"
    )
    assert setpoint_event.setpoints.sha256 == (
        "6066dc1d012b7466eb34eb32baba8b58cdd1b909f6743d5f753c7fd0d1531cdf"
    )
    assert setpoint_event.first_step == historical_event.first_step
    assert setpoint_event.settings_sha256 != historical_event.settings_sha256
    assert setpoint_event.model_sha256 != historical_event.model_sha256
    assert setpoint_event.plan_sha256 != historical_event.plan_sha256


def test_complete_short_episode_closes_every_plan_without_fallback() -> None:
    env, controller = _reset_pair()
    infos: list[dict[str, object]] = []
    finished = False
    while not finished:
        decision = _compute(env, controller)
        observation, _, terminated, truncated, info = env.step_setpoints(
            decision.setpoints
        )
        del observation
        execution = env.last_setpoint_execution
        assert isinstance(execution, MPCSetpointIntervalExecutionV1)
        controller.observe_transition(
            env.controller_observation,
            execution=execution,
        )
        infos.append(info)
        finished = terminated or truncated

    events = controller.decision_events
    assert len(events) == 4
    assert all(isinstance(event, MPCSetpointDecisionEvidenceV1) for event in events)
    assert all(
        event.closed
        for event in events
        if isinstance(event, MPCSetpointDecisionEvidenceV1)
    )
    assert [
        event.horizon_intervals
        for event in events
        if isinstance(event, MPCSetpointDecisionEvidenceV1)
    ] == [4, 3, 2, 1]
    assert all(
        info["setpoint_execution"]["instantaneous_electrolyser_target_error_kw"][
            "requested_max_overshoot"
        ]
        == 0.0
        for info in infos
    )
    assert env.controller_observation.progress == 1.0
    assert not controller.has_pending_transition
    json.dumps(controller.evidence(), allow_nan=False, sort_keys=True)


def test_transition_rejects_execution_from_another_command() -> None:
    env, controller = _reset_pair(seed=0)
    foreign_env, foreign_controller = _reset_pair(seed=1)
    decision = _compute(env, controller)
    foreign_decision = _compute(foreign_env, foreign_controller)
    env.step_setpoints(decision.setpoints)
    foreign_env.step_setpoints(foreign_decision.setpoints)
    foreign_execution = foreign_env.last_setpoint_execution
    assert isinstance(foreign_execution, MPCSetpointIntervalExecutionV1)

    with pytest.raises(ValueError, match="another setpoint command"):
        controller.observe_transition(
            env.controller_observation,
            execution=foreign_execution,
        )

    assert controller.has_pending_transition


def test_decision_evidence_rejects_identity_stage_and_closure_substitution() -> None:
    env, controller = _reset_pair()
    decision = _compute(env, controller)
    _apply_and_close(env, controller, decision)
    event = controller.decision_events[0]
    assert isinstance(event, MPCSetpointDecisionEvidenceV1)
    assert event.residual is not None

    with pytest.raises(ValueError, match="plan identity"):
        replace(
            event,
            setpoints=replace(event.setpoints, plan_sha256="0" * 64),
        )
    with pytest.raises(ValueError, match="selection evidence"):
        replace(
            event,
            stages=(
                replace(event.stages[0], solution_selection=None),
                event.stages[1],
            ),
        )
    with pytest.raises(ValueError, match="close together"):
        replace(event, execution=None)


def test_missing_forecast_records_typed_failure_and_emits_no_command() -> None:
    env, controller = _reset_pair()

    with pytest.raises(ValueError, match="current public forecast"):
        controller.compute_setpoints(
            env.controller_observation.without_forecast(),
            constraints=env.constraints,
            config=env.config,
            commitment_profile=env.episode_config.electrolyser_commitment,
        )

    assert not controller.has_pending_transition
    failure = controller.decision_events[0]
    assert isinstance(failure, MPCSetpointDecisionFailureV1)
    assert failure.phase == "input_validation"
    assert failure.observation_sha256 is None
    assert failure.model_sha256 is None
    assert failure.setpoint_sha256 is None


def test_solver_failure_retains_attempted_stage_and_has_stable_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, controller = _reset_pair()
    failed_stage = MPCStageSolveV1(
        stage="primary_delivery",
        success=False,
        status=1,
        message="deliberate timeout fixture",
        elapsed_ns=123,
        objective_value=None,
        mip_dual_bound=None,
        mip_gap=None,
        mip_node_count=None,
        solution_sha256=None,
        scipy_version="test-scipy",
        highs_version="test-highs",
    )

    def fail_solver(*_: object, **__: object) -> NoReturn:
        raise ServiceMPCSolveError(
            "deliberate timeout fixture",
            stage_evidence=failed_stage,
        )

    monkeypatch.setattr(controller_module, "solve_service_mpc_model", fail_solver)
    with pytest.raises(ServiceMPCSolveError, match="deliberate timeout"):
        _compute(env, controller)

    failure = controller.decision_events[0]
    assert isinstance(failure, MPCSetpointDecisionFailureV1)
    assert failure.phase == "solver"
    assert failure.observation_sha256 is not None
    assert failure.public_forecast_sha256 is not None
    assert failure.projection_sha256 is not None
    assert failure.model_sha256 is not None
    assert failure.plan_sha256 is None
    assert failure.stages == (failed_stage,)
    assert replace(failure, elapsed_ns=failure.elapsed_ns + 1).sha256 == failure.sha256


def test_next_solve_requires_explicit_physical_transition_closure() -> None:
    env, controller = _reset_pair()
    _compute(env, controller)

    with pytest.raises(ValueError, match="must be reconciled"):
        _compute(env, controller)

    assert controller.has_pending_transition
    assert len(controller.decision_events) == 2
    assert isinstance(controller.decision_events[0], MPCSetpointDecisionEvidenceV1)
    failure = controller.decision_events[1]
    assert isinstance(failure, MPCSetpointDecisionFailureV1)
    assert failure.phase == "transition_reconciliation"

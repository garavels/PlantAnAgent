"""Causal, deterministic, and exact-transition tests for service MPC."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import NoReturn

import pytest
from plant_sim import Action

from plant_backend.controllers.base import Constraints, ControlDecision
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.control import ControllerV2
from plant_backend.rl.v2.env import PlantEnvV2
from plant_backend.rl.v2.mpc_controller import ServiceMPCControllerV1
from plant_backend.rl.v2.mpc_execution import (
    MPCActionRealizationV2,
    MPCDecisionEvidenceV1,
    MPCDecisionFailureV2,
    SERVICE_MPC_ACTION_SCHEMA,
)
from plant_backend.rl.v2.mpc_schema import (
    MPCStageSolveV1,
    ServiceMPCPlanError,
    ServiceMPCSettingsV1,
    ServiceMPCSolveError,
    canonical_json,
)
import plant_backend.rl.v2.mpc_controller as mpc_controller_module
from plant_backend.safety.validation import validate


SHORT_EPISODE = EpisodeConfigV2(
    horizon_h=1.0,
    delivery_period_h=1.0,
    delivery_rate_kg_h=35.0,
)
SHORT_SETTINGS = ServiceMPCSettingsV1(
    task_horizon_h=1.0,
    planning_horizon_h=1.0,
)


def _reset_pair(
    *,
    start_hour: float = 6.0,
    seed: int = 0,
) -> tuple[PlantEnvV2, ServiceMPCControllerV1]:
    pytest.importorskip("scipy")
    episode = replace(SHORT_EPISODE, start_hour=start_hour)
    env = PlantEnvV2(episode=episode)
    env.reset(seed=seed)
    return env, ServiceMPCControllerV1(SHORT_SETTINGS)


def _compute(
    env: PlantEnvV2,
    controller: ServiceMPCControllerV1,
) -> ControlDecision:
    return controller.compute_action(
        env.controller_observation,
        constraints=env.constraints,
        config=env.config,
        commitment_profile=env.episode_config.electrolyser_commitment,
    )


def test_controller_descriptor_is_immutable_and_declares_no_fallback() -> None:
    controller = ServiceMPCControllerV1(SHORT_SETTINGS)
    descriptor = controller.to_dict()

    assert isinstance(controller, ControllerV2)
    assert descriptor["id"] == "service-mpc-issued-mean-v1"
    assert descriptor["uses_forecast"] is True
    assert descriptor["settings"] == SHORT_SETTINGS.to_dict()
    failure_policy = descriptor["failure_policy"]
    information_set = descriptor["information_set"]
    assert isinstance(failure_policy, str)
    assert isinstance(information_set, str)
    assert "no controller-internal fallback" in failure_policy
    assert "private weather truth" in information_set
    assert descriptor["decision_failure_schema"] == "service-mpc-decision-failure-v2"
    assert descriptor["run_evidence_format"] == "service-mpc-controller-run-evidence-v2"
    assert descriptor["action_realization_schema"] == SERVICE_MPC_ACTION_SCHEMA
    assert "separately evidenced" in descriptor["support_semantics"]
    json.dumps(descriptor, allow_nan=False, sort_keys=True)


def test_first_decision_is_valid_and_fully_hash_bound() -> None:
    env, controller = _reset_pair()
    descriptor = controller.to_dict()
    decision = _compute(env, controller)

    assert controller.to_dict() == descriptor
    assert isinstance(decision.action, Action)
    assert Action.sanitised(decision.action) == decision.action
    assert not decision.invalid
    assert len(decision.notes) == 3
    assert controller.has_pending_transition
    assert len(controller.decision_events) == 1
    event = controller.decision_events[0]
    assert isinstance(event, MPCDecisionEvidenceV1)
    assert isinstance(event.action, MPCActionRealizationV2)
    assert event.decision_index == 0
    assert event.horizon_intervals == 4
    assert event.settings_sha256 == SHORT_SETTINGS.sha256
    assert event.action.action == decision.action
    assert event.action.plan_sha256 == event.plan_sha256
    assert event.action.to_dict()["schema"] == SERVICE_MPC_ACTION_SCHEMA
    assert (
        event.action.planned_battery_support_kw == event.first_step.battery_support_kw
    )
    assert event.action.battery_support_authority_kw == 0.0
    assert event.action.action.battery_support_kw == 0.0
    assert event.first_step.index == 0
    assert event.residual is None
    assert all(stage.success for stage in event.stages)
    assert len(event.sha256) == 64
    json.dumps(event.to_dict(), allow_nan=False, sort_keys=True)


def test_identical_public_inputs_produce_identical_plan_action_and_evidence_identity() -> (
    None
):
    env, first = _reset_pair()
    second = ServiceMPCControllerV1(SHORT_SETTINGS)
    observation = env.controller_observation

    first_decision = first.compute_action(
        observation,
        constraints=env.constraints,
        config=env.config,
        commitment_profile=env.episode_config.electrolyser_commitment,
    )
    second_decision = second.compute_action(
        observation,
        constraints=env.constraints,
        config=env.config,
        commitment_profile=env.episode_config.electrolyser_commitment,
    )
    first_event = first.decision_events[0]
    second_event = second.decision_events[0]
    assert isinstance(first_event, MPCDecisionEvidenceV1)
    assert isinstance(second_event, MPCDecisionEvidenceV1)

    assert first_decision == second_decision
    assert first_event.model_sha256 == second_event.model_sha256
    assert first_event.plan_sha256 == second_event.plan_sha256
    assert first_event.action.sha256 == second_event.action.sha256
    assert first_event.sha256 == second_event.sha256
    assert first_event.decision_elapsed_ns > 0
    assert second_event.decision_elapsed_ns > 0


def test_private_truth_seed_and_future_vintages_cannot_change_same_public_request() -> (
    None
):
    left_env, left = _reset_pair(seed=0)
    right_env, right = _reset_pair(seed=97)
    left_private = left_env._episode
    right_private = right_env._episode
    assert left_private is not None
    assert right_private is not None
    assert (
        left_private.exogenous_episode_sha256 != right_private.exogenous_episode_sha256
    )
    assert left_private.weather.truth_sha256 != right_private.weather.truth_sha256
    assert (
        left_private.weather.forecasts_sha256 != right_private.weather.forecasts_sha256
    )
    assert (
        left_private.weather.issues[1]._issue_stream_sha256
        != right_private.weather.issues[1]._issue_stream_sha256
    )

    shared_public_input = left_env.controller_observation
    left_decision = left.compute_action(
        shared_public_input,
        constraints=left_env.constraints,
        config=left_env.config,
        commitment_profile=left_env.episode_config.electrolyser_commitment,
    )
    right_decision = right.compute_action(
        shared_public_input,
        constraints=right_env.constraints,
        config=right_env.config,
        commitment_profile=right_env.episode_config.electrolyser_commitment,
    )
    left_event = left.decision_events[0]
    right_event = right.decision_events[0]
    assert isinstance(left_event, MPCDecisionEvidenceV1)
    assert isinstance(right_event, MPCDecisionEvidenceV1)

    assert canonical_json(left_decision.action.to_dict()) == canonical_json(
        right_decision.action.to_dict()
    )
    assert left_event.projection_sha256 == right_event.projection_sha256
    assert left_event.model_sha256 == right_event.model_sha256
    assert left_event.plan_sha256 == right_event.plan_sha256
    assert left_event.action.sha256 == right_event.action.sha256
    assert left_event.sha256 == right_event.sha256


def test_complete_short_episode_reconciles_every_plan_with_public_plant_truth() -> None:
    env, controller = _reset_pair()
    descriptor = controller.to_dict()
    infos: list[dict[str, object]] = []
    finished = False
    while not finished:
        decision = _compute(env, controller)
        _, _, terminated, truncated, info = env.step_action(decision.action)
        infos.append(info)
        controller.observe_transition(env.controller_observation)
        finished = terminated or truncated

    assert controller.to_dict() == descriptor
    assert not controller.has_pending_transition
    assert len(controller.decision_events) == 4
    events = tuple(
        event
        for event in controller.decision_events
        if isinstance(event, MPCDecisionEvidenceV1)
    )
    assert len(events) == 4
    assert [event.horizon_intervals for event in events] == [
        4,
        3,
        2,
        1,
    ]
    for event, info in zip(events, infos, strict=True):
        residual = event.residual
        assert residual is not None
        interval_power = info["interval_power"]
        assert isinstance(interval_power, dict)
        applied = interval_power["applied_mean_kw"]
        assert isinstance(applied, dict)
        assert residual.observed_electrolyser_kw == pytest.approx(
            applied["electrolyser"]
        )
        assert residual.observed_dac_kw == pytest.approx(applied["dac"])
        assert residual.observed_methanator_kw == pytest.approx(applied["methanator"])
        assert residual.process_power_l1_error_kwh >= 0.0
        assert len(residual.sha256) == 64
    assert env.controller_observation.progress == 1.0
    json.dumps(controller.evidence(), allow_nan=False, sort_keys=True)


def test_one_interval_execution_closes_the_first_plan_against_public_truth() -> None:
    env, controller = _reset_pair()
    decision = _compute(env, controller)
    event_before = controller.decision_events[0]
    assert isinstance(event_before, MPCDecisionEvidenceV1)
    assert event_before.residual is None

    _, _, _, _, info = env.step_action(decision.action)
    controller.observe_transition(env.controller_observation)

    event_after = controller.decision_events[0]
    assert isinstance(event_after, MPCDecisionEvidenceV1)
    residual = event_after.residual
    assert residual is not None
    interval_power = info["interval_power"]
    assert isinstance(interval_power, dict)
    applied = interval_power["applied_mean_kw"]
    assert isinstance(applied, dict)
    assert residual.observed_battery_net_kw == pytest.approx(
        applied["battery"] - applied["battery_support"]
    )
    assert residual.observed_electrolyser_kw == pytest.approx(applied["electrolyser"])
    assert residual.observed_dac_kw == pytest.approx(applied["dac"])
    assert residual.observed_methanator_kw == pytest.approx(applied["methanator"])
    assert not controller.has_pending_transition


def test_registered_seed_one_first_transition_has_non_negative_curtailment() -> None:
    pytest.importorskip("scipy")
    env = PlantEnvV2(
        episode=EpisodeConfigV2(
            horizon_h=72.0,
            delivery_rate_kg_h=35.0,
        )
    )
    env.reset(seed=1)
    controller = ServiceMPCControllerV1()

    decision = _compute(env, controller)
    env.step_action(decision.action)
    controller.observe_transition(env.controller_observation)

    event = controller.decision_events[0]
    assert isinstance(event, MPCDecisionEvidenceV1)
    assert event.residual is not None
    assert event.residual.observed_curtailed_kw == 0.0
    assert not controller.has_pending_transition


def test_commitment_aware_support_preserves_registered_seed_zero_cold_start() -> None:
    pytest.importorskip("scipy")
    env = PlantEnvV2(
        episode=EpisodeConfigV2(
            horizon_h=72.0,
            delivery_rate_kg_h=35.0,
        )
    )
    env.reset(seed=0)
    controller = ServiceMPCControllerV1()

    # The first two requests do not target the electrolyser and must therefore
    # remain identical to the frozen 3796eb5 plans/actions.
    frozen_prefix = (
        (
            "782220137621a4578abac42ebd36800b028d22f6aea25f86e2c8dbe93b6d08de",
            Action(
                battery_share=0.0,
                electrolyser_share=0.0,
                dac_share=0.0,
                methanator_share=1.0,
                battery_support_kw=0.0,
            ),
        ),
        (
            "ea115d2fbb37894c46684c88c8bc8a8dc3fa6a4ace2962fe4521610f8f5c15d4",
            Action(
                battery_share=0.0,
                electrolyser_share=0.0,
                dac_share=0.4336130390869113,
                methanator_share=0.5663869609130887,
                battery_support_kw=0.0,
            ),
        ),
    )
    for plan_sha256, frozen_action in frozen_prefix:
        decision = _compute(env, controller)
        event = controller.decision_events[-1]
        assert isinstance(event, MPCDecisionEvidenceV1)
        assert event.plan_sha256 == plan_sha256
        assert event.action.battery_support_authority_kw == 0.0
        assert decision.action == frozen_action
        env.step_action(decision.action)
        controller.observe_transition(env.controller_observation)

    decision = _compute(env, controller)
    event = controller.decision_events[-1]
    assert isinstance(event, MPCDecisionEvidenceV1)
    action = event.action

    # These identities are from the pre-bridge artifact.  They prove the public
    # boundary, forecast projection, MILP and selected plan did not change.
    assert event.observation_sha256 == (
        "3f9bbe6cc62ec3dfaaf486c20a818ee90bed95984ffe9e7bc9e4708f8dee5aac"
    )
    assert event.model_sha256 == (
        "25c5505091125fd36d5b8a79d5691e843cfcf12063a6280bc14392ef2dd56355"
    )
    assert event.plan_sha256 == (
        "36af91aee783fe89979a92f4fb0e62b2f0dd31c04b09467d5c4403378cd2054f"
    )
    assert action.electrolyser_target_kw == 600.0
    assert action.planned_battery_support_kw == pytest.approx(68.15390458875038)
    assert action.battery_support_authority_kw == 600.0
    assert decision.action.electrolyser_share == 1.0
    assert decision.action.battery_support_kw == 600.0

    # The unchanged reserve guard can still remove the new authority and reports
    # the intervention.  The actual episode uses its ordinary 15% reserve.
    boundary = env.controller_observation.state.boundary
    blocked = validate(
        decision.action,
        boundary,
        constraints=Constraints(battery_min_soc=boundary.battery_soc),
        config=env.config,
        dispatch_solar_available_kw=env.simulator.next_dispatch_solar_kw(),
    )
    assert blocked.action.battery_support_kw == 0.0
    assert any("reserve" in adjustment for adjustment in blocked.adjustments)

    # Four held decisions cover the exact one-hour cold start.  Contingent
    # authority fills only the minute-level uncovered gap, so the plant accepts
    # the target without weakening the one-minute commitment kernel.
    _, _, _, _, start_info = env.step_action(decision.action)
    controller.observe_transition(env.controller_observation)
    first_start_event = controller.decision_events[2]
    assert isinstance(first_start_event, MPCDecisionEvidenceV1)
    assert first_start_event.residual is not None
    assert first_start_event.residual.observed_electrolyser_kw == pytest.approx(600.0)
    interval_power = start_info["interval_power"]
    assert isinstance(interval_power, dict)
    guarded_kwh = interval_power["guarded_kwh"]
    applied_kwh = interval_power["applied_kwh"]
    assert isinstance(guarded_kwh, dict)
    assert isinstance(applied_kwh, dict)
    assert guarded_kwh["battery_support"] == pytest.approx(150.0)
    assert 0.0 < applied_kwh["battery_support"] < guarded_kwh["battery_support"]
    assert applied_kwh["electrolyser"] == pytest.approx(150.0)
    for _ in range(3):
        continued = _compute(env, controller)
        continued_event = controller.decision_events[-1]
        assert isinstance(continued_event, MPCDecisionEvidenceV1)
        assert continued_event.action.battery_support_authority_kw == 600.0
        env.step_action(continued.action)
        controller.observe_transition(env.controller_observation)

    snapshot = env.controller_observation.state.electrolyser_commitment
    assert snapshot is not None
    assert snapshot.mode == "running"
    assert snapshot.cold_starts == 1
    assert snapshot.aborted_starts == 0
    assert snapshot.forced_stops == 0


def test_next_compute_auto_reconciles_an_unobserved_pending_transition() -> None:
    env, controller = _reset_pair()
    first = _compute(env, controller)
    env.step_action(first.action)

    second = _compute(env, controller)

    assert isinstance(second.action, Action)
    first_event = controller.decision_events[0]
    assert isinstance(first_event, MPCDecisionEvidenceV1)
    assert first_event.residual is not None
    assert len(controller.decision_events) == 2
    assert controller.has_pending_transition


def test_dark_first_interval_emits_zero_solar_shares() -> None:
    env, controller = _reset_pair(start_hour=20.0)
    decision = _compute(env, controller)
    event = controller.decision_events[0]
    assert isinstance(event, MPCDecisionEvidenceV1)

    assert event.first_step.forecast_solar_kw == 0.0
    assert decision.action.battery_share == 0.0
    assert decision.action.electrolyser_share == 0.0
    assert decision.action.dac_share == 0.0
    assert decision.action.methanator_share == 0.0
    assert Action.sanitised(decision.action) == decision.action


def test_missing_forecast_records_failure_and_emits_no_action() -> None:
    env, controller = _reset_pair()
    observation = env.controller_observation.without_forecast()

    with pytest.raises(ValueError, match="current public forecast"):
        controller.compute_action(
            observation,
            constraints=env.constraints,
            config=env.config,
            commitment_profile=env.episode_config.electrolyser_commitment,
        )
    assert not controller.has_pending_transition
    assert len(controller.decision_events) == 1
    failure = controller.decision_events[0]
    assert isinstance(failure, MPCDecisionFailureV2)
    assert failure.phase == "input_validation"
    assert failure.observation_sha256 is None
    assert failure.model_sha256 is None
    assert failure.plan_sha256 is None


def test_wrong_task_progress_fails_before_projection_or_solver() -> None:
    env, controller = _reset_pair()
    observation = replace(env.controller_observation, progress=0.1)

    with pytest.raises(ValueError, match="task horizon"):
        controller.compute_action(
            observation,
            constraints=env.constraints,
            config=env.config,
            commitment_profile=env.episode_config.electrolyser_commitment,
        )
    failure = controller.decision_events[0]
    assert isinstance(failure, MPCDecisionFailureV2)
    assert failure.phase == "input_validation"
    assert failure.projection_sha256 is None


def test_solver_failure_event_retains_public_model_and_stage_evidence(
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

    monkeypatch.setattr(
        mpc_controller_module,
        "solve_service_mpc_model",
        fail_solver,
    )
    with pytest.raises(ServiceMPCSolveError, match="deliberate timeout"):
        _compute(env, controller)
    failure = controller.decision_events[0]
    assert isinstance(failure, MPCDecisionFailureV2)
    assert failure.phase == "solver"
    assert failure.observation_sha256 is not None
    assert failure.public_forecast_sha256 is not None
    assert failure.projection_sha256 is not None
    assert failure.model_sha256 is not None
    assert failure.plan_sha256 is None
    assert failure.stages == (failed_stage,)
    assert failure.stage_evidence == failed_stage


def test_first_action_requiring_sanitizer_repair_fails_closed_with_plan_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, controller = _reset_pair()

    def corrupt_sanitizer(cls: type[Action], action: Action) -> Action:
        del cls, action
        return Action(
            battery_share=0.0,
            electrolyser_share=0.0,
            dac_share=0.0,
            methanator_share=0.0,
            battery_support_kw=123_456.0,
        )

    monkeypatch.setattr(Action, "sanitised", classmethod(corrupt_sanitizer))
    with pytest.raises(ServiceMPCPlanError, match="sanitizer correction"):
        _compute(env, controller)

    assert not controller.has_pending_transition
    assert len(controller.decision_events) == 1
    failure = controller.decision_events[0]
    assert isinstance(failure, MPCDecisionFailureV2)
    assert failure.phase == "action_realization"
    assert failure.plan_sha256 is not None
    assert len(failure.stages) == 2
    assert all(stage.success for stage in failure.stages)


def test_repeated_compute_without_advancing_fails_transition_reconciliation() -> None:
    env, controller = _reset_pair()
    _compute(env, controller)

    with pytest.raises(ValueError, match="not contiguous"):
        _compute(env, controller)
    assert len(controller.decision_events) == 2
    assert isinstance(controller.decision_events[0], MPCDecisionEvidenceV1)
    failure = controller.decision_events[1]
    assert isinstance(failure, MPCDecisionFailureV2)
    assert failure.phase == "transition_reconciliation"

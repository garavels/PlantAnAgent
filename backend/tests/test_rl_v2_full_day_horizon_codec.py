"""Detached semantic admission tests for full-day setpoint-MPC evidence."""

from __future__ import annotations

from copy import deepcopy
import json

import pytest

from plant_sim import Action, ElectrolyserCommitmentSnapshot

from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.env import PlantEnvV2
from plant_backend.rl.v2.full_day_horizon_codec import (
    decode_mpc_setpoint_event,
    decode_mpc_setpoint_run,
    verify_mpc_setpoint_run_semantics,
)
from plant_backend.rl.v2.mpc_execution import public_observation_sha256
from plant_backend.rl.v2.mpc_schema import (
    MPCStageSolveV1,
    ServiceMPCSetpointSettingsV1,
    domain_sha256,
)
from plant_backend.rl.v2.mpc_setpoint_controller import (
    ServiceMPCSetpointControllerV1,
)
from plant_backend.rl.v2.mpc_setpoint_decision import (
    MPCSetpointDecisionEvidenceV1,
    MPCSetpointDecisionFailureV1,
)
from plant_backend.rl.v2.mpc_setpoint_execution import (
    MPCMinuteApplicationV1,
    MPCSetpointIntervalExecutionV1,
)
from plant_backend.rl.v2.mpc_setpoint_run import (
    SERVICE_MPC_SETPOINT_EVIDENCE_GRAPH_DOMAIN,
    SERVICE_MPC_SETPOINT_RUN_EVIDENCE_DOMAIN,
    build_mpc_setpoint_run_evidence,
    verify_mpc_setpoint_run_digests,
)
from plant_backend.rl.v2.mpc_setpoints import MPCMinuteDispatchV1


_EPISODE = EpisodeConfigV2(
    horizon_h=1.0,
    delivery_period_h=1.0,
    delivery_rate_kg_h=35.0,
)
_SETTINGS = ServiceMPCSetpointSettingsV1(
    task_horizon_h=1.0,
    planning_horizon_h=1.0,
)
_GRAPH_KEYS = (
    "format",
    "controller_id",
    "settings_sha256",
    "status",
    "expected_decisions",
    "completed_decisions",
    "runner_failure_phase",
    "pending_transition",
    "final_observation_sha256",
    "counts",
    "energy_kwh",
    "target_tracking",
    "target_deficit_attribution_kwh",
    "event_sha256",
    "execution_sha256",
    "application_sha256",
)


@pytest.fixture(scope="module")
def complete_evidence() -> dict[str, object]:
    pytest.importorskip("scipy")
    env = PlantEnvV2(episode=_EPISODE)
    env.reset(seed=905)
    controller = ServiceMPCSetpointControllerV1(_SETTINGS)
    dispatcher_elapsed: list[tuple[int, ...]] = []
    finished = False
    while not finished:
        decision = controller.compute_setpoints(
            env.controller_observation,
            constraints=env.constraints,
            config=env.config,
            commitment_profile=env.episode_config.electrolyser_commitment,
        )
        _, _, terminated, truncated, info = env.step_setpoints(decision.setpoints)
        execution = env.last_setpoint_execution
        assert isinstance(execution, MPCSetpointIntervalExecutionV1)
        controller.observe_transition(
            env.controller_observation,
            execution=execution,
        )
        dispatcher_elapsed.append(tuple(info["dispatcher_elapsed_ns"]))
        finished = terminated or truncated

    events = controller.decision_events
    assert all(isinstance(item, MPCSetpointDecisionEvidenceV1) for item in events)
    return build_mpc_setpoint_run_evidence(
        events,
        tuple(dispatcher_elapsed),
        settings=_SETTINGS,
        completed_decisions=len(events),
        run_succeeded=True,
        pending_transition=False,
        runner_failure_phase=None,
        final_observation_sha256=public_observation_sha256(env.controller_observation),
    )


def _coherently_rehash_outer(
    evidence: dict[str, object],
    *,
    graph_changed: bool = False,
) -> None:
    if graph_changed:
        evidence["evidence_graph_sha256"] = domain_sha256(
            SERVICE_MPC_SETPOINT_EVIDENCE_GRAPH_DOMAIN,
            {key: evidence[key] for key in _GRAPH_KEYS},
        )
    body = dict(evidence)
    body.pop("sha256", None)
    evidence["sha256"] = domain_sha256(
        SERVICE_MPC_SETPOINT_RUN_EVIDENCE_DOMAIN,
        body,
    )


def test_complete_run_round_trips_every_typed_layer(
    complete_evidence: dict[str, object],
) -> None:
    detached = json.loads(json.dumps(complete_evidence, allow_nan=False))
    rebuilt = verify_mpc_setpoint_run_semantics(detached, settings=_SETTINGS)
    assert rebuilt == detached
    assert rebuilt is not detached

    events, dispatcher_elapsed = decode_mpc_setpoint_run(
        detached,
        settings=_SETTINGS,
    )
    assert len(events) == _SETTINGS.task_interval_count
    assert len(dispatcher_elapsed) == _SETTINGS.task_interval_count
    first = events[0]
    assert isinstance(first, MPCSetpointDecisionEvidenceV1)
    assert first.to_dict() == detached["events"][0]
    assert isinstance(first.execution, MPCSetpointIntervalExecutionV1)
    application = first.execution.applications[0]
    assert isinstance(application, MPCMinuteApplicationV1)
    assert isinstance(application.dispatch, MPCMinuteDispatchV1)
    assert isinstance(application.dispatch.action, Action)
    assert isinstance(application.guarded_action, Action)
    if application.commitment_before is not None:
        assert isinstance(application.commitment_before, ElectrolyserCommitmentSnapshot)
    if application.commitment_after is not None:
        assert isinstance(application.commitment_after, ElectrolyserCommitmentSnapshot)


def test_failure_event_and_run_round_trip_without_solver_fallback() -> None:
    env = PlantEnvV2(episode=_EPISODE)
    env.reset(seed=906)
    controller = ServiceMPCSetpointControllerV1(_SETTINGS)
    with pytest.raises(ValueError, match="current public forecast"):
        controller.compute_setpoints(
            env.controller_observation.without_forecast(),
            constraints=env.constraints,
            config=env.config,
            commitment_profile=env.episode_config.electrolyser_commitment,
        )
    event = controller.decision_events[0]
    assert isinstance(event, MPCSetpointDecisionFailureV1)
    raw_event = event.to_dict()
    assert decode_mpc_setpoint_event(raw_event) == event

    evidence = build_mpc_setpoint_run_evidence(
        (event,),
        (),
        settings=_SETTINGS,
        completed_decisions=0,
        run_succeeded=False,
        pending_transition=False,
        runner_failure_phase="controller",
        final_observation_sha256=public_observation_sha256(env.controller_observation),
    )
    assert verify_mpc_setpoint_run_semantics(evidence, settings=_SETTINGS) == evidence


def test_failed_solver_stage_is_rebuilt_as_typed_failure_evidence() -> None:
    stage = MPCStageSolveV1(
        stage="primary_delivery",
        success=False,
        status=1,
        message="fabricated bounded solver refusal",
        elapsed_ns=17,
        objective_value=None,
        mip_dual_bound=None,
        mip_gap=None,
        mip_node_count=None,
        solution_sha256=None,
        scipy_version="fabricated-test-runtime",
        highs_version="fabricated-test-runtime",
    )
    event = MPCSetpointDecisionFailureV1(
        decision_index=0,
        phase="solver",
        exception_type="ServiceMPCSolveError",
        message="fabricated bounded solver refusal",
        elapsed_ns=19,
        settings_sha256=_SETTINGS.sha256,
        observation_sha256="a" * 64,
        public_forecast_sha256="b" * 64,
        projection_sha256="c" * 64,
        model_sha256="d" * 64,
        stages=(stage,),
    )

    decoded = decode_mpc_setpoint_event(event.to_dict())
    assert isinstance(decoded, MPCSetpointDecisionFailureV1)
    assert decoded == event
    assert decoded.stages == (stage,)


def test_coherently_rehashed_aggregate_mutation_is_semantically_rejected(
    complete_evidence: dict[str, object],
) -> None:
    mutated = deepcopy(complete_evidence)
    counts = mutated["counts"]
    assert isinstance(counts, dict)
    counts["fallback_action_count"] = 1
    _coherently_rehash_outer(mutated, graph_changed=True)
    verify_mpc_setpoint_run_digests(mutated)

    with pytest.raises(ValueError, match="exactly rebuild"):
        verify_mpc_setpoint_run_semantics(mutated, settings=_SETTINGS)


def test_coherently_rehashed_dispatcher_latency_mutation_is_recomputed(
    complete_evidence: dict[str, object],
) -> None:
    mutated = deepcopy(complete_evidence)
    latency_groups = mutated["dispatcher_elapsed_ns"]
    assert isinstance(latency_groups, list)
    first_group = latency_groups[0]
    assert isinstance(first_group, list)
    first_group[0] += 1
    _coherently_rehash_outer(mutated)
    verify_mpc_setpoint_run_digests(mutated)

    with pytest.raises(ValueError, match="exactly rebuild"):
        verify_mpc_setpoint_run_semantics(mutated, settings=_SETTINGS)


def test_coherently_rehashed_nested_guard_claim_is_rejected_at_typed_leaf(
    complete_evidence: dict[str, object],
) -> None:
    mutated = deepcopy(complete_evidence)
    events = mutated["events"]
    assert isinstance(events, list)
    application = events[0]["execution"]["applications"][0]
    application["guard"]["changed_request"] = not application["guard"][
        "changed_request"
    ]
    _coherently_rehash_outer(mutated)
    verify_mpc_setpoint_run_digests(mutated)

    with pytest.raises(ValueError, match="minute application.*exactly rebuild"):
        verify_mpc_setpoint_run_semantics(mutated, settings=_SETTINGS)


def test_event_decoder_rejects_unknown_and_non_exact_schemas(
    complete_evidence: dict[str, object],
) -> None:
    events = complete_evidence["events"]
    assert isinstance(events, list)
    unknown = deepcopy(events[0])
    unknown["schema"] = "service-mpc-setpoint-decision-evidence-v999"
    with pytest.raises(ValueError, match="unsupported"):
        decode_mpc_setpoint_event(unknown)

    extra = deepcopy(events[0])
    extra["untyped_extension"] = True
    with pytest.raises(ValueError, match="exactly rebuild"):
        decode_mpc_setpoint_event(extra)


def test_run_decoder_rejects_wrong_settings_before_event_use(
    complete_evidence: dict[str, object],
) -> None:
    wrong = ServiceMPCSetpointSettingsV1(
        task_horizon_h=2.0,
        planning_horizon_h=1.0,
    )
    with pytest.raises(ValueError, match="settings identity"):
        decode_mpc_setpoint_run(complete_evidence, settings=wrong)

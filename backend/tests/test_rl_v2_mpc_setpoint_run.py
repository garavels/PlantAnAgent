"""Frozen identity-equivalence and deep setpoint-MPC run-evidence tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.env import PlantEnvV2
from plant_backend.rl.v2.mpc_execution import (
    public_forecast_issue_sha256,
    public_observation_sha256,
    realize_first_mpc_action,
)
from plant_backend.rl.v2.mpc_forecast import project_issued_mean_solar
from plant_backend.rl.v2.mpc_model import build_service_mpc_model
from plant_backend.rl.v2.mpc_schema import (
    MPCPlanV1,
    ServiceMPCSetpointSettingsV1,
    ServiceMPCSettingsV1,
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
    MPCSetpointIntervalExecutionV1,
)
from plant_backend.rl.v2.mpc_setpoint_run import (
    SERVICE_MPC_SETPOINT_RUN_EVIDENCE_DOMAIN,
    build_mpc_setpoint_run_evidence,
    verify_mpc_setpoint_run_digests,
)
from plant_backend.rl.v2.mpc_setpoints import realize_first_mpc_setpoints
from plant_backend.rl.v2.mpc_solver import solve_service_mpc_model


SHORT_EPISODE = EpisodeConfigV2(
    horizon_h=1.0,
    delivery_period_h=1.0,
    delivery_rate_kg_h=35.0,
)
SHORT_SETTINGS = ServiceMPCSetpointSettingsV1(
    task_horizon_h=1.0,
    planning_horizon_h=1.0,
)

_FROZEN_IDENTITIES = (
    {
        "observation": "d1d2b3de1cf6314d5cfe857c3f49f7fe48744929853439fa77ebd07b0531ec1c",
        "forecast": "d95901492f5041dce9ff73d87944028b51639b2dddbd3a739c0cc888e55477ef",
        "projection": "d0c1e03dc9199267cbdf32d516d3c9f4c2fe009081c35add7a59c47d0d02d917",
        "historical_model": "a0e5b312ffa5ddd055d502ce5205c520dc642cd801e4286ed82924108c5d54d9",
        "historical_plan": "782220137621a4578abac42ebd36800b028d22f6aea25f86e2c8dbe93b6d08de",
        "new_model": "9fc997f369fc7d83d50b5ade834e4ef63bbb1f720a4aab47923a1d997821d120",
        "new_plan": "a26f020ef6e55b4a225f796d940d7329f235db16731e07ce5310362d83b8dbcc",
        "setpoint": "6066dc1d012b7466eb34eb32baba8b58cdd1b909f6743d5f753c7fd0d1531cdf",
    },
    {
        "observation": "00219b326f2655b53fe515bd16cc7971f837950a65e07e3d2f0895883b088fac",
        "forecast": "daad4374d39eed1f88696c6bdb4d5866633cfb2bd4061cb4612ca10bce2ddf12",
        "projection": "493cd521d74d770199559af107c00d52ba654ab5061a29b9f1a52349b850d86f",
        "historical_model": "def36b41b5018d2a5cafaf2a42ef19873ba2a4f9c4d304260e14f1c6d9611356",
        "historical_plan": "ea115d2fbb37894c46684c88c8bc8a8dc3fa6a4ace2962fe4521610f8f5c15d4",
        "new_model": "6ee66de7f236c40849e43edda51691ffcda2cff149bd26ff3c64d4a61fc9e07c",
        "new_plan": "5056690517984e5c09d182018e6a6b22f74fb2353f140fa7901448b99ef32ed2",
        "setpoint": "9c3190dac0658d9976c0b1c5d3e52d185e6a2419286622be4bee77cd147ec8d2",
    },
    {
        "observation": "3f9bbe6cc62ec3dfaaf486c20a818ee90bed95984ffe9e7bc9e4708f8dee5aac",
        "forecast": "9b83eb27b1025942e8d368388812d056990e987f7874904292ddf50c090e9c5e",
        "projection": "34f11a731abf616da01bd09549973ae1409cca510eb9c6acf114bc6609ebdabf",
        "historical_model": "25c5505091125fd36d5b8a79d5691e843cfcf12063a6280bc14392ef2dd56355",
        "historical_plan": "36af91aee783fe89979a92f4fb0e62b2f0dd31c04b09467d5c4403378cd2054f",
        "new_model": "6119c5ddb4269684a57308f2e95331520f9085830ef4ea1265aed758af3aabb8",
        "new_plan": "7acc746a9f630f7964def4973225ef82d954a43df66cde9bef9a3bca55b20c60",
        "setpoint": "de427a3d69913a8f28529474f8b8815156c89522815944d983f24f431ad29488",
    },
)


def _plan_semantics(plan: MPCPlanV1) -> tuple[object, ...]:
    return (
        plan.horizon_intervals,
        plan.deliveries,
        plan.steps,
        plan.primary_delivered_optimum_kg,
        plan.selected_delivered_kg,
        plan.delivery_lock_tolerance_kg,
        plan.minimum_commitment_fill,
        plan.terminal_ch4_equivalent_kg,
        plan.secondary_utility,
    )


def test_first_three_frozen_boundaries_have_exact_structural_equivalence() -> None:
    """Execute clarification 001 without opening the new physical trajectory."""

    pytest.importorskip("scipy")
    historical_settings = ServiceMPCSettingsV1()
    setpoint_settings = ServiceMPCSetpointSettingsV1()
    assert historical_settings.sha256 == (
        "0e8896cfb9fe47dc3ac3b285fc729e029a80803e31eddf4a885795d2d04253f2"
    )
    assert setpoint_settings.sha256 == (
        "e418dc37401482cf78036daefa61d80fa23fef0411a7215c3fb549a91085fb1b"
    )
    env = PlantEnvV2(episode=EpisodeConfigV2(horizon_h=72.0, delivery_rate_kg_h=35.0))
    env.reset(seed=0)

    for decision_index, frozen in enumerate(_FROZEN_IDENTITIES):
        observation = env.controller_observation
        assert observation.forecast is not None
        historical_projection = project_issued_mean_solar(
            observation.forecast,
            boundary_time_s=observation.state.boundary.time_s,
            config=env.config,
            interval_s=historical_settings.interval_s,
            energy_tolerance=historical_settings.energy_reconstruction_tolerance,
        )
        setpoint_projection = project_issued_mean_solar(
            observation.forecast,
            boundary_time_s=observation.state.boundary.time_s,
            config=env.config,
            interval_s=setpoint_settings.interval_s,
            energy_tolerance=setpoint_settings.energy_reconstruction_tolerance,
        )
        assert historical_projection == setpoint_projection
        horizon = min(
            historical_settings.maximum_planning_intervals,
            historical_settings.task_interval_count - decision_index,
        )
        historical_model = build_service_mpc_model(
            observation,
            historical_projection,
            settings=historical_settings,
            constraints=env.constraints,
            config=env.config,
            commitment_profile=env.episode_config.electrolyser_commitment,
            horizon_intervals=horizon,
        )
        setpoint_model = build_service_mpc_model(
            observation,
            setpoint_projection,
            settings=setpoint_settings,
            constraints=env.constraints,
            config=env.config,
            commitment_profile=env.episode_config.electrolyser_commitment,
            horizon_intervals=horizon,
        )
        assert (
            replace(
                setpoint_model,
                settings_sha256=historical_model.settings_sha256,
            )
            == historical_model
        )
        historical_plan = solve_service_mpc_model(
            historical_model,
            settings=historical_settings,
        )
        setpoint_plan = solve_service_mpc_model(
            setpoint_model,
            settings=setpoint_settings,
        )
        assert _plan_semantics(setpoint_plan) == _plan_semantics(historical_plan)
        assert tuple(stage.solution_sha256 for stage in setpoint_plan.stages) != tuple(
            stage.solution_sha256 for stage in historical_plan.stages
        )
        setpoints = realize_first_mpc_setpoints(
            setpoint_plan,
            setpoint_projection,
            decision_index=decision_index,
            observation_sha256=public_observation_sha256(observation),
            public_forecast_sha256=public_forecast_issue_sha256(observation),
            model_sha256=setpoint_model.sha256,
            constraints=env.constraints,
            config=env.config,
        )

        assert public_observation_sha256(observation) == frozen["observation"]
        assert public_forecast_issue_sha256(observation) == frozen["forecast"]
        assert setpoint_projection.sha256 == frozen["projection"]
        assert historical_model.sha256 == frozen["historical_model"]
        assert historical_plan.sha256 == frozen["historical_plan"]
        assert setpoint_model.sha256 == frozen["new_model"]
        assert setpoint_plan.sha256 == frozen["new_plan"]
        assert setpoints.sha256 == frozen["setpoint"]

        historical_action = realize_first_mpc_action(
            historical_plan,
            battery_support_limit_kw=env.constraints.support_limit_kw(env.config),
        )
        env.step_action(historical_action.action)


@pytest.fixture(scope="module")
def complete_short_run() -> tuple[
    tuple[MPCSetpointDecisionEvidenceV1, ...],
    tuple[tuple[int, ...], ...],
    str,
]:
    pytest.importorskip("scipy")
    env = PlantEnvV2(episode=SHORT_EPISODE)
    env.reset(seed=0)
    controller = ServiceMPCSetpointControllerV1(SHORT_SETTINGS)
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
    assert all(isinstance(event, MPCSetpointDecisionEvidenceV1) for event in events)
    return (
        tuple(
            event
            for event in events
            if isinstance(event, MPCSetpointDecisionEvidenceV1)
        ),
        tuple(dispatcher_elapsed),
        public_observation_sha256(env.controller_observation),
    )


def _complete_evidence(
    run: tuple[
        tuple[MPCSetpointDecisionEvidenceV1, ...],
        tuple[tuple[int, ...], ...],
        str,
    ],
) -> dict[str, object]:
    events, dispatcher_elapsed, final_sha = run
    return build_mpc_setpoint_run_evidence(
        events,
        dispatcher_elapsed,
        settings=SHORT_SETTINGS,
        completed_decisions=4,
        run_succeeded=True,
        pending_transition=False,
        runner_failure_phase=None,
        final_observation_sha256=final_sha,
    )


def test_complete_short_run_is_deeply_validated_and_aggregated(
    complete_short_run: tuple[
        tuple[MPCSetpointDecisionEvidenceV1, ...],
        tuple[tuple[int, ...], ...],
        str,
    ],
) -> None:
    evidence = _complete_evidence(complete_short_run)
    counts = evidence["counts"]
    assert counts["successful_decision_count"] == 4
    assert counts["closed_decision_count"] == 4
    assert counts["physical_minute_count"] == 60
    assert counts["successful_solver_stage_count"] == 8
    assert counts["fallback_action_count"] == 0
    assert counts["requested_target_overshoot_minutes"] == 0
    assert counts["reachable_tracking_violation_minutes"] == 0
    assert counts["simultaneous_charge_support_request_minutes"] == 0
    assert counts["unattributed_target_deficit_minutes"] == 0
    assert counts["accounting_violation_count"] == 0
    assert counts["chain_violation_count"] == 0
    assert evidence["decision_latency"]["count"] == 4
    assert evidence["solver_stage_latency"]["count"] == 8
    assert evidence["dispatcher_latency"]["count"] == 60
    assert len(evidence["event_sha256"]) == 4
    assert len(evidence["execution_sha256"]) == 4
    assert len(evidence["application_sha256"]) == 60
    assert evidence["target_tracking"]["requested_max_overshoot_kw"] == 0.0
    verify_mpc_setpoint_run_digests(evidence)


def test_timing_changes_outer_record_but_not_deterministic_evidence_graph(
    complete_short_run: tuple[
        tuple[MPCSetpointDecisionEvidenceV1, ...],
        tuple[tuple[int, ...], ...],
        str,
    ],
) -> None:
    events, dispatcher_elapsed, final_sha = complete_short_run
    original = _complete_evidence(complete_short_run)
    retimed_events = tuple(
        replace(event, decision_elapsed_ns=event.decision_elapsed_ns + 1)
        for event in events
    )
    retimed_dispatcher = tuple(
        tuple(value + 1 for value in group) for group in dispatcher_elapsed
    )
    retimed = build_mpc_setpoint_run_evidence(
        retimed_events,
        retimed_dispatcher,
        settings=SHORT_SETTINGS,
        completed_decisions=4,
        run_succeeded=True,
        pending_transition=False,
        runner_failure_phase=None,
        final_observation_sha256=final_sha,
    )

    assert retimed["evidence_graph_sha256"] == original["evidence_graph_sha256"]
    assert retimed["sha256"] != original["sha256"]


def test_run_builder_rejects_public_chain_and_latency_substitution(
    complete_short_run: tuple[
        tuple[MPCSetpointDecisionEvidenceV1, ...],
        tuple[tuple[int, ...], ...],
        str,
    ],
) -> None:
    events, dispatcher_elapsed, final_sha = complete_short_run
    first = events[0]
    assert first.residual is not None
    substituted = (
        replace(
            first,
            residual=replace(first.residual, next_observation_sha256="0" * 64),
        ),
        *events[1:],
    )
    with pytest.raises(ValueError, match="public observation chain"):
        build_mpc_setpoint_run_evidence(
            substituted,
            dispatcher_elapsed,
            settings=SHORT_SETTINGS,
            completed_decisions=4,
            run_succeeded=True,
            pending_transition=False,
            runner_failure_phase=None,
            final_observation_sha256=final_sha,
        )
    with pytest.raises(ValueError, match="latency count"):
        build_mpc_setpoint_run_evidence(
            events,
            (dispatcher_elapsed[0][:-1], *dispatcher_elapsed[1:]),
            settings=SHORT_SETTINGS,
            completed_decisions=4,
            run_succeeded=True,
            pending_transition=False,
            runner_failure_phase=None,
            final_observation_sha256=final_sha,
        )

    last = events[-1]
    assert last.residual is not None
    changed_terminal = replace(
        last.residual,
        observed_battery_end_kwh=last.residual.observed_battery_end_kwh + 1.0,
        battery_end_error_kwh=last.residual.battery_end_error_kwh + 1.0,
    )
    with pytest.raises(ValueError, match="residual battery state"):
        build_mpc_setpoint_run_evidence(
            (*events[:-1], replace(last, residual=changed_terminal)),
            dispatcher_elapsed,
            settings=SHORT_SETTINGS,
            completed_decisions=4,
            run_succeeded=True,
            pending_transition=False,
            runner_failure_phase=None,
            final_observation_sha256=final_sha,
        )


def test_outer_and_graph_digest_substitution_fail_closed(
    complete_short_run: tuple[
        tuple[MPCSetpointDecisionEvidenceV1, ...],
        tuple[tuple[int, ...], ...],
        str,
    ],
) -> None:
    evidence = _complete_evidence(complete_short_run)
    outer_substitution = deepcopy(evidence)
    outer_substitution["counts"]["fallback_action_count"] = 1
    with pytest.raises(ValueError, match="run evidence digest"):
        verify_mpc_setpoint_run_digests(outer_substitution)

    graph_substitution = deepcopy(evidence)
    graph_substitution["evidence_graph_sha256"] = "0" * 64
    graph_body = dict(graph_substitution)
    del graph_body["sha256"]
    graph_substitution["sha256"] = domain_sha256(
        SERVICE_MPC_SETPOINT_RUN_EVIDENCE_DOMAIN,
        graph_body,
    )
    with pytest.raises(ValueError, match="evidence graph digest"):
        verify_mpc_setpoint_run_digests(graph_substitution)


def test_explicitly_failed_or_pending_run_is_retained_without_fallback() -> None:
    pytest.importorskip("scipy")
    env = PlantEnvV2(episode=SHORT_EPISODE)
    env.reset(seed=0)
    controller = ServiceMPCSetpointControllerV1(SHORT_SETTINGS)
    with pytest.raises(ValueError, match="current public forecast"):
        controller.compute_setpoints(
            env.controller_observation.without_forecast(),
            constraints=env.constraints,
            config=env.config,
            commitment_profile=env.episode_config.electrolyser_commitment,
        )
    failure = controller.decision_events[0]
    assert isinstance(failure, MPCSetpointDecisionFailureV1)
    final_sha = public_observation_sha256(env.controller_observation)
    failed = build_mpc_setpoint_run_evidence(
        (failure,),
        (),
        settings=SHORT_SETTINGS,
        completed_decisions=0,
        run_succeeded=False,
        pending_transition=False,
        runner_failure_phase="controller",
        final_observation_sha256=final_sha,
    )
    assert failed["status"] == "failed"
    assert failed["counts"]["failed_decision_count"] == 1
    assert failed["counts"]["fallback_action_count"] == 0

    controller.reset()
    controller.compute_setpoints(
        env.controller_observation,
        constraints=env.constraints,
        config=env.config,
        commitment_profile=env.episode_config.electrolyser_commitment,
    )
    open_event = controller.decision_events[0]
    assert isinstance(open_event, MPCSetpointDecisionEvidenceV1)
    pending = build_mpc_setpoint_run_evidence(
        (open_event,),
        (),
        settings=SHORT_SETTINGS,
        completed_decisions=0,
        run_succeeded=False,
        pending_transition=True,
        runner_failure_phase="before_environment_step",
        final_observation_sha256=final_sha,
    )
    assert pending["pending_transition"] is True
    assert pending["counts"]["closed_decision_count"] == 0

"""Commission controller correctness; disposable fixture, no registered seeds."""

from dataclasses import replace
import json

import pytest
from plant_sim import ElectrolyserMode, PlantSimulator

from plant_backend import demo_recordings, operational_mpc
from plant_backend.operational_mpc import POLICY_ID
from plant_backend.rl.v2.economic_mpc_schema import (
    EconomicMPCPlanError,
    EconomicMPCSolveError,
    EconomicMPCStageSolveV1,
)
from plant_backend.rl.v2.mpc_forecast import project_issued_mean_solar
from plant_backend.rl.v2.mpc_schema import ServiceMPCInputError
from plant_backend.rl.v2.setpoint_controllers import LegacyHeuristicSetpointControllerV1
from plant_backend.rl.v2.setpoint_command import dispatch_process_setpoint_command
from plant_backend.safety import validate
from plant_backend.v2_demo import StepRequest, V2DemoSession


@pytest.fixture
def operating():
    demo = V2DemoSession()
    demo.reset("MPC")
    return demo, demo._env, demo._controller


def arguments(env):
    return dict(
        constraints=env.constraints,
        config=env.config,
        commitment_profile=env.episode_config.electrolyser_commitment,
    )


def solver_failure(status):
    return EconomicMPCSolveError(
        "private solver detail",
        stages=(
            EconomicMPCStageSolveV1(
                stage="service_maximization",
                status=status,
                message="private solver detail",
                elapsed_ns=1,
                objective_value=None,
                mip_gap=None,
                mip_node_count=None,
                solution_sha256=None,
                scipy_version="fixture",
                highs_version="fixture",
                success=False,
            ),
        ),
    )


def fail_with(error):
    def fail(*args, **kwargs):
        raise error

    return fail


def test_mean_model_keeps_physics_and_exact_shipments_without_daily_returns(operating):
    _, env, controller = operating
    observation = env.cyclic_market_controller_observation
    model = controller.build_model(observation, **arguments(env))
    rows = {row.name: row for row in model.physical_model.rows}
    assert any(name.startswith("mandatory_delivery_request[") for name in rows)
    assert any(name.startswith("mandatory_delivery_empty[") for name in rows)
    assert len([name for name in rows if "constant_solar_power" in name]) == 96
    assert rows["beta_constant_solar_power[0]"].upper == 0
    assert not any("early_reserve" in name or "cycle_return" in name for name in rows)
    physical = model.physical_model
    reserve = physical.metadata()["constants"]["battery_reserve_energy_kwh"]
    assert reserve > 0
    assert all(
        physical.variable_lower[i] == reserve
        for i in physical.group("battery_energy_kwh")[1:]
    )
    projection = project_issued_mean_solar(
        observation.plant.forecast,
        boundary_time_s=observation.plant.state.boundary.time_s,
        config=env.config,
        interval_s=controller.settings.interval_s,
    )
    assert physical.forecast_sha256 == projection.sha256
    assert model.metadata()["operating_policy"]["id"] == POLICY_ID
    altered = replace(
        model,
        physical_model=replace(physical, rows=physical.rows[:-1]),
    )
    assert altered.sha256 != model.sha256


@pytest.mark.parametrize(
    "error,reason",
    [
        (solver_failure(2), "infeasible"),
        (solver_failure(1), "time_limit"),
        (EconomicMPCPlanError("private solver detail"), "plan_rejected"),
    ],
)
def test_typed_failure_executes_visible_heuristic_and_preserves_evidence(
    operating, monkeypatch, error, reason
):
    demo, env, controller = operating
    observation = env.cyclic_market_controller_observation
    expected = LegacyHeuristicSetpointControllerV1().compute_command(
        observation.plant, **arguments(env)
    )
    monkeypatch.setattr(operational_mpc, "solve_economic_mpc_model", fail_with(error))
    initial = demo.snapshot()
    result = demo.step(
        StepRequest(
            expected_run_id=initial["run_id"],
            expected_revision=initial["revision"],
            decisions=1,
        )
    )
    assert result["status"] == "ready"
    execution = env.last_process_setpoint_execution
    assert execution.command.targets == expected.command.targets
    assert execution.command.issuer_id == POLICY_ID
    assert len(execution.applications) == 15
    operation = result["frame"]["decision"]["operation"]
    assert operation["mode"] == "fallback"
    assert operation["solver_failure"]["reason"] == reason
    assert operation["fallback_controller"] == LegacyHeuristicSetpointControllerV1.id
    assert operation["projected_shortfall_kg"] is None
    assert all(r["projected"] is None for r in operation["resources"])
    assert result["frame"]["plan"] is None
    assert "private solver detail" not in json.dumps(result)
    assert "private solver detail" not in controller.decision_events[0].payload_json
    assert any(e["title"] == "Control mode: Fallback" for e in result["events"])
    record = demo_recordings.validate(demo.recording())
    assert demo_recordings.summarize(record, "fixture")["fallback_decisions"] == 1


def test_fallback_is_not_sticky_and_next_real_plan_matches_resource_outlook(
    operating, monkeypatch
):
    _, env, controller = operating
    with monkeypatch.context() as context:
        context.setattr(
            operational_mpc, "solve_economic_mpc_model", fail_with(solver_failure(1))
        )
        fallback = controller.compute_command(
            env.cyclic_market_controller_observation, **arguments(env)
        )
    env.step_process_setpoints(fallback.command, controller_invalid=fallback.invalid)
    observation = env.cyclic_market_controller_observation
    decision = controller.compute_command(observation, **arguments(env))
    plan = decision.evidence.plan
    assert plan is not None
    assert decision.command.decision_index == 1
    operation = decision.evidence.operation()
    assert operation["mode"] == "recovery"
    assert operation["solver_failure"] is None
    assert len(controller.decision_events) == 2
    assert sum(d["predicted_kg"] for d in operation["deliveries"]) == pytest.approx(
        plan.selected_delivered_kg
    )
    for resource, (_, _, _, _, field, _) in zip(
        operation["resources"], operational_mpc.RESOURCE_FIELDS, strict=True
    ):
        assert resource["projected"] == getattr(plan.steps[-1], field)
        assert resource["projected_deficit"] == max(
            0, resource["reference"] - resource["projected"]
        )
    for target in plan.deliveries:
        step = plan.steps[target.due_boundary - 1]
        assert step.delivered_kg == pytest.approx(
            min(target.requested_kg, step.ch4_end_kg + step.delivered_kg), abs=1e-6
        )
    payload = json.loads(decision.evidence.payload_json)
    assert payload["observation_sha256"] == observation.sha256
    assert payload["plan_sha256"] == plan.sha256
    assert decision.command.source_sha256 == decision.evidence.sha256


def test_stale_input_and_changed_reference_fail_before_solver_without_fallback(
    operating, monkeypatch
):
    _, env, controller = operating
    initial = env.cyclic_market_controller_observation
    decision = controller.compute_command(initial, **arguments(env))
    env.step_process_setpoints(decision.command)
    monkeypatch.setattr(
        operational_mpc,
        "solve_economic_mpc_model",
        fail_with(AssertionError("must not solve")),
    )
    with pytest.raises(ValueError, match="stale"):
        controller.compute_command(initial, **arguments(env))
    current = env.cyclic_market_controller_observation
    altered = replace(
        current,
        cycle=replace(current.cycle, h2_margin_kg=current.cycle.h2_margin_kg - 1),
    )
    with pytest.raises(ValueError, match="reference changed"):
        controller.compute_command(altered, **arguments(env))
    assert len(controller.decision_events) == 1


def test_operation_crosses_daily_deficit_with_fixed_endowment_and_full_horizon(
    operating, monkeypatch
):
    _, env, controller = operating
    initial = env.cyclic_market_controller_observation
    initial_model = controller.build_model(initial, **arguments(env))
    monkeypatch.setattr(
        operational_mpc, "solve_economic_mpc_model", fail_with(solver_failure(1))
    )
    for _ in range(97):
        observation = env.cyclic_market_controller_observation
        if env.decisions_taken in (95, 96):
            model = controller.build_model(observation, **arguments(env))
            assert model.horizon_intervals == 96
            assert (
                model.initial_battery_energy_kwh
                == initial_model.initial_battery_energy_kwh
            )
            assert (
                model.initial_ch4_equivalent_kg
                == initial_model.initial_ch4_equivalent_kg
            )
        decision = controller.compute_command(observation, **arguments(env))
        _, _, terminated, truncated, _ = env.step_process_setpoints(
            decision.command, controller_invalid=decision.invalid
        )
        assert not terminated and not truncated
    assert env.physics_steps_taken == 1455
    cycle = env.cyclic_market_controller_observation.cycle
    assert min(cycle.h2_margin_kg, cycle.co2_margin_kg, cycle.ch4_margin_kg) < 0
    assert len(controller.decision_events) == 97


def startup_fixture(operating, monkeypatch, mode, *, solar_kw=650.0, **limits):
    """Assemble a synthetic boundary from component advances, without seed tapes."""
    _, env, controller = operating
    config = replace(env.config, battery=replace(env.config.battery, initial_soc=0.16))
    profile = env.episode_config.electrolyser_commitment
    sim = PlantSimulator(config, electrolyser_commitment=profile)
    monkeypatch.setattr(sim.solar, "clear_sky_kw", lambda _: solar_kw)
    boundary = sim.reset()
    observation = env.cyclic_market_controller_observation
    sim.time_s = observation.plant.state.boundary.time_s
    boundary = replace(boundary, time_s=sim.time_s)
    sim.state = sim.telemetry = boundary
    stack = sim.electrolyser

    def advance(power_kw, count):
        for _ in range(count):
            stack.advance_commitment(
                power_kw,
                config.dt_h,
                h2_headroom_kg=1000.0,
                commitment_intent=True,
            )

    if mode is ElectrolyserMode.COLD_STARTING:
        advance(600.0, round((profile.cold_start_s - 900.0) / config.dt_s))
        assert sim.electrolyser_commitment.startup_remaining_s == 900.0
    elif mode in {ElectrolyserMode.HOT_STARTING, ElectrolyserMode.RUNNING}:
        advance(600.0, round(profile.cold_start_s / config.dt_s))
        if mode is ElectrolyserMode.HOT_STARTING:
            advance(60.0, 1)
            advance(3000.0, round(profile.hot_start_s / config.dt_s) - 1)
            assert sim.electrolyser_commitment.startup_remaining_s == 60.0
    assert sim.electrolyser_commitment.mode is mode
    observation = replace(
        observation,
        plant=replace(
            observation.plant,
            state=replace(
                observation.plant.state,
                boundary=boundary,
                electrolyser_commitment=sim.electrolyser_commitment,
            ),
        ),
    )
    return (
        sim,
        observation,
        controller,
        dict(
            config=config,
            constraints=replace(env.constraints, **limits),
            commitment_profile=profile,
        ),
    )


def advance_held_command(sim, command, arguments):
    snapshots = []
    for _ in range(15):
        boundary = sim.get_telemetry()
        solar_kw = sim.solar.clear_sky_kw(sim.time_s + sim.config.dt_s)
        dispatch = dispatch_process_setpoint_command(
            command,
            boundary,
            solar_available_kw=solar_kw,
            constraints=arguments["constraints"],
            config=sim.config,
        )
        guarded = validate(
            dispatch.action,
            boundary,
            constraints=arguments["constraints"],
            config=sim.config,
            dispatch_solar_available_kw=solar_kw,
        )
        sim.step(guarded.action)
        snapshots.append(sim.electrolyser_commitment)
    return snapshots


@pytest.mark.parametrize(
    "mode", [ElectrolyserMode.COLD_STARTING, ElectrolyserMode.HOT_STARTING]
)
@pytest.mark.parametrize("solar_kw", [550.0, 650.0])
@pytest.mark.parametrize("use_operating_controller", [False, True])
def test_fallback_continues_active_startup_through_shared_dispatch_and_physics(
    operating, monkeypatch, mode, solar_kw, use_operating_controller
):
    sim, observation, controller, args = startup_fixture(
        operating, monkeypatch, mode, solar_kw=solar_kw
    )
    legacy = LegacyHeuristicSetpointControllerV1().compute_command(
        observation.plant, **args
    )
    assert legacy.command.targets.electrolyser_target_kw == 450.0
    monkeypatch.setattr(
        operational_mpc, "solve_economic_mpc_model", fail_with(solver_failure(2))
    )
    decision = controller.compute_command(observation, **args)
    assert decision.command.targets == replace(
        legacy.command.targets, electrolyser_target_kw=600.0
    )
    operation = decision.evidence.operation()
    assert operation["mode"] == "fallback"
    assert operation["solver_failure"]["reason"] == "infeasible"
    assert operation["startup_continuation"]["status"] == "adjusted"
    assert operation["startup_continuation"]["requested_kw"] == 450.0
    assert operation["startup_continuation"]["issued_kw"] == 600.0
    assert decision.command.source_sha256 == decision.evidence.sha256
    snapshots = advance_held_command(
        sim, decision.command if use_operating_controller else legacy.command, args
    )
    final = snapshots[-1]
    if use_operating_controller:
        assert final.forced_stops == 0
        assert final.aborted_starts == 0
        assert final.startup_remaining_s == 0.0
        assert final.mode in {ElectrolyserMode.RUNNING, ElectrolyserMode.HOT_STANDBY}
    else:
        assert final.forced_stops == 1
        assert final.aborted_starts == 1


def test_startup_continuation_does_not_override_power_shortage(operating, monkeypatch):
    sim, observation, controller, args = startup_fixture(
        operating,
        monkeypatch,
        ElectrolyserMode.COLD_STARTING,
        solar_kw=500.0,
        battery_support_max_kw=0.0,
    )
    legacy = LegacyHeuristicSetpointControllerV1().compute_command(
        observation.plant, **args
    )
    monkeypatch.setattr(
        operational_mpc, "solve_economic_mpc_model", fail_with(solver_failure(2))
    )
    decision = controller.compute_command(observation, **args)
    assert decision.command.targets == legacy.command.targets
    assert decision.command.targets.electrolyser_target_kw <= args[
        "constraints"
    ].electrolyser_limit_kw(sim.config)
    assert (
        decision.evidence.operation()["startup_continuation"]["status"]
        == "current_power_shortage"
    )
    assert advance_held_command(sim, decision.command, args)[-1].forced_stops == 1


def test_operator_cap_below_startup_requirement_still_rejects_before_solver(
    operating, monkeypatch
):
    _, observation, controller, args = startup_fixture(
        operating,
        monkeypatch,
        ElectrolyserMode.COLD_STARTING,
        solar_kw=800.0,
        electrolyser_max_kw=450.0,
    )
    monkeypatch.setattr(
        operational_mpc,
        "solve_economic_mpc_model",
        fail_with(AssertionError("must not solve")),
    )
    with pytest.raises(ServiceMPCInputError, match="cap cannot sustain startup power"):
        controller.compute_command(observation, **args)
    assert controller.decision_events == ()


def test_current_startup_feasibility_does_not_guarantee_future_weather(
    operating, monkeypatch
):
    sim, observation, controller, args = startup_fixture(
        operating,
        monkeypatch,
        ElectrolyserMode.COLD_STARTING,
        battery_support_max_kw=0.0,
    )
    monkeypatch.setattr(
        operational_mpc, "solve_economic_mpc_model", fail_with(solver_failure(2))
    )
    decision = controller.compute_command(observation, **args)
    assert decision.evidence.operation()["startup_continuation"]["status"] == "adjusted"
    monkeypatch.setattr(sim.solar, "clear_sky_kw", lambda _: 300.0)
    assert advance_held_command(sim, decision.command, args)[-1].forced_stops == 1
    assert sim.get_telemetry().battery_soc >= args["constraints"].reserve_soc(
        sim.config
    )


@pytest.mark.parametrize("mode", [ElectrolyserMode.OFF, ElectrolyserMode.RUNNING])
def test_low_power_without_active_startup_preserves_existing_fallback(
    operating, monkeypatch, mode
):
    sim, observation, controller, args = startup_fixture(operating, monkeypatch, mode)
    legacy = LegacyHeuristicSetpointControllerV1().compute_command(
        observation.plant, **args
    )
    monkeypatch.setattr(
        operational_mpc, "solve_economic_mpc_model", fail_with(solver_failure(2))
    )
    decision = controller.compute_command(observation, **args)
    assert decision.command.targets == legacy.command.targets
    assert decision.evidence.operation()["startup_continuation"] is None
    assert advance_held_command(sim, decision.command, args)[-1].forced_stops == 0

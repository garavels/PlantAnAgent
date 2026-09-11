"""Commission controller correctness; disposable fixture, no registered seeds."""

from dataclasses import replace
import json

import pytest

from plant_backend import demo_recordings, operational_mpc
from plant_backend.operational_mpc import POLICY_ID
from plant_backend.rl.v2.economic_mpc_schema import (
    EconomicMPCPlanError,
    EconomicMPCSolveError,
    EconomicMPCStageSolveV1,
)
from plant_backend.rl.v2.mpc_forecast import project_issued_mean_solar
from plant_backend.rl.v2.setpoint_controllers import LegacyHeuristicSetpointControllerV1
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

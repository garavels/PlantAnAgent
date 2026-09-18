"""Hard mission rows, public boundary geometry, and fail-closed plan evidence."""

from dataclasses import replace

import pytest

from plant_backend import mission_mpc
from plant_backend.mission_mpc import MissionMPCControllerV1, RESOURCE_GROUPS
from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.economic_mpc_schema import (
    EconomicMPCSolveError,
    EconomicMPCStageSolveV1,
)
from plant_backend.rl.v2.mpc_execution import public_observation_sha256
from plant_backend.rl.v2.setpoint_command import issue_process_setpoint_command
from plant_backend.rl.v2.setpoint_control import ProcessSetpointTargetsV1


@pytest.fixture
def mission():
    env = PlantCyclicMarketEnvV2Beta(episode=EpisodeConfigV2(delivery_rate_kg_h=25))
    env.reset(seed=12345)
    controller = MissionMPCControllerV1(
        horizon_h=168,
        interval_s=900,
        assumptions=env.economic_assumptions,
        beta_task=env.beta_task,
        electricity_model={"id": "test-public-price-semantics"},
    )
    yield env, controller
    env.close()


def arguments(env):
    return {
        "constraints": env.constraints,
        "config": env.config,
        "commitment_profile": env.episode_config.electrolyser_commitment,
    }


def idle_interval(env):
    observation = env.cyclic_market_controller_observation
    command = issue_process_setpoint_command(
        ProcessSetpointTargetsV1(0, 0, 0, 0),
        decision_index=observation.market.issued_at_decision,
        observation_sha256=public_observation_sha256(observation.plant),
        hold_duration_s=900,
        issuer_id="mission-test",
        source_schema="mission-test-v1",
        source_sha256="1" * 64,
        constraints=env.constraints,
        config=env.config,
    )
    env.step_process_setpoints(command)


@pytest.mark.parametrize("offset", [0, 1])
def test_all_daily_resources_are_hard_at_the_actual_boundary(mission, offset):
    env, controller = mission
    if offset:
        idle_interval(env)
        controller._next_decision = 1
    observation = env.cyclic_market_controller_observation
    model = controller.build_model(observation, **arguments(env))
    boundary = 96 - offset
    assert model.horizon_intervals == 96
    assert model.metadata()["mission"]["daily_boundary"] == boundary
    rows = {row.name: row for row in model.rows}
    requirements = model.metadata()["mission"]["resources"]
    assert [item["resource"] for item in requirements] == ["battery", "h2", "co2", "ch4"]
    for requirement, group in zip(requirements, RESOURCE_GROUPS, strict=True):
        index = model.group(group)[boundary]
        row = rows[f"mission_daily_return_{requirement['resource']}[{boundary}]"]
        assert row.coefficients == ((index, 1.0),)
        assert row.lower == requirement["minimum"]
        assert model.variable_lower[index] >= requirement["minimum"]
        assert requirement["minimum"] == pytest.approx(requirement["target"] - 1e-6)
    # All declarations and sparse rows affect the model digest.
    changed = replace(
        model,
        physical_model=replace(
            model.physical_model, rows=model.physical_model.rows[:-1]
        ),
    )
    assert changed.sha256 != model.sha256


def test_deliveries_are_hard_including_the_daily_boundary_item(mission):
    env, controller = mission
    model = controller.build_model(
        env.cyclic_market_controller_observation, **arguments(env)
    )
    assert [item.due_boundary for item in model.deliveries] == [24, 48, 72, 96]
    rows = {row.name: row for row in model.rows}
    for delivery in model.deliveries:
        index = model.group("delivered_kg")[delivery.due_boundary - 1]
        row = rows[f"mission_delivery[{delivery.due_boundary}]"]
        assert row.coefficients == ((index, 1.0),)
        assert row.lower == 150
        assert model.variable_lower[index] == model.variable_upper[index] == 150


def test_intraday_policy_protects_battery_and_both_feedstocks(mission):
    env, controller = mission
    model = controller.build_model(
        env.cyclic_market_controller_observation, **arguments(env)
    )
    rows = {row.name: row for row in model.rows}
    requirements = model.metadata()["mission"]["resources"]
    for requirement, group in zip(requirements[:3], RESOURCE_GROUPS[:3], strict=True):
        for boundary in (1, 48, 96):
            index = model.group(group)[boundary]
            row = rows[f"mission_protected_{requirement['resource']}[{boundary}]"]
            assert row.coefficients == ((index, 1.0),)
            assert row.lower == requirement["minimum"]
            assert model.variable_lower[index] >= row.lower
    assert not any(name.startswith("mission_protected_ch4") for name in rows)


def test_unrepresented_daily_boundary_is_rejected(mission):
    env, controller = mission
    controller.settings = replace(controller.settings, planning_horizon_h=12)
    with pytest.raises(ValueError, match="beyond the supported public horizon"):
        controller.build_model(
            env.cyclic_market_controller_observation, **arguments(env)
        )


def test_stale_or_corrupt_public_cycle_cannot_move_the_deadline(mission):
    env, controller = mission
    observation = env.cyclic_market_controller_observation
    corrupted = replace(
        observation,
        cycle=replace(observation.cycle, decision_in_cycle=1, progress=1 / 96),
    )
    with pytest.raises(ValueError, match="cycle counters"):
        controller.build_model(corrupted, **arguments(env))
    assert controller._next_decision == 0


def test_solver_rejection_is_explicit_and_does_not_advance_state(mission, monkeypatch):
    env, controller = mission
    observation = env.cyclic_market_controller_observation
    original = env.simulator.state.to_dict()

    def rejected(*args, **kwargs):
        raise EconomicMPCSolveError("infeasible hard mission", stages=())

    monkeypatch.setattr(mission_mpc, "solve_economic_mpc_model", rejected)
    with pytest.raises(EconomicMPCSolveError, match="infeasible hard mission"):
        controller.compute_command(observation, **arguments(env))
    assert controller._next_decision == 0
    assert controller._reference is None
    assert controller.decision_events == ()
    assert env.simulator.state.to_dict() == original
    assert controller.to_dict()["fallback"] is None


def test_selected_plan_reports_forecast_feasibility_without_claiming_execution(mission):
    pytest.importorskip("scipy.optimize")
    env, controller = mission
    decision = controller.compute_command(
        env.cyclic_market_controller_observation, **arguments(env)
    )
    evidence = decision.evidence.to_dict()
    assert evidence["audited_hard_return_plan"] is True
    assert evidence["realized_mission_passed"] is None
    assert len(evidence["resources"]) == 4
    for resource in evidence["resources"]:
        assert resource["predicted"] >= resource["minimum"] - 1e-7
    assert decision.command.issuer_id == controller.id
    assert controller._next_decision == 1
    assert evidence["controller"]["electricity_model"] == {"id": "test-public-price-semantics"}
    env.step_process_setpoints(decision.command)
    assert env.simulator.state.step == 15


def test_rejected_probe_retains_solver_stage_and_controller_identity(monkeypatch, tmp_path):
    from docs.research.tools.run_mission_mpc_v1 import run

    stage = EconomicMPCStageSolveV1(
        stage="service_maximization",
        status=2,
        message="infeasible test model",
        elapsed_ns=123,
        objective_value=None,
        mip_gap=None,
        mip_node_count=None,
        solution_sha256=None,
        scipy_version="test-scipy",
        highs_version="test-highs",
        success=False,
    )

    def rejected(*args, **kwargs):
        raise EconomicMPCSolveError("test rejection", stages=(stage,))

    monkeypatch.setattr(mission_mpc, "solve_economic_mpc_model", rejected)
    destination = tmp_path / "rejected.json.gz"
    result = run(9920000, split="development", output=destination, max_decisions=1)
    assert result["mission_passed"] is False
    assert result["decisions"] == 0
    assert result["controller_configuration"]["id"] == MissionMPCControllerV1.id
    assert result["rejected_plans"][0]["stages"] == [stage.to_dict()]
    assert result["rejected_plans"][0]["audited_hard_return_plan"] is False
    assert result["rejected_plans"][0]["fallback"] is None
    assert "electricity" in result["change"]
    assert result["source_files_unchanged_during_run"] is True
    with pytest.raises(FileExistsError):
        run(9920000, split="development", output=destination, max_decisions=1)

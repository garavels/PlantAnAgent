"""V8 behaviour: a supervisor above a controller, and the seams between them.

The claims worth defending are structural, not textual. A handover must not
disturb the plant. One envelope must reach whoever is driving. Health must
measure the *controller*, not the plant's luck. Switching must be braked, so a
supervisor with a bad idea twice in a row cannot make the control room flicker.
And every failure -- the model, the policy, the checkpoint -- must leave someone
competent in the chair.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from plant_sim import Action, PlantConfig, PlantState

from plant_backend.agent import AgentToolbox, AgentTrace, RuleEngine, SupervisorAgent
from plant_backend.agent import hierarchy
from plant_backend.controllers import Constraints, ControlDecision, HeuristicController
from plant_backend.controllers.health import ControllerHealth, StepRecord, assess
from plant_backend.controllers.manager import ControllerManager
from plant_backend.controllers.safe import SAFE_CONTROLLER_ID
from plant_backend.controllers.rl import RL_CONTROLLER_ID
from plant_backend.orchestration import PlantRuntime, foresight
from plant_backend.rl import checkpoints

#: A morning of driving: enough for a health grade and a scenario to bite.
MORNING_STEPS = 420
#: A steady, unremarkable policy vector, and one that drives like a vandal.
STEADY = hierarchy.STEADY_POLICY
RECKLESS = hierarchy.RECKLESS_POLICY


@pytest.fixture
def runtime() -> PlantRuntime:
    plant = PlantRuntime()
    plant.advance(MORNING_STEPS)
    return plant


@pytest.fixture
def rl_runtime(runtime: PlantRuntime, tmp_path: Path) -> PlantRuntime:
    """The same plant, driving on a fixed checkpoint instead of the rules."""
    runtime.load_rl_checkpoint(str(checkpoints.write_fixed_checkpoint(tmp_path / "steady", STEADY)))
    runtime.set_controller(RL_CONTROLLER_ID, reason="test setup")
    runtime.advance(30)
    return runtime


def clocked(time_s: float) -> ControllerManager:
    """A manager whose clock the test moves, so cooldowns can be reached."""
    times = {"s": time_s}
    manager = ControllerManager(now=lambda: (times["s"], "D1 00:00"))
    manager.clock = times  # type: ignore[attr-defined]  # test handle only
    return manager


# --- switching ------------------------------------------------------------
def test_a_handover_does_not_disturb_the_plant(rl_runtime: PlantRuntime) -> None:
    """Selection only: the same plant, mid-run, under a different driver."""
    before = rl_runtime.session.telemetry
    rl_runtime.configure_constraints({"electrolyser_max_kw": 900.0})

    switch = rl_runtime.switch_controller(
        HeuristicController.id, reason="RL misbehaving", by="agent"
    )

    after = rl_runtime.session.telemetry
    assert switch is not None and switch.previous == RL_CONTROLLER_ID
    assert rl_runtime.controller_id == HeuristicController.id
    assert after.time_s == before.time_s  # the clock did not move
    assert after.h2_level_kg == before.h2_level_kg
    assert after.battery_soc == before.battery_soc
    assert after.ch4_produced_total_kg == before.ch4_produced_total_kg
    assert rl_runtime.constraints.electrolyser_max_kw == 900.0  # envelope carried over


def test_every_switch_is_logged_with_a_reason_and_an_author(runtime: PlantRuntime) -> None:
    runtime.switch_controller(SAFE_CONTROLLER_ID, reason="critical condition", by="agent")

    switch = runtime.control.last_switch
    assert switch is not None
    assert switch.summary == "HEURISTIC -> SAFE"
    assert switch.reason == "critical condition"
    assert switch.by == "agent"
    assert switch.clock and switch.time_s > 0.0


def test_a_switch_without_a_reason_is_still_attributable(runtime: PlantRuntime) -> None:
    runtime.switch_controller(SAFE_CONTROLLER_ID, reason="   ", by="agent")

    assert runtime.control.last_switch is not None
    assert runtime.control.last_switch.reason == "no reason given"


def test_the_agent_may_not_switch_twice_in_a_row(runtime: PlantRuntime) -> None:
    """Cooldown, not opinion: the second switch is refused with the time left."""
    runtime.switch_controller(HeuristicController.id, reason="first", by="agent")
    runtime.switch_controller(SAFE_CONTROLLER_ID, reason="critical", by="agent")

    with pytest.raises(ValueError, match="cooldown"):
        runtime.switch_controller(HeuristicController.id, reason="second thoughts", by="agent")
    assert runtime.controller_id == SAFE_CONTROLLER_ID


def test_the_operator_is_never_held_by_the_agents_cooldown(runtime: PlantRuntime) -> None:
    runtime.switch_controller(SAFE_CONTROLLER_ID, reason="critical", by="agent")

    runtime.set_controller(HeuristicController.id, reason="operator takes it back")

    assert runtime.controller_id == HeuristicController.id


def test_a_controller_taken_off_for_its_behaviour_cannot_come_straight_back() -> None:
    """Return hysteresis: a demoted controller waits, even after the cooldown."""
    manager = clocked(0.0)
    manager.controllers[RL_CONTROLLER_ID] = _StandInController()  # type: ignore[assignment]
    manager.health[RL_CONTROLLER_ID] = _poor_health(RL_CONTROLLER_ID)
    manager.active_id = RL_CONTROLLER_ID

    manager.select(HeuristicController.id, reason="rl graded poor", by="agent")
    manager.clock["s"] = manager.switch_cooldown_s + 60.0  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="taken off the plant for its behaviour"):
        manager.select(RL_CONTROLLER_ID, reason="giving it another go", by="agent")

    manager.clock["s"] = manager.return_cooldown_s + 60.0  # type: ignore[attr-defined]
    manager.select(RL_CONTROLLER_ID, reason="the fault has cleared", by="agent")
    assert manager.active_id == RL_CONTROLLER_ID


def test_safe_mode_is_never_held_up_by_a_cooldown(runtime: PlantRuntime) -> None:
    """A brake against oscillation must not be a brake against safety."""
    runtime.switch_controller(HeuristicController.id, reason="noop", by="agent")
    runtime.set_controller(SAFE_CONTROLLER_ID, reason="setup")
    runtime.set_controller(HeuristicController.id, reason="setup")

    runtime.set_operating_state("safe", reason="battery on the reserve", by="agent")

    assert runtime.controller_id == SAFE_CONTROLLER_ID
    assert runtime.control.operating_state == "safe"


def test_leaving_safe_mode_returns_the_plant_to_the_driver_it_had(rl_runtime: PlantRuntime) -> None:
    rl_runtime.set_operating_state("safe", reason="critical condition", by="agent")
    assert rl_runtime.controller_id == SAFE_CONTROLLER_ID

    rl_runtime.set_operating_state("normal", reason="condition cleared", by="agent")

    assert rl_runtime.controller_id == RL_CONTROLLER_ID
    assert rl_runtime.control.operating_state == "normal"


def test_degraded_running_is_a_label_not_a_handover(rl_runtime: PlantRuntime) -> None:
    rl_runtime.set_operating_state("degraded", reason="stack degraded", by="agent")

    assert rl_runtime.control.operating_state == "degraded"
    assert rl_runtime.controller_id == RL_CONTROLLER_ID


def test_an_unknown_operating_state_is_refused(runtime: PlantRuntime) -> None:
    with pytest.raises(ValueError, match="unknown operating state"):
        runtime.set_operating_state("emergency")


# --- one envelope, every controller ---------------------------------------
def test_a_retuned_cap_reaches_the_policy_and_the_rules_alike(rl_runtime: PlantRuntime) -> None:
    """The V8 promise: one envelope, no retraining, both controllers inside it."""
    rl_runtime.configure_constraints({"electrolyser_max_kw": 200.0}, by="agent")
    rl_runtime.advance(30)

    for controller_id in (RL_CONTROLLER_ID, HeuristicController.id):
        rl_runtime.set_controller(controller_id, reason="test sweep")
        for _ in range(10):
            rl_runtime.advance(1)
            assert rl_runtime.session.telemetry.electrolyser_power_kw <= 200.0 + 1e-6


def test_the_manager_holds_the_envelope_the_runtime_reports(runtime: PlantRuntime) -> None:
    runtime.configure_constraints({"dac_max_kw": 120.0}, by="agent")

    assert runtime.constraints is runtime.control.constraints
    assert runtime.control_status()["constraints"]["dac_max_kw"] == 120.0
    # the envelope remembers what the supervisor overwrote, so a reset can undo it
    assert "dac_max_kw" in runtime.control.agent_envelope
    assert runtime.control.agent_envelope["dac_max_kw"] != 120.0


# --- controller health ----------------------------------------------------
def test_a_cap_clamp_alone_never_condemns_a_controller() -> None:
    """A policy clamped to the supervisor's own cap is not a policy to sack."""
    health = ControllerHealth("rl")
    for _ in range(hierarchy.BATCH_MIN * 20):
        health.record(StepRecord(safety_correction=True))

    assert health.grade in {"good", "suspect"}
    assert health.counts()["safety_corrections"] == health.steps


def test_asking_an_offline_unit_for_power_does_condemn_one() -> None:
    health = ControllerHealth("rl")
    for _ in range(hierarchy.BATCH_MIN * 20):
        health.record(StepRecord(safety_correction=True, unsafe_request=True))

    assert health.grade == "poor"
    assert any("offline" in concern for concern in health.concerns())


def test_health_is_unrated_until_a_controller_has_shown_something() -> None:
    health = ControllerHealth("rl")
    health.record(StepRecord(invalid_action=True, unsafe_request=True))

    assert health.grade == "unrated" and health.concerns() == []


def test_totals_outlive_the_window(runtime: PlantRuntime) -> None:
    """The window forgets so the grade stays current; the totals do not."""
    health = ControllerHealth("rl", window=10)
    for _ in range(40):
        health.record(StepRecord(safety_correction=True))

    assert health.steps == 10
    assert health.total_steps == 40
    assert health.totals["safety_corrections"] == 40


def test_a_hand_to_mouth_reactor_is_not_a_starved_one(runtime: PlantRuntime) -> None:
    """An empty tank while methane is being made is throughput, not starvation."""
    before = runtime.session.telemetry
    empty = _bend(before, h2_level_kg=0.0, co2_level_kg=0.0, ch4_production_kg_h=0.0)
    producing = _bend(before, h2_level_kg=0.0, co2_level_kg=0.0, ch4_production_kg_h=4.0)
    requested = Action.sanitised({"methanator_share": 0.2})

    starved = assess(
        before=empty,
        after=empty,
        requested=requested,
        adjustments=(),
        invalid_action=False,
        critical_events=0,
        reserve_soc=0.2,
    )
    fed_by_flow = assess(
        before=empty,
        after=producing,
        requested=requested,
        adjustments=(),
        invalid_action=False,
        critical_events=0,
        reserve_soc=0.2,
    )

    assert starved.resource_starvation
    assert not fed_by_flow.resource_starvation


def test_poor_health_alerts_once_per_spell() -> None:
    manager = clocked(0.0)
    manager.health[HeuristicController.id] = _poor_health(HeuristicController.id)

    assert manager.health_alert() is not None
    assert manager.health_alert() is None  # the same bad spell does not shout twice


def test_the_health_alert_wakes_the_supervisor(runtime: PlantRuntime) -> None:
    runtime.supervisor = SupervisorAgent(runtime, RuleEngine(), enabled=True)
    runtime.control.health[HeuristicController.id] = _poor_health(HeuristicController.id)

    trigger = runtime.supervisor.notice(runtime.session.recent(10))

    assert trigger is not None and trigger.reason == "controller_health"
    assert "poor" in trigger.detail


# --- failure paths --------------------------------------------------------
def test_a_policy_that_throws_hands_the_plant_back(rl_runtime: PlantRuntime) -> None:
    controller = rl_runtime.control.controllers[RL_CONTROLLER_ID]

    def explode(_vector: object) -> None:
        raise RuntimeError("inference exploded")

    controller.loaded.policy.act = explode  # type: ignore[method-assign]

    rl_runtime.advance(2)

    assert rl_runtime.controller_id == HeuristicController.id
    assert rl_runtime.control.last_switch is not None
    assert rl_runtime.control.last_switch.by == "system"
    assert not rl_runtime.control.available(RL_CONTROLLER_ID)


def test_unloading_a_driving_policy_is_not_a_gap_in_control(rl_runtime: PlantRuntime) -> None:
    rl_runtime.unload_rl_checkpoint()

    assert rl_runtime.controller_id == HeuristicController.id
    assert "RL checkpoint unloaded" in (rl_runtime.control.last_switch or _no_switch()).reason
    rl_runtime.advance(2)  # and the plant carries on


def test_rl_is_refused_with_a_reason_rather_than_invented(runtime: PlantRuntime) -> None:
    with pytest.raises(ValueError, match="RL controller unavailable"):
        runtime.set_controller(RL_CONTROLLER_ID)


# --- the supervisor's hands -----------------------------------------------
def test_the_supervisor_sees_who_is_driving_and_how_it_drives(rl_runtime: PlantRuntime) -> None:
    toolbox = AgentToolbox(rl_runtime, AgentTrace())

    active = toolbox.get_active_controller()
    metrics = toolbox.get_controller_metrics()

    assert active["active"]["id"] == RL_CONTROLLER_ID
    assert active["operating_state"] == "normal"
    assert {entry["id"] for entry in active["available"]} >= {"heuristic", "safe", "rl"}
    assert metrics["active"] == RL_CONTROLLER_ID
    assert "grade" in metrics and RL_CONTROLLER_ID in metrics["controllers"]


def test_a_switch_the_supervisor_cannot_justify_is_refused(rl_runtime: PlantRuntime) -> None:
    toolbox = AgentToolbox(rl_runtime, AgentTrace())

    result = toolbox.switch_controller(HeuristicController.id, reason="  ")

    assert result["ok"] is False and "reason is required" in result["error"]
    assert rl_runtime.controller_id == RL_CONTROLLER_ID


def test_a_refused_switch_is_an_answer_not_an_exception(rl_runtime: PlantRuntime) -> None:
    toolbox = AgentToolbox(rl_runtime, AgentTrace())
    toolbox.switch_controller(HeuristicController.id, reason="rl graded poor")

    result = toolbox.switch_controller(RL_CONTROLLER_ID, reason="changed my mind")

    assert result["ok"] is False and "cooldown" in result["error"]
    assert rl_runtime.controller_id == HeuristicController.id


def test_an_agent_switch_is_recorded_as_an_action_it_took(rl_runtime: PlantRuntime) -> None:
    trace = AgentTrace()
    toolbox = AgentToolbox(rl_runtime, trace)

    toolbox.switch_controller(HeuristicController.id, reason="repeated unsafe requests")

    assert any(
        entry["kind"] == "action" and "RL -> HEURISTIC" in entry["message"]
        for entry in trace.recent()
    )


def test_safe_mode_needs_a_reason_too(runtime: PlantRuntime) -> None:
    toolbox = AgentToolbox(runtime, AgentTrace())

    result = toolbox.set_operating_state("safe", reason="")

    assert result["ok"] is False
    assert runtime.controller_id == HeuristicController.id


# --- short-horizon simulation --------------------------------------------
def test_a_projection_never_touches_the_live_plant(rl_runtime: PlantRuntime) -> None:
    before = rl_runtime.session.simulator.state
    envelope = rl_runtime.constraints.to_dict()
    toolbox = AgentToolbox(rl_runtime, AgentTrace())

    forecast = toolbox.simulate_strategy(
        controller=HeuristicController.id,
        constraints={"electrolyser_max_kw": 100.0},
        horizon_min=60,
    )

    after = rl_runtime.session.simulator.state
    assert forecast["ok"] and forecast["controller"] == HeuristicController.id
    assert set(forecast) >= {
        "predicted_ch4_kg",
        "min_battery_soc",
        "h2_shortage_steps",
        "co2_shortage_steps",
        "critical_events",
        "safety_corrections",
    }
    assert after.time_s == before.time_s and after.ch4_level_kg == before.ch4_level_kg
    assert rl_runtime.constraints.to_dict() == envelope
    assert rl_runtime.controller_id == RL_CONTROLLER_ID


def test_a_projection_costs_the_supervisor_no_write_budget(runtime: PlantRuntime) -> None:
    toolbox = AgentToolbox(runtime, AgentTrace())

    for _ in range(8):
        assert toolbox.simulate_strategy(horizon_min=15)["ok"]

    assert toolbox.set_controller_constraints({"dac_max_kw": 150.0}, reason="still allowed")["ok"]


def test_a_projection_of_an_impossible_strategy_is_refused_not_run(runtime: PlantRuntime) -> None:
    toolbox = AgentToolbox(runtime, AgentTrace())

    assert toolbox.simulate_strategy(controller=RL_CONTROLLER_ID)["ok"] is False
    assert toolbox.simulate_strategy(constraints={"electrolyser_max_kw": -5.0})["ok"] is False


def test_the_horizon_is_bounded_at_both_ends(runtime: PlantRuntime) -> None:
    assert foresight.look_ahead(runtime, horizon_min=1)["horizon_min"] == foresight.MIN_HORIZON_MIN
    assert (
        foresight.look_ahead(runtime, horizon_min=10_000)["horizon_min"]
        == foresight.MAX_HORIZON_MIN
    )


# --- the hierarchy, end to end -------------------------------------------
def test_the_hierarchy_status_is_one_place_the_frontend_can_read(rl_runtime: PlantRuntime) -> None:
    hierarchy_status = rl_runtime.control_status()["hierarchy"]

    assert hierarchy_status["active_id"] == RL_CONTROLLER_ID
    assert hierarchy_status["operating_state"] == "normal"
    assert hierarchy_status["health"]["controller"] == RL_CONTROLLER_ID
    assert "grade" in hierarchy_status["health"]
    assert hierarchy_status["switch_count"] >= 1
    assert hierarchy_status["cooldown_s"] > 0.0


@pytest.mark.parametrize("case", hierarchy.CASES, ids=lambda case: case.name)
def test_hierarchical_control_regression_suite(case: hierarchy.Case) -> None:
    """Heuristic, RL and RL+supervisor on the same scenario, scored the same way."""
    outcome = asyncio.run(hierarchy.run_case_with_checkpoints(case, hours=6.0))

    assert outcome.passed, f"{case.name}: {'; '.join(outcome.failures)}"
    assert len(outcome.runs) == len(case.arms)


# --- helpers --------------------------------------------------------------
class _StandInController:
    """Something to hold RL's place in a manager built without a checkpoint."""

    id = RL_CONTROLLER_ID
    label = "RL Policy"
    description = "a stand-in policy, for the switching tests"

    def reset(self) -> None:
        return None

    def compute_action(
        self, observation: PlantState, *, constraints: Constraints, config: PlantConfig
    ) -> ControlDecision:
        return ControlDecision(Action.idle(), ("stand-in",))

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "label": self.label, "description": self.description}


def _poor_health(controller_id: str) -> ControllerHealth:
    health = ControllerHealth(controller_id)
    for _ in range(120):
        health.record(StepRecord(invalid_action=True, unsafe_request=True))
    return health


def _bend(state: PlantState, **overrides: float) -> PlantState:
    from dataclasses import replace

    return replace(state, **overrides)  # type: ignore[arg-type]


def _bus_kw(runtime: PlantRuntime) -> float:
    return max(0.0, runtime.session.telemetry.solar_available_kw)


def _no_switch():
    raise AssertionError("expected a switch to have been logged")

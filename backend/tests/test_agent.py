"""V6 behaviour: the supervisor observes, investigates and retunes -- nothing more.

The properties worth defending are the boundaries, not the wording of a
diagnosis: what the supervisor can see (telemetry, never the simulator's truth),
what it can do (the envelope and the mode, always through validation), how often
it is allowed to think, and what happens when the model falls over.
"""

from __future__ import annotations

import asyncio

import pytest
from plant_sim import Action

from plant_backend.agent import AgentToolbox, AgentTrace, RuleEngine, SupervisorAgent
from plant_backend.agent import evaluation, tools as tools_module
from plant_backend.agent.engines import Diagnosis
from plant_backend.agent.observations import consistency
from plant_backend.agent.supervisor import API_KEY_ENV, default_engine
from plant_backend.agent.triggers import Trigger, TriggerWatch
from plant_backend.orchestration import PlantRuntime

#: Enough steps to have a morning of history to look at.
MORNING_STEPS = 420


@pytest.fixture
def runtime() -> PlantRuntime:
    """A plant with a rule-driven supervisor and a few hours behind it."""
    plant = PlantRuntime()
    plant.supervisor = SupervisorAgent(plant, RuleEngine(), enabled=True)
    plant.advance(MORNING_STEPS)
    return plant


@pytest.fixture
def toolbox(runtime: PlantRuntime) -> AgentToolbox:
    return AgentToolbox(runtime, AgentTrace())


def investigate(supervisor: SupervisorAgent, trigger: Trigger | None = None) -> object:
    return asyncio.run(supervisor.investigate(trigger))


# --- what the supervisor may see -----------------------------------------
def test_observations_are_telemetry_not_simulator_truth(toolbox: AgentToolbox) -> None:
    """Sensor faults distort what the supervisor sees, exactly as for the control room."""
    runtime = toolbox._runtime  # noqa: SLF001 - the seam under test
    runtime.session.inject_fault({"type": "sensor_drift", "severity": 0.2, "channel": "h2_level_kg"})
    runtime.advance(2)

    reading = toolbox.get_current_telemetry()
    truth = runtime.session.simulator.state

    assert reading["h2_level_kg"] == pytest.approx(runtime.session.telemetry.h2_level_kg, abs=0.1)
    assert reading["h2_level_kg"] > truth.h2_level_kg


def test_inspection_exposes_observables_never_the_hidden_fault(toolbox: AgentToolbox) -> None:
    """A degraded stack is visible as bad yield, not as the simulator's severity."""
    runtime = toolbox._runtime  # noqa: SLF001
    runtime.session.inject_fault({"type": "electrolyser_degradation", "severity": 0.4})
    runtime.advance(5)

    reading = toolbox.inspect_component("electrolyser")

    assert "degradation" not in reading and "health" not in reading
    assert reading["yield_ratio"] < 0.8  # same kW in, less H2 out
    assert reading["observed_kwh_per_kg"] > reading["design_kwh_per_kg"]


def test_unknown_component_is_refused_not_raised(toolbox: AgentToolbox) -> None:
    assert toolbox.inspect_component("reactor_47")["ok"] is False


def test_sensor_drift_shows_up_as_an_internal_inconsistency(runtime: PlantRuntime) -> None:
    """The level and the flows are two readings of one truth; a biased one disagrees."""
    before = consistency(runtime.session.recent(120), runtime.config)
    assert before["inconsistent"] == []

    runtime.session.inject_fault({"type": "sensor_drift", "severity": 0.2, "channel": "h2_level_kg"})
    runtime.advance(5)

    after = consistency(runtime.session.recent(120), runtime.config)
    assert [finding["channel"] for finding in after["inconsistent"]] == ["h2_level_kg"]


# --- what the supervisor may do ------------------------------------------
def test_the_toolbox_cannot_reach_the_simulator_or_set_an_action() -> None:
    """The whole point of the seam: no action tool, no simulator, no fault clearing."""
    surface = {
        name
        for name in vars(AgentToolbox)
        if not name.startswith("_") and callable(getattr(AgentToolbox, name))
    }

    assert surface == {
        "get_current_telemetry",
        "get_recent_telemetry",
        "get_recent_events",
        "inspect_component",
        "run_diagnostic",
        "get_controller_status",
        "get_controller_constraints",
        "observe",
        "set_controller_constraints",
        "set_operating_mode",
        "pause_plant",
        "resume_plant",
        "request_safe_shutdown",
    }


def test_constraints_are_applied_through_the_runtime_and_hold(toolbox: AgentToolbox) -> None:
    runtime = toolbox._runtime  # noqa: SLF001
    result = toolbox.set_controller_constraints({"electrolyser_max_kw": 900.0}, reason="degraded")

    assert result["ok"] is True
    assert runtime.constraints.electrolyser_max_kw == 900.0
    runtime.advance(30)
    assert max(frame["electrolyser_power_kw"] for frame in runtime.session.recent(30)) <= 900.0 + 1e-6


def test_validation_stays_authoritative_over_the_supervisor(toolbox: AgentToolbox) -> None:
    """Even in manual mode with an absurd setpoint, the guard clamps the plant.

    The cap is applied to a *share* of the last observed bus, so a rising sun
    can carry the stack a few kW past it for one timestep; what must not happen
    is the requested 5 MW of battery support getting through.
    """
    runtime = toolbox._runtime  # noqa: SLF001
    toolbox.set_controller_constraints({"electrolyser_max_kw": 500.0}, reason="test")
    toolbox.set_operating_mode("manual", reason="test")
    runtime.set_manual_action(Action(electrolyser_share=1.0, battery_support_kw=5_000.0))

    frames = runtime.advance(10)

    assert max(frame["electrolyser_power_kw"] for frame in frames) <= 500.0 * 1.05
    assert runtime.adjustments  # the guard had to say something


def test_invalid_constraints_are_refused_without_touching_the_envelope(
    toolbox: AgentToolbox,
) -> None:
    runtime = toolbox._runtime  # noqa: SLF001
    before = runtime.constraints.to_dict()

    assert toolbox.set_controller_constraints({"turbine_max_kw": 10.0}, reason="?")["ok"] is False
    assert toolbox.set_controller_constraints({"battery_min_soc": 5.0}, reason="?")["ok"] is False
    assert toolbox.set_controller_constraints({}, reason="?")["ok"] is False
    assert runtime.constraints.to_dict() == before


def test_one_run_may_only_change_so_much(toolbox: AgentToolbox) -> None:
    for _ in range(tools_module.MAX_WRITES_PER_RUN):
        assert toolbox.set_operating_mode("auto", reason="churn")["ok"] is True

    refused = toolbox.set_operating_mode("manual", reason="one too many")

    assert refused["ok"] is False and "budget" in refused["error"]


def test_safe_shutdown_needs_a_reason_and_stops_the_process(toolbox: AgentToolbox) -> None:
    runtime = toolbox._runtime  # noqa: SLF001
    assert toolbox.request_safe_shutdown(reason=" ")["ok"] is False

    toolbox.request_safe_shutdown(reason="stack overheating with no cooling")

    assert runtime.constraints.electrolyser_max_kw == 0.0
    assert runtime.playing is False
    frames = runtime.advance(5)
    assert all(frame["electrolyser_power_kw"] == 0.0 for frame in frames)


def test_a_summary_never_claims_a_change_the_envelope_already_had(runtime: PlantRuntime) -> None:
    """The prose has to describe what happened, not what the rule intended.

    With the stack already capped where a degradation diagnosis would put it,
    there is nothing to apply -- and the summary must say so, because an
    operator reads the summary and believes it.
    """
    runtime.session.inject_fault({"type": "electrolyser_degradation", "severity": 0.5})
    runtime.advance(20)
    reading = AgentToolbox(runtime, AgentTrace()).inspect_component("electrolyser")
    runtime.configure_constraints({"electrolyser_max_kw": reading["min_load_kw"]})

    incident = investigate(runtime.supervisor)

    assert incident.category == "electrolyser_degradation"
    assert incident.actions == []
    assert "already holds" in incident.summary


def test_an_agent_retune_does_not_survive_a_reset_but_the_operators_does(
    runtime: PlantRuntime,
) -> None:
    """The envelope is the operator's configuration; a retune belongs to its incident."""
    runtime.configure_constraints({"dac_max_kw": 700.0})
    toolbox = AgentToolbox(runtime, AgentTrace())
    toolbox.set_controller_constraints({"electrolyser_max_kw": 800.0}, reason="degraded stack")

    runtime.reset()

    assert runtime.constraints.electrolyser_max_kw is None
    assert runtime.constraints.dac_max_kw == 700.0


def test_the_operator_takes_a_field_back_from_the_supervisor(runtime: PlantRuntime) -> None:
    toolbox = AgentToolbox(runtime, AgentTrace())
    toolbox.set_controller_constraints({"electrolyser_max_kw": 800.0}, reason="degraded stack")
    runtime.configure_constraints({"electrolyser_max_kw": 1_200.0})

    runtime.reset()

    assert runtime.constraints.electrolyser_max_kw == 1_200.0


# --- when the supervisor thinks ------------------------------------------
def test_triggers_fire_on_situations_not_on_ticks(runtime: PlantRuntime) -> None:
    watch = TriggerWatch()
    assert watch.evaluate(runtime.session.recent(60), runtime.session.events(60)) is None

    runtime.session.inject_fault({"type": "dac_outage"})
    frames = runtime.advance(3)

    trigger = watch.evaluate(frames, runtime.session.events(60))
    assert trigger is not None
    assert trigger.reason in {"critical_event", "plant_critical"}
    assert trigger.priority >= 80


def test_the_supervisors_own_events_do_not_wake_it(runtime: PlantRuntime) -> None:
    runtime.session.simulator.events.emit("warning", "agent.error", "Supervisory agent failed")
    frames = runtime.advance(1)

    assert TriggerWatch().evaluate(frames, runtime.session.events(60)) is None


def test_one_fault_does_not_start_a_run_every_tick(runtime: PlantRuntime) -> None:
    supervisor = runtime.supervisor
    runtime.session.inject_fault({"type": "dac_outage"})

    started = 0
    for _ in range(30):
        frames = runtime.advance(5)
        trigger = supervisor.notice(frames)
        if trigger is not None:
            started += 1
            investigate(supervisor, trigger)

    assert 1 <= started <= 3, f"debounce let {started} runs through"


def test_a_manual_request_always_runs(runtime: PlantRuntime) -> None:
    supervisor = runtime.supervisor
    first = investigate(supervisor)
    second = investigate(supervisor)

    assert supervisor.runs == 2
    assert first.trigger == second.trigger == "manual"


def test_a_disabled_supervisor_leaves_v5_behaviour_exactly(runtime: PlantRuntime) -> None:
    runtime.supervisor.set_enabled(False)
    runtime.session.inject_fault({"type": "electrolyser_degradation", "severity": 0.5})
    before = runtime.constraints.to_dict()

    for _ in range(20):
        runtime.supervisor.observe(runtime.advance(5))

    assert runtime.supervisor.runs == 0
    assert runtime.constraints.to_dict() == before


# --- when the model falls over -------------------------------------------
class BrokenEngine:
    """An engine that fails the way a missing API key or a dead network does."""

    label = "broken"

    async def investigate(self, toolbox: AgentToolbox, trigger: dict[str, object]) -> Diagnosis:
        raise RuntimeError("connection refused")


def test_a_failing_engine_falls_back_to_rules_and_keeps_the_plant_running(
    runtime: PlantRuntime,
) -> None:
    supervisor = SupervisorAgent(runtime, BrokenEngine(), enabled=True)
    runtime.supervisor = supervisor
    runtime.session.inject_fault({"type": "electrolyser_degradation", "severity": 0.5})
    runtime.advance(10)

    incident = investigate(supervisor)

    assert supervisor.last_error is not None
    assert incident.category == "electrolyser_degradation"  # the rule book stepped in
    assert runtime.constraints.electrolyser_max_kw is not None
    codes = [event["code"] for event in runtime.session.events(60)]
    assert "agent.error" in codes  # the failure is on the plant's event stream
    assert len(runtime.advance(5)) == 5  # and the loop is untouched


def test_a_broken_supervisor_never_breaks_the_simulation_loop(runtime: PlantRuntime) -> None:
    class Exploding(SupervisorAgent):
        def notice(self, frames: list[dict[str, object]]) -> Trigger | None:
            raise RuntimeError("supervisor is on fire")

    from plant_backend.orchestration.broadcast import Broadcaster
    from plant_backend.orchestration.loop import SimulationClock

    runtime.supervisor = Exploding(runtime, RuleEngine())
    clock = SimulationClock(runtime, Broadcaster())

    clock._supervise(runtime.advance(1))  # noqa: SLF001 - the guard under test

    assert len(runtime.advance(5)) == 5


def test_without_credentials_the_supervisor_uses_the_rule_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    assert isinstance(default_engine(), RuleEngine)


# --- what the control room sees ------------------------------------------
def test_the_trace_is_operational_and_ordered(runtime: PlantRuntime) -> None:
    runtime.session.inject_fault({"type": "electrolyser_degradation", "severity": 0.5})
    runtime.advance(10)

    incident = investigate(runtime.supervisor)
    kinds = [entry["kind"] for entry in runtime.supervisor.trace.entries_for_run(incident.run)]

    assert kinds[0] == "trigger"
    assert "tool" in kinds and "action" in kinds
    assert kinds[-2:] == ["diagnosis", "result"]
    assert incident.summary and incident.actions


def test_a_second_look_at_one_situation_replaces_the_first(runtime: PlantRuntime) -> None:
    """History is only useful if 'open' means 'still true', so a re-run supersedes."""
    runtime.session.inject_fault({"type": "electrolyser_degradation", "severity": 0.5})
    runtime.advance(10)

    first = investigate(runtime.supervisor)
    second = investigate(runtime.supervisor)

    assert first.key == second.key
    assert first.resolved is True and second.resolved is False
    assert [incident.run for incident in runtime.supervisor.trace.unresolved] == [second.run]


def test_the_status_chip_and_the_banner_cannot_disagree(runtime: PlantRuntime) -> None:
    """One drifting sensor asks for a human; both chip and banner say so, and stop together."""
    runtime.session.inject_fault({"type": "sensor_drift", "severity": 0.35, "channel": "h2_level_kg"})
    runtime.advance(20)

    investigate(runtime.supervisor)
    flagged = runtime.supervisor.snapshot()
    assert flagged["status"] == "attention"
    assert flagged["human_intervention_required"] is True

    runtime.reset()
    cleared = runtime.supervisor.snapshot()

    assert cleared["status"] == "monitoring"
    assert cleared["human_intervention_required"] is False


def test_the_payload_carries_the_agent_for_the_frontend(runtime: PlantRuntime) -> None:
    snapshot = runtime.payload()["agent"]

    assert snapshot["status"] in {"monitoring", "investigating", "attention", "disabled"}
    assert snapshot["engine"] == "rules"
    assert snapshot["incident"] is None and snapshot["trace"] == []


# --- the regression suite -------------------------------------------------
@pytest.mark.parametrize("case", evaluation.CASES, ids=lambda case: case.scenario)
def test_scenario_regression_suite(case: evaluation.Case) -> None:
    """Deterministic: the rule supervisor on the five scenarios V6 promises."""
    outcome = asyncio.run(evaluation.run_case(case))

    assert outcome.passed, f"{case.scenario}: {'; '.join(outcome.failures)}"

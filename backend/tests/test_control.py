"""V5 behaviour: the controller decides, the guard clamps, the backend drives."""

from __future__ import annotations

from dataclasses import replace

import pytest
from plant_sim import Action, PlantSimulator, PlantState

from plant_backend.controllers import Constraints, HeuristicController
from plant_backend.orchestration import PlantRuntime
from plant_backend.safety import validate

#: Midday, so there is power to allocate.
NOON_H = 12.0


@pytest.fixture
def sim() -> PlantSimulator:
    simulator = PlantSimulator()
    simulator.run(int(NOON_H * 3600 / simulator.config.dt_s), Action.idle())
    return simulator


def observe(sim: PlantSimulator, **overrides: object) -> PlantState:
    """Telemetry, optionally bent into the situation under test."""
    return replace(sim.get_telemetry(), **overrides)  # type: ignore[arg-type]


def kw_of(action: Action, bus_kw: float) -> dict[str, float]:
    return {
        "battery": action.battery_share * bus_kw,
        "electrolyser": action.electrolyser_share * bus_kw,
        "dac": action.dac_share * bus_kw,
        "methanator": action.methanator_share * bus_kw,
    }


# --- the heuristic --------------------------------------------------------
def test_heuristic_actions_are_always_valid(sim: PlantSimulator) -> None:
    """Over a full simulated day, every action is one the simulator accepts as-is."""
    controller = HeuristicController()
    constraints = Constraints()
    simulator = PlantSimulator()
    for _ in range(simulator.config.steps_per_day):
        action = controller.compute_action(
            simulator.get_telemetry(), constraints=constraints, config=simulator.config
        ).action
        shares = (
            action.battery_share,
            action.electrolyser_share,
            action.dac_share,
            action.methanator_share,
        )
        assert all(0.0 <= share <= 1.0 for share in shares)
        assert sum(shares) <= 1.0 + 1e-9
        assert action.battery_support_kw >= 0.0
        assert action == Action.sanitised(action)  # nothing left to clean up
        simulator.step(action)


def test_heuristic_runs_the_whole_chain_unattended() -> None:
    """A day under the heuristic alone produces hydrogen, CO2 and methane."""
    runtime = PlantRuntime()
    runtime.advance(runtime.config.steps_per_day)
    state = runtime.session.simulator.state
    assert state.h2_produced_total_kg > 0.0
    assert state.co2_captured_total_kg > 0.0
    assert state.ch4_produced_total_kg > 0.0


def test_heuristic_respects_power_constraints(sim: PlantSimulator) -> None:
    controller = HeuristicController()
    constraints = Constraints(electrolyser_max_kw=400.0, dac_max_kw=100.0, methanator_max_kw=50.0)
    observation = sim.get_telemetry()
    decision = controller.compute_action(observation, constraints=constraints, config=sim.config)
    demand = kw_of(decision.action, observation.solar_available_kw)
    assert demand["electrolyser"] <= 400.0 + 1e-6
    assert demand["dac"] <= 100.0 + 1e-6
    assert demand["methanator"] <= 50.0 + 1e-6


def test_heuristic_will_not_discharge_into_the_battery_reserve(sim: PlantSimulator) -> None:
    """At the reserve the stack goes without support rather than draining the pack."""
    controller = HeuristicController()
    constraints = Constraints(battery_min_soc=0.4)
    # dusk: the only way to run the stack would be to borrow from the battery
    observation = observe(sim, solar_available_kw=50.0, battery_soc=0.4)
    decision = controller.compute_action(observation, constraints=constraints, config=sim.config)
    assert decision.action.battery_support_kw == 0.0


def test_heuristic_lends_from_the_battery_above_the_reserve(sim: PlantSimulator) -> None:
    observation = observe(sim, solar_available_kw=50.0, battery_soc=0.8)
    decision = HeuristicController().compute_action(
        observation, constraints=Constraints(battery_min_soc=0.2), config=sim.config
    )
    assert decision.action.battery_support_kw > 0.0


def test_heuristic_does_not_command_offline_units(sim: PlantSimulator) -> None:
    sim.inject_fault({"type": "dac_outage"})
    sim.inject_fault({"type": "methanator_outage"})
    sim.step(Action.idle())
    observation = sim.get_telemetry()
    assert observation.health["dac"].offline
    decision = HeuristicController().compute_action(
        observation, constraints=Constraints(), config=sim.config
    )
    assert decision.action.dac_share == 0.0
    assert decision.action.methanator_share == 0.0
    assert any("offline" in note for note in decision.notes)


def test_heuristic_stops_producing_into_full_storage(sim: PlantSimulator) -> None:
    observation = observe(sim, h2_fill=1.0, co2_fill=1.0, ch4_fill=1.0)
    decision = HeuristicController().compute_action(
        observation, constraints=Constraints(), config=sim.config
    )
    assert decision.action.electrolyser_share == 0.0
    assert decision.action.dac_share == 0.0
    assert decision.action.methanator_share == 0.0


def test_heuristic_idles_at_night(sim: PlantSimulator) -> None:
    observation = observe(sim, solar_available_kw=0.0, battery_soc=0.06)
    decision = HeuristicController().compute_action(
        observation, constraints=Constraints(), config=sim.config
    )
    assert decision.action.electrolyser_share == 0.0
    assert decision.action.battery_support_kw == 0.0


# --- constraints ---------------------------------------------------------
def test_constraints_only_ever_tighten(sim: PlantSimulator) -> None:
    """Asking for more than the equipment is rated for changes nothing."""
    generous = Constraints(electrolyser_max_kw=99_000.0)
    assert generous.electrolyser_limit_kw(sim.config) == sim.config.electrolyser.rated_kw
    assert Constraints(battery_min_soc=0.0).reserve_soc(sim.config) == sim.config.battery.min_soc


def test_constraints_merge_rejects_nonsense() -> None:
    constraints = Constraints()
    assert constraints.merged({"dac_max_kw": 40.0}).dac_max_kw == 40.0
    assert constraints.merged({"dac_max_kw": None}).dac_max_kw is None
    with pytest.raises(ValueError):
        constraints.merged({"battery_min_soc": 1.4})
    with pytest.raises(ValueError):
        constraints.merged({"dac_max_kw": -1.0})
    with pytest.raises(ValueError):
        constraints.merged({"turbine_max_kw": 10.0})


# --- the guard -----------------------------------------------------------
def test_validation_clamps_power_beyond_a_constraint(sim: PlantSimulator) -> None:
    observation = sim.get_telemetry()
    validated = validate(
        Action(battery_share=0.0, electrolyser_share=1.0),
        observation,
        constraints=Constraints(electrolyser_max_kw=300.0),
        config=sim.config,
    )
    assert validated.clamped
    assert validated.action.electrolyser_share * observation.solar_available_kw <= 300.0 + 1e-6


def test_validation_drops_allocations_to_offline_units(sim: PlantSimulator) -> None:
    sim.inject_fault({"type": "methanator_outage"})
    sim.step(Action.idle())
    validated = validate(
        Action(methanator_share=0.5),
        sim.get_telemetry(),
        constraints=Constraints(),
        config=sim.config,
    )
    assert validated.action.methanator_share == 0.0
    assert any("offline" in note for note in validated.adjustments)


def test_validation_protects_the_battery_reserve(sim: PlantSimulator) -> None:
    observation = observe(sim, battery_soc=0.2)
    validated = validate(
        Action(battery_support_kw=5000.0),
        observation,
        constraints=Constraints(battery_min_soc=0.2),
        config=sim.config,
    )
    assert validated.action.battery_support_kw == 0.0
    validated = validate(
        Action(battery_support_kw=5000.0),
        observe(sim, battery_soc=0.9),
        constraints=Constraints(battery_min_soc=0.2, battery_support_max_kw=400.0),
        config=sim.config,
    )
    assert validated.action.battery_support_kw == 400.0


def test_validation_rejects_junk_without_raising(sim: PlantSimulator) -> None:
    validated = validate(
        {"battery_share": -3.0, "electrolyser_share": 4.0, "battery_support_kw": -10.0},
        sim.get_telemetry(),
        constraints=Constraints(),
        config=sim.config,
    )
    action = validated.action
    assert action.battery_share >= 0.0
    assert action.battery_support_kw == 0.0
    assert (
        action.battery_share + action.electrolyser_share + action.dac_share + action.methanator_share
        <= 1.0 + 1e-9
    )


# --- the runtime ---------------------------------------------------------
def test_runtime_reports_telemetry_not_hidden_truth() -> None:
    """A sensor fault must show up in what the frontend sees, and only there."""
    runtime = PlantRuntime()
    runtime.advance(200)
    runtime.session.inject_fault({"type": "sensor_drift", "severity": 0.8, "channel": "h2_level_kg"})
    runtime.advance(1)
    reported = runtime.payload()["state"]["h2_level_kg"]
    assert reported != pytest.approx(runtime.session.simulator.state.h2_level_kg)
    assert reported == pytest.approx(runtime.session.telemetry.h2_level_kg)


def test_manual_mode_applies_the_operator_setpoint_after_validation() -> None:
    runtime = PlantRuntime()
    runtime.advance(400)  # get to daylight
    runtime.set_mode("manual")
    runtime.set_manual_action({"battery_share": 0.0, "electrolyser_share": 1.0})
    runtime.configure_constraints({"electrolyser_max_kw": 200.0})
    runtime.advance(1)
    assert runtime.session.simulator.state.electrolyser_power_kw <= 200.0 + 1e-6
    assert runtime.adjustments  # the operator was told it was clamped


def test_switching_modes_changes_who_decides() -> None:
    runtime = PlantRuntime()
    runtime.advance(400)
    assert runtime.mode == "auto"
    runtime.advance(1)
    auto_note = runtime.notes

    runtime.set_mode("manual")
    runtime.advance(1)
    assert runtime.notes != auto_note
    assert "Manual control: operator allocation" in runtime.notes

    assert runtime.set_mode("auto") == "auto"
    with pytest.raises(ValueError):
        runtime.set_mode("autopilot")


def test_pause_stops_simulated_time() -> None:
    """The clock is the only thing that advances the plant; pausing stops it."""
    runtime = PlantRuntime()
    runtime.play()
    runtime.advance(10)
    at_pause = runtime.session.simulator.time_s
    runtime.pause()
    assert not runtime.playing
    assert runtime.session.simulator.time_s == at_pause


def test_reset_clears_the_run_but_keeps_the_operator_settings() -> None:
    runtime = PlantRuntime()
    runtime.set_mode("manual")
    runtime.configure_constraints({"dac_max_kw": 100.0})
    runtime.play()
    runtime.advance(50)
    runtime.session.inject_fault({"type": "dac_outage"})

    runtime.reset()

    assert not runtime.playing
    assert runtime.session.simulator.time_s == 0.0
    assert runtime.session.faults == []
    assert runtime.notes == ()
    assert runtime.applied_action == Action.idle()
    assert len(runtime.session.history) == 1
    # configuration is not part of the run
    assert runtime.mode == "manual"
    assert runtime.constraints.dac_max_kw == 100.0


def test_speed_is_clamped_and_scales_steps_per_tick() -> None:
    runtime = PlantRuntime()
    assert runtime.set_speed(16) == 16
    assert runtime.set_speed(9999) == 64
    assert runtime.set_speed(0) == 1
    before = runtime.session.simulator.step_index
    runtime.advance(runtime.set_speed(16))
    assert runtime.session.simulator.step_index - before == 16

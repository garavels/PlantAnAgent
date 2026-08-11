"""Behavioural tests for the digital twin: limits, conservation, determinism."""

from __future__ import annotations

import math

import pytest

from plant_sim import Action, PlantConfig, PlantSimulator, StorageConfig
from plant_sim.config import BatteryConfig, SolarConfig

NOON = SolarConfig().sunrise_h + (SolarConfig().sunset_h - SolarConfig().sunrise_h) / 2


@pytest.fixture
def sim() -> PlantSimulator:
    return PlantSimulator()


def advance_to_hour(sim: PlantSimulator, hour: float, action: Action | None = None) -> None:
    """Fast-forward to a given time of day (day 1)."""
    steps = int(hour * 3600 / sim.config.dt_s)
    sim.run(steps, action or Action.idle())


# --- solar ----------------------------------------------------------------
def test_solar_is_dark_at_night_and_peaks_at_solar_noon(sim: PlantSimulator) -> None:
    assert sim.solar.power_at(3 * 3600) == 0.0
    assert sim.solar.power_at(23 * 3600) == 0.0
    assert sim.solar.power_at(NOON * 3600) == pytest.approx(sim.config.solar.peak_kw)


def test_solar_curve_repeats_every_day(sim: PlantSimulator) -> None:
    assert sim.solar.power_at(10 * 3600) == pytest.approx(sim.solar.power_at(34 * 3600))


# --- battery --------------------------------------------------------------
def test_battery_never_charges_past_capacity() -> None:
    config = PlantConfig(battery=BatteryConfig(capacity_kwh=100.0, initial_soc=0.99))
    sim = PlantSimulator(config)
    advance_to_hour(sim, NOON, Action.all_to_battery())
    assert sim.battery.soc <= 1.0
    assert sim.state.solar_curtailed_kw > 0  # nowhere left to put the power


def test_battery_never_discharges_past_min_soc() -> None:
    config = PlantConfig(battery=BatteryConfig(initial_soc=0.06, min_soc=0.05))
    sim = PlantSimulator(config)
    advance_to_hour(sim, 2.0, Action(battery_share=0.0, electrolyser_share=0.0, battery_support_kw=2_000.0))
    assert sim.battery.soc >= config.battery.min_soc - 1e-9


def test_battery_charging_respects_power_limit() -> None:
    sim = PlantSimulator()
    advance_to_hour(sim, NOON, Action.all_to_battery())
    assert sim.state.solar_to_battery_kw <= sim.config.battery.max_charge_kw + 1e-9


def test_battery_supports_the_stack_at_night(sim: PlantSimulator) -> None:
    advance_to_hour(sim, 22.0, Action(battery_support_kw=1_000.0))
    assert sim.state.solar_available_kw == 0.0
    assert sim.state.battery_to_electrolyser_kw == pytest.approx(1_000.0)
    assert sim.state.electrolyser_online
    assert sim.state.h2_production_kg_h > 0


# --- electrolyser ---------------------------------------------------------
def test_stack_stays_off_below_minimum_load(sim: PlantSimulator) -> None:
    # early morning: a trickle of solar, all of it aimed at the stack
    advance_to_hour(sim, 6.2, Action.all_to_electrolyser())
    assert 0 < sim.state.solar_available_kw < sim.electrolyser.min_load_kw
    assert not sim.state.electrolyser_online
    assert sim.state.h2_production_kg_h == 0.0


def test_stack_never_exceeds_rated_power(sim: PlantSimulator) -> None:
    advance_to_hour(sim, NOON, Action.all_to_electrolyser())
    assert sim.state.electrolyser_power_kw <= sim.config.electrolyser.rated_kw + 1e-9
    assert sim.state.solar_curtailed_kw > 0 or sim.state.solar_to_battery_kw > 0


def test_hydrogen_matches_energy_consumed(sim: PlantSimulator) -> None:
    advance_to_hour(sim, NOON, Action.all_to_electrolyser())
    state = sim.step(Action.all_to_electrolyser())
    expected = state.electrolyser_power_kw / sim.config.electrolyser.specific_energy_kwh_per_kg
    assert state.h2_production_kg_h == pytest.approx(expected)


# --- storage --------------------------------------------------------------
def test_tank_never_overfills_and_throttles_the_stack() -> None:
    config = PlantConfig(h2_tank=StorageConfig(capacity_kg=5.0, initial_kg=0.0))
    sim = PlantSimulator(config)
    advance_to_hour(sim, NOON, Action.all_to_electrolyser())
    # a tiny tank fills within minutes, then the stack has nowhere to put the H2:
    # it cannot turn down below its minimum load, so it shuts off entirely
    assert sim.h2_tank.level_kg <= config.h2_tank.capacity_kg + 1e-9
    assert sim.state.h2_fill > 0.95
    assert sim.state.electrolyser_power_kw == 0.0
    assert sim.state.solar_curtailed_kw > 0


# --- action handling ------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"battery_share": 5.0, "electrolyser_share": 0.0}, (1.0, 0.0)),
        ({"battery_share": -3.0, "electrolyser_share": 0.5}, (0.0, 0.5)),
        ({"battery_share": 1.0, "electrolyser_share": 1.0}, (0.5, 0.5)),  # normalised
        ({"battery_share": float("nan"), "electrolyser_share": 0.2}, (0.0, 0.2)),
        ({"battery_share": "nonsense", "electrolyser_share": None}, (0.0, 0.0)),
    ],
)
def test_invalid_actions_are_constrained(raw: dict, expected: tuple[float, float]) -> None:
    action = Action.sanitised(raw)
    assert (action.battery_share, action.electrolyser_share) == pytest.approx(expected)
    assert action.battery_share + action.electrolyser_share <= 1.0 + 1e-9


def test_simulator_accepts_garbage_actions_without_breaking(sim: PlantSimulator) -> None:
    for garbage in (None, {}, {"battery_share": float("inf")}, Action(2.0, 2.0, -5.0)):
        state = sim.step(garbage)
        assert math.isfinite(state.battery_soc)
        assert 0.0 <= state.battery_soc <= 1.0
        assert state.battery_to_electrolyser_kw >= 0.0


# --- invariants -----------------------------------------------------------
def test_solar_is_always_fully_accounted_for(sim: PlantSimulator) -> None:
    for index in range(sim.config.steps_per_day):
        state = sim.step(Action(battery_share=0.3, electrolyser_share=0.5, battery_support_kw=index % 400))
        routed = state.solar_to_battery_kw + state.solar_to_electrolyser_kw + state.solar_curtailed_kw
        assert routed == pytest.approx(state.solar_available_kw, abs=1e-6)
        assert state.electrolyser_power_kw == pytest.approx(
            state.solar_to_electrolyser_kw + state.battery_to_electrolyser_kw, abs=1e-6
        )


def test_battery_never_charges_and_discharges_at_once(sim: PlantSimulator) -> None:
    for _ in range(sim.config.steps_per_day):
        state = sim.step(Action(battery_share=0.5, electrolyser_share=0.5, battery_support_kw=800.0))
        assert not (state.solar_to_battery_kw > 0 and state.battery_to_electrolyser_kw > 0)


def test_reset_restores_the_initial_condition(sim: PlantSimulator) -> None:
    advance_to_hour(sim, 12.0, Action(battery_share=0.5, electrolyser_share=0.5))
    state = sim.reset()
    assert state.step == 0
    assert state.time_s == 0.0
    assert state.battery_soc == pytest.approx(sim.config.battery.initial_soc)
    assert state.h2_level_kg == pytest.approx(sim.config.h2_tank.initial_kg)
    assert state.h2_produced_total_kg == 0.0


def test_runs_are_deterministic() -> None:
    action = Action(battery_share=0.35, electrolyser_share=0.65)
    runs = []
    for _ in range(2):
        sim = PlantSimulator()
        runs.append(sim.run(sim.config.steps_per_day, action).to_dict())
    assert runs[0] == runs[1]


def test_a_full_day_actually_makes_hydrogen(sim: PlantSimulator) -> None:
    sim.run(sim.config.steps_per_day, Action(battery_share=0.3, electrolyser_share=0.7, battery_support_kw=600.0))
    assert sim.state.h2_produced_total_kg > 100.0
    assert sim.state.day == 1
    assert sim.state.clock.startswith("D2 00:00")

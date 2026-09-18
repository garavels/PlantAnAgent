"""Behavioural tests for the digital twin: limits, conservation, determinism."""

from __future__ import annotations

import math

import pytest

from plant_sim import Action, PlantConfig, PlantSimulator, StorageConfig, dispatch
from plant_sim.config import BatteryConfig, MethanatorConfig, SolarConfig

NOON = SolarConfig().sunrise_h + (SolarConfig().sunset_h - SolarConfig().sunrise_h) / 2

#: A balanced daytime split for the full four-consumer bus.
FULL_MIX = Action(battery_share=0.2, electrolyser_share=0.45, dac_share=0.2, methanator_share=0.15)


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


def test_methanator_round_trip_never_manufactures_bus_power_or_negative_spill(
    sim: PlantSimulator,
) -> None:
    # This exact binary64 offer previously reconstructed one ULP too high after
    # kW -> kg -> kW conversion, exposing impossible negative curtailment.
    offered_kw = float.fromhex("0x1.2e35a8ccf9062p+6")
    flows = dispatch(
        Action(
            battery_share=0.0,
            electrolyser_share=0.0,
            dac_share=0.0,
            methanator_share=1.0,
        ),
        offered_kw,
        battery=sim.battery,
        electrolyser=sim.electrolyser,
        h2_tank=sim.h2_tank,
        dac=sim.dac,
        co2_tank=sim.co2_tank,
        methanator=sim.methanator,
        ch4_tank=sim.ch4_tank,
        dt_h=sim.config.dt_h,
    )

    assert flows.methanator_power_kw == offered_kw
    assert flows.solar_to_methanator_kw == offered_kw
    assert flows.solar_to_battery_kw == 0.0
    assert flows.solar_curtailed_kw == 0.0


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
    total = action.battery_share + action.electrolyser_share + action.dac_share + action.methanator_share
    assert total <= 1.0 + 1e-9


def test_a_four_way_oversubscription_is_normalised() -> None:
    action = Action.sanitised(
        {"battery_share": 1.0, "electrolyser_share": 1.0, "dac_share": 1.0, "methanator_share": 1.0}
    )
    assert (action.battery_share, action.electrolyser_share, action.dac_share, action.methanator_share) == (
        pytest.approx(0.25),
        pytest.approx(0.25),
        pytest.approx(0.25),
        pytest.approx(0.25),
    )


def test_simulator_accepts_garbage_actions_without_breaking(sim: PlantSimulator) -> None:
    for garbage in (None, {}, {"battery_share": float("inf")}, Action(2.0, 2.0, -5.0)):
        state = sim.step(garbage)
        assert math.isfinite(state.battery_soc)
        assert 0.0 <= state.battery_soc <= 1.0
        assert state.battery_to_electrolyser_kw >= 0.0


# --- invariants -----------------------------------------------------------
def test_solar_is_always_fully_accounted_for(sim: PlantSimulator) -> None:
    for index in range(sim.config.steps_per_day):
        state = sim.step(
            Action(
                battery_share=0.3,
                electrolyser_share=0.5,
                dac_share=0.2,
                methanator_share=0.1,
                battery_support_kw=index % 400,
            )
        )
        routed = (
            state.solar_to_battery_kw
            + state.solar_to_electrolyser_kw
            + state.solar_to_dac_kw
            + state.solar_to_methanator_kw
            + state.solar_curtailed_kw
        )
        assert routed == pytest.approx(state.solar_available_kw, abs=1e-6)
        assert state.electrolyser_power_kw == pytest.approx(
            state.solar_to_electrolyser_kw + state.battery_to_electrolyser_kw, abs=1e-6
        )


def test_battery_never_charges_and_discharges_at_once(sim: PlantSimulator) -> None:
    for _ in range(sim.config.steps_per_day):
        state = sim.step(Action(battery_share=0.5, electrolyser_share=0.5, battery_support_kw=800.0))
        assert not (state.solar_to_battery_kw > 0 and state.battery_to_electrolyser_kw > 0)


def test_reset_restores_the_initial_condition(sim: PlantSimulator) -> None:
    advance_to_hour(sim, 12.0, FULL_MIX)
    state = sim.reset()
    assert state.step == 0
    assert state.time_s == 0.0
    assert state.battery_soc == pytest.approx(sim.config.battery.initial_soc)
    assert state.h2_level_kg == pytest.approx(sim.config.h2_tank.initial_kg)
    assert state.co2_level_kg == pytest.approx(sim.config.co2_tank.initial_kg)
    assert state.ch4_level_kg == pytest.approx(sim.config.ch4_tank.initial_kg)
    assert state.h2_produced_total_kg == 0.0
    assert state.co2_captured_total_kg == 0.0
    assert state.ch4_produced_total_kg == 0.0
    assert not (state.electrolyser_online or state.dac_online or state.methanator_online)


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


# --- DAC --------------------------------------------------------------------
def test_dac_captures_in_proportion_to_power(sim: PlantSimulator) -> None:
    advance_to_hour(sim, NOON, Action(battery_share=0.0, electrolyser_share=0.0, dac_share=0.5))
    state = sim.step(Action(dac_share=0.5))
    assert state.dac_online
    assert state.dac_power_kw == pytest.approx(state.solar_to_dac_kw)
    expected = state.dac_power_kw / sim.config.dac.specific_energy_kwh_per_kg
    assert state.co2_capture_kg_h == pytest.approx(expected)


def test_dac_never_exceeds_rated_power(sim: PlantSimulator) -> None:
    advance_to_hour(sim, NOON, Action(battery_share=0.0, electrolyser_share=0.0, dac_share=1.0))
    assert sim.state.dac_power_kw <= sim.config.dac.rated_kw + 1e-9


def test_dac_throttles_when_co2_tank_is_full() -> None:
    config = PlantConfig(co2_tank=StorageConfig(capacity_kg=50.0, initial_kg=50.0))
    sim = PlantSimulator(config)
    advance_to_hour(sim, NOON, Action(battery_share=0.0, electrolyser_share=0.0, dac_share=1.0))
    assert sim.state.co2_level_kg <= config.co2_tank.capacity_kg + 1e-9
    assert sim.state.dac_power_kw == 0.0  # nowhere to put the CO2
    assert not sim.state.dac_online


# --- methanation -------------------------------------------------------------
def test_methanation_produces_ch4_from_stored_gases(sim: PlantSimulator) -> None:
    # idle until noon so the morning sun does not drain the initial stocks
    advance_to_hour(sim, NOON)
    state = sim.step(Action(battery_share=0.0, electrolyser_share=0.0, methanator_share=0.5))
    assert state.methanator_online
    assert state.ch4_production_kg_h > 0.0
    assert state.ch4_level_kg > sim.config.ch4_tank.initial_kg


def test_methanation_stops_without_hydrogen() -> None:
    config = PlantConfig(h2_tank=StorageConfig(capacity_kg=100.0, initial_kg=0.0))
    sim = PlantSimulator(config)
    advance_to_hour(sim, NOON, Action(battery_share=0.0, electrolyser_share=0.0, methanator_share=0.5))
    assert sim.state.ch4_production_kg_h == 0.0
    assert not sim.state.methanator_online
    assert sim.state.methanator_power_kw == 0.0  # offered power is refused, not burned


def test_methanation_stops_without_co2() -> None:
    config = PlantConfig(co2_tank=StorageConfig(capacity_kg=100.0, initial_kg=0.0))
    sim = PlantSimulator(config)
    advance_to_hour(sim, NOON, Action(battery_share=0.0, electrolyser_share=0.0, methanator_share=0.5))
    assert sim.state.ch4_production_kg_h == 0.0
    assert not sim.state.methanator_online


def test_methanation_is_limited_by_the_scarcest_gas() -> None:
    # plenty of power and H2, almost no CO2: CO2 must set the pace
    config = PlantConfig(
        h2_tank=StorageConfig(capacity_kg=500.0, initial_kg=500.0),
        co2_tank=StorageConfig(capacity_kg=100.0, initial_kg=27.5),
        methanator=MethanatorConfig(rated_kw=1_000.0),
    )
    sim = PlantSimulator(config)
    advance_to_hour(sim, NOON, Action(battery_share=0.0, electrolyser_share=0.0, methanator_share=1.0))
    # 27.5 kg CO2 at 2.75 kg per kg CH4 -> exactly 10 kg CH4 before starvation
    assert sim.state.ch4_produced_total_kg == pytest.approx(10.0, abs=1e-6)
    assert sim.state.co2_level_kg == pytest.approx(0.0, abs=1e-9)
    assert sim.state.ch4_production_kg_h == 0.0
    assert sim.state.h2_level_kg == pytest.approx(500.0 - 5.0, abs=1e-6)


def test_methanation_consumes_the_fixed_recipe(sim: PlantSimulator) -> None:
    advance_to_hour(sim, NOON)
    h2_before, co2_before = sim.h2_tank.level_kg, sim.co2_tank.level_kg
    state = sim.step(Action(battery_share=0.0, electrolyser_share=0.0, methanator_share=0.4))
    dt_h = sim.config.dt_h
    ch4_kg = state.ch4_production_kg_h * dt_h
    assert ch4_kg > 0.0
    recipe = sim.config.methanator
    assert h2_before - sim.h2_tank.level_kg == pytest.approx(ch4_kg * recipe.h2_kg_per_kg_ch4)
    assert co2_before - sim.co2_tank.level_kg == pytest.approx(ch4_kg * recipe.co2_kg_per_kg_ch4)
    expected_kw = ch4_kg * recipe.specific_energy_kwh_per_kg / dt_h
    assert state.methanator_power_kw == pytest.approx(expected_kw)


def test_ch4_tank_throttles_methanation() -> None:
    config = PlantConfig(ch4_tank=StorageConfig(capacity_kg=20.0, initial_kg=20.0))
    sim = PlantSimulator(config)
    advance_to_hour(sim, NOON, Action(battery_share=0.0, electrolyser_share=0.0, methanator_share=0.5))
    assert sim.state.ch4_level_kg <= config.ch4_tank.capacity_kg + 1e-9
    assert sim.state.ch4_production_kg_h == 0.0


# --- storage bounds under load ------------------------------------------------
def test_gas_storages_stay_within_bounds_all_day(sim: PlantSimulator) -> None:
    for _ in range(sim.config.steps_per_day):
        state = sim.step(FULL_MIX)
        assert 0.0 <= state.h2_level_kg <= state.h2_capacity_kg + 1e-9
        assert 0.0 <= state.co2_level_kg <= state.co2_capacity_kg + 1e-9
        assert 0.0 <= state.ch4_level_kg <= state.ch4_capacity_kg + 1e-9


def test_a_full_day_actually_makes_methane(sim: PlantSimulator) -> None:
    sim.run(
        sim.config.steps_per_day,
        Action(battery_share=0.2, electrolyser_share=0.5, dac_share=0.2, methanator_share=0.1, battery_support_kw=600.0),
    )
    state = sim.state
    assert state.ch4_produced_total_kg > 100.0
    assert state.co2_captured_total_kg > 0.0
    # methane output never outruns what the feedstocks could have supplied
    assert state.ch4_produced_total_kg * sim.config.methanator.h2_kg_per_kg_ch4 <= (
        state.h2_produced_total_kg + sim.config.h2_tank.initial_kg + 1e-6
    )

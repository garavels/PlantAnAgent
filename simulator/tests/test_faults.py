"""V4 behaviour: faults change physics, telemetry can lie, scenarios repeat."""

from __future__ import annotations

import pytest

from plant_sim import Action, FaultSpec, PlantConfig, PlantSimulator
from plant_sim.config import BatteryConfig, StorageConfig
from plant_sim.events import EVENT_LIMIT

#: Midday, so the stack and DAC have power to work with.
NOON_H = 12.0
FULL_MIX = Action(battery_share=0.15, electrolyser_share=0.45, dac_share=0.25, methanator_share=0.15)


@pytest.fixture
def sim() -> PlantSimulator:
    return PlantSimulator()


def at_hour(sim: PlantSimulator, hour: float, action: Action | None = None) -> None:
    sim.run(int(hour * 3600 / sim.config.dt_s), action or Action.idle())


def run_hours(sim: PlantSimulator, hours: float, action: Action) -> None:
    sim.run(int(hours * 3600 / sim.config.dt_s), action)


# --- electrolyser ---------------------------------------------------------
def test_degraded_stack_makes_less_h2_from_the_same_power() -> None:
    healthy, degraded = PlantSimulator(), PlantSimulator()
    at_hour(healthy, NOON_H)
    at_hour(degraded, NOON_H)
    degraded.inject_fault(FaultSpec(type="electrolyser_degradation", severity=0.4))

    run_hours(healthy, 1.0, FULL_MIX)
    run_hours(degraded, 1.0, FULL_MIX)

    assert degraded.state.electrolyser_power_kw == pytest.approx(
        healthy.state.electrolyser_power_kw, rel=1e-6
    )
    assert degraded.state.h2_production_kg_h < healthy.state.h2_production_kg_h * 0.7
    assert degraded.state.electrolyser_temperature_c > healthy.state.electrolyser_temperature_c


def test_degradation_severity_scales_the_yield_loss(sim: PlantSimulator) -> None:
    at_hour(sim, NOON_H)
    sim.inject_fault(FaultSpec(type="electrolyser_degradation", severity=0.5))
    sim.step(FULL_MIX)
    assert sim.state.electrolyser_efficiency == pytest.approx(0.5)
    assert sim.state.health["electrolyser"].status == "degraded"


# --- DAC ------------------------------------------------------------------
def test_dac_outage_stops_capture(sim: PlantSimulator) -> None:
    at_hour(sim, NOON_H)
    sim.inject_fault(FaultSpec(type="dac_outage"))
    before = sim.co2_tank.level_kg
    run_hours(sim, 1.0, FULL_MIX)

    assert sim.state.co2_capture_kg_h == 0.0
    assert sim.state.dac_power_kw == 0.0
    assert not sim.state.dac_online
    assert sim.co2_tank.level_kg <= before


def test_dac_degradation_only_reduces_capture(sim: PlantSimulator) -> None:
    at_hour(sim, NOON_H)
    sim.inject_fault(FaultSpec(type="dac_degradation", severity=0.5))
    run_hours(sim, 1.0, FULL_MIX)
    assert 0.0 < sim.state.co2_capture_kg_h
    assert sim.state.dac_online


def test_dac_outage_eventually_starves_the_methanator() -> None:
    config = PlantConfig(co2_tank=StorageConfig(capacity_kg=400.0, initial_kg=8.0))
    sim = PlantSimulator(config)
    at_hour(sim, NOON_H)
    sim.inject_fault(FaultSpec(type="dac_outage"))
    run_hours(sim, 6.0, FULL_MIX)

    assert sim.co2_tank.level_kg < 1.0
    assert sim.state.ch4_production_kg_h == 0.0
    assert any(event.code == "methanator.co2_starved" for event in sim.events.events)


# --- battery --------------------------------------------------------------
def test_battery_degradation_shrinks_capacity_and_clamps_stored_energy() -> None:
    config = PlantConfig(battery=BatteryConfig(capacity_kwh=100.0, initial_soc=0.9))
    sim = PlantSimulator(config)
    sim.inject_fault(FaultSpec(type="battery_degradation", severity=0.3))

    assert sim.battery.capacity_kwh == pytest.approx(70.0)
    assert sim.battery.energy_kwh == pytest.approx(70.0)
    assert sim.battery.soc == pytest.approx(1.0)
    sim.step(Action.idle())
    assert sim.state.battery_capacity_kwh == pytest.approx(70.0)
    assert sim.battery.energy_kwh <= sim.battery.capacity_kwh + 1e-9


# --- methanator -----------------------------------------------------------
def test_methanator_outage_stops_ch4_without_eating_feedstock() -> None:
    sim = PlantSimulator()
    at_hour(sim, NOON_H, FULL_MIX)
    sim.inject_fault(FaultSpec(type="methanator_outage"))
    h2_before, co2_before = sim.h2_tank.level_kg, sim.co2_tank.level_kg

    sim.step(Action(methanator_share=1.0))

    assert sim.state.ch4_production_kg_h == 0.0
    assert sim.state.methanator_power_kw == 0.0
    assert sim.h2_tank.level_kg >= h2_before
    assert sim.co2_tank.level_kg >= co2_before


# --- disturbance ----------------------------------------------------------
def test_solar_drop_reduces_generation_below_clear_sky(sim: PlantSimulator) -> None:
    at_hour(sim, NOON_H)
    clear_sky = sim.state.solar_available_kw
    sim.inject_fault(FaultSpec(type="solar_drop", severity=0.6, duration_s=3 * 3600))
    sim.step(FULL_MIX)

    assert sim.state.solar_available_kw == pytest.approx(sim.state.solar_potential_kw * 0.4, rel=1e-6)
    assert sim.state.solar_available_kw < clear_sky
    assert sim.state.solar_potential_kw > sim.state.solar_available_kw


def test_a_disturbance_ends_after_its_duration(sim: PlantSimulator) -> None:
    at_hour(sim, 9.0)
    sim.inject_fault(FaultSpec(type="solar_drop", severity=0.6, duration_s=3600))
    sim.step(FULL_MIX)
    assert sim.state.solar_output_factor < 1.0

    run_hours(sim, 1.5, FULL_MIX)
    assert sim.state.solar_output_factor == pytest.approx(1.0)
    assert sim.state.health["solar"].status == "nominal"


# --- scheduling and clearing ---------------------------------------------
def test_a_scheduled_fault_waits_for_its_start_time(sim: PlantSimulator) -> None:
    at_hour(sim, NOON_H)
    sim.inject_fault(FaultSpec(type="dac_outage", start_s=sim.time_s + 2 * 3600))
    sim.step(FULL_MIX)
    assert sim.state.health["dac"].status == "nominal"

    run_hours(sim, 2.5, FULL_MIX)
    assert sim.state.health["dac"].status == "offline"


def test_clearing_a_fault_restores_the_component(sim: PlantSimulator) -> None:
    at_hour(sim, NOON_H)
    fault = sim.inject_fault(FaultSpec(type="electrolyser_degradation", severity=0.5))
    sim.step(FULL_MIX)
    assert sim.state.electrolyser_efficiency < 1.0

    assert sim.clear_fault(fault.id)
    sim.step(FULL_MIX)
    assert sim.state.electrolyser_efficiency == pytest.approx(1.0)
    assert not sim.clear_fault("nope")


def test_reset_clears_faults_scenario_and_events(sim: PlantSimulator) -> None:
    sim.load_scenario("dac_failure")
    sim.run(60, FULL_MIX)
    sim.reset()

    assert sim.scenario is None
    assert sim.faults.to_list() == []
    assert sim.events.events == ()
    assert sim.state.plant_status == "normal"


# --- telemetry ------------------------------------------------------------
def test_a_sensor_fault_biases_telemetry_but_not_the_truth() -> None:
    config = PlantConfig(h2_tank=StorageConfig(capacity_kg=1000.0, initial_kg=550.0))
    sim = PlantSimulator(config)
    sim.inject_fault(FaultSpec(type="sensor_drift", severity=0.17, channel="h2_level_kg"))
    sim.step(Action.idle())

    truth, observed = sim.state, sim.get_telemetry()
    assert truth.h2_level_kg == pytest.approx(sim.h2_tank.level_kg)
    assert observed.h2_level_kg > truth.h2_level_kg
    assert observed.h2_fill == pytest.approx(observed.h2_level_kg / 1000.0)
    # everything the sensor does not touch still agrees
    assert observed.battery_soc == truth.battery_soc


def test_telemetry_matches_truth_without_sensor_faults(sim: PlantSimulator) -> None:
    run_hours(sim, 2.0, FULL_MIX)
    assert sim.get_telemetry().to_dict() == sim.state.to_dict()


# --- scenarios ------------------------------------------------------------
@pytest.mark.parametrize(
    "scenario_id",
    ["clear_day", "electrolyser_degradation", "cloud_cover", "dac_failure", "multiple_problems"],
)
def test_every_scenario_is_reproducible(scenario_id: str) -> None:
    def trace(sim: PlantSimulator) -> list[tuple[float, float, str]]:
        sim.load_scenario(scenario_id)
        out = []
        for _ in range(180):
            state = sim.step(FULL_MIX)
            out.append((state.h2_level_kg, state.ch4_level_kg, state.plant_status))
        return out

    assert trace(PlantSimulator()) == trace(PlantSimulator())


def test_a_scenario_schedules_its_faults_relative_to_its_start() -> None:
    sim = PlantSimulator()
    sim.load_scenario("electrolyser_degradation")
    assert sim.scenario == "electrolyser_degradation"
    assert sim.state.health["electrolyser"].status == "nominal"

    run_hours(sim, 3.5, FULL_MIX)
    assert sim.state.health["electrolyser"].status == "degraded"


def test_a_faulty_scenario_diverges_from_the_clear_day() -> None:
    clear, faulty = PlantSimulator(), PlantSimulator()
    clear.load_scenario("clear_day")
    faulty.load_scenario("electrolyser_degradation")
    run_hours(clear, 8.0, FULL_MIX)
    run_hours(faulty, 8.0, FULL_MIX)

    assert faulty.state.h2_produced_total_kg < clear.state.h2_produced_total_kg
    assert faulty.state.plant_status == "degraded"
    assert clear.state.plant_status == "normal"


# --- events ---------------------------------------------------------------
def test_faults_and_conditions_emit_events(sim: PlantSimulator) -> None:
    at_hour(sim, NOON_H)
    sim.inject_fault(FaultSpec(type="methanator_outage"))
    sim.step(FULL_MIX)

    codes = [event.code for event in sim.events.events]
    assert "fault.methanator_outage" in codes
    assert any(event.level == "critical" for event in sim.events.events)
    assert all(event.clock for event in sim.events.events)


def test_the_event_log_stays_bounded(sim: PlantSimulator) -> None:
    run_hours(sim, 48.0, FULL_MIX)
    assert len(sim.events.events) <= EVENT_LIMIT

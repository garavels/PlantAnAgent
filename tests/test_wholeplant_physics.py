"""Conservation and counterfactual transition checks; no held-out scenarios."""
from dataclasses import fields, replace
import unittest
from unittest.mock import patch

import numpy as np

from docs.research.tools.wholeplant_ppo_v1.physics import (
    Action, WholePlantAction, WholePlantSimulator, WholePlantState, DirectPlantSimulator,
    DynamicDAC, DynamicMethanator, DAC_READINESS, METHANATOR_READINESS,
    action_to_vector, decode_action, _frozen_bootstrap,
)
from plant_sim.config import PlantConfig, SolarConfig
from plant_sim.faults import ComponentEffect
from plant_sim.state import PlantState


class WholePlantPhysicsTests(unittest.TestCase):
    def assert_balance(self, before, after, simulator):
        dt = simulator.config.dt_h
        self.assertAlmostEqual(after.solar_available_kw,
            after.solar_to_battery_kw + after.solar_to_electrolyser_kw + after.solar_to_dac_kw
            + after.solar_to_methanator_kw + after.solar_curtailed_kw, places=8)
        self.assertAlmostEqual(after.electrolyser_power_kw,
            after.solar_to_electrolyser_kw + after.battery_to_electrolyser_kw, places=8)
        self.assertAlmostEqual(after.dac_power_kw, after.solar_to_dac_kw + after.battery_to_dac_kw, places=8)
        self.assertAlmostEqual(after.methanator_power_kw,
            after.solar_to_methanator_kw + after.battery_to_methanator_kw, places=8)
        discharged = after.battery_to_electrolyser_kw + after.battery_to_dac_kw + after.battery_to_methanator_kw
        self.assertLessEqual(discharged, simulator.config.battery.max_discharge_kw + 1e-8)
        self.assertFalse(discharged > 1e-8 and after.solar_to_battery_kw > 1e-8)
        self.assertAlmostEqual(after.battery_power_kw, after.solar_to_battery_kw - discharged, places=8)
        expected_energy = (before.battery_energy_kwh + after.solar_to_battery_kw * dt
            * simulator.config.battery.charge_efficiency - discharged * dt
            / simulator.config.battery.discharge_efficiency)
        self.assertAlmostEqual(after.battery_energy_kwh, expected_energy, places=7)
        self.assertGreaterEqual(after.battery_soc, simulator.config.battery.min_soc - 1e-10)
        ch4 = after.ch4_production_kg_h * dt
        self.assertAlmostEqual(after.h2_level_kg - before.h2_level_kg,
            after.h2_production_kg_h * dt - simulator.config.methanator.h2_kg_per_kg_ch4 * ch4, places=8)
        self.assertAlmostEqual(after.co2_level_kg - before.co2_level_kg,
            after.co2_capture_kg_h * dt - simulator.config.methanator.co2_kg_per_kg_ch4 * ch4, places=8)
        self.assertAlmostEqual(after.ch4_level_kg - before.ch4_level_kg,
                               ch4 - simulator.delivery_step.delivered_kg, places=8)
        for name, profile in (("dac", DAC_READINESS), ("methanator", METHANATOR_READINESS)):
            start, standby, productive, loss = [getattr(after, f"{name}_{key}") for key in
                ("startup_power_kw", "standby_power_kw", "productive_power_kw", "heat_loss_kw")]
            self.assertAlmostEqual(getattr(after, f"{name}_power_kw"), start + standby + productive, places=8)
            q_before, q_after = [getattr(state, f"{name}_readiness_kwh") for state in (before, after)]
            self.assertAlmostEqual(q_after, q_before + (start + standby - loss) * dt, places=8)
            self.assertGreaterEqual(q_after, -1e-9)
            self.assertLessEqual(q_after, profile.capacity_kwh + 1e-9)
            if simulator.downstream_dynamics:
                self.assertAlmostEqual(loss, min(profile.standby_kw, q_before / dt + start + standby), places=8)
                if productive > 1e-8:
                    self.assertAlmostEqual(q_before, profile.capacity_kwh, places=8)
                self.assertFalse(start > 1e-8 and productive > 1e-8)
                if name == "dac":
                    self.assertAlmostEqual(after.co2_capture_kg_h,
                        productive / simulator.dac.productive_specific_energy_kwh_per_kg, places=8)
                else:
                    self.assertAlmostEqual(after.ch4_production_kg_h,
                        productive / simulator.methanator.productive_specific_energy_kwh_per_kg, places=8)

    def test_decode_inverse_and_joint_cap(self):
        action, evidence = decode_action([1.0] * 7)
        self.assertEqual(evidence["solar_normalization_divisor"], 4.0)
        self.assertEqual(evidence["battery_normalization_divisor"], 3.0)
        self.assertAlmostEqual(action.battery_support_kw + action.battery_dac_support_kw
                               + action.battery_methanator_support_kw, 2500)
        restored, _ = decode_action(action_to_vector(action))
        np.testing.assert_allclose(list(restored.to_dict().values()), list(action.to_dict().values()), atol=1e-12)
        for bad in ([0.0] * 5, [0.0] * 6 + [float("nan")], ["x"] * 7):
            with self.assertRaises(ValueError):
                decode_action(bad)

    def test_restricted_instant_is_old_transition(self):
        cfg = _frozen_bootstrap.ControllerNeutralTrainingConfigV1.from_dict(
            _frozen_bootstrap.legacy.load_benchmark_contract()["source_template"])
        for commitment in (None, cfg.episode.electrolyser_commitment):
            old = DirectPlantSimulator(cfg.plant, electrolyser_commitment=commitment)
            new = WholePlantSimulator(cfg.plant, electrolyser_commitment=commitment,
                                      shared_bus=False, downstream_dynamics=False)
            rng = np.random.default_rng(30000123)
            for simulator in (old, new):
                simulator.inject_fault({"type": "dac_degradation", "severity": .35,
                                        "start_s": 10000, "duration_s": 10000})
                simulator.set_delivery_schedule([{"id": "physics-fixture", "due_step": 900,
                                                  "requested_kg": 100.0}])
            for _ in range(1500):
                values = rng.uniform(-1, 1, 5)
                action, _ = _frozen_bootstrap.decode_action(values)
                expected = old.step(action)
                actual = new.step(WholePlantAction(**action.to_dict()))
                for field in fields(PlantState):
                    if field.name != "action":
                        self.assertEqual(getattr(expected, field.name), getattr(actual, field.name), field.name)
                for key, value in expected.action.to_dict().items():
                    self.assertEqual(actual.action.to_dict()[key], value)
                self.assertEqual(actual.action.battery_dac_support_kw, 0.0)
                self.assertEqual(actual.action.battery_methanator_support_kw, 0.0)
                self.assertEqual(old.delivery_step, new.delivery_step)

    def test_battery_powers_both_downstream_units_at_night(self):
        simulator = WholePlantSimulator()
        action = WholePlantAction(0, 0, 0, 0, 0, 1200, 180)
        for minute in range(61):
            before = simulator.state
            after = simulator.step(action)
            self.assertEqual(after.solar_available_kw, 0)
            self.assert_balance(before, after, simulator)
            if minute < 30:
                self.assertEqual(after.co2_capture_kg_h, 0)
            if minute < 60:
                self.assertEqual(after.ch4_production_kg_h, 0)
        self.assertAlmostEqual(after.co2_capture_kg_h, 600.0)
        self.assertAlmostEqual(after.ch4_production_kg_h, 150.0)
        self.assertEqual(after.dac_starts, 1)
        self.assertEqual(after.methanator_starts, 1)
        self.assertGreater(after.battery_to_dac_kw, 0)
        self.assertGreater(after.battery_to_methanator_kw, 0)

    def test_restricted_bus_cannot_feed_downstream_at_night(self):
        for dynamic in (False, True):
            simulator = WholePlantSimulator(shared_bus=False, downstream_dynamics=dynamic)
            after = simulator.step(WholePlantAction(0, 0, 0, 0, 0, 1200, 180))
            self.assertEqual(after.dac_power_kw, 0)
            self.assertEqual(after.methanator_power_kw, 0)
            self.assertEqual(after.battery_power_kw, 0)

    def test_shared_inverter_cap_and_floor(self):
        simulator = WholePlantSimulator(downstream_dynamics=False)
        # Solar-free, all requested draws combined must fit one inverter.
        before = simulator.state
        after = simulator.step(WholePlantAction(0, 0, 0, 0, 2500, 2500, 2500))
        self.assert_balance(before, after, simulator)
        self.assertLessEqual(-after.battery_power_kw, 2500.0)
        self.assertGreater(after.battery_to_dac_kw, 0)
        self.assertGreater(after.battery_to_methanator_kw, 0)
        # Exactly 1kWh above the floor permits only 57.6kW total for one minute.
        simulator.battery.energy_kwh = 401.0
        simulator.state = replace(simulator.state, battery_energy_kwh=401.0, battery_soc=401 / 8000)
        before = simulator.state
        after = simulator.step(WholePlantAction(0, 0, 0, 0, 0, 1000, 1000))
        self.assert_balance(before, after, simulator)
        self.assertAlmostEqual(-after.battery_power_kw, 57.6)
        self.assertAlmostEqual(after.battery_energy_kwh, 400)
        before = after
        after = simulator.step(WholePlantAction(0, 0, 0, 0, 0, 1000, 1000))
        self.assert_balance(before, after, simulator)
        self.assertEqual(after.battery_power_kw, 0)

    def test_one_actual_discharge_and_refused_reservation_retraction(self):
        simulator = WholePlantSimulator()
        # Cold downstream units can only accept their heaters, regardless of
        # the larger support request; the battery must lose only actual use.
        before = simulator.state
        with patch.object(simulator.battery, "discharge", wraps=simulator.battery.discharge) as discharge:
            after = simulator.step(WholePlantAction(0, 0, 0, 0, 0, 1200, 180))
            self.assertEqual(discharge.call_count, 1)
            self.assertAlmostEqual(discharge.call_args.args[0], 390.0)
        self.assert_balance(before, after, simulator)
        self.assertAlmostEqual(before.battery_energy_kwh - after.battery_energy_kwh, 390 / 60 / .96)
        # Hot methanation with no feedstock retracts all productive support.
        simulator.methanator.readiness_kwh = 72
        simulator.h2_tank.level_kg = 0
        simulator.co2_tank.level_kg = 0
        with patch.object(simulator.battery, "discharge", wraps=simulator.battery.discharge) as discharge:
            after = simulator.step(WholePlantAction(0, 0, 0, 0, 0, 0, 180))
            self.assertEqual(discharge.call_count, 1)
            self.assertAlmostEqual(discharge.call_args.args[0], 18.0)
        self.assertEqual(after.ch4_production_kg_h, 0)
        self.assertEqual(after.battery_to_methanator_kw, 18)

    def test_outage_lifecycle_and_observation_preserve_thermal_state(self):
        simulator = WholePlantSimulator()
        simulator.inject_fault({"type": "dac_outage", "severity": 1.0,
                                "start_s": 1800, "duration_s": 120})
        action = WholePlantAction(0, 0, 0, 0, 0, 1200, 0)
        for _ in range(30):
            simulator.step(action)
        self.assertEqual(simulator.state.dac_readiness_kwh, 120)
        before = simulator.state
        after = simulator.step(action)
        self.assert_balance(before, after, simulator)
        self.assertEqual(after.dac_power_kw, 0)
        self.assertEqual(after.dac_mode, "fault")
        self.assertEqual(after.dac_readiness_kwh, 119)
        simulator.step(action)
        after = simulator.step(action)
        self.assertEqual(after.dac_mode, "starting")
        self.assertEqual(after.dac_readiness_kwh, 120)
        self.assertEqual(after.co2_capture_kg_h, 0)
        self.assertEqual(simulator.telemetry.dac_readiness_kwh, after.dac_readiness_kwh)

    def test_interruption_and_passive_cooling_cost_energy(self):
        for unit, profile in ((DynamicDAC(), DAC_READINESS), (DynamicMethanator(), METHANATOR_READINESS)):
            def advance(power):
                if isinstance(unit, DynamicDAC):
                    return unit.run(power, 1 / 60)
                return unit.run(power, 1 / 60, h2_available_kg=100, co2_available_kg=1000, ch4_headroom_kg=1000).ch4_kg
            for _ in range(10):
                self.assertEqual(advance(profile.startup_heater_kw), 0)
            before = unit.readiness_kwh
            self.assertEqual(advance(0), 0)
            self.assertAlmostEqual(unit.readiness_kwh, before - profile.standby_kw / 60)
            self.assertEqual(unit.mode, "cooling")
            advance(profile.startup_heater_kw)
            self.assertEqual(unit.starts, 2)
            self.assertEqual(unit.cold_starts, 1)
            for _ in range(300):
                self.assertEqual(advance(0), 0)
            self.assertEqual(unit.readiness_kwh, 0)
            self.assertEqual(unit.mode, "off")
            # Heater power below losses cannot accumulate readiness from cold.
            advance(profile.standby_kw / 2)
            self.assertEqual(unit.readiness_kwh, 0)

    def test_no_feedstocks_or_storage_room_only_accepts_paid_standby(self):
        dac = DynamicDAC()
        meth = DynamicMethanator()
        for _ in range(60):
            dac.run(1200, 1 / 60, co2_headroom_kg=0)
            meth.run(180, 1 / 60, h2_available_kg=0, co2_available_kg=0, ch4_headroom_kg=100)
        self.assertEqual(dac.run(1200, 1 / 60, co2_headroom_kg=0), 0)
        result = meth.run(180, 1 / 60, h2_available_kg=0, co2_available_kg=0, ch4_headroom_kg=100)
        self.assertEqual(result.ch4_kg, 0)
        self.assertEqual(dac.power_kw, 60)
        self.assertEqual(meth.power_kw, 18)
        self.assertEqual(dac.mode, "standby")
        self.assertEqual(meth.mode, "standby")

    def test_fault_blocks_power_but_keeps_passive_cooling(self):
        for unit, profile in ((DynamicDAC(), DAC_READINESS), (DynamicMethanator(), METHANATOR_READINESS)):
            unit.readiness_kwh = profile.capacity_kwh
            unit.apply_effect(ComponentEffect(offline=True))
            if isinstance(unit, DynamicDAC):
                self.assertEqual(unit.run(1200, 1 / 60), 0)
            else:
                self.assertEqual(unit.run(180, 1 / 60, h2_available_kg=100,
                    co2_available_kg=1000, ch4_headroom_kg=1000).ch4_kg, 0)
            self.assertEqual(unit.power_kw, 0)
            self.assertEqual(unit.mode, "fault")
            self.assertAlmostEqual(unit.readiness_kwh, profile.capacity_kwh - profile.standby_kw / 60)

    def test_nominal_hot_full_load_yield_preserved(self):
        dac, meth = DynamicDAC(), DynamicMethanator()
        dac.readiness_kwh, meth.readiness_kwh = 120, 72
        self.assertAlmostEqual(dac.run(1200, 1 / 60), 10)
        result = meth.run(180, 1 / 60, h2_available_kg=100, co2_available_kg=1000, ch4_headroom_kg=1000)
        self.assertAlmostEqual(result.ch4_kg, 2.5)
        dac.apply_effect(ComponentEffect(degradation=.5))
        self.assertAlmostEqual(dac.run(1200, 1 / 60), 5)

    def test_refused_solar_is_never_automatically_charged(self):
        simulator = WholePlantSimulator()
        simulator.time_s = 12 * 3600
        before = simulator.battery.energy_kwh
        after = simulator.step(WholePlantAction(0, 0, 1, 0))
        self.assertEqual(after.dac_power_kw, 300)
        self.assertEqual(after.solar_to_battery_kw, 0)
        self.assertGreater(after.solar_curtailed_kw, 0)
        self.assertEqual(after.battery_energy_kwh, before)

    def test_solar_charge_support_overlap_is_netted_once(self):
        simulator = WholePlantSimulator(downstream_dynamics=False)
        simulator.time_s = 12 * 3600
        before = simulator.state
        after = simulator.step(WholePlantAction(1, 0, 0, 0, 1200, 800, 180))
        self.assert_balance(before, after, simulator)
        self.assertEqual(after.battery_to_electrolyser_kw, 0)
        self.assertEqual(after.battery_to_dac_kw, 0)
        self.assertEqual(after.battery_to_methanator_kw, 0)
        self.assertAlmostEqual(after.solar_to_electrolyser_kw, 1200)
        self.assertAlmostEqual(after.solar_to_dac_kw, 800)
        self.assertAlmostEqual(after.solar_to_methanator_kw, 180)

    def test_random_actions_conserve_energy_mass_and_readiness(self):
        cfg = _frozen_bootstrap.ControllerNeutralTrainingConfigV1.from_dict(
            _frozen_bootstrap.legacy.load_benchmark_contract()["source_template"])
        for shared in (False, True):
            for dynamic in (False, True):
                simulator = WholePlantSimulator(cfg.plant, electrolyser_commitment=cfg.episode.electrolyser_commitment,
                    shared_bus=shared, downstream_dynamics=dynamic)
                simulator.set_delivery_schedule([{"id": "fixture", "due_step": 900, "requested_kg": 150}])
                rng = np.random.default_rng(30000777)
                for _ in range(1500):
                    action, _ = decode_action(rng.uniform(-1, 1, 7))
                    before = simulator.state
                    after = simulator.step(action)
                    self.assert_balance(before, after, simulator)
                    self.assertIsInstance(after, WholePlantState)


if __name__ == "__main__":
    unittest.main()

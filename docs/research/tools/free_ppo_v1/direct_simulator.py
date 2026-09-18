"""Isolated dispatch change: refused process solar is curtailed, not autocharged.

All component calls use the frozen dispatch. The battery proxy changes only the
upper bound on that dispatch's final charge request. Charge/discharge netting is
computed exactly as in the frozen dispatch before component acceptance.
"""
from __future__ import annotations

from plant_sim import Action, PlantSimulator
from plant_sim.allocation import dispatch as frozen_dispatch


class _ExplicitChargeBattery:
    def __init__(self, battery, charge_limit_kw):
        self._battery = battery
        self._charge_limit_kw = charge_limit_kw

    def __getattr__(self, name):
        return getattr(self._battery, name)

    def charge(self, requested_kw, dt_h):
        return self._battery.charge(min(requested_kw, self._charge_limit_kw), dt_h)


def dispatch(action, solar_kw, *, battery, electrolyser, h2_tank, dac, co2_tank,
             methanator, ch4_tank, dt_h):
    # Match frozen dispatch's read-only initial support/netting calculation.
    solar_electrolyser = action.electrolyser_share * solar_kw
    explicit_charge = action.battery_share * solar_kw
    intake_limit = electrolyser.intake_limit_kw(dt_h, h2_tank.headroom_kg)
    offered_support = min(action.battery_support_kw, battery.discharge_headroom_kw(dt_h),
                          max(0.0, intake_limit - solar_electrolyser))
    netted_overlap = min(offered_support, explicit_charge)
    allowed_charge = max(0.0, explicit_charge - netted_overlap)
    return frozen_dispatch(action, solar_kw,
        battery=_ExplicitChargeBattery(battery, allowed_charge), electrolyser=electrolyser,
        h2_tank=h2_tank, dac=dac, co2_tank=co2_tank, methanator=methanator,
        ch4_tank=ch4_tank, dt_h=dt_h)


class DirectPlantSimulator(PlantSimulator):
    """Frozen step order, using only the local explicit-charge dispatch above."""

    def step(self, action: Action | dict | None = None):
        safe_action = Action.sanitised(action)
        dt_h = self.config.dt_h
        if self._solar_factor_tape is not None:
            self._solar_factor_tape.for_interval(self.step_index)
        self.events.at(self.time_s, self.state.clock)
        self._apply_faults()
        solar_kw = self._update_solar(step_index=self.step_index + 1)
        flows = dispatch(safe_action, solar_kw,
            battery=self.battery, electrolyser=self.electrolyser, h2_tank=self.h2_tank,
            dac=self.dac, co2_tank=self.co2_tank, methanator=self.methanator,
            ch4_tank=self.ch4_tank, dt_h=dt_h)
        self.time_s += self.config.dt_s
        self.step_index += 1
        self.energy_produced_kwh += flows.solar_available_kw * dt_h
        self.energy_curtailed_kwh += flows.solar_curtailed_kw * dt_h
        self.h2_produced_total_kg += flows.h2_produced_kg
        self.co2_captured_total_kg += flows.co2_captured_kg
        self.ch4_produced_total_kg += flows.ch4_produced_kg
        delivery = self.delivery_ledger.settle(self.step_index, available_kg=self.ch4_tank.level_kg)
        if delivery.delivered_kg > 0:
            self.ch4_tank.withdraw(delivery.delivered_kg)
        self.state = self._build_state(flows, safe_action)
        self.events.at(self.time_s, self.state.clock)
        self._observe_conditions(self.state)
        self.telemetry = self._observe(self.state)
        return self.state

"""The digital twin: `reset()` / `step(action)` over discrete timesteps.

    sim = PlantSimulator()
    state = sim.reset()
    state = sim.step(Action(battery_share=0.2, electrolyser_share=0.5, dac_share=0.2))

The simulator owns time and bookkeeping; the components own physics; `dispatch`
owns routing. It knows nothing about HTTP, agents, or 3D scenes -- it is only
ever driven through `Action` and only ever answers with `PlantState`.
"""

from __future__ import annotations

from typing import Any

from .allocation import PowerFlows, dispatch
from .components import (
    Battery,
    Component,
    DirectAirCapture,
    Electrolyser,
    GasStorage,
    Methanator,
    SolarArray,
)
from .config import PlantConfig
from .state import Action, PlantState


class PlantSimulator:
    """The synthetic-methane plant, advanced one timestep at a time.

    Solar -> {battery, electrolyser, DAC, methanator} on the power bus;
    electrolyser -> H2 tank, DAC -> CO2 tank, both tanks -> methanator -> CH4
    tank on the material side.
    """

    def __init__(self, config: PlantConfig | None = None) -> None:
        self.config = config or PlantConfig()
        self.solar = SolarArray(self.config.solar)
        self.battery = Battery(self.config.battery)
        self.electrolyser = Electrolyser(self.config.electrolyser)
        self.h2_tank = GasStorage(self.config.h2_tank, name="h2_tank", species="H2")
        self.dac = DirectAirCapture(self.config.dac)
        self.co2_tank = GasStorage(self.config.co2_tank, name="co2_tank", species="CO2")
        self.methanator = Methanator(self.config.methanator)
        self.ch4_tank = GasStorage(self.config.ch4_tank, name="ch4_tank", species="CH4")

        #: Registry every component registers into: reset + telemetry for free.
        self.components: dict[str, Component] = {
            c.name: c
            for c in (
                self.solar,
                self.battery,
                self.electrolyser,
                self.h2_tank,
                self.dac,
                self.co2_tank,
                self.methanator,
                self.ch4_tank,
            )
        }

        self.step_index = 0
        self.time_s = 0.0
        self.energy_produced_kwh = 0.0
        self.energy_curtailed_kwh = 0.0
        self.h2_produced_total_kg = 0.0
        self.co2_captured_total_kg = 0.0
        self.ch4_produced_total_kg = 0.0
        self.state = self.reset()

    # --- interface --------------------------------------------------------
    def reset(self) -> PlantState:
        """Return the plant to its initial condition and report the state."""
        for component in self.components.values():
            component.reset()
        self.step_index = 0
        self.time_s = 0.0
        self.energy_produced_kwh = 0.0
        self.energy_curtailed_kwh = 0.0
        self.h2_produced_total_kg = 0.0
        self.co2_captured_total_kg = 0.0
        self.ch4_produced_total_kg = 0.0
        self.solar.update(self.time_s)
        self.state = self._build_state(
            PowerFlows(
                solar_available_kw=self.solar.power_kw,
                solar_to_battery_kw=0.0,
                solar_to_electrolyser_kw=0.0,
                solar_to_dac_kw=0.0,
                solar_to_methanator_kw=0.0,
                battery_to_electrolyser_kw=0.0,
                electrolyser_power_kw=0.0,
                dac_power_kw=0.0,
                methanator_power_kw=0.0,
                solar_curtailed_kw=self.solar.power_kw,
                h2_produced_kg=0.0,
                co2_captured_kg=0.0,
                ch4_produced_kg=0.0,
            ),
            Action.idle(),
        )
        return self.state

    def step(self, action: Action | dict[str, Any] | None = None) -> PlantState:
        """Advance one timestep under `action` and report the new state."""
        safe_action = Action.sanitised(action)
        dt_h = self.config.dt_h

        solar_kw = self.solar.update(self.time_s)
        flows = dispatch(
            safe_action,
            solar_kw,
            battery=self.battery,
            electrolyser=self.electrolyser,
            h2_tank=self.h2_tank,
            dac=self.dac,
            co2_tank=self.co2_tank,
            methanator=self.methanator,
            ch4_tank=self.ch4_tank,
            dt_h=dt_h,
        )

        self.time_s += self.config.dt_s
        self.step_index += 1
        self.energy_produced_kwh += flows.solar_available_kw * dt_h
        self.energy_curtailed_kwh += flows.solar_curtailed_kw * dt_h
        self.h2_produced_total_kg += flows.h2_produced_kg
        self.co2_captured_total_kg += flows.co2_captured_kg
        self.ch4_produced_total_kg += flows.ch4_produced_kg

        self.state = self._build_state(flows, safe_action)
        return self.state

    def run(self, steps: int, action: Action | dict[str, Any] | None = None) -> PlantState:
        """Convenience: repeat the same action `steps` times (fast-forward)."""
        for _ in range(max(0, steps)):
            self.step(action)
        return self.state

    # --- internals --------------------------------------------------------
    def _build_state(self, flows: PowerFlows, action: Action) -> PlantState:
        dt_h = self.config.dt_h
        return PlantState(
            step=self.step_index,
            time_s=self.time_s,
            solar_available_kw=flows.solar_available_kw,
            solar_to_battery_kw=flows.solar_to_battery_kw,
            solar_to_electrolyser_kw=flows.solar_to_electrolyser_kw,
            solar_to_dac_kw=flows.solar_to_dac_kw,
            solar_to_methanator_kw=flows.solar_to_methanator_kw,
            solar_curtailed_kw=flows.solar_curtailed_kw,
            battery_soc=self.battery.soc,
            battery_energy_kwh=self.battery.energy_kwh,
            battery_power_kw=self.battery.power_kw,
            battery_to_electrolyser_kw=flows.battery_to_electrolyser_kw,
            electrolyser_power_kw=self.electrolyser.power_kw,
            electrolyser_load_frac=self.electrolyser.load_frac,
            electrolyser_online=self.electrolyser.online,
            h2_production_kg_h=flows.h2_produced_kg / dt_h,
            h2_level_kg=self.h2_tank.level_kg,
            h2_capacity_kg=self.h2_tank.capacity_kg,
            h2_fill=self.h2_tank.fill,
            dac_power_kw=self.dac.power_kw,
            dac_load_frac=self.dac.load_frac,
            dac_online=self.dac.online,
            co2_capture_kg_h=flows.co2_captured_kg / dt_h,
            co2_level_kg=self.co2_tank.level_kg,
            co2_capacity_kg=self.co2_tank.capacity_kg,
            co2_fill=self.co2_tank.fill,
            methanator_power_kw=self.methanator.power_kw,
            methanator_load_frac=self.methanator.load_frac,
            methanator_online=self.methanator.online,
            ch4_production_kg_h=flows.ch4_produced_kg / dt_h,
            ch4_level_kg=self.ch4_tank.level_kg,
            ch4_capacity_kg=self.ch4_tank.capacity_kg,
            ch4_fill=self.ch4_tank.fill,
            energy_produced_kwh=self.energy_produced_kwh,
            energy_curtailed_kwh=self.energy_curtailed_kwh,
            h2_produced_total_kg=self.h2_produced_total_kg,
            co2_captured_total_kg=self.co2_captured_total_kg,
            ch4_produced_total_kg=self.ch4_produced_total_kg,
            action=action,
        )

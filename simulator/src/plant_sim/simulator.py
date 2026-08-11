"""The digital twin: `reset()` / `step(action)` over discrete timesteps.

    sim = PlantSimulator()
    state = sim.reset()
    state = sim.step(Action(battery_share=0.3, electrolyser_share=0.7))

The simulator owns time and bookkeeping; the components own physics; `dispatch`
owns routing. It knows nothing about HTTP, agents, or 3D scenes -- it is only
ever driven through `Action` and only ever answers with `PlantState`.
"""

from __future__ import annotations

from typing import Any

from .allocation import PowerFlows, dispatch
from .components import Battery, Component, Electrolyser, GasStorage, SolarArray
from .config import PlantConfig
from .state import Action, PlantState


class PlantSimulator:
    """Solar -> {battery, electrolyser} -> H2 tank, advanced one timestep at a time."""

    def __init__(self, config: PlantConfig | None = None) -> None:
        self.config = config or PlantConfig()
        self.solar = SolarArray(self.config.solar)
        self.battery = Battery(self.config.battery)
        self.electrolyser = Electrolyser(self.config.electrolyser)
        self.h2_tank = GasStorage(self.config.h2_tank, name="h2_tank", species="H2")

        #: Registry every component registers into. Add DAC / methanation / CH4
        #: storage here and they become part of reset + telemetry for free.
        self.components: dict[str, Component] = {
            c.name: c for c in (self.solar, self.battery, self.electrolyser, self.h2_tank)
        }

        self.step_index = 0
        self.time_s = 0.0
        self.energy_produced_kwh = 0.0
        self.energy_curtailed_kwh = 0.0
        self.h2_produced_total_kg = 0.0
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
        self.solar.update(self.time_s)
        self.state = self._build_state(
            PowerFlows(
                solar_available_kw=self.solar.power_kw,
                solar_to_battery_kw=0.0,
                solar_to_electrolyser_kw=0.0,
                battery_to_electrolyser_kw=0.0,
                electrolyser_power_kw=0.0,
                solar_curtailed_kw=self.solar.power_kw,
                h2_produced_kg=0.0,
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
            dt_h=dt_h,
        )

        self.time_s += self.config.dt_s
        self.step_index += 1
        self.energy_produced_kwh += flows.solar_available_kw * dt_h
        self.energy_curtailed_kwh += flows.solar_curtailed_kw * dt_h
        self.h2_produced_total_kg += flows.h2_produced_kg

        self.state = self._build_state(flows, safe_action)
        return self.state

    def run(self, steps: int, action: Action | dict[str, Any] | None = None) -> PlantState:
        """Convenience: repeat the same action `steps` times (fast-forward)."""
        for _ in range(max(0, steps)):
            self.step(action)
        return self.state

    # --- internals --------------------------------------------------------
    def _build_state(self, flows: PowerFlows, action: Action) -> PlantState:
        return PlantState(
            step=self.step_index,
            time_s=self.time_s,
            solar_available_kw=flows.solar_available_kw,
            solar_to_battery_kw=flows.solar_to_battery_kw,
            solar_to_electrolyser_kw=flows.solar_to_electrolyser_kw,
            solar_curtailed_kw=flows.solar_curtailed_kw,
            battery_soc=self.battery.soc,
            battery_energy_kwh=self.battery.energy_kwh,
            battery_power_kw=self.battery.power_kw,
            battery_to_electrolyser_kw=flows.battery_to_electrolyser_kw,
            electrolyser_power_kw=self.electrolyser.power_kw,
            electrolyser_load_frac=self.electrolyser.load_frac,
            electrolyser_online=self.electrolyser.online,
            h2_production_kg_h=flows.h2_produced_kg / self.config.dt_h,
            h2_level_kg=self.h2_tank.level_kg,
            h2_capacity_kg=self.h2_tank.capacity_kg,
            h2_fill=self.h2_tank.fill,
            energy_produced_kwh=self.energy_produced_kwh,
            energy_curtailed_kwh=self.energy_curtailed_kwh,
            h2_produced_total_kg=self.h2_produced_total_kg,
            action=action,
        )

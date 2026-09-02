"""The digital twin: `reset()` / `step(action)` over discrete timesteps.

    sim = PlantSimulator()
    state = sim.reset()
    state = sim.step(Action(battery_share=0.2, electrolyser_share=0.5, dac_share=0.2))

The simulator owns time and bookkeeping; the components own physics; `dispatch`
owns routing. It knows nothing about HTTP, agents, or 3D scenes -- it is only
ever driven through `Action` and only ever answers with `PlantState`.

Since V4 it also owns the abnormal side of the plant: a `FaultRegistry` of
scheduled faults and disturbances, an `EventLog` of what it noticed, and the
distinction between the truth (`state`) and what an observer is told
(`telemetry`).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping

from . import scenarios as scenario_library
from . import telemetry as telemetry_module
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
from .delivery import DeliveryCommitment, DeliveryLedger, DeliveryStepResult
from .events import EventLog
from .faults import Fault, FaultRegistry, FaultSpec
from .forcing import SolarFactorTape
from .scenarios import Scenario
from .state import (
    CRITICAL,
    DEGRADED,
    NORMAL,
    OFFLINE,
    SECONDS_PER_HOUR,
    Action,
    ComponentHealth,
    PlantState,
)

#: SOC margin above `min_soc` that counts as "running on the reserve".
LOW_CHARGE_MARGIN = 0.05
#: Tank level below which a feedstock counts as effectively empty (kg).
EMPTY_KG = 0.05


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

        #: Faults and disturbances currently scheduled or running.
        self.faults = FaultRegistry()
        #: What the plant noticed, oldest first.
        self.events = EventLog()
        #: Opt-in methane delivery schedule and accounting, kept outside V1 state.
        self.delivery_ledger = DeliveryLedger()
        #: Id of the scenario in force, if any.
        self.scenario: str | None = None
        #: Last reported plant status, so status changes emit exactly one event.
        self._status = NORMAL
        #: Optional episode-relative exogenous solar resource trajectory.
        self._solar_factor_tape: SolarFactorTape | None = None

        self.step_index = 0
        self.time_s = 0.0
        self.energy_produced_kwh = 0.0
        self.energy_curtailed_kwh = 0.0
        self.h2_produced_total_kg = 0.0
        self.co2_captured_total_kg = 0.0
        self.ch4_produced_total_kg = 0.0
        self.state = self.reset()

    # --- interface --------------------------------------------------------
    def reset(
        self,
        *,
        keep_faults: bool = False,
        solar_factor_tape: SolarFactorTape | None = None,
    ) -> PlantState:
        """Return the plant to its initial condition and report the state.

        Faults, the event stream and any loaded scenario go with it: a reset
        plant is a healthy plant unless a caller explicitly keeps the faults.
        A solar factor tape is opt-in and scoped to this episode; passing
        ``None`` retains the exact clear-sky V1 behaviour.
        """
        if solar_factor_tape is not None:
            if not isinstance(solar_factor_tape, SolarFactorTape):
                raise TypeError("solar_factor_tape must be a SolarFactorTape or None")
            if solar_factor_tape.dt_s != self.config.dt_s:
                raise ValueError(
                    "solar factor tape timestep mismatch: "
                    f"tape dt_s={solar_factor_tape.dt_s}, simulator dt_s={self.config.dt_s}"
                )
        for component in self.components.values():
            component.reset()
        if not keep_faults:
            self.faults.clear_all()
            self.scenario = None
        self.events.clear()
        self.delivery_ledger.clear()
        self._status = NORMAL
        self._solar_factor_tape = solar_factor_tape
        self.step_index = 0
        self.time_s = 0.0
        self.energy_produced_kwh = 0.0
        self.energy_curtailed_kwh = 0.0
        self.h2_produced_total_kg = 0.0
        self.co2_captured_total_kg = 0.0
        self.ch4_produced_total_kg = 0.0
        self._apply_faults()
        self._update_solar(step_index=0)
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
        self.telemetry = self._observe(self.state)
        return self.state

    def step(self, action: Action | dict[str, Any] | None = None) -> PlantState:
        """Advance one timestep under `action` and report the new state."""
        safe_action = Action.sanitised(action)
        dt_h = self.config.dt_h

        # Exhaustion is checked before advancing faults, events, or equipment,
        # so a failed call cannot leave a partially advanced episode behind.
        if self._solar_factor_tape is not None:
            self._solar_factor_tape.for_interval(self.step_index)

        self.events.at(self.time_s, self.state.clock)
        self._apply_faults()
        solar_kw = self._update_solar(step_index=self.step_index + 1)
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

        # A tick-n deadline sits after production over [n - 1, n), so methane
        # made in this dispatch is available to the delivery.  Tick indices,
        # rather than float timestamps, make the boundary unambiguous.
        delivery = self.delivery_ledger.settle(
            self.step_index,
            available_kg=self.ch4_tank.level_kg,
        )
        if delivery.delivered_kg > 0.0:
            self.ch4_tank.withdraw(delivery.delivered_kg)

        self.state = self._build_state(flows, safe_action)
        self.events.at(self.time_s, self.state.clock)
        self._observe_conditions(self.state)
        self.telemetry = self._observe(self.state)
        return self.state

    def run(self, steps: int, action: Action | dict[str, Any] | None = None) -> PlantState:
        """Convenience: repeat the same action `steps` times (fast-forward)."""
        for _ in range(max(0, steps)):
            self.step(action)
        return self.state

    def get_telemetry(self) -> PlantState:
        """What an observer sees: the state as reported by the instruments.

        Identical to `state` unless a sensor fault is active. Controllers,
        agents and the control room should read *this*, not `state`.
        """
        return self.telemetry

    # --- deliveries ------------------------------------------------------
    def set_delivery_schedule(
        self,
        commitments: Iterable[DeliveryCommitment | Mapping[str, Any]],
        *,
        freeze: bool = False,
    ) -> DeliveryStepResult:
        """Install an episode-relative methane delivery schedule atomically.

        ``due_step`` values are counted from the most recent ``reset()`` or
        ``load_scenario()``.  When installing after the episode has advanced,
        every deadline must still be in the future.  A reset or scenario load
        clears both the schedule and all delivery accounting.
        """
        result = self.delivery_ledger.set_schedule(
            commitments,
            current_step=self.step_index,
        )
        if freeze:
            self.delivery_ledger.freeze()
        return result

    @property
    def delivery_step(self) -> DeliveryStepResult:
        """Immutable delivery result for the current simulator boundary."""
        return self.delivery_ledger.current

    @property
    def next_delivery(self) -> DeliveryCommitment | None:
        """Next pending commitment, without exposing the whole future tape."""
        return self.delivery_ledger.next_pending

    def delivery_audit(self) -> dict[str, Any]:
        """Serialize the full schedule, settlements, and cumulative totals."""
        return self.delivery_ledger.audit()

    # --- faults and scenarios ---------------------------------------------
    def inject_fault(self, spec: FaultSpec | dict[str, Any]) -> Fault:
        """Schedule a fault or disturbance; without a `start_s` it starts now."""
        fault = self.faults.inject(spec, now_s=self.time_s)
        self.events.at(self.time_s, self.state.clock)
        if fault.start_s > self.time_s:
            self.events.emit(
                "info",
                "fault.scheduled",
                f"{fault.spec.kind.label} scheduled at {_clock(fault.start_s)}",
            )
        self._apply_faults()
        return fault

    def clear_fault(self, fault_id: str) -> bool:
        """Clear one fault by id; returns whether it was found."""
        cleared = self.faults.clear(fault_id)
        if cleared:
            self.events.at(self.time_s, self.state.clock)
            self.events.emit("info", "fault.cleared", f"Fault {fault_id} cleared by operator")
            self._apply_faults()
        return cleared

    def clear_faults(self) -> None:
        """Clear every fault and disturbance, leaving the plant running."""
        self.faults.clear_all()
        self.scenario = None
        self.events.at(self.time_s, self.state.clock)
        self.events.emit("info", "fault.cleared_all", "All faults cleared by operator")
        self._apply_faults()

    def load_scenario(
        self,
        scenario: Scenario | str,
        *,
        solar_factor_tape: SolarFactorTape | None = None,
    ) -> PlantState:
        """Reset the plant into a repeatable situation and schedule its faults."""
        chosen = scenario_library.get(scenario)
        self.reset(solar_factor_tape=solar_factor_tape)
        self.scenario = chosen.id
        self.time_s = chosen.start_hour * SECONDS_PER_HOUR

        initial = chosen.initial
        if initial.battery_soc is not None:
            self.battery.energy_kwh = initial.battery_soc * self.battery.capacity_kwh
        for level_kg, tank in (
            (initial.h2_kg, self.h2_tank),
            (initial.co2_kg, self.co2_tank),
            (initial.ch4_kg, self.ch4_tank),
        ):
            if level_kg is not None:
                tank.level_kg = min(max(0.0, level_kg), tank.capacity_kg)

        # scenario offsets are relative to its start hour; the registry wants
        # absolute simulator time, so resolve them here
        for spec in chosen.faults:
            offset_s = 0.0 if spec.start_s is None else spec.start_s
            self.faults.inject(replace(spec, start_s=None), now_s=self.time_s + offset_s)

        self.events.at(self.time_s, _clock(self.time_s))
        self._apply_faults()
        self._update_solar(step_index=0)
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
        self.events.at(self.time_s, self.state.clock)
        self.events.emit("info", "scenario.loaded", f"Scenario loaded: {chosen.name}")
        self.telemetry = self._observe(self.state)
        return self.state

    # --- internals --------------------------------------------------------
    def _update_solar(self, *, step_index: int) -> float:
        """Apply one episode-boundary resource factor at the legacy solar time.

        Tape lookup intentionally uses the episode-relative integer step, not
        the absolute scenario clock.  The clear-sky curve keeps its established
        V1 sampling convention so an all-ones tape is exactly equivalent to no
        tape at all.
        """
        resource_factor = (
            1.0
            if self._solar_factor_tape is None
            else self._solar_factor_tape.at_boundary(step_index)
        )
        return self.solar.update(self.time_s, resource_factor=resource_factor)

    def _apply_faults(self) -> None:
        """Advance fault lifecycles and hand each component its current effect."""
        for transition, fault in self.faults.update(self.time_s):
            label = fault.spec.kind.label
            if transition == "activated":
                level = "critical" if fault.spec.kind.mode == "outage" else "warning"
                self.events.emit(
                    level,
                    f"fault.{fault.spec.type}",
                    f"{label} started (severity {fault.spec.severity:.0%})",
                )
            else:
                self.events.emit("info", f"fault.{fault.spec.type}.ended", f"{label} ended")
        for name, component in self.components.items():
            component.apply_effect(self.faults.effect_for(name))

    def _observe(self, state: PlantState) -> PlantState:
        """Apply sensor faults: the only place truth and telemetry diverge."""
        return telemetry_module.observe(state, self.faults.sensor_biases())

    def _plant_status(self, action: Action) -> str:
        """Plant condition, derived from components and inventories."""
        units = (self.electrolyser, self.dac, self.methanator, self.battery, self.solar)
        if all(
            not unit.online for unit in (self.electrolyser, self.dac, self.methanator)
        ) and self.solar.power_kw <= 0.0 and self.battery.reserve_kwh <= 0.0:
            return OFFLINE
        if any(unit.offline for unit in units) or self.battery.soc <= self.config.battery.min_soc:
            return CRITICAL
        if any(unit.effect.degradation > 0.0 for unit in units):
            return DEGRADED
        if self._feedstock_starved(action):
            return DEGRADED
        return NORMAL

    def _feedstock_starved(self, action: Action) -> bool:
        """The reactor was offered power but a feedstock tank was empty."""
        return (
            not self.methanator.online
            and action.methanator_share > 0.0
            and (self.h2_tank.level_kg < EMPTY_KG or self.co2_tank.level_kg < EMPTY_KG)
        )

    def _observe_conditions(self, state: PlantState) -> None:
        """Turn continuous plant conditions into edge-triggered events."""
        battery = self.config.battery
        self.events.transition(
            "battery.low_charge",
            active=state.battery_soc <= battery.min_soc + LOW_CHARGE_MARGIN,
            level="warning",
            message=f"Battery entered low-charge state ({state.battery_soc:.0%})",
            recovery="Battery recovered above the low-charge threshold",
        )
        self.events.transition(
            "methanator.co2_starved",
            active=state.action.methanator_share > 0.0
            and not state.methanator_online
            and self.co2_tank.level_kg < EMPTY_KG,
            level="warning",
            message="Methanator stopped: insufficient CO2",
            recovery="Methanator has CO2 again",
        )
        self.events.transition(
            "methanator.h2_starved",
            active=state.action.methanator_share > 0.0
            and not state.methanator_online
            and self.h2_tank.level_kg < EMPTY_KG,
            level="warning",
            message="Methanator stopped: insufficient H2",
            recovery="Methanator has H2 again",
        )
        self.events.transition(
            "h2_tank.full",
            active=state.h2_fill >= 0.995,
            level="info",
            message="H2 storage full: the stack is being throttled",
        )
        self.events.transition(
            "co2_tank.full",
            active=state.co2_fill >= 0.995,
            level="info",
            message="CO2 storage full: capture is being throttled",
        )
        if state.plant_status != self._status:
            self._status = state.plant_status
            self.events.emit(
                "critical" if state.plant_status == CRITICAL else "info",
                "plant.status",
                f"Plant status: {state.plant_status}",
            )

    def _health(self) -> dict[str, ComponentHealth]:
        return {
            name: ComponentHealth(
                status=component.status,
                degradation=component.effect.degradation,
                offline=component.effect.offline,
            )
            for name, component in self.components.items()
        }

    def _build_state(self, flows: PowerFlows, action: Action) -> PlantState:
        dt_h = self.config.dt_h
        return PlantState(
            step=self.step_index,
            time_s=self.time_s,
            solar_available_kw=flows.solar_available_kw,
            solar_potential_kw=self.solar.potential_kw,
            solar_output_factor=self.solar.output_factor,
            solar_to_battery_kw=flows.solar_to_battery_kw,
            solar_to_electrolyser_kw=flows.solar_to_electrolyser_kw,
            solar_to_dac_kw=flows.solar_to_dac_kw,
            solar_to_methanator_kw=flows.solar_to_methanator_kw,
            solar_curtailed_kw=flows.solar_curtailed_kw,
            battery_soc=self.battery.soc,
            battery_energy_kwh=self.battery.energy_kwh,
            battery_power_kw=self.battery.power_kw,
            battery_to_electrolyser_kw=flows.battery_to_electrolyser_kw,
            battery_capacity_kwh=self.battery.capacity_kwh,
            electrolyser_power_kw=self.electrolyser.power_kw,
            electrolyser_load_frac=self.electrolyser.load_frac,
            electrolyser_online=self.electrolyser.online,
            h2_production_kg_h=flows.h2_produced_kg / dt_h,
            electrolyser_efficiency=self.electrolyser.efficiency,
            electrolyser_temperature_c=self.electrolyser.temperature_c,
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
            plant_status=self._plant_status(action),
            health=self._health(),
            action=action,
        )


def _clock(time_s: float) -> str:
    """`D1 13:45` for an absolute simulation time."""
    minutes = int(round(time_s / 60.0))
    day, minute_of_day = divmod(minutes, 24 * 60)
    return f"D{day + 1} {minute_of_day // 60:02d}:{minute_of_day % 60:02d}"

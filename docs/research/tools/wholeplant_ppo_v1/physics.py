"""Isolated whole-plant bus and paid downstream readiness experiment.

The frozen simulator supplies solar, battery, electrolyser, inventories, faults,
deliveries and observation order. This module adds only the declared shared bus
and two explicit electrical-equivalent readiness reservoirs. They are engineering
assumptions, not a validated thermal process or a reaction-heat model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Any

# This import pins plant_sim before any component import; never use live sources.
from docs.research.tools.free_ppo_v1 import environment as _frozen_bootstrap
from docs.research.tools.free_ppo_v1.direct_simulator import DirectPlantSimulator
from plant_sim import Action
from plant_sim.allocation import PowerFlows
from plant_sim.commitment import COMMITMENT_INTENT_DEADBAND_FRAC
from plant_sim.components import DirectAirCapture, Methanator
from plant_sim.components.methanator import MethanatorResult
from plant_sim.state import PlantState
import numpy as np


@dataclass(frozen=True)
class ReadinessProfile:
    capacity_kwh: float
    startup_heater_kw: float
    standby_kw: float

    @property
    def cold_start_h(self) -> float:
        return self.capacity_kwh / (self.startup_heater_kw - self.standby_kw)

    @property
    def passive_cooldown_h(self) -> float:
        return self.capacity_kwh / self.standby_kw


DAC_READINESS = ReadinessProfile(120.0, 300.0, 60.0)
METHANATOR_READINESS = ReadinessProfile(72.0, 90.0, 18.0)
READINESS_MODES = ("off", "cooling", "starting", "standby", "running", "fault", "instant")
ACTION_ORDER = ("battery_solar_share", "electrolyser_solar_share", "dac_solar_share",
                "methanator_solar_share", "battery_support", "battery_dac_support",
                "battery_methanator_support")


@dataclass(frozen=True)
class WholePlantAction(Action):
    battery_dac_support_kw: float = 0.0
    battery_methanator_support_kw: float = 0.0

    @classmethod
    def sanitised(cls, action: Any, *, max_discharge_kw: float = 2500.0):
        if action is None:
            return cls(0.0, 0.0, 0.0, 0.0)
        raw = action if isinstance(action, dict) else asdict(action)
        base = Action.sanitised(raw)
        support = []
        for key in ("battery_support_kw", "battery_dac_support_kw", "battery_methanator_support_kw"):
            try:
                value = float(raw.get(key, 0.0))
            except (ValueError, TypeError):
                value = 0.0
            support.append(max(0.0, value) if math.isfinite(value) else 0.0)
        scale = max(1.0, sum(support) / max_discharge_kw)
        return cls(base.battery_share, base.electrolyser_share, base.dac_share,
                   base.methanator_share, *(value / scale for value in support))


def decode_action(action: Any, *, max_discharge_kw: float = 2500.0):
    raw = np.asarray(action)
    if raw.shape != (7,) or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError("Expected exactly seven finite real action coordinates")
    raw = raw.astype(np.float64)
    bounded = np.clip(raw, -1.0, 1.0)
    values = (bounded + 1.0) / 2.0
    solar_divisor = max(1.0, float(values[:4].sum()))
    support_divisor = max(1.0, float(values[4:].sum()))
    request = WholePlantAction(*map(float, values[:4] / solar_divisor),
                              *map(float, values[4:] / support_divisor * max_discharge_kw))
    return request, {"raw_action": raw.tolist(), "bounded_action": bounded.tolist(),
                     "box_clipped": bool(np.any(raw != bounded)),
                     "solar_normalization_divisor": solar_divisor,
                     "battery_normalization_divisor": support_divisor,
                     "requested_action": request.to_dict()}


def action_to_vector(action: WholePlantAction, *, max_discharge_kw: float = 2500.0):
    safe = WholePlantAction.sanitised(action, max_discharge_kw=max_discharge_kw)
    if any(abs(float(safe.to_dict()[key]) - float(value)) > 1e-9
           for key, value in action.to_dict().items()):
        raise ValueError("Inverse requires an admissible solar and battery allocation")
    values = [safe.battery_share, safe.electrolyser_share, safe.dac_share,
              safe.methanator_share, safe.battery_support_kw / max_discharge_kw,
              safe.battery_dac_support_kw / max_discharge_kw,
              safe.battery_methanator_support_kw / max_discharge_kw]
    return np.asarray(values, dtype=np.float64) * 2.0 - 1.0


@dataclass(frozen=True)
class WholePlantState(PlantState):
    battery_to_dac_kw: float = 0.0
    battery_to_methanator_kw: float = 0.0
    dac_mode: str = "instant"
    dac_readiness_kwh: float = 0.0
    dac_readiness_fraction: float = 0.0
    dac_startup_power_kw: float = 0.0
    dac_standby_power_kw: float = 0.0
    dac_productive_power_kw: float = 0.0
    dac_heat_loss_kw: float = 0.0
    dac_starts: int = 0
    dac_cold_starts: int = 0
    methanator_mode: str = "instant"
    methanator_readiness_kwh: float = 0.0
    methanator_readiness_fraction: float = 0.0
    methanator_startup_power_kw: float = 0.0
    methanator_standby_power_kw: float = 0.0
    methanator_productive_power_kw: float = 0.0
    methanator_heat_loss_kw: float = 0.0
    methanator_starts: int = 0
    methanator_cold_starts: int = 0


class _Readiness:
    """Paid readiness, integrated at the frozen one-minute physics timestep.

    Q_next = Q + (startup + standby - loss) * dt. Loss is the standby
    rating, capped by energy present during the interval; no negative Q.
    A partial-ready unit accepts at most startup_heater_kw, produces no gas,
    and cannot produce until a subsequent interval begins fully ready. A hot
    unit first pays standby power, then converts additional power to gas.
    """

    readiness_profile: ReadinessProfile

    def _reset_readiness(self):
        self.readiness_kwh = 0.0
        self.mode = "off"
        self.startup_power_kw = self.standby_power_kw = self.productive_power_kw = 0.0
        self.heat_loss_kw = 0.0
        self.starts = self.cold_starts = 0

    @property
    def readiness_fraction(self):
        return self.readiness_kwh / self.readiness_profile.capacity_kwh

    def electrical_intake_limit_kw(self, dt_h):
        if self.offline:
            return 0.0
        profile = self.readiness_profile
        if self.readiness_kwh < profile.capacity_kwh - 1e-9:
            return min(self.config.rated_kw, profile.startup_heater_kw,
                       (profile.capacity_kwh - self.readiness_kwh) / dt_h + profile.standby_kw)
        return self.config.rated_kw

    def _advance_readiness(self, offered_kw, dt_h, productive_limit_kw):
        if dt_h <= 0.0:
            raise ValueError("Readiness integration requires a positive interval")
        profile = self.readiness_profile
        before = self.readiness_kwh
        previous_mode = self.mode
        offered = 0.0 if self.offline else max(0.0, min(offered_kw, self.config.rated_kw))
        self.startup_power_kw = self.standby_power_kw = self.productive_power_kw = 0.0
        if before < profile.capacity_kwh - 1e-9:
            self.startup_power_kw = min(offered, profile.startup_heater_kw,
                                       (profile.capacity_kwh - before) / dt_h + profile.standby_kw)
            if self.startup_power_kw > 0.0:
                self.mode = "starting"
                if previous_mode != "starting":
                    self.starts += 1
                    self.cold_starts += int(before <= 1e-9)
            else:
                self.mode = "cooling" if before > 0.0 else "off"
        else:
            self.standby_power_kw = min(offered, profile.standby_kw)
            if offered >= profile.standby_kw:
                self.productive_power_kw = min(max(0.0, productive_limit_kw),
                                               offered - profile.standby_kw)
            self.mode = "running" if self.productive_power_kw > 0.0 else (
                "standby" if self.standby_power_kw >= profile.standby_kw else "cooling")
        heating_kw = self.startup_power_kw + self.standby_power_kw
        self.heat_loss_kw = min(profile.standby_kw, before / dt_h + heating_kw)
        self.readiness_kwh = min(profile.capacity_kwh,
                                 max(0.0, before + (heating_kw - self.heat_loss_kw) * dt_h))
        if self.offline:
            self.mode = "fault"
        elif self.readiness_kwh <= 1e-9 and heating_kw == 0.0:
            self.readiness_kwh = 0.0
            self.mode = "off"
        self.power_kw = heating_kw + self.productive_power_kw
        self.online = self.productive_power_kw > 0.0
        return self.productive_power_kw

    def readiness_snapshot(self):
        return {"mode": self.mode, "readiness_kwh": self.readiness_kwh,
                "readiness_fraction": self.readiness_fraction,
                "startup_power_kw": self.startup_power_kw, "standby_power_kw": self.standby_power_kw,
                "productive_power_kw": self.productive_power_kw, "heat_loss_kw": self.heat_loss_kw,
                "starts": self.starts, "cold_starts": self.cold_starts}


class DynamicDAC(_Readiness, DirectAirCapture):
    readiness_profile = DAC_READINESS

    def __init__(self, config=None):
        super().__init__(config)
        self._reset_readiness()

    def reset(self):
        super().reset()
        self._reset_readiness()

    @property
    def productive_specific_energy_kwh_per_kg(self):
        return (self.specific_energy_kwh_per_kg *
                (1.0 - self.readiness_profile.standby_kw / self.config.rated_kw))

    def intake_limit_kw(self, dt_h, co2_headroom_kg):
        electrical_limit = self.electrical_intake_limit_kw(dt_h)
        if self.readiness_kwh < self.readiness_profile.capacity_kwh - 1e-9:
            return electrical_limit
        production_kw = max(0.0, co2_headroom_kg) * self.productive_specific_energy_kwh_per_kg / dt_h
        return min(electrical_limit, self.readiness_profile.standby_kw + production_kw)

    def run(self, power_kw, dt_h, *, co2_headroom_kg=math.inf):
        production_limit = max(0.0, co2_headroom_kg) * self.productive_specific_energy_kwh_per_kg / dt_h
        productive = self._advance_readiness(power_kw, dt_h, production_limit)
        return productive * dt_h / self.productive_specific_energy_kwh_per_kg

    def snapshot(self):
        return {**super().snapshot(), **self.readiness_snapshot()}


class DynamicMethanator(_Readiness, Methanator):
    readiness_profile = METHANATOR_READINESS

    def __init__(self, config=None):
        super().__init__(config)
        self._reset_readiness()

    def reset(self):
        super().reset()
        self._reset_readiness()

    @property
    def productive_specific_energy_kwh_per_kg(self):
        return (self.config.specific_energy_kwh_per_kg *
                (1.0 - self.readiness_profile.standby_kw / self.config.rated_kw))

    def electrical_intake_limit_kw(self, dt_h):
        return min(super().electrical_intake_limit_kw(dt_h), self.available_kw)

    def max_power_kw(self, dt_h, h2_available_kg, co2_available_kg, ch4_headroom_kg):
        electrical_limit = self.electrical_intake_limit_kw(dt_h)
        if self.readiness_kwh < self.readiness_profile.capacity_kwh - 1e-9:
            return electrical_limit
        mass_limit = min(max(0.0, h2_available_kg) / self.config.h2_kg_per_kg_ch4,
                         max(0.0, co2_available_kg) / self.config.co2_kg_per_kg_ch4,
                         max(0.0, ch4_headroom_kg))
        return min(electrical_limit, self.readiness_profile.standby_kw +
                   mass_limit * self.productive_specific_energy_kwh_per_kg / dt_h)

    def run(self, power_kw, dt_h, *, h2_available_kg, co2_available_kg, ch4_headroom_kg):
        mass_limit = min(max(0.0, h2_available_kg) / self.config.h2_kg_per_kg_ch4,
                         max(0.0, co2_available_kg) / self.config.co2_kg_per_kg_ch4,
                         max(0.0, ch4_headroom_kg))
        productive = self._advance_readiness(min(power_kw, self.available_kw), dt_h,
            mass_limit * self.productive_specific_energy_kwh_per_kg / dt_h)
        ch4_kg = min(mass_limit, productive * dt_h / self.productive_specific_energy_kwh_per_kg)
        return MethanatorResult(ch4_kg, ch4_kg * self.config.h2_kg_per_kg_ch4,
                               ch4_kg * self.config.co2_kg_per_kg_ch4)

    def snapshot(self):
        return {**super().snapshot(), **self.readiness_snapshot()}


@dataclass(frozen=True)
class WholePlantFlows(PowerFlows):
    battery_to_dac_kw: float = 0.0
    battery_to_methanator_kw: float = 0.0


def dispatch(action, solar_kw, *, battery, electrolyser, h2_tank, dac, co2_tank,
             methanator, ch4_tank, dt_h, shared_bus):
    """Reserve support once, net once, run units, then discharge exactly once.

    Reservations are proportional after consumer intake clipping. Refused
    reservations are retracted, not reassigned. Explicit battery charging is
    the sole charging allocation; no refused process solar is autocharged.
    """
    solar = [action.electrolyser_share * solar_kw, action.dac_share * solar_kw,
             action.methanator_share * solar_kw]
    charge = action.battery_share * solar_kw
    limits = [electrolyser.intake_limit_kw(dt_h, h2_tank.headroom_kg),
              dac.intake_limit_kw(dt_h, co2_tank.headroom_kg),
              (methanator.electrical_intake_limit_kw(dt_h) if isinstance(methanator, DynamicMethanator)
               else methanator.available_kw)]
    requests = [action.battery_support_kw,
                action.battery_dac_support_kw if shared_bus else 0.0,
                action.battery_methanator_support_kw if shared_bus else 0.0]
    support = [min(request, max(0.0, limit - pv))
               for request, limit, pv in zip(requests, limits, solar)]
    total = sum(support)
    available = battery.discharge_headroom_kw(dt_h)
    if total > available:
        support = [value * available / total for value in support]
        total = sum(support)
    overlap = min(charge, total)
    if total > 0.0:
        netting = [overlap * value / total for value in support]
        solar = [pv + direct for pv, direct in zip(solar, netting)]
        support = [max(0.0, value - direct) for value, direct in zip(support, netting)]
    charge = max(0.0, charge - overlap)
    solar = [min(pv, limit) for pv, limit in zip(solar, limits)]
    offered = [pv + extra for pv, extra in zip(solar, support)]

    if electrolyser.commitment_enabled:
        result = electrolyser.advance_commitment(offered[0], dt_h,
            h2_headroom_kg=h2_tank.headroom_kg,
            commitment_intent=offered[0] > COMMITMENT_INTENT_DEADBAND_FRAC * electrolyser.config.rated_kw)
        h2_kg = h2_tank.store(result.h2_produced_kg)
    elif offered[0] < electrolyser.min_load_kw:
        electrolyser.run(0.0, dt_h)
        h2_kg = 0.0
    else:
        h2_kg = h2_tank.store(electrolyser.run(offered[0], dt_h))

    if isinstance(dac, DynamicDAC):
        co2_kg = co2_tank.store(dac.run(offered[1], dt_h, co2_headroom_kg=co2_tank.headroom_kg))
    else:
        co2_kg = co2_tank.store(dac.run(offered[1], dt_h))
    result = methanator.run(offered[2], dt_h, h2_available_kg=h2_tank.level_kg,
                           co2_available_kg=co2_tank.level_kg, ch4_headroom_kg=ch4_tank.headroom_kg)
    h2_tank.withdraw(result.h2_kg)
    co2_tank.withdraw(result.co2_kg)
    ch4_kg = ch4_tank.store(result.ch4_kg)

    accepted = [electrolyser.power_kw, dac.power_kw, methanator.power_kw]
    used_solar = [min(pv, used) for pv, used in zip(solar, accepted)]
    actual_support = [max(0.0, used - pv) for used, pv in zip(accepted, used_solar)]
    discharge = sum(actual_support)
    if discharge > 0.0:
        discharged = battery.discharge(discharge, dt_h)
        if abs(discharged - discharge) > 1e-8:
            raise RuntimeError("Whole-plant support changed after proportional reservation")
        charged = 0.0
    else:
        charged = battery.charge(charge, dt_h)
    curtailed = max(0.0, solar_kw - charged - sum(used_solar))
    return WholePlantFlows(solar_kw, charged, *used_solar, actual_support[0],
                           *accepted, curtailed, h2_kg, co2_kg, ch4_kg,
                           actual_support[1], actual_support[2])


class WholePlantSimulator(DirectPlantSimulator):
    def __init__(self, config=None, *, electrolyser_commitment=None,
                 shared_bus=True, downstream_dynamics=True):
        self.shared_bus = bool(shared_bus)
        self.downstream_dynamics = bool(downstream_dynamics)
        super().__init__(config, electrolyser_commitment=electrolyser_commitment)
        if self.downstream_dynamics:
            self.dac = DynamicDAC(self.config.dac)
            self.methanator = DynamicMethanator(self.config.methanator)
            self.components.update(dac=self.dac, methanator=self.methanator)
            self.reset()

    def _build_state(self, flows, action):
        base = super()._build_state(flows, action)
        extras = {"battery_to_dac_kw": getattr(flows, "battery_to_dac_kw", 0.0),
                  "battery_to_methanator_kw": getattr(flows, "battery_to_methanator_kw", 0.0)}
        for name in ("dac", "methanator"):
            unit = getattr(self, name)
            if isinstance(unit, _Readiness):
                extras.update({f"{name}_{key}": value for key, value in unit.readiness_snapshot().items()})
            else:
                extras[f"{name}_productive_power_kw"] = unit.power_kw
        original = {field.name: getattr(base, field.name) for field in fields(PlantState)}
        # The exact legacy dispatch constructs its five-field Action. Preserve
        # that command while making unavailable downstream routes explicit in
        # the common seven-field witness used by every experimental arm.
        original["action"] = WholePlantAction(**base.action.to_dict())
        return WholePlantState(**original, **extras)

    def step(self, action=None):
        safe = WholePlantAction.sanitised(action, max_discharge_kw=self.config.battery.max_discharge_kw)
        # This branch is the unchanged old transition for embedded old actions.
        if not self.shared_bus and not self.downstream_dynamics:
            return super().step(safe)
        dt_h = self.config.dt_h
        if self._solar_factor_tape is not None:
            self._solar_factor_tape.for_interval(self.step_index)
        self.events.at(self.time_s, self.state.clock)
        self._apply_faults()
        solar_kw = self._update_solar(step_index=self.step_index + 1)
        flows = dispatch(safe, solar_kw, battery=self.battery, electrolyser=self.electrolyser,
                         h2_tank=self.h2_tank, dac=self.dac, co2_tank=self.co2_tank,
                         methanator=self.methanator, ch4_tank=self.ch4_tank,
                         dt_h=dt_h, shared_bus=self.shared_bus)
        self.time_s += self.config.dt_s
        self.step_index += 1
        self.energy_produced_kwh += flows.solar_available_kw * dt_h
        self.energy_curtailed_kwh += flows.solar_curtailed_kw * dt_h
        self.h2_produced_total_kg += flows.h2_produced_kg
        self.co2_captured_total_kg += flows.co2_captured_kg
        self.ch4_produced_total_kg += flows.ch4_produced_kg
        delivery = self.delivery_ledger.settle(self.step_index, available_kg=self.ch4_tank.level_kg)
        if delivery.delivered_kg > 0.0:
            self.ch4_tank.withdraw(delivery.delivered_kg)
        self.state = self._build_state(flows, safe)
        self.events.at(self.time_s, self.state.clock)
        self._observe_conditions(self.state)
        self.telemetry = self._observe(self.state)
        return self.state

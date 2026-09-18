"""Causal inventory and readiness rules for all four whole-plant experiment arms.

The same algorithm uses each arm's declared static electrical authority and
readiness parameters. It reads only the raw 149-coordinate public observation;
the environment is not retained by the policy. Rules apply to these comparators,
not to PPO. Constants are engineering assumptions, not calibrated plant data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np

from docs.research.tools.comparison_ppo_v2.baselines import (
    BaselineConfig as ParentConfig, OBSERVATION_NAMES as PARENT_NAMES,
)

SCHEMA = "wholeplant-causal-baseline-v1"
EXTRA_NAMES = tuple(f"{unit}_{name}" for unit in ("dac", "methanator") for name in
    ("readiness_fraction", "off", "starting", "standby", "producing", "power_fraction")) + (
    "battery_to_dac_fraction", "battery_to_methanator_fraction")
OBSERVATION_NAMES = (*PARENT_NAMES, *EXTRA_NAMES)
INDEX = {name: index for index, name in enumerate(OBSERVATION_NAMES)}
FORECAST_FIELDS = ("solar_forecast_0_1h_mean", "solar_forecast_0_1h_p10", "solar_forecast_1_6h_mean")
FORECAST_INDICES = tuple(INDEX[name] for name in FORECAST_FIELDS)
KINDS = ("daylight", "storage", "storage-persistence")
ARMS = ("restricted-instant", "shared-instant", "restricted-dynamic", "shared-dynamic")


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class BaselineConfig(ParentConfig):
    """Public ratings, specified readiness model and outcome-independent rules."""

    shared_bus: bool = True
    downstream_dynamics: bool = True
    dac_readiness_kwh: float = 120.0
    dac_startup_heater_kw: float = 300.0
    dac_standby_kw: float = 60.0
    methanator_readiness_kwh: float = 72.0
    methanator_startup_heater_kw: float = 90.0
    methanator_standby_kw: float = 18.0
    electrolyser_retention_max_h: float = 2.0

    def __post_init__(self):
        if not isinstance(self.shared_bus, bool) or not isinstance(self.downstream_dynamics, bool):
            raise ValueError("Arm authority and dynamics must be Boolean")
        numeric = {key: value for key, value in asdict(self).items()
                   if key not in {"shared_bus", "downstream_dynamics"}}
        if any(not math.isfinite(float(value)) or value <= 0 for value in numeric.values()):
            raise ValueError("Baseline constants must be finite and positive")
        if not 0 < self.battery_floor_soc < self.battery_target_soc <= 1:
            raise ValueError("Invalid battery floor or target")
        if not 0 < self.sunrise_h < self.sunset_h < 24:
            raise ValueError("Invalid daylight window")
        for unit in ("dac", "methanator"):
            if not 0 < getattr(self, unit + "_standby_kw") < getattr(self, unit + "_startup_heater_kw") <= getattr(self, unit + "_rated_kw"):
                raise ValueError("Readiness heating and standby must fit the unit rating")

    @classmethod
    def from_plant_config(cls, plant, *, shared_bus=True, downstream_dynamics=True):
        return cls(**asdict(ParentConfig.from_plant_config(plant)),
                   shared_bus=shared_bus, downstream_dynamics=downstream_dynamics)


class BaselinePolicy:
    """Stateless, deterministic policy; only public inputs enter each decision."""

    def __init__(self, kind="daylight", config: BaselineConfig | None = None):
        if kind == "persistence":
            kind = "storage-persistence"
        if kind not in KINDS:
            raise ValueError(f"Unknown whole-plant baseline: {kind}")
        self.kind = kind
        self.config = config or BaselineConfig()

    def metadata(self):
        return {"schema": SCHEMA, "kind": self.kind, "config": asdict(self.config),
            "input": "actual raw public 149-coordinate observation; static plant/arm constants only",
            "observation_names": list(OBSERVATION_NAMES), "consumed_forecast_fields": list(FORECAST_FIELDS),
            "consumed_forecast_indices": list(FORECAST_INDICES), "current_solar_index": INDEX["solar_available"],
            "action": "four jointly normalized solar shares and three jointly bounded battery supports; 15-minute hold",
            "additional_privileged_information": False, "stateful_memory": False,
            "fitted_or_selected_using_outcomes": False,
            "profile_provenance": "declared synthetic electrical-equivalent readiness assumptions, not plant calibration",
            "buffer_rule": "48 hours of methane plus 12 hours of feedstock; replenish upstream deficits over six productive hours",
            "readiness_rule": "reserve useful heater offers before discretionary nameplate production; pay ready-unit standby within productive offers",
            "storage_rule": "surplus charging toward 35%; commitment and startup bridging; short-gap readiness retention; next-delivery urgency may justify battery production",
            "retention_rule": "electrolyser at most two hours; downstream no longer than cold-restart electricity divided by standby power",
            "forecast_caveat": "one-hour mean/p10 are window statistics, not guaranteed minute power",
            "numerical_margin_kw": self.config.support_rounding_margin_kw,
            "transformation": ("replace the three consumed forecasts jointly with current solar availability" if self.kind == "storage-persistence" else "identity")}

    @staticmethod
    def _validate(observation):
        raw = np.asarray(observation)
        if raw.dtype.kind not in "iuf" or raw.ndim not in (1, 2) or raw.shape[-1] != 149 or not np.isfinite(raw).all():
            raise ValueError("Expected a finite public149 observation or batch")
        return raw

    def transform_observation(self, observation):
        raw = self._validate(observation)
        result = raw.copy()
        if self.kind == "storage-persistence":
            result[..., list(FORECAST_INDICES)] = raw[..., INDEX["solar_available"], None]
        return result

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        obs = self.transform_observation(observation)
        if obs.ndim == 1:
            return self._choose(obs.astype(np.float64)), None
        return np.stack([self._choose(row.astype(np.float64)) for row in obs]), None

    def inference(self, observation):
        raw = self._validate(observation)
        if raw.ndim != 1:
            raise ValueError("Inference evidence records one actual decision input")
        controller_input = self.transform_observation(raw)
        action = self._choose(controller_input.astype(np.float64))
        replacements = [] if self.kind != "storage-persistence" else [
            {"field": name, "index": i, "original": float(raw[i]), "replacement": float(controller_input[i])}
            for name, i in zip(FORECAST_FIELDS, FORECAST_INDICES)]
        return action, {"actual_observation": raw.tolist(), "actual_observation_sha256": _sha(raw.tolist()),
            "controller_input": controller_input.tolist(), "controller_input_sha256": _sha(controller_input.tolist()),
            "transformation": "current-solar-persistence" if replacements else "identity",
            "replacements": replacements, "action": action.tolist()}

    def _choose(self, obs):
        c = self.config
        v = lambda name: float(obs[INDEX[name]])
        solar = max(0.0, v("solar_available")) * c.solar_peak_kw
        mean = max(0.0, v(FORECAST_FIELDS[0])) * c.solar_peak_kw
        p10 = max(0.0, v(FORECAST_FIELDS[1])) * c.solar_peak_kw
        next_mean = max(0.0, v(FORECAST_FIELDS[2])) * c.solar_peak_kw
        working_solar = max(1.0, 0.5 * (solar + mean))
        conservative_solar = min(solar, p10)
        hour = math.atan2(v("time_of_day_sin"), v("time_of_day_cos")) * 12 / math.pi % 24
        until_sun = (c.sunrise_h - hour) % 24
        in_daylight = c.sunrise_h <= hour < c.sunset_h
        requested_kg = max(0.0, v("delivery_next_amount")) * c.ch4_capacity_kg
        demand = requested_kg / c.delivery_period_h
        until_delivery = max(c.decision_h, v("delivery_time_to_next") * c.delivery_period_h)
        h2, co2, ch4 = (max(0.0, v(name + "_fill")) * getattr(c, name + "_capacity_kg") for name in ("h2", "co2", "ch4"))
        missing_methane = max(0.0, min(c.ch4_capacity_kg, demand * c.methane_buffer_h) - ch4)
        target_feedstock_ch4 = missing_methane + demand * c.feedstock_buffer_h
        deficits = np.array([max(0.0, min(c.h2_capacity_kg, target_feedstock_ch4*c.h2_per_ch4)-h2),
                             max(0.0, min(c.co2_capacity_kg, target_feedstock_ch4*c.co2_per_ch4)-co2)])
        urgent_ch4 = max(0.0, requested_kg - ch4)
        urgent_h2 = max(0.0, urgent_ch4*c.h2_per_ch4-h2)
        urgent_co2 = max(0.0, urgent_ch4*c.co2_per_ch4-co2)
        health = np.array([max(0.0, v(unit+"_health")) for unit in ("electrolyser", "dac", "methanator")])
        limits = np.array([getattr(c, unit+"_rated_kw")*max(0.0,v(unit+"_cap")) for unit in ("electrolyser", "dac", "methanator")])
        limits[health <= 0] = 0.0
        energy = max(0.0, v("battery_soc")-c.battery_floor_soc)*c.battery_capacity_kwh*c.battery_discharge_efficiency
        budget = min(c.battery_max_discharge_kw*max(0.0,v("support_cap")), energy/c.decision_h) if self.kind != "daylight" else 0.0
        access = np.array([budget > 0, budget > 0 and c.shared_bus, budget > 0 and c.shared_bus])

        starting = v("electrolyser_mode_cold_starting") > .5 or v("electrolyser_mode_hot_starting") > .5
        running = v("electrolyser_mode_running") > .5
        standby = v("electrolyser_mode_hot_standby") > .5
        active = starting or running or standby
        blocked = v("electrolyser_mode_cooldown") > .5 or v("electrolyser_min_down_remaining") > 1e-9
        compulsory = starting or (running and v("electrolyser_min_up_remaining") > 1e-9)
        e_start_remaining = v("electrolyser_startup_remaining")*c.cold_start_h
        e_required = (c.electrolyser_start_kw if starting and e_start_remaining > c.decision_h else c.electrolyser_min_stable_kw)
        desired = np.zeros(3)
        desired[0] = min(limits[0], deficits[0]/c.replenish_h*c.electrolyser_specific_energy_kwh_per_kg)
        if desired[0] > 0 or compulsory:
            desired[0] = min(limits[0], max(desired[0], c.electrolyser_min_stable_kw))
        if blocked:
            desired[0] = 0.0

        # Productive power includes readiness hold power and adjusts the
        # productive SEC so the original nameplate yields are preserved.
        standby_kw = np.array([c.dac_standby_kw,c.methanator_standby_kw]) if c.downstream_dynamics else np.zeros(2)
        heater_kw = np.array([c.dac_startup_heater_kw,c.methanator_startup_heater_kw])
        capacities = np.array([c.dac_readiness_kwh,c.methanator_readiness_kwh])
        readiness = np.array([np.clip(v(unit+"_readiness_fraction"),0,1) for unit in ("dac","methanator")]) if c.downstream_dynamics else np.ones(2)
        sec = np.array([c.dac_specific_energy_kwh_per_kg*(1-standby_kw[0]/c.dac_rated_kw)/max(health[1],1e-12),
                        c.methanator_specific_energy_kwh_per_kg*(1-standby_kw[1]/c.methanator_rated_kw)])
        rates = np.array([deficits[1]/c.replenish_h,
            min(missing_methane,h2/c.h2_per_ch4,co2/c.co2_per_ch4)/c.decision_h])
        desired[1:] = np.minimum(limits[1:], np.where(rates>0, standby_kw+rates*sec, 0))
        needed = np.array([deficits[1]>0, missing_methane>0 and (min(h2/c.h2_per_ch4,co2/c.co2_per_ch4)>0 or desired[0]>0)])
        cold_h = (1-readiness)*capacities/(heater_kw-np.maximum(standby_kw,1e-12)) if c.downstream_dynamics else np.zeros(2)

        def return_h(required_power):
            """Conservative coarse lead time, using only declared public windows."""
            if mean >= required_power:
                return c.decision_h if solar >= required_power else 1.0
            if not in_daylight and until_sun <= 1.0 and next_mean >= required_power:
                return until_sun + 1.0
            if next_mean >= required_power:
                return 6.0
            return until_sun + 1.0 if not in_daylight else 24.0

        warming = np.zeros(2,dtype=bool)
        retention = np.zeros(2,dtype=bool)
        for i in range(2):
            unit_index=i+1
            if limits[unit_index] <= 0:
                desired[unit_index]=0
                continue
            if c.downstream_dynamics and readiness[i] < 1-1e-7:
                # Warm before productive sunlight arrives when future useful
                # work is indicated; never heat a full-product, idle system.
                urgent = urgent_co2>0 if i==0 else urgent_ch4>0
                can_begin = mean>standby_kw[i] or (access[unit_index] and (
                    return_h(heater_kw[i]) <= cold_h[i]+1.0 or urgent))
                warming[i] = needed[i] and can_begin
                desired[unit_index] = min(limits[unit_index],heater_kw[i]) if warming[i] else 0.0
            elif c.downstream_dynamics and needed[i]:
                # Restart energy / hold power gives an explicit retention
                # break-even bound. It is not a mandatory reserve or daily rule.
                restart_electricity = capacities[i]*heater_kw[i]/(heater_kw[i]-standby_kw[i])
                retention[i] = return_h(max(heater_kw[i],desired[unit_index])) <= restart_electricity/standby_kw[i]
                if desired[unit_index] <= standby_kw[i] and retention[i]:
                    desired[unit_index] = standby_kw[i]

        def allocate(requests):
            assigned=np.zeros(3);remaining=working_solar
            # A committed electrolyser's required load precedes discretionary
            # warming. Downstream heater offers precede extra nameplate load.
            if compulsory and requests[0]>0:
                assigned[0]=min(requests[0],e_required,remaining);remaining-=assigned[0]
            for j in ([2,1] if urgent_ch4>0 else [1,2]):
                if warming[j-1]:
                    assigned[j]=min(requests[j],remaining);remaining-=assigned[j]
            residual=np.maximum(0,requests-assigned)
            if residual.sum()>0:
                assigned+=residual*min(1.0,remaining/float(residual.sum()))
            return assigned/working_solar

        solar_shares=allocate(desired)
        # E starts need a plausible productive window; remaining battery
        # energy, not its instantaneous power rating alone, funds any bridge.
        e_low=solar_shares[0]*conservative_solar
        e_expected=solar_shares[0]*mean
        plausible = e_expected>=c.electrolyser_min_stable_kw or (
            solar_shares[0]*next_mean>=c.electrolyser_min_stable_kw and e_expected>=c.electrolyser_start_kw)
        bridge_h = max(0,c.electrolyser_start_kw-e_low)*c.cold_start_h + max(0,c.electrolyser_min_stable_kw-e_expected)*c.minimum_up_h
        affordable = e_low>=c.electrolyser_start_kw or (access[0] and energy>=bridge_h)
        if not active and not (plausible and affordable) and urgent_h2<=0:
            desired[0]=0.0
        if active and desired[0]>0 and not compulsory and e_expected<c.electrolyser_min_stable_kw and solar_shares[0]*solar<c.electrolyser_min_stable_kw:
            keep = access[0] and return_h(c.electrolyser_min_stable_kw)<=c.electrolyser_retention_max_h
            desired[0] = c.electrolyser_standby_kw if keep else (desired[0] if urgent_h2>0 else 0.0)
        solar_shares=allocate(desired)

        required=np.zeros(3)
        if access[0] and desired[0]>0:
            if desired[0]<=c.electrolyser_standby_kw+1e-9:
                required[0]=c.electrolyser_standby_kw
            elif compulsory:
                required[0]=e_required
            elif not active:
                required[0]=c.electrolyser_start_kw
            elif plausible:
                required[0]=c.electrolyser_min_stable_kw
            if urgent_h2>0:
                productive_h=max(c.decision_h,until_delivery-(e_start_remaining if active else c.cold_start_h))
                required[0]=max(required[0],min(limits[0],max(c.electrolyser_min_stable_kw,urgent_h2/productive_h*c.electrolyser_specific_energy_kwh_per_kg)))
        for i in range(2):
            j=i+1
            if not access[j] or desired[j]<=0:
                continue
            if warming[i]:
                # An incomplete warmup can be paid before sun reaches its
                # heater rating, or when service urgency warrants starting.
                if return_h(heater_kw[i])<=cold_h[i]+1.0 or (urgent_co2>0 if i==0 else urgent_ch4>0):
                    required[j]=min(limits[j],heater_kw[i])
            elif retention[i]:
                required[j]=standby_kw[i]
            urgent_rate=(urgent_co2 if i==0 else min(urgent_ch4,h2/c.h2_per_ch4,co2/c.co2_per_ch4))/max(c.decision_h,until_delivery-cold_h[i])
            if urgent_rate>0 and not warming[i]:
                required[j]=min(limits[j],max(required[j],standby_kw[i]+urgent_rate*sec[i]))

        deficits_kw=np.maximum(0.0,required-solar_shares*conservative_solar)
        deficits_kw=np.where(deficits_kw>0, deficits_kw+c.support_rounding_margin_kw,0)
        supports=np.zeros(3);remaining=budget
        priority=([2,0,1] if urgent_ch4>0 else [0,1,2])
        for i in priority:
            supports[i]=min(deficits_kw[i],remaining) if access[i] and limits[i]>0 else 0
            remaining-=supports[i]
        # Charging is explicit and never relies on automatic recovery of
        # refused process power. Shared dispatch cancels any simultaneous net.
        shares=np.r_[0.0,solar_shares]
        if self.kind!='daylight':
            charge_room=max(0,c.battery_target_soc-v("battery_soc"))*c.battery_capacity_kwh
            charge_kw=min(c.battery_max_charge_kw,charge_room/(c.decision_h*c.battery_charge_efficiency))
            shares[0]=min(max(0,1-float(shares.sum())),charge_kw/working_solar)
        if solar<=1e-12 and mean<=1e-12:
            shares[:]=0.0
        shares=np.clip(shares,0,1);shares/=max(1,float(shares.sum()))
        supports=np.clip(supports,0,c.battery_max_discharge_kw)
        supports/=max(1,float(supports.sum())/c.battery_max_discharge_kw)
        return np.clip(np.r_[shares,supports/c.battery_max_discharge_kw]*2-1,-1,1).astype(np.float32)


def baseline(label, env):
    """Construct from public static configuration; never retain the environment."""
    if env.arm not in ARMS:
        raise ValueError("Unknown static plant arm")
    config=BaselineConfig.from_plant_config(env.config,shared_bus=env.shared_bus,
                                           downstream_dynamics=env.downstream_dynamics)
    return BaselinePolicy(label,config)


__all__=["BaselineConfig","BaselinePolicy","baseline","OBSERVATION_NAMES","INDEX","FORECAST_FIELDS","FORECAST_INDICES"]

"""Transparent inventory-aware comparators for the direct five-share interface.

Both controllers read exactly the raw 135-number policy observation.  They never
read an environment, scenario tape, future disturbance, or realised future PV.
The settings below are engineering choices fixed before development comparison,
not fitted parameters.  These rules constrain these comparators only, never PPO.

The basic rule builds two days of saleable methane plus half a day of feedstock,
then idles.  Upstream deficits are replenished over a six-hour production window;
power is shared in proportion to the resulting physical process requests.  The
daylight version uses no battery.  The storage version can bridge commitment
shortfalls and retain hot standby overnight, and explicitly charges surplus PV.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Literal

import numpy as np

SCHEMA = "comparison-direct-baseline-v2"
# This local, explicit schema keeps the policy independent of any environment
# object or simulator import. The fixture validation checks exact equality with
# the frozen beta observation names before it evaluates a policy.
OBSERVATION_NAMES = (
    "time_of_day_sin", "time_of_day_cos", "solar_available", "solar_potential",
    "solar_output_factor", "solar_trend", "battery_soc", "battery_power",
    "battery_reserve_margin", "h2_fill", "co2_fill", "ch4_fill",
    "electrolyser_load", "electrolyser_online", "electrolyser_health",
    "dac_load", "dac_online", "dac_health", "methanator_load",
    "methanator_online", "methanator_health", "h2_production", "co2_production",
    "ch4_production", "electrolyser_cap", "dac_cap", "methanator_cap",
    "support_cap", "reserve_soc", "episode_progress", "delivery_time_to_next",
    "delivery_next_amount", "delivery_inventory_coverage", "delivery_fill_rate_to_date",
    *(f"solar_forecast_{window}_{stat}" for window in ("0_1h", "1_6h", "6_12h", "12_24h") for stat in ("mean", "p10")),
    *(f"electrolyser_mode_{mode}" for mode in ("legacy", "off", "cold_starting", "running", "hot_standby", "hot_starting", "cooldown")),
    "electrolyser_previous_power", "electrolyser_ramp_up_headroom",
    "electrolyser_ramp_down_headroom", "electrolyser_startup_remaining",
    "electrolyser_cooldown_remaining", "electrolyser_min_up_remaining",
    "electrolyser_min_down_remaining", "electrolyser_shortfall_elapsed",
    "market_price_current",
    *(f"market_price_h{hour:02d}_{stat}" for hour in range(24) for stat in ("mean", "p10", "p90")),
    "cycle_progress", "cycle_battery_return_margin", "cycle_h2_return_margin",
    "cycle_co2_return_margin", "cycle_ch4_return_margin",
)
INDEX = {name: i for i, name in enumerate(OBSERVATION_NAMES)}
READ_FIELDS = (
    "time_of_day_sin", "time_of_day_cos", "solar_available", "battery_soc",
    "h2_fill", "co2_fill", "ch4_fill", "electrolyser_health", "dac_health",
    "methanator_health", "electrolyser_cap", "dac_cap", "methanator_cap",
    "support_cap", "delivery_next_amount", "solar_forecast_0_1h_mean",
    "solar_forecast_0_1h_p10", "solar_forecast_1_6h_mean",
    "electrolyser_mode_off", "electrolyser_mode_cold_starting",
    "electrolyser_mode_running", "electrolyser_mode_hot_standby",
    "electrolyser_mode_hot_starting", "electrolyser_mode_cooldown",
    "electrolyser_startup_remaining", "electrolyser_min_up_remaining",
    "electrolyser_min_down_remaining",
)


@dataclass(frozen=True)
class BaselineConfig:
    """Explicit plant constants and engineering decisions, in physical units."""

    solar_peak_kw: float = 5000.0
    battery_capacity_kwh: float = 8000.0
    battery_max_charge_kw: float = 2500.0
    battery_max_discharge_kw: float = 2500.0
    battery_charge_efficiency: float = 0.96
    battery_discharge_efficiency: float = 0.96
    battery_floor_soc: float = 0.05
    electrolyser_rated_kw: float = 3000.0
    electrolyser_specific_energy_kwh_per_kg: float = 52.0
    electrolyser_min_stable_kw: float = 1200.0
    electrolyser_start_kw: float = 600.0
    electrolyser_standby_kw: float = 60.0
    cold_start_h: float = 1.0
    minimum_up_h: float = 1.0
    dac_rated_kw: float = 1200.0
    dac_specific_energy_kwh_per_kg: float = 2.0
    methanator_rated_kw: float = 180.0
    methanator_specific_energy_kwh_per_kg: float = 1.2
    h2_per_ch4: float = 0.5
    co2_per_ch4: float = 2.75
    h2_capacity_kg: float = 2000.0
    co2_capacity_kg: float = 6000.0
    ch4_capacity_kg: float = 3000.0
    delivery_period_h: float = 6.0
    decision_h: float = 0.25
    sunrise_h: float = 6.0
    sunset_h: float = 19.5
    methane_buffer_h: float = 48.0
    feedstock_buffer_h: float = 12.0
    replenish_h: float = 6.0
    battery_target_soc: float = 0.35
    support_rounding_margin_kw: float = 0.001

    def __post_init__(self) -> None:
        if any(not math.isfinite(float(v)) or float(v) <= 0 for v in asdict(self).values()):
            raise ValueError("All baseline constants must be finite and positive")
        if not 0 < self.battery_floor_soc < self.battery_target_soc <= 1:
            raise ValueError("Battery target must exceed its physical floor")
        if self.sunrise_h >= self.sunset_h or self.sunset_h >= 24:
            raise ValueError("Expected a same-day daylight window")

    @classmethod
    def from_plant_config(cls, plant: Any) -> "BaselineConfig":
        """Copy public static ratings, with the experiment's declared AEL profile.

        The 0.4 stable, 0.2 start and 0.02 standby fractions are the frozen
        commitment profile, not estimates derived from controller outcomes.
        """
        return replace(cls(), solar_peak_kw=plant.solar.peak_kw,
            battery_capacity_kwh=plant.battery.capacity_kwh,
            battery_max_charge_kw=plant.battery.max_charge_kw,
            battery_max_discharge_kw=plant.battery.max_discharge_kw,
            battery_charge_efficiency=plant.battery.charge_efficiency,
            battery_discharge_efficiency=plant.battery.discharge_efficiency,
            battery_floor_soc=plant.battery.min_soc,
            electrolyser_rated_kw=plant.electrolyser.rated_kw,
            electrolyser_specific_energy_kwh_per_kg=plant.electrolyser.specific_energy_kwh_per_kg,
            electrolyser_min_stable_kw=0.4 * plant.electrolyser.rated_kw,
            electrolyser_start_kw=0.2 * plant.electrolyser.rated_kw,
            electrolyser_standby_kw=0.02 * plant.electrolyser.rated_kw,
            dac_rated_kw=plant.dac.rated_kw,
            dac_specific_energy_kwh_per_kg=plant.dac.specific_energy_kwh_per_kg,
            methanator_rated_kw=plant.methanator.rated_kw,
            methanator_specific_energy_kwh_per_kg=plant.methanator.specific_energy_kwh_per_kg,
            h2_per_ch4=plant.methanator.h2_kg_per_kg_ch4,
            co2_per_ch4=plant.methanator.co2_kg_per_kg_ch4,
            h2_capacity_kg=plant.h2_tank.capacity_kg,
            co2_capacity_kg=plant.co2_tank.capacity_kg,
            ch4_capacity_kg=plant.ch4_tank.capacity_kg,
            sunrise_h=plant.solar.sunrise_h, sunset_h=plant.solar.sunset_h)


class BaselinePolicy:
    """Deterministic, stateless policy compatible with SB3's predict signature."""

    def __init__(self, kind: Literal["daylight", "storage"] = "daylight",
                 config: BaselineConfig | None = None):
        if kind not in {"daylight", "storage"}:
            raise ValueError("Baseline kind must be daylight or storage")
        self.kind = kind
        self.config = config or BaselineConfig()

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA, "kind": self.kind, "config": asdict(self.config),
            "input": "raw public 135-coordinate PPO observation, before VecNormalize",
            "observation_names": list(OBSERVATION_NAMES),
            "read_observation_fields": list(READ_FIELDS),
            "action": "five direct coordinates held for 15 minutes, exactly as PPO",
            "privileged_future_information": False,
            "fitted_or_selected_using_outcomes": False,
            "stateful_memory": False,
            "forecast_interpretation": "p10 is a causal one-hour average forecast, not a guaranteed within-interval lower bound",
            "numerical_margin": "Add 0.001 kW to positive support requests, within physical and energy caps, so float32 action serialization cannot turn exact 60-kW standby into 59.999973 kW",
            "battery_policy": ("no charging or support" if self.kind == "daylight" else
                "surplus charge up to 35%; commitment bridging and affordable overnight hot standby; no daily restoration requirement"),
        }

    def predict(self, observation: Any, state=None, episode_start=None,
                deterministic: bool = True):
        obs = np.asarray(observation)
        if obs.dtype.kind not in "iuf" or not np.isfinite(obs).all():
            raise ValueError("Baseline requires finite numeric observations")
        if obs.ndim == 1 and obs.shape == (len(INDEX),):
            return self._choose(obs.astype(np.float64)), None
        if obs.ndim == 2 and obs.shape[1] == len(INDEX):
            return np.stack([self._choose(row.astype(np.float64)) for row in obs]), None
        raise ValueError("Baseline requires one 135-coordinate observation or a batch")

    def _choose(self, obs: np.ndarray) -> np.ndarray:
        c = self.config
        def value(name: str) -> float:
            return float(obs[INDEX[name]])

        hour = math.atan2(value("time_of_day_sin"), value("time_of_day_cos")) * 12 / math.pi % 24
        solar = max(0.0, value("solar_available")) * c.solar_peak_kw
        mean = max(0.0, value("solar_forecast_0_1h_mean")) * c.solar_peak_kw
        p10 = max(0.0, value("solar_forecast_0_1h_p10")) * c.solar_peak_kw
        next_mean = max(0.0, value("solar_forecast_1_6h_mean")) * c.solar_peak_kw
        # This working power is used to choose a *fraction*, not a promise that
        # the same absolute kW will persist during the next 15 minutes.
        working_solar = max(1.0, 0.5 * (solar + mean))
        conservative_solar = min(solar, p10)
        h2 = value("h2_fill") * c.h2_capacity_kg
        co2 = value("co2_fill") * c.co2_capacity_kg
        ch4 = value("ch4_fill") * c.ch4_capacity_kg
        demand = max(0.0, value("delivery_next_amount")) * c.ch4_capacity_kg / c.delivery_period_h
        methane_target = min(c.ch4_capacity_kg, demand * c.methane_buffer_h)
        missing_methane = max(0.0, methane_target - ch4)
        feedstock_target_ch4 = missing_methane + demand * c.feedstock_buffer_h
        h2_deficit = max(0.0, min(c.h2_capacity_kg, feedstock_target_ch4 * c.h2_per_ch4) - h2)
        co2_deficit = max(0.0, min(c.co2_capacity_kg, feedstock_target_ch4 * c.co2_per_ch4) - co2)

        e_limit = c.electrolyser_rated_kw * value("electrolyser_cap")
        d_limit = c.dac_rated_kw * value("dac_cap")
        m_limit = c.methanator_rated_kw * value("methanator_cap")
        support_limit = c.battery_max_discharge_kw * value("support_cap")
        e_wanted = min(e_limit, h2_deficit / c.replenish_h * c.electrolyser_specific_energy_kwh_per_kg)
        if e_wanted > 0:
            e_wanted = max(min(e_limit, c.electrolyser_min_stable_kw), e_wanted)
        d_wanted = min(d_limit, co2_deficit / c.replenish_h * c.dac_specific_energy_kwh_per_kg)
        # Methanation has no synthetic minimum load or startup in this model.
        # Convert available feedstocks promptly into the delivery buffer.
        m_wanted = min(m_limit, min(missing_methane, h2 / c.h2_per_ch4,
                                   co2 / c.co2_per_ch4) / c.decision_h * c.methanator_specific_energy_kwh_per_kg)
        if value("electrolyser_health") <= 0:
            e_wanted = 0.0
        if value("dac_health") <= 0:
            d_wanted = 0.0
        if value("methanator_health") <= 0:
            m_wanted = 0.0

        running = value("electrolyser_mode_running") > 0.5
        starting = value("electrolyser_mode_cold_starting") > 0.5 or value("electrolyser_mode_hot_starting") > 0.5
        standby = value("electrolyser_mode_hot_standby") > 0.5
        cooling = value("electrolyser_mode_cooldown") > 0.5
        blocked = cooling or value("electrolyser_min_down_remaining") > 1e-9
        battery_usable = max(0.0, value("battery_soc") - c.battery_floor_soc) * c.battery_capacity_kwh * c.battery_discharge_efficiency
        can_use_storage = self.kind == "storage" and battery_usable > 1e-9 and support_limit > 0
        active = running or starting or standby
        shares = np.zeros(4, dtype=np.float64)

        def allocate(e_kw: float, d_kw: float, m_kw: float) -> np.ndarray:
            # Treat process requests together so downstream electricity is not
            # accidentally omitted when the electrolyser is active.
            requests = np.asarray([0.0, e_kw, d_kw, m_kw], dtype=np.float64)
            return requests / max(working_solar, float(requests.sum()), 1.0)

        shares = allocate(e_wanted, d_wanted, m_wanted)
        prospective_e_share = float(shares[1])
        expected_e = prospective_e_share * mean
        low_e = prospective_e_share * conservative_solar
        # Cold starts require both a plausible productive hour and enough
        # forecast power for the 600-kW start. Battery can bridge its deficit.
        startup_bridge = max(0.0, c.electrolyser_start_kw - low_e) * c.cold_start_h
        productive_bridge = max(0.0, c.electrolyser_min_stable_kw - expected_e) * c.minimum_up_h
        enough_start_power = low_e >= c.electrolyser_start_kw or (
            can_use_storage and battery_usable >= startup_bridge + productive_bridge)
        daylight_promising = expected_e >= c.electrolyser_min_stable_kw or (
            next_mean * prospective_e_share >= c.electrolyser_min_stable_kw
            and mean * prospective_e_share >= c.electrolyser_start_kw)
        start_allowed = not blocked and enough_start_power and daylight_promising
        if not active and not start_allowed:
            e_wanted = 0.0
        if blocked:
            e_wanted = 0.0

        support = 0.0
        if active and e_wanted > 0:
            # Preserve an ongoing start or compulsory minimum-up interval if
            # energy is available. Otherwise production may shut down normally.
            compulsory = starting or value("electrolyser_min_up_remaining") > 1e-9
            productive_now = prospective_e_share * min(solar, mean) >= c.electrolyser_min_stable_kw
            daylight_soon = prospective_e_share * mean >= c.electrolyser_min_stable_kw
            if not compulsory and not productive_now and not daylight_soon:
                # A storage comparator may buy the 60-kW standby service until
                # sunrise; the daylight comparator simply shuts the unit down.
                until_sun = (c.sunrise_h - hour) % 24
                if c.sunrise_h <= hour < c.sunset_h:
                    until_sun = c.decision_h
                standby_energy = c.electrolyser_standby_kw * max(c.decision_h, until_sun)
                if can_use_storage and battery_usable >= standby_energy:
                    e_wanted = c.electrolyser_standby_kw
                else:
                    e_wanted = 0.0

        shares = allocate(e_wanted, d_wanted, m_wanted)
        if can_use_storage and e_wanted > 0:
            if e_wanted <= c.electrolyser_standby_kw + 1e-9:
                required = c.electrolyser_standby_kw
            elif starting or (not active and start_allowed):
                # A start may finish inside this action hold; budget the stable
                # load as well on the last startup quarter-hour.
                remaining = value("electrolyser_startup_remaining") * c.cold_start_h
                required = c.electrolyser_min_stable_kw if starting and remaining <= c.decision_h else c.electrolyser_start_kw
            else:
                required = c.electrolyser_min_stable_kw
            support = min(support_limit, battery_usable / c.decision_h,
                          max(0.0, required - shares[1] * conservative_solar))
            if support > 0:
                support = min(support_limit, battery_usable / c.decision_h,
                              support + c.support_rounding_margin_kw)

        if self.kind == "storage":
            remaining_share = max(0.0, 1.0 - float(shares.sum()))
            charging_room = max(0.0, c.battery_target_soc - value("battery_soc")) * c.battery_capacity_kwh
            charge_kw = min(c.battery_max_charge_kw, charging_room / (c.decision_h * c.battery_charge_efficiency))
            shares[0] = min(remaining_share, charge_kw / working_solar)
        # Off has an exact reachable representation; no infinitesimal positive
        # E request is added to dodge the plant's declared commitment deadband.
        if solar <= 1e-12 and mean <= 1e-12:
            shares[:] = 0.0
        shares = np.clip(shares, 0.0, 1.0)
        shares /= max(1.0, float(shares.sum()))
        action = np.r_[shares, support / c.battery_max_discharge_kw] * 2.0 - 1.0
        return np.clip(action, -1.0, 1.0).astype(np.float32)


__all__ = ["BaselineConfig", "BaselinePolicy", "INDEX", "OBSERVATION_NAMES", "READ_FIELDS", "SCHEMA"]

"""Independent physical auditor copied from frozen wholeplant evaluator.
Only change: declared outages are checked against each explicit target, rather
than assuming every outage is DAC. All interval balances are unchanged.
"""
import hashlib
import json
import math
PROCESSES = ("electrolyser", "dac", "methanator")

class MinuteAudit:
    """Independent interval balances, including all battery outlets and readiness.

    Readiness is an assumed electrical-equivalent reservoir, not a calibrated
    temperature model. Its coefficients are deliberately declared here rather
    than calling the simulator's transition function to audit itself.
    """
    READINESS = {"dac": {"capacity_kwh": 120.0, "heater_kw": 300.0, "loss_kw": 60.0},
                 "methanator": {"capacity_kwh": 72.0, "heater_kw": 90.0, "loss_kw": 18.0}}

    def __init__(self, env):
        self.env, self.previous, self.count = env, None, 0
        self.failures, self.failure_count = [], 0
        self.sha = hashlib.sha256()
        self.minimum_soc = float("inf")
        self.maxima = {}
        self.delivery_cache_decision = None
        self.deliveries_by_step = {}
        self.delivered_checked_kg = 0.0
        self.dynamic = env.arm.endswith("dynamic")
        self.shared = env.arm.startswith("shared")

    def fail(self, rule, value, step):
        self.failure_count += 1
        if len(self.failures) < 25:
            self.failures.append({"rule": rule, "value": value, "step": step})

    def equal(self, rule, observed, expected, step, tolerance=1e-6):
        residual = abs(observed - expected)
        self.maxima[rule] = max(self.maxima.get(rule, 0.0), residual)
        if not math.isfinite(residual) or residual > tolerance:
            self.fail(rule, {"observed": observed, "expected": expected, "residual": residual}, step)

    def bounded(self, rule, value, low, high, step, tolerance=1e-7):
        if not math.isfinite(value) or not low-tolerance <= value <= high+tolerance:
            self.fail(rule, {"value": value, "low": low, "high": high}, step)

    def add(self, state):
        s = state.to_dict() if hasattr(state, "to_dict") else state
        env, cfg, step = self.env, self.env.config, self.count
        dt = cfg.dt_h
        raw = json.dumps(s, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.sha.update(len(raw).to_bytes(8, "big")); self.sha.update(raw)
        self.equal("clock_step", s["step"], step, step, 0)
        self.equal("clock_seconds", s["time_s"], env.episode_config.start_hour*3600 + step*cfg.dt_s, step, 1e-7)
        for name, level, cap in (("battery", "battery_energy_kwh", "battery_capacity_kwh"),
                                 ("h2", "h2_level_kg", "h2_capacity_kg"),
                                 ("co2", "co2_level_kg", "co2_capacity_kg"),
                                 ("ch4", "ch4_level_kg", "ch4_capacity_kg")):
            self.bounded(name+"_bounds", s[level], 0, s[cap], step)
        self.bounded("physical_battery_floor", s["battery_soc"], cfg.battery.min_soc, 1, step, 1e-9)
        self.equal("battery_soc_definition", s["battery_soc"]*s["battery_capacity_kwh"], s["battery_energy_kwh"], step)
        self.minimum_soc = min(self.minimum_soc, s["battery_soc"])
        for unit in PROCESSES:
            self.bounded(unit+"_power_limit", s[unit+"_power_kw"], 0, getattr(cfg, unit).rated_kw, step)
            self.bounded("solar_to_"+unit+"_nonnegative", s["solar_to_"+unit+"_kw"], 0, s["solar_available_kw"], step)
            self.bounded("battery_to_"+unit+"_limit", s["battery_to_"+unit+"_kw"], 0, cfg.battery.max_discharge_kw, step)
        for unit, constants in self.READINESS.items():
            if self.previous is None:
                self.equal(unit+"_initial_readiness", s[unit+"_readiness_kwh"], 0, step)
                if s[unit+"_mode"] != ("off" if self.dynamic else "instant"):
                    self.fail(unit+"_initial_mode", s[unit+"_mode"], step)
            self.bounded(unit+"_readiness_bounds", s[unit+"_readiness_kwh"], 0, constants["capacity_kwh"], step)
            self.equal(unit+"_readiness_fraction", s[unit+"_readiness_fraction"], s[unit+"_readiness_kwh"]/constants["capacity_kwh"], step, 1e-9)
        if self.previous is not None:
            previous = self.previous
            if self.delivery_cache_decision != env.decisions_taken:
                self.deliveries_by_step = {}
                for item in env.simulator.delivery_audit()["settlements"]:
                    due = item["due_step"]
                    self.deliveries_by_step[due] = self.deliveries_by_step.get(due, 0.0) + item["delivered_kg"]
                    self.equal("delivery_mass_accounting", item["requested_kg"], item["delivered_kg"]+item["unmet_kg"], step)
                    self.bounded("delivered_mass_nonnegative", item["delivered_kg"], 0, item["requested_kg"], step)
                    self.bounded("unmet_mass_nonnegative", item["unmet_kg"], 0, item["requested_kg"], step)
                self.delivery_cache_decision = env.decisions_taken
            produced = {"h2": s["h2_produced_total_kg"]-previous["h2_produced_total_kg"],
                        "co2": s["co2_captured_total_kg"]-previous["co2_captured_total_kg"],
                        "ch4": s["ch4_produced_total_kg"]-previous["ch4_produced_total_kg"]}
            delivered = self.deliveries_by_step.get(step, 0.0)
            self.delivered_checked_kg += delivered
            changes = {"h2": produced["h2"]-produced["ch4"]*cfg.methanator.h2_kg_per_kg_ch4,
                       "co2": produced["co2"]-produced["ch4"]*cfg.methanator.co2_kg_per_kg_ch4,
                       "ch4": produced["ch4"]-delivered}
            for resource, change in changes.items():
                self.equal(resource+"_mass_balance", s[resource+"_level_kg"]-previous[resource+"_level_kg"], change, step)
                self.bounded(resource+"_production_nonnegative", produced[resource], 0, 1e9, step)
            for resource, rate in (("h2", "h2_production_kg_h"), ("co2", "co2_capture_kg_h"), ("ch4", "ch4_production_kg_h")):
                self.equal(resource+"_reported_production", produced[resource], s[rate]*dt, step)
            self.bounded("electrolyser_production_energy", produced["h2"]*cfg.electrolyser.specific_energy_kwh_per_kg,
                         0, s["electrolyser_power_kw"]*dt, step)
            self.equal("solar_forcing", s["solar_available_kw"], env.frozen_source.expected_solar_kw[step-1], step, 1e-7)
            solar_total = s["solar_to_battery_kw"]+s["solar_curtailed_kw"]+sum(s["solar_to_"+u+"_kw"] for u in PROCESSES)
            self.equal("solar_power_balance", solar_total, s["solar_available_kw"], step)
            discharge = sum(s["battery_to_"+u+"_kw"] for u in PROCESSES)
            self.bounded("battery_total_discharge_limit", discharge, 0, cfg.battery.max_discharge_kw, step)
            self.bounded("battery_charge_limit", s["solar_to_battery_kw"], 0, cfg.battery.max_charge_kw, step)
            self.bounded("explicit_battery_charge", s["solar_to_battery_kw"], 0, s["action"]["battery_share"]*s["solar_available_kw"], step)
            self.equal("whole_bus_balance", s["solar_available_kw"]+discharge,
                       sum(s[u+"_power_kw"] for u in PROCESSES)+s["solar_to_battery_kw"]+s["solar_curtailed_kw"], step)
            self.equal("battery_energy_balance", s["battery_energy_kwh"]-previous["battery_energy_kwh"],
                       (s["solar_to_battery_kw"]*cfg.battery.charge_efficiency-discharge/cfg.battery.discharge_efficiency)*dt, step)
            self.equal("battery_power_balance", s["battery_power_kw"], s["solar_to_battery_kw"]-discharge, step)
            if s["solar_to_battery_kw"] > 1e-7 and discharge > 1e-7:
                self.fail("simultaneous_physical_charge_discharge", {"charge": s["solar_to_battery_kw"], "discharge": discharge}, step)
            for unit in PROCESSES:
                self.equal(unit+"_power_balance", s[unit+"_power_kw"], s["solar_to_"+unit+"_kw"]+s["battery_to_"+unit+"_kw"], step)
            if not self.shared:
                self.equal("restricted_dac_battery", s["battery_to_dac_kw"], 0, step)
                self.equal("restricted_methanator_battery", s["battery_to_methanator_kw"], 0, step)
            schedule = env.frozen_source.original_manifest["disturbances"]["schedule"]
            for unit, health in s["health"].items():
                outage = any(f["target"] == unit and f["mode"] == "outage" and f["start_s"] <= (step-1)*cfg.dt_s < f["start_s"]+f["duration_s"] for f in schedule)
                if health["offline"] != outage or health["degradation"] != 0:
                    self.fail("declared_health_schedule", unit, step)
            for unit in PROCESSES:
                if s["health"][unit]["offline"]:
                    self.equal(unit+"_offline_power", s[unit+"_power_kw"], 0, step)
            for unit, resource in (("dac", "co2"), ("methanator", "ch4")):
                c = self.READINESS[unit]
                startup, standby, productive, loss = (s[unit+"_"+name+"_kw"] for name in ("startup_power", "standby_power", "productive_power", "heat_loss"))
                for name, power, bound in (("startup", startup, c["heater_kw"]), ("standby", standby, c["loss_kw"]), ("productive", productive, getattr(cfg, unit).rated_kw), ("heat_loss", loss, c["loss_kw"])):
                    self.bounded(unit+"_"+name+"_limit", power, 0, bound, step)
                self.equal(unit+"_paid_power_split", s[unit+"_power_kw"], startup+standby+productive, step)
                if self.dynamic:
                    before = previous[unit+"_readiness_kwh"]
                    expected_loss = min(c["loss_kw"], before/dt+startup+standby)
                    self.equal(unit+"_heat_loss", loss, expected_loss, step)
                    self.equal(unit+"_readiness_balance", s[unit+"_readiness_kwh"]-before, (startup+standby-loss)*dt, step)
                    if before < c["capacity_kwh"]-1e-9:
                        self.equal(unit+"_cold_standby_split", standby, 0, step)
                    else:
                        self.equal(unit+"_hot_startup_split", startup, 0, step)
                    if before < c["capacity_kwh"]-1e-7 or startup > 1e-7:
                        self.equal(unit+"_unready_productive_power", productive, 0, step)
                        self.equal(unit+"_unready_production", produced[resource], 0, step)
                    if productive > 1e-7:
                        self.equal(unit+"_running_standby", standby, c["loss_kw"], step)
                    if s["health"][unit]["offline"]:
                        expected_mode = "fault"
                    elif s[unit+"_readiness_kwh"] <= 1e-9 and startup+standby == 0:
                        expected_mode = "off"
                    elif before < c["capacity_kwh"]-1e-9:
                        expected_mode = "starting" if startup > 0 else ("cooling" if before > 0 else "off")
                    else:
                        expected_mode = "running" if productive > 0 else ("standby" if standby >= c["loss_kw"] else "cooling")
                    if s[unit+"_mode"] != expected_mode:
                        self.fail(unit+"_mode_transition", {"observed": s[unit+"_mode"], "expected": expected_mode}, step)
                    started = int(before < c["capacity_kwh"]-1e-9 and startup > 0 and previous[unit+"_mode"] != "starting")
                    self.equal(unit+"_start_counter", s[unit+"_starts"]-previous[unit+"_starts"], started, step, 0)
                    self.equal(unit+"_cold_start_counter", s[unit+"_cold_starts"]-previous[unit+"_cold_starts"], started*int(before <= 1e-9), step, 0)
                    specific = getattr(cfg, unit).specific_energy_kwh_per_kg*(1-c["loss_kw"]/getattr(cfg, unit).rated_kw)
                else:
                    if s[unit+"_mode"] != "instant":
                        self.fail(unit+"_instant_mode", s[unit+"_mode"], step)
                    self.equal(unit+"_instant_startup", startup, 0, step)
                    self.equal(unit+"_instant_standby", standby, 0, step)
                    self.equal(unit+"_instant_loss", loss, 0, step)
                    self.equal(unit+"_instant_readiness_unchanged", s[unit+"_readiness_kwh"], previous[unit+"_readiness_kwh"], step)
                    specific = getattr(cfg, unit).specific_energy_kwh_per_kg
                health_factor = max(0.0, 1-s["health"][unit]["degradation"])
                self.equal(unit+"_productive_energy_recipe", produced[resource]*specific, productive*dt*health_factor, step)
                for counter in ("starts", "cold_starts"):
                    delta = s[unit+"_"+counter]-previous[unit+"_"+counter]
                    self.bounded(unit+"_"+counter+"_increment", delta, 0, 1, step)
                self.bounded(unit+"_cold_starts_subset", s[unit+"_cold_starts"], 0, s[unit+"_starts"], step)
        self.previous = s
        self.count += 1

    def result(self):
        return {"all_physical_invariants_passed": self.failure_count == 0,
                "states_checked": self.count, "failure_count": self.failure_count, "first_failures": self.failures,
                "minimum_battery_soc": self.minimum_soc if self.count else None,
                "physical_battery_floor": self.env.config.battery.min_soc,
                "maximum_balance_residuals": self.maxima,
                "maximum_solar_power_balance_residual_kw": self.maxima.get("solar_power_balance", 0),
                "maximum_battery_energy_balance_residual_kwh": self.maxima.get("battery_energy_balance", 0),
                "maximum_gas_mass_balance_residual_kg": {r: self.maxima.get(r+"_mass_balance", 0) for r in ("h2", "co2", "ch4")},
                "settled_delivery_mass_checked_kg": self.delivered_checked_kg,
                "readiness_model": "Assumed electrical-equivalent process readiness, not calibrated temperature",
                "readiness_coefficients": self.READINESS,
                "stream_sha256": self.sha.hexdigest(),
                "stream_encoding": "SHA256 over each canonical JSON state prefixed by unsigned8-byte big-endian byte length",
                "daily_returns_gate": False, "forced_stops_gate": False, "critical_status_gate": False}



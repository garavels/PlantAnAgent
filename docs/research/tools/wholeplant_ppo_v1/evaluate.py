"""Frozen one-attempt whole-plant evaluations with independent bus/readiness audits.

This is a new experiment. No completed study module or global is modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import signal
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.research.tools.free_ppo_v1.analyze import canonical_sha, file_pin, publish, RESOURCES
from docs.research.tools.free_ppo_v1.evaluate import economic_audit as original_economic_audit, checkpoint_record

SCHEMA = "wholeplant-ppo-v1-evaluation"
ARMS = ("restricted-instant", "shared-instant", "restricted-dynamic", "shared-dynamic")
BASELINES = ("daylight", "storage", "storage-persistence")
PROCESSES = ("electrolyser", "dac", "methanator")


def environment():
    from docs.research.tools.wholeplant_ppo_v1 import environment as module
    return module


def sources():
    paths = {Path(row["path"]) for row in environment().source_records()}
    paths.update(Path(__file__).parent / name for name in ("evaluate.py", "analyze.py", "baselines.py", "training.py", "physics.py"))
    paths.update(Path(__file__).parent.parent / "free_ppo_v1" / name for name in ("evaluate.py", "analyze.py", "train.py"))
    protocol = ROOT / "docs/research/WHOLEPLANT_PPO_V1_PROTOCOL.md"
    if protocol.exists():
        paths.add(protocol)
    return [file_pin(path) for path in sorted(paths)]


def freeze_evaluation(output, *, controllers, conditions, panels, gate_path=None, description=None):
    """Pin all evaluated identities and panel membership before running any case."""
    controller_map = {}
    for original in controllers:
        item = dict(original)
        label = item["label"]
        if label in controller_map or item["kind"] not in ("ppo", "baseline"):
            raise ValueError("Duplicate or invalid controller")
        if item["kind"] == "ppo":
            item["checkpoint"] = checkpoint_record(item["checkpoint"])
            metadata = json.loads(Path(item["checkpoint"]["metadata"]["path"]).read_text())
            item["checkpoint_timesteps"] = metadata["num_timesteps"]
            item.setdefault("phase", "initial" if metadata["num_timesteps"] == 0 else "final")
            item["checkpoint_environment_identity"] = metadata["environment_identity"]
            if metadata["environment_identity"] != environment().environment_identity():
                raise ValueError("PPO checkpoint does not belong to the new whole-plant environment")
            if item.get("optimizer_seed", metadata["optimizer_seed"]) != metadata["optimizer_seed"]:
                raise ValueError("PPO optimizer label differs from checkpoint metadata")
            item["optimizer_seed"] = metadata["optimizer_seed"]
            if item["phase"] != ("initial" if metadata["num_timesteps"] == 0 else "final"):
                raise ValueError("Initial/final checkpoint phase is inconsistent")
        elif item.get("baseline") not in BASELINES or label != item.get("baseline"):
            raise ValueError("Unknown or ambiguously labelled baseline")
        controller_map[label] = item
    condition_map = {}
    for original in conditions:
        item = {"label": original} if isinstance(original, str) else dict(original)
        label = item.pop("label", item.pop("arm", None))
        if label not in ARMS or label in condition_map:
            raise ValueError("Duplicate or unknown whole-plant arm")
        expected = {"arm": label, "delivery_rate_kg_h": 30.0, "opening_battery": "sampled",
                    "shared_bus": label.startswith("shared"),
                    "downstream_dynamics": label.endswith("dynamic")}
        if any(key not in expected or expected[key] != value for key, value in item.items()):
            raise ValueError("Condition changes a fixed whole-plant assumption")
        condition_map[label] = expected
    selections = []
    seen = set()
    for original in panels:
        item = dict(original)
        item["condition"] = item.pop("arm", item.get("condition"))
        key = item["split"], item["controller"], item["condition"]
        if key in seen or key[1] not in controller_map or key[2] not in condition_map or key[0] not in ("train", "development", "held_out"):
            raise ValueError("Duplicate or invalid panel")
        chosen_controller = controller_map[key[1]]
        if chosen_controller["kind"] == "ppo" and key[2] != "shared-dynamic":
            raise ValueError("Fresh PPO is prescribed only on shared-dynamic")
        if key[0] == "held_out" and chosen_controller.get("phase") == "initial":
            raise ValueError("Initial-policy learning checks use development only")
        if chosen_controller.get("baseline") == "storage-persistence" and key[0] != "train" and (key[0], key[2]) != ("held_out", "shared-dynamic"):
            raise ValueError("Persistence is prescribed only on held-out shared-dynamic")
        seen.add(key)
        expected = list(environment().DEVELOPMENT_SEEDS if key[0] == "development" else environment().HELD_OUT_SEEDS) if key[0] != "train" else item.get("seeds")
        if item.get("seeds", expected) != expected or not expected or len(set(expected)) != len(expected):
            raise ValueError("Evaluation panels must contain the complete prescribed split")
        if key[0] == "train" and any(isinstance(seed, bool) or not isinstance(seed, int) or not environment().TRAIN_BASE <= seed < environment().TRAIN_STOP for seed in expected):
            raise ValueError("Engineering fixtures require admitted training seeds")
        item["seeds"] = expected
        selections.append(item)
    if not selections:
        raise ValueError("No panels selected")
    payload = {"schema": SCHEMA + "-freeze", "created_at": datetime.now(timezone.utc).isoformat(),
               "description": description, "controllers": controller_map, "conditions": condition_map, "panels": selections,
               "environment_identity_sha256": environment().environment_identity_sha256(),
               "source_identity_sha256": environment().source_identity_sha256(), "sources": sources(),
               "gate": file_pin(gate_path) if gate_path else None, "attempts_per_case": 1,
               "full_trace_rule": "First prescribed scenario per controller-condition panel; every minute audited in every case",
               "bootstrap": {"seed": 20260917, "replicates": 10000, "unit": "paired environmental scenario, conditional on fixed policies"}}
    payload["sha256"] = canonical_sha(payload)
    publish(output, payload)
    return payload


def require_freeze(path):
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != SCHEMA + "-freeze" or payload.get("sha256") != canonical_sha({k: v for k, v in payload.items() if k != "sha256"}):
        raise ValueError("Whole-plant freeze identity differs")
    if payload["sources"] != sources() or payload["environment_identity_sha256"] != environment().environment_identity_sha256() or payload["source_identity_sha256"] != environment().source_identity_sha256():
        raise ValueError("Frozen whole-plant source changed")
    for controller in payload["controllers"].values():
        if controller["kind"] == "ppo":
            for pin in controller["checkpoint"].values():
                if file_pin(pin["path"]) != pin:
                    raise ValueError("Frozen policy, metadata, or normalizer changed")
    if payload["gate"] and file_pin(payload["gate"]["path"]) != payload["gate"]:
        raise ValueError("Frozen evaluation gate changed")
    return payload


def admit(freeze, *, controller, condition, split, seed, gate_path=None, max_decisions=None):
    matching = [p for p in freeze["panels"] if (p["controller"], p["condition"], p["split"]) == (controller, condition, split)]
    if len(matching) != 1 or seed not in matching[0]["seeds"]:
        raise ValueError("Case is outside the frozen panel")
    if max_decisions is not None and (split != "train" or not 1 <= max_decisions <= 672):
        raise ValueError("Short engineering fixtures require training seeds")
    if freeze["gate"] and file_pin(gate_path or freeze["gate"]["path"]) != freeze["gate"]:
        raise ValueError("Case gate differs from frozen gate")
    if split != "train":
        selected_gate = gate_path or (freeze["gate"]["path"] if freeze["gate"] else None)
        environment()._admit(split, Path(selected_gate) if selected_gate else None)
    return matching[0]


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
            outage = any(f["start_s"] <= (step-1)*cfg.dt_s < f["start_s"]+f["duration_s"] for f in schedule)
            for unit, health in s["health"].items():
                if health["offline"] != (outage if unit == "dac" else False) or health["degradation"] != 0:
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


class OperatingAudit:
    """Integrate actual delivered power and accepted process services."""
    def __init__(self, env):
        self.dt_h, self.previous = env.config.dt_h, None
        self.totals = {name: 0.0 for name in ("solar_available_kwh", "battery_charge_kwh", "battery_discharge_kwh",
            "zero_solar_ch4_produced_kg", "solar_present_ch4_produced_kg", "zero_solar_battery_discharge_kwh",
            "zero_solar_h2_produced_kg", "zero_solar_co2_captured_kg")}
        for unit in PROCESSES:
            self.totals.update({unit+suffix: 0.0 for suffix in ("_kwh", "_on_minutes", "_battery_kwh")})
            self.totals["zero_solar_"+unit+"_kwh"] = 0.0
        for unit in ("dac", "methanator"):
            self.totals.update({unit+suffix: 0.0 for suffix in ("_startup_kwh", "_standby_kwh", "_productive_kwh", "_heat_loss_kwh", "_ready_minutes", "_starts", "_cold_starts")})

    def add(self, state):
        s = state.to_dict() if hasattr(state, "to_dict") else state
        if self.previous is not None:
            dt, dark = self.dt_h, s["solar_available_kw"] <= 1e-8
            self.totals["solar_available_kwh"] += s["solar_available_kw"]*dt
            self.totals["battery_charge_kwh"] += s["solar_to_battery_kw"]*dt
            discharge = sum(s["battery_to_"+u+"_kw"] for u in PROCESSES)
            self.totals["battery_discharge_kwh"] += discharge*dt
            if dark:
                self.totals["zero_solar_battery_discharge_kwh"] += discharge*dt
            for unit in PROCESSES:
                self.totals[unit+"_kwh"] += s[unit+"_power_kw"]*dt
                self.totals[unit+"_on_minutes"] += (s[unit+"_power_kw"] > 1e-8)*dt*60
                self.totals[unit+"_battery_kwh"] += s["battery_to_"+unit+"_kw"]*dt
                if dark:
                    self.totals["zero_solar_"+unit+"_kwh"] += s[unit+"_power_kw"]*dt
            if dark:
                self.totals["zero_solar_h2_produced_kg"] += s["h2_produced_total_kg"]-self.previous["h2_produced_total_kg"]
                self.totals["zero_solar_co2_captured_kg"] += s["co2_captured_total_kg"]-self.previous["co2_captured_total_kg"]
            methane = s["ch4_produced_total_kg"]-self.previous["ch4_produced_total_kg"]
            self.totals[("zero_solar" if dark else "solar_present")+"_ch4_produced_kg"] += methane
            for unit in ("dac", "methanator"):
                for key, field in (("startup", "startup_power"), ("standby", "standby_power"), ("productive", "productive_power"), ("heat_loss", "heat_loss")):
                    self.totals[unit+"_"+key+"_kwh"] += s[unit+"_"+field+"_kw"]*dt
                self.totals[unit+"_ready_minutes"] += (s[unit+"_readiness_fraction"] >= 1-1e-9)*dt*60
                for counter in ("starts", "cold_starts"):
                    self.totals[unit+"_"+counter] += s[unit+"_"+counter]-self.previous[unit+"_"+counter]
        self.previous = s


def economic_audit(env, initial, final, reward_sum, operating=None):
    result = original_economic_audit(env, initial, final, reward_sum)
    if operating is not None:
        totals = operating.totals if hasattr(operating, "totals") else operating
        economic = env.economic_metrics.to_dict()
        throughput = totals["battery_charge_kwh"] + totals["battery_discharge_kwh"]
        assumptions = env.economic_assumptions
        price = assumptions.battery_replacement_eur_per_kwh/(2*assumptions.battery_full_cycle_life)
        result["residuals"].update(
            whole_plant_battery_throughput_kwh=economic["battery_throughput_kwh"]-throughput,
            whole_plant_battery_cycles=economic["battery_equivalent_full_cycles"]-throughput/(2*env.config.battery.capacity_kwh),
            whole_plant_battery_wear_eur=economic["components_eur"]["battery_throughput_reserve"]+throughput*price)
        result["all_destinations_independently_reconciled"] = True
        result["passed"] = all(math.isfinite(value) and abs(value) <= 1e-6 for value in result["residuals"].values())
    else:
        result["all_destinations_independently_reconciled"] = False
    return result


def empty_case(*, seed, controller, condition, split, freeze, failure, elapsed=0):
    spec = freeze.get("controllers", {}).get(controller, {})
    return {"schema": SCHEMA + "-case", "seed": seed, "controller": controller, "condition": condition, "arm": condition, "variant": freeze["conditions"][condition], "split": split,
            "controller_kind": spec.get("kind"), "phase": spec.get("phase"), "optimizer_seed": spec.get("optimizer_seed"),
            "checkpoint_timesteps": spec.get("checkpoint_timesteps"), "checkpoint": spec.get("checkpoint"),
            "evaluation_freeze_sha256": freeze["sha256"], "execution_complete": False, "every_delivery_passed": False,
            "physical_invariants_passed": False, "delivery_and_physics_passed": False, "failure": failure,
            "decisions": 0, "physical_steps": 0, "wall_time_s": elapsed, "delivery_settlements": 0, "maximum_delivery_unmet_kg": None,
            "daily_returns_met": 0, "stock_deficits": {r: {"observed_daily_deficits": [], "sum": 0, "max": 0} for r in RESOURCES},
            "physical_metrics": {}, "economic_metrics": {}, "operating_metrics": {}, "minute_audit": {"all_physical_invariants_passed": False, "states_checked": 0},
            "economic_audit": {"passed": False}, "recording": None}


def write_record(output, record):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(gzip.compress(json.dumps(record, separators=(",", ":"), allow_nan=False).encode(), mtime=0))
    return file_pin(output)


def audit_intervention(env, action, info, trace):
    requested, _ = environment().decode_action(action, max_discharge_kw=env.config.battery.max_discharge_kw)
    request = requested.to_dict()
    expected = dict(request)
    if not env.shared_bus:
        expected.update(battery_dac_support_kw=0.0, battery_methanator_support_kw=0.0)
    witness = info["battery_intervention"]
    if witness["mode"] != env.battery_mode or witness["requested_action"] != request or witness["applied_action"] != expected or info["requested"]["action"] != request or info["direct_control"]["intervention_applied_action"] != expected:
        raise ValueError("Requested/applied arm intervention evidence differs")
    if len(env.last_direct_minute_actions or ()) != 15:
        raise ValueError("Missing minute action witnesses")
    totals = {k: 0.0 for k in ("battery", "battery_support", *PROCESSES)}
    for minute, state in zip(env.last_direct_minute_actions, trace[1:]):
        s = state.to_dict() if hasattr(state, "to_dict") else state
        if minute["requested_action"] != request or minute["intervention_applied_action"] != expected or set(s["action"]) != set(expected) or any(abs(s["action"][k]-v) > 1e-9 for k,v in expected.items()):
            raise ValueError("Minute-level experimental action differs")
        if not env.shared_bus and (abs(s["battery_to_dac_kw"]) > 1e-8 or abs(s["battery_to_methanator_kw"]) > 1e-8):
            raise ValueError("Restricted arm supplied a downstream battery route")
        totals["battery"] += s["solar_to_battery_kw"]*env.config.dt_h
        totals["battery_support"] += sum(s["battery_to_"+u+"_kw"] for u in PROCESSES)*env.config.dt_h
        for unit in PROCESSES:
            totals[unit] += s[unit+"_power_kw"]*env.config.dt_h
    applied = info["direct_control"]["physical_applied_energy_kwh"]
    if any(abs(applied[key]-value) > 1e-7 for key, value in totals.items()):
        raise ValueError("Interval accounting omits or changes actual process/battery energy")


def evaluate_case(*, freeze_path, gate_path=None, controller, split, seed, output, condition=None, arm=None, retain_minutes=False, max_decisions=None, deadline_utc=None):
    condition = condition or arm
    if arm is not None and arm != condition:
        raise ValueError("Arm and condition disagree")
    import numpy as np
    freeze = require_freeze(freeze_path)
    panel = admit(freeze, controller=controller, condition=condition, split=split, seed=seed, gate_path=gate_path, max_decisions=max_decisions)
    if retain_minutes and seed != panel["seeds"][0]:
        raise ValueError("Full trace selection was fixed to the first scenario")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    deadline = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00")) if deadline_utc else None
    if deadline and deadline.tzinfo is None:
        raise ValueError("Deadline must include timezone")
    spec = freeze["controllers"][controller]
    env = model = audit = operating = None
    failure = None
    commands, states, latencies = [], [], []
    reward_sum = 0.0
    terminated = truncated = False
    clipped_means = 0
    input_digest = hashlib.sha256()
    try:
        env = environment().WholePlantEnv(arm=condition, split=split,
            gate_path=gate_path or (freeze["gate"]["path"] if freeze["gate"] else None), record_minutes=True)
        obs, _ = env.reset(seed=seed)
        initial = env.simulator.state.to_dict()
        audit, operating = MinuteAudit(env), OperatingAudit(env)
        audit.add(initial); operating.add(initial)
        if retain_minutes:
            states.append(initial)
        if spec["kind"] == "ppo":
            from docs.research.tools.wholeplant_ppo_v1 import training as train
            train.configure_torch()
            model = train.load_checkpoint(Path(spec["checkpoint"]["checkpoint"]["path"]))
            if model.research_identity != spec["checkpoint_environment_identity"] or model.num_timesteps != spec["checkpoint_timesteps"]:
                raise ValueError("Loaded policy differs from frozen identity")
        else:
            from docs.research.tools.wholeplant_ppo_v1.baselines import baseline
            model = baseline(spec["baseline"], env)
        for index in range(max_decisions or 672):
            if deadline and datetime.now(timezone.utc) >= deadline:
                raise TimeoutError("Campaign UTC deadline reached; partial attempt retained")
            if np.asarray(obs).shape != (149,) or not np.array_equal(obs, env.current_policy_observation):
                raise ValueError("Policy input differs from actual current observation")
            t = time.perf_counter()
            action, _ = model.predict(obs.copy(), deterministic=True)
            if np.asarray(action).shape != (7,) or not np.isfinite(action).all():
                raise ValueError("Policy produced an invalid seven-action command")
            evidence = {"observation_sha256": canonical_sha(obs.tolist()), "observation": obs.tolist(), "action": np.asarray(action).tolist()}
            if spec["kind"] == "baseline":
                verified_action, baseline_evidence = model.inference(obs.copy())
                if not np.array_equal(action, verified_action) or not np.array_equal(obs, baseline_evidence["actual_observation"]):
                    raise ValueError("Baseline inference evidence differs from actual policy input")
                transformed = np.asarray(baseline_evidence["controller_input"])
                if transformed.shape != (149,) or not np.isfinite(transformed).all():
                    raise ValueError("Invalid transformed baseline input")
                evidence.update(baseline_inference=baseline_evidence, controller_input_sha256=canonical_sha(transformed.tolist()))
            if spec["kind"] == "ppo":
                details = model.deterministic_details(obs.copy())
                if not np.array_equal(action, details["clipped_action"]):
                    raise ValueError("Recorded inference differs from prediction")
                mean = np.asarray(details["unclipped_mean"])
                clipped_means += int(np.any(mean != action))
                evidence.update(normalized_observation_sha256=canonical_sha(np.asarray(details["normalized_input"]).tolist()), raw_mean=mean.tolist())
                evidence["normalized_observation"] = np.asarray(details["normalized_input"]).tolist()
            if retain_minutes:
                evidence["observation"] = obs.tolist()
            latencies.append(time.perf_counter() - t)
            input_digest.update(json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            trace = env.last_minute_truth_states
            if trace is None or len(trace) != 16:
                raise ValueError("Missing exact 15-minute physical trace")
            audit_intervention(env, action, info, trace)
            for state in trace[1:]:
                audit.add(state); operating.add(state)
                if retain_minutes:
                    states.append(state.to_dict())
            commands.append({"decision_index": index, "action": np.asarray(action).tolist(), "evidence": evidence,
                             "reward_eur": reward, "direct_control": info["direct_control"], "battery_intervention": info.get("battery_intervention"),
                             "state": {k: getattr(env.simulator.state, k) for k in ("battery_energy_kwh", "battery_soc", "h2_level_kg", "co2_level_kg", "ch4_level_kg", "solar_available_kw", "electrolyser_power_kw", "dac_power_kw", "methanator_power_kw", "dac_readiness_kwh", "methanator_readiness_kwh")}})
            if terminated or truncated:
                break
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    try:
        if env is None or audit is None:
            row = empty_case(seed=seed, controller=controller, condition=condition, split=split, freeze=freeze,
                             failure=failure or {"type": "SetupFailure"}, elapsed=time.monotonic() - started)
            row.pop("recording")
            return {**row, "recording": write_record(output, {"schema": SCHEMA + "-recording", "summary": row, "decision_trace": commands, "minute_truth_states": states or None})}
        final = env.simulator.state.to_dict()
        physical, economic = env.metrics.to_dict(), env.economic_metrics.to_dict()
        delivery = env.simulator.delivery_audit()
        cycles = [s.to_dict() for s in env.cycle_settlements]
        complete = failure is None and terminated and not truncated and env.decisions_taken == 672 and env.physics_steps_taken == 10080 and audit.count == 10081
        audit_result = audit.result()
        delivery_pass = complete and len(delivery["settlements"]) == 28 and all(r["unmet_kg"] <= 1e-9 for r in delivery["settlements"])
        physical_pass = complete and audit_result["all_physical_invariants_passed"]
        manifest = env.audit_manifest()
        paired_source = env.paired_source_manifest()
        row = {"schema": SCHEMA + "-case", "seed": seed, "controller": controller, "condition": condition, "split": split,
            "arm": condition, "variant": freeze["conditions"][condition], "controller_kind": spec["kind"], "training_family": spec.get("training_family"), "phase": spec.get("phase"),
            "optimizer_seed": spec.get("optimizer_seed"), "checkpoint_timesteps": spec.get("checkpoint_timesteps"), "checkpoint": spec.get("checkpoint"),
            "evaluation_freeze_sha256": freeze["sha256"], "execution_complete": bool(complete), "every_delivery_passed": bool(delivery_pass),
            "physical_invariants_passed": bool(physical_pass), "delivery_and_physics_passed": bool(delivery_pass and physical_pass),
            "failure": failure, "decisions": env.decisions_taken, "physical_steps": env.physics_steps_taken, "wall_time_s": time.monotonic() - started,
            "delivery_settlements": len(delivery["settlements"]), "maximum_delivery_unmet_kg": max((r["unmet_kg"] for r in delivery["settlements"]), default=0),
            "daily_returns_met": sum(c["feasible"] for c in cycles),
            "stock_deficits": {r: {"observed_daily_deficits": [c["deficits"][r] for c in cycles], "sum": sum(c["deficits"][r] for c in cycles), "max": max((c["deficits"][r] for c in cycles), default=0)} for r in RESOURCES},
            "physical_metrics": physical, "economic_metrics": economic, "operating_metrics": operating.totals, "minute_audit": audit_result,
            "economic_audit": economic_audit(env, initial, final, reward_sum, operating), "initial_state": initial, "final_state": final,
            "paired_source_sha256": paired_source["paired_source_sha256"], "paired_source_manifest": paired_source,
            "exogenous_episode_sha256": manifest["beta_exogenous_episode_sha256"], "manifest_sha256": manifest["manifest_sha256"],
            "controller_latency_s": {"mean": statistics.mean(latencies) if latencies else None, "max": max(latencies, default=0), "calls": len(latencies)},
            "inference": {"verified_decisions": len(commands), "actual_observation_inputs": True, "raw_mean_clipped_decisions": clipped_means, "witness_sha256": input_digest.hexdigest()},
            "baseline_metadata": model.metadata() if spec["kind"] == "baseline" and model is not None else None,
            "gate": file_pin(gate_path or freeze["gate"]["path"]) if gate_path or freeze["gate"] else None,
            "retained_minute_truth_sha256": canonical_sha(states) if states else None}
        record = {"schema": SCHEMA + "-recording", "summary": row, "decision_trace": commands, "cycle_settlements": cycles, "delivery": delivery,
                  "manifest": manifest, "minute_truth_states": states or None,
                  "trace_selection": "First prescribed scenario" if retain_minutes else "All minute states audited and streaming-hashed; full raw minute trace omitted"}
        return {**row, "recording": write_record(output, record)}
    finally:
        if env is not None:
            env.close()
        if model is not None and hasattr(model, "close"):
            model.close()


def freeze_manifest(spec, output=None):
    """Freeze a JSON specification; this function never opens an evaluation case."""
    spec = json.loads(Path(spec).read_text()) if isinstance(spec, (str, Path)) else dict(spec)
    destination = output or spec.pop("output", None)
    if destination is None:
        raise ValueError("Freeze output is required")
    return freeze_evaluation(destination, controllers=spec["controllers"],
        conditions=spec.get("conditions", spec.get("arms", ARMS)), panels=spec["panels"],
        gate_path=spec.get("gate_path", spec.get("gate")), description=spec.get("description"))


def run_panel(*, freeze_path, controller, split, output, arm=None, condition=None,
              gate_path=None, workers=3, deadline_utc=None, max_decisions=None):
    """One attempt for every frozen case, with at most three child processes.

    Panels are externally orchestrated. Caller must enforce the campaign-wide
    worker ceiling when other panels or training jobs are active.
    """
    condition = condition or arm
    if workers not in (1, 2, 3):
        raise ValueError("Local evaluation worker limit is three")
    frozen = require_freeze(freeze_path)
    selected = [p for p in frozen["panels"] if (p["controller"],p["condition"],p["split"]) == (controller,condition,split)]
    if len(selected) != 1:
        raise ValueError("Panel is outside the frozen specification")
    panel = selected[0]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/"cases").mkdir()
    started = time.monotonic()
    def attempt(seed):
        result_path = output/"cases"/f"{seed}.json"
        trace_path = output/"cases"/f"{seed}.json.gz"
        command = [sys.executable, str(Path(__file__).resolve()), "case", "--freeze", str(Path(freeze_path).resolve()),
            "--controller", controller, "--arm", condition, "--split", split, "--seed", str(seed),
            "--output", str(trace_path), "--result", str(result_path)]
        if gate_path:
            command += ["--gate", str(Path(gate_path).resolve())]
        if seed == panel["seeds"][0]:
            command += ["--retain-minutes"]
        if deadline_utc:
            command += ["--deadline-utc", deadline_utc]
        if max_decisions is not None:
            command += ["--max-decisions", str(max_decisions)]
        t = time.monotonic()
        expired = deadline_utc and datetime.now(timezone.utc) >= datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
        if expired:
            failure = {"type": "DeadlineBeforeStart", "message": "Not launched after campaign deadline; prescribed case retained"}
            return_code = None
        else:
            with (output/"cases"/f"{seed}.log").open("x") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                remaining = max(0.01, (datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))-datetime.now(timezone.utc)).total_seconds()) if deadline_utc else None
                try:
                    process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
            return_code = process.returncode
            failure = {"type": "CaseProcessFailure", "message": f"Child exit {return_code}; no compact result"}
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if return_code != 0:
                raise RuntimeError("Case wrote a result but did not close successfully; retain artifacts for inspection")
            return result
        result = empty_case(seed=seed, controller=controller, condition=condition, split=split,
            freeze=frozen, failure=failure, elapsed=time.monotonic()-t)
        publish(result_path, result)
        return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(attempt, seed): seed for seed in panel["seeds"]}
        rows = [future.result() for future in as_completed(futures)]
    rows.sort(key=lambda r: r["seed"])
    unchanged = require_freeze(freeze_path)["sha256"] == frozen["sha256"]
    summary = {"schema": SCHEMA+"-panel", "split": split, "controller": controller,
        "condition": condition, "arm": condition, "evaluation_freeze": file_pin(freeze_path),
        "evaluation_freeze_sha256": frozen["sha256"], "source_files_unchanged_during_panel": unchanged,
        "requested_seeds": panel["seeds"], "attempts_per_case": 1, "runs": rows,
        "wall_time_s": time.monotonic()-started,
        "execution_complete_count": sum(r["execution_complete"] for r in rows),
        "delivery_success_count": sum(r["every_delivery_passed"] for r in rows),
        "all_case_audits_passed": all(r["physical_invariants_passed"] and r["economic_audit"]["passed"] for r in rows)}
    publish(output/"summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="operation", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--spec", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    for name in ("case", "panel"):
        q = commands.add_parser(name)
        q.add_argument("--freeze", type=Path, required=True)
        q.add_argument("--gate", type=Path)
        q.add_argument("--controller", required=True)
        q.add_argument("--arm", "--condition", dest="condition", choices=ARMS, required=True)
        q.add_argument("--split", choices=("train", "development", "held_out"), required=True)
        q.add_argument("--output", type=Path, required=True)
        q.add_argument("--max-decisions", type=int)
        q.add_argument("--deadline-utc")
        if name == "case":
            q.add_argument("--seed", type=int, required=True)
            q.add_argument("--result", type=Path, required=True)
            q.add_argument("--retain-minutes", action="store_true")
        else:
            q.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    if args.operation == "freeze":
        result = freeze_manifest(args.spec, args.output)
        print(json.dumps({"freeze": str(args.output), "sha256": result["sha256"]}), flush=True)
        return
    def interrupted(signum, frame):
        raise TimeoutError(f"Parent watchdog signal {signum}; partial attempt retained")
    signal.signal(signal.SIGTERM, interrupted)
    common = dict(freeze_path=args.freeze, gate_path=args.gate, controller=args.controller,
                  condition=args.condition, split=args.split, output=args.output,
                  max_decisions=args.max_decisions, deadline_utc=args.deadline_utc)
    if args.operation == "panel":
        result = run_panel(**common, workers=args.workers)
        print(json.dumps({"panel": str(args.output), "cases": len(result["runs"]),
                          "complete": result["execution_complete_count"]}), flush=True)
    else:
        result = evaluate_case(**common, seed=args.seed, retain_minutes=args.retain_minutes)
        publish(args.result, result)
        print(json.dumps({"controller": args.controller, "arm": args.condition, "seed": args.seed,
                          "complete": result["execution_complete"], "failure": result["failure"]}), flush=True)


if __name__ == "__main__":
    main()


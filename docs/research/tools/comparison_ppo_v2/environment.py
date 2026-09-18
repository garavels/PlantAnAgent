"""Paired demand and battery experiments around the unchanged direct-control plant.

The only interventions are requested delivery quantities, initial battery charge,
and optional removal of BOTH battery charge and battery support channels. Sampling
always occurs under the original source configuration before these interventions.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from numbers import Integral
from pathlib import Path
from typing import Any

from docs.research.tools.free_ppo_v1 import environment as parent
from docs.research.tools.free_ppo_v1.environment import (
    Action, Box, CommitmentExposureV2, ControlStateV2, ControllerNeutralTrainingConfigV1,
    CRITICAL, DirectPlantSimulator, IntervalOutcome, IntervalPowerAccounting,
    IntervalTelemetryAccumulator, OFFLINE, PlantCyclicMarketEnvV2Beta,
    canonical, contract_digest, decode_action, action_to_vector, diagnostic_reward, legacy, np,
)

ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = parent.SOURCE_ROOT
ENVIRONMENT_ID = "plant-comparison-direct-control-v2"
SCHEMA = "comparison-ppo-direct-environment-v2"
TRAIN_BASE = 20_000_000
TRAIN_STOP = 20_100_000
DEVELOPMENT_SEEDS = tuple(range(20_100_000, 20_100_008))
HELD_OUT_SEEDS = tuple(range(20_200_000, 20_200_032))
ACTION_ORDER = parent.ACTION_ORDER


def environment_identity() -> dict[str, Any]:
    identity = parent.environment_identity()
    identity.update(schema=SCHEMA, environment_id=ENVIRONMENT_ID,
        parent_environment_identity_sha256=parent.environment_identity_sha256(),
        delivery_rate_kg_h=[25.0, 30.0],
        variants={"opening_battery": ["sampled", "floor"], "battery_mode": ["enabled", "disabled"]},
        intervention_semantics={
            "floor": "Replace sampled battery SOC by physical 0.05 before simulator load, observation and economic/cycle references; preserve gas stocks and all exogenous tapes.",
            "disabled": "Set decoded battery solar share and support kW to zero; do not reallocate that solar share; preserve and distinguish original policy request, intervention-applied action and actual physical flows.",
            "delivery": "Replace only each delivery commitment quantity; preserve original physical inventory sampling and deadline schedule.",
        },
        splits={"train": {"first": TRAIN_BASE, "stop_exclusive": TRAIN_STOP},
                "development": list(DEVELOPMENT_SEEDS), "held_out": list(HELD_OUT_SEEDS)})
    return identity


def environment_identity_sha256() -> str:
    return hashlib.sha256(canonical(environment_identity())).hexdigest()


def source_records() -> list[dict[str, Any]]:
    paths = {Path(item["path"]).resolve() for item in parent.source_records()}
    paths.update({Path(__file__).resolve(), Path(__file__).with_name("__init__.py").resolve(),
                  ROOT / "docs/research/COMPARISON_PPO_V2_PROTOCOL.md"})
    return [{"path": str(path), "bytes": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(paths)]


def source_identity_sha256() -> str:
    records = [{**row, "path": Path(row["path"]).relative_to(ROOT).as_posix()}
               for row in source_records()]
    return hashlib.sha256(canonical(records)).hexdigest()


def _admit(split: str, gate_path: Path | None) -> None:
    if split == "train":
        return
    if split not in {"development", "held_out"}:
        raise ValueError(f"Unknown comparison split: {split}")
    if gate_path is None:
        raise ValueError("Development and held-out access require an external frozen gate")
    gate = json.loads(gate_path.read_text())
    if (gate.get("schema") != "comparison-ppo-v2-evaluation-gate"
        or gate.get("environment_identity_sha256") != environment_identity_sha256()
        or gate.get("source_identity_sha256") != source_identity_sha256()
        or split not in gate.get("allowed_splits", [])
        or (split == "held_out" and gate.get("endpoints_frozen") is not True)):
        raise ValueError("External gate does not admit this source, environment and split")


class ComparisonEnv(parent.DirectPlantEnv):
    """Same seven-day dynamics, direct five controls, observation and EUR reward."""

    def __init__(self, delivery_rate_kg_h=25.0, opening_battery="sampled",
                 battery_mode="enabled", split="train", record_minutes=False, gate_path=None):
        if isinstance(delivery_rate_kg_h, bool) or delivery_rate_kg_h not in (25.0, 30.0):
            raise ValueError("Declared delivery rates are 25 and 30 kg/h")
        if opening_battery not in {"sampled", "floor"} or battery_mode not in {"enabled", "disabled"}:
            raise ValueError("Unknown opening-battery or battery-channel intervention")
        self.variant = {"delivery_rate_kg_h": float(delivery_rate_kg_h),
                        "opening_battery": opening_battery, "battery_mode": battery_mode}
        self.opening_battery, self.battery_mode = opening_battery, battery_mode
        self.gate_path = None if gate_path is None else Path(gate_path)
        _admit(split, self.gate_path)
        self._gate_sha256 = None if self.gate_path is None else hashlib.sha256(self.gate_path.read_bytes()).hexdigest()
        self.split, self.record_minutes = split, bool(record_minutes)
        self.allowed_seeds = range(TRAIN_BASE, TRAIN_STOP) if split == "train" else (
            DEVELOPMENT_SEEDS if split == "development" else HELD_OUT_SEEDS)
        physical = legacy.load_benchmark_contract()
        cfg = ControllerNeutralTrainingConfigV1.from_dict(physical["source_template"])
        cfg = replace(cfg, constraints=replace(cfg.constraints, battery_min_soc=cfg.plant.battery.min_soc))
        self.source_config = cfg
        self.variant_rate_kg_h = float(delivery_rate_kg_h)
        self.electricity_model = legacy.ElectricityModelV1(**physical["electricity"])
        public_episode = replace(cfg.episode, delivery_rate_kg_h=self.variant_rate_kg_h,
            ch4_coverage_h=tuple(x * cfg.episode.delivery_rate_kg_h / self.variant_rate_kg_h
                                for x in cfg.episode.ch4_coverage_h))
        self.benchmark_contract = {"schema": SCHEMA, "contract_sha256": environment_identity_sha256(),
            "parent_benchmark_sha256": physical["contract_sha256"], "splits": environment_identity()["splits"]}
        PlantCyclicMarketEnvV2Beta.__init__(self, config=cfg.plant, episode=public_episode,
            constraints=cfg.constraints, weights=cfg.diagnostic_reward,
            economic_assumptions=cfg.economic_assumptions, beta_task=cfg.beta_task)
        self.simulator = DirectPlantSimulator(self.config, electrolyser_commitment=self.episode_config.electrolyser_commitment)
        self.action_space = Box(-1, 1, shape=(5,), dtype=np.float32)
        payload = {"schema": "comparison-direct-checkpoint-v2", "interface": environment_identity(),
            "variant": self.variant, "plant_config": cfg.plant.to_dict(),
            "episode_config": public_episode.to_dict(), "constraints": cfg.constraints.to_dict(),
            "economic_assumptions": cfg.economic_assumptions.to_dict()}
        self._checkpoint_contract = {"checkpoint_contract_digest": contract_digest(payload), **payload}
        self._last_benchmark_seed = None
        self.last_direct_minute_actions = None

    def _sample_episode(self, seed):
        episode = legacy.BenchmarkEnv._sample_episode(self, seed)
        if self.opening_battery == "floor":
            initial = replace(episode.scenario.initial, battery_soc=self.config.battery.min_soc)
            episode = replace(episode, scenario=replace(episode.scenario, initial=initial))
        return episode

    def reset(self, *, seed=None, options=None):
        if self.split != "train" and hashlib.sha256(self.gate_path.read_bytes()).hexdigest() != self._gate_sha256:
            raise ValueError("External evaluation gate changed after environment admission")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
            raise ValueError("Seed must be an admitted integer")
        result = legacy.BenchmarkEnv.reset(self, seed=seed, options=options)
        self.last_direct_minute_actions = None
        self.frozen_source.source_pin = {"path": str(Path(__file__).resolve()),
                                         "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        self.frozen_source.validation.update(
            source_initial_gas_preserved=all(getattr(self._episode.scenario.initial, key)
                == getattr(self._original_sampled_episode.scenario.initial, key)
                for key in ("h2_kg", "co2_kg", "ch4_kg")),
            battery_initialization_intervention=self.opening_battery,
            paired_source_sha256=self.paired_source_manifest()["paired_source_sha256"])
        return result

    def _identify(self, info):
        info = legacy.BenchmarkEnv._identify(self, info)
        info.update(environment_id=ENVIRONMENT_ID, environment_contract_digest=environment_identity_sha256(),
            stage="comparison-ppo-v2", variant=deepcopy(self.variant),
            daily_return_constraints_in_learning_objective=False,
            operator_reserve_equals_physical_floor=True)
        return info

    def public_control_input(self):
        """Detached causal public input also used to construct the 135-vector."""
        return self.controller_observation

    def paired_source_manifest(self):
        """PRIVATE audit data; never provided to a policy or baseline."""
        if self._episode is None:
            raise RuntimeError("Reset before reading paired source evidence")
        source = self._original_sampled_episode
        payload = {"schema": "comparison-paired-exogenous-v2", "seed": source.seed,
            "weather_episode_sha256": source.weather.episode_sha256,
            "market_episode_sha256": self.market_episode.episode_sha256,
            "disturbance_episode_sha256": self.disturbance_episode.episode_sha256,
            "original_initial": source.scenario.initial.to_dict(),
            "start_hour": source.scenario.start_hour,
            "delivery_due_steps": [item.due_step for item in source.commitments],
            "physics_steps": source.physics_steps, "decision_stride": source.decision_stride}
        return {"paired_source_sha256": hashlib.sha256(canonical(payload)).hexdigest(), **payload}

    def audit_manifest(self):
        payload = legacy.BenchmarkEnv.audit_manifest(self)
        payload.pop("manifest_sha256")
        payload.update(environment=environment_identity(), variant=deepcopy(self.variant),
                       paired_source=self.paired_source_manifest())
        accounting = {key: payload.pop(key) for key in ("delivery_accounting", "economic_accounting", "cycle_accounting")}
        return {"manifest_sha256": hashlib.sha256(canonical(payload)).hexdigest(), **payload, **accounting}

    def step(self, action):
        requested, evidence = decode_action(action, max_discharge_kw=self.config.battery.max_discharge_kw)
        applied = (replace(requested, battery_share=0.0, battery_support_kw=0.0)
                   if self.battery_mode == "disabled" else requested)
        evidence.update(battery_intervention=self.battery_mode,
                        intervention_applied_action=applied.to_dict())
        return self._step_direct(requested, evidence, applied=applied)

    def _step_direct(self, requested: Action, evidence: dict, *, applied: Action | None = None):
        applied = requested if applied is None else applied
        if self._episode is None:
            raise RuntimeError("Reset an admitted episode before stepping")
        steps = min(self._episode.decision_stride, self.episode_physics_steps - self.physics_steps_taken)
        if steps <= 0:
            raise RuntimeError("Episode is complete; reset before stepping")
        start = self._capture_economic_step()
        before = self.simulator.state
        commitment_before = self.simulator.electrolyser_commitment
        self._last_setpoint_execution = self._last_process_setpoint_execution = None
        self._last_minute_truth_states = None
        self.last_direct_minute_actions = None
        minutes = [deepcopy(before)] if self.record_minutes else None
        minute_actions = [] if self.record_minutes else None
        power = IntervalPowerAccounting()
        telemetry = IntervalTelemetryAccumulator()
        min_soc = before.battery_soc
        discharged = 0.0
        shutdown_steps = 0
        settlements = []
        requested_kg = delivered_kg = unmet_kg = 0.0
        for ordinal in range(steps):
            # No validate(), target dispatch, commitment projection or allocation
            # policy is called. The simulator owns all physical consequences.
            state = self.simulator.step(applied)
            boundary = self.telemetry
            telemetry.record(boundary, dt_h=self.config.dt_h)
            power.record(requested, applied, after=state, dt_h=self.config.dt_h)
            if minutes is not None:
                minutes.append(deepcopy(state))
                minute_actions.append({"ordinal": ordinal, "requested_action": requested.to_dict(),
                                       "intervention_applied_action": applied.to_dict(),
                                       "physical_flows_kw": {"battery_charge": state.solar_to_battery_kw,
                                           "electrolyser": state.electrolyser_power_kw, "dac": state.dac_power_kw,
                                           "methanator": state.methanator_power_kw,
                                           "battery_support": state.battery_to_electrolyser_kw},
                                       "commitment": self.simulator.electrolyser_commitment.to_dict()})
            delivery = self.simulator.delivery_step
            settlements.extend(delivery.settlements)
            requested_kg += delivery.requested_kg
            delivered_kg += delivery.delivered_kg
            unmet_kg += delivery.unmet_kg
            discharged += max(0.0, -state.battery_power_kw) * self.config.dt_h
            min_soc = min(min_soc, state.battery_soc)
            shutdown_steps += int(state.plant_status in (CRITICAL, OFFLINE))
        after = self.simulator.state
        commitment_after = self.simulator.electrolyser_commitment
        self.physics_steps_taken += steps
        self.decisions_taken += 1
        exposure = CommitmentExposureV2.between(commitment_before, commitment_after, interval_s=steps * self.config.dt_s)
        self.control_state = ControlStateV2(boundary=self.telemetry, previous_interval=telemetry.summary(),
                                            electrolyser_commitment=commitment_after)
        terminated = self.physics_steps_taken >= self.episode_physics_steps
        outcome = IntervalOutcome(hours=steps * self.config.dt_h, physical_steps=steps,
            requested_kg=requested_kg, delivered_kg=delivered_kg, unmet_kg=unmet_kg,
            produced_kg=max(0.0, after.ch4_produced_total_kg - before.ch4_produced_total_kg),
            solar_kwh=max(0.0, after.energy_produced_kwh - before.energy_produced_kwh),
            curtailed_kwh=max(0.0, after.energy_curtailed_kwh - before.energy_curtailed_kwh),
            battery_discharged_kwh=discharged, min_battery_soc=min_soc, shutdown_steps=shutdown_steps,
            corrections=0, guard_intervention_kwh=0.0, dispatch_gap_kwh=power.dispatch_gap_integral_l1_kwh,
            request_application_gap_kwh=power.request_application_integral_l1_kwh,
            out_of_range=evidence["box_clipped"], terminal_ch4_kg=after.ch4_level_kg,
            reference_delivery_kg=self.episode_config.delivery_amount_kg,
            reference_period_h=self.episode_config.delivery_period_h, commitment=exposure)
        diagnostic = diagnostic_reward(outcome, config=self.config, constraints=self.constraints,
                                       terminal=terminated, weights=self.weights)
        self.metrics.record(outcome, diagnostic, settlements, corrected_physics_steps=0,
            battery_charge_applied_kwh=power.applied_kwh["battery"],
            battery_support_applied_kwh=power.applied_kwh["battery_support"],
            electrolyser_applied_energy_kwh=power.applied_kwh["electrolyser"],
            dac_applied_energy_kwh=power.applied_kwh["dac"],
            methanator_applied_energy_kwh=power.applied_kwh["methanator"], terminal_state=after)
        if minutes is not None:
            self._last_minute_truth_states = tuple(minutes)
            self.last_direct_minute_actions = tuple(minute_actions)
        info = {**self._public_info(), "direct_control": {**evidence,
                    "controller_guard": "absent; explicit battery experimental intervention and physical constraints retained",
                    "physical_applied_energy_kwh": dict(power.applied_kwh)},
                "requested": {"action": requested.to_dict()}, "guarded_action": applied.to_dict(),
                "guard": {"mode": "absent", "interventions": 0, "corrected_physics_steps": 0,
                          "out_of_range": evidence["box_clipped"], "intervention_l1_kwh": 0.0},
                "interval_power": power.to_dict(), "adjustments": [],
                "battery_intervention": {"mode": self.battery_mode, "requested_action": requested.to_dict(),
                    "applied_action": applied.to_dict(),
                    "request_intervention_integral_l1_kwh": power.guard_intervention_integral_l1_kwh},
                "control_authority": {"guard_intervention_integral_l1_kwh": 0.0,
                    "dispatch_gap_integral_l1_kwh": power.dispatch_gap_integral_l1_kwh,
                    "request_application_integral_l1_kwh": power.request_application_integral_l1_kwh},
                "delivery": {"commitments_settled": len(settlements), "requested_kg": requested_kg,
                             "delivered_kg": delivered_kg, "unmet_kg": unmet_kg},
                "electrolyser_commitment_interval": exposure.to_dict(), "reward": diagnostic.to_dict(),
                "metrics": self.metrics.to_dict(), "plant_status": after.plant_status}
        return self._finish_economic_step(start, (self._observe(), diagnostic.total, terminated, False, info))


DirectPlantEnv = ComparisonEnv
FreePlantEnv = ComparisonEnv

def make_env(split="train", **kwargs):
    return ComparisonEnv(split=split, **kwargs)

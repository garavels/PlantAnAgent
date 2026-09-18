"""Seven-day, seven-action whole-plant experiment with explicit downstream readiness."""
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
ENVIRONMENT_ID = "plant-wholeplant-direct-control-v1"
SCHEMA = "wholeplant-ppo-environment-v1"
TRAIN_BASE = 30_000_000
TRAIN_STOP = 30_100_000
DEVELOPMENT_SEEDS = tuple(range(30_100_000, 30_100_008))
HELD_OUT_SEEDS = tuple(range(30_200_000, 30_200_032))
from docs.research.tools.wholeplant_ppo_v1.physics import (WholePlantAction as Action,
    WholePlantSimulator, decode_action, action_to_vector)
from plant_backend.rl.v2.accounting import IntervalPowerAccounting as LegacyPowerAccounting

ARMS = ("restricted-instant", "shared-instant", "restricted-dynamic", "shared-dynamic")
ACTION_ORDER = (*parent.ACTION_ORDER, "battery_dac_support", "battery_methanator_support")
EXTRA_OBSERVATIONS = tuple(f"{unit}_{field}" for unit in ("dac", "methanator") for field in
    ("readiness_fraction", "off", "starting", "standby", "producing", "power_fraction")) + (
    "battery_to_dac_fraction", "battery_to_methanator_fraction")


def environment_identity():
    identity = parent.environment_identity()
    identity.update(schema=SCHEMA, environment_id=ENVIRONMENT_ID,
        parent_environment_identity_sha256=parent.environment_identity_sha256(),
        delivery_rate_kg_h=30.0, arms=list(ARMS),
        observation={"size":149, "base_size":135, "extra_order":list(EXTRA_OBSERVATIONS),
                     "source":"causal public boundary measurements and forecasts; downstream readiness measured"},
        action={"size":7,"order":list(ACTION_ORDER),"low":[-1.0]*7,"high":[1.0]*7,
                "hold_seconds":900,"physics_seconds":60,
                "mapping":"four solar shares normalized jointly; three battery power shares normalized jointly against shared 2500kW rating, then physically capped by available stored energy",
                "controller_guard":False},
        readiness={"initial":"cold (zero paid readiness)",
                   "dac":{"capacity_kwh":120.0,"heater_kw":300.0,"standby_kw":60.0},
                   "methanator":{"capacity_kwh":72.0,"heater_kw":90.0,"standby_kw":18.0},
                   "model":"paid aggregate readiness reservoir; no reaction heat recovery; running standby is included by recalibrating productive specific energy to preserve nominal full-load yield"},
        reward="unchanged scalar economic operating EUR; total battery discharge charged across all three destinations; no downstream cycle-wear penalty",
        splits={"train":{"first":TRAIN_BASE,"stop_exclusive":TRAIN_STOP},
                "development":list(DEVELOPMENT_SEEDS),"held_out":list(HELD_OUT_SEEDS)})
    # Remove historical dispatch descriptions that apply only to E support.
    identity.pop("battery_netting",None)
    identity["physical_constraints"] = "original ratings, efficiency, floors, faults and electrolyser dynamics, plus separately declared shared bus and downstream readiness"
    return identity


def environment_identity_sha256() -> str:
    return hashlib.sha256(canonical(environment_identity())).hexdigest()


def source_records() -> list[dict[str, Any]]:
    paths = {Path(item["path"]).resolve() for item in parent.source_records()}
    paths.update({Path(__file__).resolve(), Path(__file__).with_name("__init__.py").resolve(),
                  ROOT / "docs/research/WHOLEPLANT_PPO_V1_PROTOCOL.md"})
    paths.update(Path(__file__).with_name(name) for name in ("physics.py", "baselines.py", "training.py", "evaluate.py", "analyze.py"))
    for package in ("free_ppo_v1", "comparison_ppo_v2", "forecast_ablation_v1"):
        paths.update((Path(__file__).parent.parent / package).glob("*.py"))
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
        raise ValueError(f"Unknown whole-plant split: {split}")
    if gate_path is None:
        raise ValueError("Development and held-out access require an external frozen gate")
    gate = json.loads(gate_path.read_text())
    if (gate.get("schema") != "wholeplant-ppo-v1-evaluation-gate"
        or gate.get("environment_identity_sha256") != environment_identity_sha256()
        or gate.get("source_identity_sha256") != source_identity_sha256()
        or split not in gate.get("allowed_splits", [])
        or (split == "held_out" and gate.get("endpoints_frozen") is not True)):
        raise ValueError("External gate does not admit this source, environment and split")


class IntervalPowerAccounting(LegacyPowerAccounting):
    def record(self, requested, guarded, *, after, dt_h):
        if not np.isfinite(dt_h) or dt_h<=0:
            raise ValueError("Invalid interval")
        bus=float(after.solar_available_kw)
        def command(action):
            return {"battery":action.battery_share*bus,
                "electrolyser":action.electrolyser_share*bus+action.battery_support_kw,
                "dac":action.dac_share*bus+action.battery_dac_support_kw,
                "methanator":action.methanator_share*bus+action.battery_methanator_support_kw,
                "battery_support":action.battery_support_kw+action.battery_dac_support_kw+action.battery_methanator_support_kw}
        rq,gq=command(requested),command(guarded)
        aq={"battery":after.solar_to_battery_kw,"electrolyser":after.electrolyser_power_kw,
            "dac":after.dac_power_kw,"methanator":after.methanator_power_kw,
            "battery_support":after.battery_to_electrolyser_kw+after.battery_to_dac_kw+after.battery_to_methanator_kw}
        if any(not np.isfinite(v) or v < -1e-9 for x in (rq,gq,aq) for v in x.values()):
            raise ValueError("Nonfinite or negative power")
        self.hours+=dt_h
        for key in self.applied_kwh:
            self.requested_kwh[key]+=rq[key]*dt_h
            self.guarded_kwh[key]+=gq[key]*dt_h
            self.applied_kwh[key]+=aq[key]*dt_h
        channels=("battery","electrolyser","dac","methanator")
        self._guard_intervention_l1_kwh+=dt_h*sum(abs(rq[k]-gq[k]) for k in channels)
        self._dispatch_gap_l1_kwh+=dt_h*sum(abs(gq[k]-aq[k]) for k in channels)
        self._request_application_l1_kwh+=dt_h*sum(abs(rq[k]-aq[k]) for k in channels)

    def to_dict(self):
        result=super().to_dict()
        result["battery_support_is_also_in_electrolyser_total"]=False
        result["battery_support_is_sum_of_three_process_supports"]=True
        return result


class WholePlantEnv(parent.DirectPlantEnv):
    """Matched four-arm plant, seven direct controls and unchanged economic objective."""

    def __init__(self, arm="shared-dynamic", split="train", record_minutes=False, gate_path=None):
        if arm not in ARMS:
            raise ValueError(f"Unknown arm: {arm}")
        self.arm=arm
        self.shared_bus=arm.startswith("shared-")
        self.downstream_dynamics=arm.endswith("-dynamic")
        self.variant={"arm":arm,"delivery_rate_kg_h":30.0,"opening_battery":"sampled",
                      "shared_bus":self.shared_bus,"downstream_dynamics":self.downstream_dynamics}
        self.battery_mode="shared" if self.shared_bus else "electrolyser_only"
        self.opening_battery="sampled"
        self.gate_path=None if gate_path is None else Path(gate_path)
        _admit(split,self.gate_path)
        self._gate_sha256=None if self.gate_path is None else hashlib.sha256(self.gate_path.read_bytes()).hexdigest()
        self.split,self.record_minutes=split,bool(record_minutes)
        self.allowed_seeds=range(TRAIN_BASE,TRAIN_STOP) if split=="train" else (DEVELOPMENT_SEEDS if split=="development" else HELD_OUT_SEEDS)
        physical=legacy.load_benchmark_contract()
        cfg=ControllerNeutralTrainingConfigV1.from_dict(physical["source_template"])
        cfg=replace(cfg,constraints=replace(cfg.constraints,battery_min_soc=cfg.plant.battery.min_soc))
        self.source_config=cfg
        self.variant_rate_kg_h=30.0
        self.electricity_model=legacy.ElectricityModelV1(**physical["electricity"])
        public_episode=replace(cfg.episode,delivery_rate_kg_h=30.0,
            ch4_coverage_h=tuple(x*cfg.episode.delivery_rate_kg_h/30.0 for x in cfg.episode.ch4_coverage_h))
        self.benchmark_contract={"schema":SCHEMA,"contract_sha256":environment_identity_sha256(),
            "parent_benchmark_sha256":physical["contract_sha256"],"splits":environment_identity()["splits"]}
        PlantCyclicMarketEnvV2Beta.__init__(self,config=cfg.plant,episode=public_episode,
            constraints=cfg.constraints,weights=cfg.diagnostic_reward,
            economic_assumptions=cfg.economic_assumptions,beta_task=cfg.beta_task)
        self.simulator=WholePlantSimulator(self.config,electrolyser_commitment=self.episode_config.electrolyser_commitment,
            shared_bus=self.shared_bus,downstream_dynamics=self.downstream_dynamics)
        self.action_space=Box(-1,1,shape=(7,),dtype=np.float32)
        self.observation_space=Box(low=np.concatenate((parent.beta_spaces.observation_low(),np.zeros(14,dtype=np.float32))),
            high=np.concatenate((parent.beta_spaces.observation_high(),np.ones(14,dtype=np.float32))),dtype=np.float32)
        payload={"schema":"wholeplant-direct-checkpoint-v1","interface":environment_identity(),
            "variant":self.variant,"plant_config":cfg.plant.to_dict(),"episode_config":public_episode.to_dict(),
            "constraints":cfg.constraints.to_dict(),"economic_assumptions":cfg.economic_assumptions.to_dict()}
        self._checkpoint_contract={"checkpoint_contract_digest":contract_digest(payload),**payload}
        self._last_benchmark_seed=None
        self.last_direct_minute_actions=None

    def _sample_episode(self,seed):
        return legacy.BenchmarkEnv._sample_episode(self,seed)

    def _observe(self):
        physical=self.encoder.encode_controller_observation(self.controller_observation,constraints=self.constraints)
        market=np.asarray(self.current_market_issue.normalized(),dtype=np.float32)
        cycle=np.asarray(self.cycle_observation.normalized(self.config),dtype=np.float32)
        boundary=self.telemetry
        extra=[]
        for unit in ("dac","methanator"):
            mode=getattr(boundary,f"{unit}_mode","off")
            accepted=getattr(boundary,f"{unit}_power_kw")
            extra.extend([getattr(boundary,f"{unit}_readiness_fraction",0.0),
                float(mode in ("off","cooling","fault") or (mode=="instant" and accepted<=1e-9)),
                float(mode=="starting"),float(mode=="standby"),
                float(mode in ("producing","running") or (mode=="instant" and accepted>1e-9)),
                accepted/getattr(self.config,unit).rated_kw])
        extra.extend([getattr(boundary,"battery_to_dac_kw",0.0)/self.config.battery.max_discharge_kw,
                      getattr(boundary,"battery_to_methanator_kw",0.0)/self.config.battery.max_discharge_kw])
        observation=np.concatenate((physical,market,cycle,np.clip(extra,0.0,1.0))).astype(np.float32)
        if observation.shape!=(149,) or not self.observation_space.contains(observation):
            raise RuntimeError("Whole-plant public observation violates declared149-vector")
        return observation

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
            stage="wholeplant-ppo-v1", variant=deepcopy(self.variant),
            daily_return_constraints_in_learning_objective=False,
            operator_reserve_equals_physical_floor=True)
        return info

    def public_control_input(self):
        """Return the actual issued observation without advancing encoder history."""
        return self.current_policy_observation.copy()

    def paired_source_manifest(self):
        """PRIVATE audit data; never provided to a policy or baseline."""
        if self._episode is None:
            raise RuntimeError("Reset before reading paired source evidence")
        source = self._original_sampled_episode
        payload = {"schema": "wholeplant-paired-exogenous-v1", "seed": source.seed,
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
        applied = (replace(requested, battery_dac_support_kw=0.0, battery_methanator_support_kw=0.0)
                   if not self.shared_bus else requested)
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
                                           "battery_support": state.battery_to_electrolyser_kw + state.battery_to_dac_kw + state.battery_to_methanator_kw,
                                           "battery_to_electrolyser": state.battery_to_electrolyser_kw,
                                           "battery_to_dac": state.battery_to_dac_kw, "battery_to_methanator": state.battery_to_methanator_kw},
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


DirectPlantEnv = WholePlantEnv
FreePlantEnv = WholePlantEnv

def make_env(split="train", **kwargs):
    return WholePlantEnv(split=split, **kwargs)

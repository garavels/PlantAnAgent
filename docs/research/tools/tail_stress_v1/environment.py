"""Endogenous-inventory stress tests over unchanged seven-day plant physics.

The controller operates normally before each disruption. Only the declared
exogenous schedule changes between paired conditions; no inventory, readiness,
history, observation, reward or action is edited after reset.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np

from docs.research.tools.wholeplant_ppo_v1 import environment as parent
from docs.research.tools.wholeplant_ppo_v1.environment import canonical, legacy
from plant_sim import SolarArray, SolarFactorTape
from plant_sim.faults import FaultSpec
from plant_backend.rl.v2.weather import build_weather_episode_from_tape

ENVIRONMENT_ID = "plant-wholeplant-tail-stress-v1"
SCHEMA = "wholeplant-tail-stress-environment-v1"
SEEDS = tuple(range(43_000_000, 43_000_016))
CONDITIONS = ("control", "pv6", "pv18", "pv36", "meth24", "meth48")
PV_ONSET_H = 48.0
METHANATOR_REPAIR_H = 86.0


def _sha(payload):
    return hashlib.sha256(canonical(payload)).hexdigest()


def condition_spec(condition):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown tail condition: {condition}")
    if condition == "control":
        return {"family": "control", "onset_h": None, "duration_h": 0.0, "end_h": None}
    if condition.startswith("pv"):
        duration = float(condition[2:])
        return {"family": "pv", "onset_h": PV_ONSET_H, "duration_h": duration,
                "end_h": PV_ONSET_H + duration}
    duration = float(condition[4:])
    return {"family": "methanator", "onset_h": METHANATOR_REPAIR_H - duration,
            "duration_h": duration, "end_h": METHANATOR_REPAIR_H}


def environment_identity():
    return {
        "schema": SCHEMA, "environment_id": ENVIRONMENT_ID,
        "parent_environment_identity_sha256": parent.environment_identity_sha256(),
        "arm": "shared-dynamic", "seeds": list(SEEDS),
        "sampling": "Unchanged frozen parent opening stocks and weather at each new seed; original random component faults disabled",
        "conditions": {condition: condition_spec(condition) for condition in CONDITIONS},
        "timing": "Offsets are hours from start at06:00; PV begins day3at06:00; methanator repair is day4at20:00",
        "horizon_h": 168.0, "delivery_period_h": 6.0, "delivery_amount_kg": 180.0,
        "faults": "Only the prescribed methanator outage; no other scheduled faults or degradation",
        "pv_truth": "Zero resource factor only in declared outage intervals; original sampled factors restored afterward",
        "forecast": "Unchanged original causal OU ensemble generator, configuration, clear-sky envelope and paired forecast seed; no future repair knowledge",
        "boundary_convention": "PV outage physical intervals [start,end) use tape boundaries start+1..end; forecasts at k use only current factor k; component health reflects the last completed physical interval",
        "endogenous_state": "No state intervention after reset; stocks, battery and readiness develop under each controller",
        "unchanged": "Seven actions,149observations,physical dynamics,accounting,economic objective,delivery timing",
        "statistical_status": "16paired scenario seeds under constructed stress families; no estimate of event frequency",
        "guard_added": False,
    }


def environment_identity_sha256():
    return _sha(environment_identity())


@dataclass(frozen=True)
class PrescribedDisturbances:
    """Private declared fault schedule for the existing physical fault engine."""
    faults: tuple[FaultSpec, ...]

    def to_dict(self):
        return {"schema": "tail-stress-prescribed-disturbances-v1",
                "schedule": [{**fault.to_dict(), "target": fault.kind.target, "mode": fault.kind.mode}
                             for fault in self.faults],
                "time_basis": "seconds since episode start; intervals [start,end)",
                "random_component_faults": False}

    @property
    def episode_sha256(self):
        return _sha(self.to_dict())

    def private_audit_manifest(self):
        return {**self.to_dict(), "episode_sha256": self.episode_sha256}


@lru_cache(maxsize=len(SEEDS))
def _healthy_parent_draw(seed):
    """Reuse immutable controller-independent draws within a worker process."""
    env = parent.WholePlantEnv(arm="shared-dynamic", split="train")
    try:
        env.beta_task = replace(env.beta_task,
            disturbances=replace(env.beta_task.disturbances, enabled=False))
        env.source_config = replace(env.source_config, beta_task=env.beta_task)
        episode = env._sample_episode(seed)
        return episode, env.market_episode
    finally:
        env.close()


def _masked_weather(base, *, config, episode_config, onset_step, end_step):
    original = base.weather
    factors = tuple(0.0 if onset_step < k <= end_step else factor
                    for k, factor in enumerate(original.tape.boundary_factors))
    tape = SolarFactorTape(dt_s=config.dt_s, boundary_factors=factors)
    solar = SolarArray(config.solar)
    forecast_horizon = round(24 * 3600 / config.dt_s)
    envelope = tuple(solar.clear_sky_kw(episode_config.start_hour * 3600 + k * config.dt_s)
                     / config.solar.peak_kw
                     for k in range(base.physics_steps - base.decision_stride + forecast_horizon))
    if original.forecast_stream_key.source_kind != "numpy.random.SeedSequence":
        raise RuntimeError("Original forecast seed replay type changed")
    weather = build_weather_episode_from_tape(original.config, tape=tape,
        decision_stride=base.decision_stride, truth_stream_key=original.truth_stream_key,
        forecast_seed=np.random.SeedSequence(**json.loads(original.forecast_stream_key.initial_state_json)),
        clear_sky_peak_fractions=envelope, clear_sky_envelope_id=original.clear_sky_envelope_id,
        truth_origin_kind="external_tape")
    if (weather.clear_sky_envelope_sha256 != original.clear_sky_envelope_sha256
            or weather.forecast_stream_key != original.forecast_stream_key):
        raise RuntimeError("Original forecast envelope or random stream changed")
    return weather


class TailStressEnv(parent.WholePlantEnv):
    """Matched disturbances after a period of controller-dependent operation."""

    def __init__(self, condition="control", *, record_minutes=False):
        specification = condition_spec(condition)
        super().__init__(arm="shared-dynamic", split="train", record_minutes=record_minutes)
        if (self.episode_config.start_hour != 6 or self.episode_config.horizon_h != 168
                or self.config.dt_s != 60 or self.episode_config.delivery_amount_kg != 180
                or self.episode_config.decision_interval_s != 900):
            raise RuntimeError("Parent geometry no longer matches the tail study")
        self.condition, self.specification = condition, specification
        self.split = "tail_stress"
        self.allowed_seeds = frozenset(SEEDS)
        self.beta_task = replace(self.beta_task,
            disturbances=replace(self.beta_task.disturbances, enabled=False))
        self.source_config = replace(self.source_config, beta_task=self.beta_task)
        self.variant = {**self.variant, "tail_condition": condition}
        self.benchmark_contract = {"schema": SCHEMA,
            "contract_sha256": environment_identity_sha256(),
            "parent_benchmark_sha256": self.benchmark_contract["contract_sha256"],
            "splits": {"tail_stress": list(SEEDS)}}
        payload = {"schema": "wholeplant-tail-stress-checkpoint-interface-v1",
            "parent_checkpoint_contract_digest": self.checkpoint_contract_digest,
            "diagnostic": environment_identity(), "variant": self.variant,
            "unchanged_policy_interface": parent.environment_identity()["observation"],
            "beta_task": self.beta_task.to_dict()}
        self._checkpoint_contract = {"checkpoint_contract_digest": _sha(payload), **payload}

    def _sample_episode(self, seed):
        base, market = _healthy_parent_draw(int(seed))
        self._base_episode = base
        self._market_episode = market
        spec = self.specification
        weather, faults = base.weather, ()
        if spec["family"] == "pv":
            weather = _masked_weather(base, config=self.config, episode_config=self.episode_config,
                onset_step=round(spec["onset_h"] * 60), end_step=round(spec["end_h"] * 60))
        elif spec["family"] == "methanator":
            faults = (FaultSpec.parse(FaultSpec(type="methanator_outage", severity=1.0,
                start_s=spec["onset_h"] * 3600, duration_s=spec["duration_h"] * 3600)),)
        self._disturbance_episode = PrescribedDisturbances(faults)
        scenario = replace(base.scenario, id=f"tail-stress-v1-{self.condition}",
            name=f"Paired tail condition: {self.condition}",
            description="Original sampled opening stocks and weather, with only the declared tail intervention.",
            faults=faults)
        drawn = replace(base, scenario=scenario, weather=weather)
        self._original_sampled_episode = drawn
        return drawn

    def reset(self, *, seed=None, options=None):
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
            raise ValueError("Seed must be an admitted tail-study integer")
        observation, info = legacy.BenchmarkEnv.reset(self, seed=seed, options=options)
        self.last_direct_minute_actions = None
        path = Path(__file__).resolve()
        self.frozen_source.source_pin = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.frozen_source.validation.update(kind="paired_tail_stress", tail_condition=self.condition,
            original_initial_inventory_preserved=self._episode.scenario.initial == self._base_episode.scenario.initial,
            base_paired_source_sha256=self.base_source_manifest()["base_paired_source_sha256"])
        return observation, info

    def _identify(self, info):
        info = super()._identify(info)
        info.update(environment_id=ENVIRONMENT_ID,
            environment_contract_digest=environment_identity_sha256(), stage="tail-stress-v1")
        return info

    def base_source_manifest(self):
        """Private evidence of pairing before applying the declared condition."""
        payload = {"schema": "tail-stress-base-source-v1",
            "episode": self._base_episode.exogenous_manifest(),
            "market_episode_sha256": self.market_episode.episode_sha256,
            "random_component_faults": False}
        return {"base_paired_source_sha256": _sha(payload), **payload}

    def paired_source_manifest(self):
        payload = super().paired_source_manifest()
        payload.pop("paired_source_sha256")
        payload.update(schema="tail-stress-paired-exogenous-v1", tail_condition=self.condition,
            diagnostic_identity_sha256=environment_identity_sha256(), base_source=self.base_source_manifest())
        return {"paired_source_sha256": _sha(payload), **payload}

    def audit_manifest(self):
        payload = super().audit_manifest()
        payload.pop("manifest_sha256")
        payload.update(environment=environment_identity(), tail_condition=self.condition,
            condition_specification=self.specification, base_source=self.base_source_manifest())
        accounting = {key: payload.pop(key) for key in
                      ("delivery_accounting", "economic_accounting", "cycle_accounting")}
        return {"manifest_sha256": _sha(payload), **payload, **accounting}

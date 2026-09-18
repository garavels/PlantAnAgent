"""A fixed, causal startup stress case, separate from all prior study splits.

Only exogenous conditions change. Every action, plant transition, settlement,
observation encoding and economic calculation comes from the frozen whole-plant
environment. The diagnostic episode is constructed before simulator reset.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np

from docs.research.tools.wholeplant_ppo_v1 import environment as parent
from docs.research.tools.wholeplant_ppo_v1.environment import canonical, legacy
from plant_sim import InitialState, Scenario, SolarArray, SolarFactorTape
from plant_backend.rl.v2.weather import FORECAST_WINDOWS_H, build_weather_episode_from_tape

ENVIRONMENT_ID = "plant-wholeplant-startup-stress-v1"
SCHEMA = "wholeplant-startup-stress-environment-v1"
DIAGNOSTIC_SEED = 42_000_000
CONDITIONS = ("solar-loss", "clear-sky")
OUTAGE_STEPS = 360


def _sha(payload):
    return hashlib.sha256(canonical(payload)).hexdigest()


def environment_identity():
    return {
        "schema": SCHEMA,
        "environment_id": ENVIRONMENT_ID,
        "parent_environment_identity_sha256": parent.environment_identity_sha256(),
        "arm": "shared-dynamic",
        "diagnostic_seed": DIAGNOSTIC_SEED,
        "statistical_status": "One constructed case with one matched control; no population-frequency estimate",
        "initial": {"battery_soc": .25, "h2_kg": 100.0, "co2_kg": 600.0, "ch4_kg": 120.0},
        "start_hour": 6.0,
        "horizon_h": 168.0,
        "delivery_period_h": 6.0,
        "delivery_amount_kg": 180.0,
        "equipment": "Healthy; no scheduled faults; all inherited cold readiness and electrolyser initial state",
        "conditions": {
            "solar-loss": "Resource factor zero for physics intervals 0..359; one thereafter",
            "clear-sky": "Resource factor one throughout",
        },
        "forecast": {
            "method": "Unchanged frozen synthetic-logit-ou-v1 ensemble forecast service, rebuilt from external truth tape",
            "windows_h": list(FORECAST_WINDOWS_H),
            "configuration_and_randomness": "Original WeatherConfigV2 and forecast SeedSequence state, paired across conditions",
            "causality": "An issue at boundary k uses only tape.at_boundary(k), never any subsequent factor",
            "no_repair_knowledge": "The unchanged OU model can predict recovery probabilistically; it receives no outage duration or repair schedule",
            "repair_visibility": "At boundary 360 the last completed interval is still blocked; restoration appears physically in interval 360 and publicly by the next decision at boundary 375",
        },
        "unchanged": "Plant ratings, efficiencies, readiness, physical commitments, seven actions, 149 observations, objective and delivery settlement",
        "guard_added": False,
    }


def environment_identity_sha256():
    return _sha(environment_identity())


def make_weather(*, config, episode_config, physics_steps, decision_stride, condition, base_weather):
    """Build physical truth and causal issued forecasts from separate inputs."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown startup condition: {condition}")
    factors = tuple(0.0 if condition == "solar-loss" and step <= OUTAGE_STEPS else 1.0
                    for step in range(physics_steps + 1))
    tape = SolarFactorTape(dt_s=config.dt_s, boundary_factors=factors)
    solar = SolarArray(config.solar)
    horizon_steps = round(24 * 3600 / config.dt_s)
    envelope = tuple(solar.clear_sky_kw(episode_config.start_hour * 3600 + step * config.dt_s)
                     / config.solar.peak_kw
                     for step in range(physics_steps - decision_stride + horizon_steps))
    if base_weather.forecast_stream_key.source_kind != "numpy.random.SeedSequence":
        raise RuntimeError("The original forecast seed no longer has the expected replay type")
    forecast_seed = np.random.SeedSequence(**json.loads(base_weather.forecast_stream_key.initial_state_json))
    weather = build_weather_episode_from_tape(base_weather.config, tape=tape,
        decision_stride=decision_stride, truth_stream_key=base_weather.truth_stream_key,
        forecast_seed=forecast_seed, clear_sky_peak_fractions=envelope,
        clear_sky_envelope_id=base_weather.clear_sky_envelope_id,
        truth_origin_kind="external_tape")
    if (weather.clear_sky_envelope_sha256 != base_weather.clear_sky_envelope_sha256
            or weather.forecast_stream_key != base_weather.forecast_stream_key):
        raise RuntimeError("The diagnostic changed the original forecast envelope or random stream")
    return weather


class StartupStressEnv(parent.WholePlantEnv):
    """Whole-plant direct control under one of two matched startup conditions."""

    def __init__(self, condition="solar-loss", *, record_minutes=False):
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown startup condition: {condition}")
        super().__init__(arm="shared-dynamic", split="train", record_minutes=record_minutes)
        if (self.episode_config.start_hour != 6 or self.episode_config.horizon_h != 168
                or self.config.dt_s != 60 or self.episode_config.delivery_amount_kg != 180
                or self.episode_config.decision_interval_s != 900):
            raise RuntimeError("Frozen parent no longer matches the diagnostic geometry")
        self.condition = condition
        self.split = "diagnostic"
        self.allowed_seeds = frozenset((DIAGNOSTIC_SEED,))
        self.opening_battery = "fixed_25_percent"
        self.beta_task = replace(self.beta_task,
            disturbances=replace(self.beta_task.disturbances, enabled=False))
        self.source_config = replace(self.source_config, beta_task=self.beta_task)
        self.variant = {**self.variant, "opening_battery": self.opening_battery,
                        "diagnostic_condition": condition}
        self.benchmark_contract = {"schema": SCHEMA,
            "contract_sha256": environment_identity_sha256(),
            "parent_benchmark_sha256": self.benchmark_contract["contract_sha256"],
            "splits": {"diagnostic": [DIAGNOSTIC_SEED]}}
        payload = {"schema": "wholeplant-startup-stress-checkpoint-interface-v1",
            "parent_checkpoint_contract_digest": self.checkpoint_contract_digest,
            "diagnostic": environment_identity(), "variant": self.variant,
            "unchanged_policy_interface": parent.environment_identity()["observation"],
            "beta_task": self.beta_task.to_dict()}
        self._checkpoint_contract = {"checkpoint_contract_digest": _sha(payload), **payload}

    def _sample_episode(self, seed):
        base = super()._sample_episode(seed)
        scenario = Scenario(id=f"startup-stress-v1-{self.condition}",
            name=f"Synthetic startup diagnostic: {self.condition}",
            description="Fixed low opening stocks and healthy equipment; solar availability is the paired intervention.",
            start_hour=6.0, initial=InitialState(**environment_identity()["initial"]), faults=())
        weather = make_weather(config=self.config, episode_config=self.episode_config,
            physics_steps=base.physics_steps, decision_stride=base.decision_stride,
            condition=self.condition, base_weather=base.weather)
        drawn = replace(base, scenario=scenario, weather=weather, stream_spawn_keys={})
        # The source being loaded and audited is this constructed episode. The
        # parent's discarded inventory/weather draws are never treated as data.
        self._original_sampled_episode = drawn
        return drawn

    def reset(self, *, seed=None, options=None):
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
            raise ValueError("Seed must be the admitted diagnostic integer")
        # This separate split deliberately does not invoke the old study's
        # held-out gates; the diagnostic runner freezes its own evaluation plan.
        observation, info = legacy.BenchmarkEnv.reset(self, seed=seed, options=options)
        self.last_direct_minute_actions = None
        path = Path(__file__).resolve()
        self.frozen_source.source_pin = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.frozen_source.validation.update(kind="constructed_startup_diagnostic",
            diagnostic_condition=self.condition, source_initial_gas_preserved=True,
            battery_initialization_intervention=self.opening_battery,
            paired_source_sha256=self.paired_source_manifest()["paired_source_sha256"])
        return observation, info

    def _identify(self, info):
        info = super()._identify(info)
        info.update(environment_id=ENVIRONMENT_ID,
            environment_contract_digest=environment_identity_sha256(), stage="startup-stress-v1")
        return info

    def paired_source_manifest(self):
        payload = super().paired_source_manifest()
        payload.pop("paired_source_sha256")
        payload.update(schema="startup-stress-paired-exogenous-v1", diagnostic_condition=self.condition,
            diagnostic_identity_sha256=environment_identity_sha256())
        return {"paired_source_sha256": _sha(payload), **payload}

    def audit_manifest(self):
        payload = super().audit_manifest()
        payload.pop("manifest_sha256")
        payload.update(environment=environment_identity(), diagnostic_condition=self.condition)
        accounting = {key: payload.pop(key) for key in
                      ("delivery_accounting", "economic_accounting", "cycle_accounting")}
        return {"manifest_sha256": _sha(payload), **payload, **accounting}

"""Direct five-coordinate control over the frozen plant's supported Action API.

Only conservation normalization remains in this adapter. The optional controller
guard, common setpoint allocator, coverage prior and commitment projector are not
called. The frozen simulator owns physical clipping and commitment consequences.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
from numbers import Integral
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = ROOT / "runs/authority-source-20260917"
for folder in (SOURCE_ROOT, SOURCE_ROOT / "backend/src", SOURCE_ROOT / "simulator/src"):
    if str(folder) in sys.path:
        sys.path.remove(str(folder))
    sys.path.insert(0, str(folder))

import numpy as np
from gymnasium.spaces import Box
from plant_sim import Action
from plant_sim.state import CRITICAL, OFFLINE
from plant_backend.rl.v2 import beta_spaces
from plant_backend.rl.v2.accounting import IntervalPowerAccounting
from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta
from plant_backend.rl.v2.boundary import CommitmentExposureV2, ControlStateV2, IntervalTelemetryAccumulator
from plant_backend.rl.v2.contract import contract_digest
from plant_backend.rl.v2.reward import IntervalOutcome, compute as diagnostic_reward
from plant_backend.rl.v2.training.config import ControllerNeutralTrainingConfigV1
from docs.research.tools import feasible_benchmark_v1 as legacy
from docs.research.tools.free_ppo_v1.direct_simulator import DirectPlantSimulator

ENVIRONMENT_ID = "plant-free-direct-control-v1"
SCHEMA = "free-ppo-direct-environment-v1"
TRAIN_BASE = 10_000_000
TRAIN_STOP = 10_100_000
DEVELOPMENT_SEEDS = tuple(range(10_100_000, 10_100_008))
HELD_OUT_SEEDS = tuple(range(10_200_000, 10_200_032))
ACTION_ORDER = ("battery_solar_share", "electrolyser_solar_share", "dac_solar_share",
                "methanator_solar_share", "battery_support")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _assert_frozen_imports() -> None:
    for name, module in tuple(sys.modules.items()):
        if (name.startswith(("plant_backend", "plant_sim")) or name in {
            "docs.research.tools.feasible_benchmark_v1", "docs.research.tools.feasible_mission_replay",
        }) and getattr(module, "__file__", None):
            if not Path(module.__file__).resolve().is_relative_to(SOURCE_ROOT):
                raise RuntimeError(f"Mutable working source was imported: {name}")


_assert_frozen_imports()


@lru_cache(maxsize=1)
def _environment_identity() -> dict[str, Any]:
    parent = legacy.load_benchmark_contract()
    return {
        "schema": SCHEMA, "environment_id": ENVIRONMENT_ID,
        "physical_parent_benchmark_sha256": parent["contract_sha256"],
        "observation": {"size": 135, "source": "actual current public beta observation; no appended prior targets",
                        "low": beta_spaces.observation_low().tolist(), "high": beta_spaces.observation_high().tolist()},
        "action": {"size": 5, "low": [-1.0] * 5, "high": [1.0] * 5, "order": list(ACTION_ORDER),
                   "mapping": "s=(clip(a[:4],-1,1)+1)/2; solar shares=s/max(1,sum(s)); support=(clip(a[4],-1,1)+1)/2*max_discharge_kw",
                   "coverage_prior": False, "commitment_projector": False, "optional_controller_guard": False,
                   "hold_seconds": 900, "physics_seconds": 60},
        "battery_floor": "modeled plant floor 0.05; no additional operator reserve",
        "refused_process_solar": "curtailed, never automatically added to battery charge allocation",
        "battery_netting": "unchanged frozen pre-component charge/support overlap cancellation; charge cap equals explicit battery solar allocation minus that overlap; actual discharge excludes charging",
        "reward": "unscaled existing economic operating EUR: delivery value, missed-delivery penalty, wear, inventory-liability changes; dedicated PV price zero",
        "daily_stock_returns": "diagnostic telemetry only; no PI costs in learner objective and no action restrictions",
        "physical_constraints": "unchanged simulator power/rating, inventory, battery efficiency/floor, faults and electrolyser commitment dynamics",
        "delivery_rate_kg_h": 25.0, "episode_decisions": 672,
        "splits": {"train": {"first": TRAIN_BASE, "stop_exclusive": TRAIN_STOP},
                   "development": list(DEVELOPMENT_SEEDS), "held_out": list(HELD_OUT_SEEDS)},
    }


def environment_identity() -> dict[str, Any]:
    return deepcopy(_environment_identity())


@lru_cache(maxsize=1)
def environment_identity_sha256() -> str:
    return hashlib.sha256(canonical(_environment_identity())).hexdigest()


def source_records() -> list[dict[str, Any]]:
    paths = {Path(__file__), Path(__file__).with_name("__init__.py"), Path(__file__).with_name("direct_simulator.py"),
             SOURCE_ROOT / "docs/research/benchmarks/feasible25_offgrid_v1.json",
             SOURCE_ROOT / "docs/research/tools/feasible_benchmark_v1.py",
             SOURCE_ROOT / "docs/research/tools/feasible_mission_replay.py",
             SOURCE_ROOT / "uv.lock"}
    for folder in (SOURCE_ROOT / "backend/src/plant_backend", SOURCE_ROOT / "simulator/src/plant_sim"):
        paths.update(folder.rglob("*.py"))
    return [{"path": str(path.resolve()), "bytes": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(paths)]


def source_identity_sha256() -> str:
    records = []
    for row in source_records():
        path = Path(row["path"])
        relative = ("frozen/" + path.relative_to(SOURCE_ROOT).as_posix()) if path.is_relative_to(SOURCE_ROOT) else ("experiment/" + path.name)
        records.append({"path": relative, "bytes": row["bytes"], "sha256": row["sha256"]})
    return hashlib.sha256(canonical(sorted(records, key=lambda row: row["path"]))).hexdigest()


def _admit(split: str, gate_path: Path | None) -> None:
    if split == "train":
        return
    if split not in {"development", "held_out"}:
        raise ValueError(f"Unknown direct-control split: {split}")
    if gate_path is None:
        raise ValueError("Development and held-out access require an external frozen gate")
    gate = json.loads(Path(gate_path).read_text())
    if (gate.get("schema") != "free-ppo-v1-evaluation-gate"
        or gate.get("environment_identity_sha256") != environment_identity_sha256()
        or gate.get("source_identity_sha256") != source_identity_sha256()
        or split not in gate.get("allowed_splits", [])
        or (split == "held_out" and gate.get("endpoints_frozen") is not True)):
        raise ValueError("External gate does not admit this source, environment and split")


def decode_action(action: Any, *, max_discharge_kw: float = 2500.0) -> tuple[Action, dict[str, Any]]:
    raw = np.asarray(action)
    if raw.shape != (5,) or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError("Expected exactly five finite real action coordinates")
    raw = raw.astype(np.float64)
    bounded = np.clip(raw, -1, 1)
    requested_shares = (bounded[:4] + 1) / 2
    scale = max(1.0, float(requested_shares.sum()))
    shares = requested_shares / scale
    request = Action(*map(float, shares), float((bounded[4] + 1) / 2 * max_discharge_kw))
    return request, {"raw_action": raw.tolist(), "bounded_action": bounded.tolist(),
                     "box_clipped": bool(np.any(raw != bounded)),
                     "solar_normalization_divisor": scale, "requested_action": request.to_dict()}


def action_to_vector(action: Action, *, max_discharge_kw: float = 2500.0) -> np.ndarray:
    """Exact inverse for any already admissible supported Action, up to float32."""
    safe = Action.sanitised(action)
    if safe != action or safe.battery_support_kw > max_discharge_kw:
        raise ValueError("Inverse requires an admissible solar allocation and support request")
    values = [safe.battery_share, safe.electrolyser_share, safe.dac_share,
              safe.methanator_share, safe.battery_support_kw / max_discharge_kw]
    return np.asarray(values, dtype=np.float64) * 2 - 1


class DirectPlantEnv(legacy.BenchmarkEnv):
    """Same frozen stochastic plant; independent direct controls and EUR reward."""

    def __init__(self, split: str = "train", *, gate_path: Path | str | None = None,
                 record_minutes: bool = False):
        self.gate_path = None if gate_path is None else Path(gate_path)
        _admit(split, self.gate_path)
        self._gate_sha256 = None if self.gate_path is None else hashlib.sha256(self.gate_path.read_bytes()).hexdigest()
        self.split = split
        self.record_minutes = bool(record_minutes)
        self.allowed_seeds = range(TRAIN_BASE, TRAIN_STOP) if split == "train" else (
            DEVELOPMENT_SEEDS if split == "development" else HELD_OUT_SEEDS)
        parent = legacy.load_benchmark_contract()
        cfg = ControllerNeutralTrainingConfigV1.from_dict(parent["source_template"])
        cfg = replace(cfg, constraints=replace(cfg.constraints, battery_min_soc=cfg.plant.battery.min_soc))
        self.source_config = cfg
        self.variant_rate_kg_h = 25.0
        self.electricity_model = legacy.ElectricityModelV1(**parent["electricity"])
        public_episode = replace(cfg.episode, delivery_rate_kg_h=self.variant_rate_kg_h,
                                 ch4_coverage_h=tuple(x * cfg.episode.delivery_rate_kg_h / self.variant_rate_kg_h
                                                     for x in cfg.episode.ch4_coverage_h))
        self.benchmark_contract = {"schema": SCHEMA, "contract_sha256": environment_identity_sha256(),
                                   "parent_benchmark_sha256": parent["contract_sha256"],
                                   "splits": environment_identity()["splits"]}
        PlantCyclicMarketEnvV2Beta.__init__(self, config=cfg.plant, episode=public_episode,
                                           constraints=cfg.constraints, weights=cfg.diagnostic_reward,
                                           economic_assumptions=cfg.economic_assumptions, beta_task=cfg.beta_task)
        self.simulator = DirectPlantSimulator(self.config, electrolyser_commitment=self.episode_config.electrolyser_commitment)
        self.action_space = Box(-1, 1, shape=(5,), dtype=np.float32)
        payload = {"schema": "free-direct-checkpoint-v1", "interface": environment_identity(),
                   "plant_config": cfg.plant.to_dict(), "episode_config": public_episode.to_dict(),
                   "constraints": cfg.constraints.to_dict(), "economic_assumptions": cfg.economic_assumptions.to_dict()}
        self._checkpoint_contract = {"checkpoint_contract_digest": contract_digest(payload), **payload}
        self._last_benchmark_seed = None
        self.last_direct_minute_actions: tuple[dict, ...] | None = None

    def reset(self, *, seed=None, options=None):
        if self.split != "train" and hashlib.sha256(self.gate_path.read_bytes()).hexdigest() != self._gate_sha256:
            raise ValueError("External evaluation gate changed after environment admission")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
            raise ValueError("Seed must be an admitted integer")
        result = super().reset(seed=seed, options=options)
        self.last_direct_minute_actions = None
        self.frozen_source.source_pin = {"path": str(Path(__file__).resolve()),
                                         "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        return result

    def _identify(self, info):
        info = super()._identify(info)
        info.update(environment_id=ENVIRONMENT_ID, environment_contract_digest=environment_identity_sha256(),
                    stage="free-ppo-v1", daily_return_constraints_in_learning_objective=False,
                    operator_reserve_equals_physical_floor=True)
        return info

    def audit_manifest(self):
        payload = super().audit_manifest()
        payload.pop("manifest_sha256")
        payload["environment"] = environment_identity()
        accounting = {key: payload.pop(key) for key in ("delivery_accounting", "economic_accounting", "cycle_accounting")}
        return {"manifest_sha256": hashlib.sha256(canonical(payload)).hexdigest(), **payload, **accounting}

    def step(self, action):
        requested, evidence = decode_action(action, max_discharge_kw=self.config.battery.max_discharge_kw)
        return self._step_direct(requested, evidence)

    def step_action(self, action: Action, *, controller_invalid: bool = False):
        raise RuntimeError("Use the declared five-coordinate step interface for every evaluated controller")

    def step_process_setpoints(self, *args, **kwargs):
        raise RuntimeError("Common target dispatcher is outside the direct-control experiment")

    def step_setpoints(self, *args, **kwargs):
        raise RuntimeError("MPC target dispatcher is outside the direct-control experiment")

    def _step_direct(self, requested: Action, evidence: dict):
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
            state = self.simulator.step(requested)
            boundary = self.telemetry
            telemetry.record(boundary, dt_h=self.config.dt_h)
            power.record(requested, requested, after=state, dt_h=self.config.dt_h)
            if minutes is not None:
                minutes.append(deepcopy(state))
                minute_actions.append({"ordinal": ordinal, "requested_action": requested.to_dict(),
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
                    "controller_guard": "bypassed; simulator physical constraints retained",
                    "physical_applied_energy_kwh": dict(power.applied_kwh)},
                "requested": {"action": requested.to_dict()}, "guarded_action": requested.to_dict(),
                "guard": {"mode": "absent", "interventions": 0, "corrected_physics_steps": 0,
                          "out_of_range": evidence["box_clipped"], "intervention_l1_kwh": 0.0},
                "interval_power": power.to_dict(), "adjustments": [],
                "control_authority": {"guard_intervention_integral_l1_kwh": 0.0,
                    "dispatch_gap_integral_l1_kwh": power.dispatch_gap_integral_l1_kwh,
                    "request_application_integral_l1_kwh": power.request_application_integral_l1_kwh},
                "delivery": {"commitments_settled": len(settlements), "requested_kg": requested_kg,
                             "delivered_kg": delivered_kg, "unmet_kg": unmet_kg},
                "electrolyser_commitment_interval": exposure.to_dict(), "reward": diagnostic.to_dict(),
                "metrics": self.metrics.to_dict(), "plant_status": after.plant_status}
        return self._finish_economic_step(start, (self._observe(), diagnostic.total, terminated, False, info))


FreePlantEnv = DirectPlantEnv


def make_env(split: str = "train", **kwargs) -> DirectPlantEnv:
    return DirectPlantEnv(split=split, **kwargs)

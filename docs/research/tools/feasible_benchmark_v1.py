"""Versioned off-grid feasible mission; historical beta contracts stay unchanged.

This research environment reuses the original physical distribution while
separately binding delivery demand, effective electricity prices and seed splits.
Public policy observations contain no underlying synthetic market or future tape.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import gzip
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for folder in ("backend/src", "simulator/src"):
    if str(ROOT / folder) not in sys.path:
        sys.path.insert(0, str(ROOT / folder))

from plant_sim import SolarArray
from plant_backend.rl.v2 import beta_spaces
from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta
from plant_backend.rl.v2.contract import contract_digest
from plant_backend.rl.v2.market import PriceForecastBinV1, PublicPriceForecastIssueV1
from plant_backend.rl.v2.training.config import ControllerNeutralTrainingConfigV1
from docs.research.tools.feasible_mission_replay import (
    canonical_sha,
    file_pin,
    load_frozen_scenario,
    run_controller,
)

CONTRACT_PATH = ROOT / "docs/research/benchmarks/feasible25_offgrid_v1.json"
ENVIRONMENT_ID = "plant-env-feasible25-offgrid-v1"
STAGE = "feasible-mission-research-v1"
INTERFACE_CONTRACT = {
    "schema": "feasible-mission-interface-v1",
    "environment_id": ENVIRONMENT_ID,
    "base_layout": beta_spaces.contract_payload(),
    "market_fields": "Effective accepted-solar dispatch charge; zero for dedicated PV without export. Underlying synthetic market remains private audit data.",
}
INTERFACE_DIGEST = contract_digest(INTERFACE_CONTRACT)


def load_benchmark_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    digest = payload.get("contract_sha256")
    if digest != canonical_sha(
        {k: v for k, v in payload.items() if k != "contract_sha256"}
    ):
        raise ValueError("Benchmark contract checksum differs")
    if payload["schema"] != "feasible-plant-benchmark-v1":
        raise ValueError("Unknown benchmark schema")
    sets = [
        set(payload["splits"][name])
        for name in ("train", "development", "held_out", "legacy_development")
    ]
    if any(a & b for i, a in enumerate(sets) for b in sets[i + 1 :]):
        raise ValueError("Benchmark seed splits overlap")
    return payload


@dataclass(frozen=True)
class ElectricityModelV1:
    mode: str = "dedicated_pv_no_export"
    marginal_solar_cost_eur_per_mwh: float = 0.0
    solar_capital_recovery_eur_per_week: float | None = None
    solar_fixed_om_eur_per_week: float | None = None
    calibration: str = (
        "Fixed solar costs uncalibrated; operating contribution excludes them."
    )

    def __post_init__(self):
        if self.mode not in {"dedicated_pv_no_export", "solar_export_opportunity"}:
            raise ValueError("Unknown electricity mode")
        price = self.marginal_solar_cost_eur_per_mwh
        if (
            isinstance(price, bool)
            or not isinstance(price, (int, float))
            or not math.isfinite(price)
            or not 0 <= price <= 300
        ):
            raise ValueError("Marginal solar price must be a finite number in [0,300]")
        if (
            self.solar_capital_recovery_eur_per_week is not None
            or self.solar_fixed_om_eur_per_week is not None
        ):
            raise ValueError(
                "Fixed solar costs await calibration; this version requires null reporting placeholders"
            )
        for key, value in asdict(self).items():
            if key in {"mode", "calibration"} or value is None:
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid electricity assumption: {key}")
        if (
            self.mode == "solar_export_opportunity"
            and self.marginal_solar_cost_eur_per_mwh
        ):
            raise ValueError(
                "Export sensitivity uses the declared market opportunity price only"
            )

    def effective_price(self, market_price: float) -> float:
        return (
            max(0.0, market_price)
            if self.mode == "solar_export_opportunity"
            else self.marginal_solar_cost_eur_per_mwh
        )

    def to_dict(self):
        return {"schema": "solar-electricity-boundary-v1", **asdict(self)}


class BenchmarkEnv(PlantCyclicMarketEnvV2Beta):
    """Reusable unreset Gym environment, with explicit permitted reset seeds."""

    def __init__(
        self,
        split: str = "train",
        *,
        electricity_model: ElectricityModelV1 | None = None,
    ):
        self.benchmark_contract = load_benchmark_contract()
        if split not in self.benchmark_contract["splits"]:
            raise ValueError("Unknown benchmark split")
        self.split = split
        self.allowed_seeds = frozenset(self.benchmark_contract["splits"][split])
        self.electricity_model = electricity_model or ElectricityModelV1(
            **self.benchmark_contract["electricity"]
        )
        cfg = ControllerNeutralTrainingConfigV1.from_dict(
            self.benchmark_contract["source_template"]
        )
        self.source_config = cfg
        self.variant_rate_kg_h = self.benchmark_contract["delivery_rate_kg_h"]
        ratio = cfg.episode.delivery_rate_kg_h / self.variant_rate_kg_h
        public_episode = replace(
            cfg.episode,
            delivery_rate_kg_h=self.variant_rate_kg_h,
            ch4_coverage_h=tuple(x * ratio for x in cfg.episode.ch4_coverage_h),
        )
        super().__init__(
            config=cfg.plant,
            episode=public_episode,
            constraints=cfg.constraints,
            weights=cfg.diagnostic_reward,
            economic_assumptions=cfg.economic_assumptions,
            beta_task=cfg.beta_task,
        )
        payload = self.checkpoint_contract
        payload.pop("checkpoint_contract_digest")
        payload.update(
            environment_id=ENVIRONMENT_ID,
            interface_contract=INTERFACE_CONTRACT,
            interface_contract_digest=INTERFACE_DIGEST,
            benchmark_contract_sha256=self.benchmark_contract["contract_sha256"],
            electricity_model=self.electricity_model.to_dict(),
            original_inventory_sampling_episode=cfg.episode.to_dict(),
        )
        payload["economic_objective"][
            "realized_price_source"
        ] = "Effective electricity model price; dedicated PV has no export or purchase charge"
        self._checkpoint_contract = {
            "checkpoint_contract_digest": contract_digest(payload),
            **payload,
        }
        self._last_benchmark_seed: int | None = None

    def _sample_episode(self, seed):
        cfg = self.source_config
        if self.split == "legacy_development":
            source = load_frozen_scenario(seed)
            episode, market, disturbances = (
                source.episode,
                source.market,
                source.disturbances,
            )
        else:
            original = PlantCyclicMarketEnvV2Beta(
                config=cfg.plant,
                episode=cfg.episode,
                constraints=cfg.constraints,
                weights=cfg.diagnostic_reward,
                economic_assumptions=cfg.economic_assumptions,
                beta_task=cfg.beta_task,
            )
            try:
                episode = original._sample_episode(seed)
                market, disturbances = (
                    original.market_episode,
                    original.disturbance_episode,
                )
            finally:
                original.close()
        self._market_episode, self._disturbance_episode = market, disturbances
        self._original_sampled_episode = episode
        return replace(
            episode,
            commitments=tuple(
                replace(
                    c,
                    requested_kg=self.variant_rate_kg_h
                    * self.episode_config.delivery_period_h,
                )
                for c in episode.commitments
            ),
        )

    def reset(self, *, seed=None, options=None):
        seed = self._last_benchmark_seed if seed is None else seed
        if isinstance(seed, bool) or seed not in self.allowed_seeds:
            raise ValueError(f"Seed {seed} is outside benchmark split {self.split}")
        self._last_benchmark_seed = int(seed)
        observation, info = super().reset(seed=int(seed), options=options)
        self.current_policy_observation = observation.copy()
        solar = SolarArray(self.config.solar)
        episode = self._episode
        expected = tuple(
            solar.clear_sky_kw(
                self.episode_config.start_hour * 3600 + i * self.config.dt_s
            )
            * episode.weather.tape.for_interval(i)
            for i in range(episode.physics_steps)
        )
        self.frozen_source = SimpleNamespace(
            seed=int(seed),
            source_pin=file_pin(CONTRACT_PATH),
            validation={
                "kind": "seeded_benchmark",
                "source_initial_inventory_preserved": episode.scenario.initial
                == self._original_sampled_episode.scenario.initial,
                "source_weather_preserved": episode.weather
                == self._original_sampled_episode.weather,
                "split": self.split,
                "benchmark_contract_sha256": self.benchmark_contract["contract_sha256"],
            },
            original_manifest=self.audit_manifest(),
            expected_solar_kw=expected,
        )
        return observation, info

    @property
    def current_market_issue(self):
        original = super().current_market_issue
        if self.electricity_model.mode == "solar_export_opportunity":
            return original
        price = self.electricity_model.marginal_solar_cost_eur_per_mwh
        return PublicPriceForecastIssueV1(
            issued_at_decision=original.issued_at_decision,
            issued_at_offset_s=original.issued_at_offset_s,
            current_eur_per_mwh=price,
            bins=tuple(PriceForecastBinV1(i, price, price, price) for i in range(24)),
            market_id="dedicated-pv-effective-dispatch-price-v1",
        )

    def _opportunity_price_eur_per_mwh(self, *, time_h, decision_index):
        return self.electricity_model.effective_price(
            self.market_episode.price_at_decision(decision_index)
        )

    def _economic_price_context(self, *, time_h, decision_index):
        original = self.market_episode.issue_at_decision(decision_index)
        return {
            "electricity_model": self.electricity_model.to_dict(),
            "effective_price_eur_per_mwh": self.electricity_model.effective_price(
                original.current_eur_per_mwh
            ),
        }

    def _public_info(self):
        return self._identify(super()._public_info())

    def _identify(self, info):
        info.update(
            environment_id=ENVIRONMENT_ID,
            environment_contract_digest=INTERFACE_DIGEST,
            stage=STAGE,
            benchmark_contract_sha256=self.benchmark_contract["contract_sha256"],
            electricity_model=self.electricity_model.to_dict(),
        )
        info["economic_context"] = {
            "assumptions_sha256": self.economic_assumptions.sha256,
            "effective_price_eur_per_mwh": self.electricity_model.effective_price(
                self.current_market_issue.current_eur_per_mwh
            ),
            "price_source": "explicit benchmark electricity model",
        }
        return info

    def _finish_economic_step(self, start, result):
        observation, reward, terminated, truncated, info = (
            super()._finish_economic_step(start, result)
        )
        self.current_policy_observation = observation.copy()
        return observation, reward, terminated, truncated, self._identify(info)

    def audit_manifest(self):
        payload = super().audit_manifest()
        payload.pop("manifest_sha256")
        payload.update(
            environment={
                "id": ENVIRONMENT_ID,
                "stage": STAGE,
                "interface": INTERFACE_CONTRACT,
            },
            benchmark_contract_sha256=self.benchmark_contract["contract_sha256"],
            electricity_model=self.electricity_model.to_dict(),
        )
        accounting = {
            k: payload.pop(k)
            for k in ("delivery_accounting", "economic_accounting", "cycle_accounting")
        }
        return {"manifest_sha256": canonical_sha(payload), **payload, **accounting}


def evaluate_controller(
    seed, controller, *, split="development", output_path=None, retain_minutes=True
):
    """Use the existing exact-minute auditor with this task's explicit identity."""
    env = BenchmarkEnv(split=split)
    try:
        env.reset(seed=seed)
        if isinstance(controller, type):
            raise TypeError(
                "Supply an already constructed controller; constructors do not receive the private environment"
            )
        result = run_controller(env, controller, retain_minutes=retain_minutes)
        result.update(
            schema="feasible-offgrid-benchmark-recording-v1",
            split=split,
            benchmark_contract_sha256=env.benchmark_contract["contract_sha256"],
            electricity_model=env.electricity_model.to_dict(),
            benchmark_source_code=file_pin(Path(__file__)),
            change="25kg/h delivery mission with original sampled physical inventories, weather and faults; dedicated off-grid effective electricity prices; original daily returns and safety.",
        )
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite recording {path}")
            raw = json.dumps(result, separators=(",", ":"), allow_nan=False).encode()
            path.write_bytes(
                gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw
            )
        return result
    finally:
        env.close()

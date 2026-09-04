"""Opt-in common-setpoint environment with explicit economic/wear accounting."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from numbers import Real
from typing import Any, cast

import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray
from plant_sim import PlantConfig

from ...controllers.base import Constraints
from . import economic_spaces
from .boundary import CommitmentExposureV2
from .config import EpisodeConfigV2
from .contract import contract_digest
from .economics import (
    ECONOMIC_OBJECTIVE_VERSION,
    EconomicAssumptionsV1,
    EconomicControllerObservationV1,
    EconomicEpisodeMetricsV1,
    EconomicIntervalOutcomeV1,
    ch4_equivalent_inventory_kg,
    compute_economic_reward,
)
from .reward import OBJECTIVE_VERSION, RewardWeightsV2
from .setpoint_command import ProcessSetpointCommandV1
from .setpoint_env import (
    PlantSetpointEnvV2,
    describe_setpoint_checkpoint_contract,
)

ECONOMIC_CHECKPOINT_CONTRACT_SCHEMA_VERSION = 5


def _canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_tariff_cadence(episode: EpisodeConfigV2) -> None:
    """Guarantee one declared hourly price applies to every decision interval."""

    interval_s = episode.decision_interval_s
    if (
        isinstance(interval_s, bool)
        or not isinstance(interval_s, Real)
        or not math.isfinite(float(interval_s))
        or interval_s <= 0.0
    ):
        raise ValueError(
            "economic task decision_interval_s must be positive and finite"
        )
    intervals_per_hour = 3_600.0 / interval_s
    if interval_s > 3_600.0 or not math.isclose(
        intervals_per_hour,
        round(intervals_per_hour),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "economic task decision_interval_s must divide one hour exactly"
        )
    start_intervals = episode.start_hour * 3_600.0 / interval_s
    if not math.isclose(
        start_intervals,
        round(start_intervals),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "economic task start_hour must align with its decision interval"
        )


def describe_economic_checkpoint_contract(
    *,
    plant: PlantConfig,
    episode: EpisodeConfigV2,
    constraints: Constraints,
    weights: RewardWeightsV2,
    assumptions: EconomicAssumptionsV1,
) -> dict[str, Any]:
    """Return the full task identity, distinct from both earlier V2 interfaces."""

    _validate_tariff_cadence(episode)
    assumptions.validate()
    inherited = describe_setpoint_checkpoint_contract(
        plant=plant,
        episode=episode,
        constraints=constraints,
        weights=weights,
    )
    inherited.pop("checkpoint_contract_digest")
    diagnostic_legacy_objective = inherited.pop("objective")
    inherited.update(
        {
            "schema_version": ECONOMIC_CHECKPOINT_CONTRACT_SCHEMA_VERSION,
            "environment_id": economic_spaces.ENVIRONMENT_ID,
            "interface_contract": economic_spaces.contract_payload(),
            "interface_contract_digest": (economic_spaces.ENVIRONMENT_CONTRACT_DIGEST),
            "objective": {
                "version": ECONOMIC_OBJECTIVE_VERSION,
                "unit": "EUR per controller interval",
                "assumptions": assumptions.to_dict(),
                "hard_safety_is_monetized": False,
                "terminal_endowment_semantics": (
                    "per-step liability changes telescope exactly to the final "
                    "deficit below initial CH4-equivalent and battery inventories"
                ),
            },
            "diagnostic_legacy_objective": diagnostic_legacy_objective,
            "tariff_cadence": (
                "hourly piecewise-constant public tariff; controller decisions "
                "align with and do not cross tariff boundaries"
            ),
        }
    )
    return {"checkpoint_contract_digest": contract_digest(inherited), **inherited}


class PlantEconomicSetpointEnvV2(PlantSetpointEnvV2):  # type: ignore[misc]
    """Four-target task whose returned reward is auditable operating value."""

    def __init__(
        self,
        *,
        config: PlantConfig | None = None,
        episode: EpisodeConfigV2 | None = None,
        constraints: Constraints | None = None,
        weights: RewardWeightsV2 | None = None,
        economic_assumptions: EconomicAssumptionsV1 | None = None,
    ) -> None:
        self.economic_assumptions = (
            economic_assumptions or EconomicAssumptionsV1()
        ).validate()
        super().__init__(
            config=config,
            episode=episode,
            constraints=constraints,
            weights=weights,
        )
        _validate_tariff_cadence(self.episode_config)
        self.observation_space = Box(
            low=economic_spaces.observation_low(),
            high=economic_spaces.observation_high(),
            shape=(economic_spaces.OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self._checkpoint_contract = describe_economic_checkpoint_contract(
            plant=self.config,
            episode=self.episode_config,
            constraints=self.constraints,
            weights=self.weights,
            assumptions=self.economic_assumptions,
        )
        self.economic_metrics = EconomicEpisodeMetricsV1(
            initial_ch4_equivalent_kg=0.0,
            initial_battery_energy_kwh=0.0,
            battery_capacity_kwh=self.config.battery.capacity_kwh,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.generic], dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        state = self.telemetry
        self.economic_metrics = EconomicEpisodeMetricsV1(
            initial_ch4_equivalent_kg=ch4_equivalent_inventory_kg(state, self.config),
            initial_battery_energy_kwh=state.battery_energy_kwh,
            battery_capacity_kwh=self.config.battery.capacity_kwh,
        )
        info["economic_metrics"] = self.economic_metrics.to_dict()
        return observation, info

    def step_process_setpoints(
        self,
        command: ProcessSetpointCommandV1,
        *,
        controller_invalid: bool = False,
    ) -> tuple[NDArray[np.generic], float, bool, bool, dict[str, Any]]:
        """Run the common physical door, then replace only its scalar objective."""

        if self.control_state is None:
            raise RuntimeError("reset() must be called before step_process_setpoints()")
        before = self.telemetry
        before_commitment = self.simulator.electrolyser_commitment
        before_decisions = self.decisions_taken
        before_solar_kwh = self.metrics.solar_kwh
        before_curtailed_kwh = self.metrics.curtailed_kwh
        before_battery_throughput_kwh = self.metrics.battery_throughput_kwh
        before_delivered_kg = self.metrics.delivered_kg
        before_unmet_kg = self.metrics.unmet_kg
        price = self.economic_assumptions.tariff.price_at(before.time_h)

        (
            observation,
            legacy_reward,
            terminated,
            truncated,
            info,
        ) = super().step_process_setpoints(
            command,
            controller_invalid=controller_invalid,
        )
        after = self.telemetry
        interval_s = (self.decisions_taken - before_decisions) * (
            self.episode_config.decision_interval_s
        )
        exposure = CommitmentExposureV2.between(
            before_commitment,
            self.simulator.electrolyser_commitment,
            interval_s=interval_s,
        )
        interval_solar_kwh = self.metrics.solar_kwh - before_solar_kwh
        interval_curtailed_kwh = self.metrics.curtailed_kwh - before_curtailed_kwh
        accepted_solar_kwh = max(0.0, interval_solar_kwh - interval_curtailed_kwh)
        outcome = EconomicIntervalOutcomeV1(
            opportunity_price_eur_per_mwh=price,
            accepted_solar_kwh=accepted_solar_kwh,
            delivered_kg=max(0.0, self.metrics.delivered_kg - before_delivered_kg),
            unmet_kg=max(0.0, self.metrics.unmet_kg - before_unmet_kg),
            battery_throughput_kwh=max(
                0.0,
                self.metrics.battery_throughput_kwh - before_battery_throughput_kwh,
            ),
            initial_ch4_equivalent_kg=(self.economic_metrics.initial_ch4_equivalent_kg),
            ch4_equivalent_before_kg=ch4_equivalent_inventory_kg(before, self.config),
            ch4_equivalent_after_kg=ch4_equivalent_inventory_kg(after, self.config),
            initial_battery_energy_kwh=(
                self.economic_metrics.initial_battery_energy_kwh
            ),
            battery_energy_before_kwh=before.battery_energy_kwh,
            battery_energy_after_kwh=after.battery_energy_kwh,
            commitment=exposure,
        )
        breakdown = compute_economic_reward(
            outcome,
            config=self.config,
            assumptions=self.economic_assumptions,
        )
        self.economic_metrics.record(outcome, breakdown)

        legacy_breakdown = info.get("reward")
        if not isinstance(legacy_breakdown, dict):
            raise RuntimeError("legacy reward evidence is malformed")
        info["legacy_reward"] = {
            "scalar": legacy_reward,
            "breakdown": deepcopy(legacy_breakdown),
            "diagnostic_only": True,
        }
        info["reward"] = breakdown.to_dict()
        info["economic_metrics"] = self.economic_metrics.to_dict()
        info["economic_interval"] = {
            **breakdown.signals,
            "tariff": self.economic_assumptions.tariff.observe(before.time_h).to_dict(),
        }
        return observation, breakdown.total_eur, terminated, truncated, info

    @property
    def economic_controller_observation(self) -> EconomicControllerObservationV1:
        """Detached typed source carrying the same tariff encoded for Gym policies."""

        physical = self.controller_observation
        return EconomicControllerObservationV1(
            plant=physical,
            tariff=self.economic_assumptions.tariff.observe(
                physical.state.boundary.time_h
            ),
            tariff_id=self.economic_assumptions.tariff.id,
            economic_assumptions_sha256=self.economic_assumptions.sha256,
        )

    def _observe(self) -> NDArray[np.float32]:
        physical = np.asarray(super()._observe(), dtype=np.float32)
        tariff = np.asarray(
            self.economic_controller_observation.tariff.normalized(), dtype=np.float32
        )
        observation = np.concatenate((physical, tariff)).astype(np.float32, copy=False)
        if observation.shape != (economic_spaces.OBSERVATION_SIZE,):
            raise RuntimeError("economic observation does not match its contract")
        return observation

    def _public_info(self) -> dict[str, Any]:
        info = cast(dict[str, Any], super()._public_info())
        info.update(
            {
                "environment_id": economic_spaces.ENVIRONMENT_ID,
                "environment_contract_digest": (
                    economic_spaces.ENVIRONMENT_CONTRACT_DIGEST
                ),
                "stage": economic_spaces.STAGE,
                "economic_context": {
                    "assumptions_sha256": self.economic_assumptions.sha256,
                    "tariff": self.economic_assumptions.tariff.observe(
                        self.telemetry.time_h
                    ).to_dict(),
                },
            }
        )
        return info

    def audit_manifest(self) -> dict[str, Any]:
        """Bind physical episode, tariff, and economics without hiding calibration gaps."""

        inherited = super().audit_manifest()
        economic_exogenous = {
            "physical_exogenous_episode_sha256": inherited["exogenous_episode_sha256"],
            "tariff": self.economic_assumptions.tariff.to_dict(),
        }
        immutable = {
            "manifest_schema_version": 5,
            "environment": economic_spaces.describe(),
            "checkpoint_contract": self.checkpoint_contract,
            "objective_version": ECONOMIC_OBJECTIVE_VERSION,
            "economic_assumptions": self.economic_assumptions.to_dict(),
            "diagnostic_legacy_objective_version": OBJECTIVE_VERSION,
            "episode_config": inherited["episode_config"],
            "plant_config": inherited["plant_config"],
            "constraints": inherited["constraints"],
            "diagnostic_legacy_reward_weights": inherited["reward_weights"],
            "episode": inherited["episode"],
            "economic_exogenous": economic_exogenous,
            "provenance_gaps": inherited["provenance_gaps"],
        }
        return {
            "manifest_sha256": _canonical_sha256(immutable),
            "exogenous_episode_sha256": inherited["exogenous_episode_sha256"],
            "economic_episode_sha256": _canonical_sha256(economic_exogenous),
            **immutable,
            "delivery_accounting": inherited["delivery_accounting"],
            "economic_accounting": self.economic_metrics.to_dict(),
        }


__all__ = [
    "ECONOMIC_CHECKPOINT_CONTRACT_SCHEMA_VERSION",
    "PlantEconomicSetpointEnvV2",
    "describe_economic_checkpoint_contract",
]

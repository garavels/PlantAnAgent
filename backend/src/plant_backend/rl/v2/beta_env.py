"""Cyclic stochastic-market Environment V2 beta.

This task layers public price forecasts, exact per-day resource-return checks,
and private finite DAC outages over the established controller-neutral physical
door.  Alpha environments and their episode streams are not changed.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from typing import Any, cast

import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray
from plant_sim import PlantConfig, Scenario

from ...controllers.base import Constraints
from . import beta_spaces
from .beta_config import (
    CycleReferenceV1,
    CycleSettlementV1,
    CyclicMarketTaskConfigV1,
    canonical_sha256,
)
from .beta_control import CycleObservationV1, CyclicMarketControllerObservationV1
from .beta_disturbances import DisturbanceEpisodeV1, generate_disturbance_episode
from .config import EpisodeConfigV2, EpisodeV2, sample_episode
from .contract import contract_digest
from .economic_env import PlantEconomicSetpointEnvV2
from .economics import ECONOMIC_OBJECTIVE_VERSION, EconomicAssumptionsV1
from .market import MarketEpisodeV1, PublicPriceForecastIssueV1, generate_market_episode
from .reward import OBJECTIVE_VERSION, RewardWeightsV2
from .setpoint_env import PlantSetpointEnvV2, describe_setpoint_checkpoint_contract


BETA_CHECKPOINT_CONTRACT_SCHEMA_VERSION = 1
BETA_MANIFEST_SCHEMA_VERSION = 1
BETA_ROOT_STREAM_NAMES = (
    "inventories",
    "weather",
    "contracts",
    "market",
    "disturbances",
)


def describe_beta_checkpoint_contract(
    *,
    plant: PlantConfig,
    episode: EpisodeConfigV2,
    constraints: Constraints,
    weights: RewardWeightsV2,
    assumptions: EconomicAssumptionsV1,
    beta_task: CyclicMarketTaskConfigV1,
) -> dict[str, Any]:
    """Bind the complete beta task without aliasing an alpha checkpoint."""

    beta_task.validate(episode=episode, plant=plant, assumptions=assumptions)
    inherited = describe_setpoint_checkpoint_contract(
        plant=plant,
        episode=episode,
        constraints=constraints,
        weights=weights,
    )
    inherited.pop("checkpoint_contract_digest")
    inherited.update(
        {
            "schema_version": BETA_CHECKPOINT_CONTRACT_SCHEMA_VERSION,
            "environment_id": beta_spaces.ENVIRONMENT_ID,
            "interface_contract": beta_spaces.contract_payload(),
            "interface_contract_digest": beta_spaces.ENVIRONMENT_CONTRACT_DIGEST,
            "economic_objective": {
                "version": ECONOMIC_OBJECTIVE_VERSION,
                "unit": "EUR per controller interval",
                "assumptions": assumptions.to_dict(),
                "realized_price_source": "market truth at interval opening",
                "negative_price_semantics": (
                    "accepted renewable opportunity cost remains floored at zero"
                ),
            },
            "cyclic_market_beta": beta_task.to_dict(),
            "cyclic_market_beta_sha256": beta_task.sha256,
            "random_stream_contract": {
                "root_children": list(BETA_ROOT_STREAM_NAMES),
                "alpha_children_unchanged": list(BETA_ROOT_STREAM_NAMES[:3]),
                "beta_children": list(BETA_ROOT_STREAM_NAMES[3:]),
            },
            "cycle_constraint_semantics": (
                "battery, H2, CO2, and CH4 settle separately against one "
                "non-ratcheting episode-opening reference at each 24 h boundary; "
                "constraint cost is reported separately and is not monetized"
            ),
            "disturbance_visibility": (
                "only currently observable plant effects; no future schedule, seed, "
                "or selected market regime"
            ),
            "execution_doors": {
                "normalized_four_target_gym": "realized EUR objective",
                "controller_neutral_process_command": "realized EUR objective",
                "mpc_setpoint_with_rich_execution_evidence": ("realized EUR objective"),
                "legacy_share_action": "rejected before physical state movement",
            },
        }
    )
    return {"checkpoint_contract_digest": contract_digest(inherited), **inherited}


class PlantCyclicMarketEnvV2Beta(PlantEconomicSetpointEnvV2):  # type: ignore[misc]
    """Uninterrupted economic task with causal market and cycle constraints."""

    def __init__(
        self,
        *,
        config: PlantConfig | None = None,
        episode: EpisodeConfigV2 | None = None,
        constraints: Constraints | None = None,
        weights: RewardWeightsV2 | None = None,
        economic_assumptions: EconomicAssumptionsV1 | None = None,
        beta_task: CyclicMarketTaskConfigV1 | None = None,
    ) -> None:
        self.beta_task = beta_task or CyclicMarketTaskConfigV1()
        self._market_episode: MarketEpisodeV1 | None = None
        self._disturbance_episode: DisturbanceEpisodeV1 | None = None
        self._cycle_reference: CycleReferenceV1 | None = None
        self._cycle_settlements: list[CycleSettlementV1] = []
        super().__init__(
            config=config,
            episode=episode,
            constraints=constraints,
            weights=weights,
            economic_assumptions=economic_assumptions,
        )
        self.beta_task.validate(
            episode=self.episode_config,
            plant=self.config,
            assumptions=self.economic_assumptions,
        )
        self.observation_space = Box(
            low=beta_spaces.observation_low(),
            high=beta_spaces.observation_high(),
            shape=(beta_spaces.OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self._checkpoint_contract = describe_beta_checkpoint_contract(
            plant=self.config,
            episode=self.episode_config,
            constraints=self.constraints,
            weights=self.weights,
            assumptions=self.economic_assumptions,
            beta_task=self.beta_task,
        )

    def _sample_episode(self, seed: int) -> EpisodeV2:
        """Attach independent market and disturbance streams to an alpha draw."""

        base = sample_episode(self.episode_config, self.config, seed=seed)
        children = np.random.SeedSequence(seed).spawn(len(BETA_ROOT_STREAM_NAMES))
        decision_count = base.physics_steps // base.decision_stride
        cycle_decisions = self.beta_task.cycle.cycle_decisions(self.episode_config)
        if decision_count % cycle_decisions:
            raise RuntimeError("validated beta horizon lost cycle alignment")
        cycle_count = decision_count // cycle_decisions
        self._market_episode = generate_market_episode(
            self.beta_task.market,
            seed=children[3],
            decision_count=decision_count,
            decision_interval_s=self.episode_config.decision_interval_s,
            start_hour=self.episode_config.start_hour,
            baseline=self.economic_assumptions.tariff,
        )
        self._disturbance_episode = generate_disturbance_episode(
            self.beta_task.disturbances,
            seed=children[4],
            cycle_h=self.beta_task.cycle.cycle_h,
            cycle_count=cycle_count,
            physics_dt_s=self.config.dt_s,
        )
        scenario = Scenario(
            id="v2_cyclic_market_beta_v1",
            name="Environment V2 cyclic stochastic-market beta",
            description=(
                "Recurring service under stochastic prices, non-ratcheting cycle "
                "returns, causal weather/market forecasts, and finite DAC outages."
            ),
            start_hour=base.scenario.start_hour,
            initial=base.scenario.initial,
            faults=self._disturbance_episode.faults,
        )
        return replace(base, scenario=scenario)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.generic], dict[str, Any]]:
        self._cycle_reference = None
        self._cycle_settlements = []
        super().reset(seed=seed, options=options)
        # Use the simulator's authoritative state. The first beta disturbance
        # profile has no sensor faults, but this keeps reference semantics exact.
        self._cycle_reference = CycleReferenceV1.from_state(self.simulator.state)
        return self._observe(), self._public_info()

    @property
    def market_episode(self) -> MarketEpisodeV1:
        """Private episode object for audit/evaluation code, never Gym info."""

        if self._market_episode is None:
            raise RuntimeError("reset() must be called before market audit")
        return self._market_episode

    @property
    def disturbance_episode(self) -> DisturbanceEpisodeV1:
        """Private scheduled disturbances for replay, never controller input."""

        if self._disturbance_episode is None:
            raise RuntimeError("reset() must be called before disturbance audit")
        return self._disturbance_episode

    @property
    def cycle_reference(self) -> CycleReferenceV1:
        if self._cycle_reference is None:
            raise RuntimeError("reset() must be called before cycle observation")
        return self._cycle_reference

    @property
    def cycle_settlements(self) -> tuple[CycleSettlementV1, ...]:
        return tuple(self._cycle_settlements)

    @property
    def current_market_issue(self) -> PublicPriceForecastIssueV1:
        episode = self.market_episode
        index = min(self.decisions_taken, len(episode.issues) - 1)
        return episode.issue_at_decision(index)

    @property
    def cycle_observation(self) -> CycleObservationV1:
        cycle_decisions = self.beta_task.cycle.cycle_decisions(self.episode_config)
        terminal = self.physics_steps_taken >= self.episode_physics_steps
        if terminal:
            cycle_index = max(1, math.ceil(self.decisions_taken / cycle_decisions))
            decision_in_cycle = cycle_decisions
        else:
            cycle_index = self.decisions_taken // cycle_decisions + 1
            decision_in_cycle = self.decisions_taken % cycle_decisions
        if self._cycle_reference is None:
            targets = {name: 0.0 for name in ("battery", "h2", "co2", "ch4")}
        else:
            targets = self._cycle_reference.targets(self.beta_task.cycle)
        state = self.telemetry
        return CycleObservationV1(
            cycle_index=cycle_index,
            decision_in_cycle=decision_in_cycle,
            cycle_decisions=cycle_decisions,
            progress=decision_in_cycle / cycle_decisions,
            battery_margin_kwh=state.battery_energy_kwh - targets["battery"],
            h2_margin_kg=state.h2_level_kg - targets["h2"],
            co2_margin_kg=state.co2_level_kg - targets["co2"],
            ch4_margin_kg=state.ch4_level_kg - targets["ch4"],
        )

    @property
    def cyclic_market_controller_observation(
        self,
    ) -> CyclicMarketControllerObservationV1:
        issue = self.current_market_issue
        if self.decisions_taken < self.episode_decisions and (
            issue.issued_at_decision != self.decisions_taken
        ):
            raise RuntimeError("market issue is stale at a live decision boundary")
        return CyclicMarketControllerObservationV1(
            plant=self.controller_observation,
            market=issue,
            cycle=self.cycle_observation,
            economic_assumptions_sha256=self.economic_assumptions.sha256,
            beta_task_sha256=self.beta_task.sha256,
        )

    @property
    def economic_controller_observation(  # type: ignore[override]
        self,
    ) -> CyclicMarketControllerObservationV1:
        """Route generic economic callers to the beta information contract.

        Returning the richer typed object makes an alpha-only economic MPC fail
        its input type check instead of silently receiving the deterministic
        baseline tariff and bypassing the common market forecast.
        """

        return self.cyclic_market_controller_observation

    def _opportunity_price_eur_per_mwh(
        self,
        *,
        time_h: float,
        decision_index: int,
    ) -> float:
        del time_h
        return self.market_episode.price_at_decision(decision_index)

    def _economic_price_context(
        self,
        *,
        time_h: float,
        decision_index: int,
    ) -> dict[str, Any]:
        del time_h
        return self.market_episode.issue_at_decision(decision_index).to_dict()

    def _finish_economic_step(  # type: ignore[override]
        self,
        start: Any,
        result: tuple[NDArray[np.generic], float, bool, bool, dict[str, Any]],
    ) -> tuple[NDArray[np.generic], float, bool, bool, dict[str, Any]]:
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = super()._finish_economic_step(
            start,
            result,
        )
        cycle_decisions = self.beta_task.cycle.cycle_decisions(self.episode_config)
        settlement: CycleSettlementV1 | None = None
        if self.decisions_taken > 0 and self.decisions_taken % cycle_decisions == 0:
            expected_cycle = self.decisions_taken // cycle_decisions
            if len(self._cycle_settlements) == expected_cycle - 1:
                settlement = CycleSettlementV1.evaluate(
                    cycle_index=expected_cycle,
                    boundary_decision=self.decisions_taken,
                    state=self.simulator.state,
                    reference=self.cycle_reference,
                    config=self.beta_task.cycle,
                )
                self._cycle_settlements.append(settlement)
            elif len(self._cycle_settlements) != expected_cycle:
                raise RuntimeError("cycle settlements lost exact boundary cardinality")
        info["cycle_constraint"] = self._cycle_constraint_info(settlement=settlement)
        info["environment_id"] = beta_spaces.ENVIRONMENT_ID
        info["environment_contract_digest"] = beta_spaces.ENVIRONMENT_CONTRACT_DIGEST
        info["stage"] = beta_spaces.STAGE
        return observation, reward, terminated, truncated, info

    def _cycle_constraint_info(
        self,
        *,
        settlement: CycleSettlementV1 | None,
    ) -> dict[str, Any]:
        return {
            "scalar_reward_contains_constraint_penalty": False,
            "current": self.cycle_observation.to_dict(),
            "settled_this_step": None if settlement is None else settlement.to_dict(),
            "settled_cycle_count": len(self._cycle_settlements),
            "all_settled_cycles_feasible": all(
                item.feasible for item in self._cycle_settlements
            ),
            "cumulative_constraint_cost": math.fsum(
                item.total_constraint_cost for item in self._cycle_settlements
            ),
        }

    def _observe(self) -> NDArray[np.float32]:
        physical = np.asarray(PlantSetpointEnvV2._observe(self), dtype=np.float32)
        market = np.asarray(self.current_market_issue.normalized(), dtype=np.float32)
        cycle = np.asarray(
            self.cycle_observation.normalized(self.config), dtype=np.float32
        )
        observation = np.concatenate((physical, market, cycle)).astype(
            np.float32,
            copy=False,
        )
        if observation.shape != (beta_spaces.OBSERVATION_SIZE,):
            raise RuntimeError("beta observation does not match its declared contract")
        if not self.observation_space.contains(observation):
            raise RuntimeError("beta observation lies outside its declared space")
        return observation

    def _public_info(self) -> dict[str, Any]:
        info = cast(dict[str, Any], PlantSetpointEnvV2._public_info(self))
        info.update(
            {
                "environment_id": beta_spaces.ENVIRONMENT_ID,
                "environment_contract_digest": beta_spaces.ENVIRONMENT_CONTRACT_DIGEST,
                "checkpoint_contract_digest": self.checkpoint_contract_digest,
                "stage": beta_spaces.STAGE,
                "market": self.current_market_issue.to_dict(),
                "cycle_constraint": self._cycle_constraint_info(settlement=None),
                "economic_context": {
                    "assumptions_sha256": self.economic_assumptions.sha256,
                    "realized_price_eur_per_mwh": (
                        self.current_market_issue.current_eur_per_mwh
                    ),
                    "price_source": "causal stochastic-market issue",
                },
                "economic_metrics": self.economic_metrics.to_dict(),
            }
        )
        return info

    def audit_manifest(self) -> dict[str, Any]:
        """Bind every private beta source while keeping it outside Gym info."""

        if self._episode is None:
            raise RuntimeError("reset() must be called before requesting a manifest")
        beta_exogenous = {
            "alpha_physical_exogenous_episode_sha256": (
                self._episode.exogenous_episode_sha256
            ),
            "market_episode_sha256": self.market_episode.episode_sha256,
            "disturbance_episode_sha256": self.disturbance_episode.episode_sha256,
            "cycle_task_sha256": self.beta_task.sha256,
        }
        immutable = {
            "manifest_schema_version": BETA_MANIFEST_SCHEMA_VERSION,
            "environment": beta_spaces.describe(),
            "checkpoint_contract": self.checkpoint_contract,
            "objective_version": ECONOMIC_OBJECTIVE_VERSION,
            "diagnostic_legacy_objective_version": OBJECTIVE_VERSION,
            "episode_config": self.episode_config.to_dict(),
            "beta_task": self.beta_task.to_dict(),
            "economic_assumptions": self.economic_assumptions.to_dict(),
            "plant_config": self.config.to_dict(),
            "constraints": self.constraints.to_dict(),
            "diagnostic_legacy_reward_weights": self.weights.to_dict(),
            "episode": self._episode.audit_manifest(),
            "market": self.market_episode.private_audit_manifest(),
            "disturbances": self.disturbance_episode.private_audit_manifest(),
            "cycle_reference": self.cycle_reference.to_dict(),
            "beta_exogenous": beta_exogenous,
            "provenance_gaps": [
                "source revision and dependency-lock digest must be added by the experiment runner"
            ],
        }
        return {
            "manifest_sha256": canonical_sha256(immutable),
            "alpha_physical_exogenous_episode_sha256": (
                self._episode.exogenous_episode_sha256
            ),
            "beta_exogenous_episode_sha256": canonical_sha256(beta_exogenous),
            **immutable,
            "delivery_accounting": self.simulator.delivery_audit(),
            "economic_accounting": deepcopy(self.economic_metrics.to_dict()),
            "cycle_accounting": [item.to_dict() for item in self._cycle_settlements],
        }


__all__ = [
    "BETA_CHECKPOINT_CONTRACT_SCHEMA_VERSION",
    "BETA_MANIFEST_SCHEMA_VERSION",
    "BETA_ROOT_STREAM_NAMES",
    "PlantCyclicMarketEnvV2Beta",
    "describe_beta_checkpoint_contract",
]

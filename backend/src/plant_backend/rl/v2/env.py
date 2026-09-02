"""Delivery-only Environment V2 alpha around the unchanged plant simulator.

One policy decision is held for several one-minute physics steps.  The guard is
re-run before every physical step because storage and availability can change
while the requested action remains constant.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from plant_sim import Action, DeliverySettlement, PlantConfig, PlantSimulator, PlantState
from plant_sim.state import CRITICAL, OFFLINE

from ...controllers.base import Constraints
from ...safety.validation import validate
from ..actions import ActionAdapter, DecodedAction
from . import spaces
from .config import EpisodeConfigV2, EpisodeV2, sample_episode
from .metrics import EpisodeMetricsV2
from .observation import DeliveryObservation, ObservationEncoderV2
from .reward import OBJECTIVE_VERSION, IntervalOutcome, RewardWeightsV2, compute


class PlantEnvV2(gym.Env[np.ndarray, np.ndarray]):
    """Explicit V2 alpha: recurring delivery service, not tank filling."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        config: PlantConfig | None = None,
        episode: EpisodeConfigV2 | None = None,
        constraints: Constraints | None = None,
        weights: RewardWeightsV2 | None = None,
    ) -> None:
        self.config = config or PlantConfig()
        self.episode_config = (episode or EpisodeConfigV2()).validate(self.config)
        self.constraints = constraints or Constraints()
        self.weights = (weights or RewardWeightsV2()).validate()

        self.simulator = PlantSimulator(self.config)
        self.encoder = ObservationEncoderV2(self.config)
        self.adapter = ActionAdapter(self.config)
        self.observation_space = Box(
            low=spaces.observation_low(),
            high=spaces.observation_high(),
            dtype=np.float32,
        )
        self.action_space = Box(
            low=spaces.action_low(),
            high=spaces.action_high(),
            shape=(spaces.ACTION_SIZE,),
            dtype=np.float32,
        )

        self.episode: EpisodeV2 | None = None
        self.metrics = EpisodeMetricsV2()
        self.physics_steps_taken = 0
        self.decisions_taken = 0
        self.episode_physics_steps = self.episode_config.physics_steps(self.config)
        self.episode_decisions = self.episode_config.decision_steps(self.config)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Draw a replayable episode without returning its private future tape."""
        super().reset(seed=seed)
        episode_seed = (
            int(seed)
            if seed is not None
            else int(self.np_random.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        )
        drawn = sample_episode(self.episode_config, self.config, seed=episode_seed)
        self.episode = drawn
        self.episode_physics_steps = drawn.physics_steps
        self.episode_decisions = drawn.physics_steps // drawn.decision_stride
        self.simulator.load_scenario(drawn.scenario)
        # Evaluation episodes are immutable after reset. Allowing schedule
        # replacement would erase settlement history while retaining the
        # originally sampled manifest identity.
        self.simulator.set_delivery_schedule(drawn.commitments, freeze=True)
        self.encoder.reset()
        self.metrics = EpisodeMetricsV2()
        self.physics_steps_taken = 0
        self.decisions_taken = 0
        return self._observe(), self._public_info()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        decoded = self.adapter.decode(
            action,
            constraints=self.constraints,
            solar_available_kw=self.telemetry.solar_available_kw,
        )
        return self._apply(decoded)

    def step_action(
        self,
        action: Action,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Controller door sharing the exact V2 guard, reward, and metrics path."""
        safe = Action.sanitised(action)
        bus_kw = max(0.0, self.telemetry.solar_available_kw)
        decoded = DecodedAction(
            action=safe,
            clipped=safe != action,
            requested_kw={
                "battery": safe.battery_share * bus_kw,
                "electrolyser": safe.electrolyser_share * bus_kw
                + safe.battery_support_kw,
                "dac": safe.dac_share * bus_kw,
                "methanator": safe.methanator_share * bus_kw,
                "battery_support": safe.battery_support_kw,
            },
        )
        return self._apply(decoded)

    def _apply(
        self,
        decoded: DecodedAction,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.episode is None:
            raise RuntimeError("reset() must be called before step()")
        remaining = self.episode_physics_steps - self.physics_steps_taken
        inner_steps = min(self.episode.decision_stride, remaining)
        if inner_steps <= 0:
            raise RuntimeError("episode is already truncated; call reset()")

        before = self.simulator.state
        min_soc = before.battery_soc
        battery_discharged_kwh = 0.0
        shutdown_steps = 0
        adjustments: list[str] = []
        corrected_steps = 0
        settlements: list[DeliverySettlement] = []
        requested_kg = delivered_kg = unmet_kg = 0.0
        last_applied = Action.idle()

        for _ in range(inner_steps):
            validated = validate(
                decoded.action,
                self.telemetry,
                constraints=self.constraints,
                config=self.config,
            )
            last_applied = validated.action
            if validated.adjustments:
                corrected_steps += 1
                adjustments.extend(validated.adjustments)
            after_step = self.simulator.step(validated.action)
            delivery = self.simulator.delivery_step
            settlements.extend(delivery.settlements)
            requested_kg += delivery.requested_kg
            delivered_kg += delivery.delivered_kg
            unmet_kg += delivery.unmet_kg
            battery_discharged_kwh += max(0.0, -after_step.battery_power_kw) * self.config.dt_h
            min_soc = min(min_soc, after_step.battery_soc)
            shutdown_steps += int(after_step.plant_status in (CRITICAL, OFFLINE))

        after = self.simulator.state
        self.physics_steps_taken += inner_steps
        self.decisions_taken += 1
        truncated = self.physics_steps_taken >= self.episode_physics_steps
        outcome = IntervalOutcome(
            hours=inner_steps * self.config.dt_h,
            physical_steps=inner_steps,
            requested_kg=requested_kg,
            delivered_kg=delivered_kg,
            unmet_kg=unmet_kg,
            produced_kg=max(
                0.0, after.ch4_produced_total_kg - before.ch4_produced_total_kg
            ),
            solar_kwh=max(0.0, after.energy_produced_kwh - before.energy_produced_kwh),
            curtailed_kwh=max(
                0.0, after.energy_curtailed_kwh - before.energy_curtailed_kwh
            ),
            battery_discharged_kwh=battery_discharged_kwh,
            min_battery_soc=min_soc,
            shutdown_steps=shutdown_steps,
            corrections=len(adjustments),
            out_of_range=decoded.clipped,
            terminal_ch4_kg=after.ch4_level_kg,
            reference_delivery_kg=self.episode_config.delivery_amount_kg,
        )
        breakdown = compute(
            outcome,
            config=self.config,
            constraints=self.constraints,
            terminal=truncated,
            weights=self.weights,
        )
        self.metrics.record(
            outcome,
            breakdown,
            settlements,
            corrected_physics_steps=corrected_steps,
        )
        info = {
            **self._public_info(),
            "requested": decoded.to_dict(),
            "applied_action": last_applied.to_dict(),
            "guard": {
                "interventions": len(adjustments),
                "corrected_physics_steps": corrected_steps,
                "out_of_range": decoded.clipped,
            },
            "adjustments": adjustments,
            "delivery": {
                "commitments_settled": len(settlements),
                "requested_kg": requested_kg,
                "delivered_kg": delivered_kg,
                "unmet_kg": unmet_kg,
            },
            "reward": breakdown.to_dict(),
            "metrics": self.metrics.to_dict(),
            "plant_status": after.plant_status,
        }
        return self._observe(), breakdown.total, False, truncated, info

    @property
    def telemetry(self) -> PlantState:
        return self.simulator.get_telemetry()

    @property
    def progress(self) -> float:
        if self.episode_decisions <= 0:
            return 0.0
        return min(1.0, self.decisions_taken / self.episode_decisions)

    def _observe(self) -> np.ndarray:
        next_delivery = self.simulator.next_delivery
        current = self.simulator.delivery_step
        delivery = DeliveryObservation(
            steps_to_next=(
                None
                if next_delivery is None
                else max(0, next_delivery.due_step - self.simulator.step_index)
            ),
            delivery_period_steps=self.episode_config.delivery_stride(self.config),
            next_requested_kg=(
                0.0 if next_delivery is None else next_delivery.requested_kg
            ),
            cumulative_requested_kg=current.cumulative_requested_kg,
            cumulative_delivered_kg=current.cumulative_delivered_kg,
        )
        return self.encoder.encode(
            self.telemetry,
            constraints=self.constraints,
            progress=self.progress,
            delivery=delivery,
        )

    def _public_info(self) -> dict[str, Any]:
        next_delivery = self.simulator.next_delivery
        return {
            "environment_id": spaces.ENVIRONMENT_ID,
            "environment_contract_digest": spaces.ENVIRONMENT_CONTRACT_DIGEST,
            "stage": spaces.STAGE,
            "clock": self.telemetry.clock,
            "decision_index": self.decisions_taken,
            "physics_step": self.physics_steps_taken,
            "next_delivery": (
                None
                if next_delivery is None
                else {
                    "steps_until_due": next_delivery.due_step - self.simulator.step_index,
                    "requested_kg": next_delivery.requested_kg,
                }
            ),
        }

    def audit_manifest(self) -> dict[str, Any]:
        """Private reproducibility record, deliberately separate from Gym info."""
        if self.episode is None:
            raise RuntimeError("reset() must be called before requesting a manifest")
        immutable = {
            "manifest_schema_version": 1,
            "environment": spaces.describe(),
            "objective_version": OBJECTIVE_VERSION,
            "episode_config": self.episode_config.to_dict(),
            "plant_config": self.config.to_dict(),
            "constraints": self.constraints.to_dict(),
            "reward_weights": self.weights.to_dict(),
            "episode": self.episode.audit_manifest(),
            "provenance_gaps": [
                "source revision and dependency-lock digest must be added by the experiment runner"
            ],
        }
        canonical = json.dumps(
            immutable,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return {
            "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
            **immutable,
            "delivery_accounting": self.simulator.delivery_audit(),
        }


__all__ = ["PlantEnvV2"]

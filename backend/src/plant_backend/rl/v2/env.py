"""Weather-aware Environment V2 alpha around the shared plant simulator.

One policy decision is held for several one-minute physics steps.  The guard is
re-run before every physical step because storage and availability can change
while the requested action remains constant.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from numbers import Integral
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
from .accounting import IntervalPowerAccounting
from .boundary import ControlStateV2, IntervalTelemetryAccumulator
from .config import EpisodeConfigV2, EpisodeV2, sample_episode
from .contract import describe_checkpoint_contract
from .metrics import EpisodeMetricsV2
from .observation import DeliveryObservation, ObservationEncoderV2
from .reward import OBJECTIVE_VERSION, IntervalOutcome, RewardWeightsV2, compute
from .weather import ForecastIssueV2


class PlantEnvV2(gym.Env[np.ndarray, np.ndarray]):
    """Explicit V2 alpha: delivery service under uncertain solar resource."""

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
        self._checkpoint_contract = describe_checkpoint_contract(
            plant=self.config,
            episode=self.episode_config,
            constraints=self.constraints,
            weights=self.weights,
        )

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

        # Full weather truth and future forecast vintages are deliberately
        # private. Controllers receive only ``current_forecast``.
        self._episode: EpisodeV2 | None = None
        self.control_state: ControlStateV2 | None = None
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
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0
        ):
            raise ValueError("seed must be a non-negative integer or None")
        super().reset(seed=seed)
        episode_seed = (
            int(seed)
            if seed is not None
            else int(self.np_random.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        )
        drawn = sample_episode(self.episode_config, self.config, seed=episode_seed)
        self._episode = drawn
        self.episode_physics_steps = drawn.physics_steps
        self.episode_decisions = drawn.physics_steps // drawn.decision_stride
        self.simulator.load_scenario(
            drawn.scenario,
            solar_factor_tape=drawn.weather.tape,
        )
        # Evaluation episodes are immutable after reset. Allowing schedule
        # replacement would erase settlement history while retaining the
        # originally sampled manifest identity.
        self.simulator.set_delivery_schedule(drawn.commitments, freeze=True)
        self.encoder.reset()
        self.metrics = EpisodeMetricsV2()
        self.physics_steps_taken = 0
        self.decisions_taken = 0
        self.control_state = ControlStateV2(boundary=self.telemetry)
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
        if self._episode is None:
            raise RuntimeError("reset() must be called before step()")
        remaining = self.episode_physics_steps - self.physics_steps_taken
        inner_steps = min(self._episode.decision_stride, remaining)
        if inner_steps <= 0:
            raise RuntimeError("episode is already terminated; call reset()")

        before = self.simulator.state
        min_soc = before.battery_soc
        battery_discharged_kwh = 0.0
        shutdown_steps = 0
        adjustments: list[str] = []
        corrected_steps = 0
        settlements: list[DeliverySettlement] = []
        requested_kg = delivered_kg = unmet_kg = 0.0
        last_applied = Action.idle()
        power = IntervalPowerAccounting()
        interval_telemetry = IntervalTelemetryAccumulator()

        for _ in range(inner_steps):
            step_before = self.telemetry
            validated = validate(
                decoded.action,
                step_before,
                constraints=self.constraints,
                config=self.config,
                dispatch_solar_available_kw=(
                    self.simulator.next_dispatch_solar_kw()
                ),
            )
            last_applied = validated.action
            if validated.adjustments:
                corrected_steps += 1
                adjustments.extend(validated.adjustments)
            after_step = self.simulator.step(validated.action)
            interval_telemetry.record(self.telemetry, dt_h=self.config.dt_h)
            power.record(
                decoded.action,
                validated.action,
                after=after_step,
                dt_h=self.config.dt_h,
            )
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
        self.control_state = ControlStateV2(
            boundary=self.telemetry,
            previous_interval=interval_telemetry.summary(),
        )
        terminated = self.physics_steps_taken >= self.episode_physics_steps
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
            guard_intervention_kwh=power.guard_intervention_l1_kwh,
            out_of_range=decoded.clipped,
            terminal_ch4_kg=after.ch4_level_kg,
            reference_delivery_kg=self.episode_config.delivery_amount_kg,
            reference_period_h=self.episode_config.delivery_period_h,
        )
        breakdown = compute(
            outcome,
            config=self.config,
            constraints=self.constraints,
            terminal=terminated,
            weights=self.weights,
        )
        self.metrics.record(
            outcome,
            breakdown,
            settlements,
            corrected_physics_steps=corrected_steps,
        )
        requested = decoded.to_dict()
        requested["requested_kw_basis"] = (
            "latest completed one-minute solar sample at the decision boundary; "
            "authoritative interval energy is reported in interval_power"
        )
        info = {
            **self._public_info(),
            "requested": requested,
            "applied_action": last_applied.to_dict(),
            "guard": {
                "interventions": len(adjustments),
                "corrected_physics_steps": corrected_steps,
                "out_of_range": decoded.clipped,
                "intervention_l1_kwh": power.guard_intervention_l1_kwh,
            },
            "interval_power": power.to_dict(),
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
        # The contract horizon is part of the finite-horizon MDP and pays a
        # one-time inventory salvage value. It is therefore a true termination,
        # not an external time-limit truncation that an RL library may bootstrap.
        return self._observe(), breakdown.total, terminated, False, info

    @property
    def telemetry(self) -> PlantState:
        raw = self.simulator.get_telemetry()
        # The simulator retains the private resource factor even when the
        # clear-sky envelope is zero. A PV meter cannot observe that latent
        # cloud state at night, so V2 exposes only the measured power ratio.
        measured_factor = (
            raw.solar_available_kw / raw.solar_potential_kw
            if raw.solar_potential_kw > 1e-12
            else 0.0
        )
        return replace(raw, solar_output_factor=measured_factor)

    @property
    def current_forecast(self) -> ForecastIssueV2:
        """Latest immutable issuance available to every controller family."""
        if self._episode is None:
            raise RuntimeError("reset() must be called before requesting a forecast")
        return self._episode.weather.current_issue(self.physics_steps_taken)

    @property
    def progress(self) -> float:
        if self.episode_decisions <= 0:
            return 0.0
        return min(1.0, self.decisions_taken / self.episode_decisions)

    def _observe(self) -> np.ndarray:
        if self._episode is None or self.control_state is None:
            raise RuntimeError("reset() must be called before observing")
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
            self.control_state,
            constraints=self.constraints,
            progress=self.progress,
            delivery=delivery,
            forecast=self.current_forecast,
        )

    def _public_info(self) -> dict[str, Any]:
        if self._episode is None:
            raise RuntimeError("reset() must be called before requesting public info")
        next_delivery = self.simulator.next_delivery
        return {
            "environment_id": spaces.ENVIRONMENT_ID,
            "environment_contract_digest": spaces.ENVIRONMENT_CONTRACT_DIGEST,
            "checkpoint_contract_digest": self.checkpoint_contract_digest,
            "stage": spaces.STAGE,
            "clock": self.telemetry.clock,
            "decision_index": self.decisions_taken,
            "physics_step": self.physics_steps_taken,
            "forecast": self.current_forecast.to_dict(),
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
        if self._episode is None:
            raise RuntimeError("reset() must be called before requesting a manifest")
        immutable = {
            "manifest_schema_version": 2,
            "environment": spaces.describe(),
            "checkpoint_contract": self.checkpoint_contract,
            "objective_version": OBJECTIVE_VERSION,
            "episode_config": self.episode_config.to_dict(),
            "plant_config": self.config.to_dict(),
            "constraints": self.constraints.to_dict(),
            "reward_weights": self.weights.to_dict(),
            "episode": self._episode.audit_manifest(),
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

    @property
    def checkpoint_contract_digest(self) -> str:
        return str(self._checkpoint_contract["checkpoint_contract_digest"])

    @property
    def checkpoint_contract(self) -> dict[str, Any]:
        """Detached full identity that every future V2 checkpoint must declare."""
        return deepcopy(self._checkpoint_contract)


__all__ = ["PlantEnvV2"]

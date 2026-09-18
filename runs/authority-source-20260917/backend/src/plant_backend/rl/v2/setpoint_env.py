"""Opt-in Gym environment whose policy controls four absolute process targets."""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray
from plant_sim import PlantConfig

from ...controllers.base import Constraints
from .config import EpisodeConfigV2
from .contract import checkpoint_contract_payload, contract_digest
from . import setpoint_spaces
from .env import PlantEnvV2
from .mpc_execution import public_observation_sha256
from .reward import OBJECTIVE_VERSION, RewardWeightsV2
from .setpoint_command import issue_process_setpoint_command
from .setpoint_control import ProcessSetpointActionAdapterV1


SETPOINT_CHECKPOINT_CONTRACT_SCHEMA_VERSION = 4
NORMALIZED_SETPOINT_ISSUER_ID = "normalized-process-setpoint-policy-v1"


def describe_setpoint_checkpoint_contract(
    *,
    plant: PlantConfig,
    episode: EpisodeConfigV2,
    constraints: Constraints,
    weights: RewardWeightsV2,
) -> dict[str, Any]:
    """Return a checkpoint identity that cannot alias the historical five-action task."""

    payload = checkpoint_contract_payload(
        plant=plant,
        episode=episode,
        constraints=constraints,
        weights=weights,
    )
    payload["schema_version"] = SETPOINT_CHECKPOINT_CONTRACT_SCHEMA_VERSION
    payload["environment_id"] = setpoint_spaces.ENVIRONMENT_ID
    payload["interface_contract"] = setpoint_spaces.contract_payload()
    payload["interface_contract_digest"] = setpoint_spaces.ENVIRONMENT_CONTRACT_DIGEST
    payload["common_setpoint_authority"] = {
        "normalized_action_schema": "normalized-process-setpoint-action-v1",
        "target_schema": "process-setpoint-targets-v1",
        "command_schema": "process-setpoint-command-v1",
        "minute_dispatch_schema": "process-setpoint-minute-dispatch-v1",
        "minute_application_schema": "process-setpoint-minute-application-v1",
        "interval_execution_schema": "process-setpoint-interval-execution-v1",
        "support_semantics": (
            "dispatcher-owned contemporaneous electrolyser-target shortfall, bounded "
            "by reserve-aware safety availability"
        ),
    }
    return {"checkpoint_contract_digest": contract_digest(payload), **payload}


class PlantSetpointEnvV2(PlantEnvV2):  # type: ignore[misc]
    """V2 dynamics and observations with a four-target policy action contract."""

    def __init__(
        self,
        *,
        config: PlantConfig | None = None,
        episode: EpisodeConfigV2 | None = None,
        constraints: Constraints | None = None,
        weights: RewardWeightsV2 | None = None,
    ) -> None:
        super().__init__(
            config=config,
            episode=episode,
            constraints=constraints,
            weights=weights,
        )
        self.setpoint_adapter = ProcessSetpointActionAdapterV1()
        self.action_space = Box(
            low=setpoint_spaces.action_low(),
            high=setpoint_spaces.action_high(),
            shape=(setpoint_spaces.ACTION_SIZE,),
            dtype=np.float32,
        )
        self._checkpoint_contract = describe_setpoint_checkpoint_contract(
            plant=self.config,
            episode=self.episode_config,
            constraints=self.constraints,
            weights=self.weights,
        )

    def step(
        self,
        action: NDArray[np.generic],
    ) -> tuple[NDArray[np.generic], float, bool, bool, dict[str, Any]]:
        observation = self.controller_observation
        decoded = self.setpoint_adapter.decode(
            action,
            constraints=self.constraints,
            config=self.config,
        )
        command = issue_process_setpoint_command(
            decoded.targets,
            decision_index=self.decisions_taken,
            observation_sha256=public_observation_sha256(observation),
            hold_duration_s=self.episode_config.decision_interval_s,
            issuer_id=NORMALIZED_SETPOINT_ISSUER_ID,
            source_schema=decoded.schema,
            source_sha256=decoded.sha256,
            constraints=self.constraints,
            config=self.config,
        )
        next_observation, reward, terminated, truncated, info = (
            self.step_process_setpoints(
                command,
                controller_invalid=decoded.clipped,
            )
        )
        dispatcher_elapsed_ns = info.pop("dispatcher_elapsed_ns")
        if not isinstance(dispatcher_elapsed_ns, list):
            raise RuntimeError("common dispatcher timing evidence is malformed")
        # Wall-clock timings are intentionally omitted from the Gym info contract:
        # they make identical seeded transitions compare unequal. Dedicated
        # evaluation runners may still use ``step_process_setpoints`` directly.
        info["dispatcher_call_count"] = len(dispatcher_elapsed_ns)
        info["source_decision"] = decoded.to_dict()
        return next_observation, reward, terminated, truncated, info

    def _public_info(self) -> dict[str, Any]:
        info = cast(dict[str, Any], super()._public_info())
        info.update(
            {
                "environment_id": setpoint_spaces.ENVIRONMENT_ID,
                "environment_contract_digest": (
                    setpoint_spaces.ENVIRONMENT_CONTRACT_DIGEST
                ),
                "stage": setpoint_spaces.STAGE,
            }
        )
        return info

    def audit_manifest(self) -> dict[str, Any]:
        """Bind the distinct setpoint interface without changing episode sampling."""

        inherited = super().audit_manifest()
        immutable = {
            "manifest_schema_version": 4,
            "environment": setpoint_spaces.describe(),
            "checkpoint_contract": self.checkpoint_contract,
            "objective_version": OBJECTIVE_VERSION,
            "episode_config": inherited["episode_config"],
            "plant_config": inherited["plant_config"],
            "constraints": inherited["constraints"],
            "reward_weights": inherited["reward_weights"],
            "episode": inherited["episode"],
            "provenance_gaps": inherited["provenance_gaps"],
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
            "exogenous_episode_sha256": inherited["exogenous_episode_sha256"],
            **immutable,
            "delivery_accounting": inherited["delivery_accounting"],
        }


__all__ = [
    "NORMALIZED_SETPOINT_ISSUER_ID",
    "SETPOINT_CHECKPOINT_CONTRACT_SCHEMA_VERSION",
    "PlantSetpointEnvV2",
    "describe_setpoint_checkpoint_contract",
]

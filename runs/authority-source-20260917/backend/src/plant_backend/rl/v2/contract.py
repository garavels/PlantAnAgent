"""Full checkpoint-compatibility identity for Environment V2.

The static digest in :mod:`spaces` identifies vector layout only.  A V2 policy
also depends on timing, dynamics, episode distribution, constraints, and reward
semantics.  This module makes that wider contract canonical and refuses legacy
or mismatched V2 checkpoint metadata instead of guessing compatibility.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import plant_sim
from plant_sim import COMMITMENT_DYNAMICS_VERSION, PlantConfig, SOLAR_FORCING_VERSION

from ...controllers.base import Constraints
from . import spaces
from .config import CLEAR_SKY_ENVELOPE_ID, EpisodeConfigV2
from .reward import OBJECTIVE_VERSION, RewardWeightsV2
from .weather import WEATHER_GENERATOR_VERSION

CHECKPOINT_CONTRACT_SCHEMA_VERSION = 3
FLOW_SAMPLING_SEMANTICS = (
    "timestamp and inventories are current after the preceding decision interval; "
    "solar availability/potential/output factor and equipment health/online state "
    "are the final one-minute dispatch sample in that interval; power, load, and "
    "production are time-weighted means over the whole interval (reset uses the "
    "initial boundary sample because no interval exists); electrolyser commitment "
    "mode, timers, prior accepted power, and ramp headroom are current boundary state"
)
GUARD_SEMANTICS = (
    "controllers decide from causal boundary telemetry; before every physical "
    "step the actuator guard applies absolute kW caps against the simulator-owned "
    "solar bus that same step will dispatch"
)
EPISODE_END_SEMANTICS = (
    "the configured contract horizon is an intrinsic finite-horizon termination "
    "(terminated=true, truncated=false); terminal inventory is valued once and "
    "value functions must not bootstrap beyond the final observation"
)
TIME_SEMANTICS = (
    "observe boundary t; hold one decision over [t,t+decision_interval); "
    "settle a deadline after production in its final physical interval; expose only "
    "the causal forecast issued at t with half-open lead windows"
)


class CheckpointContractError(ValueError):
    """A checkpoint cannot be shown to implement this exact V2 task."""


def checkpoint_contract_payload(
    *,
    plant: PlantConfig,
    episode: EpisodeConfigV2,
    constraints: Constraints,
    weights: RewardWeightsV2,
) -> dict[str, Any]:
    """Return every runtime setting that can change learned policy semantics."""
    episode.validate(plant)
    weights.validate()
    return {
        "schema_version": CHECKPOINT_CONTRACT_SCHEMA_VERSION,
        "environment_id": spaces.ENVIRONMENT_ID,
        "interface_contract": spaces.contract_payload(),
        "interface_contract_digest": spaces.ENVIRONMENT_CONTRACT_DIGEST,
        "timing": {
            "physics_timestep_s": plant.dt_s,
            "decision_interval_s": episode.decision_interval_s,
            "time_semantics": TIME_SEMANTICS,
            "flow_sampling_semantics": FLOW_SAMPLING_SEMANTICS,
            "guard_semantics": GUARD_SEMANTICS,
            "episode_end_semantics": EPISODE_END_SEMANTICS,
        },
        "dynamics_versions": {
            "plant_sim": plant_sim.__version__,
            "solar_forcing": SOLAR_FORCING_VERSION,
            "clear_sky_envelope": CLEAR_SKY_ENVELOPE_ID,
            "weather_generator": WEATHER_GENERATOR_VERSION,
            "electrolyser_commitment": COMMITMENT_DYNAMICS_VERSION,
        },
        "episode_distribution": episode.to_dict(),
        "plant_config": plant.to_dict(),
        "constraints": constraints.to_dict(),
        "objective": {
            "version": OBJECTIVE_VERSION,
            "reward_weights": weights.to_dict(),
        },
    }


def contract_digest(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def describe_checkpoint_contract(
    *,
    plant: PlantConfig,
    episode: EpisodeConfigV2,
    constraints: Constraints,
    weights: RewardWeightsV2,
) -> dict[str, Any]:
    payload = checkpoint_contract_payload(
        plant=plant,
        episode=episode,
        constraints=constraints,
        weights=weights,
    )
    return {
        "checkpoint_contract_digest": contract_digest(payload),
        **payload,
    }


def require_compatible_checkpoint(
    metadata: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> None:
    """Fail unless metadata declares the exact current full V2 contract."""
    environment_id = metadata.get("environment_id")
    if environment_id != expected.get("environment_id"):
        raise CheckpointContractError(
            f"checkpoint environment {environment_id!r} is incompatible with "
            f"{expected.get('environment_id')!r}"
        )
    declared = metadata.get("checkpoint_contract_digest")
    if not isinstance(declared, str) or not declared:
        raise CheckpointContractError(
            "V2 checkpoint metadata is missing checkpoint_contract_digest"
        )
    current = expected.get("checkpoint_contract_digest")
    if declared != current:
        raise CheckpointContractError(
            f"V2 checkpoint contract mismatch: checkpoint={declared}, current={current}"
        )


__all__ = [
    "CHECKPOINT_CONTRACT_SCHEMA_VERSION",
    "EPISODE_END_SEMANTICS",
    "FLOW_SAMPLING_SEMANTICS",
    "GUARD_SEMANTICS",
    "TIME_SEMANTICS",
    "CheckpointContractError",
    "checkpoint_contract_payload",
    "contract_digest",
    "describe_checkpoint_contract",
    "require_compatible_checkpoint",
]

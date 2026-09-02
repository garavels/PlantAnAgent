"""Explicit Environment V2 research contract.

This namespace is intentionally parallel to :mod:`plant_backend.rl`.  Importing
``PlantEnv`` from the parent package continues to mean the frozen V1 task; V2
callers must request ``PlantEnvV2`` by name.

The first alpha contains recurring methane delivery commitments and a slower
controller cadence.  Forecast uncertainty and equipment commitment dynamics
are later, separately switchable increments.
"""

from .config import EpisodeConfigV2, EpisodeV2, sample_episode
from .env import PlantEnvV2
from .metrics import EpisodeMetricsV2
from .reward import RewardWeightsV2
from .spaces import (
    ACTION_NAMES,
    ACTION_SIZE,
    ENVIRONMENT_CONTRACT_DIGEST,
    ENVIRONMENT_ID,
    OBSERVATION_NAMES,
    OBSERVATION_SIZE,
    SPEC_VERSION,
    describe,
)

__all__ = [
    "ACTION_NAMES",
    "ACTION_SIZE",
    "ENVIRONMENT_CONTRACT_DIGEST",
    "ENVIRONMENT_ID",
    "EpisodeConfigV2",
    "EpisodeMetricsV2",
    "EpisodeV2",
    "OBSERVATION_NAMES",
    "OBSERVATION_SIZE",
    "PlantEnvV2",
    "RewardWeightsV2",
    "SPEC_VERSION",
    "describe",
    "sample_episode",
]

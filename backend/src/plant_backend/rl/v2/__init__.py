"""Explicit Environment V2 research contract.

This namespace is intentionally parallel to :mod:`plant_backend.rl`.  Importing
``PlantEnv`` from the parent package continues to mean the frozen V1 task; V2
callers must request ``PlantEnvV2`` by name.

The current alpha contains recurring methane delivery commitments, a slower
controller cadence, seeded solar-resource truth, and causal forecast vintages.
Equipment commitment dynamics remain a later, separately switchable increment.
"""

from .config import EpisodeConfigV2, EpisodeV2, sample_episode
from .boundary import ControlStateV2, IntervalSummaryV2
from .env import PlantEnvV2
from .evaluation import BaselineEvaluation, evaluate
from .metrics import EpisodeMetricsV2
from .reward import RewardWeightsV2
from .weather import ForecastIssueV2, WeatherConfigV2
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
    "BaselineEvaluation",
    "ControlStateV2",
    "EpisodeConfigV2",
    "EpisodeMetricsV2",
    "EpisodeV2",
    "ForecastIssueV2",
    "IntervalSummaryV2",
    "OBSERVATION_NAMES",
    "OBSERVATION_SIZE",
    "PlantEnvV2",
    "RewardWeightsV2",
    "SPEC_VERSION",
    "WeatherConfigV2",
    "describe",
    "evaluate",
    "sample_episode",
]

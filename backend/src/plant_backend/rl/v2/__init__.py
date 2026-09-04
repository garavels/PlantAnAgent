"""Explicit Environment V2 research contract.

This namespace is intentionally parallel to :mod:`plant_backend.rl`.  Importing
``PlantEnv`` from the parent package continues to mean the frozen V1 task; V2
callers must request ``PlantEnvV2`` by name.

The current alpha contains recurring methane delivery commitments, a slower
controller cadence, seeded solar-resource truth, causal forecast vintages, and
separately switchable electrolyser commitment dynamics.
"""

from .config import EpisodeConfigV2, EpisodeV2, sample_episode
from .boundary import ControlStateV2, IntervalSummaryV2
from .control import ControllerObservationV2, ControllerV2, DeliveryObservation
from .controllers import (
    ClimatologyControllerV2,
    DeliveryAwareControllerV2,
    DeliveryPlannerSettingsV2,
    IssuedForecastControllerV2,
    RiskAdjustedIssuedForecastControllerV2,
)
from .env import PlantEnvV2
from .evaluation import (
    BaselineEvaluation,
    CausalControllerEvaluationV2,
    ControllerEvaluationV2,
    evaluate,
    evaluate_controllers,
)
from .metrics import EpisodeMetricsV2
from .mpc_controller import ServiceMPCControllerV1
from .mpc_setpoint_controller import (
    ServiceMPCSetpointControllerV1,
    SetpointControllerV2,
)
from .mpc_setpoint_decision import (
    MPCSetpointDecisionEvidenceV1,
    MPCSetpointDecisionFailureV1,
    MPCSetpointDecisionV1,
)
from .mpc_schema import ServiceMPCSetpointSettingsV1, ServiceMPCSettingsV1
from .mpc_setpoint_execution import (
    MPCMinuteApplicationV1,
    MPCSetpointIntervalExecutionV1,
)
from .mpc_setpoints import (
    MPCMinuteDispatchV1,
    MPCProcessSetpointsV1,
    dispatch_mpc_setpoints,
)
from .reward import RewardWeightsV2
from .weather import ForecastIssueV2, PublicForecastIssueV2, WeatherConfigV2
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
    "CausalControllerEvaluationV2",
    "ControlStateV2",
    "ControllerObservationV2",
    "ControllerV2",
    "ControllerEvaluationV2",
    "ClimatologyControllerV2",
    "DeliveryAwareControllerV2",
    "DeliveryObservation",
    "DeliveryPlannerSettingsV2",
    "EpisodeConfigV2",
    "EpisodeMetricsV2",
    "EpisodeV2",
    "ForecastIssueV2",
    "PublicForecastIssueV2",
    "IntervalSummaryV2",
    "IssuedForecastControllerV2",
    "MPCMinuteDispatchV1",
    "MPCMinuteApplicationV1",
    "MPCProcessSetpointsV1",
    "MPCSetpointDecisionEvidenceV1",
    "MPCSetpointDecisionFailureV1",
    "MPCSetpointDecisionV1",
    "MPCSetpointIntervalExecutionV1",
    "OBSERVATION_NAMES",
    "OBSERVATION_SIZE",
    "PlantEnvV2",
    "RewardWeightsV2",
    "RiskAdjustedIssuedForecastControllerV2",
    "SPEC_VERSION",
    "ServiceMPCControllerV1",
    "ServiceMPCSetpointControllerV1",
    "ServiceMPCSetpointSettingsV1",
    "ServiceMPCSettingsV1",
    "SetpointControllerV2",
    "WeatherConfigV2",
    "describe",
    "dispatch_mpc_setpoints",
    "evaluate",
    "evaluate_controllers",
    "sample_episode",
]

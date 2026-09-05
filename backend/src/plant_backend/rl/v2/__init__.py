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
from .economic_env import PlantEconomicSetpointEnvV2
from .economics import (
    DayAheadTariffV1,
    EconomicAssumptionsV1,
    EconomicControllerObservationV1,
    EconomicEpisodeMetricsV1,
    EconomicIntervalOutcomeV1,
    EconomicRewardBreakdownV1,
    TariffObservationV1,
    ch4_equivalent_inventory_kg,
    compute_economic_reward,
)
from .economic_mpc_controller import TariffAwareEconomicMPCControllerV1
from .economic_mpc_model import EconomicMPCModelV1, build_economic_mpc_model
from .economic_mpc_schema import (
    EconomicMPCDecisionEvidenceV1,
    EconomicMPCDecisionV1,
    EconomicMPCPlanV1,
    EconomicMPCSettingsV1,
)
from .economic_mpc_solver import solve_economic_mpc_model
from .setpoint_env import PlantSetpointEnvV2
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
from .setpoint_control import (
    DecodedProcessSetpointActionV1,
    ProcessSetpointActionAdapterV1,
    ProcessSetpointAllocationV1,
    ProcessSetpointTargetsV1,
    allocate_process_setpoint_targets,
)
from .setpoint_command import (
    ProcessSetpointCommandV1,
    ProcessSetpointEnvelopeV1,
    ProcessSetpointMinuteDispatchV1,
    adapt_mpc_process_setpoints,
    dispatch_process_setpoint_command,
    issue_process_setpoint_command,
    resolve_process_setpoint_envelope,
)
from .setpoint_controllers import (
    LegacyHeuristicSetpointControllerV1,
    LegacyHeuristicSetpointDecisionV1,
    LegacyHeuristicSetpointEvidenceV1,
)
from .policy_setpoint_controller import (
    NORMALIZED_POLICY_SETPOINT_CONTROLLER_ID,
    NormalizedSetpointPolicyControllerV1,
    PolicySetpointDecisionEvidenceV1,
    PolicySetpointDecisionV1,
    policy_contract_for_environment,
)
from .setpoint_execution import (
    ProcessSetpointIntervalExecutionV1,
    ProcessSetpointMinuteApplicationV1,
    record_process_setpoint_application,
)
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
    "DecodedProcessSetpointActionV1",
    "DayAheadTariffV1",
    "EconomicAssumptionsV1",
    "EconomicControllerObservationV1",
    "EconomicEpisodeMetricsV1",
    "EconomicIntervalOutcomeV1",
    "EconomicMPCDecisionEvidenceV1",
    "EconomicMPCDecisionV1",
    "EconomicMPCModelV1",
    "EconomicMPCPlanV1",
    "EconomicMPCSettingsV1",
    "EconomicRewardBreakdownV1",
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
    "NORMALIZED_POLICY_SETPOINT_CONTROLLER_ID",
    "OBSERVATION_NAMES",
    "OBSERVATION_SIZE",
    "PlantEnvV2",
    "PlantEconomicSetpointEnvV2",
    "PlantSetpointEnvV2",
    "NormalizedSetpointPolicyControllerV1",
    "LegacyHeuristicSetpointControllerV1",
    "LegacyHeuristicSetpointDecisionV1",
    "LegacyHeuristicSetpointEvidenceV1",
    "ProcessSetpointActionAdapterV1",
    "ProcessSetpointAllocationV1",
    "ProcessSetpointCommandV1",
    "ProcessSetpointEnvelopeV1",
    "ProcessSetpointIntervalExecutionV1",
    "ProcessSetpointMinuteApplicationV1",
    "ProcessSetpointMinuteDispatchV1",
    "ProcessSetpointTargetsV1",
    "PolicySetpointDecisionEvidenceV1",
    "PolicySetpointDecisionV1",
    "RewardWeightsV2",
    "RiskAdjustedIssuedForecastControllerV2",
    "SPEC_VERSION",
    "ServiceMPCControllerV1",
    "ServiceMPCSetpointControllerV1",
    "ServiceMPCSetpointSettingsV1",
    "ServiceMPCSettingsV1",
    "SetpointControllerV2",
    "TariffObservationV1",
    "TariffAwareEconomicMPCControllerV1",
    "WeatherConfigV2",
    "adapt_mpc_process_setpoints",
    "allocate_process_setpoint_targets",
    "ch4_equivalent_inventory_kg",
    "compute_economic_reward",
    "build_economic_mpc_model",
    "describe",
    "dispatch_mpc_setpoints",
    "dispatch_process_setpoint_command",
    "evaluate",
    "evaluate_controllers",
    "issue_process_setpoint_command",
    "resolve_process_setpoint_envelope",
    "record_process_setpoint_application",
    "policy_contract_for_environment",
    "sample_episode",
    "solve_economic_mpc_model",
]

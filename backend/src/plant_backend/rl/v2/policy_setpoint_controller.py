"""Checkpoint-bound learned-policy adapter for the common setpoint command door."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from types import ModuleType
from typing import Any, ClassVar, TypeAlias

import numpy as np
from numpy.typing import NDArray
from plant_sim import ElectrolyserCommitmentProfile, PlantConfig

from ...controllers.base import Constraints
from .. import checkpoints
from ..policies import LoadedPolicy
from . import beta_spaces, economic_spaces, setpoint_spaces
from .beta_control import CyclicMarketControllerObservationV1
from .beta_env import PlantCyclicMarketEnvV2Beta
from .control import ControllerObservationV2
from .economic_env import PlantEconomicSetpointEnvV2
from .economics import EconomicControllerObservationV1
from .mpc_execution import public_observation_sha256
from .mpc_schema import domain_sha256
from .observation import ObservationEncoderV2
from .setpoint_command import ProcessSetpointCommandV1, issue_process_setpoint_command
from .setpoint_control import (
    DecodedProcessSetpointActionV1,
    ProcessSetpointActionAdapterV1,
)
from .setpoint_env import PlantSetpointEnvV2


NORMALIZED_POLICY_SETPOINT_CONTROLLER_ID = "rl-normalized-setpoint-policy-v1"
NORMALIZED_POLICY_SETPOINT_EVIDENCE_SCHEMA = "rl-setpoint-decision-evidence-v1"
NORMALIZED_POLICY_SETPOINT_EVIDENCE_DOMAIN = NORMALIZED_POLICY_SETPOINT_EVIDENCE_SCHEMA

PolicyObservationV1: TypeAlias = (
    ControllerObservationV2
    | EconomicControllerObservationV1
    | CyclicMarketControllerObservationV1
)
SetpointPolicyEnvironmentV1: TypeAlias = (
    PlantSetpointEnvV2 | PlantEconomicSetpointEnvV2 | PlantCyclicMarketEnvV2Beta
)


def policy_contract_for_environment(
    environment: SetpointPolicyEnvironmentV1,
) -> checkpoints.PolicyEnvironmentContract:
    """Detach the exact policy interface and full task digest from an environment."""

    interface: ModuleType
    if isinstance(environment, PlantCyclicMarketEnvV2Beta):
        interface = beta_spaces
    elif isinstance(environment, PlantEconomicSetpointEnvV2):
        interface = economic_spaces
    elif isinstance(environment, PlantSetpointEnvV2):
        interface = setpoint_spaces
    else:  # pragma: no cover - protected by the public type and exercised defensively
        raise TypeError("environment must use the common setpoint policy door")
    return checkpoints.PolicyEnvironmentContract(
        spec_version=interface.SPEC_VERSION,
        observation_fields=interface.OBSERVATION_NAMES,
        action_fields=interface.ACTION_NAMES,
        environment_id=interface.ENVIRONMENT_ID,
        environment_contract_digest=interface.ENVIRONMENT_CONTRACT_DIGEST,
        checkpoint_contract_digest=environment.checkpoint_contract_digest,
    )


@dataclass(frozen=True, slots=True)
class PolicySetpointDecisionEvidenceV1:
    """Exact learned-policy input, output, checkpoint, and target conversion."""

    decision_index: int
    observation_sha256: str
    policy_observation_sha256: str
    economic_observation_sha256: str | None
    checkpoint_metadata_sha256: str
    policy_contract_sha256: str
    decoded_action: DecodedProcessSetpointActionV1

    schema: ClassVar[str] = NORMALIZED_POLICY_SETPOINT_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if (
            isinstance(self.decision_index, bool)
            or not isinstance(self.decision_index, Integral)
            or self.decision_index < 0
        ):
            raise ValueError("decision_index must be non-negative")
        for name in (
            "observation_sha256",
            "policy_observation_sha256",
            "checkpoint_metadata_sha256",
            "policy_contract_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.economic_observation_sha256 is not None:
            _require_sha256(
                "economic_observation_sha256",
                self.economic_observation_sha256,
            )
        if not isinstance(self.decoded_action, DecodedProcessSetpointActionV1):
            raise TypeError("decoded_action must be DecodedProcessSetpointActionV1")

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "controller_id": NORMALIZED_POLICY_SETPOINT_CONTROLLER_ID,
            "decision_index": int(self.decision_index),
            "observation_sha256": self.observation_sha256,
            "policy_observation_sha256": self.policy_observation_sha256,
            "economic_observation_sha256": self.economic_observation_sha256,
            "checkpoint_metadata_sha256": self.checkpoint_metadata_sha256,
            "policy_contract_sha256": self.policy_contract_sha256,
            "decoded_action": self.decoded_action.to_dict(),
        }

    @property
    def sha256(self) -> str:
        return str(
            domain_sha256(NORMALIZED_POLICY_SETPOINT_EVIDENCE_DOMAIN, self.body())
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class PolicySetpointDecisionV1:
    """Learned decision evidence paired with its controller-neutral command."""

    evidence: PolicySetpointDecisionEvidenceV1
    command: ProcessSetpointCommandV1

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, PolicySetpointDecisionEvidenceV1):
            raise TypeError("evidence must be PolicySetpointDecisionEvidenceV1")
        if not isinstance(self.command, ProcessSetpointCommandV1):
            raise TypeError("command must be ProcessSetpointCommandV1")
        if self.command.decision_index != self.evidence.decision_index:
            raise ValueError("command decision index differs from policy evidence")
        if self.command.observation_sha256 != self.evidence.observation_sha256:
            raise ValueError("command observation differs from policy evidence")
        if self.command.issuer_id != NORMALIZED_POLICY_SETPOINT_CONTROLLER_ID:
            raise ValueError("command issuer is not the learned setpoint adapter")
        if self.command.source_schema != self.evidence.schema:
            raise ValueError("command source schema differs from policy evidence")
        if self.command.source_sha256 != self.evidence.sha256:
            raise ValueError("command source digest differs from policy evidence")
        if self.command.targets != self.evidence.decoded_action.targets:
            raise ValueError("command targets differ from decoded policy action")

    @property
    def invalid(self) -> bool:
        return bool(self.evidence.decoded_action.clipped)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": self.evidence.to_dict(),
            "command": self.command.to_dict(),
            "invalid": self.invalid,
        }


class NormalizedSetpointPolicyControllerV1:
    """Deterministic checkpoint inference followed by the common target adapter."""

    id: ClassVar[str] = NORMALIZED_POLICY_SETPOINT_CONTROLLER_ID
    label: ClassVar[str] = "RL policy (absolute-setpoint authority)"
    description: ClassVar[str] = (
        "A compatible deterministic policy whose four normalized outputs are bound "
        "to its exact observation and checkpoint before common setpoint execution."
    )
    uses_forecast: ClassVar[bool] = True
    information_set: ClassVar[str] = (
        "the checkpoint-declared V2 public observation; economic policies additionally "
        "receive the declared public tariff or beta market/cycle fields; no simulator, "
        "private weather or price truth, future forecast vintage, disturbance schedule, "
        "market regime, or hidden delivery schedule"
    )

    def __init__(
        self,
        loaded: LoadedPolicy,
        *,
        contract: checkpoints.PolicyEnvironmentContract,
        config: PlantConfig,
        constraints: Constraints,
        commitment_profile: ElectrolyserCommitmentProfile | None,
        economic_assumptions_sha256: str | None = None,
        tariff_id: str | None = None,
        beta_task_sha256: str | None = None,
    ) -> None:
        if not isinstance(loaded, LoadedPolicy):
            raise TypeError("loaded must be LoadedPolicy")
        if not isinstance(contract, checkpoints.PolicyEnvironmentContract):
            raise TypeError("contract must be PolicyEnvironmentContract")
        if not isinstance(config, PlantConfig):
            raise TypeError("config must be PlantConfig")
        if not isinstance(constraints, Constraints):
            raise TypeError("constraints must be Constraints")
        loaded.metadata.check_compatible(path=loaded.path, contract=contract)
        self.loaded = loaded
        self.contract = contract
        self.config = config
        self.constraints = constraints
        self.commitment_profile = commitment_profile
        expects_alpha_economic = (
            contract.environment_id == economic_spaces.ENVIRONMENT_ID
        )
        expects_beta = contract.environment_id == beta_spaces.ENVIRONMENT_ID
        if expects_alpha_economic or expects_beta:
            _require_sha256(
                "economic_assumptions_sha256",
                economic_assumptions_sha256,
            )
        if expects_alpha_economic:
            if not isinstance(tariff_id, str) or not tariff_id:
                raise ValueError(
                    "economic policy controller requires a tariff identity"
                )
            if beta_task_sha256 is not None:
                raise ValueError("alpha economic policy cannot declare a beta task")
        elif expects_beta:
            _require_sha256("beta_task_sha256", beta_task_sha256)
            if tariff_id is not None:
                raise ValueError(
                    "beta policy uses a causal market, not an alpha tariff"
                )
        elif (
            economic_assumptions_sha256 is not None
            or tariff_id is not None
            or beta_task_sha256 is not None
        ):
            raise ValueError(
                "physical setpoint policy cannot declare economic identities"
            )
        self.economic_assumptions_sha256 = economic_assumptions_sha256
        self.tariff_id = tariff_id
        self.beta_task_sha256 = beta_task_sha256
        self.encoder = ObservationEncoderV2(config, commitment_profile)
        self.adapter = ProcessSetpointActionAdapterV1()
        self._decision_count = 0
        self._metadata_sha256 = domain_sha256(
            "rl-checkpoint-metadata-v1",
            loaded.metadata.to_dict(),
        )
        self._contract_sha256 = domain_sha256(
            "rl-policy-environment-contract-v1",
            contract.to_dict(),
        )

    @classmethod
    def from_environment(
        cls,
        loaded: LoadedPolicy,
        environment: SetpointPolicyEnvironmentV1,
    ) -> "NormalizedSetpointPolicyControllerV1":
        if isinstance(environment, PlantCyclicMarketEnvV2Beta):
            return cls(
                loaded,
                contract=policy_contract_for_environment(environment),
                config=environment.config,
                constraints=environment.constraints,
                commitment_profile=environment.episode_config.electrolyser_commitment,
                economic_assumptions_sha256=(environment.economic_assumptions.sha256),
                beta_task_sha256=environment.beta_task.sha256,
            )
        if isinstance(environment, PlantEconomicSetpointEnvV2):
            return cls(
                loaded,
                contract=policy_contract_for_environment(environment),
                config=environment.config,
                constraints=environment.constraints,
                commitment_profile=environment.episode_config.electrolyser_commitment,
                economic_assumptions_sha256=(environment.economic_assumptions.sha256),
                tariff_id=environment.economic_assumptions.tariff.id,
            )
        return cls(
            loaded,
            contract=policy_contract_for_environment(environment),
            config=environment.config,
            constraints=environment.constraints,
            commitment_profile=environment.episode_config.electrolyser_commitment,
        )

    def reset(self) -> None:
        self.encoder.reset()
        self._decision_count = 0

    def encode_observation(
        self,
        observation: PolicyObservationV1,
        *,
        constraints: Constraints,
    ) -> NDArray[np.float32]:
        """Produce exactly the checkpoint-declared vector from the common typed source."""

        expects_alpha_economic = (
            self.contract.environment_id == economic_spaces.ENVIRONMENT_ID
        )
        expects_beta = self.contract.environment_id == beta_spaces.ENVIRONMENT_ID
        market = None
        cycle = None
        if isinstance(observation, CyclicMarketControllerObservationV1):
            if not expects_beta:
                raise TypeError("checkpoint does not accept the beta observation")
            if observation.economic_assumptions_sha256 != (
                self.economic_assumptions_sha256
            ):
                raise ValueError("beta economic assumptions identity drifted")
            if observation.beta_task_sha256 != self.beta_task_sha256:
                raise ValueError("beta task identity drifted")
            physical = observation.plant
            market = observation.market
            cycle = observation.cycle
            tariff = None
        elif isinstance(observation, EconomicControllerObservationV1):
            if not expects_alpha_economic:
                raise TypeError(
                    "checkpoint does not accept the alpha economic observation"
                )
            if observation.economic_assumptions_sha256 != (
                self.economic_assumptions_sha256
            ):
                raise ValueError("economic observation assumptions identity drifted")
            if observation.tariff_id != self.tariff_id:
                raise ValueError("economic observation tariff identity drifted")
            physical = observation.plant
            tariff = observation.tariff
        elif isinstance(observation, ControllerObservationV2):
            if expects_alpha_economic or expects_beta:
                raise TypeError("checkpoint requires its richer task observation")
            physical = observation
            tariff = None
        else:  # pragma: no cover - defensive runtime type check
            raise TypeError("observation must be a supported controller observation")
        encoded = self.encoder.encode_controller_observation(
            physical,
            constraints=constraints,
        )
        if tariff is not None:
            encoded = np.concatenate(
                (encoded, np.asarray(tariff.normalized(), dtype=np.float32))
            ).astype(np.float32, copy=False)
        if market is not None and cycle is not None:
            encoded = np.concatenate(
                (
                    encoded,
                    np.asarray(market.normalized(), dtype=np.float32),
                    np.asarray(cycle.normalized(self.config), dtype=np.float32),
                )
            ).astype(np.float32, copy=False)
        if encoded.shape != (len(self.contract.observation_fields),):
            raise RuntimeError(
                "encoded policy observation differs from checkpoint contract"
            )
        return encoded

    def compute_command(
        self,
        observation: PolicyObservationV1,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: ElectrolyserCommitmentProfile | None = None,
    ) -> PolicySetpointDecisionV1:
        """Infer one normalized action and bind it to a common process command."""

        if config != self.config:
            raise ValueError(
                "controller plant configuration differs from its checkpoint task"
            )
        if constraints != self.constraints:
            raise ValueError("controller constraints differ from its checkpoint task")
        if commitment_profile != self.commitment_profile:
            raise ValueError(
                "controller commitment profile differs from its checkpoint task"
            )
        physical = (
            observation.plant
            if isinstance(
                observation,
                (EconomicControllerObservationV1, CyclicMarketControllerObservationV1),
            )
            else observation
        )
        if not isinstance(physical, ControllerObservationV2):
            raise TypeError("observation must be a supported controller observation")
        vector = self.encode_observation(observation, constraints=constraints)
        action = self.loaded.policy.act(vector.copy())
        decoded = self.adapter.decode(action, constraints=constraints, config=config)
        evidence = PolicySetpointDecisionEvidenceV1(
            decision_index=self._decision_count,
            observation_sha256=public_observation_sha256(physical),
            policy_observation_sha256=domain_sha256(
                "rl-policy-observation-vector-v1",
                [float(value) for value in vector],
            ),
            economic_observation_sha256=(
                observation.sha256
                if isinstance(
                    observation,
                    (
                        EconomicControllerObservationV1,
                        CyclicMarketControllerObservationV1,
                    ),
                )
                else None
            ),
            checkpoint_metadata_sha256=self._metadata_sha256,
            policy_contract_sha256=self._contract_sha256,
            decoded_action=decoded,
        )
        command = issue_process_setpoint_command(
            decoded.targets,
            decision_index=evidence.decision_index,
            observation_sha256=evidence.observation_sha256,
            hold_duration_s=physical.decision_interval_s,
            issuer_id=self.id,
            source_schema=evidence.schema,
            source_sha256=evidence.sha256,
            constraints=constraints,
            config=config,
        )
        decision = PolicySetpointDecisionV1(evidence=evidence, command=command)
        self._decision_count += 1
        return decision

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "uses_forecast": self.uses_forecast,
            "information_set": self.information_set,
            "checkpoint": self.loaded.to_dict(),
            "policy_contract": self.contract.to_dict(),
            "constraints": self.constraints.to_dict(),
            "checkpoint_metadata_sha256": self._metadata_sha256,
            "policy_contract_sha256": self._contract_sha256,
            "economic_assumptions_sha256": self.economic_assumptions_sha256,
            "tariff_id": self.tariff_id,
            "beta_task_sha256": self.beta_task_sha256,
            "inference": "deterministic",
            "failure_policy": "raise; no controller-internal fallback",
        }


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as reason:
        raise ValueError(f"{name} must be a SHA-256 digest") from reason
    return value


__all__ = [
    "NORMALIZED_POLICY_SETPOINT_CONTROLLER_ID",
    "NORMALIZED_POLICY_SETPOINT_EVIDENCE_DOMAIN",
    "NORMALIZED_POLICY_SETPOINT_EVIDENCE_SCHEMA",
    "NormalizedSetpointPolicyControllerV1",
    "PolicyObservationV1",
    "PolicySetpointDecisionEvidenceV1",
    "PolicySetpointDecisionV1",
    "SetpointPolicyEnvironmentV1",
    "policy_contract_for_environment",
]

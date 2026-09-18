"""Frozen MPC/PPO comparators for the separate delivery-mission experiment.

These adapters consume a reset environment from ``feasible_mission_replay``.
They never sample scenarios, change physics, train, or publish historical results.
The PPO checkpoint is admitted against its ORIGINAL training configuration;
applying it to a changed delivery mission is explicitly a transfer experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for folder in ("backend/src", "simulator/src"):
    path = str(REPO / folder)
    if path not in sys.path:
        sys.path.insert(0, path)

from plant_backend.rl.v2.training.config import ControllerNeutralTrainingConfigV1
from plant_backend.rl.v2.training.research_mpc_residual import (
    describe_interface_contract,
    extend_observation,
    make_mpc_controller,
    prepare_residual_action,
)
from plant_backend.rl.v2.contract import contract_digest

POLICY_ID = "B-scalar-gamma1000"
CHECKPOINT_BYTES = 815_887
CHECKPOINT_SHA256 = "be134c0809b77896695d7482db4e432afe730c7e75cf89780a454b12e2bbe907"
TRAINING_DIRECTORY = (
    REPO.parents[1] / "PlantAnAgentResearch/adaptive-rl-batch-009/training" / POLICY_ID
)


@dataclass(frozen=True)
class LoadedMissionPolicy:
    policy: Any
    original_config: ControllerNeutralTrainingConfigV1
    admission: dict[str, Any]


@dataclass(frozen=True)
class ResidualTransferDecision:
    """A normal absolute process command plus its separately retained PPO record."""

    command: Any
    invalid: bool
    prior: Any
    policy_observation: list[float]
    raw_action: list[float]
    residual_evidence: dict[str, Any]

    @property
    def evidence(self) -> Any:
        """Retain both nominal MPC evidence and actual PPO inference inputs."""
        return {
            "nominal_mpc": json.loads(self.prior.evidence.payload_json),
            "policy_observation": self.policy_observation,
            "raw_action": self.raw_action,
            "residual_evidence": self.residual_evidence,
        }


def load_gamma_one_policy(
    directory: Path = TRAINING_DIRECTORY,
) -> LoadedMissionPolicy:
    """Verify original frozen identity and bytes, then load locally on CPU.

    No network or optimizer is used. This intentionally fails if the original
    loader cannot admit the checkpoint under the current source/runtime. It does
    not waive that failure or call SB3 directly as an alternative loading path.
    """
    from plant_backend.rl.v2.training import research_mpc_constraint_batch as source
    from plant_backend.rl.v2.training.research_reward_alignment_batch import (
        ARMS,
        verify_endpoint_model,
    )

    directory = Path(directory)
    checkpoint = directory / "final-policy.zip"
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("Frozen PPO checkpoint must be an ordinary local file")
    data = checkpoint.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != CHECKPOINT_BYTES or digest != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint differs from the published batch-009 endpoint")
    result_path = directory / "result.json"
    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes)
    original = ControllerNeutralTrainingConfigV1.from_dict(
        result["environment_template"]
    )
    if original.episode.delivery_rate_kg_h != 40.0:
        raise ValueError("Original PPO mission is no longer the published 40 kg/h")
    arm = next(arm for arm in ARMS if arm.name == POLICY_ID)
    policy = source.load_final(directory, original, arm)
    verify_endpoint_model(policy, arm)
    return LoadedMissionPolicy(
        policy=policy,
        original_config=original,
        admission={
            "policy_id": POLICY_ID,
            "checkpoint": {
                "path": str(checkpoint),
                "bytes": len(data),
                "sha256": digest,
            },
            "training_result_sha256": hashlib.sha256(result_bytes).hexdigest(),
            "original_training_contract_admitted": True,
            "original_delivery_rate_kg_h": 40.0,
            "optimizer_steps_executed": 0,
            "inference": "deterministic CPU inference from the fixed final endpoint",
            "scope": "Checkpoint admission only; changed mission performance is untested.",
        },
    )


def make_operating_mpc(environment: Any) -> Any:
    """Construct a fresh public-observation MPC for the supplied reset scenario."""
    return SerializableOperatingMPC(environment)


class SerializableOperatingMPC:
    """Preserve the native policy while exposing JSON-ready research evidence."""

    def __init__(self, environment):
        self.native = make_mpc_controller(environment)
        self.id = self.native.id
        self.reset()

    def reset(self):
        self.native.reset()

    def to_dict(self):
        return self.native.to_dict()

    def compute_command(self, observation, **kwargs):
        decision = self.native.compute_command(observation, **kwargs)
        return SimpleNamespace(
            command=decision.command,
            invalid=decision.invalid,
            evidence=json.loads(decision.evidence.payload_json),
        )


class MPCResidualTransferController:
    """Expose the same compute_command interface as native Operating MPC.

    Call ``set_observation`` with the ACTUAL reset/step return before each
    compute. The adapter never re-encodes observations or accesses future tapes.
    A new instance is required for each scenario. Command execution remains the
    harness's responsibility through ``environment.step_process_setpoints``.
    """

    def __init__(self, environment: Any, loaded: LoadedMissionPolicy):
        self.environment = environment
        self.loaded = loaded
        self.prior = make_mpc_controller(environment)
        self.prior.reset()
        self.id = "frozen-scalar-ppo-gamma1000-mission-transfer-v1"
        self.interface_contract = describe_interface_contract(environment)
        self.interface_contract_digest = contract_digest(self.interface_contract)
        self._observation: np.ndarray[Any, np.dtype[np.float32]] | None = None
        self._observation_decision: int | None = None
        self.last_decision: ResidualTransferDecision | None = None
        # Shape compatibility is distinct from the admitted historical task.
        if tuple(loaded.policy.observation_space.shape) != (139,) or tuple(
            loaded.policy.action_space.shape
        ) != (4,):
            raise ValueError("Frozen policy does not expose the MPC-residual interface")

    @property
    def metadata(self) -> dict[str, Any]:
        rate = self.environment.episode_config.delivery_rate_kg_h
        transferred = rate != self.loaded.original_config.episode.delivery_rate_kg_h
        return {
            **self.loaded.admission,
            "evaluation_delivery_rate_kg_h": rate,
            "mission_transfer": transferred,
            "training_task_contract_unchanged": True,
            "evaluation_delivery_rate_matches_training": not transferred,
            "claim": (
                "Exploratory delivery-mission transfer with frozen weights; "
                "no retraining, registered admission, held-out confirmation, "
                "or original-task policy promotion."
            ),
        }

    def to_dict(self):
        return {"id": self.id, "transfer": self.metadata}

    def set_observation(self, observation: Any) -> None:
        if (
            not isinstance(observation, np.ndarray)
            or observation.shape != (135,)
            or observation.dtype != np.float32
            or not np.isfinite(observation).all()
        ):
            raise ValueError("Supply the actual finite float32 135-field plant vector")
        self._observation = observation.copy()
        self._observation_decision = self.environment.decisions_taken

    def compute_command(
        self,
        observation: Any,
        *,
        constraints: Any,
        config: Any,
        commitment_profile: Any,
    ) -> ResidualTransferDecision:
        env = self.environment
        if (
            self._observation is None
            or self._observation_decision != env.decisions_taken
        ):
            raise ValueError("Set the actual current reset/step observation first")
        if observation.sha256 != env.cyclic_market_controller_observation.sha256:
            raise ValueError("Public controller observation is stale or mismatched")
        prior = self.prior.compute_command(
            observation,
            constraints=constraints,
            config=config,
            commitment_profile=commitment_profile,
        )
        vector = extend_observation(self._observation, env, prior)
        action, _ = self.loaded.policy.predict(vector.copy(), deterministic=True)
        raw = np.asarray(action)
        if (
            raw.shape != (4,)
            or raw.dtype.kind not in "iuf"
            or not np.isfinite(raw).all()
        ):
            raise ValueError("Frozen policy returned an invalid four-action residual")
        command, invalid, residual = prepare_residual_action(
            env,
            prior,
            raw,
            interface_contract_digest=self.interface_contract_digest,
        )
        decision = ResidualTransferDecision(
            command=command,
            invalid=invalid,
            prior=prior,
            policy_observation=vector.tolist(),
            raw_action=raw.tolist(),
            residual_evidence=residual,
        )
        self.last_decision = decision
        self._observation = None  # Do not admit a second inference for this vector.
        self._observation_decision = None
        return decision

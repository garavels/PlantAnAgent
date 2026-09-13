"""Batch 007 matched scalar/PI MPC residual development with power admission."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import plistlib
import re
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Callable, cast

from .. import beta_spaces
from ..beta_config import CyclicMarketTaskConfigV1
from ..beta_env import PlantCyclicMarketEnvV2Beta
from . import ppo
from .research_batch import OptimizerRewardScale, file_record, now, training_callback
from .research_effort_batch import (
    Arm,
    TrainingIntegrityError,
    TrainingUnavailable,
    preflight_checkpoint,
)
from .research_mpc_residual import MPCResidualIntegrityError
from .config import (
    BETA_SCALAR_PPO_ALGORITHM_ID as SCALAR,
    BETA_PI_LAGRANGIAN_ALGORITHM_ID as PI,
    ControllerNeutralTrainingConfigV1,
)
from .custody_io import (
    mkdir_exclusive,
    publish_bytes,
    publish_json,
    read_regular,
    require_directory,
)
from .custody_json import canonical_bytes
from .custody_process import capture_process
from .episode_allocator import EpisodeSeedAllocatorV1, EpisodeSeedCustodyError
from .runtime_provenance import (
    BetaTrainingRuntimeProvenanceV1,
    capture,
    verify_unchanged,
)

MODULE = "plant_backend.rl.v2.training.research_mpc_constraint_batch"
ACTIVE_BASE = Path("/Users/filippogombac/Developer/PlantAnAgentResearch")
ROOT = ACTIVE_BASE / "adaptive-rl-batch-007"
SCHEMA = "plant-adaptive-rl-mpc-constraint-batch-007"
STORAGE_SCHEMA = "plant-adaptive-internal-storage-v1"
POWER_SAMPLE_SCHEMA = "plant-adaptive-power-sample-v1"
POWER_COMMAND = ("/usr/bin/pmset", "-g", "batt")
POWER_TIMEOUT = 2.0
POWER_OUTPUT_LIMIT = 8192
MINIMUM_BATTERY_PERCENT = 20
AC_BATTERY_STATES = ("charging", "charged", "not charging")
_INTERNAL_VOLUME_UUID = "86093AA6-4234-4173-AD1C-F1762C98870E"
_INTERNAL_MOUNT = Path("/System/Volumes/Data")
TOTAL_STEPS = 65_536
OPTIMIZER_SEED = 101
TRAIN_SEED_BASE = 9_630_000
TRAIN_SEED_LAST = 9_630_099
DEV_SEEDS = tuple(range(9_700_300, 9_700_308))
BASELINES = ("H", "OperatingMPC", "M24", "E-beta")
LOG_LIMIT = 64 * 1024**2
ARCHIVE_LIMIT = 256 * 1024**2
TRAIN_TIMEOUT = 12 * 3600
EVAL_TIMEOUT = 600
MINIMUM_FREE = 10 * 1024**3
INTEGRITY_EXIT = 86
INTEGRITY_CHECK_INTERVAL = 30.0
ORDINARY_CHILD_CODES = frozenset(
    {
        "child_spawn_failed",
        "child_timeout",
        "child_log_limit_exceeded",
        "child_signaled",
        "child_nonzero_exit",
    }
)


ARMS = (
    Arm("A-mpc-residual-scalar", SCALAR, 0.001, 0.2, True),
    Arm("B-mpc-residual-pi", PI, 0.001, 0.2, True),
)
INITIAL_SOURCES = {"initial-mpc-residual": ARMS[0]}

ACTORS = (*BASELINES, *INITIAL_SOURCES, *(a.name for a in ARMS))


def template(root: Path) -> ControllerNeutralTrainingConfigV1:
    return ControllerNeutralTrainingConfigV1(
        environment_id=beta_spaces.ENVIRONMENT_ID,
        seed=OPTIMIZER_SEED,
        total_timesteps=TOTAL_STEPS,
        checkpoint_every=0,
        output_dir=str(root / "artifacts"),
        beta_task=CyclicMarketTaskConfigV1(),
        notes="Adaptive research environment template; learner and optimizer reward units in batch plan.",
    ).validate()


def specification(root: Path) -> dict[str, Any]:
    from .research_parallel_supervision import SCHEMA as supervision_schema
    from .research_parallel_supervision import monitor_settings

    return {
        "schema": SCHEMA,
        "root": str(root),
        "repair_replication": {
            "previous_batch": "adaptive-rl-batch-006",
            "previous_source_commit": "c59ba89aa301b863e50be7246fe5003c6fb731dc",
            "previous_report_sha256": "7ec0ca88e83f5e98743ee396251941e3f61973c8f5610e6f4878bd75c886cbcf",
            "previous_interruption_inventory_sha256": "e19ff6d6f37c452af00e43125a0ab7be109e5218ad988c2fd1c0a40e1f9e66fe",
            "previous_os_evidence_sha256": "51b02b6fd7823534cf20053afe717bab1350e6e0733e100d80aac9deeea1365a",
            "previous_status": "low-battery hibernation followed by closed abort; no final trained policies or evaluation; exact subsequent partial-rollout progress unavailable",
            "training_seed_status": "reused development stream; batch 004 opened eight reset seeds; batch 005 retained the first 44 reset records per arm; batch 006 retained the first 40 reset records and 24,576 completed-rollout transitions per arm",
            "development_seed_status": "planned for batches 004 through 006 but never opened; adaptive development only",
            "initialization": "fresh matched models and optimizer/allocator state; no resume",
            "storage_change": "unchanged persistent internal active artifacts and admission evidence; Lexar archive only after closure; retain internal originals",
            "power_change": "AC and at least 20 percent reserve before root claim, at training integrity callbacks and stage entry/exit; no retry or resume",
            "operating_solar_projection": "issued-mean-physics-left-samples-v1",
            "task_reward_learner_authority_changes": False,
            "registered_projection_changes": False,
        },
        "arms": [asdict(a) for a in ARMS],
        "environment_template": template(root).to_dict(),
        "training_seed_base": TRAIN_SEED_BASE,
        "training_seed_last": TRAIN_SEED_LAST,
        "training_rollouts": 8,
        "transitions_per_rollout": 8192,
        "training_environments_per_arm": 4,
        "training_resets_per_arm": 100,
        "training_episodes_per_environment_including_partial": 25,
        "training_concurrency": "two isolated supervisors; evaluation/replay sequential",
        "evaluation_slots": 56,
        "replay_slots": 56,
        "framework_environment_seed": None,
        "minibatch_optimizer_steps": 2560,
        "development_seeds": list(DEV_SEEDS),
        "baselines": list(BASELINES),
        "initial_baseline_sources": {
            name: arm.name for name, arm in INITIAL_SOURCES.items()
        },
        "actors": list(ACTORS),
        "policy_identities": {
            arm.name: policy_identity(template(root), arm) for arm in ARMS
        },
        "checkpoint_selection": "fixed final endpoint only",
        "torch_intraop_threads": 1,
        "torch_interop_threads": 1,
        "training_integrity_check_interval_seconds": INTEGRITY_CHECK_INTERVAL,
        "training_supervisor_startup_seconds": 30.0,
        "training_supervision_schema": supervision_schema,
        "training_log_monitor": monitor_settings(),
        "train_timeout_seconds": TRAIN_TIMEOUT,
        "evaluation_timeout_seconds": EVAL_TIMEOUT,
        "stream_limit_bytes": LOG_LIMIT,
        "archive_limit_bytes": ARCHIVE_LIMIT,
        "minimum_free_bytes": MINIMUM_FREE,
        "power_policy": power_policy(),
        "registered_evaluation_eligible": False,
        "authorization": "User granted full research autonomy to diagnose learning failure and iterate experiments toward verified operational benefit.",
    }


def control_environment(config: ControllerNeutralTrainingConfigV1, arm: Arm) -> Any:
    """Construct the actual policy interface without consuming a reset."""
    from .research_mpc_residual import ResearchMPCResidualWrapper

    env = ppo.make_env(config)
    if not isinstance(env, PlantCyclicMarketEnvV2Beta):
        cast(Any, env).close()
        raise ValueError("adaptive effort screen requires the beta task")
    if not arm.residual or arm.algorithm not in (SCALAR, PI):
        cast(Any, env).close()
        raise ValueError(
            "MPC residual screen requires its scalar/PI residual interface"
        )
    try:
        return ResearchMPCResidualWrapper(env)
    except BaseException:
        cast(Any, env).close()
        raise


def policy_identity(
    config: ControllerNeutralTrainingConfigV1, arm: Arm
) -> dict[str, Any]:
    from .constrained_ppo import ConstrainedPPO

    env = control_environment(config, arm)
    try:
        checkpoint = env.checkpoint_contract
        return {
            "schema": "plant-research-mpc-constraint-policy-identity-v1",
            "representation": "residual-mpc",
            "checkpoint_contract": checkpoint,
            "checkpoint_contract_digest": env.checkpoint_contract_digest,
            "interface_contract": checkpoint["interface_contract"],
            "interface_contract_digest": checkpoint["interface_contract_digest"],
            "observation_size": int(env.observation_space.shape[0]),
            "action_size": int(env.action_space.shape[0]),
            "observation_space": ConstrainedPPO._box_space_record(
                "PPO observation space", env.observation_space
            ),
            "action_space": ConstrainedPPO._box_space_record(
                "PPO action space", env.action_space
            ),
            "initialization": {
                "optimizer_seed": config.seed,
                "actor_output_head_weight": "exactly zero",
                "actor_output_head_bias": "exactly zero",
                "initial_gaussian_std": arm.initial_std,
                "deterministic_initial_command": "exact cached OperatingMPC",
                "matching_scope": "exact actor and reward critic tensors across both fresh arms; independent existing cost-critic initialization",
            },
            "registered_evaluation_eligible": False,
        }
    finally:
        env.close()


def verify_model_spaces(
    model: Any, config: ControllerNeutralTrainingConfigV1, arm: Arm
) -> None:
    from ...checkpoints import verify_policy_model_spaces

    env = control_environment(config, arm)
    try:
        verify_policy_model_spaces(
            model,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
    finally:
        env.close()


def make_learner(
    config: ControllerNeutralTrainingConfigV1, arm: Arm, *, seed_base: int
) -> tuple[Any, EpisodeSeedAllocatorV1, Any]:
    import torch
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from .constrained_ppo import ConstrainedPPOSettingsV1, build_constrained_model

    config.validate()
    if config.beta_rl is not None or config.n_envs != 4 or config.resume_from:
        raise ValueError("adaptive research requires a fresh four-environment template")
    allocator = EpisodeSeedAllocatorV1.fixture(
        base_seed=seed_base,
        episode_decisions=config.episode.decision_steps(config.plant),
    )

    def factory(index: int) -> Any:
        return lambda: OptimizerRewardScale(
            Monitor(
                allocator.wrap(
                    control_environment(config, arm), environment_index=index
                )
            ),
            arm.reward_scale,
        )

    vector = DummyVecEnv([factory(index) for index in range(4)])
    try:
        model = build_constrained_model(
            vector,
            settings=ConstrainedPPOSettingsV1.for_algorithm(
                arm.algorithm, optimizer_seed=config.seed
            ),
            ppo_kwargs=asdict(config.ppo),
            verbose=1,
        )
        with torch.no_grad():
            model.policy.log_std.fill_(math.log(arm.initial_std))
            action_head = cast(torch.nn.Linear, model.policy.action_net)
            action_head.weight.zero_()
            action_head.bias.zero_()
        setattr(model, "research_identity", policy_identity(config, arm))
        verify_model_spaces(model, config, arm)
        return model, allocator, vector
    except BaseException:
        vector.close()
        raise


def allocator_prefix(allocator: EpisodeSeedAllocatorV1) -> dict[str, Any]:
    """Retain observed custody state without asserting a successful endpoint."""
    return {
        "schema": "plant-research-mpc-constraint-allocator-prefix-v1",
        "allocator_identity": allocator.identity,
        "ledger": [entry.to_dict() for entry in allocator.ledger],
        "cursors": [cursor.to_dict() for cursor in allocator.cursors],
        "decisions_consumed_by_environment": list(
            allocator.decisions_consumed_by_environment
        ),
        "successful_endpoint": False,
    }


def tensor_records(weights: dict[str, Any], *, shared_only: bool) -> dict[str, Any]:
    """Bind exact tensors, independent of Torch archive serialization details."""
    import torch

    records = {}
    for name, value in sorted(weights.items()):
        if shared_only and name.startswith("cost_critic."):
            continue
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(
                "initial/final policy contains a nonfinite or nontensor value"
            )
        tensor = value.detach().cpu().contiguous()
        records[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        }
    if not records:
        raise ValueError("policy tensor binding is empty")
    return records


def initial_policy_record(
    directory: Path, model: Any, config: Any, arm: Arm
) -> dict[str, Any]:
    return {
        "schema": "plant-research-mpc-constraint-initial-policy-v1",
        "arm": asdict(arm),
        "environment_template": config.to_dict(),
        "policy_identity": model.research_identity,
        "initial_policy": file_record(directory / "initial-policy.pt"),
        "all_tensors": tensor_records(model.policy.state_dict(), shared_only=False),
        "actor_reward_tensors": tensor_records(
            model.policy.state_dict(), shared_only=True
        ),
        "successful_endpoint": False,
    }


def read_initial_policy(directory: Path, config: Any, arm: Arm) -> dict[str, Any]:
    import torch

    record = json.loads(read_regular(directory / "initial-policy-record.json"))
    if (
        record.get("schema") != "plant-research-mpc-constraint-initial-policy-v1"
        or record.get("arm") != asdict(arm)
        or canonical_bytes(record.get("environment_template"))
        != canonical_bytes(config.to_dict())
        or canonical_bytes(record.get("policy_identity"))
        != canonical_bytes(policy_identity(config, arm))
        or record.get("initial_policy") != file_record(directory / "initial-policy.pt")
    ):
        raise ValueError("adaptive initial policy binding or bytes changed")
    weights = torch.load(
        io.BytesIO(read_regular(directory / "initial-policy.pt")),
        map_location="cpu",
        weights_only=True,
    )
    if tensor_records(weights, shared_only=False) != record.get(
        "all_tensors"
    ) or tensor_records(weights, shared_only=True) != record.get(
        "actor_reward_tensors"
    ):
        raise ValueError("adaptive initial policy tensor binding differs")
    return cast(dict[str, Any], weights)


def verify_initial_matching(root: Path, config: Any) -> dict[str, Any]:
    """Admit exactly the two reached initial artifacts, including failed arms."""
    import torch

    reached = []
    unavailable = []
    tensors = []
    weights_by_arm = []
    for arm in ARMS:
        directory = root / "training" / arm.name
        initial = directory / "initial-policy-record.json"
        if initial.is_symlink():
            raise TrainingIntegrityError("adaptive initial policy binding is a symlink")
        if not initial.exists():
            require_successful_training_or_unavailable(arm)
            unavailable.append(arm.name)
            continue
        weights = read_initial_policy(directory, config, arm)
        weights_by_arm.append(weights)
        tensors.append(tensor_records(weights, shared_only=True))
        reached.append({"arm": arm.name, "record": file_record(initial)})
    if len(tensors) == 2:
        if tensors[0] != tensors[1] or any(
            not torch.equal(weights_by_arm[0][key], weights_by_arm[1][key])
            for key in tensors[0]
        ):
            raise TrainingIntegrityError(
                "adaptive actor/reward initial tensors differ across arms"
            )
    return {
        "schema": "plant-research-mpc-constraint-initial-matching-v1",
        "expected_arms": [arm.name for arm in ARMS],
        "initial_policy_records": reached,
        "unavailable_arms": unavailable,
        "matched": len(tensors) == 2,
        "shared_tensors": tensors[0] if len(tensors) == 2 else None,
        "cost_critics_excluded": True,
        "initial_comparator_source": ARMS[0].name,
    }


def require_successful_training_or_unavailable(arm: Arm) -> None:
    try:
        require_successful_training(arm)
    except TrainingUnavailable:
        return
    raise TrainingIntegrityError(
        "successful training omitted its initial policy binding"
    )


def validate_training_endpoint(
    model: Any, allocator: Any, config: Any, seed_base: int
) -> dict[str, Any]:
    from .pilot import validate_endpoint

    snapshot = validate_endpoint(model, allocator, config)
    episode_decisions = config.episode.decision_steps(config.plant)
    expected_resets = config.n_envs * (
        config.total_timesteps // config.n_envs // episode_decisions + 1
    )
    if len(allocator.ledger) != expected_resets or any(
        entry.episode_seed != seed_base + entry.ledger_index
        or entry.ignored_framework_seed is not None
        for entry in allocator.ledger
    ):
        raise EpisodeSeedCustodyError("adaptive reset stream or framework seed differs")
    return snapshot


def cell_callback(directory: Path, allocator: EpisodeSeedAllocatorV1) -> Any:
    """Reuse immutable optimizer diagnostics and retain reached allocator prefixes."""
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    class CustodyRecorder(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.written_resets = 0
            self.rollouts = 0

        def _on_step(self) -> bool:
            for entry in allocator.ledger[self.written_resets :]:
                publish_json(
                    directory / "allocator" / f"reset-{entry.ledger_index:03d}.json",
                    entry.to_dict(),
                )
                self.written_resets += 1
            return True

        def _on_rollout_end(self) -> None:
            self.rollouts += 1
            publish_json(
                directory / "allocator" / f"rollout-{self.rollouts:03d}.json",
                allocator_prefix(allocator),
            )

    class EvidenceCallbacks(CallbackList):
        def checked(self, action: Callable[[], Any]) -> Any:
            try:
                return action()
            except Exception as exc:
                raise TrainingIntegrityError(
                    "adaptive training evidence callback failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        def _on_step(self) -> bool:
            return bool(self.checked(super()._on_step))

        def _on_rollout_start(self) -> None:
            self.checked(super()._on_rollout_start)

        def _on_rollout_end(self) -> None:
            self.checked(super()._on_rollout_end)

        def _on_training_end(self) -> None:
            self.checked(super()._on_training_end)

    return EvidenceCallbacks(
        [training_callback(directory / "diagnostics"), CustodyRecorder()]
    )


def retain_training_failure(
    directory: Path,
    model: Any,
    allocator: EpisodeSeedAllocatorV1,
    reason: BaseException,
) -> None:
    """Keep failures descriptive; never authorize loading an unfinished policy."""
    try:
        publish_json(
            directory / "training-failure.json",
            {
                "schema": SCHEMA,
                "failed_at": now(),
                "failure": {"type": type(reason).__name__, "message": str(reason)},
                "num_timesteps": model.num_timesteps,
                "allocator_prefix": allocator_prefix(allocator),
                "rollout_ledgers": [
                    ledger.to_dict() for ledger in model.rollout_ledgers
                ],
                "successful_endpoint": False,
                "registered_evaluation_eligible": False,
            },
        )
    except Exception as exc:
        raise TrainingIntegrityError(
            "adaptive failed-training evidence publication failed: "
            f"{type(exc).__name__}: {exc}; original {type(reason).__name__}: {reason}"
        ) from exc


def train_cell(
    directory: Path,
    config: ControllerNeutralTrainingConfigV1,
    arm: Arm,
    *,
    seed_base: int,
) -> dict[str, Any]:
    import torch
    from .constrained_ppo import ConstrainedPPO

    try:
        publish_json(
            directory / "attempt.json",
            {
                "started_at": now(),
                "arm": asdict(arm),
                "template": config.to_dict(),
                "seed_base": seed_base,
                "policy_identity": policy_identity(config, arm),
            },
        )
        mkdir_exclusive(directory / "diagnostics")
        mkdir_exclusive(directory / "allocator")
        model, allocator, vector = make_learner(config, arm, seed_base=seed_base)
    except Exception as exc:
        raise TrainingIntegrityError(
            "adaptive pre-learning evidence or interface admission failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    try:
        try:
            buffer = io.BytesIO()
            torch.save(model.policy.state_dict(), buffer)
            publish_bytes(directory / "initial-policy.pt", buffer.getvalue())
            publish_json(
                directory / "initial-policy-record.json",
                initial_policy_record(directory, model, config, arm),
            )
        except Exception as exc:
            raise TrainingIntegrityError(
                "adaptive initial-policy publication failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=cell_callback(directory, allocator),
        )
        try:
            snapshot = validate_training_endpoint(model, allocator, config, seed_base)
            artifact = directory / "final-policy.zip"
            model.save(artifact)
            state = model.embedded_constraint_state
            identity = file_record(artifact)
            preflight_checkpoint(artifact, model.research_identity, state)
            reopened = ConstrainedPPO.load(artifact, device="cpu")
            verify_model_spaces(reopened, config, arm)
            attestation = reopened.loaded_constraint_state_attestation()
            if (
                attestation.artifact_sha256 != identity["sha256"]
                or attestation.constraint_state != state
                or attestation.num_timesteps != config.total_timesteps
                or canonical_bytes(getattr(reopened, "research_identity", None))
                != canonical_bytes(model.research_identity)
                or tensor_records(reopened.policy.state_dict(), shared_only=False)
                != tensor_records(model.policy.state_dict(), shared_only=False)
            ):
                raise ValueError(
                    "adaptive final model failed exact reopen/endpoint verification"
                )
            result = {
                "schema": SCHEMA,
                "completed_at": now(),
                "arm": asdict(arm),
                "environment_template": config.to_dict(),
                "total_timesteps": model.num_timesteps,
                "initial_policy": file_record(directory / "initial-policy.pt"),
                "initial_policy_record": file_record(
                    directory / "initial-policy-record.json"
                ),
                "final_policy": identity,
                "final_policy_tensors": tensor_records(
                    model.policy.state_dict(), shared_only=False
                ),
                "policy_identity": model.research_identity,
                "checkpoint_contract_digest": model.research_identity[
                    "checkpoint_contract_digest"
                ],
                "constraint_state": state,
                "allocator": snapshot,
                "training_reward_units": f"EUR multiplied by {arm.reward_scale:g}; evaluation always raw EUR",
                "registered_evaluation_eligible": False,
            }
            publish_json(directory / "result.json", result)
            return result
        except Exception as exc:
            raise TrainingIntegrityError(
                "adaptive post-learning endpoint or publication failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    except BaseException as exc:
        retain_training_failure(directory, model, allocator, exc)
        raise
    finally:
        vector.close()


def read_training_result(
    directory: Path, config: ControllerNeutralTrainingConfigV1, arm: Arm
) -> dict[str, Any]:
    result = json.loads(read_regular(directory / "result.json"))
    if canonical_bytes(result["environment_template"]) != canonical_bytes(
        config.to_dict()
    ) or result["arm"] != asdict(arm):
        raise ValueError("adaptive checkpoint task or learner differs from frozen cell")
    identity = policy_identity(config, arm)
    if result.get("schema") != SCHEMA or canonical_bytes(
        result.get("policy_identity")
    ) != canonical_bytes(identity):
        raise ValueError(
            "adaptive policy interface or initialization differs from frozen cell"
        )
    if (
        result.get("checkpoint_contract_digest")
        != identity["checkpoint_contract_digest"]
    ):
        raise ValueError("adaptive checkpoint identity digest differs")
    if result.get("total_timesteps") != config.total_timesteps:
        raise ValueError("adaptive completed training budget differs")
    if result.get("initial_policy_record") != file_record(
        directory / "initial-policy-record.json"
    ):
        raise ValueError("adaptive initial policy record changed")
    return cast(dict[str, Any], result)


def load_final(
    directory: Path, config: ControllerNeutralTrainingConfigV1, arm: Arm
) -> Any:
    from .constrained_ppo import ConstrainedPPO

    result = read_training_result(directory, config, arm)
    if file_record(directory / "final-policy.zip") != result["final_policy"]:
        raise ValueError("adaptive policy bytes changed")
    preflight_checkpoint(
        directory / "final-policy.zip",
        result["policy_identity"],
        result["constraint_state"],
    )
    model = ConstrainedPPO.load(directory / "final-policy.zip", device="cpu")
    verify_model_spaces(model, config, arm)
    if (
        model.algorithm_id != arm.algorithm
        or model.num_timesteps != config.total_timesteps
        or model.embedded_constraint_state != result["constraint_state"]
        or canonical_bytes(getattr(model, "research_identity", None))
        != canonical_bytes(result["policy_identity"])
        or tensor_records(model.policy.state_dict(), shared_only=False)
        != result["final_policy_tensors"]
    ):
        raise ValueError("adaptive policy state differs from completed cell")
    return model


def load_initial(
    directory: Path, config: ControllerNeutralTrainingConfigV1, arm: Arm
) -> Any:
    import torch
    from stable_baselines3.common.policies import ActorCriticPolicy

    result = read_training_result(directory, config, arm)
    if file_record(directory / "initial-policy.pt") != result["initial_policy"]:
        raise ValueError("adaptive initial policy bytes changed")
    from .constrained_ppo import LearnerRNGStateV1

    caller_rng = LearnerRNGStateV1.capture()
    env = control_environment(config, arm)
    try:
        policy = ActorCriticPolicy(
            env.observation_space,
            env.action_space,
            lr_schedule=lambda _: 0.0,
            net_arch={"pi": [64, 64], "vf": [64, 64]},
        )
        weights = read_initial_policy(directory, config, arm)
        policy.load_state_dict(
            {
                name: value
                for name, value in weights.items()
                if not name.startswith("cost_critic.")
            },
            strict=True,
        )
        action_head = cast(torch.nn.Linear, policy.action_net)
        if not torch.equal(
            action_head.weight, torch.zeros_like(action_head.weight)
        ) or not torch.equal(action_head.bias, torch.zeros_like(action_head.bias)):
            raise ValueError("adaptive initial actor output head is not exactly zero")
        if not torch.equal(
            policy.log_std, torch.full_like(policy.log_std, math.log(arm.initial_std))
        ):
            raise ValueError("adaptive initial exploration differs")
        verify_model_spaces(policy, config, arm)
        return policy
    finally:
        env.close()
        caller_rng.restore()


def power_policy() -> dict[str, Any]:
    """Return only the frozen policy; observations never enter its identity."""
    return {
        "schema": "plant-adaptive-ac-power-policy-v1",
        "command": list(POWER_COMMAND),
        "required_source": "AC Power",
        "minimum_battery_percent": MINIMUM_BATTERY_PERCENT,
        "accepted_battery_states": list(AC_BATTERY_STATES),
        "command_timeout_seconds": POWER_TIMEOUT,
        "maximum_observation_bytes": POWER_OUTPUT_LIMIT,
        "training_check_interval_seconds": INTEGRITY_CHECK_INTERVAL,
        "sampling": [
            "before_root_claim",
            "training_integrity_callbacks",
            "stage_entry_and_exit",
        ],
        "failure_action": "abort-no-retry",
    }


class PowerAdmissionError(ValueError):
    """An identifiable refusal whose bounded observation remains on disk."""


def capture_power_sample() -> dict[str, Any]:
    """Sample macOS power once, retaining bounded query and timing evidence."""
    sample: dict[str, Any] = {
        "schema": POWER_SAMPLE_SCHEMA,
        "command": list(POWER_COMMAND),
        "pid": os.getpid(),
        "started_at": now(),
        "started_monotonic_ns": time.monotonic_ns(),
        "status": "refused",
        "code": None,
        "source": None,
        "battery_percent": None,
        "battery_state": None,
        "battery_present": None,
        "command_exit_code": None,
        "stdout": "",
        "stderr": "",
        "output_truncated": False,
        "error": None,
    }
    stdout, stderr = b"", b""
    try:
        result = subprocess.run(
            list(POWER_COMMAND),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=POWER_TIMEOUT,
            check=True,
        )
        stdout, stderr = result.stdout, result.stderr
        sample["command_exit_code"] = result.returncode
    except subprocess.TimeoutExpired as exc:
        sample["code"] = "power_query_timeout"
        stdout, stderr = exc.output or b"", exc.stderr or b""
        sample["error"] = {"type": type(exc).__name__, "message": str(exc)[:2048]}
    except subprocess.CalledProcessError as exc:
        sample["code"] = "power_query_failed"
        sample["command_exit_code"] = exc.returncode
        stdout, stderr = exc.output or b"", exc.stderr or b""
        sample["error"] = {"type": type(exc).__name__, "message": str(exc)[:2048]}
    except OSError as exc:
        sample["code"] = "power_query_unavailable"
        sample["error"] = {"type": type(exc).__name__, "message": str(exc)[:2048]}
    finally:
        sample["closed_monotonic_ns"] = time.monotonic_ns()
        sample["closed_at"] = now()
        sample["elapsed_ns"] = (
            sample["closed_monotonic_ns"] - sample["started_monotonic_ns"]
        )
    sample["output_truncated"] = len(stdout) + len(stderr) > POWER_OUTPUT_LIMIT
    bounded_stdout = stdout[:POWER_OUTPUT_LIMIT]
    bounded_stderr = stderr[: POWER_OUTPUT_LIMIT - len(bounded_stdout)]
    sample["stdout"] = (
        bounded_stdout.decode("utf-8", errors="replace")
        .encode("utf-8")[:POWER_OUTPUT_LIMIT]
        .decode("utf-8", errors="ignore")
    )
    remaining = POWER_OUTPUT_LIMIT - len(sample["stdout"].encode("utf-8"))
    sample["stderr"] = (
        bounded_stderr.decode("utf-8", errors="replace")
        .encode("utf-8")[:remaining]
        .decode("utf-8", errors="ignore")
    )
    if sample["code"] is not None:
        return sample
    if sample["elapsed_ns"] > int(POWER_TIMEOUT * 1e9):
        sample["code"] = "power_query_timeout"
        return sample
    if sample["output_truncated"]:
        sample["code"] = "power_observation_oversized"
        return sample
    try:
        lines = stdout.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        sample["code"] = "power_observation_malformed"
        return sample
    if len(lines) != 2:
        sample["code"] = "power_observation_malformed"
        return sample
    source = re.fullmatch(r"Now drawing from '([^']+)'", lines[0])
    battery = re.fullmatch(
        r"\s*-InternalBattery-\d+ \(id=\d+\)\s+(\d{1,3})%; "
        r"([a-z ]+); (?:\d+:\d{2} remaining|\(no estimate\)) present: (true|false)",
        lines[1],
    )
    if battery is None:
        battery = re.fullmatch(
            r"\s*-InternalBattery-\d+ \(id=\d+\)\s+(\d{1,3})%; "
            r"AC attached; (not charging) present: (true|false)",
            lines[1],
        )
    if source is None or battery is None:
        sample["code"] = "power_observation_malformed"
        return sample
    sample.update(
        source=source.group(1),
        battery_percent=int(battery.group(1)),
        battery_state=battery.group(2),
        battery_present=battery.group(3) == "true",
    )
    if not 0 <= sample["battery_percent"] <= 100:
        sample["code"] = "power_observation_malformed"
    elif not sample["battery_present"]:
        sample["code"] = "power_battery_absent"
    elif sample["source"] != "AC Power":
        sample["code"] = "power_source_not_ac"
    elif sample["battery_state"] not in AC_BATTERY_STATES:
        sample["code"] = "power_battery_state_rejected"
    elif sample["battery_percent"] < MINIMUM_BATTERY_PERCENT:
        sample["code"] = "power_reserve_low"
    else:
        sample["status"] = "accepted"
    return sample


def validate_power_admission(value: object) -> None:
    """Reject invalid retained admission without comparing dynamic charge levels."""
    if not isinstance(value, dict) or (
        value.get("schema") != POWER_SAMPLE_SCHEMA
        or value.get("command") != list(POWER_COMMAND)
        or value.get("status") != "accepted"
        or value.get("code") is not None
        or value.get("error") is not None
        or type(value.get("command_exit_code")) is not int
        or value.get("command_exit_code") != 0
        or value.get("output_truncated") is not False
        or value.get("source") != "AC Power"
        or value.get("battery_present") is not True
        or type(value.get("battery_percent")) is not int
        or not MINIMUM_BATTERY_PERCENT <= value["battery_percent"] <= 100
        or value.get("battery_state") not in AC_BATTERY_STATES
        or type(value.get("pid")) is not int
        or value["pid"] <= 0
        or any(
            type(value.get(key)) is not int
            for key in ("started_monotonic_ns", "closed_monotonic_ns", "elapsed_ns")
        )
        or value["started_monotonic_ns"] < 0
        or not 0 <= value["elapsed_ns"] <= int(POWER_TIMEOUT * 1e9)
        or value["elapsed_ns"]
        != value["closed_monotonic_ns"] - value["started_monotonic_ns"]
        or any(
            not isinstance(value.get(key), str) or not value[key]
            for key in ("started_at", "closed_at")
        )
        or any(not isinstance(value.get(key), str) for key in ("stdout", "stderr"))
        or len((value["stdout"] + value["stderr"]).encode("utf-8")) > POWER_OUTPUT_LIMIT
    ):
        raise ValueError("adaptive initial power admission differs")


def require_power(
    *, failure_directory: Path, admission: bool = False
) -> dict[str, Any]:
    """Preserve refusals exclusively before raising through integrity handling."""
    if not admission and os.path.lexists(failure_directory):
        raise PowerAdmissionError(
            f"power_previous_refusal; evidence={failure_directory}"
        )
    sample = capture_power_sample()
    if sample["status"] == "accepted":
        validate_power_admission(sample)
        return sample
    prefix = f"{ROOT.name}-power-admission" if admission else "power"
    path = failure_directory / (
        f"{prefix}-{sample['pid']}-{sample['closed_monotonic_ns']}.json"
    )
    try:
        if not admission and not os.path.lexists(failure_directory):
            try:
                mkdir_exclusive(failure_directory)
            except FileExistsError:
                # Concurrent root and supervisor refusals share this directory.
                require_directory(failure_directory)
        publish_json(path, sample)
    except Exception as exc:
        raise PowerAdmissionError(
            f"{sample['code']}; power evidence publication failed: {type(exc).__name__}"
        ) from exc
    raise PowerAdmissionError(f"{sample['code']}; evidence={path}")


def _storage_identity(gate: Path, *, require_gate: bool) -> dict[str, Any]:
    """Check current paths/device against the fixed internal-storage contract."""
    if (
        ACTIVE_BASE != ACTIVE_BASE.resolve()
        or ROOT.parent != ACTIVE_BASE
        or ROOT != ROOT.resolve()
        or not gate.is_absolute()
        or not gate.is_relative_to(ACTIVE_BASE)
        or gate.is_relative_to(ROOT)
        or gate != gate.resolve()
    ):
        raise ValueError("adaptive storage paths must remain canonical and internal")
    require_directory(ACTIVE_BASE)
    require_directory(_INTERNAL_MOUNT)
    require_directory(gate.parent)
    paths = [
        ACTIVE_BASE,
        _INTERNAL_MOUNT,
        Path(__file__).resolve().parents[6],
        gate.parent,
    ]
    if os.path.lexists(ROOT):
        require_directory(ROOT)
        paths.append(ROOT)
    if os.path.lexists(gate):
        observed = gate.lstat()
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise ValueError(
                "adaptive storage gate must be a singly linked regular file"
            )
        paths.append(gate)
    elif require_gate:
        raise ValueError("adaptive storage gate is absent")
    device = ACTIVE_BASE.stat().st_dev
    if any(path.stat().st_dev != device for path in paths):
        raise ValueError(
            "adaptive storage device differs from the internal repository volume"
        )
    if shutil.disk_usage(ACTIVE_BASE).free < MINIMUM_FREE:
        raise ValueError("adaptive batch needs 10 GiB free on internal active storage")
    return {
        "schema": STORAGE_SCHEMA,
        "active_base": str(ACTIVE_BASE),
        "batch_root": str(ROOT),
        "gate_path": str(gate),
        "volume": {
            "uuid": _INTERNAL_VOLUME_UUID,
            "mount_point": str(_INTERNAL_MOUNT),
            "filesystem_type": "apfs",
            "internal": True,
            "removable": False,
            "ejectable": False,
            "st_dev": device,
        },
    }


def capture_storage_identity(gate: Path) -> dict[str, Any]:
    """Verify OS volume metadata before admission; the gate may not yet exist."""
    if os.path.lexists(ROOT):
        raise ValueError("adaptive root must be absent; no retry")
    identity = _storage_identity(gate, require_gate=False)
    try:
        raw = subprocess.check_output(
            ["/usr/sbin/diskutil", "info", "-plist", str(_INTERNAL_MOUNT)],
            stderr=subprocess.PIPE,
            timeout=10,
        )
        volume = plistlib.loads(raw)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise ValueError("adaptive internal storage OS admission failed") from exc
    expected = {
        "VolumeUUID": _INTERNAL_VOLUME_UUID,
        "MountPoint": str(_INTERNAL_MOUNT),
        "FilesystemType": "apfs",
        "Internal": True,
        "Removable": False,
        "Ejectable": False,
    }
    if not isinstance(volume, dict) or any(
        type(volume.get(key)) is not type(value) or volume.get(key) != value
        for key, value in expected.items()
    ):
        raise ValueError(
            "adaptive storage must use the frozen internal nonremovable APFS volume"
        )
    if os.path.lexists(ROOT):
        raise ValueError("adaptive root appeared during OS admission; no retry")
    if _storage_identity(gate, require_gate=False) != identity:
        raise ValueError("adaptive storage changed during OS admission")
    return identity


def require_storage(gate: Path, expected: object) -> dict[str, Any]:
    """Recheck frozen paths/device and free space without recurring OS commands."""
    identity = _storage_identity(gate, require_gate=True)
    if canonical_bytes(identity) != canonical_bytes(expected):
        raise ValueError("adaptive storage identity changed")
    return identity


def require_plan() -> dict[str, Any]:
    from .capacity_benchmark import _host_identity

    plan = json.loads(read_regular(ROOT / "plan.json"))
    gate_path = Path(plan["gate"]["path"])
    storage = require_storage(gate_path, plan["storage"])
    gate = json.loads(read_regular(gate_path))
    if canonical_bytes(gate.get("storage")) != canonical_bytes(storage):
        raise ValueError("adaptive gate and plan storage identities differ")
    if canonical_bytes(plan["specification"]) != canonical_bytes(specification(ROOT)):
        raise ValueError("adaptive batch specification changed")
    if canonical_bytes(gate.get("power_policy")) != canonical_bytes(power_policy()):
        raise ValueError("adaptive gate and plan power policies differ")
    validate_power_admission(plan.get("power_admission"))
    verify_unchanged(
        BetaTrainingRuntimeProvenanceV1.from_dict(plan["runtime"]),
        capture(require_clean=True),
    )
    if _host_identity() != plan["host"] or file_record(gate_path) != plan["gate"]:
        raise ValueError("adaptive batch host or gate changed")
    require_power(failure_directory=ROOT / "power-failures")
    return cast(dict[str, Any], plan)


def require_successful_training(arm: Arm) -> None:
    receipt = json.loads(read_regular(ROOT / "receipts" / f"train-{arm.name}.json"))
    if (
        receipt["stage"] != "_train"
        or receipt["name"] != f"train-{arm.name}"
        or not isinstance(receipt["codes"], list)
    ):
        raise ValueError("adaptive training receipt identity is inconsistent")
    validate_receipt_closure(receipt, ["--cell", arm.name])
    if receipt["codes"]:
        raise TrainingUnavailable(
            "adaptive policy has no successfully closed training child"
        )
    if (
        receipt["process"]["exit_code"] != 0
        or receipt["process"].get("started") is not True
        or not receipt.get("closed_at")
        or receipt["process"].get("command")
        != [sys.executable, "-m", MODULE, "_train", "--cell", arm.name]
    ):
        raise ValueError("adaptive training receipt omits a child failure")


def validate_receipt_closure(receipt: dict[str, Any], extra: list[str]) -> None:
    """Unavailable means an ordinary closed attempt, never hidden integrity loss."""
    if (
        receipt["process"].get("exit_code") == INTEGRITY_EXIT
        or "root_custody_lost" in receipt["codes"]
    ):
        raise TrainingIntegrityError("adaptive receipt records evidence integrity loss")
    if (
        any(code not in ORDINARY_CHILD_CODES for code in receipt["codes"])
        or not receipt.get("closed_at")
        or receipt["process"].get("command")
        != [sys.executable, "-m", MODULE, receipt["stage"], *extra]
    ):
        raise ValueError("adaptive receipt is not an ordinary closed attempt")


def evaluate_cell(actor: str, seed: int) -> None:
    from .research_evaluation import evaluate_episode

    directory = ROOT / "evaluations" / f"{seed}-{actor}"
    publish_json(
        directory / "attempt.json", {"actor": actor, "seed": seed, "started_at": now()}
    )
    config = template(ROOT)
    policy = None
    model_actor = actor
    arm = None
    if actor in INITIAL_SOURCES:
        arm = INITIAL_SOURCES[actor]
        require_successful_training(arm)
        policy = load_initial(ROOT / "training" / arm.name, config, arm)
        model_actor = "PPO"
    elif actor not in BASELINES:
        arm = next(a for a in ARMS if a.name == actor)
        require_successful_training(arm)
        policy = load_final(ROOT / "training" / actor, config, arm)
        model_actor = "PPO"
    if arm is not None and arm.residual:
        from .research_mpc_residual_evaluation import evaluate_residual_episode

        episode = evaluate_residual_episode(
            config, policy=policy, seed=seed, policy_id=actor
        )
    else:
        episode = evaluate_episode(
            config, actor=model_actor, policy=policy, seed=seed, policy_id=actor
        )
    encoded = gzip.compress(canonical_bytes(episode), compresslevel=6, mtime=0)
    if len(encoded) > ARCHIVE_LIMIT:
        raise ValueError("adaptive evaluation archive exceeded stored-size cap")
    publish_bytes(directory / "episode.json.gz", encoded)
    publish_json(
        directory / "summary.json",
        {
            "actor": actor,
            "seed": seed,
            "summary": episode["summary"],
            "archive": file_record(directory / "episode.json.gz"),
        },
    )


def replay_cell(actor: str, seed: int) -> None:
    from .research_evaluation import verify_episode_replay
    from .research_mpc_residual_evaluation import ACTOR as residual_actor

    directory = ROOT / "evaluations" / f"{seed}-{actor}"
    publish_json(
        directory / "replay-attempt.json",
        {"actor": actor, "seed": seed, "started_at": now()},
    )
    require_successful_evaluation(actor, seed)
    summary = json.loads(read_regular(directory / "summary.json"))
    archive = directory / "episode.json.gz"
    if file_record(archive) != summary["archive"]:
        raise ValueError("adaptive trajectory changed before replay")
    if summary["archive"]["bytes"] > ARCHIVE_LIMIT:
        raise ValueError("adaptive replay archive exceeded stored-size cap")
    episode = json.loads(gzip.decompress(read_regular(archive)))
    if (
        summary.get("actor") != actor
        or canonical_bytes(summary.get("seed")) != canonical_bytes(seed)
        or canonical_bytes(episode.get("seed")) != canonical_bytes(seed)
        or canonical_bytes(summary.get("summary"))
        != canonical_bytes(episode.get("summary"))
        or episode["summary"].get("policy_id") != actor
        or episode["summary"].get("actor")
        != (actor if actor in BASELINES else residual_actor)
    ):
        raise ValueError("adaptive replay slot identity or summary differs")
    if actor == "initial-mpc-residual" or any(
        a.name == actor and a.residual for a in ARMS
    ):
        from .research_mpc_residual_evaluation import verify_residual_episode_replay

        result = verify_residual_episode_replay(template(ROOT), episode)
    else:
        result = verify_episode_replay(template(ROOT), episode)
    publish_json(directory / "replay.json", result)


def require_successful_evaluation(actor: str, seed: int) -> None:
    name = f"evaluate-{seed}-{actor}"
    receipt = json.loads(read_regular(ROOT / "receipts" / f"{name}.json"))
    if (
        receipt.get("stage") != "_evaluate"
        or receipt.get("name") != name
        or not isinstance(receipt.get("codes"), list)
    ):
        raise ValueError("adaptive evaluation receipt identity is inconsistent")
    validate_receipt_closure(receipt, ["--actor", actor, "--seed", str(seed)])
    if receipt["codes"]:
        raise TrainingUnavailable(
            "adaptive replay has no successfully closed evaluation child"
        )
    if (
        receipt["process"].get("exit_code") != 0
        or receipt["process"].get("started") is not True
        or not receipt.get("closed_at")
        or receipt["process"].get("command")
        != [
            sys.executable,
            "-m",
            MODULE,
            "_evaluate",
            "--actor",
            actor,
            "--seed",
            str(seed),
        ]
    ):
        raise ValueError("adaptive evaluation receipt omits a child failure")


def training_wave() -> dict[str, Any]:
    from .research_parallel_supervision import ParallelAttempt, run_training_wave

    attempts = []
    for arm in ARMS:
        name = f"train-{arm.name}"
        extra = ("--cell", arm.name)
        attempts.append(
            ParallelAttempt(
                name=name,
                stage="_train",
                timeout_seconds=TRAIN_TIMEOUT,
                extra=extra,
                command=(sys.executable, "-m", MODULE, "_train", *extra),
                stdout=ROOT / "logs" / f"{name}.stdout.log",
                stderr=ROOT / "logs" / f"{name}.stderr.log",
                receipt=ROOT / "receipts" / f"{name}.json",
            )
        )
    return run_training_wave(
        attempts,
        supervisor_directory=ROOT / "supervisors",
        worker_module=MODULE,
        worker_function="closed_child",
        check_integrity=require_plan,
        minimum_free_bytes=MINIMUM_FREE,
        supervisor_log_limit_bytes=LOG_LIMIT,
        child_log_limit_bytes=LOG_LIMIT,
        integrity_check_interval_seconds=INTEGRITY_CHECK_INTERVAL,
    )


def configure_threads() -> None:
    """Apply the same host identity in the parent, supervisors and task children."""
    import torch

    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def closed_child(
    stage: str, name: str, timeout: int, extra: list[str]
) -> dict[str, Any]:
    configure_threads()
    require_plan()
    if shutil.disk_usage(ROOT).free < MINIMUM_FREE:
        raise RuntimeError("adaptive batch needs 10 GiB free before child admission")
    process, codes = capture_process(
        [sys.executable, "-m", MODULE, stage, *extra],
        stdout=ROOT / "logs" / f"{name}.stdout.log",
        stderr=ROOT / "logs" / f"{name}.stderr.log",
        timeout_seconds=timeout,
        log_limit_bytes=LOG_LIMIT,
    )
    receipt = {
        "stage": stage,
        "name": name,
        "process": process,
        "codes": sorted(codes),
        "closed_at": now(),
    }
    publish_json(ROOT / "receipts" / f"{name}.json", receipt)
    if "root_custody_lost" in codes or process["exit_code"] == INTEGRITY_EXIT:
        raise RuntimeError(
            "operator interruption, root custody loss or evidence integrity failure; batch aborted"
        )
    require_plan()
    return receipt


def run(gate: Path, *, authorized: bool) -> dict[str, Any]:
    from .capacity_benchmark import _host_identity

    if not authorized:
        raise ValueError("adaptive batch requires research authorization assertion")
    if ROOT.exists() or ROOT != ROOT.resolve():
        raise ValueError("adaptive root must be absent and canonical; no retry")
    _storage_identity(gate, require_gate=True)
    runtime = capture(require_clean=True)
    gate_raw = read_regular(gate)
    checked = json.loads(gate_raw)
    if (
        checked.get("passed") is not True
        or checked.get("source_commit") != runtime.git_head
    ):
        raise ValueError("adaptive gate must pass at the exact clean source")
    if canonical_bytes(checked.get("power_policy")) != canonical_bytes(power_policy()):
        raise ValueError("adaptive gate power policy differs")
    storage = capture_storage_identity(gate)
    require_storage(gate, checked.get("storage"))
    gate_record = file_record(gate)
    if gate_record["sha256"] != hashlib.sha256(gate_raw).hexdigest():
        raise ValueError("adaptive gate changed during storage admission")
    plan = {
        "specification": specification(ROOT),
        "runtime": runtime.to_dict(),
        "host": _host_identity(),
        "gate": gate_record,
        "storage": storage,
        "started_at": now(),
    }
    plan["power_admission"] = require_power(
        failure_directory=gate.parent, admission=True
    )
    mkdir_exclusive(ROOT)
    publish_json(ROOT / "plan.json", plan)
    for name in ("logs", "receipts", "training", "evaluations", "supervisors"):
        mkdir_exclusive(ROOT / name)
    receipts = []
    failure = None
    try:
        for arm in ARMS:
            mkdir_exclusive(ROOT / "training" / arm.name)
        wave = training_wave()
        publish_json(ROOT / "training-wave.json", wave)
        receipts.extend(wave["receipts"])
        if wave["fatal"] is not None:
            raise TrainingIntegrityError(
                f"training supervisor integrity loss: {wave['fatal']}"
            )
        require_plan()
        publish_json(
            ROOT / "initial-matching.json",
            verify_initial_matching(ROOT, template(ROOT)),
        )
        for seed in DEV_SEEDS:
            for actor in ACTORS:
                name = f"{seed}-{actor}"
                mkdir_exclusive(ROOT / "evaluations" / name)
                receipt = closed_child(
                    "_evaluate",
                    f"evaluate-{name}",
                    EVAL_TIMEOUT,
                    ["--actor", actor, "--seed", str(seed)],
                )
                receipts.append(receipt)
                receipts.append(
                    closed_child(
                        "_replay",
                        f"replay-{name}",
                        EVAL_TIMEOUT,
                        ["--actor", actor, "--seed", str(seed)],
                    )
                )
    except BaseException as reason:
        failure = {"type": type(reason).__name__, "message": str(reason)}
    report = {
        "schema": SCHEMA,
        "status": "aborted" if failure else "closed",
        "closed_at": now(),
        "failure": failure,
        "plan": file_record(ROOT / "plan.json"),
        "receipts": receipts,
        "planned_slots": {
            "training": [a.name for a in ARMS],
            "evaluation": [f"{seed}-{actor}" for seed in DEV_SEEDS for actor in ACTORS],
            "replay": [f"{seed}-{actor}" for seed in DEV_SEEDS for actor in ACTORS],
        },
        "claim": "Adaptive diagnostic development; no selected policy or confirmation claim.",
        "registered_evaluation_eligible": False,
    }
    publish_json(ROOT / "report.json", report)
    return report


def execute_child(action: Callable[[], Any], *, artifact_phase: bool) -> int:
    """Report a reserved exit for integrity failures without hiding local failures."""

    def failed_integrity(exc: Exception) -> int:
        print(
            json.dumps({"integrity_failure": type(exc).__name__, "message": str(exc)}),
            file=sys.stderr,
        )
        return INTEGRITY_EXIT

    try:
        require_plan()
    except Exception as exc:
        return failed_integrity(exc)
    try:
        action()
    except (
        TrainingIntegrityError,
        EpisodeSeedCustodyError,
        MPCResidualIntegrityError,
    ) as exc:
        return failed_integrity(exc)
    except TrainingUnavailable:
        raise
    except Exception as exc:
        if not artifact_phase:
            raise
        # Expected controller failures are already preserved inside the evaluator.
        # Any unhandled publication/load/replay exception prevents trusted evidence.
        return failed_integrity(exc)
    try:
        require_plan()
    except Exception as exc:
        return failed_integrity(exc)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "_train", "_evaluate", "_replay"))
    parser.add_argument("--assert-authorized", action="store_true")
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--cell", choices=[a.name for a in ARMS])
    parser.add_argument("--actor", choices=ACTORS)
    parser.add_argument("--seed", type=int, choices=DEV_SEEDS)
    args = parser.parse_args(argv)
    configure_threads()
    if args.command == "run":
        if args.gate is None:
            parser.error("run requires --gate")
        result = run(args.gate, authorized=args.assert_authorized)
        print(
            json.dumps(
                {"status": result["status"], "report": str(ROOT / "report.json")}
            )
        )
        return 1 if result["failure"] else 0
    if args.command == "_train":
        if args.cell is None:
            parser.error("_train requires --cell")
        arm = next(a for a in ARMS if a.name == args.cell)
        return execute_child(
            lambda: train_cell(
                ROOT / "training" / arm.name,
                template(ROOT),
                arm,
                seed_base=TRAIN_SEED_BASE,
            ),
            artifact_phase=False,
        )
    else:
        if args.actor is None or args.seed is None:
            parser.error("evaluation/replay requires --actor and --seed")
        return execute_child(
            lambda: (evaluate_cell if args.command == "_evaluate" else replay_cell)(
                args.actor, args.seed
            ),
            artifact_phase=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())

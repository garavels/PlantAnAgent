"""Paired heuristic/RL evaluation through the controller-neutral setpoint door."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from numbers import Integral
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .. import checkpoints
from ..policies import LoadedPolicy, load_policy
from .economic_env import PlantEconomicSetpointEnvV2
from .mpc_schema import domain_sha256
from .policy_setpoint_controller import (
    NORMALIZED_POLICY_SETPOINT_CONTROLLER_ID,
    NormalizedSetpointPolicyControllerV1,
    SetpointPolicyEnvironmentV1,
    policy_contract_for_environment,
)
from .setpoint_controllers import LegacyHeuristicSetpointControllerV1
from .setpoint_env import PlantSetpointEnvV2
from .training.config import ControllerNeutralTrainingConfigV1
from .training.ppo import make_env


CONTROLLER_NEUTRAL_POLICY_EVALUATION_FORMAT = "controller-neutral-policy-evaluation-v1"


def _vector_sha256(vector: NDArray[np.generic]) -> str:
    return str(
        domain_sha256(
            "rl-policy-observation-vector-v1",
            [float(value) for value in np.asarray(vector).reshape(-1)],
        )
    )


def _manifest_identity(environment: SetpointPolicyEnvironmentV1) -> dict[str, str]:
    manifest = environment.audit_manifest()
    identity = {
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "physical_exogenous_episode_sha256": str(manifest["exogenous_episode_sha256"]),
        "checkpoint_contract_sha256": environment.checkpoint_contract_digest,
    }
    if isinstance(environment, PlantEconomicSetpointEnvV2):
        identity.update(
            {
                "economic_episode_sha256": str(manifest["economic_episode_sha256"]),
                "economic_assumptions_sha256": (
                    environment.economic_assumptions.sha256
                ),
            }
        )
    return identity


def _validated_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    values: list[int] = []
    for index, seed in enumerate(seeds):
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError(f"seed {index} must be a non-negative integer")
        values.append(int(seed))
    if not values:
        raise ValueError("paired evaluation requires at least one seed")
    if len(set(values)) != len(values):
        raise ValueError("paired evaluation seeds must be unique")
    return tuple(values)


def _preflight_loaded_policy_task(
    loaded: LoadedPolicy,
    config: ControllerNeutralTrainingConfigV1,
) -> LoadedPolicy:
    """Refuse a checkpoint/task mismatch before either paired arm is reset.

    A policy adapter also checks this contract when it is constructed, but the
    paired evaluator deliberately runs the heuristic first.  Keeping the check
    here prevents an incompatible policy from producing a half-evaluation in
    which only the heuristic has moved through the episode.
    """

    if not isinstance(loaded, LoadedPolicy):
        raise TypeError("loaded must be LoadedPolicy")
    environment = make_env(config)
    try:
        contract = policy_contract_for_environment(environment)
        loaded.metadata.check_compatible(path=loaded.path, contract=contract)
        persisted = checkpoints.load_metadata(
            loaded.path,
            contract=contract,
        )
        if persisted != loaded.metadata:
            raise checkpoints.CorruptCheckpointError(
                f"checkpoint metadata changed after policy loading ({loaded.path})"
            )
        # Re-admit the persisted artifact with the concrete task environment.
        # ``LoadedPolicy`` is also a public seam, so callers are not required to
        # have used ``load_policy_for_task`` and its embedded-space check.
        readmitted = load_policy(
            loaded.path,
            contract=contract,
            environment=environment,
        )
        if readmitted.metadata != loaded.metadata:
            raise checkpoints.CorruptCheckpointError(
                f"checkpoint changed during evaluation admission ({loaded.path})"
            )
        return readmitted
    finally:
        environment.close()


def _episode_result(
    *,
    controller: Mapping[str, Any],
    seed: int,
    environment: SetpointPolicyEnvironmentV1,
    rewards: Sequence[float],
    decisions: Sequence[Mapping[str, Any]],
    executions: Sequence[Mapping[str, Any]],
    inference_elapsed_ns: Sequence[int],
    dispatcher_elapsed_ns: Sequence[Sequence[int]],
    terminated: bool,
    truncated: bool,
    failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    status = (
        "complete" if failure is None and terminated and not truncated else "failed"
    )
    deterministic = {
        "controller_id": controller["id"],
        "seed": seed,
        "manifest_identity": _manifest_identity(environment),
        "completed_decisions": environment.decisions_taken,
        "physical_steps": environment.physics_steps_taken,
        "terminated": terminated,
        "truncated": truncated,
        "reward_trace": list(rewards),
        "physical_metrics": environment.metrics.to_dict(),
        "economic_metrics": (
            environment.economic_metrics.to_dict()
            if isinstance(environment, PlantEconomicSetpointEnvV2)
            else None
        ),
        "decisions": [dict(item) for item in decisions],
        "executions": [dict(item) for item in executions],
        "failure": None if failure is None else dict(failure),
    }
    return {
        "status": status,
        "controller": dict(controller),
        **deterministic,
        "reward_trace_sum": float(math.fsum(rewards)),
        "execution_count": len(executions),
        "application_count": sum(
            int(execution["application_count"]) for execution in executions
        ),
        "inference_elapsed_ns": list(inference_elapsed_ns),
        "dispatcher_elapsed_ns": [list(values) for values in dispatcher_elapsed_ns],
        "deterministic_evidence_sha256": domain_sha256(
            "controller-neutral-policy-episode-evidence-v1",
            deterministic,
        ),
    }


def execute_policy_episode(
    environment: SetpointPolicyEnvironmentV1,
    loaded: LoadedPolicy,
    *,
    seed: int,
) -> dict[str, Any]:
    """Run deterministic checkpoint inference with complete common-door evidence."""

    observation, _ = environment.reset(seed=seed)
    controller = NormalizedSetpointPolicyControllerV1.from_environment(
        loaded,
        environment,
    )
    controller.reset()
    rewards: list[float] = []
    decisions: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    inference_elapsed_ns: list[int] = []
    dispatcher_elapsed_ns: list[list[int]] = []
    terminated = truncated = False
    failure: dict[str, Any] | None = None
    try:
        while not terminated and not truncated:
            if isinstance(environment, PlantEconomicSetpointEnvV2):
                source = environment.economic_controller_observation
            else:
                source = environment.controller_observation
            started_ns = time.perf_counter_ns()
            decision = controller.compute_command(
                source,
                constraints=environment.constraints,
                config=environment.config,
                commitment_profile=(environment.episode_config.electrolyser_commitment),
            )
            inference_elapsed_ns.append(time.perf_counter_ns() - started_ns)
            decisions.append(decision.to_dict())
            if decision.evidence.policy_observation_sha256 != _vector_sha256(
                observation
            ):
                raise RuntimeError(
                    "controller encoding differs from the environment observation"
                )
            observation, reward, terminated, truncated, info = (
                environment.step_process_setpoints(
                    decision.command,
                    controller_invalid=decision.invalid,
                )
            )
            execution = environment.last_process_setpoint_execution
            if execution is None:
                raise RuntimeError("policy step omitted common execution evidence")
            elapsed = info.get("dispatcher_elapsed_ns")
            if not isinstance(elapsed, list) or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in elapsed
            ):
                raise RuntimeError("policy dispatcher timing evidence is malformed")
            executions.append(execution.to_dict())
            dispatcher_elapsed_ns.append([int(value) for value in elapsed])
            rewards.append(float(reward))
    except Exception as reason:  # retained; evaluation never invents a fallback
        failure = {
            "phase": "policy_episode",
            "exception_type": type(reason).__name__,
            "message": " ".join(str(reason).split())[:500],
        }
    return _episode_result(
        controller=controller.to_dict(),
        seed=seed,
        environment=environment,
        rewards=rewards,
        decisions=decisions,
        executions=executions,
        inference_elapsed_ns=inference_elapsed_ns,
        dispatcher_elapsed_ns=dispatcher_elapsed_ns,
        terminated=terminated,
        truncated=truncated,
        failure=failure,
    )


def execute_heuristic_episode(
    environment: SetpointPolicyEnvironmentV1,
    *,
    seed: int,
) -> dict[str, Any]:
    """Run the transparent heuristic through the identical generic command door."""

    environment.reset(seed=seed)
    controller = LegacyHeuristicSetpointControllerV1()
    controller.reset()
    rewards: list[float] = []
    decisions: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    inference_elapsed_ns: list[int] = []
    dispatcher_elapsed_ns: list[list[int]] = []
    terminated = truncated = False
    failure: dict[str, Any] | None = None
    try:
        while not terminated and not truncated:
            started_ns = time.perf_counter_ns()
            decision = controller.compute_command(
                environment.controller_observation,
                constraints=environment.constraints,
                config=environment.config,
                commitment_profile=(environment.episode_config.electrolyser_commitment),
            )
            inference_elapsed_ns.append(time.perf_counter_ns() - started_ns)
            decisions.append(decision.to_dict())
            _, reward, terminated, truncated, info = environment.step_process_setpoints(
                decision.command,
                controller_invalid=decision.invalid,
            )
            execution = environment.last_process_setpoint_execution
            if execution is None:
                raise RuntimeError("heuristic step omitted common execution evidence")
            elapsed = info.get("dispatcher_elapsed_ns")
            if not isinstance(elapsed, list) or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in elapsed
            ):
                raise RuntimeError("heuristic dispatcher timing evidence is malformed")
            executions.append(execution.to_dict())
            dispatcher_elapsed_ns.append([int(value) for value in elapsed])
            rewards.append(float(reward))
    except Exception as reason:
        failure = {
            "phase": "heuristic_episode",
            "exception_type": type(reason).__name__,
            "message": " ".join(str(reason).split())[:500],
        }
    return _episode_result(
        controller=controller.to_dict(),
        seed=seed,
        environment=environment,
        rewards=rewards,
        decisions=decisions,
        executions=executions,
        inference_elapsed_ns=inference_elapsed_ns,
        dispatcher_elapsed_ns=dispatcher_elapsed_ns,
        terminated=terminated,
        truncated=truncated,
        failure=failure,
    )


def compare_policy_to_heuristic(
    loaded: LoadedPolicy,
    config: ControllerNeutralTrainingConfigV1,
    *,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Pair both arms on exact exogenous episodes and report, never rank, outcomes."""

    seed_values = _validated_seeds(seeds)
    loaded = _preflight_loaded_policy_task(loaded, config)
    runs: list[dict[str, Any]] = []
    for seed in seed_values:
        heuristic = execute_heuristic_episode(make_env(config), seed=seed)
        policy = execute_policy_episode(make_env(config), loaded, seed=seed)
        if heuristic["manifest_identity"] != policy["manifest_identity"]:
            raise RuntimeError(f"paired manifest identity differs for seed {seed}")
        runs.extend((heuristic, policy))
    body = {
        "format": CONTROLLER_NEUTRAL_POLICY_EVALUATION_FORMAT,
        "smoke_test": not loaded.metadata.trained,
        "claim_boundary": (
            "integration smoke; no learned-controller claim"
            if not loaded.metadata.trained
            else "descriptive paired evaluation; no statistical or commercial claim"
        ),
        "training_task": config.to_dict(),
        "checkpoint": loaded.to_dict(),
        "seeds": list(seed_values),
        "controller_order": [
            LegacyHeuristicSetpointControllerV1.id,
            NORMALIZED_POLICY_SETPOINT_CONTROLLER_ID,
        ],
        "runs": runs,
    }
    body["deterministic_evidence_sha256"] = domain_sha256(
        "controller-neutral-policy-evaluation-evidence-v1",
        {
            "format": body["format"],
            "smoke_test": body["smoke_test"],
            "training_task": body["training_task"],
            "checkpoint_metadata": loaded.metadata.to_dict(),
            "seeds": body["seeds"],
            "controller_order": body["controller_order"],
            "run_evidence": [run["deterministic_evidence_sha256"] for run in runs],
        },
    )
    body["sha256"] = domain_sha256(
        CONTROLLER_NEUTRAL_POLICY_EVALUATION_FORMAT,
        body,
    )
    return body


def load_training_task(path: Path | str) -> ControllerNeutralTrainingConfigV1:
    metadata = checkpoints.load_metadata(path, check=False)
    if not metadata.training_config:
        raise checkpoints.CorruptCheckpointError(
            f"{path} has no controller-neutral training task in its metadata"
        )
    return ControllerNeutralTrainingConfigV1.from_dict(metadata.training_config)


def load_policy_for_task(
    path: Path | str,
    config: ControllerNeutralTrainingConfigV1,
) -> LoadedPolicy:
    environment = make_env(config)
    try:
        return load_policy(
            path,
            contract=policy_contract_for_environment(environment),
            environment=environment,
        )
    finally:
        environment.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a compatible four-target RL checkpoint against the heuristic."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--json", type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.seed_start < 0 or arguments.count <= 0:
            raise ValueError(
                "seed-start must be non-negative and count must be positive"
            )
        if arguments.json is not None and arguments.json.exists():
            raise ValueError(
                f"evaluation output already exists at {arguments.json}; choose a new path"
            )
        config = load_training_task(arguments.checkpoint)
        loaded = load_policy_for_task(arguments.checkpoint, config)
        result = compare_policy_to_heuristic(
            loaded,
            config,
            seeds=range(arguments.seed_start, arguments.seed_start + arguments.count),
        )
    except (
        checkpoints.CheckpointError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as reason:
        print(f"controller-neutral evaluation refused: {reason}", file=sys.stderr)
        return 2
    summary = {
        "format": result["format"],
        "smoke_test": result["smoke_test"],
        "seeds": result["seeds"],
        "runs": [
            {
                "controller": run["controller_id"],
                "seed": run["seed"],
                "status": run["status"],
                "reward": run["reward_trace_sum"],
            }
            for run in result["runs"]
        ],
    }
    print(json.dumps(summary, indent=2))
    if arguments.json is not None:
        try:
            arguments.json.parent.mkdir(parents=True, exist_ok=True)
            with arguments.json.open("x", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
                )
        except OSError as reason:
            print(
                f"controller-neutral evaluation could not be saved: {reason}",
                file=sys.stderr,
            )
            return 2
    return 0 if all(run["status"] == "complete" for run in result["runs"]) else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CONTROLLER_NEUTRAL_POLICY_EVALUATION_FORMAT",
    "compare_policy_to_heuristic",
    "execute_heuristic_episode",
    "execute_policy_episode",
    "load_policy_for_task",
    "load_training_task",
    "main",
]

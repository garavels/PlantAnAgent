"""Residual-policy development evidence with independent saved-command replay."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import research_evaluation as physical
from .config import ControllerNeutralTrainingConfigV1
from .custody_json import canonical_bytes
from .research_residual import (
    ResearchResidualHWrapper,
    extend_observation,
    prepare_residual_action,
)
from ..beta_controller_benchmark_verifier import _decode_command
from ..setpoint_controllers import LegacyHeuristicSetpointControllerV1

SCHEMA = "plant-adaptive-residual-episode-v1"
ACTOR = "ResidualPPO"


def _source() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _list(vector: Any) -> list[float]:
    return [float(value) for value in np.asarray(vector)]


def _recorded_action(row: dict[str, Any], *, failed_inference: bool = False) -> Any:
    raw, shape = row["raw_action"], row["raw_action_shape"]
    if (
        not isinstance(raw, list)
        or not isinstance(shape, list)
        or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in shape)
        or math.prod(shape) != len(raw)
        or any(
            isinstance(value, bool)
            or not (
                isinstance(value, (float, int))
                or (failed_inference and value in ("nan", "inf", "-inf"))
            )
            for value in raw
        )
    ):
        raise ValueError("residual replay action serialization differs")
    array = np.asarray(raw, dtype=np.float64)
    if not failed_inference and (shape != [4] or not np.isfinite(array).all()):
        raise ValueError("residual replay action is invalid")
    return array


def _check_mapping(base: Any, decision: Any, row: dict[str, Any], digest: str) -> None:
    raw = _recorded_action(row)
    command, invalid, residual = prepare_residual_action(
        base, decision, raw, interface_contract_digest=digest
    )
    if (
        not isinstance(row["controller_invalid"], bool)
        or row["controller_invalid"] is not invalid
        or not physical._json_equal(row["command"], command.to_dict())
        or not physical._json_equal(row["residual_evidence"], residual)
    ):
        raise ValueError("residual replay command mapping differs")


def evaluate_residual_episode(
    config: ControllerNeutralTrainingConfigV1,
    *,
    policy: Any,
    seed: int,
    fixture: bool = False,
    max_decisions: int | None = None,
    policy_id: str = "research-residual-ppo",
) -> dict[str, Any]:
    """Evaluate deterministic 139-input residual inference through the common plant.

    Raw physical observations remain 135-dimensional in the physical evidence;
    the actual augmented policy input is recorded separately. Checkpoint custody
    belongs to the caller. No registered or live-policy admission is implied.
    """
    physical._validate(seed, fixture=fixture, max_decisions=max_decisions)
    if policy is None or not isinstance(policy_id, str) or not policy_id.strip():
        raise ValueError("residual evaluation requires a policy and nonempty identity")
    base = physical._environment(config)
    wrapper = ResearchResidualHWrapper(base)
    history: list[dict[str, Any]] = []
    minutes: list[dict[str, Any]] = []
    failure = None
    pending: dict[str, Any] | None = None
    terminated = truncated = False
    phase = "reset"
    try:
        observation, _ = wrapper.reset(seed=seed)
        initial = physical._observation(base, wrapper.current_base_observation)
        initial_policy_observation = _list(observation)
        minutes.append(base.telemetry.to_dict())
        for _ in range(
            min(max_decisions or base.episode_decisions, base.episode_decisions)
        ):
            pending = {
                "decision_index": base.decisions_taken,
                "observation": physical._observation(
                    base, wrapper.current_base_observation
                ),
                "policy_observation": _list(observation),
            }
            phase = "policy_inference"
            raw, _ = policy.predict(np.asarray(observation).copy(), deterministic=True)
            array = np.asarray(raw)
            pending["raw_action"] = [
                float(value) if math.isfinite(float(value)) else str(float(value))
                for value in array.reshape(-1)
            ]
            pending["raw_action_shape"] = list(array.shape)
            if array.shape != (4,) or not np.isfinite(array).all():
                raise ValueError("residual policy action must be four finite values")
            phase = "residual_command"
            cached_heuristic = wrapper.current_heuristic_decision
            if cached_heuristic is None:
                raise RuntimeError("live residual decision has no cached heuristic")
            command, invalid, residual = prepare_residual_action(
                base,
                cached_heuristic,
                array,
                interface_contract_digest=wrapper.interface_contract_digest,
            )
            pending.update(
                command=command.to_dict(),
                controller_invalid=invalid,
                residual_evidence=residual,
            )
            phase = "environment_execution"
            observation, reward, terminated, truncated, info = wrapper.step(array)
            phase = "evidence_capture"
            if not physical._json_equal(residual, wrapper.last_residual_evidence):
                raise RuntimeError(
                    "executed residual evidence differs from prepared command"
                )
            transition = (
                wrapper.current_base_observation,
                reward,
                terminated,
                truncated,
                info,
            )
            pending.update(physical._interval(base, transition, minutes, actor=ACTOR))
            pending["after_policy_observation"] = _list(observation)
            history.append(pending)
            pending = None
            if terminated or truncated:
                break
    except Exception as exc:
        failure = physical._failure(phase, exc)
        if phase in {"reset", "environment_execution", "evidence_capture"}:
            wrapper.close()
            # Physical/evidence exceptions may leave a partially advanced plant.
            # No replayable controller-prefix claim can be made from that state.
            # The batch artifact child treats this as an integrity failure.
            raise RuntimeError(
                f"residual episode integrity failure in {phase}"
            ) from exc
    try:
        result = {
            "schema": SCHEMA,
            "seed": seed,
            "fixture": fixture,
            "prefix_limit": max_decisions,
            "config": config.to_dict(),
            "evaluator_source_sha256": _source(),
            "interface_contract": wrapper.interface_contract,
            "interface_contract_digest": wrapper.interface_contract_digest,
            "manifest": base.audit_manifest(),
            "controller": {
                "id": policy_id,
                "deterministic": True,
                "interface": "residual-H-139",
            },
            "initial_observation": initial,
            "initial_policy_observation": initial_policy_observation,
            "final_observation": physical._observation(
                base, wrapper.current_base_observation
            ),
            "final_policy_observation": _list(wrapper.current_observation),
            "history": history,
            "minute_truth_states": minutes,
            "failed_decision": pending,
            "summary": physical._summary(
                base,
                actor=ACTOR,
                policy_id=policy_id,
                history=history,
                minute_states=minutes,
                failure=failure,
                terminated=terminated,
                truncated=truncated,
            ),
            "claim": "Adaptive residual development only; no selection, held-out, registered or live admission.",
        }
        canonical_bytes(result)
        return result
    finally:
        wrapper.close()


def verify_residual_episode_replay(
    config: ControllerNeutralTrainingConfigV1,
    evidence: dict[str, Any],
    *,
    fixture: bool = False,
) -> dict[str, Any]:
    """Replay stored absolute commands, separately verifying their residual mapping.

    The learned policy is never loaded or run. H is recomputed on each public
    boundary to check derived features and mapped-command identity, but physical
    replay executes the independently decoded saved command. It does not prove
    that the checkpoint produced a stored residual or that an exception occurred.
    """
    canonical_bytes(evidence)
    physical._validate(
        evidence["seed"], fixture=fixture, max_decisions=evidence["prefix_limit"]
    )
    if (
        evidence["schema"] != SCHEMA
        or evidence["fixture"] is not fixture
        or not physical._json_equal(evidence["config"], config.to_dict())
        or evidence["evaluator_source_sha256"] != _source()
        or evidence["summary"]["actor"] != ACTOR
    ):
        raise ValueError(
            "residual replay schema, authority, configuration or source differs"
        )
    identity = evidence["summary"]["policy_id"]
    if (
        not isinstance(identity, str)
        or not identity.strip()
        or not physical._json_equal(
            evidence["controller"],
            {"id": identity, "deterministic": True, "interface": "residual-H-139"},
        )
    ):
        raise ValueError("residual replay controller identity differs")
    base = physical._environment(config)
    contract_wrapper = ResearchResidualHWrapper(base)
    try:
        if (
            not physical._json_equal(
                evidence["interface_contract"], contract_wrapper.interface_contract
            )
            or evidence["interface_contract_digest"]
            != contract_wrapper.interface_contract_digest
        ):
            raise ValueError("residual replay interface differs")
        observation, _ = base.reset(seed=evidence["seed"])
        controller = LegacyHeuristicSetpointControllerV1()

        def heuristic() -> Any:
            return controller.compute_command(
                base.controller_observation,
                constraints=base.constraints,
                config=base.config,
                commitment_profile=base.episode_config.electrolyser_commitment,
            )

        decision = heuristic()
        augmented = extend_observation(observation, base, decision)
        if not physical._json_equal(
            evidence["initial_observation"], physical._observation(base, observation)
        ) or not physical._json_equal(
            evidence["initial_policy_observation"], _list(augmented)
        ):
            raise ValueError("residual replay initial observation differs")
        minutes = [base.telemetry.to_dict()]
        terminated = truncated = False
        history = evidence["history"]
        for index, row in enumerate(history):
            if (
                row["decision_index"] != index
                or not physical._json_equal(
                    row["observation"], physical._observation(base, observation)
                )
                or not physical._json_equal(row["policy_observation"], _list(augmented))
            ):
                raise ValueError(
                    "residual replay policy input or decision boundary differs"
                )
            _check_mapping(
                base, decision, row, contract_wrapper.interface_contract_digest
            )
            transition = base.step_process_setpoints(
                _decode_command(row["command"], name="research residual command"),
                controller_invalid=row["controller_invalid"],
            )
            actual = physical._interval(base, transition, minutes, actor=ACTOR)
            if any(
                not physical._json_equal(row[key], value)
                for key, value in actual.items()
            ):
                raise ValueError(
                    f"residual replay interval {index} physical evidence differs"
                )
            observation, _, terminated, truncated, _ = transition
            decision = None if terminated or truncated else heuristic()
            augmented = extend_observation(observation, base, decision)
            if not physical._json_equal(
                row["after_policy_observation"], _list(augmented)
            ):
                raise ValueError("residual replay returned policy input differs")
        if not physical._json_equal(minutes, evidence["minute_truth_states"]):
            raise ValueError("residual replay minute truth differs")
        pending = evidence["failed_decision"]
        failure = evidence["summary"]["failure"]
        if (pending is None) != (failure is None):
            raise ValueError("residual replay failure/pending coherence differs")
        if failure is None:
            limit = min(
                evidence["prefix_limit"] or base.episode_decisions,
                base.episode_decisions,
            )
            if len(history) > limit or (
                len(history) < limit and not (terminated or truncated)
            ):
                raise ValueError("residual replay declared stopping budget differs")
        else:
            if (
                not isinstance(failure, dict)
                or set(failure) != {"phase", "type", "message"}
                or failure["phase"] not in {"policy_inference", "residual_command"}
                or not isinstance(failure["type"], str)
                or not failure["type"]
                or not isinstance(failure["message"], str)
                or not isinstance(pending, dict)
            ):
                raise ValueError(
                    "residual replay controller failure descriptor differs"
                )
            expected_keys = {"decision_index", "observation", "policy_observation"}
            if "raw_action" in pending or "raw_action_shape" in pending:
                expected_keys |= {"raw_action", "raw_action_shape"}
            if set(pending) != expected_keys:
                raise ValueError("residual replay failed decision fields differ")
            if failure["phase"] == "residual_command" and "raw_action" not in pending:
                raise ValueError("residual replay failed mapping omits its input")
            if "raw_action" in pending:
                _recorded_action(
                    pending, failed_inference=failure["phase"] == "policy_inference"
                )
        if pending is not None and (
            pending["decision_index"] != len(history)
            or not physical._json_equal(
                pending["observation"], physical._observation(base, observation)
            )
            or not physical._json_equal(pending["policy_observation"], _list(augmented))
        ):
            raise ValueError("residual replay failed decision boundary differs")
        if (
            not physical._json_equal(
                evidence["final_observation"], physical._observation(base, observation)
            )
            or not physical._json_equal(
                evidence["final_policy_observation"], _list(augmented)
            )
            or not physical._json_equal(evidence["manifest"], base.audit_manifest())
        ):
            raise ValueError("residual replay final observation or manifest differs")
        summary = physical._summary(
            base,
            actor=ACTOR,
            policy_id=evidence["summary"]["policy_id"],
            history=history,
            minute_states=minutes,
            failure=evidence["summary"]["failure"],
            terminated=terminated,
            truncated=truncated,
        )
        if not physical._json_equal(summary, evidence["summary"]):
            raise ValueError("residual replay outcomes, safety or settlements differ")
        return {
            "schema": "plant-adaptive-residual-replay-v1",
            "verified": True,
            "seed": evidence["seed"],
            "actor": ACTOR,
            "decisions": len(history),
            "physical_steps": base.physics_steps_taken,
            "execution_complete": summary["execution_complete"],
            "scope": (
                "complete_episode"
                if summary["execution_complete"]
                else "retained_physical_prefix"
            ),
            "saved_absolute_commands_executed": True,
            "heuristic_reexecuted_for_mapping_check": True,
            "learned_policy_reexecuted": False,
            "controller_failure_authenticity_verified": False,
            "registered_evaluation_eligible": False,
            "evidence_sha256": hashlib.sha256(canonical_bytes(evidence)).hexdigest(),
        }
    finally:
        contract_wrapper.close()

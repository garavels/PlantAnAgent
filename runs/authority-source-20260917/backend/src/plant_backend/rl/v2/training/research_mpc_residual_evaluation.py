"""MPC residual development evidence with detached saved-command replay."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import research_evaluation as physical
from .config import ControllerNeutralTrainingConfigV1
from .custody_json import canonical_bytes
from .research_mpc_residual import (
    EVIDENCE_SCHEMA,
    ENVIRONMENT_ID,
    MPCResidualIntegrityError,
    ResearchMPCResidualWrapper,
    make_mpc_controller,
    mpc_decision_record,
    extend_observation,
    prepare_residual_action,
)
from ..beta_controller_benchmark_verifier import _decode_command
from ..economic_mpc_schema import ECONOMIC_MPC_PLAN_DOMAIN
from ..mpc_schema import domain_sha256
from ....operational_mpc import DECISION_SCHEMA, POLICY_ID

SCHEMA = "plant-adaptive-mpc-residual-episode-v1"
ACTOR = "MPCResidualPPO"


def _zero_stage_times(stages: Any) -> None:
    """Normalize only the declared solver clock field, never solver semantics."""
    if not isinstance(stages, list):
        raise ValueError("MPC stage evidence differs")
    for stage in stages:
        if (
            not isinstance(stage, dict)
            or isinstance(stage.get("elapsed_ns"), bool)
            or not isinstance(stage.get("elapsed_ns"), int)
            or stage["elapsed_ns"] < 0
        ):
            raise ValueError("MPC stage timing differs")
        stage["elapsed_ns"] = 0


def operation_semantic_projection(operation: dict[str, Any]) -> dict[str, Any]:
    """Keep the entire operation, except the explicitly recorded solver clock."""
    result = deepcopy(operation)
    failure = result.get("solver_failure")
    if isinstance(failure, dict) and "stages" in failure:
        _zero_stage_times(failure["stages"])
    return result


def mpc_decision_semantic_projection(record: dict[str, Any]) -> dict[str, Any]:
    """Validate raw digests, then rehash only clock-dependent descendants.

    This is not a recursive hash/timing filter. Plan stage clocks and fallback
    stage clocks are the two permitted differences. Stage messages, solution
    hashes, statuses, limits, targets, public bindings and every plan value stay.
    """
    if not isinstance(record, dict) or set(record) != {
        "command",
        "invalid",
        "evidence",
    }:
        raise ValueError("MPC decision record fields differ")
    if not isinstance(record["invalid"], bool):
        raise ValueError("MPC invalid flag differs")
    result = deepcopy(record)
    original = _decode_command(record["command"], name="recorded nominal MPC command")
    evidence = result["evidence"]
    if (
        not isinstance(evidence, dict)
        or set(evidence) != {"payload", "sha256", "source_notes", "plan"}
        or not isinstance(evidence["source_notes"], list)
        or not all(isinstance(note, str) for note in evidence["source_notes"])
    ):
        raise ValueError("MPC source evidence fields differ")
    payload = evidence["payload"]
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != DECISION_SCHEMA
        or evidence["sha256"] != domain_sha256(DECISION_SCHEMA, payload)
        or original.source_schema != DECISION_SCHEMA
        or original.issuer_id != POLICY_ID
        or original.source_sha256 != evidence["sha256"]
        or not physical._json_equal(payload.get("targets"), original.targets.to_dict())
    ):
        raise ValueError("MPC payload or command digest binding differs")
    plan = evidence["plan"]
    if plan is not None:
        if not isinstance(plan, dict) or not isinstance(plan.get("stages"), list):
            raise ValueError("MPC plan evidence differs")
        body = {key: value for key, value in plan.items() if key != "sha256"}
        body["stages"] = [
            {"stage": stage["stage"], "solution_sha256": stage["solution_sha256"]}
            for stage in plan["stages"]
        ]
        if (
            plan.get("sha256") != domain_sha256(ECONOMIC_MPC_PLAN_DOMAIN, body)
            or payload.get("plan_sha256") != plan["sha256"]
        ):
            raise ValueError("MPC plan digest binding differs")
        _zero_stage_times(plan["stages"])
    elif payload.get("plan_sha256") is not None:
        raise ValueError("MPC absent plan binding differs")
    operation = payload.get("operation")
    if not isinstance(operation, dict):
        raise ValueError("MPC operation evidence differs")
    payload["operation"] = operation_semantic_projection(operation)
    evidence["sha256"] = domain_sha256(DECISION_SCHEMA, payload)
    result["command"] = replace(original, source_sha256=evidence["sha256"]).to_dict()
    return result


def residual_evidence_semantic_projection(record: dict[str, Any]) -> dict[str, Any]:
    """Validate the issued residual chain before normalizing its MPC clocks."""
    if (
        not isinstance(record, dict)
        or set(record) != {"status", "body", "sha256", "command", "controller_invalid"}
        or record["status"] != "issued"
        or not isinstance(record["controller_invalid"], bool)
    ):
        raise ValueError("MPC residual evidence fields differ")
    result = deepcopy(record)
    body = result["body"]
    if (
        body.get("schema") != EVIDENCE_SCHEMA
        or record["sha256"] != domain_sha256(EVIDENCE_SCHEMA, body)
        or not isinstance(body.get("exact_zero_mpc_command"), bool)
    ):
        raise ValueError("MPC residual evidence digest differs")
    command = _decode_command(record["command"], name="recorded MPC residual command")
    if body["exact_zero_mpc_command"]:
        if not physical._json_equal(record["command"], body["mpc_decision"]["command"]):
            raise ValueError("MPC exact zero command differs")
    elif (
        command.issuer_id != ENVIRONMENT_ID
        or command.source_schema != EVIDENCE_SCHEMA
        or command.source_sha256 != record["sha256"]
    ):
        raise ValueError("MPC residual command source binding differs")
    if not physical._json_equal(body["mapped_targets"], command.targets.to_dict()):
        raise ValueError("MPC residual target binding differs")
    body["mpc_decision"] = mpc_decision_semantic_projection(body["mpc_decision"])
    result["sha256"] = domain_sha256(EVIDENCE_SCHEMA, body)
    result["command"] = (
        body["mpc_decision"]["command"]
        if body["exact_zero_mpc_command"]
        else replace(command, source_sha256=result["sha256"]).to_dict()
    )
    return result


def _check_prior(record: Any, decision: Any) -> None:
    actual = None if decision is None else mpc_decision_record(decision)
    if record is None or actual is None:
        if record is not actual:
            raise ValueError("MPC residual terminal prior differs")
    elif not physical._json_equal(
        mpc_decision_semantic_projection(record),
        mpc_decision_semantic_projection(actual),
    ):
        raise ValueError("MPC residual causal prior semantics differ")


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
        or not physical._json_equal(row["command"], row["residual_evidence"]["command"])
        or not physical._json_equal(
            row["mpc_decision"], row["residual_evidence"]["body"]["mpc_decision"]
        )
        or not physical._json_equal(
            residual_evidence_semantic_projection(row["residual_evidence"]),
            residual_evidence_semantic_projection(residual),
        )
    ):
        raise ValueError("residual replay command mapping differs")


def evaluate_residual_episode(
    config: ControllerNeutralTrainingConfigV1,
    *,
    policy: Any,
    seed: int,
    fixture: bool = False,
    max_decisions: int | None = None,
    policy_id: str = "research-mpc-residual-ppo",
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
    wrapper = ResearchMPCResidualWrapper(base)
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
        first_mpc = wrapper.current_mpc_decision
        if first_mpc is None:
            raise MPCResidualIntegrityError("reset omitted the initial MPC decision")
        initial_mpc_decision = mpc_decision_record(first_mpc)
        minutes.append(base.telemetry.to_dict())
        for _ in range(
            min(max_decisions or base.episode_decisions, base.episode_decisions)
        ):
            phase = "prior_integrity"
            cached_mpc = wrapper.current_mpc_decision
            if cached_mpc is None:
                raise MPCResidualIntegrityError("live decision has no cached MPC")
            pending = {
                "decision_index": base.decisions_taken,
                "observation": physical._observation(
                    base, wrapper.current_base_observation
                ),
                "policy_observation": _list(observation),
                "mpc_decision": mpc_decision_record(cached_mpc),
                "operation": cached_mpc.evidence.operation(),
            }
            phase = "policy_inference"
            raw, _ = policy.predict(np.asarray(observation).copy(), deterministic=True)
            array = np.asarray(raw)
            if array.dtype.kind not in "iuf":
                raise ValueError(
                    "residual policy action must contain real numeric values"
                )
            pending["raw_action"] = [
                float(value) if math.isfinite(float(value)) else str(float(value))
                for value in array.reshape(-1)
            ]
            pending["raw_action_shape"] = list(array.shape)
            if (
                array.shape != (4,)
                or not np.isfinite(np.asarray(array, dtype=np.float64)).all()
            ):
                raise ValueError("residual policy action must be four finite values")
            phase = "prior_integrity"
            # A known-valid residual separates lost public/cache/authority
            # bindings from a rejection of the caller's command. No MPC solve,
            # observation encoding, reset or physical step is performed here.
            prepare_residual_action(
                base,
                cached_mpc,
                np.zeros(4, dtype=np.float64),
                interface_contract_digest=wrapper.interface_contract_digest,
            )
            phase = "residual_command"
            command, invalid, residual = prepare_residual_action(
                base,
                cached_mpc,
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
        if isinstance(exc, MPCResidualIntegrityError) or phase in {
            "reset",
            "prior_integrity",
            "environment_execution",
            "evidence_capture",
        }:
            wrapper.close()
            # Physical/evidence exceptions may leave a partially advanced plant.
            # No replayable controller-prefix claim can be made from that state.
            # The batch artifact child treats this as an integrity failure.
            raise MPCResidualIntegrityError(
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
                "interface": "residual-MPC-139",
            },
            "initial_observation": initial,
            "initial_policy_observation": initial_policy_observation,
            "initial_mpc_decision": initial_mpc_decision,
            "final_observation": physical._observation(
                base, wrapper.current_base_observation
            ),
            "final_policy_observation": _list(wrapper.current_observation),
            "final_mpc_decision": (
                None
                if wrapper.current_mpc_decision is None
                else mpc_decision_record(wrapper.current_mpc_decision)
            ),
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

    The learned policy is never loaded or run. Causal, stateful MPC is recomputed
    on every nonterminal public boundary, including a fixture prefix's final
    cached boundary. Raw evidence is hash checked before narrowly normalizing
    solver clocks and their derived hashes. Physical replay executes the saved
    command unchanged. This neither proves policy inference nor authenticates a
    recorded controller exception. Callers enforce process wall-clock limits.
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
            {"id": identity, "deterministic": True, "interface": "residual-MPC-139"},
        )
    ):
        raise ValueError("residual replay controller identity differs")
    base = physical._environment(config)
    contract_wrapper = ResearchMPCResidualWrapper(base)
    try:
        if (
            not physical._json_equal(
                evidence["interface_contract"], contract_wrapper.interface_contract
            )
            or evidence["interface_contract_digest"]
            != contract_wrapper.interface_contract_digest
        ):
            raise ValueError("residual replay interface differs")
        history = evidence["history"]
        limit = min(
            evidence["prefix_limit"] or base.episode_decisions,
            base.episode_decisions,
        )
        if not isinstance(history, list) or len(history) > limit:
            raise ValueError("residual replay declared stopping budget differs")
        observation, _ = base.reset(seed=evidence["seed"])
        controller = make_mpc_controller(base)
        controller.reset()

        def prior() -> Any:
            return controller.compute_command(
                base.cyclic_market_controller_observation,
                constraints=base.constraints,
                config=base.config,
                commitment_profile=base.episode_config.electrolyser_commitment,
            )

        decision = prior()
        _check_prior(evidence["initial_mpc_decision"], decision)
        augmented = extend_observation(observation, base, decision)
        if not physical._json_equal(
            evidence["initial_observation"], physical._observation(base, observation)
        ) or not physical._json_equal(
            evidence["initial_policy_observation"], _list(augmented)
        ):
            raise ValueError("residual replay initial observation differs")
        minutes = [base.telemetry.to_dict()]
        terminated = truncated = False
        for index, row in enumerate(history):
            if (
                terminated
                or truncated
                or isinstance(row["decision_index"], bool)
                or not isinstance(row["decision_index"], int)
                or row["decision_index"] != index
                or not physical._json_equal(
                    row["observation"], physical._observation(base, observation)
                )
                or not physical._json_equal(row["policy_observation"], _list(augmented))
            ):
                raise ValueError(
                    "residual replay policy input or decision boundary differs"
                )
            _check_prior(row["mpc_decision"], decision)
            if not physical._json_equal(
                row["operation"],
                row["mpc_decision"]["evidence"]["payload"]["operation"],
            ):
                raise ValueError("residual replay operation binding differs")
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
            decision = None if terminated or truncated else prior()
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
            if len(history) > limit or (
                len(history) < limit and not (terminated or truncated)
            ):
                raise ValueError("residual replay declared stopping budget differs")
        else:
            if (
                len(history) >= limit
                or terminated
                or truncated
                or not isinstance(failure, dict)
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
            expected_keys = {
                "decision_index",
                "observation",
                "policy_observation",
                "mpc_decision",
                "operation",
            }
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
            isinstance(pending["decision_index"], bool)
            or not isinstance(pending["decision_index"], int)
            or pending["decision_index"] != len(history)
            or not physical._json_equal(
                pending["observation"], physical._observation(base, observation)
            )
            or not physical._json_equal(pending["policy_observation"], _list(augmented))
        ):
            raise ValueError("residual replay failed decision boundary differs")
        if pending is not None:
            _check_prior(pending["mpc_decision"], decision)
            if not physical._json_equal(
                pending["operation"],
                pending["mpc_decision"]["evidence"]["payload"]["operation"],
            ):
                raise ValueError("residual replay pending operation binding differs")
        _check_prior(evidence["final_mpc_decision"], decision)
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
            "schema": "plant-adaptive-mpc-residual-replay-v1",
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
            "mpc_reexecuted_for_mapping_check": True,
            "solver_timing_semantics_verified": True,
            "excluded_clock_paths": [
                "mpc_decision.evidence.plan.stages[*].elapsed_ns",
                "mpc_decision.evidence.payload.operation.solver_failure.stages[*].elapsed_ns",
            ],
            "derived_clock_hashes_recomputed": True,
            "learned_policy_reexecuted": False,
            "controller_failure_authenticity_verified": False,
            "registered_evaluation_eligible": False,
            "evidence_sha256": hashlib.sha256(canonical_bytes(evidence)).hexdigest(),
        }
    finally:
        contract_wrapper.close()

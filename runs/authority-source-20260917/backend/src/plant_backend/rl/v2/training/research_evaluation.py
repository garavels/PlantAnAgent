"""Adaptive research evidence and physical replay, outside registered admission.

Only the explicitly separate development namespace may be opened. A test-only
fixture flag permits another disjoint namespace and intentional short prefixes.
The caller owns source/checkpoint custody and publication; this module neither
loads checkpoints nor selects a policy nor claims registered-study admission.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import ppo
from .config import ControllerNeutralTrainingConfigV1
from .custody_json import canonical_bytes
from ..beta_config import RESOURCE_NAMES
from ..beta_controller_benchmark import (
    _hard_safety_classification,
    _real_beta_economic_controller,
    _service_settings,
)
from ..beta_controller_benchmark_verifier import _decode_command
from ..beta_env import PlantCyclicMarketEnvV2Beta
from ..full_day_horizon_codec import _decode_setpoints
from ..mpc_setpoint_controller import ServiceMPCSetpointControllerV1
from ..mpc_setpoint_execution import MPCMinuteApplicationV1
from ..setpoint_execution import ProcessSetpointMinuteApplicationV1
from ..setpoint_command import process_setpoint_boundary_sha256
from ..setpoint_controllers import LegacyHeuristicSetpointControllerV1
from ....operational_mpc import OperationalMPCController

SCHEMA = "plant-adaptive-research-episode-v1"
ACTORS = ("PPO", "H", "OperatingMPC", "M24", "E-beta")
DEVELOPMENT_FIRST = 9_700_000
DEVELOPMENT_LAST = 9_709_999
FIXTURE_FIRST = 9_990_000
FIXTURE_LAST = 9_990_999


def _validate(seed: int, *, fixture: bool, max_decisions: int | None) -> None:
    if (
        not isinstance(fixture, bool)
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ValueError("seed must be an integer and fixture must be boolean")
    first, last = (
        (FIXTURE_FIRST, FIXTURE_LAST)
        if fixture
        else (DEVELOPMENT_FIRST, DEVELOPMENT_LAST)
    )
    if not first <= seed <= last:
        raise ValueError("seed is outside the explicitly allowed research namespace")
    if max_decisions is not None and (
        not fixture
        or isinstance(max_decisions, bool)
        or not isinstance(max_decisions, int)
        or max_decisions <= 0
    ):
        raise ValueError("only fixtures may request a positive short prefix")


def _environment(
    config: ControllerNeutralTrainingConfigV1,
) -> PlantCyclicMarketEnvV2Beta:
    config.validate()
    if config.beta_rl is not None or config.beta_task is None:
        raise ValueError(
            "research requires a beta environment template, not a registered learner"
        )
    env = ppo.make_env(config)
    if not isinstance(env, PlantCyclicMarketEnvV2Beta):
        env.close()
        raise ValueError("research evaluation requires the production beta environment")
    return env


def _controller(config: ControllerNeutralTrainingConfigV1, actor: str) -> Any:
    if actor == "H":
        return LegacyHeuristicSetpointControllerV1()
    if actor == "OperatingMPC":
        assert config.beta_task is not None
        return OperationalMPCController(
            horizon_h=config.episode.horizon_h,
            interval_s=config.episode.decision_interval_s,
            assumptions=config.economic_assumptions,
            beta_task=config.beta_task,
        )
    if actor == "M24":
        return ServiceMPCSetpointControllerV1(_service_settings(config, None))
    if actor == "E-beta":
        return _real_beta_economic_controller(config)
    return None


def _json_equal(left: object, right: object) -> bool:
    """Compare the published JSON value, including exact scalar types.

    Dataclass ``to_dict`` methods may retain tuples. JSON turns these into arrays;
    their publication bytes remain identical after a canonical gzip round trip.
    Comparing bytes also avoids Python treating ``True`` and ``1`` as equal.
    """
    return canonical_bytes(left) == canonical_bytes(right)


def _failure(phase: str, exc: Exception) -> dict[str, str]:
    return {"phase": phase, "type": type(exc).__name__, "message": str(exc)}


def _observation(env: PlantCyclicMarketEnvV2Beta, vector: Any) -> dict[str, Any]:
    typed = env.cyclic_market_controller_observation
    return {
        "normalized": np.asarray(vector).tolist(),
        "typed": typed.to_dict(),
        "typed_sha256": typed.sha256,
    }


def _interval(
    env: PlantCyclicMarketEnvV2Beta,
    transition: Any,
    minute_states: list[dict[str, Any]],
    *,
    actor: str,
) -> dict[str, Any]:
    observation, reward, terminated, truncated, info = transition
    info = deepcopy(info)
    # Timings are not physical evidence and cannot be reproduced by replay.
    info.pop("dispatcher_elapsed_ns", None)
    execution = (
        env.last_setpoint_execution
        if actor == "M24"
        else env.last_process_setpoint_execution
    )
    truth = env.last_minute_truth_states
    if (
        execution is None
        or truth is None
        or len(truth) != len(execution.applications) + 1
    ):
        raise RuntimeError("execution omitted its complete minute truth chain")
    if truth[0].to_dict() != minute_states[-1]:
        raise RuntimeError("minute truth chain is discontinuous")
    for application, before, after in zip(
        execution.applications, truth[:-1], truth[1:], strict=True
    ):
        if actor == "M24":
            if not isinstance(application, MPCMinuteApplicationV1):
                raise TypeError("M24 application type differs")
            valid = (
                application.boundary_record == before.to_dict()
                and application.next_boundary_record == after.to_dict()
            )
        else:
            if not isinstance(application, ProcessSetpointMinuteApplicationV1):
                raise TypeError("process application type differs")
            valid = application.boundary_sha256 == process_setpoint_boundary_sha256(
                before
            ) and application.next_boundary_sha256 == process_setpoint_boundary_sha256(
                after
            )
        if not valid:
            raise RuntimeError("minute truth chain differs from dispatcher evidence")
    minute_states.extend(state.to_dict() for state in truth[1:])
    return {
        "after_observation": _observation(env, observation),
        "reward_eur": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info,
        "execution": execution.to_dict(),
    }


def _summary(
    env: PlantCyclicMarketEnvV2Beta,
    *,
    actor: str,
    policy_id: str,
    history: list[dict[str, Any]],
    minute_states: list[dict[str, Any]],
    failure: dict[str, Any] | None,
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    physical = env.metrics.to_dict()
    economic = env.economic_metrics.to_dict()
    safety_failure = None
    safety_metrics: dict[str, int | float | bool] = {}
    safety_pass = False
    try:
        safety_pass, safety_metrics = _hard_safety_classification(
            env, physical, minute_states
        )
    except Exception as exc:
        safety_failure = _failure("hard_safety_classification", exc)
    complete = (
        failure is None
        and terminated
        and not truncated
        and len(history) == env.decisions_taken == env.episode_decisions
        and env.physics_steps_taken == env.episode_physics_steps
        and len(minute_states) == env.physics_steps_taken + 1
    )
    cycles = [item.to_dict() for item in env.cycle_settlements]
    stride = env.beta_task.cycle.cycle_decisions(env.episode_config)
    expected_cycles = env.episode_decisions // stride
    all_cycles = bool(
        complete
        and len(cycles) == expected_cycles
        and all(
            row["feasible"] and row["boundary_decision"] == (index + 1) * stride
            for index, row in enumerate(cycles)
        )
    )
    modes = [row.get("operation", {}).get("mode") for row in history]
    return {
        "actor": actor,
        "policy_id": policy_id,
        "execution_complete": bool(complete),
        "decisions": len(history),
        "physical_steps": env.physics_steps_taken,
        "expected_decisions": env.episode_decisions,
        "expected_physical_steps": env.episode_physics_steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "failure": failure,
        "decision_failures": int(failure is not None),
        "hard_safety_passed": bool(safety_pass and safety_failure is None),
        "hard_safety_classifier_failure": safety_failure,
        "hard_safety_metrics": safety_metrics,
        "all_cycles_feasible": all_cycles,
        "physical_metrics": physical,
        "economic_metrics": economic,
        "cycle_settlements": cycles,
        "resource_order": list(RESOURCE_NAMES),
        "cycle_deficit_vectors": [
            [row["deficits"][name] for name in RESOURCE_NAMES] for row in cycles
        ],
        "out_of_range_decisions": physical["out_of_range_decisions"],
        "minimum_battery_soc": physical["min_battery_soc"],
        "forced_stops": physical["electrolyser_forced_stops"],
        "recovery_decisions": modes.count("recovery"),
        "fallback_decisions": modes.count("fallback"),
        "registered_evaluation_eligible": False,
    }


def evaluate_episode(
    config: ControllerNeutralTrainingConfigV1,
    *,
    actor: str,
    policy: Any = None,
    seed: int,
    fixture: bool = False,
    max_decisions: int | None = None,
    policy_id: str | None = None,
) -> dict[str, Any]:
    """Execute one complete research episode, preserving failures and every minute.

    ``PPO`` accepts an in-memory policy with deterministic ``predict``; the caller
    records the checkpoint identity. Native actors never receive the episode seed
    or private manifest. ``max_decisions`` is an explicit test-fixture seam only.
    """
    _validate(seed, fixture=fixture, max_decisions=max_decisions)
    if (
        actor not in ACTORS
        or (actor == "PPO" and policy is None)
        or (actor != "PPO" and policy is not None)
    ):
        raise ValueError("unknown actor or incompatible policy argument")
    if policy_id is not None and (
        not isinstance(policy_id, str) or not policy_id.strip()
    ):
        raise ValueError("policy_id must be non-empty text")
    env = _environment(config)
    history: list[dict[str, Any]] = []
    minutes: list[dict[str, Any]] = []
    failure = None
    pending: dict[str, Any] | None = None
    terminated = truncated = False
    phase = "reset"
    try:
        observation, _ = env.reset(seed=seed)
        initial = _observation(env, observation)
        minutes.append(env.telemetry.to_dict())
        controller = _controller(config, actor)
        descriptor = (
            {"id": policy_id or "research-ppo", "deterministic": True}
            if actor == "PPO"
            else controller.to_dict()
        )
        identity = policy_id or str(descriptor["id"])
        phase = "controller_compute"
        for _ in range(
            min(max_decisions or env.episode_decisions, env.episode_decisions)
        ):
            pending = {
                "decision_index": env.decisions_taken,
                "observation": _observation(env, observation),
            }
            phase = "controller_compute"
            if actor == "PPO":
                raw, _ = policy.predict(
                    np.asarray(observation).copy(), deterministic=True
                )
                array = np.asarray(raw)
                # Invalid outputs remain visible but cannot produce non-finite JSON.
                pending["raw_action"] = [
                    float(x) if math.isfinite(float(x)) else str(float(x))
                    for x in array.reshape(-1)
                ]
                pending["raw_action_shape"] = list(array.shape)
                if array.shape != (4,) or not np.isfinite(array).all():
                    raise ValueError(
                        "policy action must be four finite normalized targets"
                    )
                phase = "environment_execution"
                transition = env.step(array)
            elif actor == "M24":
                decision = controller.compute_setpoints(
                    env.controller_observation,
                    constraints=env.constraints,
                    config=env.config,
                    commitment_profile=env.episode_config.electrolyser_commitment,
                )
                pending["command"] = decision.setpoints.to_dict()
                pending["controller_invalid"] = False
                phase = "environment_execution"
                transition = env.step_setpoints(decision.setpoints)
            else:
                public = (
                    env.controller_observation
                    if actor == "H"
                    else env.cyclic_market_controller_observation
                )
                decision = controller.compute_command(
                    public,
                    constraints=env.constraints,
                    config=env.config,
                    commitment_profile=env.episode_config.electrolyser_commitment,
                )
                pending["command"] = decision.command.to_dict()
                pending["controller_invalid"] = decision.invalid
                if actor == "OperatingMPC":
                    pending["operation"] = decision.evidence.operation()
                    pending["controller_evidence"] = decision.evidence.payload_json
                else:
                    pending["controller_evidence"] = decision.to_dict()
                phase = "environment_execution"
                transition = env.step_process_setpoints(
                    decision.command, controller_invalid=decision.invalid
                )
            phase = "evidence_capture"
            pending.update(_interval(env, transition, minutes, actor=actor))
            history.append(pending)
            pending = None
            observation, _, terminated, truncated, _ = transition
            if actor == "M24":
                phase = "controller_reconciliation"
                controller.observe_transition(
                    env.controller_observation, execution=env.last_setpoint_execution
                )
            if terminated or truncated:
                break
    except Exception as exc:
        failure = _failure(phase, exc)
        if phase == "reset":
            env.close()
            raise  # No physical prefix or task manifest exists to substantiate an episode.
    try:
        result = {
            "schema": SCHEMA,
            "seed": seed,
            "fixture": fixture,
            "prefix_limit": max_decisions,
            "config": config.to_dict(),
            "evaluator_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "manifest": env.audit_manifest(),
            "controller": descriptor,
            "initial_observation": initial,
            "final_observation": _observation(env, observation),
            "history": history,
            "minute_truth_states": minutes,
            "failed_decision": pending,
            "summary": _summary(
                env,
                actor=actor,
                policy_id=identity,
                history=history,
                minute_states=minutes,
                failure=failure,
                terminated=terminated,
                truncated=truncated,
            ),
            "claim": "Adaptive research only. No registered admission, policy selection, or held-out claim.",
        }
        canonical_bytes(result)
        return result
    finally:
        env.close()


def verify_episode_replay(
    config: ControllerNeutralTrainingConfigV1,
    evidence: dict[str, Any],
    *,
    fixture: bool = False,
) -> dict[str, Any]:
    """Replay saved commands through production physics without any controller.

    Exact equality covers every physical minute, public observation, reward,
    execution/guard record, economics and daily settlement. This verifies physical
    consequences, not that a claimed model generated a command or threw an error.
    The caller must separately bind source, checkpoint and raw artifact custody.
    """
    canonical_bytes(evidence)
    _validate(evidence["seed"], fixture=fixture, max_decisions=evidence["prefix_limit"])
    if (
        evidence["schema"] != SCHEMA
        or evidence["fixture"] is not fixture
        or not _json_equal(evidence["config"], config.to_dict())
    ):
        raise ValueError("replay schema, fixture authority or configuration differs")
    if (
        evidence["evaluator_source_sha256"]
        != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ):
        raise ValueError("replay evaluator source differs")
    actor = evidence["summary"]["actor"]
    if actor not in ACTORS:
        raise ValueError("unknown replay actor")
    env = _environment(config)
    try:
        observation, _ = env.reset(seed=evidence["seed"])
        if not _json_equal(
            evidence["initial_observation"], _observation(env, observation)
        ):
            raise ValueError("replay initial public observation differs")
        minutes = [env.telemetry.to_dict()]
        terminated = truncated = False
        history = evidence["history"]
        for index, row in enumerate(history):
            if row["decision_index"] != index or not _json_equal(
                row["observation"], _observation(env, observation)
            ):
                raise ValueError("replay decision public observation differs")
            if actor == "PPO":
                raw = np.asarray(row["raw_action"], dtype=np.float64)
                if (
                    list(raw.shape) != row["raw_action_shape"]
                    or raw.shape != (4,)
                    or not np.isfinite(raw).all()
                ):
                    raise ValueError("replay action shape or values are invalid")
                transition = env.step(raw)
            elif actor == "M24":
                transition = env.step_setpoints(_decode_setpoints(row["command"]))
            else:
                if not isinstance(row["controller_invalid"], bool):
                    raise ValueError("replay controller invalid flag is not boolean")
                transition = env.step_process_setpoints(
                    _decode_command(row["command"], name="research command"),
                    controller_invalid=row["controller_invalid"],
                )
            actual = _interval(env, transition, minutes, actor=actor)
            if any(not _json_equal(row[key], value) for key, value in actual.items()):
                raise ValueError(f"replay interval {index} evidence differs")
            observation, _, terminated, truncated, _ = transition
        if not _json_equal(minutes, evidence["minute_truth_states"]):
            raise ValueError("replay minute truth chain differs")
        pending = evidence["failed_decision"]
        if pending is not None and (
            pending["decision_index"] != len(history)
            or not _json_equal(pending["observation"], _observation(env, observation))
        ):
            raise ValueError(
                "replay failed decision boundary differs from retained prefix"
            )
        if not _json_equal(
            evidence["final_observation"], _observation(env, observation)
        ):
            raise ValueError("replay final public observation differs")
        if not _json_equal(evidence["manifest"], env.audit_manifest()):
            raise ValueError("replay physical source manifest differs")
        expected = _summary(
            env,
            actor=actor,
            policy_id=evidence["summary"]["policy_id"],
            history=history,
            minute_states=minutes,
            failure=evidence["summary"]["failure"],
            terminated=terminated,
            truncated=truncated,
        )
        if not _json_equal(expected, evidence["summary"]):
            raise ValueError("replay outcome metrics, safety or settlements differ")
        return {
            "schema": "plant-adaptive-research-replay-v1",
            "verified": True,
            "seed": evidence["seed"],
            "actor": actor,
            "decisions": len(history),
            "physical_steps": env.physics_steps_taken,
            "execution_complete": expected["execution_complete"],
            "scope": (
                "complete_episode"
                if expected["execution_complete"]
                else "retained_physical_prefix"
            ),
            "controller_reexecuted": False,
            "controller_failure_authenticity_verified": False,
            "registered_evaluation_eligible": False,
            "evidence_sha256": hashlib.sha256(canonical_bytes(evidence)).hexdigest(),
        }
    finally:
        env.close()

"""Research-only operating-effort residual around the unchanged heuristic.

This restricted prior is neither a feasibility supervisor nor a safety guarantee.
The wrapper consumes the base environment's returned observation exactly once;
the allocator belongs outside this wrapper so each physical transition is counted.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import numpy as np

from ....controllers import base as controller_base, heuristic
from .. import beta_spaces, setpoint_command, setpoint_controllers
from ..beta_env import PlantCyclicMarketEnvV2Beta
from ..contract import contract_digest
from ..mpc_schema import domain_sha256, require_sha256
from ..setpoint_command import (
    ProcessSetpointCommandV1,
    issue_process_setpoint_command,
    resolve_process_setpoint_envelope,
)
from ..setpoint_control import ProcessSetpointTargetsV1
from ..setpoint_controllers import (
    LegacyHeuristicSetpointControllerV1,
    LegacyHeuristicSetpointDecisionV1,
)


ENVIRONMENT_ID = "plant-research-residual-h-beta-v1"
INTERFACE_SCHEMA = "plant-research-residual-h-interface-v1"
EVIDENCE_SCHEMA = "plant-research-residual-h-decision-v1"
OBSERVATION_SIZE = 139
ACTION_SIZE = 4
TARGET_NAMES = (
    "electrolyser_target_kw",
    "dac_target_kw",
    "methanator_target_kw",
    "battery_charge_ceiling_kw",
)
HEURISTIC_FEATURE_NAMES = tuple(f"heuristic_{name}_fraction" for name in TARGET_NAMES)
OBSERVATION_NAMES = (*beta_spaces.OBSERVATION_NAMES, *HEURISTIC_FEATURE_NAMES)
ACTION_NAMES = tuple(f"residual_{name}" for name in TARGET_NAMES)


def _safe_input(value: Any) -> Any:
    """Preserve rejected numeric evidence without serializing NaN or infinity."""
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "values": _safe_input(value.tolist())}
    if isinstance(value, (list, tuple)):
        return [_safe_input(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else str(number)
    if value is None or isinstance(value, str):
        return value
    return {"rejected_type": type(value).__name__}


def _residual(
    action: Any,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]
]:
    array = np.asarray(action, dtype=object)
    if array.shape != (ACTION_SIZE,):
        raise ValueError("research residual action must have shape (4,)")
    for value in array:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError("research residual action must contain real numbers")
        if not math.isfinite(float(value)):
            raise ValueError("research residual action must be finite")
    raw = np.asarray(array, dtype=np.float64)
    return raw, np.clip(raw, -1.0, 1.0)


def _installed_ratings(
    base: PlantCyclicMarketEnvV2Beta,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.asarray(
        [
            base.config.electrolyser.rated_kw,
            base.config.dac.rated_kw,
            base.config.methanator.rated_kw,
            base.config.battery.max_charge_kw,
        ],
        dtype=np.float64,
    )


def _floor(base: PlantCyclicMarketEnvV2Beta) -> float:
    profile = base.episode_config.electrolyser_commitment
    if profile is None:
        raise ValueError("residual prior requires the declared commitment profile")
    return float(profile.min_stable_load_frac * base.config.electrolyser.rated_kw)


def extend_observation(
    base_vector: Any,
    base_env: PlantCyclicMarketEnvV2Beta,
    heuristic_decision: LegacyHeuristicSetpointDecisionV1 | None,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Append cached H features without observing, resetting or computing H."""
    if (
        not isinstance(base_vector, np.ndarray)
        or base_vector.dtype != np.float32
        or base_vector.shape != (135,)
        or not np.isfinite(base_vector).all()
    ):
        raise ValueError("base must return its finite float32 135-field observation")
    features = np.zeros(4, dtype=np.float32)
    if heuristic_decision is not None:
        if not isinstance(heuristic_decision, LegacyHeuristicSetpointDecisionV1):
            raise TypeError("heuristic decision must be a cached H decision or None")
        if heuristic_decision.command.decision_index != base_env.decisions_taken:
            raise ValueError("heuristic features belong to another decision")
        values = np.asarray(
            [
                getattr(heuristic_decision.command.targets, name)
                for name in TARGET_NAMES
            ],
            dtype=np.float64,
        )
        features = np.clip(values / _installed_ratings(base_env), 0.0, 1.0).astype(
            np.float32
        )
    return np.concatenate((base_vector, features))


def describe_interface_contract(base: PlantCyclicMarketEnvV2Beta) -> dict[str, Any]:
    modules = (heuristic, controller_base, setpoint_controllers, setpoint_command)
    sources = {
        module.__name__: hashlib.sha256(
            Path(str(module.__file__)).read_bytes()
        ).hexdigest()
        for module in modules
    }
    appended = [
        {
            "name": name,
            "source": f"cached H command.{target}",
            "normalization": "clip(target / installed rating, 0, 1)",
            "installed_rating_kw": float(rating),
            "low": 0.0,
            "high": 1.0,
        }
        for name, target, rating in zip(
            HEURISTIC_FEATURE_NAMES, TARGET_NAMES, _installed_ratings(base)
        )
    ]
    return {
        "schema": INTERFACE_SCHEMA,
        "environment_id": ENVIRONMENT_ID,
        "base_interface_contract": beta_spaces.contract_payload(),
        "base_interface_contract_digest": beta_spaces.ENVIRONMENT_CONTRACT_DIGEST,
        "base_checkpoint_contract_digest": base.checkpoint_contract_digest,
        "heuristic_source_sha256": sources,
        "residual_mapping_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "heuristic_controller_id": LegacyHeuristicSetpointControllerV1.id,
        "observation_size": OBSERVATION_SIZE,
        "observation_fields": [
            *(field.to_dict() for field in beta_spaces.OBSERVATION_FIELDS),
            *appended,
        ],
        "observation_names": list(OBSERVATION_NAMES),
        "observation_dtype": "float32",
        "observation_capture": "actual vector returned by base.reset/step_process_setpoints; never re-encode",
        "action_size": ACTION_SIZE,
        "action_names": list(ACTION_NAMES),
        "action_bounds": {"low": [-1.0] * 4, "high": [1.0] * 4},
        "mapping": {
            "electrolyser_floor_kw": _floor(base),
            "floor_definition": "commitment_profile.min_stable_load_frac * installed electrolyser rated_kw",
            "electrolyser": "H_E <= F: H_E unchanged; otherwise min(active_limit_E, F + (H_E-F)*(1+0.5*r_E))",
            "other_targets": "min(active_limit_i, H_i*(1+0.5*r_i)); an H zero stays zero",
            "zero_residual": "execute cached original H command and original invalid flag, with no reconstruction, clipping or float32 target round trip",
            "nonzero_invalid": "original H invalid OR any raw residual outside [-1,1]",
            "malformed_or_nonfinite": "reject before physical movement; retain serializable failure evidence",
        },
        "cache": "one H decision at reset and each returned nonterminal boundary; no H computation on property access",
        "terminal_observation": "actual returned base135 plus four zeros; no unused terminal H decision",
        "authority": "restricted operating-effort prior, not a safety guarantee or feasibility supervisor",
        "reward_unit": "unmodified base EUR per controller interval",
        "constraint_output": "unmodified base cycle_constraint_transition",
        "registered_evaluation_eligible": False,
    }


def prepare_residual_action(
    base_env: PlantCyclicMarketEnvV2Beta,
    cached_H_decision: LegacyHeuristicSetpointDecisionV1,
    action: Any,
    *,
    interface_contract_digest: str | None = None,
) -> tuple[ProcessSetpointCommandV1, bool, dict[str, Any]]:
    """Map an action from already cached public H data without physical movement."""
    if not isinstance(cached_H_decision, LegacyHeuristicSetpointDecisionV1):
        raise TypeError("residual action requires a cached H decision")
    if interface_contract_digest is not None:
        require_sha256("interface_contract_digest", interface_contract_digest)
    cached = cached_H_decision
    if cached.command.decision_index != base_env.decisions_taken:
        raise ValueError("cached heuristic command is stale")
    raw, residual = _residual(action)
    envelope = resolve_process_setpoint_envelope(
        constraints=base_env.constraints, config=base_env.config
    )
    if envelope != cached.command.envelope:
        raise ValueError("cached heuristic command authority is stale")
    floor = _floor(base_env)
    exact_zero = bool(np.all(raw == 0.0))
    clipped = bool(np.any(raw != residual))
    h = cached.command.targets
    if exact_zero:
        targets = h
    else:
        e = h.electrolyser_target_kw
        targets = ProcessSetpointTargetsV1(
            electrolyser_target_kw=(
                e
                if e <= floor
                else min(
                    envelope.electrolyser_limit_kw,
                    floor + (e - floor) * (1.0 + 0.5 * float(residual[0])),
                )
            ),
            dac_target_kw=min(
                envelope.dac_limit_kw,
                h.dac_target_kw * (1.0 + 0.5 * float(residual[1])),
            ),
            methanator_target_kw=min(
                envelope.methanator_limit_kw,
                h.methanator_target_kw * (1.0 + 0.5 * float(residual[2])),
            ),
            battery_charge_ceiling_kw=min(
                envelope.battery_charge_limit_kw,
                h.battery_charge_ceiling_kw * (1.0 + 0.5 * float(residual[3])),
            ),
        )
    body = {
        "schema": EVIDENCE_SCHEMA,
        "interface_contract_digest": interface_contract_digest
        or contract_digest(describe_interface_contract(base_env)),
        "decision_index": int(base_env.decisions_taken),
        "heuristic_decision": cached.to_dict(),
        "raw_residual": raw.tolist(),
        "bounded_residual": residual.tolist(),
        "residual_clipped": clipped,
        "exact_zero_heuristic_command": exact_zero,
        "electrolyser_floor_kw": floor,
        "mapped_targets": targets.to_dict(),
    }
    digest = domain_sha256(EVIDENCE_SCHEMA, body)
    command = (
        cached.command
        if exact_zero
        else issue_process_setpoint_command(
            targets,
            decision_index=cached.command.decision_index,
            observation_sha256=cached.command.observation_sha256,
            hold_duration_s=cached.command.hold_duration_s,
            issuer_id=ENVIRONMENT_ID,
            source_schema=EVIDENCE_SCHEMA,
            source_sha256=digest,
            constraints=base_env.constraints,
            config=base_env.config,
        )
    )
    invalid = cached.invalid or clipped
    evidence = {
        "status": "issued",
        "body": body,
        "sha256": digest,
        "command": command.to_dict(),
        "controller_invalid": invalid,
    }
    return command, invalid, evidence


class ResearchResidualHWrapper(gym.Wrapper[Any, Any, Any, Any]):
    """Cache H once per live boundary and adjust only its operating effort."""

    def __init__(self, env: PlantCyclicMarketEnvV2Beta) -> None:
        if not isinstance(env, PlantCyclicMarketEnvV2Beta) or env.unwrapped is not env:
            raise TypeError("residual wrapper requires a plain beta environment")
        if (
            not isinstance(env.observation_space, gym.spaces.Box)
            or env.observation_space.shape != (beta_spaces.OBSERVATION_SIZE,)
            or env.observation_space.dtype != np.dtype(np.float32)
            or not np.array_equal(
                env.observation_space.low, beta_spaces.observation_low()
            )
            or not np.array_equal(
                env.observation_space.high, beta_spaces.observation_high()
            )
            or not isinstance(env.action_space, gym.spaces.Box)
            or env.action_space.shape != (beta_spaces.ACTION_SIZE,)
            or env.action_space.dtype != np.dtype(np.float32)
            or not np.array_equal(env.action_space.low, beta_spaces.action_low())
            or not np.array_equal(env.action_space.high, beta_spaces.action_high())
            or env.checkpoint_contract.get("interface_contract_digest")
            != beta_spaces.ENVIRONMENT_CONTRACT_DIGEST
        ):
            raise ValueError(
                "residual wrapper requires the complete 135-field beta interface"
            )
        profile = env.episode_config.electrolyser_commitment
        if profile is None:
            raise ValueError("residual prior requires the declared commitment profile")
        super().__init__(env)
        self.base = env
        self._ratings = np.asarray(
            [
                env.config.electrolyser.rated_kw,
                env.config.dac.rated_kw,
                env.config.methanator.rated_kw,
                env.config.battery.max_charge_kw,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(self._ratings).all() or np.any(self._ratings <= 0.0):
            raise ValueError(
                "heuristic feature normalization requires positive installed ratings"
            )
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(
                (beta_spaces.observation_low(), np.zeros(4, dtype=np.float32))
            ),
            high=np.concatenate(
                (beta_spaces.observation_high(), np.ones(4, dtype=np.float32))
            ),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self._heuristic = LegacyHeuristicSetpointControllerV1()
        self._cached: LegacyHeuristicSetpointDecisionV1 | None = None
        self._base_observation: np.ndarray[Any, np.dtype[np.float32]] | None = None
        self._observation: np.ndarray[Any, np.dtype[np.float32]] | None = None
        self._features = np.zeros(4, dtype=np.float32)
        self._last_evidence: dict[str, Any] | None = None
        self._last_timings: tuple[int, ...] = ()
        self._interface = describe_interface_contract(env)
        self._interface_digest = contract_digest(self._interface)
        checkpoint = {
            "schema": "plant-research-residual-h-checkpoint-v1",
            "environment_id": ENVIRONMENT_ID,
            "interface_contract": self.interface_contract,
            "interface_contract_digest": self.interface_contract_digest,
            "base_checkpoint_contract": env.checkpoint_contract,
            "registered_evaluation_eligible": False,
        }
        self._checkpoint = {
            **checkpoint,
            "checkpoint_contract_digest": contract_digest(checkpoint),
        }

    @property
    def interface_contract(self) -> dict[str, Any]:
        return deepcopy(self._interface)

    @property
    def interface_contract_digest(self) -> str:
        return self._interface_digest

    def require_interface_contract(self, expected: Mapping[str, Any]) -> None:
        """Refuse a 135-field checkpoint or a changed residual interface."""
        if dict(expected) != self._interface:
            raise ValueError("research residual interface contract mismatch")

    @property
    def checkpoint_contract(self) -> dict[str, Any]:
        return deepcopy(self._checkpoint)

    @property
    def checkpoint_contract_digest(self) -> str:
        return str(self._checkpoint["checkpoint_contract_digest"])

    @property
    def current_heuristic_decision(self) -> LegacyHeuristicSetpointDecisionV1 | None:
        return deepcopy(self._cached)

    @property
    def current_heuristic_features(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        if self._observation is None:
            raise RuntimeError("reset() is required before reading heuristic features")
        return self._features.copy()

    @property
    def current_base_observation(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        if self._base_observation is None:
            raise RuntimeError(
                "reset() is required before reading the base observation"
            )
        return self._base_observation.copy()

    @property
    def current_observation(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        if self._observation is None:
            raise RuntimeError(
                "reset() is required before reading the residual observation"
            )
        return self._observation.copy()

    @property
    def current_decision_index(self) -> int:
        if self._observation is None:
            raise RuntimeError("reset() is required before reading the decision index")
        return int(self.base.decisions_taken)

    @property
    def last_residual_evidence(self) -> dict[str, Any] | None:
        return deepcopy(self._last_evidence)

    @property
    def last_dispatcher_elapsed_ns(self) -> tuple[int, ...]:
        return self._last_timings

    def _cache_boundary(
        self, observation: Any, *, terminal: bool
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        # Validate the actual returned vector before advancing the H cache.
        extend_observation(observation, self.base, None)
        self._base_observation = observation.copy()
        self._cached = None
        if not terminal:
            self._cached = self._heuristic.compute_command(
                self.base.controller_observation,
                constraints=self.base.constraints,
                config=self.base.config,
                commitment_profile=self.base.episode_config.electrolyser_commitment,
            )
        self._observation = extend_observation(observation, self.base, self._cached)
        self._features = self._observation[-4:].copy()
        return self._observation.copy()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], dict[str, Any]]:
        self._cached = None
        self._observation = None
        self._base_observation = None
        self._last_evidence = None
        self._last_timings = ()
        self._heuristic.reset()
        observation, info = self.base.reset(seed=seed, options=options)
        return self._cache_boundary(observation, terminal=False), info

    def step(
        self, action: Any
    ) -> tuple[
        np.ndarray[Any, np.dtype[np.float32]], float, bool, bool, dict[str, Any]
    ]:
        cached = self._cached
        if cached is None:
            raise RuntimeError("reset() is required before a live residual step")
        if cached.command.decision_index != self.base.decisions_taken:
            raise RuntimeError("cached heuristic command is stale")
        try:
            command, invalid, evidence = prepare_residual_action(
                self.base,
                cached,
                action,
                interface_contract_digest=self.interface_contract_digest,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self._last_evidence = {
                "schema": EVIDENCE_SCHEMA,
                "status": "rejected",
                "decision_index": int(self.base.decisions_taken),
                "raw_action": _safe_input(action),
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            }
            raise
        self._last_evidence = evidence
        result = self.base.step_process_setpoints(command, controller_invalid=invalid)
        observation, reward, terminated, truncated, base_info = result
        self._last_timings = tuple(
            int(item) for item in base_info["dispatcher_elapsed_ns"]
        )
        observation139 = self._cache_boundary(
            observation, terminal=bool(terminated or truncated)
        )
        return (
            observation139,
            float(reward),
            bool(terminated),
            bool(truncated),
            base_info,
        )


__all__ = [
    "ACTION_NAMES",
    "ACTION_SIZE",
    "ENVIRONMENT_ID",
    "EVIDENCE_SCHEMA",
    "HEURISTIC_FEATURE_NAMES",
    "INTERFACE_SCHEMA",
    "OBSERVATION_NAMES",
    "OBSERVATION_SIZE",
    "ResearchResidualHWrapper",
    "TARGET_NAMES",
    "describe_interface_contract",
    "extend_observation",
    "prepare_residual_action",
]

"""Research-only operating-effort residual around the corrected product MPC.

This restricted prior is neither a feasibility supervisor nor a safety guarantee.
The wrapper consumes the base environment's returned observation exactly once;
the allocator belongs outside this wrapper so each physical transition is counted.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
from importlib.machinery import EXTENSION_SUFFIXES
import json
import math
from numbers import Real
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import gymnasium as gym
import numpy as np

from ....operational_mpc import OperationalMPCController, OperatingDecision
from .... import operational_mpc
from ....controllers import base as controller_base, heuristic
from .. import beta_spaces, setpoint_command, setpoint_controllers
from ..beta_env import PlantCyclicMarketEnvV2Beta
from ..contract import contract_digest
from ..mpc_schema import domain_sha256, require_sha256
from ..mpc_execution import public_observation_sha256
from ..setpoint_command import (
    ProcessSetpointCommandV1,
    issue_process_setpoint_command,
    resolve_process_setpoint_envelope,
)
from ..setpoint_control import ProcessSetpointTargetsV1


ENVIRONMENT_ID = "plant-research-residual-mpc-beta-v1"
INTERFACE_SCHEMA = "plant-research-residual-mpc-interface-v1"
EVIDENCE_SCHEMA = "plant-research-residual-mpc-decision-v1"
OBSERVATION_SIZE = 139
ACTION_SIZE = 4
TARGET_NAMES = (
    "electrolyser_target_kw",
    "dac_target_kw",
    "methanator_target_kw",
    "battery_charge_ceiling_kw",
)
MPC_FEATURE_NAMES = tuple(f"mpc_{name}_fraction" for name in TARGET_NAMES)
OBSERVATION_NAMES = (*beta_spaces.OBSERVATION_NAMES, *MPC_FEATURE_NAMES)
ACTION_NAMES = tuple(f"residual_{name}" for name in TARGET_NAMES)


class MPCResidualIntegrityError(RuntimeError):
    """The prior or physical/cache boundary failed; no valid continuation exists."""


def make_mpc_controller(base: PlantCyclicMarketEnvV2Beta) -> OperationalMPCController:
    """Construct the unchanged stateful prior without a reset or solve."""
    return OperationalMPCController(
        horizon_h=base.episode_config.horizon_h,
        interval_s=base.episode_config.decision_interval_s,
        assumptions=base.economic_assumptions,
        beta_task=base.beta_task,
    )


def mpc_decision_record(decision: OperatingDecision) -> dict[str, Any]:
    """Keep the original causal decision, including fallback and solver evidence."""
    return {
        "command": decision.command.to_dict(),
        "invalid": decision.invalid,
        "evidence": {
            "payload": json.loads(decision.evidence.payload_json),
            "sha256": decision.evidence.sha256,
            "source_notes": list(decision.evidence.source_notes),
            "plan": (
                None
                if decision.evidence.plan is None
                else decision.evidence.plan.to_dict()
            ),
        },
    }


def _is_native_module(module: ModuleType) -> bool:
    return any(
        str(getattr(module, "__file__", None)).endswith(suffix)
        for suffix in EXTENSION_SUFFIXES
    )


def _numpy_native_module() -> ModuleType:
    # NumPy 1.26 and 2.x expose different package names. Bind the extension,
    # never the compatibility Python shim present under the old name in 2.x.
    for name in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name is None or not (
                name == exc.name or name.startswith(f"{exc.name}.")
            ):
                raise
            continue
        if _is_native_module(module):
            return module
    raise RuntimeError("cannot bind the loaded NumPy native implementation")


def _numerical_sources() -> dict[str, Any]:
    numpy_native = _numpy_native_module()
    scipy = importlib.import_module("scipy")
    milp = importlib.import_module("scipy.optimize._milp")
    # SciPy 1.14 calls a compiled Cython wrapper; newer releases call a Python
    # wrapper holding the native HiGHS core/options modules in its globals.
    highs_wrapper = importlib.import_module(str(milp._highs_wrapper.__module__))
    highs_modules = (
        highs_wrapper,
        *(
            value
            for value in vars(highs_wrapper).values()
            if isinstance(value, ModuleType)
            and value.__name__.startswith("scipy.optimize.")
        ),
    )
    core = (
        highs_wrapper
        if _is_native_module(highs_wrapper)
        else vars(highs_wrapper).get("_h")
    )
    if not isinstance(core, ModuleType) or not _is_native_module(core):
        raise RuntimeError("cannot bind the loaded SciPy HiGHS native implementation")
    highs_native = tuple(
        module for module in highs_modules if _is_native_module(module)
    )
    parts = ("MAJOR", "MINOR", "PATCH")
    version = next(
        (
            ".".join(str(getattr(module, f"HIGHS_VERSION_{part}")) for part in parts)
            for module in highs_native
            if all(hasattr(module, f"HIGHS_VERSION_{part}") for part in parts)
        ),
        None,
    )
    modules = (
        np,
        numpy_native,
        scipy,
        milp,
        *highs_modules,
        importlib.import_module("scipy.optimize._constraints"),
        importlib.import_module("scipy.sparse._csc"),
    )
    return {
        "source_sha256": {
            module.__name__: hashlib.sha256(
                Path(str(module.__file__)).read_bytes()
            ).hexdigest()
            for module in modules
        },
        "native_implementation_modules": {
            "numpy": numpy_native.__name__,
            "scipy_highs": sorted({module.__name__ for module in highs_native}),
        },
        "numpy_version": np.__version__,
        "scipy_version": str(scipy.__version__),
        "highs_version": version,
    }


def _mpc_sources() -> dict[str, Any]:
    """Bind local planner dependencies plus the installed numerical implementation.

    Full process provenance remains the runner's responsibility. These identities
    make a changed prior/solver incompatible with a saved residual interface.
    """
    local = (
        operational_mpc,
        controller_base,
        heuristic,
        setpoint_controllers,
        setpoint_command,
        *(
            importlib.import_module(f"plant_backend.rl.v2.{name}")
            for name in (
                "beta_config",
                "beta_control",
                "beta_mpc_schema",
                "beta_mpc_uncertainty",
                "beta_spaces",
                "control",
                "boundary",
                "contract",
                "economics",
                "economic_mpc_model",
                "economic_mpc_schema",
                "economic_mpc_solver",
                "mpc_model",
                "mpc_schema",
                "mpc_forecast",
                "operating_mpc_forecast",
                "mpc_execution",
                "mpc_setpoint_execution",
                "mpc_setpoints",
                "market",
                "weather",
                "setpoint_control",
                "spaces",
            )
        ),
        importlib.import_module("plant_backend.safety.validation"),
        importlib.import_module("plant_sim.commitment"),
        importlib.import_module("plant_sim.config"),
        importlib.import_module("plant_sim.components.solar"),
        importlib.import_module("plant_sim.state"),
    )
    numerical = _numerical_sources()
    hashes = {
        module.__name__: hashlib.sha256(
            Path(str(module.__file__)).read_bytes()
        ).hexdigest()
        for module in local
    }
    return {
        **numerical,
        "source_sha256": {**hashes, **numerical["source_sha256"]},
    }


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
    mpc_decision: OperatingDecision | None,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Append cached MPC features without observing, resetting or computing MPC."""
    if (
        not isinstance(base_vector, np.ndarray)
        or base_vector.dtype != np.float32
        or base_vector.shape != (135,)
        or not np.isfinite(base_vector).all()
    ):
        raise ValueError("base must return its finite float32 135-field observation")
    features = np.zeros(4, dtype=np.float32)
    if mpc_decision is not None:
        if not isinstance(mpc_decision, OperatingDecision):
            raise TypeError("MPC decision must be a cached MPC decision or None")
        if mpc_decision.command.decision_index != base_env.decisions_taken:
            raise ValueError("MPC features belong to another decision")
        values = np.asarray(
            [getattr(mpc_decision.command.targets, name) for name in TARGET_NAMES],
            dtype=np.float64,
        )
        features = np.clip(values / _installed_ratings(base_env), 0.0, 1.0).astype(
            np.float32
        )
    return np.concatenate((base_vector, features))


def describe_interface_contract(base: PlantCyclicMarketEnvV2Beta) -> dict[str, Any]:
    appended = [
        {
            "name": name,
            "source": f"cached MPC command.{target}",
            "normalization": "clip(target / installed rating, 0, 1)",
            "installed_rating_kw": float(rating),
            "low": 0.0,
            "high": 1.0,
        }
        for name, target, rating in zip(
            MPC_FEATURE_NAMES, TARGET_NAMES, _installed_ratings(base)
        )
    ]
    return {
        "schema": INTERFACE_SCHEMA,
        "environment_id": ENVIRONMENT_ID,
        "base_interface_contract": beta_spaces.contract_payload(),
        "base_interface_contract_digest": beta_spaces.ENVIRONMENT_CONTRACT_DIGEST,
        "base_checkpoint_contract_digest": base.checkpoint_contract_digest,
        "mpc_dependencies": _mpc_sources(),
        "residual_mapping_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "mpc_controller": make_mpc_controller(base).to_dict(),
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
            "electrolyser": "MPC_E <= F: MPC_E unchanged; otherwise min(active_limit_E, F + (MPC_E-F)*(1+0.25*r_E))",
            "other_targets": "min(active_limit_i, MPC_i*(1+0.25*r_i)); an MPC zero stays zero",
            "zero_residual": "execute cached original MPC command and original invalid flag, with no reconstruction, clipping or float32 target round trip",
            "nonzero_invalid": "original MPC invalid OR any raw residual outside [-1,1]",
            "malformed_or_nonfinite": "reject before physical movement; retain serializable failure evidence",
        },
        "cache": "one MPC decision at reset and each returned nonterminal boundary; no MPC computation on property access",
        "terminal_observation": "actual returned base135 plus four zeros; no unused terminal MPC decision",
        "authority": "restricted operating-effort prior, not a safety guarantee or feasibility supervisor",
        "reward_unit": "unmodified base EUR per controller interval",
        "constraint_output": "unmodified base cycle_constraint_transition",
        "registered_evaluation_eligible": False,
    }


def _validate_cached_mpc(
    base_env: PlantCyclicMarketEnvV2Beta, cached: OperatingDecision
) -> None:
    if not isinstance(cached, OperatingDecision):
        raise TypeError("residual action requires a cached MPC decision")
    if cached.command.decision_index != base_env.decisions_taken:
        raise ValueError("cached MPC command is stale")
    envelope = resolve_process_setpoint_envelope(
        constraints=base_env.constraints, config=base_env.config
    )
    if envelope != cached.command.envelope:
        raise ValueError("cached MPC command authority is stale")
    if (
        not isinstance(cached.invalid, bool)
        or cached.command.issuer_id != OperationalMPCController.id
        or cached.command.source_schema != operational_mpc.DECISION_SCHEMA
        or cached.command.source_sha256 != cached.evidence.sha256
        or cached.command.observation_sha256
        != public_observation_sha256(base_env.controller_observation)
        or json.loads(cached.evidence.payload_json)["observation_sha256"]
        != base_env.cyclic_market_controller_observation.sha256
    ):
        raise ValueError("cached MPC evidence or public boundary differs")


def prepare_residual_action(
    base_env: PlantCyclicMarketEnvV2Beta,
    cached_MPC_decision: OperatingDecision,
    action: Any,
    *,
    interface_contract_digest: str | None = None,
) -> tuple[ProcessSetpointCommandV1, bool, dict[str, Any]]:
    """Map an action from already cached public MPC data without physical movement."""
    cached = cached_MPC_decision
    _validate_cached_mpc(base_env, cached)
    if interface_contract_digest is not None:
        require_sha256("interface_contract_digest", interface_contract_digest)
    raw, residual = _residual(action)
    envelope = resolve_process_setpoint_envelope(
        constraints=base_env.constraints, config=base_env.config
    )
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
                    floor + (e - floor) * (1.0 + 0.25 * float(residual[0])),
                )
            ),
            dac_target_kw=min(
                envelope.dac_limit_kw,
                h.dac_target_kw * (1.0 + 0.25 * float(residual[1])),
            ),
            methanator_target_kw=min(
                envelope.methanator_limit_kw,
                h.methanator_target_kw * (1.0 + 0.25 * float(residual[2])),
            ),
            battery_charge_ceiling_kw=min(
                envelope.battery_charge_limit_kw,
                h.battery_charge_ceiling_kw * (1.0 + 0.25 * float(residual[3])),
            ),
        )
    body = {
        "schema": EVIDENCE_SCHEMA,
        "interface_contract_digest": interface_contract_digest
        or contract_digest(describe_interface_contract(base_env)),
        "decision_index": int(base_env.decisions_taken),
        "mpc_decision": mpc_decision_record(cached),
        "raw_residual": raw.tolist(),
        "bounded_residual": residual.tolist(),
        "residual_clipped": clipped,
        "exact_zero_mpc_command": exact_zero,
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


class ResearchMPCResidualWrapper(gym.Wrapper[Any, Any, Any, Any]):
    """Cache MPC once per live boundary and adjust only its operating effort."""

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
                "MPC feature normalization requires positive installed ratings"
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
        self._mpc = make_mpc_controller(env)
        self._cached: OperatingDecision | None = None
        self._base_observation: np.ndarray[Any, np.dtype[np.float32]] | None = None
        self._observation: np.ndarray[Any, np.dtype[np.float32]] | None = None
        self._features = np.zeros(4, dtype=np.float32)
        self._last_evidence: dict[str, Any] | None = None
        self._last_mpc_failure: dict[str, Any] | None = None
        self._last_timings: tuple[int, ...] = ()
        self._interface = describe_interface_contract(env)
        self._interface_digest = contract_digest(self._interface)
        checkpoint = {
            "schema": "plant-research-residual-mpc-checkpoint-v1",
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
        if contract_digest(dict(expected)) != self._interface_digest:
            raise ValueError("research residual interface contract mismatch")

    @property
    def checkpoint_contract(self) -> dict[str, Any]:
        return deepcopy(self._checkpoint)

    @property
    def checkpoint_contract_digest(self) -> str:
        return str(self._checkpoint["checkpoint_contract_digest"])

    @property
    def current_mpc_decision(self) -> OperatingDecision | None:
        return deepcopy(self._cached)

    @property
    def current_mpc_features(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        if self._observation is None:
            raise RuntimeError("reset() is required before reading MPC features")
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
    def last_mpc_failure(self) -> dict[str, Any] | None:
        """Fatal prior errors remain visible; only the MPC owns typed fallback."""
        return deepcopy(self._last_mpc_failure)

    @property
    def last_dispatcher_elapsed_ns(self) -> tuple[int, ...]:
        return self._last_timings

    def _cache_boundary(
        self, observation: Any, *, terminal: bool
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        # Validate the actual returned vector before advancing the MPC cache.
        extend_observation(observation, self.base, None)
        self._base_observation = observation.copy()
        self._cached = None
        self._observation = None
        self._features = np.zeros(4, dtype=np.float32)
        if not terminal:
            try:
                self._cached = self._mpc.compute_command(
                    self.base.cyclic_market_controller_observation,
                    constraints=self.base.constraints,
                    config=self.base.config,
                    commitment_profile=self.base.episode_config.electrolyser_commitment,
                )
                _validate_cached_mpc(self.base, self._cached)
            except Exception as exc:
                self._last_mpc_failure = {
                    "decision_index": int(self.base.decisions_taken),
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                raise MPCResidualIntegrityError(f"MPC prior failed: {exc}") from exc
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
        self._last_mpc_failure = None
        self._last_timings = ()
        try:
            self._mpc.reset()
            observation, info = self.base.reset(seed=seed, options=options)
            return self._cache_boundary(observation, terminal=False), info
        except Exception as exc:
            self._cached = None
            self._observation = None
            if isinstance(exc, MPCResidualIntegrityError):
                raise
            raise MPCResidualIntegrityError(
                f"MPC residual reset failed: {exc}"
            ) from exc

    def step(
        self, action: Any
    ) -> tuple[
        np.ndarray[Any, np.dtype[np.float32]], float, bool, bool, dict[str, Any]
    ]:
        cached = self._cached
        if cached is None:
            raise MPCResidualIntegrityError(
                "reset() is required before a live residual step"
            )
        try:
            _validate_cached_mpc(self.base, cached)
        except Exception as exc:
            self._cached = None
            self._observation = None
            self._last_evidence = {
                "schema": EVIDENCE_SCHEMA,
                "status": "rejected",
                "decision_index": int(self.base.decisions_taken),
                "raw_action": _safe_input(action),
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            }
            raise MPCResidualIntegrityError(
                f"MPC residual cached boundary failed: {exc}"
            ) from exc
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
        try:
            result = self.base.step_process_setpoints(
                command, controller_invalid=invalid
            )
            observation, reward, terminated, truncated, base_info = result
            self._last_timings = tuple(
                int(item) for item in base_info["dispatcher_elapsed_ns"]
            )
            observation139 = self._cache_boundary(
                observation, terminal=bool(terminated or truncated)
            )
        except Exception as exc:
            self._cached = None
            self._observation = None
            if isinstance(exc, MPCResidualIntegrityError):
                raise
            raise MPCResidualIntegrityError(
                f"MPC residual execution failed: {exc}"
            ) from exc
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
    "MPC_FEATURE_NAMES",
    "MPCResidualIntegrityError",
    "INTERFACE_SCHEMA",
    "OBSERVATION_NAMES",
    "OBSERVATION_SIZE",
    "ResearchMPCResidualWrapper",
    "TARGET_NAMES",
    "describe_interface_contract",
    "extend_observation",
    "prepare_residual_action",
    "make_mpc_controller",
    "mpc_decision_record",
]

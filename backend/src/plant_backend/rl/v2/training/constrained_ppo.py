"""Auditable four-channel PPO-Lagrangian training for the beta environment.

The module deliberately keeps its mathematical and provenance primitives usable
without the optional ``train`` dependency group.  Torch and Stable-Baselines3
are imported opportunistically; only model construction and loading require
them.  This lets ordinary API, simulator, and verifier processes inspect a
checkpoint's constrained-training contract without importing a trainer.

The implementation follows ``plant-beta-constrained-rl-protocol-v1``:

* reward and four resource-cost GAE streams remain separate;
* each minibatch normalizes reward and each cost advantage independently;
* PI/I multipliers update once per complete rollout from completed episodes;
* a separate two-layer, four-head cost critic is initialized in an isolated RNG
  scope after the unchanged SB3 actor and reward critic; and
* scalar PPO records the same evidence while cost data cannot affect gradients.
"""

from __future__ import annotations

import base64
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass, replace as dataclass_replace
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import random
import time
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, cast

import numpy as np
import numpy.typing as npt

from ..beta_constraint_transition import (
    CYCLE_CONSTRAINT_RESOURCE_ORDER,
    CYCLE_CONSTRAINT_TRANSITION_SCHEMA,
)
from ..beta_config import canonical_sha256


PROTOCOL_ID = "plant-beta-constrained-rl-protocol-v1"
PI_LAGRANGIAN_ALGORITHM_ID = "plant-beta-ppo-pi-lagrangian-v1"
I_LAGRANGIAN_ALGORITHM_ID = "plant-beta-ppo-i-lagrangian-v1"
SCALAR_PPO_ALGORITHM_ID = "plant-beta-sb3-ppo-scalar-v1"
SUPPORTED_ALGORITHM_IDS = (
    PI_LAGRANGIAN_ALGORITHM_ID,
    I_LAGRANGIAN_ALGORITHM_ID,
    SCALAR_PPO_ALGORITHM_ID,
)
CONSTRAINED_ALGORITHM_IDS = (
    PI_LAGRANGIAN_ALGORITHM_ID,
    I_LAGRANGIAN_ALGORITHM_ID,
)
RESOURCE_ORDER = CYCLE_CONSTRAINT_RESOURCE_ORDER
COST_CHANNEL_COUNT = len(RESOURCE_ORDER)
TRANSITION_INFO_KEY = "cycle_constraint_transition"

CONSTRAINED_SETTINGS_SCHEMA = "constrained-ppo-settings-v1"
MULTIPLIER_STATE_SCHEMA = "pi-lagrange-state-v1"
EPISODE_COST_ACCUMULATOR_SCHEMA = "episode-cost-accumulator-v1"
EPISODE_COST_SUMMARY_SCHEMA = "episode-cost-summary-v1"
ROLLOUT_LEDGER_SCHEMA = "constrained-ppo-rollout-ledger-v1"
CONSTRAINT_STATE_SCHEMA = "constrained-ppo-state-v1"
CONSTRAINT_STATE_ENVELOPE_SCHEMA = "constrained-ppo-state-envelope-v1"
EXTERNAL_EPISODE_STATE_PIN_SCHEMA = "external-episode-state-pin-v1"
ABORTED_ROLLOUT_SCHEMA = "constrained-ppo-aborted-rollout-v1"
PPO_CONFIGURATION_SCHEMA = "constrained-ppo-configuration-v1"
POLICY_ARCHITECTURE_SCHEMA = "constrained-ppo-policy-architecture-v1"
LEARNER_RNG_STATE_SCHEMA = "constrained-ppo-learner-rng-state-v1"
RESUME_CONTINUATION_RECEIPT_SCHEMA = "constrained-ppo-resume-continuation-receipt-v1"
CHECKPOINT_OBSERVATION_DIGEST_SCHEMA = "constrained-ppo-checkpoint-observation-v1"

REGISTERED_OPTIMIZER_SEEDS = (101, 211, 307, 401, 503)
COST_NETWORK_SEED_OFFSET = 10_000_000
BETA_EPISODE_CYCLE_COUNT = 7
BETA_DECISIONS_PER_CYCLE = 96
BETA_DECISIONS_PER_EPISODE = BETA_EPISODE_CYCLE_COUNT * BETA_DECISIONS_PER_CYCLE
_CONSTRAINT_STATE_PAYLOAD_FIELDS = frozenset(
    {
        "schema",
        "protocol_id",
        "algorithm_id",
        "settings",
        "ppo_configuration",
        "policy_parameter_sha256",
        "optimizer_state_sha256",
        "learner_rng_state",
        "multiplier_state",
        "episode_cost_accumulator",
        "rollout_ledgers",
        "rollout_ledger_sha256",
        "num_timesteps",
        "ppo_epoch_update_count",
        "external_episode_state_pin",
        "checkpoint_observation_sha256",
        "aborted_rollout",
        "unledgered_transition_count",
        "pending_rollout",
        "resume_continuation_receipt",
    }
)

FloatVector4 = tuple[float, float, float, float]
BoolVector4 = tuple[bool, bool, bool, bool]


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return 0.0 if number == 0.0 else number


def _non_negative_float(name: str, value: object) -> float:
    number = _finite_float(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _float_vector4(
    name: str,
    values: object,
    *,
    non_negative: bool = False,
) -> FloatVector4:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a four-value sequence")
    if len(values) != COST_CHANNEL_COUNT:
        raise ValueError(f"{name} must contain four values")
    parser = _non_negative_float if non_negative else _finite_float
    normalized = tuple(
        parser(f"{name}[{index}]", value) for index, value in enumerate(values)
    )
    return cast(FloatVector4, normalized)


def _bool_vector4(name: str, values: object) -> BoolVector4:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a four-value sequence")
    if len(values) != COST_CHANNEL_COUNT or any(
        not isinstance(value, bool) for value in values
    ):
        raise TypeError(f"{name} must contain four booleans")
    return cast(BoolVector4, tuple(values))


def _non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _canonical_digest(domain: str, body: Mapping[str, Any]) -> str:
    encoded = _canonical_json(body).encode("utf-8")
    return hashlib.sha256(domain.encode("ascii") + b"\x00" + encoded).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_object_copy(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed object")
    try:
        encoded = _canonical_json(value)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as reason:
        raise ValueError(f"{name} must be finite canonical JSON: {reason}") from reason
    if not isinstance(decoded, dict):  # pragma: no cover - mapping guarantees object
        raise TypeError(f"{name} did not encode as an object")
    return cast(dict[str, Any], decoded)


def _sha256_hex(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex string")
    try:
        bytes.fromhex(value)
    except ValueError as reason:
        raise ValueError(f"{name} must be a SHA-256 hex string") from reason
    return value


def _positive_int(name: str, value: object) -> int:
    result = _non_negative_int(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _strict_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


_ATTESTATION_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class ValidatedConstraintStateAttestationV1:
    """Opaque proof minted only after a policy artifact validates on load.

    This capability is deliberately non-serializable.  Its guarded constructor
    prevents an ordinary caller from turning a copied metadata mapping into
    evidence that a ``ConstrainedPPO`` artifact was actually deserialized and
    checked against its live policy, optimizer, configuration, and ledgers.
    """

    artifact_sha256: str
    algorithm_id: str
    num_timesteps: int
    constraint_state_sha256: str
    checkpoint_observation_sha256: str
    _constraint_state_json: str

    def __init__(
        self,
        *,
        artifact_sha256: object,
        algorithm_id: object,
        num_timesteps: object,
        constraint_state: object,
        checkpoint_observation_sha256: object,
        _issuer: object | None = None,
    ) -> None:
        if _issuer is not _ATTESTATION_ISSUER:
            raise RuntimeError(
                "constraint-state attestations are issued only by "
                "ConstrainedPPO.load"
            )
        artifact_digest = _sha256_hex("policy artifact digest", artifact_sha256)
        if artifact_digest != artifact_digest.lower():
            raise ValueError("policy artifact digest must be lowercase")
        if not isinstance(algorithm_id, str) or algorithm_id not in (
            SUPPORTED_ALGORITHM_IDS
        ):
            raise ValueError("attested algorithm is unsupported")
        timestep_cut = _non_negative_int("attested num_timesteps", num_timesteps)
        envelope = _canonical_object_copy("attested constraint state", constraint_state)
        if set(envelope) != {"schema", "payload", "sha256"}:
            raise ValueError("attested constraint-state envelope fields mismatch")
        if envelope["schema"] != CONSTRAINT_STATE_ENVELOPE_SCHEMA:
            raise ValueError("attested constraint-state envelope schema mismatch")
        payload = _canonical_object_copy(
            "attested constraint-state payload", envelope["payload"]
        )
        digest = _sha256_hex("attested constraint-state digest", envelope["sha256"])
        if digest != _canonical_digest(CONSTRAINT_STATE_ENVELOPE_SCHEMA, payload):
            raise ValueError("attested constraint-state digest mismatch")
        if payload.get("algorithm_id") != algorithm_id:
            raise ValueError("attested constraint-state algorithm mismatch")
        if payload.get("num_timesteps") != timestep_cut:
            raise ValueError("attested constraint-state timestep mismatch")
        observation_digest = _sha256_hex(
            "checkpoint observation digest", checkpoint_observation_sha256
        )
        object.__setattr__(self, "artifact_sha256", artifact_digest)
        object.__setattr__(self, "algorithm_id", algorithm_id)
        object.__setattr__(self, "num_timesteps", timestep_cut)
        object.__setattr__(self, "constraint_state_sha256", digest)
        object.__setattr__(self, "checkpoint_observation_sha256", observation_digest)
        object.__setattr__(self, "_constraint_state_json", _canonical_json(envelope))

    @property
    def constraint_state(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self._constraint_state_json))


def _policy_artifact_sha256(path: object) -> str | None:
    """Hash the exact filesystem artifact supplied to ``load`` when possible."""

    if not isinstance(path, (str, Path)):
        return None
    candidate = Path(path)
    if not candidate.is_file():
        appended = Path(f"{candidate}.zip")
        if not appended.is_file():
            return None
        candidate = appended
    digest = hashlib.sha256()
    with candidate.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_observation_sha256(value: npt.ArrayLike) -> str:
    observation = np.asarray(value)
    if not np.issubdtype(observation.dtype, np.number) or not np.all(
        np.isfinite(observation)
    ):
        raise ValueError("checkpoint observation must contain finite numeric values")
    contiguous = np.ascontiguousarray(observation)
    digest = hashlib.sha256(CHECKPOINT_OBSERVATION_DIGEST_SCHEMA.encode("ascii"))
    digest.update(b"\x00")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\x00")
    digest.update(_canonical_json(list(contiguous.shape)).encode("ascii"))
    digest.update(b"\x00")
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ResumeContinuationReceiptV1:
    """Artifact-bound proof of the exact state at a resume boundary."""

    parent_artifact_sha256: str
    parent_constraint_state_sha256: str
    parent_policy_parameter_sha256: str
    parent_optimizer_state_sha256: str
    parent_learner_rng_state_sha256: str
    parent_multiplier_state_sha256: str
    parent_episode_cost_accumulator_sha256: str
    parent_external_episode_state_pin_sha256: str
    parent_checkpoint_observation_sha256: str
    parent_num_timesteps: int
    parent_rollout_ledger_sha256: str | None
    first_resumed_rollout_index: int
    first_resumed_rollout_sha256: str | None = None

    schema: ClassVar[str] = RESUME_CONTINUATION_RECEIPT_SCHEMA

    @classmethod
    def from_parent(
        cls,
        *,
        artifact_sha256: str,
        constraint_state: Mapping[str, Any],
        checkpoint_observation_sha256: str,
    ) -> ResumeContinuationReceiptV1:
        payload = _canonical_object_copy(
            "resume parent constraint payload", constraint_state["payload"]
        )
        external_pin = _canonical_object_copy(
            "resume parent external episode pin",
            payload["external_episode_state_pin"],
        )
        ledgers = payload["rollout_ledgers"]
        if not isinstance(ledgers, list):
            raise TypeError("resume parent rollout ledgers must be an array")
        if payload.get("checkpoint_observation_sha256") != (
            checkpoint_observation_sha256
        ):
            raise ValueError(
                "resume parent checkpoint observation differs from loaded artifact"
            )
        return cls(
            parent_artifact_sha256=artifact_sha256,
            parent_constraint_state_sha256=cast(str, constraint_state["sha256"]),
            parent_policy_parameter_sha256=cast(
                str, payload["policy_parameter_sha256"]
            ),
            parent_optimizer_state_sha256=cast(str, payload["optimizer_state_sha256"]),
            parent_learner_rng_state_sha256=_canonical_digest(
                "resume-parent-learner-rng-state-v1",
                _canonical_object_copy(
                    "resume parent learner RNG", payload["learner_rng_state"]
                ),
            ),
            parent_multiplier_state_sha256=_canonical_digest(
                "resume-parent-multiplier-state-v1",
                _canonical_object_copy(
                    "resume parent multiplier", payload["multiplier_state"]
                ),
            ),
            parent_episode_cost_accumulator_sha256=_canonical_digest(
                "resume-parent-episode-cost-accumulator-v1",
                _canonical_object_copy(
                    "resume parent accumulator", payload["episode_cost_accumulator"]
                ),
            ),
            parent_external_episode_state_pin_sha256=cast(str, external_pin["sha256"]),
            parent_checkpoint_observation_sha256=checkpoint_observation_sha256,
            parent_num_timesteps=cast(int, payload["num_timesteps"]),
            parent_rollout_ledger_sha256=cast(
                str | None, payload["rollout_ledger_sha256"]
            ),
            first_resumed_rollout_index=len(ledgers) + 1,
        )

    def __post_init__(self) -> None:
        for name in (
            "parent_artifact_sha256",
            "parent_constraint_state_sha256",
            "parent_policy_parameter_sha256",
            "parent_optimizer_state_sha256",
            "parent_learner_rng_state_sha256",
            "parent_multiplier_state_sha256",
            "parent_episode_cost_accumulator_sha256",
            "parent_external_episode_state_pin_sha256",
            "parent_checkpoint_observation_sha256",
        ):
            _sha256_hex(name, getattr(self, name))
        _non_negative_int("parent_num_timesteps", self.parent_num_timesteps)
        _positive_int("first_resumed_rollout_index", self.first_resumed_rollout_index)
        if self.parent_rollout_ledger_sha256 is not None:
            _sha256_hex(
                "parent_rollout_ledger_sha256", self.parent_rollout_ledger_sha256
            )
        if self.first_resumed_rollout_sha256 is not None:
            _sha256_hex(
                "first_resumed_rollout_sha256", self.first_resumed_rollout_sha256
            )

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "parent_artifact_sha256": self.parent_artifact_sha256,
            "parent_constraint_state_sha256": self.parent_constraint_state_sha256,
            "parent_policy_parameter_sha256": self.parent_policy_parameter_sha256,
            "parent_optimizer_state_sha256": self.parent_optimizer_state_sha256,
            "parent_learner_rng_state_sha256": self.parent_learner_rng_state_sha256,
            "parent_multiplier_state_sha256": self.parent_multiplier_state_sha256,
            "parent_episode_cost_accumulator_sha256": (
                self.parent_episode_cost_accumulator_sha256
            ),
            "parent_external_episode_state_pin_sha256": (
                self.parent_external_episode_state_pin_sha256
            ),
            "parent_checkpoint_observation_sha256": (
                self.parent_checkpoint_observation_sha256
            ),
            "parent_num_timesteps": self.parent_num_timesteps,
            "parent_rollout_ledger_sha256": self.parent_rollout_ledger_sha256,
            "first_resumed_rollout_index": self.first_resumed_rollout_index,
            "first_resumed_rollout_sha256": self.first_resumed_rollout_sha256,
        }

    @property
    def sha256(self) -> str:
        return _canonical_digest(self.schema, self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: object) -> ResumeContinuationReceiptV1:
        raw = _canonical_object_copy("resume continuation receipt", value)
        body = dict(raw)
        supplied = _sha256_hex("resume continuation receipt digest", body.pop("sha256"))
        expected = {
            "schema",
            "parent_artifact_sha256",
            "parent_constraint_state_sha256",
            "parent_policy_parameter_sha256",
            "parent_optimizer_state_sha256",
            "parent_learner_rng_state_sha256",
            "parent_multiplier_state_sha256",
            "parent_episode_cost_accumulator_sha256",
            "parent_external_episode_state_pin_sha256",
            "parent_checkpoint_observation_sha256",
            "parent_num_timesteps",
            "parent_rollout_ledger_sha256",
            "first_resumed_rollout_index",
            "first_resumed_rollout_sha256",
        }
        if set(body) != expected or body["schema"] != cls.schema:
            raise ValueError("resume continuation receipt fields/schema mismatch")
        kwargs = dict(body)
        kwargs.pop("schema")
        receipt = cls(**kwargs)
        if supplied != receipt.sha256:
            raise ValueError("resume continuation receipt digest mismatch")
        return receipt

    def bind_first_rollout(self, rollout_sha256: str) -> ResumeContinuationReceiptV1:
        if self.first_resumed_rollout_sha256 is not None:
            raise RuntimeError("resume continuation receipt is already finalized")
        return dataclass_replace(self, first_resumed_rollout_sha256=rollout_sha256)


@dataclass(frozen=True, slots=True)
class ConstrainedPPOSettingsV1:
    """Immutable, serializable optimization contract for one learner."""

    algorithm_id: str
    optimizer_seed: int
    resource_order: tuple[str, str, str, str] = RESOURCE_ORDER
    cost_limits: FloatVector4 = (0.0, 0.0, 0.0, 0.0)
    cost_gamma: float = 1.0
    cost_gae_lambda: float = 0.95
    cost_value_loss_coefficient: float = 0.5
    cost_hidden_layers: tuple[int, int] = (64, 64)
    initial_integral: FloatVector4 = (1.0, 1.0, 1.0, 1.0)
    proportional_gain: float = 0.25
    integral_gain: float = 0.01
    proportional_ema_alpha: float = 0.95
    multiplier_min: float = 0.0
    multiplier_max: float = 100.0
    transition_info_key: str = TRANSITION_INFO_KEY

    schema: ClassVar[str] = CONSTRAINED_SETTINGS_SCHEMA

    @classmethod
    def for_algorithm(
        cls,
        algorithm_id: str,
        *,
        optimizer_seed: int,
    ) -> "ConstrainedPPOSettingsV1":
        """Construct the exact frozen settings for one registered family."""

        if algorithm_id == PI_LAGRANGIAN_ALGORITHM_ID:
            result = cls(
                algorithm_id=algorithm_id,
                optimizer_seed=optimizer_seed,
            )
        elif algorithm_id == I_LAGRANGIAN_ALGORITHM_ID:
            result = cls(
                algorithm_id=algorithm_id,
                optimizer_seed=optimizer_seed,
                proportional_gain=0.0,
            )
        elif algorithm_id == SCALAR_PPO_ALGORITHM_ID:
            result = cls(
                algorithm_id=algorithm_id,
                optimizer_seed=optimizer_seed,
                initial_integral=(0.0, 0.0, 0.0, 0.0),
                proportional_gain=0.0,
                integral_gain=0.0,
            )
        else:
            raise ValueError(f"unsupported constrained-RL algorithm {algorithm_id!r}")
        return result.validate()

    @property
    def uses_constraints(self) -> bool:
        return self.algorithm_id in CONSTRAINED_ALGORITHM_IDS

    @property
    def cost_network_seed(self) -> int:
        return COST_NETWORK_SEED_OFFSET + self.optimizer_seed

    def validate(self) -> "ConstrainedPPOSettingsV1":
        if self.algorithm_id not in SUPPORTED_ALGORITHM_IDS:
            raise ValueError("algorithm_id is not a frozen constrained-RL family")
        optimizer_seed = _non_negative_int("optimizer_seed", self.optimizer_seed)
        if optimizer_seed != self.optimizer_seed:
            raise ValueError("optimizer_seed lost canonical integer form")
        if tuple(self.resource_order) != RESOURCE_ORDER:
            raise ValueError("resource_order must be battery, h2, co2, ch4")
        if _float_vector4("cost_limits", self.cost_limits, non_negative=True) != (
            0.0,
            0.0,
            0.0,
            0.0,
        ):
            raise ValueError("the frozen cost limits are exactly zero")
        if _finite_float("cost_gamma", self.cost_gamma) != 1.0:
            raise ValueError("the frozen cost_gamma is 1.0")
        if _finite_float("cost_gae_lambda", self.cost_gae_lambda) != 0.95:
            raise ValueError("the frozen cost_gae_lambda is 0.95")
        if (
            _finite_float(
                "cost_value_loss_coefficient",
                self.cost_value_loss_coefficient,
            )
            != 0.5
        ):
            raise ValueError("the frozen cost-value loss coefficient is 0.5")
        if tuple(self.cost_hidden_layers) != (64, 64):
            raise ValueError("the frozen cost critic architecture is [64, 64]")
        for index, width in enumerate(self.cost_hidden_layers):
            if isinstance(width, bool) or not isinstance(width, Integral):
                raise TypeError(f"cost_hidden_layers[{index}] must be an integer")
        initial = _float_vector4(
            "initial_integral", self.initial_integral, non_negative=True
        )
        proportional_gain = _non_negative_float(
            "proportional_gain", self.proportional_gain
        )
        integral_gain = _non_negative_float("integral_gain", self.integral_gain)
        if self.algorithm_id == PI_LAGRANGIAN_ALGORITHM_ID:
            if initial != (1.0, 1.0, 1.0, 1.0):
                raise ValueError("PI initial integral must be four ones")
            if proportional_gain != 0.25 or integral_gain != 0.01:
                raise ValueError("PI gains must be K_P=0.25 and K_I=0.01")
        elif self.algorithm_id == I_LAGRANGIAN_ALGORITHM_ID:
            if initial != (1.0, 1.0, 1.0, 1.0):
                raise ValueError("integral-only initial integral must be four ones")
            if proportional_gain != 0.0 or integral_gain != 0.01:
                raise ValueError("integral-only gains must be K_P=0 and K_I=0.01")
        else:
            if initial != (0.0, 0.0, 0.0, 0.0):
                raise ValueError("scalar PPO has no active multiplier state")
            if proportional_gain != 0.0 or integral_gain != 0.0:
                raise ValueError("scalar PPO must not update multipliers")
        if _finite_float("proportional_ema_alpha", self.proportional_ema_alpha) != 0.95:
            raise ValueError("the frozen proportional EMA alpha is 0.95")
        if _finite_float("multiplier_min", self.multiplier_min) != 0.0:
            raise ValueError("the frozen multiplier minimum is zero")
        if _finite_float("multiplier_max", self.multiplier_max) != 100.0:
            raise ValueError("the frozen multiplier maximum is 100")
        if (
            not isinstance(self.transition_info_key, str)
            or self.transition_info_key != TRANSITION_INFO_KEY
        ):
            raise ValueError(
                f"transition_info_key must be exactly {TRANSITION_INFO_KEY!r}"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "protocol_id": PROTOCOL_ID,
            "algorithm_id": self.algorithm_id,
            "optimizer_seed": self.optimizer_seed,
            "cost_network_seed": self.cost_network_seed,
            "resource_order": list(self.resource_order),
            "cost_limits": list(self.cost_limits),
            "cost_gamma": self.cost_gamma,
            "cost_gae_lambda": self.cost_gae_lambda,
            "cost_value_loss_coefficient": self.cost_value_loss_coefficient,
            "cost_hidden_layers": list(self.cost_hidden_layers),
            "initial_integral": list(self.initial_integral),
            "proportional_gain": self.proportional_gain,
            "integral_gain": self.integral_gain,
            "derivative_gain": 0.0,
            "proportional_ema_alpha": self.proportional_ema_alpha,
            "multiplier_min": self.multiplier_min,
            "multiplier_max": self.multiplier_max,
            "transition_info_key": self.transition_info_key,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ConstrainedPPOSettingsV1":
        if not isinstance(raw, Mapping):
            raise TypeError("constrained PPO settings must be an object")
        expected = {
            "schema",
            "protocol_id",
            "algorithm_id",
            "optimizer_seed",
            "cost_network_seed",
            "resource_order",
            "cost_limits",
            "cost_gamma",
            "cost_gae_lambda",
            "cost_value_loss_coefficient",
            "cost_hidden_layers",
            "initial_integral",
            "proportional_gain",
            "integral_gain",
            "derivative_gain",
            "proportional_ema_alpha",
            "multiplier_min",
            "multiplier_max",
            "transition_info_key",
        }
        unknown = set(raw) - expected
        missing = expected - set(raw)
        if unknown or missing:
            detail = []
            if unknown:
                detail.append(f"unknown: {', '.join(sorted(unknown))}")
            if missing:
                detail.append(f"missing: {', '.join(sorted(missing))}")
            raise ValueError(
                "invalid constrained PPO settings (" + "; ".join(detail) + ")"
            )
        if raw["schema"] != cls.schema or raw["protocol_id"] != PROTOCOL_ID:
            raise ValueError("constrained PPO settings schema/protocol mismatch")
        if _finite_float("derivative_gain", raw["derivative_gain"]) != 0.0:
            raise ValueError("derivative_gain must be zero")
        result = cls(
            algorithm_id=str(raw["algorithm_id"]),
            optimizer_seed=_non_negative_int("optimizer_seed", raw["optimizer_seed"]),
            resource_order=cast(
                tuple[str, str, str, str], tuple(raw["resource_order"])
            ),
            cost_limits=_float_vector4(
                "cost_limits", raw["cost_limits"], non_negative=True
            ),
            cost_gamma=_finite_float("cost_gamma", raw["cost_gamma"]),
            cost_gae_lambda=_finite_float("cost_gae_lambda", raw["cost_gae_lambda"]),
            cost_value_loss_coefficient=_finite_float(
                "cost_value_loss_coefficient",
                raw["cost_value_loss_coefficient"],
            ),
            cost_hidden_layers=cast(tuple[int, int], tuple(raw["cost_hidden_layers"])),
            initial_integral=_float_vector4(
                "initial_integral",
                raw["initial_integral"],
                non_negative=True,
            ),
            proportional_gain=_finite_float(
                "proportional_gain", raw["proportional_gain"]
            ),
            integral_gain=_finite_float("integral_gain", raw["integral_gain"]),
            proportional_ema_alpha=_finite_float(
                "proportional_ema_alpha", raw["proportional_ema_alpha"]
            ),
            multiplier_min=_finite_float("multiplier_min", raw["multiplier_min"]),
            multiplier_max=_finite_float("multiplier_max", raw["multiplier_max"]),
            transition_info_key=str(raw["transition_info_key"]),
        ).validate()
        if raw["cost_network_seed"] != result.cost_network_seed:
            raise ValueError("cost_network_seed is not derived from optimizer_seed")
        return result


@dataclass(frozen=True, slots=True)
class PILagrangeStateV1:
    """Four independent PI controllers after a complete rollout."""

    proportional: FloatVector4
    integral: FloatVector4
    multipliers: FloatVector4
    completed_update_count: int

    schema: ClassVar[str] = MULTIPLIER_STATE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "proportional",
            _float_vector4("proportional", self.proportional),
        )
        object.__setattr__(
            self,
            "integral",
            _float_vector4("integral", self.integral, non_negative=True),
        )
        object.__setattr__(
            self,
            "multipliers",
            _float_vector4("multipliers", self.multipliers, non_negative=True),
        )
        object.__setattr__(
            self,
            "completed_update_count",
            _non_negative_int("completed_update_count", self.completed_update_count),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "proportional": list(self.proportional),
            "integral": list(self.integral),
            "multipliers": list(self.multipliers),
            "completed_update_count": self.completed_update_count,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PILagrangeStateV1":
        if set(raw) != {
            "schema",
            "proportional",
            "integral",
            "multipliers",
            "completed_update_count",
        }:
            raise ValueError("invalid PI multiplier state fields")
        if raw["schema"] != cls.schema:
            raise ValueError("PI multiplier state schema mismatch")
        return cls(
            proportional=_float_vector4("proportional", raw["proportional"]),
            integral=_float_vector4("integral", raw["integral"], non_negative=True),
            multipliers=_float_vector4(
                "multipliers", raw["multipliers"], non_negative=True
            ),
            completed_update_count=_non_negative_int(
                "completed_update_count", raw["completed_update_count"]
            ),
        )


class PILagrangeControllerV1:
    """Stateful, channel-independent PI/I multiplier update."""

    def __init__(
        self,
        settings: ConstrainedPPOSettingsV1,
        state: PILagrangeStateV1 | None = None,
    ) -> None:
        self.settings = settings.validate()
        if state is None:
            initial = settings.initial_integral
            state = PILagrangeStateV1(
                proportional=(0.0, 0.0, 0.0, 0.0),
                integral=initial,
                multipliers=initial,
                completed_update_count=0,
            )
        self._validate_state(state)
        self.state = state

    def _validate_state(self, state: PILagrangeStateV1) -> None:
        minimum = self.settings.multiplier_min
        maximum = self.settings.multiplier_max
        for name, values in (
            ("integral", state.integral),
            ("multipliers", state.multipliers),
        ):
            if any(value < minimum or value > maximum for value in values):
                raise ValueError(f"PI state {name} lies outside multiplier bounds")
        if not self.settings.uses_constraints and (
            any(state.proportional)
            or any(state.integral)
            or any(state.multipliers)
            or state.completed_update_count != 0
        ):
            raise ValueError("scalar PPO cannot carry active PI state")

    def update(
        self,
        completed_episode_costs: Sequence[Sequence[float]],
    ) -> PILagrangeStateV1:
        """Update once from complete episode sums, or remain bitwise unchanged."""

        if not completed_episode_costs or not self.settings.uses_constraints:
            return self.state
        costs = np.asarray(completed_episode_costs, dtype=np.float64)
        if costs.ndim != 2 or costs.shape[1] != COST_CHANNEL_COUNT:
            raise ValueError("completed episode costs must have shape (N, 4)")
        if not np.all(np.isfinite(costs)) or np.any(costs < 0.0):
            raise ValueError("completed episode costs must be finite and non-negative")
        error = costs.mean(axis=0) - np.asarray(
            self.settings.cost_limits, dtype=np.float64
        )
        previous_p = np.asarray(self.state.proportional, dtype=np.float64)
        previous_i = np.asarray(self.state.integral, dtype=np.float64)
        alpha = self.settings.proportional_ema_alpha
        proportional = alpha * previous_p + (1.0 - alpha) * error
        integral = np.clip(
            previous_i + self.settings.integral_gain * error,
            self.settings.multiplier_min,
            self.settings.multiplier_max,
        )
        multipliers = np.clip(
            integral + self.settings.proportional_gain * proportional,
            self.settings.multiplier_min,
            self.settings.multiplier_max,
        )
        self.state = PILagrangeStateV1(
            proportional=cast(FloatVector4, tuple(float(x) for x in proportional)),
            integral=cast(FloatVector4, tuple(float(x) for x in integral)),
            multipliers=cast(FloatVector4, tuple(float(x) for x in multipliers)),
            completed_update_count=self.state.completed_update_count + 1,
        )
        return self.state

    def load_state(self, state: PILagrangeStateV1) -> None:
        self._validate_state(state)
        self.state = state


def normalize_advantage(
    advantage: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Sample-normalize one minibatch, replacing constants by exact zeros."""

    values = np.asarray(advantage, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("advantage must be a one-dimensional minibatch")
    if not np.all(np.isfinite(values)):
        raise ValueError("advantage must be finite")
    if values.size <= 1 or np.all(values == values[0]):
        return np.zeros_like(values, dtype=np.float64)
    return (values - values.mean()) / (values.std(ddof=1) + 1e-8)


def combine_advantages(
    reward_advantage: npt.ArrayLike,
    cost_advantages: npt.ArrayLike,
    multipliers: Sequence[float],
    *,
    constrained: bool = True,
) -> npt.NDArray[np.float64]:
    """Independently normalize all streams and build the frozen actor signal."""

    reward = normalize_advantage(reward_advantage)
    costs = np.asarray(cost_advantages, dtype=np.float64)
    if costs.shape != (reward.size, COST_CHANNEL_COUNT):
        raise ValueError("cost_advantages must have shape (minibatch, 4)")
    if not np.all(np.isfinite(costs)):
        raise ValueError("cost advantages must be finite")
    if not constrained:
        return reward
    multiplier = np.asarray(
        _float_vector4("multipliers", multipliers, non_negative=True),
        dtype=np.float64,
    )
    normalized_costs = np.column_stack(
        [normalize_advantage(costs[:, index]) for index in range(COST_CHANNEL_COUNT)]
    )
    return (reward - normalized_costs @ multiplier) / (1.0 + multiplier.sum())


def compute_vector_gae(
    signals: npt.ArrayLike,
    values: npt.ArrayLike,
    episode_starts: npt.ArrayLike,
    last_values: npt.ArrayLike,
    dones: npt.ArrayLike,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute GAE for one or many heads with exact terminal/cut semantics.

    ``signals`` and ``values`` are ``(steps, envs, heads)``.  A true terminal
    is represented by ``dones=True`` at a rollout's last transition or by an
    episode-start marker on the following transition.  A rollout cut has a
    false done and therefore uses ``last_values`` from the same continuing
    environment state.
    """

    gamma_value = _non_negative_float("gamma", gamma)
    lambda_value = _non_negative_float("gae_lambda", gae_lambda)
    if gamma_value > 1.0 or lambda_value > 1.0:
        raise ValueError("gamma and gae_lambda must not exceed one")
    signal_array = np.asarray(signals, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    starts = np.asarray(episode_starts, dtype=np.float64)
    final_values = np.asarray(last_values, dtype=np.float64)
    final_dones = np.asarray(dones)
    if signal_array.ndim != 3:
        raise ValueError("signals must have shape (steps, envs, heads)")
    if value_array.shape != signal_array.shape:
        raise ValueError("values must have the same shape as signals")
    steps, envs, heads = signal_array.shape
    if starts.shape != (steps, envs):
        raise ValueError("episode_starts must have shape (steps, envs)")
    if final_values.shape != (envs, heads):
        raise ValueError("last_values must have shape (envs, heads)")
    if final_dones.shape != (envs,) or final_dones.dtype != np.bool_:
        raise TypeError("dones must be a boolean vector with one value per env")
    for name, array in (
        ("signals", signal_array),
        ("values", value_array),
        ("episode_starts", starts),
        ("last_values", final_values),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
    if np.any((starts != 0.0) & (starts != 1.0)):
        raise ValueError("episode_starts must contain only zero or one")

    advantages = np.zeros_like(signal_array, dtype=np.float64)
    last_gae = np.zeros((envs, heads), dtype=np.float64)
    for step in reversed(range(steps)):
        if step == steps - 1:
            next_non_terminal = (~final_dones).astype(np.float64)
            next_values = final_values
        else:
            next_non_terminal = 1.0 - starts[step + 1]
            next_values = value_array[step + 1]
        mask = next_non_terminal[:, None]
        delta = (
            signal_array[step] + gamma_value * next_values * mask - value_array[step]
        )
        last_gae = delta + gamma_value * lambda_value * mask * last_gae
        advantages[step] = last_gae
    return advantages, advantages + value_array


@dataclass(frozen=True, slots=True)
class ConstraintTransitionCostV1:
    """Trainer-facing validated view of an environment transition record."""

    decision_index: int
    cycle_settled: bool
    cycle_index: int | None
    episode_cycle_count: int
    raw_cost: FloatVector4
    training_cost: FloatVector4
    source_sha256: str


def extract_constraint_transition(
    info: Mapping[str, Any],
    *,
    key: str = TRANSITION_INFO_KEY,
) -> ConstraintTransitionCostV1:
    """Fail closed while extracting the exact versioned four-channel cost."""

    if not isinstance(info, Mapping):
        raise TypeError("environment info must be a mapping")
    raw = info.get(key)
    if not isinstance(raw, Mapping):
        raise ValueError(f"environment info omitted {key!r}")
    expected_fields = {
        "schema",
        "decision_index",
        "cycle_settled",
        "cycle_index",
        "episode_cycle_count",
        "resource_order",
        "raw_cost",
        "training_cost",
        "raw_cost_sum",
        "training_cost_sum",
        "sha256",
    }
    if set(raw) != expected_fields:
        raise ValueError("constraint transition fields do not match the V1 schema")
    if raw["schema"] != CYCLE_CONSTRAINT_TRANSITION_SCHEMA:
        raise ValueError("constraint transition schema mismatch")
    if tuple(raw["resource_order"]) != RESOURCE_ORDER:
        raise ValueError("constraint transition resource order mismatch")
    decision_index = _non_negative_int("decision_index", raw["decision_index"])
    if not 1 <= decision_index <= BETA_DECISIONS_PER_EPISODE:
        raise ValueError("decision_index must lie in the beta episode range 1..672")
    cycle_settled = raw["cycle_settled"]
    if not isinstance(cycle_settled, bool):
        raise TypeError("cycle_settled must be a bool")
    cycle_index_raw = raw["cycle_index"]
    if cycle_settled:
        cycle_index = _non_negative_int("cycle_index", cycle_index_raw)
        if not 1 <= cycle_index <= BETA_EPISODE_CYCLE_COUNT:
            raise ValueError("settlement cycle_index must lie in the range 1..7")
        if decision_index != cycle_index * BETA_DECISIONS_PER_CYCLE:
            raise ValueError("settlement decision_index does not match cycle_index")
    else:
        if cycle_index_raw is not None:
            raise ValueError("non-settlement cycle_index must be None")
        cycle_index = None
    episode_cycle_count = _non_negative_int(
        "episode_cycle_count", raw["episode_cycle_count"]
    )
    if episode_cycle_count != BETA_EPISODE_CYCLE_COUNT:
        raise ValueError("constrained beta episode_cycle_count must be exactly seven")
    raw_cost = _float_vector4("raw_cost", raw["raw_cost"], non_negative=True)
    training_cost = _float_vector4(
        "training_cost", raw["training_cost"], non_negative=True
    )
    expected_training = tuple(value / episode_cycle_count for value in raw_cost)
    if training_cost != expected_training:
        raise ValueError("training cost does not equal raw cost / episode cycles")
    if not cycle_settled and (any(raw_cost) or any(training_cost)):
        raise ValueError("non-settlement transition cost must be exactly zero")
    if _finite_float("raw_cost_sum", raw["raw_cost_sum"]) != math.fsum(raw_cost):
        raise ValueError("raw constraint cost checksum mismatch")
    if _finite_float("training_cost_sum", raw["training_cost_sum"]) != math.fsum(
        training_cost
    ):
        raise ValueError("training constraint cost checksum mismatch")
    body = {name: raw[name] for name in expected_fields if name != "sha256"}
    source_sha256 = raw["sha256"]
    if not isinstance(source_sha256, str) or source_sha256 != canonical_sha256(body):
        raise ValueError("constraint transition digest mismatch")
    return ConstraintTransitionCostV1(
        decision_index=decision_index,
        cycle_settled=cycle_settled,
        cycle_index=cycle_index,
        episode_cycle_count=episode_cycle_count,
        raw_cost=raw_cost,
        training_cost=training_cost,
        source_sha256=source_sha256,
    )


@dataclass(frozen=True, slots=True)
class CompletedEpisodeCostV1:
    """A complete episode's cost totals emitted exactly once."""

    environment_index: int
    raw_cost: FloatVector4
    training_cost: FloatVector4
    settled_cycle_count: int
    expected_cycle_count: int

    schema: ClassVar[str] = EPISODE_COST_SUMMARY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "environment_index",
            _non_negative_int("environment_index", self.environment_index),
        )
        object.__setattr__(
            self,
            "raw_cost",
            _float_vector4("raw_cost", self.raw_cost, non_negative=True),
        )
        object.__setattr__(
            self,
            "training_cost",
            _float_vector4("training_cost", self.training_cost, non_negative=True),
        )
        object.__setattr__(
            self,
            "settled_cycle_count",
            _non_negative_int("settled_cycle_count", self.settled_cycle_count),
        )
        object.__setattr__(
            self,
            "expected_cycle_count",
            _non_negative_int("expected_cycle_count", self.expected_cycle_count),
        )
        if self.expected_cycle_count != BETA_EPISODE_CYCLE_COUNT:
            raise ValueError("completed beta episode must expect exactly seven cycles")
        if self.settled_cycle_count != self.expected_cycle_count:
            raise ValueError("completed episode lost a cycle settlement")
        if not np.allclose(
            np.asarray(self.training_cost),
            np.asarray(self.raw_cost) / BETA_EPISODE_CYCLE_COUNT,
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError("completed episode cost scaling is incoherent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "environment_index": self.environment_index,
            "raw_cost": list(self.raw_cost),
            "training_cost": list(self.training_cost),
            "settled_cycle_count": self.settled_cycle_count,
            "expected_cycle_count": self.expected_cycle_count,
        }


class EpisodeCostAccumulatorV1:
    """Carry incomplete episode costs across rollout boundaries."""

    n_envs: int
    raw_cost: npt.NDArray[np.float64]
    training_cost: npt.NDArray[np.float64]
    settled_cycles: npt.NDArray[np.int64]
    expected_cycles: npt.NDArray[np.int64]
    next_cycle_index: npt.NDArray[np.int64]
    next_decision_index: npt.NDArray[np.int64]

    def __init__(self, n_envs: int) -> None:
        if isinstance(n_envs, bool) or not isinstance(n_envs, Integral):
            raise TypeError("n_envs must be an integer")
        if n_envs <= 0:
            raise ValueError("n_envs must be positive")
        self.n_envs = int(n_envs)
        self.raw_cost = np.zeros((self.n_envs, COST_CHANNEL_COUNT), dtype=np.float64)
        self.training_cost = np.zeros_like(self.raw_cost)
        self.settled_cycles = np.zeros(self.n_envs, dtype=np.int64)
        self.expected_cycles = np.zeros(self.n_envs, dtype=np.int64)
        self.next_cycle_index = np.ones(self.n_envs, dtype=np.int64)
        self.next_decision_index = np.ones(self.n_envs, dtype=np.int64)

    @property
    def has_partial_episode(self) -> bool:
        return bool(np.any(self.next_decision_index != 1))

    @property
    def current_decision_indices(self) -> tuple[int, ...]:
        return tuple(int(value - 1) for value in self.next_decision_index)

    def observe(
        self,
        transitions: Sequence[ConstraintTransitionCostV1],
        dones: npt.ArrayLike,
    ) -> tuple[CompletedEpisodeCostV1, ...]:
        if len(transitions) != self.n_envs:
            raise ValueError("one constraint transition is required per environment")
        done_array = np.asarray(dones)
        if done_array.shape != (self.n_envs,) or done_array.dtype != np.bool_:
            raise TypeError("dones must be a boolean vector")
        completed: list[CompletedEpisodeCostV1] = []
        for index, transition in enumerate(transitions):
            if transition.decision_index != int(self.next_decision_index[index]):
                raise RuntimeError(
                    "constraint transition/reset discontinuity for environment "
                    f"{index}: expected decision {self.next_decision_index[index]}, "
                    f"received {transition.decision_index}"
                )
            expected = int(self.expected_cycles[index])
            if expected not in (0, transition.episode_cycle_count):
                raise RuntimeError("episode cycle count changed before termination")
            if transition.episode_cycle_count != BETA_EPISODE_CYCLE_COUNT:
                raise RuntimeError("constrained beta episode must contain seven cycles")
            self.expected_cycles[index] = transition.episode_cycle_count
            boundary = transition.decision_index % BETA_DECISIONS_PER_CYCLE == 0
            if transition.cycle_settled != boundary:
                raise RuntimeError(
                    "cycle settlement presence does not match its 24-hour boundary"
                )
            if transition.cycle_settled:
                expected_cycle_index = int(self.next_cycle_index[index])
                if transition.cycle_index != expected_cycle_index:
                    raise RuntimeError(
                        "cycle settlements are missing, duplicated, or out of order"
                    )
                if (
                    transition.decision_index
                    != expected_cycle_index * BETA_DECISIONS_PER_CYCLE
                ):
                    raise RuntimeError(
                        "cycle settlement has the wrong decision boundary"
                    )
            self.raw_cost[index] += np.asarray(transition.raw_cost, dtype=np.float64)
            self.training_cost[index] += np.asarray(
                transition.training_cost, dtype=np.float64
            )
            if transition.cycle_settled:
                self.settled_cycles[index] += 1
                self.next_cycle_index[index] += 1
            if done_array[index]:
                if transition.decision_index != BETA_DECISIONS_PER_EPISODE:
                    raise RuntimeError("beta episode terminated before decision 672")
                summary = CompletedEpisodeCostV1(
                    environment_index=index,
                    raw_cost=cast(
                        FloatVector4,
                        tuple(float(value) for value in self.raw_cost[index]),
                    ),
                    training_cost=cast(
                        FloatVector4,
                        tuple(float(value) for value in self.training_cost[index]),
                    ),
                    settled_cycle_count=int(self.settled_cycles[index]),
                    expected_cycle_count=int(self.expected_cycles[index]),
                )
                completed.append(summary)
                self.raw_cost[index].fill(0.0)
                self.training_cost[index].fill(0.0)
                self.settled_cycles[index] = 0
                self.expected_cycles[index] = 0
                self.next_cycle_index[index] = 1
                self.next_decision_index[index] = 1
            else:
                if transition.decision_index >= BETA_DECISIONS_PER_EPISODE:
                    raise RuntimeError("beta episode did not terminate at decision 672")
                self.next_decision_index[index] += 1
        return tuple(completed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EPISODE_COST_ACCUMULATOR_SCHEMA,
            "n_envs": self.n_envs,
            "raw_cost": self.raw_cost.tolist(),
            "training_cost": self.training_cost.tolist(),
            "settled_cycles": self.settled_cycles.tolist(),
            "expected_cycles": self.expected_cycles.tolist(),
            "next_cycle_index": self.next_cycle_index.tolist(),
            "next_decision_index": self.next_decision_index.tolist(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EpisodeCostAccumulatorV1":
        fields = {
            "schema",
            "n_envs",
            "raw_cost",
            "training_cost",
            "settled_cycles",
            "expected_cycles",
            "next_cycle_index",
            "next_decision_index",
        }
        if set(raw) != fields or raw["schema"] != EPISODE_COST_ACCUMULATOR_SCHEMA:
            raise ValueError("episode cost accumulator schema/fields mismatch")
        result = cls(_non_negative_int("n_envs", raw["n_envs"]))
        raw_cost = np.asarray(raw["raw_cost"], dtype=np.float64)
        training_cost = np.asarray(raw["training_cost"], dtype=np.float64)
        settled = np.asarray(raw["settled_cycles"])
        expected = np.asarray(raw["expected_cycles"])
        next_cycle = np.asarray(raw["next_cycle_index"])
        next_decision = np.asarray(raw["next_decision_index"])
        vector_shape = (result.n_envs, COST_CHANNEL_COUNT)
        if raw_cost.shape != vector_shape or training_cost.shape != vector_shape:
            raise ValueError("episode accumulator cost arrays have the wrong shape")
        if not np.all(np.isfinite(raw_cost)) or not np.all(np.isfinite(training_cost)):
            raise ValueError("episode accumulator costs must be finite")
        if np.any(raw_cost < 0.0) or np.any(training_cost < 0.0):
            raise ValueError("episode accumulator costs must be non-negative")
        if (
            settled.shape != (result.n_envs,)
            or expected.shape != (result.n_envs,)
            or next_cycle.shape != (result.n_envs,)
            or next_decision.shape != (result.n_envs,)
        ):
            raise ValueError("episode accumulator cursor arrays have the wrong shape")
        if any(
            array.dtype.kind not in "iu"
            for array in (settled, expected, next_cycle, next_decision)
        ):
            raise TypeError("episode accumulator cursors must be integers")
        if (
            np.any(settled < 0)
            or np.any(expected < 0)
            or np.any(next_cycle <= 0)
            or np.any(next_decision <= 0)
            or np.any(next_decision > BETA_DECISIONS_PER_EPISODE)
        ):
            raise ValueError("episode accumulator cursors lie outside their domain")
        fresh = next_decision == 1
        if np.any(expected[~fresh] != BETA_EPISODE_CYCLE_COUNT) or np.any(
            expected[fresh] != 0
        ):
            raise ValueError("episode accumulator expected-cycle state is incoherent")
        expected_settled = (next_decision - 1) // BETA_DECISIONS_PER_CYCLE
        if np.any(settled != expected_settled) or np.any(next_cycle != settled + 1):
            raise ValueError("episode accumulator cycle cursors are incoherent")
        if np.any(fresh[:, None] & ((raw_cost != 0.0) | (training_cost != 0.0))):
            raise ValueError("fresh episode accumulator must have zero costs")
        before_first_settlement = settled == 0
        if np.any(
            before_first_settlement[:, None]
            & ((raw_cost != 0.0) | (training_cost != 0.0))
        ):
            raise ValueError("pre-settlement accumulator must have zero costs")
        if not np.allclose(
            training_cost,
            raw_cost / BETA_EPISODE_CYCLE_COUNT,
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError("episode accumulator cost scaling is incoherent")
        result.raw_cost[:] = raw_cost
        result.training_cost[:] = training_cost
        result.settled_cycles[:] = settled
        result.expected_cycles[:] = expected
        result.next_cycle_index[:] = next_cycle
        result.next_decision_index[:] = next_decision
        return result


@dataclass(frozen=True, slots=True)
class RolloutLedgerV1:
    """Hash-chained optimizer and constraint evidence for one rollout."""

    algorithm_id: str
    rollout_index: int
    transition_count: int
    completed_episode_count: int
    complete_cycle_count: int
    reward_return: float
    raw_cost: FloatVector4
    training_cost: FloatVector4
    mean_completed_episode_cost: FloatVector4 | None
    cost_value_losses: FloatVector4
    multipliers: FloatVector4
    proportional_state: FloatVector4
    integral_state: FloatVector4
    multiplier_completed_update_count: int
    multiplier_saturated: BoolVector4
    approximate_kl: float
    entropy_loss: float
    clip_fraction: float
    optimizer_steps: int
    elapsed_compute_s: float
    rollout_discontinuities: int
    previous_sha256: str | None
    sha256: str

    schema: ClassVar[str] = ROLLOUT_LEDGER_SCHEMA

    @classmethod
    def build(
        cls,
        *,
        algorithm_id: str,
        rollout_index: int,
        transition_count: int,
        completed_episode_count: int,
        complete_cycle_count: int,
        reward_return: float,
        raw_cost: Sequence[float],
        training_cost: Sequence[float],
        mean_completed_episode_cost: Sequence[float] | None,
        cost_value_losses: Sequence[float],
        multiplier_state: PILagrangeStateV1,
        multiplier_min: float,
        multiplier_max: float,
        approximate_kl: float,
        entropy_loss: float,
        clip_fraction: float,
        optimizer_steps: int,
        elapsed_compute_s: float,
        rollout_discontinuities: int = 0,
        previous_sha256: str | None = None,
    ) -> "RolloutLedgerV1":
        if algorithm_id not in SUPPORTED_ALGORITHM_IDS:
            raise ValueError("rollout algorithm is not registered")
        multipliers = multiplier_state.multipliers
        saturated = cast(
            BoolVector4,
            tuple(
                value == multiplier_min or value == multiplier_max
                for value in multipliers
            ),
        )
        normalized_mean = (
            None
            if mean_completed_episode_cost is None
            else _float_vector4(
                "mean_completed_episode_cost",
                mean_completed_episode_cost,
                non_negative=True,
            )
        )
        normalized_discontinuities = _non_negative_int(
            "rollout_discontinuities", rollout_discontinuities
        )
        if normalized_discontinuities != 0:
            raise ValueError("registered rollouts must contain zero discontinuities")
        values: dict[str, Any] = {
            "schema": cls.schema,
            "protocol_id": PROTOCOL_ID,
            "algorithm_id": algorithm_id,
            "rollout_index": _non_negative_int("rollout_index", rollout_index),
            "transition_count": _non_negative_int("transition_count", transition_count),
            "completed_episode_count": _non_negative_int(
                "completed_episode_count", completed_episode_count
            ),
            "complete_cycle_count": _non_negative_int(
                "complete_cycle_count", complete_cycle_count
            ),
            "reward_return": _finite_float("reward_return", reward_return),
            "raw_cost": list(_float_vector4("raw_cost", raw_cost, non_negative=True)),
            "training_cost": list(
                _float_vector4("training_cost", training_cost, non_negative=True)
            ),
            "mean_completed_episode_cost": (
                None if normalized_mean is None else list(normalized_mean)
            ),
            "cost_value_losses": list(
                _float_vector4(
                    "cost_value_losses", cost_value_losses, non_negative=True
                )
            ),
            "multipliers": list(multipliers),
            "proportional_state": list(multiplier_state.proportional),
            "integral_state": list(multiplier_state.integral),
            "multiplier_completed_update_count": (
                multiplier_state.completed_update_count
            ),
            "multiplier_saturated": list(saturated),
            "approximate_kl": _non_negative_float("approximate_kl", approximate_kl),
            "entropy_loss": _finite_float("entropy_loss", entropy_loss),
            "clip_fraction": _non_negative_float("clip_fraction", clip_fraction),
            "optimizer_steps": _non_negative_int("optimizer_steps", optimizer_steps),
            "elapsed_compute_s": _non_negative_float(
                "elapsed_compute_s", elapsed_compute_s
            ),
            "rollout_discontinuities": normalized_discontinuities,
            "previous_sha256": previous_sha256,
        }
        if values["transition_count"] <= 0:
            raise ValueError("a rollout ledger must contain transitions")
        if values["clip_fraction"] > 1.0:
            raise ValueError("clip_fraction cannot exceed one")
        if previous_sha256 is not None and (
            not isinstance(previous_sha256, str) or len(previous_sha256) != 64
        ):
            raise ValueError("previous rollout digest must be a SHA-256 hex string")
        digest = _canonical_digest(cls.schema, values)
        return cls(
            algorithm_id=algorithm_id,
            rollout_index=values["rollout_index"],
            transition_count=values["transition_count"],
            completed_episode_count=values["completed_episode_count"],
            complete_cycle_count=values["complete_cycle_count"],
            reward_return=values["reward_return"],
            raw_cost=cast(FloatVector4, tuple(values["raw_cost"])),
            training_cost=cast(FloatVector4, tuple(values["training_cost"])),
            mean_completed_episode_cost=normalized_mean,
            cost_value_losses=cast(FloatVector4, tuple(values["cost_value_losses"])),
            multipliers=multipliers,
            proportional_state=multiplier_state.proportional,
            integral_state=multiplier_state.integral,
            multiplier_completed_update_count=(multiplier_state.completed_update_count),
            multiplier_saturated=saturated,
            approximate_kl=values["approximate_kl"],
            entropy_loss=values["entropy_loss"],
            clip_fraction=values["clip_fraction"],
            optimizer_steps=values["optimizer_steps"],
            elapsed_compute_s=values["elapsed_compute_s"],
            rollout_discontinuities=values["rollout_discontinuities"],
            previous_sha256=previous_sha256,
            sha256=digest,
        )

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol_id": PROTOCOL_ID,
            "algorithm_id": self.algorithm_id,
            "rollout_index": self.rollout_index,
            "transition_count": self.transition_count,
            "completed_episode_count": self.completed_episode_count,
            "complete_cycle_count": self.complete_cycle_count,
            "reward_return": self.reward_return,
            "raw_cost": list(self.raw_cost),
            "training_cost": list(self.training_cost),
            "mean_completed_episode_cost": (
                None
                if self.mean_completed_episode_cost is None
                else list(self.mean_completed_episode_cost)
            ),
            "cost_value_losses": list(self.cost_value_losses),
            "multipliers": list(self.multipliers),
            "proportional_state": list(self.proportional_state),
            "integral_state": list(self.integral_state),
            "multiplier_completed_update_count": (
                self.multiplier_completed_update_count
            ),
            "multiplier_saturated": list(self.multiplier_saturated),
            "approximate_kl": self.approximate_kl,
            "entropy_loss": self.entropy_loss,
            "clip_fraction": self.clip_fraction,
            "optimizer_steps": self.optimizer_steps,
            "elapsed_compute_s": self.elapsed_compute_s,
            "rollout_discontinuities": self.rollout_discontinuities,
            "previous_sha256": self.previous_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}

    def verify(self, *, expected_previous_sha256: str | None) -> None:
        if self.previous_sha256 != expected_previous_sha256:
            raise ValueError("rollout ledger chain is discontinuous")
        if _canonical_digest(self.schema, self.body()) != self.sha256:
            raise ValueError("rollout ledger digest mismatch")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RolloutLedgerV1":
        expected_fields = {
            "schema",
            "protocol_id",
            "algorithm_id",
            "rollout_index",
            "transition_count",
            "completed_episode_count",
            "complete_cycle_count",
            "reward_return",
            "raw_cost",
            "training_cost",
            "mean_completed_episode_cost",
            "cost_value_losses",
            "multipliers",
            "proportional_state",
            "integral_state",
            "multiplier_completed_update_count",
            "multiplier_saturated",
            "approximate_kl",
            "entropy_loss",
            "clip_fraction",
            "optimizer_steps",
            "elapsed_compute_s",
            "rollout_discontinuities",
            "previous_sha256",
            "sha256",
        }
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            raise ValueError("rollout ledger fields do not match the V1 schema")
        if raw.get("schema") != cls.schema:
            raise ValueError("rollout ledger schema mismatch")
        if raw.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("rollout ledger protocol mismatch")
        state = PILagrangeStateV1(
            proportional=_float_vector4(
                "proportional_state", raw["proportional_state"]
            ),
            integral=_float_vector4(
                "integral_state", raw["integral_state"], non_negative=True
            ),
            multipliers=_float_vector4(
                "multipliers", raw["multipliers"], non_negative=True
            ),
            completed_update_count=_non_negative_int(
                "multiplier_completed_update_count",
                raw["multiplier_completed_update_count"],
            ),
        )
        rebuilt = cls.build(
            algorithm_id=str(raw["algorithm_id"]),
            rollout_index=_non_negative_int("rollout_index", raw["rollout_index"]),
            transition_count=_non_negative_int(
                "transition_count", raw["transition_count"]
            ),
            completed_episode_count=_non_negative_int(
                "completed_episode_count", raw["completed_episode_count"]
            ),
            complete_cycle_count=_non_negative_int(
                "complete_cycle_count", raw["complete_cycle_count"]
            ),
            reward_return=_finite_float("reward_return", raw["reward_return"]),
            raw_cost=raw["raw_cost"],
            training_cost=raw["training_cost"],
            mean_completed_episode_cost=raw["mean_completed_episode_cost"],
            cost_value_losses=raw["cost_value_losses"],
            multiplier_state=state,
            multiplier_min=0.0,
            multiplier_max=100.0,
            approximate_kl=_non_negative_float("approximate_kl", raw["approximate_kl"]),
            entropy_loss=_finite_float("entropy_loss", raw["entropy_loss"]),
            clip_fraction=_non_negative_float("clip_fraction", raw["clip_fraction"]),
            optimizer_steps=_non_negative_int(
                "optimizer_steps", raw["optimizer_steps"]
            ),
            elapsed_compute_s=_non_negative_float(
                "elapsed_compute_s", raw["elapsed_compute_s"]
            ),
            rollout_discontinuities=_non_negative_int(
                "rollout_discontinuities", raw["rollout_discontinuities"]
            ),
            previous_sha256=cast(str | None, raw["previous_sha256"]),
        )
        if (
            _bool_vector4("multiplier_saturated", raw["multiplier_saturated"])
            != rebuilt.multiplier_saturated
        ):
            raise ValueError("rollout multiplier saturation disclosure mismatch")
        if raw.get("sha256") != rebuilt.sha256:
            raise ValueError("rollout ledger digest mismatch")
        return rebuilt


@dataclass(frozen=True, slots=True)
class ExternalEpisodeStatePinV1:
    """Canonical opaque pin supplied by the episode allocator/runtime layer."""

    payload_json: str
    sha256: str

    schema: ClassVar[str] = EXTERNAL_EPISODE_STATE_PIN_SCHEMA

    @classmethod
    def build(
        cls,
        payload: Mapping[str, Any],
        *,
        sha256: str | None = None,
    ) -> "ExternalEpisodeStatePinV1":
        normalized = _canonical_object_copy("external episode state", payload)
        payload_json = _canonical_json(normalized)
        expected = _canonical_digest(cls.schema, normalized)
        if sha256 is not None and sha256 != expected:
            raise ValueError("external episode state digest mismatch")
        return cls(payload_json=payload_json, sha256=expected)

    @property
    def payload(self) -> dict[str, Any]:
        decoded = json.loads(self.payload_json)
        if not isinstance(decoded, dict):  # pragma: no cover - build/load validate this
            raise ValueError("external episode state payload is not an object")
        return cast(dict[str, Any], decoded)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "payload": self.payload,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExternalEpisodeStatePinV1":
        if set(raw) != {"schema", "payload", "sha256"}:
            raise ValueError("external episode state pin fields mismatch")
        if raw["schema"] != cls.schema:
            raise ValueError("external episode state pin schema mismatch")
        digest = raw["sha256"]
        if not isinstance(digest, str):
            raise TypeError("external episode state digest must be a string")
        return cls.build(raw["payload"], sha256=digest)


@dataclass(frozen=True, slots=True)
class AbortedRolloutV1:
    """Permanent evidence that transitions were consumed without optimization."""

    reason: str
    transition_count: int
    num_timesteps_after_abort: int
    sha256: str

    schema: ClassVar[str] = ABORTED_ROLLOUT_SCHEMA

    @classmethod
    def build(
        cls,
        *,
        reason: str,
        transition_count: int,
        num_timesteps_after_abort: int,
    ) -> "AbortedRolloutV1":
        if reason not in {
            "callback-returned-false",
            "rollout-exception",
            "optimizer-exception",
        }:
            raise ValueError("unknown aborted-rollout reason")
        normalized_transition_count = _non_negative_int(
            "transition_count", transition_count
        )
        normalized_num_timesteps = _non_negative_int(
            "num_timesteps_after_abort", num_timesteps_after_abort
        )
        body: dict[str, Any] = {
            "schema": cls.schema,
            "reason": reason,
            "transition_count": normalized_transition_count,
            "num_timesteps_after_abort": normalized_num_timesteps,
        }
        if normalized_transition_count <= 0:
            raise ValueError("an aborted rollout must retain consumed transitions")
        return cls(
            reason=reason,
            transition_count=normalized_transition_count,
            num_timesteps_after_abort=normalized_num_timesteps,
            sha256=_canonical_digest(cls.schema, body),
        )

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "reason": self.reason,
            "transition_count": self.transition_count,
            "num_timesteps_after_abort": self.num_timesteps_after_abort,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AbortedRolloutV1":
        if set(raw) != {
            "schema",
            "reason",
            "transition_count",
            "num_timesteps_after_abort",
            "sha256",
        }:
            raise ValueError("aborted rollout fields mismatch")
        if raw["schema"] != cls.schema:
            raise ValueError("aborted rollout schema mismatch")
        rebuilt = cls.build(
            reason=str(raw["reason"]),
            transition_count=_non_negative_int(
                "transition_count", raw["transition_count"]
            ),
            num_timesteps_after_abort=_non_negative_int(
                "num_timesteps_after_abort", raw["num_timesteps_after_abort"]
            ),
        )
        if raw["sha256"] != rebuilt.sha256:
            raise ValueError("aborted rollout digest mismatch")
        return rebuilt


try:
    import torch as th
    from torch import nn
    from torch.nn import functional as F

    from gymnasium import spaces
    from stable_baselines3 import PPO as _SB3PPO
    from stable_baselines3.common.buffers import RolloutBuffer
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.type_aliases import GymEnv, Schedule
    from stable_baselines3.common.utils import (
        explained_variance,
        obs_as_tensor,
        set_random_seed as sb3_set_random_seed,
    )
    from stable_baselines3.common.vec_env import VecEnv

    _TRAINING_IMPORT_ERROR: ModuleNotFoundError | None = None
except (
    ModuleNotFoundError
) as reason:  # pragma: no cover - exercised without train group
    th = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    _SB3PPO = object  # type: ignore[assignment,misc]
    RolloutBuffer = object  # type: ignore[assignment,misc]
    BaseCallback = Any  # type: ignore[assignment,misc]
    ActorCriticPolicy = object  # type: ignore[assignment,misc]
    GymEnv = Any  # type: ignore[assignment,misc]
    Schedule = Any  # type: ignore[assignment,misc]
    VecEnv = Any  # type: ignore[assignment,misc]
    sb3_set_random_seed = None  # type: ignore[assignment]
    _TRAINING_IMPORT_ERROR = reason


def training_dependencies_available() -> bool:
    """Return whether model APIs are usable in this interpreter."""

    return _TRAINING_IMPORT_ERROR is None


def _require_training_dependencies() -> None:
    if _TRAINING_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "constrained PPO requires the optional 'train' dependency group "
            "(Stable-Baselines3 and Torch)"
        ) from _TRAINING_IMPORT_ERROR


class ConstrainedRolloutBufferSamplesV1(NamedTuple):
    observations: Any
    actions: Any
    old_values: Any
    old_log_prob: Any
    advantages: Any
    returns: Any
    old_cost_values: Any
    cost_advantages: Any
    cost_returns: Any
    raw_costs: Any
    training_costs: Any
    cycle_settled: Any


if TYPE_CHECKING or _TRAINING_IMPORT_ERROR is None:

    def _require_finite_array(name: str, value: object) -> None:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as reason:
            raise ValueError(f"{name} must be numeric and finite") from reason
        if array.dtype.kind not in "biufc" or not bool(np.isfinite(array).all()):
            raise ValueError(f"{name} must be finite")

    def _require_finite_tensor(name: str, value: object) -> None:
        if not isinstance(value, th.Tensor):
            raise TypeError(f"{name} must be a Torch tensor")
        if not bool(th.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")

    def _state_key(value: object) -> tuple[str, str]:
        if isinstance(value, bool):
            return ("bool", "1" if value else "0")
        if isinstance(value, Integral):
            return ("int", str(int(value)))
        if isinstance(value, str):
            return ("str", value)
        raise TypeError("state mappings may use only integer or string keys")

    def _update_state_digest(digest: Any, value: object) -> None:
        """Hash a Torch/optimizer state tree with explicit type boundaries."""

        if isinstance(value, th.Tensor):
            tensor = value.detach().cpu().contiguous()
            _require_finite_tensor("checkpoint tensor", tensor)
            digest.update(b"tensor\x00")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b"\x00")
            digest.update(_canonical_json(list(tensor.shape)).encode("ascii"))
            digest.update(b"\x00")
            digest.update(tensor.numpy().tobytes())
            return
        if isinstance(value, np.ndarray):
            if not bool(np.isfinite(value).all()):
                raise ValueError("checkpoint array must be finite")
            contiguous = np.ascontiguousarray(value)
            digest.update(b"ndarray\x00")
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(b"\x00")
            digest.update(_canonical_json(list(contiguous.shape)).encode("ascii"))
            digest.update(b"\x00")
            digest.update(contiguous.tobytes())
            return
        if isinstance(value, Mapping):
            digest.update(b"mapping\x00")
            keys = sorted(value, key=_state_key)
            digest.update(str(len(keys)).encode("ascii"))
            digest.update(b"\x00")
            for key in keys:
                _update_state_digest(digest, key)
                _update_state_digest(digest, value[key])
            return
        if isinstance(value, tuple):
            digest.update(b"tuple\x00")
            digest.update(str(len(value)).encode("ascii"))
            digest.update(b"\x00")
            for item in value:
                _update_state_digest(digest, item)
            return
        if isinstance(value, list):
            digest.update(b"list\x00")
            digest.update(str(len(value)).encode("ascii"))
            digest.update(b"\x00")
            for item in value:
                _update_state_digest(digest, item)
            return
        if value is None:
            digest.update(b"none\x00")
            return
        if isinstance(value, bool):
            digest.update(b"bool\x00\x01" if value else b"bool\x00\x00")
            return
        if isinstance(value, Integral):
            digest.update(b"int\x00")
            digest.update(str(int(value)).encode("ascii"))
            digest.update(b"\x00")
            return
        if isinstance(value, Real):
            number = _finite_float("checkpoint scalar", value)
            digest.update(b"float\x00")
            digest.update(number.hex().encode("ascii"))
            digest.update(b"\x00")
            return
        if isinstance(value, str):
            encoded = value.encode("utf-8")
            digest.update(b"str\x00")
            digest.update(str(len(encoded)).encode("ascii"))
            digest.update(b"\x00")
            digest.update(encoded)
            return
        raise TypeError(f"unsupported checkpoint state value {type(value).__name__}")

    def _typed_state_sha256(domain: str, value: object) -> str:
        digest = hashlib.sha256(domain.encode("ascii") + b"\x00")
        _update_state_digest(digest, value)
        return digest.hexdigest()

    def _qualified_type(value: object) -> str:
        value_type = value if isinstance(value, type) else type(value)
        return f"{value_type.__module__}.{value_type.__qualname__}"

    def _array_sha256(domain: str, value: npt.ArrayLike) -> str:
        array = np.ascontiguousarray(value)
        if not bool(np.isfinite(array).all()):
            raise ValueError(f"{domain} must be finite")
        digest = hashlib.sha256(domain.encode("ascii") + b"\x00")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\x00")
        digest.update(_canonical_json(list(array.shape)).encode("ascii"))
        digest.update(b"\x00")
        digest.update(array.tobytes())
        return digest.hexdigest()

    def _decode_canonical_base64(name: str, value: object) -> bytes:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a base64 string")
        try:
            decoded = base64.b64decode(value.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError) as reason:
            raise ValueError(f"{name} must be canonical base64") from reason
        if not decoded or base64.b64encode(decoded).decode("ascii") != value:
            raise ValueError(f"{name} must be non-empty canonical base64")
        return decoded

    @dataclass(frozen=True, slots=True)
    class LearnerRNGStateV1:
        """Exact Python, NumPy, and Torch learner RNG state at a ledger cut."""

        python_version: int
        python_internal_state: tuple[int, ...]
        python_gauss_next: float | None
        numpy_bit_generator: str
        numpy_state: tuple[int, ...]
        numpy_position: int
        numpy_has_gauss: int
        numpy_cached_gaussian: float
        torch_cpu_state_b64: str
        torch_cuda_states_b64: tuple[str, ...]

        schema: ClassVar[str] = LEARNER_RNG_STATE_SCHEMA

        def __post_init__(self) -> None:
            if self.python_version != 3:
                raise ValueError("Python RNG state version must be three")
            if len(self.python_internal_state) != 625 or any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in self.python_internal_state
            ):
                raise ValueError("Python RNG internal state must contain 625 integers")
            if self.python_gauss_next is not None:
                object.__setattr__(
                    self,
                    "python_gauss_next",
                    _finite_float("Python RNG Gaussian cache", self.python_gauss_next),
                )
            if self.numpy_bit_generator != "MT19937":
                raise ValueError("NumPy learner RNG must use MT19937")
            if len(self.numpy_state) != 624 or any(
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or not 0 <= int(value) <= (2**32 - 1)
                for value in self.numpy_state
            ):
                raise ValueError("NumPy MT19937 state must contain 624 uint32 values")
            if not 0 <= self.numpy_position <= 624:
                raise ValueError("NumPy RNG position lies outside 0..624")
            if self.numpy_has_gauss not in (0, 1):
                raise ValueError("NumPy RNG Gaussian-cache flag must be zero or one")
            object.__setattr__(
                self,
                "numpy_cached_gaussian",
                _finite_float("NumPy RNG Gaussian cache", self.numpy_cached_gaussian),
            )
            _decode_canonical_base64("Torch CPU RNG state", self.torch_cpu_state_b64)
            for index, encoded in enumerate(self.torch_cuda_states_b64):
                _decode_canonical_base64(f"Torch CUDA RNG state {index}", encoded)

        @classmethod
        def capture(cls) -> "LearnerRNGStateV1":
            python_state = random.getstate()
            numpy_state = cast(
                tuple[str, npt.NDArray[np.uint32], int, int, float],
                np.random.get_state(),
            )

            def encoded(tensor: Any) -> str:
                return base64.b64encode(
                    tensor.detach().cpu().contiguous().numpy().tobytes()
                ).decode("ascii")

            cuda_states = (
                tuple(encoded(state) for state in th.cuda.get_rng_state_all())
                if th.cuda.is_available()
                else ()
            )
            return cls(
                python_version=int(python_state[0]),
                python_internal_state=tuple(int(value) for value in python_state[1]),
                python_gauss_next=(
                    None if python_state[2] is None else float(python_state[2])
                ),
                numpy_bit_generator=str(numpy_state[0]),
                numpy_state=tuple(int(value) for value in numpy_state[1]),
                numpy_position=int(numpy_state[2]),
                numpy_has_gauss=int(numpy_state[3]),
                numpy_cached_gaussian=float(numpy_state[4]),
                torch_cpu_state_b64=encoded(th.random.get_rng_state()),
                torch_cuda_states_b64=cuda_states,
            )

        def restore(self) -> None:
            random.setstate(
                (
                    self.python_version,
                    self.python_internal_state,
                    self.python_gauss_next,
                )
            )
            np.random.set_state(
                (
                    self.numpy_bit_generator,
                    np.asarray(self.numpy_state, dtype=np.uint32),
                    self.numpy_position,
                    self.numpy_has_gauss,
                    self.numpy_cached_gaussian,
                )
            )

            def tensor(encoded: str) -> Any:
                raw = _decode_canonical_base64("Torch RNG state", encoded)
                array = np.frombuffer(raw, dtype=np.uint8).copy()
                return th.from_numpy(array)

            th.random.set_rng_state(tensor(self.torch_cpu_state_b64))
            if self.torch_cuda_states_b64:
                if not th.cuda.is_available() or (
                    len(self.torch_cuda_states_b64) != th.cuda.device_count()
                ):
                    raise RuntimeError(
                        "checkpoint CUDA RNG topology differs from this runtime"
                    )
                th.cuda.set_rng_state_all(
                    [tensor(value) for value in self.torch_cuda_states_b64]
                )

        def to_dict(self) -> dict[str, Any]:
            return {
                "schema": self.schema,
                "python": {
                    "version": self.python_version,
                    "internal_state": list(self.python_internal_state),
                    "gauss_next": self.python_gauss_next,
                },
                "numpy": {
                    "bit_generator": self.numpy_bit_generator,
                    "state": list(self.numpy_state),
                    "position": self.numpy_position,
                    "has_gauss": self.numpy_has_gauss,
                    "cached_gaussian": self.numpy_cached_gaussian,
                },
                "torch": {
                    "cpu_state_b64": self.torch_cpu_state_b64,
                    "cuda_states_b64": list(self.torch_cuda_states_b64),
                },
            }

        @classmethod
        def from_dict(cls, raw: Mapping[str, Any]) -> "LearnerRNGStateV1":
            if set(raw) != {"schema", "python", "numpy", "torch"}:
                raise ValueError("learner RNG state fields mismatch")
            if raw["schema"] != cls.schema:
                raise ValueError("learner RNG state schema mismatch")
            python_raw = raw["python"]
            numpy_raw = raw["numpy"]
            torch_raw = raw["torch"]
            if not isinstance(python_raw, Mapping) or set(python_raw) != {
                "version",
                "internal_state",
                "gauss_next",
            }:
                raise ValueError("Python RNG state fields mismatch")
            if not isinstance(numpy_raw, Mapping) or set(numpy_raw) != {
                "bit_generator",
                "state",
                "position",
                "has_gauss",
                "cached_gaussian",
            }:
                raise ValueError("NumPy RNG state fields mismatch")
            if not isinstance(torch_raw, Mapping) or set(torch_raw) != {
                "cpu_state_b64",
                "cuda_states_b64",
            }:
                raise ValueError("Torch RNG state fields mismatch")
            python_internal = python_raw["internal_state"]
            numpy_values = numpy_raw["state"]
            cuda_values = torch_raw["cuda_states_b64"]
            for name, value in (
                ("Python RNG internal state", python_internal),
                ("NumPy RNG state", numpy_values),
                ("Torch CUDA RNG states", cuda_values),
            ):
                if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                    raise TypeError(f"{name} must be an array")
            gauss_raw = python_raw["gauss_next"]
            return cls(
                python_version=_non_negative_int(
                    "Python RNG version", python_raw["version"]
                ),
                python_internal_state=tuple(
                    _non_negative_int(f"Python RNG state[{index}]", value)
                    for index, value in enumerate(python_internal)
                ),
                python_gauss_next=(
                    None
                    if gauss_raw is None
                    else _finite_float("Python RNG Gaussian cache", gauss_raw)
                ),
                numpy_bit_generator=str(numpy_raw["bit_generator"]),
                numpy_state=tuple(
                    _non_negative_int(f"NumPy RNG state[{index}]", value)
                    for index, value in enumerate(numpy_values)
                ),
                numpy_position=_non_negative_int(
                    "NumPy RNG position", numpy_raw["position"]
                ),
                numpy_has_gauss=_non_negative_int(
                    "NumPy RNG Gaussian flag", numpy_raw["has_gauss"]
                ),
                numpy_cached_gaussian=_finite_float(
                    "NumPy RNG Gaussian cache", numpy_raw["cached_gaussian"]
                ),
                torch_cpu_state_b64=str(torch_raw["cpu_state_b64"]),
                torch_cuda_states_b64=tuple(str(value) for value in cuda_values),
            )

    def _policy_parameter_digest(policy: Any) -> str:
        if not hasattr(policy, "state_dict"):
            raise TypeError("policy has no Torch state_dict")
        return _typed_state_sha256(
            "constrained-ppo-policy-parameter-state-v1",
            policy.state_dict(),
        )

    def _optimizer_state_digest(policy: Any) -> str:
        optimizer = getattr(policy, "optimizer", None)
        if optimizer is None or not hasattr(optimizer, "state_dict"):
            raise TypeError("policy has no optimizer state_dict")
        return _typed_state_sha256(
            "constrained-ppo-optimizer-state-v1",
            optimizer.state_dict(),
        )

    class FourHeadCostCriticV1(nn.Module):  # type: ignore[misc]
        """Four fully independent [64, 64] tanh scalar cost critics.

        There is intentionally no shared trainable trunk: changing one
        resource's target cannot alter another resource head through shared
        parameters.
        """

        def __init__(
            self,
            input_dim: int,
            *,
            hidden_layers: tuple[int, int] = (64, 64),
        ) -> None:
            super().__init__()
            if isinstance(input_dim, bool) or not isinstance(input_dim, Integral):
                raise TypeError("cost critic input_dim must be an integer")
            if input_dim <= 0:
                raise ValueError("cost critic input_dim must be positive")
            if tuple(hidden_layers) != (64, 64):
                raise ValueError("cost critic hidden layers must be [64, 64]")
            self.heads: nn.ModuleList = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(int(input_dim), 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1),
                    )
                    for _ in range(COST_CHANNEL_COUNT)
                ]
            )

        def forward(self, features: Any) -> Any:
            return th.cat([head(features) for head in self.heads], dim=-1)

    class ConstrainedActorCriticPolicyV1(ActorCriticPolicy):  # type: ignore[misc]
        """SB3 policy carrying the separate cost critic in its state dict."""

        def __init__(
            self,
            *args: Any,
            cost_network_seed: int,
            cost_hidden_layers: tuple[int, int] = (64, 64),
            **kwargs: Any,
        ) -> None:
            self.cost_network_seed = _non_negative_int(
                "cost_network_seed", cost_network_seed
            )
            self.cost_hidden_layers = tuple(cost_hidden_layers)
            super().__init__(*args, **kwargs)

        def _build(self, lr_schedule: Schedule) -> None:
            # This is intentionally first: actor and reward critic follow the
            # unchanged SB3 initialization and consume exactly the same RNG.
            super()._build(lr_schedule)
            cpu_rng_before = th.random.get_rng_state().clone()
            with th.random.fork_rng(devices=[], enabled=True):
                th.manual_seed(self.cost_network_seed)
                cost_critic = FourHeadCostCriticV1(
                    self.features_dim,
                    hidden_layers=cast(tuple[int, int], self.cost_hidden_layers),
                )
            if not th.equal(cpu_rng_before, th.random.get_rng_state()):
                raise RuntimeError("cost critic initialization leaked Torch RNG state")
            self.cost_critic = cost_critic.to(self.device)
            self.optimizer.add_param_group(
                {"params": tuple(self.cost_critic.parameters())}
            )

        def cost_values(self, observations: Any) -> Any:
            features = self.extract_features(observations)
            # The flattened Box extractor has no parameters. Detaching here is
            # still an explicit guarantee that cost loss cannot train a shared
            # feature extractor in a substituted policy architecture.
            if not isinstance(features, th.Tensor):
                raise TypeError("cost critic requires a tensor feature representation")
            return self.cost_critic(features.detach())

        def predict_cost_values(self, observations: Any) -> Any:
            return self.cost_values(observations)

    class CostRolloutBufferV1(RolloutBuffer):  # type: ignore[misc]
        """SB3 rollout storage extended by four raw/cost/value/GAE streams."""

        cost_gamma: float
        cost_gae_lambda: float
        raw_costs: npt.NDArray[np.float32]
        training_costs: npt.NDArray[np.float32]
        cost_values: npt.NDArray[np.float32]
        cost_advantages: npt.NDArray[np.float32]
        cost_returns: npt.NDArray[np.float32]
        cycle_settled: npt.NDArray[np.float32]
        generator_ready: bool

        def __init__(
            self,
            *args: Any,
            cost_gamma: float = 1.0,
            cost_gae_lambda: float = 0.95,
            **kwargs: Any,
        ) -> None:
            self.cost_gamma = cost_gamma
            self.cost_gae_lambda = cost_gae_lambda
            super().__init__(*args, **kwargs)

        def reset(self) -> None:
            super().reset()
            shape = (self.buffer_size, self.n_envs, COST_CHANNEL_COUNT)
            self.raw_costs = np.zeros(shape, dtype=np.float32)
            self.training_costs = np.zeros(shape, dtype=np.float32)
            self.cost_values = np.zeros(shape, dtype=np.float32)
            self.cost_advantages = np.zeros(shape, dtype=np.float32)
            self.cost_returns = np.zeros(shape, dtype=np.float32)
            self.cycle_settled = np.zeros(
                (self.buffer_size, self.n_envs), dtype=np.float32
            )

        def add(  # type: ignore[override]
            self,
            obs: Any,
            action: Any,
            reward: Any,
            episode_start: Any,
            value: Any,
            log_prob: Any,
            *,
            raw_cost: npt.NDArray[np.generic],
            training_cost: npt.NDArray[np.generic],
            cost_value: Any,
            cycle_settled: npt.NDArray[np.bool_],
        ) -> None:
            position = self.pos
            raw_array = np.asarray(raw_cost, dtype=np.float32)
            training_array = np.asarray(training_cost, dtype=np.float32)
            settled_array = np.asarray(cycle_settled)
            expected = (self.n_envs, COST_CHANNEL_COUNT)
            if raw_array.shape != expected or training_array.shape != expected:
                raise ValueError("rollout transition costs must have shape (n_envs, 4)")
            if settled_array.shape != (self.n_envs,) or settled_array.dtype != np.bool_:
                raise TypeError("cycle_settled must be a boolean vector")
            if not isinstance(cost_value, th.Tensor):
                raise TypeError("cost_value must be a tensor")
            cost_value_array = cost_value.detach().cpu().numpy()
            if cost_value_array.shape != expected:
                raise ValueError("cost critic output must have shape (n_envs, 4)")
            if any(
                not np.all(np.isfinite(array))
                for array in (raw_array, training_array, cost_value_array)
            ):
                raise ValueError("rollout cost data must be finite")
            if np.any(raw_array < 0.0) or np.any(training_array < 0.0):
                raise ValueError("rollout costs must be non-negative")
            super().add(obs, action, reward, episode_start, value, log_prob)
            self.raw_costs[position] = raw_array
            self.training_costs[position] = training_array
            self.cost_values[position] = cost_value_array
            self.cycle_settled[position] = settled_array.astype(np.float32)

        def compute_returns_and_advantage(  # type: ignore[override]
            self,
            last_values: Any,
            dones: npt.NDArray[np.bool_],
            *,
            last_cost_values: Any,
        ) -> None:
            super().compute_returns_and_advantage(last_values=last_values, dones=dones)
            if not isinstance(last_cost_values, th.Tensor):
                raise TypeError("last_cost_values must be a tensor")
            final_cost_values = last_cost_values.detach().cpu().numpy()
            advantages, returns = compute_vector_gae(
                self.training_costs,
                self.cost_values,
                self.episode_starts,
                final_cost_values,
                np.asarray(dones, dtype=np.bool_),
                gamma=self.cost_gamma,
                gae_lambda=self.cost_gae_lambda,
            )
            self.cost_advantages[:] = advantages.astype(np.float32)
            self.cost_returns[:] = returns.astype(np.float32)

        def get(  # type: ignore[override]
            self,
            batch_size: int | None = None,
        ) -> Generator[ConstrainedRolloutBufferSamplesV1, None, None]:
            if not self.full:
                raise RuntimeError("cannot sample an incomplete constrained rollout")
            indices = np.random.permutation(self.buffer_size * self.n_envs)
            if not self.generator_ready:
                tensor_names = (
                    "observations",
                    "actions",
                    "values",
                    "log_probs",
                    "advantages",
                    "returns",
                    "raw_costs",
                    "training_costs",
                    "cost_values",
                    "cost_advantages",
                    "cost_returns",
                    "cycle_settled",
                )
                for tensor_name in tensor_names:
                    self.__dict__[tensor_name] = self.swap_and_flatten(
                        self.__dict__[tensor_name]
                    )
                self.generator_ready = True
            if batch_size is None:
                batch_size = self.buffer_size * self.n_envs
            start = 0
            while start < self.buffer_size * self.n_envs:
                yield self._get_constrained_samples(indices[start : start + batch_size])
                start += batch_size

        def _get_constrained_samples(
            self,
            indices: npt.NDArray[np.integer[Any]],
        ) -> ConstrainedRolloutBufferSamplesV1:
            return ConstrainedRolloutBufferSamplesV1(
                observations=self.to_torch(self.observations[indices]),
                actions=self.to_torch(
                    self.actions[indices].astype(np.float32, copy=False)
                ),
                old_values=self.to_torch(self.values[indices].flatten()),
                old_log_prob=self.to_torch(self.log_probs[indices].flatten()),
                advantages=self.to_torch(self.advantages[indices].flatten()),
                returns=self.to_torch(self.returns[indices].flatten()),
                old_cost_values=self.to_torch(self.cost_values[indices]),
                cost_advantages=self.to_torch(self.cost_advantages[indices]),
                cost_returns=self.to_torch(self.cost_returns[indices]),
                raw_costs=self.to_torch(self.raw_costs[indices]),
                training_costs=self.to_torch(self.training_costs[indices]),
                cycle_settled=self.to_torch(self.cycle_settled[indices].flatten()),
            )

    @dataclass(slots=True)
    class _PendingRolloutV1:
        started_at: float
        transition_count: int
        completed: tuple[CompletedEpisodeCostV1, ...]
        complete_cycle_count: int
        reward_return: float
        raw_cost: FloatVector4
        training_cost: FloatVector4
        mean_completed_episode_cost: FloatVector4 | None

    def _torch_normalize_advantage(values: Any) -> Any:
        flat = values.flatten()
        if not bool(th.isfinite(flat).all()):
            raise ValueError("optimizer advantages must be finite")
        if flat.numel() <= 1 or bool(th.all(flat == flat[0])):
            return th.zeros_like(flat)
        return (flat - flat.mean()) / (flat.std(correction=1) + 1e-8)

    class ConstrainedPPO(_SB3PPO):  # type: ignore[misc]
        """Controller-neutral PPO with auditable, optional PI constraints.

        The scalar family deliberately uses SB3's stock policy, rollout buffer,
        optimizer, and :meth:`PPO.train` implementation.  Constraint evidence is
        observed beside that path and cannot enter its gradients.  PI/I families
        select the separate cost policy and buffer explicitly.
        """

        policy_aliases = {"MlpPolicy": ActorCriticPolicy}
        n_envs: int
        policy: ActorCriticPolicy
        rollout_buffer: RolloutBuffer
        _last_obs: Any
        _last_episode_starts: npt.NDArray[np.bool_] | None

        def __init__(
            self,
            policy: str | type[ActorCriticPolicy],
            env: GymEnv | str,
            learning_rate: float | Schedule = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 256,
            n_epochs: int = 10,
            gamma: float = 0.995,
            gae_lambda: float = 0.95,
            clip_range: float | Schedule = 0.2,
            clip_range_vf: None | float | Schedule = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: float | None = None,
            stats_window_size: int = 100,
            tensorboard_log: str | None = None,
            policy_kwargs: dict[str, Any] | None = None,
            verbose: int = 0,
            seed: int | None = None,
            device: Any = "cpu",
            constraint_settings: ConstrainedPPOSettingsV1 | None = None,
            _init_setup_model: bool = True,
        ) -> None:
            # The auditable beta contract admits constant schedules only.  A
            # callable cannot be serialized as exact, language-neutral state.
            learning_rate_value = _finite_float("learning_rate", learning_rate)
            clip_range_value = _finite_float("clip_range", clip_range)
            clip_range_vf_value = (
                None
                if clip_range_vf is None
                else _finite_float("clip_range_vf", clip_range_vf)
            )
            if learning_rate_value <= 0.0:
                raise ValueError("learning_rate must be positive")
            if clip_range_value <= 0.0:
                raise ValueError("clip_range must be positive")
            if clip_range_vf_value is not None and clip_range_vf_value <= 0.0:
                raise ValueError("clip_range_vf must be positive")
            optimizer_seed = 0 if seed is None else seed
            settings = constraint_settings or ConstrainedPPOSettingsV1.for_algorithm(
                PI_LAGRANGIAN_ALGORITHM_ID,
                optimizer_seed=optimizer_seed,
            )
            settings.validate()
            if seed is not None and seed != settings.optimizer_seed:
                raise ValueError("SB3 seed must equal constrained optimizer_seed")
            loading_shell = not _init_setup_model and constraint_settings is None
            expected_policy_class: type[Any] = (
                ConstrainedActorCriticPolicyV1
                if settings.uses_constraints
                else ActorCriticPolicy
            )
            if isinstance(policy, str):
                if policy != "MlpPolicy":
                    raise ValueError("the frozen learner requires MlpPolicy")
                selected_policy_class = expected_policy_class
            else:
                if policy is not expected_policy_class and not (
                    loading_shell
                    and policy in (ActorCriticPolicy, ConstrainedActorCriticPolicyV1)
                ):
                    raise ValueError("learner algorithm and policy class disagree")
                selected_policy_class = policy

            expected_policy_kwargs: dict[str, Any] = {
                "net_arch": {"pi": [64, 64], "vf": [64, 64]},
                "activation_fn": nn.Tanh,
            }
            if settings.uses_constraints:
                expected_policy_kwargs.update(
                    {
                        "cost_network_seed": settings.cost_network_seed,
                        "cost_hidden_layers": settings.cost_hidden_layers,
                    }
                )
            supplied_policy_kwargs = dict(policy_kwargs or {})
            unknown_policy_kwargs = set(supplied_policy_kwargs) - set(
                expected_policy_kwargs
            )
            if unknown_policy_kwargs:
                raise ValueError(
                    "policy_kwargs contain fields outside the frozen contract: "
                    + ", ".join(sorted(unknown_policy_kwargs))
                )
            for name, supplied in supplied_policy_kwargs.items():
                if supplied != expected_policy_kwargs[name]:
                    raise ValueError(
                        f"policy_kwargs[{name!r}] violates the frozen contract"
                    )
            self.constraint_settings = settings
            self.algorithm_id = settings.algorithm_id
            self._multiplier_controller = PILagrangeControllerV1(settings)
            self._episode_cost_accumulator: EpisodeCostAccumulatorV1 | None = None
            self._rollout_ledgers: tuple[RolloutLedgerV1, ...] = ()
            self._pending_rollout: _PendingRolloutV1 | None = None
            self._aborted_rollout: AbortedRolloutV1 | None = None
            self._unledgered_transition_count = 0
            self._external_episode_state_pin: ExternalEpisodeStatePinV1 | None = None
            self._embedded_constraint_state_schema: str | None = None
            self._embedded_constraint_state_payload_json: str | None = None
            self._embedded_constraint_state_sha256: str | None = None
            self._learner_rng_state: LearnerRNGStateV1 | None = None
            self._resume_continuation_receipt: ResumeContinuationReceiptV1 | None = None
            self._resume_continuity_required = False
            self._loaded_force_reset: bool | None = None
            self._loaded_policy_artifact_sha256: str | None = None
            rollout_buffer_class: type[Any] = (
                CostRolloutBufferV1 if settings.uses_constraints else RolloutBuffer
            )
            rollout_buffer_kwargs = (
                {
                    "cost_gamma": settings.cost_gamma,
                    "cost_gae_lambda": settings.cost_gae_lambda,
                }
                if settings.uses_constraints
                else None
            )
            super().__init__(
                selected_policy_class,
                env,
                learning_rate=learning_rate_value,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range_value,
                clip_range_vf=clip_range_vf_value,
                normalize_advantage=normalize_advantage,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                use_sde=use_sde,
                sde_sample_freq=sde_sample_freq,
                rollout_buffer_class=rollout_buffer_class,
                rollout_buffer_kwargs=rollout_buffer_kwargs,
                target_kl=target_kl,
                stats_window_size=stats_window_size,
                tensorboard_log=tensorboard_log,
                policy_kwargs=expected_policy_kwargs,
                verbose=verbose,
                seed=settings.optimizer_seed,
                device=device,
                _init_setup_model=_init_setup_model,
            )

        def _setup_model(self) -> None:
            if (self.n_steps * self.n_envs) % self.batch_size:
                raise ValueError(
                    "frozen PPO rollout size must divide into exact minibatches"
                )
            super()._setup_model()
            expected_buffer: type[Any] = (
                CostRolloutBufferV1
                if self.constraint_settings.uses_constraints
                else RolloutBuffer
            )
            expected_policy: type[Any] = (
                ConstrainedActorCriticPolicyV1
                if self.constraint_settings.uses_constraints
                else ActorCriticPolicy
            )
            if type(self.rollout_buffer) is not expected_buffer:
                raise RuntimeError("algorithm family lost its exact rollout buffer")
            if type(self.policy) is not expected_policy:
                raise RuntimeError("algorithm family lost its exact policy class")
            if self._episode_cost_accumulator is None:
                self._episode_cost_accumulator = EpisodeCostAccumulatorV1(self.n_envs)
            elif self._episode_cost_accumulator.n_envs != self.n_envs:
                raise ValueError("restored episode accumulator n_envs mismatch")

        @staticmethod
        def _constant_schedule_value(name: str, schedule: object) -> float:
            if not callable(schedule):
                return _finite_float(name, schedule)
            samples = tuple(
                _finite_float(f"{name}({progress})", schedule(progress))
                for progress in (0.0, 0.5, 1.0)
            )
            if samples[1:] != samples[:1] * 2:
                raise ValueError(f"{name} must be a constant schedule")
            return samples[0]

        @staticmethod
        def _box_space_record(name: str, space: object) -> dict[str, Any]:
            if type(space) is not spaces.Box:
                raise TypeError(f"{name} must be an exact Gymnasium Box")
            box = cast(Any, space)
            return {
                "class": _qualified_type(box),
                "shape": list(box.shape),
                "dtype": str(box.dtype),
                "low_sha256": _array_sha256(f"{name}-low-v1", box.low),
                "high_sha256": _array_sha256(f"{name}-high-v1", box.high),
            }

        def _live_policy_architecture(self) -> dict[str, Any]:
            expected_policy: type[Any] = (
                ConstrainedActorCriticPolicyV1
                if self.constraint_settings.uses_constraints
                else ActorCriticPolicy
            )
            if type(self.policy) is not expected_policy:
                raise ValueError("loaded policy class differs from algorithm family")
            expected_architecture = {"pi": [64, 64], "vf": [64, 64]}
            if self.policy.net_arch != expected_architecture:
                raise ValueError("policy network architecture differs from [64, 64]")
            if self.policy.activation_fn is not nn.Tanh:
                raise ValueError("policy activation must be exact Torch Tanh")
            for name, actual, expected in (
                ("ortho_init", self.policy.ortho_init, True),
                (
                    "share_features_extractor",
                    self.policy.share_features_extractor,
                    True,
                ),
                ("normalize_images", self.policy.normalize_images, True),
                ("squash_output", self.policy.squash_output, False),
            ):
                if actual is not expected:
                    raise ValueError(f"policy {name} differs from the frozen contract")
            if _finite_float("policy log_std_init", self.policy.log_std_init) != 0.0:
                raise ValueError("policy log_std_init must be zero")
            if _qualified_type(self.policy.features_extractor) != (
                "stable_baselines3.common.torch_layers.FlattenExtractor"
            ):
                raise ValueError("policy must use the stock FlattenExtractor")
            observation_shape = tuple(self.observation_space.shape or ())
            expected_features = math.prod(observation_shape)
            if self.policy.features_dim != expected_features:
                raise ValueError("policy feature dimension differs from observation")
            feature_children = dict(self.policy.features_extractor.named_children())
            if (
                set(feature_children) != {"flatten"}
                or type(feature_children["flatten"]) is not nn.Flatten
            ):
                raise ValueError("feature extractor module graph was substituted")
            flatten = feature_children["flatten"]
            if flatten.start_dim != 1 or flatten.end_dim != -1:
                raise ValueError("feature extractor flatten dimensions changed")

            mlp_extractor = self.policy.mlp_extractor
            if _qualified_type(mlp_extractor) != (
                "stable_baselines3.common.torch_layers.MlpExtractor"
            ):
                raise ValueError("policy MLP extractor class was substituted")
            if (
                int(mlp_extractor.latent_dim_pi) != 64
                or int(mlp_extractor.latent_dim_vf) != 64
            ):
                raise ValueError("policy MLP latent dimensions changed")
            if set(dict(mlp_extractor.named_children())) != {
                "policy_net",
                "value_net",
            }:
                raise ValueError("policy MLP extractor module graph changed")

            def validate_hidden_network(name: str, network: object) -> None:
                if type(network) is not nn.Sequential:
                    raise ValueError(f"{name} must be an exact Sequential")
                layers = tuple(cast(Any, network))
                if tuple(type(layer) for layer in layers) != (
                    nn.Linear,
                    nn.Tanh,
                    nn.Linear,
                    nn.Tanh,
                ):
                    raise ValueError(f"{name} module topology changed")
                dimensions = [
                    (int(layer.in_features), int(layer.out_features))
                    for layer in (layers[0], layers[2])
                ]
                if dimensions != [(expected_features, 64), (64, 64)] or any(
                    layer.bias is None for layer in (layers[0], layers[2])
                ):
                    raise ValueError(f"{name} linear dimensions changed")

            validate_hidden_network("actor hidden network", mlp_extractor.policy_net)
            validate_hidden_network("value hidden network", mlp_extractor.value_net)
            action_shape = tuple(self.action_space.shape or ())
            action_dimension = math.prod(action_shape)
            if (
                type(self.policy.action_net) is not nn.Linear
                or int(self.policy.action_net.in_features) != 64
                or int(self.policy.action_net.out_features) != action_dimension
                or self.policy.action_net.bias is None
            ):
                raise ValueError("policy action head topology changed")
            if (
                type(self.policy.value_net) is not nn.Linear
                or int(self.policy.value_net.in_features) != 64
                or int(self.policy.value_net.out_features) != 1
                or self.policy.value_net.bias is None
            ):
                raise ValueError("policy reward-value head topology changed")
            if (
                _qualified_type(self.policy.action_dist)
                != ("stable_baselines3.common.distributions.DiagGaussianDistribution")
                or int(self.policy.action_dist.action_dim) != action_dimension
            ):
                raise ValueError("policy action distribution topology changed")
            if (
                not isinstance(self.policy.log_std, nn.Parameter)
                or tuple(self.policy.log_std.shape) != (action_dimension,)
                or self.policy.log_std.dtype != th.float32
            ):
                raise ValueError("policy log-standard-deviation parameter changed")
            if type(self.policy.optimizer) is not th.optim.Adam:
                raise ValueError("policy optimizer must be exact Torch Adam")
            if self.policy.optimizer_class is not th.optim.Adam:
                raise ValueError("policy optimizer class declaration was substituted")
            if self.policy.optimizer_kwargs != {"eps": 1e-5}:
                raise ValueError("policy optimizer kwargs differ from SB3 PPO")

            policy_parameters = tuple(self.policy.parameters())
            grouped_parameters = tuple(
                parameter
                for group in self.policy.optimizer.param_groups
                for parameter in group["params"]
            )
            if len(grouped_parameters) != len(policy_parameters) or (
                {id(value) for value in grouped_parameters}
                != {id(value) for value in policy_parameters}
            ):
                raise ValueError("optimizer parameter ownership differs from policy")
            if len({id(value) for value in grouped_parameters}) != len(
                grouped_parameters
            ):
                raise ValueError("optimizer parameter groups overlap")

            expected_parameter_shapes: dict[str, tuple[int, ...]] = {
                "log_std": (action_dimension,),
                "mlp_extractor.policy_net.0.weight": (64, expected_features),
                "mlp_extractor.policy_net.0.bias": (64,),
                "mlp_extractor.policy_net.2.weight": (64, 64),
                "mlp_extractor.policy_net.2.bias": (64,),
                "mlp_extractor.value_net.0.weight": (64, expected_features),
                "mlp_extractor.value_net.0.bias": (64,),
                "mlp_extractor.value_net.2.weight": (64, 64),
                "mlp_extractor.value_net.2.bias": (64,),
                "action_net.weight": (action_dimension, 64),
                "action_net.bias": (action_dimension,),
                "value_net.weight": (1, 64),
                "value_net.bias": (1,),
            }
            cost_head_specs: list[list[dict[str, Any]]] = []
            if self.constraint_settings.uses_constraints:
                constrained_policy = cast(ConstrainedActorCriticPolicyV1, self.policy)
                if constrained_policy.cost_network_seed != (
                    self.constraint_settings.cost_network_seed
                ) or constrained_policy.cost_hidden_layers != (64, 64):
                    raise ValueError("cost critic declaration differs from settings")
                if type(constrained_policy.cost_critic) is not FourHeadCostCriticV1:
                    raise ValueError("cost critic class was substituted")
                if len(constrained_policy.cost_critic.heads) != COST_CHANNEL_COUNT:
                    raise ValueError("cost critic must contain exactly four heads")
                expected_layer_types = (
                    nn.Linear,
                    nn.Tanh,
                    nn.Linear,
                    nn.Tanh,
                    nn.Linear,
                )
                cost_parameter_ids: set[int] = set()
                for index, head in enumerate(constrained_policy.cost_critic.heads):
                    if (
                        type(head) is not nn.Sequential
                        or tuple(type(layer) for layer in head) != expected_layer_types
                    ):
                        raise ValueError(
                            f"cost critic head {index} architecture changed"
                        )
                    linears = [layer for layer in head if type(layer) is nn.Linear]
                    dimensions = [
                        (int(layer.in_features), int(layer.out_features))
                        for layer in linears
                    ]
                    if dimensions != [
                        (self.policy.features_dim, 64),
                        (64, 64),
                        (64, 1),
                    ]:
                        raise ValueError(f"cost critic head {index} dimensions changed")
                    head_specs = []
                    for name, parameter in head.named_parameters():
                        parameter_id = id(parameter)
                        if parameter_id in cost_parameter_ids:
                            raise ValueError(
                                "cost critic heads share trainable parameters"
                            )
                        cost_parameter_ids.add(parameter_id)
                        expected_parameter_shapes[
                            f"cost_critic.heads.{index}.{name}"
                        ] = tuple(parameter.shape)
                        head_specs.append(
                            {
                                "name": name,
                                "shape": list(parameter.shape),
                                "dtype": str(parameter.dtype),
                            }
                        )
                    cost_head_specs.append(head_specs)
                if len(self.policy.optimizer.param_groups) != 2:
                    raise ValueError(
                        "constrained optimizer must have two parameter groups"
                    )
                if {
                    id(parameter)
                    for parameter in self.policy.optimizer.param_groups[1]["params"]
                } != cost_parameter_ids:
                    raise ValueError(
                        "cost optimizer group does not own exactly cost heads"
                    )
                if any(
                    id(parameter) in cost_parameter_ids
                    for parameter in self.policy.optimizer.param_groups[0]["params"]
                ):
                    raise ValueError(
                        "actor/reward optimizer group contains cost weights"
                    )
            elif len(self.policy.optimizer.param_groups) != 1:
                raise ValueError("scalar PPO optimizer must have one parameter group")

            policy_state = self.policy.state_dict()
            if set(policy_state) != set(expected_parameter_shapes):
                raise ValueError("policy parameter set differs from module contract")
            for name, expected_shape in expected_parameter_shapes.items():
                tensor = policy_state[name]
                if tuple(tensor.shape) != expected_shape or tensor.dtype != th.float32:
                    raise ValueError(
                        f"policy parameter specification changed for {name}"
                    )

            expected_module_classes = {
                "": _qualified_type(expected_policy),
                "features_extractor": (
                    "stable_baselines3.common.torch_layers.FlattenExtractor"
                ),
                "features_extractor.flatten": _qualified_type(nn.Flatten),
                "mlp_extractor": ("stable_baselines3.common.torch_layers.MlpExtractor"),
                "mlp_extractor.policy_net": _qualified_type(nn.Sequential),
                "mlp_extractor.policy_net.0": _qualified_type(nn.Linear),
                "mlp_extractor.policy_net.1": _qualified_type(nn.Tanh),
                "mlp_extractor.policy_net.2": _qualified_type(nn.Linear),
                "mlp_extractor.policy_net.3": _qualified_type(nn.Tanh),
                "mlp_extractor.value_net": _qualified_type(nn.Sequential),
                "mlp_extractor.value_net.0": _qualified_type(nn.Linear),
                "mlp_extractor.value_net.1": _qualified_type(nn.Tanh),
                "mlp_extractor.value_net.2": _qualified_type(nn.Linear),
                "mlp_extractor.value_net.3": _qualified_type(nn.Tanh),
                "action_net": _qualified_type(nn.Linear),
                "value_net": _qualified_type(nn.Linear),
            }
            if self.constraint_settings.uses_constraints:
                expected_module_classes.update(
                    {
                        "cost_critic": _qualified_type(FourHeadCostCriticV1),
                        "cost_critic.heads": _qualified_type(nn.ModuleList),
                    }
                )
                for head_index in range(COST_CHANNEL_COUNT):
                    prefix = f"cost_critic.heads.{head_index}"
                    expected_module_classes[prefix] = _qualified_type(nn.Sequential)
                    for layer_index, layer_type in enumerate(
                        (nn.Linear, nn.Tanh, nn.Linear, nn.Tanh, nn.Linear)
                    ):
                        expected_module_classes[f"{prefix}.{layer_index}"] = (
                            _qualified_type(layer_type)
                        )
            actual_modules = dict(self.policy.named_modules())
            if set(actual_modules) != set(expected_module_classes) or any(
                _qualified_type(actual_modules[name]) != expected_class
                for name, expected_class in expected_module_classes.items()
            ):
                raise ValueError("policy live module topology changed")
            module_topology = [
                {"path": name, "class": expected_module_classes[name]}
                for name in sorted(expected_module_classes)
            ]
            parameter_specification = [
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }
                for name, tensor in sorted(self.policy.state_dict().items())
            ]
            optimizer_defaults = _canonical_object_copy(
                "optimizer defaults", self.policy.optimizer.defaults
            )
            optimizer_groups = [
                _canonical_object_copy(
                    f"optimizer group {index}",
                    {name: value for name, value in group.items() if name != "params"},
                )
                for index, group in enumerate(self.policy.optimizer.param_groups)
            ]
            return {
                "schema": POLICY_ARCHITECTURE_SCHEMA,
                "policy_class": _qualified_type(self.policy),
                "features_extractor_class": _qualified_type(
                    self.policy.features_extractor
                ),
                "features_dim": int(self.policy.features_dim),
                "net_arch": expected_architecture,
                "activation_class": _qualified_type(self.policy.activation_fn),
                "ortho_init": True,
                "share_features_extractor": True,
                "normalize_images": True,
                "squash_output": False,
                "log_std_init": 0.0,
                "action_distribution_class": _qualified_type(self.policy.action_dist),
                "module_topology": module_topology,
                "optimizer_class": _qualified_type(self.policy.optimizer),
                "optimizer_defaults": optimizer_defaults,
                "optimizer_groups": optimizer_groups,
                "parameter_specification": parameter_specification,
                "cost_head_parameter_specification": cost_head_specs,
            }

        def _live_ppo_configuration(self) -> dict[str, Any]:
            n_steps = _positive_int("PPO n_steps", self.n_steps)
            batch_size = _positive_int("PPO batch_size", self.batch_size)
            n_epochs = _positive_int("PPO n_epochs", self.n_epochs)
            n_envs = _positive_int("PPO n_envs", self.n_envs)
            if (n_steps * n_envs) % batch_size:
                raise ValueError("PPO rollout size must divide into exact minibatches")
            gamma = _non_negative_float("PPO gamma", self.gamma)
            gae_lambda = _non_negative_float("PPO gae_lambda", self.gae_lambda)
            if gamma > 1.0 or gae_lambda > 1.0:
                raise ValueError("PPO gamma and gae_lambda must not exceed one")
            learning_rate = _finite_float("PPO learning_rate", self.learning_rate)
            if learning_rate <= 0.0:
                raise ValueError("PPO learning_rate must be positive")
            if self._constant_schedule_value("PPO lr_schedule", self.lr_schedule) != (
                learning_rate
            ):
                raise ValueError("PPO learning-rate schedule differs from its setting")
            clip_range = self._constant_schedule_value(
                "PPO clip_range", self.clip_range
            )
            if clip_range <= 0.0:
                raise ValueError("PPO clip_range must be positive")
            clip_range_vf = (
                None
                if self.clip_range_vf is None
                else self._constant_schedule_value(
                    "PPO clip_range_vf", self.clip_range_vf
                )
            )
            if clip_range_vf is not None and clip_range_vf <= 0.0:
                raise ValueError("PPO clip_range_vf must be positive")
            if not isinstance(self.seed, Integral) or isinstance(self.seed, bool):
                raise TypeError("PPO optimizer seed must be an integer")
            if int(self.seed) != self.constraint_settings.optimizer_seed:
                raise ValueError("PPO optimizer seed differs from constrained settings")
            expected_buffer: type[Any] = (
                CostRolloutBufferV1
                if self.constraint_settings.uses_constraints
                else RolloutBuffer
            )
            if type(self.rollout_buffer) is not expected_buffer:
                raise ValueError("loaded rollout buffer differs from algorithm family")
            if not isinstance(self.normalize_advantage, bool):
                raise TypeError("PPO normalize_advantage must be a bool")
            if not isinstance(self.use_sde, bool):
                raise TypeError("PPO use_sde must be a bool")
            target_kl = (
                None
                if self.target_kl is None
                else _non_negative_float("PPO target_kl", self.target_kl)
            )
            return {
                "schema": PPO_CONFIGURATION_SCHEMA,
                "learning_rate": learning_rate,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "n_envs": n_envs,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_range": clip_range,
                "clip_range_vf": clip_range_vf,
                "normalize_advantage": self.normalize_advantage,
                "ent_coef": _non_negative_float("PPO ent_coef", self.ent_coef),
                "vf_coef": _non_negative_float("PPO vf_coef", self.vf_coef),
                "max_grad_norm": _non_negative_float(
                    "PPO max_grad_norm", self.max_grad_norm
                ),
                "use_sde": self.use_sde,
                "sde_sample_freq": int(self.sde_sample_freq),
                "target_kl": target_kl,
                "stats_window_size": _positive_int(
                    "PPO stats_window_size", self._stats_window_size
                ),
                "seed": int(self.seed),
                "device": str(self.device),
                "rollout_buffer_class": _qualified_type(self.rollout_buffer),
                "observation_space": self._box_space_record(
                    "PPO observation space", self.observation_space
                ),
                "action_space": self._box_space_record(
                    "PPO action space", self.action_space
                ),
                "policy_architecture": self._live_policy_architecture(),
            }

        def set_random_seed(self, seed: int | None = None) -> None:
            """Seed optimizer randomness without ever assigning episode seeds."""

            if seed is None:
                return
            sb3_set_random_seed(
                seed,
                using_cuda=self.device.type == th.device("cuda").type,
            )
            self.action_space.seed(seed)

        def _validate_rollout_buffer_finite(
            self,
            rollout_buffer: RolloutBuffer,
        ) -> None:
            if not rollout_buffer.full:
                raise RuntimeError("optimizer requires a complete rollout buffer")
            for field in (
                "observations",
                "actions",
                "rewards",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "episode_starts",
            ):
                _require_finite_array(
                    f"rollout buffer {field}", getattr(rollout_buffer, field)
                )
            if type(rollout_buffer) is CostRolloutBufferV1:
                cost_buffer = cast(CostRolloutBufferV1, rollout_buffer)
                for field in (
                    "raw_costs",
                    "training_costs",
                    "cost_values",
                    "cost_advantages",
                    "cost_returns",
                    "cycle_settled",
                ):
                    _require_finite_array(
                        f"rollout buffer {field}", getattr(cost_buffer, field)
                    )

        def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
        ) -> bool:
            if self._last_obs is None:
                raise RuntimeError("no previous observation was provided")
            if self._episode_cost_accumulator is None:
                raise RuntimeError("episode cost accumulator is not initialized")
            if self._last_episode_starts is None:
                raise RuntimeError("episode-start state is not initialized")
            episode_accumulator = self._episode_cost_accumulator
            last_episode_starts = self._last_episode_starts
            last_obs = self._last_obs
            if self._pending_rollout is not None:
                raise RuntimeError("previous rollout was not optimized")
            if self._aborted_rollout is not None:
                raise RuntimeError("an aborted rollout permanently blocks training")
            if self._unledgered_transition_count:
                raise RuntimeError(
                    "consumed transitions are not represented by a ledger"
                )
            if self._resume_continuity_required:
                raise RuntimeError(
                    "loaded checkpoint requires external-state admission"
                )
            constrained = self.constraint_settings.uses_constraints
            if constrained:
                if type(rollout_buffer) is not CostRolloutBufferV1:
                    raise TypeError("constrained PPO requires CostRolloutBufferV1")
                if type(self.policy) is not ConstrainedActorCriticPolicyV1:
                    raise TypeError("constrained PPO requires its cost-aware policy")
                cost_policy: ConstrainedActorCriticPolicyV1 | None = cast(
                    ConstrainedActorCriticPolicyV1,
                    self.policy,
                )
            elif type(rollout_buffer) is not RolloutBuffer:
                raise TypeError("scalar PPO requires the stock SB3 RolloutBuffer")
            else:
                cost_policy = None
            self._live_policy_architecture()
            # Any previously pinned runtime snapshot becomes stale as soon as a
            # new environment transition may be consumed.
            self._external_episode_state_pin = None
            self._learner_rng_state = None
            self._embedded_constraint_state_schema = None
            self._embedded_constraint_state_payload_json = None
            self._embedded_constraint_state_sha256 = None
            self.policy.set_training_mode(False)
            rollout_buffer.reset()
            if self.use_sde:
                self.policy.reset_noise(env.num_envs)
            callback.on_rollout_start()
            started_at = time.perf_counter()
            n_steps = 0
            completed: list[CompletedEpisodeCostV1] = []
            reward_return = 0.0
            raw_total = np.zeros(COST_CHANNEL_COUNT, dtype=np.float64)
            training_total = np.zeros(COST_CHANNEL_COUNT, dtype=np.float64)
            complete_cycle_count = 0
            dones = np.zeros(env.num_envs, dtype=np.bool_)
            new_obs: Any = self._last_obs

            while n_steps < n_rollout_steps:
                if (
                    self.use_sde
                    and self.sde_sample_freq > 0
                    and n_steps % self.sde_sample_freq == 0
                ):
                    self.policy.reset_noise(env.num_envs)
                _require_finite_array("rollout observation", last_obs)
                with th.no_grad():
                    obs_tensor = obs_as_tensor(last_obs, self.device)
                    actions, values, log_probs = self.policy(obs_tensor)
                    cost_values = (
                        cost_policy.predict_cost_values(obs_tensor)
                        if cost_policy is not None
                        else None
                    )
                _require_finite_tensor("policy actions", actions)
                _require_finite_tensor("policy values", values)
                _require_finite_tensor("policy log probabilities", log_probs)
                if cost_values is not None:
                    _require_finite_tensor("policy cost values", cost_values)
                actions_array = actions.cpu().numpy()
                clipped_actions = actions_array
                if isinstance(self.action_space, spaces.Box):
                    if self.policy.squash_output:
                        clipped_actions = self.policy.unscale_action(clipped_actions)
                    else:
                        clipped_actions = np.clip(
                            actions_array,
                            self.action_space.low,
                            self.action_space.high,
                        )
                _require_finite_array("environment actions", clipped_actions)
                new_obs, rewards, dones, infos = env.step(clipped_actions)
                self.num_timesteps += env.num_envs
                self._unledgered_transition_count += env.num_envs
                try:
                    _require_finite_array("next rollout observation", new_obs)
                    _require_finite_array("rollout rewards", rewards)
                    done_array = np.asarray(dones)
                    if (
                        done_array.shape != (env.num_envs,)
                        or done_array.dtype != np.bool_
                    ):
                        raise TypeError("environment dones must be a boolean vector")
                    if any(
                        bool(info.get("TimeLimit.truncated", False)) for info in infos
                    ):
                        raise RuntimeError(
                            "registered beta transitions cannot use time-limit truncation"
                        )
                    transitions = tuple(
                        extract_constraint_transition(
                            info,
                            key=self.constraint_settings.transition_info_key,
                        )
                        for info in infos
                    )
                    raw_costs64 = np.asarray(
                        [transition.raw_cost for transition in transitions],
                        dtype=np.float64,
                    )
                    training_costs64 = np.asarray(
                        [transition.training_cost for transition in transitions],
                        dtype=np.float64,
                    )
                    raw_costs = raw_costs64.astype(np.float32)
                    training_costs = training_costs64.astype(np.float32)
                    settled = np.asarray(
                        [transition.cycle_settled for transition in transitions],
                        dtype=np.bool_,
                    )
                    # Extraction and accumulation deliberately precede use of
                    # VecEnv's auto-reset observation.
                    completed.extend(episode_accumulator.observe(transitions, dones))
                    raw_total += raw_costs64.sum(axis=0, dtype=np.float64)
                    training_total += training_costs64.sum(axis=0, dtype=np.float64)
                    complete_cycle_count += int(settled.sum())
                    reward_return += math.fsum(float(value) for value in rewards)

                    callback.update_locals(locals())
                    if not callback.on_step():
                        self._mark_aborted_rollout("callback-returned-false")
                        return False
                    self._update_info_buffer(infos, dones)
                    n_steps += 1
                    if isinstance(self.action_space, spaces.Discrete):
                        actions_array = actions_array.reshape(-1, 1)

                    # A beta terminal is a true terminal at decision 672, so no
                    # time-limit bootstrap is admitted.  This loop is retained
                    # only for an explicit invariant check on future wrappers.
                    for index, done in enumerate(dones):
                        if (
                            done
                            and infos[index].get("terminal_observation") is not None
                            and infos[index].get("TimeLimit.truncated", False)
                        ):
                            raise RuntimeError("beta terminal was mislabeled truncated")

                    if constrained:
                        cast(CostRolloutBufferV1, rollout_buffer).add(
                            last_obs,
                            actions_array,
                            rewards,
                            last_episode_starts,
                            values,
                            log_probs,
                            raw_cost=raw_costs,
                            training_cost=training_costs,
                            cost_value=cost_values,
                            cycle_settled=settled,
                        )
                    else:
                        # This is exactly the stock SB3 buffer call.  None of the
                        # observed cost evidence can enter scalar optimization.
                        rollout_buffer.add(
                            cast(npt.NDArray[Any], last_obs),
                            actions_array,
                            rewards,
                            last_episode_starts,
                            values,
                            log_probs,
                        )
                    self._last_obs = new_obs
                    last_obs = new_obs
                    self._last_episode_starts = dones
                    last_episode_starts = dones
                except Exception:
                    self._mark_aborted_rollout("rollout-exception")
                    raise

            try:
                with th.no_grad():
                    _require_finite_array("final rollout observation", new_obs)
                    final_obs_tensor = obs_as_tensor(new_obs, self.device)
                    last_values = self.policy.predict_values(final_obs_tensor)
                    last_cost_values = (
                        cost_policy.predict_cost_values(final_obs_tensor)
                        if cost_policy is not None
                        else None
                    )
                _require_finite_tensor("final policy values", last_values)
                if last_cost_values is not None:
                    _require_finite_tensor("final policy cost values", last_cost_values)
                if constrained:
                    cast(
                        CostRolloutBufferV1, rollout_buffer
                    ).compute_returns_and_advantage(
                        last_values=last_values,
                        dones=dones,
                        last_cost_values=last_cost_values,
                    )
                else:
                    rollout_buffer.compute_returns_and_advantage(
                        last_values=last_values,
                        dones=dones,
                    )
                self._validate_rollout_buffer_finite(rollout_buffer)
            except Exception:
                self._mark_aborted_rollout("rollout-exception")
                raise
            completed_costs = [item.training_cost for item in completed]
            state = self._multiplier_controller.update(completed_costs)
            mean_completed = (
                None
                if not completed_costs
                else cast(
                    FloatVector4,
                    tuple(
                        float(value)
                        for value in np.asarray(completed_costs, dtype=np.float64).mean(
                            axis=0
                        )
                    ),
                )
            )
            self._pending_rollout = _PendingRolloutV1(
                started_at=started_at,
                transition_count=n_steps * env.num_envs,
                completed=tuple(completed),
                complete_cycle_count=complete_cycle_count,
                reward_return=reward_return,
                raw_cost=cast(FloatVector4, tuple(float(x) for x in raw_total)),
                training_cost=cast(
                    FloatVector4, tuple(float(x) for x in training_total)
                ),
                mean_completed_episode_cost=mean_completed,
            )
            for index, resource in enumerate(RESOURCE_ORDER):
                self.logger.record(
                    f"constraint/multiplier_{resource}", state.multipliers[index]
                )
                self.logger.record(
                    f"constraint/rollout_cost_{resource}", training_total[index]
                )
            self.logger.record("constraint/completed_episodes", len(completed))
            self.logger.record("constraint/complete_cycles", complete_cycle_count)
            callback.update_locals(locals())
            callback.on_rollout_end()
            return True

        def _mark_aborted_rollout(self, reason: str) -> None:
            if self._aborted_rollout is not None:
                return
            if self._unledgered_transition_count <= 0:
                return
            self._aborted_rollout = AbortedRolloutV1.build(
                reason=reason,
                transition_count=self._unledgered_transition_count,
                num_timesteps_after_abort=self.num_timesteps,
            )

        def train(self) -> None:
            """Optimize one collected rollout as an irreversible transaction."""

            if self._aborted_rollout is not None:
                raise RuntimeError("an aborted rollout permanently blocks training")
            if self._pending_rollout is None:
                raise RuntimeError("constrained PPO has no collected rollout to train")
            try:
                self._train_pending_rollout()
            except BaseException:
                self._mark_aborted_rollout("optimizer-exception")
                raise

        def _train_pending_rollout(self) -> None:
            if self._pending_rollout is None:
                raise RuntimeError("constrained PPO has no collected rollout to train")
            self._live_policy_architecture()
            self._validate_rollout_buffer_finite(self.rollout_buffer)
            # These digests also fail closed on non-finite policy or optimizer
            # state before either the stock or constrained optimizer is entered.
            _policy_parameter_digest(self.policy)
            _optimizer_state_digest(self.policy)
            if not self.constraint_settings.uses_constraints:
                if type(self.policy) is not ActorCriticPolicy:
                    raise TypeError("scalar PPO lost its stock SB3 policy")
                if type(self.rollout_buffer) is not RolloutBuffer:
                    raise TypeError("scalar PPO lost its stock SB3 rollout buffer")
                optimizer_step_before = self._optimizer_step_count()
                super().train()
                _policy_parameter_digest(self.policy)
                _optimizer_state_digest(self.policy)
                optimizer_steps = self._optimizer_step_count() - optimizer_step_before
                logger_values = self.logger.name_to_value
                self._finish_pending_rollout(
                    cost_value_losses=(0.0, 0.0, 0.0, 0.0),
                    approximate_kl=_finite_float(
                        "stock PPO approximate_kl",
                        logger_values.get("train/approx_kl", 0.0),
                    ),
                    entropy_loss=_finite_float(
                        "stock PPO entropy_loss",
                        logger_values.get("train/entropy_loss", 0.0),
                    ),
                    clip_fraction=_non_negative_float(
                        "stock PPO clip_fraction",
                        logger_values.get("train/clip_fraction", 0.0),
                    ),
                    optimizer_steps=optimizer_steps,
                )
                return
            if type(self.rollout_buffer) is not CostRolloutBufferV1:
                raise TypeError("constrained PPO lost its cost rollout buffer")
            if type(self.policy) is not ConstrainedActorCriticPolicyV1:
                raise TypeError("constrained PPO lost its cost-aware policy")
            constrained_policy = cast(ConstrainedActorCriticPolicyV1, self.policy)
            self.policy.set_training_mode(True)
            self._update_learning_rate(self.policy.optimizer)
            clip_range_setting = self.clip_range
            clip_range = (
                clip_range_setting(self._current_progress_remaining)
                if callable(clip_range_setting)
                else float(clip_range_setting)
            )
            clip_range_vf_setting = self.clip_range_vf
            clip_range_vf = None
            if clip_range_vf_setting is not None:
                clip_range_vf = (
                    clip_range_vf_setting(self._current_progress_remaining)
                    if callable(clip_range_vf_setting)
                    else float(clip_range_vf_setting)
                )
            entropy_losses: list[float] = []
            policy_losses: list[float] = []
            value_losses: list[float] = []
            cost_value_losses: list[list[float]] = [[], [], [], []]
            clip_fractions: list[float] = []
            approximate_kls: list[float] = []
            optimizer_steps = 0
            continue_training = True
            final_loss: Any = th.zeros((), device=self.device)
            multipliers = th.as_tensor(
                self._multiplier_controller.state.multipliers,
                dtype=th.float32,
                device=self.device,
            )

            for _epoch in range(self.n_epochs):
                epoch_kls: list[float] = []
                for rollout_data in self.rollout_buffer.get(self.batch_size):
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        actions = actions.long().flatten()
                    values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations,
                        actions,
                    )
                    _require_finite_tensor("optimizer reward values", values)
                    _require_finite_tensor("optimizer log probabilities", log_prob)
                    if entropy is not None:
                        _require_finite_tensor("optimizer entropy", entropy)
                    values = values.flatten()
                    reward_advantage = _torch_normalize_advantage(
                        rollout_data.advantages
                    )
                    if self.constraint_settings.uses_constraints:
                        normalized_costs = th.stack(
                            [
                                _torch_normalize_advantage(
                                    rollout_data.cost_advantages[:, index]
                                )
                                for index in range(COST_CHANNEL_COUNT)
                            ],
                            dim=1,
                        )
                        actor_advantage = (
                            reward_advantage
                            - (normalized_costs * multipliers).sum(dim=1)
                        ) / (1.0 + multipliers.sum())
                    else:
                        actor_advantage = reward_advantage
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    _require_finite_tensor("optimizer probability ratio", ratio)
                    policy_loss_1 = actor_advantage * ratio
                    policy_loss_2 = actor_advantage * th.clamp(
                        ratio,
                        1.0 - clip_range,
                        1.0 + clip_range,
                    )
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                    policy_losses.append(float(policy_loss.item()))
                    clip_fraction = float(
                        th.mean((th.abs(ratio - 1.0) > clip_range).float()).item()
                    )
                    clip_fractions.append(clip_fraction)

                    if clip_range_vf is None:
                        values_pred = values
                    else:
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values,
                            -clip_range_vf,
                            clip_range_vf,
                        )
                    reward_value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(float(reward_value_loss.item()))
                    if entropy is None:
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)
                    entropy_losses.append(float(entropy_loss.item()))

                    aggregate_cost_value_loss: Any = th.zeros((), device=self.device)
                    if self.constraint_settings.uses_constraints:
                        predicted_cost_values = constrained_policy.cost_values(
                            rollout_data.observations
                        )
                        _require_finite_tensor(
                            "optimizer predicted cost values", predicted_cost_values
                        )
                        per_channel_losses = [
                            F.mse_loss(
                                rollout_data.cost_returns[:, index],
                                predicted_cost_values[:, index],
                            )
                            for index in range(COST_CHANNEL_COUNT)
                        ]
                        for index, loss_value in enumerate(per_channel_losses):
                            cost_value_losses[index].append(float(loss_value.item()))
                        aggregate_cost_value_loss = (
                            self.constraint_settings.cost_value_loss_coefficient
                            * th.stack(per_channel_losses).mean()
                        )
                    else:
                        for channel_losses in cost_value_losses:
                            channel_losses.append(0.0)

                    final_loss = (
                        policy_loss
                        + self.ent_coef * entropy_loss
                        + self.vf_coef * reward_value_loss
                        + aggregate_cost_value_loss
                    )
                    _require_finite_tensor("optimizer loss", final_loss)
                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approximate_kl = float(
                            th.mean((th.exp(log_ratio) - 1.0) - log_ratio).cpu().item()
                        )
                    epoch_kls.append(approximate_kl)
                    approximate_kls.append(approximate_kl)
                    if (
                        self.target_kl is not None
                        and approximate_kl > 1.5 * self.target_kl
                    ):
                        continue_training = False
                        break
                    self.policy.optimizer.zero_grad()
                    final_loss.backward()
                    for name, parameter in self.policy.named_parameters():
                        if parameter.grad is not None:
                            _require_finite_tensor(
                                f"optimizer gradient {name}", parameter.grad
                            )
                    actor_reward_parameters = tuple(
                        parameter
                        for name, parameter in self.policy.named_parameters()
                        if not name.startswith("cost_critic.")
                    )
                    th.nn.utils.clip_grad_norm_(
                        actor_reward_parameters,
                        self.max_grad_norm,
                    )
                    # Each cost head is clipped independently.  Together with
                    # the disjoint networks this prevents one resource target
                    # from changing another resource head's optimizer update.
                    for head in constrained_policy.cost_critic.heads:
                        th.nn.utils.clip_grad_norm_(
                            head.parameters(),
                            self.max_grad_norm,
                        )
                    for name, parameter in self.policy.named_parameters():
                        if parameter.grad is not None:
                            _require_finite_tensor(
                                f"clipped optimizer gradient {name}", parameter.grad
                            )
                    self.policy.optimizer.step()
                    _policy_parameter_digest(self.policy)
                    _optimizer_state_digest(self.policy)
                    optimizer_steps += 1
                self._n_updates += 1
                if not continue_training:
                    break

            explained_var = explained_variance(
                self.rollout_buffer.values.flatten(),
                self.rollout_buffer.returns.flatten(),
            )
            self.logger.record("train/entropy_loss", float(np.mean(entropy_losses)))
            self.logger.record(
                "train/policy_gradient_loss", float(np.mean(policy_losses))
            )
            self.logger.record("train/value_loss", float(np.mean(value_losses)))
            self.logger.record("train/approx_kl", float(np.mean(approximate_kls)))
            self.logger.record("train/clip_fraction", float(np.mean(clip_fractions)))
            self.logger.record("train/loss", float(final_loss.item()))
            self.logger.record("train/explained_variance", explained_var)
            if hasattr(self.policy, "log_std"):
                self.logger.record(
                    "train/std", float(th.exp(self.policy.log_std).mean().item())
                )
            self.logger.record(
                "train/n_updates", self._n_updates, exclude="tensorboard"
            )
            self.logger.record("train/clip_range", clip_range)
            if clip_range_vf is not None:
                self.logger.record("train/clip_range_vf", clip_range_vf)

            per_channel_mean = cast(
                FloatVector4,
                tuple(
                    float(np.mean(losses)) if losses else 0.0
                    for losses in cost_value_losses
                ),
            )
            for index, resource in enumerate(RESOURCE_ORDER):
                self.logger.record(
                    f"train/cost_value_loss_{resource}", per_channel_mean[index]
                )
            self._finish_pending_rollout(
                cost_value_losses=per_channel_mean,
                approximate_kl=(
                    float(np.mean(approximate_kls)) if approximate_kls else 0.0
                ),
                entropy_loss=(
                    float(np.mean(entropy_losses)) if entropy_losses else 0.0
                ),
                clip_fraction=(
                    float(np.mean(clip_fractions)) if clip_fractions else 0.0
                ),
                optimizer_steps=optimizer_steps,
            )

        def _optimizer_step_count(self) -> int:
            steps: list[int] = []
            for state in self.policy.optimizer.state.values():
                step = state.get("step")
                if step is None:
                    continue
                if hasattr(step, "item"):
                    step = step.item()
                if (
                    isinstance(step, bool)
                    or not isinstance(step, Real)
                    or not math.isfinite(float(step))
                    or float(step) < 0.0
                    or not float(step).is_integer()
                ):
                    raise ValueError("optimizer step counter is invalid")
                steps.append(int(step))
            if steps and any(step != steps[0] for step in steps[1:]):
                raise ValueError("optimizer parameter step counters diverged")
            return max(steps, default=0)

        def _finish_pending_rollout(
            self,
            *,
            cost_value_losses: FloatVector4,
            approximate_kl: float,
            entropy_loss: float,
            clip_fraction: float,
            optimizer_steps: int,
        ) -> None:
            pending = self._pending_rollout
            if pending is None:  # pragma: no cover - callers guard this
                raise RuntimeError("rollout finalization lost pending evidence")
            ledger = RolloutLedgerV1.build(
                algorithm_id=self.algorithm_id,
                rollout_index=len(self._rollout_ledgers) + 1,
                transition_count=pending.transition_count,
                completed_episode_count=len(pending.completed),
                complete_cycle_count=pending.complete_cycle_count,
                reward_return=pending.reward_return,
                raw_cost=pending.raw_cost,
                training_cost=pending.training_cost,
                mean_completed_episode_cost=pending.mean_completed_episode_cost,
                cost_value_losses=cost_value_losses,
                multiplier_state=self._multiplier_controller.state,
                multiplier_min=self.constraint_settings.multiplier_min,
                multiplier_max=self.constraint_settings.multiplier_max,
                approximate_kl=approximate_kl,
                entropy_loss=entropy_loss,
                clip_fraction=clip_fraction,
                optimizer_steps=optimizer_steps,
                elapsed_compute_s=time.perf_counter() - pending.started_at,
                previous_sha256=(
                    None
                    if not self._rollout_ledgers
                    else self._rollout_ledgers[-1].sha256
                ),
            )
            self._rollout_ledgers = (*self._rollout_ledgers, ledger)
            receipt = self._resume_continuation_receipt
            if receipt is not None and receipt.first_resumed_rollout_sha256 is None:
                if ledger.rollout_index != receipt.first_resumed_rollout_index:
                    raise RuntimeError(
                        "first resumed rollout index differs from continuation receipt"
                    )
                self._resume_continuation_receipt = receipt.bind_first_rollout(
                    ledger.sha256
                )
            self._pending_rollout = None
            self._unledgered_transition_count = 0

        @property
        def multiplier_state(self) -> PILagrangeStateV1:
            return self._multiplier_controller.state

        @property
        def rollout_ledgers(self) -> tuple[RolloutLedgerV1, ...]:
            return self._rollout_ledgers

        @property
        def rollout_ledger_sha256(self) -> str | None:
            return (
                None if not self._rollout_ledgers else self._rollout_ledgers[-1].sha256
            )

        @property
        def ppo_configuration(self) -> dict[str, Any]:
            """Canonical live PPO, space, policy, and optimizer architecture."""

            return self._live_ppo_configuration()

        @property
        def policy_parameter_sha256(self) -> str:
            return _policy_parameter_digest(self.policy)

        @property
        def optimizer_state_sha256(self) -> str:
            return _optimizer_state_digest(self.policy)

        @property
        def learner_rng_state(self) -> dict[str, Any]:
            state = LearnerRNGStateV1.capture()
            self._learner_rng_state = state
            return state.to_dict()

        @property
        def episode_decision_indices(self) -> tuple[int, ...]:
            if self._episode_cost_accumulator is None:
                raise RuntimeError("episode cost accumulator is not initialized")
            return self._episode_cost_accumulator.current_decision_indices

        @property
        def checkpoint_observation(self) -> npt.NDArray[Any]:
            """Return a defensive copy of the vectorized continuation observation."""

            if self._last_obs is None:
                raise RuntimeError("checkpoint has no continuation observation")
            observation = np.asarray(self._last_obs)
            _require_finite_array("checkpoint observation", observation)
            return observation.copy()

        def _checkpoint_observation_digest(self) -> str:
            if self._last_obs is not None:
                return _checkpoint_observation_sha256(self._last_obs)
            payload_json = self._embedded_constraint_state_payload_json
            if not isinstance(payload_json, str):
                raise RuntimeError("checkpoint has no observation identity")
            payload = json.loads(payload_json)
            if not isinstance(payload, dict):  # pragma: no cover - load validates
                raise RuntimeError("checkpoint observation identity is malformed")
            return _sha256_hex(
                "checkpoint observation digest",
                payload.get("checkpoint_observation_sha256"),
            )

        @property
        def external_episode_state_pin(self) -> dict[str, Any] | None:
            return (
                None
                if self._external_episode_state_pin is None
                else self._external_episode_state_pin.to_dict()
            )

        @property
        def aborted_rollout(self) -> dict[str, Any] | None:
            return (
                None
                if self._aborted_rollout is None
                else self._aborted_rollout.to_dict()
            )

        @property
        def requires_external_episode_state_admission(self) -> bool:
            return self._resume_continuity_required

        def pin_external_episode_state(
            self,
            payload: Mapping[str, Any],
            *,
            sha256: str | None = None,
        ) -> dict[str, Any]:
            """Pin opaque runtime state at a safe ledger cut.

            ``sha256`` is this API's domain-separated pin digest (normally the
            value returned by an earlier call), not a digest embedded inside
            the allocator payload.  Callers pinning a fresh snapshot omit it.
            """

            if self._aborted_rollout is not None:
                raise RuntimeError("cannot pin state after an aborted rollout")
            if self._pending_rollout is not None or self._unledgered_transition_count:
                raise RuntimeError("external state can be pinned only at a ledger cut")
            pin = ExternalEpisodeStatePinV1.build(payload, sha256=sha256)
            self._external_episode_state_pin = pin
            self._embedded_constraint_state_schema = None
            self._embedded_constraint_state_payload_json = None
            self._embedded_constraint_state_sha256 = None
            return pin.to_dict()

        def admit_external_episode_state(
            self,
            payload: Mapping[str, Any],
            *,
            decision_indices: Sequence[int],
            runtime_observation: npt.ArrayLike,
            sha256: str | None = None,
        ) -> None:
            """Admit a caller-restored runtime snapshot for continuation.

            As in :meth:`pin_external_episode_state`, ``sha256`` refers only to
            this wrapper's domain-separated pin digest.  The opaque payload is
            responsible for validating any digest it contains itself.  This
            method does not inspect a ``VecEnv``; the runtime layer must derive
            ``runtime_observation`` from its restored state, and this core then
            compares it exactly with the checkpoint observation.
            """

            if self._external_episode_state_pin is None:
                raise RuntimeError("checkpoint has no pinned external episode state")
            supplied = ExternalEpisodeStatePinV1.build(payload, sha256=sha256)
            if supplied != self._external_episode_state_pin:
                raise ValueError("external episode state differs from checkpoint pin")
            if isinstance(decision_indices, (str, bytes)) or not isinstance(
                decision_indices, Sequence
            ):
                raise TypeError("decision_indices must be a sequence")
            normalized = tuple(
                _non_negative_int(f"decision_indices[{index}]", value)
                for index, value in enumerate(decision_indices)
            )
            if len(normalized) != self.n_envs or any(
                value >= BETA_DECISIONS_PER_EPISODE for value in normalized
            ):
                raise ValueError("external decision indices have the wrong domain")
            if normalized != self.episode_decision_indices:
                raise ValueError("external decision cursors differ from cost state")
            if self._resume_continuity_required:
                if self._loaded_force_reset is not False:
                    raise RuntimeError(
                        "checkpoint continuation requires "
                        "ConstrainedPPO.load(force_reset=False)"
                    )
                if self._last_obs is None:
                    raise RuntimeError(
                        "checkpoint continuation lost its checkpoint observation"
                    )
            expected_observation = self.checkpoint_observation
            supplied_observation = np.asarray(runtime_observation)
            _require_finite_array("restored runtime observation", supplied_observation)
            if supplied_observation.shape != expected_observation.shape:
                raise ValueError(
                    "restored runtime observation shape differs from checkpoint"
                )
            if supplied_observation.dtype != expected_observation.dtype:
                raise ValueError(
                    "restored runtime observation dtype differs from checkpoint"
                )
            if not np.array_equal(supplied_observation, expected_observation):
                raise ValueError("restored runtime observation differs from checkpoint")
            if self._learner_rng_state is None:
                raise RuntimeError("checkpoint omitted exact learner RNG state")
            if self._resume_continuity_required:
                artifact_sha256 = self._loaded_policy_artifact_sha256
                if artifact_sha256 is None:
                    raise RuntimeError(
                        "checkpoint continuation requires a filesystem artifact identity"
                    )
                parent_state = self.embedded_constraint_state
                self._resume_continuation_receipt = (
                    ResumeContinuationReceiptV1.from_parent(
                        artifact_sha256=artifact_sha256,
                        constraint_state=parent_state,
                        checkpoint_observation_sha256=(
                            _checkpoint_observation_sha256(expected_observation)
                        ),
                    )
                )
            self._learner_rng_state.restore()
            self._resume_continuity_required = False

        def learn(
            self,
            total_timesteps: int,
            callback: Any = None,
            log_interval: int = 1,
            tb_log_name: str = "OnPolicyAlgorithm",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
        ) -> "ConstrainedPPO":
            if self._aborted_rollout is not None:
                raise RuntimeError("an aborted rollout permanently blocks training")
            if self._resume_continuity_required:
                raise RuntimeError(
                    "checkpoint continuation requires external-state admission"
                )
            # SB3 resets the vector environment inside ``_setup_learn``.  Prove
            # the complete live learner contract before delegating so a
            # substituted policy graph cannot consume even one allocator-owned
            # episode identity before it is rejected.
            self._live_ppo_configuration()
            if self._rollout_ledgers and reset_num_timesteps:
                raise ValueError(
                    "ledgered training must continue with reset_num_timesteps=False"
                )
            return cast(
                ConstrainedPPO,
                super().learn(
                    total_timesteps,
                    callback=callback,
                    log_interval=log_interval,
                    tb_log_name=tb_log_name,
                    reset_num_timesteps=reset_num_timesteps,
                    progress_bar=progress_bar,
                ),
            )

        def constraint_provenance(self) -> dict[str, Any]:
            """Return canonical, digest-bound optimizer evidence for metadata."""

            return self.constraint_state_dict()

        @property
        def embedded_constraint_state(self) -> dict[str, Any]:
            """Return the validated envelope stored in a checkpoint artifact.

            Unlike :meth:`constraint_state_dict`, this accessor never captures
            live RNG state. It reconstructs a fresh defensive object solely
            from the artifact's canonical JSON and digest, making it suitable
            for exact comparison with separately admitted metadata evidence.
            """

            if self._embedded_constraint_state_schema != (
                CONSTRAINT_STATE_ENVELOPE_SCHEMA
            ):
                raise ValueError("artifact omitted constraint-state envelope schema")
            payload_json = self._embedded_constraint_state_payload_json
            digest = self._embedded_constraint_state_sha256
            if not isinstance(payload_json, str) or not isinstance(digest, str):
                raise ValueError("artifact omitted canonical constraint payload/digest")
            try:
                payload = json.loads(payload_json)
            except (TypeError, ValueError) as reason:
                raise ValueError("artifact constraint payload is not JSON") from reason
            if (
                not isinstance(payload, dict)
                or _canonical_json(payload) != payload_json
            ):
                raise ValueError("artifact constraint payload is not canonical JSON")
            envelope = {
                "schema": CONSTRAINT_STATE_ENVELOPE_SCHEMA,
                "payload": payload,
                "sha256": digest,
            }
            self._validate_constraint_state_envelope(envelope, compare_live=True)
            # ``payload`` came from a fresh JSON decode and validation retains
            # no reference to it, so callers cannot mutate model state.
            return envelope

        def constraint_state_dict(self) -> dict[str, Any]:
            payload = self._constraint_state_payload()
            return {
                "schema": CONSTRAINT_STATE_ENVELOPE_SCHEMA,
                "payload": payload,
                "sha256": _canonical_digest(
                    CONSTRAINT_STATE_ENVELOPE_SCHEMA,
                    payload,
                ),
            }

        def _constraint_state_payload(self) -> dict[str, Any]:
            if self._episode_cost_accumulator is None:
                raise RuntimeError("episode cost accumulator is not initialized")
            rng_state = (
                self._learner_rng_state
                if self._resume_continuity_required
                and self._learner_rng_state is not None
                else LearnerRNGStateV1.capture()
            )
            self._learner_rng_state = rng_state
            return {
                "schema": CONSTRAINT_STATE_SCHEMA,
                "protocol_id": PROTOCOL_ID,
                "algorithm_id": self.algorithm_id,
                "settings": self.constraint_settings.to_dict(),
                "ppo_configuration": self.ppo_configuration,
                "policy_parameter_sha256": self.policy_parameter_sha256,
                "optimizer_state_sha256": self.optimizer_state_sha256,
                "learner_rng_state": rng_state.to_dict(),
                "multiplier_state": self.multiplier_state.to_dict(),
                "episode_cost_accumulator": self._episode_cost_accumulator.to_dict(),
                "rollout_ledgers": [item.to_dict() for item in self._rollout_ledgers],
                "rollout_ledger_sha256": self.rollout_ledger_sha256,
                "num_timesteps": int(self.num_timesteps),
                "ppo_epoch_update_count": int(self._n_updates),
                "external_episode_state_pin": self.external_episode_state_pin,
                "checkpoint_observation_sha256": (
                    self._checkpoint_observation_digest()
                ),
                "resume_continuation_receipt": (
                    None
                    if self._resume_continuation_receipt is None
                    else self._resume_continuation_receipt.to_dict()
                ),
                "aborted_rollout": self.aborted_rollout,
                "unledgered_transition_count": (self._unledgered_transition_count),
                "pending_rollout": self._pending_rollout is not None,
            }

        def _validate_constraint_state_envelope(
            self,
            raw: Mapping[str, Any],
            *,
            compare_live: bool,
        ) -> dict[str, Any]:
            if not isinstance(raw, Mapping) or set(raw) != {
                "schema",
                "payload",
                "sha256",
            }:
                raise ValueError("constraint-state envelope fields mismatch")
            if raw["schema"] != CONSTRAINT_STATE_ENVELOPE_SCHEMA:
                raise ValueError("constraint-state envelope schema mismatch")
            payload = _canonical_object_copy("constraint state payload", raw["payload"])
            if set(payload) != _CONSTRAINT_STATE_PAYLOAD_FIELDS:
                raise ValueError("constraint-state payload fields mismatch")
            supplied_digest = raw["sha256"]
            if not isinstance(supplied_digest, str) or supplied_digest != (
                _canonical_digest(CONSTRAINT_STATE_ENVELOPE_SCHEMA, payload)
            ):
                raise ValueError("constraint-state envelope digest mismatch")
            if payload["schema"] != CONSTRAINT_STATE_SCHEMA:
                raise ValueError("constrained PPO state schema mismatch")
            if payload["protocol_id"] != PROTOCOL_ID:
                raise ValueError("constrained PPO state protocol mismatch")
            algorithm_id = payload["algorithm_id"]
            if not isinstance(algorithm_id, str):
                raise TypeError("constraint-state algorithm_id must be a string")
            settings_raw = payload["settings"]
            if not isinstance(settings_raw, Mapping):
                raise TypeError("constraint-state settings must be an object")
            settings = ConstrainedPPOSettingsV1.from_dict(settings_raw)
            if algorithm_id != settings.algorithm_id:
                raise ValueError("constraint-state algorithm and settings disagree")
            ppo_raw = payload["ppo_configuration"]
            if not isinstance(ppo_raw, Mapping):
                raise TypeError("constraint-state PPO configuration must be an object")
            ppo_configuration = _canonical_object_copy(
                "constraint-state PPO configuration", ppo_raw
            )
            if ppo_configuration.get("schema") != PPO_CONFIGURATION_SCHEMA:
                raise ValueError("constraint-state PPO configuration schema mismatch")
            policy_architecture = ppo_configuration.get("policy_architecture")
            if not isinstance(policy_architecture, Mapping) or (
                policy_architecture.get("schema") != POLICY_ARCHITECTURE_SCHEMA
            ):
                raise ValueError("constraint-state policy architecture schema mismatch")
            if ppo_configuration != self._live_ppo_configuration():
                raise ValueError("embedded PPO configuration differs from loaded model")
            policy_digest = _sha256_hex(
                "constraint-state policy parameter digest",
                payload["policy_parameter_sha256"],
            )
            optimizer_digest = _sha256_hex(
                "constraint-state optimizer state digest",
                payload["optimizer_state_sha256"],
            )
            if policy_digest != self.policy_parameter_sha256:
                raise ValueError("embedded policy parameters differ from loaded model")
            if optimizer_digest != self.optimizer_state_sha256:
                raise ValueError("embedded optimizer state differs from loaded model")
            rng_raw = payload["learner_rng_state"]
            if not isinstance(rng_raw, Mapping):
                raise TypeError("constraint-state learner RNG state must be an object")
            learner_rng_state = LearnerRNGStateV1.from_dict(rng_raw)
            multiplier_raw = payload["multiplier_state"]
            if not isinstance(multiplier_raw, Mapping):
                raise TypeError("constraint-state multiplier must be an object")
            multiplier_state = PILagrangeStateV1.from_dict(multiplier_raw)
            PILagrangeControllerV1(settings, multiplier_state)
            accumulator_raw = payload["episode_cost_accumulator"]
            if not isinstance(accumulator_raw, Mapping):
                raise TypeError("constraint-state accumulator must be an object")
            accumulator = EpisodeCostAccumulatorV1.from_dict(accumulator_raw)
            if accumulator.n_envs != self.n_envs:
                raise ValueError("checkpoint episode accumulator n_envs mismatch")
            num_timesteps = _non_negative_int(
                "constraint-state num_timesteps", payload["num_timesteps"]
            )
            ppo_updates = _non_negative_int(
                "constraint-state ppo_epoch_update_count",
                payload["ppo_epoch_update_count"],
            )
            unledgered = _non_negative_int(
                "constraint-state unledgered_transition_count",
                payload["unledgered_transition_count"],
            )
            pending = payload["pending_rollout"]
            if not isinstance(pending, bool):
                raise TypeError("constraint-state pending_rollout must be a bool")

            external_raw = payload["external_episode_state_pin"]
            if external_raw is None:
                external_pin = None
            elif isinstance(external_raw, Mapping):
                external_pin = ExternalEpisodeStatePinV1.from_dict(external_raw)
            else:
                raise TypeError("external episode state pin must be an object or null")
            checkpoint_observation_digest = _sha256_hex(
                "constraint-state checkpoint observation digest",
                payload["checkpoint_observation_sha256"],
            )
            receipt_raw = payload["resume_continuation_receipt"]
            if receipt_raw is None:
                resume_receipt = None
            elif isinstance(receipt_raw, Mapping):
                resume_receipt = ResumeContinuationReceiptV1.from_dict(receipt_raw)
            else:
                raise TypeError("resume continuation receipt must be an object or null")
            aborted_raw = payload["aborted_rollout"]
            if aborted_raw is None:
                aborted = None
            elif isinstance(aborted_raw, Mapping):
                aborted = AbortedRolloutV1.from_dict(aborted_raw)
            else:
                raise TypeError("aborted rollout must be an object or null")
            if pending and (unledgered == 0 or aborted is not None):
                raise ValueError("pending rollout accounting is incoherent")
            if aborted is not None and (
                pending
                or aborted.transition_count != unledgered
                or aborted.num_timesteps_after_abort != num_timesteps
            ):
                raise ValueError("aborted rollout accounting is incoherent")
            if unledgered and not pending and aborted is None:
                raise ValueError("unledgered transitions have no terminal state")

            ledger_raw = payload["rollout_ledgers"]
            if not isinstance(ledger_raw, list):
                raise TypeError("rollout_ledgers must be an array")
            ledgers: list[RolloutLedgerV1] = []
            previous: str | None = None
            cumulative_transitions = 0
            cumulative_ledger_raw = np.zeros(COST_CHANNEL_COUNT, dtype=np.float64)
            cumulative_ledger_training = np.zeros(COST_CHANNEL_COUNT, dtype=np.float64)
            cumulative_completed_training = np.zeros(
                COST_CHANNEL_COUNT, dtype=np.float64
            )
            replayed_controller = PILagrangeControllerV1(settings)
            for index, item in enumerate(ledger_raw, start=1):
                if not isinstance(item, Mapping):
                    raise TypeError("each rollout ledger must be an object")
                ledger = RolloutLedgerV1.from_dict(item)
                if ledger.rollout_index != index:
                    raise ValueError("rollout ledger indices are not contiguous")
                if ledger.algorithm_id != algorithm_id:
                    raise ValueError("rollout ledger algorithm substitution")
                ledger.verify(expected_previous_sha256=previous)
                if ledger.rollout_discontinuities != 0:
                    raise ValueError("rollout ledger records a discontinuity")
                if ledger.transition_count != self.n_steps * self.n_envs:
                    raise ValueError("rollout size differs from PPO configuration")
                if ledger.transition_count % self.n_envs:
                    raise ValueError("rollout transition count is not vector-aligned")
                before = cumulative_transitions // self.n_envs
                cumulative_transitions += ledger.transition_count
                after = cumulative_transitions // self.n_envs
                expected_completed = (
                    after // BETA_DECISIONS_PER_EPISODE
                    - before // BETA_DECISIONS_PER_EPISODE
                ) * self.n_envs
                expected_cycles = (
                    after // BETA_DECISIONS_PER_CYCLE
                    - before // BETA_DECISIONS_PER_CYCLE
                ) * self.n_envs
                if ledger.completed_episode_count != expected_completed:
                    raise ValueError("rollout completed-episode count is incoherent")
                if ledger.complete_cycle_count != expected_cycles:
                    raise ValueError("rollout settled-cycle count is incoherent")
                if (ledger.mean_completed_episode_cost is None) != (
                    expected_completed == 0
                ):
                    raise ValueError("rollout completed-cost mean is incoherent")
                if not np.allclose(
                    np.asarray(ledger.training_cost),
                    np.asarray(ledger.raw_cost) / BETA_EPISODE_CYCLE_COUNT,
                    rtol=0.0,
                    atol=1e-15,
                ):
                    raise ValueError("rollout cost scaling is incoherent")
                cumulative_ledger_raw += np.asarray(ledger.raw_cost, dtype=np.float64)
                cumulative_ledger_training += np.asarray(
                    ledger.training_cost, dtype=np.float64
                )
                if ledger.mean_completed_episode_cost is not None:
                    cumulative_completed_training += (
                        np.asarray(ledger.mean_completed_episode_cost, dtype=np.float64)
                        * ledger.completed_episode_count
                    )
                completed_costs = (
                    []
                    if ledger.mean_completed_episode_cost is None
                    else [ledger.mean_completed_episode_cost]
                )
                expected_multiplier = replayed_controller.update(completed_costs)
                ledger_multiplier = PILagrangeStateV1(
                    proportional=ledger.proportional_state,
                    integral=ledger.integral_state,
                    multipliers=ledger.multipliers,
                    completed_update_count=(ledger.multiplier_completed_update_count),
                )
                if ledger_multiplier != expected_multiplier:
                    raise ValueError("rollout multiplier history is incoherent")
                if not settings.uses_constraints and any(ledger.cost_value_losses):
                    raise ValueError("scalar PPO ledger contains cost-critic losses")
                previous = ledger.sha256
                ledgers.append(ledger)
            if payload["rollout_ledger_sha256"] != previous:
                raise ValueError("rollout ledger terminal digest mismatch")
            if resume_receipt is not None:
                receipt_index = resume_receipt.first_resumed_rollout_index
                if resume_receipt.first_resumed_rollout_sha256 is None:
                    raise ValueError(
                        "persisted resume continuation receipt is unfinished"
                    )
                if receipt_index > len(ledgers):
                    raise ValueError(
                        "resume continuation receipt has no first resumed rollout"
                    )
                first_resumed = ledgers[receipt_index - 1]
                if first_resumed.sha256 != resume_receipt.first_resumed_rollout_sha256:
                    raise ValueError(
                        "resume continuation receipt first rollout differs"
                    )
                transitions_before_resume = sum(
                    ledger.transition_count for ledger in ledgers[: receipt_index - 1]
                )
                if transitions_before_resume != resume_receipt.parent_num_timesteps:
                    raise ValueError(
                        "resume continuation receipt parent timestep cut differs"
                    )
                if (
                    first_resumed.previous_sha256
                    != resume_receipt.parent_rollout_ledger_sha256
                ):
                    raise ValueError(
                        "resume continuation receipt parent rollout tip differs"
                    )
            if not ledgers and ppo_updates != 0:
                raise ValueError("PPO update count exists without a rollout ledger")
            if ledgers:
                if not len(ledgers) <= ppo_updates <= len(ledgers) * self.n_epochs:
                    raise ValueError("PPO update count differs from rollout history")
                if (
                    self.target_kl is None
                    and ppo_updates != len(ledgers) * self.n_epochs
                ):
                    raise ValueError(
                        "PPO epoch update count differs from full training"
                    )
                expected_optimizer_steps = self.n_epochs * (
                    self.n_steps * self.n_envs // self.batch_size
                )
                if self.target_kl is None and any(
                    ledger.optimizer_steps != expected_optimizer_steps
                    for ledger in ledgers
                ):
                    raise ValueError(
                        "optimizer step count differs from PPO configuration"
                    )
            declared_optimizer_steps = sum(ledger.optimizer_steps for ledger in ledgers)
            if self._optimizer_step_count() != declared_optimizer_steps:
                raise ValueError(
                    "live optimizer step counter differs from rollout ledgers"
                )
            if cumulative_transitions + unledgered != num_timesteps:
                raise ValueError("constraint-state timestep accounting is incoherent")
            if num_timesteps % self.n_envs:
                raise ValueError("constraint-state timesteps are not vector-aligned")
            decisions_per_env = num_timesteps // self.n_envs
            decision_cursor = decisions_per_env % BETA_DECISIONS_PER_EPISODE
            if accumulator.current_decision_indices != (decision_cursor,) * self.n_envs:
                raise ValueError("episode accumulator decision cursor is incoherent")
            expected_total_cycles = (
                decisions_per_env // BETA_DECISIONS_PER_CYCLE
            ) * self.n_envs
            recorded_total_cycles = sum(
                ledger.complete_cycle_count for ledger in ledgers
            )
            expected_current_cycles = (
                decision_cursor // BETA_DECISIONS_PER_CYCLE
            ) * self.n_envs
            if int(accumulator.settled_cycles.sum()) != expected_current_cycles:
                raise ValueError(
                    "episode accumulator settled-cycle count is incoherent"
                )
            if recorded_total_cycles != expected_total_cycles:
                raise ValueError(
                    "constraint-state cumulative cycle count is incoherent"
                )
            if multiplier_state != replayed_controller.state:
                raise ValueError("final multiplier state differs from rollout history")
            expected_cumulative_training = (
                cumulative_completed_training
                + accumulator.training_cost.sum(axis=0, dtype=np.float64)
            )
            expected_cumulative_raw = (
                BETA_EPISODE_CYCLE_COUNT * cumulative_completed_training
                + accumulator.raw_cost.sum(axis=0, dtype=np.float64)
            )
            if not np.allclose(
                cumulative_ledger_training,
                expected_cumulative_training,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError(
                    "cumulative ledger training costs differ from completed and live episodes"
                )
            if not np.allclose(
                cumulative_ledger_raw,
                expected_cumulative_raw,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError(
                    "cumulative ledger raw costs differ from completed and live episodes"
                )

            parsed = {
                "payload": payload,
                "settings": settings,
                "ppo_configuration": ppo_configuration,
                "policy_digest": policy_digest,
                "optimizer_digest": optimizer_digest,
                "learner_rng_state": learner_rng_state,
                "multiplier_state": multiplier_state,
                "accumulator": accumulator,
                "ledgers": tuple(ledgers),
                "external_pin": external_pin,
                "checkpoint_observation_sha256": checkpoint_observation_digest,
                "resume_receipt": resume_receipt,
                "aborted": aborted,
                "num_timesteps": num_timesteps,
                "ppo_updates": ppo_updates,
                "unledgered": unledgered,
                "pending": pending,
                "sha256": supplied_digest,
            }
            if compare_live:
                self._crosscheck_live_constraint_state(parsed)
            return parsed

        def _crosscheck_live_constraint_state(self, parsed: Mapping[str, Any]) -> None:
            if parsed["settings"] != self.constraint_settings:
                raise ValueError("embedded settings differ from loaded model")
            if parsed["ppo_configuration"] != self._live_ppo_configuration():
                raise ValueError("embedded PPO configuration differs from loaded model")
            if parsed["policy_digest"] != self.policy_parameter_sha256:
                raise ValueError("embedded policy parameters differ from loaded model")
            if parsed["optimizer_digest"] != self.optimizer_state_sha256:
                raise ValueError("embedded optimizer state differs from loaded model")
            if parsed["learner_rng_state"] != self._learner_rng_state:
                raise ValueError("embedded learner RNG state differs from loaded model")
            if self.algorithm_id != self.constraint_settings.algorithm_id:
                raise ValueError("loaded algorithm and settings disagree")
            if self.seed != self.constraint_settings.optimizer_seed:
                raise ValueError("loaded optimizer seed differs from settings")
            if self._multiplier_controller.settings != self.constraint_settings:
                raise ValueError("loaded multiplier controller settings differ")
            if parsed["multiplier_state"] != self.multiplier_state:
                raise ValueError("embedded multiplier state differs from loaded model")
            if self._episode_cost_accumulator is None or (
                parsed["accumulator"].to_dict()
                != self._episode_cost_accumulator.to_dict()
            ):
                raise ValueError("embedded accumulator differs from loaded model")
            if parsed["ledgers"] != self._rollout_ledgers:
                raise ValueError("embedded ledgers differ from loaded model")
            if parsed["external_pin"] != self._external_episode_state_pin:
                raise ValueError("embedded external pin differs from loaded model")
            if self._last_obs is not None and parsed[
                "checkpoint_observation_sha256"
            ] != _checkpoint_observation_sha256(self._last_obs):
                raise ValueError(
                    "embedded checkpoint observation differs from loaded model"
                )
            if parsed["resume_receipt"] != self._resume_continuation_receipt:
                raise ValueError("embedded resume receipt differs from loaded model")
            if parsed["aborted"] != self._aborted_rollout:
                raise ValueError("embedded abort state differs from loaded model")
            if parsed["num_timesteps"] != self.num_timesteps:
                raise ValueError("embedded timestep count differs from loaded model")
            if parsed["ppo_updates"] != self._n_updates:
                raise ValueError("embedded PPO update count differs from loaded model")
            if parsed["unledgered"] != self._unledgered_transition_count:
                raise ValueError("embedded unledgered count differs from loaded model")
            if parsed["pending"] != (self._pending_rollout is not None):
                raise ValueError("embedded pending state differs from loaded model")
            expected_policy: type[Any] = (
                ConstrainedActorCriticPolicyV1
                if self.constraint_settings.uses_constraints
                else ActorCriticPolicy
            )
            expected_buffer: type[Any] = (
                CostRolloutBufferV1
                if self.constraint_settings.uses_constraints
                else RolloutBuffer
            )
            if type(self.policy) is not expected_policy:
                raise ValueError("loaded policy class differs from algorithm family")
            if type(self.rollout_buffer) is not expected_buffer:
                raise ValueError("loaded rollout buffer differs from algorithm family")

        def load_constraint_state_dict(self, raw: Mapping[str, Any]) -> None:
            """Validate and restore canonical JSON metadata independently."""

            parsed = self._validate_constraint_state_envelope(
                raw,
                compare_live=False,
            )
            if parsed["settings"] != self.constraint_settings:
                raise ValueError("constrained PPO checkpoint settings mismatch")
            if (
                parsed["pending"]
                or parsed["aborted"] is not None
                or parsed["unledgered"]
            ):
                raise ValueError("cannot restore an unsafe rollout cut")
            self._multiplier_controller.load_state(parsed["multiplier_state"])
            restored_accumulator = cast(
                EpisodeCostAccumulatorV1,
                parsed["accumulator"],
            )
            self._episode_cost_accumulator = restored_accumulator
            self._rollout_ledgers = parsed["ledgers"]
            self._pending_rollout = None
            self._aborted_rollout = None
            self._unledgered_transition_count = 0
            self._external_episode_state_pin = parsed["external_pin"]
            self._resume_continuation_receipt = parsed["resume_receipt"]
            self._learner_rng_state = parsed["learner_rng_state"]
            self.num_timesteps = parsed["num_timesteps"]
            self._n_updates = parsed["ppo_updates"]
            self._embedded_constraint_state_schema = CONSTRAINT_STATE_ENVELOPE_SCHEMA
            self._embedded_constraint_state_payload_json = _canonical_json(
                parsed["payload"]
            )
            self._embedded_constraint_state_sha256 = parsed["sha256"]
            self._resume_continuity_required = True

        def _assert_checkpointable(self) -> None:
            if self._aborted_rollout is not None:
                raise RuntimeError("cannot save after an aborted rollout")
            if self._pending_rollout is not None:
                raise RuntimeError("cannot save before the rollout is optimized")
            if self._unledgered_transition_count:
                raise RuntimeError("cannot save unledgered transitions")
            if self._resume_continuity_required:
                raise RuntimeError("cannot save an unadmitted partial resume")
            if self._episode_cost_accumulator is None:
                raise RuntimeError("episode cost accumulator is not initialized")
            if self._external_episode_state_pin is None:
                raise RuntimeError(
                    "continuation-capable checkpoint requires pinned external state"
                )
            if (
                self._resume_continuation_receipt is not None
                and self._resume_continuation_receipt.first_resumed_rollout_sha256
                is None
            ):
                raise RuntimeError(
                    "cannot save before the first resumed rollout is receipt-bound"
                )

        def save(
            self,
            path: Any,
            exclude: Any = None,
            include: Any = None,
        ) -> None:
            self._assert_checkpointable()
            rng_state = LearnerRNGStateV1.capture()
            self._learner_rng_state = rng_state
            envelope = self.constraint_state_dict()
            self._validate_constraint_state_envelope(envelope, compare_live=True)
            protected = {
                "constraint_settings",
                "algorithm_id",
                "_multiplier_controller",
                "_episode_cost_accumulator",
                "_rollout_ledgers",
                "_external_episode_state_pin",
                "_learner_rng_state",
                "_resume_continuation_receipt",
                "_embedded_constraint_state_schema",
                "_embedded_constraint_state_payload_json",
                "_embedded_constraint_state_sha256",
            }
            excluded = set(exclude or ())
            if protected & excluded:
                raise ValueError(
                    "cannot exclude protected constraint state from artifact"
                )
            excluded.add("_loaded_policy_artifact_sha256")
            self._embedded_constraint_state_schema = CONSTRAINT_STATE_ENVELOPE_SCHEMA
            self._embedded_constraint_state_payload_json = _canonical_json(
                envelope["payload"]
            )
            self._embedded_constraint_state_sha256 = cast(str, envelope["sha256"])
            try:
                super().save(path, exclude=excluded, include=include)
            finally:
                # Saving must be transparent to the uninterrupted comparison arm.
                rng_state.restore()

        def _validate_embedded_constraint_state(self) -> None:
            _ = self.embedded_constraint_state

        def loaded_constraint_state_attestation(
            self,
        ) -> ValidatedConstraintStateAttestationV1:
            """Issue proof that this exact filesystem artifact passed ``load``."""

            artifact_sha256 = self._loaded_policy_artifact_sha256
            if artifact_sha256 is None:
                raise RuntimeError(
                    "constraint-state attestation requires a validated filesystem "
                    "artifact loaded by ConstrainedPPO.load"
                )
            envelope = self.embedded_constraint_state
            return ValidatedConstraintStateAttestationV1(
                artifact_sha256=artifact_sha256,
                algorithm_id=self.algorithm_id,
                num_timesteps=self.num_timesteps,
                constraint_state=envelope,
                checkpoint_observation_sha256=envelope["payload"][
                    "checkpoint_observation_sha256"
                ],
                _issuer=_ATTESTATION_ISSUER,
            )

        @classmethod
        def load(
            cls,
            path: Any,
            env: Any = None,
            device: Any = "auto",
            custom_objects: dict[str, Any] | None = None,
            print_system_info: bool = False,
            force_reset: bool = True,
            **kwargs: Any,
        ) -> "ConstrainedPPO":
            caller_rng_state = LearnerRNGStateV1.capture()
            artifact_sha256_before = _policy_artifact_sha256(path)
            try:
                model = cast(
                    ConstrainedPPO,
                    super().load(
                        path,
                        env=env,
                        device=device,
                        custom_objects=custom_objects,
                        print_system_info=print_system_info,
                        force_reset=force_reset,
                        **kwargs,
                    ),
                )
                model._loaded_force_reset = force_reset
                model._validate_embedded_constraint_state()
                artifact_sha256_after = _policy_artifact_sha256(path)
                if artifact_sha256_before != artifact_sha256_after:
                    raise ValueError("policy artifact changed while it was loading")
                model._loaded_policy_artifact_sha256 = artifact_sha256_after
                if model._episode_cost_accumulator is None:  # pragma: no cover
                    raise RuntimeError("loaded artifact omitted episode accumulator")
                model._resume_continuity_required = True
                if model._learner_rng_state is None:
                    raise ValueError("artifact omitted exact learner RNG state")
                return model
            finally:
                # Inference-only loading must not replace process-global RNG.
                # The checkpoint RNG is restored only by explicit admission.
                caller_rng_state.restore()

else:

    class LearnerRNGStateV1:  # pragma: no cover - dependency-error shim
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_training_dependencies()

    class FourHeadCostCriticV1:  # pragma: no cover - dependency-error shim
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_training_dependencies()

    class ConstrainedActorCriticPolicyV1:  # pragma: no cover - error shim
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_training_dependencies()

    class CostRolloutBufferV1:  # pragma: no cover - dependency-error shim
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_training_dependencies()

    class ConstrainedPPO:  # pragma: no cover - dependency-error shim
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_training_dependencies()

        @classmethod
        def load(cls, *_args: Any, **_kwargs: Any) -> Any:
            _require_training_dependencies()


def build_constrained_model(
    env: Any,
    *,
    settings: ConstrainedPPOSettingsV1,
    ppo_kwargs: Mapping[str, Any],
    verbose: int = 1,
) -> ConstrainedPPO:
    """Build one exact family from a settings record and SB3 PPO arguments."""

    _require_training_dependencies()
    settings.validate()
    arguments = dict(ppo_kwargs)
    policy = arguments.pop("policy", "MlpPolicy")
    if "seed" in arguments and arguments["seed"] != settings.optimizer_seed:
        raise ValueError("ppo_kwargs seed disagrees with constrained settings")
    if "constraint_settings" in arguments:
        raise ValueError("constraint_settings must not be hidden in ppo_kwargs")
    if "rollout_buffer_class" in arguments or "rollout_buffer_kwargs" in arguments:
        raise ValueError("the constrained rollout buffer cannot be overridden")
    arguments["seed"] = settings.optimizer_seed
    arguments["verbose"] = verbose
    return ConstrainedPPO(
        policy,
        env,
        constraint_settings=settings,
        **arguments,
    )


def actor_reward_parameter_sha256(model_or_policy: Any) -> str:
    """Digest actor/reward parameters while deliberately excluding cost critic."""

    _require_training_dependencies()
    policy = getattr(model_or_policy, "policy", model_or_policy)
    if not hasattr(policy, "state_dict"):
        raise TypeError("model_or_policy has no Torch state_dict")
    digest = hashlib.sha256(b"actor-reward-parameter-state-v1\x00")
    included = 0
    for name, tensor in sorted(policy.state_dict().items()):
        if name.startswith("cost_critic."):
            continue
        included += 1
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\x00")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\x00")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    if included == 0:
        raise ValueError("policy state contained no actor/reward parameters")
    return digest.hexdigest()


def policy_parameter_sha256(model_or_policy: Any) -> str:
    """Digest every actor, reward-critic, and optional cost-critic tensor."""

    _require_training_dependencies()
    policy = getattr(model_or_policy, "policy", model_or_policy)
    return _policy_parameter_digest(policy)


def optimizer_state_sha256(model_or_policy: Any) -> str:
    """Digest the complete optimizer state and parameter-group configuration."""

    _require_training_dependencies()
    policy = getattr(model_or_policy, "policy", model_or_policy)
    return _optimizer_state_digest(policy)


__all__ = [
    "ABORTED_ROLLOUT_SCHEMA",
    "BETA_DECISIONS_PER_CYCLE",
    "BETA_DECISIONS_PER_EPISODE",
    "BETA_EPISODE_CYCLE_COUNT",
    "CHECKPOINT_OBSERVATION_DIGEST_SCHEMA",
    "CONSTRAINED_ALGORITHM_IDS",
    "CONSTRAINED_SETTINGS_SCHEMA",
    "CONSTRAINT_STATE_ENVELOPE_SCHEMA",
    "CONSTRAINT_STATE_SCHEMA",
    "COST_CHANNEL_COUNT",
    "COST_NETWORK_SEED_OFFSET",
    "EXTERNAL_EPISODE_STATE_PIN_SCHEMA",
    "LEARNER_RNG_STATE_SCHEMA",
    "POLICY_ARCHITECTURE_SCHEMA",
    "PPO_CONFIGURATION_SCHEMA",
    "AbortedRolloutV1",
    "CostRolloutBufferV1",
    "CompletedEpisodeCostV1",
    "ConstrainedActorCriticPolicyV1",
    "ConstrainedPPO",
    "ConstrainedPPOSettingsV1",
    "ConstrainedRolloutBufferSamplesV1",
    "ConstraintTransitionCostV1",
    "EpisodeCostAccumulatorV1",
    "ExternalEpisodeStatePinV1",
    "FourHeadCostCriticV1",
    "I_LAGRANGIAN_ALGORITHM_ID",
    "LearnerRNGStateV1",
    "PI_LAGRANGIAN_ALGORITHM_ID",
    "PILagrangeControllerV1",
    "PILagrangeStateV1",
    "PROTOCOL_ID",
    "REGISTERED_OPTIMIZER_SEEDS",
    "RESUME_CONTINUATION_RECEIPT_SCHEMA",
    "RESOURCE_ORDER",
    "ROLLOUT_LEDGER_SCHEMA",
    "RolloutLedgerV1",
    "ResumeContinuationReceiptV1",
    "SCALAR_PPO_ALGORITHM_ID",
    "SUPPORTED_ALGORITHM_IDS",
    "TRANSITION_INFO_KEY",
    "ValidatedConstraintStateAttestationV1",
    "actor_reward_parameter_sha256",
    "build_constrained_model",
    "combine_advantages",
    "compute_vector_gae",
    "extract_constraint_transition",
    "normalize_advantage",
    "optimizer_state_sha256",
    "policy_parameter_sha256",
    "training_dependencies_available",
]

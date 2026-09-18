"""Deterministic episode-seed custody for constrained beta training.

The constrained-RL protocol deliberately separates optimizer randomness from
environment randomness.  This module owns the latter: four synchronous Gym
environments receive a fixed, resource-independent stream, every reset is
explicitly seeded, and only compact private-manifest identities enter an
immutable custody ledger.

The wrapper is intentionally independent of Stable-Baselines3.  It implements
the Gymnasium API expected by ``DummyVecEnv``, including the unseeded automatic
``reset()`` that follows a terminal transition, while still supplying a fresh
explicit seed to the wrapped plant environment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from numbers import Integral
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import gymnasium as gym


ALLOCATOR_ID = "plant-beta-four-env-episode-seed-allocator-v1"
ALLOCATOR_IDENTITY_SCHEMA = "constrained-training-seed-allocator-identity-v1"
ALLOCATOR_SNAPSHOT_SCHEMA = "constrained-training-seed-allocator-snapshot-v1"
FIXTURE_RESUME_SCHEMA = "constrained-training-fixture-resume-v1"
LEDGER_SCHEMA = "constrained-training-reset-manifest-ledger-v1"
LEDGER_ENTRY_SCHEMA = "constrained-training-reset-manifest-entry-v1"
MANIFEST_IDENTITY_SCHEMA = "beta-private-manifest-identity-v1"
RESERVATION_SCHEMA = "constrained-training-reset-reservation-v1"

VECTOR_ENV_COUNT = 4
REGISTERED_EPISODE_DECISIONS = 672

CONTAMINATED_SEED_MIN = 0
CONTAMINATED_SEED_MAX = 999_999
TRAINING_SEED_MIN = 1_000_000
TRAINING_SEED_MAX = 1_999_999
DEVELOPMENT_SEED_MIN = 2_000_000
DEVELOPMENT_SEED_MAX = 2_000_031
HELDOUT_SEED_MIN = 3_000_000
HELDOUT_SEED_MAX = 3_000_031
FIXTURE_SEED_MIN = 9_000_000
FIXTURE_SEED_MAX = 9_999_999

_SHA256_HEX_LENGTH = 64
_MANIFEST_DIGEST_KEYS = (
    "manifest_sha256",
    "alpha_physical_exogenous_episode_sha256",
    "beta_exogenous_episode_sha256",
)
_MAPPING_DESCRIPTION = (
    "episode_seed = base_seed + 4 * local_episode_ordinal + environment_index"
)


class EpisodeSeedCustodyError(ValueError):
    """The requested operation would break episode-seed custody."""


class EpisodeSeedExhaustedError(EpisodeSeedCustodyError):
    """The next vector-mapped seed lies outside its frozen split."""


class RegisteredResumeForbiddenError(EpisodeSeedCustodyError):
    """Registered training cannot import or export resumable allocator state."""


class EpisodeSeedSplitV1(str, Enum):
    """Frozen, disjoint seed purposes from the constrained-RL beta protocol."""

    CONTAMINATED = "contaminated"
    TRAINING = "training"
    DEVELOPMENT = "development"
    HELDOUT = "heldout"
    FIXTURE = "fixture"


class EpisodeAllocatorModeV1(str, Enum):
    """Only fixture mode permits allocator resume."""

    REGISTERED = "registered"
    FIXTURE = "fixture"


@dataclass(frozen=True, slots=True)
class EpisodeSeedDomainV1:
    """Inclusive seed interval assigned to one protocol purpose."""

    split: EpisodeSeedSplitV1
    first_seed: int
    last_seed: int

    def __post_init__(self) -> None:
        _integer("first_seed", self.first_seed, minimum=0)
        _integer("last_seed", self.last_seed, minimum=self.first_seed)

    @property
    def size(self) -> int:
        return self.last_seed - self.first_seed + 1

    def contains(self, seed: object) -> bool:
        return (
            isinstance(seed, Integral)
            and not isinstance(seed, bool)
            and self.first_seed <= int(seed) <= self.last_seed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split.value,
            "first_seed": self.first_seed,
            "last_seed": self.last_seed,
            "size": self.size,
        }


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as reason:
        raise EpisodeSeedCustodyError(
            f"seed-custody evidence is not canonical JSON: {reason}"
        ) from reason


def canonical_sha256(value: object) -> str:
    """Return the protocol's canonical JSON digest."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _integer(name: str, value: object, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise EpisodeSeedCustodyError(f"{name} must be >= {minimum}")
    return result


def _boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _deep_detach_json(value: object) -> Any:
    """Return a fresh JSON tree with no mutable aliases to ``value``."""

    if isinstance(value, Mapping):
        return {str(key): _deep_detach_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_deep_detach_json(item) for item in value]
    return value


def _deep_freeze_json(value: object) -> object:
    """Recursively freeze a detached canonical JSON tree."""

    detached = _deep_detach_json(value)

    def freeze(item: object) -> object:
        if isinstance(item, Mapping):
            return MappingProxyType({key: freeze(child) for key, child in item.items()})
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        return item

    return freeze(detached)


def _sequence(name: str, value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    return value


def _exact_keys(name: str, value: Mapping[str, Any], expected: set[str]) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing:
        raise EpisodeSeedCustodyError(
            f"{name} is missing fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise EpisodeSeedCustodyError(
            f"{name} has unknown fields: {', '.join(sorted(unknown))}"
        )


def _sha256(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA-256 string")
    if len(value) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise EpisodeSeedCustodyError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    return value


SEED_DOMAINS: tuple[EpisodeSeedDomainV1, ...] = (
    EpisodeSeedDomainV1(
        EpisodeSeedSplitV1.CONTAMINATED,
        CONTAMINATED_SEED_MIN,
        CONTAMINATED_SEED_MAX,
    ),
    EpisodeSeedDomainV1(
        EpisodeSeedSplitV1.TRAINING,
        TRAINING_SEED_MIN,
        TRAINING_SEED_MAX,
    ),
    EpisodeSeedDomainV1(
        EpisodeSeedSplitV1.DEVELOPMENT,
        DEVELOPMENT_SEED_MIN,
        DEVELOPMENT_SEED_MAX,
    ),
    EpisodeSeedDomainV1(
        EpisodeSeedSplitV1.HELDOUT,
        HELDOUT_SEED_MIN,
        HELDOUT_SEED_MAX,
    ),
    EpisodeSeedDomainV1(
        EpisodeSeedSplitV1.FIXTURE,
        FIXTURE_SEED_MIN,
        FIXTURE_SEED_MAX,
    ),
)
_DOMAIN_BY_SPLIT = {domain.split: domain for domain in SEED_DOMAINS}


def _validate_identity(value: Mapping[str, Any]) -> None:
    expected = {
        "schema",
        "allocator_id",
        "mode",
        "split",
        "base_seed",
        "seed_domain",
        "vector_environment_count",
        "episode_decisions",
        "mapping",
        "sha256",
    }
    _exact_keys("allocator identity", value, expected)
    if value["schema"] != ALLOCATOR_IDENTITY_SCHEMA:
        raise EpisodeSeedCustodyError("allocator identity schema differs")
    if value["allocator_id"] != ALLOCATOR_ID:
        raise EpisodeSeedCustodyError("allocator implementation identity differs")
    try:
        mode = EpisodeAllocatorModeV1(value["mode"])
        split = EpisodeSeedSplitV1(value["split"])
    except (TypeError, ValueError) as reason:
        raise EpisodeSeedCustodyError("allocator mode or split is invalid") from reason
    if (
        mode is EpisodeAllocatorModeV1.REGISTERED
        and split is not EpisodeSeedSplitV1.TRAINING
    ):
        raise EpisodeSeedCustodyError(
            "registered allocator identity has a non-training split"
        )
    if (
        mode is EpisodeAllocatorModeV1.FIXTURE
        and split is not EpisodeSeedSplitV1.FIXTURE
    ):
        raise EpisodeSeedCustodyError(
            "fixture allocator identity has a non-fixture split"
        )
    base_seed = _integer("identity base_seed", value["base_seed"], minimum=0)
    episode_decisions = _integer(
        "identity episode_decisions", value["episode_decisions"], minimum=1
    )
    if (
        _integer(
            "identity vector_environment_count",
            value["vector_environment_count"],
            minimum=1,
        )
        != VECTOR_ENV_COUNT
    ):
        raise EpisodeSeedCustodyError("allocator vector-environment count differs")
    if value["mapping"] != _MAPPING_DESCRIPTION:
        raise EpisodeSeedCustodyError("allocator seed mapping differs")
    if (
        dict(_mapping("identity seed_domain", value["seed_domain"]))
        != seed_domain(split).to_dict()
    ):
        raise EpisodeSeedCustodyError("allocator seed-domain identity differs")
    domain = seed_domain(split)
    if not domain.contains(base_seed):
        raise EpisodeSeedCustodyError("allocator base seed crosses its split")
    if mode is EpisodeAllocatorModeV1.REGISTERED and (
        base_seed != TRAINING_SEED_MIN
        or episode_decisions != REGISTERED_EPISODE_DECISIONS
    ):
        raise EpisodeSeedCustodyError("registered allocator identity is not frozen")
    supplied = _sha256("allocator identity sha256", value["sha256"])
    body = dict(value)
    body.pop("sha256")
    if canonical_sha256(body) != supplied:
        raise EpisodeSeedCustodyError("allocator identity digest differs")


def seed_domain(split: EpisodeSeedSplitV1 | str) -> EpisodeSeedDomainV1:
    """Return the frozen interval for ``split``."""

    try:
        normalized = EpisodeSeedSplitV1(split)
    except ValueError as reason:
        raise EpisodeSeedCustodyError(
            f"unknown episode seed split {split!r}"
        ) from reason
    return _DOMAIN_BY_SPLIT[normalized]


def classify_episode_seed(seed: object) -> EpisodeSeedSplitV1 | None:
    """Classify a seed, returning ``None`` for deliberately unassigned gaps."""

    normalized = _integer("episode seed", seed, minimum=0)
    for domain in SEED_DOMAINS:
        if domain.contains(normalized):
            return domain.split
    return None


def episode_seed_for(
    *,
    base_seed: object,
    environment_index: object,
    local_episode_ordinal: object,
) -> int:
    """Apply the frozen four-environment map ``base + 4*k + j``."""

    base = _integer("base_seed", base_seed, minimum=0)
    index = _integer("environment_index", environment_index, minimum=0)
    ordinal = _integer("local_episode_ordinal", local_episode_ordinal, minimum=0)
    if index >= VECTOR_ENV_COUNT:
        raise EpisodeSeedCustodyError(
            f"environment_index must be in 0..{VECTOR_ENV_COUNT - 1}"
        )
    return base + VECTOR_ENV_COUNT * ordinal + index


@dataclass(frozen=True, slots=True)
class BetaManifestIdentityV1:
    """The immutable identity subset of one private beta audit manifest."""

    manifest_sha256: str
    alpha_physical_exogenous_episode_sha256: str
    beta_exogenous_episode_sha256: str

    def __post_init__(self) -> None:
        for name in _MANIFEST_DIGEST_KEYS:
            _sha256(name, getattr(self, name))

    def body(self) -> dict[str, Any]:
        return {
            "schema": MANIFEST_IDENTITY_SCHEMA,
            **{name: getattr(self, name) for name in _MANIFEST_DIGEST_KEYS},
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}

    @classmethod
    def from_audit_manifest(
        cls,
        manifest: object,
        *,
        expected_episode_seed: int,
    ) -> BetaManifestIdentityV1:
        source = _mapping("beta audit manifest", manifest)
        episode = _mapping("beta audit manifest episode", source.get("episode"))
        observed_seed = _integer(
            "beta audit manifest episode seed", episode.get("seed")
        )
        if observed_seed != expected_episode_seed:
            raise EpisodeSeedCustodyError(
                "beta audit manifest seed does not match the allocator reservation: "
                f"expected {expected_episode_seed}, observed {observed_seed}"
            )
        return cls(
            **{
                name: _sha256(f"beta audit manifest {name}", source.get(name))
                for name in _MANIFEST_DIGEST_KEYS
            }
        )

    @classmethod
    def from_dict(cls, value: object) -> BetaManifestIdentityV1:
        raw = _mapping("manifest identity", value)
        _exact_keys(
            "manifest identity",
            raw,
            {"schema", *_MANIFEST_DIGEST_KEYS, "sha256"},
        )
        if raw["schema"] != MANIFEST_IDENTITY_SCHEMA:
            raise EpisodeSeedCustodyError("manifest identity schema differs")
        result = cls(
            **{
                name: _sha256(f"manifest identity {name}", raw[name])
                for name in _MANIFEST_DIGEST_KEYS
            }
        )
        if _sha256("manifest identity sha256", raw["sha256"]) != result.sha256:
            raise EpisodeSeedCustodyError("manifest identity digest differs")
        return result


@dataclass(frozen=True, slots=True)
class ResetReservationV1:
    """A single-use reservation made before calling the underlying reset."""

    allocator_identity_sha256: str
    environment_index: int
    local_episode_ordinal: int
    global_episode_ordinal: int
    episode_seed: int
    ignored_framework_seed: int | None

    def __post_init__(self) -> None:
        _sha256("allocator_identity_sha256", self.allocator_identity_sha256)
        _integer("environment_index", self.environment_index, minimum=0)
        _integer("local_episode_ordinal", self.local_episode_ordinal, minimum=0)
        _integer("global_episode_ordinal", self.global_episode_ordinal, minimum=0)
        _integer("episode_seed", self.episode_seed, minimum=0)
        if self.ignored_framework_seed is not None:
            _integer("ignored_framework_seed", self.ignored_framework_seed, minimum=0)

    def body(self) -> dict[str, Any]:
        return {
            "schema": RESERVATION_SCHEMA,
            "allocator_identity_sha256": self.allocator_identity_sha256,
            "environment_index": self.environment_index,
            "local_episode_ordinal": self.local_episode_ordinal,
            "global_episode_ordinal": self.global_episode_ordinal,
            "episode_seed": self.episode_seed,
            "ignored_framework_seed": self.ignored_framework_seed,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.body())


@dataclass(frozen=True, slots=True)
class _FixtureReattachmentPlanV1:
    environment_index: int
    episode_seed: int
    ignored_framework_seed: int | None
    expected_manifest_identity: BetaManifestIdentityV1


@dataclass(frozen=True, slots=True)
class ResetManifestLedgerEntryV1:
    """One successful explicit reset bound to its private manifest identity."""

    ledger_index: int
    allocator_identity_sha256: str
    split: EpisodeSeedSplitV1
    environment_index: int
    local_episode_ordinal: int
    global_episode_ordinal: int
    episode_seed: int
    ignored_framework_seed: int | None
    manifest_identity: BetaManifestIdentityV1

    def __post_init__(self) -> None:
        _integer("ledger_index", self.ledger_index, minimum=0)
        _sha256("allocator_identity_sha256", self.allocator_identity_sha256)
        _integer("environment_index", self.environment_index, minimum=0)
        _integer("local_episode_ordinal", self.local_episode_ordinal, minimum=0)
        _integer("global_episode_ordinal", self.global_episode_ordinal, minimum=0)
        _integer("episode_seed", self.episode_seed, minimum=0)
        if self.ignored_framework_seed is not None:
            _integer("ignored_framework_seed", self.ignored_framework_seed, minimum=0)
        if not isinstance(self.split, EpisodeSeedSplitV1):
            raise TypeError("split must be an EpisodeSeedSplitV1")
        if not isinstance(self.manifest_identity, BetaManifestIdentityV1):
            raise TypeError("manifest_identity must be a BetaManifestIdentityV1")

    def body(self) -> dict[str, Any]:
        return {
            "schema": LEDGER_ENTRY_SCHEMA,
            "ledger_index": self.ledger_index,
            "allocator_identity_sha256": self.allocator_identity_sha256,
            "split": self.split.value,
            "environment_index": self.environment_index,
            "local_episode_ordinal": self.local_episode_ordinal,
            "global_episode_ordinal": self.global_episode_ordinal,
            "episode_seed": self.episode_seed,
            "ignored_framework_seed": self.ignored_framework_seed,
            "manifest_identity": self.manifest_identity.to_dict(),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: object) -> ResetManifestLedgerEntryV1:
        raw = _mapping("reset-manifest ledger entry", value)
        expected = {
            "schema",
            "ledger_index",
            "allocator_identity_sha256",
            "split",
            "environment_index",
            "local_episode_ordinal",
            "global_episode_ordinal",
            "episode_seed",
            "ignored_framework_seed",
            "manifest_identity",
            "sha256",
        }
        _exact_keys("reset-manifest ledger entry", raw, expected)
        if raw["schema"] != LEDGER_ENTRY_SCHEMA:
            raise EpisodeSeedCustodyError("reset-manifest ledger entry schema differs")
        try:
            split = EpisodeSeedSplitV1(raw["split"])
        except (TypeError, ValueError) as reason:
            raise EpisodeSeedCustodyError(
                "reset-manifest ledger entry split is invalid"
            ) from reason
        ignored_framework_seed = raw["ignored_framework_seed"]
        if ignored_framework_seed is not None:
            ignored_framework_seed = _integer(
                "ignored_framework_seed", ignored_framework_seed, minimum=0
            )
        result = cls(
            ledger_index=_integer("ledger_index", raw["ledger_index"], minimum=0),
            allocator_identity_sha256=_sha256(
                "allocator_identity_sha256", raw["allocator_identity_sha256"]
            ),
            split=split,
            environment_index=_integer(
                "environment_index", raw["environment_index"], minimum=0
            ),
            local_episode_ordinal=_integer(
                "local_episode_ordinal", raw["local_episode_ordinal"], minimum=0
            ),
            global_episode_ordinal=_integer(
                "global_episode_ordinal", raw["global_episode_ordinal"], minimum=0
            ),
            episode_seed=_integer("episode_seed", raw["episode_seed"], minimum=0),
            ignored_framework_seed=ignored_framework_seed,
            manifest_identity=BetaManifestIdentityV1.from_dict(
                raw["manifest_identity"]
            ),
        )
        if _sha256("ledger entry sha256", raw["sha256"]) != result.sha256:
            raise EpisodeSeedCustodyError("reset-manifest ledger entry digest differs")
        return result


@dataclass(frozen=True, slots=True)
class EpisodeCursorV1:
    """Exact position of one vector environment in its episode stream."""

    environment_index: int
    current_episode_ordinal: int | None
    decision_offset: int
    next_episode_ordinal: int
    awaiting_reset: bool

    def __post_init__(self) -> None:
        _integer("environment_index", self.environment_index, minimum=0)
        if self.current_episode_ordinal is not None:
            _integer(
                "current_episode_ordinal",
                self.current_episode_ordinal,
                minimum=0,
            )
        _integer("decision_offset", self.decision_offset, minimum=0)
        _integer("next_episode_ordinal", self.next_episode_ordinal, minimum=0)
        _boolean("awaiting_reset", self.awaiting_reset)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_index": self.environment_index,
            "current_episode_ordinal": self.current_episode_ordinal,
            "decision_offset": self.decision_offset,
            "next_episode_ordinal": self.next_episode_ordinal,
            "awaiting_reset": self.awaiting_reset,
        }

    @classmethod
    def from_dict(cls, value: object) -> EpisodeCursorV1:
        raw = _mapping("episode cursor", value)
        expected = {
            "environment_index",
            "current_episode_ordinal",
            "decision_offset",
            "next_episode_ordinal",
            "awaiting_reset",
        }
        _exact_keys("episode cursor", raw, expected)
        current = raw["current_episode_ordinal"]
        if current is not None:
            current = _integer("current_episode_ordinal", current, minimum=0)
        return cls(
            environment_index=_integer(
                "environment_index", raw["environment_index"], minimum=0
            ),
            current_episode_ordinal=current,
            decision_offset=_integer(
                "decision_offset", raw["decision_offset"], minimum=0
            ),
            next_episode_ordinal=_integer(
                "next_episode_ordinal", raw["next_episode_ordinal"], minimum=0
            ),
            awaiting_reset=_boolean("awaiting_reset", raw["awaiting_reset"]),
        )


@dataclass(frozen=True, slots=True)
class EpisodeAllocatorSnapshotV1:
    """Immutable checkpoint evidence for one allocator."""

    allocator_identity: Mapping[str, Any]
    cursors: tuple[EpisodeCursorV1, ...]
    ledger: tuple[ResetManifestLedgerEntryV1, ...]

    def __post_init__(self) -> None:
        identity = _mapping("snapshot allocator identity", self.allocator_identity)
        _validate_identity(identity)
        frozen_identity = _deep_freeze_json(identity)
        if not isinstance(frozen_identity, Mapping):  # pragma: no cover - defensive
            raise TypeError("frozen allocator identity must remain a mapping")
        object.__setattr__(self, "allocator_identity", frozen_identity)
        if not isinstance(self.cursors, tuple) or any(
            not isinstance(cursor, EpisodeCursorV1) for cursor in self.cursors
        ):
            raise TypeError("snapshot cursors must be an immutable cursor tuple")
        if not isinstance(self.ledger, tuple) or any(
            not isinstance(entry, ResetManifestLedgerEntryV1) for entry in self.ledger
        ):
            raise TypeError("snapshot ledger must be an immutable entry tuple")

    def _ledger_body(self) -> dict[str, Any]:
        return {
            "schema": LEDGER_SCHEMA,
            "allocator_identity_sha256": self.allocator_identity["sha256"],
            "entries": [entry.to_dict() for entry in self.ledger],
        }

    @property
    def ledger_sha256(self) -> str:
        return canonical_sha256(self._ledger_body())

    def body(self) -> dict[str, Any]:
        return {
            "schema": ALLOCATOR_SNAPSHOT_SCHEMA,
            "allocator_identity": _deep_detach_json(self.allocator_identity),
            "cursors": [cursor.to_dict() for cursor in self.cursors],
            "ledger": {
                **self._ledger_body(),
                "sha256": self.ledger_sha256,
            },
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


class EpisodeSeedAllocatorV1:
    """Stateful four-stream allocator with fail-closed reset custody."""

    def __init__(
        self,
        *,
        mode: EpisodeAllocatorModeV1,
        split: EpisodeSeedSplitV1,
        base_seed: int,
        episode_decisions: int,
    ) -> None:
        if not isinstance(mode, EpisodeAllocatorModeV1):
            raise TypeError("mode must be an EpisodeAllocatorModeV1")
        if not isinstance(split, EpisodeSeedSplitV1):
            raise TypeError("split must be an EpisodeSeedSplitV1")
        self._mode = mode
        self._split = split
        self._domain = seed_domain(split)
        self._base_seed = _integer("base_seed", base_seed, minimum=0)
        self._episode_decisions = _integer(
            "episode_decisions", episode_decisions, minimum=1
        )
        if mode is EpisodeAllocatorModeV1.REGISTERED:
            if split is not EpisodeSeedSplitV1.TRAINING:
                raise EpisodeSeedCustodyError(
                    "registered allocator mode requires the training split"
                )
            if self._base_seed != TRAINING_SEED_MIN:
                raise EpisodeSeedCustodyError(
                    "registered allocator mode requires the frozen training base"
                )
            if self._episode_decisions != REGISTERED_EPISODE_DECISIONS:
                raise EpisodeSeedCustodyError(
                    "registered allocator mode requires the frozen 672 decisions"
                )
        elif split is not EpisodeSeedSplitV1.FIXTURE:
            raise EpisodeSeedCustodyError(
                "fixture allocator mode requires the fixture split"
            )
        if not self._domain.contains(self._base_seed):
            raise EpisodeSeedCustodyError(
                f"base_seed is outside the {split.value} seed domain"
            )
        first_wave_last = episode_seed_for(
            base_seed=self._base_seed,
            environment_index=VECTOR_ENV_COUNT - 1,
            local_episode_ordinal=0,
        )
        if not self._domain.contains(first_wave_last):
            raise EpisodeSeedCustodyError(
                "base_seed does not leave room for all four initial environments"
            )
        self._cursors = [
            EpisodeCursorV1(
                environment_index=index,
                current_episode_ordinal=None,
                decision_offset=0,
                next_episode_ordinal=0,
                awaiting_reset=True,
            )
            for index in range(VECTOR_ENV_COUNT)
        ]
        self._ledger: list[ResetManifestLedgerEntryV1] = []
        self._pending: dict[int, ResetReservationV1] = {}
        self._used_seeds: set[int] = set()
        self._reattachment_required: set[int] = set()
        self._pending_reattachment: set[int] = set()
        self._failed_reason: str | None = None

    @classmethod
    def registered(
        cls,
        *,
        resume_from: object | None = None,
    ) -> EpisodeSeedAllocatorV1:
        """Create the sole registered-training allocator; resume is impossible."""

        if resume_from is not None:
            raise RegisteredResumeForbiddenError(
                "registered training always starts from four zero cursors"
            )
        return cls(
            mode=EpisodeAllocatorModeV1.REGISTERED,
            split=EpisodeSeedSplitV1.TRAINING,
            base_seed=TRAINING_SEED_MIN,
            episode_decisions=REGISTERED_EPISODE_DECISIONS,
        )

    @classmethod
    def fixture(
        cls,
        *,
        base_seed: int = FIXTURE_SEED_MIN,
        episode_decisions: int = REGISTERED_EPISODE_DECISIONS,
    ) -> EpisodeSeedAllocatorV1:
        """Create an implementation-only allocator in the fixture domain."""

        return cls(
            mode=EpisodeAllocatorModeV1.FIXTURE,
            split=EpisodeSeedSplitV1.FIXTURE,
            base_seed=base_seed,
            episode_decisions=episode_decisions,
        )

    @property
    def mode(self) -> EpisodeAllocatorModeV1:
        return self._mode

    @property
    def split(self) -> EpisodeSeedSplitV1:
        return self._split

    @property
    def base_seed(self) -> int:
        return self._base_seed

    @property
    def episode_decisions(self) -> int:
        return self._episode_decisions

    @property
    def identity(self) -> dict[str, Any]:
        body = {
            "schema": ALLOCATOR_IDENTITY_SCHEMA,
            "allocator_id": ALLOCATOR_ID,
            "mode": self._mode.value,
            "split": self._split.value,
            "base_seed": self._base_seed,
            "seed_domain": self._domain.to_dict(),
            "vector_environment_count": VECTOR_ENV_COUNT,
            "episode_decisions": self._episode_decisions,
            "mapping": _MAPPING_DESCRIPTION,
        }
        return {**body, "sha256": canonical_sha256(body)}

    @property
    def identity_sha256(self) -> str:
        return str(self.identity["sha256"])

    @property
    def cursors(self) -> tuple[EpisodeCursorV1, ...]:
        return tuple(self._cursors)

    @property
    def ledger(self) -> tuple[ResetManifestLedgerEntryV1, ...]:
        return tuple(self._ledger)

    @property
    def decisions_consumed_by_environment(self) -> tuple[int, ...]:
        """Return the exact transition count implied by every cursor."""

        return tuple(
            (
                0
                if cursor.current_episode_ordinal is None
                else cursor.current_episode_ordinal * self._episode_decisions
                + cursor.decision_offset
            )
            for cursor in self._cursors
        )

    def seed_for(self, environment_index: int, local_episode_ordinal: int) -> int:
        seed = episode_seed_for(
            base_seed=self._base_seed,
            environment_index=environment_index,
            local_episode_ordinal=local_episode_ordinal,
        )
        if not self._domain.contains(seed):
            raise EpisodeSeedExhaustedError(
                f"mapped seed {seed} crosses the {self._split.value} split boundary"
            )
        return seed

    def _usable(self) -> None:
        if self._failed_reason is not None:
            raise EpisodeSeedCustodyError(
                f"episode allocator is failed closed: {self._failed_reason}"
            )

    def _index(self, environment_index: object) -> int:
        index = _integer("environment_index", environment_index, minimum=0)
        if index >= VECTOR_ENV_COUNT:
            raise EpisodeSeedCustodyError(
                f"environment_index must be in 0..{VECTOR_ENV_COUNT - 1}"
            )
        return index

    def _normalize_framework_seed(self, value: object | None) -> int | None:
        if value is None:
            return None
        seed = _integer("ignored framework seed", value, minimum=0)
        if classify_episode_seed(seed) is not EpisodeSeedSplitV1.CONTAMINATED:
            raise EpisodeSeedCustodyError(
                "a framework reset seed is accepted only from the contaminated "
                "episode domain and is never used for episode selection"
            )
        return seed

    def _validate_initial_framework_seed(
        self,
        environment_index: int,
        ignored_framework_seed: int | None,
    ) -> None:
        initial_entries = [
            entry for entry in self._ledger if entry.local_episode_ordinal == 0
        ]
        if not initial_entries:
            if (
                ignored_framework_seed is not None
                and ignored_framework_seed - environment_index < 0
            ):
                raise EpisodeSeedCustodyError("framework seed base cannot be negative")
            return
        first = initial_entries[0]
        if first.ignored_framework_seed is None:
            if ignored_framework_seed is not None:
                raise EpisodeSeedCustodyError(
                    "initial vector resets mix absent and supplied framework seeds"
                )
            return
        expected_base = first.ignored_framework_seed - first.environment_index
        expected = expected_base + environment_index
        if ignored_framework_seed != expected:
            raise EpisodeSeedCustodyError(
                "initial vector framework seeds must be one common base plus "
                "environment index"
            )

    def begin_reset(
        self,
        environment_index: int,
        *,
        ignored_framework_seed: object | None = None,
    ) -> ResetReservationV1:
        """Reserve the only legal seed before an underlying Gym reset."""

        self._usable()
        index = self._index(environment_index)
        cursor = self._cursors[index]
        if not cursor.awaiting_reset:
            raise EpisodeSeedCustodyError(
                f"environment {index} reset before its current episode terminated"
            )
        if index in self._pending:
            raise EpisodeSeedCustodyError(
                f"environment {index} already has a pending reset reservation"
            )
        framework_seed = self._normalize_framework_seed(ignored_framework_seed)
        if framework_seed is not None and cursor.current_episode_ordinal is not None:
            raise EpisodeSeedCustodyError(
                "a framework seed may be absorbed only on the initial vector reset"
            )
        if cursor.current_episode_ordinal is None:
            self._validate_initial_framework_seed(index, framework_seed)
        ordinal = cursor.next_episode_ordinal
        seed = self.seed_for(index, ordinal)
        if seed in self._used_seeds or any(
            pending.episode_seed == seed for pending in self._pending.values()
        ):
            raise EpisodeSeedCustodyError(f"episode seed {seed} was already consumed")
        reservation = ResetReservationV1(
            allocator_identity_sha256=self.identity_sha256,
            environment_index=index,
            local_episode_ordinal=ordinal,
            global_episode_ordinal=VECTOR_ENV_COUNT * ordinal + index,
            episode_seed=seed,
            ignored_framework_seed=framework_seed,
        )
        self._pending[index] = reservation
        return reservation

    def commit_reset(
        self,
        reservation: ResetReservationV1,
        manifest_identity: BetaManifestIdentityV1,
    ) -> ResetManifestLedgerEntryV1:
        """Commit a successful reset and advance one environment cursor."""

        self._usable()
        if not isinstance(reservation, ResetReservationV1):
            raise TypeError("reservation must be a ResetReservationV1")
        if not isinstance(manifest_identity, BetaManifestIdentityV1):
            raise TypeError("manifest_identity must be a BetaManifestIdentityV1")
        pending = self._pending.get(reservation.environment_index)
        if pending is None or pending.sha256 != reservation.sha256:
            raise EpisodeSeedCustodyError(
                "reset reservation is missing, stale, or foreign"
            )
        if reservation.episode_seed in self._used_seeds:
            raise EpisodeSeedCustodyError(
                f"episode seed {reservation.episode_seed} was already consumed"
            )
        entry = ResetManifestLedgerEntryV1(
            ledger_index=len(self._ledger),
            allocator_identity_sha256=self.identity_sha256,
            split=self._split,
            environment_index=reservation.environment_index,
            local_episode_ordinal=reservation.local_episode_ordinal,
            global_episode_ordinal=reservation.global_episode_ordinal,
            episode_seed=reservation.episode_seed,
            ignored_framework_seed=reservation.ignored_framework_seed,
            manifest_identity=manifest_identity,
        )
        self._ledger.append(entry)
        self._used_seeds.add(reservation.episode_seed)
        del self._pending[reservation.environment_index]
        self._cursors[reservation.environment_index] = EpisodeCursorV1(
            environment_index=reservation.environment_index,
            current_episode_ordinal=reservation.local_episode_ordinal,
            decision_offset=0,
            next_episode_ordinal=reservation.local_episode_ordinal + 1,
            awaiting_reset=False,
        )
        return entry

    def fixture_reattachment_required(self, environment_index: int) -> bool:
        """Whether a restored live environment must replay its opening reset."""

        return self._index(environment_index) in self._reattachment_required

    def begin_fixture_reattachment(
        self,
        environment_index: int,
        *,
        ignored_framework_seed: object | None = None,
    ) -> _FixtureReattachmentPlanV1:
        """Prepare a no-ledger replay of an offset-zero fixture reset."""

        self._usable()
        index = self._index(environment_index)
        if self._mode is not EpisodeAllocatorModeV1.FIXTURE:
            raise RegisteredResumeForbiddenError(
                "registered training cannot reattach allocator state"
            )
        if index not in self._reattachment_required:
            raise EpisodeSeedCustodyError(
                f"environment {index} has no fixture reattachment pending"
            )
        if index in self._pending_reattachment:
            raise EpisodeSeedCustodyError(
                f"environment {index} already has a pending reattachment"
            )
        cursor = self._cursors[index]
        if cursor.current_episode_ordinal is None or cursor.awaiting_reset:
            raise EpisodeSeedCustodyError(
                f"environment {index} has no active fixture episode to reattach"
            )
        if cursor.decision_offset != 0:
            reason = EpisodeSeedCustodyError(
                f"environment {index} cannot reattach at decision offset "
                f"{cursor.decision_offset}; physical state and action history are "
                "not present"
            )
            self.fail_closed(reason)
            raise reason
        matches = [
            entry
            for entry in self._ledger
            if entry.environment_index == index
            and entry.local_episode_ordinal == cursor.current_episode_ordinal
        ]
        if len(matches) != 1:
            raise EpisodeSeedCustodyError(
                f"environment {index} cannot identify its current ledger entry"
            )
        entry = matches[0]
        framework_seed = self._normalize_framework_seed(ignored_framework_seed)
        if framework_seed != entry.ignored_framework_seed:
            raise EpisodeSeedCustodyError(
                "fixture reattachment framework seed differs from the audited "
                "opening reset"
            )
        self._pending_reattachment.add(index)
        return _FixtureReattachmentPlanV1(
            environment_index=index,
            episode_seed=entry.episode_seed,
            ignored_framework_seed=framework_seed,
            expected_manifest_identity=entry.manifest_identity,
        )

    def commit_fixture_reattachment(
        self,
        plan: _FixtureReattachmentPlanV1,
        manifest_identity: BetaManifestIdentityV1,
    ) -> None:
        """Accept a replay only when it regenerates the identical manifest."""

        self._usable()
        if not isinstance(plan, _FixtureReattachmentPlanV1):
            raise TypeError("plan must be a fixture reattachment plan")
        if plan.environment_index not in self._pending_reattachment:
            raise EpisodeSeedCustodyError("fixture reattachment plan is stale")
        if manifest_identity != plan.expected_manifest_identity:
            raise EpisodeSeedCustodyError(
                "fixture reattachment regenerated a different manifest identity"
            )
        self._pending_reattachment.remove(plan.environment_index)
        self._reattachment_required.remove(plan.environment_index)

    def fail_closed(self, reason: object) -> None:
        """Make a partially executed reset/step permanently unusable."""

        text = str(reason).strip() or type(reason).__name__
        if self._failed_reason is None:
            self._failed_reason = text

    def record_transition(
        self,
        environment_index: int,
        *,
        terminated: bool,
        truncated: bool,
    ) -> EpisodeCursorV1:
        """Advance the exact decision offset after one successful Gym step."""

        self._usable()
        index = self._index(environment_index)
        terminal = _boolean("terminated", terminated)
        cutoff = _boolean("truncated", truncated)
        cursor = self._cursors[index]
        if index in self._reattachment_required or index in self._pending_reattachment:
            reason = EpisodeSeedCustodyError(
                f"environment {index} must reattach before recording transitions"
            )
            self.fail_closed(reason)
            raise reason
        if cursor.current_episode_ordinal is None or cursor.awaiting_reset:
            raise EpisodeSeedCustodyError(
                f"environment {index} stepped without an active seeded episode"
            )
        if index in self._pending:
            raise EpisodeSeedCustodyError(
                f"environment {index} stepped during a pending reset"
            )
        next_offset = cursor.decision_offset + 1
        try:
            if terminal and cutoff:
                raise EpisodeSeedCustodyError(
                    "a beta episode cannot be both terminated and truncated"
                )
            if cutoff:
                raise EpisodeSeedCustodyError(
                    "the finite-horizon beta task cannot truncate"
                )
            if terminal and next_offset != self._episode_decisions:
                raise EpisodeSeedCustodyError(
                    f"environment {index} terminated at decision {next_offset}; "
                    f"expected {self._episode_decisions}"
                )
            if not terminal and next_offset >= self._episode_decisions:
                raise EpisodeSeedCustodyError(
                    f"environment {index} did not terminate at its exact "
                    f"{self._episode_decisions}-decision horizon"
                )
        except EpisodeSeedCustodyError as reason:
            self.fail_closed(reason)
            raise
        updated = replace(
            cursor,
            decision_offset=next_offset,
            awaiting_reset=terminal,
        )
        self._cursors[index] = updated
        return updated

    def snapshot(self) -> EpisodeAllocatorSnapshotV1:
        """Return immutable checkpoint evidence without authorizing resume."""

        self._usable()
        if self._pending or self._pending_reattachment:
            raise EpisodeSeedCustodyError(
                "cannot snapshot while an environment reset or reattachment is pending"
            )
        self._validate_state(self._cursors, self._ledger)
        return EpisodeAllocatorSnapshotV1(
            allocator_identity=self.identity,
            cursors=tuple(self._cursors),
            ledger=tuple(self._ledger),
        )

    def export_fixture_resume(
        self,
        *,
        parent_checkpoint_sha256: str,
    ) -> dict[str, Any]:
        """Export fixture state bound to an external parent checkpoint."""

        if self._mode is EpisodeAllocatorModeV1.REGISTERED:
            raise RegisteredResumeForbiddenError(
                "registered training forbids allocator resume export"
            )
        parent_sha256 = _sha256("parent checkpoint sha256", parent_checkpoint_sha256)
        snapshot = self.snapshot().to_dict()
        body = {
            "schema": FIXTURE_RESUME_SCHEMA,
            "fixture_only": True,
            "eligible_for_registered_evaluation": False,
            "parent_checkpoint_sha256": parent_sha256,
            "snapshot": snapshot,
        }
        return {**body, "sha256": canonical_sha256(body)}

    @classmethod
    def restore_fixture(
        cls,
        value: object,
        *,
        expected_parent_checkpoint_sha256: str,
        expected_resume_sha256: str,
    ) -> EpisodeSeedAllocatorV1:
        """Restore state pinned by separately admitted checkpoint evidence.

        The two expected digests must come from immutable checkpoint custody,
        never from the untrusted resume payload being opened.  Self-hashes alone
        cannot authenticate a plausible cursor or private-manifest substitution.
        """

        raw = _mapping("fixture allocator resume", value)
        _exact_keys(
            "fixture allocator resume",
            raw,
            {
                "schema",
                "fixture_only",
                "eligible_for_registered_evaluation",
                "parent_checkpoint_sha256",
                "snapshot",
                "sha256",
            },
        )
        if raw["schema"] != FIXTURE_RESUME_SCHEMA:
            raise EpisodeSeedCustodyError("fixture allocator resume schema differs")
        if _boolean("fixture_only", raw["fixture_only"]) is not True:
            raise EpisodeSeedCustodyError("allocator resume is not marked fixture-only")
        if (
            _boolean(
                "eligible_for_registered_evaluation",
                raw["eligible_for_registered_evaluation"],
            )
            is not False
        ):
            raise EpisodeSeedCustodyError(
                "fixture allocator resume cannot be evaluation-eligible"
            )
        admitted_parent_sha256 = _sha256(
            "expected parent checkpoint sha256",
            expected_parent_checkpoint_sha256,
        )
        if (
            _sha256(
                "fixture resume parent checkpoint sha256",
                raw["parent_checkpoint_sha256"],
            )
            != admitted_parent_sha256
        ):
            raise EpisodeSeedCustodyError(
                "fixture allocator resume parent checkpoint differs from custody"
            )
        supplied_resume_sha = _sha256("fixture resume sha256", raw["sha256"])
        if supplied_resume_sha != _sha256(
            "expected fixture resume sha256", expected_resume_sha256
        ):
            raise EpisodeSeedCustodyError(
                "fixture allocator resume differs from the checkpoint-pinned digest"
            )
        resume_body = dict(raw)
        resume_body.pop("sha256")
        if canonical_sha256(resume_body) != supplied_resume_sha:
            raise EpisodeSeedCustodyError("fixture allocator resume digest differs")

        snapshot = _mapping("allocator snapshot", raw["snapshot"])
        _exact_keys(
            "allocator snapshot",
            snapshot,
            {"schema", "allocator_identity", "cursors", "ledger", "sha256"},
        )
        if snapshot["schema"] != ALLOCATOR_SNAPSHOT_SCHEMA:
            raise EpisodeSeedCustodyError("allocator snapshot schema differs")
        supplied_snapshot_sha = _sha256("allocator snapshot sha256", snapshot["sha256"])
        snapshot_body = dict(snapshot)
        snapshot_body.pop("sha256")
        if canonical_sha256(snapshot_body) != supplied_snapshot_sha:
            raise EpisodeSeedCustodyError("allocator snapshot digest differs")

        identity = _mapping("allocator identity", snapshot["allocator_identity"])
        _validate_identity(identity)
        if identity["mode"] != EpisodeAllocatorModeV1.FIXTURE.value:
            raise RegisteredResumeForbiddenError(
                "only fixture allocator state can be restored"
            )
        if identity["split"] != EpisodeSeedSplitV1.FIXTURE.value:
            raise EpisodeSeedCustodyError("restored allocator split is not fixture")
        allocator = cls.fixture(
            base_seed=_integer("identity base_seed", identity["base_seed"], minimum=0),
            episode_decisions=_integer(
                "identity episode_decisions", identity["episode_decisions"], minimum=1
            ),
        )
        if allocator.identity != dict(identity):
            raise EpisodeSeedCustodyError(
                "restored allocator identity is not canonical"
            )

        cursor_items = _sequence("allocator cursors", snapshot["cursors"])
        cursors = [EpisodeCursorV1.from_dict(item) for item in cursor_items]
        ledger_payload = _mapping("reset-manifest ledger", snapshot["ledger"])
        _exact_keys(
            "reset-manifest ledger",
            ledger_payload,
            {"schema", "allocator_identity_sha256", "entries", "sha256"},
        )
        if ledger_payload["schema"] != LEDGER_SCHEMA:
            raise EpisodeSeedCustodyError("reset-manifest ledger schema differs")
        if (
            _sha256(
                "ledger allocator identity",
                ledger_payload["allocator_identity_sha256"],
            )
            != allocator.identity_sha256
        ):
            raise EpisodeSeedCustodyError("ledger allocator identity differs")
        ledger = [
            ResetManifestLedgerEntryV1.from_dict(item)
            for item in _sequence(
                "reset-manifest ledger entries", ledger_payload["entries"]
            )
        ]
        ledger_body = dict(ledger_payload)
        supplied_ledger_sha = _sha256("ledger sha256", ledger_body.pop("sha256"))
        if canonical_sha256(ledger_body) != supplied_ledger_sha:
            raise EpisodeSeedCustodyError("reset-manifest ledger digest differs")
        allocator._validate_state(cursors, ledger)
        allocator._cursors = cursors
        allocator._ledger = ledger
        allocator._used_seeds = {entry.episode_seed for entry in ledger}
        allocator._reattachment_required = {
            cursor.environment_index
            for cursor in cursors
            if cursor.current_episode_ordinal is not None and not cursor.awaiting_reset
        }
        return allocator

    def _validate_state(
        self,
        cursors: Sequence[EpisodeCursorV1],
        ledger: Sequence[ResetManifestLedgerEntryV1],
    ) -> None:
        if len(cursors) != VECTOR_ENV_COUNT:
            raise EpisodeSeedCustodyError(
                f"allocator must contain exactly {VECTOR_ENV_COUNT} cursors"
            )
        if [cursor.environment_index for cursor in cursors] != list(
            range(VECTOR_ENV_COUNT)
        ):
            raise EpisodeSeedCustodyError("allocator cursors are missing or reordered")
        seeds: set[int] = set()
        ordinals_by_environment: list[list[int]] = [[] for _ in range(VECTOR_ENV_COUNT)]
        initial_framework_seeds: list[tuple[int, int | None]] = []
        for expected_ledger_index, entry in enumerate(ledger):
            if entry.ledger_index != expected_ledger_index:
                raise EpisodeSeedCustodyError("ledger indices are not contiguous")
            if entry.allocator_identity_sha256 != self.identity_sha256:
                raise EpisodeSeedCustodyError("ledger entry allocator identity differs")
            if entry.split is not self._split:
                raise EpisodeSeedCustodyError(
                    "ledger entry crosses the allocator split"
                )
            index = self._index(entry.environment_index)
            expected_global = VECTOR_ENV_COUNT * entry.local_episode_ordinal + index
            if entry.global_episode_ordinal != expected_global:
                raise EpisodeSeedCustodyError("ledger global episode ordinal differs")
            expected_seed = self.seed_for(index, entry.local_episode_ordinal)
            if entry.episode_seed != expected_seed:
                raise EpisodeSeedCustodyError(
                    "ledger seed differs from the fixed mapping"
                )
            if entry.episode_seed in seeds:
                raise EpisodeSeedCustodyError(
                    f"ledger duplicates episode seed {entry.episode_seed}"
                )
            seeds.add(entry.episode_seed)
            ordinals_by_environment[index].append(entry.local_episode_ordinal)
            if entry.local_episode_ordinal == 0:
                initial_framework_seeds.append((index, entry.ignored_framework_seed))
            elif entry.ignored_framework_seed is not None:
                raise EpisodeSeedCustodyError(
                    "only an initial vector reset may ignore a framework seed"
                )

        framework_presence = {value is not None for _, value in initial_framework_seeds}
        if len(framework_presence) > 1:
            raise EpisodeSeedCustodyError(
                "initial vector resets mix absent and supplied framework seeds"
            )
        framework_bases = {
            value - index
            for index, value in initial_framework_seeds
            if value is not None
        }
        if len(framework_bases) > 1 or any(base < 0 for base in framework_bases):
            raise EpisodeSeedCustodyError(
                "initial vector framework seeds do not share one valid base"
            )
        for _, value in initial_framework_seeds:
            if value is not None and (
                classify_episode_seed(value) is not EpisodeSeedSplitV1.CONTAMINATED
            ):
                raise EpisodeSeedCustodyError(
                    "audited framework seed is outside the contaminated episode domain"
                )

        for cursor in cursors:
            index = cursor.environment_index
            ordinals = ordinals_by_environment[index]
            if ordinals != list(range(len(ordinals))):
                raise EpisodeSeedCustodyError(
                    f"environment {index} ledger ordinals are not contiguous"
                )
            if cursor.next_episode_ordinal != len(ordinals):
                raise EpisodeSeedCustodyError(
                    f"environment {index} next episode ordinal differs from its ledger"
                )
            if not ordinals:
                if (
                    cursor.current_episode_ordinal is not None
                    or cursor.decision_offset != 0
                    or not cursor.awaiting_reset
                ):
                    raise EpisodeSeedCustodyError(
                        f"environment {index} empty cursor is inconsistent"
                    )
                continue
            if cursor.current_episode_ordinal != ordinals[-1]:
                raise EpisodeSeedCustodyError(
                    f"environment {index} current episode ordinal differs"
                )
            if cursor.decision_offset > self._episode_decisions:
                raise EpisodeSeedCustodyError(
                    f"environment {index} decision offset exceeds its horizon"
                )
            if cursor.awaiting_reset != (
                cursor.decision_offset == self._episode_decisions
            ):
                raise EpisodeSeedCustodyError(
                    f"environment {index} reset state disagrees with decision offset"
                )

    def wrap(
        self, environment: gym.Env[Any, Any], *, environment_index: int
    ) -> SeedCustodyWrapperV1:
        """Wrap one of the allocator's four independently constructed envs."""

        return SeedCustodyWrapperV1(
            environment,
            allocator=self,
            environment_index=environment_index,
        )


class SeedCustodyWrapperV1(gym.Wrapper[Any, Any, Any, Any]):
    """Supply allocator-owned seeds on explicit and DummyVecEnv resets."""

    def __init__(
        self,
        environment: gym.Env[Any, Any],
        *,
        allocator: EpisodeSeedAllocatorV1,
        environment_index: int,
    ) -> None:
        super().__init__(environment)
        if not isinstance(allocator, EpisodeSeedAllocatorV1):
            raise TypeError("allocator must be an EpisodeSeedAllocatorV1")
        self.seed_allocator = allocator
        self.environment_index = allocator._index(environment_index)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if self.seed_allocator.fixture_reattachment_required(self.environment_index):
            try:
                plan = self.seed_allocator.begin_fixture_reattachment(
                    self.environment_index,
                    ignored_framework_seed=seed,
                )
                observation, info = self.env.reset(
                    seed=plan.episode_seed,
                    options=options,
                )
                audit_manifest = getattr(self.env.unwrapped, "audit_manifest", None)
                if not callable(audit_manifest):
                    raise EpisodeSeedCustodyError(
                        "wrapped beta environment does not expose audit_manifest()"
                    )
                identity = BetaManifestIdentityV1.from_audit_manifest(
                    audit_manifest(),
                    expected_episode_seed=plan.episode_seed,
                )
                self.seed_allocator.commit_fixture_reattachment(plan, identity)
            except Exception as reason:
                self.seed_allocator.fail_closed(reason)
                raise
            return observation, info

        reservation = self.seed_allocator.begin_reset(
            self.environment_index,
            ignored_framework_seed=seed,
        )
        try:
            observation, info = self.env.reset(
                seed=reservation.episode_seed,
                options=options,
            )
            audit_manifest = getattr(self.env.unwrapped, "audit_manifest", None)
            if not callable(audit_manifest):
                raise EpisodeSeedCustodyError(
                    "wrapped beta environment does not expose audit_manifest()"
                )
            identity = BetaManifestIdentityV1.from_audit_manifest(
                audit_manifest(),
                expected_episode_seed=reservation.episode_seed,
            )
            self.seed_allocator.commit_reset(reservation, identity)
        except Exception as reason:
            self.seed_allocator.fail_closed(reason)
            raise
        # The private seed and manifest identity deliberately do not enter info.
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        try:
            observation, reward, terminated, truncated, info = self.env.step(action)
            reward = float(reward)
            self.seed_allocator.record_transition(
                self.environment_index,
                terminated=terminated,
                truncated=truncated,
            )
        except Exception as reason:
            self.seed_allocator.fail_closed(reason)
            raise
        return observation, reward, terminated, truncated, info


def find_seed_allocator(vector_environment: object) -> EpisodeSeedAllocatorV1:
    """Extract and validate the shared allocator from an SB3-like VecEnv."""

    environments = getattr(vector_environment, "envs", None)
    if not isinstance(environments, Sequence) or isinstance(environments, (str, bytes)):
        raise EpisodeSeedCustodyError("vector environment does not expose envs")
    if len(environments) != VECTOR_ENV_COUNT:
        raise EpisodeSeedCustodyError(
            f"vector environment must contain exactly {VECTOR_ENV_COUNT} envs"
        )
    wrappers: list[SeedCustodyWrapperV1] = []
    for position, outer in enumerate(environments):
        current = outer
        seen: set[int] = set()
        found: SeedCustodyWrapperV1 | None = None
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, SeedCustodyWrapperV1):
                found = current
                break
            current = getattr(current, "env", None)
        if found is None:
            raise EpisodeSeedCustodyError(
                f"vector environment {position} has no seed-custody wrapper"
            )
        if found.environment_index != position:
            raise EpisodeSeedCustodyError(
                f"vector environment {position} is bound to custody index "
                f"{found.environment_index}"
            )
        wrappers.append(found)
    allocator = wrappers[0].seed_allocator
    if any(wrapper.seed_allocator is not allocator for wrapper in wrappers[1:]):
        raise EpisodeSeedCustodyError(
            "vector environments do not share one episode-seed allocator"
        )
    return allocator


__all__ = [
    "ALLOCATOR_ID",
    "BetaManifestIdentityV1",
    "DEVELOPMENT_SEED_MAX",
    "DEVELOPMENT_SEED_MIN",
    "EpisodeAllocatorModeV1",
    "EpisodeAllocatorSnapshotV1",
    "EpisodeCursorV1",
    "EpisodeSeedCustodyError",
    "EpisodeSeedDomainV1",
    "EpisodeSeedExhaustedError",
    "EpisodeSeedAllocatorV1",
    "EpisodeSeedSplitV1",
    "FIXTURE_SEED_MAX",
    "FIXTURE_SEED_MIN",
    "HELDOUT_SEED_MAX",
    "HELDOUT_SEED_MIN",
    "REGISTERED_EPISODE_DECISIONS",
    "RegisteredResumeForbiddenError",
    "ResetManifestLedgerEntryV1",
    "SEED_DOMAINS",
    "SeedCustodyWrapperV1",
    "TRAINING_SEED_MAX",
    "TRAINING_SEED_MIN",
    "VECTOR_ENV_COUNT",
    "canonical_sha256",
    "classify_episode_seed",
    "episode_seed_for",
    "find_seed_allocator",
    "seed_domain",
]

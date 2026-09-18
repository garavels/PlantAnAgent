"""Strict, artifact-bound checkpoint evidence for beta RL training.

This module is deliberately independent of the optional trainer runtime.  It
authenticates the stable checkpoint boundary (campaign identity, allocator
custody, runtime provenance, and the constrained-state envelope) while leaving
the trainer itself responsible for fully replaying and validating every
optimizer-specific field inside that envelope.

Parsing evidence never restores an allocator and never authorizes an
environment reset.  The allocator snapshot is reconstructed only as immutable
evidence and its cursor/ledger invariants are checked without creating a live
seed stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from numbers import Integral
from typing import Any, Mapping, Protocol, Sequence, cast

from .config import (
    BETA_CONSTRAINED_RL_PROTOCOL_ID,
    BETA_CORE_ALGORITHM_IDS,
    BETA_FIXTURE_MAX_COLLECTED_TIMESTEPS,
    BETA_FIXTURE_RUN_KIND,
    BETA_OPTIMIZER_SEEDS,
    BETA_REGISTERED_RUN_KIND,
    BETA_REGISTERED_TOTAL_TIMESTEPS,
    BETA_RUN_KINDS,
)
from .episode_allocator import (
    ALLOCATOR_SNAPSHOT_SCHEMA,
    FIXTURE_RESUME_SCHEMA,
    LEDGER_SCHEMA,
    VECTOR_ENV_COUNT,
    EpisodeAllocatorSnapshotV1,
    EpisodeCursorV1,
    EpisodeSeedSplitV1,
    ResetManifestLedgerEntryV1,
    canonical_sha256,
    classify_episode_seed,
    episode_seed_for,
    seed_domain,
)
from .runtime_provenance import BetaTrainingRuntimeProvenanceV1


BETA_CHECKPOINT_EVIDENCE_SCHEMA = "beta-rl-checkpoint-evidence-v1"
BETA_CHECKPOINT_RESUME_PARENT_SCHEMA = "beta-rl-checkpoint-resume-parent-v1"
CONSTRAINT_STATE_ENVELOPE_SCHEMA = "constrained-ppo-state-envelope-v1"
CONSTRAINT_STATE_SCHEMA = "constrained-ppo-state-v1"
EXTERNAL_EPISODE_STATE_PIN_SCHEMA = "external-episode-state-pin-v1"

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "protocol_id",
        "algorithm_id",
        "run_kind",
        "optimizer_seed",
        "total_timesteps",
        "artifact_sha256",
        "runtime_provenance",
        "allocator_snapshot",
        "constraint_state",
        "fixture_resume",
        "resume_parent",
        "resumed_from_parent",
        "exact_endpoint_admitted",
        "eligible_for_registered_evaluation",
        "sha256",
    }
)
_SNAPSHOT_FIELDS = frozenset(
    {"schema", "allocator_identity", "cursors", "ledger", "sha256"}
)
_LEDGER_FIELDS = frozenset({"schema", "allocator_identity_sha256", "entries", "sha256"})
_FIXTURE_RESUME_FIELDS = frozenset(
    {
        "schema",
        "fixture_only",
        "eligible_for_registered_evaluation",
        "parent_checkpoint_sha256",
        "snapshot",
        "sha256",
    }
)
_CONSTRAINT_ENVELOPE_FIELDS = frozenset({"schema", "payload", "sha256"})
_CONSTRAINT_REQUIRED_PAYLOAD_FIELDS = frozenset(
    {
        "schema",
        "protocol_id",
        "algorithm_id",
        "settings",
        "num_timesteps",
        "external_episode_state_pin",
    }
)
_CONSTRAINT_REQUIRED_SETTINGS_FIELDS = frozenset(
    {"schema", "protocol_id", "algorithm_id", "optimizer_seed"}
)
_EXTERNAL_PIN_FIELDS = frozenset({"schema", "payload", "sha256"})
_RESUME_PARENT_FIELDS = frozenset(
    {
        "schema",
        "metadata_sha256",
        "policy_artifact_sha256",
        "checkpoint_contract_digest",
        "checkpoint_evidence_sha256",
        "allocator_snapshot_sha256",
        "constraint_state_sha256",
        "total_timesteps",
        "sha256",
    }
)

_REGISTERED_ROLLOUT_TRANSITIONS = 8_192
_REGISTERED_ROLLOUT_COUNT = (
    BETA_REGISTERED_TOTAL_TIMESTEPS // _REGISTERED_ROLLOUT_TRANSITIONS
)
_REGISTERED_DECISIONS_PER_ENVIRONMENT = (
    BETA_REGISTERED_TOTAL_TIMESTEPS // VECTOR_ENV_COUNT
)
_REGISTERED_COMPLETED_EPISODES_PER_ENVIRONMENT = 390
_REGISTERED_FINAL_DECISION_OFFSET = 64
_REGISTERED_RESETS_PER_ENVIRONMENT = 391
_REGISTERED_RESET_COUNT = _REGISTERED_RESETS_PER_ENVIRONMENT * VECTOR_ENV_COUNT


class BetaCheckpointEvidenceError(ValueError):
    """Checkpoint evidence is malformed, inconsistent, or unauthenticated."""


class _ConstraintStateAttestationView(Protocol):
    artifact_sha256: str
    algorithm_id: str
    num_timesteps: int
    constraint_state_sha256: str
    checkpoint_observation_sha256: str

    @property
    def constraint_state(self) -> dict[str, Any]: ...


def _canonical_json(value: object, *, ensure_ascii: bool = False) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=ensure_ascii,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as reason:
        raise BetaCheckpointEvidenceError(
            f"checkpoint evidence is not finite canonical JSON: {reason}"
        ) from reason


def _canonical_copy(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed JSON object")
    decoded = json.loads(_canonical_json(value))
    if not isinstance(decoded, dict):  # pragma: no cover - guarded above
        raise TypeError(f"{name} did not encode as a JSON object")
    return cast(dict[str, Any], decoded)


def _canonical_json_text(name: str, value: object) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, str):
        raise TypeError(f"{name} JSON must be a string")
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError) as reason:
        raise BetaCheckpointEvidenceError(
            f"{name} JSON is invalid: {reason}"
        ) from reason
    normalized = _canonical_copy(name, decoded)
    canonical = _canonical_json(normalized)
    if value != canonical:
        raise BetaCheckpointEvidenceError(f"{name} JSON is not canonical")
    return canonical, normalized


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed JSON object")
    return value


def _sequence(name: str, value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a JSON array")
    return value


def _exact_keys(
    name: str,
    value: Mapping[str, Any],
    expected: frozenset[str],
) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing:
        raise BetaCheckpointEvidenceError(
            f"{name} is missing fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise BetaCheckpointEvidenceError(
            f"{name} has unknown fields: {', '.join(sorted(unknown))}"
        )


def _required_keys(
    name: str,
    value: Mapping[str, Any],
    required: frozenset[str],
) -> None:
    missing = required - set(value)
    if missing:
        raise BetaCheckpointEvidenceError(
            f"{name} is missing fields: {', '.join(sorted(missing))}"
        )


def _sha256(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA-256 string")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise BetaCheckpointEvidenceError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    return value


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise BetaCheckpointEvidenceError(f"{name} must be >= {minimum}")
    return normalized


def _boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _domain_separated_sha256(schema: str, payload: Mapping[str, Any]) -> str:
    """Reproduce the trainer's stable envelope digest without importing it."""

    encoded = _canonical_json(payload, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(schema.encode("ascii") + b"\x00" + encoded).hexdigest()


def _validate_allocator_state(snapshot: EpisodeAllocatorSnapshotV1) -> None:
    """Validate snapshot cursors and ledgers without restoring a live allocator."""

    identity = snapshot.allocator_identity
    identity_sha256 = cast(str, identity["sha256"])
    split = EpisodeSeedSplitV1(cast(str, identity["split"]))
    base_seed = _integer("allocator base_seed", identity["base_seed"])
    episode_decisions = _integer(
        "allocator episode_decisions", identity["episode_decisions"], minimum=1
    )

    if len(snapshot.cursors) != VECTOR_ENV_COUNT:
        raise BetaCheckpointEvidenceError(
            f"allocator snapshot must contain exactly {VECTOR_ENV_COUNT} cursors"
        )
    if [cursor.environment_index for cursor in snapshot.cursors] != list(
        range(VECTOR_ENV_COUNT)
    ):
        raise BetaCheckpointEvidenceError(
            "allocator snapshot cursors are missing or reordered"
        )

    seeds: set[int] = set()
    ordinals: list[list[int]] = [[] for _ in range(VECTOR_ENV_COUNT)]
    initial_framework_seeds: list[tuple[int, int | None]] = []
    domain = seed_domain(split)
    for expected_index, entry in enumerate(snapshot.ledger):
        if entry.ledger_index != expected_index:
            raise BetaCheckpointEvidenceError(
                "allocator ledger indices are not contiguous"
            )
        if entry.allocator_identity_sha256 != identity_sha256:
            raise BetaCheckpointEvidenceError("allocator ledger entry identity differs")
        if entry.split is not split:
            raise BetaCheckpointEvidenceError("allocator ledger entry split differs")
        environment_index = entry.environment_index
        if not 0 <= environment_index < VECTOR_ENV_COUNT:
            raise BetaCheckpointEvidenceError(
                "allocator ledger environment index is outside the vector"
            )
        expected_global = VECTOR_ENV_COUNT * entry.local_episode_ordinal + (
            environment_index
        )
        if entry.global_episode_ordinal != expected_global:
            raise BetaCheckpointEvidenceError(
                "allocator ledger global episode ordinal differs"
            )
        expected_seed = episode_seed_for(
            base_seed=base_seed,
            environment_index=environment_index,
            local_episode_ordinal=entry.local_episode_ordinal,
        )
        if entry.episode_seed != expected_seed or not domain.contains(expected_seed):
            raise BetaCheckpointEvidenceError(
                "allocator ledger seed differs from its fixed split mapping"
            )
        if entry.episode_seed in seeds:
            raise BetaCheckpointEvidenceError("allocator ledger duplicates a seed")
        seeds.add(entry.episode_seed)
        ordinals[environment_index].append(entry.local_episode_ordinal)
        if entry.local_episode_ordinal == 0:
            initial_framework_seeds.append(
                (environment_index, entry.ignored_framework_seed)
            )
        elif entry.ignored_framework_seed is not None:
            raise BetaCheckpointEvidenceError(
                "only an initial vector reset may ignore a framework seed"
            )

    framework_presence = {value is not None for _, value in initial_framework_seeds}
    if len(framework_presence) > 1:
        raise BetaCheckpointEvidenceError(
            "initial vector resets mix absent and supplied framework seeds"
        )
    framework_bases = {
        value - index for index, value in initial_framework_seeds if value is not None
    }
    if len(framework_bases) > 1 or any(base < 0 for base in framework_bases):
        raise BetaCheckpointEvidenceError(
            "initial framework seeds do not share one valid base"
        )
    if any(
        value is not None
        and classify_episode_seed(value) is not EpisodeSeedSplitV1.CONTAMINATED
        for _, value in initial_framework_seeds
    ):
        raise BetaCheckpointEvidenceError(
            "ignored framework seed is outside the contaminated split"
        )

    for cursor in snapshot.cursors:
        environment_index = cursor.environment_index
        environment_ordinals = ordinals[environment_index]
        if environment_ordinals != list(range(len(environment_ordinals))):
            raise BetaCheckpointEvidenceError(
                f"allocator environment {environment_index} ordinals are not contiguous"
            )
        if cursor.next_episode_ordinal != len(environment_ordinals):
            raise BetaCheckpointEvidenceError(
                f"allocator environment {environment_index} next ordinal differs"
            )
        if not environment_ordinals:
            if (
                cursor.current_episode_ordinal is not None
                or cursor.decision_offset != 0
                or not cursor.awaiting_reset
            ):
                raise BetaCheckpointEvidenceError(
                    f"allocator environment {environment_index} empty cursor differs"
                )
            continue
        if cursor.current_episode_ordinal != environment_ordinals[-1]:
            raise BetaCheckpointEvidenceError(
                f"allocator environment {environment_index} current ordinal differs"
            )
        if cursor.decision_offset > episode_decisions:
            raise BetaCheckpointEvidenceError(
                f"allocator environment {environment_index} offset exceeds horizon"
            )
        if cursor.awaiting_reset != (cursor.decision_offset == episode_decisions):
            raise BetaCheckpointEvidenceError(
                f"allocator environment {environment_index} reset state differs"
            )


def _validate_allocator_progress(
    snapshot: EpisodeAllocatorSnapshotV1, *, total_timesteps: int
) -> None:
    """Bind the reset ledger and all four cursors to the checkpoint cut.

    Beta training uses a synchronous four-environment vector.  Consequently
    all environments have consumed the same number of decisions, every reset
    wave is ordered by environment index, and the ledger contains the initial
    reset plus one reset for every completed episode.
    """

    episode_decisions = _integer(
        "allocator episode_decisions",
        snapshot.allocator_identity["episode_decisions"],
        minimum=1,
    )
    if total_timesteps % VECTOR_ENV_COUNT:
        raise BetaCheckpointEvidenceError(
            "checkpoint timesteps are not vector-environment aligned"
        )
    decisions_per_environment = total_timesteps // VECTOR_ENV_COUNT
    expected_current_ordinal, expected_offset = divmod(
        decisions_per_environment, episode_decisions
    )
    expected_resets_per_environment = expected_current_ordinal + 1

    expected_global_ordinals = list(range(len(snapshot.ledger)))
    observed_global_ordinals = [
        entry.global_episode_ordinal for entry in snapshot.ledger
    ]
    if observed_global_ordinals != expected_global_ordinals:
        raise BetaCheckpointEvidenceError(
            "allocator reset ledger is not in exact synchronous global order"
        )
    if len(snapshot.ledger) != expected_resets_per_environment * VECTOR_ENV_COUNT:
        raise BetaCheckpointEvidenceError(
            "allocator reset count differs from checkpoint timesteps"
        )

    for cursor in snapshot.cursors:
        if (
            cursor.current_episode_ordinal != expected_current_ordinal
            or cursor.decision_offset != expected_offset
            or cursor.next_episode_ordinal != expected_resets_per_environment
            or cursor.awaiting_reset
        ):
            raise BetaCheckpointEvidenceError(
                "allocator cursor progress differs from checkpoint timesteps"
            )


def _parse_allocator_snapshot(value: object) -> EpisodeAllocatorSnapshotV1:
    raw = _canonical_copy("allocator snapshot", value)
    _exact_keys("allocator snapshot", raw, _SNAPSHOT_FIELDS)
    if raw["schema"] != ALLOCATOR_SNAPSHOT_SCHEMA:
        raise BetaCheckpointEvidenceError("allocator snapshot schema differs")

    identity = _canonical_copy("allocator identity", raw["allocator_identity"])
    cursors = tuple(
        EpisodeCursorV1.from_dict(item)
        for item in _sequence("allocator snapshot cursors", raw["cursors"])
    )
    ledger_raw = _canonical_copy("allocator ledger", raw["ledger"])
    _exact_keys("allocator ledger", ledger_raw, _LEDGER_FIELDS)
    if ledger_raw["schema"] != LEDGER_SCHEMA:
        raise BetaCheckpointEvidenceError("allocator ledger schema differs")
    entries = tuple(
        ResetManifestLedgerEntryV1.from_dict(item)
        for item in _sequence("allocator ledger entries", ledger_raw["entries"])
    )
    snapshot = EpisodeAllocatorSnapshotV1(
        allocator_identity=identity,
        cursors=cursors,
        ledger=entries,
    )
    if snapshot.to_dict() != raw:
        raise BetaCheckpointEvidenceError(
            "allocator snapshot or nested ledger digest differs"
        )
    _validate_allocator_state(snapshot)
    return snapshot


def _parse_constraint_state(
    value: object,
    *,
    protocol_id: str,
    algorithm_id: str,
    optimizer_seed: int,
    total_timesteps: int,
    allocator_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    envelope = _canonical_copy("constraint state envelope", value)
    _exact_keys("constraint state envelope", envelope, _CONSTRAINT_ENVELOPE_FIELDS)
    if envelope["schema"] != CONSTRAINT_STATE_ENVELOPE_SCHEMA:
        raise BetaCheckpointEvidenceError("constraint state envelope schema differs")
    payload = _canonical_copy("constraint state payload", envelope["payload"])
    _required_keys(
        "constraint state payload",
        payload,
        _CONSTRAINT_REQUIRED_PAYLOAD_FIELDS,
    )
    supplied_digest = _sha256("constraint state envelope sha256", envelope["sha256"])
    if supplied_digest != _domain_separated_sha256(
        CONSTRAINT_STATE_ENVELOPE_SCHEMA, payload
    ):
        raise BetaCheckpointEvidenceError("constraint state envelope digest differs")
    if payload["schema"] != CONSTRAINT_STATE_SCHEMA:
        raise BetaCheckpointEvidenceError("constraint state payload schema differs")
    if payload["protocol_id"] != protocol_id:
        raise BetaCheckpointEvidenceError("constraint state protocol differs")
    if payload["algorithm_id"] != algorithm_id:
        raise BetaCheckpointEvidenceError("constraint state algorithm differs")
    if (
        _integer("constraint state num_timesteps", payload["num_timesteps"])
        != total_timesteps
    ):
        raise BetaCheckpointEvidenceError("constraint state timesteps differ")

    settings = _canonical_copy("constraint settings", payload["settings"])
    _required_keys(
        "constraint settings", settings, _CONSTRAINT_REQUIRED_SETTINGS_FIELDS
    )
    if settings["protocol_id"] != protocol_id:
        raise BetaCheckpointEvidenceError("constraint settings protocol differs")
    if settings["algorithm_id"] != algorithm_id:
        raise BetaCheckpointEvidenceError("constraint settings algorithm differs")
    if (
        _integer("constraint settings optimizer_seed", settings["optimizer_seed"])
        != optimizer_seed
    ):
        raise BetaCheckpointEvidenceError("constraint settings optimizer seed differs")

    external_pin = _canonical_copy(
        "external episode state pin", payload["external_episode_state_pin"]
    )
    _exact_keys("external episode state pin", external_pin, _EXTERNAL_PIN_FIELDS)
    if external_pin["schema"] != EXTERNAL_EPISODE_STATE_PIN_SCHEMA:
        raise BetaCheckpointEvidenceError("external episode state pin schema differs")
    external_payload = _parse_allocator_snapshot(external_pin["payload"]).to_dict()
    if external_payload != dict(allocator_snapshot):
        raise BetaCheckpointEvidenceError(
            "constraint external episode pin differs from allocator snapshot"
        )
    external_digest = _sha256(
        "external episode state pin sha256", external_pin["sha256"]
    )
    if external_digest != _domain_separated_sha256(
        EXTERNAL_EPISODE_STATE_PIN_SCHEMA, external_payload
    ):
        raise BetaCheckpointEvidenceError("external episode state pin digest differs")
    return envelope


def _validate_registered_endpoint(
    *,
    allocator_snapshot: Mapping[str, Any],
    constraint_state: Mapping[str, Any],
) -> None:
    """Prove the frozen registered endpoint from authenticated live-state evidence.

    The full trainer-specific ledger replay remains the responsibility of
    ``ConstrainedPPO.load``.  This independent boundary nevertheless refuses a
    positive eligibility bit unless the externally pinned allocator and the
    generic constrained-state envelope both disclose the exact registered cut.
    """

    snapshot = _parse_allocator_snapshot(allocator_snapshot)
    identity = snapshot.allocator_identity
    if (
        identity["mode"] != BETA_REGISTERED_RUN_KIND
        or identity["split"] != EpisodeSeedSplitV1.TRAINING.value
        or identity["base_seed"] != 1_000_000
        or identity["episode_decisions"] != 672
    ):
        raise BetaCheckpointEvidenceError(
            "registered endpoint allocator identity differs from the protocol"
        )
    if len(snapshot.ledger) != _REGISTERED_RESET_COUNT:
        raise BetaCheckpointEvidenceError(
            "registered endpoint allocator ledger must contain exactly 1,564 resets"
        )
    observed_seeds = sorted(entry.episode_seed for entry in snapshot.ledger)
    if observed_seeds != list(range(1_000_000, 1_001_564)):
        raise BetaCheckpointEvidenceError(
            "registered endpoint allocator seeds do not cover 1,000,000..1,001,563"
        )
    for cursor in snapshot.cursors:
        if (
            cursor.current_episode_ordinal
            != _REGISTERED_COMPLETED_EPISODES_PER_ENVIRONMENT
            or cursor.decision_offset != _REGISTERED_FINAL_DECISION_OFFSET
            or cursor.next_episode_ordinal != _REGISTERED_RESETS_PER_ENVIRONMENT
            or cursor.awaiting_reset
        ):
            raise BetaCheckpointEvidenceError(
                "registered endpoint allocator cursor differs from "
                "ordinal 390 / offset 64 / next 391"
            )
        consumed = cursor.current_episode_ordinal * 672 + cursor.decision_offset
        if consumed != _REGISTERED_DECISIONS_PER_ENVIRONMENT:
            raise BetaCheckpointEvidenceError(
                "registered endpoint allocator does not disclose 262,144 "
                "decisions per environment"
            )

    payload = _canonical_copy(
        "registered constraint state payload", constraint_state["payload"]
    )
    if payload["num_timesteps"] != BETA_REGISTERED_TOTAL_TIMESTEPS:
        raise BetaCheckpointEvidenceError(
            "registered endpoint must contain exactly 1,048,576 transitions"
        )
    pending = payload.get("pending_rollout")
    if not isinstance(pending, bool):
        raise TypeError("registered endpoint pending_rollout must be a boolean")
    if pending:
        raise BetaCheckpointEvidenceError(
            "registered endpoint cannot contain a pending rollout"
        )
    if payload.get("aborted_rollout", object()) is not None:
        raise BetaCheckpointEvidenceError(
            "registered endpoint cannot contain an aborted rollout"
        )
    if (
        _integer(
            "registered endpoint unledgered_transition_count",
            payload.get("unledgered_transition_count"),
        )
        != 0
    ):
        raise BetaCheckpointEvidenceError(
            "registered endpoint cannot contain unledgered transitions"
        )

    ledgers = _sequence(
        "registered endpoint rollout ledgers", payload.get("rollout_ledgers")
    )
    if len(ledgers) != _REGISTERED_ROLLOUT_COUNT:
        raise BetaCheckpointEvidenceError(
            "registered endpoint must contain exactly 128 rollout ledgers"
        )
    previous_sha256: str | None = None
    for expected_index, value in enumerate(ledgers, start=1):
        ledger = _mapping("registered endpoint rollout ledger", value)
        if (
            _integer("registered rollout index", ledger.get("rollout_index"))
            != expected_index
        ):
            raise BetaCheckpointEvidenceError(
                "registered endpoint rollout indices are not contiguous"
            )
        if (
            _integer(
                "registered rollout transition_count",
                ledger.get("transition_count"),
                minimum=1,
            )
            != _REGISTERED_ROLLOUT_TRANSITIONS
        ):
            raise BetaCheckpointEvidenceError(
                "registered endpoint rollout must contain exactly 8,192 transitions"
            )
        if ledger.get("algorithm_id") != payload["algorithm_id"]:
            raise BetaCheckpointEvidenceError(
                "registered endpoint rollout algorithm differs"
            )
        if ledger.get("previous_sha256") != previous_sha256:
            raise BetaCheckpointEvidenceError(
                "registered endpoint rollout digest chain differs"
            )
        previous_sha256 = _sha256("registered rollout sha256", ledger.get("sha256"))
    if payload.get("rollout_ledger_sha256") != previous_sha256:
        raise BetaCheckpointEvidenceError(
            "registered endpoint terminal rollout digest differs"
        )

    accumulator = _canonical_copy(
        "registered endpoint episode cost accumulator",
        payload.get("episode_cost_accumulator"),
    )
    if _integer("registered accumulator n_envs", accumulator.get("n_envs")) != 4:
        raise BetaCheckpointEvidenceError(
            "registered endpoint accumulator must contain four environments"
        )
    next_decisions = _sequence(
        "registered accumulator next_decision_index",
        accumulator.get("next_decision_index"),
    )
    if [
        _integer(f"registered next decision {index}", value, minimum=1)
        for index, value in enumerate(next_decisions)
    ] != [65, 65, 65, 65]:
        raise BetaCheckpointEvidenceError(
            "registered endpoint cost cursors must all be at decision offset 64"
        )


def _parse_fixture_resume(
    value: object,
    *,
    artifact_sha256: str,
    allocator_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _canonical_copy("fixture resume envelope", value)
    _exact_keys("fixture resume envelope", raw, _FIXTURE_RESUME_FIELDS)
    if raw["schema"] != FIXTURE_RESUME_SCHEMA:
        raise BetaCheckpointEvidenceError("fixture resume schema differs")
    if _boolean("fixture_only", raw["fixture_only"]) is not True:
        raise BetaCheckpointEvidenceError("fixture resume is not fixture-only")
    if (
        _boolean(
            "fixture resume eligibility",
            raw["eligible_for_registered_evaluation"],
        )
        is not False
    ):
        raise BetaCheckpointEvidenceError(
            "fixture resume cannot be registered-evaluation eligible"
        )
    if (
        _sha256(
            "fixture resume parent checkpoint sha256",
            raw["parent_checkpoint_sha256"],
        )
        != artifact_sha256
    ):
        raise BetaCheckpointEvidenceError(
            "fixture resume is not bound to this checkpoint artifact"
        )
    resumed_snapshot = _parse_allocator_snapshot(raw["snapshot"]).to_dict()
    if resumed_snapshot != dict(allocator_snapshot):
        raise BetaCheckpointEvidenceError(
            "fixture resume snapshot differs from checkpoint allocator snapshot"
        )
    supplied_digest = _sha256("fixture resume sha256", raw["sha256"])
    body = dict(raw)
    body.pop("sha256")
    if supplied_digest != canonical_sha256(body):
        raise BetaCheckpointEvidenceError("fixture resume digest differs")
    return raw


def _parse_resume_parent(
    value: object,
    *,
    current_artifact_sha256: str,
    current_total_timesteps: int,
) -> dict[str, Any]:
    raw = _canonical_copy("beta checkpoint resume parent", value)
    _exact_keys("beta checkpoint resume parent", raw, _RESUME_PARENT_FIELDS)
    if raw["schema"] != BETA_CHECKPOINT_RESUME_PARENT_SCHEMA:
        raise BetaCheckpointEvidenceError("beta resume-parent schema differs")
    _sha256("resume-parent metadata_sha256", raw["metadata_sha256"])
    parent_artifact = _sha256(
        "resume-parent policy_artifact_sha256",
        raw["policy_artifact_sha256"],
    )
    _sha256(
        "resume-parent checkpoint_contract_digest",
        raw["checkpoint_contract_digest"],
    )
    _sha256(
        "resume-parent checkpoint_evidence_sha256",
        raw["checkpoint_evidence_sha256"],
    )
    _sha256(
        "resume-parent allocator_snapshot_sha256",
        raw["allocator_snapshot_sha256"],
    )
    _sha256(
        "resume-parent constraint_state_sha256",
        raw["constraint_state_sha256"],
    )
    parent_timesteps = _integer(
        "resume-parent total_timesteps", raw["total_timesteps"], minimum=1
    )
    if parent_timesteps % VECTOR_ENV_COUNT:
        raise BetaCheckpointEvidenceError(
            "resume-parent timesteps must be vector-aligned"
        )
    if parent_timesteps >= current_total_timesteps:
        raise BetaCheckpointEvidenceError(
            "resume parent must precede the current checkpoint timestep cut"
        )
    if parent_artifact == current_artifact_sha256:
        raise BetaCheckpointEvidenceError(
            "resume parent and current policy artifacts must differ"
        )
    body = dict(raw)
    supplied_digest = _sha256("resume-parent sha256", body.pop("sha256"))
    if supplied_digest != canonical_sha256(body):
        raise BetaCheckpointEvidenceError("beta resume-parent digest differs")
    return raw


def build_beta_resume_parent_identity(
    *,
    metadata_sha256: object,
    policy_artifact_sha256: object,
    checkpoint_contract_digest: object,
    checkpoint_evidence_sha256: object,
    allocator_snapshot_sha256: object,
    constraint_state_sha256: object,
    total_timesteps: object,
) -> dict[str, Any]:
    """Build the exact portable identity of a fixture resume parent."""

    body: dict[str, Any] = {
        "schema": BETA_CHECKPOINT_RESUME_PARENT_SCHEMA,
        "metadata_sha256": _sha256("resume-parent metadata_sha256", metadata_sha256),
        "policy_artifact_sha256": _sha256(
            "resume-parent policy_artifact_sha256", policy_artifact_sha256
        ),
        "checkpoint_contract_digest": _sha256(
            "resume-parent checkpoint_contract_digest",
            checkpoint_contract_digest,
        ),
        "checkpoint_evidence_sha256": _sha256(
            "resume-parent checkpoint_evidence_sha256",
            checkpoint_evidence_sha256,
        ),
        "allocator_snapshot_sha256": _sha256(
            "resume-parent allocator_snapshot_sha256",
            allocator_snapshot_sha256,
        ),
        "constraint_state_sha256": _sha256(
            "resume-parent constraint_state_sha256",
            constraint_state_sha256,
        ),
        "total_timesteps": _integer(
            "resume-parent total_timesteps", total_timesteps, minimum=1
        ),
    }
    if body["total_timesteps"] % VECTOR_ENV_COUNT:
        raise BetaCheckpointEvidenceError(
            "resume-parent timesteps must be vector-aligned"
        )
    return {**body, "sha256": canonical_sha256(body)}


def _runtime_provenance(value: object) -> BetaTrainingRuntimeProvenanceV1:
    if isinstance(value, BetaTrainingRuntimeProvenanceV1):
        return BetaTrainingRuntimeProvenanceV1.from_dict(value.to_dict())
    return BetaTrainingRuntimeProvenanceV1.from_dict(
        _mapping("expected runtime provenance", value)
    )


def _validate_artifact_attestation(
    evidence: BetaRLCheckpointEvidenceV1,
    value: object,
    *,
    name: str,
) -> _ConstraintStateAttestationView:
    """Require a nominal capability minted by ``ConstrainedPPO.load``."""

    # Lazy import preserves dependency-light parsing for API/verifier processes.
    from .constrained_ppo import ValidatedConstraintStateAttestationV1

    if type(value) is not ValidatedConstraintStateAttestationV1:
        raise BetaCheckpointEvidenceError(
            f"{name} must be issued by ConstrainedPPO.load"
        )
    attestation = value
    if attestation.artifact_sha256 != evidence.artifact_sha256:
        raise BetaCheckpointEvidenceError(
            f"{name} artifact differs from checkpoint evidence"
        )
    if attestation.algorithm_id != evidence.algorithm_id:
        raise BetaCheckpointEvidenceError(
            f"{name} algorithm differs from checkpoint evidence"
        )
    if attestation.num_timesteps != evidence.total_timesteps:
        raise BetaCheckpointEvidenceError(
            f"{name} timesteps differ from checkpoint evidence"
        )
    if (
        attestation.constraint_state_sha256 != evidence.constraint_state["sha256"]
        or attestation.constraint_state != evidence.constraint_state
    ):
        raise BetaCheckpointEvidenceError(
            f"{name} constraint state differs from checkpoint evidence"
        )
    return cast(_ConstraintStateAttestationView, attestation)


def _validate_resume_receipt(
    *,
    child: BetaRLCheckpointEvidenceV1,
    parent: BetaRLCheckpointEvidenceV1,
    parent_attestation: _ConstraintStateAttestationView,
) -> None:
    """Bind the child's first new rollout to every exact parent-state facet."""

    parent_payload = parent.constraint_state["payload"]
    child_payload = child.constraint_state["payload"]
    receipt = _canonical_copy(
        "resume continuation receipt",
        child_payload.get("resume_continuation_receipt"),
    )
    parent_rng = _canonical_copy(
        "parent learner RNG state", parent_payload.get("learner_rng_state")
    )
    parent_multiplier = _canonical_copy(
        "parent multiplier state", parent_payload.get("multiplier_state")
    )
    parent_accumulator = _canonical_copy(
        "parent episode cost accumulator",
        parent_payload.get("episode_cost_accumulator"),
    )
    parent_pin = _canonical_copy(
        "parent external episode state pin",
        parent_payload.get("external_episode_state_pin"),
    )
    parent_ledgers = _sequence(
        "parent constraint rollout ledgers", parent_payload.get("rollout_ledgers")
    )
    child_ledgers = _sequence(
        "child constraint rollout ledgers", child_payload.get("rollout_ledgers")
    )
    first_index = len(parent_ledgers) + 1
    expected = {
        "parent_artifact_sha256": parent_attestation.artifact_sha256,
        "parent_constraint_state_sha256": parent_attestation.constraint_state_sha256,
        "parent_policy_parameter_sha256": parent_payload.get("policy_parameter_sha256"),
        "parent_optimizer_state_sha256": parent_payload.get("optimizer_state_sha256"),
        "parent_learner_rng_state_sha256": _domain_separated_sha256(
            "resume-parent-learner-rng-state-v1", parent_rng
        ),
        "parent_multiplier_state_sha256": _domain_separated_sha256(
            "resume-parent-multiplier-state-v1", parent_multiplier
        ),
        "parent_episode_cost_accumulator_sha256": _domain_separated_sha256(
            "resume-parent-episode-cost-accumulator-v1", parent_accumulator
        ),
        "parent_external_episode_state_pin_sha256": parent_pin.get("sha256"),
        "parent_checkpoint_observation_sha256": (
            parent_attestation.checkpoint_observation_sha256
        ),
        "parent_num_timesteps": parent.total_timesteps,
        "parent_rollout_ledger_sha256": parent_payload.get("rollout_ledger_sha256"),
        "first_resumed_rollout_index": first_index,
    }
    for field_name, expected_value in expected.items():
        if receipt.get(field_name) != expected_value:
            raise BetaCheckpointEvidenceError(
                f"resume continuation receipt {field_name} differs from parent"
            )
    if first_index > len(child_ledgers):
        raise BetaCheckpointEvidenceError(
            "resume continuation receipt has no first child rollout"
        )
    first_ledger = _mapping(
        "first resumed rollout ledger", child_ledgers[first_index - 1]
    )
    if receipt.get("first_resumed_rollout_sha256") != first_ledger.get("sha256"):
        raise BetaCheckpointEvidenceError(
            "resume continuation receipt does not bind the first child rollout"
        )


def _validate_resume_lineage(
    *,
    child: BetaRLCheckpointEvidenceV1,
    parent: BetaRLCheckpointEvidenceV1,
    parent_identity: Mapping[str, Any],
) -> None:
    """Prove exact parent identity plus allocator and optimizer continuation."""

    if (
        parent.run_kind != BETA_FIXTURE_RUN_KIND
        or parent.algorithm_id != child.algorithm_id
        or parent.optimizer_seed != child.optimizer_seed
        or parent.runtime_provenance != child.runtime_provenance
    ):
        raise BetaCheckpointEvidenceError(
            "resume parent campaign/runtime differs from child"
        )
    expected_identity_fields = {
        "policy_artifact_sha256": parent.artifact_sha256,
        "checkpoint_evidence_sha256": parent.sha256,
        "allocator_snapshot_sha256": parent.allocator_snapshot["sha256"],
        "constraint_state_sha256": parent.constraint_state["sha256"],
        "total_timesteps": parent.total_timesteps,
    }
    for field_name, expected in expected_identity_fields.items():
        if parent_identity[field_name] != expected:
            raise BetaCheckpointEvidenceError(
                f"resume-parent {field_name} differs from parent evidence"
            )

    parent_snapshot = parent.allocator_snapshot
    child_snapshot = child.allocator_snapshot
    if child_snapshot["allocator_identity"] != parent_snapshot["allocator_identity"]:
        raise BetaCheckpointEvidenceError(
            "resumed allocator identity differs from parent"
        )
    parent_entries = parent_snapshot["ledger"]["entries"]
    child_entries = child_snapshot["ledger"]["entries"]
    if child_entries[: len(parent_entries)] != parent_entries:
        raise BetaCheckpointEvidenceError(
            "resumed allocator ledger is not an exact parent extension"
        )

    parent_payload = parent.constraint_state["payload"]
    child_payload = child.constraint_state["payload"]
    for immutable_field in ("settings", "ppo_configuration"):
        if (
            immutable_field not in parent_payload
            or immutable_field not in child_payload
        ):
            raise BetaCheckpointEvidenceError(
                f"resume lineage requires constraint {immutable_field}"
            )
        if child_payload[immutable_field] != parent_payload[immutable_field]:
            raise BetaCheckpointEvidenceError(
                f"resumed constraint {immutable_field} differs from parent"
            )
    parent_ledgers = _sequence(
        "parent constraint rollout ledgers", parent_payload.get("rollout_ledgers")
    )
    child_ledgers = _sequence(
        "child constraint rollout ledgers", child_payload.get("rollout_ledgers")
    )
    if list(child_ledgers[: len(parent_ledgers)]) != list(parent_ledgers):
        raise BetaCheckpointEvidenceError(
            "resumed rollout ledger is not an exact parent extension"
        )
    if len(child_ledgers) <= len(parent_ledgers):
        raise BetaCheckpointEvidenceError(
            "resumed rollout ledger does not advance beyond parent"
        )


def _validate_campaign_identity(
    *,
    protocol_id: object,
    algorithm_id: object,
    run_kind: object,
    optimizer_seed: object,
    total_timesteps: object,
    artifact_sha256: object,
) -> tuple[str, str, str, int, int, str]:
    if protocol_id != BETA_CONSTRAINED_RL_PROTOCOL_ID:
        raise BetaCheckpointEvidenceError("beta checkpoint protocol differs")
    if not isinstance(algorithm_id, str) or algorithm_id not in (
        BETA_CORE_ALGORITHM_IDS
    ):
        raise BetaCheckpointEvidenceError(
            "beta checkpoint algorithm is not a frozen PI/I/scalar family"
        )
    if not isinstance(run_kind, str) or run_kind not in BETA_RUN_KINDS:
        raise BetaCheckpointEvidenceError(
            "beta checkpoint run kind must be fixture or registered"
        )
    normalized_seed = _integer("optimizer_seed", optimizer_seed)
    if normalized_seed not in BETA_OPTIMIZER_SEEDS:
        raise BetaCheckpointEvidenceError(
            "optimizer_seed is not in the frozen five-seed set"
        )
    normalized_timesteps = _integer("total_timesteps", total_timesteps, minimum=1)
    if normalized_timesteps % VECTOR_ENV_COUNT:
        raise BetaCheckpointEvidenceError(
            "total_timesteps must be aligned to four vector environments"
        )
    if (
        run_kind == BETA_FIXTURE_RUN_KIND
        and normalized_timesteps > BETA_FIXTURE_MAX_COLLECTED_TIMESTEPS
    ):
        raise BetaCheckpointEvidenceError(
            "fixture checkpoint exceeds the bounded fixture decision budget"
        )
    if (
        run_kind == BETA_REGISTERED_RUN_KIND
        and normalized_timesteps != BETA_REGISTERED_TOTAL_TIMESTEPS
    ):
        raise BetaCheckpointEvidenceError(
            "registered checkpoint must contain exactly 1,048,576 transitions"
        )
    return (
        BETA_CONSTRAINED_RL_PROTOCOL_ID,
        algorithm_id,
        run_kind,
        normalized_seed,
        normalized_timesteps,
        _sha256("artifact_sha256", artifact_sha256),
    )


@dataclass(frozen=True, slots=True)
class BetaRLCheckpointEvidenceV1:
    """Immutable canonical evidence binding one beta policy artifact."""

    protocol_id: str
    algorithm_id: str
    run_kind: str
    optimizer_seed: int
    total_timesteps: int
    artifact_sha256: str
    runtime_provenance: BetaTrainingRuntimeProvenanceV1
    resumed_from_parent: bool
    exact_endpoint_admitted: bool
    eligible_for_registered_evaluation: bool
    _allocator_snapshot_json: str = field(repr=False)
    _constraint_state_json: str = field(repr=False)
    _fixture_resume_json: str | None = field(repr=False)
    _resume_parent_json: str | None = field(repr=False)

    def __post_init__(self) -> None:
        (
            protocol_id,
            algorithm_id,
            run_kind,
            optimizer_seed,
            total_timesteps,
            artifact_sha256,
        ) = _validate_campaign_identity(
            protocol_id=self.protocol_id,
            algorithm_id=self.algorithm_id,
            run_kind=self.run_kind,
            optimizer_seed=self.optimizer_seed,
            total_timesteps=self.total_timesteps,
            artifact_sha256=self.artifact_sha256,
        )
        for name, observed, expected in (
            ("protocol_id", self.protocol_id, protocol_id),
            ("algorithm_id", self.algorithm_id, algorithm_id),
            ("run_kind", self.run_kind, run_kind),
            ("optimizer_seed", self.optimizer_seed, optimizer_seed),
            ("total_timesteps", self.total_timesteps, total_timesteps),
            ("artifact_sha256", self.artifact_sha256, artifact_sha256),
        ):
            if observed != expected:  # pragma: no cover - normalized inputs agree
                raise BetaCheckpointEvidenceError(f"{name} is not canonical")
        if not isinstance(self.runtime_provenance, BetaTrainingRuntimeProvenanceV1):
            raise TypeError(
                "runtime_provenance must be BetaTrainingRuntimeProvenanceV1"
            )
        # Reparse so a future alternate constructor cannot bypass its digest.
        BetaTrainingRuntimeProvenanceV1.from_dict(self.runtime_provenance.to_dict())
        resumed = _boolean("resumed_from_parent", self.resumed_from_parent)
        endpoint = _boolean("exact_endpoint_admitted", self.exact_endpoint_admitted)
        eligible = _boolean(
            "eligible_for_registered_evaluation",
            self.eligible_for_registered_evaluation,
        )
        _, snapshot_raw = _canonical_json_text(
            "allocator snapshot", self._allocator_snapshot_json
        )
        snapshot_object = _parse_allocator_snapshot(snapshot_raw)
        snapshot = snapshot_object.to_dict()
        _validate_allocator_progress(snapshot_object, total_timesteps=total_timesteps)
        _, constraint_raw = _canonical_json_text(
            "constraint state", self._constraint_state_json
        )
        _parse_constraint_state(
            constraint_raw,
            protocol_id=protocol_id,
            algorithm_id=algorithm_id,
            optimizer_seed=optimizer_seed,
            total_timesteps=total_timesteps,
            allocator_snapshot=snapshot,
        )

        fixture_resume: dict[str, Any] | None
        if self._fixture_resume_json is None:
            fixture_resume = None
        else:
            _, fixture_raw = _canonical_json_text(
                "fixture resume", self._fixture_resume_json
            )
            fixture_resume = _parse_fixture_resume(
                fixture_raw,
                artifact_sha256=artifact_sha256,
                allocator_snapshot=snapshot,
            )

        resume_parent: dict[str, Any] | None
        if self._resume_parent_json is None:
            resume_parent = None
        else:
            _, resume_parent_raw = _canonical_json_text(
                "resume parent", self._resume_parent_json
            )
            resume_parent = _parse_resume_parent(
                resume_parent_raw,
                current_artifact_sha256=artifact_sha256,
                current_total_timesteps=total_timesteps,
            )

        identity = cast(Mapping[str, Any], snapshot["allocator_identity"])
        expected_mode = run_kind
        expected_split = (
            EpisodeSeedSplitV1.FIXTURE.value
            if run_kind == BETA_FIXTURE_RUN_KIND
            else EpisodeSeedSplitV1.TRAINING.value
        )
        if identity["mode"] != expected_mode or identity["split"] != expected_split:
            raise BetaCheckpointEvidenceError(
                "allocator mode/split differs from checkpoint run kind"
            )
        if endpoint != eligible:
            raise BetaCheckpointEvidenceError(
                "exact_endpoint_admitted must be true exactly when the checkpoint "
                "is registered-evaluation eligible"
            )
        if resumed != (resume_parent is not None):
            raise BetaCheckpointEvidenceError(
                "resumed_from_parent must be true exactly when resume_parent is present"
            )
        if run_kind == BETA_FIXTURE_RUN_KIND:
            if fixture_resume is None:
                raise BetaCheckpointEvidenceError(
                    "fixture checkpoint requires an artifact-bound resume envelope"
                )
            if endpoint or eligible:
                raise BetaCheckpointEvidenceError(
                    "fixture checkpoint is never registered-evaluation eligible"
                )
        else:
            if fixture_resume is not None:
                raise BetaCheckpointEvidenceError(
                    "registered checkpoint forbids fixture resume evidence"
                )
            if resumed:
                raise BetaCheckpointEvidenceError(
                    "registered checkpoint forbids a resume parent"
                )
            if resume_parent is not None:  # pragma: no cover - equivalence catches this
                raise BetaCheckpointEvidenceError(
                    "registered checkpoint requires a null resume parent"
                )
            if not self.runtime_provenance.worktree_clean:
                raise BetaCheckpointEvidenceError(
                    "registered checkpoint requires clean runtime provenance"
                )
            if eligible:
                _validate_registered_endpoint(
                    allocator_snapshot=snapshot,
                    constraint_state=constraint_raw,
                )

    @property
    def allocator_snapshot(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self._allocator_snapshot_json))

    @property
    def constraint_state(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self._constraint_state_json))

    @property
    def fixture_resume(self) -> dict[str, Any] | None:
        if self._fixture_resume_json is None:
            return None
        return cast(dict[str, Any], json.loads(self._fixture_resume_json))

    @property
    def resume_parent(self) -> dict[str, Any] | None:
        if self._resume_parent_json is None:
            return None
        return cast(dict[str, Any], json.loads(self._resume_parent_json))

    def body(self) -> dict[str, Any]:
        return {
            "schema": BETA_CHECKPOINT_EVIDENCE_SCHEMA,
            "protocol_id": self.protocol_id,
            "algorithm_id": self.algorithm_id,
            "run_kind": self.run_kind,
            "optimizer_seed": self.optimizer_seed,
            "total_timesteps": self.total_timesteps,
            "artifact_sha256": self.artifact_sha256,
            "runtime_provenance": self.runtime_provenance.to_dict(),
            "allocator_snapshot": self.allocator_snapshot,
            "constraint_state": self.constraint_state,
            "fixture_resume": self.fixture_resume,
            "resume_parent": self.resume_parent,
            "resumed_from_parent": self.resumed_from_parent,
            "exact_endpoint_admitted": self.exact_endpoint_admitted,
            "eligible_for_registered_evaluation": (
                self.eligible_for_registered_evaluation
            ),
        }

    @property
    def sha256(self) -> str:
        return cast(str, canonical_sha256(self.body()))

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: object) -> BetaRLCheckpointEvidenceV1:
        raw = _canonical_copy("beta checkpoint evidence", value)
        _exact_keys("beta checkpoint evidence", raw, _TOP_LEVEL_FIELDS)
        if raw["schema"] != BETA_CHECKPOINT_EVIDENCE_SCHEMA:
            raise BetaCheckpointEvidenceError("beta checkpoint schema differs")
        result = _build_beta_checkpoint_evidence(
            protocol_id=raw["protocol_id"],
            algorithm_id=raw["algorithm_id"],
            run_kind=raw["run_kind"],
            optimizer_seed=raw["optimizer_seed"],
            total_timesteps=raw["total_timesteps"],
            artifact_sha256=raw["artifact_sha256"],
            runtime_provenance=raw["runtime_provenance"],
            allocator_snapshot=raw["allocator_snapshot"],
            constraint_state=raw["constraint_state"],
            fixture_resume=raw["fixture_resume"],
            resume_parent=raw["resume_parent"],
            resumed_from_parent=raw["resumed_from_parent"],
            exact_endpoint_admitted=raw["exact_endpoint_admitted"],
            eligible_for_registered_evaluation=(
                raw["eligible_for_registered_evaluation"]
            ),
        )
        supplied_digest = _sha256("beta checkpoint sha256", raw["sha256"])
        if supplied_digest != result.sha256:
            raise BetaCheckpointEvidenceError("beta checkpoint digest differs")
        return result


def _build_beta_checkpoint_evidence(
    *,
    algorithm_id: object,
    run_kind: object,
    optimizer_seed: object,
    total_timesteps: object,
    artifact_sha256: object,
    runtime_provenance: BetaTrainingRuntimeProvenanceV1 | Mapping[str, Any],
    allocator_snapshot: Mapping[str, Any],
    constraint_state: Mapping[str, Any],
    fixture_resume: Mapping[str, Any] | None,
    resume_parent: Mapping[str, Any] | None,
    resumed_from_parent: object,
    exact_endpoint_admitted: object,
    eligible_for_registered_evaluation: object,
    protocol_id: object = BETA_CONSTRAINED_RL_PROTOCOL_ID,
) -> BetaRLCheckpointEvidenceV1:
    """Construct canonical evidence, including unauthenticated parsed claims."""

    (
        normalized_protocol,
        normalized_algorithm,
        normalized_kind,
        normalized_seed,
        normalized_timesteps,
        normalized_artifact,
    ) = _validate_campaign_identity(
        protocol_id=protocol_id,
        algorithm_id=algorithm_id,
        run_kind=run_kind,
        optimizer_seed=optimizer_seed,
        total_timesteps=total_timesteps,
        artifact_sha256=artifact_sha256,
    )
    if isinstance(runtime_provenance, BetaTrainingRuntimeProvenanceV1):
        runtime = BetaTrainingRuntimeProvenanceV1.from_dict(
            runtime_provenance.to_dict()
        )
    else:
        runtime = BetaTrainingRuntimeProvenanceV1.from_dict(runtime_provenance)
    snapshot = _parse_allocator_snapshot(allocator_snapshot).to_dict()
    constraint = _parse_constraint_state(
        constraint_state,
        protocol_id=normalized_protocol,
        algorithm_id=normalized_algorithm,
        optimizer_seed=normalized_seed,
        total_timesteps=normalized_timesteps,
        allocator_snapshot=snapshot,
    )
    resume = (
        None
        if fixture_resume is None
        else _parse_fixture_resume(
            fixture_resume,
            artifact_sha256=normalized_artifact,
            allocator_snapshot=snapshot,
        )
    )
    parent = (
        None
        if resume_parent is None
        else _parse_resume_parent(
            resume_parent,
            current_artifact_sha256=normalized_artifact,
            current_total_timesteps=normalized_timesteps,
        )
    )
    return BetaRLCheckpointEvidenceV1(
        protocol_id=normalized_protocol,
        algorithm_id=normalized_algorithm,
        run_kind=normalized_kind,
        optimizer_seed=normalized_seed,
        total_timesteps=normalized_timesteps,
        artifact_sha256=normalized_artifact,
        runtime_provenance=runtime,
        resumed_from_parent=_boolean("resumed_from_parent", resumed_from_parent),
        exact_endpoint_admitted=_boolean(
            "exact_endpoint_admitted", exact_endpoint_admitted
        ),
        eligible_for_registered_evaluation=_boolean(
            "eligible_for_registered_evaluation",
            eligible_for_registered_evaluation,
        ),
        _allocator_snapshot_json=_canonical_json(snapshot),
        _constraint_state_json=_canonical_json(constraint),
        _fixture_resume_json=(None if resume is None else _canonical_json(resume)),
        _resume_parent_json=(None if parent is None else _canonical_json(parent)),
    )


def build_beta_checkpoint_evidence(
    *,
    algorithm_id: object,
    run_kind: object,
    optimizer_seed: object,
    total_timesteps: object,
    artifact_sha256: object,
    runtime_provenance: BetaTrainingRuntimeProvenanceV1 | Mapping[str, Any],
    allocator_snapshot: Mapping[str, Any],
    constraint_state: Mapping[str, Any],
    fixture_resume: Mapping[str, Any] | None,
    resume_parent: Mapping[str, Any] | None,
    resumed_from_parent: object,
    exact_endpoint_admitted: object,
    eligible_for_registered_evaluation: object,
    artifact_attestation: object | None = None,
    protocol_id: object = BETA_CONSTRAINED_RL_PROTOCOL_ID,
) -> BetaRLCheckpointEvidenceV1:
    """Build evidence; positive eligibility requires a load-issued capability."""

    requested_endpoint = _boolean("exact_endpoint_admitted", exact_endpoint_admitted)
    requested_eligibility = _boolean(
        "eligible_for_registered_evaluation",
        eligible_for_registered_evaluation,
    )
    derived_eligibility = (
        run_kind == BETA_REGISTERED_RUN_KIND and artifact_attestation is not None
    )
    if requested_endpoint != derived_eligibility or (
        requested_eligibility != derived_eligibility
    ):
        raise BetaCheckpointEvidenceError(
            "registered endpoint admission and eligibility are derived only from "
            "an artifact-load attestation issued by ConstrainedPPO.load"
        )
    if artifact_attestation is not None and run_kind != BETA_REGISTERED_RUN_KIND:
        raise BetaCheckpointEvidenceError(
            "only a registered checkpoint can consume an eligibility attestation"
        )
    evidence = _build_beta_checkpoint_evidence(
        algorithm_id=algorithm_id,
        run_kind=run_kind,
        optimizer_seed=optimizer_seed,
        total_timesteps=total_timesteps,
        artifact_sha256=artifact_sha256,
        runtime_provenance=runtime_provenance,
        allocator_snapshot=allocator_snapshot,
        constraint_state=constraint_state,
        fixture_resume=fixture_resume,
        resume_parent=resume_parent,
        resumed_from_parent=resumed_from_parent,
        exact_endpoint_admitted=derived_eligibility,
        eligible_for_registered_evaluation=derived_eligibility,
        protocol_id=protocol_id,
    )
    if derived_eligibility:
        _validate_artifact_attestation(
            evidence,
            artifact_attestation,
            name="artifact attestation",
        )
    return evidence


def parse_beta_checkpoint_evidence(value: object) -> BetaRLCheckpointEvidenceV1:
    """Strictly parse self-authenticating evidence without admitting an artifact."""

    return BetaRLCheckpointEvidenceV1.from_dict(value)


def admit_beta_checkpoint_evidence(
    value: object,
    *,
    expected_artifact_sha256: object,
    expected_algorithm_id: object,
    expected_run_kind: object,
    expected_optimizer_seed: object,
    expected_resume_parent: Mapping[str, Any] | None,
    expected_runtime_provenance: BetaTrainingRuntimeProvenanceV1 | Mapping[str, Any],
    expected_artifact_attestation: object,
    expected_total_timesteps: object | None = None,
    expected_resume_parent_evidence: Mapping[str, Any] | None = None,
    expected_parent_artifact_attestation: object | None = None,
    require_registered_evaluation_eligibility: bool = False,
) -> BetaRLCheckpointEvidenceV1:
    """Admit evidence against external artifact and campaign custody.

    Self-hashes prove integrity, not authenticity.  Callers must supply the hash
    calculated from the artifact they actually opened and the campaign identity
    they intended to load.
    """

    evidence = parse_beta_checkpoint_evidence(value)
    expected_artifact = _sha256("expected_artifact_sha256", expected_artifact_sha256)
    if evidence.artifact_sha256 != expected_artifact:
        raise BetaCheckpointEvidenceError(
            "checkpoint artifact differs from external custody"
        )
    if expected_algorithm_id != evidence.algorithm_id:
        raise BetaCheckpointEvidenceError(
            "checkpoint algorithm differs from requested campaign"
        )
    if expected_run_kind != evidence.run_kind:
        raise BetaCheckpointEvidenceError(
            "checkpoint run kind differs from requested campaign"
        )
    expected_seed = _integer("expected_optimizer_seed", expected_optimizer_seed)
    if expected_seed != evidence.optimizer_seed:
        raise BetaCheckpointEvidenceError(
            "checkpoint optimizer seed differs from requested campaign"
        )
    expected_runtime = _runtime_provenance(expected_runtime_provenance)
    if evidence.runtime_provenance != expected_runtime:
        raise BetaCheckpointEvidenceError(
            "checkpoint runtime/source identity differs from external custody"
        )
    if expected_total_timesteps is not None:
        expected_timesteps = _integer(
            "expected_total_timesteps", expected_total_timesteps, minimum=1
        )
        if expected_timesteps != evidence.total_timesteps:
            raise BetaCheckpointEvidenceError(
                "checkpoint timesteps differ from requested campaign"
            )
    expected_parent = (
        None
        if expected_resume_parent is None
        else _parse_resume_parent(
            expected_resume_parent,
            current_artifact_sha256=evidence.artifact_sha256,
            current_total_timesteps=evidence.total_timesteps,
        )
    )
    if evidence.resume_parent != expected_parent:
        raise BetaCheckpointEvidenceError(
            "checkpoint resume parent differs from external custody"
        )
    if evidence.resumed_from_parent:
        if expected_resume_parent_evidence is None:
            raise BetaCheckpointEvidenceError(
                "resumed checkpoint requires externally admitted parent evidence"
            )
        if expected_parent_artifact_attestation is None:
            raise BetaCheckpointEvidenceError(
                "resumed checkpoint requires a parent artifact-load attestation"
            )
        parent_evidence = parse_beta_checkpoint_evidence(
            expected_resume_parent_evidence
        )
        if expected_parent is None:  # pragma: no cover - model invariant
            raise BetaCheckpointEvidenceError("resumed checkpoint has no parent")
        _validate_resume_lineage(
            child=evidence,
            parent=parent_evidence,
            parent_identity=expected_parent,
        )
        parent_attestation = _validate_artifact_attestation(
            parent_evidence,
            expected_parent_artifact_attestation,
            name="parent artifact attestation",
        )
        _validate_artifact_attestation(
            evidence,
            expected_artifact_attestation,
            name="artifact attestation",
        )
        _validate_resume_receipt(
            child=evidence,
            parent=parent_evidence,
            parent_attestation=parent_attestation,
        )
    elif (
        expected_resume_parent_evidence is not None
        or expected_parent_artifact_attestation is not None
    ):
        raise BetaCheckpointEvidenceError(
            "fresh checkpoint cannot admit parent lineage evidence"
        )
    if not isinstance(require_registered_evaluation_eligibility, bool):
        raise TypeError("require_registered_evaluation_eligibility must be a boolean")
    if (
        require_registered_evaluation_eligibility
        and not evidence.eligible_for_registered_evaluation
    ):
        raise BetaCheckpointEvidenceError(
            "checkpoint is not admitted for registered evaluation"
        )
    _validate_artifact_attestation(
        evidence,
        expected_artifact_attestation,
        name="artifact attestation",
    )
    return evidence


__all__ = [
    "BETA_CHECKPOINT_EVIDENCE_SCHEMA",
    "BETA_CHECKPOINT_RESUME_PARENT_SCHEMA",
    "BetaCheckpointEvidenceError",
    "BetaRLCheckpointEvidenceV1",
    "CONSTRAINT_STATE_ENVELOPE_SCHEMA",
    "CONSTRAINT_STATE_SCHEMA",
    "EXTERNAL_EPISODE_STATE_PIN_SCHEMA",
    "admit_beta_checkpoint_evidence",
    "build_beta_checkpoint_evidence",
    "build_beta_resume_parent_identity",
    "parse_beta_checkpoint_evidence",
]

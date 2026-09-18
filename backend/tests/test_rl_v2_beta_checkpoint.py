"""Strict beta-RL checkpoint evidence tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from typing import Any, Mapping

import pytest

from plant_backend.rl.v2.training.beta_checkpoint import (
    BETA_CHECKPOINT_EVIDENCE_SCHEMA,
    BETA_CHECKPOINT_RESUME_PARENT_SCHEMA,
    CONSTRAINT_STATE_ENVELOPE_SCHEMA,
    CONSTRAINT_STATE_SCHEMA,
    EXTERNAL_EPISODE_STATE_PIN_SCHEMA,
    BetaCheckpointEvidenceError,
    BetaRLCheckpointEvidenceV1,
    admit_beta_checkpoint_evidence,
    build_beta_checkpoint_evidence,
    build_beta_resume_parent_identity,
    parse_beta_checkpoint_evidence,
)
from plant_backend.rl.v2.training.config import (
    BETA_CONSTRAINED_RL_PROTOCOL_ID,
    BETA_CORE_ALGORITHM_IDS,
    BETA_FIXTURE_RUN_KIND,
    BETA_PI_LAGRANGIAN_ALGORITHM_ID,
    BETA_REGISTERED_RUN_KIND,
)
from plant_backend.rl.v2.training.episode_allocator import (
    FIXTURE_RESUME_SCHEMA,
    BetaManifestIdentityV1,
    EpisodeAllocatorSnapshotV1,
    EpisodeCursorV1,
    EpisodeSeedAllocatorV1,
    EpisodeSeedSplitV1,
    ResetManifestLedgerEntryV1,
    canonical_sha256,
)
from plant_backend.rl.v2.training.runtime_provenance import (
    BetaTrainingRuntimeProvenanceV1,
    WORKTREE_IDENTITY_SCHEMA,
)


ARTIFACT_SHA = hashlib.sha256(b"policy artifact").hexdigest()
OTHER_ARTIFACT_SHA = hashlib.sha256(b"substituted artifact").hexdigest()
OPTIMIZER_SEED = 101
TIMESTEPS = 2_688


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _digest_segment(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def _empty_worktree_content_sha256() -> str:
    digest = hashlib.sha256()
    _digest_segment(digest, WORKTREE_IDENTITY_SCHEMA.encode("ascii"))
    _digest_segment(digest, b"")
    _digest_segment(digest, b"")
    return digest.hexdigest()


def _runtime(*, clean: bool) -> BetaTrainingRuntimeProvenanceV1:
    return BetaTrainingRuntimeProvenanceV1(
        git_head="a" * 40,
        worktree_clean=clean,
        worktree_status_sha256=_sha("" if clean else "dirty status"),
        worktree_content_sha256=(
            _empty_worktree_content_sha256() if clean else _sha("dirty content")
        ),
        uv_lock_sha256=_sha("lock"),
        dependency_entrypoints_sha256=_sha("entrypoints"),
        python_version="3.12.11",
        python_implementation="CPython",
        platform="macOS-15-arm64",
        numpy_version="2.2.6",
        scipy_version="1.15.3",
        torch_version="2.7.1",
        stable_baselines3_version="2.9.0",
        gymnasium_version="1.2.0",
        plant_sim_version="0.5.2",
        plant_sim_module_path="simulator/src/plant_sim/__init__.py",
        plant_sim_module_sha256=_sha("plant-sim"),
    )


def _domain_digest(schema: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(schema.encode("ascii") + b"\x00" + encoded).hexdigest()


def _external_pin(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(snapshot))
    return {
        "schema": EXTERNAL_EPISODE_STATE_PIN_SCHEMA,
        "payload": payload,
        "sha256": _domain_digest(EXTERNAL_EPISODE_STATE_PIN_SCHEMA, payload),
    }


def _constraint_state(
    *,
    snapshot: Mapping[str, Any],
    algorithm_id: str = BETA_PI_LAGRANGIAN_ALGORITHM_ID,
    optimizer_seed: int = OPTIMIZER_SEED,
    total_timesteps: int = TIMESTEPS,
    extra_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": CONSTRAINT_STATE_SCHEMA,
        "protocol_id": BETA_CONSTRAINED_RL_PROTOCOL_ID,
        "algorithm_id": algorithm_id,
        "settings": {
            "schema": "constrained-ppo-settings-v1",
            "protocol_id": BETA_CONSTRAINED_RL_PROTOCOL_ID,
            "algorithm_id": algorithm_id,
            "optimizer_seed": optimizer_seed,
        },
        "num_timesteps": total_timesteps,
        "external_episode_state_pin": _external_pin(snapshot),
    }
    if extra_payload is not None:
        payload.update(deepcopy(dict(extra_payload)))
    return {
        "schema": CONSTRAINT_STATE_ENVELOPE_SCHEMA,
        "payload": payload,
        "sha256": _domain_digest(CONSTRAINT_STATE_ENVELOPE_SCHEMA, payload),
    }


def _fixture_resume(
    *, artifact_sha256: str, snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    body = {
        "schema": FIXTURE_RESUME_SCHEMA,
        "fixture_only": True,
        "eligible_for_registered_evaluation": False,
        "parent_checkpoint_sha256": artifact_sha256,
        "snapshot": deepcopy(dict(snapshot)),
    }
    return {**body, "sha256": canonical_sha256(body)}


def _parent_evidence(*, total_timesteps: int = TIMESTEPS) -> BetaRLCheckpointEvidenceV1:
    return build_beta_checkpoint_evidence(
        **_fixture_inputs(
            total_timesteps=total_timesteps,
            artifact_sha256=OTHER_ARTIFACT_SHA,
        )
    )


def _resume_parent(*, total_timesteps: int = TIMESTEPS) -> dict[str, Any]:
    parent = _parent_evidence(total_timesteps=total_timesteps)
    return build_beta_resume_parent_identity(
        metadata_sha256=_sha("parent metadata"),
        policy_artifact_sha256=OTHER_ARTIFACT_SHA,
        checkpoint_contract_digest=_sha("parent contract"),
        checkpoint_evidence_sha256=parent.sha256,
        allocator_snapshot_sha256=parent.allocator_snapshot["sha256"],
        constraint_state_sha256=parent.constraint_state["sha256"],
        total_timesteps=total_timesteps,
    )


def _snapshot_at(*, total_timesteps: int, registered: bool = False) -> dict[str, Any]:
    allocator = (
        EpisodeSeedAllocatorV1.registered()
        if registered
        else EpisodeSeedAllocatorV1.fixture()
    )
    decisions = total_timesteps // 4
    current_ordinal, decision_offset = divmod(decisions, allocator.episode_decisions)
    entries: list[ResetManifestLedgerEntryV1] = []
    for ordinal in range(current_ordinal + 1):
        for environment_index in range(4):
            episode_seed = allocator.seed_for(environment_index, ordinal)
            entries.append(
                ResetManifestLedgerEntryV1(
                    ledger_index=len(entries),
                    allocator_identity_sha256=allocator.identity_sha256,
                    split=allocator.split,
                    environment_index=environment_index,
                    local_episode_ordinal=ordinal,
                    global_episode_ordinal=4 * ordinal + environment_index,
                    episode_seed=episode_seed,
                    ignored_framework_seed=(
                        101 + environment_index if ordinal == 0 else None
                    ),
                    manifest_identity=BetaManifestIdentityV1(
                        manifest_sha256=_sha(f"manifest:{episode_seed}"),
                        alpha_physical_exogenous_episode_sha256=_sha(
                            f"alpha:{episode_seed}"
                        ),
                        beta_exogenous_episode_sha256=_sha(f"beta:{episode_seed}"),
                    ),
                )
            )
    cursors = tuple(
        EpisodeCursorV1(
            environment_index=environment_index,
            current_episode_ordinal=current_ordinal,
            decision_offset=decision_offset,
            next_episode_ordinal=current_ordinal + 1,
            awaiting_reset=False,
        )
        for environment_index in range(4)
    )
    return EpisodeAllocatorSnapshotV1(
        allocator_identity=allocator.identity,
        cursors=cursors,
        ledger=tuple(entries),
    ).to_dict()


def _snapshot_object(value: Mapping[str, Any]) -> EpisodeAllocatorSnapshotV1:
    ledger = value["ledger"]
    assert isinstance(ledger, Mapping)
    return EpisodeAllocatorSnapshotV1(
        allocator_identity=value["allocator_identity"],
        cursors=tuple(EpisodeCursorV1.from_dict(item) for item in value["cursors"]),
        ledger=tuple(
            ResetManifestLedgerEntryV1.from_dict(item) for item in ledger["entries"]
        ),
    )


def _fixture_constraint_progress_fields(
    algorithm_id: str, *, total_timesteps: int
) -> dict[str, Any]:
    ledgers = []
    previous: str | None = None
    for rollout_index in range(1, total_timesteps // 384 + 1):
        digest = _sha(f"fixture-rollout:{algorithm_id}:{rollout_index}")
        ledgers.append(
            {
                "algorithm_id": algorithm_id,
                "rollout_index": rollout_index,
                "transition_count": 384,
                "previous_sha256": previous,
                "sha256": digest,
            }
        )
        previous = digest
    return {
        "ppo_configuration": {"schema": "fixture-test-ppo-v1", "n_steps": 96},
        "rollout_ledgers": ledgers,
        "rollout_ledger_sha256": previous,
    }


def _fixture_inputs(
    *,
    algorithm_id: str = BETA_PI_LAGRANGIAN_ALGORITHM_ID,
    optimizer_seed: int = OPTIMIZER_SEED,
    total_timesteps: int = TIMESTEPS,
    artifact_sha256: str = ARTIFACT_SHA,
) -> dict[str, Any]:
    snapshot = _snapshot_at(total_timesteps=total_timesteps)
    return {
        "algorithm_id": algorithm_id,
        "run_kind": BETA_FIXTURE_RUN_KIND,
        "optimizer_seed": optimizer_seed,
        "total_timesteps": total_timesteps,
        "artifact_sha256": artifact_sha256,
        "runtime_provenance": _runtime(clean=False),
        "allocator_snapshot": snapshot,
        "constraint_state": _constraint_state(
            snapshot=snapshot,
            algorithm_id=algorithm_id,
            optimizer_seed=optimizer_seed,
            total_timesteps=total_timesteps,
            extra_payload=_fixture_constraint_progress_fields(
                algorithm_id, total_timesteps=total_timesteps
            ),
        ),
        "fixture_resume": _fixture_resume(
            artifact_sha256=artifact_sha256,
            snapshot=snapshot,
        ),
        "resume_parent": None,
        "resumed_from_parent": False,
        "exact_endpoint_admitted": False,
        "eligible_for_registered_evaluation": False,
    }


def _registered_inputs(
    *,
    algorithm_id: str = BETA_PI_LAGRANGIAN_ALGORITHM_ID,
    optimizer_seed: int = OPTIMIZER_SEED,
    total_timesteps: int = 1_048_576,
    eligible: bool = True,
) -> dict[str, Any]:
    snapshot = _registered_endpoint_snapshot()
    endpoint_fields = _registered_constraint_endpoint_fields(algorithm_id)
    if not eligible:
        endpoint_fields = {}
    return {
        "algorithm_id": algorithm_id,
        "run_kind": BETA_REGISTERED_RUN_KIND,
        "optimizer_seed": optimizer_seed,
        "total_timesteps": total_timesteps,
        "artifact_sha256": ARTIFACT_SHA,
        "runtime_provenance": _runtime(clean=True),
        "allocator_snapshot": snapshot,
        "constraint_state": _constraint_state(
            snapshot=snapshot,
            algorithm_id=algorithm_id,
            optimizer_seed=optimizer_seed,
            total_timesteps=total_timesteps,
            extra_payload=endpoint_fields,
        ),
        "fixture_resume": None,
        "resume_parent": None,
        "resumed_from_parent": False,
        "exact_endpoint_admitted": eligible,
        "eligible_for_registered_evaluation": eligible,
    }


def _registered_endpoint_snapshot() -> dict[str, Any]:
    allocator = EpisodeSeedAllocatorV1.registered()
    identity = allocator.identity
    entries: list[ResetManifestLedgerEntryV1] = []
    for ordinal in range(391):
        for environment_index in range(4):
            episode_seed = 1_000_000 + 4 * ordinal + environment_index
            manifest = BetaManifestIdentityV1(
                manifest_sha256=_sha(f"manifest:{episode_seed}"),
                alpha_physical_exogenous_episode_sha256=_sha(f"alpha:{episode_seed}"),
                beta_exogenous_episode_sha256=_sha(f"beta:{episode_seed}"),
            )
            entries.append(
                ResetManifestLedgerEntryV1(
                    ledger_index=len(entries),
                    allocator_identity_sha256=allocator.identity_sha256,
                    split=EpisodeSeedSplitV1.TRAINING,
                    environment_index=environment_index,
                    local_episode_ordinal=ordinal,
                    global_episode_ordinal=4 * ordinal + environment_index,
                    episode_seed=episode_seed,
                    ignored_framework_seed=(
                        101 + environment_index if ordinal == 0 else None
                    ),
                    manifest_identity=manifest,
                )
            )
    cursors = tuple(
        EpisodeCursorV1(
            environment_index=environment_index,
            current_episode_ordinal=390,
            decision_offset=64,
            next_episode_ordinal=391,
            awaiting_reset=False,
        )
        for environment_index in range(4)
    )
    return EpisodeAllocatorSnapshotV1(
        allocator_identity=identity,
        cursors=cursors,
        ledger=tuple(entries),
    ).to_dict()


def _registered_constraint_endpoint_fields(
    algorithm_id: str,
) -> dict[str, Any]:
    ledgers: list[dict[str, Any]] = []
    previous_sha256: str | None = None
    for rollout_index in range(1, 129):
        digest = _sha(f"rollout:{algorithm_id}:{rollout_index}")
        ledgers.append(
            {
                "algorithm_id": algorithm_id,
                "rollout_index": rollout_index,
                "transition_count": 8_192,
                "previous_sha256": previous_sha256,
                "sha256": digest,
            }
        )
        previous_sha256 = digest
    return {
        "rollout_ledgers": ledgers,
        "rollout_ledger_sha256": previous_sha256,
        "episode_cost_accumulator": {
            "n_envs": 4,
            "next_decision_index": [65, 65, 65, 65],
        },
        "pending_rollout": False,
        "aborted_rollout": None,
        "unledgered_transition_count": 0,
    }


def _rehash_constraint(state: dict[str, Any]) -> None:
    state["sha256"] = _domain_digest(CONSTRAINT_STATE_ENVELOPE_SCHEMA, state["payload"])


def _rehash_top_level(payload: dict[str, Any]) -> None:
    body = dict(payload)
    body.pop("sha256", None)
    payload["sha256"] = canonical_sha256(body)


def _parsed_registered_claim() -> dict[str, Any]:
    inputs = _registered_inputs(eligible=True)
    inputs["exact_endpoint_admitted"] = False
    inputs["eligible_for_registered_evaluation"] = False
    payload = build_beta_checkpoint_evidence(**inputs).to_dict()
    payload["exact_endpoint_admitted"] = True
    payload["eligible_for_registered_evaluation"] = True
    _rehash_top_level(payload)
    return parse_beta_checkpoint_evidence(payload).to_dict()


@pytest.mark.parametrize("algorithm_id", BETA_CORE_ALGORITHM_IDS)
def test_fixture_round_trip_is_canonical_for_every_algorithm(
    algorithm_id: str,
) -> None:
    evidence = build_beta_checkpoint_evidence(
        **_fixture_inputs(algorithm_id=algorithm_id)
    )

    reopened = parse_beta_checkpoint_evidence(evidence.to_dict())

    assert reopened == evidence
    assert reopened.algorithm_id == algorithm_id
    assert reopened.to_dict() == evidence.to_dict()
    assert reopened.to_dict()["schema"] == BETA_CHECKPOINT_EVIDENCE_SCHEMA
    assert reopened.fixture_resume is not None
    assert not reopened.eligible_for_registered_evaluation


def test_registered_evidence_requires_explicit_endpoint_admission() -> None:
    with pytest.raises(BetaCheckpointEvidenceError, match="artifact-load attestation"):
        build_beta_checkpoint_evidence(**_registered_inputs(eligible=True))


def test_registered_positive_bits_cannot_bless_an_earlier_snapshot() -> None:
    inputs = _registered_inputs(eligible=False)
    snapshot = EpisodeSeedAllocatorV1.registered().snapshot().to_dict()
    inputs["allocator_snapshot"] = snapshot
    inputs["constraint_state"] = _constraint_state(
        snapshot=snapshot,
        total_timesteps=1_048_576,
    )
    inputs["exact_endpoint_admitted"] = True
    inputs["eligible_for_registered_evaluation"] = True

    with pytest.raises(BetaCheckpointEvidenceError, match="derived only"):
        build_beta_checkpoint_evidence(**inputs)


def test_registered_checkpoint_rejects_nonprotocol_transition_count() -> None:
    inputs = _registered_inputs(eligible=False)
    inputs["total_timesteps"] = 1_048_572
    snapshot = inputs["allocator_snapshot"]
    assert isinstance(snapshot, Mapping)
    inputs["constraint_state"] = _constraint_state(
        snapshot=snapshot,
        total_timesteps=1_048_572,
    )

    with pytest.raises(BetaCheckpointEvidenceError, match="exactly 1,048,576"):
        build_beta_checkpoint_evidence(**inputs)


def test_builder_detaches_nested_inputs_and_returned_properties() -> None:
    inputs = _fixture_inputs()
    snapshot = inputs["allocator_snapshot"]
    assert isinstance(snapshot, dict)
    evidence = build_beta_checkpoint_evidence(**inputs)
    original = evidence.to_dict()

    snapshot["cursors"][0]["decision_offset"] = 999
    returned_snapshot = evidence.allocator_snapshot
    returned_snapshot["cursors"][0]["decision_offset"] = 888
    returned_state = evidence.constraint_state
    returned_state["payload"]["num_timesteps"] = 0

    assert evidence.to_dict() == original
    with pytest.raises(FrozenInstanceError):
        evidence.total_timesteps = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    "field",
    sorted(
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
    ),
)
def test_parser_rejects_every_missing_top_level_field(field: str) -> None:
    payload = build_beta_checkpoint_evidence(**_fixture_inputs()).to_dict()
    payload.pop(field)

    with pytest.raises(BetaCheckpointEvidenceError, match="missing fields"):
        parse_beta_checkpoint_evidence(payload)


def test_parser_rejects_unknown_top_level_field() -> None:
    payload = build_beta_checkpoint_evidence(**_fixture_inputs()).to_dict()
    payload["unregistered_claim"] = True
    _rehash_top_level(payload)

    with pytest.raises(BetaCheckpointEvidenceError, match="unknown fields"):
        parse_beta_checkpoint_evidence(payload)


def test_parser_rejects_top_level_and_runtime_tampering() -> None:
    payload = build_beta_checkpoint_evidence(**_fixture_inputs()).to_dict()
    payload["sha256"] = "d" * 64
    with pytest.raises(BetaCheckpointEvidenceError, match="digest differs"):
        parse_beta_checkpoint_evidence(payload)

    runtime_tampered = build_beta_checkpoint_evidence(**_fixture_inputs()).to_dict()
    runtime_tampered["runtime_provenance"]["platform"] = "substituted"
    _rehash_top_level(runtime_tampered)
    with pytest.raises(ValueError, match="runtime provenance digest differs"):
        parse_beta_checkpoint_evidence(runtime_tampered)


def test_parser_rejects_allocator_digest_and_self_rehashed_semantic_tamper() -> None:
    payload = build_beta_checkpoint_evidence(**_fixture_inputs()).to_dict()
    payload["allocator_snapshot"]["cursors"][0]["decision_offset"] = 1
    _rehash_top_level(payload)
    with pytest.raises(BetaCheckpointEvidenceError, match="snapshot.*digest"):
        parse_beta_checkpoint_evidence(payload)

    inputs = _fixture_inputs()
    snapshot = deepcopy(inputs["allocator_snapshot"])
    snapshot["cursors"][0]["awaiting_reset"] = True
    snapshot_body = dict(snapshot)
    snapshot_body.pop("sha256")
    snapshot["sha256"] = canonical_sha256(snapshot_body)
    inputs["allocator_snapshot"] = snapshot
    inputs["constraint_state"] = _constraint_state(snapshot=snapshot)
    inputs["fixture_resume"] = _fixture_resume(
        artifact_sha256=ARTIFACT_SHA, snapshot=snapshot
    )
    with pytest.raises(BetaCheckpointEvidenceError, match="reset state differs"):
        build_beta_checkpoint_evidence(**inputs)


def test_constraint_envelope_is_forward_compatible_but_digest_bound() -> None:
    inputs = _fixture_inputs()
    snapshot = inputs["allocator_snapshot"]
    assert isinstance(snapshot, Mapping)
    inputs["constraint_state"] = _constraint_state(
        snapshot=snapshot,
        extra_payload={
            "ppo_policy_identity": {"schema": "future-v1", "sha256": "b" * 64},
            "rng_identity": {"torch": "isolated"},
        },
    )
    evidence = build_beta_checkpoint_evidence(**inputs)
    assert evidence.constraint_state["payload"]["rng_identity"] == {"torch": "isolated"}

    tampered = evidence.to_dict()
    tampered["constraint_state"]["payload"]["rng_identity"]["torch"] = "shared"
    _rehash_top_level(tampered)
    with pytest.raises(BetaCheckpointEvidenceError, match="envelope digest differs"):
        parse_beta_checkpoint_evidence(tampered)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("protocol", "constraint state protocol differs"),
        ("algorithm", "constraint state algorithm differs"),
        ("settings_algorithm", "constraint settings algorithm differs"),
        ("settings_seed", "constraint settings optimizer seed differs"),
        ("timesteps", "constraint state timesteps differ"),
    ],
)
def test_self_rehashed_constraint_substitutions_are_rejected(
    mutation: str, message: str
) -> None:
    inputs = _fixture_inputs()
    state = deepcopy(inputs["constraint_state"])
    if mutation == "protocol":
        state["payload"]["protocol_id"] = "substituted-protocol"
    elif mutation == "algorithm":
        state["payload"]["algorithm_id"] = BETA_CORE_ALGORITHM_IDS[1]
    elif mutation == "settings_algorithm":
        state["payload"]["settings"]["algorithm_id"] = BETA_CORE_ALGORITHM_IDS[1]
    elif mutation == "settings_seed":
        state["payload"]["settings"]["optimizer_seed"] = 211
    else:
        state["payload"]["num_timesteps"] += 4
    _rehash_constraint(state)
    inputs["constraint_state"] = state

    with pytest.raises(BetaCheckpointEvidenceError, match=message):
        build_beta_checkpoint_evidence(**inputs)


def test_constraint_external_pin_must_equal_allocator_snapshot() -> None:
    inputs = _fixture_inputs()
    substituted = (
        EpisodeSeedAllocatorV1.fixture(base_seed=9_000_100).snapshot().to_dict()
    )
    state = deepcopy(inputs["constraint_state"])
    state["payload"]["external_episode_state_pin"] = _external_pin(substituted)
    _rehash_constraint(state)
    inputs["constraint_state"] = state

    with pytest.raises(BetaCheckpointEvidenceError, match="pin differs"):
        build_beta_checkpoint_evidence(**inputs)


def test_external_pin_digest_is_independently_checked() -> None:
    inputs = _fixture_inputs()
    state = deepcopy(inputs["constraint_state"])
    state["payload"]["external_episode_state_pin"]["sha256"] = "c" * 64
    _rehash_constraint(state)
    inputs["constraint_state"] = state

    with pytest.raises(BetaCheckpointEvidenceError, match="pin digest differs"):
        build_beta_checkpoint_evidence(**inputs)


def test_fixture_resume_is_artifact_bound_and_snapshot_exact() -> None:
    inputs = _fixture_inputs()
    resume = deepcopy(inputs["fixture_resume"])
    resume["parent_checkpoint_sha256"] = OTHER_ARTIFACT_SHA
    body = dict(resume)
    body.pop("sha256")
    resume["sha256"] = canonical_sha256(body)
    inputs["fixture_resume"] = resume
    with pytest.raises(BetaCheckpointEvidenceError, match="not bound"):
        build_beta_checkpoint_evidence(**inputs)

    inputs = _fixture_inputs()
    substituted = (
        EpisodeSeedAllocatorV1.fixture(base_seed=9_000_100).snapshot().to_dict()
    )
    inputs["fixture_resume"] = _fixture_resume(
        artifact_sha256=ARTIFACT_SHA, snapshot=substituted
    )
    with pytest.raises(BetaCheckpointEvidenceError, match="snapshot differs"):
        build_beta_checkpoint_evidence(**inputs)


def test_fixture_requires_resume_and_is_never_eligible() -> None:
    missing_resume = _fixture_inputs()
    missing_resume["fixture_resume"] = None
    with pytest.raises(BetaCheckpointEvidenceError, match="requires.*resume"):
        build_beta_checkpoint_evidence(**missing_resume)

    eligible = _fixture_inputs()
    eligible["exact_endpoint_admitted"] = True
    eligible["eligible_for_registered_evaluation"] = True
    with pytest.raises(BetaCheckpointEvidenceError, match="derived only"):
        build_beta_checkpoint_evidence(**eligible)


def test_fixture_resume_parent_presence_exactly_matches_flag() -> None:
    resumed = _fixture_inputs(total_timesteps=TIMESTEPS * 2)
    resumed["resume_parent"] = _resume_parent(total_timesteps=TIMESTEPS)
    resumed["resumed_from_parent"] = True

    evidence = build_beta_checkpoint_evidence(**resumed)

    assert evidence.resumed_from_parent
    assert evidence.resume_parent == resumed["resume_parent"]
    assert evidence.resume_parent["schema"] == BETA_CHECKPOINT_RESUME_PARENT_SCHEMA
    assert parse_beta_checkpoint_evidence(evidence.to_dict()) == evidence

    missing_identity = _fixture_inputs(total_timesteps=TIMESTEPS * 2)
    missing_identity["resumed_from_parent"] = True
    with pytest.raises(BetaCheckpointEvidenceError, match="true exactly when"):
        build_beta_checkpoint_evidence(**missing_identity)

    false_flag = _fixture_inputs(total_timesteps=TIMESTEPS * 2)
    false_flag["resume_parent"] = _resume_parent(total_timesteps=TIMESTEPS)
    with pytest.raises(BetaCheckpointEvidenceError, match="true exactly when"):
        build_beta_checkpoint_evidence(**false_flag)


def test_resume_parent_is_strict_digest_bound_and_precedes_current_cut() -> None:
    inputs = _fixture_inputs(total_timesteps=TIMESTEPS * 2)
    parent = _resume_parent(total_timesteps=TIMESTEPS)
    parent["metadata_sha256"] = _sha("substituted metadata")
    inputs["resume_parent"] = parent
    inputs["resumed_from_parent"] = True
    with pytest.raises(BetaCheckpointEvidenceError, match="digest differs"):
        build_beta_checkpoint_evidence(**inputs)

    inputs = _fixture_inputs(total_timesteps=TIMESTEPS * 2)
    parent = _resume_parent(total_timesteps=TIMESTEPS)
    parent["unknown_parent_field"] = True
    body = dict(parent)
    body.pop("sha256")
    parent["sha256"] = canonical_sha256(body)
    inputs["resume_parent"] = parent
    inputs["resumed_from_parent"] = True
    with pytest.raises(BetaCheckpointEvidenceError, match="unknown fields"):
        build_beta_checkpoint_evidence(**inputs)

    inputs = _fixture_inputs(total_timesteps=TIMESTEPS * 2)
    inputs["resume_parent"] = _resume_parent(total_timesteps=TIMESTEPS * 2)
    inputs["resumed_from_parent"] = True
    with pytest.raises(BetaCheckpointEvidenceError, match="must precede"):
        build_beta_checkpoint_evidence(**inputs)


def test_admission_binds_resume_parent_to_external_custody() -> None:
    inputs = _fixture_inputs(total_timesteps=TIMESTEPS * 2)
    parent = _resume_parent(total_timesteps=TIMESTEPS)
    inputs["resume_parent"] = parent
    inputs["resumed_from_parent"] = True
    evidence = build_beta_checkpoint_evidence(**inputs).to_dict()

    with pytest.raises(BetaCheckpointEvidenceError, match="issued by ConstrainedPPO"):
        admit_beta_checkpoint_evidence(
            evidence,
            expected_artifact_sha256=ARTIFACT_SHA,
            expected_algorithm_id=BETA_PI_LAGRANGIAN_ALGORITHM_ID,
            expected_run_kind=BETA_FIXTURE_RUN_KIND,
            expected_optimizer_seed=OPTIMIZER_SEED,
            expected_resume_parent=parent,
            expected_runtime_provenance=_runtime(clean=False),
            expected_total_timesteps=TIMESTEPS * 2,
            expected_artifact_attestation=evidence["constraint_state"],
            expected_resume_parent_evidence=_parent_evidence().to_dict(),
            expected_parent_artifact_attestation=(_parent_evidence().constraint_state),
        )

    substituted_parent = build_beta_resume_parent_identity(
        metadata_sha256=_sha("different parent metadata"),
        policy_artifact_sha256=OTHER_ARTIFACT_SHA,
        checkpoint_contract_digest=_sha("parent contract"),
        checkpoint_evidence_sha256=_sha("different evidence"),
        allocator_snapshot_sha256=_sha("different allocator"),
        constraint_state_sha256=_sha("different constraint"),
        total_timesteps=TIMESTEPS,
    )
    with pytest.raises(BetaCheckpointEvidenceError, match="external custody"):
        admit_beta_checkpoint_evidence(
            evidence,
            expected_artifact_sha256=ARTIFACT_SHA,
            expected_algorithm_id=BETA_PI_LAGRANGIAN_ALGORITHM_ID,
            expected_run_kind=BETA_FIXTURE_RUN_KIND,
            expected_optimizer_seed=OPTIMIZER_SEED,
            expected_resume_parent=substituted_parent,
            expected_runtime_provenance=_runtime(clean=False),
            expected_artifact_attestation=None,
            expected_total_timesteps=TIMESTEPS * 2,
        )


@pytest.mark.parametrize(("endpoint", "eligible"), [(True, False), (False, True)])
def test_exact_endpoint_flag_is_equivalent_to_eligibility(
    endpoint: bool, eligible: bool
) -> None:
    inputs = _registered_inputs(eligible=False)
    inputs["exact_endpoint_admitted"] = endpoint
    inputs["eligible_for_registered_evaluation"] = eligible

    with pytest.raises(BetaCheckpointEvidenceError, match="derived only"):
        build_beta_checkpoint_evidence(**inputs)


def test_registered_forbids_dirty_runtime_resume_and_fixture_envelope() -> None:
    dirty = _registered_inputs(eligible=False)
    dirty["runtime_provenance"] = _runtime(clean=False)
    with pytest.raises(BetaCheckpointEvidenceError, match="requires clean"):
        build_beta_checkpoint_evidence(**dirty)

    resumed = _registered_inputs(eligible=False)
    resumed["resumed_from_parent"] = True
    resumed["resume_parent"] = _resume_parent(total_timesteps=TIMESTEPS)
    with pytest.raises(BetaCheckpointEvidenceError, match="forbids a resume parent"):
        build_beta_checkpoint_evidence(**resumed)

    with_resume = _registered_inputs(eligible=False)
    snapshot = with_resume["allocator_snapshot"]
    assert isinstance(snapshot, Mapping)
    with_resume["fixture_resume"] = _fixture_resume(
        artifact_sha256=ARTIFACT_SHA, snapshot=snapshot
    )
    with pytest.raises(BetaCheckpointEvidenceError, match="forbids fixture resume"):
        build_beta_checkpoint_evidence(**with_resume)


def test_allocator_run_kind_substitution_is_rejected() -> None:
    inputs = _fixture_inputs()
    registered_snapshot = _snapshot_at(total_timesteps=TIMESTEPS, registered=True)
    inputs["allocator_snapshot"] = registered_snapshot
    inputs["constraint_state"] = _constraint_state(snapshot=registered_snapshot)
    inputs["fixture_resume"] = _fixture_resume(
        artifact_sha256=ARTIFACT_SHA, snapshot=registered_snapshot
    )

    with pytest.raises(BetaCheckpointEvidenceError, match="mode/split differs"):
        build_beta_checkpoint_evidence(**inputs)


def test_admission_binds_external_artifact_and_campaign_identity() -> None:
    evidence = build_beta_checkpoint_evidence(**_fixture_inputs()).to_dict()
    with pytest.raises(BetaCheckpointEvidenceError, match="issued by ConstrainedPPO"):
        admit_beta_checkpoint_evidence(
            evidence,
            expected_artifact_sha256=ARTIFACT_SHA,
            expected_algorithm_id=BETA_PI_LAGRANGIAN_ALGORITHM_ID,
            expected_run_kind=BETA_FIXTURE_RUN_KIND,
            expected_optimizer_seed=OPTIMIZER_SEED,
            expected_resume_parent=None,
            expected_runtime_provenance=_runtime(clean=False),
            expected_artifact_attestation=None,
            expected_total_timesteps=TIMESTEPS,
        )

    cases = (
        ({"expected_artifact_sha256": OTHER_ARTIFACT_SHA}, "artifact differs"),
        (
            {"expected_algorithm_id": BETA_CORE_ALGORITHM_IDS[1]},
            "algorithm differs",
        ),
        ({"expected_run_kind": BETA_REGISTERED_RUN_KIND}, "run kind differs"),
        ({"expected_optimizer_seed": 211}, "optimizer seed differs"),
        ({"expected_total_timesteps": TIMESTEPS + 4}, "timesteps differ"),
    )
    base: dict[str, Any] = {
        "expected_artifact_sha256": ARTIFACT_SHA,
        "expected_algorithm_id": BETA_PI_LAGRANGIAN_ALGORITHM_ID,
        "expected_run_kind": BETA_FIXTURE_RUN_KIND,
        "expected_optimizer_seed": OPTIMIZER_SEED,
        "expected_resume_parent": None,
        "expected_runtime_provenance": _runtime(clean=False),
        "expected_artifact_attestation": None,
        "expected_total_timesteps": TIMESTEPS,
    }
    for update, message in cases:
        kwargs: dict[str, Any] = {**base, **update}
        with pytest.raises(BetaCheckpointEvidenceError, match=message):
            admit_beta_checkpoint_evidence(evidence, **kwargs)


def test_admission_rejects_fully_rehashed_algorithm_substitution() -> None:
    substituted_algorithm = BETA_CORE_ALGORITHM_IDS[1]
    evidence = build_beta_checkpoint_evidence(
        **_fixture_inputs(algorithm_id=substituted_algorithm)
    ).to_dict()

    with pytest.raises(BetaCheckpointEvidenceError, match="algorithm differs"):
        admit_beta_checkpoint_evidence(
            evidence,
            expected_artifact_sha256=ARTIFACT_SHA,
            expected_algorithm_id=BETA_PI_LAGRANGIAN_ALGORITHM_ID,
            expected_run_kind=BETA_FIXTURE_RUN_KIND,
            expected_optimizer_seed=OPTIMIZER_SEED,
            expected_resume_parent=None,
            expected_runtime_provenance=_runtime(clean=False),
            expected_artifact_attestation=None,
            expected_total_timesteps=TIMESTEPS,
        )


def test_registered_eligibility_must_be_explicitly_required_by_evaluator() -> None:
    ineligible = build_beta_checkpoint_evidence(
        **_registered_inputs(eligible=False)
    ).to_dict()
    with pytest.raises(BetaCheckpointEvidenceError, match="not admitted"):
        admit_beta_checkpoint_evidence(
            ineligible,
            expected_artifact_sha256=ARTIFACT_SHA,
            expected_algorithm_id=BETA_PI_LAGRANGIAN_ALGORITHM_ID,
            expected_run_kind=BETA_REGISTERED_RUN_KIND,
            expected_optimizer_seed=OPTIMIZER_SEED,
            expected_resume_parent=None,
            expected_runtime_provenance=_runtime(clean=True),
            expected_artifact_attestation=None,
            expected_total_timesteps=1_048_576,
            require_registered_evaluation_eligibility=True,
        )


def test_allocator_progress_is_exactly_bound_to_total_timesteps() -> None:
    inputs = _fixture_inputs()
    stale_snapshot = _snapshot_at(total_timesteps=TIMESTEPS // 2)
    inputs["allocator_snapshot"] = stale_snapshot
    inputs["constraint_state"] = _constraint_state(
        snapshot=stale_snapshot,
        total_timesteps=TIMESTEPS,
        extra_payload=_fixture_constraint_progress_fields(
            BETA_PI_LAGRANGIAN_ALGORITHM_ID,
            total_timesteps=TIMESTEPS,
        ),
    )
    inputs["fixture_resume"] = _fixture_resume(
        artifact_sha256=ARTIFACT_SHA,
        snapshot=stale_snapshot,
    )

    with pytest.raises(BetaCheckpointEvidenceError, match="reset count differs"):
        build_beta_checkpoint_evidence(**inputs)


def test_allocator_rejects_fully_rehashed_out_of_order_reset_wave() -> None:
    inputs = _fixture_inputs()
    snapshot_raw = inputs["allocator_snapshot"]
    assert isinstance(snapshot_raw, Mapping)
    parsed = _snapshot_object(snapshot_raw)
    entries = list(parsed.ledger)
    entries[0], entries[1] = (
        replace(entries[1], ledger_index=0),
        replace(entries[0], ledger_index=1),
    )
    reordered = EpisodeAllocatorSnapshotV1(
        allocator_identity=deepcopy(dict(snapshot_raw["allocator_identity"])),
        cursors=parsed.cursors,
        ledger=tuple(entries),
    ).to_dict()
    inputs["allocator_snapshot"] = reordered
    inputs["constraint_state"] = _constraint_state(
        snapshot=reordered,
        extra_payload=_fixture_constraint_progress_fields(
            BETA_PI_LAGRANGIAN_ALGORITHM_ID,
            total_timesteps=TIMESTEPS,
        ),
    )
    inputs["fixture_resume"] = _fixture_resume(
        artifact_sha256=ARTIFACT_SHA,
        snapshot=reordered,
    )

    with pytest.raises(BetaCheckpointEvidenceError, match="synchronous global order"):
        build_beta_checkpoint_evidence(**inputs)


def test_registered_claim_requires_exact_artifact_loaded_second_stage() -> None:
    with pytest.raises(BetaCheckpointEvidenceError, match="artifact-load attestation"):
        build_beta_checkpoint_evidence(**_registered_inputs(eligible=True))
    evidence = _parsed_registered_claim()
    base: dict[str, Any] = {
        "expected_artifact_sha256": ARTIFACT_SHA,
        "expected_algorithm_id": BETA_PI_LAGRANGIAN_ALGORITHM_ID,
        "expected_run_kind": BETA_REGISTERED_RUN_KIND,
        "expected_optimizer_seed": OPTIMIZER_SEED,
        "expected_resume_parent": None,
        "expected_runtime_provenance": _runtime(clean=True),
        "expected_artifact_attestation": None,
        "expected_total_timesteps": 1_048_576,
        "require_registered_evaluation_eligibility": True,
    }
    with pytest.raises(BetaCheckpointEvidenceError, match="issued by ConstrainedPPO"):
        admit_beta_checkpoint_evidence(evidence, **base)

    with pytest.raises(
        BetaCheckpointEvidenceError, match="must be issued by ConstrainedPPO.load"
    ):
        admit_beta_checkpoint_evidence(
            evidence,
            **{**base, "expected_artifact_attestation": evidence["constraint_state"]},
        )


def test_admission_rejects_fully_rehashed_runtime_source_substitution() -> None:
    inputs = _fixture_inputs()
    substituted_runtime = _runtime(clean=False).to_dict()
    substituted_runtime["plant_sim_module_sha256"] = _sha("substituted source")
    runtime_body = dict(substituted_runtime)
    runtime_body.pop("sha256")
    substituted_runtime["sha256"] = canonical_sha256(runtime_body)
    inputs["runtime_provenance"] = substituted_runtime
    evidence = build_beta_checkpoint_evidence(**inputs).to_dict()

    with pytest.raises(BetaCheckpointEvidenceError, match="runtime/source identity"):
        admit_beta_checkpoint_evidence(
            evidence,
            expected_artifact_sha256=ARTIFACT_SHA,
            expected_algorithm_id=BETA_PI_LAGRANGIAN_ALGORITHM_ID,
            expected_run_kind=BETA_FIXTURE_RUN_KIND,
            expected_optimizer_seed=OPTIMIZER_SEED,
            expected_resume_parent=None,
            expected_runtime_provenance=_runtime(clean=False),
            expected_artifact_attestation=None,
            expected_total_timesteps=TIMESTEPS,
        )


def _resumed_admission_kwargs(
    *,
    evidence: Mapping[str, Any],
    parent_identity: Mapping[str, Any],
    parent_evidence: BetaRLCheckpointEvidenceV1,
) -> dict[str, Any]:
    return {
        "expected_artifact_sha256": ARTIFACT_SHA,
        "expected_algorithm_id": BETA_PI_LAGRANGIAN_ALGORITHM_ID,
        "expected_run_kind": BETA_FIXTURE_RUN_KIND,
        "expected_optimizer_seed": OPTIMIZER_SEED,
        "expected_resume_parent": parent_identity,
        "expected_runtime_provenance": _runtime(clean=False),
        "expected_total_timesteps": TIMESTEPS * 2,
        "expected_artifact_attestation": evidence["constraint_state"],
        "expected_resume_parent_evidence": parent_evidence.to_dict(),
        "expected_parent_artifact_attestation": (parent_evidence.constraint_state),
    }


def test_resume_lineage_rejects_rehashed_parent_evidence_identity_forgery() -> None:
    parent_evidence = _parent_evidence()
    parent_identity = _resume_parent()
    parent_identity["checkpoint_evidence_sha256"] = _sha("forged evidence")
    body = dict(parent_identity)
    body.pop("sha256")
    parent_identity["sha256"] = canonical_sha256(body)
    inputs = _fixture_inputs(total_timesteps=TIMESTEPS * 2)
    inputs["resume_parent"] = parent_identity
    inputs["resumed_from_parent"] = True
    evidence = build_beta_checkpoint_evidence(**inputs).to_dict()

    with pytest.raises(
        BetaCheckpointEvidenceError, match="checkpoint_evidence_sha256 differs"
    ):
        admit_beta_checkpoint_evidence(
            evidence,
            **_resumed_admission_kwargs(
                evidence=evidence,
                parent_identity=parent_identity,
                parent_evidence=parent_evidence,
            ),
        )


def test_resume_lineage_rejects_allocator_and_constraint_forks() -> None:
    parent_evidence = _parent_evidence()
    parent_identity = _resume_parent()

    allocator_fork = _fixture_inputs(
        total_timesteps=TIMESTEPS * 2,
    )
    forked_snapshot = _snapshot_at(total_timesteps=TIMESTEPS * 2)
    # A real independent allocator makes every nested digest self-consistent.
    independent = EpisodeSeedAllocatorV1.fixture(base_seed=9_000_100)
    original = _snapshot_object(forked_snapshot)
    forked_entries = tuple(
        replace(
            entry,
            allocator_identity_sha256=independent.identity_sha256,
            episode_seed=independent.seed_for(
                entry.environment_index, entry.local_episode_ordinal
            ),
        )
        for entry in original.ledger
    )
    forked_snapshot = EpisodeAllocatorSnapshotV1(
        allocator_identity=independent.identity,
        cursors=original.cursors,
        ledger=forked_entries,
    ).to_dict()
    allocator_fork["allocator_snapshot"] = forked_snapshot
    allocator_fork["constraint_state"] = _constraint_state(
        snapshot=forked_snapshot,
        total_timesteps=TIMESTEPS * 2,
        extra_payload=_fixture_constraint_progress_fields(
            BETA_PI_LAGRANGIAN_ALGORITHM_ID,
            total_timesteps=TIMESTEPS * 2,
        ),
    )
    allocator_fork["fixture_resume"] = _fixture_resume(
        artifact_sha256=ARTIFACT_SHA,
        snapshot=forked_snapshot,
    )
    allocator_fork["resume_parent"] = parent_identity
    allocator_fork["resumed_from_parent"] = True
    allocator_evidence = build_beta_checkpoint_evidence(**allocator_fork).to_dict()
    with pytest.raises(BetaCheckpointEvidenceError, match="allocator identity differs"):
        admit_beta_checkpoint_evidence(
            allocator_evidence,
            **_resumed_admission_kwargs(
                evidence=allocator_evidence,
                parent_identity=parent_identity,
                parent_evidence=parent_evidence,
            ),
        )

    constraint_fork = _fixture_inputs(total_timesteps=TIMESTEPS * 2)
    state = deepcopy(constraint_fork["constraint_state"])
    state["payload"]["rollout_ledgers"][0]["sha256"] = _sha("forked rollout")
    _rehash_constraint(state)
    constraint_fork["constraint_state"] = state
    constraint_fork["resume_parent"] = parent_identity
    constraint_fork["resumed_from_parent"] = True
    constraint_evidence = build_beta_checkpoint_evidence(**constraint_fork).to_dict()
    with pytest.raises(BetaCheckpointEvidenceError, match="rollout ledger.*extension"):
        admit_beta_checkpoint_evidence(
            constraint_evidence,
            **_resumed_admission_kwargs(
                evidence=constraint_evidence,
                parent_identity=parent_identity,
                parent_evidence=parent_evidence,
            ),
        )


@pytest.mark.parametrize(
    ("updates", "exception", "message"),
    [
        ({"protocol_id": "wrong"}, ValueError, "protocol differs"),
        ({"algorithm_id": "ppo-ish"}, ValueError, "algorithm"),
        ({"run_kind": "research"}, ValueError, "run kind"),
        ({"optimizer_seed": True}, TypeError, "must be an integer"),
        ({"optimizer_seed": 999}, ValueError, "five-seed"),
        ({"total_timesteps": 3}, ValueError, "aligned"),
        ({"artifact_sha256": "A" * 64}, ValueError, "lowercase"),
        ({"resumed_from_parent": 0}, TypeError, "must be a boolean"),
    ],
)
def test_builder_rejects_noncanonical_campaign_fields(
    updates: dict[str, Any],
    exception: type[Exception],
    message: str,
) -> None:
    inputs = _fixture_inputs()
    inputs.update(updates)

    with pytest.raises(exception, match=message):
        build_beta_checkpoint_evidence(**inputs)

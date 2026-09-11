"""All-slot synthetic scorecards reconstructed from sealed physical arm files."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..training import custody_io as io
from ..training.custody_json import (
    authenticate,
    canonical_bytes,
    decode,
    require_envelope,
)
from . import arm_evidence as arms
from .contract import EPISODE_COUNT, PREFIX
from .operational import operational_contract, policy_scorecard

SCHEMA = PREFIX + "synthetic-operational-evidence-v1"
FILENAME = "synthetic-operational.json"
CLAIM_BOUNDARY = "synthetic descriptive evidence; no registered checkpoint, stage, registry or result claim"
FIELDS = frozenset(
    {
        "schema",
        "operational_contract_sha256",
        "episode_seeds",
        "cycle_count",
        "arms",
        "result",
        "claim_boundary",
        "result_claim",
        "sha256",
    }
)


def synthetic_operational_evidence(
    episode_seeds: Sequence[int],
    paths: Sequence[Path | str | None],
    *,
    cycle_count: int = 7,
) -> dict[str, Any]:
    # Close every seed/path scope before any underlying manifest can be reopened.
    if (
        len(episode_seeds) != EPISODE_COUNT
        or len(paths) != EPISODE_COUNT
        or len(set(episode_seeds)) != EPISODE_COUNT
    ):
        raise ValueError("synthetic scorecard requires 32 unique fixture slots")
    for seed in episode_seeds:
        arms._fixture_seed(seed)
    targets = [
        (
            None
            if path is None
            else arms.require_synthetic_path(path, filename=arms.FILENAME)
        )
        for path in paths
    ]
    present = [path for path in targets if path is not None]
    if len(set(present)) != len(present):
        raise ValueError("physical arm references must be unique")
    stamps = {path: arms.file_stamp(path) for path in present}
    references: list[dict[str, Any] | None] = []
    records: list[dict[str, Any] | None] = []
    identity = None
    for seed, path in zip(episode_seeds, targets, strict=True):
        if path is None:
            references.append(None)
            records.append(None)
            continue
        raw = io.read_regular(path)
        admitted = arms.verify_synthetic_arm_file(path)
        if admitted["seed"] != seed:
            raise ValueError("physical arm seed differs from its scorecard slot")
        envelope = decode(raw)
        current = {
            key: envelope[key]
            for key in (
                "actor",
                "training_task",
                "checkpoint",
                "source_provenance",
                "service_mpc_settings",
            )
        }
        if identity is not None and canonical_bytes(identity) != canonical_bytes(
            current
        ):
            raise ValueError(
                "operational scorecard mixes actor, task, checkpoint or source"
            )
        identity = current
        if io.read_regular(path) != raw:
            raise ValueError("physical arm changed during operational admission")
        references.append(
            {"file": admitted["file"], "envelope_sha256": admitted["envelope_sha256"]}
        )
        records.append(admitted["operational"])
    result = policy_scorecard(episode_seeds, records, cycle_count=cycle_count)
    for path, reference in zip(targets, references, strict=True):
        if (
            path is not None
            and reference is not None
            and (
                arms.file_stamp(path) != stamps[path]
                or io.file_record(path) != reference["file"]
            )
        ):
            raise ValueError("operational evidence graph changed during admission")
    return authenticate(
        {
            "schema": SCHEMA,
            "operational_contract_sha256": operational_contract()["sha256"],
            "episode_seeds": list(episode_seeds),
            "cycle_count": cycle_count,
            "arms": references,
            "result": result,
            "claim_boundary": CLAIM_BOUNDARY,
            "result_claim": None,
        }
    )


def verify_synthetic_operational_file(path: Path | str) -> dict[str, Any]:
    target = arms.require_synthetic_path(path, filename=FILENAME)
    stamp = arms.file_stamp(target)
    raw = io.read_regular(target)
    report = require_envelope(raw, schema=SCHEMA, fields=FIELDS)
    if (
        report["claim_boundary"] != CLAIM_BOUNDARY
        or report["result_claim"] is not None
        or report["operational_contract_sha256"] != operational_contract()["sha256"]
    ):
        raise ValueError("synthetic scorecard claim/contract differs")
    paths = [
        None if item is None else Path(item["file"]["path"]) for item in report["arms"]
    ]
    source_stamps = {
        p: arms.file_stamp(arms.require_synthetic_path(p, filename=arms.FILENAME))
        for p in paths
        if p is not None
    }
    rebuilt = synthetic_operational_evidence(
        report["episode_seeds"], paths, cycle_count=report["cycle_count"]
    )
    if canonical_bytes(rebuilt) != raw:
        raise ValueError(
            "operational scorecard differs from independently reconstructed arms"
        )
    record = io.file_record(target)
    if (
        io.read_regular(target) != raw
        or arms.file_stamp(target) != stamp
        or any(arms.file_stamp(p) != before for p, before in source_stamps.items())
        or any(
            item is not None
            and io.file_record(Path(item["file"]["path"])) != item["file"]
            for item in rebuilt["arms"]
        )
    ):
        raise ValueError("operational evidence graph changed during verification")
    return {
        "verification": "synthetic_operational_reconstructed",
        "source_evidence_verified": True,
        "registered_checkpoint_verified": False,
        "registered_registry_verified": False,
        "result_claim": None,
        "file": record,
        "result": rebuilt["result"],
    }

"""Detached physical/timing reconstruction from bounded fixture archives."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..training import custody_io as io
from ..training.custody_json import authenticate, canonical_bytes, require_envelope
from . import archive, arm_evidence as arms, latency_evidence as timing
from .contract import PREFIX
from .execution_budget import EvidenceLimits as EvidenceLimits
from .latency import latency_contract

ARM_FILENAME = "synthetic-arm.json.gz"
TIMING_FILENAME = "synthetic-latency.json.gz"
TIMING_SCHEMA = PREFIX + "synthetic-compressed-latency-v1"
TIMING_FIELDS = (timing.FIELDS - {"arm_file"}) | {"arm_archive"}


def _recheck(
    path: Path, record: dict[str, Any], stamp: tuple[int, ...], limit: int
) -> None:
    current = archive._descriptor(
        path,
        plain_bytes=record["uncompressed_bytes"],
        plain_sha256=record["uncompressed_sha256"],
        stored_limit=limit,
    )
    if (
        canonical_bytes(current) != canonical_bytes(record)
        or arms.file_stamp(path) != stamp
    ):
        raise ValueError("compressed study evidence changed during reconstruction")


def _read_arm(
    path: Path, limits: EvidenceLimits
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload, record = archive.read_archive(path, **limits.arm())
    envelope = require_envelope(
        canonical_bytes(payload), schema=arms.SCHEMA, fields=arms.FIELDS
    )
    admitted = arms._verify_fixture_envelope(envelope)
    return envelope, admitted, record


def synthetic_timing_archive(
    arm_path: Path | str,
    measurement: Mapping[str, Any],
    host_before: Mapping[str, Any],
    host_after: Mapping[str, Any],
    *,
    limits: EvidenceLimits,
) -> dict[str, Any]:
    target = arms.require_synthetic_path(arm_path, filename=ARM_FILENAME)
    stamp = arms.file_stamp(target)
    envelope, admitted, record = _read_arm(target, limits)
    before, after, result = timing._project_latency(
        envelope, admitted, measurement, host_before, host_after
    )
    document = authenticate(
        {
            "schema": TIMING_SCHEMA,
            "latency_contract_sha256": latency_contract()["sha256"],
            "arm_archive": record,
            "arm_envelope_sha256": admitted["envelope_sha256"],
            "host_before": before,
            "host_after": after,
            "measurement": dict(measurement),
            "result": result,
            "claim_boundary": timing.CLAIM_BOUNDARY,
            "result_claim": None,
        }
    )
    _recheck(target, record, stamp, limits.arm_stored_bytes)
    return document


def verify_synthetic_execution(
    directory: Path | str, *, limits: EvidenceLimits
) -> dict[str, Any]:
    """Reopen both archives and every physical/timing witness; never execute a run."""
    root = Path(directory)
    arm_path = arms.require_synthetic_path(root / ARM_FILENAME, filename=ARM_FILENAME)
    timing_path = arms.require_synthetic_path(
        root / TIMING_FILENAME, filename=TIMING_FILENAME
    )
    io.require_directory(root)
    if {path.name for path in root.iterdir()} != {ARM_FILENAME, TIMING_FILENAME}:
        raise ValueError("synthetic execution contains undeclared or missing evidence")
    root_stamp = arms.file_stamp(root)
    stamps = {path: arms.file_stamp(path) for path in (arm_path, timing_path)}
    raw_timing, timing_record = archive.read_archive(timing_path, **limits.timing())
    document = require_envelope(
        canonical_bytes(raw_timing), schema=TIMING_SCHEMA, fields=TIMING_FIELDS
    )
    envelope, admitted, arm_record = _read_arm(arm_path, limits)
    before, after, result = timing._project_latency(
        envelope,
        admitted,
        document["measurement"],
        document["host_before"],
        document["host_after"],
    )
    rebuilt = authenticate(
        {
            "schema": TIMING_SCHEMA,
            "latency_contract_sha256": latency_contract()["sha256"],
            "arm_archive": arm_record,
            "arm_envelope_sha256": admitted["envelope_sha256"],
            "host_before": before,
            "host_after": after,
            "measurement": document["measurement"],
            "result": result,
            "claim_boundary": timing.CLAIM_BOUNDARY,
            "result_claim": None,
        }
    )
    if canonical_bytes(rebuilt) != canonical_bytes(document):
        raise ValueError(
            "compressed timing differs from reconstructed physical evidence"
        )
    for path, record, limit in (
        (arm_path, arm_record, limits.arm_stored_bytes),
        (timing_path, timing_record, limits.timing_stored_bytes),
    ):
        _recheck(path, record, stamps[path], limit)
    if arms.file_stamp(root) != root_stamp:
        raise ValueError("compressed execution directory changed during verification")
    return authenticate(
        {
            "schema": PREFIX + "synthetic-compressed-execution-verification-v1",
            "arm_archive": arm_record,
            "timing_archive": timing_record,
            "physical": admitted,
            "latency": result,
            "source_evidence_verified": True,
            "registered_checkpoint_verified": False,
            "execution_authorized": False,
            "result_claim": None,
        }
    )

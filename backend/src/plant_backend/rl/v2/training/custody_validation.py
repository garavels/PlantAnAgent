"""Strict nested receipt shapes reconstructed independently from artifact bytes."""

from __future__ import annotations

from typing import Any

from . import custody_documents as documents
from .custody_json import canonical_bytes
from .registered_contract import LOG_LIMIT_BYTES, TIMEOUT_SECONDS


def exact(name: str, actual: object, expected: object) -> None:
    if canonical_bytes(actual) != canonical_bytes(expected):
        raise ValueError(f"{name} differs")


def integer(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("custody integer is outside its exact range")
    return value


def mapping(value: object, fields: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("custody nested fields differ")
    return value


def process_failures(
    process: object, logs: dict[str, Any], command: list[str]
) -> dict[str, str]:
    record = mapping(process, documents.PROCESS_FIELDS)
    exact("child command", record["command"], command)
    if integer(record["timeout_seconds"], minimum=1) != TIMEOUT_SECONDS:
        raise ValueError("child timeout differs from the frozen limit")
    if type(record["started"]) is not bool:
        raise ValueError("child started flag must be boolean")
    failures: dict[str, str] = {}
    for name in ("stdout", "stderr"):
        descriptor = mapping(logs[name], documents.FILE_FIELDS)
        size = integer(descriptor["logical_bytes"])
        integer(descriptor["allocated_bytes"])
        if size > LOG_LIMIT_BYTES:
            raise ValueError("sealed log exceeds the frozen byte limit")
        if size == LOG_LIMIT_BYTES:
            failures["child_log_limit_exceeded"] = "child"
    if not record["started"]:
        if any(
            record[key] is not None
            for key in documents.PROCESS_FIELDS
            - {"command", "timeout_seconds", "started"}
        ):
            raise ValueError("unstarted child contains process measurements")
        if any(logs[name]["logical_bytes"] != 0 for name in ("stdout", "stderr")):
            raise ValueError("unstarted child has nonempty logs")
        failures["child_spawn_failed"] = "child"
        return failures
    elapsed = integer(record["wall_elapsed_ns"], minimum=1)
    integer(record["child_user_cpu_ns"])
    integer(record["child_system_cpu_ns"])
    exit_code = record["exit_code"]
    if type(exit_code) is not int:
        raise ValueError("a reaped child must have an integer exit status")
    exact(
        "child termination signal",
        record["termination_signal"],
        -exit_code if exit_code < 0 else None,
    )
    if elapsed >= TIMEOUT_SECONDS * 1_000_000_000:
        failures["child_timeout"] = "child"
    if exit_code < 0:
        failures["child_signaled"] = "child"
    elif exit_code != 0:
        failures["child_nonzero_exit"] = "child"
    return failures


def checkpoint_descriptor(value: object, runtime_sha256: str) -> dict[str, Any]:
    record = mapping(value, documents.CHECKPOINT_FIELDS)
    for key, expected in (
        ("rollout_entry_count", 128),
        ("allocator_entry_count", 1564),
        ("optimizer_steps", 40960),
    ):
        if integer(record[key]) != expected:
            raise ValueError("checkpoint endpoint count differs")
    if (
        record["registered_evaluation_eligible"] is not True
        or record["runtime_provenance_sha256"] != runtime_sha256
    ):
        raise ValueError("checkpoint eligibility or runtime differs")
    files = record["files"]
    if not isinstance(files, list) or len(files) != 3:
        raise ValueError("checkpoint must describe three files")
    for item in files:
        mapping(item, documents.FILE_FIELDS)
        integer(item["logical_bytes"])
        integer(item["allocated_bytes"])
    if [item["path"] for item in files] != [
        f"{record['path']}/{name}"
        for name in ("metadata.json", "policy.zip", "training_config.json")
    ]:
        raise ValueError("checkpoint file paths or order differ")
    exact(
        "checkpoint logical size",
        record["logical_bytes"],
        sum(item["logical_bytes"] for item in files),
    )
    exact(
        "checkpoint allocated size",
        record["allocated_bytes"],
        sum(item["allocated_bytes"] for item in files),
    )
    exact("checkpoint metadata identity", record["metadata_sha256"], files[0]["sha256"])
    return record

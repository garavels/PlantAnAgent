"""Closed stage records and bounded control I/O, with no execution authority."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
from typing import Any

from ..training import custody_io as io
from ..training.custody_authorization import utc_timestamp
from ..training.custody_json import authenticate, canonical_bytes, require_envelope
from ..training.custody_validation import exact, integer, mapping
from ..training.custody_documents import PROCESS_FIELDS, FILE_FIELDS
from .archive import _read_stored
from .contract import PREFIX, PROTOCOL_ID, PROTOCOL_SHA256
from .execution_budget import (
    ExecutionBudget,
    ROOT_CONTROL_LIMIT_BYTES,
    CELL_CONTROL_LIMIT_BYTES,
    control_bytes,
)

AUTH_SCHEMA = PREFIX + "stage-authorization-v1"
OPEN_SCHEMA = PREFIX + "stage-opening-v1"
CLAIM_SCHEMA = PREFIX + "stage-cell-claim-v1"
TERMINAL_SCHEMA = PREFIX + "stage-cell-terminal-v1"
REPORT_SCHEMA = PREFIX + "stage-report-v1"
VERDICT_SCHEMA = PREFIX + "stage-verification-v1"
LAUNCH_SCHEMA = PREFIX + "stage-child-launch-v1"
AUTH_FIELDS = frozenset(
    {
        "schema",
        "protocol",
        "authorization_id",
        "authorized_at_utc",
        "stage",
        "registry_sha256",
        "training_inputs_sha256",
        "training_report",
        "source_provenance",
        "latency_host",
        "budget",
        "development",
        "stage_execution_authorized",
        "operator_assertions",
        "sha256",
    }
)
OPEN_FIELDS = frozenset(
    {
        "schema",
        "authorization_sha256",
        "registry_sha256",
        "training_inputs_sha256",
        "stage",
        "parent_pid",
        "lock",
        "opened_at_utc",
        "sha256",
    }
)
CLAIM_FIELDS = frozenset(
    {
        "schema",
        "authorization_sha256",
        "registry_sha256",
        "opening_sha256",
        "cell",
        "previous_terminal_sha256",
        "claimed_at_utc",
        "sha256",
    }
)
TERMINAL_FIELDS = frozenset(
    {
        "schema",
        "claim_sha256",
        "cell",
        "process",
        "logs",
        "inventory",
        "state",
        "failure",
        "arm_archive",
        "timing_archive",
        "finished_at_utc",
        "sha256",
    }
)
REPORT_FIELDS = frozenset(
    {
        "schema",
        "protocol",
        "stage",
        "registry",
        "authorization",
        "opening",
        "training_inputs_sha256",
        "cells",
        "abort",
        "campaign_status",
        "analysis",
        "result_claim",
        "sha256",
    }
)
ASSERTIONS = {
    "no_concurrent_benchmark_jobs": True,
    "budget_and_capacity_reviewed": True,
    "no_retry_repair_or_replacement": True,
    "no_outcome_inspection_before_detached_closure": True,
}
PROCESS_FAILURE_ORDER = (
    "child_log_limit_exceeded",
    "child_timeout",
    "child_spawn_failed",
    "child_signaled",
    "child_nonzero_exit",
)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def read(
    path: Path, schema: str, fields: frozenset[str], *, cell: bool = False
) -> dict[str, Any]:
    raw = _read_stored(
        path, CELL_CONTROL_LIMIT_BYTES if cell else ROOT_CONTROL_LIMIT_BYTES
    )
    return require_envelope(raw, schema=schema, fields=fields)


def publish(path: Path, value: dict[str, Any], *, cell: bool = False) -> None:
    io.publish_bytes(path, control_bytes(value, cell=cell))


def reference(
    path: Path, value: dict[str, Any], *, cell: bool = False
) -> dict[str, Any]:
    raw = _read_stored(
        path, CELL_CONTROL_LIMIT_BYTES if cell else ROOT_CONTROL_LIMIT_BYTES
    )
    if raw != canonical_bytes(value):
        raise ValueError("stage control reference changed")
    return {
        "path": str(path),
        "raw_file_sha256": hashlib.sha256(raw).hexdigest(),
        "sha256": value["sha256"],
    }


def protocol() -> dict[str, str]:
    return {"id": PROTOCOL_ID, "raw_sha256": PROTOCOL_SHA256}


def opening(
    authority: dict[str, Any],
    registry: dict[str, Any],
    *,
    parent_pid: int,
    lock: dict[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    integer(parent_pid, minimum=1)
    utc_timestamp(timestamp)
    if timestamp < authority["authorized_at_utc"]:
        raise ValueError("stage opened before its authorization")
    return authenticate(
        {
            "schema": OPEN_SCHEMA,
            "authorization_sha256": authority["sha256"],
            "registry_sha256": registry["sha256"],
            "training_inputs_sha256": authority["training_inputs_sha256"],
            "stage": registry["stage"],
            "parent_pid": parent_pid,
            "lock": lock,
            "opened_at_utc": timestamp,
        }
    )


def claim(
    registry: dict[str, Any],
    opened: dict[str, Any],
    cell: dict[str, Any],
    previous: dict[str, Any] | None,
    *,
    timestamp: str,
) -> dict[str, Any]:
    utc_timestamp(timestamp)
    if timestamp < (
        opened["opened_at_utc"] if previous is None else previous["finished_at_utc"]
    ):
        raise ValueError("stage claim chronology differs")
    return authenticate(
        {
            "schema": CLAIM_SCHEMA,
            "authorization_sha256": opened["authorization_sha256"],
            "registry_sha256": registry["sha256"],
            "opening_sha256": opened["sha256"],
            "cell": cell,
            "previous_terminal_sha256": (
                None if previous is None else previous["sha256"]
            ),
            "claimed_at_utc": timestamp,
        }
    )


def command(registry: dict[str, Any], cell: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "plant_backend.rl.v2.study.stage_cli",
        "child",
        "--training-root",
        registry["training_root"],
        "--stage",
        registry["stage"],
        "--order",
        str(cell["order"]),
    ]


def process_failure(
    process: object,
    logs: dict[str, Any],
    expected_command: list[str],
    budget: ExecutionBudget,
) -> str | None:
    record = mapping(process, PROCESS_FIELDS)
    exact("stage child command", record["command"], expected_command)
    exact(
        "stage child timeout",
        integer(record["timeout_seconds"], minimum=1),
        budget.child_timeout_seconds,
    )
    if type(record["started"]) is not bool or set(logs) != {"stdout", "stderr"}:
        raise ValueError("stage process/log fields differ")
    failures = set()
    for name in ("stdout", "stderr"):
        log = mapping(logs[name], FILE_FIELDS)
        size = integer(log["logical_bytes"])
        integer(log["allocated_bytes"])
        if size > budget.log_limit_bytes:
            raise ValueError("stage log exceeds its declared limit")
        if size == budget.log_limit_bytes:
            failures.add("child_log_limit_exceeded")
    if not record["started"]:
        if any(
            record[key] is not None
            for key in PROCESS_FIELDS - {"command", "timeout_seconds", "started"}
        ) or any(logs[name]["logical_bytes"] for name in logs):
            raise ValueError("unstarted stage child retains measurements")
        failures.add("child_spawn_failed")
    else:
        elapsed = integer(record["wall_elapsed_ns"], minimum=1)
        integer(record["child_user_cpu_ns"])
        integer(record["child_system_cpu_ns"])
        code = record["exit_code"]
        if type(code) is not int:
            raise ValueError("stage child was not reaped")
        exact(
            "stage termination signal",
            record["termination_signal"],
            -code if code < 0 else None,
        )
        if elapsed >= budget.child_timeout_seconds * 1_000_000_000:
            failures.add("child_timeout")
        if code < 0:
            failures.add("child_signaled")
        elif code:
            failures.add("child_nonzero_exit")
    return next((code for code in PROCESS_FAILURE_ORDER if code in failures), None)


def inventory(directory: Path) -> dict[str, Any]:
    """Full retained cell inventory, excluding only its self-referencing terminal."""
    from ..training.custody_json import domain_digest

    result = io.inventory(directory)
    rows = [row for row in result["entries"] if row["path"] != "terminal.json"]
    return {
        **result,
        "entries": rows,
        "regular_file_count": sum(row["node_type"] == "regular_file" for row in rows),
        "directory_count": sum(row["node_type"] == "directory" for row in rows),
        "forbidden_count": sum(row["node_type"] == "forbidden" for row in rows),
        "logical_bytes": sum(row["logical_bytes"] or 0 for row in rows),
        "allocated_bytes": sum(row["allocated_bytes"] or 0 for row in rows),
        "tree_sha256": domain_digest(io.INVENTORY_DOMAIN, rows),
    }

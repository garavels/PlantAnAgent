"""Closed, outcome-free schemas and ordered receipt identities."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

from .custody_authorization import EXECUTION, utc_timestamp
from .custody_json import authenticate, canonical_bytes, domain_digest, require_envelope
from .registered_contract import TEMPLATE_DOMAIN, TEMPLATE_SHA256, matrix

PREFIX = "plant-beta-constrained-rl-registered-training-"
PLAN_SCHEMA = PREFIX + "plan-v1"
CLAIM_SCHEMA = PREFIX + "claim-v1"
TERMINAL_SCHEMA = PREFIX + "terminal-v1"
REPORT_SCHEMA = PREFIX + "report-v1"
SYNTHETIC_REPORT_SCHEMA = PREFIX + "synthetic-report-v1"
SOURCE_DOMAIN = PREFIX + "source-v1"
HOST_DOMAIN = PREFIX + "host-v1"
DETERMINISTIC_DOMAIN = PREFIX + "plan-deterministic-evidence-v1"

PLAN_IDENTITY_FIELDS = (
    "protocols",
    "authorization",
    "campaign_root",
    "source",
    "runtime_provenance",
    "host",
    "execution",
    "task_template",
    "cells",
)
PLAN_FIELDS = frozenset(
    {"schema", *PLAN_IDENTITY_FIELDS, "deterministic_evidence_sha256", "sha256"}
)
CLAIM_FIELDS = frozenset(
    {
        "schema",
        "protocols",
        "authorization_sha256",
        "plan_sha256",
        "cell",
        "source_sha256",
        "runtime_provenance_sha256",
        "host_sha256",
        "previous_terminal_sha256",
        "claimed_at_utc",
        "sha256",
    }
)
TERMINAL_FIELDS = frozenset(
    {
        "schema",
        "status",
        "protocols",
        "authorization_sha256",
        "plan_sha256",
        "claim_sha256",
        "cell",
        "process",
        "logs",
        "artifact_inventory",
        "checkpoint",
        "failure",
        "source_before",
        "source_after",
        "runtime_before",
        "runtime_after",
        "host_before",
        "host_after",
        "finished_at_utc",
        "sha256",
    }
)
REPORT_FIELDS = frozenset(
    {
        "schema",
        "protocols",
        "claim_boundary",
        "result_claim",
        "authorization",
        "plan",
        "source",
        "runtime_provenance",
        "host",
        "execution",
        "cells",
        "status_counts",
        "campaign_status",
        "abort",
        "deterministic_evidence_sha256",
        "sha256",
    }
)
REPORT_CELL_FIELDS = frozenset(
    {
        "cell",
        "state",
        "config",
        "claim",
        "terminal",
        "logs",
        "artifact_inventory",
        "checkpoint",
        "failure",
    }
)
PROCESS_FIELDS = frozenset(
    {
        "command",
        "timeout_seconds",
        "started",
        "exit_code",
        "termination_signal",
        "wall_elapsed_ns",
        "child_user_cpu_ns",
        "child_system_cpu_ns",
    }
)
CHECKPOINT_FIELDS = frozenset(
    {
        "path",
        "metadata_sha256",
        "checkpoint_evidence_sha256",
        "runtime_provenance_sha256",
        "environment_contract_sha256",
        "checkpoint_contract_sha256",
        "files",
        "logical_bytes",
        "allocated_bytes",
        "rollout_entry_count",
        "allocator_entry_count",
        "optimizer_steps",
        "registered_evaluation_eligible",
    }
)
FILE_FIELDS = frozenset({"path", "logical_bytes", "allocated_bytes", "sha256"})
ENVELOPE_REFERENCE_FIELDS = frozenset({"path", "raw_file_sha256", "sha256"})
CELL_STATES = (
    "not_started",
    "claimed_interrupted",
    "process_failed",
    "checkpoint_rejected",
    "campaign_integrity_failed",
    "admitted",
)
CLAIM_BOUNDARY = "registered-training-custody-only; no controller-performance evidence"
SUCCESS_CLAIM = (
    "On the authorized host and clean source revision, the frozen 15-run constrained-RL "
    "training matrix completed once and every final checkpoint passed registered production "
    "admission. No development seed, held-out seed, controller comparison, or policy selection was opened."
)

# Insertion order is the frozen precedence, never exception arrival order.
FAILURES = {
    "source_mutation": ("campaign_integrity_failed", "Source identity changed."),
    "runtime_mutation": ("campaign_integrity_failed", "Runtime identity changed."),
    "host_mutation": ("campaign_integrity_failed", "Host identity changed."),
    "authorization_mutation": (
        "campaign_integrity_failed",
        "Authorization identity changed.",
    ),
    "plan_mutation": ("campaign_integrity_failed", "Plan identity changed."),
    "root_custody_lost": (
        "campaign_integrity_failed",
        "Campaign root custody was lost.",
    ),
    "child_log_limit_exceeded": (
        "process_failed",
        "A child log reached its declared byte limit.",
    ),
    "child_timeout": ("process_failed", "The child exceeded its declared timeout."),
    "child_spawn_failed": ("process_failed", "The child process could not be started."),
    "child_signaled": ("process_failed", "The child terminated by signal."),
    "child_nonzero_exit": ("process_failed", "The child returned a nonzero exit code."),
    "checkpoint_missing": ("checkpoint_rejected", "The final checkpoint is missing."),
    "artifact_inventory_invalid": (
        "checkpoint_rejected",
        "The artifact inventory is ineligible.",
    ),
    "checkpoint_malformed": (
        "checkpoint_rejected",
        "The checkpoint cannot be decoded.",
    ),
    "checkpoint_admission_failed": (
        "checkpoint_rejected",
        "Registered checkpoint admission failed.",
    ),
}
STAGES = frozenset({"pre_child", "child", "post_child", "inventory", "admission"})


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def failure(code: str, stage: str) -> dict[str, str]:
    if stage not in STAGES or code not in FAILURES:
        raise ValueError("failure code or stage is not in the closed contract")
    return {"code": code, "stage": stage, "message": FAILURES[code][1]}


def choose_failure(observations: dict[str, str]) -> tuple[str, dict[str, str] | None]:
    if set(observations) - set(FAILURES):
        raise ValueError("unknown campaign failure observation")
    for code, (status, _) in FAILURES.items():
        if code in observations:
            return status, failure(code, observations[code])
    return "admitted", None


def envelope_reference(path: Path, value: dict[str, Any]) -> dict[str, str]:
    return {
        "path": str(path),
        "raw_file_sha256": hashlib.sha256(canonical_bytes(value)).hexdigest(),
        "sha256": value["sha256"],
    }


def make_plan(
    root: Path,
    *,
    authorization: dict[str, Any],
    source: dict[str, Any],
    runtime: dict[str, Any],
    host: dict[str, Any],
) -> dict[str, Any]:
    """Build pre-run identity only; this grants no training authority."""
    identity = {
        "protocols": authorization["protocols"],
        "authorization": envelope_reference(root / "authorization.json", authorization),
        "campaign_root": str(root),
        "source": source,
        "runtime_provenance": runtime,
        "host": host,
        "execution": dict(EXECUTION),
        "task_template": {"domain": TEMPLATE_DOMAIN, "sha256": TEMPLATE_SHA256},
        "cells": matrix(root),
    }
    return authenticate(
        {
            "schema": PLAN_SCHEMA,
            **identity,
            "deterministic_evidence_sha256": domain_digest(
                DETERMINISTIC_DOMAIN, identity
            ),
        }
    )


def make_claim(
    plan: dict[str, Any],
    cell: dict[str, Any],
    previous_terminal: dict[str, Any] | None,
    *,
    timestamp: str,
) -> dict[str, Any]:
    utc_timestamp(timestamp)
    require_envelope(canonical_bytes(plan), schema=PLAN_SCHEMA, fields=PLAN_FIELDS)
    if previous_terminal is not None:
        require_envelope(
            canonical_bytes(previous_terminal),
            schema=TERMINAL_SCHEMA,
            fields=TERMINAL_FIELDS,
        )
    order = cell["order"]
    if (
        type(order) is not int
        or not 1 <= order <= 15
        or plan["cells"][order - 1] != cell
    ):
        raise ValueError("claim does not match the next declared cell")
    if order == 1:
        if previous_terminal is not None:
            raise ValueError("first claim cannot have a preceding terminal")
    elif (
        previous_terminal is None
        or previous_terminal["cell"] != plan["cells"][order - 2]
        or previous_terminal["plan_sha256"] != plan["sha256"]
        or previous_terminal["status"]
        not in {"admitted", "process_failed", "checkpoint_rejected"}
    ):
        raise ValueError("claim requires the preceding non-aborting terminal")
    return authenticate(
        {
            "schema": CLAIM_SCHEMA,
            "protocols": plan["protocols"],
            "authorization_sha256": plan["authorization"]["sha256"],
            "plan_sha256": plan["sha256"],
            "cell": cell,
            "source_sha256": domain_digest(SOURCE_DOMAIN, plan["source"]),
            "runtime_provenance_sha256": plan["runtime_provenance"]["sha256"],
            "host_sha256": domain_digest(HOST_DOMAIN, plan["host"]),
            "previous_terminal_sha256": (
                None if previous_terminal is None else previous_terminal["sha256"]
            ),
            "claimed_at_utc": timestamp,
        }
    )

"""Detached reconstruction of registered custody, with a separate synthetic door."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any

from . import capacity_benchmark as capacity
from . import custody_authorization as authority
from . import custody_documents as documents
from . import custody_io as io
from .custody_checkpoint import CheckpointAdmissionFailure, admit_checkpoint
from .custody_json import canonical_bytes, require_envelope
from .custody_runtime import capture_identity
from .custody_state import (
    CampaignContext,
    CheckpointAdmitter,
    IdentitySnapshot,
    cell_config,
    command_for,
    identity_failures,
    _stamp,
)
from .custody_validation import checkpoint_descriptor, exact, mapping, process_failures
from .registered_contract import matrix
from .runtime_provenance import BetaTrainingRuntimeProvenanceV1


@dataclass(frozen=True)
class _CellEvidence:
    cell: dict[str, Any]
    config: dict[str, Any]
    claim: dict[str, Any] | None
    terminal: dict[str, Any] | None
    logs: dict[str, Any] | None
    inventory: dict[str, Any] | None
    failures: dict[str, str]


def _envelope(path: Path, schema: str, fields: frozenset[str]) -> dict[str, Any]:
    return require_envelope(io.read_regular(path), schema=schema, fields=fields)


def _read_report(path: Path, *, synthetic: bool) -> dict[str, Any]:
    expected_name = "synthetic-report.json" if synthetic else "report.json"
    schema = documents.SYNTHETIC_REPORT_SCHEMA if synthetic else documents.REPORT_SCHEMA
    if path.name != expected_name or not path.is_absolute() or path != path.resolve():
        raise ValueError("custody report filename or canonical path differs")
    matrix(path.parent)
    # Schema rejection precedes all source/host/runtime/checkpoint recapture.
    return _envelope(path, schema, documents.REPORT_FIELDS)


def _read_context(
    root: Path, authorization: dict[str, Any], identity: IdentitySnapshot
) -> CampaignContext:
    exact(
        "authorization copy",
        io.read_regular(root / "authorization.json").decode("utf-8"),
        canonical_bytes(authorization).decode("utf-8"),
    )
    plan = _envelope(root / "plan.json", documents.PLAN_SCHEMA, documents.PLAN_FIELDS)
    expected = documents.make_plan(
        root,
        authorization=authorization,
        source=identity.source,
        runtime=identity.runtime,
        host=identity.host,
    )
    exact("pre-run plan", plan, expected)
    return CampaignContext(root, authorization, plan, identity)


def _snapshot(source: object, runtime: object, host: object) -> IdentitySnapshot:
    source_map = mapping(
        source,
        frozenset(
            {
                "authorization_commit",
                "approved_implementation_commit",
                "approved_implementation_tree",
                "uv_lock_sha256",
                "authorization_parent",
                "changed_paths",
            }
        ),
    )
    rt = BetaTrainingRuntimeProvenanceV1.from_dict(runtime)
    host_map = capacity._verify_host(host)
    if (
        source_map["authorization_commit"] != rt.git_head
        or source_map["uv_lock_sha256"] != rt.uv_lock_sha256
    ):
        raise ValueError("source snapshot differs from runtime provenance")
    for key in (
        "authorization_commit",
        "approved_implementation_commit",
        "approved_implementation_tree",
        "authorization_parent",
    ):
        value = source_map[key]
        if (
            not isinstance(value, str)
            or len(value) != 40
            or set(value) - set("0123456789abcdef")
        ):
            raise ValueError("source snapshot Git identity is malformed")
    paths = source_map["changed_paths"]
    if (
        not isinstance(paths, list)
        or any(
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            for path in paths
        )
        or paths != sorted(set(paths))
    ):
        raise ValueError("source changed paths are malformed")
    return IdentitySnapshot(source_map, rt.to_dict(), host_map)


def _read_cells(
    context: CampaignContext, *, stop_before: int | None = None
) -> list[_CellEvidence]:
    cells: list[_CellEvidence] = []
    previous: dict[str, Any] | None = None
    gap = False
    for cell in context.plan["cells"]:
        if stop_before is not None and cell["order"] >= stop_before:
            break
        config = cell_config(cell)
        config_record = io.file_record(Path(cell["config_path"]))
        exact(
            "canonical cell config",
            io.read_regular(Path(cell["config_path"])).decode("utf-8"),
            canonical_bytes(config.to_dict()).decode("utf-8"),
        )
        exact("cell config digest", config_record["sha256"], cell["config_sha256"])
        claim_path = Path(cell["claim_path"])
        terminal_paths = [
            Path(cell[key])
            for key in ("completed_receipt_path", "failed_receipt_path")
            if os.path.lexists(cell[key])
        ]
        if not os.path.lexists(claim_path):
            if terminal_paths or any(
                os.path.lexists(cell[key])
                for key in ("stdout_path", "stderr_path", "output_path")
            ):
                raise ValueError(
                    "unclaimed cell has process, receipt, or artifact evidence"
                )
            gap = True
            cells.append(_CellEvidence(cell, config_record, None, None, None, None, {}))
            continue
        if gap:
            raise ValueError("claim appears after an unclaimed cell")
        claim = _envelope(claim_path, documents.CLAIM_SCHEMA, documents.CLAIM_FIELDS)
        expected_claim = documents.make_claim(
            context.plan,
            cell,
            previous,
            timestamp=authority.utc_timestamp(claim["claimed_at_utc"]),
        )
        exact("sequential claim", claim, expected_claim)
        if not terminal_paths:
            raise ValueError(
                "claimed_interrupted: the consumed claim has no terminal receipt"
            )
        if len(terminal_paths) != 1:
            raise ValueError("a claimed cell must have exactly one terminal receipt")
        terminal = _envelope(
            terminal_paths[0], documents.TERMINAL_SCHEMA, documents.TERMINAL_FIELDS
        )
        for key, expected in (
            ("protocols", context.plan["protocols"]),
            ("authorization_sha256", context.authorization["sha256"]),
            ("plan_sha256", context.plan["sha256"]),
            ("claim_sha256", claim["sha256"]),
            ("cell", cell),
        ):
            exact(f"terminal {key}", terminal[key], expected)
        authority.utc_timestamp(terminal["finished_at_utc"])
        before = _snapshot(
            terminal["source_before"],
            terminal["runtime_before"],
            terminal["host_before"],
        )
        after = _snapshot(
            terminal["source_after"], terminal["runtime_after"], terminal["host_after"]
        )
        if identity_failures(context.identity, before, "pre_child"):
            raise ValueError("claimed cell began with an unadmitted identity")
        failures = identity_failures(context.identity, after, "post_child")
        logs = {
            name: io.file_record(Path(cell[f"{name}_path"]))
            for name in ("stdout", "stderr")
        }
        exact("sealed log descriptors", terminal["logs"], logs)
        failures.update(process_failures(terminal["process"], logs, command_for(cell)))
        observed_inventory = io.inventory(Path(cell["output_path"]))
        exact(
            "retained artifact inventory",
            terminal["artifact_inventory"],
            observed_inventory,
        )
        if terminal["failure"] is not None:
            reported_failure = mapping(
                terminal["failure"], frozenset({"code", "stage", "message"})
            )
            exact(
                "closed failure",
                reported_failure,
                documents.failure(reported_failure["code"], reported_failure["stage"]),
            )
            if reported_failure["code"] == "root_custody_lost":
                # A handled interruption/root replacement is a terminal event,
                # not recoverable from later byte hashes. It can only revoke a
                # campaign claim and forbid all subsequent attempts.
                if reported_failure["stage"] not in {"child", "post_child"}:
                    raise ValueError("root custody failure stage differs")
                failures["root_custody_lost"] = reported_failure["stage"]
        expected_filename = (
            "completed.json" if terminal["status"] == "admitted" else "failed.json"
        )
        if terminal_paths[0].name != expected_filename:
            raise ValueError("terminal filename differs from status")
        previous = terminal
        cells.append(
            _CellEvidence(
                cell, config_record, claim, terminal, logs, observed_inventory, failures
            )
        )
    return cells


def _admit_production_cells(
    cells: list[_CellEvidence], context: CampaignContext
) -> dict[int, tuple[dict[str, Any] | None, str | None]]:
    results: dict[int, tuple[dict[str, Any] | None, str | None]] = {}
    runtime = BetaTrainingRuntimeProvenanceV1.from_dict(context.identity.runtime)
    for item in cells:
        if item.terminal is None or item.failures:
            continue
        try:
            admitted = admit_checkpoint(cell_config(item.cell), runtime)
        except CheckpointAdmissionFailure as reason:
            results[item.cell["order"]] = (None, reason.code)
        except Exception:
            results[item.cell["order"]] = (None, "checkpoint_admission_failed")
        else:
            results[item.cell["order"]] = (admitted, None)
    return results


def _admit_synthetic_cells(
    cells: list[_CellEvidence], context: CampaignContext, checkpoint: CheckpointAdmitter
) -> dict[int, tuple[dict[str, Any] | None, str | None]]:
    results: dict[int, tuple[dict[str, Any] | None, str | None]] = {}
    runtime = BetaTrainingRuntimeProvenanceV1.from_dict(context.identity.runtime)
    for item in cells:
        if item.terminal is None or item.failures:
            continue
        try:
            admitted = checkpoint(cell_config(item.cell), runtime)
        except CheckpointAdmissionFailure as reason:
            results[item.cell["order"]] = (None, reason.code)
        except Exception:
            results[item.cell["order"]] = (None, "checkpoint_admission_failed")
        else:
            results[item.cell["order"]] = (admitted, None)
    return results


def _reconstruct_records(
    cells: list[_CellEvidence],
    admissions: dict[int, tuple[dict[str, Any] | None, str | None]],
    context: CampaignContext,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in cells:
        cell = item.cell
        if item.terminal is None:
            records.append(
                {
                    "cell": cell,
                    "state": "not_started",
                    "config": item.config,
                    "claim": None,
                    "terminal": None,
                    "logs": None,
                    "artifact_inventory": None,
                    "checkpoint": None,
                    "failure": None,
                }
            )
            continue
        failures = dict(item.failures)
        admitted = None
        if not failures:
            admitted, rejection = admissions[cell["order"]]
            if rejection:
                failures[rejection] = (
                    "inventory"
                    if rejection == "artifact_inventory_invalid"
                    else "admission"
                )
            else:
                checkpoint_descriptor(admitted, context.identity.runtime["sha256"])
        state, failure = documents.choose_failure(failures)
        exact("terminal status", item.terminal["status"], state)
        exact("terminal failure", item.terminal["failure"], failure)
        exact(
            "terminal checkpoint",
            item.terminal["checkpoint"],
            admitted if state == "admitted" else None,
        )
        terminal_path = Path(
            cell["completed_receipt_path"]
            if state == "admitted"
            else cell["failed_receipt_path"]
        )
        assert item.claim is not None
        records.append(
            {
                "cell": cell,
                "state": state,
                "config": item.config,
                "claim": documents.envelope_reference(
                    Path(cell["claim_path"]), item.claim
                ),
                "terminal": documents.envelope_reference(terminal_path, item.terminal),
                "logs": item.logs,
                "artifact_inventory": item.inventory,
                "checkpoint": admitted if state == "admitted" else None,
                "failure": failure,
            }
        )
    return records


def _verify_layout(
    context: CampaignContext, records: list[dict[str, Any]], report_name: str
) -> None:
    root = context.root
    expected_root = {
        "authorization.json",
        "plan.json",
        "configs",
        "logs",
        "attempts",
        "artifacts",
        report_name,
    }
    if {path.name for path in root.iterdir()} != expected_root:
        raise ValueError("campaign root has missing or undeclared entries")
    for name in ("configs", "logs", "attempts", "artifacts"):
        io.require_directory(root / name)
    expected: dict[str, set[str]] = {
        name: set() for name in ("configs", "logs", "attempts", "artifacts")
    }
    for record in records:
        cell = record["cell"]
        expected["configs"].add(Path(cell["config_path"]).name)
        attempt = Path(cell["claim_path"]).parent
        io.require_directory(attempt)
        expected["attempts"].add(attempt.name)
        attempt_names = {path.name for path in attempt.iterdir()}
        if record["state"] == "not_started":
            if attempt_names:
                raise ValueError("unstarted attempt directory contains evidence")
            continue
        if attempt_names != {"claimed.json", Path(record["terminal"]["path"]).name}:
            raise ValueError("attempt directory contains undeclared evidence")
        expected["logs"].update(
            Path(cell[f"{name}_path"]).name for name in ("stdout", "stderr")
        )
        if record["artifact_inventory"]["exists"]:
            expected["artifacts"].add(Path(cell["output_path"]).name)
    for directory, names in expected.items():
        if {path.name for path in (root / directory).iterdir()} != names:
            raise ValueError("campaign directory contents differ from the exact matrix")


def _verify_report_record(
    report: dict[str, Any],
    context: CampaignContext,
    records: list[dict[str, Any]],
    *,
    synthetic: bool,
) -> None:
    for key, expected in (
        ("protocols", context.plan["protocols"]),
        ("authorization", context.plan["authorization"]),
        (
            "plan",
            documents.envelope_reference(context.root / "plan.json", context.plan),
        ),
        ("source", context.identity.source),
        ("runtime_provenance", context.identity.runtime),
        ("host", context.identity.host),
        ("execution", context.plan["execution"]),
        ("cells", records),
        (
            "deterministic_evidence_sha256",
            context.plan["deterministic_evidence_sha256"],
        ),
    ):
        exact(f"report {key}", report[key], expected)
    counts = {
        state: sum(cell["state"] == state for cell in records)
        for state in documents.CELL_STATES
    }
    exact("status counts", report["status_counts"], counts)
    abort = report["abort"]
    if abort is None:
        if counts["not_started"] or counts["campaign_integrity_failed"]:
            raise ValueError("incomplete or interrupted campaign has no abort")
    else:
        mapping(abort, frozenset({"code", "stage", "message"}))
        if abort["code"] == "storage_budget_unavailable":
            exact(
                "storage abort",
                abort,
                {
                    "code": "storage_budget_unavailable",
                    "stage": "pre_child",
                    "message": "The remaining campaign storage budget is unavailable.",
                },
            )
            if not counts["not_started"] or counts["campaign_integrity_failed"]:
                raise ValueError("storage abort does not explain campaign termination")
        else:
            exact(
                "campaign abort",
                abort,
                documents.failure(abort["code"], abort["stage"]),
            )
            if documents.FAILURES[abort["code"]][0] != "campaign_integrity_failed":
                raise ValueError("cell-local failure cannot abort a campaign")
            stopped = [
                cell for cell in records if cell["state"] == "campaign_integrity_failed"
            ]
            if stopped:
                exact("terminal campaign abort", abort, stopped[-1]["failure"])
            elif not counts["not_started"] or abort["stage"] != "pre_child":
                raise ValueError("campaign abort has no interrupted boundary")
    status = (
        "aborted" if abort else ("completed" if counts["admitted"] == 15 else "failed")
    )
    exact("campaign status", report["campaign_status"], status)
    expected_claim = (
        None if synthetic or status != "completed" else documents.SUCCESS_CLAIM
    )
    exact("result claim", report["result_claim"], expected_claim)
    expected_boundary = (
        "synthetic-custody-fixture; no registered training or production claim"
        if synthetic
        else documents.CLAIM_BOUNDARY
    )
    exact("claim boundary", report["claim_boundary"], expected_boundary)
    _verify_layout(
        context, records, "synthetic-report.json" if synthetic else "report.json"
    )


def _recheck_evidence(
    target: Path,
    context: CampaignContext,
    report: dict[str, Any],
    records: list[dict[str, Any]],
    admissions: dict[int, tuple[dict[str, Any] | None, str | None]],
    root_stamp: tuple[int, ...],
    *,
    synthetic: bool,
) -> None:
    """Reopen the complete evidence graph after checkpoint admission finishes."""
    exact("root after admission", _stamp(context.root), root_stamp)
    exact("report after admission", _read_report(target, synthetic=synthetic), report)
    _read_context(context.root, context.authorization, context.identity)
    exact(
        "evidence after admission",
        _reconstruct_records(_read_cells(context), admissions, context),
        records,
    )
    _verify_layout(context, records, target.name)


def verify_report(path: Path | str) -> dict[str, Any]:
    """Production verifier: no injectable runtime, host, process, or policy hooks."""
    target = Path(path)
    report = _read_report(target, synthetic=False)
    host = capacity._verify_host(capacity._host_identity())
    authorization, source = authority.admit_authorization(target.parent, host)
    identity = capture_identity(authorization)
    exact("current authorization source", identity.source, source)
    exact("current authorized host", identity.host, host)
    if not identity.runtime["worktree_clean"]:
        raise ValueError("production verification requires clean runtime provenance")
    context = _read_context(target.parent, authorization, identity)
    root_stamp = _stamp(context.root)
    cells = _read_cells(context)
    admissions = _admit_production_cells(cells, context)
    records = _reconstruct_records(cells, admissions, context)
    _verify_report_record(report, context, records, synthetic=False)
    after = capture_identity(authorization)
    if identity_failures(identity, after, "post_child"):
        raise ValueError("verification source/runtime/host changed")
    _recheck_evidence(
        target, context, report, records, admissions, root_stamp, synthetic=False
    )
    return {
        "verified": True,
        "schema": documents.REPORT_SCHEMA,
        "report_sha256": report["sha256"],
        "campaign_status": report["campaign_status"],
        "status_counts": report["status_counts"],
        "result_claim": report["result_claim"],
    }


def verify_synthetic_report(
    path: Path | str, *, checkpoint: CheckpointAdmitter
) -> dict[str, Any]:
    """Synthetic-only admission is unavailable until filename and schema match."""
    target = Path(path)
    report = _read_report(target, synthetic=True)
    authorization = _envelope(
        target.parent / "authorization.json",
        authority.AUTHORIZATION_SCHEMA,
        authority.AUTHORIZATION_FIELDS,
    )
    identity = _snapshot(report["source"], report["runtime_provenance"], report["host"])
    context = _read_context(target.parent, authorization, identity)
    root_stamp = _stamp(context.root)
    cells = _read_cells(context)
    admissions = _admit_synthetic_cells(cells, context, checkpoint)
    records = _reconstruct_records(cells, admissions, context)
    _verify_report_record(report, context, records, synthetic=True)
    _recheck_evidence(
        target, context, report, records, admissions, root_stamp, synthetic=True
    )
    return {
        "verified": True,
        "schema": documents.SYNTHETIC_REPORT_SCHEMA,
        "report_sha256": report["sha256"],
        "result_claim": None,
    }


def inspect_root(root: Path | str) -> dict[str, Any]:
    """Read-only forensic classification; it grants no retry or admission."""
    target = Path(root)
    expected_cells = matrix(target)
    io.require_directory(target)
    plan = _envelope(target / "plan.json", documents.PLAN_SCHEMA, documents.PLAN_FIELDS)
    exact("forensic plan matrix", plan["cells"], expected_cells)
    records: list[dict[str, Any]] = []
    for cell in expected_cells:
        claim_path = Path(cell["claim_path"])
        terminals = [
            Path(cell[key])
            for key in ("completed_receipt_path", "failed_receipt_path")
            if os.path.lexists(cell[key])
        ]
        claim_valid = False
        terminal_valid = False
        state = "not_started"
        if os.path.lexists(claim_path):
            try:
                claim = _envelope(
                    claim_path, documents.CLAIM_SCHEMA, documents.CLAIM_FIELDS
                )
                exact("forensic claim cell", claim["cell"], cell)
                exact("forensic claim plan", claim["plan_sha256"], plan["sha256"])
                claim_valid = True
                state = "claimed_interrupted"
                if terminals:
                    if len(terminals) != 1:
                        raise ValueError("multiple terminal receipts")
                    terminal = _envelope(
                        terminals[0],
                        documents.TERMINAL_SCHEMA,
                        documents.TERMINAL_FIELDS,
                    )
                    exact(
                        "forensic terminal claim",
                        terminal["claim_sha256"],
                        claim["sha256"],
                    )
                    exact("forensic terminal cell", terminal["cell"], cell)
                    if terminal["status"] not in {
                        "admitted",
                        "process_failed",
                        "checkpoint_rejected",
                        "campaign_integrity_failed",
                    }:
                        raise ValueError("invalid terminal status")
                    terminal_valid = True
                    state = terminal["status"]
            except (ValueError, OSError, TypeError):
                state = "invalid_evidence"
        elif terminals:
            state = "invalid_evidence"
        records.append(
            {
                "order": cell["order"],
                "state": state,
                "claim_valid": claim_valid,
                "terminal_valid": terminal_valid,
            }
        )
    return {
        "schema": documents.PREFIX + "forensic-inspection-v1",
        "root": str(target),
        "plan_sha256": plan["sha256"],
        "cells": records,
        "verified": False,
        "result_claim": None,
        "retry_permitted": False,
    }

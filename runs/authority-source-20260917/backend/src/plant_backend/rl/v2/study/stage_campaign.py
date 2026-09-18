"""Fixed sequential stage parent; source/authority gates precede every attempt."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

from ..training import custody_io as io
from ..training.custody_json import authenticate
from ..training.custody_launch import LAUNCH_FD_ENV
from ..training.custody_process import capture_process
from ..training.custody_validation import exact
from . import archive, stage_documents as documents, stage_state, stage_lock
from .stage_authority import StageAdmission, admit_stage
from .stage_launch import permission
from .stage_verifier import build_report, verdict
from .training_evidence import _tree_stamps
from .execution_budget import execution_contract
from .arm_evidence import file_stamp


def _budget_available(context: StageAdmission, retained_bytes: int) -> bool:
    fs = os.statvfs(context.root)
    return (
        fs.f_bavail * fs.f_frsize + retained_bytes
        >= context.budget.opening_free_bytes(context.registry["stage"])
    )


def _outside_current(
    stamps: tuple[tuple[str, tuple[int, ...]], ...], cell: dict[str, Any]
) -> dict[str, tuple[int, ...]]:
    prefix = "cells/" + Path(cell["directory"]).name
    return {
        name: stamp
        for name, stamp in stamps
        if name != prefix and not name.startswith(prefix + "/")
    }


def _run_child(
    context: StageAdmission, cell: dict[str, Any], claim: dict[str, Any]
) -> tuple[dict[str, Any], set[str]]:
    return capture_process(
        documents.command(context.registry, cell),
        stdout=Path(cell["stdout_path"]),
        stderr=Path(cell["stderr_path"]),
        timeout_seconds=context.budget.child_timeout_seconds,
        log_limit_bytes=context.budget.log_limit_bytes,
        launch_permission=permission(context.registry, claim, context.authorization),
    )


def _collect(
    context: StageAdmission, lock: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    io.mkdir_exclusive(context.root)
    documents.publish(
        Path(context.registry["authorization_path"]), context.authorization
    )
    documents.publish(Path(context.registry["registry_path"]), context.registry)
    io.mkdir_exclusive(context.root / "cells")
    for cell in context.registry["cells"]:
        io.mkdir_exclusive(Path(cell["directory"]))
    opened = documents.opening(
        context.authorization,
        context.registry,
        parent_pid=os.getpid(),
        lock=lock,
        timestamp=documents.now(),
    )
    documents.publish(Path(context.registry["opening_claim_path"]), opened)
    snapshot = _tree_stamps(context.root)
    previous = None
    abort = None
    retained_bytes = io.inventory(context.root)["allocated_bytes"]
    for cell in context.registry["cells"]:
        try:
            context.recheck()
            stage_lock.verify_lock_identity(lock)
            stage_state.check_layout(context, closed=False)
            if _tree_stamps(context.root) != snapshot:
                raise ValueError(
                    "closed stage evidence or unclaimed destinations changed"
                )
        except (ValueError, OSError, RuntimeError):
            abort = {"code": "stage_integrity_lost", "before_order": cell["order"]}
            break
        try:
            available = _budget_available(context, retained_bytes)
        except KeyboardInterrupt:
            abort = {"code": "operator_interrupted", "before_order": cell["order"]}
            break
        if not available:
            abort = {
                "code": "storage_budget_unavailable",
                "before_order": cell["order"],
            }
            break
        claim = documents.claim(
            context.registry, opened, cell, previous, timestamp=documents.now()
        )
        documents.publish(Path(cell["claim_path"]), claim, cell=True)
        claim_stamp = file_stamp(Path(cell["claim_path"]))
        process = None
        integrity = False
        if context.training_failure(cell) is None:
            process, failures = _run_child(context, cell, claim)
            integrity = "root_custody_lost" in failures
        try:
            context.recheck()
            stage_lock.verify_lock_identity(lock)
        except (ValueError, OSError, RuntimeError):
            integrity = True
        after = _tree_stamps(context.root)
        before_root = dict(snapshot)["."]
        after_root = dict(after)["."]
        if before_root[:4] != after_root[:4] or before_root[-2:] != after_root[-2:]:
            raise ValueError(
                "consumed stage root was replaced; no new publication is permitted"
            )
        if _outside_current(snapshot, cell) != _outside_current(after, cell):
            integrity = True
        exact(
            "closed child claim",
            documents.read(
                Path(cell["claim_path"]),
                documents.CLAIM_SCHEMA,
                documents.CLAIM_FIELDS,
                cell=True,
            ),
            claim,
        )
        if file_stamp(Path(cell["claim_path"])) != claim_stamp:
            raise ValueError("consumed child claim was replaced")
        rebuilt, _ = stage_state.classify_cell(
            context, cell, claim, process, integrity_failure=integrity
        )
        terminal = authenticate(
            {
                "schema": documents.TERMINAL_SCHEMA,
                "claim_sha256": claim["sha256"],
                "cell": cell,
                **rebuilt,
                "finished_at_utc": documents.now(),
            }
        )
        documents.publish(Path(cell["terminal_path"]), terminal, cell=True)
        previous = terminal
        retained_bytes += (
            rebuilt["inventory"]["allocated_bytes"]
            + Path(cell["terminal_path"]).stat().st_blocks * 512
        )
        snapshot = _tree_stamps(context.root)
        if integrity:
            abort = {"code": "stage_integrity_lost", "before_order": cell["order"] + 1}
            break
    stage_state.check_layout(context, closed=False)
    records, inputs = stage_state.read_cells(context, opened)
    report = build_report(context, opened, records, inputs, abort)
    record = archive.publish_archive(
        Path(context.registry["report_path"]),
        report,
        stored_limit=context.budget.report_stored_bytes,
        uncompressed_limit=context.budget.report_decoded_bytes,
    )
    return report, record


def _detached(
    context: StageAdmission, report: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    registry = context.registry
    command = [
        sys.executable,
        "-m",
        "plant_backend.rl.v2.study.stage_cli",
        "verify",
        "--training-root",
        registry["training_root"],
        "--stage",
        registry["stage"],
        "--publish",
    ]
    process, failures = capture_process(
        command,
        stdout=Path(registry["verification_stdout_path"]),
        stderr=Path(registry["verification_stderr_path"]),
        timeout_seconds=execution_contract()["detached_verifier_timeout_seconds"],
        log_limit_bytes=context.budget.log_limit_bytes,
    )
    receipt = authenticate(
        {
            "schema": documents.VERDICT_SCHEMA + "-process",
            "report_archive": record,
            "process": process,
            "failure_codes": sorted(failures),
            "logs": {
                name: io.file_record(Path(registry["verification_" + name + "_path"]))
                for name in ("stdout", "stderr")
            },
        }
    )
    documents.publish(Path(registry["verification_process_path"]), receipt)
    if failures:
        raise ValueError(
            "detached stage verification failed; all evidence remains consumed"
        )
    expected = verdict(report, record)
    actual = documents.read(
        Path(registry["verification_path"]),
        documents.VERDICT_SCHEMA,
        frozenset(expected),
    )
    exact("detached stage verdict", actual, expected)
    context.recheck()
    return actual


def run_stage(
    training_root: Path | str, stage: str, *, confirm_operator_assertions: bool = False
) -> dict[str, Any]:
    if confirm_operator_assertions is not True or LAUNCH_FD_ENV in os.environ:
        raise ValueError(
            "stage parent requires confirmed operator assertions and no inherited child channel"
        )
    context = admit_stage(training_root, stage)
    for key in (
        "verification_stdout_path",
        "verification_stderr_path",
        "verification_process_path",
    ):
        if os.path.lexists(context.registry[key]):
            raise ValueError(
                "stage detached-verification destination was already consumed"
            )
    from ..training.custody_authorization import repository_root

    io.require_new_root(
        context.root, repository_root(), context.budget.opening_free_bytes(stage)
    )
    with stage_lock.acquire() as lock:
        context.recheck()
        report, record = _collect(context, lock)
        # Always attempt a fresh detached reconstruction after report publication,
        # even when a local identity recapture has already failed.
        return _detached(context, report, record)

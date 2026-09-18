"""Durable stage layout, sequential claims and independently reconstructed cells."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..training import custody_io as io
from ..training.custody_authorization import utc_timestamp
from ..training.custody_json import canonical_bytes
from ..training.custody_validation import exact
from . import stage_documents as documents, stage_arm
from .stage_authority import StageAdmission
from .campaign_analysis import CellInput
from .registry import cells_for_seeds
from .archive import _read_stored
from .execution_budget import ROOT_CONTROL_LIMIT_BYTES


def opened_context(context: StageAdmission) -> dict[str, Any]:
    registry = context.registry
    exact(
        "stage registry copy",
        _read_stored(
            Path(registry["registry_path"]), ROOT_CONTROL_LIMIT_BYTES
        ).decode(),
        canonical_bytes(registry).decode(),
    )
    exact(
        "stage authorization copy",
        documents.read(
            Path(registry["authorization_path"]),
            documents.AUTH_SCHEMA,
            documents.AUTH_FIELDS,
        ),
        context.authorization,
    )
    opened = documents.read(
        Path(registry["opening_claim_path"]),
        documents.OPEN_SCHEMA,
        documents.OPEN_FIELDS,
    )
    exact(
        "stage opening",
        opened,
        documents.opening(
            context.authorization,
            registry,
            parent_pid=opened["parent_pid"],
            lock=opened["lock"],
            timestamp=opened["opened_at_utc"],
        ),
    )
    if opened["opened_at_utc"] < context.authorization["authorized_at_utc"]:
        raise ValueError("stage opened before its authorization")
    from .stage_lock import verify_lock_identity

    verify_lock_identity(opened["lock"])
    return opened


def check_layout(context: StageAdmission, *, closed: bool) -> None:
    io.require_directory(context.root)
    expected = {"registry.json", "authorization.json", "opened.json", "cells"}
    if closed:
        expected.add("report.json.gz")
        if os.path.lexists(context.registry["verification_path"]):
            expected.add("verification.json")
    if {path.name for path in context.root.iterdir()} != expected:
        raise ValueError("stage root has missing or undeclared entries")
    parent = context.root / "cells"
    io.require_directory(parent)
    if {path.name for path in parent.iterdir()} != {
        Path(cell["directory"]).name for cell in context.registry["cells"]
    }:
        raise ValueError("stage lacks the complete predeclared cell directories")
    for cell in context.registry["cells"]:
        io.require_directory(Path(cell["directory"]))


def classify_cell(
    context: StageAdmission,
    cell: dict[str, Any],
    claim: dict[str, Any],
    process: object,
    *,
    integrity_failure: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Reconstruct process/artifact classification; failures never disappear."""
    training_failure = context.training_failure(cell)
    failure: str | None
    logs = None
    admitted = None
    if training_failure is not None:
        if process is not None or {
            path.name for path in Path(cell["directory"]).iterdir()
        } - {"claimed.json", "terminal.json"}:
            raise ValueError("unadmitted training slot contains process evidence")
        state, failure = "training_not_admitted", training_failure
    else:
        logs = {
            name: io.file_record(Path(cell[name + "_path"]))
            for name in ("stdout", "stderr")
        }
        failure = documents.process_failure(
            process, logs, documents.command(context.registry, cell), context.budget
        )
        if integrity_failure:
            state, failure = "evidence_rejected", "stage_integrity_lost"
        elif failure is not None:
            state = "process_failed"
        else:
            try:
                names = {path.name for path in Path(cell["directory"]).iterdir()} - {
                    "terminal.json"
                }
                if names != {
                    "claimed.json",
                    "stdout.log",
                    "stderr.log",
                    "arm.json.gz",
                    "latency.json.gz",
                }:
                    raise ValueError("completed child output layout differs")
                admitted = stage_arm.verify_cell(context, cell, claim)
            except (ValueError, TypeError, KeyError, OSError, RuntimeError):
                state, failure = "evidence_rejected", "arm_evidence_rejected"
            else:
                state = "verified"
    return {
        "state": state,
        "failure": failure,
        "process": process,
        "logs": logs,
        "inventory": documents.inventory(Path(cell["directory"])),
        "arm_archive": None if admitted is None else admitted["arm_archive"],
        "timing_archive": None if admitted is None else admitted["timing_archive"],
    }, admitted


def read_cells(
    context: StageAdmission, opened: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[CellInput]]:
    previous = None
    gap = False
    stopped = False
    records, inputs = [], []
    identities = cells_for_seeds(context.registry["episode_seeds"])
    for cell, identity in zip(context.registry["cells"], identities, strict=True):
        directory = Path(cell["directory"])
        claim_path = Path(cell["claim_path"])
        if not os.path.lexists(claim_path):
            if list(directory.iterdir()):
                raise ValueError("unclaimed stage cell contains evidence")
            gap = True
            # Unattempted failed-training slots retain their known reason even
            # when a campaign abort occurs before their sequential claim.
            training_failure = context.training_failure(cell)
            failure = "not_started" if training_failure is None else training_failure
            state = (
                "not_started" if training_failure is None else "training_not_admitted"
            )
            records.append(
                {
                    "cell": cell,
                    "claim": None,
                    "terminal": None,
                    "state": state,
                    "failure": failure,
                }
            )
            inputs.append(CellInput(identity, state, failure, None, None))
            continue
        if gap or stopped:
            raise ValueError(
                "stage attempt skips a missing or terminal integrity boundary"
            )
        claim = documents.read(
            claim_path, documents.CLAIM_SCHEMA, documents.CLAIM_FIELDS, cell=True
        )
        exact(
            "sequential stage claim",
            claim,
            documents.claim(
                context.registry,
                opened,
                cell,
                previous,
                timestamp=claim["claimed_at_utc"],
            ),
        )
        terminal_path = Path(cell["terminal_path"])
        if not os.path.lexists(terminal_path):
            raise ValueError("claimed_interrupted: stage attempt has no terminal")
        terminal = documents.read(
            terminal_path,
            documents.TERMINAL_SCHEMA,
            documents.TERMINAL_FIELDS,
            cell=True,
        )
        exact("stage terminal claim", terminal["claim_sha256"], claim["sha256"])
        exact("stage terminal cell", terminal["cell"], cell)
        utc_timestamp(terminal["finished_at_utc"])
        if terminal["finished_at_utc"] < claim["claimed_at_utc"]:
            raise ValueError("stage terminal predates its claim")
        stopped = terminal["failure"] == "stage_integrity_lost"
        rebuilt, admitted = classify_cell(
            context, cell, claim, terminal["process"], integrity_failure=stopped
        )
        for key, value in rebuilt.items():
            exact("stage terminal " + key, terminal[key], value)
        records.append(
            {
                "cell": cell,
                "claim": documents.reference(claim_path, claim, cell=True),
                "terminal": documents.reference(terminal_path, terminal, cell=True),
                **rebuilt,
            }
        )
        inputs.append(
            CellInput(
                identity,
                rebuilt["state"],
                rebuilt["failure"],
                None if admitted is None else admitted["physical"],
                None if admitted is None else admitted["latency"],
            )
        )
        previous = terminal
    return records, inputs

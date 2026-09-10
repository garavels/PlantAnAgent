"""Sequential attempt state machine; collection alone cannot issue a report claim."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
from typing import Any

from . import custody_documents as documents
from . import custody_io as io
from .config import ControllerNeutralTrainingConfigV1
from .custody_checkpoint import CheckpointAdmissionFailure
from .custody_json import authenticate, canonical_bytes, require_envelope
from .custody_validation import checkpoint_descriptor, process_failures
from .registered_contract import MINIMUM_FREE_BYTES, registered_config
from .runtime_provenance import BetaTrainingRuntimeProvenanceV1


class CampaignClosedError(RuntimeError):
    """The root is consumed, including when no complete report can be published."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class IdentitySnapshot:
    source: dict[str, Any]
    runtime: dict[str, Any]
    host: dict[str, Any]


@dataclass(frozen=True)
class CampaignContext:
    root: Path
    authorization: dict[str, Any]
    plan: dict[str, Any]
    identity: IdentitySnapshot


ProcessRunner = Callable[[dict[str, Any]], tuple[dict[str, Any], set[str]]]
CheckpointAdmitter = Callable[
    [ControllerNeutralTrainingConfigV1, BetaTrainingRuntimeProvenanceV1], dict[str, Any]
]
IdentityCapture = Callable[[], IdentitySnapshot]


@dataclass(frozen=True)
class CollectionServices:
    process: ProcessRunner
    checkpoint: CheckpointAdmitter
    identity: IdentityCapture
    # Production passes only the real filesystem function. Synthetic fixtures
    # may simulate resource exhaustion without reserving five GiB.
    budget: Callable[[Path, int], bool]


@dataclass(frozen=True)
class CollectedCampaign:
    context: CampaignContext
    cells: tuple[dict[str, Any], ...]
    abort: dict[str, str] | None


def command_for(cell: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "plant_backend.rl.v2.training",
        "--config",
        cell["config_path"],
    ]


def cell_config(cell: dict[str, Any]) -> ControllerNeutralTrainingConfigV1:
    return registered_config(
        Path(cell["output_path"]), cell["algorithm"], cell["optimizer_seed"]
    )


def identity_failures(
    expected: IdentitySnapshot, observed: IdentitySnapshot, stage: str
) -> dict[str, str]:
    failures: dict[str, str] = {}
    source_runtime_fields = (
        "git_head",
        "worktree_clean",
        "worktree_status_sha256",
        "worktree_content_sha256",
        "uv_lock_sha256",
        "plant_sim_module_path",
        "plant_sim_module_sha256",
    )
    if canonical_bytes(expected.source) != canonical_bytes(observed.source) or any(
        expected.runtime[key] != observed.runtime[key] for key in source_runtime_fields
    ):
        failures["source_mutation"] = stage
    if canonical_bytes(expected.runtime) != canonical_bytes(observed.runtime):
        failures["runtime_mutation"] = stage
    if canonical_bytes(expected.host) != canonical_bytes(observed.host):
        failures["host_mutation"] = stage
    return failures


def _stamp(root: Path) -> tuple[int, int, int, int, int]:
    io.require_directory(root)
    observed = root.lstat()
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_uid,
        observed.st_gid,
    )


def static_failures(
    context: CampaignContext, stamp: tuple[int, ...], stage: str
) -> dict[str, str]:
    failures: dict[str, str] = {}
    try:
        if _stamp(context.root) != stamp:
            failures["root_custody_lost"] = stage
    except (OSError, ValueError):
        failures["root_custody_lost"] = stage
    try:
        if io.read_regular(context.root / "authorization.json") != canonical_bytes(
            context.authorization
        ):
            failures["authorization_mutation"] = stage
    except (OSError, ValueError):
        failures["authorization_mutation"] = stage
    try:
        if io.read_regular(context.root / "plan.json") != canonical_bytes(context.plan):
            failures["plan_mutation"] = stage
        config_paths = {Path(cell["config_path"]) for cell in context.plan["cells"]}
        if set((context.root / "configs").iterdir()) != config_paths:
            failures["plan_mutation"] = stage
        for cell in context.plan["cells"]:
            raw = io.read_regular(Path(cell["config_path"]))
            if hashlib.sha256(raw).hexdigest() != cell[
                "config_sha256"
            ] or raw != canonical_bytes(cell_config(cell).to_dict()):
                failures["plan_mutation"] = stage
    except (OSError, ValueError):
        failures["plan_mutation"] = stage
    return failures


def _publish_initial(context: CampaignContext) -> tuple[int, ...]:
    io.mkdir_exclusive(context.root)
    try:
        for directory in ("configs", "logs", "attempts", "artifacts"):
            io.mkdir_exclusive(context.root / directory)
        io.publish_json(context.root / "authorization.json", context.authorization)
        io.publish_json(context.root / "plan.json", context.plan)
        for cell in context.plan["cells"]:
            io.mkdir_exclusive(Path(cell["claim_path"]).parent)
            io.publish_json(Path(cell["config_path"]), cell_config(cell).to_dict())
    except (OSError, ValueError) as reason:
        raise CampaignClosedError("evidence_publication_failed") from reason
    return _stamp(context.root)


def _unstarted(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "cell": cell,
        "state": "not_started",
        "config": io.file_record(Path(cell["config_path"])),
        "claim": None,
        "terminal": None,
        "logs": None,
        "artifact_inventory": None,
        "checkpoint": None,
        "failure": None,
    }


def _prior_evidence_unchanged(cells: list[dict[str, Any]]) -> bool:
    for record in cells:
        if record["state"] == "not_started":
            break
        for key in ("claim", "terminal"):
            reference = record[key]
            if (
                hashlib.sha256(io.read_regular(Path(reference["path"]))).hexdigest()
                != reference["raw_file_sha256"]
            ):
                return False
        if (
            io.inventory(Path(record["cell"]["output_path"]))
            != record["artifact_inventory"]
        ):
            return False
        for stream in ("stdout", "stderr"):
            descriptor = record["logs"][stream]
            if io.file_record(Path(descriptor["path"])) != descriptor:
                return False
    return True


def _layout_unchanged(
    context: CampaignContext,
    records: list[dict[str, Any]],
    active_cell: dict[str, Any] | None = None,
) -> bool:
    """Check the complete open-campaign namespace before consuming another claim."""
    directories = ("configs", "logs", "attempts", "artifacts")
    if {path.name for path in context.root.iterdir()} != {
        "authorization.json",
        "plan.json",
        *directories,
    }:
        return False
    expected: dict[str, set[str]] = {name: set() for name in directories}
    for name in directories:
        io.require_directory(context.root / name)
    for record in records:
        cell = record["cell"]
        attempt = Path(cell["claim_path"]).parent
        io.require_directory(attempt)
        expected["configs"].add(Path(cell["config_path"]).name)
        expected["attempts"].add(attempt.name)
        is_active = active_cell is not None and cell["order"] == active_cell["order"]
        claimed = is_active or record["state"] != "not_started"
        receipts = {"claimed.json"} if claimed else set()
        if claimed and not is_active:
            receipts.add(Path(record["terminal"]["path"]).name)
        if {path.name for path in attempt.iterdir()} != receipts:
            return False
        if claimed:
            expected["logs"].update(
                Path(cell[f"{name}_path"]).name for name in ("stdout", "stderr")
            )
            if os.path.lexists(cell["output_path"]):
                expected["artifacts"].add(Path(cell["output_path"]).name)
    return all(
        {path.name for path in (context.root / name).iterdir()} == names
        for name, names in expected.items()
    )


def collect_campaign(
    context: CampaignContext, services: CollectionServices
) -> CollectedCampaign:
    """Collect ordered receipts, with no production report or success-claim path.

    A publication/inventory failure raises with the root permanently consumed;
    callers must never retry, reconstruct a missing receipt, or resume this loop.
    """
    stamp = _publish_initial(context)
    records = [_unstarted(cell) for cell in context.plan["cells"]]
    previous: dict[str, Any] | None = None
    abort: dict[str, str] | None = None
    for index, cell in enumerate(context.plan["cells"]):
        try:
            before = services.identity()
            failures = identity_failures(context.identity, before, "pre_child")
            failures.update(static_failures(context, stamp, "pre_child"))
            if (
                not _prior_evidence_unchanged(records[:index])
                or not _layout_unchanged(context, records)
                or os.path.lexists(cell["output_path"])
            ):
                failures["root_custody_lost"] = "pre_child"
            if failures:
                _, abort = documents.choose_failure(failures)
                break
            if not services.budget(context.root, MINIMUM_FREE_BYTES):
                abort = {
                    "code": "storage_budget_unavailable",
                    "stage": "pre_child",
                    "message": "The remaining campaign storage budget is unavailable.",
                }
                break
        except (OSError, ValueError, RuntimeError) as reason:
            raise CampaignClosedError("root_custody_lost") from reason

        claim = documents.make_claim(
            context.plan, cell, previous, timestamp=documents.now_utc()
        )
        try:
            io.publish_json(Path(cell["claim_path"]), claim)
            if io.read_regular(Path(cell["claim_path"])) != canonical_bytes(claim):
                raise ValueError("claim bytes changed at the persistence barrier")
        except (OSError, ValueError) as reason:
            raise CampaignClosedError("claim_publication_failed") from reason

        try:
            # This is the first operation that can start a process. Both the
            # complete file and directory fsync barriers have returned.
            process, process_codes = services.process(dict(cell))
            failures = {code: "child" for code in process_codes}
            after = services.identity()
            failures.update(identity_failures(context.identity, after, "post_child"))
            failures.update(static_failures(context, stamp, "post_child"))
            if not _prior_evidence_unchanged(records[:index]) or not _layout_unchanged(
                context, records, cell
            ):
                failures["root_custody_lost"] = "post_child"
            artifact_inventory = io.inventory(Path(cell["output_path"]))
            logs = {
                name: io.file_record(Path(cell[f"{name}_path"]))
                for name in ("stdout", "stderr")
            }
            measured_failures = process_failures(process, logs, command_for(cell))
            if set(process_codes) - {"root_custody_lost"} != set(measured_failures):
                raise ValueError("process result codes differ from measured evidence")
            failures.update(measured_failures)
            checkpoint = None
            if not failures:
                try:
                    checkpoint = services.checkpoint(
                        cell_config(cell),
                        BetaTrainingRuntimeProvenanceV1.from_dict(before.runtime),
                    )
                    checkpoint_descriptor(checkpoint, before.runtime["sha256"])
                except CheckpointAdmissionFailure as reason:
                    failures[reason.code] = (
                        "inventory"
                        if reason.code == "artifact_inventory_invalid"
                        else "admission"
                    )
                except Exception:
                    failures["checkpoint_admission_failed"] = "admission"
                # Admission itself is inside custody, including model-load
                # attestation. Retain any identity drift before a receipt exists.
                after = services.identity()
                failures.update(
                    identity_failures(context.identity, after, "post_child")
                )
                failures.update(static_failures(context, stamp, "post_child"))
                if not _prior_evidence_unchanged(
                    records[:index]
                ) or not _layout_unchanged(context, records, cell):
                    failures["root_custody_lost"] = "post_child"
                if io.inventory(Path(cell["output_path"])) != artifact_inventory:
                    failures["root_custody_lost"] = "post_child"
                    artifact_inventory = io.inventory(Path(cell["output_path"]))
            status, selected_failure = documents.choose_failure(failures)
            terminal = authenticate(
                {
                    "schema": documents.TERMINAL_SCHEMA,
                    "status": status,
                    "protocols": context.plan["protocols"],
                    "authorization_sha256": context.authorization["sha256"],
                    "plan_sha256": context.plan["sha256"],
                    "claim_sha256": claim["sha256"],
                    "cell": cell,
                    "process": process,
                    "logs": logs,
                    "artifact_inventory": artifact_inventory,
                    "checkpoint": checkpoint if status == "admitted" else None,
                    "failure": selected_failure,
                    "source_before": before.source,
                    "source_after": after.source,
                    "runtime_before": before.runtime,
                    "runtime_after": after.runtime,
                    "host_before": before.host,
                    "host_after": after.host,
                    "finished_at_utc": documents.now_utc(),
                }
            )
        except BaseException as reason:
            # Incomplete evidence never becomes a fabricated terminal. The
            # durable claim remains available to detached forensic inspection.
            raise CampaignClosedError("root_custody_lost") from reason
        terminal_path = Path(
            cell["completed_receipt_path"]
            if status == "admitted"
            else cell["failed_receipt_path"]
        )
        try:
            io.publish_json(terminal_path, terminal)
            persisted = require_envelope(
                io.read_regular(terminal_path),
                schema=documents.TERMINAL_SCHEMA,
                fields=documents.TERMINAL_FIELDS,
            )
            if persisted != terminal:
                raise ValueError("terminal receipt changed at publication")
        except (OSError, ValueError) as reason:
            raise CampaignClosedError("evidence_publication_failed") from reason
        records[index] = {
            "cell": cell,
            "state": status,
            "config": records[index]["config"],
            "claim": documents.envelope_reference(Path(cell["claim_path"]), claim),
            "terminal": documents.envelope_reference(terminal_path, terminal),
            "logs": logs,
            "artifact_inventory": artifact_inventory,
            "checkpoint": terminal["checkpoint"],
            "failure": selected_failure,
        }
        previous = terminal
        if status == "campaign_integrity_failed":
            abort = selected_failure
            break
    return CollectedCampaign(context, tuple(records), abort)

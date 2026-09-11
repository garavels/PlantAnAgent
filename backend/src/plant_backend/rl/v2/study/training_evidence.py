"""Read-only bridge from complete registered training custody to study inputs.

The production entry point reopens training custody and all registered models
through its fixed verifier. It never authorizes an evaluation or resets a plant.
Keep this admission in the parent process under the original training thread
contract; evaluation's single-thread configuration belongs in fresh children.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
import stat
from typing import Any

from ..training import custody_documents as documents
from ..training import custody_io as io
from ..training import custody_verifier
from ..training.beta_checkpoint import parse_beta_checkpoint_evidence
from ..training.custody_json import authenticate, canonical_bytes, decode
from ..training.custody_runtime import capture_identity
from ..training.custody_state import IdentitySnapshot, identity_failures
from ..training.custody_validation import exact
from ..training.ppo import BETA_CHECKPOINT_EXTRA_KEY
from ..training.registered_contract import matrix
from .contract import PREFIX
from .registry import actors
from .training_scorecards import TrainingSlot, training_scorecard

SCHEMA = PREFIX + "registered-training-inputs-v1"


def _tree_stamps(root: Path) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Metadata-only, stable, non-following walk; detects identical replacements."""
    io.require_directory(root)
    result: list[tuple[str, tuple[int, ...]]] = []

    def visit(path: Path) -> None:
        before = path.lstat()
        stamp = (*io._identity(before), before.st_uid, before.st_gid)
        result.append((path.relative_to(root).as_posix(), stamp))
        if stat.S_ISDIR(before.st_mode):
            for child in sorted(
                path.iterdir(), key=lambda item: os.fsencode(item.name)
            ):
                visit(child)
        # Retain forbidden artifact nodes without following/opening them. The
        # complete training verifier checks their location and failed receipt;
        # their presence must not erase another policy's admitted checkpoint.
        after = path.lstat()
        if (*io._identity(after), after.st_uid, after.st_gid) != stamp:
            raise ValueError("training evidence graph changed during its snapshot")

    visit(root)
    return tuple(result)


def _rollouts(checkpoint: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Reopen the exact already-admitted metadata and complete endpoint ledger."""
    target = Path(checkpoint["path"]) / "metadata.json"
    expected = checkpoint["files"][0]
    exact("admitted metadata path", expected["path"], str(target))
    exact("admitted metadata file", io.file_record(target), expected)
    metadata = decode(io.read_regular(target))
    evidence = parse_beta_checkpoint_evidence(
        metadata["extra"][BETA_CHECKPOINT_EXTRA_KEY]
    )
    exact(
        "admitted checkpoint evidence",
        evidence.sha256,
        checkpoint["checkpoint_evidence_sha256"],
    )
    exact(
        "admitted checkpoint runtime",
        evidence.runtime_provenance.sha256,
        checkpoint["runtime_provenance_sha256"],
    )
    rows = evidence.constraint_state["payload"]["rollout_ledgers"]
    exact(
        "complete endpoint rollout count", len(rows), checkpoint["rollout_entry_count"]
    )
    exact("registered rollout count", len(rows), 128)
    exact(
        "complete endpoint optimizer steps",
        sum(row["optimizer_steps"] for row in rows),
        40960,
    )
    exact("metadata after ledger decoding", io.file_record(target), expected)
    return tuple(deepcopy(rows))


def _slots(report: dict[str, Any], root: Path) -> tuple[TrainingSlot, ...]:
    """Projection of admitted records; callable arithmetic grants no authority."""
    expected_cells = matrix(root)
    if len(report["cells"]) != len(expected_cells):
        raise ValueError("training inputs require every fixed training slot")
    slots = []
    for record, cell, actor in zip(
        report["cells"], expected_cells, actors()[3:], strict=True
    ):
        exact("registered training input cell", record["cell"], cell)
        state = record["state"]
        if state not in documents.CELL_STATES or state == "claimed_interrupted":
            raise ValueError("training inputs require closed verified custody")
        admitted = state == "admitted"
        if admitted != (record["checkpoint"] is not None):
            raise ValueError("training checkpoint differs from custody status")
        if admitted:
            if record["failure"] is not None:
                raise ValueError("admitted training slot has a failure")
            rows = _rollouts(record["checkpoint"])
            failure = None
        else:
            rows = None
            failure = (
                "not_started" if state == "not_started" else record["failure"]["code"]
            )
        slots.append(
            TrainingSlot(
                actor["algorithm"],
                actor["optimizer_seed"],
                state in {"admitted", "checkpoint_rejected"},
                admitted,
                failure,
                rows,
            )
        )
    return tuple(slots)


@dataclass(frozen=True)
class TrainingAdmission:
    """In-memory result only; persisted documents must be independently readmitted."""

    path: Path
    report: dict[str, Any]
    slots: tuple[TrainingSlot, ...]
    evidence: dict[str, Any]
    identity: IdentitySnapshot
    _stamps: tuple[tuple[str, tuple[int, ...]], ...]

    def recheck(self) -> None:
        """Before/after parent work, reject any graph or runtime/host change."""
        if _tree_stamps(self.path.parent) != self._stamps:
            raise ValueError("training evidence graph changed after admission")
        if io.read_regular(self.path) != canonical_bytes(self.report):
            raise ValueError("training report changed after admission")
        authorization = decode(io.read_regular(self.path.parent / "authorization.json"))
        if identity_failures(
            self.identity, capture_identity(authorization), "post_child"
        ):
            raise ValueError("training admission source/runtime/host changed")
        if _tree_stamps(self.path.parent) != self._stamps:
            raise ValueError("training evidence graph changed during recapture")


def admit_training(path: Path | str) -> TrainingAdmission:
    """Fresh production admission; no fixture flag, dependency hook or reset."""
    target = Path(path)
    # Reject synthetic/cache/ambiguous entry points before any graph walk,
    # host probe or model loading. Production authority is checked by the
    # complete verifier before it opens logs/checkpoints.
    custody_verifier._read_report(target, synthetic=False)
    stamps = _tree_stamps(target.parent)
    report = custody_verifier.read_verified_report(target)
    slots = _slots(report, target.parent)
    scorecard = training_scorecard(slots)
    report_reference = documents.envelope_reference(target, report)
    identity = IdentitySnapshot(
        report["source"], report["runtime_provenance"], report["host"]
    )
    evidence = authenticate(
        {
            "schema": SCHEMA,
            "training_report": report_reference,
            "training_campaign_status": report["campaign_status"],
            "training_abort": deepcopy(report["abort"]),
            "source": deepcopy(report["source"]),
            "runtime_provenance": deepcopy(report["runtime_provenance"]),
            "training_host": deepcopy(report["host"]),
            "actors": actors()[3:],
            "checkpoints": [
                deepcopy(record["checkpoint"]) for record in report["cells"]
            ],
            "scorecard": scorecard,
            "source_evidence_verified": True,
            "registered_checkpoint_admission_performed": True,
            "evaluation_authorized": False,
            "result_claim": None,
        }
    )
    result = TrainingAdmission(target, report, slots, evidence, identity, stamps)
    result.recheck()
    return result

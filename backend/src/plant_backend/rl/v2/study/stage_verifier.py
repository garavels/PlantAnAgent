"""Full detached stage reconstruction; cached verdicts never admit a stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..training.custody_json import authenticate, canonical_bytes, require_envelope
from ..training.custody_validation import exact
from . import archive, stage_documents as documents, stage_state
from .campaign_analysis import CellInput, CustodyFacts, analyze_campaign
from .stage_authority import StageAdmission, admit_stage
from .training_evidence import _tree_stamps

SUCCESS_CLAIM = "Under the frozen synthetic cyclic-market beta and declared economic assumptions, a preregistered PI-Lagrangian PPO controller improved verified risk-adjusted simulated operating value over the beta economic MPC while preserving matched service, hard safety, and every cyclic resource return."


def build_report(
    context: StageAdmission,
    opened: dict[str, Any],
    records: list[dict[str, Any]],
    inputs: list[CellInput],
    abort: dict[str, Any] | None,
) -> dict[str, Any]:
    unclaimed = [row for row in records if row["claim"] is None]
    integrity = [row for row in records if row["failure"] == "stage_integrity_lost"]
    if abort is None:
        if unclaimed or integrity:
            raise ValueError("incomplete stage has no abort")
    else:
        if set(abort) != {"code", "before_order"} or abort["code"] not in {
            "storage_budget_unavailable",
            "stage_integrity_lost",
            "operator_interrupted",
        }:
            raise ValueError("stage abort fields differ")
        if type(abort["before_order"]) is not int or abort["before_order"] != 577 - len(
            unclaimed
        ):
            raise ValueError("stage abort is not at the next unconsumed boundary")
        if abort["code"] == "storage_budget_unavailable" and (
            not unclaimed or integrity
        ):
            raise ValueError("storage abort does not explain the retained stage")
        if integrity and abort["code"] != "stage_integrity_lost":
            raise ValueError("stage integrity terminal lacks its matching abort")
    analysis = analyze_campaign(
        stage=context.registry["stage"],
        episode_seeds=context.registry["episode_seeds"],
        cells=inputs,
        training_slots=context.training.slots,
        facts=CustodyFacts(
            training_verified=True,
            stage_verified=abort is None,
            registered_hardware_verified=True,
            identities_frozen=True,
            published_fields_verified=True,
        ),
        frozen_champions=(
            None
            if context.development is None
            else context.development.report["analysis"]["frozen_champions"]
        ),
        expected_host_sha256=context.authorization["latency_host"]["sha256"],
    )
    success = analysis["heldout_success_requirements"]
    return authenticate(
        {
            "schema": documents.REPORT_SCHEMA,
            "protocol": documents.protocol(),
            "stage": context.registry["stage"],
            "registry": documents.reference(
                Path(context.registry["registry_path"]), context.registry
            ),
            "authorization": documents.reference(
                Path(context.registry["authorization_path"]), context.authorization
            ),
            "opening": documents.reference(
                Path(context.registry["opening_claim_path"]), opened
            ),
            "training_inputs_sha256": context.training.evidence["sha256"],
            "cells": records,
            "abort": abort,
            "campaign_status": "completed" if abort is None else "aborted",
            "analysis": analysis,
            "result_claim": (
                SUCCESS_CLAIM
                if abort is None
                and success is not None
                and success["all_calculated_requirements_met"]
                else None
            ),
        }
    )


def verdict(report: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    return authenticate(
        {
            "schema": documents.VERDICT_SCHEMA,
            "stage": report["stage"],
            "report_archive": record,
            "report_sha256": report["sha256"],
            "campaign_status": report["campaign_status"],
            "state_counts": report["analysis"]["state_counts"],
            "frozen_champions": report["analysis"]["frozen_champions"],
            "source_evidence_verified": True,
            "execution_authorized": False,
            "result_claim": report["result_claim"],
        }
    )


@dataclass(frozen=True)
class StageVerification:
    context: StageAdmission
    report: dict[str, Any]
    archive: dict[str, Any]
    stamps: tuple[tuple[str, tuple[int, ...]], ...]

    def recheck(self) -> None:
        self.context.recheck()
        if _tree_stamps(self.context.root) != self.stamps:
            raise ValueError("closed stage evidence changed after reconstruction")


def verify_stage(training_root: Path | str, stage: str) -> StageVerification:
    context = admit_stage(training_root, stage)
    stage_state.check_layout(context, closed=True)
    before = _tree_stamps(context.root)
    opened = stage_state.opened_context(context)
    raw, record = archive.read_archive(
        Path(context.registry["report_path"]),
        stored_limit=context.budget.report_stored_bytes,
        uncompressed_limit=context.budget.report_decoded_bytes,
    )
    report = require_envelope(
        canonical_bytes(raw),
        schema=documents.REPORT_SCHEMA,
        fields=documents.REPORT_FIELDS,
    )
    records, inputs = stage_state.read_cells(context, opened)
    rebuilt = build_report(context, opened, records, inputs, report["abort"])
    exact("complete registered stage report", report, rebuilt)
    path = Path(context.registry["verification_path"])
    if path.exists():
        expected = verdict(rebuilt, record)
        exact(
            "stored stage verdict",
            documents.read(path, documents.VERDICT_SCHEMA, frozenset(expected)),
            expected,
        )
    stage_state.check_layout(context, closed=True)
    result = StageVerification(context, rebuilt, record, before)
    result.recheck()
    return result

"""Explicitly synthetic state-machine fixtures, incapable of a production claim."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import custody_documents as documents
from . import custody_io as io
from .custody_authorization import repository_root
from .custody_json import authenticate
from .custody_state import (
    CampaignClosedError,
    CampaignContext,
    CollectionServices,
    collect_campaign,
)
from .registered_contract import MINIMUM_FREE_BYTES


def run_synthetic(
    context: CampaignContext, services: CollectionServices
) -> dict[str, Any]:
    """Injected execution always has a synthetic filename, schema, and null claim."""
    io.require_new_root(context.root, repository_root(), 0)
    if not services.budget(context.root.parent, MINIMUM_FREE_BYTES):
        raise ValueError("synthetic campaign admission storage is unavailable")
    trace = collect_campaign(context, services)
    counts = {
        state: sum(cell["state"] == state for cell in trace.cells)
        for state in documents.CELL_STATES
    }
    report = authenticate(
        {
            "schema": documents.SYNTHETIC_REPORT_SCHEMA,
            "protocols": context.plan["protocols"],
            "claim_boundary": "synthetic-custody-fixture; no registered training or production claim",
            "result_claim": None,
            "authorization": context.plan["authorization"],
            "plan": documents.envelope_reference(
                context.root / "plan.json", context.plan
            ),
            "source": context.identity.source,
            "runtime_provenance": context.identity.runtime,
            "host": context.identity.host,
            "execution": context.plan["execution"],
            "cells": list(trace.cells),
            "status_counts": counts,
            "campaign_status": (
                "aborted"
                if trace.abort
                else ("completed" if counts["admitted"] == 15 else "failed")
            ),
            "abort": trace.abort,
            "deterministic_evidence_sha256": context.plan[
                "deterministic_evidence_sha256"
            ],
        }
    )
    try:
        io.publish_json(context.root / "synthetic-report.json", report)
    except (OSError, ValueError) as reason:
        raise CampaignClosedError("evidence_publication_failed") from reason
    return report


def require_synthetic_path(path: Path) -> None:
    if path.name != "synthetic-report.json":
        raise ValueError("injected custody verification requires synthetic-report.json")

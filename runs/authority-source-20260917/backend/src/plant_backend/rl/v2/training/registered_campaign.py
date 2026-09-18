"""One-shot registered training custody and its detached, read-only verifier."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from . import capacity_benchmark as capacity
from . import custody_authorization as authority
from . import custody_documents as documents
from . import custody_io as io
from .custody_checkpoint import admit_checkpoint
from .custody_json import authenticate, canonical_bytes, decode
from .custody_launch import LAUNCH_FD_ENV, launch_message
from .custody_process import capture_process
from .custody_runtime import capture_identity
from .custody_state import (
    CampaignClosedError,
    CampaignContext,
    CollectionServices,
    collect_campaign,
    command_for,
    identity_failures,
    static_failures,
    _stamp,
)
from .custody_validation import exact
from .custody_verifier import inspect_root, verify_report
from .registered_contract import LOG_LIMIT_BYTES, MINIMUM_FREE_BYTES, TIMEOUT_SECONDS


def _run_cell(cell: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    return capture_process(
        command_for(cell),
        stdout=Path(cell["stdout_path"]),
        stderr=Path(cell["stderr_path"]),
        timeout_seconds=TIMEOUT_SECONDS,
        log_limit_bytes=LOG_LIMIT_BYTES,
        launch_permission=launch_message(cell),
    )


def _post_publication_verification(
    context: CampaignContext, report: dict[str, Any], root_stamp: tuple[int, ...]
) -> dict[str, Any]:
    identity_changed = False
    try:
        after = capture_identity(context.authorization)
        identity_changed = bool(
            identity_failures(context.identity, after, "post_child")
            or static_failures(context, root_stamp, "post_child")
        )
        authority.admit_authorization(context.root, after.host)
    except BaseException:
        identity_changed = True
    # This fresh-process invocation is unconditional after publication, even
    # when the local recapture just failed. Its failure never rewrites a report.
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "plant_backend.rl.v2.training.registered_campaign",
                "verify",
                str(context.root / "report.json"),
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=TIMEOUT_SECONDS,
        )
    except BaseException as reason:
        raise CampaignClosedError("post_publication_verification_failed") from reason
    if (
        identity_changed
        or result.returncode != 0
        or result.stderr
        or len(result.stdout) > 1024 * 1024
    ):
        raise CampaignClosedError("post_publication_verification_failed")
    try:
        verification = decode(result.stdout)
        exact(
            "detached verification",
            verification,
            {
                "verified": True,
                "schema": documents.REPORT_SCHEMA,
                "report_sha256": report["sha256"],
                "campaign_status": report["campaign_status"],
                "status_counts": report["status_counts"],
                "result_claim": report["result_claim"],
            },
        )
    except (ValueError, TypeError) as reason:
        raise CampaignClosedError("post_publication_verification_failed") from reason
    return verification


def run_registered_campaign(
    root: Path | str,
    *,
    assert_exclusive: bool = False,
    assert_first_attempt: bool = False,
    assert_authorized: bool = False,
) -> dict[str, Any]:
    """Fixed production dependencies only; never accept synthetic execution hooks."""
    if any(
        flag is not True
        for flag in (assert_exclusive, assert_first_attempt, assert_authorized)
    ):
        raise ValueError(
            "registered custody requires all three explicit operator assertions"
        )
    if LAUNCH_FD_ENV in os.environ:
        raise ValueError("a registered child launch cannot create another campaign")
    target = Path(root)
    host = capacity._verify_host(capacity._host_identity())
    authorization, source = authority.admit_authorization(target, host)
    identity = capture_identity(authorization)
    exact("admitted source", identity.source, source)
    exact("admitted host", identity.host, host)
    if not identity.runtime["worktree_clean"]:
        raise ValueError("registered custody requires clean runtime provenance")
    io.require_new_root(target, authority.repository_root(), MINIMUM_FREE_BYTES)
    plan = documents.make_plan(
        target,
        authorization=authorization,
        source=source,
        runtime=identity.runtime,
        host=identity.host,
    )
    context = CampaignContext(target, authorization, plan, identity)
    trace = collect_campaign(
        context,
        CollectionServices(
            process=_run_cell,
            checkpoint=admit_checkpoint,
            identity=lambda: capture_identity(authorization),
            budget=io.budget_available,
        ),
    )
    counts = {
        state: sum(cell["state"] == state for cell in trace.cells)
        for state in documents.CELL_STATES
    }
    status = (
        "aborted"
        if trace.abort
        else ("completed" if counts["admitted"] == 15 else "failed")
    )
    # This production report is constructed only here, after the non-injected
    # authority, source, process and registered-loader path has completed.
    report = authenticate(
        {
            "schema": documents.REPORT_SCHEMA,
            "protocols": plan["protocols"],
            "claim_boundary": documents.CLAIM_BOUNDARY,
            "result_claim": documents.SUCCESS_CLAIM if status == "completed" else None,
            "authorization": plan["authorization"],
            "plan": documents.envelope_reference(target / "plan.json", plan),
            "source": source,
            "runtime_provenance": identity.runtime,
            "host": identity.host,
            "execution": plan["execution"],
            "cells": list(trace.cells),
            "status_counts": counts,
            "campaign_status": status,
            "abort": trace.abort,
            "deterministic_evidence_sha256": plan["deterministic_evidence_sha256"],
        }
    )
    root_stamp = _stamp(target)
    try:
        io.publish_json(target / "report.json", report)
    except (OSError, ValueError) as reason:
        raise CampaignClosedError("evidence_publication_failed") from reason
    return _post_publication_verification(context, report, root_stamp)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Registered training custody; execution requires committed authorization."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="consume the exact authorized 15-cell campaign once"
    )
    run.add_argument("root", type=Path)
    for name in ("exclusive", "first-attempt", "authorized"):
        run.add_argument(f"--assert-{name}", action="store_true")
    verify = commands.add_parser(
        "verify",
        help="independently verify an existing production report without any reset",
    )
    verify.add_argument("report", type=Path)
    inspect = commands.add_parser(
        "inspect",
        help="inspect consumed or interrupted claims without admitting a campaign",
    )
    inspect.add_argument("root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_report(args.report)
        elif args.command == "inspect":
            result = inspect_root(args.root)
        else:
            result = run_registered_campaign(
                args.root,
                assert_exclusive=args.assert_exclusive,
                assert_first_attempt=args.assert_first_attempt,
                assert_authorized=args.assert_authorized,
            )
    except BaseException:
        # Exception strings can contain child-controlled metadata. Keep them
        # out of the operator channel; failed roots and sealed evidence remain.
        print(
            "Registered custody admission or verification failed; no verified campaign claim was issued.",
            file=sys.stderr,
        )
        return 2
    print(canonical_bytes(result).decode("utf-8"), end="")
    return (
        0
        if args.command in {"verify", "inspect"}
        or result["campaign_status"] == "completed"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

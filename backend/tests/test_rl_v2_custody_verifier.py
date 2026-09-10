"""Detached verification and corruption checks using permanently synthetic traces."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from custody_fixtures import FixtureServices, admit, context
from plant_backend.rl.v2.training import custody_documents as docs
from plant_backend.rl.v2.training import custody_verifier as verifier
from plant_backend.rl.v2.training.custody_checkpoint import CheckpointAdmissionFailure
from plant_backend.rl.v2.training.custody_json import (
    authenticate,
    canonical_bytes,
    decode,
)
from plant_backend.rl.v2.training.custody_synthetic import run_synthetic


def test_detached_synthetic_verification_readmits_every_checkpoint(
    tmp_path: Path,
) -> None:
    ctx = context(tmp_path / "campaign")
    fixture = FixtureServices(ctx)
    report = run_synthetic(ctx, fixture.services)
    admitted: list[int] = []

    def checkpoint(config: Any, runtime: Any) -> dict[str, Any]:
        admitted.append(config.seed)
        return admit(config, runtime)

    result = verifier.verify_synthetic_report(
        ctx.root / "synthetic-report.json", checkpoint=checkpoint
    )
    assert result["verified"] is True and result["result_claim"] is None
    assert result["report_sha256"] == report["sha256"]
    assert admitted == [101, 211, 307, 401, 503] * 3


@pytest.mark.parametrize(
    "code",
    ["child_timeout", "child_signaled", "child_spawn_failed", "child_nonzero_exit"],
)
def test_detached_failure_accounting_does_not_drop_failed_cells(
    tmp_path: Path, code: str
) -> None:
    ctx = context(tmp_path / "campaign")
    fixture = FixtureServices(ctx)
    fixture.process_code = code
    run_synthetic(ctx, fixture.services)
    admitted: list[int] = []

    def checkpoint(config: Any, runtime: Any) -> dict[str, Any]:
        admitted.append(config.seed)
        return admit(config, runtime)

    result = verifier.verify_synthetic_report(
        ctx.root / "synthetic-report.json", checkpoint=checkpoint
    )
    assert result["verified"] is True and len(admitted) == 14


def test_detached_storage_abort_accounts_for_unclaimed_cells(tmp_path: Path) -> None:
    ctx = context(tmp_path / "campaign")
    fixture = FixtureServices(ctx)
    budget_calls = 0

    def budget(root: Path, minimum: int) -> bool:
        nonlocal budget_calls
        budget_calls += 1
        return budget_calls <= 2

    run_synthetic(ctx, replace(fixture.services, budget=budget))
    assert (
        verifier.verify_synthetic_report(
            ctx.root / "synthetic-report.json", checkpoint=admit
        )["verified"]
        is True
    )


@pytest.mark.parametrize(
    "part",
    [
        "config",
        "claim",
        "terminal",
        "log",
        "policy",
        "metadata",
        "checkpoint_config",
        "partial",
        "plan",
        "authorization",
        "extra_root",
        "extra_attempt",
        "extra_config",
        "symlink",
    ],
)
def test_detached_verifier_reopens_every_declared_file(
    tmp_path: Path, part: str
) -> None:
    ctx = context(tmp_path / "campaign")
    fixture = FixtureServices(ctx)
    run_synthetic(ctx, fixture.services)
    cell = ctx.plan["cells"][0]
    paths = {
        "config": Path(cell["config_path"]),
        "claim": Path(cell["claim_path"]),
        "terminal": Path(cell["completed_receipt_path"]),
        "log": Path(cell["stdout_path"]),
        "policy": Path(cell["output_path"]) / "checkpoint/policy.zip",
        "metadata": Path(cell["output_path"]) / "checkpoint/metadata.json",
        "checkpoint_config": Path(cell["output_path"])
        / "checkpoint/training_config.json",
        "plan": ctx.root / "plan.json",
        "authorization": ctx.root / "authorization.json",
    }
    if part in paths:
        with paths[part].open("ab") as handle:
            handle.write(b"changed")
    elif part == "partial":
        (Path(cell["output_path"]) / "unexpected.bin").write_bytes(b"retained extra")
    elif part == "extra_root":
        (ctx.root / "unexpected.txt").write_text("extra")
    elif part == "extra_attempt":
        Path(cell["failed_receipt_path"]).write_bytes(
            Path(cell["completed_receipt_path"]).read_bytes()
        )
    elif part == "extra_config":
        (ctx.root / "configs/16-extra.json").write_text("{}")
    else:
        path = Path(cell["stdout_path"])
        path.unlink()
        path.symlink_to(Path(cell["stderr_path"]))
    with pytest.raises((ValueError, RuntimeError, OSError)):
        verifier.verify_synthetic_report(
            ctx.root / "synthetic-report.json", checkpoint=admit
        )


@pytest.mark.parametrize(
    "field",
    [
        "source",
        "runtime_provenance",
        "host",
        "execution",
        "status_counts",
        "campaign_status",
        "result_claim",
        "claim_boundary",
        "cells",
        "deterministic_evidence_sha256",
    ],
)
def test_rehashing_a_report_cannot_change_its_semantics(
    tmp_path: Path, field: str
) -> None:
    ctx = context(tmp_path / "campaign")
    report = run_synthetic(ctx, FixtureServices(ctx).services)
    changed = deepcopy(report)
    changed[field] = "coherently rehashed mutation"
    (ctx.root / "synthetic-report.json").write_bytes(
        canonical_bytes(authenticate(changed))
    )
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        verifier.verify_synthetic_report(
            ctx.root / "synthetic-report.json", checkpoint=admit
        )


def test_checkpoint_admission_is_repeated_independently(tmp_path: Path) -> None:
    ctx = context(tmp_path / "campaign")
    run_synthetic(ctx, FixtureServices(ctx).services)

    def reject(config: Any, runtime: Any) -> dict[str, Any]:
        raise CheckpointAdmissionFailure("checkpoint_admission_failed")

    with pytest.raises(ValueError, match="terminal status"):
        verifier.verify_synthetic_report(
            ctx.root / "synthetic-report.json", checkpoint=reject
        )


def test_production_and_synthetic_doors_reject_each_other_before_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = context(tmp_path / "campaign")
    report = run_synthetic(ctx, FixtureServices(ctx).services)

    def forbidden(*args: object, **kwargs: object) -> Any:
        pytest.fail(
            "wrong schema accessed runtime, host, authorization or checkpoint dependencies"
        )

    monkeypatch.setattr(verifier, "capture_identity", forbidden)
    monkeypatch.setattr(verifier.capacity, "_host_identity", forbidden)
    monkeypatch.setattr(verifier.authority, "admit_authorization", forbidden)
    monkeypatch.setattr(verifier, "admit_checkpoint", forbidden)
    with pytest.raises(ValueError):
        verifier.verify_report(ctx.root / "synthetic-report.json")
    (ctx.root / "report.json").write_bytes(canonical_bytes(report))
    with pytest.raises(ValueError, match="schema"):
        verifier.verify_report(ctx.root / "report.json")
    with pytest.raises(ValueError):
        verifier.verify_synthetic_report(ctx.root / "report.json", checkpoint=forbidden)
    changed = authenticate({**report, "schema": docs.REPORT_SCHEMA})
    (ctx.root / "synthetic-report.json").write_bytes(canonical_bytes(changed))
    with pytest.raises(ValueError, match="schema"):
        verifier.verify_synthetic_report(
            ctx.root / "synthetic-report.json", checkpoint=forbidden
        )


def test_interrupted_claim_cannot_be_verified_as_complete(tmp_path: Path) -> None:
    ctx = context(tmp_path / "campaign")
    run_synthetic(ctx, FixtureServices(ctx).services)
    Path(ctx.plan["cells"][-1]["completed_receipt_path"]).unlink()
    with pytest.raises(ValueError, match="claimed_interrupted"):
        verifier.verify_synthetic_report(
            ctx.root / "synthetic-report.json", checkpoint=admit
        )


@pytest.mark.parametrize(
    "part", ["report", "plan", "authorization", "claim", "log", "policy", "extra_root"]
)
def test_verification_rechecks_evidence_after_all_checkpoint_loads(
    tmp_path: Path, part: str
) -> None:
    ctx = context(tmp_path / "campaign")
    run_synthetic(ctx, FixtureServices(ctx).services)
    first = ctx.plan["cells"][0]
    paths = {
        "report": ctx.root / "synthetic-report.json",
        "plan": ctx.root / "plan.json",
        "authorization": ctx.root / "authorization.json",
        "claim": Path(first["claim_path"]),
        "log": Path(first["stdout_path"]),
        "policy": Path(first["output_path"]) / "checkpoint/policy.zip",
        "extra_root": ctx.root / "unlisted.txt",
    }
    calls = 0

    def mutate(config: Any, runtime: Any) -> dict[str, Any]:
        nonlocal calls
        descriptor = admit(config, runtime)
        calls += 1
        if calls == 15:
            with paths[part].open("ab") as handle:
                handle.write(b"changed during final checkpoint admission")
        return descriptor

    with pytest.raises((ValueError, OSError)):
        verifier.verify_synthetic_report(
            ctx.root / "synthetic-report.json", checkpoint=mutate
        )
    assert calls == 15

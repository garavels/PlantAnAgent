"""End-to-end synthetic receipt custody with no protected reset or training."""

from dataclasses import replace
from pathlib import Path
import shutil
from typing import Any

import pytest

from custody_fixtures import FixtureServices, admit, context
from plant_backend.rl.v2.training import custody_documents as docs
from plant_backend.rl.v2.training import custody_io as io
from plant_backend.rl.v2.training.custody_checkpoint import CheckpointAdmissionFailure
from plant_backend.rl.v2.training.custody_json import decode
from plant_backend.rl.v2.training.custody_state import CampaignClosedError
from plant_backend.rl.v2.training.custody_synthetic import run_synthetic
from plant_backend.rl.v2.training.custody_verifier import verify_synthetic_report


def test_all_fifteen_attempts_are_predeclared_and_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = context(tmp_path / "campaign")
    fixture = FixtureServices(ctx)
    barriers: list[str] = []
    fsync = io.fsync_directory

    def record_barrier(path: Path) -> None:
        fsync(path)
        barriers.append(str(path))

    monkeypatch.setattr(io, "fsync_directory", record_barrier)

    def process(cell: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
        assert str(Path(cell["claim_path"]).parent) == barriers[-1]
        return fixture.process(cell)

    report = run_synthetic(ctx, replace(fixture.services, process=process))
    assert fixture.started == list(range(1, 16))
    assert report["status_counts"]["admitted"] == 15
    assert report["result_claim"] is None
    assert report["schema"] == docs.SYNTHETIC_REPORT_SCHEMA
    assert (ctx.root / "synthetic-report.json").exists()
    assert not (ctx.root / "report.json").exists()
    previous = None
    for item in report["cells"]:
        claim = decode(Path(item["claim"]["path"]).read_bytes())
        terminal = decode(Path(item["terminal"]["path"]).read_bytes())
        assert claim["previous_terminal_sha256"] == previous
        assert terminal["claim_sha256"] == claim["sha256"]
        assert terminal["status"] == "admitted"
        previous = terminal["sha256"]
    with pytest.raises(ValueError, match="already exists"):
        run_synthetic(ctx, fixture.services)
    assert len(fixture.started) == 15


@pytest.mark.parametrize(
    "code",
    [
        "child_spawn_failed",
        "child_nonzero_exit",
        "child_signaled",
        "child_timeout",
        "child_log_limit_exceeded",
    ],
)
def test_process_failures_continue_the_exact_remaining_matrix(
    tmp_path: Path, code: str
) -> None:
    ctx = context(tmp_path / "campaign")
    fixture = FixtureServices(ctx)
    fixture.process_code = code
    report = run_synthetic(ctx, fixture.services)
    assert fixture.started == list(range(1, 16))
    assert report["cells"][1]["failure"]["code"] == code
    assert report["cells"][1]["state"] == "process_failed"
    assert report["cells"][1]["checkpoint"] is None
    assert report["cells"][1]["artifact_inventory"]["regular_file_count"] == 3
    assert report["campaign_status"] == "failed" and report["result_claim"] is None
    assert report["status_counts"]["admitted"] == 14


@pytest.mark.parametrize(
    "code",
    [
        "checkpoint_missing",
        "artifact_inventory_invalid",
        "checkpoint_malformed",
        "checkpoint_admission_failed",
    ],
)
def test_checkpoint_rejections_are_retained_and_never_retried(
    tmp_path: Path, code: str
) -> None:
    ctx = context(tmp_path / "campaign")
    fixture = FixtureServices(ctx)

    def reject(config: Any, runtime: Any) -> dict[str, Any]:
        if config.seed == 211 and config.beta_rl.algorithm.endswith("pi-lagrangian-v1"):
            raise CheckpointAdmissionFailure(code)
        return admit(config, runtime)

    report = run_synthetic(ctx, replace(fixture.services, checkpoint=reject))
    assert fixture.started == list(range(1, 16))
    assert report["cells"][1]["failure"]["code"] == code
    assert report["cells"][1]["state"] == "checkpoint_rejected"
    assert report["status_counts"]["admitted"] == 14


@pytest.mark.parametrize(
    "code",
    [
        "source_mutation",
        "runtime_mutation",
        "host_mutation",
        "authorization_mutation",
        "plan_mutation",
        "root_custody_lost",
    ],
)
def test_integrity_failure_aborts_every_unclaimed_cell(
    tmp_path: Path, code: str
) -> None:
    ctx = context(tmp_path / "campaign")
    fixture = FixtureServices(ctx)
    fixture.mutation = code
    fixture.process_code = "child_nonzero_exit"
    report = run_synthetic(ctx, fixture.services)
    assert fixture.started == [1, 2]
    assert report["cells"][1]["failure"]["code"] == code
    assert report["cells"][1]["state"] == "campaign_integrity_failed"
    assert report["status_counts"]["not_started"] == 13
    assert report["campaign_status"] == "aborted"
    for cell in report["cells"][2:]:
        assert cell["state"] == "not_started" and cell["claim"] is None
        assert not Path(cell["cell"]["claim_path"]).exists()


@pytest.mark.parametrize(
    "filename,code,started",
    [
        ("claimed.json", "claim_publication_failed", 0),
        ("completed.json", "evidence_publication_failed", 1),
        ("synthetic-report.json", "evidence_publication_failed", 15),
    ],
)
def test_publication_failure_closes_the_root_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    code: str,
    started: int,
) -> None:
    ctx = context(tmp_path / "campaign")
    fixture = FixtureServices(ctx)
    original = io.publish_json

    def fail(path: Path, value: object) -> None:
        if path.name == filename:
            raise OSError("synthetic publication failure")
        original(path, value)

    monkeypatch.setattr(io, "publish_json", fail)
    with pytest.raises(CampaignClosedError) as raised:
        run_synthetic(ctx, fixture.services)
    assert raised.value.code == code
    assert len(fixture.started) == started
    assert not (ctx.root / "synthetic-report.json").exists()
    with pytest.raises(ValueError, match="already exists"):
        run_synthetic(ctx, fixture.services)


def test_storage_rechecks_abort_before_consuming_next_claim(tmp_path: Path) -> None:
    ctx = context(tmp_path / "campaign")
    fixture = FixtureServices(ctx)
    budgets: list[int] = []

    def budget(root: Path, minimum: int) -> bool:
        budgets.append(minimum)
        return len(budgets) <= 3

    report = run_synthetic(ctx, replace(fixture.services, budget=budget))
    assert budgets == [5_368_709_120] * 4
    assert fixture.started == [1, 2]
    assert report["abort"]["code"] == "storage_budget_unavailable"
    assert report["status_counts"]["not_started"] == 13


def test_inventory_failure_leaves_a_consumed_interrupted_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = context(tmp_path / "campaign")
    fixture = FixtureServices(ctx)
    original = io.inventory

    def fail(root: Path) -> dict[str, Any]:
        if root == Path(ctx.plan["cells"][0]["output_path"]):
            raise OSError("synthetic unwalkable tree")
        return original(root)

    monkeypatch.setattr(io, "inventory", fail)
    with pytest.raises(CampaignClosedError, match="root_custody_lost"):
        run_synthetic(ctx, fixture.services)
    first = ctx.plan["cells"][0]
    assert Path(first["claim_path"]).exists()
    assert not Path(first["completed_receipt_path"]).exists()
    assert not Path(first["failed_receipt_path"]).exists()
    assert fixture.started == [1]
    assert not (ctx.root / "synthetic-report.json").exists()


def test_failed_partial_and_forbidden_artifacts_remain_in_detached_evidence(
    tmp_path: Path,
) -> None:
    ctx = context(tmp_path / "campaign")
    fixture = FixtureServices(ctx)
    fixture.process_code = "child_nonzero_exit"
    unopened = tmp_path / "outside.bin"
    unopened.write_bytes(b"external fixture; must not be followed")

    def process(cell: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
        result = fixture.process(cell)
        if cell["order"] == 2:
            output = Path(cell["output_path"])
            (output / "partial.bin").write_bytes(b"partial serialization")
            (output / "unexpected-link").symlink_to(unopened)
        return result

    report = run_synthetic(ctx, replace(fixture.services, process=process))
    failed = report["cells"][1]
    inventory = failed["artifact_inventory"]
    assert fixture.started == list(range(1, 16))
    assert failed["failure"]["code"] == "child_nonzero_exit"
    assert inventory["regular_file_count"] == 4
    assert inventory["forbidden_count"] == 1
    entry = next(e for e in inventory["entries"] if e["path"] == "unexpected-link")
    assert entry["node_type"] == "forbidden" and entry["sha256"] is None
    assert (
        verify_synthetic_report(ctx.root / "synthetic-report.json", checkpoint=admit)[
            "verified"
        ]
        is True
    )
    assert unopened.read_bytes() == b"external fixture; must not be followed"


@pytest.mark.parametrize("part", ["log", "artifact", "claim", "root"])
def test_prior_evidence_or_root_replacement_stops_subsequent_attempts(
    tmp_path: Path, part: str
) -> None:
    ctx = context(tmp_path / "campaign")
    fixture = FixtureServices(ctx)
    first = ctx.plan["cells"][0]

    def process(cell: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
        result = fixture.process(cell)
        if cell["order"] == 2:
            if part == "root":
                retained = tmp_path / "replaced-campaign"
                ctx.root.rename(retained)
                shutil.copytree(retained, ctx.root)
            else:
                target = {
                    "log": Path(first["stdout_path"]),
                    "artifact": Path(first["output_path"]) / "checkpoint/policy.zip",
                    "claim": Path(first["claim_path"]),
                }[part]
                with target.open("ab") as handle:
                    handle.write(b"mutation during later cell")
        return result

    report = run_synthetic(ctx, replace(fixture.services, process=process))
    assert fixture.started == [1, 2]
    assert report["cells"][1]["failure"]["code"] == "root_custody_lost"
    assert report["status_counts"]["not_started"] == 13
    assert report["result_claim"] is None
    assert all(not Path(cell["claim_path"]).exists() for cell in ctx.plan["cells"][2:])


def test_config_mutation_before_claim_cannot_start_any_child(
    tmp_path: Path,
) -> None:
    ctx = context(tmp_path / "campaign")
    fixture = FixtureServices(ctx)

    def identity() -> Any:
        Path(ctx.plan["cells"][-1]["config_path"]).write_bytes(b"{}")
        return ctx.identity

    report = run_synthetic(ctx, replace(fixture.services, identity=identity))
    assert fixture.started == []
    assert report["abort"]["code"] == "plan_mutation"
    assert report["status_counts"]["not_started"] == 15
    assert all(not Path(cell["claim_path"]).exists() for cell in ctx.plan["cells"])


@pytest.mark.parametrize("when", ["pre_child", "post_child", "admission"])
@pytest.mark.parametrize("part", ["root", "log", "attempt", "artifact"])
def test_undeclared_campaign_entries_abort_before_another_claim(
    tmp_path: Path, when: str, part: str
) -> None:
    ctx = context(tmp_path / "campaign")
    fixture = FixtureServices(ctx)
    calls = 0

    def mutate() -> None:
        target = {
            "root": ctx.root / "unexpected.bin",
            "log": ctx.root / "logs/unlisted.log",
            "attempt": Path(ctx.plan["cells"][-1]["claim_path"]),
            "artifact": Path(ctx.plan["cells"][-1]["output_path"]),
        }[part]
        target.write_bytes(b"unexpected custody entry")

    def identity() -> Any:
        nonlocal calls
        calls += 1
        if calls == 1 and when == "pre_child":
            mutate()
        return ctx.identity

    def process(cell: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
        result = fixture.process(cell)
        if when == "post_child":
            mutate()
        return result

    def checkpoint(config: Any, runtime: Any) -> dict[str, Any]:
        result = admit(config, runtime)
        if when == "admission":
            mutate()
        return result

    report = run_synthetic(
        ctx,
        replace(
            fixture.services, identity=identity, process=process, checkpoint=checkpoint
        ),
    )
    assert fixture.started == ([] if when == "pre_child" else [1])
    assert report["abort"]["code"] == "root_custody_lost"
    assert report["status_counts"]["not_started"] == (15 if when == "pre_child" else 14)
    assert report["result_claim"] is None
    assert not Path(ctx.plan["cells"][1]["claim_path"]).exists()

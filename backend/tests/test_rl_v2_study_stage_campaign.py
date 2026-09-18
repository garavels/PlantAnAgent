"""Stage state-machine integration with explicit fake physical/training inputs.

All 576 claims, report analysis and detached reconstruction use production code.
Physical projection and process execution are test substitutes; protected seeds
are identifiers only and no plant may be constructed or reset.
"""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from study_stage_fixtures import (
    authorize,
    training,
    patch_admission,
    process_record,
    rehash_file,
)
from plant_backend.rl.v2.study import (
    archive,
    registry,
    stage_arm,
    stage_authority,
    stage_campaign,
    stage_documents as documents,
    stage_lock,
    stage_state,
    stage_verifier,
    stage_launch,
)
from plant_backend.rl.v2.study.campaign_analysis import CellInput
from plant_backend.rl.v2.training import custody_io as io
from plant_backend.rl.v2.training.custody_json import authenticate, canonical_bytes
from test_rl_v2_study_campaign_analysis import inputs as arithmetic_inputs


@pytest.fixture(autouse=True)
def no_execution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from plant_backend.rl.v2 import beta_env

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("stage state-machine fixture reached physical execution")

    monkeypatch.setattr(beta_env.PlantCyclicMarketEnvV2Beta, "__init__", forbidden)
    monkeypatch.setattr(beta_env.PlantCyclicMarketEnvV2Beta, "reset", forbidden)
    monkeypatch.setattr(stage_launch, "_execute_arm", forbidden)
    monkeypatch.setattr(
        stage_lock, "lock_path", lambda: tmp_path.resolve() / "fixture.lock"
    )


def setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, failed: tuple[int, ...] = ()
) -> Any:
    trained = training(tmp_path.resolve() / "training", failed=failed)
    authorize(trained)
    patch_admission(monkeypatch, trained)
    monkeypatch.setattr(stage_campaign, "_budget_available", lambda *_: True)
    return stage_authority.admit_stage(trained.path.parent, "development")


def failed_child(context: Any, cell: dict[str, Any], claim: dict[str, Any]) -> Any:
    assert Path(cell["claim_path"]).is_file()
    for name in ("stdout_path", "stderr_path"):
        io.publish_bytes(Path(cell[name]), b"fixture failed process\n")
    return process_record(context.registry, cell, code=2), {"child_nonzero_exit"}


def install_arithmetic_arms(context: Any, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    originals = arithmetic_inputs.__wrapped__()["cells"]
    projected: dict[int, CellInput] = {}
    for cell, original, identity in zip(
        context.registry["cells"],
        originals,
        registry.cells_for_seeds(context.registry["episode_seeds"]),
        strict=True,
    ):
        arm, timing = deepcopy(original.arm), deepcopy(original.latency)
        assert arm is not None and timing is not None
        arm.update(stage=context.registry["stage"], seed=cell["episode_seed"])
        arm["operational"] = authenticate(
            {**arm["operational"], "seed": cell["episode_seed"]}
        )
        timing["host_sha256"] = context.authorization["latency_host"]["sha256"]
        projected[cell["order"]] = replace(
            original, identity=identity, arm=arm, latency=timing
        )
    calls: list[int] = []

    def process(ctx: Any, cell: dict[str, Any], claim: dict[str, Any]) -> Any:
        assert Path(cell["claim_path"]).read_bytes() == canonical_bytes(claim)
        assert calls == list(range(1, cell["order"]))
        assert all(
            not list(Path(later["directory"]).iterdir())
            for later in ctx.registry["cells"][cell["order"] :]
        )
        calls.append(cell["order"])
        for name in ("stdout_path", "stderr_path"):
            io.publish_bytes(Path(cell[name]), b"")
        for key in ("arm_path", "latency_path"):
            archive.publish_archive(
                Path(cell[key]),
                {"fixture": key, "order": cell["order"]},
                stored_limit=1000,
                uncompressed_limit=1000,
            )
        return process_record(ctx.registry, cell), set()

    def verify(ctx: Any, cell: dict[str, Any], claim: dict[str, Any]) -> dict[str, Any]:
        refs = []
        for key in ("arm_path", "latency_path"):
            raw, descriptor = archive.read_archive(
                Path(cell[key]), stored_limit=1000, uncompressed_limit=1000
            )
            if raw != {"fixture": key, "order": cell["order"]}:
                raise ValueError("explicit fixture projection does not match its input")
            refs.append(descriptor)
        value = projected[cell["order"]]
        return {
            "arm_archive": refs[0],
            "timing_archive": refs[1],
            "physical": value.arm,
            "latency": value.latency,
        }

    monkeypatch.setattr(stage_campaign, "_run_child", process)
    monkeypatch.setattr(stage_arm, "verify_cell", verify)
    return calls


def rewrite_report(context: Any, report: dict[str, Any]) -> None:
    path = Path(context.registry["report_path"])
    path.unlink()
    archive.publish_archive(
        path,
        authenticate(report),
        stored_limit=context.budget.report_stored_bytes,
        uncompressed_limit=context.budget.report_decoded_bytes,
    )


def test_all_576_sequential_claims_fresh_analysis_and_report_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = setup(tmp_path, monkeypatch)
    calls = install_arithmetic_arms(context, monkeypatch)
    with stage_lock.acquire() as lock:
        report, record = stage_campaign._collect(context, lock)
    assert calls == list(range(1, 577))
    assert report["analysis"]["state_counts"]["verified"] == 576
    assert (
        report["analysis"]["development_opening_requirements"][
            "all_calculated_requirements_met"
        ]
        is True
    )
    assert len(report["cells"]) == 576
    assert report["result_claim"] is None
    verified = stage_verifier.verify_stage(context.training.path.parent, "development")
    assert verified.report == report and verified.archive == record
    before_admissions = context.training.calls.count("fresh_training_admission")
    expected = stage_verifier.verdict(report, record)
    documents.publish(Path(context.registry["verification_path"]), expected)
    assert (
        stage_verifier.verify_stage(context.training.path.parent, "development").report
        == report
    )
    assert (
        context.training.calls.count("fresh_training_admission")
        == before_admissions + 1
    )
    # A cached success marker is neither sufficient nor a substitute for raw cells.
    verdict_path = Path(context.registry["verification_path"])
    rehash_file(
        verdict_path, lambda value: value.update(source_evidence_verified=False)
    )
    with pytest.raises(ValueError, match="verdict"):
        stage_verifier.verify_stage(context.training.path.parent, "development")
    verdict_path.unlink()
    for mutate in (
        lambda value: value.update(result_claim="unearned result"),
        lambda value: value["analysis"]["state_counts"].update(verified=575),
        lambda value: value["analysis"]["frozen_champions"].update(
            {registry.BETA_CORE_ALGORITHM_IDS[0]: 211}
        ),
        lambda value: value["cells"].pop(),
    ):
        changed = deepcopy(report)
        mutate(changed)
        rewrite_report(context, changed)
        with pytest.raises(ValueError, match="complete registered stage report"):
            stage_verifier.verify_stage(context.training.path.parent, "development")
    rewrite_report(context, report)
    arm_path = Path(context.registry["cells"][0]["arm_path"])
    original = arm_path.read_bytes()
    arm_path.unlink()
    archive.publish_archive(
        arm_path,
        {"fixture": "arm_path", "order": 2},
        stored_limit=1000,
        uncompressed_limit=1000,
    )
    with pytest.raises(ValueError, match="terminal"):
        stage_verifier.verify_stage(context.training.path.parent, "development")
    arm_path.write_bytes(original)
    with pytest.raises(ValueError, match="changed after reconstruction"):
        verified.recheck()
    with stage_lock.acquire() as lock:
        with pytest.raises(FileExistsError):
            stage_campaign._collect(context, lock)
    assert calls == list(range(1, 577))


def test_failed_training_slots_and_processes_remain_all_576(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = setup(tmp_path, monkeypatch, failed=tuple(range(1, 16)))
    started: list[int] = []

    def process(ctx: Any, cell: dict[str, Any], claim: dict[str, Any]) -> Any:
        started.append(cell["order"])
        return failed_child(ctx, cell, claim)

    monkeypatch.setattr(stage_campaign, "_run_child", process)
    with stage_lock.acquire() as lock:
        report, _ = stage_campaign._collect(context, lock)
    assert started == [
        cell["order"]
        for cell in context.registry["cells"]
        if cell["actor"]["arm"] != "PPO"
    ]
    assert report["analysis"]["state_counts"]["training_not_admitted"] == 480
    assert report["analysis"]["state_counts"]["process_failed"] == 96
    assert all(row["claim"] is not None for row in report["cells"])
    assert (
        report["analysis"]["development_opening_requirements"][
            "all_calculated_requirements_met"
        ]
        is False
    )
    assert (
        stage_verifier.verify_stage(context.training.path.parent, "development").report
        == report
    )


@pytest.mark.parametrize(
    "defect",
    ["storage", "interrupt", "child_interrupt", "claim_replaced", "later_consumed"],
)
def test_first_failure_consumes_no_later_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    context = setup(tmp_path, monkeypatch)
    started: list[int] = []

    def process(ctx: Any, cell: dict[str, Any], claim: dict[str, Any]) -> Any:
        started.append(cell["order"])
        result, failures = failed_child(ctx, cell, claim)
        if defect == "child_interrupt":
            failures.add("root_custody_lost")
        elif defect == "claim_replaced":
            path = Path(cell["claim_path"])
            replacement = path.with_suffix(".replacement")
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
        elif defect == "later_consumed":
            Path(ctx.registry["cells"][2]["claim_path"]).write_bytes(b"undeclared")
        return result, failures

    monkeypatch.setattr(stage_campaign, "_run_child", process)
    checks = 0

    def available(*_: Any) -> bool:
        nonlocal checks
        checks += 1
        if checks == 2:
            if defect == "interrupt":
                raise KeyboardInterrupt
            return False
        return True

    if defect in {"storage", "interrupt"}:
        monkeypatch.setattr(stage_campaign, "_budget_available", available)
    with stage_lock.acquire() as lock:
        if defect in {"claim_replaced", "later_consumed"}:
            with pytest.raises(ValueError):
                stage_campaign._collect(context, lock)
            assert not Path(context.registry["report_path"]).exists()
        else:
            report, _ = stage_campaign._collect(context, lock)
            expected = {
                "storage": "storage_budget_unavailable",
                "interrupt": "operator_interrupted",
                "child_interrupt": "stage_integrity_lost",
            }[defect]
            assert report["abort"] == {"code": expected, "before_order": 2}
            assert report["campaign_status"] == "aborted"
            assert report["result_claim"] is None
            assert (
                stage_verifier.verify_stage(
                    context.training.path.parent, "development"
                ).report
                == report
            )
    assert started == [1]
    assert not Path(context.registry["cells"][1]["claim_path"]).exists()


@pytest.mark.parametrize(
    "defect",
    [
        "missing_directory",
        "extra_root",
        "symlink_cell",
        "claim_without_terminal",
        "terminal_before_claim",
        "gap",
    ],
)
def test_layout_and_sequential_history_rejected_before_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    from study_stage_fixtures import opening_files

    context = setup(tmp_path, monkeypatch)
    with stage_lock.acquire() as lock:
        opened = opening_files(context, lock)
    cell = context.registry["cells"][0]
    if defect == "missing_directory":
        Path(cell["directory"]).rmdir()
    elif defect == "extra_root":
        (context.root / "extra").write_text("extra")
    elif defect == "symlink_cell":
        directory = Path(cell["directory"])
        directory.rmdir()
        directory.symlink_to(
            Path(context.registry["cells"][1]["directory"]), target_is_directory=True
        )
    else:
        if defect == "gap":
            cell = context.registry["cells"][1]
        claim = documents.claim(
            context.registry, opened, cell, None, timestamp=documents.now()
        )
        documents.publish(Path(cell["claim_path"]), claim, cell=True)
        if defect == "terminal_before_claim":
            process, _ = failed_child(context, cell, claim)
            rebuilt, _ = stage_state.classify_cell(context, cell, claim, process)
            documents.publish(
                Path(cell["terminal_path"]),
                authenticate(
                    {
                        "schema": documents.TERMINAL_SCHEMA,
                        "claim_sha256": claim["sha256"],
                        "cell": cell,
                        **rebuilt,
                        "finished_at_utc": "2026-09-01T00:00:00.000000Z",
                    }
                ),
                cell=True,
            )
    with pytest.raises((ValueError, FileNotFoundError)):
        stage_state.check_layout(context, closed=False)
        stage_state.read_cells(context, opened)


def test_detached_verifier_is_attempted_after_publication_without_local_recapture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(stage_campaign, "admit_stage", lambda *_: context)
    monkeypatch.setattr(io, "require_new_root", lambda *_: None)
    events: list[str] = []

    def collect(*_: Any) -> Any:
        events.append("report_published")
        context.training.recheck = lambda: (_ for _ in ()).throw(
            ValueError("identity lost")
        )
        return {"fixture_report": True}, {"fixture_record": True}

    def detached(*_: Any) -> Any:
        events.append("detached_attempted")
        raise ValueError("detached rejected mutated identity")

    monkeypatch.setattr(stage_campaign, "_collect", collect)
    monkeypatch.setattr(stage_campaign, "_detached", detached)
    with pytest.raises(ValueError, match="detached rejected"):
        stage_campaign.run_stage(
            context.training.path.parent,
            "development",
            confirm_operator_assertions=True,
        )
    assert events == ["report_published", "detached_attempted"]


@pytest.mark.parametrize("fails", [False, True])
def test_detached_process_keeps_bounded_logs_and_exclusive_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fails: bool,
) -> None:
    from study_stage_fixtures import opening_files
    import sys

    context = setup(tmp_path, monkeypatch)
    with stage_lock.acquire() as lock:
        opening_files(context, lock)
    report = {
        "sha256": "a" * 64,
        "stage": "development",
        "campaign_status": "completed",
        "analysis": {"state_counts": {"verified": 576}, "frozen_champions": {}},
        "result_claim": None,
    }
    record = {"fixture_archive": True}
    expected = stage_verifier.verdict(report, record)
    calls = []

    def capture(command: list[str], **kwargs: Any) -> Any:
        calls.append(command)
        assert command == [
            sys.executable,
            "-m",
            "plant_backend.rl.v2.study.stage_cli",
            "verify",
            "--training-root",
            str(context.training.path.parent),
            "--stage",
            "development",
            "--publish",
        ]
        assert kwargs["timeout_seconds"] == 86_400
        assert kwargs["log_limit_bytes"] == context.budget.log_limit_bytes
        assert "launch_permission" not in kwargs
        for name in ("stdout", "stderr"):
            assert kwargs[name] == Path(
                context.registry["verification_" + name + "_path"]
            )
            io.publish_bytes(
                kwargs[name], b"fixture verifier failure" if fails else b""
            )
        if not fails:
            documents.publish(Path(context.registry["verification_path"]), expected)
        process = process_record(
            context.registry, context.registry["cells"][0], code=2 if fails else 0
        )
        process.update(command=command, timeout_seconds=86_400)
        return process, {"child_nonzero_exit"} if fails else set()

    monkeypatch.setattr(stage_campaign, "capture_process", capture)
    if fails:
        with pytest.raises(ValueError, match="detached stage verification failed"):
            stage_campaign._detached(context, report, record)
    else:
        assert stage_campaign._detached(context, report, record) == expected
    assert len(calls) == 1
    from plant_backend.rl.v2.training.custody_json import decode

    receipt = decode(Path(context.registry["verification_process_path"]).read_bytes())
    assert receipt["failure_codes"] == (["child_nonzero_exit"] if fails else [])
    assert receipt["report_archive"] == record
    assert set(receipt["logs"]) == {"stdout", "stderr"}
    assert receipt["process"]["exit_code"] == (2 if fails else 0)

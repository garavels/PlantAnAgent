"""Real pipe/process/lock checks and reset-free child admission ordering."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from custody_fixtures import fixture_runtime
from study_stage_fixtures import (
    authorize,
    training,
    patch_admission,
    opening_files,
    process_record,
)
from plant_backend.rl.v2.study import (
    stage_launch as launch,
    stage_documents as documents,
)
from plant_backend.rl.v2.study import stage_lock, stage_authority, stage_arm
from plant_backend.rl.v2.training import custody_io as io
from plant_backend.rl.v2.training.custody_json import (
    authenticate,
    canonical_bytes,
    decode,
)
from plant_backend.rl.v2.training.custody_process import capture_process


@pytest.fixture(autouse=True)
def no_real_execution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from plant_backend.rl.v2 import beta_env
    from plant_backend.rl.v2.training import custody_checkpoint

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("launch fixture attempted a real model or plant")

    monkeypatch.setattr(beta_env.PlantCyclicMarketEnvV2Beta, "__init__", forbidden)
    monkeypatch.setattr(beta_env.PlantCyclicMarketEnvV2Beta, "reset", forbidden)
    monkeypatch.setattr(custody_checkpoint, "load_policy", forbidden)
    monkeypatch.setattr(launch, "_execute_arm", forbidden)
    monkeypatch.setattr(
        stage_lock, "lock_path", lambda: tmp_path.resolve() / "fixture.lock"
    )


def payload(**changes: Any) -> bytes:
    return canonical_bytes(
        {
            "schema": documents.LAUNCH_SCHEMA,
            "parent_pid": os.getppid(),
            "registry_sha256": "a" * 64,
            "claim_sha256": "b" * 64,
            "authorization_sha256": "c" * 64,
            **changes,
        }
    )


def consume(raw: bytes, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    descriptor, writer = os.pipe()
    try:
        os.write(writer, raw)
    finally:
        os.close(writer)
    monkeypatch.setenv(launch.LAUNCH_FD_ENV, str(descriptor))
    try:
        return launch.consume_permission()
    finally:
        assert launch.LAUNCH_FD_ENV not in os.environ
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "training_schema",
        "parent",
        "extra",
        "oversized",
        "noncanonical",
        "duplicate",
    ],
)
def test_one_use_anonymous_pipe_rejects_replay_and_cross_schema(
    monkeypatch: pytest.MonkeyPatch, defect: str | None
) -> None:
    raw = payload()
    if defect == "training_schema":
        raw = payload(
            schema="plant-beta-constrained-rl-registered-training-child-launch-v1"
        )
    elif defect == "parent":
        raw = payload(parent_pid=os.getppid() + 1)
    elif defect == "extra":
        raw = payload(extra=True)
    elif defect == "oversized":
        raw = payload(claim_sha256="b" * 1100)
    elif defect == "noncanonical":
        raw = b" " + raw
    elif defect == "duplicate":
        raw = raw.replace(b"{", b'{"schema":"duplicated",', 1)
    if defect is None:
        assert consume(raw, monkeypatch)["schema"] == documents.LAUNCH_SCHEMA
        with pytest.raises(ValueError, match="live admitted parent"):
            launch.consume_permission()
    else:
        with pytest.raises(ValueError):
            consume(raw, monkeypatch)


def test_regular_file_cannot_supply_launch_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "permission.json"
    path.write_bytes(payload())
    descriptor = os.open(path, os.O_RDONLY)
    monkeypatch.setenv(launch.LAUNCH_FD_ENV, str(descriptor))
    with pytest.raises(ValueError, match="anonymous pipe"):
        launch.consume_permission()
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_actual_child_consumes_only_its_live_parent_pipe(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "from plant_backend.rl.v2.study.stage_launch import consume_permission; v=consume_permission(); assert v['schema'].endswith('stage-child-launch-v1');\ntry: consume_permission()\nexcept ValueError: pass\nelse: raise AssertionError('replayed')\n",
    ]
    permission = payload(parent_pid=os.getpid())
    process, failures = capture_process(
        command,
        stdout=tmp_path / "stdout.log",
        stderr=tmp_path / "stderr.log",
        timeout_seconds=30,
        log_limit_bytes=4096,
        launch_permission=permission,
    )
    assert failures == set(), (process, (tmp_path / "stderr.log").read_text())
    assert process["started"] is True and process["exit_code"] == 0


def lock_child(identity: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    code = "import json,sys; from pathlib import Path; from plant_backend.rl.v2.study import stage_lock; identity=json.loads(sys.argv[1]); stage_lock.lock_path=lambda:Path(identity['path']); stage_lock.require_locked(identity)"
    return subprocess.run(
        [sys.executable, "-c", code, json.dumps(identity)],
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_lock_is_live_cross_process_and_replacement_is_detected(tmp_path: Path) -> None:
    with stage_lock.acquire() as identity:
        assert lock_child(identity).returncode == 0
        with pytest.raises(BlockingIOError):
            with stage_lock.acquire():
                pytest.fail("second parent acquired live lock")
    assert lock_child(identity).returncode != 0
    replacement = tmp_path.resolve() / "replacement.lock"
    replacement.write_bytes(b"")
    replacement.replace(Path(identity["path"]))
    with pytest.raises(ValueError, match="identity"):
        stage_lock.verify_lock_identity(identity)


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "unadmitted",
        "prior_integrity",
        "later_attempt",
        "symlink_cell",
        "changed_authority",
        "changed_training",
        "changed_claim",
        "runtime",
        "host",
    ],
)
def test_child_checkpoint_thread_and_reset_ordering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str | None,
) -> None:
    trained = training(
        tmp_path.resolve() / "training", failed=(1,) if defect == "unadmitted" else ()
    )
    if defect != "unadmitted":
        trained.report["cells"][0]["checkpoint"]["path"] = str(
            trained.path.parent / "fixture-checkpoint"
        )
        trained.report = authenticate(trained.report)
        trained.path.write_bytes(canonical_bytes(trained.report))
        trained.evidence = authenticate(
            {
                **trained.evidence,
                "training_report": documents.reference(trained.path, trained.report),
            }
        )
    authorize(trained)
    patch_admission(monkeypatch, trained)
    context = stage_authority.admit_stage(trained.path.parent, "development")
    cell = context.registry["cells"][3]
    events: list[str] = []
    with stage_lock.acquire() as identity:
        opened = opening_files(context, identity)
        opened = authenticate({**opened, "parent_pid": os.getppid()})
        Path(context.registry["opening_claim_path"]).write_bytes(
            canonical_bytes(opened)
        )
        previous = None
        for prior in context.registry["cells"][:3]:
            claim = documents.claim(
                context.registry, opened, prior, previous, timestamp=documents.now()
            )
            documents.publish(Path(prior["claim_path"]), claim, cell=True)
            # The parent has independently verified these terminals. The child
            # checks the immediately previous bound terminal without trusting a
            # free-standing attempt number.
            previous = authenticate(
                {
                    "schema": documents.TERMINAL_SCHEMA,
                    "claim_sha256": claim["sha256"],
                    "cell": prior,
                    "process": process_record(context.registry, prior, code=2),
                    "logs": None,
                    "inventory": None,
                    "state": "process_failed",
                    "failure": (
                        "stage_integrity_lost"
                        if defect == "prior_integrity"
                        else "child_nonzero_exit"
                    ),
                    "arm_archive": None,
                    "timing_archive": None,
                    "finished_at_utc": documents.now(),
                }
            )
            documents.publish(Path(prior["terminal_path"]), previous, cell=True)
        claim = documents.claim(
            context.registry, opened, cell, previous, timestamp=documents.now()
        )
        documents.publish(Path(cell["claim_path"]), claim, cell=True)
        for name in ("stdout_path", "stderr_path"):
            io.publish_bytes(Path(cell[name]), b"")
        if defect == "later_attempt":
            Path(context.registry["cells"][4]["claim_path"]).write_bytes(b"used")
        elif defect == "symlink_cell":
            path = Path(context.registry["cells"][4]["directory"])
            path.rmdir()
            path.symlink_to(
                Path(context.registry["cells"][5]["directory"]),
                target_is_directory=True,
            )
        permission = decode(
            launch.permission(context.registry, claim, context.authorization)
        )
        permission["parent_pid"] = os.getppid()
        monkeypatch.setattr(launch, "consume_permission", lambda: permission)
        monkeypatch.setattr(
            launch.training_verifier,
            "_read_report",
            lambda *_, **__: deepcopy(trained.report),
        )
        runtime = fixture_runtime()
        monkeypatch.setattr(
            launch,
            "capture",
            lambda **_: (
                runtime if defect != "runtime" else replace(runtime, git_head="d" * 40)
            ),
        )

        def checkpoint(*_: Any) -> Any:
            events.append("own_checkpoint_admission")
            if defect in {"changed_authority", "changed_training", "changed_claim"}:
                path = {
                    "changed_authority": Path(
                        context.registry["authorization_input_path"]
                    ),
                    "changed_training": trained.path,
                    "changed_claim": Path(cell["claim_path"]),
                }[defect]
                replacement = path.with_suffix(".replacement")
                replacement.write_bytes(path.read_bytes())
                replacement.replace(path)
            return trained.report["cells"][0]["checkpoint"]

        monkeypatch.setattr(launch, "admit_checkpoint", checkpoint)
        monkeypatch.setattr(
            launch, "configure_latency_threads", lambda: events.append("threads_1_1")
        )
        monkeypatch.setattr(
            launch,
            "capture_latency_host",
            lambda: {} if defect == "host" else context.authorization["latency_host"],
        )

        def execute(config: Any, **kwargs: Any) -> str:
            assert events == ["own_checkpoint_admission", "threads_1_1"]
            assert kwargs["expected_training_runtime"] == runtime
            assert kwargs["seed"] == cell["episode_seed"]
            events.append("single_arm_test_substitute")
            return "no physical execution"

        monkeypatch.setattr(launch, "_execute_arm", execute)
        monkeypatch.setattr(
            stage_arm, "publish_arm", lambda *args: events.append("publish")
        )
        if defect is None:
            launch.run_child(trained.path.parent, "development", 4)
            assert events == [
                "own_checkpoint_admission",
                "threads_1_1",
                "single_arm_test_substitute",
                "publish",
            ]
        else:
            with pytest.raises((ValueError, RuntimeError)):
                launch.run_child(trained.path.parent, "development", 4)
            assert "single_arm_test_substitute" not in events

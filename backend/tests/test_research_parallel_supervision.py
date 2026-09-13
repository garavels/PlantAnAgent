"""Tiny fabricated commands only: these tests never import or run learning."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Callable

import pytest

from plant_backend.rl.v2.training.custody_json import canonical_bytes
from plant_backend.rl.v2.training.research_parallel_supervision import (
    ParallelAttempt,
    run_training_wave,
)


WORKER = """
import json, os, signal, subprocess, sys, threading, time
from pathlib import Path
from plant_backend.rl.v2.training.custody_io import publish_json
from plant_backend.rl.v2.training.custody_process import capture_process

def closed_child(stage, name, timeout, extra):
    spec = json.loads(Path(extra[0]).read_text())
    root = Path(spec["root"])
    # Frozen-source checks can themselves spawn Git without interception.
    subprocess.run([sys.executable, "-c", "pass"], check=True)
    behavior = spec.get("behavior", "normal")
    if behavior == "wrong_command":
        spec["command"] = [sys.executable, "-c", "raise SystemExit(99)"]
    if behavior == "spawn_failure":
        os.environ["PLANT_FIXTURE_OVERSIZED_ENVIRONMENT"] = "x" * (os.sysconf("SC_ARG_MAX") + 1)
    if behavior == "startup_sleep":
        time.sleep(30)
    if behavior == "corrupt_completion":
        (root / "supervisors" / (name + ".supervisor.json")).write_text("{corrupt")
    if behavior == "supervisor_logs":
        os.write(1, b"sealed supervisor content" * 65536)
    if behavior == "missing_evidence":
        os._exit(0)
    if behavior == "killed_supervisor":
        def kill_later():
            while not (root / (name + ".ready")).exists():
                time.sleep(.005)
            os.kill(os.getpid(), signal.SIGKILL)
        threading.Thread(target=kill_later, daemon=True).start()
    process, codes = capture_process(
        spec["command"], stdout=root / (name + ".stdout.log"),
        stderr=root / (name + ".stderr.log"), timeout_seconds=timeout,
        log_limit_bytes=spec["log_limit"],
    )
    receipt = {"stage": stage, "name": name, "process": process,
               "codes": sorted(codes), "closed_at": "fixture closed"}
    if behavior == "wrong_identity":
        receipt["name"] = "wrong"
    if behavior == "corrupt_receipt":
        (root / (name + ".json")).write_text("{corrupt")
    else:
        publish_json(root / (name + ".json"), receipt)
    if "root_custody_lost" in codes or process["exit_code"] == 86:
        raise RuntimeError("fixture integrity loss")
    return receipt
"""


@pytest.fixture
def fixture_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path.resolve()
    (root / "supervisors").mkdir()
    (root / "fixture_parallel_worker.py").write_text(WORKER)
    monkeypatch.setenv(
        "PYTHONPATH", str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    return root


def attempt(
    root: Path,
    name: str,
    code: str,
    *,
    behavior: str = "normal",
    timeout: int = 5,
    log_limit: int = 4096,
) -> ParallelAttempt:
    command = (sys.executable, "-c", code)
    spec = root / (name + ".fixture.json")
    spec.write_text(
        json.dumps(
            {
                "root": str(root),
                "command": command,
                "behavior": behavior,
                "log_limit": log_limit,
            }
        )
    )
    return ParallelAttempt(
        name=name,
        stage="_train",
        timeout_seconds=timeout,
        extra=(str(spec),),
        command=command,
        stdout=root / (name + ".stdout.log"),
        stderr=root / (name + ".stderr.log"),
        receipt=root / (name + ".json"),
    )


def wave(
    root: Path, arms: list[ParallelAttempt], check: Callable[[], None] = lambda: None
) -> dict[str, Any]:
    return run_training_wave(
        arms,
        supervisor_directory=root / "supervisors",
        worker_module="fixture_parallel_worker",
        check_integrity=check,
        minimum_free_bytes=0,
        supervisor_log_limit_bytes=4096,
        poll_seconds=0.02,
        integrity_check_interval_seconds=0.02,
    )


def pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def assert_groups_closed(result: dict[str, Any]) -> None:
    for supervisor in result["supervisors"]:
        assert supervisor["supervisor_group_closed"] is True
        assert not pid_exists(supervisor["supervisor_pid"])
        if supervisor["trainer"] is not None:
            assert supervisor["trainer_group_closed"] is True
            with pytest.raises(ProcessLookupError):
                os.killpg(supervisor["trainer"]["pgid"], 0)


def waiting_code(root: Path, name: str) -> str:
    return (
        "import os,time; from pathlib import Path; "
        f"Path({str(root / (name + '.ready'))!r}).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )


def test_two_arms_overlap_and_keep_exact_distinct_identities(
    fixture_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = fixture_root
    code = (
        "import time; from pathlib import Path; "
        f"p=Path({str(root)!r}); "
        "(p / '{name}.ready').write_text('ready'); "
        "deadline=time.monotonic()+3\n"
        "while not (p / '{peer}.ready').exists():\n"
        " assert time.monotonic()<deadline\n"
        " time.sleep(.005)\n"
        "print('sealed tiny fixture')"
    )
    arms = [
        attempt(root, name, code.format(name=name, peer=peer))
        for name, peer in (("A", "B"), ("B", "A"))
    ]
    result = wave(root, arms)
    assert result["fatal"] is None
    assert [r["name"] for r in result["receipts"]] == ["A", "B"]
    assert all(r["codes"] == [] for r in result["receipts"])
    assert len({r["supervisor_pid"] for r in result["supervisors"]}) == 2
    for arm, supervisor in zip(arms, result["supervisors"]):
        assert supervisor["trainer"]["command"] == list(arm.command)
        assert supervisor["trainer"]["pid"] != supervisor["supervisor_pid"]
        assert supervisor["evidence"]["ordinary_receipt"]["path"] == str(arm.receipt)
        assert supervisor["files"]["trainer"]["sha256"]
        assert supervisor["files"]["completion"]["sha256"]
    assert_groups_closed(result)
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("kind", ["numerical", "timeout", "child_logs"])
def test_ordinary_failure_leaves_peer_running(fixture_root: Path, kind: str) -> None:
    root = fixture_root
    codes = {
        "numerical": "raise SystemExit(3)",
        "timeout": "import time; time.sleep(30)",
        "child_logs": "import os; os.write(1,b'x'*8192)",
    }
    a = attempt(root, "A", codes[kind], timeout=1)
    b = attempt(root, "B", "import time; time.sleep(1.2); print('healthy')")
    result = wave(root, [a, b])
    assert result["fatal"] is None
    assert result["receipts"][0]["codes"]
    assert result["receipts"][1]["codes"] == []
    assert a.stdout.stat().st_size <= 4096
    assert_groups_closed(result)


@pytest.mark.parametrize(
    "kind",
    [
        "integrity_exit",
        "missing_evidence",
        "corrupt_receipt",
        "corrupt_completion",
        "wrong_identity",
        "supervisor_logs",
        "killed_supervisor",
        "wrong_command",
    ],
)
def test_fatal_supervision_loss_cancels_healthy_peer(
    fixture_root: Path, kind: str
) -> None:
    root = fixture_root
    code = (
        "import time; time.sleep(.2); raise SystemExit(86)"
        if kind == "integrity_exit"
        else waiting_code(root, "A") if kind == "killed_supervisor" else "pass"
    )
    a = attempt(
        root, "A", code, behavior="normal" if kind == "integrity_exit" else kind
    )
    b = attempt(root, "B", waiting_code(root, "B"))
    started = time.monotonic()
    result = wave(root, [a, b])
    assert result["fatal"] is not None
    assert time.monotonic() - started < 12
    assert_groups_closed(result)
    assert all(
        path.stat().st_size <= 4096 for path in (root / "supervisors").glob("*.log")
    )


@pytest.mark.parametrize("kind", ["source", "request", "storage", "operator"])
def test_live_integrity_checks_abort_both(
    fixture_root: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    root = fixture_root
    arms = [attempt(root, name, waiting_code(root, name)) for name in ("A", "B")]
    calls = 0

    def check() -> None:
        nonlocal calls
        calls += 1
        if not all((root / (name + ".ready")).exists() for name in ("A", "B")):
            return
        if kind == "source":
            raise ValueError("fixture source changed")
        if kind == "operator":
            raise KeyboardInterrupt
        if kind == "request":
            path = root / "supervisors" / "A.request.json"
            value = json.loads(path.read_text())
            value["worker_function"] = "changed"
            path.write_bytes(canonical_bytes(value))
        if kind == "storage":
            from types import SimpleNamespace

            monkeypatch.setattr(
                "plant_backend.rl.v2.training.research_parallel_supervision.shutil.disk_usage",
                lambda _: SimpleNamespace(free=-1),
            )

    result = wave(root, arms, check)
    assert result["fatal"] is not None
    assert calls > 3
    assert_groups_closed(result)


@pytest.mark.parametrize("fatal", [False, True])
def test_descendants_are_closed_before_return(fixture_root: Path, fatal: bool) -> None:
    root = fixture_root
    descendant = root / "descendant.pid"
    code = (
        "import os,time; from pathlib import Path; "
        "pid=os.fork(); "
        f"Path({str(descendant)!r}).write_text(str(os.getpid())) if pid==0 else None; "
        "time.sleep(30)"
    )

    def check() -> None:
        if fatal and descendant.exists():
            raise ValueError("fixture source integrity loss with descendant")

    result = wave(
        root, [attempt(root, "A", code, timeout=1), attempt(root, "B", "pass")], check
    )
    assert (result["fatal"] is not None) is fatal
    assert not pid_exists(int(descendant.read_text()))
    assert_groups_closed(result)


def test_ordinary_spawn_failure_has_no_trainer_identity_and_no_retry(
    fixture_root: Path,
) -> None:
    root = fixture_root
    result = wave(
        root,
        [
            attempt(root, "A", "pass", behavior="spawn_failure"),
            attempt(root, "B", "pass"),
        ],
    )
    assert result["fatal"] is None
    assert result["receipts"][0]["codes"] == ["child_spawn_failed"]
    assert result["receipts"][0]["process"]["started"] is False
    assert result["supervisors"][0]["trainer"] is None
    assert result["receipts"][1]["codes"] == []
    assert_groups_closed(result)


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGKILL])
def test_parent_signal_closes_supervisors_and_trainers(
    fixture_root: Path, signum: int
) -> None:
    root = fixture_root
    arms = [attempt(root, name, waiting_code(root, name)) for name in ("A", "B")]
    record = root / "parent-result.json"
    script = (
        "import json; from pathlib import Path; "
        "from plant_backend.rl.v2.training.research_parallel_supervision import ParallelAttempt,run_training_wave; "
        f"specs=json.loads({json.dumps([a.record() for a in arms])!r}); "
        "arms=[ParallelAttempt(**{**a,'extra':tuple(a['extra']),'command':tuple(a['command']),"
        "'stdout':Path(a['stdout']),'stderr':Path(a['stderr']),'receipt':Path(a['receipt'])}) for a in specs]; "
        f"result=run_training_wave(arms,supervisor_directory=Path({str(root / 'supervisors')!r}),"
        "worker_module='fixture_parallel_worker',check_integrity=lambda:None,"
        "minimum_free_bytes=0,supervisor_log_limit_bytes=4096,poll_seconds=.02); "
        f"Path({str(record)!r}).write_text(json.dumps(result))"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not all((root / (name + ".ready")).exists() for name in ("A", "B")):
            assert parent.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        trainers = [
            json.loads((root / "supervisors" / (name + ".trainer.json")).read_text())
            for name in ("A", "B")
        ]
        os.kill(parent.pid, signum)
        parent.communicate(timeout=20)
        deadline = time.monotonic() + 10
        while (
            any(
                pid_exists(t["pid"]) or pid_exists(t["supervisor_pid"])
                for t in trainers
            )
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        for trainer in trainers:
            assert not pid_exists(trainer["pid"])
            assert not pid_exists(trainer["supervisor_pid"])
        if signum == signal.SIGTERM:
            assert parent.returncode == 0
            assert (
                json.loads(record.read_text())["fatal"]["code"]
                == "operator_interruption"
            )
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()


def test_nonretry_and_exclusive_paths(fixture_root: Path) -> None:
    root = fixture_root
    arms = [attempt(root, name, "pass") for name in ("A", "B")]
    arms[0].stdout.write_bytes(b"preexisting")
    with pytest.raises(ValueError, match="already exist"):
        wave(root, arms)
    assert not list((root / "supervisors").iterdir())
    arms[0].stdout.unlink()
    result = wave(root, arms)
    assert result["fatal"] is None
    with pytest.raises(ValueError, match="admission"):
        wave(root, arms)


def test_exec_barrier_timeout_prevents_unacknowledged_command_and_restores_binding(
    fixture_root: Path,
) -> None:
    from plant_backend.rl.v2.training.research_parallel_supervision import (
        MODULE,
        SCHEMA,
    )

    root = fixture_root
    forbidden = root / "unacknowledged-command-ran"
    arm = attempt(
        root, "A", f"from pathlib import Path; Path({str(forbidden)!r}).touch()"
    )
    request_path = root / "supervisors" / "A.request.json"
    request_path.write_bytes(
        canonical_bytes(
            {
                "schema": SCHEMA,
                "attempt": arm.record(),
                "worker_module": "fixture_parallel_worker",
                "worker_function": "closed_child",
                "parent_pid": os.getpid(),
            }
        )
    )
    parent_control, supervisor_control = socket.socketpair()
    script = (
        "import sys; from pathlib import Path; "
        "from plant_backend.rl.v2.training import custody_process; "
        f"from {MODULE} import _supervise; "
        "original=custody_process.subprocess; "
        f"result=_supervise(Path({str(request_path)!r}), {supervisor_control.fileno()}); "
        "assert custody_process.subprocess is original; raise SystemExit(result)"
    )
    supervisor = subprocess.Popen(
        [sys.executable, "-c", script],
        pass_fds=(supervisor_control.fileno(),),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    supervisor_control.close()
    try:
        parent_control.settimeout(10)
        raw = b""
        while b"\n" not in raw:
            raw += parent_control.recv(65536)
        trainer = json.loads(raw)
        supervisor.communicate(timeout=12)
        assert supervisor.returncode == 86
        assert not forbidden.exists()
        assert not pid_exists(trainer["pid"])
        with pytest.raises(ProcessLookupError):
            os.killpg(trainer["pgid"], 0)
        completion = json.loads(
            (root / "supervisors" / "A.supervisor.json").read_text()
        )
        assert completion["fatal"] == "RuntimeError"
    finally:
        parent_control.close()
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait()


def test_slow_full_checks_do_not_delay_barriers_or_run_at_process_poll_cadence(
    fixture_root: Path,
) -> None:
    root = fixture_root
    arms = [attempt(root, name, "import time; time.sleep(.5)") for name in ("A", "B")]
    checks: list[float] = []

    def full_check() -> dict[str, bool]:
        checks.append(time.monotonic())
        if len(checks) == 2:
            assert all(
                (root / "supervisors" / (name + ".trainer.json")).exists()
                for name in ("A", "B")
            )
        time.sleep(0.08)
        return {"checked": True}

    result = run_training_wave(
        arms,
        supervisor_directory=root / "supervisors",
        worker_module="fixture_parallel_worker",
        check_integrity=full_check,
        minimum_free_bytes=0,
        supervisor_log_limit_bytes=4096,
        poll_seconds=0.01,
        integrity_check_interval_seconds=30.0,
    )
    assert result["fatal"] is None
    assert len(checks) == 3  # Prelaunch, after acknowledged launches, final closure.
    admission = json.loads((root / "supervisors" / "admission.json").read_text())
    assert admission["poll_seconds"] == 0.01
    assert admission["integrity_check_interval_seconds"] == 30.0
    assert_groups_closed(result)


def test_startup_bound_cancels_without_blocking_launch_acknowledgements(
    fixture_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = fixture_root
    monkeypatch.setattr(
        "plant_backend.rl.v2.training.research_parallel_supervision._STARTUP_SECONDS",
        0.4,
    )
    checks = 0

    def check() -> None:
        nonlocal checks
        checks += 1

    result = wave(
        root,
        [
            attempt(root, "A", "pass", behavior="startup_sleep"),
            attempt(root, "B", waiting_code(root, "B")),
        ],
        check,
    )
    assert result["fatal"] is not None
    assert checks == 1
    assert not (root / "supervisors" / "A.trainer.json").exists()
    assert_groups_closed(result)


def test_unresolved_group_closure_never_publishes_a_final_file_inventory(
    fixture_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plant_backend.rl.v2.training import research_parallel_supervision as parallel

    root = fixture_root
    closure_failed = False
    late_inventory: list[Path] = []
    original = getattr(parallel, "file_record")

    def fail_closure(pid: int) -> None:
        nonlocal closure_failed
        # The real groups have already closed in the normal completion loop;
        # fabricate an inability to establish that fact in final cleanup.
        assert not pid_exists(pid)
        closure_failed = True
        raise RuntimeError("fixture closure observation failure")

    def inventory(path: Path) -> Any:
        if closure_failed:
            late_inventory.append(path)
        return original(path)

    monkeypatch.setattr(parallel, "_close_foreign_group", fail_closure)
    monkeypatch.setattr(parallel, "file_record", inventory)
    result = wave(root, [attempt(root, "A", "pass"), attempt(root, "B", "pass")])
    assert result["fatal"]["code"] == "process_or_evidence_closure_failed"
    assert not late_inventory
    for supervisor in result["supervisors"]:
        assert supervisor["trainer_group_closed"] is False
        assert not pid_exists(supervisor["trainer"]["pid"])
        assert all(
            record.get("unavailable") is True and "sha256" not in record
            for record in supervisor["files"].values()
        )

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
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from plant_backend.rl.v2.training import custody_io
from plant_backend.rl.v2.training.custody_json import canonical_bytes
from plant_backend.rl.v2.training.research_parallel_supervision import (
    ParallelAttempt,
    run_training_wave,
)


@pytest.mark.parametrize(
    "cleanup",
    [
        None,
        {},
        ["error"],
        [{}],
        [
            {
                "code": "io",
                "site": "fixture",
                "exception_type": "OSError",
                "message": "x" * 321,
            }
        ],
        [
            {
                "code": "io",
                "site": "fixture",
                "exception_type": "OSError",
                "message": "failure",
            }
        ]
        * 17,
    ],
)
def test_completion_rejects_malformed_or_unbounded_cleanup(cleanup: Any) -> None:
    from plant_backend.rl.v2.training.research_parallel_supervision import (
        _validate_completion_failures,
    )

    with pytest.raises(ValueError, match="supervisor (cleanup|failure)"):
        _validate_completion_failures({"fatal": None, "cleanup_failures": cleanup})


def test_completion_cannot_omit_a_cleanup_failure_from_fatal_state() -> None:
    from plant_backend.rl.v2.training.research_parallel_supervision import (
        _validate_completion_failures,
    )

    failure = {
        "code": "control_shutdown_failed",
        "site": "supervisor.control_shutdown",
        "exception_type": "OSError",
        "message": "fixture failure",
    }
    with pytest.raises(ValueError, match="omits cleanup failures"):
        _validate_completion_failures({"fatal": None, "cleanup_failures": [failure]})
    _validate_completion_failures({"fatal": failure, "cleanup_failures": [failure]})
    _validate_completion_failures({"fatal": None, "cleanup_failures": []})


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
    if behavior == "repeated_cancellation":
        from plant_backend.rl.v2.training import custody_process
        original_stop = custody_process._stop_group
        def interrupted_stop(child):
            os.kill(os.getpid(), signal.SIGTERM)
            os.kill(os.getpid(), signal.SIGHUP)
            os.kill(os.getpid(), signal.SIGINT)
            original_stop(child)
        custody_process._stop_group = interrupted_stop
        def cancel_when_ready():
            while not (root / (name + ".ready")).exists():
                time.sleep(.005)
            os.kill(os.getpid(), signal.SIGTERM)
        threading.Thread(target=cancel_when_ready, daemon=True).start()
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
    if behavior == "receipt_without_completion":
        os._exit(86)
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
    if kind in {"corrupt_receipt", "wrong_identity"}:
        assert all(receipt["name"] == "B" for receipt in result["receipts"])
        assert result["supervisors"][0]["receipt_status"] == "unavailable"


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
        assert completion["fatal"]["exception_type"] == "RuntimeError"
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


@pytest.mark.parametrize("exit_code", [0, 3])
def test_missing_completion_retains_only_independently_validated_receipt(
    fixture_root: Path,
    exit_code: int,
) -> None:
    root = fixture_root
    result = wave(
        root,
        [
            attempt(
                root,
                "A",
                f"raise SystemExit({exit_code})",
                behavior="receipt_without_completion",
            ),
            attempt(root, "B", waiting_code(root, "B")),
        ],
    )
    assert result["fatal"] is not None
    receipt = next(receipt for receipt in result["receipts"] if receipt["name"] == "A")
    assert receipt["process"]["exit_code"] == exit_code
    assert receipt["codes"] == (["child_nonzero_exit"] if exit_code else [])
    supervisor = result["supervisors"][0]
    assert supervisor["evidence"] is None
    assert supervisor["receipt_status"] == "independent_unadmitted"
    assert supervisor["admitted_success"] is False
    assert supervisor["files"]["completion"]["unavailable"] is True
    assert result["group_closure_known"] is True
    assert_groups_closed(result)


def test_repeated_cancellation_preserves_capture_receipt_and_completion(
    fixture_root: Path,
) -> None:
    root = fixture_root
    result = wave(
        root,
        [
            attempt(
                root,
                "A",
                waiting_code(root, "A"),
                behavior="repeated_cancellation",
            ),
            attempt(root, "B", waiting_code(root, "B")),
        ],
    )
    assert result["fatal"] is not None
    receipt = next(receipt for receipt in result["receipts"] if receipt["name"] == "A")
    assert "root_custody_lost" in receipt["codes"]
    assert receipt["process"]["exit_code"] == -signal.SIGTERM
    completion = json.loads((root / "supervisors" / "A.supervisor.json").read_text())
    assert completion["fatal"]["exception_type"] == "RuntimeError"
    assert completion["ordinary_receipt"]["sha256"]
    assert_groups_closed(result)


@pytest.mark.parametrize("failure", ["publication_signals", "disconnected_shutdown"])
def test_completion_publication_survives_signals_and_disconnected_control(
    fixture_root: Path, failure: str
) -> None:
    from plant_backend.rl.v2.training.research_parallel_supervision import (
        MODULE,
        SCHEMA,
    )

    root = fixture_root
    arm = attempt(root, "A", "raise SystemExit(86)")
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
    script = f"""
import errno, os, signal, socket
from pathlib import Path
from plant_backend.rl.v2.training import custody_process
from {MODULE.rsplit('.', 1)[0]} import research_parallel_supervision as parallel
from plant_backend.rl.v2.training.custody_json import canonical_bytes
original_subprocess = custody_process.subprocess
original_signal = custody_process.signal
original_handlers = {{s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}}
if {failure!r} == 'disconnected_shutdown':
    class DisconnectedSocket(socket.socket):
        def shutdown(self, how):
            raise OSError(errno.ENOTCONN, 'fixture disconnected transport')
    parallel.socket.socket = DisconnectedSocket
else:
    real_publish = parallel.publish_json
    def interrupted_publish(path, value):
        if path.name.endswith('.supervisor.json'):
            raw = canonical_bytes(value)
            with path.open('xb') as handle:
                handle.write(raw[:len(raw)//2])
                handle.flush()
                for signum in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT,signal.SIGTERM):
                    os.kill(os.getpid(), signum)
                handle.write(raw[len(raw)//2:])
                handle.flush()
                os.fsync(handle.fileno())
            parallel.fsync_directory(path.parent)
        else:
            real_publish(path, value)
    parallel.publish_json = interrupted_publish
result = parallel._supervise(Path({str(request_path)!r}), DESCRIPTOR)
assert custody_process.subprocess is original_subprocess
assert custody_process.signal is original_signal
assert all(signal.getsignal(s) == h for s,h in original_handlers.items())
raise SystemExit(result)
"""
    parent_control, supervisor_control = socket.socketpair()
    supervisor = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script.replace("DESCRIPTOR", str(supervisor_control.fileno())),
        ],
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
            chunk = parent_control.recv(65536)
            assert chunk
            raw += chunk
        trainer = json.loads(raw)
        parent_control.sendall(b"A")
        stdout, stderr = supervisor.communicate(timeout=12)
        assert stdout == b""
        assert stderr == b""
        assert supervisor.returncode == 86
        assert not pid_exists(trainer["pid"])
        with pytest.raises(ProcessLookupError):
            os.killpg(trainer["pgid"], 0)
        completion_path = root / "supervisors" / "A.supervisor.json"
        completion = json.loads(completion_path.read_text())
        assert completion_path.read_bytes() == canonical_bytes(completion)
        assert completion["fatal"]["exception_type"] == "RuntimeError"
        assert completion["fatal"]["message"] == "fixture integrity loss"
        assert completion["ordinary_receipt"]["sha256"]
        if failure == "disconnected_shutdown":
            assert any(
                error["code"] == "control_already_disconnected"
                for error in completion["cleanup_failures"]
            )
    finally:
        parent_control.close()
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait()


def test_first_failure_is_preserved_separately_from_cleanup_failure(
    fixture_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plant_backend.rl.v2.training import research_parallel_supervision as parallel

    root = fixture_root
    real_close = parallel._close_foreign_group

    def check() -> None:
        if all((root / (name + ".ready")).exists() for name in ("A", "B")):
            raise ValueError("fixture initiating failure\n" + "x" * 600)

    def close(pid: int) -> None:
        real_close(pid)
        raise RuntimeError("fixture later cleanup failure")

    monkeypatch.setattr(parallel, "_close_foreign_group", close)
    result = wave(
        root,
        [attempt(root, name, waiting_code(root, name)) for name in ("A", "B")],
        check,
    )
    assert result["fatal"]["exception_type"] == "ValueError"
    assert result["fatal"]["message"].startswith("fixture initiating failure")
    assert len(result["fatal"]["message"]) <= 320
    assert "\n" not in result["fatal"]["message"]
    assert result["fatal"]["site"]
    assert any(
        error["exception_type"] == "RuntimeError"
        and "fixture later cleanup failure" in error["message"]
        for error in result["cleanup_failures"]
    )
    assert result["group_closure_known"] is False
    assert all(not item["admitted_success"] for item in result["supervisors"])
    for supervisor in result["supervisors"]:
        assert not pid_exists(supervisor["trainer"]["pid"])
        assert all(
            record.get("unavailable") is True for record in supervisor["files"].values()
        )


def test_retained_descriptor_accepts_observed_exfat_allocation_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plant_backend.rl.v2.training import research_parallel_supervision as parallel

    path = tmp_path / "allocation.log"
    path.touch()
    monitor = parallel._LogMonitor()
    monitor.add(path, 4096)
    descriptor = monitor.logs[0].fd
    real_fstat, real_lstat = os.fstat, Path.lstat
    view = {"descriptor": 18446744073709490217, "path": 18446744073709490217}

    def identity(info: os.stat_result, inode: int) -> Any:
        fields = {
            name: getattr(info, name) for name in dir(info) if name.startswith("st_")
        }
        return SimpleNamespace(**{**fields, "st_ino": inode})

    def fstat(fd: int) -> Any:
        value = real_fstat(fd)
        return identity(value, view["descriptor"]) if fd == descriptor else value

    def lstat(target: Path, *args: Any, **kwargs: Any) -> Any:
        value = real_lstat(target, *args, **kwargs)
        return identity(value, view["path"]) if target == path else value

    monkeypatch.setattr(os, "fstat", fstat)
    monkeypatch.setattr(Path, "lstat", lstat)
    try:
        monitor.poll()
        with path.open("ab") as writer:
            writer.write(b"ordinary allocation")
            writer.flush()
            os.fsync(writer.fileno())
        # Probe: after writer flush its inode changed while path/reader lagged;
        # a bounded inconsistent view must keep the same retained reader.
        view["path"] = 2636951
        monitor.poll()
        view["descriptor"] = 2636951
        monitor.poll()
        monitor.finish()
        assert monitor.logs[0].fd == descriptor
        assert monitor.logs[0].hashed_bytes == path.stat().st_size
    finally:
        monitor.close()
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_persistent_descriptor_path_disagreement_has_a_bounded_settle_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plant_backend.rl.v2.training import research_parallel_supervision as parallel

    path = tmp_path / "unresolved.log"
    path.touch()
    monitor = parallel._LogMonitor()
    monitor.add(path, 4096)
    real_lstat = Path.lstat

    def mismatched(target: Path, *args: Any, **kwargs: Any) -> Any:
        value = real_lstat(target, *args, **kwargs)
        if target != path:
            return value
        fields = {
            name: getattr(value, name) for name in dir(value) if name.startswith("st_")
        }
        return SimpleNamespace(**{**fields, "st_ino": value.st_ino + 1})

    monkeypatch.setattr(Path, "lstat", mismatched)
    monkeypatch.setattr(parallel, "_IDENTITY_SETTLE_SECONDS", 0.005)
    try:
        monitor.poll()
        time.sleep(0.01)
        with pytest.raises(parallel._Failure):
            monitor.poll()
    finally:
        monitor.close()


def test_retained_descriptor_size_decrease_within_one_bracket_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plant_backend.rl.v2.training import research_parallel_supervision as parallel

    path = tmp_path / "bracket-truncation.log"
    path.touch()
    monitor = parallel._LogMonitor()
    monitor.add(path, 4096)
    descriptor = monitor.logs[0].fd
    real_fstat = os.fstat
    observations = iter((100, 0))

    def fstat(fd: int) -> Any:
        value = real_fstat(fd)
        if fd != descriptor:
            return value
        fields = {
            name: getattr(value, name) for name in dir(value) if name.startswith("st_")
        }
        return SimpleNamespace(**{**fields, "st_size": next(observations)})

    monkeypatch.setattr(os, "fstat", fstat)
    try:
        with pytest.raises(parallel._Failure) as raised:
            monitor.poll()
        assert raised.value.code == "log_truncated"
        assert raised.value.site == "log.metadata"
    finally:
        monitor.close()


@pytest.mark.parametrize(
    "mutation", ["replacement", "truncation", "regrowth", "prefix", "partial_tail"]
)
def test_log_monitor_rejects_persistent_replacement_and_byte_mutation(
    tmp_path: Path, mutation: str
) -> None:
    from plant_backend.rl.v2.training import research_parallel_supervision as parallel

    path = tmp_path / "sealed.log"
    path.touch()
    monitor = parallel._LogMonitor()
    monitor.add(path, 4 * parallel._HASH_BLOCK_BYTES)
    original = b"a" * (2 * parallel._HASH_BLOCK_BYTES + 31)
    path.write_bytes(original)
    try:
        monitor.poll()
        assert monitor.logs[0].hashed_bytes == len(original)
        if mutation == "replacement":
            replacement = tmp_path / "replacement.log"
            replacement.write_bytes(original)
            replacement.replace(path)
        elif mutation == "truncation":
            with path.open("r+b") as handle:
                handle.truncate(3)
        elif mutation == "regrowth":
            path.write_bytes(b"b" * len(original))
        else:
            with path.open("r+b") as handle:
                handle.seek(0 if mutation == "prefix" else len(original) - 1)
                handle.write(b"b")
                if mutation == "partial_tail":
                    handle.seek(0, os.SEEK_END)
                    handle.write(b"new append")
        with pytest.raises(parallel._Failure):
            monitor.finish()
    finally:
        monitor.close()


@pytest.mark.parametrize("forbidden", ["symlink", "hardlink", "nonempty"])
def test_log_monitor_rejects_unanchored_or_aliased_input(
    tmp_path: Path, forbidden: str
) -> None:
    from plant_backend.rl.v2.training import research_parallel_supervision as parallel

    original = tmp_path / "original.log"
    original.touch()
    path = tmp_path / "anchor.log"
    if forbidden == "symlink":
        path.symlink_to(original)
    elif forbidden == "hardlink":
        path.hardlink_to(original)
    else:
        path.write_bytes(b"already written")
    monitor = parallel._LogMonitor()
    try:
        with pytest.raises((parallel._Failure, OSError)):
            monitor.add(path, 4096)
    finally:
        monitor.close()


def test_monitor_read_budget_and_final_verification_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plant_backend.rl.v2.training import research_parallel_supervision as parallel

    path = tmp_path / "bounded.log"
    path.touch()
    monitor = parallel._LogMonitor()
    monitor.add(path, 4 * 1024 * 1024)
    path.write_bytes(b"a" * (3 * 1024 * 1024))
    read_sizes: list[int] = []
    real_pread = os.pread

    def pread(fd: int, length: int, offset: int) -> bytes:
        value = real_pread(fd, length, offset)
        read_sizes.append(len(value))
        return value

    monkeypatch.setattr(os, "pread", pread)
    try:
        monitor.poll()
        assert 0 < sum(read_sizes) <= 2 * 1024 * 1024
        assert max(read_sizes) <= 65536
        assert monitor.logs[0].hashed_bytes < path.stat().st_size
        assert parallel._MONITOR_READ_BYTES == 2 * 1024 * 1024
        assert parallel._MONITOR_SLICE_SECONDS == 0.01
        read_sizes.clear()
        monitor.finish()
        assert monitor.logs[0].hashed_bytes == path.stat().st_size
        assert sum(read_sizes) >= path.stat().st_size - 2 * 1024 * 1024
        record = monitor.record()
        assert record["final_verified"] is True
        assert record["backlog_bytes"] == 0
        assert record["max_backlog_bytes"] > 0
        assert record["max_poll_bytes"] <= 2 * 1024 * 1024
        assert record["bytes_read"] >= path.stat().st_size
        assert 0 < record["max_poll_wall_ns"] <= record["wall_ns"]
        assert record["configuration"]["prefix_deadline_seconds"] == 60.0
        assert record["configuration"]["hash_block_bytes"] == 65536
        assert record["logs"][0]["final_sha256"]
    finally:
        monitor.close()


def test_monitor_soft_time_slice_stops_after_one_slow_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plant_backend.rl.v2.training import research_parallel_supervision as parallel

    path = tmp_path / "slow.log"
    path.touch()
    monitor = parallel._LogMonitor()
    monitor.add(path, 4 * parallel._HASH_BLOCK_BYTES)
    path.write_bytes(b"a" * (3 * parallel._HASH_BLOCK_BYTES))
    reads = 0
    real_pread = os.pread

    def slow_pread(fd: int, length: int, offset: int) -> bytes:
        nonlocal reads
        reads += 1
        time.sleep(0.02)
        return real_pread(fd, length, offset)

    monkeypatch.setattr(os, "pread", slow_pread)
    try:
        monitor.poll()
        assert reads == 1
        assert 0 < monitor.logs[0].hashed_bytes < path.stat().st_size
        record = monitor.record()
        assert record["max_poll_wall_ns"] >= 20_000_000
        assert record["backlog_bytes"] > 0
    finally:
        monitor.close()


def test_verification_backlog_exceeding_deadline_is_explicit_failure(
    tmp_path: Path,
) -> None:
    from plant_backend.rl.v2.training import research_parallel_supervision as parallel

    path = tmp_path / "overdue.log"
    path.touch()
    monitor = parallel._LogMonitor()
    monitor.add(path, 4096)
    path.write_bytes(b"pending verification")
    monitor.logs[0].last_verified = time.monotonic() - 61
    try:
        with pytest.raises(parallel._Failure) as raised:
            monitor.poll()
        assert raised.value.code == "log_verification_overdue"
        assert raised.value.site == "log.budget"
        assert monitor.record()["backlog_bytes"] == path.stat().st_size
        assert monitor.record()["final_verified"] is False
    finally:
        monitor.close()


def test_final_verification_has_its_own_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plant_backend.rl.v2.training import research_parallel_supervision as parallel

    path = tmp_path / "final-deadline.log"
    path.touch()
    monitor = parallel._LogMonitor(poll_seconds=0.01)
    monitor.add(path, 4 * parallel._HASH_BLOCK_BYTES)
    path.write_bytes(b"a" * (3 * parallel._HASH_BLOCK_BYTES))
    monitor.logs[0].last_verified = time.monotonic() + 60
    real_pread = os.pread

    def slow_pread(fd: int, length: int, offset: int) -> bytes:
        time.sleep(0.02)
        return real_pread(fd, length, offset)

    monkeypatch.setattr(parallel, "_FINAL_VERIFICATION_SECONDS", 0.01)
    monkeypatch.setattr(os, "pread", slow_pread)
    try:
        with pytest.raises(parallel._Failure) as raised:
            monitor.finish()
        assert raised.value.code == "log_final_verification_overdue"
        assert raised.value.site == "log.final"
        assert monitor.record()["final_verified"] is False
        assert monitor.record()["backlog_bytes"] > 0
        assert monitor.record()["logs"][0]["final_sha256"] is None
    finally:
        monitor.close()


def test_final_verification_rejects_an_overdue_last_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plant_backend.rl.v2.training import research_parallel_supervision as parallel

    path = tmp_path / "final-last-read.log"
    path.touch()
    monitor = parallel._LogMonitor(poll_seconds=0.01)
    monitor.add(path, 1)
    path.write_bytes(b"a")
    real_pread = os.pread

    def slow_pread(fd: int, length: int, offset: int) -> bytes:
        time.sleep(0.02)
        return real_pread(fd, length, offset)

    monkeypatch.setattr(parallel, "_FINAL_VERIFICATION_SECONDS", 0.01)
    monkeypatch.setattr(os, "pread", slow_pread)
    try:
        with pytest.raises(parallel._Failure) as raised:
            monitor.finish()
        assert raised.value.code == "log_final_verification_overdue"
        assert raised.value.site == "log.final"
        assert monitor.record()["backlog_bytes"] == 0
        assert monitor.record()["final_verified"] is False
        assert monitor.record()["logs"][0]["final_sha256"] is None
        with pytest.raises(parallel._Failure):
            monitor.file_record(path)
    finally:
        monitor.close()


def test_periodic_live_audit_detects_an_earlier_prefix_rewrite(tmp_path: Path) -> None:
    from plant_backend.rl.v2.training import research_parallel_supervision as parallel

    path = tmp_path / "periodic.log"
    path.touch()
    monitor = parallel._LogMonitor()
    monitor.add(path, 4 * parallel._HASH_BLOCK_BYTES)
    path.write_bytes(b"a" * (3 * parallel._HASH_BLOCK_BYTES))
    try:
        monitor.poll()
        assert monitor.logs[0].hashed_bytes == path.stat().st_size
        with path.open("r+b") as handle:
            handle.write(b"changed prefix")
        monitor.logs[0].last_verified = time.monotonic() - 31
        with pytest.raises(parallel._Failure) as raised:
            monitor.poll()
        assert raised.value.code == "log_prefix_changed"
        assert raised.value.site == "log.prefix"
    finally:
        monitor.close()


def test_eight_streams_share_one_monitor_read_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plant_backend.rl.v2.training import research_parallel_supervision as parallel

    monitor = parallel._LogMonitor()
    monkeypatch.setattr(parallel, "_MONITOR_SLICE_SECONDS", 1.0)
    try:
        for index in range(8):
            path = tmp_path / f"stream-{index}.log"
            path.touch()
            monitor.add(path, 8 * parallel._HASH_BLOCK_BYTES)
            path.write_bytes(b"x" * (5 * parallel._HASH_BLOCK_BYTES))
        monitor.poll()
        assert all(log.hashed_bytes > 0 for log in monitor.logs)
        record = monitor.record()
        assert record["bytes_read"] <= 2 * 1024 * 1024
        assert record["backlog_bytes"] > 0
        monitor.finish()
        assert monitor.record()["max_poll_bytes"] <= 2 * 1024 * 1024
        assert monitor.record()["final_verified"] is True
    finally:
        monitor.close()


def test_unknown_trainer_custody_prevents_final_byte_reads_and_inventory(
    fixture_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plant_backend.rl.v2.training import research_parallel_supervision as parallel

    root = fixture_root
    observed_unknown = False
    late_reads: list[Path] = []
    late_inventories: list[Path] = []
    real_closed = parallel._all_groups_closed
    real_read = custody_io.read_regular
    real_inventory = custody_io.file_record

    def closed(running: Any) -> bool:
        nonlocal observed_unknown
        result = real_closed(running)
        if not result and all(
            item.child is None or item.supervisor_group_closed for item in running
        ):
            observed_unknown = True
        return result

    def read(path: Path) -> bytes:
        if observed_unknown:
            late_reads.append(path)
        return real_read(path)

    def inventory(path: Path) -> Any:
        if observed_unknown:
            late_inventories.append(path)
        return real_inventory(path)

    monkeypatch.setattr(parallel, "_all_groups_closed", closed)
    monkeypatch.setattr(parallel, "read_regular", read)
    monkeypatch.setattr(parallel, "file_record", inventory)
    result = wave(
        root,
        [
            attempt(root, "A", "pass", behavior="missing_evidence"),
            attempt(root, "B", waiting_code(root, "B")),
        ],
    )
    assert observed_unknown
    assert result["group_closure_known"] is False
    assert result["fatal"] is not None
    assert result["receipts"] == []
    assert not late_reads
    assert not late_inventories
    assert result["monitoring"]["final_verified"] is False
    for supervisor in result["supervisors"]:
        assert supervisor["admitted_success"] is False
        assert supervisor["receipt_status"] == "unavailable"
        assert all(
            record.get("unavailable") is True for record in supervisor["files"].values()
        )
    assert_groups_closed(result)

"""Two isolated, bounded serial custody supervisors; no learner is imported here.

The parent sees process identities and ordinary custody receipts only. A short
exec barrier registers each trainer's process group before its command can run.
The existing serial capture runs on each supervisor's main thread, preserving
its signal handling and per-process RUSAGE_CHILDREN accounting.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import importlib
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from types import FrameType, SimpleNamespace
from typing import Any, BinaryIO, Callable, Sequence

from .custody_io import (
    file_record,
    fsync_directory,
    publish_json,
    read_regular,
    require_directory,
)
from .custody_json import canonical_bytes, decode
from .custody_process import _stop_group

MODULE = "plant_backend.rl.v2.training.research_parallel_supervision"
SCHEMA = "plant-research-parallel-training-wave-v1"
INTEGRITY_EXIT = 86
_EVIDENCE_LIMIT = 1024 * 1024
_CONTROL_LIMIT = 64 * 1024
_CLOSURE_SECONDS = 15.0
_STARTUP_SECONDS = 30.0
_CODES = {
    "child_spawn_failed",
    "child_timeout",
    "child_log_limit_exceeded",
    "child_signaled",
    "child_nonzero_exit",
    "root_custody_lost",
}


@dataclass(frozen=True)
class ParallelAttempt:
    name: str
    stage: str
    timeout_seconds: int
    extra: tuple[str, ...]
    command: tuple[str, ...]
    stdout: Path
    stderr: Path
    receipt: Path

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "timeout_seconds": self.timeout_seconds,
            "extra": list(self.extra),
            "command": list(self.command),
            "stdout": str(self.stdout),
            "stderr": str(self.stderr),
            "receipt": str(self.receipt),
        }


@dataclass
class _Running:
    attempt: ParallelAttempt
    request: dict[str, Any]
    paths: dict[str, Path]
    logs: list[BinaryIO]
    child: subprocess.Popen[bytes] | None = None
    control: socket.socket | None = None
    buffer: bytes = b""
    sizes: list[int] = field(default_factory=lambda: [0, 0])
    trainer: dict[str, Any] | None = None
    started: float = 0.0
    eof: bool = False
    closed: bool = False
    evidence: dict[str, Any] | None = None
    receipt: dict[str, Any] | None = None
    started_record: dict[str, Any] | None = None
    supervisor_group_closed: bool = False
    trainer_group_closed: bool | None = None
    log_inodes: dict[str, tuple[int, int]] = field(default_factory=dict)


def _bounded_json(path: Path) -> dict[str, Any]:
    if path.lstat().st_size > _EVIDENCE_LIMIT:
        raise ValueError("supervision evidence exceeds the bounded object size")
    raw = read_regular(path)
    value = decode(raw)
    if raw != canonical_bytes(value):
        raise ValueError("supervision evidence is not canonical")
    return value


def _group_exists(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _signal_group(pid: int, signum: int) -> None:
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass


def _close_foreign_group(pid: int) -> None:
    """Fallback for a trainer whose supervisor died before reaping it."""
    if not _group_exists(pid):
        return
    _signal_group(pid, signal.SIGTERM)
    deadline = time.monotonic() + 1
    while _group_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    if _group_exists(pid):
        _signal_group(pid, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while _group_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    if _group_exists(pid):
        raise RuntimeError("trainer process group could not be closed")


def _validate_receipt(
    receipt: dict[str, Any], attempt: ParallelAttempt, trainer: dict[str, Any] | None
) -> None:
    if set(receipt) != {"stage", "name", "process", "codes", "closed_at"}:
        raise ValueError("ordinary receipt fields differ")
    if receipt["stage"] != attempt.stage or receipt["name"] != attempt.name:
        raise ValueError("ordinary receipt identity differs")
    codes = receipt["codes"]
    if (
        not isinstance(codes, list)
        or any(not isinstance(code, str) or code not in _CODES for code in codes)
        or codes != sorted(set(codes))
        or not isinstance(receipt["closed_at"], str)
        or not receipt["closed_at"]
    ):
        raise ValueError("ordinary receipt codes or closure differ")
    process = receipt["process"]
    if (
        not isinstance(process, dict)
        or set(process)
        != {
            "command",
            "timeout_seconds",
            "started",
            "exit_code",
            "termination_signal",
            "wall_elapsed_ns",
            "child_user_cpu_ns",
            "child_system_cpu_ns",
        }
        or (
            process.get("command") != list(attempt.command)
            or process.get("timeout_seconds") != attempt.timeout_seconds
            or type(process.get("started")) is not bool
        )
    ):
        raise ValueError("ordinary child command or process identity differs")
    started = process["started"]
    exit_code = process.get("exit_code")
    if started:
        if trainer is None or type(exit_code) is not int:
            raise ValueError("ordinary child has no registered closed identity")
        if exit_code > 0 and "child_nonzero_exit" not in codes:
            raise ValueError("ordinary receipt omitted nonzero exit")
        if exit_code < 0 and (
            "child_signaled" not in codes
            or process.get("termination_signal") != -exit_code
        ):
            raise ValueError("ordinary receipt omitted child signal")
        for key in ("wall_elapsed_ns", "child_user_cpu_ns", "child_system_cpu_ns"):
            if type(process.get(key)) is not int or process[key] < 0:
                raise ValueError("ordinary child accounting is incomplete")
        if (
            process["wall_elapsed_ns"] >= attempt.timeout_seconds * 1_000_000_000
            and "child_timeout" not in codes
        ):
            raise ValueError("ordinary receipt omitted timeout")
    elif (
        trainer is not None
        or exit_code is not None
        or "child_spawn_failed" not in codes
    ):
        raise ValueError("ordinary spawn failure identity differs")


def _read_completion(item: _Running) -> None:
    assert item.child is not None
    evidence = _bounded_json(item.paths["completion"])
    if set(evidence) != {
        "schema",
        "request",
        "supervisor_pid",
        "supervisor_pgid",
        "trainer",
        "ordinary_receipt",
        "ordinary_logs",
        "fatal",
        "closed_at_ns",
    } or (
        evidence["schema"] != SCHEMA
        or evidence["request"] != file_record(item.paths["request"])
        or evidence["supervisor_pid"] != item.child.pid
        or evidence["supervisor_pgid"] != item.child.pid
        or evidence["trainer"] != item.trainer
        or type(evidence["closed_at_ns"]) is not int
    ):
        raise ValueError("supervisor completion identity differs")
    if evidence["ordinary_receipt"] is not None:
        if evidence["ordinary_receipt"] != file_record(item.attempt.receipt):
            raise ValueError("ordinary receipt bytes changed after supervision")
        receipt = _bounded_json(item.attempt.receipt)
        _validate_receipt(receipt, item.attempt, item.trainer)
        item.receipt = receipt
    if evidence["ordinary_logs"] is not None and evidence["ordinary_logs"] != {
        "stdout": file_record(item.attempt.stdout),
        "stderr": file_record(item.attempt.stderr),
    }:
        raise ValueError("ordinary logs changed after supervision")
    item.evidence = evidence
    if evidence["fatal"] is not None:
        raise ValueError("supervisor reported an integrity failure")
    if (
        item.child.returncode != 0
        or item.receipt is None
        or evidence["ordinary_logs"] is None
    ):
        raise ValueError("supervisor did not close with ordinary receipt evidence")
    if (
        item.receipt["process"]["exit_code"] == INTEGRITY_EXIT
        or "root_custody_lost" in item.receipt["codes"]
    ):
        raise ValueError("trainer reported integrity loss or operator interruption")


def _drain(
    selector: selectors.BaseSelector,
    limit: int,
    *,
    authorize: bool,
    timeout: float,
) -> None:
    for key, _ in selector.select(timeout):
        item, stream_index = key.data
        if stream_index is None:
            assert item.control is not None and item.child is not None
            chunk = item.control.recv(_CONTROL_LIMIT)
            if not chunk:
                item.eof = True
                selector.unregister(key.fileobj)
                continue
            item.buffer += chunk
            if len(item.buffer) > _CONTROL_LIMIT:
                raise ValueError("supervisor control stream exceeds its bound")
            while b"\n" in item.buffer:
                raw, item.buffer = item.buffer.split(b"\n", 1)
                event = decode(raw)
                if (
                    item.trainer is not None
                    or set(event)
                    != {"pid", "pgid", "supervisor_pid", "command", "bootstrap_command"}
                    or (
                        type(event.get("pid")) is not int
                        or event["pid"] <= 1
                        or event["pid"] in {os.getpid(), os.getpgrp(), item.child.pid}
                        or event["pgid"] != event["pid"]
                        or event["supervisor_pid"] != item.child.pid
                        or event["command"] != list(item.attempt.command)
                        or event["bootstrap_command"]
                        != [
                            sys.executable,
                            "-m",
                            MODULE,
                            "_exec",
                            "--",
                            *item.attempt.command,
                        ]
                    )
                ):
                    raise ValueError("trainer launch identity differs")
                item.trainer = event
                item.trainer_group_closed = False
                publish_json(item.paths["trainer"], event)
                if authorize:
                    item.control.sendall(b"A")
        else:
            remaining = limit - item.sizes[stream_index]
            chunk = os.read(key.fd, max(1, min(65_536, remaining)))
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            if remaining:
                item.logs[stream_index].write(chunk[:remaining])
                item.sizes[stream_index] += min(len(chunk), remaining)
            if item.sizes[stream_index] >= limit:
                raise ValueError("supervisor log limit reached")


def _check_live_files(item: _Running, child_limit: int, supervisor_limit: int) -> None:
    if _bounded_json(item.paths["request"]) != item.request:
        raise ValueError("supervisor request changed")
    if (
        item.started_record is not None
        and _bounded_json(item.paths["started"]) != item.started_record
    ):
        raise ValueError("supervisor start evidence changed")
    if (
        item.trainer is not None
        and _bounded_json(item.paths["trainer"]) != item.trainer
    ):
        raise ValueError("trainer start evidence changed")
    for path, limit in (
        (item.attempt.stdout, child_limit),
        (item.attempt.stderr, child_limit),
        (item.paths["stdout"], supervisor_limit),
        (item.paths["stderr"], supervisor_limit),
    ):
        require_directory(path.parent)
        try:
            info = path.lstat()
        except FileNotFoundError:
            if str(path) in item.log_inodes:
                raise ValueError("training log disappeared") from None
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limit:
            raise ValueError("training log identity or bound differs")
        identity = (info.st_dev, info.st_ino)
        if item.log_inodes.setdefault(str(path), identity) != identity:
            raise ValueError("training log was replaced")


def run_training_wave(
    attempts: Sequence[ParallelAttempt],
    *,
    supervisor_directory: Path,
    worker_module: str,
    check_integrity: Callable[[], Any],
    minimum_free_bytes: int,
    supervisor_log_limit_bytes: int,
    worker_function: str = "closed_child",
    child_log_limit_bytes: int | None = None,
    poll_seconds: float = 0.1,
    integrity_check_interval_seconds: float = 30.0,
) -> dict[str, Any]:
    """Attempt exactly two arms once, close all groups, and return sealed metadata.

    ``check_integrity`` must inspect only frozen source/host/admission evidence,
    and raise on a loss. It runs before launch, throughout the wave, and after
    closure. It must not inspect learner outcomes. The serial worker must map
    allocator/checkpoint integrity loss to exit 86. Ordinary child failures do
    not cancel their peer. Any supervision/evidence failure cancels the wave.

    The directory must exist and be empty. Exclusive request, supervisor-start,
    and trainer-start JSON records offer read-only live process state. No retry
    or serial fallback is performed. Pre-admission errors raise; once admitted,
    failures return ``fatal`` only after both supervisors and trainers close.
    """
    require_directory(supervisor_directory)
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("training-wave signal custody requires the main thread")
    child_limit = (
        supervisor_log_limit_bytes
        if child_log_limit_bytes is None
        else child_log_limit_bytes
    )
    if (
        len(attempts) != 2
        or len({a.name for a in attempts}) != 2
        or minimum_free_bytes < 0
        or supervisor_log_limit_bytes <= 0
        or child_limit <= 0
        or not 0.01 <= poll_seconds <= 1
        or not 0.01 <= integrity_check_interval_seconds <= 30.0
        or not re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", worker_module)
        or not re.fullmatch(r"[A-Za-z_]\w*", worker_function)
        or any(supervisor_directory.iterdir())
    ):
        raise ValueError("parallel training wave admission differs")
    paths: set[Path] = set()
    for attempt in attempts:
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}", attempt.name)
            or attempt.timeout_seconds <= 0
            or not attempt.command
            or not all(
                isinstance(arg, str) and "\0" not in arg for arg in attempt.command
            )
            or attempt.stdout.parent != attempt.stderr.parent
        ):
            raise ValueError("parallel attempt identity or limits differ")
        for path in (attempt.stdout, attempt.stderr, attempt.receipt):
            require_directory(path.parent)
            if (
                path in paths
                or os.path.lexists(path)
                or path.is_relative_to(supervisor_directory)
            ):
                raise ValueError("parallel attempt paths collide or already exist")
            paths.add(path)
    check_integrity()
    if shutil.disk_usage(supervisor_directory).free < minimum_free_bytes:
        raise ValueError("training wave storage floor is unavailable")
    publish_json(
        supervisor_directory / "admission.json",
        {
            "schema": SCHEMA,
            "attempts": [a.record() for a in attempts],
            "worker_module": worker_module,
            "worker_function": worker_function,
            "minimum_free_bytes": minimum_free_bytes,
            "supervisor_log_limit_bytes": supervisor_log_limit_bytes,
            "child_log_limit_bytes": child_limit,
            "poll_seconds": poll_seconds,
            "integrity_check_interval_seconds": integrity_check_interval_seconds,
            "closure_seconds": _CLOSURE_SECONDS,
            "startup_seconds": _STARTUP_SECONDS,
        },
    )
    running: list[_Running] = []
    fatal: dict[str, str] | None = None
    cleanup_errors: list[str] = []
    selector = selectors.DefaultSelector()

    def interrupt(signum: int, frame: FrameType | None) -> None:
        raise KeyboardInterrupt

    previous = {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        for signum in previous:
            signal.signal(signum, interrupt)
        for attempt in attempts:
            request = {
                "schema": SCHEMA,
                "attempt": attempt.record(),
                "worker_module": worker_module,
                "worker_function": worker_function,
                "parent_pid": os.getpid(),
            }
            item_paths = {
                key: supervisor_directory / f"{attempt.name}.{suffix}"
                for key, suffix in {
                    "request": "request.json",
                    "started": "started.json",
                    "trainer": "trainer.json",
                    "completion": "supervisor.json",
                    "stdout": "supervisor.stdout.log",
                    "stderr": "supervisor.stderr.log",
                }.items()
            }
            item = _Running(attempt, request, item_paths, [])
            running.append(item)
            publish_json(item_paths["request"], request)
            for key in ("stdout", "stderr"):
                fd = os.open(
                    item_paths[key],
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                )
                item.logs.append(os.fdopen(fd, "wb"))
            parent_control, supervisor_control = socket.socketpair()
            item.control = parent_control
            try:
                command = [
                    sys.executable,
                    "-m",
                    MODULE,
                    "_supervise",
                    str(item_paths["request"]),
                    str(supervisor_control.fileno()),
                ]
                item.child = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    pass_fds=(supervisor_control.fileno(),),
                )
            finally:
                supervisor_control.close()
            item.started = time.monotonic()
            item.started_record = {
                "schema": SCHEMA,
                "supervisor_pid": item.child.pid,
                "supervisor_pgid": item.child.pid,
                "command": command,
                "request": file_record(item_paths["request"]),
                "started_at_ns": time.time_ns(),
            }
            publish_json(item_paths["started"], item.started_record)
            for index, stream in enumerate((item.child.stdout, item.child.stderr)):
                assert stream is not None
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, (item, index))
            parent_control.setblocking(False)
            selector.register(parent_control, selectors.EVENT_READ, (item, None))
        next_integrity_check = time.monotonic() + integrity_check_interval_seconds
        checked_after_launch = False
        while any(not item.closed for item in running):
            if shutil.disk_usage(supervisor_directory).free < minimum_free_bytes:
                raise ValueError("training wave storage floor was lost")
            for item in running:
                _check_live_files(item, child_limit, supervisor_log_limit_bytes)
            _drain(
                selector,
                supervisor_log_limit_bytes,
                authorize=True,
                timeout=poll_seconds,
            )
            for item in running:
                if item.closed:
                    continue
                assert item.child is not None
                if (
                    time.monotonic() - item.started
                    > item.attempt.timeout_seconds + _CLOSURE_SECONDS
                ):
                    raise ValueError("supervisor exceeded its total wall bound")
                if (
                    item.trainer is None
                    and time.monotonic() - item.started > _STARTUP_SECONDS
                ):
                    raise ValueError(
                        "supervisor did not establish its trainer within the startup bound"
                    )
                if item.child.poll() is not None and item.eof:
                    if item.buffer:
                        raise ValueError("supervisor control stream ended mid-record")
                    _stop_group(item.child)
                    if item.trainer is not None and _group_exists(item.trainer["pgid"]):
                        raise ValueError(
                            "trainer group survived its supervisor closure"
                        )
                    _read_completion(item)
                    item.closed = True
            # Both barriers are acknowledged before an expensive source/host
            # check. Process, log and storage polling uses the shorter cadence.
            launches_known = all(
                item.trainer is not None or item.closed for item in running
            )
            if launches_known and (
                time.monotonic() >= next_integrity_check or not checked_after_launch
            ):
                check_integrity()
                checked_after_launch = launches_known
                next_integrity_check = (
                    time.monotonic() + integrity_check_interval_seconds
                )
        check_integrity()
        for item in running:
            _check_live_files(item, child_limit, supervisor_log_limit_bytes)
            _read_completion(item)
    except BaseException as exc:
        fatal = {
            "code": (
                "operator_interruption"
                if isinstance(exc, (KeyboardInterrupt, SystemExit))
                else "supervision_integrity_loss"
            ),
            "detail": type(exc).__name__,
        }
    finally:
        # Repeated operator signals cannot interrupt the bounded cleanup sequence.
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        try:
            if fatal is not None:
                for item in running:
                    try:
                        if item.child is not None and item.child.poll() is None:
                            _signal_group(item.child.pid, signal.SIGTERM)
                    except BaseException as exc:
                        cleanup_errors.append(type(exc).__name__)
                    if item.control is not None:
                        item.control.close()  # Parent-loss watchdog also wakes.
                deadline = time.monotonic() + _CLOSURE_SECONDS
                while (
                    any(
                        item.child is not None and item.child.poll() is None
                        for item in running
                    )
                    and time.monotonic() < deadline
                ):
                    # Draining only logs avoids blocking a supervisor during closure.
                    for item in running:
                        if item.child is None:
                            continue
                        for index, stream in enumerate(
                            (item.child.stdout, item.child.stderr)
                        ):
                            if stream is None:
                                continue
                            try:
                                chunk = os.read(stream.fileno(), 65_536)
                            except (BlockingIOError, OSError):
                                continue
                            room = max(
                                0, supervisor_log_limit_bytes - item.sizes[index]
                            )
                            item.logs[index].write(chunk[:room])
                            item.sizes[index] += min(room, len(chunk))
                    time.sleep(0.01)
            for item in running:
                try:
                    if item.child is not None:
                        _stop_group(item.child)
                        item.supervisor_group_closed = True
                except BaseException as exc:
                    cleanup_errors.append(type(exc).__name__)
                try:
                    if item.trainer is not None:
                        _close_foreign_group(item.trainer["pgid"])
                        item.trainer_group_closed = True
                except BaseException as exc:
                    cleanup_errors.append(type(exc).__name__)
                if item.child is not None:
                    for index, stream in enumerate(
                        (item.child.stdout, item.child.stderr)
                    ):
                        if stream is None:
                            continue
                        try:
                            while True:
                                chunk = os.read(stream.fileno(), 65_536)
                                if not chunk:
                                    break
                                room = max(
                                    0, supervisor_log_limit_bytes - item.sizes[index]
                                )
                                item.logs[index].write(chunk[:room])
                                item.sizes[index] += min(room, len(chunk))
                                if len(chunk) >= room:
                                    fatal = fatal or {
                                        "code": "supervision_integrity_loss",
                                        "detail": "supervisor_log_limit_reached",
                                    }
                        except (BlockingIOError, OSError):
                            pass
                        finally:
                            stream.close()
                if item.control is not None:
                    item.control.close()
                for handle in item.logs:
                    try:
                        handle.flush()
                        os.fsync(handle.fileno())
                    except BaseException as exc:
                        cleanup_errors.append(type(exc).__name__)
                    finally:
                        handle.close()
            if _all_groups_closed(running):
                for item in running:
                    if item.evidence is None and item.child is not None:
                        try:
                            _read_completion(item)
                        except BaseException:
                            pass  # Missing/corrupt evidence has already aborted the wave.
            try:
                fsync_directory(supervisor_directory)
            except BaseException as exc:
                cleanup_errors.append(type(exc).__name__)
        finally:
            selector.close()
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    if cleanup_errors:
        fatal = {
            "code": "process_or_evidence_closure_failed",
            "detail": ",".join(cleanup_errors),
        }
    supervisors = []
    for item in running:
        records: dict[str, Any] = {}
        for key, path in item.paths.items():
            if not _all_groups_closed(running):
                records[key] = {
                    "path": str(path),
                    "unavailable": True,
                    "reason": "process_group_not_closed",
                }
                continue
            try:
                records[key] = file_record(path)
            except (OSError, ValueError):
                records[key] = {"path": str(path), "unavailable": True}
                if key != "trainer" or item.trainer is not None:
                    fatal = fatal or {
                        "code": "supervision_integrity_loss",
                        "detail": "missing_supervisor_evidence",
                    }
        supervisors.append(
            {
                "name": item.attempt.name,
                "supervisor_pid": item.child.pid if item.child is not None else None,
                "supervisor_pgid": item.child.pid if item.child is not None else None,
                "exit_code": item.child.returncode if item.child is not None else None,
                "supervisor_group_closed": item.supervisor_group_closed,
                "trainer_group_closed": item.trainer_group_closed,
                "trainer": item.trainer,
                "evidence": item.evidence,
                "files": records,
            }
        )
    return {
        "schema": SCHEMA,
        "fatal": fatal,
        "receipts": [item.receipt for item in running if item.receipt is not None],
        "supervisors": supervisors,
        "closed_at_ns": time.time_ns(),
    }


def _all_groups_closed(running: Sequence[_Running]) -> bool:
    return all(
        (item.child is None or item.supervisor_group_closed)
        and (item.trainer is None or item.trainer_group_closed is True)
        for item in running
    )


def _supervise(request_path: Path, descriptor: int) -> int:
    """One serial worker, with a watchdog that never performs process capture."""
    from . import custody_process

    request = _bounded_json(request_path)
    attempt = request["attempt"]
    completion_path = request_path.with_name(f"{attempt['name']}.supervisor.json")
    control = socket.socket(fileno=descriptor)
    acknowledged = threading.Event()
    finished = threading.Event()
    trainer: dict[str, Any] | None = None
    real_popen = subprocess.Popen

    def parent_watchdog() -> None:
        try:
            while not finished.is_set():
                value = control.recv(1)
                if value == b"A" and not acknowledged.is_set():
                    acknowledged.set()
                else:
                    break
        except OSError:
            pass
        if not finished.is_set():
            os.kill(os.getpid(), signal.SIGTERM)

    def interrupt(signum: int, frame: FrameType | None) -> None:
        raise KeyboardInterrupt

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupt)
    watcher = threading.Thread(target=parent_watchdog, daemon=True)
    watcher.start()

    def tracked_popen(command: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        nonlocal trainer
        if (
            trainer is not None
            or command != attempt["command"]
            or kwargs.get("start_new_session") is not True
            or kwargs.get("stdout") != subprocess.PIPE
            or kwargs.get("stderr") != subprocess.PIPE
            or kwargs.get("pass_fds")
        ):
            raise ValueError("serial worker launch differs from its one frozen command")
        reader, writer = os.pipe()
        child: subprocess.Popen[bytes] | None = None
        try:
            bootstrap = [sys.executable, "-m", MODULE, "_exec", "--", *command]
            env = dict(kwargs.pop("env", None) or os.environ)
            env["PLANT_RESEARCH_EXEC_BARRIER_FD"] = str(reader)
            kwargs["pass_fds"] = (reader,)
            child = real_popen(bootstrap, env=env, **kwargs)
            trainer = {
                "pid": child.pid,
                "pgid": child.pid,
                "supervisor_pid": os.getpid(),
                "command": command,
                "bootstrap_command": bootstrap,
            }
            import json

            control.sendall(json.dumps(trainer, separators=(",", ":")).encode() + b"\n")
            if not acknowledged.wait(timeout=5):
                raise RuntimeError("parent did not acknowledge trainer identity")
            os.write(writer, b"A")
            return child
        except BaseException:
            if child is not None:
                _stop_group(child)
            raise
        finally:
            os.close(reader)
            os.close(writer)

    receipt_record = None
    ordinary_logs = None
    fatal = None
    try:
        worker = getattr(
            importlib.import_module(request["worker_module"]),
            request["worker_function"],
        )
        # Replace only this module's subprocess binding. Source/host admission
        # may legitimately invoke Git via the real subprocess module.
        custody_process.subprocess = SimpleNamespace(  # type: ignore[attr-defined,assignment]
            Popen=tracked_popen,
            PIPE=subprocess.PIPE,
            DEVNULL=subprocess.DEVNULL,
            TimeoutExpired=subprocess.TimeoutExpired,
        )
        result = worker(
            attempt["stage"],
            attempt["name"],
            attempt["timeout_seconds"],
            attempt["extra"],
        )
        receipt_path = Path(attempt["receipt"])
        if _bounded_json(receipt_path) != result:
            raise ValueError("worker returned different ordinary receipt evidence")
        receipt_record = file_record(receipt_path)
        ordinary_logs = {
            "stdout": file_record(Path(attempt["stdout"])),
            "stderr": file_record(Path(attempt["stderr"])),
        }
    except BaseException as exc:
        fatal = type(exc).__name__
        try:
            receipt_record = file_record(Path(attempt["receipt"]))
        except (OSError, ValueError):
            pass
    finally:
        custody_process.subprocess = subprocess  # type: ignore[attr-defined]
        finished.set()
        control.shutdown(socket.SHUT_RDWR)
        control.close()
        watcher.join(timeout=1)
    publish_json(
        completion_path,
        {
            "schema": SCHEMA,
            "request": file_record(request_path),
            "supervisor_pid": os.getpid(),
            "supervisor_pgid": os.getpgrp(),
            "trainer": trainer,
            "ordinary_receipt": receipt_record,
            "ordinary_logs": ordinary_logs,
            "fatal": fatal,
            "closed_at_ns": time.time_ns(),
        },
    )
    return INTEGRITY_EXIT if fatal is not None else 0


def _exec_barrier(command: list[str]) -> int:
    descriptor = int(os.environ.pop("PLANT_RESEARCH_EXEC_BARRIER_FD"))
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(descriptor, selectors.EVENT_READ)
            if not selector.select(timeout=5) or os.read(descriptor, 1) != b"A":
                return INTEGRITY_EXIT
    finally:
        os.close(descriptor)
    os.execvpe(command[0], command, os.environ)
    return INTEGRITY_EXIT  # pragma: no cover: exec does not return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("_supervise", "_exec"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.mode == "_exec":
        command = args.arguments
        if command and command[0] == "--":
            command = command[1:]
        return _exec_barrier(command)
    if len(args.arguments) != 2:
        parser.error("supervisor needs its frozen request and inherited control fd")
    return _supervise(Path(args.arguments[0]), int(args.arguments[1]))


if __name__ == "__main__":
    raise SystemExit(main())

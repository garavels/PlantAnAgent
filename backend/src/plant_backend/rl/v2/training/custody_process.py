"""Sealed bounded subprocess capture. No stream content enters a report or UI."""

from __future__ import annotations

import os
from pathlib import Path
import resource
import selectors
import signal
import subprocess
from types import FrameType
import time
from typing import Any, BinaryIO

from .custody_io import fsync_directory, require_directory


def _group_exists(child: subprocess.Popen[bytes]) -> bool:
    deadline = time.monotonic() + 1.0
    while True:
        child.poll()
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # Darwin can report EPERM for an exiting, unreaped group leader.
            # Retry after reaping; only ESRCH establishes group disappearance.
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)
        else:
            return True


def _signal_group(child: subprocess.Popen[bytes], signum: int) -> None:
    try:
        os.killpg(child.pid, signum)
    except ProcessLookupError:
        pass
    except PermissionError:
        if _group_exists(child):
            raise


def _stop_group(child: subprocess.Popen[bytes]) -> None:
    """Reap the leader and stop surviving descendants before any inventory."""
    child.poll()
    if _group_exists(child):
        _signal_group(child, signal.SIGTERM)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            child.poll()
            if not _group_exists(child):
                break
            time.sleep(0.01)
        if _group_exists(child):
            _signal_group(child, signal.SIGKILL)
    child.wait()
    # Orphaned grandchildren are reaped by the host's init process. If the
    # process group persists, no caller may publish an artifact inventory.
    deadline = time.monotonic() + 5.0
    while _group_exists(child) and time.monotonic() < deadline:
        time.sleep(0.01)
    if _group_exists(child):
        raise RuntimeError("child process group could not be closed")


def capture_process(
    command: list[str],
    *,
    stdout: Path,
    stderr: Path,
    timeout_seconds: int,
    log_limit_bytes: int,
) -> tuple[dict[str, Any], set[str]]:
    """Handle operator interruption with the same complete process-group cleanup."""

    def interrupt(signum: int, frame: FrameType | None) -> None:
        raise KeyboardInterrupt

    previous = {
        signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGHUP)
    }
    try:
        for signum in previous:
            signal.signal(signum, interrupt)
        return _capture_process(
            command,
            stdout=stdout,
            stderr=stderr,
            timeout_seconds=timeout_seconds,
            log_limit_bytes=log_limit_bytes,
        )
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _capture_process(
    command: list[str],
    *,
    stdout: Path,
    stderr: Path,
    timeout_seconds: int,
    log_limit_bytes: int,
) -> tuple[dict[str, Any], set[str]]:
    """Execute one command and return only process metadata and closed codes.

    The campaign supplies the frozen command and limits; fixture tests use tiny
    processes here. Publication errors propagate so the claim stays consumed.
    """
    if timeout_seconds <= 0 or log_limit_bytes <= 0:
        raise ValueError("process limits must be positive")
    if stdout.parent != stderr.parent:
        raise ValueError("sealed logs must share the declared directory")
    require_directory(stdout.parent)
    logs: list[BinaryIO] = []
    child: subprocess.Popen[bytes] | None = None
    observed: set[str] = set()
    started_at = time.perf_counter_ns()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    process: dict[str, Any] = {
        "command": command,
        "timeout_seconds": timeout_seconds,
        "started": False,
        "exit_code": None,
        "termination_signal": None,
        "wall_elapsed_ns": None,
        "child_user_cpu_ns": None,
        "child_system_cpu_ns": None,
    }
    try:
        for path in (stdout, stderr):
            descriptor = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            logs.append(os.fdopen(descriptor, "wb"))
        fsync_directory(stdout.parent)
        try:
            child = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError:
            observed.add("child_spawn_failed")
        if child is not None:
            process["started"] = True
            assert child.stdout is not None and child.stderr is not None
            streams = (child.stdout, child.stderr)
            sizes = [0, 0]
            deadline = time.monotonic() + timeout_seconds
            try:
                with selectors.DefaultSelector() as selector:
                    for index, stream in enumerate(streams):
                        os.set_blocking(stream.fileno(), False)
                        selector.register(stream, selectors.EVENT_READ, index)
                    while selector.get_map():
                        if time.monotonic() >= deadline:
                            observed.add("child_timeout")
                            break
                        for key, _ in selector.select(
                            timeout=min(0.05, max(0.0, deadline - time.monotonic()))
                        ):
                            index = key.data
                            chunk = os.read(
                                key.fd, min(65_536, log_limit_bytes - sizes[index])
                            )
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            logs[index].write(chunk)
                            sizes[index] += len(chunk)
                            if sizes[index] == log_limit_bytes:
                                observed.add("child_log_limit_exceeded")
                                break
                        if "child_log_limit_exceeded" in observed:
                            break
                    # EOF can arrive while the leader continues running.
                    if not observed:
                        try:
                            child.wait(timeout=max(0.0, deadline - time.monotonic()))
                        except subprocess.TimeoutExpired:
                            observed.add("child_timeout")
            except (KeyboardInterrupt, SystemExit):
                observed.add("root_custody_lost")
            finally:
                try:
                    _stop_group(child)
                    # Preserve buffered bytes after termination, still enforcing
                    # both hard caps. The group is gone, so EOF is authoritative.
                    for index, stream in enumerate(streams):
                        while sizes[index] < log_limit_bytes:
                            chunk = os.read(
                                stream.fileno(),
                                min(65_536, log_limit_bytes - sizes[index]),
                            )
                            if not chunk:
                                break
                            logs[index].write(chunk)
                            sizes[index] += len(chunk)
                        if sizes[index] == log_limit_bytes:
                            observed.add("child_log_limit_exceeded")
                finally:
                    for stream in streams:
                        stream.close()
            process["exit_code"] = child.returncode
            if child.returncode is not None and child.returncode < 0:
                process["termination_signal"] = -child.returncode
                observed.add("child_signaled")
            elif child.returncode:
                observed.add("child_nonzero_exit")
    finally:
        cleanup_errors: list[BaseException] = []
        try:
            if child is not None and child.poll() is None:
                _stop_group(child)
        except BaseException as reason:
            cleanup_errors.append(reason)
        for handle in logs:
            try:
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException as reason:
                cleanup_errors.append(reason)
            finally:
                handle.close()
        try:
            fsync_directory(stdout.parent)
        except BaseException as reason:
            cleanup_errors.append(reason)
        if cleanup_errors:
            raise BaseExceptionGroup("sealed process cleanup failed", cleanup_errors)
    if process["started"]:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        process["wall_elapsed_ns"] = time.perf_counter_ns() - started_at
        process["child_user_cpu_ns"] = max(
            0, round((after.ru_utime - before.ru_utime) * 1e9)
        )
        process["child_system_cpu_ns"] = max(
            0, round((after.ru_stime - before.ru_stime) * 1e9)
        )
    return process, observed

"""One user-host study lock held by the live sequential parent, never a PID cache."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat
import tempfile
from collections.abc import Iterator
from typing import Any

from ..training import custody_io as io
from ..training.custody_validation import exact


def lock_path() -> Path:
    return (
        Path(tempfile.gettempdir()).resolve() / "plant-beta-constrained-study-v1.lock"
    )


def _identity(path: Path) -> dict[str, Any]:
    io.require_directory(path.parent)
    value = path.lstat()
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_nlink != 1
        or value.st_uid != os.getuid()
    ):
        raise ValueError("study lock is not a regular owned file")
    return {
        "path": str(path),
        "device": value.st_dev,
        "inode": value.st_ino,
        "uid": value.st_uid,
        "mode": value.st_mode,
    }


def _check_descriptor(descriptor: int, identity: dict[str, Any]) -> None:
    value = os.fstat(descriptor)
    if (
        value.st_dev != identity["device"]
        or value.st_ino != identity["inode"]
        or value.st_uid != identity["uid"]
        or value.st_mode != identity["mode"]
        or value.st_nlink != 1
    ):
        raise ValueError("study lock was replaced while opening")


def verify_lock_identity(raw: dict[str, Any]) -> None:
    exact("study lock identity", raw, _identity(lock_path()))


@contextmanager
def acquire() -> Iterator[dict[str, Any]]:
    path = lock_path()
    io.require_directory(path.parent)
    descriptor = os.open(
        path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
    )
    try:
        identity = _identity(path)
        _check_descriptor(descriptor, identity)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.fsync(descriptor)
        io.fsync_directory(path.parent)
        yield identity
        _check_descriptor(descriptor, identity)
        verify_lock_identity(identity)
    finally:
        os.close(descriptor)


def require_locked(raw: dict[str, Any]) -> None:
    """Child verifies that another process still holds the recorded lock."""
    verify_lock_identity(raw)
    descriptor = os.open(lock_path(), os.O_RDWR | os.O_NOFOLLOW)
    try:
        _check_descriptor(descriptor, raw)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            verify_lock_identity(raw)
            return
        raise ValueError("stage parent no longer holds its study lock")
    finally:
        os.close(descriptor)

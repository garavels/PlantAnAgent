"""Exclusive durable publications and complete, non-following artifact walks."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
from typing import Any

from .custody_json import canonical_bytes, domain_digest

INVENTORY_DOMAIN = "plant-beta-constrained-rl-registered-training-artifact-inventory-v1"


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def require_directory(path: Path) -> None:
    """Check every existing path component, including the root, without resolution."""
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("custody path must be absolute without traversal")
    for parent in reversed((path, *path.parents)):
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise ValueError("custody directory is not a regular nonsymlink directory")


def fsync_directory(path: Path) -> None:
    require_directory(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def mkdir_exclusive(path: Path) -> None:
    require_directory(path.parent)
    path.mkdir(mode=0o700, exist_ok=False)
    fsync_directory(path)
    fsync_directory(path.parent)


def publish_bytes(path: Path, raw: bytes) -> None:
    """Consume a filename permanently, including when a write or fsync fails."""
    require_directory(path.parent)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(path.parent)


def publish_json(path: Path, value: object) -> None:
    publish_bytes(path, canonical_bytes(value))


def read_regular(path: Path) -> bytes:
    require_directory(path.parent)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("custody file must be regular, nonsymlink and singly linked")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as handle:
        if _identity(os.fstat(handle.fileno())) != _identity(before):
            raise ValueError("custody file changed while opening")
        raw = handle.read()
        if _identity(os.fstat(handle.fileno())) != _identity(before):
            raise ValueError("custody file changed while reading")
    if len(raw) != before.st_size or _identity(path.lstat()) != _identity(before):
        raise ValueError("custody file changed while closing")
    return raw


def file_record(path: Path) -> dict[str, Any]:
    before = path.lstat()
    raw = read_regular(path)
    if _identity(path.lstat()) != _identity(before):
        raise ValueError("custody file changed while describing")
    return {
        "path": str(path),
        "logical_bytes": len(raw),
        "allocated_bytes": before.st_blocks * 512,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def inventory(root: Path) -> dict[str, Any]:
    """Retain all entries, including forbidden ones. An unstable walk raises.

    A forbidden root is represented by a single '.' entry. A missing root is
    distinct from an empty directory. No content is deleted or repaired here.
    """
    require_directory(root.parent)
    entries: list[dict[str, Any]] = []

    def visit(path: Path) -> None:
        before = path.lstat()
        name = path.relative_to(root).as_posix()
        record: dict[str, Any] = {
            "path": name,
            "node_type": "forbidden",
            "logical_bytes": before.st_size,
            "allocated_bytes": before.st_blocks * 512,
            "sha256": None,
        }
        if stat.S_ISDIR(before.st_mode):
            children = sorted(path.iterdir(), key=lambda child: os.fsencode(child.name))
            if path != root:
                entries.append(
                    {
                        **record,
                        "node_type": "directory",
                        "logical_bytes": None,
                        "allocated_bytes": None,
                    }
                )
            for child in children:
                visit(child)
            if _identity(path.lstat()) != _identity(before):
                raise ValueError("artifact directory changed during inventory")
            return
        if (
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and before.st_mode & 0o444
        ):
            try:
                raw = read_regular(path)
            except PermissionError:
                pass
            else:
                record["node_type"] = "regular_file"
                record["sha256"] = hashlib.sha256(raw).hexdigest()
        if _identity(path.lstat()) != _identity(before):
            raise ValueError("artifact entry changed during inventory")
        entries.append(record)

    try:
        root.lstat()
    except FileNotFoundError:
        exists = False
    else:
        exists = True
        visit(root)
    entries.sort(key=lambda entry: os.fsencode(entry["path"]))
    return {
        "root": str(root),
        "exists": exists,
        "entries": entries,
        "regular_file_count": sum(
            item["node_type"] == "regular_file" for item in entries
        ),
        "directory_count": sum(item["node_type"] == "directory" for item in entries),
        "forbidden_count": sum(item["node_type"] == "forbidden" for item in entries),
        "logical_bytes": sum(item["logical_bytes"] or 0 for item in entries),
        "allocated_bytes": sum(item["allocated_bytes"] or 0 for item in entries),
        "tree_sha256": domain_digest(INVENTORY_DOMAIN, entries),
    }


def require_new_root(root: Path, repository: Path, minimum_free_bytes: int) -> None:
    if not root.is_absolute() or root != root.resolve():
        raise ValueError("campaign root must be absolute, canonical and nonsymlink")
    if root.is_relative_to(repository.resolve()):
        raise ValueError("campaign root must be external to the repository")
    require_directory(root.parent)
    if os.path.lexists(root):
        raise ValueError("campaign root already exists; attempts cannot be retried")
    if not os.access(root.parent, os.W_OK | os.X_OK):
        raise ValueError("campaign parent is not writable")
    filesystem = os.statvfs(root.parent)
    if filesystem.f_bavail * filesystem.f_frsize < minimum_free_bytes:
        raise ValueError("campaign storage budget is unavailable")


def budget_available(root: Path, minimum_free_bytes: int) -> bool:
    require_directory(root)
    retained = inventory(root)
    filesystem = os.statvfs(root)
    return bool(
        filesystem.f_bavail * filesystem.f_frsize + retained["allocated_bytes"]
        >= minimum_free_bytes
    )

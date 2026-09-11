"""Byte-preserving storage fixtures; no controller, checkpoint or plant access."""

import gzip
import hashlib
import os
from pathlib import Path
import stat
from typing import Any

import pytest

from plant_backend.rl.v2.study import archive
from plant_backend.rl.v2.training.custody_json import canonical_bytes

LIMITS = {"stored_limit": 16_000_000, "uncompressed_limit": 16_000_000}


def packed(raw: bytes) -> bytes:
    # Independent standard-library encoder; normalize its platform header.
    encoded = gzip.compress(raw, compresslevel=1, mtime=0)
    return archive.HEADER + encoded[10:]


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"unicode": "🌱 météo / \ud800", "array": [1, -0.0, True, None, 1e-210]},
        {"nested": {"b": [False, {"a": 3.14}], "a": 'quotes"\n\t'}},
        {"long": "x" * (archive.CHUNK * 5 + 31)},
    ],
)
def test_lossless_bytes_and_deterministic_exclusive_publication(
    tmp_path: Path, value: dict[str, Any]
) -> None:
    plain = canonical_bytes(value)
    assert b"".join(archive.canonical_chunks(value)) == plain
    path = tmp_path / "arm.json.gz"
    record = archive.publish_archive(path, value, **LIMITS)
    encoded = path.read_bytes()
    assert gzip.decompress(encoded) == plain
    assert encoded[:10] == archive.HEADER
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    decoded, reopened = archive.read_archive(path, **LIMITS)
    assert decoded == value and reopened == record
    assert record["uncompressed_sha256"] == hashlib.sha256(plain).hexdigest()
    assert record["uncompressed_bytes"] == len(plain)
    assert record["file"]["sha256"] == hashlib.sha256(encoded).hexdigest()
    second = tmp_path / "second.json.gz"
    archive.publish_archive(second, value, **LIMITS)
    assert second.read_bytes() == encoded
    with pytest.raises(FileExistsError):
        archive.publish_archive(path, {"replaced": True}, **LIMITS)
    assert path.read_bytes() == encoded


def test_independent_member_and_exact_inclusive_limits(tmp_path: Path) -> None:
    plain = canonical_bytes({"body": [str(i) for i in range(20_000)]})
    path = tmp_path / "arm.json.gz"
    path.write_bytes(packed(plain))
    limits = {"stored_limit": path.stat().st_size, "uncompressed_limit": len(plain)}
    assert archive.read_archive(path, **limits)[0]["body"][-1] == "19999"
    for key in limits:
        with pytest.raises(ValueError, match="byte limit"):
            archive.read_archive(path, **{**limits, key: limits[key] - 1})


@pytest.mark.parametrize("key", ["stored_limit", "uncompressed_limit"])
@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "100", None])
def test_caps_have_exact_types_before_consuming_destination(
    tmp_path: Path, key: str, bad: object
) -> None:
    path = tmp_path / "arm.json.gz"
    with pytest.raises(ValueError, match="positive integer"):
        archive.publish_archive(path, {}, **{**LIMITS, key: bad})
    assert not path.exists()
    with pytest.raises(ValueError, match="positive integer"):
        archive.read_archive(path, **{**LIMITS, key: bad})


@pytest.mark.parametrize(
    "plain",
    [
        b"{}",
        b'{"x": 1}\n',
        b'{"x":1,"x":2}',
        b'{"x":NaN}',
        b'{"x":1e999}',
        b"[]\n",
        b"\xff",
        b'{"outer":{"a":1,"a":2}}',
    ],
)
def test_noncanonical_ambiguous_or_nonfinite_json_rejected(
    tmp_path: Path, plain: bytes
) -> None:
    path = tmp_path / "arm.json.gz"
    path.write_bytes(packed(plain))
    with pytest.raises(ValueError):
        archive.read_archive(path, **LIMITS)


@pytest.mark.parametrize(
    "defect", ["crc", "size", "header", "empty", "truncated", "tail", "member"]
)
def test_corruption_truncation_and_concatenation_rejected(
    tmp_path: Path, defect: str
) -> None:
    raw = packed(canonical_bytes({"repeated": "x" * (archive.CHUNK * 3)}))
    if defect in {"crc", "size", "header"}:
        index = {"crc": -8, "size": -4, "header": 4}[defect]
        changed = bytearray(raw)
        changed[index] ^= 1
        raw = bytes(changed)
    elif defect == "empty":
        raw = b""
    elif defect == "truncated":
        raw = raw[:-1]
    elif defect == "tail":
        raw += b"\0"
    else:
        raw += raw
    path = tmp_path / "arm.json.gz"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        archive.read_archive(path, **LIMITS)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("key", ["stored_limit", "uncompressed_limit"])
def test_failed_write_retains_consumed_filename(tmp_path: Path, key: str) -> None:
    path = tmp_path / "arm.json.gz"
    with pytest.raises(ValueError, match="byte limit"):
        archive.publish_archive(path, {"retained": "x" * 400}, **{**LIMITS, key: 11})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        archive.publish_archive(path, {}, **LIMITS)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "defect", ["symlink", "hardlink", "directory", "parent_symlink"]
)
def test_file_and_directory_indirection_is_rejected(
    tmp_path: Path, defect: str
) -> None:
    original = tmp_path / "original.json.gz"
    original.write_bytes(packed(canonical_bytes({})))
    target = tmp_path / "arm.json.gz"
    if defect == "symlink":
        target.symlink_to(original)
    elif defect == "hardlink":
        os.link(original, target)
    elif defect == "directory":
        target.mkdir()
    else:
        parent = tmp_path / "link"
        parent.symlink_to(tmp_path, target_is_directory=True)
        target = parent / original.name
    with pytest.raises((ValueError, OSError)):
        archive.read_archive(target, **LIMITS)


@pytest.mark.parametrize("operation", ["read", "publish"])
def test_identical_byte_replacement_is_not_stable_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    path = tmp_path / "arm.json.gz"
    if operation == "read":
        archive.publish_archive(path, {}, **LIMITS)
    describe = archive._descriptor

    def replace(target: Path, **kwargs: Any) -> dict[str, Any]:
        record = describe(target, **kwargs)
        replacement = tmp_path / "replacement"
        replacement.write_bytes(target.read_bytes())
        replacement.replace(target)
        return record

    monkeypatch.setattr(archive, "_descriptor", replace)
    with pytest.raises(ValueError, match="changed during"):
        if operation == "read":
            archive.read_archive(path, **LIMITS)
        else:
            archive.publish_archive(path, {}, **LIMITS)


def test_stored_read_is_bounded_even_when_file_grows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "arm.json.gz"
    path.write_bytes(packed(canonical_bytes({})))
    original = os.fdopen
    seen: list[int] = []

    class Reader:
        def __init__(self, descriptor: int):
            self.handle = original(descriptor, "rb")

        def __enter__(self) -> "Reader":
            return self

        def __exit__(self, *args: object) -> None:
            self.handle.close()

        def fileno(self) -> int:
            return self.handle.fileno()

        def read(self, size: int) -> bytes:
            seen.append(size)
            with path.open("ab") as writer:
                writer.write(b"x" * 1_000)
            return self.handle.read(size)

    monkeypatch.setattr(archive.os, "fdopen", lambda fd, mode: Reader(fd))
    with pytest.raises(ValueError, match="stored byte limit"):
        archive.read_archive(path, stored_limit=100, uncompressed_limit=1_000)
    assert seen == [101]


def test_invalid_document_and_filename_are_preflight_errors(tmp_path: Path) -> None:
    path = tmp_path / "arm.json.gz"
    with pytest.raises(ValueError, match="object"):
        archive.publish_archive(path, [], **LIMITS)
    assert not path.exists()
    with pytest.raises(ValueError, match="filename"):
        archive.publish_archive(tmp_path / "arm.json", {}, **LIMITS)
    with pytest.raises(ValueError, match="filename"):
        archive.read_archive(tmp_path / "arm.json", **LIMITS)

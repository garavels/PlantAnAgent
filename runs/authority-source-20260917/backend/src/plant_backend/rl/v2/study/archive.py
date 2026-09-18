"""Bounded, lossless canonical-JSON archives with durable exclusive publication."""

from __future__ import annotations

from collections.abc import Iterator
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
from typing import Any
import zlib

from ..training import custody_io as io
from ..training.custody_json import decode
from .contract import PREFIX

SCHEMA = PREFIX + "canonical-json-gzip-v1"
HEADER = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x04\xff"
CHUNK = 1_048_576


def _limit(value: int, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer byte limit")
    return value


def canonical_chunks(value: object) -> Iterator[bytes]:
    """Exactly the custody canonical encoding, without a second complete string."""
    encoder = json.JSONEncoder(
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        separators=(",", ": "),
        sort_keys=True,
    )
    for part in encoder.iterencode(value):
        # Large individual strings are bounded at the compressor boundary too.
        for start in range(0, len(part), CHUNK):
            yield part[start : start + CHUNK].encode("ascii")
    yield b"\n"


def _stamp(path: Path) -> tuple[int, ...]:
    value = path.lstat()
    return (*io._identity(value), value.st_uid, value.st_gid)


def _read_stored(path: Path, limit: int) -> bytes:
    """Enforce the stored cap on the descriptor, including concurrent growth."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("study archive must be regular and singly linked")
    if before.st_size > limit:
        raise ValueError("study archive exceeds its stored byte limit")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as handle:
        if io._identity(os.fstat(handle.fileno())) != io._identity(before):
            raise ValueError("study archive changed while opening")
        raw = handle.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("study archive exceeds its stored byte limit")
        if io._identity(os.fstat(handle.fileno())) != io._identity(before):
            raise ValueError("study archive changed while reading")
    if len(raw) != before.st_size or io._identity(path.lstat()) != io._identity(before):
        raise ValueError("study archive changed while closing")
    return raw


def _descriptor(
    path: Path, *, plain_bytes: int, plain_sha256: str, stored_limit: int
) -> dict[str, Any]:
    before = path.lstat()
    raw = _read_stored(path, stored_limit)
    if io._identity(path.lstat()) != io._identity(before):
        raise ValueError("study archive changed while describing")
    return {
        "schema": SCHEMA,
        "file": {
            "path": str(path),
            "logical_bytes": len(raw),
            "allocated_bytes": before.st_blocks * 512,
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "uncompressed_bytes": plain_bytes,
        "uncompressed_sha256": plain_sha256,
    }


def publish_archive(
    path: Path | str, value: object, *, stored_limit: int, uncompressed_limit: int
) -> dict[str, Any]:
    """Consume the filename once, retaining any partial failed publication."""
    target = Path(path)
    _limit(stored_limit, "stored limit")
    _limit(uncompressed_limit, "uncompressed limit")
    if not isinstance(value, dict):
        raise ValueError("study archive document must be an object")
    if target.suffixes[-2:] != [".json", ".gz"]:
        raise ValueError("study archive filename must end in .json.gz")
    io.require_directory(target.parent)
    descriptor = os.open(
        target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    plain_hash = hashlib.sha256()
    plain_size = 0
    stored_size = 0
    stored_hash = hashlib.sha256()
    crc = 0
    compressor = zlib.compressobj(level=1, method=zlib.DEFLATED, wbits=-zlib.MAX_WBITS)
    try:
        with os.fdopen(descriptor, "wb") as handle:

            def write(raw: bytes) -> None:
                nonlocal stored_size
                if stored_size + len(raw) > stored_limit:
                    raise ValueError("study archive exceeds its stored byte limit")
                handle.write(raw)
                stored_size += len(raw)
                stored_hash.update(raw)

            try:
                write(HEADER)
                for raw in canonical_chunks(value):
                    if plain_size + len(raw) > uncompressed_limit:
                        raise ValueError(
                            "study archive exceeds its uncompressed byte limit"
                        )
                    plain_size += len(raw)
                    plain_hash.update(raw)
                    crc = zlib.crc32(raw, crc)
                    write(compressor.compress(raw))
                write(compressor.flush())
                write(struct.pack("<II", crc & 0xFFFFFFFF, plain_size & 0xFFFFFFFF))
            finally:
                handle.flush()
                os.fsync(handle.fileno())
                finished = os.fstat(handle.fileno())
    finally:
        io.fsync_directory(target.parent)
    expected_stamp = (*io._identity(finished), finished.st_uid, finished.st_gid)
    record = _descriptor(
        target,
        plain_bytes=plain_size,
        plain_sha256=plain_hash.hexdigest(),
        stored_limit=stored_limit,
    )
    if (
        _stamp(target) != expected_stamp
        or record["file"]["logical_bytes"] != stored_size
        or record["file"]["sha256"] != stored_hash.hexdigest()
    ):
        raise ValueError("study archive changed during publication")
    return record


def read_archive(
    path: Path | str, *, stored_limit: int, uncompressed_limit: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decode one bounded member; reject corruption, trailing data and ambiguity."""
    target = Path(path)
    _limit(stored_limit, "stored limit")
    _limit(uncompressed_limit, "uncompressed limit")
    if target.suffixes[-2:] != [".json", ".gz"]:
        raise ValueError("study archive filename must end in .json.gz")
    io.require_directory(target.parent)
    before = _stamp(target)
    compressed = _read_stored(target, stored_limit)
    if not compressed.startswith(HEADER):
        raise ValueError("study archive header differs from the fixed encoding")
    decompressor = zlib.decompressobj(wbits=31)
    chunks = []
    plain_size = 0
    plain_hash = hashlib.sha256()
    try:
        for offset in range(0, len(compressed), CHUNK):
            pending = compressed[offset : offset + CHUNK]
            while pending:
                decoded = decompressor.decompress(
                    pending, min(CHUNK, uncompressed_limit - plain_size + 1)
                )
                if decompressor.unused_data:
                    raise ValueError(
                        "study archive has trailing data or multiple members"
                    )
                plain_size += len(decoded)
                if plain_size > uncompressed_limit:
                    raise ValueError(
                        "study archive exceeds its uncompressed byte limit"
                    )
                chunks.append(decoded)
                plain_hash.update(decoded)
                pending = decompressor.unconsumed_tail
        if not decompressor.eof:
            raise ValueError("study archive is truncated")
    except zlib.error as reason:
        raise ValueError("study archive compression or checksum is invalid") from reason
    payload = decode(b"".join(chunks))
    canonical_hash = hashlib.sha256()
    canonical_size = 0
    for raw in canonical_chunks(payload):
        canonical_size += len(raw)
        canonical_hash.update(raw)
    if canonical_size != plain_size or canonical_hash.digest() != plain_hash.digest():
        raise ValueError("study archive JSON is not the exact canonical encoding")
    record = _descriptor(
        target,
        plain_bytes=plain_size,
        plain_sha256=plain_hash.hexdigest(),
        stored_limit=stored_limit,
    )
    if (
        _stamp(target) != before
        or record["file"]["sha256"] != hashlib.sha256(compressed).hexdigest()
    ):
        raise ValueError("study archive changed during verification")
    return payload, record

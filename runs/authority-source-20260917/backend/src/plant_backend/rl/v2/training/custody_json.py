"""Strict JSON and domain identities shared by registered custody boundaries.

This module imports no learner or environment. Decode before inspecting run kind:
untrusted duplicate keys must never choose which admission rules apply.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, NoReturn


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite JSON number is forbidden")
    return result


def decode(raw: bytes | str) -> dict[str, Any]:
    """Decode a finite object, rejecting duplicates at every nesting depth."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    value = json.loads(
        raw, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float
    )
    if not isinstance(value, dict):
        raise ValueError("JSON document must be an object")
    return value


def canonical_bytes(value: object) -> bytes:
    """The protocol's complete, pretty-printed publication encoding."""
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            separators=(",", ": "),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def domain_digest(domain: str, value: object) -> str:
    payload = json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(domain.encode("ascii") + b"\0" + payload).hexdigest()


def authenticate(value: dict[str, Any]) -> dict[str, Any]:
    """Return a fresh signed envelope; hashes establish integrity, not authority."""
    payload = {key: item for key, item in value.items() if key != "sha256"}
    return {**payload, "sha256": domain_digest(payload["schema"], payload)}


def require_envelope(
    raw: bytes, *, schema: str, fields: set[str] | frozenset[str]
) -> dict[str, Any]:
    value = decode(raw)
    if value.get("schema") != schema:
        raise ValueError("custody document schema differs")
    if set(value) != fields:
        raise ValueError("custody document fields differ")
    if raw != canonical_bytes(value):
        raise ValueError("custody document is not canonical")
    if value != authenticate(value):
        raise ValueError("custody document digest differs")
    return value

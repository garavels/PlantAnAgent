"""Integrity-checked public fixture recordings, independent of research reports."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from fastapi import HTTPException

RECORDING_ROOT = Path(__file__).resolve().parents[3] / "runs" / "commission-demo"
SCHEMA = "plant-public-recording-v1"
_ID = re.compile(r"[a-f0-9-]{36}\Z")


def digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def seal_recording(
    snapshot: dict[str, Any], frames: list[dict[str, Any]]
) -> dict[str, Any]:
    payload = {
        "schema": SCHEMA,
        "snapshot": deepcopy(snapshot),
        "frames": deepcopy(frames),
        "evidence_level": "disposable fixture; checksum integrity only, not a verified research result",
    }
    return {**payload, "sha256": digest(payload)}


def validate(record: dict[str, Any]) -> dict[str, Any]:
    if (
        set(record) != {"schema", "snapshot", "frames", "evidence_level", "sha256"}
        or record["schema"] != SCHEMA
    ):
        raise ValueError("unsupported recording schema")
    payload = {k: v for k, v in record.items() if k != "sha256"}
    if digest(payload) != record["sha256"]:
        raise ValueError("recording checksum mismatch")
    snap, frames = record["snapshot"], record["frames"]
    if (
        snap["environment"]["fixture_seed"] != 9_000_034
        or snap["mode"] != "interactive_fixture"
    ):
        raise ValueError("only disposable demo fixtures can be replayed here")
    if not 1 <= len(frames) <= 673 or frames[-1] != snap["frame"]:
        raise ValueError("incomplete recording boundaries")
    if len(snap["history"]) != len(frames):
        raise ValueError("recording history mismatch")
    for index, frame in enumerate(frames):
        if (
            frame["clock"]["decision_index"] != index
            or frame["clock"]["elapsed_hours"] != index / 4
        ):
            raise ValueError("recording decision sequence mismatch")
    return record


def summarize(record: dict[str, Any], source: str) -> dict[str, Any]:
    snap = record["snapshot"]
    return {
        "id": snap["run_id"],
        "controller": snap["controller"]["id"],
        "hours": snap["frame"]["clock"]["elapsed_hours"],
        "status": snap["status"],
        "source": source,
        "sha256": record["sha256"],
        "available": True,
        "delivered_kg": snap["frame"]["delivery"]["delivered_kg"],
        "unmet_kg": snap["frame"]["delivery"]["unmet_kg"],
        "net_value_eur": snap["frame"]["economics"]["net_value_eur"],
        "fallback_decisions": sum(
            (f.get("decision") or {}).get("operation", {}).get("mode") == "fallback"
            for f in record["frames"]
            if (f.get("decision") or {}).get("operation")
        ),
        "recovery_decisions": sum(
            (f.get("decision") or {}).get("operation", {}).get("mode") == "recovery"
            for f in record["frames"]
            if (f.get("decision") or {}).get("operation")
        ),
        "failed_cycles": sum(
            not c["feasible"] for c in snap["frame"]["cycle_settlements"]
        ),
    }


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128_000_000:
        raise ValueError("unavailable recording file")
    record = validate(json.loads(path.read_text()))
    if record["snapshot"]["run_id"] != path.stem:
        raise ValueError("recording identity differs from filename")
    return record


def catalog() -> list[dict[str, Any]]:
    results = []
    for path in sorted(
        p for p in RECORDING_ROOT.glob("*.json") if _ID.fullmatch(p.stem)
    )[:8]:
        try:
            results.append(summarize(_read(path), "saved"))
        except (ValueError, KeyError, TypeError, OSError):
            results.append(
                {
                    "id": path.stem,
                    "available": False,
                    "source": "saved",
                    "status": "invalid",
                }
            )
    return results


def read_recording(identifier: str) -> dict[str, Any]:
    if not _ID.fullmatch(identifier):
        raise HTTPException(status_code=404, detail="Unknown recording")
    try:
        return _read(RECORDING_ROOT / f"{identifier}.json")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise HTTPException(
            status_code=404, detail="Recording missing or failed integrity checks"
        ) from exc

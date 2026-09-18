"""Read-only fixture recordings and separately admitted research playback."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from . import research_playback

RECORDING_ROOT = Path(__file__).resolve().parents[3] / "runs" / "commission-demo"
RESEARCH_RECORDING_ROOT = Path(__file__).resolve().parents[3] / "demo" / "recordings"
SCHEMA = "plant-public-recording-v1"
SITE_SCHEMA = "plant-site-recording-v1"
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
    site = snapshot.get("mode") == "site_forecast"
    if site:
        from .site_workspace import EVIDENCE_LEVEL
    payload = {
        "schema": SITE_SCHEMA if site else SCHEMA,
        "snapshot": deepcopy(snapshot),
        "frames": deepcopy(frames),
        "evidence_level": (
            EVIDENCE_LEVEL
            if site
            else "disposable fixture; checksum integrity only, not a verified research result"
        ),
    }
    return {**payload, "sha256": digest(payload)}


def seal_research_recording(
    snapshot: dict[str, Any],
    frames: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Seal an export candidate; only validate() grants playback admission."""
    payload = {
        "schema": research_playback.SCHEMA,
        "snapshot": deepcopy(snapshot),
        "frames": deepcopy(frames),
        "provenance": deepcopy(provenance),
        "evidence_level": research_playback.EVIDENCE_LEVEL,
    }
    return {**payload, "sha256": digest(payload)}


def validate(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("unsupported recording schema")
    research = record.get("schema") == research_playback.SCHEMA
    fields = {"schema", "snapshot", "frames", "evidence_level", "sha256"}
    if research:
        fields.add("provenance")
    if set(record) != fields or record["schema"] not in (
        SCHEMA,
        SITE_SCHEMA,
        research_playback.SCHEMA,
    ):
        raise ValueError("unsupported recording schema")
    payload = {k: v for k, v in record.items() if k != "sha256"}
    if digest(payload) != record["sha256"]:
        raise ValueError("recording checksum mismatch")
    if research:
        return research_playback.validate_research(record)
    if record["schema"] == SITE_SCHEMA:
        from .site_workspace import validate_site_recording

        return validate_site_recording(record)
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
        **(
            {
                "mode": "recorded_research",
                "featured": True,
                "title": "Successful trained PPO week",
                "provenance": deepcopy(record["provenance"]),
            }
            if record["schema"] == research_playback.SCHEMA
            else {}
        ),
        **(
            {
                "mode": "site_forecast",
                "title": (snap.get("site_parameters", {}).get("scenario") or {}).get(
                    "name", snap["site_forecast"]["site"]["name"]
                ),
                "forecast_id": snap["site_forecast"]["id"],
            }
            if record["schema"] == SITE_SCHEMA
            else {}
        ),
    }


def _read(path: Path, *, research_only: bool = False) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128_000_000:
        raise ValueError("unavailable recording file")
    record = validate(json.loads(path.read_text()))
    if research_only and record["schema"] != research_playback.SCHEMA:
        raise ValueError("research path requires approved research evidence")
    if record["snapshot"]["run_id"] != path.stem:
        raise ValueError("recording identity differs from filename")
    return record


def catalog() -> list[dict[str, Any]]:
    results = []
    paths = sorted(
        p
        for p in RECORDING_ROOT.glob("*.json")
        if _ID.fullmatch(p.stem) and p.stem != research_playback.APPROVED_RECORDING_ID
    )[:8]
    # The reviewed research asset has its own fixed path and admission gate;
    # fixture filename ordering cannot hide it behind the bounded legacy list.
    research_path = (
        RESEARCH_RECORDING_ROOT / f"{research_playback.APPROVED_RECORDING_ID}.json"
    )
    if research_playback.APPROVED_RECORDING_ID and research_path.exists():
        paths.insert(0, research_path)
    for path in paths:
        try:
            results.append(
                summarize(_read(path, research_only=path == research_path), "saved")
            )
        except (ValueError, KeyError, TypeError, AttributeError, OSError):
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
        root = (
            RESEARCH_RECORDING_ROOT
            if identifier == research_playback.APPROVED_RECORDING_ID
            else RECORDING_ROOT
        )
        return _read(
            root / f"{identifier}.json",
            research_only=identifier == research_playback.APPROVED_RECORDING_ID,
        )
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
        raise HTTPException(
            status_code=404, detail="Recording missing or failed integrity checks"
        ) from exc

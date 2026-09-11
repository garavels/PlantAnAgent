"""Summarize a saved commission rehearsal without a reset, solver or replay.

Usage: python docs/research/tools/commission_operating_result.py MARKER RECORD OUTPUT
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

from plant_backend.demo_recordings import summarize, validate


def build(marker_path: Path, record_path: Path) -> dict[str, Any]:
    marker = json.loads(marker_path.read_text())
    record_bytes = record_path.read_bytes()
    record = validate(json.loads(record_bytes))
    snapshot = record["snapshot"]
    summary = summarize(record, "saved")
    if marker["controller"] != "MPC" or marker["git_dirty"]:
        raise ValueError("expected clean-source product MPC rehearsal")
    if any(marker[key] != value for key, value in summary.items()):
        raise ValueError("attempt manifest differs from saved recording")
    decisions = [frame["decision"] for frame in record["frames"] if frame["decision"]]
    fallbacks = []
    for index, decision in enumerate(decisions):
        operation = decision["operation"]
        if operation["mode"] != "fallback":
            continue
        fallbacks.append(
            {
                "issued_at_hours": decision["at_hours"],
                "completed_boundary": index + 1,
                "solver_failure": operation["solver_failure"],
                "following_mode": (
                    decisions[index + 1]["operation"]["mode"]
                    if index + 1 < len(decisions)
                    else None
                ),
            }
        )
    latencies = [decision["computation_ms"] for decision in decisions]
    frame = snapshot["frame"]
    return {
        "schema": "commission-operating-rehearsal-result-v1",
        "evidence_level": record["evidence_level"],
        "attempt": marker,
        "recording": {
            "repository_relative_path": f"runs/commission-demo/{record_path.name}",
            "bytes": len(record_bytes),
            "file_sha256": hashlib.sha256(record_bytes).hexdigest(),
            "content_sha256": record["sha256"],
            "attempt_file_sha256": hashlib.sha256(marker_path.read_bytes()).hexdigest(),
        },
        "controller": snapshot["controller"],
        "frame_count": len(record["frames"]),
        "mode_counts": dict(Counter(d["operation"]["mode"] for d in decisions)),
        "computation_ms": {
            "median": statistics.median(latencies) if latencies else None,
            "maximum": max(latencies) if latencies else None,
        },
        "fallbacks": fallbacks,
        "metrics": frame["metrics"],
        "final_resources": frame["resources"],
        "cycle_settlements": frame["cycle_settlements"],
        "economics": frame["economics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("marker", type=Path)
    parser.add_argument("recording", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = build(args.marker, args.recording)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()

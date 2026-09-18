#!/usr/bin/env python3
"""Refresh the inline inspection fragment with exact saved baseline trajectories.

The durable fragment is the literal presentation template. This builder only
replaces its marked JSON data slot, so it can be rerun as recordings finish.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO / "docs/research/results/feasible_mission_20260915/verified"
DEFAULT_FRAGMENT = Path(
    "/Users/filippogombac/.codex/visualizations/2026/09/15/01a0a466-8883-7cb1-8cef-99e03183ca0b/feasible-mission.html"
)
SEEDS = (*range(9700300, 9700308), 9000034)
COLUMNS = [
    "battery_energy_kwh",
    "h2_level_kg",
    "co2_level_kg",
    "ch4_level_kg",
    "solar_available_kw",
    "electrolyser_power_kw",
    "dac_power_kw",
    "methanator_power_kw",
    "battery_power_kw",
]


def build(input_dir: Path, fragment: Path, *, require_all: bool = False):
    cases = []
    for seed in SEEDS:
        path = input_dir / f"coverage-25-{seed}.json.gz"
        if not path.exists():
            if require_all:
                raise ValueError(f"Missing final recording: {path}")
            continue
        raw = path.read_bytes()
        data = json.loads(gzip.decompress(raw))
        if not data["execution_complete"] or len(data["decision_frames"]) != 673:
            raise ValueError(f"Incomplete source: {path}")
        if data["seed"] != seed or data["delivery_rate_kg_h"] != 25:
            raise ValueError(f"Source identity/rate mismatch: {path}")
        frames = [
            [round(f["state"][k], 5) for k in COLUMNS] for f in data["decision_frames"]
        ]
        settlements = data["cycle_settlements"]
        targets = settlements[0]["targets"]
        cases.append(
            {
                "seed": seed,
                "controller": data["controller"],
                "mission": data["mission_passed"],
                "safety": data["hard_safety_passed"],
                "forcedStops": data["physical_metrics"]["electrolyser_forced_stops"],
                "delivered": data["physical_metrics"]["delivered_kg"],
                "requested": data["physical_metrics"]["requested_kg"],
                "targets": [targets[k] for k in ["battery", "h2", "co2", "ch4"]],
                "cycles": [
                    {
                        "day": s["cycle_index"],
                        "pass": s["feasible"],
                        "deficits": s["deficits"],
                    }
                    for s in settlements
                ],
                "deliveries": [
                    {
                        "hour": s["due_step"] / 60,
                        "requested": s["requested_kg"],
                        "delivered": s["delivered_kg"],
                        "pass": s["unmet_kg"] <= data["delivery_unmet_tolerance_kg"],
                    }
                    for s in data["delivery"]["settlements"]
                ],
                "frames": frames,
                "source": {
                    "path": str(path),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                },
            }
        )
    if not cases:
        raise ValueError("No completed recordings available")
    packed = json.dumps(
        {"columns": COLUMNS, "cases": cases}, separators=(",", ":"), allow_nan=False
    ).replace("<", "\\u003c")
    template = fragment.read_text()
    pattern = r"(?<=<!--FEASIBLE_DATA_BEGIN-->)[\s\S]*?(?=<!--FEASIBLE_DATA_END-->)"
    replacement = (
        '\n<script id="fm-data" type="application/json">' + packed + "</script>\n"
    )
    updated, count = re.subn(pattern, lambda _: replacement, template)
    if count != 1:
        raise ValueError("Expected one data slot in fragment")
    if len(updated.encode()) >= 1_000_000:
        raise ValueError("Inline fragment exceeds 1 MB")
    fragment.write_text(updated)
    return {
        "fragment": str(fragment),
        "bytes": len(updated.encode()),
        "case_count": len(cases),
        "passed_cases": sum(c["mission"] for c in cases),
        "source_files": [c["source"] for c in cases],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--fragment", type=Path, default=DEFAULT_FRAGMENT)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.input_dir, args.fragment, require_all=args.require_all), indent=2
        )
    )


if __name__ == "__main__":
    main()

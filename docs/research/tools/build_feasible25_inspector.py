#!/usr/bin/env python3
"""Embed published result rows in the existing literal policy-inspector fragment.

Reads JSON only. Does not import a simulator, sample a seed, or open a recording.
Accepts the final inspection-data.json rows or an existing benchmark panel summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "docs/research/results/feasible25_offgrid_20260915/inspection-data.json"
DEFAULT_FRAGMENT = Path(
    "/Users/filippogombac/.codex/visualizations/2026/09/15/01a0a466-8883-7cb1-8cef-99e03183ca0b/feasible25-policy-inspector.html"
)
RESOURCES = ("battery", "h2", "co2", "ch4")
LABELS = {"daily-coverage-offgrid-v1": "Coverage baseline", "daily-coverage-feasibility-v2": "Original coverage", "operational-economic-mpc-v1": "Operating MPC"}


def finite(value, name, *, nullable=False):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def normalize(raw, payload):
    controller = raw["controller"]
    if not isinstance(controller, str) or not controller:
        raise ValueError("Controller identity is missing")
    physical = raw.get("physical_metrics", {})
    economic = raw.get("economic_metrics", {})
    row = {
        "controller": controller,
        "label": raw.get("label", LABELS.get(controller, controller)),
        "split": raw.get("split", payload.get("split")),
        "seed": raw["seed"],
        "execution_complete": boolean(raw["execution_complete"], "execution_complete"),
        "mission_passed": boolean(raw["mission_passed"], "mission_passed"),
        "hard_safety_passed": raw.get("hard_safety_passed"),
        "unmet_kg": finite(raw.get("unmet_kg", physical.get("unmet_kg")), "unmet_kg", nullable=True),
        "modeled_value_eur": finite(raw.get("modeled_value_eur", economic.get("net_operating_value_eur")), "modeled_value_eur", nullable=True),
        "decisions": raw["decisions"],
        "cycle_settlements": [],
    }
    if not isinstance(row["label"], str) or not isinstance(row["split"], str) or not row["split"]:
        raise ValueError("Label and split must be strings")
    for key in ("seed", "decisions"):
        if isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if row["decisions"] > 672:
        raise ValueError("Decision count exceeds a benchmark week")
    if row["hard_safety_passed"] is not None:
        boolean(row["hard_safety_passed"], "hard_safety_passed")
    for cycle in raw.get("cycle_settlements", []):
        index = cycle["cycle_index"]
        if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= 7:
            raise ValueError("Cycle index must be between 1 and 7")
        row["cycle_settlements"].append({
            "cycle_index": index,
            **{key: {resource: finite(cycle[key][resource], f"{key}.{resource}") for resource in RESOURCES} for key in ("actual", "targets", "deficits")},
            "feasible": boolean(cycle["feasible"], "cycle.feasible"),
        })
    row["cycle_settlements"].sort(key=lambda cycle: cycle["cycle_index"])
    if [cycle["cycle_index"] for cycle in row["cycle_settlements"]] != list(range(1, len(row["cycle_settlements"]) + 1)):
        raise ValueError("Recorded daily boundaries must be unique and consecutive")
    if row["mission_passed"] and not (
        row["execution_complete"] and row["hard_safety_passed"] is True
        and row["decisions"] == 672 and row["unmet_kg"] is not None
        and row["unmet_kg"] <= 1e-9 and len(row["cycle_settlements"]) == 7
        and all(cycle["feasible"] for cycle in row["cycle_settlements"])
    ):
        raise ValueError("Mission pass contradicts recorded outcome")
    return row


def build(source: Path, fragment: Path):
    source = source.resolve()
    content = source.read_bytes()
    payload = json.loads(content)
    if isinstance(payload, list):
        payload = {"rows": payload}
    raw_rows = payload.get("rows", payload.get("runs"))
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("No published result rows in input")
    rows = [normalize(raw, payload) for raw in raw_rows]
    identities = {(row["controller"], row["split"], row["seed"]) for row in rows}
    if len(identities) != len(rows):
        raise ValueError("Duplicate controller/split/scenario row")
    splits = {row["split"] for row in rows}
    default_split = next((split for split in ("held_out", "development") if split in splits), rows[0]["split"])
    seeds = {row["seed"] for row in rows if row["split"] == default_split}
    default_seed = 9930000 if default_split == "held_out" and 9930000 in seeds else min(seeds)
    peers = [row for row in rows if row["split"] == default_split and row["seed"] == default_seed]
    controllers = {row["controller"] for row in peers}
    default_controller = next(
        (controller for controller in ("coverage-ppo-101", "daily-coverage-offgrid-v1") if controller in controllers),
        peers[0]["controller"],
    )
    data = {
        "schema": "feasible25-policy-inspection-v1",
        "source": {"path": str(source), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()},
        "benchmark_contract_sha256": payload.get("benchmark_contract_sha256"),
        "default_selection": {"split": default_split, "seed": default_seed, "controller": default_controller},
        "rows": rows,
    }
    packed = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    literal = fragment.read_text()
    pattern = r"(?<=<!--F25_DATA_BEGIN-->)[\s\S]*?(?=<!--F25_DATA_END-->)"
    replacement = '\n<script id="f25-data" type="application/json">' + packed + "</script>\n"
    updated, count = re.subn(pattern, lambda _: replacement, literal)
    if count != 1:
        raise ValueError("Expected exactly one data slot in the literal fragment")
    if len(updated.encode()) >= 1_000_000:
        raise ValueError("Fragment exceeds 1 MB")
    if re.search(r"<!doctype\s|<\s*(?:html|head|body)(?:\s|>)", updated, re.I):
        raise ValueError("Expected a fragment, not a complete document")
    fragment.write_text(updated)
    print(json.dumps({"fragment": str(fragment), "bytes": len(updated.encode()), "rows": len(rows), "splits": sorted(splits), "default_selection": data["default_selection"], "source_sha256": data["source"]["sha256"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--fragment", type=Path, default=DEFAULT_FRAGMENT)
    args = parser.parse_args()
    build(args.input, args.fragment)


if __name__ == "__main__":
    main()

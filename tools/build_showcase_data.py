"""Export compact, read-only data for the static showcase site in site/data/.

Reads a `tools/reproduce.py evaluate` output directory. Week replays come from
the first final seed, whose minute states the evaluation retains for every
controller and condition; aggregate results need the completed
summary.json and analysis.json.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site/data"
WEEK_CONTROLLERS = ("cem-ridge-bias-g16-5901", "storage-m48-b60-r3", "original-daylight")
STEP = 15  # minutes per exported sample, one controller decision
START_H = 6.0
FLOWS = (
    "solar_to_battery_kw", "solar_to_electrolyser_kw", "solar_to_dac_kw", "solar_to_methanator_kw",
    "battery_to_electrolyser_kw", "battery_to_dac_kw", "battery_to_methanator_kw",
    "h2_level_kg", "co2_level_kg", "ch4_level_kg",
)


def r(value, digits=1):
    return None if value is None else round(float(value), digits)


def compact(values):
    return [round(v) for v in values]


def capacities():
    spec = json.loads((ROOT / "experiments/reproduction.json").read_text())
    p = spec["original-storage"]["teacher_parameters"]
    keys = ("solar_peak_kw", "battery_capacity_kwh", "electrolyser_rated_kw", "dac_rated_kw",
            "methanator_rated_kw", "h2_capacity_kg", "co2_capacity_kg", "ch4_capacity_kg")
    return {k: p[k] for k in keys}


def week(directory: Path, condition: str, seed: int):
    out = {"condition": condition, "seed": seed, "start_h": START_H, "step_min": STEP, "controllers": {}}
    solar = None
    for controller in WEEK_CONTROLLERS:
        case = directory / "cases" / f"{condition}--{controller}--{seed}"
        result = json.loads((case / "result.json").read_text())
        states = json.loads(gzip.decompress((case / "recording.json.gz").read_bytes()))["minute_truth_states"]
        if not states:
            raise ValueError(f"{case} has no retained minute states")
        states = states[::STEP]
        potential = compact(s["solar_potential_kw"] for s in states)
        if solar is None:
            solar = potential
            out["solar_potential_kw"] = potential
            out["solar_available"] = [int(s["solar_available_kw"] > 0 or s["solar_potential_kw"] <= 0) for s in states]
        elif potential != solar:
            raise ValueError(f"{condition}: solar forcing differs between paired controllers")
        series = {k: compact(s[k] for s in states) for k in FLOWS}
        series["battery_soc"] = [round(s["battery_soc"] * 1000) for s in states]
        economic = result["economic_metrics"]
        out["controllers"][controller] = {
            "series": series,
            "deliveries": [{"due_h": d["due_step"] / 60, "requested_kg": d["requested_kg"],
                            "delivered_kg": r(d["delivered_kg"]), "unmet_kg": r(d["unmet_kg"])}
                           for d in result["delivery"]["settlements"]],
            "unmet_kg": r(economic["unmet_kg"]),
            "delivered_kg": r(economic["delivered_kg"]),
            "contribution_eur": r(economic["net_operating_value_eur"], 0),
        }
        event = result["event_metrics"] or {}
        out["event"] = {"onset_h": event.get("onset_h"), "end_h": event.get("end_h")}
    return out


def _strip(value):
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in ("values", "per_seed_differences")}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def evaluation(directory: Path):
    analysis = json.loads((directory / "analysis.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    rows = []
    for row in summary["rows"]:
        economic = row["economic_metrics"]
        rows.append({
            "seed": row["seed"], "condition": row["condition"], "controller": row["controller"],
            "unmet_kg": r(economic["unmet_kg"], 2),
            "contribution_eur": r(economic["net_operating_value_eur"], 1),
            "missed": sum(d["unmet_kg"] > 1e-9 for d in row["delivery"]["settlements"]),
            "all_deliveries": row["every_delivery_passed"],
        })
    return {
        "case_count": analysis["case_count"],
        "all_cases_audited": analysis["all_cases_audited"],
        "split": analysis["split"],
        "groups": analysis["groups"],
        "primary_claim": _strip(analysis["primary_claim"]),
        "distribution_caveat": analysis["distribution_caveat"],
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path, help="reproduce.py evaluate output directory")
    parser.add_argument("--weeks-only", action="store_true", help="export week replays before the evaluation completes")
    args = parser.parse_args()
    spec = json.loads((ROOT / "experiments/reproduction.json").read_text())
    seed, conditions = spec["seeds"][0], spec["conditions"]
    (OUT / "weeks").mkdir(parents=True, exist_ok=True)
    dump = lambda path, data: path.write_text(json.dumps(data, separators=(",", ":")))  # noqa: E731
    dump(OUT / "plant.json", {"capacities": capacities(), "conditions": conditions, "week_seed": seed,
                              "week_controllers": list(WEEK_CONTROLLERS)})
    for condition in conditions:
        dump(OUT / "weeks" / f"{condition}.json", week(args.evaluation, condition, seed))
    if not args.weeks_only:
        dump(OUT / "evaluation.json", evaluation(args.evaluation))


if __name__ == "__main__":
    main()

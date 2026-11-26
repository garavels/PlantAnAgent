"""Export compact, read-only data for the static showcase site in site/data/.

Inputs are the committed demo recording and, optionally, a completed
`tools/reproduce.py evaluate` output directory (summary.json + analysis.json).
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDING = ROOT / "demo/recordings/cd1ff361-33ac-5f84-9c94-8363ffdbe12f.json"
OUT = ROOT / "site/data"


def r(value, digits=1):
    return None if value is None else round(float(value), digits)


def featured_week():
    recording = json.loads(RECORDING.read_text())
    frames = recording["frames"]
    keys = {
        "solar_potential_kw": 0, "solar_available_kw": 0, "solar_curtailed_kw": 0,
        "electrolyser_power_kw": 0, "dac_power_kw": 0, "methanator_power_kw": 0,
        "battery_power_kw": 0, "battery_soc": 4, "h2_level_kg": 1, "co2_level_kg": 1,
        "ch4_level_kg": 1, "ch4_production_kg_h": 2, "h2_production_kg_h": 2, "co2_capture_kg_h": 2,
    }
    series = {k: [r(f["state"][k], d) for f in frames] for k, d in keys.items()}
    series["t_h"] = [r(f["clock"]["elapsed_hours"], 2) for f in frames]
    series["mode"] = [f["commitment"]["mode"] for f in frames]
    series["delivered_kg"] = [r(f["delivery"]["delivered_kg"], 1) for f in frames]
    series["unmet_kg"] = [r(f["delivery"]["unmet_kg"], 1) for f in frames]
    series["net_value_eur"] = [r(f["economics"]["net_value_eur"], 0) for f in frames]
    last = frames[-1]
    config = recording["snapshot"]["config"]
    return {
        "source": RECORDING.relative_to(ROOT).as_posix(),
        "evidence_level": recording["evidence_level"],
        "provenance": recording["provenance"],
        "capacities": {
            "solar_peak_kw": config["solar"]["peak_kw"],
            "battery_kwh": config["battery"]["capacity_kwh"],
            "electrolyser_kw": config["electrolyser"]["rated_kw"],
            "dac_kw": config["dac"]["rated_kw"],
            "methanator_kw": config["methanator"]["rated_kw"],
            "h2_kg": config["h2_tank"]["capacity_kg"],
            "co2_kg": config["co2_tank"]["capacity_kg"],
            "ch4_kg": config["ch4_tank"]["capacity_kg"],
        },
        "deliveries": [{"due_h": c["due_hours"], "requested_kg": c["requested_kg"],
                        "delivered_kg": c["delivered_kg"], "status": c["status"]}
                       for c in last["delivery"]["commitments"]],
        "totals": last["metrics"],
        "economics": last["economics"],
        "series": series,
    }


TRACE_CONDITIONS = ("ordinary", "pv36", "meth48")
TRACE_CONTROLLERS = ("cem-ridge-bias-g16-5901", "storage-m48-b60-r3", "original-daylight")


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
        economic = row["economic_metrics"] or {}
        rows.append({
            "seed": row["seed"], "condition": row["condition"], "controller": row["controller"],
            "unmet_kg": r(economic.get("unmet_kg"), 2),
            "contribution_eur": r(economic.get("net_operating_value_eur"), 1),
            "missed": sum(d["unmet_kg"] > 1e-9 for d in row["delivery"]["settlements"]),
            "all_deliveries": row["every_delivery_passed"],
        })
    first_seed = min(row["seed"] for row in summary["rows"])
    traces = {}
    for row in summary["rows"]:
        if row["seed"] != first_seed or row["condition"] not in TRACE_CONDITIONS \
                or row["controller"] not in TRACE_CONTROLLERS:
            continue
        case = directory / "cases" / f"{row['condition']}--{row['controller']}--{row['seed']}" / "recording.json.gz"
        states = json.loads(gzip.decompress(case.read_bytes()))["minute_truth_states"][::15]
        traces.setdefault(row["condition"], {})[row["controller"]] = {
            "ch4_level_kg": [r(s["ch4_level_kg"]) for s in states],
            "battery_soc": [r(s["battery_soc"], 3) for s in states],
            "process_kw": [r(s["electrolyser_power_kw"] + s["dac_power_kw"] + s["methanator_power_kw"], 0)
                           for s in states],
            "delivered_kg": [r(d["delivered_kg"]) for d in row["delivery"]["settlements"]],
            "event": {k: (row["event_metrics"] or {}).get(k)
                      for k in ("onset_h", "end_h", "event_unmet_kg", "post_event_unmet_kg")},
        }
    return {
        "case_count": analysis["case_count"],
        "all_cases_audited": analysis["all_cases_audited"],
        "split": analysis["split"],
        "groups": analysis["groups"],
        "primary_claim": _strip(analysis["primary_claim"]),
        "distribution_caveat": analysis["distribution_caveat"],
        "paired_forcing_and_openings_equal": summary.get("paired_forcing_and_openings_equal"),
        "rows": rows,
        "trace_seed": first_seed,
        "traces": traces,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, help="completed reproduce.py evaluate output directory")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "featured-week.json").write_text(json.dumps(featured_week(), separators=(",", ":")))
    if args.evaluation:
        (OUT / "evaluation.json").write_text(json.dumps(evaluation(args.evaluation), separators=(",", ":")))


if __name__ == "__main__":
    main()

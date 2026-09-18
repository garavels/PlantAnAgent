"""Build a checksum-verified, offline comparison inspector without a simulator.

Design: a quiet laboratory notebook, with one continuous energy/material path;
solar amber, electrical teal, DAC blue, methane red, and a cool white background.
Avenir Next carries headings and body text, with local system fallbacks.  Results
form a full-width comparison table; plots share one precise minute cursor.  Night
shading, battery zoom, and explicit requested/applied/actual controls address the
scientific questions rather than decorate a generic dashboard.

Final mode admits only an analysis containing every prescribed held-out primary
panel. Train-only fixture mode is visibly distinct and never claims final results.
No network, simulator, training library, or external JavaScript is used.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

SCHEMA = "comparison-ppo-v2-evaluation"
FORECAST_SCHEMA = "forecast-ablation-v1"
PRIMARY_CONDITIONS = ("d25-sampled", "d25-floor", "d30-sampled", "d30-floor")
PRIMARY_CONTROLLERS = ("daylight", "storage", "ppo25-101", "ppo25-211", "ppo25-307")
FIELDS = (
    "step", "time_s", "solar_available_kw", "solar_potential_kw", "solar_output_factor", "solar_to_battery_kw", "solar_curtailed_kw",
    "battery_to_electrolyser_kw", "electrolyser_power_kw", "dac_power_kw", "methanator_power_kw",
    "battery_energy_kwh", "battery_soc", "h2_level_kg", "co2_level_kg", "ch4_level_kg",
    "h2_production_kg_h", "co2_capture_kg_h", "ch4_production_kg_h",
)
STOCKS = ("battery_energy_kwh", "battery_soc", "h2_level_kg", "co2_level_kg", "ch4_level_kg")
GROUP_METRICS = ("net_operating_value_eur", "unmet_kg", "ch4_produced_kg", "curtailed_kwh",
    "electrolyser_forced_stops", "operating:battery_charge_kwh", "operating:battery_discharge_kwh",
    "operating:zero_solar_electrolyser_kwh", "operating:zero_solar_ch4_produced_kg",
    *(f"initial:{key}" for key in STOCKS), *(f"final:{key}" for key in STOCKS))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def pin(path):
    path = Path(path)
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def resolve_pin(record_pin, root=None):
    original = Path(record_pin["path"])
    candidates = [original] if root is None else list(Path(root).rglob(original.name))
    matches = [path for path in candidates if path.is_file() and path.stat().st_size == record_pin["bytes"]
               and pin(path)["sha256"] == record_pin["sha256"]]
    if len(matches) != 1:
        raise ValueError(f"Expected one checksum-matching file for {original.name}, found {len(matches)}")
    return matches[0]


def key_for(row):
    return row["condition"] + "|" + row["controller"]


def compact_summary(row):
    fields = ("controller", "condition", "split", "seed", "variant", "controller_kind",
        "training_family", "optimizer_seed", "checkpoint_timesteps", "execution_complete",
        "every_delivery_passed", "physical_invariants_passed", "failure", "decisions",
        "physical_steps", "minute_audit", "economic_audit", "physical_metrics",
        "economic_metrics", "operating_metrics", "baseline_metadata", "inference",
        "evaluation_freeze_sha256", "paired_source_sha256")
    item = {name: row.get(name) for name in fields}
    item["initial_state"] = {name: row.get("initial_state", {}).get(name) for name in STOCKS}
    item["final_state"] = {name: row.get("final_state", {}).get(name) for name in STOCKS}
    return item


def compact_record(path, expected_row=None, fixture=False):
    with gzip.open(path, "rt") as stream:
        record = json.load(stream)
    row = record["summary"]
    expected_schema = FORECAST_SCHEMA if row.get("controller") == "daylight-persistence" else SCHEMA
    if record.get("schema") != expected_schema + "-recording" or row.get("schema") != expected_schema + "-case":
        raise ValueError("Unexpected recording schema")
    if expected_row is not None and row != {k: v for k, v in expected_row.items() if k != "recording"}:
        raise ValueError("Analysis row and recording summary disagree")
    if fixture and row["split"] != "train":
        raise ValueError("Fixture mode accepts training recordings only")
    minutes, decisions = record.get("minute_truth_states"), record.get("decision_trace", [])
    if not minutes:
        if row["execution_complete"]:
            raise ValueError("Completed prescribed trace lacks minute truth")
        return {"summary": compact_summary(row), "unavailable": "This attempt has no retained minute trace.", "recording": pin(path)}
    if canonical_sha(minutes) != row["retained_minute_truth_sha256"]:
        raise ValueError("Retained minute truth checksum differs")
    if not decisions:
        if row["execution_complete"]:
            raise ValueError("Completed attempt has no decisions")
        return {"summary": compact_summary(row), "unavailable": "This incomplete attempt ended before its first recorded decision.", "recording": pin(path)}
    if len(minutes) != row["physical_steps"] + 1 or len(decisions) != row["decisions"] or len(minutes) != 15 * len(decisions) + 1:
        raise ValueError("Recording length does not match 15-minute decisions")
    for i, minute in enumerate(minutes):
        if minute["step"] != i or minute["time_s"] != minutes[0]["time_s"] + 60 * i:
            raise ValueError("Minute truth is not contiguous")
    witness = hashlib.sha256()
    commands = []
    for i, decision in enumerate(decisions):
        if decision["decision_index"] != i:
            raise ValueError("Decision order differs")
        closing = minutes[(i + 1) * 15]
        if any(closing[k] != v for k, v in decision["state"].items()):
            raise ValueError("Decision closing state differs from minute truth")
        evidence = decision["evidence"]
        observation = evidence.get("observation")
        if not observation or len(observation) != 135 or canonical_sha(observation) != evidence["observation_sha256"]:
            raise ValueError("Prescribed public observation is missing or altered")
        if evidence["action"] != decision["action"]:
            raise ValueError("Inference action differs from executed request")
        transformed = evidence.get("transformed_observation")
        if expected_schema == FORECAST_SCHEMA:
            target = list(observation)
            for slot in (34, 35, 36):
                target[slot] = observation[2]
            if transformed != target or canonical_sha(transformed) != evidence.get("transformed_observation_sha256"):
                raise ValueError("Persistence input differs from the three prescribed replacements")
        witness.update(json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        commands.append({"index": i, "action": decision["action"], "raw_mean": evidence.get("raw_mean"),
            "observation": observation, "observation_sha256": evidence["observation_sha256"],
            "normalized_observation_sha256": evidence.get("normalized_observation_sha256"),
            "transformed_observation": transformed, "replacements": evidence.get("replacements"),
            "direct_control": decision["direct_control"], "battery_intervention": decision.get("battery_intervention"),
            "reward_eur": decision["reward_eur"]})
    if witness.hexdigest() != row["inference"]["witness_sha256"]:
        raise ValueError("Actual inference witness hash differs")
    if minutes[0] != row["initial_state"] or minutes[-1] != row["final_state"]:
        raise ValueError("Opening or closing state differs")
    return {"summary": compact_summary(row),
        "minutes": [[m[name] for name in FIELDS] + [int(m["health"]["dac"]["offline"])] for m in minutes],
        "commands": commands, "delivery": record["delivery"], "recording": pin(path),
        "capacity": {name: minutes[0][name] for name in ("battery_capacity_kwh", "h2_capacity_kg", "co2_capacity_kg", "ch4_capacity_kg")}}


def assemble(*, analysis_path=None, forecast_analysis_path=None, recordings_root=None, fixtures=None):
    fixture = bool(fixtures)
    provenance = {"builder": pin(__file__), "created_at": datetime.now(timezone.utc).isoformat()}
    records, rows, groups, comparisons = {}, [], [], []
    supplement = None
    if fixture:
        if analysis_path or forecast_analysis_path:
            raise ValueError("Do not mix final analysis and train-only fixtures")
        for path in fixtures:
            record = compact_record(path, fixture=True)
            row = record["summary"]
            key = key_for(row)
            if key in records:
                raise ValueError("Duplicate fixture controller and condition")
            records[key] = record
            rows.append(row)
        seeds = {r["seed"] for r in rows}
        if len(seeds) != 1:
            raise ValueError("Fixture traces must share one training seed")
        scenario = seeds.pop()
    else:
        if analysis_path is None:
            raise ValueError("Final inspector requires a completed canonical analysis")
        analysis = json.loads(Path(analysis_path).read_text())
        if analysis.get("schema") != SCHEMA + "-analysis":
            raise ValueError("Unexpected analysis schema")
        original_rows = analysis["rows"]
        groups_by = defaultdict(list)
        freezes = {}
        seen = set()
        for row in original_rows:
            case = row["split"], row["condition"], row["controller"], row["seed"]
            if case in seen or row["split"] not in {"development", "held_out"}:
                raise ValueError("Duplicate outcome or noncanonical split")
            seen.add(case)
            groups_by[case[:3]].append(row)
        for item in analysis["freezes"]:
            path = resolve_pin(item["file"])
            freeze = json.loads(path.read_text())
            if freeze.get("schema") != SCHEMA + "-freeze" or freeze["sha256"] != item["sha256"] or canonical_sha({k: v for k, v in freeze.items() if k != "sha256"}) != freeze["sha256"]:
                raise ValueError("Frozen comparison identity differs")
            freezes[freeze["sha256"]] = freeze
            for panel in freeze["panels"]:
                if panel["split"] == "train":
                    continue
                found = groups_by[panel["split"], panel["condition"], panel["controller"]]
                if sorted(r["seed"] for r in found) != panel["seeds"]:
                    raise ValueError("A prescribed panel is incomplete; final inspector is unavailable")
        expected_seeds = list(range(20200000, 20200032))
        for condition in PRIMARY_CONDITIONS:
            for controller in PRIMARY_CONTROLLERS:
                if sorted(r["seed"] for r in groups_by["held_out", condition, controller]) != expected_seeds:
                    raise ValueError("Final mode requires all four primary conditions and all five controllers")
        scenario = expected_seeds[0]
        for row in original_rows:
            if row["evaluation_freeze_sha256"] not in freezes:
                raise ValueError("Outcome lacks a declared comparison freeze")
            if row.get("recording"):
                path = resolve_pin(row["recording"], recordings_root)
                if row["split"] == "held_out" and row["seed"] == scenario:
                    records[key_for(row)] = compact_record(path, expected_row=row)
            elif row["split"] == "held_out" and row["seed"] == scenario:
                records[key_for(row)] = {"summary": compact_summary(row), "unavailable": "The retained attempt has no recording."}
            rows.append(compact_summary(row))
        for group in analysis["groups"]:
            found = groups_by[group["split"], group["condition"], group["controller"]]
            if len(found) != group["n_scenarios"] or group["delivery_successes"] != sum(r["every_delivery_passed"] for r in found) or group["complete_weeks"] != sum(r["execution_complete"] for r in found):
                raise ValueError("Overview group disagrees with retained individual outcomes")
            groups.append({k: v for k, v in group.items() if k != "metrics"} | {
                "metrics": {k: group["metrics"][k] for k in GROUP_METRICS if k in group["metrics"]}})
        for item in analysis.get("aggregate_comparisons", []):
            comparisons.append({k: v for k, v in item.items() if k != "metrics"} | {
                "contribution": item["metrics"]["net_operating_value_eur"]})
        provenance.update(analysis=pin(analysis_path), freezes=analysis["freezes"], bootstrap=analysis["bootstrap"])
        if forecast_analysis_path:
            extra = json.loads(Path(forecast_analysis_path).read_text())
            if extra.get("schema") != FORECAST_SCHEMA + "-analysis" or extra.get("supplementary") is not True:
                raise ValueError("Unexpected supplementary forecast analysis")
            if extra["reference_main_analysis"] != pin(analysis_path):
                raise ValueError("Supplement refers to a different main analysis")
            found = extra["rows"]
            if len(found) != 32 or sorted(r["seed"] for r in found) != expected_seeds or any(
                (r["controller"], r["condition"], r["split"]) != ("daylight-persistence", "d30-sampled", "held_out") for r in found):
                raise ValueError("Supplement must retain exactly its 32 prescribed persistence outcomes")
            reference = sorted((r for r in original_rows if (r["controller"], r["condition"], r["split"]) == ("daylight", "d30-sampled", "held_out")), key=lambda r: r["seed"])
            if sorted(extra["reference_rows"], key=lambda r: r["seed"]) != reference:
                raise ValueError("Supplementary daylight reference differs from the main outcomes")
            supplementary_freezes = set()
            for item in extra["freezes"]:
                freeze_path = resolve_pin(item["file"])
                freeze = json.loads(freeze_path.read_text())
                if freeze.get("schema") != FORECAST_SCHEMA + "-freeze" or freeze["sha256"] != item["sha256"] or canonical_sha({k: v for k, v in freeze.items() if k != "sha256"}) != freeze["sha256"]:
                    raise ValueError("Supplement freeze identity differs")
                supplementary_freezes.add(freeze["sha256"])
            for row in found:
                if row["evaluation_freeze_sha256"] not in supplementary_freezes:
                    raise ValueError("Supplementary outcome lacks its frozen comparison identity")
                path = resolve_pin(row["recording"], recordings_root) if row.get("recording") else None
                if row["seed"] == scenario:
                    records[key_for(row)] = compact_record(path, expected_row=row) if path else {
                        "summary": compact_summary(row), "unavailable": "Supplementary attempt has no recording."}
                item = compact_summary(row)
                item["supplementary"] = True
                rows.append(item)
            if len(extra["groups"]) != 1:
                raise ValueError("Expected one supplementary summary group")
            group = extra["groups"][0]
            if group["n_scenarios"] != 32 or group["delivery_successes"] != sum(r["every_delivery_passed"] for r in found) or group["complete_weeks"] != sum(r["execution_complete"] for r in found):
                raise ValueError("Supplementary group differs from retained cases")
            groups.append({k: v for k, v in group.items() if k != "metrics"} | {"supplementary": True,
                "metrics": {k: group["metrics"][k] for k in GROUP_METRICS if k in group["metrics"]}})
            supplement = {"criterion": extra["criterion"], "interpretation": extra["interpretation"],
                "condition": "d30-sampled", "n_scenarios": 32, "reference_controller": "daylight",
                "reference_delivery_successes": sum(r["every_delivery_passed"] for r in reference),
                "persistence_delivery_successes": sum(r["every_delivery_passed"] for r in found),
                "analysis": pin(forecast_analysis_path)}
            provenance["forecast_analysis"] = pin(forecast_analysis_path)
            provenance["forecast_freezes"] = extra["freezes"]
    pairings = {r["summary"].get("paired_source_sha256") for r in records.values() if r["summary"].get("paired_source_sha256")}
    if len(pairings) > 1:
        raise ValueError("Selected traces do not share the prescribed weather, faults and original opening gas stocks")
    default = "d25-sampled|ppo25-101"
    if default not in records:
        default = next(iter(records))
    return {"schema": "comparison-ppo-v2-offline-inspector", "fixture": fixture, "scenario": scenario,
        "fields": [*FIELDS, "dac_offline"], "default": default, "records": records, "rows": rows,
        "groups": groups, "comparisons": comparisons, "supplement": supplement, "provenance": provenance}


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; connect-src 'none'">
<title>Daylight production and learned control — PlantAnAgent</title>
<style>
:root{font-family:"Avenir Next",Avenir,"Trebuchet MS",sans-serif;color:#183345;background:#f3f7fa;font-variant-numeric:tabular-nums;--muted:#516879;--line:#cbd8e2;--solar:#ad7815;--electric:#087c83;--dac:#4c6caf;--meth:#ae5547}*{box-sizing:border-box}body{margin:0}main{max-width:1320px;margin:auto;padding:32px 36px 56px}h1{font-size:clamp(31px,3.6vw,47px);font-weight:600;line-height:1.14;letter-spacing:-.035em;margin:20px 0}h2{font-size:24px;font-weight:600;line-height:1.2;letter-spacing:-.02em;margin:0 0 12px}h3{font-size:16px;font-weight:600;margin:0 0 10px}p{max-width:78ch;font-size:15px;line-height:1.65;color:var(--muted);margin:8px 0 18px}a{color:#066b73}button,select,input{font:inherit}button,select{background:white;color:#183345;border:1px solid #a7bccb;border-radius:5px;padding:9px 12px;max-width:100%}button{cursor:pointer;font-size:13px}button:hover{background:#e4f0f2}button:focus-visible,select:focus-visible,input:focus-visible,summary:focus-visible,a:focus-visible{outline:3px solid #bf821d;outline-offset:3px}button[aria-pressed=true]{background:#087c83;color:white;border-color:#087c83}label{font-size:13px;font-weight:600}header{display:grid;grid-template-columns:minmax(0,1fr) 260px;gap:28px;border-bottom:1px solid var(--line);padding-bottom:24px}.brand{font-size:15px;font-weight:600}.context{align-self:end;border-left:3px solid #9bbbc8;padding-left:18px;font-size:14px;line-height:1.7}.process{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;gap:14px;align-items:center;margin:22px 0 0;color:var(--muted);font-size:14px}.process strong{display:block;color:#183345;font-size:15px;font-weight:600}.process .arrow{color:#087c83;font-size:24px}.section{margin-top:34px}.bar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:15px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:18px 0}.control{display:flex;gap:8px;align-items:center}.control select{min-width:175px}.note{font-size:13px;line-height:1.55}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.5}caption{font-size:13px;color:var(--muted);text-align:left;padding:0 0 9px}th,td{padding:10px 9px;border-bottom:1px solid var(--line);vertical-align:top;text-align:right}th{color:var(--muted);font-weight:500;white-space:nowrap}th:first-child,td:first-child{text-align:left}td:first-child{font-weight:500}td{white-space:nowrap}tbody tr.selected{background:#deeff1}tbody tr[data-record]{cursor:pointer}tbody tr[data-record]:hover{background:#e8f1f6}.sub{display:block;font-size:11px;color:var(--muted);font-weight:400}.good{color:#056b6d}.bad{color:#9b4237}.warning{background:#fff0e3;border-left:4px solid #ad7815;color:#6d4a21;line-height:1.6;padding:15px 18px;margin:20px 0}.warning strong{font-weight:650}.method{display:grid;grid-template-columns:1fr 1fr;gap:30px;padding:20px 0;border-bottom:1px solid var(--line)}.method p{margin-bottom:0}.trace-head{display:flex;justify-content:space-between;align-items:end;gap:20px}.trace-head .stamp{font-size:14px;color:var(--muted);text-align:right}.transport{padding:16px;background:#e7f0f5;display:grid;grid-template-columns:1fr auto;gap:14px;margin:18px 0}.transport .controls{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.scrub{grid-column:1/-1;display:flex;align-items:center;gap:14px}.scrub input{flex:1;accent-color:#087c83;min-width:100px}.clock{font-size:21px;font-weight:600;white-space:nowrap}.status{font-size:14px;line-height:1.6;min-height:24px;margin:10px 0 16px}.plot{background:white;border:1px solid #dce5eb;padding:16px 15px 9px;min-width:0}.plot svg{width:100%;display:block;overflow:visible}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;line-height:1.8;color:var(--muted);margin:7px 0 0}.legend span:before{content:"";display:inline-block;width:19px;height:3px;background:var(--c);vertical-align:middle;margin-right:5px}.two>*,.stocks>*,.method>*,.transport>*{min-width:0}.two{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:20px}.stocks{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;margin-top:20px}.plothead{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:5px}.plothead h3{margin:0}.plothead button{padding:5px 9px;font-size:12px}.readout{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;margin:18px 0}.readout div{border-left:2px solid var(--line);padding-left:11px}.readout strong{display:block;font-size:21px;font-weight:600;white-space:nowrap}.readout span{font-size:12px;color:var(--muted)}details{margin-top:20px}summary{font-size:15px;font-weight:600;cursor:pointer;line-height:1.6}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;border:1px solid var(--line);padding:16px;font-size:12px;line-height:1.5;max-height:440px;overflow:auto}.ledger{max-height:520px}.ledger thead{position:sticky;top:0;background:#f3f7fa}.hidden{display:none!important}.load{max-width:700px;margin:40px auto;padding:30px;background:white;border:1px solid var(--line);font-size:17px;line-height:1.6}.outcomes span{display:inline-block;width:7px;height:12px;margin-right:2px;background:#0e8084}.outcomes span.failed{background:#b65a4c}.outcomes span.incomplete{background:#8092a0}.outcomes{line-height:1;white-space:nowrap}footer{font-size:13px;line-height:1.6;color:var(--muted);border-top:1px solid var(--line);margin-top:34px;padding-top:20px}.skip{position:absolute;top:-60px;left:15px;background:white;padding:10px;z-index:10}.skip:focus{top:10px}.small-actions{display:flex;gap:9px;flex-wrap:wrap}.auditline{font-size:13px;margin-top:10px;color:var(--muted)}@media(max-width:800px){main{padding:22px 16px 40px}header{grid-template-columns:1fr;gap:10px}.context{border-left:0;border-top:1px solid var(--line);padding:12px 0 0}.two,.stocks,.method{grid-template-columns:1fr}.readout{grid-template-columns:1fr 1fr;gap:16px}.transport{grid-template-columns:1fr}.clock{text-align:left}.trace-head{display:block}.trace-head .stamp{text-align:left}.control{width:100%;justify-content:space-between}.control select{min-width:0;max-width:72%}.process{gap:8px;font-size:12px}.process strong{font-size:13px}h2{font-size:22px}.table-wrap{margin-right:-2px}}@media print{.bar,.transport,.small-actions,button,.skip{display:none}.plot{break-inside:avoid}body{background:white}main{padding:0}details{display:none}}
</style></head><body><a class="skip" href="#trace-section">Skip to operating trace</a>
<div id="loading" class="load" role="status">Opening the verified experiment records…<p class="note">This file works offline. It contains the results and the prescribed operating traces.</p></div>
<main id="app" class="hidden">
<header><div><div class="brand">PlantAnAgent research</div><h1>Does planning improve on <br>daylight production?</h1><p>Compare learned policies with explicit operating rules, then inspect how the same simulated plant uses sunlight, stored electricity and gas to meet deliveries.</p><div class="process"><div><strong>Electricity</strong>Sunlight and battery</div><div class="arrow">→</div><div><strong>Feedstocks</strong>Hydrogen and captured CO₂</div><div class="arrow">→</div><div><strong>Delivery</strong>Stored methane</div></div></div><div class="context" id="context"></div></header>
<div id="fixture"></div>
<section class="method"><div><h3>What changes in this comparison</h3><p class="note">Demand is 25 or 30 kg/h. The opening battery is sampled at 25–65% or starts at its 5% modeled floor. Weather, faults and opening gas stocks are paired by scenario. Every PPO model was trained with 25–65% opening charge: its 5% test changes both the training initial-condition distribution and available energy. The episode stays seven days.</p></div><div><h3>What this plant assumes</h3><p class="note">This assumed 5 MW solar-to-methane benchmark studies scheduling and inventories. Its conversion-energy magnitudes have literature support; it is not a calibrated replica of Rivan. The battery powers electrolysis only, and carbon capture and methanation have no startup or thermal states.</p></div></section>
<section class="section" id="overview-section"><h2>Compare complete panels</h2><p class="note">Every prescribed outcome is included. One small mark represents one environmental scenario; green meets all deliveries, red misses at least one, and gray is incomplete. Repeated controllers and conditions do not create additional independent weather scenarios.</p><div class="bar"><div class="control"><label for="split">Evidence</label><select id="split"></select></div><div class="control"><label for="overview-condition">Operating condition</label><select id="overview-condition"></select></div></div><div class="table-wrap"><table><caption id="overview-caption"></caption><thead><tr><th>Controller</th><th>All deliveries</th><th>Mean contribution / €</th><th>Mean missed / kg</th><th>Battery charge / kWh</th><th>Battery discharge / kWh</th><th>Forced stops</th></tr></thead><tbody id="overview"></tbody></table></div><p class="note" id="paired"></p><p class="auditline" id="auditline"></p></section>
<section class="section hidden" id="learning-section"><h2>Did the new policies learn?</h2><p class="note">All three new PPO30 policies began from random weights and were trained at 30 kg/h. Initial and final checkpoints are evaluated on the same eight development scenarios, with each checkpoint’s saved observation normalizer. Every final seed is retained; no best checkpoint is selected.</p><div class="table-wrap"><table><thead><tr><th>Training seed</th><th>Initial all deliveries</th><th>Initial contribution / €</th><th>Final transitions</th><th>Final all deliveries</th><th>Final contribution / €</th></tr></thead><tbody id="learning"></tbody></table></div></section>
<section class="section hidden" id="supplement-section"><h2>Does the forecast input help?</h2><p class="note">A supplementary test keeps the daylight controller unchanged and replaces its three consumed forecast values with current solar availability. It uses the same 32 test scenarios at 30 kg/h with sampled opening charge. This measures the value of those inputs within this rule, not an optimum or proof that PPO needs forecasts.</p><div class="table-wrap"><table><thead><tr><th>Comparison</th><th>All deliveries</th><th>Forecast minus persistence / €</th></tr></thead><tbody id="supplement"></tbody></table></div><p class="note" id="supplement-criterion"></p></section>
<section class="section" id="trace-section"><div class="trace-head"><div><h2>Inspect one operating week</h2><p class="note">The first scenario was prescribed before outcomes were known. It is an example, not the average result. Click a plot or move the cursor to inspect an exact minute.</p></div><div class="stamp" id="scenario-stamp"></div></div><div class="bar"><div class="control"><label for="condition">Operating condition</label><select id="condition"></select></div><div class="control"><label for="controller">Controller</label><select id="controller"></select></div><div class="control"><label for="window">Plot window</label><select id="window"><option value="168">Full week</option><option value="24">Selected day</option><option value="6">Six hours</option></select></div></div><p class="note" id="policy-note"></p><div id="trace-missing" class="warning hidden"></div><div id="trace-content"><div class="transport"><div class="controls"><button id="previous">−15 min</button><button id="next">+15 min</button><button id="next-night">Next night</button><button id="next-dip">Next solar dip</button><button id="next-fault">Next DAC outage</button><button id="next-delivery">Next delivery</button></div><output id="clock" class="clock"></output><div class="scrub"><label for="timeline">Recorded minute</label><input id="timeline" type="range" min="0" max="10080" step="1" value="0"><output id="elapsed"></output></div></div><p id="status" class="status" role="status" aria-live="polite"></p><div class="readout" id="readout"></div>
<section class="plot"><h3>Actual electrical power / kW</h3><svg id="power" role="img" aria-label="Actual solar and process power"></svg><div class="legend"><span style="--c:#ad7815">Solar available</span><span style="--c:#087c83">Electrolyser</span><span style="--c:#4c6caf">Carbon capture</span><span style="--c:#ae5547">Methanator</span></div><p class="note">Pale shading marks minutes with zero solar power; red bands mark observed DAC outages. Electrolyser power includes battery support. Lines show actual consumed power, not the controller's requested power. “Next solar dip” finds a fall of at least 30% over 15 minutes while clear-sky potential remains above 1 MW. Coincident actions and disturbances alone do not establish anticipation.</p></section>
<div class="two"><section class="plot"><div class="plothead"><h3>Battery charge level / %</h3><button id="battery-scale" aria-pressed="false">Zoom to variation</button></div><svg id="soc" role="img" aria-label="Battery charge level with physical floor"></svg><p class="note" id="soc-caption"></p></section><section class="plot"><h3>Battery routing and curtailed sunlight / kW</h3><svg id="routing" role="img" aria-label="Actual battery charging, discharging and curtailment"></svg><div class="legend"><span style="--c:#4c6caf">Charging</span><span style="--c:#087c83">Supporting electrolysis</span><span style="--c:#ad7815">Curtailed solar</span></div></section></div>
<div class="stocks"><section class="plot"><h3>Hydrogen stored / kg</h3><svg id="h2" role="img" aria-label="Stored hydrogen"></svg></section><section class="plot"><h3>CO₂ stored / kg</h3><svg id="co2" role="img" aria-label="Stored carbon dioxide"></svg></section><section class="plot"><h3>Methane stored / kg</h3><svg id="ch4" role="img" aria-label="Stored methane and opening amount"></svg></section></div><p class="note">Dashed inventory lines mark the opening amounts. Restoring each inventory is not required daily. Material-equivalent accounting does not imply that each feedstock was individually restored.</p>
<div class="two"><section><h3>Request, intervention and physical response</h3><p class="note" id="command-time"></p><div class="table-wrap"><table><thead><tr><th>Destination</th><th>Requested</th><th>After intervention</th><th>Actual mean / kW</th></tr></thead><tbody id="commands"></tbody></table></div><p class="note" id="command-note"></p></section><section class="plot"><h3>Public sunlight forecast at that decision / kW</h3><svg id="forecast" role="img" aria-label="Causal mean and p10 solar forecast in four time windows"></svg><div class="legend"><span style="--c:#ad7815">Mean</span><span style="--c:#4c6caf">p10</span></div><p class="note">These are averages over the four forecast windows, available when the action was chosen. The p10 is an ensemble percentile of each window average; it is not a guaranteed minimum minute power.</p><p class="note" id="forecast-replacement"></p></section></div>
<div class="two"><section class="plot"><h3>Methane produced / kg per hour</h3><svg id="production" role="img" aria-label="Actual methane production rate"></svg><p class="note">Deliveries can draw stored methane while production is stopped. Nighttime methane production is zero in this electrical architecture.</p></section><section><h3>Selected week's economic account / €</h3><div class="table-wrap"><table><thead><tr><th>Item</th><th>Contribution</th></tr></thead><tbody id="economics"></tbody></table></div><p class="note">Delivery value less missed-delivery penalties, battery throughput wear, electrolyser life consumption and any ending stock-depletion liability. Dedicated sunlight has zero marginal opportunity cost here; fixed plant costs are excluded.</p></section></div>
<details><summary>Delivery ledger and individual opening/ending stocks</summary><div class="two"><div class="table-wrap"><table><thead><tr><th>Delivery</th><th>Requested / kg</th><th>Delivered / kg</th><th>Missed / kg</th></tr></thead><tbody id="deliveries"></tbody></table></div><div><table><thead><tr><th>Inventory</th><th>Opening</th><th>Ending</th></tr></thead><tbody id="stock-ledger"></tbody></table><p class="note" id="night-account"></p></div></div></details>
<details><summary>Recorded inputs, actions and audit evidence</summary><pre id="detail"></pre></details><div class="small-actions section"><button id="download-minutes">Download this minute series</button><button id="download-decisions">Download these decision inputs</button></div></div></section>
<section class="section"><h2>Every retained case</h2><p class="note">This ledger includes development and held-out outcomes, including incomplete or failed attempts. Filter the view without changing the underlying evidence.</p><div class="bar"><div class="control"><label for="case-split">Evidence</label><select id="case-split"></select></div><div class="control"><label for="case-condition">Condition</label><select id="case-condition"></select></div><div class="control"><label for="case-controller">Controller</label><select id="case-controller"></select></div><button id="download-cases">Download all case outcomes</button></div><div class="table-wrap ledger"><table><caption id="cases-caption"></caption><thead><tr><th>Controller / condition</th><th>Scenario</th><th>All deliveries</th><th>Contribution / €</th><th>Missed / kg</th><th>Physical / economic audit</th></tr></thead><tbody id="cases"></tbody></table></div></section>
<section class="section"><h2>Read the result within its scope</h2><p>Full PPO chooses all five controls directly. The modeled battery floor, capacities, losses and electrolyser commitment dynamics remain physical assumptions. Neither daily stock restoration nor an extra operator reserve is imposed. A baseline's own operating rules constrain that baseline only.</p><p>Good performance here establishes useful control in this seven-day simulation. It does not establish an optimal schedule, realistic downstream thermal cycling or indefinite operation. Plausible assumptions and evidence that control choices matter are the benchmark’s purpose. Removing battery access from an existing policy tests its dependence; it is not a retrained batteryless optimum.</p><details><summary>Reproducibility and artifact provenance</summary><pre id="provenance"></pre></details></section>
<footer id="footer"></footer></main>
<script id="payload" type="application/octet-stream">__PAYLOAD__</script>
<script>
(async()=>{'use strict';
const $=id=>document.getElementById(id),set=(id,s)=>{$(id).textContent=s},esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
try{
if(!('DecompressionStream' in window))throw new Error('This offline file needs a current browser with gzip decompression support. Open it in a recent Chrome, Edge, Firefox or Safari.');
const raw=Uint8Array.from(atob($('payload').textContent.trim()),c=>c.charCodeAt(0));
const data=JSON.parse(await new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream('gzip'))).text());
$('payload').remove();const I=Object.fromEntries(data.fields.map((name,i)=>[name,i]));
let key=data.default,minute=0,zoomBattery=false,plots=[],lastWindow='';
const fmt=(x,n=0)=>x===null||x===undefined||!Number.isFinite(x)?'Unavailable':new Intl.NumberFormat('en-GB',{minimumFractionDigits:n,maximumFractionDigits:n}).format(x);
const name=id=>id==='daylight'?'Daylight rules':id==='storage'?'Storage rules':id==='daylight-persistence'?'Persistence (supplementary)':id.startsWith('ppo30-initial-')?'PPO30 initial · '+id.slice(14):id.startsWith('ppo25-')?'PPO25 · '+id.slice(6):id.startsWith('ppo30-')?'PPO30 · '+id.slice(6):id;
const condition=id=>({'d25-sampled':'25 kg/h · sampled opening charge','d25-floor':'25 kg/h · 5% opening charge','d30-sampled':'30 kg/h · sampled opening charge','d30-floor':'30 kg/h · 5% opening charge','d25-sampled-no-battery':'25 kg/h · battery access removed'}[id]||id);
const splitName=s=>s==='held_out'?'Unseen test scenarios':s==='development'?'Development scenarios':'Training fixture only';
const order=id=>id==='daylight'?0:id==='storage'?1:id.startsWith('ppo25-')?2:id.startsWith('ppo30-')?3:4;
const sortControllers=(a,b)=>order(a)-order(b)||a.localeCompare(b);
const record=()=>data.records[key],val=(row,name)=>row[I[name]],currentCondition=()=>record().summary.condition;
const metric=(g,k)=>g.metrics[k]?.all_case||{},mean=(g,k)=>metric(g,k).mean;
const options=(id,values,label=x=>x,all=false)=>{const old=$(id).value;$(id).innerHTML=(all?'<option value="">All</option>':'')+values.map(v=>`<option value="${esc(v)}">${esc(label(v))}</option>`).join('');if(values.includes(old))$(id).value=old};
const rowValue=(r,k)=>r.execution_complete?(r.economic_metrics?.[k]??r.physical_metrics?.[k]??null):null;
const download=(filename,content,type='text/plain')=>{const url=URL.createObjectURL(new Blob([content],{type})),a=document.createElement('a');a.href=url;a.download=filename;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000)};
const csvCell=v=>{const s=String(v??'');return /[",\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s};
const description=r=>{if(r.controller==='daylight-persistence')return 'Supplementary persistence input: the unchanged daylight rule receives current solar availability instead of the 0–1 h mean, 0–1 h p10 and 1–6 h mean forecasts. All other inputs and operating rules are unchanged.';if(r.controller_kind==='baseline')return r.controller==='daylight'?'Inventory-aware solar allocation, with no battery use.':'Inventory-aware allocation with explicit charging, startup support and hot standby.';if(r.checkpoint_timesteps===0)return 'Untrained initialization; included to assess learning.';const source=r.controller.startsWith('ppo25-')?'25':'30';return `Trained at ${source} kg/h for ${fmt(r.checkpoint_timesteps)} transitions. ${+source!==r.variant.delivery_rate_kg_h?'This demand is a transfer test. ':''}${r.variant.opening_battery==='floor'?'Every PPO model was trained with opening charge sampled at 25–65%; starting at 5% tests both initial-condition transfer and reduced available energy. ':''}${r.variant.battery_mode==='disabled'?'Battery requests are removed after prediction; the policy is unchanged.':''}`};
const actualTraces=Object.keys(data.records).filter(k=>data.records[k].minutes),conditions=[...new Set(Object.values(data.records).map(r=>r.summary.condition))];
const allSplits=[...new Set(data.rows.map(r=>r.split))].sort((a,b)=>({'held_out':0,'development':1,'train':2}[a]-{'held_out':0,'development':1,'train':2}[b]));
set('context',`${data.fixture?'Training fixture':'Completed comparison'}\nSeven-day episodes\n15-minute controls · 1-minute physics\n${fmt(data.rows.filter(r=>!r.supplementary).length)} main case outcomes${data.supplement?' + 32 supplementary':''}`);$('context').style.whiteSpace='pre-line';
set('scenario-stamp',`${data.fixture?'Training fixture':'Prescribed test scenario'} ${data.scenario}`);
if(data.fixture){$('fixture').innerHTML='<div class="warning"><strong>Training-only interface fixture.</strong> These recordings validate the inspector. They are not development or held-out results, and they do not support comparison claims.</div>';$('overview-section').classList.add('hidden')}
options('split',allSplits,splitName);options('overview-condition',[...new Set(data.rows.map(r=>r.condition))],condition);options('condition',conditions,condition);
options('case-split',allSplits,splitName,true);options('case-condition',[...new Set(data.rows.map(r=>r.condition))],condition,true);options('case-controller',[...new Set(data.rows.map(r=>r.controller))].sort(sortControllers),name,true);
$('condition').value=currentCondition();
function setupControllers(){const labels=Object.values(data.records).filter(r=>r.summary.condition===$('condition').value).map(r=>r.summary.controller).sort(sortControllers);options('controller',labels,name);const old=record().summary.controller;const chosen=labels.includes(old)?old:labels[0];$('controller').value=chosen;key=$('condition').value+'|'+chosen}
setupControllers();
function learningAndSupplement(){
const learning=[101,211,307].map(seed=>{const initial=data.groups.find(g=>g.split==='development'&&g.condition==='d30-sampled'&&g.controller==='ppo30-initial-'+seed),final=data.groups.find(g=>g.split==='development'&&g.condition==='d30-sampled'&&g.controller==='ppo30-'+seed),row=data.rows.find(r=>r.split==='development'&&r.condition==='d30-sampled'&&r.controller==='ppo30-'+seed);return{seed,initial,final,row}});
if(learning.some(p=>p.initial||p.final)){$('learning-section').classList.remove('hidden');$('learning').innerHTML=learning.map(p=>`<tr><td>${p.seed}</td><td>${p.initial?p.initial.delivery_successes+'/'+p.initial.n_scenarios:'Unavailable'}</td><td>${p.initial?fmt(mean(p.initial,'net_operating_value_eur')):'Unavailable'}</td><td>${fmt(p.row?.checkpoint_timesteps)}</td><td>${p.final?p.final.delivery_successes+'/'+p.final.n_scenarios:'Unavailable'}</td><td>${p.final?fmt(mean(p.final,'net_operating_value_eur')):'Unavailable'}</td></tr>`).join('')}
if(data.supplement){const s=data.supplement,c=s.criterion,m=c.forecast_minus_persistence_eur;$('supplement-section').classList.remove('hidden');$('supplement').innerHTML=`<tr><td>Daylight with issued forecasts</td><td>${s.reference_delivery_successes}/32</td><td rowspan="2">${fmt(m.mean)}<span class="sub">${m.ci95?'95% interval ['+fmt(m.ci95[0])+', '+fmt(m.ci95[1])+']':'No interval available'}</span></td></tr><tr><td>Persistence (supplementary)</td><td>${s.persistence_delivery_successes}/32</td></tr>`;set('supplement-criterion',`${c.practical_evidence_bar_met?'The predeclared practical evidence bar was met.':'The predeclared practical evidence bar was not met.'} Forecasts rescued ${c.forecast_rescues} scenario(s) and regressed ${c.forecast_regressions}. The bar required at least four rescues with no regressions, or a contribution gain of at least €250/week with a positive paired interval and no fewer successful weeks; all pairs must complete and pass their audits. This supplement was declared after development and before fresh test outcomes.`)}
}
learningAndSupplement();
function overview(){const split=$('split').value,cond=$('overview-condition').value,groups=data.groups.filter(g=>g.split===split&&g.condition===cond).sort((a,b)=>sortControllers(a.controller,b.controller));set('overview-caption',`${splitName(split)} · ${condition(cond)}. Values are means over all prescribed scenarios; an incomplete monetary outcome leaves that panel mean unavailable. Brackets give 95% scenario-bootstrap intervals for contribution.`);$('overview').innerHTML=groups.map(g=>{const e=metric(g,'net_operating_value_eur'),cases=data.rows.filter(r=>r.split===split&&r.condition===cond&&r.controller===g.controller).sort((a,b)=>a.seed-b.seed),recordKey=cond+'|'+g.controller;return `<tr class="${recordKey===key?'selected':''}" ${data.records[recordKey]?`data-record="${esc(recordKey)}" tabindex="0"`:''}><td>${esc(name(g.controller))}<span class="sub">${esc(g.controller==='daylight-persistence'?'Supplement: three forecast inputs replaced':g.controller.startsWith('ppo30-initial-')?'Random initialization':g.controller.startsWith('ppo25-')&&cond.startsWith('d30')?'25 kg/h policy transferred to 30 kg/h':g.controller.startsWith('ppo30-')?'Trained at 30 kg/h':g.controller==='daylight'?'No battery requests':g.controller==='storage'?'Battery access and operating rules':'Trained at 25 kg/h')}</span></td><td>${g.delivery_successes}/${g.n_scenarios}<span class="sub outcomes">${cases.map(r=>`<span class="${!r.execution_complete?'incomplete':r.every_delivery_passed?'':'failed'}" title="Scenario ${r.seed}: ${!r.execution_complete?'incomplete':r.every_delivery_passed?'all deliveries met':'delivery missed'}"></span>`).join('')}</span></td><td>${fmt(e.mean)}<span class="sub">${e.ci95?'['+fmt(e.ci95[0])+', '+fmt(e.ci95[1])+']':'No interval available'}</span></td><td>${fmt(mean(g,'unmet_kg'),1)}</td><td>${fmt(mean(g,'operating:battery_charge_kwh'))}</td><td>${fmt(mean(g,'operating:battery_discharge_kwh'))}</td><td>${fmt(mean(g,'electrolyser_forced_stops'),1)}</td></tr>`}).join('');
const paired=data.comparisons.filter(p=>p.split===split&&p.condition===cond&&p.reference==='daylight');set('paired',paired.map(p=>{const m=p.contribution.all_case;return `${p.treatment_controllers[0].startsWith('ppo25')?'PPO25':'PPO30'} mean across three fixed training seeds minus daylight: ${fmt(m.mean)} €/week${m.ci95?' [95% interval '+fmt(m.ci95[0])+', '+fmt(m.ci95[1])+']':''}.`}).join(' ')+' Intervals condition on these policies; they do not include uncertainty from retraining.');
const rs=data.rows.filter(r=>r.split===split&&r.condition===cond);set('auditline',`${rs.filter(r=>r.execution_complete).length}/${rs.length} completed attempts; ${rs.filter(r=>r.physical_invariants_passed).length}/${rs.length} physical audits passed; ${rs.filter(r=>r.economic_audit?.passed).length}/${rs.length} economic audits passed.`);
document.querySelectorAll('[data-record]').forEach(el=>{const go=()=>{key=el.dataset.record;$('condition').value=currentCondition();setupControllers();$('controller').value=data.records[key].summary.controller;drawStatic();$('trace-section').scrollIntoView({behavior:'auto'})};el.onclick=go;el.onkeydown=e=>{if(e.key==='Enter'){go()}}})}
function casesTable(){const rows=data.rows.filter(r=>(!$('case-split').value||r.split===$('case-split').value)&&(!$('case-condition').value||r.condition===$('case-condition').value)&&(!$('case-controller').value||r.controller===$('case-controller').value));set('cases-caption',`${rows.length} of ${data.rows.length} retained cases shown`);$('cases').innerHTML=rows.map(r=>`<tr><td>${esc(name(r.controller))}<span class="sub">${esc(condition(r.condition))} · ${esc(splitName(r.split))}</span></td><td>${r.seed}</td><td class="${r.every_delivery_passed?'good':'bad'}">${r.execution_complete?(r.every_delivery_passed?'All met':'Missed'):'Incomplete'}</td><td>${fmt(rowValue(r,'net_operating_value_eur'))}</td><td>${fmt(rowValue(r,'unmet_kg'),1)}</td><td>${r.physical_invariants_passed?'Pass':'Fail'} / ${r.economic_audit?.passed?'Pass':'Fail'}${r.failure?`<span class="sub">${esc(r.failure.type+': '+r.failure.message)}</span>`:''}</td></tr>`).join('')}
const ranges=()=>{const count=record().minutes.length-1,span=+$('window').value*60;if(span>=count)return[0,count];const start=Math.max(0,Math.min(count-span,Math.floor(minute/span)*span));return[start,start+span]};
function bandSegments(rows,predicate,start,end){const result=[];let first=null;for(let i=start;i<=end;i++){const match=predicate(rows[i]);if(match&&first===null)first=i;if(first!==null&&(!match||i===end)){result.push([first,match?i:i-1]);first=null}}return result}
function chart(id,series,{max=null,min=0,reference=null,floor=null,shading=true,height=190,percentage=false}={}){const el=$(id),r=record(),rows=r.minutes,[start,end]=ranges(),W=Math.max(380,el.parentElement.clientWidth-30),H=height,L=49,R=13,T=10,B=31;el.setAttribute('viewBox',`0 0 ${W} ${H}`);const x=n=>L+(n-start)/(end-start||1)*(W-L-R);let hi=max;if(hi===null){hi=Math.max(...series.map(s=>{let m=0;for(let i=start;i<=end;i++)m=Math.max(m,(val(rows[i],s.field)* (s.scale||1)));return m}))*1.08}if(hi<=min)hi=min+1;const y=n=>H-B-(n-min)/(hi-min)*(H-T-B);let html='';if(shading){for(const [a,b]of bandSegments(rows,m=>val(m,'solar_available_kw')<=1e-9,start,end))html+=`<rect x="${x(a)}" y="${T}" width="${Math.max(1,x(b)-x(a))}" height="${H-T-B}" fill="#eaf0f4"/>`;for(const[a,b]of bandSegments(rows,m=>val(m,'dac_offline')>0,start,end))html+=`<rect x="${x(a)}" y="${T}" width="${Math.max(1,x(b)-x(a))}" height="${H-T-B}" fill="#b45c4c" opacity=".10"/>`}
for(let j=0;j<4;j++){const v=min+(hi-min)*j/3;html+=`<line x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}" stroke="#dce5eb"/><text x="${L-8}" y="${y(v)+4}" text-anchor="end" fill="#617888" font-size="11">${fmt(v,percentage&&hi-min<10?1:0)}</text>`}for(let j=0;j<=4;j++){const n=start+(end-start)*j/4;html+=`<text x="${x(n)}" y="${H-8}" text-anchor="middle" fill="#617888" font-size="11">${fmt(n/60,(end-start)<1440?1:0)} h</text>`}
for(const [v,stroke,dash]of [[reference,'#7e929f','5 4'],[floor,'#ae5547','2 3']])if(v!==null&&v>=min&&v<=hi)html+=`<line x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}" stroke="${stroke}" stroke-dasharray="${dash}"/>`;
const stride=Math.max(1,Math.ceil((end-start)/1200));for(const s of series){let path='';for(let i=start;i<=end;i+=stride){path+=(path?'L':'M')+x(i).toFixed(2)+','+y(val(rows[i],s.field)*(s.scale||1)).toFixed(2)}if((end-start)%stride)path+='L'+x(end).toFixed(2)+','+y(val(rows[end],s.field)*(s.scale||1)).toFixed(2);html+=`<path d="${path}" fill="none" stroke="${s.color}" stroke-width="1.5" stroke-linejoin="round"/>`}
html+=`<line data-cursor x1="${x(minute)}" x2="${x(minute)}" y1="${T}" y2="${H-B}" stroke="#183345" stroke-width="1"/><rect data-click x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="transparent" style="cursor:crosshair"/>`;el.innerHTML=html;el.onclick=e=>{const box=el.getBoundingClientRect(),px=(e.clientX-box.left)/box.width*W;minute=Math.round(start+Math.max(0,Math.min(1,(px-L)/(W-L-R)))*(end-start));drawCursor()};plots.push({el,x,start,end})}
function drawPlots(){if(!record().minutes)return;plots=[];const r=record(),[start,end]=ranges();lastWindow=`${start}:${end}`;chart('power',[{field:'solar_available_kw',color:'#ad7815'},{field:'electrolyser_power_kw',color:'#087c83'},{field:'dac_power_kw',color:'#4c6caf'},{field:'methanator_power_kw',color:'#ae5547'}],{max:5200,height:225});let min=0,max=100;if(zoomBattery){let lo=100,hi=0;for(let i=start;i<=end;i++){const v=val(r.minutes[i],'battery_soc')*100;lo=Math.min(lo,v);hi=Math.max(hi,v)}const pad=Math.max(.15,(hi-lo)*.12);min=Math.max(0,Math.min(5,lo)-pad);max=Math.min(100,hi+pad)}chart('soc',[{field:'battery_soc',scale:100,color:'#087c83'}],{min,max,reference:val(r.minutes[0],'battery_soc')*100,floor:5,percentage:true});set('soc-caption',`${zoomBattery?'Zoomed vertical scale':'Full 0–100% scale'}. Dashed line: opening charge. Dotted line: 5% physical floor. The readout always shows the exact selected value.`);chart('routing',[{field:'solar_to_battery_kw',color:'#4c6caf'},{field:'battery_to_electrolyser_kw',color:'#087c83'},{field:'solar_curtailed_kw',color:'#ad7815'}]);for(const[id,field,color]of [['h2','h2_level_kg','#087c83'],['co2','co2_level_kg','#4c6caf'],['ch4','ch4_level_kg','#ae5547']])chart(id,[{field,color}],{reference:val(r.minutes[0],field)});chart('production',[{field:'ch4_production_kg_h',color:'#ae5547'}],{max:160});}
function forecast(command){const obs=command.observation,el=$('forecast'),W=Math.max(390,el.parentElement.clientWidth-30),H=220,L=52,R=15,T=16,B=44,group=(W-L-R)/4,y=v=>H-B-v/5000*(H-T-B);let html='';for(const v of[0,2500,5000])html+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="#dce5eb"/><text x="${L-8}" y="${y(v)+4}" text-anchor="end" fill="#617888" font-size="11">${fmt(v)}</text>`;['0–1 h','1–6 h','6–12 h','12–24 h'].forEach((label,i)=>{const mean=obs[34+2*i]*5000,p10=obs[35+2*i]*5000,x=L+group*i+group*.20,bw=group*.25;html+=`<rect x="${x}" y="${y(mean)}" width="${bw}" height="${H-B-y(mean)}" fill="#ad7815"/><rect x="${x+bw+3}" y="${y(p10)}" width="${bw}" height="${H-B-y(p10)}" fill="#4c6caf"/><text x="${L+group*(i+.5)}" y="${H-24}" text-anchor="middle" fill="#516879" font-size="11">${label}</text><text x="${L+group*(i+.5)}" y="${H-8}" text-anchor="middle" fill="#516879" font-size="10">${fmt(mean)} / ${fmt(p10)}</text>`});el.setAttribute('viewBox',`0 0 ${W} ${H}`);el.innerHTML=html}
function drawCursor(){const r=record();if(!r.minutes)return;minute=Math.max(0,Math.min(r.minutes.length-1,minute));const bounds=ranges();if(`${bounds[0]}:${bounds[1]}`!==lastWindow)drawPlots();const m=r.minutes[minute],day=Math.floor((minute+360)/1440)+1,t=(minute+360)%1440,hr=Math.floor(t/60),mm=t%60;set('clock',`Day ${day} · ${String(hr).padStart(2,'0')}:${String(mm).padStart(2,'0')}`);set('elapsed',`${fmt(minute/60,2)} / ${fmt((r.minutes.length-1)/60)} h`);$('timeline').value=minute;const commandIndex=Math.min(Math.floor(minute/15),r.commands.length-1),command=r.commands[commandIndex],start=commandIndex*15;set('status',`${val(m,'solar_available_kw')<=1e-9?'Night / no available solar power':'Available sunlight: '+fmt(val(m,'solar_available_kw'))+' kW'}. ${val(m,'dac_offline')?'DAC outage is active.':'DAC is available.'} ${r.summary.every_delivery_passed?'This policy met every delivery in this week.':'At least one delivery was missed or the attempt is incomplete.'}`);
$('readout').innerHTML=[['Battery charge',fmt(val(m,'battery_soc')*100,3)+'%'],['Battery support',fmt(val(m,'battery_to_electrolyser_kw'),1)+' kW'],['Methane stored',fmt(val(m,'ch4_level_kg'),1)+' kg'],['Methane production',fmt(val(m,'ch4_production_kg_h'),1)+' kg/h']].map(([a,b])=>`<div><span>${a}</span><strong>${b}</strong></div>`).join('');
const requested=command.direct_control.requested_action,applied=command.direct_control.intervention_applied_action||requested,energy=command.direct_control.physical_applied_energy_kwh;const modeNames=['legacy','off','cold starting','running','hot standby','hot starting','cooldown'];const mode=modeNames[command.observation.slice(42,49).findIndex(v=>v>.5)]||'unknown';set('command-time',`Decision ${commandIndex+1}, chosen at elapsed ${fmt(start/60,2)} h and held until ${fmt((start+15)/60,2)} h. Electrolyser mode at that decision: ${mode}.`);
$('commands').innerHTML=[['Battery charging','battery_share','battery'],['Electrolyser','electrolyser_share','electrolyser'],['Carbon capture','dac_share','dac'],['Methanator','methanator_share','methanator'],['Battery support','battery_support_kw','battery_support']].map(([label,k,e])=>`<tr><td>${label}</td><td>${fmt(requested[k]*(k.endsWith('_share')?100:1),2)}${k.endsWith('_share')?'%':' kW'}</td><td>${fmt(applied[k]*(k.endsWith('_share')?100:1),2)}${k.endsWith('_share')?'%':' kW'}</td><td>${fmt(energy[e]/.25,1)}</td></tr>`).join('');set('command-note',`${r.summary.variant.battery_mode==='disabled'?'Battery charging and support were explicitly zeroed after the policy requested them. ':'No experimental action removal was applied. '}Solar shares are held for 15 minutes, so requested kW varies with sunlight. Actual mean power is the energy consumed over that entire interval; it differs from the selected minute.`);forecast(command);set('forecast-replacement',command.replacements?'This policy did not consume the three issued values it uses for startup decisions. Replacements (issued → controller input): '+command.replacements.map(v=>v.field.replace('solar_forecast_','').replaceAll('_',' ')+' '+fmt(v.original*5000)+' → '+fmt(v.replacement*5000)+' kW').join('; ')+'. The other forecast slots are ignored by this rule.':'These are the original causal public forecasts available at this decision.');
set('detail',JSON.stringify({controller:r.summary.controller,condition:r.summary.condition,scenario:data.scenario,selected_minute:minute,decision_index:commandIndex,public_observation:command.observation,public_observation_sha256:command.observation_sha256,normalized_observation_sha256:command.normalized_observation_sha256,persistence_transformed_observation:command.transformed_observation,persistence_replacements:command.replacements,raw_policy_mean:command.raw_mean,bounded_action:command.action,requested_and_applied_control:command.direct_control,battery_intervention:command.battery_intervention,interval_reward_eur:command.reward_eur,physical_audit:r.summary.minute_audit,economic_audit:r.summary.economic_audit,recording:r.recording},null,2));for(const p of plots){const line=p.el.querySelector('[data-cursor]');line.setAttribute('x1',p.x(minute));line.setAttribute('x2',p.x(minute))}}
function drawStatic(){const r=record();set('policy-note',description(r.summary));$('trace-content').classList.toggle('hidden',!r.minutes);$('trace-missing').classList.toggle('hidden',!!r.minutes);if(!r.minutes){set('trace-missing',r.unavailable);return}minute=Math.min(minute,r.minutes.length-1);$('timeline').max=r.minutes.length-1;const labels={delivered_service_value:'Delivered methane value',unmet_delivery_penalty:'Missed-delivery penalty',solar_opportunity_cost:'Solar opportunity cost',battery_throughput_reserve:'Battery throughput wear',electrolyser_life_reserve:'Electrolyser life consumption',methane_endowment_liability_change:'Material-depletion liability change',battery_endowment_liability_change:'Battery-depletion liability change'};$('economics').innerHTML=Object.entries(r.summary.economic_metrics.components_eur).map(([k,v])=>`<tr><td>${esc(labels[k]||k)}</td><td>${fmt(v,2)}</td></tr>`).join('')+`<tr><td><strong>Modeled contribution</strong></td><td><strong>${fmt(r.summary.economic_metrics.net_operating_value_eur,2)}</strong></td></tr>`;
$('deliveries').innerHTML=r.delivery.settlements.map((d,i)=>`<tr><td>${i+1} · ${fmt(d.due_step/60)} h</td><td>${fmt(d.requested_kg)}</td><td>${fmt(d.delivered_kg,1)}</td><td class="${d.unmet_kg>1e-9?'bad':'good'}">${fmt(d.unmet_kg,1)}</td></tr>`).join('');$('stock-ledger').innerHTML=[['Battery / kWh','battery_energy_kwh'],['Hydrogen / kg','h2_level_kg'],['CO₂ / kg','co2_level_kg'],['Methane / kg','ch4_level_kg']].map(([label,k])=>`<tr><td>${label}</td><td>${fmt(r.summary.initial_state[k],1)}</td><td>${fmt(r.summary.final_state[k],1)}</td></tr>`).join('');const op=r.summary.operating_metrics;set('night-account',`Across zero-solar minutes: electrolysis consumed ${fmt(op.zero_solar_electrolyser_kwh,1)} kWh; battery support supplied ${fmt(op.zero_solar_battery_discharge_kwh,1)} kWh; methane production was ${fmt(op.zero_solar_ch4_produced_kg,2)} kg. These are observed flows, not a claim that nighttime production is required.`);drawPlots();drawCursor();overview()}
$('condition').onchange=()=>{setupControllers();drawStatic()};$('controller').onchange=()=>{key=$('condition').value+'|'+$('controller').value;drawStatic()};$('window').onchange=()=>{drawPlots();drawCursor()};$('timeline').oninput=e=>{minute=+e.target.value;drawCursor()};$('previous').onclick=()=>{minute-=15;drawCursor()};$('next').onclick=()=>{minute+=15;drawCursor()};$('battery-scale').onclick=()=>{zoomBattery=!zoomBattery;$('battery-scale').setAttribute('aria-pressed',zoomBattery);$('battery-scale').textContent=zoomBattery?'Show 0–100%':'Zoom to variation';drawPlots();drawCursor()};
function jump(predicate,message){const rows=record().minutes;const found=rows.findIndex((m,i)=>i>minute&&predicate(m,i,rows));if(found>=0){minute=found;drawCursor()}else set('status',message)}
$('next-dip').onclick=()=>jump((m,i,rows)=>i>=15&&val(rows[i-15],'solar_available_kw')>=1000&&val(m,'solar_potential_kw')>=1000&&val(m,'solar_available_kw')<=.7*val(rows[i-15],'solar_available_kw'),'No later qualifying daytime solar dip is present.');$('next-night').onclick=()=>jump((m,i,rows)=>val(m,'solar_available_kw')<=1e-9&&i>0&&val(rows[i-1],'solar_available_kw')>1e-9,'No later night begins in this trace.');$('next-fault').onclick=()=>jump((m,i,rows)=>val(m,'dac_offline')>0&&(i===0||!val(rows[i-1],'dac_offline')),'No later DAC outage begins in this trace.');$('next-delivery').onclick=()=>{const d=record().delivery.settlements.find(v=>v.due_step>minute);if(d){minute=d.due_step;drawCursor()}else set('status','No later delivery remains in this trace.')};
$('split').onchange=overview;$('overview-condition').onchange=overview;for(const id of['case-split','case-condition','case-controller'])$(id).onchange=casesTable;
$('download-minutes').onclick=()=>download(`${key.replace('|','-')}-scenario-${data.scenario}-minutes.csv`,[data.fields.join(','),...record().minutes.map(m=>m.join(','))].join('\n'),'text/csv;charset=utf-8');$('download-decisions').onclick=()=>download(`${key.replace('|','-')}-scenario-${data.scenario}-decisions.json`,JSON.stringify(record().commands,null,2),'application/json');$('download-cases').onclick=()=>{const fields=['split','condition','controller','seed','complete','all_deliveries','contribution_eur','missed_kg','physical_audit','economic_audit'];download('comparison-all-case-outcomes.csv',[fields.join(','),...data.rows.map(r=>[r.split,r.condition,r.controller,r.seed,r.execution_complete,r.every_delivery_passed,rowValue(r,'net_operating_value_eur'),rowValue(r,'unmet_kg'),r.physical_invariants_passed,!!r.economic_audit?.passed].map(csvCell).join(','))].join('\n'),'text/csv;charset=utf-8')};
set('provenance',JSON.stringify(data.provenance,null,2));set('footer',`PlantAnAgent · Filippo Gombac and Vittorio Garavelli. ${data.fixture?'Training-only interface fixture.':'Recorded comparison for the commission research note.'} This is a self-contained, read-only file; no simulation, training or network request runs in this view. Built ${data.provenance.created_at}.`);$('loading').classList.add('hidden');$('app').classList.remove('hidden');casesTable();drawStatic();let resize;window.addEventListener('resize',()=>{clearTimeout(resize);resize=setTimeout(()=>{drawPlots();drawCursor()},120)});window.__inspectorQA={ready:true,fixture:data.fixture,cases:data.rows.length,traces:Object.keys(data.records).length,scenario:data.scenario};
}catch(error){$('loading').textContent='Unable to open the recording: '+error.message;console.error(error)}})();
</script></body></html>'''


def write_inspector(data, output):
    output = Path(output)
    sidecar = output.with_suffix(".build.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("Inspector or provenance already exists")
    serialized = json.dumps(data, separators=(",", ":"), allow_nan=False).encode()
    payload = base64.b64encode(gzip.compress(serialized, compresslevel=9, mtime=0)).decode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(HTML.replace("__PAYLOAD__", payload))
    provenance = {"schema": data["schema"] + "-build", "fixture": data["fixture"],
        "scenario": data["scenario"], "cases": len(data["rows"]), "traces": len(data["records"]),
        "data_sha256": hashlib.sha256(serialized).hexdigest(), "html": pin(output),
        **data["provenance"], "recordings": {key: value.get("recording") for key, value in data["records"].items()}}
    with sidecar.open("x") as stream:
        json.dump(provenance, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--forecast-analysis", type=Path, help="Optional supplementary persistence analysis pinned to the supplied main analysis")
    parser.add_argument("--recordings-root", type=Path)
    parser.add_argument("--fixture-recordings", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = assemble(analysis_path=args.analysis, forecast_analysis_path=args.forecast_analysis,
                    recordings_root=args.recordings_root, fixtures=args.fixture_recordings)
    print(json.dumps(write_inspector(data, args.output), indent=2))


if __name__ == "__main__":
    main()

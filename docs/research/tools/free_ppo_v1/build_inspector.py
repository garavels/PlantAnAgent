"""Build a self-contained, read-only inspector from verified direct-control traces.

Final mode requires the frozen five-controller comparison and its full held-out
analysis. Fixture mode accepts existing training recordings only, visibly labels
the output, and never runs a simulator or imports the learning runtime.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

SCHEMA = "free-ppo-v1-evaluation"
CONTROLLERS = ("coverage-direct", "untrained-101", "free-ppo-v1-101", "free-ppo-v1-211", "free-ppo-v1-307")
FIELDS = ("step", "time_s", "solar_available_kw", "solar_to_battery_kw", "solar_curtailed_kw",
          "battery_to_electrolyser_kw", "electrolyser_power_kw", "dac_power_kw", "methanator_power_kw",
          "battery_energy_kwh", "battery_soc", "h2_level_kg", "co2_level_kg", "ch4_level_kg")


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def pin(path):
    path = Path(path)
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def resolve_record(record_pin, root=None):
    original = Path(record_pin["path"])
    candidates = [original] if root is None else list(Path(root).rglob(original.name))
    matches = [p for p in candidates if p.is_file() and pin(p)["sha256"] == record_pin["sha256"]
               and p.stat().st_size == record_pin["bytes"]]
    if len(matches) != 1:
        raise ValueError(f"Expected one checksum-matching recording: {original.name}; found {len(matches)}")
    return matches[0]


def compact_record(path, expected_row=None, fixture=False):
    raw = Path(path).read_bytes()
    record = json.loads(gzip.decompress(raw))
    summary = record["summary"]
    if record.get("schema") != SCHEMA + "-recording" or summary.get("schema") != SCHEMA + "-case":
        raise ValueError("Unexpected recording schema")
    if expected_row is not None and summary != {k: v for k, v in expected_row.items() if k != "recording"}:
        raise ValueError("Analysis row and retained recording disagree")
    if fixture and summary["split"] != "train":
        raise ValueError("Fixture mode is restricted to training recordings")
    minutes = record.get("minute_truth_states")
    if not minutes or canonical_sha(minutes) != summary["retained_minute_truth_sha256"]:
        raise ValueError("Missing or altered retained minute truth")
    decisions = record["decision_trace"]
    if len(minutes) != summary["physical_steps"] + 1 or len(decisions) != summary["decisions"]:
        raise ValueError("Retained trace length differs from summary")
    if len(minutes) != 15 * len(decisions) + 1:
        raise ValueError("Expected fifteen physical minutes per recorded decision")
    for index, state in enumerate(minutes):
        if state["step"] != index or state["time_s"] != minutes[0]["time_s"] + 60 * index:
            raise ValueError("Minute truth is not contiguous")
    for index, decision in enumerate(decisions):
        if decision["decision_index"] != index:
            raise ValueError("Decision order differs")
        final = minutes[(index + 1) * 15]
        if any(final[k] != value for k, value in decision["state"].items()):
            raise ValueError("Decision closing state differs from minute truth")
    if summary["initial_state"] != minutes[0] or summary["final_state"] != minutes[-1]:
        raise ValueError("Opening or closing state differs")
    commands = []
    for decision in decisions:
        evidence = decision["evidence"]
        commands.append({"index": decision["decision_index"], "action": decision["action"],
                         "raw_mean": evidence.get("raw_mean"), "baseline_targets": evidence.get("baseline_targets"),
                         "observation_sha256": evidence.get("observation_sha256"),
                         "normalized_observation_sha256": evidence.get("normalized_observation_sha256"),
                         "direct_control": decision["direct_control"], "reward_eur": decision["reward_eur"]})
    return {"controller": summary["controller"], "summary": summary,
            "minutes": [[m[k] for k in FIELDS] + [int(m["health"]["dac"]["offline"])] for m in minutes],
            "commands": commands, "delivery": record["delivery"], "cycles": record["cycle_settlements"],
            "capacity": {k: minutes[0][k] for k in ("battery_capacity_kwh", "h2_capacity_kg", "co2_capacity_kg", "ch4_capacity_kg")},
            "environment": record["manifest"]["environment"], "recording": pin(path)}


def assemble(*, analysis_path=None, freeze_path=None, recordings_root=None, fixtures=None):
    provenance = {"builder": pin(__file__), "created_at": datetime.now(timezone.utc).isoformat()}
    records = {}
    groups = []
    fixture = bool(fixtures)
    if fixture:
        if analysis_path or freeze_path:
            raise ValueError("Fixture and final comparison inputs cannot be mixed")
        for path in fixtures:
            item = compact_record(path, fixture=True)
            if item["controller"] in records:
                raise ValueError("Duplicate fixture controller")
            records[item["controller"]] = item
        scenario_ids = {r["summary"]["seed"] for r in records.values()}
        if len(scenario_ids) != 1:
            raise ValueError("Fixture controllers must share the same training scenario")
        scenario = scenario_ids.pop()
    else:
        if not analysis_path or not freeze_path:
            raise ValueError("Final inspector requires --analysis and --evaluation-freeze")
        analysis = json.loads(Path(analysis_path).read_text())
        freeze = json.loads(Path(freeze_path).read_text())
        if (freeze.get("schema") != SCHEMA + "-freeze" or
            canonical_sha({k: v for k, v in freeze.items() if k != "sha256"}) != freeze.get("sha256")):
            raise ValueError("Evaluation freeze identity differs")
        if analysis.get("schema") != SCHEMA + "-analysis" or analysis["evaluation_freeze_sha256"] != freeze["sha256"]:
            raise ValueError("Analysis belongs to a different comparison")
        if {k: v for k, v in analysis["evaluation_freeze"].items() if k != "path"} != {k: v for k, v in pin(freeze_path).items() if k != "path"}:
            raise ValueError("Analysis freeze byte pin differs")
        selection = freeze["selection"]
        if tuple(selection["primary_controllers"]) != CONTROLLERS or selection["held_out_seeds"] != list(range(10200000, 10200032)):
            raise ValueError("Unexpected prospective controller/scenario selection")
        scenario = selection["held_out_seeds"][0]
        for controller in CONTROLLERS:
            rows = [r for r in analysis["rows"] if r["split"] == "held_out" and r["controller"] == controller]
            if sorted(r["seed"] for r in rows) != selection["held_out_seeds"]:
                raise ValueError(f"Missing or duplicate held-out outcomes for {controller}")
            row = next(r for r in rows if r["seed"] == scenario)
            if row["evaluation_freeze_sha256"] != freeze["sha256"] or not row.get("endpoint_freeze_sha256"):
                raise ValueError("Selected trace lacks comparison/endpoint identity")
            expected_steps = selection["final_timesteps"] if controller.startswith("free-ppo-v1-") else 0
            if row["checkpoint_timesteps"] != expected_steps:
                raise ValueError("Selected trace does not use its prespecified checkpoint budget")
            item = compact_record(resolve_record(row["recording"], recordings_root), expected_row=row)
            records[controller] = item
            matched = [g for g in analysis["groups"] if g["split"] == "held_out" and g["controller"] == controller]
            if len(matched) != 1 or matched[0]["n_scenarios"] != 32:
                raise ValueError("Missing or duplicate full-panel group")
            group = matched[0]
            if any(group[field] != sum(r[flag] for r in rows) for field, flag in (
                ("complete_weeks", "execution_complete"), ("delivery_successes", "every_delivery_passed"),
                ("delivery_and_physics_successes", "delivery_and_physics_passed"))):
                raise ValueError("Group counts disagree with retained outcomes")
            groups.append(group)
        provenance.update(analysis=pin(analysis_path), evaluation_freeze=pin(freeze_path),
                          evaluation_freeze_sha256=freeze["sha256"], bootstrap=analysis["bootstrap"])
    identities = {(r["summary"]["exogenous_episode_sha256"], r["summary"]["manifest_sha256"]) for r in records.values()}
    if len(identities) != 1:
        raise ValueError("Controllers do not share the same exogenous scenario")
    if not fixture and len({r["summary"]["endpoint_freeze_sha256"] for r in records.values()}) != 1:
        raise ValueError("Controllers do not share the frozen endpoint set")
    return {"schema": "free-ppo-v1-offline-inspector", "fixture": fixture, "scenario": scenario,
            "default": "free-ppo-v1-101" if "free-ppo-v1-101" in records else next(iter(records)),
            "fields": [*FIELDS, "dac_offline"], "records": records, "groups": groups, "provenance": provenance}


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Direct plant control — recorded experiment</title>
<style>
:root{font-family:Aptos,"Segoe UI",-apple-system,BlinkMacSystemFont,sans-serif;color:#183445;background:#f6f8fa;font-size:16px;font-variant-numeric:tabular-nums;--muted:#516675;--line:#cfdae1;--teal:#187f83;--blue:#466db1;--amber:#ad781d;--brick:#a3453d}*{box-sizing:border-box}body{margin:0}main{max-width:1240px;margin:auto;padding:30px 30px 55px}h1{font-size:clamp(28px,3vw,40px);line-height:1.12;letter-spacing:-.025em;margin:12px 0 14px;font-weight:640}h2{font-size:20px;letter-spacing:-.015em;margin:0 0 10px;font-weight:620}h3{font-size:16px;margin:0 0 7px}p{line-height:1.55;margin:8px 0 18px;max-width:90ch;color:var(--muted)}header{display:grid;grid-template-columns:1fr auto;gap:25px;border-bottom:1px solid var(--line);padding-bottom:20px}.brand{font-size:14px;font-weight:600}.case-note{font-size:14px;border-left:2px solid var(--teal);padding:7px 0 7px 16px;align-self:end;max-width:240px}.flow{display:flex;gap:13px;flex-wrap:wrap;align-items:center;font-size:14px;line-height:1.7;margin:21px 0}.flow span{border-bottom:2px solid var(--line)}.flow b{font-weight:500;color:var(--teal)}.controls{display:grid;grid-template-columns:auto minmax(170px,260px) 1fr auto;gap:12px;align-items:center;background:#eaf0f4;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:16px;margin:20px 0 14px}.scrub{grid-column:1/-1;display:flex;align-items:center;gap:14px}.scrub input{flex:1;min-width:100px;accent-color:var(--teal)}label{font-size:14px;font-weight:600}button,select{font:inherit;color:inherit;background:#fff;border:1px solid #adbec8;border-radius:4px;padding:9px 11px;max-width:100%}button{cursor:pointer;font-size:14px}button:hover{background:#e1ebee}button:focus-visible,select:focus-visible,input:focus-visible,summary:focus-visible,a:focus-visible{outline:3px solid #ad781d;outline-offset:3px}.clock{font-size:20px;font-weight:650;justify-self:end}.note{font-size:13px}.status{font-size:14px;padding:4px 0 13px;margin:0;min-height:38px}.status strong{color:#183445}.warning{background:#f9eee7;color:#7d382b;padding:14px 17px;border-left:4px solid var(--brick);line-height:1.5;margin:16px 0}.plot{padding:18px 14px 8px;background:#fff;border:1px solid #e1e7eb;min-width:0}.plot svg{display:block;width:100%;height:auto}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--muted);line-height:1.6;margin:6px 0 2px}.legend span:before{content:"";display:inline-block;width:20px;height:3px;background:var(--c);vertical-align:middle;margin-right:6px}.two{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:20px 0}.inventory{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:18px 0 8px}.section{margin-top:32px}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.5}caption{text-align:left;font-size:13px;color:var(--muted);padding-bottom:10px}th{color:var(--muted);font-weight:550;text-align:left}th,td{padding:10px 7px;border-bottom:1px solid var(--line);vertical-align:top}td:not(:first-child),th:not(:first-child){text-align:right}td:first-child{font-weight:500}tbody tr.selected{background:#e5f0f1}.table-wrap td{white-space:nowrap}.label{font-size:12px;color:var(--muted);font-weight:400;display:block}.pass{color:#146467}.fail{color:#913e35}.muted{color:var(--muted)}.outcome{font-weight:620}.assumptions{border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:19px 0;margin-top:28px}.assumptions p{margin-bottom:10px}.assumptions li{margin:9px 0;color:var(--muted);line-height:1.5}details{margin-top:20px}summary{font-size:15px;font-weight:600;cursor:pointer;line-height:1.6}pre{font-size:12px;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere;max-height:500px;overflow-y:auto;background:#fff;border:1px solid var(--line);padding:16px}footer{border-top:1px solid var(--line);margin-top:30px;padding-top:18px;color:var(--muted);font-size:13px;line-height:1.6}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}.skip{position:absolute;left:12px;top:-60px;background:#fff;padding:10px;z-index:10}.skip:focus{top:12px}.actions{display:flex;gap:8px;flex-wrap:wrap}a{color:#16696d}@media(max-width:760px){main{padding:22px 15px 38px}header{grid-template-columns:1fr;gap:4px}.case-note{max-width:none}.controls{grid-template-columns:1fr 1fr;padding:13px}.controls>label{grid-column:1}.controls>select{grid-column:2}.clock{font-size:16px}.controls>.actions{grid-column:1/-1}.scrub{flex-wrap:wrap}.scrub input{min-width:75%}.two,.inventory{grid-template-columns:1fr;gap:16px}table{font-size:13px}.plot{padding:14px 9px 6px}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}@media print{.controls,.actions,.skip{display:none}body{background:white}main{padding:0}.plot{break-inside:avoid}details{display:none}}
</style></head><body><a class="skip" href="#time-controls">Skip to recording controls</a><main>
<header><div><div class="brand">Plant An Agent · Direct-control experiment</div><h1>From sunlight to methane,<br>one decision at a time.</h1><p>A policy chooses how to share solar power and when to draw from the battery. These recordings show what the simulated plant actually did, minute by minute, over a delivery week.</p></div><div class="case-note" id="case-note"></div></header>
<div id="fixture-warning"></div>
<div class="flow" aria-label="Plant material and energy path"><span>Sunlight + battery</span><b>→</b><span>Electrolyser → hydrogen</span><b>+</b><span>Air capture → CO₂</span><b>→</b><span>Methanator → stored methane → deliveries</span></div>
<p class="note">Direct air capture (DAC) collects CO₂ from air. Battery support powers the electrolyser; the other process units use solar power. The controller acts every 15 minutes; physical equipment and inventories evolve every minute.</p>
<section class="controls" id="time-controls" aria-label="Recording controls"><label for="policy">Controller</label><select id="policy"></select><div class="actions"><button id="previous">−15 minutes</button><button id="next">+15 minutes</button><button id="delivery-next">Next delivery</button><button id="outage-next">Next DAC outage</button></div><output class="clock" id="clock" for="timeline"></output><div class="scrub"><label for="timeline">Recording time</label><input id="timeline" type="range" min="0" max="10080" value="0" step="1"><output id="elapsed" for="timeline"></output></div></section>
<p class="status" id="status" role="status" aria-live="polite"></p>
<section class="plot"><h2>Power actually used / kW</h2><svg id="power" role="img" aria-label="Recorded solar availability and physical process power over elapsed hours"></svg><div class="legend"><span style="--c:#ad781d">Solar available</span><span style="--c:#187f83">Electrolyser</span><span style="--c:#466db1">Air capture</span><span style="--c:#a3453d">Methanator</span></div><p class="note">The electrolyser total includes solar and battery power. Shading marks actual DAC outages. The vertical cursor selects an exact recorded minute; the horizontal axis is elapsed hours from the 06:00 start.</p></section>
<div class="two"><section class="plot"><h2>Battery and unused sunlight / kW</h2><svg id="routing" role="img" aria-label="Actual battery charge, battery support and solar curtailment"></svg><div class="legend"><span style="--c:#466db1">Battery charge</span><span style="--c:#187f83">Battery support</span><span style="--c:#ad781d">Curtailed solar</span></div></section><section><h2>At the selected minute</h2><p class="note">Power describes the minute just completed. Stocks are measured after that minute, including any scheduled delivery.</p><table><thead><tr><th scope="col">Quantity</th><th scope="col">Recorded value</th></tr></thead><tbody id="readout"></tbody></table></section></div>
<section class="section"><h2>Stored energy and feedstock</h2><div class="inventory"><section class="plot"><h3>Battery / kWh</h3><svg id="stock0" role="img" aria-label="Battery inventory and opening reference"></svg></section><section class="plot"><h3>Hydrogen / kg</h3><svg id="stock1" role="img" aria-label="Hydrogen inventory and opening reference"></svg></section><section class="plot"><h3>CO₂ / kg</h3><svg id="stock2" role="img" aria-label="Carbon dioxide inventory and opening reference"></svg></section><section class="plot"><h3>Methane / kg</h3><svg id="stock3" role="img" aria-label="Methane inventory and opening reference"></svg></section></div><p class="note">Dashed lines mark the opening stock, for interpretation. Returning each stock every day is <strong>descriptive only</strong>: it is neither an action restriction nor a success condition. The battery's dotted lower line is its modeled 5% physical floor.</p></section>
<div class="two section"><section><h2>Requested control and physical response</h2><p class="note" id="commandtime"></p><div class="table-wrap"><table><thead><tr><th scope="col">Destination</th><th scope="col">Solar share</th><th scope="col">Applied mean kW</th></tr></thead><tbody id="commands"></tbody></table></div><p class="note" id="support"></p><p class="note">Four action coordinates request solar shares; the fifth requests battery support. The same solar share is held across the 15-minute interval, so its requested kW follows changing sunlight. Physical response is the interval energy divided by 0.25 hours.</p></section><section><h2>Delivered methane</h2><svg id="delivery-plot" role="img" aria-label="Cumulative requested and delivered methane at each settlement"></svg><div class="legend"><span style="--c:#183445">Requested</span><span style="--c:#187f83">Delivered</span></div><p class="note" id="delivery-note"></p><details><summary>All recorded delivery settlements</summary><div class="table-wrap"><table><thead><tr><th scope="col">Elapsed h</th><th scope="col">Due kg</th><th scope="col">Delivered kg</th><th scope="col">Unmet kg</th></tr></thead><tbody id="deliveries"></tbody></table></div></details></section></div>
<section class="section"><h2>This scenario, all controllers</h2><p class="note">The scenario was chosen by its index before test outcomes. The default display is PPO seed 101, fixed in advance; it is not selected for performance. Wear counters below cover the recorded duration.</p><div class="table-wrap"><table><thead><tr><th scope="col">Controller</th><th scope="col">Week outcome</th><th scope="col">Delivered kg</th><th scope="col">Unmet kg</th><th scope="col">Cold / hot starts</th><th scope="col">Aborted / forced stops</th><th scope="col">Battery cycles</th><th scope="col">Net value EUR</th></tr></thead><tbody id="outcomes"></tbody></table></div><p class="note">Battery cycles are equivalent full cycles. Electrolyser starts, aborted starts and forced stops affect modeled wear; they are reported, not separate success gates. Net value is displayed for complete weeks only.</p></section>
<div class="two section"><section><h2>Economic account for this recording / EUR</h2><table><thead><tr><th scope="col">Signed contribution</th><th scope="col">Value</th></tr></thead><tbody id="economics"></tbody></table><p class="note">Positive delivery value is reduced by unmet delivery, equipment wear and ending inventory liabilities. Opening inventories are an endowment, not free product. Positive excess ending stocks receive no additional salvage credit.</p></section><section><h2>Daily stock changes</h2><p class="note">Shortfall relative to each opening stock at a daily boundary, truncated at zero. These values explain resource use and do not gate this experiment.</p><div class="table-wrap"><table><thead><tr><th scope="col">Elapsed day</th><th scope="col">Battery kWh</th><th scope="col">H₂ kg</th><th scope="col">CO₂ kg</th><th scope="col">CH₄ kg</th></tr></thead><tbody id="daily"></tbody></table></div></section></div>
<section class="section" id="panel-section"><h2>The full held-out comparison</h2><p class="note">Each controller faces the same 32 environmental scenarios. A successful week completes all 28 deliveries and all exact-minute physical checks. Intervals describe scenario variation with policies held fixed; they do not estimate uncertainty from retraining.</p><div class="table-wrap"><table><thead><tr><th scope="col">Controller</th><th scope="col">Successful weeks</th><th scope="col">Mean unmet kg</th><th scope="col">Mean net value EUR [95% CI]</th></tr></thead><tbody id="comparison"></tbody></table></div></section>
<section class="assumptions"><h2>What this simulation assumes</h2><ul><li>The learned controllers begin from random parameters. They receive 135 current public observation values and no handcrafted coverage targets. The baseline translates a handwritten coverage policy into this same direct-action runtime.</li><li>Actions are clipped to their supported range; four nonnegative solar requests are divided by their sum when needed to conserve available solar. Battery support is requested independently, up to the modeled 2,500 kW rating. There is no discretionary reserve above the battery's physical 5% floor.</li><li>Power ratings, battery efficiency, tank capacities, material requirements, faults and electrolyser startup dynamics still apply. Process-refused solar is curtailed, not reassigned to battery charging. Existing charge/support overlap cancellation remains part of the modeled electrical bus.</li><li>Dedicated solar has zero marginal purchase price in this accounting. Fixed capital costs are unestimated and excluded. Wear and inventory-liability coefficients are modeling assumptions, so operating value is not a commercial profit forecast.</li></ul></section>
<details><summary>Exact action and verification evidence</summary><p class="note">The policy mean is shown separately from its clipped action and the mapped physical request. Full observation vectors remain in the source recording; their hashes are retained here.</p><pre id="evidence"></pre><button id="download">Download selected minute series as CSV</button></details>
<details><summary>Source checksums and build provenance</summary><pre id="provenance"></pre></details>
<footer>This page contains recorded data and works offline. It does not run a policy or simulate future behavior. Minute values are retained at source precision; labels are rounded for readability. The display can be navigated with the controller selector, arrow keys on the timeline, and the step buttons.</footer>
</main><script id="data" type="application/json">__DATA__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),ix=Object.fromEntries(data.fields.map((f,i)=>[f,i]));
let key=data.default,minute=0,plots=[];
const pretty=k=>k==='coverage-direct'?'Coverage baseline':k==='untrained-101'?'Untrained network (101)':`PPO · seed ${k.split('-').at(-1)}`;
const fmt=(n,d=1)=>n===null||n===undefined?'—':Number(n).toLocaleString('en',{maximumFractionDigits:d});
const val=(m,f)=>m[ix[f]],record=()=>data.records[key],text=(id,v)=>{$(id).textContent=v};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const series=(r,f)=>r.minutes.map(m=>[val(m,'step')/60,val(m,f)]);
const palette=['#ad781d','#187f83','#466db1','#a3453d'];
for(const k of Object.keys(data.records)){const option=document.createElement('option');option.value=k;option.textContent=pretty(k);$('policy').append(option)}$('policy').value=key;
text('case-note',data.fixture?`Training fixture · scenario ${data.scenario}. Short engineering trace, not a research result.`:`Scenario ${data.scenario} · the first prespecified held-out case. All five controllers share the same weather and faults.`);
if(data.fixture){$('fixture-warning').className='warning';text('fixture-warning','Training fixture preview. This short recording tests the viewer only. It is not a complete week and must not be used to infer final policy performance.');$('panel-section').hidden=true}
$('provenance').textContent=JSON.stringify({...data.provenance,recordings:Object.fromEntries(Object.entries(data.records).map(([k,r])=>[k,r.recording]))},null,2);
function table(id,rows){$(id).innerHTML=rows.map(row=>'<tr>'+row.map((v,i)=>`<${i?'td':'th scope="row"'}>${esc(v)}</${i?'td':'th'}>`).join('')+'</tr>').join('')}
function shadeIntervals(r){const result=[];let start=null;for(const m of r.minutes){const h=val(m,'step')/60;if(val(m,'dac_offline')&&start===null)start=h;if(!val(m,'dac_offline')&&start!==null){result.push([start,h]);start=null}}if(start!==null)result.push([start,val(r.minutes.at(-1),'step')/60]);return result}
function plot(id,lines,{height=225,refs=[],shades=true}={}){const el=$(id),W=Math.max(290,Math.round(el.clientWidth)),H=height,L=57,R=17,T=12,B=32,w=W-L-R,h=H-T-B,r=record(),end=Math.max(1/60,val(r.minutes.at(-1),'step')/60);let lo=0,hi=1;for(const a of lines)for(const [,v]of a.values){lo=Math.min(lo,v);hi=Math.max(hi,v)}for(const ref of refs){lo=Math.min(lo,ref.value);hi=Math.max(hi,ref.value)}hi+=(hi-lo)*.07;const x=t=>L+t/end*w,y=v=>T+(hi-v)/(hi-lo)*h;el.setAttribute('viewBox',`0 0 ${W} ${H}`);let svg='';if(shades)for(const [a,b]of shadeIntervals(r))svg+=`<rect x="${x(a)}" y="${T}" width="${x(b)-x(a)}" height="${h}" fill="#f7e9e3"/>`;
for(let i=0;i<=4;i++){const v=hi*i/4;svg+=`<line x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}" stroke="#e8edf0"/><text x="${L-7}" y="${y(v)+4}" text-anchor="end" font-size="11" fill="#516675">${fmt(v,0)}</text>`}
const count=end>=24?Math.ceil(end/24):4;for(let i=0;i<=count;i++){const t=end>=24?Math.min(end,i*24):end*i/count;svg+=`<line x1="${x(t)}" y1="${T}" x2="${x(t)}" y2="${H-B}" stroke="#edf1f4"/><text x="${x(t)}" y="${H-10}" text-anchor="middle" font-size="11" fill="#516675">${fmt(t,1)}h</text>`}
for(const ref of refs)svg+=`<line x1="${L}" y1="${y(ref.value)}" x2="${W-R}" y2="${y(ref.value)}" stroke="${ref.color||'#8395a0'}" stroke-dasharray="${ref.dash||'5 4'}"/>`;
for(const line of lines){const points=line.values.map(([t,v])=>`${x(t).toFixed(2)},${y(v).toFixed(2)}`).join(' ');svg+=`<polyline points="${points}" fill="none" stroke="${line.color}" stroke-width="1.7" ${line.dash?'stroke-dasharray="5 4"':''}/>`}
svg+=`<line data-cursor x1="${x(minute/60)}" x2="${x(minute/60)}" y1="${T}" y2="${H-B}" stroke="#183445" stroke-width="1.2"/>`;el.innerHTML=svg;plots.push({el,x})}
function drawPlots(){plots=[];const r=record();plot('power',['solar_available_kw','electrolyser_power_kw','dac_power_kw','methanator_power_kw'].map((f,i)=>({values:series(r,f),color:palette[i]})),{height:265});plot('routing',['solar_to_battery_kw','battery_to_electrolyser_kw','solar_curtailed_kw'].map((f,i)=>({values:series(r,f),color:['#466db1','#187f83','#ad781d'][i]})),{height:240});['battery_energy_kwh','h2_level_kg','co2_level_kg','ch4_level_kg'].forEach((f,i)=>{const refs=[{value:val(r.minutes[0],f)}];if(!i)refs.push({value:r.capacity.battery_capacity_kwh*r.summary.minute_audit.physical_battery_floor,color:'#a3453d',dash:'2 4'});plot('stock'+i,[{values:series(r,f),color:['#466db1','#187f83','#ad781d','#a3453d'][i]}],{height:205,refs})});let requested=0,delivered=0,a=[[0,0]],b=[[0,0]];for(const s of r.delivery.settlements){const h=s.due_step/60;a.push([h,requested]);b.push([h,delivered]);requested+=s.requested_kg;delivered+=s.delivered_kg;a.push([h,requested]);b.push([h,delivered])}const end=val(r.minutes.at(-1),'step')/60;a.push([end,requested]);b.push([end,delivered]);plot('delivery-plot',[{values:a,color:'#183445',dash:true},{values:b,color:'#187f83'}],{height:230,shades:false})}
function drawStatic(){const r=record(),p=r.summary.physical_metrics,e=r.summary.economic_metrics;table('deliveries',r.delivery.settlements.map(s=>[fmt(s.due_step/60),fmt(s.requested_kg),fmt(s.delivered_kg),fmt(s.unmet_kg)]));text('delivery-note',`${r.delivery.settlements.length} of 28 scheduled settlements recorded; ${fmt(p.delivered_kg)} kg delivered and ${fmt(p.unmet_kg)} kg unmet. Each delivery requests 150 kg every six hours (25 kg/h).`);
const components={delivered_service_value:'Delivered service value',unmet_delivery_penalty:'Unmet delivery penalty',solar_opportunity_cost:'Solar opportunity cost',battery_throughput_reserve:'Battery wear reserve',electrolyser_life_reserve:'Electrolyser wear reserve',methane_endowment_liability_change:'Methane-equivalent endowment change',battery_endowment_liability_change:'Battery endowment change'};table('economics',[...Object.entries(components).map(([k,label])=>[label,fmt(e.components_eur[k],2)]),[data.fixture?'Recorded-interval net value':'Complete-week net value',r.summary.execution_complete||data.fixture?fmt(e.net_operating_value_eur,2):'Unavailable: incomplete week']]);table('daily',r.cycles.map((c,i)=>[String(i+1),...['battery','h2','co2','ch4'].map(k=>fmt(c.deficits[k]))]));
$('outcomes').innerHTML=Object.entries(data.records).map(([k,v])=>{const s=v.summary,p=s.physical_metrics,e=s.economic_metrics;const status=!s.execution_complete?'Incomplete':s.delivery_and_physics_passed?'Success':'Failed';return `<tr class="${k===key?'selected':''}"><th scope="row">${esc(pretty(k))}</th><td class="outcome ${s.delivery_and_physics_passed?'pass':'fail'}">${status}</td><td>${fmt(p.delivered_kg)}</td><td>${fmt(p.unmet_kg)}</td><td>${fmt(p.electrolyser_cold_starts,0)} / ${fmt(p.electrolyser_hot_starts,0)}</td><td>${fmt(p.electrolyser_aborted_starts,0)} / ${fmt(p.electrolyser_forced_stops,0)}</td><td>${fmt(e.battery_equivalent_full_cycles,2)}</td><td>${s.execution_complete?fmt(e.net_operating_value_eur,0):'—'}</td></tr>`}).join('');
table('comparison',data.groups.map(g=>{const m=g.metrics,eur=m.net_operating_value_eur.all_case;return [pretty(g.controller),`${g.delivery_and_physics_successes} / ${g.n_scenarios}`,fmt(m.unmet_kg.all_case.mean),eur.mean===null?'Unavailable (incomplete cases)':`${fmt(eur.mean,0)} [${fmt(eur.ci95[0],0)}, ${fmt(eur.ci95[1],0)}]`]}));drawPlots()}
function drawCursor(){const r=record();minute=Math.max(0,Math.min(minute,r.minutes.length-1));const m=r.minutes[minute],s=r.summary,c=r.commands[Math.max(0,Math.ceil(minute/15)-1)],elapsed=minute/60,clockMinutes=Math.round(val(m,'time_s')/60),day=Math.floor(clockMinutes/1440)+1,hh=Math.floor(clockMinutes/60)%24,mm=clockMinutes%60;$('timeline').max=r.minutes.length-1;$('timeline').value=minute;text('clock',`Day ${day} · ${String(hh).padStart(2,'0')}:${String(mm).padStart(2,'0')}`);text('elapsed',`${fmt(elapsed,2)} h elapsed`);$('timeline').setAttribute('aria-valuetext',`${elapsed.toFixed(2)} elapsed hours, day ${day}, ${hh}:${String(mm).padStart(2,'0')}`);const settled=r.delivery.settlements.filter(d=>d.due_step<=minute),delivered=settled.reduce((n,d)=>n+d.delivered_kg,0),unmet=settled.reduce((n,d)=>n+d.unmet_kg,0);text('status',`${pretty(key)} · ${val(m,'dac_offline')?'DAC outage active':'DAC available'} · ${fmt(delivered)} kg delivered / ${fmt(unmet)} kg unmet at ${settled.length} settled deliveries.`);
table('readout',[['Solar available',`${fmt(val(m,'solar_available_kw'))} kW`],['Electrolyser / air capture / methanator',`${fmt(val(m,'electrolyser_power_kw'))} / ${fmt(val(m,'dac_power_kw'))} / ${fmt(val(m,'methanator_power_kw'))} kW`],['Battery state of charge',`${fmt(100*val(m,'battery_soc'))}%`],['Stored H₂ / CO₂ / methane',`${fmt(val(m,'h2_level_kg'))} / ${fmt(val(m,'co2_level_kg'))} / ${fmt(val(m,'ch4_level_kg'))} kg`]]);
if(c){const d=c.direct_control,a=d.requested_action,e=d.physical_applied_energy_kwh;table('commands',[['Battery charging',`${fmt(100*a.battery_share)}%`,fmt(e.battery/.25)],['Electrolyser (solar share; total applied)',`${fmt(100*a.electrolyser_share)}%`,fmt(e.electrolyser/.25)],['Air capture',`${fmt(100*a.dac_share)}%`,fmt(e.dac/.25)],['Methanator',`${fmt(100*a.methanator_share)}%`,fmt(e.methanator/.25)]]);text('commandtime',`${minute?'Command governing the preceding minute':'First command; no minute has yet executed at the cursor'}. Issued at ${fmt(c.index/4,2)} elapsed h; response summarizes its complete 15-minute recorded interval.`);text('support',`Battery support request: ${fmt(a.battery_support_kw)} kW; applied interval mean: ${fmt(e.battery_support/.25)} kW. Solar normalization divisor: ${fmt(d.solar_normalization_divisor,3)}.`);text('evidence',JSON.stringify({controller:key,scenario:data.scenario,minute,decision_index:c.index,checkpoint:s.checkpoint,raw_policy_mean:c.raw_mean,clipped_action:c.action,mapped_and_applied:c.direct_control,coverage_targets_if_baseline:c.baseline_targets,observation_sha256:c.observation_sha256,normalized_observation_sha256:c.normalized_observation_sha256,interval_reward_eur:c.reward_eur,minute_physical_audit:s.minute_audit,economic_audit:s.economic_audit,failure:s.failure,recording_sha256:r.recording.sha256},null,2))}for(const p of plots){const line=p.el.querySelector('[data-cursor]');line.setAttribute('x1',p.x(elapsed));line.setAttribute('x2',p.x(elapsed))}}
$('policy').onchange=e=>{key=e.target.value;minute=Math.min(minute,record().minutes.length-1);drawStatic();drawCursor()};$('timeline').oninput=e=>{minute=+e.target.value;drawCursor()};$('previous').onclick=()=>{minute-=15;drawCursor()};$('next').onclick=()=>{minute+=15;drawCursor()};$('delivery-next').onclick=()=>{const next=record().delivery.settlements.find(s=>s.due_step>minute);if(next){minute=next.due_step;drawCursor()}else text('status','No later settled delivery is present in this recording.')};$('outage-next').onclick=()=>{const r=record(),found=r.minutes.findIndex((m,i)=>i>minute&&val(m,'dac_offline')&&(i===0||!val(r.minutes[i-1],'dac_offline')));if(found>=0){minute=found;drawCursor()}else text('status','No later DAC outage begins in this recording.')};
$('download').onclick=()=>{const r=record(),csv=[data.fields.join(','),...r.minutes.map(m=>m.join(','))].join('\n'),url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=`${key}-scenario-${data.scenario}-minute-series.csv`;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000)};let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{drawPlots();drawCursor()},150)});drawStatic();drawCursor();
</script></body></html>'''


def write_inspector(data, output):
    output = Path(output)
    sidecar = output.with_suffix(".build.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("Inspector or build provenance already exists")
    serialized = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    content = HTML.replace("__DATA__", serialized)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(content)
    provenance = {"schema": data["schema"] + "-build", "fixture": data["fixture"], "scenario": data["scenario"],
                  "default_controller": data["default"], "html": pin(output), **data["provenance"],
                  "recordings": {k: r["recording"] for k, r in data["records"].items()}}
    with sidecar.open("x") as stream:
        json.dump(provenance, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--evaluation-freeze", type=Path)
    parser.add_argument("--recordings-root", type=Path, help="Optional relocated recording directory; bytes must match analysis pins")
    parser.add_argument("--fixture-recordings", type=Path, nargs="+", help="Existing training recordings only; output is visibly marked as a fixture")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = assemble(analysis_path=args.analysis, freeze_path=args.evaluation_freeze,
                    recordings_root=args.recordings_root, fixtures=args.fixture_recordings)
    write_inspector(data, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

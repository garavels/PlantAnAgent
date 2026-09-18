"""Build a self-contained reader for the prescribed, unselected first-seed traces.

This is an artifact generator, not part of the frozen experiment or inference.
The visual samples every 15 minutes; full minute records remain in the package.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from docs.research.tools.tail_stress_v1.analyze import CONDITIONS, CONTROLLERS, SEEDS
from docs.research.tools.wholeplant_ppo_v1.portable import file_pin

FIELDS = ("solar_available_kw", "battery_soc", "battery_energy_kwh", "h2_level_kg", "co2_level_kg", "ch4_level_kg", "dac_readiness_fraction", "methanator_readiness_fraction", "electrolyser_power_kw", "dac_power_kw", "methanator_power_kw", "battery_to_electrolyser_kw", "battery_to_dac_kw", "battery_to_methanator_kw", "ch4_production_kg_h", "h2_production_kg_h", "co2_capture_kg_h")


def collect(results):
    results = Path(results)
    analysis_path = results / "analysis/analysis.json"
    analysis = json.loads(analysis_path.read_text())
    if analysis["design"]["n_rollouts"] != 480 or not analysis["audits"]["all_rollouts_complete_and_audited"]:
        raise ValueError("Inspector requires all 480 audited results")
    records = {}
    for condition in CONDITIONS:
        for controller in CONTROLLERS:
            path = results / "cases" / f"{condition}--{controller}--{SEEDS[0]}" / "result.json"
            row = json.loads(path.read_text())
            if (row["condition"], row["controller"], row["seed"]) != (condition, controller, SEEDS[0]):
                raise ValueError("Trace identity differs")
            if file_pin(row["recording"]["path"]) != row["recording"]:
                raise ValueError("Retained recording changed")
            with gzip.open(row["recording"]["path"], "rt") as stream:
                recording = json.load(stream)
            if recording["summary"] != {k: v for k, v in row.items() if k != "recording"}:
                raise ValueError("Summary and full recording differ")
            states, commands = recording["minute_truth_states"], recording["decision_trace"]
            if len(states) != 10081 or len(commands) != 672:
                raise ValueError("Prescribed trace is incomplete")
            if any(len(c["action"]) != 7 or len(c["evidence"]["observation"]) != 149 for c in commands):
                raise ValueError("Action/input witness shape differs")
            records[f"{condition}|{controller}"] = {
                "condition": condition, "controller": controller, "seed": row["seed"],
                "states": [[state[field] for field in FIELDS] for state in states[::15]],
                "modes": [[state["dac_mode"], state["methanator_mode"]] for state in states[::15]],
                "actions": [c["action"] for c in commands],
                "delivery": row["delivery"]["settlements"], "event": row["event_metrics"],
                "economic": row["economic_metrics"], "operating": row["operating_metrics"],
                "all_delivery_success": row["every_delivery_passed"],
                "recording": row["recording"], "result": file_pin(path),
            }
    return {"schema": "tail-stress-offline-inspector-v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "analysis": file_pin(analysis_path), "fields": FIELDS, "sample_minutes": 15, "seed": SEEDS[0],
            "conditions": CONDITIONS, "controllers": CONTROLLERS, "records": records,
            "groups": analysis["groups"], "pooled_stress_groups": analysis["pooled_stress_groups"],
            "primary_claims": analysis["primary_claims"], "audits": analysis["audits"]}


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; img-src data:">
<title>PlantAnAgent · Tail stress evidence</title>
<style>
:root{--ink:#172923;--muted:#516760;--paper:#f5f5ef;--line:#c8d1c8;--green:#17684a;--amber:#ad5f16;--bad:#9a3041;--card:white}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:system-ui,-apple-system,sans-serif;font-size:15px;line-height:1.55}main{max-width:1260px;margin:auto;padding:35px 32px 70px}h1{font-size:clamp(30px,5vw,52px);font-weight:650;line-height:1.05;letter-spacing:-2px;margin:12px 0 18px}h2{font-size:23px;margin:0 0 10px;letter-spacing:-.5px}h3{font-size:16px;margin:0}.eyebrow{text-transform:uppercase;letter-spacing:2px;font-size:12px;color:var(--green);font-weight:750}.intro{max-width:900px;font-size:17px;color:var(--muted)}.ribbon{display:flex;flex-wrap:wrap;gap:9px;margin:20px 0}.pill{border:1px solid var(--line);padding:5px 11px;border-radius:30px;font-size:12px}.section{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:24px;margin-top:24px}.lede{color:var(--muted);margin:0 0 16px}.controls{display:flex;gap:16px;flex-wrap:wrap;align-items:end;margin:18px 0}label{display:block;font-size:12px;text-transform:uppercase;letter-spacing:1px;font-weight:650;color:var(--muted)}select,button{font:inherit;padding:9px 12px;border:1px solid #a7b9aa;border-radius:7px;background:#fff;color:var(--ink);min-height:42px}select{margin-top:5px;min-width:170px}button{cursor:pointer}button:hover{background:#edf5ee}.notice{background:#e9f1e8;border-left:4px solid var(--green);padding:14px 16px;margin:18px 0;font-size:14px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}.stat{border-top:2px solid var(--line);padding:10px 0}.stat span{font-size:12px;color:var(--muted)}.stat strong{display:block;font-size:24px;line-height:1.3;font-weight:600}.timeline{background:#f4f6f0;border-radius:8px;padding:16px;margin:20px 0}.timeline input{width:100%;accent-color:var(--green);cursor:pointer}.timehead{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}.timehead strong{font-size:18px}.plotgrid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.plot{border:1px solid #e0e6dc;border-radius:9px;padding:14px 14px 7px;min-width:0}.plot p{font-size:12px;color:var(--muted);margin:0 0 2px}.plot svg{width:100%;display:block;overflow:visible}.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-bottom:6px}.legend i{display:inline-block;width:17px;height:3px;margin-right:5px;vertical-align:middle}.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th,td{padding:9px 12px;text-align:right;border-bottom:1px solid #e5e9e1}th:first-child,td:first-child{text-align:left}th{font-weight:600;background:#f5f7f2;color:var(--muted)}tr.selected{background:#eaf3e9}.small{font-size:12px;color:var(--muted)}.mono{font-family:ui-monospace,monospace;font-size:12px;overflow-wrap:anywhere}summary{cursor:pointer;font-weight:600;padding:8px 0}details{margin:15px 0}pre{font-size:11px;background:#f5f7f2;white-space:pre-wrap;overflow-wrap:anywhere;padding:15px;max-height:300px;overflow:auto}.good{color:var(--green)}.bad{color:var(--bad)}.split{display:grid;grid-template-columns:1fr 1fr;gap:20px}footer{color:var(--muted);font-size:12px;margin-top:24px}a{color:var(--green)}@media(max-width:720px){main{padding:22px 15px 45px}.section{padding:17px}.plotgrid,.split{grid-template-columns:1fr}.stats{grid-template-columns:1fr 1fr}select{min-width:135px}.controls{gap:10px}h1{letter-spacing:-1px}.tablewrap:before{content:'Swipe horizontally to inspect every column';display:block;font-size:11px;color:var(--muted);padding:5px 0}th,td{padding:8px}.plot{padding:12px}}
</style></head><body><main>
<div class="eyebrow">PlantAnAgent · Commission research evidence</div><h1>Reliability under disruption</h1>
<p class="intro">Can a frozen learned controller turn battery access into reliable delivery when solar power or methanation disappears? These paired seven-day tests let each controller build—and spend—its own reserves before the event.</p>
<div class="ribbon"><span class="pill">480 audited rollouts</span><span class="pill">16 independent base scenarios</span><span class="pill">3 frozen PPO policies</span><span class="pill">7-day horizon</span><span class="pill">Entirely offline</span></div>
<section class="section"><h2>The complete challenge set</h2><p class="lede">All outcomes are retained. The severity mix is a constructed experiment, with no assigned real-world occurrence probabilities.</p>
<div class="controls"><div><label for="summary-condition">Summary condition</label><select id="summary-condition"></select></div></div>
<div class="tablewrap"><table><thead><tr><th>Controller</th><th>Every delivery met</th><th>Mean unmet kg</th><th>Worst-quarter mean unmet kg</th><th>Mean modeled €/week</th></tr></thead><tbody id="summary-rows"></tbody></table></div>
<p class="small">Worst-quarter loss is the mean of the four largest losses among 16 scenarios, or 20 among 80 pooled stress cells. It is descriptive, not a population tail-risk estimate. Pooled conditions remain correlated within each of the 16 base seeds.</p>
<details open><summary>Prespecified broad reliability criteria</summary><p class="small">For each PPO against each heuristic: at least eight rescued stress cells, zero regressions, no higher mean unmet mass, and a positive adjusted lower confidence bound for the all-delivery-success difference. Resampling keeps all five conditions together within each base seed; 2,000 draws, 99.1667% intervals adjusted for six primary comparisons.</p><div class="tablewrap"><table><thead><tr><th>Policy / reference</th><th>Rescued / regressed</th><th>Success difference [adjusted CI], pp</th><th>Criterion</th></tr></thead><tbody id="claim-rows"></tbody></table></div></details>
</section>
<section class="section"><h2>A full week, decision by decision</h2>
<div class="notice"><strong>Prescribed first-seed example: 43,000,000.</strong> This is the first seed for every condition/controller pair, chosen before results. These 30 traces illustrate behavior; use the complete 16-seed tables above for comparative conclusions.</div>
<div class="controls"><div><label for="condition">Disruption</label><select id="condition"></select></div><div><label for="controller">Controller</label><select id="controller"></select></div><button id="jump-start">Jump to event start</button><button id="jump-end">Jump to event end</button><button id="download">Download selected trace</button></div>
<p id="event-description" class="lede"></p>
<div class="stats" id="week-stats"></div>
<div class="timeline"><div class="timehead"><strong id="time-label"></strong><span class="small">Elapsed hours from D1 06:00 · samples every 15 min</span></div><input id="timeline" type="range" min="0" max="672" step="1" value="192" aria-label="Elapsed time, 15-minute intervals"></div>
<div class="stats" id="state-stats"></div>
<div class="plotgrid" id="plots"></div>
<p class="small">Shading marks the imposed disruption; the vertical line marks the selected time. Charts connect 15-minute samples of the physical state. Every physical minute was audited and the original complete traces remain in the research package. Power values at a boundary describe the last completed minute.</p>
<div class="split"><div><h3>At the event boundary</h3><div class="tablewrap"><table><thead><tr><th>Stock</th><th>Before event</th><th>After event</th></tr></thead><tbody id="stock-rows"></tbody></table></div></div><div><h3>Delivery losses through the week</h3><div class="tablewrap"><table><thead><tr><th>Period</th><th>Unmet kg</th></tr></thead><tbody id="event-rows"></tbody></table></div><p class="small">The boundary delivery belongs to the period ending at that boundary. The control panel uses hour 48 as a reference and has no imposed event.</p></div></div>
<details><summary>All 28 delivery settlements</summary><div class="tablewrap"><table><thead><tr><th>Elapsed hour</th><th>Demand kg</th><th>Delivered kg</th><th>Unmet kg</th></tr></thead><tbody id="delivery-rows"></tbody></table></div></details>
<details><summary>Selected state and raw policy command</summary><p class="small">Commands have seven normalized coordinates in [−1,1]. Physical power can differ because resource and equipment limits are applied by the simulator. At the final boundary there is no further action. The downloaded trace includes sampled states and all 672 commands; full 149-coordinate policy inputs remain in the retained research recordings.</p><pre id="detail"></pre></details>
</section><footer id="provenance"></footer></main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
'use strict';
const D=JSON.parse(document.getElementById('payload').textContent),$=id=>document.getElementById(id),F=Object.fromEntries(D.fields.map((f,i)=>[f,i]));
const labels={control:'Control · no imposed event',pv6:'Solar loss · 6 h',pv18:'Solar loss · 18 h',pv36:'Solar loss · 36 h',meth24:'Methanator outage · 24 h',meth48:'Methanator outage · 48 h','pooled-stress':'All five stress conditions'};
const policy={daylight:'Daylight',storage:'Storage heuristic','ppo-101':'PPO 101','ppo-211':'PPO 211','ppo-307':'PPO 307'};
const fmt=(v,d=1)=>Number.isFinite(v)?new Intl.NumberFormat('en-GB',{minimumFractionDigits:d,maximumFractionDigits:d}).format(v):'Unavailable';
const options=(el,values,display)=>{for(const v of values){let o=document.createElement('option');o.value=v;o.textContent=display[v];el.append(o)}};
options($('condition'),D.conditions,labels);options($('controller'),D.controllers,policy);options($('summary-condition'),[...D.conditions,'pooled-stress'],labels);
$('controller').value='ppo-101';$('summary-condition').value='pooled-stress';
function summary(){const v=$('summary-condition').value,groups=v==='pooled-stress'?D.pooled_stress_groups:D.groups.filter(g=>g.panel===v);$('summary-rows').innerHTML=groups.map(g=>`<tr><td>${policy[g.controller]}</td><td>${g.all_delivery_successes}/${g.n_cells}</td><td>${fmt(g.metrics.unmet_kg.mean)}</td><td>${fmt(g.worst_quarter.mean_unmet_kg)}</td><td>${fmt(g.metrics.net_operating_value_eur.mean,0)}</td></tr>`).join('')}
$('claim-rows').innerHTML=D.primary_claims.map(p=>{const x=p.primary_criterion.adjusted_success_difference;return `<tr><td>${policy[p.treatment]} / ${policy[p.reference]}</td><td>${p.rescues} / ${p.regressions}</td><td>${fmt(100*x.mean)} [${fmt(100*x.interval[0])}, ${fmt(100*x.interval[1])}]</td><td class="${p.primary_criterion.met?'good':'bad'}">${p.primary_criterion.met?'Met':'Not met'}</td></tr>`}).join('');
const chartSpecs=[['Solar available','MW',[['solar_available_kw','Solar','#bb7a19',.001]],null],['Battery','% of capacity',[['battery_soc','Charge','#17684a',100]],100],['Methane stock','kg',[['ch4_level_kg','Methane','#356396',1]],null],['Process readiness','% of full readiness',[['dac_readiness_fraction','Capture','#7c568f',100],['methanator_readiness_fraction','Methanation','#bc572e',100]],100],['Feedstock reserves','kg',[['h2_level_kg','Hydrogen','#268678',1],['co2_level_kg','Carbon dioxide','#7c568f',1]],null],['Battery power to processes','kW',[['battery_to_electrolyser_kw','Electrolysis','#268678',1],['battery_to_dac_kw','Capture','#7c568f',1],['battery_to_methanator_kw','Methanation','#bc572e',1]],null]];
function selected(){return D.records[$('condition').value+'|'+$('controller').value]}
function chart(spec,r,tick){const [title,unit,series,fixed]=spec,W=530,H=190,L=43,R=13,T=12,B=30,PW=W-L-R,PH=H-T-B,max=fixed||Math.max(1,...series.map(s=>Math.max(...r.states.map(row=>row[F[s[0]]]*s[3]))))*1.06;const x=i=>L+i/672*PW,y=v=>T+PH-v/max*PH;let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title} across seven days">`;
if(r.event.is_intervention)svg+=`<rect x="${x(r.event.onset_h*4)}" y="${T}" width="${x(r.event.end_h*4)-x(r.event.onset_h*4)}" height="${PH}" fill="#f8e5c9"/>`;
for(let i=0;i<=4;i++){const val=max*i/4,Y=y(val);svg+=`<line x1="${L}" y1="${Y}" x2="${W-R}" y2="${Y}" stroke="#dce4db"/><text x="${L-6}" y="${Y+3}" text-anchor="end" font-size="10" fill="#607568">${fmt(val,max>10?0:1)}</text>`}for(let hour=0;hour<=168;hour+=24){const X=x(hour*4);svg+=`<text x="${X}" y="${H-8}" text-anchor="middle" font-size="10" fill="#607568">${hour}h</text>`}
for(const [field,label,color,scale] of series){const path=r.states.map((row,i)=>(i?'L':'M')+x(i).toFixed(2)+','+y(row[F[field]]*scale).toFixed(2)).join(' ');svg+=`<path d="${path}" stroke="${color}" stroke-width="1.6" fill="none"/>`}
svg+=`<line x1="${x(tick)}" y1="${T}" x2="${x(tick)}" y2="${T+PH}" stroke="#253d2e" stroke-width="1.2" stroke-dasharray="4 3"/></svg>`;return `<div class="plot"><h3>${title}</h3><p>${unit}</p><div class="legend">${series.map(s=>`<span><i style="background:${s[2]}"></i>${s[1]}</span>`).join('')}</div>${svg}</div>`}
function stat(label,value){return `<div class="stat"><span>${label}</span><strong>${value}</strong></div>`}
function timeLabel(tick){const minutes=360+tick*15,day=Math.floor(minutes/1440)+1,h=Math.floor(minutes%1440/60),m=minutes%60;return `${fmt(tick/4,2)} h · D${day} ${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}`}
function render(){const r=selected(),tick=+$('timeline').value,s=r.states[tick],e=r.event,passed=r.delivery.filter(d=>d.unmet_kg<=1e-9).length;
$('event-description').textContent=e.is_intervention?`${labels[r.condition]} from elapsed hour ${e.onset_h} to ${e.end_h}. ${r.condition.startsWith('meth')?'Repair occurs at D4 20:00, after sunset.':'The original sampled solar resource returns after the imposed interruption.'} Controllers receive no future repair schedule.`:'No imposed event. Ordinary sampled weather is retained; unrelated random component faults are disabled in every condition.';
$('week-stats').innerHTML=stat('Deliveries met',passed+' / 28')+stat('Week unmet methane',fmt(r.economic.unmet_kg)+' kg')+stat('Modeled week contribution','€'+fmt(r.economic.net_operating_value_eur,0))+stat('Battery electricity used',fmt(r.operating.battery_discharge_kwh,0)+' kWh');
$('time-label').textContent=timeLabel(tick);$('state-stats').innerHTML=stat('Battery charge',fmt(s[F.battery_soc]*100)+'%')+stat('Methane in store',fmt(s[F.ch4_level_kg])+' kg')+stat('Capture readiness',fmt(s[F.dac_readiness_fraction]*100)+'%')+stat('Methanator readiness',fmt(s[F.methanator_readiness_fraction]*100)+'%');
$('plots').innerHTML=chartSpecs.map(spec=>chart(spec,r,tick)).join('');
const stocks=[['Battery / kWh','battery_energy_kwh'],['Hydrogen / kg','h2_level_kg'],['Carbon dioxide / kg','co2_level_kg'],['Methane / kg','ch4_level_kg']];$('stock-rows').innerHTML=stocks.map(([label,key])=>`<tr><td>${label}</td><td>${fmt(e.pre_event_state[key])}</td><td>${fmt(e.post_event_state[key])}</td></tr>`).join('');
$('event-rows').innerHTML=[['Before / at onset','pre_event_unmet_kg'],['During event','event_unmet_kg'],['After event','post_event_unmet_kg']].map(([label,key])=>`<tr><td>${label}</td><td>${fmt(e[key])}</td></tr>`).join('');
$('delivery-rows').innerHTML=r.delivery.map(d=>`<tr class="${Math.abs(d.due_step/15-tick)<1?'selected':''}"><td>${d.due_step/60}</td><td>${fmt(d.requested_kg)}</td><td>${fmt(d.delivered_kg)}</td><td class="${d.unmet_kg>1e-9?'bad':''}">${fmt(d.unmet_kg)}</td></tr>`).join('');
const detail={seed:r.seed,condition:r.condition,controller:r.controller,elapsed_h:tick/4,physical_state:Object.fromEntries(D.fields.map((f,i)=>[f,s[i]])),dac_mode:r.modes[tick][0],methanator_mode:r.modes[tick][1],next_decision_index:tick<672?tick:null,next_raw_action:tick<672?r.actions[tick]:null};$('detail').textContent=JSON.stringify(detail,null,2);window.__tailQA.selection={condition:r.condition,controller:r.controller,tick};}
$('condition').addEventListener('change',render);$('controller').addEventListener('change',render);$('timeline').addEventListener('input',render);$('summary-condition').addEventListener('change',summary);$('jump-start').addEventListener('click',()=>{$('timeline').value=selected().event.onset_h*4;render()});$('jump-end').addEventListener('click',()=>{$('timeline').value=selected().event.end_h*4;render()});
$('download').addEventListener('click',()=>{const r=selected(),blob=new Blob([JSON.stringify({schema:D.schema,fields:D.fields,sample_minutes:D.sample_minutes,trace:r},null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`tail-${r.condition}-${r.controller}-${r.seed}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});
$('provenance').innerHTML=`Analysis SHA-256 <span class="mono">${D.analysis.sha256}</span><br>30 prescribed first-seed traces · 673 sampled states and 672 actual seven-coordinate commands per trace · No remote assets, tracking or network requests.`;
window.__tailQA={ready:false,cases:480,traces:Object.keys(D.records).length,seed:D.seed,selection:null};summary();render();window.__tailQA.ready=true;
</script></body></html>'''


def build(results, output):
    payload = collect(results)
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(HTML.replace("__PAYLOAD__", encoded))
    return file_pin(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.results, args.output or args.results / "inspector.html")))


if __name__ == "__main__":
    main()

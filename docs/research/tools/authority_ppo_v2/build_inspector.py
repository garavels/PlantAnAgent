"""Build an offline replay from retained, checksum-verified evaluation traces."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read_record(path):
    raw=path.read_bytes(); record=json.loads(gzip.decompress(raw))
    minutes=record['minute_truth_states']
    if canonical_sha(minutes)!=record['minute_truth_sha256']:
        raise ValueError('Retained minute truth checksum differs')
    for i,frame in enumerate(record['decision_frames']):
        if frame['decision_index']!=i or frame['state']!=minutes[i*15]:
            raise ValueError('Boundary differs from minute truth')
    return record,hashlib.sha256(raw).hexdigest()


def compact(path):
    r,sha=read_record(path); minutes=r['minute_truth_states']; opening=minutes[0]
    fields=['solar_available_kw','electrolyser_power_kw','dac_power_kw','methanator_power_kw',
            'battery_energy_kwh','h2_level_kg','co2_level_kg','ch4_level_kg']
    frames=[]
    for i,f in enumerate(r['decision_frames']):
        state=f['state']; interval=minutes[max(0,(i-1)*15)+1:i*15+1]
        item={'index':i,'time':i/4,'clock_hour':state['time_of_day_h'],'values':[state[x] for x in fields],
              'dac_offline':state['health']['dac']['offline'],'mode':f['commitment']['mode'],
              'delivered':sum(x['delivered_kg'] for x in r['delivery']['settlements'] if x['due_step']<=i*15),
              'unmet':sum(x['unmet_kg'] for x in r['delivery']['settlements'] if x['due_step']<=i*15),
              'returns':[c['feasible'] for c in r['cycle_settlements'] if c['boundary_decision']<=i]}
        if i:
            command=r['commands'][i-1]; evidence=command.get('evidence') or {}
            item.update(targets=command['command']['targets']['setpoints_kw'],
                        prior=evidence.get('nominal_command',{}).get('targets',{}).get('setpoints_kw'),
                        adapter_action=evidence.get('raw_residual'),
                        raw_policy_mean=evidence.get('inference_witness',{}).get('unclipped_deterministic_action',evidence.get('deterministic_raw_action_mean')),
                        upstream_action_clipped=evidence.get('inference_witness',{}).get('upstream_action_clipped',evidence.get('deterministic_action_clipped')),
                        actual=[sum(m[x] for m in interval)/len(interval) for x in [*fields[1:4],'solar_to_battery_kw']],
                        projection=evidence.get('electrolyser_projection_reason'))
        frames.append(item)
    return {'label':r['controller'],'arm':r.get('arm'),'optimizer_seed':r.get('optimizer_seed'),
            'scenario':r['seed'],'frames':frames,'opening':[opening[x] for x in fields],
            'mission_passed':r['mission_passed'],'safe':r['hard_safety_passed'],
            'execution_complete':r['execution_complete'],
            'delivery_count':sum(s['unmet_kg']<=1e-9 for s in r['delivery']['settlements']),
            'return_count':sum(s['feasible'] for s in r['cycle_settlements']),
            'contribution':r['economic_metrics']['net_operating_value_eur'],
            'failure':r.get('failure'),'recording_sha256':sha,
            'checkpoint':None,
            'evaluation_freeze_sha256':r.get('evaluation_freeze_sha256'),
            'dac_outages':[{'start_h':f['start_s']/3600,'end_h':(f['start_s']+f['duration_s'])/3600}
                           for f in r['manifest']['disturbances']['schedule'] if f['type']=='dac_outage'],
            'contract_sha256':r['benchmark_contract_sha256']}


HTML=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Plant An Agent | Research replay</title><style>
:root{font-family:"Aptos","Segoe UI",-apple-system,BlinkMacSystemFont,sans-serif;color:#26382f;background:#edf1e9;font-size:16px;font-variant-numeric:tabular-nums}*{box-sizing:border-box}body{margin:0}main{max-width:1200px;margin:auto;padding:34px 28px 50px}header{display:flex;justify-content:space-between;gap:30px;align-items:start}h1{font-size:clamp(27px,3vw,39px);line-height:1.13;letter-spacing:-.035em;max-width:700px;margin:10px 0 15px;font-weight:620}h2{font-size:18px;font-weight:600;margin:0 0 12px}p{color:#5c6d62;line-height:1.55;max-width:78ch;margin:8px 0 20px}a{color:#315e47}.brand{font-size:15px;font-weight:600}.controls{padding:20px 0;display:flex;gap:10px;align-items:center;flex-wrap:wrap;border-top:1px solid #bfcdbf;border-bottom:1px solid #bfcdbf;margin:20px 0}.controls input{flex:1;min-width:180px;accent-color:#3c775a}.controls select{max-width:270px}button,select{font:inherit;color:inherit;background:white;border:1px solid #b6c6b8;border-radius:5px;padding:10px 12px}button{cursor:pointer}button:hover{background:#dce7d7}button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #466e86;outline-offset:3px}.clock{font-size:20px;font-weight:600;min-width:150px;text-align:right}.plot{background:white;padding:18px 14px 6px;border-radius:8px;margin-bottom:18px}.plot svg{display:block;width:100%;height:auto}.stocks{display:grid;grid-template-columns:1fr 1fr;gap:18px}.stocks .plot{margin:0}.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:13px;margin:8px 0 0;color:#5c6d62}.legend span:before{content:"";display:inline-block;width:22px;height:3px;background:var(--c);vertical-align:middle;margin-right:7px}.note{font-size:13px}.tables{display:grid;grid-template-columns:1fr 1.2fr;gap:28px;margin-top:28px}table{width:100%;border-collapse:collapse;font-size:14px}th{text-align:left;font-weight:550;color:#5c6d62}th,td{padding:10px 6px;border-bottom:1px solid #c8d3c7}td:not(:first-child),th:not(:first-child){text-align:right}.status{font-weight:650}.pass{color:#326b4a}.fail{color:#984337}.pill{background:#dce7d7;border-radius:4px;padding:5px 8px;font-size:13px;display:inline-block}.fault{background:#f0ded8;color:#843f37}details{margin-top:25px}summary{cursor:pointer}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5;background:white;padding:18px;border-radius:6px}footer{border-top:1px solid #bfcdbf;margin-top:25px;padding-top:16px;font-size:13px;color:#5c6d62}@media(max-width:720px){main{padding:22px 14px}header{display:block}.stocks,.tables{grid-template-columns:1fr}.clock{text-align:left}.controls input{flex-basis:100%}table{font-size:12px}}
</style><main><header><div><div class="brand">Plant An Agent / Research replay</div><h1>One week of learned plant control</h1><p>A recorded 25 kg/h methane mission, with four deliveries each day and daily restoration of battery, hydrogen, CO₂ and methane stocks.</p></div><div><span class="pill">Recorded simulation</span><p class="note">Scenario 9960000<br>Chosen before test outcomes</p></div></header>
<div class="controls"><label for="policy">Controller</label><select id="policy"></select><button id="play">Play</button><button id="next">Next daily return</button><button id="fault">Next DAC outage</button><input id="timeline" type="range" min="0" max="672" value="0" aria-label="Recording time"><output class="clock" id="clock"></output></div>
<p id="state" role="status"></p><section class="plot"><h2>Power allocation and sunlight / kW</h2><svg id="power" viewBox="0 0 1100 260" aria-label="Full week of power allocation"></svg><div class="legend"><span style="--c:#ae7d28">Available solar</span><span style="--c:#3c775a">Electrolyser</span><span style="--c:#466e86">Direct air capture</span><span style="--c:#855664">Methanator</span></div><p class="note">Plots use elapsed hours from the 06:00 mission start and show 15-minute boundaries; qualification uses every physical minute. The cursor marks playback time and shaded bands show exact recorded DAC outage intervals.</p></section>
<div class="stocks"><section class="plot"><h2>Battery margin / kWh</h2><svg id="stock0" viewBox="0 0 530 210"></svg></section><section class="plot"><h2>Hydrogen margin / kg</h2><svg id="stock1" viewBox="0 0 530 210"></svg></section><section class="plot"><h2>CO₂ margin / kg</h2><svg id="stock2" viewBox="0 0 530 210"></svg></section><section class="plot"><h2>Methane margin / kg</h2><svg id="stock3" viewBox="0 0 530 210"></svg></section></div>
<p class="note">Stock margin is relative to the opening inventory. Each stock must be at or above zero at every daily boundary. Solid: selected controller. Dashed: handwritten coverage reference.</p>
<div class="tables"><section><h2>Current power command</h2><p class="note" id="commandtime"></p><table><thead><tr><th>Equipment</th><th>Requested kW</th><th>Actual mean kW</th></tr></thead><tbody id="commands"></tbody></table><p class="note" id="commandnote"></p></section><section><h2>Complete-week outcomes</h2><p class="note">This single preselected case is an explanation of behavior. Use the full 32-scenario results for comparisons.</p><table><thead><tr><th>Controller</th><th>Mission</th><th>Deliveries</th><th>Returns</th></tr></thead><tbody id="outcomes"></tbody></table></section></div>
<details><summary>Recorded actions and evidence</summary><pre id="evidence"></pre></details><footer>Hybrid PPO adjusts a handwritten coverage controller through the same guarded simulator. No electricity purchase/export charge; fixed solar costs are unestimated and excluded. Complete mission qualification is separate from modeled operating contribution. The recording opens paused and runs entirely offline.</footer></main>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);let key=data.default,index=0,timer=null;
const pretty=r=>r.arm?`${r.arm==='narrow'?'Narrow':'Broad'} PPO ${r.optimizer_seed}`:'Coverage reference';
for(const [k,r] of Object.entries(data.records)){const o=document.createElement('option');o.value=k;o.textContent=pretty(r);$('policy').append(o)}$('policy').value=key;
const fmt=n=>Number(n).toLocaleString('en',{maximumFractionDigits:1}), colors=['#ae7d28','#3c775a','#466e86','#855664'];
function plot(id,series,colors,dashed=false){let el=$(id),W=Math.max(280,Math.round(el.clientWidth)),H=id==='power'?260:210,L=60,R=16,T=12,B=29,w=W-L-R,h=H-T-B;el.setAttribute('viewBox',`0 0 ${W} ${H}`);const all=series.flatMap(x=>x.map(v=>v[1]));let lo=Math.min(0,...all),hi=Math.max(1,...all),span=hi-lo;hi+=span*.06;lo-=span*.04;const x=t=>L+t/168*w,y=v=>T+(hi-v)/(hi-lo)*h;let s='';for(const outage of data.records[key].dac_outages){s+=`<rect x="${x(outage.start_h)}" y="${T}" width="${x(outage.end_h)-x(outage.start_h)}" height="${h}" fill="#f6ece6"/>`}
for(let d=0;d<=7;d++){s+=`<line x1="${x(d*24)}" y1="${T}" x2="${x(d*24)}" y2="${H-B}" stroke="#e1e8df"/><text x="${x(d*24)}" y="${H-8}" font-size="11" fill="#5c6d62" text-anchor="middle">${d*24}h</text>`}
for(let n=0;n<4;n++){let v=lo+(hi-lo)*n/3;s+=`<line x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}" stroke="#edf1e9"/><text x="${L-7}" y="${y(v)+4}" font-size="11" fill="#5c6d62" text-anchor="end">${fmt(v)}</text>`}
s+=`<line x1="${L}" y1="${y(0)}" x2="${W-R}" y2="${y(0)}" stroke="#81937e"/>`;
series.forEach((a,i)=>{s+=`<polyline fill="none" stroke="${colors[i]}" stroke-width="${i===1&&dashed?1.4:1.8}" ${i===1&&dashed?'stroke-dasharray="5 4"':''} points="${a.map(v=>`${x(v[0])},${y(v[1])}`).join(' ')}"/>`});s+=`<line x1="${x(index/4)}" y1="${T}" x2="${x(index/4)}" y2="${H-B}" stroke="#26382f" stroke-width="1.4"/>`;el.innerHTML=s}
function draw(){const r=data.records[key],f=r.frames[Math.min(index,r.frames.length-1)],base=data.records.coverage;index=f.index;$('timeline').max=r.frames.at(-1).index;$('timeline').value=index;$('clock').textContent=index===672?'Week complete':`Day ${Math.floor((r.frames[0].clock_hour+index/4)/24)+1} / ${String(Math.floor(f.clock_hour)).padStart(2,'0')}:${String(Math.round((f.clock_hour%1)*60)).padStart(2,'0')}`;
$('state').innerHTML=`<span class="pill ${f.dac_offline?'fault':''}">${f.dac_offline?'DAC outage active':'DAC available'}</span> &nbsp; Electrolyser: ${f.mode.replaceAll('_',' ')}. Delivered ${fmt(f.delivered)} kg; unmet ${fmt(f.unmet)} kg. Daily returns passed: ${f.returns.filter(Boolean).length}/${f.returns.length} settled.`;
plot('power',[0,1,2,3].map(n=>r.frames.map(v=>[v.time,v.values[n]])),colors);for(let n=0;n<4;n++)plot('stock'+n,[r.frames.map(v=>[v.time,v.values[n+4]-r.opening[n+4]]),base.frames.map(v=>[v.time,v.values[n+4]-base.opening[n+4]])],[['#677b45','#3c775a','#466e86','#855664'][n],'#9fa99e'],true);
$('commandtime').textContent=index?`Command issued at hour ${(index-1)/4}; actual mean over the following 15 minutes.`:'No command has executed at the opening boundary.';
$('commands').innerHTML=index?['Electrolyser','Direct air capture','Methanator','Battery charging'].map((label,n)=>`<tr><td>${label}</td><td>${fmt(f.targets[['electrolyser','dac','methanator','battery_charge_ceiling'][n]])}</td><td>${fmt(f.actual[n])}</td></tr>`).join(''):'';
$('commandnote').textContent='Battery request is a charging ceiling. '+(f.projection?`Electrolyser projection: ${f.projection.replaceAll('_',' ')}.`:'The simulator limits actual flow by available energy, reactants and equipment state.');
$('outcomes').innerHTML=Object.values(data.records).map(v=>`<tr><td>${pretty(v)}</td><td class="status ${v.mission_passed?'pass':'fail'}">${v.mission_passed?'Pass':'Fail'}</td><td>${v.delivery_count}/28</td><td>${v.return_count}/7</td></tr>`).join('');
$('evidence').textContent=JSON.stringify({controller:pretty(r),scenario:r.scenario,selected_before_test_outcomes:true,source_recording_sha256:r.recording_sha256,benchmark_contract_sha256:r.contract_sha256,hard_safety_passed:r.safe,mission_passed:r.mission_passed,execution_complete:r.execution_complete,failure:r.failure,decision_index:index,checkpoint:r.checkpoint,raw_policy_mean:f.raw_policy_mean,adapter_action:f.adapter_action,upstream_action_clipped:f.upstream_action_clipped,nominal_coverage_targets:f.prior,requested_targets:f.targets,projection:f.projection},null,2)}
function stop(){clearInterval(timer);timer=null;$('play').textContent='Play'}$('play').onclick=()=>{if(timer){stop();return}const end=data.records[key].frames.at(-1).index;if(index>=end)index=0;$('play').textContent='Pause';timer=setInterval(()=>{index=Math.min(end,index+4);draw();if(index>=end)stop()},150)};
$('timeline').oninput=e=>{stop();index=+e.target.value;draw()};$('policy').onchange=e=>{stop();key=e.target.value;draw()};$('next').onclick=()=>{stop();index=Math.min(data.records[key].frames.at(-1).index,(Math.floor(index/96)+1)*96);draw()};$('fault').onclick=()=>{stop();const a=data.records[key].frames;const found=a.find((f,i)=>i>index&&f.dac_offline&&(i===0||!a[i-1].dac_offline));if(found){index=found.index;draw()}else $('state').textContent='No later DAC outage occurs in this recording.'};window.addEventListener('resize',draw);draw();
</script></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluations',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--evaluation-freeze',type=Path,required=True)
    a=parser.parse_args(); records={}
    freeze=json.loads(a.evaluation_freeze.read_text())
    if canonical_sha({k:v for k,v in freeze.items() if k!='sha256'})!=freeze['sha256']:
        raise ValueError('Evaluation freeze checksum differs')
    endpoints={(v['arm'],v['optimizer_seed']):v for v in freeze['endpoints']['endpoints']}
    for path in sorted(a.evaluations.rglob('9960000.json.gz')):
        r=compact(path)
        if r['evaluation_freeze_sha256']!=freeze['sha256']:
            raise ValueError('Recording belongs to another evaluation freeze')
        if r['arm'] in ('narrow','broad'):
            key=f'{r["arm"]}-{r["optimizer_seed"]}'
            pin=endpoints[(r['arm'],r['optimizer_seed'])]['checkpoint']
            if hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest()!=pin['sha256']:
                raise ValueError('Checkpoint differs from the evaluated endpoint')
            r['checkpoint']={k:pin[k] for k in ('sha256','bytes')}
        elif 'coverage' in r['label']:
            key='coverage'
        else:
            continue
        if key in records:
            raise ValueError(f'Duplicate recording for {key}')
        records[key]=r
    if set(records)!={'coverage',*(f'{arm}-{s}' for arm in ('narrow','broad') for s in (101,211,307))}:
        raise ValueError('Inspector requires coverage and all six final policies')
    records={key:records[key] for key in ['coverage',*(f'{arm}-{s}' for arm in ('narrow','broad') for s in (101,211,307))]}
    data={'schema':'authority-ppo-inspector-v2','default':'narrow-101','records':records}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    serialized=json.dumps(data,separators=(',',':'),allow_nan=False).replace('</',r'<\/')
    a.output.write_text(HTML.replace('__DATA__',serialized))
    a.output.with_suffix('.json').write_text(json.dumps({'schema':data['schema'],'default':data['default'],
        'recordings':{key:{k:r[k] for k in ('recording_sha256','contract_sha256','scenario','mission_passed')} for key,r in records.items()}},indent=2)+'\n')
    print(a.output)


if __name__=='__main__':
    main()

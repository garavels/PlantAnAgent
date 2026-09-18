"""Build the offline whole-plant research inspector from retained evidence only.

Design: continue the laboratory notebook used by the earlier study, with solar
amber #ad7815, electrical teal #087c83, DAC blue #4c6caf, methane red #ae5547,
ink #183345 and cool paper #f3f7fa. Avenir/system text, left-aligned explanations,
full-width comparisons, and one shared minute cursor keep the physical process
legible. The distinctive addition is paid readiness: preparation, holding and
production are separated along the actual energy path, not rendered as generic
scorecards. The existing mobile-tested controls are reused without changing the
older artifact. This builder is artifact-only and imports no simulator.
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
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from docs.research.tools.comparison_ppo_v2 import build_inspector as previous

SCHEMA = 'wholeplant-ppo-v1-evaluation'
ARMS = ('restricted-instant', 'shared-instant', 'restricted-dynamic', 'shared-dynamic')
SEEDS = (101, 211, 307)
TRAIN_START, TRAIN_STOP = 30000000, 30100000
DEVELOPMENT = tuple(range(30100000, 30100008))
HELD_OUT = tuple(range(30200000, 30200032))
PROCESSES = ('electrolyser', 'dac', 'methanator')
FIELDS = (*previous.FIELDS, 'battery_to_dac_kw', 'battery_to_methanator_kw',
    'solar_to_electrolyser_kw', 'solar_to_dac_kw', 'solar_to_methanator_kw',
    *(u+'_'+k for u in ('dac', 'methanator') for k in
      ('readiness_kwh', 'readiness_fraction', 'startup_power_kw', 'standby_power_kw',
       'productive_power_kw', 'heat_loss_kw', 'starts', 'cold_starts')))
STOCKS = (*previous.STOCKS, 'dac_readiness_kwh', 'methanator_readiness_kwh')
GROUP_METRICS = (*previous.GROUP_METRICS,
    *(f'operating:{u}_{k}' for u in PROCESSES for k in ('battery_kwh', 'kwh')),
    *(f'operating:{u}_{k}' for u in ('dac', 'methanator') for k in
      ('startup_kwh', 'standby_kwh', 'productive_kwh', 'heat_loss_kwh', 'starts', 'cold_starts')),
    *(f'{phase}:{u}_readiness_kwh' for phase in ('initial', 'final') for u in ('dac', 'methanator')))
canonical_sha, pin, resolve_pin = previous.canonical_sha, previous.pin, previous.resolve_pin


def key_for(row):
    return row['condition']+'|'+row['controller']


def compact_summary(row):
    item = previous.compact_summary(row)
    item.update(phase=row.get('phase'), arm=row['condition'])
    for phase in ('initial', 'final'):
        item[phase+'_state'] = {k: row.get(phase+'_state', {}).get(k) for k in STOCKS}
    return item


def engineering_record(record):
    """Normalize our admitted train-only direct-audit fixture without simulation."""
    row = record['summary'].copy()
    if row.get('split') != 'train' or not TRAIN_START <= row['seed'] < TRAIN_STOP:
        raise ValueError('Engineering fixture must use the new training namespace')
    arm = row['arm']
    row.update(schema=SCHEMA+'-case', condition=arm, controller_kind='baseline',
        variant={'arm':arm, 'delivery_rate_kg_h':30., 'opening_battery':'sampled',
            'shared_bus':arm.startswith('shared'), 'downstream_dynamics':arm.endswith('dynamic')},
        physical_steps=len(record['minute_truth_states'])-1,
        physical_invariants_passed=row['minute_audit']['all_physical_invariants_passed'],
        paired_source_sha256=row['paired_source_manifest']['paired_source_sha256'],
        retained_minute_truth_sha256=canonical_sha(record['minute_truth_states']))
    digest = hashlib.sha256()
    commands = []
    for old in record['decision_trace']:
        b = old['evidence']
        evidence = {'observation':b['actual_observation'], 'observation_sha256':canonical_sha(b['actual_observation']),
            'action':b['action'], 'baseline_inference':b,
            'controller_input_sha256':canonical_sha(b['controller_input'])}
        digest.update(json.dumps(evidence,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
        commands.append({'decision_index':old['decision'], 'action':b['action'], 'evidence':evidence,
            'reward_eur':old['reward_eur'], 'direct_control':old['direct_control'],
            'battery_intervention':old['battery_intervention'], 'state':{}})
    row['inference'] = {'witness_sha256':digest.hexdigest(), 'verified_decisions':len(commands), 'actual_observation_inputs':True}
    return {**record, 'schema':SCHEMA+'-recording', 'summary':row, 'decision_trace':commands}


def compact_record(path, expected_row=None, fixture=False):
    with gzip.open(path, 'rt') as stream:
        record = json.load(stream)
    if fixture and not record.get('schema'):
        record = engineering_record(record)
    row = record['summary']
    if record.get('schema') != SCHEMA+'-recording' or row.get('schema') != SCHEMA+'-case':
        raise ValueError('Unexpected whole-plant recording schema')
    if expected_row is not None and row != {k:v for k,v in expected_row.items() if k!='recording'}:
        raise ValueError('Analysis row and retained recording disagree')
    if fixture and (row['split'] != 'train' or not TRAIN_START <= row['seed'] < TRAIN_STOP):
        raise ValueError('Fixture mode admits new training cases only')
    minutes, decisions = record.get('minute_truth_states'), record.get('decision_trace', [])
    if not minutes or not decisions:
        if row['execution_complete']:
            raise ValueError('Complete first case lacks retained trace')
        return {'summary':compact_summary(row), 'unavailable':'This attempt ended before a complete decision trace was recorded.', 'recording':pin(path)}
    if canonical_sha(minutes) != row['retained_minute_truth_sha256']:
        raise ValueError('Retained minute truth checksum differs')
    if len(minutes) != row['physical_steps']+1 or len(decisions) != row['decisions'] or len(minutes) != 15*len(decisions)+1:
        raise ValueError('Minute/action lengths differ from the15-minute hold')
    for i, minute in enumerate(minutes):
        if minute['step'] != i or minute['time_s'] != minutes[0]['time_s']+i*60:
            raise ValueError('Minute trace is not contiguous')
    digest, commands = hashlib.sha256(), []
    for i, decision in enumerate(decisions):
        if decision['decision_index'] != i or any(minutes[(i+1)*15][k] != v for k,v in decision['state'].items()):
            raise ValueError('Decision index or closing state differs')
        evidence = decision['evidence']; obs = evidence.get('observation')
        if not obs or len(obs)!=149 or canonical_sha(obs)!=evidence['observation_sha256']:
            raise ValueError('Actual149-value public input missing or changed')
        if evidence['action'] != decision['action'] or len(decision['action'])!=7:
            raise ValueError('Actual seven-action inference witness differs')
        transformed, replacements = None, None
        if row['controller_kind']=='baseline':
            b=evidence['baseline_inference']; transformed=b['controller_input']; replacements=b['replacements']
            target=list(obs)
            if row['controller']=='storage-persistence':
                if row['baseline_metadata']['consumed_forecast_indices'] != [34,35,36] or row['baseline_metadata']['current_solar_index'] != 2:
                    raise ValueError('Persistence metadata changes prescribed inputs')
                for index in (34,35,36): target[index]=obs[2]
            if b['actual_observation']!=obs or b['action']!=decision['action'] or transformed!=target or canonical_sha(transformed)!=evidence['controller_input_sha256']:
                raise ValueError('Baseline input transformation or action differs')
        else:
            normalized=evidence.get('normalized_observation')
            if not normalized or len(normalized)!=149 or canonical_sha(normalized)!=evidence['normalized_observation_sha256']:
                raise ValueError('PPO normalized input witness differs')
            mean=evidence.get('raw_mean')
            if not mean or len(mean)!=7 or [max(-1,min(1,v)) for v in mean]!=decision['action']:
                raise ValueError('PPO bounded action differs from its raw mean')
        digest.update(json.dumps(evidence,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
        commands.append({'index':i, 'action':decision['action'], 'raw_mean':evidence.get('raw_mean'),
            'observation':obs, 'observation_sha256':evidence['observation_sha256'],
            'normalized_observation_sha256':evidence.get('normalized_observation_sha256'),
            'transformed_observation':transformed, 'replacements':replacements,
            'direct_control':decision['direct_control'], 'battery_intervention':decision['battery_intervention'],
            'reward_eur':decision['reward_eur']})
    if digest.hexdigest()!=row['inference']['witness_sha256']:
        raise ValueError('Inference stream hash differs')
    if minutes[0]!=row['initial_state'] or minutes[-1]!=row['final_state']:
        raise ValueError('Opening or closing state differs')
    return {'summary':compact_summary(row), 'commands':commands,
        'minutes':[[m[k] for k in FIELDS]+[int(m['health']['dac']['offline'])] for m in minutes],
        'delivery':record['delivery'], 'recording':pin(path),
        'capacity':{k:minutes[0][k] for k in ('battery_capacity_kwh','h2_capacity_kg','co2_capacity_kg','ch4_capacity_kg')}}


def prescribed_panels():
    panels={}
    for split,seeds in (('development',DEVELOPMENT),('held_out',HELD_OUT)):
        for arm in ARMS:
            for controller in ('daylight','storage'):
                panels[split,arm,controller]=seeds
        for seed in SEEDS:
            panels[split,'shared-dynamic',f'ppo-{seed}']=seeds
            if split=='development': panels[split,'shared-dynamic',f'ppo-initial-{seed}']=seeds
    panels['held_out','shared-dynamic','storage-persistence']=HELD_OUT
    return panels


def assemble(*, analysis_path=None, recordings_root=None, fixtures=None):
    fixture=bool(fixtures)
    provenance={'builder':pin(__file__), 'base_renderer':pin(previous.__file__),
        'created_at':datetime.now(timezone.utc).isoformat()}
    records, rows, groups, comparisons, arm_comparisons, factorial = {}, [], [], [], [], []
    supplement=None
    if fixture:
        if analysis_path: raise ValueError('Do not mix canonical analysis and engineering fixtures')
        for path in fixtures:
            record=compact_record(path,fixture=True);row=record['summary'];key=key_for(row)
            if key in records: raise ValueError('Duplicate fixture panel')
            records[key]=record;rows.append(row)
        seeds={r['seed'] for r in rows}
        if len(seeds)!=1: raise ValueError('Fixture traces must share a training scenario')
        scenario=seeds.pop()
    else:
        if not analysis_path: raise ValueError('Final mode requires the complete canonical analysis')
        analysis=json.loads(Path(analysis_path).read_text())
        if analysis.get('schema')!=SCHEMA+'-analysis': raise ValueError('Unexpected analysis schema')
        original=analysis['rows'];expected=prescribed_panels();by=defaultdict(list);seen=set()
        for row in original:
            key=row['split'],row['condition'],row['controller'];case=(*key,row['seed'])
            if case in seen or key not in expected: raise ValueError('Duplicate or nonprescribed canonical outcome')
            seen.add(case);by[key].append(row)
        if len(original)!=496 or set(by)!=set(expected): raise ValueError('Final inspector requires all496 prescribed outcomes')
        for key,seeds in expected.items():
            if sorted(r['seed'] for r in by[key])!=list(seeds): raise ValueError('A prescribed panel lacks retained attempts')
        freezes={};frozen_panels={}
        for item in analysis['freezes']:
            freeze=json.loads(resolve_pin(item['file']).read_text())
            if freeze.get('schema')!=SCHEMA+'-freeze' or freeze['sha256']!=item['sha256'] or canonical_sha({k:v for k,v in freeze.items() if k!='sha256'})!=freeze['sha256']:
                raise ValueError('Canonical freeze identity differs')
            for source in freeze['sources']: resolve_pin(source)
            for controller in freeze['controllers'].values():
                if controller['kind']=='ppo':
                    for checkpoint_pin in controller['checkpoint'].values(): resolve_pin(checkpoint_pin)
            freezes[freeze['sha256']]=freeze
            for panel in freeze['panels']:
                key=panel['split'],panel['condition'],panel['controller']
                if key not in expected or panel['seeds']!=list(expected[key]): raise ValueError('Freeze panel differs from protocol')
                if key in frozen_panels: raise ValueError('Canonical panel declared in multiple freezes')
                frozen_panels[key]=freeze['sha256']
        if set(frozen_panels)!=set(expected): raise ValueError('A canonical panel lacks a source freeze')
        scenario=HELD_OUT[0]
        for row in original:
            key=row['split'],row['condition'],row['controller']
            if row['evaluation_freeze_sha256']!=frozen_panels[key]: raise ValueError('Outcome freeze differs from its panel')
            path=resolve_pin(row['recording'],recordings_root) if row.get('recording') else None
            if row['execution_complete'] and not path: raise ValueError('Complete attempt lacks recording')
            if row['split']=='held_out' and row['seed']==scenario:
                records[key_for(row)]=compact_record(path,expected_row=row) if path else {'summary':compact_summary(row),'unavailable':'This retained attempt has no recording.'}
            rows.append(compact_summary(row))
        if len(records)!=12: raise ValueError('Expected exactly12 prescribed first-scenario traces')
        for group in analysis['groups']:
            found=by[group['split'],group['condition'],group['controller']]
            if len(found)!=group['n_scenarios'] or group['delivery_successes']!=sum(r['every_delivery_passed'] for r in found) or group['complete_weeks']!=sum(r['execution_complete'] for r in found):
                raise ValueError('Group summary differs from retained outcomes')
            groups.append({k:v for k,v in group.items() if k!='metrics'}|{'metrics':{k:group['metrics'][k] for k in GROUP_METRICS if k in group['metrics']}})
        if len(groups)!=26 or len({(g['split'],g['condition'],g['controller']) for g in groups})!=26:
            raise ValueError('Expected26 unique overview groups')
        def compact_comparison(item):
            return {k:v for k,v in item.items() if k!='metrics'}|{'contribution':item['metrics']['net_operating_value_eur']}
        comparisons=[compact_comparison(x) for x in analysis['aggregate_comparisons']]
        arm_comparisons=[compact_comparison(x) for x in analysis['condition_comparisons']]
        factorial=[compact_comparison(x) for x in analysis['factorial_contrasts']]
        forecast=analysis.get('forecast_comparison')
        if forecast is None: raise ValueError('Canonical storage forecast comparison missing')
        criterion=dict(forecast['criterion'])
        criterion['forecast_minus_persistence_eur']=forecast['metrics']['net_operating_value_eur']['all_case']
        supplement={'criterion':criterion, 'n_scenarios':32,
            'reference_delivery_successes':sum(r['every_delivery_passed'] for r in by['held_out','shared-dynamic','storage']),
            'persistence_delivery_successes':sum(r['every_delivery_passed'] for r in by['held_out','shared-dynamic','storage-persistence'])}
        provenance.update(analysis=pin(analysis_path),freezes=analysis['freezes'],bootstrap=analysis['bootstrap'],
            learning_comparisons=analysis['learning_comparisons'],interpretation=analysis['interpretation'])
    pairings={r['summary'].get('paired_source_sha256') for r in records.values() if r['summary'].get('paired_source_sha256')}
    if len(pairings)>1: raise ValueError('Selected traces do not share identical exogenous scenarios')
    default='shared-dynamic|ppo-101'
    if default not in records: default='shared-dynamic|storage' if 'shared-dynamic|storage' in records else next(iter(records))
    return {'schema':'wholeplant-ppo-v1-offline-inspector','fixture':fixture,'scenario':scenario,
        'fields':[*FIELDS,'dac_offline'],'default':default,'records':records,'rows':rows,'groups':groups,
        'comparisons':comparisons,'arm_comparisons':arm_comparisons,'factorial':factorial,
        'supplement':supplement,'provenance':provenance}


def replace_once(text, old, new):
    if text.count(old)!=1: raise ValueError('Base renderer changed around '+old[:90])
    return text.replace(old,new,1)


def replace_block(text, start, stop, new):
    first=text.index(start);last=text.index(stop,first)
    return text[:first]+new+text[last:]


def render_template():
    html=previous.HTML
    # Reuse stable offline controls, typography and responsive plotting; every
    # experiment-specific claim and input dimension is replaced below.
    html=html.replace('Daylight production and learned control — PlantAnAgent','Whole-plant storage and readiness — PlantAnAgent')
    html=replace_once(html,'</style>', '.mobile-table-hint{display:none}@media(max-width:700px){.mobile-table-hint{display:block}}</style>')
    html=replace_once(html,'<p class="note" id="command-note">','<p class="note mobile-table-hint">Swipe the table sideways to see the actual-power column.</p><p class="note" id="command-note">')
    html=replace_once(html,'Does planning improve on <br>daylight production?','What changes when the battery <br>can power the whole plant?')
    html=replace_once(html,'Compare learned policies with explicit operating rules, then inspect how the same simulated plant uses sunlight, stored electricity and gas to meet deliveries.','A solar-powered plant converts water and captured CO₂ into methane. Compare where stored electricity can go, whether equipment must prepare before producing, and what the controllers actually learn.')
    html=replace_block(html,'<section class="method">','<section class="section" id="overview-section">',METHOD+ARM_SECTION)
    html=html.replace('All three new PPO30 policies began from random weights and were trained at 30 kg/h.','All three new PPO policies began from random weights and were trained at30 kg/h on the shared-battery, finite-readiness arm.')
    html=replace_once(html,'A supplementary test keeps the daylight controller unchanged and replaces its three consumed forecast values with current solar availability. It uses the same 32 test scenarios at 30 kg/h with sampled opening charge. This measures the value of those inputs within this rule, not an optimum or proof that PPO needs forecasts.','The same storage controller is evaluated with its issued forecasts and with its three consumed forecast inputs replaced by current sunlight. All other rules and inputs remain identical. The32 shared-dynamic test scenarios measure information value within this fixed rule, not an optimum or proof that PPO needs forecasts.')
    html=html.replace('Electrolyser power includes battery support.','Every process power trace includes its accepted battery support.')
    html=html.replace('Night / no available solar power','No available solar power')
    html=replace_once(html,'<span style="--c:#4c6caf">Charging</span><span style="--c:#087c83">Supporting electrolysis</span><span style="--c:#ad7815">Curtailed solar</span>','<span style="--c:#79549b">Charging</span><span style="--c:#087c83">Battery to electrolysis</span><span style="--c:#4c6caf">Battery to capture</span><span style="--c:#ae5547">Battery to methanation</span><span style="--c:#ad7815">Curtailed solar</span>')
    html=replace_once(html,'<div class="stocks"><section class="plot"><h3>Hydrogen stored / kg</h3>',READINESS+'<div class="stocks"><section class="plot"><h3>Hydrogen stored / kg</h3>')
    html=html.replace('Deliveries can draw stored methane while production is stopped. Nighttime methane production is zero in this electrical architecture.','Deliveries can draw stored methane while production is stopped. Shared battery access permits production without sunlight when readiness, feedstocks and charge allow it; the trace shows whether that option is used.')
    html=replace_once(html,'Full PPO chooses all five controls directly. The modeled battery floor, capacities, losses and electrolyser commitment dynamics remain physical assumptions. Neither daily stock restoration nor an extra operator reserve is imposed. A baseline\'s own operating rules constrain that baseline only.','Full PPO chooses all seven controls directly. Capacities, electrical losses, the battery floor, electrolysis commitment dynamics and the declared downstream readiness model remain physical assumptions. Neither daily stock restoration nor an extra operator reserve is imposed. Each baseline\'s rules constrain that baseline only.')
    html=replace_once(html,'Good performance here establishes useful control in this seven-day simulation. It does not establish an optimal schedule, realistic downstream thermal cycling or indefinite operation. Plausible assumptions and evidence that control choices matter are the benchmark’s purpose. Removing battery access from an existing policy tests its dependence; it is not a retrained batteryless optimum.','A completed learning run and its initial-to-final comparison can establish learning in this seven-day simulation. They do not establish an optimal schedule or indefinite operation. The paid readiness reservoirs are transparent aggregate assumptions; they do not establish realistic thermochemical trajectories. Four-arm contrasts condition on the same fixed baseline algorithms, while PPO is trained only on shared-dynamic. A negative battery, forecasting or PPO result is retained as evidence about this benchmark.')
    html=replace_block(html,'const name=id=>','const splitName=',LABELS)
    html=replace_block(html,'const order=id=>','const sortControllers=',"const order=id=>id==='daylight'?0:id==='storage'?1:id==='storage-persistence'?3:2;\n")
    html=replace_block(html,'const description=r=>','const actualTraces=',DESCRIPTION)
    html=replace_once(html,"${fmt(data.rows.filter(r=>!r.supplementary).length)} main case outcomes${data.supplement?' + 32 supplementary':''}","${fmt(data.rows.length)} retained case outcomes${data.fixture?'':' ·32 independent test scenarios'}")
    html=replace_block(html,'function learningAndSupplement(){','function overview(){',LEARNING)
    html=replace_block(html,'function overview(){','function casesTable(){',OVERVIEW)
    html=replace_once(html,"chart('routing',[{field:'solar_to_battery_kw',color:'#4c6caf'},{field:'battery_to_electrolyser_kw',color:'#087c83'},{field:'solar_curtailed_kw',color:'#ad7815'}]);", "chart('routing',[{field:'solar_to_battery_kw',color:'#79549b'},{field:'battery_to_electrolyser_kw',color:'#087c83'},{field:'battery_to_dac_kw',color:'#4c6caf'},{field:'battery_to_methanator_kw',color:'#ae5547'},{field:'solar_curtailed_kw',color:'#ad7815'}]);")
    html=replace_once(html,"chart('production',[{field:'ch4_production_kg_h',color:'#ae5547'}],{max:160});}","chart('production',[{field:'ch4_production_kg_h',color:'#ae5547'}],{max:160});"+READINESS_PLOTS+'}')
    html=replace_block(html,"$('readout').innerHTML=","const requested=command.direct_control",READOUT)
    html=replace_block(html,"$('commands').innerHTML=","set('detail',",COMMANDS)
    html=replace_once(html,"['Methane / kg','ch4_level_kg']].map", "['Methane / kg','ch4_level_kg'],['Capture readiness / kWh','dac_readiness_kwh'],['Methanator readiness / kWh','methanator_readiness_kwh']].map")
    html=replace_once(html,"set('night-account',`Across zero-solar minutes: electrolysis consumed ${fmt(op.zero_solar_electrolyser_kwh,1)} kWh; battery support supplied ${fmt(op.zero_solar_battery_discharge_kwh,1)} kWh; methane production was ${fmt(op.zero_solar_ch4_produced_kg,2)} kg. These are observed flows, not a claim that nighttime production is required.`);",NIGHT_ACCOUNT)
    html=html.replace('comparison-all-case-outcomes.csv','wholeplant-all-case-outcomes.csv')
    html=html.replace('Trained at30','Trained at 30').replace('trained at30','trained at 30').replace('The32','The 32').replace('·32','· 32')
    return html


METHOD=r'''<section class="method"><div><h3>Two physical changes, four matched plants</h3><p class="note">In the restricted arms, the battery powers only electrolysis. In the shared arms, it can also power capture and methanation. Independently, downstream equipment either responds instantly or must pay for preparation, holding readiness and cooling. Every arm faces the same30 kg/h demand, seven-day horizon, weather, faults and opening inventories.</p></div><div><h3>What readiness means here</h3><p class="note">Readiness is an assumed store of paid preparation energy. Capture stores up to120 kWh and loses60 kW when ready; methanation stores72 kWh and loses18 kW. Preparation makes no product. These are transparent aggregate design assumptions, not measured temperatures or a calibrated replica of a commercial plant.</p></div></section>
<p class="note">Every controller chooses seven direct commands every15 minutes from149 public measurements and forecast values. PPO has no heuristic operating shield or daily inventory-restoration requirement. Original electrolysis constraints and the declared equipment physics still apply. Opening charge is sampled at25–65%; this experiment does not change the delivery rate or episode length.</p>
'''.replace('same30','same 30').replace('to120','to 120').replace('loses60','loses 60').replace('stores72','stores 72').replace('loses18','loses 18').replace('every15','every 15').replace('from149','from 149').replace('at25','at 25')

ARM_SECTION=r'''<section class="section" id="arm-section"><h2>What changes across the four plants?</h2><p class="note">The same fixed rule is used in all four arms, with the declared routing and readiness model available to it. These contrasts describe that rule under different physics; they do not show independently optimized architectural value. Zero-solar periods have no available PV and can include daytime outages.</p><div class="table-wrap"><table><thead><tr><th>Physical arm</th><th>Controller</th><th>All deliveries</th><th>Contribution / €</th><th>Zero-solar methane / kg</th><th>Battery to capture / kWh</th><th>Battery to methane / kWh</th></tr></thead><tbody id="arms"></tbody></table></div><p class="note" id="arm-effects"></p></section>
'''

READINESS=r'''<div class="two"><section class="plot"><h3>Readiness retained / %</h3><svg id="readiness" role="img" aria-label="Capture and methanator readiness fractions"></svg><div class="legend"><span style="--c:#4c6caf">Capture</span><span style="--c:#ae5547">Methanator</span></div><p class="note">A dynamic unit must reach100% before a later minute can produce. Below that level, accepted heater power rebuilds readiness after losses. Instantaneous arms do not use these reservoirs; zero is not a failure.</p></section><section><h3>Where downstream electricity went</h3><div class="table-wrap"><table><thead><tr><th>Selected week</th><th>Capture</th><th>Methanator</th></tr></thead><tbody id="thermal-account"></tbody></table></div><p class="note">Preparation and ready standby consume real electricity. Productive power accounts for actual conversion. Losses remove readiness energy; they are not an additional electrical purchase.</p></section></div><div class="two"><section class="plot"><h3>Capture: paid power services / kW</h3><svg id="dac-services" role="img" aria-label="Capture startup standby and productive power"></svg><div class="legend"><span style="--c:#ad7815">Preparation</span><span style="--c:#79549b">Ready standby</span><span style="--c:#4c6caf">Productive</span></div></section><section class="plot"><h3>Methanation: paid power services / kW</h3><svg id="meth-services" role="img" aria-label="Methanator startup standby and productive power"></svg><div class="legend"><span style="--c:#ad7815">Preparation</span><span style="--c:#79549b">Ready standby</span><span style="--c:#ae5547">Productive</span></div></section></div>
'''.replace('reach100','reach 100')

LABELS=r'''const name=id=>id==='daylight'?'Daylight rules':id==='storage'?'Storage rules':id==='storage-persistence'?'Storage with persistence':id.startsWith('ppo-initial-')?'PPO initial · '+id.split('-').at(-1):id.startsWith('ppo-')?'PPO · '+id.split('-').at(-1):id;
const condition=id=>({'restricted-instant':'Electrolysis-only battery · instant downstream','shared-instant':'Whole-plant battery · instant downstream','restricted-dynamic':'Electrolysis-only battery · paid readiness','shared-dynamic':'Whole-plant battery · paid readiness'}[id]||id);
'''

DESCRIPTION=r'''const description=r=>{if(r.controller==='storage-persistence')return 'The unchanged storage rule receives current sunlight in place of its0–1h mean,0–1h p10 and1–6h mean forecasts. Its other inputs and operating rules are identical.';if(r.controller_kind==='baseline')return r.controller==='daylight'?'Inventory and readiness-aware solar allocation, with no battery use.':'Inventory and readiness-aware allocation with explicit charging, startup bridging, useful readiness retention and delivery support. Access follows this arm’s public routing configuration.';if(r.checkpoint_timesteps===0)return 'Random initialization, included only in development to assess learning.';return `Fresh PPO trained only on shared-dynamic at30 kg/h for ${fmt(r.checkpoint_timesteps)} transitions. No demonstrations, pretrained weights, heuristic operating shield or daily stock-restoration objective.`};
'''.replace('its0','its 0').replace(',0',', 0').replace('and1','and 1').replace('at30','at 30')

LEARNING=r'''function learningAndSupplement(){
const learning=[101,211,307].map(seed=>{const initial=data.groups.find(g=>g.split==='development'&&g.condition==='shared-dynamic'&&g.controller==='ppo-initial-'+seed),final=data.groups.find(g=>g.split==='development'&&g.condition==='shared-dynamic'&&g.controller==='ppo-'+seed),row=data.rows.find(r=>r.split==='development'&&r.controller==='ppo-'+seed);return{seed,initial,final,row}});
if(learning.some(p=>p.initial||p.final)){$('learning-section').classList.remove('hidden');$('learning').innerHTML=learning.map(p=>`<tr><td>${p.seed}</td><td>${p.initial?p.initial.delivery_successes+'/'+p.initial.n_scenarios:'Unavailable'}</td><td>${p.initial?fmt(mean(p.initial,'net_operating_value_eur')):'Unavailable'}</td><td>${fmt(p.row?.checkpoint_timesteps)}</td><td>${p.final?p.final.delivery_successes+'/'+p.final.n_scenarios:'Unavailable'}</td><td>${p.final?fmt(mean(p.final,'net_operating_value_eur')):'Unavailable'}</td></tr>`).join('')}
if(data.supplement){const s=data.supplement,c=s.criterion,m=c.forecast_minus_persistence_eur;$('supplement-section').classList.remove('hidden');$('supplement').innerHTML=`<tr><td>Storage with issued forecasts</td><td>${s.reference_delivery_successes}/32</td><td rowspan="2">${fmt(m.mean)}<span class="sub">${m.ci95?'95% interval ['+fmt(m.ci95[0])+', '+fmt(m.ci95[1])+']':'No interval available'}</span></td></tr><tr><td>Storage with persistence</td><td>${s.persistence_delivery_successes}/32</td></tr>`;set('supplement-criterion',`${c.practical_evidence_criterion_met?'The predeclared practical evidence criterion was met.':'The predeclared practical evidence criterion was not met.'} Forecasts rescued ${c.forecast_rescues} scenario(s) and regressed ${c.forecast_regressions}. The criterion required at least four rescues with no regressions, or at least€250/week gain with a positive paired interval and no fewer successful weeks; all paired attempts must complete and pass audits. This criterion and the forecast intervention were declared before development.`)}
if(data.fixture){$('arm-section').classList.add('hidden')}else{const arms=['restricted-instant','shared-instant','restricted-dynamic','shared-dynamic'];$('arms').innerHTML=arms.flatMap(arm=>['daylight','storage'].map(label=>{const g=data.groups.find(g=>g.split==='held_out'&&g.condition===arm&&g.controller===label);return `<tr><td>${esc(condition(arm))}</td><td>${esc(name(label))}</td><td>${g.delivery_successes}/${g.n_scenarios}</td><td>${fmt(mean(g,'net_operating_value_eur'))}</td><td>${fmt(mean(g,'operating:zero_solar_ch4_produced_kg'),1)}</td><td>${fmt(mean(g,'operating:dac_battery_kwh'))}</td><td>${fmt(mean(g,'operating:methanator_battery_kwh'))}</td></tr>`})).join('');const effects=data.arm_comparisons.filter(c=>c.split==='held_out'&&c.effect==='shared_bus_with_readiness');set('arm-effects',effects.map(c=>{const m=c.contribution.all_case;return `${name(c.reference)}: adding downstream battery access with paid readiness changes contribution by ${fmt(m.mean)} €/week${m.ci95?' [95% paired interval '+fmt(m.ci95[0])+', '+fmt(m.ci95[1])+']':''}.`}).join(' '));}
}
learningAndSupplement();
'''.replace('least€','least €')

OVERVIEW=r'''function overview(){const split=$('split').value,cond=$('overview-condition').value,groups=data.groups.filter(g=>g.split===split&&g.condition===cond).sort((a,b)=>sortControllers(a.controller,b.controller));set('overview-caption',`${splitName(split)} · ${condition(cond)}. Means include every prescribed scenario; incomplete weekly value remains unavailable. Brackets give95% scenario-bootstrap intervals.`);$('overview').innerHTML=groups.map(g=>{const e=metric(g,'net_operating_value_eur'),cases=data.rows.filter(r=>r.split===split&&r.condition===cond&&r.controller===g.controller).sort((a,b)=>a.seed-b.seed),recordKey=cond+'|'+g.controller;return `<tr class="${recordKey===key?'selected':''}" ${data.records[recordKey]?`data-record="${esc(recordKey)}" tabindex="0"`:''}><td>${esc(name(g.controller))}<span class="sub">${esc(g.controller==='storage-persistence'?'Three forecasts replaced by current sunlight':g.controller.startsWith('ppo-initial-')?'Random initialization':g.controller.startsWith('ppo-')?'Trained on shared-dynamic':g.controller==='daylight'?'No battery requests':'Same fixed storage algorithm')}</span></td><td>${g.delivery_successes}/${g.n_scenarios}<span class="sub outcomes">${cases.map(r=>`<span class="${!r.execution_complete?'incomplete':r.every_delivery_passed?'':'failed'}" title="Scenario ${r.seed}: ${!r.execution_complete?'incomplete':r.every_delivery_passed?'all deliveries met':'delivery missed'}"></span>`).join('')}</span></td><td>${fmt(e.mean)}<span class="sub">${e.ci95?'['+fmt(e.ci95[0])+', '+fmt(e.ci95[1])+']':'No interval available'}</span></td><td>${fmt(mean(g,'unmet_kg'),1)}</td><td>${fmt(mean(g,'operating:battery_charge_kwh'))}</td><td>${fmt(mean(g,'operating:battery_discharge_kwh'))}</td><td>${fmt(mean(g,'electrolyser_forced_stops'),1)}</td></tr>`}).join('');
const paired=data.comparisons.filter(p=>p.split===split&&p.condition===cond);set('paired',paired.map(p=>{const m=p.contribution.all_case;return `Mean of the three fixed PPO policies minus ${name(p.reference).toLowerCase()}: ${fmt(m.mean)} €/week${m.ci95?' [95% paired interval '+fmt(m.ci95[0])+', '+fmt(m.ci95[1])+']':''}.`}).join(' ')+' Intervals condition on these fixed policies and do not include retraining uncertainty.');
const rs=data.rows.filter(r=>r.split===split&&r.condition===cond);set('auditline',`${rs.filter(r=>r.execution_complete).length}/${rs.length} completed attempts; ${rs.filter(r=>r.physical_invariants_passed).length}/${rs.length} physical audits passed; ${rs.filter(r=>r.economic_audit?.passed).length}/${rs.length} economic audits passed.`);
document.querySelectorAll('[data-record]').forEach(el=>{const go=()=>{key=el.dataset.record;$('condition').value=currentCondition();setupControllers();$('controller').value=data.records[key].summary.controller;drawStatic();$('trace-section').scrollIntoView({behavior:'auto'})};el.onclick=go;el.onkeydown=e=>{if(e.key==='Enter')go()}})}
'''.replace('give95','give 95')

READINESS_PLOTS=r'''chart('readiness',[{field:'dac_readiness_fraction',scale:100,color:'#4c6caf'},{field:'methanator_readiness_fraction',scale:100,color:'#ae5547'}],{max:100,percentage:true});for(const[id,u,color]of[['dac-services','dac','#4c6caf'],['meth-services','methanator','#ae5547']])chart(id,[{field:u+'_startup_power_kw',color:'#ad7815'},{field:u+'_standby_power_kw',color:'#79549b'},{field:u+'_productive_power_kw',color}],{max:u==='dac'?1250:190});'''

READOUT=r'''$('readout').innerHTML=[['Battery charge',fmt(val(m,'battery_soc')*100,3)+'%'],['Total battery support',fmt(['electrolyser','dac','methanator'].reduce((s,u)=>s+val(m,'battery_to_'+u+'_kw'),0),1)+' kW'],['Methane stored',fmt(val(m,'ch4_level_kg'),1)+' kg'],['Methane production',fmt(val(m,'ch4_production_kg_h'),1)+' kg/h'],['Capture readiness',fmt(val(m,'dac_readiness_fraction')*100,1)+'%'],['Methanator readiness',fmt(val(m,'methanator_readiness_fraction')*100,1)+'%'],['Capture battery support',fmt(val(m,'battery_to_dac_kw'),1)+' kW'],['Methanator battery support',fmt(val(m,'battery_to_methanator_kw'),1)+' kW']].map(([a,b])=>`<div><span>${a}</span><strong>${b}</strong></div>`).join('');
'''

COMMANDS=r'''const intervalMean=field=>{let sum=0;for(let j=start+1;j<=Math.min(start+15,r.minutes.length-1);j++)sum+=val(r.minutes[j],field);return sum/15};
$('commands').innerHTML=[['Solar to charging','battery_share','solar_to_battery_kw'],['Solar to electrolysis','electrolyser_share','solar_to_electrolyser_kw'],['Solar to capture','dac_share','solar_to_dac_kw'],['Solar to methanation','methanator_share','solar_to_methanator_kw'],['Battery to electrolysis','battery_support_kw','battery_to_electrolyser_kw'],['Battery to capture','battery_dac_support_kw','battery_to_dac_kw'],['Battery to methanation','battery_methanator_support_kw','battery_to_methanator_kw']].map(([label,k,field])=>`<tr><td>${label}</td><td>${fmt(requested[k]*(k.endsWith('_share')?100:1),2)}${k.endsWith('_share')?'%':' kW'}</td><td>${fmt(applied[k]*(k.endsWith('_share')?100:1),2)}${k.endsWith('_share')?'%':' kW'}</td><td>${fmt(intervalMean(field),1)}</td></tr>`).join('');set('command-note',`${r.summary.variant.shared_bus?'All three battery routes are available.':'The restricted arm explicitly removes battery requests to capture and methanation after inference.'} Solar shares and the three joint battery requests are held for15 minutes. Actual means are independently summed from accepted minute flows, after limits, readiness, inventories and bus netting.`);forecast(command);set('forecast-replacement',command.replacements?.length?'Persistence replacements (issued → controller input): '+command.replacements.map(v=>v.field.replace('solar_forecast_','').replaceAll('_',' ')+' '+fmt(v.original*5000)+' → '+fmt(v.replacement*5000)+' kW').join('; ')+'. Other forecast slots are unused by this rule.':'These are the original causal public forecasts available when this decision was chosen.');
'''.replace('for15','for 15')

NIGHT_ACCOUNT=r'''set('night-account',`Across zero-solar minutes: electrolysis consumed ${fmt(op.zero_solar_electrolyser_kwh,1)} kWh; capture ${fmt(op.zero_solar_dac_kwh,1)} kWh; methanation ${fmt(op.zero_solar_methanator_kwh,1)} kWh. Battery support supplied ${fmt(op.zero_solar_battery_discharge_kwh,1)} kWh and methane production was ${fmt(op.zero_solar_ch4_produced_kg,2)} kg. Zero-solar periods can include daytime outages. These are observed flows, not a requirement to run at night.`);$('thermal-account').innerHTML=[['Preparation / kWh','startup_kwh'],['Ready standby / kWh','standby_kwh'],['Productive / kWh','productive_kwh'],['Readiness losses / kWh','heat_loss_kwh'],['Starts','starts'],['Cold starts','cold_starts'],['Battery input / kWh','battery_kwh']].map(([label,k])=>`<tr><td>${label}</td><td>${fmt(op['dac_'+k],1)}</td><td>${fmt(op['methanator_'+k],1)}</td></tr>`).join('');'''


def write_inspector(data, output):
    output=Path(output);sidecar=output.with_suffix('.build.json')
    if output.exists() or sidecar.exists(): raise FileExistsError('Inspector or build receipt already exists')
    serialized=json.dumps(data,separators=(',',':'),allow_nan=False).encode()
    payload=base64.b64encode(gzip.compress(serialized,compresslevel=9,mtime=0)).decode()
    html=render_template().replace('__PAYLOAD__',payload)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream: stream.write(html)
    receipt={'schema':data['schema']+'-build','fixture':data['fixture'],'scenario':data['scenario'],
        'cases':len(data['rows']),'traces':len(data['records']),'data_sha256':hashlib.sha256(serialized).hexdigest(),
        'html':pin(output),**data['provenance'],
        'recordings':{k:r.get('recording') for k,r in data['records'].items()}}
    with sidecar.open('x') as stream: json.dump(receipt,stream,indent=2,allow_nan=False)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path)
    parser.add_argument('--recordings-root',type=Path)
    parser.add_argument('--fixture-recordings',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=assemble(analysis_path=args.analysis,recordings_root=args.recordings_root,fixtures=args.fixture_recordings)
    print(json.dumps(write_inspector(data,args.output),indent=2))


if __name__=='__main__': main()

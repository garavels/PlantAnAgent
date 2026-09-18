"""Prospectively frozen paired tail panel with endogenous controller reserves."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import traceback
import numpy as np
from docs.research.tools.wholeplant_ppo_v1 import evaluate as audit_tools, training
from docs.research.tools.wholeplant_ppo_v1.baselines import baseline
from docs.research.tools.wholeplant_ppo_v1.portable import file_pin, digest
from docs.research.tools.tail_stress_v1.audit import MinuteAudit
from docs.research.tools.tail_stress_v1.environment import TailStressEnv, SEEDS, CONDITIONS, condition_spec

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT / 'docs/research/results/wholeplant_ppo_20260917'
CONTROLLERS = ('daylight', 'storage', 'ppo-101', 'ppo-211', 'ppo-307')
SCHEMA = 'endogenous-tail-stress-v1'
DEADLINE = '2026-09-17T22:53:00+00:00'

def publish(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')

def freeze(output):
    original_path = OLD / 'held-out-freeze.json'
    original = audit_tools.require_freeze(original_path)
    paths = {Path(p['path']) for p in original['sources']}
    paths.add(ROOT/'docs/research/tools/wholeplant_ppo_v1/portable.py')
    paths.update(Path(__file__).parent / name for name in ('__init__.py','environment.py','audit.py','run.py','analyze.py'))
    paths.update((ROOT/'docs/research/TAIL_STRESS_V1_PROTOCOL.md', ROOT/'tests/test_tail_stress_environment.py', ROOT/'tests/test_tail_stress_analysis.py', ROOT/'tests/test_tail_stress_audit.py'))
    spec = {'schema': SCHEMA, 'created_at': datetime.now(timezone.utc).isoformat(),
        'deadline_utc': DEADLINE, 'seeds': list(SEEDS), 'conditions': list(CONDITIONS),
        'condition_specifications': {c: condition_spec(c) for c in CONDITIONS},
        'controller_order': list(CONTROLLERS), 'controllers': {c: original['controllers'][c] for c in CONTROLLERS},
        'cases': [{'seed': s,'condition':c,'controller':p} for s in SEEDS for c in CONDITIONS for p in CONTROLLERS],
        'expected_cases': len(SEEDS)*len(CONDITIONS)*len(CONTROLLERS), 'attempts_per_case':1,
        'episode_hours':168, 'original_evaluation':file_pin(original_path),
        'sources':[file_pin(p) for p in sorted(paths)],
        'trace_rule':'All actual policy inputs/actions retained; every physical minute audited; full states for first seed of every controller-condition panel',
        'bootstrap':{'unit':'base scenario seed, all five stress conditions kept together','replicates':2000,'confidence_percent':99.1666666667,'primary_comparisons':6},
        'primary':'For each fixed PPO versus each heuristic: >=8 rescued stress cells, zero regressed cells, no higher mean unmet kg, adjusted paired success CI strictly positive',
        'interpretation':'Constructed stress distribution; no estimate of natural event frequency or new-training-seed reliability'}
    spec['sha256'] = digest(spec)
    publish(output,spec)
    return spec

def require_freeze(path):
    spec=json.loads(Path(path).read_text())
    if spec['schema'] != SCHEMA or spec['sha256'] != digest({k:v for k,v in spec.items() if k!='sha256'}):
        raise ValueError('Tail freeze digest differs')
    for record in [spec['original_evaluation'],*spec['sources']]:
        if file_pin(record['path']) != record:
            raise ValueError('Frozen source changed: '+record['path'])
    for controller in spec['controllers'].values():
        for record in controller.get('checkpoint',{}).values():
            if file_pin(record['path']) != record:
                raise ValueError('Frozen policy/normalizer changed')
    return spec

def run_case(freeze_path, seed, condition, controller, directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
    spec=require_freeze(freeze_path)
    if {'seed':seed,'condition':condition,'controller':controller} not in spec['cases']:
        raise ValueError('Case not prescribed')
    deadline=datetime.fromisoformat(spec['deadline_utc'])
    started=time.monotonic(); env=policy=auditor=operating=None
    initial=final=failure=None; states=[]; commands=[]
    timing=condition_spec(condition)
    onset,end=timing['onset_h'],timing['end_h']
    if onset is None: onset=end=48.
    event={'onset_h':onset,'end_h':end,'is_intervention':condition!='control'}
    reward_sum=0.; terminated=truncated=False; evidence_hash=hashlib.sha256()
    retain=seed==SEEDS[0]; forcing=None; base_source=None
    try:
        if datetime.now(timezone.utc)>=deadline: raise TimeoutError('Tail deadline reached')
        env=TailStressEnv(condition=condition,record_minutes=True)
        obs,_=env.reset(seed=seed)
        initial=env.simulator.state.to_dict()
        if retain: states.append(initial)
        forcing=digest([float(v) for v in env.frozen_source.expected_solar_kw])
        base_source=env.base_source_manifest()['base_paired_source_sha256']
        auditor=MinuteAudit(env); operating=audit_tools.OperatingAudit(env)
        auditor.add(initial); operating.add(initial)
        cs=spec['controllers'][controller]
        if cs['kind']=='ppo':
            policy=training.load_checkpoint(Path(cs['checkpoint']['checkpoint']['path']))
            if policy.num_timesteps!=786432 or policy.research_identity!=cs['checkpoint_environment_identity']:
                raise ValueError('Wrong trained policy')
        else: policy=baseline(controller,env)
        for index in range(672):
            if datetime.now(timezone.utc)>=deadline: raise TimeoutError('Tail deadline reached')
            if obs.shape!=(149,) or not np.array_equal(obs,env.current_policy_observation):
                raise ValueError('Actual input mismatch')
            action,_=policy.predict(obs.copy(),deterministic=True)
            evidence={'observation':obs.tolist(),'observation_sha256':digest(obs.tolist())}
            if cs['kind']=='ppo':
                details=policy.deterministic_details(obs.copy())
                if not np.array_equal(action,details['clipped_action']): raise ValueError('PPO witness mismatch')
                evidence.update(normalized_observation=details['normalized_input'].tolist(),raw_mean=details['unclipped_mean'].tolist())
            else:
                checked,details=policy.inference(obs.copy())
                if not np.array_equal(action,checked): raise ValueError('Baseline witness mismatch')
                evidence['baseline_inference']=details
            action=np.asarray(action)
            if action.shape!=(7,) or not np.isfinite(action).all(): raise ValueError('Invalid action')
            evidence['action']=action.tolist()
            evidence_hash.update(json.dumps(evidence,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
            obs,reward,terminated,truncated,info=env.step(action); reward_sum+=float(reward)
            trace=env.last_minute_truth_states
            if trace is None or len(trace)!=16: raise ValueError('Missing minute trace')
            audit_tools.audit_intervention(env,action,info,trace)
            for state in trace[1:]:
                auditor.add(state); operating.add(state)
                if retain: states.append(state.to_dict())
            commands.append({'decision_index':index,'evidence':evidence,'action':action.tolist(),'reward_eur':float(reward)})
            for key,boundary in (('pre_event_state',onset),('post_event_state',end)):
                if env.physics_steps_taken==round(boundary*60): event[key]=deepcopy(env.simulator.state.to_dict())
            if terminated or truncated: break
    except BaseException as exc:
        failure={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    try:
        final=env.simulator.state.to_dict() if initial is not None else None
        delivery=env.simulator.delivery_audit() if final else None
        if delivery:
            settlements=delivery['settlements']
            event['pre_event_unmet_kg']=sum(d['unmet_kg'] for d in settlements if d['due_step']<=event['onset_h']*60)
            event['event_unmet_kg']=sum(d['unmet_kg'] for d in settlements if event['onset_h']*60<d['due_step']<=event['end_h']*60)
            event['post_event_unmet_kg']=sum(d['unmet_kg'] for d in settlements if d['due_step']>event['end_h']*60)
        complete=bool(failure is None and terminated and not truncated and env.decisions_taken==672 and auditor.count==10081)
        try: require_freeze(freeze_path); unchanged=True
        except Exception as exc:
            unchanged=False; complete=False; failure=failure or {'type':type(exc).__name__,'message':str(exc)}
        row={'schema':SCHEMA+'-case','seed':seed,'condition':condition,'controller':controller,
            'freeze_sha256':spec['sha256'],'execution_complete':complete,'failure':failure,'sources_unchanged':unchanged,
            'decisions':len(commands),'physical_states':auditor.count if auditor else 0,
            'initial_state':initial,'final_state':final,'delivery':delivery,
            'economic_metrics':env.economic_metrics.to_dict() if final else None,
            'physical_metrics':env.metrics.to_dict() if final else None,
            'operating_metrics':operating.totals if operating else None,
            'minute_audit':auditor.result() if auditor else {'all_physical_invariants_passed':False},
            'economic_audit':audit_tools.economic_audit(env,initial,final,reward_sum,operating) if final else {'passed':False},
            'every_delivery_passed':bool(complete and len(delivery['settlements'])==28 and all(d['unmet_kg']<=1e-9 for d in delivery['settlements'])),
            'event_metrics':event,'base_paired_source_sha256':base_source,'physical_solar_forcing_sha256':forcing,
            'input_action_witness_sha256':evidence_hash.hexdigest(),'wall_time_s':time.monotonic()-started}
        recording={'schema':SCHEMA+'-recording','summary':row,'decision_trace':commands,'minute_truth_states':states,
            'environment_manifest':env.audit_manifest() if retain and final else None}
        recordpath=directory/'recording.json.gz'
        with recordpath.open('xb') as stream:
            stream.write(gzip.compress(json.dumps(recording,separators=(',',':'),allow_nan=False).encode(),compresslevel=1,mtime=0))
        row['recording']=file_pin(recordpath)
        publish(directory/'result.json',row)
        return row
    finally:
        if policy is not None and hasattr(policy,'close'): policy.close()
        if env is not None: env.close()

def _worker(args):
    return run_case(**args)

def campaign(output):
    output=Path(output).resolve(); spec=require_freeze(output/'freeze.json')
    tasks=[{'freeze_path':output/'freeze.json',**c,'directory':output/'cases'/f"{c['condition']}--{c['controller']}--{c['seed']}"} for c in spec['cases']]
    rows=[]
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn'),initializer=training.configure_torch) as pool:
        pending={pool.submit(_worker,args):args for args in tasks}
        for future in as_completed(pending):
            row=future.result(); rows.append(row)
            print(json.dumps({'completed':len(rows),'seed':row['seed'],'condition':row['condition'],'controller':row['controller'],
                'complete':row['execution_complete'],'audit':row['minute_audit']['all_physical_invariants_passed'],
                'unmet_kg':row['economic_metrics']['unmet_kg'] if row['economic_metrics'] else None,'failure':row['failure']}),flush=True)
    checks=all(r['execution_complete'] and r['sources_unchanged'] and r['minute_audit']['all_physical_invariants_passed'] and r['economic_audit']['passed'] for r in rows)
    pairing=all(len({r['base_paired_source_sha256'] for r in rows if r['seed']==s})==1 and len({digest(r['initial_state']) for r in rows if r['seed']==s})==1 for s in SEEDS)
    forcing=all(len({r['physical_solar_forcing_sha256'] for r in rows if r['seed']==s and r['condition']==c})==1 for s in SEEDS for c in CONDITIONS)
    publish(output/'summary.json',{'schema':SCHEMA+'-summary','completed_at':datetime.now(timezone.utc).isoformat(),
        'freeze':file_pin(output/'freeze.json'),'case_count':len(rows),'all_executions_and_audits_passed':checks,
        'paired_base_sources_and_openings_equal':pairing,'paired_forcing_equal':forcing,'rows':rows})
    if not(checks and pairing and forcing): raise RuntimeError('Incomplete or invalid tail evidence')

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('freeze','run')); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='freeze': print(json.dumps({'freeze_sha256':freeze(args.output/'freeze.json')['sha256']}))
    else: campaign(args.output)

if __name__=='__main__': main()

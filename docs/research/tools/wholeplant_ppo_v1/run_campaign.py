"""Bounded one-attempt supervisor for the authorized whole-plant campaign.

Scientific sources and case membership are frozen before canonical evaluation.
The supervisor does not tune physics/controllers based on canonical outcomes.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
RESULTS=ROOT/'docs/research/results/wholeplant_ppo_20260917'
PREFIX='docs.research.tools.wholeplant_ppo_v1'


def now():return datetime.now(timezone.utc)
def dt(value):return datetime.fromisoformat(value.replace('Z','+00:00'))
def pin(path):
 path=Path(path).resolve(); data=path.read_bytes()
 return {'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def publish(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def read(path):return json.loads(Path(path).read_text())


class Campaign:
 def __init__(self,results):
  self.results=Path(results).resolve();self.supervision=self.results/'supervision'
  self.supervision.mkdir(exist_ok=False)
  self.auth=read(self.results/'authorization.json')
  self.deadline=dt(self.auth['deadline_at']);self.training_deadline=dt(self.auth['training_cutoff_at'])
  self.child=None;self.stage='preflight';self.started=time.monotonic();self.receipts=[]
  self.sources=None;self.self_pin=pin(__file__)

 def event(self,event,**details):
  record={'at':now().isoformat(),'stage':self.stage,'event':event,**details}
  with (self.supervision/'events.jsonl').open('a') as f:f.write(json.dumps(record,sort_keys=True)+'\n')
  temporary=self.supervision/'status.writing.json';temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(self.supervision/'status.json')
  print(json.dumps(record),flush=True)

 def check(self):
  if now()>=self.deadline:raise TimeoutError('Authorized four-hour deadline reached')
  if pin(__file__)!=self.self_pin:raise ValueError('Supervisor source changed after launch')
  if self.sources is not None:
   from docs.research.tools.wholeplant_ppo_v1.environment import source_records
   if source_records()!=self.sources:raise ValueError('Scientific sources changed during campaign')

 def command(self,name,args,*,timeout_at=None):
  self.check();self.stage=name;self.event('started',command=args)
  started=time.monotonic();log=self.supervision/(name+'.log')
  deadline=timeout_at or self.deadline
  with log.open('x') as stream:
   self.child=subprocess.Popen(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
   last=0.0
   while self.child.poll() is None:
    if now()>=deadline:
     os.killpg(self.child.pid,signal.SIGTERM)
     try:self.child.wait(timeout=15)
     except subprocess.TimeoutExpired:
      os.killpg(self.child.pid,signal.SIGKILL);self.child.wait()
     raise TimeoutError(f'{name} exceeded its stage deadline; retained logs')
    if time.monotonic()-last>=60:
     self.event('running',pid=self.child.pid,elapsed_s=round(time.monotonic()-started,1));last=time.monotonic()
    time.sleep(1)
   code=self.child.returncode;self.child=None
  receipt={'stage':name,'exit_code':code,'elapsed_s':time.monotonic()-started,'log':pin(log)}
  self.receipts.append(receipt);self.event('finished',**receipt)
  if code:raise RuntimeError(f'{name} failed with exit {code}; see retained log')
  self.check()
  return receipt

 def gate(self,split,endpoints=None):
  from docs.research.tools.wholeplant_ppo_v1 import environment as e
  path=self.results/(split.replace('_','-')+'-gate.json')
  publish(path,{'schema':'wholeplant-ppo-v1-evaluation-gate','created_at':now().isoformat(),
   'environment_identity_sha256':e.environment_identity_sha256(),'source_identity_sha256':e.source_identity_sha256(),
   'allowed_splits':[split],'endpoints_frozen':split=='held_out','endpoints':endpoints})
  return path

 def spec(self,name,controllers,panels,gate):
  from docs.research.tools.wholeplant_ppo_v1 import environment as e,evaluate
  path=self.results/(name+'-spec.json');freeze=self.results/(name+'-freeze.json')
  payload={'controllers':controllers,'arms':list(e.ARMS),'panels':panels,'gate_path':str(gate),'description':name}
  publish(path,payload);evaluate.freeze_manifest(path,freeze)
  return freeze,payload

 def wave(self,name,freeze,spec,gate):
  root=self.results/name;root.mkdir(exist_ok=False)
  count=0
  for panel in spec['panels']:
   label=f"{name}--{panel['controller']}--{panel['arm']}"
   folder=root/f"{panel['controller']}--{panel['arm']}"
   self.command(label,[sys.executable,'-m',PREFIX+'.evaluate','panel','--freeze',str(freeze),'--gate',str(gate),
    '--controller',panel['controller'],'--arm',panel['arm'],'--split',panel['split'],'--output',str(folder),
    '--workers','3','--deadline-utc',self.auth['deadline_at']])
   summary=read(folder/'summary.json');count+=len(summary['runs'])
   if not summary['all_case_audits_passed'] or summary['execution_complete_count']!=len(summary['requested_seeds']):
    raise RuntimeError(f'{label} has incomplete execution or failed accounting; retained without replacement')
  self.event('wave_completed',wave=name,retained_cases=count)
  return count

 def main(self):
  from docs.research.tools.wholeplant_ppo_v1 import environment as e,training as t,evaluate
  self.sources=e.source_records();self.check()
  if self.auth['compute']!='local CPU only' or self.auth['horizon_hours']!=168:raise ValueError('Unexpected authorization')
  publish(self.supervision/'launch.json',{'started_at':now().isoformat(),'authorization':pin(self.results/'authorization.json'),
    'supervisor_source':self.self_pin,'scientific_sources':self.sources,'expected_total_cases':496})
  baselines=[{'label':name,'kind':'baseline','baseline':name} for name in ('daylight','storage')]
  devgate=self.gate('development')
  devpanels=[{'controller':c['label'],'arm':arm,'split':'development'} for c in baselines for arm in e.ARMS]
  freeze,spec=self.spec('development-baselines',baselines,devpanels,devgate)
  # Membership of later panels is registered before any development outcome.
  future={'learning':[{'controller':f'ppo-{phase+"-" if phase=="initial" else ""}{seed}',
       'arm':'shared-dynamic','split':'development'} for seed in (101,211,307) for phase in ('initial','final')],
    'held_out':[{'controller':c['label'],'arm':arm,'split':'held_out'} for c in baselines for arm in e.ARMS]+
       [{'controller':f'ppo-{seed}','arm':'shared-dynamic','split':'held_out'} for seed in (101,211,307)]+
       [{'controller':'storage-persistence','arm':'shared-dynamic','split':'held_out'}],
    'fixed_optimizer_seeds':[101,211,307], 'checkpoint_selection':'initial and exact fixed final; no score selection'}
  publish(self.results/'prescribed-endpoints-and-panels.json',future)
  baseline_count=self.wave('development-baselines',freeze,spec,devgate)
  # Discarded pilot: only throughput determines the production budget.
  pilot=self.results/'preflight/throughput-pilot'
  t.freeze_plan(pilot,total_timesteps=16384,deadline_utc=self.auth['training_cutoff_at'],checkpoint_steps=65536)
  self.command('throughput-pilot',[sys.executable,'-m',PREFIX+'.training','campaign','--root',str(pilot)],timeout_at=min(self.deadline,self.training_deadline+timedelta(seconds=60)))
  closure=read(pilot/'campaign-closure.json')
  if not closure['all_fixed_endpoints_completed']:raise RuntimeError('Pilot did not complete all learners')
  available=(self.training_deadline-now()).total_seconds()-90
  budgets=[786432,655360,524288,393216,262144]
  predictions=[{'transitions':budget,'predicted_s':closure['elapsed_s']*budget/16384,
                'with_25_percent_margin_s':1.25*closure['elapsed_s']*budget/16384} for budget in budgets]
  feasible=[row for row in predictions if row['with_25_percent_margin_s']<=available]
  decision={'created_at':now().isoformat(),'pilot_closure':pin(pilot/'campaign-closure.json'),
    'available_training_seconds_after_90s_reserve':available,'candidates':predictions,
    'train_new_policies':bool(feasible),'total_timesteps':feasible[0]['transitions'] if feasible else None,
    'selection_uses_only_throughput_and_remaining_time':True,'pilot_policies_discarded':True}
  publish(self.results/'training-decision.json',decision)
  if not feasible:raise TimeoutError('No prospective minimum training budget fits; baseline evidence retained')
  trainroot=self.results/'training'
  plan=t.freeze_plan(trainroot,total_timesteps=decision['total_timesteps'],deadline_utc=self.auth['training_cutoff_at'])
  self.command('production-training',[sys.executable,'-m',PREFIX+'.training','campaign','--root',str(trainroot)],timeout_at=min(self.deadline,self.training_deadline+timedelta(seconds=60)))
  closure=read(trainroot/'campaign-closure.json')
  if not closure['all_fixed_endpoints_completed']:raise RuntimeError('Not all production endpoints completed')
  endpoints={};learning=[];finals=[]
  t.configure_torch()
  for seed in (101,211,307):
   resultpath=trainroot/f'seed-{seed}/result.json';result=read(resultpath)
   if (result['status']!='completed' or result['num_timesteps']!=plan['total_timesteps'] or not result['actor_mean_parameters_changed']):
    raise ValueError('Final endpoint lacks prescribed actor learning evidence')
   entries={'result':pin(resultpath)}
   for phase,steps in (('initial',0),('final',plan['total_timesteps'])):
    checkpoint=trainroot/f'seed-{seed}/{phase}-policy.zip'
    loaded=t.load_checkpoint(checkpoint,trainroot/'plan.json')
    try:
     if loaded.num_timesteps!=steps or loaded.seed!=seed:raise ValueError('Checkpoint identity mismatch')
    finally:loaded.close()
    entries[phase]=evaluate.checkpoint_record(checkpoint)
    ctrl={'label':f'ppo-{seed}' if phase=='final' else f'ppo-initial-{seed}',
       'kind':'ppo','checkpoint':str(checkpoint),'optimizer_seed':seed,'phase':phase}
    learning.append(ctrl)
    if phase=='final':finals.append(ctrl)
   endpoints[str(seed)]=entries
  endpointfile=self.supervision/'endpoints-frozen.json'
  publish(endpointfile,{'created_at':now().isoformat(),'training_plan':pin(trainroot/'plan.json'),
      'closure':pin(trainroot/'campaign-closure.json'),'endpoints':endpoints})
  learningfreeze,learningspec=self.spec('development-learning',learning,future['learning'],devgate)
  learning_count=self.wave('development-learning',learningfreeze,learningspec,devgate)
  heldgate=self.gate('held_out',pin(endpointfile))
  heldcontrollers=baselines+finals+[{'label':'storage-persistence','kind':'baseline','baseline':'storage-persistence'}]
  heldfreeze,heldspec=self.spec('held-out',heldcontrollers,future['held_out'],heldgate)
  held_count=self.wave('held-out',heldfreeze,heldspec,heldgate)
  self.command('analysis',[sys.executable,'-m',PREFIX+'.analyze','--root',str(self.results/'development-baselines'),
     str(self.results/'development-learning'),str(self.results/'held-out'),'--output',str(self.results/'analysis')])
  analysis=read(self.results/'analysis/analysis.json')
  if baseline_count+learning_count+held_count!=496 or len(analysis['rows'])!=496:raise ValueError('Prescribed case total mismatch')
  self.check()
  receipt={'status':'completed','completed_at':now().isoformat(),'elapsed_s':time.monotonic()-self.started,
      'rows':496,'training_transitions_per_seed':plan['total_timesteps'],'analysis':pin(self.results/'analysis/analysis.json'),
      'endpoints':pin(endpointfile),'stages':self.receipts,'scientific_sources_unchanged':True}
  publish(self.supervision/'completion.json',receipt);self.event('completed',rows=496,analysis=receipt['analysis'])
  return receipt


def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=RESULTS);args=p.parse_args()
 campaign=Campaign(args.results)
 try:campaign.main()
 except BaseException as exc:
  if campaign.child is not None and campaign.child.poll() is None:
   os.killpg(campaign.child.pid,signal.SIGTERM)
  publish(campaign.supervision/'failure.json',{'at':now().isoformat(),'stage':campaign.stage,'type':type(exc).__name__,'message':str(exc),'stages':campaign.receipts})
  raise
if __name__=='__main__':main()

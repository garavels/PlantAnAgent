"""Fit one declared ON-mean row with parent-centered ridge regression."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['VECLIB_MAXIMUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import torch
from stable_baselines3 import PPO
from docs.research.tools.learned_control_hybrid_v1 import training as hybrid
from docs.research.tools.learned_control_hybrid_v1.policy import HybridActorCriticPolicy, distribution_identity
from docs.research.tools.learned_control_v1 import imitation, environment
from docs.research.tools.wholeplant_ppo_v1 import training as common

ROOT=Path(__file__).resolve().parents[4]
SCHEMA='learned-control-mean-head-refit-v1'

def source_records(extra=()):
    paths={Path(row['path']) for row in environment.source_records()}
    paths.update(Path(hybrid.__file__).parent.glob('*.py'))
    paths.update(Path(__file__).parent.glob('*.py'))
    paths.update([Path(common.__file__),Path(imitation.__file__),Path(imitation.__file__).with_name('training.py'),ROOT/'uv.lock',*map(Path,extra)])
    return [common.file_record(p) for p in sorted(paths)]

def input_records(parent,dataset):
    metadata=imitation.require_dataset(dataset)
    return [common.file_record(parent.with_suffix(s)) for s in ['.zip','.json','.vecnormalize.pkl']]+[
        common.file_record(dataset),common.file_record(dataset.with_suffix('.json')),metadata['plan']]

def freeze_plan(root,parent,dataset,*,optimizer_seed=903,deadline_utc='2026-09-18T02:00:00Z',diagnostic_paths=()):
    root,parent,dataset=map(lambda p:Path(p).resolve(),(root,parent,dataset))
    common._deadline(deadline_utc)
    loaded=hybrid.load_checkpoint(parent)
    try:
        if type(loaded.model.policy) is not HybridActorCriticPolicy or loaded.model.num_timesteps!=0 or int(loaded.model.seed)!=optimizer_seed:
            raise ValueError('Require the declared initialized plain hybrid parent and its inherited seed')
        parent_optimizer=common.optimizer_sha256(loaded.model)
    finally:loaded.close()
    plan={'schema':SCHEMA,'created_at':common.now(),'recipe':{'parent_checkpoint':str(parent),'dataset':str(dataset),
        'optimizer_seed':optimizer_seed,'coordinates':[1],'regularization':.001,'holdout_modulus':5,'holdout_residue':0,
        'deadline_utc':deadline_utc},'method':'Supervised parent-centered ridge mean-head correction, conditional on teacher ON labels; no PPO or gradient steps',
        'selection':'Solar-electrolysis row retained after the separately pinned repair probe improved held-out ON MAE31.7%; DAC candidate was rejected because MAE worsened8.6%. No development/final outcomes selected these rows.',
        'holdout_caveat':'The parent representation already saw all120weeks. Only this head correction withholds episodes whose index modulo5 is0; this is a repair holdout, not independent policy-generalization evidence.',
        'seed_semantics':'Inherit original neural-training seed; ridge solve is deterministic and is not an independent optimizer replicate',
        'environment_identity':environment.environment_identity(),'source_files':source_records(),
        'dependencies':input_records(parent,dataset)+[common.file_record(Path(p)) for p in diagnostic_paths],
        'parent_optimizer_sha256':parent_optimizer,'runtime':common.runtime_record()}
    common.publish_json(root/'plan.json',plan)
    return plan

def require_plan(root):
    plan=json.loads((Path(root)/'plan.json').read_text())
    if plan['schema']!=SCHEMA or plan['source_files']!=source_records() or plan['runtime']!=common.runtime_record() or plan['environment_identity']!=environment.environment_identity():
        raise ValueError('Frozen refit source/runtime/environment changed')
    for row in plan['dependencies']:
        if common.file_record(Path(row['path']))!=row:raise ValueError('Frozen refit dependency changed')
    return plan

def ridge_correction(latent,target,parent_weight,parent_bias,selected,regularization):
    x=np.column_stack((latent[selected],np.ones(int(selected.sum())))).astype(np.float64)
    old=np.r_[parent_weight,parent_bias].astype(np.float64)
    residual=np.asarray(target)[selected]-x@old
    delta=np.linalg.solve(x.T@x+len(x)*regularization*np.eye(x.shape[1]),x.T@residual)
    fitted=(old+delta).astype(np.float32)
    return fitted[:-1],fitted[-1],{'fit_rows':len(x),'coefficient_correction_norm':float(np.linalg.norm(delta))}

def fidelity(action,labels,selected):
    a,y=action[selected],labels[selected]
    return {'rows':len(a),'mean_absolute_action_error':float(abs(a-y).mean()),
        'on_mean_absolute_error':[float(abs(a[:,j]-y[:,j])[y[:,j]>-1].mean()) for j in range(7)],
        'on_to_exact_off_fraction':[float((a[y[:,j]>-1,j]==-1).mean()) for j in range(7)]}

def save_authenticated(policy,root,plan,*,kind,witness):
    """Shared serializer for supervised initializers; caller proves its intervention."""
    root=Path(root);out=root/'training';out.mkdir(exist_ok=False)
    model=policy.model
    model.research_plan_sha256=common.sha(plan)
    model.research_identity=plan['environment_identity']
    model.research_policy_distribution=distribution_identity()
    model.research_supervised_initialization={'method':plan['method'],'witness':witness}
    model.seed=plan['recipe']['optimizer_seed']
    checkpoint=out/'initial-policy.zip'
    model.save(checkpoint)
    normalizer=checkpoint.with_suffix('.vecnormalize.pkl')
    normalizer.write_bytes(Path(plan['recipe']['parent_checkpoint']).with_suffix('.vecnormalize.pkl').read_bytes())
    states=random.getstate(),np.random.get_state(),torch.get_rng_state()
    try:restored=PPO.load(checkpoint,device='cpu')
    finally:random.setstate(states[0]);np.random.set_state(states[1]);torch.set_rng_state(states[2])
    hashes={'policy_tensor_sha256':common.policy_parameter_sha256(model),'actor_tensor_sha256':common.actor_parameter_sha256(model),
        'actor_mean_tensor_sha256':common.actor_mean_parameter_sha256(model),'optimizer_state_sha256':common.optimizer_sha256(model)}
    if common.policy_parameter_sha256(restored)!=hashes['policy_tensor_sha256'] or common.optimizer_sha256(restored)!=hashes['optimizer_state_sha256']:
        raise AssertionError('Supervised checkpoint did not reopen exactly')
    record={'schema':plan['schema'],'kind':kind,'created_at':common.now(),**hashes,'checkpoint':common.file_record(checkpoint),
        'normalizer':common.file_record(normalizer),'plan_sha256':common.sha(plan),'training_plan':common.file_record(root/'plan.json'),
        'environment_identity':plan['environment_identity'],'policy_distribution':distribution_identity(),
        'optimizer_seed':int(model.seed),'optimizer_algorithm':plan['method'],'seed_semantics':plan['seed_semantics'],
        'num_timesteps':model.num_timesteps,'additional_timesteps':0,'ppo_update_count':model._n_updates,'ppo_updates_this_run':0,
        'gradient_updates_this_run':0,'normalizer_observation_count':float(policy.normalizer.obs_rms.count),'gaussian_log_std':None,
        'normalizer_bytes_identical_to_parent':True,'checkpoint_reopen_verified':True,'witness':witness}
    common.publish_json(checkpoint.with_suffix('.json'),record)
    return record

def train(root):
    common.configure_torch();root=Path(root).resolve();plan=require_plan(root);r=plan['recipe']
    if datetime.now(timezone.utc)>=common._deadline(r['deadline_utc']):raise TimeoutError('Refit deadline passed')
    started=time.monotonic();policy=hybrid.load_checkpoint(r['parent_checkpoint'])
    try:
        with np.load(r['dataset'],allow_pickle=False) as data:obs,labels,episodes=data['observations'],data['actions'],data['episode_indices']
        normal=policy.normalize_observation(obs)
        with torch.no_grad():latent=policy.model.policy.mlp_extractor.forward_actor(torch.from_numpy(normal)).numpy()
        before=policy.predict(obs,deterministic=True)[0]
        tensors={name:value.detach().clone() for name,value in policy.model.policy.state_dict().items()}
        holdout=episodes%r['holdout_modulus']==r['holdout_residue']
        fits={}
        for coordinate in r['coordinates']:
            selected=(~holdout)&(labels[:,coordinate]>-1)
            row=7+coordinate
            weight,bias,fit=ridge_correction(latent,labels[:,coordinate],tensors['action_net.weight'][row].numpy(),
                float(tensors['action_net.bias'][row]),selected,r['regularization'])
            with torch.no_grad():
                policy.model.policy.action_net.weight[row].copy_(torch.from_numpy(weight))
                policy.model.policy.action_net.bias[row].copy_(torch.tensor(bias))
            fits[str(coordinate)]=fit
        after=policy.predict(obs,deterministic=True)[0]
        mutable=[7+c for c in r['coordinates']]
        for name,value in policy.model.policy.state_dict().items():
            keep=[i for i in range(21) if i not in mutable] if name in ('action_net.weight','action_net.bias') else None
            if not torch.equal(value if keep is None else value[keep],tensors[name] if keep is None else tensors[name][keep]):
                raise AssertionError('Refit changed an undeclared tensor')
        other=[c for c in range(7) if c not in r['coordinates']]
        if not np.array_equal(before[:,other],after[:,other]):raise AssertionError('Unselected action coordinates changed')
        if common.optimizer_sha256(policy.model)!=plan['parent_optimizer_sha256']:raise AssertionError('PPO optimizer changed')
        metrics={name:{'before':fidelity(before,labels,mask),'after':fidelity(after,labels,mask)}
            for name,mask in [('fit_rows',~holdout),('repair_holdout',holdout),('all_training_rows',np.ones(len(obs),dtype=bool))]}
        witness={'all_other_tensors_bitwise_unchanged':True,'other_action_coordinates_bitwise_unchanged':other,
            'action_witness_observations':len(obs),'ppo_optimizer_exactly_preserved':True,'fit':fits,
            'fit_episode_indices':np.unique(episodes[~holdout]).tolist(),'repair_holdout_episode_indices':np.unique(episodes[holdout]).tolist(),
            'metrics':metrics}
        if not all(metrics['repair_holdout']['after']['on_mean_absolute_error'][j]<.9*metrics['repair_holdout']['before']['on_mean_absolute_error'][j] for j in r['coordinates']):
            raise ValueError('Registered refit does not reproduce the material repair-holdout gain')
        require_plan(root)
        if datetime.now(timezone.utc)>=common._deadline(r['deadline_utc']):raise TimeoutError('Refit finished after deadline')
        record=save_authenticated(policy,root,plan,kind='supervised-solar-mean-refit',witness=witness)
        restored=hybrid.load_checkpoint(record['checkpoint']['path'])
        try:
            if not np.array_equal(restored.predict(obs,deterministic=True)[0],after):raise AssertionError('Reopened refit actions changed')
        finally:restored.close()
        result={'status':'completed','elapsed_s':time.monotonic()-started,'checkpoint':record,'witness':witness}
        common.publish_json(root/'result.json',result);return result
    finally:policy.close()

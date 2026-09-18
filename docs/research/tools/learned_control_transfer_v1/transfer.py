"""Merge learned trunks into one ordinary dense hybrid actor; no operating rules."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
from docs.research.tools.learned_control_refit_v1 import refit
from docs.research.tools.learned_control_hybrid_v1.policy import HybridActorCriticPolicy

common,hybrid=refit.common,refit.hybrid
SCHEMA='learned-control-algebraic-actor-transfer-v1'

def source_records():
    return refit.source_records(Path(__file__).parent.glob('*.py'))

def freeze_plan(root,parent,gaussian,dataset,*,optimizer_seed=903,deadline_utc='2026-09-18T02:00:00Z'):
    root,parent,gaussian,dataset=map(lambda p:Path(p).resolve(),(root,parent,gaussian,dataset))
    common._deadline(deadline_utc)
    plan={'schema':SCHEMA,'created_at':common.now(),'recipe':{'parent_checkpoint':str(parent),'gaussian_checkpoint':str(gaussian),
        'dataset':str(dataset),'optimizer_seed':optimizer_seed,'gaussian_mean_coordinates':[1,2],
        'holdout_modulus':5,'holdout_residue':0,'deadline_utc':deadline_utc},
        'method':'Algebraic concatenation and block-diagonal initialization from two supervised neural actors; Gaussian supplies solar-electrolysis/DAC means, hybrid supplies all gates/scales/other means',
        'seed_semantics':'Both neural parents use the stated original training seed; deterministic merging creates no independent optimizer replicate',
        'architecture':'One ordinary512x512tanh actor: concatenate first-layer rows, block-diagonal second-layer weights, concatenate second-layer biases, and place each learned output row in its source columns. All resulting parameters remain trainable.',
        'selection':'Solar-electrolysis and DAC means selected from the Gaussian parent because their supervised training ON errors were substantially lower; all other coordinates retain the hybrid source. No deployment/final outcome selects this mapping.',
        'holdout_caveat':'Both parents already saw the complete dataset; episode-index modulo5 holdout is a diagnostic subset, not independent policy-generalization evidence.',
        'environment_identity':refit.environment.environment_identity(),'runtime':common.runtime_record(),
        'source_files':source_records(),'dependencies':refit.input_records(parent,dataset)+[
            common.file_record(gaussian.with_suffix(s)) for s in ['.zip','.json','.vecnormalize.pkl']]}
    common.publish_json(root/'plan.json',plan);return plan

def require_plan(root):
    plan=json.loads((Path(root)/'plan.json').read_text())
    if plan['schema']!=SCHEMA or plan['source_files']!=source_records() or plan['runtime']!=common.runtime_record() or plan['environment_identity']!=refit.environment.environment_identity():
        raise ValueError('Frozen transfer sources/runtime/environment changed')
    for pin in plan['dependencies']:
        if common.file_record(Path(pin['path']))!=pin:raise ValueError('Frozen transfer dependency changed')
    return plan

def merged_policy(hybrid_model,gaussian_model,coordinates=(1,2)):
    h,g=hybrid_model.policy,gaussian_model.policy
    if type(h) is not HybridActorCriticPolicy or hybrid_model.num_timesteps or gaussian_model.num_timesteps:
        raise ValueError('Merge only initialized hybrid and Gaussian neural parents')
    if h.optimizer.state or g.optimizer.state:
        raise ValueError('The algebraic initializer requires empty parent PPO optimizer states')
    if h.features_extractor.state_dict() or g.features_extractor.state_dict():
        raise ValueError('Public features must be unchanged parameter-free flattening')
    for policy in (h,g):
        net=policy.mlp_extractor.policy_net
        if len(net)!=4 or not isinstance(net[1],torch.nn.Tanh) or not isinstance(net[3],torch.nn.Tanh):
            raise ValueError('Expected the existing two-layer tanh actors')
        if net[0].weight.shape!=(256,149) or net[2].weight.shape!=(256,256):
            raise ValueError('Expected both parent actors to have256x256hidden dimensions')
    kwargs=copy.deepcopy(hybrid_model.policy_kwargs)
    kwargs['net_arch']['pi']=[512,512]
    new=HybridActorCriticPolicy(h.observation_space,h.action_space,hybrid_model.lr_schedule,**kwargs)
    with torch.no_grad():
        destination=new.mlp_extractor.policy_net
        for layer in (0,2):
            destination[layer].bias.copy_(torch.cat((h.mlp_extractor.policy_net[layer].bias,g.mlp_extractor.policy_net[layer].bias)))
        destination[0].weight.copy_(torch.cat((h.mlp_extractor.policy_net[0].weight,g.mlp_extractor.policy_net[0].weight)))
        destination[2].weight.zero_()
        destination[2].weight[:256,:256].copy_(h.mlp_extractor.policy_net[2].weight)
        destination[2].weight[256:,256:].copy_(g.mlp_extractor.policy_net[2].weight)
        new.mlp_extractor.value_net.load_state_dict(h.mlp_extractor.value_net.state_dict())
        new.value_net.load_state_dict(h.value_net.state_dict())
        new.action_net.weight.zero_();new.action_net.weight[:,:256].copy_(h.action_net.weight)
        new.action_net.bias.copy_(h.action_net.bias)
        for coordinate in coordinates:
            new.action_net.weight[7+coordinate,:256].zero_()
            new.action_net.weight[7+coordinate,256:].copy_(g.action_net.weight[coordinate])
            new.action_net.bias[7+coordinate].copy_(g.action_net.bias[coordinate])
    return new,kwargs

def outputs(policy,normalized):
    with torch.no_grad():
        parameters=torch.cat([policy.get_distribution(torch.from_numpy(batch)).parameters.detach().clone()
            for batch in np.array_split(normalized,40)])
    distribution=policy.action_dist.proba_distribution(parameters)
    return parameters,distribution.mode().numpy()

def train(root):
    common.configure_torch();root=Path(root).resolve();plan=require_plan(root);r=plan['recipe']
    if datetime.now(timezone.utc)>=common._deadline(r['deadline_utc']):raise TimeoutError('Transfer deadline passed')
    h=hybrid.load_checkpoint(r['parent_checkpoint']);g=common.load_checkpoint(r['gaussian_checkpoint']);started=time.monotonic()
    try:
        if h.model.seed!=r['optimizer_seed'] or g.model.seed!=r['optimizer_seed']:raise ValueError('Parent training seeds differ from declared lineage')
        with np.load(r['dataset'],allow_pickle=False) as data:obs,labels,episodes=data['observations'],data['actions'],data['episode_indices']
        normalized=h.normalize_observation(obs)
        if not np.array_equal(normalized,g.normalize_observation(obs)):raise ValueError('Parent normalizers differ on authenticated inputs')
        for attr in ('mean','var','count'):
            if not np.array_equal(getattr(h.normalizer.obs_rms,attr),getattr(g.normalizer.obs_rms,attr)):raise ValueError('Parent observation statistics differ')
        for attr in ('clip_obs','epsilon','norm_obs'):
            if getattr(h.normalizer,attr)!=getattr(g.normalizer,attr):raise ValueError('Parent observation normalization rules differ')
        before_parameters,before=outputs(h.model.policy,normalized)
        gaussian=g.predict(obs,deterministic=True)[0]
        with torch.no_grad():gaussian_means=g.model.policy.get_distribution(torch.from_numpy(normalized)).distribution.mean.detach().clone()
        old_optimizer=common.optimizer_sha256(h.model)
        merged,kwargs=merged_policy(h.model,g.model,r['gaussian_mean_coordinates'])
        parameters,after=outputs(merged,normalized)
        preserved_rows=[i for i in range(21) if i not in [7+c for c in r['gaussian_mean_coordinates']]]
        other=[c for c in range(7) if c not in r['gaussian_mean_coordinates']]
        if not torch.equal(parameters[:,preserved_rows],before_parameters[:,preserved_rows]):
            raise AssertionError('Merged actor changed preserved gate/mean/scale outputs numerically')
        if not np.array_equal(before[:,other],after[:,other]):raise AssertionError('Merged actor changed preserved action coordinates')
        if not torch.equal(parameters[:,[7+c for c in r['gaussian_mean_coordinates']]],gaussian_means[:,r['gaussian_mean_coordinates']]):
            raise AssertionError('Transferred Gaussian means differ numerically')
        h.model.policy,h.model.policy_kwargs=merged,kwargs
        if common.optimizer_sha256(h.model)!=old_optimizer:raise AssertionError('Empty PPO optimizer configuration changed')
        holdout=episodes%r['holdout_modulus']==r['holdout_residue']
        metrics={name:{'hybrid_parent':refit.fidelity(before,labels,mask),'gaussian_parent':refit.fidelity(gaussian,labels,mask),
            'merged':refit.fidelity(after,labels,mask)} for name,mask in [('repair_holdout',holdout),('all_training_rows',np.ones(len(obs),dtype=bool))]}
        if not all(metrics['repair_holdout']['merged']['on_mean_absolute_error'][j]<.9*metrics['repair_holdout']['hybrid_parent']['on_mean_absolute_error'][j] for j in r['gaussian_mean_coordinates']):
            raise ValueError('Transferred mean rows did not materially improve diagnostic fidelity')
        witness={'normalizer_statistics_identical_between_parents':True,'preserved_head_outputs_bitwise_equal':preserved_rows,
            'gaussian_mean_outputs_bitwise_equal':r['gaussian_mean_coordinates'],'preserved_action_coordinates_bitwise_equal':other,
            'exact_observation_witnesses':len(obs),'empty_PPO_optimizer_preserved':True,'all_parameters_trainable':all(p.requires_grad for p in merged.parameters()),
            'metrics':metrics}
        require_plan(root)
        if datetime.now(timezone.utc)>=common._deadline(r['deadline_utc']):raise TimeoutError('Transfer completed after deadline')
        record=refit.save_authenticated(h,root,plan,kind='algebraically-merged-neural-initialization',witness=witness)
        restored=hybrid.load_checkpoint(record['checkpoint']['path'])
        try:
            if not np.array_equal(restored.predict(obs,deterministic=True)[0],after):raise AssertionError('Merged actions changed on reopening')
        finally:restored.close()
        result={'status':'completed','elapsed_s':time.monotonic()-started,'checkpoint':record,'witness':witness}
        common.publish_json(root/'result.json',result);return result
    finally:h.close();g.close()

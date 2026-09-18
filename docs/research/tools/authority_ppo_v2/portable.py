"""Load and replay an admitted policy from the portable research package.

This checks relative source/checkpoint hashes, so the published artifact can be
moved without weakening its model/task identity or requiring original host paths.
Reproducing an already published case creates no new independent test evidence.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

SOURCE_ROOT = Path(__file__).resolve().parents[4]
for folder in (SOURCE_ROOT,SOURCE_ROOT/'backend/src',SOURCE_ROOT/'simulator/src'):
    if str(folder) not in sys.path:
        sys.path.insert(0,str(folder))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def confined(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or not (root / relative).resolve().is_relative_to(root.resolve()):
        raise ValueError('Artifact path escapes the portable bundle')
    return root / relative


def load_policy(bundle, arm, optimizer_seed, *, require_runtime_match=True):
    from docs.research.tools.authority_ppo_v2 import benchmark,policy
    from plant_backend.rl.v2.training.constrained_ppo import ConstrainedPPO,policy_parameter_sha256
    from docs.research.tools.authority_ppo_v2.train import actor_parameter_sha256,actor_mean_parameter_sha256
    from plant_backend.rl.v2.training.research_effort_batch import preflight_checkpoint
    bundle=Path(bundle).resolve()
    manifest=json.loads((bundle/'manifest.json').read_text())
    if manifest['schema']!='plant-authority-portable-bundle-v2':
        raise ValueError('Unknown portable bundle')
    pairs=[(r['arm'],r['optimizer_seed']) for r in manifest['endpoints']]
    if len(pairs)!=6 or set(pairs)!={(a,s) for a in ('narrow','broad') for s in (101,211,307)}:
        raise ValueError('Bundle must contain exactly six distinct admitted endpoints')
    for item in manifest['source_files']:
        path=confined(bundle/'source',item['path'])
        if sha(path)!=item['sha256']:
            raise ValueError(f'Source differs: {item["path"]}')
    if SOURCE_ROOT != bundle/'source':
        raise ValueError('Run this loader from the source directory inside the bundle')
    mismatches={}
    for name,version in manifest['runtime_packages'].items():
        try:
            actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual=None
        if actual!=version:
            mismatches[name]={'recorded':version,'current':actual}
    if mismatches and require_runtime_match:
        raise ValueError(f'Runtime differs: {mismatches}')
    row=next((r for r in manifest['endpoints'] if r['arm']==arm and r['optimizer_seed']==optimizer_seed),None)
    if row is None:
        raise ValueError('Policy is outside the six admitted endpoints')
    checkpoint=confined(bundle,row['checkpoint'])
    metadata_path=confined(bundle,row['metadata'])
    if sha(checkpoint)!=row['checkpoint_sha256'] or sha(metadata_path)!=row['metadata_sha256']:
        raise ValueError('Checkpoint or endpoint metadata differs')
    result=json.loads(metadata_path.read_text())
    if result['arm']!=arm or result['optimizer_seed']!=optimizer_seed:
        raise ValueError('Endpoint metadata belongs to a different arm or optimizer seed')
    preflight_checkpoint(checkpoint,result['policy_identity'],result['constraint_state'])
    model=ConstrainedPPO.load(checkpoint,device='cpu')
    if (model.num_timesteps!=262144 or model.seed!=optimizer_seed
        or policy_parameter_sha256(model)!=result['final_tensor_sha256']
        or actor_parameter_sha256(model)!=result['final_actor_sha256']
        or actor_mean_parameter_sha256(model)!=result['final_actor_mean_sha256']
        or model.embedded_constraint_state!=result['constraint_state']):
        raise ValueError('Loaded endpoint tensors or transition count differ')
    env=benchmark.BenchmarkEnv(split='held_out')
    try:
        expected=policy.interface_identity(env,arm)
        if model.research_identity!=expected:
            raise ValueError('Policy/environment interface differs')
    finally:
        env.close()
    return model,{'checkpoint_sha256':row['checkpoint_sha256'],'runtime_mismatches':mismatches}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--arm',choices=['narrow','broad'],required=True)
    parser.add_argument('--optimizer-seed',type=int,choices=[101,211,307],required=True)
    parser.add_argument('--scenario',type=int,default=9960000)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-runtime-difference',action='store_true')
    args=parser.parse_args()
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model,provenance=load_policy(args.bundle,args.arm,args.optimizer_seed,
                                 require_runtime_match=not args.allow_runtime_difference)
    from docs.research.tools.authority_ppo_v2.benchmark import BenchmarkEnv
    from docs.research.tools.authority_ppo_v2.policy import AuthorityPPOController
    from docs.research.tools.feasible_mission_replay import run_controller
    env=BenchmarkEnv('held_out')
    try:
        env.reset(seed=args.scenario)
        result=run_controller(env,AuthorityPPOController(env,model,args.arm))
        result.update(reproduction_of_published_case=True,portable_provenance=provenance,
                      independent_new_test_evidence=False)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('xb') as stream:
            stream.write(gzip.compress(json.dumps(result,separators=(',',':'),allow_nan=False).encode(),mtime=0))
        print(json.dumps({k:result[k] for k in ('mission_passed','hard_safety_passed','every_delivery_passed','all_daily_returns_passed','minute_truth_sha256')},indent=2))
    finally:
        env.close()


if __name__=='__main__':
    main()

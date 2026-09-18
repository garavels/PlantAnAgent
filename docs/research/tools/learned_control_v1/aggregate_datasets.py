"""Aggregate authenticated train-only demonstrations without dropping rare actions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from docs.research.tools.learned_control_v1 import imitation, training


def aggregate(root, sources):
    root = Path(root).resolve()
    sources = [Path(p).resolve() for p in sources]
    if len(sources) < 2 or len(set(sources)) != len(sources):
        raise ValueError('Aggregation requires at least two distinct authenticated datasets')
    metadata = [imitation.require_dataset(p) for p in sources]
    teacher = metadata[0]['teacher_parameters']
    if any(m['teacher_parameters'] != teacher for m in metadata):
        raise ValueError('A DAgger aggregate must use the same teacher configuration')
    cases = [case for m in metadata for case in m['cases']]
    identities = [(c['condition'], c['seed']) for c in cases]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate demonstration scenarios would silently reweight the dataset')
    if any(not 50_000_000 <= c['seed'] < 51_000_000 for c in cases):
        raise ValueError('Every aggregated scenario must be in the training domain')
    root.mkdir(parents=True, exist_ok=False)
    source_files = {p['path']:p for m in metadata for p in m['source_files']}
    own = training.file_record(Path(__file__).resolve())
    source_files[own['path']] = own
    plan = {'schema': imitation.SCHEMA+'-aggregation-plan', 'created_at': training.now(),
        'split':'train', 'cases':cases, 'teacher_parameters':teacher,
        'input_datasets':[{'dataset':training.file_record(p),
                           'metadata':training.file_record(p.with_suffix('.json'))} for p in sources],
        'source_files':sorted(source_files.values(), key=lambda x:x['path']),
        'method':'Concatenate all rows without sampling; offset episode indices; preserve behavior flags and actual trajectory returns'}
    training.publish_json(root/'collection-plan.json',plan)
    arrays = {k:[] for k in ('observations','actions','rewards','returns','episode_indices','behavior_teacher')}
    offset = 0
    for source, m in zip(sources,metadata):
        with np.load(source,allow_pickle=False) as data:
            episode_ids = data['episode_indices']
            if set(episode_ids.tolist()) != set(range(m['episodes'])):
                raise ValueError('Source episode indices do not match its authenticated ledger')
            for key in arrays:
                value = data[key].copy()
                arrays[key].append(value+offset if key == 'episode_indices' else value)
        offset += m['episodes']
    path = root/'demonstrations.npz'
    with path.open('xb') as stream:
        np.savez_compressed(stream,**{k:np.concatenate(v) for k,v in arrays.items()})
    result = {'schema':imitation.SCHEMA, 'completed_at':training.now(),
        'dataset':training.file_record(path), 'plan':training.file_record(root/'collection-plan.json'),
        'plan_sha256':training.sha(plan), 'source_files':plan['source_files'],
        'samples':sum(m['samples'] for m in metadata), 'episodes':offset,
        'observation_shape':[149], 'action_shape':[7], 'teacher_parameters':teacher,
        'cases':cases, 'summary':[row for m in metadata for row in m['summary']],
        'aggregation':{'source_datasets':plan['input_datasets'],'new_simulations':0}}
    training.publish_json(path.with_suffix('.json'),result)
    imitation.require_dataset(path)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--sources',type=Path,nargs='+',required=True)
    args=parser.parse_args()
    print(json.dumps(aggregate(args.root,args.sources)),flush=True)


if __name__=='__main__':
    main()

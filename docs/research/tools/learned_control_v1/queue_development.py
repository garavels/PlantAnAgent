"""Wait for named immutable checkpoints, then run a prescribed development panel."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from docs.research.tools.learned_control_v1 import evaluate, analyze


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controllers',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--seeds',type=int,nargs='+',default=list(evaluate.DEVELOPMENT_SEEDS[:4]))
    parser.add_argument('--conditions',nargs='+',default=list(evaluate.CONDITIONS))
    parser.add_argument('--source',type=Path,action='append',default=[])
    parser.add_argument('--deadline-utc',default='2026-09-18T05:45:00+00:00')
    args=parser.parse_args()
    controllers=json.loads(args.controllers.read_text())
    deadline=datetime.fromisoformat(args.deadline_utc)
    dependencies=[Path(c['checkpoint']).with_suffix('.json') for c in controllers if c['kind']!='baseline']
    while not all(p.exists() for p in dependencies):
        if datetime.now(timezone.utc)>=deadline:
            raise TimeoutError('A prescribed development checkpoint did not arrive')
        time.sleep(10)
    spec=evaluate.freeze_panel(args.output/'freeze.json',controllers,args.seeds,args.conditions,
        source_paths=[args.controllers,*args.source],deadline_utc=args.deadline_utc)
    summary=evaluate.campaign(args.output,workers=args.workers)
    result=analyze.analyze_rows(summary['rows'],spec)
    evaluate.publish(args.output/'analysis.json',result)
    for row in result['summaries']:
        if row['panel'] in ('mixture','ordinary','stress'):
            print(json.dumps({'population':row['population'],'panel':row['panel'],
                'contribution':row['metrics']['net_operating_value_eur']['mean'],
                'unmet':row['metrics']['unmet_kg']['mean'],
                'success_rate':row['metrics']['all_delivery_success']['mean']}),flush=True)


if __name__=='__main__':
    main()

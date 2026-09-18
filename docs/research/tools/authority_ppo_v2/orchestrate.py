"""Host-side source snapshot and bounded, three-process training supervision."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[4]


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def snapshot(destination, output):
    destination, output = Path(destination).resolve(), Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=True)
    selected = set()
    for folder in ('backend/src', 'simulator/src', 'docs/research/tools'):
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and path.suffix in {'.py', '.json', '.toml'} and '__pycache__' not in path.parts:
                selected.add(path)
    for relative in ('uv.lock', 'pyproject.toml', 'backend/pyproject.toml', 'simulator/pyproject.toml',
                     'docs/research/benchmarks/feasible25_offgrid_v1.json',
                     'docs/research/benchmarks/authority_ppo_v2.json',
                     'docs/research/COVERAGE_PPO_V1_PROTOCOL.md',
                     'docs/research/AUTHORITY_PPO_V2_PROTOCOL.md'):
        selected.add(ROOT / relative)
    records = []
    for source in sorted(selected):
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        target.write_bytes(data)
        records.append({'path': str(relative), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    archive = output / 'source-initial-snapshot.tar.gz'
    with tarfile.open(archive, 'x:gz') as tar:
        for record in records:
            tar.add(destination / record['path'], arcname=record['path'], recursive=False)
    with tarfile.open(archive, 'r:gz') as tar:
        for record in records:
            assert hashlib.sha256(tar.extractfile(record['path']).read()).hexdigest() == record['sha256']
    write(output / 'source-initial-snapshot.json', {
        'schema':'authority-source-snapshot-v1', 'created_at':utc(), 'root':str(destination),
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'source_files':records, 'archive':str(archive),
        'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'note':'Training executes this copied source with explicit PYTHONPATH; unrelated workspace edits cannot change it. Evaluator may be added before its later separate freeze.'})
    (output / 'working-tree-at-snapshot.patch').write_bytes(subprocess.check_output(['git','diff','--binary'], cwd=ROOT))
    return {'snapshot':str(destination), 'files':len(records), 'archive':str(archive)}


def train_waves(source, training_root, output, deadline):
    source, training_root, output = map(lambda p:Path(p).resolve(), (source, training_root, output))
    output.mkdir(parents=True, exist_ok=True)
    deadline_s = datetime.fromisoformat(deadline.replace('Z','+00:00')).timestamp()
    python = ROOT / '.venv/bin/python'
    script = source / 'docs/research/tools/authority_ppo_v2/train.py'
    environment = dict(os.environ)
    environment.update(PYTHONPATH=os.pathsep.join(map(str,(source,source/'backend/src',source/'simulator/src'))),
                       OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                       VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    records = []
    for arm in ('narrow','broad'):
        children = []
        for seed in (101,211,307):
            if time.time() >= deadline_s:
                raise TimeoutError('No remaining training time')
            log_path = output/f'{arm}-{seed}.log'
            log = log_path.open('xb')
            cmd = [str(python),str(script),'train','--root',str(training_root),'--arm',arm,'--optimizer-seed',str(seed)]
            process = subprocess.Popen(cmd,cwd=source,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record = {'arm':arm,'optimizer_seed':seed,'pid':process.pid,'started_at':utc(),'command':cmd,'log':str(log_path),'exit_code':None}
            records.append(record); children.append((process,log,record))
            print(json.dumps({'event':'launched',**record}),flush=True)
        while any(p.poll() is None for p,_,_ in children):
            if time.time() > deadline_s + 60:
                for process,_,record in children:
                    if process.poll() is None:
                        os.killpg(process.pid,signal.SIGTERM)
                        record['supervisor_terminated_at']=utc()
            if time.time() > deadline_s + 120:
                for process,_,_ in children:
                    if process.poll() is None:
                        os.killpg(process.pid,signal.SIGKILL)
            for process,_,record in children:
                record['exit_code']=process.poll()
                progress=training_root/f'{record["arm"]}-seed-{record["optimizer_seed"]}'/'progress.jsonl'
                if progress.exists():
                    content=progress.read_text()
                    lines=content.splitlines()
                    if not content.endswith('\n'):
                        lines=lines[:-1]
                    if lines:
                        record['latest_progress']=json.loads(lines[-1])
            write(output/'supervision.json',{'schema':'authority-training-supervision-v1','updated_at':utc(),'deadline_utc':deadline,'processes':records})
            time.sleep(10)
        for process,log,record in children:
            record.update(exit_code=process.wait(),finished_at=utc()); log.close()
            print(json.dumps({'event':'closed',**record}),flush=True)
        write(output/'supervision.json',{'schema':'authority-training-supervision-v1','updated_at':utc(),'deadline_utc':deadline,'processes':records})
        if any(record['exit_code'] != 0 for _,_,record in children):
            raise RuntimeError(f'{arm} wave contains a failed attempt; preserved for review, not retried')
    return {'completed':True,'processes':records}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['snapshot','train-waves'])
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--training-root',type=Path)
    parser.add_argument('--deadline',default='2026-09-17T05:26:17Z')
    a=parser.parse_args()
    result=snapshot(a.source,a.output) if a.command=='snapshot' else train_waves(a.source,a.training_root,a.output,a.deadline)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()

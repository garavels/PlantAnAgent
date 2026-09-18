"""Package the final paper and complete commission evidence without altering releases."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile
from docs.research.tools.wholeplant_ppo_v1.portable import file_pin, digest

ROOT=Path(__file__).resolve().parents[4]
RESULTS=ROOT/'docs/research/results/tail_stress_20260918'
NAME='plantanagent-commission-final'

def source_archive(output):
    source=RESULTS/'writeup'
    paths=sorted(p for p in source.rglob('*') if p.is_file() and not any(x in {'.build','.layout','qa','__pycache__'} for x in p.relative_to(source).parts) and p.suffix not in {'.aux','.log','.out'})
    files=[{'path':p.relative_to(source).as_posix(),**{k:v for k,v in file_pin(p).items() if k!='path'}} for p in paths]
    manifest={'created_at':datetime.now(timezone.utc).isoformat(),'files':files,'build':f'pdflatex -no-shell-escape {NAME}.tex (twice)'}
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in paths: archive.write(p,p.relative_to(source).as_posix())
        archive.writestr('SOURCE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with tempfile.TemporaryDirectory(prefix='commission-paper-check-') as tmp:
        extracted=Path(tmp)
        with zipfile.ZipFile(output) as archive:
            if archive.testzip(): raise ValueError('Source archive CRC failed')
            archive.extractall(extracted)
        for row in files:
            pin=file_pin(extracted/row['path'])
            if pin['sha256']!=row['sha256'] or pin['bytes']!=row['bytes']: raise ValueError('Extracted source differs')
        for _ in range(2):
            built=subprocess.run(['/Library/TeX/texbin/pdflatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error',NAME+'.tex'],cwd=extracted,capture_output=True,text=True,timeout=45)
            if built.returncode: raise RuntimeError(built.stdout[-6000:])
        log=(extracted/(NAME+'.log')).read_text()
        if any(s in log for s in ('Overfull','undefined','multiply defined')): raise ValueError('Standalone LaTeX warnings')
        info=subprocess.check_output(['/opt/homebrew/bin/pdfinfo',str(extracted/(NAME+'.pdf'))],text=True)
        if not any(line.split()==['Pages:','2'] for line in info.splitlines()): raise ValueError('Paper must be two pages')
        texts=[subprocess.check_output(['/opt/homebrew/bin/pdftotext',str(p),'-'],text=True) for p in (extracted/(NAME+'.pdf'),source/(NAME+'.pdf'))]
        if ' '.join(texts[0].split())!=' '.join(texts[1].split()): raise ValueError('Standalone paper text differs')
    return {'archive':file_pin(output),'files':len(files),'standalone_compilation':True,'pages':2,'text_matches_reviewed_pdf':True}

def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip(): raise ValueError('Evidence CRC failed')
        manifest=json.loads(archive.read('commission-manifest.json'))
        if digest({k:v for k,v in manifest.items() if k!='sha256'})!=manifest['sha256']: raise ValueError('Manifest digest failed')
        if len(archive.namelist())!=len(set(archive.namelist())): raise ValueError('Duplicate archive members')
        if set(archive.namelist())!={r['path'] for r in manifest['files']}|{'commission-manifest.json'}:
            raise ValueError('Archive contains missing or unmanifested members')
        for row in manifest['files']:
            data=archive.read(row['path'])
            if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']: raise ValueError('Evidence changed: '+row['path'])
        for pin in manifest['scientific_bindings']:
            mapped=next((r for r in manifest['files'] if r.get('original_path')==pin['path']),None)
            if mapped is None or any(mapped[k]!=pin[k] for k in ('bytes','sha256')): raise ValueError('Missing immutable evidence binding: '+pin['path'])
    return manifest

def evidence_archive(output):
    from docs.research.tools.tail_stress_v1.run import require_freeze
    freeze=require_freeze(RESULTS/'freeze.json')
    summary=json.loads((RESULTS/'summary.json').read_text())
    analysis=json.loads((RESULTS/'analysis/analysis.json').read_text())
    if summary['case_count']!=480 or not summary['all_executions_and_audits_passed'] or not analysis['audits']['all_rollouts_complete_and_audited']:
        raise ValueError('Incomplete tail study cannot be released')
    old_path=ROOT/'output/research/plantanagent-wholeplant-ppo-research-package.zip'
    paths=set()
    for folder in ('docs/research/tools/startup_stress_v1','docs/research/tools/tail_stress_v1','docs/research/results/startup_stress_20260917','docs/research/results/tail_stress_20260918'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and not p.is_symlink() and not any(s in {'__pycache__','.build','.layout','qa'} for s in p.relative_to(ROOT).parts) and p.suffix not in {'.pyc','.aux','.log','.out'} and p.name!='package-receipt.json' and p!=RESULTS/'completion.json':
                paths.add(p)
    for relative in ('docs/research/TAIL_STRESS_V1_PROTOCOL.md','docs/research/STARTUP_STRESS_V1_PROTOCOL.md','tests/test_startup_stress_environment.py','tests/test_tail_stress_environment.py','tests/test_tail_stress_audit.py','tests/test_tail_stress_analysis.py','output/pdf/'+NAME+'.pdf','output/research/'+NAME+'-latex-source.zip'):
        paths.add(ROOT/relative)
    if not (RESULTS/'postflight/review.json').exists(): raise ValueError('Independent final verification missing')
    review=json.loads((RESULTS/'postflight/review.json').read_text())
    if review.get('status')!='passed' or not all(review.get(k) is True for k in ('all_sources_unchanged','all480_case_artifacts_verified','exact_replays_passed','prefixes_match_paired_control')):
        raise ValueError('Independent final verification did not pass every required check')
    if not (RESULTS/'inspector.html').exists(): raise ValueError('Dynamic inspector missing')
    added={p.relative_to(ROOT).as_posix():p for p in paths}
    files=[]
    with zipfile.ZipFile(old_path) as previous, zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as target:
        old_manifest=json.loads(previous.read('bundle-manifest.json'))
        old_entries={r['path']:r for r in old_manifest['files']}
        for member in previous.namelist():
            if member in added: raise ValueError('Refusing to overwrite prior release member '+member)
            data=previous.read(member)
            target.writestr(member,data,compress_type=zipfile.ZIP_STORED if member.endswith(('.gz','.zip')) else zipfile.ZIP_DEFLATED)
            row={'path':member,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            if member in old_entries and old_entries[member].get('original_path'): row['original_path']=old_entries[member]['original_path']
            files.append(row)
        for name,p in sorted(added.items()):
            pin=file_pin(p); data=p.read_bytes()
            target.writestr(name,data,compress_type=zipfile.ZIP_STORED if name.endswith(('.gz','.zip')) else zipfile.ZIP_DEFLATED)
            files.append({'path':name,'original_path':pin['path'],'bytes':pin['bytes'],'sha256':pin['sha256']})
        readme='''# PlantAnAgent — final commission review package

Start with output/pdf/plantanagent-commission-final.pdf (two pages).
Editable LaTeX and a separately verified source archive are under output/research.
The final manuscript source is docs/research/results/tail_stress_20260918/writeup.
Open docs/research/results/tail_stress_20260918/inspector.html in a browser for
offline dynamic responses. It shows the first prescribed scenario, not a selected win;
the tables and analysis retain all scenarios. No web server or network is needed.

This package contains the unchanged prior whole-plant release (496 outcomes),
the constructed startup diagnostic (12 outcomes), and the endogenous-reserve tail
study (480 outcomes), all sources, policies, normalizers, inputs/actions and audits.
No files, results or models from the older release are silently replaced.

Tail physical traces exist for the first seed in each of 30 panels. Every minute
of every rollout was audited; actual inputs and actions are retained for all480.
Read docs/research/results/tail_stress_20260918/INTERPRETATION.md and its
analysis/REPORT.md for limitations.
The stress mix is artificial and cannot estimate real-world event frequencies.

Use Python3.12 and requirements-replay.txt (dependencies are not bundled). From
the extracted root, the older portable verifier still authenticates its unchanged
release: python -m docs.research.tools.wholeplant_ppo_v1.portable verify
The final package has commission-manifest.json with relative member hashes and
original absolute paths preserved as provenance. All frozen paths can be mapped
through this manifest without editing the underlying research records.

Verify the complete extracted package in a fresh Python process from its root:
python -m docs.research.tools.tail_stress_v1.verify_release manifest --root .
Replay all30 retained tail traces into a new receipt directory:
python -m docs.research.tools.tail_stress_v1.verify replay --output /tmp/plantanagent-tail-replay --max-seconds 600
The output directory must be new so earlier receipts are preserved. Exact replay
was checked under the captured runtime; different hardware may differ numerically.

This is a local review package. It has not been published or submitted. The
commission also asks for a public repository/design package and a short demo.
The included offline inspector supplies a working dynamic inspection artifact;
public hosting and submission remain the authors' final external actions.
'''.encode()
        target.writestr('README-COMMISSION.md',readme)
        files.append({'path':'README-COMMISSION.md','bytes':len(readme),'sha256':hashlib.sha256(readme).hexdigest()})
        startup=json.loads((ROOT/'docs/research/results/startup_stress_20260917/freeze.json').read_text())
        bindings=[*freeze['sources'],*startup['sources'],freeze['original_evaluation'],file_pin(RESULTS/'freeze.json')]
        bindings += [r['recording'] for r in summary['rows']]
        for c in freeze['controllers'].values(): bindings+=list(c.get('checkpoint',{}).values())
        manifest={'schema':'plantanagent-final-commission-package-v1','created_at':datetime.now(timezone.utc).isoformat(),
            'prior_release_archive':file_pin(old_path),'files':sorted(files,key=lambda x:x['path']),
            'scientific_bindings':list({r['path']:r for r in bindings}.values()),
            'studies':{'ordinary_cases':496,'startup_cases':12,'tail_cases':480},'local_review_only':True}
        manifest['sha256']=digest(manifest)
        target.writestr('commission-manifest.json',json.dumps(manifest,indent=2)+'\n')
    checked=verify_archive(output)
    return {'archive':file_pin(output),'manifest_sha256':checked['sha256'],'files':len(files),'all_member_hashes_verified':True,'all_scientific_bindings_verified':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=('paper','evidence','verify'));p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    result=source_archive(args.output) if args.kind=='paper' else evidence_archive(args.output) if args.kind=='evidence' else {'verified':True,'manifest_sha256':verify_archive(args.output)['sha256']}
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()

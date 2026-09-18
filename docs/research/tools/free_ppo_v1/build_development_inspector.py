"""Build a clearly labeled interim development snapshot; never a final viewer.

Requires all eight development scenarios for the three fixed checkpoints of
each optimizer, plus both references. No model or simulator is imported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from docs.research.tools.free_ppo_v1 import build_inspector as base

STEPS = (0, 65536, 393216)
SEEDS = (101, 211, 307)
SCENARIOS = list(range(10100000, 10100008))
DETAILS = ("coverage-direct", "untrained-101", *(f"free-ppo-v1-{s}-step-393216" for s in SEEDS))


def assemble(analysis_path, freeze_path):
    analysis = json.loads(Path(analysis_path).read_text())
    freeze = json.loads(Path(freeze_path).read_text())
    if (freeze.get("schema") != base.SCHEMA + "-freeze" or
        base.canonical_sha({k: v for k, v in freeze.items() if k != "sha256"}) != freeze.get("sha256")):
        raise ValueError("Evaluation freeze changed")
    if analysis.get("schema") != base.SCHEMA + "-analysis" or analysis["evaluation_freeze_sha256"] != freeze["sha256"]:
        raise ValueError("Analysis comparison identity differs")
    if any(r["split"] != "development" for r in analysis["rows"]):
        raise ValueError("This snapshot requires development outcomes only")
    if freeze["selection"]["development_seeds"] != SCENARIOS or freeze["selection"]["final_timesteps"] != 786432:
        raise ValueError("Unexpected prospective development design")
    if any(analysis["evaluation_freeze"][k] != base.pin(freeze_path)[k] for k in ("bytes", "sha256")):
        raise ValueError("Analysis freeze pin differs")
    for p in analysis["summaries"]:
        if base.pin(p["path"]) != p:
            raise ValueError("Canonical development panel changed after analysis")
    expected = {"coverage-direct", "untrained-101", *(f"free-ppo-v1-{s}-step-{n}" for s in SEEDS for n in STEPS)}
    if set(g["controller"] for g in analysis["groups"]) != expected or len(analysis["groups"]) != 11:
        raise ValueError("Expected all eleven fixed development groups")
    records, groups, names = {}, {}, {}
    for g in analysis["groups"]:
        label = g["controller"]
        rows = [r for r in analysis["rows"] if r["controller"] == label]
        if g["split"] != "development" or g["n_scenarios"] != 8 or sorted(r["seed"] for r in rows) != SCENARIOS:
            raise ValueError("Missing or duplicate development scenario")
        steps = int(label.rsplit("-step-", 1)[1]) if "-step-" in label else 0
        for row in rows:
            if (not row["execution_complete"] or row["decisions"] != 672 or row["physical_steps"] != 10080 or
                row["checkpoint_timesteps"] != steps or row["evaluation_freeze_sha256"] != freeze["sha256"] or row.get("endpoint_freeze_sha256") is not None):
                raise ValueError("Unexpected complete development checkpoint identity")
            record_path = base.resolve_record(row["recording"])
            if row["seed"] == SCENARIOS[0] and label in DETAILS:
                records[label] = base.compact_record(record_path, expected_row=row)
        if (g["complete_weeks"] != 8 or g["delivery_successes"] != sum(r["every_delivery_passed"] for r in rows)
            or g["delivery_and_physics_successes"] != sum(r["delivery_and_physics_passed"] for r in rows)):
            raise ValueError("Group counts disagree with canonical cases")
        groups[label] = g
        if label == "coverage-direct": names[label] = "Translated coverage reference"
        elif label == "untrained-101": names[label] = "Untrained network · seed 101"
        else:
            seed = label.split("-")[3]
            names[label] = f"PPO · seed {seed} · {steps:,} steps"
    if set(records) != set(DETAILS):
        raise ValueError("Missing prospectively selected first-scenario traces")
    identities = {(r["summary"]["exogenous_episode_sha256"], r["summary"]["manifest_sha256"]) for r in records.values()}
    if len(identities) != 1:
        raise ValueError("Selected controllers faced different scenarios")
    curves = analysis["learning_curves"]
    if sorted(c["controller"] for c in curves) != [f"free-ppo-v1-{s}" for s in SEEDS]:
        raise ValueError("Missing optimizer learning curve")
    for curve in curves:
        if [p["transitions"] for p in curve["points"]] != list(STEPS) or any(p["n_scenarios"] != 8 for p in curve["points"]):
            raise ValueError("Expected three actual fixed checkpoints and a pending final endpoint")
        for point in curve["points"]:
            for metric in ("net_operating_value_eur", "every_delivery_passed"):
                if point["metrics"][metric] != groups[point["controller"]]["metrics"][metric]["all_case"]:
                    raise ValueError("Learning curve and canonical group differ")
    return {"schema": "free-ppo-v1-interim-development-inspector", "fixture": False, "interim": True,
            "scenario": SCENARIOS[0], "default": "free-ppo-v1-101-step-393216", "names": names,
            "fields": [*base.FIELDS, "dac_offline"], "records": {k: records[k] for k in DETAILS},
            "groups": [groups[k] for k in DETAILS], "all_groups": list(groups.values()), "learning_curves": curves,
            "final_steps": 786432, "provenance": {"builder": base.pin(__file__), "shared_viewer": base.pin(base.__file__),
            "analysis": base.pin(analysis_path), "evaluation_freeze": base.pin(freeze_path),
            "evaluation_freeze_sha256": freeze["sha256"], "created_at": datetime.now(timezone.utc).isoformat(),
            "bootstrap": analysis["bootstrap"], "interpretation": "Interim development snapshot. All final endpoints are pending here; no held-out outcomes are included. Development scenarios may inform exploratory interpretation but never endpoint selection."}}


LEARNING = '''<section class="section"><h2>Learning progress on eight development weeks</h2><p class="note">All three optimizer seeds are shown at their original, 65,536-step and 393,216-step checkpoints. No intermediate policy replaces the fixed final endpoint. Shading marks training beyond the latest evaluated checkpoint; final results are pending.</p><div class="two"><section class="plot"><h3>All 28 deliveries met / % of weeks</h3><svg id="learning-success" role="img" aria-label="Actual development delivery success at three training checkpoints; final checkpoint pending"></svg></section><section class="plot"><h3>Mean modeled operating value / EUR per week</h3><svg id="learning-value" role="img" aria-label="Actual development operating value at three training checkpoints; final checkpoint pending"></svg></section></div><div class="legend"><span style="--c:#187f83">Seed 101</span><span style="--c:#466db1">Seed 211</span><span style="--c:#a3453d">Seed 307</span></div><div class="table-wrap"><table><thead><tr><th>Optimizer seed</th><th>Training steps</th><th>All deliveries met</th><th>Mean unmet kg</th><th>Mean net value EUR</th></tr></thead><tbody id="learning-rows"></tbody></table></div><p class="note">The translated reference changes a power target into a fixed solar share once every 15 minutes. When sunlight falls, this can interrupt startup. Its outcome does not establish the performance of the original reference with minute-by-minute allocation, or of an adapted rule-based controller.</p></section>'''

LEARNING_JS = r'''
function drawLearning(){const colors=['#187f83','#466db1','#a3453d'];for(const [id,metric,scale]of [['learning-success','every_delivery_passed',100],['learning-value','net_operating_value_eur',1]]){const el=$(id),W=Math.max(290,Math.round(el.clientWidth)),H=245,L=65,R=22,T=26,B=38,w=W-L-R,h=H-T-B,points=data.learning_curves.flatMap(c=>c.points.map(p=>p.metrics[metric].mean*scale)),lo=Math.min(0,...points),hi=Math.max(metric==='every_delivery_passed'?100:1,...points),pad=(hi-lo)*.1||1,x=t=>L+t/data.final_steps*w,y=v=>T+(hi+pad-v)/(hi+pad-lo)*h;el.setAttribute('viewBox',`0 0 ${W} ${H}`);let svg=`<rect x="${x(393216)}" y="${T}" width="${x(786432)-x(393216)}" height="${h}" fill="#edf1f4"/><text x="${x(589824)}" y="${T+14}" text-anchor="middle" font-size="12" fill="#516675">Final pending</text>`;for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4;svg+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="#e1e8ed"/><text x="${L-7}" y="${y(v)+4}" font-size="11" text-anchor="end" fill="#516675">${fmt(v,0)}</text>`}for(const t of [0,393216,786432])svg+=`<text x="${x(t)}" y="${H-17}" text-anchor="${t===786432?'end':'middle'}" font-size="11" fill="#516675">${fmt(t,0)}</text>`;data.learning_curves.forEach((c,i)=>{svg+=`<polyline fill="none" stroke="${colors[i]}" stroke-width="2" points="${c.points.map(p=>`${x(p.transitions)},${y(p.metrics[metric].mean*scale)}`).join(' ')}"/>`;for(const p of c.points)svg+=`<circle cx="${x(p.transitions)}" cy="${y(p.metrics[metric].mean*scale)}" r="3.6" fill="${colors[i]}"><title>Seed ${c.controller.split('-').at(-1)}; ${fmt(p.transitions,0)} steps: ${fmt(p.metrics[metric].mean*scale,2)}</title></circle>`});el.innerHTML=svg}table('learning-rows',data.learning_curves.flatMap(c=>[...c.points.map(p=>[c.controller.split('-').at(-1),fmt(p.transitions,0),`${fmt(p.metrics.every_delivery_passed.mean*8,0)} / 8`,fmt(p.metrics.unmet_kg.mean),fmt(p.metrics.net_operating_value_eur.mean,0)]),[c.controller.split('-').at(-1),fmt(data.final_steps,0),'Pending','Pending','Pending']]));}
drawLearning();window.addEventListener('resize',drawLearning);
'''


def document(data):
    html = base.HTML
    def replace(old, new):
        nonlocal html
        if html.count(old) != 1:
            raise ValueError(f"Shared viewer template changed: {old[:70]}")
        html = html.replace(old, new, 1)
    replace('<title>Direct plant control — recorded experiment</title>', '<title>Interim development results — Plant An Agent</title>')
    replace('<body><a class="skip"', '<body><div style="position:sticky;top:0;z-index:30;background:#f8e7c5;color:#5c421b;border-bottom:2px solid #ad781d;padding:12px 20px;font-weight:650;line-height:1.45" role="note">Interim development results — training in progress; final test set unopened.</div><a class="skip"')
    replace('From sunlight to methane,<br>one decision at a time.', 'What has the policy learned so far?')
    replace('const pretty=k=>k===\'coverage-direct\'?\'Coverage baseline\':k===\'untrained-101\'?\'Untrained network (101)\':`PPO · seed ${k.split(\'-\').at(-1)}`;', 'const pretty=k=>data.names[k]||k;')
    replace('`Scenario ${data.scenario} · the first prespecified held-out case. All five controllers share the same weather and faults.`', '`Development scenario ${data.scenario} · first selected scenario. Snapshot ${data.provenance.created_at}. All five controllers share the same weather and faults.`')
    replace('<section class="controls"', LEARNING + '<section class="controls"')
    replace('The full held-out comparison', 'The current eight-week development comparison')
    replace('Each controller faces the same 32 environmental scenarios.', 'Each displayed controller faces the same eight development scenarios. PPO rows use the midpoint checkpoint; final results are pending.')
    replace('The scenario was chosen by its index before test outcomes. The default display is PPO seed 101, fixed in advance; it is not selected for performance.', 'This is the first development scenario by its prescribed index. The default is midpoint PPO seed 101, not the best-performing seed.')
    replace("</script></body></html>", LEARNING_JS + "</script></body></html>")
    return html.replace('__DATA__', json.dumps(data, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--evaluation-freeze', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = assemble(args.analysis, args.evaluation_freeze)
    html = document(data)
    sidecar = args.output.with_suffix('.build.json')
    readme = args.output.parent / 'README.md'
    if any(p.exists() for p in (args.output, sidecar, readme)):
        raise FileExistsError('Interim output already exists; preserve previous snapshots')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream: stream.write(html)
    with sidecar.open('x') as stream:
        json.dump({'schema': data['schema']+'-build', **data['provenance'], 'html': base.pin(args.output),
                   'recordings': {k:r['recording'] for k,r in data['records'].items()}}, stream, indent=2)
        stream.write('\n')
    with readme.open('x') as stream:
        stream.write('# Interim development inspection\n\nOpen `inspector.html` in a browser; it works entirely offline. This is a saved development snapshot, not the final test result. Training was still in progress and the final test set was unopened when this snapshot was built.\n\nThe viewer includes all three seeds at 0, 65,536 and 393,216 steps on the same eight development scenarios. Final 786,432-step results are explicitly pending. Detailed minute traces show the first prescribed development scenario, 10100000, for the three midpoint policies, the untrained seed-101 policy and the translated coverage reference. Default selection is midpoint seed 101.\n\nUse the selector and timeline to inspect requested versus applied power, inventories, delivery settlements, wear and modeled economic contribution. Daily inventory return is descriptive only. The translated reference uses a 15-minute fixed solar share; its outcome does not demonstrate superiority over the original minute-by-minute reference.\n\nAll recording pins, selected minute hashes, comparison identities, checkpoint budgets and complete eight-scenario panels were verified. No policy or simulator ran to build this page. Build provenance is in `inspector.build.json`; full results remain in `analysis/analysis.json`. The separate final held-out inspector has not been relaxed or replaced.\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()

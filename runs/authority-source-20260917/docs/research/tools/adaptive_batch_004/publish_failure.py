"""Archive closed failure evidence without admitting a performance result."""

import hashlib
import json
from pathlib import Path

ROOT = Path('/Volumes/Lexar/PlantAnAgentResearch')
OUT = Path('/Users/filippogombac/Developer/GitHub/PlantAnAgent/docs/research/results')


def record(path):
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def check(path, expected):
    actual = record(path)
    assert actual['sha256'] == expected['sha256'], path
    if 'bytes' in expected:
        assert actual['bytes'] == expected['bytes'], path
    return actual


inventory_path = ROOT / 'adaptive-rl-batch-004-analysis/abort-inventory.json'
check(inventory_path, {'sha256': '43da491388c802597866703dc1030b0dacee30429e9c7de90a0b91f20dd9b017'})
inventory = json.loads(inventory_path.read_text())
assert inventory['performance_admitted'] is False
assert inventory['policy_endpoints'] == inventory['evaluations_attempted'] == 0
assert inventory['completed_scalar_optimizer_rounds'] == 0
for item in inventory['files'] + inventory['macos_metadata_files']:
    check(Path(item['path']), item)

manifest_paths = [
    ROOT / 'mpc-constraint-analysis-tools-4f908fc/manifest.json',
    ROOT / 'adaptive-rl-batch-004-analysis/forecast-diagnosis/manifest.json',
    ROOT / 'supervision-exfat-probe-004/manifest.json',
]
manifest_digests = [
    '514d4c54482e864b45538361d94a839641b24c232799054928e2fe785ec8a60a',
    'fc1ee1a173170e0c385eb8e79b3bc5e9c2bf884b2132b3fbe31bacdf6e1a806d',
    'd7dc4dab0139813f40d776ac76fbf7de91d7879b4623eaabcf8462ceb235902d',
]
for path, digest in zip(manifest_paths, manifest_digests, strict=True):
    check(path, {'sha256': digest})
    for item in json.loads(path.read_text())['files']:
        archive = Path(item['archive']) if 'archive' in item else path.parent / item['name']
        check(archive, item)

readiness = ROOT / 'mpc-constraint-readiness-4f908fc'
gate_path = readiness / 'gate.json'
check(gate_path, {'sha256': '98efb720738e434bdde7ad7b35a87c8268e706af22eb59af9103164dd7dcd91b'})
gate = json.loads(gate_path.read_text())
for item in gate['checks']:
    assert item['exit_code'] == 0
    check(Path(item['log']), {'sha256': item['log_sha256']})

diagnosis_path = manifest_paths[1].parent / 'plant-mpc-forecast-sampling-diagnosis.json'
copies = [
    (inventory_path, OUT / 'adaptive_rl_batch_004_results.json'),
    (diagnosis_path, OUT / 'adaptive_rl_batch_004_forecast_diagnosis.json'),
]
copy_records = []
for source, target in copies:
    with target.open('xb') as handle:
        handle.write(source.read_bytes())
    assert record(source)['sha256'] == record(target)['sha256']
    copy_records.append({'external': record(source), 'git': record(target)})

summary = {
    'schema': 'plant-adaptive-rl-batch-004-failure-publication-v1',
    'source_commit': inventory['source_commit'],
    'source_tree': inventory['source_tree'],
    'performance_admitted': False,
    'kind': 'closed-file forensic publication; normal closure auditor not executed',
    'byte_identical_copies': copy_records,
    'gate': record(gate_path),
    'gate_checks': [{k: v for k, v in item.items() if k != 'argv'} for item in gate['checks']],
    'python_regressions': {'passed': 698, 'pytest_seconds': 1140.75},
    'launch_claim': record(readiness / 'launch-claim.json'),
    'launch_receipt': record(readiness / 'launch-receipt.json'),
    'analysis_tools_manifest': record(manifest_paths[0]),
    'forecast_diagnosis_manifest': record(manifest_paths[1]),
    'exfat_probe_manifest': record(manifest_paths[2]),
    'reverified_file_counts': {
        'batch_ordinary': len(inventory['files']),
        'batch_macos_metadata': len(inventory['macos_metadata_files']),
        'tool_archive': len(json.loads(manifest_paths[0].read_text())['files']),
        'forecast_diagnosis': len(json.loads(manifest_paths[1].read_text())['files']),
        'exfat_probe': len(json.loads(manifest_paths[2].read_text())['files']),
    },
    'script': record(Path(__file__)),
    'new_resets': 0,
    'new_policy_inferences': 0,
    'new_tests': 0,
}
target = OUT / 'adaptive_rl_batch_004_evidence_index.json'
with target.open('x') as handle:
    json.dump(summary, handle, indent=2)
    handle.write('\n')
print(json.dumps({'publication': record(target), 'copies': copy_records,
                  'reverified_file_counts': summary['reverified_file_counts']}, indent=2))

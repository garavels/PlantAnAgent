# Closed batch 002 analysis tools

These four files are byte-identical copies of the outcome-blind tools frozen
on Lexar at `2026-09-12T15:35:59.051255Z`, while batch 002 was running at clean
source `a442a4a9f5b028568d36bbbddf21a34c3609a7cc`. Their manifest is retained at
`/Volumes/Lexar/PlantAnAgentResearch/effort-analysis-tools-a442a4a/manifest.json`,
SHA-256 `4082675893b6eb321f9c665aa4938554eede6444ebd88beefd9a30d599d6368c`.

The auditor requires that exact clean source, its passing gate, normal batch
closure and complete artifact bindings. It reads checkpoint/allocator metadata
and cross-binds the previously completed physical replay receipts; it does not
reset a production environment or reexecute controller inference. A later
checkout must not bypass its pinned-source requirement to reissue an audit.
The reporter requires the resulting audit and preserves all fixed result slots.

The original external copies produced the closed audit and descriptive report
linked by [the result index](../../results/adaptive_rl_batch_002_results.json).
The archived fabricated-artifact suites passed **88 tests in 31.07 seconds**:

```sh
PYTHONPATH=backend/src:simulator/src .venv/bin/python -m pytest -q \
  docs/research/tools/adaptive_batch_002/test_plant_effort_closure_audit.py \
  docs/research/tools/adaptive_batch_002/test_plant_effort_report.py
```

The tests include malformed custody graphs, missing/failed slots, initial-H
equivalence, diagnostic/unit distinctions and a guarded fake-audit-to-report
integration. They use fabricated data, never production outcome selection.

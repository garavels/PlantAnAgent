# Closed batch 003 analysis tools

The original auditor, reporter and their fabricated tests are byte-identical to
the files frozen before launch under
`/Volumes/Lexar/PlantAnAgentResearch/mpc-residual-analysis-tools-4072959`.
Its manifest SHA-256 is
`85291086efc1289c340ec9b0005ba0364dfda9d049358529010d18560c030e50`.
The original suites passed 64 auditor tests and 60 reporter tests.

**Use the versioned `plant-mpc-residual-closure-audit-v2.py` for this closed run.**
The original auditor stopped before performance analysis because its fabricated
allocator fixture incorrectly expected numeric framework reset seeds. The fixed
trainer deliberately seeds optimizer RNG and action space only; physical episode
seeds come exclusively from the allocator. The corrected auditor requires null
framework reset metadata and retains every physical seed, order, progress,
checkpoint, source and replay check. It passed 72 fabricated tests, including
eight new regressions and a spy test of the actual optimizer-seeding override.
The original auditor remains here as historical evidence of the failed attempt.

Both corrected files are byte-identical to the checked correction archived at
`/Volumes/Lexar/PlantAnAgentResearch/adaptive-rl-batch-003-analysis/auditor-correction-001`.
Its manifest SHA-256 is
`0ce985e1a84c9274a1c76e875f69f4afe2c49bf2365983441ef035655819ea6a`.
That directory retains the narrow diff, original failure reference, custody-only
input digests and complete validation note. The experiment and original tools
were never modified or retried.

The auditor requires exact clean source
`4072959f7093ddf672b08386929330545dc98658`, its pinned passing gate, normal closure
and complete artifact bindings. Do not bypass the source pin from a later checkout
to reissue an audit. It checks checkpoint/allocator metadata and previously
completed replay receipts without resetting a production environment or running
policy inference. The unchanged reporter consumes the successfully corrected audit
and preserves all 48 slots, individual deliveries and daily resource vectors.

The original external implementations produced the audit and descriptive report
bound by the [compact result index](../../results/adaptive_rl_batch_003_results.json).
Read-only learning/operational diagnoses and the paired scientific figure have
their own external source and input/output digests in that index. Large evidence
and checkpoints remain outside Git.

The Git SVG removes only Matplotlib's line-end whitespace from the archived
original. Its element sequence, non-path attributes, text and every path-data
token were checked unchanged. The original digest remains in the compact index;
the publication mapping is retained in the external analysis directory's
`publication-formatting/plant-mpc-residual-figure-publication.json`.

Fabricated checks for the corrected auditor:

```sh
PYTHONPATH=backend/src:simulator/src .venv/bin/python -m pytest -q \
  docs/research/tools/adaptive_batch_003/test_plant_mpc_residual_closure_audit_v2.py
```

The reporter's archived suite includes its original fabricated-auditor integration;
the real corrected audit also passed through the unchanged reporter successfully.
All six archived Python files are checked by byte identity, compilation and test
collection after copying. Those archive checks do not constitute another execution
of the full test suites or a new production audit.

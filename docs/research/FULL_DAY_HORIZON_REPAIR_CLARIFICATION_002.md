# Full-day horizon repair clarification 002: numerical and causal admission semantics

Status: **frozen before implementation and before any M24 action on seeds 20--24**

Date: **2026-09-05**

This second clarification resolves numerical, aggregation, and cross-revision admission details
identified by the completed pre-code audit. It is jointly binding with
`FULL_DAY_HORIZON_REPAIR_PROTOCOL.md` and clarification 001. It changes no controller,
environment, objective, matrix, scientific hypothesis, or outcome.

## Exact numerical interpretations

The shared service tolerance is `1e-6 kg`. The following inclusive rules are canonical:

- “positive fourth delivery” means fourth-settlement delivery **greater than** `1e-6 kg`;
- no-regression means
  `M24 delivered >= M6 delivered - 1e-6 kg` and
  `M24 unmet <= M6 unmet + 1e-6 kg` in each cell;
- full request means absolute total-delivery error no greater than `1e-6 kg`, total unmet no
  greater than `1e-6 kg`, and exactly four commitments met;
- H comparability means absolute M24-minus-H delivered and unmet differences are each no
  greater than `1e-6 kg`;
- a 90% mean-unmet reduction means the known M6 mean unmet is greater than `1e-6 kg` and
  `mean(M24 unmet) <= 0.10 * mean(M6 unmet) + 1e-6 kg`; and
- post-third adequacy means
  `CH4 immediately after settlement 3 >= complete request 4 - 1e-6 kg`.

The fourth-settlement material balance is production-aware:

```text
delivered_4 = CH4_before_final_interval
              + cumulative_CH4_production_delta_during_final_interval
              - CH4_after_settlement_4
```

The balance-derived value must agree within `1e-6 kg` with both the public residual's observed
delivery and the retained M24 interval delivery. Comparing only pre- and post-inventory is
invalid because production occurs before settlement.

Economic value separation and raw-driver materiality remain strict `>` comparisons at the
existing R5/R6 thresholds: 1 EUR; 1 kWh for accepted solar, accepted process energy, and battery
throughput; 0.01 h for electrolyser exposure; 0.001 kg for methane-equivalent deficit; and
0.1 kWh for terminal battery deficit. R6's nonterminal driver set remains accepted solar,
battery throughput, and electrolyser equivalent operating hours; accepted process energy is
not added retrospectively.

## Gate evaluation and aggregation

Every intrinsic observation is reported whenever its evidence is available. Gate admission is
chained, matching the robustness tournament:

```text
F1 -> F2 -> F3 -> F4 -> F5 -> F6 -> F7
```

Thus a later gate is recorded `failed` when a prerequisite fails even if its own descriptive
counts happen to meet threshold. F8 reports its intrinsic latency check and passes only when
that check and the exact claim-boundary text are valid; whole-report success still requires all
F1--F8 gates.

Each per-rate numeric summary must retain ordered values by seed, `expected_count`,
`observed_count`, `missing_count`, and explicit `missing_seed_ids`. Mean, population standard
deviation, median, minimum, and maximum are reported when available. `minimum_seed_ids` and
`maximum_seed_ids` contain **every** tied seed. Positive/negative/tie sign counts apply only to
declared paired contrasts (`M24-H` or `M24-M6`); absolute nonnegative measurements label sign
counts `not_applicable`.

The prohibition on mixing rates applies to comparative performance and mechanism summaries.
Global artifact cardinality, source/reference custody, all-gate status, and whole-run timing may
necessarily cover both rates.

## Reference availability and citations

The compact report must cite the root report and all 20 historical cell files so F1 proves the
complete 21-entry reference directory. Only historical positions 11--20 enter M24 analysis.
Complete H or M6 arm/evidence bodies must not be copied. Compact measurements algebraically
derived from those verified bodies are permitted and must retain their source arm digest.

Identity availability is explicit:

| Object | raw file | full/domain | deterministic | evidence graph |
|---|---:|---:|---:|---:|
| old root report | available | available | available | not applicable |
| old cell | available | available | available | not applicable |
| embedded old H | not applicable | H evidence + deterministic arm | available | not retained |
| embedded old M6 | not applicable | run evidence + deterministic arm | available | available |
| new M24 cell | available | available | available | available through embedded M24 run |
| new root report | returned by writer/verifier | available | available | not applicable |

A JSON report cannot contain its own raw-file digest without self-reference. The exclusive
writer and independent verifier return that raw digest out of band. This is the canonical
meaning of the protocol's report-identity requirement.

## Cross-revision causal fidelity

Manifest and settings hashes do not prove that executable implementations stayed unchanged
between the historical run and M24. Admission must therefore compare the registered execution
revision with historical source `7b1b656964dcac4a25c6f5d4d89fed8abf4d4e0e` and reject any
modified or deleted file under these existing runtime surfaces:

```text
simulator/src/
backend/src/plant_backend/controllers/
backend/src/plant_backend/safety/
backend/src/plant_backend/rl/v2/
```

New experiment, detached-verifier, and codec modules added under the last surface are allowed
only because they cannot be imported by or alter the historical controller/environment path.
The admission audit must enumerate those new paths and prove that no pre-existing runtime file
changed. Tests, documentation, and Makefile entry points may change. `uv.lock` must remain at
the frozen SHA-256.

Every M24 solver stage must report SciPy `1.18.1` and HiGHS `1.12.0`, matching the historical M6
evidence. Python and platform identities are reported and compared; inequality is a named
experimental limitation unless a later registration explicitly makes equality an execution
gate.

The first M24 public-observation and public-issue hashes must equal M6 event zero. The complete
reconstructed projection must contain 96 intervals of 900 seconds, bind the same issue, and
have the same projection SHA-256 as M6 event zero. M24 event zero must cite those exact hashes.
The only permitted settings-body difference from M6 is `planning_horizon_h`; `sha256` changes as
a derived identity, while the 96-interval maximum is verified through event-zero behavior.

## Custody and outcome boundary

Reference admission requires exactly 21 regular, nonsymlink directory entries: `report.json`
and the 20 registered cell files. JSON must have canonical bytes, a single trailing newline,
no duplicate keys, and no non-finite constants. The reference report's exact registered source,
worktree, lock, contract, raw, full, and deterministic identities must match. The proposed M24
output may be neither inside the repository nor inside or below the reference directory.

No protected reset or action may occur until the implementation, adversarial tests, and an
admission audit prove every clarified obligation, followed by a separate execution registration.
No H/M6 rerun, outcome, training, controller comparison, or commercial claim is authorized by
this clarification.

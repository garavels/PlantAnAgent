# Research programme

This directory is the auditable record for the Environment V2 research programme.
The programme's objective is to turn Plant An Agent from a convincing integration
prototype into a discriminative, reproducible benchmark for layered industrial
autonomy.

The governing question is not whether a particular learning algorithm can achieve
a high scalar reward. It is whether each intelligence layer produces measurable
operational value under realistic uncertainty, while retaining explicit safety and
fallback behavior.

## Files

- [`JOURNAL.md`](JOURNAL.md) is chronological and append-only in spirit. It records
  commands, observations, decisions, failures, and changes of direction.
- [`ROADMAP.md`](ROADMAP.md) defines hypotheses, experimental gates, benchmark arms,
  and the staged implementation programme.

Large generated datasets, model checkpoints, and raw run outputs remain outside Git.
Every reported experiment must nevertheless record enough provenance to reproduce it:
commit, configuration, seeds, environment contract version, dependency versions, and
the location or digest of generated artifacts.

## Research rules

1. Preserve Environment V1 as a reproducible baseline.
2. Change one scientifically meaningful difficulty class at a time.
3. Compare against the heuristic and at least one model-based or oracle baseline.
4. Separate training, validation, and held-out stress regimes.
5. Report distributions and tail risk, not a single best rollout.
6. Keep hard safety in the guard; do not hide violations in reward weights.
7. Use ablations to attribute value to RL, supervision, and HEPA separately.
8. Treat negative results as results and record them.
9. Do not claim commercial value without translating metrics into operational units.
10. Commit coherent, tested increments with the journal entry that explains them.

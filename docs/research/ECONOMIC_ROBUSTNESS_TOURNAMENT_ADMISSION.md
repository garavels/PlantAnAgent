# Economic robustness tournament — implementation admission

Status: **runner/verifier implementation gate; no seed 20--24 reset or trajectory opened**

Date: **2026-09-04**

Protocol: [`ECONOMIC_ROBUSTNESS_TOURNAMENT_PROTOCOL.md`](ECONOMIC_ROBUSTNESS_TOURNAMENT_PROTOCOL.md)

Raw protocol SHA-256:
`6fb5eb11933ba5ee601849210b4e61278e4cec2ddb97c8e31b05817ee7141412`

Dependency-lock SHA-256:
`080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd`

## Admission purpose

The seed-0 mechanism smoke showed that the alpha economic task can separate two safe controller
trajectories at equal methane service. It did not show that the separation repeats, survives a
full tariff day, or remains after removing the terminal methane-liability term that dominated
that cell. The frozen robustness protocol answers that narrower task-readiness question before
compute is spent on a substantive RL policy.

This document admits the software used to open the matrix. It contains no controller outcome.

## Frozen execution path

The implementation consists of:

- `economic_robustness.py`, the outcome-independent cell schema, paired metrics, algebraic
  sensitivities, R1--R8 aggregation, deterministic evidence projection, and cell verifier;
- `economic_robustness_experiment.py`, the clean-source runner, exclusive publisher, final
  report builder, and deep directory verifier; and
- `test_rl_v2_economic_robustness.py`, which exercises the contract with fabricated evidence
  and non-protected seed 901/902 identifiers only.

The registered execution has no configuration surface. It expands exactly, in order:

```text
horizon 6 h
  rate 35 kg/h, seeds 20 21 22 23 24
  rate 42 kg/h, seeds 20 21 22 23 24
horizon 24 h
  rate 35 kg/h, seeds 20 21 22 23 24
  rate 42 kg/h, seeds 20 21 22 23 24
```

Within every cell it constructs both environments, resets both with the same seed, and compares
the complete manifest, physical exogenous episode, economic episode, checkpoint contract,
tariff-assumption identity, and controller task before H can act. H then executes through the
generic common-setpoint command. M follows through the rich transition-reconciled MPC door. No
cell runs concurrently.

The MPC task horizon follows the cell while its planning horizon remains six hours. The code
checks the two frozen settings hashes before execution:

- 6-hour task: `7f89b35d23ee9313f9a8723050a2dcbc94f9303e18e96cab81e86fa9f2a0261e`;
- 24-hour task: `28dd0dbbd732f4defcc700e3fcb91e59b2d14d6df7bac728963090e37ba48b0f`.

## Evidence and publication

The runner refuses a relative path, a path inside the repository, an existing destination, a
dirty or unidentified Git source, a changed protocol, or a changed dependency lock before it
creates the output directory or resets an environment.

Each cell is written immediately to a new numbered JSON file. Publication uses a flushed and
synced temporary file followed by an exclusive hard link, so an existing artifact is never
overwritten. The runner then reopens the cell and verifies:

- the exact cell, episode, controller, and MPC-settings identities;
- the complete heuristic decision/command/execution digest ledger;
- the service-MPC event, solve-stage, transition, execution, minute-application, authority,
  energy, and latency digest graph;
- reward/component/interval/final-liability accounting;
- the recomputed per-cell R1--R8 inputs; and
- the full and deterministic cell hashes.

After all cells, the runner rechecks the source revision and worktree identity. A compact
`report.json` references every raw file hash, internal report hash, and deterministic evidence
hash. Its verifier reopens all 20 cell files, rejects missing, duplicate, reordered, renamed, or
extra artifacts, recomputes every cell analysis and stratum aggregate, and verifies the report
hashes. A cell failure is retained and later cells continue; missing evidence cannot be replaced
by a surviving-subset aggregate.

Controller, solver, dispatcher, cell, and tournament wall times are reported. They are removed
from the deterministic cell and matrix projections, so repeated physical evidence can retain
the same outcome identity without pretending wall-clock latency is deterministic.

## Frozen analysis coverage

For each horizon/rate stratum, the compact report contains all five seed values, complete and
missing counts, mean, population standard deviation, median, extrema, raw minimum seed, and sign
counts for every registered paired metric. It separately retains controller/solver/dispatcher
latency by seed.

The report also includes, by seed and stratum:

- default M-minus-H value;
- methane-liability, declared-wear, and solar-opportunity multipliers 0, 0.5, 1, and 2;
- no methane liability;
- no declared battery/electrolyser wear reserve;
- service plus solar opportunity cost only;
- the higher arm and every sign flip from the default ledger; and
- the change between six-hour and 24-hour default/no-methane distributions, without a direction
  threshold.

These are algebraic rescoring checks of one retained physical trajectory, not rerun controllers,
calibrated uncertainty ranges, or causal counterfactuals.

## Outcome-free acceptance evidence

The tests prove, without constructing or resetting seeds 20--24. Fabricated matrix fixtures use
identifiers 901 and 902; one 15-minute generic-door wiring test resets non-protected seed 903:

1. protocol and dependency-lock hashes, matrix order, and 2,400/36,000 arithmetic;
2. strict cell serialization and the two frozen MPC settings identities;
3. exact additive sensitivity algebra and coverage of every multiplier;
4. execution, identity, accounting, safety, service, task-expression, nonterminal-expression,
   settlement, and latency calculations;
5. four separate five-seed stratum aggregates, horizon comparison, and R1--R8 decisions;
6. deep cell recomputation and tamper rejection;
7. exclusion of all wall-clock fields from deterministic evidence;
8. absolute/new/outside-repository destination enforcement;
9. construction and reset of both paired environments before H acts;
10. retention of a preflight failure while the next declared cell proceeds; and
11. raw cell/report verification plus rejection of post-publication byte changes.

The existing real seed-0 smoke artifact was also reopened read-only and accepted by the new
analysis and evidence schemas. No simulator reset or controller action occurred during that
compatibility check.

## Authorized command after commit

Only a clean committed revision may run the registered matrix:

```bash
make v2-economic-robustness \
  OUTPUT=/absolute/new/directory/outside/the/repository

make v2-economic-robustness-verify \
  REPORT=/absolute/new/directory/outside/the/repository/report.json
```

The output directory is consumed even when a gate fails. The matrix must not be retried at the
same revision, and seeds 20--24 become inspected development seeds after their first reset.

## Claim boundary

Passing R1--R7 would establish only that this alpha task repeatedly separates the two declared
safe behavioral probes, including without terminal methane liability, and is ready for a
separately registered bounded PPO baseline and tariff-aware economic MPC work. It would not
establish RL value, controller superiority, generalization, commercial savings, or commission
readiness. A failed gate is equally actionable evidence: it sends execution, service
comparability, objective design, or horizon realism back for repair before training.

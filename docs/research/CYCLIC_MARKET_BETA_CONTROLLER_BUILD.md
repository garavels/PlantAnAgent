# Cyclic-market beta controller build

Status: **release-validated controller-neutral software boundary; no trained-policy result**

Date: **2026-09-05**

## Why this increment exists

The alpha environment established the execution architecture, but its deterministic tariff,
single finite horizon, and spendable opening inventory made controller parity difficult to
interpret. A heuristic and a learned policy could look similar because the task itself admitted a
wide set of successful behaviours. The cyclic-market beta was introduced to create recurring
operational obligations under revised forecasts, stochastic electricity value, and unannounced
finite DAC loss without changing the common four-target actuation door.

This increment closes the next software dependency: the beta task now has a native economic MPC,
a four-arm controller-neutral runner, and a detached semantic verifier. The repository can execute
and independently audit one matched comparison among the physical heuristic, rolling service MPC,
beta-native economic MPC, and a genuinely trained PPO checkpoint. It does not contain such a
trained beta checkpoint, and no protected controller ranking has been opened.

## End-to-end architecture

```text
public beta boundary (135 values)
  = physical plant (57)
  + current market issue (73)
  + cycle progress/return margins (5)
                  |
       four independently reset twins
                  |
     +------------+-------------+----------------+---------------+
     |                          |                |               |
 H: physical heuristic   M24: service MPC   E-beta: MPC   PPO: learned policy
     |                          |                |               |
     +------------+-------------+----------------+---------------+
                  |
       ProcessSetpointCommandV1 (four absolute targets)
                  |
     common minute dispatcher -> independent safety guard -> plant_sim
                  |
    physical + economic + cycle settlement + complete evidence ledger
                  |
       eligibility gates -> aggregate -> detached verifier
```

Every arm owns an independent environment instance reconstructed from the same checkpoint-bound
task and reset seed. All four environments must agree on their initial physical, economic,
market, disturbance, cycle, interface, and exogenous identities before any controller is allowed
to move. Controllers cannot replace one another after failure.

## Beta-native economic MPC

`CyclicMarketEconomicMPCControllerV1` is a separately identified controller rather than an alpha
controller pointed at new observations. At every decision it consumes only:

- the detached public physical boundary and current issued-mean solar forecast;
- the already-known current electricity price and the current market issue's 24 hourly means;
- the public cycle position and signed resource-return margins;
- declared plant, constraint, commitment, task, and economic-assumption contracts; and
- episode-opening battery and methane-equivalent endowments captured at its first decision.

It cannot read future price truth, future forecast vintages, the private market regime, the
private outage schedule or recovery time, random seeds, simulator internals, or another
controller's result.

The optimizer retains the proven two-stage lexicographic structure. Stage one maximizes delivery
service. Stage two locks that service level and minimizes the declared opportunity-price,
battery-throughput, modelled electrolyser-life, cold-start, and non-rewarding endowment-deficit
costs. Four additional hard terminal rows require battery, H2, CO2, and saleable CH4 to meet their
independent return minima at the exact next 24-hour cycle boundary. Those rows constrain both
solver stages and use the same absolute numerical feasibility epsilon as environment settlement.

The deterministic model uses the exact public current price for interval zero and only the
current issue's hourly means thereafter. Public p10/p90 values are hash-bound but deliberately
unused by this baseline. A currently observed DAC outage disables DAC in interval zero; later
intervals assume nominal availability because no future recovery schedule is public. Minute-scale
commitment, nonlinear response, guard intervention, forecast error, and unexpected later
availability remain measured model mismatch.

The implementation keeps alpha and beta evidence identities separate. Shared numerical solving
returns an in-memory, model-bound solve result and identity-neutral validated plan values. Only
the alpha or beta wrapper can turn those values into its own serializable plan. Exact-type checks
prevent beta settings, plans, or evidence from passing through public alpha APIs.

## Four-arm comparison and information policy

The registered arm order is fixed:

1. **H** — forecast-blind physical heuristic; receives only the physical observation.
2. **M24** — rolling 24-hour service MPC; receives the physical observation and current public
   solar forecast, but no market issue or cycle-return objective.
3. **E-beta** — beta-native economic MPC; receives the complete typed public beta observation and
   explicitly models the next cycle return.
4. **PPO** — deterministic inference from a checkpoint admitted against the exact 135/4 beta task;
   receives the normalized full public beta observation.

H and M24 are intentional ablations, not equal-information economic contenders. E-beta and PPO
share the public market/cycle surface, while E-beta has an explicit deterministic optimization
model and PPO has only its trained function. The report records these differences so a future
claim cannot silently attribute value to a controller family when it came from extra information.

Each command is bound to its exact decision boundary. The common dispatcher, safety guard, and
simulator remain the physical authority. The report retains all 135 typed and normalized values
at every decision and terminal boundary, controller-specific decision evidence, every minute
application, realized EUR ledger, physical metrics, cycle settlements, failures, and latency. It
also retains one exhaustive run-wide `PlantState` truth ledger from episode step zero through the
last executed minute. The environment supplies this ledger through a protected audit seam; it is
not added to the Gym observation or `info`, so it cannot expand controller information.

## Eligibility and failure semantics

A cell enters value aggregation only when all four arms:

- complete the expected decision and minute execution lattice;
- satisfy every 24-hour battery/H2/CO2/CH4 return settlement;
- pass the hard-safety classification; and
- deliver comparable service within the registered absolute tolerance.

The service gate prevents an apparently cheaper controller from winning by declining the common
job. Cycle infeasibility prevents value created by consuming non-renewed starting resources from
being counted. A failed arm, preflight failure, incomplete cell, and every exclusion reason remain
in the report; no surviving-arm average, imputation, fallback, or dropped seed is allowed.

The simulator labels the complete plant `critical` whenever the deliberately injected DAC outage
is active, so raw `shutdown_steps == 0` would automatically reject roughly the configured outage
fraction independently of controller behaviour. The beta safety classifier therefore retains raw
critical/offline steps, reconstructs the exact executed half-open DAC-outage lattice, and gates on
**unexpected** critical/offline steps. An outage minute is exempted only when the exact scheduled
step is active, DAC is offline, plant status is critical/offline, every non-DAC component remains
online, and battery SOC remains strictly above reserve. Any simultaneous non-DAC outage, reserve
breach, missing outage indication, or off-lattice attribution remains a safety failure. The
classifier additionally requires zero evidence-derived out-of-range decisions and zero
electrolyser forced stops.

The detached verifier does not trust the report's private outage schedule. It deterministically
regenerates the disturbance episode from the registered cell seed and beta root-stream allocation,
requires an exact match with the private audit manifest, and then checks that every regenerated
outage step is expressed by the minute truth ledger. It independently derives component-health
status, aggregate plant status, minimum SOC, critical/offline count, forced-stop delta, and the
controller-specific out-of-range count before rebuilding hard-safety eligibility. Declared DAC
exposure therefore remains evidence and a stress stratum without becoming a blanket exemption for
controller or plant failures.

A qualified run additionally requires canonical H/M24/E-beta implementations, canonical settings,
a trained Stable-Baselines3 PPO artifact, exact checkpoint/training-task readmission, a clean
committed source tree, a repository-local `plant_sim`, and stable source/runtime provenance from
the beginning to the end of execution. Filesystem paths and wall-clock timings remain visible but
are excluded from deterministic scientific identities; model, source, lockfile, task, checkpoint,
and numerical-runtime digests remain bound.

## Detached verification

The verifier does not trust the runner's booleans, summaries, or hashes. It reopens canonical JSON
as a stable regular nonsymlink file, rejects duplicate keys and non-finite values, reconstructs the
training and checkpoint contracts, re-encodes typed and normalized observations in sequence,
checks command-to-boundary and execution continuity, rebuilds physical/economic/cycle metrics and
classifications, and then recomputes arm, cell, aggregate, and report identities.

For E-beta it also reconstructs the public weather and price projections, cycle-return target,
model identity, plan identity, first-step command targets, and decision evidence chain. The
selected stage-two solver vector is retained explicitly in the typed plan. The verifier checks its
cardinality and finite values, reconstructs the service-lock row, binds the vector to the reported
stage-two solution digest, audits bounds/integrality/model rows and the service lock, and recomputes
the named cost components and economic objective from that same vector. Coherent edits to the
vector, digest, plan projection, or cost ledger therefore fail rather than becoming a new
self-consistent report.

The stage-one service vector is not retained. Its reported solution digest remains bound
provenance, not an independently reconstructed proof that the primary optimum was solved. More
generally, the verifier performs arithmetic and contract reconstruction; it does not rerun either
controller or MILP and does not independently certify global solver optimality. Verification proves
the retained stage-two witness is internally model-feasible and evidence-consistent, not that
HiGHS could not have found a better primary or secondary solution.

This auditability intentionally makes the report large and unsuitable as ordinary telemetry. It
contains the private disturbance audit manifest, every controller boundary and minute state, and
solver witnesses. Qualified runs must therefore write to a new external absolute path; protected
reports should remain access-controlled until their preregistered opening rather than being copied
into routine logs or committed by default. A later product telemetry surface should derive a
minimal operator-safe view without weakening this research artifact.

## Outcome-free integration smoke

A bounded fixed-policy seed-34 run was used only to exercise the complete implementation path. All
four arms completed 1,440 physical minute steps and passed hard safety. All four failed the strict
cycle-return gate, their delivered service differed, and the cell therefore produced zero
value-eligible comparisons. The policy was an explicitly untrained fixed-action fixture.

This is not evidence for or against PPO, H, M24, or E-beta performance. It demonstrates that the
runner retains physically complete but scientifically ineligible trajectories instead of turning
them into a ranking. No optimizer training, protected seed, policy selection, or commercial result
was consumed.

## Current verification status

The four-arm runner's dedicated suite passes **17/17** tests, and a focused **37/37** combined
selection covers the runner, beta MPC schema and solver, and environment minute-truth seam. The
expanded detached-verifier suite passes **35/35** tests in 142.40 seconds, including coherent
nested/rehashed safety attacks, disturbance substitution, and retained-vector/digest mutations.
That run exposed and closed two production defects: an incorrect strict transition pairing and an
E-beta out-of-range derivation that assumed a nonexistent generic `invalid` field. Verifier formatting,
Python compilation, and focused typing pass; verifier-test formatting, compilation, and Git
whitespace validation also pass.

The final complete Python repository gate, rerun after the metadata and seed annotations, passes
**1,535 tests with 4 optional skips** in 198.61 seconds. It emits the one existing Starlette/httpx
deprecation warning. Strict scoped MyPy passes for all five touched production modules. Black
passes for all 10 changed Python files; `compileall`, Git whitespace validation, and both beta
command-line `--help` imports pass. The frontend production build and lint pass with the existing
advisory for a bundle larger than 500 kB. No training or protected rollout was part of any release
gate.

## Maintained commands

A qualified comparison requires a real trained beta checkpoint and a new absolute output path
outside the repository:

```text
make v2-beta-controller-benchmark \
  CHECKPOINT=/absolute/path/to/trained-beta-checkpoint \
  SEED_START=<registered-development-seed> \
  SEEDS=<bounded-count> \
  OUTPUT=/absolute/path/to/new-report.json

make v2-beta-controller-benchmark-verify \
  REPORT=/absolute/path/to/report.json
```

The runner intentionally returns a nonzero status when any registered cell is excluded, even
though it still publishes the complete report. Verification success means the artifact is
internally valid; it does not convert an ineligible or unqualified report into performance
evidence.

## What is complete and what is not

Complete in this increment:

- causal beta-native deterministic economic MPC with hard next-cycle resource returns;
- common four-arm execution with explicit information profiles and no fallback;
- checkpoint, source, simulator, runtime, boundary, command, minute, settlement, and metric
  evidence binding;
- fail-closed scientific eligibility and complete failure retention; and
- independent semantic report verification and passing expanded adversarial tamper coverage.

Still required before a controller claim:

1. freeze an outcome-blind beta train/development/held-out regime and seed split;
2. define a finite multi-training-seed budget and checkpoint-selection rule;
3. add constrained RL (ordinary scalar PPO remains an ablation because observing cycle cost does
   not guarantee feasibility);
4. run only the registered development campaign, then lock the controller;
5. open held-out results once and report uncertainty, tails, failures, and ablations; and
6. calibrate or uncertainty-bound economic and degradation coefficients before translating
   simulated EUR differences into a commercial claim.

HEPA remains a predictive state/risk layer and natural language remains an operator-governance
interface. Neither is part of this controller ranking yet. Their value must later be introduced as
separate factorial ablations so detection or explanation benefit is not confused with policy
benefit.

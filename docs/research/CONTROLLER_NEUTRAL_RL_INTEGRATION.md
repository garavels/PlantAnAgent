# Controller-neutral RL integration contract

Status: **implementation gate; frozen before any performance-bearing four-target training or
policy outcome**

Date: **2026-09-04**

## Purpose

The common-setpoint and economic-task phases established the physical environment that an RL
policy should eventually solve, but the maintained training/checkpoint/evaluation code still
targeted the historical 30-observation, five-action V1 environment. A Gym class alone is not a
finished RL integration. A credible integration must make the following path runnable and
auditable without allowing an old checkpoint to acquire new semantics:

```text
typed public observation
        -> exact checkpoint observation vector
        -> deterministic policy inference
        -> four normalized target coordinates
        -> checkpoint-bound decision evidence
        -> ProcessSetpointCommandV1
        -> common minute dispatcher
        -> independent safety guard
        -> plant and economic ledger
```

This phase wires that path. Its acceptance check may train a disposable two-transition policy
and resume it for two more transitions solely to exercise the real trainer and serializer. It
does not produce a research policy or claim that PPO, RL, or the new task is performant.

## Supported tasks

The integration supports only the two separately identified four-target tasks:

- `plant-env-v2-common-setpoint-alpha`: 57 public physical/forecast observations, the
  diagnostic V2 service reward, and four normalized absolute targets;
- `plant-env-v2-economic-wear-alpha`: the same physical authority, five append-only public
  tariff features, and the additive EUR ledger.

Historical `plant-env-v1` and the five-action `PlantEnvV2` retain their current checkpoint,
training, runtime, and evaluation behavior. No implicit environment upgrade is permitted.

### Completion boundary: research task, not the legacy live runtime

“Integrated” in this document means that both four-target tasks have one maintained,
fail-closed train/resume/load/infer/evaluate path through the common absolute-setpoint command.
It does **not** mean that a four-target checkpoint can be selected in the existing control-room
runtime. That runtime still advances the historical one-minute `plant-env-v1` interface, and
its `/api/controller/rl` slot therefore admits only the 30-observation/five-action V1 contract.

A safe live V2 deployment requires a separately versioned runtime that supplies the same causal
forecast vintage, commitment context, 15-minute decision cadence, tariff context for the
economic task, and held-setpoint dispatcher used in training. Adapting a V2 policy to V1 by
dropping fields, inventing forecasts, or converting targets back to instantaneous shares would
violate its checkpoint contract. The repository deliberately rejects that shortcut. Live V2
orchestration is a product milestone after the research task is admitted, not part of this
training-integration gate.

## Exact checkpoint admission

`PolicyEnvironmentContract` detaches the policy-facing contract from any globally imported
default environment. A four-target checkpoint must match all of:

1. observation and action field order;
2. interface specification version;
3. environment identifier;
4. observation/action semantic digest; and
5. the full task checkpoint digest, which includes plant design, episode distribution,
   constraints, objective, tariff, economic assumptions, and diagnostic weights.

Real policy artifacts additionally carry a SHA-256 content digest. A missing or changed model
file is corruption, not a request to initialize another policy. Historical V1 metadata is
still accepted by historical callers, but absence of the new full-task identity fails closed
when an explicit V2 contract is requested.

## Learned controller adapter

`NormalizedSetpointPolicyControllerV1` consumes a compatible loaded policy plus the detached
public controller observation. It reconstructs exactly the vector seen during training,
including the five tariff values for the economic task. It rejects the wrong observation
kind, tariff identity, assumptions identity, plant design, safety constraints, or commitment
profile.

Every decision records:

- physical and, where applicable, economic observation identities;
- the exact float32 policy-vector identity;
- checkpoint-metadata and policy-contract identities;
- raw and clipped normalized outputs;
- decoded absolute targets; and
- the resulting observation-bound common command.

The adapter raises on inference or contract failure and has no learned, heuristic, or idle
fallback. Evaluation retains that failure before any state movement. Deployment fallback is a
separate supervisory responsibility.

## Training boundary

`ControllerNeutralTrainingConfigV1` is a strict JSON-round-trippable description of one run:
task, plant, episode distribution, safety constraints, diagnostic reward, economic assumptions,
PPO settings, seed, vectorization, checkpoint cadence, output, and resume source. Unknown keys,
wrong scalar types, non-finite values, invalid PPO ranges, and inconsistent rollout/batch sizes
fail rather than being ignored. Defaults select the economic task because that is the active
research direction; the physical-only four-target task remains available as an ablation.

The dry-run path constructs and resets the exact environment, emits its interface/full-task
digests and simulation arithmetic—including SB3's minimum whole-rollout collection above the
requested timestep count—and imports no training library. Real training remains an explicit
opt-in operation. Resume checks the full task contract before model loading. Every
intermediate and final save is a complete checkpoint with metadata and artifact digest; a raw
SB3 zip is not treated as a checkpoint. Existing output directories are never overwritten, and
dependency or resume-admission failure does not leave an empty run directory.

The closure audit additionally requires resume truthfulness. A resumed run may change its
output, duration, vector count, run seed, checkpoint cadence, notes, and explicitly applied
device, but it may not relabel the loaded model with different policy or PPO optimization
settings. The child checkpoint records the parent metadata bytes, model bytes, task digest, and
parent timestep count, and rechecks both parent files before deserialization and every save.
Real model artifacts must also embed concrete Gym observation/action spaces equal to the
declared task; metadata with the right names cannot legitimize model bytes for another shape,
dtype, or bound.

PPO remains the first maintained algorithm only to establish a reproducible baseline. The
controller and checkpoint contracts are algorithm-neutral; adding SAC or another algorithm
later must not create another physical action path.

## Evaluation boundary

The paired evaluator runs the common-door heuristic and a compatible deterministic checkpoint
in independent environments with the same seed. Both arms bind decisions to the current public
observation and execute `ProcessSetpointCommandV1` through the same dispatcher, guard, plant,
and reward. Each interval retains command and physical-minute application evidence. Paired full
manifest identities must match or evaluation aborts. Wall-clock timings remain reportable but
are excluded from the separate deterministic evidence digest. Optional JSON output is
exclusive and never overwrites an earlier evaluation.

A fixed-action checkpoint is always labelled an integration smoke. A trained checkpoint result
is descriptive until a separate protocol freezes seeds, benchmark arms, statistics, selection
rules, and protected evaluation boundaries. The existing service MPC keeps its richer evidence
door because transition reconciliation requires it; the common allocation and economic layers
under both doors are already tested as equal.

## Outcome-free acceptance gate

Before any performance-bearing four-target training run or RL result:

1. legacy V1 checkpoint loading and runtime tests remain unchanged;
2. both four-target tasks produce exact 57/4 and 62/4 dry-run contracts;
3. configuration round-trips all task-defining values and rejects unknown nested settings,
   nonnumeric or nonfinite constraint values, and direct-construction attempts to bypass that
   validation;
4. V2 checkpoints fail on the wrong interface, physical/economic task, task digest, artifact
   bytes, or model-internal Gym spaces;
5. a normalized policy decision binds its exact observation, checkpoint, decoded targets, and
   command;
6. the same normalized vector through Gym and the checkpoint-bound controller produces exact
   physical/economic parity;
7. one tiny policy episode contains one typed interval execution and all fifteen minute
   applications;
8. inference failure produces no action, fallback, or state movement;
9. paired heuristic/policy smoke environments have identical manifest identity, and checkpoint
   compatibility is rechecked before either arm moves;
10. changed embedded PPO settings are rejected on resume and every child binds its exact parent;
11. disposable real-SB3 tests train two transitions, save, resume to four, reload, and evaluate
    both the 57/4 and 62/4 tasks when the optional dependency is available; and
12. focused and complete tests, formatting, compilation, strict targeted type checks, Git
    whitespace, dependency lock, and clean status pass.

Tiny one-decision fixed-policy fixtures and the disposable two-plus-two-transition PPO smoke are
wiring tests, not controller outcomes. No research policy, hyperparameter search, multi-seed
benchmark, HEPA fit, NLP evaluation, or commercial claim is authorized by this integration
phase.

## Next decision

The subsequent frozen 20-cell heuristic/MPC robustness tournament completed from revision
`7b1b656` with R1--R3 passed and R4--R7 failed; see
`results/v2_economic_robustness_7b1b656.md`. Both six-hour strata were service-comparable in
5/5 cells, but both 24-hour strata were comparable in 0/5 because the six-hour-lookahead MPC
failed the sunless fourth commitment while retaining battery energy and convertible feedstock.
The alpha task is therefore not admitted for substantive PPO training. The integration remains
ready, but the next work is a newly versioned full-day service/terminal-adequacy repair under a
new protocol. Integration readiness and one-cell task expression must not be confused with
evidence that learning is useful.

# Controller-neutral benchmark build

Status: **implemented software boundary; no controller-performance outcome opened**

Date: **2026-09-05**

## Purpose

The four-target PPO integration already provides real training, resume, checkpoint admission,
deterministic inference, common-command execution, and paired heuristic evaluation. The M24
horizon repair separately established a working causal service optimizer. The missing software
boundary was a single comparison path in which the learned policy and the strongest implemented
model-based controllers act on independently reset copies of the same economic task.

This build closes that boundary. It does not manufacture a trained policy and it does not turn
the historical M24 development result into an RL result.

## Controllers

The benchmark has four fixed arms:

1. **H** — the frozen forecast-blind heuristic, converted transparently to the common absolute
   process targets;
2. **M24** — the causal service MPC with a rolling horizon of up to 24 hours and its existing
   rich solver, setpoint, minute-application, and transition evidence;
3. **E** — a separately identified service-constrained economic MPC that first maximizes
   delivered methane and then minimizes predicted EUR cost without relaxing the service lock;
4. **PPO** — a real trained Stable-Baselines3 checkpoint admitted against its complete 62/4
   economic-task contract and executed deterministically through the common command door.

There is no substitute policy, idle action, heuristic takeover, or surviving-arm replacement.
An arm failure is retained as the outcome of that arm.

## Economic MPC boundary

The economic optimizer is not a renamed M24 objective. It reuses the validated physical MILP
but has separate settings, model, plan, solver-stage, decision, and controller identities. Its
second stage includes:

- interval-specific accepted-solar opportunity cost using the declared day-ahead tariff and the
  same zero floor used by the realized economic ledger;
- bus-side battery charge and support throughput reserve;
- the modeled running-time and cold-start portion of electrolyser life reserve; and
- non-negative terminal methane-equivalent and battery endowment-deficit hinges.

The hinge formulation is important: terminal surplus has zero liability credit. A simple linear
terminal-inventory reward would recreate the horizon artifact already observed in the alpha
task.

The controller validates the five-value public tariff summary against the complete declared
schedule at every boundary. The full schedule is static, public task data, not private weather
truth. Nevertheless, the alpha PPO observation contains only summary values through six hours,
whereas the model-based controller projects the declared schedule across its horizon. E is
therefore an unequal-representation model-based reference on the alpha task, not yet proof of an
equal-information tariff contest. Environment V2 beta must expose a common versioned price
forecast before tariff-profile generalization can be claimed.

The optimizer explicitly declares the remaining model mismatch. Its issued-mean, 15-minute MILP
does not exactly predict minute-scale ramping, low-load classification, hot starts, aborted
starts, forced stops, forecast errors, guard intervention, or nonlinear plant response. Those
effects remain measured by the authoritative simulator and realized EUR ledger.

## Admission and execution rules

Before any requested cell is allowed to move, the evaluator reopens the checkpoint metadata and
policy bytes, validates the embedded spaces and complete task contract, and binds their content
digests. A qualified run additionally requires a clean committed worktree, records the full Git
revision, worktree-status digest, dependency-lock digest, and Python/numerical runtime versions,
and refuses injected E implementations or noncanonical M24 tuning. The injectable library seams
remain available only in reports explicitly labelled as unqualified integration fixtures. For
every requested seed, the evaluator then:

1. constructs four independent instances of the exact economic task recovered from the
   checkpoint;
2. resets all instances to the same seed and compares their physical, economic, assumptions,
   and checkpoint-contract manifest identities before any action;
3. constructs and resets every controller without moving an environment;
4. executes each arm through its typed command path, independent safety guard, one-minute plant,
   delivery settlement, and economic ledger;
5. retains decisions, interval economics, complete physical-minute execution evidence, failures,
   typed solver-stage failure evidence, and latency; and
6. includes a cell in numeric aggregation only when all four arms completed that paired episode.

Incomplete cells and every failed arm remain in the report and denominator accounting. The
deterministic evidence identity excludes wall-clock measurements and checkpoint filesystem
location while the raw report retains both. JSON publication is exclusive and refuses to
overwrite an existing artifact. The maintained CLI requires an absolute destination outside the
source repository.

## What this build can prove

With a compatible trained checkpoint, this runner can establish descriptive same-task paired
behavior under new episode seeds: service, unmet delivery, economic value, physical utilization,
wear proxies, safety outcomes, controller failure, and latency for all four controllers.

It cannot by itself establish:

- that PPO is better than a heuristic or MPC;
- generalization to a different horizon, demand, plant, tariff, constraint set, or objective;
- calibrated savings or a bankable business case;
- field, hardware-in-the-loop, operator, or site performance; or
- commission or commercial superiority.

Those claims require a prospectively frozen train/development/held-out split, multiple policy
training seeds, calibrated coefficient uncertainty, a common price-forecast interface, genuinely
new stochastic regimes, a separately frozen protocol and detached semantic report verifier, and
later hardware/site evidence. The checkpoint contract remains strict instead of being weakened
to simulate such generalization.

## Maintained command

The benchmark accepts only a real trained economic-task checkpoint and requires a new output
path:

```text
make v2-controller-benchmark \
  CHECKPOINT=/absolute/path/to/checkpoint \
  SEED_START=<unopened-development-seed> \
  SEEDS=<bounded-count> \
  OUTPUT=/absolute/path/to/new-report.json
```

No research-scale PPO checkpoint is stored in this repository. A colleague's policy can be used
only after the complete checkpoint directory and its embedded training task pass admission.

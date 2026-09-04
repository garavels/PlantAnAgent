# Controller-neutral setpoint authority — phase 1 protocol

Status: **frozen before implementation**

Date: **2026-09-04**

## Evidence that motivates this phase

The first absolute-setpoint MPC mechanism cell passed S1--S8 and restored full contracted
delivery, proving that the deterministic target dispatcher is a credible actuation seam.
However, the exact-cell A0 heuristic also delivered the complete 2,520 kg contract through
the historical five-field share action. A0 used 3.856 MWh less battery throughput and 1.043
kWh/kg less accepted process energy. This is not yet a fair controller comparison because
the algorithms do not have the same action representation or real-time support semantics.

Before adding economics or training RL, every online policy needs the same physical
authority. This phase extracts the already-tested target-allocation kernel from its MPC
provenance wrapper and defines the normalized action that a learned policy will eventually
use. It opens no controller trajectory or performance outcome.

## Layer boundary

```text
heuristic / MPC / RL
        |
        | four non-negative absolute targets
        v
controller-neutral target allocation
        |
        | ordinary Action for one contemporaneous solar bus
        v
independent safety guard -> plant
```

The four supervisory targets are:

1. total electrolyser input power, including startup and productive power;
2. DAC input power;
3. methanator input power; and
4. battery-charge ceiling.

Battery support is intentionally not a fifth controller action. The deterministic dispatcher
requests only the contemporaneous shortfall needed to reach the electrolyser target, bounded
by the shared reserve-aware availability calculation. This prevents learned or heuristic
controllers from reintroducing the failed additive-support ambiguity.

## Controller-neutral target value

`ProcessSetpointTargetsV1` contains exactly the four absolute kW values above. Every value is
finite and non-negative. A target value is not a command by itself: decision index,
observation identity, controller/checkpoint identity, active envelope, hold duration, and
execution evidence will be bound by the later common-command phase.

The type is deliberately free of MPC model, plan, forecast, settings, and solver identities.
Those remain in `MPCProcessSetpointsV1` and its decision evidence. A transparent heuristic
and an RL policy must not fabricate optimizer provenance to use the common actuator.

## Common allocation kernel

`allocate_process_setpoint_targets` receives:

- one `ProcessSetpointTargetsV1`;
- one finite, non-negative contemporaneous solar bus value; and
- one typed `BatterySupportAvailability` produced by the existing shared safety primitive.

It implements exactly the already-frozen `reserve-aware-proportional-unsupported-loads-v1`
algorithm. It returns a typed allocation containing four solar routes, battery support,
unallocated solar, target reachability, and the ordinary five-field plant `Action`.

The current MPC dispatcher must delegate to this kernel without changing any command,
dispatch, application, run, model, plan, or evidence identity. The kernel does not inspect a
forecast, reward, delivery request, seed, controller identity, or private plant future.

## Normalized learned-policy action

`ProcessSetpointActionAdapterV1` defines a new four-dimensional continuous action with fields
in the target order above. Each raw coordinate has policy range `[-1, 1]`; after finite-value
validation it is clipped to that range and mapped linearly to `[0, active_limit_kw]`.

The adapter retains both the raw and bounded vector and an explicit `clipped` flag. It rejects
the wrong shape and non-finite or non-numeric values. It performs no resource allocation,
safety correction, forecast logic, reward logic, or physical step. The resulting target type
is shared with heuristic and MPC paths.

This phase does not change `PlantEnvV2.action_space` or checkpoint compatibility. The later
common-command/environment phase will introduce a separately versioned Gym contract, so
historical PPO artifacts cannot be silently interpreted under new action semantics.

## Outcome-free implementation gate

No heuristic, MPC, or RL comparison is authorized by this phase until all of the following
pass:

1. target values reject booleans, non-numeric values, non-finite values, and negatives;
2. the normalized adapter rejects wrong shape/non-finite input and explicitly reports
   clipping;
3. `-1`, `0`, and `+1` map exactly to zero, half, and full active limits for all four fields;
4. controller limits and operator derating change decoded targets without changing action
   semantics;
5. the common allocation reproduces day, night, surplus, deficit, reserve-bound, and
   proportional-scarcity behavior;
6. identical inputs produce equal typed allocations and ordinary actions;
7. the MPC dispatcher delegates to the common kernel and remains byte-identical on the
   existing grid and frozen three-boundary tests;
8. historical Gym/action paths and the registered setpoint runner contract remain unchanged;
9. focused and complete tests, formatting, compilation, strict targeted types, lock identity,
   whitespace, and clean Git status pass.

Tiny pure fixtures and existing short episodes are allowed. No full controller episode,
multi-seed matrix, RL training, HEPA fitting, or economic outcome may be opened in this
phase.

## Next admission boundary

After this kernel passes, a separately frozen phase will add:

- a controller-neutral observation-bound command and physical execution ledger;
- an opt-in Gym environment with the four-dimensional normalized target action;
- a transparent setpoint heuristic using that exact command;
- an adapter from service-MPC evidence to the common command without deleting its original
  provenance; and
- paired equivalence tests proving all policies reach the same dispatcher and guard.

Only then is it scientifically meaningful to add the first economic/wear pressure or train a
new policy.

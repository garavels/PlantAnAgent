# Common setpoint command and execution door

Status: **frozen before implementation**

Date: **2026-09-04**

## Purpose

The shared allocation kernel proves that absolute targets can be controller-neutral, but
only the service MPC can currently bind those targets to an observation and execute them in
`PlantEnvV2`. The historical Gym policy still emits five share/support values. Comparing
controllers before they cross the same command, dispatcher, guard, and plant boundary would
confound algorithm quality with actuator semantics.

This phase creates a separately versioned, opt-in control door. It must not change the
historical V2 observation vector, five-dimensional action, checkpoint contract, default
environment, registered MPC artifact, or any frozen MPC evidence identity.

## Common command

`ProcessSetpointCommandV1` binds:

- a non-negative decision index;
- the exact detached `ControllerObservationV2` identity;
- a positive hold duration;
- a non-empty issuer/controller identifier;
- the schema and SHA-256 identity of the decision evidence that produced the targets;
- one `ProcessSetpointTargetsV1`; and
- the resolved process, battery-charge, battery-support, and reserve envelope.

The command contains no optimizer-only field. A typed adapter may project an
`MPCProcessSetpointsV1` into the common command, but its source schema and complete original
digest must be retained. A heuristic and learned policy must bind their own decision
evidence rather than fabricate MPC provenance.

## Shared dispatch and safety

`dispatch_process_setpoint_command` must validate the active envelope, compute battery
support from the current physical boundary through the existing shared safety primitive,
and call `allocate_process_setpoint_targets`. It may inspect only the held command, current
plant boundary, current solar bus, static plant configuration, and active constraints.

The existing `dispatch_mpc_setpoints` must delegate to that function and wrap the result in
its unchanged MPC evidence schema. The ordinary independent `validate` guard remains the
only interface safety filter, and the simulator remains the final physical authority.

## Opt-in Gym contract

A new environment class will retain the existing 57-field observation semantics and expose
exactly four normalized target actions in the order frozen by
`COMMON_SETPOINT_AUTHORITY_PROTOCOL.md`. It will have a new environment identifier,
interface digest, and checkpoint identity. Historical checkpoints must fail compatibility
rather than being interpreted as four-target policies.

Each normalized action is retained as source evidence, decoded once into absolute targets,
bound to the current public observation and active envelope, and held as a command for one
controller interval. The dispatcher recomputes the ordinary plant action at every physical
minute. The returned info must expose requested targets, minute dispatch/guard/application
evidence, reward, and physical metrics.

## Transparent comparator

The first common-door heuristic is an explicitly labelled authority adapter around the
current A0 decision rule. At the decision boundary it converts A0's sanitized shares and
support into absolute process targets using only the same measured solar sample A0 used:

- electrolyser target = routed solar plus requested support;
- DAC and methanator targets = routed solar;
- battery-charge ceiling = routed solar.

This does not claim behavioral equivalence over the held interval: unlike A0's held shares,
the new command holds absolute targets while solar changes. That difference is precisely the
actuation abstraction being isolated. A later benchmark must label A0-share and A0-setpoint
as distinct arms.

## Outcome-free acceptance gate

Before any controller outcome, economics change, or training:

1. invalid indices, identities, durations, targets, and envelopes fail closed;
2. target values cannot exceed the command envelope;
3. identical command inputs produce identical commands and dispatches;
4. current constraints or reserve changes invalidate stale commands;
5. MPC adaptation retains the exact original command digest and values;
6. MPC dispatch outputs and all frozen identities remain unchanged;
7. normalized Gym actions use the four-target adapter and a new contract identity;
8. heuristic, normalized-action, and MPC adapters can issue the same command values;
9. every path reaches the same common dispatcher and independent guard;
10. historical V2 environment/action/checkpoint contracts remain byte-identical;
11. focused and complete tests, formatting, compilation, strict targeted types, lock
    identity, whitespace, and clean Git status pass.

Tiny fixtures and short physical intervals are allowed. No protected 72-hour rollout,
multi-seed comparison, RL training, HEPA fit, economic result, or commercial claim is opened
by this phase.

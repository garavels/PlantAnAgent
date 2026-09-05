# Full-day horizon repair — implementation admission

Status: **implementation and evidence gate passed; no M24 action on seeds 20--24 has run**

Date: **2026-09-05**

This document admits the software for the single-variable M24 mechanism probe frozen by
`FULL_DAY_HORIZON_REPAIR_PROTOCOL.md` and clarifications 001 and 002. It contains no protected
controller outcome. A separate execution registration must bind the final clean revision,
detached worktree, and unused external destination before any seed 20--24 environment is reset.

## Frozen inputs

| Input | SHA-256 |
|---|---|
| Protocol | `e203d1bb12246bc43278b686779afdfde310f8b56e487dac7f9d05947079e0e1` |
| Clarification 001 | `8dc341b142a41214b7845b0f5debab29b7f23751ff8f7cd71ab68ac30f6b7ce3` |
| Clarification 002 | `e2ed43908242a228669859fb7111e4b4f24513d23f467daf60b8c9b1f7692966` |
| Dependency lock | `080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd` |
| Historical executable source | `7b1b656964dcac4a25c6f5d4d89fed8abf4d4e0e` |
| Historical raw report | `26c9ab4a85693dab05d93f7a1b65e86564c74bfe667ee54813901a466d1e0755` |
| Implementation commit before this admission record | `f7ee3041ad2ea20fb57bce26a621178c118b8df5` |

The experiment remains exactly the registered horizon-only intervention. M24 uses the existing
`ServiceMPCSetpointControllerV1`, 24-hour task, 15-minute control cadence, and unchanged
controller/environment stack; only `planning_horizon_h` changes from 6 to 24. Its frozen settings
identity is `67877c94438fc1bf1bd3ad564f17d860f77cea10b82f07bba0d52b6a6f6b1778`.

## Implemented production path

The implementation is additive and split into four bounded modules:

- `full_day_horizon_admission.py` performs preflight and postflight custody. It accepts exactly
  the registered 21-file historical directory, canonical regular nonsymlink JSON, every frozen
  report/cell/source/lock identity, a clean authenticated Git diff, and a new external output
  destination. It rejects changes or deletions to every pre-existing simulator, controller,
  safety, and V2 runtime path. Only the four new M24 modules, tests, documentation, README, and
  Makefile entry points are admitted.
- `full_day_horizon_codec.py` reconstructs every retained M6/M24 target, solve stage, selection,
  setpoint, dispatch, guard action, commitment boundary, minute application, interval execution,
  residual, success/failure event, and complete run through the existing typed contracts. Exact
  round trips and a rebuilt run graph prevent a coherently rehashed nested mutation from passing.
- `full_day_horizon_repair.py` owns the immutable experiment contract, first-boundary forecast
  audit, 96-row interval ledger, physical and economic reconstruction, explicit legacy
  availability states, per-seed/per-rate summaries, and chained F1--F8 decision logic. It
  independently closes delivery, energy, exposure, tariff, terminal-liability, and all seven
  additive objective components.
- `full_day_horizon_repair_experiment.py` executes the ten cells sequentially, publishes each
  artifact immediately by durable exclusive creation, retains failure without retry, performs
  source/reference admission before and after execution, publishes the compact report once, and
  reopens the complete old/new evidence graph with the detached verifier. Wall-clock records and
  host paths remain observable but are removed from deterministic scientific identities.

The Makefile exposes only the frozen `v2-full-day-horizon-repair` run command and the detached
`v2-full-day-horizon-repair-verify` command. There is no horizon sweep, outcome-dependent option,
fallback controller, retry switch, surviving-subset aggregate, or alternate seed surface.

## Outcome-free verification evidence

The four focused test files contain 96 passing tests. They cover:

1. exact protocol, clarification, lock, historical-source, matrix, settings, and filename bytes;
2. canonical JSON, duplicate-key, non-finite, symlink, non-regular-entry, path-escape, TOCTOU,
   missing/extra reference, and external-destination rejection;
3. rejection of every modified/deleted pre-existing protected runtime path and every undeclared
   new runtime or packaging surface;
4. complete typed reconstruction of successful and failed MPC runs and coherent-rehash attacks;
5. complete first-boundary issue/projection/event-zero binding and exact four-target semantics;
6. the 96-interval delivery, economic, energy, cadence, boundary-chain, settlement, and terminal
   accounting ledger;
7. exact service tolerances, product-form adequacy, paired economic sensitivity, F7 eligibility,
   five-seed denominators, missingness, tied extrema, and F1--F8 prerequisite chaining;
8. deterministic exclusion of latency and failure paths while retaining physical outcomes;
9. durable exclusive publication and failure retention without constructing a protected
   environment; and
10. an unmocked run-to-publish-to-detached-verify fixture over ten synthetic failed cells,
    including rejection of extra files, reordered and coherently rehashed references,
    noncanonical bytes, canonical body tampering, and postflight source drift.

That end-to-end test found two real serialization-boundary defects before admission: Python tuple
values in `EpisodeConfigV2.to_dict()` were compared directly with JSON arrays after reopening.
Both the task comparison and independently rebuilt complete/failed cell comparison now use the
same canonical JSON value representation. The adversarial fixture then passed without weakening
any field or digest check.

Two real 24-hour integration exercises used only non-protected seeds 905 and 906. Each completed
96 decisions, 1,440 physical steps, 96 interval executions, 1,440 minute applications, and four
settlements through the production M24 action/execution/ledger path. Seed 905 exposed and fixed a
real time-origin assumption: the first projection begins at the episode's 06:00 public boundary,
not at simulation second zero. Read-only compatibility then decoded and measured all ten selected
historical H/M6 cells. This found and fixed the retained heuristic allocation path: electrolyser
target and request live under `dispatch.allocation.targets.setpoints_kw.electrolyser` and
`dispatch.allocation.request_kw.electrolyser_total`. No H or M6 arm was rerun.

The full repository gate at the implementation commit passed **1,372 tests with 4 skips**. Strict
typing passed for all four new production modules with import following, Black accepted all eight
new source/test files, bytecode compilation passed, and Git whitespace checks passed. The
frontend TypeScript/Vite production build also passed after installing the locked dependencies;
its existing large 3D chunk warning is unrelated to this backend-only addition.

## Read-only historical admission result

From a clean worktree based on the frozen pre-implementation revision, the production admission
code reopened the 215 MB historical directory, deeply verified the old robustness report, and
typed-rebuilt every M6 run across all 20 cells. It accepted exactly the complete reference set and
selected, in order:

```text
h24-r35-s20, h24-r35-s21, h24-r35-s22, h24-r35-s23, h24-r35-s24,
h24-r42-s20, h24-r42-s21, h24-r42-s22, h24-r42-s23, h24-r42-s24
```

The authenticated source diff reports a clean target, all four required new M24 runtime modules,
no modified or deleted pre-existing protected runtime file, and only admitted test,
documentation, README, and Makefile changes. The historical raw report identity reproduced as
`26c9ab4a85693dab05d93f7a1b65e86564c74bfe667ee54813901a466d1e0755`.

## Admission decision and remaining boundary

The implementation, adversarial suite, non-protected integration checks, clean-source audit,
dependency lock, and complete historical evidence graph pass the outcome-free gate. The next
authorized step is a separate registration followed by one sequential execution of the ten M24
development cells from that exact clean revision and a fresh external destination.

Passing the later probe may show only that complete public-horizon lookahead repairs this known
development failure while preserving the frozen stack. It cannot show RL value, controller
superiority, calibrated economics, held-out generalization, site transfer, commercial savings,
commission merit, HEPA value, or language-supervisor value.

# Feasible planner repair 001 — legacy binary64 closure

Status: **repair rule frozen before implementation or another registered execution**

## Trigger and scope

The first invocation of the registered 35 kg/h, seed-0 development cell at clean revision
`0059626313866d7a2bda2a44f64926a757a9f5d6` failed closed after exact A0 generation and
replay. The retained artifact reports `beam_search/search_incomplete` with the message
`action shares must have a binary64 sum no greater than one`. No searched terminal outcome
was observed and the other five matrix cells were not opened.

This repair may reconcile only the representation of an exact legacy-controller request with
the planner action-record invariant. It may not change the search library size or order,
template targets, beam width, macro length, ranking, score, polishing, resource limits, task,
seeds, delivery rates, A0 implementation, safety guard, plant dynamics, or relaxation.

## Root cause and noncanonical witness

`legacy_live` is the only search template that intentionally bypasses the future-truth target
router and returns the exact request emitted by a freshly reset `LegacyHeuristicControllerV2`.
The controller constructs component powers by repeated subtraction and then divides each by
the solar bus. Binary64 non-associativity means the resulting shares can sum to one plus one
ULP even though they encode the controller's complete bus allocation.

A synthetic telemetry-only witness uses solar bus
`3097.9458568525442 == 0x1.833e4475970c0p+11` with the default plant and constraints, low
H2/CO2/CH4 fills, and 35% battery SOC. The shipped heuristic emits the ordered four-share
tuple:

```text
battery       0x0.0p+0
electrolyser  0x1.1bed21723950fp-1
DAC           0x1.8ca661a8a7735p-2
methanator    0x1.dbfadb972f572p-5
sum           0x1.0000000000001p+0
```

Python's exact `Action.sanitised` operation divides the four shares by that total, but every
quotient rounds back to the identical binary64 input. The complete five-double action is
therefore already a bit-identical fixed point of the simulator's authoritative sanitizer. The
planner schema nevertheless rejects it solely because it tests `sum > 1.0`. The target-router
templates do not have this ambiguity: the frozen proportional closure explicitly moves their
common scale downward until their ordered sum is no greater than one.

This witness does not recreate or inspect a registered episode. It establishes the arithmetic
mechanism independently of the protected result.

## Frozen repair rule

Preserve the exact legacy request bytes. Do not apply planner proportional closure to
`legacy_live`, because that would change the action bound to the A0 controller descriptor and
silently change a frozen search template after an outcome was opened.

`ActionRecordV1` continues to require every share to be finite and individually within
`[0, 1]`. An ordered sum no greater than one remains valid for every lineage. If the ordered
sum exceeds one, validation may accept the record only when both conditions hold:

1. its lineage is exact A0 capture (`source_code=-1`) or the live legacy template
   (`source_code=0`); and
2. recomputing the simulator's four normalizing divisions in field order leaves every share
   bit-identical under `struct.pack(">d", value)`.

All other oversubscribed records remain invalid. In particular, templates 1--14 remain bound
to the protocol's strict proportional-closure rule, and a legacy request that normalization
would actually change remains invalid. The schema implementation remains pure and reproduces
the five-line binary64 normalization locally rather than importing simulator or controller
code.

This is an operational-closedness correction, not an action-library or controller change: the
accepted request, `Action.sanitised` result, guard input, plant action, trace bytes, source
binding, and replay action are identical. Source provenance will distinguish the repaired
implementation from the failed revision.

## Required evidence before re-freeze

- Reproduce the exact synthetic heuristic vector and its one-ULP ordered sum.
- Prove its sanitizer output is bit-identical.
- Prove both A0-capture and `legacy_live` records preserve all five bytes and reconstruct.
- Reject the same fixed-point vector under every nonlegacy template lineage.
- Reject a legacy oversubscription whose normalization changes any share.
- Preserve the existing strict closure tests and frozen target-router reference vector.
- Run the complete planner gate, full repository suite, formatter, combined strict type check,
  compilation, lock validation, whitespace validation, and a clean-worktree check.

After a new clean execution revision is recorded, run all six repaired development cells in
the declared rate-major, seed-minor order. The original failed artifact remains reported
separately; no repaired artifact may overwrite it or be described as a rerun of the original
frozen implementation.

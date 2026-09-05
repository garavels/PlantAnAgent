# Constrained-RL capacity repair clarification 002 — canonical host thread settings

Status: **frozen after re-audit and before host-identity code changes or repaired execution**

Date: **2026-09-05**

Clarification identifier: **`plant-beta-constrained-rl-capacity-repair-clarification-002`**

## Finding

Clarification 001 requires the deterministic pre-run identity to bind host capacity and forbids
observed controller outcomes. The existing host record copied `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and `NUMEXPR_NUM_THREADS` as unrestricted strings.
An arbitrary string could therefore carry outcome, timing, or operator commentary into both the
canonical report and deterministic digest while pretending to be a thread-count setting.

No repaired capacity execution has occurred. This is an evidence-representation repair only; it
does not change the measured workload, subprocess environment, host, learner, seed, or projection.

## Required representation

For each frozen thread environment variable, host capture must record either:

- `null` when the variable is absent; or
- a JSON integer parsed from one canonical positive base-10 string in the inclusive range
  `1..2,147,483,647`.

Signs, leading zeros, whitespace, separators, lists, expressions, assignments, annotations,
non-ASCII digits, zero, negative values, and values above the bound must fail before output-root
creation. Verification must require the same integer-or-null representation and must independently
recapture it. The benchmark must not silently rewrite or inject these environment variables into
the child; the record describes the environment already selected by the operator/runtime.

## Required evidence

Before repaired execution, adversarial tests must prove that valid absent and canonical-positive
settings are retained exactly, while representative outcome text, timestamp text, whitespace,
leading-zero, signed, zero, negative, list, and overflow forms are rejected. The production and
synthetic report schemas remain V2 and synthetic V1 respectively; both must bind this clarification
through the protocol identity.

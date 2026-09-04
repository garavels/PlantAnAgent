# Absolute-setpoint service-MPC mechanism result

Date: **2026-09-04**

Source revision: `a968a343f50e5724ea74b75cc3debd1509fc1d80`

Result: **all S1--S8 mechanism gates pass; actuation architecture accepted; no controller
superiority claim**

## Evidence boundary

This is the single rate-35/seed-0 development mechanism cell registered before execution in
`CAUSAL_SERVICE_MPC_SETPOINT_EXECUTION_REGISTRATION.md`. The cell and its heuristic result
had already been inspected. It is not a holdout, multi-seed replication, RL comparison,
field result, financial result, or commission claim. There was one execution and no retry.

The runner executed in a clean detached checkout of the admitted implementation revision.
It completed in 166.057 seconds and wrote the preregistered external artifact. The artifact
was then reopened under the repository verifier, which recomputed the report digest, run and
evidence-graph digests, source relationships, S1--S8, and status.

## Frozen gate result

| Gate | Result | Observed |
|---|---|---|
| S1 execution | **Pass** | 288/288 closed decisions, 576/576 successful solver stages, 4,320/4,320 minute applications, zero failures and zero fallback. |
| S2 no overshoot | **Pass** | Maximum pre-guard target overshoot `4.55e-13 kW`; zero overshoot minutes; zero simultaneous charge/support requests. |
| S3 reachable tracking | **Pass** | Zero reachable-tracking violations, zero unattributed deficits, maximum decomposition error `4.55e-13 kW`. |
| S4 mechanism | **Pass** | 102,600 s productive operation, zero aborted starts, zero forced stops. |
| S5 safety | **Pass** | Zero reserve penalty, shutdown, or out-of-range decision; minimum SOC 57.76%, terminal SOC 99.53%. |
| S6 accounting | **Pass** | Zero accounting, chain, target-decomposition, or support-authority violations. |
| S7 direction | **Pass** | 2,520.000 kg delivered versus the frozen 647.714 kg prior MPC result, a +1,872.286 kg diagnostic difference. |
| S8 latency | **Pass** | Controller p99 0.988 s, max 1.401 s; dispatcher p99 0.135 ms, max 0.352 ms. |

The result establishes the frozen mechanism hypotheses: absolute targets remove additive
overshoot, the minute dispatcher maintains every reachable pre-guard target, the guard stays
authoritative, and discrepancies at the plant remain attributable.

## What changed relative to the failed MPC

The earlier share-action MPC delivered only 647.714 kg in this exact exogenous episode and
repeatedly lost the electrolyser startup commitment. The setpoint controller delivered all
2,520 kg, had three cold starts, ten hot starts, three controlled stops, and no aborted or
forced stop. This is strong evidence that the dominant prior failure was the lossy
optimizer-to-actuator interface rather than infeasibility of the causal MILP.

The new evidence also separates remaining mismatches. The dispatcher requested the target
within numerical precision for every minute. The plant accepted 1,210.584 kWh less
electrolyser energy than targeted over 76 ramp/storage-refusal or rerouting minutes. Resource
shortfall and guard reduction contributed effectively zero to that deficit, and the complete
decomposition closes to `2.65e-14 kWh`. These 76 residuals are now model-correction data,
not hidden controller failure.

## Same-cell heuristic comparison

The registered episode has the same manifest and exogenous SHA-256 identities as the
previously retained A0 heuristic and future-information feasibility study. A0 and the
certified delivery ceiling both already delivered all 2,520 kg. Consequently, the new MPC
has **zero delivery gain** on this cell.

| Metric | Setpoint MPC | A0 heuristic | MPC minus A0 |
|---|---:|---:|---:|
| Delivered / requested kg | 2,520 / 2,520 | 2,520 / 2,520 | 0 |
| Commitments met | 12/12 | 12/12 | 0 |
| Reward | 11.9959 | 11.9253 | +0.0707 |
| Accepted process energy | 77.896 MWh | 75.268 MWh | +2.628 MWh |
| Process energy intensity | 30.911 kWh/kg | 29.868 kWh/kg | +1.043 kWh/kg (+3.49%) |
| Battery throughput | 10.463 MWh | 6.607 MWh | +3.856 MWh (+58.36%) |
| Battery discharge | 3.547 MWh | 2.014 MWh | +1.532 MWh |
| Curtailed solar | 3.025 MWh | 6.444 MWh | -3.418 MWh |
| Terminal battery SOC | 99.53% | 91.60% | +7.93 percentage points |
| Cold / aborted / forced starts | 3 / 0 / 0 | 1 / 0 / 0 | +2 / 0 / 0 |

The small reward advantage reflects the current synthetic objective and terminal inventory,
not a commercially established benefit. The MPC trades lower curtailment and more terminal
inventory for more conversion energy, battery cycling, and operating time. Since electricity
price, wear, degradation, and replacement cost are absent, the task cannot decide whether
that trade is economically desirable.

One planning-to-realization mismatch is especially important: the MILP planned only 28.500
kWh of battery support, while the minute dispatcher requested 3,602.453 kWh and the plant
applied 3,546.863 kWh to maintain absolute targets under realized solar. This authority was
safe and the battery ended nearly full, but the planning objective does not currently price
or accurately predict the realized cycling consequence.

## Architecture decision

Accept the layered actuation architecture:

```text
causal supervisory policy -> absolute process setpoints -> deterministic minute dispatcher
                           -> independent safety guard -> physical plant -> typed residual
```

Do not interpret this cell as evidence that MPC is better than the heuristic or RL. The
delivery objective is saturated by both A0 and MPC, and the controllers currently use
different command abstractions. A fair algorithm comparison requires heuristic, MPC, and RL
to issue the same typed setpoints through the same dispatcher and guard.

The result supports the original project diagnosis: the environment now validates a real
control interface, but its service objective remains too easy to rank competent policies.
Adding more controller complexity before adding decision pressure would reproduce the
earlier heuristic/RL tie under a more elaborate implementation.

## Next bounded work package

1. Implement a transparent setpoint heuristic as the common-interface incumbent and expose
   the identical setpoint door to RL. Preserve the historical share controller only as a
   compatibility reference.
2. Run the separately preregistered multi-seed mechanism matrix to establish that the new
   seam is robust, while keeping any comparison with legacy A0 explicitly descriptive.
3. Add one commercially interpretable source of decision pressure at a time. The first
   candidate should be time-varying electricity value plus explicit battery/equipment wear,
   because this result exposes a 58% battery-throughput difference at identical delivery.
4. Calibrate the new cost and degradation ranges from traceable industrial sources or a
   declared customer case, then freeze economic, safety, and service metrics before
   controller evaluation.
5. Train RL only after the common action contract and new stress suite exist. RL must improve
   profit, lifecycle cost, service tail risk, or robustness—not merely match saturated
   delivery.

HEPA remains a predictive/risk layer and natural language remains a bounded supervisory
interface. Neither should receive minute dispatch authority or be evaluated until the
common control benchmark exposes a measurable residual problem for that layer.

## Artifact custody

- external JSON: `/Users/filippogombac/.codex/visualizations/2026/08/31/01a05842-8e4b-73b1-8810-42296fdaca5b/v2_service_mpc_setpoint_a968a34_rate35_seed0.json`
- bytes: `82,419,626`
- raw file SHA-256: `6b90ea6eb907c338fdf1aaa695ace173ad4cbe0b23b6eb7fef18bb3c513b4cf0`
- report SHA-256: `dfb3bac1b19afd78306f40110502fadfacd585ca26d54855a952a28e0b1cb5cc`
- run-evidence SHA-256: `1060a46c275277025add905d2c6aa5345b5c8dfe3b7c2bc1ee7b720a0772a592`
- deterministic evidence-graph SHA-256:
  `dc0dbf78c4670a41473e9a2ccb8f406e73f009cfbde5f40ea5d81de8056407e9`

The 82.4 MB artifact retains all 288 controller events, 576 solver stages, 288 interval
executions, 4,320 minute applications, timing records, terminal metrics, source provenance,
and digest graph. It remains external so experimental evidence does not contaminate the
source revision.

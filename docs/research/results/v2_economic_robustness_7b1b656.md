# Economic task robustness tournament

Date: **2026-09-05**

Executable source: `7b1b656964dcac4a25c6f5d4d89fed8abf4d4e0e`

Result: **complete, execution-valid negative task-readiness result; R1--R3 pass, R4--R7 fail,
R8 reported**

## Executive conclusion

The alpha economic task is **not admitted for substantive RL training**. All 20 registered
paired cells and all 40 controller trajectories completed safely with intact provenance,
accounting, authority, and physical evidence. The two six-hour strata are service-comparable
and discriminate the controllers in all ten cells. Both 24-hour strata, however, are
service-comparable in **0/5 cells**. The service MPC delivers only 70.8--73.7% of requested
methane on average while the heuristic delivers 98.0--100%.

This failure is useful. It demonstrates that the longer episode exposes intertemporal control
pressure that the six-hour endpoint hides. It also localizes the immediate defect: the frozen
MPC has a six-hour lookahead, sees only one six-hour commitment at a time, and treats finished
CH4 and convertible H2/CO2 as equal terminal inventory even though only the electrolyser can
receive battery support. It therefore fails to manufacture enough saleable methane before the
sunless final delivery window. Training RL now would confound economic control with a known
benchmark horizon defect and a coefficient-sensitive synthetic objective.

The next bounded work is a newly versioned, preregistered full-day service benchmark with
lookahead through the next renewable window and intervening commitments, explicit product-form
inventory adequacy, and lexicographic service protection. Seeds 20--24 are now inspected
development seeds and cannot become validation evidence.

## Evidence boundary and provenance

The tournament was frozen in `ECONOMIC_ROBUSTNESS_TOURNAMENT_PROTOCOL.md`, implemented and
committed, and then bound to one exact execution in
`ECONOMIC_ROBUSTNESS_TOURNAMENT_REGISTRATION.md`. It ran once, sequentially, from a clean
detached worktree. H ran before M in every cell. No cell was retried, substituted, or selected
after outcome.

| Field | Retained value |
|---|---|
| Report status | `complete_gate_failed` |
| Source revision | `7b1b656964dcac4a25c6f5d4d89fed8abf4d4e0e` |
| Runner contract | `6f789629393188f406c929305721aa257e33e329fc3b5b85da9e5d62d77d7d38` |
| Protocol SHA-256 | `6fb5eb11933ba5ee601849210b4e61278e4cec2ddb97c8e31b05817ee7141412` |
| Dependency-lock SHA-256 | `080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd` |
| Clean worktree identity | `eb34c2f8415f865ca6626a5e37e276441f3401e6d2c8cfdda3d4a13b82c5bbd6` |
| Runtime | CPython 3.12.7, macOS arm64 |
| UTC interval | 2026-09-05 06:36:55.269212 to 06:41:44.100754 |
| Runner elapsed time | 288.832 s |
| Matrix | 20 paired cells, 40 trajectories |
| Execution volume | 2,400 controller decisions, 36,000 one-minute applications |
| Artifacts | 20 complete cell files plus one report; 220,620 KiB on disk |
| Raw report SHA-256 | `26c9ab4a85693dab05d93f7a1b65e86564c74bfe667ee54813901a466d1e0755` |
| Report-domain SHA-256 | `e0fe451861cda01f0a788adafed9bf2767f778d0141d7c76147b6eb0f3831f7f` |
| Deterministic evidence SHA-256 | `cff97bfc678f0b7ca47c516f3f7a80e39434aae83a6597576b1539eba5110b96` |

The runner's own verifier and a separate post-run invocation of the deep whole-directory
verifier independently reproduced the report, cell ordering, raw/full/deterministic hashes,
source gate, and R1--R8 result. The verifier rejected neither a cell nor an evidence chain.

## Frozen gate result

| Gate | Result | Observed |
|---|---|---|
| R1 complete execution | **Pass** | 20/20 cells, 40/40 arms, exact decisions/minutes, all M transitions and solver stages closed, zero fallback. |
| R2 identity/accounting | **Pass** | All paired manifests, artifact graphs, reward traces, additive components, interval drivers, and terminal liabilities close. |
| R3 safety/authority | **Pass** | Every stratum passes; zero shutdown, out-of-range decisions, forced stops, authority overshoot, or unattributed deficits. |
| R4 service comparability | **Fail** | Six-hour strata 5/5 and 5/5; 24-hour strata 0/5 and 0/5. Four of five were required in every stratum. |
| R5 task expression | **Fail** | Six-hour strata 5/5 and 5/5; 24-hour strata 0/5 and 0/5 because task expression requires comparable service. |
| R6 nonterminal expression | **Fail** | Six-hour strata 5/5 and 5/5; 24-hour strata 0/5 and 0/5 because service remains non-comparable after removing methane liability. |
| R7 full-day robustness | **Fail** | Both 24-hour strata fail the R4 prerequisite despite complete settlements and valid execution. |
| R8 sensitivity | **Reported** | Default direction is stable in the confounded full-day cells but coefficient changes flip many equal-service six-hour cells. |

R5 and R6 are chained failures. They do not establish that the full-day trajectories lack
physical or economic separation; they establish that this separation cannot be interpreted as
an operating-cost tie-break while service differs materially.

## Controller outcomes

Values below are descriptive means over five development seeds. `M-H` is positive when M has
the higher net operating value. Only the six-hour rows satisfy the registered equal-service
condition.

| Horizon | Rate | H fill | M fill | H delivered | M delivered | Comparable | H net EUR | M net EUR | M-H EUR | M-H without CH4 liability |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 h | 35 kg/h | 100.00% | 100.00% | 210.000 kg | 210.000 kg | 5/5 | -225.529 | -375.362 | -149.833 | -228.949 |
| 6 h | 42 kg/h | 100.00% | 100.00% | 252.000 kg | 252.000 kg | 5/5 | -190.799 | -323.800 | -133.001 | -223.751 |
| 24 h | 35 kg/h | 100.00% | 73.66% | 840.000 kg | 618.738 kg | 0/5 | 605.570 | -3,952.274 | -4,557.844 | -5,427.167 |
| 24 h | 42 kg/h | 97.97% | 70.82% | 987.527 kg | 713.840 kg | 0/5 | 278.305 | -5,169.494 | -5,447.800 | -6,686.385 |

The per-seed full-day delivery deficit is consistent:

- at 35 kg/h, M delivers 215.680--236.226 kg less than H in every seed;
- at 42 kg/h, M delivers 200.823--308.744 kg less than H in every seed;
- H meets all four rate-35 commitments in every cell and all four rate-42 commitments in four
  cells; and
- M meets only two commitments in every rate-35 cell, two in four rate-42 cells, and one in the
  remaining rate-42 cell.

The raw net differences in the 24-hour rows are not controller-cost estimates. Mean M-minus-H
unmet-delivery penalties alone are -4,425.239 EUR at rate 35 and -5,473.728 EUR at rate 42;
lower delivered-service value contributes another -885.048 and -1,094.746 EUR. M's favorable
methane-endowment term offsets 869.323 and 1,238.585 EUR respectively. Removing that liability
makes the full-day gap more negative rather than resolving the service confound.

## Full-day failure mechanism

The retained MPC decisions expose the same sequence in every 24-hour cell:

| Delivery ordinal | Rate 35 outcome | Rate 42 outcome |
|---:|---|---|
| 1 | complete in 5/5 | complete in 5/5 |
| 2 | complete in 5/5 | complete in 4/5 and 0.103 kg short in one |
| 3 | partial: 183.774--204.320 of 210 kg | partial: 195.359--247.604 of 252 kg |
| 4 | zero in 5/5 | zero in 5/5 |

This is not an energy-exhaustion or solver-failure signature. Immediately after the third
settlement, M has zero CH4 in all ten full-day cells but retains:

- 7.122--8.000 MWh battery energy;
- 177.575--470.290 kg H2; and
- 976.664--2,543.948 kg CO2.

Three frozen design choices explain the stranded state:

1. M has a six-hour planning horizon and the commitment period is also six hours. Its inferred
   delivery set therefore contains only the next settlement. It cannot value preparation for
   the following sunless obligation.
2. The secondary objective rewards terminal CH4 plus convertible H2/CO2 as one
   methane-equivalent quantity. It does not distinguish immediately deliverable product from
   feedstock that still requires conversion power and time.
3. The authoritative plant and common setpoint door expose battery support only for the
   electrolyser. DAC and methanation consume contemporaneous routed solar. Once the final
   six-hour window is dark, the accumulated battery and feedstocks cannot be converted to CH4.

M therefore optimizes the declared six-hour problem correctly, but that local optimum is not a
feasible full-day service strategy. The longer environment is doing useful work: it reveals a
causal inventory-horizon problem that a single-settlement episode cannot express.

The common physical integration itself survives the test. Across all cells M has zero failed
decisions, fallback actions, evidence-chain violations, requested-target overshoots,
reachable-tracking violations, or unattributed target deficits. The largest observed M decision
time is 557.907 ms and the largest solver-stage time is 307.866 ms, far below the 15-minute
control interval. The largest dispatcher time is 0.594 ms. Performance failure is therefore
eligible evidence rather than an execution artifact.

## Six-hour economics and assumption sensitivity

The one-cell seed-0 mechanism result favored M by +382.532 EUR, driven by avoided methane
liability. That direction does not replicate on fresh six-hour development seeds: H has higher
default value in all ten equal-service cells. This is still not a robust controller-selection
result because the direction changes under registered coefficient ablations.

| Equal-service stratum | Default M-H | No CH4 liability | No declared wear | Service + solar only | M higher at CH4 liability x2 |
|---|---:|---:|---:|---:|---:|
| 6 h, 35 kg/h | -149.833 EUR | -228.949 EUR | +87.259 EUR | +8.144 EUR | 2/5 cells |
| 6 h, 42 kg/h | -133.001 EUR | -223.751 EUR | +100.694 EUR | +9.944 EUR | 2/5 cells |

Under the default ledger, M's extra battery throughput and electrolyser exposure outweigh its
terminal-inventory and solar-opportunity terms. Remove declared wear and M becomes higher in
four of five rate-35 cells and all five rate-42 cells. Retain only service and solar opportunity
and M is higher in three of five and five of five. Double methane-endowment liability and two
cells in each stratum also flip to M. These are algebraic sensitivities over synthetic
coefficients, not calibrated commercial scenarios.

The result rejects two tempting narratives:

- the seed-0 M advantage was not a stable operating-value advantage; and
- the fresh-seed H direction is not yet a commercially meaningful advantage because it depends
  strongly on uncalibrated wear and endowment prices.

## Research and architecture decision

Do not train or tune RL on `plant-env-v2-economic-wear-alpha` as a substantive contender yet.
Do not tune the existing MPC weights against these outcomes and call the result a replication.
The registered failure points to a new versioned work package:

1. make service a lexicographic or constrained requirement before economic optimization;
2. give the causal benchmark lookahead through the next renewable window and every intervening
   commitment, within the already public forecast support;
3. distinguish deliverable CH4 from convertible feedstock in terminal adequacy and expose the
   time/power required to convert it;
4. retain the common four-setpoint command, dispatcher, independent guard, and complete evidence
   path;
5. test the repair first on the now-contaminated development seeds 20--24 under a new protocol;
   and
6. reserve new weather, demand, tariff, plant, fault, and equipment regimes for later formal
   evaluation.

Once an equal-information benchmark maintains comparable service over a full day, the economic
layer can progress from synthetic additive terms to calibrated tariff, offtake, degradation,
inventory, and availability costs. A bounded RL baseline is then justified only if it can
improve cost, tail service, robustness, or latency under the same service constraint and the
same physical authority.

This advances the product thesis from “RL controls a simulator” to a more defensible industrial
problem: auditable scheduling of a renewable-to-product plant under intermittent supply,
multi-stage inventory conversion, delivery obligations, equipment limits, and uncertain
economics. The current result validates the need for that problem formulation; it does not yet
validate an RL, controller, savings, profitability, commission, or commercial claim.

## Artifact custody

The 21 immutable artifacts remain outside Git at:

`/Users/filippogombac/.codex/visualizations/2026/08/31/01a05842-8e4b-73b1-8810-42296fdaca5b/v2_economic_robustness_7b1b656`

This repository stores the compact interpretation and cryptographic identities, not a duplicate
of the approximately 215 MiB evidence directory. Seeds 20--24 and every controller outcome in
that directory are development evidence forever.

# Environment V2 full-day horizon repair — registered result

Status: **execution-valid positive mechanism result; F1--F8 passed**

Date: **2026-09-05**

## Conclusion

Extending only the existing causal service MPC's planning horizon from 6 to 24 hours repaired
the known fourth-commitment failure on all ten registered 24-hour development cells. M24 met all
four commitments and the complete request in every cell, with effectively zero unmet methane.
The frozen six-hour controller met only 1.8--2.0 commitments on average and left 221.262 kg or
294.160 kg unmet, depending on delivery rate.

This isolates insufficient lookahead as sufficient to explain and repair the observed failure in
these contaminated development cases. It does not establish that the environment is realistic,
that M24 is economically optimal, or that RL adds value. The result instead creates a
service-credible, horizon-aware model-based baseline against which the already completed
controller-neutral RL integration can be evaluated under a separately frozen protocol.

## Provenance and custody

| Field | Executed value |
|---|---|
| Executed source revision | `d18082594b7fea692c9fff146c54d9b0f2f31733` |
| Source branch | `codex/m24-horizon-execution` |
| Historical source revision | `7b1b656964dcac4a25c6f5d4d89fed8abf4d4e0e` |
| Runner contract | `d70857ef63a838ce9eb94767739e7492517983665977039e9d8d39547c7faae8` |
| M24 settings | `67877c94438fc1bf1bd3ad564f17d860f77cea10b82f07bba0d52b6a6f6b1778` |
| Historical M6 settings | `28dd0dbbd732f4defcc700e3fcb91e59b2d14d6df7bac728963090e37ba48b0f` |
| Historical raw report | `26c9ab4a85693dab05d93f7a1b65e86564c74bfe667ee54813901a466d1e0755` |
| Dependency lock | `080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd` |
| Output directory | `/Users/filippogombac/.codex/visualizations/2026/08/31/01a05842-8e4b-73b1-8810-42296fdaca5b/v2_full_day_horizon_repair_d180825` |
| Raw report SHA-256 | `44a11d071e3f4820757c2174c35119fe99aa4016a0cfdc73a2e6196e53c32974` |
| Report-domain SHA-256 | `eb3a68232ddef74d96e391c4979ab7aa2c15238b1ca08a4a0a4162fc04a664c4` |
| Deterministic-evidence SHA-256 | `1d12982aed4f9e32f8868393a1b226525eb268023bdf77b7b286e3f872307c12` |
| Report status | `complete_all_gates_passed` |
| Runtime | CPython 3.12.7; macOS 26.6.2 arm64 |
| Retained artifacts | 10 cell files plus `report.json`; about 140 MB |
| Timed matrix window | 2026-09-05 09:35:59Z to 09:58:59Z |
| Recorded matrix elapsed | 1,380.627498334 s |

The one-shot runner deeply admitted the complete immutable 215 MB H/M6 reference directory
before creating the destination, executed cells sequentially, exclusively published and verified
each one, repeated source/reference admission after execution, published the report, and invoked
the independent verifier. A second detached verifier invocation later reopened both evidence
directories and returned successfully. The registered run and the second verification both
returned exit code zero. The detached worktree remained clean at the exact revision.

## Frozen gates

| Gate | Result | Exact observation |
|---|---:|---|
| F1 source/reference custody | Pass | Every source, protocol, clarification, lock, historical report/cell, directory, and destination check passed. |
| F2 complete execution/evidence | Pass | 10/10 cells; 960 decisions; 14,400 physical minutes; 1,920 successful solver stages; no fallback or incomplete transition. |
| F3 identity/accounting/safety/authority | Pass | Every identity and additive ledger closed; all prohibited violation counts were zero. |
| F4 horizon fidelity | Pass | Every first decision used the unchanged public issue, 96 consecutive intervals, and four targets at boundaries 24/48/72/96; only planning horizon differed from M6. |
| F5 service repair | Pass | Full request: 5/5 at both rates. H-comparable service: 5/5 at 35 kg/h and 4/5 at 42 kg/h. Positive fourth delivery: 10/10. |
| F6 mechanism magnitude | Pass | Mean M24 unmet passed the 90% reduction rule at both rates; post-third CH4 adequacy: 5/5 at both rates; all final material balances closed. |
| F7 economic expression | Pass | Both expression rules: 5/5 at 35 kg/h and 4/5 at 42 kg/h; the one non-comparable cell stayed visible but ineligible. |
| F8 operational latency | Pass | Worst cell controller p99/max 0.965/0.965 s; worst dispatcher p99/max 0.587/2.955 ms; exact claim boundary retained. |

## Service result

The values below are five-seed means. Exact tolerance checks use the unrounded values in the
artifact.

| Rate | Arm | Requested (kg) | Delivered (kg) | Unmet (kg) | Commitments met |
|---:|---|---:|---:|---:|---:|
| 35 kg/h | H | 840 | 840.000 | 0.000 | 4.0 |
| 35 kg/h | M6 | 840 | 618.738 | 221.262 | 2.0 |
| 35 kg/h | M24 | 840 | 840.000 | effectively 0 | 4.0 |
| 42 kg/h | H | 1,008 | 987.527 | 20.473 | 3.8 |
| 42 kg/h | M6 | 1,008 | 713.840 | 294.160 | 1.8 |
| 42 kg/h | M24 | 1,008 | 1,008.000 | 0.000 | 4.0 |

M24 improved delivered methane over M6 in every cell. Its mean improvements were 221.262 kg at
35 kg/h and 294.160 kg at 42 kg/h, exactly closing the corresponding mean M6 deficits within
floating-point tolerance. Every fourth settlement delivered the complete 210 kg or 252 kg.
Every cell already held at least that request as CH4 immediately after settlement three, and the
production-aware final material balance agreed with both interval and residual delivery evidence.

| Cell | M6 delivered | M24 delivered | Fourth delivery | Post-third CH4 | H-comparable | Default M24-H value | No-methane M24-H value |
|---|---:|---:|---:|---:|---:|---:|---:|
| r35 s20 | 623.101 | 840.000 | 210.000 | 210.000 | yes | +85.659 EUR | -192.852 EUR |
| r35 s21 | 624.320 | 840.000 | 210.000 | 215.513 | yes | +255.705 EUR | -154.475 EUR |
| r35 s22 | 620.810 | 840.000 | 210.000 | 210.000 | yes | +311.723 EUR | -165.765 EUR |
| r35 s23 | 603.774 | 840.000 | 210.000 | 269.275 | yes | -120.426 EUR | -120.426 EUR |
| r35 s24 | 621.685 | 840.000 | 210.000 | 210.000 | yes | +156.758 EUR | -221.958 EUR |
| r42 s20 | 710.370 | 1,008.000 | 252.000 | 352.508 | yes | +47.343 EUR | -172.075 EUR |
| r42 s21 | 751.604 | 1,008.000 | 252.000 | 448.636 | yes | +211.829 EUR | -153.730 EUR |
| r42 s22 | 704.810 | 1,008.000 | 252.000 | 262.676 | no | ineligible | ineligible |
| r42 s23 | 699.256 | 1,008.000 | 252.000 | 266.007 | yes | +119.015 EUR | -141.966 EUR |
| r42 s24 | 703.161 | 1,008.000 | 252.000 | 340.633 | yes | +125.706 EUR | -186.315 EUR |

The exceptional r42/s22 cell is not a failure: M24 delivered the full 1,008 kg while H delivered
905.633 kg. Because service differed by 102.367 kg, the frozen protocol correctly excluded that
cell from an equal-service economic comparison while retaining it in every service denominator.

## What the mechanism evidence says

The result supports the preregistered causal diagnosis. M6 sees one six-hour commitment at a
time, assigns equivalent terminal value to methane and convertible feedstock, and reaches the
sunless final period without enough deliverable product. M24 receives no new private information
and uses no new objective term; the existing recurring-target inference simply exposes all four
already public commitments inside its first optimization. The resulting trajectory holds the
complete fourth request as finished methane before the final interval in every seed.

The change is not merely more aggressive production. Relative to equal-service H cells, M24
used on average 4.294 MWh less accepted process energy at 35 kg/h and 4.417 MWh less at 42 kg/h.
It also used 2.539/2.391 MWh more battery throughput and 2.865/2.913 more equivalent electrolyser
operating hours, accepted about 0.210/0.206 MWh less solar, and reduced terminal methane-
equivalent deficit by 77.245/72.374 kg. The controller trades different inventory, cycling, and
equipment-exposure profiles to secure product-form adequacy.

The battery still ended near full: mean terminal energy was 7.986 MWh at 35 kg/h and 7.984 MWh
at 42 kg/h. This reinforces that the historical M6 service failure was informational/temporal,
not caused by physical exhaustion of stored energy.

## Economics: discriminative, not calibrated

F7 passed because operating behavior remained materially distinguishable at equal service both
under the default ledger and after removing methane-endowment liability, with nonterminal raw
drivers. It did **not** establish a stable economic winner.

- At 35 kg/h, default M24-minus-H value averaged +137.884 EUR; four cells favored M24 and one
  favored H. Removing methane-endowment liability made all five deltas negative, averaging
  -171.095 EUR.
- At 42 kg/h, the four eligible default deltas all favored M24 and averaged +125.973 EUR.
  Removing methane-endowment liability made all four negative, averaging -163.522 EUR.
- Doubling the declared wear coefficients changed the mean delta to -57.920 EUR at 35 kg/h and
  -62.291 EUR at 42 kg/h, with H favored in three cells of each eligible stratum.
- With only service and solar opportunity cost retained, all nine eligible cells favored M24,
  but by much smaller mean margins of 24.709 EUR and 24.742 EUR.

The default positive direction is therefore strongly influenced by the synthetic methane-
endowment liability, while the no-methane ledger penalizes M24's extra battery throughput and
equipment exposure. The environment now expresses real operational trade-offs, but coefficient
calibration and uncertainty analysis are mandatory before reward optimization or savings claims.

## Safety, authority, and operational cost

Every cell reported zero shutdowns, forced stops, out-of-range decisions, chain/accounting
violations, requested-target overshoots, reachable-tracking violations, support-authority
violations, and unattributed target deficits. There was no controller fallback.

The safety/plant layers were active rather than bypassed. M24 recorded 3--7 guard corrections and
23--46 plant refusals/target deficits per cell; all deficits were independently reconstructed and
attributed, and every cell still met service. This is the intended layered behavior: planning
proposes setpoints, the common dispatcher and guard retain final authority, and evidence explains
the difference between request and physical application.

M24's worst per-cell controller p99 and maximum were both 965.081 ms, below the 2 s and 5 s
limits. The worst dispatcher p99 was 0.587 ms and worst absolute dispatcher latency was 2.955 ms,
below the 5 ms and 25 ms limits. Each cell reported SciPy 1.18.1 and HiGHS 1.12.0, matching the
historical evidence.

## Architecture and next direction

This result closes the immediate task-readiness blocker but raises the benchmark bar. A
horizon-aware model-based controller can now solve the deterministic development commitments, so
an RL policy should not be credited merely for delivering all product. The next comparison must
test whether learning improves economic/risk trade-offs under information and dynamics that M24
cannot perfectly encode.

The next bounded program is:

1. preregister a finite PPO baseline using the completed controller-neutral 62/4 economic task,
   with training/development seeds separated from genuinely new held-out weather, tariff, demand,
   initial-state, and plant-parameter regimes;
2. implement a tariff-aware economic MPC as the principal model-based comparator, while retaining
   H, M6, and M24 for ablation;
3. calibrate or uncertainty-bound service value, wear, cycling, inventory, and energy coefficients
   before selecting a policy from scalar return;
4. version Environment V2 beta around cyclic multi-day operation, forecast revisions, correlated
   weather/tariff uncertainty, sensor imperfections, equipment faults/degradation, grid or backup
   energy constraints, and product-quality/storage logistics; and
5. only after held-out evidence, add a live V2 orchestration surface where HEPA provides forecast/
   anomaly context and the language supervisor explains, constrains, and escalates decisions
   without becoming the millisecond physical controller.

The commercially relevant problem is reliable, auditable scheduling of intermittently powered
industrial production under delivery obligations, uncertain renewable energy, inventory limits,
equipment wear, and human safety authority. This run demonstrates one useful piece: long-horizon
planning can prevent a large service failure with public information and sub-second decisions.
Commission or customer value still requires held-out savings/reliability evidence, calibrated
site economics, hardware/operator validation, and a credible deployment path.

## Artifact identities

| File | Raw SHA-256 |
|---|---|
| `01_h24-r35-s20.json` | `c7b6552425b1633176bfbfeff2ca51d453ea2ad59257e694e4398c2806f791e9` |
| `02_h24-r35-s21.json` | `e78a97f6bf291230d523422b8b709e96d678b580b2e3a351bd69b5bed256f733` |
| `03_h24-r35-s22.json` | `04ed8187c652845ab7be7b43be20e4765c5001ff0c8ef0224a3e99abd6bbfdcf` |
| `04_h24-r35-s23.json` | `56f3a05f5ed9c9a4bdb45c8110ffb72e4c0752dbf7ebd7535606908d1ee68eb8` |
| `05_h24-r35-s24.json` | `78492f940242284e667465e8e24cba9bd3792394c95fdad706e0d7309d8daef6` |
| `06_h24-r42-s20.json` | `60dbb02360f0453c5308576b494dff9be50f0f40af7201f69d5e08fbac642369` |
| `07_h24-r42-s21.json` | `01dc7c6e37a82cd5109d0ab594ab81bae274de94163664e38f9f3e051ce0fcf2` |
| `08_h24-r42-s22.json` | `fa469693fe6f384ecbee98cbe9f6a598748852ed8e2e349b2b9e356d9070fe29` |
| `09_h24-r42-s23.json` | `a592a15a9b40cdf97c2cf6a823f1a20ff3a14afbdb9118b1e1e743a0483fe38e` |
| `10_h24-r42-s24.json` | `d1aa987fd061178abb2bb2b40dbb15f8a42f647554526c8ee07fe31b6a69c9fb` |
| `report.json` | `44a11d071e3f4820757c2174c35119fe99aa4016a0cfdc73a2e6196e53c32974` |

## Claim boundary

> Contaminated development mechanism only; no RL, controller-superiority, calibrated-economic,
> held-out, site, commercial, savings, commission, or generalization claim.

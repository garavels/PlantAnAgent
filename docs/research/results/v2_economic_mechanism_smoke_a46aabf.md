# Economic/wear task-expression mechanism result

Date: **2026-09-04**

Source revision: `a46aabf9cd3d8ee7387055c07b1c689e2d53f74f`

Result: **E1--E5 pass; the task separates equal-service controller trajectories on the
registered development cell; no controller, RL, commercial, or commission superiority claim**

## Evidence boundary

This is the single six-hour, rate-35, seed-0 development cell frozen before execution in
`ECONOMIC_MECHANISM_SMOKE_PROTOCOL.md`. It asks whether the new economic/wear ledger can
distinguish the common-door heuristic H from the existing service MPC M when safety and methane
delivery are equal. It is not a holdout, multi-seed comparison, tariff-aware optimization test,
RL result, calibrated financial model, or proof that M is generally better.

The first shell launch using `uv run --frozen` stopped before Python initialized because the
global uv cache was outside the writable sandbox. It created no artifact, reset no environment,
and opened no controller trajectory. The committed runner was then launched once through the
already-installed project `.venv`, against the same clean source and dependency-lock identity.
That registered execution completed in 5.514 seconds and exclusively published the external
artifact. No parameter, controller, source, or protocol changed between preflight and result.

The artifact was reopened independently. Its raw and domain-separated report hashes were
recomputed; the frozen contract was compared byte-for-value; the heuristic evidence digest and
complete MPC evidence graph were recomputed; the MPC run verifier passed; and E1--E6 were
re-evaluated from detached arm payloads with exact equality to the published gates.

## Frozen gates

| Gate | Result | Observed |
|---|---|---|
| E1 execution | **Pass** | Both arms completed 24/24 decisions and 360/360 physical minutes, terminated normally, and reported no failure. |
| E2 safety and authority | **Pass** | Both retained 24 interval executions and 360 minute applications with zero shutdown, out-of-range decision, or forced stop. M's deep evidence graph reports 48/48 successful solver stages, zero fallback, zero chain/accounting/tracking violation, and seven fully attributed plant-refusal minutes. |
| E3 comparable service | **Pass** | H and M each delivered exactly 210.000 kg with zero unmet methane; both deltas are exactly zero. |
| E4 task expression | **Pass** | M minus H net operating value is **+382.532 EUR**. All five registered physical-driver comparisons are reportable and four are material. |
| E5 accounting | **Pass** | Reward trace, component sum, interval/episode drivers, terminal liabilities, frozen identity, and paired identity close for both arms. |
| E6 interpretation | **Reported** | M is descriptively higher value on this one development cell only. |

## Controller outcomes

| Metric | H heuristic | M service MPC | M minus H |
|---|---:|---:|---:|
| Delivered / unmet methane | 210.000 / 0.000 kg | 210.000 / 0.000 kg | 0 / 0 kg |
| Net operating value | -556.315 EUR | -173.783 EUR | **+382.532 EUR** |
| Accepted solar | 10,825.869 kWh | 10,899.793 kWh | +73.924 kWh |
| Accepted process energy | 9,033.996 kWh | 7,782.135 kWh | -1,251.861 kWh |
| Battery throughput | 1,924.532 kWh | 3,371.180 kWh | +1,446.648 kWh |
| Electrolyser equivalent exposure | 3.300 h | 7.725 h | +4.425 h |
| Terminal CH4-equivalent deficit | 135.754 kg | 0.000 kg | -135.754 kg |
| Terminal battery energy / SOC | 6,732.181 kWh / 84.15% | 8,000.000 kWh / 100.00% | +1,267.819 kWh / +15.85 pp |
| Terminal CH4 | 511.960 kg | effectively 0 kg | -511.960 kg |
| Terminal H2 | 0.000 kg | 337.967 kg | +337.967 kg |
| Guard interventions | 6 | 0 | -6 |
| Cold / hot / forced starts | 1 / 1 / 0 | 1 / 1 / 0 | 0 / 0 / 0 |

M did not win this cell by serving more product, seeing tariff information, or avoiding a hard
safety event. It met the same commitment while ending with enough H2 potential to restore the
initial CH4-equivalent endowment and a full battery. H converted much more feedstock to methane,
delivered the same 210 kg, and ended 135.754 kg CH4-equivalent below its starting endowment.

The physical trajectories are meaningfully different. M used 13.86% less process energy and
curtailed 73.924 kWh less solar, but moved 75.17% more energy through the battery and incurred
134.09% more declared electrolyser exposure. The task is therefore no longer a scalar delivery
tie over physically indistinguishable behavior.

## Exact economic decomposition

| Component | H | M | M minus H |
|---|---:|---:|---:|
| Delivered-service value | +840.000 EUR | +840.000 EUR | 0.000 EUR |
| Unmet-delivery penalty | 0.000 EUR | 0.000 EUR | 0.000 EUR |
| Solar opportunity cost | -684.551 EUR | -683.188 EUR | +1.362 EUR |
| Battery-throughput reserve | -109.348 EUR | -191.544 EUR | -82.196 EUR |
| Electrolyser-life reserve | -59.400 EUR | -139.050 EUR | -79.650 EUR |
| Methane-endowment liability change | -543.016 EUR | 0.000 EUR | +543.016 EUR |
| Battery-endowment liability change | 0.000 EUR | 0.000 EUR | 0.000 EUR |
| **Net operating value** | **-556.315 EUR** | **-173.783 EUR** | **+382.532 EUR** |

The accounting says exactly why the task separated the arms. M's 543.016 EUR avoided methane
endowment liability is partially offset by 161.846 EUR of additional battery and electrolyser
reserves; the small solar-cost improvement closes the 382.532 EUR gap. The methane-liability
term alone is 142% of the net difference, so the direction is economically interpretable but
not yet robust to assumption or horizon choice.

Both absolute net values are negative under the synthetic alpha assumptions. That does not mean
the plant or either controller destroys commercial value: the six-hour ledger omits capital
recovery, product-market structure, grid services, operating overhead, and calibrated customer
prices. Conversely, a positive controller delta here is not a savings estimate.

## Mechanism conclusion

The narrow mechanism hypothesis passes. The economic task can break a saturated delivery tie
using additive, auditable physical drivers while preserving common actuator authority and exact
accounting. This is the first evidence in the repository that the new task creates decision
pressure relevant to intertemporal inventory, energy acceptance, battery cycling, and equipment
exposure.

The result also exposes the next scientific risk. A six-hour endpoint with substantial sampled
inventory may reward retaining convertible H2 rather than completing methanation. The
CH4-equivalent liability was designed to make H2 and CO2 fungible at stoichiometric potential,
and its telescoping identity closes exactly, but its value coefficient and interaction with
actual delivered service remain synthetic. Multi-seed consistency, longer/cyclic horizons, and
predeclared coefficient ablations are required before calling this a stable optimization
problem rather than an endpoint-sensitive one.

M is still the delivery-oriented service MPC. It ignores tariff and every economic coefficient,
while H also ignores tariff and has less forecast/delivery context. This cell proves task
expression, not equal-information algorithm quality. An economic MPC, learned policy, and
heuristic should ultimately receive the same public information and action authority.

## Decision and next bounded experiment

E1--E5 authorize a separately frozen economic development tournament; they do not authorize an
unbounded PPO run. The next protocol should be written before opening any new controller seed
and should:

1. reserve seed 0 as inspected mechanism evidence and use a declared fresh development-seed set;
2. compare H and M on identical 35 and 42 kg/h episodes through their admitted setpoint doors;
3. report paired net value, service, safety, accepted energy, cycling, equipment exposure, and
   terminal inventories with distributions and worst cases rather than only means;
4. include predeclared objective ablations that isolate methane endowment, battery wear,
   electrolyser exposure, and tariff opportunity cost without tuning coefficients to outcomes;
5. test at least one longer or cyclic terminal horizon to detect endpoint deferral; and
6. define the residual-headroom gate that must pass before a substantive RL baseline is trained.

If the value separation is consistent, safe, service-comparable, and not carried solely by one
synthetic terminal term, the environment is ready for a bounded PPO baseline through the now
completed controller-neutral integration. If not, the economic model or horizon must be repaired
before learning compute is spent.

## Artifact custody

- external JSON: `/private/tmp/plantanagent_economic_smoke_a46aabf.json`
- bytes: `4,635,756`
- raw file SHA-256:
  `10711d88d7c22c5601ff4abd4282349d468a4e35c8019229a06445cb95637454`
- report SHA-256:
  `acca646c7b2705bd278d022fd687d34303ba9b9acef5a0ae56d3b8d9503b684b`
- heuristic evidence SHA-256:
  `ff2554b778e4e98787764946db17d4249d8e6d49b6e22b24bda28818d699fa72`
- MPC run SHA-256:
  `8bd6e0630c589ad077d6abaf495027c26fa63934b7495f2a7b7e320885ee8125`
- MPC evidence-graph SHA-256:
  `8623359f02d7fe2ca375b59cff89bb97b0a927575387dbf8607720eae1e0197e`
- contract SHA-256:
  `ac2d923dc5634b220452cc70e63fb2985ebffc8880fddac334aefb96854cec55`
- protocol raw SHA-256:
  `41c0ce9a1d49bcbfca34d88c0fb20b7fc7eb0ab91d15ed405aded512135cd795`
- dependency-lock SHA-256:
  `080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd`

The source gate records the same clean revision and worktree identity before and after execution.
The raw artifact remains outside Git; this compact report retains the admitted result and its
claim boundary without adding 4.6 MB of controller ledgers to the repository.

# Environment V2 delivery-alpha baseline result

Status: **formal baseline-only calibration; not an RL, MPC, or product-performance
result.**

## Question

Does replacing V1's terminal tank-filling objective with recurring methane delivery
create a measurable performance gap between transparent dispatch strategies, while
remaining feasible for the shipped heuristic?

The intended answer was neither “make the heuristic win” nor “prove RL is needed.” The
acceptance condition was a useful but limited decision frontier: idle and poorly chosen
fixed allocations should miss product commitments, while at least one credible baseline
should remain feasible. Forecast uncertainty, equipment commitment, an optimiser, and
RL were deliberately absent so this increment could be attributed to delivery alone.

## Frozen experiment

Command:

```text
UV_CACHE_DIR=/tmp/plant-an-agent-uv-cache uv run python -m plant_backend.rl.v2 \
  --seed-start 0 --count 20 --hours 168 --delivery-rate 55 \
  --json docs/research/results/v2_delivery_alpha_baselines_seed0_19.json
```

Provenance:

| item | identity |
| --- | --- |
| source revision | `1200a6171969f70486d911fa2858042571f08a48` |
| source dirty | `false` |
| dependency lock SHA-256 | `2c70ff89decad894dab67e079166cdec4723fc71b587b27a3185b67a86ebf7a3` |
| environment | `plant-env-v2-delivery-alpha` |
| interface digest | `7fc3808d1a393fb59ccd34c7b6cd31114530348a37fcb66f054604877951316f` |
| full checkpoint-contract digest | `fcc560585f03f2ba29faa5b21ccdf26d93ca861d97274bd72c0d3ebd9dd31129` |
| result JSON SHA-256 | `5d5b34f538f5ec8bb6ff015b80ccfec494a50f01b772606e81e745a5aa8b3711` |

All four arms saw the same immutable episode-manifest hash on every seed. Population
standard deviation (`ddof=0`) is used. The complete configuration, per-run records,
reward components, distributions, and pairing hashes are in the committed JSON.

## Result

Twenty seeds vary initial battery, H2, CO2, and CH4 inventories. Solar is the same
deterministic clear-sky profile in every run, which is why total available solar has
zero variance and why this is not a generalization test.

| controller | delivery fill mean ± std [min, max] | service level mean [min, max] | unmet kg mean ± std [min, max] | CH4 produced kg mean ± std | curtailment kWh mean ± std | minimum SOC mean [min, max] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shipped heuristic | 100.0% ± 0.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] | 0.0 ± 0.0 [0.0, 0.0] | 9364.4 ± 178.7 | 595 ± 469 | 21.9% [15.0%, 31.3%] |
| idle | 3.3% ± 0.9% [1.9%, 4.9%] | 1.2% [0.0%, 3.6%] | 8938.2 ± 83.5 [8788.3, 9068.2] | 0.0 ± 0.0 | 300802 ± 0 | 53.1% [32.6%, 64.6%] |
| fixed balanced mix | 71.4% ± 2.6% [66.8%, 76.5%] | 61.8% [53.6%, 67.9%] | 2639.4 ± 243.7 [2167.0, 3064.2] | 6298.7 ± 245.4 | 97887 ± 2826 | 53.1% [32.6%, 64.6%] |
| fixed production mix | 88.8% ± 2.6% [84.2%, 93.9%] | 85.2% [78.6%, 89.3%] | 1034.3 ± 243.7 [561.9, 1459.1] | 7903.9 ± 245.4 | 45400 ± 2826 | 53.1% [32.6%, 64.6%] |

The heuristic delivered all 9,240 kg requested in every run. It used the battery and
finished with 426.2 ± 181.2 kg CH4, whereas the fixed arms never discharged the battery
and ended with no methane. No arm entered a critical/offline plant state.

## Safety and action-authority finding

The new interval accounting found that successful delivery does not mean the requested
action passed through unchanged:

| controller | adjustment messages mean ± std | corrected physics steps mean ± std | request-to-guard L1 gap kWh mean ± std |
| --- | ---: | ---: | ---: |
| shipped heuristic | 4046.6 ± 49.5 | 2410.1 ± 15.3 | 3216.0 ± 74.2 |
| idle | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| fixed balanced mix | 4347.0 ± 0.0 | 4347.0 ± 0.0 | 30082.8 ± 0.0 |
| fixed production mix | 5768.0 ± 0.0 | 4347.0 ± 0.0 | 37979.0 ± 0.0 |

An adjustment message is not a safety violation: the independent guard prevented the
request from exceeding the current envelope. The magnitude does show that the present
share-based action and smart guard/router can make materially different requests converge
toward similar physical dispatch. Later controller comparisons must report this gap and
test whether absolute setpoints provide a cleaner authority contract.

## Decision

The delivery increment passes its narrow calibration condition: it eliminates V1's
one-way terminal tank ceiling and rejects idle and weak fixed policies. It does **not**
pass the complete Gate B in the roadmap:

- the heuristic still has 100% delivery service under repeated deterministic weather;
- only initial inventories vary between seeds;
- no forecast-aware controller, MPC, or perfect-information bound exists;
- memoryless process equipment creates no startup/ramp/degradation trade-off;
- 20 inventory seeds are not held-out sites, years, or weather regimes.

The next experiment therefore adds seeded autocorrelated renewable truth and strictly
causal issued forecasts before training RL. The contract rate must then be recalibrated.
After that, one technology-labelled electrolyser commitment state and a deterministic
economic MPC/perfect-foresight oracle are required. RL training is justified only when
those additions produce a nontrivial controllable gap.

## Empirical-data path

Synthetic weather is useful first because causal forecast leakage and ablations can be
tested exactly. It is not the final evidence source. The intended empirical benchmark
adapter will ingest multi-year site profiles with explicit timestamps and provenance.
The European Commission Joint Research Centre's PVGIS service is a suitable first open
source because it exposes multi-year hourly solar radiation and PV output through a
documented API. Hourly resolution is insufficient for every fast transient, so it will
support regime/site generalization while controlled higher-frequency perturbations or
plant data cover the 15-minute control cadence.

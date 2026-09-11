# Commission operating rehearsal

The single authorized product MPC rehearsal completed **168 hours / 672 decisions /
10,080 physical minutes** at clean implementation revision
`ccc86903cd7dbfcad0c2ebc2f93b7cc6c0f6ecaa`, on disposable fixture 9,000,034.
It ran once in 309.71 seconds, within the 1,200-second execution limit. No retry,
parameter search, training, development selection or held-out evaluation followed.
The prospective scope is [COMMISSION_OPERATING_SCOPE.md](../COMMISSION_OPERATING_SCOPE.md).

| Observation | Retained result |
| --- | --- |
| Controller | `operational-economic-mpc-v1` / product MPC |
| Methane delivered / requested | 5,938.0718 / 6,720 kg (88.3642%) |
| Methane unmet | 781.9282 kg |
| Fully met commitments | 22 / 28 |
| Operating modes | 667 Recovery, 5 Fallback, 0 Scheduled |
| Daily inventory reviews with deficits | 7 / 7 |
| Net simulated operating value | −18,929.62 EUR |
| Median / maximum controller computation | 413.43 / 1,646.40 ms |
| Guard interventions | 35 |
| Electrolyser aborted starts / forced stops | 0 / 0 |

These are descriptive observations from a synthetic presentation fixture. The
public recording is checksum- and sequence-validated; it is not a detached-verified
research comparison. No safety certification, commercial savings, RL superiority
or indefinite sustainable production follows. Minimum recorded battery SOC was
0.14999999999999997 against a 0.15 reserve, a floating-point-sized difference;
the raw metrics and their tiny reserve-breach term remain in the compact result.
The result also retains 617 shutdown steps and all request/application differences.

## Fallback and restoration

All five fallbacks record solver status 2 (infeasible), then return to a valid
Recovery plan at the immediately following decision. Four occur in the economic
second stage after the primary delivery stage succeeds; one occurs in the primary
stage. These statuses do not establish physical plant infeasibility. Their precise
numerical/model causes remain undiagnosed; no tuning or retry hid them.

| Issued elapsed hour | Failed stage | Completed replay boundary |
| --- | --- | --- |
| 31.50 | Economic cost | 127 |
| 32.50 | Economic cost | 131 |
| 56.50 | Economic cost | 227 |
| 128.50 | Economic cost | 515 |
| 145.25 | Delivery service | 582 |

Each failed interval publishes the guarded heuristic's actual targets, the typed
failure and null inventory/shortfall predictions. The subsequent public plan is
retained independently. Browser inspection of boundaries 127 and 128 verifies
Fallback with no invented schedule followed by Recovery with an 82.4 kg forecast
shortfall and explicit inventory deficits.

All opening endowments remain fixed throughout the run. Final battery inventory
is 1,272.97 kWh versus 3,993.58 kWh initially; H2 is 3.8744 kg versus 112.6731 kg;
CO2 is approximately zero versus 1,706.8181 kg; CH4 is zero versus 215.2820 kg.
The product continued, but did not restore inventories. “Recovery” identifies the
operating deficit state; it is not evidence that restoration eventually succeeds.

## Custody and reproduction

The 31,438,173-byte public recording remains outside Git at
`runs/commission-demo/5a25ba80-6227-4c07-8790-adc38f02a3a1.json`;
`MPC.attempt.json` is its exclusive attempt marker. Existing H and E-beta files
and markers remain unchanged. The GUI prefers the complete MPC recording at the
existing eight-hour cursor and retains access to both historical records.

- Recording content SHA-256: `a42cf8729324454fe785b575db584b97ebc5f8fb9537e987bcf24d347b9021bb`.
- Recording file SHA-256: `bc3a5e7381c9f0a45870b6e3d9200163d08856df256110f21cb6f6de8897114b`.
- Compact JSON SHA-256: `323e0b34d07a75ed366b2f2111d175e328c153596b6c76c48459f151de148999`.

[The compact result](commission_operating_ccc8690.json) retains the attempt,
source digest, solver versions, every fallback stage record, all daily inventories,
physical metrics and final EUR accounting. It can be reconstructed without any
environment reset or solver call:

```sh
.venv/bin/python docs/research/tools/commission_operating_result.py \
  runs/commission-demo/MPC.attempt.json \
  runs/commission-demo/5a25ba80-6227-4c07-8790-adc38f02a3a1.json \
  /private/tmp/commission-operating-result.json
```

The research E-beta implementation, simulator, frozen three-family/five-seed study
and admission requirements remain unchanged. This closes the scoped product
rehearsal, not research admission or commissioning readiness.

# E-beta uncertainty and delivery repair result

Date: **2026-09-11**. Implementation: **`a82f11c44e36`**.

The authorized repair is implemented, but the revised comparator has **not passed
the full-week fixture gate**. All three fixed fixtures stopped at their opening
decision with primary-model infeasibility. The current product scenario also
stops explicitly on its opening E-beta decision. No physical step, automatic
fallback, protected reset, PPO training or policy selection follows these stops.

## Implemented corrections

- The saved sunset interval had more available PV than forecast. Its fixed
  charging ceiling curtailed early surplus and could not capture enough later
  power. The revised model constrains held targets by the minimum sampled public
  daylight envelope, separately from forecast interval energy.
- Public-band p10 solar and fixed 1% resource buffers, restored two hours before
  sunset, are declared and reconstructed. Buffers are spendable after that
  deadline; original daily return requirements and tolerances remain unchanged.
- Methane shipments now match physical settlement: `min(available, requested)`.
  The controller cannot withhold shipment to protect a predicted tank reserve.
  Exact full/partial settlement is enforced independently of objective weights.

The policy identity is `public-p10-early-reserves-v2`. Explicit historical mean
construction, M24, V1, the simulator/guard and the frozen PPO study design remain
unchanged. Prior policy revisions and all failed evidence remain in Git history
and retained artifacts. The frontend and visual identity were not modified.

## Fixed fixture outcome

| Seed | Completed decisions | Physical minutes | Child wall time | Result |
| --- | ---: | ---: | ---: | --- |
| 9,100,001 | 0 | 0 | 4.90 s | Primary MILP infeasible |
| 9,100,002 | 0 | 0 | 4.03 s | Primary MILP infeasible |
| 9,100,003 | 0 | 0 | 4.03 s | Primary MILP infeasible |

All three child processes exited zero: this means the runner successfully
retained their controller failures, not that control succeeded. Every archive
passed detached physical and timing reconstruction in the parent. No process,
log or memory cap was reached. Zero physical steps provide no operating safety
or full-week capacity evidence. The earlier revision's 72/144/25-decision failures
remain in [the first result](v2_beta_uncertainty_c5b049b.json).

Raw evidence is outside Git at
`/private/tmp/plantanagent-beta-uncertainty-a82f11c44e36/`.
The report-domain digest is
`ae2c31f2044f41bfcfb7d0aeb54594cbc385dd9865c9bdc50a35178827748814`.
The [compact result](v2_beta_uncertainty_a82f11c.json) includes source, raw/archive
digests, classifications and the public inputs for the following calculation.

## Independent necessary energy bound

This calculation reads saved public observations only. It invokes no environment
reset or solver and makes no assumption about unseen weather truth.

The plant starts at 06:00, sunsets at 19:30 and owes 240 kg at 12:00, 18:00,
00:00 and 06:00. The model permits no nighttime methane production. A positive
final methane return requires both nighttime shipments to be full. A partial
18:00 shipment would empty the tank; the remaining 1.5 hours can produce at most
225 kg, less than the 480 kg nighttime requests plus the required final stock.
Thus 18:00 must also ship fully. The first shipment is at least
`min(initial methane, 240 kg)` because no earlier shipment or methane-consuming
process can remove that stock.

Consequently daily methane production must be at least
`720 + min(initial methane, 240) + required final methane − initial methane`.
H2 and CO2 balances and their original return minima then bound the required
electrolysis and capture. Multiplying by the model's specific energies gives:

| Seed | Minimum CH4 production | Minimum process electricity | Issued mean PV | Issued p10 PV |
| --- | ---: | ---: | ---: | ---: |
| 9,100,001 | 916.17 kg | 32.82 MWh | 28.40 MWh | 23.77 MWh |
| 9,100,002 | 960.00 kg | 34.39 MWh | 27.83 MWh | 20.51 MWh |
| 9,100,003 | 960.00 kg | 34.39 MWh | 29.43 MWh | 25.29 MWh |

Battery return allows only a sub-micro-kWh endowment contribution. These bounds
already exceed **mean** forecast energy, even ignoring added buffers, the held
power restriction, startup energy and storage losses. Thus this opening failure
is not a solver timeout or merely excessive p10 reserve conservatism. The former
withholding relaxation hid a forecast-scenario feasibility deficit.

This is a proof about the declared planning model and issued scenarios. It is
not proof that the physical plant is infeasible under realized weather, nor a
test of unopened development/held-out episodes. A p10 band is not a joint weather
guarantee. The reproducible calculation is
[`beta_uncertainty_result.py`](../tools/beta_uncertainty_result.py).

## Remaining decision

E-beta is not ready to freeze as the reliable primary comparator. A subsequent
scope must address how supervisory control acts when public forecasts cannot
satisfy the daily return obligations, and separate model conservatism from task
and plant feasibility. Any recovery mode must explicitly record predicted
shortfalls while preserving safety and operator authority; it cannot silently
pretend the existing hard-return MILP has a feasible plan. Changing the task's
obligations, limits or study eligibility is a separate protocol decision.

No such recovery policy, parameter search or task relaxation is introduced by
this repair. The training and held-out boundaries remain closed.

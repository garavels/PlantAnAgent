# Whole-plant battery access and downstream readiness: prospective experiment

Authorized 17 September 2026; local CPU only, seven-day episodes. Campaign starts 19:09:38 UTC and ends by 23:09:38 UTC. Training stops by 22:09:38 UTC, reserving the final hour for held-out evaluation and artifacts. Previous experiments, models, source captures and outcomes remain unchanged.

## Question and four physical arms

Does extending battery access to all processes and adding finite downstream readiness change operation, forecast value or the performance of learned control? All arms keep 30 kg/hour methane demand (180 kg every six hours), 28 deliveries, 15-minute held controls, one-minute physics, original ratings, gas capacities, sampled 25–65% opening battery and original gas distributions. Weather/fault tapes and opening inventories match within scenario seed.

| Arm | Battery access | Downstream behavior |
|---|---|---|
| restricted-instant | Electrolysis only | Original instantaneous DAC and methanation |
| shared-instant | All three processes | Original instantaneous DAC and methanation |
| restricted-dynamic | Electrolysis only | Paid readiness, standby and cooldown |
| shared-dynamic, primary | All three processes | Paid readiness, standby and cooldown |

Original electrolyser startup, minimum load, ramping, standby and shutdown dynamics remain in every arm. No compulsory nighttime operation, extra operator reserve, daily stock-restoration objective, imitation controller or discretionary PPO action shield is introduced. Restricted routes are explicit experimental architectural interventions.

## Declared dynamics and energy boundaries

These are assumed aggregate readiness dynamics, not a calibrated thermochemical model. The bounded primary-source review is retained under results/wholeplant_ppo_20260917/assumptions/thermal-readiness-review.md. It supports finite preparation, retained readiness and fast hot restart; exact coefficients below are design assumptions.

| Unit | Paid readiness capacity | Maximum preparation power | Ready standby draw | Cold preparation at maximum | Unpowered linear cooldown |
|---|---:|---:|---:|---:|---:|
| DAC | 120 kWh | 300 kW | 60 kW | 30 minutes | 2 hours |
| Methanator | 72 kWh | 90 kW | 18 kW | 60 minutes | 4 hours |

Both dynamic units start cold. Accepted preparation power fills the readiness reservoir after its modeled loss. No product is made while preparing; elapsed time alone cannot create readiness. Readiness can be retained with paid standby, or lost on insufficient power. Outages admit no electricity or production and readiness cools. No reaction heat is recovered. A hot unit can resume production without an additional arbitrary delay.

When ready, accepted power first maintains readiness. Remaining power is productive. To avoid adding maintenance twice to the original nominal full-load demand, productive specific energy is original specific energy multiplied by (1 − standby power/rated power). Consequently, full-load production matches the original nominal coefficient; lower loads bear the declared fixed maintenance overhead. Feedstock or output-capacity limitations can leave a unit in paid standby. No separate downstream minimum up/down time or cycle-wear penalty is imposed. Record preparation, standby, productive power, readiness energy and losses separately. These reservoirs are control abstractions, especially for aggregated cyclic DAC modules; they are not measured bed temperatures.

## Equal control and information

Every arm has seven bounded direct controls: four solar shares, followed by battery power requests to electrolysis, DAC and methanation. Solar shares are jointly normalized. The three battery requests share one 2.5 MW inverter and available battery energy, using proportional reservation rather than a hidden process priority. Simultaneous explicit charging/support requests are netted transparently on the electrical bus. Refused process solar is curtailed; it is not automatically allocated to another process or stored. Battery energy is withdrawn once for total accepted discharge; no simultaneous actual charging/discharging.

Public observation is the original 135 causal values plus 14 measurements: for each downstream unit, readiness fraction, off/starting/standby/producing indicators and accepted-power fraction; then separate DAC and methanator battery-power fractions. Static arm/configuration constants are public. Both baselines and PPO hold the same actions for 15 minutes, including through equipment transitions. They cannot read future realized weather/faults or private simulator tapes.

Inherited dispatch convention: available battery support is capped before charge/support netting. At the battery floor, an allocation to charging cannot be rerouted through a simultaneous support request; processes can still receive direct solar allocations. This is an explicit action-mapping limitation, not an energy source. Upstream gas production remains available to downstream processing within the same one-minute physical step.

## Controllers and training budget

Two fixed, inspectable baselines run in all four arms: daylight production coordinating inventories and readiness without battery use, and storage-assisted production that can allocate battery power to startup, production and useful standby. Their rules adapt to the declared arm's physical availability, and are frozen before development outcomes. Engineering fixtures use training seeds only. No baseline tuning uses development or held-out results.

Three fresh PPO policies, optimizer seeds 101/211/307, train only on shared-dynamic. No existing policy weights, normalizers, demonstrations or heuristic targets are imported. Retain the previous PPO settings: two 64-unit layers, four environments, 2,048 steps per environment/rollout, learning rate 3e-4, batch 256, ten epochs, gamma 1, GAE 0.95, clip 0.2, entropy coefficient 0.01, value coefficient 0.5, gradient norm 0.5, CPU and one Torch thread per learner. Scale the scalar economic euro reward by 0.001; normalize observations during training and freeze normalization in evaluation.

Proposed fixed budget is 786,432 transitions per seed. Before production training, a discarded concurrent pilot estimates throughput. Select the largest common whole-rollout budget among 786,432, 655,360, 524,288, 393,216 and 262,144 that fits the remaining training window with a 25% time margin. Record the chosen common budget and pilot calculation before training; no score-based extension or best-checkpoint selection. If none fits, report the engineering and baseline experiment without claiming completed PPO training. All final seeds and incomplete attempts are retained. Intermediate checkpoints are diagnostics only.

## Fixed cases and forecast intervention

New scenario domains are training 30,000,000–30,099,999; development 30,100,000–30,100,007; held-out 30,200,000–30,200,031. They are disjoint from previous studies. Primary development: two baselines × four arms × eight scenarios = 64 cases. Learning check: each of three initial/final PPO pairs × eight development scenarios = 48 cases.

Held-out access follows freezing scientific source and every final endpoint. Main held-out comparison: two baselines × four arms × 32 scenarios = 256 cases, plus three PPO policies × shared-dynamic × 32 = 96 cases. One supplementary 32-case comparator replaces all forecast inputs consumed by the storage baseline with current-sunlight persistence on shared-dynamic, preserving its other rules. Register exact consumed coordinates in the frozen baseline metadata. Total prescribed evaluation cases: 496, all retained. These reuse only 32 independent held-out environmental scenarios.

Practical forecast evidence criterion, fixed before development: either at least four forecast-rescued all-delivery weeks with no regressions, or no lower all-delivery count plus at least EUR 250/week paired contribution gain with a positive lower 95% bootstrap bound. Report both delivery and contribution results even if the criterion is not met. This is incremental information value within one fixed heuristic, not proof of universal planning or RL necessity.

## Outcomes, checks and inference

Keep the original economic objective and coefficients. Total battery throughput includes all three destinations. Warmup/standby incur real energy/time consequences; no invented downstream cycle penalty or reward for staying warm is added. Report full-week and all-delivery counts, every economic component, individual opening/ending stocks, per-process nighttime production, battery energy by destination, preparation/standby energy, starts and ending readiness.

Ending readiness receives no terminal salvage value. Cold initial readiness and zero terminal readiness credit define a finite-week startup/operation comparison, not a steady-state thermal-efficiency estimate.

Independent minute checks must reconcile solar allocation, each process's solar-plus-battery input, shared inverter power, battery energy/efficiency/floor, H2/CO2/CH4 conservation, delivery settlement, productive-energy-to-mass conversion, readiness input/loss/storage balance and zero product during preparation/standby. Audit actual accepted energy, not commands. Independently reconcile the economic ledger with summed scalar rewards. The prior electrolyser-only physical auditor is not sufficient.

Retain every case, including execution failures. Keep complete minute/action/input traces for the first prescribed scenario in every panel; retain compact audits and inference witnesses for all cases. Publish actual random-initialization versus final learning checks, source/model/normalizer hashes and a portable package. No held-out tuning, replacement, demand increase or architecture revision based on who wins.

Paired baseline contrasts across arms estimate routing, dynamics and interaction effects under those fixed algorithms, not the value after independently optimizing each architecture. Average three fixed PPO outcomes within each scenario before paired bootstrap versus baselines. Intervals resample environmental scenarios; they do not quantify retraining uncertainty. Seven-day success does not prove indefinite operation. Startup parameters are assumed: a negative result for battery or nighttime operation remains scientifically valid.

# When does learned control improve on daylight production?

Authorized by the user on 17 September 2026 as the comparison-first follow-up to the completed direct-control PPO study. The user keeps the seven-day episode horizon and a maximum four-hour local-compute campaign. Start: **2026-09-17 14:42:48 UTC**. Finish by **18:42:48 UTC**. Any fresh training must finish by **17:27:48 UTC**, preserving at least 75 minutes for evaluation and reporting. No remote compute or external submission is authorized by this protocol.

## Scientific question and limits

Does a learned direct-control policy improve delivery and modeled contribution over competent, transparent daylight-production and storage-assisted controllers? Is low battery use a consequence of opening energy, moderate demand, or a useful strategy under the implemented plant assumptions?

The prior experiment establishes genuine PPO learning in the simulator. It does not establish superiority to a strong matched heuristic, optimality, realistic downstream shutdown behavior, or indefinite operation. This follow-up keeps seven-day episodes and does not claim indefinite sustainability. It permits findings that simple control is sufficient, that the current PPO is already adequate, or that the selected challenge remains unresolved.

Preserve the original frozen physical simulator, its five direct controls, 15-minute decisions, minute physics, 135 public observations, efficiencies, storage capacities, reward and electrolyser dynamics. Battery power continues to supply only electrolysis. Carbon capture and methanation remain memoryless conversions without thermal or startup states; their limitations must be prominent in the report. No unsupported thermal parameters, compulsory battery cycling, daily stock-return constraint, controller prior or discretionary action shield will be added to PPO.

## Four paired operating conditions

| Label | Delivery rate | Opening battery |
|---|---:|---|
| d25-sampled | 25 kg/hour; 150 kg every six hours | Original sampled distribution, 25–65% |
| d25-floor | 25 kg/hour; 150 kg every six hours | Modeled 5% floor |
| d30-sampled | 30 kg/hour; 180 kg every six hours | Original sampled distribution, 25–65% |
| d30-floor | 30 kg/hour; 180 kg every six hours | Modeled 5% floor |

All last 168 hours, with 28 scheduled deliveries and 672 control decisions. Within a scenario seed, preserve the realized sunlight, faults and opening gas quantities across conditions. Changing demand must not change opening gas stocks, weather draws or fault draws. The opening battery intervention occurs before economic references and the first observation are initialized. It changes an explicitly disclosed initial condition, not the battery physics.

The 30 kg/hour challenge is fixed before seeing its development outcomes. Nominal process energy for 5,040 kg/week is 164.808 MWh before startup, part-load and other losses. This is an optimistic lower bound, not proof that a feasible schedule exists. The rate will not be increased or replaced because of which controller wins. Opening at the floor tests dependence on initial electrical energy; it does not establish repeatability across weeks.

## Controllers and fair information

Retain every final policy from the previous study: optimizer seeds 101, 211 and 307, trained at 25 kg/hour for 786,432 transitions. Labels are `ppo25-101`, `ppo25-211` and `ppo25-307`. Use each saved observation normalizer unchanged. Their performance at 30 kg/hour or a floor opening is a transfer test, not evidence of training under those conditions.

Implement two transparent, state-aware comparators:

- `daylight`: coordinates gas inventories and upcoming deliveries using direct solar allocations and no battery charging or support.
- `storage`: the same basic production logic with explicit battery charging and support for useful electrolyser startup, operation and shortfalls.

Both receive only the current 135 public observation values and declared static plant constants, output the same five bounded actions and hold them for the same 15 minutes. They may use only the causal forecast information available to PPO. No realized future tape, privileged physical truth or minute-by-minute target allocator may inform a baseline action. A baseline may follow its own startup rules; these do not constrain PPO. Document every rule and constant. Engineering fixtures use training seeds and retain their settings and outcomes; no baseline tuning uses held-out outcomes. Freeze baseline source before canonical development comparison.

The former translated prior is not the principal comparator because its target-to-share interface impairs startup. The new baseline must be checked for valid causal inputs, correct action mapping and coherent startup/inventory behavior, even if its economic performance is poor. No comparator may be silently dropped.

## Prospective decision on additional training

Run all five controllers on all four conditions and all eight development scenarios: 160 primary development cases. After that panel is complete, apply the following declared decision rule to `d30-sampled`:

1. At least one of the five controllers must meet every delivery in at least six of eight scenarios. This is empirical evidence that the challenge is often attainable; it is not a feasibility proof for all cases.
2. Additionally, either at least one original PPO policy misses a delivery on that panel, or the better of the two baseline mean contributions exceeds the mean contribution of the three PPO policies by at least EUR 250/week.

If both conditions hold and the runtime pilot leaves the training and reporting reserve intact, train three new policies from random weights at 30 kg/hour with the sampled opening-battery distribution. This threshold determines whether to spend the next training budget; it does not select a final policy, change the main evaluation cases, or select a winning training seed.

Otherwise, retain the completed comparison as the result and explain why extra training was not undertaken. If all approaches fail frequently, do not call the task impossible: its feasibility remains unresolved. If simple control matches already-successful PPO, report that result rather than manufacturing a harder problem until PPO wins.

Fresh training, if triggered, uses optimizer seeds 101/211/307, the previous architecture and PPO settings, four synchronous environments and one Torch CPU thread per learner. Proposed fixed budget: 786,432 transitions each, subject to a discarded throughput pilot before the plan is frozen. The training budget must be fixed for all three runs before production begins; it may be reduced prospectively if the remaining wall clock cannot accommodate it. No continuation of old weights, checkpoint selection by score, or outcome-dependent extension. Retain initial, intermediate and every final policy with its normalizer and actual optimizer-update evidence. Incomplete attempts remain labeled incomplete. Never invent a full-budget endpoint from a partial checkpoint.

## Scenario separation and endpoint freeze

Use new namespaces: training 20,000,000 through 20,099,999; development 20,100,000 through 20,100,007; held-out 20,200,000 through 20,200,031. These are disjoint from the completed study. The same realized scenario seed is shared across controllers and operating conditions.

Freeze source, comparison specification and checkpoint identities before held-out access. Evaluate all five original/comparator controllers in all four conditions on the 32 fresh shared scenarios (640 controller-condition-scenario outcomes). If new PPO30 policies finish, also evaluate all three in `d30-sampled` and `d30-floor` (192 additional outcomes). These are 32 shared environmental scenarios, not hundreds of independent weather samples. All attempted cases and failures remain in the report.

New PPO30 initial and final checkpoints are evaluated on development scenarios to verify learning; intermediate checkpoint scores cannot substitute for final endpoints. Held-out outcomes cannot trigger tuning, retraining, scenario replacement or extension.

## Battery dependence diagnostic

On the same scenarios, evaluate each original PPO25 policy in a separately labeled `d25-sampled-no-battery` condition. Its charge and support requests are explicitly disabled after prediction, while requested and executed actions are recorded separately. The actual physics still conserves energy. This intervention measures the existing policy's dependence on battery access. It does not estimate the optimal batteryless controller or prove that a battery is unnecessary. The daylight baseline provides a separately designed batteryless comparator, still without an optimality claim.

The floor-opening and battery-disabled conditions answer different questions and must not be conflated. Battery-disabled results are secondary diagnostics. If the time reserve requires reducing diagnostic scope, retain and disclose the complete primary panel first; never retain only favorable cases.

## Outcomes, auditing and interpretation

Report every controller's complete-week count, all-deliveries count, delivered/missed kilograms, modeled contribution and all economic components. Also report produced methane, solar curtailment, actual battery charge/discharge throughput, electrolyser cold/hot/forced stops, and individual opening/ending battery, hydrogen, CO2 and methane inventories. Aggregated methane-equivalent inventory must not imply restoration of all feedstocks.

Primary controller contrasts are paired within condition and scenario. For an aggregate PPO result, average all three fixed training-seed outcomes within each environmental scenario before bootstrap resampling scenarios. Give per-seed results too. Confidence intervals condition on the trained policies and omit broader retraining uncertainty. Report condition interactions descriptively and identify exploratory comparisons; do not count repeated weather across conditions as independent evidence.

Audit every physical minute for energy, mass, bounds and actual process power, and independently reconcile economic rewards. Verify that policy inference receives the actual public observation and uses the matching frozen normalizer. Retain compact per-case ledgers and source/model hashes for all cases, with full minute/action/input traces for the prospectively selected first scenario per panel. Verify paired realized weather, fault and gas-stock identity, allowing only the declared intervention fields to differ.

Deliver an inspectable comparison, the numerical evidence, baseline source, any new trained policies, and an updated two-page LaTeX research note with prominent model limitations. Include a representative cloud/fault response where supported by the prescribed traces. Distinguish modeled contribution from calibrated profit and successful control from optimality. Preserve the original completed experiment and its archives unchanged.

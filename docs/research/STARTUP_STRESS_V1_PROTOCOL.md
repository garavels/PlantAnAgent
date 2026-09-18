# Six-hour loss of solar at startup: diagnostic protocol

Authorized 17 September 2026 after the completed whole-plant study. Local evaluation only; no training, checkpoint selection, controller changes or changes to completed study files. This is a constructed mechanism test, not an estimate of tail-event frequency or operational reliability.

## Fixed question

Can the frozen PPO policies recover a physically feasible first delivery when daylight-only production necessarily fails?

Start at 06:00 with 120 kg methane, 100 kg hydrogen, 600 kg CO2, battery SOC 25%, cold downstream readiness and healthy equipment. These opening quantities lie within the previous benchmark's marginal ranges; their simultaneous selection is a deliberate diagnostic intervention. Keep the shared battery, dynamic downstream model, all component ratings, economic coefficients, 15-minute actions, one-minute physics, seven-day episode, and 180 kg deliveries every six hours. Remove other faults to isolate the solar interruption.

Two paired deterministic cases use the same initial state and all other exogenous inputs: (1) solar unavailable for the first 360 physical intervals, then the ordinary clear-sky profile; (2) the ordinary clear-sky profile throughout. Solar resource factors are 0/1 respectively, with factor 1 after interval 359. Forecasts retain the original stochastic ensemble forecast service and its configuration, initialized causally from the current resource factor, with paired fixed forecast randomness. They do not read future realized weather or the scheduled repair time; forecasts may anticipate resource recovery without knowing when it occurs. No onset/repair countdown is added to observations. Astronomical sunrise and sunset are unchanged.

The diagnostic seed 42,000,000 is a fixed reproducibility identifier, not an independent random sample. No confidence interval or population reliability rate will be inferred from it. A separately identified environment records the interventions; this is not part of the original held-out panel.

## Frozen controllers and feasibility witness

Run daylight, storage, and the three final whole-plant PPO policies (optimizer seeds 101, 211, 307; 786,432 transitions each), using the same raw 149-coordinate observations and seven-action interface. PPO uses deterministic means and its frozen normalizer exactly as in the completed study. All policies and both conditions are retained, including failures. No best-seed selection or retraining follows a result within this diagnostic.

An additional feasibility reference uses battery support only for methanation: 90 kW for four 15-minute decisions, then 150 kW for two decisions, then all equipment off until the first delivery. From the next decision onward, it uses the frozen storage baseline. The fixed first-six-hour schedule is identical under both solar conditions. It is an engineered physical witness, not learned autonomy or a competitive autonomous controller.

The witness should consume 90 kWh during preparation plus 75 kWh for 30 minutes of production, producing approximately 61.111 kg methane. Its 165 kWh bus input debits about 171.875 kWh from the battery. There is approximately 1.111 kg of margin above the first delivery requirement. This schedule is fixed by the equations before policy outcomes; it must be verified by an actual rollout with minute-level audits.

## Measurements and execution

Primary outcome: first delivery at step 360, requested/delivered/unmet kg. Also record methane produced, actual battery discharge by process, readiness progression, economic components and battery SOC during the first six hours. In the zero-solar case daylight's 120 kg opening methane bounds its delivery to 120 kg; any larger result would invalidate the diagnostic.

Secondary outcomes: all 28 weekly delivery settlements, full-week unmet kg and modeled contribution, individual final stocks and physical/economic audits. Do not conflate first-delivery recovery with completing the week, or completion with meeting all deliveries.

Six controllers times two conditions give 12 full-week rollouts. Preserve full minute states and actual raw/normalized PPO inputs, raw means, requested actions and accounting witnesses. Use the unchanged independent whole-plant minute and economic auditors. Pin the protocol, diagnostic source, inherited source and final checkpoints before the first evaluated policy rollout. Retain execution failures rather than silently retrying cases; engineering test attempts are separately labeled. Limit the evaluation phase to 30 local minutes.

## Interpretation fixed before results

The expected daylight failure plus a successful physical witness establishes that the disruption is discriminating and recoverable. PPO success would establish recovery by that particular frozen policy in this constructed state. Storage-heuristic success would show that this capability is not exclusive to PPO. PPO failure despite witness success would identify a policy capability gap. None of those outcomes establishes rare-event frequency, broad tail reliability, a normal-operation insurance premium or outperformance on independently chosen full-week disruptions. Such claims require a separate stress family and confirmation panel.

# Batch 009 economic and reward-timing audit: prospective analysis plan

Written before new saved-trajectory value calculations on 2026-09-14. This is a descriptive analysis of closed batch 008, not policy selection, training, calibration, or a new experiment. The known batch 008 weekly outcomes and prior conversational component decomposition are not blinded.

## Fixed inputs and inclusion

Use the frozen `adaptive-rl-batch-008-analysis/report/analysis.json`, SHA-256 `9b4e86afe39148a9ad5f9877c8a2ddcdb5e135bbb104ba6043d51a76da56934c`, and only its digest-bound saved episode artifacts. Retain all seven actors and all eight seeds 9700300–9700307. Preserve stopped actors and failure reasons. Aggregate performance and paired differences require all eight complete episodes; stopped prefixes are never substituted for complete outcomes. No simulator reset, model inference, training, price mutation, or protected-seed access is permitted.

## Exact estimands

1. Reconcile every recorded economic component with undiscounted net operating value for every actor/seed. Report each complete actor's eight-case mean and each scalar/PI candidate-minus-OperatingMPC difference, including all component contributions and per-case values.
2. Rescore unchanged physical trajectories using one-at-a-time multipliers `[0, 0.5, 1, 1.5, 2]` on each recorded battery-throughput reserve, solar opportunity cost, electrolyser-life reserve and unmet-delivery penalty. Delivered-service value and all remaining components remain fixed. Also report the full 5-by-5 battery-wear × unmet-delivery-penalty grid. These are illustrative sensitivity multipliers, not credible calibrated ranges. Report mean, worst paired difference and positive-case count, with all paired values retained. Compute the analytic one-factor mean break-even multiplier where its slope is nonzero; explicitly flag negative factors and unchanged-slope cases. A zero/negative penalty is a mathematical sensitivity only, not a proposed service contract.
3. Report battery throughput and electrolyser nonproductive energy divided separately by produced CH4 and delivered CH4, both ratio of panel totals and the eight per-case ratios. Retain production, delivery and terminal inventory information. Call these throughput/intensity proxies, not thermodynamic efficiency or measured equipment life.
4. If the frozen saved episodes contain the full per-decision reward/component series, validate step count, sums, end-of-episode correspondence and digest identity, then compute start-of-episode discounted returns `sum(gamma**t * reward[t])`, with first reward at t=0, for gamma `[0.99, 0.995, 0.999, 1.0]` plus the exact recorded training gamma if different. Decompose the same discounted return by each economic component and by seven 24-hour blocks. Compare complete actor rankings and candidate-minus-OperatingMPC paired differences at each gamma. Report discount weights at 6/24/168 hours and the effective horizon in hours. These deterministic realized-trajectory returns diagnose objective sensitivity; they do not reconstruct PPO advantages, prove causal credit-assignment failure, or predict retraining benefit.

## Validation and limits

Small fixtures validate field extraction, negative component signs, reconciliation, ratios, analytic break-even arithmetic and discount indexing. Source data must be stable, nonsymlink, singly linked regular files with exact path/size/hash checks; decompression is bounded. All deviations and unavailable fields are recorded. No economic coefficients are changed in the environment. Existing safety, individual-delivery and daily-resource regressions remain binding, regardless of rescored mean value. Root owns any subsequent training protocol, admission, conclusions and commits.


## Recorded descriptive extension

The original committed plan named battery throughput and electrolyser nonproductive energy. During implementation, the analysis additionally normalized accepted solar and accepted process energy by produced/delivered methane to distinguish cycling from plant energy intensity. This extension was added after the original plan commit; its timing relative to the first inspection of those raw fields is not separately attested. It is explicitly additional descriptive analysis, not a preregistered selection criterion. The original sensitivity grid, gamma grid, actor/seed inclusion and no-selection rules are unchanged.

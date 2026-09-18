# Supplementary test: does forecast information help competent control?

Declared 17 September 2026 during the running comparison campaign, after its development results and before any of its fresh held-out outcomes. The user clarified that the standard is plausible assumptions and evidence of an interesting control problem, rather than commercial calibration of a Rivan plant. This supplementary test does not change the frozen comparison experiment, training, physical model or four-hour deadline of 18:42:48 UTC.

## Question and exact intervention

Does the available weather forecast improve a competent fixed controller relative to a current-solar persistence input?

The reference is the already-frozen `daylight` baseline. The sole additional comparator, `daylight-persistence`, runs the same baseline code, constants, inventory buffers, commitment logic, physical limits and five-action interface. It receives the actual 135-dimensional public observation. Within its own policy it replaces exactly the three forecast coordinates consumed by the baseline—0–1 hour solar mean, 0–1 hour solar p10, and 1–6 hour solar mean—with current observed `solar_available`, using the observation's actual normalization and units. It changes no other input. Original and transformed observations, the replacement indices and values, and requested/applied actions must be recorded distinctly.

This is a matched persistence-forecast comparator, not a completely memoryless or inventory-blind controller. It retains commitment state and stock targets. The intervention removes both forecast means and the p10 uncertainty estimate, so it cannot identify their individual contributions. It does not establish the optimal forecastless controller or compare all planning algorithms.

## Frozen panel and scope

Evaluate the new comparator only in `d30-sampled`: 30 kg/hour demand, the original sampled opening charge, a seven-day episode and 15-minute actions. Use all 32 already-prescribed fresh environmental seeds, 20,200,000 through 20,200,031. Pair each case with the unchanged daylight outcome from the main experiment, retaining every success and failure. Both controllers are batteryless, so duplicating the floor-opening condition is unnecessary for this question.

The original 160-case development comparison informed the choice to ask this supplementary question. The original daylight baseline and the additional persistence rule will not be tuned using any new development or held-out performance. Engineering fixtures use only the training namespace and retain their attempts. Freeze this protocol and all supplementary code before opening any fresh held-out result, and require the main campaign's endpoint gate before execution. Use a separate evaluator and source manifest; never modify or monkeypatch frozen scientific modules while presenting them as unchanged.

One attempt per case; retain failures and incomplete cases. Audit electricity, modeled gas inventories, economic accounting and actual policy inputs using the unchanged main-study audit definitions. Verify identical weather, faults, opening stocks and delivery schedule against each reference case. Preserve the first prescribed scenario's complete minute trajectory. Keep concurrency at three evaluation workers globally. Run only after the main campaign's evaluation work has closed; the main study and final report retain priority if time expires.

## Outcomes and interpretation

Report all-deliveries counts, delivered/missed mass, modeled contribution, cold/aborted/forced starts, curtailment and individual opening/ending inventories. Compute paired differences on the 32 environmental scenarios; report a descriptive 95% paired bootstrap interval using the main study's fixed bootstrap settings. Do not treat repeated policies or conditions as extra independent weather samples.

For a concrete, prospectively stated evidence bar, forecast information provides a practically useful contribution in this fixed controller if either:

1. it rescues at least four of the 32 complete-delivery weeks without causing any delivery-success regressions; or
2. it has no lower all-deliveries count and improves mean modeled contribution by at least EUR 250/week, with the lower bound of the descriptive paired 95% interval above zero.

These are reporting criteria, not tuning or policy-selection criteria. An improvement supports a narrow claim that anticipation helps this controller in the assumed environment. A null or negative result supports the adequacy of this simpler persistence approach on the tested distribution. Neither establishes mathematical optimality, universal planning value, or that RL is necessary. The existing PPO-versus-baseline comparison separately tests whether the learned policies capture useful control behavior.

The test is supplementary and declared after inspecting development results. The paper must not present it as part of the original comparison protocol or as an independently replicated weather sample. A negative outcome is retained without adding constraints, changing the demand, or extending training to manufacture a positive result.

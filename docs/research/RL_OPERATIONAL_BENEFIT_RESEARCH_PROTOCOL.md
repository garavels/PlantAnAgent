# Adaptive research toward operational RL benefit

Authorized 2026-09-12 by the user's explicit research-autonomy goal: diagnose the
learning failure and iterate experiments until verified operational benefit is
obtained. This is a separate adaptive development programme. The original
three-family/five-seed registered protocol, its protected seeds and negative
pilot remain intact. A positive result is a hypothesis, never an output that
may be manufactured by excluding failures or weakening comparisons afterward.

## Outcome required

The target is an inspectable, reproducible learned supervisory policy with a
material operational gain over strong public-information control under the same
simulator, actions, dispatcher, guard, weather/market information, delivery
obligations and economic accounting. All individual resource settlements remain
visible. The running product's operating MPC is a required economic incumbent;
H and M24 are additional baselines. E-beta's model infeasibility is retained and
cannot supply an artificial numeric win. No real-site, deployment or commission
readiness claim follows from synthetic scenarios alone.

The user's service-ceiling clarification is part of this scope: where incumbent
delivery is already high, the primary opportunity is better economics at preserved
service, with less wear or more reliable recovery as measured mechanisms. High
contracted production rates and high fulfillment percentages are distinct; retain
both the absolute obligation and the fulfilled fraction. Do not increase demand,
degrade forecasts or weaken a baseline simply to make PPO appear useful. The
current learning failure must first be repaired on the unchanged task. Any later
stress panel must be justified and frozen prospectively, with ordinary operation
still reported. Historical deterministic full-delivery cells do not establish
full service on the current stochastic seven-day task.

Before opening final confirmation data, fix the chosen policy/hybrid design,
training budgets and seeds, every comparator identity, scenario panel, practical
effect threshold and analysis. Require complete paired episodes, absolute
inherited hard-safety pass for the learned policy, service noninferiority and
individual resource-deficit noninferiority at every daily boundary. A scoped
operational gain may coexist with unresolved absolute daily restoration, but
must say so explicitly; it is not sustainable-feasibility certification.
Independent replay must reconstruct controller outcomes from saved actions and
minute physics. Replicate the selected method across at least three training
seeds before an algorithm-level claim. Subsequent confirmation uses a new,
sealed panel and cannot be recycled for development after a failure without
declaring it contaminated and freezing a new confirmation experiment.

The numeric materiality and confirmation panel are intentionally not selected
from confirmation outcomes: they will be frozen after engineering diagnosis and
before that panel is opened. This development phase itself cannot complete the
research goal.

## Current diagnosis

The first PI-101 pilot completed 1,048,576 transitions but reduced mean delivery
fill from 84.3% at initialization to 30.6%; H achieved 89.5%. All four scenarios
show the same ordering. None of these controllers passed all daily returns.
Saved weights/trajectories show the final reward critic's second tanh layer is
fully saturated and its prediction is effectively constant near -806 EUR, while
realized discounted returns average roughly -22,000 to -27,000 EUR. The actor
requests almost maximum electrolysis and much less methanation, fills hydrogen
storage and incurs much more startup/nonproductive energy. This establishes the
behavioral failure and poor critic fit, not a unique causal explanation.

Read-only source review found no demonstrated action-order, PPO-sign or pilot
wiring defect. Hypotheses to discriminate are numerical reward conditioning;
the four independently standardized sparse cost-advantage signals competing with
one economic signal; and very broad Gaussian exploration relative to deterministic
operation. GAE credit assignment may also matter, but changes to the frozen cost
trace require a separately named subsequent learner experiment.

Primary implementation references: [SB3 custom-environment guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html),
[PPO implementation details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/),
and [Stooke et al. on PID Lagrangian control](https://proceedings.mlr.press/v119/stooke20a.html).
These motivate experiments; they do not establish the cause in this plant.

## Development batch 1: fixed before its first reset

Five fresh seed-101 cells, each **131,072 transitions**, 16 rollouts of 8,192,
four synchronous environments, 5,120 minibatch updates. The full original
168-hour beta plant/task and PPO hyperparameters remain unchanged.

| Cell | Learner | Reward passed to optimizer | Initial Gaussian std |
| --- | --- | --- | ---: |
| A-pi-raw | Existing PI-Lagrangian | interval EUR | 1.0 |
| B-pi-scaled | Existing PI-Lagrangian | interval EUR / 1000 | 1.0 |
| C-scalar-raw | Existing scalar PPO | interval EUR | 1.0 |
| D-scalar-scaled | Existing scalar PPO | interval EUR / 1000 | 1.0 |
| E-pi-scaled-lowstd | Existing PI-Lagrangian | interval EUR / 1000 | 0.2 |

Scaling affects optimizer rewards only. Physical/economic info and all four
constraint signals remain byte-equivalent for identical actions. Monitor episode
returns remain raw EUR. No actor receives a different plant or an evaluation
exemption. Initial actor/reward weights are matched; E changes only `log_std`
after the common initialization, before any training transition. All saved
artifacts are labelled adaptive research, never registered or bounded fixtures.

Training uses the same allocator stream in every cell, seeds 9,600,000–9,600,195.
The old pilot's training and evaluation seeds are development-contaminated and
not reused here. The batch endpoint alone is used for comparison; no best-time
checkpoint is selected. Preserve initial weights, every completed rollout,
numeric diagnostics and the final checkpoint (including failure receipts).

Diagnostics record reward/cost advantage scales, reward/cost explained variance,
pre-clipping actor/reward/cost gradient norms, critic saturation, actor means/std
and clipping frequency. Diagnostics run on the just-collected training buffer
without changing parameters, gradients, RNG state or optimizer behavior.

After all five attempts close, evaluate all final policies, the common initial
policy, H, the product operating MPC, M24 and E-beta on **all eight development
seeds 9,700,000–9,700,007**. The dedicated evidence adapter supports all four
comparators; the ten-actor list is fixed in the launch plan. Report all failures and all
component ledgers. Learned-policy episodes and replay use deterministic actions.
No stochastic-policy operating claim or formal selection is made in this batch.
Only successfully closed training children supply a learned policy or the common
initial baseline. If cell A fails, the initial-policy comparison is unavailable;
no other cell replaces it. Initial and final weights are checked against their
recorded hashes before loading. Operator interruption stops the complete batch.

Per-cell training limit: 3 hours; per-evaluation child: 10 minutes; stdout/stderr
limit: 64 MiB each. Sequential children use one Torch intra-op/inter-op thread
on the existing Mac. This avoids tiny-network thread overhead; every arm uses
the same host settings. Opening free-space floor: 10 GiB on Lexar. Root:
`/Volumes/Lexar/PlantAnAgentResearch/adaptive-rl-batch-001`.
Expected training is approximately 2–3 hours from prior throughput, plus
comparison/replay time; this is a planning estimate, not a capacity guarantee.
No cloud resources, purchases or messages to others are authorized by this batch.

Freeze source and manifest before execution. Claim the root and each child
exclusively; no retries/resume/replacement within a batch. Recheck source/runtime
before and after children. Local failures remain data; source/integrity failures
stop the campaign. Any next batch requires a new prospectively committed design,
new root and journalled decision, covered by the user's research authorization.

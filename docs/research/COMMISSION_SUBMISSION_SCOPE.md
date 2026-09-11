# Commission submission scope

Updated 2026-09-11 following the user's requirement that PPO training be essential
to the submission. This supersedes the recommendation to defer PPO until after
submission. The commission, rubric, deadline and presentation duration remain
unspecified.

## Objective and required deliverables

Present one working virtual-commissioning and production-planning workspace for
a renewable methane plant, together with executed, reproducible PPO training and
evaluation. Demonstrate how the environment supports learning a four-target policy
and how that policy behaves through the same dispatcher, guard and physics as
the existing controllers.

1. Preserve the unified product, visual identity, causal forecasts, equipment
   dynamics, delivery/economic accounting, inventory outlook and explicit MPC
   fallback already implemented.
2. Execute the preserved PI-Lagrangian, integral-only Lagrangian and scalar PPO
   design with five optimizer seeds per family. Retain all 15 attempts and admit
   final checkpoints through production custody. A successful learned-controller
   demonstration requires a valid trained artifact; fixture smokes and untrained
   networks are not substitutes.
3. Report training curves, economic return, per-resource costs, multipliers and
   completion/failure evidence after custody permits opening it. Changing weights
   or training loss alone does not establish useful learning or generalization.
4. Evaluate through the frozen development procedure against H, M24 and research
   E-beta, retaining failed/ineligible arms. Open held-out only if the preregistered
   gates pass and that stage is authorized. A negative development result remains
   reportable and leaves held-out sealed. PPO superiority is an empirical result,
   not a promised deliverable or a prerequisite to demonstrating RL capability.
5. Make factual PPO results and a policy trajectory inspectable in the existing
   workspace. Prepare and fixture-test the inference/export/presentation interfaces
   before source freeze because checkpoint loading binds runtime identity. Use a
   prospectively specified demonstration policy/scenario or admitted selection;
   do not search for a pleasing trajectory or present an ineligible policy as a
   winner. Interactive availability needs declared acceptance checks; displaying
   an experimental result does not authorize automatic operational takeover.
6. Package a reproducible release, dependable walkthrough/backup recording,
   submission materials and an evidence appendix linking claims to results.

## Current position and work order

Scalar PPO, PI/I training, seed allocation, checkpoint custody, registered campaign
execution, development selection and detached verification are implemented.
No registered beta campaign has run. The remaining work is admission, execution,
evaluation and product integration, with defects repaired before source freeze.

The product's mean-forecast MPC measures inventory restoration as an operating
target. Research E-beta retains strict daily returns. Its recorded failures do
not prevent PPO gradient updates, but can prevent every policy from passing the
current development selection gate. Product continuity does not prove strict
research feasibility and product MPC cannot silently replace E-beta in that
comparison. A deficit in an issued planning scenario is not proof that the physical
plant task is impossible.

Complete the following in order:

1. Confirm submission timing and criteria, retaining trained PPO evidence as
   mandatory and superiority as conditional on results.
2. Close the bounded fixture-based task/comparator feasibility review and decide
   its contract before training. Resolve the limitations or explicitly retain
   their failure risk. An incompatible task/selection change needs a separately
   versioned protocol and seed plan, not edits under the frozen identities.
3. Prepare the PPO product adapter and evidence views using fixture artifacts;
   finalize evaluation and reporting interfaces before checkpoint source freeze.
4. Resolve host, capacity, storage and execution budgets. Complete the clean
   repository/static/frontend release gate and prepare the concrete source,
   artifact-root, host and budget authorization packet.
5. After authorization, execute the fixed 15-run campaign once and verify its
   artifacts. Keep outcomes sealed during execution and avoid commentary merely
   waiting for experiments. Preserve failed, interrupted and rejected attempts.
6. Open development under its separate authority, apply preregistered selection,
   and open held-out only if its exact gate passes. No bypass for presentation.
7. Populate the workspace with resulting evidence, rehearse and freeze the
   submitted application, recordings and claims.

The historical M4 Pro sequential training projection is 62.49 hours, with mechanical
1.25x/1.50x scenarios of 78.10/93.71 hours, plus evaluation and verification. These
are planning estimates. The training admission floor is 5 GiB. The earlier draft
evaluation storage scenario requires 46.80 GiB before additional training allocation;
that budget remains unfrozen. The user subsequently authorized `/Volumes/Lexar`,
where about 326.6 GiB is free and production custody I/O checks pass. The new
proposal keeps all archive caps and increases the filesystem margin to 4 GiB,
requiring 48.80 GiB for evaluation plus training storage. See
[PPO_LAUNCH_PLAN.md](PPO_LAUNCH_PLAN.md). The registered root and tracked execution
authorization remain absent.

This is authorization to prioritize PPO in the submission objective, not the
canonical execution authorization for a particular multi-day campaign. It opens
no protected seed and authorizes no reduced campaign, auxiliary evaluation or
change to the frozen three-family/five-seed design. HEPA/RUL, LLM takeover,
real-site calibration and hardware-in-the-loop deployment remain later milestones
unless the commission's requirements change.

## Governing evidence

- [Frozen learning protocol](CONSTRAINED_RL_BETA_PROTOCOL.md)
- [Registered training custody](CONSTRAINED_RL_REGISTERED_TRAINING_CUSTODY_PROTOCOL.md)
- [Training readiness review](TRAINING_READINESS_REVIEW.md)
- [Product operating scope](COMMISSION_OPERATING_SCOPE.md)
- [Completed product MPC rehearsal](results/commission_operating_ccc8690.md)

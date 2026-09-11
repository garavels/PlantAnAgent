# E-beta uncertainty handling

Status: **prospective implementation scope, authorized by the user on 2026-09-11**

Hypothesis: planning against public lower solar estimates and restoring resource
buffers before the last daylight window reduces loss of daily feasibility under
forecast and model error. This is a controller policy change, not a task repair.
The mean-based policy and both previous failed capacity attempts remain retained.

The new policy is identified as `public-p10-early-reserves-v1`. It uses only the
current public forecast's four band p10 values, reconstructed with the existing
clear-sky integration. A band quantile is not an interval quantile or a joint
probabilistic guarantee. All issue fields remain bound; market pricing remains
the existing known-current/issued-mean path.

Resource buffers are fixed prospectively at 1% of each resource's capacity,
limited by headroom above the original daily minimum. The planning restoration
deadline is two hours before the last daylight boundary preceding settlement.
Before that deadline, predicted inventories must retain those buffers from the
deadline through settlement. Once the deadline has passed, the additional buffer
is limited to the observed surplus at the current boundary. This makes buffers
spendable against realized error rather than requiring their impossible recovery
after sunset. Original four daily minima and all safety constraints remain hard;
no added variable relaxes them. Buffers never accumulate across daily cycles.

The policy replans from each current public boundary. Its model and descriptor
bind the uncertainty policy, forecast statistic, restoration boundary, requested
and retained resource buffers, and added constraints. Solver failure remains a
typed, retained failure with no silent retry, mean-policy fallback or tolerance
relaxation. This is a conservative heuristic inside MPC, not a proof of robust
feasibility for every possible weather realization.

Implementation includes public-record forecast/model discrepancy decomposition,
projection tests, early restoration and spendable-buffer tests, invalid settings
and evidence-tampering rejection, real-solver fixtures and detached reconstruction.
Existing mean-policy tests must retain their original numerical semantics.

After committing the implementation and focused checks, one bounded E-beta
fixture per seed 9,100,001 / 9,100,002 / 9,100,003 may run at that revision, in
that order, with the unchanged 168-hour task and 15-minute/minute cadence.
Seed 9,100,001 is the disclosed regression seed; the other two are implementation
fixtures, not held-out research evidence. Each child is capped at 600 seconds,
logs at 1 MiB per stream, and physical/timing archives at the existing capacity
fixture limits. All failures and attempts remain external and are summarized
with digests. There is no same-revision retry or outcome-based parameter search.
The maximum summed child budget is 30 minutes, excluding parent verification.

The new policy is connected to the product and study comparator with its complete
descriptor. Registered training, development and held-out evaluation remain
closed. Their 3-family × 5-seed design, protected seeds, costs, simulator, guard
and task tolerances are unchanged. Fixture success cannot establish PPO value or
commission readiness. Any later repair or capacity run requires a separately
recorded scope and retains these results.

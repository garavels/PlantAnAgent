# Commission operating scope

Authorized by the user on 2026-09-11, before implementation and rehearsal.

This document records the completed operating-controller increment and its
rehearsal authorization. The broader
[commission submission scope](COMMISSION_SUBMISSION_SCOPE.md) now also requires
PPO training, following the user's subsequent instruction. Registered execution
and evaluation gates remain separate from this product rehearsal.

The product receives a separate `operational-economic-mpc-v1` controller, shown
as Economic MPC (`MPC`). The offline E-beta implementation, 3-family × 5-seed
PPO design, registered task and protected seeds remain unchanged. Historical
E-beta recordings keep their original identity and failure outcomes.

The operating controller uses issued mean solar and public market forecasts over
a rolling 24 hours, clipped only at episode end. It retains exact mandatory
methane settlement and the sampled held-power constraint. The p10 policy and
extra early buffers are deferred. Equipment limits, battery safety reserve,
commitment dynamics, common dispatcher and independent guard remain hard.

Daily opening-reference inventory targets remain visible measurements; they do
not terminate product operation or constrain its optimizer. Existing terminal
battery and methane-equivalent depletion costs persist against the episode's
opening endowments, including across daily boundaries. The controller maximizes
forecast delivery service first, then minimizes the existing economic objective
including depletion and wear. It does not promise restoration when insufficient
energy is forecast. Individual resource levels/deficits remain visible alongside
aggregate depletion accounting, with no invented future recovery date.

A valid plan with forecast shortfalls or resource deficits is explicitly marked
Recovery. A typed solver/plan rejection uses the existing reactive heuristic for
that interval through the same execution path, explicitly marked Fallback with
its reason and no invented plan. The next interval tries the planner again.
Invalid/stale observations and unexpected software errors still stop the session.
Operator pause, step, reset and controller choice retain their existing authority.

The existing visual identity and panel structure are preserved. Planning exposes
mode, predicted delivery shortfalls and individual resource outlooks. Inventory
targets and daily records are labelled as operating references for new product
runs; old recordings retain strict daily-return labels. Mode transitions and
fallbacks appear in Events and public recordings. Evidence distinguishes product
MPC from the offline research baseline.

Validation covers real opening plans, resource/depletion accounting, immutable
reference levels, explicit fallback and return to planning, stale-input rejection,
daily continuation, API/recording integrity, frontend checks and browser review.
After a clean implementation commit, one product MPC rehearsal may run on fixture
9,000,034, with a 168-hour target, existing 20-minute wall cap and a new exclusive
MPC attempt marker. All outcomes and fallback counts are retained. No training,
development/held-out reset, parameter search or performance claim is authorized.

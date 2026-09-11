# E-beta capacity boundary diagnosis

Scope fixed before model execution: **one reconstruction of the retained public
boundary at decision 52**, from the failed seed 9,100,001 capacity fixture at
`17331fe5b89a8e30769cdd656a826683071dae29`.

The original capacity result and archives remain unchanged. This is a numerical
implementation diagnosis, not another capacity attempt or controller evaluation.
No environment, controller, checkpoint, optimizer, seed reset, trajectory replay
or action dispatch is permitted. Only the saved current public observation,
declared configuration/settings and initial economic endowments enter the model.
Private future weather, prices and fault schedules are not solver inputs.

Rebuild the unchanged beta model and execute its existing two solver stages once,
with the saved settings (two-second solver limit per stage). Retain both selected
vectors, solver records, input/model identities and four terminal return residuals
outside Git. Reuse that solve result for ordinary plan validation; do not solve
again to obtain a different candidate. Record failure as well as success. The
original failing candidate was not archived, so a new numerical result is a
reproduction attempt, not evidence of that original candidate's exact residual.

Hypothesis: hard return rows use the solver's residual allowance, while plan
validation applies the task's absolute 1e-12 resource epsilon. The economic
canonicalizer repairs small violations of explicit variable bounds only; the
beta terminal returns are currently singleton rows rather than variable bounds.
Inspect whether this distinction explains the rejected plan. Do not change a
task tolerance, solver setting, controller policy or physical constraint during
this diagnosis. Any repair requires a separately recorded implementation scope
and regression evidence before a new capacity protocol can be considered.

The three-family/five-seed study, protected seed domains and training authorization
remain unchanged and unopened.

## Retained result

At documentation-only revision `9f16e3f`, the unchanged numerical implementation
reproduced the rejection with both stages reporting optimal solutions. The
economic candidate's battery was `5013.628497150462` kWh against a required minimum
of `5013.628497150464`: a deficit of `1.8189894035458565e-12` kWh (two floating-point
ULPs). The explicit variable lower bound was only 2000 kWh. The other three
resource minima passed. Thus solver residual admission succeeded while strict
plan validation rejected the battery. This is a numerical representation defect;
it is not evidence that the plant can physically meet every daily return.

The single reconstruction's inputs, both vectors and stage records are retained
at `/private/tmp/plantanagent-beta-boundary-diagnosis-17331fe/`. Its `report.json`
raw SHA-256 is
`4af1b7e0dfd06af838b8aae39774295a2bf13bd09f3e477f748510296b1f8d2b`.
The model digest is
`409a3778b6751def0ffdb3d492d1cf02aefbf92f3b35a32121d791b5386e1481`.
No plant reset, controller call or action dispatch occurred. Process exit was zero.

## Separate implementation repair scope

Before any registered training or further capacity execution, represent each
existing terminal resource row also as the identical variable lower bound,
intersected with any stronger existing bound. Retain all four rows. These are
mathematically redundant bounds: neither feasible region nor objective changes.
The shared economic solver already projects small explicit-bound violations and
reaudits the coupled rows; reuse that behavior without changing its tolerances.
Version the beta model as `cyclic-return-economic-mpc-v2` so settings, model,
solution and controller evidence bind the changed numerical representation.

Regression scope: retain the saved public inputs and candidate in a compact test
fixture; exercise ordinary stage extraction with fixed solver responses and no
environment reset or new optimization. Verify the original two-ULP deficit is
corrected, feasible interior values are preserved, material infeasibility still
fails, all four bounds match their rows, and a stronger battery reserve cannot be
weakened. Existing strict plan validation (including its rejection of a 5e-8 kWh
deficit), physical balances, actual solver and detached evidence tests must pass.
No alpha/service-MPC behavior, plant dynamics, task tolerance, learning rule,
optimizer seed or budget is changed. A new capacity attempt is outside this
repair scope and must remain separately declared with the original failure linked.

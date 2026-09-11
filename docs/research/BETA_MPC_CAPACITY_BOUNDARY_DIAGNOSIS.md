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

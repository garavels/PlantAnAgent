# Setpoint-dispatch protocol clarification 001 — identity equivalence

Status: **frozen before the admission audit and before any setpoint-controller rollout**

Date: **2026-09-04**

## Trigger

Implementation gate 9 of
[`CAUSAL_SERVICE_MPC_SETPOINT_DISPATCH_PROTOCOL.md`](CAUSAL_SERVICE_MPC_SETPOINT_DISPATCH_PROTOCOL.md)
requires the first three frozen rate-35/seed-0 observation, model, and plan identities to
remain byte-identical before the old and new controller trajectories diverge. The governing
protocol has raw SHA-256
`21b12eaa245b21d20d9458b9e3d0b8d163c682ef85ad9d5579fde392647e8782`.

The requirement is internally inconsistent with two other frozen requirements. The new
controller must use a separately versioned settings/hash domain and the
feasibility-preserving solution-selection policy. `ServiceMPCModelV1` binds the settings
digest, while `MPCPlanV1` binds the controller, settings, model, and selected solver-vector
identities. Honest provenance therefore requires the model and plan digests to change even
when the mathematical model and extracted operating plan do not.

This contradiction was found before the implementation-admission audit and before any
setpoint-controller trajectory, 72-hour mechanism cell, performance metric, or gate outcome
was opened. It will not be resolved by erasing the new provenance or by editing the original
frozen document. This addendum narrows only the executable interpretation of gate 9.

## Unchanged scientific boundary

This clarification changes no hypothesis, simulator state or physics, episode, weather,
delivery demand, public information, controller cadence, MILP variable, bound, row,
objective, horizon, tolerance, solver option, setpoint, dispatcher rule, guard, outcome gate,
or retry policy. Gates 1--8 and 10--12, S1--S8, and the decision rule remain exactly as
frozen. It authorizes no full rollout.

## Frozen three-boundary counterfactual

The comparison path is the already frozen historical `service-mpc-issued-mean-v1` controller
through the unchanged `step_action(Action)` door at delivery rate 35 kg/h and seed 0. Its
first two actions generate three public decision boundaries. At each detached boundary, the
historical and new model are independently constructed from the exact same observation,
public forecast issue, plant configuration, constraints, commitment profile, horizon, and
issued-mean projection. The new setpoint command is constructed but never executed.

The immutable identities are:

| Decision | Public observation | Public forecast issue | Issued-mean projection | Historical model | Historical plan | New model | New plan | New setpoint |
|---:|---|---|---|---|---|---|---|---|
| 0 | `d1d2b3de1cf6314d5cfe857c3f49f7fe48744929853439fa77ebd07b0531ec1c` | `d95901492f5041dce9ff73d87944028b51639b2dddbd3a739c0cc888e55477ef` | `d0c1e03dc9199267cbdf32d516d3c9f4c2fe009081c35add7a59c47d0d02d917` | `a0e5b312ffa5ddd055d502ce5205c520dc642cd801e4286ed82924108c5d54d9` | `782220137621a4578abac42ebd36800b028d22f6aea25f86e2c8dbe93b6d08de` | `9fc997f369fc7d83d50b5ade834e4ef63bbb1f720a4aab47923a1d997821d120` | `a26f020ef6e55b4a225f796d940d7329f235db16731e07ce5310362d83b8dbcc` | `6066dc1d012b7466eb34eb32baba8b58cdd1b909f6743d5f753c7fd0d1531cdf` |
| 1 | `00219b326f2655b53fe515bd16cc7971f837950a65e07e3d2f0895883b088fac` | `daad4374d39eed1f88696c6bdb4d5866633cfb2bd4061cb4612ca10bce2ddf12` | `493cd521d74d770199559af107c00d52ba654ab5061a29b9f1a52349b850d86f` | `def36b41b5018d2a5cafaf2a42ef19873ba2a4f9c4d304260e14f1c6d9611356` | `ea115d2fbb37894c46684c88c8bc8a8dc3fa6a4ace2962fe4521610f8f5c15d4` | `6ee66de7f236c40849e43edda51691ffcda2cff149bd26ff3c64d4a61fc9e07c` | `5056690517984e5c09d182018e6a6b22f74fb2353f140fa7901448b99ef32ed2` | `9c3190dac0658d9976c0b1c5d3e52d185e6a2419286622be4bee77cd147ec8d2` |
| 2 | `3f9bbe6cc62ec3dfaaf486c20a818ee90bed95984ffe9e7bc9e4708f8dee5aac` | `9b83eb27b1025942e8d368388812d056990e987f7874904292ddf50c090e9c5e` | `34f11a731abf616da01bd09549973ae1409cca510eb9c6acf114bc6609ebdabf` | `25c5505091125fd36d5b8a79d5691e843cfcf12063a6280bc14392ef2dd56355` | `36af91aee783fe89979a92f4fb0e62b2f0dd31c04b09467d5c4403378cd2054f` | `6119c5ddb4269684a57308f2e95331520f9085830ef4ea1265aed758af3aabb8` | `7acc746a9f630f7964def4973225ef82d954a43df66cde9bef9a3bca55b20c60` | `de427a3d69913a8f28529474f8b8815156c89522815944d983f24f431ad29488` |

The historical and setpoint settings digests are respectively
`0e8896cfb9fe47dc3ac3b285fc729e029a80803e31eddf4a885795d2d04253f2`
and `e418dc37401482cf78036daefa61d80fa23fef0411a7215c3fb549a91085fb1b`.

## Replacement for implementation gate 9

Gate 9 passes only if one executable test proves all of the following at all three frozen
counterfactual boundaries:

1. The public observation, public forecast issue, issued-mean projection, historical model,
   historical plan, new model, new plan, and unexecuted new setpoint reproduce every digest
   in the table exactly.
2. The historical and new forecast projections are equal as complete typed values.
3. The historical and new `ServiceMPCModelV1` values become exactly equal after replacing
   only the new model's `settings_sha256` with the historical settings digest. This proves
   that no variable, bound, integrality flag, objective coefficient, sparse row, delivery,
   forecast value, initial condition, or metadata changed.
4. The historical and new plans have exactly equal horizon, deliveries, complete ordered
   `MPCPlanStepV1` tuples, primary delivered optimum, selected delivery, delivery-lock
   tolerance, minimum commitment fill, terminal methane-equivalent inventory, and secondary
   utility. No approximate or newly chosen tolerance is permitted for these frozen fixtures.
5. The settings, model, plan, and solver-stage solution identities remain honestly
   version-separated. They are not normalized in stored evidence and are not required to be
   byte-identical across controller versions.

The actual setpoint trajectory may diverge immediately after the first minute dispatcher is
applied. No post-dispatch boundary may be substituted into this historical counterfactual,
and no equality on future physical outcomes is implied. This replacement tests the intended
scientific invariant—unchanged forecast, optimization problem, and selected operating
quantities—while retaining truthful version and solution-selection provenance.

## Admission consequence

The clarification itself is not evidence that gate 9 passes. The admission audit must bind
the table and all five obligations to a focused regression test from a committed revision,
then satisfy every other frozen implementation gate from a committed clean worktree. Until
that audit is complete, the 72-hour setpoint mechanism run remains forbidden.

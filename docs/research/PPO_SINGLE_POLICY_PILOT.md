# Single-policy PPO pilot v1

Prospective specification, 2026-09-11. The user authorized: “Let's start by
training one policy on one seed and see what happens.” This authorizes this
separate exploratory run and its fixed descriptive checks. The registered
three-family/five-seed study and its protocols remain unchanged and unopened.

## Fixed experiment

- PI-Lagrangian PPO, optimizer seed 101, fresh initialization, CPU.
- 1,048,576 transitions, four synchronous environments, 128 complete rollouts,
  40,960 minibatch optimizer steps. Existing PPO and multiplier defaults.
- Unchanged cyclic-market beta task: 168 hours, seven daily return constraints,
  135 observations, four actions, 15-minute decisions, one-minute physics.
- Explicit training episode seeds 9,500,000 through 9,501,563, allocated by the
  existing four-environment allocator. Its `fixture` mode identifies the
  nonprotected seed namespace; this **long pilot is not a bounded fixture run**.
- Separate pilot manifest/checkpoint format, never a registered or fixture
  checkpoint. No changes to the production training-config limits, checkpoint
  admission, seed domains, task, reward, guard, or registered campaign runner.
- Root: `/Volumes/Lexar/PlantAnAgentResearch/pilot-pi-seed101-v1`.
- This Apple M4 Pro host; Torch 8 intra-op / 12 inter-op threads. Historical
  planning estimate 4–6 hours, hard training child limit 10 hours, log limit
  64 MiB per stream, opening free-space floor 5 GiB. These are caps/estimates,
  not a measured pilot duration. No cloud resources or spending.
- One attempt; exclusively claim the root before any reset. Preserve failure,
  logs and completed rollout records. No retry, resume, replacement seed,
  intermediate-policy selection or outcome-dependent parameter changes.

The pilot uses the production constrained learner, the production episode
allocator, and the actual beta environment. A separate runner composes these
public components directly. Its environment configuration is explicitly a task
template, not a declaration that this is generic scalar PPO. The pilot manifest
binds that template, the constrained learner settings, the full budget, exact
clean source/runtime and host. Final model bytes are reopened by the production
constrained loader and checked against the saved optimizer and allocator state.
Registered evidence exporters cannot admit this separate pilot format.

## Fixed descriptive check after training

Use all four nontraining scenario seeds 9,800,000–9,800,003, in that order, with
the final PPO policy, its saved untrained actor/reward policy, and H. Each runs
for the full 168-hour episode with deterministic PPO actions and the same
dispatcher, guard and simulator. Save each result, including exceptions and
stopped prefixes; never replace a failed case. Limit the evaluation child to
20 minutes and each stored trajectory to 64 MiB. Read no protected training,
development or held-out seed. These four exploratory scenarios are not the
registered held-out evaluation and do not select a checkpoint.

Retain decision observations/actions, public outcome info, exact episode
manifests, delivery/service metrics, economic ledgers, all seven resource-return
settlements, physical safety diagnostics, and complete rollout learning curves.
Report raw safety diagnostics without claiming the registered detached safety
gate passed. Changes from initialization and comparison with H are descriptive;
four scenarios and one training seed establish neither generalization nor
superiority to MPC. M24 and E-beta comparisons remain future work. An increase
in training objective alone is not evidence of useful plant control.

## Gate and custody

Before launch, run targeted real-learning/save/reload and pilot-boundary tests,
existing constrained-training/allocator/config/runtime regressions, strict typing,
formatting and CLI checks; retain failures and fixes. The complete 3,052-test
gate at `b5ea55e` is inherited evidence, not a complete gate at the pilot source.
Commit the isolated runner, tests and specification before execution. Capture
the clean source and exact check report in the attempt manifest. Recheck runtime
before/after each child; a mutation prevents admitted completion/evaluation.
Archive logs, models and results only outside Git. A compact result and journal
entry can be committed after the pilot closes.

The user's later one-policy request changes the immediate execution scope;
it does not authorize the canonical 15-cell campaign or its protected evaluation.
This specification is frozen before this pilot's first training reset.

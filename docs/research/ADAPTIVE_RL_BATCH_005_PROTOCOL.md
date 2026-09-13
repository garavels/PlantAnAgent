# Adaptive batch 005: repair replication of the matched residual screen

Prospective design, 2026-09-13, before any batch 005 reset. Existing local research
autonomy authorizes this experiment after the implementation and admission gates
below. This is a fresh execution following the [aborted batch 004](results/ADAPTIVE_RL_BATCH_004_RESULTS.md),
not a continuation of its partial learner state.

## Hypothesis and fixed design

Retain batch 004's hypothesis: whether existing PI-Lagrangian resource-cost
optimization improves the tradeoff relative to scalar PPO around an equal MPC
prior at an equal budget. The aborted attempt produced no optimizer round, final
checkpoint or development comparison. Repair its forecast and supervision
incompatibilities before testing that same hypothesis. Do not tune rewards,
constraints, optimizer settings, demand or control authority based on the failure.

All scientific and comparison provisions of the
[batch 004 protocol](ADAPTIVE_RL_BATCH_004_PROTOCOL.md) apply, with only the explicit
implementation, provenance and root changes below. In particular:

| Contract | Fixed value |
| --- | --- |
| Task | 168 hours, 28 deliveries, 6,720 kg methane, unchanged daily returns and economics |
| Timing | One-minute physics, 15-minute control decisions |
| Interface | 135 public inputs plus four causal cached MPC targets; four bounded residual actions |
| Arms | A-mpc-residual-scalar and B-mpc-residual-pi |
| Initial policy | Optimizer seed 101, exact zero action head, Gaussian std 0.2, matched actor/reward tensors |
| Budget per arm | 65,536 transitions, eight 8,192-transition rollouts, 2,560 minibatch updates |
| Environments | Four synchronous environments per arm |
| Reward scale | 0.001 for optimizer rewards only |
| Training seed range | 9,630,000–9,630,099, exactly 100 allocator resets per successful arm |
| Development seeds | 9,700,300–9,700,307 |
| Actors | H, OperatingMPC, M24, E-beta, initial MPC residual from successful A only, final A and final B |
| Evaluation/replay | All 56 slots each, sequential after training closes |
| Endpoint | Fixed final only; no selection, retry, resume or extension |

Preserve the existing PPO and PI parameters, cost advantages/critics/dual updates,
zero-command and stable-running-floor behavior, 25% operating-effort adjustment,
authority caps, one-minute dispatcher and independent guard. Scalar's economic
objective already includes methane-equivalent resource incentives; PI adds four
separate resource cost channels. No new reward or multiplier remedy is assumed.

The training range is deliberately **reused development material**, not fresh or
held-out evidence. Batch 004 consumed its first eight reset seeds; keeping the
same streams includes the failing forecast and prevents avoiding it by seed
replacement. Both arms start from independently reconstructed fresh weights and
empty optimizer/allocator state. No earlier checkpoint, partial state or diagnosis
is supplied to either learner. The development panel was planned for batch 004
but never opened; it remains an adaptive development panel, not confirmation.

## Exact repair boundary

Implement only the [frozen repair scope](ADAPTIVE_RL_BATCH_004_FAILURE_AND_REPAIR.md)
plus new run identity/provenance and matching detached analysis-tool joins:

- OperatingMPC uses the separately identified `issued-mean-physics-left-samples-v1`
  projection. It reconstructs issued mean band energy using the same public solar
  geometry and physics-start samples as the producer. Firm-power rows use that
  sampled envelope. Full public issue, timestep, geometry and trajectory are bound.
- The OperatingMPC comparator and both residual priors use this same repaired
  implementation. Their model, optimization stages, limits and explicit fallback
  otherwise stay fixed. Initial residual/native MPC parity must again be verified
  physically across all eight development scenarios.
- Old registered M24/E-beta projection semantics remain unchanged. All their slots
  are attempted; stopped runs remain unavailable for full-week comparisons. No
  forecast clipping, wider impossible-energy tolerance or concealed fallback is
  introduced. The weather producer and physical simulator remain unchanged.
- Parallel supervision schema v2 uses retained descriptors and bounded byte-prefix
  verification compatible with Lexar allocation, robust cancellation/completion
  publication, specific first-failure evidence and independently retained ordinary
  receipts after all groups close. Fatal/missing completion evidence never admits
  a policy or permits evaluation. Unknown/unclosed groups prohibit file inventory.
- The existing matched runner's default root/schema advance to batch 005 with an
  explicit repair-replication record. Its frozen batch 004 implementation remains
  at `4f908fc`; no old files or artifacts are modified. No generic runner refactor
  or duplicate learner implementation is part of this change.

New residual checkpoint dependency hashes bind the sampled projection and solar
implementation/constants. A new source/gate identity is required; historical
artifacts cannot satisfy this admission or be interpreted with changed semantics.
No causal comparison of the projection repair against old batches is proposed.

## Execution and evidence bounds

Exclusively claim `/Volumes/Lexar/PlantAnAgentResearch/adaptive-rl-batch-005`.
The default runner must reject any existing root. Preserve two isolated training
supervisors with one Torch intra-op/inter-op thread, four synchronous environments
per arm and sequential evaluation/replay afterward. No serial substitution or
cloud execution is allowed.

Per-arm cap remains 12 hours; each evaluation/replay remains capped at ten minutes.
Keep 64 MiB per log, 256 MiB compressed episode, 10 GiB free-space floor, 0.1-second
process polling, 30-second full integrity/startup bounds, five-second execution
acknowledgement and 15-second supervisor closure allowance. Prefix monitoring
uses 64 KiB blocks, at most 2 MiB read per poll, a soft 10 ms slice and a 60-second
verification deadline, 30-second scan interval, one-second metadata-settling bound
and 60-second final-verification bound. Neither backlog nor final verification
failure can be hidden as successful custody. The complete
executable settings and observed monitoring costs are retained in schema v2.

Keep all original initial tensor inventories, eight diagnostics/optimizer ledgers
and allocator prefixes per successful arm, source/host/gate/command identities,
ordinary receipts, actual observations, absolute commands, full trajectories,
detached physics replay and causal prior recomputation. Retain all failures and
unavailable slots. Receipt recovery is evidence retention, not success admission.

The detached auditor must be updated and tested on fabricated schema-v2 artifacts
before launch. It independently validates the complete supervisor/receipt graph,
prefix monitoring closure, fresh initial tensor construction, checkpoint/source
joins, actual allocator seeds and replay. The reporter retains both final arms,
all incumbent comparisons, PI-minus-scalar direction, every individual delivery
and daily resource vector, diagnostics, losses and unavailable comparisons.
Full-panel means require all eight complete replayed weeks; missing policies do
not receive substitute endpoints.

## Admission and interpretation

Before any production reset: review both repairs and runner joins; pass focused
fixtures including the exact failed public forecast, independent sampled issues,
legacy projection identities, residual/native MPC initialization and actual tiny
Lexar I/O; commit clean source; pass a revision-bound regression/static gate; and
freeze the detached tools with their check evidence. No previously claimed root
or old normal-closure auditor is reused. Retain unsuccessful checks honestly.

Once training starts, keep the design/source frozen and remain outcome-blind and
quiet. Process-only monitoring may inspect identities and closure existence. An
integrity failure aborts the campaign with its partial graph; ordinary failed arms
remain explicit under the fixed slot rules. No opportunistic intervention, reward
inspection or extension while a learner is running.

This is still a single-optimizer-seed development screen. The
[parent protocol](RL_OPERATIONAL_BENEFIT_RESEARCH_PROTOCOL.md) requires absolute
hard safety, paired individual service/resource noninferiority, prospectively
frozen materiality and sealed fresh confirmation. At least three optimizer seeds
precede an algorithm-level claim. Absolute resource restoration failures remain
visible even if paired noninferiority is achieved. High fulfillment does not justify
raising delivery requirements to create a gain: seek cost, wear or recovery value
at preserved service. No protected registered seed, live deployment, frontend
change, site-generalization claim or operational-benefit declaration is included.

# Adaptive batch 008: evaluate the preserved batch-007 endpoints

Prospective design, 2026-09-14, before implementation, checkpoint interpretation
or any batch-008 evaluation reset. The user explicitly authorized correcting the
power guard, resolving the audit issue and evaluating the saved checkpoints.
This is a new evaluation-only attempt. [Batch 007](results/ADAPTIVE_RL_BATCH_007_RESULTS.md)
remains aborted, with its original protocol, logs, receipts and tools unchanged.

## Fixed inputs and scientific question

Keep the matched scalar/PI question and every scientific setting of
[batch 007](ADAPTIVE_RL_BATCH_007_PROTOCOL.md). Reuse its fixed final endpoints,
without training, optimizer updates, checkpoint selection, budget extensions or
replacement policies. Both source arms completed 65,536 transitions, eight
rollouts and 2,560 minibatch updates, optimizer seed 101. Their training material
was development seeds 9,630,000–9,630,099.

Source root is
`/Users/filippogombac/Developer/PlantAnAgentResearch/adaptive-rl-batch-007`,
source revision `61b97851637a0f5a16e56c479e1a436cd30cf68d`, tree
`311f8d3b5d048643e4a38b638c73ef95363b4d27`.
These exact files under its `training` directory are the operating inputs:

| Input | Bytes | SHA-256 |
| --- | ---: | --- |
| A-mpc-residual-scalar/final-policy.zip | 972,651 | `4031802ee0d8b7042de4b5193ab82797479dfa800d6943f62b134aa0e415e524` |
| B-mpc-residual-pi/final-policy.zip | 1,639,652 | `881140cb3ddfa52d975017bb6cbbcbcecb8fbeb147a91cdd6a7db8430fd8e05d` |
| A-mpc-residual-scalar/initial-policy.pt | 111,356 | `ba9ab03b73f403c05664d03beb39909c0df53e8e459980768673c86819e7bf70` |

Other training records, including B's initial weights, remain required provenance
and matching evidence. Independently check their original result/receipt bindings.
The source configuration retains its original output-directory field when loading
checkpoints and constructing evaluation episodes; all new output writers use the
new root. Never rewrite source results or embedded checkpoint identities.

| Evaluation contract | Fixed value |
| --- | --- |
| Task | 168 hours, 28 deliveries, 6,720 kg methane; unchanged economics and daily returns |
| Physics/control | One minute / 15 minutes |
| Learned interface | 135 public inputs plus four causal MPC targets; four bounded residual actions |
| Controllers | H, OperatingMPC, M24, E-beta, A's initial residual, final A, final B |
| Development scenarios | All eight seeds 9,700,300–9,700,307 |
| Work | 56 fresh evaluations and 56 independent replays, sequential |
| Prediction | Existing deterministic policy inference and unchanged baseline implementations |
| Selection | None; exact fixed endpoints only |

The old attempt opened 26 evaluations and 26 replay attempts: all controllers
on seeds 9,700,300–9,700,302, then the four baselines and initial residual on
9,700,303. This is reused development material. None of those partial results
is substituted into the new comparison. No outcome was interpreted to choose
these endpoints or change the panel. The original registered three-family,
five-seed study and protected seeds remain unopened.

## Prospective verification of imported training evidence

Before evaluation, independently verify the preserved source inventory, source
gate/tool identities, original training process/monitor/completion graph, reset
allocation, endpoint budgets, model contracts, tensor identities and initial
matching. Reopen the exact saved checkpoints without learning or environment
resets. Record the resulting import admission separately from any normal batch-007
admission. A missing or inconsistent content/provenance check prevents launch.

The frozen batch-007 supervision verifier rejected A stdout's physical allocation:
589,824 recorded versus 581,632 observed bytes. Its logical size (580,389) and
SHA-256 match. When or why allocation changed is unknown. Preserve that failed
check explicitly; do not edit its record or relabel the old verifier as passing.

For this new post-closure import verification, file content identity consists of
canonical path, logical byte count and SHA-256. Retain existing nonsymlink,
single-link, descriptor/path identity and stable-read checks. Record original
and currently observed allocation separately; physical allocation is storage
metadata, not a content digest. This rule is fixed before opening model contents
and applies uniformly to post-closure custody records. It does not waive any
other discrepancy or change live allocation, log, storage or integrity budgets.
Test unchanged bytes with changed allocation, changed bytes/length, symlinks,
path races, missing completion and altered checkpoint/source bindings.

Bind the import packet and every consumed source artifact by digest into the
new gate and plan, and recheck the consumed checkpoint/result dependencies at
stage boundaries. Verify that inherited plant, controller, learner, forecast,
reward and replay code are unchanged from the training revision. Only explicit
power handling and new orchestration/verification code may differ. Imported
training provenance and new execution provenance remain separate.

## Bounded execution repair

Accept the exact additional macOS battery state `finishing charge` when the
source is AC, the battery is present and charge is at least 20%. Continue to
reject discharge, unknown/ambiguous states, insufficient reserve, missing battery,
command failure and timeout. Keep the two-second query limit and 8,192-byte
observation cap. Test the captured AC/99% incident plus source/reserve/presence
boundaries and continued rejection of unknown labels. Never alter actual power
settings, simulate host hibernation or broaden acceptance to arbitrary strings.

Use a dedicated evaluation-only CLI with no training command. Reuse unchanged
scientific evaluation and replay functions; do not temporarily monkeypatch the
old runner's global root or module identity. Keep exclusive paths, source/host
verification, clean-revision admission, bounded child cancellation and durable
receipts. Include the final failed child's disk receipt in forensic accounting.

Exclusive new root:
`/Users/filippogombac/Developer/PlantAnAgentResearch/adaptive-rl-batch-008`.
Use the same verified internal APFS volume and at least 10 GiB free space.
Keep ten minutes per evaluation/replay, 64 MiB per log, 256 MiB per compressed
episode, Torch 1/1 threads and `caffeinate -i -s`. Sample power before root claim
and at stage entry/exit; no continuous sampling inside evaluation is claimed.
Abort on integrity failure; no retry, partial-result reuse or resume within this
attempt. No training supervisor wave is created. Lexar remains an optional
verified archive destination after closure; retain internal originals.

## Admission and reporting

Commit this protocol before implementation. Review and commit source, pass an
exact clean-revision scoped regression/static gate, and freeze the detached import
verifier, auditor, reporter and launcher with prospective tests before execution.
Use fabricated checkpoint/custody graphs and a tiny actual internal-storage
process fixture for orchestration tests; no production episode before admission.
Retain failed checks and distinguish tested tool behavior from final constant pins.

During execution inspect only process identity and closure-marker existence.
After all writers close, audit the complete evaluation/replay graph before
interpreting performance. Preserve every failed controller, fallback, stopped
comparison, individual delivery/resource regression and paired economic result.
Check initial/native-MPC physical parity on all eight scenarios. An abort permits
forensic reporting only. No source edits occur while the campaign is active.

The [parent research protocol](RL_OPERATIONAL_BENEFIT_RESEARCH_PROTOCOL.md)
still governs hard safety, paired service/resource noninferiority, prospective
materiality, fresh sealed confirmation and multiple optimizer seeds. This is a
single-seed development comparison; no operational-benefit, registered selection,
commercial savings, site-generalization or live-deployment claim follows merely
from executing it.

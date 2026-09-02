# Research journal

## 2026-09-02 — Programme initiation

### Mandate

The user granted broad research autonomy to evolve the repository toward a product
that is technically competitive, commercially relevant, and demonstrably addresses a
real industrial problem. Work is isolated on `codex/environment-v2-research` and will
be organized as small, auditable commits.

### Starting evidence

A colleague has trained an RL policy against Environment V1 and reports that it
performs similarly to the shipped heuristic. The working hypothesis is that this is
environment saturation rather than evidence that the two policies are behaviorally
equivalent: the deterministic, mostly linear plant admits a broad plateau of adequate
allocations, and the heuristic already encodes the dominant process priorities.

This report has not yet been independently reproduced. The first research phase must
locate or request the checkpoint, its training configuration, learning curves, seeds,
and evaluation output before treating parity as a quantified result.

### Initial audit observations

- Environment V1 uses a deterministic clear-sky curve plus scheduled `solar_drop`
  disturbances. Once an episode is sampled, the trajectory is deterministic for a
  given action sequence.
- Default RL episodes last 48 simulated hours and draw from `clear_day` and
  `cloud_cover`, randomized starting inventories, and optionally one additional solar
  disturbance.
- Conversion is predominantly linear. The electrolyser has a minimum load, but there
  are no startup delays, ramp limits, minimum up/down times, delivery commitments,
  prices, or physical wear state.
- The observation has 30 values and includes the current solar value, clear-sky
  potential, one-step trend, storage, unit condition, controller constraints, and
  episode progress. It contains no exogenous forecast or demand commitment.
- The five-dimensional action allocates contemporaneous solar shares and battery
  support. It expresses dispatch, not equipment commitment or scheduling.
- The reward is methane production minus curtailment, battery drain, reserve breach,
  tank saturation, shutdown, safety correction, and cycling penalties. Its weights
  were scale-selected before any documented training evidence existed.
- The existing evaluation compares controllers on identical episode seeds, but it
  does not estimate an optimality gap, separate in-distribution validation from
  out-of-distribution stress, or report confidence intervals and tail risk.

### Main falsifiable hypothesis

`H1`: On Environment V1, the heuristic and trained RL policy are both close to a
perfect-information upper bound, and many materially different actions lie on a broad
performance plateau.

Evidence against `H1` would include a large gap between both controllers and an oracle,
or a large train/evaluation discrepancy for RL. In that case, reward design, training,
or observability should be investigated before increasing simulator complexity.

### Immediate research actions

1. Audit the current environment and training/evaluation contracts.
2. Locate the colleague's training artifacts and establish exact provenance.
3. Research current control, safe-RL, and renewable hydrogen/e-fuel benchmarks.
4. Define an Environment V1 ceiling study with heuristic, RL, simple-policy,
   model-based, and perfect-information comparisons.
5. Specify the smallest Environment V2 slice that adds genuine temporal commitment
   and uncertainty without prematurely building a high-fidelity chemical simulator.

### Tooling state

At the start of the programme, the repository had no local `.venv` or frontend
`node_modules`. Python source parsed successfully in the earlier audit, but the full
test and build suites could not yet be executed without installing locked dependencies.

### Reproducibility and dependency setup

The locked Python dependencies, including the optional `train` group, were installed
with:

```text
UV_CACHE_DIR=/tmp/plant-an-agent-uv-cache uv sync --group train --frozen
```

The first sandboxed attempt could not resolve `files.pythonhosted.org`; the identical
locked command was then authorized with network access and completed successfully. It
installed Stable-Baselines3 2.9.0, Torch 2.13.0, Gymnasium 1.3.0, and the workspace
packages into `.venv`. No dependency specification or lockfile changed.

Baseline verification:

```text
UV_CACHE_DIR=/tmp/plant-an-agent-uv-cache uv run pytest -q
321 passed, 1 warning in 16.03 s
```

The warning is an upstream Starlette/FastAPI TestClient deprecation concerning `httpx`;
it is not a failing behavior test. The branch therefore begins from a green Python
baseline.

### Artifact audit

Neither this repository nor the adjacent `latent-world-models` checkout contains the
reported trained PlantAnAgent policy. There is no PlantAnAgent `policy.zip`, checkpoint
metadata, training configuration, learning curve, or evaluation JSON available locally,
and no Git ref contains an otherwise-unmerged training result. The adjacent repository's
artifacts concern Atari and Doom experiments and contain no `PlantEnv` or plant-simulator
integration.

The minimum provenance required from the colleague is:

- the complete checkpoint directory (`policy.zip` and metadata);
- the exact source commit and dependency-lock digest;
- full training configuration and policy seeds;
- learning curves and training duration/hardware;
- per-seed paired heuristic/RL evaluation output;
- any uncommitted reward or environment modifications.

Until those artifacts arrive, this branch will create its own reproducible baseline and
will label the colleague's reported parity as unverified external evidence.

### Initial saturation evidence

A 100-seed, 48-hour exploratory heuristic audit exposed a structural ceiling: the CH4
tank has no offtake, starts at a randomized 0--20% fill, and can only fill. The heuristic
deliberately stops methanation near 97% fill. Over seeds 1--100, 73 runs ended at or above
96% fill; within those ceiling-limited runs, the correlation between starting CH4
inventory and reported CH4 production was approximately `-0.99994`.

This means the headline production metric is, for most nominal episodes, nearly the tank's
remaining headroom rather than a sensitive measure of dispatch quality. Mean heuristic
solar utilization was approximately 99.4%, no shutdown step occurred, and the positive
methane reward contribution was roughly two orders of magnitude larger than the combined
penalties. These observations strongly support `H1`, but they will be reproduced through
a committed diagnostic command before being treated as a branch result.

### Committed saturation diagnostic

`plant_backend.rl.evaluation.ceiling` now formalizes the preceding exploratory audit. It
can evaluate the heuristic alone or pair it with any repository-compatible checkpoint on
identical seeds, prints an explicit evidence scope, and writes every rollout plus
population distributions to JSON. `make rl-ceiling` is its maintained entry point.

The independently repeated baseline over seeds 0--99 produced:

```text
72/100 runs at or above 96% terminal CH4 fill
Pearson r(initial CH4, produced CH4) = -1.000 on those 72 runs
mean terminal fill                  = 94.7%
mean CH4 production                 = 2538.69 kg
mean solar utilization              = 99.4%
mean shutdown steps                 = 0
```

The exploratory 73/100 result used the shifted seed range 1--100. Both are retained in
the record. The conclusion is insensitive to that choice: roughly three quarters of
default episodes terminate against storage headroom, and their production is nearly a
deterministic function of randomized starting inventory.

## 2026-09-02 — Battery reserve consistency correction

### Finding

The V1 audit found a mismatch between the supervisory control envelope and the
physical battery. `Battery.discharge_headroom_kw` correctly converts stored energy
into bus-side power through discharge efficiency, but both the safety validator and
the heuristic divided usable stored energy by timestep duration without applying the
efficiency loss. Near the configured reserve, they could therefore request more
bus-side support than the battery could deliver without crossing that reserve.

The correct relationship is:

```text
deliverable bus power = usable stored kWh * discharge efficiency / timestep hours
```

### Change and verification

The validator and heuristic now use the same relationship as the physical component.
Two regression tests cover different failure modes:

- the validator clamps a deliberately excessive request to the loss-adjusted boundary
  and the projected ending SOC is exactly the declared reserve;
- the heuristic refuses a minimum-stack-load top-up for which the lossless estimate
  appears sufficient but the physically deliverable power is insufficient.

Focused verification before commit:

```text
backend/tests/test_control.py                         22 passed
simulator battery-focused tests                       5 passed
```

This fix preserves existing nominal behavior away from the boundary. The envelope
still operates on reported SOC and reported effective capacity by design; a positive
SOC sensor bias can consequently make it optimistic. Sensor uncertainty and robust
reserve estimation are separate V2 research questions and are not silently folded
into this correctness fix.

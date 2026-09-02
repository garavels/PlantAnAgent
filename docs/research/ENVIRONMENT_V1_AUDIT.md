# Environment V1 saturation audit

## Finding

Environment V1 is structurally unable to discriminate strongly between competent
dispatch policies over its default 48-hour episodes. The dominant cause is not merely
that the physics are coarse: most runs terminate against a randomized methane-storage
ceiling, so cumulative methane is largely determined before the controller acts.

This finding explains why a trained policy and the heuristic may report similar product
output. It does not establish that either policy is optimal, and the colleague's policy
is not available in the local repositories for an independent paired comparison.

## Reproducible diagnostic

The branch adds a read-only diagnostic that rolls the heuristic and an optional
checkpoint through independent `PlantEnv` instances on identical episode seeds:

```bash
make rl-ceiling
make rl-ceiling CHECKPOINT=runs/ppo/checkpoint JSON=/tmp/v1-ceiling.json
```

It records every run's true starting and terminal CH4 inventory, theoretical headroom,
production, reward, and existing operational metrics. It reports population
distributions, the fraction ending above a declared fill threshold, and the correlation
between starting inventory and production among those ceiling-limited episodes.

## Baseline result

Configuration:

```text
source commit before diagnostic: 8e81c39
controller: shipped heuristic
episode: default Environment V1, 48 hours
seeds: 0 through 99
plant and constraints: defaults
terminal ceiling criterion: CH4 fill >= 96%
```

Result:

```text
runs                                              100
ceiling-limited runs                               72 (72.0%)
correlation(initial CH4, produced CH4)             -1.000
mean initial CH4                                   303.58 kg
mean theoretical tank headroom                   2696.42 kg
mean CH4 produced                                 2538.69 kg
mean final CH4 fill                                  94.7%
mean solar utilization                               99.4%
mean shutdown steps                                  0
mean reward                                         987.13
```

The complete machine-readable result was generated at
`/tmp/plant-an-agent-v1-ceiling-100.json`. It is intentionally not committed because
generated experimental outputs belong outside source control; the command and seed range
above reproduce it.

An earlier exploratory run over seeds 1 through 100 found 73 ceiling-limited episodes,
a 94.8% mean final fill, and a `-0.99994` initial-inventory/production correlation. The
one-run difference is expected from the shifted seed range and is recorded to avoid
quietly selecting the stronger-looking result.

## Structural causes

### Methane storage is the terminal objective

- Episodes randomize initial CH4 independently between 0% and 20% of capacity.
- The CH4 tank has no delivery or offtake; methane can only accumulate.
- The heuristic stops methanation at 97% fill.
- The reward's saturation penalty begins at 98%, leaving a narrow rewarded stopping
  point just below the penalty.

For a run that reaches the ceiling:

```text
reported CH4 production approximately equals
0.97 * tank capacity - randomized initial CH4
```

The benchmark is therefore measuring remaining storage capacity more strongly than
dispatch quality.

### The dominant bottleneck never changes

At default ratings:

```text
electrolyser H2 supports about 115.4 kg CH4/h
DAC CO2 supports about 218.2 kg CH4/h
methanator power supports 150.0 kg CH4/h
```

Hydrogen is the fixed bottleneck. A policy that feeds methanation and sends most surplus
power to electrolysis is close to the obvious nominal strategy, which the heuristic
already encodes.

### Equipment has almost no temporal commitment

- Solar repeats a deterministic clear-sky profile.
- Battery efficiency and power limits are static.
- Electrolyser and methanator start and stop instantly.
- DAC modulates freely.
- There are no ramp limits, minimum up/down times, startup energy, thermal dynamics,
  physical cycling wear, delivery deadlines, or maintenance decisions.
- H2 and CO2 produced earlier in a timestep are available to methanation in that same
  timestep.

The result is primarily a sequence of memoryless allocation decisions rather than a
scheduling problem.

### Weather variation is narrow

The base solar curve is exactly repeated. Default episodes choose `clear_day` or one
predefined rectangular cloud-cover scenario and add at most one extra rectangular solar
drop with 30% probability. There is no controller-visible forecast, autocorrelated
weather regime, or uncertainty that grows with forecast horizon.

### The router rescues poor commands

Policy outputs are normalized into shares, excess component requests spill into the
battery, and the safety layer redirects shares removed from unavailable equipment.
Different raw actions therefore collapse into similar physical allocations. Exploratory
constant policies still produced within roughly 3% of heuristic methane, although they
incurred many more corrections and earned lower total reward.

### Reward and discounting weaken long-horizon pressure

The positive methane term dominates nominal returns while shutdown and reserve terms are
almost inactive. Starting H2 and CO2 inventories are free, so consuming inherited
feedstock earns reward without an inventory cost.

With the default per-minute `gamma = 0.995`, reward weight decays approximately to:

```text
1 hour       0.740
6 hours      0.165
24 hours     0.00073
48 hours     0.00000054
```

A 48-hour episode consequently does not imply that PPO meaningfully values the second
day. Environment V2 must choose policy decision frequency and discounting from an
explicit economic time constant.

## Additional audit risks

- Checkpoint compatibility records but does not validate the simulator package version
  or plant configuration.
- The package reports inconsistent simulator versions in `pyproject.toml` and
  `plant_sim.__version__`.
- Training persists no learning curves, validation score, Git revision, lockfile digest,
  artifact hash, or independently loadable metadata for intermediate checkpoints.
- Evaluation defaults to three seeds and means only; it has no paired uncertainty,
  lower-tail statistic, or held-out regime split.
- The controller-side battery-reserve calculation appears not to account for discharge
  efficiency, unlike the physical battery model. This requires a focused correctness
  test and separate fix before interpreting reserve margins.
- Simulator state is stamped after advancing time while solar flow is calculated at the
  prior instant. Environment V2 needs an explicit `[t, t + dt)` convention before adding
  weather forecasts or delivery schedules.

## Decision

Gate A is not complete until the colleague's checkpoint is evaluated, but the storage
ceiling is sufficiently established to reject “add more PPO tuning” as the next research
step. The first Environment V2 slice will target the commercial scheduling problem:

> Meet contracted methane deliveries under uncertain renewable generation while
> preserving equipment condition.

The minimal discriminative bundle is:

1. recurring methane delivery commitments and explicit shortfall accounting;
2. seeded stochastic solar with an imperfect controller-visible forecast;
3. one genuine commitment dynamic on the hydrogen bottleneck, beginning with
   electrolyser startup/ramp behavior.

These features will be independently switchable so their value can be measured by
ablation.

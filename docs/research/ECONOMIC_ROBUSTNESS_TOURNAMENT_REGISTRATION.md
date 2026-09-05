# Economic robustness tournament — execution registration

Status: **registered before outcome; one matrix execution only**

Date: **2026-09-05**

This registration opens only the development task-readiness matrix admitted by
`ECONOMIC_ROBUSTNESS_TOURNAMENT_ADMISSION.md`.

| Field | Frozen value |
|---|---|
| Executable source revision | `7b1b656964dcac4a25c6f5d4d89fed8abf4d4e0e` |
| Runner contract | `6f789629393188f406c929305721aa257e33e329fc3b5b85da9e5d62d77d7d38` |
| Protocol SHA-256 | `6fb5eb11933ba5ee601849210b4e61278e4cec2ddb97c8e31b05817ee7141412` |
| Dependency-lock SHA-256 | `080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd` |
| Horizons | 6 h, 24 h |
| Delivery rates | 35 kg/h, 42 kg/h |
| Development seeds | 20, 21, 22, 23, 24 |
| Controllers / order | H then M within every cell |
| Cells / trajectories | 20 paired cells / 40 trajectories |
| Decisions / physical minutes | 2,400 / 36,000 |
| Execution worktree | `/private/tmp/plantanagent-economic-robustness-7b1b656` |
| Output directory | `/Users/filippogombac/.codex/visualizations/2026/08/31/01a05842-8e4b-73b1-8810-42296fdaca5b/v2_economic_robustness_7b1b656` |

Execution must occur sequentially in a fresh clean detached worktree at the exact source
revision above. The output directory and worktree must not exist at registration time. The
project's already-installed Python runtime may be used only with the detached `backend/src` and
`simulator/src` placed first on its module search path; the runner locates and records the Git
root from its detached source file.

The registered module invocation is logically:

```text
python -m plant_backend.rl.v2.economic_robustness_experiment run \
  /Users/filippogombac/.codex/visualizations/2026/08/31/01a05842-8e4b-73b1-8810-42296fdaca5b/v2_economic_robustness_7b1b656
```

The first execution is final for this source revision whether a controller fails, an R gate
fails, or all gates pass. The runner publishes each cell before proceeding and retains failure.
No second destination, same-revision retry, seed substitution, controller repair, parameter
adjustment, or concurrent controller cell is authorized. The process may inspect controller
outcomes only after `report.json` has closed and the deep whole-directory verifier has returned.

An infrastructure failure before Python initialization, output-directory creation, environment
reset, or controller action does not consume a cell, but it must be recorded before any corrected
launch. Once any seed 20--24 environment resets, all five seeds remain development evidence and
may never be relabelled as holdout or formal validation data.

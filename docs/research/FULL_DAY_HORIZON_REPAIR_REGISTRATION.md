# Full-day horizon repair — execution registration

Status: **registered before outcome; one ten-cell M24 execution only**

Date: **2026-09-05**

This registration opens only the contaminated-development mechanism probe admitted by
`FULL_DAY_HORIZON_REPAIR_ADMISSION.md`. It authorizes no horizon sweep, RL training, H/M6 rerun,
held-out evaluation, or commercial interpretation.

| Field | Frozen value |
|---|---|
| Executable source revision | `d18082594b7fea692c9fff146c54d9b0f2f31733` |
| Source branch | `codex/m24-horizon-execution` |
| Runner contract | `d70857ef63a838ce9eb94767739e7492517983665977039e9d8d39547c7faae8` |
| Protocol SHA-256 | `e203d1bb12246bc43278b686779afdfde310f8b56e487dac7f9d05947079e0e1` |
| Clarification 001 SHA-256 | `8dc341b142a41214b7845b0f5debab29b7f23751ff8f7cd71ab68ac30f6b7ce3` |
| Clarification 002 SHA-256 | `e2ed43908242a228669859fb7111e4b4f24513d23f467daf60b8c9b1f7692966` |
| Admission-record SHA-256 | `0a96b0763696e01bf8c54f0a250ba2a332d3a5dd6560a5c8e4bbcf6abbf33bd6` |
| Dependency-lock SHA-256 | `080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd` |
| Historical source revision | `7b1b656964dcac4a25c6f5d4d89fed8abf4d4e0e` |
| Historical raw report | `26c9ab4a85693dab05d93f7a1b65e86564c74bfe667ee54813901a466d1e0755` |
| Historical reference report | `/Users/filippogombac/.codex/visualizations/2026/08/31/01a05842-8e4b-73b1-8810-42296fdaca5b/v2_economic_robustness_7b1b656/report.json` |
| Planning-horizon intervention | M6 6 h to M24 24 h; every other setting fixed |
| Delivery rates / order | 35 kg/h then 42 kg/h |
| Development seeds / order | 20, 21, 22, 23, 24 within each rate |
| New trajectories | 10 M24 cells, sequential |
| Expected decisions / solver stages / physical minutes | 960 / 1,920 / 14,400 |
| Detached execution worktree | `/private/tmp/plantanagent-m24-d180825` |
| New external output directory | `/Users/filippogombac/.codex/visualizations/2026/08/31/01a05842-8e4b-73b1-8810-42296fdaca5b/v2_full_day_horizon_repair_d180825` |

The worktree and output directory were confirmed absent when this registration was written. The
execution must occur from a fresh detached worktree at the exact revision above, with that
worktree's `backend/src` and `simulator/src` first on the already installed Python runtime's
module search path. The runner must first deeply admit the immutable 21-file historical
directory and current clean source before it creates the destination or resets an environment.

The registered invocation is logically:

```text
python -m plant_backend.rl.v2.full_day_horizon_repair_experiment run \
  /Users/filippogombac/.codex/visualizations/2026/08/31/01a05842-8e4b-73b1-8810-42296fdaca5b/v2_economic_robustness_7b1b656/report.json \
  /Users/filippogombac/.codex/visualizations/2026/08/31/01a05842-8e4b-73b1-8810-42296fdaca5b/v2_full_day_horizon_repair_d180825
```

The first protected execution is final for this revision whether a cell fails, a scientific gate
fails, or all gates pass. Every cell is published and deeply verified immediately; a failed cell
continues to later cells without retry, substitution, parameter change, or surviving-subset
analysis. H and M6 remain immutable references and are never rerun.

An infrastructure failure before Python initialization, destination creation, environment reset,
or controller action does not consume the registered run, but it must be recorded before a
corrected launch. After any seed 20--24 reset, there is no same-revision retry and these seeds
remain development-only. Outcomes may be inspected only after `report.json` is exclusively
published and the independent whole-directory verifier returns.

The mandatory interpretation remains:

> Contaminated development mechanism only; no RL, controller-superiority, calibrated-economic,
> held-out, site, commercial, savings, commission, or generalization claim.

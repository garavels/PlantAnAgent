# Absolute-setpoint MPC one-cell execution registration

Status: **registered before outcome; one execution only**

Date: **2026-09-04**

This registration opens only the single mechanism cell admitted by
`CAUSAL_SERVICE_MPC_SETPOINT_DISPATCH_ADMISSION.md`.

| Field | Frozen value |
|---|---|
| Executable source revision | `a968a343f50e5724ea74b75cc3debd1509fc1d80` |
| Runner contract | `421561b2ee75650024c11fdf55d806012275b44838a33f6b3714635fa689e304` |
| Delivery rate | 35 kg/h |
| Seed | 0 |
| Simulated horizon | 72 h |
| Decisions / physical minutes | 288 / 4,320 |
| Output | `/Users/filippogombac/.codex/visualizations/2026/08/31/01a05842-8e4b-73b1-8810-42296fdaca5b/v2_service_mpc_setpoint_a968a34_rate35_seed0.json` |

Execution must occur in a fresh clean detached worktree at the exact revision above. The
output must not already exist. The command is:

```text
python -m plant_backend.rl.v2.mpc_setpoint_experiment \
  --output /Users/filippogombac/.codex/visualizations/2026/08/31/01a05842-8e4b-73b1-8810-42296fdaca5b/v2_service_mpc_setpoint_a968a34_rate35_seed0.json
```

The first published artifact is final for this revision whether execution fails, an S-gate
fails, or all gates pass. No second destination, same-revision retry, parameter adjustment,
or concurrent experiment is authorized. The process may only inspect the result after the
runner has closed and published it.

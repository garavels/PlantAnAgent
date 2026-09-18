# PlantAnAgent

A digital twin and learning environment for an off-grid, solar-powered methane plant.

PlantAnAgent connects solar generation, battery storage, hydrogen production, direct air capture, methanation, and scheduled fuel deliveries. Its control room lets you inspect the plant in 3D, follow controller decisions, explore weather and faults, and compare learned policies with rule-based operation.

**[Run the app](#run-the-app) · [Explore the demo](#a-five-minute-walkthrough) · [Reproduce the experiments](#reproduce-the-experiments) · [Verify the code](#verification)**

## Run the app

Requires **Python 3.11+**, [uv](https://docs.astral.sh/uv/), **Node.js 22.12+**, npm, and Make. The commands below run from the repository root.

```bash
git clone --depth 1 https://github.com/garavels/PlantAnAgent.git
cd PlantAnAgent
make setup
make dev
```

Open **http://localhost:5173**. The backend runs on port **8000**; its API explorer is at **http://localhost:8000/docs**. Stop both services with Ctrl+C.

You can also use GitHub's **Code → Download ZIP**, extract it, and run `make setup` and `make dev` inside the extracted directory. Git is not needed to open the app or use the reproduction commands below.

The featured recording and saved site forecasts are included. **No API key or GPU is required to explore them.** Installation needs internet access; fetching fresh weather also needs a connection.

To run the services separately:

```bash
make backend
# In another terminal:
make frontend
```

## A five-minute walkthrough

1. **Overview:** open the featured trained-policy recording. Use **Play recording** or **Final result** to inspect a completed seven-day simulation.
2. **Plant:** follow electricity and gas flows in the 3D twin. Select equipment to inspect its state; turn on **Agent view** for an explanation of the recorded control decisions.
3. **Deliveries and Economics:** inspect delivery obligations, unmet demand, stored resources, and modeled operating contribution.
4. **Models:** inspect the policy and its evaluation context. The featured successful week is one example from a broader comparison.
5. **Site and Scenarios:** explore saved European forecasts or build a scenario using the editable controls. Interactive simulations are separate from the read-only recording.

Optional language-model features:

| Feature | Environment variables |
| --- | --- |
| Natural-language scenario generation | `OPENROUTER_API_KEY`; optionally `PLANT_SCENARIO_MODEL` |
| LLM supervisory agent | `OPENAI_API_KEY`; optionally `PLANT_AGENT_MODEL` |

Export variables in the terminal before starting the backend. Editable scenarios and the recorded demo work without them. Live economic planning uses SciPy, installed by `make setup`.

## What is included

- A deterministic component simulator with energy and material accounting, storage limits, equipment readiness, faults, and deliveries.
- A FastAPI backend and React/Three.js control room with saved playback and interactive scenarios.
- Controller implementations, training code, evaluation and physical-audit code, and regression tests.
- Small runtime fixtures, benchmark definitions, the frozen experimental simulator, and the selected learned-policy checkpoints required for reproduction.

The app's featured playback and the latest research comparison are **different experiments**:

| | Featured app playback | Learned-control experiment |
| --- | --- | --- |
| Methane demand | 25 kg/h; 150 kg every six hours | 30 kg/h; 180 kg every six hours |
| Duration | Seven days | Seven days per case |
| Purpose | Inspect one successful trained-policy week | Compare policies across ordinary operation and prescribed disruptions |
| Controllers | Earlier coverage-residual PPO | Imitation-initialized policies improved with cross-entropy search, plus daylight and storage baselines |

The latest selected policies use **derivative-free cross-entropy search**, not PPO updates. The repository also contains the earlier PPO experiments. Do not interpret the successful playback as proof that learned control always outperforms storage-aware rules. All results concern a simplified simulator; operating contribution excludes uncalibrated solar capital and fixed costs.

## Reproduce the experiments

Install the optional training dependencies:

```bash
make setup-research
```

The reproduction runner uses the included frozen physical source and authenticates checkpoint bytes. Every run requires a **new output directory**, so it cannot overwrite previous evidence.

### Quick engineering check

```bash
make research-smoke
# To repeat it, choose another output directory:
make research-smoke OUT=output/smoke-2
```

This exercises the selected models and baselines for four decisions on a training seed. It is a wiring check, not a complete week or a performance result.

### Re-evaluate the published policies

```bash
uv run --no-sync python tools/reproduce.py evaluate \
  --output output/reproduction-evaluation --workers 3
```

This runs the predetermined 32-scenario, seven-condition comparison using the included checkpoints and writes the frozen run specification, individual case records, `summary.json`, and `analysis.json`. It audits physical steps and retains failed outcomes. This is a substantial CPU workload, not part of app startup. Reusing the published scenarios is reproduction, not new held-out evidence.

Inspect the complete recipe or prepare an evaluation without executing it:

```bash
uv run --no-sync python tools/reproduce.py plan
uv run --no-sync python tools/reproduce.py evaluate \
  --output output/prepared-evaluation --prepare-only
```

### Regenerate the selected training pipeline

```bash
uv run --no-sync python tools/reproduce.py train \
  --output output/reproduction-training --workers 3

uv run --no-sync python tools/reproduce.py evaluate \
  --training-root output/reproduction-training \
  --output output/retrained-evaluation --workers 3
```

Training regenerates teacher demonstrations, Gaussian imitation, learner-visited demonstrations, hybrid imitation, the solar-action refit, and all three selected CEM training attempts. The evaluation checkpoint is fixed at generation 15, matching the published selection; it is not chosen by the new evaluation outcomes. This reproduces the selected pipeline, not every exploratory candidate investigated during development.

The original seed allocation and numerical settings are in [experiments/reproduction.json](experiments/reproduction.json). Runtime deadlines are refreshed for a new execution; override them with `--deadline` using a future ISO timestamp with a timezone. The default deadline is seven days, a runtime cap rather than a completion estimate. Hardware and numerical-library differences can affect newly trained weights; bitwise-identical retraining across platforms is not promised.

## Verification

```bash
make test              # current simulator and backend
make test-frontend     # frontend regression tests
make build             # TypeScript checks and production frontend build

# After make setup-research:
make test-research     # frozen experiment and learning tests
```

The Python test targets install the locked training and optimization dependencies, since they include controller-training fixtures. Use a Git clone for the full backend suite: historical benchmark tests verify commit provenance. App tests and experiment tests run in separate Python processes because they intentionally use different simulator versions. `make research-smoke` additionally checks checkpoint loading and the experiment execution path.

## Code map

| Path | Purpose |
| --- | --- |
| `simulator/src/plant_sim/` | Current app physics and component state |
| `backend/src/plant_backend/` | API, simulation clock, controllers and validation |
| `frontend/src/` | Control room and 3D plant |
| `docs/research/tools/` | Experiment implementations, retained at their original import paths |
| `runs/authority-source-20260917/` | Frozen physical source required by the experiments |
| `experiments/` | Portable recipes, selected checkpoint inputs and reference evaluation specification |
| `tools/reproduce.py` | Fresh-directory reproduction entry point |
| `tests/`, `backend/tests/`, `simulator/tests/`, `frontend/tests/` | Verification suites |

Protocol files under `docs/research/` are immutable inputs whose bytes are checked by the experiment code. New training outputs and evaluation traces are written outside the tracked source. Third-party attribution is included beside the corresponding asset.

Built by **Filippo Gombac** and **Vittorio Garavelli**.

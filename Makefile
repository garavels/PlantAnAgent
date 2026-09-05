.PHONY: help setup dev backend frontend sim test eval foresight-eval rl-eval rl-ceiling v2-eval v2-weather-eval v2-controller-eval v2-planner v2-mpc v2-mpc-report v2-mpc-setpoint v2-economic-smoke v2-economic-robustness v2-economic-robustness-verify v2-full-day-horizon-repair v2-full-day-horizon-repair-verify v2-train-dry v2-train v2-rl-eval v2-controller-benchmark train-dry train \
	hepa-collect hepa-pretrain-dry hepa-pretrain hepa-finetune-dry hepa-finetune \
	hepa-eval build clean

help:             ## list targets
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

setup:            ## install python + node dependencies
	uv sync
	cd frontend && npm install

backend:          ## run the control backend on :8000
	uv run uvicorn plant_backend.app:app --reload --port 8000

frontend:         ## run the control room on :5173
	cd frontend && npm run dev

dev:              ## run both (ctrl-c stops both)
	@$(MAKE) -j2 backend frontend

eval:             ## run the supervisory agent against the five scenarios
	uv run python -m plant_backend.agent.evaluation

foresight-eval:   ## compare controller / +supervisor / +HEPA (fixture forecasts)
	uv run python -m plant_backend.agent.foresight_evaluation

rl-eval:          ## evaluate the heuristic baseline (add CHECKPOINT=<dir> for a policy)
	uv run python -m plant_backend.rl.evaluation --seeds 1 2 3 --hours 24 \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),)

rl-ceiling:       ## diagnose whether CH4 tank headroom caps the V1 benchmark
	uv run python -m plant_backend.rl.evaluation.ceiling --seed-start 0 \
		--count $(or $(SEEDS),100) --hours $(or $(HOURS),48) \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) $(if $(JSON),--json $(JSON),)

v2-eval:          ## run paired transparent baselines on the V2 commitment alpha
	uv run python -m plant_backend.rl.v2 --seed-start $(or $(SEED_START),0) \
		--count $(or $(SEEDS),20) --hours $(or $(HOURS),168) \
		--delivery-rate $(or $(DELIVERY_RATE),40) $(if $(JSON),--json $(JSON),)

v2-weather-eval: v2-eval  ## compatibility alias for the pre-commitment target name

v2-controller-eval: ## compare causal delivery-aware controllers on the development task
	uv run python -m plant_backend.rl.v2.compare \
		--seed-start $(or $(SEED_START),0) --count $(or $(SEEDS),3) \
		--hours $(or $(HOURS),72) --delivery-rate $(or $(DELIVERY_RATE),35) \
		$(if $(ALLOW_DIRTY_DEVELOPMENT),--allow-dirty-development,) \
		$(if $(ACKNOWLEDGE_SEED_CONTAMINATION),--acknowledge-seed-contamination,) \
		$(if $(JSON),--json $(JSON),)

v2-planner:       ## run one registered future-information planner cell (external OUTPUT required)
	$(if $(OUTPUT),,$(error OUTPUT=/absolute/path/outside/repository.json is required))
	uv run --frozen python -m plant_backend.rl.v2.planner \
		--delivery-rate $(or $(DELIVERY_RATE),35) --seed $(or $(SEED),0) \
		--output $(OUTPUT)

v2-mpc:           ## run one registered causal service-MPC cell (external OUTPUT required)
	$(if $(OUTPUT),,$(error OUTPUT=/absolute/path/outside/repository.json is required))
	uv run --frozen python -m plant_backend.rl.v2.mpc_experiment \
		--delivery-rate $(or $(DELIVERY_RATE),35) --seed $(or $(SEED),0) \
		--output $(OUTPUT)

v2-mpc-report:    ## aggregate six registered causal service-MPC cells (external paths required)
	$(if $(CELL_35_0),,$(error CELL_35_0=/absolute/rate35-seed0.json is required))
	$(if $(CELL_35_1),,$(error CELL_35_1=/absolute/rate35-seed1.json is required))
	$(if $(CELL_35_2),,$(error CELL_35_2=/absolute/rate35-seed2.json is required))
	$(if $(CELL_42_0),,$(error CELL_42_0=/absolute/rate42-seed0.json is required))
	$(if $(CELL_42_1),,$(error CELL_42_1=/absolute/rate42-seed1.json is required))
	$(if $(CELL_42_2),,$(error CELL_42_2=/absolute/rate42-seed2.json is required))
	$(if $(OUTPUT),,$(error OUTPUT=/absolute/path/outside/repository.json is required))
	uv run --frozen python -m plant_backend.rl.v2.mpc_report \
		--cell $(CELL_35_0) --cell $(CELL_35_1) --cell $(CELL_35_2) \
		--cell $(CELL_42_0) --cell $(CELL_42_1) --cell $(CELL_42_2) \
		--output $(OUTPUT)

v2-mpc-setpoint:  ## run the single frozen absolute-setpoint mechanism cell (external OUTPUT required)
	$(if $(OUTPUT),,$(error OUTPUT=/absolute/path/outside/repository.json is required))
	uv run --frozen python -m plant_backend.rl.v2.mpc_setpoint_experiment \
		--output $(OUTPUT)

v2-economic-smoke: ## run the single frozen heuristic/MPC economic mechanism cell
	$(if $(OUTPUT),,$(error OUTPUT=/absolute/path/outside/repository.json is required))
	uv run --frozen python -m plant_backend.rl.v2.economic_smoke_experiment $(OUTPUT)

v2-economic-robustness: ## run the frozen 20-cell task-robustness tournament
	$(if $(OUTPUT),,$(error OUTPUT=/absolute/new/directory/outside/repository is required))
	uv run --frozen python -m plant_backend.rl.v2.economic_robustness_experiment \
		run $(OUTPUT)

v2-economic-robustness-verify: ## deeply verify a robustness report and all cell artifacts
	$(if $(REPORT),,$(error REPORT=/absolute/path/to/report.json is required))
	uv run --frozen python -m plant_backend.rl.v2.economic_robustness_experiment \
		verify $(REPORT)

v2-full-day-horizon-repair: ## run the frozen ten-cell M24 horizon-only probe
	$(if $(REFERENCE_REPORT),,$(error REFERENCE_REPORT=/absolute/path/to/historical/report.json is required))
	$(if $(OUTPUT),,$(error OUTPUT=/absolute/new/directory/outside/repository is required))
	uv run --frozen python -m plant_backend.rl.v2.full_day_horizon_repair_experiment \
		run $(REFERENCE_REPORT) $(OUTPUT)

v2-full-day-horizon-repair-verify: ## deeply verify an M24 report and both evidence directories
	$(if $(REPORT),,$(error REPORT=/absolute/path/to/report.json is required))
	uv run --frozen python -m plant_backend.rl.v2.full_day_horizon_repair_experiment \
		verify $(REPORT)

v2-train-dry:     ## identify the controller-neutral RL task; trains nothing
	uv run --frozen python -m plant_backend.rl.v2.training --dry-run \
		$(if $(CONFIG),--config $(CONFIG),)

v2-train:         ## train PPO through the four-target door (long; opt-in)
	uv run --frozen python -m plant_backend.rl.v2.training \
		$(if $(CONFIG),--config $(CONFIG),) $(if $(OUTPUT),--output $(OUTPUT),)

v2-rl-eval:       ## pair a compatible four-target checkpoint with the heuristic
	$(if $(CHECKPOINT),,$(error CHECKPOINT=<checkpoint-directory> is required))
	uv run --frozen python -m plant_backend.rl.v2.policy_evaluation \
		--checkpoint $(CHECKPOINT) --seed-start $(or $(SEED_START),0) \
		--count $(or $(SEEDS),1) $(if $(JSON),--json $(JSON),)

v2-controller-benchmark: ## compare H, M24, economic MPC, and trained PPO (exclusive OUTPUT required)
	$(if $(CHECKPOINT),,$(error CHECKPOINT=<trained-checkpoint-directory> is required))
	$(if $(OUTPUT),,$(error OUTPUT=/absolute/new/report.json is required))
	uv run --frozen --group train --group optimisation \
		python -m plant_backend.rl.v2.controller_benchmark \
		--checkpoint $(CHECKPOINT) --seed-start $(or $(SEED_START),0) \
		--count $(or $(SEEDS),1) --json $(OUTPUT)

train-dry:        ## build the RL environment and print the training plan; trains nothing
	uv run python -m plant_backend.rl.training.train --dry-run

train:            ## train PPO (long; needs `uv sync --group train`)
	uv run python -m plant_backend.rl.training.train $(if $(CONFIG),--config $(CONFIG),)

sim:              ## run one simulated day headless
	uv run python -m plant_sim --hours 24

hepa-collect:     ## collect a HEPA telemetry dataset (EPISODES=<n> DATASET=<dir>)
	uv run python -m plant_backend.hepa.data.collect \
		--config $(or $(CONFIG),backend/src/plant_backend/hepa/configs/collect.json) \
		--out $(or $(DATASET),datasets/hepa) $(if $(EPISODES),--episodes $(EPISODES),)

hepa-pretrain-dry: ## print the stage-1 plan; trains nothing
	uv run python -m plant_backend.hepa.pretraining \
		--dataset $(or $(DATASET),datasets/hepa) --dry-run

hepa-pretrain:    ## stage-1 self-supervised pretraining (long; needs `uv sync --group hepa`)
	uv run python -m plant_backend.hepa.pretraining \
		--config $(or $(CONFIG),backend/src/plant_backend/hepa/configs/pretrain.json) \
		--dataset $(or $(DATASET),datasets/hepa) --out $(or $(OUT),checkpoints/hepa/stage1)

hepa-finetune-dry: ## print the stage-2 plan; trains nothing
	uv run python -m plant_backend.hepa.finetuning \
		--dataset $(or $(DATASET),datasets/hepa) \
		--pretrained $(or $(PRETRAINED),checkpoints/hepa/stage1) --dry-run

hepa-finetune:    ## stage-2 event fine-tuning (long; needs `uv sync --group hepa`)
	uv run python -m plant_backend.hepa.finetuning \
		--config $(or $(CONFIG),backend/src/plant_backend/hepa/configs/finetune.json) \
		--dataset $(or $(DATASET),datasets/hepa) \
		--pretrained $(or $(PRETRAINED),checkpoints/hepa/stage1) \
		--out $(or $(OUT),checkpoints/hepa/stage2)

hepa-eval:        ## score a stage-2 checkpoint per event and per horizon
	uv run python -m plant_backend.hepa.evaluation \
		--config $(or $(CONFIG),backend/src/plant_backend/hepa/configs/evaluate.json) \
		--checkpoint $(or $(CHECKPOINT),checkpoints/hepa/stage2) \
		--dataset $(or $(DATASET),datasets/hepa)

test:             ## simulator + backend test suite
	uv run pytest -q

build:            ## typecheck + build the frontend
	cd frontend && npm run build

clean:
	rm -rf .venv frontend/node_modules frontend/dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

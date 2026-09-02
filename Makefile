.PHONY: help setup dev backend frontend sim test eval foresight-eval rl-eval rl-ceiling v2-weather-eval train-dry train \
	hepa-collect hepa-pretrain-dry hepa-pretrain hepa-finetune-dry hepa-finetune \
	hepa-eval build clean

help:             ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

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

v2-weather-eval:  ## run paired transparent baselines on the V2 weather alpha
	uv run python -m plant_backend.rl.v2 --seed-start $(or $(SEED_START),0) \
		--count $(or $(SEEDS),20) --hours $(or $(HOURS),168) \
		--delivery-rate $(or $(DELIVERY_RATE),40) $(if $(JSON),--json $(JSON),)

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

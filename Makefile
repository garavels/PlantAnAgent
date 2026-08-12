.PHONY: help setup dev backend frontend sim test build clean

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

sim:              ## run one simulated day headless
	uv run python -m plant_sim --hours 24

test:             ## simulator + backend test suite
	uv run pytest -q

build:            ## typecheck + build the frontend
	cd frontend && npm run build

clean:
	rm -rf .venv frontend/node_modules frontend/dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

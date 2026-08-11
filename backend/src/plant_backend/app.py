"""Development API in front of the simulator.

Deliberately thin: it holds one `PlantSession` in memory and exposes it over
HTTP so the frontend never imports the simulator directly. The clock lives in
the browser -- the UI decides when to call `/api/step`, the backend only decides
what happens when it does. That is the seam where controllers and agents will
eventually take over.

    uv run uvicorn plant_backend.app:app --reload --port 8000
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .session import PlantSession

app = FastAPI(title="Plant An Agent -- dev API", version="0.1.0")

# Dev only: the Vite dev server proxies /api, but allow direct calls too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

session = PlantSession()


class ActionModel(BaseModel):
    """How to split the available power this timestep. See `plant_sim.Action`."""

    battery_share: float = 0.5
    electrolyser_share: float = 0.5
    battery_support_kw: float = 0.0


class StepRequest(BaseModel):
    action: ActionModel = Field(default_factory=ActionModel)
    #: Fast-forward: apply the same action this many timesteps in one call.
    repeat: int = Field(default=1, ge=1, le=1000)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "step": session.simulator.step_index}


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    return session.config


@app.get("/api/state")
def get_state(history: int = 240) -> dict[str, Any]:
    """Current state, plus recent frames so a reloaded UI can redraw its charts."""
    return {"state": session.state, "frames": session.recent(history), "config": session.config}


@app.post("/api/step")
def post_step(request: StepRequest) -> dict[str, Any]:
    """Advance the plant. Returns every frame produced, newest last."""
    frames = session.step(request.action.model_dump(), request.repeat)
    return {"state": frames[-1], "frames": frames}


@app.post("/api/reset")
def post_reset() -> dict[str, Any]:
    state = session.reset()
    return {"state": state, "frames": [state]}

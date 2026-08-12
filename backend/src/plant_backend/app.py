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

from fastapi import FastAPI, HTTPException
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
    dac_share: float = 0.0
    methanator_share: float = 0.0
    battery_support_kw: float = 0.0


class StepRequest(BaseModel):
    action: ActionModel = Field(default_factory=ActionModel)
    #: Fast-forward: apply the same action this many timesteps in one call.
    repeat: int = Field(default=1, ge=1, le=1000)


class FaultRequest(BaseModel):
    """A fault or disturbance to inject. See `plant_sim.faults.CATALOGUE`."""

    type: str
    severity: float = Field(default=0.4, ge=0.0, le=1.0)
    #: Absolute simulator time; omitted means "start now".
    start_s: float | None = None
    #: Omitted means "until cleared".
    duration_s: float | None = Field(default=None, ge=0.0)
    #: Sensor faults only: which telemetry channel to bias.
    channel: str | None = None


class ScenarioRequest(BaseModel):
    scenario: str


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "step": session.simulator.step_index}


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    return session.config


@app.get("/api/state")
def get_state(history: int = 240) -> dict[str, Any]:
    """Current telemetry, plus recent frames so a reloaded UI can redraw its charts."""
    return {
        "state": session.state,
        "frames": session.recent(history),
        "config": session.config,
        **session.conditions(),
    }


@app.post("/api/step")
def post_step(request: StepRequest) -> dict[str, Any]:
    """Advance the plant. Returns every frame produced, newest last."""
    frames = session.step(request.action.model_dump(), request.repeat)
    return {"state": frames[-1], "frames": frames, **session.conditions()}


@app.post("/api/reset")
def post_reset() -> dict[str, Any]:
    state = session.reset()
    return {"state": state, "frames": [state], **session.conditions()}


# --- abnormal operation ----------------------------------------------------
@app.get("/api/events")
def get_events(limit: int = 60) -> dict[str, Any]:
    """What the plant noticed, oldest first."""
    return {"events": session.events(limit)}


@app.get("/api/faults")
def get_faults() -> dict[str, Any]:
    """Injectable fault types, and the ones currently scheduled or running."""
    return {"catalogue": session.fault_catalogue(), "faults": session.faults}


@app.post("/api/faults")
def post_fault(request: FaultRequest) -> dict[str, Any]:
    """Inject a fault or disturbance, immediately or at a scheduled time."""
    try:
        fault = session.inject_fault(request.model_dump())
    except ValueError as reason:
        raise HTTPException(status_code=400, detail=str(reason)) from reason
    return {"fault": fault, "state": session.state, **session.conditions()}


@app.delete("/api/faults")
def delete_faults() -> dict[str, Any]:
    """Clear every fault, leaving the plant where it is."""
    session.clear_faults()
    return {"state": session.state, **session.conditions()}


@app.delete("/api/faults/{fault_id}")
def delete_fault(fault_id: str) -> dict[str, Any]:
    if not session.clear_fault(fault_id):
        raise HTTPException(status_code=404, detail=f"unknown fault: {fault_id}")
    return {"state": session.state, **session.conditions()}


@app.get("/api/scenarios")
def get_scenarios() -> dict[str, Any]:
    return {"scenarios": session.scenario_catalogue(), "scenario": session.scenario}


@app.post("/api/scenarios")
def post_scenario(request: ScenarioRequest) -> dict[str, Any]:
    """Reset the plant into a predefined, repeatable situation."""
    try:
        state = session.load_scenario(request.scenario)
    except ValueError as reason:
        raise HTTPException(status_code=404, detail=str(reason)) from reason
    return {"state": state, "frames": [state], **session.conditions()}

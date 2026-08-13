"""HTTP surface of the plant-control backend.

    uv run uvicorn plant_backend.app:app --reload --port 8000

The frontend reaches the plant only through here, and only ever *asks*: what is
the state, please run, please stop, please inject this fault, please switch to
manual. The simulation loop, the controller and the action guard all live behind
this layer (`orchestration`, `controllers`, `safety`), so there is no way for a
client to hand the simulator an action that was not validated first.

Live state arrives over SSE at `/api/stream` -- one self-contained payload per
tick. The REST routes are for asking questions and for changing things; the
stream is for watching.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .hepa.api import configure as configure_hepa
from .hepa.api import router as hepa_router
from .orchestration import Broadcaster, PlantRuntime, SimulationClock

#: Seconds without an update before the stream sends a keepalive comment.
KEEPALIVE_S = 15.0

runtime = PlantRuntime()
broadcast = Broadcaster()
clock = SimulationClock(runtime, broadcast)


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    clock.start()
    try:
        yield
    finally:
        await clock.stop()


app = FastAPI(title="Plant An Agent -- control backend", version="0.6.0", lifespan=lifespan)

# V9: event forecasting, advisory only. The router is read-only and never touches the
# controller; with no checkpoint configured its routes answer "unavailable" and the
# rest of the backend is unaffected. It reads the session's rolling telemetry history.
app.include_router(hepa_router)
configure_hepa(lambda: list(runtime.session.history))

# Dev only: the Vite dev server proxies /api, but allow direct calls too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ActionModel(BaseModel):
    """How to split the available power. See `plant_sim.Action`."""

    battery_share: float = 0.5
    electrolyser_share: float = 0.5
    dac_share: float = 0.0
    methanator_share: float = 0.0
    battery_support_kw: float = 0.0


class StepRequest(BaseModel):
    """Advance the plant by hand. Used when the clock is paused."""

    #: Manual mode only: the allocation to apply (and remember).
    action: ActionModel | None = None
    repeat: int = Field(default=1, ge=1, le=1000)


class SpeedRequest(BaseModel):
    #: Timesteps applied per tick.
    speed: int = Field(ge=1, le=64)


class ControllerConfigRequest(BaseModel):
    """Change how the plant is driven. Every field is optional."""

    mode: str | None = None
    controller: str | None = None
    #: Partial update of the operating envelope. See `controllers.Constraints`.
    constraints: dict[str, float | None] | None = None
    #: The manual setpoint; stored now, validated when it is applied.
    action: ActionModel | None = None


class RLCheckpointRequest(BaseModel):
    """Load an RL checkpoint, or unload the current one with a null path."""

    checkpoint: str | None = None


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


class AgentConfigRequest(BaseModel):
    """Turn the supervisory agent on or off. Off leaves V5 behaviour exactly."""

    enabled: bool


def _publish(payload: dict[str, Any]) -> dict[str, Any]:
    """Send an update to every connected client and answer the caller with it."""
    broadcast.publish(payload)
    return payload


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "step": runtime.session.simulator.step_index,
        "clock": clock.running,
        "clients": broadcast.subscribers,
    }


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    return runtime.session.config


@app.get("/api/state")
def get_state(history: int = 240) -> dict[str, Any]:
    """Everything a freshly loaded control room needs, including its charts."""
    return {
        **runtime.payload(runtime.session.recent(history), replace=True),
        "config": runtime.session.config,
    }


@app.get("/api/stream")
async def stream() -> StreamingResponse:
    """Server-sent events: one payload per tick, plus one on every change."""
    queue = broadcast.subscribe()

    async def events() -> AsyncIterator[str]:
        try:
            snapshot = {
                **runtime.payload(runtime.session.recent(240), replace=True),
                "config": runtime.session.config,
            }
            yield f"data: {json.dumps(snapshot)}\n\n"
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_S)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(payload)}\n\n"
        finally:
            broadcast.unsubscribe(queue)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"cache-control": "no-store", "x-accel-buffering": "no"},
    )


# --- simulation controls ---------------------------------------------------
@app.post("/api/simulation/play")
async def post_play() -> dict[str, Any]:
    runtime.play()
    return _publish(runtime.payload())


@app.post("/api/simulation/pause")
async def post_pause() -> dict[str, Any]:
    runtime.pause()
    return _publish(runtime.payload())


@app.post("/api/simulation/speed")
async def post_speed(request: SpeedRequest) -> dict[str, Any]:
    runtime.set_speed(request.speed)
    return _publish(runtime.payload())


@app.post("/api/simulation/step")
@app.post("/api/step")
async def post_step(request: StepRequest | None = None) -> dict[str, Any]:
    """Advance the plant by hand: the same decide/validate/apply path as the clock."""
    ask = request or StepRequest()
    if ask.action is not None:
        runtime.set_manual_action(ask.action.model_dump())
    frames = runtime.advance(ask.repeat)
    return _publish(runtime.payload(frames))


@app.post("/api/simulation/reset")
@app.post("/api/reset")
async def post_reset() -> dict[str, Any]:
    state = runtime.reset()
    return _publish(runtime.payload([state], replace=True))


# --- control ---------------------------------------------------------------
@app.get("/api/controller")
def get_controller() -> dict[str, Any]:
    """Mode, controller, envelope, and why the last action looked as it did."""
    return runtime.control_status()


@app.post("/api/controller/config")
async def post_controller_config(request: ControllerConfigRequest) -> dict[str, Any]:
    """Switch AUTO/MANUAL, pick a controller, retune constraints, set the setpoint."""
    try:
        if request.mode is not None:
            runtime.set_mode(request.mode)
        if request.controller is not None:
            runtime.set_controller(request.controller)
        if request.constraints is not None:
            runtime.configure_constraints(dict(request.constraints))
        if request.action is not None:
            runtime.set_manual_action(request.action.model_dump())
    except ValueError as reason:
        raise HTTPException(status_code=400, detail=str(reason)) from reason
    return _publish(runtime.payload())


@app.post("/api/controller/rl")
async def post_controller_rl(request: RLCheckpointRequest) -> dict[str, Any]:
    """Load or unload the RL checkpoint. Loading never switches controller by itself.

    A checkpoint that is missing, corrupt or trained against a different
    observation is a 400 with the reason: the plant carries on with the heuristic
    controller rather than running something that cannot be verified.
    """
    try:
        if request.checkpoint:
            runtime.load_rl_checkpoint(request.checkpoint)
        else:
            runtime.unload_rl_checkpoint()
    except ValueError as reason:
        raise HTTPException(status_code=400, detail=str(reason)) from reason
    return _publish(runtime.payload())


# --- abnormal operation ----------------------------------------------------
@app.get("/api/events")
def get_events(limit: int = 60) -> dict[str, Any]:
    """What the plant noticed, oldest first."""
    return {"events": runtime.session.events(limit)}


@app.get("/api/faults")
def get_faults() -> dict[str, Any]:
    """Injectable fault types, and the ones currently scheduled or running."""
    return {
        "catalogue": runtime.session.fault_catalogue(),
        "faults": runtime.session.faults,
    }


@app.post("/api/faults")
async def post_fault(request: FaultRequest) -> dict[str, Any]:
    """Inject a fault or disturbance, immediately or at a scheduled time."""
    try:
        fault = runtime.session.inject_fault(request.model_dump())
    except ValueError as reason:
        raise HTTPException(status_code=400, detail=str(reason)) from reason
    return {"fault": fault, **_publish(runtime.payload())}


@app.delete("/api/faults")
async def delete_faults() -> dict[str, Any]:
    """Clear every fault, leaving the plant where it is."""
    runtime.session.clear_faults()
    return _publish(runtime.payload())


@app.delete("/api/faults/{fault_id}")
async def delete_fault(fault_id: str) -> dict[str, Any]:
    if not runtime.session.clear_fault(fault_id):
        raise HTTPException(status_code=404, detail=f"unknown fault: {fault_id}")
    return _publish(runtime.payload())


# --- supervisory agent -----------------------------------------------------
@app.get("/api/agent")
def get_agent(trace: int = Query(60, ge=0, le=240)) -> dict[str, Any]:
    """Supervisor status, the current incident, and its recent trace."""
    return runtime.supervisor.snapshot(trace_limit=trace)


@app.get("/api/agent/trace")
def get_agent_trace(
    limit: int = Query(120, ge=0, le=240),
    run: int | None = None,
) -> dict[str, Any]:
    """The operational trace, whole or for one run. Never chain-of-thought."""
    agent_trace = runtime.supervisor.trace
    entries = agent_trace.entries_for_run(run) if run is not None else agent_trace.recent(limit)
    return {"trace": entries, "incidents": agent_trace.incidents()}


@app.post("/api/agent/investigate")
async def post_agent_investigate() -> dict[str, Any]:
    """"Investigate current plant": one supervisor run, now, on demand."""
    incident = await runtime.supervisor.investigate()
    return {"incident": incident.to_dict(), **_publish(runtime.payload())}


@app.post("/api/agent/config")
async def post_agent_config(request: AgentConfigRequest) -> dict[str, Any]:
    runtime.supervisor.set_enabled(request.enabled)
    return _publish(runtime.payload())


@app.get("/api/scenarios")
def get_scenarios() -> dict[str, Any]:
    return {
        "scenarios": runtime.session.scenario_catalogue(),
        "scenario": runtime.session.scenario,
    }


@app.post("/api/scenarios")
async def post_scenario(request: ScenarioRequest) -> dict[str, Any]:
    """Reset the plant into a predefined, repeatable situation."""
    try:
        state = runtime.load_scenario(request.scenario)
    except ValueError as reason:
        raise HTTPException(status_code=404, detail=str(reason)) from reason
    return _publish(runtime.payload([state], replace=True))

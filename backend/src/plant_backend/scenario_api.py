"""Describe, review and activate scenarios in the shared V2 digital twin."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import os
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from plant_sim import SENSOR_CHANNELS
from plant_sim.faults import CATALOGUE as FAULT_CATALOGUE
from pydantic import Field

from . import scenario_authoring as authoring
from . import site_weather

router = APIRouter(prefix="/api/scenario-studio", tags=["scenario-studio"])


class GenerateRequest(authoring.StrictModel):
    prompt: str = Field(min_length=1, max_length=authoring.PROMPT_LIMIT)


class PreviewRequest(authoring.StrictModel):
    scenario: authoring.ScenarioSpec


class ActivateRequest(authoring.StrictModel):
    preview_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    controller: Literal["MPC", "H"]


class ScenarioPreviews:
    """Keep at most 16 reviewed scenarios with their exact weather snapshots.

    Copies on both sides prevent edits or a forecast refresh from changing what
    an already-reviewed preview will activate. This store never mutates a run.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._previews: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def save(self, scenario: authoring.ScenarioSpec, forecast: dict[str, Any]) -> dict[str, Any]:
        identity = uuid4().hex
        start = datetime.fromisoformat(forecast["start_time"].replace("Z", "+00:00"))
        end = (start + timedelta(days=scenario.duration_days)).astimezone(timezone.utc)
        value = {
            "preview_id": identity,
            "scenario": scenario.model_dump(),
            "forecast": deepcopy(forecast),
            "summary": {
                "duration_hours": scenario.duration_days * 24,
                "start_time": forecast["start_time"],
                "end_time": end.isoformat().replace("+00:00", "Z"),
                "total_delivery_kg": sum(item.kg for item in scenario.deliveries),
                "delivery_count": len(scenario.deliveries),
            },
            "warnings": authoring.warnings(scenario),
        }
        with self._lock:
            self._previews[identity] = value
            while len(self._previews) > 16:
                self._previews.popitem(last=False)
        public = deepcopy(value)
        public["forecast"] = {key: public["forecast"][key] for key in ("id", "site", "start_time", "source")}
        return public

    def get(self, identity: str) -> dict[str, Any]:
        with self._lock:
            if identity not in self._previews:
                raise HTTPException(404, "This scenario preview has expired. Preview the scenario again before running it.")
            return deepcopy(self._previews[identity])


previews = ScenarioPreviews()


@router.get("/capabilities")
def capabilities() -> dict[str, Any]:
    return {
        "llm_available": bool(os.environ.get(authoring.API_KEY_ENV, "").strip()),
        "model": os.environ.get(authoring.MODEL_ENV) or authoring.DEFAULT_MODEL,
        "defaults": authoring.defaults(),
        "examples": authoring.examples(),
        "faults": [
            {"type": item.type, "label": item.label, "default_severity": item.default_severity}
            for item in FAULT_CATALOGUE.values()
        ],
        "sensor_channels": list(SENSOR_CHANNELS),
    }


@router.post("/generate")
async def generate(request: GenerateRequest) -> dict[str, Any]:
    try:
        scenario = await authoring.author(request.prompt)
    except authoring.AuthoringUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    except authoring.ScenarioValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"scenario": scenario.model_dump(), "warnings": authoring.warnings(scenario)}


@router.post("/preview")
def preview(request: PreviewRequest) -> dict[str, Any]:
    scenario = request.scenario
    if scenario.unsupported:
        raise HTTPException(
            422,
            "Resolve the unsupported requests before previewing: " + "; ".join(scenario.unsupported),
        )
    try:
        if scenario.site.source == "saved":
            forecast = site_weather.load_example(scenario.site.example_id)
        else:
            forecast = site_weather.fetch_forecast(
                latitude=scenario.site.latitude,
                longitude=scenario.site.longitude,
                name=scenario.site.name,
            )
    except site_weather.ForecastUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    except site_weather.SiteWeatherError as exc:
        raise HTTPException(422, str(exc)) from exc
    if len(forecast["hours"]) < scenario.duration_days * 24 + 24:
        raise HTTPException(422, "The weather snapshot does not cover the scenario and its final day of planning.")
    return previews.save(scenario, forecast)


@router.post("/activate")
def activate(request: ActivateRequest) -> dict[str, Any]:
    from . import v2_demo

    value = previews.get(request.preview_id)
    # Validate again at the state-changing boundary. Clients provide only the
    # preview id and controller, so reviewed data cannot be swapped on activation.
    scenario = authoring.build(value["scenario"])
    try:
        return v2_demo.session.activate_site(
            value["forecast"],
            controller_id=request.controller,
            solar_kw=scenario.solar_kw,
            battery_kwh=scenario.battery_kwh,
            inject_fault=False,
            solar_scale=1.0,
            scenario=scenario.model_dump(),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

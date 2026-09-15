"""Bounded site studies, independent of the live plant and research runs."""

from __future__ import annotations

import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from . import site_weather

router = APIRouter(prefix="/api/sites", tags=["site-studies"])
logger = logging.getLogger(__name__)


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    latitude: float = Field(ge=34, le=72, allow_inf_nan=False, strict=True)
    longitude: float = Field(ge=-25, le=45, allow_inf_nan=False, strict=True)
    name: str = Field(default="", max_length=100, strict=True)
    refresh: bool = Field(default=False, strict=True)


class SiteActivateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    forecast_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    solar_kw: float = Field(
        default=5000, ge=100, le=50000, allow_inf_nan=False, strict=True
    )
    battery_kwh: float = Field(
        default=8000, ge=100, le=100000, allow_inf_nan=False, strict=True
    )
    controller: Literal["MPC", "H"] = "MPC"
    inject_fault: bool = Field(default=True, strict=True)
    solar_scale: float = Field(
        default=1.0, ge=0.1, le=1, allow_inf_nan=False, strict=True
    )


class SiteRunRequest(SiteActivateRequest):
    compare_baseline: bool = Field(default=True, strict=True)


def _weather_error(exc: ValueError, *, missing: bool = False) -> HTTPException:
    unavailable = isinstance(exc, site_weather.ForecastUnavailableError)
    return HTTPException(
        status_code=404 if missing else 503 if unavailable else 422, detail=str(exc)
    )


@router.post("/forecast")
def forecast(request: ForecastRequest) -> dict[str, Any]:
    try:
        return site_weather.fetch_forecast(**request.model_dump())
    except site_weather.SiteWeatherError as exc:
        raise _weather_error(exc) from exc


@router.get("/examples")
def examples() -> list[dict[str, Any]]:
    return site_weather.list_examples()


@router.get("/examples/{example_id}")
def example(example_id: str) -> dict[str, Any]:
    try:
        return site_weather.load_example(example_id)
    except site_weather.SiteWeatherError as exc:
        raise _weather_error(exc, missing=True) from exc


@router.get("/forecasts/{forecast_id}/download")
def download_forecast(forecast_id: str) -> JSONResponse:
    try:
        value = site_weather.get_forecast(forecast_id)
    except site_weather.SiteWeatherError as exc:
        raise _weather_error(exc, missing=True) from exc
    return JSONResponse(
        value,
        headers={
            "Content-Disposition": f'attachment; filename="site-forecast-{value["id"][:12]}.json"',
        },
    )


def _compute(
    request: SiteRunRequest, forecast: dict[str, Any], progress: Any
) -> dict[str, Any]:
    # Keep optional optimisation imports out of the ordinary application's startup.
    from .site_simulation import run_site_simulation

    paired = request.compare_baseline and request.controller != "H"
    options: dict[str, Any] = {
        "solar_kw": request.solar_kw,
        "battery_kwh": request.battery_kwh,
        "inject_fault": request.inject_fault,
        "solar_scale": request.solar_scale,
    }
    result = run_site_simulation(
        forecast,
        **options,
        controller=request.controller,
        progress_callback=lambda fraction: progress(
            fraction * (0.8 if paired else 1.0)
        ),
    )
    result["baseline"] = (
        run_site_simulation(
            forecast,
            **options,
            controller="H",
            progress_callback=lambda fraction: progress(0.8 + 0.2 * fraction),
        )
        if paired
        else None
    )
    return result


class SiteJobs:
    """One computation at a time, at most two outstanding and eight retained."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="site-study"
        )
        self._jobs: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def submit(
        self, request: SiteRunRequest, forecast: dict[str, Any]
    ) -> dict[str, Any]:
        with self._lock:
            active = sum(
                j["status"] in ("queued", "running") for j in self._jobs.values()
            )
            if active >= 2:
                raise HTTPException(
                    429, "Two site studies are already pending. Wait for one to finish."
                )
            while len(self._jobs) >= 8:
                oldest = next(
                    k
                    for k, v in self._jobs.items()
                    if v["status"] in ("complete", "failed")
                )
                del self._jobs[oldest]
            identity = uuid4().hex
            value = {
                "id": identity,
                "status": "queued",
                "progress": 0.0,
                "result": None,
                "error": None,
            }
            self._jobs[identity] = value
            initial = deepcopy(value)
        try:
            self._executor.submit(self._run, identity, request, deepcopy(forecast))
        except RuntimeError:
            with self._lock:
                self._jobs[identity].update(
                    status="failed",
                    error="Site worker is shutting down. Restart the backend.",
                )
            return self.get(identity)
        return initial

    def _run(
        self, identity: str, request: SiteRunRequest, forecast: dict[str, Any]
    ) -> None:
        def progress(fraction: float) -> None:
            with self._lock:
                self._jobs[identity]["progress"] = max(
                    self._jobs[identity]["progress"], min(0.99, max(0, float(fraction)))
                )

        with self._lock:
            self._jobs[identity]["status"] = "running"
        try:
            result = _compute(request, forecast, progress)
            with self._lock:
                self._jobs[identity].update(
                    status="complete", result=result, progress=1.0
                )
        except Exception as exc:
            logger.exception("Site study %s failed", identity)
            # Internal solver/filesystem exception details remain in server logs.
            message = (
                str(exc)
                if isinstance(exc, (ValueError, ImportError))
                else f"The site simulation stopped ({type(exc).__name__}). Try the reactive baseline or another forecast."
            )
            with self._lock:
                self._jobs[identity].update(status="failed", error=message)

    def get(self, identity: str) -> dict[str, Any]:
        with self._lock:
            if identity not in self._jobs:
                raise HTTPException(
                    404, "This site study is no longer available. Run the study again."
                )
            return deepcopy(self._jobs[identity])

    def close(self) -> None:
        self._executor.shutdown(wait=True)


jobs = SiteJobs()


@router.post("/runs", status_code=202)
def start_run(request: SiteRunRequest) -> dict[str, Any]:
    try:
        data = site_weather.get_forecast(request.forecast_id)
    except site_weather.SiteWeatherError as exc:
        raise _weather_error(exc, missing=True) from exc
    return jobs.submit(request, data)


@router.post("/activate")
def activate_site(request: SiteActivateRequest) -> dict[str, Any]:
    """Use this location/weather as the real shared workspace plant."""
    from . import v2_demo

    try:
        data = site_weather.get_forecast(request.forecast_id)
    except site_weather.SiteWeatherError as exc:
        raise _weather_error(exc, missing=True) from exc
    options = request.model_dump(exclude={"forecast_id", "controller"})
    try:
        return v2_demo.session.activate_site(
            data, controller_id=request.controller, **options
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/deactivate")
def deactivate_site() -> dict[str, Any]:
    from . import v2_demo

    return v2_demo.session.reset_fixture()


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    return jobs.get(run_id)


@router.get("/runs/{run_id}/download")
def download_run(run_id: str) -> JSONResponse:
    job = jobs.get(run_id)
    if job["status"] != "complete":
        raise HTTPException(409, "The study has not completed yet.")
    return JSONResponse(
        job["result"],
        headers={
            "Content-Disposition": f'attachment; filename="site-study-{job["id"]}.json"',
        },
    )

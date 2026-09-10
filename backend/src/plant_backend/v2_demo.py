"""Isolated, bounded V2 fixture visibility; never a research execution API.

The only seed is a disposable fixture seed. Importing this module creates no
environment, controller, checkpoint, or background task. All operations share
one lock; FastAPI executes these synchronous routes in its worker threadpool.
"""

from __future__ import annotations

import importlib.util
import time
import uuid
from collections import deque
from copy import deepcopy
from threading import Lock
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from .demo_projection import boundary_events, forecast_view, weather_view
from .rl.v2 import beta_spaces
from .rl.v2.beta_env import PlantCyclicMarketEnvV2Beta
from .rl.v2.setpoint_controllers import LegacyHeuristicSetpointControllerV1

FIXTURE_SEED = 9_000_034
HISTORY_LIMIT = 673
ControllerId = Literal["H", "E-beta"]
router = APIRouter(prefix="/api/v2/demo", tags=["v2-demo"])


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    controller: ControllerId = "H"


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_run_id: str = Field(min_length=1, max_length=64, strict=True)
    expected_revision: int = Field(ge=0, strict=True)
    decisions: int = Field(default=1, ge=1, le=4, strict=True)


def _has_scipy() -> bool:
    return importlib.util.find_spec("scipy") is not None


def _controller(
    controller_id: ControllerId, environment: PlantCyclicMarketEnvV2Beta
) -> Any:
    if controller_id == "H":
        return LegacyHeuristicSetpointControllerV1()
    if not _has_scipy():
        raise HTTPException(
            status_code=503, detail="E-beta requires the optional optimisation runtime"
        )
    from .rl.v2.beta_mpc_controller import CyclicMarketEconomicMPCControllerV1

    return CyclicMarketEconomicMPCControllerV1(
        assumptions=environment.economic_assumptions, beta_task=environment.beta_task
    )


class V2DemoSession:
    """One fixture, independent of the historical live plant and experiment custody."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._environment: PlantCyclicMarketEnvV2Beta | None = None
        self._controller: Any = None
        self._controller_id: ControllerId = "H"
        self._run_id = ""
        self._revision = 0
        self._status = "ready"
        self._error: str | None = None
        self._decision: dict[str, Any] | None = None
        self._plan: dict[str, Any] | None = None
        self._history: deque[dict[str, Any]] = deque(maxlen=HISTORY_LIMIT)
        self._settlements: dict[float, dict[str, Any]] = {}
        self._committed: dict[str, Any] | None = None
        self._frames: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._archives: deque[dict[str, Any]] = deque(maxlen=3)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self._environment is None:
                self._reset("H")
            return self._view()

    def reset(self, controller_id: ControllerId) -> dict[str, Any]:
        with self._lock:
            self._reset(controller_id)
            return self._view()

    def _reset(self, controller_id: ControllerId) -> None:
        # Construct first so an unavailable optional controller preserves the old run.
        if controller_id == "E-beta" and not _has_scipy():
            raise HTTPException(
                status_code=503,
                detail="E-beta requires the optional optimisation runtime",
            )
        environment = PlantCyclicMarketEnvV2Beta()
        try:
            environment.reset(seed=FIXTURE_SEED)
            controller = _controller(controller_id, environment)
            controller.reset()
        except Exception:
            environment.close()
            raise
        if self._committed is not None:
            self._archives.append(self._recording())
        if self._environment is not None:
            self._environment.close()
        self._environment = environment
        self._controller = controller
        self._controller_id = controller_id
        self._run_id = str(uuid.uuid4())
        self._revision = 0
        self._status = "ready"
        self._error = None
        self._decision = self._plan = None
        self._frames.clear()
        self._events.clear()
        self._settlements.clear()
        self._history.clear()
        self._remember()
        self._commit()

    def step(self, request: StepRequest) -> dict[str, Any]:
        with self._lock:
            if self._environment is None or (
                request.expected_run_id != self._run_id
                or request.expected_revision != self._revision
            ):
                raise HTTPException(
                    status_code=409,
                    detail="Stale demo command; refresh the current run",
                )
            if self._status != "ready":
                raise HTTPException(
                    status_code=409, detail="Demo is stopped; reset before advancing"
                )
            for _ in range(request.decisions):
                try:
                    self._advance()
                    # Commit each successfully projected boundary, not just the
                    # end of the batch: later failures may leave physics partly
                    # advanced, or even prevent another observation being read.
                    self._commit()
                except Exception as exc:
                    # Raw exception text can contain solver internals. Publish a typed
                    # failure, never substitute a controller or expose private evidence.
                    self._status = "failed"
                    self._error = (
                        f"{self._controller_id} stopped ({type(exc).__name__}). "
                        "No fallback was applied. Last completed boundary shown; "
                        "a partial interval may have executed. Reset to begin another fixture run."
                    )
                    try:
                        self._environment.close()
                    except Exception:
                        pass  # Failed cleanup must not prevent the committed view.
                if self._status != "ready":
                    break
            self._revision += 1
            return self._view()

    def _advance(self) -> None:
        environment = self._env
        observation = environment.cyclic_market_controller_observation
        at_hours = environment.physics_steps_taken * environment.config.dt_h
        input_forecasts = forecast_view(observation, environment.config.solar.peak_kw)
        source = observation.plant if self._controller_id == "H" else observation
        started = time.perf_counter()
        decision = self._controller.compute_command(
            source,
            constraints=environment.constraints,
            config=environment.config,
            commitment_profile=environment.episode_config.electrolyser_commitment,
        )
        computation_ms = (time.perf_counter() - started) * 1000.0
        _, _, terminated, truncated, info = environment.step_process_setpoints(
            decision.command, controller_invalid=getattr(decision, "invalid", False)
        )
        execution = environment.last_process_setpoint_execution
        if execution is None:
            raise RuntimeError("missing common execution evidence")
        applications = execution.applications
        targets = decision.command.targets
        target_fields = (
            (
                "electrolyser",
                "Electrolyser",
                "electrolyser_target_kw",
                "applied_electrolyser_kw",
            ),
            ("dac", "Direct air capture", "dac_target_kw", "applied_dac_kw"),
            (
                "methanator",
                "Methanator",
                "methanator_target_kw",
                "applied_methanator_kw",
            ),
            (
                "battery",
                "Battery charge ceiling",
                "battery_charge_ceiling_kw",
                "applied_solar_to_battery_kw",
            ),
        )
        notes = list(getattr(decision.evidence, "source_notes", ()))
        if self._controller_id == "E-beta":
            notes.append(
                "Experimental economic MPC: current public mean forecasts and next-cycle resource returns."
            )
        notes.append(
            "Targets held for 15 minutes; dispatcher and safety guard run every minute."
        )
        notes.append(
            "Applied battery charging includes recovered spill and can exceed the direct-charge ceiling."
        )
        notes.extend(dict.fromkeys(info["adjustments"]))
        self._decision = {
            "at_hours": at_hours,
            "input_forecasts": input_forecasts,
            "input_state": observation.plant.state.boundary.to_dict(),
            "minutes": [
                {
                    "offset_minutes": item.ordinal,
                    "solar_kw": item.applied_solar_available_kw,
                    "electrolyser_kw": item.applied_electrolyser_kw,
                    "dac_kw": item.applied_dac_kw,
                    "methanator_kw": item.applied_methanator_kw,
                    "battery_charge_kw": item.applied_solar_to_battery_kw,
                    "battery_support_kw": item.applied_battery_support_kw,
                    "adjustments": list(item.adjustments),
                }
                for item in applications
            ],
            "controller_id": self._controller_id,
            "computation_ms": computation_ms,
            "notes": notes,
            "targets": [
                {
                    "id": identifier,
                    "label": label,
                    "requested_kw": float(getattr(targets, target)),
                    "applied_mean_kw": sum(
                        float(getattr(item, applied)) for item in applications
                    )
                    / len(applications),
                }
                for identifier, label, target, applied in target_fields
            ],
            "guard_adjustments": int(info["guard"]["interventions"]),
            "execution_digest": execution.sha256,
            "command_digest": decision.command.sha256,
        }
        self._plan = None
        if self._controller_id == "E-beta":
            self._plan = {
                "issued_at_hours": at_hours,
                "points": [
                    {
                        "offset_hours": item.index
                        * environment.episode_config.decision_interval_s
                        / 3600.0,
                        "electrolyser_kw": item.electrolyser_startup_kw
                        + item.electrolyser_productive_kw,
                        "dac_kw": item.solar_to_dac_kw,
                        "methanator_kw": item.solar_to_methanator_kw,
                        "battery_charge_kw": item.solar_to_battery_kw,
                        "battery_kwh": item.battery_energy_end_kwh,
                        "h2_kg": item.h2_end_kg,
                        "co2_kg": item.co2_end_kg,
                        "ch4_kg": item.ch4_end_kg,
                        "delivered_kg": item.delivered_kg,
                    }
                    for item in decision.evidence.plan.steps
                ],
            }
        delivery = info["delivery"]
        if delivery["commitments_settled"]:
            # The fixed public task has one recurring commitment every six hours,
            # aligned to decision boundaries. Do not inspect the private schedule.
            if delivery["commitments_settled"] != 1:
                raise RuntimeError("fixture delivery cadence changed")
            due = environment.physics_steps_taken * environment.config.dt_h
            self._settlements[due] = {
                "due_hours": due,
                "requested_kg": delivery["requested_kg"],
                "delivered_kg": delivery["delivered_kg"],
                "unmet_kg": delivery["unmet_kg"],
                "status": "met" if delivery["unmet_kg"] <= 1e-9 else "shortfall",
            }
        self._remember()
        if terminated or truncated:
            self._status = "complete"
            environment.close()

    @property
    def _env(self) -> PlantCyclicMarketEnvV2Beta:
        assert self._environment is not None
        return self._environment

    def _remember(self) -> None:
        environment = self._env
        state = environment.telemetry
        self._history.append(
            {
                "elapsed_hours": environment.physics_steps_taken
                * environment.config.dt_h,
                "solar_kw": state.solar_available_kw,
                "clear_sky_kw": state.solar_potential_kw,
                "h2_kg": state.h2_level_kg,
                "co2_kg": state.co2_level_kg,
                "battery_soc": state.battery_soc,
                "ch4_kg": state.ch4_level_kg,
                "delivered_kg": environment.economic_metrics.delivered_kg,
                "net_value_eur": environment.economic_metrics.reward_total_eur,
            }
        )

    def _frame(self) -> dict[str, Any]:
        environment = self._env
        observation = environment.cyclic_market_controller_observation
        state = environment.telemetry
        cycle = observation.cycle
        delivery = observation.plant.delivery
        elapsed = environment.physics_steps_taken * environment.config.dt_h
        episode = environment.episode_config
        economic = environment.economic_metrics
        forecast = observation.plant.forecast
        assert forecast is not None
        period = delivery.delivery_period_steps * environment.config.dt_h
        commitments = []
        # sample_episode's contract stream is deliberately fixed-period and fixed-
        # quantity. Project that PUBLIC configuration, never its private manifest.
        for index in range(1, int(round(episode.horizon_h / period)) + 1):
            due = index * period
            commitments.append(
                self._settlements.get(
                    due,
                    {
                        "due_hours": due,
                        "requested_kg": episode.delivery_amount_kg,
                        "delivered_kg": None,
                        "unmet_kg": None,
                        "status": "pending",
                    },
                )
            )
        resource_values = (
            (
                "battery",
                "Battery",
                "kWh",
                state.battery_energy_kwh,
                cycle.battery_margin_kwh,
            ),
            ("h2", "Hydrogen", "kg", state.h2_level_kg, cycle.h2_margin_kg),
            ("co2", "Carbon dioxide", "kg", state.co2_level_kg, cycle.co2_margin_kg),
            ("ch4", "Methane", "kg", state.ch4_level_kg, cycle.ch4_margin_kg),
        )
        return {
            "state": state.to_dict(),
            "weather": weather_view(state.to_dict(), elapsed),
            "commitment": (
                observation.plant.state.electrolyser_commitment.to_dict()
                if observation.plant.state.electrolyser_commitment
                else None
            ),
            "metrics": environment.metrics.to_dict(),
            "assumptions": environment.economic_assumptions.to_dict(),
            "commitment_profile": (
                episode.electrolyser_commitment.to_dict()
                if episode.electrolyser_commitment
                else None
            ),
            "clock": {
                "elapsed_hours": elapsed,
                "decision_index": environment.decisions_taken,
                "decision_count": environment.episode_decisions,
                "cycle_index": cycle.cycle_index,
                "cycle_count": int(
                    episode.horizon_h / environment.beta_task.cycle.cycle_h
                ),
                "hours_to_cycle_end": (cycle.cycle_decisions - cycle.decision_in_cycle)
                * episode.decision_interval_s
                / 3600.0,
            },
            "delivery": {
                "next_due_hours": (
                    None
                    if delivery.steps_to_next is None
                    else elapsed + delivery.steps_to_next * environment.config.dt_h
                ),
                "next_requested_kg": delivery.next_requested_kg,
                "delivered_kg": economic.delivered_kg,
                "unmet_kg": economic.unmet_kg,
                "requested_kg": delivery.cumulative_requested_kg,
                "commitments": commitments,
            },
            "forecasts": forecast_view(observation, environment.config.solar.peak_kw),
            "resources": [
                {
                    "id": identifier,
                    "label": label,
                    "unit": unit,
                    "current": current,
                    "required": current - margin,
                    "margin": margin,
                }
                for identifier, label, unit, current, margin in resource_values
            ],
            "economics": {
                "net_value_eur": economic.reward_total_eur,
                "entries": [
                    {
                        "id": key,
                        "label": key.removesuffix("_eur")
                        .replace("_", " ")
                        .capitalize(),
                        "value_eur": value,
                    }
                    for key, value in economic.components_eur.items()
                ],
                "assumption_note": "Illustrative simulator economics and wear assumptions; no calibrated commercial savings claim.",
            },
            "decision": self._decision,
            "plan": self._plan,
            "cycle_settlements": [
                item.to_dict() for item in environment.cycle_settlements
            ],
        }

    def _commit(self) -> None:
        candidate = self._snapshot()
        frame = candidate["frame"]
        new_events = boundary_events(self._frames[-1] if self._frames else None, frame)
        self._events.extend(new_events)
        self._frames.append(deepcopy(frame))
        candidate["events"] = deepcopy(self._events)
        self._committed = candidate

    def frame(self, run_id: str, index: int) -> dict[str, Any]:
        with self._lock:
            if run_id != self._run_id:
                raise HTTPException(
                    status_code=409,
                    detail="Run changed; refresh before inspecting history",
                )
            if index < 0 or index >= len(self._frames):
                raise HTTPException(
                    status_code=404, detail="No completed frame at this decision"
                )
            return deepcopy(self._frames[index])

    def _recording(self) -> dict[str, Any]:
        from .demo_recordings import seal_recording

        return seal_recording(self._view(), self._frames)

    def recording(self) -> dict[str, Any]:
        with self._lock:
            if self._committed is None:
                raise HTTPException(status_code=404, detail="No active run to export")
            return self._recording()

    def archives(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(list(self._archives))

    def _snapshot(self) -> dict[str, Any]:
        environment = self._env
        scipy_available = _has_scipy()
        config = environment.config
        return deepcopy(
            {
                "schema": "plant-v2-demo-v1",
                "run_id": self._run_id,
                "revision": self._revision,
                "mode": "interactive_fixture",
                "status": self._status,
                "error": self._error,
                "controller": {
                    "id": self._controller_id,
                    "label": (
                        "Reactive heuristic"
                        if self._controller_id == "H"
                        else "Economic MPC (experimental)"
                    ),
                    "uses_forecast": self._controller_id == "E-beta",
                },
                "available_controllers": [
                    {
                        "id": "H",
                        "label": "Reactive heuristic",
                        "available": True,
                        "reason": None,
                    },
                    {
                        "id": "E-beta",
                        "label": "Economic MPC (experimental)",
                        "available": scipy_available,
                        "reason": (
                            None
                            if scipy_available
                            else "Optional optimisation runtime is not installed"
                        ),
                    },
                ],
                "environment": {
                    "id": beta_spaces.ENVIRONMENT_ID,
                    "observation_count": beta_spaces.OBSERVATION_SIZE,
                    "action_count": beta_spaces.ACTION_SIZE,
                    "decision_minutes": environment.episode_config.decision_interval_s
                    / 60.0,
                    "physics_minutes": config.dt_s / 60.0,
                    "horizon_hours": environment.episode_config.horizon_h,
                    "fixture_seed": FIXTURE_SEED,
                    "interface_digest": beta_spaces.ENVIRONMENT_CONTRACT_DIGEST,
                    "task_digest": environment.checkpoint_contract_digest,
                },
                "config": {
                    **config.to_dict(),
                    "dt_h": config.dt_h,
                    "steps_per_day": config.steps_per_day,
                    "electrolyser_min_load_kw": config.electrolyser.rated_kw
                    * config.electrolyser.min_load_frac,
                },
                "frame": self._frame(),
                "history": list(self._history),
                "events": list(self._events),
            }
        )

    def _view(self) -> dict[str, Any]:
        """Render the last complete public boundary without touching the plant."""
        assert self._committed is not None
        view = deepcopy(self._committed)
        view.update(revision=self._revision, status=self._status, error=self._error)
        scipy_available = _has_scipy()
        view["available_controllers"][1].update(
            available=scipy_available,
            reason=(
                None
                if scipy_available
                else "Optional optimisation runtime is not installed"
            ),
        )
        return view


session = V2DemoSession()


def _reject_query(request: Request) -> None:
    if request.query_params:
        raise HTTPException(
            status_code=422, detail="The demo accepts no query parameters"
        )


@router.get("")
def get_snapshot(request: Request) -> dict[str, Any]:
    _reject_query(request)
    return session.snapshot()


@router.get("/frames/{run_id}/{index}")
def get_frame(request: Request, run_id: str, index: int) -> dict[str, Any]:
    _reject_query(request)
    return session.frame(run_id, index)


@router.get("/recording")
def get_recording(request: Request) -> dict[str, Any]:
    _reject_query(request)
    return session.recording()


@router.get("/recordings")
def get_recordings(request: Request) -> list[dict[str, Any]]:
    from .demo_recordings import catalog, summarize

    _reject_query(request)
    return catalog() + [summarize(item, "recent") for item in session.archives()]


@router.get("/recordings/{recording_id}")
def get_saved_recording(request: Request, recording_id: str) -> dict[str, Any]:
    from .demo_recordings import read_recording

    _reject_query(request)
    for item in session.archives():
        if item["snapshot"]["run_id"] == recording_id:
            return item
    return read_recording(recording_id)


@router.post("/reset")
def reset_demo(request: Request, body: ResetRequest) -> dict[str, Any]:
    _reject_query(request)
    return session.reset(body.controller)


@router.post("/step")
def step_demo(request: Request, body: StepRequest) -> dict[str, Any]:
    _reject_query(request)
    return session.step(body)


__all__ = ["router"]

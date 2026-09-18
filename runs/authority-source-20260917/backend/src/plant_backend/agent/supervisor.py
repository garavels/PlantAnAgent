"""The supervisor: one slow loop above the fast one.

    clock tick -> runtime.advance(speed)      # the controller, every 60 s
               -> supervisor.observe(frames)  # cheap: looks for a *situation*
                     -> maybe one run         # a model call, in the background

`observe` is called on every tick and is deliberately almost free: it evaluates
the triggers and returns. Since V10 it also keeps the predicted-risk monitor up to
date -- but on the monitor's own cadence (15 simulated minutes by default), not per
tick: HEPA answers questions about hours, and asking it every 60 s would be a waste
of a forecast. A risk that crosses its configured level, or that is rising towards
one, becomes one more trigger competing on priority with the plant's own alarms.

When a trigger does fire, the run happens in a background task, so the simulation
keeps stepping while the supervisor thinks -- and if the model is slow, hangs or
fails, the plant is still being controlled by the same heuristic controller as
before.

Gating, in order: the agent must be enabled, no run may already be in flight, a
minimum gap must have passed since the last run, and the trigger's key must not
be on cooldown or already attached to an open incident. That is what keeps one
degraded electrolyser from starting an investigation every tick.

Failure is a first-class path. Any exception from the engine -- no API key, a
network error, a timeout, a malformed response -- is recorded in the trace,
emitted as a plant event, and then handed to `RuleEngine`, which needs neither
network nor credentials. The worst case is a supervisor that reasons less well,
never a plant that stops being controlled.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from ..hepa import service as hepa_service
from .engines import Diagnosis, Engine, LLMEngine, RuleEngine
from .risk import RiskMonitor, RiskPolicy
from .tools import AgentToolbox
from .trace import AgentTrace, Incident
from .triggers import Trigger, TriggerWatch, manual

log = logging.getLogger(__name__)

#: Simulated seconds that must pass before the same situation is raised again.
COOLDOWN_S = 3600.0
#: Simulated seconds between any two runs, whatever the reason.
MIN_GAP_S = 600.0
#: Wall-clock seconds one run may take before it is abandoned.
RUN_TIMEOUT_S = 120.0
#: History the trigger watch looks back over, in frames.
WATCH_FRAMES = 180

#: Environment: an API key turns the model on, `PLANT_AGENT_MODEL` picks it,
#: `PLANT_AGENT_ENABLED=0` turns the supervisor off entirely.
API_KEY_ENV = "OPENAI_API_KEY"
MODEL_ENV = "PLANT_AGENT_MODEL"
ENABLED_ENV = "PLANT_AGENT_ENABLED"


def default_engine() -> Engine:
    """The LLM supervisor when it can run, the rule supervisor when it cannot."""
    if not os.environ.get(API_KEY_ENV):
        return RuleEngine()
    model = os.environ.get(MODEL_ENV)
    return LLMEngine(model) if model else LLMEngine()


class SupervisorAgent:
    """Watches the plant, investigates situations, retunes the envelope."""

    def __init__(
        self,
        runtime: Any,
        engine: Engine | None = None,
        *,
        cooldown_s: float = COOLDOWN_S,
        min_gap_s: float = MIN_GAP_S,
        timeout_s: float = RUN_TIMEOUT_S,
        enabled: bool | None = None,
        forecaster: hepa_service.Forecaster | None = None,
        risk_policy: RiskPolicy | None = None,
    ) -> None:
        self.runtime = runtime
        self.engine = engine or default_engine()
        self.trace = AgentTrace()
        self.watch = TriggerWatch()
        #: V10: "what may happen?", on its own cadence. Unavailable, and silent,
        #: until a HEPA checkpoint is configured.
        self.risk = RiskMonitor(runtime, forecaster, risk_policy, trace=self.trace)
        self.cooldown_s = cooldown_s
        self.min_gap_s = min_gap_s
        self.timeout_s = timeout_s
        self.enabled = _enabled_by_default() if enabled is None else enabled

        self.status = "monitoring" if self.enabled else "disabled"
        self.runs = 0
        self.last_run_time_s: float | None = None
        self.last_run_clock: str | None = None
        self.last_error: str | None = None
        self._cooldowns: dict[str, float] = {}
        self._task: asyncio.Task[Any] | None = None

    # --- the slow loop ----------------------------------------------------
    def observe(self, frames: list[dict[str, Any]]) -> Trigger | None:
        """Called every tick. Starts a background run if a trigger gets through."""
        trigger = self.notice(frames)
        if trigger is None:
            return None
        self._task = asyncio.ensure_future(self.investigate(trigger))
        self._task.add_done_callback(self._finished)
        return trigger

    def notice(self, frames: list[dict[str, Any]]) -> Trigger | None:
        """Evaluate the triggers and the gating. No model call, no side effects."""
        if not self.enabled or self.busy or not frames:
            return None
        trigger = self.watch.evaluate(
            self._window(),
            self.runtime.session.events(60),
            # V8: the driver itself is something worth waking up for
            self.runtime.control.health_alert(),
            # V10: and so is something that has not happened yet
            self._risk_trigger(),
        )
        if trigger is None:
            return None
        return trigger if self._allowed(trigger, float(frames[-1]["time_s"])) else None

    async def investigate(self, trigger: Trigger | None = None) -> Incident:
        """Run the supervisor once. Never raises: failure becomes a diagnosis."""
        observation = self.runtime.session.telemetry
        trigger = trigger or manual(observation.step)
        self.runs += 1
        self.status = "investigating"
        self.last_run_time_s = observation.time_s
        self.last_run_clock = observation.clock
        self._cooldowns[trigger.key] = observation.time_s

        self.trace.at(observation.time_s, observation.clock)
        run = self.trace.start_run()
        self.trace.record("trigger", f"{trigger.reason}: {trigger.detail}", **trigger.to_dict())
        incident = self.trace.open_incident(
            Incident(
                id=f"INC-{run:03d}",
                run=run,
                opened_clock=observation.clock,
                opened_time_s=observation.time_s,
                trigger=trigger.reason,
                trigger_detail=trigger.detail,
                key=trigger.key,
            )
        )
        toolbox = AgentToolbox(self.runtime, self.trace, self.risk)
        try:
            diagnosis = await self._decide(toolbox, trigger)
        finally:
            self._retime()
        self._close(incident, diagnosis, toolbox)
        return incident

    async def _decide(self, toolbox: AgentToolbox, trigger: Trigger) -> Diagnosis:
        """The engine, with the rule book behind it if the engine falls over."""
        try:
            return await asyncio.wait_for(
                self.engine.investigate(toolbox, trigger.to_dict()), self.timeout_s
            )
        except Exception as failure:  # noqa: BLE001 - the plant outranks the agent
            self._failed(failure)
            if isinstance(self.engine, RuleEngine):
                return Diagnosis(
                    category="unclear",
                    summary=(
                        "The supervisor could not complete its investigation "
                        f"({type(failure).__name__}). The controller continues to run the plant "
                        "inside the existing envelope; a human should take a look."
                    ),
                    human_intervention_required=True,
                )
            self.trace.record("decision", "Falling back to rule-based supervision")
            try:
                return await RuleEngine().investigate(toolbox, trigger.to_dict())
            except Exception as fallback_failure:  # noqa: BLE001
                self._failed(fallback_failure)
                return Diagnosis(
                    category="unclear",
                    summary=(
                        "Both the model and the rule-based supervisor failed to complete an "
                        "investigation. Nothing was changed; the plant is still under controller "
                        "and safety limits."
                    ),
                    human_intervention_required=True,
                )

    # --- bookkeeping ------------------------------------------------------
    def _close(self, incident: Incident, diagnosis: Diagnosis, toolbox: AgentToolbox) -> None:
        """Write the conclusion into the trace, the incident and the event log."""
        self._retime()
        incident.category = diagnosis.category
        incident.component = diagnosis.component
        incident.confidence = diagnosis.confidence
        incident.summary = diagnosis.summary
        incident.actions = list(toolbox.actions)
        incident.human_intervention_required = diagnosis.human_intervention_required

        label = diagnosis.category.replace("_", " ")
        self.trace.record(
            "diagnosis",
            f"{label} (confidence: {diagnosis.confidence})",
            **diagnosis.to_dict(),
            engine=self.engine.label,
        )
        status = self.runtime.session.telemetry.plant_status
        if not toolbox.actions:
            self.trace.record("result", f"No change made; plant {status}", plant_status=status)
        else:
            self.trace.record(
                "result",
                f"{len(toolbox.actions)} change(s) applied; plant {status}",
                plant_status=status,
                actions=list(toolbox.actions),
            )
        if diagnosis.category == "normal":
            # Nothing is wrong, so nothing is still open. The cooldowns stay:
            # a situation that looked normal once is not worth re-opening at the
            # next tick just because the trigger is still in the window.
            self.trace.resolve_all()
        else:
            # An incident is only *open* while it still says something about the
            # plant now: a later look at the same situation replaces the earlier
            # one, and a run that ends with the plant normal and nobody needed
            # is over as soon as it is written. Otherwise the history is all
            # open incidents and tells the shift nothing.
            for earlier in self.trace.unresolved:
                same_situation = (earlier.category, earlier.component) == (
                    incident.category,
                    incident.component,
                )
                if earlier is not incident and same_situation:
                    self.trace.resolve(earlier)
            if status == "normal" and not diagnosis.human_intervention_required:
                self.trace.resolve(incident)
        self._emit(
            "warning" if diagnosis.human_intervention_required else "info",
            f"agent.{diagnosis.category}",
            f"Supervisor: {diagnosis.summary}",
        )
        self.status = "attention" if self.needs_human else "monitoring"

    def _failed(self, failure: BaseException) -> None:
        self.last_error = f"{type(failure).__name__}: {failure}"
        log.warning("supervisor run failed: %s", self.last_error)
        self.trace.record("error", f"Supervisor run failed: {self.last_error}")
        self._emit("warning", "agent.error", f"Supervisory agent failed: {self.last_error}")

    def _emit(self, level: str, code: str, message: str) -> None:
        """Put the supervisor's own conclusion on the plant's event stream."""
        try:
            self.runtime.session.simulator.events.emit(level, code, message)
        except Exception:  # noqa: BLE001 - logging must never break a run
            log.debug("could not emit agent event %s", code, exc_info=True)

    def _risk_trigger(self) -> Trigger | None:
        """Refresh the forecast if it is due, and offer what it raises, if anything.

        Cheap by construction: the monitor's cadence decides whether the model is
        asked at all, and a monitor with no checkpoint answers `unavailable` without
        loading anything. Failure here is not the plant's problem -- the monitor
        turns every failure into a status -- so nothing is caught again.
        """
        self.risk.update()
        return self.risk.trigger()

    def _allowed(self, trigger: Trigger, time_s: float) -> bool:
        if self.last_run_time_s is not None and time_s - self.last_run_time_s < self.min_gap_s:
            return False
        seen = self._cooldowns.get(trigger.key)
        if seen is not None and time_s - seen < self.cooldown_s:
            return False
        return not any(incident.key == trigger.key for incident in self.trace.unresolved)

    def _window(self) -> list[dict[str, Any]]:
        return self.runtime.session.recent(WATCH_FRAMES)

    def _retime(self) -> None:
        observation = self.runtime.session.telemetry
        self.trace.at(observation.time_s, observation.clock)

    def _finished(self, task: asyncio.Task[Any]) -> None:
        self._task = None
        if not task.cancelled() and task.exception() is not None:
            self._failed(task.exception())  # type: ignore[arg-type]  # checked above

    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def needs_human(self) -> bool:
        """One source for both the status chip and the banner, so they agree."""
        return any(incident.human_intervention_required for incident in self.trace.unresolved)

    # --- control and reporting --------------------------------------------
    def set_enabled(self, enabled: bool) -> bool:
        self.enabled = bool(enabled)
        self.status = "monitoring" if self.enabled else "disabled"
        return self.enabled

    def reset(self) -> None:
        """Forget the run history. Called when the plant itself is reset."""
        self.trace.clear()
        self.watch.reset()
        self.risk.reset()
        self._cooldowns.clear()
        self.runs = 0
        self.last_run_time_s = None
        self.last_run_clock = None
        self.last_error = None
        self.status = "monitoring" if self.enabled else "disabled"

    def snapshot(self, trace_limit: int = 60) -> dict[str, Any]:
        """Everything the control room needs about the supervisor, in one dict."""
        incident = self.trace.last_incident
        return {
            "enabled": self.enabled,
            "status": self.status,
            "busy": self.busy,
            "engine": self.engine.label,
            "runs": self.runs,
            "last_run_clock": self.last_run_clock,
            "last_error": self.last_error,
            "cooldown_s": self.cooldown_s,
            "min_gap_s": self.min_gap_s,
            "human_intervention_required": self.needs_human,
            "incident": incident.to_dict() if incident else None,
            "incidents": self.trace.incidents(10),
            "trace": self.trace.recent(trace_limit),
            # V10: what may happen, next to what happened.
            "risk": self.risk.snapshot(),
        }


def _enabled_by_default() -> bool:
    return os.environ.get(ENABLED_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


__all__ = ["COOLDOWN_S", "MIN_GAP_S", "SupervisorAgent", "default_engine"]

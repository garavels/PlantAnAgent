"""The controller manager: who is driving, inside what envelope, and how well.

V7 gave the runtime two controllers and a registry. V8 gives that registry a
name and a memory, because a supervisor above it needs three things the registry
could not answer:

    who is driving now, and what else could      `status()` / `catalogue()`
    how well is the driver doing                 `health` / `metrics()`
    change the driver, and say why               `select(..., reason=...)`

The manager owns exactly that -- active controller, the shared envelope, the
switch log and the health windows -- and deliberately nothing else. It never
steps the plant, never validates an action and never holds a simulator
reference, so the chain of command stays:

    supervisor -> manager -> controller -> safety.validate -> simulator

Switching is *selection only*: the plant is not reset, not paused and not
touched, the envelope carries over unchanged, and the new controller starts with
a clean health window (it should be judged on its own driving, not its
predecessor's). Two brakes stop an anxious supervisor from oscillating: no
agent-initiated switch within `SWITCH_COOLDOWN_S` of the last one, and no return
to a controller that was demoted for poor behaviour within `RETURN_COOLDOWN_S`
of that demotion. Both are measured in *simulated* seconds, both are exempt for
an operator, and both are exempt for escalation into safe mode -- a brake that
can delay protecting the plant is a bug, not a brake.

This is not an ensemble: exactly one controller drives at a time.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from plant_sim import Action, PlantConfig, PlantState

from .base import Constraints, ControlDecision, Controller
from .health import ControllerHealth, StepRecord, assess
from .heuristic import HeuristicController
from .rl import RL_CONTROLLER_ID, RLControllerSlot
from .safe import SAFE_CONTROLLER_ID, SafeController

log = logging.getLogger(__name__)

#: Who asked for a change. An operator is trusted more than the supervisor.
Actor = Literal["operator", "agent", "system"]
#: How the plant is being run, as opposed to *who* is running it.
OperatingState = Literal["normal", "degraded", "safe"]
OPERATING_STATES: tuple[OperatingState, ...] = ("normal", "degraded", "safe")

#: Simulated seconds an agent must leave between two controller switches.
SWITCH_COOLDOWN_S = 1800.0
#: Simulated seconds before a controller demoted for poor behaviour may return.
RETURN_COOLDOWN_S = 7200.0
#: Switch entries kept for the control room.
SWITCH_LOG_LIMIT = 40


@dataclass(frozen=True)
class ControllerSwitch:
    """One handover, with the reason that justified it. Written, never edited."""

    time_s: float
    clock: str
    previous: str
    current: str
    reason: str
    by: Actor
    previous_grade: str = "unrated"

    @property
    def summary(self) -> str:
        return f"{self.previous.upper()} -> {self.current.upper()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s,
            "clock": self.clock,
            "previous": self.previous,
            "current": self.current,
            "summary": self.summary,
            "reason": self.reason,
            "by": self.by,
            "previous_grade": self.previous_grade,
        }


class ControllerManager:
    """Active controller, shared envelope, switch history and controller health."""

    def __init__(
        self,
        *,
        constraints: Constraints | None = None,
        now: Callable[[], tuple[float, str]] | None = None,
        switch_cooldown_s: float = SWITCH_COOLDOWN_S,
        return_cooldown_s: float = RETURN_COOLDOWN_S,
    ) -> None:
        self._now = now or (lambda: (0.0, "D1 00:00"))
        self.switch_cooldown_s = switch_cooldown_s
        self.return_cooldown_s = return_cooldown_s

        self.controllers: dict[str, Controller] = {}
        for controller in (HeuristicController(), SafeController()):
            self.controllers[controller.id] = controller
        #: The RL checkpoint slot. RL joins the registry only once it loads.
        self.rl = RLControllerSlot()
        self.rl.autoload()
        if self.rl.controller is not None:
            self.controllers[RL_CONTROLLER_ID] = self.rl.controller

        self.active_id = HeuristicController.id
        #: Who drives again when safe mode is left. Never the safe controller.
        self.preferred_id = HeuristicController.id
        self.operating_state: OperatingState = "normal"
        self.constraints = constraints or Constraints()
        #: Envelope fields the supervisor overwrote, and what they were before.
        self._agent_envelope: dict[str, Any] = {}

        self.health: dict[str, ControllerHealth] = {
            controller_id: ControllerHealth(controller_id) for controller_id in self.controllers
        }
        self.switches: list[ControllerSwitch] = []
        #: controller id -> simulated time it was taken off the plant for behaviour.
        self._demoted: dict[str, float] = {}
        #: Poor-health alerts already announced, so one bad spell alerts once.
        self._alerted: set[str] = set()

    # --- who is driving ---------------------------------------------------
    @property
    def active(self) -> Controller:
        return self.controllers[self.active_id]

    @property
    def active_health(self) -> ControllerHealth:
        return self.health.setdefault(self.active_id, ControllerHealth(self.active_id))

    @property
    def last_switch(self) -> ControllerSwitch | None:
        return self.switches[-1] if self.switches else None

    def available(self, controller_id: str) -> bool:
        return controller_id in self.controllers

    def decide(
        self, observation: PlantState, *, config: PlantConfig
    ) -> ControlDecision:
        """Ask the active controller for this timestep. No validation here."""
        return self.active.compute_action(
            observation, constraints=self.constraints, config=config
        )

    # --- switching --------------------------------------------------------
    def select(
        self,
        controller_id: str,
        *,
        reason: str = "operator selection",
        by: Actor = "operator",
        force: bool = False,
    ) -> ControllerSwitch | None:
        """Hand the plant to `controller_id`. Returns the switch, or `None` if unchanged.

        Raises `ValueError` for an unknown or unavailable controller and for a
        switch a brake refuses; the caller (a route, or a supervisor tool) turns
        that into an error the asker can read.
        """
        if controller_id not in self.controllers:
            raise ValueError(self.unavailable(controller_id))
        if controller_id == self.active_id:
            self._remember_preference(controller_id)
            return None

        escalation = controller_id == SAFE_CONTROLLER_ID
        if by == "agent" and not force and not escalation:
            self._check_brakes(controller_id)

        previous = self.active_id
        grade = self.active_health.grade
        self.active_id = controller_id
        self._remember_preference(controller_id)
        # a handover is a fresh assessment: judge the new driver on its own driving
        self.health[controller_id] = ControllerHealth(controller_id)
        self._alerted.discard(controller_id)
        if by == "agent" and grade in {"poor", "suspect"} and previous != SAFE_CONTROLLER_ID:
            self._demoted[previous] = self._time_s
        self.operating_state = "safe" if escalation else self._state_after_switch()

        time_s, clock = self._now()
        switch = ControllerSwitch(
            time_s=time_s,
            clock=clock,
            previous=previous,
            current=controller_id,
            reason=reason.strip() or "no reason given",
            by=by,
            previous_grade=grade,
        )
        self.switches.append(switch)
        del self.switches[:-SWITCH_LOG_LIMIT]
        log.info(
            "controller switch %s -> %s by %s at %s: %s",
            previous,
            controller_id,
            by,
            clock,
            switch.reason,
        )
        return switch

    def set_operating_state(
        self, state: str, *, reason: str = "", by: Actor = "operator"
    ) -> OperatingState:
        """Move between normal, degraded and safe running.

        `safe` puts the safe controller on the plant; leaving it hands back to
        whoever was driving before. `degraded` is a label for "running on a
        deliberately tightened envelope" -- it changes no controller by itself.
        """
        if state not in OPERATING_STATES:
            raise ValueError(f"unknown operating state: {state} (expected {', '.join(OPERATING_STATES)})")
        if state == "safe":
            self.select(
                SAFE_CONTROLLER_ID,
                reason=reason or "entering safe mode",
                by=by,
            )
            return self.operating_state
        if self.active_id == SAFE_CONTROLLER_ID:
            self.select(
                self.preferred_id,
                reason=reason or f"leaving safe mode: returning to {state} operation",
                by=by,
                force=True,
            )
        self.operating_state = state  # type: ignore[assignment]  # guarded above
        return self.operating_state

    def _check_brakes(self, controller_id: str) -> None:
        """Hysteresis: refuse a switch that would be oscillation, and say so."""
        now = self._time_s
        last = self.last_switch
        if last is not None and now - last.time_s < self.switch_cooldown_s:
            wait = self.switch_cooldown_s - (now - last.time_s)
            raise ValueError(
                f"controller switch on cooldown: {last.summary} was {int((now - last.time_s) / 60)} min "
                f"ago, {int(wait / 60)} min still to run"
            )
        demoted_at = self._demoted.get(controller_id)
        if demoted_at is not None and now - demoted_at < self.return_cooldown_s:
            wait = self.return_cooldown_s - (now - demoted_at)
            raise ValueError(
                f"{controller_id} was taken off the plant for its behaviour "
                f"{int((now - demoted_at) / 60)} min ago; it may return in {int(wait / 60)} min"
            )

    def _remember_preference(self, controller_id: str) -> None:
        if controller_id != SAFE_CONTROLLER_ID:
            self.preferred_id = controller_id
            self._demoted.pop(controller_id, None)

    def _state_after_switch(self) -> OperatingState:
        """Leaving safe mode by picking a controller directly still leaves it."""
        return "normal" if self.operating_state == "safe" else self.operating_state

    # --- the envelope -----------------------------------------------------
    def configure_constraints(
        self, changes: dict[str, Any], *, by: Actor = "operator"
    ) -> Constraints:
        """Tighten or relax the shared envelope. Applies to every controller.

        One envelope, held here, passed to whoever is driving -- so a supervisor
        that lowers the electrolyser cap has lowered it for the RL policy and the
        heuristic alike, with no retraining and no per-controller bookkeeping.

        Who asked matters for one reason: the operator's envelope is
        configuration and survives a reset, while a supervisor's retune belongs
        to the incident that caused it.
        """
        before = self.constraints.to_dict()
        self.constraints = self.constraints.merged(changes)
        after = self.constraints.to_dict()
        for field in changes:
            if field not in after or after[field] == before[field]:
                continue
            if by == "agent":
                self._agent_envelope.setdefault(field, before[field])
            else:
                self._agent_envelope.pop(field, None)
        return self.constraints

    @property
    def agent_envelope(self) -> dict[str, Any]:
        """Fields the supervisor currently owns, and what they were before it did."""
        return dict(self._agent_envelope)

    def revert_agent_constraints(self) -> Constraints:
        """Put back whatever the supervisor overwrote. Called on reset."""
        if self._agent_envelope:
            self.constraints = self.constraints.merged(dict(self._agent_envelope))
            self._agent_envelope.clear()
        return self.constraints

    # --- health -----------------------------------------------------------
    def record_step(
        self,
        *,
        before: PlantState,
        after: PlantState,
        requested: Action,
        adjustments: tuple[str, ...] | list[str],
        config: PlantConfig,
        invalid_action: bool = False,
        critical_events: int = 0,
    ) -> StepRecord:
        """Score the timestep that just ran against the controller that asked for it."""
        record = assess(
            before=before,
            after=after,
            requested=requested,
            adjustments=adjustments,
            invalid_action=invalid_action,
            critical_events=critical_events,
            reserve_soc=self.constraints.reserve_soc(config),
        )
        self.active_health.record(record)
        return record

    def health_alert(self) -> dict[str, Any] | None:
        """The first look at a controller whose health has just gone poor.

        Returned once per spell, so the supervisor's trigger watch can wake on it
        without being woken by it every timestep afterwards.
        """
        health = self.active_health
        if health.grade != "poor":
            if health.grade == "good":
                self._alerted.discard(self.active_id)
            return None
        if self.active_id in self._alerted:
            return None
        self._alerted.add(self.active_id)
        return health.to_dict()

    # --- reporting --------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Everything about control that is not an action. The stable interface."""
        last = self.last_switch
        return {
            "active": self.active.to_dict(),
            "active_id": self.active_id,
            "preferred_id": self.preferred_id,
            "operating_state": self.operating_state,
            "controllers": self.catalogue(),
            "health": self.active_health.to_dict(),
            "last_switch": last.to_dict() if last else None,
            "switches": [switch.to_dict() for switch in self.switches[-10:]],
            "switch_count": len(self.switches),
            "cooldown_s": self.switch_cooldown_s,
            "agent_envelope": sorted(self._agent_envelope),
        }

    def metrics(self) -> dict[str, Any]:
        """Health for every controller that has driven, plus the active one's."""
        return {
            "active": self.active_id,
            "controllers": {
                controller_id: health.to_dict()
                for controller_id, health in self.health.items()
                if health.steps or controller_id == self.active_id
            },
        }

    def catalogue(self) -> list[dict[str, Any]]:
        """Every controller the control room may offer, available or not.

        RL appears even when it cannot be selected, carrying the reason: a
        disabled option with an explanation is honest, a missing option is not.
        """
        catalogue = [
            {
                **controller.to_dict(),
                "available": True,
                "state": "ready",
                "reason": "",
                "grade": self.health[controller_id].grade if controller_id in self.health else "unrated",
            }
            for controller_id, controller in self.controllers.items()
            if controller_id != RL_CONTROLLER_ID
        ]
        rl = self.rl.descriptor()
        rl["grade"] = self.health[RL_CONTROLLER_ID].grade if RL_CONTROLLER_ID in self.health else "unrated"
        catalogue.append(rl)
        return catalogue

    # --- the RL slot ------------------------------------------------------
    def load_rl(self, path: str) -> dict[str, Any]:
        """Load a checkpoint and make RL selectable. Loading never switches to it."""
        try:
            controller = self.rl.load(path)
        except ValueError:
            # a bad checkpoint must not leave the registry and the slot disagreeing
            # about whether RL can drive: whoever was driving carries on
            self._reconcile_rl()
            raise
        self.controllers[RL_CONTROLLER_ID] = controller
        self.health.setdefault(RL_CONTROLLER_ID, ControllerHealth(RL_CONTROLLER_ID)).clear()
        return self.rl.status.to_dict()

    def unload_rl(self) -> dict[str, Any]:
        """Drop the policy; if it was driving, the heuristic takes over immediately."""
        status = self.rl.unload().to_dict()
        self._reconcile_rl(reason="RL checkpoint unloaded")
        return status

    def rl_failed(self, reason: str) -> ControllerSwitch | None:
        """A policy that cannot answer stops driving. The fallback path.

        Recorded on the slot, out of the registry (so it cannot be reselected
        until a checkpoint is loaded again), and the plant carries on under the
        controller that always can.
        """
        self.rl.mark_failed(reason)
        return self._reconcile_rl(reason=f"RL inference failed: {reason}")

    def _reconcile_rl(self, *, reason: str = "RL unavailable") -> ControllerSwitch | None:
        """Make the registry agree with the slot, and never strand RL as active."""
        if self.rl.controller is not None:
            self.controllers[RL_CONTROLLER_ID] = self.rl.controller
            return None
        self.controllers.pop(RL_CONTROLLER_ID, None)
        if self.active_id != RL_CONTROLLER_ID:
            return None
        fallback = (
            self.preferred_id
            if self.preferred_id not in {RL_CONTROLLER_ID, self.active_id}
            else HeuristicController.id
        )
        return self.select(fallback, reason=reason, by="system", force=True)

    # --- lifecycle --------------------------------------------------------
    def reset(self) -> None:
        """Called when the plant is reset: the run is over, the setup survives.

        Who is driving and the operator's envelope are configuration and stay.
        The health windows, the switch log and the supervisor's retunes describe
        a run that no longer exists, so they go -- as does safe mode, which was
        a response to a condition that no longer exists either.
        """
        for controller in self.controllers.values():
            controller.reset()
        for health in self.health.values():
            health.clear()
        self.switches.clear()
        self._demoted.clear()
        self._alerted.clear()
        self.revert_agent_constraints()
        if self.active_id == SAFE_CONTROLLER_ID:
            self.active_id = self.preferred_id
        self.operating_state = "normal"

    def unavailable(self, controller_id: str) -> str:
        if controller_id == RL_CONTROLLER_ID:
            # selectable only once a checkpoint is loaded; say why it is not
            return f"RL controller unavailable: {self.rl.status.reason}"
        known = ", ".join(sorted(self.controllers))
        return f"unknown controller: {controller_id} (available: {known})"

    @property
    def _time_s(self) -> float:
        return self._now()[0]


__all__ = [
    "OPERATING_STATES",
    "RETURN_COOLDOWN_S",
    "SWITCH_COOLDOWN_S",
    "Actor",
    "ControllerManager",
    "ControllerSwitch",
    "OperatingState",
]

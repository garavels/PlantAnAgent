"""The supervisor's hands: the only way it can touch the plant.

Read tools answer from telemetry (`observations`), write tools go through the
same two levers an operator has -- the operating envelope and the control mode --
and therefore through the same validation guard:

    toolbox.set_controller_constraints(electrolyser_max_kw=1800)
        -> PlantRuntime.configure_constraints   (Constraints.merged validates)
        -> HeuristicController                  (plans inside the envelope)
        -> safety.validate                      (clamps whatever still escapes)
        -> PlantSimulator.step

There is deliberately no tool that sets an `Action`, and none that reaches the
simulator, the fault registry or a component object. A supervisor that could
allocate kilowatts would end up doing the controller's job badly; a supervisor
that could clear its own faults would be marking its own homework.

Every call is recorded in the trace, writes are capped per run, and a failed
write returns `{"ok": False, "error": ...}` rather than raising: the model is
expected to read the answer and carry on, not to crash the plant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import observations
from .trace import AgentTrace

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..orchestration.runtime import PlantRuntime

#: Frames of history a `get_recent_telemetry` call may look back over.
MAX_HISTORY_FRAMES = 720
#: Default look-back for history and diagnostics, in simulated minutes.
DEFAULT_WINDOW_MIN = 60
#: Events a single call may fetch.
MAX_EVENTS = 60
#: Writes one run is allowed. Investigation is unlimited; intervention is not.
MAX_WRITES_PER_RUN = 6

#: The envelope fields a supervisor may retune, with their meaning for the model.
CONSTRAINT_FIELDS = {
    "electrolyser_max_kw": "Cap on electrolyser power. Lower it when the stack cannot use it.",
    "dac_max_kw": "Cap on DAC power. Lower it to protect the energy budget.",
    "methanator_max_kw": "Cap on methanation power, i.e. how much CH4 we ask for.",
    "battery_support_max_kw": "Cap on battery discharge lent to the electrolyser.",
    "battery_min_soc": "Battery reserve nothing may eat into, 0..0.95. Raise it to hold energy back.",
}


class AgentToolbox:
    """The tool surface for one supervisor run, bound to the live runtime."""

    def __init__(self, runtime: "PlantRuntime", trace: AgentTrace) -> None:
        self._runtime = runtime
        self._trace = trace
        #: One line per applied change, for the incident record.
        self.actions: list[str] = []
        self.writes = 0

    # --- observation ------------------------------------------------------
    def get_current_telemetry(self) -> dict[str, Any]:
        """The latest reported plant state."""
        reading = observations.telemetry_view(self._runtime.session.telemetry)
        self._log("get_current_telemetry", plant_status=reading["plant_status"])
        return reading

    def get_recent_telemetry(self, minutes: int = DEFAULT_WINDOW_MIN) -> dict[str, Any]:
        """A thinned trend of the last `minutes` of telemetry, with means."""
        frames = self._frames(minutes)
        self._log("get_recent_telemetry", minutes=minutes, frames=len(frames))
        return observations.history_view(frames)

    def get_recent_events(self, limit: int = 20) -> dict[str, Any]:
        """What the plant itself flagged, oldest first."""
        limit = max(1, min(int(limit), MAX_EVENTS))
        events = self._runtime.session.events(limit)
        self._log("get_recent_events", limit=limit, returned=len(events))
        return {"events": events}

    def inspect_component(self, component_id: str) -> dict[str, Any]:
        """Observed performance of one unit against its design figures."""
        try:
            reading = observations.inspect(
                component_id,
                self._runtime.session.telemetry,
                self._frames(DEFAULT_WINDOW_MIN),
                self._runtime.config,
            )
        except ValueError as reason:
            self._log("inspect_component", component=component_id, error=str(reason))
            return {"ok": False, "error": str(reason)}
        self._log("inspect_component", component=component_id)
        return reading

    def run_diagnostic(self, component_id: str = "plant") -> dict[str, Any]:
        """A deeper look: mass balances, trends, and what they suggest.

        This is the tool that can tell instrument problems from process problems,
        because it is the only one that cross-checks two independent readings of
        the same thing (a tank level and the flows through that tank).
        """
        frames = self._frames(2 * DEFAULT_WINDOW_MIN)
        balance = observations.consistency(frames, self._runtime.config)
        report: dict[str, Any] = {
            "component": component_id,
            "mass_balance": balance,
            "trend": observations.history_view(frames, points=6),
            "notes": _diagnostic_notes(balance),
        }
        if component_id != "plant":
            component = self.inspect_component(component_id)
            if component.get("ok") is False:
                return component
            report["reading"] = component
        self._log("run_diagnostic", component=component_id, inconsistent=len(balance["inconsistent"]))
        return report

    def get_controller_status(self) -> dict[str, Any]:
        """Mode, controller, the last applied action and why it looked like that."""
        status = self._runtime.control_status()
        self._log("get_controller_status", mode=status["mode"], playing=status["playing"])
        return {
            "mode": status["mode"],
            "controller": status["controller"],
            "playing": status["playing"],
            "speed": status["speed"],
            "applied_action": status["applied_action"],
            "notes": status["notes"],
            "adjustments": status["adjustments"],
        }

    def get_controller_constraints(self) -> dict[str, Any]:
        """The operating envelope in force, with the ratings it sits inside."""
        config = self._runtime.config
        constraints = self._runtime.constraints
        self._log("get_controller_constraints")
        return {
            "constraints": constraints.to_dict(),
            "effective_kw": {
                "electrolyser": round(constraints.electrolyser_limit_kw(config), 1),
                "dac": round(constraints.dac_limit_kw(config), 1),
                "methanator": round(constraints.methanator_limit_kw(config), 1),
                "battery_support": round(constraints.support_limit_kw(config), 1),
            },
            "rated_kw": {
                "electrolyser": config.electrolyser.rated_kw,
                "dac": config.dac.rated_kw,
                "methanator": config.methanator.rated_kw,
                "battery_support": config.battery.max_discharge_kw,
            },
            "battery_reserve_soc": round(constraints.reserve_soc(config), 3),
            "fields": CONSTRAINT_FIELDS,
        }

    def observe(self, message: str, **detail: Any) -> None:
        """Write one observation into the trace. Touches nothing but the log.

        The numbers a diagnosis rests on belong in the shift log next to it, so
        the control room can see *why* the supervisor concluded what it did
        without being shown how it thought.
        """
        self._trace.record("observation", message, **detail)

    # --- intervention -----------------------------------------------------
    def set_controller_constraints(self, changes: dict[str, float | None], reason: str) -> dict[str, Any]:
        """Retune the envelope the controller plans inside. The main lever."""
        if not isinstance(changes, dict) or not changes:
            return self._refuse("set_controller_constraints", "no constraint changes given")
        blocked = self._budget("set_controller_constraints")
        if blocked:
            return blocked
        before = self._runtime.constraints.to_dict()
        try:
            after = self._runtime.configure_constraints(dict(changes)).to_dict()
        except ValueError as reason_text:
            return self._refuse("set_controller_constraints", str(reason_text))
        applied = {key: after[key] for key in changes if key in after and after[key] != before[key]}
        for key, value in applied.items():
            self._act(f"{key}: {_kw(before[key])} -> {_kw(value)}", reason=reason, field=key)
        if not applied:
            self._log("set_controller_constraints", note="no change", reason=reason)
        return {"ok": True, "before": before, "after": after, "applied": applied}

    def set_operating_mode(self, mode: str, reason: str) -> dict[str, Any]:
        """Switch between AUTO (the controller drives) and MANUAL (the operator does)."""
        blocked = self._budget("set_operating_mode")
        if blocked:
            return blocked
        before = self._runtime.mode
        try:
            after = self._runtime.set_mode(mode)
        except ValueError as reason_text:
            return self._refuse("set_operating_mode", str(reason_text))
        if after != before:
            self._act(f"operating mode: {before} -> {after}", reason=reason)
        return {"ok": True, "mode": after, "previous": before}

    def pause_plant(self, reason: str) -> dict[str, Any]:
        """Stop simulated time. Nothing is lost; nothing advances either."""
        blocked = self._budget("pause_plant")
        if blocked:
            return blocked
        was_playing = self._runtime.playing
        self._runtime.pause()
        if was_playing:
            self._act("plant paused", reason=reason)
        return {"ok": True, "playing": self._runtime.playing, "was_playing": was_playing}

    def resume_plant(self, reason: str) -> dict[str, Any]:
        blocked = self._budget("resume_plant")
        if blocked:
            return blocked
        was_playing = self._runtime.playing
        self._runtime.play()
        if not was_playing:
            self._act("plant resumed", reason=reason)
        return {"ok": True, "playing": self._runtime.playing, "was_playing": was_playing}

    def request_safe_shutdown(self, reason: str, keep_running: bool = False) -> dict[str, Any]:
        """Take the process loads to zero and hand the plant back to a human.

        High impact, so it is one explicit tool with a mandatory reason: the caps
        go to zero (the controller then asks for nothing), the battery reserve is
        raised, and unless `keep_running` is set, time stops as well.
        """
        if not reason or not reason.strip():
            return self._refuse("request_safe_shutdown", "a reason is required")
        blocked = self._budget("request_safe_shutdown")
        if blocked:
            return blocked
        before = self._runtime.constraints.to_dict()
        after = self._runtime.configure_constraints(
            {
                "electrolyser_max_kw": 0.0,
                "dac_max_kw": 0.0,
                "methanator_max_kw": 0.0,
                "battery_support_max_kw": 0.0,
                "battery_min_soc": max(before["battery_min_soc"], 0.4),
            }
        ).to_dict()
        self._act("safe shutdown: all process loads capped at 0 kW", reason=reason)
        if not keep_running:
            self._runtime.pause()
            self._act("plant paused for safe shutdown", reason=reason)
        return {"ok": True, "before": before, "after": after, "playing": self._runtime.playing}

    # --- plumbing ---------------------------------------------------------
    def _frames(self, minutes: int) -> list[dict[str, Any]]:
        steps = int(max(1.0, minutes) * 60.0 / self._runtime.config.dt_s)
        return self._runtime.session.recent(min(steps, MAX_HISTORY_FRAMES))

    def _budget(self, tool: str) -> dict[str, Any] | None:
        """Refuse a write once this run has already changed enough."""
        if self.writes >= MAX_WRITES_PER_RUN:
            return self._refuse(tool, f"write budget exhausted ({MAX_WRITES_PER_RUN} per run)")
        self.writes += 1
        return None

    def _refuse(self, tool: str, error: str) -> dict[str, Any]:
        self._trace.record("error", f"{tool} refused: {error}", tool=tool)
        return {"ok": False, "error": error}

    def _log(self, tool: str, **detail: Any) -> None:
        self._trace.record("tool", f"{tool}({_arguments(detail)})", tool=tool, **detail)

    def _act(self, message: str, *, reason: str, **detail: Any) -> None:
        self.actions.append(message)
        self._trace.record("action", message, reason=reason, **detail)


def _diagnostic_notes(balance: dict[str, Any]) -> list[str]:
    """Plain-language readings of the mass balance, for whoever gets the payload."""
    notes: list[str] = []
    for finding in balance["inconsistent"]:
        notes.append(
            f"{finding['channel']} moved {finding['reported_change_kg']} kg at {finding['clock']} "
            f"while the flows account for {finding['flow_change_kg']} kg: "
            f"the level reading and the flow readings disagree"
        )
    if not notes:
        notes.append("Inventories and flows agree: the instruments are consistent with each other")
    return notes


def _arguments(detail: dict[str, Any]) -> str:
    interesting = {k: v for k, v in detail.items() if k in {"component", "minutes", "limit"}}
    return ", ".join(f"{key}={value}" for key, value in interesting.items())


def _kw(value: Any) -> str:
    return "unset" if value is None else f"{float(value):g}"


__all__ = ["AgentToolbox", "CONSTRAINT_FIELDS", "MAX_WRITES_PER_RUN"]

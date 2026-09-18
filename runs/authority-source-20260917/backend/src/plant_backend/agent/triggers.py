"""When the supervisor is worth waking up.

The controller answers every 60 simulated seconds. The supervisor must not: an
LLM call per timestep would be slow, expensive and pointless, because nothing
interesting happens in most timesteps. So the loop watches for *situations* and
only then asks for an opinion:

    critical event        the plant shouted
    plant CRITICAL        the derived status says so
    controller health     the *driver* has started making mistakes
    warning event         the plant grumbled
    plant DEGRADED        something is derated or starved
    predicted risk        HEPA says something is becoming likely (V10)
    production drop       methane output collapsed while the sun was up
    manual               a human pressed "investigate"

Predicted risk is the V10 addition, and it is the first trigger that is not about
the present at all: every other one fires because something already happened, this
one fires because something probably will. It sits *below* the plant's own alarms
on purpose -- a forecast never outranks a critical event -- and the risk itself is
never a decision: it buys an investigation, and the supervisor decides the rest
(see `risk.RiskMonitor.trigger`).

Controller health is the V8 addition, and it is a different kind of trigger from
the rest: every other one says something about the plant, this one says something
about whoever is driving it. It fires once per bad spell (the manager only offers
the alert on the edge, see `controllers.ControllerManager.health_alert`), because
a controller that is driving badly now will still be driving badly next tick.

Each trigger carries a `key`, and the key is what stops one fault from starting
thirty investigations: while an incident with the same key is open, and for a
cooldown afterwards, the same situation is not raised again (see
`supervisor.SupervisorAgent`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Fraction of the earlier baseline that counts as a production collapse.
DROP_RATIO = 0.5
#: Frames of "now" and "before" the production check compares.
RECENT_FRAMES = 30
BASELINE_FRAMES = 120
#: Below this the baseline is too small for a ratio to mean anything (kg/h).
MIN_BASELINE_KG_H = 1.0
#: The supervisor's own events never wake the supervisor.
SELF_PREFIX = "agent."


@dataclass(frozen=True)
class Trigger:
    """One reason to run the supervisor, and how urgent it is."""

    reason: str
    detail: str
    key: str
    priority: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "detail": self.detail,
            "key": self.key,
            "priority": self.priority,
        }


def manual(step: int) -> Trigger:
    """A human asked. Always runs: no debounce, no dedupe."""
    return Trigger(
        reason="manual",
        detail="An operator asked the supervisor to investigate the current plant",
        key=f"manual:{step}",
        priority=100,
    )


class TriggerWatch:
    """Turns the stream of frames and events into at most one trigger per tick."""

    def __init__(self) -> None:
        self._last_event_seq = 0
        self._last_status = "normal"

    def reset(self) -> None:
        self._last_event_seq = 0
        self._last_status = "normal"

    def evaluate(
        self,
        frames: list[dict[str, Any]],
        events: list[dict[str, Any]],
        health_alert: dict[str, Any] | None = None,
        risk: Trigger | None = None,
    ) -> Trigger | None:
        """The most urgent trigger raised by `frames`, or `None` to stay asleep.

        `events` is the plant's own stream; only entries newer than the last call
        are considered, so a warning that is still in the window does not fire
        again on the next tick. `risk` is the V10 predicted-risk candidate, already
        formed by the risk monitor, and competes on priority like the rest.
        """
        if not frames:
            return None
        fresh = [event for event in events if event["seq"] > self._last_event_seq]
        if events:
            self._last_event_seq = max(self._last_event_seq, max(e["seq"] for e in events))

        status = str(frames[-1]["plant_status"])
        previous_status, self._last_status = self._last_status, status

        candidates = [
            candidate
            for candidate in (
                _from_events(fresh, "critical", 90),
                _from_status(status, previous_status),
                _from_health(health_alert),
                _from_events(fresh, "warning", 60),
                risk,
                _production_drop(frames),
            )
            if candidate is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda trigger: trigger.priority)


def _from_events(events: list[dict[str, Any]], level: str, priority: int) -> Trigger | None:
    matching = [
        event
        for event in events
        if event["level"] == level and not event["code"].startswith(SELF_PREFIX)
    ]
    if not matching:
        return None
    event = matching[-1]
    return Trigger(
        reason=f"{level}_event",
        detail=f"{event['clock']} {event['message']}",
        key=f"event:{event['code']}",
        priority=priority,
    )


def _from_health(alert: dict[str, Any] | None) -> Trigger | None:
    """The active controller's health has just gone poor. Look at the driver."""
    if not alert:
        return None
    concerns = alert.get("concerns") or ["repeated corrections"]
    controller = alert.get("controller", "controller")
    return Trigger(
        reason="controller_health",
        detail=f"{controller} health graded poor: {'; '.join(concerns[:2])}",
        key=f"controller:{controller}",
        priority=70,
    )


def _from_status(status: str, previous: str) -> Trigger | None:
    if status == previous or status in {"normal", "offline"}:
        return None
    return Trigger(
        reason=f"plant_{status}",
        detail=f"Plant status changed {previous} -> {status}",
        key=f"status:{status}",
        priority=80 if status == "critical" else 50,
    )


def _production_drop(frames: list[dict[str, Any]]) -> Trigger | None:
    """Methane output well below its recent baseline, with power still available."""
    if len(frames) < BASELINE_FRAMES + RECENT_FRAMES:
        return None
    recent = frames[-RECENT_FRAMES:]
    baseline = frames[-(BASELINE_FRAMES + RECENT_FRAMES) : -RECENT_FRAMES]
    if not any(frame["solar_available_kw"] > 0.0 for frame in recent):
        return None  # night: producing nothing is the correct behaviour
    before = _mean(baseline, "ch4_production_kg_h")
    now = _mean(recent, "ch4_production_kg_h")
    if before < MIN_BASELINE_KG_H or now >= DROP_RATIO * before:
        return None
    return Trigger(
        reason="production_drop",
        detail=f"CH4 output fell from {before:.1f} to {now:.1f} kg/h while power was available",
        key="production:drop",
        priority=40,
    )


def _mean(frames: list[dict[str, Any]], key: str) -> float:
    return sum(frame[key] for frame in frames) / len(frames) if frames else 0.0


__all__ = ["BASELINE_FRAMES", "DROP_RATIO", "RECENT_FRAMES", "Trigger", "TriggerWatch", "manual"]

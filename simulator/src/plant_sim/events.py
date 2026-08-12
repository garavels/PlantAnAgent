"""The event stream: what the plant thinks is worth telling someone about.

Events are emitted by the simulator from its own conditions -- never by the
frontend. Two ways in:

    log.emit("warning", "fault.activated", "Electrolyser efficiency reduced")
    log.transition("battery.low", active=soc < 0.1, level="warning",
                   message="Battery entered low-charge state")

`transition()` is the one that matters for continuous conditions: it fires once
on the edge, not every timestep, and emits a matching `info` when the condition
clears. The log is a bounded stream with monotonic sequence numbers, so a
consumer (control room today, agent tomorrow) can ask for everything after the
last event it saw.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

EventLevel = Literal["info", "warning", "critical"]

#: How many events are kept before the oldest are dropped.
EVENT_LIMIT = 400


@dataclass(frozen=True)
class Event:
    """One thing that happened, at one simulated moment."""

    seq: int
    time_s: float
    clock: str
    level: EventLevel
    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "time_s": self.time_s,
            "clock": self.clock,
            "level": self.level,
            "code": self.code,
            "message": self.message,
        }


class EventLog:
    """A bounded, ordered stream of plant events plus the conditions behind it."""

    def __init__(self, limit: int = EVENT_LIMIT) -> None:
        self._events: deque[Event] = deque(maxlen=limit)
        self._conditions: dict[str, bool] = {}
        self._sequence = 0
        self.time_s = 0.0
        self.clock = "D1 00:00"

    def at(self, time_s: float, clock: str) -> None:
        """Set the timestamp events are stamped with (called once per step)."""
        self.time_s = time_s
        self.clock = clock

    def emit(self, level: EventLevel, code: str, message: str) -> Event:
        self._sequence += 1
        event = Event(
            seq=self._sequence,
            time_s=self.time_s,
            clock=self.clock,
            level=level,
            code=code,
            message=message,
        )
        self._events.append(event)
        return event

    def transition(
        self,
        key: str,
        *,
        active: bool,
        level: EventLevel,
        message: str,
        recovery: str | None = None,
    ) -> Event | None:
        """Emit only when `active` flips. Returns the event, if one was emitted."""
        was_active = self._conditions.get(key, False)
        self._conditions[key] = active
        if active and not was_active:
            return self.emit(level, key, message)
        if was_active and not active and recovery:
            return self.emit("info", f"{key}.cleared", recovery)
        return None

    # --- read-only views --------------------------------------------------
    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        events = list(self._events)
        return [event.to_dict() for event in events[-max(1, limit):]]

    def since(self, seq: int) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self._events if event.seq > seq]

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    def clear(self) -> None:
        """Wipe the stream and every remembered condition (used by `reset`)."""
        self._events.clear()
        self._conditions.clear()
        self._sequence = 0


__all__ = ["EVENT_LIMIT", "Event", "EventLevel", "EventLog"]

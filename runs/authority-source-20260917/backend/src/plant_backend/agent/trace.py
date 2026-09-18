"""What the supervisor did, in the only form anyone else gets to see it.

The trace is a bounded stream of *operational* entries -- observation, tool call,
diagnosis, decision, action, result -- not the model's reasoning. Nothing here
carries chain-of-thought: an entry is a short line a human operator could have
written in a shift log, plus a small dict of numbers behind it.

    14:31  prediction   HEPA: CO2 starvation 12h risk increased to 74%
    14:32  observation  H2 production 38% below the last clear hour
    14:32  tool         inspect_component(electrolyser)
    14:33  diagnosis    Probable electrolyser degradation (confidence: high)
    14:33  action       electrolyser_max_kw 3000 -> 1800
    14:34  result       Plant operating in DEGRADED mode

`Incident` is the other half: the current situation the supervisor believes it
is dealing with, so the control room can answer "what is wrong, and did anyone
do anything about it" without replaying the whole trace.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

#: Entry kinds, in the order they usually appear in a run.
EntryKind = Literal[
    "trigger",
    #: V10: what HEPA says may happen. Kept distinct from `observation`, which is
    #: something the plant is doing now -- a forecast is not a measurement.
    "prediction",
    "observation",
    "tool",
    "diagnosis",
    "decision",
    "action",
    "result",
    "error",
]

#: How much confidence the supervisor claims in its diagnosis.
Confidence = Literal["low", "medium", "high"]

#: Trace entries kept before the oldest are dropped.
TRACE_LIMIT = 240
#: Incidents kept in the history handed to the control room.
INCIDENT_LIMIT = 20


@dataclass(frozen=True)
class TraceEntry:
    """One operational step of one supervisor run."""

    seq: int
    run: int
    clock: str
    time_s: float
    kind: EntryKind
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "run": self.run,
            "clock": self.clock,
            "time_s": self.time_s,
            "kind": self.kind,
            "message": self.message,
            "detail": self.detail,
        }


@dataclass
class Incident:
    """One situation the supervisor picked up, and what it made of it."""

    id: str
    run: int
    opened_clock: str
    opened_time_s: float
    #: Why the supervisor woke up (`triggers.Trigger.reason`).
    trigger: str
    trigger_detail: str
    #: Dedupe key: the same key does not re-trigger while the incident is open.
    key: str
    category: str = "investigating"
    component: str | None = None
    confidence: Confidence = "low"
    summary: str = ""
    #: One line per action actually applied, already validated by the backend.
    actions: list[str] = field(default_factory=list)
    human_intervention_required: bool = False
    resolved: bool = False
    closed_clock: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run": self.run,
            "opened_clock": self.opened_clock,
            "opened_time_s": self.opened_time_s,
            "trigger": self.trigger,
            "trigger_detail": self.trigger_detail,
            "key": self.key,
            "category": self.category,
            "component": self.component,
            "confidence": self.confidence,
            "summary": self.summary,
            "actions": list(self.actions),
            "human_intervention_required": self.human_intervention_required,
            "resolved": self.resolved,
            "closed_clock": self.closed_clock,
        }


class AgentTrace:
    """The supervisor's log: a bounded entry stream plus its incident history."""

    def __init__(self, limit: int = TRACE_LIMIT) -> None:
        self._entries: deque[TraceEntry] = deque(maxlen=limit)
        self._incidents: deque[Incident] = deque(maxlen=INCIDENT_LIMIT)
        self._sequence = 0
        self.run = 0
        self.clock = "D1 00:00"
        self.time_s = 0.0

    # --- writing ----------------------------------------------------------
    def at(self, time_s: float, clock: str) -> None:
        """Set the simulated moment entries are stamped with."""
        self.time_s = time_s
        self.clock = clock

    def start_run(self) -> int:
        self.run += 1
        return self.run

    def record(self, kind: EntryKind, message: str, **detail: Any) -> TraceEntry:
        self._sequence += 1
        entry = TraceEntry(
            seq=self._sequence,
            run=self.run,
            clock=self.clock,
            time_s=self.time_s,
            kind=kind,
            message=message,
            detail=detail,
        )
        self._entries.append(entry)
        return entry

    def open_incident(self, incident: Incident) -> Incident:
        self._incidents.append(incident)
        return incident

    # --- reading ----------------------------------------------------------
    def recent(self, limit: int = 60) -> list[dict[str, Any]]:
        entries = list(self._entries)
        return [entry.to_dict() for entry in entries[-max(1, limit) :]]

    def entries_for_run(self, run: int) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self._entries if entry.run == run]

    def incidents(self, limit: int = INCIDENT_LIMIT) -> list[dict[str, Any]]:
        incidents = list(self._incidents)
        return [incident.to_dict() for incident in incidents[-max(1, limit) :]]

    @property
    def last_incident(self) -> Incident | None:
        return self._incidents[-1] if self._incidents else None

    @property
    def unresolved(self) -> list[Incident]:
        """Incidents nobody has declared over yet: the dedupe and status source."""
        return [incident for incident in self._incidents if not incident.resolved]

    def resolve(self, incident: Incident) -> None:
        """Declare one incident over, stamped with the current clock."""
        if not incident.resolved:
            incident.resolved = True
            incident.closed_clock = self.clock

    def resolve_all(self) -> None:
        """Declare every open incident over, stamped with the current clock."""
        for incident in self.unresolved:
            incident.resolved = True
            incident.closed_clock = self.clock

    def clear(self) -> None:
        """Forget everything. Used when the plant itself is reset."""
        self._entries.clear()
        self._incidents.clear()
        self._sequence = 0
        self.run = 0


__all__ = [
    "AgentTrace",
    "Confidence",
    "EntryKind",
    "INCIDENT_LIMIT",
    "Incident",
    "TRACE_LIMIT",
    "TraceEntry",
]

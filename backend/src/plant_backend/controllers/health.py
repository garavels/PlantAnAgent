"""Controller health: is whoever is driving actually doing a good job?

The supervisor needs an answer to that question before it can decide to take a
controller off the plant, and the answer has to be cheap, deterministic and
attributable. So this file counts *controller mistakes* over a rolling window,
never plant misfortune:

    invalid actions        the policy asked for something outside its own range
    unsafe requests        power sent to an offline unit, or the reserve raided
    safety corrections     the guard had to change what the controller asked for
    battery depletion      the pack sat on its reserve while solar was available
    resource starvation    methanation was commanded with an empty feedstock tank
    lost production        no CH4 while both gases, power and the reactor were there
    forced shutdowns       the plant fell into a critical or offline condition
    critical events        the plant shouted while this controller was driving

A DAC outage starves the reactor whoever is driving, which is why starvation is
counted only when the controller *asked for* methanation it could not feed and
got nothing for it: the heuristic declines, a bad policy asks anyway. A reactor
running hand-to-mouth on gas as fast as it is made keeps an empty tank and still
produces, and that is not starvation. The same reasoning shapes every counter
above -- each one is something a competent controller would not do in the
situation it was in.

Corrections are split for the same reason. A policy trained at an 80 kW cap and
now running under a supervisor's 45 kW cap gets clamped every single timestep
while driving the plant perfectly well; grading that as misbehaviour would make
the supervisor's own intervention look like a reason to sack the controller. So
a cap clamp is reported and can raise a concern, but never a grade of `poor`,
while asking an offline unit for power or discharging past the reserve -- wrong
whatever the envelope says -- is graded strictly.

No model, no learning, no scoring subtleties: rates over a window, two
thresholds each, and a grade of `good`, `suspect` or `poor`. Below
`MIN_GRADED_STEPS` the grade is `unrated`, because a controller that has been
driving for four minutes has not shown anything yet.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from plant_sim import Action, PlantState
from plant_sim.state import CRITICAL, OFFLINE

#: Timesteps the window keeps (3 simulated hours at the default dt).
WINDOW_STEPS = 180
#: Steps needed before a grade is anything other than `unrated`.
MIN_GRADED_STEPS = 45
#: Feedstock below this is "not there" (kg).
SHORTAGE_KG = 0.05
#: Bus power above which the plant could have been producing (kW).
PRODUCTIVE_BUS_KW = 50.0
#: SOC margin above the reserve that still counts as running on the reserve.
RESERVE_MARGIN = 0.02

#: Rate thresholds: (suspect, poor) as a fraction of the window's steps.
RATE_THRESHOLDS: dict[str, tuple[float, float]] = {
    "invalid_actions": (0.10, 0.30),
    "unsafe_requests": (0.10, 0.25),
    "battery_depletion": (0.30, 0.60),
    "resource_starvation": (0.20, 0.45),
    "lost_production": (0.35, 0.65),
}
#: Rates that raise a concern but never a grade of `poor`: a controller clamped
#: to a cap every timestep is worth mentioning, not worth sacking.
SOFT_RATE_THRESHOLDS: dict[str, float] = {
    "safety_corrections": 0.75,
}
#: Count thresholds for the things that are bad once, not bad often.
COUNT_THRESHOLDS: dict[str, tuple[int, int]] = {
    "forced_shutdowns": (1, 3),
    "critical_events": (2, 4),
}
#: Counter name -> the `StepRecord` field behind it. The counters are the metrics.
COUNTERS: dict[str, str] = {
    "invalid_actions": "invalid_action",
    "unsafe_requests": "unsafe_request",
    "safety_corrections": "safety_correction",
    "battery_depletion": "battery_depletion",
    "resource_starvation": "resource_starvation",
    "lost_production": "lost_production",
    "forced_shutdowns": "forced_shutdown",
    "critical_events": "critical_events",
}
#: How each counter reads in a sentence.
CONCERN_WORDS: dict[str, str] = {
    "invalid_actions": "actions outside the policy's own range",
    "unsafe_requests": "requests for offline units or the battery reserve",
    "safety_corrections": "actions the safety layer had to correct",
    "battery_depletion": "steps with the battery held on its reserve",
    "resource_starvation": "steps commanding methanation with no feedstock",
    "lost_production": "steps making no methane with everything available",
    "forced_shutdowns": "forced shutdowns",
    "critical_events": "critical plant events",
}


def _value(record: "StepRecord", key: str) -> int:
    """One counter's contribution from one step. Booleans count as one."""
    return int(getattr(record, COUNTERS[key]))


@dataclass(frozen=True)
class StepRecord:
    """One timestep as the health tracker sees it. All booleans on purpose."""

    invalid_action: bool = False
    unsafe_request: bool = False
    safety_correction: bool = False
    battery_depletion: bool = False
    resource_starvation: bool = False
    lost_production: bool = False
    forced_shutdown: bool = False
    critical_events: int = 0


class ControllerHealth:
    """A rolling read on one controller's behaviour. Counts, rates, a grade."""

    def __init__(self, controller_id: str, *, window: int = WINDOW_STEPS) -> None:
        self.controller_id = controller_id
        self.window = window
        self._records: deque[StepRecord] = deque(maxlen=window)
        #: Totals since this controller last took over; the window can forget.
        self.total_steps = 0
        self.totals: dict[str, int] = dict.fromkeys(COUNTERS, 0)

    def clear(self) -> None:
        """Start the assessment again -- a fresh handover deserves a fresh look."""
        self._records.clear()
        self.total_steps = 0
        self.totals = dict.fromkeys(COUNTERS, 0)

    def record(self, record: StepRecord) -> None:
        self._records.append(record)
        self.total_steps += 1
        for key in COUNTERS:
            self.totals[key] += _value(record, key)

    # --- reading ----------------------------------------------------------
    @property
    def steps(self) -> int:
        return len(self._records)

    def counts(self) -> dict[str, int]:
        return {key: sum(_value(record, key) for record in self._records) for key in COUNTERS}

    def concerns(self) -> list[str]:
        """Every counter past its `suspect` threshold, in operator words."""
        if self.steps < MIN_GRADED_STEPS:
            return []
        counts = self.counts()
        found: list[str] = []
        for key, (suspect, _) in RATE_THRESHOLDS.items():
            rate = counts[key] / self.steps
            if rate >= suspect:
                found.append(f"{CONCERN_WORDS[key]}: {rate:.0%} of the last {self.steps} steps")
        for key, suspect in SOFT_RATE_THRESHOLDS.items():
            rate = counts[key] / self.steps
            if rate >= suspect:
                found.append(f"{CONCERN_WORDS[key]}: {rate:.0%} of the last {self.steps} steps")
        for key, (suspect, _) in COUNT_THRESHOLDS.items():
            if counts[key] >= suspect:
                found.append(f"{CONCERN_WORDS[key]}: {counts[key]} in the last {self.steps} steps")
        return found

    @property
    def grade(self) -> str:
        """`unrated`, `good`, `suspect` or `poor`. The supervisor reads this."""
        if self.steps < MIN_GRADED_STEPS:
            return "unrated"
        counts = self.counts()
        poor = any(
            counts[key] / self.steps >= threshold[1] for key, threshold in RATE_THRESHOLDS.items()
        ) or any(counts[key] >= threshold[1] for key, threshold in COUNT_THRESHOLDS.items())
        if poor:
            return "poor"
        return "suspect" if self.concerns() else "good"

    @property
    def healthy(self) -> bool:
        return self.grade in {"unrated", "good"}

    @property
    def summary(self) -> str:
        """One line an operator can read: the grade, and what is behind it."""
        if self.steps < MIN_GRADED_STEPS:
            return f"{self.steps} of {MIN_GRADED_STEPS} steps driven: not graded yet"
        concerns = self.concerns()
        if not concerns:
            return f"Nothing of concern in the last {self.steps} steps"
        return f"Graded {self.grade}: {'; '.join(concerns[:2])}"

    def to_dict(self) -> dict[str, Any]:
        counts = self.counts()
        steps = max(1, self.steps)
        return {
            "controller": self.controller_id,
            "grade": self.grade,
            "steps": self.steps,
            "total_steps": self.total_steps,
            "counts": counts,
            "totals": dict(self.totals),
            "rates": {
                key: round(counts[key] / steps, 3)
                for key in (*RATE_THRESHOLDS, *SOFT_RATE_THRESHOLDS)
            },
            "concerns": self.concerns(),
            "summary": self.summary,
        }


def assess(
    *,
    before: PlantState,
    after: PlantState,
    requested: Action,
    adjustments: tuple[str, ...] | list[str],
    invalid_action: bool,
    critical_events: int,
    reserve_soc: float,
) -> StepRecord:
    """Turn one applied timestep into the record the tracker keeps.

    `before` is what the controller saw, `after` is what the plant did with the
    action -- both telemetry, never the simulator's private truth.
    """
    bus_kw = max(0.0, before.solar_available_kw)
    fed = min(before.h2_level_kg, before.co2_level_kg) > SHORTAGE_KG
    reactor = after.health.get("methanator")
    reactor_online = after.methanator_online and not (reactor is not None and reactor.offline)
    return StepRecord(
        invalid_action=invalid_action,
        unsafe_request=any(_unsafe(text) for text in adjustments),
        safety_correction=bool(adjustments),
        battery_depletion=(
            bus_kw > PRODUCTIVE_BUS_KW
            and after.battery_soc <= reserve_soc + RESERVE_MARGIN
            and after.battery_soc < before.battery_soc
        ),
        resource_starvation=(
            requested.methanator_share > 0.0 and not fed and after.ch4_production_kg_h <= 0.0
        ),
        lost_production=(
            bus_kw > PRODUCTIVE_BUS_KW
            and fed
            and reactor_online
            and after.ch4_production_kg_h <= 0.0
            and after.ch4_fill < 0.97
        ),
        forced_shutdown=after.plant_status in {CRITICAL, OFFLINE}
        and before.plant_status not in {CRITICAL, OFFLINE},
        critical_events=critical_events,
    )


def _unsafe(adjustment: str) -> bool:
    """Was this correction of the kind no envelope could make reasonable?"""
    text = adjustment.lower()
    return "offline" in text or "reserve" in text


__all__ = [
    "COUNTERS",
    "COUNT_THRESHOLDS",
    "MIN_GRADED_STEPS",
    "RATE_THRESHOLDS",
    "SOFT_RATE_THRESHOLDS",
    "WINDOW_STEPS",
    "ControllerHealth",
    "StepRecord",
    "assess",
]

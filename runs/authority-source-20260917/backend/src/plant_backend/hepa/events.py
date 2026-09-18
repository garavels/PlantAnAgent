"""What counts as an event -- defined once, for every module that needs it.

Label leakage between modules is the classic way a predictive pipeline lies to
itself: the collector calls a tank "starved" below 5 kg, the label generator uses
1 kg, the dashboard uses "status != normal", and the reported AUROC measures
nothing in particular. So the rules live here, in one table, and the collector,
the label generator, the evaluation code and the API all read *this*.

    for name, active in occurring(telemetry).items(): ...

Each rule is a pure predicate over a single `PlantState` *telemetry* sample (what
an operator could see, sensor faults included) plus `EventThresholds`. An event
"occurs at step t" if its predicate holds for the telemetry of step t; horizon
labels are built from those per-step occurrences in `data/labels.py`.

The five events, and exactly what qualifies:

`battery_reserve_violation`
    Observed SOC at or below the operating reserve (default 0.15, the default
    `Constraints.battery_min_soc`, not the pack's hard floor of 0.05). The plant
    still runs here -- this is the envelope being eaten into, which is why it is
    worth predicting hours ahead.

`h2_starvation`
    The methanator was offered power this step, is *not* running, and observed H2
    inventory is below `empty_kg`. Mirrors the simulator's `methanator.h2_starved`
    condition. A merely low tank is not starvation: nothing has stopped yet.

`co2_starvation`
    The same three conditions with the CO2 tank: methanator commanded, methanator
    not online, observed CO2 inventory below `empty_kg`. Stated explicitly because
    "CO2 starvation" is otherwise read three different ways -- it is *not* "the
    CO2 tank is low", *not* "DAC is down", and *not* "capture rate is zero"
    (capture is zero every night by design).

`forced_production_shutdown`
    A production unit was offered power and did not run: electrolyser or
    methanator commanded (`share > 0`) while `*_online` is false. Covers both
    starvation and an outage or a stack falling under its minimum load, so it
    deliberately overlaps the two starvation events -- the labels are multi-label,
    not mutually exclusive.

`critical_plant_state`
    Reported `plant_status` is `critical` or `offline`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from plant_sim import PlantState
from plant_sim.state import CRITICAL, OFFLINE

#: Bumped when a rule below changes meaning. Checkpoints record it and refuse to
#: serve predictions trained against a different definition of the same word.
EVENT_SPEC_VERSION = 1


@dataclass(frozen=True)
class EventThresholds:
    """The numbers the rules compare against. Configurable, recorded, shared."""

    #: SOC at or below which the reserve counts as violated.
    reserve_soc: float = 0.15
    #: Inventory (kg) below which a feedstock tank counts as empty. The
    #: simulator's own `EMPTY_KG`, restated here so a rule change is visible.
    empty_kg: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {"reserve_soc": self.reserve_soc, "empty_kg": self.empty_kg}

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "EventThresholds":
        if not raw:
            return cls()
        return cls(
            reserve_soc=float(raw.get("reserve_soc", cls.reserve_soc)),
            empty_kg=float(raw.get("empty_kg", cls.empty_kg)),
        )


@dataclass(frozen=True)
class EventDefinition:
    """One predictable failure: its name, its rule, and why it matters."""

    name: str
    summary: str
    rule: str
    predicate: Callable[[PlantState, EventThresholds], bool]

    def occurs(self, state: PlantState, thresholds: EventThresholds) -> bool:
        return bool(self.predicate(state, thresholds))

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "summary": self.summary, "rule": self.rule}


def _starved(state: PlantState, level_kg: float, thresholds: EventThresholds) -> bool:
    return (
        state.action.methanator_share > 0.0
        and not state.methanator_online
        and level_kg < thresholds.empty_kg
    )


EVENTS: tuple[EventDefinition, ...] = (
    EventDefinition(
        name="battery_reserve_violation",
        summary="Stored energy fell into the reserve the operator set aside.",
        rule="battery_soc <= thresholds.reserve_soc",
        predicate=lambda state, thresholds: state.battery_soc <= thresholds.reserve_soc,
    ),
    EventDefinition(
        name="h2_starvation",
        summary="The methanator was asked to run and had no hydrogen.",
        rule=(
            "action.methanator_share > 0 and not methanator_online "
            "and h2_level_kg < thresholds.empty_kg"
        ),
        predicate=lambda state, thresholds: _starved(state, state.h2_level_kg, thresholds),
    ),
    EventDefinition(
        name="co2_starvation",
        summary="The methanator was asked to run and had no captured CO2.",
        rule=(
            "action.methanator_share > 0 and not methanator_online "
            "and co2_level_kg < thresholds.empty_kg"
        ),
        predicate=lambda state, thresholds: _starved(state, state.co2_level_kg, thresholds),
    ),
    EventDefinition(
        name="forced_production_shutdown",
        summary="A unit was offered power and did not run.",
        rule=(
            "(action.electrolyser_share > 0 and not electrolyser_online) or "
            "(action.methanator_share > 0 and not methanator_online)"
        ),
        predicate=lambda state, _: (
            (state.action.electrolyser_share > 0.0 and not state.electrolyser_online)
            or (state.action.methanator_share > 0.0 and not state.methanator_online)
        ),
    ),
    EventDefinition(
        name="critical_plant_state",
        summary="The plant reported itself critical or offline.",
        rule="plant_status in {critical, offline}",
        predicate=lambda state, _: state.plant_status in (CRITICAL, OFFLINE),
    ),
)

EVENT_NAMES: tuple[str, ...] = tuple(event.name for event in EVENTS)
EVENT_COUNT = len(EVENTS)


def definition(name: str) -> EventDefinition:
    for event in EVENTS:
        if event.name == name:
            return event
    raise KeyError(f"unknown event: {name!r} (known: {', '.join(EVENT_NAMES)})")


def occurring(
    state: PlantState, thresholds: EventThresholds | None = None
) -> dict[str, bool]:
    """Which events are happening in this one telemetry sample."""
    limits = thresholds or EventThresholds()
    return {event.name: event.occurs(state, limits) for event in EVENTS}


def occurrence_matrix(
    states: Iterable[PlantState],
    thresholds: EventThresholds | None = None,
    names: Sequence[str] = EVENT_NAMES,
) -> np.ndarray:
    """`(T, E)` bool: per step, per event, did it occur. The label primitive."""
    limits = thresholds or EventThresholds()
    definitions = [definition(name) for name in names]
    rows = [
        [event.occurs(state, limits) for event in definitions] for state in states
    ]
    if not rows:
        return np.zeros((0, len(definitions)), dtype=bool)
    return np.asarray(rows, dtype=bool)


def describe() -> dict[str, Any]:
    """The event contract as JSON, for the docs, the API and the checkpoints."""
    return {
        "spec_version": EVENT_SPEC_VERSION,
        "events": [event.to_dict() for event in EVENTS],
        "thresholds": EventThresholds().to_dict(),
    }


__all__ = [
    "EVENTS",
    "EVENT_COUNT",
    "EVENT_NAMES",
    "EVENT_SPEC_VERSION",
    "EventDefinition",
    "EventThresholds",
    "definition",
    "describe",
    "occurrence_matrix",
    "occurring",
]

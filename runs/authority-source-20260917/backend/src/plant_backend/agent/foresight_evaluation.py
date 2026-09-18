"""V10: does foresight change anything? The A/B/C harness.

    uv run python -m plant_backend.agent.foresight_evaluation

Three arms of the same scenario, same seed, same controller:

    A  controller           the plant drives itself, nobody supervises
    B  controller + supervisor      V8: it reacts to what has happened
    C  controller + supervisor + HEPA   V10: it also sees what may happen

and the questions worth asking of advance warning:

    events         did the negative event actually happen?
    shutdowns      did a unit get asked to run and fail to?
    acted          how often was the envelope retuned?
    early          how many of those changes came *before* the event
    unnecessary    how many came on a plant that had nothing wrong with it
    ch4            what did it cost in product
    lead           simulated time between the first warning and the event
    risk           predicted risk at the first intervention, and at the end

**These runs are not evidence about HEPA.** There is no trained checkpoint in
this repository, so arm C is driven by `hepa.fixtures.ScriptedForecaster` -- a
hand-written script, labelled `fixture:*` in every forecast, snapshot and report
line it touches. What the harness tests is the *integration*: that a rising
predicted risk reaches the supervisor, that the supervisor corroborates it
against telemetry before acting, that the envelope change lands in the controller
and not in the simulator, and that the arms are otherwise identical. Replace the
fixture with a checkpoint (`PLANT_HEPA_CHECKPOINT`) and the same harness measures
a model; until then it measures wiring.

Two columns need reading carefully. `events` counts the *strict* event from
`hepa.events` -- the methanator commanded, not running, the tank under `empty_kg`
-- so a zero there means the plant never actually starved, in any arm, which is
what the heuristic controller is for; `shut` is the broader forced shutdown.
And `risk` after an intervention comes from the script, which cannot respond to
what the supervisor did: a flat `0.81 -> 0.81` is the fixture repeating itself,
not a risk that refused to fall.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Mapping

from plant_sim import PlantState

from ..hepa import fixtures
from ..hepa.events import EVENT_NAMES, occurring
from ..orchestration import PlantRuntime
from .engines import Engine, RuleEngine
from .risk import LEVEL_RANK, RISING, RiskPolicy, replace_refresh
from .supervisor import SupervisorAgent

#: Simulated minutes of plant per supervisory offer, and how long a case runs.
BATCH_MIN = 10
DEFAULT_HOURS = 12.0
#: Forecast cadence for the harness: often enough that a 12 h case sees the
#: script move, still far from per-timestep.
REFRESH_S = 900.0

#: The three arms, in the order they are reported.
CONTROLLER = "controller"
SUPERVISOR = "supervisor"
SUPERVISOR_HEPA = "supervisor+hepa"
ARMS: tuple[str, ...] = (CONTROLLER, SUPERVISOR, SUPERVISOR_HEPA)

#: One hand-written forecast per refresh; the last entry repeats. Fixtures.
Script = tuple[Mapping[str, Mapping[str, float]], ...]

CO2_SCRIPT: Script = (
    {"co2_starvation": {"1h": 0.02, "3h": 0.08, "6h": 0.18, "12h": 0.34}},
    {"co2_starvation": {"1h": 0.03, "3h": 0.14, "6h": 0.38, "12h": 0.62}},
    {"co2_starvation": {"1h": 0.05, "3h": 0.24, "6h": 0.56, "12h": 0.81}},
)

BATTERY_SCRIPT: Script = (
    {"battery_reserve_violation": {"1h": 0.04, "3h": 0.12, "6h": 0.26, "12h": 0.38}},
    {"battery_reserve_violation": {"1h": 0.06, "3h": 0.22, "6h": 0.44, "12h": 0.66}},
    {"battery_reserve_violation": {"1h": 0.09, "3h": 0.31, "6h": 0.58, "12h": 0.79}},
)

CALM_SCRIPT: Script = (
    {name: {horizon: 0.02 for horizon in ("1h", "3h", "6h", "12h")} for name in EVENT_NAMES},
)


@dataclass(frozen=True)
class ForesightCase:
    """One scenario where advance warning ought to matter, and its fixture."""

    scenario: str
    #: The negative event this case is about, from `hepa.events`.
    event: str
    #: The fixture forecast arm C is driven by. Not a prediction.
    script: Script
    hours: float = DEFAULT_HOURS
    #: `False` for the healthy plant: there is nothing to be early about, so any
    #: envelope change is counted as an unnecessary intervention.
    at_risk: bool = True


CASES: tuple[ForesightCase, ...] = (
    ForesightCase(scenario="dac_failure", event="co2_starvation", script=CO2_SCRIPT),
    ForesightCase(
        scenario="cloud_cover", event="battery_reserve_violation", script=BATTERY_SCRIPT
    ),
    ForesightCase(
        scenario="clear_day", event="co2_starvation", script=CALM_SCRIPT, at_risk=False
    ),
)


@dataclass
class Intervention:
    """One supervisory run that changed the envelope, and its context."""

    time_s: float
    clock: str
    actions: tuple[str, ...]
    #: Whether the case's event had already occurred when this was applied.
    after_event: bool
    #: Predicted risk of the case's event at the longest horizon, at the time.
    risk: float | None


@dataclass
class ArmOutcome:
    """What one arm of one case did. Counts and totals only, no verdicts."""

    scenario: str
    arm: str
    #: Copied from the case, so `unnecessary` can be read without it.
    at_risk: bool = True
    #: `checkpoint`, or `fixture:*` when the forecast was hand-written.
    source: str = "none"
    runs: int = 0
    #: Occurrences of the case's event, counted as episodes, not steps.
    events: int = 0
    #: Episodes of `forced_production_shutdown`.
    shutdowns: int = 0
    interventions: list[Intervention] = field(default_factory=list)
    ch4_kg: float = 0.0
    min_battery_soc: float = 1.0
    #: First time the event occurred, and first time HEPA flagged it.
    first_event_s: float | None = None
    first_warning_s: float | None = None
    risk_at_end: float | None = None
    elapsed_s: float = 0.0

    @property
    def acted(self) -> int:
        return len(self.interventions)

    @property
    def early(self) -> int:
        """Envelope changes applied before the event they were meant to head off."""
        return sum(1 for step in self.interventions if not step.after_event)

    @property
    def unnecessary(self) -> int:
        """Changes made to a plant that had nothing wrong with it."""
        return 0 if self.at_risk else self.acted

    @property
    def prevented(self) -> bool:
        """Warned, intervened early, and the event never happened. Not a claim."""
        return self.first_warning_s is not None and self.early > 0 and self.events == 0

    @property
    def lead_s(self) -> float | None:
        """Simulated seconds from the first warning to the event, or to the end."""
        if self.first_warning_s is None:
            return None
        end = self.first_event_s if self.first_event_s is not None else self.elapsed_s
        return max(0.0, end - self.first_warning_s)

    @property
    def risk_before(self) -> float | None:
        return next((step.risk for step in self.interventions if step.risk is not None), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "arm": self.arm,
            "source": self.source,
            "runs": self.runs,
            "events": self.events,
            "shutdowns": self.shutdowns,
            "acted": self.acted,
            "early": self.early,
            "unnecessary": self.unnecessary,
            "prevented": self.prevented,
            "ch4_kg": round(self.ch4_kg, 3),
            "min_battery_soc": round(self.min_battery_soc, 3),
            "lead_s": self.lead_s,
            "risk_before": self.risk_before,
            "risk_after": self.risk_at_end,
            "actions": [action for step in self.interventions for action in step.actions],
        }


async def run_arm(
    case: ForesightCase,
    arm: str,
    *,
    engine: Engine | None = None,
    hours: float | None = None,
    batch_min: int = BATCH_MIN,
) -> ArmOutcome:
    """Run one scenario under one arm and count what happened.

    Identical in every arm but the supervision: arm A runs with the supervisor
    switched off, arm B with it on and no forecast available (exactly the V8
    plant), arm C with it on and the case's fixture script behind the HEPA tool.
    """
    runtime = PlantRuntime()
    forecaster = (
        fixtures.ScriptedForecaster(list(case.script), name=f"{case.scenario}")
        if arm == SUPERVISOR_HEPA
        else fixtures.UnavailableForecaster()
    )
    supervisor = SupervisorAgent(
        runtime,
        engine or RuleEngine(),
        enabled=arm != CONTROLLER,
        forecaster=forecaster,
        risk_policy=replace_refresh(RiskPolicy.from_env(), REFRESH_S),
    )
    runtime.supervisor = supervisor
    runtime.load_scenario(case.scenario)

    outcome = ArmOutcome(scenario=case.scenario, arm=arm, at_risk=case.at_risk)
    occurring_now = {case.event: False, "forced_production_shutdown": False}
    batch: list[dict[str, Any]] = []
    for step in range(int(case.hours if hours is None else hours) * 60):
        batch.extend(runtime.advance(1))
        observation: PlantState = runtime.session.telemetry
        outcome.elapsed_s = float(observation.time_s)
        outcome.ch4_kg = float(observation.ch4_produced_total_kg)
        outcome.min_battery_soc = min(outcome.min_battery_soc, float(observation.battery_soc))
        _count_events(outcome, case, observation, occurring_now)
        if (step + 1) % batch_min:
            continue
        before = len(supervisor.trace.incidents())
        trigger = supervisor.notice(batch)
        batch = []
        _note_warning(outcome, supervisor, case)
        if trigger is None:
            continue
        await supervisor.investigate(trigger)
        _note_intervention(outcome, supervisor, case, occurring_now[case.event], before)

    outcome.runs = supervisor.runs
    outcome.source = supervisor.risk.source if arm == SUPERVISOR_HEPA else "none"
    outcome.risk_at_end = _risk(supervisor, case)
    return outcome


def _count_events(
    outcome: ArmOutcome,
    case: ForesightCase,
    observation: PlantState,
    active: dict[str, bool],
) -> None:
    """Rising edges only: one drained tank is one event, not one per timestep."""
    happening = occurring(observation)
    for name in active:
        now = bool(happening.get(name, False))
        if now and not active[name]:
            if name == case.event:
                outcome.events += 1
                if outcome.first_event_s is None:
                    outcome.first_event_s = float(observation.time_s)
            else:
                outcome.shutdowns += 1
        active[name] = now


def _risk(supervisor: SupervisorAgent, case: ForesightCase) -> float | None:
    """The case's event at the longest configured horizon, if HEPA answered."""
    horizon = supervisor.risk.policy.horizons[-1]
    entry = supervisor.risk.risks.get(case.event, {}).get(horizon)
    return None if entry is None else round(entry.probability, 4)


def _note_warning(
    outcome: ArmOutcome, supervisor: SupervisorAgent, case: ForesightCase
) -> None:
    """The first moment the forecast for this event was worth attention."""
    if outcome.first_warning_s is not None:
        return
    horizon = supervisor.risk.policy.horizons[-1]
    entry = supervisor.risk.risks.get(case.event, {}).get(horizon)
    if entry is None:
        return
    policy = supervisor.risk.policy
    worth = LEVEL_RANK.get(entry.level, 0) >= LEVEL_RANK[policy.rising_attention] and (
        entry.trend == RISING or LEVEL_RANK[entry.level] >= LEVEL_RANK[policy.attention]
    )
    if worth:
        outcome.first_warning_s = supervisor.risk.updated_time_s


def _note_intervention(
    outcome: ArmOutcome,
    supervisor: SupervisorAgent,
    case: ForesightCase,
    after_event: bool,
    before: int,
) -> None:
    """Record the run only if it actually changed the envelope."""
    incidents = supervisor.trace.incidents()
    if len(incidents) <= before:
        return
    latest = incidents[-1]
    actions = tuple(str(action) for action in latest["actions"])
    if not actions:
        return
    observation = supervisor.runtime.session.telemetry
    outcome.interventions.append(
        Intervention(
            time_s=float(observation.time_s),
            clock=observation.clock,
            actions=actions,
            after_event=after_event,
            risk=_risk(supervisor, case),
        )
    )


async def run_case(
    case: ForesightCase, *, engine: Engine | None = None, hours: float | None = None
) -> list[ArmOutcome]:
    """All three arms of one case, in order. One plant at a time, on purpose."""
    return [await run_arm(case, arm, engine=engine, hours=hours) for arm in ARMS]


async def run_suite(
    cases: tuple[ForesightCase, ...] = CASES,
    *,
    engine: Engine | None = None,
    hours: float | None = None,
) -> list[ArmOutcome]:
    outcomes: list[ArmOutcome] = []
    for case in cases:
        outcomes.extend(await run_case(case, engine=engine, hours=hours))
    return outcomes


def report(outcomes: list[ArmOutcome]) -> str:
    """The comparison as a table, with the fixture caveat attached to it."""
    lines = [
        f"{'scenario':24} {'arm':16} {'events':>6} {'shut':>5} {'acted':>5} "
        f"{'early':>5} {'unnec':>5} {'ch4 kg':>8} {'lead':>7} {'risk':>11}",
        "-" * 104,
    ]
    for outcome in outcomes:
        lead = "-" if outcome.lead_s is None else f"{outcome.lead_s / 3600.0:.1f}h"
        before = "-" if outcome.risk_before is None else f"{outcome.risk_before:.2f}"
        after = "-" if outcome.risk_at_end is None else f"{outcome.risk_at_end:.2f}"
        lines.append(
            f"{outcome.scenario:24} {outcome.arm:16} {outcome.events:>6} "
            f"{outcome.shutdowns:>5} {outcome.acted:>5} {outcome.early:>5} "
            f"{outcome.unnecessary:>5} {outcome.ch4_kg:>8.1f} {lead:>7} "
            f"{before + ' -> ' + after:>11}"
        )
    sources = sorted({outcome.source for outcome in outcomes if outcome.source != "none"})
    lines.append("-" * 104)
    lines.append(
        "Arm C forecasts came from: "
        + (", ".join(sources) or "nothing")
        + ". A fixture source means these numbers describe the integration, "
        "not the model: they are not evidence that HEPA improves the plant."
    )
    return "\n".join(lines)


def main() -> int:
    print(report(asyncio.run(run_suite())))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "ARMS",
    "CASES",
    "CONTROLLER",
    "SUPERVISOR",
    "SUPERVISOR_HEPA",
    "ArmOutcome",
    "ForesightCase",
    "Intervention",
    "main",
    "report",
    "run_arm",
    "run_case",
    "run_suite",
]

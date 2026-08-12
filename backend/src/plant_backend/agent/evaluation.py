"""A small regression harness: five scenarios, six questions each.

    uv run python -m plant_backend.agent.evaluation

This is not a benchmark. It answers the only questions that matter for a
supervisor that is supposed to help rather than meddle:

    triggered            did it wake up at all?
    investigated         did it look at the unit that was actually misbehaving?
    identified           did it land on the right category?
    acted                did it change the envelope (or correctly leave it alone)?
    reserve held         did the battery stay above its floor?
    no meddling          did it keep its hands off a healthy plant?

It runs the plant exactly as the backend does -- controller, envelope, validation
guard -- driving `advance()` in batches instead of on a wall clock, and offering
each batch to the supervisor. With `RuleEngine` (the default) the whole run is
deterministic: same scenario, same trace, every time, which is what makes it
usable as a test. Pass `LLMEngine()` to exercise the model path instead; results
then vary and this becomes a smoke test rather than a regression one.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from ..orchestration import PlantRuntime
from .engines import Engine, RuleEngine
from .supervisor import SupervisorAgent

#: Simulated minutes applied per batch, and how long a case runs for.
BATCH_MIN = 10
DEFAULT_HOURS = 8.0


@dataclass(frozen=True)
class Case:
    """One scenario and what a competent supervisor should make of it."""

    scenario: str
    #: `False` for the healthy plant: waking up is allowed, intervening is not.
    should_trigger: bool
    #: Diagnoses accepted as correct. The first is the one we are aiming at.
    categories: tuple[str, ...]
    #: The unit we expect it to have inspected, if any.
    component: str | None = None
    #: Whether the envelope should have been retuned.
    should_act: bool = True


CASES: tuple[Case, ...] = (
    Case(
        scenario="clear_day",
        should_trigger=False,
        categories=("normal",),
        should_act=False,
    ),
    Case(
        scenario="electrolyser_degradation",
        should_trigger=True,
        categories=("electrolyser_degradation",),
        component="electrolyser",
    ),
    Case(
        scenario="dac_failure",
        should_trigger=True,
        categories=("dac_unavailable",),
        component="dac",
    ),
    Case(
        scenario="cloud_cover",
        should_trigger=True,
        categories=("energy_shortfall", "battery_low"),
        component="solar",
    ),
    Case(
        scenario="sensor_drift",
        should_trigger=True,
        categories=("sensor_inconsistency",),
        should_act=False,
    ),
)


@dataclass
class Outcome:
    """What one case actually did, and where it disagreed with the case."""

    scenario: str
    runs: int = 0
    categories: list[str] = field(default_factory=list)
    inspected: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    plant_status: str = "normal"
    min_battery_soc: float = 1.0
    reserve_soc: float = 0.0
    failures: list[str] = field(default_factory=list)

    @property
    def triggered(self) -> bool:
        return self.runs > 0

    @property
    def reserve_held(self) -> bool:
        return self.min_battery_soc >= self.reserve_soc - 1e-6

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "runs": self.runs,
            "triggered": self.triggered,
            "categories": list(self.categories),
            "inspected": list(self.inspected),
            "actions": list(self.actions),
            "plant_status": self.plant_status,
            "min_battery_soc": round(self.min_battery_soc, 3),
            "reserve_held": self.reserve_held,
            "passed": self.passed,
            "failures": list(self.failures),
        }


async def run_case(
    case: Case,
    *,
    engine: Engine | None = None,
    hours: float = DEFAULT_HOURS,
    batch_min: int = BATCH_MIN,
) -> Outcome:
    """Run one scenario end to end and score the supervisor against `case`."""
    runtime = PlantRuntime()
    supervisor = SupervisorAgent(runtime, engine or RuleEngine(), enabled=True)
    runtime.supervisor = supervisor
    runtime.load_scenario(case.scenario)

    outcome = Outcome(
        scenario=case.scenario,
        reserve_soc=runtime.constraints.reserve_soc(runtime.config),
    )
    for _ in range(int(hours * 60 / batch_min)):
        frames = runtime.advance(batch_min)
        outcome.min_battery_soc = min(
            outcome.min_battery_soc, *(frame["battery_soc"] for frame in frames)
        )
        trigger = supervisor.notice(frames)
        if trigger is not None:
            await supervisor.investigate(trigger)

    outcome.runs = supervisor.runs
    outcome.plant_status = runtime.session.telemetry.plant_status
    outcome.categories = [incident["category"] for incident in supervisor.trace.incidents()]
    outcome.actions = [
        action for incident in supervisor.trace.incidents() for action in incident["actions"]
    ]
    outcome.inspected = sorted(
        {
            str(entry["detail"].get("component"))
            for entry in supervisor.trace.recent(240)
            if entry["kind"] == "tool" and entry["detail"].get("component")
        }
    )
    outcome.failures = _score(case, outcome)
    return outcome


def _score(case: Case, outcome: Outcome) -> list[str]:
    """Every way this run fell short. Empty means the case passed."""
    failures: list[str] = []
    diagnosed = [category for category in outcome.categories if category != "normal"]
    if case.should_trigger and not outcome.triggered:
        failures.append("supervisor never woke up")
    if case.should_trigger and not any(
        category in case.categories for category in outcome.categories
    ):
        failures.append(
            f"expected one of {case.categories}, diagnosed {outcome.categories or ['nothing']}"
        )
    if not case.should_trigger and diagnosed:
        failures.append(f"diagnosed {diagnosed} on a healthy plant")
    if case.component and case.component not in outcome.inspected:
        failures.append(f"never inspected the {case.component} (looked at {outcome.inspected})")
    if case.should_act and not outcome.actions:
        failures.append("diagnosed a problem but changed nothing")
    if not case.should_act and outcome.actions:
        failures.append(f"intervened when it should not have: {outcome.actions}")
    if not outcome.reserve_held:
        failures.append(
            f"battery fell to {outcome.min_battery_soc:.2f}, below its "
            f"{outcome.reserve_soc:.2f} floor"
        )
    return failures


async def run_suite(
    cases: tuple[Case, ...] = CASES,
    *,
    engine: Engine | None = None,
    hours: float = DEFAULT_HOURS,
) -> list[Outcome]:
    """Every case, in order. Sequential on purpose: one plant, one supervisor."""
    return [await run_case(case, engine=engine, hours=hours) for case in cases]


def report(outcomes: list[Outcome]) -> str:
    """The suite as a table a human can read in one glance."""
    lines = [
        f"{'scenario':26} {'runs':>4}  {'diagnosis':26} {'acted':>5}  result",
        "-" * 84,
    ]
    for outcome in outcomes:
        diagnosis = next(
            (category for category in outcome.categories if category != "normal"),
            outcome.categories[0] if outcome.categories else "-",
        )
        lines.append(
            f"{outcome.scenario:26} {outcome.runs:>4}  {diagnosis:26} "
            f"{len(outcome.actions):>5}  {'pass' if outcome.passed else 'FAIL'}"
        )
        lines.extend(f"{'':32} - {failure}" for failure in outcome.failures)
    passed = sum(1 for outcome in outcomes if outcome.passed)
    lines.append("-" * 84)
    lines.append(f"{passed}/{len(outcomes)} cases passed")
    return "\n".join(lines)


def main() -> int:
    outcomes = asyncio.run(run_suite())
    print(report(outcomes))
    return 0 if all(outcome.passed for outcome in outcomes) else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = ["BATCH_MIN", "CASES", "Case", "Outcome", "main", "report", "run_case", "run_suite"]

"""V8 regression harness: the same scenarios, driven three different ways.

    uv run python -m plant_backend.agent.hierarchy

`evaluation.py` asks whether the supervisor diagnoses a fault correctly. This
file asks the question V8 adds: does putting a supervisor *above* a timestep
controller make the plant behave better, or does it just make it twitch? So each
scenario is run three times --

    heuristic        rules only, no supervisor          (the V0-V6 baseline)
    rl               a policy only, no supervisor       (V7)
    rl+supervisor    the same policy, supervised        (V8)

-- and the same figures are collected from all three: methane made, critical
events, forced shutdowns, safety corrections, controller switches, and whether
the plant ended the run running normally.

Two notes on honesty. First, the RL arms drive a *fixed* checkpoint, not a
trained policy: nothing here trains anything, and a constant allocation is
enough to exercise the hierarchy (the `poor_rl` arm deliberately drives badly,
which is exactly the situation the supervisor is supposed to notice). Second,
this is regression infrastructure and not a benchmark: the numbers are only
meaningful against each other, on the same scenario, in the same version.
"""

from __future__ import annotations

import asyncio
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ..controllers import SAFE_CONTROLLER_ID
from ..orchestration import PlantRuntime
from ..rl import checkpoints
from .engines import Engine, RuleEngine
from .supervisor import SupervisorAgent

#: Simulated minutes applied per batch, and how long an arm runs for.
BATCH_MIN = 10
DEFAULT_HOURS = 10.0

#: A steady, unambitious allocation: half the bus to the stack, the rest stored.
#: In policy units, where -1 is nothing and +1 is everything.
STEADY_POLICY = (-0.4, 0.0, -0.7, -0.8, -1.0)
#: Everything, at once, past the top of its own range: the controller-from-hell.
#: Out of range every step (invalid), all four consumers maxed on an
#: oversubscribed bus, and the battery lent out until it hits its floor.
RECKLESS_POLICY = (2.0, 2.0, 2.0, 2.0, 2.0)


@dataclass(frozen=True)
class Arm:
    """One way of driving the plant for a whole run."""

    id: str
    controller: str
    #: Whether the supervisory agent is awake above the controller.
    supervised: bool
    #: Fixed policy vector for the RL arms; `None` for the heuristic.
    policy: tuple[float, ...] | None = None


ARMS: dict[str, Arm] = {
    "heuristic": Arm("heuristic", controller="heuristic", supervised=False),
    "rl": Arm("rl", controller="rl", supervised=False, policy=STEADY_POLICY),
    "rl+supervisor": Arm("rl+supervisor", controller="rl", supervised=True, policy=STEADY_POLICY),
    "poor_rl": Arm("poor_rl", controller="rl", supervised=False, policy=RECKLESS_POLICY),
    "poor_rl+supervisor": Arm(
        "poor_rl+supervisor", controller="rl", supervised=True, policy=RECKLESS_POLICY
    ),
}


@dataclass(frozen=True)
class Case:
    """One scenario, the arms to run it with, and what V8 promises about them."""

    name: str
    scenario: str
    arms: tuple[str, ...]
    #: What the supervised arm must end up doing. All optional, all checked.
    #: Controller the supervised arm must still have on the plant at the end.
    ends_with: str | None = None
    #: Upper bound on switches: 0 for "leave it alone", 1 for "hand over once".
    max_switches: int | None = None
    #: Lower bound, for the cases where a handover is the whole point.
    min_switches: int = 0
    #: Whether the envelope must have been retuned.
    expect_retune: bool | None = None
    #: Whether the plant must be running normally again by the end.
    expect_recovery: bool = False
    #: Whether safe mode is an acceptable place to end the run.
    allow_safe_mode: bool = False


CASES: tuple[Case, ...] = (
    Case(
        name="normal",
        scenario="clear_day",
        arms=("heuristic", "rl", "rl+supervisor"),
        ends_with="rl",
        max_switches=0,
        expect_retune=False,
        expect_recovery=True,
    ),
    Case(
        name="electrolyser_degradation",
        scenario="electrolyser_degradation",
        arms=("heuristic", "rl", "rl+supervisor"),
        ends_with="rl",
        max_switches=0,
        expect_retune=True,
    ),
    Case(
        name="poor_rl",
        scenario="clear_day",
        arms=("poor_rl", "poor_rl+supervisor"),
        ends_with="heuristic",
        min_switches=1,
        max_switches=2,
    ),
    Case(
        name="energy_disturbance",
        scenario="multiple_problems",
        arms=("heuristic", "rl", "rl+supervisor"),
        max_switches=2,
        expect_retune=True,
        allow_safe_mode=True,
    ),
    Case(
        name="recovery",
        scenario="cloud_cover",
        arms=("heuristic", "rl", "rl+supervisor"),
        max_switches=2,
        expect_recovery=True,
        allow_safe_mode=True,
    ),
)


@dataclass
class Run:
    """What one arm did on one scenario. The comparable figures, and nothing else."""

    case: str
    arm: str
    ch4_kg: float = 0.0
    critical_events: int = 0
    shutdowns: int = 0
    safety_corrections: int = 0
    switches: int = 0
    switch_log: list[str] = field(default_factory=list)
    supervisor_runs: int = 0
    retunes: list[str] = field(default_factory=list)
    final_controller: str = "heuristic"
    operating_state: str = "normal"
    plant_status: str = "normal"
    min_battery_soc: float = 1.0
    grade: str = "unrated"

    @property
    def recovered(self) -> bool:
        """Running normally at the end, under a controller that is not the safe one."""
        return (
            self.plant_status == "normal"
            and self.operating_state == "normal"
            and self.final_controller != SAFE_CONTROLLER_ID
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "arm": self.arm,
            "ch4_kg": round(self.ch4_kg, 2),
            "critical_events": self.critical_events,
            "shutdowns": self.shutdowns,
            "safety_corrections": self.safety_corrections,
            "switches": self.switches,
            "switch_log": list(self.switch_log),
            "supervisor_runs": self.supervisor_runs,
            "retunes": list(self.retunes),
            "final_controller": self.final_controller,
            "operating_state": self.operating_state,
            "plant_status": self.plant_status,
            "min_battery_soc": round(self.min_battery_soc, 3),
            "grade": self.grade,
            "recovered": self.recovered,
        }


@dataclass
class Outcome:
    """One case: every arm's figures, plus where the supervised arm fell short."""

    case: str
    runs: list[Run] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def supervised(self) -> Run | None:
        return next((run for run in self.runs if ARMS[run.arm].supervised), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "runs": [run.to_dict() for run in self.runs],
            "passed": self.passed,
            "failures": list(self.failures),
        }


async def run_arm(
    case: Case,
    arm: Arm,
    *,
    checkpoint: Path | None = None,
    engine: Engine | None = None,
    hours: float = DEFAULT_HOURS,
    batch_min: int = BATCH_MIN,
) -> Run:
    """Drive one scenario with one arm, exactly as the backend would drive it."""
    runtime = PlantRuntime()
    supervisor = SupervisorAgent(runtime, engine or RuleEngine(), enabled=arm.supervised)
    runtime.supervisor = supervisor
    runtime.load_scenario(case.scenario)
    if arm.policy is not None:
        if checkpoint is None:
            raise ValueError(f"arm {arm.id} needs a checkpoint")
        runtime.load_rl_checkpoint(str(checkpoint))
    runtime.set_controller(arm.controller, reason=f"evaluation arm {arm.id}")
    # Selecting the arm's controller is the run's starting condition, not a
    # switch the run made: only handovers decided during the run are counted.
    runtime.control.switches.clear()

    run = Run(case=case.name, arm=arm.id)
    for _ in range(int(hours * 60 / batch_min)):
        frames = runtime.advance(batch_min)
        run.min_battery_soc = min(
            run.min_battery_soc, *(frame["battery_soc"] for frame in frames)
        )
        if not arm.supervised:
            continue
        trigger = supervisor.notice(frames)
        if trigger is not None:
            await supervisor.investigate(trigger)

    return _figures(run, runtime, supervisor)


def _figures(run: Run, runtime: PlantRuntime, supervisor: SupervisorAgent) -> Run:
    """Read the run off the plant and the manager. Totals, not windows."""
    control = runtime.control
    telemetry = runtime.session.telemetry
    totals = {"critical_events": 0, "forced_shutdowns": 0, "safety_corrections": 0}
    for health in control.health.values():
        for key in totals:
            totals[key] += health.totals[key]

    run.ch4_kg = telemetry.ch4_produced_total_kg
    run.critical_events = totals["critical_events"]
    run.shutdowns = totals["forced_shutdowns"]
    run.safety_corrections = totals["safety_corrections"]
    run.switches = len(control.switches)
    run.switch_log = [
        f"{switch.clock} {switch.summary} ({switch.by}: {switch.reason})"
        for switch in control.switches
    ]
    run.supervisor_runs = supervisor.runs
    run.retunes = [
        action for incident in supervisor.trace.incidents() for action in incident["actions"]
    ]
    run.final_controller = control.active_id
    run.operating_state = control.operating_state
    run.plant_status = telemetry.plant_status
    run.grade = control.active_health.grade
    return run


async def run_case(
    case: Case,
    *,
    checkpoints_by_arm: dict[str, Path] | None = None,
    engine: Engine | None = None,
    hours: float = DEFAULT_HOURS,
) -> Outcome:
    """Every arm of one case, then the V8 promises checked against the supervised arm."""
    outcome = Outcome(case=case.name)
    for arm_id in case.arms:
        outcome.runs.append(
            await run_arm(
                case,
                ARMS[arm_id],
                checkpoint=(checkpoints_by_arm or {}).get(arm_id),
                engine=engine,
                hours=hours,
            )
        )
    outcome.failures = _score(case, outcome)
    return outcome


@contextmanager
def fixed_checkpoints() -> Iterator[dict[str, Path]]:
    """A throwaway `fixed` checkpoint per RL arm, for the length of a suite."""
    with tempfile.TemporaryDirectory(prefix="v8-eval-") as directory:
        yield {
            arm.id: checkpoints.write_fixed_checkpoint(Path(directory) / arm.id, arm.policy)
            for arm in ARMS.values()
            if arm.policy is not None
        }


async def run_case_with_checkpoints(
    case: Case, *, engine: Engine | None = None, hours: float = DEFAULT_HOURS
) -> Outcome:
    """`run_case`, with the checkpoints its RL arms need written and cleaned up."""
    with fixed_checkpoints() as checkpoints_by_arm:
        return await run_case(
            case, checkpoints_by_arm=checkpoints_by_arm, engine=engine, hours=hours
        )


def _score(case: Case, outcome: Outcome) -> list[str]:
    """Where the supervised arm disagreed with what V8 promises. Empty is a pass."""
    run = outcome.supervised
    if run is None:
        return ["no supervised arm in this case"]
    failures: list[str] = []
    in_safe_mode = run.final_controller == SAFE_CONTROLLER_ID or run.operating_state == "safe"

    if case.max_switches is not None and run.switches > case.max_switches:
        failures.append(
            f"{run.switches} controller switches, expected at most {case.max_switches}: "
            f"{run.switch_log}"
        )
    if run.switches < case.min_switches:
        failures.append(f"{run.switches} controller switches, expected at least {case.min_switches}")
    if case.ends_with is not None and run.final_controller != case.ends_with:
        failures.append(f"ended under {run.final_controller}, expected {case.ends_with}")
    if case.expect_retune is True and not run.retunes:
        failures.append("the envelope was never retuned")
    if case.expect_retune is False and run.retunes:
        failures.append(f"retuned a healthy plant: {run.retunes}")
    if case.expect_recovery and not run.recovered:
        failures.append(
            f"did not return to normal running: plant {run.plant_status}, "
            f"{run.operating_state} under {run.final_controller}"
        )
    if in_safe_mode and not case.allow_safe_mode:
        failures.append("ended in safe mode, which this case does not call for")
    return failures


async def run_suite(
    cases: tuple[Case, ...] = CASES,
    *,
    engine: Engine | None = None,
    hours: float = DEFAULT_HOURS,
) -> list[Outcome]:
    """Every case, sequentially: one plant and one supervisor at a time."""
    with fixed_checkpoints() as checkpoints_by_arm:
        return [
            await run_case(
                case, checkpoints_by_arm=checkpoints_by_arm, engine=engine, hours=hours
            )
            for case in cases
        ]


def report(outcomes: list[Outcome]) -> str:
    """The suite as a table: one row per arm, one block per case."""
    header = (
        f"{'case':22} {'arm':20} {'CH4 kg':>8} {'crit':>5} {'shut':>5} "
        f"{'corr':>5} {'sw':>3} {'grade':>8}  end"
    )
    lines = [header, "-" * len(header)]
    for outcome in outcomes:
        for run in outcome.runs:
            lines.append(
                f"{outcome.case:22} {run.arm:20} {run.ch4_kg:>8.1f} {run.critical_events:>5} "
                f"{run.shutdowns:>5} {run.safety_corrections:>5} {run.switches:>3} "
                f"{run.grade:>8}  {run.final_controller}/{run.operating_state}"
            )
        lines.append(f"{'':22} {'-> ' + ('pass' if outcome.passed else 'FAIL'):20}")
        lines.extend(f"{'':22}    - {failure}" for failure in outcome.failures)
    passed = sum(1 for outcome in outcomes if outcome.passed)
    lines.append("-" * len(header))
    lines.append(f"{passed}/{len(outcomes)} cases passed")
    return "\n".join(lines)


def main() -> int:
    outcomes = asyncio.run(run_suite())
    print(report(outcomes))
    return 0 if all(outcome.passed for outcome in outcomes) else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "ARMS",
    "CASES",
    "Arm",
    "Case",
    "Outcome",
    "Run",
    "fixed_checkpoints",
    "main",
    "report",
    "run_arm",
    "run_case",
    "run_case_with_checkpoints",
    "run_suite",
]

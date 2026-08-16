"""V10 behaviour: a forecast reaches the supervisor, and never reaches the plant.

Every prediction in this file is a hand-written fixture (`hepa.fixtures`), not a
model output: no checkpoint, no torch, no training. What is under test is the path
between the two -- level, trend, cadence, trigger, investigation, intervention --
and the three ways it is allowed to end when HEPA cannot answer: unavailable,
failed, and simply not due yet.

The property defended throughout is the hierarchy. A probability may make the
supervisor look; only the supervisor moves the envelope, and only through the
controller manager it already had in V8.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from plant_backend.agent import AgentToolbox, AgentTrace, RuleEngine, SupervisorAgent
from plant_backend.agent import foresight_evaluation as foresight
from plant_backend.agent import risk as risk_module
from plant_backend.agent.risk import RiskMonitor, RiskPolicy, Thresholds
from plant_backend.hepa import fixtures, service
from plant_backend.orchestration import PlantRuntime

#: Enough steps that the window rules have a few hours of history to read.
MORNING_STEPS = 420

#: An hour of capture outage: the CO2 buffer is falling but not yet empty, which is
#: exactly the state a forecast is useful in.
CO2_DRAIN_STEPS = 60
#: Two hours of thin generation: the pack is above its reserve but working for it.
SOLAR_SHORTFALL_STEPS = 120

#: A forecast nobody needs to act on, and one nobody can ignore.
CALM = {"co2_starvation": {"1h": 0.01, "3h": 0.02, "6h": 0.04, "12h": 0.06}}
CO2_RISING = (
    {"co2_starvation": {"1h": 0.02, "3h": 0.10, "6h": 0.18, "12h": 0.40}},
    {"co2_starvation": {"1h": 0.04, "3h": 0.22, "6h": 0.52, "12h": 0.76}},
    {"co2_starvation": {"1h": 0.03, "3h": 0.12, "6h": 0.30, "12h": 0.48}},
)
BATTERY_HIGH = {"battery_reserve_violation": {"1h": 0.05, "6h": 0.44, "12h": 0.72}}


def plant(forecaster: service.Forecaster | None = None) -> PlantRuntime:
    """A plant with a rule-driven supervisor and, optionally, a fixture forecast."""
    runtime = PlantRuntime()
    runtime.supervisor = SupervisorAgent(
        runtime, RuleEngine(), enabled=True, forecaster=forecaster
    )
    runtime.advance(MORNING_STEPS)
    return runtime


def investigate(runtime: PlantRuntime) -> object:
    return asyncio.run(runtime.supervisor.investigate())


def monitor(runtime: PlantRuntime) -> RiskMonitor:
    return runtime.supervisor.risk


# --- what a level and a trend are ----------------------------------------
def test_levels_come_from_configurable_thresholds_per_event_and_horizon() -> None:
    policy = RiskPolicy(
        events={"co2_starvation": Thresholds(elevated=0.35, high=0.60, critical=0.85)},
        per_horizon={"co2_starvation": {"1h": Thresholds(elevated=0.20, high=0.40, critical=0.65)}},
    )

    assert policy.level("co2_starvation", "12h", 0.50) == risk_module.ELEVATED
    assert policy.level("co2_starvation", "12h", 0.61) == risk_module.HIGH
    assert policy.level("co2_starvation", "12h", 0.90) == risk_module.CRITICAL
    # The same number is worse an hour out than half a day out.
    assert policy.level("co2_starvation", "1h", 0.50) == risk_module.HIGH


def test_the_shipped_risk_config_parses_into_the_shipped_defaults() -> None:
    """`configs/risk.json` documents the defaults, so it has to reproduce them."""
    path = Path(risk_module.__file__).with_name("configs") / "risk.json"
    policy = RiskPolicy.from_dict(json.loads(path.read_text()))

    assert policy.horizons == risk_module.DEFAULT_HORIZONS
    assert policy.refresh_s == risk_module.REFRESH_S
    assert policy.events["co2_starvation"].high == pytest.approx(0.60)
    assert policy.thresholds("co2_starvation", "1h").high == pytest.approx(0.40)


def test_a_broken_config_is_ignored_rather_than_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(risk_module.CONFIG_ENV, "/nonexistent/risk.json")
    monkeypatch.setenv(risk_module.REFRESH_ENV, "not-a-number")

    policy = RiskPolicy.from_env()

    assert policy.refresh_s == risk_module.REFRESH_S
    assert policy.horizons == risk_module.DEFAULT_HORIZONS


def test_a_trend_is_this_forecast_against_the_previous_one() -> None:
    runtime = plant(fixtures.ScriptedForecaster(list(CO2_RISING)))
    risk = monitor(runtime)

    risk.update(force=True)
    first = risk.risks["co2_starvation"]["12h"]
    risk.update(force=True)
    second = risk.risks["co2_starvation"]["12h"]
    risk.update(force=True)
    third = risk.risks["co2_starvation"]["12h"]

    assert first.trend == risk_module.STABLE  # nothing to compare against yet
    assert (second.trend, second.previous) == (risk_module.RISING, first.probability)
    assert third.trend == risk_module.FALLING
    assert [entry["kind"] for entry in runtime.supervisor.trace.recent(20)].count("prediction") >= 2


def test_the_model_is_not_asked_every_timestep() -> None:
    """The cadence, not the caller, decides when HEPA is asked."""
    forecaster = fixtures.ScriptedForecaster([CALM])
    runtime = plant(forecaster)
    risk = monitor(runtime)

    risk.update()
    calls = forecaster.calls
    for _ in range(5):
        runtime.advance(1)
        assert risk.update() is None  # 60 s of plant time, 900 s of cadence

    assert forecaster.calls == calls
    runtime.advance(int(risk.policy.refresh_s / 60) + 1)
    assert risk.update() is not None
    assert forecaster.calls == calls + 1


# --- what the tool answers ------------------------------------------------
def test_the_tool_answers_semantic_probabilities_and_nothing_else() -> None:
    runtime = plant(fixtures.FixedForecaster(BATTERY_HIGH))
    trace = AgentTrace()
    toolbox = AgentToolbox(runtime, trace, monitor(runtime))

    answer = toolbox.get_predicted_event_risk([1, 6, 12])

    assert answer["status"] == "active"
    assert answer["probabilities"]["battery_reserve_violation"]["12h"] == pytest.approx(0.72)
    assert answer["levels"]["battery_reserve_violation"]["12h"] == risk_module.HIGH
    assert set(answer["probabilities"]["battery_reserve_violation"]) == {"1h", "6h", "12h"}
    assert "tensor" not in repr(answer) and "checkpoint" not in answer
    logged = [entry for entry in trace.recent(10) if entry["kind"] == "tool"]
    assert logged and logged[-1]["detail"]["tool"] == "get_predicted_event_risk"
    assert toolbox.actions == []  # a read, and only a read


def test_the_tool_says_unavailable_rather_than_inventing_numbers() -> None:
    runtime = plant(fixtures.UnavailableForecaster())
    toolbox = AgentToolbox(runtime, AgentTrace(), monitor(runtime))

    answer = toolbox.get_predicted_event_risk()

    assert answer["status"] == "unavailable"
    assert answer["probabilities"] == {}
    assert answer["attention"] == []
    assert answer["reason"]


# --- low risk: no intervention -------------------------------------------
def test_a_calm_forecast_raises_no_trigger_and_changes_nothing() -> None:
    runtime = plant(fixtures.FixedForecaster(CALM))
    before = runtime.constraints.to_dict()

    risk = monitor(runtime)
    risk.update(force=True)

    assert risk.available and risk.concerns() == []
    assert risk.trigger() is None
    assert runtime.supervisor.notice(runtime.session.recent(10)) is None
    assert runtime.constraints.to_dict() == before


# --- rising CO2 risk: investigate before the buffer empties ---------------
def test_rising_co2_risk_wakes_the_supervisor_before_anything_has_stopped() -> None:
    runtime = plant(fixtures.ScriptedForecaster([CO2_RISING[1]]))
    risk = monitor(runtime)
    risk.update(force=True)

    trigger = risk.trigger()

    assert trigger is not None
    assert trigger.reason == "predicted_risk"
    assert trigger.key == "risk:co2_starvation"
    assert "co2 starvation" in trigger.detail.lower()
    assert runtime.session.telemetry.co2_level_kg > 1.0  # not starved: this is foresight


def test_a_credible_co2_forecast_tightens_methanation_while_capture_recovers() -> None:
    runtime = plant(fixtures.FixedForecaster(CO2_RISING[1]))
    runtime.session.inject_fault({"type": "dac_outage", "severity": 1.0})
    runtime.advance(CO2_DRAIN_STEPS)
    monitor(runtime).update(force=True)
    before = runtime.constraints.methanator_limit_kw(runtime.config)

    incident = investigate(runtime)

    assert incident.category == "predicted_risk"
    assert incident.component == "dac"
    assert runtime.constraints.methanator_limit_kw(runtime.config) < before
    trace = " ".join(entry["message"] for entry in runtime.supervisor.trace.recent(40))
    assert "HEPA" in trace and "co2" in trace.lower()


def test_an_uncorroborated_forecast_is_logged_and_not_acted_on() -> None:
    """The buffer is full and capture is fine: the prediction is noted, not obeyed."""
    runtime = plant(fixtures.FixedForecaster(CO2_RISING[1]))
    monitor(runtime).update(force=True)

    incident = investigate(runtime)

    assert incident.category != "predicted_risk"
    trace = " ".join(entry["message"] for entry in runtime.supervisor.trace.recent(40))
    assert "not corroborated" in trace


# --- high battery risk: shed early ---------------------------------------
def test_a_credible_reserve_forecast_sheds_discretionary_load_early() -> None:
    runtime = plant(fixtures.FixedForecaster(BATTERY_HIGH))
    runtime.session.inject_fault({"type": "solar_drop", "severity": 0.3})
    runtime.advance(SOLAR_SHORTFALL_STEPS)
    monitor(runtime).update(force=True)
    soc = runtime.session.telemetry.battery_soc
    reserve = runtime.constraints.reserve_soc(runtime.config)
    before = runtime.constraints.dac_limit_kw(runtime.config)

    incident = investigate(runtime)

    assert soc > reserve  # the pack still looks healthy
    assert incident.category == "predicted_risk"
    assert incident.component == "battery"
    assert runtime.constraints.dac_limit_kw(runtime.config) < before


# --- the risk falls again ------------------------------------------------
def test_a_falling_risk_after_an_intervention_is_traced_as_such() -> None:
    runtime = plant(fixtures.ScriptedForecaster(list(CO2_RISING)))
    runtime.session.inject_fault({"type": "dac_outage", "severity": 1.0})
    runtime.advance(CO2_DRAIN_STEPS)
    risk = monitor(runtime)
    risk.update(force=True)  # the rising forecast
    risk.update(force=True)  # the concerning one
    investigate(runtime)
    risk.update(force=True)  # and after the intervention

    entry = risk.risks["co2_starvation"]["12h"]

    assert entry.trend == risk_module.FALLING
    assert entry.delta is not None and entry.delta < 0
    predictions = [
        item for item in runtime.supervisor.trace.recent(60) if item["kind"] == "prediction"
    ]
    assert any("decreased to" in item["message"] for item in predictions)


# --- without a model ------------------------------------------------------
def test_with_no_checkpoint_the_plant_runs_exactly_as_it_did_in_v8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PLANT_HEPA_CHECKPOINT", raising=False)
    service.reset_cache()
    runtime = plant()  # the real CheckpointForecaster, with nothing to load
    risk = monitor(runtime)

    forecast = risk.update(force=True)
    runtime.advance(60)
    incident = investigate(runtime)

    assert forecast is not None and forecast.status == service.UNAVAILABLE
    assert risk.risks == {} and risk.trigger() is None
    assert risk.snapshot()["available"] is False
    assert incident.summary  # the supervisor still investigated, on telemetry alone
    assert runtime.payload()["agent"]["risk"]["status"] == service.UNAVAILABLE


def test_a_forecaster_that_raises_becomes_a_status_not_an_outage() -> None:
    runtime = plant(fixtures.BrokenForecaster())
    risk = monitor(runtime)

    forecast = risk.update(force=True)
    runtime.advance(60)
    incident = investigate(runtime)

    assert forecast is not None and forecast.status == service.ERROR
    assert forecast.reason is not None and "fixture inference failure" in forecast.reason
    assert risk.risks == {} and risk.trigger() is None
    assert incident.summary
    assert runtime.supervisor.last_error is None  # the supervisor never saw a failure


def test_a_broken_forecast_does_not_stop_the_clock() -> None:
    runtime = plant(fixtures.BrokenForecaster())
    frames = runtime.advance(30)

    assert len(frames) == 30
    assert runtime.session.telemetry.plant_status in {"normal", "degraded", "critical"}


# --- the hierarchy --------------------------------------------------------
def test_a_forecast_alone_never_moves_the_plant() -> None:
    """HEPA predicts. It does not cap, switch, stop or set anything."""
    runtime = plant(fixtures.FixedForecaster(BATTERY_HIGH))
    risk = monitor(runtime)
    before = (
        runtime.constraints.to_dict(),
        runtime.controller_id,
        runtime.control.operating_state,
        runtime.session.telemetry.step,
    )

    for _ in range(3):
        risk.update(force=True)

    assert risk.concerns()  # it had plenty to say
    assert (
        runtime.constraints.to_dict(),
        runtime.controller_id,
        runtime.control.operating_state,
        runtime.session.telemetry.step,
    ) == before


def test_fixture_forecasts_are_labelled_all_the_way_to_the_control_room() -> None:
    runtime = plant(fixtures.FixedForecaster(CALM))
    monitor(runtime).update(force=True)

    snapshot = runtime.payload()["agent"]["risk"]

    assert snapshot["source"] == f"{fixtures.SOURCE_PREFIX}:fixed"
    assert snapshot["status"] == service.ACTIVE


# --- the A/B/C harness ----------------------------------------------------
def test_the_foresight_harness_runs_every_arm_and_labels_the_fixture() -> None:
    """A/B/C on one short case: same plant, three levels of supervision."""
    case = foresight.ForesightCase(
        scenario="dac_failure",
        event="co2_starvation",
        script=foresight.CO2_SCRIPT,
        hours=2.0,
    )
    outcomes = asyncio.run(foresight.run_case(case))

    assert [outcome.arm for outcome in outcomes] == list(foresight.ARMS)
    controller, supervised, foresighted = outcomes
    # arm A is unsupervised: the envelope is never touched
    assert controller.runs == 0 and controller.acted == 0
    # arm B is the V8 plant: supervision, but nothing to be early about
    assert supervised.source == "none" and supervised.first_warning_s is None
    # arm C saw the fixture, and says so
    assert foresighted.source.startswith(fixtures.SOURCE_PREFIX)
    assert foresighted.first_warning_s is not None
    assert "not evidence" in foresight.report(outcomes)


def test_the_harness_counts_nothing_unnecessary_on_a_healthy_plant() -> None:
    case = foresight.ForesightCase(
        scenario="clear_day",
        event="co2_starvation",
        script=foresight.CALM_SCRIPT,
        hours=2.0,
        at_risk=False,
    )
    outcome = asyncio.run(foresight.run_arm(case, foresight.SUPERVISOR_HEPA))

    assert outcome.events == 0
    assert outcome.unnecessary == 0
    assert outcome.acted == 0

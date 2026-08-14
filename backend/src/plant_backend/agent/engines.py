"""Two ways to be the supervisor: an LLM, or a rule book.

Both implement the same tiny contract -- given a toolbox and a brief, investigate
and return a `Diagnosis` -- and both act only through the toolbox, so the plant
cannot tell them apart:

    LLMEngine    OpenAI Agents SDK. The real thing: it chooses which tools to
                 call, in what order, and what to conclude.
    RuleEngine   No model, no network, no cost. It walks the same tools in a
                 fixed order and applies explicit thresholds.

`RuleEngine` is not a toy: it is the fallback when the LLM is unavailable or
fails, and it is what the regression harness runs so the evaluation stays
deterministic. The plant must keep running when the model does not, and the
cheapest way to guarantee that is to have the non-model path be a first-class
implementation of the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol

from .prompts import SYSTEM_PROMPT, brief as build_brief
from .tools import AgentToolbox
from .trace import Confidence

#: What the supervisor thinks it is looking at. `normal` means "nothing wrong".
CATEGORIES = (
    "normal",
    #: V10: a credible HEPA forecast, acted on before the event.
    "predicted_risk",
    "controller_degradation",
    "critical_condition",
    "electrolyser_degradation",
    "dac_unavailable",
    "dac_degradation",
    "methanator_unavailable",
    "energy_shortfall",
    "battery_low",
    "feedstock_shortage",
    "sensor_inconsistency",
    "unclear",
)

#: Model used when none is configured.
DEFAULT_MODEL = "gpt-4.1-mini"
#: Tool-calling turns one LLM run may take.
MAX_TURNS = 12

# --- rule-engine thresholds -----------------------------------------------
#: Observed yield below this fraction of design counts as a derated unit.
YIELD_SUSPECT = 0.9
#: ... and below this it is not a subtle problem.
YIELD_SEVERE = 0.75
#: Generation below this fraction of clear-sky counts as an energy problem.
SOLAR_SUSPECT = 0.8
#: Tank fill above which throttling, not degradation, explains a low yield.
TANK_BUSY_FILL = 0.98
#: SOC margin above the reserve that still counts as "running on the reserve".
RESERVE_MARGIN = 0.05
#: Battery reserve the supervisor holds when energy is short.
SHORTFALL_RESERVE_SOC = 0.3
#: Discretionary load left during an energy shortfall, as a fraction of rating.
SHORTFALL_ELECTROLYSER_FRAC = 0.5
SHORTFALL_DAC_FRAC = 0.4
#: Methanation left when CO2 supply is gone, as a fraction of rating.
CO2_STARVED_METHANATOR_FRAC = 0.4
#: Methanation left when CO2 starvation is *predicted* but not yet happening: a
#: forecast buys a gentler change than a fault does.
CO2_RISK_METHANATOR_FRAC = 0.6
#: Fall in mean CO2 inventory over the window that counts as "heading down".
CO2_FALL_KG = 1.0
#: Fall in mean SOC over the window that counts as the pack losing ground.
SOC_FALL = 0.02
#: Plant conditions that call for safe mode rather than a retuned envelope.
CRITICAL_STATUSES = ("critical", "offline")
#: SOC above the reserve at which there is still an energy budget to retune.
CRITICAL_SOC_MARGIN = 0.03
#: The controller the plant falls back to when a driver is graded poor.
FALLBACK_CONTROLLER = "heuristic"


@dataclass(frozen=True)
class Diagnosis:
    """The supervisor's conclusion about one incident."""

    category: str
    summary: str
    component: str | None = None
    confidence: Confidence = "low"
    human_intervention_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "summary": self.summary,
            "component": self.component,
            "confidence": self.confidence,
            "human_intervention_required": self.human_intervention_required,
        }


class Engine(Protocol):
    """Anything that can supervise the plant through a toolbox."""

    label: str

    async def investigate(self, toolbox: AgentToolbox, trigger: dict[str, Any]) -> Diagnosis: ...


class RuleEngine:
    """The deterministic supervisor: the same tools, walked in a fixed order.

    The order is the diagnostic one, not the alphabetical one. A critical plant
    condition comes first (protect first, explain afterwards), then the driver
    itself (a controller misbehaving invalidates conclusions about the plant it is
    driving), then instruments before the process (a bad reading invalidates
    everything downstream), then the units that convert power into product, then
    the energy budget, then the feedstock chain.

    V10 inserts one rule into that order, after the instrument check and before the
    component rules: a HEPA forecast that the telemetry corroborates. It sits there
    because the point of a forecast is to act while the plant still looks fine --
    the component rules below all describe something that has already gone wrong --
    and behind the instrument check because a forecast built on a lying sensor is
    worth exactly as much as the sensor.
    """

    label = "rules"

    async def investigate(self, toolbox: AgentToolbox, trigger: dict[str, Any]) -> Diagnosis:
        telemetry = toolbox.get_current_telemetry()
        toolbox.get_recent_events(20)
        trend = toolbox.get_recent_telemetry(120)
        constraints = toolbox.get_controller_constraints()
        diagnostic = toolbox.run_diagnostic("plant")
        # V10: what may happen, before the rules about what already has. It is only
        # ever evidence: `_predicted_risk` acts on a forecast the telemetry agrees
        # with, and otherwise stands aside for the present-tense rules below.
        risk = toolbox.get_predicted_event_risk()

        for rule in (
            _critical_condition,
            _controller_behaviour,
            _recovery,
            _sensor_inconsistency,
            partial(_predicted_risk, risk=risk, trend=trend),
            _electrolyser,
            _dac,
            _methanator,
            _solar,
            _battery,
            _feedstock,
        ):
            diagnosis = rule(toolbox, telemetry, constraints, diagnostic)
            if diagnosis is not None:
                return diagnosis

        if telemetry["plant_status"] == "normal":
            return _normal(toolbox)
        return Diagnosis(
            category="unclear",
            summary=(
                f"Plant status is {telemetry['plant_status']} but no single component explains it "
                "from the available telemetry. Envelope left unchanged."
            ),
            confidence="low",
            human_intervention_required=True,
        )


def _critical_condition(
    toolbox: AgentToolbox,
    telemetry: dict[str, Any],
    constraints: dict[str, Any],
    diagnostic: dict[str, Any],
) -> Diagnosis | None:
    """A critical plant with no energy left to work with gets the safe controller.

    Both halves of that matter. A critical status on its own usually has a
    component behind it, and that component wants its cap moved, not the whole
    plant put to bed -- V6's component diagnoses are still the right answer there.
    But a critical plant sitting on its battery reserve has no budget left to
    retune: whatever the envelope permits, there is no power to spend inside it.
    Then the safe controller is the right driver, because it stops the process
    loads, recharges the pack and works off the feedstock already in the tanks
    while a human looks at the plant.
    """
    if telemetry["plant_status"] not in CRITICAL_STATUSES:
        return None
    reserve_soc = float(constraints["battery_reserve_soc"])
    if float(telemetry["battery_soc"]) > reserve_soc + CRITICAL_SOC_MARGIN:
        return None
    control = toolbox.get_active_controller()
    if control["operating_state"] == "safe":
        return Diagnosis(
            category="critical_condition",
            summary=(
                f"Plant status is {telemetry['plant_status']} and the plant is already in safe "
                "mode under the safe/degraded controller. Left there; the condition needs a human, "
                "not another supervisory change."
            ),
            confidence="high",
            human_intervention_required=True,
        )
    reason = f"plant status {telemetry['plant_status']}: protecting the plant"
    result = toolbox.set_operating_state("safe", reason=reason)
    entered = bool(result.get("ok"))
    return Diagnosis(
        category="critical_condition",
        summary=(
            f"Plant status is {telemetry['plant_status']} while {control['active']['label']} was "
            "driving. "
            + (
                "Moved to safe mode: process loads stop, the battery recharges and only existing "
                "feedstock is consumed."
                if entered
                else f"Safe mode was refused ({result.get('error')}), so the envelope is unchanged "
                "and the condition needs a human now."
            )
        ),
        confidence="high",
        human_intervention_required=True,
    )


def _controller_behaviour(
    toolbox: AgentToolbox,
    telemetry: dict[str, Any],
    constraints: dict[str, Any],
    diagnostic: dict[str, Any],
) -> Diagnosis | None:
    """Is the driver the problem? Only a `poor` grade is grounds for a handover.

    Anything short of that -- `suspect`, or a poor grade on the fallback
    controller, which has nowhere safer to hand the plant -- is written into the
    shift log and then left to the plant rules below. Taking a controller off the
    plant for a bad half-hour is how a supervisor turns one incident into an
    oscillation, and a genuinely faulty component still wants its cap moved.
    """
    metrics = toolbox.get_controller_metrics()
    grade = metrics["grade"]
    if grade not in {"poor", "suspect"}:
        return None
    control = toolbox.get_active_controller()
    active = control["active"]["id"]
    concerns = metrics["concerns"]
    evidence = "; ".join(concerns[:2]) or "repeated corrections"
    toolbox.observe(f"{active} graded {grade}: {evidence}", controller=active, grade=grade)
    if grade == "suspect" or active == FALLBACK_CONTROLLER:
        return None
    result = toolbox.switch_controller(
        FALLBACK_CONTROLLER, reason=f"{active} graded poor: {evidence}"
    )
    switched = bool(result.get("ok")) and result.get("switched")
    return Diagnosis(
        category="controller_degradation",
        component=None,
        summary=(
            f"{control['active']['label']} is driving badly: {evidence}. "
            + (
                "Handed the plant to the heuristic controller, which keeps running from the same "
                "state under the same envelope."
                if switched
                else f"The handover was refused ({result.get('error')}), so it keeps driving and "
                "will be reassessed."
            )
        ),
        confidence="high",
        human_intervention_required=not switched,
    )


def _recovery(
    toolbox: AgentToolbox,
    telemetry: dict[str, Any],
    constraints: dict[str, Any],
    diagnostic: dict[str, Any],
) -> Diagnosis | None:
    """Coming back out of safe mode, before the component rules get a say.

    Safe mode stops the electrolyser and the DAC, so the plant it produces looks
    starved of feedstock for as long as it lasts. Read from below, that is an
    unresolved feedstock fault and the supervisor would keep re-diagnosing its own
    intervention forever. So a healthy plant under a healthy driver leaves safe
    mode here, on the strength of the plant status the safe controller was put
    there to restore -- not on inventories the safe controller is not refilling.
    """
    if telemetry["plant_status"] != "normal":
        return None
    if toolbox.get_active_controller()["operating_state"] == "normal":
        return None
    if toolbox.get_controller_metrics()["grade"] == "poor":
        return None
    return _normal(toolbox)


def _normal(toolbox: AgentToolbox) -> Diagnosis:
    """Nothing is wrong. Which may itself be news, if the plant is still in safe mode.

    Coming back is a one-way step taken once, and only from a plant that reports
    normal with a driver that is not misbehaving -- both already established by
    the time this runs.
    """
    control = toolbox.get_active_controller()
    if control["operating_state"] == "normal":
        return Diagnosis(
            category="normal",
            summary=(
                "Telemetry, inventories and component performance are all consistent with "
                f"normal operation under {control['active']['label']}. No intervention taken."
            ),
            confidence="high",
        )
    result = toolbox.set_operating_state(
        "normal", reason="plant back to normal status with a healthy controller"
    )
    if not result.get("ok"):
        return Diagnosis(
            category="normal",
            summary=(
                f"The plant reports normal operation but leaving {control['operating_state']} mode "
                f"was refused ({result.get('error')}). Left as it is."
            ),
            confidence="medium",
        )
    return Diagnosis(
        category="normal",
        summary=(
            f"The condition that put the plant into {control['operating_state']} mode has cleared: "
            "telemetry, inventories and component performance are all consistent with normal "
            f"operation. Returned to normal running under {result['controller']}."
        ),
        confidence="high",
    )


def _tighten(
    toolbox: AgentToolbox,
    changes: dict[str, float],
    current: dict[str, float],
    *,
    reason: str,
) -> dict[str, Any]:
    """Apply the changes that are genuine tightenings; report what took effect.

    A summary claiming a cap the envelope already had is a lie an operator would
    act on, so every rule below writes its prose from this return value rather
    than from what it meant to do.
    """
    tightening = {
        key: value for key, value in changes.items() if key not in current or value < current[key]
    }
    if not tightening:
        return {}
    result = toolbox.set_controller_constraints(dict(tightening), reason=reason)
    return dict(result.get("applied") or {}) if result.get("ok") else {}


#: How to say each envelope field out loud, for summaries.
_PHRASES = {
    "battery_min_soc": lambda value: f"battery reserve raised to {float(value):.0%}",
    "electrolyser_max_kw": lambda value: f"stack capped at {float(value):.0f} kW",
    "dac_max_kw": lambda value: f"capture capped at {float(value):.0f} kW",
    "methanator_max_kw": lambda value: f"methanation capped at {float(value):.0f} kW",
    "battery_support_max_kw": lambda value: f"battery support capped at {float(value):.0f} kW",
}


def _changed(applied: dict[str, Any]) -> str:
    """Name what actually moved, in the words a shift log would use."""
    parts = [_PHRASES[field](value) for field, value in applied.items() if field in _PHRASES]
    if not parts:
        return "the envelope was retuned"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _sensor_inconsistency(
    toolbox: AgentToolbox,
    telemetry: dict[str, Any],
    constraints: dict[str, Any],
    diagnostic: dict[str, Any],
) -> Diagnosis | None:
    """A level that moved without a flow to explain it: suspect the transmitter."""
    findings = diagnostic["mass_balance"]["inconsistent"]
    if not findings:
        return None
    worst = max(findings, key=lambda finding: abs(finding["residual_kg"]))
    toolbox.observe(
        f"{worst['channel']} and the flow readings disagree by "
        f"{abs(worst['residual_kg']):.1f} kg at {worst['clock']}",
        channel=worst["channel"],
        residual_kg=worst["residual_kg"],
    )
    return Diagnosis(
        category="sensor_inconsistency",
        component=None,
        summary=(
            f"{worst['channel']} reported a {worst['reported_change_kg']} kg change at "
            f"{worst['clock']} while the flow readings account for {worst['flow_change_kg']} kg. "
            "The observations are internally inconsistent, which points at the instrument rather "
            "than the process; no envelope change made on an unreliable reading."
        ),
        confidence="medium",
        human_intervention_required=True,
    )


def _predicted_risk(
    toolbox: AgentToolbox,
    telemetry: dict[str, Any],
    constraints: dict[str, Any],
    diagnostic: dict[str, Any],
    *,
    risk: dict[str, Any],
    trend: dict[str, Any],
) -> Diagnosis | None:
    """V10: act on a forecast, but only one the plant's own telemetry supports.

    HEPA says *what* may happen, never *why*, so this rule does what a supervisor
    would do with a warning: name the risk, then go and look. Something in the
    telemetry has to corroborate the forecast before the envelope moves -- a
    falling buffer, a unit under-performing, generation short of clear-sky. If
    nothing does, the risk is written into the shift log as a prediction that was
    not confirmed, and the present-tense rules below get their turn.

    Nothing here shuts anything down: a probability is not grounds for stopping a
    plant that is running.
    """
    if risk.get("status") != "active":
        return None
    concerns = risk.get("attention") or []
    if not concerns:
        return None
    worst = concerns[0]
    toolbox.observe(
        f"HEPA: {worst['event'].replace('_', ' ')} within {worst['horizon']} at "
        f"{worst['probability']:.0%} ({worst['level']}, {worst['trend']})",
        event=worst["event"],
        horizon=worst["horizon"],
        probability=worst["probability"],
        level=worst["level"],
        trend=worst["trend"],
    )
    handler = {
        "co2_starvation": _predicted_co2_starvation,
        "battery_reserve_violation": _predicted_battery_reserve,
    }.get(worst["event"])
    if handler is None:
        return None
    return handler(toolbox, telemetry, constraints, trend, worst)


def _predicted_co2_starvation(
    toolbox: AgentToolbox,
    telemetry: dict[str, Any],
    constraints: dict[str, Any],
    trend: dict[str, Any],
    worst: dict[str, Any],
) -> Diagnosis | None:
    """CO2 starvation ahead: is the capture chain actually losing ground?

    The plant can look entirely normal here -- the reactor is running, the buffer
    is not empty -- which is the whole point: by the time CO2 starvation is a fact,
    methanation has already stopped. The evidence that makes the forecast credible
    is capture, not inventory: a unit that is unavailable or converting power at
    well under its design rate, with the buffer draining.
    """
    co2_kg = float(telemetry["co2_level_kg"])
    if co2_kg < 1.0:
        return None  # already starved: the present-tense rules own this
    reading = toolbox.inspect_component("dac")
    ratio = reading.get("yield_ratio")
    draining = _fell(trend, "co2_level_kg", CO2_FALL_KG)
    weak_capture = not reading["available"] or (ratio is not None and ratio < YIELD_SUSPECT)
    if not (draining and weak_capture):
        toolbox.observe(
            "Predicted CO2 starvation not corroborated: capture is at design rate and the "
            f"buffer holds {co2_kg:.0f} kg",
            co2_level_kg=co2_kg,
        )
        return None
    toolbox.observe(
        f"CO2 buffer at {co2_kg:.0f} kg and falling, capture "
        + (
            "unavailable"
            if not reading["available"]
            else f"at {ratio:.0%} of design ({reading['co2_capture_kg_h']:.1f} kg/h)"
        ),
        co2_level_kg=co2_kg,
        capture_kg_h=reading["co2_capture_kg_h"],
    )
    target_kw = round(constraints["rated_kw"]["methanator"] * CO2_RISK_METHANATOR_FRAC, -1)
    applied = _tighten(
        toolbox,
        {"methanator_max_kw": target_kw},
        {"methanator_max_kw": constraints["effective_kw"]["methanator"]},
        reason=(
            f"predicted CO2 starvation within {worst['horizon']} "
            f"({worst['probability']:.0%}); stretch the buffer while capture recovers"
        ),
    )
    return Diagnosis(
        category="predicted_risk",
        component="dac",
        summary=(
            f"CO2 starvation is predicted within {worst['horizon']} at "
            f"{worst['probability']:.0%} and the telemetry agrees: the buffer is down to "
            f"{co2_kg:.0f} kg and falling while capture is "
            + ("unavailable" if not reading["available"] else f"at {ratio:.0%} of design")
            + ". Nothing has stopped yet. "
            + (
                f"Methanation capped at {target_kw:.0f} kW so the remaining CO2 lasts long enough "
                "for capture to catch up."
                if applied
                else "Methanation is already held low enough to stretch the remaining CO2, so the "
                "envelope is unchanged."
            )
        ),
        confidence="medium",
    )


def _predicted_battery_reserve(
    toolbox: AgentToolbox,
    telemetry: dict[str, Any],
    constraints: dict[str, Any],
    trend: dict[str, Any],
    worst: dict[str, Any],
) -> Diagnosis | None:
    """Reserve violation ahead while the pack still looks healthy: shed early.

    Corroboration is generation short of clear-sky, or a pack that is simply
    losing charge over the window. The intervention is the ordinary one -- hold
    more charge back, take the discretionary capture load down -- applied while
    there is still charge to protect, and left to the controller to work inside.
    """
    soc = float(telemetry["battery_soc"])
    reserve_soc = float(constraints["battery_reserve_soc"])
    if soc <= reserve_soc + RESERVE_MARGIN:
        return None  # already on the reserve: `_battery` owns this
    solar = toolbox.inspect_component("solar")
    short_generation = solar["clear_sky_kw"] > 0.0 and solar["output_ratio"] < SOLAR_SUSPECT
    if not (short_generation or _fell(trend, "battery_soc", SOC_FALL)):
        toolbox.observe(
            f"Predicted reserve violation not corroborated: pack at {soc:.0%} and holding, "
            f"generation at {solar['output_ratio']:.0%} of clear-sky",
            battery_soc=soc,
        )
        return None
    toolbox.observe(
        f"Pack at {soc:.0%} against a {reserve_soc:.0%} reserve, generation at "
        f"{solar['output_ratio']:.0%} of clear-sky",
        battery_soc=soc,
        output_ratio=solar["output_ratio"],
    )
    reason = (
        f"predicted reserve violation within {worst['horizon']} "
        f"({worst['probability']:.0%}); protect stored energy early"
    )
    applied = _tighten(
        toolbox,
        {"dac_max_kw": round(constraints["rated_kw"]["dac"] * SHORTFALL_DAC_FRAC, -1)},
        {"dac_max_kw": constraints["effective_kw"]["dac"]},
        reason=reason,
    )
    if reserve_soc < SHORTFALL_RESERVE_SOC:
        result = toolbox.set_controller_constraints(
            {"battery_min_soc": SHORTFALL_RESERVE_SOC}, reason=reason
        )
        if result.get("ok"):
            applied.update(result.get("applied") or {})
    return Diagnosis(
        category="predicted_risk",
        component="battery",
        summary=(
            f"A battery reserve violation is predicted within {worst['horizon']} at "
            f"{worst['probability']:.0%}. The pack is still at {soc:.0%}, above its "
            f"{reserve_soc:.0%} reserve, but generation is at {solar['output_ratio']:.0%} of "
            "clear-sky. "
            + (
                f"{_changed(applied)} while there is still charge to protect."
                if applied
                else "The envelope already holds discretionary load low enough, so it is left "
                "unchanged."
            )
        ),
        confidence="medium",
    )


def _fell(trend: dict[str, Any], channel: str, by: float) -> bool:
    """Whether a channel's mean is at least `by` lower now than earlier in the window."""
    earlier = trend.get("earlier") or {}
    recent = trend.get("recent") or {}
    if channel not in earlier or channel not in recent:
        return False
    return float(earlier[channel]) - float(recent[channel]) >= by


def _electrolyser(
    toolbox: AgentToolbox,
    telemetry: dict[str, Any],
    constraints: dict[str, Any],
    diagnostic: dict[str, Any],
) -> Diagnosis | None:
    """Same kilowatts in, less hydrogen out: a derated stack."""
    reading = toolbox.inspect_component("electrolyser")
    ratio = reading.get("yield_ratio")
    if ratio is None or ratio >= YIELD_SUSPECT or reading["h2_fill"] >= TANK_BUSY_FILL:
        return None
    rated_kw = reading["rated_kw"]
    safe_kw = max(reading["min_load_kw"], round(rated_kw * ratio, -1))
    hot = reading["temperature_c"] > reading["nominal_temperature_c"] + 5.0
    toolbox.observe(
        f"Electrical input normal at {reading['power_kw']:.0f} kW; H2 yield {ratio:.0%} of design "
        f"({reading['observed_kwh_per_kg']} vs {reading['design_kwh_per_kg']} kWh/kg)"
        + (f", stack at {reading['temperature_c']:.0f} degC" if hot else ""),
        yield_ratio=ratio,
    )
    effective_kw = constraints["effective_kw"]["electrolyser"]
    applied = _tighten(
        toolbox,
        {"electrolyser_max_kw": safe_kw},
        {"electrolyser_max_kw": effective_kw},
        reason=f"stack yield at {ratio:.0%} of design; hold it at power it can convert",
    )
    return Diagnosis(
        category="electrolyser_degradation",
        component="electrolyser",
        summary=(
            f"Electrolyser is producing {ratio:.0%} of the hydrogen its power should give "
            f"({reading['observed_kwh_per_kg']} vs {reading['design_kwh_per_kg']} kWh/kg)"
            + (f" and running hot at {reading['temperature_c']} degC" if hot else "")
            + ". Electrical input looks normal, so this is stack degradation. "
            + (
                f"Capped stack power at {safe_kw:.0f} kW so the plant keeps running in a "
                "degraded mode."
                if applied
                else f"The envelope already holds the stack at {effective_kw:.0f} kW, at or below "
                "what it can convert, so it is left unchanged."
            )
        ),
        confidence="high" if ratio < YIELD_SEVERE else "medium",
    )


def _dac(
    toolbox: AgentToolbox,
    telemetry: dict[str, Any],
    constraints: dict[str, Any],
    diagnostic: dict[str, Any],
) -> Diagnosis | None:
    """No CO2 at all, or CO2 at a worse rate than the sorbent should manage."""
    reading = toolbox.inspect_component("dac")
    rated_methanator_kw = constraints["rated_kw"]["methanator"]
    if not reading["available"]:
        target_kw = round(rated_methanator_kw * CO2_STARVED_METHANATOR_FRAC, -1)
        co2_kg = telemetry["co2_level_kg"]
        toolbox.observe(
            f"Capture unit unavailable: {reading['power_kw']:.0f} kW drawn, "
            f"{reading['co2_capture_kg_h']:.1f} kg/h captured, {co2_kg:.0f} kg left in the buffer",
            co2_level_kg=co2_kg,
        )
        methanator_kw = constraints["effective_kw"]["methanator"]
        applied = _tighten(
            toolbox,
            {"methanator_max_kw": target_kw},
            {"methanator_max_kw": methanator_kw},
            reason="no CO2 being captured; stretch the remaining buffer",
        )
        return Diagnosis(
            category="dac_unavailable",
            component="dac",
            summary=(
                f"The capture unit reports unavailable and CO2 capture has stopped, leaving "
                f"{co2_kg:.0f} kg in the buffer as the only CO2 the plant has. "
                + (
                    f"Methanation capped at {target_kw:.0f} kW to stretch that inventory instead "
                    "of burning through it; "
                    if applied
                    else f"Methanation is already held at {methanator_kw:.0f} kW, low enough to "
                    "stretch that inventory, so the envelope is unchanged; "
                )
                + "hydrogen production is unaffected and continues."
            ),
            confidence="high",
            human_intervention_required=True,
        )
    ratio = reading.get("yield_ratio")
    if ratio is None or ratio >= YIELD_SUSPECT or reading["co2_fill"] >= TANK_BUSY_FILL:
        return None
    return Diagnosis(
        category="dac_degradation",
        component="dac",
        summary=(
            f"DAC is capturing {ratio:.0%} of the CO2 its power should give "
            f"({reading['observed_kwh_per_kg']} vs {reading['design_kwh_per_kg']} kWh/kg), which "
            "looks like sorbent performance rather than an energy problem. Envelope left as is; "
            "CO2 supply will limit methanation if it continues."
        ),
        confidence="medium" if ratio < YIELD_SEVERE else "low",
    )


def _methanator(
    toolbox: AgentToolbox,
    telemetry: dict[str, Any],
    constraints: dict[str, Any],
    diagnostic: dict[str, Any],
) -> Diagnosis | None:
    if telemetry["methanator_available"]:
        return None
    applied = _tighten(
        toolbox,
        {"dac_max_kw": round(constraints["rated_kw"]["dac"] * SHORTFALL_DAC_FRAC, -1)},
        {"dac_max_kw": constraints["effective_kw"]["dac"]},
        reason="reactor unavailable; no point filling the CO2 buffer at full rate",
    )
    return Diagnosis(
        category="methanator_unavailable",
        component="methanator",
        summary=(
            "The methanation reactor reports unavailable, so no CH4 is being made whatever the "
            "feedstock situation. "
            + (
                "Capture rate reduced to stop the CO2 buffer filling pointlessly; "
                if applied
                else "Capture rate is already low enough that the buffer will not fill pointlessly; "
            )
            + "the reactor itself needs a human."
        ),
        confidence="high",
        human_intervention_required=True,
    )


def _solar(
    toolbox: AgentToolbox,
    telemetry: dict[str, Any],
    constraints: dict[str, Any],
    diagnostic: dict[str, Any],
) -> Diagnosis | None:
    """Generation well below clear-sky: an energy problem, not a broken unit."""
    reading = toolbox.inspect_component("solar")
    ratio = reading["output_ratio"]
    if reading["clear_sky_kw"] <= 0.0 or ratio >= SOLAR_SUSPECT:
        return None
    toolbox.observe(
        f"Generation {reading['available_kw']:.0f} kW against {reading['clear_sky_kw']:.0f} kW "
        f"clear-sky ({ratio:.0%}), with every unit converting to specification",
        output_ratio=ratio,
    )
    rated = constraints["rated_kw"]
    reserve_soc = constraints["constraints"]["battery_min_soc"] or 0.0
    applied = _tighten(
        toolbox,
        {
            "electrolyser_max_kw": round(rated["electrolyser"] * SHORTFALL_ELECTROLYSER_FRAC, -1),
            "dac_max_kw": round(rated["dac"] * SHORTFALL_DAC_FRAC, -1),
        },
        {
            "electrolyser_max_kw": constraints["effective_kw"]["electrolyser"],
            "dac_max_kw": constraints["effective_kw"]["dac"],
        },
        reason=f"generation at {ratio:.0%} of clear-sky; protect stored energy",
    )
    # A reserve is raised, not tightened downwards, so it does not go through `_tighten`.
    if reserve_soc < SHORTFALL_RESERVE_SOC:
        result = toolbox.set_controller_constraints(
            {"battery_min_soc": SHORTFALL_RESERVE_SOC},
            reason=f"generation at {ratio:.0%} of clear-sky; hold more charge back",
        )
        if result.get("ok"):
            applied.update(result.get("applied") or {})
    return Diagnosis(
        category="energy_shortfall",
        component="solar",
        summary=(
            f"Generation is at {ratio:.0%} of clear-sky ({reading['available_kw']:.0f} of "
            f"{reading['clear_sky_kw']:.0f} kW) while every unit is performing to specification. "
            "This is an energy-availability problem, not a component failure: "
            + (
                f"{_changed(applied)} until generation recovers."
                if applied
                else "the envelope is already tight enough for the energy on hand, so it is left "
                "unchanged."
            )
        ),
        confidence="high",
    )


def _battery(
    toolbox: AgentToolbox,
    telemetry: dict[str, Any],
    constraints: dict[str, Any],
    diagnostic: dict[str, Any],
) -> Diagnosis | None:
    if telemetry["battery_soc"] > constraints["battery_reserve_soc"] + RESERVE_MARGIN:
        return None
    applied = _tighten(
        toolbox,
        {"dac_max_kw": round(constraints["rated_kw"]["dac"] * SHORTFALL_DAC_FRAC, -1)},
        {"dac_max_kw": constraints["effective_kw"]["dac"]},
        reason="battery on its reserve; shed discretionary load",
    )
    return Diagnosis(
        category="battery_low",
        component="battery",
        summary=(
            f"Battery is down to {telemetry['battery_soc']:.0%}, on top of its "
            f"{constraints['battery_reserve_soc']:.0%} reserve, so there is nothing left to lend "
            "the stack overnight. "
            + (
                "Discretionary capture load reduced; "
                if applied
                else "Discretionary capture load is already reduced, so the envelope is unchanged; "
            )
            + "the plant rides on solar until the pack recovers."
        ),
        confidence="medium",
    )


def _feedstock(
    toolbox: AgentToolbox,
    telemetry: dict[str, Any],
    constraints: dict[str, Any],
    diagnostic: dict[str, Any],
) -> Diagnosis | None:
    """The reactor is fine but one of its two gases is not there."""
    reading = toolbox.inspect_component("methanator")
    if reading["ch4_production_kg_h"] > 0.0:
        return None
    short = [
        species
        for species, level in (("H2", telemetry["h2_level_kg"]), ("CO2", telemetry["co2_level_kg"]))
        if level < 1.0
    ]
    if not short:
        return None
    return Diagnosis(
        category="feedstock_shortage",
        component="methanator",
        summary=(
            f"Methanation has stopped for want of {' and '.join(short)}: the reactor is available "
            "but its tanks are empty. Upstream supply is the constraint, so the envelope is left "
            "alone and the controller will restart the reactor as inventory rebuilds."
        ),
        confidence="medium",
    )


class LLMEngine:
    """The supervisory agent proper, on the OpenAI Agents SDK.

    The SDK is imported lazily so the backend runs (and the tests pass) without
    it, and so an import problem shows up as one failed run rather than a broken
    process. Everything the model can do is a `function_tool` around a method of
    `AgentToolbox`; there is no other surface.
    """

    def __init__(self, model: str = DEFAULT_MODEL, max_turns: int = MAX_TURNS) -> None:
        self.model = model
        self.max_turns = max_turns
        self.label = f"llm:{model}"

    async def investigate(self, toolbox: AgentToolbox, trigger: dict[str, Any]) -> Diagnosis:
        from agents import Agent, Runner  # noqa: PLC0415  (lazy on purpose)

        from .sdk import agent_tools, report_model

        agent = Agent(
            name="Plant supervisor",
            instructions=SYSTEM_PROMPT,
            model=self.model,
            tools=agent_tools(toolbox),
            output_type=report_model(),
        )
        message = build_brief(
            trigger,
            toolbox.get_current_telemetry(),
            toolbox.get_recent_events(10)["events"],
            toolbox.get_predicted_event_risk(),
        )
        result = await Runner.run(agent, message, max_turns=self.max_turns)
        report = result.final_output
        category = report.category if report.category in CATEGORIES else "unclear"
        return Diagnosis(
            category=category,
            summary=report.summary.strip(),
            component=report.component or None,
            confidence=report.confidence,
            human_intervention_required=report.human_intervention_required,
        )


__all__ = ["CATEGORIES", "DEFAULT_MODEL", "Diagnosis", "Engine", "LLMEngine", "RuleEngine"]

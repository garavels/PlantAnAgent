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
from typing import Any, Protocol

from .prompts import SYSTEM_PROMPT, brief as build_brief
from .tools import AgentToolbox
from .trace import Confidence

#: What the supervisor thinks it is looking at. `normal` means "nothing wrong".
CATEGORIES = (
    "normal",
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

    The order is the diagnostic one, not the alphabetical one. Instruments are
    checked before the process (a bad reading invalidates everything downstream),
    then the units that convert power into product, then the energy budget, then
    the feedstock chain.
    """

    label = "rules"

    async def investigate(self, toolbox: AgentToolbox, trigger: dict[str, Any]) -> Diagnosis:
        telemetry = toolbox.get_current_telemetry()
        toolbox.get_recent_events(20)
        toolbox.get_recent_telemetry(120)
        constraints = toolbox.get_controller_constraints()
        diagnostic = toolbox.run_diagnostic("plant")

        for rule in (
            _sensor_inconsistency,
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
            return Diagnosis(
                category="normal",
                summary=(
                    "Telemetry, inventories and component performance are all consistent with "
                    "normal operation. No intervention taken."
                ),
                confidence="high",
            )
        return Diagnosis(
            category="unclear",
            summary=(
                f"Plant status is {telemetry['plant_status']} but no single component explains it "
                "from the available telemetry. Envelope left unchanged."
            ),
            confidence="low",
            human_intervention_required=True,
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

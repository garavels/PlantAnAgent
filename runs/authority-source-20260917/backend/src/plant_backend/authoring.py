"""Natural language in, a `Scenario` the simulator already runs out.

    scenario, warnings = await author("solar drops by half, then the DAC fails")
    session.save_scenario(scenario)
    runtime.load_scenario(scenario.id)

The model never sees the simulator and never writes code: it fills in one JSON
schema built from the simulator's own catalogues (`plant_sim.faults.CATALOGUE`,
`plant_sim.SENSOR_CHANNELS`), and what comes back is validated here before it
becomes a `Scenario`. Anything out of range is clamped with a warning the
operator reads; anything that cannot be expressed at all is a validation error.
So a generated scenario is the same kind of data as a predefined one, and it
runs down the same path.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from plant_sim import SENSOR_CHANNELS, PlantConfig
from plant_sim.faults import CATALOGUE as FAULT_CATALOGUE
from plant_sim.faults import FaultSpec
from plant_sim.scenarios import SCENARIOS, InitialState, Scenario
from plant_sim.state import SECONDS_PER_HOUR

#: Model used when `PLANT_AUTHOR_MODEL` is not set.
DEFAULT_MODEL = "gpt-4.1-mini"
MODEL_ENV = "PLANT_AUTHOR_MODEL"
API_KEY_ENV = "OPENAI_API_KEY"

#: Wall-clock seconds one generation may take.
TIMEOUT_S = 60.0
#: Longest description the operator may send.
PROMPT_LIMIT = 2_000
#: Faults one generated scenario may schedule.
MAX_FAULTS = 6
#: How far into a run a fault may be scheduled, in simulated hours.
MAX_START_H = 24.0
#: Longest a fault may last, in simulated hours.
MAX_DURATION_H = 24.0
NAME_LIMIT = 60
DESCRIPTION_LIMIT = 400

SYSTEM_PROMPT = """You turn an operator's description of a situation into one scenario for a
solar-to-methane plant simulator.

A scenario is data: a starting hour of the simulated day, optional inventory
overrides, and a schedule of faults and disturbances taken from the fixed
catalogue below. You cannot invent fault types, components, channels or
parameters, and you never write code or formulas.

Rules:
- Use only the fault types in the catalogue. If the operator asks for something
  that is not there, use the closest supported fault and say so in `notes`.
- `severity` is a fraction of nominal, 0..1, and only ever what the operator
  actually asked for ("drops by half" is 0.5). With no number given, use the
  catalogue's default severity: a fault nobody quantified is a moderate one, not
  a total one. Outage faults ignore severity.
- The scenario's `start_hour` is a time of day; use 6 unless the operator asks
  for another hour, so the plant starts the run with sun to work with.
- `start_hour` on a fault is an offset from the scenario start, in hours, not a
  time of day. `duration_hours` is null for "until cleared".
- `channel` is only for sensor faults; use null for every other type.
- Leave an `initial` field null unless the operator asks for that inventory.
- Keep at most {max_faults} faults, `start_hour` within 0..{max_start_h} h.
- `name` is a short label, `description` two sentences an operator can read.
- `notes` records anything you mapped, dropped or assumed. Empty otherwise.

Fault catalogue:
{catalogue}

Sensor channels (for sensor faults): {channels}

Inventory overrides: battery_soc is 0..1; h2_kg, co2_kg and ch4_kg are
kilograms in tanks of {h2_capacity:.0f}, {co2_capacity:.0f} and
{ch4_capacity:.0f} kg."""


class ScenarioValidationError(ValueError):
    """The described scenario cannot be expressed in the simulator's format."""


def instructions(config: PlantConfig | None = None) -> str:
    """The system prompt, with the simulator's catalogues written into it."""
    plant = config or PlantConfig()
    catalogue = "\n".join(
        f"- {kind.type} (target {kind.target}, mode {kind.mode}, "
        f"default severity {kind.default_severity}): {kind.description}"
        for kind in FAULT_CATALOGUE.values()
    )
    return SYSTEM_PROMPT.format(
        max_faults=MAX_FAULTS,
        max_start_h=int(MAX_START_H),
        catalogue=catalogue,
        channels=", ".join(SENSOR_CHANNELS),
        h2_capacity=plant.h2_tank.capacity_kg,
        co2_capacity=plant.co2_tank.capacity_kg,
        ch4_capacity=plant.ch4_tank.capacity_kg,
    )


def schema() -> dict[str, Any]:
    """The JSON schema the model must fill in, derived from the simulator.

    Strict Structured Outputs allows types and enums but not numeric bounds, so
    ranges are the validator's job (`build`); the schema's work is to make an
    unsupported fault type, component or channel impossible to express at all.
    """
    number_or_null = {"type": ["number", "null"]}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "description", "start_hour", "initial", "faults", "notes"],
        "properties": {
            "name": {"type": "string", "description": "Short label, a few words."},
            "description": {
                "type": "string",
                "description": "Two sentences: the situation and what it tests.",
            },
            "start_hour": {
                "type": "number",
                "description": "Time of day the scenario starts at, in hours, 0..24.",
            },
            "initial": {
                "type": "object",
                "additionalProperties": False,
                "required": ["battery_soc", "h2_kg", "co2_kg", "ch4_kg"],
                "properties": {
                    "battery_soc": number_or_null,
                    "h2_kg": number_or_null,
                    "co2_kg": number_or_null,
                    "ch4_kg": number_or_null,
                },
                "description": "Inventory overrides; null keeps the plant default.",
            },
            "faults": {
                "type": "array",
                "description": "Faults and disturbances, scheduled from the scenario start.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "severity", "start_hour", "duration_hours", "channel"],
                    "properties": {
                        "type": {"type": "string", "enum": list(FAULT_CATALOGUE)},
                        "severity": {"type": "number", "description": "Fraction 0..1."},
                        "start_hour": {
                            "type": "number",
                            "description": "Hours after the scenario start.",
                        },
                        "duration_hours": {
                            **number_or_null,
                            "description": "Null means until cleared.",
                        },
                        "channel": {
                            "anyOf": [
                                {"type": "string", "enum": list(SENSOR_CHANNELS)},
                                {"type": "null"},
                            ],
                            "description": "Sensor faults only; null otherwise.",
                        },
                    },
                },
            },
            "notes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "What was mapped, assumed or left out.",
            },
        },
    }


def build(
    payload: dict[str, Any],
    *,
    config: PlantConfig | None = None,
    taken: tuple[str, ...] = (),
) -> tuple[Scenario, list[str]]:
    """Validate a scenario payload into a `Scenario`, with the warnings it raised.

    Every number is clamped to what the simulator accepts and every fault goes
    through `FaultSpec.parse`, so what comes out is indistinguishable from a
    scenario written by hand in `plant_sim.scenarios`.
    """
    if not isinstance(payload, dict):
        raise ScenarioValidationError("the generated scenario is not an object")
    plant = config or PlantConfig()
    warnings: list[str] = [str(note) for note in _sequence(payload.get("notes")) if str(note)]

    name = _text(payload.get("name"), NAME_LIMIT) or "Generated scenario"
    description = _text(payload.get("description"), DESCRIPTION_LIMIT) or (
        "Generated from an operator description."
    )
    start_hour, clamped = _clamp(_number(payload.get("start_hour"), 6.0), 0.0, 24.0)
    if clamped:
        warnings.append(f"start hour moved into the simulated day: {start_hour:.1f} h")

    raw_faults = _sequence(payload.get("faults"))
    if len(raw_faults) > MAX_FAULTS:
        warnings.append(f"kept the first {MAX_FAULTS} of {len(raw_faults)} faults")
        raw_faults = raw_faults[:MAX_FAULTS]

    specs: list[FaultSpec] = []
    for entry in raw_faults:
        specs.append(_fault(entry, warnings))
    if not specs and not raw_faults:
        warnings.append("no faults were requested: this is a nominal day")

    scenario = Scenario(
        id=_identifier(name, taken),
        name=name,
        description=description,
        start_hour=start_hour,
        initial=_initial(payload.get("initial"), plant, warnings),
        faults=tuple(specs),
    )
    return scenario, warnings


async def author(
    prompt: str,
    *,
    model: str | None = None,
    config: PlantConfig | None = None,
    taken: tuple[str, ...] = (),
) -> tuple[Scenario, list[str]]:
    """Ask the model for a scenario, then validate it. Raises on both halves."""
    text = prompt.strip()
    if not text:
        raise ScenarioValidationError("describe the scenario you want first")
    if len(text) > PROMPT_LIMIT:
        raise ScenarioValidationError(f"description is longer than {PROMPT_LIMIT} characters")
    if not os.environ.get(API_KEY_ENV):
        raise ScenarioValidationError(
            f"scenario generation needs {API_KEY_ENV}; the predefined scenarios still run"
        )
    payload = await _ask(text, model or os.environ.get(MODEL_ENV) or DEFAULT_MODEL, config)
    return build(payload, config=config, taken=taken)


async def _ask(prompt: str, model: str, config: PlantConfig | None) -> dict[str, Any]:
    """One Responses API call, constrained to `schema()`."""
    from openai import AsyncOpenAI, OpenAIError  # noqa: PLC0415  (lazy on purpose)

    client = AsyncOpenAI(timeout=TIMEOUT_S)
    try:
        response = await client.responses.create(
            model=model,
            instructions=instructions(config),
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "plant_scenario",
                    "strict": True,
                    "schema": schema(),
                }
            },
        )
    except OpenAIError as reason:
        raise ScenarioValidationError(f"the model could not be reached: {reason}") from reason
    try:
        parsed = json.loads(response.output_text)
    except (json.JSONDecodeError, TypeError) as reason:
        raise ScenarioValidationError("the model did not return a scenario") from reason
    if not isinstance(parsed, dict):
        raise ScenarioValidationError("the model did not return a scenario object")
    return parsed


def _fault(entry: Any, warnings: list[str]) -> FaultSpec:
    if not isinstance(entry, dict):
        raise ScenarioValidationError("a scheduled fault is not an object")
    fault_type = str(entry.get("type", ""))
    kind = FAULT_CATALOGUE.get(fault_type)
    if kind is None:
        raise ScenarioValidationError(
            f"unsupported fault {fault_type!r}; the simulator offers: "
            + ", ".join(FAULT_CATALOGUE)
        )

    start_h, clamped = _clamp(_number(entry.get("start_hour"), 0.0), 0.0, MAX_START_H)
    if clamped:
        warnings.append(f"{fault_type}: start moved to {start_h:.1f} h after the scenario start")

    duration_h = _optional_number(entry.get("duration_hours"))
    if duration_h is not None:
        duration_h, clamped = _clamp(duration_h, 0.0, MAX_DURATION_H)
        if clamped:
            warnings.append(f"{fault_type}: duration clamped to {duration_h:.1f} h")

    channel = entry.get("channel")
    if kind.mode == "sensor":
        if channel not in SENSOR_CHANNELS:
            warnings.append(
                f"{fault_type}: channel {channel!r} is not measured; using {kind.default_channel}"
            )
            channel = kind.default_channel
    elif channel is not None:
        warnings.append(f"{fault_type}: dropped the channel, it only applies to sensor faults")
        channel = None

    severity = _number(entry.get("severity"), kind.default_severity)
    spec = FaultSpec.parse(
        {
            "type": fault_type,
            "severity": severity,
            "start_s": start_h * SECONDS_PER_HOUR,
            "duration_s": None if duration_h is None else duration_h * SECONDS_PER_HOUR,
            "channel": channel,
        }
    )
    # FaultSpec.parse enforces the simulator's own limits (outages are total,
    # degradations cap below 1.0); say so rather than silently disagreeing.
    if abs(spec.severity - severity) > 1e-6 and kind.mode != "outage":
        warnings.append(f"{fault_type}: severity limited to {spec.severity:.2f}")
    return spec


def _initial(raw: Any, plant: PlantConfig, warnings: list[str]) -> InitialState:
    if not isinstance(raw, dict):
        return InitialState()
    values: dict[str, float | None] = {}
    for field, high, unit in (
        ("battery_soc", 1.0, ""),
        ("h2_kg", plant.h2_tank.capacity_kg, " kg"),
        ("co2_kg", plant.co2_tank.capacity_kg, " kg"),
        ("ch4_kg", plant.ch4_tank.capacity_kg, " kg"),
    ):
        value = _optional_number(raw.get(field))
        if value is not None:
            value, clamped = _clamp(value, 0.0, high)
            if clamped:
                warnings.append(f"initial {field} clamped to {value:g}{unit}")
        values[field] = value
    return InitialState(**values)  # type: ignore[arg-type]  # keys are the field names


def _identifier(name: str, taken: tuple[str, ...]) -> str:
    """A scenario id that collides with nothing already in the catalogue."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "scenario"
    candidate = f"ai_{slug}"[:48]
    existing = set(SCENARIOS) | set(taken)
    if candidate not in existing:
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in existing:
        suffix += 1
    return f"{candidate}_{suffix}"


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _text(value: Any, limit: int) -> str:
    text = " ".join(str(value).split()) if isinstance(value, str) else ""
    return text[:limit].strip()


def _clamp(value: float, low: float, high: float) -> tuple[float, bool]:
    if value < low:
        return low, True
    if value > high:
        return high, True
    return value, False


def _number(value: Any, fallback: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return float(value)


def _optional_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


__all__ = [
    "DEFAULT_MODEL",
    "MAX_FAULTS",
    "ScenarioValidationError",
    "author",
    "build",
    "instructions",
    "schema",
]

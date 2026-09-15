"""Validated scenario descriptions for the active, forecast-driven digital twin.

The model writes data only. Manual edits and model output use the same strict
validation before weather is resolved or the shared simulation is changed.
"""

from __future__ import annotations

import json
import math
import os
from typing import Annotated, Any, Literal

from plant_sim import SENSOR_CHANNELS, PlantConfig
from plant_sim.faults import CATALOGUE as FAULT_CATALOGUE
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .site_weather import EXAMPLE_SITES

API_KEY_ENV = "OPENROUTER_API_KEY"
MODEL_ENV = "PLANT_SCENARIO_MODEL"
DEFAULT_MODEL = "openai/gpt-4.1-mini"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TIMEOUT_S = 60.0
PROMPT_LIMIT = 4000
MAX_SUNLIGHT_PERIODS = 16
MAX_FAULTS = 12
MAX_DELIVERIES = 80
Note = Annotated[str, Field(min_length=1, max_length=500)]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False, str_strip_whitespace=True
    )


class ScenarioSite(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    latitude: float = Field(ge=34, le=72)
    longitude: float = Field(ge=-25, le=45)
    source: Literal["saved", "live"]
    example_id: Literal["seville", "trieste", "stockholm"] | None

    @model_validator(mode="after")
    def consistent_source(self) -> ScenarioSite:
        if self.source == "saved":
            if self.example_id is None:
                raise ValueError("Saved weather requires an example_id.")
            example = EXAMPLE_SITES[self.example_id]
            if any(abs(getattr(self, key) - example[key]) > 1e-5 for key in ("latitude", "longitude")):
                raise ValueError("Saved weather coordinates must match the selected example location.")
        elif self.example_id is not None:
            raise ValueError("Live weather requires example_id to be null.")
        return self


class InitialInventories(StrictModel):
    battery_soc: float = Field(ge=0, le=1)
    h2_kg: float = Field(ge=0, le=PlantConfig().h2_tank.capacity_kg)
    co2_kg: float = Field(ge=0, le=PlantConfig().co2_tank.capacity_kg)
    ch4_kg: float = Field(ge=0, le=PlantConfig().ch4_tank.capacity_kg)


class SunlightPeriod(StrictModel):
    start_hour: float = Field(ge=0, lt=240)
    end_hour: float = Field(gt=0, le=240)
    scale: float = Field(ge=0, le=1)
    forecast_known: bool

    @model_validator(mode="after")
    def ordered_hours(self) -> SunlightPeriod:
        if self.end_hour <= self.start_hour:
            raise ValueError("Sunlight end_hour must be later than start_hour.")
        if not _aligned(self.start_hour, 1) or not _aligned(self.end_hour, 1):
            raise ValueError("Sunlight periods must start and end on whole hours.")
        return self


class ScheduledFault(StrictModel):
    type: str = Field(min_length=1, max_length=60)
    severity: float = Field(ge=0, le=1)
    start_hour: float = Field(ge=0, lt=240)
    duration_hours: float | None = Field(gt=0, le=240)
    channel: str | None = Field(max_length=100)

    @model_validator(mode="after")
    def supported_fault(self) -> ScheduledFault:
        kind = FAULT_CATALOGUE.get(self.type)
        if kind is None:
            raise ValueError("Unsupported fault type. Choose a fault from the catalogue.")
        if kind.mode == "outage" and self.severity != 1:
            raise ValueError("Outages require severity 1; use a degradation fault for a partial loss.")
        if kind.mode == "degradation" and self.severity > 0.95:
            raise ValueError("Degradation severity cannot exceed 0.95.")
        if kind.mode == "sensor":
            if self.channel not in SENSOR_CHANNELS:
                raise ValueError("Sensor faults require a supported sensor channel.")
        elif self.channel is not None:
            raise ValueError("Only sensor faults accept a channel; use null for other faults.")
        if not _aligned(self.start_hour, 0.25) or (
            self.duration_hours is not None and not _aligned(self.duration_hours, 0.25)
        ):
            raise ValueError("Fault start and duration must align to 15-minute intervals.")
        return self


class ScenarioDelivery(StrictModel):
    due_hour: float = Field(gt=0, le=240)
    kg: float = Field(gt=0, le=100000)

    @model_validator(mode="after")
    def quarter_hour(self) -> ScenarioDelivery:
        if not _aligned(self.due_hour, 0.25):
            raise ValueError("Delivery deadlines must align to 15-minute intervals.")
        return self


class ScenarioEconomics(StrictModel):
    delivered_service_value_eur_per_kg: float = Field(ge=0, le=10000)
    unmet_delivery_penalty_eur_per_kg: float = Field(gt=0, le=100000)

    @model_validator(mode="after")
    def penalty_exceeds_value(self) -> ScenarioEconomics:
        if self.unmet_delivery_penalty_eur_per_kg <= self.delivered_service_value_eur_per_kg:
            raise ValueError("The missed-delivery penalty must exceed the delivered methane value.")
        return self


class ScenarioSpec(StrictModel):
    version: Literal["plant-scenario-v1"]
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=800)
    duration_days: int = Field(ge=1, le=10)
    seed: int = Field(ge=0, le=2147483647)
    site: ScenarioSite
    solar_kw: float = Field(ge=100, le=50000)
    battery_kwh: float = Field(ge=100, le=100000)
    initial: InitialInventories
    sunlight: list[SunlightPeriod] = Field(max_length=MAX_SUNLIGHT_PERIODS)
    faults: list[ScheduledFault] = Field(max_length=MAX_FAULTS)
    deliveries: list[ScenarioDelivery] = Field(max_length=MAX_DELIVERIES)
    economics: ScenarioEconomics
    notes: list[Note] = Field(max_length=32)
    unsupported: list[Note] = Field(max_length=32)

    @model_validator(mode="after")
    def bounded_schedule(self) -> ScenarioSpec:
        horizon = self.duration_days * 24
        previous_end = -1.0
        for period in sorted(self.sunlight, key=lambda item: item.start_hour):
            if period.end_hour > horizon:
                raise ValueError("Sunlight periods must fit within the scenario duration.")
            if period.start_hour < previous_end:
                raise ValueError("Sunlight periods cannot overlap.")
            previous_end = period.end_hour
        for fault in self.faults:
            if fault.start_hour >= horizon or (
                fault.duration_hours is not None and fault.start_hour + fault.duration_hours > horizon
            ):
                raise ValueError("Faults must start and finish within the scenario duration.")
        if any(delivery.due_hour > horizon for delivery in self.deliveries):
            raise ValueError("Delivery deadlines must fit within the scenario duration.")
        if len({delivery.due_hour for delivery in self.deliveries}) != len(self.deliveries):
            raise ValueError("Delivery deadlines must be unique; combine quantities due at the same time.")
        self.deliveries = sorted(self.deliveries, key=lambda item: item.due_hour)
        return self


def _aligned(value: float, interval: float) -> bool:
    return math.isclose(value / interval, round(value / interval), abs_tol=1e-9)


def defaults() -> dict[str, Any]:
    """A complete offline-editable week, with no implicit inventory settings."""
    return {
        "version": "plant-scenario-v1",
        "name": "A week in Trieste",
        "description": "A seven-day plant scenario using a saved Trieste weather forecast.",
        "duration_days": 7, "seed": 42,
        "site": {**EXAMPLE_SITES["trieste"], "source": "saved", "example_id": "trieste"},
        "solar_kw": 5000.0, "battery_kwh": 8000.0,
        "initial": {"battery_soc": 0.5, "h2_kg": 300.0, "co2_kg": 1500.0, "ch4_kg": 150.0},
        "sunlight": [], "faults": [],
        "deliveries": [{"due_hour": float(hour), "kg": 150.0} for hour in range(6, 169, 6)],
        "economics": {"delivered_service_value_eur_per_kg": 4.0, "unmet_delivery_penalty_eur_per_kg": 20.0},
        "notes": [], "unsupported": [],
    }


def examples() -> list[dict[str, Any]]:
    cloudy = defaults()
    cloudy.update(
        name="Cloudy week, low reserves",
        description="Two days of reduced sunlight, low initial reserves and a six-hour capture outage.",
        initial={"battery_soc": 0.3, "h2_kg": 200.0, "co2_kg": 3600.0, "ch4_kg": 150.0},
        sunlight=[{"start_hour": 24.0, "end_hour": 72.0, "scale": 0.3, "forecast_known": True}],
        faults=[{"type": "dac_outage", "severity": 1.0, "start_hour": 84.0, "duration_hours": 6.0, "channel": None}],
        deliveries=[{"due_hour": float(18 + day * 24), "kg": 150.0} for day in range(7)],
        notes=["Evening deliveries mean 18 hours into each scenario day, measured from the forecast start.",
               "Days two and three receive 30% of saved forecast sunlight, and the controller knows this schedule."],
    )
    surprise = defaults()
    surprise.update(
        name="Unexpected cloud cover",
        description="An unforecast reduction in sunlight tests the plant's reserve management.",
        sunlight=[{"start_hour": 48.0, "end_hour": 72.0, "scale": 0.2, "forecast_known": False}],
        notes=["Day three receives 20% of the saved sunlight forecast; the controller retains the nominal weather forecast."],
    )
    return [
        {"title": "Cloudy week and capture outage", "prompt": "Simulate one week near Trieste using saved weather. Start with battery at 30%, hydrogen at 10% and CO2 at 60%. Days two and three receive 70% less sunlight. Shut down CO2 capture for six hours starting 84 hours into the run. Deliver 150 kg of methane each evening.", "scenario": cloudy},
        {"title": "Unexpected clouds", "prompt": "Use a saved week in Trieste. On day three, sunlight unexpectedly falls to 20% of forecast. Keep the default delivery commitments and starting inventories.", "scenario": surprise},
        {"title": "Start from the defaults", "prompt": "Run a nominal week in Trieste with saved weather, 5 MW solar, an 8 MWh battery and 150 kg of methane due every six hours.", "scenario": defaults()},
    ]


def build(payload: dict[str, Any]) -> ScenarioSpec:
    """Reject invalid manual or generated data without silently clamping it."""
    try:
        return ScenarioSpec.model_validate(payload)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_input=False)
        message = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'scenario'}: {error['msg']}"
            for error in errors[:8]
        )
        raise ScenarioValidationError(message) from exc


def warnings(scenario: ScenarioSpec) -> list[str]:
    messages = list(scenario.notes)
    messages.extend(f"Unsupported request: {item}" for item in scenario.unsupported)
    messages.append("This plant is off-grid. Methane value and missed-delivery penalties change economic reporting; the controller prioritizes delivery service.")
    messages.append("Weather affects production through sunlight. Temperature, humidity and wind do not change equipment physics.")
    if scenario.site.source == "saved":
        messages.append("Weather is a saved forecast with its original dates, not a live forecast.")
    if any(fault.type not in ("dac_outage",) for fault in scenario.faults):
        messages.append("During faults outside the MPC planner's equipment model, control may fall back to the guarded reactive controller.")
    return list(dict.fromkeys(messages))


def schema() -> dict[str, Any]:
    """All properties are required, and every object rejects additional keys."""
    result = ScenarioSpec.model_json_schema()
    fault = result["$defs"]["ScheduledFault"]["properties"]
    fault["type"]["enum"] = list(FAULT_CATALOGUE)
    fault["channel"]["anyOf"][0]["enum"] = list(SENSOR_CHANNELS)
    return result


def instructions() -> str:
    catalogue = "\n".join(
        f"- {kind.type}: {kind.description} Mode={kind.mode}; default severity={kind.default_severity}."
        for kind in FAULT_CATALOGUE.values()
    )
    return f"""Translate the operator's situation into supported data for an off-grid solar-to-methane digital twin.
Return the complete schema. Never emit code, formulas, URLs, commands, or extra keys.
Use this default scenario unless the operator requests a change: {json.dumps(defaults())}

Interpretation rules:
- Set unsupported to an explicit list of requests the simulator cannot model. Do not substitute a different physical effect or silently drop a request. Keep supported settings editable. An unsupported request prevents activation until the operator removes or revises it.
- Record every assumption, interpretation and default that materially affects the requested scenario in notes. Do not invent observed data or claim that the weather forecast was fetched.
- Whole-day duration is 1..10, seed 0..2147483647. Use at most 16 sunlight periods, 12 faults, 80 deliveries.
- Location is European, latitude 34..72, longitude -25..45. Saved weather is only available for these exact sites: {json.dumps(EXAMPLE_SITES)}. Saved coordinates must match the selected example. Use saved Trieste when location is unspecified. A request for another location or current weather uses source live and example_id null; note that inferred city coordinates are approximate. No geocoder or weather API is available to you.
- The timeline begins at the acquired forecast's first timestamp in UTC, not at an invented date or local midnight. Day one is hours 0..24. Explicit weekday/date requests that cannot be anchored to that unknown forecast start must be flagged unsupported; otherwise note relative-hour interpretations. An evening without a precise time means offset 18 + 24*day and this interpretation must be noted.
- Sunlight periods use whole-hour [start_hour,end_hour) offsets, cannot overlap, and must fit the duration. scale is a fraction of the forecast sunlight: 70% less means 0.3. forecast_known=true makes the modified weather known to the controller; false means an unexpected change with the original forecast retained. Use true unless the user requests a surprise. All other hours preserve the forecast. Do not invent brighter-than-forecast sunlight: only scale 0..1 is supported.
- Weather physics supports sunlight only. Wind, temperature, humidity, snow, atmospheric gas concentrations and gas purity have no modeled equipment effect. Atmospheric gas is not tank inventory. Flag such requests unsupported.
- Initial inventories: battery_soc is 0..1; H2, CO2 and CH4 values are kg with tank capacities 2000, 6000 and 3000 respectively. Convert requested fill percentages using those capacities. Nearly empty means 10%, nearly full 90%, with explicit notes. Retain defaults for unspecified inventory.
- Solar 100..50000 kW; battery 100..100000 kWh. Other tank capacities, equipment ratings and control policies are not configurable here; flag them unsupported.
- Fault starts and durations use 0.25-hour increments and must fit within the scenario. duration_hours=null means until run end. severity is fractional loss, not remaining performance. Outages require severity=1, degradations 0..0.95, sensor drift 0..1. Use catalogue defaults for unquantified severity and explain the assumption. Sensor drift needs a channel; every other channel must be null.
- Deliveries are explicit individual deadlines in hours after scenario start on 0.25-hour increments, strictly positive, and kg positive up to100000. Deadlines must be unique and chronological; combine quantities due at the same time. Deadlines may equal the end of the run. Expand recurring requests into the list; an empty list means no delivery commitments.
- Economics supports delivered_service_value_eur_per_kg 0..10000 and unmet_delivery_penalty_eur_per_kg > delivered value and <=100000. If requested values conflict, flag the conflict unsupported and retain valid defaults. These change accounting; the controller prioritizes fulfilling deliveries. Grid electricity import/export, price series, trading and changing market prices over time are unsupported.

Fault catalogue:
{catalogue}
Sensor channels: {', '.join(SENSOR_CHANNELS)}.
"""


class ScenarioValidationError(ValueError):
    """A draft contains unsupported or out-of-range settings."""


class AuthoringUnavailableError(ScenarioValidationError):
    """The model is unavailable; manual scenario authoring still works."""


async def author(prompt: str, *, model: str | None = None) -> ScenarioSpec:
    """Translate one description; never install it or fetch its weather here."""
    text = prompt.strip()
    if not text:
        raise ScenarioValidationError("Describe the scenario you want first.")
    if len(text) > PROMPT_LIMIT:
        raise ScenarioValidationError(f"Descriptions can contain at most {PROMPT_LIMIT} characters.")
    if not os.environ.get(API_KEY_ENV, "").strip():
        raise AuthoringUnavailableError(
            "Natural-language generation needs OPENROUTER_API_KEY in the backend environment. "
            "Set it and restart the backend, or edit the scenario manually using a saved weather example."
        )
    payload = await _ask(text, model or os.environ.get(MODEL_ENV) or DEFAULT_MODEL)
    return build(payload)


async def _ask(prompt: str, model: str) -> dict[str, Any]:
    """Request strict JSON from OpenRouter, then validate it before use."""
    from openai import AsyncOpenAI, OpenAIError

    unavailable_message = (
        "OpenRouter could not generate the scenario. Check the backend API key, "
        "credits, network connection and that PLANT_SCENARIO_MODEL names an OpenRouter "
        "model supporting structured outputs, or edit the scenario manually."
    )
    try:
        async with AsyncOpenAI(
            api_key=os.environ.get(API_KEY_ENV, "").strip(),
            base_url=OPENROUTER_BASE_URL,
            timeout=TIMEOUT_S,
            max_retries=0,
        ) as client:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": instructions()},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=6000,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "digital_twin_scenario", "strict": True, "schema": schema(),
                    },
                },
                extra_body={"provider": {"require_parameters": True}},
            )
    except OpenAIError as exc:
        # Provider exceptions can contain request details; retain no secrets in
        # browser errors. The failure is explicit and does not invent a draft.
        raise AuthoringUnavailableError(unavailable_message) from exc
    choices = getattr(response, "choices", None) or []
    choice = choices[0] if choices else None
    if (
        getattr(response, "error", None)
        or getattr(choice, "error", None)
        or getattr(choice, "finish_reason", None) == "error"
    ):
        raise AuthoringUnavailableError(unavailable_message)
    message = getattr(choice, "message", None)
    if getattr(message, "refusal", None) or getattr(choice, "finish_reason", None) == "content_filter":
        raise ScenarioValidationError(
            "The model declined this description. Describe plant conditions and events, "
            "or use the manual editor."
        )
    if getattr(choice, "finish_reason", None) != "stop":
        raise ScenarioValidationError(
            "The model returned an incomplete scenario. Shorten the description and try again."
        )
    try:
        payload = json.loads(message.content)
    except (AttributeError, json.JSONDecodeError, TypeError) as exc:
        raise ScenarioValidationError("The model did not return a valid scenario. Try again.") from exc
    if not isinstance(payload, dict):
        raise ScenarioValidationError("The model did not return a scenario object. Try again.")
    return payload

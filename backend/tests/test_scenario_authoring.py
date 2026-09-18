"""The model and manual editor share strict simulation-specific validation."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from plant_backend import scenario_authoring as authoring


def fault(**changes):
    return {"type": "dac_outage", "severity": 1.0, "start_hour": 84.0,
            "duration_hours": 6.0, "channel": None, **changes}


def sunlight(**changes):
    return {"start_hour": 24.0, "end_hour": 48.0, "scale": 0.3,
            "forecast_known": True, **changes}


def test_defaults_and_examples_are_complete_valid_editable_scenarios() -> None:
    default = authoring.build(authoring.defaults())
    assert default.duration_days == 7
    assert default.initial.h2_kg == 300
    assert default.deliveries[-1].due_hour == 168
    for example in authoring.examples():
        assert authoring.build(example["scenario"]).unsupported == []
    independent = authoring.defaults()
    independent["initial"]["h2_kg"] = 0
    assert authoring.defaults()["initial"]["h2_kg"] == 300


def test_structured_schema_requires_every_property_and_forbids_unknown_keys() -> None:
    schema = authoring.schema()

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    fault_schema = schema["$defs"]["ScheduledFault"]["properties"]
    assert fault_schema["type"]["enum"] == list(authoring.FAULT_CATALOGUE)
    assert fault_schema["channel"]["anyOf"][0]["enum"] == list(authoring.SENSOR_CHANNELS)


@pytest.mark.parametrize("change", [
    {"duration_days": 0}, {"duration_days": 11}, {"duration_days": True},
    {"duration_days": 7.0}, {"duration_days": "7"}, {"seed": -1}, {"seed": True},
    {"seed": 2147483648}, {"solar_kw": True}, {"solar_kw": "5000"},
    {"solar_kw": 50001}, {"solar_kw": float("inf")}, {"battery_kwh": float("nan")},
    {"version": "v2"}, {"name": " "}, {"code": "print('bad')"},
])
def test_invalid_scalars_and_unknown_fields_are_rejected_without_clamping(change) -> None:
    with pytest.raises(authoring.ScenarioValidationError):
        authoring.build({**authoring.defaults(), **change})


@pytest.mark.parametrize("change", [
    {"battery_soc": 1.01}, {"h2_kg": 2001}, {"co2_kg": 6001}, {"ch4_kg": 3001},
    {"battery_soc": -0.1}, {"h2_kg": True}, {"co2_kg": "50"}, {"ch4_kg": None},
])
def test_initial_inventories_respect_real_tank_limits(change) -> None:
    value = authoring.defaults()
    value["initial"].update(change)
    with pytest.raises(authoring.ScenarioValidationError):
        authoring.build(value)


def test_saved_weather_cannot_be_relabeled_as_another_location() -> None:
    value = authoring.defaults()
    value["site"]["latitude"] = 50
    with pytest.raises(authoring.ScenarioValidationError, match="match"):
        authoring.build(value)
    value["site"].update(source="live", example_id=None)
    assert authoring.build(value).site.latitude == 50
    value["site"]["example_id"] = "trieste"
    with pytest.raises(authoring.ScenarioValidationError, match="null"):
        authoring.build(value)


@pytest.mark.parametrize("periods", [
    [sunlight(start_hour=24.5)], [sunlight(end_hour=24)], [sunlight(end_hour=169)],
    [sunlight(scale=1.1)], [sunlight(forecast_known=1)],
    [sunlight(), sunlight(start_hour=47, end_hour=60)],
    [sunlight(start_hour=float(i), end_hour=float(i + 1)) for i in range(17)],
])
def test_sunlight_periods_must_fit_hourly_nonoverlapping_schedule(periods) -> None:
    with pytest.raises(authoring.ScenarioValidationError):
        authoring.build({**authoring.defaults(), "sunlight": periods})


def test_adjacent_periods_and_explicit_no_deliveries_are_valid() -> None:
    scenario = authoring.build({**authoring.defaults(), "sunlight": [
        sunlight(start_hour=0, end_hour=24, scale=0), sunlight(),
    ], "deliveries": []})
    assert scenario.sunlight[0].scale == 0
    assert scenario.deliveries == []


@pytest.mark.parametrize("entry", [
    fault(type="turbine_trip"), fault(severity=0.5),
    fault(type="dac_degradation", severity=1), fault(start_hour=84.1),
    fault(duration_hours=0), fault(duration_hours=0.1), fault(start_hour=168),
    fault(duration_hours=100), fault(channel="h2_level_kg"),
    fault(type="sensor_drift", severity=0.2, channel=None),
    fault(type="sensor_drift", severity=0.2, channel="not-a-sensor"),
])
def test_faults_match_simulator_semantics_and_run_boundaries(entry) -> None:
    with pytest.raises(authoring.ScenarioValidationError):
        authoring.build({**authoring.defaults(), "faults": [entry]})


def test_supported_sensor_and_until_end_faults_remain_exact() -> None:
    scenario = authoring.build({**authoring.defaults(), "faults": [
        fault(type="sensor_drift", severity=0.2, duration_hours=None, channel="h2_level_kg")
    ]})
    assert scenario.faults[0].duration_hours is None
    assert scenario.faults[0].severity == 0.2


@pytest.mark.parametrize("deliveries", [
    [{"due_hour": 0, "kg": 10}], [{"due_hour": 169, "kg": 10}],
    [{"due_hour": 1.1, "kg": 10}], [{"due_hour": 1, "kg": -1}],
    [{"due_hour": 1, "kg": True}], [{"due_hour": 1, "kg": 100001}],
    [{"due_hour": 1, "kg": 10}] * 81,
])
def test_delivery_deadlines_mass_and_count_are_bounded(deliveries) -> None:
    with pytest.raises(authoring.ScenarioValidationError):
        authoring.build({**authoring.defaults(), "deliveries": deliveries})


def test_penalty_must_exceed_service_value() -> None:
    value = authoring.defaults()
    value["economics"]["delivered_service_value_eur_per_kg"] = 25
    with pytest.raises(authoring.ScenarioValidationError, match="penalty"):
        authoring.build(value)


def test_duplicate_delivery_deadlines_are_rejected_and_order_is_canonical() -> None:
    value = authoring.defaults()
    value["deliveries"] = [{"due_hour": 12, "kg": 10}, {"due_hour": 6, "kg": 20}]
    assert [item.due_hour for item in authoring.build(value).deliveries] == [6, 12]
    value["deliveries"].append({"due_hour": 6, "kg": 30})
    with pytest.raises(authoring.ScenarioValidationError, match="unique"):
        authoring.build(value)


def test_missing_key_is_actionable_and_does_not_call_model(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-only-must-not-enable-studio")
    with pytest.raises(authoring.AuthoringUnavailableError, match="OPENROUTER_API_KEY") as error:
        asyncio.run(authoring.author("A cloudy week"))
    assert "manually" in str(error.value)
    assert "openai-only-must-not-enable-studio" not in str(error.value)


def test_generation_uses_configured_model_and_strict_build(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("PLANT_SCENARIO_MODEL", "configured-model")
    monkeypatch.setenv("PLANT_AUTHOR_MODEL", "legacy-model-must-not-be-used")
    calls = []

    async def ask(prompt, model):
        calls.append((prompt, model))
        return authoring.defaults()

    monkeypatch.setattr(authoring, "_ask", ask)
    assert asyncio.run(authoring.author("  A week  ")).duration_days == 7
    assert calls == [("A week", "configured-model")]


def test_generation_ignores_legacy_model_and_defaults_to_openrouter_identifier(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("PLANT_SCENARIO_MODEL", raising=False)
    monkeypatch.setenv("PLANT_AUTHOR_MODEL", "legacy-direct-openai-model")
    calls = []

    async def ask(prompt, model):
        calls.append(model)
        return authoring.defaults()

    monkeypatch.setattr(authoring, "_ask", ask)
    asyncio.run(authoring.author("A week"))
    assert calls == ["openai/gpt-4.1-mini"]


def completion(content, *, finish_reason="stop", refusal=None, error=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason,
        error=error,
        message=SimpleNamespace(content=content, refusal=refusal),
    )])


def response_client(monkeypatch, response):
    import openai

    monkeypatch.setenv("OPENROUTER_API_KEY", "router-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-openai-key")
    calls = []

    class Client:
        def __init__(self, **options):
            calls.append(options)
            self.chat = SimpleNamespace(completions=self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            calls.append({"closed": True})
            return False

        async def create(self, **options):
            calls.append(options)
            if isinstance(response, Exception):
                raise response
            return response

    monkeypatch.setattr(openai, "AsyncOpenAI", Client)
    return calls


def test_openrouter_chat_uses_strict_json_schema_and_closes_client(monkeypatch) -> None:
    import json

    calls = response_client(monkeypatch, completion(json.dumps(authoring.defaults())))
    assert asyncio.run(authoring._ask("A week", "example-model")) == authoring.defaults()
    assert calls[0]["base_url"] == "https://openrouter.ai/api/v1"
    assert calls[0]["api_key"] == "router-test-key"
    assert calls[0]["timeout"] == 60
    assert calls[1]["response_format"]["type"] == "json_schema"
    assert calls[1]["response_format"]["json_schema"] == {
        "name": "digital_twin_scenario", "strict": True, "schema": authoring.schema(),
    }
    assert calls[1]["extra_body"] == {"provider": {"require_parameters": True}}
    assert calls[1]["messages"] == [
        {"role": "system", "content": authoring.instructions()},
        {"role": "user", "content": "A week"},
    ]
    assert calls[1]["max_tokens"] == 6000
    assert calls[1]["model"] == "example-model"
    assert calls[0]["max_retries"] == 0
    assert calls[-1] == {"closed": True}


def test_openrouter_sdk_serializes_expected_url_auth_and_structured_output(monkeypatch) -> None:
    import json

    import httpx
    import openai

    real_client = openai.AsyncOpenAI
    requests = []
    clients = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-http-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-openai-key")

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "gen-scenario-test", "object": "chat.completion", "created": 1,
            "model": "openai/gpt-4.1-mini",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(authoring.defaults()),
            }}],
        })

    def client(**options):
        transport_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        clients.append(transport_client)
        return real_client(**options, http_client=transport_client)

    monkeypatch.setattr(openai, "AsyncOpenAI", client)
    assert asyncio.run(authoring._ask("A saved week", "openai/gpt-4.1-mini")) == authoring.defaults()
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.method == "POST"
    assert request.headers["authorization"] == "Bearer router-http-test-key"
    body = json.loads(request.content)
    assert body["provider"] == {"require_parameters": True}
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"] == authoring.schema()
    assert body["messages"][-1] == {"role": "user", "content": "A saved week"}
    assert body["model"] == "openai/gpt-4.1-mini"
    assert all(item.is_closed for item in clients)


@pytest.mark.parametrize("response,match", [
    (completion("{}", finish_reason="length"), "incomplete"),
    (completion("{}", finish_reason=None), "incomplete"),
    (completion(None, refusal="Provider refusal text"), "declined"),
    (completion(None, finish_reason="content_filter"), "declined"),
    (completion("{bad"), "valid scenario"),
    (completion(None), "valid scenario"),
    (completion("[]"), "object"),
    (SimpleNamespace(choices=[]), "scenario"),
])
def test_incomplete_refused_and_malformed_model_responses_fail(monkeypatch, response, match) -> None:
    response_client(monkeypatch, response)
    with pytest.raises(authoring.ScenarioValidationError, match=match):
        asyncio.run(authoring._ask("A week", "model"))


@pytest.mark.parametrize("response", [
    SimpleNamespace(error={"message": "secret-provider-key internal details", "code": 502}, choices=[]),
    completion(None, error={"message": "secret-provider-key internal details"}),
    completion("secret-provider-key internal details", finish_reason="error"),
])
def test_http_success_envelope_provider_errors_are_sanitized(monkeypatch, response) -> None:
    response_client(monkeypatch, response)
    with pytest.raises(authoring.AuthoringUnavailableError) as error:
        asyncio.run(authoring._ask("A week", "model"))
    assert "secret-provider-key" not in str(error.value)
    assert "internal details" not in str(error.value)
    assert "router-test-key" not in str(error.value)
    assert "manually" in str(error.value)


def test_provider_error_does_not_expose_credentials_or_request_details(monkeypatch) -> None:
    import httpx
    import openai

    response_client(monkeypatch, openai.APIConnectionError(
        message="secret-key request details", request=httpx.Request("POST", "https://example.invalid")
    ))
    with pytest.raises(authoring.AuthoringUnavailableError) as error:
        asyncio.run(authoring._ask("A week", "model"))
    assert "secret-key" not in str(error.value)
    assert "router-test-key" not in str(error.value)
    assert "manually" in str(error.value)

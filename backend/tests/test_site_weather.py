"""Provider boundary and immutable cache contracts for site forecasts."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

import pytest

from plant_backend import site_weather as weather


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(weather, "CACHE_DIR", tmp_path / "cache")
    weather._MEMORY.clear()
    yield
    weather._MEMORY.clear()


def provider_payload():
    first = datetime(2026, 9, 15)
    return {
        "latitude": 45.64, "longitude": 13.78, "elevation": 18.0,
        "timezone": "GMT", "utc_offset_seconds": 0,
        "hourly_units": {"time": "iso8601", "shortwave_radiation": "W/m²",
                         "temperature_2m": "°C", "relative_humidity_2m": "%", "wind_speed_10m": "m/s",
                         "cloud_cover": "%", "cloud_cover_low": "%", "cloud_cover_mid": "%", "cloud_cover_high": "%"},
        "hourly": {
            "time": [(first + timedelta(hours=index)).isoformat(timespec="minutes") for index in range(264)],
            "shortwave_radiation": [float(index % 24) for index in range(264)],
            "temperature_2m": [20.0] * 264, "relative_humidity_2m": [60] * 264,
            "wind_speed_10m": [2.5] * 264,
            "cloud_cover": [0, None, 100] * 88,
            "cloud_cover_low": [0, None, 80] * 88,
            "cloud_cover_mid": [0, None, 40] * 88,
            "cloud_cover_high": [0, None, 20] * 88,
        },
    }


def mock_response(monkeypatch, payload=None):
    requests = []

    def urlopen(request, *, timeout):
        requests.append((request, timeout))
        return BytesIO(json.dumps(payload if payload is not None else provider_payload()).encode())

    monkeypatch.setattr(weather, "urlopen", urlopen)
    return requests


def fetch(**kwargs):
    return weather.fetch_forecast(45.6495, 13.7768, "Trieste", **kwargs)


def test_fetch_normalizes_preceding_hour_radiation_without_moving_values(monkeypatch):
    requests = mock_response(monkeypatch)
    result = fetch()
    query = parse_qs(urlparse(requests[0][0].full_url).query)
    assert query["forecast_days"] == ["11"]
    assert query["timezone"] == ["GMT"]
    assert query["wind_speed_unit"] == ["ms"]
    assert set(weather._PROVIDER_CLOUD_VARIABLES) <= set(query["hourly"][0].split(","))
    assert requests[0][1] == weather.REQUEST_TIMEOUT_S
    assert result["start_time"] == "2026-09-14T23:00:00Z"
    assert len(result["hours"]) == 264
    assert result["hours"][1]["time"] == "2026-09-15T00:00:00Z"
    assert result["hours"][1]["shortwave_w_m2"] == 1.0
    assert result["hours"][1]["wind_m_s"] == 2.5
    assert result["hours"][0]["cloud_cover_pct"] == 0
    assert result["hours"][1]["cloud_cover_pct"] is None
    assert result["hours"][2]["cloud_cover_pct"] == 100
    assert result["hours"][2]["cloud_cover_low_pct"] == 80
    assert result["hours"][2]["cloud_cover_mid_pct"] == 40
    assert result["hours"][2]["cloud_cover_high_pct"] == 20
    assert result["site"]["latitude"] == 45.6495  # requested site, not the provider grid centre
    assert result["site"]["elevation_m"] == 18.0
    assert result["source"]["kind"] == "live_forecast"
    assert result["source"]["forecast_issued_at"] is None


def test_repeated_requests_and_restarts_reuse_valid_cache_and_return_copies(monkeypatch):
    requests = mock_response(monkeypatch)
    first = fetch()
    first["hours"][0]["shortwave_w_m2"] = 999
    second = fetch()
    assert second["hours"][0]["shortwave_w_m2"] == 0
    assert second["source"]["kind"] == "cached_forecast"
    assert second["id"] == first["id"]
    weather._MEMORY.clear()
    third = fetch()
    assert third["id"] == first["id"]
    assert len(requests) == 1
    assert weather.get_forecast(first["id"])["id"] == first["id"]


def test_content_identity_ignores_acquisition_time_and_delivery_kind(monkeypatch):
    mock_response(monkeypatch)
    now = datetime(2026, 9, 15, tzinfo=timezone.utc)
    monkeypatch.setattr(weather, "_utc_now", lambda: now)
    first = fetch()
    monkeypatch.setattr(weather, "_utc_now", lambda: now + timedelta(hours=1))
    refreshed = fetch(refresh=True)
    assert refreshed["source"]["retrieved_at"] != first["source"]["retrieved_at"]
    assert refreshed["id"] == first["id"]
    assert weather._forecast_id(weather.get_forecast(first["id"])) == first["id"]


def test_expired_cache_does_not_silently_masquerade_as_current_forecast(monkeypatch):
    requests = mock_response(monkeypatch)
    now = datetime(2026, 9, 15, tzinfo=timezone.utc)
    monkeypatch.setattr(weather, "_utc_now", lambda: now)
    first = fetch()
    monkeypatch.setattr(weather, "_utc_now", lambda: now + timedelta(hours=7))
    assert fetch()["source"]["kind"] == "live_forecast"
    assert len(requests) == 2
    assert weather.get_forecast(first["id"])["source"]["kind"] == "cached_forecast"


@pytest.mark.parametrize("latitude,longitude", [(float("nan"), 15), (45, float("inf")),
                                               (True, 15), (33, 15), (73, 15), (45, -26), (45, 46)])
def test_rejects_invalid_or_outside_europe_coordinates_before_network(monkeypatch, latitude, longitude):
    requests = mock_response(monkeypatch)
    with pytest.raises(weather.SiteWeatherError):
        weather.fetch_forecast(latitude, longitude)
    assert not requests


@pytest.mark.parametrize("malformation", ["gap", "duplicate", "null", "nan", "infinity", "short", "mismatch", "units", "timezone", "negative_solar", "elevation"])
def test_rejects_incomplete_or_invalid_provider_data(monkeypatch, malformation):
    payload = provider_payload()
    if malformation == "gap":
        payload["hourly"]["time"][12] = "2026-09-15T13:00"
    elif malformation == "duplicate":
        payload["hourly"]["time"][12] = payload["hourly"]["time"][11]
    elif malformation in ("null", "nan", "infinity"):
        payload["hourly"]["shortwave_radiation"][100] = {"null": None, "nan": float("nan"), "infinity": float("inf")}[malformation]
    elif malformation == "short":
        for key in payload["hourly"]:
            payload["hourly"][key] = payload["hourly"][key][:263]
    elif malformation == "mismatch":
        payload["hourly"]["temperature_2m"].pop()
    elif malformation == "units":
        payload["hourly_units"]["wind_speed_10m"] = "km/h"
    elif malformation == "timezone":
        payload["utc_offset_seconds"] = 3600
    elif malformation == "negative_solar":
        payload["hourly"]["shortwave_radiation"][1] = -10
    elif malformation == "elevation":
        payload["elevation"] = None
    mock_response(monkeypatch, payload)
    with pytest.raises(weather.ForecastUnavailableError):
        fetch()
    assert not weather._MEMORY


@pytest.mark.parametrize("error", [URLError("DNS failed"), TimeoutError(), HTTPError("https://api.open-meteo.com", 429, "limited", {}, None)])
def test_provider_failures_are_actionable(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(weather, "urlopen", fail)
    with pytest.raises(weather.ForecastUnavailableError, match="saved example"):
        fetch()


def test_hash_lookup_rejects_paths_and_corrupted_weather(monkeypatch):
    mock_response(monkeypatch)
    record = fetch()
    path = weather.CACHE_DIR / f"{record['id']}.json"
    corrupt = deepcopy(record)
    corrupt["hours"][0]["shortwave_w_m2"] = 10
    path.write_text(json.dumps(corrupt))
    weather._MEMORY.clear()
    with pytest.raises(weather.ForecastUnavailableError):
        weather.get_forecast(record["id"])
    for bad_id in ("../../etc/passwd", "x" * 64, "abc", "A" * 64):
        with pytest.raises(weather.SiteWeatherError):
            weather.get_forecast(bad_id)


def test_memory_and_disk_caches_are_bounded(monkeypatch):
    mock_response(monkeypatch)
    monkeypatch.setattr(weather, "MAX_CACHE_ENTRIES", 2)
    for index in range(4):
        weather.fetch_forecast(45.5 + index / 100, 13.7)
    assert len(weather._MEMORY) == 2
    assert len(list(weather.CACHE_DIR.glob("*.json"))) == 2


def test_missing_provider_clouds_are_unavailable_without_filling_or_repeated_downloads(monkeypatch):
    payload = provider_payload()
    for key in weather._PROVIDER_CLOUD_VARIABLES:
        del payload["hourly"][key]
        del payload["hourly_units"][key]
    requests = mock_response(monkeypatch, payload)
    record = fetch()
    assert all(row[key] is None for row in record["hours"] for key in weather.CLOUD_COVER_FIELDS)
    assert fetch()["id"] == record["id"]
    assert len(requests) == 1


@pytest.mark.parametrize("provider_key", weather._PROVIDER_CLOUD_VARIABLES)
@pytest.mark.parametrize("bad_value", [-1, 101, True, "50", float("nan"), float("inf")])
def test_rejects_invalid_cloud_percentages(monkeypatch, provider_key, bad_value):
    payload = provider_payload()
    payload["hourly"][provider_key][1] = bad_value
    mock_response(monkeypatch, payload)
    with pytest.raises(weather.ForecastUnavailableError):
        fetch()
    assert not weather._MEMORY


@pytest.mark.parametrize("malformation", ["unit", "missing_unit", "short", "scalar", "null_array"])
def test_rejects_malformed_provided_cloud_arrays(monkeypatch, malformation):
    payload = provider_payload()
    if malformation == "unit":
        payload["hourly_units"]["cloud_cover"] = "fraction"
    elif malformation == "missing_unit":
        del payload["hourly_units"]["cloud_cover"]
    elif malformation == "short":
        payload["hourly"]["cloud_cover"].pop()
    elif malformation == "scalar":
        payload["hourly"]["cloud_cover"] = 50
    else:
        payload["hourly"]["cloud_cover"] = None
    mock_response(monkeypatch, payload)
    with pytest.raises(weather.ForecastUnavailableError):
        fetch()


@pytest.mark.parametrize("from_disk", [False, True])
def test_legacy_cache_upgrades_without_mutating_old_saved_content(monkeypatch, from_disk):
    requests = mock_response(monkeypatch)
    fresh = fetch()
    legacy = deepcopy(fresh)
    for row in legacy["hours"]:
        for key in weather.CLOUD_COVER_FIELDS:
            del row[key]
    legacy["id"] = weather._forecast_id(legacy)
    weather._MEMORY.clear()
    (weather.CACHE_DIR / f"{fresh['id']}.json").unlink()
    weather._remember(legacy)
    if from_disk:
        weather._MEMORY.clear()
    upgraded = fetch()
    assert len(requests) == 2
    assert upgraded["id"] != legacy["id"]
    saved_legacy = weather.get_forecast(legacy["id"])
    assert saved_legacy["hours"] == legacy["hours"]
    assert weather._forecast_id(saved_legacy) == legacy["id"]


def test_cloud_content_participates_in_snapshot_hash_and_cache_validation(monkeypatch):
    mock_response(monkeypatch)
    record = fetch()
    changed = deepcopy(record)
    changed["hours"][0]["cloud_cover_pct"] = 20
    assert weather._forecast_id(changed) != record["id"]
    changed["hours"][0]["cloud_cover_pct"] = 120
    changed["id"] = weather._forecast_id(changed)
    path = weather.CACHE_DIR / f"{changed['id']}.json"
    path.write_text(json.dumps(changed))
    assert weather._read_record(path) is None


def test_saved_examples_have_genuine_provenance_and_are_labelled_as_cached():
    examples = weather.list_examples()
    assert {item["id"] for item in examples} == {"seville", "trieste", "stockholm"}
    assert all(item["available"] for item in examples)
    for example in examples:
        if example["available"]:
            record = weather.load_example(example["id"])
            assert record["source"]["provider"] == "Open-Meteo"
            assert record["source"]["kind"] == "cached_forecast"
            assert record["source"]["forecast_issued_at"] is None
            assert record["source"]["url"].startswith("https://api.open-meteo.com/v1/forecast?")
            assert len(record["hours"]) >= 264
            assert weather.get_forecast(record["id"])["id"] == record["id"]

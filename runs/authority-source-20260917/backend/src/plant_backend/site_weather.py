"""Validated, reproducible European site forecasts from Open-Meteo.

Radiation timestamps are normalized to the START of the represented hour:
Open-Meteo labels its preceding-hour mean at the END of that hour. Other
weather variables retain the provider's sample at the end of that interval.
No missing weather is filled, and a cached forecast is never described as live.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import tempfile
from threading import RLock
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


EUROPE_BOUNDS = {"south": 34.0, "north": 72.0, "west": -25.0, "east": 45.0}
MIN_FORECAST_HOURS = 264
MAX_FORECAST_HOURS = 384
REQUEST_TIMEOUT_S = 20
MAX_RESPONSE_BYTES = 1_000_000
MAX_CACHE_ENTRIES = 32
CACHE_TTL = timedelta(hours=6)
CACHE_DIR = Path(os.environ.get(
    "PLANT_SITE_WEATHER_CACHE",
    str(Path(tempfile.gettempdir()) / f"plant-an-agent-site-forecasts-{os.getuid()}"),
))
EXAMPLE_DIR = Path(__file__).with_name("data") / "site_forecasts"
EXAMPLE_SITES = {
    "seville": {"name": "Seville, Spain", "latitude": 37.3891, "longitude": -5.9845},
    "trieste": {"name": "Trieste, Italy", "latitude": 45.6495, "longitude": 13.7768},
    "stockholm": {"name": "Stockholm, Sweden", "latitude": 59.3293, "longitude": 18.0686},
}
_MEMORY: OrderedDict[str, dict[str, Any]] = OrderedDict()
_LOCK = RLock()
_PROVIDER_VARIABLES = {
    "shortwave_radiation": ("shortwave_w_m2", 0.0, 1600.0, "W/m²"),
    "temperature_2m": ("temperature_c", -100.0, 70.0, "°C"),
    "relative_humidity_2m": ("humidity_pct", 0.0, 100.0, "%"),
    "wind_speed_10m": ("wind_m_s", 0.0, 150.0, "m/s"),
}
_PROVIDER_CLOUD_VARIABLES = {
    "cloud_cover": "cloud_cover_pct",
    "cloud_cover_low": "cloud_cover_low_pct",
    "cloud_cover_mid": "cloud_cover_mid_pct",
    "cloud_cover_high": "cloud_cover_high_pct",
}
CLOUD_COVER_FIELDS = tuple(_PROVIDER_CLOUD_VARIABLES.values())


class SiteWeatherError(ValueError):
    """Invalid coordinates, dataset, or forecast identifier."""


class ForecastUnavailableError(SiteWeatherError):
    """The provider or requested saved forecast is unavailable."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise SiteWeatherError("Forecast timestamps must be ISO dates.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SiteWeatherError("Forecast contains an invalid timestamp.") from exc
    # Provider requests always specify GMT, and its ISO timestamps are naive.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if parsed.utcoffset() != timedelta(0):
        raise SiteWeatherError("Forecast timestamps must use UTC.")
    return parsed


def _number(value: object, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise SiteWeatherError(f"{name} must be a finite number.")
    if not low <= value <= high:
        raise SiteWeatherError(f"{name} must lie between {low:g} and {high:g}.")
    return float(value)


def _coordinates(latitude: float, longitude: float) -> tuple[float, float]:
    lat = _number(latitude, "European latitude", EUROPE_BOUNDS["south"], EUROPE_BOUNDS["north"])
    lon = _number(longitude, "European longitude", EUROPE_BOUNDS["west"], EUROPE_BOUNDS["east"])
    return round(lat, 5), round(lon, 5)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _forecast_id(record: dict[str, Any]) -> str:
    # Acquisition/delivery metadata can change without changing weather content.
    content = {
        "schema": "site-weather-interval-start-v1",
        "site": record["site"],
        "source": {key: record["source"][key] for key in ("provider", "url", "model")},
        "start_time": record["start_time"],
        "hours": record["hours"],
        "assumptions": record["assumptions"],
    }
    return hashlib.sha256(_canonical(content).encode("utf-8")).hexdigest()


def _validate_hours(hours: object) -> list[dict[str, Any]]:
    if not isinstance(hours, list) or not MIN_FORECAST_HOURS <= len(hours) <= MAX_FORECAST_HOURS:
        raise SiteWeatherError(f"Forecast needs at least {MIN_FORECAST_HOURS} complete hourly intervals.")
    previous = None
    for row in hours:
        if not isinstance(row, dict):
            raise SiteWeatherError("Forecast contains a malformed hourly interval.")
        current = _timestamp(row.get("time"))
        if current.minute or current.second or current.microsecond:
            raise SiteWeatherError("Forecast timestamps must align to whole hours.")
        if previous is not None and current - previous != timedelta(hours=1):
            raise SiteWeatherError("Forecast has missing, duplicate, or unordered hours.")
        previous = current
        for key, low, high, _ in _PROVIDER_VARIABLES.values():
            _number(row.get(key), key, low, high)
        # Saved forecasts predate cloud downloads. Preserve their exact content
        # and hashes; absence, like a provider null, means unavailable.
        for key in CLOUD_COVER_FIELDS:
            if row.get(key) is not None:
                _number(row[key], key, 0.0, 100.0)
    return hours


def _normalize(payload: object, latitude: float, longitude: float, name: str, url: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("error"):
        raise SiteWeatherError("Open-Meteo returned an invalid forecast response.")
    if payload.get("utc_offset_seconds") != 0 or payload.get("timezone") not in ("GMT", "UTC"):
        raise SiteWeatherError("Open-Meteo returned weather outside the requested UTC timezone.")
    elevation = _number(payload.get("elevation"), "Elevation", -500.0, 9000.0)
    hourly = payload.get("hourly")
    units = payload.get("hourly_units")
    if not isinstance(hourly, dict) or not isinstance(units, dict) or not isinstance(hourly.get("time"), list):
        raise SiteWeatherError("Open-Meteo returned no usable hourly weather.")
    times = hourly["time"]
    for provider_key, (_, _, _, expected_unit) in _PROVIDER_VARIABLES.items():
        if units.get(provider_key) != expected_unit:
            raise SiteWeatherError(f"Open-Meteo returned unexpected units for {provider_key}.")
        if not isinstance(hourly.get(provider_key), list) or len(hourly[provider_key]) != len(times):
            raise SiteWeatherError(f"Open-Meteo returned incomplete {provider_key} coverage.")
    for provider_key in _PROVIDER_CLOUD_VARIABLES:
        if provider_key not in hourly:
            continue
        if units.get(provider_key) != "%":
            raise SiteWeatherError(f"Open-Meteo returned unexpected units for {provider_key}.")
        if not isinstance(hourly[provider_key], list) or len(hourly[provider_key]) != len(times):
            raise SiteWeatherError(f"Open-Meteo returned incomplete {provider_key} coverage.")
    hours = []
    for index, raw_time in enumerate(times):
        row = {"time": _iso(_timestamp(raw_time) - timedelta(hours=1))}
        for provider_key, (key, low, high, _) in _PROVIDER_VARIABLES.items():
            row[key] = _number(hourly[provider_key][index], key, low, high)
        for provider_key, key in _PROVIDER_CLOUD_VARIABLES.items():
            value = hourly[provider_key][index] if provider_key in hourly else None
            row[key] = None if value is None else _number(value, key, 0.0, 100.0)
        hours.append(row)
    _validate_hours(hours)
    record = {
        "site": {"name": name, "latitude": latitude, "longitude": longitude,
                 "elevation_m": elevation, "timezone": "UTC"},
        "source": {
            "provider": "Open-Meteo", "url": url, "retrieved_at": _iso(_utc_now()),
            "forecast_issued_at": None, "model": "Best Match (automatic model blend)",
            "kind": "live_forecast",
            "note": "Downloaded forecast; provider model issuance time is unavailable. "
                    "Weather uses a nearby model grid cell; elevation is provider terrain metadata.",
        },
        "start_time": hours[0]["time"], "hours": hours,
        "assumptions": [
            "Each time marks the start of a one-hour interval in UTC. Open-Meteo's preceding-hour "
            "mean radiation timestamp is shifted back one hour.",
            "Temperature, humidity, wind and cloud cover are provider samples at the end of each interval.",
            "Cloud cover percentages are forecast grid-cell coverage. Missing cloud data remains "
            "unavailable; individual rendered cloud shapes and positions are illustrative.",
            "This is a saved forecast projection, not observed weather or a historical backtest.",
            "The forecast begins at 23:00 UTC before the provider's first forecast date. "
            "Eleven downloaded days support ten simulation days plus one day of planning.",
            "Shortwave radiation is global horizontal irradiance; panel tilt, shading, snow and "
            "equipment temperature effects require additional models.",
            "Data attribution: Open-Meteo (https://open-meteo.com/), CC BY 4.0.",
        ],
    }
    record["id"] = _forecast_id(record)
    return record


def _cached(record: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(record)
    result["source"]["kind"] = "cached_forecast"
    result["source"]["note"] = (
        "Saved forecast downloaded on " + result["source"]["retrieved_at"]
        + "; replay uses those forecast dates. Provider model issuance time is unavailable. "
        "Elevation is provider terrain metadata."
    )
    return result


def _read_record(path: Path) -> dict[str, Any] | None:
    try:
        if path.stat().st_size > MAX_RESPONSE_BYTES:
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            return None
        _validate_hours(record["hours"])
        site, source = record["site"], record["source"]
        _coordinates(site["latitude"], site["longitude"])
        _number(site["elevation_m"], "Elevation", -500.0, 9000.0)
        _timestamp(source["retrieved_at"])
        if source["kind"] not in ("live_forecast", "cached_forecast") or site["timezone"] != "UTC":
            return None
        if record["start_time"] != record["hours"][0]["time"] or record["id"] != _forecast_id(record):
            return None
        return record
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return None


def _remember(record: dict[str, Any], *, persist: bool = True) -> None:
    with _LOCK:
        _MEMORY[record["id"]] = deepcopy(record)
        _MEMORY.move_to_end(record["id"])
        while len(_MEMORY) > MAX_CACHE_ENTRIES:
            _MEMORY.popitem(last=False)
        if not persist:
            return
        temporary_path: str | None = None
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE_DIR, suffix=".tmp", delete=False) as output:
                temporary_path = output.name
                output.write(_canonical(record))
            os.replace(temporary_path, CACHE_DIR / f"{record['id']}.json")
            paths = sorted(CACHE_DIR.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
            for path in paths[MAX_CACHE_ENTRIES:]:
                path.unlink(missing_ok=True)
        except OSError:
            # Read-only installs still work through the bounded memory cache.
            pass
        finally:
            if temporary_path:
                try:
                    Path(temporary_path).unlink(missing_ok=True)
                except OSError:
                    pass


def _recent(latitude: float, longitude: float, name: str) -> dict[str, Any] | None:
    with _LOCK:
        records = list(_MEMORY.values())
        try:
            paths = sorted(CACHE_DIR.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        except OSError:
            paths = []
        for path in paths[:MAX_CACHE_ENTRIES]:
            if path.stem not in _MEMORY:
                record = _read_record(path)
                if record:
                    records.append(record)
        matches = [record for record in records if
                   record["site"]["latitude"] == latitude and record["site"]["longitude"] == longitude
                   and record["site"]["name"] == name
                   and all(all(key in row for key in CLOUD_COVER_FIELDS) for row in record["hours"])
                   and timedelta(0) <= _utc_now() - _timestamp(record["source"]["retrieved_at"]) <= CACHE_TTL]
        return max(matches, key=lambda item: item["source"]["retrieved_at"], default=None)


def fetch_forecast(latitude: float, longitude: float, name: str = "", refresh: bool = False) -> dict[str, Any]:
    """Fetch eleven provider days; reuse a clearly labelled cache for six hours.

    Provider errors are explicit. There is no silent fallback to old examples.
    """
    latitude, longitude = _coordinates(latitude, longitude)
    if not isinstance(name, str) or len(name) > 120:
        raise SiteWeatherError("Site name must be text of at most 120 characters.")
    name = name.strip() or f"Selected site ({latitude:.4f}, {longitude:.4f})"
    if not refresh:
        record = _recent(latitude, longitude, name)
        if record:
            _remember(record, persist=False)
            return _cached(record)
    url = "https://api.open-meteo.com/v1/forecast?" + urlencode({
        "latitude": latitude, "longitude": longitude,
        "forecast_days": 11, "timezone": "GMT", "hourly": ",".join((*_PROVIDER_VARIABLES, *_PROVIDER_CLOUD_VARIABLES)),
        "wind_speed_unit": "ms", "temperature_unit": "celsius",
    })
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "PlantAnAgent/1.0"})
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ForecastUnavailableError("Open-Meteo response exceeds the supported size.")
        payload = json.loads(raw)
        record = _normalize(payload, latitude, longitude, name, url)
    except HTTPError as exc:
        raise ForecastUnavailableError(f"Open-Meteo returned HTTP {exc.code}. Try again or choose a saved example.") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ForecastUnavailableError("Could not reach Open-Meteo. Try again or choose a saved example.") from exc
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ForecastUnavailableError(f"Open-Meteo forecast is incomplete or invalid: {exc}") from exc
    _remember(record)
    return deepcopy(record)


def get_forecast(id: str) -> dict[str, Any]:
    """Retrieve immutable weather by content hash; arbitrary filesystem paths fail."""
    if not isinstance(id, str) or re.fullmatch(r"[a-f0-9]{64}", id) is None:
        raise SiteWeatherError("Forecast id must be a lowercase SHA-256 digest.")
    with _LOCK:
        record = _MEMORY.get(id)
        if record:
            _MEMORY.move_to_end(id)
            return _cached(record)
        record = _read_record(CACHE_DIR / f"{id}.json")
        if record is not None and record["id"] != id:
            record = None
        if record is None:
            for example in EXAMPLE_SITES:
                candidate = _read_record(EXAMPLE_DIR / f"{example}.json")
                if candidate and candidate["id"] == id:
                    record = candidate
                    break
        if record:
            _remember(record, persist=False)
            return _cached(record)
    raise ForecastUnavailableError("Saved forecast is unavailable. Download weather again or load a saved example.")


def list_examples() -> list[dict[str, Any]]:
    """List fixed example locations and whether genuine saved weather is present."""
    return [{"id": id, **site, "available": _read_record(EXAMPLE_DIR / f"{id}.json") is not None}
            for id, site in EXAMPLE_SITES.items()]


def load_example(id: str) -> dict[str, Any]:
    if id not in EXAMPLE_SITES:
        raise SiteWeatherError("Unknown site example.")
    record = _read_record(EXAMPLE_DIR / f"{id}.json")
    if record is None:
        raise ForecastUnavailableError("This saved example is unavailable. Download a live forecast for the site.")
    _remember(record, persist=False)
    return _cached(record)

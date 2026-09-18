"""Public projections and recording integrity for an active geographic site.

These helpers consume the retained downloaded snapshot and public plant frames.
They never inspect future simulated weather, faults or controller-private state.
"""

from __future__ import annotations

from datetime import timedelta, timezone, tzinfo
import math
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .site_simulation import SitePVProfile, _digest, _iso, _utc, scenario_solar_scale
from .site_weather import CLOUD_COVER_FIELDS, _number

RECORDING_SCHEMA = "plant-site-recording-v1"
EVIDENCE_LEVEL = (
    "forecast-based site simulation; checksum integrity only, "
    "not measured plant output or validated controller evidence"
)


def annotate_site_frame(
    frame: dict[str, Any],
    forecast: dict[str, Any],
    parameters: dict[str, Any],
    *,
    physics_dt_s: float,
) -> dict[str, Any]:
    """Attach date-aligned forecast context without relabeling it as clear sky."""
    elapsed_s = frame["clock"]["elapsed_hours"] * 3600
    start = _utc(forecast["start_time"])
    boundary = start + timedelta(seconds=elapsed_s)
    sample_offset_s = max(0.0, elapsed_s - physics_dt_s)
    hour = forecast["hours"][int(sample_offset_s // 3600)]
    weather = frame["weather"]
    scenario = parameters.get("scenario")
    known_scale = scenario_solar_scale(scenario, sample_offset_s, forecast_only=True)
    actual_scale = scenario_solar_scale(scenario, sample_offset_s)
    weather.update(
        reference_kind="forecast_pv",
        resource_factor=None,
        attenuation=None,
        spatial_basis="downloaded site forecast; cloud percentages describe the forecast model grid cell",
        stress_factor=actual_scale if scenario is not None else parameters["solar_scale"],
        site_sample={
            "time": _iso(boundary),
            "interval_start": hour["time"],
            "conditions_time": _iso(_utc(hour["time"]) + timedelta(hours=1)),
            "shortwave_w_m2": hour["shortwave_w_m2"],
            "temperature_c": hour.get("temperature_c"),
            "humidity_pct": hour.get("humidity_pct"),
            "wind_m_s": hour.get("wind_m_s"),
            **{key: hour.get(key) for key in CLOUD_COVER_FIELDS},
            "forecast_solar_kw": weather["clear_sky_kw"] * known_scale,
            "simulated_solar_kw": weather["available_kw"],
            "daylight": hour["shortwave_w_m2"] > 1.0,
        },
    )
    if scenario is not None:
        weather.update(
            scenario_name=scenario["name"],
            scenario_forecast_scale=known_scale,
            spatial_basis="Downloaded site weather with an authored sunlight schedule; cloud cover remains provider context.",
        )
    zone_name = forecast["site"].get("timezone") or "UTC"
    zone: tzinfo
    try:
        zone = ZoneInfo(zone_name)
    except (ValueError, TypeError, ZoneInfoNotFoundError):
        zone = timezone.utc
    frame["clock"].update(
        time_utc=_iso(boundary),
        local_time=boundary.astimezone(zone).isoformat(),
        timezone=zone_name,
    )
    source = forecast["source"]
    issued = source.get("forecast_issued_at")
    issue_age = (boundary - _utc(issued)).total_seconds() / 3600 if issued else None
    view_start = start + timedelta(hours=frame["forecasts"]["issued_at_hours"])
    frame["forecasts"].update(
        kind="fixed_snapshot_projection_view",
        forecast_id=forecast["id"],
        forecast_issued_at=issued,
        retrieved_at=source.get("retrieved_at"),
        view_start_utc=_iso(view_start),
        issue_age_hours=issue_age,
        uncertainty="deterministic scenario; no calibrated p10 interval",
        price_basis="off-grid; no imported or exported electricity",
    )
    frame["economics"]["assumption_note"] = (
        "Off-grid: no imported/exported electricity and zero marginal solar tariff. "
        "Equipment wear, service and inventory values remain illustrative assumptions."
    )
    frame["electricity"] = {
        "mode": "dedicated_pv_no_export",
        "marginal_eur_per_mwh": 0,
        "fixed_costs_included": False,
    }
    return frame


def task_identity(
    forecast: dict[str, Any], parameters: dict[str, Any], base_digest: str
) -> str:
    return _digest(
        {
            "kind": "site-workspace-v1",
            "forecast": forecast,
            "parameters": parameters,
            "physical_contract": base_digest,
        }
    )


def validate_site_recording(record: dict[str, Any]) -> dict[str, Any]:
    """Validate retained site boundaries without accepting research provenance."""
    snapshot, frames = record["snapshot"], record["frames"]
    if (
        record["evidence_level"] != EVIDENCE_LEVEL
        or snapshot.get("mode") != "site_forecast"
    ):
        raise ValueError("site recording mode or evidence level is invalid")
    forecast = snapshot.get("site_forecast")
    parameters = snapshot.get("site_parameters")
    if not isinstance(forecast, dict) or not isinstance(parameters, dict):
        raise ValueError("site recording must retain its forecast and parameters")
    profile = SitePVProfile.from_forecast(forecast)
    scenario = parameters.get("scenario")
    horizon = 240
    if scenario is not None:
        from .scenario_authoring import ScenarioSpec

        spec = ScenarioSpec.model_validate(scenario)
        if spec.unsupported:
            raise ValueError("site recording has unresolved scenario requests")
        horizon = spec.duration_days * 24
        if parameters["solar_kw"] != spec.solar_kw or parameters["battery_kwh"] != spec.battery_kwh:
            raise ValueError("scenario capacities differ from recording parameters")
    if (
        snapshot["environment"]["id"] != "plant-site-forecast-v1"
        or snapshot["environment"]["horizon_hours"] != horizon
        or snapshot["environment"].get("fixture_seed") is not None
        or not isinstance(frames, list)
        or not 1 <= len(frames) <= horizon * 4 + 1
        or frames[-1] != snapshot["frame"]
        or len(snapshot["history"]) != len(frames)
    ):
        raise ValueError("incomplete site recording boundaries")
    for index, (frame, history) in enumerate(
        zip(frames, snapshot["history"], strict=True)
    ):
        elapsed_h = index / 4
        if (
            frame["clock"]["decision_index"] != index
            or frame["clock"]["elapsed_hours"] != elapsed_h
            or frame["clock"]["decision_count"] != horizon * 4
            or frame["state"]["step"] != index * 15
            or history["elapsed_hours"] != elapsed_h
        ):
            raise ValueError("site recording decision sequence mismatch")
        weather = frame["weather"]
        sample = weather.get("site_sample", {})
        expected_time = _iso(profile.start_time + timedelta(hours=elapsed_h))
        forcing_offset = max(0.0, elapsed_h * 3600 - 60)
        hour = forecast["hours"][int(forcing_offset // 3600)]
        expected_power = profile.at(forcing_offset) * parameters["solar_kw"]
        expected_power *= scenario_solar_scale(scenario, forcing_offset, forecast_only=True)
        if scenario is not None:
            if (
                weather.get("scenario_name") != scenario["name"]
                or weather.get("scenario_forecast_scale") != scenario_solar_scale(scenario, forcing_offset, forecast_only=True)
                or weather.get("stress_factor") != scenario_solar_scale(scenario, forcing_offset)
                or [(item["due_hours"], item["requested_kg"]) for item in frame["delivery"]["commitments"]]
                != [(item["due_hour"], item["kg"]) for item in scenario["deliveries"]]
            ):
                raise ValueError("site recording scenario context mismatch")
        for key in CLOUD_COVER_FIELDS:
            if hour.get(key) is not None:
                _number(hour[key], key, 0.0, 100.0)
            if sample.get(key) is not None:
                _number(sample[key], key, 0.0, 100.0)
        cloud_mismatch = any(sample.get(key) != hour.get(key) for key in CLOUD_COVER_FIELDS)
        # Earlier site recordings have no separate instantaneous sample time.
        # Accept those immutable exports while checking every supplied timestamp.
        conditions_mismatch = "conditions_time" in sample and sample["conditions_time"] != _iso(
            _utc(hour["time"]) + timedelta(hours=1)
        )
        if (
            weather.get("reference_kind") != "forecast_pv"
            or weather.get("attenuation") is not None
            or sample.get("time") != expected_time
            or sample.get("interval_start") != hour["time"]
            or sample.get("shortwave_w_m2") != hour["shortwave_w_m2"]
            or cloud_mismatch
            or conditions_mismatch
            or not math.isclose(
                sample.get("forecast_solar_kw", -1), expected_power, abs_tol=1e-8
            )
            or frame["forecasts"].get("forecast_id") != forecast["id"]
            or frame["forecasts"].get("forecast_issued_at")
            != forecast["source"].get("forecast_issued_at")
        ):
            raise ValueError(
                "site recording weather context does not match its retained forecast"
            )
        for key, field in (
            ("solar_kw", "solar_available_kw"),
            ("clear_sky_kw", "solar_potential_kw"),
            ("battery_soc", "battery_soc"),
            ("h2_kg", "h2_level_kg"),
            ("co2_kg", "co2_level_kg"),
            ("ch4_kg", "ch4_level_kg"),
        ):
            if history[key] != frame["state"][field]:
                raise ValueError(
                    "site recording history differs from its plant boundary"
                )
    return record

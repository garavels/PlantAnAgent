"""Machine-readable observation/action contract for the weather V2 alpha."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Mapping

import numpy as np

from .. import spaces as v1

ENVIRONMENT_ID = "plant-env-v2-weather-alpha"
SPEC_VERSION = 2
STAGE = "weather-forecast-alpha"


_V2_SOURCES = {
    "time_of_day_sin": "ControlStateV2.boundary.time_of_day_h (current decision boundary)",
    "time_of_day_cos": "ControlStateV2.boundary.time_of_day_h (current decision boundary)",
    "solar_available": (
        "ControlStateV2.boundary.solar_available_kw "
        "(latest completed one-minute dispatch sample reported at the current "
        "decision boundary; reset uses the initial boundary sample)"
    ),
    "solar_potential": (
        "ControlStateV2.boundary.solar_potential_kw "
        "(clear-sky value sampled for the latest completed one-minute dispatch; "
        "reset uses the initial boundary sample)"
    ),
    "solar_output_factor": (
        "V2 controller telemetry solar_available_kw / solar_potential_kw for the "
        "latest completed one-minute dispatch sample; zero when clear-sky power "
        "is zero"
    ),
    "solar_trend": (
        "final one-minute solar dispatch samples reported at the current and "
        "previous decision boundaries"
    ),
    "battery_soc": "ControlStateV2.boundary.battery_soc (current boundary inventory)",
    "battery_power": (
        "ControlStateV2.previous_interval.battery_power_mean_kw "
        "(mean over the previous completed decision interval)"
    ),
    "battery_reserve_margin": (
        "ControlStateV2.boundary.battery_soc (current boundary inventory), "
        "Constraints.reserve_soc"
    ),
    "h2_fill": "ControlStateV2.boundary.h2_fill (current boundary inventory)",
    "co2_fill": "ControlStateV2.boundary.co2_fill (current boundary inventory)",
    "ch4_fill": "ControlStateV2.boundary.ch4_fill (current boundary inventory)",
    "electrolyser_load": (
        "ControlStateV2.previous_interval.electrolyser_load_mean "
        "(mean over the previous completed decision interval)"
    ),
    "electrolyser_online": (
        "ControlStateV2.boundary.electrolyser_online "
        "(status during the latest completed physical interval)"
    ),
    "electrolyser_health": (
        "ControlStateV2.boundary.health['electrolyser'] "
        "(condition during the latest completed physical interval)"
    ),
    "dac_load": (
        "ControlStateV2.previous_interval.dac_load_mean "
        "(mean over the previous completed decision interval)"
    ),
    "dac_online": (
        "ControlStateV2.boundary.dac_online "
        "(status during the latest completed physical interval)"
    ),
    "dac_health": (
        "ControlStateV2.boundary.health['dac'] "
        "(condition during the latest completed physical interval)"
    ),
    "methanator_load": (
        "ControlStateV2.previous_interval.methanator_load_mean "
        "(mean over the previous completed decision interval)"
    ),
    "methanator_online": (
        "ControlStateV2.boundary.methanator_online "
        "(status during the latest completed physical interval)"
    ),
    "methanator_health": (
        "ControlStateV2.boundary.health['methanator'] "
        "(condition during the latest completed physical interval)"
    ),
    "h2_production": (
        "ControlStateV2.previous_interval.h2_production_mean_kg_h "
        "(mean over the previous completed decision interval)"
    ),
    "co2_production": (
        "ControlStateV2.previous_interval.co2_capture_mean_kg_h "
        "(mean over the previous completed decision interval)"
    ),
    "ch4_production": (
        "ControlStateV2.previous_interval.ch4_production_mean_kg_h "
        "(mean over the previous completed decision interval)"
    ),
    "electrolyser_cap": "Constraints.electrolyser_limit_kw at current decision boundary",
    "dac_cap": "Constraints.dac_limit_kw at current decision boundary",
    "methanator_cap": "Constraints.methanator_limit_kw at current decision boundary",
    "support_cap": "Constraints.support_limit_kw at current decision boundary",
    "reserve_soc": "Constraints.reserve_soc at current decision boundary",
    "episode_progress": "PlantEnvV2 decision counter at current decision boundary",
}


def _v2_prefix() -> tuple[v1.ObservationField, ...]:
    """Retain V1 field order while declaring the V2 sampling semantics."""
    fields: list[v1.ObservationField] = []
    for field in v1.OBSERVATION_FIELDS:
        field = replace(field, source=_V2_SOURCES[field.name])
        if field.name == "solar_trend":
            field = replace(
                field,
                normalisation=(
                    "(latest dispatch-sample kw - previous decision-boundary "
                    "dispatch-sample kw) / solar.peak_kw, "
                    "clipped to [-1, 1]"
                ),
            )
        elif field.name == "episode_progress":
            field = replace(
                field,
                normalisation="decisions_taken / episode_decisions",
            )
        fields.append(field)
    return tuple(fields)


def _forecast_fields() -> tuple[v1.ObservationField, ...]:
    """Append mean then p10 for each canonical forecast window."""
    windows = (
        ("0_1h", "0-1h"),
        ("1_6h", "1-6h"),
        ("6_12h", "6-12h"),
        ("12_24h", "12-24h"),
    )
    fields: list[v1.ObservationField] = []
    for slug, window_name in windows:
        fields.extend(
            (
                v1.ObservationField(
                    f"solar_forecast_{slug}_mean",
                    (
                        f"ForecastIssueV2.windows['{window_name}']."
                        "mean_peak_power_fraction"
                    ),
                    "fraction of PV nameplate",
                    "identity (window-mean available PV kW / solar.peak_kw)",
                ),
                v1.ObservationField(
                    f"solar_forecast_{slug}_p10",
                    (
                        f"ForecastIssueV2.windows['{window_name}']."
                        "p10_peak_power_fraction"
                    ),
                    "fraction of PV nameplate",
                    (
                        "identity (10th ensemble percentile after member-wise "
                        "window aggregation, divided by solar.peak_kw)"
                    ),
                ),
            )
        )
    return tuple(fields)


OBSERVATION_FIELDS: tuple[v1.ObservationField, ...] = _v2_prefix() + (
    v1.ObservationField(
        "delivery_time_to_next",
        "DeliveryLedger.next_pending.due_step and current physics boundary",
        "delivery periods",
        "remaining physics ticks / delivery-period ticks, clipped to [0, 1]",
    ),
    v1.ObservationField(
        "delivery_next_amount",
        "DeliveryLedger.next_pending.requested_kg",
        "kg",
        "requested kg / CH4 tank capacity",
    ),
    v1.ObservationField(
        "delivery_inventory_coverage",
        (
            "ControlStateV2.boundary.ch4_level_kg (current boundary inventory), "
            "next requested delivery"
        ),
        "commitments",
        "CH4 kg / next requested kg, clipped to [0, 2]",
        high=2.0,
    ),
    v1.ObservationField(
        "delivery_fill_rate_to_date",
        "DeliveryLedger cumulative delivered/requested kg",
        "fraction",
        "delivered kg / requested kg; 1 before the first deadline",
    ),
) + _forecast_fields()

ACTION_FIELDS = v1.ACTION_FIELDS
OBSERVATION_NAMES = tuple(field.name for field in OBSERVATION_FIELDS)
ACTION_NAMES = tuple(field.name for field in ACTION_FIELDS)
OBSERVATION_SIZE = len(OBSERVATION_FIELDS)
ACTION_SIZE = len(ACTION_FIELDS)


def observation_low() -> np.ndarray:
    return np.asarray([field.low for field in OBSERVATION_FIELDS], dtype=np.float32)


def observation_high() -> np.ndarray:
    return np.asarray([field.high for field in OBSERVATION_FIELDS], dtype=np.float32)


def action_low() -> np.ndarray:
    return np.asarray([field.low for field in ACTION_FIELDS], dtype=np.float32)


def action_high() -> np.ndarray:
    return np.asarray([field.high for field in ACTION_FIELDS], dtype=np.float32)


def contract_payload() -> dict[str, object]:
    """Descriptor only; episode/business parameters live in the audit manifest."""
    return {
        "environment_id": ENVIRONMENT_ID,
        "spec_version": SPEC_VERSION,
        "stage": STAGE,
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
        "observation_fields": [field.to_dict() for field in OBSERVATION_FIELDS],
        "action_fields": [field.to_dict() for field in ACTION_FIELDS],
    }


def contract_digest(payload: Mapping[str, object] | None = None) -> str:
    canonical = json.dumps(
        contract_payload() if payload is None else payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


ENVIRONMENT_CONTRACT_DIGEST = contract_digest()


def describe() -> dict[str, object]:
    return {
        **contract_payload(),
        "environment_contract_digest": ENVIRONMENT_CONTRACT_DIGEST,
        "digest_scope": "observation/action interface only",
        "full_checkpoint_contract": "PlantEnvV2.checkpoint_contract",
        "limitations": [
            (
                "synthetic weather and forecast uncertainty are engineering "
                "fixtures, not calibrated site data"
            ),
            "mean and p10 forecast summaries do not expose a full predictive distribution",
            "memoryless V1 process equipment",
            "delivery objective and plant model are not a calibrated physical twin",
        ],
    }


__all__ = [
    "ACTION_FIELDS",
    "ACTION_NAMES",
    "ACTION_SIZE",
    "ENVIRONMENT_CONTRACT_DIGEST",
    "ENVIRONMENT_ID",
    "OBSERVATION_FIELDS",
    "OBSERVATION_NAMES",
    "OBSERVATION_SIZE",
    "SPEC_VERSION",
    "STAGE",
    "action_high",
    "action_low",
    "contract_digest",
    "contract_payload",
    "describe",
    "observation_high",
    "observation_low",
]

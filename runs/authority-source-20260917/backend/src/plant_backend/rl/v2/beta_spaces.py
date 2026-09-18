"""Machine-readable Gym interface for the cyclic stochastic-market beta."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from ..spaces import ObservationField
from . import setpoint_spaces
from .economics import MAX_ABS_TARIFF_EUR_PER_MWH


ENVIRONMENT_ID = "plant-env-v2-cyclic-market-beta-v1"
SPEC_VERSION = 1
STAGE = "cyclic-market-beta-v1"
PRICE_FORECAST_BIN_COUNT = 24
PRICE_QUANTILES = ("mean", "p10", "p90")


MARKET_OBSERVATION_FIELDS = (
    ObservationField(
        "market_price_current",
        "current realized public opportunity price",
        "EUR/MWh",
        f"price / {MAX_ABS_TARIFF_EUR_PER_MWH:g}, clipped to [-1, 1]",
        low=-1.0,
    ),
    *tuple(
        ObservationField(
            f"market_price_h{hour:02d}_{statistic}",
            (
                "current PublicPriceForecastIssueV1 "
                f"{statistic} over [{hour}, {hour + 1}) h from issue"
            ),
            "EUR/MWh",
            f"price / {MAX_ABS_TARIFF_EUR_PER_MWH:g}, clipped to [-1, 1]",
            low=-1.0,
        )
        for hour in range(PRICE_FORECAST_BIN_COUNT)
        for statistic in PRICE_QUANTILES
    ),
)

CYCLE_OBSERVATION_FIELDS = (
    ObservationField(
        "cycle_progress",
        "completed decisions in the current non-resetting 24 h cycle",
        "fraction",
        "decision_in_cycle / decisions_per_cycle",
    ),
    ObservationField(
        "cycle_battery_return_margin",
        "current battery energy minus non-ratcheting cycle target",
        "kWh",
        "signed margin / installed battery capacity, clipped to [-1, 1]",
        low=-1.0,
    ),
    ObservationField(
        "cycle_h2_return_margin",
        "current H2 inventory minus non-ratcheting cycle target",
        "kg",
        "signed margin / H2 capacity, clipped to [-1, 1]",
        low=-1.0,
    ),
    ObservationField(
        "cycle_co2_return_margin",
        "current CO2 inventory minus non-ratcheting cycle target",
        "kg",
        "signed margin / CO2 capacity, clipped to [-1, 1]",
        low=-1.0,
    ),
    ObservationField(
        "cycle_ch4_return_margin",
        "current saleable CH4 inventory minus non-ratcheting cycle target",
        "kg",
        "signed margin / CH4 capacity, clipped to [-1, 1]",
        low=-1.0,
    ),
)

OBSERVATION_FIELDS = (
    *setpoint_spaces.OBSERVATION_FIELDS,
    *MARKET_OBSERVATION_FIELDS,
    *CYCLE_OBSERVATION_FIELDS,
)
ACTION_FIELDS = setpoint_spaces.ACTION_FIELDS
OBSERVATION_NAMES = tuple(field.name for field in OBSERVATION_FIELDS)
ACTION_NAMES = tuple(field.name for field in ACTION_FIELDS)
OBSERVATION_SIZE = len(OBSERVATION_FIELDS)
ACTION_SIZE = len(ACTION_FIELDS)


def observation_low() -> NDArray[np.float32]:
    return np.asarray([field.low for field in OBSERVATION_FIELDS], dtype=np.float32)


def observation_high() -> NDArray[np.float32]:
    return np.asarray([field.high for field in OBSERVATION_FIELDS], dtype=np.float32)


def action_low() -> NDArray[np.float32]:
    return np.asarray(setpoint_spaces.action_low(), dtype=np.float32)


def action_high() -> NDArray[np.float32]:
    return np.asarray(setpoint_spaces.action_high(), dtype=np.float32)


def contract_payload() -> dict[str, object]:
    return {
        "environment_id": ENVIRONMENT_ID,
        "spec_version": SPEC_VERSION,
        "stage": STAGE,
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
        "observation_fields": [field.to_dict() for field in OBSERVATION_FIELDS],
        "action_fields": [field.to_dict() for field in ACTION_FIELDS],
        "physical_authority_contract": setpoint_spaces.ENVIRONMENT_CONTRACT_DIGEST,
        "market_information": {
            "current_realized_price": True,
            "forecast_horizon_h": 24,
            "forecast_bin_h": 1,
            "statistics_per_bin": list(PRICE_QUANTILES),
            "field_order": "current then chronological mean,p10,p90 triples",
        },
        "cycle_information": (
            "24 h progress plus signed battery/H2/CO2/CH4 margins to one "
            "non-ratcheting opening reference"
        ),
        "structured_information_schema": ("cyclic-market-controller-observation-v1"),
        "reward_unit": "EUR per controller interval",
        "constraint_output": "per-cycle resource return feasibility and cost vector",
    }


def contract_digest(payload: Mapping[str, object] | None = None) -> str:
    encoded = json.dumps(
        contract_payload() if payload is None else payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


ENVIRONMENT_CONTRACT_DIGEST = contract_digest()


def describe() -> dict[str, object]:
    return {
        **contract_payload(),
        "environment_contract_digest": ENVIRONMENT_CONTRACT_DIGEST,
        "digest_scope": "observation/action/online information interface only",
        "full_checkpoint_contract": ("PlantCyclicMarketEnvV2Beta.checkpoint_contract"),
        "limitations": [
            "market truth and forecasts are synthetic rather than site-calibrated",
            "cycle deficits are hard evaluation constraints, not hidden EUR penalties",
            "the first disturbance profile includes finite DAC outages only",
            "future disturbance schedules and market regimes are deliberately private",
            "no controller, commercial, field, HEPA, or language-layer result follows",
        ],
    }


__all__ = [
    "ACTION_FIELDS",
    "ACTION_NAMES",
    "ACTION_SIZE",
    "CYCLE_OBSERVATION_FIELDS",
    "ENVIRONMENT_CONTRACT_DIGEST",
    "ENVIRONMENT_ID",
    "MARKET_OBSERVATION_FIELDS",
    "OBSERVATION_FIELDS",
    "OBSERVATION_NAMES",
    "OBSERVATION_SIZE",
    "PRICE_FORECAST_BIN_COUNT",
    "PRICE_QUANTILES",
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

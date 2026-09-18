"""Machine-readable interface for the opt-in economic/wear setpoint task."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from ..spaces import ObservationField
from . import setpoint_spaces
from .economics import MAX_ABS_TARIFF_EUR_PER_MWH

ENVIRONMENT_ID = "plant-env-v2-economic-wear-alpha"
SPEC_VERSION = 1
STAGE = "economic-wear-alpha"

TARIFF_OBSERVATION_FIELDS = (
    ObservationField(
        "energy_price_current",
        "DayAheadTariffV1.price_at(current decision boundary)",
        "EUR/MWh",
        f"price / {MAX_ABS_TARIFF_EUR_PER_MWH:g}, clipped to [-1, 1]",
        low=-1.0,
    ),
    ObservationField(
        "energy_price_next_1h_mean",
        "DayAheadTariffV1 exact public mean over [t, t+1h)",
        "EUR/MWh",
        f"price / {MAX_ABS_TARIFF_EUR_PER_MWH:g}, clipped to [-1, 1]",
        low=-1.0,
    ),
    ObservationField(
        "energy_price_next_6h_min",
        "DayAheadTariffV1 exact public minimum over [t, t+6h)",
        "EUR/MWh",
        f"price / {MAX_ABS_TARIFF_EUR_PER_MWH:g}, clipped to [-1, 1]",
        low=-1.0,
    ),
    ObservationField(
        "energy_price_next_6h_mean",
        "DayAheadTariffV1 exact public mean over [t, t+6h)",
        "EUR/MWh",
        f"price / {MAX_ABS_TARIFF_EUR_PER_MWH:g}, clipped to [-1, 1]",
        low=-1.0,
    ),
    ObservationField(
        "energy_price_next_6h_max",
        "DayAheadTariffV1 exact public maximum over [t, t+6h)",
        "EUR/MWh",
        f"price / {MAX_ABS_TARIFF_EUR_PER_MWH:g}, clipped to [-1, 1]",
        low=-1.0,
    ),
)

OBSERVATION_FIELDS = setpoint_spaces.OBSERVATION_FIELDS + TARIFF_OBSERVATION_FIELDS
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
        "price_information": (
            "current deterministic day-ahead tariff plus exact public 1 h and 6 h "
            "summaries; no realized future physical state"
        ),
        "structured_information_schema": "economic-controller-observation-v1",
        "reward_unit": "EUR per controller interval",
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
    inherited_limitations = setpoint_spaces.describe().get("limitations")
    if not isinstance(inherited_limitations, list) or any(
        not isinstance(item, str) for item in inherited_limitations
    ):
        raise RuntimeError("setpoint limitations are not a text list")
    superseded = {
        (
            "the five-field share action expresses commitment intent implicitly through "
            "requested electrolyser power, with a synthetic one-percent zero-command deadband"
        ),
        "controller authority parity only; economic and wear pressure not added",
    }
    return {
        **contract_payload(),
        "environment_contract_digest": ENVIRONMENT_CONTRACT_DIGEST,
        "digest_scope": "observation/action/economic interface only",
        "full_checkpoint_contract": "PlantEconomicSetpointEnvV2.checkpoint_contract",
        "limitations": [
            *(item for item in inherited_limitations if item not in superseded),
            "tariff is a deterministic stress profile, not historical market data",
            "wear is an explicit linear exposure proxy, not calibrated degradation",
            "DAC and methanator life consumption are not yet monetized",
            "no controller outcome or commercial claim has been established",
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
    "TARIFF_OBSERVATION_FIELDS",
    "action_high",
    "action_low",
    "contract_digest",
    "contract_payload",
    "describe",
    "observation_high",
    "observation_low",
]

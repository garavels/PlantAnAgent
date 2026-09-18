"""Separately versioned Gym contract for four absolute process targets."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from ..spaces import ActionField
from . import spaces as legacy_spaces
from .setpoint_control import PROCESS_SETPOINT_ACTION_NAMES


ENVIRONMENT_ID = "plant-env-v2-common-setpoint-alpha"
SPEC_VERSION = 1
STAGE = "controller-authority-parity-alpha"

OBSERVATION_FIELDS = legacy_spaces.OBSERVATION_FIELDS
ACTION_FIELDS = (
    ActionField(
        "electrolyser_target",
        "ProcessSetpointTargetsV1.electrolyser_target_kw",
        "(a + 1) / 2 * active electrolyser limit",
    ),
    ActionField(
        "dac_target",
        "ProcessSetpointTargetsV1.dac_target_kw",
        "(a + 1) / 2 * active DAC limit",
    ),
    ActionField(
        "methanator_target",
        "ProcessSetpointTargetsV1.methanator_target_kw",
        "(a + 1) / 2 * active methanator limit",
    ),
    ActionField(
        "battery_charge_ceiling",
        "ProcessSetpointTargetsV1.battery_charge_ceiling_kw",
        "(a + 1) / 2 * battery charge rating",
    ),
)

OBSERVATION_NAMES = tuple(field.name for field in OBSERVATION_FIELDS)
ACTION_NAMES = tuple(field.name for field in ACTION_FIELDS)
OBSERVATION_SIZE = len(OBSERVATION_FIELDS)
ACTION_SIZE = len(ACTION_FIELDS)

if ACTION_NAMES != PROCESS_SETPOINT_ACTION_NAMES:
    raise RuntimeError("setpoint Gym action order differs from the common adapter")


def observation_low() -> NDArray[np.float32]:
    return np.asarray(legacy_spaces.observation_low(), dtype=np.float32)


def observation_high() -> NDArray[np.float32]:
    return np.asarray(legacy_spaces.observation_high(), dtype=np.float32)


def action_low() -> NDArray[np.float32]:
    return np.asarray([field.low for field in ACTION_FIELDS], dtype=np.float32)


def action_high() -> NDArray[np.float32]:
    return np.asarray([field.high for field in ACTION_FIELDS], dtype=np.float32)


def contract_payload() -> dict[str, object]:
    return {
        "environment_id": ENVIRONMENT_ID,
        "spec_version": SPEC_VERSION,
        "stage": STAGE,
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
        "observation_fields": [field.to_dict() for field in OBSERVATION_FIELDS],
        "action_fields": [field.to_dict() for field in ACTION_FIELDS],
        "allocation": (
            "four absolute targets -> reserve-aware common minute dispatcher -> "
            "independent guard -> plant"
        ),
        "battery_support": (
            "not a policy action; deterministic contemporaneous target shortfall "
            "bounded by the active reserve-aware safety envelope"
        ),
        "battery_charge_ceiling": (
            "caps the dispatcher's direct battery route; process power refused by "
            "the physical plant may still be recovered into the battery as spill"
        ),
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
    inherited_limitations = legacy_spaces.describe().get("limitations")
    if not isinstance(inherited_limitations, list) or any(
        not isinstance(item, str) for item in inherited_limitations
    ):
        raise RuntimeError("historical V2 limitations are not a text list")
    return {
        **contract_payload(),
        "environment_contract_digest": ENVIRONMENT_CONTRACT_DIGEST,
        "digest_scope": "observation/action/dispatch interface only",
        "full_checkpoint_contract": "PlantSetpointEnvV2.checkpoint_contract",
        "limitations": [
            *inherited_limitations,
            "controller authority parity only; economic and wear pressure not added",
            "no trained four-target policy has been evaluated",
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

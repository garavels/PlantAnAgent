"""Machine-readable observation/action contract for the delivery-only V2 alpha."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Mapping

import numpy as np

from .. import spaces as v1

ENVIRONMENT_ID = "plant-env-v2-delivery-alpha"
SPEC_VERSION = 1
STAGE = "delivery-only-alpha"


def _v2_prefix() -> tuple[v1.ObservationField, ...]:
    """Reuse V1 values only where V2 documents their changed sampling cadence."""
    fields: list[v1.ObservationField] = []
    for field in v1.OBSERVATION_FIELDS:
        if field.name == "solar_trend":
            field = replace(
                field,
                source="PlantState.solar_available_kw, previous V2 decision boundary",
                normalisation=(
                    "(kw - previous decision-boundary kw) / solar.peak_kw, "
                    "clipped to [-1, 1]"
                ),
            )
        elif field.name == "episode_progress":
            field = replace(
                field,
                source="PlantEnvV2 decision counter",
                normalisation="decisions_taken / episode_decisions",
            )
        fields.append(field)
    return tuple(fields)


OBSERVATION_FIELDS: tuple[v1.ObservationField, ...] = _v2_prefix() + (
    v1.ObservationField(
        "delivery_time_to_next",
        "DeliveryLedger.next_pending.due_step, simulator.step_index",
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
        "PlantState.ch4_level_kg, next requested delivery",
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
)

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
        "limitations": [
            "deterministic V1 clear-sky solar; no forecast uncertainty yet",
            "memoryless V1 process equipment",
            "delivery-only objective; not a calibrated physical twin",
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

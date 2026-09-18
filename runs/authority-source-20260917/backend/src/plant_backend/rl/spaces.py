"""The contract between a policy and the plant: what it sees, what it may say.

Both spaces are described *as data* -- a tuple of field specs each carrying its
source, its unit before normalisation and the formula used -- for three reasons:
the encoder cannot drift from the documentation, the tests can assert on the
declared bounds instead of hard-coded indices, and a checkpoint can record the
exact field order it was trained against (`checkpoints.py`), so loading a policy
into a plant whose observation changed fails loudly instead of quietly feeding
it nonsense.

Observation: 30 floats, mostly in [0, 1]. Action: 5 floats in [-1, 1], the range
Gaussian/tanh policies are comfortable in, converted to the simulator's shares
and kilowatts by `actions.py`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ObservationField:
    """One slot of the observation vector, and where its number comes from."""

    name: str
    #: Where the raw number is read: telemetry field, constraint, or the episode.
    source: str
    #: Unit *before* normalisation.
    unit: str
    #: How the raw number becomes the vector entry.
    normalisation: str
    low: float = 0.0
    high: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source": self.source,
            "unit": self.unit,
            "normalisation": self.normalisation,
            "low": self.low,
            "high": self.high,
        }


@dataclass(frozen=True)
class ActionField:
    """One slot of the action vector, and what the simulator does with it."""

    name: str
    #: What the number means once converted.
    target: str
    #: How the policy output becomes a simulator quantity.
    conversion: str
    low: float = -1.0
    high: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "target": self.target,
            "conversion": self.conversion,
            "low": self.low,
            "high": self.high,
        }


#: Order matters and is part of the checkpoint contract: never reorder in place,
#: append instead, and treat any change as a new observation version.
OBSERVATION_FIELDS: tuple[ObservationField, ...] = (
    ObservationField(
        "time_of_day_sin",
        "PlantState.time_of_day_h",
        "h",
        "sin(2*pi*h/24)",
        low=-1.0,
    ),
    ObservationField(
        "time_of_day_cos",
        "PlantState.time_of_day_h",
        "h",
        "cos(2*pi*h/24)",
        low=-1.0,
    ),
    ObservationField(
        "solar_available",
        "PlantState.solar_available_kw",
        "kW",
        "kw / solar.peak_kw",
    ),
    ObservationField(
        "solar_potential",
        "PlantState.solar_potential_kw",
        "kW",
        "kw / solar.peak_kw",
    ),
    ObservationField(
        "solar_output_factor",
        "PlantState.solar_output_factor",
        "fraction",
        "identity (already 0..1)",
    ),
    ObservationField(
        "solar_trend",
        "PlantState.solar_available_kw, previous step",
        "kW",
        "(kw - previous_kw) / solar.peak_kw, clipped to [-1, 1]",
        low=-1.0,
    ),
    ObservationField(
        "battery_soc",
        "PlantState.battery_soc",
        "fraction",
        "identity (already 0..1)",
    ),
    ObservationField(
        "battery_power",
        "PlantState.battery_power_kw",
        "kW (>0 charging)",
        "kw / battery.max_discharge_kw, clipped to [-1, 1]",
        low=-1.0,
    ),
    ObservationField(
        "battery_reserve_margin",
        "PlantState.battery_soc, Constraints.reserve_soc",
        "fraction",
        "soc - reserve_soc, clipped to [-1, 1]",
        low=-1.0,
    ),
    ObservationField("h2_fill", "PlantState.h2_fill", "fraction", "identity (already 0..1)"),
    ObservationField("co2_fill", "PlantState.co2_fill", "fraction", "identity (already 0..1)"),
    ObservationField("ch4_fill", "PlantState.ch4_fill", "fraction", "identity (already 0..1)"),
    ObservationField(
        "electrolyser_load",
        "PlantState.electrolyser_load_frac",
        "fraction of rating",
        "identity (already 0..1)",
    ),
    ObservationField(
        "electrolyser_online",
        "PlantState.electrolyser_online",
        "bool",
        "1.0 if online else 0.0",
    ),
    ObservationField(
        "electrolyser_health",
        "PlantState.health['electrolyser']",
        "fraction",
        "0 if offline else 1 - degradation",
    ),
    ObservationField(
        "dac_load", "PlantState.dac_load_frac", "fraction of rating", "identity (already 0..1)"
    ),
    ObservationField("dac_online", "PlantState.dac_online", "bool", "1.0 if online else 0.0"),
    ObservationField(
        "dac_health",
        "PlantState.health['dac']",
        "fraction",
        "0 if offline else 1 - degradation",
    ),
    ObservationField(
        "methanator_load",
        "PlantState.methanator_load_frac",
        "fraction of rating",
        "identity (already 0..1)",
    ),
    ObservationField(
        "methanator_online",
        "PlantState.methanator_online",
        "bool",
        "1.0 if online else 0.0",
    ),
    ObservationField(
        "methanator_health",
        "PlantState.health['methanator']",
        "fraction",
        "0 if offline else 1 - degradation",
    ),
    ObservationField(
        "h2_production",
        "PlantState.h2_production_kg_h",
        "kg/h",
        "kg_h / (electrolyser.rated_kw / specific_energy_kwh_per_kg)",
    ),
    ObservationField(
        "co2_production",
        "PlantState.co2_capture_kg_h",
        "kg/h",
        "kg_h / (dac.rated_kw / specific_energy_kwh_per_kg)",
    ),
    ObservationField(
        "ch4_production",
        "PlantState.ch4_production_kg_h",
        "kg/h",
        "kg_h / (methanator.rated_kw / specific_energy_kwh_per_kg)",
    ),
    ObservationField(
        "electrolyser_cap",
        "Constraints.electrolyser_limit_kw",
        "kW",
        "limit_kw / electrolyser.rated_kw",
    ),
    ObservationField(
        "dac_cap", "Constraints.dac_limit_kw", "kW", "limit_kw / dac.rated_kw"
    ),
    ObservationField(
        "methanator_cap",
        "Constraints.methanator_limit_kw",
        "kW",
        "limit_kw / methanator.rated_kw",
    ),
    ObservationField(
        "support_cap",
        "Constraints.support_limit_kw",
        "kW",
        "limit_kw / battery.max_discharge_kw",
    ),
    ObservationField(
        "reserve_soc", "Constraints.reserve_soc", "fraction", "identity (already 0..1)"
    ),
    ObservationField(
        "episode_progress",
        "PlantEnv step counter",
        "steps",
        "steps_taken / episode_steps",
    ),
)

#: Order matters for the same reason as the observation order.
ACTION_FIELDS: tuple[ActionField, ...] = (
    ActionField(
        "battery_share",
        "Action.battery_share",
        "(a + 1) / 2, then shares normalised by Action.sanitised if they sum above 1",
    ),
    ActionField(
        "electrolyser_share",
        "Action.electrolyser_share",
        "(a + 1) / 2, then shares normalised by Action.sanitised if they sum above 1",
    ),
    ActionField(
        "dac_share",
        "Action.dac_share",
        "(a + 1) / 2, then shares normalised by Action.sanitised if they sum above 1",
    ),
    ActionField(
        "methanator_share",
        "Action.methanator_share",
        "(a + 1) / 2, then shares normalised by Action.sanitised if they sum above 1",
    ),
    ActionField(
        "battery_support",
        "Action.battery_support_kw",
        "(a + 1) / 2 * Constraints.support_limit_kw(config)",
    ),
)

OBSERVATION_NAMES: tuple[str, ...] = tuple(field.name for field in OBSERVATION_FIELDS)
ACTION_NAMES: tuple[str, ...] = tuple(field.name for field in ACTION_FIELDS)
OBSERVATION_SIZE = len(OBSERVATION_FIELDS)
ACTION_SIZE = len(ACTION_FIELDS)
#: Bumped whenever the meaning of a slot changes rather than a slot being added.
SPEC_VERSION = 1
#: Stable identity for this observation/action interface. It deliberately does
#: not promise compatibility of the reward function or plant dynamics.
ENVIRONMENT_ID = "plant-env-v1"


def observation_low() -> np.ndarray:
    return np.array([field.low for field in OBSERVATION_FIELDS], dtype=np.float32)


def observation_high() -> np.ndarray:
    return np.array([field.high for field in OBSERVATION_FIELDS], dtype=np.float32)


def action_low() -> np.ndarray:
    return np.array([field.low for field in ACTION_FIELDS], dtype=np.float32)


def action_high() -> np.ndarray:
    return np.array([field.high for field in ACTION_FIELDS], dtype=np.float32)


def contract_payload() -> dict[str, object]:
    """Return the machine-readable observation/action contract to be hashed.

    The digest covers the ordered fields and their semantics, but intentionally
    does not cover rewards, episode construction, or the simulator's dynamics.
    """
    return {
        "spec_version": SPEC_VERSION,
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
        "observation_fields": [field.to_dict() for field in OBSERVATION_FIELDS],
        "action_fields": [field.to_dict() for field in ACTION_FIELDS],
    }


def contract_digest(payload: Mapping[str, object] | None = None) -> str:
    """Return the SHA-256 of a canonical JSON representation of the contract."""
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
    """Describe this observation/action interface, including its identity."""
    return {
        "environment_id": ENVIRONMENT_ID,
        "environment_contract_digest": ENVIRONMENT_CONTRACT_DIGEST,
        **contract_payload(),
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
    "ActionField",
    "ObservationField",
    "action_high",
    "action_low",
    "contract_digest",
    "contract_payload",
    "describe",
    "observation_high",
    "observation_low",
]

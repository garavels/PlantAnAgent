"""What HEPA is allowed to look at: the observed plant, as a vector.

HEPA reads *telemetry*, the same distortable observation the controller and the
control room read (`PlantSimulator.get_telemetry()`), never the simulator's
private truth. That is deliberate: a predictor trained on hidden state would
learn to depend on numbers no instrument can produce.

The schema is data, in the spirit of `rl/spaces.py`, with one difference: each
feature carries the function that computes it, so the documented formula and the
number in the tensor cannot drift apart.

    encoder = FeatureEncoder(PlantConfig())
    vector = encoder.encode(telemetry)          # (S,) float32, mostly 0..1

Two normalisation layers sit on top of each other and answer different questions:

1. *Physical scaling*, here: every channel divided by the thing that bounds it
   (rated power, tank capacity, peak generation), so a feature means the same
   fraction whatever the plant is sized like. Time of day becomes sin/cos so
   midnight is next to 23:59.
2. *Dataset standardisation*, `NormalisationStats`: per-feature mean and standard
   deviation measured on the training split only and carried in the checkpoint,
   so pretraining, fine-tuning and serving see identically scaled inputs.

The model itself adds per-context instance normalisation (the paper's choice),
which removes the slow offset inside one window; the dataset statistics remain
useful because they keep the *variances* of unlike channels comparable.

Nothing in this module knows about events, labels or horizons: the feature vector
must never contain the answer (see `data/labels.py`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from plant_sim import Action, PlantConfig, PlantState
from plant_sim.state import CRITICAL, DEGRADED, NORMAL, OFFLINE, ComponentHealth

#: Bumped when the meaning of a slot changes rather than a slot being appended.
#: A checkpoint records it and refuses to run against a different one.
SCHEMA_VERSION = 1

#: Ordinal severity of the plant-level status. Ordered on purpose: a network
#: reading a single monotone number learns "worse" without four one-hot slots.
_STATUS_SEVERITY: dict[str, float] = {
    NORMAL: 0.0,
    DEGRADED: 1.0 / 3.0,
    CRITICAL: 2.0 / 3.0,
    OFFLINE: 1.0,
}


@dataclass(frozen=True)
class Scales:
    """The plant's own bounds, precomputed once: what each channel is divided by."""

    solar_peak_kw: float
    battery_capacity_kwh: float
    battery_max_discharge_kw: float
    electrolyser_rated_kw: float
    dac_rated_kw: float
    methanator_rated_kw: float
    h2_max_kg_h: float
    co2_max_kg_h: float
    ch4_max_kg_h: float
    electrolyser_temperature_c: float

    @classmethod
    def of(cls, config: PlantConfig) -> "Scales":
        electrolyser = config.electrolyser
        return cls(
            solar_peak_kw=config.solar.peak_kw,
            battery_capacity_kwh=config.battery.capacity_kwh,
            battery_max_discharge_kw=config.battery.max_discharge_kw,
            electrolyser_rated_kw=electrolyser.rated_kw,
            dac_rated_kw=config.dac.rated_kw,
            methanator_rated_kw=config.methanator.rated_kw,
            h2_max_kg_h=_ratio(electrolyser.rated_kw, electrolyser.specific_energy_kwh_per_kg),
            co2_max_kg_h=_ratio(config.dac.rated_kw, config.dac.specific_energy_kwh_per_kg),
            ch4_max_kg_h=_ratio(
                config.methanator.rated_kw, config.methanator.specific_energy_kwh_per_kg
            ),
            electrolyser_temperature_c=(
                electrolyser.nominal_temperature_c + electrolyser.fault_temperature_rise_c
            ),
        )


@dataclass(frozen=True)
class TelemetryFeature:
    """One slot of the feature vector: where it comes from and how it is scaled."""

    name: str
    source: str
    unit: str
    formula: str
    extract: Callable[[PlantState, Scales], float]

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "source": self.source,
            "unit": self.unit,
            "formula": self.formula,
        }


def _ratio(value: float, reference: float) -> float:
    return value / reference if reference > 0.0 else 0.0


def _flag(value: bool) -> float:
    return 1.0 if value else 0.0


def _health(state: PlantState, unit: str) -> float:
    """1.0 healthy, 0.0 offline, in between while derated. Reported, not true."""
    health = state.health.get(unit)
    if health is None:
        return 1.0
    return 0.0 if health.offline else max(0.0, 1.0 - health.degradation)


def _day_angle(state: PlantState) -> float:
    return 2.0 * math.pi * state.time_of_day_h / 24.0


#: Order matters: a checkpoint records these names, in order, and refuses a plant
#: whose schema no longer matches (`checkpoints.py`).
FEATURES: tuple[TelemetryFeature, ...] = (
    TelemetryFeature(
        "time_of_day_sin",
        "PlantState.time_of_day_h",
        "h",
        "sin(2*pi*h/24)",
        lambda state, _: math.sin(_day_angle(state)),
    ),
    TelemetryFeature(
        "time_of_day_cos",
        "PlantState.time_of_day_h",
        "h",
        "cos(2*pi*h/24)",
        lambda state, _: math.cos(_day_angle(state)),
    ),
    TelemetryFeature(
        "solar_available",
        "PlantState.solar_available_kw",
        "kW",
        "kw / solar.peak_kw",
        lambda state, scales: _ratio(state.solar_available_kw, scales.solar_peak_kw),
    ),
    TelemetryFeature(
        "solar_potential",
        "PlantState.solar_potential_kw",
        "kW",
        "kw / solar.peak_kw",
        lambda state, scales: _ratio(state.solar_potential_kw, scales.solar_peak_kw),
    ),
    TelemetryFeature(
        "solar_output_factor",
        "PlantState.solar_output_factor",
        "fraction",
        "identity (already 0..1)",
        lambda state, _: state.solar_output_factor,
    ),
    TelemetryFeature(
        "solar_curtailed",
        "PlantState.solar_curtailed_kw",
        "kW",
        "kw / solar.peak_kw",
        lambda state, scales: _ratio(state.solar_curtailed_kw, scales.solar_peak_kw),
    ),
    TelemetryFeature(
        "battery_soc",
        "PlantState.battery_soc",
        "fraction",
        "identity (already 0..1)",
        lambda state, _: state.battery_soc,
    ),
    TelemetryFeature(
        "battery_power",
        "PlantState.battery_power_kw",
        "kW",
        "kw / battery.max_discharge_kw (signed: >0 charging)",
        lambda state, scales: _ratio(state.battery_power_kw, scales.battery_max_discharge_kw),
    ),
    TelemetryFeature(
        "battery_support",
        "PlantState.battery_to_electrolyser_kw",
        "kW",
        "kw / battery.max_discharge_kw",
        lambda state, scales: _ratio(
            state.battery_to_electrolyser_kw, scales.battery_max_discharge_kw
        ),
    ),
    TelemetryFeature(
        "electrolyser_power",
        "PlantState.electrolyser_power_kw",
        "kW",
        "kw / electrolyser.rated_kw",
        lambda state, scales: _ratio(state.electrolyser_power_kw, scales.electrolyser_rated_kw),
    ),
    TelemetryFeature(
        "electrolyser_load",
        "PlantState.electrolyser_load_frac",
        "fraction",
        "identity (already 0..1)",
        lambda state, _: state.electrolyser_load_frac,
    ),
    TelemetryFeature(
        "electrolyser_online",
        "PlantState.electrolyser_online",
        "bool",
        "1.0 if online else 0.0",
        lambda state, _: _flag(state.electrolyser_online),
    ),
    TelemetryFeature(
        "electrolyser_efficiency",
        "PlantState.electrolyser_efficiency",
        "fraction",
        "identity (already 0..1)",
        lambda state, _: state.electrolyser_efficiency,
    ),
    TelemetryFeature(
        "electrolyser_temperature",
        "PlantState.electrolyser_temperature_c",
        "degC",
        "degC / (nominal + fault_rise)",
        lambda state, scales: _ratio(
            state.electrolyser_temperature_c, scales.electrolyser_temperature_c
        ),
    ),
    TelemetryFeature(
        "electrolyser_health",
        "PlantState.health['electrolyser']",
        "fraction",
        "0 if offline else 1 - degradation",
        lambda state, _: _health(state, "electrolyser"),
    ),
    TelemetryFeature(
        "h2_production",
        "PlantState.h2_production_kg_h",
        "kg/h",
        "kg_h / (electrolyser.rated_kw / specific_energy_kwh_per_kg)",
        lambda state, scales: _ratio(state.h2_production_kg_h, scales.h2_max_kg_h),
    ),
    TelemetryFeature(
        "h2_fill",
        "PlantState.h2_fill",
        "fraction",
        "identity (already 0..1, a drifting sensor may exceed it)",
        lambda state, _: state.h2_fill,
    ),
    TelemetryFeature(
        "dac_power",
        "PlantState.dac_power_kw",
        "kW",
        "kw / dac.rated_kw",
        lambda state, scales: _ratio(state.dac_power_kw, scales.dac_rated_kw),
    ),
    TelemetryFeature(
        "dac_load",
        "PlantState.dac_load_frac",
        "fraction",
        "identity (already 0..1)",
        lambda state, _: state.dac_load_frac,
    ),
    TelemetryFeature(
        "dac_online",
        "PlantState.dac_online",
        "bool",
        "1.0 if online else 0.0",
        lambda state, _: _flag(state.dac_online),
    ),
    TelemetryFeature(
        "dac_health",
        "PlantState.health['dac']",
        "fraction",
        "0 if offline else 1 - degradation",
        lambda state, _: _health(state, "dac"),
    ),
    TelemetryFeature(
        "co2_capture",
        "PlantState.co2_capture_kg_h",
        "kg/h",
        "kg_h / (dac.rated_kw / specific_energy_kwh_per_kg)",
        lambda state, scales: _ratio(state.co2_capture_kg_h, scales.co2_max_kg_h),
    ),
    TelemetryFeature(
        "co2_fill",
        "PlantState.co2_fill",
        "fraction",
        "identity (already 0..1, a drifting sensor may exceed it)",
        lambda state, _: state.co2_fill,
    ),
    TelemetryFeature(
        "methanator_power",
        "PlantState.methanator_power_kw",
        "kW",
        "kw / methanator.rated_kw",
        lambda state, scales: _ratio(state.methanator_power_kw, scales.methanator_rated_kw),
    ),
    TelemetryFeature(
        "methanator_load",
        "PlantState.methanator_load_frac",
        "fraction",
        "identity (already 0..1)",
        lambda state, _: state.methanator_load_frac,
    ),
    TelemetryFeature(
        "methanator_online",
        "PlantState.methanator_online",
        "bool",
        "1.0 if online else 0.0",
        lambda state, _: _flag(state.methanator_online),
    ),
    TelemetryFeature(
        "methanator_health",
        "PlantState.health['methanator']",
        "fraction",
        "0 if offline else 1 - degradation",
        lambda state, _: _health(state, "methanator"),
    ),
    TelemetryFeature(
        "ch4_production",
        "PlantState.ch4_production_kg_h",
        "kg/h",
        "kg_h / (methanator.rated_kw / specific_energy_kwh_per_kg)",
        lambda state, scales: _ratio(state.ch4_production_kg_h, scales.ch4_max_kg_h),
    ),
    TelemetryFeature(
        "ch4_fill",
        "PlantState.ch4_fill",
        "fraction",
        "identity (already 0..1, a drifting sensor may exceed it)",
        lambda state, _: state.ch4_fill,
    ),
    TelemetryFeature(
        "plant_status",
        "PlantState.plant_status",
        "category",
        "normal 0, degraded 1/3, critical 2/3, offline 1",
        lambda state, _: _STATUS_SEVERITY.get(state.plant_status, 0.0),
    ),
)

FEATURE_NAMES: tuple[str, ...] = tuple(feature.name for feature in FEATURES)
FEATURE_COUNT = len(FEATURES)


class FeatureEncoder:
    """`PlantState` telemetry -> the physically scaled feature vector, in order."""

    def __init__(self, config: PlantConfig | None = None) -> None:
        self.config = config or PlantConfig()
        self.scales = Scales.of(self.config)

    @property
    def names(self) -> tuple[str, ...]:
        return FEATURE_NAMES

    def encode(self, state: PlantState) -> np.ndarray:
        """One observation -> `(FEATURE_COUNT,)` float32. Never NaN, never inf."""
        values = [feature.extract(state, self.scales) for feature in FEATURES]
        vector = np.asarray(values, dtype=np.float32)
        return np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)

    def encode_many(self, states: Iterable[PlantState]) -> np.ndarray:
        """A sequence of observations -> `(T, FEATURE_COUNT)` float32."""
        rows = [self.encode(state) for state in states]
        if not rows:
            return np.zeros((0, FEATURE_COUNT), dtype=np.float32)
        return np.stack(rows).astype(np.float32)


@dataclass(frozen=True)
class NormalisationStats:
    """Per-feature mean and standard deviation, measured on the training split.

    Carried by the dataset manifest *and* by every checkpoint, because a model
    served with different statistics than it was trained with is silently wrong.
    """

    names: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]

    #: Below this a channel is treated as constant and left unscaled: dividing by
    #: a near-zero spread turns numerical noise into a large input.
    MIN_STD = 1e-6

    def __post_init__(self) -> None:
        if not (len(self.names) == len(self.mean) == len(self.std)):
            raise ValueError("normalisation statistics must have one entry per feature")

    @classmethod
    def identity(cls, names: Sequence[str] = FEATURE_NAMES) -> "NormalisationStats":
        """Leave the features as the schema scaled them. The honest default."""
        return cls(
            names=tuple(names),
            mean=tuple(0.0 for _ in names),
            std=tuple(1.0 for _ in names),
        )

    @classmethod
    def fit(
        cls, windows: Iterable[np.ndarray], names: Sequence[str] = FEATURE_NAMES
    ) -> "NormalisationStats":
        """Measure statistics over `(T, S)` blocks, streaming, in one pass."""
        count = 0
        total = np.zeros(len(names), dtype=np.float64)
        total_squared = np.zeros(len(names), dtype=np.float64)
        for block in windows:
            values = np.asarray(block, dtype=np.float64).reshape(-1, len(names))
            count += values.shape[0]
            total += values.sum(axis=0)
            total_squared += np.square(values).sum(axis=0)
        if count == 0:
            return cls.identity(names)
        mean = total / count
        variance = np.maximum(total_squared / count - np.square(mean), 0.0)
        return cls(
            names=tuple(names),
            mean=tuple(float(value) for value in mean),
            std=tuple(float(value) for value in np.sqrt(variance)),
        )

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Standardise a `(..., S)` array. Constant channels are passed through."""
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != len(self.names):
            raise ValueError(
                f"expected {len(self.names)} features, got {array.shape[-1]}"
            )
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        scale = np.where(std < self.MIN_STD, np.float32(1.0), std)
        return ((array - mean) / scale).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "mean": list(self.mean),
            "std": list(self.std),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "NormalisationStats":
        if not raw:
            return cls.identity()
        return cls(
            names=tuple(str(name) for name in raw.get("names", FEATURE_NAMES)),
            mean=tuple(float(value) for value in raw["mean"]),
            std=tuple(float(value) for value in raw["std"]),
        )


def state_from_frame(frame: dict[str, Any]) -> PlantState:
    """A serialised telemetry frame back into a `PlantState`.

    The backend keeps its rolling history as `PlantState.to_dict()` frames (that is
    what the API and the stream send), so serving a prediction from the live plant
    means going back the other way. Derived keys (`time_h`, `clock`, ...) are
    recomputed by `PlantState` itself and are dropped here rather than trusted.

    Unknown keys are ignored: a frame from a newer build stays readable, and a frame
    missing a channel gets that channel's documented default instead of an error --
    the alternative is refusing to predict because a field was added.
    """
    fields = {field.name: field for field in dataclass_fields(PlantState)}
    values: dict[str, Any] = {}
    for key, value in frame.items():
        if key not in fields or key in ("health", "action"):
            continue
        values[key] = value
    health = frame.get("health") or {}
    if isinstance(health, dict):
        values["health"] = {
            name: entry if isinstance(entry, ComponentHealth) else ComponentHealth(**entry)
            for name, entry in health.items()
        }
    action = frame.get("action")
    if isinstance(action, dict):
        values["action"] = Action(**action)
    elif isinstance(action, Action):
        values["action"] = action
    return PlantState(**values)


def describe() -> dict[str, Any]:
    """The whole feature contract as JSON: what the docs, the API and tests read."""
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_count": FEATURE_COUNT,
        "features": [feature.to_dict() for feature in FEATURES],
    }


__all__ = [
    "FEATURES",
    "FEATURE_COUNT",
    "FEATURE_NAMES",
    "SCHEMA_VERSION",
    "FeatureEncoder",
    "NormalisationStats",
    "Scales",
    "TelemetryFeature",
    "describe",
    "state_from_frame",
]

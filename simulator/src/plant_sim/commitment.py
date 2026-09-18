"""Technology-labelled commitment dynamics for the opt-in electrolyser model.

The profile is deliberately separate from :mod:`plant_sim.config`: enabling a
stateful scheduling archetype must not change the stable V1 plant configuration
or state payloads.  Both the profile and every state/snapshot value are frozen,
so callers can retain them as audit records without observing later mutation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from enum import Enum
from numbers import Real
from typing import Any


COMMITMENT_DYNAMICS_VERSION = "electrolyser-commitment-alpha-v3"
COMMITMENT_INTENT_DEADBAND_FRAC = 0.01
LEGACY_NOMINAL_SPECIFIC_ENERGY_KWH_PER_KG = 52.0


class ElectrolyserMode(str, Enum):
    """Mutually exclusive operating modes at a simulation boundary."""

    OFF = "off"
    COLD_STARTING = "cold_starting"
    RUNNING = "running"
    HOT_STANDBY = "hot_standby"
    HOT_STARTING = "hot_starting"
    COOLDOWN = "cooldown"


_MODE_ORDER = tuple(ElectrolyserMode)


def _require_text(name: str, value: Any) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _require_number(
    name: str,
    value: Any,
    *,
    low: float | None = None,
    high: float | None = None,
    low_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if low is not None:
        invalid = number < low if low_inclusive else number <= low
        if invalid:
            relation = ">=" if low_inclusive else ">"
            raise ValueError(f"{name} must be {relation} {low}")
    if high is not None and number > high:
        raise ValueError(f"{name} must be <= {high}")
    return number


@dataclass(frozen=True)
class ElectrolyserCommitmentProfile:
    """Immutable parameters for one control-oriented equipment archetype.

    ``calibration_basis`` is metadata with scientific meaning, not decoration:
    the default states plainly that this is a literature-bounded synthetic
    benchmark and not a vendor-calibrated equipment model.
    """

    profile_id: str = "ael_pressurised_fixed_bop_alpha_v1"
    technology: str = "pressurised alkaline"
    calibration_basis: str = "literature-bounded/synthetic; not vendor-calibrated"
    dynamics_version: str = COMMITMENT_DYNAMICS_VERSION

    cold_start_s: float = 3_600.0
    hot_start_s: float = 300.0
    start_load_frac: float = 0.20
    min_stable_load_frac: float = 0.40
    hot_standby_load_frac: float = 0.02
    cooldown_s: float = 900.0
    min_up_s: float = 3_600.0
    min_down_s: float = 1_800.0
    ramp_up_frac_per_min: float = 0.25
    ramp_down_frac_per_min: float = 0.25
    shortfall_grace_s: float = 120.0
    low_load_threshold_frac: float = 0.60
    nominal_specific_energy_kwh_per_kg: float = (
        LEGACY_NOMINAL_SPECIFIC_ENERGY_KWH_PER_KG
    )
    specific_energy_multiplier_anchors: tuple[tuple[float, float], ...] = (
        (0.40, 1.12),
        (0.60, 1.06),
        (0.80, 1.02),
        (1.00, 1.00),
    )

    def __post_init__(self) -> None:
        for name in ("profile_id", "technology", "calibration_basis", "dynamics_version"):
            _require_text(name, getattr(self, name))
        if self.dynamics_version != COMMITMENT_DYNAMICS_VERSION:
            raise ValueError(
                "dynamics_version must match the implemented commitment dynamics: "
                f"{COMMITMENT_DYNAMICS_VERSION}"
            )

        for name in ("cold_start_s", "hot_start_s", "min_up_s"):
            _require_number(name, getattr(self, name), low=0.0, low_inclusive=False)
        for name in ("cooldown_s", "min_down_s", "shortfall_grace_s"):
            _require_number(name, getattr(self, name), low=0.0)
        for name in (
            "start_load_frac",
            "min_stable_load_frac",
            "hot_standby_load_frac",
            "ramp_up_frac_per_min",
            "ramp_down_frac_per_min",
            "low_load_threshold_frac",
        ):
            _require_number(name, getattr(self, name), low=0.0, high=1.0)

        if self.start_load_frac <= 0.0:
            raise ValueError("start_load_frac must be > 0")
        if self.hot_standby_load_frac >= self.start_load_frac:
            raise ValueError("hot_standby_load_frac must be below start_load_frac")
        if self.start_load_frac >= self.min_stable_load_frac:
            raise ValueError("start_load_frac must be below min_stable_load_frac")
        if not (
            self.min_stable_load_frac
            <= self.low_load_threshold_frac
            <= 1.0
        ):
            raise ValueError(
                "low_load_threshold_frac must be between min_stable_load_frac and 1"
            )
        if self.ramp_up_frac_per_min <= 0.0 or self.ramp_down_frac_per_min <= 0.0:
            raise ValueError("running ramp fractions must be > 0")

        nominal = _require_number(
            "nominal_specific_energy_kwh_per_kg",
            self.nominal_specific_energy_kwh_per_kg,
            low=LEGACY_NOMINAL_SPECIFIC_ENERGY_KWH_PER_KG,
        )
        if nominal < LEGACY_NOMINAL_SPECIFIC_ENERGY_KWH_PER_KG:
            raise ValueError(
                "nominal_specific_energy_kwh_per_kg must not be below the legacy "
                f"{LEGACY_NOMINAL_SPECIFIC_ENERGY_KWH_PER_KG:g} kWh/kg reference"
            )

        anchors = self.specific_energy_multiplier_anchors
        if not isinstance(anchors, tuple):
            raise TypeError("specific_energy_multiplier_anchors must be an immutable tuple")
        if len(anchors) < 2:
            raise ValueError("specific_energy_multiplier_anchors requires at least two anchors")
        previous_load = -math.inf
        previous_multiplier = math.inf
        for index, anchor in enumerate(anchors):
            if not isinstance(anchor, tuple) or len(anchor) != 2:
                raise TypeError(f"specific-energy anchor {index} must be a (load, multiplier) tuple")
            load = _require_number(f"anchor[{index}].load", anchor[0], low=0.0, high=1.0)
            multiplier = _require_number(
                f"anchor[{index}].multiplier", anchor[1], low=1.0
            )
            if load <= previous_load:
                raise ValueError("specific-energy anchor loads must increase strictly")
            if multiplier > previous_multiplier:
                raise ValueError(
                    "specific-energy multipliers must be non-increasing as load rises"
                )
            previous_load = load
            previous_multiplier = multiplier
        if not math.isclose(anchors[0][0], self.min_stable_load_frac):
            raise ValueError("the first specific-energy anchor must equal min stable load")
        if not math.isclose(anchors[-1][0], 1.0):
            raise ValueError("the final specific-energy anchor must be at full load")
        if not math.isclose(anchors[-1][1], 1.0):
            raise ValueError("the full-load specific-energy multiplier must equal 1")

    def specific_energy_multiplier(self, load_frac: float) -> float:
        """Linearly interpolate the frozen part-load energy penalty."""
        load = _require_number("load_frac", load_frac, low=0.0, high=1.0)
        anchors = self.specific_energy_multiplier_anchors
        if load <= anchors[0][0]:
            return anchors[0][1]
        for (left_x, left_y), (right_x, right_y) in zip(anchors, anchors[1:]):
            if load <= right_x:
                fraction = (load - left_x) / (right_x - left_x)
                return left_y + fraction * (right_y - left_y)
        return anchors[-1][1]

    def validate_timestep(self, dt_s: float) -> None:
        """Reject a physics grid that would blur a declared time boundary.

        Commitment events are boundary-discrete.  Requiring exact alignment
        prevents a partial final interval from being charged at full start or
        standby power.  The first producing interval must also be able to move
        from fixed start power to minimum stable load within the running ramp.
        """
        timestep_s = _require_number("dt_s", dt_s, low=0.0, low_inclusive=False)
        for name in (
            "cold_start_s",
            "hot_start_s",
            "cooldown_s",
            "min_up_s",
            "min_down_s",
            "shortfall_grace_s",
        ):
            duration_s = float(getattr(self, name))
            intervals = duration_s / timestep_s
            if not math.isclose(intervals, round(intervals), abs_tol=1e-9):
                raise ValueError(
                    f"{name}={duration_s:g}s must align exactly to dt_s={timestep_s:g}s"
                )
        first_running_load = (
            self.start_load_frac
            + self.ramp_up_frac_per_min * timestep_s / 60.0
        )
        if first_running_load + 1e-12 < self.min_stable_load_frac:
            raise ValueError(
                "dt_s is too short for the running ramp to reach minimum stable "
                "load from fixed start power in the first producing interval"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready copy without exposing nested mutable state."""
        result = asdict(self)
        result["specific_energy_multiplier_anchors"] = [
            [load, multiplier]
            for load, multiplier in self.specific_energy_multiplier_anchors
        ]
        return result

    @property
    def profile_sha256(self) -> str:
        """Canonical identity for every numerical and provenance parameter."""
        canonical = json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE = ElectrolyserCommitmentProfile()


def _empty_mode_times() -> tuple[tuple[ElectrolyserMode, float], ...]:
    return tuple((mode, 0.0) for mode in _MODE_ORDER)


@dataclass(frozen=True)
class ElectrolyserCommitmentState:
    """Immutable internal state at the boundary before the next interval."""

    mode: ElectrolyserMode = ElectrolyserMode.OFF
    mode_elapsed_s: float = 0.0
    startup_remaining_s: float = 0.0
    cooldown_remaining_s: float = 0.0
    min_up_remaining_s: float = 0.0
    min_down_remaining_s: float = 0.0
    shortfall_elapsed_s: float = 0.0
    previous_power_kw: float = 0.0

    cold_starts: int = 0
    hot_starts: int = 0
    aborted_starts: int = 0
    forced_stops: int = 0
    controlled_stops: int = 0
    startup_energy_kwh: float = 0.0
    standby_energy_kwh: float = 0.0
    operating_time_s: float = 0.0
    low_load_time_s: float = 0.0
    ramp_limited_intervals: int = 0
    absolute_load_movement_kw: float = 0.0
    mode_time_s: tuple[tuple[ElectrolyserMode, float], ...] = _empty_mode_times()

    def with_mode_time(self, mode: ElectrolyserMode, dt_s: float) -> "ElectrolyserCommitmentState":
        times = tuple(
            (key, value + dt_s if key is mode else value)
            for key, value in self.mode_time_s
        )
        return replace(self, mode_time_s=times)


@dataclass(frozen=True)
class ElectrolyserCommitmentSnapshot:
    """Read-only public audit view of commitment state and provenance."""

    profile_id: str
    technology: str
    calibration_basis: str
    dynamics_version: str
    profile_sha256: str
    mode: ElectrolyserMode
    mode_elapsed_s: float
    startup_remaining_s: float
    cooldown_remaining_s: float
    min_up_remaining_s: float
    min_down_remaining_s: float
    shortfall_elapsed_s: float
    previous_power_kw: float
    ramp_up_headroom_kw: float
    ramp_down_headroom_kw: float
    cold_starts: int
    hot_starts: int
    aborted_starts: int
    forced_stops: int
    controlled_stops: int
    startup_energy_kwh: float
    standby_energy_kwh: float
    operating_time_s: float
    low_load_time_s: float
    ramp_limited_intervals: int
    absolute_load_movement_kw: float
    mode_time_s: tuple[tuple[ElectrolyserMode, float], ...]

    @property
    def committed(self) -> bool:
        return self.mode in {
            ElectrolyserMode.COLD_STARTING,
            ElectrolyserMode.RUNNING,
            ElectrolyserMode.HOT_STANDBY,
            ElectrolyserMode.HOT_STARTING,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "technology": self.technology,
            "calibration_basis": self.calibration_basis,
            "dynamics_version": self.dynamics_version,
            "profile_sha256": self.profile_sha256,
            "mode": self.mode.value,
            "committed": self.committed,
            "mode_elapsed_s": self.mode_elapsed_s,
            "startup_remaining_s": self.startup_remaining_s,
            "cooldown_remaining_s": self.cooldown_remaining_s,
            "min_up_remaining_s": self.min_up_remaining_s,
            "min_down_remaining_s": self.min_down_remaining_s,
            "shortfall_elapsed_s": self.shortfall_elapsed_s,
            "previous_power_kw": self.previous_power_kw,
            "ramp_up_headroom_kw": self.ramp_up_headroom_kw,
            "ramp_down_headroom_kw": self.ramp_down_headroom_kw,
            "cold_starts": self.cold_starts,
            "hot_starts": self.hot_starts,
            "aborted_starts": self.aborted_starts,
            "forced_stops": self.forced_stops,
            "controlled_stops": self.controlled_stops,
            "startup_energy_kwh": self.startup_energy_kwh,
            "standby_energy_kwh": self.standby_energy_kwh,
            "operating_time_s": self.operating_time_s,
            "low_load_time_s": self.low_load_time_s,
            "ramp_limited_intervals": self.ramp_limited_intervals,
            "absolute_load_movement_kw": self.absolute_load_movement_kw,
            "mode_time_s": {mode.value: seconds for mode, seconds in self.mode_time_s},
        }


@dataclass(frozen=True)
class ElectrolyserCommitmentStep:
    """Accepted bus power and saleable hydrogen over the interval just ended."""

    accepted_power_kw: float
    h2_produced_kg: float


__all__ = [
    "COMMITMENT_DYNAMICS_VERSION",
    "COMMITMENT_INTENT_DEADBAND_FRAC",
    "DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE",
    "ElectrolyserCommitmentProfile",
    "ElectrolyserCommitmentSnapshot",
    "ElectrolyserCommitmentState",
    "ElectrolyserCommitmentStep",
    "ElectrolyserMode",
    "LEGACY_NOMINAL_SPECIFIC_ENERGY_KWH_PER_KG",
]

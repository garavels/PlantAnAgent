"""Privileged delivery relaxation for Environment V2.

This module is a research instrument, not an online controller.  It consumes a
complete sampled episode (including future solar truth), constructs a
physics-step continuous relaxation (one minute in the registered benchmark),
and asks HiGHS for a numerical primal witness. A delivery upper bound for the
encoded LP is published only after independent primal validation and an
exact-rational box-Lagrangian certificate. Applicability to simulator traces is
a separately tested numerical embedding claim, not a bit-exact theorem.

SciPy is deliberately imported only inside :func:`solve_delivery_relaxation`.
Importing the normal backend therefore does not require the optional
``optimisation`` dependency group.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import struct
import time
from dataclasses import dataclass, replace
from fractions import Fraction
from numbers import Integral, Real
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from plant_sim import PlantConfig, SolarArray

from ...controllers.base import Constraints
from .config import (
    CLEAR_SKY_ENVELOPE_ID,
    STREAM_NAMES,
    EpisodeConfigV2,
    EpisodeV2,
    sample_episode,
)
from .contract import contract_digest, describe_checkpoint_contract
from .reward import RewardWeightsV2
from .weather import WEATHER_GENERATOR_VERSION

MODEL_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
MODEL_KIND = "v2-perfect-truth-physics-step-delivery-relaxation"
PRIVILEGED_INFORMATION_LABEL = (
    "offline perfect-information research bound; not a deployable controller"
)
CERTIFICATE_SCOPE = "formal certificate for the encoded finite-box linear program only"
SIMULATOR_EMBEDDING_STATUS = (
    "relaxation embedding is numerically validated within declared tolerances; "
    "not bit-exact certified"
)
DEFAULT_FEASIBILITY_TOLERANCE = 1e-7
DEFAULT_PHYSICAL_TOLERANCE = 1e-6
DEFAULT_OPTIMALITY_TOLERANCE_KG = 1e-5
DEFAULT_MARGINAL_SIGN_TOLERANCE = 1e-9


class DeliveryRelaxationError(ValueError):
    """The requested episode cannot be represented by the declared relaxation."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 digest") from exc
    return value


def _real(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DeliveryRelaxationError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise DeliveryRelaxationError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise DeliveryRelaxationError(f"{name} must be >= {minimum:g}")
    return number


def _positive(name: str, value: object) -> float:
    number = _real(name, value)
    if number <= 0.0:
        raise DeliveryRelaxationError(f"{name} must be positive")
    return number


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise DeliveryRelaxationError(f"{name} must be an integer")
    number = int(value)
    if number < minimum:
        raise DeliveryRelaxationError(f"{name} must be >= {minimum}")
    return number


def _finite_tuple(name: str, values: Iterable[object]) -> tuple[float, ...]:
    return tuple(_real(f"{name}[{index}]", value) for index, value in enumerate(values))


@dataclass(frozen=True, slots=True)
class SparseCooMatrix:
    """Small immutable COO representation independent of SciPy."""

    shape: tuple[int, int]
    row_indices: tuple[int, ...]
    column_indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.shape, tuple)
            or len(self.shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in self.shape
            )
        ):
            raise TypeError("sparse matrix shape must be a pair of integers")
        rows, columns = (int(value) for value in self.shape)
        if rows < 0 or columns < 0:
            raise ValueError("sparse matrix dimensions must be non-negative")
        object.__setattr__(self, "shape", (rows, columns))

        try:
            raw_row_indices = tuple(self.row_indices)
            raw_column_indices = tuple(self.column_indices)
            raw_values = tuple(self.values)
        except TypeError as exc:
            raise TypeError("sparse COO arrays must be iterable") from exc

        canonical_rows: list[int] = []
        for index, row in enumerate(raw_row_indices):
            if isinstance(row, bool) or not isinstance(row, Integral):
                raise TypeError(f"sparse row index {index} must be an integer")
            canonical_rows.append(int(row))

        canonical_columns: list[int] = []
        for index, column in enumerate(raw_column_indices):
            if isinstance(column, bool) or not isinstance(column, Integral):
                raise TypeError(f"sparse column index {index} must be an integer")
            canonical_columns.append(int(column))

        canonical_values = tuple(
            _real(f"sparse value {index}", value)
            for index, value in enumerate(raw_values)
        )
        object.__setattr__(self, "row_indices", tuple(canonical_rows))
        object.__setattr__(self, "column_indices", tuple(canonical_columns))
        object.__setattr__(self, "values", canonical_values)

        if not (len(canonical_rows) == len(canonical_columns) == len(canonical_values)):
            raise ValueError("sparse COO arrays must have equal length")

        previous: tuple[int, int] | None = None
        for index, (row, column, value) in enumerate(
            zip(self.row_indices, self.column_indices, self.values, strict=True)
        ):
            coordinate = (row, column)
            if not (0 <= coordinate[0] < rows and 0 <= coordinate[1] < columns):
                raise ValueError(
                    f"sparse coordinate {coordinate} lies outside {self.shape}"
                )
            if previous is not None and coordinate <= previous:
                raise ValueError(
                    "sparse coordinates must be unique and row-major sorted"
                )
            previous = coordinate
            if value == 0.0:
                raise ValueError("stored sparse values must be non-zero")

    @property
    def nnz(self) -> int:
        return len(self.values)

    @property
    def data(self) -> tuple[float, ...]:
        """NumPy/SciPy-compatible name for the stored non-zero values."""
        return self.values

    def matvec(self, vector: Sequence[float]) -> tuple[float, ...]:
        if len(vector) != self.shape[1]:
            raise ValueError("matrix/vector dimensions do not agree")
        result = [0.0] * self.shape[0]
        for row, column, value in zip(
            self.row_indices,
            self.column_indices,
            self.values,
            strict=True,
        ):
            result[row] += value * float(vector[column])
        return tuple(result)

    def to_dict(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "row_indices": list(self.row_indices),
            "column_indices": list(self.column_indices),
            "values": list(self.values),
        }


@dataclass(frozen=True, slots=True)
class FiniteBoxLinearProgram:
    """Normalized minimization LP with explicit finite independent boxes."""

    objective: tuple[float, ...]
    a_ub: SparseCooMatrix
    b_ub: tuple[float, ...]
    a_eq: SparseCooMatrix
    b_eq: tuple[float, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]

    def __post_init__(self) -> None:
        objective = _finite_tuple("objective", self.objective)
        b_ub = _finite_tuple("b_ub", self.b_ub)
        b_eq = _finite_tuple("b_eq", self.b_eq)
        lower = _finite_tuple("lower_bounds", self.lower_bounds)
        upper = _finite_tuple("upper_bounds", self.upper_bounds)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "b_ub", b_ub)
        object.__setattr__(self, "b_eq", b_eq)
        object.__setattr__(self, "lower_bounds", lower)
        object.__setattr__(self, "upper_bounds", upper)
        variables = len(objective)
        if variables == 0:
            raise ValueError("linear program must contain at least one variable")
        if self.a_ub.shape != (len(b_ub), variables):
            raise ValueError("A_ub shape does not match b_ub and objective")
        if self.a_eq.shape != (len(b_eq), variables):
            raise ValueError("A_eq shape does not match b_eq and objective")
        if len(lower) != variables or len(upper) != variables:
            raise ValueError("variable-bound dimensions do not match objective")
        for index, (low, high) in enumerate(zip(lower, upper, strict=True)):
            if low > high:
                raise ValueError(f"variable {index} has lower bound above upper bound")

    @property
    def variable_count(self) -> int:
        return len(self.objective)

    @property
    def c(self) -> tuple[float, ...]:
        """Conventional short name for the minimization objective vector."""
        return self.objective

    def to_dict(self) -> dict[str, object]:
        return {
            "sense": "minimize",
            "objective": list(self.objective),
            "a_ub": self.a_ub.to_dict(),
            "b_ub": list(self.b_ub),
            "a_eq": self.a_eq.to_dict(),
            "b_eq": list(self.b_eq),
            "lower_bounds": list(self.lower_bounds),
            "upper_bounds": list(self.upper_bounds),
        }


@dataclass(frozen=True, slots=True)
class ProgramResiduals:
    """Direct residuals of a normalized finite-box program."""

    max_equality_absolute: float
    max_inequality_violation: float
    max_lower_bound_violation: float
    max_upper_bound_violation: float

    @property
    def max_box_violation(self) -> float:
        return max(self.max_lower_bound_violation, self.max_upper_bound_violation)

    @property
    def maximum(self) -> float:
        return max(
            self.max_equality_absolute,
            self.max_inequality_violation,
            self.max_box_violation,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "max_equality_absolute": self.max_equality_absolute,
            "max_inequality_violation": self.max_inequality_violation,
            "max_lower_bound_violation": self.max_lower_bound_violation,
            "max_upper_bound_violation": self.max_upper_bound_violation,
            "max_box_violation": self.max_box_violation,
            "maximum": self.maximum,
        }


def evaluate_program_residuals(
    program: FiniteBoxLinearProgram,
    normalized_variables: Sequence[float],
) -> ProgramResiduals:
    """Recompute normalized equation, inequality, and box residuals."""
    if not isinstance(program, FiniteBoxLinearProgram):
        raise TypeError("program must be a FiniteBoxLinearProgram")
    values = _finite_tuple("normalized_variables", normalized_variables)
    if len(values) != program.variable_count:
        raise ValueError("normalized variable vector has the wrong size")
    eq_values = program.a_eq.matvec(values)
    ub_values = program.a_ub.matvec(values)
    return ProgramResiduals(
        max_equality_absolute=max(
            (
                abs(value - rhs)
                for value, rhs in zip(eq_values, program.b_eq, strict=True)
            ),
            default=0.0,
        ),
        max_inequality_violation=max(
            (
                max(0.0, value - rhs)
                for value, rhs in zip(ub_values, program.b_ub, strict=True)
            ),
            default=0.0,
        ),
        max_lower_bound_violation=max(
            (
                max(0.0, low - value)
                for value, low in zip(values, program.lower_bounds, strict=True)
            ),
            default=0.0,
        ),
        max_upper_bound_violation=max(
            (
                max(0.0, value - high)
                for value, high in zip(values, program.upper_bounds, strict=True)
            ),
            default=0.0,
        ),
    )


@dataclass(frozen=True, slots=True)
class VariableBlock:
    """One contiguous physical quantity in the normalized decision vector."""

    name: str
    start: int
    size: int
    unit: str

    def __post_init__(self) -> None:
        if not self.name or not self.unit:
            raise ValueError("variable block name and unit must not be empty")
        _integer("variable block start", self.start)
        _integer("variable block size", self.size, minimum=1)

    @property
    def stop(self) -> int:
        return self.start + self.size

    def to_dict(self) -> dict[str, str | int]:
        return {
            "name": self.name,
            "start": self.start,
            "size": self.size,
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class DeliveryRelaxationModel:
    """Complete normalized LP plus enough physical metadata for independent audit."""

    program: FiniteBoxLinearProgram
    blocks: tuple[VariableBlock, ...]
    physical_scales: tuple[float, ...]
    physical_lower_bounds: tuple[float, ...]
    physical_upper_bounds: tuple[float, ...]
    solar_available_kw: tuple[float, ...]
    solar_truth_sha256: str
    commitments: tuple[tuple[str, int, float], ...]
    physics_steps: int
    dt_h: float
    initial_battery_energy_kwh: float
    initial_h2_kg: float
    initial_co2_kg: float
    initial_ch4_kg: float
    battery_capacity_kwh: float
    battery_reserve_kwh: float
    battery_charge_efficiency: float
    battery_discharge_efficiency: float
    battery_charge_limit_kw: float
    electrolyser_limit_kw: float
    battery_support_limit_kw: float
    dac_limit_kw: float
    methanator_limit_kw: float
    best_h2_specific_energy_kwh_per_kg: float
    dac_specific_energy_kwh_per_kg: float
    methanator_specific_energy_kwh_per_kg: float
    h2_kg_per_kg_ch4: float
    co2_kg_per_kg_ch4: float
    h2_capacity_kg: float
    co2_capacity_kg: float
    ch4_capacity_kg: float
    preserve_initial_endowment: bool
    canonical_episode_verified: bool
    profile_id: str
    profile_sha256: str | None
    coefficient_proof: str
    checkpoint_contract_json: str
    checkpoint_contract_digest: str
    episode_manifest_json: str
    exogenous_episode_sha256: str
    schema_version: int = MODEL_SCHEMA_VERSION
    model_kind: str = MODEL_KIND

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_SCHEMA_VERSION or self.model_kind != MODEL_KIND:
            raise ValueError("unsupported delivery-relaxation model identity")
        if not isinstance(self.program, FiniteBoxLinearProgram):
            raise TypeError("program must be a FiniteBoxLinearProgram")
        if not isinstance(self.preserve_initial_endowment, bool):
            raise TypeError("preserve_initial_endowment must be a bool")
        if not isinstance(self.canonical_episode_verified, bool):
            raise TypeError("canonical_episode_verified must be a bool")
        for name, value in (
            ("blocks", self.blocks),
            ("physical_scales", self.physical_scales),
            ("physical_lower_bounds", self.physical_lower_bounds),
            ("physical_upper_bounds", self.physical_upper_bounds),
            ("solar_available_kw", self.solar_available_kw),
            ("commitments", self.commitments),
        ):
            if not isinstance(value, tuple):
                raise TypeError(f"{name} must be an immutable tuple")
        physics_steps = _integer("physics_steps", self.physics_steps, minimum=1)
        n = self.program.variable_count
        if not (
            len(self.physical_scales)
            == len(self.physical_lower_bounds)
            == len(self.physical_upper_bounds)
            == n
        ):
            raise ValueError("physical variable metadata dimensions do not match LP")
        if len(self.solar_available_kw) != self.physics_steps:
            raise ValueError("solar vector length must match physics_steps")
        for index, (width, low, high) in enumerate(
            zip(
                self.physical_scales,
                self.physical_lower_bounds,
                self.physical_upper_bounds,
                strict=True,
            )
        ):
            width = _real(f"physical_scales[{index}]", width, minimum=0.0)
            low = _real(f"physical_lower_bounds[{index}]", low, minimum=0.0)
            high = _real(f"physical_upper_bounds[{index}]", high, minimum=0.0)
            if low > high or width != high - low:
                raise ValueError(
                    "physical offsets, widths, and upper bounds are incoherent"
                )
        if any(value != 0.0 for value in self.program.lower_bounds) or any(
            value != 1.0 for value in self.program.upper_bounds
        ):
            raise ValueError("delivery-relaxation solver boxes must all be [0, 1]")

        expected_blocks = (
            ("solar_to_battery_kw", physics_steps, "kW"),
            ("solar_to_electrolyser_kw", physics_steps, "kW"),
            ("solar_to_dac_kw", physics_steps, "kW"),
            ("solar_to_methanator_kw", physics_steps, "kW"),
            ("battery_to_electrolyser_kw", physics_steps, "kW"),
            ("curtailed_solar_kw", physics_steps, "kW"),
            ("h2_produced_kg", physics_steps, "kg"),
            ("co2_captured_kg", physics_steps, "kg"),
            ("ch4_produced_kg", physics_steps, "kg"),
            ("battery_energy_kwh", physics_steps + 1, "kWh"),
            ("h2_inventory_kg", physics_steps + 1, "kg"),
            ("co2_inventory_kg", physics_steps + 1, "kg"),
            ("ch4_inventory_kg", physics_steps + 1, "kg"),
            ("delivered_ch4_kg", max(len(self.commitments), 1), "kg"),
        )
        actual_blocks = tuple(
            (block.name, block.size, block.unit) for block in self.blocks
        )
        if actual_blocks != expected_blocks:
            raise ValueError(
                "delivery-relaxation variable block schema is inconsistent"
            )
        expected_start = 0
        seen: set[str] = set()
        for block in self.blocks:
            if block.name in seen or block.start != expected_start:
                raise ValueError(
                    "variable blocks must be unique, contiguous, and ordered"
                )
            seen.add(block.name)
            expected_start = block.stop
        if expected_start != n:
            raise ValueError("variable blocks do not cover the LP vector")

        solar = _finite_tuple("solar_available_kw", self.solar_available_kw)
        if any(value < 0.0 for value in solar):
            raise ValueError("solar availability must be non-negative")
        _require_sha256("solar_truth_sha256", self.solar_truth_sha256)
        if self.solar_truth_sha256 != _solar_digest(solar):
            raise ValueError("solar truth digest does not match the serialized vector")

        previous_due = 0
        seen_ids: set[str] = set()
        for index, commitment in enumerate(self.commitments):
            if not isinstance(commitment, tuple) or len(commitment) != 3:
                raise TypeError("commitments must contain immutable three-field tuples")
            commitment_id, due_raw, request_raw = commitment
            if not isinstance(commitment_id, str) or not commitment_id.strip():
                raise ValueError("commitment ids must be non-empty strings")
            if commitment_id in seen_ids:
                raise ValueError("commitment ids must be unique")
            seen_ids.add(commitment_id)
            due = _integer(f"commitments[{index}].due_step", due_raw, minimum=1)
            request = _real(
                f"commitments[{index}].requested_kg", request_raw, minimum=0.0
            )
            if due < previous_due or due > physics_steps:
                raise ValueError(
                    "commitment deadlines must be ordered inside the horizon"
                )
            previous_due = due
            delivery_column = self.block("delivered_ch4_kg").start + index
            if (
                self.physical_lower_bounds[delivery_column] != 0.0
                or self.physical_upper_bounds[delivery_column] != request
            ):
                raise ValueError("delivery variable boxes do not match commitments")

        objective_expected = [0.0] * n
        delivery_block = self.block("delivered_ch4_kg")
        for index in range(len(self.commitments)):
            column = delivery_block.start + index
            objective_expected[column] = -self.physical_scales[column]
        if self.program.objective != tuple(objective_expected):
            raise ValueError("LP objective does not match cumulative physical delivery")

        positive_fields = (
            "dt_h",
            "battery_capacity_kwh",
            "battery_charge_efficiency",
            "battery_discharge_efficiency",
            "best_h2_specific_energy_kwh_per_kg",
            "dac_specific_energy_kwh_per_kg",
            "methanator_specific_energy_kwh_per_kg",
            "h2_kg_per_kg_ch4",
            "co2_kg_per_kg_ch4",
            "h2_capacity_kg",
            "co2_capacity_kg",
            "ch4_capacity_kg",
        )
        for name in positive_fields:
            _positive(name, getattr(self, name))
        for name in (
            "initial_battery_energy_kwh",
            "initial_h2_kg",
            "initial_co2_kg",
            "initial_ch4_kg",
            "battery_reserve_kwh",
            "battery_charge_limit_kw",
            "electrolyser_limit_kw",
            "battery_support_limit_kw",
            "dac_limit_kw",
            "methanator_limit_kw",
        ):
            _real(name, getattr(self, name), minimum=0.0)
        if not (
            self.battery_reserve_kwh
            <= self.initial_battery_energy_kwh
            <= self.battery_capacity_kwh
        ):
            raise ValueError("initial battery energy is outside its active box")
        for name, initial, capacity in (
            ("H2", self.initial_h2_kg, self.h2_capacity_kg),
            ("CO2", self.initial_co2_kg, self.co2_capacity_kg),
            ("CH4", self.initial_ch4_kg, self.ch4_capacity_kg),
        ):
            if initial > capacity:
                raise ValueError(f"initial {name} exceeds capacity")
        for name in ("battery_charge_efficiency", "battery_discharge_efficiency"):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must not exceed one")
        if self.program.a_eq.shape[0] != 4 + 5 * physics_steps:
            raise ValueError("LP equality-row schema is inconsistent")
        if self.program.a_ub.shape[0] != 10 * physics_steps:
            raise ValueError("LP inequality-row schema is inconsistent")

        def physical_bounds(name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
            block = self.block(name)
            return (
                self.physical_lower_bounds[block.start : block.stop],
                self.physical_upper_bounds[block.start : block.stop],
            )

        for name in (
            "solar_to_battery_kw",
            "solar_to_electrolyser_kw",
            "solar_to_dac_kw",
            "solar_to_methanator_kw",
            "curtailed_solar_kw",
        ):
            low, high = physical_bounds(name)
            if low != (0.0,) * physics_steps or high != solar:
                raise ValueError(f"{name} physical boxes do not match solar truth")
        production_caps = {
            "h2_produced_kg": (
                self.electrolyser_limit_kw
                * self.dt_h
                / self.best_h2_specific_energy_kwh_per_kg
            ),
            "co2_captured_kg": (
                self.dac_limit_kw * self.dt_h / self.dac_specific_energy_kwh_per_kg
            ),
            "ch4_produced_kg": (
                self.methanator_limit_kw
                * self.dt_h
                / self.methanator_specific_energy_kwh_per_kg
            ),
        }
        expected_constant_boxes = {
            "battery_to_electrolyser_kw": self.battery_support_limit_kw,
            **production_caps,
        }
        for name, upper in expected_constant_boxes.items():
            low_values, high_values = physical_bounds(name)
            if (
                low_values != (0.0,) * physics_steps
                or high_values != (upper,) * physics_steps
            ):
                raise ValueError(f"{name} physical boxes are inconsistent")
        state_boxes = {
            "battery_energy_kwh": (
                self.battery_reserve_kwh,
                self.initial_battery_energy_kwh,
                self.battery_capacity_kwh,
            ),
            "h2_inventory_kg": (0.0, self.initial_h2_kg, self.h2_capacity_kg),
            "co2_inventory_kg": (0.0, self.initial_co2_kg, self.co2_capacity_kg),
            "ch4_inventory_kg": (0.0, self.initial_ch4_kg, self.ch4_capacity_kg),
        }
        for name, (ordinary_low, endowment_low, capacity) in state_boxes.items():
            low_values, high_values = physical_bounds(name)
            expected_low = [ordinary_low] * (physics_steps + 1)
            if self.preserve_initial_endowment:
                expected_low[-1] = max(ordinary_low, endowment_low)
            if low_values != tuple(expected_low) or high_values != (capacity,) * (
                physics_steps + 1
            ):
                raise ValueError(f"{name} physical boxes are inconsistent")
        if not self.profile_id or not self.coefficient_proof:
            raise ValueError("profile and coefficient proof metadata must not be empty")
        if self.profile_sha256 is not None:
            _require_sha256("profile_sha256", self.profile_sha256)

        _require_sha256("checkpoint_contract_digest", self.checkpoint_contract_digest)
        _require_sha256("exogenous_episode_sha256", self.exogenous_episode_sha256)
        try:
            checkpoint = json.loads(self.checkpoint_contract_json)
            manifest = json.loads(self.episode_manifest_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("model identity JSON is malformed") from exc
        if _canonical_json(checkpoint) != self.checkpoint_contract_json:
            raise ValueError("checkpoint contract JSON must be canonical")
        if _canonical_json(manifest) != self.episode_manifest_json:
            raise ValueError("episode manifest JSON must be canonical")
        if (
            checkpoint.get("checkpoint_contract_digest")
            != self.checkpoint_contract_digest
        ):
            raise ValueError("checkpoint digest metadata is inconsistent")
        checkpoint_payload = dict(checkpoint)
        checkpoint_payload.pop("checkpoint_contract_digest", None)
        if contract_digest(checkpoint_payload) != self.checkpoint_contract_digest:
            raise ValueError("checkpoint contract payload does not match its digest")
        if manifest.get("exogenous_episode_sha256") != self.exogenous_episode_sha256:
            raise ValueError("episode manifest metadata is inconsistent")

    @property
    def certificate_scope(self) -> str:
        return CERTIFICATE_SCOPE

    @property
    def simulator_embedding_status(self) -> str:
        return SIMULATOR_EMBEDDING_STATUS

    @property
    def evidence_provenance_status(self) -> str:
        return (
            "canonical sample_episode regeneration verified"
            if self.canonical_episode_verified
            else "noncanonical research fixture; not evidence-verified"
        )

    @property
    def requested_kg(self) -> float:
        exact = sum(
            (Fraction.from_float(item[2]) for item in self.commitments),
            start=Fraction(0),
        )
        return next_float_up(exact)

    @property
    def is_task_upper_bound(self) -> bool:
        return not self.preserve_initial_endowment

    def block(self, name: str) -> VariableBlock:
        for block in self.blocks:
            if block.name == name:
                return block
        raise KeyError(name)

    def physical_vector(self, normalized: Sequence[float]) -> tuple[float, ...]:
        if len(normalized) != self.program.variable_count:
            raise ValueError("normalized vector has the wrong size")
        return tuple(
            low + float(value) * scale
            for value, scale, low in zip(
                normalized,
                self.physical_scales,
                self.physical_lower_bounds,
                strict=True,
            )
        )

    def normalized_vector(
        self,
        physical: Sequence[float],
        *,
        tolerance: float = 1e-9,
    ) -> tuple[float, ...]:
        """Map a physical trace into solver coordinates, including fixed boxes."""
        if len(physical) != self.program.variable_count:
            raise ValueError("physical vector has the wrong size")
        tolerance = _real("tolerance", tolerance, minimum=0.0)
        result: list[float] = []
        for index, (value_raw, low, width) in enumerate(
            zip(
                physical,
                self.physical_lower_bounds,
                self.physical_scales,
                strict=True,
            )
        ):
            value = _real(f"physical[{index}]", value_raw)
            if width == 0.0:
                if abs(value - low) > tolerance:
                    raise ValueError(
                        f"physical variable {index} differs from its fixed value"
                    )
                result.append(0.0)
            else:
                result.append((value - low) / width)
        return tuple(result)

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_kind": self.model_kind,
            "program": self.program.to_dict(),
            "blocks": [block.to_dict() for block in self.blocks],
            "physical_scales": list(self.physical_scales),
            "physical_lower_bounds": list(self.physical_lower_bounds),
            "physical_upper_bounds": list(self.physical_upper_bounds),
            "solar_available_kw": list(self.solar_available_kw),
            "solar_truth_sha256": self.solar_truth_sha256,
            "commitments": [
                {"id": item[0], "due_step": item[1], "requested_kg": item[2]}
                for item in self.commitments
            ],
            "physics_steps": self.physics_steps,
            "dt_h": self.dt_h,
            "initial": {
                "battery_energy_kwh": self.initial_battery_energy_kwh,
                "h2_kg": self.initial_h2_kg,
                "co2_kg": self.initial_co2_kg,
                "ch4_kg": self.initial_ch4_kg,
            },
            "limits": {
                "battery_capacity_kwh": self.battery_capacity_kwh,
                "battery_reserve_kwh": self.battery_reserve_kwh,
                "battery_charge_kw": self.battery_charge_limit_kw,
                "electrolyser_kw": self.electrolyser_limit_kw,
                "battery_support_kw": self.battery_support_limit_kw,
                "dac_kw": self.dac_limit_kw,
                "methanator_kw": self.methanator_limit_kw,
                "h2_capacity_kg": self.h2_capacity_kg,
                "co2_capacity_kg": self.co2_capacity_kg,
                "ch4_capacity_kg": self.ch4_capacity_kg,
            },
            "coefficients": {
                "battery_charge_efficiency": self.battery_charge_efficiency,
                "battery_discharge_efficiency": self.battery_discharge_efficiency,
                "best_h2_specific_energy_kwh_per_kg": (
                    self.best_h2_specific_energy_kwh_per_kg
                ),
                "dac_specific_energy_kwh_per_kg": self.dac_specific_energy_kwh_per_kg,
                "methanator_specific_energy_kwh_per_kg": (
                    self.methanator_specific_energy_kwh_per_kg
                ),
                "h2_kg_per_kg_ch4": self.h2_kg_per_kg_ch4,
                "co2_kg_per_kg_ch4": self.co2_kg_per_kg_ch4,
                "proof": self.coefficient_proof,
            },
            "preserve_initial_endowment": self.preserve_initial_endowment,
            "canonical_episode_verified": self.canonical_episode_verified,
            "evidence_provenance_status": self.evidence_provenance_status,
            "profile": {"id": self.profile_id, "sha256": self.profile_sha256},
            "checkpoint_contract": json.loads(self.checkpoint_contract_json),
            "checkpoint_contract_digest": self.checkpoint_contract_digest,
            "episode_manifest": json.loads(self.episode_manifest_json),
            "exogenous_episode_sha256": self.exogenous_episode_sha256,
            "privileged_information": PRIVILEGED_INFORMATION_LABEL,
            "certificate_scope": self.certificate_scope,
            "simulator_embedding_status": self.simulator_embedding_status,
        }

    @property
    def model_sha256(self) -> str:
        return _sha256_json(self.identity_payload())

    def to_dict(self) -> dict[str, object]:
        payload = self.identity_payload()
        payload["model_sha256"] = self.model_sha256
        return payload


@dataclass(frozen=True, slots=True)
class PrimalResiduals:
    max_equality_normalized: float
    max_inequality_violation_normalized: float
    max_box_violation_normalized: float
    max_solar_balance_kw: float
    max_battery_balance_kwh: float
    max_h2_balance_kg: float
    max_co2_balance_kg: float
    max_ch4_balance_kg: float
    max_headroom_violation_kg: float
    max_yield_violation_kg: float
    max_delivery_bound_violation_kg: float
    max_physical_box_violation: float
    max_component_power_violation_kw: float

    @property
    def max_physical_violation(self) -> float:
        return max(
            self.max_solar_balance_kw,
            self.max_battery_balance_kwh,
            self.max_h2_balance_kg,
            self.max_co2_balance_kg,
            self.max_ch4_balance_kg,
            self.max_headroom_violation_kg,
            self.max_yield_violation_kg,
            self.max_delivery_bound_violation_kg,
            self.max_physical_box_violation,
            self.max_component_power_violation_kw,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            field: float(getattr(self, field)) for field in self.__dataclass_fields__
        } | {"max_physical_violation": self.max_physical_violation}


@dataclass(frozen=True, slots=True)
class DeliveryWitness:
    id: str
    due_step: int
    requested_kg: float
    delivered_kg: float

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "id": self.id,
            "due_step": self.due_step,
            "requested_kg": self.requested_kg,
            "delivered_kg": self.delivered_kg,
        }


@dataclass(frozen=True, slots=True)
class DeliveryRelaxationWitness:
    normalized_variables: tuple[float, ...]
    solar_to_battery_kw: tuple[float, ...]
    solar_to_electrolyser_kw: tuple[float, ...]
    solar_to_dac_kw: tuple[float, ...]
    solar_to_methanator_kw: tuple[float, ...]
    battery_to_electrolyser_kw: tuple[float, ...]
    curtailed_solar_kw: tuple[float, ...]
    h2_produced_kg: tuple[float, ...]
    co2_captured_kg: tuple[float, ...]
    ch4_produced_kg: tuple[float, ...]
    battery_energy_kwh: tuple[float, ...]
    h2_inventory_kg: tuple[float, ...]
    co2_inventory_kg: tuple[float, ...]
    ch4_inventory_kg: tuple[float, ...]
    deliveries: tuple[DeliveryWitness, ...]
    residuals: PrimalResiduals

    @property
    def delivered_kg(self) -> float:
        return math.fsum(item.delivered_kg for item in self.deliveries)

    def to_dict(self) -> dict[str, object]:
        return {
            "normalized_variables": list(self.normalized_variables),
            "solar_to_battery_kw": list(self.solar_to_battery_kw),
            "solar_to_electrolyser_kw": list(self.solar_to_electrolyser_kw),
            "solar_to_dac_kw": list(self.solar_to_dac_kw),
            "solar_to_methanator_kw": list(self.solar_to_methanator_kw),
            "battery_to_electrolyser_kw": list(self.battery_to_electrolyser_kw),
            "curtailed_solar_kw": list(self.curtailed_solar_kw),
            "h2_produced_kg": list(self.h2_produced_kg),
            "co2_captured_kg": list(self.co2_captured_kg),
            "ch4_produced_kg": list(self.ch4_produced_kg),
            "battery_energy_kwh": list(self.battery_energy_kwh),
            "h2_inventory_kg": list(self.h2_inventory_kg),
            "co2_inventory_kg": list(self.co2_inventory_kg),
            "ch4_inventory_kg": list(self.ch4_inventory_kg),
            "deliveries": [item.to_dict() for item in self.deliveries],
            "delivered_kg": self.delivered_kg,
            "residuals": self.residuals.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DualDeliveryCertificate:
    """Formal encoded-LP ceiling; simulator embedding is a separate validation."""

    minimization_lower_bound_numerator: int
    minimization_lower_bound_denominator: int
    uncapped_delivery_upper_bound_numerator: int
    uncapped_delivery_upper_bound_denominator: int
    requested_kg_numerator: int
    requested_kg_denominator: int
    certified_delivery_upper_bound_numerator: int
    certified_delivery_upper_bound_denominator: int
    dual_certified_delivery_upper_bound_kg: float
    positive_inequality_marginal_count: int
    max_positive_inequality_marginal: float

    @property
    def minimization_lower_bound(self) -> Fraction:
        return Fraction(
            self.minimization_lower_bound_numerator,
            self.minimization_lower_bound_denominator,
        )

    @property
    def certified_delivery_upper_bound(self) -> Fraction:
        return Fraction(
            self.certified_delivery_upper_bound_numerator,
            self.certified_delivery_upper_bound_denominator,
        )

    @property
    def certificate_scope(self) -> str:
        return CERTIFICATE_SCOPE

    def to_dict(self) -> dict[str, object]:
        def exact(numerator: int, denominator: int) -> str:
            return f"{numerator}/{denominator}"

        return {
            "minimization_lower_bound_exact": exact(
                self.minimization_lower_bound_numerator,
                self.minimization_lower_bound_denominator,
            ),
            "uncapped_delivery_upper_bound_exact": exact(
                self.uncapped_delivery_upper_bound_numerator,
                self.uncapped_delivery_upper_bound_denominator,
            ),
            "requested_kg_exact": exact(
                self.requested_kg_numerator,
                self.requested_kg_denominator,
            ),
            "certified_delivery_upper_bound_exact": exact(
                self.certified_delivery_upper_bound_numerator,
                self.certified_delivery_upper_bound_denominator,
            ),
            "dual_certified_delivery_upper_bound_kg": (
                self.dual_certified_delivery_upper_bound_kg
            ),
            "positive_inequality_marginal_count": (
                self.positive_inequality_marginal_count
            ),
            "max_positive_inequality_marginal": (self.max_positive_inequality_marginal),
            "arithmetic": "exact Fraction.from_float over stored IEEE-754 coefficients",
            "float_conversion": "directed toward positive infinity",
            "certificate_scope": self.certificate_scope,
        }


@dataclass(frozen=True, slots=True)
class SolveSettings:
    """Frozen numerical settings and dependency identity for one solve."""

    feasibility_tolerance: float
    physical_tolerance: float
    optimality_tolerance_kg: float
    marginal_sign_tolerance: float
    solver_options_json: str
    highs_runtime_version: str | None

    @property
    def solver_options(self) -> dict[str, object]:
        value = json.loads(self.solver_options_json)
        if not isinstance(value, dict):  # Construction guarantees this invariant.
            raise RuntimeError("serialized solver options are not an object")
        return value

    @property
    def provenance_gaps(self) -> tuple[str, ...]:
        gaps = [
            "git source revision and dependency-lock digest must be attached by the evaluation runner"
        ]
        if self.highs_runtime_version is None:
            gaps.append(
                "linked HiGHS runtime version is not exposed by this SciPy build"
            )
        return tuple(gaps)

    def to_dict(self) -> dict[str, object]:
        return {
            "feasibility_tolerance": self.feasibility_tolerance,
            "physical_tolerance": self.physical_tolerance,
            "optimality_tolerance_kg": self.optimality_tolerance_kg,
            "marginal_sign_tolerance": self.marginal_sign_tolerance,
            "solver_options": self.solver_options,
            "highs_runtime_version": self.highs_runtime_version,
            "provenance_gaps": list(self.provenance_gaps),
        }


@dataclass(frozen=True, slots=True)
class DeliveryRelaxationResult:
    schema_version: int
    success: bool
    status: int | None
    failure_code: str | None
    message: str
    solver: str
    solver_version: str | None
    solve_settings: SolveSettings
    model_sha256: str
    checkpoint_contract_digest: str
    exogenous_episode_sha256: str
    canonical_episode_verified: bool
    preserve_initial_endowment: bool
    is_task_upper_bound: bool
    requested_kg: float
    numerical_primal_delivery_kg: float | None
    dual_certified_delivery_upper_bound_kg: float | None
    relaxed_fill_ceiling: float | None
    primal_dual_gap_kg: float | None
    numerical_objective: float | None
    iterations: int | None
    elapsed_s: float
    marginal_projection_within_tolerance: bool | None
    certificate: DualDeliveryCertificate | None
    witness: DeliveryRelaxationWitness | None

    @property
    def certificate_scope(self) -> str:
        return CERTIFICATE_SCOPE

    @property
    def simulator_embedding_status(self) -> str:
        return SIMULATOR_EMBEDDING_STATUS

    @property
    def evidence_provenance_status(self) -> str:
        return (
            "canonical sample_episode regeneration verified"
            if self.canonical_episode_verified
            else "noncanonical research fixture; not evidence-verified"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "success": self.success,
            "status": self.status,
            "failure_code": self.failure_code,
            "message": self.message,
            "solver": self.solver,
            "solver_version": self.solver_version,
            "solve_settings": self.solve_settings.to_dict(),
            "model_sha256": self.model_sha256,
            "checkpoint_contract_digest": self.checkpoint_contract_digest,
            "exogenous_episode_sha256": self.exogenous_episode_sha256,
            "canonical_episode_verified": self.canonical_episode_verified,
            "evidence_provenance_status": self.evidence_provenance_status,
            "preserve_initial_endowment": self.preserve_initial_endowment,
            "is_task_upper_bound": self.is_task_upper_bound,
            "requested_kg": self.requested_kg,
            "numerical_primal_delivery_kg": self.numerical_primal_delivery_kg,
            "dual_certified_delivery_upper_bound_kg": (
                self.dual_certified_delivery_upper_bound_kg
            ),
            "relaxed_fill_ceiling": self.relaxed_fill_ceiling,
            "primal_dual_gap_kg": self.primal_dual_gap_kg,
            "numerical_objective": self.numerical_objective,
            "iterations": self.iterations,
            "elapsed_s": self.elapsed_s,
            "marginal_projection_within_tolerance": (
                self.marginal_projection_within_tolerance
            ),
            "certificate": (
                None if self.certificate is None else self.certificate.to_dict()
            ),
            "witness": None if self.witness is None else self.witness.to_dict(),
            "privileged_information": PRIVILEGED_INFORMATION_LABEL,
            "certificate_scope": self.certificate_scope,
            "simulator_embedding_status": self.simulator_embedding_status,
        }


class _SparseRows:
    def __init__(self) -> None:
        self._rows: list[tuple[tuple[tuple[int, float], ...], float]] = []

    def add(
        self, coefficients: Mapping[int, float], rhs: float, *, scale: float
    ) -> None:
        row_scale = _positive("constraint row scale", scale)
        combined: dict[int, float] = {}
        for column, coefficient in coefficients.items():
            value = _real("constraint coefficient", coefficient) / row_scale
            if value != 0.0:
                combined[int(column)] = combined.get(int(column), 0.0) + value
        entries = tuple(
            (column, value)
            for column, value in sorted(combined.items())
            if value != 0.0
        )
        self._rows.append((entries, _real("constraint rhs", rhs) / row_scale))

    def build(self, columns: int) -> tuple[SparseCooMatrix, tuple[float, ...]]:
        row_indices: list[int] = []
        column_indices: list[int] = []
        values: list[float] = []
        rhs: list[float] = []
        for row, (entries, bound) in enumerate(self._rows):
            rhs.append(bound)
            for column, value in entries:
                if not 0 <= column < columns:
                    raise ValueError("constraint references an unknown variable")
                row_indices.append(row)
                column_indices.append(column)
                values.append(value)
        return (
            SparseCooMatrix(
                shape=(len(self._rows), columns),
                row_indices=tuple(row_indices),
                column_indices=tuple(column_indices),
                values=tuple(values),
            ),
            tuple(rhs),
        )


class _Variables:
    def __init__(self) -> None:
        self.blocks: list[VariableBlock] = []
        self.scales: list[float] = []
        self.physical_lower: list[float] = []
        self.physical_upper: list[float] = []

    def add(
        self,
        name: str,
        unit: str,
        lower: Sequence[float],
        upper: Sequence[float],
        fallback_scale: float,
    ) -> VariableBlock:
        if len(lower) != len(upper) or not lower:
            raise ValueError("variable block bounds must be non-empty and aligned")
        # Retain a validated fallback argument in the builder API so every
        # block declares its natural engineering scale. Fixed physical
        # variables nevertheless use an affine width of exactly zero.
        _positive(f"{name} fallback scale", fallback_scale)
        block = VariableBlock(name, len(self.scales), len(lower), unit)
        self.blocks.append(block)
        for index, (low_raw, high_raw) in enumerate(zip(lower, upper, strict=True)):
            low = _real(f"{name}[{index}] lower", low_raw, minimum=0.0)
            high = _real(f"{name}[{index}] upper", high_raw, minimum=0.0)
            if low > high:
                raise DeliveryRelaxationError(
                    f"{name}[{index}] lower bound exceeds its upper bound"
                )
            self.scales.append(high - low)
            self.physical_lower.append(low)
            self.physical_upper.append(high)
        return block


def _validate_configuration(config: PlantConfig, constraints: Constraints) -> None:
    _positive("plant dt_s", config.dt_s)
    _positive("solar peak_kw", config.solar.peak_kw)
    for name, value in (
        ("battery capacity_kwh", config.battery.capacity_kwh),
        ("electrolyser rated_kw", config.electrolyser.rated_kw),
        (
            "electrolyser specific_energy_kwh_per_kg",
            config.electrolyser.specific_energy_kwh_per_kg,
        ),
        ("DAC rated_kw", config.dac.rated_kw),
        ("DAC specific_energy_kwh_per_kg", config.dac.specific_energy_kwh_per_kg),
        ("methanator rated_kw", config.methanator.rated_kw),
        (
            "methanator specific_energy_kwh_per_kg",
            config.methanator.specific_energy_kwh_per_kg,
        ),
        ("methanator H2 ratio", config.methanator.h2_kg_per_kg_ch4),
        ("methanator CO2 ratio", config.methanator.co2_kg_per_kg_ch4),
        ("H2 capacity", config.h2_tank.capacity_kg),
        ("CO2 capacity", config.co2_tank.capacity_kg),
        ("CH4 capacity", config.ch4_tank.capacity_kg),
    ):
        _positive(name, value)
    for name, value in (
        ("battery max_charge_kw", config.battery.max_charge_kw),
        ("battery max_discharge_kw", config.battery.max_discharge_kw),
    ):
        _real(name, value, minimum=0.0)
    for name, value in (
        ("battery charge efficiency", config.battery.charge_efficiency),
        ("battery discharge efficiency", config.battery.discharge_efficiency),
    ):
        efficiency = _positive(name, value)
        if efficiency > 1.0:
            raise DeliveryRelaxationError(f"{name} must not exceed one")
    for name, value in (
        ("battery min_soc", config.battery.min_soc),
        ("active battery reserve", constraints.battery_min_soc),
    ):
        fraction = _real(name, value, minimum=0.0)
        if fraction > 1.0:
            raise DeliveryRelaxationError(f"{name} must not exceed one")
    for field in (
        "electrolyser_max_kw",
        "dac_max_kw",
        "methanator_max_kw",
        "battery_support_max_kw",
    ):
        value = getattr(constraints, field)
        if value is not None:
            _real(f"constraints.{field}", value, minimum=0.0)


def _validate_checkpoint_contract(
    checkpoint_contract: Mapping[str, Any],
    *,
    episode_config: EpisodeConfigV2,
    config: PlantConfig,
    constraints: Constraints,
) -> tuple[str, str]:
    if not isinstance(checkpoint_contract, Mapping):
        raise DeliveryRelaxationError("checkpoint_contract must be a mapping")
    try:
        detached = json.loads(_canonical_json(dict(checkpoint_contract)))
    except (TypeError, ValueError) as exc:
        raise DeliveryRelaxationError(
            "checkpoint_contract must be finite canonical-JSON data"
        ) from exc
    declared = detached.get("checkpoint_contract_digest")
    if not isinstance(declared, str) or len(declared) != 64:
        raise DeliveryRelaxationError(
            "checkpoint contract digest is missing or malformed"
        )
    payload = dict(detached)
    payload.pop("checkpoint_contract_digest", None)
    if contract_digest(payload) != declared:
        raise DeliveryRelaxationError(
            "checkpoint contract digest does not match its payload"
        )
    expected_fields = {
        "plant_config": config.to_dict(),
        "episode_distribution": episode_config.to_dict(),
        "constraints": constraints.to_dict(),
    }
    for field, expected in expected_fields.items():
        # Compare the JSON data model, not Python container implementation
        # details (for example immutable tuples become JSON arrays).
        canonical_expected = json.loads(_canonical_json(expected))
        if detached.get(field) != canonical_expected:
            raise DeliveryRelaxationError(
                f"checkpoint contract {field} does not match the relaxation input"
            )
    timing = detached.get("timing")
    if (
        not isinstance(timing, Mapping)
        or timing.get("physics_timestep_s") != config.dt_s
    ):
        raise DeliveryRelaxationError(
            "checkpoint contract has the wrong physics timestep"
        )
    objective = detached.get("objective")
    weights_payload = (
        objective.get("reward_weights") if isinstance(objective, Mapping) else None
    )
    if not isinstance(weights_payload, Mapping):
        raise DeliveryRelaxationError(
            "checkpoint contract is missing its complete reward identity"
        )
    try:
        weights = RewardWeightsV2(**dict(weights_payload)).validate()
        expected_complete = describe_checkpoint_contract(
            plant=config,
            episode=episode_config,
            constraints=constraints,
            weights=weights,
        )
    except (TypeError, ValueError) as exc:
        raise DeliveryRelaxationError(
            "checkpoint contract reward identity is invalid"
        ) from exc
    if detached != json.loads(_canonical_json(expected_complete)):
        raise DeliveryRelaxationError(
            "checkpoint contract does not describe the current V2 runtime"
        )
    return _canonical_json(detached), declared


def _validate_weather_identity(
    episode: EpisodeV2,
    episode_config: EpisodeConfigV2,
    config: PlantConfig,
) -> None:
    """Recompute the sampled weather identities used by the LP input."""
    weather = episode.weather
    if weather.decision_stride_steps != episode.decision_stride:
        raise DeliveryRelaxationError(
            "weather decision stride does not match the sampled episode"
        )
    if weather.truth_origin_kind != "synthetic_generator":
        raise DeliveryRelaxationError(
            "delivery relaxation currently requires sample_episode synthetic weather"
        )
    if weather.clear_sky_envelope_id != CLEAR_SKY_ENVELOPE_ID:
        raise DeliveryRelaxationError("weather uses an unknown clear-sky envelope")
    if json.loads(_canonical_json(weather.config.to_dict())) != json.loads(
        _canonical_json(episode_config.weather.to_dict())
    ):
        raise DeliveryRelaxationError(
            "sampled weather configuration does not match episode_config"
        )

    expected_truth = _sha256_json(
        {
            "semantics": "factor at boundary k+1 applies to interval [k, k+1)",
            **weather.tape.to_dict(),
        }
    )
    if weather.truth_sha256 != expected_truth:
        raise DeliveryRelaxationError("weather truth hash does not match its tape")
    expected_forecasts = _sha256_json([issue.to_dict() for issue in weather.issues])
    if weather.forecasts_sha256 != expected_forecasts:
        raise DeliveryRelaxationError("weather forecast hash does not match its issues")

    max_forecast_h = max(
        end_h for _, end_h in episode_config.weather.forecast_windows_h
    )
    horizon_steps = int(round(max_forecast_h * 3_600.0 / config.dt_s))
    final_issue_step = episode.physics_steps - episode.decision_stride
    required_intervals = final_issue_step + horizon_steps
    solar = SolarArray(config.solar)
    start_s = episode.scenario.start_hour * 3_600.0
    envelope = tuple(
        solar.clear_sky_kw(start_s + interval * config.dt_s) / config.solar.peak_kw
        for interval in range(required_intervals)
    )
    expected_envelope = _sha256_json(
        {
            "unit": "fraction of PV nameplate peak power",
            "interval_values": envelope,
        }
    )
    if weather.clear_sky_envelope_sha256 != expected_envelope:
        raise DeliveryRelaxationError(
            "weather clear-sky envelope hash does not match the plant and episode"
        )
    identity_payload = {
        "weather_generator_version": WEATHER_GENERATOR_VERSION,
        "config": weather.config.to_dict(),
        "dt_s": weather.dt_s,
        "physics_steps": weather.physics_steps,
        "decision_stride_steps": weather.decision_stride_steps,
        "truth_stream": weather.truth_stream_key.to_dict(),
        "forecast_stream": weather.forecast_stream_key.to_dict(),
        "clear_sky_envelope_id": weather.clear_sky_envelope_id,
        "clear_sky_envelope_sha256": weather.clear_sky_envelope_sha256,
        "truth_origin_kind": weather.truth_origin_kind,
        "truth_sha256": weather.truth_sha256,
        "forecasts_sha256": weather.forecasts_sha256,
        "issue_stream_sha256": [issue._issue_stream_sha256 for issue in weather.issues],
    }
    if weather.episode_sha256 != _sha256_json(identity_payload):
        raise DeliveryRelaxationError("weather episode identity is inconsistent")


def _best_h2_coefficient(
    episode_config: EpisodeConfigV2,
    config: PlantConfig,
) -> tuple[float, str, str | None, str]:
    configured = _positive(
        "electrolyser specific energy",
        config.electrolyser.specific_energy_kwh_per_kg,
    )
    profile = episode_config.electrolyser_commitment
    if profile is None:
        return (
            configured,
            "legacy-memoryless-electrolyser",
            None,
            "fault-free exact yield uses configured specific energy; relaxation uses the same best case",
        )
    profile.validate_timestep(config.dt_s)
    nominal = _positive(
        "commitment nominal specific energy",
        profile.nominal_specific_energy_kwh_per_kg,
    )
    multipliers = tuple(
        _positive(f"specific-energy multiplier {index}", multiplier)
        for index, (_, multiplier) in enumerate(
            profile.specific_energy_multiplier_anchors
        )
    )
    if min(multipliers) < 1.0 or not math.isclose(multipliers[-1], 1.0, abs_tol=1e-12):
        raise DeliveryRelaxationError(
            "commitment profile cannot prove a full-load best-case multiplier of one"
        )
    best = max(configured, nominal)
    proof = (
        "exact commitment yield denominator is max(configured, profile nominal) "
        "times a profile multiplier >= 1 and divided by health <= 1; the fault-free "
        f"relaxation therefore uses the attainable lower denominator {best!r} kWh/kg"
    )
    return best, profile.profile_id, profile.profile_sha256, proof


def _solar_truth(episode: EpisodeV2, config: PlantConfig) -> tuple[float, ...]:
    tape = episode.weather.tape
    if not math.isclose(tape.dt_s, config.dt_s, rel_tol=0.0, abs_tol=1e-12):
        raise DeliveryRelaxationError(
            "weather tape timestep does not match plant timestep"
        )
    if tape.interval_count != episode.physics_steps:
        raise DeliveryRelaxationError(
            "weather tape does not cover the complete episode"
        )
    solar = SolarArray(config.solar)
    start_s = episode.scenario.start_hour * 3_600.0
    values: list[float] = []
    for step in range(episode.physics_steps):
        factor = tape.for_interval(step)
        if factor != tape.boundary_factors[step + 1]:
            raise DeliveryRelaxationError(
                "weather tape violates t+1 interval semantics"
            )
        available = solar.clear_sky_kw(start_s + step * config.dt_s) * factor
        values.append(_real(f"solar truth interval {step}", available, minimum=0.0))
    return tuple(values)


def _solar_digest(values: Sequence[float]) -> str:
    digest = hashlib.sha256()
    digest.update(b"plantanagent-v2-solar-kw-ieee754-v1\0")
    digest.update(struct.pack(">Q", len(values)))
    for value in values:
        digest.update(struct.pack(">d", float(value)))
    return digest.hexdigest()


def build_delivery_relaxation(
    *,
    episode: EpisodeV2,
    episode_config: EpisodeConfigV2,
    config: PlantConfig,
    constraints: Constraints,
    checkpoint_contract: Mapping[str, Any],
    preserve_initial_endowment: bool = False,
    require_canonical_episode: bool = True,
) -> DeliveryRelaxationModel:
    """Build the protocol's physics-step, perfect-truth LP.

    The returned mathematical program is normalized column-wise.  If ``z`` is
    the solver vector, physical variable ``x[i]`` is
    ``model.physical_lower_bounds[i] + z[i] * model.physical_scales[i]``. Every
    normalized and physical bound is finite and serialized.
    """
    if not isinstance(episode, EpisodeV2):
        raise TypeError("episode must be an EpisodeV2")
    if not isinstance(episode_config, EpisodeConfigV2):
        raise TypeError("episode_config must be an EpisodeConfigV2")
    if not isinstance(config, PlantConfig):
        raise TypeError("config must be a PlantConfig")
    if not isinstance(constraints, Constraints):
        raise TypeError("constraints must be Constraints")
    if not isinstance(preserve_initial_endowment, bool):
        raise TypeError("preserve_initial_endowment must be a bool")
    if not isinstance(require_canonical_episode, bool):
        raise TypeError("require_canonical_episode must be a bool")

    episode_config.validate(config)
    _validate_configuration(config, constraints)
    if episode.scenario.faults:
        raise DeliveryRelaxationError(
            "faulted episodes are outside the delivery-relaxation scope"
        )
    n_steps = _integer("episode.physics_steps", episode.physics_steps, minimum=1)
    episode_seed = _integer("episode.seed", episode.seed)
    if require_canonical_episode:
        canonical_episode = sample_episode(episode_config, config, seed=episode_seed)
        if _canonical_json(episode.audit_manifest()) != _canonical_json(
            canonical_episode.audit_manifest()
        ):
            raise DeliveryRelaxationError(
                "episode does not match canonical sample_episode regeneration"
            )
    if n_steps != episode_config.physics_steps(config):
        raise DeliveryRelaxationError("episode horizon does not match episode_config")
    if episode.decision_stride != episode_config.decision_stride(config):
        raise DeliveryRelaxationError(
            "episode decision stride does not match episode_config"
        )
    if episode.delivery_stride != episode_config.delivery_stride(config):
        raise DeliveryRelaxationError(
            "episode delivery stride does not match episode_config"
        )
    if not math.isclose(
        episode.scenario.start_hour,
        episode_config.start_hour,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise DeliveryRelaxationError(
            "episode start hour does not match episode_config"
        )
    if episode.weather.physics_steps != n_steps:
        raise DeliveryRelaxationError(
            "weather horizon does not match the sampled episode"
        )
    _validate_weather_identity(episode, episode_config, config)
    if set(episode.stream_spawn_keys) != set(STREAM_NAMES):
        raise DeliveryRelaxationError(
            "sampled episode does not contain the canonical random streams"
        )
    for stream_name, spawn_key in episode.stream_spawn_keys.items():
        if not isinstance(spawn_key, tuple):
            raise DeliveryRelaxationError(
                f"random stream {stream_name!r} spawn key must be an immutable tuple"
            )
        for index, value in enumerate(spawn_key):
            _integer(f"random stream {stream_name} spawn_key[{index}]", value)

    checkpoint_json, checkpoint_sha = _validate_checkpoint_contract(
        checkpoint_contract,
        episode_config=episode_config,
        config=config,
        constraints=constraints,
    )
    initial = episode.scenario.initial
    missing = [
        name
        for name in ("battery_soc", "h2_kg", "co2_kg", "ch4_kg")
        if getattr(initial, name) is None
    ]
    if missing:
        raise DeliveryRelaxationError(
            "sampled initial state is incomplete: " + ", ".join(missing)
        )

    battery_capacity = _positive("battery capacity", config.battery.capacity_kwh)
    initial_soc = _real("initial battery SOC", initial.battery_soc, minimum=0.0)
    if initial_soc > 1.0:
        raise DeliveryRelaxationError("initial battery SOC must not exceed one")
    initial_energy = initial_soc * battery_capacity
    reserve_soc = constraints.reserve_soc(config)
    reserve_energy = reserve_soc * battery_capacity
    if initial_energy < reserve_energy:
        raise DeliveryRelaxationError(
            "initial battery energy is below the active reserve"
        )

    h2_capacity = _positive("H2 tank capacity", config.h2_tank.capacity_kg)
    co2_capacity = _positive("CO2 tank capacity", config.co2_tank.capacity_kg)
    ch4_capacity = _positive("CH4 tank capacity", config.ch4_tank.capacity_kg)
    initial_h2 = _real("initial H2", initial.h2_kg, minimum=0.0)
    initial_co2 = _real("initial CO2", initial.co2_kg, minimum=0.0)
    initial_ch4 = _real("initial CH4", initial.ch4_kg, minimum=0.0)
    for name, level, capacity in (
        ("H2", initial_h2, h2_capacity),
        ("CO2", initial_co2, co2_capacity),
        ("CH4", initial_ch4, ch4_capacity),
    ):
        if level > capacity:
            raise DeliveryRelaxationError(f"initial {name} exceeds configured capacity")

    commitment_rows: list[tuple[str, int, float]] = []
    seen_ids: set[str] = set()
    previous_due = 0
    for index, commitment in enumerate(episode.commitments):
        if commitment.id in seen_ids:
            raise DeliveryRelaxationError("delivery commitment ids must be unique")
        seen_ids.add(commitment.id)
        due = _integer(f"commitment {index} due_step", commitment.due_step, minimum=1)
        if due > n_steps:
            raise DeliveryRelaxationError("delivery deadline lies beyond the episode")
        if due < previous_due:
            raise DeliveryRelaxationError(
                "delivery commitments must be ordered by deadline"
            )
        previous_due = due
        amount = _real(
            f"commitment {index} requested_kg",
            commitment.requested_kg,
            minimum=0.0,
        )
        if due % episode.delivery_stride:
            raise DeliveryRelaxationError(
                "delivery deadline does not align with episode_config delivery cadence"
            )
        if not math.isclose(
            amount,
            episode_config.delivery_amount_kg,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise DeliveryRelaxationError(
                "delivery request does not match episode_config delivery amount"
            )
        commitment_rows.append((commitment.id, due, amount))

    solar_kw = _solar_truth(episode, config)
    best_h2_energy, profile_id, profile_sha, coefficient_proof = _best_h2_coefficient(
        episode_config,
        config,
    )
    dt_h = _positive("plant dt_h", config.dt_h)
    electrolyser_limit = _real(
        "active electrolyser limit",
        constraints.electrolyser_limit_kw(config),
        minimum=0.0,
    )
    support_limit = _real(
        "active battery support limit",
        constraints.support_limit_kw(config),
        minimum=0.0,
    )
    dac_limit = _real("active DAC limit", constraints.dac_limit_kw(config), minimum=0.0)
    methanator_limit = _real(
        "active methanator limit",
        constraints.methanator_limit_kw(config),
        minimum=0.0,
    )
    support_limit = min(support_limit, config.battery.max_discharge_kw)
    charge_limit = config.battery.max_charge_kw
    dac_energy = config.dac.specific_energy_kwh_per_kg
    methanator_energy = config.methanator.specific_energy_kwh_per_kg
    h2_ratio = config.methanator.h2_kg_per_kg_ch4
    co2_ratio = config.methanator.co2_kg_per_kg_ch4

    variables = _Variables()
    zeros_n = (0.0,) * n_steps
    pv_upper = solar_kw
    pv_fallback = config.solar.peak_kw
    sb = variables.add("solar_to_battery_kw", "kW", zeros_n, pv_upper, pv_fallback)
    se = variables.add("solar_to_electrolyser_kw", "kW", zeros_n, pv_upper, pv_fallback)
    sd = variables.add("solar_to_dac_kw", "kW", zeros_n, pv_upper, pv_fallback)
    sm = variables.add("solar_to_methanator_kw", "kW", zeros_n, pv_upper, pv_fallback)
    bd = variables.add(
        "battery_to_electrolyser_kw",
        "kW",
        zeros_n,
        (support_limit,) * n_steps,
        max(config.battery.max_discharge_kw, 1.0),
    )
    curtail = variables.add("curtailed_solar_kw", "kW", zeros_n, pv_upper, pv_fallback)
    h2_prod_cap = electrolyser_limit * dt_h / best_h2_energy
    co2_prod_cap = dac_limit * dt_h / dac_energy
    ch4_prod_cap = methanator_limit * dt_h / methanator_energy
    hp = variables.add(
        "h2_produced_kg", "kg", zeros_n, (h2_prod_cap,) * n_steps, max(h2_prod_cap, 1.0)
    )
    cp = variables.add(
        "co2_captured_kg",
        "kg",
        zeros_n,
        (co2_prod_cap,) * n_steps,
        max(co2_prod_cap, 1.0),
    )
    mp = variables.add(
        "ch4_produced_kg",
        "kg",
        zeros_n,
        (ch4_prod_cap,) * n_steps,
        max(ch4_prod_cap, 1.0),
    )

    energy_lower = [reserve_energy] * (n_steps + 1)
    h2_lower = [0.0] * (n_steps + 1)
    co2_lower = [0.0] * (n_steps + 1)
    ch4_lower = [0.0] * (n_steps + 1)
    if preserve_initial_endowment:
        energy_lower[-1] = max(energy_lower[-1], initial_energy)
        h2_lower[-1] = initial_h2
        co2_lower[-1] = initial_co2
        ch4_lower[-1] = initial_ch4
    energy = variables.add(
        "battery_energy_kwh",
        "kWh",
        energy_lower,
        (battery_capacity,) * (n_steps + 1),
        battery_capacity,
    )
    h2 = variables.add(
        "h2_inventory_kg",
        "kg",
        h2_lower,
        (h2_capacity,) * (n_steps + 1),
        h2_capacity,
    )
    co2 = variables.add(
        "co2_inventory_kg",
        "kg",
        co2_lower,
        (co2_capacity,) * (n_steps + 1),
        co2_capacity,
    )
    ch4 = variables.add(
        "ch4_inventory_kg",
        "kg",
        ch4_lower,
        (ch4_capacity,) * (n_steps + 1),
        ch4_capacity,
    )
    requested = tuple(item[2] for item in commitment_rows)
    delivery = variables.add(
        "delivered_ch4_kg",
        "kg",
        (0.0,) * len(requested) if requested else (0.0,),
        requested if requested else (0.0,),
        max(max(requested, default=0.0), 1.0),
    )
    # A no-commitment fixture carries one fixed-zero sentinel delivery variable.
    delivery_variable_count = len(requested)

    scales = tuple(variables.scales)
    offsets = tuple(variables.physical_lower)
    lower_normalized = (0.0,) * len(scales)
    upper_normalized = (1.0,) * len(scales)
    objective = [0.0] * len(scales)
    for index in range(delivery_variable_count):
        column = delivery.start + index
        objective[column] = -scales[column]

    eq = _SparseRows()
    ub = _SparseRows()

    def transformed(
        terms: Mapping[int, float],
        rhs: float,
    ) -> tuple[dict[int, float], float]:
        """Substitute x = physical_lower + physical_width * z."""
        normalized: dict[int, float] = {}
        adjusted_rhs = rhs
        for column, multiplier in terms.items():
            adjusted_rhs -= multiplier * offsets[column]
            value = multiplier * scales[column]
            if value != 0.0:
                normalized[column] = value
        return normalized, adjusted_rhs

    def add_eq(terms: Mapping[int, float], rhs: float, *, scale: float) -> None:
        normalized, adjusted_rhs = transformed(terms, rhs)
        eq.add(normalized, adjusted_rhs, scale=scale)

    def add_ub(terms: Mapping[int, float], rhs: float, *, scale: float) -> None:
        normalized, adjusted_rhs = transformed(terms, rhs)
        ub.add(normalized, adjusted_rhs, scale=scale)

    # Initial boundary conditions.
    add_eq({energy.start: 1.0}, initial_energy, scale=battery_capacity)
    add_eq({h2.start: 1.0}, initial_h2, scale=h2_capacity)
    add_eq({co2.start: 1.0}, initial_co2, scale=co2_capacity)
    add_eq({ch4.start: 1.0}, initial_ch4, scale=ch4_capacity)

    deliveries_at_step: dict[int, list[int]] = {}
    for index, (_, due, _) in enumerate(commitment_rows):
        deliveries_at_step.setdefault(due, []).append(index)

    for step, available_kw in enumerate(solar_kw):
        columns = {
            sb.start + step: 1.0,
            se.start + step: 1.0,
            sd.start + step: 1.0,
            sm.start + step: 1.0,
            curtail.start + step: 1.0,
        }
        add_eq(
            columns,
            available_kw,
            scale=available_kw if available_kw > 0.0 else pv_fallback,
        )

        # Boundary recurrences. Delivery due at tick step+1 uses methane made
        # in this interval, exactly matching simulator production ordering.
        add_eq(
            {
                energy.start + step + 1: 1.0,
                energy.start + step: -1.0,
                sb.start + step: -config.battery.charge_efficiency * dt_h,
                bd.start + step: dt_h / config.battery.discharge_efficiency,
            },
            0.0,
            scale=battery_capacity,
        )
        add_eq(
            {
                h2.start + step + 1: 1.0,
                h2.start + step: -1.0,
                hp.start + step: -1.0,
                mp.start + step: h2_ratio,
            },
            0.0,
            scale=h2_capacity,
        )
        add_eq(
            {
                co2.start + step + 1: 1.0,
                co2.start + step: -1.0,
                cp.start + step: -1.0,
                mp.start + step: co2_ratio,
            },
            0.0,
            scale=co2_capacity,
        )
        methane_row = {
            ch4.start + step + 1: 1.0,
            ch4.start + step: -1.0,
            mp.start + step: -1.0,
        }
        for delivery_index in deliveries_at_step.get(step + 1, ()):
            column = delivery.start + delivery_index
            methane_row[column] = 1.0
        add_eq(methane_row, 0.0, scale=ch4_capacity)

        # Active component caps.
        add_ub(
            {
                sb.start + step: 1.0,
            },
            charge_limit,
            scale=max(charge_limit, 1.0),
        )
        add_ub(
            {
                se.start + step: 1.0,
                bd.start + step: 1.0,
            },
            electrolyser_limit,
            scale=max(electrolyser_limit, config.electrolyser.rated_kw, 1.0),
        )
        add_ub(
            {sd.start + step: 1.0},
            dac_limit,
            scale=max(dac_limit, config.dac.rated_kw, 1.0),
        )
        add_ub(
            {sm.start + step: 1.0},
            methanator_limit,
            scale=max(methanator_limit, config.methanator.rated_kw, 1.0),
        )

        # Best-case upper-yield inequalities. Accepted power may produce less,
        # allowing startup, standby, part-load, and full-tank traces to embed.
        add_ub(
            {
                hp.start + step: 1.0,
                se.start + step: -dt_h / best_h2_energy,
                bd.start + step: -dt_h / best_h2_energy,
            },
            0.0,
            scale=max(h2_prod_cap, 1.0),
        )
        add_ub(
            {
                cp.start + step: 1.0,
                sd.start + step: -dt_h / dac_energy,
            },
            0.0,
            scale=max(co2_prod_cap, 1.0),
        )
        add_ub(
            {
                mp.start + step: 1.0,
                sm.start + step: -dt_h / methanator_energy,
            },
            0.0,
            scale=max(ch4_prod_cap, 1.0),
        )

        # Simulator ordering: freshly produced gas/product must fit before
        # same-minute methanation and settlement release storage headroom.
        add_ub(
            {
                h2.start + step: 1.0,
                hp.start + step: 1.0,
            },
            h2_capacity,
            scale=h2_capacity,
        )
        add_ub(
            {
                co2.start + step: 1.0,
                cp.start + step: 1.0,
            },
            co2_capacity,
            scale=co2_capacity,
        )
        add_ub(
            {
                ch4.start + step: 1.0,
                mp.start + step: 1.0,
            },
            ch4_capacity,
            scale=ch4_capacity,
        )

    a_ub, b_ub = ub.build(len(scales))
    a_eq, b_eq = eq.build(len(scales))
    program = FiniteBoxLinearProgram(
        objective=tuple(objective),
        a_ub=a_ub,
        b_ub=b_ub,
        a_eq=a_eq,
        b_eq=b_eq,
        lower_bounds=lower_normalized,
        upper_bounds=upper_normalized,
    )
    manifest_json = _canonical_json(episode.audit_manifest())
    return DeliveryRelaxationModel(
        program=program,
        blocks=tuple(variables.blocks),
        physical_scales=scales,
        physical_lower_bounds=tuple(variables.physical_lower),
        physical_upper_bounds=tuple(variables.physical_upper),
        solar_available_kw=solar_kw,
        solar_truth_sha256=_solar_digest(solar_kw),
        commitments=tuple(commitment_rows),
        physics_steps=n_steps,
        dt_h=dt_h,
        initial_battery_energy_kwh=initial_energy,
        initial_h2_kg=initial_h2,
        initial_co2_kg=initial_co2,
        initial_ch4_kg=initial_ch4,
        battery_capacity_kwh=battery_capacity,
        battery_reserve_kwh=reserve_energy,
        battery_charge_efficiency=config.battery.charge_efficiency,
        battery_discharge_efficiency=config.battery.discharge_efficiency,
        battery_charge_limit_kw=charge_limit,
        electrolyser_limit_kw=electrolyser_limit,
        battery_support_limit_kw=support_limit,
        dac_limit_kw=dac_limit,
        methanator_limit_kw=methanator_limit,
        best_h2_specific_energy_kwh_per_kg=best_h2_energy,
        dac_specific_energy_kwh_per_kg=dac_energy,
        methanator_specific_energy_kwh_per_kg=methanator_energy,
        h2_kg_per_kg_ch4=h2_ratio,
        co2_kg_per_kg_ch4=co2_ratio,
        h2_capacity_kg=h2_capacity,
        co2_capacity_kg=co2_capacity,
        ch4_capacity_kg=ch4_capacity,
        preserve_initial_endowment=preserve_initial_endowment,
        canonical_episode_verified=require_canonical_episode,
        profile_id=profile_id,
        profile_sha256=profile_sha,
        coefficient_proof=coefficient_proof,
        checkpoint_contract_json=checkpoint_json,
        checkpoint_contract_digest=checkpoint_sha,
        episode_manifest_json=manifest_json,
        exogenous_episode_sha256=episode.exogenous_episode_sha256,
    )


def next_float_up(value: Fraction) -> float:
    """Convert an exact rational to the least adjacent float not below it."""
    if not isinstance(value, Fraction):
        raise TypeError("value must be a Fraction")
    rounded = float(value)
    if not math.isfinite(rounded):
        raise OverflowError("exact rational does not have a finite float bound")
    if Fraction.from_float(rounded) < value:
        rounded = math.nextafter(rounded, math.inf)
    if not math.isfinite(rounded):
        raise OverflowError("upward rounding escaped the finite float range")
    if Fraction.from_float(rounded) < value:  # Defensive against non-IEEE runtimes.
        raise ArithmeticError("failed to round exact rational upward")
    return rounded


def box_lagrangian_delivery_certificate(
    program: FiniteBoxLinearProgram,
    *,
    inequality_marginals: Sequence[float],
    equality_marginals: Sequence[float],
    requested_kg: Sequence[float] | float,
) -> DualDeliveryCertificate:
    """Evaluate a valid dual bound exactly, without assuming stationarity.

    SciPy/HiGHS reports ``d objective / d rhs`` marginals.  Inequality
    marginals therefore have the ``<= 0`` convention used here; positive
    numerical noise is projected to zero.  The independent minimization over
    each finite box makes the resulting Lagrangian value dual-feasible for any
    finite projected multipliers.
    """
    if not isinstance(program, FiniteBoxLinearProgram):
        raise TypeError("program must be a FiniteBoxLinearProgram")
    y_ub_float = _finite_tuple("inequality_marginals", inequality_marginals)
    y_eq_float = _finite_tuple("equality_marginals", equality_marginals)
    if len(y_ub_float) != program.a_ub.shape[0]:
        raise ValueError("inequality marginal count does not match A_ub")
    if len(y_eq_float) != program.a_eq.shape[0]:
        raise ValueError("equality marginal count does not match A_eq")
    positive = tuple(value for value in y_ub_float if value > 0.0)
    y_ub = tuple(Fraction.from_float(min(value, 0.0)) for value in y_ub_float)
    y_eq = tuple(Fraction.from_float(value) for value in y_eq_float)

    q = [Fraction.from_float(value) for value in program.objective]
    for row, column, value in zip(
        program.a_ub.row_indices,
        program.a_ub.column_indices,
        program.a_ub.values,
        strict=True,
    ):
        q[column] -= Fraction.from_float(value) * y_ub[row]
    for row, column, value in zip(
        program.a_eq.row_indices,
        program.a_eq.column_indices,
        program.a_eq.values,
        strict=True,
    ):
        q[column] -= Fraction.from_float(value) * y_eq[row]

    lower = sum(
        (
            Fraction.from_float(rhs) * multiplier
            for rhs, multiplier in zip(program.b_ub, y_ub, strict=True)
        ),
        start=Fraction(0),
    )
    lower += sum(
        (
            Fraction.from_float(rhs) * multiplier
            for rhs, multiplier in zip(program.b_eq, y_eq, strict=True)
        ),
        start=Fraction(0),
    )
    for reduced, low, high in zip(
        q,
        program.lower_bounds,
        program.upper_bounds,
        strict=True,
    ):
        low_value = reduced * Fraction.from_float(low)
        high_value = reduced * Fraction.from_float(high)
        lower += min(low_value, high_value)

    if isinstance(requested_kg, Real) and not isinstance(requested_kg, bool):
        requested_values = (_real("requested_kg", requested_kg, minimum=0.0),)
    else:
        requested_values = tuple(
            _real(f"requested_kg[{index}]", value, minimum=0.0)
            for index, value in enumerate(requested_kg)
        )
    requested_exact = sum(
        (Fraction.from_float(value) for value in requested_values),
        start=Fraction(0),
    )
    uncapped_delivery = -lower
    certified = min(requested_exact, uncapped_delivery)
    # The all-zero decision is feasible for the generated delivery LP. A
    # negative delivery ceiling therefore signals incompatible multipliers or
    # a non-delivery test program, and must not be published as mass.
    if certified < 0:
        raise ValueError("dual certificate produced a negative delivery ceiling")
    upper_float = next_float_up(certified)
    return DualDeliveryCertificate(
        minimization_lower_bound_numerator=lower.numerator,
        minimization_lower_bound_denominator=lower.denominator,
        uncapped_delivery_upper_bound_numerator=uncapped_delivery.numerator,
        uncapped_delivery_upper_bound_denominator=uncapped_delivery.denominator,
        requested_kg_numerator=requested_exact.numerator,
        requested_kg_denominator=requested_exact.denominator,
        certified_delivery_upper_bound_numerator=certified.numerator,
        certified_delivery_upper_bound_denominator=certified.denominator,
        dual_certified_delivery_upper_bound_kg=upper_float,
        positive_inequality_marginal_count=len(positive),
        max_positive_inequality_marginal=max(positive, default=0.0),
    )


def _maximum(values: Iterable[float], *, default: float = 0.0) -> float:
    return max((float(value) for value in values), default=default)


def _block_values(
    model: DeliveryRelaxationModel,
    physical: Sequence[float],
    name: str,
) -> tuple[float, ...]:
    block = model.block(name)
    return tuple(float(value) for value in physical[block.start : block.stop])


def _primal_residuals(
    model: DeliveryRelaxationModel,
    normalized: Sequence[float],
    physical: Sequence[float],
) -> PrimalResiduals:
    program = model.program
    program_residuals = evaluate_program_residuals(program, normalized)

    sb = _block_values(model, physical, "solar_to_battery_kw")
    se = _block_values(model, physical, "solar_to_electrolyser_kw")
    sd = _block_values(model, physical, "solar_to_dac_kw")
    sm = _block_values(model, physical, "solar_to_methanator_kw")
    bd = _block_values(model, physical, "battery_to_electrolyser_kw")
    curtail = _block_values(model, physical, "curtailed_solar_kw")
    hp = _block_values(model, physical, "h2_produced_kg")
    cp = _block_values(model, physical, "co2_captured_kg")
    mp = _block_values(model, physical, "ch4_produced_kg")
    energy = _block_values(model, physical, "battery_energy_kwh")
    h2 = _block_values(model, physical, "h2_inventory_kg")
    co2 = _block_values(model, physical, "co2_inventory_kg")
    ch4 = _block_values(model, physical, "ch4_inventory_kg")
    delivered_all = _block_values(model, physical, "delivered_ch4_kg")
    delivered = delivered_all[: len(model.commitments)]
    delivery_by_step: dict[int, float] = {}
    for (_, due, _), value in zip(model.commitments, delivered, strict=True):
        delivery_by_step[due] = delivery_by_step.get(due, 0.0) + value

    solar_residual: list[float] = []
    battery_residual: list[float] = [abs(energy[0] - model.initial_battery_energy_kwh)]
    h2_residual: list[float] = [abs(h2[0] - model.initial_h2_kg)]
    co2_residual: list[float] = [abs(co2[0] - model.initial_co2_kg)]
    ch4_residual: list[float] = [abs(ch4[0] - model.initial_ch4_kg)]
    headroom: list[float] = []
    yield_violations: list[float] = []
    component_power: list[float] = []
    for step in range(model.physics_steps):
        solar_residual.append(
            abs(
                sb[step]
                + se[step]
                + sd[step]
                + sm[step]
                + curtail[step]
                - model.solar_available_kw[step]
            )
        )
        battery_residual.append(
            abs(
                energy[step + 1]
                - energy[step]
                - model.battery_charge_efficiency * sb[step] * model.dt_h
                + bd[step] * model.dt_h / model.battery_discharge_efficiency
            )
        )
        h2_residual.append(
            abs(h2[step + 1] - h2[step] - hp[step] + model.h2_kg_per_kg_ch4 * mp[step])
        )
        co2_residual.append(
            abs(
                co2[step + 1]
                - co2[step]
                - cp[step]
                + model.co2_kg_per_kg_ch4 * mp[step]
            )
        )
        ch4_residual.append(
            abs(
                ch4[step + 1]
                - ch4[step]
                - mp[step]
                + delivery_by_step.get(step + 1, 0.0)
            )
        )
        headroom.extend(
            (
                max(0.0, h2[step] + hp[step] - model.h2_capacity_kg),
                max(0.0, co2[step] + cp[step] - model.co2_capacity_kg),
                max(0.0, ch4[step] + mp[step] - model.ch4_capacity_kg),
            )
        )
        yield_violations.extend(
            (
                max(
                    0.0,
                    hp[step]
                    - (se[step] + bd[step])
                    * model.dt_h
                    / model.best_h2_specific_energy_kwh_per_kg,
                ),
                max(
                    0.0,
                    cp[step]
                    - sd[step] * model.dt_h / model.dac_specific_energy_kwh_per_kg,
                ),
                max(
                    0.0,
                    mp[step]
                    - sm[step]
                    * model.dt_h
                    / model.methanator_specific_energy_kwh_per_kg,
                ),
            )
        )
        component_power.extend(
            (
                max(0.0, sb[step] - model.battery_charge_limit_kw),
                max(
                    0.0,
                    se[step] + bd[step] - model.electrolyser_limit_kw,
                ),
                max(0.0, bd[step] - model.battery_support_limit_kw),
                max(0.0, sd[step] - model.dac_limit_kw),
                max(0.0, sm[step] - model.methanator_limit_kw),
            )
        )
    delivery_violation = _maximum(
        max(0.0, value - request)
        for value, (_, _, request) in zip(delivered, model.commitments, strict=True)
    )
    physical_box_violation = _maximum(
        max(0.0, low - value, value - high)
        for value, low, high in zip(
            physical,
            model.physical_lower_bounds,
            model.physical_upper_bounds,
            strict=True,
        )
    )
    return PrimalResiduals(
        max_equality_normalized=program_residuals.max_equality_absolute,
        max_inequality_violation_normalized=(
            program_residuals.max_inequality_violation
        ),
        max_box_violation_normalized=program_residuals.max_box_violation,
        max_solar_balance_kw=_maximum(solar_residual),
        max_battery_balance_kwh=_maximum(battery_residual),
        max_h2_balance_kg=_maximum(h2_residual),
        max_co2_balance_kg=_maximum(co2_residual),
        max_ch4_balance_kg=_maximum(ch4_residual),
        max_headroom_violation_kg=_maximum(headroom),
        max_yield_violation_kg=_maximum(yield_violations),
        max_delivery_bound_violation_kg=delivery_violation,
        max_physical_box_violation=physical_box_violation,
        max_component_power_violation_kw=_maximum(component_power),
    )


def _make_witness(
    model: DeliveryRelaxationModel,
    normalized: Sequence[float],
) -> DeliveryRelaxationWitness:
    normalized_tuple = tuple(float(value) for value in normalized)
    physical = model.physical_vector(normalized_tuple)
    delivery_values = _block_values(model, physical, "delivered_ch4_kg")
    deliveries = tuple(
        DeliveryWitness(item[0], item[1], item[2], delivery_values[index])
        for index, item in enumerate(model.commitments)
    )
    return DeliveryRelaxationWitness(
        normalized_variables=normalized_tuple,
        solar_to_battery_kw=_block_values(model, physical, "solar_to_battery_kw"),
        solar_to_electrolyser_kw=_block_values(
            model, physical, "solar_to_electrolyser_kw"
        ),
        solar_to_dac_kw=_block_values(model, physical, "solar_to_dac_kw"),
        solar_to_methanator_kw=_block_values(model, physical, "solar_to_methanator_kw"),
        battery_to_electrolyser_kw=_block_values(
            model, physical, "battery_to_electrolyser_kw"
        ),
        curtailed_solar_kw=_block_values(model, physical, "curtailed_solar_kw"),
        h2_produced_kg=_block_values(model, physical, "h2_produced_kg"),
        co2_captured_kg=_block_values(model, physical, "co2_captured_kg"),
        ch4_produced_kg=_block_values(model, physical, "ch4_produced_kg"),
        battery_energy_kwh=_block_values(model, physical, "battery_energy_kwh"),
        h2_inventory_kg=_block_values(model, physical, "h2_inventory_kg"),
        co2_inventory_kg=_block_values(model, physical, "co2_inventory_kg"),
        ch4_inventory_kg=_block_values(model, physical, "ch4_inventory_kg"),
        deliveries=deliveries,
        residuals=_primal_residuals(model, normalized_tuple, physical),
    )


def _linked_highs_version() -> str | None:
    """Best-effort identity of the HiGHS runtime linked inside SciPy.

    A separately installed ``highspy`` package is intentionally not inspected:
    it need not be the library used by ``scipy.optimize.linprog``.
    """
    try:
        core = importlib.import_module("scipy.optimize._highspy._core")
    except (ImportError, ModuleNotFoundError):
        return None
    for name in ("HIGHS_VERSION", "__highs_version__", "__version__"):
        value = getattr(core, name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for name in ("Highs", "_Highs"):
        constructor = getattr(core, name, None)
        if constructor is None:
            continue
        try:
            value = constructor().version()
        except Exception:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _failure_result(
    model: DeliveryRelaxationModel,
    *,
    status: int | None,
    code: str,
    message: str,
    solver_version: str | None,
    solve_settings: SolveSettings,
    elapsed_s: float,
    numerical_objective: float | None = None,
    iterations: int | None = None,
    witness: DeliveryRelaxationWitness | None = None,
    marginal_projection_within_tolerance: bool | None = None,
) -> DeliveryRelaxationResult:
    return DeliveryRelaxationResult(
        schema_version=RESULT_SCHEMA_VERSION,
        success=False,
        status=status,
        failure_code=code,
        message=message,
        solver="scipy.optimize.linprog(method='highs')",
        solver_version=solver_version,
        solve_settings=solve_settings,
        model_sha256=model.model_sha256,
        checkpoint_contract_digest=model.checkpoint_contract_digest,
        exogenous_episode_sha256=model.exogenous_episode_sha256,
        canonical_episode_verified=model.canonical_episode_verified,
        preserve_initial_endowment=model.preserve_initial_endowment,
        is_task_upper_bound=model.is_task_upper_bound,
        requested_kg=model.requested_kg,
        numerical_primal_delivery_kg=(
            None if witness is None else witness.delivered_kg
        ),
        dual_certified_delivery_upper_bound_kg=None,
        relaxed_fill_ceiling=None,
        primal_dual_gap_kg=None,
        numerical_objective=numerical_objective,
        iterations=iterations,
        elapsed_s=elapsed_s,
        marginal_projection_within_tolerance=marginal_projection_within_tolerance,
        # Failed solves never serialize a numeric-ceiling-bearing certificate.
        certificate=None,
        witness=witness,
    )


def solve_delivery_relaxation(
    model: DeliveryRelaxationModel,
    *,
    feasibility_tolerance: float = DEFAULT_FEASIBILITY_TOLERANCE,
    physical_tolerance: float = DEFAULT_PHYSICAL_TOLERANCE,
    optimality_tolerance_kg: float = DEFAULT_OPTIMALITY_TOLERANCE_KG,
    marginal_sign_tolerance: float = DEFAULT_MARGINAL_SIGN_TOLERANCE,
    solver_options: Mapping[str, object] | None = None,
) -> DeliveryRelaxationResult:
    """Solve, independently validate, and (only then) publish a delivery bound."""
    if not isinstance(model, DeliveryRelaxationModel):
        raise TypeError("model must be a DeliveryRelaxationModel")
    feasibility_tolerance = _positive("feasibility_tolerance", feasibility_tolerance)
    physical_tolerance = _positive("physical_tolerance", physical_tolerance)
    optimality_tolerance_kg = _positive(
        "optimality_tolerance_kg", optimality_tolerance_kg
    )
    marginal_sign_tolerance = _real(
        "marginal_sign_tolerance", marginal_sign_tolerance, minimum=0.0
    )
    try:
        options_payload = {} if solver_options is None else dict(solver_options)
        options_json = _canonical_json(options_payload)
        options = json.loads(options_json)
    except (TypeError, ValueError) as exc:
        raise DeliveryRelaxationError(
            "solver_options must be a finite JSON object"
        ) from exc
    if not isinstance(options, dict):
        raise DeliveryRelaxationError("solver_options must be a JSON object")
    solve_settings = SolveSettings(
        feasibility_tolerance=feasibility_tolerance,
        physical_tolerance=physical_tolerance,
        optimality_tolerance_kg=optimality_tolerance_kg,
        marginal_sign_tolerance=marginal_sign_tolerance,
        solver_options_json=options_json,
        highs_runtime_version=None,
    )
    started = time.perf_counter()
    try:
        import scipy
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except (ImportError, ModuleNotFoundError) as exc:
        return _failure_result(
            model,
            status=None,
            code="solver_unavailable",
            message=f"optional SciPy optimisation runtime is unavailable: {exc}",
            solver_version=None,
            solve_settings=solve_settings,
            elapsed_s=time.perf_counter() - started,
        )

    solve_settings = replace(
        solve_settings,
        highs_runtime_version=_linked_highs_version(),
    )

    program = model.program
    try:
        a_ub = coo_matrix(
            (
                np.asarray(program.a_ub.values, dtype=np.float64),
                (
                    np.asarray(program.a_ub.row_indices, dtype=np.int64),
                    np.asarray(program.a_ub.column_indices, dtype=np.int64),
                ),
            ),
            shape=program.a_ub.shape,
        ).tocsr()
        a_eq = coo_matrix(
            (
                np.asarray(program.a_eq.values, dtype=np.float64),
                (
                    np.asarray(program.a_eq.row_indices, dtype=np.int64),
                    np.asarray(program.a_eq.column_indices, dtype=np.int64),
                ),
            ),
            shape=program.a_eq.shape,
        ).tocsr()
        result = linprog(
            c=np.asarray(program.objective, dtype=np.float64),
            A_ub=a_ub,
            b_ub=np.asarray(program.b_ub, dtype=np.float64),
            A_eq=a_eq,
            b_eq=np.asarray(program.b_eq, dtype=np.float64),
            bounds=list(zip(program.lower_bounds, program.upper_bounds, strict=True)),
            method="highs",
            options=options,
        )
    except Exception as exc:  # Solver/API failure is retained, never disguised.
        return _failure_result(
            model,
            status=None,
            code="solver_exception",
            message=f"HiGHS invocation failed: {type(exc).__name__}: {exc}",
            solver_version=scipy.__version__,
            solve_settings=solve_settings,
            elapsed_s=time.perf_counter() - started,
        )

    status = int(result.status) if isinstance(result.status, Integral) else None
    iterations = int(result.nit) if isinstance(result.nit, Integral) else None
    objective = (
        float(result.fun)
        if result.fun is not None and math.isfinite(result.fun)
        else None
    )
    if not bool(result.success) or status != 0:
        return _failure_result(
            model,
            status=status,
            code="solver_not_optimal",
            message=str(result.message),
            solver_version=scipy.__version__,
            solve_settings=solve_settings,
            elapsed_s=time.perf_counter() - started,
            numerical_objective=objective,
            iterations=iterations,
        )
    if (
        result.x is None
        or len(result.x) != program.variable_count
        or not np.all(np.isfinite(result.x))
    ):
        return _failure_result(
            model,
            status=status,
            code="nonfinite_primal",
            message="solver returned a missing, malformed, or non-finite primal vector",
            solver_version=scipy.__version__,
            solve_settings=solve_settings,
            elapsed_s=time.perf_counter() - started,
            numerical_objective=objective,
            iterations=iterations,
        )

    witness = _make_witness(model, result.x)
    residuals = witness.residuals
    if (
        residuals.max_equality_normalized > feasibility_tolerance
        or residuals.max_inequality_violation_normalized > feasibility_tolerance
        or residuals.max_box_violation_normalized > feasibility_tolerance
        or residuals.max_physical_violation > physical_tolerance
    ):
        return _failure_result(
            model,
            status=status,
            code="primal_validation_failed",
            message="independently recomputed primal residual exceeded tolerance",
            solver_version=scipy.__version__,
            solve_settings=solve_settings,
            elapsed_s=time.perf_counter() - started,
            numerical_objective=objective,
            iterations=iterations,
            witness=witness,
        )

    recomputed_objective = math.fsum(
        coefficient * value
        for coefficient, value in zip(
            program.objective,
            witness.normalized_variables,
            strict=True,
        )
    )
    objective_tolerance = max(
        physical_tolerance,
        feasibility_tolerance * max(1.0, model.requested_kg),
    )
    if (
        objective is None
        or abs(objective - recomputed_objective) > objective_tolerance
        or abs(-recomputed_objective - witness.delivered_kg) > objective_tolerance
    ):
        return _failure_result(
            model,
            status=status,
            code="objective_validation_failed",
            message="solver objective, recomputed objective, and delivered mass disagree",
            solver_version=scipy.__version__,
            solve_settings=solve_settings,
            elapsed_s=time.perf_counter() - started,
            numerical_objective=objective,
            iterations=iterations,
            witness=witness,
        )

    try:
        inequality_marginals = tuple(float(value) for value in result.ineqlin.marginals)
        equality_marginals = tuple(float(value) for value in result.eqlin.marginals)
        certificate = box_lagrangian_delivery_certificate(
            program,
            inequality_marginals=inequality_marginals,
            equality_marginals=equality_marginals,
            requested_kg=tuple(item[2] for item in model.commitments),
        )
    except (
        AttributeError,
        TypeError,
        ValueError,
        OverflowError,
        ArithmeticError,
    ) as exc:
        return _failure_result(
            model,
            status=status,
            code="dual_certificate_failed",
            message=f"dual certificate construction failed: {type(exc).__name__}: {exc}",
            solver_version=scipy.__version__,
            solve_settings=solve_settings,
            elapsed_s=time.perf_counter() - started,
            numerical_objective=objective,
            iterations=iterations,
            witness=witness,
        )

    primal = witness.delivered_kg
    upper = certificate.dual_certified_delivery_upper_bound_kg
    gap = upper - primal
    projection_ok = (
        certificate.max_positive_inequality_marginal <= marginal_sign_tolerance
    )
    if certificate.certified_delivery_upper_bound < Fraction.from_float(primal):
        return _failure_result(
            model,
            status=status,
            code="dual_below_primal",
            message="exact-rational dual delivery ceiling lies below the numerical primal",
            solver_version=scipy.__version__,
            solve_settings=solve_settings,
            elapsed_s=time.perf_counter() - started,
            numerical_objective=objective,
            iterations=iterations,
            witness=witness,
            marginal_projection_within_tolerance=projection_ok,
        )
    if gap > optimality_tolerance_kg or not projection_ok:
        reason = (
            "primal-dual gap exceeds tolerance"
            if gap > optimality_tolerance_kg
            else "positive inequality marginal projection exceeds tolerance"
        )
        return _failure_result(
            model,
            status=status,
            code="certificate_not_numerically_tight",
            message=reason,
            solver_version=scipy.__version__,
            solve_settings=solve_settings,
            elapsed_s=time.perf_counter() - started,
            numerical_objective=objective,
            iterations=iterations,
            witness=witness,
            marginal_projection_within_tolerance=projection_ok,
        )
    if primal > model.requested_kg + physical_tolerance:
        return _failure_result(
            model,
            status=status,
            code="delivery_exceeds_request",
            message="numerical primal delivery exceeds requested methane",
            solver_version=scipy.__version__,
            solve_settings=solve_settings,
            elapsed_s=time.perf_counter() - started,
            numerical_objective=objective,
            iterations=iterations,
            witness=witness,
            marginal_projection_within_tolerance=projection_ok,
        )

    fill = 1.0 if model.requested_kg == 0.0 else upper / model.requested_kg
    return DeliveryRelaxationResult(
        schema_version=RESULT_SCHEMA_VERSION,
        success=True,
        status=status,
        failure_code=None,
        message=str(result.message),
        solver="scipy.optimize.linprog(method='highs')",
        solver_version=scipy.__version__,
        solve_settings=solve_settings,
        model_sha256=model.model_sha256,
        checkpoint_contract_digest=model.checkpoint_contract_digest,
        exogenous_episode_sha256=model.exogenous_episode_sha256,
        canonical_episode_verified=model.canonical_episode_verified,
        preserve_initial_endowment=model.preserve_initial_endowment,
        is_task_upper_bound=model.is_task_upper_bound,
        requested_kg=model.requested_kg,
        numerical_primal_delivery_kg=primal,
        dual_certified_delivery_upper_bound_kg=upper,
        relaxed_fill_ceiling=fill,
        primal_dual_gap_kg=gap,
        numerical_objective=objective,
        iterations=iterations,
        elapsed_s=time.perf_counter() - started,
        marginal_projection_within_tolerance=projection_ok,
        certificate=certificate,
        witness=witness,
    )


def evaluate_delivery_relaxation(
    *,
    episode: EpisodeV2,
    episode_config: EpisodeConfigV2,
    config: PlantConfig,
    constraints: Constraints,
    checkpoint_contract: Mapping[str, Any],
    preserve_initial_endowment: bool = False,
    require_canonical_episode: bool = True,
    feasibility_tolerance: float = DEFAULT_FEASIBILITY_TOLERANCE,
    physical_tolerance: float = DEFAULT_PHYSICAL_TOLERANCE,
    optimality_tolerance_kg: float = DEFAULT_OPTIMALITY_TOLERANCE_KG,
    marginal_sign_tolerance: float = DEFAULT_MARGINAL_SIGN_TOLERANCE,
    solver_options: Mapping[str, object] | None = None,
) -> DeliveryRelaxationResult:
    """Build and solve one privileged relaxation without exposing it as control."""
    model = build_delivery_relaxation(
        episode=episode,
        episode_config=episode_config,
        config=config,
        constraints=constraints,
        checkpoint_contract=checkpoint_contract,
        preserve_initial_endowment=preserve_initial_endowment,
        require_canonical_episode=require_canonical_episode,
    )
    return solve_delivery_relaxation(
        model,
        feasibility_tolerance=feasibility_tolerance,
        physical_tolerance=physical_tolerance,
        optimality_tolerance_kg=optimality_tolerance_kg,
        marginal_sign_tolerance=marginal_sign_tolerance,
        solver_options=solver_options,
    )


# Protocol terminology aliases.  The longer implementation names above remain
# descriptive in stack traces; these concise names are the public research API.
RelaxedDeliveryModel = DeliveryRelaxationModel
solve_delivery_upper_bound = solve_delivery_relaxation
solar_truth_kw = _solar_truth
exact_box_lagrangian_certificate = box_lagrangian_delivery_certificate
outward_float = next_float_up


__all__ = [
    "CERTIFICATE_SCOPE",
    "DEFAULT_FEASIBILITY_TOLERANCE",
    "DEFAULT_MARGINAL_SIGN_TOLERANCE",
    "DEFAULT_OPTIMALITY_TOLERANCE_KG",
    "DEFAULT_PHYSICAL_TOLERANCE",
    "DeliveryRelaxationError",
    "DeliveryRelaxationModel",
    "DeliveryRelaxationResult",
    "DeliveryRelaxationWitness",
    "DeliveryWitness",
    "DualDeliveryCertificate",
    "FiniteBoxLinearProgram",
    "MODEL_KIND",
    "MODEL_SCHEMA_VERSION",
    "PRIVILEGED_INFORMATION_LABEL",
    "PrimalResiduals",
    "ProgramResiduals",
    "RelaxedDeliveryModel",
    "RESULT_SCHEMA_VERSION",
    "SIMULATOR_EMBEDDING_STATUS",
    "SolveSettings",
    "SparseCooMatrix",
    "VariableBlock",
    "box_lagrangian_delivery_certificate",
    "build_delivery_relaxation",
    "evaluate_program_residuals",
    "evaluate_delivery_relaxation",
    "exact_box_lagrangian_certificate",
    "next_float_up",
    "outward_float",
    "solar_truth_kw",
    "solve_delivery_relaxation",
    "solve_delivery_upper_bound",
]

"""Canonical sparse MILP construction for causal service MPC.

This module constructs and validates matrix algebra without importing SciPy.  The
optional optimization runtime is isolated in the later solver adapter; unit tests can
therefore audit every variable, constraint, and objective coefficient independently.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping, Sequence, cast

from plant_sim import (
    ElectrolyserCommitmentProfile,
    ElectrolyserCommitmentSnapshot,
    ElectrolyserMode,
    PlantConfig,
)

from ...controllers.base import Constraints
from .control import ControllerObservationV2
from .mpc_forecast import MPCForecastTrajectoryV1
from .mpc_schema import (
    MODEL_BALANCE_TOLERANCE,
    MPCDeliveryTargetV1,
    SERVICE_MPC_MODEL_VERSION,
    ServiceMPCInputError,
    ServiceMPCSettingsV1,
    canonical_json,
    domain_sha256,
    require_sha256,
)

SERVICE_MPC_MODEL_DOMAIN = "service-mpc-sparse-model-v1"
SECONDS_PER_HOUR = 3_600.0


def _finite(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum:g}")
    return number


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    integer = int(value)
    if integer < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return integer


def _aligned_intervals(name: str, seconds: float, interval_s: float) -> int:
    duration = _finite(name, seconds, minimum=0.0)
    if duration <= 0.0:
        return 0
    intervals = int(math.ceil((duration - 1e-9) / interval_s))
    if intervals <= 0:
        raise ValueError(f"{name} cannot be represented on the MPC grid")
    return intervals


@dataclass(frozen=True, slots=True)
class MPCSparseRowV1:
    """One named finite sparse row with optional lower and upper sides."""

    name: str
    lower: float | None
    upper: float | None
    coefficients: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("MPC row name must be non-empty")
        if self.lower is None and self.upper is None:
            raise ValueError("MPC row must have at least one finite side")
        lower = None if self.lower is None else _finite("row lower", self.lower)
        upper = None if self.upper is None else _finite("row upper", self.upper)
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("MPC row lower bound exceeds upper bound")
        if not isinstance(self.coefficients, tuple) or not self.coefficients:
            raise ValueError("MPC row must contain immutable coefficients")
        previous = -1
        for index, coefficient in self.coefficients:
            checked = _integer("row variable index", index)
            if checked <= previous:
                raise ValueError("MPC row variable indices must increase strictly")
            value = _finite("row coefficient", coefficient)
            if value == 0.0:
                raise ValueError("MPC row must not retain zero coefficients")
            previous = checked

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lower": self.lower,
            "upper": self.upper,
            "coefficients": [
                {"variable": index, "coefficient": coefficient}
                for index, coefficient in self.coefficients
            ],
        }


@dataclass(frozen=True, slots=True)
class MPCVariableGroupV1:
    """Canonical extraction indices for one repeated variable family."""

    name: str
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("MPC variable-group name must be non-empty")
        if not isinstance(self.indices, tuple) or not self.indices:
            raise ValueError("MPC variable group must be a non-empty tuple")
        if any(
            isinstance(index, bool) or not isinstance(index, Integral) or index < 0
            for index in self.indices
        ):
            raise ValueError("MPC variable-group indices must be non-negative integers")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("MPC variable-group indices must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "indices": list(self.indices)}


@dataclass(frozen=True, slots=True)
class MPCSolutionAuditV1:
    """Independent residual summary for one candidate model solution."""

    valid: bool
    max_bound_violation: float
    max_row_violation: float
    max_integrality_violation: float
    worst_row: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError("solution-audit valid must be a bool")
        for name in (
            "max_bound_violation",
            "max_row_violation",
            "max_integrality_violation",
        ):
            _finite(name, getattr(self, name), minimum=0.0)
        if self.worst_row is not None and (
            not isinstance(self.worst_row, str) or not self.worst_row
        ):
            raise ValueError("worst_row must be a non-empty string or None")

    def to_dict(self) -> dict[str, bool | float | str | None]:
        return {
            "valid": self.valid,
            "max_bound_violation": self.max_bound_violation,
            "max_row_violation": self.max_row_violation,
            "max_integrality_violation": self.max_integrality_violation,
            "worst_row": self.worst_row,
        }


@dataclass(frozen=True, slots=True)
class ServiceMPCModelV1:
    """Solver-independent finite MILP with a canonical evidence identity."""

    model_version: str
    settings_sha256: str
    forecast_sha256: str
    interval_s: float
    horizon_intervals: int
    deliveries: tuple[MPCDeliveryTargetV1, ...]
    variable_names: tuple[str, ...]
    variable_lower: tuple[float, ...]
    variable_upper: tuple[float, ...]
    integrality: tuple[int, ...]
    primary_objective: tuple[float, ...]
    secondary_objective: tuple[float, ...]
    groups: tuple[MPCVariableGroupV1, ...]
    rows: tuple[MPCSparseRowV1, ...]
    metadata_json: str

    def __post_init__(self) -> None:
        if self.model_version != SERVICE_MPC_MODEL_VERSION:
            raise ValueError("service-MPC model version is unsupported")
        require_sha256("settings_sha256", self.settings_sha256)
        require_sha256("forecast_sha256", self.forecast_sha256)
        interval_s = _finite("interval_s", self.interval_s, minimum=0.0)
        if interval_s <= 0.0:
            raise ValueError("model interval must be positive")
        horizon = _integer("horizon_intervals", self.horizon_intervals, minimum=1)
        if not isinstance(self.deliveries, tuple) or any(
            not isinstance(item, MPCDeliveryTargetV1) for item in self.deliveries
        ):
            raise TypeError("model deliveries must be an immutable typed tuple")
        if any(item.due_boundary > horizon for item in self.deliveries):
            raise ValueError("model delivery lies beyond the horizon")
        if tuple(item.ordinal for item in self.deliveries) != tuple(
            range(len(self.deliveries))
        ):
            raise ValueError("model delivery ordinals are not contiguous")
        if any(
            left.due_boundary >= right.due_boundary
            for left, right in zip(self.deliveries, self.deliveries[1:])
        ):
            raise ValueError("model delivery boundaries must increase strictly")
        if any(
            not math.isclose(
                item.due_offset_s,
                item.due_boundary * interval_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for item in self.deliveries
        ):
            raise ValueError("model delivery offsets disagree with their boundaries")

        sequence_fields = (
            "variable_names",
            "variable_lower",
            "variable_upper",
            "integrality",
            "primary_objective",
            "secondary_objective",
        )
        if any(not isinstance(getattr(self, name), tuple) for name in sequence_fields):
            raise TypeError("model variable vectors must be immutable tuples")
        lengths = {len(getattr(self, name)) for name in sequence_fields}
        if len(lengths) != 1 or not self.variable_names:
            raise ValueError("model variable vectors must be equal and non-empty")
        if len(set(self.variable_names)) != len(self.variable_names) or any(
            not isinstance(name, str) or not name for name in self.variable_names
        ):
            raise ValueError("model variable names must be unique and non-empty")
        for index, (lower, upper, integer, primary, secondary) in enumerate(
            zip(
                self.variable_lower,
                self.variable_upper,
                self.integrality,
                self.primary_objective,
                self.secondary_objective,
                strict=True,
            )
        ):
            low = _finite(f"variable_lower[{index}]", lower)
            high = _finite(f"variable_upper[{index}]", upper)
            if low > high:
                raise ValueError("model variable lower bound exceeds upper bound")
            if integer not in (0, 1):
                raise ValueError("model integrality codes must be zero or one")
            if integer == 1 and (low < 0.0 or high > 1.0):
                raise ValueError("integer variables must be binary in service MPC V1")
            _finite(f"primary_objective[{index}]", primary)
            _finite(f"secondary_objective[{index}]", secondary)

        if not isinstance(self.groups, tuple) or not self.groups:
            raise ValueError("model variable groups must be a non-empty tuple")
        if len({group.name for group in self.groups}) != len(self.groups):
            raise ValueError("model variable-group names must be unique")
        flattened = [index for group in self.groups for index in group.indices]
        if sorted(flattened) != list(range(len(self.variable_names))):
            raise ValueError("model variable groups must partition every variable")
        if not isinstance(self.rows, tuple) or not self.rows:
            raise ValueError("model rows must be a non-empty tuple")
        if len({row.name for row in self.rows}) != len(self.rows):
            raise ValueError("model row names must be unique")
        if any(
            not isinstance(row, MPCSparseRowV1)
            or row.coefficients[-1][0] >= len(self.variable_names)
            for row in self.rows
        ):
            raise ValueError("model row references an invalid variable")
        if not isinstance(self.metadata_json, str):
            raise TypeError("model metadata_json must be a string")
        try:
            metadata = json.loads(self.metadata_json)
        except json.JSONDecodeError as exc:
            raise ValueError("model metadata_json is invalid") from exc
        if not isinstance(metadata, dict):
            raise ValueError("model metadata_json must encode an object")
        if canonical_json(metadata) != self.metadata_json:
            raise ValueError("model metadata_json must be canonical strict JSON")

    @property
    def variable_count(self) -> int:
        return len(self.variable_names)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def nonzero_count(self) -> int:
        return sum(len(row.coefficients) for row in self.rows)

    @property
    def binary_variable_count(self) -> int:
        return sum(self.integrality)

    def group(self, name: str) -> tuple[int, ...]:
        for group in self.groups:
            if group.name == name:
                return group.indices
        raise KeyError(name)

    def metadata(self) -> dict[str, Any]:
        value = json.loads(self.metadata_json)
        if not isinstance(value, dict):
            raise TypeError("model metadata is not an object")
        return value

    def body(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "settings_sha256": self.settings_sha256,
            "forecast_sha256": self.forecast_sha256,
            "interval_s": float(self.interval_s),
            "horizon_intervals": int(self.horizon_intervals),
            "deliveries": [item.to_dict() for item in self.deliveries],
            "variable_names": list(self.variable_names),
            "variable_lower": list(self.variable_lower),
            "variable_upper": list(self.variable_upper),
            "integrality": list(self.integrality),
            "primary_objective": list(self.primary_objective),
            "secondary_objective": list(self.secondary_objective),
            "groups": [group.to_dict() for group in self.groups],
            "rows": [row.to_dict() for row in self.rows],
            "metadata": self.metadata(),
        }

    @property
    def sha256(self) -> str:
        return domain_sha256(SERVICE_MPC_MODEL_DOMAIN, self.body())

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "settings_sha256": self.settings_sha256,
            "forecast_sha256": self.forecast_sha256,
            "interval_s": float(self.interval_s),
            "horizon_intervals": int(self.horizon_intervals),
            "deliveries": [item.to_dict() for item in self.deliveries],
            "dimensions": {
                "variables": self.variable_count,
                "binary_variables": self.binary_variable_count,
                "rows": self.row_count,
                "nonzeros": self.nonzero_count,
            },
            "metadata": self.metadata(),
            "sha256": self.sha256,
        }

    def audit_solution(
        self,
        solution: Sequence[float],
        *,
        tolerance: float = MODEL_BALANCE_TOLERANCE,
        extra_rows: Sequence[MPCSparseRowV1] = (),
    ) -> MPCSolutionAuditV1:
        """Recompute every bound, integrality condition, and sparse row."""

        tolerance_value = _finite("solution tolerance", tolerance, minimum=0.0)
        if tolerance_value <= 0.0:
            raise ValueError("solution tolerance must be positive")
        values = tuple(solution)
        if len(values) != self.variable_count:
            raise ValueError("solution length does not match model variables")
        checked = tuple(
            _finite(f"solution[{index}]", value) for index, value in enumerate(values)
        )
        bound_violation = 0.0
        integrality_violation = 0.0
        for value, lower, upper, integer in zip(
            checked,
            self.variable_lower,
            self.variable_upper,
            self.integrality,
            strict=True,
        ):
            bound_violation = max(
                bound_violation,
                max(0.0, lower - value),
                max(0.0, value - upper),
            )
            if integer:
                integrality_violation = max(
                    integrality_violation, abs(value - round(value))
                )

        rows = (*self.rows, *tuple(extra_rows))
        if len({row.name for row in rows}) != len(rows):
            raise ValueError("solution-audit row names must be unique")
        row_violation = 0.0
        worst_row: str | None = None
        for row in rows:
            if not isinstance(row, MPCSparseRowV1):
                raise TypeError("extra solution-audit rows must be MPCSparseRowV1")
            if row.coefficients[-1][0] >= self.variable_count:
                raise ValueError("solution-audit row references an invalid variable")
            activity = math.fsum(
                coefficient * checked[index] for index, coefficient in row.coefficients
            )
            violation = max(
                0.0 if row.lower is None else row.lower - activity,
                0.0 if row.upper is None else activity - row.upper,
            )
            if violation > row_violation:
                row_violation = violation
                worst_row = row.name
        valid = (
            bound_violation <= tolerance_value
            and row_violation <= tolerance_value
            and integrality_violation <= tolerance_value
        )
        return MPCSolutionAuditV1(
            valid=valid,
            max_bound_violation=bound_violation,
            max_row_violation=row_violation,
            max_integrality_violation=integrality_violation,
            worst_row=worst_row,
        )


class _VariableBuilder:
    def __init__(self) -> None:
        self.names: list[str] = []
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.integrality: list[int] = []
        self.primary: list[float] = []
        self.secondary: list[float] = []
        self.groups: list[MPCVariableGroupV1] = []

    def add_group(
        self,
        name: str,
        count: int,
        *,
        lower: float | Sequence[float] = 0.0,
        upper: float | Sequence[float],
        binary: bool = False,
    ) -> tuple[int, ...]:
        if any(group.name == name for group in self.groups):
            raise ValueError(f"duplicate MPC variable group: {name}")
        size = _integer(f"{name} count", count, minimum=1)

        def expand(value: float | Sequence[float], field: str) -> tuple[float, ...]:
            if isinstance(value, Real) and not isinstance(value, bool):
                scalar = _finite(f"{name} {field}", value)
                return (scalar,) * size
            values = tuple(cast(Sequence[float], value))
            if len(values) != size:
                raise ValueError(f"{name} {field} length does not match group")
            return tuple(
                _finite(f"{name} {field}[{index}]", item)
                for index, item in enumerate(values)
            )

        lows = expand(lower, "lower")
        highs = expand(upper, "upper")
        if any(low > high for low, high in zip(lows, highs, strict=True)):
            raise ValueError(f"{name} contains an inverted bound")
        first = len(self.names)
        indices = tuple(range(first, first + size))
        self.names.extend(f"{name}[{index}]" for index in range(size))
        self.lower.extend(lows)
        self.upper.extend(highs)
        self.integrality.extend([int(binary)] * size)
        self.primary.extend([0.0] * size)
        self.secondary.extend([0.0] * size)
        self.groups.append(MPCVariableGroupV1(name=name, indices=indices))
        return indices


class _RowBuilder:
    def __init__(self) -> None:
        self.rows: list[MPCSparseRowV1] = []
        self.names: set[str] = set()

    def add(
        self,
        name: str,
        coefficients: Mapping[int, float],
        *,
        lower: float | None = None,
        upper: float | None = None,
    ) -> None:
        if name in self.names:
            raise ValueError(f"duplicate MPC row: {name}")
        retained = tuple(
            sorted(
                (int(index), float(value))
                for index, value in coefficients.items()
                if value != 0.0
            )
        )
        self.rows.append(
            MPCSparseRowV1(
                name=name,
                lower=lower,
                upper=upper,
                coefficients=retained,
            )
        )
        self.names.add(name)

    def equality(
        self, name: str, coefficients: Mapping[int, float], value: float
    ) -> None:
        self.add(name, coefficients, lower=value, upper=value)


def infer_delivery_targets(
    observation: ControllerObservationV2,
    *,
    config: PlantConfig,
    interval_s: float,
    horizon_intervals: int,
) -> tuple[MPCDeliveryTargetV1, ...]:
    """Infer only the fixed recurrence declared by the public delivery view."""

    if not isinstance(observation, ControllerObservationV2):
        raise TypeError("observation must be ControllerObservationV2")
    if not isinstance(config, PlantConfig):
        raise TypeError("config must be PlantConfig")
    step_s = _finite("interval_s", interval_s, minimum=0.0)
    horizon = _integer("horizon_intervals", horizon_intervals, minimum=1)
    if step_s <= 0.0:
        raise ValueError("interval_s must be positive")
    delivery = observation.delivery
    if delivery.steps_to_next is None:
        if delivery.next_requested_kg != 0.0:
            raise ServiceMPCInputError(
                "delivery view has a request without a next settlement"
            )
        return ()
    requested = _finite("next_requested_kg", delivery.next_requested_kg, minimum=0.0)
    if requested <= 0.0:
        raise ServiceMPCInputError("announced delivery request must be positive")
    due_s = delivery.steps_to_next * config.dt_s
    period_s = delivery.delivery_period_steps * config.dt_s
    for name, seconds in (("next delivery", due_s), ("delivery period", period_s)):
        intervals = seconds / step_s
        if intervals <= 0.0 or not math.isclose(
            intervals, round(intervals), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ServiceMPCInputError(
                f"{name} does not align to the service-MPC interval"
            )
    due_boundary = int(round(due_s / step_s))
    period_intervals = int(round(period_s / step_s))
    targets: list[MPCDeliveryTargetV1] = []
    current = due_boundary
    while current <= horizon:
        ordinal = len(targets)
        targets.append(
            MPCDeliveryTargetV1(
                ordinal=ordinal,
                due_boundary=current,
                due_offset_s=current * step_s,
                requested_kg=requested,
                source="announced" if ordinal == 0 else "inferred_recurring",
            )
        )
        current += period_intervals
    return tuple(targets)


def _require_nominal_health(observation: ControllerObservationV2) -> None:
    health = observation.state.boundary.health
    for component in ("battery", "electrolyser", "dac", "methanator"):
        item = health.get(component)
        if item is None:
            raise ServiceMPCInputError(f"missing public health for {component}")
        if item.offline or not math.isclose(
            float(item.degradation), 0.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ServiceMPCInputError(
                f"service MPC V1 does not model degraded/offline {component}"
            )


def _validate_commitment(
    snapshot: ElectrolyserCommitmentSnapshot | None,
    profile: ElectrolyserCommitmentProfile | None,
) -> tuple[ElectrolyserCommitmentSnapshot, ElectrolyserCommitmentProfile]:
    if snapshot is None or profile is None:
        raise ServiceMPCInputError(
            "service MPC V1 requires the commitment-alpha profile and snapshot"
        )
    if not isinstance(snapshot, ElectrolyserCommitmentSnapshot):
        raise TypeError("commitment snapshot must be ElectrolyserCommitmentSnapshot")
    if not isinstance(profile, ElectrolyserCommitmentProfile):
        raise TypeError("commitment profile must be ElectrolyserCommitmentProfile")
    if not isinstance(snapshot.mode, ElectrolyserMode):
        raise TypeError("commitment snapshot mode must be ElectrolyserMode")
    if snapshot.profile_sha256 != profile.profile_sha256:
        raise ServiceMPCInputError("commitment snapshot/profile digest mismatch")
    for name in ("profile_id", "technology", "calibration_basis", "dynamics_version"):
        if getattr(snapshot, name) != getattr(profile, name):
            raise ServiceMPCInputError("commitment snapshot/profile metadata mismatch")
    for name in (
        "mode_elapsed_s",
        "startup_remaining_s",
        "cooldown_remaining_s",
        "min_up_remaining_s",
        "min_down_remaining_s",
        "shortfall_elapsed_s",
        "previous_power_kw",
        "ramp_up_headroom_kw",
        "ramp_down_headroom_kw",
    ):
        try:
            _finite(f"commitment snapshot {name}", getattr(snapshot, name), minimum=0.0)
        except (TypeError, ValueError) as exc:
            raise ServiceMPCInputError(
                "commitment snapshot state must be finite and non-negative"
            ) from exc
    return snapshot, profile


def _initial_commitment_schedule(
    snapshot: ElectrolyserCommitmentSnapshot,
    profile: ElectrolyserCommitmentProfile,
    *,
    interval_s: float,
    horizon: int,
) -> tuple[bool, tuple[int, ...], tuple[float, ...], tuple[int, ...], tuple[int, ...]]:
    """Map the public minute state to conservative coarse-grid fixed prefixes.

    Returns initial on-state, warmup flags, warmup offer adjustments above cold-start
    power, forced-on intervals, and forced-off intervals.
    """

    warmup = [0] * horizon
    offer_adjustment = [0.0] * horizon
    force_on: set[int] = set()
    force_off: set[int] = set()
    mode = snapshot.mode
    active_modes = {
        ElectrolyserMode.COLD_STARTING,
        ElectrolyserMode.RUNNING,
        ElectrolyserMode.HOT_STANDBY,
        ElectrolyserMode.HOT_STARTING,
    }
    initial_on = mode in active_modes
    start_kw = profile.start_load_frac
    stable_kw = profile.min_stable_load_frac
    min_up = _aligned_intervals("minimum up time", profile.min_up_s, interval_s)

    if mode is ElectrolyserMode.COLD_STARTING:
        residual = min(
            horizon,
            _aligned_intervals(
                "cold-start remainder", snapshot.startup_remaining_s, interval_s
            ),
        )
        for index in range(residual):
            warmup[index] = 1
            force_on.add(index)
            if snapshot.startup_remaining_s - index * interval_s < interval_s - 1e-9:
                offer_adjustment[index] = stable_kw - start_kw
        force_on.update(range(min(horizon, residual + min_up)))
    elif mode is ElectrolyserMode.RUNNING:
        force_on.update(
            range(
                min(
                    horizon,
                    _aligned_intervals(
                        "minimum-up remainder",
                        snapshot.min_up_remaining_s,
                        interval_s,
                    ),
                )
            )
        )
    elif mode in {ElectrolyserMode.HOT_STANDBY, ElectrolyserMode.HOT_STARTING}:
        # A full stable-load offer is required to trigger/reliably finish the
        # minute-scale hot start. The coarse model conservatively assigns no H2
        # to this complete first interval.
        warmup[0] = 1
        offer_adjustment[0] = stable_kw - start_kw
        force_on.add(0)
        remaining_up = _aligned_intervals(
            "minimum-up remainder", snapshot.min_up_remaining_s, interval_s
        )
        force_on.update(range(min(horizon, max(1, remaining_up))))
    elif mode is ElectrolyserMode.COOLDOWN:
        unavailable_s = max(
            snapshot.cooldown_remaining_s,
            snapshot.min_down_remaining_s,
        )
        force_off.update(
            range(
                min(
                    horizon,
                    _aligned_intervals(
                        "cooldown/minimum-down remainder", unavailable_s, interval_s
                    ),
                )
            )
        )
    elif mode is ElectrolyserMode.OFF:
        force_off.update(
            range(
                min(
                    horizon,
                    _aligned_intervals(
                        "minimum-down remainder",
                        snapshot.min_down_remaining_s,
                        interval_s,
                    ),
                )
            )
        )
    else:  # pragma: no cover - exhaustive enum defence
        raise ServiceMPCInputError(f"unsupported commitment mode: {mode!r}")
    return (
        initial_on,
        tuple(warmup),
        tuple(offer_adjustment),
        tuple(sorted(force_on)),
        tuple(sorted(force_off)),
    )


def build_service_mpc_model(
    observation: ControllerObservationV2,
    forecast: MPCForecastTrajectoryV1,
    *,
    settings: ServiceMPCSettingsV1,
    constraints: Constraints,
    config: PlantConfig,
    commitment_profile: ElectrolyserCommitmentProfile | None,
    horizon_intervals: int,
) -> ServiceMPCModelV1:
    """Build the frozen finite service-MPC model from public inputs only."""

    if not isinstance(observation, ControllerObservationV2):
        raise TypeError("observation must be ControllerObservationV2")
    if not isinstance(forecast, MPCForecastTrajectoryV1):
        raise TypeError("forecast must be MPCForecastTrajectoryV1")
    if not isinstance(settings, ServiceMPCSettingsV1):
        raise TypeError("settings must be ServiceMPCSettingsV1")
    if not isinstance(constraints, Constraints):
        raise TypeError("constraints must be Constraints")
    if not isinstance(config, PlantConfig):
        raise TypeError("config must be PlantConfig")
    horizon = _integer("horizon_intervals", horizon_intervals, minimum=1)
    if (
        horizon > settings.maximum_planning_intervals
        or horizon > forecast.interval_count
    ):
        raise ServiceMPCInputError("model horizon exceeds settings or forecast support")
    if not math.isclose(
        settings.interval_s,
        observation.decision_interval_s,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ServiceMPCInputError("MPC interval differs from controller cadence")
    if not math.isclose(
        forecast.interval_s, settings.interval_s, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ServiceMPCInputError("forecast and MPC intervals differ")
    if not math.isclose(
        forecast.boundary_time_s,
        observation.state.boundary.time_s,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ServiceMPCInputError("forecast projection is for a different boundary")
    if forecast.issued_at_physics_boundary_step != observation.state.boundary.step:
        raise ServiceMPCInputError(
            "forecast projection is for a different physics step"
        )
    if not math.isclose(
        forecast.peak_kw,
        config.solar.peak_kw,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ServiceMPCInputError("forecast projection uses a different solar plant")
    if config.dt_s <= 0.0 or not math.isclose(
        settings.interval_s / config.dt_s,
        round(settings.interval_s / config.dt_s),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ServiceMPCInputError("MPC interval must contain whole physics steps")

    _require_nominal_health(observation)
    snapshot, profile = _validate_commitment(
        observation.state.electrolyser_commitment, commitment_profile
    )
    profile.validate_timestep(config.dt_s)
    boundary = observation.state.boundary
    dt_h = settings.interval_s / SECONDS_PER_HOUR
    forecast_kw = forecast.prefix(horizon)
    deliveries = infer_delivery_targets(
        observation,
        config=config,
        interval_s=settings.interval_s,
        horizon_intervals=horizon,
    )

    battery_capacity = _finite(
        "battery effective capacity", boundary.battery_capacity_kwh, minimum=0.0
    )
    battery_initial = _finite(
        "battery energy", boundary.battery_energy_kwh, minimum=0.0
    )
    if battery_capacity <= 0.0 or battery_initial > battery_capacity + 1e-9:
        raise ServiceMPCInputError("public battery state/capacity is invalid")
    reserve_soc = _finite(
        "active battery reserve SOC",
        constraints.reserve_soc(config),
        minimum=0.0,
        maximum=1.0,
    )
    reserve_energy = reserve_soc * battery_capacity
    if battery_initial < reserve_energy - 1e-9:
        raise ServiceMPCInputError("public battery energy is below the active reserve")
    terminal_energy_floor = max(
        reserve_energy,
        min(
            battery_initial,
            settings.terminal_battery_soc_target * battery_capacity,
        ),
    )
    h2_capacity = _finite("H2 capacity", boundary.h2_capacity_kg, minimum=0.0)
    co2_capacity = _finite("CO2 capacity", boundary.co2_capacity_kg, minimum=0.0)
    ch4_capacity = _finite("CH4 capacity", boundary.ch4_capacity_kg, minimum=0.0)
    h2_initial = _finite("H2 level", boundary.h2_level_kg, minimum=0.0)
    co2_initial = _finite("CO2 level", boundary.co2_level_kg, minimum=0.0)
    ch4_initial = _finite("CH4 level", boundary.ch4_level_kg, minimum=0.0)
    for name, level, capacity in (
        ("H2", h2_initial, h2_capacity),
        ("CO2", co2_initial, co2_capacity),
        ("CH4", ch4_initial, ch4_capacity),
    ):
        if capacity <= 0.0 or level > capacity + 1e-9:
            raise ServiceMPCInputError(f"public {name} state/capacity is invalid")

    electrolyser_cap = _finite(
        "electrolyser limit", constraints.electrolyser_limit_kw(config), minimum=0.0
    )
    dac_cap = _finite("DAC limit", constraints.dac_limit_kw(config), minimum=0.0)
    methanator_cap = _finite(
        "methanator limit", constraints.methanator_limit_kw(config), minimum=0.0
    )
    support_cap = _finite(
        "battery support limit", constraints.support_limit_kw(config), minimum=0.0
    )
    charge_cap = _finite(
        "battery charge limit", config.battery.max_charge_kw, minimum=0.0
    )
    if min(electrolyser_cap, dac_cap, methanator_cap, charge_cap) <= 0.0:
        raise ServiceMPCInputError("service MPC V1 requires positive component limits")
    if support_cap < 0.0:  # Defensive; Constraints already clamps this.
        raise ServiceMPCInputError("battery support limit cannot be negative")

    start_power = profile.start_load_frac * config.electrolyser.rated_kw
    stable_power = profile.min_stable_load_frac * config.electrolyser.rated_kw
    if electrolyser_cap + 1e-9 < start_power:
        raise ServiceMPCInputError("electrolyser cap cannot sustain startup power")
    charge_efficiency = _finite(
        "battery charge efficiency",
        config.battery.charge_efficiency,
        minimum=0.0,
        maximum=1.0,
    )
    discharge_efficiency = _finite(
        "battery discharge efficiency",
        config.battery.discharge_efficiency,
        minimum=0.0,
        maximum=1.0,
    )
    electrolyser_specific_energy = _finite(
        "electrolyser specific energy",
        config.electrolyser.specific_energy_kwh_per_kg,
        minimum=0.0,
    )
    dac_specific_energy = _finite(
        "DAC specific energy",
        config.dac.specific_energy_kwh_per_kg,
        minimum=0.0,
    )
    methanator_specific_energy = _finite(
        "methanator specific energy",
        config.methanator.specific_energy_kwh_per_kg,
        minimum=0.0,
    )
    h2_per_ch4 = _finite(
        "methanator H2 ratio",
        config.methanator.h2_kg_per_kg_ch4,
        minimum=0.0,
    )
    co2_per_ch4 = _finite(
        "methanator CO2 ratio",
        config.methanator.co2_kg_per_kg_ch4,
        minimum=0.0,
    )
    if (
        min(
            charge_efficiency,
            discharge_efficiency,
            electrolyser_specific_energy,
            dac_specific_energy,
            methanator_specific_energy,
            h2_per_ch4,
            co2_per_ch4,
        )
        <= 0.0
    ):
        raise ServiceMPCInputError(
            "service MPC V1 requires positive efficiencies, intensities, and ratios"
        )
    worst_specific_energy = max(
        electrolyser_specific_energy,
        profile.nominal_specific_energy_kwh_per_kg,
    ) * max(multiplier for _, multiplier in profile.specific_energy_multiplier_anchors)
    h2_rate = dt_h / worst_specific_energy
    co2_rate = dt_h / dac_specific_energy
    ch4_rate = dt_h / methanator_specific_energy
    feedstock_capacity = min(
        h2_capacity / h2_per_ch4,
        co2_capacity / co2_per_ch4,
    )
    equivalent_capacity = ch4_capacity + feedstock_capacity

    (
        initial_on,
        initial_warmup,
        initial_offer_adjustment_fraction,
        force_on,
        force_off,
    ) = _initial_commitment_schedule(
        snapshot,
        profile,
        interval_s=settings.interval_s,
        horizon=horizon,
    )
    initial_offer_adjustment = tuple(
        value * config.electrolyser.rated_kw
        for value in initial_offer_adjustment_fraction
    )
    startup_intervals = _aligned_intervals(
        "cold-start duration", profile.cold_start_s, settings.interval_s
    )
    minimum_up_intervals = _aligned_intervals(
        "minimum-up duration", profile.min_up_s, settings.interval_s
    )
    minimum_down_intervals = _aligned_intervals(
        "minimum-down duration", profile.min_down_s, settings.interval_s
    )

    variables = _VariableBuilder()
    max_solar = tuple(float(value) for value in forecast_kw)
    solar_battery = variables.add_group("solar_to_battery_kw", horizon, upper=max_solar)
    solar_electrolyser = variables.add_group(
        "solar_to_electrolyser_kw", horizon, upper=max_solar
    )
    solar_dac = variables.add_group("solar_to_dac_kw", horizon, upper=max_solar)
    solar_methanator = variables.add_group(
        "solar_to_methanator_kw", horizon, upper=max_solar
    )
    solar_curtailed = variables.add_group(
        "solar_curtailed_kw", horizon, upper=max_solar
    )
    battery_support = variables.add_group(
        "battery_support_kw", horizon, upper=support_cap
    )
    startup_power = variables.add_group(
        "electrolyser_startup_kw", horizon, upper=electrolyser_cap
    )
    productive_power = variables.add_group(
        "electrolyser_productive_kw", horizon, upper=electrolyser_cap
    )
    h2_produced = variables.add_group(
        "h2_produced_kg", horizon, upper=electrolyser_cap * h2_rate
    )
    co2_produced = variables.add_group(
        "co2_captured_kg", horizon, upper=dac_cap * co2_rate
    )
    ch4_produced = variables.add_group(
        "ch4_produced_kg", horizon, upper=methanator_cap * ch4_rate
    )
    request_by_boundary = {item.due_boundary: item.requested_kg for item in deliveries}
    delivered = variables.add_group(
        "delivered_kg",
        horizon,
        upper=tuple(
            request_by_boundary.get(index + 1, 0.0) for index in range(horizon)
        ),
    )
    cold_start = variables.add_group("cold_start", horizon, upper=1.0, binary=True)
    warmup = variables.add_group("warmup", horizon, upper=1.0, binary=True)
    running = variables.add_group("running", horizon, upper=1.0, binary=True)
    on = variables.add_group("on", horizon, upper=1.0, binary=True)
    stop = variables.add_group("stop", horizon, upper=1.0, binary=True)
    battery_charging = variables.add_group(
        "battery_charging", horizon, upper=1.0, binary=True
    )

    # A start too near the finite horizon would hide its startup/minimum-up
    # obligation beyond the model. Make it unavailable rather than grant free
    # terminal commitment.
    required_on_intervals = startup_intervals + minimum_up_intervals
    for index in range(max(0, horizon - required_on_intervals + 1), horizon):
        variables.upper[cold_start[index]] = 0.0

    battery_lower = [reserve_energy] * (horizon + 1)
    battery_upper = [battery_capacity] * (horizon + 1)
    battery_lower[0] = battery_upper[0] = battery_initial
    battery_lower[-1] = max(battery_lower[-1], terminal_energy_floor)
    battery_energy = variables.add_group(
        "battery_energy_kwh",
        horizon + 1,
        lower=tuple(battery_lower),
        upper=tuple(battery_upper),
    )

    def state_bounds(
        initial: float,
        capacity: float,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        lower = [0.0] * (horizon + 1)
        upper = [capacity] * (horizon + 1)
        lower[0] = upper[0] = initial
        return tuple(lower), tuple(upper)

    h2_lower, h2_upper = state_bounds(h2_initial, h2_capacity)
    co2_lower, co2_upper = state_bounds(co2_initial, co2_capacity)
    ch4_lower, ch4_upper = state_bounds(ch4_initial, ch4_capacity)
    h2_state = variables.add_group(
        "h2_inventory_kg", horizon + 1, lower=h2_lower, upper=h2_upper
    )
    co2_state = variables.add_group(
        "co2_inventory_kg", horizon + 1, lower=co2_lower, upper=co2_upper
    )
    ch4_state = variables.add_group(
        "ch4_inventory_kg", horizon + 1, lower=ch4_lower, upper=ch4_upper
    )
    minimum_fill = variables.add_group(
        "minimum_commitment_fill", 1, lower=1.0 if not deliveries else 0.0, upper=1.0
    )[0]
    terminal_feedstock = variables.add_group(
        "terminal_feedstock_ch4_equivalent_kg", 1, upper=feedstock_capacity
    )[0]

    for index in delivered:
        variables.primary[index] = -1.0
    variables.secondary[minimum_fill] = -settings.minimum_fill_weight
    variables.secondary[ch4_state[-1]] = (
        -settings.terminal_ch4_equivalent_weight / equivalent_capacity
    )
    variables.secondary[terminal_feedstock] = (
        -settings.terminal_ch4_equivalent_weight / equivalent_capacity
    )
    variables.secondary[battery_energy[-1]] = (
        -settings.terminal_battery_weight / battery_capacity
    )
    for index in cold_start:
        variables.secondary[index] = settings.cold_start_weight / horizon
    solar_energy_denominator_kw = config.solar.peak_kw * horizon
    for index in solar_curtailed:
        variables.secondary[index] = (
            settings.curtailment_weight / solar_energy_denominator_kw
        )
    battery_power_denominator_kw = max(charge_cap, support_cap) * horizon
    for index in (*solar_battery, *battery_support):
        variables.secondary[index] = (
            settings.battery_throughput_weight / battery_power_denominator_kw
        )

    rows = _RowBuilder()
    for time in range(horizon):
        rows.equality(
            f"solar_bus[{time}]",
            {
                solar_battery[time]: 1.0,
                solar_electrolyser[time]: 1.0,
                solar_dac[time]: 1.0,
                solar_methanator[time]: 1.0,
                solar_curtailed[time]: 1.0,
            },
            forecast_kw[time],
        )
        rows.equality(
            f"electrolyser_bus[{time}]",
            {
                solar_electrolyser[time]: 1.0,
                battery_support[time]: 1.0,
                startup_power[time]: -1.0,
                productive_power[time]: -1.0,
            },
            0.0,
        )
        rows.equality(
            f"startup_power[{time}]",
            {
                startup_power[time]: 1.0,
                warmup[time]: -start_power,
            },
            initial_warmup[time] * initial_offer_adjustment[time],
        )
        rows.add(
            f"productive_minimum[{time}]",
            {
                productive_power[time]: 1.0,
                running[time]: -stable_power,
            },
            lower=0.0,
        )
        rows.add(
            f"productive_maximum[{time}]",
            {
                productive_power[time]: 1.0,
                running[time]: -electrolyser_cap,
            },
            upper=0.0,
        )
        rows.equality(
            f"h2_yield[{time}]",
            {h2_produced[time]: 1.0, productive_power[time]: -h2_rate},
            0.0,
        )
        rows.equality(
            f"co2_yield[{time}]",
            {co2_produced[time]: 1.0, solar_dac[time]: -co2_rate},
            0.0,
        )
        rows.equality(
            f"ch4_yield[{time}]",
            {ch4_produced[time]: 1.0, solar_methanator[time]: -ch4_rate},
            0.0,
        )
        rows.add(f"dac_cap[{time}]", {solar_dac[time]: 1.0}, upper=dac_cap)
        rows.add(
            f"methanator_cap[{time}]",
            {solar_methanator[time]: 1.0},
            upper=methanator_cap,
        )
        rows.add(
            f"battery_charge_direction[{time}]",
            {
                solar_battery[time]: 1.0,
                battery_charging[time]: -charge_cap,
            },
            upper=0.0,
        )
        rows.add(
            f"battery_support_direction[{time}]",
            {
                battery_support[time]: 1.0,
                battery_charging[time]: support_cap,
            },
            upper=support_cap,
        )
        rows.equality(
            f"battery_balance[{time}]",
            {
                battery_energy[time + 1]: 1.0,
                battery_energy[time]: -1.0,
                solar_battery[time]: -charge_efficiency * dt_h,
                battery_support[time]: dt_h / discharge_efficiency,
            },
            0.0,
        )
        rows.equality(
            f"h2_balance[{time}]",
            {
                h2_state[time + 1]: 1.0,
                h2_state[time]: -1.0,
                h2_produced[time]: -1.0,
                ch4_produced[time]: h2_per_ch4,
            },
            0.0,
        )
        rows.equality(
            f"co2_balance[{time}]",
            {
                co2_state[time + 1]: 1.0,
                co2_state[time]: -1.0,
                co2_produced[time]: -1.0,
                ch4_produced[time]: co2_per_ch4,
            },
            0.0,
        )
        rows.equality(
            f"ch4_balance[{time}]",
            {
                ch4_state[time + 1]: 1.0,
                ch4_state[time]: -1.0,
                ch4_produced[time]: -1.0,
                delivered[time]: 1.0,
            },
            0.0,
        )
        rows.add(
            f"h2_preconversion_headroom[{time}]",
            {h2_state[time]: 1.0, h2_produced[time]: 1.0},
            upper=h2_capacity,
        )
        rows.add(
            f"co2_preconversion_headroom[{time}]",
            {co2_state[time]: 1.0, co2_produced[time]: 1.0},
            upper=co2_capacity,
        )
        rows.add(
            f"ch4_predelivery_headroom[{time}]",
            {ch4_state[time]: 1.0, ch4_produced[time]: 1.0},
            upper=ch4_capacity,
        )

        recent_starts = range(max(0, time - startup_intervals + 1), time + 1)
        warmup_coefficients = {warmup[time]: 1.0}
        warmup_coefficients.update({cold_start[index]: -1.0 for index in recent_starts})
        rows.equality(
            f"warmup_definition[{time}]",
            warmup_coefficients,
            float(initial_warmup[time]),
        )
        rows.equality(
            f"on_partition[{time}]",
            {on[time]: 1.0, warmup[time]: -1.0, running[time]: -1.0},
            0.0,
        )
        transition = {
            on[time]: 1.0,
            cold_start[time]: -1.0,
            stop[time]: 1.0,
        }
        transition_rhs = float(initial_on) if time == 0 else 0.0
        if time > 0:
            transition[on[time - 1]] = -1.0
        rows.equality(f"on_transition[{time}]", transition, transition_rhs)
        rows.add(
            f"exclusive_start_stop[{time}]",
            {cold_start[time]: 1.0, stop[time]: 1.0},
            upper=1.0,
        )
        if time in force_on:
            rows.equality(f"initial_force_on[{time}]", {on[time]: 1.0}, 1.0)
        if time in force_off:
            rows.equality(f"initial_force_off[{time}]", {on[time]: 1.0}, 0.0)

    for start_time in range(horizon):
        for occupied in range(
            start_time,
            min(horizon, start_time + required_on_intervals),
        ):
            rows.add(
                f"minimum_up[{start_time},{occupied}]",
                {on[occupied]: 1.0, cold_start[start_time]: -1.0},
                lower=0.0,
            )
    for stop_time in range(horizon):
        for inactive in range(
            stop_time,
            min(horizon, stop_time + minimum_down_intervals),
        ):
            rows.add(
                f"minimum_down[{stop_time},{inactive}]",
                {on[inactive]: 1.0, stop[stop_time]: 1.0},
                upper=1.0,
            )
    for target in deliveries:
        rows.add(
            f"minimum_fill[{target.ordinal}]",
            {
                delivered[target.due_boundary - 1]: 1.0,
                minimum_fill: -target.requested_kg,
            },
            lower=0.0,
        )
    rows.add(
        "terminal_feedstock_h2",
        {terminal_feedstock: 1.0, h2_state[-1]: -1.0 / h2_per_ch4},
        upper=0.0,
    )
    rows.add(
        "terminal_feedstock_co2",
        {terminal_feedstock: 1.0, co2_state[-1]: -1.0 / co2_per_ch4},
        upper=0.0,
    )

    metadata = {
        "formulation_scope": (
            "deterministic public-issued-mean service model; conservative 15-minute "
            "off/cold-start/running commitment approximation; exact simulator remains "
            "authoritative"
        ),
        "initial_state": {
            "battery_energy_kwh": battery_initial,
            "battery_capacity_kwh": battery_capacity,
            "h2_kg": h2_initial,
            "co2_kg": co2_initial,
            "ch4_kg": ch4_initial,
            "commitment": snapshot.to_dict(),
        },
        "terminal_battery_energy_floor_kwh": terminal_energy_floor,
        "constants": {
            "dt_h": dt_h,
            "forecast_available_kw": list(forecast_kw),
            "electrolyser_cap_kw": electrolyser_cap,
            "dac_cap_kw": dac_cap,
            "methanator_cap_kw": methanator_cap,
            "battery_support_cap_kw": support_cap,
            "battery_charge_cap_kw": charge_cap,
            "battery_reserve_energy_kwh": reserve_energy,
            "electrolyser_start_power_kw": start_power,
            "electrolyser_minimum_stable_power_kw": stable_power,
            "conservative_electrolyser_specific_energy_kwh_per_kg": worst_specific_energy,
            "startup_intervals": startup_intervals,
            "minimum_up_intervals": minimum_up_intervals,
            "minimum_down_intervals": minimum_down_intervals,
            "initial_on": initial_on,
            "initial_warmup": list(initial_warmup),
            "initial_warmup_offer_adjustment_kw": list(initial_offer_adjustment),
            "initial_force_on": list(force_on),
            "initial_force_off": list(force_off),
            "h2_kg_per_kg_ch4": h2_per_ch4,
            "co2_kg_per_kg_ch4": co2_per_ch4,
            "maximum_ch4_equivalent_inventory_kg": equivalent_capacity,
        },
    }
    return ServiceMPCModelV1(
        model_version=SERVICE_MPC_MODEL_VERSION,
        settings_sha256=settings.sha256,
        forecast_sha256=forecast.sha256,
        interval_s=settings.interval_s,
        horizon_intervals=horizon,
        deliveries=deliveries,
        variable_names=tuple(variables.names),
        variable_lower=tuple(variables.lower),
        variable_upper=tuple(variables.upper),
        integrality=tuple(variables.integrality),
        primary_objective=tuple(variables.primary),
        secondary_objective=tuple(variables.secondary),
        groups=tuple(variables.groups),
        rows=tuple(rows.rows),
        metadata_json=canonical_json(metadata),
    )


def delivery_lock_row(
    model: ServiceMPCModelV1,
    *,
    primary_delivered_kg: float,
    tolerance_kg: float,
) -> MPCSparseRowV1:
    """Construct the deterministic second-stage primary-delivery lock."""

    if not isinstance(model, ServiceMPCModelV1):
        raise TypeError("model must be ServiceMPCModelV1")
    optimum = _finite("primary_delivered_kg", primary_delivered_kg, minimum=0.0)
    tolerance = _finite("tolerance_kg", tolerance_kg, minimum=0.0)
    if tolerance <= 0.0:
        raise ValueError("delivery-lock tolerance must be positive")
    delivered = model.group("delivered_kg")
    return MPCSparseRowV1(
        name="primary_delivery_lock",
        lower=max(0.0, optimum - tolerance),
        upper=None,
        coefficients=tuple((index, 1.0) for index in delivered),
    )


__all__ = [
    "MPCSolutionAuditV1",
    "MPCSparseRowV1",
    "MPCVariableGroupV1",
    "SERVICE_MPC_MODEL_DOMAIN",
    "ServiceMPCModelV1",
    "build_service_mpc_model",
    "delivery_lock_row",
    "infer_delivery_targets",
]

"""Immutable contracts for the causal service-MPC research controller."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, ClassVar

from .mpc_forecast import FORECAST_PROJECTION_VERSION

SERVICE_MPC_SCHEMA_VERSION = 1
SERVICE_MPC_CONTROLLER_ID = "service-mpc-issued-mean-v1"
SERVICE_MPC_MODEL_VERSION = "service-mpc-linear-commitment-v1"
SERVICE_MPC_OBJECTIVE_VERSION = "delivery-lock-service-endowment-v1"
SERVICE_MPC_SOLVER = "scipy.optimize.milp"
SERVICE_MPC_PLAN_DOMAIN = "service-mpc-plan-v1"
SERVICE_MPC_SETTINGS_DOMAIN = "service-mpc-settings-v1"
SERVICE_MPC_SOLUTION_DOMAIN = "service-mpc-solution-vector-v1"
MODEL_BALANCE_TOLERANCE = 1e-7


def canonical_json(value: object) -> str:
    """Return the strict canonical JSON used by all MPC evidence identities."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def domain_sha256(domain: str, value: object) -> str:
    if not isinstance(domain, str) or not domain:
        raise ValueError("hash domain must be a non-empty string")
    return hashlib.sha256(
        f"{domain}\0{canonical_json(value)}".encode("utf-8")
    ).hexdigest()


def require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 digest") from exc
    return value


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


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class ServiceMPCSettingsV1:
    """Complete predeclared tuning and solver surface for service MPC V1."""

    task_horizon_h: float = 72.0
    planning_horizon_h: float = 24.0
    interval_s: float = 900.0
    terminal_battery_soc_target: float = 0.25
    delivery_lock_tolerance_kg: float = 1e-6
    energy_reconstruction_tolerance: float = 1e-12
    solver_presolve: bool = True
    solver_mip_rel_gap: float = 1e-6
    solver_time_limit_s: float = 2.0
    minimum_fill_weight: float = 100.0
    terminal_ch4_equivalent_weight: float = 10.0
    terminal_battery_weight: float = 1.0
    cold_start_weight: float = 0.10
    curtailment_weight: float = 0.01
    battery_throughput_weight: float = 0.01

    schema_version: ClassVar[int] = SERVICE_MPC_SCHEMA_VERSION
    controller_id: ClassVar[str] = SERVICE_MPC_CONTROLLER_ID
    model_version: ClassVar[str] = SERVICE_MPC_MODEL_VERSION
    objective_version: ClassVar[str] = SERVICE_MPC_OBJECTIVE_VERSION
    forecast_projection_version: ClassVar[str] = FORECAST_PROJECTION_VERSION
    solver: ClassVar[str] = SERVICE_MPC_SOLVER

    def __post_init__(self) -> None:
        task_h = _finite("task_horizon_h", self.task_horizon_h, minimum=0.0)
        planning_h = _finite("planning_horizon_h", self.planning_horizon_h, minimum=0.0)
        interval_s = _finite("interval_s", self.interval_s, minimum=0.0)
        if task_h <= 0.0 or planning_h <= 0.0 or interval_s <= 0.0:
            raise ValueError("task, planning, and interval durations must be positive")
        if planning_h > 24.0:
            raise ValueError("planning_horizon_h cannot exceed public forecast support")
        if planning_h > task_h:
            raise ValueError("planning_horizon_h cannot exceed task_horizon_h")
        for name, hours in (
            ("task_horizon_h", task_h),
            ("planning_horizon_h", planning_h),
        ):
            intervals = hours * 3_600.0 / interval_s
            if not math.isclose(intervals, round(intervals), rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"{name} must contain whole optimization intervals")
        _finite(
            "terminal_battery_soc_target",
            self.terminal_battery_soc_target,
            minimum=0.0,
            maximum=1.0,
        )
        delivery_tolerance = _finite(
            "delivery_lock_tolerance_kg",
            self.delivery_lock_tolerance_kg,
            minimum=0.0,
        )
        reconstruction_tolerance = _finite(
            "energy_reconstruction_tolerance",
            self.energy_reconstruction_tolerance,
            minimum=0.0,
        )
        if delivery_tolerance <= 0.0 or reconstruction_tolerance <= 0.0:
            raise ValueError("numerical tolerances must be positive")
        if not isinstance(self.solver_presolve, bool):
            raise TypeError("solver_presolve must be a bool")
        _finite("solver_mip_rel_gap", self.solver_mip_rel_gap, minimum=0.0, maximum=1.0)
        time_limit = _finite(
            "solver_time_limit_s", self.solver_time_limit_s, minimum=0.0
        )
        if time_limit <= 0.0:
            raise ValueError("solver_time_limit_s must be positive")
        for name in (
            "minimum_fill_weight",
            "terminal_ch4_equivalent_weight",
            "terminal_battery_weight",
            "cold_start_weight",
            "curtailment_weight",
            "battery_throughput_weight",
        ):
            _finite(name, getattr(self, name), minimum=0.0)
        if self.minimum_fill_weight <= 0.0:
            raise ValueError("minimum_fill_weight must be positive")

    @property
    def task_interval_count(self) -> int:
        return int(round(self.task_horizon_h * 3_600.0 / self.interval_s))

    @property
    def maximum_planning_intervals(self) -> int:
        return int(round(self.planning_horizon_h * 3_600.0 / self.interval_s))

    def body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "controller_id": self.controller_id,
            "model_version": self.model_version,
            "objective_version": self.objective_version,
            "forecast_projection_version": self.forecast_projection_version,
            "forecast_statistic": "issued_mean",
            "solver": self.solver,
            "controller_fallback": None,
            "task_horizon_h": float(self.task_horizon_h),
            "planning_horizon_h": float(self.planning_horizon_h),
            "interval_s": float(self.interval_s),
            "terminal_battery_soc_target": float(self.terminal_battery_soc_target),
            "delivery_lock_tolerance_kg": float(self.delivery_lock_tolerance_kg),
            "energy_reconstruction_tolerance": float(
                self.energy_reconstruction_tolerance
            ),
            "solver_options": {
                "presolve": self.solver_presolve,
                "mip_rel_gap": float(self.solver_mip_rel_gap),
                "time_limit": float(self.solver_time_limit_s),
            },
            "secondary_objective_weights": {
                "minimum_commitment_fill": float(self.minimum_fill_weight),
                "terminal_ch4_equivalent_inventory": float(
                    self.terminal_ch4_equivalent_weight
                ),
                "terminal_battery_energy": float(self.terminal_battery_weight),
                "cold_starts": float(self.cold_start_weight),
                "curtailed_energy": float(self.curtailment_weight),
                "battery_throughput": float(self.battery_throughput_weight),
            },
            "task_interval_count": self.task_interval_count,
            "maximum_planning_intervals": self.maximum_planning_intervals,
        }

    @property
    def sha256(self) -> str:
        return domain_sha256(SERVICE_MPC_SETTINGS_DOMAIN, self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


DEFAULT_SERVICE_MPC_SETTINGS_V1 = ServiceMPCSettingsV1()


@dataclass(frozen=True, slots=True)
class MPCDeliveryTargetV1:
    """One announced or causally inferred settlement in the MPC horizon."""

    ordinal: int
    due_boundary: int
    due_offset_s: float
    requested_kg: float
    source: str

    def __post_init__(self) -> None:
        for name in ("ordinal", "due_boundary"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"delivery target {name} must be an integer")
        if self.ordinal < 0 or self.due_boundary <= 0:
            raise ValueError("delivery target ordinal/boundary is invalid")
        due_s = _finite("due_offset_s", self.due_offset_s, minimum=0.0)
        requested = _finite("requested_kg", self.requested_kg, minimum=0.0)
        if due_s <= 0.0 or requested <= 0.0:
            raise ValueError("delivery target offset and request must be positive")
        if self.source not in {"announced", "inferred_recurring"}:
            raise ValueError("delivery target source is invalid")
        if self.ordinal == 0 and self.source != "announced":
            raise ValueError("first delivery target must be the announced request")
        if self.ordinal > 0 and self.source != "inferred_recurring":
            raise ValueError("later delivery targets must be explicitly inferred")

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "ordinal": int(self.ordinal),
            "due_boundary": int(self.due_boundary),
            "due_offset_s": float(self.due_offset_s),
            "requested_kg": float(self.requested_kg),
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class MPCStageSolveV1:
    """Bounded solver-status evidence for one lexicographic stage."""

    stage: str
    success: bool
    status: int
    message: str
    elapsed_ns: int
    objective_value: float | None
    mip_dual_bound: float | None
    mip_gap: float | None
    mip_node_count: int | None
    solution_sha256: str | None
    scipy_version: str
    highs_version: str

    def __post_init__(self) -> None:
        if self.stage not in {"primary_delivery", "secondary_tiebreak"}:
            raise ValueError("MPC solve stage is invalid")
        if not isinstance(self.success, bool):
            raise TypeError("MPC solve success must be a bool")
        if isinstance(self.status, bool) or not isinstance(self.status, Integral):
            raise TypeError("MPC solve status must be an integer")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("MPC solve message must be non-empty")
        if len(self.message) > 500:
            raise ValueError("MPC solve message exceeds the evidence bound")
        if (
            isinstance(self.elapsed_ns, bool)
            or not isinstance(self.elapsed_ns, Integral)
            or self.elapsed_ns < 0
        ):
            raise ValueError("MPC solve elapsed_ns must be non-negative")
        for name in ("scipy_version", "highs_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty")
        if self.success:
            if self.status != 0:
                raise ValueError("successful MPC solve requires optimal status zero")
            for name in ("objective_value", "mip_dual_bound", "mip_gap"):
                if getattr(self, name) is None:
                    raise ValueError(f"successful MPC solve requires {name}")
                _finite(name, getattr(self, name))
            if self.mip_gap is not None and self.mip_gap < 0.0:
                raise ValueError("MPC MIP gap must be non-negative")
            if (
                self.mip_node_count is None
                or isinstance(self.mip_node_count, bool)
                or not isinstance(self.mip_node_count, Integral)
                or self.mip_node_count < 0
            ):
                raise ValueError("successful MPC solve requires a node count")
            require_sha256("solution_sha256", self.solution_sha256)
        else:
            if self.status == 0:
                raise ValueError("failed MPC solve cannot carry optimal status zero")
            if self.solution_sha256 is not None:
                raise ValueError("failed MPC solve cannot publish a solution digest")
            for name in ("objective_value", "mip_dual_bound", "mip_gap"):
                value = getattr(self, name)
                if value is not None:
                    _finite(name, value)
            if self.mip_node_count is not None:
                if (
                    isinstance(self.mip_node_count, bool)
                    or not isinstance(self.mip_node_count, Integral)
                    or self.mip_node_count < 0
                ):
                    raise ValueError("MPC node count must be non-negative when present")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "success": self.success,
            "status": int(self.status),
            "message": self.message,
            "elapsed_ns": int(self.elapsed_ns),
            "objective_value": self.objective_value,
            "mip_dual_bound": self.mip_dual_bound,
            "mip_gap": self.mip_gap,
            "mip_node_count": self.mip_node_count,
            "solution_sha256": self.solution_sha256,
            "solver": SERVICE_MPC_SOLVER,
            "scipy_version": self.scipy_version,
            "highs_version": self.highs_version,
        }


@dataclass(frozen=True, slots=True)
class MPCPlanStepV1:
    """Selected model quantities for one future control interval."""

    index: int
    forecast_solar_kw: float
    solar_to_battery_kw: float
    solar_to_electrolyser_kw: float
    solar_to_dac_kw: float
    solar_to_methanator_kw: float
    solar_curtailed_kw: float
    battery_support_kw: float
    electrolyser_startup_kw: float
    electrolyser_productive_kw: float
    h2_produced_kg: float
    co2_captured_kg: float
    ch4_produced_kg: float
    delivered_kg: float
    battery_energy_end_kwh: float
    h2_end_kg: float
    co2_end_kg: float
    ch4_end_kg: float
    cold_start: bool
    warmup: bool
    running: bool
    battery_charging: bool

    _NON_NEGATIVE_FIELDS: ClassVar[tuple[str, ...]] = (
        "forecast_solar_kw",
        "solar_to_battery_kw",
        "solar_to_electrolyser_kw",
        "solar_to_dac_kw",
        "solar_to_methanator_kw",
        "solar_curtailed_kw",
        "battery_support_kw",
        "electrolyser_startup_kw",
        "electrolyser_productive_kw",
        "h2_produced_kg",
        "co2_captured_kg",
        "ch4_produced_kg",
        "delivered_kg",
        "battery_energy_end_kwh",
        "h2_end_kg",
        "co2_end_kg",
        "ch4_end_kg",
    )

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, Integral):
            raise TypeError("MPC plan-step index must be an integer")
        if self.index < 0:
            raise ValueError("MPC plan-step index must be non-negative")
        for name in self._NON_NEGATIVE_FIELDS:
            _finite(name, getattr(self, name), minimum=0.0)
        for name in ("cold_start", "warmup", "running", "battery_charging"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"MPC plan-step {name} must be a bool")
        if self.warmup and self.running:
            raise ValueError("MPC interval cannot be both warmup and productive")
        solar_routed = math.fsum(
            (
                self.solar_to_battery_kw,
                self.solar_to_electrolyser_kw,
                self.solar_to_dac_kw,
                self.solar_to_methanator_kw,
                self.solar_curtailed_kw,
            )
        )
        if not math.isclose(
            solar_routed,
            self.forecast_solar_kw,
            rel_tol=0.0,
            abs_tol=MODEL_BALANCE_TOLERANCE,
        ):
            raise ValueError("MPC plan-step solar bus does not balance")
        if not math.isclose(
            self.solar_to_electrolyser_kw + self.battery_support_kw,
            self.electrolyser_startup_kw + self.electrolyser_productive_kw,
            rel_tol=0.0,
            abs_tol=MODEL_BALANCE_TOLERANCE,
        ):
            raise ValueError("MPC plan-step electrolyser bus does not balance")
        if self.battery_charging and self.battery_support_kw > MODEL_BALANCE_TOLERANCE:
            raise ValueError("MPC plan-step cannot charge and discharge simultaneously")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MPCPlanV1:
    """Complete selected second-stage plan and its two solver attestations."""

    model_sha256: str
    forecast_sha256: str
    settings_sha256: str
    horizon_intervals: int
    deliveries: tuple[MPCDeliveryTargetV1, ...]
    stages: tuple[MPCStageSolveV1, MPCStageSolveV1]
    steps: tuple[MPCPlanStepV1, ...]
    primary_delivered_optimum_kg: float
    selected_delivered_kg: float
    delivery_lock_tolerance_kg: float
    minimum_commitment_fill: float
    terminal_ch4_equivalent_kg: float
    secondary_utility: float

    def __post_init__(self) -> None:
        for name in ("model_sha256", "forecast_sha256", "settings_sha256"):
            require_sha256(name, getattr(self, name))
        count = _positive_integer("horizon_intervals", self.horizon_intervals)
        if not isinstance(self.deliveries, tuple) or any(
            not isinstance(item, MPCDeliveryTargetV1) for item in self.deliveries
        ):
            raise TypeError("MPC plan deliveries must be an immutable typed tuple")
        if not isinstance(self.stages, tuple) or len(self.stages) != 2:
            raise ValueError("MPC plan must retain exactly two solve stages")
        if tuple(stage.stage for stage in self.stages) != (
            "primary_delivery",
            "secondary_tiebreak",
        ):
            raise ValueError("MPC solve stages are missing or out of order")
        if not all(stage.success for stage in self.stages):
            raise ValueError("published MPC plan requires two successful solve stages")
        if not isinstance(self.steps, tuple) or len(self.steps) != count:
            raise ValueError("MPC plan steps do not cover the declared horizon")
        if any(
            not isinstance(step, MPCPlanStepV1) or step.index != index
            for index, step in enumerate(self.steps)
        ):
            raise ValueError("MPC plan steps are not typed and contiguous")
        if tuple(item.ordinal for item in self.deliveries) != tuple(
            range(len(self.deliveries))
        ):
            raise ValueError("MPC delivery ordinals are not contiguous")
        if any(item.due_boundary > count for item in self.deliveries):
            raise ValueError("MPC delivery lies beyond the selected horizon")
        primary = _finite(
            "primary_delivered_optimum_kg",
            self.primary_delivered_optimum_kg,
            minimum=0.0,
        )
        selected = _finite(
            "selected_delivered_kg", self.selected_delivered_kg, minimum=0.0
        )
        lock_tolerance = _finite(
            "delivery_lock_tolerance_kg",
            self.delivery_lock_tolerance_kg,
            minimum=0.0,
        )
        if lock_tolerance <= 0.0:
            raise ValueError("MPC delivery-lock tolerance must be positive")
        fill = _finite(
            "minimum_commitment_fill",
            self.minimum_commitment_fill,
            minimum=0.0,
            maximum=1.0,
        )
        _finite(
            "terminal_ch4_equivalent_kg",
            self.terminal_ch4_equivalent_kg,
            minimum=0.0,
        )
        _finite("secondary_utility", self.secondary_utility)
        if not math.isclose(
            math.fsum(step.delivered_kg for step in self.steps),
            selected,
            rel_tol=0.0,
            abs_tol=MODEL_BALANCE_TOLERANCE,
        ):
            raise ValueError("MPC plan selected delivery does not match its steps")
        if selected > primary + lock_tolerance + MODEL_BALANCE_TOLERANCE:
            raise ValueError("selected delivery exceeds the primary optimum")
        if primary - selected > lock_tolerance + MODEL_BALANCE_TOLERANCE:
            raise ValueError("selected delivery violates the primary delivery lock")
        if not self.deliveries and (selected > MODEL_BALANCE_TOLERANCE or fill != 1.0):
            raise ValueError(
                "delivery-free MPC horizon must have zero delivery and unit fill"
            )
        if self.deliveries:
            observed_fill = min(
                self.steps[item.due_boundary - 1].delivered_kg / item.requested_kg
                for item in self.deliveries
            )
            if any(
                self.steps[item.due_boundary - 1].delivered_kg
                > item.requested_kg + MODEL_BALANCE_TOLERANCE
                for item in self.deliveries
            ):
                raise ValueError("MPC plan delivers above a declared request")
            if not math.isclose(
                observed_fill,
                fill,
                rel_tol=0.0,
                abs_tol=MODEL_BALANCE_TOLERANCE,
            ):
                raise ValueError("MPC minimum fill does not match delivery steps")

    def body(self) -> dict[str, Any]:
        return {
            "schema_version": SERVICE_MPC_SCHEMA_VERSION,
            "controller_id": SERVICE_MPC_CONTROLLER_ID,
            "model_version": SERVICE_MPC_MODEL_VERSION,
            "objective_version": SERVICE_MPC_OBJECTIVE_VERSION,
            "model_sha256": self.model_sha256,
            "forecast_sha256": self.forecast_sha256,
            "settings_sha256": self.settings_sha256,
            "horizon_intervals": int(self.horizon_intervals),
            "deliveries": [item.to_dict() for item in self.deliveries],
            "stages": [stage.to_dict() for stage in self.stages],
            "steps": [step.to_dict() for step in self.steps],
            "primary_delivered_optimum_kg": float(self.primary_delivered_optimum_kg),
            "selected_delivered_kg": float(self.selected_delivered_kg),
            "delivery_lock_tolerance_kg": float(self.delivery_lock_tolerance_kg),
            "minimum_commitment_fill": float(self.minimum_commitment_fill),
            "terminal_ch4_equivalent_kg": float(self.terminal_ch4_equivalent_kg),
            "secondary_utility": float(self.secondary_utility),
        }

    @property
    def sha256(self) -> str:
        return domain_sha256(SERVICE_MPC_PLAN_DOMAIN, self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


class ServiceMPCError(RuntimeError):
    """Base class for explicit service-MPC controller failures."""


class ServiceMPCInputError(ServiceMPCError):
    """The current public boundary is invalid or outside the frozen model scope."""


class ServiceMPCSolveError(ServiceMPCError):
    """One solver stage failed; no fallback action may be emitted as MPC."""


class ServiceMPCPlanError(ServiceMPCError):
    """An apparently successful numerical solution failed plan validation."""


__all__ = [
    "DEFAULT_SERVICE_MPC_SETTINGS_V1",
    "MODEL_BALANCE_TOLERANCE",
    "MPCDeliveryTargetV1",
    "MPCPlanStepV1",
    "MPCPlanV1",
    "MPCStageSolveV1",
    "SERVICE_MPC_CONTROLLER_ID",
    "SERVICE_MPC_MODEL_VERSION",
    "SERVICE_MPC_OBJECTIVE_VERSION",
    "SERVICE_MPC_PLAN_DOMAIN",
    "SERVICE_MPC_SCHEMA_VERSION",
    "SERVICE_MPC_SETTINGS_DOMAIN",
    "SERVICE_MPC_SOLUTION_DOMAIN",
    "SERVICE_MPC_SOLVER",
    "ServiceMPCError",
    "ServiceMPCInputError",
    "ServiceMPCPlanError",
    "ServiceMPCSettingsV1",
    "ServiceMPCSolveError",
    "canonical_json",
    "domain_sha256",
    "require_sha256",
]

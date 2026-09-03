"""Optional SciPy/HiGHS adapter for the immutable service-MPC model.

The ordinary backend can import every service-MPC schema and model module without
installing SciPy.  Only :func:`solve_service_mpc_model` crosses the optional runtime
boundary.  It performs both preregistered stages, independently audits the returned
vectors against the solver-independent sparse model, and never substitutes a fallback
plan when optimization fails.
"""

from __future__ import annotations

import importlib
import math
import time
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, NoReturn, Sequence

from .mpc_model import MPCSparseRowV1, ServiceMPCModelV1, delivery_lock_row
from .mpc_schema import (
    MODEL_BALANCE_TOLERANCE,
    MPCPlanStepV1,
    MPCPlanV1,
    MPCStageSolveV1,
    SERVICE_MPC_SOLUTION_DOMAIN,
    ServiceMPCInputError,
    ServiceMPCPlanError,
    ServiceMPCSettingsV1,
    ServiceMPCSolveError,
    domain_sha256,
)

_INTERVAL_GROUP_NAMES = (
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
    "cold_start",
    "warmup",
    "running",
    "on",
    "stop",
    "battery_charging",
)
_BOUNDARY_GROUP_NAMES = (
    "battery_energy_kwh",
    "h2_inventory_kg",
    "co2_inventory_kg",
    "ch4_inventory_kg",
)
_SCALAR_GROUP_NAMES = (
    "minimum_commitment_fill",
    "terminal_feedstock_ch4_equivalent_kg",
)
_BINARY_GROUP_NAMES = frozenset(
    ("cold_start", "warmup", "running", "on", "stop", "battery_charging")
)


@dataclass(frozen=True, slots=True)
class _SciPyRuntime:
    numpy: Any
    milp: Any
    bounds: Any
    linear_constraint: Any
    csc_array: Any
    scipy_version: str
    highs_version: str


@dataclass(frozen=True, slots=True)
class _SolvedStage:
    solution: tuple[float, ...]
    evidence: MPCStageSolveV1


def _validate_solver_model(model: ServiceMPCModelV1) -> None:
    expected_names = (
        *_INTERVAL_GROUP_NAMES,
        *_BOUNDARY_GROUP_NAMES,
        *_SCALAR_GROUP_NAMES,
    )
    if tuple(group.name for group in model.groups) != expected_names:
        raise ServiceMPCInputError(
            "service-MPC model variable groups do not match the frozen formulation"
        )
    groups = {group.name: group.indices for group in model.groups}
    if any(
        len(groups[name]) != model.horizon_intervals for name in _INTERVAL_GROUP_NAMES
    ):
        raise ServiceMPCInputError("service-MPC interval groups have invalid lengths")
    if any(
        len(groups[name]) != model.horizon_intervals + 1
        for name in _BOUNDARY_GROUP_NAMES
    ):
        raise ServiceMPCInputError("service-MPC boundary groups have invalid lengths")
    if any(len(groups[name]) != 1 for name in _SCALAR_GROUP_NAMES):
        raise ServiceMPCInputError("service-MPC scalar groups have invalid lengths")
    for name, indices in groups.items():
        expected_integrality = int(name in _BINARY_GROUP_NAMES)
        if any(model.integrality[index] != expected_integrality for index in indices):
            raise ServiceMPCInputError(
                f"service-MPC group {name!r} has invalid integrality"
            )


def _load_scipy_runtime() -> _SciPyRuntime:
    """Load the optimization extra lazily and report its bundled solver version."""

    try:
        numpy = importlib.import_module("numpy")
        scipy = importlib.import_module("scipy")
        optimize = importlib.import_module("scipy.optimize")
        sparse = importlib.import_module("scipy.sparse")
    except ImportError as exc:
        raise ServiceMPCSolveError(
            "service MPC requires the optional optimisation dependencies; "
            "install the workspace optimisation group"
        ) from exc

    scipy_version = str(getattr(scipy, "__version__", "")).strip()
    if not scipy_version:
        raise ServiceMPCSolveError("SciPy did not publish a runtime version")
    try:
        highs_core = importlib.import_module("scipy.optimize._highspy._core")
        version_parts = tuple(
            getattr(highs_core, name)
            for name in (
                "HIGHS_VERSION_MAJOR",
                "HIGHS_VERSION_MINOR",
                "HIGHS_VERSION_PATCH",
            )
        )
        if any(
            isinstance(part, bool) or not isinstance(part, Integral)
            for part in version_parts
        ):
            raise TypeError("HiGHS version constants are not integers")
        highs_version = ".".join(str(int(part)) for part in version_parts)
    except (ImportError, AttributeError, TypeError):
        # SciPy does not expose a public HiGHS-version API. Older supported
        # wheels may omit the private constants, so preserve that fact rather
        # than inventing a version or rejecting an otherwise valid runtime.
        highs_version = f"unreported-by-scipy-{scipy_version}"

    return _SciPyRuntime(
        numpy=numpy,
        milp=getattr(optimize, "milp"),
        bounds=getattr(optimize, "Bounds"),
        linear_constraint=getattr(optimize, "LinearConstraint"),
        csc_array=getattr(sparse, "csc_array"),
        scipy_version=scipy_version,
        highs_version=highs_version,
    )


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ServiceMPCPlanError(f"{name} is not a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ServiceMPCPlanError(f"{name} is not finite")
    return number


def _optional_finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _optional_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    count = int(value)
    return count if count >= 0 else None


def _bounded_message(value: object, *, fallback: str) -> str:
    message = " ".join(str(value).split())
    if not message:
        message = fallback
    return message[:500]


def _solver_constraint(
    runtime: _SciPyRuntime,
    model: ServiceMPCModelV1,
    extra_rows: Sequence[MPCSparseRowV1],
) -> Any:
    rows = (*model.rows, *tuple(extra_rows))
    row_indices: list[int] = []
    column_indices: list[int] = []
    coefficients: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, MPCSparseRowV1):
            raise TypeError("solver extra rows must be MPCSparseRowV1 records")
        lower.append(-math.inf if row.lower is None else row.lower)
        upper.append(math.inf if row.upper is None else row.upper)
        for column_index, coefficient in row.coefficients:
            row_indices.append(row_index)
            column_indices.append(column_index)
            coefficients.append(coefficient)
    matrix = runtime.csc_array(
        (coefficients, (row_indices, column_indices)),
        shape=(len(rows), model.variable_count),
        dtype=float,
    )
    return runtime.linear_constraint(
        matrix,
        runtime.numpy.asarray(lower, dtype=float),
        runtime.numpy.asarray(upper, dtype=float),
    )


def _canonical_solution(
    model: ServiceMPCModelV1,
    raw_solution: object,
) -> tuple[float, ...]:
    try:
        raw_values: tuple[object, ...] = tuple(raw_solution)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ServiceMPCPlanError("solver solution is not a finite vector") from exc
    if len(raw_values) != model.variable_count:
        raise ServiceMPCPlanError("solver solution length does not match the model")

    values: list[float] = []
    for index, (raw, lower, upper, integer) in enumerate(
        zip(
            raw_values,
            model.variable_lower,
            model.variable_upper,
            model.integrality,
            strict=True,
        )
    ):
        value = _finite(f"solver solution[{index}]", raw)
        if abs(value - lower) <= MODEL_BALANCE_TOLERANCE:
            value = lower
        elif abs(value - upper) <= MODEL_BALANCE_TOLERANCE:
            value = upper
        if integer and abs(value - round(value)) <= MODEL_BALANCE_TOLERANCE:
            value = float(round(value))
        if value == 0.0:
            value = 0.0  # Canonicalize negative zero in solution identities.
        values.append(value)
    return tuple(values)


def _solution_sha256(
    model: ServiceMPCModelV1,
    *,
    stage: str,
    solution: Sequence[float],
    extra_rows: Sequence[MPCSparseRowV1],
) -> str:
    return domain_sha256(
        SERVICE_MPC_SOLUTION_DOMAIN,
        {
            "model_sha256": model.sha256,
            "stage": stage,
            "extra_rows": [row.to_dict() for row in extra_rows],
            "values": [float(value) for value in solution],
        },
    )


def _failed_evidence(
    runtime: _SciPyRuntime,
    *,
    stage: str,
    status: int,
    message: str,
    elapsed_ns: int,
    result: object | None,
) -> MPCStageSolveV1:
    return MPCStageSolveV1(
        stage=stage,
        success=False,
        status=status,
        message=message,
        elapsed_ns=elapsed_ns,
        objective_value=(
            None if result is None else _optional_finite(getattr(result, "fun", None))
        ),
        mip_dual_bound=(
            None
            if result is None
            else _optional_finite(getattr(result, "mip_dual_bound", None))
        ),
        mip_gap=(
            None
            if result is None
            else _optional_finite(getattr(result, "mip_gap", None))
        ),
        mip_node_count=(
            None
            if result is None
            else _optional_count(getattr(result, "mip_node_count", None))
        ),
        solution_sha256=None,
        scipy_version=runtime.scipy_version,
        highs_version=runtime.highs_version,
    )


def _raise_stage_validation_failure(
    runtime: _SciPyRuntime,
    *,
    stage: str,
    status: int,
    solver_message: str,
    elapsed_ns: int,
    result: object | None,
    exception: Exception,
    prior_stages: tuple[MPCStageSolveV1, ...],
) -> NoReturn:
    validation_message = _bounded_message(
        f"{solver_message}; independent adapter validation rejected the result: "
        f"{type(exception).__name__}: {exception}",
        fallback="independent adapter validation rejected the solver result",
    )
    evidence = _failed_evidence(
        runtime,
        stage=stage,
        status=status,
        message=validation_message,
        elapsed_ns=elapsed_ns,
        result=result,
    )
    raise ServiceMPCSolveError(
        f"service-MPC {stage} stage validation failed: {validation_message}",
        stages=(*prior_stages, evidence),
    ) from exception


def _solve_stage(
    runtime: _SciPyRuntime,
    model: ServiceMPCModelV1,
    settings: ServiceMPCSettingsV1,
    *,
    stage: str,
    objective: Sequence[float],
    extra_rows: Sequence[MPCSparseRowV1] = (),
    prior_stages: tuple[MPCStageSolveV1, ...] = (),
) -> _SolvedStage:
    started_ns = time.perf_counter_ns()
    result: object | None = None
    try:
        constraint = _solver_constraint(runtime, model, extra_rows)
        result = runtime.milp(
            c=runtime.numpy.asarray(tuple(objective), dtype=float),
            integrality=runtime.numpy.asarray(model.integrality, dtype=int),
            bounds=runtime.bounds(
                runtime.numpy.asarray(model.variable_lower, dtype=float),
                runtime.numpy.asarray(model.variable_upper, dtype=float),
            ),
            constraints=constraint,
            options={
                "disp": False,
                "presolve": settings.solver_presolve,
                "mip_rel_gap": settings.solver_mip_rel_gap,
                "time_limit": settings.solver_time_limit_s,
            },
        )
    except Exception as exc:
        elapsed_ns = time.perf_counter_ns() - started_ns
        message = _bounded_message(
            f"{type(exc).__name__}: {exc}",
            fallback="SciPy milp raised without a message",
        )
        evidence = _failed_evidence(
            runtime,
            stage=stage,
            status=-1,
            message=message,
            elapsed_ns=elapsed_ns,
            result=result,
        )
        raise ServiceMPCSolveError(
            f"service-MPC {stage} stage raised: {message}",
            stages=(*prior_stages, evidence),
        ) from exc

    elapsed_ns = time.perf_counter_ns() - started_ns
    message = _bounded_message(
        getattr(result, "message", ""),
        fallback="SciPy milp returned no status message",
    )
    raw_status = getattr(result, "status", None)
    if isinstance(raw_status, bool) or not isinstance(raw_status, Integral):
        exception = ServiceMPCPlanError(f"service-MPC {stage} status is not an integer")
        _raise_stage_validation_failure(
            runtime,
            stage=stage,
            status=-2,
            solver_message=message,
            elapsed_ns=elapsed_ns,
            result=result,
            exception=exception,
            prior_stages=prior_stages,
        )
    status = int(raw_status)
    if status != 0:
        evidence = _failed_evidence(
            runtime,
            stage=stage,
            status=status,
            message=message,
            elapsed_ns=elapsed_ns,
            result=result,
        )
        raise ServiceMPCSolveError(
            f"service-MPC {stage} stage failed with status {status}: {message}",
            stages=(*prior_stages, evidence),
        )
    try:
        if getattr(result, "success", None) is not True:
            raise ServiceMPCPlanError(
                f"service-MPC {stage} returned status zero without success"
            )
        raw_solution = getattr(result, "x", None)
        if raw_solution is None:
            raise ServiceMPCPlanError(
                f"service-MPC {stage} returned status zero without a solution"
            )
        solution = _canonical_solution(model, raw_solution)
        audit = model.audit_solution(solution, extra_rows=extra_rows)
        if not audit.valid:
            raise ServiceMPCPlanError(
                f"service-MPC {stage} solution failed independent residual audit: "
                f"bounds={audit.max_bound_violation:.6g}, "
                f"rows={audit.max_row_violation:.6g}, "
                f"integrality={audit.max_integrality_violation:.6g}, "
                f"worst_row={audit.worst_row!r}"
            )

        if len(objective) != model.variable_count:
            raise ServiceMPCPlanError(
                "solver objective length does not match the model"
            )
        objective_value = math.fsum(
            float(coefficient) * value
            for coefficient, value in zip(objective, solution, strict=True)
        )
        reported_objective = _finite(
            f"service-MPC {stage} reported objective",
            getattr(result, "fun", None),
        )
        objective_tolerance = max(
            MODEL_BALANCE_TOLERANCE,
            1e-9 * max(1.0, abs(objective_value)),
        )
        if not math.isclose(
            reported_objective,
            objective_value,
            rel_tol=0.0,
            abs_tol=objective_tolerance,
        ):
            raise ServiceMPCPlanError(
                f"service-MPC {stage} objective disagrees with its solution"
            )
        dual_bound = _finite(
            f"service-MPC {stage} MIP dual bound",
            getattr(result, "mip_dual_bound", None),
        )
        mip_gap = _finite(
            f"service-MPC {stage} MIP gap",
            getattr(result, "mip_gap", None),
        )
        if mip_gap < 0.0:
            raise ServiceMPCPlanError(f"service-MPC {stage} MIP gap is negative")
        node_count = _optional_count(getattr(result, "mip_node_count", None))
        if node_count is None:
            raise ServiceMPCPlanError(f"service-MPC {stage} node count is invalid")

        evidence = MPCStageSolveV1(
            stage=stage,
            success=True,
            status=status,
            message=message,
            elapsed_ns=elapsed_ns,
            objective_value=objective_value,
            mip_dual_bound=dual_bound,
            mip_gap=mip_gap,
            mip_node_count=node_count,
            solution_sha256=_solution_sha256(
                model,
                stage=stage,
                solution=solution,
                extra_rows=extra_rows,
            ),
            scipy_version=runtime.scipy_version,
            highs_version=runtime.highs_version,
        )
        return _SolvedStage(solution=solution, evidence=evidence)
    except (ServiceMPCPlanError, TypeError, ValueError) as exc:
        _raise_stage_validation_failure(
            runtime,
            stage=stage,
            status=status,
            solver_message=message,
            elapsed_ns=elapsed_ns,
            result=result,
            exception=exc,
            prior_stages=prior_stages,
        )


def _boolean_solution(name: str, value: float) -> bool:
    rounded = round(value)
    if rounded not in (0, 1) or abs(value - rounded) > MODEL_BALANCE_TOLERANCE:
        raise ServiceMPCPlanError(f"selected {name} is not binary")
    return bool(rounded)


def _extract_plan(
    model: ServiceMPCModelV1,
    settings: ServiceMPCSettingsV1,
    *,
    primary: _SolvedStage,
    secondary: _SolvedStage,
) -> MPCPlanV1:
    solution = secondary.solution

    def group(name: str) -> tuple[int, ...]:
        try:
            return model.group(name)
        except KeyError as exc:
            raise ServiceMPCPlanError(
                f"service-MPC model is missing variable group {name!r}"
            ) from exc

    solar_battery = group("solar_to_battery_kw")
    solar_electrolyser = group("solar_to_electrolyser_kw")
    solar_dac = group("solar_to_dac_kw")
    solar_methanator = group("solar_to_methanator_kw")
    solar_curtailed = group("solar_curtailed_kw")
    battery_support = group("battery_support_kw")
    startup_power = group("electrolyser_startup_kw")
    productive_power = group("electrolyser_productive_kw")
    h2_produced = group("h2_produced_kg")
    co2_captured = group("co2_captured_kg")
    ch4_produced = group("ch4_produced_kg")
    delivered = group("delivered_kg")
    cold_start = group("cold_start")
    warmup = group("warmup")
    running = group("running")
    battery_charging = group("battery_charging")
    battery_energy = group("battery_energy_kwh")
    h2_inventory = group("h2_inventory_kg")
    co2_inventory = group("co2_inventory_kg")
    ch4_inventory = group("ch4_inventory_kg")
    minimum_fill = group("minimum_commitment_fill")
    terminal_feedstock = group("terminal_feedstock_ch4_equivalent_kg")

    interval_groups = (
        solar_battery,
        solar_electrolyser,
        solar_dac,
        solar_methanator,
        solar_curtailed,
        battery_support,
        startup_power,
        productive_power,
        h2_produced,
        co2_captured,
        ch4_produced,
        delivered,
        cold_start,
        warmup,
        running,
        battery_charging,
    )
    if any(len(indices) != model.horizon_intervals for indices in interval_groups):
        raise ServiceMPCPlanError("service-MPC interval groups have invalid lengths")
    if any(
        len(indices) != model.horizon_intervals + 1
        for indices in (battery_energy, h2_inventory, co2_inventory, ch4_inventory)
    ):
        raise ServiceMPCPlanError("service-MPC boundary groups have invalid lengths")
    if len(minimum_fill) != 1 or len(terminal_feedstock) != 1:
        raise ServiceMPCPlanError("service-MPC scalar groups have invalid lengths")

    constants = model.metadata().get("constants")
    if not isinstance(constants, dict):
        raise ServiceMPCPlanError("service-MPC metadata has no constants object")
    forecast_values = constants.get("forecast_available_kw")
    if (
        not isinstance(forecast_values, list)
        or len(forecast_values) != model.horizon_intervals
    ):
        raise ServiceMPCPlanError("service-MPC metadata forecast has invalid length")
    forecast = tuple(
        _finite(f"forecast_available_kw[{index}]", value)
        for index, value in enumerate(forecast_values)
    )

    steps = tuple(
        MPCPlanStepV1(
            index=index,
            forecast_solar_kw=forecast[index],
            solar_to_battery_kw=solution[solar_battery[index]],
            solar_to_electrolyser_kw=solution[solar_electrolyser[index]],
            solar_to_dac_kw=solution[solar_dac[index]],
            solar_to_methanator_kw=solution[solar_methanator[index]],
            solar_curtailed_kw=solution[solar_curtailed[index]],
            battery_support_kw=solution[battery_support[index]],
            electrolyser_startup_kw=solution[startup_power[index]],
            electrolyser_productive_kw=solution[productive_power[index]],
            h2_produced_kg=solution[h2_produced[index]],
            co2_captured_kg=solution[co2_captured[index]],
            ch4_produced_kg=solution[ch4_produced[index]],
            delivered_kg=solution[delivered[index]],
            battery_energy_end_kwh=solution[battery_energy[index + 1]],
            h2_end_kg=solution[h2_inventory[index + 1]],
            co2_end_kg=solution[co2_inventory[index + 1]],
            ch4_end_kg=solution[ch4_inventory[index + 1]],
            cold_start=_boolean_solution(
                f"cold_start[{index}]", solution[cold_start[index]]
            ),
            warmup=_boolean_solution(f"warmup[{index}]", solution[warmup[index]]),
            running=_boolean_solution(f"running[{index}]", solution[running[index]]),
            battery_charging=_boolean_solution(
                f"battery_charging[{index}]",
                solution[battery_charging[index]],
            ),
        )
        for index in range(model.horizon_intervals)
    )
    primary_delivered = math.fsum(primary.solution[index] for index in delivered)
    selected_delivered = math.fsum(solution[index] for index in delivered)
    secondary_objective = math.fsum(
        coefficient * value
        for coefficient, value in zip(
            model.secondary_objective,
            solution,
            strict=True,
        )
    )
    return MPCPlanV1(
        model_sha256=model.sha256,
        forecast_sha256=model.forecast_sha256,
        settings_sha256=model.settings_sha256,
        horizon_intervals=model.horizon_intervals,
        deliveries=model.deliveries,
        stages=(primary.evidence, secondary.evidence),
        steps=steps,
        primary_delivered_optimum_kg=primary_delivered,
        selected_delivered_kg=selected_delivered,
        delivery_lock_tolerance_kg=settings.delivery_lock_tolerance_kg,
        minimum_commitment_fill=solution[minimum_fill[0]],
        terminal_ch4_equivalent_kg=(
            solution[ch4_inventory[-1]] + solution[terminal_feedstock[0]]
        ),
        secondary_utility=-secondary_objective,
    )


def solve_service_mpc_model(
    model: ServiceMPCModelV1,
    *,
    settings: ServiceMPCSettingsV1,
) -> MPCPlanV1:
    """Run the frozen two-stage solve and return only an independently valid plan."""

    if not isinstance(model, ServiceMPCModelV1):
        raise TypeError("model must be ServiceMPCModelV1")
    if not isinstance(settings, ServiceMPCSettingsV1):
        raise TypeError("settings must be ServiceMPCSettingsV1")
    if model.settings_sha256 != settings.sha256:
        raise ServiceMPCInputError("service-MPC model/settings identity mismatch")
    _validate_solver_model(model)
    runtime = _load_scipy_runtime()
    primary = _solve_stage(
        runtime,
        model,
        settings,
        stage="primary_delivery",
        objective=model.primary_objective,
    )
    primary_delivered = math.fsum(
        primary.solution[index] for index in model.group("delivered_kg")
    )
    try:
        lock = delivery_lock_row(
            model,
            primary_delivered_kg=primary_delivered,
            tolerance_kg=settings.delivery_lock_tolerance_kg,
        )
    except (TypeError, ValueError, ServiceMPCPlanError) as exc:
        raise ServiceMPCPlanError(
            f"service-MPC delivery lock construction failed: {exc}",
            stages=(primary.evidence,),
        ) from exc
    secondary = _solve_stage(
        runtime,
        model,
        settings,
        stage="secondary_tiebreak",
        objective=model.secondary_objective,
        extra_rows=(lock,),
        prior_stages=(primary.evidence,),
    )
    try:
        return _extract_plan(
            model,
            settings,
            primary=primary,
            secondary=secondary,
        )
    except (ServiceMPCPlanError, KeyError, TypeError, ValueError) as exc:
        raise ServiceMPCPlanError(
            f"service-MPC selected plan failed schema extraction: {exc}",
            stages=(primary.evidence, secondary.evidence),
        ) from exc


__all__ = ["solve_service_mpc_model"]

"""Fail-closed SciPy/HiGHS adapter for economic MPC V1."""

from __future__ import annotations

import importlib
import math
import time
import warnings
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Sequence

from .economic_mpc_model import (
    EconomicMPCModelProtocol,
    EconomicMPCModelV1,
    economic_delivery_lock_row,
)
from .economic_mpc_schema import (
    ECONOMIC_MPC_SOLUTION_DOMAIN,
    EconomicMPCInputError,
    EconomicMPCPlanError,
    EconomicMPCPlanV1,
    EconomicMPCSettingsV1,
    EconomicMPCSolveError,
    EconomicMPCStageSolveV1,
)
from .mpc_model import MPCSparseRowV1
from .mpc_schema import (
    MODEL_BALANCE_TOLERANCE,
    MPCDeliveryTargetV1,
    MPCPlanStepV1,
    domain_sha256,
    require_sha256,
)


@dataclass(frozen=True, slots=True)
class _Runtime:
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
    evidence: EconomicMPCStageSolveV1


@dataclass(frozen=True, slots=True)
class EconomicMPCSolveResultV1:
    """Controller-neutral numerical result before a task-specific plan schema.

    The vectors stay in memory and are never treated as standalone scientific
    evidence.  Their stage records and the task-specific model digest provide the
    auditable bridge used by alpha and beta plan builders.
    """

    model_sha256: str
    solution_domain: str
    primary_solution: tuple[float, ...]
    economic_solution: tuple[float, ...]
    stages: tuple[EconomicMPCStageSolveV1, EconomicMPCStageSolveV1]

    def __post_init__(self) -> None:
        require_sha256("model_sha256", self.model_sha256)
        if (
            not isinstance(self.solution_domain, str)
            or not self.solution_domain.strip()
        ):
            raise ValueError("solution_domain must be non-empty")
        if not self.primary_solution or not self.economic_solution:
            raise ValueError("economic solve vectors must be non-empty")
        if len(self.primary_solution) != len(self.economic_solution):
            raise ValueError("economic solve vectors must have equal length")
        if tuple(stage.stage for stage in self.stages) != (
            "service_maximization",
            "economic_cost_minimization",
        ) or not all(stage.success for stage in self.stages):
            raise ValueError("economic solve result requires two successful stages")
        for name, vector in (
            ("primary_solution", self.primary_solution),
            ("economic_solution", self.economic_solution),
        ):
            if not isinstance(vector, tuple):
                raise TypeError(f"{name} must be an immutable tuple")
            for index, value in enumerate(vector):
                _finite(f"{name}[{index}]", value)


@dataclass(frozen=True, slots=True)
class _EconomicMPCPlanValuesV1:
    """Identity-neutral validated values shared by alpha and beta plan schemas."""

    model_sha256: str
    forecast_sha256: str
    settings_sha256: str
    assumptions_sha256: str
    horizon_intervals: int
    deliveries: tuple[MPCDeliveryTargetV1, ...]
    stages: tuple[EconomicMPCStageSolveV1, EconomicMPCStageSolveV1]
    steps: tuple[MPCPlanStepV1, ...]
    primary_delivered_optimum_kg: float
    selected_delivered_kg: float
    delivery_lock_tolerance_kg: float
    predicted_cost_components_eur: tuple[tuple[str, float], ...]
    terminal_ch4_equivalent_deficit_kg: float
    terminal_battery_deficit_kwh: float


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise EconomicMPCPlanError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise EconomicMPCPlanError(f"{name} must be finite")
    return result


def _load_runtime() -> _Runtime:
    try:
        numpy = importlib.import_module("numpy")
        scipy = importlib.import_module("scipy")
        optimize = importlib.import_module("scipy.optimize")
        sparse = importlib.import_module("scipy.sparse")
    except ImportError as exc:
        raise EconomicMPCSolveError(
            "economic MPC requires the optional optimization dependencies"
        ) from exc
    scipy_version = str(getattr(scipy, "__version__", "")).strip()
    if not scipy_version:
        raise EconomicMPCSolveError("SciPy did not publish a runtime version")
    try:
        core = importlib.import_module("scipy.optimize._highspy._core")
        parts = tuple(
            getattr(core, name)
            for name in (
                "HIGHS_VERSION_MAJOR",
                "HIGHS_VERSION_MINOR",
                "HIGHS_VERSION_PATCH",
            )
        )
        if any(
            isinstance(part, bool) or not isinstance(part, Integral) for part in parts
        ):
            raise TypeError("HiGHS version constants are invalid")
        highs_version = ".".join(str(int(part)) for part in parts)
    except (ImportError, AttributeError, TypeError):
        highs_version = f"unreported-by-scipy-{scipy_version}"
    return _Runtime(
        numpy=numpy,
        milp=getattr(optimize, "milp"),
        bounds=getattr(optimize, "Bounds"),
        linear_constraint=getattr(optimize, "LinearConstraint"),
        csc_array=getattr(sparse, "csc_array"),
        scipy_version=scipy_version,
        highs_version=highs_version,
    )


def _constraint(
    runtime: _Runtime,
    model: EconomicMPCModelProtocol,
    extra_rows: Sequence[MPCSparseRowV1],
) -> Any:
    rows = (*model.rows, *tuple(extra_rows))
    row_indices: list[int] = []
    column_indices: list[int] = []
    coefficients: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for row_index, row in enumerate(rows):
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
    model: EconomicMPCModelProtocol,
    raw: object,
    *,
    tolerance: float,
    extra_rows: Sequence[MPCSparseRowV1],
) -> tuple[float, ...]:
    try:
        raw_values: tuple[object, ...] = tuple(raw)  # type: ignore[arg-type]
    except TypeError as exc:
        raise EconomicMPCPlanError("solver solution is not a vector") from exc
    if len(raw_values) != model.variable_count:
        raise EconomicMPCPlanError("solver solution length differs from the model")
    values = tuple(
        _finite(f"solution[{index}]", value) for index, value in enumerate(raw_values)
    )
    raw_audit = model.audit_solution(
        values,
        tolerance=tolerance,
        extra_rows=extra_rows,
    )
    if not raw_audit.valid:
        raise EconomicMPCPlanError(
            "raw solver solution failed independent residual audit: "
            f"bounds={raw_audit.max_bound_violation:.6g}, "
            f"rows={raw_audit.max_row_violation:.6g}, "
            f"integrality={raw_audit.max_integrality_violation:.6g}, "
            f"worst_row={raw_audit.worst_row!r}"
        )
    projected: list[float] = []
    for value, lower, upper, integer in zip(
        values,
        model.variable_lower,
        model.variable_upper,
        model.integrality,
        strict=True,
    ):
        # Correct only an actual bound violation. A small but feasible interior
        # flow is physical signal, not solver noise, and snapping it can disturb
        # a coupled material or energy balance.
        if math.isfinite(lower) and value < lower and lower - value <= tolerance:
            value = lower
        elif math.isfinite(upper) and value > upper and value - upper <= tolerance:
            value = upper
        if integer and abs(value - round(value)) <= tolerance:
            value = float(round(value))
        projected.append(0.0 if value == 0.0 else value)
    candidate = tuple(projected)
    projected_audit = model.audit_solution(
        candidate,
        # A row can accumulate one bounded projection delta per coefficient.
        # The raw vector has already passed at ``tolerance``; two tolerances
        # therefore cover the raw residual plus the complete projection budget.
        tolerance=2.0 * tolerance,
        extra_rows=extra_rows,
    )
    if not projected_audit.valid:
        raise EconomicMPCPlanError(
            "canonical solver solution failed independent residual audit: "
            f"bounds={projected_audit.max_bound_violation:.6g}, "
            f"rows={projected_audit.max_row_violation:.6g}, "
            f"integrality={projected_audit.max_integrality_violation:.6g}, "
            f"worst_row={projected_audit.worst_row!r}"
        )
    return candidate


def _failed_stage(
    runtime: _Runtime,
    *,
    stage: str,
    status: int,
    message: str,
    elapsed_ns: int,
    result: object | None,
) -> EconomicMPCStageSolveV1:
    def optional_real(name: str) -> float | None:
        value = None if result is None else getattr(result, name, None)
        if isinstance(value, bool) or not isinstance(value, Real):
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    raw_nodes = None if result is None else getattr(result, "mip_node_count", None)
    nodes = (
        int(raw_nodes)
        if isinstance(raw_nodes, Integral)
        and not isinstance(raw_nodes, bool)
        and int(raw_nodes) >= 0
        else None
    )
    return EconomicMPCStageSolveV1(
        stage=stage,
        success=False,
        status=status,
        message=(" ".join(message.split()) or "solver returned no message")[:500],
        elapsed_ns=elapsed_ns,
        objective_value=optional_real("fun"),
        mip_gap=optional_real("mip_gap"),
        mip_node_count=nodes,
        solution_sha256=None,
        scipy_version=runtime.scipy_version,
        highs_version=runtime.highs_version,
    )


def _solve_stage(
    runtime: _Runtime,
    model: EconomicMPCModelProtocol,
    settings: EconomicMPCSettingsV1,
    *,
    stage: str,
    objective: Sequence[float],
    solution_domain: str,
    extra_rows: Sequence[MPCSparseRowV1] = (),
    prior: tuple[EconomicMPCStageSolveV1, ...] = (),
) -> _SolvedStage:
    started = time.perf_counter_ns()
    result: object | None = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Unrecognized options detected:.*mip_feasibility_tolerance.*",
                category=RuntimeWarning,
            )
            result = runtime.milp(
                c=runtime.numpy.asarray(tuple(objective), dtype=float),
                integrality=runtime.numpy.asarray(model.integrality, dtype=int),
                bounds=runtime.bounds(
                    runtime.numpy.asarray(model.variable_lower, dtype=float),
                    runtime.numpy.asarray(model.variable_upper, dtype=float),
                ),
                constraints=_constraint(runtime, model, extra_rows),
                options={
                    "disp": False,
                    "presolve": settings.solver_presolve,
                    "mip_rel_gap": settings.solver_mip_rel_gap,
                    "mip_feasibility_tolerance": (
                        settings.solver_mip_feasibility_tolerance
                    ),
                    "time_limit": settings.solver_time_limit_s,
                },
            )
    except Exception as exc:
        elapsed = time.perf_counter_ns() - started
        failed = _failed_stage(
            runtime,
            stage=stage,
            status=-1,
            message=f"{type(exc).__name__}: {exc}",
            elapsed_ns=elapsed,
            result=result,
        )
        raise EconomicMPCSolveError(
            f"economic-MPC {stage} stage raised: {failed.message}",
            stages=(*prior, failed),
        ) from exc
    elapsed = time.perf_counter_ns() - started
    raw_status = getattr(result, "status", None)
    status = (
        int(raw_status)
        if isinstance(raw_status, Integral) and not isinstance(raw_status, bool)
        else -2
    )
    message = " ".join(str(getattr(result, "message", "")).split())
    if status != 0 or getattr(result, "success", None) is not True:
        failed = _failed_stage(
            runtime,
            stage=stage,
            status=status,
            message=message,
            elapsed_ns=elapsed,
            result=result,
        )
        raise EconomicMPCSolveError(
            f"economic-MPC {stage} failed with status {status}: {failed.message}",
            stages=(*prior, failed),
        )
    try:
        raw_solution = tuple(
            _finite(f"raw solution[{index}]", value)
            for index, value in enumerate(getattr(result, "x", ()))
        )
        if len(raw_solution) != model.variable_count:
            raise EconomicMPCPlanError(
                "solver solution length differs from the economic model"
            )
        solution = _canonical_solution(
            model,
            raw_solution,
            tolerance=settings.solver_mip_feasibility_tolerance,
            extra_rows=extra_rows,
        )
        value = math.fsum(
            float(coefficient) * selected
            for coefficient, selected in zip(objective, solution, strict=True)
        )
        reported = _finite("reported solver objective", getattr(result, "fun", None))
        raw_value = math.fsum(
            float(coefficient) * selected
            for coefficient, selected in zip(objective, raw_solution, strict=True)
        )
        objective_tolerance = max(
            MODEL_BALANCE_TOLERANCE,
            1e-9 * max(1.0, abs(raw_value), abs(reported)),
        )
        if not math.isclose(
            raw_value,
            reported,
            rel_tol=0.0,
            abs_tol=objective_tolerance,
        ):
            raise EconomicMPCPlanError(
                "reported solver objective disagrees with the raw solution"
            )
        gap = _finite("solver MIP gap", getattr(result, "mip_gap", None))
        if gap < 0.0:
            raise EconomicMPCPlanError("solver MIP gap is negative")
        raw_nodes = getattr(result, "mip_node_count", None)
        if (
            isinstance(raw_nodes, bool)
            or not isinstance(raw_nodes, Integral)
            or raw_nodes < 0
        ):
            raise EconomicMPCPlanError("solver MIP node count is invalid")
        solution_sha = domain_sha256(
            solution_domain,
            {
                "model_sha256": model.sha256,
                "stage": stage,
                "extra_rows": [row.to_dict() for row in extra_rows],
                "values": list(solution),
            },
        )
        evidence = EconomicMPCStageSolveV1(
            stage=stage,
            success=True,
            status=0,
            message=(message or "optimal")[:500],
            elapsed_ns=elapsed,
            objective_value=value,
            mip_gap=gap,
            mip_node_count=int(raw_nodes),
            solution_sha256=solution_sha,
            scipy_version=runtime.scipy_version,
            highs_version=runtime.highs_version,
        )
        return _SolvedStage(solution=solution, evidence=evidence)
    except (EconomicMPCPlanError, TypeError, ValueError) as exc:
        failed = _failed_stage(
            runtime,
            stage=stage,
            status=-3,
            message=f"independent validation rejected result: {exc}",
            elapsed_ns=elapsed,
            result=result,
        )
        raise EconomicMPCSolveError(
            f"economic-MPC {stage} validation failed: {failed.message}",
            stages=(*prior, failed),
        ) from exc


def _binary(name: str, value: float) -> bool:
    rounded = round(value)
    if rounded not in (0, 1) or abs(value - rounded) > MODEL_BALANCE_TOLERANCE:
        raise EconomicMPCPlanError(f"selected {name} is not binary")
    return bool(rounded)


def _extract_plan_values(
    model: EconomicMPCModelProtocol,
    settings: EconomicMPCSettingsV1,
    *,
    primary: _SolvedStage,
    economic: _SolvedStage,
) -> _EconomicMPCPlanValuesV1:
    values = economic.solution
    group = model.group
    solar_battery = group("solar_to_battery_kw")
    solar_electrolyser = group("solar_to_electrolyser_kw")
    solar_dac = group("solar_to_dac_kw")
    solar_methanator = group("solar_to_methanator_kw")
    solar_curtailed = group("solar_curtailed_kw")
    support = group("battery_support_kw")
    startup = group("electrolyser_startup_kw")
    productive = group("electrolyser_productive_kw")
    h2_produced = group("h2_produced_kg")
    co2_produced = group("co2_captured_kg")
    ch4_produced = group("ch4_produced_kg")
    delivered = group("delivered_kg")
    cold_start = group("cold_start")
    warmup = group("warmup")
    running = group("running")
    charging = group("battery_charging")
    battery = group("battery_energy_kwh")
    h2 = group("h2_inventory_kg")
    co2 = group("co2_inventory_kg")
    ch4 = group("ch4_inventory_kg")
    forecast_raw = model.physical_model.metadata()["constants"]["forecast_available_kw"]
    if (
        not isinstance(forecast_raw, list)
        or len(forecast_raw) != model.horizon_intervals
    ):
        raise EconomicMPCPlanError("physical forecast metadata is malformed")
    forecast = tuple(_finite("forecast value", item) for item in forecast_raw)
    steps = tuple(
        MPCPlanStepV1(
            index=index,
            forecast_solar_kw=forecast[index],
            solar_to_battery_kw=values[solar_battery[index]],
            solar_to_electrolyser_kw=values[solar_electrolyser[index]],
            solar_to_dac_kw=values[solar_dac[index]],
            solar_to_methanator_kw=values[solar_methanator[index]],
            solar_curtailed_kw=values[solar_curtailed[index]],
            battery_support_kw=values[support[index]],
            electrolyser_startup_kw=values[startup[index]],
            electrolyser_productive_kw=values[productive[index]],
            h2_produced_kg=values[h2_produced[index]],
            co2_captured_kg=values[co2_produced[index]],
            ch4_produced_kg=values[ch4_produced[index]],
            delivered_kg=values[delivered[index]],
            battery_energy_end_kwh=values[battery[index + 1]],
            h2_end_kg=values[h2[index + 1]],
            co2_end_kg=values[co2[index + 1]],
            ch4_end_kg=values[ch4[index + 1]],
            cold_start=_binary("cold_start", values[cold_start[index]]),
            warmup=_binary("warmup", values[warmup[index]]),
            running=_binary("running", values[running[index]]),
            battery_charging=_binary("battery_charging", values[charging[index]]),
        )
        for index in range(model.horizon_intervals)
    )
    primary_delivery = math.fsum(primary.solution[index] for index in delivered)
    selected_delivery = math.fsum(values[index] for index in delivered)
    components = model.predicted_cost_components(values)
    predicted_cost = math.fsum(value for _, value in components)
    objective = math.fsum(
        coefficient * value
        for coefficient, value in zip(model.economic_objective, values, strict=True)
    )
    if not math.isclose(
        predicted_cost,
        objective,
        rel_tol=0.0,
        abs_tol=max(1e-7, 1e-9 * max(1.0, abs(objective))),
    ):
        raise EconomicMPCPlanError(
            "economic cost ledger does not reconstruct the solver objective"
        )
    return _EconomicMPCPlanValuesV1(
        model_sha256=model.sha256,
        forecast_sha256=model.forecast_sha256,
        settings_sha256=model.settings_sha256,
        assumptions_sha256=model.assumptions_sha256,
        horizon_intervals=model.horizon_intervals,
        deliveries=model.deliveries,
        stages=(primary.evidence, economic.evidence),
        steps=steps,
        primary_delivered_optimum_kg=primary_delivery,
        selected_delivered_kg=selected_delivery,
        delivery_lock_tolerance_kg=settings.delivery_lock_tolerance_kg,
        predicted_cost_components_eur=components,
        terminal_ch4_equivalent_deficit_kg=values[
            group("terminal_ch4_equivalent_deficit_kg")[0]
        ],
        terminal_battery_deficit_kwh=values[group("terminal_battery_deficit_kwh")[0]],
    )


def solve_economic_mpc_stages(
    model: EconomicMPCModelProtocol,
    *,
    settings: EconomicMPCSettingsV1,
    solution_domain: str = ECONOMIC_MPC_SOLUTION_DOMAIN,
) -> EconomicMPCSolveResultV1:
    """Solve any validated economic model satisfying the structural contract."""

    if not isinstance(model, EconomicMPCModelProtocol):
        raise TypeError("model must satisfy EconomicMPCModelProtocol")
    if not isinstance(settings, EconomicMPCSettingsV1):
        raise TypeError("settings must be EconomicMPCSettingsV1")
    if not isinstance(solution_domain, str) or not solution_domain.strip():
        raise ValueError("solution_domain must be non-empty")
    if model.settings_sha256 != settings.sha256:
        raise EconomicMPCInputError("economic model/settings identity mismatch")
    runtime = _load_runtime()
    primary = _solve_stage(
        runtime,
        model,
        settings,
        stage="service_maximization",
        objective=model.primary_objective,
        solution_domain=solution_domain,
    )
    primary_delivery = math.fsum(
        primary.solution[index] for index in model.group("delivered_kg")
    )
    lock = economic_delivery_lock_row(
        model,
        delivered_optimum_kg=primary_delivery,
        tolerance_kg=settings.delivery_lock_tolerance_kg,
    )
    economic = _solve_stage(
        runtime,
        model,
        settings,
        stage="economic_cost_minimization",
        objective=model.economic_objective,
        solution_domain=solution_domain,
        extra_rows=(lock,),
        prior=(primary.evidence,),
    )
    return EconomicMPCSolveResultV1(
        model_sha256=model.sha256,
        solution_domain=solution_domain,
        primary_solution=primary.solution,
        economic_solution=economic.solution,
        stages=(primary.evidence, economic.evidence),
    )


def _economic_mpc_plan_values_from_solve_result(
    model: EconomicMPCModelProtocol,
    *,
    settings: EconomicMPCSettingsV1,
    solved: EconomicMPCSolveResultV1,
    expected_solution_domain: str,
) -> _EconomicMPCPlanValuesV1:
    """Extract validated identity-neutral values for a task-specific plan."""

    if not isinstance(model, EconomicMPCModelProtocol):
        raise TypeError("model must satisfy EconomicMPCModelProtocol")
    if not isinstance(settings, EconomicMPCSettingsV1):
        raise TypeError("settings must be EconomicMPCSettingsV1")
    if not isinstance(solved, EconomicMPCSolveResultV1):
        raise TypeError("solved must be EconomicMPCSolveResultV1")
    if solved.solution_domain != expected_solution_domain:
        raise EconomicMPCPlanError("economic solve/plan solution domain mismatch")
    if model.settings_sha256 != settings.sha256:
        raise EconomicMPCInputError("economic model/settings identity mismatch")
    if solved.model_sha256 != model.sha256:
        raise EconomicMPCPlanError("economic solve/model identity mismatch")
    if len(solved.economic_solution) != model.variable_count:
        raise EconomicMPCPlanError("economic solve/model vector length mismatch")
    tolerance = settings.solver_mip_feasibility_tolerance
    primary_audit = model.audit_solution(
        solved.primary_solution,
        tolerance=tolerance,
    )
    if not primary_audit.valid:
        raise EconomicMPCPlanError(
            "retained primary solution failed independent model audit"
        )
    primary_delivery = math.fsum(
        solved.primary_solution[index] for index in model.group("delivered_kg")
    )
    lock = economic_delivery_lock_row(
        model,
        delivered_optimum_kg=primary_delivery,
        tolerance_kg=settings.delivery_lock_tolerance_kg,
    )
    economic_audit = model.audit_solution(
        solved.economic_solution,
        tolerance=2.0 * tolerance,
        extra_rows=(lock,),
    )
    if not economic_audit.valid:
        raise EconomicMPCPlanError(
            "retained economic solution failed independent model audit"
        )
    objectives = (
        math.fsum(
            coefficient * value
            for coefficient, value in zip(
                model.primary_objective,
                solved.primary_solution,
                strict=True,
            )
        ),
        math.fsum(
            coefficient * value
            for coefficient, value in zip(
                model.economic_objective,
                solved.economic_solution,
                strict=True,
            )
        ),
    )
    extra_rows = ((), (lock,))
    for vector, stage, expected_objective, stage_rows in zip(
        (solved.primary_solution, solved.economic_solution),
        solved.stages,
        objectives,
        extra_rows,
        strict=True,
    ):
        objective_tolerance = max(
            MODEL_BALANCE_TOLERANCE,
            1e-9 * max(1.0, abs(expected_objective)),
        )
        if stage.objective_value is None or not math.isclose(
            stage.objective_value,
            expected_objective,
            rel_tol=0.0,
            abs_tol=objective_tolerance,
        ):
            raise EconomicMPCPlanError(
                f"retained {stage.stage} objective evidence is inconsistent"
            )
        expected_solution_sha256 = domain_sha256(
            solved.solution_domain,
            {
                "model_sha256": model.sha256,
                "stage": stage.stage,
                "extra_rows": [row.to_dict() for row in stage_rows],
                "values": list(vector),
            },
        )
        if stage.solution_sha256 != expected_solution_sha256:
            raise EconomicMPCPlanError(
                f"retained {stage.stage} solution digest is inconsistent"
            )
    primary = _SolvedStage(
        solution=solved.primary_solution,
        evidence=solved.stages[0],
    )
    economic = _SolvedStage(
        solution=solved.economic_solution,
        evidence=solved.stages[1],
    )
    try:
        return _extract_plan_values(
            model,
            settings,
            primary=primary,
            economic=economic,
        )
    except (EconomicMPCPlanError, KeyError, TypeError, ValueError) as exc:
        raise EconomicMPCPlanError(
            f"economic-MPC selected plan failed schema extraction: {exc}"
        ) from exc


def economic_mpc_plan_from_solve_result(
    model: EconomicMPCModelV1,
    *,
    settings: EconomicMPCSettingsV1,
    solved: EconomicMPCSolveResultV1,
) -> EconomicMPCPlanV1:
    """Build an alpha plan from an exact alpha model and alpha settings."""

    if type(model) is not EconomicMPCModelV1:
        raise TypeError("model must be EconomicMPCModelV1")
    if type(settings) is not EconomicMPCSettingsV1:
        raise TypeError("settings must be EconomicMPCSettingsV1")
    values = _economic_mpc_plan_values_from_solve_result(
        model,
        settings=settings,
        solved=solved,
        expected_solution_domain=ECONOMIC_MPC_SOLUTION_DOMAIN,
    )
    return EconomicMPCPlanV1(
        model_sha256=values.model_sha256,
        forecast_sha256=values.forecast_sha256,
        settings_sha256=values.settings_sha256,
        assumptions_sha256=values.assumptions_sha256,
        horizon_intervals=values.horizon_intervals,
        deliveries=values.deliveries,
        stages=values.stages,
        steps=values.steps,
        primary_delivered_optimum_kg=values.primary_delivered_optimum_kg,
        selected_delivered_kg=values.selected_delivered_kg,
        delivery_lock_tolerance_kg=values.delivery_lock_tolerance_kg,
        predicted_cost_components_eur=values.predicted_cost_components_eur,
        terminal_ch4_equivalent_deficit_kg=(values.terminal_ch4_equivalent_deficit_kg),
        terminal_battery_deficit_kwh=values.terminal_battery_deficit_kwh,
    )


def solve_economic_mpc_model(
    model: EconomicMPCModelV1,
    *,
    settings: EconomicMPCSettingsV1,
) -> EconomicMPCPlanV1:
    """Maximize service, lock it, then build the alpha economic plan."""

    if type(model) is not EconomicMPCModelV1:
        raise TypeError("model must be EconomicMPCModelV1")
    if type(settings) is not EconomicMPCSettingsV1:
        raise TypeError("settings must be EconomicMPCSettingsV1")
    solved = solve_economic_mpc_stages(model, settings=settings)
    return economic_mpc_plan_from_solve_result(
        model,
        settings=settings,
        solved=solved,
    )


__all__ = [
    "EconomicMPCSolveResultV1",
    "economic_mpc_plan_from_solve_result",
    "solve_economic_mpc_model",
    "solve_economic_mpc_stages",
]

"""Versioned contracts for the service-constrained economic MPC controller.

The economic controller is deliberately a different algorithm from the historical
service MPC.  It shares the latter's physical formulation, but not its objective or
identity: service is optimized first and a monetary cost is optimized only inside
the service-optimal face.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, ClassVar

from .mpc_schema import (
    MODEL_BALANCE_TOLERANCE,
    MPCDeliveryTargetV1,
    MPCPlanStepV1,
    ServiceMPCSetpointSettingsV1,
    canonical_json,
    domain_sha256,
    require_sha256,
)
from .setpoint_command import ProcessSetpointCommandV1
from .setpoint_control import ProcessSetpointTargetsV1


ECONOMIC_MPC_SCHEMA_VERSION = 1
ECONOMIC_MPC_CONTROLLER_ID = "tariff-aware-economic-mpc-v1"
ECONOMIC_MPC_MODEL_VERSION = "service-constrained-economic-mpc-v1"
ECONOMIC_MPC_OBJECTIVE_VERSION = "lexicographic-service-then-eur-v1"
ECONOMIC_MPC_SETTINGS_DOMAIN = "economic-mpc-settings-v1"
ECONOMIC_MPC_PLAN_DOMAIN = "economic-mpc-plan-v1"
ECONOMIC_MPC_SOLUTION_DOMAIN = "economic-mpc-solution-vector-v1"
ECONOMIC_MPC_DECISION_SCHEMA = "economic-mpc-decision-evidence-v1"
ECONOMIC_MPC_DECISION_DOMAIN = ECONOMIC_MPC_DECISION_SCHEMA
ECONOMIC_MPC_SOLVER = "scipy.optimize.milp"
ECONOMIC_MPC_MODEL_MISMATCH_DECLARATION = (
    "Solar opportunity cost and bus-side battery throughput are exact linear "
    "counterparts of EconomicObjectiveV1 for the issued-mean plan. Electrolyser "
    "wear covers modeled running intervals and cold starts; minute-scale ramping, "
    "low-load classification, hot starts, aborted starts, forced stops, forecast "
    "error, guard intervention, and nonlinear plant response remain measured "
    "realization/model-mismatch terms rather than optimizer claims."
)


def _finite(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum:g}")
    return result


def _index(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class EconomicMPCSettingsV1:
    """Complete immutable tuning and solver surface for economic MPC V1."""

    task_horizon_h: float = 72.0
    planning_horizon_h: float = 24.0
    interval_s: float = 900.0
    terminal_battery_soc_target: float = 0.25
    delivery_lock_tolerance_kg: float = 1e-6
    energy_reconstruction_tolerance: float = 1e-12
    solver_presolve: bool = True
    solver_mip_rel_gap: float = 1e-6
    solver_mip_feasibility_tolerance: float = 1e-7
    solver_time_limit_s: float = 2.0

    schema_version: ClassVar[int] = ECONOMIC_MPC_SCHEMA_VERSION
    controller_id: ClassVar[str] = ECONOMIC_MPC_CONTROLLER_ID
    model_version: ClassVar[str] = ECONOMIC_MPC_MODEL_VERSION
    objective_version: ClassVar[str] = ECONOMIC_MPC_OBJECTIVE_VERSION
    solver: ClassVar[str] = ECONOMIC_MPC_SOLVER

    def __post_init__(self) -> None:
        # Reuse the mature physical-grid and solver-option validator.  Its
        # secondary weights are irrelevant because the economic model replaces
        # the complete second-stage vector.
        self.physical_settings

    @property
    def physical_settings(self) -> ServiceMPCSetpointSettingsV1:
        return ServiceMPCSetpointSettingsV1(
            task_horizon_h=self.task_horizon_h,
            planning_horizon_h=self.planning_horizon_h,
            interval_s=self.interval_s,
            terminal_battery_soc_target=self.terminal_battery_soc_target,
            delivery_lock_tolerance_kg=self.delivery_lock_tolerance_kg,
            energy_reconstruction_tolerance=self.energy_reconstruction_tolerance,
            solver_presolve=self.solver_presolve,
            solver_mip_rel_gap=self.solver_mip_rel_gap,
            solver_mip_feasibility_tolerance=self.solver_mip_feasibility_tolerance,
            solver_time_limit_s=self.solver_time_limit_s,
        )

    @property
    def task_interval_count(self) -> int:
        return self.physical_settings.task_interval_count

    @property
    def maximum_planning_intervals(self) -> int:
        return self.physical_settings.maximum_planning_intervals

    def body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "controller_id": self.controller_id,
            "model_version": self.model_version,
            "objective_version": self.objective_version,
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
                "mip_feasibility_tolerance": float(
                    self.solver_mip_feasibility_tolerance
                ),
                "time_limit": float(self.solver_time_limit_s),
            },
            "lexicographic_order": [
                "maximize delivered service",
                "minimize predicted EUR cost under delivery lock",
            ],
            "terminal_endowment_semantics": (
                "non-negative hinge deficits relative to episode-opening methane-"
                "equivalent and battery-energy endowments; surplus has zero credit"
            ),
            "model_mismatch_declaration": ECONOMIC_MPC_MODEL_MISMATCH_DECLARATION,
            "task_interval_count": self.task_interval_count,
            "maximum_planning_intervals": self.maximum_planning_intervals,
        }

    @property
    def sha256(self) -> str:
        return domain_sha256(ECONOMIC_MPC_SETTINGS_DOMAIN, self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


DEFAULT_ECONOMIC_MPC_SETTINGS_V1 = EconomicMPCSettingsV1()


@dataclass(frozen=True, slots=True)
class EconomicMPCStageSolveV1:
    """Fail-closed numerical evidence for one lexicographic solve stage."""

    stage: str
    status: int
    message: str
    elapsed_ns: int
    objective_value: float | None
    mip_gap: float | None
    mip_node_count: int | None
    solution_sha256: str | None
    scipy_version: str
    highs_version: str
    success: bool

    def __post_init__(self) -> None:
        if self.stage not in {"service_maximization", "economic_cost_minimization"}:
            raise ValueError("economic-MPC solve stage is invalid")
        if not isinstance(self.success, bool):
            raise TypeError("success must be a bool")
        if isinstance(self.status, bool) or not isinstance(self.status, Integral):
            raise TypeError("status must be an integer")
        if (
            not isinstance(self.message, str)
            or not self.message
            or len(self.message) > 500
        ):
            raise ValueError("message must contain at most 500 characters")
        _index("elapsed_ns", self.elapsed_ns)
        if not isinstance(self.scipy_version, str) or not self.scipy_version:
            raise ValueError("scipy_version must be non-empty")
        if not isinstance(self.highs_version, str) or not self.highs_version:
            raise ValueError("highs_version must be non-empty")
        for name in ("objective_value", "mip_gap"):
            value = getattr(self, name)
            if value is not None:
                _finite(name, value)
        if self.mip_gap is not None and self.mip_gap < 0.0:
            raise ValueError("mip_gap must be non-negative")
        if self.mip_node_count is not None:
            _index("mip_node_count", self.mip_node_count)
        if self.success:
            if self.status != 0 or self.objective_value is None:
                raise ValueError("successful stage requires status zero and objective")
            if self.mip_gap is None or self.mip_node_count is None:
                raise ValueError("successful stage requires complete MIP evidence")
            require_sha256("solution_sha256", self.solution_sha256)
        elif self.solution_sha256 is not None:
            raise ValueError("failed stage cannot publish a solution digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "success": self.success,
            "status": int(self.status),
            "message": self.message,
            "elapsed_ns": int(self.elapsed_ns),
            "objective_value": self.objective_value,
            "mip_gap": self.mip_gap,
            "mip_node_count": self.mip_node_count,
            "solution_sha256": self.solution_sha256,
            "solver": ECONOMIC_MPC_SOLVER,
            "scipy_version": self.scipy_version,
            "highs_version": self.highs_version,
        }


@dataclass(frozen=True, slots=True)
class EconomicMPCPlanV1:
    """Selected economic plan with an explicit predicted-cost ledger."""

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

    def __post_init__(self) -> None:
        for name in (
            "model_sha256",
            "forecast_sha256",
            "settings_sha256",
            "assumptions_sha256",
        ):
            require_sha256(name, getattr(self, name))
        count = _index("horizon_intervals", self.horizon_intervals)
        if count == 0:
            raise ValueError("horizon_intervals must be positive")
        if not isinstance(self.deliveries, tuple) or any(
            not isinstance(item, MPCDeliveryTargetV1) for item in self.deliveries
        ):
            raise TypeError("economic plan deliveries must be an immutable typed tuple")
        if tuple(item.ordinal for item in self.deliveries) != tuple(
            range(len(self.deliveries))
        ):
            raise ValueError("economic plan delivery ordinals must be contiguous")
        if any(item.due_boundary > count for item in self.deliveries):
            raise ValueError("economic plan delivery lies beyond the selected horizon")
        if (
            not isinstance(self.stages, tuple)
            or len(self.stages) != 2
            or any(
                not isinstance(stage, EconomicMPCStageSolveV1) for stage in self.stages
            )
        ):
            raise ValueError("economic plan requires exactly two stages")
        if tuple(stage.stage for stage in self.stages) != (
            "service_maximization",
            "economic_cost_minimization",
        ) or not all(stage.success for stage in self.stages):
            raise ValueError("economic plan requires two ordered successful stages")
        if not isinstance(self.steps, tuple) or len(self.steps) != count:
            raise ValueError("economic plan steps must cover the horizon")
        if any(
            not isinstance(step, MPCPlanStepV1) or step.index != index
            for index, step in enumerate(self.steps)
        ):
            raise ValueError("economic plan steps must be typed and contiguous")
        primary = _finite(
            "primary_delivered_optimum_kg",
            self.primary_delivered_optimum_kg,
            minimum=0.0,
        )
        selected = _finite(
            "selected_delivered_kg", self.selected_delivered_kg, minimum=0.0
        )
        tolerance = _finite(
            "delivery_lock_tolerance_kg",
            self.delivery_lock_tolerance_kg,
            minimum=0.0,
        )
        if tolerance <= 0.0 or primary - selected > (
            tolerance + MODEL_BALANCE_TOLERANCE
        ):
            raise ValueError("selected delivery violates the primary service lock")
        if selected > primary + tolerance + MODEL_BALANCE_TOLERANCE:
            raise ValueError("selected delivery exceeds the primary service optimum")
        if not math.isclose(
            math.fsum(step.delivered_kg for step in self.steps),
            selected,
            rel_tol=0.0,
            abs_tol=MODEL_BALANCE_TOLERANCE,
        ):
            raise ValueError("economic plan selected delivery does not match its steps")
        if any(
            self.steps[item.due_boundary - 1].delivered_kg
            > item.requested_kg + MODEL_BALANCE_TOLERANCE
            for item in self.deliveries
        ):
            raise ValueError("economic plan delivers above a declared request")
        if not isinstance(self.predicted_cost_components_eur, tuple):
            raise TypeError("predicted cost components must be immutable")
        names = tuple(name for name, _ in self.predicted_cost_components_eur)
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("predicted cost component names must be unique")
        for name, value in self.predicted_cost_components_eur:
            if not isinstance(name, str):
                raise TypeError("predicted cost component name must be text")
            _finite(name, value, minimum=0.0)
        stage_one_objective = self.stages[0].objective_value
        stage_two_objective = self.stages[1].objective_value
        assert stage_one_objective is not None
        assert stage_two_objective is not None
        if not math.isclose(
            -stage_one_objective,
            primary,
            rel_tol=0.0,
            abs_tol=MODEL_BALANCE_TOLERANCE,
        ):
            raise ValueError(
                "primary service evidence does not match the selected plan"
            )
        if not math.isclose(
            math.fsum(value for _, value in self.predicted_cost_components_eur),
            stage_two_objective,
            rel_tol=0.0,
            abs_tol=max(
                MODEL_BALANCE_TOLERANCE,
                1e-9 * max(1.0, abs(stage_two_objective)),
            ),
        ):
            raise ValueError("predicted cost ledger does not match stage-two evidence")
        _finite(
            "terminal_ch4_equivalent_deficit_kg",
            self.terminal_ch4_equivalent_deficit_kg,
            minimum=0.0,
        )
        _finite(
            "terminal_battery_deficit_kwh",
            self.terminal_battery_deficit_kwh,
            minimum=0.0,
        )

    @property
    def predicted_economic_cost_eur(self) -> float:
        return math.fsum(value for _, value in self.predicted_cost_components_eur)

    def body(self) -> dict[str, Any]:
        return {
            "schema_version": ECONOMIC_MPC_SCHEMA_VERSION,
            "controller_id": ECONOMIC_MPC_CONTROLLER_ID,
            "model_version": ECONOMIC_MPC_MODEL_VERSION,
            "objective_version": ECONOMIC_MPC_OBJECTIVE_VERSION,
            "model_sha256": self.model_sha256,
            "forecast_sha256": self.forecast_sha256,
            "settings_sha256": self.settings_sha256,
            "assumptions_sha256": self.assumptions_sha256,
            "horizon_intervals": int(self.horizon_intervals),
            "deliveries": [item.to_dict() for item in self.deliveries],
            "stages": [stage.to_dict() for stage in self.stages],
            "steps": [step.to_dict() for step in self.steps],
            "primary_delivered_optimum_kg": float(self.primary_delivered_optimum_kg),
            "selected_delivered_kg": float(self.selected_delivered_kg),
            "delivery_lock_tolerance_kg": float(self.delivery_lock_tolerance_kg),
            "predicted_cost_components_eur": {
                name: float(value) for name, value in self.predicted_cost_components_eur
            },
            "predicted_economic_cost_eur": self.predicted_economic_cost_eur,
            "terminal_ch4_equivalent_deficit_kg": float(
                self.terminal_ch4_equivalent_deficit_kg
            ),
            "terminal_battery_deficit_kwh": float(self.terminal_battery_deficit_kwh),
        }

    @property
    def sha256(self) -> str:
        # Timing and solver prose are evidence, not semantic plan identity.
        body = self.body()
        body["stages"] = [
            {"stage": stage.stage, "solution_sha256": stage.solution_sha256}
            for stage in self.stages
        ]
        return domain_sha256(ECONOMIC_MPC_PLAN_DOMAIN, body)

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class EconomicMPCDecisionEvidenceV1:
    """Hash-bound source evidence used to issue one common process command."""

    decision_index: int
    economic_observation_sha256: str
    physical_observation_sha256: str
    projection_sha256: str
    settings_sha256: str
    assumptions_sha256: str
    model_sha256: str
    plan_sha256: str
    tariff_id: str
    targets: ProcessSetpointTargetsV1
    plan: EconomicMPCPlanV1

    schema: ClassVar[str] = ECONOMIC_MPC_DECISION_SCHEMA

    def __post_init__(self) -> None:
        _index("decision_index", self.decision_index)
        for name in (
            "economic_observation_sha256",
            "physical_observation_sha256",
            "projection_sha256",
            "settings_sha256",
            "assumptions_sha256",
            "model_sha256",
            "plan_sha256",
        ):
            require_sha256(name, getattr(self, name))
        if not isinstance(self.tariff_id, str) or not self.tariff_id.strip():
            raise ValueError("tariff_id must be non-empty")
        if not isinstance(self.targets, ProcessSetpointTargetsV1):
            raise TypeError("targets must be ProcessSetpointTargetsV1")
        if not isinstance(self.plan, EconomicMPCPlanV1):
            raise TypeError("plan must be EconomicMPCPlanV1")
        if self.plan.sha256 != self.plan_sha256:
            raise ValueError("decision plan digest differs from its typed plan")
        if self.plan.model_sha256 != self.model_sha256:
            raise ValueError("decision model digest differs from its typed plan")
        if self.plan.forecast_sha256 != self.projection_sha256:
            raise ValueError("decision projection digest differs from its typed plan")
        if self.plan.settings_sha256 != self.settings_sha256:
            raise ValueError("decision settings digest differs from its typed plan")
        if self.plan.assumptions_sha256 != self.assumptions_sha256:
            raise ValueError("decision assumptions digest differs from its typed plan")
        first = self.plan.steps[0]
        expected_targets = ProcessSetpointTargetsV1(
            electrolyser_target_kw=(
                first.electrolyser_startup_kw + first.electrolyser_productive_kw
            ),
            dac_target_kw=first.solar_to_dac_kw,
            methanator_target_kw=first.solar_to_methanator_kw,
            battery_charge_ceiling_kw=first.solar_to_battery_kw,
        )
        if self.targets != expected_targets:
            raise ValueError(
                "decision targets do not match the first economic plan step"
            )

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "controller_id": ECONOMIC_MPC_CONTROLLER_ID,
            "decision_index": int(self.decision_index),
            "economic_observation_sha256": self.economic_observation_sha256,
            "physical_observation_sha256": self.physical_observation_sha256,
            "projection_sha256": self.projection_sha256,
            "settings_sha256": self.settings_sha256,
            "assumptions_sha256": self.assumptions_sha256,
            "model_sha256": self.model_sha256,
            "plan_sha256": self.plan_sha256,
            "tariff_id": self.tariff_id,
            "targets": self.targets.to_dict(),
            "plan": self.plan.to_dict(),
        }

    @property
    def sha256(self) -> str:
        return domain_sha256(ECONOMIC_MPC_DECISION_DOMAIN, self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class EconomicMPCDecisionV1:
    evidence: EconomicMPCDecisionEvidenceV1
    command: ProcessSetpointCommandV1

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, EconomicMPCDecisionEvidenceV1):
            raise TypeError("evidence must be EconomicMPCDecisionEvidenceV1")
        if not isinstance(self.command, ProcessSetpointCommandV1):
            raise TypeError("command must be ProcessSetpointCommandV1")
        if self.command.decision_index != self.evidence.decision_index:
            raise ValueError("command and evidence decision indices differ")
        if self.command.observation_sha256 != self.evidence.physical_observation_sha256:
            raise ValueError("command and evidence observations differ")
        if self.command.issuer_id != ECONOMIC_MPC_CONTROLLER_ID:
            raise ValueError("command issuer is not economic MPC")
        if self.command.source_schema != self.evidence.schema:
            raise ValueError("command source schema differs from evidence")
        if self.command.source_sha256 != self.evidence.sha256:
            raise ValueError("command source digest differs from evidence")
        if self.command.targets != self.evidence.targets:
            raise ValueError("command targets differ from evidence")

    def to_dict(self) -> dict[str, Any]:
        return {"evidence": self.evidence.to_dict(), "command": self.command.to_dict()}


class EconomicMPCError(RuntimeError):
    """Base class for explicit economic-MPC failures."""


class EconomicMPCInputError(EconomicMPCError):
    """The declared public economic input is inconsistent or unsupported."""


class EconomicMPCSolveError(EconomicMPCError):
    """A solver stage failed; no substitute command may be emitted."""

    def __init__(
        self,
        message: str,
        *,
        stages: tuple[EconomicMPCStageSolveV1, ...] = (),
    ) -> None:
        super().__init__(message)
        self.stages = stages


class EconomicMPCPlanError(EconomicMPCError):
    """A numerical solution failed independent plan validation."""


__all__ = [
    "DEFAULT_ECONOMIC_MPC_SETTINGS_V1",
    "ECONOMIC_MPC_CONTROLLER_ID",
    "ECONOMIC_MPC_DECISION_SCHEMA",
    "ECONOMIC_MPC_MODEL_MISMATCH_DECLARATION",
    "ECONOMIC_MPC_MODEL_VERSION",
    "ECONOMIC_MPC_OBJECTIVE_VERSION",
    "ECONOMIC_MPC_SCHEMA_VERSION",
    "EconomicMPCDecisionEvidenceV1",
    "EconomicMPCDecisionV1",
    "EconomicMPCError",
    "EconomicMPCInputError",
    "EconomicMPCPlanError",
    "EconomicMPCPlanV1",
    "EconomicMPCSettingsV1",
    "EconomicMPCSolveError",
    "EconomicMPCStageSolveV1",
]

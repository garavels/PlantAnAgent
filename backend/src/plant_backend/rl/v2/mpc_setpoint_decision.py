"""Typed supervisory decisions and failures for the absolute-setpoint MPC."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, ClassVar

from .mpc_execution import MPCFirstIntervalResidualV1
from .mpc_schema import (
    MODEL_BALANCE_TOLERANCE,
    MPCDeliveryTargetV1,
    MPCPlanStepV1,
    MPCStageSolveV1,
    SERVICE_MPC_SETPOINT_CONTROLLER_ID,
    domain_sha256,
    require_sha256,
)
from .mpc_setpoint_execution import MPCSetpointIntervalExecutionV1
from .mpc_setpoints import MPCProcessSetpointsV1

SERVICE_MPC_SETPOINT_DECISION_SCHEMA = "service-mpc-setpoint-decision-v1"
SERVICE_MPC_SETPOINT_DECISION_EVIDENCE_SCHEMA = (
    "service-mpc-setpoint-decision-evidence-v1"
)
SERVICE_MPC_SETPOINT_DECISION_EVIDENCE_DOMAIN = (
    SERVICE_MPC_SETPOINT_DECISION_EVIDENCE_SCHEMA
)
SERVICE_MPC_SETPOINT_FAILURE_SCHEMA = "service-mpc-setpoint-decision-failure-v1"
SERVICE_MPC_SETPOINT_FAILURE_DOMAIN = SERVICE_MPC_SETPOINT_FAILURE_SCHEMA


def _index(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_index(name: str, value: object) -> int:
    result = _index(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


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


def _notes(value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError("setpoint decision notes must be a non-empty immutable tuple")
    return value


def _same(left: float, right: float) -> bool:
    scale = max(1.0, abs(left), abs(right))
    tolerance = MODEL_BALANCE_TOLERANCE + MODEL_BALANCE_TOLERANCE * scale
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


@dataclass(frozen=True, slots=True)
class MPCSetpointDecisionV1:
    """One controller output: immutable absolute intent plus operator notes."""

    setpoints: MPCProcessSetpointsV1
    notes: tuple[str, ...]

    schema: ClassVar[str] = SERVICE_MPC_SETPOINT_DECISION_SCHEMA
    controller_id: ClassVar[str] = SERVICE_MPC_SETPOINT_CONTROLLER_ID

    def __post_init__(self) -> None:
        if not isinstance(self.setpoints, MPCProcessSetpointsV1):
            raise TypeError("decision setpoints must be MPCProcessSetpointsV1")
        _notes(self.notes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "controller_id": self.controller_id,
            "setpoints": self.setpoints.to_dict(),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class MPCSetpointDecisionEvidenceV1:
    """One successful plan and its optional closed physical realization."""

    decision_index: int
    observation_sha256: str
    public_forecast_sha256: str
    projection_sha256: str
    settings_sha256: str
    model_sha256: str
    model_variable_count: int
    model_binary_variable_count: int
    model_row_count: int
    model_nonzero_count: int
    horizon_intervals: int
    deliveries: tuple[MPCDeliveryTargetV1, ...]
    stages: tuple[MPCStageSolveV1, MPCStageSolveV1]
    plan_sha256: str
    primary_delivered_optimum_kg: float
    selected_delivered_kg: float
    minimum_commitment_fill: float
    terminal_ch4_equivalent_kg: float
    secondary_utility: float
    first_step: MPCPlanStepV1
    setpoints: MPCProcessSetpointsV1
    notes: tuple[str, ...]
    decision_elapsed_ns: int
    execution: MPCSetpointIntervalExecutionV1 | None = None
    residual: MPCFirstIntervalResidualV1 | None = None

    schema: ClassVar[str] = SERVICE_MPC_SETPOINT_DECISION_EVIDENCE_SCHEMA
    controller_id: ClassVar[str] = SERVICE_MPC_SETPOINT_CONTROLLER_ID

    def __post_init__(self) -> None:
        index = _index("decision_index", self.decision_index)
        for name in (
            "observation_sha256",
            "public_forecast_sha256",
            "projection_sha256",
            "settings_sha256",
            "model_sha256",
            "plan_sha256",
        ):
            require_sha256(name, getattr(self, name))
        for name in (
            "model_variable_count",
            "model_row_count",
            "model_nonzero_count",
            "horizon_intervals",
        ):
            _positive_index(name, getattr(self, name))
        binary_count = _index(
            "model_binary_variable_count", self.model_binary_variable_count
        )
        if binary_count > self.model_variable_count:
            raise ValueError("binary variable count exceeds total model variables")
        if not isinstance(self.deliveries, tuple) or any(
            not isinstance(item, MPCDeliveryTargetV1) for item in self.deliveries
        ):
            raise TypeError("decision deliveries must be an immutable typed tuple")
        if not isinstance(self.stages, tuple) or len(self.stages) != 2:
            raise ValueError("decision evidence requires exactly two solver stages")
        if tuple(stage.stage for stage in self.stages) != (
            "primary_delivery",
            "secondary_tiebreak",
        ) or not all(stage.success for stage in self.stages):
            raise ValueError("decision evidence requires ordered successful stages")
        if any(stage.solution_selection is None for stage in self.stages):
            raise ValueError(
                "setpoint decision requires feasibility-selection evidence"
            )
        for name in (
            "primary_delivered_optimum_kg",
            "selected_delivered_kg",
            "terminal_ch4_equivalent_kg",
        ):
            _finite(name, getattr(self, name), minimum=0.0)
        _finite(
            "minimum_commitment_fill",
            self.minimum_commitment_fill,
            minimum=0.0,
            maximum=1.0,
        )
        _finite("secondary_utility", self.secondary_utility)
        if not isinstance(self.first_step, MPCPlanStepV1) or self.first_step.index != 0:
            raise ValueError("decision evidence requires plan interval zero")
        if not isinstance(self.setpoints, MPCProcessSetpointsV1):
            raise TypeError("decision evidence requires typed setpoints")
        _notes(self.notes)
        if (
            isinstance(self.decision_elapsed_ns, bool)
            or not isinstance(self.decision_elapsed_ns, Integral)
            or self.decision_elapsed_ns < 0
        ):
            raise ValueError("decision_elapsed_ns must be non-negative")
        if self.decision_elapsed_ns < sum(stage.elapsed_ns for stage in self.stages):
            raise ValueError("decision latency cannot be below summed solver latency")

        command = self.setpoints
        bindings = (
            ("decision", command.decision_index, index),
            ("observation", command.observation_sha256, self.observation_sha256),
            ("forecast", command.public_forecast_sha256, self.public_forecast_sha256),
            ("projection", command.projection_sha256, self.projection_sha256),
            ("settings", command.settings_sha256, self.settings_sha256),
            ("model", command.model_sha256, self.model_sha256),
            ("plan", command.plan_sha256, self.plan_sha256),
        )
        if any(left != right for _, left, right in bindings):
            mismatch = next(name for name, left, right in bindings if left != right)
            raise ValueError(f"setpoint decision {mismatch} identity differs")
        power_bindings = (
            (
                "electrolyser",
                command.electrolyser_target_kw,
                self.first_step.electrolyser_startup_kw
                + self.first_step.electrolyser_productive_kw,
            ),
            ("DAC", command.dac_target_kw, self.first_step.solar_to_dac_kw),
            (
                "methanator",
                command.methanator_target_kw,
                self.first_step.solar_to_methanator_kw,
            ),
            (
                "battery charge",
                command.battery_charge_ceiling_kw,
                self.first_step.solar_to_battery_kw,
            ),
            (
                "expected support",
                command.expected_battery_support_kw,
                self.first_step.battery_support_kw,
            ),
        )
        if any(not _same(left, right) for _, left, right in power_bindings):
            mismatch = next(
                name for name, left, right in power_bindings if not _same(left, right)
            )
            raise ValueError(f"setpoint decision {mismatch} differs from the plan")

        if (self.execution is None) != (self.residual is None):
            raise ValueError("execution and public residual must close together")
        if self.execution is not None:
            if (
                not isinstance(self.execution, MPCSetpointIntervalExecutionV1)
                or self.execution.setpoints != command
            ):
                raise ValueError("decision execution belongs to another setpoint")
            if (
                not isinstance(self.residual, MPCFirstIntervalResidualV1)
                or self.residual.decision_index != index
            ):
                raise ValueError("decision residual belongs to another decision")

    @property
    def closed(self) -> bool:
        return self.execution is not None and self.residual is not None

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "controller_id": self.controller_id,
            "decision_index": int(self.decision_index),
            "observation_sha256": self.observation_sha256,
            "public_forecast_sha256": self.public_forecast_sha256,
            "projection_sha256": self.projection_sha256,
            "settings_sha256": self.settings_sha256,
            "model": {
                "sha256": self.model_sha256,
                "variables": int(self.model_variable_count),
                "binary_variables": int(self.model_binary_variable_count),
                "rows": int(self.model_row_count),
                "nonzeros": int(self.model_nonzero_count),
            },
            "horizon_intervals": int(self.horizon_intervals),
            "deliveries": [item.to_dict() for item in self.deliveries],
            "stages": [stage.to_dict() for stage in self.stages],
            "plan": {
                "sha256": self.plan_sha256,
                "primary_delivered_optimum_kg": float(
                    self.primary_delivered_optimum_kg
                ),
                "selected_delivered_kg": float(self.selected_delivered_kg),
                "minimum_commitment_fill": float(self.minimum_commitment_fill),
                "terminal_ch4_equivalent_kg": float(self.terminal_ch4_equivalent_kg),
                "secondary_utility": float(self.secondary_utility),
            },
            "first_step": self.first_step.to_dict(),
            "setpoints": self.setpoints.to_dict(),
            "notes": list(self.notes),
            "decision_elapsed_ns": int(self.decision_elapsed_ns),
            "execution": (None if self.execution is None else self.execution.to_dict()),
            "residual": None if self.residual is None else self.residual.to_dict(),
        }

    def identity_body(self) -> dict[str, Any]:
        body = self.body()
        del body["decision_elapsed_ns"]
        del body["stages"]
        body["stage_solutions"] = [
            {
                "stage": stage.stage,
                "solution_sha256": stage.solution_sha256,
                "solution_selection": (
                    None
                    if stage.solution_selection is None
                    else stage.solution_selection.to_dict()
                ),
            }
            for stage in self.stages
        ]
        return body

    @property
    def sha256(self) -> str:
        return str(
            domain_sha256(
                SERVICE_MPC_SETPOINT_DECISION_EVIDENCE_DOMAIN,
                self.identity_body(),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class MPCSetpointDecisionFailureV1:
    """One failed setpoint-controller attempt; no command or fallback is invented."""

    decision_index: int
    phase: str
    exception_type: str
    message: str
    elapsed_ns: int
    settings_sha256: str
    observation_sha256: str | None = None
    public_forecast_sha256: str | None = None
    projection_sha256: str | None = None
    model_sha256: str | None = None
    plan_sha256: str | None = None
    setpoint_sha256: str | None = None
    stages: tuple[MPCStageSolveV1, ...] = ()

    schema: ClassVar[str] = SERVICE_MPC_SETPOINT_FAILURE_SCHEMA
    controller_id: ClassVar[str] = SERVICE_MPC_SETPOINT_CONTROLLER_ID

    def __post_init__(self) -> None:
        _index("failure decision_index", self.decision_index)
        if self.phase not in {
            "transition_reconciliation",
            "input_validation",
            "forecast_projection",
            "model_construction",
            "solver",
            "setpoint_realization",
            "evidence_construction",
        }:
            raise ValueError("setpoint-controller failure phase is invalid")
        for name in ("exception_type", "message"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"failure {name} must be non-empty")
        if len(self.message) > 500:
            raise ValueError("failure message exceeds the evidence bound")
        if (
            isinstance(self.elapsed_ns, bool)
            or not isinstance(self.elapsed_ns, Integral)
            or self.elapsed_ns < 0
        ):
            raise ValueError("failure elapsed_ns must be non-negative")
        require_sha256("settings_sha256", self.settings_sha256)
        for name in (
            "observation_sha256",
            "public_forecast_sha256",
            "projection_sha256",
            "model_sha256",
            "plan_sha256",
            "setpoint_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                require_sha256(name, value)
        if not isinstance(self.stages, tuple) or any(
            not isinstance(stage, MPCStageSolveV1) for stage in self.stages
        ):
            raise TypeError("failure stages must be an immutable typed tuple")
        expected = ("primary_delivery", "secondary_tiebreak")[: len(self.stages)]
        if (
            len(self.stages) > 2
            or tuple(stage.stage for stage in self.stages) != expected
            or any(not stage.success for stage in self.stages[:-1])
        ):
            raise ValueError("failure stages must be an ordered attempted prefix")
        if any(
            stage.success and stage.solution_selection is None for stage in self.stages
        ):
            raise ValueError("successful failure-prefix stages need selection evidence")
        if self.stages and self.phase not in {
            "solver",
            "setpoint_realization",
            "evidence_construction",
        }:
            raise ValueError("failure phase cannot contain solver-stage evidence")
        if self.phase in {"setpoint_realization", "evidence_construction"} and any(
            not stage.success for stage in self.stages
        ):
            raise ValueError("post-solve failure cannot contain a failed solver stage")

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "controller_id": self.controller_id,
            "decision_index": int(self.decision_index),
            "phase": self.phase,
            "exception_type": self.exception_type,
            "message": self.message,
            "elapsed_ns": int(self.elapsed_ns),
            "settings_sha256": self.settings_sha256,
            "observation_sha256": self.observation_sha256,
            "public_forecast_sha256": self.public_forecast_sha256,
            "projection_sha256": self.projection_sha256,
            "model_sha256": self.model_sha256,
            "plan_sha256": self.plan_sha256,
            "setpoint_sha256": self.setpoint_sha256,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def identity_body(self) -> dict[str, Any]:
        body = self.body()
        del body["elapsed_ns"]
        stages = body.pop("stages")
        assert isinstance(stages, list)
        body["stages"] = [
            {key: value for key, value in stage.items() if key != "elapsed_ns"}
            for stage in stages
            if isinstance(stage, dict)
        ]
        return body

    @property
    def sha256(self) -> str:
        return str(
            domain_sha256(
                SERVICE_MPC_SETPOINT_FAILURE_DOMAIN,
                self.identity_body(),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


MPCSetpointDecisionEventV1 = (
    MPCSetpointDecisionEvidenceV1 | MPCSetpointDecisionFailureV1
)


__all__ = [
    "MPCSetpointDecisionEventV1",
    "MPCSetpointDecisionEvidenceV1",
    "MPCSetpointDecisionFailureV1",
    "MPCSetpointDecisionV1",
    "SERVICE_MPC_SETPOINT_DECISION_EVIDENCE_DOMAIN",
    "SERVICE_MPC_SETPOINT_DECISION_EVIDENCE_SCHEMA",
    "SERVICE_MPC_SETPOINT_DECISION_SCHEMA",
    "SERVICE_MPC_SETPOINT_FAILURE_DOMAIN",
    "SERVICE_MPC_SETPOINT_FAILURE_SCHEMA",
]

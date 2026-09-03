"""Causal receding-horizon service-MPC controller for Environment V2."""

from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import ClassVar

from plant_sim import ElectrolyserCommitmentProfile, PlantConfig

from ...controllers.base import Constraints, ControlDecision
from .control import ControllerObservationV2
from .mpc_execution import (
    MPCDecisionEventV2,
    MPCDecisionEvidenceV1,
    MPCDecisionFailureV2,
    SERVICE_MPC_FAILURE_SCHEMA,
    SERVICE_MPC_RUN_EVIDENCE_FORMAT,
    public_forecast_issue_sha256,
    public_observation_sha256,
    realize_first_mpc_action,
    reconcile_first_interval,
)
from .mpc_forecast import MPCForecastTrajectoryV1, project_issued_mean_solar
from .mpc_model import ServiceMPCModelV1, build_service_mpc_model
from .mpc_schema import (
    DEFAULT_SERVICE_MPC_SETTINGS_V1,
    MPCPlanV1,
    MPCPlanStepV1,
    MPCStageSolveV1,
    SERVICE_MPC_CONTROLLER_ID,
    ServiceMPCPlanError,
    ServiceMPCSettingsV1,
    ServiceMPCSolveError,
)
from .mpc_solver import solve_service_mpc_model
from .weather import PublicForecastIssueV2


@dataclass(frozen=True, slots=True)
class _PendingDecision:
    decision_index: int
    evidence_position: int
    before: ControllerObservationV2
    first_step: MPCPlanStepV1


class ServiceMPCControllerV1:
    """Deterministic issued-mean MILP with no controller-internal fallback."""

    id: ClassVar[str] = SERVICE_MPC_CONTROLLER_ID
    label: ClassVar[str] = "Service MPC (issued mean forecast)"
    description: ClassVar[str] = (
        "Two-stage causal receding-horizon optimizer for delivery service, "
        "inventories, battery endowment, solar routing, and electrolyser commitment."
    )
    uses_forecast: ClassVar[bool] = True
    information_set: ClassVar[str] = (
        "current detached ControllerObservationV2, current PublicForecastIssueV2 mean "
        "bands, static PlantConfig, active Constraints, public commitment profile and "
        "snapshot, immutable settings, and controller-local prior public residuals; no "
        "private weather truth, future forecast vintage, episode seed, simulator, "
        "delivery schedule, or privileged planner result"
    )

    def __init__(self, settings: ServiceMPCSettingsV1 | None = None) -> None:
        chosen = DEFAULT_SERVICE_MPC_SETTINGS_V1 if settings is None else settings
        if not isinstance(chosen, ServiceMPCSettingsV1):
            raise TypeError("settings must be ServiceMPCSettingsV1")
        self.settings = chosen
        self._events: list[MPCDecisionEventV2] = []
        self._pending: _PendingDecision | None = None
        self._decision_count = 0
        self._origin_step: int | None = None
        self._origin_time_s: float | None = None
        self._physics_stride_steps: int | None = None

    def reset(self) -> None:
        self._events.clear()
        self._pending = None
        self._decision_count = 0
        self._origin_step = None
        self._origin_time_s = None
        self._physics_stride_steps = None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "uses_forecast": self.uses_forecast,
            "information_set": self.information_set,
            "settings": self.settings.to_dict(),
            "solver_runtime_identity": (
                "SciPy and bundled HiGHS versions recorded independently in every "
                "attempted solve stage"
            ),
            "failure_policy": "raise typed error; no controller-internal fallback",
            "decision_evidence_schema": "service-mpc-decision-evidence-v1",
            "decision_failure_schema": SERVICE_MPC_FAILURE_SCHEMA,
            "run_evidence_format": SERVICE_MPC_RUN_EVIDENCE_FORMAT,
        }

    def descriptor(self) -> dict[str, object]:
        return self.to_dict()

    @property
    def decision_events(self) -> tuple[MPCDecisionEventV2, ...]:
        return tuple(self._events)

    @property
    def has_pending_transition(self) -> bool:
        return self._pending is not None

    def evidence(self) -> tuple[dict[str, object], ...]:
        return tuple(event.to_dict() for event in self._events)

    @staticmethod
    def _finite(name: str, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        return number

    def _physics_stride(self, config: PlantConfig) -> int:
        dt_s = self._finite("plant physics timestep", config.dt_s)
        if dt_s <= 0.0:
            raise ValueError("plant physics timestep must be positive")
        ratio = self.settings.interval_s / dt_s
        if ratio <= 0.0 or not math.isclose(
            ratio,
            round(ratio),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("service-MPC interval must contain whole physics steps")
        return int(round(ratio))

    def _validate_current_boundary(
        self,
        observation: ControllerObservationV2,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: ElectrolyserCommitmentProfile | None,
    ) -> int:
        if not isinstance(observation, ControllerObservationV2):
            raise TypeError("observation must be ControllerObservationV2")
        if not isinstance(constraints, Constraints):
            raise TypeError("constraints must be Constraints")
        if not isinstance(config, PlantConfig):
            raise TypeError("config must be PlantConfig")
        if not isinstance(commitment_profile, ElectrolyserCommitmentProfile):
            raise TypeError("service MPC V1 requires an ElectrolyserCommitmentProfile")
        if not isinstance(observation.forecast, PublicForecastIssueV2):
            raise ValueError("service MPC requires a current public forecast issue")
        if not math.isclose(
            observation.decision_interval_s,
            self.settings.interval_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("controller cadence differs from service-MPC settings")
        stride = self._physics_stride(config)
        if self._physics_stride_steps is None:
            self._physics_stride_steps = stride
        elif self._physics_stride_steps != stride:
            raise ValueError("plant physics stride changed within the service-MPC task")
        boundary = observation.state.boundary
        if (
            isinstance(boundary.step, bool)
            or not isinstance(boundary.step, Integral)
            or boundary.step < 0
        ):
            raise ValueError("public boundary step must be a non-negative integer")
        boundary_time_s = self._finite("public boundary time", boundary.time_s)
        if boundary_time_s < 0.0:
            raise ValueError("public boundary time must be non-negative")
        if self._origin_step is None:
            self._origin_step = int(boundary.step)
            self._origin_time_s = boundary_time_s
        assert self._origin_time_s is not None
        expected_step = self._origin_step + self._decision_count * stride
        expected_time_s = (
            self._origin_time_s + self._decision_count * self.settings.interval_s
        )
        if boundary.step != expected_step or not math.isclose(
            boundary_time_s,
            expected_time_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("public boundary is not the next service-MPC decision")
        expected_progress = self._decision_count / self.settings.task_interval_count
        if not math.isclose(
            observation.progress,
            expected_progress,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "public progress differs from the service-MPC task horizon"
            )
        if observation.forecast.issued_at_physics_boundary_step != boundary.step:
            raise ValueError(
                "public forecast was not issued at the current physics step"
            )
        expected_offset_s = self._decision_count * self.settings.interval_s
        if not math.isclose(
            observation.forecast.issued_at_offset_s,
            expected_offset_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "public forecast issue offset is not the current task offset"
            )
        remaining = self.settings.task_interval_count - self._decision_count
        if remaining <= 0:
            raise ValueError("service-MPC task horizon is already complete")
        return min(self.settings.maximum_planning_intervals, remaining)

    def _validate_transition_boundary(
        self,
        observation: ControllerObservationV2,
        *,
        pending: _PendingDecision,
    ) -> None:
        if not isinstance(observation, ControllerObservationV2):
            raise TypeError("transition observation must be ControllerObservationV2")
        if self._origin_step is None or self._origin_time_s is None:
            raise RuntimeError("service-MPC transition has no task origin")
        if self._physics_stride_steps is None:
            raise RuntimeError("service-MPC transition has no physics stride")
        next_index = pending.decision_index + 1
        expected_step = self._origin_step + next_index * self._physics_stride_steps
        expected_time_s = self._origin_time_s + next_index * self.settings.interval_s
        if observation.state.boundary.step != expected_step or not math.isclose(
            observation.state.boundary.time_s,
            expected_time_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("service-MPC transition boundary is not contiguous")
        expected_progress = min(
            1.0,
            next_index / self.settings.task_interval_count,
        )
        if not math.isclose(
            observation.progress,
            expected_progress,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("transition progress differs from the task horizon")
        if not math.isclose(
            observation.decision_interval_s,
            self.settings.interval_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("transition controller cadence changed")

    def observe_transition(self, observation: ControllerObservationV2) -> None:
        """Close the pending plan with only the next public controller boundary."""

        pending = self._pending
        if pending is None:
            raise RuntimeError("service MPC has no pending decision to reconcile")
        self._validate_transition_boundary(observation, pending=pending)
        event = self._events[pending.evidence_position]
        if not isinstance(event, MPCDecisionEvidenceV1):
            raise RuntimeError("pending service-MPC event is not successful evidence")
        residual = reconcile_first_interval(
            decision_index=pending.decision_index,
            before=pending.before,
            after=observation,
            planned=pending.first_step,
        )
        self._events[pending.evidence_position] = replace(event, residual=residual)
        self._pending = None

    def _record_failure(
        self,
        *,
        phase: str,
        exception: Exception,
        started_ns: int,
        observation_sha256: str | None,
        public_forecast_sha256: str | None,
        projection: MPCForecastTrajectoryV1 | None,
        model: ServiceMPCModelV1 | None,
        plan: MPCPlanV1 | None,
    ) -> None:
        message = " ".join(str(exception).split()) or "(no exception message)"
        stages: tuple[MPCStageSolveV1, ...]
        if plan is not None:
            stages = plan.stages
        elif isinstance(exception, (ServiceMPCSolveError, ServiceMPCPlanError)):
            stages = exception.stages
        else:
            stages = ()
        if not isinstance(stages, tuple) or any(
            not isinstance(stage, MPCStageSolveV1) for stage in stages
        ):
            stages = ()
        failure = MPCDecisionFailureV2(
            decision_index=self._decision_count,
            phase=phase,
            exception_type=type(exception).__name__,
            message=message[:500],
            elapsed_ns=time.perf_counter_ns() - started_ns,
            observation_sha256=observation_sha256,
            public_forecast_sha256=public_forecast_sha256,
            projection_sha256=None if projection is None else projection.sha256,
            model_sha256=None if model is None else model.sha256,
            plan_sha256=None if plan is None else plan.sha256,
            stages=stages,
        )
        self._events.append(failure)

    def compute_action(
        self,
        observation: ControllerObservationV2,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: ElectrolyserCommitmentProfile | None,
    ) -> ControlDecision:
        started_ns = time.perf_counter_ns()
        phase = "transition_reconciliation"
        observation_sha: str | None = None
        forecast_sha: str | None = None
        projection: MPCForecastTrajectoryV1 | None = None
        model: ServiceMPCModelV1 | None = None
        plan: MPCPlanV1 | None = None
        try:
            if self._pending is not None:
                self.observe_transition(observation)
            phase = "input_validation"
            horizon = self._validate_current_boundary(
                observation,
                constraints=constraints,
                config=config,
                commitment_profile=commitment_profile,
            )
            observation_sha = public_observation_sha256(observation)
            forecast_sha = public_forecast_issue_sha256(observation)
            assert observation.forecast is not None

            phase = "forecast_projection"
            projection = project_issued_mean_solar(
                observation.forecast,
                boundary_time_s=observation.state.boundary.time_s,
                config=config,
                interval_s=self.settings.interval_s,
                energy_tolerance=self.settings.energy_reconstruction_tolerance,
            )

            phase = "model_construction"
            model = build_service_mpc_model(
                observation,
                projection,
                settings=self.settings,
                constraints=constraints,
                config=config,
                commitment_profile=commitment_profile,
                horizon_intervals=horizon,
            )

            phase = "solver"
            solved_plan = solve_service_mpc_model(model, settings=self.settings)
            plan = solved_plan

            phase = "action_realization"
            action = realize_first_mpc_action(solved_plan)
            notes = (
                f"Service MPC decision {self._decision_count}: "
                f"{horizon} x {self.settings.interval_s:g}s intervals; "
                f"model {model.sha256[:12]}",
                f"Horizon service optimum={solved_plan.primary_delivered_optimum_kg:.3f} kg; "
                f"selected={solved_plan.selected_delivered_kg:.3f} kg; "
                f"minimum fill={solved_plan.minimum_commitment_fill:.3%}",
                "First planned mean kW: "
                f"battery={solved_plan.steps[0].solar_to_battery_kw:.2f}, "
                f"electrolyser={solved_plan.steps[0].electrolyser_startup_kw + solved_plan.steps[0].electrolyser_productive_kw:.2f}, "
                f"DAC={solved_plan.steps[0].solar_to_dac_kw:.2f}, "
                f"methanator={solved_plan.steps[0].solar_to_methanator_kw:.2f}, "
                f"support={solved_plan.steps[0].battery_support_kw:.2f}",
            )

            phase = "evidence_construction"
            elapsed_ns = time.perf_counter_ns() - started_ns
            evidence = MPCDecisionEvidenceV1(
                decision_index=self._decision_count,
                observation_sha256=observation_sha,
                public_forecast_sha256=forecast_sha,
                projection_sha256=projection.sha256,
                settings_sha256=self.settings.sha256,
                model_sha256=model.sha256,
                model_variable_count=model.variable_count,
                model_binary_variable_count=model.binary_variable_count,
                model_row_count=model.row_count,
                model_nonzero_count=model.nonzero_count,
                horizon_intervals=horizon,
                deliveries=model.deliveries,
                stages=solved_plan.stages,
                plan_sha256=solved_plan.sha256,
                primary_delivered_optimum_kg=solved_plan.primary_delivered_optimum_kg,
                selected_delivered_kg=solved_plan.selected_delivered_kg,
                minimum_commitment_fill=solved_plan.minimum_commitment_fill,
                terminal_ch4_equivalent_kg=solved_plan.terminal_ch4_equivalent_kg,
                secondary_utility=solved_plan.secondary_utility,
                first_step=solved_plan.steps[0],
                action=action,
                notes=notes,
                decision_elapsed_ns=elapsed_ns,
            )
        except Exception as exc:
            self._record_failure(
                phase=phase,
                exception=exc,
                started_ns=started_ns,
                observation_sha256=observation_sha,
                public_forecast_sha256=forecast_sha,
                projection=projection,
                model=model,
                plan=plan,
            )
            raise

        self._events.append(evidence)
        self._pending = _PendingDecision(
            decision_index=self._decision_count,
            evidence_position=len(self._events) - 1,
            before=deepcopy(observation),
            first_step=solved_plan.steps[0],
        )
        self._decision_count += 1
        return ControlDecision(action=action.action, notes=notes, invalid=False)


__all__ = ["ServiceMPCControllerV1"]

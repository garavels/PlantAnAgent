"""Tariff-aware economic MPC issuing controller-neutral process commands."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import ClassVar

from plant_sim import ElectrolyserCommitmentProfile, PlantConfig

from ...controllers.base import Constraints
from .economic_mpc_model import build_economic_mpc_model
from .economic_mpc_schema import (
    DEFAULT_ECONOMIC_MPC_SETTINGS_V1,
    ECONOMIC_MPC_CONTROLLER_ID,
    ECONOMIC_MPC_DECISION_SCHEMA,
    ECONOMIC_MPC_MODEL_MISMATCH_DECLARATION,
    EconomicMPCDecisionEvidenceV1,
    EconomicMPCDecisionV1,
    EconomicMPCSettingsV1,
)
from .economic_mpc_solver import solve_economic_mpc_model
from .economics import (
    EconomicAssumptionsV1,
    EconomicControllerObservationV1,
    ch4_equivalent_inventory_kg,
)
from .mpc_execution import public_observation_sha256
from .mpc_forecast import project_issued_mean_solar
from .setpoint_command import issue_process_setpoint_command
from .setpoint_control import ProcessSetpointTargetsV1


class TariffAwareEconomicMPCControllerV1:
    """Lexicographic service/economic optimizer with no internal fallback."""

    id: ClassVar[str] = ECONOMIC_MPC_CONTROLLER_ID
    label: ClassVar[str] = "Tariff-aware economic MPC"
    description: ClassVar[str] = (
        "Issued-mean receding-horizon MILP that first maximizes delivery service "
        "and then minimizes a declared EUR cost on the service-optimal face."
    )
    uses_forecast: ClassVar[bool] = True
    information_set: ClassVar[str] = (
        "EconomicControllerObservationV1 (detached physical boundary, current public "
        "forecast issue, and tariff summary), the controller-declared full repeating "
        "24-hour tariff, economic assumptions, active Constraints, PlantConfig, "
        "commitment profile, and episode-opening public endowments captured at the "
        "first decision; no private weather truth, future forecast vintage, seed, "
        "simulator state, planner result, or fallback controller"
    )

    def __init__(
        self,
        settings: EconomicMPCSettingsV1 | None = None,
        assumptions: EconomicAssumptionsV1 | None = None,
    ) -> None:
        chosen_settings = (
            DEFAULT_ECONOMIC_MPC_SETTINGS_V1 if settings is None else settings
        )
        chosen_assumptions = (
            EconomicAssumptionsV1() if assumptions is None else assumptions
        )
        if not isinstance(chosen_settings, EconomicMPCSettingsV1):
            raise TypeError("settings must be EconomicMPCSettingsV1")
        if not isinstance(chosen_assumptions, EconomicAssumptionsV1):
            raise TypeError("assumptions must be EconomicAssumptionsV1")
        chosen_assumptions.validate()
        self.settings = chosen_settings
        self.assumptions = chosen_assumptions
        self.reset()

    def reset(self) -> None:
        self._decision_count = 0
        self._origin_step: int | None = None
        self._origin_time_s: float | None = None
        self._physics_stride_steps: int | None = None
        self._initial_ch4_equivalent_kg: float | None = None
        self._initial_battery_energy_kwh: float | None = None
        self._events: list[EconomicMPCDecisionEvidenceV1] = []

    @property
    def decision_events(self) -> tuple[EconomicMPCDecisionEvidenceV1, ...]:
        return tuple(self._events)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "uses_forecast": self.uses_forecast,
            "information_set": self.information_set,
            "settings": self.settings.to_dict(),
            "economic_assumptions": self.assumptions.to_dict(),
            "failure_policy": "raise typed error; no controller-internal fallback",
            "output_schema": "process-setpoint-command-v1",
            "source_schema": ECONOMIC_MPC_DECISION_SCHEMA,
            "model_mismatch_declaration": ECONOMIC_MPC_MODEL_MISMATCH_DECLARATION,
        }

    def descriptor(self) -> dict[str, object]:
        return self.to_dict()

    @staticmethod
    def _finite(name: str, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    def _validate_boundary(
        self,
        observation: EconomicControllerObservationV1,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: ElectrolyserCommitmentProfile | None,
    ) -> int:
        if not isinstance(observation, EconomicControllerObservationV1):
            raise TypeError("observation must be EconomicControllerObservationV1")
        if not isinstance(constraints, Constraints):
            raise TypeError("constraints must be Constraints")
        if not isinstance(config, PlantConfig):
            raise TypeError("config must be PlantConfig")
        if not isinstance(commitment_profile, ElectrolyserCommitmentProfile):
            raise TypeError("economic MPC requires a commitment profile")
        plant = observation.plant
        if plant.forecast is None:
            raise ValueError("economic MPC requires a current public forecast")
        if observation.economic_assumptions_sha256 != self.assumptions.sha256:
            raise ValueError("economic observation assumptions digest is stale")
        if observation.tariff_id != self.assumptions.tariff.id:
            raise ValueError("economic observation tariff id is stale")
        expected_tariff = self.assumptions.tariff.observe(plant.state.boundary.time_h)
        if observation.tariff != expected_tariff:
            raise ValueError(
                "economic observation tariff summary disagrees with the full schedule"
            )
        if not math.isclose(
            plant.decision_interval_s,
            self.settings.interval_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("controller cadence differs from economic-MPC settings")
        dt_s = self._finite("plant timestep", config.dt_s)
        ratio = self.settings.interval_s / dt_s
        if dt_s <= 0.0 or not math.isclose(
            ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("economic-MPC interval must contain whole physics steps")
        stride = int(round(ratio))
        boundary = plant.state.boundary
        if (
            isinstance(boundary.step, bool)
            or not isinstance(boundary.step, Integral)
            or boundary.step < 0
        ):
            raise ValueError("public boundary step must be non-negative")
        boundary_time_s = self._finite("public boundary time", boundary.time_s)
        if self._origin_step is None:
            self._origin_step = int(boundary.step)
            self._origin_time_s = boundary_time_s
            self._physics_stride_steps = stride
        assert self._origin_time_s is not None
        assert self._origin_step is not None
        assert self._physics_stride_steps is not None
        expected_step = (
            self._origin_step + self._decision_count * self._physics_stride_steps
        )
        expected_time = (
            self._origin_time_s + self._decision_count * self.settings.interval_s
        )
        if boundary.step != expected_step or not math.isclose(
            boundary_time_s, expected_time, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("public boundary is not the next economic-MPC decision")
        expected_progress = self._decision_count / self.settings.task_interval_count
        if not math.isclose(
            plant.progress, expected_progress, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "public progress differs from the economic-MPC task horizon"
            )
        if plant.forecast.issued_at_physics_boundary_step != boundary.step:
            raise ValueError("public forecast is not from the current boundary")
        remaining = self.settings.task_interval_count - self._decision_count
        if remaining <= 0:
            raise ValueError("economic-MPC task horizon is complete")
        return min(self.settings.maximum_planning_intervals, remaining)

    def compute_command(
        self,
        observation: EconomicControllerObservationV1,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: ElectrolyserCommitmentProfile | None = None,
    ) -> EconomicMPCDecisionV1:
        """Solve and issue one held command; every failure propagates to the caller."""

        horizon = self._validate_boundary(
            observation,
            constraints=constraints,
            config=config,
            commitment_profile=commitment_profile,
        )
        plant = observation.plant
        assert plant.forecast is not None
        physical_sha = public_observation_sha256(plant)
        projection = project_issued_mean_solar(
            plant.forecast,
            boundary_time_s=plant.state.boundary.time_s,
            config=config,
            interval_s=self.settings.interval_s,
            energy_tolerance=self.settings.energy_reconstruction_tolerance,
        )
        if self._initial_ch4_equivalent_kg is None:
            self._initial_ch4_equivalent_kg = ch4_equivalent_inventory_kg(
                plant.state.boundary, config
            )
            self._initial_battery_energy_kwh = plant.state.boundary.battery_energy_kwh
        assert self._initial_battery_energy_kwh is not None
        model = build_economic_mpc_model(
            observation,
            projection,
            settings=self.settings,
            assumptions=self.assumptions,
            constraints=constraints,
            config=config,
            commitment_profile=commitment_profile,
            horizon_intervals=horizon,
            initial_ch4_equivalent_kg=self._initial_ch4_equivalent_kg,
            initial_battery_energy_kwh=self._initial_battery_energy_kwh,
        )
        plan = solve_economic_mpc_model(model, settings=self.settings)
        first = plan.steps[0]
        targets = ProcessSetpointTargetsV1(
            electrolyser_target_kw=(
                first.electrolyser_startup_kw + first.electrolyser_productive_kw
            ),
            dac_target_kw=first.solar_to_dac_kw,
            methanator_target_kw=first.solar_to_methanator_kw,
            battery_charge_ceiling_kw=first.solar_to_battery_kw,
        )
        evidence = EconomicMPCDecisionEvidenceV1(
            decision_index=self._decision_count,
            economic_observation_sha256=observation.sha256,
            physical_observation_sha256=physical_sha,
            projection_sha256=projection.sha256,
            settings_sha256=self.settings.sha256,
            assumptions_sha256=self.assumptions.sha256,
            model_sha256=model.sha256,
            plan_sha256=plan.sha256,
            tariff_id=self.assumptions.tariff.id,
            targets=targets,
            plan=plan,
        )
        command = issue_process_setpoint_command(
            targets,
            decision_index=self._decision_count,
            observation_sha256=physical_sha,
            hold_duration_s=self.settings.interval_s,
            issuer_id=self.id,
            source_schema=evidence.schema,
            source_sha256=evidence.sha256,
            constraints=constraints,
            config=config,
        )
        result = EconomicMPCDecisionV1(evidence=evidence, command=command)
        self._events.append(evidence)
        self._decision_count += 1
        return result


__all__ = ["TariffAwareEconomicMPCControllerV1"]

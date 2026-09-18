"""Causal market- and cycle-aware economic MPC process controller."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import ClassVar

from plant_sim import ElectrolyserCommitmentProfile, PlantConfig

from ...controllers.base import Constraints
from .beta_config import CyclicMarketTaskConfigV1
from .beta_control import CyclicMarketControllerObservationV1
from .beta_mpc_model import build_beta_mpc_model
from .beta_mpc_schema import (
    BETA_MPC_CONTROLLER_ID,
    BETA_MPC_DECISION_SCHEMA,
    BETA_MPC_MODEL_MISMATCH_DECLARATION,
    BETA_MPC_UNCERTAINTY_POLICY,
    DEFAULT_BETA_MPC_SETTINGS_V1,
    BetaMPCCycleReturnV1,
    BetaMPCDecisionEvidenceV1,
    BetaMPCDecisionV1,
    BetaMPCResourceReturnV1,
    BetaMPCSettingsV1,
    beta_cycle_observation_sha256,
    project_beta_mpc_prices,
)
from .beta_mpc_solver import solve_beta_mpc_model
from .economics import EconomicAssumptionsV1, ch4_equivalent_inventory_kg
from .mpc_execution import public_observation_sha256
from .beta_mpc_uncertainty import project_beta_mpc_solar
from .setpoint_command import issue_process_setpoint_command
from .setpoint_control import ProcessSetpointTargetsV1


_SECONDS_PER_DAY = 24.0 * 3_600.0
_TARGET_ABS_TOLERANCE = 1e-9


class CyclicMarketEconomicMPCControllerV1:
    """Beta-native receding-horizon MPC with four hard daily returns."""

    id: ClassVar[str] = BETA_MPC_CONTROLLER_ID
    label: ClassVar[str] = "Cyclic-market economic MPC"
    description: ClassVar[str] = (
        "Causal issued-mean economic MPC ending exactly at the next 24-hour "
        "boundary and enforcing independent battery, H2, CO2, and CH4 returns."
    )
    uses_forecast: ClassVar[bool] = True
    information_set: ClassVar[str] = (
        "CyclicMarketControllerObservationV1 only: detached physical boundary, "
        "current public issued-mean solar forecast, one current public market "
        "issue (known opening price plus hourly forecast means), public cycle "
        "counters and signed resource margins; declared economic assumptions, "
        "beta task contract, Constraints, PlantConfig, and commitment profile; "
        "no market episode, realized future prices, selected regime, disturbance "
        "schedule, fault recovery time, seed, simulator state, later forecast "
        "vintage, or fallback controller"
    )

    def __init__(
        self,
        settings: BetaMPCSettingsV1 | None = None,
        assumptions: EconomicAssumptionsV1 | None = None,
        beta_task: CyclicMarketTaskConfigV1 | None = None,
    ) -> None:
        chosen_settings = DEFAULT_BETA_MPC_SETTINGS_V1 if settings is None else settings
        chosen_assumptions = (
            EconomicAssumptionsV1() if assumptions is None else assumptions
        )
        chosen_task = CyclicMarketTaskConfigV1() if beta_task is None else beta_task
        if not isinstance(chosen_settings, BetaMPCSettingsV1):
            raise TypeError("settings must be BetaMPCSettingsV1")
        if not isinstance(chosen_assumptions, EconomicAssumptionsV1):
            raise TypeError("assumptions must be EconomicAssumptionsV1")
        if not isinstance(chosen_task, CyclicMarketTaskConfigV1):
            raise TypeError("beta_task must be CyclicMarketTaskConfigV1")
        chosen_assumptions.validate()
        chosen_task.market.validate()
        if not math.isclose(
            float(chosen_task.cycle.cycle_h), 24.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("beta MPC requires an exact 24 h cycle")
        self.settings = chosen_settings
        self.assumptions = chosen_assumptions
        self.beta_task = chosen_task
        self.reset()

    def reset(self) -> None:
        self._decision_count = 0
        self._origin_step: int | None = None
        self._origin_time_s: float | None = None
        self._physics_stride_steps: int | None = None
        self._market_id: str | None = None
        self._cycle_targets: tuple[float, float, float, float] | None = None
        self._initial_ch4_equivalent_kg: float | None = None
        self._initial_battery_energy_kwh: float | None = None
        self._events: list[BetaMPCDecisionEvidenceV1] = []

    @property
    def decision_events(self) -> tuple[BetaMPCDecisionEvidenceV1, ...]:
        return tuple(self._events)

    def to_dict(self) -> dict[str, object]:
        body: dict[str, object] = {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "uses_forecast": self.uses_forecast,
            "information_set": self.information_set,
            "settings": self.settings.to_dict(),
            "economic_assumptions": self.assumptions.to_dict(),
            "beta_task": {
                **self.beta_task.to_dict(),
                "sha256": self.beta_task.sha256,
            },
            "failure_policy": "raise typed error; no controller-internal fallback",
            "output_schema": "process-setpoint-command-v1",
            "source_schema": BETA_MPC_DECISION_SCHEMA,
            "model_mismatch_declaration": BETA_MPC_MODEL_MISMATCH_DECLARATION,
        }
        if self.settings.uncertainty_policy == BETA_MPC_UNCERTAINTY_POLICY:
            body["description"] = (
                "Economic MPC with public band-p10 solar, early inventory "
                "buffers and four unchanged hard daily returns."
            )
            body["information_set"] = self.information_set.replace(
                "issued-mean solar", "band-p10 solar"
            )
            body["model_mismatch_declaration"] = self.settings.body()[
                "model_mismatch_declaration"
            ]
        return body

    def descriptor(self) -> dict[str, object]:
        return self.to_dict()

    @staticmethod
    def _finite(name: str, value: object, *, minimum: float | None = None) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        if minimum is not None and result < minimum:
            raise ValueError(f"{name} must be >= {minimum:g}")
        return result

    @property
    def _cycle_interval_count(self) -> int:
        count = int(round(_SECONDS_PER_DAY / self.settings.interval_s))
        if count <= 0 or not math.isclose(
            count * self.settings.interval_s,
            _SECONDS_PER_DAY,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("beta-MPC cadence does not divide the 24 h cycle")
        return count

    def _physics_stride(self, config: PlantConfig) -> int:
        dt_s = self._finite("plant timestep", config.dt_s, minimum=0.0)
        if dt_s <= 0.0:
            raise ValueError("plant timestep must be positive")
        ratio = self.settings.interval_s / dt_s
        stride = int(round(ratio))
        if stride <= 0 or not math.isclose(ratio, stride, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("beta-MPC interval must contain whole physics steps")
        return stride

    def _reconstruct_cycle_return(
        self,
        observation: CyclicMarketControllerObservationV1,
        *,
        horizon: int,
    ) -> BetaMPCCycleReturnV1:
        boundary = observation.plant.state.boundary
        cycle = observation.cycle
        configured = self.beta_task.cycle
        values = (
            (
                "battery",
                "kWh",
                boundary.battery_energy_kwh,
                cycle.battery_margin_kwh,
                configured.battery_tolerance_kwh,
            ),
            (
                "h2",
                "kg",
                boundary.h2_level_kg,
                cycle.h2_margin_kg,
                configured.h2_tolerance_kg,
            ),
            (
                "co2",
                "kg",
                boundary.co2_level_kg,
                cycle.co2_margin_kg,
                configured.co2_tolerance_kg,
            ),
            (
                "ch4",
                "kg",
                boundary.ch4_level_kg,
                cycle.ch4_margin_kg,
                configured.ch4_tolerance_kg,
            ),
        )
        resources: list[BetaMPCResourceReturnV1] = []
        targets: list[float] = []
        for name, unit, raw_current, raw_margin, raw_tolerance in values:
            current = self._finite(f"{name} current", raw_current, minimum=0.0)
            margin = self._finite(f"{name} signed margin", raw_margin)
            tolerance = self._finite(
                f"{name} cycle tolerance", raw_tolerance, minimum=0.0
            )
            target = current - margin
            if target < -_TARGET_ABS_TOLERANCE:
                raise ValueError(f"reconstructed {name} cycle target is negative")
            target = max(0.0, target)
            targets.append(target)
            resources.append(
                BetaMPCResourceReturnV1(
                    resource=name,
                    unit=unit,
                    current=current,
                    signed_margin=margin,
                    target=target,
                    tolerance=tolerance,
                    required_minimum=max(0.0, target - tolerance),
                )
            )
        reconstructed = (targets[0], targets[1], targets[2], targets[3])
        if self._cycle_targets is not None and any(
            not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=_TARGET_ABS_TOLERANCE,
            )
            for actual, expected in zip(reconstructed, self._cycle_targets, strict=True)
        ):
            raise ValueError("reconstructed cycle targets changed within the beta task")
        return BetaMPCCycleReturnV1(
            cycle_index=cycle.cycle_index,
            decision_in_cycle=cycle.decision_in_cycle,
            cycle_decisions=cycle.cycle_decisions,
            issued_at_decision=observation.market.issued_at_decision,
            next_boundary_decision=(observation.market.issued_at_decision + horizon),
            remaining_intervals=horizon,
            resources=(resources[0], resources[1], resources[2], resources[3]),
        )

    def _validate_boundary(
        self,
        observation: CyclicMarketControllerObservationV1,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: ElectrolyserCommitmentProfile | None,
    ) -> tuple[int, BetaMPCCycleReturnV1]:
        if not isinstance(observation, CyclicMarketControllerObservationV1):
            raise TypeError("observation must be CyclicMarketControllerObservationV1")
        if not isinstance(constraints, Constraints):
            raise TypeError("constraints must be Constraints")
        if not isinstance(config, PlantConfig):
            raise TypeError("config must be PlantConfig")
        if not isinstance(commitment_profile, ElectrolyserCommitmentProfile):
            raise TypeError("beta MPC requires an ElectrolyserCommitmentProfile")
        if observation.economic_assumptions_sha256 != self.assumptions.sha256:
            raise ValueError("beta observation economic assumptions digest is stale")
        if observation.beta_task_sha256 != self.beta_task.sha256:
            raise ValueError("beta observation task digest is stale")

        plant = observation.plant
        if plant.forecast is None:
            raise ValueError("beta MPC requires a current public solar forecast")
        if not math.isclose(
            plant.decision_interval_s,
            self.settings.interval_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("controller cadence differs from beta-MPC settings")
        stride = self._physics_stride(config)
        if (
            self._physics_stride_steps is not None
            and self._physics_stride_steps != stride
        ):
            raise ValueError("plant physics stride changed within the beta task")

        market = observation.market
        if market.issued_at_decision != self._decision_count:
            raise ValueError("public market issue is not for the current decision")
        expected_offset_s = self._decision_count * self.settings.interval_s
        if not math.isclose(
            market.issued_at_offset_s,
            expected_offset_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "public market issue offset is not the current task offset"
            )
        if self._market_id is not None and market.market_id != self._market_id:
            raise ValueError("public market identity changed within the beta task")

        cycle = observation.cycle
        cycle_decisions = self._cycle_interval_count
        if cycle.cycle_decisions != cycle_decisions:
            raise ValueError("public cycle length differs from beta-MPC cadence")
        expected_in_cycle = self._decision_count % cycle_decisions
        expected_cycle_index = self._decision_count // cycle_decisions + 1
        if (
            cycle.decision_in_cycle != expected_in_cycle
            or cycle.cycle_index != expected_cycle_index
        ):
            raise ValueError("public cycle counters disagree with the market decision")
        expected_cycle_progress = expected_in_cycle / cycle_decisions
        if not math.isclose(
            cycle.progress,
            expected_cycle_progress,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("public cycle progress disagrees with its counters")
        expected_task_progress = (
            self._decision_count / self.settings.task_interval_count
        )
        if not math.isclose(
            plant.progress,
            expected_task_progress,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("public progress differs from the beta-MPC task horizon")

        boundary = plant.state.boundary
        if (
            isinstance(boundary.step, bool)
            or not isinstance(boundary.step, Integral)
            or boundary.step < 0
        ):
            raise ValueError("public boundary step must be non-negative")
        boundary_time_s = self._finite(
            "public boundary time", boundary.time_s, minimum=0.0
        )
        origin_step = (
            int(boundary.step) if self._origin_step is None else self._origin_step
        )
        origin_time_s = (
            boundary_time_s if self._origin_time_s is None else self._origin_time_s
        )
        expected_step = origin_step + self._decision_count * stride
        expected_time_s = (
            origin_time_s + self._decision_count * self.settings.interval_s
        )
        if boundary.step != expected_step or not math.isclose(
            boundary_time_s,
            expected_time_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("public boundary is not the next beta-MPC decision")
        if plant.forecast.issued_at_physics_boundary_step != boundary.step:
            raise ValueError("public solar forecast is not from the current boundary")
        if not math.isclose(
            plant.forecast.issued_at_offset_s,
            expected_offset_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("public solar forecast offset is not current")

        horizon = cycle_decisions - expected_in_cycle
        if horizon <= 0 or horizon > self.settings.maximum_planning_intervals:
            raise ValueError("beta-MPC horizon cannot reach the next cycle boundary")
        cycle_return = self._reconstruct_cycle_return(
            observation,
            horizon=horizon,
        )
        if self._physics_stride_steps is None:
            self._physics_stride_steps = stride
        if self._market_id is None:
            self._market_id = market.market_id
        if self._origin_step is None:
            self._origin_step = origin_step
            self._origin_time_s = origin_time_s
        if self._cycle_targets is None:
            self._cycle_targets = (
                cycle_return.resources[0].target,
                cycle_return.resources[1].target,
                cycle_return.resources[2].target,
                cycle_return.resources[3].target,
            )
        return horizon, cycle_return

    def compute_command(
        self,
        observation: CyclicMarketControllerObservationV1,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: ElectrolyserCommitmentProfile | None = None,
    ) -> BetaMPCDecisionV1:
        """Solve atomically; a rejected decision cannot alter controller history."""

        anchors = (
            self._decision_count,
            self._origin_step,
            self._origin_time_s,
            self._physics_stride_steps,
            self._market_id,
            self._cycle_targets,
            self._initial_ch4_equivalent_kg,
            self._initial_battery_energy_kwh,
            len(self._events),
        )
        try:
            return self._compute_command(
                observation,
                constraints=constraints,
                config=config,
                commitment_profile=commitment_profile,
            )
        except Exception:
            (
                self._decision_count,
                self._origin_step,
                self._origin_time_s,
                self._physics_stride_steps,
                self._market_id,
                self._cycle_targets,
                self._initial_ch4_equivalent_kg,
                self._initial_battery_energy_kwh,
                event_count,
            ) = anchors
            del self._events[event_count:]
            raise

    def _compute_command(
        self,
        observation: CyclicMarketControllerObservationV1,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: ElectrolyserCommitmentProfile | None = None,
    ) -> BetaMPCDecisionV1:
        """Build one command while the public wrapper owns rollback semantics."""

        horizon, cycle_return = self._validate_boundary(
            observation,
            constraints=constraints,
            config=config,
            commitment_profile=commitment_profile,
        )
        plant = observation.plant
        assert plant.forecast is not None
        physical_sha = public_observation_sha256(plant)
        cycle_sha = beta_cycle_observation_sha256(observation.cycle)
        weather_projection = project_beta_mpc_solar(
            plant.forecast,
            boundary_time_s=plant.state.boundary.time_s,
            config=config,
            settings=self.settings,
        )
        price_projection = project_beta_mpc_prices(
            observation.market,
            interval_s=self.settings.interval_s,
            horizon_intervals=horizon,
        )
        if self._initial_ch4_equivalent_kg is None:
            self._initial_ch4_equivalent_kg = ch4_equivalent_inventory_kg(
                plant.state.boundary,
                config,
            )
            self._initial_battery_energy_kwh = plant.state.boundary.battery_energy_kwh
        assert self._initial_battery_energy_kwh is not None
        model = build_beta_mpc_model(
            observation,
            weather_projection,
            price_projection,
            settings=self.settings,
            assumptions=self.assumptions,
            beta_task=self.beta_task,
            constraints=constraints,
            config=config,
            commitment_profile=commitment_profile,
            initial_ch4_equivalent_kg=self._initial_ch4_equivalent_kg,
            initial_battery_energy_kwh=self._initial_battery_energy_kwh,
        )
        if model.cycle_return != cycle_return:
            raise RuntimeError("controller and model cycle-return derivations disagree")
        plan = solve_beta_mpc_model(model, settings=self.settings)
        first = plan.steps[0]
        targets = ProcessSetpointTargetsV1(
            electrolyser_target_kw=(
                first.electrolyser_startup_kw + first.electrolyser_productive_kw
            ),
            dac_target_kw=first.solar_to_dac_kw,
            methanator_target_kw=first.solar_to_methanator_kw,
            battery_charge_ceiling_kw=first.solar_to_battery_kw,
        )
        evidence = BetaMPCDecisionEvidenceV1(
            decision_index=self._decision_count,
            beta_observation_sha256=observation.sha256,
            physical_observation_sha256=physical_sha,
            market_issue_sha256=observation.market.sha256,
            cycle_observation_sha256=cycle_sha,
            weather_projection_sha256=weather_projection.sha256,
            price_projection_sha256=price_projection.sha256,
            settings_sha256=self.settings.sha256,
            assumptions_sha256=self.assumptions.sha256,
            beta_task_sha256=self.beta_task.sha256,
            cycle_return_sha256=cycle_return.sha256,
            model_sha256=model.sha256,
            plan_sha256=plan.sha256,
            market_id=observation.market.market_id,
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
        result = BetaMPCDecisionV1(evidence=evidence, command=command)
        self._events.append(evidence)
        self._decision_count += 1
        return result


__all__ = ["CyclicMarketEconomicMPCControllerV1"]

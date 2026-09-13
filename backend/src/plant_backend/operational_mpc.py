"""Product MPC with measured restoration targets and explicit guarded fallback.

This controller is not the registered E-beta comparator. It consumes only public
observations and never changes the simulator, guard or research task contracts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, ClassVar

from plant_sim import ElectrolyserCommitmentProfile, ElectrolyserMode, PlantConfig

from .controllers.base import Constraints
from .safety.validation import battery_support_availability
from .rl.v2.beta_config import CyclicMarketTaskConfigV1
from .rl.v2.beta_control import CyclicMarketControllerObservationV1
from .rl.v2.beta_mpc_schema import project_beta_mpc_prices
from .rl.v2.economic_mpc_model import (
    EconomicMPCModelV1,
    _build_economic_mpc_model_from_prices,
)
from .rl.v2.economic_mpc_schema import (
    ECONOMIC_MPC_SOLUTION_DOMAIN,
    EconomicMPCPlanError,
    EconomicMPCPlanV1,
    EconomicMPCSettingsV1,
    EconomicMPCSolveError,
)
from .rl.v2.economic_mpc_solver import (
    _economic_mpc_plan_values_from_solve_result,
    solve_economic_mpc_stages,
)
from .rl.v2.economics import EconomicAssumptionsV1
from .rl.v2.mpc_execution import public_observation_sha256
from .rl.v2.mpc_schema import (
    ServiceMPCSetpointSettingsV1,
    canonical_json,
    domain_sha256,
)
from .rl.v2.operating_mpc_forecast import (
    OPERATING_FORECAST_PROJECTION_VERSION,
    operating_constant_power_rows,
    project_operating_issued_mean_solar,
)
from .rl.v2.setpoint_command import (
    ProcessSetpointCommandV1,
    issue_process_setpoint_command,
)
from .rl.v2.setpoint_control import ProcessSetpointTargetsV1
from .rl.v2.setpoint_controllers import LegacyHeuristicSetpointControllerV1

POLICY_ID = "operational-economic-mpc-v1"
DECISION_SCHEMA = "operational-mpc-decision-v1"
RESOURCE_FIELDS = (
    (
        "battery",
        "Battery",
        "kWh",
        "battery_energy_kwh",
        "battery_energy_end_kwh",
        "battery_margin_kwh",
    ),
    ("h2", "Hydrogen", "kg", "h2_level_kg", "h2_end_kg", "h2_margin_kg"),
    ("co2", "Carbon dioxide", "kg", "co2_level_kg", "co2_end_kg", "co2_margin_kg"),
    ("ch4", "Methane", "kg", "ch4_level_kg", "ch4_end_kg", "ch4_margin_kg"),
)


@dataclass(frozen=True)
class OperatingEvidence:
    plan: EconomicMPCPlanV1 | None
    source_notes: tuple[str, ...]
    payload_json: str

    @property
    def sha256(self) -> str:
        import json

        return domain_sha256(DECISION_SCHEMA, json.loads(self.payload_json))

    def operation(self) -> dict[str, Any]:
        import json

        return dict(json.loads(self.payload_json)["operation"])


@dataclass(frozen=True)
class OperatingDecision:
    evidence: OperatingEvidence
    command: ProcessSetpointCommandV1
    invalid: bool = False


@dataclass(frozen=True, slots=True)
class OperatingMPCPhysicalSettingsV1(ServiceMPCSetpointSettingsV1):
    """Operating-only projection identity; inherited numerical settings unchanged."""

    forecast_projection_version: ClassVar[str] = OPERATING_FORECAST_PROJECTION_VERSION


@dataclass(frozen=True, slots=True)
class OperatingMPCSettingsV1(EconomicMPCSettingsV1):
    @property
    def physical_settings(self) -> OperatingMPCPhysicalSettingsV1:
        return OperatingMPCPhysicalSettingsV1(
            **asdict(super(OperatingMPCSettingsV1, self).physical_settings)
        )

    def body(self) -> dict[str, Any]:
        return {
            **EconomicMPCSettingsV1.body(self),
            "forecast_projection_version": OPERATING_FORECAST_PROJECTION_VERSION,
        }


def solve_economic_mpc_model(
    model: EconomicMPCModelV1, *, settings: OperatingMPCSettingsV1
) -> EconomicMPCPlanV1:
    """Operating settings adapter; retain both existing stage and plan audits.

    The alpha entry point deliberately accepts only exact alpha settings. Use
    its identity-neutral primitives without widening that historical gate.
    """
    if type(model) is not EconomicMPCModelV1:
        raise TypeError("model must be EconomicMPCModelV1")
    if type(settings) is not OperatingMPCSettingsV1:
        raise TypeError("settings must be OperatingMPCSettingsV1")
    solved = solve_economic_mpc_stages(model, settings=settings)
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
        terminal_ch4_equivalent_deficit_kg=values.terminal_ch4_equivalent_deficit_kg,
        terminal_battery_deficit_kwh=values.terminal_battery_deficit_kwh,
    )


class OperationalMPCController:
    id = POLICY_ID

    def __init__(
        self,
        *,
        horizon_h: float,
        interval_s: float,
        assumptions: EconomicAssumptionsV1,
        beta_task: CyclicMarketTaskConfigV1,
    ) -> None:
        self.settings = OperatingMPCSettingsV1(
            task_horizon_h=horizon_h,
            interval_s=interval_s,
            # The active physical safety reserve remains hard in the base model.
            terminal_battery_soc_target=0.0,
        )
        self.assumptions = assumptions.validate()
        self.beta_task = beta_task
        self.reset()

    def reset(self) -> None:
        self._next_decision = 0
        self._reference: tuple[float, ...] | None = None
        self._events: list[OperatingEvidence] = []

    @property
    def decision_events(self) -> tuple[OperatingEvidence, ...]:
        return tuple(self._events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "forecast_statistic": "issued_mean",
            "forecast_projection_version": OPERATING_FORECAST_PROJECTION_VERSION,
            "planning_horizon_h": 24.0,
            "daily_restoration": "measured target; not an optimizer constraint",
            "inventory_accounting": "fixed episode-opening battery and methane-equivalent depletion costs",
            "priority": [
                "physical safety",
                "delivery service",
                "economic cost including inventory recovery",
            ],
            "fallback": "reactive heuristic after a typed solver or plan rejection; retry planning at next boundary",
            "startup_continuation": "preserve active startup power when currently feasible within the operator envelope; minute guard remains authoritative",
            "numerical_settings": self.settings.to_dict(),
            "research_comparator": False,
        }

    def _validate(
        self, observation: CyclicMarketControllerObservationV1, config: PlantConfig
    ) -> tuple[float, ...]:
        if not isinstance(observation, CyclicMarketControllerObservationV1):
            raise TypeError("operating MPC requires a public beta observation")
        plant = observation.plant
        forecast = plant.forecast
        if forecast is None:
            raise ValueError("current public solar forecast is required")
        offset = self._next_decision * self.settings.interval_s
        if (
            observation.market.issued_at_decision != self._next_decision
            or observation.market.issued_at_offset_s != offset
            or forecast.issued_at_offset_s != offset
            or forecast.issued_at_physics_boundary_step != plant.state.boundary.step
            or not math.isclose(
                plant.state.boundary.step * config.dt_s, offset, abs_tol=1e-9
            )
            or plant.decision_interval_s != self.settings.interval_s
            or observation.economic_assumptions_sha256 != self.assumptions.sha256
            or observation.beta_task_sha256 != self.beta_task.sha256
            or not math.isclose(
                plant.progress,
                self._next_decision / self.settings.task_interval_count,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("stale or mismatched operating observation")
        reference = tuple(
            float(getattr(plant.state.boundary, state))
            - float(getattr(observation.cycle, margin))
            for _, _, _, state, _, margin in RESOURCE_FIELDS
        )
        if any(value < -1e-9 for value in reference) or (
            self._reference is not None
            and any(
                not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)
                for a, b in zip(reference, self._reference, strict=True)
            )
        ):
            raise ValueError("opening inventory reference changed")
        return self._reference or tuple(max(0.0, x) for x in reference)

    def build_model(
        self,
        observation: CyclicMarketControllerObservationV1,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: ElectrolyserCommitmentProfile | None,
    ) -> EconomicMPCModelV1:
        reference = self._validate(observation, config)
        horizon = min(
            self.settings.maximum_planning_intervals,
            self.settings.task_interval_count - self._next_decision,
        )
        if horizon <= 0:
            raise ValueError("operating episode is complete")
        forecast = observation.plant.forecast
        assert forecast is not None
        weather = project_operating_issued_mean_solar(
            forecast,
            boundary_time_s=observation.plant.state.boundary.time_s,
            config=config,
            interval_s=self.settings.interval_s,
        )
        prices = project_beta_mpc_prices(
            observation.market,
            interval_s=self.settings.interval_s,
            horizon_intervals=horizon,
        )
        initial_methane = reference[3] + min(
            reference[1] / config.methanator.h2_kg_per_kg_ch4,
            reference[2] / config.methanator.co2_kg_per_kg_ch4,
        )
        model = _build_economic_mpc_model_from_prices(
            observation.plant,
            weather,
            settings=self.settings,
            assumptions=self.assumptions,
            constraints=constraints,
            config=config,
            commitment_profile=commitment_profile,
            horizon_intervals=horizon,
            initial_ch4_equivalent_kg=initial_methane,
            initial_battery_energy_kwh=reference[0],
            interval_prices_eur_per_mwh=prices.prices_eur_per_mwh,
            price_source_id=prices.price_source_id,
            causal_current_dac_health=True,
            mandatory_deliveries=True,
        )
        power_rows, power = operating_constant_power_rows(model, weather, config)
        physical = replace(
            model.physical_model, rows=(*model.physical_model.rows, *power_rows)
        )
        metadata = model.metadata()
        metadata["operating_policy"] = {
            **self.to_dict(),
            "market_issue_sha256": observation.market.sha256,
            "price_projection_sha256": prices.sha256,
            "constant_solar_power_kw": list(power),
        }
        return replace(
            model, physical_model=physical, metadata_json=canonical_json(metadata)
        )

    @staticmethod
    def _continue_active_startup(
        targets: ProcessSetpointTargetsV1,
        observation: CyclicMarketControllerObservationV1,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: ElectrolyserCommitmentProfile | None,
    ) -> tuple[ProcessSetpointTargetsV1, dict[str, Any] | None]:
        snapshot = observation.plant.state.electrolyser_commitment
        if (
            snapshot is None
            or commitment_profile is None
            or snapshot.mode
            not in {ElectrolyserMode.COLD_STARTING, ElectrolyserMode.HOT_STARTING}
        ):
            return targets, None
        required_kw = commitment_profile.start_load_frac * config.electrolyser.rated_kw
        boundary = observation.plant.state.boundary
        limit_kw = constraints.electrolyser_limit_kw(config)
        support = battery_support_availability(
            boundary,
            constraints=constraints,
            config=config,
            electrolyser_headroom_kw=limit_kw,
        )
        available_kw = (
            min(limit_kw, max(0.0, boundary.solar_available_kw) + support.available_kw)
            if support.electrolyser_available
            else 0.0
        )
        requested_kw = targets.electrolyser_target_kw
        if not support.electrolyser_available:
            status = "equipment_offline"
        elif limit_kw < required_kw:
            status = "operator_limit"
        elif available_kw + 1e-9 < required_kw:
            status = "current_power_shortage"
        elif requested_kw < required_kw:
            # A legacy fallback uses the V1 productive floor, which can be below
            # the active startup draw. Retain the held target through startup;
            # the common minute dispatcher still adapts to subsequent weather.
            targets = replace(targets, electrolyser_target_kw=required_kw)
            status = "adjusted"
        else:
            status = "already_sufficient"
        return targets, {
            "status": status,
            "mode": snapshot.mode.value,
            "startup_remaining_s": snapshot.startup_remaining_s,
            "required_kw": required_kw,
            "currently_available_kw": available_kw,
            "requested_kw": requested_kw,
            "issued_kw": targets.electrolyser_target_kw,
        }

    def compute_command(
        self,
        observation: CyclicMarketControllerObservationV1,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: ElectrolyserCommitmentProfile | None = None,
    ) -> OperatingDecision:
        reference = self._validate(observation, config)
        model = self.build_model(
            observation,
            constraints=constraints,
            config=config,
            commitment_profile=commitment_profile,
        )
        plan: EconomicMPCPlanV1 | None = None
        failure: dict[str, Any] | None = None
        invalid = False
        try:
            plan = solve_economic_mpc_model(model, settings=self.settings)
        except (EconomicMPCSolveError, EconomicMPCPlanError) as exc:
            stages = getattr(exc, "stages", ())
            statuses = [stage.status for stage in stages if not stage.success]
            reason = (
                "infeasible"
                if 2 in statuses
                else "time_limit" if 1 in statuses else "plan_rejected"
            )
            failure = {
                "reason": reason,
                "exception_type": type(exc).__name__,
                "stages": [
                    {k: v for k, v in stage.to_dict().items() if k != "message"}
                    for stage in stages
                ],
            }
        if plan is None:
            fallback = LegacyHeuristicSetpointControllerV1().compute_command(
                observation.plant,
                constraints=constraints,
                config=config,
                commitment_profile=commitment_profile,
            )
            targets = fallback.command.targets
            invalid = fallback.invalid
            notes = (
                *fallback.evidence.source_notes,
                "Planner unavailable; guarded heuristic fallback applied for this interval.",
            )
        else:
            first = plan.steps[0]
            targets = ProcessSetpointTargetsV1(
                electrolyser_target_kw=first.electrolyser_startup_kw
                + first.electrolyser_productive_kw,
                dac_target_kw=first.solar_to_dac_kw,
                methanator_target_kw=first.solar_to_methanator_kw,
                battery_charge_ceiling_kw=first.solar_to_battery_kw,
            )
            notes = (
                "Issued mean forecasts; rolling 24-hour plan; inventory depletion remains priced.",
            )
        targets, startup_continuation = self._continue_active_startup(
            targets,
            observation,
            constraints=constraints,
            config=config,
            commitment_profile=commitment_profile,
        )
        if startup_continuation is not None:
            notes = (
                *notes,
                "Active startup continuation: " + startup_continuation["status"] + ".",
            )
        resources: list[dict[str, Any]] = []
        for (name, label, unit, state, end, _), target in zip(
            RESOURCE_FIELDS, reference, strict=True
        ):
            current = float(getattr(observation.plant.state.boundary, state))
            predicted = None if plan is None else float(getattr(plan.steps[-1], end))
            resources.append(
                {
                    "id": name,
                    "label": label,
                    "unit": unit,
                    "reference": target,
                    "current": current,
                    "projected": predicted,
                    "projected_deficit": (
                        None if predicted is None else max(0.0, target - predicted)
                    ),
                }
            )
        deliveries: list[dict[str, Any]] = (
            []
            if plan is None
            else [
                {
                    "due_hours": observation.market.issued_at_offset_s / 3600.0
                    + target.due_offset_s / 3600.0,
                    "requested_kg": target.requested_kg,
                    "predicted_kg": plan.steps[target.due_boundary - 1].delivered_kg,
                    "shortfall_kg": max(
                        0.0,
                        target.requested_kg
                        - plan.steps[target.due_boundary - 1].delivered_kg,
                    ),
                }
                for target in plan.deliveries
            ]
        )
        recovery = any(
            r["reference"] - r["current"] > 1e-6
            or (r["projected_deficit"] or 0.0) > 1e-6
            for r in resources
        ) or any(d["shortfall_kg"] > 1e-6 for d in deliveries)
        operation = {
            "policy_id": self.id,
            "mode": (
                "fallback" if plan is None else "recovery" if recovery else "scheduled"
            ),
            "forecast_statistic": "issued_mean",
            "restoration_policy": "operating_target",
            "issued_at_hours": observation.market.issued_at_offset_s / 3600.0,
            "horizon_hours": model.horizon_intervals
            * self.settings.interval_s
            / 3600.0,
            "resources": resources,
            "deliveries": deliveries,
            "projected_shortfall_kg": (
                None if plan is None else sum(d["shortfall_kg"] for d in deliveries)
            ),
            "fallback_controller": (
                LegacyHeuristicSetpointControllerV1.id if plan is None else None
            ),
            "solver_failure": failure,
            "startup_continuation": startup_continuation,
        }
        evidence = OperatingEvidence(
            plan,
            notes,
            canonical_json(
                {
                    "schema": DECISION_SCHEMA,
                    "policy": self.to_dict(),
                    "observation_sha256": observation.sha256,
                    "model_sha256": model.sha256,
                    "plan_sha256": None if plan is None else plan.sha256,
                    "targets": targets.to_dict(),
                    "operation": operation,
                }
            ),
        )
        command = issue_process_setpoint_command(
            targets,
            decision_index=self._next_decision,
            observation_sha256=public_observation_sha256(observation.plant),
            hold_duration_s=self.settings.interval_s,
            issuer_id=self.id,
            source_schema=DECISION_SCHEMA,
            source_sha256=evidence.sha256,
            constraints=constraints,
            config=config,
        )
        self._reference = reference
        self._next_decision += 1
        self._events.append(evidence)
        return OperatingDecision(evidence, command, invalid)

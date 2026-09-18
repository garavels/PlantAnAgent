"""Versioned public-observation MPC with hard delivery and daily return rows.

Forecast feasibility is evidence about the planning model, never a claim about
realized delivery or daily restoration. A rejected solve raises a typed error;
this controller does not relax mission rows or conceal a fallback policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from typing import Any

from plant_sim import ElectrolyserCommitmentProfile, PlantConfig

from .controllers.base import Constraints
from .operational_mpc import (
    OperationalMPCController,
    RESOURCE_FIELDS,
    solve_economic_mpc_model,
)
from .rl.v2.beta_config import CyclicMarketTaskConfigV1
from .rl.v2.beta_control import CyclicMarketControllerObservationV1
from .rl.v2.economic_mpc_model import EconomicMPCModelV1
from .rl.v2.economic_mpc_schema import EconomicMPCPlanError, EconomicMPCPlanV1
from .rl.v2.economics import EconomicAssumptionsV1
from .rl.v2.mpc_execution import public_observation_sha256
from .rl.v2.mpc_model import MPCSparseRowV1
from .rl.v2.mpc_schema import canonical_json, domain_sha256
from .rl.v2.setpoint_command import (
    ProcessSetpointCommandV1,
    issue_process_setpoint_command,
)
from .rl.v2.setpoint_control import ProcessSetpointTargetsV1


POLICY_ID = "daily-return-mission-mpc-v1"
DECISION_SCHEMA = "daily-return-mission-mpc-decision-v1"
MODEL_VERSION = "hard-delivery-daily-return-mpc-v1"
RESOURCE_GROUPS = (
    "battery_energy_kwh",
    "h2_inventory_kg",
    "co2_inventory_kg",
    "ch4_inventory_kg",
)


@dataclass(frozen=True, slots=True)
class MissionMPCSettingsV1:
    """Additional declared planning reserves, separate from task tolerances."""

    restoration_lead_h: float = 2.0
    battery_buffer_kwh: float = 100.0
    h2_buffer_kg: float = 5.0
    co2_buffer_kg: float = 20.0
    ch4_buffer_kg: float = 10.0
    post_delivery_buffer_kg: float = 1.0
    solver_time_limit_s: float = 5.0
    protected_resource_fraction: float = 1.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.solver_time_limit_s <= 0 or self.restoration_lead_h >= 24:
            raise ValueError("invalid solver budget or restoration lead")
        if self.protected_resource_fraction > 1:
            raise ValueError("protected resource fraction must be at most one")


@dataclass(frozen=True, slots=True)
class MissionMPCEvidenceV1:
    plan: EconomicMPCPlanV1
    payload_json: str

    @property
    def sha256(self) -> str:
        return domain_sha256(DECISION_SCHEMA, self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.payload_json)

    @property
    def source_notes(self) -> tuple[str, ...]:
        return (
            "All deliveries and all four daily returns are hard planning constraints.",
            "Public forecast predictions are not guarantees of simulator outcomes.",
        )


@dataclass(frozen=True, slots=True)
class MissionMPCDecisionV1:
    evidence: MissionMPCEvidenceV1
    command: ProcessSetpointCommandV1
    invalid: bool = False


class MissionMPCControllerV1(OperationalMPCController):
    """Rolling 24 h commitment MPC; only public observations enter planning."""

    id = POLICY_ID

    def __init__(
        self,
        *,
        horizon_h: float,
        interval_s: float,
        assumptions: EconomicAssumptionsV1,
        beta_task: CyclicMarketTaskConfigV1,
        mission_settings: MissionMPCSettingsV1 | None = None,
        electricity_model: dict[str, Any] | None = None,
    ) -> None:
        self.mission_settings = mission_settings or MissionMPCSettingsV1()
        self.electricity_model = json.loads(canonical_json(electricity_model or {}))
        super().__init__(
            horizon_h=horizon_h,
            interval_s=interval_s,
            assumptions=assumptions,
            beta_task=beta_task,
        )
        # Historical operational settings and solver remain untouched. These
        # instance-specific numerical settings are fully included in identity.
        self.settings = replace(
            self.settings,
            solver_time_limit_s=self.mission_settings.solver_time_limit_s,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_version": MODEL_VERSION,
            "planning_horizon_h": self.settings.planning_horizon_h,
            "daily_restoration": "four independent hard rows at the next daily boundary",
            "delivery_service": "every known delivery is a hard row",
            "forecast_statistic": "issued_mean",
            "information_set": "current public plant, cycle, weather issue and market issue",
            "fallback": None,
            "plan_rejection": "raise typed error; no mission-constraint relaxation",
            "execution": "four held targets; normal minute dispatcher and guard",
            "charge_ceiling": "plant maximum to capture otherwise curtailed surplus",
            "intraday_reserves": (
                "protect opening battery and feedstocks; existing deficits "
                "cannot grow in a plan"
            ),
            "mission_settings": asdict(self.mission_settings),
            "numerical_settings": self.settings.to_dict(),
            "electricity_model": self.electricity_model,
        }

    def _cycle_boundary(self, observation: CyclicMarketControllerObservationV1) -> int:
        cycle = observation.cycle
        cycle_h = float(self.beta_task.cycle.cycle_h)
        if not math.isclose(cycle_h, 24.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("mission MPC requires a 24 h daily return cycle")
        count = cycle_h * 3600 / self.settings.interval_s
        if not math.isclose(count, round(count), rel_tol=0, abs_tol=1e-9):
            raise ValueError("daily boundary does not align with the planning grid")
        if cycle.cycle_decisions != round(count):
            raise ValueError("public daily cycle length differs from the task")
        if not 0 <= cycle.decision_in_cycle < cycle.cycle_decisions:
            raise ValueError("public daily boundary has no remaining decision")
        global_decision = (
            (cycle.cycle_index - 1) * cycle.cycle_decisions + cycle.decision_in_cycle
        )
        if global_decision != self._next_decision:
            raise ValueError("public cycle counters and controller clock differ")
        remaining = cycle.cycle_decisions - cycle.decision_in_cycle
        horizon = min(
            self.settings.maximum_planning_intervals,
            self.settings.task_interval_count - self._next_decision,
        )
        if remaining > horizon:
            raise ValueError("daily boundary lies beyond the supported public horizon")
        return remaining

    def build_model(
        self,
        observation: CyclicMarketControllerObservationV1,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: ElectrolyserCommitmentProfile | None,
    ) -> EconomicMPCModelV1:
        reference = self._validate(observation, config)
        boundary = self._cycle_boundary(observation)
        model = super().build_model(
            observation,
            constraints=constraints,
            config=config,
            commitment_profile=commitment_profile,
        )
        tuning = self.mission_settings
        cycle_config = self.beta_task.cycle
        tolerances = (
            cycle_config.battery_tolerance_kwh,
            cycle_config.h2_tolerance_kg,
            cycle_config.co2_tolerance_kg,
            cycle_config.ch4_tolerance_kg,
        )
        buffers = (
            tuning.battery_buffer_kwh,
            tuning.h2_buffer_kg,
            tuning.co2_buffer_kg,
            tuning.ch4_buffer_kg,
        )
        capacities = (
            config.battery.capacity_kwh,
            config.h2_tank.capacity_kg,
            config.co2_tank.capacity_kg,
            config.ch4_tank.capacity_kg,
        )
        lower = list(model.physical_model.variable_lower)
        rows: list[MPCSparseRowV1] = []

        def floor(name: str, index: int, minimum: float) -> None:
            lower[index] = max(lower[index], minimum)
            rows.append(MPCSparseRowV1(name, minimum, None, ((index, 1.0),)))

        now_h = observation.plant.state.boundary.time_s / 3600
        # Daylight geometry is public plant configuration, not realized weather.
        daylight = [
            i + 1
            for i in range(boundary)
            if config.solar.sunrise_h
            < (now_h + (i + 0.5) * self.settings.interval_s / 3600) % 24
            < config.solar.sunset_h
        ]
        lead = math.ceil(tuning.restoration_lead_h * 3600 / self.settings.interval_s)
        deadline = max(0, max(daylight, default=0) - lead)
        requirements = []
        for spec, group_name, target, tolerance, buffer, capacity in zip(
            RESOURCE_FIELDS,
            RESOURCE_GROUPS,
            reference,
            tolerances,
            buffers,
            capacities,
            strict=True,
        ):
            resource, _, _, state_field, plan_field, _ = spec
            current = float(getattr(observation.plant.state.boundary, state_field))
            minimum = max(0.0, target - tolerance)
            if minimum > capacity:
                raise ValueError(f"daily {resource} target exceeds capacity")
            floor(
                f"mission_daily_return_{resource}[{boundary}]",
                model.group(group_name)[boundary],
                minimum,
            )
            # With zero marginal PV price, an otherwise equivalent optimizer
            # schedule can turn opening CO2/H2 into methane and defer refill
            # until the last forecast-feasible moment. Protect those reserves
            # throughout the lookahead so a later cloud or outage has less
            # opportunity to strand a large feedstock-replenishment obligation.
            # This is an explicitly stronger planning policy, not task physics.
            if resource != "ch4":
                protected = min(
                    current, tuning.protected_resource_fraction * minimum
                )
                for point in range(1, model.horizon_intervals + 1):
                    floor(
                        f"mission_protected_{resource}[{point}]",
                        model.group(group_name)[point],
                        protected,
                    )
            requested = min(buffer, max(0.0, capacity - minimum))
            retained = (
                requested
                if deadline
                else min(requested, max(0.0, current - minimum))
            )
            for point in range(max(1, deadline), boundary + 1):
                floor(
                    f"mission_early_return_{resource}[{point}]",
                    model.group(group_name)[point],
                    minimum + retained,
                )
            requirements.append(
                {
                    "resource": resource,
                    "boundary": boundary,
                    "target": target,
                    "task_tolerance": tolerance,
                    "minimum": minimum,
                    "planning_buffer": retained,
                    "plan_field": plan_field,
                }
            )
        for delivery in model.deliveries:
            point = delivery.due_boundary
            floor(
                f"mission_delivery[{point}]",
                model.group("delivered_kg")[point - 1],
                delivery.requested_kg,
            )
            floor(
                f"mission_post_delivery_stock[{point}]",
                model.group("ch4_inventory_kg")[point],
                tuning.post_delivery_buffer_kg,
            )
        physical = replace(
            model.physical_model,
            variable_lower=tuple(lower),
            rows=(*model.physical_model.rows, *rows),
        )
        metadata = model.metadata()
        metadata["mission"] = {
            "model_version": MODEL_VERSION,
            "cycle_index": observation.cycle.cycle_index,
            "daily_boundary": boundary,
            "restoration_boundary": deadline,
            "resources": requirements,
            "hard_row_names": [row.name for row in rows],
            "uncertainty_claim": "planning reserves; no guarantee of realized weather or exact execution",
        }
        return replace(
            model, physical_model=physical, metadata_json=canonical_json(metadata)
        )

    def compute_command(
        self,
        observation: CyclicMarketControllerObservationV1,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: ElectrolyserCommitmentProfile | None = None,
    ) -> MissionMPCDecisionV1:
        reference = self._validate(observation, config)
        model = self.build_model(
            observation,
            constraints=constraints,
            config=config,
            commitment_profile=commitment_profile,
        )
        # Both solve stages and the selected complete vector are audited against
        # the augmented model. Errors propagate before advancing controller state.
        plan = solve_economic_mpc_model(model, settings=self.settings)
        mission = model.metadata()["mission"]
        cycle_step = plan.steps[mission["daily_boundary"] - 1]
        resources = []
        for requirement in mission["resources"]:
            predicted = float(getattr(cycle_step, requirement["plan_field"]))
            if predicted < requirement["minimum"] - 1e-7:
                raise EconomicMPCPlanError("selected plan fails the daily mission audit")
            resources.append({**requirement, "predicted": predicted})
        for delivery in model.deliveries:
            if (
                plan.steps[delivery.due_boundary - 1].delivered_kg
                < delivery.requested_kg - 1e-7
            ):
                raise EconomicMPCPlanError("selected plan fails a hard mission delivery")
        first = plan.steps[0]
        targets = ProcessSetpointTargetsV1(
            electrolyser_target_kw=(
                first.electrolyser_startup_kw + first.electrolyser_productive_kw
            ),
            dac_target_kw=first.solar_to_dac_kw,
            methanator_target_kw=first.solar_to_methanator_kw,
            battery_charge_ceiling_kw=config.battery.max_charge_kw,
        )
        evidence = MissionMPCEvidenceV1(
            plan,
            canonical_json(
                {
                    "schema": DECISION_SCHEMA,
                    "controller": self.to_dict(),
                    "observation_sha256": observation.sha256,
                    "model_sha256": model.sha256,
                    "plan_sha256": plan.sha256,
                    "decision": self._next_decision,
                    "audited_hard_return_plan": True,
                    "realized_mission_passed": None,
                    "daily_boundary": mission["daily_boundary"],
                    "restoration_boundary": mission["restoration_boundary"],
                    "resources": resources,
                    "targets": targets.to_dict(),
                    "plan": plan.to_dict(),
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
        return MissionMPCDecisionV1(evidence, command)

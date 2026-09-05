"""Solver-independent beta MPC built from causal public observations only."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Sequence

from plant_sim import ElectrolyserCommitmentProfile, PlantConfig

from ...controllers.base import Constraints
from .beta_config import CyclicMarketTaskConfigV1
from .beta_control import CyclicMarketControllerObservationV1
from .beta_mpc_schema import (
    BETA_MPC_DAC_AVAILABILITY_ASSUMPTION,
    BETA_MPC_MODEL_MISMATCH_DECLARATION,
    BETA_MPC_MODEL_VERSION,
    BetaMPCCycleReturnV1,
    BetaMPCInputError,
    BetaMPCPriceProjectionV1,
    BetaMPCResourceReturnV1,
    BetaMPCSettingsV1,
    beta_cycle_observation_sha256,
    project_beta_mpc_prices,
)
from .economic_mpc_model import (
    EconomicMPCModelV1,
    _build_economic_mpc_model_from_prices,
)
from .economic_mpc_schema import EconomicMPCInputError
from .economics import EconomicAssumptionsV1
from .mpc_forecast import MPCForecastTrajectoryV1
from .mpc_model import MPCSolutionAuditV1, MPCSparseRowV1, ServiceMPCModelV1
from .mpc_schema import (
    MODEL_BALANCE_TOLERANCE,
    MPCDeliveryTargetV1,
    ServiceMPCInputError,
    canonical_json,
    domain_sha256,
    require_sha256,
)


BETA_MPC_MODEL_DOMAIN = "cyclic-market-beta-mpc-sparse-model-v1"
_RETURN_ROW_NAMES = (
    "beta_cycle_return_battery",
    "beta_cycle_return_h2",
    "beta_cycle_return_co2",
    "beta_cycle_return_ch4",
)
_RESOURCE_GROUPS = {
    "battery": "battery_energy_kwh",
    "h2": "h2_inventory_kg",
    "co2": "co2_inventory_kg",
    "ch4": "ch4_inventory_kg",
}


def _finite(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}")
    return result


def _validated_cycle_geometry(
    observation: CyclicMarketControllerObservationV1,
    *,
    settings: BetaMPCSettingsV1,
    beta_task: CyclicMarketTaskConfigV1,
) -> tuple[int, int]:
    cycle = observation.cycle
    config = beta_task.cycle
    if not math.isclose(float(config.cycle_h), 24.0, rel_tol=0.0, abs_tol=1e-12):
        raise BetaMPCInputError("beta MPC requires an exact 24 h cycle")
    expected_cycle_decisions_float = config.cycle_h * 3_600.0 / settings.interval_s
    if not math.isclose(
        expected_cycle_decisions_float,
        round(expected_cycle_decisions_float),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise BetaMPCInputError("cycle does not align to the beta-MPC interval")
    expected_cycle_decisions = int(round(expected_cycle_decisions_float))
    if cycle.cycle_decisions != expected_cycle_decisions:
        raise BetaMPCInputError(
            "public cycle length disagrees with the declared beta task"
        )
    if cycle.decision_in_cycle >= cycle.cycle_decisions:
        raise BetaMPCInputError("cycle is already settled; no live decision remains")
    remaining = cycle.cycle_decisions - cycle.decision_in_cycle
    if remaining > settings.maximum_planning_intervals:
        raise BetaMPCInputError(
            "next cycle boundary exceeds beta-MPC public planning support"
        )
    expected_global_decision = (
        cycle.cycle_index - 1
    ) * cycle.cycle_decisions + cycle.decision_in_cycle
    if observation.market.issued_at_decision != expected_global_decision:
        raise BetaMPCInputError(
            "current public market issue and cycle counters are not aligned"
        )
    return remaining, expected_global_decision


def derive_beta_mpc_cycle_return(
    observation: CyclicMarketControllerObservationV1,
    *,
    settings: BetaMPCSettingsV1,
    beta_task: CyclicMarketTaskConfigV1,
    config: PlantConfig,
) -> BetaMPCCycleReturnV1:
    """Reconstruct the four immutable targets from public levels and margins."""

    if not isinstance(observation, CyclicMarketControllerObservationV1):
        raise TypeError("observation must be CyclicMarketControllerObservationV1")
    if not isinstance(settings, BetaMPCSettingsV1):
        raise TypeError("settings must be BetaMPCSettingsV1")
    if not isinstance(beta_task, CyclicMarketTaskConfigV1):
        raise TypeError("beta_task must be CyclicMarketTaskConfigV1")
    if not isinstance(config, PlantConfig):
        raise TypeError("config must be PlantConfig")
    if observation.beta_task_sha256 != beta_task.sha256:
        raise BetaMPCInputError(
            "beta observation task digest differs from the controller declaration"
        )
    remaining, issued_decision = _validated_cycle_geometry(
        observation,
        settings=settings,
        beta_task=beta_task,
    )
    cycle_config = beta_task.cycle
    for name in (
        "battery_return_fraction",
        "h2_return_fraction",
        "co2_return_fraction",
        "ch4_return_fraction",
    ):
        value = _finite(name, getattr(cycle_config, name), minimum=0.0)
        if value > 1.0:
            raise BetaMPCInputError(f"{name} must be <= 1")

    boundary = observation.plant.state.boundary
    current = {
        "battery": _finite(
            "current battery energy", boundary.battery_energy_kwh, minimum=0.0
        ),
        "h2": _finite("current H2 inventory", boundary.h2_level_kg, minimum=0.0),
        "co2": _finite("current CO2 inventory", boundary.co2_level_kg, minimum=0.0),
        "ch4": _finite("current CH4 inventory", boundary.ch4_level_kg, minimum=0.0),
    }
    margins = {
        "battery": observation.cycle.battery_margin_kwh,
        "h2": observation.cycle.h2_margin_kg,
        "co2": observation.cycle.co2_margin_kg,
        "ch4": observation.cycle.ch4_margin_kg,
    }
    tolerances = {
        "battery": cycle_config.battery_tolerance_kwh,
        "h2": cycle_config.h2_tolerance_kg,
        "co2": cycle_config.co2_tolerance_kg,
        "ch4": cycle_config.ch4_tolerance_kg,
    }
    capacities = {
        "battery": _finite(
            "battery capacity", boundary.battery_capacity_kwh, minimum=0.0
        ),
        "h2": _finite("H2 capacity", boundary.h2_capacity_kg, minimum=0.0),
        "co2": _finite("CO2 capacity", boundary.co2_capacity_kg, minimum=0.0),
        "ch4": _finite("CH4 capacity", boundary.ch4_capacity_kg, minimum=0.0),
    }
    units = {"battery": "kWh", "h2": "kg", "co2": "kg", "ch4": "kg"}
    records: list[BetaMPCResourceReturnV1] = []
    for resource in ("battery", "h2", "co2", "ch4"):
        margin = _finite(f"{resource} public margin", margins[resource])
        target = current[resource] - margin
        if -1e-9 <= target < 0.0:
            target = 0.0
        if target < 0.0 or target > capacities[resource] + 1e-9:
            raise BetaMPCInputError(
                f"reconstructed {resource} target lies outside public capacity"
            )
        tolerance = _finite(
            f"{resource} cycle tolerance", tolerances[resource], minimum=0.0
        )
        records.append(
            BetaMPCResourceReturnV1(
                resource=resource,
                unit=units[resource],
                current=current[resource],
                signed_margin=margin,
                target=target,
                tolerance=tolerance,
                required_minimum=max(0.0, target - tolerance),
            )
        )
    cycle = observation.cycle
    return BetaMPCCycleReturnV1(
        cycle_index=cycle.cycle_index,
        decision_in_cycle=cycle.decision_in_cycle,
        cycle_decisions=cycle.cycle_decisions,
        issued_at_decision=issued_decision,
        next_boundary_decision=issued_decision + remaining,
        remaining_intervals=remaining,
        resources=(records[0], records[1], records[2], records[3]),
    )


@dataclass(frozen=True, slots=True)
class BetaMPCModelV1:
    """Economic MILP plus four immutable hard next-cycle return rows."""

    economic_model: EconomicMPCModelV1
    settings_sha256: str
    beta_task_sha256: str
    market_issue_sha256: str
    cycle_observation_sha256: str
    price_projection: BetaMPCPriceProjectionV1
    cycle_return: BetaMPCCycleReturnV1
    return_rows: tuple[
        MPCSparseRowV1,
        MPCSparseRowV1,
        MPCSparseRowV1,
        MPCSparseRowV1,
    ]
    metadata_json: str

    model_version = BETA_MPC_MODEL_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.economic_model, EconomicMPCModelV1):
            raise TypeError("economic_model must be EconomicMPCModelV1")
        for name in (
            "settings_sha256",
            "beta_task_sha256",
            "market_issue_sha256",
            "cycle_observation_sha256",
        ):
            require_sha256(name, getattr(self, name))
        if self.economic_model.settings_sha256 != self.settings_sha256:
            raise ValueError("beta and economic model settings digests differ")
        if not isinstance(self.price_projection, BetaMPCPriceProjectionV1):
            raise TypeError("price_projection must be BetaMPCPriceProjectionV1")
        if self.price_projection.market_issue_sha256 != self.market_issue_sha256:
            raise ValueError("price projection is bound to another market issue")
        if self.price_projection.prices_eur_per_mwh != (
            self.economic_model.interval_prices_eur_per_mwh
        ):
            raise ValueError("economic objective prices differ from public projection")
        if self.economic_model.tariff_id != self.price_projection.price_source_id:
            raise ValueError("economic price source id differs from public projection")
        if not isinstance(self.cycle_return, BetaMPCCycleReturnV1):
            raise TypeError("cycle_return must be BetaMPCCycleReturnV1")
        if self.horizon_intervals != self.cycle_return.remaining_intervals:
            raise ValueError("beta model does not end at the next cycle boundary")
        if self.price_projection.horizon_intervals != self.horizon_intervals:
            raise ValueError("price projection does not span the beta model horizon")
        if not isinstance(self.return_rows, tuple) or len(self.return_rows) != 4:
            raise TypeError("beta return rows must be an immutable four-tuple")
        if any(not isinstance(row, MPCSparseRowV1) for row in self.return_rows):
            raise TypeError("beta return rows must be MPCSparseRowV1 records")
        if tuple(row.name for row in self.return_rows) != _RETURN_ROW_NAMES:
            raise ValueError("beta return rows are missing or out of canonical order")
        for row, resource in zip(
            self.return_rows, self.cycle_return.resources, strict=True
        ):
            expected_index = self.group(_RESOURCE_GROUPS[resource.resource])[-1]
            if row.coefficients != ((expected_index, 1.0),):
                raise ValueError(
                    f"{resource.resource} return row does not select terminal inventory"
                )
            if (
                row.lower is None
                or row.upper is not None
                or not math.isclose(
                    row.lower,
                    resource.required_minimum,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    f"{resource.resource} return row has an invalid hard lower bound"
                )
        if len({row.name for row in self.rows}) != len(self.rows):
            raise ValueError("beta model row names must remain unique")
        if not isinstance(self.metadata_json, str):
            raise TypeError("metadata_json must be text")
        try:
            metadata = json.loads(self.metadata_json)
        except json.JSONDecodeError as exc:
            raise ValueError("metadata_json is invalid") from exc
        if (
            not isinstance(metadata, dict)
            or canonical_json(metadata) != self.metadata_json
        ):
            raise ValueError("metadata_json must be a canonical strict JSON object")

    @property
    def physical_model(self) -> ServiceMPCModelV1:
        return self.economic_model.physical_model

    @property
    def assumptions_sha256(self) -> str:
        return self.economic_model.assumptions_sha256

    @property
    def interval_s(self) -> float:
        return self.economic_model.interval_s

    @property
    def horizon_intervals(self) -> int:
        return self.economic_model.horizon_intervals

    @property
    def forecast_sha256(self) -> str:
        return self.economic_model.forecast_sha256

    @property
    def deliveries(self) -> tuple[MPCDeliveryTargetV1, ...]:
        return self.economic_model.deliveries

    @property
    def variable_names(self) -> tuple[str, ...]:
        return self.economic_model.variable_names

    @property
    def variable_lower(self) -> tuple[float, ...]:
        return self.economic_model.variable_lower

    @property
    def variable_upper(self) -> tuple[float, ...]:
        return self.economic_model.variable_upper

    @property
    def integrality(self) -> tuple[int, ...]:
        return self.economic_model.integrality

    @property
    def primary_objective(self) -> tuple[float, ...]:
        return self.economic_model.primary_objective

    @property
    def economic_objective(self) -> tuple[float, ...]:
        return self.economic_model.economic_objective

    @property
    def groups(self) -> tuple[Any, ...]:
        return self.economic_model.groups

    @property
    def rows(self) -> tuple[MPCSparseRowV1, ...]:
        return (*self.economic_model.rows, *self.return_rows)

    @property
    def variable_count(self) -> int:
        return self.economic_model.variable_count

    @property
    def binary_variable_count(self) -> int:
        return self.economic_model.binary_variable_count

    @property
    def row_count(self) -> int:
        return self.economic_model.row_count + len(self.return_rows)

    @property
    def nonzero_count(self) -> int:
        return self.economic_model.nonzero_count + len(self.return_rows)

    def group(self, name: str) -> tuple[int, ...]:
        return self.economic_model.group(name)

    def metadata(self) -> dict[str, Any]:
        value = json.loads(self.metadata_json)
        if not isinstance(value, dict):  # pragma: no cover - guarded at construction
            raise TypeError("beta model metadata is not an object")
        return value

    def body(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "economic_model_sha256": self.economic_model.sha256,
            "settings_sha256": self.settings_sha256,
            "assumptions_sha256": self.assumptions_sha256,
            "beta_task_sha256": self.beta_task_sha256,
            "weather_projection_sha256": self.forecast_sha256,
            "market_issue_sha256": self.market_issue_sha256,
            "cycle_observation_sha256": self.cycle_observation_sha256,
            "price_projection": self.price_projection.to_dict(),
            "cycle_return": self.cycle_return.to_dict(),
            "return_rows": [row.to_dict() for row in self.return_rows],
            "metadata": self.metadata(),
        }

    @property
    def sha256(self) -> str:
        return domain_sha256(BETA_MPC_MODEL_DOMAIN, self.body())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.body(),
            "dimensions": {
                "variables": self.variable_count,
                "binary_variables": self.binary_variable_count,
                "rows": self.row_count,
                "nonzeros": self.nonzero_count,
            },
            "sha256": self.sha256,
        }

    def audit_solution(
        self,
        solution: Sequence[float],
        *,
        tolerance: float = MODEL_BALANCE_TOLERANCE,
        extra_rows: Sequence[MPCSparseRowV1] = (),
    ) -> MPCSolutionAuditV1:
        """Audit base physics, economics, all returns, and caller-supplied rows."""

        return self.economic_model.audit_solution(
            solution,
            tolerance=tolerance,
            extra_rows=(*self.return_rows, *tuple(extra_rows)),
        )

    def predicted_cost_components(
        self,
        solution: Sequence[float],
    ) -> tuple[tuple[str, float], ...]:
        return self.economic_model.predicted_cost_components(solution)


def build_beta_mpc_model(
    observation: CyclicMarketControllerObservationV1,
    weather_projection: MPCForecastTrajectoryV1,
    price_projection: BetaMPCPriceProjectionV1,
    *,
    settings: BetaMPCSettingsV1,
    assumptions: EconomicAssumptionsV1,
    beta_task: CyclicMarketTaskConfigV1,
    constraints: Constraints,
    config: PlantConfig,
    commitment_profile: ElectrolyserCommitmentProfile | None,
    initial_ch4_equivalent_kg: float,
    initial_battery_energy_kwh: float,
) -> BetaMPCModelV1:
    """Build a hard-return model from the current public beta information set."""

    if not isinstance(observation, CyclicMarketControllerObservationV1):
        raise TypeError("observation must be CyclicMarketControllerObservationV1")
    if not isinstance(weather_projection, MPCForecastTrajectoryV1):
        raise TypeError("weather_projection must be MPCForecastTrajectoryV1")
    if not isinstance(price_projection, BetaMPCPriceProjectionV1):
        raise TypeError("price_projection must be BetaMPCPriceProjectionV1")
    if not isinstance(settings, BetaMPCSettingsV1):
        raise TypeError("settings must be BetaMPCSettingsV1")
    if not isinstance(assumptions, EconomicAssumptionsV1):
        raise TypeError("assumptions must be EconomicAssumptionsV1")
    if not isinstance(beta_task, CyclicMarketTaskConfigV1):
        raise TypeError("beta_task must be CyclicMarketTaskConfigV1")
    if not isinstance(constraints, Constraints):
        raise TypeError("constraints must be Constraints")
    if not isinstance(config, PlantConfig):
        raise TypeError("config must be PlantConfig")
    if observation.economic_assumptions_sha256 != assumptions.sha256:
        raise BetaMPCInputError("beta observation economic assumptions digest is stale")
    if observation.beta_task_sha256 != beta_task.sha256:
        raise BetaMPCInputError("beta observation task digest is stale")
    try:
        beta_task.market.validate()
    except (TypeError, ValueError) as exc:
        raise BetaMPCInputError(f"invalid beta market declaration: {exc}") from exc
    if not math.isclose(
        observation.plant.decision_interval_s,
        settings.interval_s,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise BetaMPCInputError("controller cadence differs from beta-MPC settings")
    if config.dt_s <= 0.0 or not math.isclose(
        settings.interval_s / config.dt_s,
        round(settings.interval_s / config.dt_s),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise BetaMPCInputError(
            "beta-MPC interval must contain whole physics timesteps"
        )
    cycle_return = derive_beta_mpc_cycle_return(
        observation,
        settings=settings,
        beta_task=beta_task,
        config=config,
    )
    horizon = cycle_return.remaining_intervals
    expected_price_projection = project_beta_mpc_prices(
        observation.market,
        interval_s=settings.interval_s,
        horizon_intervals=horizon,
    )
    if price_projection != expected_price_projection:
        raise BetaMPCInputError(
            "price projection is not the canonical path from the current public issue"
        )
    stride = int(round(settings.interval_s / config.dt_s))
    expected_physics_step = observation.market.issued_at_decision * stride
    if observation.plant.state.boundary.step != expected_physics_step:
        raise BetaMPCInputError(
            "physical boundary and current public market issue are not aligned"
        )
    expected_progress = (
        observation.market.issued_at_decision / settings.task_interval_count
    )
    if not math.isclose(
        observation.plant.progress,
        expected_progress,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise BetaMPCInputError(
            "public task progress differs from beta-MPC task_horizon_h"
        )
    try:
        economic = _build_economic_mpc_model_from_prices(
            observation.plant,
            weather_projection,
            settings=settings,
            assumptions=assumptions,
            constraints=constraints,
            config=config,
            commitment_profile=commitment_profile,
            horizon_intervals=horizon,
            initial_ch4_equivalent_kg=initial_ch4_equivalent_kg,
            initial_battery_energy_kwh=initial_battery_energy_kwh,
            interval_prices_eur_per_mwh=price_projection.prices_eur_per_mwh,
            price_source_id=price_projection.price_source_id,
            causal_current_dac_health=True,
        )
    except (EconomicMPCInputError, ServiceMPCInputError) as exc:
        raise BetaMPCInputError(str(exc)) from exc

    return_rows = tuple(
        MPCSparseRowV1(
            name=f"beta_cycle_return_{resource.resource}",
            lower=resource.required_minimum,
            upper=None,
            coefficients=(
                (economic.group(_RESOURCE_GROUPS[resource.resource])[-1], 1.0),
            ),
        )
        for resource in cycle_return.resources
    )
    if len(return_rows) != 4:  # pragma: no cover - fixed typed cardinality
        raise RuntimeError("beta cycle return lost resource cardinality")
    boundary_health = observation.plant.state.boundary.health
    dac_health = boundary_health.get("dac")
    if dac_health is None:  # The reused physical model also checks this.
        raise BetaMPCInputError("missing public health for dac")
    metadata = {
        "formulation_scope": (
            "service-first causal economic MPC with four separate hard resource "
            "returns at the exact next cycle boundary"
        ),
        "information_policy": {
            "weather": "current public issued-mean projection only",
            "market": price_projection.projection_version,
            "market_quantiles": "p10/p90 bound through issue digest but unused",
            "future_weather_vintages": False,
            "future_market_vintages": False,
            "future_market_truth": False,
            "future_disturbance_schedule": False,
            "private_regime": False,
            "random_seed": False,
        },
        "dac_availability": {
            "assumption": BETA_MPC_DAC_AVAILABILITY_ASSUMPTION,
            "current_public_status": str(dac_health.status),
            "current_public_offline": bool(dac_health.offline),
            "current_public_degradation": float(dac_health.degradation),
        },
        "hard_cycle_rows": list(_RETURN_ROW_NAMES),
        "hard_cycle_row_slack_variables": 0,
        "failure_policy": "typed failure; no controller-internal fallback",
        "model_mismatch_declaration": BETA_MPC_MODEL_MISMATCH_DECLARATION,
    }
    typed_rows = (
        return_rows[0],
        return_rows[1],
        return_rows[2],
        return_rows[3],
    )
    return BetaMPCModelV1(
        economic_model=economic,
        settings_sha256=settings.sha256,
        beta_task_sha256=beta_task.sha256,
        market_issue_sha256=observation.market.sha256,
        cycle_observation_sha256=beta_cycle_observation_sha256(observation.cycle),
        price_projection=price_projection,
        cycle_return=cycle_return,
        return_rows=typed_rows,
        metadata_json=canonical_json(metadata),
    )


__all__ = [
    "BETA_MPC_MODEL_DOMAIN",
    "BetaMPCModelV1",
    "build_beta_mpc_model",
    "derive_beta_mpc_cycle_return",
]

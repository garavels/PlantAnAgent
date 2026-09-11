"""Solver-independent tariff-aware extension of the proven service MILP."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Protocol, Sequence, runtime_checkable

from plant_sim import ElectrolyserCommitmentProfile, PlantConfig

from ...controllers.base import Constraints
from .control import ControllerObservationV2
from .economics import EconomicAssumptionsV1, EconomicControllerObservationV1
from .mpc_forecast import MPCForecastTrajectoryV1
from .mpc_model import (
    MPCSolutionAuditV1,
    MPCSparseRowV1,
    MPCVariableGroupV1,
    ServiceMPCModelV1,
    build_service_mpc_model,
)
from .mpc_schema import (
    MODEL_BALANCE_TOLERANCE,
    MPCDeliveryTargetV1,
    canonical_json,
    domain_sha256,
)
from .economic_mpc_schema import (
    ECONOMIC_MPC_MODEL_MISMATCH_DECLARATION,
    ECONOMIC_MPC_MODEL_VERSION,
    EconomicMPCInputError,
    EconomicMPCSettingsV1,
)


ECONOMIC_MPC_MODEL_DOMAIN = "economic-mpc-sparse-model-v1"
_HINGE_GROUPS = (
    "terminal_ch4_equivalent_deficit_kg",
    "terminal_battery_deficit_kwh",
)


def _finite(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}")
    return result


def _economic_prices(
    observation: EconomicControllerObservationV1,
    assumptions: EconomicAssumptionsV1,
    *,
    interval_s: float,
    horizon: int,
) -> tuple[float, ...]:
    if observation.economic_assumptions_sha256 != assumptions.sha256:
        raise EconomicMPCInputError(
            "economic observation assumptions digest differs from the controller"
        )
    if observation.tariff_id != assumptions.tariff.id:
        raise EconomicMPCInputError(
            "economic observation tariff id differs from the declared schedule"
        )
    boundary_h = observation.plant.state.boundary.time_h
    expected_summary = assumptions.tariff.observe(boundary_h)
    if observation.tariff != expected_summary:
        raise EconomicMPCInputError(
            "economic observation tariff summary does not match the full declared "
            "day-ahead schedule at the current boundary"
        )
    dt_h = interval_s / 3_600.0
    return tuple(
        assumptions.tariff.price_at(boundary_h + index * dt_h)
        for index in range(horizon)
    )


@dataclass(frozen=True, slots=True)
class EconomicMPCModelV1:
    """Physical service model plus monetary objective and endowment hinges."""

    physical_model: ServiceMPCModelV1
    settings_sha256: str
    assumptions_sha256: str
    tariff_id: str
    interval_prices_eur_per_mwh: tuple[float, ...]
    initial_ch4_equivalent_kg: float
    initial_battery_energy_kwh: float
    economic_objective: tuple[float, ...]
    hinge_groups: tuple[MPCVariableGroupV1, MPCVariableGroupV1]
    hinge_rows: tuple[MPCSparseRowV1, MPCSparseRowV1]
    metadata_json: str

    model_version = ECONOMIC_MPC_MODEL_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.physical_model, ServiceMPCModelV1):
            raise TypeError("physical_model must be ServiceMPCModelV1")
        for name in ("settings_sha256", "assumptions_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError(f"{name} must be a SHA-256 digest") from exc
        if not isinstance(self.tariff_id, str) or not self.tariff_id.strip():
            raise ValueError("tariff_id must be non-empty")
        if (
            not isinstance(self.interval_prices_eur_per_mwh, tuple)
            or len(self.interval_prices_eur_per_mwh) != self.horizon_intervals
        ):
            raise ValueError("interval prices must cover the model horizon")
        for index, value in enumerate(self.interval_prices_eur_per_mwh):
            _finite(f"interval price[{index}]", value)
        _finite(
            "initial_ch4_equivalent_kg",
            self.initial_ch4_equivalent_kg,
            minimum=0.0,
        )
        _finite(
            "initial_battery_energy_kwh",
            self.initial_battery_energy_kwh,
            minimum=0.0,
        )
        if len(self.economic_objective) != self.variable_count:
            raise ValueError("economic objective does not span all model variables")
        for index, value in enumerate(self.economic_objective):
            _finite(f"economic objective[{index}]", value, minimum=0.0)
        if tuple(group.name for group in self.hinge_groups) != _HINGE_GROUPS:
            raise ValueError("economic hinge groups are missing or out of order")
        expected_hinge_indices = tuple(
            range(self.physical_model.variable_count, self.variable_count)
        )
        if tuple(index for group in self.hinge_groups for index in group.indices) != (
            expected_hinge_indices
        ):
            raise ValueError("economic hinge groups do not cover appended variables")
        if tuple(row.name for row in self.hinge_rows) != (
            "terminal_ch4_endowment_hinge",
            "terminal_battery_endowment_hinge",
        ):
            raise ValueError("economic endowment rows are missing or out of order")
        if not isinstance(self.metadata_json, str):
            raise TypeError("metadata_json must be text")
        import json

        try:
            metadata = json.loads(self.metadata_json)
        except json.JSONDecodeError as exc:
            raise ValueError("metadata_json is invalid") from exc
        if canonical_json(metadata) != self.metadata_json:
            raise ValueError("metadata_json must be canonical strict JSON")

    @property
    def interval_s(self) -> float:
        return self.physical_model.interval_s

    @property
    def horizon_intervals(self) -> int:
        return self.physical_model.horizon_intervals

    @property
    def forecast_sha256(self) -> str:
        return self.physical_model.forecast_sha256

    @property
    def deliveries(self) -> tuple[MPCDeliveryTargetV1, ...]:
        return self.physical_model.deliveries

    @property
    def variable_names(self) -> tuple[str, ...]:
        return (
            *self.physical_model.variable_names,
            "terminal_ch4_equivalent_deficit_kg",
            "terminal_battery_deficit_kwh",
        )

    @property
    def variable_lower(self) -> tuple[float, ...]:
        return (*self.physical_model.variable_lower, 0.0, 0.0)

    @property
    def variable_upper(self) -> tuple[float, ...]:
        return (
            *self.physical_model.variable_upper,
            self.initial_ch4_equivalent_kg,
            self.initial_battery_energy_kwh,
        )

    @property
    def integrality(self) -> tuple[int, ...]:
        return (*self.physical_model.integrality, 0, 0)

    @property
    def primary_objective(self) -> tuple[float, ...]:
        return (*self.physical_model.primary_objective, 0.0, 0.0)

    @property
    def groups(self) -> tuple[MPCVariableGroupV1, ...]:
        return (*self.physical_model.groups, *self.hinge_groups)

    @property
    def rows(self) -> tuple[MPCSparseRowV1, ...]:
        return (*self.physical_model.rows, *self.hinge_rows)

    @property
    def variable_count(self) -> int:
        return self.physical_model.variable_count + 2

    @property
    def binary_variable_count(self) -> int:
        return self.physical_model.binary_variable_count

    @property
    def row_count(self) -> int:
        return self.physical_model.row_count + 2

    @property
    def nonzero_count(self) -> int:
        return self.physical_model.nonzero_count + sum(
            len(row.coefficients) for row in self.hinge_rows
        )

    def group(self, name: str) -> tuple[int, ...]:
        for group in self.groups:
            if group.name == name:
                return group.indices
        raise KeyError(name)

    def metadata(self) -> dict[str, Any]:
        import json

        value = json.loads(self.metadata_json)
        if not isinstance(value, dict):
            raise TypeError("economic model metadata is not an object")
        return value

    def body(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "physical_model_sha256": self.physical_model.sha256,
            "settings_sha256": self.settings_sha256,
            "assumptions_sha256": self.assumptions_sha256,
            "tariff_id": self.tariff_id,
            "interval_prices_eur_per_mwh": list(self.interval_prices_eur_per_mwh),
            "initial_endowment": {
                "ch4_equivalent_kg": float(self.initial_ch4_equivalent_kg),
                "battery_energy_kwh": float(self.initial_battery_energy_kwh),
            },
            "economic_objective": list(self.economic_objective),
            "hinge_groups": [group.to_dict() for group in self.hinge_groups],
            "hinge_rows": [row.to_dict() for row in self.hinge_rows],
            "metadata": self.metadata(),
        }

    @property
    def sha256(self) -> str:
        return domain_sha256(ECONOMIC_MPC_MODEL_DOMAIN, self.body())

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
        """Independently recompute every bound, integrality, and sparse row."""

        tolerance = _finite("solution tolerance", tolerance, minimum=0.0)
        if tolerance <= 0.0:
            raise ValueError("solution tolerance must be positive")
        values = tuple(
            _finite(f"solution[{index}]", value) for index, value in enumerate(solution)
        )
        if len(values) != self.variable_count:
            raise ValueError("solution length does not match economic model")
        max_bound = 0.0
        max_integrality = 0.0
        valid = True
        for value, lower, upper, integer in zip(
            values,
            self.variable_lower,
            self.variable_upper,
            self.integrality,
            strict=True,
        ):
            violation = max(0.0, lower - value, value - upper)
            max_bound = max(max_bound, violation)
            scale = max(1.0, abs(value), abs(lower), abs(upper))
            valid = valid and violation <= tolerance + tolerance * scale
            if integer:
                max_integrality = max(max_integrality, abs(value - round(value)))
        max_row = 0.0
        worst_row: str | None = None
        rows = (*self.rows, *tuple(extra_rows))
        if len({row.name for row in rows}) != len(rows):
            raise ValueError("solution audit row names must be unique")
        for row in rows:
            terms = tuple(
                coefficient * values[index] for index, coefficient in row.coefficients
            )
            activity = math.fsum(terms)
            violation = max(
                0.0 if row.lower is None else row.lower - activity,
                0.0 if row.upper is None else activity - row.upper,
            )
            if violation > max_row:
                max_row = violation
                worst_row = row.name
            finite_sides = tuple(
                abs(value) for value in (row.lower, row.upper) if value is not None
            )
            scale = max(
                1.0,
                math.fsum(
                    abs(coefficient) * max(1.0, abs(values[index]))
                    for index, coefficient in row.coefficients
                ),
                *finite_sides,
            )
            valid = valid and violation <= tolerance + tolerance * scale
        valid = valid and max_integrality <= tolerance
        return MPCSolutionAuditV1(
            valid=valid,
            max_bound_violation=max_bound,
            max_row_violation=max_row,
            max_integrality_violation=max_integrality,
            worst_row=worst_row,
        )

    def predicted_cost_components(
        self,
        solution: Sequence[float],
    ) -> tuple[tuple[str, float], ...]:
        values = tuple(float(value) for value in solution)
        if len(values) != self.variable_count:
            raise ValueError("solution length does not match economic model")
        metadata = self.metadata()["cost_coefficients"]
        if not isinstance(metadata, dict):
            raise TypeError("cost coefficient metadata is malformed")
        dt_h = self.interval_s / 3_600.0
        accepted_groups = (
            "solar_to_battery_kw",
            "solar_to_electrolyser_kw",
            "solar_to_dac_kw",
            "solar_to_methanator_kw",
        )
        solar_cost = math.fsum(
            values[index] * dt_h * max(0.0, price) / 1_000.0
            for time, price in enumerate(self.interval_prices_eur_per_mwh)
            for group_name in accepted_groups
            for index in (self.group(group_name)[time],)
        )
        throughput_rate = float(metadata["battery_throughput_eur_per_kwh"])
        battery_cost = math.fsum(
            (values[charge] + values[support]) * dt_h * throughput_rate
            for charge, support in zip(
                self.group("solar_to_battery_kw"),
                self.group("battery_support_kw"),
                strict=True,
            )
        )
        reserve_per_h = float(metadata["electrolyser_reserve_eur_per_h"])
        start_h = float(metadata["cold_start_equivalent_operating_h"])
        wear_cost = math.fsum(
            values[index] * dt_h * reserve_per_h for index in self.group("running")
        ) + math.fsum(
            values[index] * start_h * reserve_per_h
            for index in self.group("cold_start")
        )
        methane_cost = values[self.group(_HINGE_GROUPS[0])[0]] * float(
            metadata["methane_endowment_eur_per_kg"]
        )
        battery_endowment_cost = values[self.group(_HINGE_GROUPS[1])[0]] * float(
            metadata["battery_endowment_eur_per_kwh"]
        )
        return (
            ("solar_opportunity_cost", solar_cost),
            ("battery_throughput_reserve", battery_cost),
            ("electrolyser_life_reserve", wear_cost),
            ("methane_endowment_liability", methane_cost),
            ("battery_endowment_liability", battery_endowment_cost),
        )


@runtime_checkable
class EconomicMPCModelProtocol(Protocol):
    """Structural contract consumed by the shared two-stage economic solver.

    Separately versioned tasks may add hard rows and their own model identity while
    reusing the numerical solver.  The protocol deliberately contains no tariff
    field, so a causal market model need not masquerade as the alpha tariff model.
    """

    @property
    def physical_model(self) -> ServiceMPCModelV1: ...

    @property
    def settings_sha256(self) -> str: ...

    @property
    def assumptions_sha256(self) -> str: ...

    @property
    def economic_objective(self) -> tuple[float, ...]: ...

    @property
    def interval_s(self) -> float: ...

    @property
    def horizon_intervals(self) -> int: ...

    @property
    def forecast_sha256(self) -> str: ...

    @property
    def deliveries(self) -> tuple[MPCDeliveryTargetV1, ...]: ...

    @property
    def variable_lower(self) -> tuple[float, ...]: ...

    @property
    def variable_upper(self) -> tuple[float, ...]: ...

    @property
    def integrality(self) -> tuple[int, ...]: ...

    @property
    def primary_objective(self) -> tuple[float, ...]: ...

    @property
    def rows(self) -> tuple[MPCSparseRowV1, ...]: ...

    @property
    def variable_count(self) -> int: ...

    @property
    def sha256(self) -> str: ...

    def group(self, name: str) -> tuple[int, ...]: ...

    def audit_solution(
        self,
        solution: Sequence[float],
        *,
        tolerance: float = MODEL_BALANCE_TOLERANCE,
        extra_rows: Sequence[MPCSparseRowV1] = (),
    ) -> MPCSolutionAuditV1: ...

    def predicted_cost_components(
        self,
        solution: Sequence[float],
    ) -> tuple[tuple[str, float], ...]: ...


def build_economic_mpc_model(
    observation: EconomicControllerObservationV1,
    forecast: MPCForecastTrajectoryV1,
    *,
    settings: EconomicMPCSettingsV1,
    assumptions: EconomicAssumptionsV1,
    constraints: Constraints,
    config: PlantConfig,
    commitment_profile: ElectrolyserCommitmentProfile | None,
    horizon_intervals: int,
    initial_ch4_equivalent_kg: float,
    initial_battery_energy_kwh: float,
) -> EconomicMPCModelV1:
    """Build a service-first model from only declared public information."""

    if not isinstance(observation, EconomicControllerObservationV1):
        raise TypeError("observation must be EconomicControllerObservationV1")
    if type(settings) is not EconomicMPCSettingsV1:
        raise TypeError("settings must be EconomicMPCSettingsV1")
    if not isinstance(assumptions, EconomicAssumptionsV1):
        raise TypeError("assumptions must be EconomicAssumptionsV1")
    prices = _economic_prices(
        observation,
        assumptions,
        interval_s=settings.interval_s,
        horizon=horizon_intervals,
    )
    return _build_economic_mpc_model_from_prices(
        observation.plant,
        forecast,
        settings=settings,
        assumptions=assumptions,
        constraints=constraints,
        config=config,
        commitment_profile=commitment_profile,
        horizon_intervals=horizon_intervals,
        initial_ch4_equivalent_kg=initial_ch4_equivalent_kg,
        initial_battery_energy_kwh=initial_battery_energy_kwh,
        interval_prices_eur_per_mwh=prices,
        price_source_id=assumptions.tariff.id,
    )


def _build_economic_mpc_model_from_prices(
    observation: ControllerObservationV2,
    forecast: MPCForecastTrajectoryV1,
    *,
    settings: EconomicMPCSettingsV1,
    assumptions: EconomicAssumptionsV1,
    constraints: Constraints,
    config: PlantConfig,
    commitment_profile: ElectrolyserCommitmentProfile | None,
    horizon_intervals: int,
    initial_ch4_equivalent_kg: float,
    initial_battery_energy_kwh: float,
    interval_prices_eur_per_mwh: Sequence[float],
    price_source_id: str,
    causal_current_dac_health: bool = False,
    mandatory_deliveries: bool = False,
) -> EconomicMPCModelV1:
    """Build the proven economic model from an explicitly supplied public price path.

    The alpha builder above remains responsible for validating its full tariff.  This
    lower-level seam lets separately identified environments supply their own causal
    public price issue without pretending that it is the alpha tariff.
    """

    if not isinstance(observation, ControllerObservationV2):
        raise TypeError("observation must be ControllerObservationV2")
    if not isinstance(settings, EconomicMPCSettingsV1):
        raise TypeError("settings must be EconomicMPCSettingsV1")
    if not isinstance(assumptions, EconomicAssumptionsV1):
        raise TypeError("assumptions must be EconomicAssumptionsV1")
    assumptions.validate()
    if not isinstance(price_source_id, str) or not price_source_id.strip():
        raise ValueError("price_source_id must be non-empty")
    initial_ch4 = _finite(
        "initial_ch4_equivalent_kg", initial_ch4_equivalent_kg, minimum=0.0
    )
    initial_battery = _finite(
        "initial_battery_energy_kwh", initial_battery_energy_kwh, minimum=0.0
    )
    physical = build_service_mpc_model(
        observation,
        forecast,
        settings=settings.physical_settings,
        constraints=constraints,
        config=config,
        commitment_profile=commitment_profile,
        horizon_intervals=horizon_intervals,
        causal_current_dac_health=causal_current_dac_health,
        mandatory_deliveries=mandatory_deliveries,
    )
    prices = tuple(
        _finite(f"interval price[{index}]", value)
        for index, value in enumerate(interval_prices_eur_per_mwh)
    )
    if len(prices) != physical.horizon_intervals:
        raise ValueError("interval prices must cover the model horizon")
    first_hinge = physical.variable_count
    methane_hinge = first_hinge
    battery_hinge = first_hinge + 1
    hinge_groups = (
        MPCVariableGroupV1(name=_HINGE_GROUPS[0], indices=(methane_hinge,)),
        MPCVariableGroupV1(name=_HINGE_GROUPS[1], indices=(battery_hinge,)),
    )
    terminal_ch4 = physical.group("ch4_inventory_kg")[-1]
    terminal_feedstock = physical.group("terminal_feedstock_ch4_equivalent_kg")[0]
    terminal_battery = physical.group("battery_energy_kwh")[-1]
    hinge_rows = (
        MPCSparseRowV1(
            name="terminal_ch4_endowment_hinge",
            lower=initial_ch4,
            upper=None,
            coefficients=tuple(
                sorted(
                    (
                        (terminal_ch4, 1.0),
                        (terminal_feedstock, 1.0),
                        (methane_hinge, 1.0),
                    )
                )
            ),
        ),
        MPCSparseRowV1(
            name="terminal_battery_endowment_hinge",
            lower=initial_battery,
            upper=None,
            coefficients=tuple(sorted(((terminal_battery, 1.0), (battery_hinge, 1.0)))),
        ),
    )

    dt_h = settings.interval_s / 3_600.0
    objective = [0.0] * (physical.variable_count + 2)
    accepted_groups = (
        "solar_to_battery_kw",
        "solar_to_electrolyser_kw",
        "solar_to_dac_kw",
        "solar_to_methanator_kw",
    )
    for time, raw_price in enumerate(prices):
        coefficient = dt_h * max(0.0, raw_price) / 1_000.0
        for group_name in accepted_groups:
            objective[physical.group(group_name)[time]] += coefficient
    throughput_rate = assumptions.battery_throughput_eur_per_kwh
    for group_name in ("solar_to_battery_kw", "battery_support_kw"):
        for index in physical.group(group_name):
            objective[index] += dt_h * throughput_rate
    reserve_per_h = (
        config.electrolyser.rated_kw
        * assumptions.electrolyser_capital_eur_per_kw
        / assumptions.electrolyser_life_h
    )
    for index in physical.group("running"):
        objective[index] += dt_h * reserve_per_h
    for index in physical.group("cold_start"):
        objective[index] += (
            assumptions.cold_start_equivalent_operating_h * reserve_per_h
        )
    objective[methane_hinge] = (
        assumptions.methane_endowment_value_eur_per_kg_ch4_equivalent
    )
    objective[battery_hinge] = assumptions.battery_endowment_value_eur_per_kwh
    metadata = {
        "objective_scope": (
            "stage 1 maximizes service; stage 2 minimizes predicted EUR costs "
            "without monetizing safety or relaxing the delivery lock"
        ),
        "accepted_solar_definition": (
            "dt_h times the sum of solar routed to battery, electrolyser, DAC, "
            "and methanator; curtailed solar is excluded"
        ),
        "battery_throughput_definition": (
            "dt_h times bus-side solar charging plus battery support"
        ),
        "negative_tariff_semantics": "opportunity price is floored at zero",
        "terminal_deficit_semantics": (
            "exact non-negative hinges against episode-opening endowments; no "
            "objective credit for terminal surplus"
        ),
        "model_mismatch_declaration": ECONOMIC_MPC_MODEL_MISMATCH_DECLARATION,
        "cost_coefficients": {
            "battery_throughput_eur_per_kwh": throughput_rate,
            "electrolyser_reserve_eur_per_h": reserve_per_h,
            "cold_start_equivalent_operating_h": (
                assumptions.cold_start_equivalent_operating_h
            ),
            "methane_endowment_eur_per_kg": (
                assumptions.methane_endowment_value_eur_per_kg_ch4_equivalent
            ),
            "battery_endowment_eur_per_kwh": (
                assumptions.battery_endowment_value_eur_per_kwh
            ),
        },
    }
    return EconomicMPCModelV1(
        physical_model=physical,
        settings_sha256=settings.sha256,
        assumptions_sha256=assumptions.sha256,
        tariff_id=price_source_id,
        interval_prices_eur_per_mwh=prices,
        initial_ch4_equivalent_kg=initial_ch4,
        initial_battery_energy_kwh=initial_battery,
        economic_objective=tuple(objective),
        hinge_groups=hinge_groups,
        hinge_rows=hinge_rows,
        metadata_json=canonical_json(metadata),
    )


def economic_delivery_lock_row(
    model: EconomicMPCModelProtocol,
    *,
    delivered_optimum_kg: float,
    tolerance_kg: float,
) -> MPCSparseRowV1:
    optimum = _finite("delivered_optimum_kg", delivered_optimum_kg, minimum=0.0)
    tolerance = _finite("tolerance_kg", tolerance_kg, minimum=0.0)
    if tolerance <= 0.0:
        raise ValueError("delivery lock tolerance must be positive")
    return MPCSparseRowV1(
        name="economic_primary_delivery_lock",
        lower=max(0.0, optimum - tolerance),
        upper=None,
        coefficients=tuple((index, 1.0) for index in model.group("delivered_kg")),
    )


__all__ = [
    "ECONOMIC_MPC_MODEL_DOMAIN",
    "EconomicMPCModelProtocol",
    "EconomicMPCModelV1",
    "build_economic_mpc_model",
    "economic_delivery_lock_row",
]

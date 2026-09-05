"""Versioned public contracts for the cyclic-market beta MPC.

The beta controller is not an alias for the alpha economic MPC.  It reuses the
validated physical and monetary plan records, but has its own information policy,
model identity, hard cycle-return contract, and command evidence.  In particular,
the price path is derived from exactly one public market issue: the already-known
current price is used for the opening interval and that issue's hourly ensemble
means are used thereafter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, ClassVar

from .beta_config import CYCLE_FEASIBILITY_EPSILON
from .economic_mpc_schema import (
    ECONOMIC_MPC_SOLVER,
    EconomicMPCPlanV1,
    EconomicMPCSettingsV1,
    EconomicMPCStageSolveV1,
)
from .market import PublicPriceForecastIssueV1
from .mpc_schema import domain_sha256, require_sha256
from .setpoint_command import ProcessSetpointCommandV1
from .setpoint_control import ProcessSetpointTargetsV1


BETA_MPC_SCHEMA_VERSION = 1
BETA_MPC_CONTROLLER_ID = "cyclic-market-economic-mpc-v1"
BETA_MPC_MODEL_VERSION = "cyclic-return-economic-mpc-v1"
BETA_MPC_OBJECTIVE_VERSION = "cycle-feasible-service-then-eur-v1"
BETA_MPC_PRICE_PROJECTION_VERSION = "current-then-issued-hourly-mean-v1"
BETA_MPC_SETTINGS_DOMAIN = "cyclic-market-beta-mpc-settings-v1"
BETA_MPC_PRICE_PROJECTION_DOMAIN = "cyclic-market-beta-mpc-price-projection-v1"
BETA_MPC_CYCLE_OBSERVATION_DOMAIN = "cyclic-market-beta-cycle-observation-v1"
BETA_MPC_CYCLE_RETURN_DOMAIN = "cyclic-market-beta-cycle-return-v1"
BETA_MPC_PLAN_DOMAIN = "cyclic-market-beta-mpc-plan-v1"
BETA_MPC_SOLUTION_DOMAIN = "cyclic-market-beta-mpc-solution-vector-v1"
BETA_MPC_DECISION_SCHEMA = "cyclic-market-beta-mpc-decision-evidence-v1"
BETA_MPC_DECISION_DOMAIN = BETA_MPC_DECISION_SCHEMA
BETA_MPC_DAC_AVAILABILITY_ASSUMPTION = (
    "current-interval-public-dac-health-then-nominal-v1"
)
BETA_MPC_MODEL_MISMATCH_DECLARATION = (
    "The current market price is exact for interval zero; later prices use only "
    "the current public issue's hourly means. Public p10/p90 values are hash-bound "
    "but unused by this deterministic baseline. Solar uses the current issued-mean "
    "weather projection. Currently observed DAC unavailability is enforced in the "
    "opening interval and nominal availability is assumed thereafter; no future "
    "outage schedule, recovery time, seed, future forecast vintage, or private "
    "market regime is used. "
    "Minute-scale commitment, nonlinear response, guard intervention, and forecast "
    "error remain measured realization/model-mismatch terms."
)

_RESOURCE_ORDER = ("battery", "h2", "co2", "ch4")
_RESOURCE_UNITS = {
    "battery": "kWh",
    "h2": "kg",
    "co2": "kg",
    "ch4": "kg",
}


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


def _index(name: str, value: object, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    lower = 1 if positive else 0
    if result < lower:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


@dataclass(frozen=True, slots=True)
class BetaMPCSettingsV1(EconomicMPCSettingsV1):
    """Complete immutable tuning surface for the beta-native MPC."""

    task_horizon_h: float = 168.0
    planning_horizon_h: float = 24.0

    schema_version: ClassVar[int] = BETA_MPC_SCHEMA_VERSION
    controller_id: ClassVar[str] = BETA_MPC_CONTROLLER_ID
    model_version: ClassVar[str] = BETA_MPC_MODEL_VERSION
    objective_version: ClassVar[str] = BETA_MPC_OBJECTIVE_VERSION
    solver: ClassVar[str] = ECONOMIC_MPC_SOLVER
    price_projection_version: ClassVar[str] = BETA_MPC_PRICE_PROJECTION_VERSION
    dac_availability_assumption: ClassVar[str] = BETA_MPC_DAC_AVAILABILITY_ASSUMPTION

    def __post_init__(self) -> None:
        EconomicMPCSettingsV1.__post_init__(self)
        if not math.isclose(
            float(self.planning_horizon_h), 24.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "beta MPC requires exactly 24 h of planning support at cycle opening"
            )
        cycles = float(self.task_horizon_h) / 24.0
        if not math.isclose(cycles, round(cycles), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("beta-MPC task_horizon_h must contain whole 24 h cycles")
        subdivisions = 3_600.0 / float(self.interval_s)
        if not math.isclose(
            subdivisions, round(subdivisions), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                "beta-MPC interval_s must be a whole subdivision of one hour"
            )

    def body(self) -> dict[str, Any]:
        body = EconomicMPCSettingsV1.body(self)
        body.update(
            {
                "schema_version": self.schema_version,
                "controller_id": self.controller_id,
                "model_version": self.model_version,
                "objective_version": self.objective_version,
                "price_projection_version": self.price_projection_version,
                "weather_forecast_statistic": "current public issued mean",
                "market_forecast_statistic": (
                    "current exact interval, then current issue hourly mean"
                ),
                "market_forecast_quantiles": "visible and bound, unused",
                "planning_horizon_semantics": (
                    "exactly the remaining controller intervals to the next 24 h "
                    "cycle settlement"
                ),
                "cycle_return_semantics": (
                    "four independent hard terminal lower bounds: battery, H2, "
                    "CO2, and CH4; no aggregate substitution and no slack"
                ),
                "dac_availability_assumption": self.dac_availability_assumption,
                "controller_fallback": None,
                "lexicographic_order": [
                    "enforce safety and all four hard cycle returns",
                    "maximize delivered service",
                    "minimize predicted EUR cost under the delivery lock",
                ],
                "model_mismatch_declaration": BETA_MPC_MODEL_MISMATCH_DECLARATION,
            }
        )
        return body

    @property
    def sha256(self) -> str:
        return domain_sha256(BETA_MPC_SETTINGS_DOMAIN, self.body())


DEFAULT_BETA_MPC_SETTINGS_V1 = BetaMPCSettingsV1()


@dataclass(frozen=True, slots=True)
class BetaMPCPriceProjectionV1:
    """Causal controller-grid price path derived from one public issue."""

    market_issue_sha256: str
    market_id: str
    issued_at_decision: int
    issued_at_offset_s: float
    interval_s: float
    prices_eur_per_mwh: tuple[float, ...]

    projection_version: ClassVar[str] = BETA_MPC_PRICE_PROJECTION_VERSION

    def __post_init__(self) -> None:
        require_sha256("market_issue_sha256", self.market_issue_sha256)
        if not isinstance(self.market_id, str) or not self.market_id.strip():
            raise ValueError("market_id must be non-empty")
        decision = _index("issued_at_decision", self.issued_at_decision)
        offset = _finite("issued_at_offset_s", self.issued_at_offset_s, minimum=0.0)
        interval = _finite("interval_s", self.interval_s, minimum=0.0)
        if interval <= 0.0:
            raise ValueError("price projection interval must be positive")
        if not math.isclose(offset, decision * interval, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "price issue offset disagrees with its decision and controller grid"
            )
        if not isinstance(self.prices_eur_per_mwh, tuple) or not (
            self.prices_eur_per_mwh
        ):
            raise ValueError("price projection must be a non-empty immutable tuple")
        for index, value in enumerate(self.prices_eur_per_mwh):
            _finite(f"prices_eur_per_mwh[{index}]", value)
        if len(self.prices_eur_per_mwh) * interval > 24.0 * 3_600.0 + 1e-9:
            raise ValueError("price projection exceeds the public 24 h horizon")

    @property
    def horizon_intervals(self) -> int:
        return len(self.prices_eur_per_mwh)

    @property
    def price_source_id(self) -> str:
        """Opaque source identity retained inside the reused economic primitive."""

        return (
            f"{self.projection_version}:{self.market_id}:" f"{self.market_issue_sha256}"
        )

    def body(self) -> dict[str, Any]:
        return {
            "projection_version": self.projection_version,
            "source_information": "one current public market issue only",
            "market_issue_sha256": self.market_issue_sha256,
            "market_id": self.market_id,
            "issued_at_decision": int(self.issued_at_decision),
            "issued_at_offset_s": float(self.issued_at_offset_s),
            "interval_s": float(self.interval_s),
            "interval_zero_statistic": "current_eur_per_mwh",
            "later_interval_statistic": "issued hourly ensemble mean",
            "p10_p90_semantics": "bound through market_issue_sha256 but unused",
            "prices_eur_per_mwh": list(self.prices_eur_per_mwh),
        }

    @property
    def sha256(self) -> str:
        return domain_sha256(BETA_MPC_PRICE_PROJECTION_DOMAIN, self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


def project_beta_mpc_prices(
    issue: PublicPriceForecastIssueV1,
    *,
    interval_s: float,
    horizon_intervals: int,
) -> BetaMPCPriceProjectionV1:
    """Project one public issue without consulting later truth or vintages."""

    if not isinstance(issue, PublicPriceForecastIssueV1):
        raise TypeError("issue must be PublicPriceForecastIssueV1")
    horizon = _index("horizon_intervals", horizon_intervals, positive=True)
    means = issue.project_mean_prices(
        interval_s=interval_s,
        horizon_intervals=horizon,
    )
    prices = (float(issue.current_eur_per_mwh), *means[1:])
    return BetaMPCPriceProjectionV1(
        market_issue_sha256=issue.sha256,
        market_id=issue.market_id,
        issued_at_decision=issue.issued_at_decision,
        issued_at_offset_s=issue.issued_at_offset_s,
        interval_s=float(interval_s),
        prices_eur_per_mwh=prices,
    )


def beta_cycle_observation_sha256(cycle: object) -> str:
    """Hash a typed cycle observation without importing private task state."""

    from .beta_control import CycleObservationV1

    if not isinstance(cycle, CycleObservationV1):
        raise TypeError("cycle must be CycleObservationV1")
    return domain_sha256(BETA_MPC_CYCLE_OBSERVATION_DOMAIN, cycle.to_dict())


@dataclass(frozen=True, slots=True)
class BetaMPCResourceReturnV1:
    """One independently enforced resource target at the next cycle boundary."""

    resource: str
    unit: str
    current: float
    signed_margin: float
    target: float
    tolerance: float
    required_minimum: float

    def __post_init__(self) -> None:
        if self.resource not in _RESOURCE_ORDER:
            raise ValueError("beta-MPC return resource is unsupported")
        if self.unit != _RESOURCE_UNITS[self.resource]:
            raise ValueError("beta-MPC return resource unit is invalid")
        current = _finite(f"{self.resource}.current", self.current, minimum=0.0)
        margin = _finite(f"{self.resource}.signed_margin", self.signed_margin)
        target = _finite(f"{self.resource}.target", self.target, minimum=0.0)
        tolerance = _finite(f"{self.resource}.tolerance", self.tolerance, minimum=0.0)
        required = _finite(
            f"{self.resource}.required_minimum",
            self.required_minimum,
            minimum=0.0,
        )
        if not math.isclose(current - margin, target, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"{self.resource} target is not reconstructed from current minus margin"
            )
        expected_required = max(0.0, target - tolerance)
        if not math.isclose(required, expected_required, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"{self.resource} required minimum disagrees with settlement tolerance"
            )

    def to_dict(self) -> dict[str, str | float]:
        return {
            "resource": self.resource,
            "unit": self.unit,
            "current": float(self.current),
            "signed_margin": float(self.signed_margin),
            "target": float(self.target),
            "tolerance": float(self.tolerance),
            "required_minimum": float(self.required_minimum),
        }


@dataclass(frozen=True, slots=True)
class BetaMPCCycleReturnV1:
    """Complete next-boundary hard-return contract reconstructed from public data."""

    cycle_index: int
    decision_in_cycle: int
    cycle_decisions: int
    issued_at_decision: int
    next_boundary_decision: int
    remaining_intervals: int
    resources: tuple[
        BetaMPCResourceReturnV1,
        BetaMPCResourceReturnV1,
        BetaMPCResourceReturnV1,
        BetaMPCResourceReturnV1,
    ]

    def __post_init__(self) -> None:
        cycle_index = _index("cycle_index", self.cycle_index, positive=True)
        decision_in_cycle = _index("decision_in_cycle", self.decision_in_cycle)
        cycle_decisions = _index("cycle_decisions", self.cycle_decisions, positive=True)
        issued = _index("issued_at_decision", self.issued_at_decision)
        boundary = _index(
            "next_boundary_decision", self.next_boundary_decision, positive=True
        )
        remaining = _index(
            "remaining_intervals", self.remaining_intervals, positive=True
        )
        if decision_in_cycle >= cycle_decisions:
            raise ValueError("cycle settlement has no remaining live decision")
        if remaining != cycle_decisions - decision_in_cycle:
            raise ValueError(
                "remaining intervals do not end at the next cycle boundary"
            )
        expected_issued = (cycle_index - 1) * cycle_decisions + decision_in_cycle
        if issued != expected_issued:
            raise ValueError("market decision and cycle counters are not aligned")
        if boundary != issued + remaining or boundary != cycle_index * cycle_decisions:
            raise ValueError("next cycle boundary decision is inconsistent")
        if not isinstance(self.resources, tuple) or len(self.resources) != 4:
            raise TypeError("cycle return resources must be an immutable four-tuple")
        if any(
            not isinstance(item, BetaMPCResourceReturnV1) for item in self.resources
        ):
            raise TypeError("cycle return resources must be typed records")
        if tuple(item.resource for item in self.resources) != _RESOURCE_ORDER:
            raise ValueError("cycle return resources must use canonical order")

    def resource(self, name: str) -> BetaMPCResourceReturnV1:
        for item in self.resources:
            if item.resource == name:
                return item
        raise KeyError(name)

    def body(self) -> dict[str, Any]:
        return {
            "cycle_index": int(self.cycle_index),
            "decision_in_cycle": int(self.decision_in_cycle),
            "cycle_decisions": int(self.cycle_decisions),
            "issued_at_decision": int(self.issued_at_decision),
            "next_boundary_decision": int(self.next_boundary_decision),
            "remaining_intervals": int(self.remaining_intervals),
            "resources": {
                item.resource: {
                    key: value
                    for key, value in item.to_dict().items()
                    if key != "resource"
                }
                for item in self.resources
            },
            "constraint_semantics": (
                "four separate hard lower bounds at the exact next cycle boundary"
            ),
        }

    @property
    def sha256(self) -> str:
        return domain_sha256(BETA_MPC_CYCLE_RETURN_DOMAIN, self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class BetaMPCPlanV1(EconomicMPCPlanV1):
    """Selected beta plan, republished without an alpha controller identity."""

    price_projection: BetaMPCPriceProjectionV1
    market_issue_sha256: str
    cycle_observation_sha256: str
    beta_task_sha256: str
    cycle_return: BetaMPCCycleReturnV1
    selected_solution_vector: tuple[float, ...]

    def __post_init__(self) -> None:
        EconomicMPCPlanV1.__post_init__(self)
        if not isinstance(self.price_projection, BetaMPCPriceProjectionV1):
            raise TypeError("price_projection must be BetaMPCPriceProjectionV1")
        for name in (
            "market_issue_sha256",
            "cycle_observation_sha256",
            "beta_task_sha256",
        ):
            require_sha256(name, getattr(self, name))
        if not isinstance(self.cycle_return, BetaMPCCycleReturnV1):
            raise TypeError("cycle_return must be BetaMPCCycleReturnV1")
        if (
            not isinstance(self.selected_solution_vector, tuple)
            or not self.selected_solution_vector
        ):
            raise ValueError(
                "selected_solution_vector must be a non-empty immutable tuple"
            )
        for index, value in enumerate(self.selected_solution_vector):
            _finite(f"selected_solution_vector[{index}]", value)
        if self.price_projection.market_issue_sha256 != self.market_issue_sha256:
            raise ValueError("plan price projection is bound to another market issue")
        if self.price_projection.horizon_intervals != self.horizon_intervals:
            raise ValueError("plan price projection does not span the complete horizon")
        if self.cycle_return.remaining_intervals != self.horizon_intervals:
            raise ValueError("plan horizon does not end at the next cycle boundary")
        terminal = self.steps[-1]
        terminal_values = {
            "battery": terminal.battery_energy_end_kwh,
            "h2": terminal.h2_end_kg,
            "co2": terminal.co2_end_kg,
            "ch4": terminal.ch4_end_kg,
        }
        for resource in self.cycle_return.resources:
            actual = terminal_values[resource.resource]
            # Cycle settlement uses an absolute resource tolerance.  Do not
            # inflate a hard terminal lower bound in proportion to inventory.
            allowed = CYCLE_FEASIBILITY_EPSILON
            if actual < resource.required_minimum - allowed:
                raise ValueError(f"plan violates hard {resource.resource} cycle return")

    def body(self) -> dict[str, Any]:
        body = EconomicMPCPlanV1.body(self)
        body.update(
            {
                "schema_version": BETA_MPC_SCHEMA_VERSION,
                "controller_id": BETA_MPC_CONTROLLER_ID,
                "model_version": BETA_MPC_MODEL_VERSION,
                "objective_version": BETA_MPC_OBJECTIVE_VERSION,
                "weather_projection_sha256": self.forecast_sha256,
                "price_projection": self.price_projection.to_dict(),
                "market_issue_sha256": self.market_issue_sha256,
                "cycle_observation_sha256": self.cycle_observation_sha256,
                "beta_task_sha256": self.beta_task_sha256,
                "cycle_return": self.cycle_return.to_dict(),
                # Retain the exact canonical stage-two vector, including
                # solver-degenerate auxiliary variables.  A detached verifier
                # cannot reconstruct its raw digest from the reduced physical
                # plan alone.
                "selected_solution_vector": [
                    float(value) for value in self.selected_solution_vector
                ],
                "predicted_cycle_boundary": {
                    "battery_kwh": float(self.steps[-1].battery_energy_end_kwh),
                    "h2_kg": float(self.steps[-1].h2_end_kg),
                    "co2_kg": float(self.steps[-1].co2_end_kg),
                    "ch4_kg": float(self.steps[-1].ch4_end_kg),
                },
            }
        )
        return body

    @property
    def sha256(self) -> str:
        body = self.body()
        body["stages"] = [
            {"stage": stage.stage, "solution_sha256": stage.solution_sha256}
            for stage in self.stages
        ]
        return domain_sha256(BETA_MPC_PLAN_DOMAIN, body)


@dataclass(frozen=True, slots=True)
class BetaMPCDecisionEvidenceV1:
    """Complete hash chain for one beta MPC process-setpoint decision."""

    decision_index: int
    beta_observation_sha256: str
    physical_observation_sha256: str
    market_issue_sha256: str
    cycle_observation_sha256: str
    weather_projection_sha256: str
    price_projection_sha256: str
    settings_sha256: str
    assumptions_sha256: str
    beta_task_sha256: str
    cycle_return_sha256: str
    model_sha256: str
    plan_sha256: str
    market_id: str
    targets: ProcessSetpointTargetsV1
    plan: BetaMPCPlanV1

    schema: ClassVar[str] = BETA_MPC_DECISION_SCHEMA

    def __post_init__(self) -> None:
        decision = _index("decision_index", self.decision_index)
        for name in (
            "beta_observation_sha256",
            "physical_observation_sha256",
            "market_issue_sha256",
            "cycle_observation_sha256",
            "weather_projection_sha256",
            "price_projection_sha256",
            "settings_sha256",
            "assumptions_sha256",
            "beta_task_sha256",
            "cycle_return_sha256",
            "model_sha256",
            "plan_sha256",
        ):
            require_sha256(name, getattr(self, name))
        if not isinstance(self.market_id, str) or not self.market_id.strip():
            raise ValueError("market_id must be non-empty")
        if not isinstance(self.targets, ProcessSetpointTargetsV1):
            raise TypeError("targets must be ProcessSetpointTargetsV1")
        if not isinstance(self.plan, BetaMPCPlanV1):
            raise TypeError("plan must be BetaMPCPlanV1")
        if self.plan.sha256 != self.plan_sha256:
            raise ValueError("decision plan digest differs from its typed plan")
        if self.plan.model_sha256 != self.model_sha256:
            raise ValueError("decision model digest differs from its typed plan")
        if self.plan.forecast_sha256 != self.weather_projection_sha256:
            raise ValueError("decision weather projection differs from its plan")
        if self.plan.price_projection.sha256 != self.price_projection_sha256:
            raise ValueError("decision price projection differs from its plan")
        if self.plan.market_issue_sha256 != self.market_issue_sha256:
            raise ValueError("decision market issue differs from its plan")
        if self.plan.cycle_observation_sha256 != self.cycle_observation_sha256:
            raise ValueError("decision cycle observation differs from its plan")
        if self.plan.settings_sha256 != self.settings_sha256:
            raise ValueError("decision settings differ from its plan")
        if self.plan.assumptions_sha256 != self.assumptions_sha256:
            raise ValueError("decision assumptions differ from its plan")
        if self.plan.beta_task_sha256 != self.beta_task_sha256:
            raise ValueError("decision beta task differs from its plan")
        if self.plan.cycle_return.sha256 != self.cycle_return_sha256:
            raise ValueError("decision cycle return differs from its plan")
        if self.plan.price_projection.market_id != self.market_id:
            raise ValueError("decision market id differs from its plan")
        if self.plan.price_projection.issued_at_decision != decision:
            raise ValueError("decision index differs from its public market issue")
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
            raise ValueError("decision targets do not match the first beta plan step")

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "controller_id": BETA_MPC_CONTROLLER_ID,
            "decision_index": int(self.decision_index),
            "beta_observation_sha256": self.beta_observation_sha256,
            "physical_observation_sha256": self.physical_observation_sha256,
            "market_issue_sha256": self.market_issue_sha256,
            "cycle_observation_sha256": self.cycle_observation_sha256,
            "weather_projection_sha256": self.weather_projection_sha256,
            "price_projection_sha256": self.price_projection_sha256,
            "settings_sha256": self.settings_sha256,
            "assumptions_sha256": self.assumptions_sha256,
            "beta_task_sha256": self.beta_task_sha256,
            "cycle_return_sha256": self.cycle_return_sha256,
            "model_sha256": self.model_sha256,
            "plan_sha256": self.plan_sha256,
            "market_id": self.market_id,
            "targets": self.targets.to_dict(),
            "plan": self.plan.to_dict(),
        }

    @property
    def sha256(self) -> str:
        # Solver wall-clock evidence stays serialized in the typed plan, but the
        # command source identity follows the plan's timing-free semantic digest.
        body = self.body()
        body["plan"] = {"sha256": self.plan_sha256}
        return domain_sha256(BETA_MPC_DECISION_DOMAIN, body)

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class BetaMPCDecisionV1:
    evidence: BetaMPCDecisionEvidenceV1
    command: ProcessSetpointCommandV1

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, BetaMPCDecisionEvidenceV1):
            raise TypeError("evidence must be BetaMPCDecisionEvidenceV1")
        if not isinstance(self.command, ProcessSetpointCommandV1):
            raise TypeError("command must be ProcessSetpointCommandV1")
        if self.command.decision_index != self.evidence.decision_index:
            raise ValueError("command and evidence decision indices differ")
        if self.command.observation_sha256 != self.evidence.physical_observation_sha256:
            raise ValueError("command and evidence physical observations differ")
        if self.command.issuer_id != BETA_MPC_CONTROLLER_ID:
            raise ValueError("command issuer is not beta MPC")
        if self.command.source_schema != self.evidence.schema:
            raise ValueError("command source schema differs from beta evidence")
        if self.command.source_sha256 != self.evidence.sha256:
            raise ValueError("command source digest differs from beta evidence")
        if self.command.targets != self.evidence.targets:
            raise ValueError("command targets differ from beta evidence")

    def to_dict(self) -> dict[str, Any]:
        return {"evidence": self.evidence.to_dict(), "command": self.command.to_dict()}


class BetaMPCError(RuntimeError):
    """Base class for explicit beta-MPC failures."""


class BetaMPCInputError(BetaMPCError):
    """A public input or declared task binding is inconsistent or unsupported."""


class BetaMPCSolveError(BetaMPCError):
    """The hard beta model did not solve; no substitute command may be emitted."""

    def __init__(
        self,
        message: str,
        *,
        stages: tuple[EconomicMPCStageSolveV1, ...] = (),
    ) -> None:
        super().__init__(message)
        if not isinstance(stages, tuple) or any(
            not isinstance(stage, EconomicMPCStageSolveV1) for stage in stages
        ):
            raise TypeError("beta solve stages must be an immutable typed tuple")
        self.stages = stages


class BetaMPCPlanError(BetaMPCError):
    """A numerical solution failed independent beta plan validation."""


__all__ = [
    "BETA_MPC_CONTROLLER_ID",
    "BETA_MPC_DAC_AVAILABILITY_ASSUMPTION",
    "BETA_MPC_DECISION_SCHEMA",
    "BETA_MPC_MODEL_MISMATCH_DECLARATION",
    "BETA_MPC_MODEL_VERSION",
    "BETA_MPC_OBJECTIVE_VERSION",
    "BETA_MPC_PRICE_PROJECTION_VERSION",
    "BETA_MPC_SCHEMA_VERSION",
    "BETA_MPC_SOLUTION_DOMAIN",
    "DEFAULT_BETA_MPC_SETTINGS_V1",
    "BetaMPCCycleReturnV1",
    "BetaMPCDecisionEvidenceV1",
    "BetaMPCDecisionV1",
    "BetaMPCError",
    "BetaMPCInputError",
    "BetaMPCPlanError",
    "BetaMPCPlanV1",
    "BetaMPCPriceProjectionV1",
    "BetaMPCResourceReturnV1",
    "BetaMPCSettingsV1",
    "BetaMPCSolveError",
    "beta_cycle_observation_sha256",
    "project_beta_mpc_prices",
]

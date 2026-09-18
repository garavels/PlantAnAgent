"""Transparent delivery controllers for the causal Environment V2 boundary.

Every public controller in this module shares one planner, immutable settings, and
dispatch path.  The clear-sky controller is a diagnostic that integrates the known
design envelope exactly.  The climatology and issued-forecast comparators form the
clean information-treatment pair: both use the same canonical aggregate bands,
piecewise-constant overlap rule, deadline transform, and next-delivery scarcity
transform.  They differ only in band values -- frozen climatological resource times
analytic clear-sky geometry versus the current public issuance.  Uniform value
within a band is an explicit disaggregation assumption; it uses no future truth and
avoids discontinuous horizon collapse at a delivery or commitment boundary.

Planning happens in physical units.  The controller first calculates methane and
stoichiometric feedstock deficits, turns those deficits into deadline-closing kW,
and allocates the observed solar bus in priority order.  Shares are only a final
encoding step; they are never deliberately oversubscribed for ``Action.sanitised``
to repair later.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import ClassVar

from plant_sim import (
    Action,
    ElectrolyserCommitmentProfile,
    ElectrolyserCommitmentSnapshot,
    ElectrolyserMode,
    PlantConfig,
)

from ...controllers.base import Constraints, ControlDecision
from .control import ControllerObservationV2, close_unit_simplex
from .weather import FORECAST_WINDOWS_H, PublicForecastIssueV2

SECONDS_PER_HOUR = 3_600.0
#: Frozen unconditional resource expectation under the synthetic generator's
#: declared stationary latent prior: E[sigmoid(N(logit(.70), .75^2))], evaluated
#: once with 128-node Gauss-Hermite quadrature.  It is calibration, never truth.
FROZEN_CLIMATOLOGICAL_RESOURCE_FACTOR_V2 = 0.6803843775134744


def _fraction(name: str, value: object, *, maximum: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= maximum:
        raise ValueError(f"{name} must be finite and within [0, {maximum:g}]")
    return number


@dataclass(frozen=True, slots=True)
class DeliveryPlannerSettingsV2:
    """Small, explicit tuning surface shared by both causal comparator arms.

    ``delivery_buffer_fraction`` is extra methane held against integration and
    settlement edge effects. ``feedstock_reserve_fraction`` expresses the H2 and
    CO2 reserve as the stoichiometric mass needed for that fraction of the next
    delivery.  Both are intentionally capped at 25% so a mislabeled value cannot
    silently turn a small operational buffer into a second delivery obligation.

    ``climatological_resource_factor`` is the frozen no-issued-forecast resource
    multiplier applied to analytic clear-sky energy inside every included band.
    Its default is the unconditional expectation of the declared synthetic
    weather prior, not a statistic fitted to episode truth or controller outcomes.

    The public forecast is converted to a configured PV factor as
    ``(1 - forecast_p10_weight) * mean + forecast_p10_weight * p10``.
    """

    delivery_buffer_fraction: float = 0.03
    feedstock_reserve_fraction: float = 0.03
    climatological_resource_factor: float = (
        FROZEN_CLIMATOLOGICAL_RESOURCE_FACTOR_V2
    )
    forecast_p10_weight: float = 0.0

    def __post_init__(self) -> None:
        _fraction(
            "delivery_buffer_fraction",
            self.delivery_buffer_fraction,
            maximum=0.25,
        )
        _fraction(
            "feedstock_reserve_fraction",
            self.feedstock_reserve_fraction,
            maximum=0.25,
        )
        _fraction(
            "climatological_resource_factor",
            self.climatological_resource_factor,
        )
        _fraction("forecast_p10_weight", self.forecast_p10_weight)

    def to_dict(self) -> dict[str, float]:
        """Return a detached, strict-JSON-ready settings descriptor."""
        return {name: float(value) for name, value in asdict(self).items()}


DEFAULT_DELIVERY_PLANNER_SETTINGS_V2 = DeliveryPlannerSettingsV2()
DEFAULT_RISK_ADJUSTED_DELIVERY_PLANNER_SETTINGS_V2 = DeliveryPlannerSettingsV2(
    forecast_p10_weight=0.50,
)


@dataclass(frozen=True, slots=True)
class _PowerPlan:
    # ``electrolyser_kw`` is direct solar. Part of ``battery_kw`` can be the
    # solar side of the explicitly contingent support route described below.
    methane_kw: float
    electrolyser_kw: float
    dac_kw: float
    battery_kw: float
    battery_support_kw: float
    routing_bus_kw: float


class _SharedDeliveryControllerV2:
    """One stateless delivery/commitment planner behind both public wrappers."""

    id: ClassVar[str] = "shared-delivery-controller-v2"
    label: ClassVar[str] = "Shared delivery controller V2"
    description: ClassVar[str] = "Internal shared delivery planner."
    uses_forecast: ClassVar[bool] = False
    information_set: ClassVar[str] = ""
    availability_note: ClassVar[str] = "unspecified availability"
    default_settings: ClassVar[DeliveryPlannerSettingsV2] = (
        DEFAULT_DELIVERY_PLANNER_SETTINGS_V2
    )

    def __init__(
        self,
        settings: DeliveryPlannerSettingsV2 | None = None,
    ) -> None:
        chosen = self.default_settings if settings is None else settings
        if not isinstance(chosen, DeliveryPlannerSettingsV2):
            raise TypeError("settings must be DeliveryPlannerSettingsV2")
        self.settings = chosen

    def reset(self) -> None:
        """The policy is stateless; reset is intentionally a no-op."""

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "uses_forecast": self.uses_forecast,
            "information_set": self.information_set,
            "settings": self.settings.to_dict(),
        }

    def descriptor(self) -> dict[str, object]:
        """Alias used by registry-style callers."""
        return self.to_dict()

    def compute_action(
        self,
        observation: ControllerObservationV2,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: ElectrolyserCommitmentProfile | None,
    ) -> ControlDecision:
        if not isinstance(observation, ControllerObservationV2):
            raise TypeError("observation must be ControllerObservationV2")
        if not isinstance(constraints, Constraints):
            raise TypeError("constraints must be Constraints")
        if not isinstance(config, PlantConfig):
            raise TypeError("config must be PlantConfig")
        if commitment_profile is not None and not isinstance(
            commitment_profile, ElectrolyserCommitmentProfile
        ):
            raise TypeError(
                "commitment_profile must be ElectrolyserCommitmentProfile or None"
            )

        boundary = observation.state.boundary
        snapshot = observation.state.electrolyser_commitment
        self._verify_commitment(snapshot, commitment_profile)

        remaining_h = self._deadline_h(observation, config)
        held_h = observation.decision_interval_s / SECONDS_PER_HOUR
        planning_h = self._planning_horizon_h(
            observation,
            config,
            start_h=0.0,
            end_h=remaining_h,
        )
        planning_h = max(config.dt_h, min(remaining_h, planning_h))
        unavailable_h = min(
            remaining_h,
            self._electrolyser_unavailable_h(snapshot, commitment_profile),
        )
        electrolyser_planning_h = self._planning_horizon_h(
            observation,
            config,
            start_h=unavailable_h,
            end_h=remaining_h,
        )
        electrolyser_planning_h = max(
            config.dt_h,
            min(remaining_h, electrolyser_planning_h),
        )

        requested_kg = _non_negative(boundary_value=observation.delivery.next_requested_kg)
        has_delivery = observation.delivery.steps_to_next is not None and requested_kg > 0.0
        ch4_capacity_kg = _non_negative(boundary_value=boundary.ch4_capacity_kg)
        ch4_level_kg = min(
            ch4_capacity_kg,
            _non_negative(boundary_value=boundary.ch4_level_kg),
        )
        carry_fraction = (
            self._next_delivery_carry_fraction(
                observation,
                config,
                remaining_h=remaining_h,
            )
            if has_delivery
            else 0.0
        )
        carry_kg = requested_kg * carry_fraction
        ch4_target_kg = (
            min(
                ch4_capacity_kg,
                requested_kg * (1.0 + self.settings.delivery_buffer_fraction)
                + carry_kg,
            )
            if has_delivery
            else ch4_level_kg
        )
        ch4_deficit_kg = max(0.0, ch4_target_kg - ch4_level_kg)

        h2_capacity_kg = _non_negative(boundary_value=boundary.h2_capacity_kg)
        h2_level_kg = min(
            h2_capacity_kg,
            _non_negative(boundary_value=boundary.h2_level_kg),
        )
        co2_capacity_kg = _non_negative(boundary_value=boundary.co2_capacity_kg)
        co2_level_kg = min(
            co2_capacity_kg,
            _non_negative(boundary_value=boundary.co2_level_kg),
        )

        reserve_ch4_equivalent_kg = (
            requested_kg * self.settings.feedstock_reserve_fraction
            if has_delivery
            else 0.0
        )
        recipe_basis_kg = ch4_deficit_kg + reserve_ch4_equivalent_kg
        h2_target_kg = min(
            h2_capacity_kg,
            recipe_basis_kg * config.methanator.h2_kg_per_kg_ch4,
        )
        co2_target_kg = min(
            co2_capacity_kg,
            recipe_basis_kg * config.methanator.co2_kg_per_kg_ch4,
        )
        h2_deficit_kg = max(
            0.0,
            h2_target_kg - h2_level_kg,
        )
        co2_deficit_kg = max(
            0.0,
            co2_target_kg - co2_level_kg,
        )

        # Methanation runs after feedstock production inside one physics step,
        # but assigning a scarce bus to an empty reactor can still deadlock the
        # upstream units.  Current tank contents are the only guaranteed inputs,
        # so product power is capped by methane that is immediately feedable.
        feedable_ch4_kg = min(
            ch4_deficit_kg,
            h2_level_kg / max(config.methanator.h2_kg_per_kg_ch4, 1e-12),
            co2_level_kg / max(config.methanator.co2_kg_per_kg_ch4, 1e-12),
        )

        methanator_kw = self._methanator_target_kw(
            feedable_ch4_kg,
            planning_h,
            boundary,
            constraints,
            config,
        )
        electrolyser_raw_kw = self._electrolyser_target_kw(
            h2_deficit_kg,
            electrolyser_planning_h,
            boundary,
            commitment_profile,
            config,
        )
        electrolyser_kw = self._commitment_target_kw(
            electrolyser_raw_kw,
            held_s=observation.decision_interval_s,
            snapshot=snapshot,
            profile=commitment_profile,
            constraints=constraints,
            config=config,
            offline=_offline(boundary, "electrolyser"),
        )
        dac_kw = self._dac_target_kw(
            co2_deficit_kg,
            planning_h,
            boundary,
            constraints,
            config,
        )

        bus_kw = _non_negative(boundary_value=boundary.solar_available_kw)
        support_cap_kw = self._sustainable_support_kw(
            boundary,
            constraints,
            config,
            held_h=held_h,
        )
        plan = self._allocate_kw(
            bus_kw=bus_kw,
            methane_target_kw=methanator_kw,
            electrolyser_target_kw=electrolyser_kw,
            dac_target_kw=dac_kw,
            support_cap_kw=support_cap_kw,
            snapshot=snapshot,
            profile=commitment_profile,
            config=config,
            battery_offline=_offline(boundary, "battery"),
        )
        action = self._action_from_plan(plan)

        mode = "legacy" if snapshot is None else snapshot.mode.value
        notes = (
            f"CH4 target {ch4_target_kg:.2f} kg includes a "
            f"{self.settings.delivery_buffer_fraction:.1%} delivery buffer and "
            f"{carry_kg:.2f} kg inferred next-delivery carry",
            f"Stoichiometric targets H2={h2_target_kg:.2f} kg, "
            f"CO2={co2_target_kg:.2f} kg; immediately feedable "
            f"CH4={feedable_ch4_kg:.2f} kg",
            f"Deadline horizon {planning_h:.3f} h ({self.availability_note}); "
            f"electrolyser production horizon={electrolyser_planning_h:.3f} h "
            f"after {unavailable_h:.3f} h unavailable; mode={mode}",
            "Ordered solar allocation (kW): "
            f"methanator={plan.methane_kw:.2f}, "
            f"electrolyser-direct={plan.electrolyser_kw:.2f}, "
            f"DAC={plan.dac_kw:.2f}, battery={plan.battery_kw:.2f}; "
            f"contingent battery top-up={plan.battery_support_kw:.2f}",
        )
        return ControlDecision(action=action, notes=notes)

    @staticmethod
    def _deadline_h(
        observation: ControllerObservationV2,
        config: PlantConfig,
    ) -> float:
        steps = observation.delivery.steps_to_next
        if steps is None:
            return max(config.dt_h, observation.decision_interval_s / SECONDS_PER_HOUR)
        return max(config.dt_h, steps * config.dt_s / SECONDS_PER_HOUR)

    def _planning_horizon_h(
        self,
        observation: ControllerObservationV2,
        config: PlantConfig,
        *,
        start_h: float,
        end_h: float,
    ) -> float:
        """Map either causal availability source to equivalent full-power hours."""
        return self._availability_equivalent_h(
            observation,
            config,
            start_h=start_h,
            end_h=end_h,
        )

    def _availability_equivalent_h(
        self,
        observation: ControllerObservationV2,
        config: PlantConfig,
        *,
        start_h: float,
        end_h: float,
    ) -> float:
        raise NotImplementedError

    def _next_delivery_carry_fraction(
        self,
        observation: ControllerObservationV2,
        config: PlantConfig,
        *,
        remaining_h: float,
    ) -> float:
        period_h = observation.delivery.delivery_period_steps * config.dt_h
        equivalent_h = self._availability_equivalent_h(
            observation,
            config,
            start_h=remaining_h,
            end_h=remaining_h + period_h,
        )
        # Both comparator arms use the identical scarcity transform.  The
        # inferred next amount is the current amount because the causal boundary
        # intentionally withholds the private future delivery schedule.
        return min(1.0, max(0.0, 1.0 - equivalent_h / period_h))

    @staticmethod
    def _electrolyser_unavailable_h(
        snapshot: ElectrolyserCommitmentSnapshot | None,
        profile: ElectrolyserCommitmentProfile | None,
    ) -> float:
        """Conservative wall time before the stack can next make hydrogen."""
        if snapshot is None or profile is None:
            return 0.0
        mode = snapshot.mode
        if mode is ElectrolyserMode.RUNNING:
            unavailable_s = 0.0
        elif mode in {
            ElectrolyserMode.COLD_STARTING,
            ElectrolyserMode.HOT_STARTING,
        }:
            unavailable_s = snapshot.startup_remaining_s
        elif mode is ElectrolyserMode.HOT_STANDBY:
            unavailable_s = profile.hot_start_s
        elif mode is ElectrolyserMode.COOLDOWN:
            unavailable_s = (
                max(
                    snapshot.cooldown_remaining_s,
                    snapshot.min_down_remaining_s,
                )
                + profile.cold_start_s
            )
        elif mode is ElectrolyserMode.OFF:
            unavailable_s = snapshot.min_down_remaining_s + profile.cold_start_s
        else:
            raise ValueError(f"unsupported electrolyser commitment mode: {mode!r}")
        return unavailable_s / SECONDS_PER_HOUR

    @staticmethod
    def _verify_commitment(
        snapshot: ElectrolyserCommitmentSnapshot | None,
        profile: ElectrolyserCommitmentProfile | None,
    ) -> None:
        if profile is None:
            if snapshot is not None:
                raise ValueError(
                    "commitment snapshot is present but commitment profile is disabled"
                )
            return
        if snapshot is None:
            raise ValueError("commitment profile is enabled but snapshot is missing")
        if not isinstance(snapshot, ElectrolyserCommitmentSnapshot):
            raise TypeError("commitment snapshot must be ElectrolyserCommitmentSnapshot")
        if not isinstance(snapshot.mode, ElectrolyserMode):
            raise TypeError("commitment snapshot mode must be ElectrolyserMode")
        if (
            snapshot.profile_id != profile.profile_id
            or snapshot.profile_sha256 != profile.profile_sha256
            or snapshot.technology != profile.technology
            or snapshot.calibration_basis != profile.calibration_basis
            or snapshot.dynamics_version != profile.dynamics_version
        ):
            raise ValueError(
                "commitment snapshot/profile identity mismatch: "
                f"snapshot={snapshot.profile_sha256}, "
                f"profile={profile.profile_sha256}"
            )
        state_values = (
            snapshot.mode_elapsed_s,
            snapshot.startup_remaining_s,
            snapshot.cooldown_remaining_s,
            snapshot.min_up_remaining_s,
            snapshot.min_down_remaining_s,
            snapshot.shortfall_elapsed_s,
            snapshot.previous_power_kw,
            snapshot.ramp_up_headroom_kw,
            snapshot.ramp_down_headroom_kw,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or value < 0.0
            for value in state_values
        ):
            raise ValueError(
                "commitment snapshot state must be finite and non-negative"
            )

    @staticmethod
    def _methanator_target_kw(
        deficit_kg: float,
        horizon_h: float,
        boundary: object,
        constraints: Constraints,
        config: PlantConfig,
    ) -> float:
        if deficit_kg <= 0.0 or _offline(boundary, "methanator"):
            return 0.0
        health = _health_fraction(boundary, "methanator")
        limit_kw = min(
            _finite_limit(constraints.methanator_limit_kw(config)),
            config.methanator.rated_kw * health,
        )
        return min(
            limit_kw,
            deficit_kg * config.methanator.specific_energy_kwh_per_kg / horizon_h,
        )

    @staticmethod
    def _electrolyser_target_kw(
        deficit_kg: float,
        horizon_h: float,
        boundary: object,
        profile: ElectrolyserCommitmentProfile | None,
        config: PlantConfig,
    ) -> float:
        if deficit_kg <= 0.0 or _offline(boundary, "electrolyser"):
            return 0.0
        health = max(_health_fraction(boundary, "electrolyser"), 1e-12)
        specific_energy = config.electrolyser.specific_energy_kwh_per_kg
        if profile is not None:
            specific_energy = max(
                specific_energy,
                profile.nominal_specific_energy_kwh_per_kg,
            ) * profile.specific_energy_multiplier(profile.min_stable_load_frac)
        return deficit_kg * specific_energy / (health * horizon_h)

    @staticmethod
    def _dac_target_kw(
        deficit_kg: float,
        horizon_h: float,
        boundary: object,
        constraints: Constraints,
        config: PlantConfig,
    ) -> float:
        if deficit_kg <= 0.0 or _offline(boundary, "dac"):
            return 0.0
        health = max(_health_fraction(boundary, "dac"), 1e-12)
        return min(
            _finite_limit(constraints.dac_limit_kw(config)),
            deficit_kg * config.dac.specific_energy_kwh_per_kg / (health * horizon_h),
        )

    @staticmethod
    def _commitment_target_kw(
        raw_target_kw: float,
        *,
        held_s: float,
        snapshot: ElectrolyserCommitmentSnapshot | None,
        profile: ElectrolyserCommitmentProfile | None,
        constraints: Constraints,
        config: PlantConfig,
        offline: bool,
    ) -> float:
        if offline:
            return 0.0
        cap_kw = _finite_limit(constraints.electrolyser_limit_kw(config))
        if snapshot is None or profile is None:
            if raw_target_kw <= 0.0:
                return 0.0
            minimum_kw = config.electrolyser.min_load_frac * config.electrolyser.rated_kw
            return min(cap_kw, max(raw_target_kw, minimum_kw)) if cap_kw >= minimum_kw else 0.0

        rated_kw = config.electrolyser.rated_kw
        start_kw = profile.start_load_frac * rated_kw
        stable_kw = profile.min_stable_load_frac * rated_kw
        standby_kw = profile.hot_standby_load_frac * rated_kw
        production_needed = raw_target_kw > 0.0
        mode = snapshot.mode

        if mode is ElectrolyserMode.COOLDOWN:
            return 0.0
        if mode is ElectrolyserMode.OFF:
            if not production_needed or snapshot.min_down_remaining_s > 0.0:
                return 0.0
            return min(cap_kw, start_kw) if cap_kw >= start_kw else 0.0
        if mode is ElectrolyserMode.COLD_STARTING:
            if not production_needed:
                return 0.0
            required_kw = (
                max(raw_target_kw, stable_kw)
                if snapshot.startup_remaining_s < held_s
                else start_kw
            )
            return min(cap_kw, required_kw) if cap_kw >= start_kw else 0.0
        if mode is ElectrolyserMode.HOT_STARTING:
            if not production_needed and snapshot.min_up_remaining_s <= 0.0:
                return 0.0
            if production_needed and snapshot.startup_remaining_s < held_s:
                required_kw = max(raw_target_kw, stable_kw)
            else:
                required_kw = start_kw
            return min(cap_kw, required_kw) if cap_kw >= start_kw else 0.0
        if mode is ElectrolyserMode.HOT_STANDBY:
            if not production_needed:
                return min(cap_kw, standby_kw) if snapshot.min_up_remaining_s > 0.0 else 0.0
            if cap_kw >= stable_kw:
                return min(cap_kw, max(raw_target_kw, stable_kw))
            return min(cap_kw, standby_kw)
        if mode is not ElectrolyserMode.RUNNING:
            raise ValueError(f"unsupported electrolyser commitment mode: {mode!r}")

        if not production_needed:
            return min(cap_kw, standby_kw) if snapshot.min_up_remaining_s > 0.0 else 0.0
        if cap_kw < stable_kw:
            return min(cap_kw, standby_kw)
        # A controller action is held for many physics intervals.  Command the
        # desired endpoint; the minute-level commitment model applies each ramp
        # step.  Capping here by one snapshot interval would strand the stack at
        # that first ramp step for the entire held decision.
        return min(cap_kw, rated_kw, max(raw_target_kw, stable_kw))

    @staticmethod
    def _sustainable_support_kw(
        boundary: object,
        constraints: Constraints,
        config: PlantConfig,
        *,
        held_h: float,
    ) -> float:
        if _offline(boundary, "battery") or held_h <= 0.0:
            return 0.0
        reserve_soc = constraints.reserve_soc(config)
        soc = _non_negative(boundary_value=getattr(boundary, "battery_soc", 0.0))
        capacity_kwh = _non_negative(
            boundary_value=getattr(boundary, "battery_capacity_kwh", 0.0)
        )
        usable_kwh = max(0.0, soc - reserve_soc) * capacity_kwh
        sustainable_kw = (
            usable_kwh * config.battery.discharge_efficiency / held_h
        )
        return min(
            _finite_limit(constraints.support_limit_kw(config)),
            sustainable_kw,
        )

    @staticmethod
    def _allocate_kw(
        *,
        bus_kw: float,
        methane_target_kw: float,
        electrolyser_target_kw: float,
        dac_target_kw: float,
        support_cap_kw: float,
        snapshot: ElectrolyserCommitmentSnapshot | None,
        profile: ElectrolyserCommitmentProfile | None,
        config: PlantConfig,
        battery_offline: bool,
    ) -> _PowerPlan:
        routing_bus_kw = bus_kw
        if routing_bus_kw <= 0.0:
            # Shares remain in force for the whole held decision.  Build them on
            # current causal process demand so solar appearing after a zero-power
            # boundary is routed rather than curtailed while storage supplies the
            # same load.  With no process demand, route emerging solar to storage.
            routing_bus_kw = (
                methane_target_kw + electrolyser_target_kw + dac_target_kw
            )
            if routing_bus_kw <= 0.0:
                reference_kw = 1.0 if not battery_offline else 0.0
                return _PowerPlan(
                    methane_kw=0.0,
                    electrolyser_kw=0.0,
                    dac_kw=0.0,
                    battery_kw=reference_kw,
                    battery_support_kw=0.0,
                    routing_bus_kw=reference_kw,
                )
        remaining_kw = routing_bus_kw

        # Consume only immediately feedable product first.  Then secure the
        # stack's physical floor, give the complementary DAC a chance to make
        # feedstock, and use any remaining bus for the stack's desired endpoint.
        methane_kw = min(methane_target_kw, remaining_kw)
        remaining_kw -= methane_kw
        support_kw = min(support_cap_kw, electrolyser_target_kw)

        # ``battery_support_kw`` is a contingent top-up, not an instruction to
        # discharge while solar is plentiful. Put the support-backed portion of
        # the stack's solar claim in the battery share. The simulator's declared
        # dispatch contract first cancels that exact charge/support overlap and
        # routes the solar straight to the stack; only a within-hold solar drop
        # draws the uncovered remainder from storage. This is distinct from
        # share sanitisation: the four solar claims below still sum to <= bus.
        direct_target_kw = max(0.0, electrolyser_target_kw - support_kw)

        # Do not signal an operating intent that cannot clear the applicable
        # physical floor. Partial intent only spills solar (or pauses a start)
        # and cannot make useful feedstock.  Battery-share overlap can replace
        # discharge, but it cannot add to ``direct solar + support``: dispatch
        # cancels the same number of support kW when it moves overlap to the
        # stack.  Feasibility must therefore secure the direct complement first.
        required_kw = config.electrolyser.min_load_frac * config.electrolyser.rated_kw
        if snapshot is not None and profile is not None:
            if snapshot.mode in {
                ElectrolyserMode.OFF,
                ElectrolyserMode.COLD_STARTING,
                ElectrolyserMode.HOT_STARTING,
            }:
                required_kw = profile.start_load_frac * config.electrolyser.rated_kw
            elif snapshot.mode in {
                ElectrolyserMode.RUNNING,
                ElectrolyserMode.HOT_STANDBY,
            }:
                standby_kw = (
                    profile.hot_standby_load_frac * config.electrolyser.rated_kw
                )
                stable_kw = profile.min_stable_load_frac * config.electrolyser.rated_kw
                required_kw = (
                    stable_kw if electrolyser_target_kw >= stable_kw else standby_kw
                )
            else:
                required_kw = 0.0
        minimum_offer_kw = min(electrolyser_target_kw, required_kw)
        minimum_direct_kw = min(
            max(0.0, minimum_offer_kw - support_kw),
            remaining_kw,
        )
        remaining_kw -= minimum_direct_kw
        electrolyser_kw = minimum_direct_kw
        contingent_solar_kw = 0.0
        electrolyser_feasible = (
            electrolyser_target_kw <= 0.0
            or electrolyser_kw + support_kw + 1e-9 >= required_kw
        )
        if not electrolyser_feasible:
            remaining_kw += electrolyser_kw
            electrolyser_kw = 0.0
            support_kw = 0.0

        dac_kw = min(dac_target_kw, remaining_kw)
        remaining_kw -= dac_kw

        if electrolyser_feasible:
            extra_direct_kw = min(
                max(0.0, direct_target_kw - electrolyser_kw),
                remaining_kw,
            )
            electrolyser_kw += extra_direct_kw
            remaining_kw -= extra_direct_kw
            # Only after useful process claims are placed do we route spare
            # solar through the battery share.  Dispatch can then substitute
            # it for up to the same support kW without changing stack power.
            contingent_solar_kw = min(support_kw, remaining_kw)
            remaining_kw -= contingent_solar_kw
        battery_kw = (
            0.0 if battery_offline else remaining_kw + contingent_solar_kw
        )
        return _PowerPlan(
            methane_kw=methane_kw,
            electrolyser_kw=electrolyser_kw,
            dac_kw=dac_kw,
            battery_kw=battery_kw,
            battery_support_kw=support_kw,
            routing_bus_kw=routing_bus_kw,
        )

    @staticmethod
    def _action_from_plan(plan: _PowerPlan) -> Action:
        bus_kw = plan.routing_bus_kw
        if bus_kw <= 0.0:
            return Action.idle()
        raw_shares = (
            plan.battery_kw / bus_kw,
            plan.electrolyser_kw / bus_kw,
            plan.dac_kw / bus_kw,
            plan.methane_kw / bus_kw,
        )
        shares = close_unit_simplex(raw_shares)
        action = Action(
            battery_share=shares[0],
            electrolyser_share=shares[1],
            dac_share=shares[2],
            methanator_share=shares[3],
            battery_support_kw=plan.battery_support_kw,
        )
        # This is an assertion about planner construction, never a correction.
        shares = (
            action.battery_share,
            action.electrolyser_share,
            action.dac_share,
            action.methanator_share,
        )
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in shares):
            raise RuntimeError("delivery planner produced a non-finite bus share")
        if sum(shares) > 1.0:
            raise RuntimeError("delivery planner oversubscribed the solar bus")
        return action


class DeliveryAwareControllerV2(_SharedDeliveryControllerV2):
    """Clear-sky diagnostic that is invariant to the forecast field."""

    id = "delivery-aware-v2"
    label = "Delivery-aware (analytic clear-sky diagnostic)"
    description = (
        "Transparent deadline-closing delivery planner using current inventories, "
        "known clear-sky plant geometry, constraints, and electrolyser commitment "
        "state without a weather forecast."
    )
    uses_forecast = False
    availability_note = "analytic clear-sky envelope"
    information_set = (
        "current V2 boundary time and preceding-interval telemetry, next delivery "
        "context, known clear-sky PlantConfig.solar geometry, safety constraints, "
        "and public commitment snapshot; issued weather forecast excluded"
    )

    def _availability_equivalent_h(
        self,
        observation: ControllerObservationV2,
        config: PlantConfig,
        *,
        start_h: float,
        end_h: float,
    ) -> float:
        return _clear_sky_equivalent_h(
            boundary_time_s=_non_negative(
                boundary_value=observation.state.boundary.time_s
            ),
            start_h=start_h,
            end_h=end_h,
            sunrise_h=config.solar.sunrise_h,
            sunset_h=config.solar.sunset_h,
        )


class _CanonicalBandControllerV2(_SharedDeliveryControllerV2):
    """Shared piecewise-constant resolution for the two forecast-information arms."""

    def _availability_equivalent_h(
        self,
        observation: ControllerObservationV2,
        config: PlantConfig,
        *,
        start_h: float,
        end_h: float,
    ) -> float:
        if end_h <= start_h:
            return 0.0
        return sum(
            overlap_h
            * self._band_peak_power_fraction(
                observation,
                config,
                band_index=band_index,
                start_h=band_start_h,
                end_h=band_end_h,
            )
            for (
                band_index,
                band_start_h,
                band_end_h,
                overlap_h,
            ) in self._overlapping_bands(
                start_h=start_h,
                end_h=end_h,
            )
        )

    @staticmethod
    def _overlapping_bands(
        *,
        start_h: float,
        end_h: float,
    ) -> tuple[tuple[int, float, float, float], ...]:
        """Return canonical bands and their exact overlap with one query interval."""
        return tuple(
            (index, band_start_h, band_end_h, overlap_h)
            for index, (band_start_h, band_end_h) in enumerate(FORECAST_WINDOWS_H)
            if (
                overlap_h := max(
                    0.0,
                    min(end_h, band_end_h) - max(start_h, band_start_h),
                )
            )
            > 0.0
        )

    def _band_peak_power_fraction(
        self,
        observation: ControllerObservationV2,
        config: PlantConfig,
        *,
        band_index: int,
        start_h: float,
        end_h: float,
    ) -> float:
        raise NotImplementedError


class ClimatologyControllerV2(_CanonicalBandControllerV2):
    """No-issued-forecast comparator using one frozen resource climatology."""

    id = "delivery-aware-climatology-v2"
    label = "Delivery-aware (fixed climatology)"
    description = (
        "Transparent deadline-closing delivery planner using canonical aggregate "
        "bands, known clear-sky geometry, and the frozen unconditional resource "
        "expectation of the declared synthetic weather prior without an issued "
        "weather forecast."
    )
    uses_forecast = False
    availability_note = "canonical-band fixed climatology"
    information_set = (
        "current V2 boundary time and preceding-interval telemetry, next delivery "
        "context, known clear-sky PlantConfig.solar geometry, safety constraints, "
        "public commitment snapshot, canonical public forecast-band boundaries, "
        "and frozen prior calibration E[sigmoid(N(logit(.70), .75^2))]="
        f"{FROZEN_CLIMATOLOGICAL_RESOURCE_FACTOR_V2:.16f}; issued weather forecast "
        "and episode truth excluded"
    )

    def _band_peak_power_fraction(
        self,
        observation: ControllerObservationV2,
        config: PlantConfig,
        *,
        band_index: int,
        start_h: float,
        end_h: float,
    ) -> float:
        del band_index
        clear_sky_h = _clear_sky_equivalent_h(
            boundary_time_s=_non_negative(
                boundary_value=observation.state.boundary.time_s
            ),
            start_h=start_h,
            end_h=end_h,
            sunrise_h=config.solar.sunrise_h,
            sunset_h=config.solar.sunset_h,
        )
        band_h = end_h - start_h
        return (
            clear_sky_h
            / band_h
            * self.settings.climatological_resource_factor
        )


class IssuedForecastControllerV2(_CanonicalBandControllerV2):
    """Same planner paced by current issued mean aggregate PV values."""

    id = "issued-forecast-v2"
    label = "Delivery-aware (issued mean forecast)"
    description = (
        "Transparent deadline-closing delivery planner using the current public "
        "mean aggregate PV issuance and no quantile weighting, private weather "
        "truth, or future vintage."
    )
    uses_forecast = True
    allow_nonzero_p10_weight: ClassVar[bool] = False
    availability_note = "canonical-band issued mean forecast"
    information_set = (
        "current V2 boundary and preceding-interval telemetry, next delivery context, "
        "plant design, safety constraints, public commitment snapshot, and only the "
        "mean values from current PublicForecastIssueV2 aggregate windows under the "
        "shared uniform-within-band overlap assumption; p10 excluded by zero weight"
    )

    def __init__(
        self,
        settings: DeliveryPlannerSettingsV2 | None = None,
    ) -> None:
        super().__init__(settings)
        if (
            not self.allow_nonzero_p10_weight
            and self.settings.forecast_p10_weight != 0.0
        ):
            raise ValueError(
                "IssuedForecastControllerV2 is the mean-only arm; use "
                "RiskAdjustedIssuedForecastControllerV2 for non-zero p10 weight"
            )

    def _band_peak_power_fraction(
        self,
        observation: ControllerObservationV2,
        config: PlantConfig,
        *,
        band_index: int,
        start_h: float,
        end_h: float,
    ) -> float:
        del config, start_h, end_h
        forecast = observation.forecast
        if not isinstance(forecast, PublicForecastIssueV2):
            raise ValueError(
                "IssuedForecastControllerV2 requires a current PublicForecastIssueV2"
            )
        window = forecast.windows[band_index]
        risk = self.settings.forecast_p10_weight
        issued_fraction = (
            (1.0 - risk) * window.mean_peak_power_fraction
            + risk * window.p10_peak_power_fraction
        )
        return issued_fraction


class RiskAdjustedIssuedForecastControllerV2(IssuedForecastControllerV2):
    """Predeclared downside-risk ablation using equal mean and p10 weights."""

    id = "risk-adjusted-issued-forecast-v2"
    label = "Delivery-aware (risk-adjusted issued forecast)"
    description = (
        "Issued-forecast comparator with the same planner and canonical band overlap "
        "rule "
        "as the mean arm and a predeclared default 50% p10 weight; explicit "
        "serialized settings support sensitivity ablations."
    )
    availability_note = "canonical-band risk-adjusted issued forecast"
    allow_nonzero_p10_weight = True
    information_set = (
        "current V2 boundary and preceding-interval telemetry, next delivery context, "
        "plant design, safety constraints, public commitment snapshot, and the mean "
        "and p10 values from current PublicForecastIssueV2 aggregate windows under "
        "the shared uniform-within-band overlap assumption; serialized risk weighting"
    )
    default_settings = DEFAULT_RISK_ADJUSTED_DELIVERY_PLANNER_SETTINGS_V2


def _clear_sky_equivalent_h(
    *,
    boundary_time_s: float,
    start_h: float,
    end_h: float,
    sunrise_h: float,
    sunset_h: float,
) -> float:
    """Integrate the configured unit-peak half-sine over an offset interval.

    This is the analytic integral of the public PlantConfig solar geometry, not
    a simulator query or a weather prediction.  Its periodic primitive handles
    intervals crossing midnight and arbitrarily many complete days without a
    sampling loop.
    """
    if not (
        math.isfinite(sunrise_h)
        and math.isfinite(sunset_h)
        and 0.0 <= sunrise_h < sunset_h <= 24.0
    ):
        raise ValueError(
            "clear-sky geometry must satisfy 0 <= sunrise_h < sunset_h <= 24"
        )
    if not (
        math.isfinite(boundary_time_s)
        and math.isfinite(start_h)
        and math.isfinite(end_h)
    ):
        raise ValueError("clear-sky integration bounds must be finite")
    if end_h <= start_h:
        return 0.0

    daylight_h = sunset_h - sunrise_h
    daily_equivalent_h = 2.0 * daylight_h / math.pi

    def primitive(absolute_h: float) -> float:
        day = math.floor(absolute_h / 24.0)
        hour = absolute_h - day * 24.0
        if hour <= sunrise_h:
            partial_h = 0.0
        elif hour >= sunset_h:
            partial_h = daily_equivalent_h
        else:
            phase = math.pi * (hour - sunrise_h) / daylight_h
            partial_h = daylight_h / math.pi * (1.0 - math.cos(phase))
        return day * daily_equivalent_h + partial_h

    absolute_start_h = boundary_time_s / SECONDS_PER_HOUR + start_h
    absolute_end_h = boundary_time_s / SECONDS_PER_HOUR + end_h
    duration_h = end_h - start_h
    equivalent_h = primitive(absolute_end_h) - primitive(absolute_start_h)
    return min(duration_h, max(0.0, equivalent_h))


def _non_negative(*, boundary_value: object) -> float:
    if isinstance(boundary_value, bool) or not isinstance(boundary_value, Real):
        return 0.0
    number = float(boundary_value)
    return max(0.0, number) if math.isfinite(number) else 0.0


def _finite_limit(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        return 0.0
    number = float(value)
    return max(0.0, number) if math.isfinite(number) else 0.0


def _offline(boundary: object, component: str) -> bool:
    health = getattr(boundary, "health", {}).get(component)
    return bool(health is not None and health.offline)


def _health_fraction(boundary: object, component: str) -> float:
    health = getattr(boundary, "health", {}).get(component)
    if health is None:
        return 1.0
    if health.offline:
        return 0.0
    degradation = _non_negative(boundary_value=health.degradation)
    return min(1.0, max(0.0, 1.0 - degradation))


__all__ = [
    "ClimatologyControllerV2",
    "DEFAULT_DELIVERY_PLANNER_SETTINGS_V2",
    "DEFAULT_RISK_ADJUSTED_DELIVERY_PLANNER_SETTINGS_V2",
    "DeliveryAwareControllerV2",
    "DeliveryPlannerSettingsV2",
    "FROZEN_CLIMATOLOGICAL_RESOURCE_FACTOR_V2",
    "IssuedForecastControllerV2",
    "RiskAdjustedIssuedForecastControllerV2",
]

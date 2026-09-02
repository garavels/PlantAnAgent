"""Human-facing delivery, energy, safety, and intervention metrics for V2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from plant_sim import DeliverySettlement

from .boundary import COMMITMENT_MODE_NAMES
from .reward import IntervalOutcome, RewardBreakdownV2
from .spaces import STAGE


@dataclass
class EpisodeMetricsV2:
    decisions: int = 0
    physical_steps: int = 0
    hours: float = 0.0
    reward_total: float = 0.0
    reward_components: dict[str, float] = field(default_factory=dict)
    commitments_settled: int = 0
    commitments_met: int = 0
    requested_kg: float = 0.0
    delivered_kg: float = 0.0
    unmet_kg: float = 0.0
    worst_commitment_fill_rate: float = 1.0
    ch4_produced_kg: float = 0.0
    terminal_ch4_kg: float = 0.0
    solar_kwh: float = 0.0
    curtailed_kwh: float = 0.0
    battery_discharged_kwh: float = 0.0
    min_battery_soc: float = 1.0
    shutdown_steps: int = 0
    guard_interventions: int = 0
    guard_intervention_kwh: float = 0.0
    dispatch_gap_kwh: float = 0.0
    request_application_gap_kwh: float = 0.0
    corrected_physics_steps: int = 0
    out_of_range_decisions: int = 0
    electrolyser_commitment_enabled: bool = False
    electrolyser_cold_starts: int = 0
    electrolyser_hot_starts: int = 0
    electrolyser_aborted_starts: int = 0
    electrolyser_forced_stops: int = 0
    electrolyser_controlled_stops: int = 0
    electrolyser_startup_energy_kwh: float = 0.0
    electrolyser_standby_energy_kwh: float = 0.0
    electrolyser_operating_time_s: float = 0.0
    electrolyser_low_load_time_s: float = 0.0
    electrolyser_ramp_limited_intervals: int = 0
    electrolyser_absolute_load_movement_kw: float = 0.0
    electrolyser_mode_time_s: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in COMMITMENT_MODE_NAMES}
    )

    def record(
        self,
        outcome: IntervalOutcome,
        breakdown: RewardBreakdownV2,
        settlements: Iterable[DeliverySettlement],
        *,
        corrected_physics_steps: int,
    ) -> None:
        self.decisions += 1
        self.physical_steps += outcome.physical_steps
        self.hours += outcome.hours
        self.reward_total += breakdown.total
        for name, value in breakdown.components.items():
            self.reward_components[name] = self.reward_components.get(name, 0.0) + value
        self.requested_kg += outcome.requested_kg
        self.delivered_kg += outcome.delivered_kg
        self.unmet_kg += outcome.unmet_kg
        self.ch4_produced_kg += outcome.produced_kg
        self.terminal_ch4_kg = outcome.terminal_ch4_kg
        self.solar_kwh += outcome.solar_kwh
        self.curtailed_kwh += outcome.curtailed_kwh
        self.battery_discharged_kwh += outcome.battery_discharged_kwh
        self.min_battery_soc = min(self.min_battery_soc, outcome.min_battery_soc)
        self.shutdown_steps += outcome.shutdown_steps
        self.guard_interventions += outcome.corrections
        self.guard_intervention_kwh += outcome.guard_intervention_kwh
        self.dispatch_gap_kwh += outcome.dispatch_gap_kwh
        self.request_application_gap_kwh += outcome.request_application_gap_kwh
        self.corrected_physics_steps += corrected_physics_steps
        self.out_of_range_decisions += int(outcome.out_of_range)
        exposure = outcome.commitment
        self.electrolyser_commitment_enabled = (
            self.electrolyser_commitment_enabled or exposure.enabled
        )
        self.electrolyser_cold_starts += exposure.cold_starts
        self.electrolyser_hot_starts += exposure.hot_starts
        self.electrolyser_aborted_starts += exposure.aborted_starts
        self.electrolyser_forced_stops += exposure.forced_stops
        self.electrolyser_controlled_stops += exposure.controlled_stops
        self.electrolyser_startup_energy_kwh += exposure.startup_energy_kwh
        self.electrolyser_standby_energy_kwh += exposure.standby_energy_kwh
        self.electrolyser_operating_time_s += exposure.operating_time_s
        self.electrolyser_low_load_time_s += exposure.low_load_time_s
        self.electrolyser_ramp_limited_intervals += exposure.ramp_limited_intervals
        self.electrolyser_absolute_load_movement_kw += (
            exposure.absolute_load_movement_kw
        )
        for mode, seconds in exposure.mode_time_s:
            self.electrolyser_mode_time_s[mode] += seconds
        for settlement in settlements:
            self.commitments_settled += 1
            fill = (
                settlement.delivered_kg / settlement.requested_kg
                if settlement.requested_kg > 0.0
                else 1.0
            )
            self.worst_commitment_fill_rate = min(self.worst_commitment_fill_rate, fill)
            if settlement.unmet_kg <= 1e-9:
                self.commitments_met += 1

    @property
    def delivery_fill_rate(self) -> float:
        return self.delivered_kg / self.requested_kg if self.requested_kg > 0.0 else 1.0

    @property
    def commitment_service_level(self) -> float:
        if self.commitments_settled == 0:
            return 1.0
        return self.commitments_met / self.commitments_settled

    @property
    def solar_utilisation(self) -> float:
        if self.solar_kwh <= 0.0:
            return 0.0
        return max(0.0, 1.0 - self.curtailed_kwh / self.solar_kwh)

    @property
    def electrolyser_nonproductive_energy_kwh(self) -> float:
        return (
            self.electrolyser_startup_energy_kwh
            + self.electrolyser_standby_energy_kwh
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": STAGE,
            "decisions": self.decisions,
            "physical_steps": self.physical_steps,
            "hours": self.hours,
            "reward_total": self.reward_total,
            "reward_components": dict(self.reward_components),
            "commitments_settled": self.commitments_settled,
            "commitments_met": self.commitments_met,
            "commitment_service_level": self.commitment_service_level,
            "requested_kg": self.requested_kg,
            "delivered_kg": self.delivered_kg,
            "unmet_kg": self.unmet_kg,
            "delivery_fill_rate": self.delivery_fill_rate,
            "worst_commitment_fill_rate": self.worst_commitment_fill_rate,
            "ch4_produced_kg": self.ch4_produced_kg,
            "terminal_ch4_kg": self.terminal_ch4_kg,
            "solar_kwh": self.solar_kwh,
            "curtailed_kwh": self.curtailed_kwh,
            "solar_utilisation": self.solar_utilisation,
            "battery_discharged_kwh": self.battery_discharged_kwh,
            "min_battery_soc": self.min_battery_soc,
            "shutdown_steps": self.shutdown_steps,
            "guard_interventions": self.guard_interventions,
            "guard_intervention_kwh": self.guard_intervention_kwh,
            "dispatch_gap_kwh": self.dispatch_gap_kwh,
            "request_application_gap_kwh": self.request_application_gap_kwh,
            "corrected_physics_steps": self.corrected_physics_steps,
            "out_of_range_decisions": self.out_of_range_decisions,
            "electrolyser_commitment_enabled": self.electrolyser_commitment_enabled,
            "electrolyser_cold_starts": self.electrolyser_cold_starts,
            "electrolyser_hot_starts": self.electrolyser_hot_starts,
            "electrolyser_aborted_starts": self.electrolyser_aborted_starts,
            "electrolyser_forced_stops": self.electrolyser_forced_stops,
            "electrolyser_controlled_stops": self.electrolyser_controlled_stops,
            "electrolyser_startup_energy_kwh": self.electrolyser_startup_energy_kwh,
            "electrolyser_standby_energy_kwh": self.electrolyser_standby_energy_kwh,
            "electrolyser_nonproductive_energy_kwh": (
                self.electrolyser_nonproductive_energy_kwh
            ),
            "electrolyser_operating_time_s": self.electrolyser_operating_time_s,
            "electrolyser_low_load_time_s": self.electrolyser_low_load_time_s,
            "electrolyser_ramp_limited_intervals": (
                self.electrolyser_ramp_limited_intervals
            ),
            "electrolyser_absolute_load_movement_kw": (
                self.electrolyser_absolute_load_movement_kw
            ),
            "electrolyser_mode_time_s": dict(self.electrolyser_mode_time_s),
        }


__all__ = ["EpisodeMetricsV2"]

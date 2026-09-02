"""Human-facing delivery, energy, safety, and intervention metrics for V2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from plant_sim import DeliverySettlement

from .reward import IntervalOutcome, RewardBreakdownV2


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
    corrected_physics_steps: int = 0
    out_of_range_decisions: int = 0

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
        self.corrected_physics_steps += corrected_physics_steps
        self.out_of_range_decisions += int(outcome.out_of_range)
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": "delivery-only-alpha",
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
            "corrected_physics_steps": self.corrected_physics_steps,
            "out_of_range_decisions": self.out_of_range_decisions,
        }


__all__ = ["EpisodeMetricsV2"]

"""Battery: an energy buffer with capacity, power and efficiency limits."""

from __future__ import annotations

from typing import Any

from ..config import BatteryConfig
from .base import Component


class Battery(Component):
    """Charges and discharges within power limits, never past its energy limits.

    Powers quoted to and from this component are always *bus-side*: efficiency
    losses happen inside the battery, so charging 100 kW stores less than
    100 kWh/h and discharging 100 kW drains more.

    Extension point: degradation (capacity fade, rising internal resistance),
    temperature limits, self-discharge.
    """

    name = "battery"

    def __init__(self, config: BatteryConfig | None = None) -> None:
        self.config = config or BatteryConfig()
        self.energy_kwh = 0.0
        self.power_kw = 0.0
        self.reset()

    def reset(self) -> None:
        self.energy_kwh = self.config.initial_soc * self.config.capacity_kwh
        self.power_kw = 0.0

    # --- read-only views --------------------------------------------------
    @property
    def soc(self) -> float:
        return self.energy_kwh / self.config.capacity_kwh

    @property
    def reserve_kwh(self) -> float:
        """Energy above the minimum SOC, i.e. what is actually usable."""
        return max(0.0, self.energy_kwh - self.config.min_soc * self.config.capacity_kwh)

    def charge_headroom_kw(self, dt_h: float) -> float:
        """Largest bus-side charge power this component can absorb for `dt_h`."""
        room_kwh = max(0.0, self.config.capacity_kwh - self.energy_kwh)
        return min(self.config.max_charge_kw, room_kwh / (dt_h * self.config.charge_efficiency))

    def discharge_headroom_kw(self, dt_h: float) -> float:
        """Largest bus-side discharge power this component can deliver for `dt_h`."""
        return min(self.config.max_discharge_kw, self.reserve_kwh * self.config.discharge_efficiency / dt_h)

    # --- actions ----------------------------------------------------------
    def charge(self, power_kw: float, dt_h: float) -> float:
        """Absorb up to `power_kw`; returns the power actually accepted."""
        accepted = min(max(0.0, power_kw), self.charge_headroom_kw(dt_h))
        self.energy_kwh += accepted * dt_h * self.config.charge_efficiency
        self.power_kw = accepted
        return accepted

    def discharge(self, power_kw: float, dt_h: float) -> float:
        """Deliver up to `power_kw`; returns the power actually delivered."""
        delivered = min(max(0.0, power_kw), self.discharge_headroom_kw(dt_h))
        self.energy_kwh -= delivered * dt_h / self.config.discharge_efficiency
        self.power_kw = -delivered
        return delivered

    def snapshot(self) -> dict[str, Any]:
        return {
            "soc": self.soc,
            "energy_kwh": self.energy_kwh,
            "power_kw": self.power_kw,
            "capacity_kwh": self.config.capacity_kwh,
        }

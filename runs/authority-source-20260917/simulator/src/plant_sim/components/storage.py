"""Gas storage: a tank with a level and a hard ceiling."""

from __future__ import annotations

from typing import Any

from ..config import StorageConfig
from .base import Component


class GasStorage(Component):
    """Mass buffer, deliberately species-agnostic.

    The H2 tank is one instance; a CH4 tank, a CO2 buffer or a water tank are the
    same object with a different `name`, `species` and `StorageConfig`.

    Extension point: pressure/temperature, boil-off and leakage, offtake demand.
    """

    def __init__(self, config: StorageConfig | None = None, *, name: str = "h2_tank", species: str = "H2") -> None:
        self.name = name
        self.species = species
        self.config = config or StorageConfig()
        self.level_kg = 0.0
        self.reset()

    def reset(self) -> None:
        self.level_kg = min(self.config.initial_kg, self.config.capacity_kg)

    @property
    def capacity_kg(self) -> float:
        return self.config.capacity_kg

    @property
    def headroom_kg(self) -> float:
        return max(0.0, self.config.capacity_kg - self.level_kg)

    @property
    def fill(self) -> float:
        return self.level_kg / self.config.capacity_kg if self.config.capacity_kg else 0.0

    def store(self, mass_kg: float) -> float:
        """Add mass; returns what actually fitted (the rest is refused, not vented)."""
        accepted = min(max(0.0, mass_kg), self.headroom_kg)
        self.level_kg += accepted
        return accepted

    def withdraw(self, mass_kg: float) -> float:
        """Remove mass; returns what was actually available. For future offtake."""
        delivered = min(max(0.0, mass_kg), self.level_kg)
        self.level_kg -= delivered
        return delivered

    def snapshot(self) -> dict[str, Any]:
        return {
            "species": self.species,
            "level_kg": self.level_kg,
            "capacity_kg": self.capacity_kg,
            "fill": self.fill,
        }

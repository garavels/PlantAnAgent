"""plant_sim -- digital twin of an intermittently powered hydrogen plant.

    from plant_sim import Action, PlantSimulator

    sim = PlantSimulator()
    state = sim.reset()
    state = sim.step(Action(battery_share=0.3, electrolyser_share=0.7))

Pure Python, no I/O, no framework. Deterministic: same actions, same states.
"""

from .allocation import PowerFlows, dispatch
from .components import Battery, Component, Electrolyser, GasStorage, SolarArray
from .config import (
    BatteryConfig,
    ElectrolyserConfig,
    PlantConfig,
    SolarConfig,
    StorageConfig,
)
from .simulator import PlantSimulator
from .state import Action, PlantState

__version__ = "0.1.0"

__all__ = [
    "Action",
    "Battery",
    "BatteryConfig",
    "Component",
    "Electrolyser",
    "ElectrolyserConfig",
    "GasStorage",
    "PlantConfig",
    "PlantSimulator",
    "PlantState",
    "PowerFlows",
    "SolarArray",
    "SolarConfig",
    "StorageConfig",
    "dispatch",
]

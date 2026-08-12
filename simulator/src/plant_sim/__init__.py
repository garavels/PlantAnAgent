"""plant_sim -- digital twin of an intermittently powered synthetic-methane plant.

    from plant_sim import Action, PlantSimulator

    sim = PlantSimulator()
    state = sim.reset()
    state = sim.step(Action(battery_share=0.2, electrolyser_share=0.5, dac_share=0.2))

Pure Python, no I/O, no framework. Deterministic: same actions, same states.
"""

from .allocation import PowerFlows, dispatch
from .components import (
    Battery,
    Component,
    DirectAirCapture,
    Electrolyser,
    GasStorage,
    Methanator,
    SolarArray,
)
from .config import (
    BatteryConfig,
    DACConfig,
    ElectrolyserConfig,
    MethanatorConfig,
    PlantConfig,
    SolarConfig,
    StorageConfig,
)
from .simulator import PlantSimulator
from .state import Action, PlantState

__version__ = "0.3.0"

__all__ = [
    "Action",
    "Battery",
    "BatteryConfig",
    "Component",
    "DACConfig",
    "DirectAirCapture",
    "Electrolyser",
    "ElectrolyserConfig",
    "GasStorage",
    "Methanator",
    "MethanatorConfig",
    "PlantConfig",
    "PlantSimulator",
    "PlantState",
    "PowerFlows",
    "SolarArray",
    "SolarConfig",
    "StorageConfig",
    "dispatch",
]

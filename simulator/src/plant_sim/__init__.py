"""plant_sim -- digital twin of an intermittently powered synthetic-methane plant.

    from plant_sim import Action, PlantSimulator

    sim = PlantSimulator()
    state = sim.reset()
    state = sim.step(Action(battery_share=0.2, electrolyser_share=0.5, dac_share=0.2))

Pure Python, no I/O, no framework. Deterministic: same actions, same states.

Abnormal operation lives beside it: `sim.inject_fault(...)` and
`sim.load_scenario(...)` change what the plant does, `sim.events` says what it
noticed, and `sim.get_telemetry()` is what an observer is allowed to see.
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
from .delivery import (
    DeliveryCommitment,
    DeliveryLedger,
    DeliverySettlement,
    DeliveryStepResult,
)
from .events import Event, EventLevel, EventLog
from .faults import (
    CATALOGUE as FAULT_CATALOGUE,
)
from .faults import (
    ComponentEffect,
    Fault,
    FaultKind,
    FaultRegistry,
    FaultSpec,
)
from .scenarios import SCENARIOS, InitialState, Scenario
from .simulator import PlantSimulator
from .state import Action, ComponentHealth, PlantState
from .telemetry import SENSOR_CHANNELS, observe

__version__ = "0.4.0"

__all__ = [
    "Action",
    "Battery",
    "BatteryConfig",
    "Component",
    "ComponentEffect",
    "ComponentHealth",
    "DACConfig",
    "DirectAirCapture",
    "DeliveryCommitment",
    "DeliveryLedger",
    "DeliverySettlement",
    "DeliveryStepResult",
    "Electrolyser",
    "ElectrolyserConfig",
    "Event",
    "EventLevel",
    "EventLog",
    "FAULT_CATALOGUE",
    "Fault",
    "FaultKind",
    "FaultRegistry",
    "FaultSpec",
    "GasStorage",
    "InitialState",
    "Methanator",
    "MethanatorConfig",
    "PlantConfig",
    "PlantSimulator",
    "PlantState",
    "PowerFlows",
    "SCENARIOS",
    "SENSOR_CHANNELS",
    "Scenario",
    "SolarArray",
    "SolarConfig",
    "StorageConfig",
    "dispatch",
    "observe",
]

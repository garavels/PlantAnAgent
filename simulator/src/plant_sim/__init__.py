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
from .commitment import (
    COMMITMENT_DYNAMICS_VERSION,
    DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE,
    ElectrolyserCommitmentProfile,
    ElectrolyserCommitmentSnapshot,
    ElectrolyserCommitmentState,
    ElectrolyserCommitmentStep,
    ElectrolyserMode,
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
from .forcing import SOLAR_FORCING_VERSION, SolarFactorTape
from .scenarios import SCENARIOS, InitialState, Scenario
from .simulator import PlantSimulator
from .state import Action, ComponentHealth, PlantState
from .telemetry import SENSOR_CHANNELS, observe

__version__ = "0.5.0"

__all__ = [
    "Action",
    "Battery",
    "BatteryConfig",
    "Component",
    "ComponentEffect",
    "ComponentHealth",
    "COMMITMENT_DYNAMICS_VERSION",
    "DACConfig",
    "DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE",
    "DirectAirCapture",
    "DeliveryCommitment",
    "DeliveryLedger",
    "DeliverySettlement",
    "DeliveryStepResult",
    "Electrolyser",
    "ElectrolyserCommitmentProfile",
    "ElectrolyserCommitmentSnapshot",
    "ElectrolyserCommitmentState",
    "ElectrolyserCommitmentStep",
    "ElectrolyserConfig",
    "ElectrolyserMode",
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
    "SOLAR_FORCING_VERSION",
    "Scenario",
    "SolarArray",
    "SolarConfig",
    "SolarFactorTape",
    "StorageConfig",
    "dispatch",
    "observe",
]

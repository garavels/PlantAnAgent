"""Controllers: telemetry in, `Action` out.

    heuristic   -- the default, and the baseline everything else is read against
    RL policy   -- V7, selectable once a checkpoint is loaded (`RLControllerSlot`)
    safe        -- V8, the protective mode a supervisor escalates into
    optimiser   -- later, same interface

V8 added the layer above them: `ControllerManager` owns which one is driving, the
envelope they all share, the log of every handover and a deterministic read on
how well the current one is doing (`health`).
"""

from .base import Constraints, ControlDecision, Controller
from .health import ControllerHealth, StepRecord
from .heuristic import HeuristicController
from .manager import ControllerManager, ControllerSwitch, OperatingState
from .rl import RL_CONTROLLER_ID, RLController, RLControllerSlot, RLStatus
from .safe import SAFE_CONTROLLER_ID, SafeController

__all__ = [
    "RL_CONTROLLER_ID",
    "SAFE_CONTROLLER_ID",
    "Constraints",
    "ControlDecision",
    "Controller",
    "ControllerHealth",
    "ControllerManager",
    "ControllerSwitch",
    "HeuristicController",
    "OperatingState",
    "RLController",
    "RLControllerSlot",
    "RLStatus",
    "SafeController",
    "StepRecord",
]

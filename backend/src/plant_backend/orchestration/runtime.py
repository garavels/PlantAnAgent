"""The orchestrator: the only thing that drives the plant.

One timestep, end to end:

    1. read telemetry                observation = session.telemetry
    2. decide                        AUTO -> controller, MANUAL -> operator setpoint
    3. validate                      safety.validate(action, observation, ...)
    4. apply                         session.step(validated.action)
    5. collect                       state / telemetry / events / faults
    6. broadcast                     handled by the clock, over SSE

The frontend does not appear anywhere in that list, which is the point of V5: it
observes and it asks, it does not drive. Whether the action came from the
heuristic controller or from a slider, it reaches the simulator by exactly one
path, through exactly one guard.

Where V6 plugged in: `self.supervisor` watches the same telemetry and events the
control room does, and steers by calling `set_mode` and `configure_constraints` --
the same two levers the operator has. It has no step-level hook, because a
supervisor that has to answer every 60 simulated seconds is not a supervisor; the
clock offers it each batch of frames (`SupervisorAgent.observe`) and it decides
whether anything is worth investigating.

Where the RL controller will plug in: `self.controllers`. A policy that implements
`Controller.compute_action` is registered there and selected with `set_controller`,
and it inherits the envelope, the guard and the supervisor unchanged.
"""

from __future__ import annotations

from typing import Any, Literal

from plant_sim import Action, PlantConfig, PlantState, Scenario

from ..agent import SupervisorAgent
from ..controllers import Constraints, Controller, HeuristicController
from ..safety import validate
from ..session import PlantSession

ControlMode = Literal["auto", "manual"]
CONTROL_MODES: tuple[ControlMode, ...] = ("auto", "manual")

#: Wall-clock milliseconds between advances while running.
TICK_MS = 200
#: Timesteps applied per tick, and the fastest we allow.
DEFAULT_SPEED = 4
MAX_SPEED = 64
#: Most timesteps one API call may advance.
MAX_STEPS = 1000
#: Trace entries carried in every payload. The full trace is a separate route.
AGENT_TRACE_IN_PAYLOAD = 12

#: Where the manual sliders start: the V1-V4 default allocation.
DEFAULT_MANUAL_ACTION = Action(
    battery_share=0.25,
    electrolyser_share=0.45,
    dac_share=0.2,
    methanator_share=0.1,
    battery_support_kw=600.0,
)


class PlantRuntime:
    """Owns the simulation loop, the control mode and the operating envelope."""

    def __init__(self, config: PlantConfig | None = None) -> None:
        self.session = PlantSession(config)
        self.controllers: dict[str, Controller] = {
            controller.id: controller for controller in (HeuristicController(),)
        }
        self.controller_id = HeuristicController.id
        self.mode: ControlMode = "auto"
        self.constraints = Constraints()
        self.manual_action = DEFAULT_MANUAL_ACTION
        #: The V6 supervisory agent. Disabled, the runtime behaves exactly as V5.
        self.supervisor = SupervisorAgent(self)

        self.playing = False
        self.speed = DEFAULT_SPEED
        #: Why the last action looked the way it did, for the control room.
        self.notes: tuple[str, ...] = ()
        self.adjustments: tuple[str, ...] = ()
        self.applied_action = Action.idle()

    @property
    def controller(self) -> Controller:
        return self.controllers[self.controller_id]

    @property
    def config(self) -> PlantConfig:
        return self.session.simulator.config

    # --- the loop ---------------------------------------------------------
    def advance(self, steps: int = 1) -> list[dict[str, Any]]:
        """Run `steps` timesteps of decide -> validate -> apply; returns the frames."""
        frames: list[dict[str, Any]] = []
        for _ in range(max(1, min(int(steps), MAX_STEPS))):
            observation = self.session.telemetry
            action, notes = self._decide(observation)
            validated = validate(
                action, observation, constraints=self.constraints, config=self.config
            )
            self.notes = notes
            self.adjustments = validated.adjustments
            self.applied_action = validated.action
            frames.extend(self.session.step(validated.action))
        return frames

    def _decide(self, observation: PlantState) -> tuple[Action, tuple[str, ...]]:
        if self.mode == "manual":
            return self.manual_action, ("Manual control: operator allocation",)
        decision = self.controller.compute_action(
            observation, constraints=self.constraints, config=self.config
        )
        return decision.action, decision.notes

    # --- simulation controls ----------------------------------------------
    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def set_speed(self, speed: int) -> int:
        self.speed = max(1, min(int(speed), MAX_SPEED))
        return self.speed

    def reset(self) -> dict[str, Any]:
        """Reset the plant *and* the orchestration around it.

        The operator's settings -- mode, controller, constraints, manual
        allocation -- are configuration and survive. Everything that describes a
        run in progress does not.
        """
        self.pause()
        for controller in self.controllers.values():
            controller.reset()
        self.notes = ()
        self.adjustments = ()
        self.applied_action = Action.idle()
        self.supervisor.reset()
        return self.session.reset()

    def load_scenario(self, scenario: Scenario | str) -> dict[str, Any]:
        self.pause()
        for controller in self.controllers.values():
            controller.reset()
        self.notes = ()
        self.adjustments = ()
        self.applied_action = Action.idle()
        self.supervisor.reset()
        return self.session.load_scenario(scenario)

    # --- control configuration --------------------------------------------
    def set_mode(self, mode: str) -> ControlMode:
        if mode not in CONTROL_MODES:
            raise ValueError(f"unknown control mode: {mode}")
        self.mode = mode  # type: ignore[assignment]  # guarded above
        return self.mode

    def set_controller(self, controller_id: str) -> str:
        if controller_id not in self.controllers:
            raise ValueError(f"unknown controller: {controller_id}")
        self.controller_id = controller_id
        return self.controller_id

    def configure_constraints(self, changes: dict[str, Any]) -> Constraints:
        """Tighten or relax the envelope. This is V6's main lever."""
        self.constraints = self.constraints.merged(changes)
        return self.constraints

    def set_manual_action(self, action: Action | dict[str, Any] | None) -> Action:
        """Store the operator's setpoint; it is validated when it is applied."""
        self.manual_action = Action.sanitised(action)
        return self.manual_action

    # --- reporting --------------------------------------------------------
    def control_status(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "controller": self.controller.to_dict(),
            "controllers": [controller.to_dict() for controller in self.controllers.values()],
            "constraints": self.constraints.to_dict(),
            "manual_action": self.manual_action.to_dict(),
            "applied_action": self.applied_action.to_dict(),
            "notes": list(self.notes),
            "adjustments": list(self.adjustments),
            "playing": self.playing,
            "speed": self.speed,
            "tick_ms": TICK_MS,
        }

    def payload(
        self, frames: list[dict[str, Any]] | None = None, *, replace: bool = False
    ) -> dict[str, Any]:
        """One self-contained update: state, new frames, conditions, control."""
        new_frames = frames if frames else [self.session.state]
        return {
            "state": new_frames[-1],
            "frames": new_frames,
            "replace": replace,
            "control": self.control_status(),
            "agent": self.supervisor.snapshot(trace_limit=AGENT_TRACE_IN_PAYLOAD),
            **self.session.conditions(),
        }

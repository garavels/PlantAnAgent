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

Where the RL controller plugged in (V7): `self.controllers` is the registry and
`self.rl` is the slot a checkpoint is loaded into. A loaded policy is registered
under `rl` and selected with `set_controller`, and it inherits the envelope, the
guard and the supervisor unchanged. With no checkpoint the registry holds the
heuristic controller alone -- which is why the heuristic is the default: there is
nothing else to select, and nothing is invented to fill the gap.
"""

from __future__ import annotations

from typing import Any, Literal

from plant_sim import Action, PlantConfig, PlantState, Scenario

from ..agent import SupervisorAgent
from ..controllers import Constraints, Controller, HeuristicController
from ..controllers.rl import RL_CONTROLLER_ID, RLController, RLControllerSlot
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
        #: The RL checkpoint slot. Empty unless one is configured and loads cleanly.
        self.rl = RLControllerSlot()
        self.rl.autoload()
        if self.rl.controller is not None:
            self.controllers[RL_CONTROLLER_ID] = self.rl.controller
        self.mode: ControlMode = "auto"
        self.constraints = Constraints()
        #: Envelope fields the supervisor overwrote, and what they were before.
        self._agent_envelope: dict[str, Any] = {}
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
        controller = self.controller
        decision = controller.compute_action(
            observation, constraints=self.constraints, config=self.config
        )
        if isinstance(controller, RLController) and controller.failure is not None:
            return self._after_rl_failure(observation, controller.failure)
        return decision.action, decision.notes

    def _after_rl_failure(
        self, observation: PlantState, reason: str
    ) -> tuple[Action, tuple[str, ...]]:
        """A policy that cannot answer stops driving: back to the heuristic, loudly.

        The failure is recorded on the slot, RL leaves the registry (so it cannot
        be reselected until a checkpoint is loaded again) and the note says what
        happened -- the plant keeps running on the controller that always can.
        """
        self.rl.mark_failed(reason)
        self.controllers.pop(RL_CONTROLLER_ID, None)
        self.controller_id = HeuristicController.id
        decision = self.controller.compute_action(
            observation, constraints=self.constraints, config=self.config
        )
        note = f"RL inference failed ({reason}): reverted to heuristic"
        return decision.action, (note, *decision.notes)

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
        run in progress does not, and that includes any envelope the supervisor
        retuned for an incident that no longer exists.
        """
        self.pause()
        for controller in self.controllers.values():
            controller.reset()
        self.notes = ()
        self.adjustments = ()
        self.applied_action = Action.idle()
        self.revert_agent_constraints()
        self.supervisor.reset()
        return self.session.reset()

    def load_scenario(self, scenario: Scenario | str) -> dict[str, Any]:
        self.pause()
        for controller in self.controllers.values():
            controller.reset()
        self.notes = ()
        self.adjustments = ()
        self.applied_action = Action.idle()
        self.revert_agent_constraints()
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
            if controller_id == RL_CONTROLLER_ID:
                # selectable only once a checkpoint is loaded; say why it is not
                raise ValueError(f"RL controller unavailable: {self.rl.status.reason}")
            raise ValueError(f"unknown controller: {controller_id}")
        self.controller_id = controller_id
        return self.controller_id

    def load_rl_checkpoint(self, path: str) -> dict[str, Any]:
        """Load an RL checkpoint and make it selectable. Raises on a bad checkpoint.

        Loading does not switch to it: selecting a controller stays a separate,
        deliberate act, so a checkpoint arriving never changes who is driving.
        """
        try:
            controller = self.rl.load(path)
        except ValueError:
            # a bad checkpoint must not leave the registry and the slot disagreeing
            # about whether RL can drive: whoever was driving carries on
            self._reconcile_rl()
            raise
        self.controllers[RL_CONTROLLER_ID] = controller
        return self.rl.status.to_dict()

    def _reconcile_rl(self) -> None:
        """Make the controller registry agree with the slot, and never strand RL."""
        if self.rl.controller is not None:
            self.controllers[RL_CONTROLLER_ID] = self.rl.controller
            return
        self.controllers.pop(RL_CONTROLLER_ID, None)
        if self.controller_id == RL_CONTROLLER_ID:
            self.controller_id = HeuristicController.id

    def unload_rl_checkpoint(self) -> dict[str, Any]:
        """Drop the policy; if it was driving, the heuristic takes over immediately."""
        self.controllers.pop(RL_CONTROLLER_ID, None)
        if self.controller_id == RL_CONTROLLER_ID:
            self.controller_id = HeuristicController.id
        return self.rl.unload().to_dict()

    def configure_constraints(
        self, changes: dict[str, Any], *, by: Literal["operator", "agent"] = "operator"
    ) -> Constraints:
        """Tighten or relax the envelope. This is V6's main lever.

        Who asked matters for exactly one reason: the operator's envelope is
        configuration and survives a reset, while a supervisor's retune belongs
        to the incident that caused it. So an agent write remembers what it
        replaced, and an operator write over the same field takes ownership back.
        """
        before = self.constraints.to_dict()
        self.constraints = self.constraints.merged(changes)
        after = self.constraints.to_dict()
        for field in changes:
            if field not in after or after[field] == before[field]:
                continue
            if by == "agent":
                self._agent_envelope.setdefault(field, before[field])
            else:
                self._agent_envelope.pop(field, None)
        return self.constraints

    def revert_agent_constraints(self) -> Constraints:
        """Put back whatever the supervisor overwrote. Called on reset."""
        if self._agent_envelope:
            self.constraints = self.constraints.merged(dict(self._agent_envelope))
            self._agent_envelope.clear()
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
            "controllers": self._controller_catalogue(),
            "rl": self.rl.status.to_dict(),
            "constraints": self.constraints.to_dict(),
            "manual_action": self.manual_action.to_dict(),
            "applied_action": self.applied_action.to_dict(),
            "notes": list(self.notes),
            "adjustments": list(self.adjustments),
            "playing": self.playing,
            "speed": self.speed,
            "tick_ms": TICK_MS,
        }

    def _controller_catalogue(self) -> list[dict[str, Any]]:
        """Every controller the control room may offer, available or not.

        RL appears even when it cannot be selected, carrying the reason: a
        disabled option with an explanation is honest, a missing option is not.
        """
        catalogue = [
            {**controller.to_dict(), "available": True, "state": "ready", "reason": ""}
            for controller_id, controller in self.controllers.items()
            if controller_id != RL_CONTROLLER_ID
        ]
        catalogue.append(self.rl.descriptor())
        return catalogue

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

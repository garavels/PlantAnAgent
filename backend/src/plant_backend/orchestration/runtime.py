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

Where V8 plugged in: that registry, the envelope and the fallback logic now live
in `ControllerManager` (`self.control`), which also grades whoever is driving.
The runtime kept the two jobs only it can do -- driving the clock and putting
every action through the guard -- so the step above is unchanged, and the layers
read exactly as the hierarchy does:

    supervisor -> self.control -> controller -> safety.validate -> simulator
"""

from __future__ import annotations

from typing import Any, Literal

from plant_sim import Action, PlantConfig, PlantState, Scenario

from ..agent import SupervisorAgent
from ..controllers import Constraints, ControlDecision, Controller
from ..controllers.manager import Actor, ControllerManager, ControllerSwitch
from ..controllers.rl import RLController, RLControllerSlot
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
        #: V8: who drives, inside what envelope, and how well they are doing it.
        self.control = ControllerManager(now=self._now)
        self.mode: ControlMode = "auto"
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
        return self.control.active

    @property
    def controllers(self) -> dict[str, Controller]:
        return self.control.controllers

    @property
    def controller_id(self) -> str:
        return self.control.active_id

    @controller_id.setter
    def controller_id(self, controller_id: str) -> None:
        """Kept for the pre-V8 spelling of `set_controller`; goes the same way."""
        self.set_controller(controller_id, reason="controller set directly")

    @property
    def rl(self) -> RLControllerSlot:
        return self.control.rl

    @property
    def constraints(self) -> Constraints:
        return self.control.constraints

    @property
    def config(self) -> PlantConfig:
        return self.session.simulator.config

    def _now(self) -> tuple[float, str]:
        """Simulated time, for anything that wants to stamp a log entry."""
        observation = self.session.telemetry
        return observation.time_s, observation.clock

    # --- the loop ---------------------------------------------------------
    def advance(self, steps: int = 1) -> list[dict[str, Any]]:
        """Run `steps` timesteps of decide -> validate -> apply; returns the frames."""
        frames: list[dict[str, Any]] = []
        for _ in range(max(1, min(int(steps), MAX_STEPS))):
            observation = self.session.telemetry
            decision = self._decide(observation)
            validated = validate(
                decision.action, observation, constraints=self.constraints, config=self.config
            )
            self.notes = decision.notes
            self.adjustments = validated.adjustments
            self.applied_action = validated.action
            seen = self._last_event_seq
            frames.extend(self.session.step(validated.action))
            # V8: grade the driver on the step it just asked for, from telemetry
            self.control.record_step(
                before=observation,
                after=self.session.telemetry,
                requested=decision.action,
                adjustments=validated.adjustments,
                config=self.config,
                invalid_action=decision.invalid,
                critical_events=self._critical_events_since(seen),
            )
        return frames

    def _decide(self, observation: PlantState) -> ControlDecision:
        if self.mode == "manual":
            return ControlDecision(
                self.manual_action, ("Manual control: operator allocation",)
            )
        controller = self.controller
        decision = self.control.decide(observation, config=self.config)
        if isinstance(controller, RLController) and controller.failure is not None:
            return self._after_rl_failure(observation, controller.failure)
        return decision

    def _after_rl_failure(self, observation: PlantState, reason: str) -> ControlDecision:
        """A policy that cannot answer stops driving: back to the heuristic, loudly.

        The manager records the failure, drops RL from the registry (so it cannot
        be reselected until a checkpoint is loaded again) and logs the handover;
        the note says what happened -- the plant keeps running on the controller
        that always can.
        """
        self.control.rl_failed(reason)
        decision = self.control.decide(observation, config=self.config)
        note = f"RL inference failed ({reason}): reverted to {self.controller.label.lower()}"
        return ControlDecision(decision.action, (note, *decision.notes), invalid=True)

    @property
    def _last_event_seq(self) -> int:
        recent = self.session.events(1)
        return int(recent[-1]["seq"]) if recent else 0

    def _critical_events_since(self, seq: int) -> int:
        return sum(
            1
            for event in self.session.events(20)
            if event["seq"] > seq and event["level"] == "critical"
        )

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
        self.control.reset()
        self.notes = ()
        self.adjustments = ()
        self.applied_action = Action.idle()
        self.supervisor.reset()
        return self.session.reset()

    def load_scenario(self, scenario: Scenario | str) -> dict[str, Any]:
        self.pause()
        self.control.reset()
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

    def set_controller(
        self,
        controller_id: str,
        *,
        reason: str = "operator selection",
        by: Actor = "operator",
        force: bool = False,
    ) -> str:
        """Hand the plant to another controller. Selection only: nothing is reset."""
        self.control.select(controller_id, reason=reason, by=by, force=force)
        return self.controller_id

    def switch_controller(
        self, controller_id: str, *, reason: str, by: Actor = "agent", force: bool = False
    ) -> ControllerSwitch | None:
        """The supervisor's version of `set_controller`: a reason, and the brakes on."""
        return self.control.select(controller_id, reason=reason, by=by, force=force)

    def set_operating_state(
        self, state: str, *, reason: str = "", by: Actor = "operator"
    ) -> str:
        """Normal, degraded or safe running. Safe mode puts the safe controller on."""
        return self.control.set_operating_state(state, reason=reason, by=by)

    def load_rl_checkpoint(self, path: str) -> dict[str, Any]:
        """Load an RL checkpoint and make it selectable. Raises on a bad checkpoint.

        Loading does not switch to it: selecting a controller stays a separate,
        deliberate act, so a checkpoint arriving never changes who is driving.
        """
        return self.control.load_rl(path)

    def unload_rl_checkpoint(self) -> dict[str, Any]:
        """Drop the policy; if it was driving, the heuristic takes over immediately."""
        return self.control.unload_rl()

    def configure_constraints(
        self, changes: dict[str, Any], *, by: Actor = "operator"
    ) -> Constraints:
        """Tighten or relax the shared envelope. This is V6's main lever.

        Held by the controller manager since V8, so one envelope reaches whoever
        is driving: the supervisor lowers a cap once and both the heuristic and
        the RL policy plan inside it, with no retraining.
        """
        return self.control.configure_constraints(changes, by=by)

    def revert_agent_constraints(self) -> Constraints:
        """Put back whatever the supervisor overwrote. Called on reset."""
        return self.control.revert_agent_constraints()

    def set_manual_action(self, action: Action | dict[str, Any] | None) -> Action:
        """Store the operator's setpoint; it is validated when it is applied."""
        self.manual_action = Action.sanitised(action)
        return self.manual_action

    # --- reporting --------------------------------------------------------
    def control_status(self) -> dict[str, Any]:
        hierarchy = self.control.status()
        return {
            "mode": self.mode,
            "controller": hierarchy["active"],
            "controllers": hierarchy["controllers"],
            "rl": self.rl.status.to_dict(),
            "constraints": self.constraints.to_dict(),
            "manual_action": self.manual_action.to_dict(),
            "applied_action": self.applied_action.to_dict(),
            "notes": list(self.notes),
            "adjustments": list(self.adjustments),
            "playing": self.playing,
            "speed": self.speed,
            "tick_ms": TICK_MS,
            #: V8: the hierarchy's own state, in one place the frontend can read.
            "hierarchy": hierarchy,
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

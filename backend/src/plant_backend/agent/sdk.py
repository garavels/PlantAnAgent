"""The OpenAI Agents SDK adapter: `AgentToolbox` methods as function tools.

Kept separate from `engines` so the SDK is imported in exactly one place, and so
the toolbox itself stays a plain Python object the tests and the rule engine can
call directly.

Two shaping decisions for the model's benefit:

* `set_controller_constraints` takes each envelope field as its own optional
  number rather than a free-form mapping. A flat signature is what tool-calling
  models get right, and it makes an invalid field name impossible to express.
* every tool returns the toolbox's own dict, including its `{"ok": False, ...}`
  refusals. A refused write is information the model should read and work with,
  not an exception that ends the run.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .tools import AgentToolbox


class SupervisorReport(BaseModel):
    """The structured conclusion the supervisor must finish with."""

    category: Literal[
        "normal",
        "controller_degradation",
        "critical_condition",
        "electrolyser_degradation",
        "dac_unavailable",
        "dac_degradation",
        "methanator_unavailable",
        "energy_shortfall",
        "battery_low",
        "feedstock_shortage",
        "sensor_inconsistency",
        "unclear",
    ] = Field(description="Best single label for what is happening.")
    component: str = Field(
        description="The unit involved, or an empty string if this is not about one unit."
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="How well the observations actually support the diagnosis."
    )
    summary: str = Field(
        description=(
            "Two or three sentences for the control room: what was observed, what it means, "
            "what was changed and why. Conclusions only, no reasoning steps."
        )
    )
    human_intervention_required: bool = Field(
        description="True if this cannot be handled by retuning the envelope alone."
    )


def report_model() -> type[SupervisorReport]:
    return SupervisorReport


def agent_tools(toolbox: AgentToolbox) -> list[Any]:
    """Wrap one run's toolbox as SDK function tools. Nothing else is exposed."""
    from agents import function_tool

    @function_tool
    def get_current_telemetry() -> dict[str, Any]:
        """Read the plant's current telemetry: power, production, levels, status."""
        return toolbox.get_current_telemetry()

    @function_tool
    def get_recent_telemetry(minutes: int = 60) -> dict[str, Any]:
        """Trend of the last `minutes` of telemetry, with earlier/recent means to compare."""
        return toolbox.get_recent_telemetry(minutes)

    @function_tool
    def get_recent_events(limit: int = 20) -> dict[str, Any]:
        """The plant's own recent events and alarms, oldest first."""
        return toolbox.get_recent_events(limit)

    @function_tool
    def inspect_component(component_id: str) -> dict[str, Any]:
        """Observed performance of one unit against its design figures.

        Args:
            component_id: One of solar, battery, electrolyser, dac, methanator,
                h2_tank, co2_tank, ch4_tank.
        """
        return toolbox.inspect_component(component_id)

    @function_tool
    def run_diagnostic(component_id: str = "plant") -> dict[str, Any]:
        """Cross-check inventories against flows to tell instrument faults from process faults.

        Args:
            component_id: A component name, or "plant" for the whole mass balance.
        """
        return toolbox.run_diagnostic(component_id)

    @function_tool
    def get_controller_status() -> dict[str, Any]:
        """Control mode, active controller, last applied action and its notes."""
        return toolbox.get_controller_status()

    @function_tool
    def get_active_controller() -> dict[str, Any]:
        """Who is driving the plant timestep by timestep, and what else could be."""
        return toolbox.get_active_controller()

    @function_tool
    def get_controller_metrics() -> dict[str, Any]:
        """Controller health: counts of mistakes over a rolling window, and a grade."""
        return toolbox.get_controller_metrics()

    @function_tool
    def get_controller_constraints() -> dict[str, Any]:
        """The operating envelope in force, with equipment ratings for context."""
        return toolbox.get_controller_constraints()

    @function_tool
    def set_controller_constraints(
        reason: str,
        electrolyser_max_kw: float | None = None,
        dac_max_kw: float | None = None,
        methanator_max_kw: float | None = None,
        battery_support_max_kw: float | None = None,
        battery_min_soc: float | None = None,
    ) -> dict[str, Any]:
        """Retune the envelope the controller plans inside. Your main lever.

        Pass only the fields you want to change; omitted fields are left alone.

        Args:
            reason: Short operational justification, logged with the change.
            electrolyser_max_kw: Cap on electrolyser power, kW.
            dac_max_kw: Cap on DAC power, kW.
            methanator_max_kw: Cap on methanation power, kW.
            battery_support_max_kw: Cap on battery discharge lent to the electrolyser, kW.
            battery_min_soc: Battery reserve nothing may eat into, 0..0.95.
        """
        changes = {
            "electrolyser_max_kw": electrolyser_max_kw,
            "dac_max_kw": dac_max_kw,
            "methanator_max_kw": methanator_max_kw,
            "battery_support_max_kw": battery_support_max_kw,
            "battery_min_soc": battery_min_soc,
        }
        given = {key: value for key, value in changes.items() if value is not None}
        return toolbox.set_controller_constraints(given, reason=reason)

    @function_tool
    def switch_controller(controller: str, reason: str) -> dict[str, Any]:
        """Hand the plant to a different controller. The plant keeps running throughout.

        Use it when the driver itself is the problem, not when the plant is: a
        degraded component is a reason to retune the envelope, repeatedly bad
        actions are a reason to change driver. May be refused while a cooldown is
        in force -- read the reason and leave it alone rather than retrying.

        Args:
            controller: "heuristic", "rl" or "safe".
            reason: Short operational justification, logged with the switch. Required.
        """
        return toolbox.switch_controller(controller, reason=reason)

    @function_tool
    def set_operating_state(state: str, reason: str) -> dict[str, Any]:
        """Set how the plant is run: normal, degraded or safe.

        Args:
            state: "normal" for ordinary running, "degraded" to record running on a
                deliberately tightened envelope, "safe" to put the safe controller
                on the plant during a critical condition.
            reason: Short operational justification, logged with the change. Required.
        """
        return toolbox.set_operating_state(state, reason=reason)

    @function_tool
    def simulate_strategy(
        controller: str | None = None,
        horizon_min: int = 120,
        electrolyser_max_kw: float | None = None,
        dac_max_kw: float | None = None,
        methanator_max_kw: float | None = None,
        battery_support_max_kw: float | None = None,
        battery_min_soc: float | None = None,
    ) -> dict[str, Any]:
        """Project one strategy on a copy of the plant. Changes nothing.

        Optional: use it to check a change you are unsure about, one option at a
        time. Omitted fields keep their live values. The projection is indicative,
        not a promise, so do not tune against small differences.

        Args:
            controller: Controller to project, or omit for the one driving now.
            horizon_min: Minutes to project, 15..240.
            electrolyser_max_kw: Cap on electrolyser power to project, kW.
            dac_max_kw: Cap on DAC power to project, kW.
            methanator_max_kw: Cap on methanation power to project, kW.
            battery_support_max_kw: Cap on battery discharge to project, kW.
            battery_min_soc: Battery reserve to project, 0..0.95.
        """
        changes = {
            "electrolyser_max_kw": electrolyser_max_kw,
            "dac_max_kw": dac_max_kw,
            "methanator_max_kw": methanator_max_kw,
            "battery_support_max_kw": battery_support_max_kw,
            "battery_min_soc": battery_min_soc,
        }
        given = {key: value for key, value in changes.items() if value is not None}
        return toolbox.simulate_strategy(
            controller=controller, constraints=given or None, horizon_min=horizon_min
        )

    @function_tool
    def set_operating_mode(mode: str, reason: str) -> dict[str, Any]:
        """Set the control mode.

        Args:
            mode: "auto" to let the controller drive, "manual" for operator setpoints.
            reason: Short operational justification, logged with the change.
        """
        return toolbox.set_operating_mode(mode, reason=reason)

    @function_tool
    def pause_plant(reason: str) -> dict[str, Any]:
        """Stop simulated time. Use when the plant should not keep running unattended."""
        return toolbox.pause_plant(reason=reason)

    @function_tool
    def resume_plant(reason: str) -> dict[str, Any]:
        """Resume simulated time after a pause."""
        return toolbox.resume_plant(reason=reason)

    @function_tool
    def request_safe_shutdown(reason: str, keep_running: bool = False) -> dict[str, Any]:
        """Take every process load to zero and hand the plant to a human. High impact.

        Args:
            reason: Why continuing is worse than stopping. Required.
            keep_running: True to leave simulated time running with loads at zero.
        """
        return toolbox.request_safe_shutdown(reason=reason, keep_running=keep_running)

    return [
        get_current_telemetry,
        get_recent_telemetry,
        get_recent_events,
        inspect_component,
        run_diagnostic,
        get_controller_status,
        get_active_controller,
        get_controller_metrics,
        get_controller_constraints,
        simulate_strategy,
        set_controller_constraints,
        switch_controller,
        set_operating_state,
        set_operating_mode,
        pause_plant,
        resume_plant,
        request_safe_shutdown,
    ]


__all__ = ["SupervisorReport", "agent_tools", "report_model"]

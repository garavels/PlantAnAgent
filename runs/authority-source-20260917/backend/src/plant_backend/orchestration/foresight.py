"""What would happen if: a short-horizon replay on a copy of the plant.

    forecast = look_ahead(runtime, controller_id="rl", constraints={...}, horizon_min=120)

The clone is a `deepcopy` of the live simulator, so the answer comes from the same
physics, the same faults and the same weather as the plant it was taken from, and
the live plant cannot be touched by asking the question: nothing here writes to
`runtime`, and the copy is discarded when the answer is returned.

Deliberately not an optimiser. It runs *one* option the supervisor names, for at
most `MAX_HORIZON_MIN` minutes, and reports six numbers about it. There is no
search, no scoring, no ranking and no recommendation, because a supervisor that
can search will search instead of thinking, and a four-hour open-loop projection
is not accurate enough to be optimised against -- it is accurate enough to tell
"this collapses the battery" from "this is fine".
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from plant_sim.state import CRITICAL, OFFLINE

from ..controllers import Constraints
from ..controllers.rl import RLController
from ..safety import validate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .runtime import PlantRuntime

#: The longest projection allowed. Beyond this the weather makes it fiction.
MAX_HORIZON_MIN = 240
#: The shortest projection worth running.
MIN_HORIZON_MIN = 15
#: Feedstock below this counts as a shortage (kg).
SHORTAGE_KG = 0.05


def look_ahead(
    runtime: "PlantRuntime",
    *,
    controller_id: str | None = None,
    constraints: dict[str, Any] | None = None,
    horizon_min: int = 120,
) -> dict[str, Any]:
    """Run one strategy on a copy of the plant and report how it went.

    `controller_id` and `constraints` default to what is driving the plant now,
    so the caller can vary one thing at a time. Raises `ValueError` for an
    unknown controller or an invalid envelope -- the same validation the live
    envelope gets, before anything is simulated.
    """
    manager = runtime.control
    controller_id = controller_id or manager.active_id
    if not manager.available(controller_id):
        raise ValueError(manager.unavailable(controller_id))
    envelope: Constraints = manager.constraints
    if constraints:
        envelope = envelope.merged(dict(constraints))

    horizon_min = max(MIN_HORIZON_MIN, min(int(horizon_min), MAX_HORIZON_MIN))
    config = runtime.config
    steps = max(1, int(horizon_min * 60.0 / config.dt_s))

    simulator = copy.deepcopy(runtime.session.simulator)
    controller = copy.deepcopy(manager.controllers[controller_id])
    controller.reset()
    if isinstance(controller, RLController):
        controller.failure = None

    start = simulator.get_telemetry()
    seen_seq = _last_seq(simulator)
    ch4_start = start.ch4_produced_total_kg
    min_soc = start.battery_soc
    corrections = 0
    h2_short = 0
    co2_short = 0
    shutdown_steps = 0
    failed: str | None = None

    for _ in range(steps):
        observation = simulator.get_telemetry()
        decision = controller.compute_action(
            observation, constraints=envelope, config=config
        )
        if isinstance(controller, RLController) and controller.failure is not None:
            failed = controller.failure
            break
        validated = validate(
            decision.action, observation, constraints=envelope, config=config
        )
        corrections += 1 if validated.adjustments else 0
        simulator.step(validated.action)
        after = simulator.get_telemetry()
        min_soc = min(min_soc, after.battery_soc)
        h2_short += 1 if after.h2_level_kg <= SHORTAGE_KG else 0
        co2_short += 1 if after.co2_level_kg <= SHORTAGE_KG else 0
        shutdown_steps += 1 if after.plant_status in {CRITICAL, OFFLINE} else 0

    end = simulator.get_telemetry()
    critical = [
        event["message"]
        for event in simulator.events.since(seen_seq)
        if event["level"] == "critical"
    ]
    return {
        "controller": controller_id,
        "horizon_min": horizon_min,
        "constraints": envelope.to_dict(),
        "from_clock": start.clock,
        "to_clock": end.clock,
        "predicted_ch4_kg": round(end.ch4_produced_total_kg - ch4_start, 2),
        "min_battery_soc": round(min_soc, 3),
        "h2_shortage_steps": h2_short,
        "co2_shortage_steps": co2_short,
        "shutdown_steps": shutdown_steps,
        "safety_corrections": corrections,
        "critical_events": critical,
        "end_plant_status": end.plant_status,
        "controller_failure": failed,
        "note": (
            "Open-loop projection on a copy of the plant: indicative, not a promise. "
            "The live plant was not touched."
        ),
    }


def _last_seq(simulator: Any) -> int:
    recent = simulator.events.recent(1)
    return int(recent[-1]["seq"]) if recent else 0


__all__ = ["MAX_HORIZON_MIN", "MIN_HORIZON_MIN", "look_ahead"]

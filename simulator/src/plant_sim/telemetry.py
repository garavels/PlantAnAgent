"""True plant state vs observed telemetry.

The simulator always knows the truth (`PlantSimulator.state`). What leaves the
simulator for a control room or an agent is an *observation* of that truth
(`PlantSimulator.telemetry`). With no sensor faults active the two are the same
object-shaped copy; a sensor fault biases one channel and they diverge:

    true H2 level    550 kg   (simulator.state.h2_level_kg)
    reported         720 kg   (simulator.telemetry.h2_level_kg)

Bias is expressed as a fraction of the channel's full scale, so a severity of
0.17 on a 1 000 kg tank over-reports by 170 kg. Keeping the mapping in one
table means a new observation channel is one entry, not a new framework.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Iterable

from .faults import SensorBias
from .state import PlantState

#: channel -> (full-scale field, fields it writes) for every biasable reading.
_CHANNELS: dict[str, Callable[[PlantState, float], dict[str, Any]]] = {
    "h2_level_kg": lambda state, bias: _tank(state.h2_level_kg, state.h2_capacity_kg, bias, "h2"),
    "co2_level_kg": lambda state, bias: _tank(state.co2_level_kg, state.co2_capacity_kg, bias, "co2"),
    "ch4_level_kg": lambda state, bias: _tank(state.ch4_level_kg, state.ch4_capacity_kg, bias, "ch4"),
    "battery_soc": lambda state, bias: _battery(state, bias),
}

#: The channels a `sensor_drift` fault can be pointed at.
SENSOR_CHANNELS = tuple(_CHANNELS)


def observe(state: PlantState, biases: Iterable[SensorBias]) -> PlantState:
    """Return the telemetry a consumer sees: `state` with sensor faults applied."""
    changes: dict[str, Any] = {}
    for bias in biases:
        apply = _CHANNELS.get(bias.channel)
        if apply is None or bias.severity == 0.0:
            continue
        changes.update(apply(state, bias.severity))
    if not changes:
        return state
    return replace(state, **changes)


def _tank(level_kg: float, capacity_kg: float, bias: float, species: str) -> dict[str, Any]:
    reported = _clamp(level_kg + bias * capacity_kg, 0.0, capacity_kg)
    return {
        f"{species}_level_kg": reported,
        f"{species}_fill": reported / capacity_kg if capacity_kg else 0.0,
    }


def _battery(state: PlantState, bias: float) -> dict[str, Any]:
    soc = _clamp(state.battery_soc + bias, 0.0, 1.0)
    return {"battery_soc": soc, "battery_energy_kwh": soc * state.battery_capacity_kwh}


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


__all__ = ["SENSOR_CHANNELS", "observe"]

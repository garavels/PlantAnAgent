"""What the supervisor is allowed to know, and how it is phrased.

Everything here is derived from *telemetry* -- the same observation the control
room gets, which a sensor fault is allowed to distort -- plus the static plant
design. Two deliberate omissions:

* the per-unit `degradation` number the simulator carries in `PlantState.health`
  is not passed on. A real plant does not publish "this stack is 40% derated";
  it publishes power in and hydrogen out, and someone works it out. The
  supervisor gets the observables and has to do the same.
* nothing reads `PlantSimulator.state`. There is no path from here to the
  simulator's private truth.

What *is* observable: whether a unit is available (a tripped unit says so),
what it was given, what it produced, and how that compares with its design
figures. `inspect()` and `consistency()` are where those comparisons live.
"""

from __future__ import annotations

from typing import Any

from plant_sim import PlantConfig, PlantState

#: Components the supervisor may ask about by name.
COMPONENTS = (
    "solar",
    "battery",
    "electrolyser",
    "dac",
    "methanator",
    "h2_tank",
    "co2_tank",
    "ch4_tank",
)

#: Tank fills at which clamping makes a mass balance meaningless.
_TANK_FULL = 0.995
_TANK_EMPTY = 0.005
#: Absolute floor on the mass-balance tolerance, in kg per timestep.
_BALANCE_FLOOR_KG = 1.0
#: Relative part of that tolerance, against the flows through the tank.
_BALANCE_FRACTION = 0.05


def telemetry_view(state: PlantState) -> dict[str, Any]:
    """The current plant as the supervisor sees it: one flat, compact reading."""
    return {
        "step": state.step,
        "clock": state.clock,
        "plant_status": state.plant_status,
        "solar_available_kw": _round(state.solar_available_kw),
        "solar_potential_kw": _round(state.solar_potential_kw),
        "solar_output_factor": _round(state.solar_output_factor, 3),
        "solar_curtailed_kw": _round(state.solar_curtailed_kw),
        "battery_soc": _round(state.battery_soc, 3),
        "battery_power_kw": _round(state.battery_power_kw),
        "battery_capacity_kwh": _round(state.battery_capacity_kwh),
        "electrolyser_power_kw": _round(state.electrolyser_power_kw),
        "electrolyser_load_frac": _round(state.electrolyser_load_frac, 3),
        "electrolyser_available": not _offline(state, "electrolyser"),
        "electrolyser_temperature_c": _round(state.electrolyser_temperature_c, 1),
        "h2_production_kg_h": _round(state.h2_production_kg_h, 2),
        "h2_level_kg": _round(state.h2_level_kg, 1),
        "h2_fill": _round(state.h2_fill, 3),
        "dac_power_kw": _round(state.dac_power_kw),
        "dac_available": not _offline(state, "dac"),
        "co2_capture_kg_h": _round(state.co2_capture_kg_h, 2),
        "co2_level_kg": _round(state.co2_level_kg, 1),
        "co2_fill": _round(state.co2_fill, 3),
        "methanator_power_kw": _round(state.methanator_power_kw),
        "methanator_load_frac": _round(state.methanator_load_frac, 3),
        "methanator_available": not _offline(state, "methanator"),
        "ch4_production_kg_h": _round(state.ch4_production_kg_h, 2),
        "ch4_level_kg": _round(state.ch4_level_kg, 1),
        "ch4_fill": _round(state.ch4_fill, 3),
    }


def history_view(frames: list[dict[str, Any]], points: int = 12) -> dict[str, Any]:
    """A thinned trend of the recent past, plus the means either side of it.

    `points` samples spread over the window keep the payload small enough to put
    in a prompt; `recent` and `earlier` are what a comparison actually needs.
    """
    if not frames:
        return {"points": [], "recent": {}, "earlier": {}, "span_h": 0.0}
    channels = (
        "solar_available_kw",
        "solar_output_factor",
        "battery_soc",
        "electrolyser_power_kw",
        "h2_production_kg_h",
        "co2_capture_kg_h",
        "ch4_production_kg_h",
        "h2_level_kg",
        "co2_level_kg",
    )
    half = max(1, len(frames) // 2)
    samples = frames[:: max(1, len(frames) // max(1, points))][-points:]
    return {
        "span_h": _round((frames[-1]["time_s"] - frames[0]["time_s"]) / 3600.0, 2),
        "points": [
            {"clock": frame["clock"], **{key: _round(frame[key], 2) for key in channels}}
            for frame in samples
        ],
        "earlier": _means(frames[:half], channels),
        "recent": _means(frames[half:], channels),
    }


def inspect(
    component: str,
    state: PlantState,
    frames: list[dict[str, Any]],
    config: PlantConfig,
) -> dict[str, Any]:
    """Observed performance of one unit against its design figures.

    The `*_ratio` fields are the interesting ones: 1.0 means the unit is doing
    what the data sheet says, and anything well below it is the evidence a
    diagnosis has to rest on.
    """
    if component not in COMPONENTS:
        raise ValueError(f"unknown component: {component!r} (known: {', '.join(COMPONENTS)})")
    reading: dict[str, Any] = {
        "component": component,
        "clock": state.clock,
        "available": not _offline(state, component),
    }
    if component == "solar":
        reading.update(
            available_kw=_round(state.solar_available_kw),
            clear_sky_kw=_round(state.solar_potential_kw),
            output_ratio=_round(state.solar_output_factor, 3),
            peak_kw=_round(config.solar.peak_kw),
            curtailed_kw=_round(state.solar_curtailed_kw),
        )
    elif component == "battery":
        reading.update(
            soc=_round(state.battery_soc, 3),
            energy_kwh=_round(state.battery_energy_kwh),
            power_kw=_round(state.battery_power_kw),
            usable_capacity_kwh=_round(state.battery_capacity_kwh),
            design_capacity_kwh=_round(config.battery.capacity_kwh),
            capacity_ratio=_ratio(state.battery_capacity_kwh, config.battery.capacity_kwh),
        )
    elif component == "electrolyser":
        design = config.electrolyser.specific_energy_kwh_per_kg
        observed = _specific_energy(state.electrolyser_power_kw, state.h2_production_kg_h)
        reading.update(
            power_kw=_round(state.electrolyser_power_kw),
            load_frac=_round(state.electrolyser_load_frac, 3),
            rated_kw=_round(config.electrolyser.rated_kw),
            min_load_kw=_round(config.electrolyser.min_load_frac * config.electrolyser.rated_kw),
            h2_production_kg_h=_round(state.h2_production_kg_h, 2),
            design_kwh_per_kg=design,
            observed_kwh_per_kg=_round(observed, 2) if observed is not None else None,
            yield_ratio=_round(design / observed, 3) if observed else None,
            temperature_c=_round(state.electrolyser_temperature_c, 1),
            nominal_temperature_c=config.electrolyser.nominal_temperature_c,
            h2_fill=_round(state.h2_fill, 3),
        )
    elif component == "dac":
        design = config.dac.specific_energy_kwh_per_kg
        observed = _specific_energy(state.dac_power_kw, state.co2_capture_kg_h)
        reading.update(
            power_kw=_round(state.dac_power_kw),
            load_frac=_round(state.dac_load_frac, 3),
            rated_kw=_round(config.dac.rated_kw),
            co2_capture_kg_h=_round(state.co2_capture_kg_h, 2),
            design_kwh_per_kg=design,
            observed_kwh_per_kg=_round(observed, 2) if observed is not None else None,
            yield_ratio=_round(design / observed, 3) if observed else None,
            co2_fill=_round(state.co2_fill, 3),
        )
    elif component == "methanator":
        recipe = config.methanator
        reading.update(
            power_kw=_round(state.methanator_power_kw),
            load_frac=_round(state.methanator_load_frac, 3),
            rated_kw=_round(recipe.rated_kw),
            ch4_production_kg_h=_round(state.ch4_production_kg_h, 2),
            h2_demand_kg_h=_round(state.ch4_production_kg_h * recipe.h2_kg_per_kg_ch4, 2),
            co2_demand_kg_h=_round(state.ch4_production_kg_h * recipe.co2_kg_per_kg_ch4, 2),
            h2_available_kg=_round(state.h2_level_kg, 1),
            co2_available_kg=_round(state.co2_level_kg, 1),
        )
    else:
        species = component.split("_")[0]
        level = float(getattr(state, f"{species}_level_kg"))
        capacity = float(getattr(state, f"{species}_capacity_kg"))
        reading.update(
            species=species.upper(),
            level_kg=_round(level, 1),
            capacity_kg=_round(capacity),
            fill=_round(getattr(state, f"{species}_fill"), 3),
            trend_kg_h=_trend_kg_h(frames, f"{species}_level_kg"),
        )
    return reading


def consistency(
    frames: list[dict[str, Any]], config: PlantConfig, tolerance_kg: float | None = None
) -> dict[str, Any]:
    """Do the reported inventories agree with the reported flows?

    Every tank is a mass balance the plant reports twice: as a level, and as the
    flows in and out of it. Sensor faults break the agreement -- a transmitter
    that starts over-reporting moves a level without any flow to explain it --
    so a large residual is evidence about the *instruments*, not the process.
    Tanks sitting at their limits are skipped: clamping is not a mismatch.
    """
    dt_h = config.dt_h
    recipe = config.methanator
    findings: list[dict[str, Any]] = []
    for species, produced, consumed in (
        ("h2", "h2_production_kg_h", recipe.h2_kg_per_kg_ch4),
        ("co2", "co2_capture_kg_h", recipe.co2_kg_per_kg_ch4),
        ("ch4", "ch4_production_kg_h", 0.0),
    ):
        worst = _worst_residual(frames, species, produced, consumed, dt_h, tolerance_kg)
        if worst is not None:
            findings.append(worst)
    return {
        "checked_steps": max(0, len(frames) - 1),
        "inconsistent": [f for f in findings if f["beyond_tolerance"]],
        "channels": findings,
    }


def _worst_residual(
    frames: list[dict[str, Any]],
    species: str,
    produced_key: str,
    per_ch4: float,
    dt_h: float,
    tolerance_kg: float | None,
) -> dict[str, Any] | None:
    worst: dict[str, Any] | None = None
    for previous, current in zip(frames, frames[1:]):
        fill = current[f"{species}_fill"]
        if fill >= _TANK_FULL or fill <= _TANK_EMPTY:
            continue
        flow_kg = (current[produced_key] - per_ch4 * current["ch4_production_kg_h"]) * dt_h
        reported_kg = current[f"{species}_level_kg"] - previous[f"{species}_level_kg"]
        residual = reported_kg - flow_kg
        limit = (
            tolerance_kg
            if tolerance_kg is not None
            else max(_BALANCE_FLOOR_KG, _BALANCE_FRACTION * abs(flow_kg))
        )
        if worst is None or abs(residual) > abs(worst["residual_kg"]):
            worst = {
                "channel": f"{species}_level_kg",
                "clock": current["clock"],
                "reported_change_kg": _round(reported_kg, 2),
                "flow_change_kg": _round(flow_kg, 2),
                "residual_kg": _round(residual, 2),
                "tolerance_kg": _round(limit, 2),
                "beyond_tolerance": abs(residual) > limit,
            }
    return worst


def _means(frames: list[dict[str, Any]], channels: tuple[str, ...]) -> dict[str, float]:
    if not frames:
        return {}
    return {key: _round(sum(f[key] for f in frames) / len(frames), 2) for key in channels}


def _trend_kg_h(frames: list[dict[str, Any]], key: str) -> float | None:
    """Average rate of change of a level over the window, in kg per hour."""
    if len(frames) < 2:
        return None
    span_h = (frames[-1]["time_s"] - frames[0]["time_s"]) / 3600.0
    if span_h <= 0.0:
        return None
    return _round((frames[-1][key] - frames[0][key]) / span_h, 2)


def _specific_energy(power_kw: float, production_kg_h: float) -> float | None:
    """kWh per kg actually observed, or `None` when the unit is not running."""
    if power_kw <= 0.0 or production_kg_h <= 0.0:
        return None
    return power_kw / production_kg_h


def _ratio(value: float, reference: float) -> float | None:
    return _round(value / reference, 3) if reference else None


def _offline(state: PlantState, component: str) -> bool:
    health = state.health.get(component)
    return bool(health.offline) if health is not None else False


def _round(value: Any, digits: int = 1) -> Any:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


__all__ = ["COMPONENTS", "consistency", "history_view", "inspect", "telemetry_view"]

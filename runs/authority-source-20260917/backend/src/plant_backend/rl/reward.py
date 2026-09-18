"""What "running the plant well" means, as a number.

The objective is the plant's objective: **make methane efficiently over time,
without operating it unsafely or unstably.** Everything else is a cost on the
ways an allocation policy can look productive while damaging the plant or
wasting the resource it is given.

    reward = + methane
             - curtailment          wasted renewable power
             - battery_drain        stored energy spent
             - reserve_breach       SOC pushed under the operating reserve
             - storage_saturation   tanks left against their ceiling
             - shutdown             plant in a critical/offline condition
             - correction           actions the guard had to clamp
             - cycling              units switched on/off for no gain

Two rules keep this honest. First, every term is read from a signal the
simulator already maintains -- cumulative counters where possible, so a term
cannot be double-counted or missed -- and never from a quantity this module
computes on its own. Second, every term is normalised by a *design* reference
(one timestep at nameplate), so a term is order-1 per step regardless of the
timestep length or the size of the plant, and the weights stay meaningful if
the plant is re-sized.

The weights below are a **starting point chosen for scale, not tuned against
training outcomes**: no policy has been trained yet, so any claim that these are
the right trade-offs would be invented. They are configuration
(`TrainingConfig.reward`) precisely so tuning them later is a config change and
not a code change.

`components` is returned through `PlantEnv.step`'s `info` so the contribution of
each term is visible per step -- the only way to tell a policy that learned to
make methane from one that learned to avoid the shutdown penalty by idling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

from plant_sim import PlantConfig, PlantState
from plant_sim.state import CRITICAL, OFFLINE

from ..controllers.base import Constraints

#: A tank at or above this fill counts as saturated: production has nowhere to go.
SATURATED_FILL = 0.98
#: Corrections per step that saturate the penalty (three consumers can be clamped).
CORRECTION_SCALE = 3.0
#: Units whose on/off transitions count as cycling.
CYCLED_UNITS = ("electrolyser", "dac", "methanator")


@dataclass(frozen=True)
class RewardWeights:
    """Multipliers on the normalised terms. Signs live in `compute`, not here."""

    #: CH4 produced this step / CH4 at rated methanator power. The only positive term.
    methane: float = 1.0
    #: Solar curtailed / solar generated this step. Wasting the resource is the
    #: main way a plant with a battery underperforms.
    curtailment: float = 0.25
    #: Battery energy discharged / one step at rated discharge. Small: discharging
    #: to hold the stack above its floor is *correct*, it is just not free.
    battery_drain: float = 0.05
    #: How far SOC ended below the reserve, in SOC points. Large: the reserve is
    #: what keeps the plant able to restart.
    reserve_breach: float = 2.0
    #: Fraction of tanks sitting at their ceiling. A full tank throttles the unit
    #: feeding it, so this is wasted capture/electrolysis waiting to happen.
    storage_saturation: float = 0.1
    #: 1.0 while the plant is `critical` or `offline`.
    shutdown: float = 0.5
    #: Guard adjustments this step / CORRECTION_SCALE, plus out-of-range output.
    #: Deliberately mild: the guard already made the action safe, so this teaches
    #: the policy to stop asking, it does not stand in for the guard.
    correction: float = 0.1
    #: Units that changed on/off state / number of units watched.
    cycling: float = 0.05

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "RewardWeights":
        if not values:
            return cls()
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown reward weights: {', '.join(sorted(unknown))}")
        return replace(cls(), **{key: float(value) for key, value in values.items()})


@dataclass(frozen=True)
class RewardBreakdown:
    """The scalar the policy learns from, plus what it was made of."""

    total: float
    #: Signed contributions, already weighted. They sum to `total`.
    components: dict[str, float] = field(default_factory=dict)
    #: The unweighted, normalised signals -- what actually happened.
    signals: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "components": dict(self.components),
            "signals": dict(self.signals),
        }


def _online(state: PlantState, unit: str) -> bool:
    return bool(
        {
            "electrolyser": state.electrolyser_online,
            "dac": state.dac_online,
            "methanator": state.methanator_online,
        }[unit]
    )


def compute(
    before: PlantState,
    after: PlantState,
    *,
    config: PlantConfig,
    constraints: Constraints,
    adjustments: tuple[str, ...] = (),
    out_of_range: bool = False,
    weights: RewardWeights | None = None,
) -> RewardBreakdown:
    """Score the timestep between `before` and `after`.

    Both states are the simulator's *truth*, not telemetry: a policy is rewarded
    for the methane the plant actually made, never for what a drifting sensor
    claims it made. Telemetry is what the policy *sees* (`observation.py`); this
    is what it is *graded on*, which is the same split a real plant has between
    its instruments and its production accounting.
    """
    weights = weights or RewardWeights()
    dt_h = config.dt_h

    ch4_reference_kg = (
        config.methanator.rated_kw * dt_h / config.methanator.specific_energy_kwh_per_kg
    )
    discharge_reference_kwh = config.battery.max_discharge_kw * dt_h

    ch4_kg = max(0.0, after.ch4_produced_total_kg - before.ch4_produced_total_kg)
    solar_kwh = max(0.0, after.energy_produced_kwh - before.energy_produced_kwh)
    curtailed_kwh = max(0.0, after.energy_curtailed_kwh - before.energy_curtailed_kwh)
    discharged_kwh = max(0.0, -after.battery_power_kw) * dt_h
    reserve_deficit = max(0.0, constraints.reserve_soc(config) - after.battery_soc)
    saturated = sum(
        1.0
        for fill in (after.h2_fill, after.co2_fill, after.ch4_fill)
        if fill >= SATURATED_FILL
    )
    cycled = sum(1.0 for unit in CYCLED_UNITS if _online(before, unit) != _online(after, unit))

    signals = {
        "methane": ch4_kg / ch4_reference_kg if ch4_reference_kg > 0.0 else 0.0,
        "curtailment": curtailed_kwh / solar_kwh if solar_kwh > 0.0 else 0.0,
        "battery_drain": (
            discharged_kwh / discharge_reference_kwh if discharge_reference_kwh > 0.0 else 0.0
        ),
        "reserve_breach": reserve_deficit,
        "storage_saturation": saturated / 3.0,
        "shutdown": 1.0 if after.plant_status in (CRITICAL, OFFLINE) else 0.0,
        "correction": min(1.0, (len(adjustments) + (1.0 if out_of_range else 0.0)) / CORRECTION_SCALE),
        "cycling": cycled / len(CYCLED_UNITS),
    }
    components = {
        "methane": weights.methane * signals["methane"],
        "curtailment": -weights.curtailment * signals["curtailment"],
        "battery_drain": -weights.battery_drain * signals["battery_drain"],
        "reserve_breach": -weights.reserve_breach * signals["reserve_breach"],
        "storage_saturation": -weights.storage_saturation * signals["storage_saturation"],
        "shutdown": -weights.shutdown * signals["shutdown"],
        "correction": -weights.correction * signals["correction"],
        "cycling": -weights.cycling * signals["cycling"],
    }
    return RewardBreakdown(
        total=float(sum(components.values())), components=components, signals=signals
    )


__all__ = ["RewardBreakdown", "RewardWeights", "compute"]

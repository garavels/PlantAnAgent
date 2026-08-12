"""A rule-based operator: enough judgement to run the plant unattended.

It is deliberately not optimal. It plans in kilowatts, in the order the plant
cares about, and only then converts the plan into the share-based `Action` the
simulator wants:

1. **Methanation first.** It is the point of the plant and the smallest load, so
   whenever both feedstocks are in the tanks it gets its power.
2. **Then CO2.** The DAC runs hard while the CO2 buffer is below target and
   trickles once it is comfortable -- CO2 starvation is what stops methanation.
3. **Then hydrogen.** Everything left goes to the stack, unless the H2 tank is
   full. Below the stack's minimum load the controller either borrows from the
   battery to lift it over the floor, or gives up and charges instead: half a
   stack is worth nothing.
4. **The rest charges the battery,** and when the pack is near its reserve the
   battery gets first claim on part of the bus instead of last.

Degraded units are still used -- they just produce less. Offline units are never
commanded. Nothing here reads the simulator: only telemetry, the constraints and
the plant's design.
"""

from __future__ import annotations

from plant_sim import Action, PlantConfig, PlantState

from .base import Constraints, ControlDecision, Controller

#: CO2 buffer the controller aims to keep in the tank (fraction of capacity).
CO2_TARGET_FILL = 0.5
#: A tank this full counts as full: stop making more of that gas.
TANK_FULL_FILL = 0.97
#: DAC effort once the CO2 buffer is at target, as a fraction of its cap.
CO2_TRICKLE = 0.3
#: SOC margin above the reserve where the controller starts protecting charge.
RECHARGE_MARGIN = 0.15
#: Share of the bus the battery claims first while it is near its reserve.
RECHARGE_CLAIM = 0.5


class HeuristicController(Controller):
    id = "heuristic"
    label = "Heuristic"
    description = (
        "Rule-based operator: methanates whenever both gases are available, keeps a CO2 "
        "buffer, sends the rest to the stack, and protects the battery reserve."
    )

    def compute_action(
        self,
        observation: PlantState,
        *,
        constraints: Constraints,
        config: PlantConfig,
    ) -> ControlDecision:
        notes: list[str] = []
        bus_kw = max(0.0, observation.solar_available_kw)
        reserve_soc = constraints.reserve_soc(config)
        offline = {name: health.offline for name, health in observation.health.items()}

        # the battery gets first claim while it is close to its reserve
        budget_kw = bus_kw
        recharge_kw = 0.0
        if observation.battery_soc < reserve_soc + RECHARGE_MARGIN:
            recharge_kw = budget_kw * RECHARGE_CLAIM
            budget_kw -= recharge_kw
            notes.append(f"Battery near reserve ({observation.battery_soc:.0%}): charging first")

        methanator_kw = self._methanator_kw(observation, constraints, config, budget_kw, offline, notes)
        budget_kw -= methanator_kw
        dac_kw = self._dac_kw(observation, constraints, config, budget_kw, offline, notes)
        budget_kw -= dac_kw

        stack_kw, support_kw = self._stack_kw(
            observation, constraints, config, budget_kw, reserve_soc, offline, notes
        )
        budget_kw -= stack_kw
        battery_kw = budget_kw + recharge_kw

        if bus_kw <= 0.0:
            # night: the only power available is whatever the battery lends the stack
            if support_kw > 0.0:
                notes.append("No solar: running the stack on battery support")
            else:
                notes.append("No solar: plant idle")
            return ControlDecision(
                Action(
                    battery_share=0.0,
                    electrolyser_share=0.0,
                    dac_share=0.0,
                    methanator_share=0.0,
                    battery_support_kw=support_kw,
                ),
                tuple(notes),
            )

        return ControlDecision(
            Action(
                battery_share=battery_kw / bus_kw,
                electrolyser_share=stack_kw / bus_kw,
                dac_share=dac_kw / bus_kw,
                methanator_share=methanator_kw / bus_kw,
                battery_support_kw=support_kw,
            ),
            tuple(notes),
        )

    # --- the rules --------------------------------------------------------
    def _methanator_kw(
        self,
        observation: PlantState,
        constraints: Constraints,
        config: PlantConfig,
        budget_kw: float,
        offline: dict[str, bool],
        notes: list[str],
    ) -> float:
        if offline.get("methanator", False):
            notes.append("Methanator offline: no methane while it is down")
            return 0.0
        if observation.ch4_fill >= TANK_FULL_FILL:
            notes.append("CH4 storage full: methanation paused")
            return 0.0
        recipe = config.methanator
        # one timestep of production has to be actually feedable
        per_step_kg = constraints.methanator_limit_kw(config) * config.dt_h / recipe.specific_energy_kwh_per_kg
        if (
            observation.h2_level_kg < per_step_kg * recipe.h2_kg_per_kg_ch4
            or observation.co2_level_kg < per_step_kg * recipe.co2_kg_per_kg_ch4
        ):
            # the reactor itself throttles to the scarcer gas; still worth running
            if observation.h2_level_kg <= 0.0 or observation.co2_level_kg <= 0.0:
                notes.append("Methanator starved: waiting for feedstock")
                return 0.0
        return min(constraints.methanator_limit_kw(config), max(0.0, budget_kw))

    def _dac_kw(
        self,
        observation: PlantState,
        constraints: Constraints,
        config: PlantConfig,
        budget_kw: float,
        offline: dict[str, bool],
        notes: list[str],
    ) -> float:
        if offline.get("dac", False):
            notes.append("DAC offline: CO2 buffer will drain")
            return 0.0
        if observation.co2_fill >= TANK_FULL_FILL:
            notes.append("CO2 storage full: capture paused")
            return 0.0
        cap_kw = constraints.dac_limit_kw(config)
        if observation.co2_fill >= CO2_TARGET_FILL:
            cap_kw *= CO2_TRICKLE
        else:
            notes.append(f"CO2 buffer low ({observation.co2_fill:.0%}): capturing at full rate")
        return min(cap_kw, max(0.0, budget_kw))

    def _stack_kw(
        self,
        observation: PlantState,
        constraints: Constraints,
        config: PlantConfig,
        budget_kw: float,
        reserve_soc: float,
        offline: dict[str, bool],
        notes: list[str],
    ) -> tuple[float, float]:
        """Power for the electrolyser, plus any battery support needed to run it."""
        if offline.get("electrolyser", False):
            notes.append("Electrolyser offline: no hydrogen production")
            return 0.0, 0.0
        if observation.h2_fill >= TANK_FULL_FILL:
            notes.append("H2 storage full: stack idle")
            return 0.0, 0.0

        stack_kw = min(constraints.electrolyser_limit_kw(config), max(0.0, budget_kw))
        min_load_kw = config.electrolyser.min_load_frac * config.electrolyser.rated_kw
        if stack_kw >= min_load_kw:
            return stack_kw, 0.0

        # below the floor: lend from the battery, but never from the reserve
        lendable_kw = 0.0
        if observation.battery_soc > reserve_soc:
            usable_kwh = (observation.battery_soc - reserve_soc) * observation.battery_capacity_kwh
            lendable_kw = min(constraints.support_limit_kw(config), usable_kwh / config.dt_h)
        support_kw = min(max(0.0, min_load_kw - stack_kw), lendable_kw)
        if stack_kw + support_kw < min_load_kw:
            notes.append("Not enough power for the stack floor: buffering instead")
            return 0.0, 0.0
        if support_kw > 0.0:
            notes.append(f"Battery lending {support_kw:.0f} kW to hold the stack above its floor")
        return stack_kw, support_kw

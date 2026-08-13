"""The controller for when the answer is "stop trying to be clever".

Safe mode is not a shutdown -- a shutdown is `Action.idle()` and the operator's
button. It is the smallest set of things a plant in trouble can usefully do:

    charge the battery       energy in the pack is the plant's only autonomy
    finish the feedstock     methanate what the tanks already hold, gently
    make nothing new         no electrolyser, no DAC, no battery lending

So it draws down risk (heat, stack cycling, deep discharge) without abandoning
the plant's purpose, and it needs no telemetry the other controllers do not have.
The supervisor selects it for a critical condition and leaves it when the
condition clears; validation still sits underneath, as it does for everyone.
"""

from __future__ import annotations

from plant_sim import Action, PlantConfig, PlantState

from .base import Constraints, ControlDecision, Controller

SAFE_CONTROLLER_ID = "safe"
#: Fraction of the methanator's cap safe mode is willing to run at.
SAFE_METHANATION_FRAC = 0.5
#: Feedstock below this is not worth starting the reactor for (kg).
SHORTAGE_KG = 0.05
#: A tank this full counts as full.
TANK_FULL_FILL = 0.97


class SafeController(Controller):
    id = SAFE_CONTROLLER_ID
    label = "Safe / Degraded"
    description = (
        "Protective mode: charges the battery, methanates existing feedstock gently, and "
        "starts no new hydrogen or CO2 production."
    )

    def compute_action(
        self,
        observation: PlantState,
        *,
        constraints: Constraints,
        config: PlantConfig,
    ) -> ControlDecision:
        notes: list[str] = ["Safe mode: protecting the plant, production is secondary"]
        bus_kw = max(0.0, observation.solar_available_kw)
        if bus_kw <= 0.0:
            notes.append("No solar: plant idle until sunrise")
            return ControlDecision(Action.idle(), tuple(notes))

        methanator_kw = 0.0
        offline = observation.health["methanator"].offline if "methanator" in observation.health else False
        fed = min(observation.h2_level_kg, observation.co2_level_kg) > SHORTAGE_KG
        if offline:
            notes.append("Methanator offline: charging only")
        elif not fed:
            notes.append("No feedstock to convert: charging only")
        elif observation.ch4_fill >= TANK_FULL_FILL:
            notes.append("CH4 storage full: charging only")
        else:
            methanator_kw = min(
                constraints.methanator_limit_kw(config) * SAFE_METHANATION_FRAC, bus_kw
            )
            notes.append(f"Converting stored feedstock at {methanator_kw:.0f} kW")

        return ControlDecision(
            Action(
                battery_share=max(0.0, (bus_kw - methanator_kw) / bus_kw),
                electrolyser_share=0.0,
                dac_share=0.0,
                methanator_share=methanator_kw / bus_kw,
                battery_support_kw=0.0,
            ),
            tuple(notes),
        )


__all__ = ["SAFE_CONTROLLER_ID", "SafeController"]

"""Headless run: `python -m plant_sim --hours 24`.

Applies one fixed action for the whole run and prints a telemetry table. Any
smarter choice of action is a *controller*, and controllers live in `backend/`.
"""

from __future__ import annotations

import argparse

from .simulator import PlantSimulator
from .state import Action

COLUMNS = (
    "  clock    solar   ->batt   ->elz   ->dac   ->met   batt%  "
    "H2 kg/h  H2 kg   CO2 kg/h  CO2 kg   CH4 kg/h  CH4 kg"
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="plant_sim", description=__doc__)
    parser.add_argument("--hours", type=float, default=24.0, help="simulated hours to run")
    parser.add_argument("--battery-share", type=float, default=0.25, help="fraction of solar to the battery")
    parser.add_argument("--electrolyser-share", type=float, default=0.45, help="fraction of solar to the stack")
    parser.add_argument("--dac-share", type=float, default=0.2, help="fraction of solar to the DAC")
    parser.add_argument("--methanator-share", type=float, default=0.1, help="fraction of solar to the methanator")
    parser.add_argument("--support-kw", type=float, default=0.0, help="battery discharge offered to the stack")
    parser.add_argument("--every", type=int, default=30, help="print one row every N steps")
    args = parser.parse_args()

    sim = PlantSimulator()
    action = Action(
        battery_share=args.battery_share,
        electrolyser_share=args.electrolyser_share,
        dac_share=args.dac_share,
        methanator_share=args.methanator_share,
        battery_support_kw=args.support_kw,
    )
    steps = int(args.hours * 3600 / sim.config.dt_s)

    print(COLUMNS)
    print("-" * len(COLUMNS))
    for index in range(steps):
        state = sim.step(action)
        if index % max(1, args.every) == 0:
            print(
                f"{state.clock:>9}"
                f"{state.solar_available_kw:9.0f}"
                f"{state.solar_to_battery_kw:9.0f}"
                f"{state.solar_to_electrolyser_kw:8.0f}"
                f"{state.solar_to_dac_kw:8.0f}"
                f"{state.solar_to_methanator_kw:8.0f}"
                f"{state.battery_soc * 100:8.1f}"
                f"{state.h2_production_kg_h:9.1f}"
                f"{state.h2_level_kg:8.1f}"
                f"{state.co2_capture_kg_h:10.1f}"
                f"{state.co2_level_kg:9.1f}"
                f"{state.ch4_production_kg_h:10.1f}"
                f"{state.ch4_level_kg:9.1f}"
            )

    final = sim.state
    print("-" * len(COLUMNS))
    print(
        f"  {final.ch4_produced_total_kg:.1f} kg CH4 from "
        f"{final.h2_produced_total_kg:.1f} kg H2 + {final.co2_captured_total_kg:.1f} kg CO2 | "
        f"{final.energy_produced_kwh:.0f} kWh solar | "
        f"{final.energy_curtailed_kwh:.0f} kWh curtailed "
        f"({final.energy_curtailed_kwh / max(1e-9, final.energy_produced_kwh):.1%}) | "
        f"battery {final.battery_soc:.1%}"
    )


if __name__ == "__main__":
    main()

"""Necessary physical feasibility conditions from saved exogenous episode facts.

This is a conservation-law bound, not a controller or a feasible schedule.
It gives a hypothetical controller all generated solar, lossless storage,
best-case conversion, no power limits, no faults and no startup overhead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

RESOURCES = ("battery", "h2", "co2", "ch4")
INITIAL_KEYS = ("battery_kwh", "h2_kg", "co2_kg", "ch4_kg")
RESOURCE_ALLOWANCE = 1e-5  # More permissive than 1e-6 + 1e-12 actual tolerance.
DELIVERY_ALLOWANCE_KG = 1e-5  # More permissive than 1e-9 per commitment.
FREE_ENERGY_KWH = 1.0  # Additional free energy per cumulative prefix.


def required_energy(delivered, initial, minimum, coefficients):
    """Lower bound on process electricity for cumulative deliveries and stocks."""
    ch4 = max(0.0, delivered + minimum["ch4"] - initial["ch4"])
    h2 = max(0.0, coefficients["h2_ratio"] * ch4 + minimum["h2"] - initial["h2"])
    co2 = max(0.0, coefficients["co2_ratio"] * ch4 + minimum["co2"] - initial["co2"])
    return math.fsum(
        (
            coefficients["h2_kwh_per_kg"] * h2,
            coefficients["co2_kwh_per_kg"] * co2,
            coefficients["ch4_kwh_per_kg"] * ch4,
        )
    )


def delivery_ceiling(available_kwh, requested_kg, initial, minimum, coefficients):
    """Upper bisection bracket, rounded outward, capped by actual contracts."""
    if required_energy(0, initial, minimum, coefficients) > available_kwh:
        raise ValueError("Even zero delivery cannot meet this stock minimum")
    if required_energy(requested_kg, initial, minimum, coefficients) <= available_kwh:
        return requested_kg
    low, high = 0.0, requested_kg
    for _ in range(100):
        middle = (low + high) / 2
        if middle == low or middle == high:
            break
        if required_energy(middle, initial, minimum, coefficients) <= available_kwh:
            low = middle
        else:
            high = middle
    return min(requested_kg, math.nextafter(high, math.inf))


def assess(scenario):
    plant = scenario["plant_config"]
    initial = {
        name: float(scenario["initial"][key])
        for name, key in zip(RESOURCES, INITIAL_KEYS)
    }
    profile = scenario.get("episode_config", {}).get(
        "electrolyser_commitment"
    ) or scenario.get("commitment_profile")
    h2_specific = plant["electrolyser"]["specific_energy_kwh_per_kg"]
    if profile:
        if any(point[1] < 1 for point in profile["specific_energy_multiplier_anchors"]):
            raise ValueError("Unreviewed efficiency multiplier below one")
        h2_specific = max(h2_specific, profile["nominal_specific_energy_kwh_per_kg"])
    coefficients = {
        "h2_kwh_per_kg": h2_specific,
        "co2_kwh_per_kg": plant["dac"]["specific_energy_kwh_per_kg"],
        "ch4_kwh_per_kg": plant["methanator"]["specific_energy_kwh_per_kg"],
        "h2_ratio": plant["methanator"]["h2_kg_per_kg_ch4"],
        "co2_ratio": plant["methanator"]["co2_kg_per_kg_ch4"],
    }
    if any(not math.isfinite(v) or v <= 0 for v in coefficients.values()):
        raise ValueError("Invalid conversion coefficients")
    nominal_specific = required_energy(1, initial, initial, coefficients)
    contracts = scenario["commitments"]
    total_requested = math.fsum(c["requested_kg"] for c in contracts)

    def due_hours(contract):
        if "due_hours" in contract:
            return contract["due_hours"]
        return contract["due_step"] * scenario["physics_dt_s"] / 3600

    days = scenario["days"]
    if len(days) != 7 or [d["cycle_index"] for d in days] != list(range(1, 8)):
        raise ValueError("This assessment requires seven contiguous daily boundaries")
    if not math.isclose(
        math.fsum(d["solar_available_kwh"] for d in days),
        scenario["weekly_solar_available_kwh"],
        abs_tol=1e-6,
    ):
        raise ValueError("Daily and weekly available solar disagree")
    prefixes = []
    for day in days:
        cycle = day["cycle_index"]
        targets = day["settlement"]["targets"]
        tolerances = day["settlement"]["tolerances"]
        if any(abs(targets[k] - initial[k]) > 1e-8 for k in RESOURCES):
            raise ValueError("Only fixed full-opening-stock returns are admitted")
        if any(tolerances[k] + 1e-12 > RESOURCE_ALLOWANCE for k in RESOURCES):
            raise ValueError("Assessment allowance must relax the actual tolerance")
        minimum = {k: max(0.0, targets[k] - RESOURCE_ALLOWANCE) for k in RESOURCES}
        due = [c for c in contracts if due_hours(c) <= cycle * 24 + 1e-9]
        requested = math.fsum(c["requested_kg"] for c in due)
        required_delivered = max(0.0, requested - len(due) * DELIVERY_ALLOWANCE_KG)
        solar = math.fsum(d["solar_available_kwh"] for d in days[:cycle])
        if not math.isfinite(solar) or solar < 0:
            raise ValueError("Invalid available solar energy")
        battery_gift = max(0.0, initial["battery"] - minimum["battery"])
        available = solar + battery_gift + FREE_ENERGY_KWH
        required = required_energy(required_delivered, initial, minimum, coefficients)
        ceiling = delivery_ceiling(available, requested, initial, minimum, coefficients)
        prefixes.append(
            {
                "cycle": cycle,
                "elapsed_hours": cycle * 24,
                "requested_kg": requested,
                "available_solar_kwh": solar,
                "minimum_process_kwh": required,
                "optimistic_energy_available_kwh": available,
                "energy_deficit_kwh": max(0.0, required - available),
                "full_service_impossible": required > available,
                "restored_inventory_delivery_ceiling_kg": ceiling,
                "implied_whole_week_delivery_ceiling_kg": min(
                    total_requested, ceiling + total_requested - requested
                ),
                "uniform_service_fraction_ceiling": ceiling / requested,
                "minimum_solar_energy_multiplier": max(
                    0.0, (required - battery_gift - FREE_ENERGY_KWH) / solar
                ),
            }
        )
    failed = [p for p in prefixes if p["full_service_impossible"]]
    weekly = prefixes[-1]
    zero = dict.fromkeys(RESOURCES, 0.0)
    # An even more generous diagnostic allows total depletion of all opening stocks.
    no_returns_available = (
        scenario["weekly_solar_available_kwh"] + initial["battery"] + FREE_ENERGY_KWH
    )
    no_returns_required = required_energy(
        total_requested - len(contracts) * DELIVERY_ALLOWANCE_KG,
        initial,
        zero,
        coefficients,
    )
    return {
        "seed": scenario["seed"],
        "evidence_kind": scenario["kind"],
        "source": scenario["source"],
        "initial": initial,
        "coefficients": coefficients,
        "ideal_replenished_specific_energy_kwh_per_kg": nominal_specific,
        "requested_kg": total_requested,
        "nominal_minimum_weekly_process_kwh": total_requested * nominal_specific,
        "weekly_available_solar_kwh": scenario["weekly_solar_available_kwh"],
        "weekly_conservative_energy_deficit_kwh": weekly["energy_deficit_kwh"],
        "impossible_with_all_returns": bool(failed),
        "first_impossible_daily_boundary": failed[0]["cycle"] if failed else None,
        "weekly_only_delivery_ceiling_kg": weekly[
            "restored_inventory_delivery_ceiling_kg"
        ],
        "all_prefix_delivery_ceiling_kg": min(
            p["implied_whole_week_delivery_ceiling_kg"] for p in prefixes
        ),
        "uniform_rate_ceiling_kg_h": total_requested
        / 168
        * min(p["uniform_service_fraction_ceiling"] for p in prefixes),
        "minimum_solar_energy_multiplier_all_prefixes": max(
            p["minimum_solar_energy_multiplier"] for p in prefixes
        ),
        "without_returns_weekly_energy_test": {
            "minimum_process_kwh": no_returns_required,
            "optimistic_available_kwh": no_returns_available,
            "energy_alone_rules_out_full_delivery": no_returns_required
            > no_returns_available,
            "interpretation": "Necessary condition only; passing does not establish a feasible schedule.",
        },
        "prefixes": prefixes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("facts", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.facts.read_bytes()
    facts = json.loads(raw)
    if facts["schema"] != "saved-plant-physical-feasibility-scenarios-v1":
        raise ValueError("Unsupported fact schema")
    scenarios = [assess(s) for s in facts["scenarios"]]
    result = {
        "schema": "plant-physical-feasibility-conservation-v1",
        "facts_sha256": hashlib.sha256(raw).hexdigest(),
        "analysis_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "allowances": {
            "resource_units": RESOURCE_ALLOWANCE,
            "per_delivery_kg": DELIVERY_ALLOWANCE_KG,
            "free_energy_per_cumulative_prefix_kwh": FREE_ENERGY_KWH,
        },
        "interpretation": "A violated necessary energy condition proves impossibility in the recorded simulator model. A passed condition or delivery ceiling is not a feasible schedule. Ceilings apply only with the stated stock-return conditions.",
        "scenarios": scenarios,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for s in scenarios:
        print(
            f"{s['seed']}: impossible={s['impossible_with_all_returns']}, first_day={s['first_impossible_daily_boundary']}, solar_MWh={s['weekly_available_solar_kwh']/1000:.3f}, conservative_gap_MWh={s['weekly_conservative_energy_deficit_kwh']/1000:.3f}, uniform_rate_ceiling={s['uniform_rate_ceiling_kg_h']:.3f} kg/h"
        )


if __name__ == "__main__":
    main()

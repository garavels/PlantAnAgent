"""Conservation-bound counterexamples, without simulator or optimization calls."""

import math

import pytest

from docs.research.tools.physical_feasibility import (
    assess,
    delivery_ceiling,
    required_energy,
)


COEFFICIENTS = {
    "h2_kwh_per_kg": 52.0,
    "co2_kwh_per_kg": 2.0,
    "ch4_kwh_per_kg": 1.2,
    "h2_ratio": 0.5,
    "co2_ratio": 2.75,
}
OPENING = {"battery": 4000.0, "h2": 300.0, "co2": 1500.0, "ch4": 240.0}
EMPTY = dict.fromkeys(OPENING, 0.0)


def test_restoration_cancels_the_opening_endowment():
    assert required_energy(6720, OPENING, OPENING, COEFFICIENTS) == pytest.approx(
        219744
    )
    another = {"battery": 100.0, "h2": 600.0, "co2": 3500.0, "ch4": 1000.0}
    assert required_energy(6720, another, another, COEFFICIENTS) == pytest.approx(
        219744
    )


def test_preexisting_methane_can_be_delivered_without_new_process_energy():
    assert required_energy(240, OPENING, EMPTY, COEFFICIENTS) == 0
    # After drawing the methane stock, the next 100 kg uses existing feedstock.
    assert required_energy(340, OPENING, EMPTY, COEFFICIENTS) == pytest.approx(120)


def test_limited_initial_hydrogen_cannot_be_counted_twice():
    opening = {**EMPTY, "h2": 5.0, "co2": 1000.0}
    # Producing 20 kg needs 10 kg H2; 5 kg still needs manufacture.
    assert required_energy(20, opening, EMPTY, COEFFICIENTS) == pytest.approx(
        5 * 52 + 20 * 1.2
    )


def test_energy_ceiling_is_outward_and_never_a_promised_feasible_yield():
    ceiling = delivery_ceiling(200000, 6720, OPENING, OPENING, COEFFICIENTS)
    assert ceiling >= 200000 / 32.7
    assert ceiling == pytest.approx(6116.207951070337)
    assert (
        required_energy(
            math.nextafter(ceiling, math.inf), OPENING, OPENING, COEFFICIENTS
        )
        > 200000
    )


def test_piecewise_ceiling_handles_free_starting_stock():
    ceiling = delivery_ceiling(120, 6720, OPENING, EMPTY, COEFFICIENTS)
    assert ceiling >= 340
    assert ceiling == pytest.approx(340)


def test_sufficient_energy_returns_only_contract_cap():
    assert delivery_ceiling(300000, 6720, OPENING, OPENING, COEFFICIENTS) == 6720


def test_impossible_stock_target_is_not_silently_treated_as_zero_service():
    with pytest.raises(ValueError):
        delivery_ceiling(0, 100, EMPTY, OPENING, COEFFICIENTS)


def test_later_sun_cannot_fix_an_already_failed_daily_boundary():
    solar = [10000, 42000, 42000, 42000, 42000, 42000, 42000]
    scenario = {
        "seed": "synthetic-prefix-counterexample",
        "kind": "unit_fixture",
        "source": {},
        "plant_config": {
            "electrolyser": {"specific_energy_kwh_per_kg": 52.0},
            "dac": {"specific_energy_kwh_per_kg": 2.0},
            "methanator": {
                "specific_energy_kwh_per_kg": 1.2,
                "h2_kg_per_kg_ch4": 0.5,
                "co2_kg_per_kg_ch4": 2.75,
            },
        },
        "initial": {
            "battery_kwh": 4000.0,
            "h2_kg": 300.0,
            "co2_kg": 1500.0,
            "ch4_kg": 240.0,
        },
        "weekly_solar_available_kwh": sum(solar),
        "commitments": [
            {"due_hours": (i + 1) * 6, "requested_kg": 240} for i in range(28)
        ],
        "days": [
            {
                "cycle_index": i + 1,
                "solar_available_kwh": value,
                "settlement": {
                    "targets": OPENING,
                    "tolerances": dict.fromkeys(OPENING, 1e-6),
                },
            }
            for i, value in enumerate(solar)
        ],
    }
    result = assess(scenario)
    assert result["weekly_conservative_energy_deficit_kwh"] == 0
    assert result["impossible_with_all_returns"]
    assert result["first_impossible_daily_boundary"] == 1
    assert result["all_prefix_delivery_ceiling_kg"] < 6720

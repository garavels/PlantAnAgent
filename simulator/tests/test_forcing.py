"""Contracts for episode-relative exogenous solar forcing."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from plant_sim import (
    Action,
    DeliveryCommitment,
    FaultSpec,
    PlantConfig,
    PlantSimulator,
    Scenario,
    SolarFactorTape,
)


NOON_SCENARIO = Scenario(
    id="forcing-noon",
    name="Forcing at noon",
    description="A bright start that makes solar factors directly observable.",
    start_hour=12.0,
)


def tape(*factors: float, dt_s: float = 60.0) -> SolarFactorTape:
    return SolarFactorTape(dt_s=dt_s, boundary_factors=factors)


def test_tape_is_immutable_and_indexes_boundaries_and_intervals() -> None:
    source = [0.1, 0.4, 0.9]
    forcing = SolarFactorTape(dt_s=60, boundary_factors=source)  # type: ignore[arg-type]
    source[1] = 0.8

    assert forcing.dt_s == 60.0
    assert forcing.boundary_factors == (0.1, 0.4, 0.9)
    assert forcing.boundary_count == 3
    assert forcing.interval_count == 2
    assert [forcing.at_boundary(index) for index in range(3)] == [0.1, 0.4, 0.9]
    assert [forcing.for_interval(index) for index in range(2)] == [0.4, 0.9]
    assert forcing.to_dict() == {
        "dt_s": 60.0,
        "boundary_factors": [0.1, 0.4, 0.9],
    }

    with pytest.raises(FrozenInstanceError):
        forcing.dt_s = 30.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"dt_s": 0.0, "boundary_factors": (1.0,)}, ValueError, "positive"),
        ({"dt_s": float("nan"), "boundary_factors": (1.0,)}, ValueError, "finite"),
        ({"dt_s": True, "boundary_factors": (1.0,)}, TypeError, "real number"),
        ({"dt_s": 60.0, "boundary_factors": ()}, ValueError, "boundary zero"),
        ({"dt_s": 60.0, "boundary_factors": (float("inf"),)}, ValueError, "finite"),
        ({"dt_s": 60.0, "boundary_factors": (float("nan"),)}, ValueError, "finite"),
        ({"dt_s": 60.0, "boundary_factors": (-0.01,)}, ValueError, "between 0 and 1"),
        ({"dt_s": 60.0, "boundary_factors": (1.01,)}, ValueError, "between 0 and 1"),
        ({"dt_s": 60.0, "boundary_factors": (True,)}, TypeError, "real number"),
        ({"dt_s": 60.0, "boundary_factors": ("0.5",)}, TypeError, "real number"),
    ],
)
def test_tape_validation_is_strict(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        SolarFactorTape(**kwargs)  # type: ignore[arg-type]


def test_hand_trace_uses_factor_zero_then_interval_end_factors() -> None:
    simulator = PlantSimulator()
    state0 = simulator.load_scenario(
        NOON_SCENARIO,
        solar_factor_tape=tape(0.2, 0.5, 0.8),
    )
    potential0 = state0.solar_potential_kw

    assert state0.step == 0
    assert state0.solar_output_factor == pytest.approx(0.2)
    assert state0.solar_available_kw == pytest.approx(potential0 * 0.2)

    state1 = simulator.step(Action.idle())
    state2 = simulator.step(Action.idle())

    assert state1.step == 1
    assert state1.solar_output_factor == pytest.approx(0.5)
    assert state1.solar_available_kw == pytest.approx(state1.solar_potential_kw * 0.5)
    assert state2.step == 2
    assert state2.solar_output_factor == pytest.approx(0.8)
    assert state2.solar_available_kw == pytest.approx(state2.solar_potential_kw * 0.8)


def test_absent_tape_and_all_ones_tape_are_exactly_equivalent() -> None:
    steps = 18
    plain = PlantSimulator()
    forced = PlantSimulator()
    plain_initial = plain.load_scenario("clear_day")
    forced_initial = forced.load_scenario(
        "clear_day",
        solar_factor_tape=tape(*(1.0 for _ in range(steps + 1))),
    )
    assert forced_initial.to_dict() == plain_initial.to_dict()
    schedule = [DeliveryCommitment(id="sample", due_step=9, requested_kg=2.0)]
    plain.set_delivery_schedule(schedule, freeze=True)
    forced.set_delivery_schedule(schedule, freeze=True)
    actions = [
        Action(battery_share=0.2, electrolyser_share=0.5, dac_share=0.2, methanator_share=0.1),
        Action(battery_share=0.1, electrolyser_share=0.6, dac_share=0.1, methanator_share=0.2),
        Action.idle(),
    ] * (steps // 3)

    plain_states = [plain.step(action).to_dict() for action in actions]
    forced_states = [forced.step(action).to_dict() for action in actions]

    assert forced_states == plain_states
    assert forced.events.events == plain.events.events
    assert forced.delivery_audit() == plain.delivery_audit()


def test_reset_with_the_same_tape_replays_exactly() -> None:
    forcing = tape(0.3, 0.8, 0.1, 1.0, 0.6)
    simulator = PlantSimulator()
    action = Action(battery_share=0.3, electrolyser_share=0.5, dac_share=0.2)
    runs: list[list[dict[str, object]]] = []

    for _ in range(2):
        simulator.reset(solar_factor_tape=forcing)
        runs.append([simulator.step(action).to_dict() for _ in range(4)])

    assert runs[0] == runs[1]


def test_factor_lookup_is_independent_of_scenario_start_clock() -> None:
    forcing = tape(0.25, 0.75)
    starts = []
    ends = []
    for start_hour in (6.0, 12.0, 18.0):
        simulator = PlantSimulator()
        state = simulator.load_scenario(
            Scenario(
                id=f"start-{start_hour}",
                name="Clock independence",
                description="Same forcing, different absolute clock.",
                start_hour=start_hour,
            ),
            solar_factor_tape=forcing,
        )
        starts.append(state.solar_output_factor)
        ends.append(simulator.step(Action.idle()).solar_output_factor)

    assert starts == pytest.approx([0.25, 0.25, 0.25])
    assert ends == pytest.approx([0.75, 0.75, 0.75])


def test_timestep_mismatch_and_tape_exhaustion_fail_loudly() -> None:
    simulator = PlantSimulator(PlantConfig(dt_s=60.0))

    with pytest.raises(ValueError, match="timestep mismatch"):
        simulator.reset(solar_factor_tape=tape(1.0, 1.0, dt_s=300.0))
    with pytest.raises(ValueError, match="timestep mismatch"):
        simulator.load_scenario(
            "clear_day",
            solar_factor_tape=tape(1.0, 1.0, dt_s=300.0),
        )

    simulator.reset(solar_factor_tape=tape(0.5, 0.7))
    simulator.step(Action.idle())
    state_before = simulator.state
    events_before = simulator.events.events
    with pytest.raises(IndexError, match="exhausted before interval"):
        simulator.step(Action.idle())
    assert simulator.state is state_before
    assert simulator.step_index == 1
    assert simulator.events.events == events_before


def test_resource_factor_multiplies_solar_fault_health_without_faking_degradation() -> None:
    resource_only = PlantSimulator()
    resource_state = resource_only.load_scenario(
        NOON_SCENARIO,
        solar_factor_tape=tape(0.5, 0.5),
    )
    assert resource_state.solar_output_factor == pytest.approx(0.5)
    assert resource_state.health["solar"].status == "nominal"
    assert resource_state.health["solar"].degradation == 0.0

    cloudy = PlantSimulator()
    cloudy_state = cloudy.load_scenario(
        Scenario(
            id="forcing-plus-fault",
            name="Forcing plus solar fault",
            description="Resource variability and equipment fault remain distinct.",
            start_hour=12.0,
            faults=(FaultSpec(type="solar_drop", severity=0.6, start_s=0.0),),
        ),
        solar_factor_tape=tape(0.5, 0.5),
    )

    assert cloudy.solar.resource_factor == pytest.approx(0.5)
    assert cloudy.solar.health == pytest.approx(0.4)
    assert cloudy_state.solar_output_factor == pytest.approx(0.2)
    assert cloudy_state.solar_available_kw == pytest.approx(
        cloudy_state.solar_potential_kw * 0.5 * 0.4
    )
    assert cloudy_state.health["solar"].status == "degraded"

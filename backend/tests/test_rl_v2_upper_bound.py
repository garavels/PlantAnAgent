"""Proof-oriented tests for the privileged Environment V2 delivery relaxation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from plant_sim import (
    Action,
    BatteryConfig,
    DACConfig,
    DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE,
    DeliveryCommitment,
    ElectrolyserCommitmentProfile,
    ElectrolyserConfig,
    ElectrolyserMode,
    FaultSpec,
    InitialState,
    MethanatorConfig,
    PlantConfig,
    PlantSimulator,
    Scenario,
    SolarArray,
    SolarConfig,
    StorageConfig,
)

from plant_backend.controllers import Constraints
from plant_backend.rl.v2 import PlantEnvV2
from plant_backend.rl.v2.config import EpisodeConfigV2, EpisodeV2, sample_episode
from plant_backend.rl.v2.contract import contract_digest, describe_checkpoint_contract
from plant_backend.rl.v2.reward import RewardWeightsV2
from plant_backend.rl.v2.upper_bound import (
    CERTIFICATE_SCOPE,
    SIMULATOR_EMBEDDING_STATUS,
    DeliveryRelaxationError,
    DualDeliveryCertificate,
    FiniteBoxLinearProgram,
    SparseCooMatrix,
    box_lagrangian_delivery_certificate,
    evaluate_program_residuals,
    next_float_up,
)


@dataclass(frozen=True)
class _Fixture:
    config: PlantConfig
    episode_config: EpisodeConfigV2
    constraints: Constraints
    episode: EpisodeV2
    checkpoint_contract: dict[str, Any]


def _plant(
    *,
    dt_s: float = 3_600.0,
    solar_peak_kw: float = 100_000.0,
    solar_sunrise_h: float = 0.0,
    solar_sunset_h: float = 24.0,
    battery_capacity_kwh: float = 100.0,
    battery_max_charge_kw: float = 100.0,
    battery_max_discharge_kw: float = 100.0,
    battery_charge_efficiency: float = 1.0,
    battery_discharge_efficiency: float = 1.0,
    electrolyser_rated_kw: float = 100_000.0,
    h2_capacity_kg: float = 100.0,
    dac_rated_kw: float = 100_000.0,
    co2_capacity_kg: float = 100.0,
    methanator_rated_kw: float = 100_000.0,
    ch4_capacity_kg: float = 100.0,
) -> PlantConfig:
    """Return a numerically simple plant with no equipment commitment model."""
    return PlantConfig(
        dt_s=dt_s,
        solar=SolarConfig(
            peak_kw=solar_peak_kw,
            sunrise_h=solar_sunrise_h,
            sunset_h=solar_sunset_h,
        ),
        battery=BatteryConfig(
            capacity_kwh=battery_capacity_kwh,
            max_charge_kw=battery_max_charge_kw,
            max_discharge_kw=battery_max_discharge_kw,
            charge_efficiency=battery_charge_efficiency,
            discharge_efficiency=battery_discharge_efficiency,
            min_soc=0.0,
            initial_soc=0.0,
        ),
        electrolyser=ElectrolyserConfig(
            rated_kw=electrolyser_rated_kw,
            min_load_frac=0.0,
            specific_energy_kwh_per_kg=52.0,
        ),
        h2_tank=StorageConfig(capacity_kg=h2_capacity_kg, initial_kg=0.0),
        dac=DACConfig(
            rated_kw=dac_rated_kw,
            specific_energy_kwh_per_kg=2.0,
        ),
        co2_tank=StorageConfig(capacity_kg=co2_capacity_kg, initial_kg=0.0),
        methanator=MethanatorConfig(
            rated_kw=methanator_rated_kw,
            specific_energy_kwh_per_kg=1.2,
            h2_kg_per_kg_ch4=0.5,
            co2_kg_per_kg_ch4=2.75,
        ),
        ch4_tank=StorageConfig(capacity_kg=ch4_capacity_kg, initial_kg=0.0),
    )


def _fixture(
    *,
    config: PlantConfig | None = None,
    steps: int = 1,
    start_hour: float = 12.0,
    initial_battery_soc: float = 0.0,
    initial_h2_kg: float = 0.0,
    initial_co2_kg: float = 0.0,
    initial_ch4_kg: float = 0.0,
    requested_kg: float = 1.0,
    delivery_period_steps: int | None = None,
    commitments: tuple[DeliveryCommitment, ...] | None = None,
    constraints: Constraints | None = None,
    electrolyser_commitment: ElectrolyserCommitmentProfile | None = None,
    seed: int = 7,
) -> _Fixture:
    plant = config or _plant()
    horizon_h = steps * plant.dt_h
    delivery_steps = steps if delivery_period_steps is None else delivery_period_steps
    delivery_period_h = delivery_steps * plant.dt_h
    episode_config = EpisodeConfigV2(
        horizon_h=horizon_h,
        decision_interval_s=plant.dt_s,
        start_hour=start_hour,
        delivery_period_h=delivery_period_h,
        delivery_rate_kg_h=requested_kg / delivery_period_h,
        battery_soc=(0.0, 0.0),
        h2_fill=(0.0, 0.0),
        co2_fill=(0.0, 0.0),
        ch4_coverage_h=(0.0, 0.0),
        electrolyser_commitment=electrolyser_commitment,
    )
    sampled = sample_episode(episode_config, plant, seed=seed)
    schedule = sampled.commitments if commitments is None else commitments
    scenario = Scenario(
        id="upper-bound-fixture",
        name="Upper-bound fixture",
        description="Deterministic hand-solvable LP test fixture.",
        start_hour=start_hour,
        initial=InitialState(
            battery_soc=initial_battery_soc,
            h2_kg=initial_h2_kg,
            co2_kg=initial_co2_kg,
            ch4_kg=initial_ch4_kg,
        ),
    )
    episode = replace(sampled, scenario=scenario, commitments=schedule)
    active_constraints = constraints or Constraints(battery_min_soc=0.0)
    checkpoint = describe_checkpoint_contract(
        plant=plant,
        episode=episode_config,
        constraints=active_constraints,
        weights=RewardWeightsV2(),
    )
    return _Fixture(
        config=plant,
        episode_config=episode_config,
        constraints=active_constraints,
        episode=episode,
        checkpoint_contract=checkpoint,
    )


def _build(fixture: _Fixture, **changes: Any) -> Any:
    from plant_backend.rl.v2.upper_bound import build_delivery_relaxation

    arguments: dict[str, Any] = {
        "episode": fixture.episode,
        "episode_config": fixture.episode_config,
        "config": fixture.config,
        "constraints": fixture.constraints,
        "checkpoint_contract": fixture.checkpoint_contract,
        "require_canonical_episode": False,
    }
    arguments.update(changes)
    return build_delivery_relaxation(**arguments)


def _solve(fixture: _Fixture, **changes: Any) -> Any:
    from plant_backend.rl.v2.upper_bound import solve_delivery_relaxation

    return solve_delivery_relaxation(_build(fixture), **changes)


def _published_bound(result: Any) -> float:
    value = result.dual_certified_delivery_upper_bound_kg
    assert value is not None, result
    return float(value)


def _one_variable_program(
    *,
    objective: float = -1.0,
    inequality_coefficient: float | None = None,
    inequality_rhs: float | None = None,
    equality_coefficient: float | None = None,
    equality_rhs: float | None = None,
    lower: float = 0.0,
    upper: float = 1.0,
) -> FiniteBoxLinearProgram:
    if inequality_coefficient is None:
        a_ub = SparseCooMatrix((0, 1), (), (), ())
        b_ub: tuple[float, ...] = ()
    else:
        assert inequality_rhs is not None
        a_ub = SparseCooMatrix((1, 1), (0,), (0,), (inequality_coefficient,))
        b_ub = (inequality_rhs,)
    if equality_coefficient is None:
        a_eq = SparseCooMatrix((0, 1), (), (), ())
        b_eq: tuple[float, ...] = ()
    else:
        assert equality_rhs is not None
        a_eq = SparseCooMatrix((1, 1), (0,), (0,), (equality_coefficient,))
        b_eq = (equality_rhs,)
    return FiniteBoxLinearProgram(
        objective=(objective,),
        a_ub=a_ub,
        b_ub=b_ub,
        a_eq=a_eq,
        b_eq=b_eq,
        lower_bounds=(lower,),
        upper_bounds=(upper,),
    )


def _exact_trace_vector(
    fixture: _Fixture,
    actions: tuple[Action, ...],
) -> tuple[Any, tuple[float, ...], float]:
    """Replay exact simulator flows and encode them into the relaxed LP vector."""
    assert len(actions) == fixture.episode.physics_steps
    model = _build(fixture)
    simulator = PlantSimulator(
        fixture.config,
        electrolyser_commitment=fixture.episode_config.electrolyser_commitment,
    )
    simulator.load_scenario(
        fixture.episode.scenario,
        solar_factor_tape=fixture.episode.weather.tape,
    )
    simulator.set_delivery_schedule(fixture.episode.commitments, freeze=True)

    initial = simulator.state
    states = []
    delivered_by_id: dict[str, float] = {}
    for action in actions:
        states.append(simulator.step(action))
        for settlement in simulator.delivery_step.settlements:
            delivered_by_id[settlement.id] = settlement.delivered_kg

    normalized = _normalize_trace(
        model,
        initial=initial,
        states=tuple(states),
        delivered_by_id=delivered_by_id,
    )
    return model, normalized, math.fsum(delivered_by_id.values())


def _normalize_trace(
    model: Any,
    *,
    initial: Any,
    states: tuple[Any, ...],
    delivered_by_id: dict[str, float],
) -> tuple[float, ...]:
    """Encode already-replayed physical states using the model's affine boxes."""

    physical = [0.0] * model.program.variable_count

    def assign(name: str, values: tuple[float, ...]) -> None:
        block = model.block(name)
        assert len(values) == block.size
        physical[block.start : block.stop] = values

    assign("solar_to_battery_kw", tuple(state.solar_to_battery_kw for state in states))
    assign(
        "solar_to_electrolyser_kw",
        tuple(state.solar_to_electrolyser_kw for state in states),
    )
    assign("solar_to_dac_kw", tuple(state.solar_to_dac_kw for state in states))
    assign(
        "solar_to_methanator_kw",
        tuple(state.solar_to_methanator_kw for state in states),
    )
    assign(
        "battery_to_electrolyser_kw",
        tuple(state.battery_to_electrolyser_kw for state in states),
    )
    assign("curtailed_solar_kw", tuple(state.solar_curtailed_kw for state in states))
    assign(
        "h2_produced_kg",
        tuple(state.h2_production_kg_h * model.dt_h for state in states),
    )
    assign(
        "co2_captured_kg",
        tuple(state.co2_capture_kg_h * model.dt_h for state in states),
    )
    assign(
        "ch4_produced_kg",
        tuple(state.ch4_production_kg_h * model.dt_h for state in states),
    )
    assign(
        "battery_energy_kwh",
        (initial.battery_energy_kwh,)
        + tuple(state.battery_energy_kwh for state in states),
    )
    assign(
        "h2_inventory_kg",
        (initial.h2_level_kg,) + tuple(state.h2_level_kg for state in states),
    )
    assign(
        "co2_inventory_kg",
        (initial.co2_level_kg,) + tuple(state.co2_level_kg for state in states),
    )
    assign(
        "ch4_inventory_kg",
        (initial.ch4_level_kg,) + tuple(state.ch4_level_kg for state in states),
    )
    delivery_values = tuple(
        delivered_by_id.get(commitment_id, 0.0)
        for commitment_id, _, _ in model.commitments
    )
    # The model uses one fixed-zero sentinel when there are no commitments.
    assign("delivered_ch4_kg", delivery_values or (0.0,))
    return model.normalized_vector(physical)


def test_sparse_coo_matrix_copies_and_canonicalizes_mutable_inputs() -> None:
    row_indices: list[object] = [np.int64(0), 1]
    column_indices: list[object] = [np.int32(1), 2]
    values: list[object] = [np.float64(1.25), Fraction(-1, 2)]

    matrix = SparseCooMatrix(
        (np.int64(2), np.int32(3)),
        row_indices,  # type: ignore[arg-type]
        column_indices,  # type: ignore[arg-type]
        values,  # type: ignore[arg-type]
    )

    assert matrix.shape == (2, 3)
    assert matrix.row_indices == (0, 1)
    assert matrix.column_indices == (1, 2)
    assert matrix.values == (1.25, -0.5)
    assert all(type(value) is int for value in (*matrix.shape, *matrix.row_indices))
    assert all(type(value) is int for value in matrix.column_indices)
    assert all(type(value) is float for value in matrix.values)

    row_indices[0] = 1
    column_indices[0] = 0
    values[0] = 99.0
    row_indices.append(1)
    column_indices.append(1)
    values.append(1.0)

    assert matrix.row_indices == (0, 1)
    assert matrix.column_indices == (1, 2)
    assert matrix.values == (1.25, -0.5)


def test_sparse_coo_matrix_accepts_one_shot_iterables() -> None:
    matrix = SparseCooMatrix(
        (2, 2),
        iter((0, 1)),  # type: ignore[arg-type]
        (column for column in (1, 0)),  # type: ignore[arg-type]
        map(float, (2, -3)),  # type: ignore[arg-type]
    )

    assert matrix.row_indices == (0, 1)
    assert matrix.column_indices == (1, 0)
    assert matrix.values == (2.0, -3.0)


@pytest.mark.parametrize(
    ("row_indices", "column_indices", "values"),
    (
        ([True], [0], [1.0]),
        ([0], [False], [1.0]),
        ([0.0], [0], [1.0]),
        ([0], [0.5], [1.0]),
        ([0], [0], [True]),
        ([0], [0], [float("nan")]),
        ([0], [0], [float("inf")]),
        ([0], [0], [-float("inf")]),
        ([0], [0], [0.0]),
        ([0], [0], [-0.0]),
    ),
)
def test_sparse_coo_matrix_rejects_noncanonical_coordinates_or_values(
    row_indices: list[object],
    column_indices: list[object],
    values: list[object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        SparseCooMatrix(
            (2, 2),
            row_indices,  # type: ignore[arg-type]
            column_indices,  # type: ignore[arg-type]
            values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("row_indices", "column_indices", "values", "message"),
    (
        ([0], [], [1.0], "equal length"),
        ([2], [0], [1.0], "outside"),
        ([0, 0], [0, 0], [1.0, 2.0], "unique and row-major sorted"),
        ([1, 0], [0, 1], [1.0, 2.0], "unique and row-major sorted"),
    ),
)
def test_sparse_coo_matrix_rejects_invalid_structure(
    row_indices: list[int],
    column_indices: list[int],
    values: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SparseCooMatrix(
            (2, 2),
            row_indices,  # type: ignore[arg-type]
            column_indices,  # type: ignore[arg-type]
            values,  # type: ignore[arg-type]
        )


def test_relaxation_uses_end_boundary_weather_factor_for_each_interval() -> None:
    fixture = _fixture(steps=2, seed=7)
    model = _build(fixture)
    solar = SolarArray(fixture.config.solar)
    start_s = fixture.episode.scenario.start_hour * 3_600.0
    expected = tuple(
        solar.clear_sky_kw(start_s + step * fixture.config.dt_s)
        * fixture.episode.weather.tape.for_interval(step)
        for step in range(fixture.episode.physics_steps)
    )
    wrong_boundary_zero = solar.clear_sky_kw(
        start_s
    ) * fixture.episode.weather.tape.at_boundary(0)

    assert tuple(model.solar_available_kw) == pytest.approx(expected, abs=1e-12)
    assert model.solar_available_kw[0] != pytest.approx(
        wrong_boundary_zero,
        rel=1e-6,
    )


def test_zero_resource_fixture_has_zero_delivery_ceiling() -> None:
    fixture = _fixture(start_hour=0.0)
    result = _solve(fixture)

    assert result.success is True
    assert result.numerical_primal_delivery_kg == pytest.approx(0.0, abs=1e-10)
    assert _published_bound(result) == pytest.approx(0.0, abs=1e-10)


def test_abundant_resource_fixture_fills_the_contract_from_same_step_production() -> (
    None
):
    fixture = _fixture(initial_h2_kg=0.5, initial_co2_kg=2.75)
    result = _solve(fixture)

    assert result.success is True
    assert result.numerical_primal_delivery_kg == pytest.approx(1.0, abs=1e-8)
    assert _published_bound(result) >= 1.0
    assert _published_bound(result) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize(
    ("tank", "capacity", "initial"),
    (
        ("h2", 0.25, 0.25),
        ("co2", 1.375, 1.375),
    ),
)
def test_pre_conversion_tank_headroom_prevents_same_minute_empty_and_refill(
    tank: str,
    capacity: float,
    initial: float,
) -> None:
    config = _plant(
        h2_capacity_kg=capacity if tank == "h2" else 100.0,
        co2_capacity_kg=capacity if tank == "co2" else 100.0,
    )
    fixture = _fixture(
        config=config,
        initial_h2_kg=initial if tank == "h2" else 0.5,
        initial_co2_kg=initial if tank == "co2" else 2.75,
    )
    result = _solve(fixture)

    # Production is stored before methanation in the simulator. A full tank
    # therefore refuses new material even if conversion would empty it later
    # in this same interval. Omitting the transient-headroom row gives 1 kg.
    assert result.success is True
    assert result.numerical_primal_delivery_kg == pytest.approx(0.5, abs=1e-8)
    assert _published_bound(result) == pytest.approx(0.5, abs=1e-8)


def test_exact_pre_conversion_headroom_is_available_in_the_same_interval() -> None:
    config = _plant(h2_capacity_kg=0.5)
    fixture = _fixture(
        config=config,
        initial_h2_kg=0.25,
        initial_co2_kg=2.75,
    )
    result = _solve(fixture)

    # The upper-yield relaxation may make exactly 0.25 kg, fill the H2 tank,
    # and then consume all 0.5 kg to produce and settle 1 kg CH4.
    assert result.success is True
    assert result.numerical_primal_delivery_kg == pytest.approx(1.0, abs=1e-8)
    assert _published_bound(result) == pytest.approx(1.0, abs=1e-8)


def test_full_methane_tank_cannot_be_refilled_before_same_step_settlement() -> None:
    config = _plant(ch4_capacity_kg=1.0)
    fixture = _fixture(
        config=config,
        initial_h2_kg=0.5,
        initial_co2_kg=2.75,
        initial_ch4_kg=1.0,
    )
    result = _solve(fixture)

    assert result.success is True
    assert result.witness is not None
    assert result.witness.delivered_kg == pytest.approx(1.0, abs=1e-9)
    assert result.witness.ch4_produced_kg == pytest.approx((0.0,), abs=1e-9)


def test_battery_support_obeys_bus_side_efficiency_and_only_feeds_electrolyser() -> (
    None
):
    config = _plant(
        solar_peak_kw=4.1,
        battery_capacity_kwh=52.0,
        battery_max_charge_kw=1.0,
        battery_max_discharge_kw=26.0,
        battery_discharge_efficiency=0.5,
    )
    charged = _fixture(
        config=config,
        initial_battery_soc=1.0,
        initial_co2_kg=2.75,
        seed=7,
    )
    empty = _fixture(
        config=config,
        initial_battery_soc=0.0,
        initial_co2_kg=2.75,
        seed=7,
    )
    charged_result = _solve(charged)
    empty_result = _solve(empty)
    solar = SolarArray(config.solar)
    empty_solar_kwh = (
        solar.clear_sky_kw(empty.episode.scenario.start_hour * 3_600.0)
        * empty.episode.weather.tape.for_interval(0)
        * config.dt_h
    )
    expected_without_battery = empty_solar_kwh / (
        config.electrolyser.specific_energy_kwh_per_kg
        * config.methanator.h2_kg_per_kg_ch4
        + config.methanator.specific_energy_kwh_per_kg
    )

    assert charged_result.success is True
    assert empty_result.success is True
    assert charged_result.numerical_primal_delivery_kg == pytest.approx(1.0, abs=1e-8)
    assert empty_result.numerical_primal_delivery_kg == pytest.approx(
        expected_without_battery,
        abs=1e-8,
    )
    assert _published_bound(charged_result) == pytest.approx(1.0, abs=1e-8)


def test_every_normalized_program_variable_has_an_explicit_finite_box() -> None:
    model = _build(_fixture(steps=2))
    program = model.program

    assert program.variable_count > 0
    assert np.asarray(program.objective).shape == (program.variable_count,)
    assert np.asarray(program.lower_bounds).shape == (program.variable_count,)
    assert np.asarray(program.upper_bounds).shape == (program.variable_count,)
    assert np.all(np.isfinite(program.objective))
    assert np.all(np.isfinite(program.lower_bounds))
    assert np.all(np.isfinite(program.upper_bounds))
    assert np.all(np.asarray(program.lower_bounds) <= np.asarray(program.upper_bounds))
    assert np.all(np.asarray(program.lower_bounds) == 0.0)
    assert np.all(np.asarray(program.upper_bounds) == 1.0)
    assert np.all(np.isfinite(model.physical_lower_bounds))
    assert np.all(np.isfinite(model.physical_upper_bounds))
    assert np.all(np.isfinite(model.physical_scales))
    assert np.all(
        np.asarray(model.physical_lower_bounds)
        <= np.asarray(model.physical_upper_bounds)
    )
    assert np.asarray(model.physical_scales) == pytest.approx(
        np.asarray(model.physical_upper_bounds)
        - np.asarray(model.physical_lower_bounds),
        abs=0.0,
    )
    assert np.all(np.isfinite(program.a_ub.values))
    assert np.all(np.isfinite(program.b_ub))
    assert np.all(np.isfinite(program.a_eq.values))
    assert np.all(np.isfinite(program.b_eq))
    assert model.certificate_scope == CERTIFICATE_SCOPE
    assert model.simulator_embedding_status == SIMULATOR_EMBEDDING_STATUS
    assert "encoded finite-box linear program only" in model.certificate_scope
    assert "numerically validated" in model.simulator_embedding_status
    assert "not bit-exact certified" in model.simulator_embedding_status


def test_untouched_sample_episode_is_verified_by_default() -> None:
    config = _plant(dt_s=60.0)
    episode_config = EpisodeConfigV2(
        horizon_h=1.0 / 60.0,
        decision_interval_s=60.0,
        start_hour=12.0,
        delivery_period_h=1.0 / 60.0,
        delivery_rate_kg_h=60.0,
        battery_soc=(0.5, 0.5),
        h2_fill=(0.005, 0.005),
        co2_fill=(0.0275, 0.0275),
        ch4_coverage_h=(0.0, 0.0),
        electrolyser_commitment=None,
    )
    constraints = Constraints(battery_min_soc=0.0)
    episode = sample_episode(episode_config, config, seed=13)
    checkpoint = describe_checkpoint_contract(
        plant=config,
        episode=episode_config,
        constraints=constraints,
        weights=RewardWeightsV2(),
    )
    from plant_backend.rl.v2.upper_bound import build_delivery_relaxation

    model = build_delivery_relaxation(
        episode=episode,
        episode_config=episode_config,
        config=config,
        constraints=constraints,
        checkpoint_contract=checkpoint,
    )
    from plant_backend.rl.v2.upper_bound import solve_delivery_relaxation

    result = solve_delivery_relaxation(model)

    assert model.canonical_episode_verified is True
    assert model.evidence_provenance_status == (
        "canonical sample_episode regeneration verified"
    )
    assert model.to_dict()["canonical_episode_verified"] is True
    assert model.to_dict()["evidence_provenance_status"] == (
        "canonical sample_episode regeneration verified"
    )
    assert result.success is True
    assert result.canonical_episode_verified is True
    assert result.evidence_provenance_status == (
        "canonical sample_episode regeneration verified"
    )


def test_default_canonical_check_rejects_a_mutated_sample_episode() -> None:
    config = _plant(dt_s=60.0)
    episode_config = EpisodeConfigV2(
        horizon_h=1.0 / 60.0,
        decision_interval_s=60.0,
        start_hour=12.0,
        delivery_period_h=1.0 / 60.0,
        delivery_rate_kg_h=60.0,
        battery_soc=(0.5, 0.5),
        h2_fill=(0.005, 0.005),
        co2_fill=(0.0275, 0.0275),
        ch4_coverage_h=(0.0, 0.0),
        electrolyser_commitment=None,
    )
    constraints = Constraints(battery_min_soc=0.0)
    canonical = sample_episode(episode_config, config, seed=13)
    assert canonical.scenario.initial.ch4_kg is not None
    mutated = replace(
        canonical,
        scenario=replace(
            canonical.scenario,
            initial=replace(
                canonical.scenario.initial,
                ch4_kg=canonical.scenario.initial.ch4_kg + 0.001,
            ),
        ),
    )
    checkpoint = describe_checkpoint_contract(
        plant=config,
        episode=episode_config,
        constraints=constraints,
        weights=RewardWeightsV2(),
    )
    from plant_backend.rl.v2.upper_bound import build_delivery_relaxation

    with pytest.raises(DeliveryRelaxationError, match="canonical sample_episode"):
        build_delivery_relaxation(
            episode=mutated,
            episode_config=episode_config,
            config=config,
            constraints=constraints,
            checkpoint_contract=checkpoint,
        )


def test_noncanonical_fixture_opt_out_is_serialized_and_not_evidence_verified() -> None:
    fixture = _fixture(initial_h2_kg=0.5, initial_co2_kg=2.75)
    model = _build(fixture)
    from plant_backend.rl.v2.upper_bound import solve_delivery_relaxation

    result = solve_delivery_relaxation(model)

    assert model.canonical_episode_verified is False
    assert model.evidence_provenance_status == (
        "noncanonical research fixture; not evidence-verified"
    )
    assert model.to_dict()["canonical_episode_verified"] is False
    assert model.to_dict()["evidence_provenance_status"] == (
        "noncanonical research fixture; not evidence-verified"
    )
    assert result.success is True
    assert result.canonical_episode_verified is False
    assert result.evidence_provenance_status == (
        "noncanonical research fixture; not evidence-verified"
    )
    assert result.to_dict()["canonical_episode_verified"] is False
    assert result.to_dict()["evidence_provenance_status"] == (
        "noncanonical research fixture; not evidence-verified"
    )


def test_relaxation_solve_is_deterministic_for_one_frozen_model() -> None:
    fixture = _fixture(steps=2, initial_h2_kg=0.5, initial_co2_kg=2.75)
    model = _build(fixture)
    from plant_backend.rl.v2.upper_bound import solve_delivery_relaxation

    first = solve_delivery_relaxation(model)
    second = solve_delivery_relaxation(model)

    assert first.success is True
    assert second.success is True
    assert first.model_sha256 == second.model_sha256 == model.model_sha256
    assert first.numerical_primal_delivery_kg == pytest.approx(
        second.numerical_primal_delivery_kg,
        abs=0.0,
    )
    assert first.dual_certified_delivery_upper_bound_kg == (
        second.dual_certified_delivery_upper_bound_kg
    )
    assert first.witness == second.witness


def test_endowment_preserving_result_is_explicitly_not_the_task_upper_bound() -> None:
    fixture = _fixture(initial_h2_kg=0.5, initial_co2_kg=2.75)
    model = _build(fixture, preserve_initial_endowment=True)
    from plant_backend.rl.v2.upper_bound import solve_delivery_relaxation

    result = solve_delivery_relaxation(model)

    assert model.preserve_initial_endowment is True
    assert model.is_task_upper_bound is False
    assert result.success is True
    assert result.preserve_initial_endowment is True
    assert result.is_task_upper_bound is False
    assert result.witness is not None
    assert result.witness.h2_inventory_kg[-1] >= model.initial_h2_kg - 1e-8
    assert result.witness.co2_inventory_kg[-1] >= model.initial_co2_kg - 1e-8
    assert result.witness.ch4_inventory_kg[-1] >= model.initial_ch4_kg - 1e-8
    assert (
        result.witness.battery_energy_kwh[-1] >= model.initial_battery_energy_kwh - 1e-8
    )


def test_exact_box_lagrangian_certificate_uses_scipy_inequality_sign() -> None:
    rhs = float(Fraction(1, 3))
    program = _one_variable_program(
        inequality_coefficient=1.0,
        inequality_rhs=rhs,
    )
    certificate = box_lagrangian_delivery_certificate(
        program,
        inequality_marginals=(-1.0,),
        equality_marginals=(),
        requested_kg=2.0,
    )

    # min -x, x <= rhs has y=-1 under SciPy's d(objective)/d(rhs)
    # convention. The certificate is evaluated over the stored IEEE float,
    # hence Fraction.from_float(rhs), rather than the symbolic one third.
    expected = Fraction.from_float(rhs)
    assert certificate.minimization_lower_bound == -expected
    assert certificate.certified_delivery_upper_bound == expected
    assert certificate.dual_certified_delivery_upper_bound_kg >= rhs
    assert certificate.positive_inequality_marginal_count == 0
    assert certificate.certificate_scope == CERTIFICATE_SCOPE
    assert certificate.to_dict()["certificate_scope"] == CERTIFICATE_SCOPE


def test_positive_inequality_marginal_is_projected_to_zero_and_remains_valid() -> None:
    program = _one_variable_program(
        inequality_coefficient=1.0,
        inequality_rhs=0.25,
    )
    certificate = box_lagrangian_delivery_certificate(
        program,
        inequality_marginals=(0.125,),
        equality_marginals=(),
        requested_kg=2.0,
    )

    # Projection drops the invalid positive multiplier. Box minimization then
    # gives the deliberately loose but still valid x <= 1 delivery ceiling.
    assert certificate.minimization_lower_bound == Fraction(-1, 1)
    assert certificate.certified_delivery_upper_bound == Fraction(1, 1)
    assert certificate.positive_inequality_marginal_count == 1
    assert certificate.max_positive_inequality_marginal == 0.125


def test_equality_multiplier_sign_and_box_minimization_are_exact() -> None:
    rhs = 0.125
    program = _one_variable_program(
        equality_coefficient=1.0,
        equality_rhs=rhs,
    )
    certificate = box_lagrangian_delivery_certificate(
        program,
        inequality_marginals=(),
        equality_marginals=(-1.0,),
        requested_kg=1.0,
    )

    assert certificate.minimization_lower_bound == -Fraction.from_float(rhs)
    assert certificate.certified_delivery_upper_bound == Fraction.from_float(rhs)


@pytest.mark.parametrize("value", (Fraction(1, 3), Fraction(1, 10), Fraction(1, 2)))
def test_outward_float_is_the_smallest_binary_float_not_below_exact_value(
    value: Fraction,
) -> None:
    outward = next_float_up(value)

    assert Fraction.from_float(outward) >= value
    assert Fraction.from_float(math.nextafter(outward, -math.inf)) < value


def test_requested_mass_is_summed_exactly_before_the_certificate_cap() -> None:
    program = _one_variable_program()
    certificate = box_lagrangian_delivery_certificate(
        program,
        inequality_marginals=(),
        equality_marginals=(),
        requested_kg=(0.1, 0.2),
    )
    exact_requested = Fraction.from_float(0.1) + Fraction.from_float(0.2)

    assert certificate.certified_delivery_upper_bound == exact_requested
    assert certificate.dual_certified_delivery_upper_bound_kg == next_float_up(
        exact_requested
    )
    assert exact_requested < Fraction.from_float(0.1 + 0.2)


@pytest.mark.parametrize(
    ("inequality_marginals", "equality_marginals", "message"),
    (
        ((), (), "inequality marginal count"),
        ((-1.0,), (0.0,), "equality marginal count"),
        ((float("nan"),), (), "inequality_marginals"),
    ),
)
def test_certificate_rejects_malformed_or_nonfinite_multipliers(
    inequality_marginals: tuple[float, ...],
    equality_marginals: tuple[float, ...],
    message: str,
) -> None:
    program = _one_variable_program(
        inequality_coefficient=1.0,
        inequality_rhs=0.5,
    )

    with pytest.raises(ValueError, match=message):
        box_lagrangian_delivery_certificate(
            program,
            inequality_marginals=inequality_marginals,
            equality_marginals=equality_marginals,
            requested_kg=1.0,
        )


def test_program_residuals_independently_detect_equation_inequality_and_box_errors() -> (
    None
):
    program = FiniteBoxLinearProgram(
        objective=(0.0, 0.0),
        a_ub=SparseCooMatrix((1, 2), (0, 0), (0, 1), (1.0, 1.0)),
        b_ub=(1.0,),
        a_eq=SparseCooMatrix((1, 2), (0, 0), (0, 1), (1.0, -1.0)),
        b_eq=(0.0,),
        lower_bounds=(0.0, 0.0),
        upper_bounds=(1.0, 1.0),
    )
    valid = evaluate_program_residuals(program, (0.5, 0.5))
    invalid = evaluate_program_residuals(program, (1.2, 0.1))

    assert valid.maximum == 0.0
    assert invalid.max_equality_absolute == pytest.approx(1.1)
    assert invalid.max_inequality_violation == pytest.approx(0.3)
    assert invalid.max_upper_bound_violation == pytest.approx(0.2)


def test_faulted_episode_fails_closed_instead_of_ignoring_capacity_loss() -> None:
    fixture = _fixture()
    faulted_scenario = replace(
        fixture.episode.scenario,
        faults=(
            FaultSpec(
                type="solar_drop",
                severity=0.5,
                start_s=0.0,
                duration_s=60.0,
            ),
        ),
    )
    faulted = replace(
        fixture, episode=replace(fixture.episode, scenario=faulted_scenario)
    )

    with pytest.raises(DeliveryRelaxationError, match="faulted episodes.*outside"):
        _build(faulted)


@pytest.mark.parametrize("field", ("battery_soc", "h2_kg", "co2_kg", "ch4_kg"))
def test_incomplete_sampled_initial_state_fails_closed(field: str) -> None:
    fixture = _fixture()
    incomplete_initial = replace(
        fixture.episode.scenario.initial,
        **{field: None},
    )
    incomplete = replace(
        fixture,
        episode=replace(
            fixture.episode,
            scenario=replace(fixture.episode.scenario, initial=incomplete_initial),
        ),
    )

    with pytest.raises(DeliveryRelaxationError, match="initial state is incomplete"):
        _build(incomplete)


def test_initial_energy_below_active_operator_reserve_fails_closed() -> None:
    fixture = _fixture(
        initial_battery_soc=0.25,
        constraints=Constraints(battery_min_soc=0.5),
    )

    with pytest.raises(DeliveryRelaxationError, match="below the active reserve"):
        _build(fixture)


def test_checkpoint_payload_or_digest_mismatch_fails_closed() -> None:
    fixture = _fixture()
    corrupted_digest = dict(fixture.checkpoint_contract)
    corrupted_digest["checkpoint_contract_digest"] = "0" * 64
    corrupted_payload = dict(fixture.checkpoint_contract)
    corrupted_payload["constraints"] = {
        **corrupted_payload["constraints"],
        "battery_min_soc": 0.75,
    }

    with pytest.raises(DeliveryRelaxationError, match="digest does not match"):
        _build(fixture, checkpoint_contract=corrupted_digest)
    with pytest.raises(DeliveryRelaxationError, match="digest does not match"):
        _build(fixture, checkpoint_contract=corrupted_payload)


def test_self_consistent_but_wrong_full_runtime_checkpoint_fails_closed() -> None:
    fixture = _fixture()
    wrong_runtime = dict(fixture.checkpoint_contract)
    wrong_runtime["environment_id"] = "plant-env-v2-counterfeit"
    payload = dict(wrong_runtime)
    payload.pop("checkpoint_contract_digest")
    wrong_runtime["checkpoint_contract_digest"] = contract_digest(payload)

    # The digest is internally correct, so this reaches the complete runtime
    # comparison rather than merely failing checksum validation.
    with pytest.raises(DeliveryRelaxationError, match="current V2 runtime"):
        _build(fixture, checkpoint_contract=wrong_runtime)


def test_episode_timing_or_delivery_order_mismatch_fails_closed() -> None:
    fixture = _fixture(steps=2)
    wrong_horizon = replace(
        fixture,
        episode=replace(fixture.episode, physics_steps=1),
    )
    unordered = replace(
        fixture,
        episode=replace(
            fixture.episode,
            commitments=(
                DeliveryCommitment("late", 2, 1.0),
                DeliveryCommitment("early", 1, 1.0),
            ),
        ),
    )

    with pytest.raises(DeliveryRelaxationError, match="horizon does not match"):
        _build(wrong_horizon)
    with pytest.raises(DeliveryRelaxationError, match="ordered by deadline"):
        _build(unordered)


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"battery": BatteryConfig(charge_efficiency=float("nan"))}, "finite"),
        ({"dac": DACConfig(specific_energy_kwh_per_kg=0.0)}, "must be positive"),
        ({"solar": SolarConfig(peak_kw=float("inf"))}, "finite"),
    ),
)
def test_nonfinite_or_nonphysical_plant_coefficients_fail_before_solver_entry(
    change: dict[str, Any],
    message: str,
) -> None:
    fixture = _fixture()
    invalid_config = replace(fixture.config, **change)
    invalid = replace(fixture, config=invalid_config)

    with pytest.raises((DeliveryRelaxationError, ValueError), match=message):
        _build(invalid)


def test_solver_exception_returns_a_failure_without_publishing_a_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scipy_optimize = pytest.importorskip("scipy.optimize")
    fixture = _fixture(initial_h2_kg=0.5, initial_co2_kg=2.75)
    model = _build(fixture)

    def fail_solver(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("deliberate solver failure")

    monkeypatch.setattr(scipy_optimize, "linprog", fail_solver)
    from plant_backend.rl.v2.upper_bound import solve_delivery_relaxation

    result = solve_delivery_relaxation(model)

    assert result.success is False
    assert result.failure_code == "solver_exception"
    assert result.dual_certified_delivery_upper_bound_kg is None
    assert result.relaxed_fill_ceiling is None
    assert result.certificate is None
    assert result.witness is None


def test_post_certificate_failure_drops_certificate_and_every_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plant_backend.rl.v2.upper_bound as upper_bound_module

    model = _build(_fixture(initial_h2_kg=0.5, initial_co2_kg=2.75))
    original = upper_bound_module.box_lagrangian_delivery_certificate

    def certificate_with_bad_sign(*args: Any, **kwargs: Any) -> DualDeliveryCertificate:
        certificate = original(*args, **kwargs)
        return replace(
            certificate,
            positive_inequality_marginal_count=1,
            max_positive_inequality_marginal=1e-3,
        )

    monkeypatch.setattr(
        upper_bound_module,
        "box_lagrangian_delivery_certificate",
        certificate_with_bad_sign,
    )
    result = upper_bound_module.solve_delivery_relaxation(
        model,
        marginal_sign_tolerance=0.0,
    )
    serialized = result.to_dict()

    assert result.success is False
    assert result.failure_code == "certificate_not_numerically_tight"
    assert result.certificate is None
    assert result.dual_certified_delivery_upper_bound_kg is None
    assert result.relaxed_fill_ceiling is None
    assert result.primal_dual_gap_kg is None
    assert serialized["certificate"] is None
    assert serialized["dual_certified_delivery_upper_bound_kg"] is None
    assert serialized["relaxed_fill_ceiling"] is None
    assert serialized["primal_dual_gap_kg"] is None


def test_exact_certificate_below_primal_is_not_hidden_by_upward_float_rounding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plant_backend.rl.v2.upper_bound as upper_bound_module

    model = _build(_fixture(start_hour=0.0, initial_ch4_kg=1.0))
    one = Fraction(1, 1)
    previous = Fraction.from_float(math.nextafter(1.0, -math.inf))
    exact_upper = (3 * one + previous) / 4
    assert exact_upper < one
    assert next_float_up(exact_upper) == 1.0

    def half_ulp_certificate(*args: Any, **kwargs: Any) -> DualDeliveryCertificate:
        del args, kwargs
        lower = -exact_upper
        return DualDeliveryCertificate(
            minimization_lower_bound_numerator=lower.numerator,
            minimization_lower_bound_denominator=lower.denominator,
            uncapped_delivery_upper_bound_numerator=exact_upper.numerator,
            uncapped_delivery_upper_bound_denominator=exact_upper.denominator,
            requested_kg_numerator=1,
            requested_kg_denominator=1,
            certified_delivery_upper_bound_numerator=exact_upper.numerator,
            certified_delivery_upper_bound_denominator=exact_upper.denominator,
            dual_certified_delivery_upper_bound_kg=1.0,
            positive_inequality_marginal_count=0,
            max_positive_inequality_marginal=0.0,
        )

    monkeypatch.setattr(
        upper_bound_module,
        "box_lagrangian_delivery_certificate",
        half_ulp_certificate,
    )
    result = upper_bound_module.solve_delivery_relaxation(model)

    assert result.numerical_primal_delivery_kg == 1.0
    assert result.success is False
    assert result.failure_code == "dual_below_primal"
    assert result.certificate is None
    assert result.dual_certified_delivery_upper_bound_kg is None
    assert result.relaxed_fill_ceiling is None
    assert result.primal_dual_gap_kg is None


def test_deadlines_cannot_borrow_production_from_later_intervals() -> None:
    config = _plant(solar_sunrise_h=6.0, solar_sunset_h=24.0)
    fixture = _fixture(
        config=config,
        steps=3,
        delivery_period_steps=1,
        start_hour=5.0,
        initial_h2_kg=0.5,
        initial_co2_kg=2.75,
    )
    result = _solve(fixture)

    assert result.success is True
    assert result.witness is not None
    assert tuple(item.due_step for item in result.witness.deliveries) == (1, 2, 3)
    assert tuple(
        item.delivered_kg for item in result.witness.deliveries[:2]
    ) == pytest.approx(
        (0.0, 0.0),
        abs=1e-9,
    )
    assert result.witness.deliveries[2].delivered_kg == pytest.approx(1.0, abs=1e-8)


def test_exact_same_interval_production_then_settlement_embeds_in_relaxation() -> None:
    fixture = _fixture(initial_h2_kg=0.5, initial_co2_kg=2.75)
    model, normalized, delivered = _exact_trace_vector(
        fixture,
        (
            Action(
                battery_share=0.0,
                electrolyser_share=0.0,
                dac_share=0.0,
                methanator_share=1.0,
            ),
        ),
    )
    residuals = evaluate_program_residuals(model.program, normalized)
    result = _solve(fixture)

    assert residuals.maximum <= 1e-10
    assert delivered == pytest.approx(1.0, abs=1e-10)
    assert delivered <= _published_bound(result)


def test_exact_guarded_v2_trace_embeds_and_never_exceeds_certificate() -> None:
    config = _plant(dt_s=60.0)
    episode_config = EpisodeConfigV2(
        horizon_h=1.0 / 60.0,
        decision_interval_s=60.0,
        start_hour=12.0,
        delivery_period_h=1.0 / 60.0,
        delivery_rate_kg_h=60.0,
        battery_soc=(0.0, 0.0),
        h2_fill=(0.005, 0.005),
        co2_fill=(0.0275, 0.0275),
        ch4_coverage_h=(0.0, 0.0),
        electrolyser_commitment=None,
    )
    constraints = Constraints(battery_min_soc=0.0)
    env = PlantEnvV2(
        config=config,
        episode=episode_config,
        constraints=constraints,
    )
    env.reset(seed=7)
    assert env._episode is not None  # test-only access to privileged truth
    fixture = _Fixture(
        config=config,
        episode_config=episode_config,
        constraints=constraints,
        episode=env._episode,
        checkpoint_contract=env.checkpoint_contract,
    )
    model = _build(fixture)
    initial = env.simulator.state
    _, _, terminated, truncated, info = env.step_action(
        Action(
            battery_share=0.0,
            electrolyser_share=0.0,
            dac_share=0.0,
            methanator_share=1.0,
        )
    )
    state = env.simulator.state
    delivered_by_id = {
        settlement.id: settlement.delivered_kg
        for settlement in env.simulator.delivery_step.settlements
    }
    normalized = _normalize_trace(
        model,
        initial=initial,
        states=(state,),
        delivered_by_id=delivered_by_id,
    )
    relaxation = _solve(fixture)

    assert terminated is True and truncated is False
    assert evaluate_program_residuals(model.program, normalized).maximum <= 1e-10
    assert info["delivery"]["delivered_kg"] == pytest.approx(1.0, abs=1e-10)
    assert info["delivery"]["delivered_kg"] <= _published_bound(relaxation)


def test_exact_startup_energy_with_zero_hydrogen_embeds_via_upper_yield_constraint() -> (
    None
):
    config = _plant(dt_s=60.0)
    fixture = _fixture(
        config=config,
        electrolyser_commitment=DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE,
    )
    model, normalized, delivered = _exact_trace_vector(
        fixture,
        (Action.all_to_electrolyser(),),
    )
    residuals = evaluate_program_residuals(model.program, normalized)
    physical = model.physical_vector(normalized)
    electrolyser = model.block("solar_to_electrolyser_kw")
    h2_produced = model.block("h2_produced_kg")

    assert residuals.maximum <= 1e-10
    assert physical[electrolyser.start] > 0.0
    assert physical[h2_produced.start] == pytest.approx(0.0, abs=1e-12)
    assert delivered == 0.0


def test_exact_hot_standby_power_with_zero_hydrogen_embeds_in_relaxation() -> None:
    profile = replace(
        DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE,
        cold_start_s=60.0,
        min_up_s=60.0,
    )
    config = _plant(
        dt_s=60.0,
        solar_peak_kw=1_000.0,
        battery_capacity_kwh=1_000.0,
        battery_max_charge_kw=1_000.0,
        battery_max_discharge_kw=1_000.0,
        electrolyser_rated_kw=1_000.0,
    )
    fixture = _fixture(
        config=config,
        steps=2,
        initial_battery_soc=1.0,
        electrolyser_commitment=profile,
    )
    model = _build(fixture)
    simulator = PlantSimulator(config, electrolyser_commitment=profile)
    simulator.load_scenario(
        fixture.episode.scenario,
        solar_factor_tape=fixture.episode.weather.tape,
    )
    simulator.set_delivery_schedule(fixture.episode.commitments, freeze=True)
    initial = simulator.state
    start_kw = profile.start_load_frac * config.electrolyser.rated_kw
    standby_kw = profile.hot_standby_load_frac * config.electrolyser.rated_kw
    start_action = Action(
        battery_share=0.0,
        electrolyser_share=0.0,
        dac_share=0.0,
        methanator_share=0.0,
        battery_support_kw=start_kw,
    )
    standby_action = replace(start_action, battery_support_kw=standby_kw)

    starting_state = simulator.step(start_action)
    after_start = simulator.electrolyser_commitment
    standby_state = simulator.step(standby_action)
    in_standby = simulator.electrolyser_commitment
    delivered_by_id = {
        settlement.id: settlement.delivered_kg
        for settlement in simulator.delivery_step.settlements
    }
    normalized = _normalize_trace(
        model,
        initial=initial,
        states=(starting_state, standby_state),
        delivered_by_id=delivered_by_id,
    )
    result = _solve(fixture)

    assert after_start is not None and after_start.mode is ElectrolyserMode.RUNNING
    assert in_standby is not None and in_standby.mode is ElectrolyserMode.HOT_STANDBY
    assert standby_state.electrolyser_power_kw == pytest.approx(standby_kw)
    assert standby_state.h2_production_kg_h == pytest.approx(0.0, abs=1e-12)
    assert evaluate_program_residuals(model.program, normalized).maximum <= 1e-10
    exact_delivered = math.fsum(delivered_by_id.values())
    assert exact_delivered == 0.0
    assert exact_delivered <= _published_bound(result)


def test_exact_full_tank_refusal_and_spill_embed_in_relaxation() -> None:
    config = _plant(h2_capacity_kg=0.25)
    fixture = _fixture(config=config, initial_h2_kg=0.25)
    model, normalized, delivered = _exact_trace_vector(
        fixture,
        (Action.all_to_electrolyser(),),
    )
    residuals = evaluate_program_residuals(model.program, normalized)
    physical = model.physical_vector(normalized)
    produced = model.block("h2_produced_kg")

    assert residuals.maximum <= 1e-10
    assert physical[produced.start] == pytest.approx(0.0, abs=1e-12)
    assert delivered == 0.0


def test_exact_requested_charge_support_overlap_embeds_after_dispatch_cancellation() -> (
    None
):
    config = _plant(
        battery_capacity_kwh=100_000.0,
        battery_max_charge_kw=50_000.0,
        battery_max_discharge_kw=50_000.0,
    )
    fixture = _fixture(
        config=config,
        initial_battery_soc=0.5,
    )
    action = Action(
        battery_share=0.5,
        electrolyser_share=0.0,
        dac_share=0.0,
        methanator_share=0.0,
        battery_support_kw=10_000.0,
    )
    model, normalized, _ = _exact_trace_vector(fixture, (action,))
    residuals = evaluate_program_residuals(model.program, normalized)
    physical = model.physical_vector(normalized)
    support = model.block("battery_to_electrolyser_kw")

    # Dispatch nets the requested round trip on the common solar bus, so no
    # battery discharge is actually needed in this exact trace.
    assert residuals.maximum <= 1e-10
    assert physical[support.start] == pytest.approx(0.0, abs=1e-10)


def test_exact_trace_and_relaxation_share_sunrise_and_midnight_solar_boundaries() -> (
    None
):
    config = _plant(solar_sunrise_h=6.0, solar_sunset_h=19.5)
    midnight_fixture = _fixture(
        config=config,
        steps=2,
        start_hour=23.0,
    )
    midnight_model, midnight_z, _ = _exact_trace_vector(
        midnight_fixture,
        (Action.idle(), Action.idle()),
    )
    sunrise_fixture = _fixture(
        config=config,
        steps=3,
        start_hour=5.0,
    )
    sunrise_model, sunrise_z, _ = _exact_trace_vector(
        sunrise_fixture,
        (Action.idle(), Action.idle(), Action.idle()),
    )

    assert midnight_model.solar_available_kw == pytest.approx((0.0, 0.0), abs=0.0)
    assert (
        evaluate_program_residuals(midnight_model.program, midnight_z).maximum <= 1e-10
    )
    assert sunrise_model.solar_available_kw[:2] == pytest.approx((0.0, 0.0), abs=0.0)
    assert sunrise_model.solar_available_kw[2] > 0.0
    assert evaluate_program_residuals(sunrise_model.program, sunrise_z).maximum <= 1e-10


def test_exact_trace_at_active_battery_reserve_is_not_excluded() -> None:
    fixture = _fixture(
        start_hour=0.0,
        initial_battery_soc=0.5,
        constraints=Constraints(battery_min_soc=0.5),
    )
    model, normalized, _ = _exact_trace_vector(fixture, (Action.idle(),))

    assert evaluate_program_residuals(model.program, normalized).maximum <= 1e-10


def test_numerical_objective_witness_and_certificate_order_are_consistent() -> None:
    result = _solve(_fixture(initial_h2_kg=0.5, initial_co2_kg=2.75))

    assert result.success is True
    assert result.witness is not None
    assert result.certificate is not None
    assert result.numerical_objective == pytest.approx(
        -result.witness.delivered_kg,
        abs=1e-9,
    )
    assert result.numerical_primal_delivery_kg == pytest.approx(
        result.witness.delivered_kg,
        abs=1e-9,
    )
    assert result.numerical_primal_delivery_kg <= _published_bound(result)
    assert _published_bound(result) <= result.requested_kg
    assert result.primal_dual_gap_kg is not None
    assert result.primal_dual_gap_kg >= 0.0
    assert result.certificate_scope == CERTIFICATE_SCOPE
    assert result.simulator_embedding_status == SIMULATOR_EMBEDDING_STATUS
    assert result.to_dict()["certificate_scope"] == CERTIFICATE_SCOPE
    assert result.to_dict()["simulator_embedding_status"] == SIMULATOR_EMBEDDING_STATUS
    solve_settings = result.solve_settings.to_dict()
    assert solve_settings["feasibility_tolerance"] == 1e-7
    assert solve_settings["physical_tolerance"] == 1e-6
    assert solve_settings["optimality_tolerance_kg"] == 1e-5
    assert solve_settings["marginal_sign_tolerance"] == 1e-9
    assert solve_settings["solver_options"] == {}
    assert any(
        "git source revision" in gap for gap in solve_settings["provenance_gaps"]
    )

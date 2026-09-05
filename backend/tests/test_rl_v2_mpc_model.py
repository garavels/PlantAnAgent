"""Structural and numerical audits for the solver-independent service-MPC MILP."""

from __future__ import annotations

from dataclasses import replace

import pytest
from plant_sim import (
    ComponentHealth,
    ElectrolyserMode,
    PlantConfig,
)

from plant_backend.controllers import Constraints
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.control import ControllerObservationV2, DeliveryObservation
from plant_backend.rl.v2.env import PlantEnvV2
from plant_backend.rl.v2.mpc_forecast import (
    MPCForecastTrajectoryV1,
    project_issued_mean_solar,
)
from plant_backend.rl.v2.mpc_model import (
    MPCSparseRowV1,
    MPCVariableGroupV1,
    ServiceMPCModelV1,
    build_service_mpc_model,
    delivery_lock_row,
    infer_delivery_targets,
)
from plant_backend.rl.v2.mpc_schema import (
    DEFAULT_SERVICE_MPC_SETTINGS_V1,
    SERVICE_MPC_MODEL_VERSION,
    ServiceMPCInputError,
    canonical_json,
)


EPISODE = EpisodeConfigV2(horizon_h=72.0, delivery_rate_kg_h=35.0)
SETTINGS = DEFAULT_SERVICE_MPC_SETTINGS_V1


@pytest.fixture(scope="module")
def public_inputs() -> tuple[
    PlantEnvV2,
    ControllerObservationV2,
    MPCForecastTrajectoryV1,
]:
    env = PlantEnvV2(episode=EPISODE)
    env.reset(seed=0)
    observation = env.controller_observation
    assert observation.forecast is not None
    forecast = project_issued_mean_solar(
        observation.forecast,
        boundary_time_s=observation.state.boundary.time_s,
        config=env.config,
        interval_s=SETTINGS.interval_s,
    )
    return env, observation, forecast


def _build(
    public_inputs: tuple[
        PlantEnvV2,
        ControllerObservationV2,
        MPCForecastTrajectoryV1,
    ],
    *,
    observation: ControllerObservationV2 | None = None,
    forecast: MPCForecastTrajectoryV1 | None = None,
    config: PlantConfig | None = None,
    constraints: Constraints | None = None,
    horizon: int = 96,
) -> ServiceMPCModelV1:
    env, original_observation, original_forecast = public_inputs
    return build_service_mpc_model(
        observation or original_observation,
        forecast or original_forecast,
        settings=SETTINGS,
        constraints=constraints or env.constraints,
        config=config or env.config,
        commitment_profile=env.episode_config.electrolyser_commitment,
        horizon_intervals=horizon,
    )


def _replace_boundary(
    observation: ControllerObservationV2,
    **changes: object,
) -> ControllerObservationV2:
    return replace(
        observation,
        state=replace(
            observation.state,
            boundary=replace(observation.state.boundary, **changes),
        ),
    )


def _replace_commitment(
    observation: ControllerObservationV2,
    **changes: object,
) -> ControllerObservationV2:
    snapshot = observation.state.electrolyser_commitment
    assert snapshot is not None
    return replace(
        observation,
        state=replace(
            observation.state,
            electrolyser_commitment=replace(snapshot, **changes),
        ),
    )


def test_delivery_targets_use_only_announced_request_and_public_recurrence(
    public_inputs: tuple[
        PlantEnvV2,
        ControllerObservationV2,
        MPCForecastTrajectoryV1,
    ],
) -> None:
    env, observation, _ = public_inputs
    targets = infer_delivery_targets(
        observation,
        config=env.config,
        interval_s=SETTINGS.interval_s,
        horizon_intervals=96,
    )

    assert tuple(target.due_boundary for target in targets) == (24, 48, 72, 96)
    assert tuple(target.requested_kg for target in targets) == (210.0,) * 4
    assert tuple(target.source for target in targets) == (
        "announced",
        "inferred_recurring",
        "inferred_recurring",
        "inferred_recurring",
    )


def test_delivery_target_inference_rejects_a_non_aligned_public_deadline(
    public_inputs: tuple[
        PlantEnvV2,
        ControllerObservationV2,
        MPCForecastTrajectoryV1,
    ],
) -> None:
    env, observation, _ = public_inputs
    misaligned = replace(
        observation,
        delivery=replace(observation.delivery, steps_to_next=1),
    )

    with pytest.raises(ServiceMPCInputError, match="does not align"):
        infer_delivery_targets(
            misaligned,
            config=env.config,
            interval_s=SETTINGS.interval_s,
            horizon_intervals=96,
        )


def test_model_has_a_reproducible_complete_sparse_identity(
    public_inputs: tuple[
        PlantEnvV2,
        ControllerObservationV2,
        MPCForecastTrajectoryV1,
    ],
) -> None:
    first = _build(public_inputs)
    second = _build(public_inputs)

    assert first.sha256 == second.sha256
    assert first.body() == second.body()
    assert first.variable_count == 2_118
    assert first.binary_variable_count == 576
    assert first.row_count == 3_145
    assert first.nonzero_count == 7_915
    assert tuple(index for group in first.groups for index in group.indices) == tuple(
        range(first.variable_count)
    )
    assert len(first.sha256) == 64


def test_primary_and_secondary_objectives_match_the_frozen_protocol(
    public_inputs: tuple[
        PlantEnvV2,
        ControllerObservationV2,
        MPCForecastTrajectoryV1,
    ],
) -> None:
    model = _build(public_inputs)
    delivered = set(model.group("delivered_kg"))
    nonzero_primary = {
        index: coefficient
        for index, coefficient in enumerate(model.primary_objective)
        if coefficient != 0.0
    }

    assert nonzero_primary == {index: -1.0 for index in delivered}
    assert model.secondary_objective[model.group("minimum_commitment_fill")[0]] == (
        -SETTINGS.minimum_fill_weight
    )
    assert model.secondary_objective[model.group("battery_energy_kwh")[-1]] < 0.0
    assert all(
        model.secondary_objective[index] > 0.0 for index in model.group("cold_start")
    )
    assert all(
        model.secondary_objective[index] > 0.0
        for index in model.group("solar_curtailed_kw")
    )


def test_model_encodes_buses_material_order_and_battery_direction_exactly(
    public_inputs: tuple[
        PlantEnvV2,
        ControllerObservationV2,
        MPCForecastTrajectoryV1,
    ],
) -> None:
    model = _build(public_inputs)
    names = {row.name for row in model.rows}

    for prefix in (
        "solar_bus",
        "electrolyser_bus",
        "battery_balance",
        "h2_balance",
        "co2_balance",
        "ch4_balance",
        "h2_preconversion_headroom",
        "co2_preconversion_headroom",
        "ch4_predelivery_headroom",
        "battery_charge_direction",
        "battery_support_direction",
        "warmup_definition",
        "on_partition",
        "on_transition",
    ):
        assert f"{prefix}[0]" in names
        assert f"{prefix}[95]" in names

    direction = next(
        row for row in model.rows if row.name == "battery_support_direction[0]"
    )
    support = model.group("battery_support_kw")[0]
    charging = model.group("battery_charging")[0]
    assert dict(direction.coefficients) == {support: 1.0, charging: 2_500.0}
    assert direction.upper == 2_500.0


def test_model_prohibits_late_starts_that_hide_terminal_obligations(
    public_inputs: tuple[
        PlantEnvV2,
        ControllerObservationV2,
        MPCForecastTrajectoryV1,
    ],
) -> None:
    model = _build(public_inputs)
    starts = model.group("cold_start")

    assert model.variable_upper[starts[88]] == 1.0
    assert tuple(model.variable_upper[index] for index in starts[89:]) == (0.0,) * 7
    constants = model.metadata()["constants"]
    assert constants["startup_intervals"] == 4
    assert constants["minimum_up_intervals"] == 4
    assert constants["minimum_down_intervals"] == 2


def test_terminal_battery_floor_uses_target_or_current_when_below_target(
    public_inputs: tuple[
        PlantEnvV2,
        ControllerObservationV2,
        MPCForecastTrajectoryV1,
    ],
) -> None:
    _, observation, _ = public_inputs
    capacity = observation.state.boundary.battery_capacity_kwh
    nominal = _build(public_inputs)
    energy = nominal.group("battery_energy_kwh")
    assert nominal.variable_lower[energy[-1]] == pytest.approx(0.25 * capacity)

    low_observation = _replace_boundary(
        observation,
        battery_soc=0.20,
        battery_energy_kwh=0.20 * capacity,
    )
    low_model = _build(public_inputs, observation=low_observation)
    low_energy = low_model.group("battery_energy_kwh")
    assert low_model.variable_lower[low_energy[-1]] == pytest.approx(0.20 * capacity)


@pytest.mark.parametrize(
    (
        "changes",
        "expected_on",
        "expected_warmup",
        "expected_force_on",
        "expected_force_off",
    ),
    (
        (
            {"mode": ElectrolyserMode.OFF, "min_down_remaining_s": 1_800.0},
            False,
            (),
            (),
            (0, 1),
        ),
        (
            {
                "mode": ElectrolyserMode.COLD_STARTING,
                "startup_remaining_s": 1_800.0,
            },
            True,
            (0, 1),
            (0, 1, 2, 3, 4, 5),
            (),
        ),
        (
            {"mode": ElectrolyserMode.RUNNING, "min_up_remaining_s": 1_800.0},
            True,
            (),
            (0, 1),
            (),
        ),
        (
            {"mode": ElectrolyserMode.HOT_STANDBY},
            True,
            (0,),
            (0,),
            (),
        ),
        (
            {
                "mode": ElectrolyserMode.HOT_STARTING,
                "min_up_remaining_s": 1_800.0,
            },
            True,
            (0,),
            (0, 1),
            (),
        ),
        (
            {
                "mode": ElectrolyserMode.COOLDOWN,
                "cooldown_remaining_s": 900.0,
                "min_down_remaining_s": 1_800.0,
            },
            False,
            (),
            (),
            (0, 1),
        ),
    ),
)
def test_public_commitment_state_is_mapped_to_conservative_fixed_prefixes(
    public_inputs: tuple[
        PlantEnvV2,
        ControllerObservationV2,
        MPCForecastTrajectoryV1,
    ],
    changes: dict[str, object],
    expected_on: bool,
    expected_warmup: tuple[int, ...],
    expected_force_on: tuple[int, ...],
    expected_force_off: tuple[int, ...],
) -> None:
    _, observation, _ = public_inputs
    model = _build(
        public_inputs,
        observation=_replace_commitment(observation, **changes),
    )
    constants = model.metadata()["constants"]

    assert constants["initial_on"] is expected_on
    assert (
        tuple(index for index, value in enumerate(constants["initial_warmup"]) if value)
        == expected_warmup
    )
    assert tuple(constants["initial_force_on"]) == expected_force_on
    assert tuple(constants["initial_force_off"]) == expected_force_off


def test_partial_initial_start_uses_a_full_stable_offer_conservatively(
    public_inputs: tuple[
        PlantEnvV2,
        ControllerObservationV2,
        MPCForecastTrajectoryV1,
    ],
) -> None:
    _, observation, _ = public_inputs
    model = _build(
        public_inputs,
        observation=_replace_commitment(
            observation,
            mode=ElectrolyserMode.COLD_STARTING,
            startup_remaining_s=600.0,
        ),
    )
    constants = model.metadata()["constants"]

    assert constants["initial_warmup"][0] == 1
    assert constants["initial_warmup_offer_adjustment_kw"][0] == pytest.approx(600.0)
    startup_row = next(row for row in model.rows if row.name == "startup_power[0]")
    assert startup_row.lower == startup_row.upper == pytest.approx(600.0)


def test_model_rejects_inputs_outside_its_declared_validity_domain(
    public_inputs: tuple[
        PlantEnvV2,
        ControllerObservationV2,
        MPCForecastTrajectoryV1,
    ],
) -> None:
    env, observation, forecast = public_inputs
    degraded_health = dict(observation.state.boundary.health)
    degraded_health["dac"] = ComponentHealth(
        status="degraded",
        degradation=0.1,
    )
    degraded = _replace_boundary(observation, health=degraded_health)
    with pytest.raises(ServiceMPCInputError, match="degraded/offline dac"):
        _build(public_inputs, observation=degraded)

    invalid_commitment = _replace_commitment(
        observation,
        previous_power_kw=float("nan"),
    )
    with pytest.raises(ServiceMPCInputError, match="finite and non-negative"):
        _build(public_inputs, observation=invalid_commitment)

    stale = replace(
        forecast,
        issued_at_physics_boundary_step=forecast.issued_at_physics_boundary_step + 1,
    )
    with pytest.raises(ServiceMPCInputError, match="different physics step"):
        _build(public_inputs, forecast=stale)

    foreign_config = replace(
        env.config,
        solar=replace(env.config.solar, peak_kw=env.config.solar.peak_kw + 1.0),
    )
    with pytest.raises(ServiceMPCInputError, match="different solar plant"):
        _build(public_inputs, config=foreign_config)

    assert observation.forecast is not None
    changed_issue = replace(
        observation.forecast,
        windows=(
            replace(
                observation.forecast.windows[0],
                p10_peak_power_fraction=(
                    observation.forecast.windows[0].p10_peak_power_fraction * 0.9
                ),
            ),
            *observation.forecast.windows[1:],
        ),
    )
    foreign_projection = project_issued_mean_solar(
        changed_issue,
        boundary_time_s=observation.state.boundary.time_s,
        config=env.config,
        interval_s=SETTINGS.interval_s,
    )
    assert foreign_projection.available_kw == forecast.available_kw
    with pytest.raises(ServiceMPCInputError, match="current public issue"):
        _build(public_inputs, forecast=foreign_projection)

    with pytest.raises(ServiceMPCInputError, match="current public forecast"):
        _build(public_inputs, observation=replace(observation, forecast=None))

    zero_efficiency = replace(
        env.config,
        battery=replace(env.config.battery, discharge_efficiency=0.0),
    )
    with pytest.raises(ServiceMPCInputError, match="positive efficiencies"):
        _build(public_inputs, config=zero_efficiency)


def test_beta_causal_dac_health_disables_only_the_observed_interval(
    public_inputs: tuple[
        PlantEnvV2,
        ControllerObservationV2,
        MPCForecastTrajectoryV1,
    ],
) -> None:
    env, observation, forecast = public_inputs
    health = dict(observation.state.boundary.health)
    health["dac"] = ComponentHealth(status="offline", offline=True)
    offline = _replace_boundary(observation, health=health)

    with pytest.raises(ServiceMPCInputError, match="degraded/offline dac"):
        _build(public_inputs, observation=offline)

    model = build_service_mpc_model(
        offline,
        forecast,
        settings=SETTINGS,
        constraints=env.constraints,
        config=env.config,
        commitment_profile=env.episode_config.electrolyser_commitment,
        horizon_intervals=4,
        causal_current_dac_health=True,
    )

    dac_rows = tuple(row for row in model.rows if row.name.startswith("dac_cap["))
    captured = model.group("co2_captured_kg")
    nominal_dac_cap = env.constraints.dac_limit_kw(env.config)
    assert dac_rows[0].upper == 0.0
    assert all(row.upper == nominal_dac_cap for row in dac_rows[1:])
    assert model.variable_upper[captured[0]] == 0.0
    assert all(model.variable_upper[index] > 0.0 for index in captured[1:])
    assert model.metadata()["causal_current_dac_health"] == {
        "observed_offline": True,
        "dac_cap_kw_by_interval": [
            0.0,
            nominal_dac_cap,
            nominal_dac_cap,
            nominal_dac_cap,
        ],
        "rule": (
            "observed offline disables interval zero; later intervals assume "
            "nominal recovery without consulting a future fault schedule"
        ),
    }


def test_no_remaining_delivery_fixes_minimum_fill_without_private_inference(
    public_inputs: tuple[
        PlantEnvV2,
        ControllerObservationV2,
        MPCForecastTrajectoryV1,
    ],
) -> None:
    _, observation, _ = public_inputs
    no_delivery = replace(
        observation,
        delivery=DeliveryObservation(
            steps_to_next=None,
            delivery_period_steps=observation.delivery.delivery_period_steps,
            next_requested_kg=0.0,
            cumulative_requested_kg=observation.delivery.cumulative_requested_kg,
            cumulative_delivered_kg=observation.delivery.cumulative_delivered_kg,
        ),
    )
    model = _build(public_inputs, observation=no_delivery)
    minimum_fill = model.group("minimum_commitment_fill")[0]

    assert model.deliveries == ()
    assert model.variable_lower[minimum_fill] == 1.0
    assert model.variable_upper[minimum_fill] == 1.0


def test_solution_audit_recomputes_bounds_rows_and_integrality() -> None:
    model = ServiceMPCModelV1(
        model_version=SERVICE_MPC_MODEL_VERSION,
        settings_sha256="a" * 64,
        forecast_sha256="b" * 64,
        interval_s=900.0,
        horizon_intervals=1,
        deliveries=(),
        variable_names=("switch",),
        variable_lower=(0.0,),
        variable_upper=(1.0,),
        integrality=(1,),
        primary_objective=(0.0,),
        secondary_objective=(0.0,),
        groups=(MPCVariableGroupV1(name="switch", indices=(0,)),),
        rows=(
            MPCSparseRowV1(
                name="must_be_on",
                lower=1.0,
                upper=1.0,
                coefficients=((0, 1.0),),
            ),
        ),
        metadata_json=canonical_json({}),
    )

    assert model.audit_solution((1.0,)).valid
    fractional = model.audit_solution((0.5,))
    assert not fractional.valid
    assert fractional.max_row_violation == pytest.approx(0.5)
    assert fractional.max_integrality_violation == pytest.approx(0.5)
    outside = model.audit_solution((2.0,))
    assert not outside.valid
    assert outside.max_bound_violation == pytest.approx(1.0)


def test_solution_audit_scales_cancelling_physical_rows_explicitly() -> None:
    model = ServiceMPCModelV1(
        model_version=SERVICE_MPC_MODEL_VERSION,
        settings_sha256="a" * 64,
        forecast_sha256="b" * 64,
        interval_s=900.0,
        horizon_intervals=1,
        deliveries=(),
        variable_names=("productive_kw", "running"),
        variable_lower=(0.0, 0.0),
        variable_upper=(1_200.0, 1.0),
        integrality=(0, 1),
        primary_objective=(0.0, 0.0),
        secondary_objective=(0.0, 0.0),
        groups=(
            MPCVariableGroupV1(name="productive_kw", indices=(0,)),
            MPCVariableGroupV1(name="running", indices=(1,)),
        ),
        rows=(
            MPCSparseRowV1(
                name="productive_minimum",
                lower=0.0,
                upper=None,
                coefficients=((0, 1.0), (1, -1_200.0)),
            ),
        ),
        metadata_json=canonical_json({}),
    )
    solution = (1_200.0 - 1.6e-5, 1.0)

    absolute_only = model.audit_solution(solution, tolerance=1e-7)
    scaled = model.audit_solution(
        solution,
        tolerance=1e-7,
        relative_tolerance=1e-7,
    )

    assert absolute_only.valid is False
    assert absolute_only.max_row_violation == pytest.approx(1.6e-5)
    assert scaled.valid is True
    assert scaled.max_row_violation == absolute_only.max_row_violation


def test_solution_audit_retains_coefficient_scale_at_binary_zero() -> None:
    model = ServiceMPCModelV1(
        model_version=SERVICE_MPC_MODEL_VERSION,
        settings_sha256="a" * 64,
        forecast_sha256="b" * 64,
        interval_s=900.0,
        horizon_intervals=1,
        deliveries=(),
        variable_names=("productive_kw", "running"),
        variable_lower=(0.0, 0.0),
        variable_upper=(1_200.0, 1.0),
        integrality=(0, 1),
        primary_objective=(0.0, 0.0),
        secondary_objective=(0.0, 0.0),
        groups=(
            MPCVariableGroupV1(name="productive_kw", indices=(0,)),
            MPCVariableGroupV1(name="running", indices=(1,)),
        ),
        rows=(
            MPCSparseRowV1(
                name="productive_maximum",
                lower=None,
                upper=0.0,
                coefficients=((0, 1.0), (1, -1_200.0)),
            ),
        ),
        metadata_json=canonical_json({}),
    )
    solver_scaled_zero = (1.6e-5, 0.0)

    absolute_only = model.audit_solution(solver_scaled_zero, tolerance=1e-7)
    scaled = model.audit_solution(
        solver_scaled_zero,
        tolerance=1e-7,
        relative_tolerance=1e-7,
    )

    assert absolute_only.valid is False
    assert absolute_only.max_row_violation == pytest.approx(1.6e-5)
    assert scaled.valid is True


def test_delivery_lock_is_a_named_lower_bound_on_total_service(
    public_inputs: tuple[
        PlantEnvV2,
        ControllerObservationV2,
        MPCForecastTrajectoryV1,
    ],
) -> None:
    model = _build(public_inputs)
    lock = delivery_lock_row(
        model,
        primary_delivered_kg=700.0,
        tolerance_kg=SETTINGS.delivery_lock_tolerance_kg,
    )

    assert lock.name == "primary_delivery_lock"
    assert lock.lower == pytest.approx(699.999999)
    assert lock.upper is None
    assert lock.coefficients == tuple(
        (index, 1.0) for index in model.group("delivered_kg")
    )


@pytest.mark.parametrize(
    "mutation",
    (
        {"metadata_json": canonical_json([])},
        {"horizon_intervals": 1},
    ),
)
def test_model_schema_rejects_non_object_metadata_and_inconsistent_mutations(
    public_inputs: tuple[
        PlantEnvV2,
        ControllerObservationV2,
        MPCForecastTrajectoryV1,
    ],
    mutation: dict[str, object],
) -> None:
    model = _build(public_inputs)
    with pytest.raises(ValueError):
        replace(model, **mutation)

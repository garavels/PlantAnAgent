"""Outcome-free contracts for absolute MPC setpoints and minute dispatch."""

from __future__ import annotations

from dataclasses import replace

import pytest
from plant_sim import Action, ComponentHealth, PlantConfig, PlantSimulator, PlantState

from plant_backend.controllers.base import Constraints
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.env import PlantEnvV2
from plant_backend.rl.v2.mpc_execution import public_observation_sha256
from plant_backend.rl.v2.mpc_forecast import (
    MPCForecastTrajectoryV1,
    project_issued_mean_solar,
)
from plant_backend.rl.v2.mpc_model import ServiceMPCModelV1, build_service_mpc_model
from plant_backend.rl.v2.mpc_schema import (
    DEFAULT_SERVICE_MPC_SETPOINT_SETTINGS_V1,
    MPCPlanV1,
    ServiceMPCPlanError,
)
from plant_backend.rl.v2.mpc_setpoints import (
    MPCProcessSetpointsV1,
    RESERVE_AWARE_DISPATCH_POLICY,
    dispatch_mpc_setpoints,
    projected_first_interval_sha256,
    realize_first_mpc_setpoints,
    resolve_setpoint_envelope,
)
from plant_backend.rl.v2.mpc_solver import solve_service_mpc_model
from plant_backend.safety import validate


_SHA = "a" * 64
_SETTINGS = DEFAULT_SERVICE_MPC_SETPOINT_SETTINGS_V1


@pytest.fixture
def boundary() -> PlantState:
    state = PlantSimulator().get_telemetry()
    return replace(
        state,
        battery_soc=0.90,
        battery_energy_kwh=0.90 * state.battery_capacity_kwh,
    )


def _command(
    *,
    config: PlantConfig | None = None,
    constraints: Constraints | None = None,
    electrolyser_kw: float = 600.0,
    dac_kw: float = 200.0,
    methanator_kw: float = 100.0,
    battery_charge_kw: float = 300.0,
    expected_support_kw: float = 0.0,
    extra_planned_curtailment_kw: float = 100.0,
) -> MPCProcessSetpointsV1:
    plant = config or PlantConfig()
    active = constraints or Constraints()
    solar_electrolyser_kw = electrolyser_kw - expected_support_kw
    forecast_kw = (
        solar_electrolyser_kw
        + dac_kw
        + methanator_kw
        + battery_charge_kw
        + extra_planned_curtailment_kw
    )
    return MPCProcessSetpointsV1(
        decision_index=0,
        observation_sha256=_SHA,
        public_forecast_sha256="b" * 64,
        projection_sha256="c" * 64,
        first_interval_sha256="d" * 64,
        settings_sha256="e" * 64,
        model_sha256="f" * 64,
        plan_sha256="1" * 64,
        hold_duration_s=900.0,
        electrolyser_target_kw=electrolyser_kw,
        dac_target_kw=dac_kw,
        methanator_target_kw=methanator_kw,
        battery_charge_ceiling_kw=battery_charge_kw,
        expected_battery_support_kw=expected_support_kw,
        planned_forecast_solar_kw=forecast_kw,
        planned_solar_to_battery_kw=battery_charge_kw,
        planned_solar_to_electrolyser_kw=solar_electrolyser_kw,
        planned_solar_to_dac_kw=dac_kw,
        planned_solar_to_methanator_kw=methanator_kw,
        planned_solar_curtailed_kw=extra_planned_curtailment_kw,
        envelope=resolve_setpoint_envelope(
            constraints=active,
            config=plant,
        ),
    )


@pytest.fixture(scope="module")
def solved_inputs() -> tuple[
    PlantEnvV2,
    MPCForecastTrajectoryV1,
    ServiceMPCModelV1,
    MPCPlanV1,
]:
    pytest.importorskip("scipy")
    env = PlantEnvV2(episode=EpisodeConfigV2(horizon_h=72.0, delivery_rate_kg_h=35.0))
    env.reset(seed=0)
    observation = env.controller_observation
    assert observation.forecast is not None
    projection = project_issued_mean_solar(
        observation.forecast,
        boundary_time_s=observation.state.boundary.time_s,
        config=env.config,
        interval_s=_SETTINGS.interval_s,
    )
    model = build_service_mpc_model(
        observation,
        projection,
        settings=_SETTINGS,
        constraints=env.constraints,
        config=env.config,
        commitment_profile=env.episode_config.electrolyser_commitment,
        horizon_intervals=8,
    )
    plan = solve_service_mpc_model(model, settings=_SETTINGS)
    return env, projection, model, plan


def test_setpoint_schema_binds_absolute_targets_plan_and_envelope() -> None:
    command = _command(expected_support_kw=100.0, battery_charge_kw=0.0)
    payload = command.to_dict()

    assert payload["schema"] == "service-mpc-process-setpoints-v1"
    assert payload["controller_id"] == "service-mpc-setpoint-dispatch-v1"
    assert payload["dispatch_policy_id"] == RESERVE_AWARE_DISPATCH_POLICY
    assert payload["setpoints_kw"]["electrolyser"] == 600.0
    assert payload["expected_battery_support_kw"] == 100.0
    assert payload["planned_first_interval_kw"]["solar_to_electrolyser"] == 500.0
    assert len(command.sha256) == 64


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"electrolyser_target_kw": float("nan")}, "finite"),
        ({"dac_target_kw": -1.0}, "must be >= 0"),
        ({"electrolyser_target_kw": 3_001.0}, "active limit"),
        ({"expected_battery_support_kw": 601.0}, "electrolyser target"),
        ({"planned_solar_curtailed_kw": 101.0}, "does not balance"),
        (
            {
                "planned_solar_to_dac_kw": 199.0,
                "planned_solar_curtailed_kw": 101.0,
            },
            "first-plan evidence",
        ),
        ({"dispatch_policy_id": "unknown"}, "unsupported"),
    ],
)
def test_setpoint_schema_rejects_invalid_or_unbound_commands(
    changes: dict[str, object],
    message: str,
) -> None:
    command = _command()
    with pytest.raises((TypeError, ValueError), match=message):
        replace(command, **changes)  # type: ignore[arg-type]


def test_surplus_dispatch_tracks_targets_without_duplicate_support(
    boundary: PlantState,
) -> None:
    command = _command()
    dispatched = dispatch_mpc_setpoints(
        command,
        boundary,
        solar_available_kw=2_000.0,
        constraints=Constraints(),
        config=PlantConfig(),
    )

    assert dispatched.solar_to_electrolyser_kw == pytest.approx(600.0)
    assert dispatched.battery_support_kw == 0.0
    assert dispatched.electrolyser_offered_kw == pytest.approx(600.0)
    assert dispatched.solar_to_dac_kw == pytest.approx(200.0)
    assert dispatched.solar_to_methanator_kw == pytest.approx(100.0)
    assert dispatched.solar_to_battery_kw == pytest.approx(300.0)
    assert dispatched.unallocated_solar_kw == pytest.approx(800.0)
    assert dispatched.electrolyser_target_reachable is True


def test_deficit_dispatch_uses_support_only_for_the_target_shortfall(
    boundary: PlantState,
) -> None:
    command = _command()
    dispatched = dispatch_mpc_setpoints(
        command,
        boundary,
        solar_available_kw=500.0,
        constraints=Constraints(),
        config=PlantConfig(),
    )

    assert dispatched.solar_to_dac_kw == pytest.approx(200.0)
    assert dispatched.solar_to_methanator_kw == pytest.approx(100.0)
    assert dispatched.solar_to_electrolyser_kw == pytest.approx(200.0)
    assert dispatched.battery_support_kw == pytest.approx(400.0)
    assert dispatched.electrolyser_offered_kw == pytest.approx(600.0)
    assert dispatched.solar_to_battery_kw == 0.0
    assert dispatched.unallocated_solar_kw == pytest.approx(0.0)


def test_unsupported_loads_share_scarcity_proportionally(
    boundary: PlantState,
) -> None:
    command = _command(battery_charge_kw=0.0)
    dispatched = dispatch_mpc_setpoints(
        command,
        boundary,
        solar_available_kw=150.0,
        constraints=Constraints(),
        config=PlantConfig(),
    )

    assert dispatched.solar_to_dac_kw == pytest.approx(100.0)
    assert dispatched.solar_to_methanator_kw == pytest.approx(50.0)
    assert (
        dispatched.solar_to_dac_kw / dispatched.solar_to_methanator_kw
        == pytest.approx(2.0)
    )
    assert dispatched.battery_support_kw == pytest.approx(600.0)
    assert dispatched.electrolyser_offered_kw == pytest.approx(600.0)


def test_reserve_bound_shortfall_is_explicit_and_never_overoffered(
    boundary: PlantState,
) -> None:
    plant = PlantConfig()
    constraints = Constraints(battery_min_soc=0.40)
    command = _command(config=plant, constraints=constraints)
    depleted = replace(boundary, battery_soc=0.40)
    dispatched = dispatch_mpc_setpoints(
        command,
        depleted,
        solar_available_kw=100.0,
        constraints=constraints,
        config=plant,
    )

    assert dispatched.support_availability.available_kw == 0.0
    assert dispatched.solar_to_electrolyser_kw == pytest.approx(100.0)
    assert dispatched.battery_support_kw == 0.0
    assert dispatched.electrolyser_offered_kw <= command.electrolyser_target_kw
    assert dispatched.electrolyser_shortfall_kw == pytest.approx(500.0)
    assert dispatched.electrolyser_target_reachable is False


def test_night_dispatch_can_reach_the_stack_target_from_reserve(
    boundary: PlantState,
) -> None:
    command = _command(battery_charge_kw=0.0)
    dispatched = dispatch_mpc_setpoints(
        command,
        boundary,
        solar_available_kw=0.0,
        constraints=Constraints(),
        config=PlantConfig(),
    )

    assert dispatched.action == Action(
        battery_share=0.0, electrolyser_share=0.0, battery_support_kw=600.0
    )
    assert dispatched.electrolyser_offered_kw == pytest.approx(600.0)
    assert dispatched.electrolyser_target_reachable is True
    assert dispatched.unallocated_solar_kw == 0.0


def test_four_fluctuating_minutes_hold_a_600_kw_cold_start_target(
    boundary: PlantState,
) -> None:
    command = _command(
        electrolyser_kw=600.0,
        dac_kw=0.0,
        methanator_kw=0.0,
        battery_charge_kw=0.0,
        extra_planned_curtailment_kw=0.0,
    )

    dispatches = tuple(
        dispatch_mpc_setpoints(
            command,
            replace(boundary, step=index),
            solar_available_kw=solar_kw,
            constraints=Constraints(),
            config=PlantConfig(),
        )
        for index, solar_kw in enumerate((0.0, 100.0, 599.0, 800.0))
    )

    assert tuple(item.electrolyser_offered_kw for item in dispatches) == pytest.approx(
        (600.0, 600.0, 600.0, 600.0)
    )
    assert tuple(item.battery_support_kw for item in dispatches) == pytest.approx(
        (600.0, 500.0, 1.0, 0.0)
    )
    assert all(item.solar_to_battery_kw == 0.0 for item in dispatches)


def test_offline_stack_is_attributed_to_the_unchanged_guard(
    boundary: PlantState,
) -> None:
    offline = replace(
        boundary,
        health={
            **boundary.health,
            "electrolyser": ComponentHealth(status="offline", offline=True),
        },
    )
    command = _command(
        dac_kw=0.0,
        methanator_kw=0.0,
        battery_charge_kw=0.0,
        extra_planned_curtailment_kw=0.0,
    )
    dispatched = dispatch_mpc_setpoints(
        command,
        offline,
        solar_available_kw=600.0,
        constraints=Constraints(),
        config=PlantConfig(),
    )
    guarded = validate(
        dispatched.action,
        offline,
        constraints=Constraints(),
        config=PlantConfig(),
        dispatch_solar_available_kw=600.0,
    )

    assert dispatched.support_availability.electrolyser_available is False
    assert dispatched.electrolyser_target_reachable is False
    assert dispatched.electrolyser_offered_kw == pytest.approx(600.0)
    assert guarded.action.electrolyser_share == 0.0
    assert any("offline" in item for item in guarded.adjustments)


def test_dispatch_is_byte_deterministic_and_rejects_a_changed_envelope(
    boundary: PlantState,
) -> None:
    plant = PlantConfig()
    command = _command(config=plant)
    first = dispatch_mpc_setpoints(
        command,
        boundary,
        solar_available_kw=723.5,
        constraints=Constraints(),
        config=plant,
    )
    second = dispatch_mpc_setpoints(
        command,
        replace(boundary),
        solar_available_kw=723.5,
        constraints=Constraints(),
        config=plant,
    )

    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.sha256 == second.sha256
    assert first.action == Action.sanitised(first.action)
    with pytest.raises(ValueError, match="envelope changed"):
        dispatch_mpc_setpoints(
            command,
            boundary,
            solar_available_kw=723.5,
            constraints=Constraints(battery_min_soc=0.30),
            config=plant,
        )


def test_minute_dispatch_evidence_rejects_policy_and_reachability_substitution(
    boundary: PlantState,
) -> None:
    dispatched = dispatch_mpc_setpoints(
        _command(),
        boundary,
        solar_available_kw=500.0,
        constraints=Constraints(),
        config=PlantConfig(),
    )

    with pytest.raises(ValueError, match="declared policy"):
        replace(
            dispatched,
            solar_to_dac_kw=dispatched.solar_to_dac_kw - 1.0,
            unallocated_solar_kw=dispatched.unallocated_solar_kw + 1.0,
        )
    with pytest.raises(ValueError, match="declared policy"):
        replace(dispatched, electrolyser_target_reachable=False)


def test_dispatch_grid_preserves_every_frozen_invariant(
    boundary: PlantState,
) -> None:
    plant = PlantConfig()
    for soc in (0.15, 0.1501, 0.90):
        current = replace(boundary, battery_soc=soc)
        for solar_kw in (0.0, 1.0, 100.0, 500.0, 1_500.0, 5_000.0):
            for electrolyser_kw in (0.0, 600.0, 3_000.0):
                command = _command(
                    electrolyser_kw=electrolyser_kw,
                    dac_kw=1_200.0,
                    methanator_kw=180.0,
                    battery_charge_kw=2_500.0,
                    extra_planned_curtailment_kw=0.0,
                )
                dispatched = dispatch_mpc_setpoints(
                    command,
                    current,
                    solar_available_kw=solar_kw,
                    constraints=Constraints(),
                    config=plant,
                )
                shares = (
                    dispatched.action.battery_share,
                    dispatched.action.electrolyser_share,
                    dispatched.action.dac_share,
                    dispatched.action.methanator_share,
                )

                assert all(0.0 <= value <= 1.0 for value in shares)
                assert sum(shares) <= 1.0
                assert Action.sanitised(dispatched.action) == dispatched.action
                assert dispatched.electrolyser_offered_kw <= electrolyser_kw + 1e-7
                assert dispatched.solar_to_dac_kw <= command.dac_target_kw + 1e-7
                assert (
                    dispatched.solar_to_methanator_kw
                    <= command.methanator_target_kw + 1e-7
                )
                assert (
                    dispatched.battery_support_kw
                    <= dispatched.support_availability.available_kw + 1e-7
                )
                assert not (
                    dispatched.solar_to_battery_kw > 0.0
                    and dispatched.battery_support_kw > 0.0
                )
                assert (
                    dispatched.solar_to_battery_kw
                    + dispatched.solar_to_electrolyser_kw
                    + dispatched.solar_to_dac_kw
                    + dispatched.solar_to_methanator_kw
                    + dispatched.unallocated_solar_kw
                ) == pytest.approx(solar_kw, abs=1e-7)
                if dispatched.electrolyser_target_reachable:
                    assert dispatched.electrolyser_offered_kw == pytest.approx(
                        electrolyser_kw,
                        abs=1e-7,
                    )


def test_realization_binds_the_selected_plan_and_projected_interval(
    solved_inputs: tuple[
        PlantEnvV2,
        MPCForecastTrajectoryV1,
        ServiceMPCModelV1,
        MPCPlanV1,
    ],
) -> None:
    env, projection, model, plan = solved_inputs
    command = realize_first_mpc_setpoints(
        plan,
        projection,
        decision_index=0,
        observation_sha256=public_observation_sha256(env.controller_observation),
        public_forecast_sha256=projection.source_issue_sha256,
        model_sha256=model.sha256,
        constraints=env.constraints,
        config=env.config,
    )

    assert command.plan_sha256 == plan.sha256
    assert command.projection_sha256 == projection.sha256
    assert command.first_interval_sha256 == projected_first_interval_sha256(projection)
    assert command.electrolyser_target_kw == pytest.approx(
        plan.steps[0].electrolyser_startup_kw + plan.steps[0].electrolyser_productive_kw
    )
    assert command.expected_battery_support_kw == plan.steps[0].battery_support_kw


def test_realization_rejects_identity_and_legacy_stage_substitution(
    solved_inputs: tuple[
        PlantEnvV2,
        MPCForecastTrajectoryV1,
        ServiceMPCModelV1,
        MPCPlanV1,
    ],
) -> None:
    env, projection, model, plan = solved_inputs

    def realize(
        selected: MPCPlanV1,
        *,
        model_sha256: str = model.sha256,
        public_forecast_sha256: str = projection.source_issue_sha256,
    ) -> MPCProcessSetpointsV1:
        return realize_first_mpc_setpoints(
            selected,
            projection,
            decision_index=0,
            observation_sha256=public_observation_sha256(env.controller_observation),
            public_forecast_sha256=public_forecast_sha256,
            model_sha256=model_sha256,
            constraints=env.constraints,
            config=env.config,
        )

    with pytest.raises(ServiceMPCPlanError, match="model.*identities differ"):
        realize(plan, model_sha256="0" * 64)
    with pytest.raises(ServiceMPCPlanError, match="public issue"):
        realize(plan, public_forecast_sha256="0" * 64)
    legacy_stages = (
        replace(plan.stages[0], solution_selection=None),
        replace(plan.stages[1], solution_selection=None),
    )
    legacy_plan = replace(plan, stages=legacy_stages)
    with pytest.raises(ServiceMPCPlanError, match="selection stage evidence"):
        realize(legacy_plan)

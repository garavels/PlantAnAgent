from __future__ import annotations

from dataclasses import replace

import pytest

from plant_backend.rl.v2.mpc_schema import (
    DEFAULT_SERVICE_MPC_SETTINGS_V1,
    MPCDeliveryTargetV1,
    MPCPlanStepV1,
    MPCPlanV1,
    MPCStageSolveV1,
    SERVICE_MPC_CONTROLLER_ID,
    SERVICE_MPC_MODEL_VERSION,
    SERVICE_MPC_OBJECTIVE_VERSION,
    ServiceMPCSettingsV1,
    ServiceMPCSolveError,
    canonical_json,
)


_SHA = "a" * 64


def _stage(name: str) -> MPCStageSolveV1:
    return MPCStageSolveV1(
        stage=name,
        success=True,
        status=0,
        message="optimal",
        elapsed_ns=10,
        objective_value=-1.0,
        mip_dual_bound=-1.0,
        mip_gap=0.0,
        mip_node_count=1,
        solution_sha256=_SHA,
        scipy_version="1.18.1",
        highs_version="1.12.0",
    )


def _step(*, delivered_kg: float = 2.0) -> MPCPlanStepV1:
    return MPCPlanStepV1(
        index=0,
        forecast_solar_kw=10.0,
        solar_to_battery_kw=2.0,
        solar_to_electrolyser_kw=3.0,
        solar_to_dac_kw=2.0,
        solar_to_methanator_kw=1.0,
        solar_curtailed_kw=2.0,
        battery_support_kw=0.0,
        electrolyser_startup_kw=0.0,
        electrolyser_productive_kw=3.0,
        h2_produced_kg=0.01,
        co2_captured_kg=0.25,
        ch4_produced_kg=2.0,
        delivered_kg=delivered_kg,
        battery_energy_end_kwh=100.0,
        h2_end_kg=20.0,
        co2_end_kg=30.0,
        ch4_end_kg=40.0,
        cold_start=False,
        warmup=False,
        running=True,
        battery_charging=True,
    )


def test_default_settings_exactly_encode_the_frozen_protocol() -> None:
    settings = DEFAULT_SERVICE_MPC_SETTINGS_V1
    payload = settings.to_dict()

    assert payload["controller_id"] == SERVICE_MPC_CONTROLLER_ID
    assert payload["model_version"] == SERVICE_MPC_MODEL_VERSION
    assert payload["objective_version"] == SERVICE_MPC_OBJECTIVE_VERSION
    assert payload["forecast_statistic"] == "issued_mean"
    assert payload["controller_fallback"] is None
    assert payload["task_interval_count"] == 288
    assert payload["maximum_planning_intervals"] == 96
    assert payload["terminal_battery_soc_target"] == 0.25
    assert payload["delivery_lock_tolerance_kg"] == 1e-6
    assert payload["solver_options"] == {
        "presolve": True,
        "mip_rel_gap": 1e-6,
        "time_limit": 2.0,
    }
    assert len(settings.sha256) == 64
    assert canonical_json(payload) == canonical_json(settings.to_dict())


def test_settings_identity_changes_with_every_tuning_surface() -> None:
    baseline = DEFAULT_SERVICE_MPC_SETTINGS_V1
    changes = {
        "task_horizon_h": 48.0,
        "planning_horizon_h": 12.0,
        "interval_s": 1_800.0,
        "terminal_battery_soc_target": 0.3,
        "delivery_lock_tolerance_kg": 2e-6,
        "energy_reconstruction_tolerance": 2e-12,
        "solver_presolve": False,
        "solver_mip_rel_gap": 2e-6,
        "solver_time_limit_s": 3.0,
        "minimum_fill_weight": 101.0,
        "terminal_ch4_equivalent_weight": 11.0,
        "terminal_battery_weight": 2.0,
        "cold_start_weight": 0.2,
        "curtailment_weight": 0.02,
        "battery_throughput_weight": 0.02,
    }
    for field, value in changes.items():
        candidate = replace(baseline, **{field: value})
        assert candidate.sha256 != baseline.sha256, field


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"planning_horizon_h": 25.0}, "public forecast support"),
        ({"planning_horizon_h": 0.1}, "whole optimization intervals"),
        ({"terminal_battery_soc_target": 1.1}, "must be <= 1"),
        ({"delivery_lock_tolerance_kg": 0.0}, "tolerances must be positive"),
        ({"solver_time_limit_s": 0.0}, "must be positive"),
        ({"minimum_fill_weight": 0.0}, "must be positive"),
    ],
)
def test_settings_reject_invalid_protocol_values(
    changes: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(DEFAULT_SERVICE_MPC_SETTINGS_V1, **changes)


def test_delivery_target_distinguishes_announced_from_inferred() -> None:
    announced = MPCDeliveryTargetV1(
        ordinal=0,
        due_boundary=4,
        due_offset_s=3_600.0,
        requested_kg=10.0,
        source="announced",
    )
    inferred = MPCDeliveryTargetV1(
        ordinal=1,
        due_boundary=28,
        due_offset_s=25_200.0,
        requested_kg=10.0,
        source="inferred_recurring",
    )
    assert announced.to_dict()["source"] == "announced"
    assert inferred.to_dict()["source"] == "inferred_recurring"
    with pytest.raises(ValueError, match="first delivery target"):
        replace(announced, source="inferred_recurring")
    with pytest.raises(ValueError, match="later delivery targets"):
        replace(inferred, source="announced")


def test_solver_status_never_turns_a_failed_stage_into_evidence() -> None:
    success = _stage("primary_delivery")
    assert success.to_dict()["success"] is True
    with pytest.raises(ValueError, match="optimal status zero"):
        replace(success, status=1)
    with pytest.raises(ValueError, match="cannot publish a solution"):
        replace(
            success,
            success=False,
            status=1,
            objective_value=None,
            mip_dual_bound=None,
            mip_gap=None,
            mip_node_count=None,
        )
    independently_rejected = replace(
        success,
        success=False,
        solution_sha256=None,
    )
    assert independently_rejected.status == 0
    assert independently_rejected.success is False
    assert independently_rejected.solution_sha256 is None
    with pytest.raises(ValueError, match="unsuccessful stage"):
        ServiceMPCSolveError("invalid evidence", stages=(success,))


def test_plan_step_validates_buses_and_battery_direction() -> None:
    step = _step()
    assert step.to_dict()["forecast_solar_kw"] == 10.0
    with pytest.raises(ValueError, match="solar bus"):
        replace(step, solar_curtailed_kw=3.0)
    with pytest.raises(ValueError, match="electrolyser bus"):
        replace(step, electrolyser_productive_kw=4.0)
    with pytest.raises(ValueError, match="charge and discharge"):
        replace(step, battery_support_kw=1.0, electrolyser_productive_kw=4.0)


def test_plan_binds_stages_steps_and_delivery_totals() -> None:
    plan = MPCPlanV1(
        model_sha256=_SHA,
        forecast_sha256=_SHA,
        settings_sha256=_SHA,
        horizon_intervals=1,
        deliveries=(
            MPCDeliveryTargetV1(
                ordinal=0,
                due_boundary=1,
                due_offset_s=900.0,
                requested_kg=2.0,
                source="announced",
            ),
        ),
        stages=(_stage("primary_delivery"), _stage("secondary_tiebreak")),
        steps=(_step(),),
        primary_delivered_optimum_kg=2.0,
        selected_delivered_kg=2.0,
        delivery_lock_tolerance_kg=1e-6,
        minimum_commitment_fill=1.0,
        terminal_ch4_equivalent_kg=50.0,
        secondary_utility=100.0,
    )

    assert plan.to_dict()["sha256"] == plan.sha256
    assert len(plan.sha256) == 64
    slower = replace(
        plan,
        stages=tuple(
            replace(stage, elapsed_ns=stage.elapsed_ns + 1) for stage in plan.stages
        ),
    )
    assert slower.body() != plan.body()
    assert slower.sha256 == plan.sha256
    different_solution = replace(
        plan,
        stages=(
            plan.stages[0],
            replace(plan.stages[1], solution_sha256="b" * 64),
        ),
    )
    assert different_solution.sha256 != plan.sha256
    with pytest.raises(ValueError, match="does not match its steps"):
        replace(plan, selected_delivered_kg=1.0)
    with pytest.raises(ValueError, match="primary delivery lock"):
        replace(
            plan,
            steps=(_step(delivered_kg=1.0),),
            selected_delivered_kg=1.0,
        )
    with pytest.raises(ValueError, match="minimum fill"):
        replace(plan, minimum_commitment_fill=0.5)
    with pytest.raises(ValueError, match="out of order"):
        replace(plan, stages=tuple(reversed(plan.stages)))


def test_delivery_free_plan_uses_unit_fill_convention() -> None:
    step = _step(delivered_kg=0.0)
    plan = MPCPlanV1(
        model_sha256=_SHA,
        forecast_sha256=_SHA,
        settings_sha256=_SHA,
        horizon_intervals=1,
        deliveries=(),
        stages=(_stage("primary_delivery"), _stage("secondary_tiebreak")),
        steps=(step,),
        primary_delivered_optimum_kg=0.0,
        selected_delivered_kg=0.0,
        delivery_lock_tolerance_kg=1e-6,
        minimum_commitment_fill=1.0,
        terminal_ch4_equivalent_kg=50.0,
        secondary_utility=1.0,
    )
    assert plan.minimum_commitment_fill == 1.0
    with pytest.raises(ValueError, match="unit fill"):
        replace(plan, minimum_commitment_fill=0.0)

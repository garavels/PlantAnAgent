"""Contract tests for the separately identified cyclic-market beta MPC."""

from __future__ import annotations

from dataclasses import replace

import pytest

from plant_backend.rl.v2.beta_mpc_schema import (
    BETA_MPC_CONTROLLER_ID,
    BETA_MPC_MODEL_VERSION,
    BETA_MPC_PRICE_PROJECTION_VERSION,
    BetaMPCCycleReturnV1,
    BetaMPCDecisionEvidenceV1,
    BetaMPCPlanV1,
    BetaMPCPriceProjectionV1,
    BetaMPCResourceReturnV1,
    BetaMPCSettingsV1,
    project_beta_mpc_prices,
)
from plant_backend.rl.v2.economic_mpc_schema import (
    EconomicMPCDecisionEvidenceV1,
    EconomicMPCPlanV1,
    EconomicMPCStageSolveV1,
)
from plant_backend.rl.v2.market import PriceForecastBinV1, PublicPriceForecastIssueV1
from plant_backend.rl.v2.mpc_schema import MPCPlanStepV1
from plant_backend.rl.v2.setpoint_control import ProcessSetpointTargetsV1


def _issue(
    *,
    current: float = 299.0,
    p90_offset: float = 10.0,
) -> PublicPriceForecastIssueV1:
    return PublicPriceForecastIssueV1(
        issued_at_decision=4,
        issued_at_offset_s=3_600.0,
        current_eur_per_mwh=current,
        bins=tuple(
            PriceForecastBinV1(
                hour_index=index,
                mean_eur_per_mwh=float(index),
                p10_eur_per_mwh=float(index) - 10.0,
                p90_eur_per_mwh=float(index) + p90_offset,
            )
            for index in range(24)
        ),
        market_id="public-market-fixture",
    )


def _return_resource(
    resource: str,
    unit: str,
    target: float,
) -> BetaMPCResourceReturnV1:
    return BetaMPCResourceReturnV1(
        resource=resource,
        unit=unit,
        current=target,
        signed_margin=0.0,
        target=target,
        tolerance=1e-6,
        required_minimum=max(0.0, target - 1e-6),
    )


def _cycle_return() -> BetaMPCCycleReturnV1:
    return BetaMPCCycleReturnV1(
        cycle_index=1,
        decision_in_cycle=23,
        cycle_decisions=24,
        issued_at_decision=23,
        next_boundary_decision=24,
        remaining_intervals=1,
        resources=(
            _return_resource("battery", "kWh", 4_000.0),
            _return_resource("h2", "kg", 200.0),
            _return_resource("co2", "kg", 600.0),
            _return_resource("ch4", "kg", 1.0),
        ),
    )


def _economic_plan() -> EconomicMPCPlanV1:
    stages = (
        EconomicMPCStageSolveV1(
            stage="service_maximization",
            status=0,
            message="optimal",
            elapsed_ns=1,
            objective_value=0.0,
            mip_gap=0.0,
            mip_node_count=0,
            solution_sha256="a" * 64,
            scipy_version="fixture",
            highs_version="fixture",
            success=True,
        ),
        EconomicMPCStageSolveV1(
            stage="economic_cost_minimization",
            status=0,
            message="optimal",
            elapsed_ns=1,
            objective_value=0.0,
            mip_gap=0.0,
            mip_node_count=0,
            solution_sha256="b" * 64,
            scipy_version="fixture",
            highs_version="fixture",
            success=True,
        ),
    )
    step = MPCPlanStepV1(
        index=0,
        forecast_solar_kw=0.0,
        solar_to_battery_kw=0.0,
        solar_to_electrolyser_kw=0.0,
        solar_to_dac_kw=0.0,
        solar_to_methanator_kw=0.0,
        solar_curtailed_kw=0.0,
        battery_support_kw=0.0,
        electrolyser_startup_kw=0.0,
        electrolyser_productive_kw=0.0,
        h2_produced_kg=0.0,
        co2_captured_kg=0.0,
        ch4_produced_kg=0.0,
        delivered_kg=0.0,
        battery_energy_end_kwh=4_000.0,
        h2_end_kg=200.0,
        co2_end_kg=600.0,
        ch4_end_kg=1.0,
        cold_start=False,
        warmup=False,
        running=False,
        battery_charging=False,
    )
    return EconomicMPCPlanV1(
        model_sha256="1" * 64,
        forecast_sha256="2" * 64,
        settings_sha256="3" * 64,
        assumptions_sha256="4" * 64,
        horizon_intervals=1,
        deliveries=(),
        stages=stages,
        steps=(step,),
        primary_delivered_optimum_kg=0.0,
        selected_delivered_kg=0.0,
        delivery_lock_tolerance_kg=1e-6,
        predicted_cost_components_eur=(),
        terminal_ch4_equivalent_deficit_kg=0.0,
        terminal_battery_deficit_kwh=0.0,
    )


def test_settings_publish_beta_identity_hard_returns_and_no_fallback() -> None:
    settings = BetaMPCSettingsV1()
    body = settings.to_dict()

    assert body["controller_id"] == BETA_MPC_CONTROLLER_ID
    assert body["model_version"] == BETA_MPC_MODEL_VERSION
    assert body["price_projection_version"] == BETA_MPC_PRICE_PROJECTION_VERSION
    assert body["controller_fallback"] is None
    assert "four independent hard" in body["cycle_return_semantics"]
    assert settings.maximum_planning_intervals * settings.interval_s == 86_400.0

    with pytest.raises(ValueError, match="exactly 24 h"):
        BetaMPCSettingsV1(planning_horizon_h=12.0)
    with pytest.raises(ValueError, match="whole 24 h cycles"):
        BetaMPCSettingsV1(task_horizon_h=36.0)
    with pytest.raises(ValueError, match="subdivision"):
        BetaMPCSettingsV1(interval_s=800.0)


def test_price_projection_uses_current_then_only_current_issue_means() -> None:
    issue = _issue()
    projection = project_beta_mpc_prices(
        issue,
        interval_s=900.0,
        horizon_intervals=9,
    )

    assert projection.prices_eur_per_mwh == (
        299.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
    )
    assert projection.market_issue_sha256 == issue.sha256
    assert issue.sha256 in projection.price_source_id
    assert projection.to_dict()["p10_p90_semantics"].endswith("unused")


def test_price_projection_hash_binds_visible_quantiles_even_when_path_is_same() -> None:
    narrow = project_beta_mpc_prices(
        _issue(p90_offset=10.0), interval_s=900.0, horizon_intervals=8
    )
    wide = project_beta_mpc_prices(
        _issue(p90_offset=20.0), interval_s=900.0, horizon_intervals=8
    )

    assert narrow.prices_eur_per_mwh == wide.prices_eur_per_mwh
    assert narrow.market_issue_sha256 != wide.market_issue_sha256
    assert narrow.sha256 != wide.sha256


def test_cycle_return_requires_exact_geometry_and_independent_tolerances() -> None:
    cycle_return = _cycle_return()

    assert cycle_return.resource("battery").required_minimum == 3_999.999999
    assert cycle_return.resource("h2").required_minimum == 199.999999
    assert cycle_return.next_boundary_decision == 24
    assert tuple(cycle_return.to_dict()["resources"]) == (
        "battery",
        "h2",
        "co2",
        "ch4",
    )

    with pytest.raises(ValueError, match="cycle counters"):
        replace(cycle_return, issued_at_decision=22)
    with pytest.raises(ValueError, match="settlement tolerance"):
        replace(
            cycle_return.resources[0],
            required_minimum=cycle_return.resources[0].target,
        )


def test_beta_plan_and_decision_evidence_republish_without_alpha_identity() -> None:
    economic = _economic_plan()
    price = BetaMPCPriceProjectionV1(
        market_issue_sha256="5" * 64,
        market_id="public-market-fixture",
        issued_at_decision=23,
        issued_at_offset_s=23 * 3_600.0,
        interval_s=3_600.0,
        prices_eur_per_mwh=(75.0,),
    )
    cycle_return = _cycle_return()
    plan = BetaMPCPlanV1(
        model_sha256=economic.model_sha256,
        forecast_sha256=economic.forecast_sha256,
        settings_sha256=economic.settings_sha256,
        assumptions_sha256=economic.assumptions_sha256,
        horizon_intervals=economic.horizon_intervals,
        deliveries=economic.deliveries,
        stages=economic.stages,
        steps=economic.steps,
        primary_delivered_optimum_kg=economic.primary_delivered_optimum_kg,
        selected_delivered_kg=economic.selected_delivered_kg,
        delivery_lock_tolerance_kg=economic.delivery_lock_tolerance_kg,
        predicted_cost_components_eur=economic.predicted_cost_components_eur,
        terminal_ch4_equivalent_deficit_kg=(
            economic.terminal_ch4_equivalent_deficit_kg
        ),
        terminal_battery_deficit_kwh=economic.terminal_battery_deficit_kwh,
        price_projection=price,
        market_issue_sha256=price.market_issue_sha256,
        cycle_observation_sha256="6" * 64,
        beta_task_sha256="7" * 64,
        cycle_return=cycle_return,
    )
    targets = ProcessSetpointTargetsV1(
        electrolyser_target_kw=0.0,
        dac_target_kw=0.0,
        methanator_target_kw=0.0,
        battery_charge_ceiling_kw=0.0,
    )
    evidence = BetaMPCDecisionEvidenceV1(
        decision_index=23,
        beta_observation_sha256="8" * 64,
        physical_observation_sha256="9" * 64,
        market_issue_sha256=price.market_issue_sha256,
        cycle_observation_sha256=plan.cycle_observation_sha256,
        weather_projection_sha256=plan.forecast_sha256,
        price_projection_sha256=price.sha256,
        settings_sha256=plan.settings_sha256,
        assumptions_sha256=plan.assumptions_sha256,
        beta_task_sha256=plan.beta_task_sha256,
        cycle_return_sha256=cycle_return.sha256,
        model_sha256=plan.model_sha256,
        plan_sha256=plan.sha256,
        market_id=price.market_id,
        targets=targets,
        plan=plan,
    )

    serialized = evidence.to_dict()
    assert serialized["controller_id"] == BETA_MPC_CONTROLLER_ID
    assert serialized["plan"]["controller_id"] == BETA_MPC_CONTROLLER_ID
    assert serialized["plan"]["model_version"] == BETA_MPC_MODEL_VERSION
    assert serialized["plan"]["price_projection"]["sha256"] == price.sha256

    retimed_stages = tuple(
        replace(stage, elapsed_ns=stage.elapsed_ns + 10_000) for stage in plan.stages
    )
    retimed_plan = replace(plan, stages=retimed_stages)
    retimed_evidence = replace(evidence, plan=retimed_plan)
    assert retimed_plan.sha256 == plan.sha256
    assert retimed_evidence.sha256 == evidence.sha256

    short_terminal = replace(
        plan.steps[-1],
        battery_energy_end_kwh=(
            cycle_return.resource("battery").required_minimum - 5e-8
        ),
    )
    with pytest.raises(ValueError, match="hard battery cycle return"):
        replace(plan, steps=(short_terminal,))

    with pytest.raises(ValueError, match="market issue differs"):
        replace(evidence, market_issue_sha256="f" * 64)

    with pytest.raises(TypeError, match="plan must be EconomicMPCPlanV1"):
        EconomicMPCDecisionEvidenceV1(
            decision_index=23,
            economic_observation_sha256="a" * 64,
            physical_observation_sha256="9" * 64,
            projection_sha256=plan.forecast_sha256,
            settings_sha256=plan.settings_sha256,
            assumptions_sha256=plan.assumptions_sha256,
            model_sha256=plan.model_sha256,
            plan_sha256=plan.sha256,
            tariff_id="alpha-tariff-fixture",
            targets=targets,
            plan=plan,
        )

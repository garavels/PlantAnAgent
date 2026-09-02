"""Focused contract tests for the transparent V2 delivery controllers."""

from __future__ import annotations

import ast
import json
import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from plant_sim import (
    Action,
    DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE,
    ComponentHealth,
    ElectrolyserMode,
    FaultSpec,
    InitialState,
    PlantConfig,
    PlantSimulator,
    PlantState,
    Scenario,
)

from plant_backend.controllers import Constraints
from plant_backend.rl.v2.boundary import ControlStateV2
from plant_backend.rl.v2.control import (
    ControllerObservationV2,
    ControllerV2,
    DeliveryObservation,
)
from plant_backend.rl.v2.controllers import (
    ClimatologyControllerV2,
    DeliveryAwareControllerV2,
    DeliveryPlannerSettingsV2,
    FROZEN_CLIMATOLOGICAL_RESOURCE_FACTOR_V2,
    IssuedForecastControllerV2,
    RiskAdjustedIssuedForecastControllerV2,
)
from plant_backend.rl.v2.weather import ForecastWindowV2, PublicForecastIssueV2


CONFIG = PlantConfig()
PROFILE = DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE
BASE_SNAPSHOT = PlantSimulator(
    CONFIG,
    electrolyser_commitment=PROFILE,
).electrolyser_commitment
assert BASE_SNAPSHOT is not None


def _forecast(mean: float, p10: float | None = None) -> PublicForecastIssueV2:
    p10 = mean if p10 is None else p10
    return _windowed_forecast(((mean, p10),) * 4)


def _windowed_forecast(
    values: tuple[tuple[float, float], ...],
) -> PublicForecastIssueV2:
    names = ("0-1h", "1-6h", "6-12h", "12-24h")
    spans = ((0.0, 1.0), (1.0, 6.0), (6.0, 12.0), (12.0, 24.0))
    return PublicForecastIssueV2(
        issued_at_physics_boundary_step=0,
        issued_at_offset_s=0.0,
        windows=tuple(
            ForecastWindowV2(
                name=name,
                start_offset_s_from_issue=start_h * 3_600.0,
                end_offset_s_from_issue=end_h * 3_600.0,
                mean_peak_power_fraction=mean,
                p10_peak_power_fraction=p10,
            )
            for name, (start_h, end_h), (mean, p10) in zip(
                names, spans, values, strict=True
            )
        ),
    )


def _boundary(**changes: object) -> PlantState:
    state = PlantState(
        time_s=12.0 * 3_600.0,
        solar_available_kw=5_000.0,
        battery_soc=0.60,
        battery_capacity_kwh=8_000.0,
        h2_level_kg=0.0,
        h2_capacity_kg=2_000.0,
        co2_level_kg=0.0,
        co2_capacity_kg=6_000.0,
        ch4_level_kg=0.0,
        ch4_capacity_kg=3_000.0,
        health={
            name: ComponentHealth()
            for name in ("battery", "electrolyser", "dac", "methanator")
        },
    )
    return replace(state, **changes)


def _delivery(**changes: object) -> DeliveryObservation:
    fields: dict[str, object] = {
        "steps_to_next": 360,
        "delivery_period_steps": 360,
        "next_requested_kg": 100.0,
        "cumulative_requested_kg": 0.0,
        "cumulative_delivered_kg": 0.0,
    }
    fields.update(changes)
    return DeliveryObservation(**fields)  # type: ignore[arg-type]


def _observation(
    *,
    boundary: PlantState | None = None,
    delivery: DeliveryObservation | None = None,
    forecast: PublicForecastIssueV2 | None = None,
    snapshot=BASE_SNAPSHOT,
) -> ControllerObservationV2:
    return ControllerObservationV2(
        state=ControlStateV2(
            boundary=boundary or _boundary(),
            electrolyser_commitment=snapshot,
        ),
        delivery=delivery or _delivery(),
        forecast=forecast,
        progress=0.25,
        decision_interval_s=900.0,
    )


def _decision(
    controller: (
        ClimatologyControllerV2
        | DeliveryAwareControllerV2
        | IssuedForecastControllerV2
        | RiskAdjustedIssuedForecastControllerV2
    ),
    observation: ControllerObservationV2,
    *,
    profile=PROFILE,
    constraints: Constraints | None = None,
):
    return controller.compute_action(
        observation,
        constraints=constraints or Constraints(),
        config=CONFIG,
        commitment_profile=profile,
    )


def _offered_stack_kw(decision, solar_kw: float) -> float:
    return (
        decision.action.electrolyser_share * solar_kw
        + decision.action.battery_support_kw
    )


def test_wrappers_share_one_planner_and_publish_stable_metadata() -> None:
    clear_sky = DeliveryAwareControllerV2()
    climatology = ClimatologyControllerV2()
    issued = IssuedForecastControllerV2()
    risk_adjusted = RiskAdjustedIssuedForecastControllerV2()

    for controller in (clear_sky, climatology, issued, risk_adjusted):
        assert isinstance(controller, ControllerV2)
        assert controller.compute_action.__func__ is clear_sky.compute_action.__func__
    assert clear_sky.settings == climatology.settings == issued.settings
    assert replace(risk_adjusted.settings, forecast_p10_weight=0.0) == issued.settings
    assert len({clear_sky.id, climatology.id, issued.id, risk_adjusted.id}) == 4
    assert not clear_sky.uses_forecast
    assert not climatology.uses_forecast
    assert issued.uses_forecast
    assert risk_adjusted.uses_forecast
    assert "excluded" in clear_sky.information_set
    assert "excluded" in climatology.information_set
    assert "PublicForecastIssueV2" in issued.information_set
    assert "PublicForecastIssueV2" in risk_adjusted.information_set
    assert climatology.to_dict()["settings"] == {
        "delivery_buffer_fraction": 0.03,
        "feedstock_reserve_fraction": 0.03,
        "climatological_resource_factor": FROZEN_CLIMATOLOGICAL_RESOURCE_FACTOR_V2,
        "forecast_p10_weight": 0.0,
    }
    assert issued.settings.forecast_p10_weight == 0.0
    assert risk_adjusted.settings.forecast_p10_weight == 0.50
    for controller in (clear_sky, climatology, issued, risk_adjusted):
        descriptor = controller.to_dict()
        assert descriptor == controller.descriptor()
        assert descriptor["id"] == controller.id
        assert set(descriptor) == {
            "id",
            "label",
            "description",
            "uses_forecast",
            "information_set",
            "settings",
        }
        json.dumps(descriptor, allow_nan=False)


def test_forecast_blind_action_is_invariant_to_present_absent_or_changed_issue() -> None:
    controller = DeliveryAwareControllerV2()
    observations = (
        _observation(forecast=None),
        _observation(forecast=_forecast(0.05, 0.01)),
        _observation(forecast=_forecast(0.95, 0.90)),
    )

    decisions = [_decision(controller, item).action for item in observations]
    assert decisions[0] == decisions[1] == decisions[2]


def test_climatology_action_is_invariant_to_issued_forecast_field() -> None:
    controller = ClimatologyControllerV2()
    observations = (
        _observation(forecast=None),
        _observation(forecast=_forecast(0.05, 0.01)),
        _observation(forecast=_forecast(0.95, 0.90)),
    )

    decisions = [_decision(controller, item).action for item in observations]
    assert decisions[0] == decisions[1] == decisions[2]


def test_blind_availability_integrates_known_sunrise_and_sunset_geometry() -> None:
    controller = DeliveryAwareControllerV2()
    daylight_h = CONFIG.solar.sunset_h - CONFIG.solar.sunrise_h
    observation = _observation(
        boundary=_boundary(time_s=CONFIG.solar.sunrise_h * 3_600.0),
    )

    full_daylight = controller._availability_equivalent_h(  # type: ignore[attr-defined]
        observation,
        CONFIG,
        start_h=0.0,
        end_h=daylight_h,
    )
    two_days = controller._availability_equivalent_h(  # type: ignore[attr-defined]
        _observation(boundary=_boundary(time_s=3.0 * 3_600.0)),
        CONFIG,
        start_h=0.0,
        end_h=48.0,
    )

    daily_equivalent_h = 2.0 * daylight_h / math.pi
    assert full_daylight == pytest.approx(daily_equivalent_h)
    assert two_days == pytest.approx(2.0 * daily_equivalent_h)


def test_blind_controller_carries_a_full_delivery_across_sunless_period() -> None:
    controller = DeliveryAwareControllerV2()
    # Current deadline is sunset; the inferred following six-hour contract lies
    # wholly in the night interval, including the midnight wrap.
    observation = _observation(
        boundary=_boundary(
            time_s=18.0 * 3_600.0,
            h2_level_kg=100.0,
            co2_level_kg=550.0,
        ),
        delivery=_delivery(steps_to_next=90),
        snapshot=None,
    )
    decision = _decision(controller, observation, profile=None)

    assert "100.00 kg inferred next-delivery carry" in decision.notes[0]


def test_horizon_and_carry_use_one_shared_availability_transform() -> None:
    assert (
        DeliveryAwareControllerV2._planning_horizon_h
        is IssuedForecastControllerV2._planning_horizon_h
    )
    assert (
        DeliveryAwareControllerV2._next_delivery_carry_fraction
        is IssuedForecastControllerV2._next_delivery_carry_fraction
    )
    assert (
        ClimatologyControllerV2._planning_horizon_h
        is IssuedForecastControllerV2._planning_horizon_h
    )
    assert (
        ClimatologyControllerV2._next_delivery_carry_fraction
        is IssuedForecastControllerV2._next_delivery_carry_fraction
    )
    period_h = _delivery().delivery_period_steps * CONFIG.dt_h
    cases = (
        (DeliveryAwareControllerV2(), _observation()),
        (ClimatologyControllerV2(), _observation()),
        (IssuedForecastControllerV2(), _observation(forecast=_forecast(0.6, 0.2))),
    )

    for controller, observation in cases:
        horizon_h = controller._planning_horizon_h(  # type: ignore[attr-defined]
            observation,
            CONFIG,
            start_h=0.0,
            end_h=6.0,
        )
        source_h = controller._availability_equivalent_h(  # type: ignore[attr-defined]
            observation,
            CONFIG,
            start_h=0.0,
            end_h=6.0,
        )
        next_source_h = controller._availability_equivalent_h(  # type: ignore[attr-defined]
            observation,
            CONFIG,
            start_h=6.0,
            end_h=6.0 + period_h,
        )
        carry = controller._next_delivery_carry_fraction(  # type: ignore[attr-defined]
            observation,
            CONFIG,
            remaining_h=6.0,
        )

        assert horizon_h == pytest.approx(source_h)
        assert carry == pytest.approx(1.0 - next_source_h / period_h)


def test_climatology_and_issued_forecast_share_piecewise_overlap_exactly() -> None:
    assert (
        ClimatologyControllerV2._availability_equivalent_h
        is IssuedForecastControllerV2._availability_equivalent_h
    )
    assert (
        RiskAdjustedIssuedForecastControllerV2._availability_equivalent_h
        is IssuedForecastControllerV2._availability_equivalent_h
    )
    assert ClimatologyControllerV2._overlapping_bands(  # type: ignore[attr-defined]
        start_h=0.0,
        end_h=5.5,
    ) == ((0, 0.0, 1.0, 1.0), (1, 1.0, 6.0, 4.5))
    cases = (
        (ClimatologyControllerV2(), _observation()),
        (IssuedForecastControllerV2(), _observation(forecast=_forecast(0.7))),
        (
            RiskAdjustedIssuedForecastControllerV2(),
            _observation(forecast=_forecast(0.7, 0.4)),
        ),
    )

    for controller, observation in cases:
        partial_h = controller._availability_equivalent_h(  # type: ignore[attr-defined]
            observation,
            CONFIG,
            start_h=0.0,
            end_h=5.5,
        )
        first_band_h = controller._availability_equivalent_h(  # type: ignore[attr-defined]
            observation,
            CONFIG,
            start_h=0.0,
            end_h=1.0,
        )
        two_bands_h = controller._availability_equivalent_h(  # type: ignore[attr-defined]
            observation,
            CONFIG,
            start_h=0.0,
            end_h=6.0,
        )

        second_fraction = controller._band_peak_power_fraction(  # type: ignore[attr-defined]
            observation,
            CONFIG,
            band_index=1,
            start_h=1.0,
            end_h=6.0,
        )
        assert partial_h == pytest.approx(first_band_h + 4.5 * second_fraction)
        assert two_bands_h == pytest.approx(first_band_h + 5.0 * second_fraction)
        assert two_bands_h > first_band_h


def test_piecewise_band_availability_is_continuous_at_canonical_edge() -> None:
    epsilon_h = 1e-6
    cases = (
        (ClimatologyControllerV2(), _observation()),
        (IssuedForecastControllerV2(), _observation(forecast=_forecast(0.7))),
        (
            RiskAdjustedIssuedForecastControllerV2(),
            _observation(forecast=_forecast(0.7, 0.4)),
        ),
    )

    for controller, observation in cases:
        left = controller._availability_equivalent_h(  # type: ignore[attr-defined]
            observation,
            CONFIG,
            start_h=0.0,
            end_h=6.0 - epsilon_h,
        )
        edge = controller._availability_equivalent_h(  # type: ignore[attr-defined]
            observation,
            CONFIG,
            start_h=0.0,
            end_h=6.0,
        )
        right = controller._availability_equivalent_h(  # type: ignore[attr-defined]
            observation,
            CONFIG,
            start_h=0.0,
            end_h=6.0 + epsilon_h,
        )

        assert 0.0 <= edge - left <= epsilon_h + 1e-12
        assert 0.0 <= right - edge <= epsilon_h + 1e-12


def test_climatology_multiplies_bandwise_clear_sky_geometry_numerically() -> None:
    factor = 0.35
    controller = ClimatologyControllerV2(
        DeliveryPlannerSettingsV2(climatological_resource_factor=factor)
    )
    observation = _observation(
        boundary=_boundary(time_s=CONFIG.solar.sunrise_h * 3_600.0),
    )
    daylight_h = CONFIG.solar.sunset_h - CONFIG.solar.sunrise_h
    interval_h = 6.0
    clear_sky_h = daylight_h / math.pi * (
        1.0 - math.cos(math.pi * interval_h / daylight_h)
    )

    actual_h = controller._availability_equivalent_h(  # type: ignore[attr-defined]
        observation,
        CONFIG,
        start_h=0.0,
        end_h=interval_h,
    )

    assert actual_h == pytest.approx(factor * clear_sky_h)
    assert controller.to_dict()["settings"]["climatological_resource_factor"] == factor


def test_issued_forecast_arm_dispatches_more_aggressively_for_low_solar() -> None:
    controller = IssuedForecastControllerV2()
    # Compatibility-off isolates forecast pacing from fixed commitment start power.
    boundary = _boundary(h2_level_kg=20.0, co2_level_kg=100.0)
    low = _observation(
        boundary=boundary,
        forecast=_forecast(0.10, 0.05),
        snapshot=None,
    )
    high = _observation(
        boundary=boundary,
        forecast=_forecast(0.90, 0.80),
        snapshot=None,
    )

    low_action = _decision(controller, low, profile=None).action
    high_action = _decision(controller, high, profile=None).action

    assert low_action.electrolyser_share > high_action.electrolyser_share
    assert low_action.dac_share > high_action.dac_share
    assert low_action.methanator_share > high_action.methanator_share
    assert low_action.battery_share < high_action.battery_share


def test_issued_mean_is_p10_invariant_and_risk_arm_is_separate_ablation() -> None:
    mean_only = IssuedForecastControllerV2()
    risk_adjusted = RiskAdjustedIssuedForecastControllerV2()
    optimistic_tail = _forecast(0.8, 0.8)
    adverse_tail = _forecast(0.8, 0.2)
    boundary = _boundary(h2_level_kg=20.0, co2_level_kg=100.0)
    optimistic = _observation(
        boundary=boundary,
        forecast=optimistic_tail,
        snapshot=None,
    )
    adverse = _observation(
        boundary=boundary,
        forecast=adverse_tail,
        snapshot=None,
    )

    assert _decision(mean_only, optimistic, profile=None).action == _decision(
        mean_only,
        adverse,
        profile=None,
    ).action
    assert _decision(risk_adjusted, optimistic, profile=None).action != _decision(
        risk_adjusted,
        adverse,
        profile=None,
    ).action

    mean_h = mean_only._availability_equivalent_h(  # type: ignore[attr-defined]
        adverse,
        CONFIG,
        start_h=0.0,
        end_h=6.0,
    )
    risk_h = risk_adjusted._availability_equivalent_h(  # type: ignore[attr-defined]
        adverse,
        CONFIG,
        start_h=0.0,
        end_h=6.0,
    )
    assert mean_h == pytest.approx(6.0 * 0.8)
    assert risk_h == pytest.approx(6.0 * 0.5)


def test_risk_adjusted_arm_uses_frozen_default_but_accepts_explicit_ablation() -> None:
    custom = DeliveryPlannerSettingsV2(forecast_p10_weight=0.25)
    custom_controller = RiskAdjustedIssuedForecastControllerV2(custom)

    assert RiskAdjustedIssuedForecastControllerV2().settings.forecast_p10_weight == 0.50
    assert custom_controller.settings is custom
    assert custom_controller.to_dict()["settings"] == custom.to_dict()
    with pytest.raises(ValueError, match="mean-only arm"):
        IssuedForecastControllerV2(custom)


def test_forecast_arm_ignores_windows_beyond_two_delivery_deadlines() -> None:
    first_six_hours = ((0.4, 0.2), (0.6, 0.3))
    first = _windowed_forecast(first_six_hours + ((0.0, 0.0), (0.0, 0.0)))
    changed_late = _windowed_forecast(first_six_hours + ((0.0, 0.0), (1.0, 1.0)))
    delivery = _delivery(steps_to_next=360)
    controller = IssuedForecastControllerV2()

    a = _decision(
        controller,
        _observation(delivery=delivery, forecast=first, snapshot=None),
        profile=None,
    ).action
    b = _decision(
        controller,
        _observation(delivery=delivery, forecast=changed_late, snapshot=None),
        profile=None,
    ).action
    assert a == b


def test_partial_aggregate_overlap_uses_explicit_uniform_band_interpretation() -> None:
    # At 5.5 h the controller credits 4.5 h of the 1-6 h published aggregate.
    # This is an explicit piecewise-constant disaggregation, not future truth.
    first = _windowed_forecast(
        ((0.4, 0.2), (0.1, 0.05), (0.3, 0.1), (0.0, 0.0))
    )
    changed_partial = _windowed_forecast(
        ((0.4, 0.2), (1.0, 1.0), (0.3, 0.1), (0.0, 0.0))
    )
    delivery = _delivery(steps_to_next=330)
    boundary = _boundary(h2_level_kg=100.0, co2_level_kg=550.0)
    controller = IssuedForecastControllerV2()

    first_observation = _observation(
        boundary=boundary,
        delivery=delivery,
        forecast=first,
        snapshot=None,
    )
    changed_observation = _observation(
        boundary=boundary,
        delivery=delivery,
        forecast=changed_partial,
        snapshot=None,
    )
    a = _decision(
        controller,
        first_observation,
        profile=None,
    ).action
    b = _decision(
        controller,
        changed_observation,
        profile=None,
    ).action
    first_h = controller._availability_equivalent_h(  # type: ignore[attr-defined]
        first_observation,
        CONFIG,
        start_h=0.0,
        end_h=5.5,
    )
    changed_h = controller._availability_equivalent_h(  # type: ignore[attr-defined]
        changed_observation,
        CONFIG,
        start_h=0.0,
        end_h=5.5,
    )

    assert first_h == pytest.approx(1.0 * 0.4 + 4.5 * 0.1)
    assert changed_h == pytest.approx(1.0 * 0.4 + 4.5 * 1.0)
    assert a != b


def test_forecast_carry_prebuilds_for_a_low_resource_next_delivery_period() -> None:
    current = ((0.6, 0.4), (0.8, 0.6))
    low_next = _windowed_forecast(current + ((0.0, 0.0), (0.0, 0.0)))
    high_next = _windowed_forecast(current + ((1.0, 1.0), (0.0, 0.0)))
    boundary = _boundary(h2_level_kg=100.0, co2_level_kg=550.0)
    controller = IssuedForecastControllerV2()

    low = _decision(
        controller,
        _observation(boundary=boundary, forecast=low_next, snapshot=None),
        profile=None,
    )
    high = _decision(
        controller,
        _observation(boundary=boundary, forecast=high_next, snapshot=None),
        profile=None,
    )

    assert low.action.methanator_share > high.action.methanator_share
    assert _offered_stack_kw(low, 5_000.0) > _offered_stack_kw(high, 5_000.0)
    assert "100.00 kg inferred next-delivery carry" in low.notes[0]
    assert "0.00 kg inferred next-delivery carry" in high.notes[0]


def test_delivery_amount_and_inventory_both_change_the_plan() -> None:
    controller = DeliveryAwareControllerV2()
    feedable = _boundary(h2_level_kg=100.0, co2_level_kg=550.0)
    small = _decision(
        controller,
        _observation(
            boundary=feedable,
            delivery=_delivery(next_requested_kg=50.0),
            snapshot=None,
        ),
        profile=None,
    ).action
    large = _decision(
        controller,
        _observation(
            boundary=feedable,
            delivery=_delivery(next_requested_kg=200.0),
            snapshot=None,
        ),
        profile=None,
    ).action
    stocked = _decision(
        controller,
        _observation(
            boundary=_boundary(
                ch4_level_kg=250.0,
                h2_level_kg=200.0,
                co2_level_kg=1_000.0,
            ),
            snapshot=None,
        ),
        profile=None,
    ).action

    assert small != large
    assert stocked != small
    assert large.methanator_share > small.methanator_share
    assert stocked.methanator_share == 0.0


def test_empty_feedstock_preempts_methanator_and_funds_upstream_units() -> None:
    controller = DeliveryAwareControllerV2()
    action = _decision(
        controller,
        _observation(
            boundary=_boundary(
                solar_available_kw=500.0,
                battery_soc=CONFIG.battery.min_soc,
                h2_level_kg=0.0,
                co2_level_kg=0.0,
            ),
            snapshot=None,
        ),
        profile=None,
    ).action

    assert action.methanator_share == 0.0
    assert action.electrolyser_share > 0.0
    assert action.dac_share > 0.0
    assert action.battery_support_kw == 0.0


def test_zero_solar_emits_forward_routing_and_sustainable_support() -> None:
    controller = DeliveryAwareControllerV2()
    observation = _observation(
        boundary=_boundary(solar_available_kw=0.0),
        snapshot=None,
    )
    action = _decision(controller, observation, profile=None).action

    assert sum(
        (
            action.battery_share,
            action.electrolyser_share,
            action.dac_share,
            action.methanator_share,
        )
    ) == pytest.approx(1.0)
    assert action.battery_share > 0.0
    assert action.dac_share > 0.0
    assert action.battery_support_kw > 0.0


@pytest.mark.parametrize(
    ("controller", "forecast"),
    (
        (DeliveryAwareControllerV2(), None),
        (ClimatologyControllerV2(), None),
        (IssuedForecastControllerV2(), _forecast(0.6, 0.3)),
        (RiskAdjustedIssuedForecastControllerV2(), _forecast(0.6, 0.3)),
    ),
)
def test_zero_boundary_routing_uses_rising_solar_before_storage(
    controller: (
        ClimatologyControllerV2
        | DeliveryAwareControllerV2
        | IssuedForecastControllerV2
        | RiskAdjustedIssuedForecastControllerV2
    ),
    forecast: PublicForecastIssueV2 | None,
) -> None:
    def simulator() -> PlantSimulator:
        item = PlantSimulator(CONFIG, electrolyser_commitment=PROFILE)
        item.load_scenario(
            Scenario(
                id="sunrise-routing",
                name="Sunrise routing",
                description="Decision boundary exactly at sunrise",
                start_hour=CONFIG.solar.sunrise_h,
                initial=InitialState(
                    battery_soc=0.80,
                    h2_kg=0.0,
                    co2_kg=0.0,
                    ch4_kg=0.0,
                ),
            )
        )
        assert item.telemetry.solar_available_kw == pytest.approx(0.0, abs=1e-9)
        return item

    routed = simulator()
    observation = _observation(
        boundary=routed.telemetry,
        forecast=forecast,
        snapshot=routed.electrolyser_commitment,
    )
    action = _decision(controller, observation).action
    routed_states = [routed.step(action) for _ in range(15)]

    all_zero = simulator()
    counterfactual = Action(
        battery_share=0.0,
        electrolyser_share=0.0,
        dac_share=0.0,
        methanator_share=0.0,
        battery_support_kw=action.battery_support_kw,
    )
    zero_states = [all_zero.step(counterfactual) for _ in range(15)]

    assert sum(state.solar_curtailed_kw for state in routed_states) == pytest.approx(0.0)
    assert sum(state.solar_curtailed_kw for state in zero_states) > 0.0
    assert sum(state.battery_to_electrolyser_kw for state in routed_states) < sum(
        state.battery_to_electrolyser_kw for state in zero_states
    )
    assert (
        routed_states[-1].battery_to_electrolyser_kw
        < routed_states[0].battery_to_electrolyser_kw
    )


def test_battery_support_preserves_constraint_reserve_for_full_held_decision() -> None:
    reserve = 0.40
    constraints = Constraints(battery_min_soc=reserve, battery_support_max_kw=2_500.0)
    boundary = _boundary(solar_available_kw=0.0, battery_soc=0.45)
    observation = _observation(boundary=boundary, snapshot=None)
    action = _decision(
        DeliveryAwareControllerV2(),
        observation,
        profile=None,
        constraints=constraints,
    ).action
    held_h = observation.decision_interval_s / 3_600.0
    usable_kwh = (boundary.battery_soc - reserve) * boundary.battery_capacity_kwh

    assert action.battery_support_kw > 0.0
    assert (
        action.battery_support_kw * held_h / CONFIG.battery.discharge_efficiency
        <= usable_kwh + 1e-9
    )

    at_reserve = replace(
        observation,
        state=replace(
            observation.state,
            boundary=replace(boundary, battery_soc=reserve),
        ),
    )
    assert (
        _decision(
            DeliveryAwareControllerV2(),
            at_reserve,
            profile=None,
            constraints=constraints,
        ).action.battery_support_kw
        == 0.0
    )


def test_running_floor_combines_direct_solar_with_support_without_false_rejection() -> None:
    snapshot = replace(
        BASE_SNAPSHOT,
        mode=ElectrolyserMode.RUNNING,
        previous_power_kw=1_200.0,
        ramp_up_headroom_kw=750.0,
        ramp_down_headroom_kw=750.0,
    )
    boundary = _boundary(
        solar_available_kw=600.0,
        h2_level_kg=0.0,
        co2_level_kg=0.0,
    )
    constraints = Constraints(battery_support_max_kw=600.0)

    decision = _decision(
        DeliveryAwareControllerV2(),
        _observation(
            boundary=boundary,
            delivery=_delivery(next_requested_kg=50.0),
            snapshot=snapshot,
        ),
        constraints=constraints,
    )
    action = decision.action

    # Contingent battery routing merely replaces discharge during dispatch; it
    # is not a third power source.  The real feasible pair is 600 kW of direct
    # solar plus 600 kW of support, exactly the 1.2 MW running floor.
    assert action.electrolyser_share == pytest.approx(1.0)
    assert action.battery_share == 0.0
    assert action.battery_support_kw == pytest.approx(600.0)
    assert _offered_stack_kw(decision, boundary.solar_available_kw) == pytest.approx(
        PROFILE.min_stable_load_frac * CONFIG.electrolyser.rated_kw
    )
    assert action == Action.sanitised(action)


def test_contingent_support_uses_solar_overlap_before_discharge() -> None:
    simulator = PlantSimulator(CONFIG, electrolyser_commitment=PROFILE)
    simulator.load_scenario(
        Scenario(
            id="contingent-support",
            name="Contingent support",
            description="Nominal noon boundary",
            start_hour=12.0,
            initial=InitialState(
                battery_soc=0.80,
                h2_kg=0.0,
                co2_kg=0.0,
                ch4_kg=0.0,
            ),
        )
    )
    observation = _observation(
        boundary=simulator.telemetry,
        snapshot=simulator.electrolyser_commitment,
    )
    action = _decision(DeliveryAwareControllerV2(), observation).action

    # The action declares a top-up and a battery solar route concurrently, but
    # dispatch cancels their overlap: abundant solar reaches the stack directly
    # and storage does not discharge.
    assert action.battery_share > 0.0
    assert action.battery_support_kw == pytest.approx(600.0)
    state = simulator.step(action)
    assert state.electrolyser_power_kw == pytest.approx(600.0)
    assert state.solar_to_electrolyser_kw == pytest.approx(600.0)
    assert state.battery_to_electrolyser_kw == 0.0
    assert state.battery_power_kw >= 0.0


def test_held_start_survives_an_intra_decision_solar_drop() -> None:
    simulator = PlantSimulator(CONFIG, electrolyser_commitment=PROFILE)
    simulator.load_scenario(
        Scenario(
            id="held-start-cloud",
            name="Held start cloud",
            description="A cloud arrives after the decision boundary",
            start_hour=12.0,
            initial=InitialState(
                battery_soc=0.80,
                h2_kg=0.0,
                co2_kg=0.0,
                ch4_kg=0.0,
            ),
        )
    )
    observation = _observation(
        boundary=simulator.telemetry,
        snapshot=simulator.electrolyser_commitment,
    )
    action = _decision(DeliveryAwareControllerV2(), observation).action

    # Compute once, then hold exactly 15 one-minute physics intervals. The 95%
    # solar drop begins after the first two intervals and is not a forecast input.
    simulator.inject_fault(
        FaultSpec(
            type="solar_drop",
            severity=0.95,
            start_s=simulator.time_s + 120.0,
            duration_s=900.0,
        )
    )
    states = [simulator.step(action) for _ in range(15)]
    snapshot = simulator.electrolyser_commitment
    assert snapshot is not None

    assert any(state.battery_to_electrolyser_kw > 0.0 for state in states)
    assert all(state.electrolyser_power_kw == pytest.approx(600.0) for state in states)
    assert snapshot.mode is ElectrolyserMode.COLD_STARTING
    assert snapshot.startup_remaining_s == pytest.approx(PROFILE.cold_start_s - 900.0)
    assert snapshot.forced_stops == 0
    assert snapshot.aborted_starts == 0


@pytest.mark.parametrize("solar_kw", (float("nan"), 0.0, 1.0, 250.0, 5_000.0))
def test_actions_are_finite_and_solar_shares_never_oversubscribe(solar_kw: float) -> None:
    observation = _observation(
        boundary=_boundary(solar_available_kw=solar_kw),
        snapshot=None,
    )
    action = _decision(
        DeliveryAwareControllerV2(), observation, profile=None
    ).action
    shares = (
        action.battery_share,
        action.electrolyser_share,
        action.dac_share,
        action.methanator_share,
    )
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in shares)
    assert sum(shares) <= 1.0
    assert math.isfinite(action.battery_support_kw)
    assert action == Action.sanitised(action)


def test_awkward_floating_point_bus_is_closed_before_action_sanitisation() -> None:
    action = _decision(
        DeliveryAwareControllerV2(),
        _observation(
            boundary=_boundary(
                solar_available_kw=3_333.3333333333335,
                h2_level_kg=37.037037037,
                co2_level_kg=203.7037037035,
            ),
            snapshot=None,
        ),
        profile=None,
    ).action

    assert sum(
        (
            action.battery_share,
            action.electrolyser_share,
            action.dac_share,
            action.methanator_share,
        )
    ) <= 1.0
    assert action == Action.sanitised(action)


def test_offline_components_are_never_commanded() -> None:
    offline = {
        name: ComponentHealth(status="offline", offline=True)
        for name in ("battery", "electrolyser", "dac", "methanator")
    }
    observation = _observation(
        boundary=_boundary(health=offline),
        snapshot=None,
    )
    action = _decision(
        DeliveryAwareControllerV2(), observation, profile=None
    ).action

    assert action == replace(
        action,
        battery_share=0.0,
        electrolyser_share=0.0,
        dac_share=0.0,
        methanator_share=0.0,
        battery_support_kw=0.0,
    )


@pytest.mark.parametrize(
    ("mode", "snapshot_changes", "expected_kw"),
    (
        (ElectrolyserMode.OFF, {"min_down_remaining_s": 0.0}, 600.0),
        (ElectrolyserMode.COOLDOWN, {"cooldown_remaining_s": 600.0}, 0.0),
        (
            ElectrolyserMode.COLD_STARTING,
            {"startup_remaining_s": 1_800.0},
            600.0,
        ),
        (
            ElectrolyserMode.RUNNING,
            {
                "previous_power_kw": 1_800.0,
                "ramp_up_headroom_kw": 750.0,
                "ramp_down_headroom_kw": 750.0,
            },
            1_200.0,
        ),
        (ElectrolyserMode.HOT_STANDBY, {"min_up_remaining_s": 600.0}, 1_200.0),
        (
            ElectrolyserMode.HOT_STARTING,
            {"startup_remaining_s": 300.0, "min_up_remaining_s": 600.0},
            1_200.0,
        ),
    ),
)
def test_each_commitment_mode_has_an_explicit_power_policy(
    mode: ElectrolyserMode,
    snapshot_changes: dict[str, float],
    expected_kw: float,
) -> None:
    snapshot = replace(BASE_SNAPSHOT, mode=mode, **snapshot_changes)
    observation = _observation(
        delivery=_delivery(next_requested_kg=50.0),
        snapshot=snapshot,
    )
    decision = _decision(DeliveryAwareControllerV2(), observation)

    assert _offered_stack_kw(decision, 5_000.0) == pytest.approx(expected_kw)


@pytest.mark.parametrize(
    ("mode", "changes", "expected_h"),
    (
        (ElectrolyserMode.RUNNING, {}, 0.0),
        (
            ElectrolyserMode.COLD_STARTING,
            {"startup_remaining_s": 1_800.0},
            0.5,
        ),
        (
            ElectrolyserMode.HOT_STARTING,
            {"startup_remaining_s": 300.0},
            300.0 / 3_600.0,
        ),
        (ElectrolyserMode.HOT_STANDBY, {}, PROFILE.hot_start_s / 3_600.0),
        (
            ElectrolyserMode.OFF,
            {"min_down_remaining_s": 600.0},
            (600.0 + PROFILE.cold_start_s) / 3_600.0,
        ),
        (
            ElectrolyserMode.COOLDOWN,
            {"cooldown_remaining_s": 600.0, "min_down_remaining_s": 1_200.0},
            (1_200.0 + PROFILE.cold_start_s) / 3_600.0,
        ),
    ),
)
def test_commitment_nonproductive_time_is_explicitly_discounted(
    mode: ElectrolyserMode,
    changes: dict[str, float],
    expected_h: float,
) -> None:
    snapshot = replace(BASE_SNAPSHOT, mode=mode, **changes)

    assert DeliveryAwareControllerV2._electrolyser_unavailable_h(  # type: ignore[attr-defined]
        snapshot,
        PROFILE,
    ) == pytest.approx(expected_h)


def test_cold_start_deadline_plan_reports_only_post_start_production_time() -> None:
    controller = DeliveryAwareControllerV2()
    observation = _observation(snapshot=BASE_SNAPSHOT)
    decision = _decision(controller, observation)
    expected_h = controller._availability_equivalent_h(  # type: ignore[attr-defined]
        observation,
        CONFIG,
        start_h=1.0,
        end_h=6.0,
    )

    assert f"electrolyser production horizon={expected_h:.3f} h" in decision.notes[2]
    assert "after 1.000 h unavailable" in decision.notes[2]


def test_running_controller_commands_endpoint_beyond_one_minute_ramp_headroom() -> None:
    snapshot = replace(
        BASE_SNAPSHOT,
        mode=ElectrolyserMode.RUNNING,
        previous_power_kw=600.0,
        ramp_up_headroom_kw=100.0,
        ramp_down_headroom_kw=100.0,
    )
    decision = _decision(
        DeliveryAwareControllerV2(),
        _observation(
            boundary=_boundary(h2_level_kg=0.0, co2_level_kg=0.0),
            delivery=_delivery(steps_to_next=60, next_requested_kg=1_000.0),
            snapshot=snapshot,
        ),
    )

    offered_kw = _offered_stack_kw(decision, 5_000.0)
    assert offered_kw == pytest.approx(CONFIG.electrolyser.rated_kw)
    assert offered_kw > snapshot.previous_power_kw + snapshot.ramp_up_headroom_kw


def test_held_endpoint_allows_minute_simulator_to_complete_successive_ramps() -> None:
    simulator = PlantSimulator(CONFIG, electrolyser_commitment=PROFILE)
    simulator.load_scenario(
        Scenario(
            id="held-ramp",
            name="Held ramp",
            description="Reach running mode before applying a held endpoint",
            start_hour=12.0,
            initial=InitialState(
                battery_soc=0.90,
                h2_kg=0.0,
                co2_kg=0.0,
                ch4_kg=0.0,
            ),
        )
    )
    for _ in range(int(PROFILE.cold_start_s / CONFIG.dt_s)):
        simulator.step(Action(electrolyser_share=1.0))
    snapshot = simulator.electrolyser_commitment
    assert snapshot is not None and snapshot.mode is ElectrolyserMode.RUNNING

    decision = _decision(
        DeliveryAwareControllerV2(),
        _observation(
            boundary=simulator.telemetry,
            delivery=_delivery(steps_to_next=60, next_requested_kg=1_000.0),
            snapshot=snapshot,
        ),
    )
    offered_kw = _offered_stack_kw(
        decision,
        simulator.telemetry.solar_available_kw,
    )
    powers = [simulator.step(decision.action).electrolyser_power_kw for _ in range(4)]

    assert offered_kw == pytest.approx(CONFIG.electrolyser.rated_kw)
    assert powers[0] == pytest.approx(
        snapshot.previous_power_kw + snapshot.ramp_up_headroom_kw
    )
    assert powers[1] > powers[0]
    assert powers[-1] > CONFIG.electrolyser.rated_kw - 1.0


def test_min_down_startup_continuity_and_running_ramp_are_respected() -> None:
    controller = DeliveryAwareControllerV2()
    blocked = replace(
        BASE_SNAPSHOT,
        mode=ElectrolyserMode.OFF,
        min_down_remaining_s=600.0,
    )
    long_start = replace(
        BASE_SNAPSHOT,
        mode=ElectrolyserMode.COLD_STARTING,
        startup_remaining_s=1_800.0,
    )
    completing_start = replace(
        BASE_SNAPSHOT,
        mode=ElectrolyserMode.COLD_STARTING,
        startup_remaining_s=300.0,
    )
    ramped = replace(
        BASE_SNAPSHOT,
        mode=ElectrolyserMode.RUNNING,
        previous_power_kw=1_300.0,
        ramp_up_headroom_kw=100.0,
        ramp_down_headroom_kw=100.0,
    )

    assert _offered_stack_kw(_decision(controller, _observation(snapshot=blocked)), 5_000.0) == 0.0
    assert _offered_stack_kw(
        _decision(
            controller,
            _observation(
                delivery=_delivery(next_requested_kg=50.0),
                snapshot=long_start,
            ),
        ),
        5_000.0,
    ) == pytest.approx(600.0)
    assert _offered_stack_kw(
        _decision(
            controller,
            _observation(
                delivery=_delivery(next_requested_kg=50.0),
                snapshot=completing_start,
            ),
        ),
        5_000.0,
    ) == pytest.approx(1_200.0)
    assert _offered_stack_kw(
        _decision(
            controller,
            _observation(
                delivery=_delivery(next_requested_kg=50.0),
                snapshot=ramped,
            ),
        ),
        5_000.0,
    ) == pytest.approx(1_200.0)


def test_min_up_uses_standby_then_no_need_commands_an_exact_controlled_stop() -> None:
    controller = DeliveryAwareControllerV2()
    stocked = _boundary(
        ch4_level_kg=250.0,
        h2_level_kg=200.0,
        co2_level_kg=1_000.0,
    )
    held = replace(
        BASE_SNAPSHOT,
        mode=ElectrolyserMode.RUNNING,
        previous_power_kw=1_200.0,
        ramp_up_headroom_kw=750.0,
        ramp_down_headroom_kw=0.0,
        min_up_remaining_s=600.0,
    )
    released = replace(held, min_up_remaining_s=0.0)

    standby = _decision(
        controller,
        _observation(boundary=stocked, snapshot=held),
    )
    stop = _decision(
        controller,
        _observation(boundary=stocked, snapshot=released),
    )
    assert _offered_stack_kw(standby, 5_000.0) == pytest.approx(60.0)
    assert stop.action.electrolyser_share == 0.0
    assert stop.action.battery_support_kw == 0.0


def test_compatibility_off_and_profile_identity_are_checked() -> None:
    controller = DeliveryAwareControllerV2()
    # The legacy path is supported when both sides declare commitment off.
    _decision(controller, _observation(snapshot=None), profile=None)

    with pytest.raises(ValueError, match="snapshot is present"):
        _decision(controller, _observation(snapshot=BASE_SNAPSHOT), profile=None)
    with pytest.raises(ValueError, match="snapshot is missing"):
        _decision(controller, _observation(snapshot=None), profile=PROFILE)
    mismatched = replace(BASE_SNAPSHOT, profile_sha256="f" * 64)
    with pytest.raises(ValueError, match="identity mismatch"):
        _decision(controller, _observation(snapshot=mismatched), profile=PROFILE)


def test_forecast_aware_controller_requires_a_public_issue() -> None:
    for controller in (
        IssuedForecastControllerV2(),
        RiskAdjustedIssuedForecastControllerV2(),
    ):
        with pytest.raises(ValueError, match="PublicForecastIssueV2"):
            _decision(controller, _observation(forecast=None))


def test_controller_module_has_no_environment_simulator_or_private_weather_access() -> None:
    import plant_backend.rl.v2.controllers as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert "ForecastIssueV2" not in imported_names
    assert "PlantSimulator" not in imported_names
    assert "PlantEnvV2" not in imported_names
    assert "solar_potential_kw" not in attributes
    assert "_episode" not in attributes


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("delivery_buffer_fraction", -0.01, ValueError),
        ("delivery_buffer_fraction", 0.26, ValueError),
        ("feedstock_reserve_fraction", float("nan"), ValueError),
        ("climatological_resource_factor", -0.01, ValueError),
        ("climatological_resource_factor", True, TypeError),
        ("forecast_p10_weight", 1.01, ValueError),
        ("forecast_p10_weight", True, TypeError),
    ),
)
def test_settings_are_validated(field: str, value: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        DeliveryPlannerSettingsV2(**{field: value})  # type: ignore[arg-type]


def test_settings_are_immutable_and_serialise_without_aliasing() -> None:
    settings = DeliveryPlannerSettingsV2(
        delivery_buffer_fraction=0.04,
        feedstock_reserve_fraction=0.02,
        climatological_resource_factor=0.35,
        forecast_p10_weight=0.75,
    )
    with pytest.raises(FrozenInstanceError):
        settings.delivery_buffer_fraction = 0.1  # type: ignore[misc]

    payload = settings.to_dict()
    payload["delivery_buffer_fraction"] = 0.2
    assert settings.delivery_buffer_fraction == 0.04
    assert json.loads(json.dumps(settings.to_dict(), allow_nan=False)) == {
        "delivery_buffer_fraction": 0.04,
        "feedstock_reserve_fraction": 0.02,
        "climatological_resource_factor": 0.35,
        "forecast_p10_weight": 0.75,
    }

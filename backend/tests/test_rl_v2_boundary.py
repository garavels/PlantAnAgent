"""Multi-rate V2 decision-boundary telemetry semantics."""

from __future__ import annotations

from dataclasses import replace

import pytest
from plant_sim import PlantState

from plant_backend.rl.v2.boundary import (
    ControlStateV2,
    IntervalSummaryV2,
    IntervalTelemetryAccumulator,
)


def test_interval_summary_integrates_flows_and_rates() -> None:
    accumulator = IntervalTelemetryAccumulator()
    first = PlantState(
        solar_available_kw=1_000.0,
        solar_potential_kw=2_000.0,
        solar_curtailed_kw=100.0,
        battery_power_kw=200.0,
        electrolyser_power_kw=400.0,
        electrolyser_load_frac=0.2,
        electrolyser_online=True,
        h2_production_kg_h=8.0,
    )
    second = replace(
        first,
        solar_available_kw=3_000.0,
        solar_potential_kw=4_000.0,
        battery_power_kw=-100.0,
        electrolyser_power_kw=800.0,
        electrolyser_load_frac=0.6,
        electrolyser_online=False,
        h2_production_kg_h=16.0,
    )
    accumulator.record(first, dt_h=0.25)
    accumulator.record(second, dt_h=0.25)
    summary = accumulator.summary()

    assert summary.physical_steps == 2
    assert summary.hours == pytest.approx(0.5)
    assert summary.solar_kwh == pytest.approx(1_000.0)
    assert summary.solar_available_mean_kw == pytest.approx(2_000.0)
    assert summary.solar_output_factor == pytest.approx(2.0 / 3.0)
    assert summary.battery_power_mean_kw == pytest.approx(50.0)
    assert summary.electrolyser_power_mean_kw == pytest.approx(600.0)
    assert summary.electrolyser_load_mean == pytest.approx(0.4)
    assert summary.h2_production_mean_kg_h == pytest.approx(12.0)
    assert summary.to_dict()["electrolyser_online_fraction"] == pytest.approx(0.5)


def test_control_state_keeps_boundary_levels_but_uses_interval_rates() -> None:
    boundary = PlantState(
        time_s=3_600.0,
        battery_soc=0.42,
        h2_level_kg=123.0,
        electrolyser_power_kw=1_000.0,
        electrolyser_load_frac=1.0 / 3.0,
        h2_production_kg_h=1.0,
    )
    interval = IntervalSummaryV2(
        physical_steps=15,
        hours=0.25,
        battery_net_kwh=-25.0,
        electrolyser_kwh=375.0,
        h2_produced_kg=10.0,
        electrolyser_load_fraction_h=0.125,
    )

    view = ControlStateV2(boundary, interval).encoder_telemetry()

    assert view.time_s == boundary.time_s
    assert view.battery_soc == boundary.battery_soc
    assert view.h2_level_kg == boundary.h2_level_kg
    assert view.battery_power_kw == pytest.approx(-100.0)
    assert view.electrolyser_power_kw == pytest.approx(1_500.0)
    assert view.electrolyser_load_frac == pytest.approx(0.5)
    assert view.h2_production_kg_h == pytest.approx(40.0)


def test_empty_interval_preserves_the_reset_boundary_exactly() -> None:
    boundary = PlantState(time_s=1_234.0, battery_soc=0.7, solar_available_kw=99.0)
    control = ControlStateV2(boundary)

    assert control.previous_interval == IntervalSummaryV2()
    assert control.encoder_telemetry() is boundary


@pytest.mark.parametrize("dt_h", [0.0, -0.1, float("nan"), float("inf")])
def test_accumulator_rejects_invalid_duration_without_mutation(dt_h: float) -> None:
    accumulator = IntervalTelemetryAccumulator()

    with pytest.raises(ValueError, match="positive and finite"):
        accumulator.record(PlantState(solar_available_kw=100.0), dt_h=dt_h)

    assert accumulator.summary() == IntervalSummaryV2()


def test_accumulator_rejects_non_numeric_duration_without_mutation() -> None:
    accumulator = IntervalTelemetryAccumulator()

    with pytest.raises(TypeError, match="real number"):
        accumulator.record(PlantState(), dt_h=True)  # type: ignore[arg-type]

    assert accumulator.summary() == IntervalSummaryV2()

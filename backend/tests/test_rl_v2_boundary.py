"""Multi-rate V2 decision-boundary telemetry semantics."""

from __future__ import annotations

from dataclasses import replace

import pytest
from plant_sim import (
    Action,
    DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE,
    PlantSimulator,
    PlantState,
)

from plant_backend.rl.v2.boundary import (
    CommitmentExposureV2,
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


def test_commitment_exposure_uses_exact_counter_deltas_and_covers_interval() -> None:
    simulator = PlantSimulator(
        electrolyser_commitment=DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE
    )
    before = simulator.electrolyser_commitment
    simulator.run(15, Action(battery_support_kw=3_000.0))
    after = simulator.electrolyser_commitment

    exposure = CommitmentExposureV2.between(before, after, interval_s=900.0)
    payload = exposure.to_dict()
    assert exposure.enabled is True
    assert exposure.cold_starts == 1
    assert exposure.hot_starts == 0
    assert exposure.startup_energy_kwh == pytest.approx(150.0)
    assert exposure.nonproductive_energy_kwh == pytest.approx(150.0)
    assert exposure.operating_time_s == 0.0
    assert exposure.absolute_load_movement_kw == pytest.approx(600.0)
    assert payload["mode_time_s"]["cold_starting"] == pytest.approx(900.0)
    assert sum(payload["mode_time_s"].values()) == pytest.approx(900.0)


def test_compatibility_off_exposure_is_explicit_legacy_mode_time() -> None:
    exposure = CommitmentExposureV2.between(None, None, interval_s=900.0)

    assert exposure.enabled is False
    assert exposure.to_dict()["mode_time_s"] == {
        "legacy": 900.0,
        "off": 0.0,
        "cold_starting": 0.0,
        "running": 0.0,
        "hot_standby": 0.0,
        "hot_starting": 0.0,
        "cooldown": 0.0,
    }


def test_commitment_exposure_rejects_a_backwards_mode_counter() -> None:
    simulator = PlantSimulator(
        electrolyser_commitment=DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE
    )
    before = simulator.electrolyser_commitment
    simulator.run(15, Action(battery_support_kw=3_000.0))
    after = simulator.electrolyser_commitment
    assert after is not None
    malformed = replace(
        after,
        mode_time_s=tuple(
            (mode, -60.0 if mode.value == "off" else seconds)
            for mode, seconds in after.mode_time_s
        ),
    )

    with pytest.raises(ValueError, match="off moved backwards"):
        CommitmentExposureV2.between(before, malformed, interval_s=900.0)


def test_commitment_exposure_compares_full_profile_identity_not_only_label() -> None:
    first = PlantSimulator(
        electrolyser_commitment=DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE
    ).electrolyser_commitment
    second = PlantSimulator(
        electrolyser_commitment=replace(
            DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE,
            cold_start_s=7_200.0,
        )
    ).electrolyser_commitment
    assert first is not None and second is not None
    assert first.profile_id == second.profile_id
    assert first.profile_sha256 != second.profile_sha256

    with pytest.raises(ValueError, match="profile cannot change"):
        CommitmentExposureV2.between(first, second, interval_s=0.0)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -1.0))
def test_commitment_exposure_rejects_corrupt_counters(value: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        CommitmentExposureV2(enabled=True, startup_energy_kwh=value)


def test_dark_interval_reports_unobservable_solar_factor_as_zero() -> None:
    summary = IntervalSummaryV2(physical_steps=15, hours=0.25)

    assert summary.solar_output_factor == 0.0
    assert summary.to_dict()["solar_output_factor"] == 0.0


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

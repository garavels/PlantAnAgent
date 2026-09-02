"""Commitment-alpha V2 observation contract and boundary semantics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from plant_sim import (
    Action,
    DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE,
    PlantConfig,
    PlantSimulator,
    PlantState,
)

from plant_backend.controllers import Constraints
from plant_backend.rl.v2 import spaces
from plant_backend.rl.v2.boundary import ControlStateV2, IntervalSummaryV2
from plant_backend.rl.v2.observation import DeliveryObservation, ObservationEncoderV2
from plant_backend.rl.v2.weather import ForecastIssueV2, ForecastWindowV2


EXPECTED_NAMES = (
    "time_of_day_sin",
    "time_of_day_cos",
    "solar_available",
    "solar_potential",
    "solar_output_factor",
    "solar_trend",
    "battery_soc",
    "battery_power",
    "battery_reserve_margin",
    "h2_fill",
    "co2_fill",
    "ch4_fill",
    "electrolyser_load",
    "electrolyser_online",
    "electrolyser_health",
    "dac_load",
    "dac_online",
    "dac_health",
    "methanator_load",
    "methanator_online",
    "methanator_health",
    "h2_production",
    "co2_production",
    "ch4_production",
    "electrolyser_cap",
    "dac_cap",
    "methanator_cap",
    "support_cap",
    "reserve_soc",
    "episode_progress",
    "delivery_time_to_next",
    "delivery_next_amount",
    "delivery_inventory_coverage",
    "delivery_fill_rate_to_date",
    "solar_forecast_0_1h_mean",
    "solar_forecast_0_1h_p10",
    "solar_forecast_1_6h_mean",
    "solar_forecast_1_6h_p10",
    "solar_forecast_6_12h_mean",
    "solar_forecast_6_12h_p10",
    "solar_forecast_12_24h_mean",
    "solar_forecast_12_24h_p10",
    "electrolyser_mode_legacy",
    "electrolyser_mode_off",
    "electrolyser_mode_cold_starting",
    "electrolyser_mode_running",
    "electrolyser_mode_hot_standby",
    "electrolyser_mode_hot_starting",
    "electrolyser_mode_cooldown",
    "electrolyser_previous_power",
    "electrolyser_ramp_up_headroom",
    "electrolyser_ramp_down_headroom",
    "electrolyser_startup_remaining",
    "electrolyser_cooldown_remaining",
    "electrolyser_min_up_remaining",
    "electrolyser_min_down_remaining",
    "electrolyser_shortfall_elapsed",
)


def _forecast() -> ForecastIssueV2:
    values = ((0.11, 0.01), (0.22, 0.02), (0.33, 0.03), (0.44, 0.04))
    names = ("0-1h", "1-6h", "6-12h", "12-24h")
    offsets_h = ((0.0, 1.0), (1.0, 6.0), (6.0, 12.0), (12.0, 24.0))
    windows = tuple(
        ForecastWindowV2(
            name=name,
            start_offset_s_from_issue=start_h * 3_600.0,
            end_offset_s_from_issue=end_h * 3_600.0,
            mean_peak_power_fraction=mean,
            p10_peak_power_fraction=p10,
        )
        for name, (start_h, end_h), (mean, p10) in zip(
            names,
            offsets_h,
            values,
            strict=True,
        )
    )
    return ForecastIssueV2(
        issued_at_physics_boundary_step=15,
        issued_at_offset_s=900.0,
        windows=windows,
        _issue_stream_sha256="0" * 64,
    )


def _delivery() -> DeliveryObservation:
    return DeliveryObservation(
        steps_to_next=30,
        delivery_period_steps=60,
        next_requested_kg=600.0,
        cumulative_requested_kg=1_200.0,
        cumulative_delivered_kg=900.0,
    )


def _encode(
    encoder: ObservationEncoderV2,
    state: ControlStateV2,
    *,
    forecast: ForecastIssueV2 | None = None,
) -> np.ndarray:
    return encoder.encode(
        state,
        constraints=Constraints(battery_min_soc=0.20),
        progress=0.35,
        delivery=_delivery(),
        forecast=forecast or _forecast(),
    )


def test_commitment_alpha_contract_has_exact_order_size_and_bounds() -> None:
    assert spaces.ENVIRONMENT_ID == "plant-env-v2-commitment-alpha"
    assert spaces.SPEC_VERSION == 3
    assert spaces.STAGE == "equipment-commitment-alpha"
    assert spaces.OBSERVATION_NAMES == EXPECTED_NAMES
    assert spaces.OBSERVATION_SIZE == 57

    negative_low = {
        "time_of_day_sin",
        "time_of_day_cos",
        "solar_trend",
        "battery_power",
        "battery_reserve_margin",
    }
    expected_low = np.asarray(
        [-1.0 if name in negative_low else 0.0 for name in EXPECTED_NAMES],
        dtype=np.float32,
    )
    expected_high = np.asarray(
        [2.0 if name == "delivery_inventory_coverage" else 1.0 for name in EXPECTED_NAMES],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(spaces.observation_low(), expected_low)
    np.testing.assert_array_equal(spaces.observation_high(), expected_high)
    actions = {field.name: field for field in spaces.ACTION_FIELDS}
    assert "deadband" in actions["electrolyser_share"].conversion
    assert "0.01" in actions["battery_support"].conversion


def test_encoder_rejects_an_invalid_commitment_profile_type() -> None:
    with pytest.raises(TypeError, match="ElectrolyserCommitmentProfile"):
        ObservationEncoderV2(PlantConfig(), commitment=True)  # type: ignore[arg-type]


def test_contract_describes_boundary_and_previous_interval_sources() -> None:
    fields = {field.name: field for field in spaces.OBSERVATION_FIELDS}

    assert "boundary.solar_available_kw" in fields["solar_available"].source
    assert "latest completed one-minute dispatch" in fields["solar_available"].source
    assert "boundary.battery_soc" in fields["battery_soc"].source
    assert "current boundary inventory" in fields["battery_soc"].source
    assert "previous_interval.battery_power_mean_kw" in fields["battery_power"].source
    assert "previous completed decision interval" in fields["battery_power"].source
    assert "previous_interval.electrolyser_load_mean" in fields["electrolyser_load"].source
    assert "boundary.ch4_level_kg" in fields["delivery_inventory_coverage"].source
    assert all(
        field.unit == "fraction of PV nameplate"
        for field in spaces.OBSERVATION_FIELDS[34:42]
    )
    assert "one-hot" in fields["electrolyser_mode_running"].source
    assert "previous_power_kw" in fields["electrolyser_previous_power"].source


def test_encoder_maps_interval_means_but_keeps_inventory_and_latest_solar_sample() -> None:
    config = PlantConfig()
    boundary = PlantState(
        time_s=9.0 * 3_600.0,
        solar_available_kw=2_500.0,
        solar_potential_kw=4_000.0,
        solar_output_factor=0.625,
        battery_soc=0.60,
        battery_power_kw=2_000.0,
        h2_fill=0.20,
        co2_fill=0.30,
        ch4_fill=0.40,
        h2_level_kg=400.0,
        h2_capacity_kg=2_000.0,
        co2_level_kg=1_800.0,
        co2_capacity_kg=6_000.0,
        ch4_level_kg=900.0,
        ch4_capacity_kg=3_000.0,
        electrolyser_load_frac=0.91,
        electrolyser_online=True,
        h2_production_kg_h=1.0,
        dac_load_frac=0.92,
        dac_online=True,
        co2_capture_kg_h=2.0,
        methanator_load_frac=0.93,
        methanator_online=True,
        ch4_production_kg_h=3.0,
    )
    interval = IntervalSummaryV2(
        physical_steps=15,
        hours=0.25,
        solar_kwh=25.0,
        solar_potential_kwh=50.0,
        battery_net_kwh=-62.5,
        h2_produced_kg=(config.electrolyser.rated_kw / 52.0) * 0.50 * 0.25,
        co2_captured_kg=(config.dac.rated_kw / 2.0) * 0.60 * 0.25,
        ch4_produced_kg=(config.methanator.rated_kw / 1.2) * 0.70 * 0.25,
        electrolyser_load_fraction_h=0.20 * 0.25,
        dac_load_fraction_h=0.30 * 0.25,
        methanator_load_fraction_h=0.40 * 0.25,
    )
    encoded = _encode(
        ObservationEncoderV2(config),
        ControlStateV2(boundary=boundary, previous_interval=interval),
    )
    at = {name: index for index, name in enumerate(spaces.OBSERVATION_NAMES)}

    assert encoded.shape == (57,)
    assert encoded.dtype == np.float32
    assert encoded[at["solar_available"]] == pytest.approx(0.50)
    assert encoded[at["solar_potential"]] == pytest.approx(0.80)
    assert encoded[at["solar_output_factor"]] == pytest.approx(0.625)
    assert encoded[at["battery_soc"]] == pytest.approx(0.60)
    assert encoded[at["battery_power"]] == pytest.approx(-0.10)
    assert encoded[at["electrolyser_load"]] == pytest.approx(0.20)
    assert encoded[at["dac_load"]] == pytest.approx(0.30)
    assert encoded[at["methanator_load"]] == pytest.approx(0.40)
    assert encoded[at["h2_production"]] == pytest.approx(0.50)
    assert encoded[at["co2_production"]] == pytest.approx(0.60)
    assert encoded[at["ch4_production"]] == pytest.approx(0.70)
    assert encoded[at["delivery_time_to_next"]] == pytest.approx(0.50)
    assert encoded[at["delivery_next_amount"]] == pytest.approx(0.20)
    assert encoded[at["delivery_inventory_coverage"]] == pytest.approx(1.50)
    assert encoded[at["delivery_fill_rate_to_date"]] == pytest.approx(0.75)
    np.testing.assert_allclose(
        encoded[34:42],
        np.asarray((0.11, 0.01, 0.22, 0.02, 0.33, 0.03, 0.44, 0.04)),
    )
    np.testing.assert_array_equal(
        encoded[42:],
        np.asarray((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) + (0.0,) * 8),
    )


def test_commitment_sidecar_encodes_mode_power_and_timer_state() -> None:
    config = PlantConfig()
    simulator = PlantSimulator(
        config,
        electrolyser_commitment=DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE,
    )
    simulator.step(Action(battery_support_kw=3_000.0))
    snapshot = simulator.electrolyser_commitment
    assert snapshot is not None
    assert len(snapshot.profile_sha256) == 64
    state = ControlStateV2(
        boundary=simulator.state,
        electrolyser_commitment=snapshot,
    )

    encoded = _encode(ObservationEncoderV2(config), state)
    at = {name: index for index, name in enumerate(spaces.OBSERVATION_NAMES)}
    mode_values = encoded[42:49]

    np.testing.assert_array_equal(
        mode_values,
        np.asarray((0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    )
    assert sum(mode_values) == pytest.approx(1.0)
    assert encoded[at["electrolyser_previous_power"]] == pytest.approx(0.20)
    assert encoded[at["electrolyser_ramp_up_headroom"]] == 0.0
    assert encoded[at["electrolyser_ramp_down_headroom"]] == 0.0
    assert encoded[at["electrolyser_startup_remaining"]] == pytest.approx(59.0 / 60.0)
    assert encoded[at["electrolyser_cooldown_remaining"]] == 0.0
    assert encoded[at["electrolyser_min_up_remaining"]] == 0.0
    assert encoded[at["electrolyser_min_down_remaining"]] == 0.0
    assert encoded[at["electrolyser_shortfall_elapsed"]] == 0.0


def test_encoder_rejects_same_label_with_different_profile_parameters() -> None:
    config = PlantConfig()
    changed = replace(
        DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE,
        cold_start_s=7_200.0,
    )
    simulator = PlantSimulator(config, electrolyser_commitment=changed)
    snapshot = simulator.electrolyser_commitment
    assert snapshot is not None
    assert snapshot.profile_id == DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE.profile_id
    assert snapshot.profile_sha256 != DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE.profile_sha256

    state = ControlStateV2(
        boundary=simulator.state,
        electrolyser_commitment=snapshot,
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        _encode(ObservationEncoderV2(config), state)


def test_encoder_rejects_nonfinite_simulator_owned_commitment_state() -> None:
    config = PlantConfig()
    simulator = PlantSimulator(
        config,
        electrolyser_commitment=DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE,
    )
    snapshot = simulator.electrolyser_commitment
    assert snapshot is not None
    state = ControlStateV2(
        boundary=simulator.state,
        electrolyser_commitment=replace(snapshot, previous_power_kw=float("nan")),
    )

    with pytest.raises(ValueError, match="finite and non-negative"):
        _encode(ObservationEncoderV2(config), state)


def test_solar_trend_uses_measured_decision_boundaries_and_reset_clears_it() -> None:
    config = PlantConfig()
    encoder = ObservationEncoderV2(config)
    interval = IntervalSummaryV2(
        physical_steps=15,
        hours=0.25,
        solar_kwh=1_000.0,
        solar_potential_kwh=1_100.0,
    )
    first = ControlStateV2(
        boundary=PlantState(solar_available_kw=1_000.0),
        previous_interval=interval,
    )
    second = replace(
        first,
        boundary=replace(first.boundary, solar_available_kw=1_750.0),
        previous_interval=replace(interval, solar_kwh=25.0),
    )
    trend_index = spaces.OBSERVATION_NAMES.index("solar_trend")

    assert _encode(encoder, first)[trend_index] == pytest.approx(0.0)
    assert _encode(encoder, second)[trend_index] == pytest.approx(0.15)

    encoder.reset()
    assert _encode(encoder, second)[trend_index] == pytest.approx(0.0)

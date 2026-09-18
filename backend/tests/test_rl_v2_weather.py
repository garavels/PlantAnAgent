"""Causality, replay, and statistical contracts for V2 synthetic weather."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from plant_sim import SolarFactorTape

from plant_backend.rl.v2.weather import (
    FORECAST_STREAM_NAME,
    FORECAST_WINDOWS_H,
    TRUTH_STREAM_NAME,
    WEATHER_GENERATOR_VERSION,
    WeatherConfigV2,
    build_weather_episode_from_tape,
    generate_truth_tape,
    generate_weather_episode,
)


DT_S = 900.0
PHYSICS_STEPS = 16
DECISION_STRIDE = 2


def _seeds(seed: int) -> tuple[np.random.SeedSequence, np.random.SeedSequence]:
    truth, forecast = np.random.SeedSequence(seed).spawn(2)
    return truth, forecast


def _episode(
    seed: int = 7,
    *,
    config: WeatherConfigV2 | None = None,
    envelope: tuple[float, ...] | None = None,
):
    truth, forecast = _seeds(seed)
    return generate_weather_episode(
        config or WeatherConfigV2(forecast_ensemble_members=16),
        physics_steps=PHYSICS_STEPS,
        dt_s=DT_S,
        decision_stride=DECISION_STRIDE,
        truth_seed=truth,
        forecast_seed=forecast,
        clear_sky_peak_fractions=envelope,
        clear_sky_envelope_id="test-envelope-v1",
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (WeatherConfigV2(latent_median_resource_factor=0.0), "strictly within"),
        (WeatherConfigV2(latent_stationary_sd=0.0), "positive and finite"),
        (WeatherConfigV2(autocorrelation_half_life_h=float("nan")), "positive"),
        (WeatherConfigV2(forecast_ensemble_members=9), "at least 10"),
        (WeatherConfigV2(forecast_analysis_latent_sd=-0.1), "non-negative"),
        (WeatherConfigV2(forecast_innovation_scale=0.0), "positive"),
        (
            WeatherConfigV2(forecast_windows_h=((0.0, 1.0), (1.0, 24.0))),
            "canonical",
        ),
    ],
)
def test_weather_configuration_is_strict_and_frozen(
    config: WeatherConfigV2,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate()

    valid = WeatherConfigV2()
    with pytest.raises(FrozenInstanceError):
        valid.latent_stationary_sd = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"physics_steps": 0}, "positive integer"),
        ({"dt_s": 0.0}, "positive and finite"),
        ({"decision_stride": 0}, "positive integer"),
        ({"physics_steps": 15}, "whole number of decisions"),
        ({"clear_sky_envelope_id": ""}, "non-empty"),
    ],
)
def test_episode_timing_and_envelope_identity_are_validated(
    changes: dict[str, object],
    message: str,
) -> None:
    truth, forecast = _seeds(9)
    kwargs: dict[str, object] = {
        "physics_steps": PHYSICS_STEPS,
        "dt_s": DT_S,
        "decision_stride": DECISION_STRIDE,
        "truth_seed": truth,
        "forecast_seed": forecast,
    }
    kwargs.update(changes)
    with pytest.raises(ValueError, match=message):
        generate_weather_episode(WeatherConfigV2(), **kwargs)  # type: ignore[arg-type]


def test_envelope_must_cover_full_horizon_and_contain_peak_fractions() -> None:
    truth, forecast = _seeds(10)
    kwargs = {
        "physics_steps": PHYSICS_STEPS,
        "dt_s": DT_S,
        "decision_stride": DECISION_STRIDE,
        "truth_seed": truth,
        "forecast_seed": forecast,
    }
    with pytest.raises(ValueError, match="full forecast horizon"):
        generate_weather_episode(
            WeatherConfigV2(),
            **kwargs,
            clear_sky_peak_fractions=(1.0,) * PHYSICS_STEPS,
        )

    truth, forecast = _seeds(10)
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        generate_weather_episode(
            WeatherConfigV2(),
            **{**kwargs, "truth_seed": truth, "forecast_seed": forecast},
            clear_sky_peak_fractions=(1.1,) * 112,
        )


def test_truth_and_forecast_streams_must_be_independent() -> None:
    with pytest.raises(ValueError, match="independent random streams"):
        generate_weather_episode(
            WeatherConfigV2(),
            physics_steps=PHYSICS_STEPS,
            dt_s=DT_S,
            decision_stride=DECISION_STRIDE,
            truth_seed=np.random.SeedSequence(44),
            forecast_seed=np.random.SeedSequence(44),
        )


def test_exact_replay_includes_truth_forecasts_and_private_hashes() -> None:
    first = _episode(81)
    replay = _episode(81)

    assert first.tape == replay.tape
    assert first.issues == replay.issues
    assert first.private_audit_manifest() == replay.private_audit_manifest()
    assert first.truth_sha256 == replay.truth_sha256
    assert first.forecasts_sha256 == replay.forecasts_sha256
    assert first.episode_sha256 == replay.episode_sha256
    assert len(first.episode_sha256) == 64
    int(first.episode_sha256, 16)
    manifest = first.private_audit_manifest()
    private_hashes = manifest["private_hashes"]
    assert isinstance(private_hashes, dict)
    issue_streams = private_hashes["forecast_issue_stream_sha256"]
    assert isinstance(issue_streams, list)
    assert len(issue_streams) == len(first.issues)
    assert len(set(issue_streams)) == len(first.issues)
    json.dumps(manifest, allow_nan=False, sort_keys=True)


def test_generator_version_has_fixed_seed_golden_digests() -> None:
    truth, forecast = _seeds(1_234)
    generated = generate_weather_episode(
        WeatherConfigV2(forecast_ensemble_members=16),
        physics_steps=16,
        dt_s=900.0,
        decision_stride=2,
        truth_seed=truth,
        forecast_seed=forecast,
        clear_sky_envelope_id="golden-unit-v1",
    )

    assert WEATHER_GENERATOR_VERSION == "synthetic-logit-ou-v1"
    assert generated.truth_sha256 == (
        "fa94db030524cdc608b3732e4900bb202ec42440e81262557393246e70324ddf"
    )
    assert generated.forecasts_sha256 == (
        "64f8497d6085adeb5a23bb80790c70c1c79de9d5f02db77f5e1b841e4f4a2b6c"
    )
    assert generated.episode_sha256 == (
        "1826925b81542376a86283404d0f2e8a6b401df3ea4de9974ab9a0761690308e"
    )


def test_seed_streams_are_named_separate_and_accept_generators() -> None:
    first = _episode(17)
    assert tuple(first.stream_keys) == (TRUTH_STREAM_NAME, FORECAST_STREAM_NAME)
    assert first.truth_stream_key.spawn_key == (0,)
    assert first.forecast_stream_key.spawn_key == (1,)

    same_truth = generate_weather_episode(
        WeatherConfigV2(forecast_ensemble_members=16),
        physics_steps=PHYSICS_STEPS,
        dt_s=DT_S,
        decision_stride=DECISION_STRIDE,
        truth_seed=np.random.SeedSequence(101),
        forecast_seed=np.random.SeedSequence(202),
    )
    changed_forecast = generate_weather_episode(
        WeatherConfigV2(forecast_ensemble_members=16),
        physics_steps=PHYSICS_STEPS,
        dt_s=DT_S,
        decision_stride=DECISION_STRIDE,
        truth_seed=np.random.SeedSequence(101),
        forecast_seed=np.random.SeedSequence(303),
    )
    assert same_truth.tape == changed_forecast.tape
    assert same_truth.truth_sha256 == changed_forecast.truth_sha256
    assert same_truth.forecasts_sha256 != changed_forecast.forecasts_sha256

    generated = generate_weather_episode(
        WeatherConfigV2(forecast_ensemble_members=10),
        physics_steps=4,
        dt_s=DT_S,
        decision_stride=2,
        truth_seed=np.random.default_rng(11),
        forecast_seed=np.random.default_rng(12),
    )
    assert generated.truth_stream_key.source_kind.endswith("PCG64")
    assert generated.forecast_stream_key.source_kind.endswith("PCG64")


def test_random_sources_are_snapshotted_not_consumed_and_can_be_reused() -> None:
    truth, forecast = _seeds(117)

    first = generate_weather_episode(
        WeatherConfigV2(forecast_ensemble_members=10),
        physics_steps=4,
        dt_s=DT_S,
        decision_stride=2,
        truth_seed=truth,
        forecast_seed=forecast,
    )
    replay = generate_weather_episode(
        WeatherConfigV2(forecast_ensemble_members=10),
        physics_steps=4,
        dt_s=DT_S,
        decision_stride=2,
        truth_seed=truth,
        forecast_seed=forecast,
    )

    assert truth.state["n_children_spawned"] == 0
    assert forecast.state["n_children_spawned"] == 0
    assert first == replay


def test_canonical_stream_identity_supports_mt19937_and_numpy_scalars() -> None:
    config = WeatherConfigV2(
        latent_median_resource_factor=np.float32(0.7),  # type: ignore[arg-type]
        latent_stationary_sd=np.float32(0.75),  # type: ignore[arg-type]
        forecast_ensemble_members=np.int64(10),  # type: ignore[arg-type]
    )
    generated = generate_weather_episode(
        config,
        physics_steps=4,
        dt_s=np.float64(DT_S),
        decision_stride=np.int64(2),
        truth_seed=np.random.Generator(np.random.MT19937(1)),
        forecast_seed=np.random.Generator(np.random.MT19937(2)),
    )

    assert generated.truth_stream_key.source_kind.endswith("MT19937")
    assert isinstance(generated.config.to_dict()["forecast_ensemble_members"], int)
    manifest = generated.private_audit_manifest()
    initial_state = manifest["random_streams"][TRUTH_STREAM_NAME][  # type: ignore[index]
        "initial_state"
    ]
    assert "key" in initial_state["state"]  # type: ignore[index,operator]
    json.dumps(manifest, allow_nan=False, sort_keys=True)


def test_truth_is_bounded_variable_and_autocorrelated() -> None:
    config = WeatherConfigV2(
        latent_stationary_sd=0.75,
        autocorrelation_half_life_h=1.0,
    )
    tape = generate_truth_tape(
        config,
        physics_steps=20_000,
        dt_s=300.0,
        truth_seed=np.random.SeedSequence(404),
    )
    factors = np.asarray(tape.boundary_factors)
    latent = np.log(factors / (1.0 - factors))
    lag_one = float(np.corrcoef(latent[:-1], latent[1:])[0, 1])
    lag_hour = float(np.corrcoef(latent[:-12], latent[12:])[0, 1])

    assert np.all((0.0 <= factors) & (factors <= 1.0))
    assert float(np.var(factors)) > 0.005
    assert 0.90 < lag_one < 0.98
    assert 0.40 < lag_hour < 0.65
    assert lag_one > lag_hour > 0.0


def test_short_valid_half_life_remains_numerically_finite() -> None:
    weather = _episode(
        411,
        config=WeatherConfigV2(
            autocorrelation_half_life_h=0.015,
            forecast_ensemble_members=10,
        ),
    )

    assert all(
        np.isfinite(value)
        for issue in weather.issues
        for window in issue.windows
        for value in (
            window.mean_peak_power_fraction,
            window.p10_peak_power_fraction,
        )
    )


def test_forecasts_are_issued_only_at_decision_boundaries_with_explicit_units() -> None:
    weather = _episode(12)

    assert weather.issue_steps == tuple(range(0, PHYSICS_STEPS, DECISION_STRIDE))
    assert weather.issue_at_boundary(3) is None
    assert weather.current_issue(3).issued_at_physics_boundary_step == 2
    assert weather.current_issue(PHYSICS_STEPS).issued_at_physics_boundary_step == 14

    issue = weather.issue_at_boundary(4)
    assert issue is not None
    assert issue.issued_at_offset_s == 4 * DT_S
    assert tuple(window.name for window in issue.windows) == (
        "0-1h",
        "1-6h",
        "6-12h",
        "12-24h",
    )
    assert tuple(
        (
            window.start_offset_s_from_issue / 3_600.0,
            window.end_offset_s_from_issue / 3_600.0,
        )
        for window in issue.windows
    ) == FORECAST_WINDOWS_H
    assert issue.to_dict()["issue_time_basis"] == "seconds since episode start"


def test_forecast_p10_is_bounded_by_mean_and_clear_sky_envelope() -> None:
    # At night (zero deterministic envelope), available PV/nameplate forecast is
    # zero even though the private weather resource factor remains non-zero.
    zero_envelope = (0.0,) * 112
    weather = _episode(24, envelope=zero_envelope)
    assert any(factor > 0.0 for factor in weather.tape.boundary_factors)

    for issue in weather.issues:
        for window in issue.windows:
            assert 0.0 <= window.p10_peak_power_fraction
            assert window.p10_peak_power_fraction <= window.mean_peak_power_fraction
            assert window.mean_peak_power_fraction == 0.0


def test_one_hot_envelope_is_assigned_to_the_correct_half_open_window() -> None:
    required = 112
    envelope = [0.0] * required
    envelope[0] = 1.0
    weather = _episode(39, envelope=tuple(envelope))
    issue = weather.current_issue(0)

    assert issue.window("0-1h").mean_peak_power_fraction > 0.0
    assert all(
        issue.window(name).mean_peak_power_fraction == 0.0
        for name in ("1-6h", "6-12h", "12-24h")
    )


def test_post_issue_truth_counterfactual_cannot_change_already_issued_forecast() -> None:
    config = WeatherConfigV2(forecast_ensemble_members=16)
    truth_seed, forecast_seed = _seeds(515)
    original = generate_weather_episode(
        config,
        physics_steps=PHYSICS_STEPS,
        dt_s=DT_S,
        decision_stride=DECISION_STRIDE,
        truth_seed=truth_seed,
        forecast_seed=forecast_seed,
    )
    issue_step = 4
    original_factors = original.tape.boundary_factors
    changed_factors = original_factors[: issue_step + 1] + tuple(
        1.0 - factor for factor in original_factors[issue_step + 1 :]
    )
    counterfactual_tape = SolarFactorTape(
        dt_s=DT_S,
        boundary_factors=changed_factors,
    )
    counterfactual = build_weather_episode_from_tape(
        config,
        tape=counterfactual_tape,
        decision_stride=DECISION_STRIDE,
        truth_stream_key=original.truth_stream_key,
        forecast_seed=_seeds(515)[1],
    )

    assert original.truth_sha256 != counterfactual.truth_sha256
    original_at_issue = original.issue_at_boundary(issue_step)
    counterfactual_at_issue = counterfactual.issue_at_boundary(issue_step)
    assert original_at_issue is not None
    assert counterfactual_at_issue is not None
    assert original_at_issue.to_dict() == counterfactual_at_issue.to_dict()
    # A later issue is allowed to change after the altered truth becomes current.
    later_step = issue_step + DECISION_STRIDE
    original_later = original.issue_at_boundary(later_step)
    counterfactual_later = counterfactual.issue_at_boundary(later_step)
    assert original_later is not None
    assert counterfactual_later is not None
    assert original_later.to_dict() != counterfactual_later.to_dict()


def test_canonical_hashes_track_envelope_values_and_identity_stably() -> None:
    required = 112
    envelope = tuple((index % 8) / 8.0 for index in range(required))
    first = _episode(901, envelope=envelope)
    replay = _episode(901, envelope=envelope)
    assert first.clear_sky_envelope_sha256 == replay.clear_sky_envelope_sha256
    assert first.episode_sha256 == replay.episode_sha256

    changed_values = list(envelope)
    changed_values[5] += 0.01
    changed = _episode(901, envelope=tuple(changed_values))
    assert changed.clear_sky_envelope_sha256 != first.clear_sky_envelope_sha256
    assert changed.forecasts_sha256 != first.forecasts_sha256
    assert changed.episode_sha256 != first.episode_sha256

    truth, forecast = _seeds(901)
    relabelled = generate_weather_episode(
        WeatherConfigV2(forecast_ensemble_members=16),
        physics_steps=PHYSICS_STEPS,
        dt_s=DT_S,
        decision_stride=DECISION_STRIDE,
        truth_seed=truth,
        forecast_seed=forecast,
        clear_sky_peak_fractions=envelope,
        clear_sky_envelope_id="same-values-different-provenance",
    )
    assert relabelled.clear_sky_envelope_sha256 == first.clear_sky_envelope_sha256
    assert relabelled.forecasts_sha256 == first.forecasts_sha256
    assert relabelled.episode_sha256 != first.episode_sha256


def test_stream_key_mapping_is_detached_from_the_frozen_episode() -> None:
    weather = _episode(3)
    detached = weather.stream_keys
    detached.pop(TRUTH_STREAM_NAME)
    assert tuple(weather.stream_keys) == (TRUTH_STREAM_NAME, FORECAST_STREAM_NAME)

    issue = weather.issues[0]
    with pytest.raises(FrozenInstanceError):
        issue.issued_at_offset_s = 1.0  # type: ignore[misc]


def test_weather_configuration_serialization_is_canonical() -> None:
    config = WeatherConfigV2()
    assert replace(config).to_dict() == config.to_dict()
    assert config.to_dict()["forecast_windows_h"] == FORECAST_WINDOWS_H


@pytest.mark.parametrize("invalid", [True, "0.5"])
def test_clear_sky_envelope_rejects_coercible_non_numbers(invalid: object) -> None:
    truth, forecast = _seeds(919)

    with pytest.raises(ValueError, match="real number"):
        generate_weather_episode(
            WeatherConfigV2(forecast_ensemble_members=10),
            physics_steps=4,
            dt_s=DT_S,
            decision_stride=2,
            truth_seed=truth,
            forecast_seed=forecast,
            clear_sky_peak_fractions=(invalid,) * 98,  # type: ignore[arg-type]
        )


def test_private_manifest_states_what_is_and_is_not_self_materialising() -> None:
    synthetic = _episode(61).private_audit_manifest()
    assert synthetic["replay"]["truth_origin_kind"] == "synthetic_generator"  # type: ignore[index]
    assert synthetic["replay"]["random_initial_states_embedded"] is True  # type: ignore[index]
    assert synthetic["replay"]["truth_tape_embedded"] is False  # type: ignore[index]

    original = _episode(62)
    external = build_weather_episode_from_tape(
        original.config,
        tape=original.tape,
        decision_stride=original.decision_stride_steps,
        truth_stream_key=original.truth_stream_key,
        forecast_seed=_seeds(62)[1],
    ).private_audit_manifest()
    assert external["replay"]["truth_origin_kind"] == "external_tape"  # type: ignore[index]
    requirements = external["replay"]["requirements"]  # type: ignore[index]
    assert any(
        "retain the external SolarFactorTape" in item
        for item in requirements  # type: ignore[union-attr]
    )

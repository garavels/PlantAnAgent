"""Episode-level integration of private weather truth and forecast vintages."""

from __future__ import annotations

from dataclasses import replace

from plant_sim import PlantConfig

from plant_backend.rl.v2.config import EpisodeConfigV2, sample_episode


SHORT = EpisodeConfigV2(horizon_h=2.0, delivery_period_h=1.0)


def test_sampled_episode_contains_a_complete_replayable_weather_contract() -> None:
    plant = PlantConfig()
    first = sample_episode(SHORT, plant, seed=73)
    replay = sample_episode(SHORT, plant, seed=73)

    assert first.weather.tape.interval_count == first.physics_steps
    assert first.weather.tape.dt_s == plant.dt_s
    assert first.weather.issue_steps == tuple(
        range(0, first.physics_steps, first.decision_stride)
    )
    assert first.weather.episode_sha256 == replay.weather.episode_sha256
    assert first.weather.tape == replay.weather.tape
    assert first.audit_manifest() == replay.audit_manifest()


def test_weather_parameter_change_does_not_perturb_initial_inventories() -> None:
    plant = PlantConfig()
    base = sample_episode(SHORT, plant, seed=19)
    changed_config = replace(
        SHORT,
        weather=replace(
            SHORT.weather,
            latent_median_resource_factor=0.55,
        ),
    )
    changed = sample_episode(changed_config, plant, seed=19)

    assert base.scenario.initial == changed.scenario.initial
    assert base.commitments == changed.commitments
    assert base.weather.truth_sha256 != changed.weather.truth_sha256
    assert base.weather.forecasts_sha256 != changed.weather.forecasts_sha256


def test_weather_child_streams_have_stable_named_spawn_keys() -> None:
    sampled = sample_episode(SHORT, PlantConfig(), seed=5)

    assert tuple(sampled.stream_spawn_keys) == ("inventories", "weather", "contracts")
    assert sampled.stream_spawn_keys["weather"] == (1,)
    assert sampled.weather.truth_stream_key.spawn_key == (1, 0)
    assert sampled.weather.forecast_stream_key.spawn_key == (1, 1)


def test_episode_audit_records_weather_provenance_without_materialising_truth() -> None:
    sampled = sample_episode(SHORT, PlantConfig(), seed=29)
    manifest = sampled.audit_manifest()
    weather = manifest["weather"]

    assert weather["weather_generator_version"] == "synthetic-logit-ou-v1"
    assert weather["time_convention"]["physics_timestep_s"] == 60.0
    assert weather["clear_sky_envelope"]["id"] == "plant-sim-half-sine-v1"
    assert len(weather["private_hashes"]["truth_sha256"]) == 64
    assert "boundary_factors" not in str(weather)


def test_clear_sky_envelope_respects_episode_clock_and_known_night_zero() -> None:
    sampled = sample_episode(SHORT, PlantConfig(), seed=31)
    initial_issue = sampled.weather.current_issue(0)

    # The episode begins at sunrise (06:00), so the first interval has exactly
    # zero deterministic clear-sky power even though its cloud factor is positive.
    first_hour = initial_issue.window("0-1h")
    assert 0.0 < first_hour.mean_peak_power_fraction < 1.0
    assert 0.0 <= first_hour.p10_peak_power_fraction <= first_hour.mean_peak_power_fraction

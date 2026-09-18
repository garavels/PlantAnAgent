"""Engineering checks for an isolated diagnostic; no PPO outcomes are used."""
import json
import numpy as np
import pytest

from docs.research.tools.startup_stress_v1.environment import (
    CONDITIONS, DIAGNOSTIC_SEED, ENVIRONMENT_ID, StartupStressEnv,
)
from docs.research.tools.wholeplant_ppo_v1.evaluate import MinuteAudit
from docs.research.tools.wholeplant_ppo_v1.physics import WholePlantAction, action_to_vector
from plant_sim import SolarArray, SolarFactorTape
from plant_backend.rl.v2.weather import build_weather_episode_from_tape


@pytest.mark.parametrize("condition", CONDITIONS)
def test_diagnostic_is_initialized_before_observation_with_unchanged_mission(condition):
    env = StartupStressEnv(condition, record_minutes=True)
    try:
        obs, info = env.reset(seed=DIAGNOSTIC_SEED)
        state = env.simulator.state
        assert info["environment_id"] == ENVIRONMENT_ID
        assert env.split == "diagnostic" and env.arm == "shared-dynamic"
        assert obs.shape == (149,) and env.observation_space.contains(obs)
        assert env.action_space.shape == (7,)
        np.testing.assert_array_equal(obs, env.public_control_input())
        env.public_control_input()[:] = -100
        np.testing.assert_array_equal(obs, env.public_control_input())
        assert (state.battery_soc, state.h2_level_kg, state.co2_level_kg, state.ch4_level_kg) == (.25, 100, 600, 120)
        assert state.time_s == 6 * 3600
        assert state.dac_readiness_kwh == state.methanator_readiness_kwh == 0
        assert env.simulator.electrolyser_commitment.mode == "off"
        assert all(not h.offline and h.degradation == 0 for h in state.health.values())
        assert env._episode.scenario.faults == env.disturbance_episode.faults == ()
        assert env.episode_physics_steps == 10080 and env.episode_decisions == 672
        assert len(env._episode.commitments) == 28
        assert [(c.due_step, c.requested_kg) for c in env._episode.commitments] == [(360*i, 180) for i in range(1, 29)]
        manifest = env.audit_manifest()
        assert manifest["environment"]["environment_id"] == ENVIRONMENT_ID
        assert manifest["disturbances"]["schedule"] == []
        assert isinstance(env.frozen_source.expected_solar_kw, tuple)
        assert len(env.frozen_source.expected_solar_kw) == 10080
        assert env._episode.weather.config == env.episode_config.weather
        assert env._episode.weather.truth_origin_kind == "external_tape"
    finally:
        env.close()


def test_zero_solar_covers_first_delivery_and_forecasts_do_not_reveal_repair():
    env = StartupStressEnv("solar-loss", record_minutes=True)
    try:
        obs, _ = env.reset(seed=DIAGNOSTIC_SEED)
        assert env._episode.weather.tape.boundary_factors[:361] == (0.0,) * 361
        assert env._episode.weather.tape.boundary_factors[361:] == (1.0,) * (10080 - 360)
        assert env.frozen_source.expected_solar_kw[:360] == (0.0,) * 360
        assert env.frozen_source.expected_solar_kw[360] > 0
        audit = MinuteAudit(env)
        audit.add(env.simulator.state)
        for _ in range(24):
            assert all(0 <= w.p10_peak_power_fraction <= w.mean_peak_power_fraction <= 1
                       for w in env.current_forecast.windows)
            np.testing.assert_array_equal(obs, env.public_control_input())
            obs, _, terminated, truncated, _ = env.step(-np.ones(7))
            assert not terminated and not truncated
            assert all(s.solar_available_kw == 0 for s in env.last_minute_truth_states[1:])
            for state in env.last_minute_truth_states[1:]:
                audit.add(state)
        assert env.physics_steps_taken == 360
        first = env.simulator.delivery_audit()["settlements"][0]
        assert (first["due_step"], first["delivered_kg"], first["unmet_kg"]) == (360, 120, 60)
        # At the deadline, repair has not yet appeared in any completed interval.
        assert env.telemetry.solar_available_kw == 0
        assert env.current_forecast == env._episode.weather.current_issue(360).public()
        assert obs[2] == 0
        obs, *_ = env.step(-np.ones(7))
        assert all(s.solar_available_kw > 0 for s in env.last_minute_truth_states[1:])
        for state in env.last_minute_truth_states[1:]:
            audit.add(state)
        assert env.physics_steps_taken == 375
        assert env.current_forecast.windows[0].mean_peak_power_fraction > 0
        assert obs[2] > 0
        np.testing.assert_array_equal(obs, env.public_control_input())
        assert audit.result()["all_physical_invariants_passed"], audit.result()
        assert audit.result()["states_checked"] == 376
    finally:
        env.close()


def test_matched_control_changes_only_declared_solar_truth_and_issued_forecasts():
    stress = StartupStressEnv("solar-loss")
    control = StartupStressEnv("clear-sky")
    try:
        stress.reset(seed=DIAGNOSTIC_SEED)
        control.reset(seed=DIAGNOSTIC_SEED)
        assert stress._episode.scenario.initial == control._episode.scenario.initial
        assert stress._episode.commitments == control._episode.commitments
        assert stress.market_episode == control.market_episode
        assert stress.disturbance_episode == control.disturbance_episode
        assert stress._episode.weather.clear_sky_envelope_sha256 == control._episode.weather.clear_sky_envelope_sha256
        assert stress._episode.weather.forecast_stream_key == control._episode.weather.forecast_stream_key
        assert control._episode.weather.tape.boundary_factors == (1.0,) * 10081
        assert stress.frozen_source.expected_solar_kw[360:] == control.frozen_source.expected_solar_kw[360:]
        # Once the latest observed factor agrees, every issued forecast agrees.
        for step in range(375, 10080, 15):
            assert stress._episode.weather.current_issue(step) == control._episode.weather.current_issue(step)
        assert any(w.mean_peak_power_fraction > 0 for w in control.current_forecast.windows)
    finally:
        stress.close()
        control.close()


def test_unchanged_forecaster_does_not_read_future_repair_from_truth_tape():
    env = StartupStressEnv("solar-loss")
    try:
        env.reset(seed=DIAGNOSTIC_SEED)
        original = env._episode.weather
        # Change all future restoration factors, while leaving the first360
        # intervals and every factor available through the deadline untouched.
        altered_tape = SolarFactorTape(dt_s=60, boundary_factors=tuple(
            0.0 if step <= 360 else .3 for step in range(10081)))
        solar = SolarArray(env.config.solar)
        envelope = tuple(solar.clear_sky_kw(6*3600 + step*60)/env.config.solar.peak_kw
                         for step in range(10080 - 15 + 1440))
        altered = build_weather_episode_from_tape(original.config, tape=altered_tape,
            decision_stride=15, truth_stream_key=original.truth_stream_key,
            forecast_seed=np.random.SeedSequence(**json.loads(original.forecast_stream_key.initial_state_json)),
            clear_sky_peak_fractions=envelope, clear_sky_envelope_id=original.clear_sky_envelope_id,
            truth_origin_kind="external_tape")
        assert altered.issues[:25] == original.issues[:25]
        assert altered.current_issue(375) != original.current_issue(375)
        assert altered.forecast_stream_key == original.forecast_stream_key
    finally:
        env.close()


def test_fixed_methanator_commands_prove_the_first_delivery_is_recoverable():
    env = StartupStressEnv("solar-loss", record_minutes=True)
    try:
        env.reset(seed=DIAGNOSTIC_SEED)
        audit = MinuteAudit(env)
        audit.add(env.simulator.state)
        energy = 0.0
        for decision in range(24):
            power = 90.0 if decision < 4 else 150.0 if decision < 6 else 0.0
            command = WholePlantAction(0, 0, 0, 0, 0, 0, power)
            env.step(action_to_vector(command))
            for state in env.last_minute_truth_states[1:]:
                energy += state.battery_to_methanator_kw * env.config.dt_h
                audit.add(state)
        first = env.simulator.delivery_audit()["settlements"][0]
        assert first["delivered_kg"] == 180 and first["unmet_kg"] == 0
        assert env.simulator.state.ch4_produced_total_kg == pytest.approx(61.111111111111, abs=1e-8)
        assert energy == pytest.approx(165.0, abs=1e-8)
        assert audit.result()["all_physical_invariants_passed"], audit.result()
    finally:
        env.close()


def test_old_study_seeds_and_unknown_conditions_are_rejected():
    with pytest.raises(ValueError, match="condition"):
        StartupStressEnv("unknown")
    env = StartupStressEnv()
    try:
        for seed in (30000000, 30200000, True, 42000000.0):
            with pytest.raises(ValueError):
                env.reset(seed=seed)
    finally:
        env.close()

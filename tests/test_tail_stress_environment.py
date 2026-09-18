"""Engineering-only validation of paired tail conditions, without PPO outcomes."""
import numpy as np
import pytest

from docs.research.tools.tail_stress_v1.environment import (
    CONDITIONS, SEEDS, TailStressEnv, condition_spec,
)
from docs.research.tools.wholeplant_ppo_v1.environment import WholePlantEnv
from docs.research.tools.wholeplant_ppo_v1.evaluate import MinuteAudit


def test_original_sampled_openings_and_weather_are_preserved_in_control():
    reference = WholePlantEnv()
    env = TailStressEnv("control")
    try:
        # Engineering source comparison only; no controller or old-study reset.
        original = reference._sample_episode(SEEDS[0])
        obs, _ = env.reset(seed=SEEDS[0])
        assert env._episode.scenario.initial == original.scenario.initial
        assert env._episode.weather == original.weather
        assert env.market_episode == reference.market_episode
        assert env._episode.commitments == original.commitments
        assert env._episode.scenario.faults == ()
        assert env.disturbance_episode.faults == ()
        assert env.episode_physics_steps == 10080 and env.episode_decisions == 672
        assert len(env._episode.commitments) == 28
        assert all(c.requested_kg == 180 and c.due_step == i*360
                   for i, c in enumerate(env._episode.commitments, 1))
        assert env.action_space.shape == (7,) and obs.shape == (149,)
        np.testing.assert_array_equal(obs, env.public_control_input())
        env.public_control_input()[:] = -100
        np.testing.assert_array_equal(obs, env.public_control_input())
    finally:
        reference.close()
        env.close()


@pytest.mark.parametrize("condition", CONDITIONS[1:])
def test_conditions_are_paired_with_no_advance_forecast_or_initial_state_change(condition):
    control, stress = TailStressEnv("control"), TailStressEnv(condition)
    try:
        obs0, _ = control.reset(seed=SEEDS[0])
        obs1, _ = stress.reset(seed=SEEDS[0])
        np.testing.assert_array_equal(obs0, obs1)
        assert control.simulator.state == stress.simulator.state
        assert control.base_source_manifest() == stress.base_source_manifest()
        assert control._episode.scenario.initial == stress._episode.scenario.initial
        assert control.market_episode == stress.market_episode
        assert control._episode.commitments == stress._episode.commitments
        original, changed = control._episode.weather, stress._episode.weather
        assert original.config == changed.config
        assert original.forecast_stream_key == changed.forecast_stream_key
        assert original.clear_sky_envelope_sha256 == changed.clear_sky_envelope_sha256
        spec = condition_spec(condition)
        start, end = int(spec["onset_h"]*60), int(spec["end_h"]*60)
        if spec["family"] == "pv":
            assert changed.tape.boundary_factors[:start+1] == original.tape.boundary_factors[:start+1]
            assert changed.tape.boundary_factors[start+1:end+1] == (0.0,)*(end-start)
            assert changed.tape.boundary_factors[end+1:] == original.tape.boundary_factors[end+1:]
            assert changed.issues[:start//15+1] == original.issues[:start//15+1]
            assert changed.issues[end//15+1:] == original.issues[end//15+1:]
            assert stress.frozen_source.expected_solar_kw[:start] == control.frozen_source.expected_solar_kw[:start]
            assert stress.frozen_source.expected_solar_kw[start:end] == (0.0,)*(end-start)
            assert stress.frozen_source.expected_solar_kw[end:] == control.frozen_source.expected_solar_kw[end:]
            assert stress.disturbance_episode.faults == ()
        else:
            assert changed == original
            faults = stress.disturbance_episode.faults
            assert len(faults) == 1 and faults[0].type == "methanator_outage"
            assert faults[0].start_s == spec["onset_h"]*3600
            assert faults[0].duration_s == spec["duration_h"]*3600
            schedule = stress.frozen_source.original_manifest["disturbances"]["schedule"]
            assert schedule[0]["target"] == "methanator"
    finally:
        control.close()
        stress.close()


def test_pv_severity_does_not_change_forecasts_before_the_shortest_repair():
    short, long = TailStressEnv("pv6"), TailStressEnv("pv36")
    try:
        short.reset(seed=SEEDS[1])
        long.reset(seed=SEEDS[1])
        assert short._episode.weather.issues[:(54*60)//15+1] == long._episode.weather.issues[:(54*60)//15+1]
        assert short._episode.weather.current_issue(54*60+15) != long._episode.weather.current_issue(54*60+15)
    finally:
        short.close()
        long.close()


@pytest.mark.parametrize("condition", ("meth24", "meth48"))
def test_methanator_physical_fault_intervals_and_repair_are_exact(condition):
    env = TailStressEnv(condition, record_minutes=True)
    try:
        env.reset(seed=SEEDS[0])
        spec = condition_spec(condition)
        start, end = int(spec["onset_h"]*60), int(spec["end_h"]*60)
        expected_state_steps = set(range(start+1, end+1))
        observed_state_steps = set()
        # All-off commands examine the forcing seam without measuring a policy.
        for _ in range(end//15+1):
            env.step(-np.ones(7))
            for state in env.last_minute_truth_states[1:]:
                if state.health["methanator"].offline:
                    observed_state_steps.add(state.step)
                    assert state.methanator_power_kw == state.ch4_production_kg_h == 0
                assert all(not health.offline for name, health in state.health.items() if name != "methanator")
                assert all(health.degradation == 0 for health in state.health.values())
        assert observed_state_steps == expected_state_steps
        assert not env.telemetry.health["methanator"].offline
        assert env.physics_steps_taken == end+15
        assert env.telemetry.time_of_day_h == pytest.approx(20.25)
    finally:
        env.close()


def test_pv_full_week_keeps_horizon_and_accounting_with_zero_physical_power_in_mask():
    env = TailStressEnv("pv6", record_minutes=True)
    try:
        env.reset(seed=SEEDS[0])
        audit = MinuteAudit(env)
        audit.add(env.simulator.state)
        for decision in range(672):
            obs, _, terminated, truncated, _ = env.step(-np.ones(7))
            assert terminated == (decision == 671) and not truncated
            np.testing.assert_array_equal(obs, env.public_control_input())
            for state in env.last_minute_truth_states[1:]:
                audit.add(state)
                if 2880 < state.step <= 3240:
                    assert state.solar_available_kw == 0
        assert env.physics_steps_taken == 10080
        assert len(env.simulator.delivery_audit()["settlements"]) == 28
        assert audit.result()["all_physical_invariants_passed"], audit.result()
        assert audit.result()["states_checked"] == 10081
    finally:
        env.close()


def test_unknown_conditions_and_prior_study_seeds_are_rejected():
    with pytest.raises(ValueError, match="condition"):
        TailStressEnv("pv100")
    env = TailStressEnv()
    try:
        for seed in (30000000, 30200000, 42000000, True, 43000000.0):
            with pytest.raises(ValueError):
                env.reset(seed=seed)
    finally:
        env.close()

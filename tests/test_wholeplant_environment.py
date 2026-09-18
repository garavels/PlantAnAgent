"""Integration contracts for new physics, observation and economic boundaries."""
import numpy as np
import pytest

from docs.research.tools.wholeplant_ppo_v1.environment import (
    ARMS, WholePlantEnv, IntervalPowerAccounting,
)
from docs.research.tools.wholeplant_ppo_v1.physics import WholePlantAction, action_to_vector


def test_all_arms_expose_current_state_and_identical_exogenous_source():
    manifests = []
    for arm in ARMS:
        env = WholePlantEnv(arm=arm, record_minutes=True)
        try:
            obs, _ = env.reset(seed=30000001)
            manifests.append(env.paired_source_manifest())
            assert obs.shape == (149,) and env.observation_space.contains(obs)
            assert env.action_space.shape == (7,)
            action = action_to_vector(WholePlantAction(0, 0, 0, 0, 0, 300, 90))
            obs, _, _, _, info = env.step(action)
            assert np.array_equal(obs, env.current_policy_observation)
            assert np.array_equal(obs, env.public_control_input())
            assert obs[135] == pytest.approx(env.telemetry.dac_readiness_fraction)
            assert obs[141] == pytest.approx(env.telemetry.methanator_readiness_fraction)
            trace = env.last_minute_truth_states
            assert len(trace) == 16
            battery_kwh = sum(s.battery_to_electrolyser_kw+s.battery_to_dac_kw+s.battery_to_methanator_kw for s in trace[1:])*env.config.dt_h
            assert info['interval_power']['applied_kwh']['battery_support'] == pytest.approx(battery_kwh)
            if not env.shared_bus:
                assert all(s.battery_to_dac_kw == s.battery_to_methanator_kw == 0 for s in trace)
                assert info['direct_control']['requested_action']['battery_dac_support_kw'] > 0
                assert info['direct_control']['intervention_applied_action']['battery_dac_support_kw'] == 0
        finally:
            env.close()
    assert all(m == manifests[0] for m in manifests)


def test_restricted_instant_preserves_old_observations_rewards_and_economic_totals():
    from docs.research.tools.comparison_ppo_v2.environment import ComparisonEnv
    new = WholePlantEnv(arm='restricted-instant')
    old = ComparisonEnv(delivery_rate_kg_h=30.0)
    # Instance-only engineering admission to feed both unchanged samplers one
    # training seed; this is not a canonical experiment or source alteration.
    old.allowed_seeds = range(30000000, 30100000)
    try:
        new_obs, _ = new.reset(seed=30000002)
        old_obs, _ = old.reset(seed=30000002)
        np.testing.assert_array_equal(new_obs[:135], old_obs)
        rng = np.random.default_rng(24)
        for _ in range(96):
            old_action = rng.uniform(-1, 1, 5)
            new_obs, new_reward, *_ = new.step(np.r_[old_action, -1., -1.])
            old_obs, old_reward, *_ = old.step(old_action)
            np.testing.assert_array_equal(new_obs[:135], old_obs)
            assert new_reward == pytest.approx(old_reward, abs=1e-8)
        for key, value in old.economic_metrics.to_dict().items():
            if isinstance(value, (float, int)):
                assert new.economic_metrics.to_dict()[key] == pytest.approx(value, abs=1e-7)
    finally:
        new.close()
        old.close()


def test_evaluation_requires_new_external_gate():
    with pytest.raises(ValueError, match='gate'):
        WholePlantEnv(split='held_out')


def test_old_five_coordinate_action_is_not_silently_accepted():
    env = WholePlantEnv()
    try:
        env.reset(seed=30000003)
        with pytest.raises(ValueError, match='seven'):
            env.step(np.zeros(5))
    finally:
        env.close()

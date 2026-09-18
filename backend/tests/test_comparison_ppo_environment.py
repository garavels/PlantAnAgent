"""Training-only fixtures for the isolated demand/battery comparison experiment.

Run with frozen backend/simulator first on PYTHONPATH, because backend conftest
imports plant_backend before test collection. No development/test seeds opened.
"""
from __future__ import annotations

import json
import math
import numpy as np
import pytest

from docs.research.tools.comparison_ppo_v2 import environment as comparison
from docs.research.tools.free_ppo_v1 import environment as old

SEED = 20_099_900


def test_exact_25_sampled_enabled_physics_parity_with_old_training_fixture():
    reference = old.DirectPlantEnv(record_minutes=True)
    current = comparison.ComparisonEnv(record_minutes=True)
    # Test-only admission of an ORIGINAL training fixture; production split is untouched.
    current.allowed_seeds = {10_099_900}
    try:
        first, _ = reference.reset(seed=10_099_900)
        second, _ = current.reset(seed=10_099_900)
        np.testing.assert_array_equal(first, second)
        rng = np.random.default_rng(442)
        for _ in range(24):
            action = rng.uniform(-1, 1, 5)
            a = reference.step(action)
            b = current.step(action)
            np.testing.assert_array_equal(a[0], b[0])
            assert a[1:4] == b[1:4]
            assert reference.simulator.state.to_dict() == current.simulator.state.to_dict()
            assert a[4]['economic_metrics'] == b[4]['economic_metrics']
            assert a[4]['interval_power'] == b[4]['interval_power']
            assert [x.to_dict() for x in reference.last_minute_truth_states] == [x.to_dict() for x in current.last_minute_truth_states]
    finally:
        reference.close()
        current.close()


def test_pairing_preserves_weather_faults_gas_and_schedule_across_all_variants():
    source = None
    for rate in (25, 30):
        for opening in ('sampled', 'floor'):
            for battery in ('enabled', 'disabled'):
                env = comparison.ComparisonEnv(rate, opening, battery)
                try:
                    env.reset(seed=SEED)
                    paired = env.paired_source_manifest()
                    source = paired if source is None else source
                    assert paired == source
                    assert env.frozen_source.validation['source_initial_gas_preserved']
                    assert all(c.requested_kg == rate * 6 for c in env._episode.commitments)
                    assert len(env._episode.commitments) == 28
                    assert sum(c.requested_kg for c in env._episode.commitments) == rate * 168
                    assert env._episode.weather == env._original_sampled_episode.weather
                    assert env._episode.scenario.faults == env._original_sampled_episode.scenario.faults
                finally:
                    env.close()


def test_floor_is_initialized_before_economic_cycle_and_public_observation_references():
    env = comparison.ComparisonEnv(30, 'floor')
    try:
        observation, _ = env.reset(seed=SEED)
        state = env.simulator.state
        assert state.battery_soc == .05
        assert state.battery_energy_kwh == 400
        assert env.economic_metrics.initial_battery_energy_kwh == 400
        assert env.cycle_reference.battery_energy_kwh == 400
        np.testing.assert_array_equal(observation, env.current_policy_observation)
        np.testing.assert_array_equal(observation, env._observe())
        _, reward, _, _, info = env.step([-1] * 5)
        assert info['economic_metrics']['components_eur']['battery_endowment_liability_change'] == 0
        assert math.isfinite(reward)
        assert env.economic_metrics.initial_battery_energy_kwh == 400
    finally:
        env.close()


def test_disabled_battery_keeps_original_request_and_actual_intervention_separate():
    env = comparison.ComparisonEnv(30, 'sampled', 'disabled', record_minutes=True)
    try:
        env.reset(seed=SEED)
        before = env.simulator.state.battery_energy_kwh
        for _ in range(24):
            _, _, _, _, info = env.step([1, 1, 1, 1, 1])
            assert info['direct_control']['requested_action']['battery_share'] == .25
            assert info['requested']['action']['battery_support_kw'] == 2500
            assert info['battery_intervention']['applied_action']['battery_share'] == 0
            assert info['battery_intervention']['applied_action']['battery_support_kw'] == 0
            assert info['interval_power']['requested_kwh']['battery_support'] > 0
            assert info['interval_power']['applied_kwh']['battery_support'] == 0
            assert all(row['requested_action']['battery_support_kw'] == 2500 for row in env.last_direct_minute_actions)
            assert all(row['intervention_applied_action']['battery_support_kw'] == 0 for row in env.last_direct_minute_actions)
            assert all(s.solar_to_battery_kw == s.battery_to_electrolyser_kw == 0 for s in env.last_minute_truth_states)
            assert env.simulator.state.battery_energy_kwh == before
    finally:
        env.close()


def test_30kg_demand_full_week_and_no_extra_objective_or_guard():
    env = comparison.ComparisonEnv(30, 'floor')
    try:
        env.reset(seed=SEED)
        rewards = []
        for step in range(672):
            _, reward, done, truncated, info = env.step([-1] * 5)
            rewards.append(reward)
            assert done == (step == 671)
            assert not truncated
            assert info['guard']['mode'] == 'absent'
            assert info['guard']['interventions'] == 0
            assert reward == info['reward']['total_eur']
        assert env.metrics.delivered_kg + env.metrics.unmet_kg == pytest.approx(5040)
        assert env.economic_metrics.reward_total_eur == pytest.approx(math.fsum(rewards))
        assert info['daily_return_constraints_in_learning_objective'] is False
        assert info['cycle_constraint']['scalar_reward_contains_constraint_penalty'] is False
    finally:
        env.close()


def test_split_gate_and_seed_domains(tmp_path):
    for split in ('development', 'held_out'):
        with pytest.raises(ValueError, match='external frozen gate'):
            comparison.ComparisonEnv(split=split)
    gate = tmp_path / 'gate.json'
    gate.write_text(json.dumps({'schema': 'comparison-ppo-v2-evaluation-gate',
        'environment_identity_sha256': comparison.environment_identity_sha256(),
        'source_identity_sha256': comparison.source_identity_sha256(),
        'allowed_splits': ['development', 'held_out'], 'endpoints_frozen': False}))
    with pytest.raises(ValueError, match='does not admit'):
        comparison.ComparisonEnv(split='held_out', gate_path=gate)
    env = comparison.ComparisonEnv(split='development', gate_path=gate)
    assert env._episode is None
    env.close()
    env = comparison.ComparisonEnv()
    try:
        for seed in (10_000_000, 20_100_000, 20_200_000, True, 20_000_000.0):
            with pytest.raises(ValueError):
                env.reset(seed=seed)
    finally:
        env.close()

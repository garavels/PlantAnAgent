"""Training-namespace physics/interface checks; no development/test reset."""
from __future__ import annotations

import json
import math
import numpy as np
import pytest

from docs.research.tools.free_ppo_v1 import environment as free
from plant_sim import Action, PlantSimulator
from plant_backend.rl.v2.env import PlantEnvV2

FIXTURE_SEED = 10_099_900


@pytest.mark.parametrize("values", [
    [-1, -1, -1, -1, -1], [1, -1, -1, -1, -1], [-1, 1, -1, -1, -1],
    [-1, -1, 1, -1, -1], [-1, -1, -1, 1, -1], [-1, -1, -1, -1, 1],
    [-.6, -.4, -.8, -.9, .2],
])
def test_admissible_action_roundtrip(values):
    action, evidence = free.decode_action(values)
    recovered, _ = free.decode_action(free.action_to_vector(action))
    assert recovered.to_dict() == pytest.approx(action.to_dict(), abs=1e-12)
    assert sum(action.to_dict()[key] for key in ("battery_share", "electrolyser_share", "dac_share", "methanator_share")) <= 1 + 1e-15
    assert not evidence["box_clipped"]


def test_only_physical_conservation_normalization_and_independent_support():
    request, evidence = free.decode_action([1, 1, 1, 1, 1])
    assert [request.battery_share, request.electrolyser_share, request.dac_share, request.methanator_share] == [.25] * 4
    assert request.battery_support_kw == 2500
    assert evidence["solar_normalization_divisor"] == 4
    request, _ = free.decode_action([-1, -1, -1, -1, -.952])
    assert request.battery_support_kw == pytest.approx(60)


@pytest.mark.parametrize("bad", [[0] * 4, [0] * 6, [0, 0, 0, 0, float("nan")], [True] * 5, ["0"] * 5])
def test_malformed_action_rejected(bad):
    with pytest.raises(ValueError):
        free.decode_action(bad)


def test_external_split_gate_and_seed_separation(tmp_path):
    for split in ("development", "held_out"):
        with pytest.raises(ValueError, match="external frozen gate"):
            free.DirectPlantEnv(split)
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"schema": "free-ppo-v1-evaluation-gate",
        "environment_identity_sha256": free.environment_identity_sha256(),
        "source_identity_sha256": free.source_identity_sha256(),
        "allowed_splits": ["development", "held_out"], "endpoints_frozen": False}))
    with pytest.raises(ValueError, match="does not admit"):
        free.DirectPlantEnv("held_out", gate_path=gate)
    # Admission constructs no physical episode; never open a development seed.
    admitted = free.DirectPlantEnv("development", gate_path=gate)
    assert admitted._episode is None
    admitted.close()
    env = free.DirectPlantEnv()
    for seed in (9_940_000, 10_100_000, 10_200_000, True, 10_000_000.0):
        with pytest.raises(ValueError):
            env.reset(seed=seed)
    env.close()


def test_current_public_observation_and_exact_minute_evidence():
    env = free.DirectPlantEnv(record_minutes=True)
    try:
        observation, info = env.reset(seed=FIXTURE_SEED)
        assert observation.shape == (135,) and observation.dtype == np.float32
        np.testing.assert_array_equal(observation, env._observe())
        assert env.constraints.battery_min_soc == env.config.battery.min_soc == .05
        observation, reward, terminated, truncated, info = env.step([-1] * 5)
        assert np.isfinite(observation).all() and math.isfinite(reward)
        assert not terminated and not truncated
        assert len(env.last_minute_truth_states) == 16
        assert len(env.last_direct_minute_actions) == 15
        assert env.last_minute_truth_states[0].time_s + 900 == env.last_minute_truth_states[-1].time_s
        assert info["guard"]["mode"] == "absent" and info["guard"]["interventions"] == 0
        assert info["legacy_reward"]["diagnostic_only"] is True
        assert reward == info["reward"]["total_eur"]
        np.testing.assert_array_equal(observation, env.current_policy_observation)
        assert env.last_process_setpoint_execution is None
    finally:
        env.close()


def test_direct_loop_preserves_accounting_when_old_guard_does_nothing():
    direct, reference = free.DirectPlantEnv(), free.DirectPlantEnv()
    try:
        reference.simulator = PlantSimulator(reference.config, electrolyser_commitment=reference.episode_config.electrolyser_commitment)
        direct.reset(seed=FIXTURE_SEED)
        reference.reset(seed=FIXTURE_SEED)
        request = Action(battery_share=.125, electrolyser_share=0, dac_share=0, methanator_share=0)
        for _ in range(12):
            obs, reward, done, trunc, info = direct.step(free.action_to_vector(request))
            start = reference._capture_economic_step()
            old_result = PlantEnvV2.step_action(reference, request)
            old_obs, old_reward, old_done, old_trunc, old_info = reference._finish_economic_step(start, old_result)
            np.testing.assert_array_equal(obs, old_obs)
            assert (reward, done, trunc) == (old_reward, old_done, old_trunc)
            assert direct.simulator.state.to_dict() == reference.simulator.state.to_dict()
            assert info["interval_power"] == old_info["interval_power"]
            assert info["economic_metrics"] == old_info["economic_metrics"]
            assert info["metrics"] == old_info["metrics"]
    finally:
        direct.close()
        reference.close()


def test_start_interruption_is_not_projected_away():
    env = free.DirectPlantEnv(record_minutes=True)
    try:
        env.reset(seed=FIXTURE_SEED)
        env.step(free.action_to_vector(Action(battery_share=0, electrolyser_share=0, battery_support_kw=1000)))
        assert env.simulator.electrolyser_commitment.mode.value == "cold_starting"
        _, _, _, _, info = env.step([-1] * 5)
        assert info["direct_control"]["requested_action"]["battery_support_kw"] == 0
        assert env.simulator.electrolyser_commitment.aborted_starts == 1
    finally:
        env.close()


def test_hot_standby_and_using_battery_below_old_reserve_are_reachable():
    env = free.DirectPlantEnv(record_minutes=True)
    try:
        env.reset(seed=FIXTURE_SEED)
        produce = Action(battery_share=0, electrolyser_share=.6, dac_share=0, methanator_share=0,
                         battery_support_kw=1800)
        for _ in range(8):
            env.step(free.action_to_vector(produce))
        assert .05 < env.telemetry.battery_soc < .15
        assert env.simulator.electrolyser_commitment.mode.value == "running"
        standby = Action(battery_share=0, electrolyser_share=0, dac_share=0, methanator_share=0,
                         battery_support_kw=60)
        env.step(free.action_to_vector(standby))
        assert env.simulator.electrolyser_commitment.mode.value == "hot_standby"
        assert env.telemetry.electrolyser_power_kw == pytest.approx(60)
        assert env.simulator.electrolyser_commitment.forced_stops == 0
    finally:
        env.close()


def test_no_optional_guard_or_dispatcher_call(monkeypatch):
    import plant_backend.rl.v2.env as old_env
    def forbidden(*args, **kwargs):
        raise AssertionError("Optional guard or target dispatcher was called")
    monkeypatch.setattr(old_env, "validate", forbidden)
    monkeypatch.setattr(old_env, "dispatch_process_setpoint_command", forbidden)
    env = free.DirectPlantEnv(record_minutes=True)
    try:
        env.reset(seed=FIXTURE_SEED)
        for _ in range(20):
            env.step([-1] * 5)
        # Overrequest the small methanator. Unused solar is not reassigned to battery.
        env.step([-1, -1, -1, 1, -1])
        assert all(state.solar_to_battery_kw == 0 for state in env.last_minute_truth_states)
        assert all(state.methanator_power_kw <= 180 + 1e-9 for state in env.last_minute_truth_states)
    finally:
        env.close()


def test_full_week_economic_ledger_and_diagnostic_only_daily_returns():
    env = free.DirectPlantEnv()
    try:
        env.reset(seed=FIXTURE_SEED)
        rewards = []
        for index in range(672):
            _, reward, done, truncated, info = env.step([-1] * 5)
            rewards.append(reward)
            assert done == (index == 671) and not truncated
        assert env.physics_steps_taken == 10080 and env.decisions_taken == 672
        assert env.metrics.delivered_kg + env.metrics.unmet_kg == pytest.approx(4200)
        assert env.economic_metrics.reward_total_eur == pytest.approx(math.fsum(rewards))
        assert len(env._cycle_settlements) == 7
        assert info["daily_return_constraints_in_learning_objective"] is False
        assert info["cycle_constraint"]["scalar_reward_contains_constraint_penalty"] is False
        assert env.last_minute_truth_states is None
    finally:
        env.close()


@pytest.mark.parametrize("condition", ["normal", "empty_feedstocks", "full_tanks", "offline_dac", "netting"])
def test_local_dispatch_conserves_energy_material_and_explicit_charge(condition):
    from docs.research.tools.free_ppo_v1.direct_simulator import dispatch
    from plant_sim.faults import ComponentEffect
    from plant_sim.allocation import dispatch as frozen_dispatch
    import plant_sim.simulator as frozen_simulator
    env = free.DirectPlantEnv()
    try:
        simulator = env.simulator
        if condition == "empty_feedstocks":
            simulator.h2_tank.level_kg = simulator.co2_tank.level_kg = 0
        elif condition == "full_tanks":
            for tank in (simulator.h2_tank, simulator.co2_tank, simulator.ch4_tank):
                tank.level_kg = tank.config.capacity_kg
        elif condition == "offline_dac":
            simulator.dac.apply_effect(ComponentEffect(offline=True))
        action = (Action(.2, .3, .3, .2, 2500) if condition == "netting"
                  else Action(0, .6, .25, .15, 2500))
        for _ in range(16):
            before = [tank.level_kg for tank in (simulator.h2_tank, simulator.co2_tank, simulator.ch4_tank)]
            flows = dispatch(action, 5000, battery=simulator.battery, electrolyser=simulator.electrolyser,
                h2_tank=simulator.h2_tank, dac=simulator.dac, co2_tank=simulator.co2_tank,
                methanator=simulator.methanator, ch4_tank=simulator.ch4_tank, dt_h=env.config.dt_h)
            assert flows.solar_available_kw == pytest.approx(flows.solar_to_battery_kw + flows.solar_to_electrolyser_kw
                + flows.solar_to_dac_kw + flows.solar_to_methanator_kw + flows.solar_curtailed_kw, abs=1e-8)
            assert flows.electrolyser_power_kw == pytest.approx(flows.solar_to_electrolyser_kw + flows.battery_to_electrolyser_kw)
            assert flows.solar_to_battery_kw <= action.battery_share * 5000 + 1e-9
            if action.battery_share == 0 or condition == "netting":
                assert flows.solar_to_battery_kw == 0
            assert simulator.battery.soc >= env.config.battery.min_soc - 1e-12
            assert simulator.h2_tank.level_kg - before[0] == pytest.approx(flows.h2_produced_kg - flows.ch4_produced_kg * env.config.methanator.h2_kg_per_kg_ch4, abs=1e-10)
            assert simulator.co2_tank.level_kg - before[1] == pytest.approx(flows.co2_captured_kg - flows.ch4_produced_kg * env.config.methanator.co2_kg_per_kg_ch4, abs=1e-10)
            assert simulator.ch4_tank.level_kg - before[2] == pytest.approx(flows.ch4_produced_kg, abs=1e-10)
            for tank in (simulator.h2_tank, simulator.co2_tank, simulator.ch4_tank):
                assert 0 <= tank.level_kg <= tank.config.capacity_kg
        assert frozen_simulator.dispatch is frozen_dispatch
    finally:
        env.close()

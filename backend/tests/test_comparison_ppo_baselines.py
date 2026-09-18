"""Meaningful controller-contract checks, separate from frozen experiment tests."""
import math

import numpy as np
import pytest

from docs.research.tools.comparison_ppo_v2.baselines import (
    BaselineConfig, BaselinePolicy, INDEX, READ_FIELDS,
)


def observation(**updates):
    values = {
        "time_of_day_sin": 0.0, "time_of_day_cos": -1.0,
        "solar_available": 0.7, "battery_soc": 0.35,
        "h2_fill": 0.05, "co2_fill": 0.1, "ch4_fill": 0.05,
        "electrolyser_health": 1.0, "dac_health": 1.0,
        "methanator_health": 1.0, "electrolyser_cap": 1.0,
        "dac_cap": 1.0, "methanator_cap": 1.0, "support_cap": 1.0,
        "delivery_next_amount": 150 / 3000,
        "solar_forecast_0_1h_mean": 0.7, "solar_forecast_0_1h_p10": 0.5,
        "solar_forecast_1_6h_mean": 0.6, "electrolyser_mode_off": 1.0,
    }
    values.update(updates)
    obs = np.zeros(len(INDEX), dtype=np.float32)
    for key, value in values.items():
        obs[INDEX[key]] = value
    return obs


def routes(action):
    values = (action.astype(np.float64) + 1) / 2
    return values[:4], values[4] * 2500


@pytest.mark.parametrize("kind", ["daylight", "storage"])
def test_action_uses_exact_direct_interface_and_batch_contract(kind):
    policy = BaselinePolicy(kind)
    obs = observation()
    action, state = policy.predict(obs)
    shares, support = routes(action)
    assert state is None and action.shape == (5,) and action.dtype == np.float32
    assert np.isfinite(action).all() and np.all(np.abs(action) <= 1)
    assert shares.sum() <= 1 + 1e-7 and 0 <= support <= 2500
    batch, _ = policy.predict(np.stack([obs, obs]))
    np.testing.assert_array_equal(batch, np.stack([action, action]))


def test_daylight_has_no_battery_use_and_exact_zero_night_command():
    policy = BaselinePolicy("daylight")
    action, _ = policy.predict(observation())
    assert action[0] == -1 and action[4] == -1
    action, _ = policy.predict(observation(time_of_day_cos=1.0, solar_available=0,
        solar_forecast_0_1h_mean=0, solar_forecast_0_1h_p10=0,
        solar_forecast_1_6h_mean=0, electrolyser_mode_off=0,
        electrolyser_mode_running=1))
    np.testing.assert_array_equal(action, -np.ones(5))


def test_starting_storage_bridges_forecast_solar_dip_not_just_current_kw():
    obs = observation(solar_available=0.6, solar_forecast_0_1h_mean=0.5,
        solar_forecast_0_1h_p10=0.05, electrolyser_mode_off=0,
        electrolyser_mode_cold_starting=1, electrolyser_startup_remaining=0.75)
    action, _ = BaselinePolicy("storage").predict(obs)
    shares, support = routes(action)
    # When current PV is 3MW but its public downside forecast is 250kW, a
    # constant-current conversion would ask for no support and abort a start.
    assert support > 0
    assert shares[1] * 250 + support >= 600 - 1e-4
    # Last startup action may enter production before the 15-minute hold ends.
    obs[INDEX["electrolyser_startup_remaining"]] = 0.25
    shares, support = routes(BaselinePolicy("storage").predict(obs)[0])
    assert shares[1] * 250 + support >= 1200 - 1e-4


def test_storage_standby_night_energy_is_budgeted_and_floor_is_respected():
    args = dict(time_of_day_cos=1, solar_available=0,
        solar_forecast_0_1h_mean=0, solar_forecast_0_1h_p10=0,
        solar_forecast_1_6h_mean=0, electrolyser_mode_off=0,
        electrolyser_mode_running=1)
    action, _ = BaselinePolicy("storage").predict(observation(**args))
    shares, support = routes(action)
    # The plant has a strict 60kW threshold. Encoding exactly60 in float32
    # decodes to59.999973 and must not silently interrupt nightly standby.
    assert shares.sum() == 0 and 60 <= support < 60.01
    action, _ = BaselinePolicy("storage").predict(observation(**args, battery_soc=0.05))
    _, support = routes(action)
    assert support == pytest.approx(0, abs=1e-4)


def test_feedstock_shortage_and_public_demand_change_actions():
    policy = BaselinePolicy("daylight")
    empty = observation(h2_fill=0, co2_fill=0)
    shares, _ = routes(policy.predict(empty)[0])
    assert shares[1] > 0 and shares[2] > 0 and shares[3] == 0
    # Near its inventory target, the same plant needs more production at30kg/h.
    at25 = observation(h2_fill=0.225, co2_fill=0.4, ch4_fill=0.25)
    at30 = at25.copy()
    at30[INDEX["delivery_next_amount"]] = 180 / 3000
    a25 = policy.predict(at25)[0]
    a30 = policy.predict(at30)[0]
    assert not np.array_equal(a25, a30)


def test_public_fault_blocks_its_process_and_unread_fields_cannot_change_action():
    policy = BaselinePolicy("storage")
    obs = observation(dac_health=0)
    action, _ = policy.predict(obs)
    assert action[2] == -1
    alternate = obs.copy()
    for name, index in INDEX.items():
        if name not in READ_FIELDS:
            alternate[index] = 0.123
    np.testing.assert_array_equal(action, policy.predict(alternate)[0])


def test_invalid_observation_is_not_silently_replaced():
    with pytest.raises(ValueError):
        BaselinePolicy().predict(np.zeros(134))
    bad = observation()
    bad[0] = math.nan
    with pytest.raises(ValueError):
        BaselinePolicy().predict(bad)
    with pytest.raises(ValueError):
        BaselineConfig(battery_target_soc=0.01)

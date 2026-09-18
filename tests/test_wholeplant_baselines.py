"""Whole-plant comparator contract and physically meaningful control checks."""
from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from docs.research.tools.wholeplant_ppo_v1.baselines import (
    BaselineConfig, BaselinePolicy, FORECAST_INDICES, INDEX, OBSERVATION_NAMES,
)


def observation(**changes):
    values={"time_of_day_sin":0,"time_of_day_cos":-1,
        "solar_available":.7,"solar_forecast_0_1h_mean":.7,
        "solar_forecast_0_1h_p10":.4,"solar_forecast_1_6h_mean":.6,
        "battery_soc":.5,"h2_fill":.05,"co2_fill":.1,"ch4_fill":.05,
        "electrolyser_health":1,"dac_health":1,"methanator_health":1,
        "electrolyser_cap":1,"dac_cap":1,"methanator_cap":1,"support_cap":1,
        "delivery_next_amount":180/3000,"delivery_time_to_next":.8,
        "electrolyser_mode_off":1,"dac_off":1,"methanator_off":1}
    values.update(changes);obs=np.zeros(149,dtype=np.float32)
    for name,value in values.items():obs[INDEX[name]]=value
    return obs


def decoded(action):
    values=(np.asarray(action,dtype=np.float64)+1)/2
    return values[:4]/max(1,values[:4].sum()),values[4:]/max(1,values[4:].sum())*2500


def night(**changes):
    values=dict(time_of_day_cos=1,solar_available=0,solar_forecast_0_1h_mean=0,
        solar_forecast_0_1h_p10=0,solar_forecast_1_6h_mean=0)
    values.update(changes)
    return observation(**values)


@pytest.mark.parametrize('kind',['daylight','storage','storage-persistence'])
@pytest.mark.parametrize('shared,dynamic',[(False,False),(True,False),(False,True),(True,True)])
def test_public_observation_and_joint_action_budgets(kind,shared,dynamic):
    p=BaselinePolicy(kind,BaselineConfig(shared_bus=shared,downstream_dynamics=dynamic))
    obs=observation();before=obs.copy();action,state=p.predict(obs)
    assert state is None and action.shape==(7,) and action.dtype==np.float32
    shares,support=decoded(action)
    assert np.isfinite(action).all() and np.max(np.abs(action))<=1
    assert shares.sum()<=1+1e-12 and support.sum()<=2500+1e-9
    if not shared:assert np.array_equal(action[5:],[-1,-1])
    if kind=='daylight':assert np.array_equal(action[[0,4,5,6]],[-1]*4)
    np.testing.assert_array_equal(obs,before)
    batch,_=p.predict(np.stack([obs,obs]));np.testing.assert_array_equal(batch,np.stack([action,action]))


def test_solar_preheat_offers_paid_heater_power_before_discretionary_load():
    obs=observation(solar_available=.7,solar_forecast_0_1h_mean=.7,
        solar_forecast_0_1h_p10=.6)
    cold=BaselinePolicy('daylight').predict(obs)[0]
    shares,_=decoded(cold)
    # Both cold units receive explicit heater offers; these are not productive
    # rates or a hidden instantaneous transition to a ready component.
    assert shares[2]*3500==pytest.approx(300,abs=.001)
    assert shares[3]*3500==pytest.approx(90,abs=.001)
    ready=obs.copy();ready[INDEX['dac_readiness_fraction']]=1;ready[INDEX['methanator_readiness_fraction']]=1
    ready_action=BaselinePolicy('daylight').predict(ready)[0]
    assert not np.array_equal(cold,ready_action)


def test_allocation_keeps_compulsory_electrolyser_before_new_heaters():
    obs=observation(solar_available=.3,solar_forecast_0_1h_mean=.3,
        electrolyser_mode_off=0,electrolyser_mode_running=1,
        electrolyser_min_up_remaining=.5)
    shares,_=decoded(BaselinePolicy('daylight').predict(obs)[0])
    assert shares[1]*1500 >= 1200-.001


def test_storage_can_supply_urgent_ready_methanation_overnight():
    obs=night(ch4_fill=0,h2_fill=.3,co2_fill=.3,delivery_time_to_next=.1,
        dac_readiness_fraction=1,methanator_readiness_fraction=1)
    action,_=BaselinePolicy('storage').predict(obs)
    shares,support=decoded(action)
    assert np.all(shares==0) and support[2]>=180
    restricted=BaselinePolicy('storage',BaselineConfig(shared_bus=False)).predict(obs)[0]
    assert restricted[6]==-1
    daylight=BaselinePolicy('daylight').predict(obs)[0]
    assert np.array_equal(daylight,[-1]*7)


def test_unready_methanator_gets_heat_before_urgent_productive_power():
    obs=night(ch4_fill=0,h2_fill=.3,co2_fill=.3,delivery_time_to_next=.1)
    _,support=decoded(BaselinePolicy('storage').predict(obs)[0])
    assert 90<=support[2]<90.01


def test_floor_has_no_requested_discharge_and_fault_has_no_unit_offer():
    obs=observation(battery_soc=.05,dac_health=0)
    action,_=BaselinePolicy('storage').predict(obs)
    assert action[2]==-1 and action[5]==-1
    assert np.allclose(decoded(action)[1],0,atol=.0001)


def test_storage_does_not_buy_long_overnight_standby_without_delivery_need():
    obs=night(ch4_fill=.3,h2_fill=.2,co2_fill=.2,
        electrolyser_mode_off=0,electrolyser_mode_running=1,
        dac_readiness_fraction=1,methanator_readiness_fraction=1)
    action,_=BaselinePolicy('storage').predict(obs)
    np.testing.assert_array_equal(decoded(action)[1],np.zeros(3))


def test_persistence_changes_only_declared_inputs_and_matches_storage_on_them():
    obs=observation();p=BaselinePolicy('storage-persistence')
    action,evidence=p.inference(obs)
    transformed=obs.copy();transformed[list(FORECAST_INDICES)]=obs[INDEX['solar_available']]
    assert evidence['actual_observation']==obs.tolist()
    assert evidence['controller_input']==transformed.tolist()
    assert [r['index'] for r in evidence['replacements']]==list(FORECAST_INDICES)
    np.testing.assert_array_equal(action,p.predict(obs)[0])
    np.testing.assert_array_equal(action,BaselinePolicy('storage').predict(transformed)[0])
    assert p.metadata()['consumed_forecast_indices']==[34,35,36]
    assert p.metadata()['current_solar_index']==2


def test_unused_prices_and_far_forecasts_do_not_change_action():
    obs=observation();changed=obs.copy()
    for name,index in INDEX.items():
        if name.startswith('market_price') or name.startswith('solar_forecast_6_12') or name.startswith('solar_forecast_12_24'):
            changed[index]=.123
    p=BaselinePolicy('storage')
    np.testing.assert_array_equal(p.predict(obs)[0],p.predict(changed)[0])


def test_instant_model_skips_thermal_state_even_when_appended_readiness_is_zero():
    p=BaselinePolicy('daylight',BaselineConfig(downstream_dynamics=False))
    a=observation();b=a.copy()
    b[135:147]=1
    np.testing.assert_array_equal(p.predict(a)[0],p.predict(b)[0])


@pytest.mark.parametrize('bad',[np.zeros(135),np.zeros(150),np.full(149,np.nan),np.ones(149,dtype=bool)])
def test_bad_inputs_rejected(bad):
    with pytest.raises(ValueError):BaselinePolicy().predict(bad)


def test_schema_and_config_are_explicit():
    assert len(OBSERVATION_NAMES)==149 and len(set(OBSERVATION_NAMES))==149
    assert INDEX['dac_readiness_fraction']==135 and INDEX['methanator_readiness_fraction']==141
    assert INDEX['battery_to_dac_fraction']==147
    with pytest.raises(ValueError):BaselineConfig(dac_startup_heater_kw=50)
    with pytest.raises(ValueError):BaselinePolicy('oracle')

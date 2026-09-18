"""Meaningful statistical/qualification checks plus training-only execution."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from docs.research.tools.free_ppo_v1 import analyze as a
from docs.research.tools.free_ppo_v1 import evaluate as e


def row(seed=1, *, complete=True, delivered=True, value=10.0, daily=0, forced=7):
    physical={name:0.0 for name in a.PHYSICAL}
    physical.update(delivered_kg=4200.0 if delivered else 4190.0,unmet_kg=0.0 if delivered else 10.0,electrolyser_forced_stops=forced)
    economic={name:0.0 for name in a.ECONOMIC};economic["net_operating_value_eur"]=value
    economic["components_eur"]={name:0.0 for name in a.COMPONENTS};economic["components_eur"]["delivered_service_value"]=value
    return {"seed":seed,"controller":"fixture","split":"held_out","execution_complete":complete,
            "every_delivery_passed":complete and delivered,"physical_invariants_passed":complete,
            "delivery_and_physics_passed":complete and delivered,"decisions":672 if complete else 10,"physical_steps":10080 if complete else 150,
            "delivery_settlements":28 if complete else 0,"maximum_delivery_unmet_kg":0 if delivered else 10,
            "daily_returns_met":daily,"stock_deficits":{resource:{"sum":100.0,"max":25.0} for resource in a.RESOURCES},
            "physical_metrics":physical,"economic_metrics":economic,"minute_audit":{"all_physical_invariants_passed":True},"economic_audit":{"passed":True},
            "controller_latency_s":{"mean":.001}}


def test_daily_failures_and_forced_stops_do_not_disqualify():
    result=row(daily=0,forced=19)
    a.validate_row(result)
    group=a.summarize_group([result])
    assert group["delivery_and_physics_successes"]==1
    assert group["metrics"]["electrolyser_forced_stops"]["all_case"]["mean"]==19
    assert group["metrics"]["daily_returns_met"]["all_case"]["mean"]==0


def test_partial_money_never_becomes_weekly_income():
    result=a.summarize_group([row(1,value=10),row(2,complete=False,value=10000)])
    metric=result["metrics"]["net_operating_value_eur"]
    assert metric["all_case"]["mean"] is None
    assert metric["complete_observed"]["mean"]==10
    assert metric["missing_cases"]==1
    assert result["metrics"]["every_delivery_passed"]["all_case"]["mean"]==.5


def test_pairing_rejects_missing_scenario_not_silent_intersection():
    with pytest.raises(ValueError,match="identical scenario"):
        a.paired_comparison({1:row(1)},{2:row(2)},label="bad",split="held_out")


def test_optimizer_aggregate_resamples_scenarios_not_pooled_replicates():
    controllers={"base":{s:row(s,value=0) for s in range(32)}}
    for optimizer in range(3):
        controllers[str(optimizer)]={s:row(s,value=s+100*optimizer) for s in range(32)}
    result=a.aggregate_comparison(controllers,["0","1","2"],"base","held_out")
    estimate=result["metrics"]["net_operating_value_eur"]["all_case"]
    assert estimate==a.bootstrap_mean([s+100 for s in range(32)])
    assert estimate["n_scenarios"]==32


def test_economic_condition_is_delivery_and_physics_not_daily_returns():
    reference={s:row(s,value=0,daily=0) for s in range(3)}
    treatment={s:row(s,value=s,delivered=s!=2,daily=0) for s in range(3)}
    result=a.paired_comparison(reference,treatment,label="x",split="held_out")
    assert result["both_delivery_and_physics_passed"]==2
    assert result["metrics"]["net_operating_value_eur"]["delivery_and_physics_passed_only"]["mean"]==.5


def test_each_delivery_tolerance_not_sum_of_tolerances():
    result=row();result["physical_metrics"]["unmet_kg"]=28*5e-10
    result["maximum_delivery_unmet_kg"]=5e-10
    a.validate_row(result)


def test_incomplete_execution_cannot_get_physical_pass():
    result=row(complete=False);result["physical_invariants_passed"]=True
    with pytest.raises(ValueError,match="improperly"):
        a.validate_row(result)


def test_untrained_and_intermediate_paths_are_unambiguous(tmp_path):
    assert e.checkpoint_path(tmp_path,"untrained")==tmp_path/"seed-101/initial-policy.zip"
    assert e.checkpoint_path(tmp_path,"ppo",211,0)==tmp_path/"seed-211/initial-policy.zip"
    assert e.checkpoint_path(tmp_path,"ppo",211,65536)==tmp_path/"seed-211/checkpoints/step-065536.zip"
    assert e.label_for("ppo",211,0)=="free-ppo-v1-211-step-0"
    with pytest.raises(ValueError):e.label_for("untrained",211)


def test_held_out_intermediate_rejected_before_any_environment_admission(monkeypatch):
    monkeypatch.setattr(e,"environment",lambda:pytest.fail("Must reject before environment access"))
    freeze={"selection":{"optimizer_seeds":[101],"development_checkpoint_timesteps":[0,65536,131072],"final_timesteps":131072}}
    with pytest.raises(ValueError,match="Intermediate"):
        e.admit(freeze,split="held_out",kind="ppo",optimizer_seed=101,checkpoint_steps=65536,gate=None,endpoints=None)


def test_terminal_inventory_liability_is_not_omitted_or_double_counted():
    initial={"ch4_level_kg":100,"h2_level_kg":50,"co2_level_kg":275,"battery_energy_kwh":1000}
    final={"ch4_level_kg":120,"h2_level_kg":20,"co2_level_kg":110,"battery_energy_kwh":800}
    # Opening equivalents200; final equivalents160:40kg liability, not sums of both gases.
    components={name:0.0 for name in a.COMPONENTS}
    components.update(methane_endowment_liability_change=-160,battery_endowment_liability_change=-30)
    env=SimpleNamespace(config=SimpleNamespace(methanator=SimpleNamespace(h2_kg_per_kg_ch4=.5,co2_kg_per_kg_ch4=2.75)),
        economic_assumptions=SimpleNamespace(methane_endowment_value_eur_per_kg_ch4_equivalent=4,battery_endowment_value_eur_per_kwh=.15,delivered_service_value_eur_per_kg=4,unmet_delivery_penalty_eur_per_kg=20),
        economic_metrics=SimpleNamespace(to_dict=lambda:{"components_eur":components,"net_operating_value_eur":-190,"delivered_kg":0,"unmet_kg":0}))
    audit=e.economic_audit(env,initial,final,-190)
    assert audit["passed"] and audit["terminal_methane_equivalent_deficit_kg"]==40
    components["methane_endowment_liability_change"]=0
    assert not e.economic_audit(env,initial,final,-190)["passed"]


def test_training_seed_baseline_translation_and_minute_auditor():
    env=e.environment().DirectPlantEnv(split="train",record_minutes=True)
    try:
        observation,_=env.reset(seed=10000000);initial=env.simulator.state.to_dict()
        audit=e.MinuteAudit(env);audit.add(initial);controller=e.CoverageTranslation();reward=0
        for decision in range(8):
            action,witness=controller.action(env)
            assert np.all(np.isfinite(action)) and action.shape==(5,)
            assert witness["translation_boundary_step"]==env.physics_steps_taken
            observation,r,terminated,truncated,info=env.step(action);reward+=r
            assert info["guard"]["mode"]=="absent"
            if decision==0:
                corrupted=env.last_minute_truth_states[1].to_dict()
                corrupted["h2_level_kg"]+=1
                negative=e.MinuteAudit(env);negative.add(initial);negative.add(corrupted)
                assert any(f["rule"]=="h2_mass_balance" for f in negative.result()["first_failures"])
            for state in env.last_minute_truth_states[1:]:audit.add(state)
        assert audit.result()["all_physical_invariants_passed"]
        assert audit.count==121
        assert e.economic_audit(env,initial,env.simulator.state.to_dict(),reward)["passed"]
    finally:env.close()


def test_training_full_week_gas_balances_during_faults_and_delivery_settlements():
    env=e.environment().DirectPlantEnv(split="train",record_minutes=True)
    try:
        env.reset(seed=10000002)
        assert env.frozen_source.original_manifest["disturbances"]["schedule"]
        audit=e.MinuteAudit(env);audit.add(env.simulator.state)
        for _ in range(672):
            _,_,terminated,truncated,_=env.step(np.zeros(5,dtype=np.float32))
            for state in env.last_minute_truth_states[1:]:audit.add(state)
        result=audit.result()
        assert terminated and not truncated and audit.count==10081
        assert result["all_physical_invariants_passed"],result["first_failures"]
        settlements=env.simulator.delivery_audit()["settlements"]
        assert len(settlements)==28
        assert result["settled_delivery_mass_checked_kg"]>0
        assert result["settled_delivery_mass_checked_kg"]==pytest.approx(sum(s["delivered_kg"] for s in settlements))
        assert max(result["maximum_gas_mass_balance_residual_kg"].values())<1e-6
    finally:env.close()

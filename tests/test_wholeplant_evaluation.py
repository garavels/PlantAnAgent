"""Whole-plant evaluation contracts: physical corruptions and paired inference."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from docs.research.tools.wholeplant_ppo_v1 import analyze as a
from docs.research.tools.wholeplant_ppo_v1 import evaluate as e
from docs.research.tools.free_ppo_v1.test_evaluate import row as old_row


def row(seed=1, arm="shared-dynamic", label="storage", **kwargs):
    result = old_row(seed, **kwargs)
    result.update(condition=arm, arm=arm, controller=label,
        variant={"arm": arm, "delivery_rate_kg_h": 30.0},
        paired_source_sha256=f"scenario-{seed}",
        initial_state={k: 0 for k in a.STOCKS}, final_state={k: 1 for k in a.STOCKS},
        operating_metrics={k: 2 for k in a.OPERATING}, economic_audit={"passed": True})
    return result


def test_failure_stays_in_denominator_without_partial_money():
    result = a.summarize_group([row(1, value=10), row(2, complete=False, value=999999)])
    assert result["n_scenarios"] == 2
    assert result["metrics"]["every_delivery_passed"]["all_case"]["mean"] == .5
    assert result["metrics"]["net_operating_value_eur"]["all_case"]["mean"] is None
    assert result["metrics"]["net_operating_value_eur"]["complete_observed"]["mean"] == 10


def test_pairing_does_not_intersect_or_replace_scenarios():
    with pytest.raises(ValueError, match="identical scenario"):
        a.paired_comparison({1: row(1)}, {2: row(2)}, label="invalid", split="development")
    changed = row(1); changed["paired_source_sha256"] = "different-weather"
    with pytest.raises(ValueError, match="weather"):
        a.paired_comparison({1: row(1)}, {1: changed}, label="invalid", split="development")


def test_factorial_interaction_has_correct_direction_and_scenario_units():
    means = {"restricted-instant": 1, "shared-instant": 3, "restricted-dynamic": 10, "shared-dynamic": 17}
    arms = {arm: {seed: row(seed, arm=arm, value=100*seed+offset) for seed in range(8)} for arm, offset in means.items()}
    result = a.factorial_interaction(arms, controller="storage", split="development")
    estimate = result["metrics"]["net_operating_value_eur"]["all_case"]
    assert estimate == {"n_scenarios": 8, "mean": 5, "ci95": [5, 5]}


def test_optimizer_bootstrap_averages_policies_within_scenario():
    by = {"storage": {seed: row(seed, value=0) for seed in range(8)}}
    for k in range(3):
        by[str(k)] = {seed: row(seed, value=seed+100*k) for seed in range(8)}
    result = a.aggregate_comparison(by, ["0", "1", "2"], "storage", split="development", condition="shared-dynamic")
    assert result["metrics"]["net_operating_value_eur"]["all_case"] == a.v1.bootstrap_mean([seed+100 for seed in range(8)])


def test_forecast_criterion_uses_forecast_minus_persistence_and_both_bars():
    forecast = {s: row(s, value=1000, label="storage") for s in range(32)}
    persistence = {s: row(s, value=1000, label="storage-persistence") for s in range(32)}
    for s in range(3):
        persistence[s]["every_delivery_passed"] = False
    assert not a.forecast_result(forecast, persistence)["criterion"]["practical_evidence_criterion_met"]
    persistence[3]["every_delivery_passed"] = False
    assert a.forecast_result(forecast, persistence)["criterion"]["reliability_bar_met"]
    forecast[4]["every_delivery_passed"] = False
    assert not a.forecast_result(forecast, persistence)["criterion"]["reliability_bar_met"]
    for s in range(32):
        forecast[s]["economic_metrics"]["net_operating_value_eur"] += 300
    result = a.forecast_result(forecast, persistence)
    assert result["metrics"]["net_operating_value_eur"]["all_case"]["mean"] == 300
    assert result["criterion"]["contribution_bar_met"]


def test_unselected_case_rejected_before_environment_access(monkeypatch):
    freeze = {"panels": [{"controller": "storage", "condition": "shared-dynamic", "split": "development", "seeds": [30100000]}]}
    monkeypatch.setattr(e, "environment", lambda: pytest.fail("Unselected case must not open environment"))
    with pytest.raises(ValueError, match="outside"):
        e.admit(freeze, controller="storage", condition="shared-dynamic", split="development", seed=30100001)


def test_persistence_recording_detects_an_extra_replaced_coordinate():
    raw = np.arange(149,dtype=float)/149
    indices = [34,35,36]
    transformed = raw.copy(); transformed[indices] = raw[2]
    action = [-1.0]*7
    evidence = {"observation": raw.tolist(), "observation_sha256": e.canonical_sha(raw.tolist()), "action": action,
        "controller_input_sha256": e.canonical_sha(transformed.tolist()),
        "baseline_inference": {"actual_observation":raw.tolist(),"controller_input":transformed.tolist()}}
    commands = [{"decision_index":0,"action":action,"evidence":evidence}]
    r = {"controller_kind":"baseline","controller":"storage-persistence",
        "baseline_metadata":{"consumed_forecast_indices":indices,"current_solar_index":2},
        "inference":{"witness_sha256":hashlib.sha256(json.dumps(evidence,sort_keys=True,separators=(",",":")).encode()).hexdigest()}}
    a.verify_inference_witnesses(commands,r)
    evidence["baseline_inference"]["controller_input"][6]=0.0
    evidence["controller_input_sha256"]=e.canonical_sha(evidence["baseline_inference"]["controller_input"])
    with pytest.raises(ValueError, match="registered persistence"):
        a.verify_inference_witnesses(commands,r)


def _physical_fixture():
    from docs.research.tools.wholeplant_ppo_v1.environment import WholePlantEnv
    for arm in e.ARMS:
        env = WholePlantEnv(arm=arm, split="train", record_minutes=True)
        try:
            obs, _ = env.reset(seed=30000000)
            initial = env.simulator.state.to_dict()
            audit, flows = e.MinuteAudit(env), e.OperatingAudit(env)
            audit.add(initial); flows.add(initial)
            reward = 0
            first = None
            for index in range(12):
                action = np.asarray([-1.,-1.,-1.,-1.,-1.,-.76,-.928]) if index < 6 else -np.ones(7)
                assert np.array_equal(obs, env.current_policy_observation)
                obs, value, _, _, info = env.step(action); reward += value
                trace = env.last_minute_truth_states
                e.audit_intervention(env, action, info, trace)
                if first is None:
                    first = trace[1].to_dict()
                for state in trace[1:]:
                    audit.add(state); flows.add(state)
            assert audit.count == 181
            assert audit.result()["all_physical_invariants_passed"], audit.result()["first_failures"]
            assert e.economic_audit(env, initial, env.simulator.state.to_dict(), reward, flows)["passed"]
            if arm.startswith("restricted"):
                assert flows.totals["dac_battery_kwh"] == flows.totals["methanator_battery_kwh"] == 0
            if arm == "shared-dynamic":
                assert flows.totals["dac_startup_kwh"] > 0 and flows.totals["methanator_startup_kwh"] > 0
                assert flows.totals["dac_productive_kwh"] > 0 and flows.totals["methanator_productive_kwh"] > 0
                broken = deepcopy(first); broken["battery_to_dac_kw"] += 1
                checker = e.MinuteAudit(env); checker.add(initial); checker.add(broken)
                failures = {f["rule"] for f in checker.result()["first_failures"]}
                assert "battery_energy_balance" in failures and "dac_power_balance" in failures
                broken = deepcopy(first); broken["dac_readiness_kwh"] += 1
                checker = e.MinuteAudit(env); checker.add(initial); checker.add(broken)
                assert "dac_readiness_balance" in {f["rule"] for f in checker.result()["first_failures"]}
        finally:
            env.close()


def test_actual_four_arm_physics_and_corruption_detection():
    subprocess.run([sys.executable,"-c",
        "import runpy; runpy.run_path('tests/test_wholeplant_evaluation.py')['_physical_fixture']()"],
        cwd=e.ROOT,check=True)


def test_expired_panel_retains_every_prescribed_case_without_launching(monkeypatch,tmp_path):
    freeze_path=tmp_path/"freeze.json";freeze_path.write_text("{}")
    frozen={"sha256":"frozen","conditions":{"shared-dynamic":{}},"panels":[{"controller":"storage","condition":"shared-dynamic","split":"development","seeds":[30100000,30100001]}]}
    monkeypatch.setattr(e,"require_freeze",lambda path:frozen)
    monkeypatch.setattr(e.subprocess,"Popen",lambda *args,**kwargs:pytest.fail("Expired case must not launch"))
    panel=e.run_panel(freeze_path=freeze_path,controller="storage",arm="shared-dynamic",split="development",output=tmp_path/"panel",workers=2,deadline_utc="2000-01-01T00:00:00Z")
    assert panel["requested_seeds"]==[30100000,30100001]
    assert len(panel["runs"])==2 and all(r["failure"]["type"]=="DeadlineBeforeStart" for r in panel["runs"])
    assert not any(r["execution_complete"] for r in panel["runs"])

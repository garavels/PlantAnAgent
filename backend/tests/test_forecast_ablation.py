"""Scientific distinction, provenance and eligibility of the supplementary test."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from docs.research.tools.forecast_ablation_v1.policy import PersistencePolicy, INDICES, INDEX, verify_witness
from docs.research.tools.comparison_ppo_v2.baselines import BaselinePolicy
from docs.research.tools.forecast_ablation_v1 import freeze as f
from docs.research.tools.forecast_ablation_v1.analyze import criterion


def observation():
    obs = np.linspace(.01, .9, 135, dtype=np.float32)
    obs[INDEX["solar_available"]] = .72
    for name in ("electrolyser_mode_cooldown", "electrolyser_min_down_remaining"):
        obs[INDEX[name]] = 0
    return obs


def test_exact_three_coordinate_transformation_preserves_input_and_batch():
    obs = observation()
    before = obs.copy()
    out = PersistencePolicy.transform(obs)
    assert np.array_equal(before, obs)
    assert np.flatnonzero(obs != out).tolist() == list(INDICES)
    assert np.all(out[list(INDICES)] == obs[INDEX["solar_available"]])
    batch = PersistencePolicy.transform(np.stack([obs, obs * .8]))
    assert np.array_equal(batch[0], out)
    assert np.all(batch[1, list(INDICES)] == obs[INDEX["solar_available"]] * np.float32(.8))


def test_persistence_is_forecast_invariant_and_matches_unmodified_baseline():
    policy, obs = PersistencePolicy(), observation()
    transformed = policy.transform(obs)
    reference, _ = BaselinePolicy("daylight").predict(transformed)
    action, witness = policy.inference(obs)
    assert np.array_equal(action, reference)
    for name, index in INDEX.items():
        if name.startswith("solar_forecast_"):
            obs[index] = .99
    assert np.array_equal(policy.predict(obs)[0], action)
    assert verify_witness(witness, policy)
    witness["transformed_observation"][INDEX["h2_fill"]] += .01
    with pytest.raises(ValueError, match="witness differs"):
        verify_witness(witness, policy)


def test_inference_evidence_rejects_relabelled_actions():
    policy = PersistencePolicy()
    _, witness = policy.inference(observation())
    witness["action"][0] += .01
    with pytest.raises(ValueError, match="witness differs"):
        verify_witness(witness, policy)


def test_unselected_case_is_rejected_before_main_gate_access(monkeypatch):
    payload = {"panels": [{"controller": f.LABEL, "condition": f.CONDITION, "split": "held_out", "seeds": [20200000]}]}
    monkeypatch.setattr(f, "main_admission", lambda *args: pytest.fail("Gate must not be read"))
    with pytest.raises(ValueError, match="outside"):
        f.admit(payload, controller=f.LABEL, condition=f.CONDITION, split="held_out", seed=20200001)


def test_late_gate_requires_closed_main_campaign_and_earlier_rule_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(f.main_e, "environment", lambda: SimpleNamespace(_admit=lambda *args: None))
    pins = []
    for name in ("endpoints", "training", "analysis"):
        path = tmp_path / (name + ".json")
        path.write_text("{}")
        pins.append(f.file_pin(path))
    gate = {"schema": "comparison-ppo-v2-evaluation-gate", "endpoints_frozen": True,
        "created_at": "2026-09-17T16:00:00+00:00", "endpoints": pins[0], "training_closure": pins[1]}
    closure = {"schema": "comparison-ppo-v2-finish-supervisor-completion", "status": "completed", "outcomes": 1160,
        "endpoints": pins[0], "analysis": pins[2], "stages": [{"stage": n, "exit_code": 0} for n in ("development-learning", "held-out-primary", "held-out-battery", "analysis")]}
    gatepath, done = tmp_path / "gate.json", tmp_path / "done.json"
    gatepath.write_text(json.dumps(gate)); done.write_text(json.dumps(closure))
    payload = {"created_at": "2026-09-17T15:00:00+00:00", "late_gate_contract": {"schema": gate["schema"], "completion_schema": closure["schema"], "main_outcomes": 1160}}
    assert f.main_admission(payload, gatepath, done)["main_analysis"] == pins[2]
    closure["status"] = "running"; done.write_text(json.dumps(closure))
    with pytest.raises(ValueError, match="not closed"):
        f.main_admission(payload, gatepath, done)
    closure["status"] = "completed"; done.write_text(json.dumps(closure))
    payload["created_at"] = "2026-09-17T17:00:00+00:00"
    with pytest.raises(ValueError, match="before"):
        f.main_admission(payload, gatepath, done)


def test_positive_bar_requires_all_pairs_and_no_hidden_failure_exclusion():
    row = {"execution_complete": True, "physical_invariants_passed": True, "economic_audit": {"passed": True}, "every_delivery_passed": True}
    ref, persist = [deepcopy(row) for _ in range(32)], [deepcopy(row) for _ in range(32)]
    paired = {"rescued_cases": [], "regressed_cases": [], "metrics": {"net_operating_value_eur": {"all_case": {"mean": 300, "ci95": [100, 500], "n_scenarios": 32}}}}
    assert criterion(paired, ref, persist, pairing_verified=True)["practical_evidence_bar_met"]
    persist[0]["execution_complete"] = False
    assert not criterion(paired, ref, persist, pairing_verified=True)["practical_evidence_bar_met"]
    persist[0]["execution_complete"] = True
    ref[0]["every_delivery_passed"] = False
    assert not criterion(paired, ref, persist, pairing_verified=True)["contribution_bar_met"]
    paired["rescued_cases"] = list(range(4)); paired["regressed_cases"] = [8]
    assert not criterion(paired, ref, persist, pairing_verified=True)["reliability_bar_met"]

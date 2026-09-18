"""Tests of case retention, pairing, direct action evidence, and physical audit."""
from copy import deepcopy
import subprocess
import sys

import numpy as np
import pytest

from docs.research.tools.comparison_ppo_v2 import analyze as a
from docs.research.tools.comparison_ppo_v2 import evaluate as e
from docs.research.tools.free_ppo_v1.test_evaluate import row as old_row


def row(seed=1, **kwargs):
    result = old_row(seed, **kwargs)
    result.update(condition="d25-sampled", variant={"delivery_rate_kg_h": 25, "opening_battery": "sampled", "battery_mode": "enabled"},
        paired_source_sha256="same-case", initial_state={k: 0 for k in a.STOCKS}, final_state={k: 1 for k in a.STOCKS},
        operating_metrics={k: 2 for k in a.OPERATING})
    return result


def test_failure_stays_in_denominator_without_partial_income():
    summary = a.summarize_group([row(1, value=10), row(2, complete=False, value=99999)])
    assert summary["n_scenarios"] == 2
    assert summary["metrics"]["every_delivery_passed"]["all_case"]["mean"] == .5
    assert summary["metrics"]["net_operating_value_eur"]["all_case"]["mean"] is None
    assert summary["metrics"]["net_operating_value_eur"]["complete_observed"]["mean"] == 10
    assert summary["metrics"]["final:ch4_level_kg"]["all_case"]["mean"] is None


def test_paired_comparison_rejects_case_and_forcing_mismatch():
    with pytest.raises(ValueError, match="identical scenario"):
        a.paired_comparison({1: row(1)}, {2: row(2)}, label="bad", split="development")
    mismatched = row()
    mismatched["paired_source_sha256"] = "changed-weather"
    with pytest.raises(ValueError, match="weather"):
        a.paired_comparison({1: row()}, {1: mismatched}, label="bad", split="development")


def test_cross_optimizer_intervals_have_environmental_sample_size():
    by = {"base": {s: row(s, value=0) for s in range(8)}}
    for optimizer in range(3):
        by[str(optimizer)] = {s: row(s, value=s + 100 * optimizer) for s in range(8)}
    result = a.aggregate_comparison(by, ["0", "1", "2"], "base", split="development", condition="d25-sampled")
    estimate = result["metrics"]["net_operating_value_eur"]["all_case"]
    assert estimate == a.v1.bootstrap_mean([s + 100 for s in range(8)])
    assert estimate["n_scenarios"] == 8


def test_unselected_case_rejected_before_environment_access(monkeypatch):
    freeze = {"panels": [{"controller": "ppo25-101", "condition": "d25-sampled", "split": "development", "seeds": [20100000]}]}
    monkeypatch.setattr(e, "environment", lambda: pytest.fail("Admission must precede environmental access"))
    with pytest.raises(ValueError, match="outside"):
        e.admit(freeze, controller="ppo25-101", condition="d25-sampled", split="development", seed=20100001)


def _actual_energy_and_intervention_fixture():
    module = e.environment()
    hashes = []
    for mode in ("enabled", "disabled"):
        env = module.ComparisonEnv(split="train", delivery_rate_kg_h=30, opening_battery="floor", battery_mode=mode, record_minutes=True)
        try:
            obs, _ = env.reset(seed=20000000)
            initial = env.simulator.state.to_dict()
            physical, flows = e.MinuteAudit(env), e.OperatingAudit(env)
            physical.add(initial); flows.add(initial)
            reward = 0
            # Charges and support are both requested; disabled treatment must remove both.
            action = np.asarray([.5, .5, -.5, -.5, .5], dtype=np.float32)
            for _ in range(8):
                assert np.array_equal(obs, env.current_policy_observation)
                obs, value, _, _, info = env.step(action)
                reward += value
                trace = env.last_minute_truth_states
                e.audit_intervention(env, action, info, trace)
                for state in trace[1:]:
                    physical.add(state); flows.add(state)
                corrupted = deepcopy(info)
                corrupted["battery_intervention"]["applied_action"]["battery_support_kw"] += 1
                with pytest.raises(ValueError, match="intervention evidence"):
                    e.audit_intervention(env, action, corrupted, trace)
            assert physical.result()["all_physical_invariants_passed"]
            assert physical.count == 121
            assert e.economic_audit(env, initial, env.simulator.state.to_dict(), reward)["passed"]
            if mode == "disabled":
                assert flows.totals["battery_charge_kwh"] == flows.totals["battery_discharge_kwh"] == 0
            hashes.append(env.paired_source_manifest()["paired_source_sha256"])
        finally:
            env.close()
    assert hashes[0] == hashes[1]


def test_actual_energy_and_intervention_evidence_on_paired_training_fixture():
    # Backend conftest imports mutable app packages; production runtime deliberately
    # rejects those. Test the frozen simulator in a clean interpreter like a case worker.
    subprocess.run([sys.executable, "-c", "from backend.tests.test_comparison_ppo_evaluation import _actual_energy_and_intervention_fixture; _actual_energy_and_intervention_fixture()"], cwd=e.ROOT, check=True)

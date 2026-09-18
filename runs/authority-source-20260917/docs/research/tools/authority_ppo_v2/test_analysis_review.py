"""Independent synthetic review fixtures; never reset an environment or open test tapes."""
from copy import deepcopy

import numpy as np
import pytest

from docs.research.tools.authority_ppo_v2 import analyze, evaluate


def outcome(scenario, *, passed=True, complete=True, contribution=100.0):
    return {
        "seed": scenario, "split": "held_out", "controller": "synthetic",
        "mission_passed": passed and complete, "execution_complete": complete,
        "every_delivery_passed": complete, "all_daily_returns_passed": passed and complete,
        "hard_safety_passed": complete,
        "economic_metrics": {"net_operating_value_eur": contribution},
    }


def test_32_scenario_bootstrap_averages_optimizer_pairs_before_resampling(monkeypatch):
    monkeypatch.setattr(analyze, "all_metrics", lambda: ("mission_passed", "net_operating_value_eur"))
    panels = {evaluate.COVERAGE_LABEL: {scenario: outcome(scenario, contribution=1000+scenario) for scenario in range(32)}}
    for opt, multiplier in zip(evaluate.OPTIMIZER_SEEDS, (1.0, 2.0, 3.0)):
        panels[evaluate.controller_label("ppo", "broad", opt)] = {
            scenario: outcome(scenario, passed=not(opt == 101 and scenario == 0),
                              contribution=1000+scenario+multiplier*(scenario-16))
            for scenario in range(32)
        }
    result = analyze.arm_comparison(panels, "broad", None, "held_out")
    metric = result["metrics"]["net_operating_value_eur"]["all_case"]
    assert result["n_scenarios"] == metric["n_scenarios"] == 32
    scenario_deltas = 2*(np.arange(32, dtype=float)-16)
    rng = np.random.default_rng(analyze.BOOTSTRAP_SEED)
    expected = scenario_deltas[rng.integers(0, 32, size=(analyze.BOOTSTRAP_REPLICATES, 32))].mean(axis=1)
    assert metric["mean"] == -1.0
    assert metric["ci95"] == np.quantile(expected, [.025, .975]).tolist()
    conditional = result["metrics"]["net_operating_value_eur"]["mutual_all_optimizer_seeds_pass"]
    assert conditional["n_scenarios"] == 31
    assert conditional["mean"] == 0.0
    # One of the three optimizers fails one environmental case. It remains a
    # primary failure with fractional arm-average delta, not a missing case.
    assert result["metrics"]["mission_passed"]["all_case"]["mean"] == pytest.approx(-1/96)


def test_partial_week_is_failure_and_cannot_be_averaged_as_weekly_economics(monkeypatch):
    monkeypatch.setattr(analyze, "all_metrics", lambda: ("mission_passed", "execution_complete", "net_operating_value_eur"))
    reference = {scenario: outcome(scenario, contribution=100) for scenario in range(32)}
    treatment = {scenario: outcome(scenario, contribution=110) for scenario in range(32)}
    treatment[7] = outcome(7, complete=False, contribution=999999)
    treatment[8] = outcome(8, passed=False, contribution=510)
    compared = analyze.paired_comparison(reference, treatment, label="synthetic", split="held_out")
    assert compared["n_scenarios"] == 32
    assert compared["metrics"]["mission_passed"]["all_case"]["mean"] == -2/32
    economics = compared["metrics"]["net_operating_value_eur"]
    assert economics["all_case"]["mean"] is None
    assert economics["complete_observed_pairs"]["n_scenarios"] == 31
    assert economics["complete_observed_pairs"]["mean"] == pytest.approx((30*10+410)/31)
    assert economics["both_missions_passed"]["n_scenarios"] == 30
    assert economics["both_missions_passed"]["mean"] == 10
    assert compared["regressed_missions"] == [7, 8]


def test_missing_optimizer_week_keeps_32_environment_denominator(monkeypatch):
    monkeypatch.setattr(analyze, "all_metrics", lambda: ("mission_passed", "net_operating_value_eur"))
    panels = {evaluate.COVERAGE_LABEL: {s: outcome(s) for s in range(32)}}
    for opt in evaluate.OPTIMIZER_SEEDS:
        panels[evaluate.controller_label("ppo", "broad", opt)] = {s: outcome(s, contribution=105) for s in range(32)}
    panels[evaluate.controller_label("ppo", "broad", 307)][4] = outcome(4, complete=False)
    result = analyze.arm_comparison(panels, "broad", None, "held_out")
    economics = result["metrics"]["net_operating_value_eur"]
    assert economics["all_case"]["n_scenarios"] == 32
    assert economics["all_case"]["mean"] is None
    assert economics["complete_observed_scenarios"]["n_scenarios"] == 31
    assert economics["missing_scenarios"] == [4]
    assert result["metrics"]["mission_passed"]["all_case"]["n_scenarios"] == 32
    assert result["metrics"]["mission_passed"]["all_case"]["mean"] == pytest.approx(-1/96)


def synthetic_record():
    record = {key: None for key in ("seed", "controller", "split", "failure", "benchmark_contract_sha256", "evaluation_freeze_sha256", "minute_truth_sha256")}
    record.update({key: False for key in analyze.BINARY_METRICS})
    record.update(decisions=3, physical_steps=45, wall_time_s=1, physical_metrics={}, economic_metrics={},
                  cycle_settlements=[], commands=[], minute_checks={}, delivery={"settlements": []},
                  manifest={"manifest_sha256": "synthetic", "beta_exogenous_episode_sha256": "synthetic", "disturbances": {"schedule": []}},
                  minute_truth_states=[])
    return record


def test_changed_commands_count_targets_not_actor_noise_or_command_provenance():
    record = synthetic_record()
    targets = {"schema": "targets", "setpoints_kw": {"dac": 100}, "sha256": "same"}
    changed = {"schema": "targets", "setpoints_kw": {"dac": 101}, "sha256": "different"}
    record["commands"] = [
        {"evidence": {"raw_residual": [0, 0, 0, 0], "nominal_command": {"targets": deepcopy(targets), "issuer": "prior"}, "mapped_targets": deepcopy(targets)}},
        {"evidence": {"raw_residual": [1, 1, 1, 1], "nominal_command": {"targets": deepcopy(targets), "issuer": "prior"}, "mapped_targets": deepcopy(targets)}},
        {"evidence": {"raw_residual": [0, .1, 0, 0], "nominal_command": {"targets": deepcopy(targets), "issuer": "prior"}, "mapped_targets": changed}},
    ]
    assert evaluate.summarize_record(record)["inference"]["commands_changed_from_coverage"] == 1


def test_daily_resource_failure_count_honors_existing_tolerance_and_epsilon():
    record = synthetic_record()
    for deficit in (1.0, 1.0+5e-13, 1.0+2e-12):
        record["cycle_settlements"].append({
            "targets": {resource: 10.0 for resource in analyze.RESOURCES},
            "actual": {resource: 10.0-deficit for resource in analyze.RESOURCES},
            "tolerances": {resource: 1.0 for resource in analyze.RESOURCES},
            "feasible": deficit <= 1.0+1e-12,
        })
    summary = evaluate.summarize_record(record)
    for resource in analyze.RESOURCES:
        assert summary["stock_deficits"][resource]["failed_daily_returns"] == 1
        assert summary["stock_deficits"][resource]["sum"] > 3
    assert summary["daily_returns_met"] == 2


def test_bootstrap_rejects_nonfinite_values_instead_of_dropping_them():
    for values in ([1.0, float("nan")], [float("inf")], [[1.0, 2.0]]):
        with pytest.raises(ValueError):
            analyze.bootstrap_mean(values)

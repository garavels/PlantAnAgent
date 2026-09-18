"""Custody/gate/statistical tests; physical smoke checks only use training seeds."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from docs.research.tools.authority_ppo_v2 import analyze, evaluate


def fake_benchmark():
    contract = {"contract_sha256": "contract", "splits": {
        "train": [9940000], "development": list(range(9950000, 9950008)),
        "held_out": list(range(9960000, 9960032)),
    }}
    return SimpleNamespace(load_benchmark_contract=lambda: contract)


def test_parent_freeze_gate_precedes_directory_and_workers(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, "benchmark", fake_benchmark)
    def blocked(*args):
        raise ValueError("Freeze unavailable")
    def forbidden(*args, **kwargs):
        pytest.fail("No workers before held-out freeze")
    monkeypatch.setattr(evaluate, "require_evaluation_freeze", blocked)
    monkeypatch.setattr(evaluate, "ThreadPoolExecutor", forbidden)
    output = tmp_path / "unopened"
    with pytest.raises(ValueError, match="unavailable"):
        evaluate.run_panel(kind="coverage", split="held_out", output=output,
                           training_root=tmp_path, freeze_path=tmp_path / "freeze.json")
    assert not output.exists()


def test_child_freeze_gate_precedes_environment_access(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, "require_evaluation_freeze", lambda *args: {"sha256": "new"})
    def forbidden():
        pytest.fail("Environment must not be accessed after freeze mismatch")
    monkeypatch.setattr(evaluate, "benchmark", forbidden)
    with pytest.raises(ValueError, match="changed before child"):
        evaluate.run_case(kind="coverage", arm=None, optimizer_seed=None, split="held_out", seed=9960000,
            output=tmp_path, training_root=tmp_path, freeze_path=tmp_path / "freeze.json",
            expected_freeze_sha256="old", min_free_gib=0)
    assert not list(tmp_path.iterdir())


def test_heldout_panel_cannot_choose_favorable_subset(monkeypatch):
    monkeypatch.setattr(evaluate, "benchmark", fake_benchmark)
    assert evaluate.select_seeds("mpc", "held_out", None) == list(range(9960000, 9960008))
    assert len(evaluate.select_seeds("ppo", "held_out", None)) == 32
    for seeds in ([9960000], [9960000, 9960000], [9950000]):
        with pytest.raises(ValueError):
            evaluate.select_seeds("coverage", "held_out", seeds)


def test_freeze_rejects_source_change_even_if_endpoints_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, "benchmark", fake_benchmark)
    monkeypatch.setattr(evaluate, "training", lambda: SimpleNamespace(require_frozen_endpoints=lambda root: {"six": "endpoints"}))
    monkeypatch.setattr(evaluate, "comparison_sources", lambda: [{"sha256": "original"}])
    path = tmp_path / "freeze.json"
    evaluate.freeze_evaluation(tmp_path, path)
    monkeypatch.setattr(evaluate, "comparison_sources", lambda: [{"sha256": "changed"}])
    with pytest.raises(ValueError, match="source changed"):
        evaluate.require_evaluation_freeze(tmp_path, path)


def test_watchdog_no_recording_failure_never_gets_mission_credit():
    row = evaluate.failure_row(9960000, "controller", "held_out", "freeze", "CaseWallTimeout", "timeout", 5.0)
    assert not any(row[field] for field in analyze.BINARY_METRICS)
    assert analyze.outcome_value(row, "net_operating_value_eur") is None
    assert analyze.outcome_value(row, "mission_passed") == 0


def test_disk_reserve_stops_before_process_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate.shutil, "disk_usage", lambda path: SimpleNamespace(free=1024))
    def forbidden(*args, **kwargs):
        pytest.fail("No subprocess after disk limit")
    monkeypatch.setattr(evaluate.subprocess, "Popen", forbidden)
    row = evaluate.launch_case(kind="coverage", arm=None, optimizer_seed=None, split="train", seed=9940000,
        output=tmp_path, training_root=tmp_path, freeze_path=tmp_path / "freeze", freeze_sha=None,
        timeout_s=1, min_free_gib=1, max_decisions=1)
    assert not row["execution_complete"]
    assert row["failure"]["type"] == "OSError"
    assert (tmp_path / "9940000-runner-failure.json").exists()


def row(seed, passed, value=100):
    return {"seed": seed, "controller": "test", "split": "held_out",
            **{key: passed for key in analyze.BINARY_METRICS},
            "economic_metrics": {"net_operating_value_eur": value}}


def test_bootstrap_reproducible_and_paired_not_independent():
    a = analyze.bootstrap_mean([0, 1, 2, 3], replicates=1000)
    assert a == analyze.bootstrap_mean([0, 1, 2, 3], replicates=1000)
    assert a["mean"] == 1.5
    assert a["n_scenarios"] == 4
    assert analyze.bootstrap_mean([2.0] * 32)["ci95"] == [2.0, 2.0]


def test_missing_full_week_is_not_treated_as_zero_cost(monkeypatch):
    monkeypatch.setattr(analyze, "all_metrics", lambda: ("mission_passed", "net_operating_value_eur"))
    reference = {1: row(1, True, 100), 2: row(2, True, 100)}
    treatment = {1: row(1, True, 105), 2: row(2, False, 0)}
    comparison = analyze.paired_comparison(reference, treatment, label="test", split="held_out")
    assert comparison["metrics"]["mission_passed"]["all_case"]["mean"] == -0.5
    assert comparison["metrics"]["net_operating_value_eur"]["all_case"]["mean"] is None
    assert comparison["metrics"]["net_operating_value_eur"]["complete_observed_pairs"]["mean"] == 5
    assert comparison["regressed_missions"] == [2]


def test_arm_bootstrap_counts_scenarios_not_optimizer_rows(monkeypatch):
    monkeypatch.setattr(analyze, "all_metrics", lambda: ("mission_passed", "net_operating_value_eur"))
    panels = {evaluate.COVERAGE_LABEL: {1: row(1, True, 100), 2: row(2, True, 100)}}
    for seed, gain in zip(evaluate.OPTIMIZER_SEEDS, (1, 2, 6)):
        panels[evaluate.controller_label("ppo", "narrow", seed)] = {1: row(1, True, 100+gain), 2: row(2, True, 100+gain)}
    result = analyze.arm_comparison(panels, "narrow", None, "held_out")
    assert result["n_scenarios"] == 2
    economic = result["metrics"]["net_operating_value_eur"]["all_case"]
    assert economic["n_scenarios"] == 2
    assert economic["mean"] == 3
    assert economic["ci95"] == [3, 3]


def test_replay_only_excludes_runtime_measurements():
    record = {"commands": [{"evidence": {"evaluation_compute_s": 1, "inference_witness": {"controller_compute_s": 2, "witness_verification_s": 3, "unclipped_deterministic_action": [0.1]}}}],
              **{key: {} for key in ("minute_truth_sha256", "physical_metrics", "economic_metrics", "cycle_settlements", "delivery", "mission_passed", "failure")}}
    other = json.loads(json.dumps(record))
    other["commands"][0]["evidence"]["evaluation_compute_s"] = 90
    assert evaluate.deterministic_replay_view(record) == evaluate.deterministic_replay_view(other)
    other["commands"][0]["evidence"]["inference_witness"]["unclipped_deterministic_action"] = [0.2]
    assert evaluate.deterministic_replay_view(record) != evaluate.deterministic_replay_view(other)


def test_actual_coverage_train_smoke_retains_exact_partial_audit(tmp_path):
    result = evaluate.run_case(kind="coverage", arm=None, optimizer_seed=None, split="train", seed=9940000,
        output=tmp_path, training_root=tmp_path, freeze_path=tmp_path / "absent.json",
        expected_freeze_sha256=None, min_free_gib=0, max_decisions=8)
    assert result["failure"] is None
    assert result["decisions"] == 8
    assert result["physical_steps"] == 120
    assert not result["execution_complete"] and not result["mission_passed"]
    assert result["controller_latency_s"]["count"] == 8
    assert result["minute_checks"]["minute_count"] == 121
    import gzip
    with gzip.open(result["recording"]["path"], "rt") as stream:
        record = json.load(stream)
    analyze.validate_record(record, result, evaluate.benchmark().load_benchmark_contract())


@pytest.mark.parametrize("arm", ("narrow", "broad"))
def test_actual_ppo_observation_and_raw_mean_witness_train_only(arm):
    from docs.research.tools.authority_ppo_v2.train import make_learner
    from docs.research.tools.feasible_mission_replay import run_controller
    import torch
    torch.set_num_threads(1)
    model, allocator, vector = make_learner(101, arm, fixture_steps=2)
    env = evaluate.benchmark().BenchmarkEnv(split="train")
    try:
        env.reset(seed=9940000)
        controller = evaluate.instrumented_ppo_controller(env, model, arm)
        record = run_controller(env, controller, max_decisions=2)
        assert record["failure"] is None
        assert record["decisions"] == 2
        for item in record["commands"]:
            evidence = item["evidence"]
            assert evidence["exact_zero_command"]
            assert evidence["inference_witness"]["adapter_input_verified"]
            assert evidence["inference_witness"]["actual_current_observation_verified"]
            assert evidence["inference_witness"]["unclipped_deterministic_action"] == [0.0] * 4
            assert not evidence["inference_witness"]["upstream_action_clipped"]
    finally:
        env.close()
        vector.close()


@pytest.mark.parametrize("kind,split,steps", (("ppo", "held_out", 65536), ("coverage", "development", 65536), ("ppo", "train", 65536), ("ppo", "development", 32768)))
def test_intermediate_checkpoint_never_opens_heldout_or_nonregistered_endpoint(kind, split, steps, monkeypatch, tmp_path):
    def forbidden():
        pytest.fail("Checkpoint restriction must precede any environment access")
    monkeypatch.setattr(evaluate, "benchmark", forbidden)
    with pytest.raises(ValueError, match="development only"):
        evaluate.run_case(kind=kind, arm="narrow", optimizer_seed=101, split=split, seed=9960000,
            output=tmp_path, training_root=tmp_path, freeze_path=tmp_path / "none", expected_freeze_sha256=None,
            min_free_gib=0, checkpoint_steps=steps)

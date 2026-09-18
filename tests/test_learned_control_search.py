"""CEM objective, search isolation, failure receipts, and real checkpoint tests."""
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from docs.research.tools.learned_control_search_v1 import training as t


PARENT = t.ROOT / "docs/research/results/learned_control_20260918/runs/hybrid-dagger-fidelity/seed-903/training/initial-policy.zip"
DATASET = t.ROOT / "docs/research/results/learned_control_20260918/demonstrations/original-plus-dagger/demonstrations.npz"


def recipe(**changes):
    return t.default_recipe(parent_checkpoint=str(PARENT), training_dataset=str(DATASET),
        deadline_utc=(datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(), **changes)


def fake_rows(cases):
    return [{**case, "status": "completed", "raw_reward_eur": 100. if case["condition"] == "ordinary" else 0.,
             "metrics": {"unmet_kg": 0.}, "decisions": 672, "physical_steps": 10080} for case in cases]


def test_exact_weighted_objective_and_no_incomplete_ranking():
    cases = t.panel_cases([50_300_000, 50_300_001])
    rows = fake_rows(cases)
    assert t.weighted_objective(rows, cases)["weighted_reward_eur"] == 50.
    assert t.weighted_objective(rows, cases)["weeks"] == 12
    for invalid in (rows[:-1], rows + rows[:1], rows[:-1] + rows[:1]):
        with pytest.raises(ValueError, match="complete matched"):
            t.weighted_objective(invalid, cases)
    rows[-1]["status"] = "deadline_incomplete"
    with pytest.raises(ValueError, match="complete matched"):
        t.weighted_objective(rows, cases)


def test_antithetic_candidates_and_current_panel_incumbent_ranking():
    r = recipe()
    mean, std, incumbent = np.ones(14), np.ones(14) * 2, np.arange(14, dtype=float)
    candidates = t.generation_candidates(mean, std, incumbent, np.random.default_rng(12), r)
    assert len(candidates) == 16
    for index in range(0, 14, 2):
        np.testing.assert_allclose(np.asarray(candidates[index]["parameters"]) + candidates[index + 1]["parameters"], 2 * mean)
    np.testing.assert_array_equal(candidates[-1]["parameters"], incumbent)
    tied = {i: {"weighted_reward_eur": 10.} for i in range(16)}
    _, _, kept, _, winner = t.update_distribution(candidates, tied, mean, std, r)
    assert winner == 15
    np.testing.assert_array_equal(kept, incumbent)
    # The refreshed score alone decides; no previous-generation score is accepted.
    tied[15]["weighted_reward_eur"] = -100
    _, _, _, _, winner = t.update_distribution(candidates, tied, mean, std, r)
    assert winner == 0


def test_elite_smoothing_and_search_spread_floor():
    r = recipe(elite_count=1, smoothing=1.)
    points = [{"index": 0, "role": "incumbent", "parameters": [7.] * 14}]
    mean, std, best, elites, _ = t.update_distribution(points, {0: {"weighted_reward_eur": 1.}}, np.zeros(14), np.ones(14), r)
    np.testing.assert_array_equal(mean, best)
    np.testing.assert_array_equal(std, [.15] * 14)
    assert elites == [0]


def test_teacher_on_scales_keep_rare_on_labels_and_no_parameter_bounds():
    actions = np.full((1000, 7), -1.)
    actions[0] = [-.99, 1., -.9, 0., -.5, -.999, -.2]
    scales = t.derive_parameter_scales(actions, recipe())
    assert scales["teacher_on_counts"] == [1] * 7
    np.testing.assert_allclose(scales["on_mean_scales"], [.001, .05, .003, .03, .015, .001, .024])
    assert scales["gate_scales"] == [2.] * 7
    policy = torch.nn.Module()
    policy.action_net = torch.nn.Linear(3, 21)
    model = SimpleNamespace(policy=policy)
    base = policy.action_net.bias.detach().clone()
    unchanged = t.unchanged_tensor_hash(model)
    for point in ([1.] * 14, [-3.] * 14, [0.] * 14, [1.] * 14, [100.] * 14):
        t.apply_parameters(model, base, point, scales["all_scales"])
        np.testing.assert_allclose(policy.action_net.bias[:14].detach().numpy(), base[:14].numpy() + np.asarray(point) * scales["all_scales"], rtol=1e-6, atol=np.finfo(np.float32).eps)
        assert t.unchanged_tensor_hash(model) == unchanged
        assert torch.equal(policy.action_net.bias[14:], base[14:])
    assert policy.action_net.bias[0] > 1  # Search does not constrain network parameters.


def test_training_only_case_recipe_and_distinct_search_seed_identity():
    r = recipe(search_seed=3913)
    assert r["optimizer_seed"] == 3913
    t._validate(r)
    for changes in ({"optimizer_seed": 901}, {"validation_seeds": [50_300_000]},
                    {"validation_seeds": [41]}, {"condition_weights": {c: 1 / 6 for c in t.CONDITIONS}}):
        with pytest.raises(ValueError):
            t._validate({**r, **changes})


def task(index=0):
    return {"phase": "search", "generation": 0, "candidate": index, "condition": "ordinary",
            "seed": 50_300_000, "parameters": [0.] * 14}


def test_broken_submission_and_worker_failure_preserve_all_tasks(tmp_path):
    class BrokenExecutor:
        calls = 0
        def submit(self, function, item):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("broken pool")
            result = Future()
            result.set_exception(RuntimeError("worker vanished after unknown progress"))
            return result
    rows = t.execute_panel(BrokenExecutor(), [task(i) for i in range(3)], tmp_path, recipe(workers=1))
    assert len(rows) == 3
    assert rows[0]["status"] == "worker_failed" and rows[0]["decisions"] is None
    assert [row["status"] for row in rows[1:]] == ["submission_failed_not_started"] * 2
    assert len(list(tmp_path.rglob("*.json"))) == 3
    ledger = t.effort_ledger(rows)
    assert ledger["unknown_partial_rollouts"] == 1
    assert ledger["physical_decisions"] == ledger["gradient_updates"] == ledger["ppo_updates"] == 0


def test_expired_deadline_never_submits_and_retains_unscheduled_cases(tmp_path):
    class NeverSubmit:
        def submit(self, *args):
            raise AssertionError("Expired deadline submitted work")
    r = recipe()
    r["deadline_utc"] = "2020-01-01T00:00:00Z"
    rows = t.execute_panel(NeverSubmit(), [task(i) for i in range(3)], tmp_path, r)
    assert len(rows) == 3 and all(r["status"] == "deadline_not_started" for r in rows)


@pytest.fixture
def real_plan(tmp_path):
    if not PARENT.exists() or not DATASET.exists():
        pytest.skip("Local campaign parent and authenticated training labels are required")
    t.common.configure_torch()
    root = tmp_path / "run"
    return root, t.freeze_plan(root, recipe())


def test_real_plan_pins_training_dependencies_and_expected_effort(real_plan, tmp_path):
    root, plan = real_plan
    assert plan["expected_rollouts"] == {"search": 768, "validation": 48, "total": 816,
        "decisions": 548352, "physical_minutes": 8225280}
    assert t.require_plan(root) == plan
    assert plan["parent_training_seed"] == 903
    extra = tmp_path / "extra.py"
    extra.write_text("VALUE = 1\n")
    second_root = tmp_path / "second"
    t.freeze_plan(second_root, recipe(extra_source_paths=[str(extra)]))
    extra.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="Frozen CEM source"):
        t.require_plan(second_root)


def test_real_checkpoint_reopens_exact_actor_optimizer_normalizer_and_search_seed(real_plan):
    root, plan = real_plan
    parent = t.hybrid.load_checkpoint(PARENT)
    try:
        point = np.linspace(-1., 1., 14)
        ledger = t.effort_ledger(fake_rows(t.panel_cases([50_300_000, 50_300_001])))
        path = root / "test-policy.zip"
        record = t.save_checkpoint(parent, path, plan=plan, parameters=point, kind="engineering", ledger=ledger, generation=0)
        loaded = t.common.load_checkpoint(path)
        try:
            assert loaded.model.seed == 3901
            assert loaded.model.num_timesteps == parent.model.num_timesteps == 0
            assert t.common.optimizer_sha256(loaded.model) == plan["parent_optimizer_sha256"]
            assert t.unchanged_tensor_hash(loaded.model) == plan["unchanged_tensor_sha256"]
            assert path.with_suffix(".vecnormalize.pkl").read_bytes() == PARENT.with_suffix(".vecnormalize.pkl").read_bytes()
            np.testing.assert_array_equal(loaded.predict(np.zeros(149, dtype=np.float32), deterministic=True)[0], parent.predict(np.zeros(149, dtype=np.float32), deterministic=True)[0])
            assert record["gradient_updates_this_run"] == record["ppo_updates_this_run"] == 0
            assert record["parent_training_seed"] == 903
        finally:
            loaded.close()
    finally:
        parent.close()


def test_one_spawned_full_week_uses_exact_original_mission(real_plan):
    _, plan = real_plan
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                             initializer=t._worker_initializer, initargs=(plan,)) as executor:
        result = executor.submit(t._evaluate_case, task()).result(timeout=60)
    assert result["status"] == "completed", result
    assert result["decisions"] == 672 and result["physical_steps"] == 10080
    assert result["raw_reward_eur"] == pytest.approx(result["economic_metrics"]["net_operating_value_eur"], abs=1e-6)


def test_campaign_bookkeeping_and_validation_cannot_select_checkpoint(real_plan, monkeypatch):
    root, _ = real_plan
    class UnusedExecutor:
        def __init__(self, **kwargs):
            pass
        def shutdown(self, **kwargs):
            pass
    def synthetic_panel(executor, tasks, output, recipe):
        rows = []
        for item in tasks:
            sign = 1 if item["phase"] == "search" else -1
            reward = sign * item["parameters"][0]
            row = {**item, "status": "completed", "raw_reward_eur": reward,
                   "metrics": {"unmet_kg": 0.}, "decisions": 672, "physical_steps": 10080,
                   "engineering_test_synthetic": True}
            rows.append(row)
        return rows
    monkeypatch.setattr(t, "ProcessPoolExecutor", UnusedExecutor)
    monkeypatch.setattr(t, "execute_panel", synthetic_panel)
    result = t.train(root)
    assert result["status"] == "completed"
    assert result["total_ledger"]["completed_rollouts"] == 816
    assert result["total_ledger"]["physical_decisions"] == 548352
    assert result["total_ledger"]["gradient_updates"] == 0
    generation = json.loads((root / "training/generation-003-result.json").read_text())
    assert result["final_parameters"] == generation["incumbent"]
    assert result["validation"]["endpoint"]["weighted_reward_eur"] < result["validation"]["parent"]["weighted_reward_eur"]
    assert result["endpoint"]["parameters"] == generation["incumbent"]

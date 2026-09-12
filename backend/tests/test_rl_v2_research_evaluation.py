"""Real beta trajectories, controller failures, and independent physical replay."""

from copy import deepcopy
import gzip
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from plant_backend import operational_mpc
from plant_backend.rl.v2.economic_mpc_schema import EconomicMPCPlanError
from plant_backend.rl.v2.training import research_evaluation as evaluation
from plant_backend.rl.v2.training.pilot import environment_template
from plant_backend.rl.v2.training.custody_json import canonical_bytes, decode


class FixedPolicy:
    def __init__(self, *, fail_after: int | None = None, action: Any = None):
        self.calls = 0
        self.fail_after = fail_after
        self.action = (
            np.array([0.123456789, -0.2, 0.7, -0.6]) if action is None else action
        )

    def predict(self, observation, *, deterministic):
        assert deterministic is True
        assert observation.shape == (135,)
        if self.calls == self.fail_after:
            raise RuntimeError("fixture policy failure")
        self.calls += 1
        return self.action, None


@pytest.fixture
def config(tmp_path: Path):
    return environment_template(tmp_path)


@pytest.mark.parametrize(
    "seed,fixture,prefix",
    [
        (0, False, None),
        (101, False, None),
        (9_500_001, False, None),
        (9_800_000, False, None),
        (9_990_601, False, None),
        (9_700_000, True, None),
        (9_700_000, False, 2),
        (True, False, None),
    ],
)
def test_seed_and_prefix_isolation_before_environment(
    config, monkeypatch, seed, fixture, prefix
):
    monkeypatch.setattr(
        evaluation.ppo, "make_env", lambda _: pytest.fail("forbidden reset path")
    )
    with pytest.raises(ValueError):
        evaluation.evaluate_episode(
            config, actor="H", seed=seed, fixture=fixture, max_decisions=prefix
        )


def test_policy_prefix_replay_is_exact_and_never_calls_controller(config, monkeypatch):
    policy = FixedPolicy(fail_after=2)
    result = evaluation.evaluate_episode(
        config,
        actor="PPO",
        policy=policy,
        policy_id="fixture policy",
        seed=9_990_601,
        fixture=True,
    )
    assert policy.calls == 2
    summary = result["summary"]
    assert summary["decisions"] == 2
    assert summary["physical_steps"] == 30
    assert len(result["minute_truth_states"]) == 31
    assert not summary["execution_complete"]
    assert summary["failure"]["phase"] == "controller_compute"
    assert summary["failure"]["message"] == "fixture policy failure"
    assert result["failed_decision"]["decision_index"] == 2
    monkeypatch.setattr(
        evaluation, "_controller", lambda *args: pytest.fail("replay ran a controller")
    )
    result = decode(gzip.decompress(gzip.compress(canonical_bytes(result), mtime=0)))
    report = evaluation.verify_episode_replay(config, result, fixture=True)
    assert report["verified"]
    assert report["scope"] == "retained_physical_prefix"
    assert not report["controller_reexecuted"]
    assert not report["execution_complete"]
    assert not report["registered_evaluation_eligible"]


@pytest.mark.parametrize(
    "change",
    [
        "minute",
        "economics",
        "settlement",
        "action",
        "command",
        "public",
        "failed_boundary",
    ],
)
def test_replay_rejects_tampering(config, change):
    actor = "H" if change == "command" else "PPO"
    result = evaluation.evaluate_episode(
        config,
        actor=actor,
        policy=FixedPolicy(fail_after=1) if actor == "PPO" else None,
        seed=9_990_602,
        fixture=True,
        max_decisions=2,
    )
    forged = deepcopy(result)
    if change == "minute":
        forged["minute_truth_states"][1]["h2_level_kg"] += 1
    elif change == "economics":
        forged["summary"]["economic_metrics"]["invented"] = 1
    elif change == "settlement":
        forged["summary"]["cycle_settlements"] = [{"feasible": True}]
    elif change == "action":
        forged["history"][0]["raw_action"][0] = 1.0
    elif change == "command":
        forged["history"][0]["command"]["targets"]["setpoints_kw"]["electrolyser"] += 1
    elif change == "public":
        forged["history"][0]["observation"]["normalized"][0] += 0.01
    else:
        forged["failed_decision"]["decision_index"] += 1
    with pytest.raises(ValueError):
        evaluation.verify_episode_replay(config, forged, fixture=True)


def test_invalid_raw_output_is_finite_serializable_and_has_no_fallback(config):
    result = evaluation.evaluate_episode(
        config,
        actor="PPO",
        policy=FixedPolicy(action=np.array([float("nan"), 0, 0, 0])),
        seed=9_990_603,
        fixture=True,
    )
    assert result["summary"]["physical_steps"] == 0
    assert result["summary"]["failure"]["type"] == "ValueError"
    assert result["failed_decision"]["raw_action"][0] == "nan"
    assert result["summary"]["fallback_decisions"] == 0
    assert evaluation.verify_episode_replay(config, result, fixture=True)["verified"]


def test_out_of_range_is_recorded_after_common_guard(config):
    result = evaluation.evaluate_episode(
        config,
        actor="PPO",
        policy=FixedPolicy(action=np.array([2.0, -2.0, 0.0, 0.0])),
        seed=9_990_604,
        fixture=True,
        max_decisions=1,
    )
    assert result["history"][0]["raw_action"] == [2.0, -2.0, 0.0, 0.0]
    assert result["summary"]["out_of_range_decisions"] == 1
    assert not result["summary"]["hard_safety_passed"]
    assert evaluation.verify_episode_replay(config, result, fixture=True)["verified"]


def test_classifier_exception_is_separate_and_never_passes(config, monkeypatch):
    def broken(*args):
        raise RuntimeError("fixture classifier failure")

    monkeypatch.setattr(evaluation, "_hard_safety_classification", broken)
    result = evaluation.evaluate_episode(
        config, actor="H", seed=9_990_605, fixture=True, max_decisions=1
    )
    assert result["summary"]["failure"] is None
    assert not result["summary"]["hard_safety_passed"]
    assert (
        result["summary"]["hard_safety_classifier_failure"]["message"]
        == "fixture classifier failure"
    )
    assert evaluation.verify_episode_replay(config, result, fixture=True)["verified"]


def test_operating_mpc_uses_168h_episode_24h_plan_and_disclosed_fallback(
    config, monkeypatch
):
    controller = evaluation._controller(config, "OperatingMPC")
    assert controller.settings.task_horizon_h == 168
    assert controller.to_dict()["planning_horizon_h"] == 24

    def rejected(*args, **kwargs):
        raise EconomicMPCPlanError("fixture solver rejection")

    monkeypatch.setattr(operational_mpc, "solve_economic_mpc_model", rejected)
    result = evaluation.evaluate_episode(
        config, actor="OperatingMPC", seed=9_990_606, fixture=True, max_decisions=1
    )
    assert result["summary"]["decisions"] == 1
    assert result["summary"]["failure"] is None
    assert result["summary"]["fallback_decisions"] == 1
    assert (
        result["history"][0]["operation"]["solver_failure"]["reason"] == "plan_rejected"
    )
    monkeypatch.setattr(
        evaluation, "_controller", lambda *args: pytest.fail("replay ran a solver")
    )
    assert evaluation.verify_episode_replay(config, result, fixture=True)["verified"]


@pytest.mark.parametrize("actor", ["M24", "E-beta"])
def test_native_research_mpc_failure_retains_opening_evidence_without_fallback(
    config, monkeypatch, actor
):
    controller = evaluation._controller(config, actor)

    def rejected(*args, **kwargs):
        raise RuntimeError("fixture native planning failure")

    monkeypatch.setattr(
        controller,
        "compute_setpoints" if actor == "M24" else "compute_command",
        rejected,
    )
    monkeypatch.setattr(evaluation, "_controller", lambda *args: controller)
    result = evaluation.evaluate_episode(
        config, actor=actor, seed=9_990_607, fixture=True
    )
    assert result["summary"]["physical_steps"] == 0
    assert result["summary"]["failure"]["message"] == "fixture native planning failure"
    assert result["summary"]["fallback_decisions"] == 0
    result = decode(gzip.decompress(gzip.compress(canonical_bytes(result), mtime=0)))
    assert evaluation.verify_episode_replay(config, result, fixture=True)["verified"]


def test_real_full_heuristic_week_settles_all_days_and_replays_every_minute(config):
    result = evaluation.evaluate_episode(
        config, actor="H", seed=9_990_608, fixture=True
    )
    summary = result["summary"]
    assert summary["execution_complete"]
    assert summary["decisions"] == 672
    assert summary["physical_steps"] == 10_080
    assert len(result["minute_truth_states"]) == 10_081
    assert len(summary["cycle_settlements"]) == 7
    assert len(summary["cycle_deficit_vectors"]) == 7
    assert all(len(row) == 4 for row in summary["cycle_deficit_vectors"])
    assert summary["hard_safety_classifier_failure"] is None
    result = decode(gzip.decompress(gzip.compress(canonical_bytes(result), mtime=0)))
    replay = evaluation.verify_episode_replay(config, result, fixture=True)
    assert replay["verified"] and replay["execution_complete"]
    assert replay["scope"] == "complete_episode"


def test_m24_actual_plan_command_and_reconciliation_replay(config, monkeypatch):
    result = evaluation.evaluate_episode(
        config,
        actor="M24",
        seed=9_990_609,
        fixture=True,
        max_decisions=1,
    )
    assert result["summary"]["failure"] is None
    assert result["summary"]["decisions"] == 1
    assert result["history"][0]["command"]["decision_index"] == 0
    monkeypatch.setattr(
        evaluation, "_controller", lambda *args: pytest.fail("replay called M24")
    )
    assert evaluation.verify_episode_replay(config, result, fixture=True)["verified"]


@pytest.mark.parametrize("actor", ["PPO", "H", "M24", "OperatingMPC"])
def test_gzip_json_round_trip_replays_and_still_rejects_tampering(
    config, tmp_path, monkeypatch, actor
):
    if actor == "OperatingMPC":

        def rejected(*args, **kwargs):
            raise EconomicMPCPlanError("fixture serialization fallback")

        monkeypatch.setattr(operational_mpc, "solve_economic_mpc_model", rejected)
    original = evaluation.evaluate_episode(
        config,
        actor=actor,
        policy=FixedPolicy() if actor == "PPO" else None,
        seed=9_990_610,
        fixture=True,
        max_decisions=1,
    )
    assert original["summary"]["decisions"] == 1
    artifact = tmp_path / "episode.json.gz"
    artifact.write_bytes(
        gzip.compress(canonical_bytes(original), compresslevel=6, mtime=0)
    )
    loaded = decode(gzip.decompress(artifact.read_bytes()))
    # Real config tuples become JSON arrays; Python-container equality is wrong.
    assert original["config"] != loaded["config"]
    assert canonical_bytes(original) == canonical_bytes(loaded)
    report = evaluation.verify_episode_replay(config, loaded, fixture=True)
    assert report["verified"] and report["physical_steps"] == 15
    assert report["scope"] == "retained_physical_prefix"
    loaded["summary"]["economic_metrics"]["net_operating_value_eur"] += 1
    with pytest.raises(ValueError, match="outcome metrics"):
        evaluation.verify_episode_replay(config, loaded, fixture=True)


def test_json_equality_keeps_scalar_types_strict():
    assert evaluation._json_equal({"items": (1.0, 2)}, {"items": [1.0, 2]})
    assert not evaluation._json_equal({"passed": False}, {"passed": 0})
    assert not evaluation._json_equal({"value": 1.0}, {"value": 1})

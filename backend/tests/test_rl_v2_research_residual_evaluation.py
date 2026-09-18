"""Real disjoint physical fixtures for residual evidence and saved-command replay."""

from __future__ import annotations

from copy import deepcopy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from plant_backend.rl.v2.training import research_batch
from plant_backend.rl.v2.training import research_residual_evaluation as evaluation
from plant_backend.rl.v2.training.custody_json import canonical_bytes


class ConstantPolicy:
    def __init__(self, action: list[float]) -> None:
        self.action = np.asarray(action, dtype=np.float32)
        self.calls = 0

    def predict(self, observation, *, deterministic):
        assert observation.shape == (139,)
        assert deterministic is True
        self.calls += 1
        return self.action.copy(), None


@pytest.fixture(scope="module")
def day(tmp_path_factory):
    config = research_batch.template(tmp_path_factory.mktemp("residual-evidence"))
    policy = ConstantPolicy([0.25, -0.3, 0.4, 0.5])
    episode = evaluation.evaluate_residual_episode(
        config, policy=policy, seed=9_990_620, fixture=True, max_decisions=96
    )
    assert policy.calls == 96
    assert episode["summary"]["failure"] is None
    return config, episode


def test_serialized_day_replays_commands_and_daily_settlement(day, monkeypatch):
    config, episode = day
    # Full compressed JSON round trip exercises tuple/list and float32 publication.
    encoded = gzip.compress(canonical_bytes(episode), mtime=0)
    reopened = json.loads(gzip.decompress(encoded))

    def no_policy_execution(*args, **kwargs):
        raise AssertionError("physical replay must not execute the learned policy")

    monkeypatch.setattr(ConstantPolicy, "predict", no_policy_execution)
    result = evaluation.verify_residual_episode_replay(config, reopened, fixture=True)
    assert result["verified"] is True
    assert result["saved_absolute_commands_executed"] is True
    assert result["learned_policy_reexecuted"] is False
    assert result["physical_steps"] == 1440
    assert result["execution_complete"] is False
    assert result["scope"] == "retained_physical_prefix"
    assert len(reopened["summary"]["cycle_settlements"]) == 1
    assert len(reopened["initial_observation"]["normalized"]) == 135
    assert len(reopened["initial_policy_observation"]) == 139


@pytest.mark.parametrize(
    "target",
    [
        "input",
        "raw",
        "command",
        "mapping",
        "physical",
        "reward",
        "summary",
        "interface",
        "identity",
        "deterministic",
        "budget",
        "extra_failure",
    ],
)
def test_tampering_with_mapping_or_physical_evidence_is_rejected(day, target):
    config, episode = day
    changed = deepcopy(episode)
    row = changed["history"][0]
    if target == "input":
        row["policy_observation"][-1] += 0.01
    elif target == "raw":
        row["raw_action"][0] += 0.1
    elif target == "command":
        row["command"]["tampered"] = True
    elif target == "mapping":
        row["residual_evidence"]["sha256"] = "0" * 64
    elif target == "physical":
        changed["minute_truth_states"][1]["battery_soc"] += 0.01
    elif target == "reward":
        row["reward_eur"] += 1
    elif target == "summary":
        changed["summary"]["hard_safety_passed"] = not changed["summary"][
            "hard_safety_passed"
        ]
    elif target == "interface":
        changed["interface_contract_digest"] = "0" * 64
    elif target == "identity":
        changed["controller"]["id"] = "another policy"
    elif target == "deterministic":
        changed["controller"]["deterministic"] = False
    elif target == "budget":
        changed["prefix_limit"] += 1
    elif target == "extra_failure":
        changed["summary"]["failure"] = {
            "phase": "policy_inference",
            "type": "ValueError",
            "message": "invented",
        }
    with pytest.raises(ValueError, match="differ"):
        evaluation.verify_residual_episode_replay(config, changed, fixture=True)


def test_invalid_policy_preserves_replayable_failure_prefix(tmp_path: Path):
    class FailsAfterTwo(ConstantPolicy):
        def predict(self, observation, *, deterministic):
            action, state = super().predict(observation, deterministic=deterministic)
            if self.calls == 3:
                action[0] = np.nan
            return action, state

    config = research_batch.template(tmp_path)
    episode = evaluation.evaluate_residual_episode(
        config,
        policy=FailsAfterTwo([0, 0, 0, 0]),
        seed=9_990_621,
        fixture=True,
        max_decisions=4,
    )
    assert episode["summary"]["failure"]["phase"] == "policy_inference"
    assert episode["summary"]["decisions"] == 2
    assert episode["failed_decision"]["raw_action"][0] == "nan"
    replay = evaluation.verify_residual_episode_replay(config, episode, fixture=True)
    assert replay["physical_steps"] == 30
    assert replay["controller_failure_authenticity_verified"] is False
    for mutation in ("shape", "mapping", "phase", "missing_failure"):
        altered = deepcopy(episode)
        pending = altered["failed_decision"]
        if mutation == "shape":
            pending["raw_action_shape"] = [3]
        elif mutation == "mapping":
            pending["command"] = {"made_up": True}
        elif mutation == "phase":
            altered["summary"]["failure"]["phase"] = "environment_execution"
        else:
            altered["summary"]["failure"] = None
        with pytest.raises(ValueError, match="differ"):
            evaluation.verify_residual_episode_replay(config, altered, fixture=True)


def test_clipped_residual_is_visible_and_replayable(tmp_path: Path):
    config = research_batch.template(tmp_path)
    episode = evaluation.evaluate_residual_episode(
        config,
        policy=ConstantPolicy([2, -3, 0.5, 0]),
        seed=9_990_622,
        fixture=True,
        max_decisions=1,
    )
    assert episode["summary"]["failure"] is None
    row = episode["history"][0]
    assert row["raw_action"] == [2, -3, 0.5, 0]
    assert row["residual_evidence"]["body"]["residual_clipped"] is True
    assert row["controller_invalid"] is True
    assert evaluation.verify_residual_episode_replay(config, episode, fixture=True)[
        "verified"
    ]


def test_post_execution_evidence_failure_is_integrity_error(tmp_path, monkeypatch):
    config = research_batch.template(tmp_path)
    advanced = []

    def broken_capture(base, transition, minutes, *, actor):
        advanced.append(base.decisions_taken)
        raise ValueError("injected archive capture failure")

    monkeypatch.setattr(evaluation.physical, "_interval", broken_capture)
    with pytest.raises(RuntimeError, match="integrity failure in evidence_capture"):
        evaluation.evaluate_residual_episode(
            config,
            policy=ConstantPolicy([0, 0, 0, 0]),
            seed=9_990_623,
            fixture=True,
            max_decisions=1,
        )
    assert advanced == [1]


@pytest.mark.parametrize(
    "seed,fixture,prefix",
    [(42, False, None), (9_990_620, False, None), (9_700_100, False, 1)],
)
def test_forbidden_seed_or_production_prefix_does_not_reset(
    tmp_path, monkeypatch, seed, fixture, prefix
):
    config = research_batch.template(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid admission reached environment construction")

    monkeypatch.setattr(evaluation.physical, "_environment", forbidden)
    with pytest.raises(ValueError):
        evaluation.evaluate_residual_episode(
            config,
            policy=ConstantPolicy([0, 0, 0, 0]),
            seed=seed,
            fixture=fixture,
            max_decisions=prefix,
        )

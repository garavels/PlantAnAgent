"""Training integrity checks use a tiny synthetic fixture, never evaluation cases."""
from __future__ import annotations

from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
import shutil
import time

import gymnasium as gym
import numpy as np
import pytest

from docs.research.tools.free_ppo_v1 import train


class FixtureEnv(gym.Env):
    observation_space = gym.spaces.Box(-100.0, 100.0, (135,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (5,), dtype=np.float32)

    def __init__(self, split="train"):
        assert split == "train"
        self.index = 0
        self.selected_seed = None

    def reset(self, *, seed=None, options=None):
        self.index = 0
        self.selected_seed = seed
        return np.full(135, (seed % 7) + 1, dtype=np.float32), {}

    def step(self, action):
        self.index += 1
        observation = np.full(135, self.index + self.selected_seed % 7, dtype=np.float32)
        return observation, float(1000.0 * (1.0 - np.square(action - 0.25).sum())), self.index == 8, False, {}


@pytest.fixture
def fixture_environment(monkeypatch):
    module = SimpleNamespace(DirectPlantEnv=FixtureEnv, source_records=lambda: [],
                             environment_identity=lambda: {"fixture": True, "observation_size": 135, "action_size": 5})
    monkeypatch.setattr(train, "_environment", lambda: module)


def test_allocator_separates_scenarios_from_optimizer_rng(fixture_environment):
    wrapper = train.TrainingEpisodes(FixtureEnv(), index=2, count=4)
    wrapper.reset(seed=101)
    assert wrapper.current_seed == 10_000_002
    wrapper.reset(seed=307)
    assert wrapper.current_seed == 10_000_006


def test_actual_updates_and_checkpoint_normalizer_reopen(tmp_path, fixture_environment):
    root = tmp_path / "training"
    train.freeze_plan(root, total_timesteps=256, optimizer_seeds=(101,), n_steps=64,
                      deadline_utc="2099-01-01T00:00:00Z")
    result = train.train(root, 101)
    assert result["status"] == "completed"
    assert result["num_timesteps"] == 256
    assert result["actor_parameters_changed"]
    assert result["completed_episodes"] == 32
    checkpoint = root / "seed-101/final-policy.zip"
    loaded = train.load_checkpoint(checkpoint, root / "plan.json")
    try:
        observation = np.full(135, 5, dtype=np.float32)
        before_count = loaded.normalizer.obs_rms.count
        details = loaded.deterministic_details(observation)
        prediction, _ = loaded.predict(observation, deterministic=True)
        np.testing.assert_array_equal(prediction, details["clipped_action"])
        assert not np.array_equal(details["normalized_input"], observation)
        assert loaded.normalizer.obs_rms.count == before_count
        assert loaded.normalizer.training is False
        assert loaded.normalizer.norm_reward is False
    finally:
        loaded.close()
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    for original in (checkpoint, checkpoint.with_suffix(".json"), checkpoint.with_suffix(".vecnormalize.pkl")):
        shutil.copy2(original, relocated / original.name)
    relocated_policy = train.load_checkpoint(relocated / checkpoint.name)
    try:
        relocated_action, _ = relocated_policy.predict(observation, deterministic=True)
        np.testing.assert_array_equal(prediction, relocated_action)
    finally:
        relocated_policy.close()
    assert train.require_plan(root)["total_timesteps"] == 256


def test_normalizer_tampering_is_rejected(tmp_path, fixture_environment):
    root = tmp_path / "training"
    plan = train.freeze_plan(root, total_timesteps=256, optimizer_seeds=(101,), n_steps=64,
                             deadline_utc="2099-01-01T00:00:00Z")
    model, vector, wrappers = train.make_learner(plan, 101)
    path = root / "initial-policy.zip"
    try:
        train.save_checkpoint(model, path, plan=plan, wrappers=wrappers, kind="initial")
    finally:
        vector.close()
    statistics = path.with_suffix(".vecnormalize.pkl")
    statistics.write_bytes(statistics.read_bytes() + b"unexpected")
    with pytest.raises(ValueError, match="observation statistics differ"):
        train.load_checkpoint(path, root / "plan.json")


def test_deadline_preserves_incomplete_without_final_endpoint(tmp_path, monkeypatch):
    class SlowFixture(FixtureEnv):
        def step(self, action):
            time.sleep(0.3)
            return super().step(action)

    module = SimpleNamespace(DirectPlantEnv=SlowFixture, source_records=lambda: [],
                             environment_identity=lambda: {"fixture": True})
    monkeypatch.setattr(train, "_environment", lambda: module)
    deadline = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()
    root = tmp_path / "deadline"
    train.freeze_plan(root, total_timesteps=256, optimizer_seeds=(101,), n_steps=64,
                      deadline_utc=deadline)
    result = train.train(root, 101)
    assert result["status"] == "deadline_incomplete"
    assert result["num_timesteps"] < 256
    assert (root / "seed-101/incomplete-policy.zip").exists()
    assert not (root / "seed-101/final-policy.zip").exists()

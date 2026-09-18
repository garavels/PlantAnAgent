"""Engineering checks for learning plumbing, not claims of policy quality."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import pytest

from docs.research.tools.learned_control_v1 import training as t
from docs.research.tools.learned_control_v1.imitation import transform_targets, undiscounted_returns


def tiny_recipe(**changes):
    recipe = t.default_recipe(total_timesteps=64, environments=2, n_steps=16,
        checkpoint_steps=32, deadline_utc=(datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat())
    recipe["ppo"].update(batch_size=16, n_epochs=1)
    recipe.update(changes)
    return recipe


def test_continuation_learning_schedule_is_relative_to_added_steps():
    schedule = t.RunLinearSchedule(.001, .0001, 1000, 100)
    assert schedule(1 - 1000 / 1100) == pytest.approx(.001)
    assert schedule(1 - 1050 / 1100) == pytest.approx(.00055)
    assert schedule(0) == pytest.approx(.0001)


def test_off_label_transform_does_not_change_physical_actions():
    actions = np.array([[-1, -.5, 0, .5, 1, -1, .9]], dtype=np.float32)
    targets = transform_targets(actions, .1, .05)
    np.testing.assert_array_equal(np.clip(targets, -1, 1), actions)
    assert targets[0, 0] == pytest.approx(-1.1)
    assert targets[0, 4] == pytest.approx(1.05)
    with pytest.raises(ValueError):
        transform_targets(np.full((2, 7), np.nan))


def test_critic_labels_are_actual_undiscounted_returns():
    np.testing.assert_allclose(undiscounted_returns([2, -3, 7]), [6, 4, 7])


def test_optimizer_independent_scenario_allocator():
    from docs.research.tools.learned_control_v1.environment import MixtureEnv
    one = t.TrainingEpisodes(MixtureEnv(sampler_seed=80), 0)
    two = t.TrainingEpisodes(MixtureEnv(sampler_seed=80), 0)
    try:
        o1, i1 = one.reset(seed=100)
        o2, i2 = two.reset(seed=999)
        np.testing.assert_array_equal(o1, o2)
        assert i1["training_scenario"] == i2["training_scenario"]
    finally:
        one.close()
        two.close()


def test_small_training_checkpoint_exact_reopen_and_resume(tmp_path):
    root = tmp_path / "fresh"
    plan = t.freeze_plan(root, tiny_recipe())
    result = t.train(root)
    assert result["status"] == "completed"
    assert result["additional_timesteps"] == 64
    assert result["actor_mean_parameters_changed"]
    assert result["endpoint"]["entropy_coefficient"] == 0.
    checkpoint = root / "training/final-policy.zip"
    loaded = t.load_checkpoint(checkpoint, root / "plan.json")
    from docs.research.tools.learned_control_v1.environment import FixedEnv
    env = FixedEnv(condition="ordinary", split="train")
    try:
        obs, _ = env.reset(seed=50_010_100)
        action, _ = loaded.predict(obs, deterministic=True)
        details = loaded.deterministic_details(obs)
        np.testing.assert_array_equal(action, details["clipped_action"])
        assert loaded.normalizer.training is False
        source_actor_hash = t.actor_mean_parameter_sha256(loaded.model)
        policy_kwargs = loaded.model.policy_kwargs
    finally:
        loaded.close()
        env.close()
    recipe = tiny_recipe(initialization="continue", initial_checkpoint=str(checkpoint),
                         optimizer_seed=1203, reset_log_std=-2.)
    recipe["ppo"]["policy_kwargs"] = policy_kwargs
    resumed_root = tmp_path / "resumed"
    resumed_plan = t.freeze_plan(resumed_root, recipe)
    model, vector, wrappers = t.make_learner(resumed_plan)
    try:
        assert model.num_timesteps == 64
        assert t.actor_mean_parameter_sha256(model) == source_actor_hash
        np.testing.assert_array_equal(model.policy.log_std.detach().numpy(), np.full(7, -2.))
    finally:
        vector.close()
    result = t.train(resumed_root)
    assert result["num_timesteps"] == 128
    assert result["additional_timesteps"] == 64
    assert result["status"] == "completed"


def test_changed_frozen_dependency_fails_before_training(tmp_path):
    extra = tmp_path / "protocol.txt"
    extra.write_text("first")
    root = tmp_path / "run"
    t.freeze_plan(root, tiny_recipe(extra_source_paths=[str(extra)]))
    extra.write_text("different")
    with pytest.raises(ValueError, match="source changed"):
        t.require_plan(root)


def test_behavior_clone_pipeline_labels_actions_and_changes_actor(tmp_path):
    from docs.research.tools.learned_control_v1.imitation import collect_dataset, require_dataset
    data_root = tmp_path / "demonstrations"
    metadata = collect_dataset(data_root, [{"condition": "ordinary", "seed": 50_020_100}],
                               deadline_utc=(datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat())
    assert metadata["samples"] == 672
    dataset = data_root / "demonstrations.npz"
    require_dataset(dataset)
    recipe = tiny_recipe(initialization="behavior_clone", dataset=str(dataset),
                         freeze_observation_statistics=True)
    recipe["behavior_clone"].update(epochs=2, critic_epochs=1, batch_size=256)
    root = tmp_path / "bc"
    t.freeze_plan(root, recipe)
    result = t.train(root)
    assert result["status"] == "completed"
    assert result["before_initialization"]["actor_mean_tensor_sha256"] != result["initial_checkpoint"]["actor_mean_tensor_sha256"]
    assert result["initial_checkpoint"]["normalizer_observation_count"] == pytest.approx(672.0001)
    assert result["endpoint"]["normalizer_observation_count"] == result["initial_checkpoint"]["normalizer_observation_count"]

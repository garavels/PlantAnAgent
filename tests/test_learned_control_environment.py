"""Engineering parity and split isolation; no final-split outcomes are evaluated."""
import hashlib
import json

import numpy as np
import pytest

from docs.research.tools.learned_control_v1 import environment as adapter
from docs.research.tools.tail_stress_v1.environment import TailStressEnv
from docs.research.tools.wholeplant_ppo_v1.environment import WholePlantEnv


@pytest.mark.parametrize("condition", adapter.CONDITIONS)
def test_every_existing_condition_replays_exactly_through_adapter(condition):
    """The original classes draw a new training seed through a test-only admission."""
    seed = adapter.TRAIN_BASE
    reference = WholePlantEnv(record_minutes=True) if condition == "ordinary" else TailStressEnv(condition, record_minutes=True)
    reference.allowed_seeds = range(seed, seed + 1)
    env = adapter.FixedEnv(condition, record_minutes=True)
    try:
        obs0, info0 = reference.reset(seed=seed)
        obs1, info1 = env.reset(seed=seed)
        np.testing.assert_array_equal(obs0, obs1)
        assert {k: v for k, v in info1.items() if k != "learned_control_admission"} == info0
        assert reference.audit_manifest() == env.env.audit_manifest()
        assert env.action_space.shape == (7,) and obs1.shape == (149,)
        assert env.episode_decisions == 672 and env.episode_physics_steps == 10080
        assert reference.frozen_source.original_manifest == env.frozen_source.original_manifest
        # An identical prescribed action tape reaches all outage and repair seams.
        actions = (np.array([0.2, 0.8, -0.4, 0.7, -0.6, -0.8, -0.95]), -np.ones(7))
        for decision in range(672):
            action = actions[(decision // 17) % 2]
            left, right = reference.step(action), env.step(action)
            np.testing.assert_array_equal(left[0], right[0])
            assert left[1:] == right[1:]
            assert reference.last_minute_truth_states == env.last_minute_truth_states
            assert reference.last_direct_minute_actions == env.last_direct_minute_actions
        assert env.simulator.delivery_audit() == reference.simulator.delivery_audit()
        assert env.audit_manifest()["learned_control_admission"]["condition"] == condition
    finally:
        reference.close()
        env.close()


def test_split_admission_rejects_prior_seeds_cross_split_and_nonintegers():
    train = adapter.FixedEnv()
    development = adapter.FixedEnv(split="development")
    try:
        for seed in (30_000_000, 43_000_000, adapter.DEVELOPMENT_SEEDS[0], adapter.FINAL_SEEDS[0],
                     True, float(adapter.TRAIN_BASE), None):
            with pytest.raises(ValueError, match="outside"):
                train.reset(seed=seed)
        for seed in (adapter.TRAIN_BASE, adapter.FINAL_SEEDS[0], True):
            with pytest.raises(ValueError, match="outside"):
                development.reset(seed=seed)
        obs, _ = development.reset(seed=adapter.DEVELOPMENT_SEEDS[0])
        assert obs.shape == (149,)
        np.testing.assert_array_equal(obs, development.reset()[0])
    finally:
        train.close()
        development.close()
    with pytest.raises(ValueError, match="gate"):
        adapter.FixedEnv(split="final")
    with pytest.raises(ValueError, match="condition"):
        adapter.FixedEnv("pv100")
    with pytest.raises(ValueError, match="split"):
        adapter.FixedEnv(split="test")


def test_final_gate_requires_actual_pinned_recipe_and_detects_changed_gate(tmp_path):
    """Validate admission controls without drawing any final episode."""
    recipe = tmp_path / "selected-recipe.json"
    recipe.write_text('{"selected": "engineering fixture, no final execution"}\n')
    gate = tmp_path / "gate.json"
    payload = {
        "schema": "learned-control-v1-evaluation-gate",
        "environment_identity_sha256": adapter.environment_identity_sha256(),
        "source_identity_sha256": adapter.source_identity_sha256(),
        "allowed_splits": ["final"], "selected_recipe_frozen": True,
        "selected_recipe": {"path": recipe.name, "sha256": hashlib.sha256(recipe.read_bytes()).hexdigest()},
    }
    gate.write_text(json.dumps(payload))
    env = adapter.FixedEnv(split="final", gate_path=gate)
    try:
        assert env._last_seed is None and env.env._episode is None
        gate.write_text(json.dumps({**payload, "extra": "changed after admission"}))
        with pytest.raises(ValueError, match="changed"):
            env.reset(seed=adapter.TRAIN_BASE)
        gate.write_text(json.dumps(payload))
        recipe.write_text('{"selected": "tampered"}\n')
        with pytest.raises(ValueError, match="recipe differs"):
            env.reset(seed=adapter.TRAIN_BASE)
    finally:
        env.close()


def test_mixture_is_reproducible_resets_full_state_and_keeps_condition_private():
    first = adapter.MixtureEnv(sampler_seed=711)
    second = adapter.MixtureEnv(sampler_seed=711)
    try:
        for _ in range(3):
            obs0, info0 = first.reset()
            obs1, info1 = second.reset()
            assert info0["training_scenario"] == info1["training_scenario"]
            np.testing.assert_array_equal(obs0, obs1)
            assert obs0.shape == (149,) and first.action_space.shape == (7,)
            scenario = first.training_scenario
            assert scenario["condition"] in adapter.DEFAULT_MIXTURE
            assert adapter.TRAIN_BASE <= scenario["seed"] < adapter.TRAIN_STOP
            reference = adapter.FixedEnv(scenario["condition"])
            try:
                expected, _ = reference.reset(seed=scenario["seed"])
                np.testing.assert_array_equal(obs0, expected)
                assert first.simulator.state == reference.simulator.state
            finally:
                reference.close()
            first.step(np.ones(7))
            second.step(-np.ones(7))
        np.testing.assert_array_equal(first.reset(seed=711)[0], second.reset(seed=711)[0])
        assert first.training_scenario["episode_index"] == 0
    finally:
        first.close()
        second.close()


def test_mixture_seed_stream_does_not_depend_on_condition_weights():
    ordinary = adapter.MixtureEnv({"ordinary": 1}, sampler_seed=91)
    stress = adapter.MixtureEnv({"pv18": 1}, sampler_seed=91)
    try:
        for _ in range(2):
            ordinary.reset()
            stress.reset()
            assert ordinary.training_scenario["seed"] == stress.training_scenario["seed"]
            assert ordinary.training_scenario["condition"] == "ordinary"
            assert stress.training_scenario["condition"] == "pv18"
    finally:
        ordinary.close()
        stress.close()
    for weights in ({}, {"pv99": 1}, {"ordinary": -1}, {"ordinary": float("nan")}, {"control": 0}):
        with pytest.raises(ValueError, match="Mixture|mixture"):
            adapter.MixtureEnv(weights)


def test_identity_pins_both_original_environment_implementations():
    identity = adapter.environment_identity()
    assert identity["ordinary_environment_identity_sha256"] == adapter.ordinary.environment_identity_sha256()
    assert identity["tail_environment_identity_sha256"] == adapter.tail.environment_identity_sha256()
    paths = {row["path"] for row in adapter.source_records()}
    assert str(adapter.Path(adapter.ordinary.__file__).resolve()) in paths
    assert str(adapter.Path(adapter.tail.__file__).resolve()) in paths
    assert sum(adapter.DEFAULT_MIXTURE.values()) == pytest.approx(1)

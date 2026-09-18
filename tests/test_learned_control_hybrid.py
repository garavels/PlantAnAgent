"""Probability and engineering checks for the unrestricted hybrid actor."""
from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.stats import norm

from docs.research.tools.learned_control_hybrid_v1.policy import (
    OffCensoredGaussianDistribution as Distribution, LOWER_CENSOR, UPPER_CENSOR,
)
from docs.research.tools.learned_control_hybrid_v1 import training as t


def one_distribution(q=.3, mean=.2, sigma=.25, *, min_std=.0001, max_std=2.):
    fraction = (sigma - min_std) / (max_std - min_std)
    parameters = torch.tensor([[math.log(q / (1 - q)), mean, math.log(fraction / (1 - fraction))]], dtype=torch.float64)
    return Distribution(1, min_on_std=min_std, max_on_std=max_std).proba_distribution(parameters)


@pytest.mark.parametrize("q,mean,sigma", [(.3, .2, .25), (.01, -1., .3), (.8, 1.2, .5)])
def test_marginal_probability_normalizes_and_matches_numerical_entropy(q, mean, sigma):
    dist = one_distribution(q, mean, sigma)
    masses = {key: float(value) for key, value in dist.coordinate_masses().items()}
    assert sum(masses.values()) == pytest.approx(1., abs=1e-13)
    expected_left = q + (1 - q) * norm.cdf((LOWER_CENSOR - mean) / sigma)
    expected_right = (1 - q) * norm.sf((UPPER_CENSOR - mean) / sigma)
    assert masses["off"] == pytest.approx(expected_left, abs=1e-13)
    assert masses["maximum"] == pytest.approx(expected_right, abs=1e-13)
    def density(action):
        return math.exp(float(dist.log_prob(torch.tensor([[action]], dtype=torch.float64))))
    interior = quad(density, LOWER_CENSOR, UPPER_CENSOR, epsabs=1e-11)[0]
    assert interior + masses["off"] + masses["maximum"] == pytest.approx(1., abs=1e-10)
    entropy = -sum(p * math.log(p) for p in (masses["off"], masses["maximum"]) if p > 0)
    entropy -= quad(lambda action: density(action) * math.log(density(action)), LOWER_CENSOR, UPPER_CENSOR, epsabs=1e-10)[0]
    assert float(dist.entropy()) == pytest.approx(entropy, abs=1e-9)


def test_sampling_endpoint_frequencies_match_marginal_scores():
    torch.manual_seed(718)
    dist = one_distribution(q=.25, mean=.6, sigma=.5)
    parameters = dist.parameters.repeat(120000, 1)
    samples = Distribution(1, min_on_std=.0001, max_on_std=2.).proba_distribution(parameters).sample()
    masses = dist.coordinate_masses()
    assert float((samples == -1).double().mean()) == pytest.approx(float(masses["off"]), abs=.005)
    assert float((samples == 1).double().mean()) == pytest.approx(float(masses["maximum"]), abs=.005)
    assert torch.isfinite(dist.log_prob(torch.tensor([[-1.], [1.], [0.]]))).all()


def test_extreme_tail_scores_entropy_and_gradients_are_finite():
    parameters = torch.tensor([[100., -100., 100., -100., 100., -100., 0.,
                               -100., -2., -.9999, 0., .9999, 2., 100.,
                               -100., -100., -100., -100., -100., -100., -100.]], requires_grad=True)
    dist = Distribution(7, min_on_std=.0001, max_on_std=.1).proba_distribution(parameters)
    for actions in (torch.full((1, 7), -1.), torch.zeros(1, 7), torch.ones(1, 7)):
        assert torch.isfinite(dist.log_prob(actions)).all()
    loss = -dist.log_prob(torch.zeros(1, 7)).mean() - .001 * dist.entropy().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(parameters.grad).all()


@pytest.mark.parametrize("action,expected_sign", [(-1., 1), (0., -1)])
def test_gate_can_learn_both_directions(action, expected_sign):
    gate = torch.tensor([[0.]], requires_grad=True)
    optimizer = torch.optim.Adam([gate], lr=.1)
    for _ in range(30):
        params = torch.cat((gate, torch.zeros_like(gate), torch.zeros_like(gate)), dim=-1)
        dist = Distribution(1, min_on_std=.01, max_on_std=.2).proba_distribution(params)
        loss = -dist.log_prob(torch.tensor([[action]])).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float(gate.detach()) * expected_sign > 1.


def test_deterministic_gate_returns_exact_off_and_retains_full_box_range():
    params = torch.tensor([[2., -2., -2., 0., .2, 5., 0., 0., 0.]])
    dist = Distribution(3).proba_distribution(params)
    np.testing.assert_array_equal(dist.mode().numpy(), np.array([[-1., .2, 1.]], dtype=np.float32))
    near_endpoint = one_distribution(q=.01, mean=1 - 2 ** -27)
    assert float(near_endpoint.mode()) == 1.
    assert torch.isfinite(near_endpoint.log_prob(near_endpoint.mode())).all()


def test_rare_on_labels_receive_equal_class_weight():
    from docs.research.tools.learned_control_hybrid_v1.imitation import balanced_gate_weights
    off = np.ones((100, 7), dtype=bool)
    off[:3] = False
    weights = balanced_gate_weights(off)
    np.testing.assert_allclose((weights * off).sum(axis=0, dtype=np.float64),
                               (weights * ~off).sum(axis=0, dtype=np.float64), rtol=1e-6)


def tiny_recipe(**changes):
    recipe = t.default_recipe(total_timesteps=64, environments=2, n_steps=16, checkpoint_steps=32,
        freeze_observation_statistics=False,
        deadline_utc=(datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat())
    recipe["ppo"].update(batch_size=16, n_epochs=1)
    recipe.update(changes)
    return recipe


def test_tiny_ppo_checkpoint_reopens_and_old_evaluator_loader_accepts(tmp_path):
    from docs.research.tools.wholeplant_ppo_v1 import training as common
    from docs.research.tools.learned_control_v1.environment import FixedEnv
    root = tmp_path / "hybrid"
    t.freeze_plan(root, tiny_recipe())
    result = t.train(root)
    assert result["status"] == "completed"
    assert result["actor_mean_parameters_changed"]
    assert result["endpoint"]["gaussian_log_std"] is None
    checkpoint = root / "training/final-policy.zip"
    loaded, compatible = t.load_checkpoint(checkpoint), common.load_checkpoint(checkpoint)
    env = FixedEnv(condition="ordinary", split="train")
    try:
        obs, _ = env.reset(seed=50_009_001)
        one = loaded.predict(obs, deterministic=True)[0]
        two = compatible.predict(obs, deterministic=True)[0]
        np.testing.assert_array_equal(one, two)
        np.testing.assert_array_equal(one, loaded.deterministic_details(obs)["clipped_action"])
        assert loaded.normalizer.training is False
    finally:
        loaded.close()
        compatible.close()
        env.close()
    resume = tiny_recipe(initialization="continue", initial_checkpoint=str(checkpoint), optimizer_seed=1103)
    resumed = tmp_path / "resumed"
    t.freeze_plan(resumed, resume)
    second = t.train(resumed)
    assert second["status"] == "completed"
    assert second["num_timesteps"] == 128


def test_checkpoint_verification_does_not_reset_training_random_streams(tmp_path):
    import random
    root = tmp_path / "rng"
    plan = t.freeze_plan(root, tiny_recipe())
    model, vector, wrappers = t.make_learner(plan)
    try:
        random.seed(7181)
        np.random.seed(7172)
        torch.manual_seed(7134)
        expected_python, expected_numpy, expected_torch = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        t.save_checkpoint(model, root / "probe.zip", plan=plan, wrappers=wrappers, kind="engineering")
        assert random.getstate() == expected_python
        actual = np.random.get_state()
        assert actual[0] == expected_numpy[0]
        np.testing.assert_array_equal(actual[1], expected_numpy[1])
        assert actual[2:] == expected_numpy[2:]
        assert torch.equal(torch.get_rng_state(), expected_torch)
    finally:
        vector.close()


def test_hybrid_clone_pipeline_changes_gate_and_preserves_exact_interface(tmp_path):
    from docs.research.tools.learned_control_v1.imitation import collect_dataset
    root = tmp_path / "data"
    collect_dataset(root, [{"condition": "ordinary", "seed": 50_030_001}],
                    deadline_utc=(datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat())
    recipe = tiny_recipe(initialization="behavior_clone", dataset=str(root / "demonstrations.npz"),
                         freeze_observation_statistics=True)
    recipe["behavior_clone"].update(epochs=2, critic_epochs=1, batch_size=256)
    attempt = tmp_path / "cloned"
    t.freeze_plan(attempt, recipe)
    result = t.train(attempt)
    assert result["status"] == "completed"
    assert result["before_initialization"]["actor_tensor_sha256"] != result["initial_checkpoint"]["actor_tensor_sha256"]
    assert result["initial_checkpoint"]["normalizer_observation_count"] == pytest.approx(672.0001)
    assert result["endpoint"]["normalizer_observation_count"] == result["initial_checkpoint"]["normalizer_observation_count"]

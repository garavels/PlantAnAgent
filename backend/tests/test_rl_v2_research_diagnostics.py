"""Read-only PPO diagnostics on tiny synthetic cost-transition fixtures."""

from __future__ import annotations

import copy
import json
import random
from typing import Any

import numpy as np
import pytest

from plant_backend.rl.v2.training.constrained_ppo import (
    BETA_DECISIONS_PER_CYCLE,
    ConstrainedPPO,
    ConstrainedPPOSettingsV1,
    PI_LAGRANGIAN_ALGORITHM_ID,
    SCALAR_PPO_ALGORITHM_ID,
    training_dependencies_available,
)
from plant_backend.rl.v2.training.research_diagnostics import (
    DIAGNOSTIC_SCHEMA,
    _explained_variance,
    capture_rollout_diagnostics,
)

from test_rl_v2_constrained_ppo import _FixtureCostEnv, _assert_nested_equal


pytestmark = pytest.mark.skipif(
    not training_dependencies_available(), reason="optional train dependencies absent"
)


def _model(algorithm_id: str, *, n_envs: int = 3) -> ConstrainedPPO:
    from stable_baselines3.common.vec_env import DummyVecEnv

    tasks = [
        _FixtureCostEnv(settlement_raw_cost=0.7 + index * 0.3)
        for index in range(n_envs)
    ]
    environment = DummyVecEnv([lambda task=task: task for task in tasks])
    model = ConstrainedPPO(
        "MlpPolicy",
        environment,
        n_steps=BETA_DECISIONS_PER_CYCLE,
        batch_size=BETA_DECISIONS_PER_CYCLE,
        n_epochs=1,
        constraint_settings=ConstrainedPPOSettingsV1.for_algorithm(
            algorithm_id, optimizer_seed=101
        ),
        seed=101,
        verbose=0,
    )
    environment.seed(9_990_100)
    return model


def _snapshot(model: ConstrainedPPO) -> dict[str, Any]:
    import torch

    policy = model.policy
    buffer = model.rollout_buffer
    return {
        "python_rng": random.getstate(),
        "numpy_rng": copy.deepcopy(np.random.get_state()),
        "torch_rng": torch.random.get_rng_state().clone(),
        "parameters": copy.deepcopy(policy.state_dict()),
        "gradients": {
            name: None if value.grad is None else value.grad.clone()
            for name, value in policy.named_parameters()
        },
        "optimizer": copy.deepcopy(policy.optimizer.state_dict()),
        "modes": tuple(module.training for module in policy.modules()),
        "buffer": {
            name: value.copy()
            for name, value in vars(buffer).items()
            if isinstance(value, np.ndarray)
        },
        "buffer_generator_ready": buffer.generator_ready,
        "timesteps": model.num_timesteps,
        "updates": model._n_updates,
        "multipliers": model.multiplier_state,
        "ledgers": model.rollout_ledgers,
        "distribution_identity": id(getattr(policy.action_dist, "distribution", None)),
    }


def _assert_unchanged(before: dict[str, Any], model: ConstrainedPPO) -> None:
    after = _snapshot(model)
    np.testing.assert_array_equal(before["numpy_rng"][1], after["numpy_rng"][1])
    assert before["numpy_rng"][0] == after["numpy_rng"][0]
    assert before["numpy_rng"][2:] == after["numpy_rng"][2:]
    before.pop("numpy_rng")
    after.pop("numpy_rng")
    assert before["buffer"].keys() == after["buffer"].keys()
    for name in before["buffer"]:
        np.testing.assert_array_equal(before["buffer"][name], after["buffer"][name])
    before.pop("buffer")
    after.pop("buffer")
    _assert_nested_equal(before, after)


@pytest.mark.parametrize(
    "algorithm_id", [PI_LAGRANGIAN_ALGORITHM_ID, SCALAR_PPO_ALGORITHM_ID]
)
def test_capture_before_optimization_preserves_all_training_state(
    algorithm_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stable_baselines3.common.callbacks import BaseCallback

    model = _model(algorithm_id)
    records: list[dict[str, Any]] = []

    class Capture(BaseCallback):
        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            before = _snapshot(model)
            with monkeypatch.context() as patch:
                # An accidental shuffled iterator or policy distribution call
                # would make the diagnostics invasive even before an update.
                def forbidden(*args: Any, **kwargs: Any) -> None:
                    raise AssertionError("diagnostic used a mutable inference path")

                patch.setattr(model.rollout_buffer, "get", forbidden)
                patch.setattr(model.policy, "evaluate_actions", forbidden)
                patch.setattr(model.policy, "get_distribution", forbidden)
                record = capture_rollout_diagnostics(model)
                assert capture_rollout_diagnostics(model) == record
            _assert_unchanged(before, model)
            model.policy.set_training_mode(True)
            before_training_mode = _snapshot(model)
            assert capture_rollout_diagnostics(model) == record
            _assert_unchanged(before_training_mode, model)
            model.policy.set_training_mode(False)
            records.append(record)

    try:
        model.learn(2 * BETA_DECISIONS_PER_CYCLE * 3, callback=Capture())
        assert len(records) == 2
        assert model._n_updates == 2
        for record in records:
            json.dumps(record, allow_nan=False)
            assert record["schema"] == DIAGNOSTIC_SCHEMA
            assert record["sample"]["count"] == 256
            assert record["rollout_transition_count"] == 288
            assert not record["sample"]["actual_shuffled_sgd_minibatch"]
            assert len(record["sample"]["reward_critic_tanh"]) == 2
            clipping = record["sample"]["gaussian"]
            assert 0 <= clipping["expected_any_action_clipped_fraction"] <= 1
            attribution = record["gradient_attribution"]
            assert 0 < attribution["implied_common_clip_factor"] <= 1
            assert attribution["reward_actor_gradient_norm"] > 0
            assert attribution["reward_critic_preclip_gradient_norm"] > 0
            if algorithm_id == SCALAR_PPO_ALGORITHM_ID:
                assert record["full_rollout"]["costs"] is None
                assert record["sample"]["costs"] is None
                assert attribution["cost_heads"] is None
                assert attribution["cost_actor_components"] is None
                assert attribution["weighted_cost_actor_gradient_norm"] is None
            else:
                assert len(attribution["cost_heads"]) == 4
                assert len(attribution["cost_actor_components"]) == 4
                for channel in record["full_rollout"]["costs"]:
                    assert channel["raw_unit_cost_advantage"]["mean"] == pytest.approx(
                        7 * channel["training_cost_advantage"]["mean"]
                    )
        # Second callback observed populated Adam state and previously accumulated
        # .grad tensors; its exact-state assertion covers both, not just an empty
        # optimizer before the first update.
        assert model.policy.optimizer.state_dict()["state"]
    finally:
        model.get_env().close()


def test_diagnostic_gaussian_and_gradient_match_production_loss() -> None:
    import torch
    from stable_baselines3.common.callbacks import BaseCallback

    from plant_backend.rl.v2.training.constrained_ppo import _torch_normalize_advantage

    model = _model(PI_LAGRANGIAN_ALGORITHM_ID, n_envs=3)

    class Capture(BaseCallback):
        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            record = capture_rollout_diagnostics(model)
            buffer = model.rollout_buffer
            tensor = lambda value: torch.as_tensor(value, device=model.device)

            def samples(field: str) -> Any:
                values = getattr(buffer, field)
                return tensor(
                    values.swapaxes(0, 1).reshape((-1, *values.shape[2:]))[:256]
                )

            obs = samples("observations")
            actions = samples("actions")
            values, log_prob, entropy = model.policy.evaluate_actions(obs, actions)
            advantages = _torch_normalize_advantage(samples("advantages"))
            all_cost_advantages = samples("cost_advantages")
            cost_adv = torch.stack(
                [
                    _torch_normalize_advantage(all_cost_advantages[:, i])
                    for i in range(4)
                ],
                dim=1,
            )
            multipliers = tensor(
                np.array(model.multiplier_state.multipliers, dtype=np.float32)
            )
            combined = (advantages - (cost_adv * multipliers).sum(dim=1)) / (
                1 + multipliers.sum()
            )
            ratio = (log_prob - samples("log_probs")).exp()
            clip = model.clip_range(model._current_progress_remaining)
            policy_loss = -torch.minimum(
                combined * ratio, combined * ratio.clamp(1 - clip, 1 + clip)
            ).mean()
            loss = (
                policy_loss
                - model.ent_coef * entropy.mean()
                + model.vf_coef
                * torch.nn.functional.mse_loss(values.flatten(), samples("returns"))
            )
            params = tuple(
                p
                for name, p in model.policy.named_parameters()
                if not name.startswith("cost_critic.")
            )
            gradients = torch.autograd.grad(loss, params, allow_unused=True)
            expected = float(
                torch.sqrt(
                    sum(g.double().square().sum() for g in gradients if g is not None)
                )
            )
            assert record["gradient_attribution"][
                "actor_and_reward_common_preclip_gradient_norm"
            ] == pytest.approx(expected, rel=2e-6)
            expected_clip = min(1.0, model.max_grad_norm / (expected + 1e-6))
            assert record["gradient_attribution"][
                "implied_common_clip_factor"
            ] == pytest.approx(expected_clip, rel=2e-6)
            means = model.policy.action_dist.distribution.mean.detach().numpy()
            assert record["sample"]["gaussian"]["mean_by_action"][0][
                "mean"
            ] == pytest.approx(float(means[:, 0].mean()), abs=1e-8)

    try:
        model.learn(3 * BETA_DECISIONS_PER_CYCLE, callback=Capture())
    finally:
        model.get_env().close()


def test_incomplete_or_flattened_rollout_rejected_without_mutation() -> None:
    model = _model(SCALAR_PPO_ALGORITHM_ID)
    try:
        before = _snapshot(model)
        with pytest.raises(ValueError, match="full, unflattened"):
            capture_rollout_diagnostics(model)
        _assert_unchanged(before, model)
        model.rollout_buffer.full = True
        model.rollout_buffer.generator_ready = True
        before = _snapshot(model)
        with pytest.raises(ValueError, match="full, unflattened"):
            capture_rollout_diagnostics(model)
        _assert_unchanged(before, model)
    finally:
        model.get_env().close()


def test_zero_variance_is_null_and_nonfinite_data_is_rejected() -> None:
    assert _explained_variance([0.0, 0.0], [4.0, 4.0]) is None
    assert _explained_variance([1.0, 2.0], [1.0, 2.0]) == 1.0
    with pytest.raises(ValueError, match="finite"):
        _explained_variance([1.0, np.nan], [1.0, 2.0])


def test_large_critic_targets_report_suppression_of_unchanged_actor_gradient() -> None:
    from stable_baselines3.common.callbacks import BaseCallback

    model = _model(SCALAR_PPO_ALGORITHM_ID, n_envs=1)

    class Capture(BaseCallback):
        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            before = capture_rollout_diagnostics(model)["gradient_attribution"]
            original = model.rollout_buffer.returns.copy()
            try:
                # Synthetic critic mismatch isolates joint clipping: actor data
                # stays identical while reward-value targets grow substantially.
                model.rollout_buffer.returns += 100_000.0
                after = capture_rollout_diagnostics(model)["gradient_attribution"]
            finally:
                model.rollout_buffer.returns[:] = original
            assert (
                after["actor_total_preclip_gradient_norm"]
                == before["actor_total_preclip_gradient_norm"]
            )
            assert (
                after["reward_critic_preclip_gradient_norm"]
                > 1000 * before["reward_critic_preclip_gradient_norm"]
            )
            assert (
                after["implied_common_clip_factor"]
                < before["implied_common_clip_factor"] / 1000
            )

    try:
        model.learn(BETA_DECISIONS_PER_CYCLE, callback=Capture())
    finally:
        model.get_env().close()


def test_known_gaussian_clipping_probability_and_saturated_tanh_are_detected() -> None:
    import math
    import torch
    from stable_baselines3.common.callbacks import BaseCallback

    model = _model(SCALAR_PPO_ALGORITHM_ID, n_envs=1)
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
        model.policy.log_std.zero_()
        for module in model.policy.mlp_extractor.value_net:
            if isinstance(module, torch.nn.Linear):
                module.weight.zero_()
                module.bias.fill_(10.0)

    class Capture(BaseCallback):
        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            sample = capture_rollout_diagnostics(model)["sample"]
            # Exact N(0, 1) tail mass outside [-1, 1], no Monte Carlo estimate.
            expected_tail = math.erfc(1 / math.sqrt(2))
            gaussian = sample["gaussian"]
            assert gaussian[
                "expected_clipped_fraction_all_action_coordinates"
            ] == pytest.approx(expected_tail, abs=1e-7)
            assert gaussian["expected_any_action_clipped_fraction"] == pytest.approx(
                1 - (1 - expected_tail) ** 2, abs=1e-7
            )
            for layer in sample["reward_critic_tanh"]:
                assert layer["fraction_abs_output_ge_0_95"] == 1.0
                assert layer["mean_tanh_derivative"] == 0.0

    try:
        model.learn(BETA_DECISIONS_PER_CYCLE, callback=Capture())
    finally:
        model.get_env().close()

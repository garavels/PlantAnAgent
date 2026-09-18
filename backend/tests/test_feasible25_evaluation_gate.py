"""An unfinished training campaign must not expose the held-out scenarios."""

import pytest
from docs.research.tools import evaluate_feasible25_v1 as evaluation


def test_missing_joint_freeze_blocks_before_any_environment_is_constructed(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(evaluation, "FREEZE_PATH", tmp_path / "absent-freeze.json")

    def forbidden_environment(*args, **kwargs):
        raise AssertionError("Held-out environment was constructed before admission")

    monkeypatch.setattr(evaluation, "BenchmarkEnv", forbidden_environment)
    with pytest.raises(FileNotFoundError):
        evaluation.evaluate(
            kind="coverage",
            seeds=[9930000],
            split="held_out",
            output=tmp_path / "results",
        )
    assert not (tmp_path / "results").exists()


def test_unclipped_actor_output_is_distinguished_from_adapter_input():
    import json
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from docs.research.tools.coverage_ppo_v1 import make_learner
    from docs.research.tools.feasible_benchmark_v1 import BenchmarkEnv
    from plant_backend.rl.v2.training.constrained_ppo import policy_parameter_sha256

    model, _, vector = make_learner(101, fixture_steps=384)
    env = BenchmarkEnv(split="development")
    try:
        observation, _ = env.reset(seed=9920000)
        # Only this disposable fixture model is changed; no saved policy is loaded.
        with torch.no_grad():
            model.policy.action_net.bias.fill_(2.0)
        # SB3 metadata loading has this exact JSON tuple-to-list effect.
        model.research_identity = json.loads(json.dumps(model.research_identity))
        tensor_sha = policy_parameter_sha256(model)
        controller = evaluation.instrumented_ppo_controller(env, model)
        assert policy_parameter_sha256(model) == tensor_sha
        controller.reset()
        controller.set_observation(observation)
        decision = controller.compute_command(
            env.cyclic_market_controller_observation,
            constraints=env.constraints,
            config=env.config,
            commitment_profile=env.episode_config.electrolyser_commitment,
        )
        witness = decision.evidence["inference_witness"]
        assert witness["unclipped_deterministic_action"] == [2.0] * 4
        assert witness["sb3_clipped_action"] == [1.0] * 4
        assert decision.evidence["raw_residual"] == [1.0] * 4
        assert witness["upstream_action_clipped"] is True
        assert witness["adapter_input_verified"] is True
        assert env.decisions_taken == 0
        model.research_identity["residual_fraction"] = 0.5
        with pytest.raises(ValueError, match="different canonical task identity"):
            evaluation.instrumented_ppo_controller(env, model)
    finally:
        env.close()
        vector.close()

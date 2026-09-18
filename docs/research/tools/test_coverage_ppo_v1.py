"""Focused interoperability checks using training seeds only; no held-out access."""

import json

import numpy as np
import pytest

from docs.research.tools import coverage_ppo_v1 as learner
from docs.research.tools.feasible_benchmark_v1 import BenchmarkEnv
from plant_backend.rl.v2.training.custody_json import canonical_bytes


def test_zero_residual_preserves_day_commands_physics_rewards_and_costs():
    baseline = BenchmarkEnv(split="train")
    base = BenchmarkEnv(split="train")
    wrapped = learner.CoverageResidualEnv(base)
    controller = learner.make_coverage_prior()
    try:
        baseline.reset(seed=learner.TRAIN_BASE)
        vector, _ = wrapped.reset(seed=learner.TRAIN_BASE)
        assert vector.shape == (139,) and vector.dtype == np.float32
        for _ in range(96):
            original = controller.compute_command(
                baseline.cyclic_market_controller_observation,
                constraints=baseline.constraints,
                config=baseline.config,
                commitment_profile=baseline.episode_config.electrolyser_commitment,
            )
            a = baseline.step_process_setpoints(
                original.command, controller_invalid=original.invalid
            )
            b = wrapped.step(np.zeros(4, dtype=np.float32))
            assert wrapped.last_decision.command.to_dict() == original.command.to_dict()
            assert np.array_equal(a[0], b[0][:135])
            assert a[1:4] == b[1:4]
            assert (
                a[4]["cycle_constraint_transition"]
                == b[4]["cycle_constraint_transition"]
            )
            assert canonical_bytes(
                baseline.simulator.state.to_dict()
            ) == canonical_bytes(base.simulator.state.to_dict())
        assert baseline.cycle_settlements == base.cycle_settlements
    finally:
        baseline.close()
        wrapped.close()


@pytest.mark.parametrize(
    "bad", [np.array([np.nan, 0, 0, 0]), np.ones(3), np.array([True] * 4)]
)
def test_malformed_action_rejected_before_physics(bad):
    wrapped = learner.CoverageResidualEnv(BenchmarkEnv(split="train"))
    try:
        wrapped.reset(seed=learner.TRAIN_BASE)
        with pytest.raises(ValueError):
            wrapped.step(bad)
        assert wrapped.base.decisions_taken == 0
    finally:
        wrapped.close()


def test_bound_projection_is_recorded_and_keeps_zero_or_start_commands():
    wrapped = learner.CoverageResidualEnv(BenchmarkEnv(split="train"))
    try:
        wrapped.reset(seed=learner.TRAIN_BASE)
        original = wrapped.cached
        decision = learner.residual_command(
            wrapped.base,
            original,
            np.array([2.0, -2.0, 2.0, -2.0]),
            wrapped.identity_sha256,
        )
        assert decision.invalid
        assert decision.evidence["action_clipped"]
        assert decision.evidence["bounded_residual"] == [1.0, -1.0, 1.0, -1.0]
        floor = (
            wrapped.base.episode_config.electrolyser_commitment.min_stable_load_frac
            * wrapped.base.config.electrolyser.rated_kw
        )
        assert original.command.targets.electrolyser_target_kw <= floor
        assert (
            decision.command.targets.electrolyser_target_kw
            == original.command.targets.electrolyser_target_kw
        )
        assert wrapped.base.decisions_taken == 0
    finally:
        wrapped.close()


def test_identity_binds_new_task_and_matches_across_evaluation_splits():
    training = BenchmarkEnv(split="train")
    development = BenchmarkEnv(split="development")
    try:
        a, b = learner.interface_identity(training), learner.interface_identity(
            development
        )
        assert a == b
        assert a["schema"] == learner.SCHEMA
        assert (
            a["base_checkpoint_contract"]["benchmark_contract_sha256"]
            == training.benchmark_contract["contract_sha256"]
        )
        assert a["electricity_model"]["mode"] == "dedicated_pv_no_export"
        json.dumps(a, allow_nan=False)
    finally:
        training.close()
        development.close()


def test_real_optimizer_fixture_records_costs_and_reopens_checkpoint(tmp_path):
    import torch
    from plant_backend.rl.v2.training.constrained_ppo import (
        ConstrainedPPO,
        policy_parameter_sha256,
    )
    from plant_backend.rl.v2.training.research_effort_batch import preflight_checkpoint

    torch.set_num_threads(1)
    model, allocator, vector = learner.make_learner(101, fixture_steps=96)
    before = learner.actor_parameter_sha256(model)
    try:
        model.learn(total_timesteps=384)
        assert model.num_timesteps == 384
        assert allocator.decisions_consumed_by_environment == (96, 96, 96, 96)
        assert len(model.rollout_ledgers) == 1
        assert model.rollout_ledgers[0].optimizer_steps == 1
        assert learner.actor_parameter_sha256(model) != before
        snapshot = allocator.snapshot().to_dict()
        model.pin_external_episode_state(snapshot)
        checkpoint = tmp_path / "fixture-only.zip"
        model.save(checkpoint)
        state = model.embedded_constraint_state
        preflight_checkpoint(checkpoint, model.research_identity, state)
        reopened = ConstrainedPPO.load(checkpoint, device="cpu")
        assert policy_parameter_sha256(reopened) == policy_parameter_sha256(model)
        assert reopened.loaded_constraint_state_attestation().num_timesteps == 384
        assert reopened.embedded_constraint_state == state
        assert state["payload"]["pending_rollout"] is False
    finally:
        vector.close()


def test_held_out_gate_requires_all_three_frozen_endpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(learner, "_require_plan", lambda root: {})
    with pytest.raises(OSError):
        learner.require_frozen_endpoints(tmp_path)

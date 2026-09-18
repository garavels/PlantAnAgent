"""Meaningful interface, commitment, optimizer and checkpoint fixtures; train only."""
from dataclasses import replace
import io
from types import SimpleNamespace

import numpy as np
import pytest

from docs.research.tools import coverage_ppo_v1 as legacy
from docs.research.tools.authority_ppo_v2 import benchmark, policy, train
from plant_backend.rl.v2.training.custody_json import canonical_bytes


@pytest.mark.parametrize("arm", policy.ARMS)
def test_zero_action_preserves_entire_training_week_exactly(arm):
    baseline = benchmark.BenchmarkEnv(split="train")
    wrapped = policy.AuthorityResidualEnv(benchmark.BenchmarkEnv(split="train"), arm)
    controller = policy.make_coverage_prior()
    try:
        baseline.reset(seed=train.TRAIN_BASE)
        vector, _ = wrapped.reset(seed=train.TRAIN_BASE)
        assert vector.dtype == np.float32 and vector.shape == (139,)
        for index in range(672):
            original = controller.compute_command(
                baseline.cyclic_market_controller_observation,
                constraints=baseline.constraints, config=baseline.config,
                commitment_profile=baseline.episode_config.electrolyser_commitment,
            )
            a = baseline.step_process_setpoints(original.command, controller_invalid=original.invalid)
            b = wrapped.step(np.zeros(4, dtype=np.float32))
            assert wrapped.last_decision.command.to_dict() == original.command.to_dict()
            assert np.array_equal(a[0], b[0][:135])
            assert a[1:4] == b[1:4]
            assert a[4]["cycle_constraint_transition"] == b[4]["cycle_constraint_transition"]
            assert canonical_bytes(baseline.simulator.state.to_dict()) == canonical_bytes(wrapped.base.simulator.state.to_dict())
        assert a[2] or a[3]
        assert baseline.cycle_settlements == wrapped.base.cycle_settlements
    finally:
        baseline.close()
        wrapped.close()


@pytest.mark.parametrize("arm", policy.ARMS)
@pytest.mark.parametrize("bad", [np.array([np.nan, 0, 0, 0]), np.ones(3), np.array([True] * 4), np.array([1j] * 4)])
def test_malformed_action_cannot_advance_physics(arm, bad):
    wrapped = policy.AuthorityResidualEnv(benchmark.BenchmarkEnv(split="train"), arm)
    try:
        wrapped.reset(seed=train.TRAIN_BASE)
        with pytest.raises(ValueError):
            wrapped.step(bad)
        assert wrapped.base.decisions_taken == 0
    finally:
        wrapped.close()


@pytest.mark.parametrize("mode,requested,min_up,min_down,offline,expected,reason", [
    ("off", 0, 0, 0, False, 0, "zero"),
    ("off", 300, 0, 0, False, 0, "nearest_feasible_off"),
    ("off", 301, 0, 0, False, 600, "nearest_feasible_floor"),
    ("off", 900, 0, 0, False, 900, "within_feasible_range"),
    ("off", 900, 0, 100, False, 0, "minimum_down"),
    ("cooldown", 2000, 0, 0, False, 0, "cooldown"),
    ("cold_starting", 0, 0, 0, False, 600, "mandatory_startup"),
    ("hot_starting", 2500, 0, 0, False, 600, "mandatory_startup"),
    ("running", 0, 100, 0, False, 1200, "mandatory_minimum_up"),
    ("running", 600, 0, 0, False, 0, "nearest_feasible_off"),
    ("running", 601, 0, 0, False, 1200, "nearest_feasible_floor"),
    ("running", 9999, 0, 0, False, 3000, "within_feasible_range"),
    ("running", 2500, 100, 0, True, 0, "offline"),
    ("hot_standby", 800, 0, 0, False, 1200, "nearest_feasible_floor"),
])
def test_public_commitment_projection(mode, requested, min_up, min_down, offline, expected, reason):
    assert policy.project_electrolyser(requested, mode=mode, start_kw=600, stable_kw=1200, limit_kw=3000,
                                      min_up_remaining_s=min_up, min_down_remaining_s=min_down, offline=offline) == (expected, reason)


def test_mandatory_floor_never_exceeds_operator_authority():
    for mode, min_up in (("cold_starting", 0), ("running", 100)):
        value, reason = policy.project_electrolyser(1000, mode=mode, start_kw=600, stable_kw=1200,
                                                   limit_kw=500, min_up_remaining_s=min_up)
        assert value == 0 and reason == "commitment_authority_conflict"


def test_broad_activates_prior_zero_and_records_requested_vs_projected():
    wrapped = policy.AuthorityResidualEnv(benchmark.BenchmarkEnv(split="train"), "broad")
    try:
        wrapped.reset(seed=train.TRAIN_BASE)
        prior = wrapped.cached
        assert prior.command.targets.electrolyser_target_kw == 0
        result = policy.residual_command(wrapped.base, prior, np.array([2, 1, 1, -2]), wrapped.identity_sha256, "broad")
        assert result.invalid and result.evidence["action_clipped"]
        assert result.evidence["prior_zero_target_enabled"][0]
        assert result.command.targets.electrolyser_target_kw == 900
        assert result.evidence["bounded_residual"] == [1, 1, 1, -1]
        assert result.evidence["requested_targets_kw"][0] == 900
        assert wrapped.base.decisions_taken == 0
    finally:
        wrapped.close()


def test_narrow_command_and_physics_match_legacy_for_nonzero_actions():
    wrapped = policy.AuthorityResidualEnv(benchmark.BenchmarkEnv(split="train"), "narrow")
    rng = np.random.default_rng(17)
    try:
        wrapped.reset(seed=train.TRAIN_BASE)
        for _ in range(96):
            action = rng.uniform(-1, 1, 4)
            old = legacy.residual_command(wrapped.base, wrapped.cached, action, wrapped.identity_sha256)
            new = policy.residual_command(wrapped.base, wrapped.cached, action, wrapped.identity_sha256, "narrow")
            assert old.command.to_dict() == new.command.to_dict()
            assert old.invalid == new.invalid
            wrapped.step(action)
    finally:
        wrapped.close()


def test_broad_cannot_cancel_actual_started_sequence():
    wrapped = policy.AuthorityResidualEnv(benchmark.BenchmarkEnv(split="train"), "broad")
    seen = set()
    try:
        wrapped.reset(seed=train.TRAIN_BASE)
        for _ in range(96):
            snap = wrapped.base.cyclic_market_controller_observation.plant.state.electrolyser_commitment
            if snap.mode.value in {"cold_starting", "hot_starting"} or snap.min_up_remaining_s > 0:
                result = policy.residual_command(wrapped.base, wrapped.cached, np.array([-1, 0, 0, 0]), wrapped.identity_sha256, "broad")
                if snap.mode.value in {"cold_starting", "hot_starting"}:
                    seen.add("startup")
                    assert result.command.targets.electrolyser_target_kw == 600
                else:
                    seen.add("minimum_up")
                    assert result.command.targets.electrolyser_target_kw >= 1200
                wrapped.step(np.array([-1, 0, 0, 0]))
            else:
                wrapped.step(np.zeros(4))
        assert seen == {"startup", "minimum_up"}
    finally:
        wrapped.close()


@pytest.mark.parametrize("arm", policy.ARMS)
def test_identity_is_split_invariant_and_binds_arm(arm):
    a = benchmark.BenchmarkEnv(split="train")
    b = benchmark.BenchmarkEnv(split="development")
    try:
        assert policy.interface_identity(a, arm) == policy.interface_identity(b, arm)
        assert policy.interface_identity(a, arm)["arm"] == arm
        assert policy.interface_identity(a, arm)["observation_size"] == 139
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize("arm", policy.ARMS)
def test_real_optimizer_changes_actor_mean_and_reopens_durable_checkpoint(tmp_path, arm):
    import torch
    from plant_backend.rl.v2.training.constrained_ppo import ConstrainedPPO, policy_parameter_sha256
    torch.set_num_threads(1)
    model, allocator, vector = train.make_learner(101, arm, fixture_steps=96)
    before = train.actor_mean_parameter_sha256(model)
    try:
        model.learn(total_timesteps=384)
        assert model.num_timesteps == 384
        assert allocator.decisions_consumed_by_environment == (96,) * 4
        assert len(model.rollout_ledgers) == 1
        assert model.rollout_ledgers[0].optimizer_steps == 1
        assert train.actor_mean_parameter_sha256(model) != before
        checkpoint = tmp_path / "fixture-only.zip"
        saved = train._save_checkpoint(model, allocator, checkpoint)
        reopened = ConstrainedPPO.load(checkpoint, device="cpu")
        assert policy_parameter_sha256(reopened) == policy_parameter_sha256(model)
        assert train.actor_mean_parameter_sha256(reopened) == train.actor_mean_parameter_sha256(model)
        assert reopened.loaded_constraint_state_attestation().num_timesteps == 384
        assert reopened.embedded_constraint_state == saved["constraint_state"]
        assert saved["constraint_state"]["payload"]["pending_rollout"] is False
        actions, _ = reopened.predict(model._last_obs.copy(), deterministic=True)
        assert np.any(actions != 0)
        env = benchmark.BenchmarkEnv(split="train")
        try:
            observation, _ = env.reset(seed=train.TRAIN_BASE)
            controller = policy.AuthorityPPOController(env, reopened)
            controller.set_observation(observation)
            decision = controller.compute_command(
                env.cyclic_market_controller_observation, constraints=env.constraints,
                config=env.config, commitment_profile=env.episode_config.electrolyser_commitment,
            )
            mean = np.asarray(decision.evidence["deterministic_raw_action_mean"])
            assert mean.shape == (4,) and np.isfinite(mean).all()
            assert np.allclose(np.clip(mean, -1, 1), decision.evidence["bounded_residual"], atol=0, rtol=0)
            env.step_process_setpoints(decision.command, controller_invalid=decision.invalid)
        finally:
            env.close()
        with pytest.raises(FileExistsError):
            train._save_checkpoint(model, allocator, checkpoint)
    finally:
        vector.close()


def test_paired_arms_start_with_same_fresh_tensors():
    from plant_backend.rl.v2.training.constrained_ppo import policy_parameter_sha256
    models = []
    try:
        for arm in policy.ARMS:
            models.append(train.make_learner(211, arm, fixture_steps=1))
        assert policy_parameter_sha256(models[0][0]) == policy_parameter_sha256(models[1][0])
        assert train.actor_mean_parameter_sha256(models[0][0]) == train.actor_mean_parameter_sha256(models[1][0])
    finally:
        for _, _, vector in models:
            vector.close()


def test_source_freeze_excludes_transient_eval_tests_and_helpers():
    paths = {row["path"] for row in train.source_records()}
    prefix = str(train.ROOT / "docs/research/tools/authority_ppo_v2") + "/"
    assert {path.removeprefix(prefix) for path in paths if path.startswith(prefix)} == {"benchmark.py", "policy.py", "train.py"}
    assert str(benchmark.CONTRACT_PATH) in paths


def test_held_out_gate_cannot_open_without_all_six_endpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "require_plan", lambda root: {})
    with pytest.raises(OSError):
        train.require_frozen_endpoints(tmp_path)


def test_stale_prior_is_rejected_before_next_physics_step():
    wrapped = policy.AuthorityResidualEnv(benchmark.BenchmarkEnv(split="train"), "broad")
    try:
        wrapped.reset(seed=train.TRAIN_BASE)
        prior = wrapped.cached
        wrapped.step(np.zeros(4))
        with pytest.raises(ValueError, match="stale"):
            policy.residual_command(wrapped.base, prior, np.ones(4), wrapped.identity_sha256, "broad")
        assert wrapped.base.decisions_taken == 1
    finally:
        wrapped.close()


def test_intermediate_rollout_start_save_allows_uninterrupted_next_rollout(tmp_path):
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from plant_backend.rl.v2.training.constrained_ppo import ConstrainedPPO, policy_parameter_sha256
    torch.set_num_threads(1)
    model, allocator, vector = train.make_learner(307, "broad", fixture_steps=96)
    saved_steps = []

    class SaveAtLedgerCut(BaseCallback):
        def _on_rollout_start(self):
            if self.num_timesteps:
                train._save_checkpoint(self.model, allocator, tmp_path / "intermediate.zip")
                saved_steps.append(self.num_timesteps)

        def _on_step(self):
            return True

    try:
        model.learn(total_timesteps=768, callback=SaveAtLedgerCut())
        assert saved_steps == [384]
        assert model.num_timesteps == 768 and len(model.rollout_ledgers) == 2
        assert allocator.decisions_consumed_by_environment == (192,) * 4
        assert model.episode_decision_indices == (192,) * 4
        intermediate = ConstrainedPPO.load(tmp_path / "intermediate.zip", device="cpu")
        assert intermediate.num_timesteps == 384
        final_path = tmp_path / "after-next-rollout.zip"
        train._save_checkpoint(model, allocator, final_path)
        reopened = ConstrainedPPO.load(final_path, device="cpu")
        assert policy_parameter_sha256(reopened) == policy_parameter_sha256(model)
        assert reopened.num_timesteps == 768
    finally:
        vector.close()

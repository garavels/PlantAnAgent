"""Genuine production-path smokes for controller-neutral constrained beta RL."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from plant_backend.rl import checkpoints
from plant_backend.rl.policies import SB3Policy, load_policy
from plant_backend.rl.training.config import PPOConfig
from plant_backend.rl.v2 import beta_spaces
from plant_backend.rl.v2.beta_config import CyclicMarketTaskConfigV1
from plant_backend.rl.v2.beta_controller_benchmark import load_beta_benchmark_policy
from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta
from plant_backend.rl.v2.policy_setpoint_controller import (
    NormalizedSetpointPolicyControllerV1,
    policy_contract_for_environment,
)
from plant_backend.rl.v2.training.beta_checkpoint import (
    BetaRLCheckpointEvidenceV1,
    parse_beta_checkpoint_evidence,
)
from plant_backend.rl.v2.training.config import (
    BETA_I_LAGRANGIAN_ALGORITHM_ID,
    BETA_PI_LAGRANGIAN_ALGORITHM_ID,
    BETA_SCALAR_PPO_ALGORITHM_ID,
    BetaRLTrainingRunV1,
    ControllerNeutralTrainingConfigV1,
)
from plant_backend.rl.v2.training.constrained_ppo import ConstrainedPPO
from plant_backend.rl.v2.training.ppo import (
    BETA_CHECKPOINT_EXTRA_KEY,
    admit_resume,
    make_env,
    restore_beta_episode_allocator,
)
from plant_backend.rl.v2.training.train import train


def _fixture_config(
    tmp_path: Path,
    *,
    algorithm: str,
    output_name: str,
    n_steps: int = 96,
    total_timesteps: int = 384,
    episode_seed_base: int = 9_600_000,
) -> ControllerNeutralTrainingConfigV1:
    return ControllerNeutralTrainingConfigV1(
        environment_id=beta_spaces.ENVIRONMENT_ID,
        seed=101,
        total_timesteps=total_timesteps,
        n_envs=4,
        checkpoint_every=0,
        output_dir=str(tmp_path / output_name),
        beta_task=CyclicMarketTaskConfigV1(),
        beta_rl=BetaRLTrainingRunV1(
            algorithm=algorithm,
            episode_seed_base=episode_seed_base,
        ),
        ppo=replace(
            PPOConfig(),
            n_steps=n_steps,
            batch_size=4 * n_steps,
            n_epochs=1,
        ),
        notes="fixture-domain production integration; no research outcome",
    ).validate()


def _metadata_and_evidence(
    checkpoint: Path,
) -> tuple[checkpoints.CheckpointMetadata, BetaRLCheckpointEvidenceV1]:
    metadata = checkpoints.load_metadata(checkpoint, check=False)
    evidence = parse_beta_checkpoint_evidence(metadata.extra[BETA_CHECKPOINT_EXTRA_KEY])
    return metadata, evidence


@pytest.mark.parametrize(
    "algorithm",
    (
        BETA_PI_LAGRANGIAN_ALGORITHM_ID,
        BETA_I_LAGRANGIAN_ALGORITHM_ID,
        BETA_SCALAR_PPO_ALGORITHM_ID,
    ),
)
def test_real_beta_family_trains_saves_loads_and_crosses_common_physical_door(
    tmp_path: Path,
    algorithm: str,
) -> None:
    pytest.importorskip("stable_baselines3")
    config = _fixture_config(
        tmp_path,
        algorithm=algorithm,
        output_name=algorithm,
    )

    checkpoint = train(config)
    metadata, evidence = _metadata_and_evidence(checkpoint)

    assert metadata.algorithm == algorithm
    assert metadata.total_timesteps == 384
    assert metadata.seed == 101
    assert evidence.algorithm_id == algorithm
    assert evidence.total_timesteps == 384
    assert evidence.eligible_for_registered_evaluation is False
    assert len(evidence.runtime_provenance.sha256) == 64

    environment = make_env(config)
    assert type(environment) is PlantCyclicMarketEnvV2Beta
    try:
        contract = policy_contract_for_environment(environment)
        loaded = load_policy(
            checkpoint,
            contract=contract,
            environment=environment,
            expected_beta_runtime_provenance=evidence.runtime_provenance,
        )
        observation, _ = environment.reset(seed=9_900_100)
        assert observation.shape == (135,)
        action = loaded.policy.act(observation)
        repeated_action = loaded.policy.act(observation.copy())
        assert action.shape == (beta_spaces.ACTION_SIZE,)
        assert action.dtype == np.float32
        assert np.all(np.isfinite(action))
        np.testing.assert_array_equal(action, repeated_action)
        assert type(loaded.policy) is SB3Policy
        loaded_model = cast(SB3Policy, loaded.policy)._model
        assert type(loaded_model) is ConstrainedPPO
        assert loaded_model.constraint_settings.algorithm_id == algorithm

        controller = NormalizedSetpointPolicyControllerV1.from_environment(
            loaded,
            environment,
        )
        controller.reset()
        decision = controller.compute_command(
            environment.cyclic_market_controller_observation,
            constraints=environment.constraints,
            config=environment.config,
            commitment_profile=environment.episode_config.electrolyser_commitment,
        )
        _, reward, terminated, truncated, info = environment.step_process_setpoints(
            decision.command
        )
        assert np.isfinite(reward)
        assert terminated is False
        assert truncated is False
        assert info["process_setpoint_execution"]["command"]["issuer"]["id"] == (
            controller.id
        )
    finally:
        environment.close()

    with pytest.raises(
        checkpoints.CheckpointError,
        match="registered|not admitted",
    ):
        load_beta_benchmark_policy(checkpoint)

    # A 96-decision-per-worker cut is valid for inference but cannot recreate
    # the missing physical/action history required for exact continuation.
    resume_config = replace(
        config,
        output_dir=str(tmp_path / f"{algorithm}-invalid-resume"),
        resume_from=str(checkpoint),
    ).validate()
    admitted = admit_resume(
        resume_config,
        runtime_provenance=evidence.runtime_provenance,
    )
    assert admitted is not None
    with pytest.raises(ValueError, match="exact episode-boundary"):
        restore_beta_episode_allocator(resume_config, admitted)


def test_real_scalar_beta_checkpoint_resumes_exactly_at_episode_boundary(
    tmp_path: Path,
) -> None:
    pytest.importorskip("stable_baselines3")
    parent_config = _fixture_config(
        tmp_path,
        algorithm=BETA_SCALAR_PPO_ALGORITHM_ID,
        output_name="scalar-boundary-parent",
        n_steps=672,
        total_timesteps=2_688,
        episode_seed_base=9_700_000,
    )
    parent_checkpoint = train(parent_config)
    parent_metadata, parent_evidence = _metadata_and_evidence(parent_checkpoint)

    child_config = replace(
        parent_config,
        output_dir=str(tmp_path / "scalar-boundary-child"),
        resume_from=str(parent_checkpoint),
        notes="exact fixture resume integration; no research outcome",
    ).validate()
    child_checkpoint = train(child_config)
    child_metadata, child_evidence = _metadata_and_evidence(child_checkpoint)

    assert parent_metadata.total_timesteps == 2_688
    assert child_metadata.total_timesteps == 5_376
    assert child_evidence.resumed_from_parent is True
    assert child_evidence.resume_parent is not None
    assert child_evidence.resume_parent["policy_artifact_sha256"] == (
        parent_metadata.policy_artifact_sha256
    )
    assert child_evidence.resume_parent["checkpoint_evidence_sha256"] == (
        parent_evidence.sha256
    )
    parent_entries = parent_evidence.allocator_snapshot["ledger"]["entries"]
    assert len(parent_entries) == 8
    assert (
        child_evidence.allocator_snapshot["ledger"]["entries"][: len(parent_entries)]
        == parent_entries
    )
    assert child_evidence.eligible_for_registered_evaluation is False

    environment = make_env(child_config)
    try:
        loaded = load_policy(
            child_checkpoint,
            contract=policy_contract_for_environment(environment),
            environment=environment,
            expected_beta_runtime_provenance=child_evidence.runtime_provenance,
        )
        assert isinstance(loaded.policy._model, ConstrainedPPO)
        observation, _ = environment.reset(seed=9_900_200)
        action = loaded.policy.act(observation)
        assert action.shape == (4,)
        assert np.all(np.isfinite(action))
    finally:
        environment.close()

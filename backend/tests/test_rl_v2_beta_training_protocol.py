"""Strict, outcome-free configuration gates for the beta RL campaign."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from plant_backend.rl.training.config import PPOConfig
from plant_backend.rl.v2 import beta_spaces, economic_spaces
from plant_backend.rl.v2.beta_config import CyclicMarketTaskConfigV1
from plant_backend.rl.v2.training.config import (
    BETA_I_LAGRANGIAN_ALGORITHM_ID,
    BETA_PI_LAGRANGIAN_ALGORITHM_ID,
    BETA_REGISTERED_RUN_KIND,
    BETA_SCALAR_PPO_ALGORITHM_ID,
    BetaRLTrainingRunV1,
    ControllerNeutralTrainingConfigV1,
)


def _beta_config(
    tmp_path: Path,
    *,
    run: BetaRLTrainingRunV1 | None = None,
) -> ControllerNeutralTrainingConfigV1:
    return ControllerNeutralTrainingConfigV1(
        environment_id=beta_spaces.ENVIRONMENT_ID,
        seed=101,
        total_timesteps=8,
        n_envs=4,
        checkpoint_every=0,
        output_dir=str(tmp_path / "fixture-run"),
        beta_task=CyclicMarketTaskConfigV1(),
        beta_rl=run or BetaRLTrainingRunV1(),
        ppo=replace(PPOConfig(), n_steps=2, batch_size=8, n_epochs=1),
    )


@pytest.mark.parametrize(
    "algorithm",
    [
        BETA_PI_LAGRANGIAN_ALGORITHM_ID,
        BETA_I_LAGRANGIAN_ALGORITHM_ID,
        BETA_SCALAR_PPO_ALGORITHM_ID,
    ],
)
def test_fixture_beta_rl_contract_round_trips_all_families(
    tmp_path: Path,
    algorithm: str,
) -> None:
    config = _beta_config(
        tmp_path,
        run=BetaRLTrainingRunV1(algorithm=algorithm, episode_seed_base=9_100_000),
    ).validate()

    assert ControllerNeutralTrainingConfigV1.from_dict(config.to_dict()) == config
    assert config.beta_rl is not None
    assert config.beta_rl.algorithm == algorithm


def test_beta_rl_contract_rejects_cross_task_seed_and_compute_leakage(
    tmp_path: Path,
) -> None:
    fixture = _beta_config(tmp_path)

    with pytest.raises(ValueError, match="only valid for the cyclic-market beta"):
        replace(
            fixture,
            environment_id=economic_spaces.ENVIRONMENT_ID,
            beta_task=None,
        ).validate()
    with pytest.raises(ValueError, match="9,000,000"):
        replace(
            fixture,
            beta_rl=replace(fixture.beta_rl, episode_seed_base=2_000_000),  # type: ignore[arg-type]
        ).validate()
    with pytest.raises(ValueError, match="at most 8,192"):
        replace(fixture, total_timesteps=8_193).validate()
    with pytest.raises(ValueError, match="five-seed"):
        replace(fixture, seed=102).validate()
    with pytest.raises(ValueError, match="exactly four"):
        replace(
            fixture,
            n_envs=2,
            ppo=replace(fixture.ppo, batch_size=4),
        ).validate()


def test_registered_beta_rl_contract_is_exact_and_non_resumable(
    tmp_path: Path,
) -> None:
    del tmp_path
    registered = ControllerNeutralTrainingConfigV1(
        environment_id=beta_spaces.ENVIRONMENT_ID,
        seed=503,
        total_timesteps=1_048_576,
        n_envs=4,
        checkpoint_every=0,
        output_dir="/private/tmp/plant-beta-pi-503",
        beta_task=CyclicMarketTaskConfigV1(),
        beta_rl=BetaRLTrainingRunV1(
            algorithm=BETA_PI_LAGRANGIAN_ALGORITHM_ID,
            run_kind=BETA_REGISTERED_RUN_KIND,
            episode_seed_base=1_000_000,
        ),
    ).validate()

    assert registered.total_timesteps == 1_048_576
    with pytest.raises(ValueError, match="forbids resume_from"):
        replace(registered, resume_from="/private/tmp/parent").validate()
    with pytest.raises(ValueError, match="frozen defaults"):
        replace(registered, ppo=replace(registered.ppo, gamma=0.99)).validate()
    with pytest.raises(ValueError, match="must be absolute"):
        replace(registered, output_dir="relative/run").validate()
    repository_output = Path(__file__).resolve().parents[2] / "forbidden-run"
    with pytest.raises(ValueError, match="external to the repository"):
        replace(registered, output_dir=str(repository_output)).validate()


def test_beta_rl_contract_rejects_unknown_nested_fields() -> None:
    with pytest.raises(ValueError, match="unknown beta RL training run"):
        BetaRLTrainingRunV1.from_dict({"algorithm": "unknown", "seed_fishing": True})
    with pytest.raises(ValueError, match="three frozen core families"):
        BetaRLTrainingRunV1.from_dict({"algorithm": "PPO-ish"})

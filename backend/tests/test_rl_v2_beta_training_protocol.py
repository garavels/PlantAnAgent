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
from plant_backend.rl.v2.training.episode_allocator import (
    EpisodeAllocatorModeV1,
    EpisodeSeedAllocatorV1,
    EpisodeSeedSplitV1,
    find_seed_allocator,
)
from plant_backend.rl.v2.training.ppo import (
    build_episode_allocator,
    build_vec_env,
    plan,
)
from plant_backend.rl.v2.training.train import build_parser, resolve


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


def _registered_config(output: Path) -> ControllerNeutralTrainingConfigV1:
    return ControllerNeutralTrainingConfigV1(
        environment_id=beta_spaces.ENVIRONMENT_ID,
        seed=503,
        total_timesteps=1_048_576,
        n_envs=4,
        checkpoint_every=0,
        output_dir=str(output),
        beta_task=CyclicMarketTaskConfigV1(),
        beta_rl=BetaRLTrainingRunV1(
            algorithm=BETA_PI_LAGRANGIAN_ALGORITHM_ID,
            run_kind=BETA_REGISTERED_RUN_KIND,
            episode_seed_base=1_000_000,
        ),
    ).validate()


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
    registered = _registered_config(tmp_path / "plant-beta-pi-503")

    assert registered.total_timesteps == 1_048_576
    described = plan(registered)
    assert described["environment_reset"] is False
    assert described["optimizer_seed"] == 503
    assert described["first_episode_seed"] is None
    assert described["first_episode_manifest_sha256"] is None
    assert described["episode_allocator"] is None
    with pytest.raises(ValueError, match="forbids resume_from"):
        replace(registered, resume_from="/private/tmp/parent").validate()
    with pytest.raises(ValueError, match="frozen defaults"):
        replace(registered, ppo=replace(registered.ppo, gamma=0.99)).validate()
    with pytest.raises(ValueError, match="must be absolute"):
        replace(registered, output_dir="relative/run").validate()
    repository_output = Path(__file__).resolve().parents[2] / "forbidden-run"
    with pytest.raises(ValueError, match="external to the repository"):
        replace(registered, output_dir=str(repository_output)).validate()


def test_fixture_output_cannot_contaminate_its_own_source_provenance(
    tmp_path: Path,
) -> None:
    fixture = _beta_config(tmp_path)
    repository = Path(__file__).resolve().parents[2]

    with pytest.raises(ValueError, match="ignored runs/ artifact root"):
        replace(
            fixture,
            output_dir=str(repository / "unignored-fixture-run"),
        ).validate()

    admitted = replace(
        fixture,
        output_dir=str(repository / "runs" / "fixture-output-contract-test"),
    ).validate()
    assert Path(admitted.output_dir).is_relative_to(repository / "runs")


def test_registered_cli_refuses_identity_overrides_but_allows_destination(
    tmp_path: Path,
) -> None:
    config_path = _registered_config(tmp_path / "registered").write(
        tmp_path / "registered.json"
    )
    parser = build_parser()

    for override in (
        ["--seed", "101"],
        ["--timesteps", "8192"],
        ["--resume", str(tmp_path / "parent")],
        ["--environment", beta_spaces.ENVIRONMENT_ID],
    ):
        with pytest.raises(ValueError, match="only an output override"):
            resolve(parser.parse_args(["--config", str(config_path), *override]))

    destination = tmp_path / "registered-override"
    resolved = resolve(
        parser.parse_args(["--config", str(config_path), "--output", str(destination)])
    )
    assert resolved.output_dir == str(destination)


def test_fixture_cli_cannot_relabel_beta_rl_as_an_alpha_task(tmp_path: Path) -> None:
    config_path = _beta_config(tmp_path).write(tmp_path / "fixture.json")

    with pytest.raises(ValueError, match="cannot be relabelled"):
        resolve(
            build_parser().parse_args(
                [
                    "--config",
                    str(config_path),
                    "--environment",
                    economic_spaces.ENVIRONMENT_ID,
                ]
            )
        )


def test_beta_rl_contract_rejects_unknown_nested_fields() -> None:
    with pytest.raises(ValueError, match="unknown beta RL training run"):
        BetaRLTrainingRunV1.from_dict({"algorithm": "unknown", "seed_fishing": True})
    with pytest.raises(ValueError, match="three frozen core families"):
        BetaRLTrainingRunV1.from_dict({"algorithm": "PPO-ish"})


def test_fixture_vector_environment_uses_allocator_owned_episode_seeds(
    tmp_path: Path,
) -> None:
    config = _beta_config(
        tmp_path,
        run=BetaRLTrainingRunV1(episode_seed_base=9_100_000),
    ).validate()
    vector = build_vec_env(config)
    try:
        observations = vector.reset()
        allocator = find_seed_allocator(vector)

        assert observations.shape == (4, beta_spaces.OBSERVATION_SIZE)
        assert allocator.mode is EpisodeAllocatorModeV1.FIXTURE
        assert allocator.split is EpisodeSeedSplitV1.FIXTURE
        assert [entry.episode_seed for entry in allocator.ledger] == [
            9_100_000,
            9_100_001,
            9_100_002,
            9_100_003,
        ]
        assert all(entry.ignored_framework_seed is None for entry in allocator.ledger)
        assert config.seed not in {entry.episode_seed for entry in allocator.ledger}
    finally:
        vector.close()


def test_fixture_dry_plan_opens_only_the_allocator_fixture_domain(
    tmp_path: Path,
) -> None:
    config = _beta_config(
        tmp_path,
        run=BetaRLTrainingRunV1(episode_seed_base=9_100_000),
    ).validate()

    described = plan(config)

    assert described["environment_reset"] is True
    assert described["optimizer_seed"] == 101
    assert described["first_episode_seed"] == 9_100_000
    assert described["first_episode_manifest_sha256"]
    assert described["episode_allocator"]["allocator_identity"]["base_seed"] == (
        9_100_000
    )
    assert [
        entry["episode_seed"]
        for entry in described["episode_allocator"]["ledger"]["entries"]
    ] == [9_100_000, 9_100_001, 9_100_002, 9_100_003]


def test_vector_environment_rejects_foreign_or_unadmitted_custody(
    tmp_path: Path,
) -> None:
    config = _beta_config(
        tmp_path,
        run=BetaRLTrainingRunV1(episode_seed_base=9_100_000),
    ).validate()
    foreign = EpisodeSeedAllocatorV1.fixture(
        base_seed=9_200_000,
        episode_decisions=config.episode.decision_steps(config.plant),
    )

    with pytest.raises(ValueError, match="base seed differs"):
        build_vec_env(config, episode_allocator=foreign)
    with pytest.raises(ValueError, match="checkpoint-admitted episode custody"):
        build_episode_allocator(
            replace(config, resume_from=str(tmp_path / "parent")).validate()
        )

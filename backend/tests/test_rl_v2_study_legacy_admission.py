"""All historical study bypasses reject before construction, loading or reset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from plant_backend.rl import checkpoints
from plant_backend.rl.policies import FixedActionPolicy, LoadedPolicy
from plant_backend.rl.v2 import beta_controller_benchmark as benchmark
from plant_backend.rl.v2 import beta_spaces, policy_evaluation, setpoint_spaces
from plant_backend.rl.v2.beta_config import CyclicMarketTaskConfigV1
from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta
from plant_backend.rl.v2.study.legacy_admission import require_legacy_evaluation_scope
from plant_backend.rl.v2.training.config import (
    BETA_CORE_ALGORITHM_IDS,
    BETA_OPTIMIZER_SEEDS,
    ControllerNeutralTrainingConfigV1,
)
from plant_backend.rl.v2.training.registered_contract import registered_config


def _forbidden(*_args: object, **_kwargs: object) -> Any:
    pytest.fail("historical evaluation crossed custody before admission")


@pytest.fixture(autouse=True)
def no_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PlantCyclicMarketEnvV2Beta, "__init__", _forbidden)
    monkeypatch.setattr(PlantCyclicMarketEnvV2Beta, "reset", _forbidden)
    monkeypatch.setattr(policy_evaluation, "make_env", _forbidden)
    monkeypatch.setattr(policy_evaluation, "load_policy", _forbidden)
    monkeypatch.setattr(benchmark, "_new_beta_environment", _forbidden)
    monkeypatch.setattr(benchmark, "_prepare_cell", _forbidden)
    monkeypatch.setattr(benchmark, "load_policy", _forbidden)
    monkeypatch.setattr(checkpoints, "load_metadata", _forbidden)


def _fixture() -> ControllerNeutralTrainingConfigV1:
    return ControllerNeutralTrainingConfigV1(
        environment_id=beta_spaces.ENVIRONMENT_ID,
        beta_task=CyclicMarketTaskConfigV1(),
        notes="reset-free scope admission fixture",
    ).validate()


def _loaded(config: ControllerNeutralTrainingConfigV1) -> LoadedPolicy:
    # No artifact is created or opened. Deliberately incomplete descriptors must
    # be rejected on scope before a concrete checkpoint can be considered.
    return LoadedPolicy(
        policy=FixedActionPolicy([-1.0] * 4),
        path=Path("/private/tmp/nonexistent-scope-fixture"),
        metadata=checkpoints.CheckpointMetadata(
            format=checkpoints.FORMAT_FIXED,
            algorithm="fixture",
            environment_id=config.environment_id,
            training_config=config.to_dict(),
        ),
    )


@pytest.mark.parametrize(
    "seed",
    [1_000_000, 1_001_563, 1_999_999, 2_000_000, 2_000_031, 3_000_000, 3_000_031],
)
@pytest.mark.parametrize(
    "route", ["benchmark", "cell", "paired", "policy", "heuristic"]
)
def test_every_public_execution_route_refuses_reserved_seeds(
    seed: int, route: str
) -> None:
    config = _fixture()
    loaded = _loaded(config)
    environment = object.__new__(PlantCyclicMarketEnvV2Beta)
    with pytest.raises(ValueError, match="study-stage custody"):
        if route == "benchmark":
            benchmark.evaluate_beta_controller_benchmark(
                loaded, config, seeds=[seed], require_trained_policy=False
            )
        elif route == "cell":
            benchmark.execute_beta_controller_benchmark_cell(loaded, config, seed=seed)
        elif route == "paired":
            policy_evaluation.compare_policy_to_heuristic(loaded, config, seeds=[seed])
        elif route == "policy":
            policy_evaluation.execute_policy_episode(environment, loaded, seed=seed)
        else:
            policy_evaluation.execute_heuristic_episode(environment, seed=seed)


@pytest.mark.parametrize("algorithm", BETA_CORE_ALGORITHM_IDS)
@pytest.mark.parametrize("seed", BETA_OPTIMIZER_SEEDS)
def test_all_registered_cells_refused_even_on_fixture_seed(
    algorithm: str, seed: int
) -> None:
    config = registered_config(
        Path("/private/tmp/unused-registered-scope"), algorithm, seed
    )
    with pytest.raises(ValueError, match="study-stage custody"):
        require_legacy_evaluation_scope(
            config.environment_id, training_config=config.to_dict(), seeds=[9_000_075]
        )


@pytest.mark.parametrize(
    "route",
    [
        "benchmark",
        "cell",
        "paired",
        "policy",
        "policy_loader",
        "benchmark_loader",
        "policy_readmission",
        "benchmark_readmission",
    ],
)
def test_registered_declaration_refused_before_loading_or_reset(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = registered_config(
        Path("/private/tmp/unused-registered-scope"), BETA_CORE_ALGORITHM_IDS[0], 101
    )
    loaded = _loaded(config)
    environment = object.__new__(PlantCyclicMarketEnvV2Beta)
    with pytest.raises(ValueError, match="study-stage custody"):
        if route == "benchmark":
            benchmark.evaluate_beta_controller_benchmark(
                loaded, config, seeds=[9_000_075], require_trained_policy=False
            )
        elif route == "cell":
            benchmark.execute_beta_controller_benchmark_cell(
                loaded, config, seed=9_000_075
            )
        elif route == "paired":
            policy_evaluation.compare_policy_to_heuristic(
                loaded, config, seeds=[9_000_075]
            )
        elif route == "policy":
            policy_evaluation.execute_policy_episode(
                environment, loaded, seed=9_000_075
            )
        elif route == "policy_loader":
            policy_evaluation.load_policy_for_task(loaded.path, config)
        elif route == "benchmark_loader":
            monkeypatch.setattr(benchmark, "load_training_task", lambda _path: config)
            benchmark.load_beta_benchmark_policy(loaded.path)
        elif route == "policy_readmission":
            policy_evaluation._preflight_loaded_policy_task(loaded, config)
        else:
            benchmark._readmit_policy(loaded, config, require_trained_policy=False)


@pytest.mark.parametrize(
    "route", ["cell", "policy_readmission", "benchmark_readmission"]
)
def test_fixture_task_does_not_override_registered_checkpoint_declaration(
    route: str,
) -> None:
    config = _fixture()
    registered = registered_config(
        Path("/private/tmp/unused-registered-scope"), BETA_CORE_ALGORITHM_IDS[0], 101
    )
    loaded = _loaded(registered)
    with pytest.raises(ValueError, match="study-stage custody"):
        if route == "cell":
            benchmark.execute_beta_controller_benchmark_cell(
                loaded, config, seed=9_000_075
            )
        elif route == "policy_readmission":
            policy_evaluation._preflight_loaded_policy_task(loaded, config)
        else:
            benchmark._readmit_policy(loaded, config, require_trained_policy=False)


def test_beta_environment_cannot_be_downgraded_by_checkpoint_environment() -> None:
    config = ControllerNeutralTrainingConfigV1()
    loaded = _loaded(config)
    environment = object.__new__(PlantCyclicMarketEnvV2Beta)
    with pytest.raises(ValueError, match="study-stage custody"):
        policy_evaluation.execute_policy_episode(environment, loaded, seed=3_000_000)


@pytest.mark.parametrize("seed", [9_000_000, 9_999_999])
def test_fixture_domain_remains_available_to_legacy_beta(seed: int) -> None:
    require_legacy_evaluation_scope(
        beta_spaces.ENVIRONMENT_ID, training_config=_fixture().to_dict(), seeds=[seed]
    )


@pytest.mark.parametrize("seed", [1_000_000, 2_000_000, 3_000_000])
def test_nonbeta_seed_semantics_are_unchanged(seed: int) -> None:
    require_legacy_evaluation_scope(setpoint_spaces.ENVIRONMENT_ID, seeds=[seed])


@pytest.mark.parametrize("seed", [True, -1, 2_000_000.0, "2000000"])
def test_direct_beta_admission_rejects_ambiguous_seed_types(seed: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        require_legacy_evaluation_scope(beta_spaces.ENVIRONMENT_ID, seeds=[seed])


def test_empty_seed_batch_does_not_admit_registered_checkpoint() -> None:
    with pytest.raises(ValueError, match="study-stage custody"):
        require_legacy_evaluation_scope(
            beta_spaces.ENVIRONMENT_ID,
            training_config={"beta_rl": {"run_kind": "registered"}},
        )


def test_supplied_training_config_cannot_downgrade_beta_scope() -> None:
    with pytest.raises(ValueError, match="study-stage custody"):
        require_legacy_evaluation_scope(
            setpoint_spaces.ENVIRONMENT_ID,
            training_config=_fixture().to_dict(),
            seeds=[3_000_031],
        )

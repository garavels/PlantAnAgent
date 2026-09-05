"""Controller-neutral RL training, checkpoint, inference, and evaluation seams."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import ModuleType

from gymnasium import spaces
import numpy as np
import pytest

from plant_backend.controllers.base import Constraints
from plant_backend.rl import checkpoints
from plant_backend.rl.policies import LoadedPolicy, load_policy
from plant_backend.rl.v2 import economic_spaces, policy_evaluation, setpoint_spaces
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.economics import DayAheadTariffV1, EconomicAssumptionsV1
from plant_backend.rl.v2.mpc_schema import domain_sha256
from plant_backend.rl.v2.policy_evaluation import (
    compare_policy_to_heuristic,
    execute_policy_episode,
    load_policy_for_task,
)
from plant_backend.rl.v2.policy_setpoint_controller import (
    NORMALIZED_POLICY_SETPOINT_CONTROLLER_ID,
    NormalizedSetpointPolicyControllerV1,
    policy_contract_for_environment,
)
from plant_backend.rl.training.config import PPOConfig
from plant_backend.rl.v2.training.config import ControllerNeutralTrainingConfigV1
from plant_backend.rl.v2.training.ppo import (
    admit_resume,
    build_model,
    make_env,
    plan,
    save_checkpoint,
)
from plant_backend.rl.v2.training.train import (
    main as training_main,
    train as train_policy,
)


def _episode() -> EpisodeConfigV2:
    return EpisodeConfigV2(
        horizon_h=0.25,
        delivery_period_h=0.25,
        delivery_rate_kg_h=1.0,
        battery_soc=(0.5, 0.5),
        h2_fill=(0.0, 0.0),
        co2_fill=(0.0, 0.0),
        ch4_coverage_h=(0.0, 0.0),
    )


def _config(**changes: object) -> ControllerNeutralTrainingConfigV1:
    values = {
        "episode": _episode(),
        "total_timesteps": 4,
        "n_envs": 1,
        "checkpoint_every": 0,
        **changes,
    }
    return ControllerNeutralTrainingConfigV1(**values).validate()


def _fixed_checkpoint(
    tmp_path: Path, config: ControllerNeutralTrainingConfigV1
) -> Path:
    contract = policy_contract_for_environment(make_env(config))
    return checkpoints.write_fixed_checkpoint(
        tmp_path / "fixed-v2",
        [-1.0, -1.0, -1.0, -1.0],
        contract=contract,
    )


class _FakeSB3Model:
    """Small artifact writer with the concrete spaces a real SB3 model embeds."""

    def __init__(self, config: ControllerNeutralTrainingConfigV1) -> None:
        environment = make_env(config)
        self.observation_space = environment.observation_space
        self.action_space = environment.action_space
        environment.close()

    def save(self, path: Path) -> None:
        path.write_bytes(b"stable policy bytes")


def test_training_config_round_trips_every_task_identity(tmp_path: Path) -> None:
    assumptions = EconomicAssumptionsV1(
        tariff=DayAheadTariffV1(
            id="unit-tariff",
            hourly_eur_per_mwh=tuple(float(hour) for hour in range(24)),
        ),
        delivered_service_value_eur_per_kg=5.0,
    )
    config = _config(economic_assumptions=assumptions, notes="unit round trip")
    target = config.write(tmp_path / "training.json")

    loaded = ControllerNeutralTrainingConfigV1.from_file(target)

    assert loaded == config
    assert loaded.to_dict() == config.to_dict()


def test_training_config_rejects_unknown_top_level_and_nested_settings() -> None:
    with pytest.raises(ValueError, match="unknown controller-neutral training"):
        ControllerNeutralTrainingConfigV1.from_dict({"optimizer_of_hope": True})
    with pytest.raises(ValueError, match="unknown episode settings"):
        ControllerNeutralTrainingConfigV1.from_dict(
            {"episode": {"private_weather_truth": True}}
        )
    with pytest.raises(ValueError, match="unknown economic assumption settings"):
        ControllerNeutralTrainingConfigV1.from_dict(
            {"economic_assumptions": {"free_energy": 1.0}}
        )


def test_training_config_normalizes_numbers_and_rejects_invalid_ppo() -> None:
    loaded = ControllerNeutralTrainingConfigV1.from_dict(
        {
            "plant": {"solar": {"peak_kw": 5000}},
            "ppo": {"n_steps": 2048, "batch_size": 256},
        }
    )
    assert loaded.plant.solar.peak_kw == 5000.0
    assert isinstance(loaded.plant.solar.peak_kw, float)

    with pytest.raises(TypeError, match="PPO settings.n_steps must be an integer"):
        ControllerNeutralTrainingConfigV1.from_dict({"ppo": {"n_steps": "2048"}})
    with pytest.raises(ValueError, match="must divide"):
        replace(
            _config(),
            ppo=replace(PPOConfig(), batch_size=7),
        ).validate()


@pytest.mark.parametrize(
    ("environment_id", "observation_size", "action_size"),
    [
        (
            economic_spaces.ENVIRONMENT_ID,
            economic_spaces.OBSERVATION_SIZE,
            economic_spaces.ACTION_SIZE,
        ),
        (
            setpoint_spaces.ENVIRONMENT_ID,
            setpoint_spaces.OBSERVATION_SIZE,
            setpoint_spaces.ACTION_SIZE,
        ),
    ],
)
def test_dry_training_plan_identifies_the_exact_four_target_task(
    environment_id: str,
    observation_size: int,
    action_size: int,
) -> None:
    config = _config(environment_id=environment_id)

    described = plan(config)

    assert described["environment_id"] == environment_id
    assert described["observation_shape"] == [observation_size]
    assert described["action_shape"] == [action_size]
    assert described["steps_per_episode"] == 1
    assert described["physics_steps_per_episode"] == 15
    assert described["training_library_imported"] is False
    assert (
        described["first_reset_checkpoint_contract_digest"]
        == described["checkpoint_contract_digest"]
    )


def test_dry_run_cli_constructs_the_task_without_training(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _config().write(tmp_path / "training.json")

    assert training_main(["--config", str(config_path), "--dry-run"]) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out)["dry_run"] is True
    assert "nothing was trained" in captured.err


def test_v2_checkpoint_requires_the_exact_interface_and_full_task(
    tmp_path: Path,
) -> None:
    config = _config()
    environment = make_env(config)
    contract = policy_contract_for_environment(environment)
    path = _fixed_checkpoint(tmp_path, config)

    metadata = checkpoints.load_metadata(path, contract=contract)

    assert metadata.observation_fields == economic_spaces.OBSERVATION_NAMES
    assert metadata.action_fields == economic_spaces.ACTION_NAMES
    assert metadata.environment_id == economic_spaces.ENVIRONMENT_ID
    assert metadata.checkpoint_contract_digest == environment.checkpoint_contract_digest
    assert load_policy(path, contract=contract).metadata == metadata
    with pytest.raises(checkpoints.IncompatibleCheckpointError):
        load_policy(path)

    changed = _config(
        economic_assumptions=EconomicAssumptionsV1(
            battery_replacement_eur_per_kwh=301.0
        )
    )
    changed_contract = policy_contract_for_environment(make_env(changed))
    assert changed_contract.environment_contract_digest == (
        contract.environment_contract_digest
    )
    assert changed_contract.checkpoint_contract_digest != (
        contract.checkpoint_contract_digest
    )
    with pytest.raises(
        checkpoints.IncompatibleCheckpointError,
        match="task contract digest",
    ):
        load_policy(path, contract=changed_contract)


def test_checkpoint_bound_policy_and_gym_action_have_identical_physics(
    tmp_path: Path,
) -> None:
    config = _config()
    direct = make_env(config)
    commanded = make_env(config)
    first_observation, _ = direct.reset(seed=3)
    commanded.reset(seed=3)
    loaded = load_policy(
        _fixed_checkpoint(tmp_path, config),
        contract=policy_contract_for_environment(commanded),
    )
    controller = NormalizedSetpointPolicyControllerV1.from_environment(
        loaded,
        commanded,
    )
    controller.reset()
    decision = controller.compute_command(
        commanded.economic_controller_observation,
        constraints=commanded.constraints,
        config=commanded.config,
        commitment_profile=commanded.episode_config.electrolyser_commitment,
    )
    expected_observation_sha256 = domain_sha256(
        "rl-policy-observation-vector-v1",
        [float(value) for value in first_observation],
    )

    direct_result = direct.step(np.full(4, -1.0, dtype=np.float32))
    commanded_result = commanded.step_process_setpoints(decision.command)

    assert decision.evidence.policy_observation_sha256 == expected_observation_sha256
    assert decision.command.issuer_id == NORMALIZED_POLICY_SETPOINT_CONTROLLER_ID
    assert np.array_equal(direct_result[0], commanded_result[0])
    assert direct_result[1:4] == commanded_result[1:4]
    assert direct.metrics.to_dict() == commanded.metrics.to_dict()
    assert direct.economic_metrics.to_dict() == commanded.economic_metrics.to_dict()
    assert commanded.last_process_setpoint_execution is not None
    assert commanded.last_process_setpoint_execution.command == decision.command


def test_policy_episode_retains_checkpoint_decision_and_application_evidence(
    tmp_path: Path,
) -> None:
    config = _config()
    environment = make_env(config)
    loaded = load_policy(
        _fixed_checkpoint(tmp_path, config),
        contract=policy_contract_for_environment(environment),
    )

    result = execute_policy_episode(environment, loaded, seed=0)

    assert result["status"] == "complete"
    assert result["controller_id"] == NORMALIZED_POLICY_SETPOINT_CONTROLLER_ID
    assert result["completed_decisions"] == 1
    assert result["physical_steps"] == 15
    assert result["execution_count"] == 1
    assert result["application_count"] == 15
    assert result["decisions"][0]["evidence"]["checkpoint_metadata_sha256"]
    assert result["executions"][0]["command"]["issuer"]["id"] == (
        NORMALIZED_POLICY_SETPOINT_CONTROLLER_ID
    )
    assert result["reward_trace_sum"] == pytest.approx(
        result["economic_metrics"]["net_operating_value_eur"]
    )


def test_policy_episode_also_supports_the_physical_four_target_ablation(
    tmp_path: Path,
) -> None:
    config = _config(environment_id=setpoint_spaces.ENVIRONMENT_ID)
    environment = make_env(config)
    loaded = load_policy(
        _fixed_checkpoint(tmp_path, config),
        contract=policy_contract_for_environment(environment),
    )

    result = execute_policy_episode(environment, loaded, seed=0)

    assert result["status"] == "complete"
    assert result["economic_metrics"] is None
    assert result["application_count"] == 15


def test_policy_encoder_stays_identical_to_gym_across_multiple_decisions(
    tmp_path: Path,
) -> None:
    config = _config(
        episode=replace(_episode(), horizon_h=0.5),
        total_timesteps=2,
    )
    environment = make_env(config)
    loaded = load_policy(
        _fixed_checkpoint(tmp_path, config),
        contract=policy_contract_for_environment(environment),
    )

    result = execute_policy_episode(environment, loaded, seed=7)

    assert result["status"] == "complete"
    assert result["completed_decisions"] == 2
    assert result["physical_steps"] == 30
    assert len(result["decisions"]) == 2


def test_policy_refuses_runtime_constraint_drift_before_state_movement(
    tmp_path: Path,
) -> None:
    config = _config()
    environment = make_env(config)
    environment.reset(seed=0)
    loaded = load_policy(
        _fixed_checkpoint(tmp_path, config),
        contract=policy_contract_for_environment(environment),
    )
    controller = NormalizedSetpointPolicyControllerV1.from_environment(
        loaded, environment
    )

    with pytest.raises(ValueError, match="constraints differ"):
        controller.compute_command(
            environment.economic_controller_observation,
            constraints=Constraints(battery_min_soc=0.25),
            config=environment.config,
            commitment_profile=environment.episode_config.electrolyser_commitment,
        )
    assert environment.decisions_taken == 0
    assert environment.physics_steps_taken == 0


def test_policy_failure_is_retained_without_a_fallback_or_state_movement(
    tmp_path: Path,
) -> None:
    config = _config()
    environment = make_env(config)
    contract = policy_contract_for_environment(environment)
    path = _fixed_checkpoint(tmp_path, config)
    metadata = checkpoints.load_metadata(path, contract=contract)

    class BrokenPolicy:
        def act(self, observation: np.ndarray) -> np.ndarray:
            del observation
            raise RuntimeError("inference failed deliberately")

    result = execute_policy_episode(
        environment,
        LoadedPolicy(BrokenPolicy(), metadata, path),
        seed=0,
    )

    assert result["status"] == "failed"
    assert result["completed_decisions"] == 0
    assert result["physical_steps"] == 0
    assert result["execution_count"] == 0
    assert result["failure"]["exception_type"] == "RuntimeError"
    assert "deliberately" in result["failure"]["message"]


def test_paired_evaluation_uses_identical_task_identity_and_labels_mock(
    tmp_path: Path,
) -> None:
    config = _config()
    path = _fixed_checkpoint(tmp_path, config)
    loaded = load_policy(
        path,
        contract=policy_contract_for_environment(make_env(config)),
    )

    result = compare_policy_to_heuristic(loaded, config, seeds=[0])

    assert result["smoke_test"] is True
    assert [run["controller_id"] for run in result["runs"]] == [
        "legacy-heuristic-setpoint-authority-v1",
        NORMALIZED_POLICY_SETPOINT_CONTROLLER_ID,
    ]
    assert (
        result["runs"][0]["manifest_identity"] == result["runs"][1]["manifest_identity"]
    )
    assert all(run["status"] == "complete" for run in result["runs"])
    assert result["deterministic_evidence_sha256"]

    with pytest.raises(ValueError, match="at least one seed"):
        compare_policy_to_heuristic(loaded, config, seeds=[])
    with pytest.raises(ValueError, match="unique"):
        compare_policy_to_heuristic(loaded, config, seeds=[0, 0])


def test_saved_policy_artifact_is_hashed_and_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    config = _config()
    model = _FakeSB3Model(config)

    invalid_path = tmp_path / "invalid-checkpoint"
    with pytest.raises(ValueError, match="total_timesteps"):
        save_checkpoint(model, config, invalid_path, total_timesteps=-1)
    assert not invalid_path.exists()

    path = save_checkpoint(model, config, tmp_path / "checkpoint", total_timesteps=4)
    contract = policy_contract_for_environment(make_env(config))
    metadata = checkpoints.load_metadata(path, contract=contract)
    assert metadata.policy_artifact_sha256

    checkpoints.policy_path(path).write_bytes(b"tampered")
    with pytest.raises(
        checkpoints.CorruptCheckpointError, match="content does not match"
    ):
        checkpoints.load_metadata(path, contract=contract)


def test_checkpoint_save_refuses_model_internal_space_mismatch(
    tmp_path: Path,
) -> None:
    config = _config()
    model = _FakeSB3Model(config)
    model.observation_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(1,),
        dtype=np.float32,
    )
    output = tmp_path / "mislabeled-checkpoint"

    with pytest.raises(
        checkpoints.IncompatibleCheckpointError,
        match="observation shape",
    ):
        save_checkpoint(model, config, output, total_timesteps=4)

    assert not output.exists()


def test_checkpoint_load_refuses_artifact_internal_space_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    environment = make_env(config)
    checkpoint = save_checkpoint(
        _FakeSB3Model(config),
        config,
        tmp_path / "checkpoint-with-mislabeled-bytes",
        total_timesteps=4,
    )

    class WrongSpaceModel:
        observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        action_space = environment.action_space

        def predict(self, observation: np.ndarray, deterministic: bool) -> object:
            del observation, deterministic
            return np.zeros(4, dtype=np.float32), None

    class FakePPO:
        @classmethod
        def load(cls, *_: object, **__: object) -> WrongSpaceModel:
            return WrongSpaceModel()

    fake_sb3 = ModuleType("stable_baselines3")
    fake_sb3.PPO = FakePPO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stable_baselines3", fake_sb3)
    contract = policy_contract_for_environment(environment)

    with pytest.raises(
        checkpoints.IncompatibleCheckpointError,
        match="observation shape",
    ):
        load_policy(
            checkpoint,
            contract=contract,
            environment=environment,
        )
    generically_loaded = load_policy(checkpoint, contract=contract)
    arm_calls: list[str] = []

    def moved(*_: object, **__: object) -> dict[str, object]:
        arm_calls.append("moved")
        raise AssertionError("an arm moved before embedded-space admission")

    monkeypatch.setattr(policy_evaluation, "execute_heuristic_episode", moved)
    monkeypatch.setattr(policy_evaluation, "execute_policy_episode", moved)
    with pytest.raises(
        checkpoints.IncompatibleCheckpointError,
        match="observation shape",
    ):
        policy_evaluation.compare_policy_to_heuristic(
            generically_loaded,
            config,
            seeds=[0],
        )
    assert arm_calls == []
    environment.close()


def test_resume_rejects_changed_embedded_ppo_settings_and_records_parent(
    tmp_path: Path,
) -> None:
    parent_config = _config(output_dir=str(tmp_path / "parent-run"))
    parent = save_checkpoint(
        _FakeSB3Model(parent_config),
        parent_config,
        tmp_path / "parent-checkpoint",
        total_timesteps=4,
    )
    resumed_config = replace(
        parent_config,
        output_dir=str(tmp_path / "resumed-run"),
        resume_from=str(parent),
    )

    admitted = admit_resume(resumed_config)

    assert admitted is not None
    assert admitted.total_timesteps == 4
    assert admitted.policy_artifact_sha256
    child = save_checkpoint(
        _FakeSB3Model(resumed_config),
        resumed_config,
        tmp_path / "resumed-checkpoint",
        total_timesteps=8,
        resume_parent=admitted,
    )
    resumed_environment = make_env(resumed_config)
    lineage = checkpoints.load_metadata(
        child,
        contract=policy_contract_for_environment(resumed_environment),
    ).extra["resume_parent"]
    resumed_environment.close()
    assert lineage == admitted.to_dict()

    changed = replace(
        resumed_config,
        ppo=replace(resumed_config.ppo, gamma=0.9),
    )
    with pytest.raises(
        checkpoints.IncompatibleCheckpointError,
        match="gamma",
    ):
        admit_resume(changed)


def test_resume_rejects_a_checkpoint_from_the_other_four_target_task(
    tmp_path: Path,
) -> None:
    physical_config = _config(environment_id=setpoint_spaces.ENVIRONMENT_ID)
    physical = save_checkpoint(
        _FakeSB3Model(physical_config),
        physical_config,
        tmp_path / "physical-checkpoint",
        total_timesteps=4,
    )
    economic_resume = replace(
        _config(),
        resume_from=str(physical),
        output_dir=str(tmp_path / "economic-resume"),
    )

    with pytest.raises(checkpoints.IncompatibleCheckpointError):
        admit_resume(economic_resume)


def test_resume_helpers_independently_reject_a_substituted_parent_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    first = save_checkpoint(
        _FakeSB3Model(config),
        config,
        tmp_path / "first-parent",
        total_timesteps=4,
    )
    second = save_checkpoint(
        _FakeSB3Model(config),
        config,
        tmp_path / "second-parent",
        total_timesteps=4,
    )
    first_resume = replace(
        config,
        output_dir=str(tmp_path / "first-child"),
        resume_from=str(first),
    )
    second_resume = replace(
        config,
        output_dir=str(tmp_path / "second-child"),
        resume_from=str(second),
    )
    substituted = admit_resume(second_resume)
    assert substituted is not None
    output = tmp_path / "substituted-child-checkpoint"

    with pytest.raises(
        checkpoints.CorruptCheckpointError,
        match="differs from independent admission",
    ):
        save_checkpoint(
            _FakeSB3Model(first_resume),
            first_resume,
            output,
            total_timesteps=8,
            resume_parent=substituted,
        )

    assert not output.exists()

    class ForbiddenPPO:
        @classmethod
        def load(cls, *_: object, **__: object) -> object:
            raise AssertionError("PPO.load ran with a substituted resume identity")

    fake_sb3 = ModuleType("stable_baselines3")
    fake_sb3.PPO = ForbiddenPPO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stable_baselines3", fake_sb3)
    with pytest.raises(
        checkpoints.CorruptCheckpointError,
        match="differs from independent admission",
    ):
        build_model(
            first_resume,
            env=object(),
            resume_parent=substituted,
        )


def test_training_setup_failure_does_not_claim_an_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "not-a-run"
    config = _config(output_dir=str(output))

    def unavailable(_: ControllerNeutralTrainingConfigV1) -> None:
        raise ImportError("deliberately unavailable")

    monkeypatch.setattr(
        "plant_backend.rl.v2.training.ppo.build_vec_env",
        unavailable,
    )
    with pytest.raises(ImportError, match="deliberately unavailable"):
        train_policy(config)
    assert not output.exists()


@pytest.mark.parametrize(
    "environment_id",
    [setpoint_spaces.ENVIRONMENT_ID, economic_spaces.ENVIRONMENT_ID],
)
def test_real_sb3_tiny_train_save_load_and_evaluate(
    tmp_path: Path,
    environment_id: str,
) -> None:
    pytest.importorskip("stable_baselines3")
    task_name = (
        "physical" if environment_id == setpoint_spaces.ENVIRONMENT_ID else "economic"
    )
    config = _config(
        environment_id=environment_id,
        total_timesteps=2,
        output_dir=str(tmp_path / f"real-sb3-{task_name}-smoke"),
        ppo=replace(
            PPOConfig(),
            n_steps=2,
            batch_size=2,
            n_epochs=1,
        ),
    )

    checkpoint = train_policy(config)
    resumed_config = replace(
        config,
        output_dir=str(tmp_path / f"real-sb3-{task_name}-resume-smoke"),
        resume_from=str(checkpoint),
    )
    resumed_checkpoint = train_policy(resumed_config)
    loaded = load_policy_for_task(resumed_checkpoint, resumed_config)
    result = compare_policy_to_heuristic(loaded, resumed_config, seeds=[11])

    assert checkpoint.is_dir()
    assert resumed_checkpoint.is_dir()
    assert loaded.metadata.trained is True
    assert loaded.metadata.total_timesteps == 4
    assert loaded.metadata.policy_artifact_sha256
    assert loaded.metadata.extra["resume_parent"]["total_timesteps"] == 2
    assert result["smoke_test"] is False
    assert all(run["status"] == "complete" for run in result["runs"])

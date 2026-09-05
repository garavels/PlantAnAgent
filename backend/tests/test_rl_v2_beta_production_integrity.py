"""Adversarial checks at the constrained beta production boundary.

The module owns one eight-transition fixture artifact.  Tests copy and mutate
that checkpoint rather than repeating optimization, so the real SB3 container
is exercised without turning integrity coverage into an experiment campaign.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, Callable, cast

import pytest

from plant_backend.rl import checkpoints
from plant_backend.rl.policies import load_policy
from plant_backend.rl.training.config import PPOConfig
from plant_backend.rl.v2 import beta_spaces
from plant_backend.rl.v2.beta_config import CyclicMarketTaskConfigV1
from plant_backend.rl.v2.policy_setpoint_controller import (
    policy_contract_for_environment,
)
from plant_backend.rl.v2.training import constrained_ppo
from plant_backend.rl.v2.training import ppo as ppo_wiring
from plant_backend.rl.v2.training.beta_checkpoint import (
    BetaRLCheckpointEvidenceV1,
    parse_beta_checkpoint_evidence,
)
from plant_backend.rl.v2.training.config import (
    BETA_SCALAR_PPO_ALGORITHM_ID,
    BetaRLTrainingRunV1,
    ControllerNeutralTrainingConfigV1,
)
from plant_backend.rl.v2.training.episode_allocator import (
    EpisodeAllocatorModeV1,
    EpisodeSeedSplitV1,
)
from plant_backend.rl.v2.training.runtime_provenance import (
    BetaTrainingRuntimeProvenanceV1,
    RuntimeProvenanceChangedError,
)
from plant_backend.rl.v2.training.train import train


@dataclass(frozen=True, slots=True)
class _GenuineFixtureCheckpoint:
    path: Path
    config: ControllerNeutralTrainingConfigV1
    evidence: BetaRLCheckpointEvidenceV1


def _fixture_config(output: Path) -> ControllerNeutralTrainingConfigV1:
    return ControllerNeutralTrainingConfigV1(
        environment_id=beta_spaces.ENVIRONMENT_ID,
        seed=101,
        total_timesteps=8,
        n_envs=4,
        checkpoint_every=0,
        output_dir=str(output),
        beta_task=CyclicMarketTaskConfigV1(),
        beta_rl=BetaRLTrainingRunV1(
            algorithm=BETA_SCALAR_PPO_ALGORITHM_ID,
            episode_seed_base=9_800_000,
        ),
        ppo=replace(
            PPOConfig(),
            n_steps=2,
            batch_size=8,
            n_epochs=1,
        ),
        notes="bounded production-integrity fixture; no research outcome",
    ).validate()


@pytest.fixture(scope="module")
def genuine_fixture_checkpoint(
    tmp_path_factory: pytest.TempPathFactory,
) -> _GenuineFixtureCheckpoint:
    """Create the smallest shared real artifact needed by the downgrade tests."""

    pytest.importorskip("stable_baselines3")
    root = tmp_path_factory.mktemp("beta-production-integrity")
    config = _fixture_config(root / "source-run")
    checkpoint = train(config)
    metadata = checkpoints.load_metadata(checkpoint, check=False)
    evidence = parse_beta_checkpoint_evidence(
        metadata.extra[ppo_wiring.BETA_CHECKPOINT_EXTRA_KEY]
    )
    return _GenuineFixtureCheckpoint(checkpoint, config, evidence)


def _copy_checkpoint(
    source: _GenuineFixtureCheckpoint,
    tmp_path: Path,
    name: str,
) -> Path:
    destination = tmp_path / name
    shutil.copytree(source.path, destination)
    return destination


def _read_metadata_payload(checkpoint: Path) -> dict[str, Any]:
    value = json.loads(
        checkpoints.metadata_path(checkpoint).read_text(encoding="utf-8")
    )
    if not isinstance(value, dict):  # pragma: no cover - source fixture is trusted
        raise TypeError("checkpoint metadata fixture must be an object")
    return cast(dict[str, Any], value)


def _write_metadata_payload(checkpoint: Path, payload: dict[str, Any]) -> None:
    checkpoints.metadata_path(checkpoint).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _config_from_payload(payload: dict[str, Any]) -> ControllerNeutralTrainingConfigV1:
    training_config = payload.get("training_config")
    if not isinstance(training_config, dict):
        raise TypeError("checkpoint training_config must be an object")
    return ControllerNeutralTrainingConfigV1.from_dict(training_config)


def _load_from_declared_task(checkpoint: Path) -> None:
    payload = _read_metadata_payload(checkpoint)
    config = _config_from_payload(payload)
    environment = ppo_wiring.make_env(config)
    try:
        arguments: dict[str, Any] = {}
        if config.beta_rl is not None:
            metadata = checkpoints.load_metadata(checkpoint, check=False)
            evidence = parse_beta_checkpoint_evidence(
                metadata.extra[ppo_wiring.BETA_CHECKPOINT_EXTRA_KEY]
            )
            arguments["expected_beta_runtime_provenance"] = evidence.runtime_provenance
        load_policy(
            checkpoint,
            contract=policy_contract_for_environment(environment),
            environment=environment,
            **arguments,
        )
    finally:
        environment.close()


def _rewrite_declared_task_contract(
    checkpoint: Path,
    payload: dict[str, Any],
) -> None:
    config = _config_from_payload(payload)
    environment = ppo_wiring.make_env(config)
    try:
        contract = policy_contract_for_environment(environment)
    finally:
        environment.close()
    payload["spec_version"] = contract.spec_version
    payload["observation_fields"] = list(contract.observation_fields)
    payload["action_fields"] = list(contract.action_fields)
    payload["environment_id"] = contract.environment_id
    payload["environment_contract_digest"] = contract.environment_contract_digest
    payload["checkpoint_contract_digest"] = contract.checkpoint_contract_digest
    _write_metadata_payload(checkpoint, payload)


def test_constrained_artifact_cannot_be_downgraded_to_generic_ppo_metadata(
    genuine_fixture_checkpoint: _GenuineFixtureCheckpoint,
    tmp_path: Path,
) -> None:
    checkpoint = _copy_checkpoint(
        genuine_fixture_checkpoint,
        tmp_path,
        "downgraded-metadata",
    )
    payload = _read_metadata_payload(checkpoint)
    payload["algorithm"] = "PPO (stable-baselines3)"
    training_config = cast(dict[str, Any], payload["training_config"])
    training_config["beta_rl"] = None
    payload["extra"] = {}
    _write_metadata_payload(checkpoint, payload)

    with pytest.raises(
        checkpoints.CorruptCheckpointError,
        match="downgraded|relabelled|constrained artifact",
    ):
        _load_from_declared_task(checkpoint)


def test_beta_metadata_seed_must_match_its_training_config(
    genuine_fixture_checkpoint: _GenuineFixtureCheckpoint,
    tmp_path: Path,
) -> None:
    checkpoint = _copy_checkpoint(
        genuine_fixture_checkpoint,
        tmp_path,
        "seed-mismatch",
    )
    payload = _read_metadata_payload(checkpoint)
    payload["seed"] = 211
    _write_metadata_payload(checkpoint, payload)

    with pytest.raises(
        checkpoints.CorruptCheckpointError,
        match="seed differs from its training contract",
    ):
        _load_from_declared_task(checkpoint)


def test_loaded_model_ppo_configuration_cannot_be_relabelled(
    genuine_fixture_checkpoint: _GenuineFixtureCheckpoint,
    tmp_path: Path,
) -> None:
    checkpoint = _copy_checkpoint(
        genuine_fixture_checkpoint,
        tmp_path,
        "ppo-config-mismatch",
    )
    payload = _read_metadata_payload(checkpoint)
    training_config = cast(dict[str, Any], payload["training_config"])
    ppo_config = cast(dict[str, Any], training_config["ppo"])
    ppo_config["gamma"] = 0.9
    _write_metadata_payload(checkpoint, payload)

    with pytest.raises(
        checkpoints.CorruptCheckpointError,
        match=r"PPO configuration differs.*gamma",
    ):
        _load_from_declared_task(checkpoint)


@pytest.mark.parametrize("mutation", ("base_seed", "episode_horizon"))
def test_allocator_identity_cannot_be_relabelled_by_training_metadata(
    genuine_fixture_checkpoint: _GenuineFixtureCheckpoint,
    tmp_path: Path,
    mutation: str,
) -> None:
    checkpoint = _copy_checkpoint(
        genuine_fixture_checkpoint,
        tmp_path,
        f"allocator-{mutation}",
    )
    payload = _read_metadata_payload(checkpoint)
    training_config = cast(dict[str, Any], payload["training_config"])
    if mutation == "base_seed":
        beta_run = cast(dict[str, Any], training_config["beta_rl"])
        beta_run["episode_seed_base"] = 9_800_100
        _write_metadata_payload(checkpoint, payload)
        expected_field = "base_seed"
    else:
        episode = cast(dict[str, Any], training_config["episode"])
        episode["horizon_h"] = 144.0
        _rewrite_declared_task_contract(checkpoint, payload)
        expected_field = "episode_decisions"

    with pytest.raises(
        checkpoints.CorruptCheckpointError,
        match=rf"allocator evidence differs.*{expected_field}",
    ):
        _load_from_declared_task(checkpoint)


def test_checkpoint_cut_must_be_explained_by_requested_and_parent_timesteps(
    genuine_fixture_checkpoint: _GenuineFixtureCheckpoint,
    tmp_path: Path,
) -> None:
    checkpoint = _copy_checkpoint(
        genuine_fixture_checkpoint,
        tmp_path,
        "cumulative-timestep-mismatch",
    )
    payload = _read_metadata_payload(checkpoint)
    training_config = cast(dict[str, Any], payload["training_config"])
    # A twelve-transition request with an eight-transition rollout implies a
    # sixteen-transition artifact, while this fixture actually contains eight.
    training_config["total_timesteps"] = 12
    _write_metadata_payload(checkpoint, payload)

    with pytest.raises(
        checkpoints.CorruptCheckpointError,
        match="cumulative timestep cut is not explained",
    ):
        _load_from_declared_task(checkpoint)


def test_resume_admission_detects_metadata_replacement_during_artifact_load(
    genuine_fixture_checkpoint: _GenuineFixtureCheckpoint,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _copy_checkpoint(
        genuine_fixture_checkpoint,
        tmp_path,
        "metadata-replacement",
    )
    resume_config = replace(
        genuine_fixture_checkpoint.config,
        output_dir=str(tmp_path / "metadata-replacement-child"),
        resume_from=str(checkpoint),
    ).validate()
    original = ppo_wiring._load_beta_artifact_evidence
    replaced = False

    def replace_metadata_after_load(
        directory: Path,
        config: ControllerNeutralTrainingConfigV1,
    ) -> tuple[dict[str, Any], Any]:
        nonlocal replaced
        result = original(directory, config)
        if not replaced:
            payload = _read_metadata_payload(checkpoint)
            payload["notes"] = "metadata replaced during artifact admission"
            _write_metadata_payload(checkpoint, payload)
            replaced = True
        return result

    monkeypatch.setattr(
        ppo_wiring,
        "_load_beta_artifact_evidence",
        replace_metadata_after_load,
    )

    with pytest.raises(
        checkpoints.CorruptCheckpointError,
        match="metadata changed during admission",
    ):
        ppo_wiring.admit_resume(
            resume_config,
            runtime_provenance=genuine_fixture_checkpoint.evidence.runtime_provenance,
        )


def test_fixture_save_rechecks_runtime_after_artifact_serialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixture_config(tmp_path / "runtime-mutation-run")
    runtime_before = ppo_wiring.capture_runtime_provenance(require_clean=False)
    runtime_after = replace(
        runtime_before,
        python_version=f"{runtime_before.python_version}-mutated",
    )
    artifact_bytes = b"bounded fake constrained artifact"
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    constraint_state = {"schema": "bounded-save-state"}

    class FakeConstrainedPPO:
        num_timesteps = 8
        episode_decision_indices = (2, 2, 2, 2)
        embedded_constraint_state = constraint_state

        def get_env(self) -> object:
            return object()

        def pin_external_episode_state(self, _snapshot: object) -> None:
            return None

        def constraint_provenance(self) -> dict[str, str]:
            return constraint_state

        def save(self, path: Path) -> None:
            path.write_bytes(artifact_bytes)

        def loaded_constraint_state_attestation(self) -> SimpleNamespace:
            return SimpleNamespace(
                artifact_sha256=artifact_sha256,
                num_timesteps=8,
                constraint_state=constraint_state,
            )

        @classmethod
        def load(cls, _path: Path, *, device: str) -> FakeConstrainedPPO:
            del device
            return cls()

    class FakeAllocator:
        mode = EpisodeAllocatorModeV1.FIXTURE
        split = EpisodeSeedSplitV1.FIXTURE
        base_seed = cast(BetaRLTrainingRunV1, config.beta_rl).episode_seed_base
        episode_decisions = config.episode.decision_steps(config.plant)
        decisions_consumed_by_environment = (2, 2, 2, 2)
        cursors = tuple(SimpleNamespace(decision_offset=2) for _ in range(4))

        def snapshot(self) -> SimpleNamespace:
            return SimpleNamespace(to_dict=lambda: {"bounded": "snapshot"})

    captures = iter((runtime_before, runtime_after))

    def capture_runtime(*, require_clean: bool) -> BetaTrainingRuntimeProvenanceV1:
        assert require_clean is False
        return next(captures)

    no_op_verifier: Callable[..., None] = lambda *_args, **_kwargs: None
    monkeypatch.setattr(constrained_ppo, "ConstrainedPPO", FakeConstrainedPPO)
    monkeypatch.setattr(
        ppo_wiring,
        "verify_beta_model_configuration",
        no_op_verifier,
    )
    monkeypatch.setattr(ppo_wiring, "_verify_beta_model_spaces", no_op_verifier)
    monkeypatch.setattr(ppo_wiring, "find_seed_allocator", lambda _env: FakeAllocator())
    monkeypatch.setattr(ppo_wiring, "capture_runtime_provenance", capture_runtime)

    destination = tmp_path / "runtime-mutation-checkpoint"
    with pytest.raises(
        RuntimeProvenanceChangedError,
        match="python_version",
    ):
        ppo_wiring.save_checkpoint(
            FakeConstrainedPPO(),
            config,
            destination,
            total_timesteps=8,
            runtime_provenance=runtime_before,
        )
    assert checkpoints.policy_path(destination).is_file()
    assert not checkpoints.metadata_path(destination).exists()

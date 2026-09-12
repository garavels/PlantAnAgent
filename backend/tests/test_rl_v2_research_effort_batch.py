"""Adaptive batch conditioning, initialization, custody and true learner tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from plant_backend.rl.v2.training import research_effort_batch as batch
from plant_backend.rl.v2.training import ppo
from plant_backend.rl.v2.training.custody_json import canonical_bytes
from plant_backend.rl.v2.training.custody_io import publish_json


def test_fixed_screen_and_untouched_task_contract() -> None:
    from plant_backend.rl.v2.training.registered_contract import registered_config

    spec = batch.specification(batch.ROOT)
    assert len(spec["arms"]) == 4
    assert batch.TOTAL_STEPS == 131072
    assert batch.TRAIN_SEED_BASE + 195 < min(batch.DEV_SEEDS)
    original = registered_config(batch.ROOT / "hypothetical-registered", batch.PI, 101)
    candidate = batch.template(batch.ROOT)
    assert ppo.policy_contract(original) == ppo.policy_contract(candidate)
    assert candidate.ppo == original.ppo
    assert spec["registered_evaluation_eligible"] is False


@pytest.mark.parametrize("arm_index", [0, 2])
def test_optimizer_scaling_preserves_physics_constraints_and_raw_economics(
    tmp_path: Path,
    arm_index: int,
) -> None:
    config = batch.template(tmp_path)
    raw = batch.control_environment(config, batch.ARMS[arm_index])
    scaled = batch.OptimizerRewardScale(
        batch.control_environment(config, batch.ARMS[arm_index]), 0.001
    )
    try:
        a, ai = raw.reset(seed=9_990_800)
        b, bi = scaled.reset(seed=9_990_800)
        assert np.array_equal(a, b)
        assert canonical_bytes(ai) == canonical_bytes(bi)
        for _ in range(96):
            action = np.array([-0.3, -0.3, -0.3, 1], dtype=np.float32)
            a, ar, at, ax, ai = raw.step(action)
            b, br, bt, bx, bi = scaled.step(action)
            assert np.array_equal(a, b)
            assert br == ar * 0.001
            assert (at, ax) == (bt, bx)
            if arm_index == 2:
                assert (
                    len(ai["dispatcher_elapsed_ns"])
                    == len(bi["dispatcher_elapsed_ns"])
                    == 15
                )
                ai = {k: v for k, v in ai.items() if k != "dispatcher_elapsed_ns"}
                bi = {k: v for k, v in bi.items() if k != "dispatcher_elapsed_ns"}
            assert canonical_bytes(ai) == canonical_bytes(bi)
        assert ai["cycle_constraint_transition"]["cycle_settled"] is True
    finally:
        raw.close()
        scaled.close()


def test_matched_zero_head_initialization_without_constructor_resets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    torch.set_num_threads(1)
    monkeypatch.setattr(
        batch.PlantCyclicMarketEnvV2Beta,
        "reset",
        lambda *args, **kwargs: pytest.fail(
            "constructor or policy identity consumed a reset"
        ),
    )
    snapshots = []
    config = batch.template(tmp_path)
    batch.specification(tmp_path)
    for arm in batch.ARMS:
        model, allocator, vector = batch.make_learner(config, arm, seed_base=9_990_810)
        try:
            assert not allocator.ledger
            assert allocator.decisions_consumed_by_environment == (0, 0, 0, 0)
            assert model.observation_space.shape == ((139,) if arm.residual else (135,))
            action, _ = model.predict(
                np.zeros(model.observation_space.shape, dtype=np.float32),
                deterministic=True,
            )
            assert np.array_equal(action, np.zeros(4))
            assert torch.equal(
                model.policy.action_net.weight,
                torch.zeros_like(model.policy.action_net.weight),
            )
            assert torch.equal(
                model.policy.action_net.bias,
                torch.zeros_like(model.policy.action_net.bias),
            )
            assert torch.equal(
                model.policy.log_std,
                torch.full_like(model.policy.log_std, float(np.log(0.2))),
            )
            snapshots.append(
                {
                    k: v.detach().clone()
                    for k, v in model.policy.state_dict().items()
                    if not k.startswith("cost_critic.")
                }
            )
        finally:
            vector.close()
    for left, right in ((0, 1), (2, 3)):
        assert all(
            torch.equal(snapshots[left][k], snapshots[right][k])
            for k in snapshots[left]
        )
    assert (
        snapshots[0]["mlp_extractor.policy_net.0.weight"].shape
        != snapshots[2]["mlp_extractor.policy_net.0.weight"].shape
    )


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_true_scaled_training_diagnostics_and_checkpoint_roundtrip(
    tmp_path: Path, index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    torch.set_num_threads(1)
    original = batch.template(tmp_path)
    config = replace(
        original,
        total_timesteps=64,
        ppo=replace(original.ppo, n_steps=8, batch_size=16, n_epochs=1),
    )
    arm = batch.ARMS[index]
    result = batch.train_cell(tmp_path, config, arm, seed_base=9_990_820)
    assert result["total_timesteps"] == 64
    assert len(result["constraint_state"]["payload"]["rollout_ledgers"]) == 2
    assert len(list((tmp_path / "diagnostics").glob("diagnostic-*.json"))) == 2
    assert result["registered_evaluation_eligible"] is False
    assert result["policy_identity"]["observation_size"] == (
        139 if arm.residual else 135
    )
    assert (
        result["checkpoint_contract_digest"]
        == result["policy_identity"]["checkpoint_contract_digest"]
    )
    assert all(c["decision_offset"] == 16 for c in result["allocator"]["cursors"])
    from copy import deepcopy
    import json
    from plant_backend.rl.v2.training.constrained_ppo import ConstrainedPPO

    tampered = deepcopy(result)
    tampered["policy_identity"]["observation_size"] = 135 if arm.residual else 139
    (tmp_path / "result.json").write_text(json.dumps(tampered))
    with monkeypatch.context() as checked:
        checked.setattr(
            ConstrainedPPO,
            "load",
            lambda *args, **kwargs: pytest.fail(
                "model loaded before interface rejection"
            ),
        )
        with pytest.raises(ValueError, match="interface"):
            batch.load_final(tmp_path, config, arm)
        with pytest.raises(ValueError, match="interface"):
            batch.load_initial(tmp_path, config, arm)
    (tmp_path / "result.json").write_text(json.dumps(result))
    model = batch.load_final(tmp_path, config, arm)
    initial = batch.load_initial(tmp_path, config, arm)
    for policy in (model, initial):
        action, _ = policy.predict(
            np.zeros(139 if arm.residual else 135, dtype=np.float32), deterministic=True
        )
        assert action.shape == (4,)
    with pytest.raises(FileExistsError):
        batch.train_cell(tmp_path, config, arm, seed_base=9_990_820)
    with (tmp_path / "initial-policy.pt").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="initial policy bytes changed"):
        batch.load_initial(tmp_path, config, arm)
    with (tmp_path / "final-policy.zip").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        batch.load_final(tmp_path, config, arm)


def test_parent_closes_failed_cells_and_keeps_every_fixed_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "batch"
    monkeypatch.setattr(batch, "ROOT", root)
    monkeypatch.setattr(batch, "DEV_SEEDS", (9_700_000,))
    runtime = SimpleNamespace(git_head="a" * 40, to_dict=lambda: {"fixture": True})
    monkeypatch.setattr(batch, "capture", lambda **kwargs: runtime)
    gate = tmp_path / "gate.json"
    publish_json(gate, {"passed": True, "source_commit": runtime.git_head})
    calls = []

    def child(
        stage: str, name: str, timeout: int, extra: list[str]
    ) -> dict[str, object]:
        calls.append((stage, name))
        return {
            "stage": stage,
            "name": name,
            "codes": ["deliberate-local-failure"] if "A-absolute-pi" in name else [],
        }

    monkeypatch.setattr(batch, "closed_child", child)
    report = batch.run(gate, authorized=True)
    assert report["status"] == "closed"
    assert sum(s == "_train" for s, _ in calls) == 4
    assert sum(s == "_evaluate" for s, _ in calls) == 10
    assert sum(s == "_replay" for s, _ in calls) == 9
    assert sum(bool(r["codes"]) for r in report["receipts"]) == 2
    with pytest.raises(ValueError, match="no retry"):
        batch.run(gate, authorized=True)


def test_authorization_and_source_gate_precede_root_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "batch"
    monkeypatch.setattr(batch, "ROOT", root)
    gate = tmp_path / "gate.json"
    publish_json(gate, {"passed": True, "source_commit": "b" * 40})
    with pytest.raises(ValueError, match="authorization"):
        batch.run(gate, authorized=False)
    monkeypatch.setattr(
        batch, "capture", lambda **kwargs: SimpleNamespace(git_head="a" * 40)
    )
    with pytest.raises(ValueError, match="exact clean source"):
        batch.run(gate, authorized=True)
    assert not root.exists()


@pytest.mark.parametrize(
    "actor", ["initial-absolute", "initial-residual", "A-absolute-pi", "C-residual-pi"]
)
def test_failed_training_cannot_be_evaluated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, actor: str
) -> None:
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    (tmp_path / "receipts").mkdir()
    directory = tmp_path / "evaluations" / f"9700000-{actor}"
    directory.mkdir(parents=True)
    arm_name = (
        "C-residual-pi"
        if actor in ("initial-residual", "C-residual-pi")
        else "A-absolute-pi"
    )
    publish_json(
        tmp_path / "receipts" / f"train-{arm_name}.json",
        {
            "stage": "_train",
            "name": f"train-{arm_name}",
            "codes": ["child_timeout"],
            "process": {"exit_code": 0},
        },
    )
    with pytest.raises(ValueError, match="successfully closed training child"):
        batch.evaluate_cell(actor, 9_700_000)
    assert (directory / "attempt.json").is_file()
    assert not (directory / "summary.json").exists()


@pytest.mark.parametrize(
    "code,exit_code",
    [("root_custody_lost", -15), ("child_nonzero_exit", batch.INTEGRITY_EXIT)],
)
def test_interruption_or_integrity_closes_receipt_and_stops_remaining_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: str, exit_code: int
) -> None:
    root = tmp_path / "batch"
    monkeypatch.setattr(batch, "ROOT", root)
    runtime = SimpleNamespace(git_head="a" * 40, to_dict=lambda: {"fixture": True})
    monkeypatch.setattr(batch, "capture", lambda **kwargs: runtime)
    monkeypatch.setattr(batch, "require_plan", lambda: {})
    calls = []

    def interrupted(
        *args: object, **kwargs: object
    ) -> tuple[dict[str, object], set[str]]:
        calls.append(args)
        return {"exit_code": exit_code}, {code}

    monkeypatch.setattr(batch, "capture_process", interrupted)
    gate = tmp_path / "gate.json"
    publish_json(gate, {"passed": True, "source_commit": runtime.git_head})
    report = batch.run(gate, authorized=True)
    assert report["status"] == "aborted"
    assert "interruption" in report["failure"]["message"]
    assert len(calls) == 1
    assert (root / "receipts" / "train-A-absolute-pi.json").is_file()
    assert not (root / "training" / "B-absolute-scalar").exists()


def test_child_distinguishes_local_failure_from_integrity_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(batch, "require_plan", lambda: {})

    def corrupt_artifact() -> None:
        raise ValueError("checksum or replay mismatch")

    assert (
        batch.execute_child(corrupt_artifact, artifact_phase=True)
        == batch.INTEGRITY_EXIT
    )
    with pytest.raises(ValueError, match="checksum"):
        batch.execute_child(corrupt_artifact, artifact_phase=False)

    def unavailable() -> None:
        raise batch.TrainingUnavailable("retained failed training")

    with pytest.raises(batch.TrainingUnavailable):
        batch.execute_child(unavailable, artifact_phase=True)
    monkeypatch.setattr(batch, "require_plan", corrupt_artifact)
    assert (
        batch.execute_child(lambda: None, artifact_phase=False) == batch.INTEGRITY_EXIT
    )


@pytest.mark.parametrize("actor", ["initial-absolute", "initial-residual"])
def test_both_initials_require_their_own_closed_training_and_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, actor: str
) -> None:
    from plant_backend.rl.v2.training import research_evaluation
    from plant_backend.rl.v2.training import research_residual_evaluation

    monkeypatch.setattr(batch, "ROOT", tmp_path)
    (tmp_path / "receipts").mkdir()
    directory = tmp_path / "evaluations" / f"9990850-{actor}"
    directory.mkdir(parents=True)
    arm = batch.INITIAL_SOURCES[actor]
    publish_json(
        tmp_path / "receipts" / f"train-{arm.name}.json",
        {
            "stage": "_train",
            "name": f"train-{arm.name}",
            "codes": [],
            "closed_at": "fixture-closed",
            "process": {
                "exit_code": 0,
                "started": True,
                "command": [
                    batch.sys.executable,
                    "-m",
                    batch.MODULE,
                    "_train",
                    "--cell",
                    arm.name,
                ],
            },
        },
    )
    calls = []
    sentinel = object()

    def initial(path: Path, config: object, selected: object) -> object:
        assert path == tmp_path / "training" / arm.name
        assert selected is arm
        return sentinel

    def native(config: object, **kwargs: object) -> dict[str, object]:
        calls.append(("native", kwargs))
        assert kwargs["policy"] is sentinel
        return {"summary": {"fixture": True}}

    def residual(config: object, **kwargs: object) -> dict[str, object]:
        calls.append(("residual", kwargs))
        assert kwargs["policy"] is sentinel
        assert "actor" not in kwargs
        return {"summary": {"fixture": True}}

    monkeypatch.setattr(batch, "load_initial", initial)
    monkeypatch.setattr(research_evaluation, "evaluate_episode", native)
    monkeypatch.setattr(
        research_residual_evaluation, "evaluate_residual_episode", residual
    )
    batch.evaluate_cell(actor, 9_990_850)
    assert calls[0][0] == ("residual" if arm.residual else "native")
    assert calls[0][1]["policy_id"] == actor
    assert (directory / "episode.json.gz").is_file()
    assert (directory / "summary.json").is_file()
    with pytest.raises(FileExistsError):
        batch.evaluate_cell(actor, 9_990_850)


@pytest.mark.parametrize("missing", ["started", "closed_at"])
def test_training_receipt_must_have_started_and_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    (tmp_path / "receipts").mkdir()
    receipt = {
        "stage": "_train",
        "name": "train-A-absolute-pi",
        "codes": [],
        "closed_at": "fixture-closed",
        "process": {
            "exit_code": 0,
            "started": True,
            "command": [
                batch.sys.executable,
                "-m",
                batch.MODULE,
                "_train",
                "--cell",
                "A-absolute-pi",
            ],
        },
    }
    if missing == "started":
        del receipt["process"]["started"]
    else:
        del receipt["closed_at"]
    publish_json(tmp_path / "receipts" / "train-A-absolute-pi.json", receipt)
    with pytest.raises(ValueError, match="receipt"):
        batch.require_successful_training(batch.ARMS[0])


def test_plain_zip_identity_and_space_mismatch_rejected_before_deserialization(
    tmp_path: Path,
) -> None:
    from copy import deepcopy
    import io
    import json
    import zipfile

    identity = batch.policy_identity(batch.template(tmp_path), batch.ARMS[2])
    state = {
        "schema": "fixture",
        "sha256": "fixture-digest",
        "payload": {
            "ppo_configuration": {
                "observation_space": identity["observation_space"],
                "action_space": identity["action_space"],
            }
        },
    }

    def archive(the_identity: object, the_state: dict[str, object]) -> Path:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as stored:
            stored.writestr(
                "data",
                json.dumps(
                    {
                        "research_identity": the_identity,
                        "_embedded_constraint_state_schema": the_state["schema"],
                        "_embedded_constraint_state_sha256": the_state["sha256"],
                        "_embedded_constraint_state_payload_json": json.dumps(
                            the_state["payload"]
                        ),
                    }
                ),
            )
        path = tmp_path / "fabricated.zip"
        path.write_bytes(buffer.getvalue())
        return path

    batch.preflight_checkpoint(archive(identity, state), identity, state)
    wrong = deepcopy(identity)
    wrong["observation_size"] = 135
    with pytest.raises(ValueError, match="interface"):
        batch.preflight_checkpoint(archive(wrong, state), identity, state)
    wrong_state = deepcopy(state)
    wrong_state["payload"]["ppo_configuration"]["observation_space"]["shape"] = [135]
    with pytest.raises(ValueError, match="space"):
        batch.preflight_checkpoint(
            archive(identity, wrong_state), identity, wrong_state
        )
    with pytest.raises(ValueError, match="endpoint"):
        batch.preflight_checkpoint(archive(identity, wrong_state), identity, state)


def test_model_space_check_rejects_a_mislabeled_residual_policy(tmp_path: Path) -> None:
    from plant_backend.rl.checkpoints import IncompatibleCheckpointError

    config = batch.template(tmp_path)
    env = batch.control_environment(config, batch.ARMS[0])
    try:
        mislabeled = SimpleNamespace(
            observation_space=env.observation_space, action_space=env.action_space
        )
        with pytest.raises(IncompatibleCheckpointError, match="shape"):
            batch.verify_model_spaces(mislabeled, config, batch.ARMS[2])
    finally:
        env.close()


@pytest.mark.parametrize("learning_fails", [False, True])
def test_post_learning_failure_is_integrity_but_learning_failure_is_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, learning_fails: bool
) -> None:
    from plant_backend.rl.v2.training import pilot

    events = []

    def learn(**kwargs: object) -> None:
        events.append("learn")
        if learning_fails:
            raise ValueError("ordinary numerical learning failure")

    def endpoint(*args: object) -> None:
        events.append("endpoint")
        raise ValueError("wrong exact endpoint")

    model = SimpleNamespace(policy=SimpleNamespace(state_dict=lambda: {}), learn=learn)
    vector = SimpleNamespace(close=lambda: events.append("closed"))
    monkeypatch.setattr(
        batch, "make_learner", lambda *args, **kwargs: (model, object(), vector)
    )
    monkeypatch.setattr(batch, "training_callback", lambda *args: object())
    monkeypatch.setattr(pilot, "validate_endpoint", endpoint)
    error = ValueError if learning_fails else batch.TrainingIntegrityError
    with pytest.raises(error):
        batch.train_cell(
            tmp_path, batch.template(tmp_path), batch.ARMS[0], seed_base=9_990_870
        )
    assert events == (
        ["learn", "closed"] if learning_fails else ["learn", "endpoint", "closed"]
    )
    assert not (tmp_path / "result.json").exists()
    monkeypatch.setattr(batch, "require_plan", lambda: {})

    def integrity() -> None:
        raise batch.TrainingIntegrityError("endpoint verification failed")

    assert batch.execute_child(integrity, artifact_phase=False) == batch.INTEGRITY_EXIT


def test_wrong_training_command_cannot_admit_a_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    (tmp_path / "receipts").mkdir()
    publish_json(
        tmp_path / "receipts" / "train-A-absolute-pi.json",
        {
            "stage": "_train",
            "name": "train-A-absolute-pi",
            "codes": [],
            "closed_at": "fixture-closed",
            "process": {
                "exit_code": 0,
                "started": True,
                "command": [
                    batch.sys.executable,
                    "-m",
                    batch.MODULE,
                    "_train",
                    "--cell",
                    "C-residual-pi",
                ],
            },
        },
    )
    with pytest.raises(ValueError, match="receipt"):
        batch.require_successful_training(batch.ARMS[0])


@pytest.mark.parametrize(
    "mutation", ["outer_actor", "outer_seed", "episode_seed", "summary", "policy_id"]
)
def test_cross_slot_substitution_rejected_before_physical_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    import gzip
    from plant_backend.rl.v2.training import research_evaluation

    monkeypatch.setattr(batch, "ROOT", tmp_path)
    actor, seed = "H", 9_990_860
    directory = tmp_path / "evaluations" / f"{seed}-{actor}"
    directory.mkdir(parents=True)
    episode = {"seed": seed, "summary": {"actor": "H", "policy_id": actor}}
    summary = {"actor": actor, "seed": seed, "summary": dict(episode["summary"])}
    if mutation == "outer_actor":
        summary["actor"] = "OperatingMPC"
    elif mutation == "outer_seed":
        summary["seed"] = seed + 1
    elif mutation == "episode_seed":
        episode["seed"] = seed + 1
    elif mutation == "summary":
        summary["summary"]["changed"] = True
    else:
        episode["summary"]["policy_id"] = "OperatingMPC"
        summary["summary"]["policy_id"] = "OperatingMPC"
    archive = directory / "episode.json.gz"
    archive.write_bytes(gzip.compress(canonical_bytes(episode)))
    summary["archive"] = batch.file_record(archive)
    publish_json(directory / "summary.json", summary)
    monkeypatch.setattr(
        research_evaluation,
        "verify_episode_replay",
        lambda *args: pytest.fail("physical replay started before slot validation"),
    )
    with pytest.raises(ValueError, match="slot identity"):
        batch.replay_cell(actor, seed)
    assert not (directory / "replay.json").exists()


def test_allocator_custody_failure_closes_receipt_and_aborts_campaign_without_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "batch"
    monkeypatch.setattr(batch, "ROOT", root)
    runtime = SimpleNamespace(git_head="a" * 40, to_dict=lambda: {"fixture": True})
    monkeypatch.setattr(batch, "capture", lambda **kwargs: runtime)
    monkeypatch.setattr(batch, "require_plan", lambda: {})
    monkeypatch.setattr(
        batch.ppo,
        "make_env",
        lambda *args: pytest.fail("fixture must not construct or reset an environment"),
    )
    monkeypatch.setattr(
        batch,
        "specification",
        lambda root: {"schema": batch.SCHEMA, "fabricated": True},
    )
    calls = []

    def custody_error() -> None:
        raise batch.EpisodeSeedCustodyError("injected unauthorized reset")

    def process(*args: object, **kwargs: object) -> tuple[dict[str, object], set[str]]:
        calls.append(args)
        exit_code = batch.execute_child(custody_error, artifact_phase=False)
        assert exit_code == batch.INTEGRITY_EXIT
        return {"exit_code": exit_code}, {"child_nonzero_exit"}

    monkeypatch.setattr(batch, "capture_process", process)
    gate = tmp_path / "gate.json"
    publish_json(gate, {"passed": True, "source_commit": runtime.git_head})
    report = batch.run(gate, authorized=True)
    assert report["status"] == "aborted"
    assert len(calls) == 1
    assert (root / "receipts" / "train-A-absolute-pi.json").is_file()
    assert not (root / "training" / "B-absolute-scalar").exists()
    assert not list((root / "evaluations").iterdir())

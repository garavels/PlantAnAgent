"""Batch003 custody and tiny scalar learning; no production MPC solves or resets.

Physical fixtures use only seeds 9_991_000..9_991_099.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from plant_backend.rl.v2.training import research_mpc_residual_batch as batch
from plant_backend.rl.v2.training import ppo
from plant_backend.rl.v2.training.custody_json import canonical_bytes
from plant_backend.rl.v2.training.custody_io import publish_json


@pytest.fixture(autouse=True)
def forbid_production_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    from plant_backend import operational_mpc

    monkeypatch.setattr(
        operational_mpc.OperationalMPCController,
        "build_model",
        lambda *args, **kwargs: pytest.fail("production MPC model construction"),
    )
    monkeypatch.setattr(
        operational_mpc,
        "solve_economic_mpc_model",
        lambda *args, **kwargs: pytest.fail("production MPC solve"),
    )


@pytest.fixture
def cheap_prior(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain real MPC fallback, mapping and physics; fabricate only the solve."""
    from plant_backend import operational_mpc
    from plant_backend.rl.v2.economic_mpc_schema import EconomicMPCPlanError
    from plant_backend.rl.v2.mpc_schema import domain_sha256

    def model(controller, observation, **kwargs):
        return SimpleNamespace(
            sha256=domain_sha256("runner-controlled-mpc-fixture", observation.sha256),
            horizon_intervals=min(
                96, controller.settings.task_interval_count - controller._next_decision
            ),
        )

    def solve(*args, **kwargs):
        raise EconomicMPCPlanError("runner controlled solver fixture")

    monkeypatch.setattr(operational_mpc.OperationalMPCController, "build_model", model)
    monkeypatch.setattr(operational_mpc, "solve_economic_mpc_model", solve)


def test_fixed_screen_and_untouched_task_contract() -> None:
    from plant_backend.rl.v2.training.registered_contract import registered_config

    spec = batch.specification(batch.ROOT)
    assert len(spec["arms"]) == 1
    assert batch.TOTAL_STEPS == 32768
    assert batch.TRAIN_SEED_BASE + 51 < min(batch.DEV_SEEDS)
    original = registered_config(
        batch.ROOT / "hypothetical-registered", batch.SCALAR, 101
    )
    candidate = batch.template(batch.ROOT)
    assert ppo.policy_contract(original) == ppo.policy_contract(candidate)
    assert candidate.ppo == original.ppo
    assert spec["registered_evaluation_eligible"] is False
    assert batch.TRAIN_SEED_BASE == 9_620_000
    assert batch.TRAIN_SEED_LAST == 9_620_051
    assert batch.DEV_SEEDS == tuple(range(9_700_200, 9_700_208))
    assert len(batch.ACTORS) == 6
    assert batch.TRAIN_TIMEOUT == 21600
    assert batch.EVAL_TIMEOUT == 600
    rollout_size = candidate.n_envs * candidate.ppo.n_steps
    assert candidate.total_timesteps // rollout_size == spec["training_rollouts"] == 4
    assert (
        candidate.total_timesteps * candidate.ppo.n_epochs // candidate.ppo.batch_size
        == spec["minibatch_optimizer_steps"]
        == 1280
    )
    decisions = candidate.episode.decision_steps(candidate.plant)
    episodes_per_env = candidate.total_timesteps // candidate.n_envs // decisions + 1
    assert batch.TRAIN_SEED_BASE + 4 * episodes_per_env - 1 == batch.TRAIN_SEED_LAST
    from plant_backend.rl.v2.training import research_effort_batch

    assert research_effort_batch.TOTAL_STEPS == 131072
    assert research_effort_batch.TRAIN_TIMEOUT == 10800
    assert len(research_effort_batch.ARMS) == 4


def test_optimizer_scaling_preserves_physics_constraints_and_raw_economics(
    tmp_path: Path,
    cheap_prior: None,
) -> None:
    config = batch.template(tmp_path)
    raw = batch.control_environment(config, batch.ARMS[0])
    scaled = batch.OptimizerRewardScale(
        batch.control_environment(config, batch.ARMS[0]), 0.001
    )
    try:
        a, ai = raw.reset(seed=9_991_000)
        b, bi = scaled.reset(seed=9_991_000)
        assert np.array_equal(a, b)
        assert canonical_bytes(ai) == canonical_bytes(bi)
        for _ in range(96):
            action = np.array([-0.3, -0.3, -0.3, 1], dtype=np.float32)
            a, ar, at, ax, ai = raw.step(action)
            b, br, bt, bx, bi = scaled.step(action)
            assert np.array_equal(a, b)
            assert br == ar * 0.001
            assert (at, ax) == (bt, bx)
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
    from stable_baselines3.common.monitor import Monitor
    from plant_backend.rl.v2.training.episode_allocator import SeedCustodyWrapperV1
    from plant_backend.rl.v2.training.research_mpc_residual import (
        ResearchMPCResidualWrapper,
    )

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
    for arm in (batch.ARMS[0], batch.ARMS[0]):
        model, allocator, vector = batch.make_learner(config, arm, seed_base=9_991_010)
        try:
            assert not allocator.ledger
            assert allocator.decisions_consumed_by_environment == (0, 0, 0, 0)
            for instance in vector.envs:
                assert isinstance(instance, batch.OptimizerRewardScale)
                assert isinstance(instance.env, Monitor)
                assert isinstance(instance.env.env, SeedCustodyWrapperV1)
                assert isinstance(instance.env.env.env, ResearchMPCResidualWrapper)
            assert model.observation_space.shape == (139,)
            assert model.research_identity["representation"] == "residual-mpc"
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
    for left, right in ((0, 1),):
        assert all(
            torch.equal(snapshots[left][k], snapshots[right][k])
            for k in snapshots[left]
        )


def test_true_scaled_training_diagnostics_and_checkpoint_roundtrip(
    tmp_path: Path, cheap_prior: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    torch.set_num_threads(1)
    original = batch.template(tmp_path)
    config = replace(
        original,
        total_timesteps=128,
        ppo=replace(original.ppo, n_steps=8, batch_size=16, n_epochs=1),
    )
    arm = batch.ARMS[0]
    result = batch.train_cell(tmp_path, config, arm, seed_base=9_991_020)
    assert result["total_timesteps"] == 128
    assert len(result["constraint_state"]["payload"]["rollout_ledgers"]) == 4
    assert len(list((tmp_path / "diagnostics").glob("diagnostic-*.json"))) == 4
    assert len(list((tmp_path / "diagnostics").glob("rollout-*.json"))) == 4
    assert len(list((tmp_path / "allocator").glob("rollout-*.json"))) == 4
    assert len(list((tmp_path / "allocator").glob("reset-*.json"))) == 4
    assert result["registered_evaluation_eligible"] is False
    assert result["policy_identity"]["observation_size"] == 139
    assert (
        result["checkpoint_contract_digest"]
        == result["policy_identity"]["checkpoint_contract_digest"]
    )
    assert all(c["decision_offset"] == 32 for c in result["allocator"]["cursors"])
    from copy import deepcopy
    import json
    from plant_backend.rl.v2.training.constrained_ppo import ConstrainedPPO

    tampered = deepcopy(result)
    tampered["policy_identity"]["representation"] = "residual-h"
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
    with monkeypatch.context() as admission:
        admission.setattr(
            batch.PlantCyclicMarketEnvV2Beta,
            "reset",
            lambda *args, **kwargs: pytest.fail("checkpoint admission reset the plant"),
        )
        model = batch.load_final(tmp_path, config, arm)
        initial = batch.load_initial(tmp_path, config, arm)
    for policy in (model, initial):
        action, _ = policy.predict(np.zeros(139, dtype=np.float32), deterministic=True)
        assert action.shape == (4,)
    with pytest.raises(batch.TrainingIntegrityError, match="pre-learning"):
        batch.train_cell(tmp_path, config, arm, seed_base=9_991_020)
    with (tmp_path / "initial-policy.pt").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="initial policy bytes changed"):
        batch.load_initial(tmp_path, config, arm)
    with (tmp_path / "final-policy.zip").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        batch.load_final(tmp_path, config, arm)


@pytest.mark.parametrize("training_failed", [False, True])
def test_parent_closes_failed_cells_and_keeps_every_fixed_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, training_failed: bool
) -> None:
    root = tmp_path / "batch"
    monkeypatch.setattr(batch, "ROOT", root)
    monkeypatch.setattr(batch, "DEV_SEEDS", tuple(range(9_991_080, 9_991_088)))
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
            "codes": (
                ["deliberate-local-failure"]
                if training_failed
                and ("A-mpc-residual-scalar" in name or "initial-mpc-residual" in name)
                else []
            ),
        }

    monkeypatch.setattr(batch, "closed_child", child)
    report = batch.run(gate, authorized=True)
    assert report["status"] == "closed"
    assert sum(s == "_train" for s, _ in calls) == 1
    assert sum(s == "_evaluate" for s, _ in calls) == 48
    assert sum(s == "_replay" for s, _ in calls) == (32 if training_failed else 48)
    assert sum(bool(r["codes"]) for r in report["receipts"]) == (
        17 if training_failed else 0
    )
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


@pytest.mark.parametrize("actor", ["initial-mpc-residual", "A-mpc-residual-scalar"])
def test_failed_training_cannot_be_evaluated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, actor: str
) -> None:
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    (tmp_path / "receipts").mkdir()
    directory = tmp_path / "evaluations" / f"9991080-{actor}"
    directory.mkdir(parents=True)
    arm_name = batch.ARMS[0].name
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
        batch.evaluate_cell(actor, 9_991_080)
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
    assert (root / "receipts" / "train-A-mpc-residual-scalar.json").is_file()
    assert not list((root / "evaluations").iterdir())


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


@pytest.mark.parametrize("actor", ["initial-mpc-residual", "A-mpc-residual-scalar"])
def test_both_policies_require_the_closed_training_and_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, actor: str
) -> None:
    from plant_backend.rl.v2.training import research_evaluation
    from plant_backend.rl.v2.training import research_mpc_residual_evaluation

    monkeypatch.setattr(batch, "ROOT", tmp_path)
    (tmp_path / "receipts").mkdir()
    directory = tmp_path / "evaluations" / f"9991050-{actor}"
    directory.mkdir(parents=True)
    arm = batch.ARMS[0]
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
    monkeypatch.setattr(batch, "load_final", initial)
    monkeypatch.setattr(research_evaluation, "evaluate_episode", native)
    monkeypatch.setattr(
        research_mpc_residual_evaluation, "evaluate_residual_episode", residual
    )
    batch.evaluate_cell(actor, 9_991_050)
    assert calls[0][0] == "residual"
    assert calls[0][1]["policy_id"] == actor
    assert (directory / "episode.json.gz").is_file()
    assert (directory / "summary.json").is_file()
    with pytest.raises(FileExistsError):
        batch.evaluate_cell(actor, 9_991_050)


@pytest.mark.parametrize("missing", ["started", "closed_at"])
def test_training_receipt_must_have_started_and_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    (tmp_path / "receipts").mkdir()
    receipt = {
        "stage": "_train",
        "name": "train-A-mpc-residual-scalar",
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
                "A-mpc-residual-scalar",
            ],
        },
    }
    if missing == "started":
        del receipt["process"]["started"]
    else:
        del receipt["closed_at"]
    publish_json(tmp_path / "receipts" / "train-A-mpc-residual-scalar.json", receipt)
    with pytest.raises(ValueError, match="receipt"):
        batch.require_successful_training(batch.ARMS[0])


def test_plain_zip_identity_and_space_mismatch_rejected_before_deserialization(
    tmp_path: Path,
) -> None:
    from copy import deepcopy
    import io
    import json
    import zipfile

    identity = batch.policy_identity(batch.template(tmp_path), batch.ARMS[0])
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
    from plant_backend.rl.v2.training import research_effort_batch

    h_identity = research_effort_batch.policy_identity(
        batch.template(tmp_path), research_effort_batch.ARMS[3]
    )
    assert h_identity["observation_size"] == identity["observation_size"] == 139
    with pytest.raises(ValueError, match="interface"):
        batch.preflight_checkpoint(archive(h_identity, state), identity, state)
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
    env = batch.ppo.make_env(config)
    try:
        mislabeled = SimpleNamespace(
            observation_space=env.observation_space, action_space=env.action_space
        )
        with pytest.raises(IncompatibleCheckpointError, match="shape"):
            batch.verify_model_spaces(mislabeled, config, batch.ARMS[0])
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

    model = SimpleNamespace(
        policy=SimpleNamespace(state_dict=lambda: {}),
        learn=learn,
        num_timesteps=0,
        rollout_ledgers=(),
    )
    vector = SimpleNamespace(close=lambda: events.append("closed"))
    monkeypatch.setattr(
        batch,
        "make_learner",
        lambda *args, **kwargs: (
            model,
            batch.EpisodeSeedAllocatorV1.fixture(base_seed=9_991_070),
            vector,
        ),
    )
    monkeypatch.setattr(batch, "cell_callback", lambda *args: object())
    monkeypatch.setattr(pilot, "validate_endpoint", endpoint)
    error = ValueError if learning_fails else batch.TrainingIntegrityError
    with pytest.raises(error):
        batch.train_cell(
            tmp_path, batch.template(tmp_path), batch.ARMS[0], seed_base=9_991_070
        )
    assert events == (
        ["learn", "closed"] if learning_fails else ["learn", "endpoint", "closed"]
    )
    assert not (tmp_path / "result.json").exists()
    import json

    failure = json.loads((tmp_path / "training-failure.json").read_text())
    assert failure["successful_endpoint"] is False
    assert failure["allocator_prefix"]["ledger"] == []
    assert failure["rollout_ledgers"] == []
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
        tmp_path / "receipts" / "train-A-mpc-residual-scalar.json",
        {
            "stage": "_train",
            "name": "train-A-mpc-residual-scalar",
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
                    "wrong-arm",
                ],
            },
        },
    )
    with pytest.raises(ValueError, match="receipt"):
        batch.require_successful_training(batch.ARMS[0])


@pytest.mark.parametrize(
    "mutation",
    [
        "outer_actor",
        "outer_seed",
        "episode_seed",
        "summary",
        "policy_id",
        "inner_actor",
    ],
)
def test_cross_slot_substitution_rejected_before_physical_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    import gzip
    from plant_backend.rl.v2.training import research_evaluation

    monkeypatch.setattr(batch, "ROOT", tmp_path)
    actor, seed = "H", 9_991_060
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
    elif mutation == "inner_actor":
        episode["summary"]["actor"] = "OperatingMPC"
        summary["summary"]["actor"] = "OperatingMPC"
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


@pytest.mark.parametrize(
    "failure_kind", ["allocator", "mpc", "callback", "endpoint", "claim"]
)
def test_custody_failure_closes_receipt_and_aborts_campaign_without_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_kind: str
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
        if failure_kind == "claim":
            original_publish = batch.publish_json

            def publish(path, value):
                if path.name == "attempt.json":
                    raise FileExistsError("injected consumed attempt claim")
                return original_publish(path, value)

            monkeypatch.setattr(batch, "publish_json", publish)
            monkeypatch.setattr(batch, "policy_identity", lambda *args: {})
            batch.train_cell(
                root / "training" / batch.ARMS[0].name,
                batch.template(root),
                batch.ARMS[0],
                seed_base=9_991_095,
            )
        elif failure_kind == "callback":
            from stable_baselines3.common.callbacks import BaseCallback

            class BrokenRecorder(BaseCallback):
                def _on_step(self) -> bool:
                    return True

                def _on_rollout_end(self) -> None:
                    raise OSError("injected evidence publication failure")

            monkeypatch.setattr(
                batch, "training_callback", lambda *args: BrokenRecorder()
            )
            allocator = batch.EpisodeSeedAllocatorV1.fixture(base_seed=9_991_095)
            batch.cell_callback(root, allocator).on_rollout_end()
        else:
            error = {
                "allocator": batch.EpisodeSeedCustodyError,
                "mpc": batch.MPCResidualIntegrityError,
                "endpoint": batch.TrainingIntegrityError,
            }[failure_kind]
            raise error("injected custody loss")

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
    assert (root / "receipts" / "train-A-mpc-residual-scalar.json").is_file()
    assert not list((root / "evaluations").iterdir())


@pytest.mark.parametrize("kind", ["allocator", "mpc", "endpoint"])
def test_all_typed_integrity_failures_abort_even_during_learning(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    monkeypatch.setattr(batch, "require_plan", lambda: {})
    errors = {
        "allocator": batch.EpisodeSeedCustodyError,
        "mpc": batch.MPCResidualIntegrityError,
        "endpoint": batch.TrainingIntegrityError,
    }

    def action() -> None:
        raise errors[kind]("controlled integrity failure")

    assert batch.execute_child(action, artifact_phase=False) == batch.INTEGRITY_EXIT


def test_callback_publication_failure_is_integrity_not_numerical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stable_baselines3.common.callbacks import BaseCallback

    class FailingEvidence(BaseCallback):
        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            raise OSError("controlled evidence write failure")

    monkeypatch.setattr(batch, "training_callback", lambda *args: FailingEvidence())
    allocator = batch.EpisodeSeedAllocatorV1.fixture(base_seed=9_991_090)
    callback = batch.cell_callback(tmp_path, allocator)
    with pytest.raises(batch.TrainingIntegrityError, match="evidence callback"):
        callback.on_rollout_end()
    assert not allocator.ledger


def test_failed_allocator_prefix_is_retained_without_asserting_snapshot_validity(
    tmp_path: Path,
) -> None:
    import json

    allocator = batch.EpisodeSeedAllocatorV1.fixture(base_seed=9_991_091)
    reason = batch.EpisodeSeedCustodyError("controlled failure before any reset")
    allocator.fail_closed(reason)
    with pytest.raises(batch.EpisodeSeedCustodyError):
        allocator.snapshot()
    batch.retain_training_failure(
        tmp_path,
        SimpleNamespace(num_timesteps=0, rollout_ledgers=()),
        allocator,
        reason,
    )
    failure = json.loads((tmp_path / "training-failure.json").read_text())
    assert failure["failure"]["type"] == "EpisodeSeedCustodyError"
    assert failure["allocator_prefix"]["successful_endpoint"] is False
    assert failure["allocator_prefix"]["decisions_consumed_by_environment"] == [0] * 4
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("changed", ["specification", "source", "host", "gate"])
def test_plan_rejects_runtime_host_gate_and_specification_drift_without_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    from plant_backend.rl.v2.training import capacity_benchmark

    monkeypatch.setattr(batch, "ROOT", tmp_path)
    monkeypatch.setattr(batch, "specification", lambda path: {"fixture": "fixed"})
    monkeypatch.setattr(batch, "capture", lambda **kwargs: {"source": "fixed"})
    monkeypatch.setattr(
        batch, "BetaTrainingRuntimeProvenanceV1", SimpleNamespace(from_dict=lambda x: x)
    )

    def verify(saved, current):
        if saved != current:
            raise ValueError("source changed")

    monkeypatch.setattr(batch, "verify_unchanged", verify)
    monkeypatch.setattr(capacity_benchmark, "_host_identity", lambda: {"host": "fixed"})
    gate = tmp_path / "gate.json"
    publish_json(gate, {"passed": True})
    plan = {
        "specification": {"fixture": "fixed"},
        "runtime": {"source": "fixed"},
        "host": {"host": "fixed"},
        "gate": batch.file_record(gate),
    }
    if changed == "specification":
        plan["specification"] = {"fixture": "changed"}
    elif changed == "source":
        plan["runtime"] = {"source": "changed"}
    elif changed == "host":
        plan["host"] = {"host": "changed"}
    else:
        gate.write_text("changed")
    publish_json(tmp_path / "plan.json", plan)
    with pytest.raises(ValueError, match="changed"):
        batch.require_plan()


@pytest.mark.parametrize("post_child_drift", [False, True])
def test_closed_child_checks_source_before_and_after_and_retains_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, post_child_drift: bool
) -> None:
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    (tmp_path / "receipts").mkdir()
    (tmp_path / "logs").mkdir()
    events = []

    def require():
        events.append("source")
        if post_child_drift and events.count("source") == 2:
            raise ValueError("source changed after child")
        return {}

    def process(command, **kwargs):
        events.append("child")
        assert command == [batch.sys.executable, "-m", batch.MODULE, "_train"]
        assert kwargs["timeout_seconds"] == 21600
        return {"exit_code": 0, "command": command}, set()

    monkeypatch.setattr(batch, "require_plan", require)
    monkeypatch.setattr(batch, "capture_process", process)
    if post_child_drift:
        with pytest.raises(ValueError, match="after child"):
            batch.closed_child("_train", "fixture", batch.TRAIN_TIMEOUT, [])
    else:
        assert (
            batch.closed_child("_train", "fixture", batch.TRAIN_TIMEOUT, [])["codes"]
            == []
        )
    assert events == ["source", "child", "source"]
    assert (tmp_path / "receipts" / "fixture.json").is_file()


def test_low_space_prevents_child_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    monkeypatch.setattr(batch, "require_plan", lambda: {})
    monkeypatch.setattr(
        batch.shutil, "disk_usage", lambda path: SimpleNamespace(free=0)
    )
    monkeypatch.setattr(
        batch, "capture_process", lambda *args, **kwargs: pytest.fail("low-space child")
    )
    with pytest.raises(RuntimeError, match="free before child"):
        batch.closed_child("_train", "fixture", batch.TRAIN_TIMEOUT, [])


@pytest.mark.parametrize("phase", ["attempt", "directory", "learner"])
def test_prelearning_claim_publication_and_interface_failures_are_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    monkeypatch.setattr(batch, "policy_identity", lambda *args: {})

    def failed(*args, **kwargs):
        raise OSError(f"controlled {phase} admission failure")

    if phase == "attempt":
        monkeypatch.setattr(batch, "publish_json", failed)
    elif phase == "directory":
        monkeypatch.setattr(batch, "mkdir_exclusive", failed)
    else:
        monkeypatch.setattr(batch, "make_learner", failed)
    with pytest.raises(batch.TrainingIntegrityError, match="pre-learning") as error:
        batch.train_cell(
            tmp_path, batch.template(tmp_path), batch.ARMS[0], seed_base=9_991_096
        )
    assert isinstance(error.value.__cause__, OSError)
    assert not (tmp_path / "result.json").exists()

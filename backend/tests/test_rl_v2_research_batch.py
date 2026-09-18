"""Adaptive batch conditioning, initialization, custody and true learner tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from plant_backend.rl.v2.training import research_batch as batch
from plant_backend.rl.v2.training import ppo
from plant_backend.rl.v2.training.custody_json import canonical_bytes
from plant_backend.rl.v2.training.custody_io import publish_json


def test_fixed_screen_and_untouched_task_contract() -> None:
    from plant_backend.rl.v2.training.registered_contract import registered_config

    spec = batch.specification(batch.ROOT)
    assert len(spec["arms"]) == 5
    assert batch.TOTAL_STEPS == 131072
    assert batch.TRAIN_SEED_BASE + 195 < min(batch.DEV_SEEDS)
    original = registered_config(batch.ROOT / "hypothetical-registered", batch.PI, 101)
    candidate = batch.template(batch.ROOT)
    assert ppo.policy_contract(original) == ppo.policy_contract(candidate)
    assert candidate.ppo == original.ppo
    assert spec["registered_evaluation_eligible"] is False


def test_optimizer_scaling_preserves_physics_constraints_and_raw_economics(
    tmp_path: Path,
) -> None:
    config = batch.template(tmp_path)
    raw = ppo.make_env(config)
    scaled = batch.OptimizerRewardScale(ppo.make_env(config), 0.001)
    try:
        a, ai = raw.reset(seed=9_990_200)
        b, bi = scaled.reset(seed=9_990_200)
        assert np.array_equal(a, b)
        assert canonical_bytes(ai) == canonical_bytes(bi)
        for _ in range(96):
            action = np.array([-0.3, -0.3, -0.3, 1], dtype=np.float32)
            a, ar, at, ax, ai = raw.step(action)
            b, br, bt, bx, bi = scaled.step(action)
            assert np.array_equal(a, b)
            assert br == ar * 0.001
            assert (at, ax) == (bt, bx)
            assert canonical_bytes(ai) == canonical_bytes(bi)
        assert ai["cycle_constraint_transition"]["cycle_settled"] is True
    finally:
        raw.close()
        scaled.close()


def test_matched_initialization_and_only_logstd_changes(tmp_path: Path) -> None:
    import torch

    torch.set_num_threads(1)
    snapshots = []
    config = batch.template(tmp_path)
    for arm in batch.ARMS:
        model, allocator, vector = batch.make_learner(config, arm, seed_base=9_990_300)
        try:
            assert not allocator.ledger
            snapshots.append(
                {
                    k: v.detach().clone()
                    for k, v in model.policy.state_dict().items()
                    if not k.startswith("cost_critic.")
                }
            )
        finally:
            vector.close()
    for current in snapshots[1:4]:
        assert all(torch.equal(current[k], snapshots[0][k]) for k in current)
    for k, v in snapshots[4].items():
        if k == "log_std":
            assert torch.allclose(v, torch.full_like(v, float(np.log(0.2))))
        else:
            assert torch.equal(v, snapshots[1][k])


@pytest.mark.parametrize("index", [1, 3, 4])
def test_true_scaled_training_diagnostics_and_checkpoint_roundtrip(
    tmp_path: Path, index: int
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
    result = batch.train_cell(tmp_path, config, arm, seed_base=9_990_400)
    assert result["total_timesteps"] == 64
    assert len(result["constraint_state"]["payload"]["rollout_ledgers"]) == 2
    assert len(list((tmp_path / "diagnostics").glob("diagnostic-*.json"))) == 2
    assert result["registered_evaluation_eligible"] is False
    model = batch.load_final(tmp_path, config, arm)
    initial = batch.load_initial(tmp_path, config, arm)
    for policy in (model, initial):
        action, _ = policy.predict(np.zeros(135, dtype=np.float32), deterministic=True)
        assert action.shape == (4,)
    with pytest.raises(FileExistsError):
        batch.train_cell(tmp_path, config, arm, seed_base=9_990_400)
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
            "codes": ["deliberate-local-failure"] if "A-pi-raw" in name else [],
        }

    monkeypatch.setattr(batch, "closed_child", child)
    report = batch.run(gate, authorized=True)
    assert report["status"] == "closed"
    assert sum(s == "_train" for s, _ in calls) == 5
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


@pytest.mark.parametrize("actor", ["initial", "A-pi-raw"])
def test_failed_training_cannot_be_evaluated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, actor: str
) -> None:
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    (tmp_path / "receipts").mkdir()
    directory = tmp_path / "evaluations" / f"9700000-{actor}"
    directory.mkdir(parents=True)
    publish_json(
        tmp_path / "receipts" / "train-A-pi-raw.json",
        {
            "stage": "_train",
            "name": "train-A-pi-raw",
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
    assert (root / "receipts" / "train-A-pi-raw.json").is_file()
    assert not (root / "training" / "B-pi-scaled").exists()


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

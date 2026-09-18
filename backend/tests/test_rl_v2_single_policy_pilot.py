"""Pilot isolation, failure retention and real learner/checkpoint integration."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from plant_backend.rl.v2.training import pilot
from plant_backend.rl.v2.training.config import BetaRLTrainingRunV1
from plant_backend.rl.v2.training.custody_io import publish_json


def test_plan_keeps_long_pilot_separate_from_bounded_fixture_and_registered() -> None:
    spec = pilot.specification(pilot.ROOT)
    assert spec["run_kind"] == "exploratory_pilot"
    assert spec["total_timesteps"] == 1_048_576
    assert spec["optimizer_seed"] == 101
    assert spec["registered_evaluation_eligible"] is False
    assert spec["training_episode_seed_base"] == 9_500_000
    assert spec["training_episode_seed_last"] < min(spec["evaluation_seeds"])
    template = pilot.environment_template(pilot.ROOT)
    assert template.beta_rl is None
    with pytest.raises(ValueError, match="8,192"):
        replace(template, beta_rl=BetaRLTrainingRunV1()).validate()
    with pytest.raises(ValueError, match="run_kind"):
        BetaRLTrainingRunV1(run_kind="exploratory_pilot").validate()


def test_no_authorization_means_no_capture_or_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(**kwargs: object) -> None:
        pytest.fail("runtime capture should not precede authorization assertion")

    monkeypatch.setattr(pilot, "capture", forbidden)
    with pytest.raises(ValueError, match="authorization"):
        pilot.run(tmp_path / "absent", tmp_path / "gate.json", authorized=False)
    assert not (tmp_path / "absent").exists()


def test_training_core_real_learn_reload_and_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("stable_baselines3")
    import torch
    from plant_backend.rl.v2.training.constrained_ppo import ConstrainedPPO

    torch.set_num_threads(1)
    root = tmp_path / "learning-fixture"
    root.mkdir()
    for name in ("artifacts", "rollouts", "evaluation"):
        (root / name).mkdir()
    original = pilot.environment_template(root)
    config = replace(
        original,
        total_timesteps=64,
        ppo=replace(original.ppo, n_steps=8, batch_size=16, n_epochs=1),
    )
    result = pilot.train_model(root, config, base_seed=9_990_000)
    payload = result["constraint_state"]["payload"]
    assert payload["num_timesteps"] == 64
    assert len(payload["rollout_ledgers"]) == 2
    assert payload["pending_rollout"] is False
    assert result["registered_evaluation_eligible"] is False
    assert [
        row["episode_seed"] for row in result["allocator"]["ledger"]["entries"]
    ] == list(range(9_990_000, 9_990_004))
    assert len(list((root / "rollouts").glob("*.json"))) == 2
    final = ConstrainedPPO.load(root / "artifacts" / "final-policy.zip", device="cpu")
    initial = pilot.load_initial_policy(root / "artifacts" / "initial-policy.pt", final)
    import numpy as np

    observation = np.zeros(135, dtype=np.float32)
    for policy in (initial, final):
        action, _ = policy.predict(observation, deterministic=True)
        assert action.shape == (4,)
        assert np.isfinite(action).all()
    assert any(
        not torch.equal(initial.state_dict()[key], value)
        for key, value in final.policy.state_dict().items()
    )
    assert not (root / "artifacts" / "metadata.json").exists()
    # One root cannot silently replace its already completed rollout history.
    with pytest.raises(FileExistsError):
        publish_json(root / "rollouts" / "001.json", {})
    # Exercise the complete post-training exporter with small synthetic episode
    # records; no production pilot scenario is opened by this integration test.
    monkeypatch.setattr(pilot, "environment_template", lambda root: config)
    monkeypatch.setattr(pilot, "STEPS", 64)
    calls = []

    def episode(
        config: object, *, actor: str, policy: object, seed: int
    ) -> dict[str, object]:
        calls.append((seed, actor))
        return {
            "actor": actor,
            "seed": seed,
            "history": [],
            "manifest": {},
            "failure": None,
        }

    monkeypatch.setattr(pilot, "evaluate_episode", episode)
    report = pilot.evaluate(root)
    assert calls == [
        (seed, actor) for seed in pilot.EVALUATION_SEEDS for actor in pilot.ACTORS
    ]
    assert len(report["cells"]) == 12
    assert len(list((root / "evaluation").glob("*.gz"))) == 12
    with (root / "artifacts" / "initial-policy.pt").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        pilot.evaluate(root)
    assert len(calls) == 12


def test_policy_failure_keeps_real_physics_prefix_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    seed = 9_990_020
    monkeypatch.setattr(pilot, "EVALUATION_SEEDS", (seed,))

    class FailingPolicy:
        count = 0

        def predict(
            self, observation: object, deterministic: bool
        ) -> tuple[object, None]:
            self.count += 1
            if self.count == 3:
                raise RuntimeError("deliberate policy fixture failure")
            return np.zeros(4, dtype=np.float32), None

    result = pilot.evaluate_episode(
        pilot.environment_template(tmp_path),
        actor="final",
        policy=FailingPolicy(),
        seed=seed,
    )
    assert result["decisions"] == 2
    assert result["failure"]["type"] == "RuntimeError"
    assert result["execution_complete"] is False
    assert result["registered_hard_safety_gate_performed"] is False
    assert result["manifest"] is not None
    assert len(result["history"][0]["observation"]) == 135
    assert len(result["history"][0]["action"]) == 4
    assert pilot.canonical_bytes(
        json.loads(pilot.canonical_bytes(result))
    ) == pilot.canonical_bytes(result)


def test_heuristic_full_fixture_serializes_all_daily_settlements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = 9_990_030
    monkeypatch.setattr(pilot, "EVALUATION_SEEDS", (seed,))
    result = pilot.evaluate_episode(
        pilot.environment_template(tmp_path), actor="H", policy=None, seed=seed
    )
    assert result["failure"] is None
    assert result["execution_complete"] is True
    assert result["decisions"] == 672
    assert len(result["cycle_settlements"]) == 7
    assert result["physical_metrics"]["physical_steps"] == 10_080
    assert result["registered_hard_safety_gate_performed"] is False
    assert pilot.canonical_bytes(
        json.loads(pilot.canonical_bytes(result))
    ) == pilot.canonical_bytes(result)


def test_evaluation_rejects_protected_seeds_before_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("protected evaluation must fail before constructing the task")

    monkeypatch.setattr(pilot.ppo, "make_env", forbidden)
    for seed in (1_000_000, 2_000_000, 3_000_000, pilot.TRAIN_BASE):
        with pytest.raises(ValueError, match="outside the pilot"):
            pilot.evaluate_episode(
                pilot.environment_template(tmp_path),
                actor="final",
                policy=None,
                seed=seed,
            )


@pytest.mark.parametrize("failure_kind", ["none", "training", "mutation"])
def test_parent_bounds_children_retains_failure_and_never_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_kind: str
) -> None:
    root = tmp_path / "pilot"
    monkeypatch.setattr(pilot, "ROOT", root)
    runtime = SimpleNamespace(git_head="a" * 40, to_dict=lambda: {"fixture": True})
    monkeypatch.setattr(pilot, "capture", lambda **kwargs: runtime)
    gate = tmp_path / "gate.json"
    publish_json(gate, {"source_commit": runtime.git_head, "passed": True})
    calls = []

    def admit(path: Path) -> None:
        if failure_kind == "mutation" and calls:
            raise ValueError("source changed")

    monkeypatch.setattr(pilot, "admit_plan", admit)

    def child(
        command: list[str], **kwargs: object
    ) -> tuple[dict[str, object], set[str]]:
        assert command[2] == pilot.MODULE
        stage = command[-1]
        calls.append(stage)
        assert kwargs["log_limit_bytes"] == 64 * 1024**2
        assert kwargs["timeout_seconds"] == (36_000 if stage == "_train" else 1_200)
        if failure_kind == "training":
            return {"exit_code": 1}, {"child_nonzero_exit"}
        filename = (
            "training-result.json" if stage == "_train" else "evaluation-result.json"
        )
        publish_json(root / filename, {"status": "completed"})
        return {"exit_code": 0}, set()

    monkeypatch.setattr(pilot, "capture_process", child)
    report = pilot.run(root, gate, authorized=True)
    assert report["status"] == ("completed" if failure_kind == "none" else "failed")
    assert calls == (["_train", "_evaluate"] if failure_kind == "none" else ["_train"])
    assert json.loads((root / "report.json").read_text()) == report
    with pytest.raises(ValueError, match="no retry"):
        pilot.run(root, gate, authorized=True)


def test_gate_must_match_source_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "pilot"
    monkeypatch.setattr(pilot, "ROOT", root)
    monkeypatch.setattr(
        pilot, "capture", lambda **kwargs: SimpleNamespace(git_head="a" * 40)
    )
    gate = tmp_path / "gate.json"
    publish_json(gate, {"source_commit": "b" * 40, "passed": True})
    with pytest.raises(ValueError, match="exact clean source"):
        pilot.run(root, gate, authorized=True)
    assert not root.exists()


def test_plan_admission_roundtrip_and_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plant_backend.rl.v2.training.runtime_provenance import capture
    from plant_backend.rl.v2.training.capacity_benchmark import _host_identity

    # Preserve real provenance parsing/rechecking; only clean-tree acquisition
    # is replaced so this fixture can run while implementation edits exist.
    runtime = capture(require_clean=False)
    root = tmp_path / "pilot"
    root.mkdir()
    gate = tmp_path / "gate.json"
    publish_json(gate, {"fixture": True})
    plan = {
        "specification": pilot.specification(root),
        "runtime": runtime.to_dict(),
        "host": _host_identity(),
        "gate": pilot.record(gate),
    }
    publish_json(root / "plan.json", plan)
    monkeypatch.setattr(pilot, "ROOT", root)
    monkeypatch.setattr(pilot, "capture", lambda **kwargs: runtime)
    admitted = pilot.admit_plan(root)
    assert admitted["specification"]["total_timesteps"] == 1_048_576
    with gate.open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="gate evidence changed"):
        pilot.admit_plan(root)


def test_child_attempt_is_claimed_once_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    monkeypatch.setattr(pilot, "configure_host", lambda: None)
    monkeypatch.setattr(pilot, "admit_plan", lambda root: {})
    calls = []
    monkeypatch.setattr(
        pilot, "train_model", lambda *args, **kwargs: calls.append(True)
    )
    assert pilot.main(["_train"]) == 0
    with pytest.raises(FileExistsError):
        pilot.main(["_train"])
    assert calls == [True]

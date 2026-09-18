"""Joint-freeze gate tests; no held-out environment is constructed or reset."""

import pytest

from docs.research.tools import evaluate_mission_mpc_heldout_v1 as launcher


def test_parent_gate_prevents_worker_creation(monkeypatch, tmp_path):
    def blocked():
        raise ValueError("Joint freeze is unavailable")

    def forbidden(*args, **kwargs):
        pytest.fail("Workers must not be created before the joint freeze passes")

    monkeypatch.setattr(launcher, "require_evaluation_freeze", blocked)
    monkeypatch.setattr(launcher, "ProcessPoolExecutor", forbidden)
    with pytest.raises(ValueError, match="unavailable"):
        launcher.run_panel(tmp_path / "unopened", workers=1)
    assert not (tmp_path / "unopened").exists()


def test_child_gate_prevents_environment_invocation(monkeypatch, tmp_path):
    monkeypatch.setattr(
        launcher,
        "verified_joint_freeze",
        lambda: {"sha256": "new-freeze", "held_out_seeds": [9930000]},
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Strict runner must not be invoked after a freeze mismatch")

    monkeypatch.setattr(launcher, "_invoke_strict_run", forbidden)
    with pytest.raises(ValueError, match="changed before child"):
        launcher.run_case(9930000, "old-freeze", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_joint_freeze_must_bind_the_launcher_itself(monkeypatch):
    monkeypatch.setattr(
        launcher,
        "require_evaluation_freeze",
        lambda: {"schema": "feasible25-comparison-freeze-v1", "sources": []},
    )
    with pytest.raises(ValueError, match="does not bind this launcher"):
        launcher.verified_joint_freeze()

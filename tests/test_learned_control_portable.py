"""Authenticate relocated evidence without rewriting its frozen provenance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from docs.research.tools.learned_control_release_v1 import portable as p
from docs.research.tools.wholeplant_ppo_v1.baselines import BaselinePolicy


def synthetic_recording():
    policy = BaselinePolicy("storage")
    evidence = p.inference(policy, "baseline", np.zeros(149, dtype=np.float32))
    commands = [{"decision_index": i, "evidence": deepcopy(evidence), "action": evidence["action"], "reward_eur": 0.0} for i in range(672)]
    witness = hashlib.sha256()
    for command in commands:
        witness.update(p.canonical(command["evidence"]))
    row = {"seed": 51_000_001, "execution_complete": True, "sources_unchanged": True, "failure": None,
           "minute_audit": {"all_physical_invariants_passed": True, "states_checked": 10081}, "economic_audit": {"passed": True},
           "decisions": 672, "physical_states": 10081, "input_action_witness_sha256": witness.hexdigest(),
           "economic_metrics": {"net_operating_value_eur": 0.0, "unmet_kg": 0.0}, "every_delivery_passed": True,
           "delivery": {"settlements": [{"due_step": (i+1)*360, "requested_kg": 180, "unmet_kg": 0} for i in range(28)]}}
    return row, {"decision_trace": commands, "minute_truth_states": []}, policy


def test_every_stored_witness_recomputed_and_tampering_rejected():
    row, record, policy = synthetic_recording()
    assert p.verify_recording(row, record, first_seed=51_000_000, policy=policy, controller_kind="baseline") == {
        "decisions": 672, "retained_states": 0, "inference_recomputed": True}
    altered = deepcopy(record)
    altered["decision_trace"][-1]["evidence"]["observation"][0] = 1
    with pytest.raises(ValueError, match="observation digest"):
        p.verify_recording(row, altered, first_seed=51_000_000)
    altered = deepcopy(record)
    altered["decision_trace"][0]["reward_eur"] = 1
    with pytest.raises(ValueError, match="reconcile"):
        p.verify_recording(row, altered, first_seed=51_000_000)
    altered = deepcopy(record)
    altered["decision_trace"].pop()
    with pytest.raises(ValueError, match="retention"):
        p.verify_recording(row, altered, first_seed=51_000_000)


def test_recomputed_policy_detects_self_consistent_forged_actions():
    row, record, policy = synthetic_recording()
    command = record["decision_trace"][0]
    command["evidence"]["action"] = [0.0]*7
    command["action"] = [0.0]*7
    with pytest.raises(ValueError, match="full inference"):
        p.verify_recording(row, record, first_seed=51_000_000, policy=policy, controller_kind="baseline")


def test_resolver_preserves_original_pins_and_rejects_changed_bytes(tmp_path):
    evidence = p.Evidence.__new__(p.Evidence)
    evidence.root, evidence.original_root = tmp_path, Path("/missing-original-workspace")
    local = tmp_path / "nested/evidence.json"
    local.parent.mkdir()
    local.write_text("immutable")
    pin = {**p.file_pin(local), "path": "/missing-original-workspace/nested/evidence.json"}
    assert evidence.resolve(pin) == local
    assert pin["path"].startswith("/missing-original-workspace")
    local.write_text("different")
    with pytest.raises(ValueError, match="changed"):
        evidence.resolve(pin)
    with pytest.raises(ValueError, match="outside"):
        evidence.path_for_original("/unrelated/data")


def test_member_paths_reject_parent_escapes_absolute_names_and_symlinks(tmp_path):
    for name in ("../escape", "/absolute"):
        with pytest.raises(ValueError, match="safe relative"):
            p.member_path(tmp_path, name)
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        p.member_path(tmp_path, "link/file")


def test_final_gate_selection_is_relocated_without_changing_original_bytes(tmp_path):
    evidence = p.Evidence.__new__(p.Evidence)
    evidence.root, evidence.original_root = tmp_path, Path("/nonexistent-original")
    policy = {"label": "learned", "kind": "learned", "checkpoint": "/nonexistent-original/policy.zip", "group": "recipe", "optimizer_seed": 101}
    champion = {"label": "storage", "kind": "baseline", "baseline": "storage"}
    primary = {"recipe": "recipe", "reference": "storage"}
    selected = {"controllers": [policy], "baseline_champion": champion, "recipe_id": "recipe", "primary": primary, "seeds": [52_000_000], "conditions": ["ordinary"]}
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(json.dumps(selected))
    selected_pin = {**p.file_pin(selected_path), "path": "/nonexistent-original/selected.json"}
    gate = {"schema": "learned-control-v1-evaluation-gate", "selected_recipe_frozen": True, "allowed_splits": ["final"],
            "selected_recipe": {"path": selected_pin["path"], "sha256": selected_pin["sha256"]}}
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate))
    original_gate_bytes = gate_path.read_bytes()
    evidence.spec = {"split": "final", "gate": {**p.file_pin(gate_path), "path": "/nonexistent-original/gate.json"},
                     "selected_recipe": selected_pin, "controllers": {"learned": policy, "storage": champion},
                     **{k: selected[k] for k in ("seeds", "conditions", "primary")}}
    assert evidence.selection() == (gate, selected)
    assert gate_path.read_bytes() == original_gate_bytes
    evidence.spec["controllers"]["learned"]["checkpoint"] = "/nonexistent-original/substituted.zip"
    with pytest.raises(ValueError, match="controller specification"):
        evidence.selection()


def test_manifest_requires_every_member_and_authentic_scientific_binding(tmp_path):
    source = tmp_path / "evidence.txt"
    source.write_text("immutable")
    pin = p.file_pin(source)
    item = {"path": source.name, "original_path": "/original/evidence.txt", "bytes": pin["bytes"], "sha256": pin["sha256"]}
    manifest = {"schema": p.SCHEMA+"-bundle", "files": [item], "scientific_bindings": [{"path": item["original_path"], "bytes": pin["bytes"], "sha256": pin["sha256"]}]}
    manifest["sha256"] = p.digest(manifest)
    (tmp_path / p.MANIFEST).write_text(json.dumps(manifest))
    assert p.verify_bundle(tmp_path)["sha256"] == manifest["sha256"]
    source.write_text("tampered")
    with pytest.raises(ValueError, match="bytes changed"):
        p.verify_bundle(tmp_path)


def test_release_packaging_rejects_development_and_incomplete_reviews(tmp_path):
    evidence = p.Evidence.__new__(p.Evidence)
    evidence.spec = {"split": "development"}
    with pytest.raises(ValueError, match="final panel"):
        p.create_bundle(evidence, tmp_path / "release.zip")
    evidence.spec = {"expected_cases": 100, "sha256": "freeze"}
    review = tmp_path / "review.json"
    replay = tmp_path / "replay.json"
    review.write_text(json.dumps({"verified": True, "all_inference_recomputed": False}))
    replay.write_text("{}")
    with pytest.raises(ValueError, match="recomputed inference"):
        p.require_release_reviews(evidence, review, replay)


def test_training_plan_dependency_discovery_authenticates_hash(tmp_path):
    root = tmp_path / "run"
    training = root / "training/checkpoints"
    training.mkdir(parents=True)
    plan = {"recipe": {"extra_source_paths": ["custom_policy.py"]}, "source_files": []}
    (root / "plan.json").write_text(json.dumps(plan))
    metadata = {"plan_sha256": p.digest(plan), "checkpoint": {}, "normalizer": {}}
    assert p.checkpoint_plan(training / "step.json", metadata, tmp_path) == root / "plan.json"
    metadata["plan_sha256"] = "wrong"
    assert p.checkpoint_plan(training / "step.json", metadata, tmp_path) is None


def test_relative_archive_entries_are_not_original_scientific_pins():
    assert list(p.iter_pins({"files": [{"path": "paper.tex", "bytes": 1, "sha256": "x"}]})) == []
    assert list(p.iter_pins({"source": {"path": "/original/source.py", "bytes": 1, "sha256": "x"}})) == [
        {"path": "/original/source.py", "bytes": 1, "sha256": "x"}]

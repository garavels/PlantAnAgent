"""Fabricated closed-artifact graphs only; no production root reads or resets."""

from dataclasses import asdict
import gzip
import importlib.util
import io
import math
from pathlib import Path
import sys

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "closure_audit", str(Path(__file__).with_name("plant-adaptive-closure-audit.py"))
)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(audit)
batch = audit.batch


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audit.canonical_bytes(value))


def identity(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": audit.sha(raw)}


def fabricate(root, *, failed_training=None, replay_timeout=False):
    root.mkdir()
    gate = root.parent / "gate.json"
    write(gate, {"passed": True, "source_commit": audit.PINNED_HEAD})
    write(
        root / "plan.json",
        {
            "specification": batch.specification(root),
            "runtime": {"fixture": True},
            "host": {},
            "gate": identity(gate),
            "started_at": "fixture",
        },
    )
    config = batch.template(root)
    receipts = []

    def receipt(stage, name, extra, fail=False):
        result = {
            "stage": stage,
            "name": name,
            "closed_at": "fixture",
            "codes": ["child_timeout"] if fail else [],
            "process": {
                "command": [sys.executable, "-m", batch.MODULE, stage, *extra],
                "timeout_seconds": (
                    batch.TRAIN_TIMEOUT if stage == "_train" else batch.EVAL_TIMEOUT
                ),
                "started": True,
                "exit_code": 0,
            },
        }
        write(root / "receipts" / f"{name}.json", result)
        receipts.append(result)
        (root / "logs").mkdir(exist_ok=True)
        (root / "logs" / f"{name}.stderr.log").write_bytes(
            b"fixture timeout or process error" if fail else b""
        )
        return result

    for arm in batch.ARMS:
        failed = arm.name == failed_training
        receipt("_train", f"train-{arm.name}", ["--cell", arm.name], fail=failed)
        if failed:
            continue
        directory = root / "training" / arm.name
        directory.mkdir(parents=True)
        data = {
            "action_net.weight": torch.ones(2, 2),
            "value_net.weight": torch.zeros(2, 2),
            "log_std": torch.full((4,), math.log(arm.initial_std)),
            "cost_critic.fixture": torch.ones(2),
        }
        buffer = io.BytesIO()
        torch.save(data, buffer)
        (directory / "initial-policy.pt").write_bytes(buffer.getvalue())
        (directory / "final-policy.zip").write_bytes(b"fixture-checkpoint-bytes")
        count = batch.TOTAL_STEPS // (config.n_envs * config.ppo.n_steps)
        ledgers = [{"rollout_index": i} for i in range(1, count + 1)]
        for i, ledger in enumerate(ledgers, 1):
            write(directory / "diagnostics" / f"rollout-{i:03d}.json", ledger)
            write(
                directory / "diagnostics" / f"diagnostic-{i:03d}.json",
                {
                    "fixture_diagnostic_index": i,
                    "schema": "plant-research-ppo-rollout-diagnostics-v1",
                    "num_timesteps": i * config.n_envs * config.ppo.n_steps,
                    "algorithm_id": arm.algorithm,
                },
            )
        write(
            directory / "attempt.json",
            {
                "arm": asdict(arm),
                "template": config.to_dict(),
                "seed_base": batch.TRAIN_SEED_BASE,
            },
        )
        write(
            directory / "result.json",
            {
                "schema": batch.SCHEMA,
                "arm": asdict(arm),
                "environment_template": config.to_dict(),
                "total_timesteps": batch.TOTAL_STEPS,
                "registered_evaluation_eligible": False,
                "checkpoint_contract_digest": audit.expected_contract(config),
                "initial_policy": identity(directory / "initial-policy.pt"),
                "final_policy": identity(directory / "final-policy.zip"),
                "constraint_state": {
                    "payload": {
                        "num_timesteps": batch.TOTAL_STEPS,
                        "pending_rollout": False,
                        "rollout_ledgers": ledgers,
                    }
                },
            },
        )
    for seed in batch.DEV_SEEDS:
        for actor in audit.ACTORS:
            failed = actor == failed_training or (
                actor == "initial" and failed_training == batch.ARMS[0].name
            )
            receipt(
                "_evaluate",
                f"evaluate-{seed}-{actor}",
                ["--actor", actor, "--seed", str(seed)],
                fail=failed,
            )
            if failed:
                continue
            timeout = replay_timeout and actor == "H" and seed == batch.DEV_SEEDS[0]
            receipt(
                "_replay",
                f"replay-{seed}-{actor}",
                ["--actor", actor, "--seed", str(seed)],
                fail=timeout,
            )
            directory = root / "evaluations" / f"{seed}-{actor}"
            write(
                directory / "attempt.json",
                {"actor": actor, "seed": seed, "started_at": "fixture"},
            )
            complete = actor == "H"
            decisions = 672 if complete else 0
            model_actor = actor if actor in batch.BASELINES else "PPO"
            resources = list(audit.evaluation.RESOURCE_NAMES)
            cycles = [
                {"feasible": False, "deficits": {name: 1.0 for name in resources}}
                for _ in range(7 if complete else 0)
            ]
            summary = {
                "actor": model_actor,
                "policy_id": actor,
                "registered_evaluation_eligible": False,
                "expected_decisions": 672,
                "expected_physical_steps": 10080,
                "decisions": decisions,
                "physical_steps": decisions * 15,
                "failure": (
                    None
                    if complete
                    else {
                        "phase": "controller_compute",
                        "message": "fabricated failure",
                    }
                ),
                "terminated": complete,
                "truncated": False,
                "execution_complete": complete,
                "hard_safety_classifier_failure": None,
                "hard_safety_passed": False,
                "cycle_settlements": cycles,
                "resource_order": resources,
                "cycle_deficit_vectors": [[1.0] * 4 for _ in cycles],
                "all_cycles_feasible": False,
                "physical_metrics": {"delivery_fill_rate": 0.5},
                "economic_metrics": {"net_operating_value_eur": -1.0},
            }
            episode = {
                "schema": audit.evaluation.SCHEMA,
                "seed": seed,
                "fixture": False,
                "prefix_limit": None,
                "config": config.to_dict(),
                "evaluator_source_sha256": audit.sha(
                    Path(audit.evaluation.__file__).read_bytes()
                ),
                "summary": summary,
                "history": [{}] * decisions,
                "minute_truth_states": [{}] * (decisions * 15 + 1),
            }
            raw = audit.canonical_bytes(episode)
            archive = directory / "episode.json.gz"
            archive.write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
            write(
                directory / "summary.json",
                {
                    "actor": actor,
                    "seed": seed,
                    "summary": summary,
                    "archive": identity(archive),
                },
            )
            if not timeout:
                write(
                    directory / "replay.json",
                    {
                        "schema": "plant-adaptive-research-replay-v1",
                        "verified": True,
                        "seed": seed,
                        "actor": model_actor,
                        "decisions": decisions,
                        "physical_steps": decisions * 15,
                        "execution_complete": complete,
                        "scope": (
                            "complete_episode"
                            if complete
                            else "retained_physical_prefix"
                        ),
                        "controller_reexecuted": False,
                        "controller_failure_authenticity_verified": False,
                        "registered_evaluation_eligible": False,
                        "evidence_sha256": audit.sha(raw),
                    },
                )
    write(
        root / "report.json",
        {
            "schema": batch.SCHEMA,
            "status": "closed",
            "failure": None,
            "closed_at": "fixture",
            "registered_evaluation_eligible": False,
            "plan": identity(root / "plan.json"),
            "receipts": receipts,
        },
    )
    return root


@pytest.fixture(autouse=True)
def forbid_production(monkeypatch):
    original = audit.read_regular

    def guarded(path):
        assert not str(path).startswith(
            str(batch.ROOT)
        ), "production outcomes must never be read by tests"
        return original(path)

    monkeypatch.setattr(audit, "read_regular", guarded)
    monkeypatch.setattr(
        batch.ppo,
        "make_env",
        lambda *args: pytest.fail("audit must not reset or construct environments"),
    )
    monkeypatch.setattr(
        audit.provenance,
        "capture",
        lambda **kwargs: pytest.fail("fixtures must not capture production runtime"),
    )


@pytest.mark.parametrize("status", [None, "running", "aborted"])
def test_closure_precedes_every_other_artifact(tmp_path, monkeypatch, status):
    root = tmp_path / "absent"
    root.mkdir()
    if status is not None:
        write(
            root / "report.json",
            {"status": status, "failure": None, "closed_at": "fixture"},
        )
    original = audit.read_regular

    def only_report(path):
        assert path == root / "report.json"
        return original(path)

    monkeypatch.setattr(audit, "read_regular", only_report)
    with pytest.raises(audit.NotClosed):
        audit.audit_closed_batch(root, fixture=True)


def test_complete_graph_retains_all_eight_slots_and_checks_initial_tensors(tmp_path):
    root = fabricate(tmp_path / "batch")
    packet = audit.audit_closed_batch(root, fixture=True)
    assert packet["batch_closed"] and packet["integrity_verified"]
    assert packet["integrity"]["training_slots"] == 5
    assert packet["integrity"]["evaluation_slots"] == 80
    assert packet["integrity"]["replay_slots"] == 80
    assert packet["initial_tensor_comparison"]["passed"]
    assert all(len(a["slots"]) == 8 for a in packet["actors"].values())
    assert all(
        s["eligible_for_descriptive_comparison"] for s in packet["actors"]["H"]["slots"]
    )
    assert all(
        not s["eligible_for_descriptive_comparison"] and s["failures"]
        for s in packet["actors"]["E-beta"]["slots"]
    )
    assert (
        packet["training"][batch.ARMS[0].name]["diagnostics"][0][
            "fixture_diagnostic_index"
        ]
        == 1
    )
    assert (
        packet["training"][batch.ARMS[0].name]["result"]["total_timesteps"]
        == batch.TOTAL_STEPS
    )
    assert packet["performance_means_computed"] is False


def test_failed_training_and_replay_remain_fixed_slots(tmp_path):
    root = fabricate(
        tmp_path / "batch", failed_training=batch.ARMS[2].name, replay_timeout=True
    )
    packet = audit.audit_closed_batch(root, fixture=True)
    training = packet["training"][batch.ARMS[2].name]
    assert not training["closed_successfully"] and training["result"] is None
    assert not packet["initial_tensor_comparison"]["performed"]
    slots = packet["actors"][batch.ARMS[2].name]["slots"]
    assert all(
        s["summary"] is None and not s["eligible_for_descriptive_comparison"]
        for s in slots
    )
    h = packet["actors"]["H"]["slots"][0]
    assert (
        h["summary"]["execution_complete"]
        and not h["eligible_for_descriptive_comparison"]
    )
    assert h["failures"][0]["kind"] == "replay_process"


@pytest.mark.parametrize(
    "target",
    [
        "checkpoint",
        "archive",
        "replay",
        "summary",
        "receipt",
        "plan",
        "diagnostic",
        "initial_tensor",
    ],
)
def test_artifact_mismatches_fail_closed(tmp_path, target):
    root = fabricate(tmp_path / "batch")
    cell = root / "evaluations" / "9700000-H"
    training = root / "training" / batch.ARMS[1].name
    if target == "checkpoint":
        (training / "final-policy.zip").write_bytes(b"changed")
    elif target == "archive":
        (cell / "episode.json.gz").write_bytes(b"changed")
    elif target == "replay":
        path = cell / "replay.json"
        value = audit.decode(path.read_bytes())
        value["evidence_sha256"] = "0" * 64
        write(path, value)
    elif target == "summary":
        path = cell / "summary.json"
        value = audit.decode(path.read_bytes())
        value["summary"]["economic_metrics"]["net_operating_value_eur"] = 999
        write(path, value)
    elif target == "receipt":
        path = root / "receipts" / "train-A-pi-raw.json"
        value = audit.decode(path.read_bytes())
        value["process"]["exit_code"] = 1
        write(path, value)
    elif target == "plan":
        path = root / "plan.json"
        value = audit.decode(path.read_bytes())
        value["specification"]["training_seed_base"] += 1
        write(path, value)
    elif target == "diagnostic":
        path = training / "diagnostics" / "rollout-001.json"
        write(path, {"rollout_index": 2})
    else:
        path = training / "initial-policy.pt"
        weights = torch.load(path, weights_only=True)
        weights["action_net.weight"] += 1
        torch.save(weights, path)
        result_path = training / "result.json"
        result = audit.decode(result_path.read_bytes())
        result["initial_policy"] = identity(path)
        write(result_path, result)
    with pytest.raises((audit.AuditError, ValueError)):
        audit.audit_closed_batch(root, fixture=True)


@pytest.mark.parametrize(
    "name,allowed",
    [
        ("._train-A-pi-raw.json", True),
        ("unexpected.json", False),
        (".unexpected.json", False),
    ],
)
def test_receipt_inventory_ignores_only_appledouble_sidecars(tmp_path, name, allowed):
    root = fabricate(tmp_path / "batch")
    (root / "receipts" / name).write_bytes(b"fixture sidecar or extra receipt")
    if allowed:
        assert audit.audit_closed_batch(root, fixture=True)["integrity_verified"]
    else:
        with pytest.raises(audit.AuditError, match="unexpected receipt artifacts"):
            audit.audit_closed_batch(root, fixture=True)


@pytest.mark.parametrize(
    "field,value",
    [("schema", "wrong"), ("num_timesteps", 1), ("algorithm_id", "wrong")],
)
def test_diagnostic_identity_checked_independently(tmp_path, field, value):
    root = fabricate(tmp_path / "batch")
    path = (
        root / "training" / batch.ARMS[0].name / "diagnostics" / "diagnostic-001.json"
    )
    diagnostic = audit.decode(path.read_bytes())
    diagnostic[field] = value
    write(path, diagnostic)
    with pytest.raises(
        audit.AuditError, match="diagnostic rollout or algorithm identity differs"
    ):
        audit.audit_closed_batch(root, fixture=True)

"""Fabricated closed-artifact graphs only; no production root reads or resets."""

from dataclasses import asdict
import gzip
import importlib.util
import io
import math
import json
import zipfile
from pathlib import Path
import sys

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "effort_closure_audit",
    str(Path(__file__).with_name("plant-effort-closure-audit.py")),
)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(audit)
batch = audit.batch
REAL_VERIFY_LOADED = audit.verify_loaded_checkpoint
REAL_VERIFY_ALLOCATOR = audit.verify_allocator


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audit.canonical_bytes(value))


def identity(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": audit.sha(raw)}


def fabricate(
    root, *, failed_training=None, replay_timeout=False, retain_failed_prefix=False
):
    root.mkdir()
    gate = root.parent / "gate.json"
    write(
        gate,
        {
            "passed": True,
            "source_commit": audit.PINNED_HEAD,
            "source_tree": audit.PINNED_TREE,
            "source_clean_and_unchanged": True,
        },
    )
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
        if failed and not retain_failed_prefix:
            continue
        directory = root / "training" / arm.name
        directory.mkdir(parents=True)
        from plant_backend.rl.v2.training.constrained_ppo import (
            RolloutLedgerV1,
            PILagrangeStateV1,
            ConstrainedPPOSettingsV1,
        )

        identity_policy = audit.expected_identity(root, arm)
        data = {
            "action_net.weight": torch.zeros(4, 64),
            "action_net.bias": torch.zeros(4),
            "value_net.weight": torch.zeros(1, 64),
            "log_std": torch.full((4,), math.log(0.2)),
            "mlp_extractor.policy_net.0.weight": torch.ones(
                64, 139 if arm.residual else 135
            ),
            "mlp_extractor.value_net.0.weight": torch.ones(
                64, 139 if arm.residual else 135
            ),
        }
        if arm.algorithm == batch.PI:
            data["cost_critic.fixture"] = torch.ones(2)
        buffer = io.BytesIO()
        torch.save(data, buffer)
        (directory / "initial-policy.pt").write_bytes(buffer.getvalue())
        settings = ConstrainedPPOSettingsV1.for_algorithm(
            arm.algorithm, optimizer_seed=101
        )
        state = PILagrangeStateV1(
            (0.0,) * 4, settings.initial_integral, settings.initial_integral, 0
        )
        ledgers = []
        for i in range(1, 2 if failed else 17):
            ledger = RolloutLedgerV1.build(
                algorithm_id=arm.algorithm,
                rollout_index=i,
                transition_count=8192,
                completed_episode_count=12,
                complete_cycle_count=88 if i % 3 == 0 else 84,
                reward_return=0.0,
                raw_cost=[0.0] * 4,
                training_cost=[0.0] * 4,
                mean_completed_episode_cost=[0.0] * 4,
                cost_value_losses=[0.0] * 4,
                multiplier_state=state,
                multiplier_min=0.0,
                multiplier_max=100.0,
                approximate_kl=0.0,
                entropy_loss=0.0,
                clip_fraction=0.0,
                optimizer_steps=320,
                elapsed_compute_s=0.0,
                previous_sha256=ledgers[-1]["sha256"] if ledgers else None,
            ).to_dict()
            ledgers.append(ledger)
            write(directory / "diagnostics" / f"rollout-{i:03d}.json", ledger)
            write(
                directory / "diagnostics" / f"diagnostic-{i:03d}.json",
                {
                    "fixture_diagnostic_index": i,
                    "schema": "plant-research-ppo-rollout-diagnostics-v1",
                    "num_timesteps": i * 8192,
                    "algorithm_id": arm.algorithm,
                },
            )
        write(
            directory / "attempt.json",
            {
                "arm": asdict(arm),
                "template": config.to_dict(),
                "seed_base": batch.TRAIN_SEED_BASE,
                "policy_identity": identity_policy,
            },
        )
        if failed:
            continue
        constraint_state = {
            "schema": "fixture",
            "sha256": "fixture",
            "payload": {
                "num_timesteps": batch.TOTAL_STEPS,
                "pending_rollout": False,
                "aborted_rollout": None,
                "unledgered_transition_count": 0,
                "resume_continuation_receipt": None,
                "ppo_epoch_update_count": 160,
                "rollout_ledgers": ledgers,
                "rollout_ledger_sha256": ledgers[-1]["sha256"],
                "settings": settings.to_dict(),
                "ppo_configuration": {
                    **asdict(config.ppo),
                    "n_envs": 4,
                    "seed": 101,
                    "observation_space": identity_policy["observation_space"],
                    "action_space": identity_policy["action_space"],
                },
            },
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "data",
                json.dumps(
                    {
                        "research_identity": identity_policy,
                        "_embedded_constraint_state_schema": constraint_state["schema"],
                        "_embedded_constraint_state_sha256": constraint_state["sha256"],
                        "_embedded_constraint_state_payload_json": json.dumps(
                            constraint_state["payload"]
                        ),
                    }
                ),
            )
        (directory / "final-policy.zip").write_bytes(buffer.getvalue())
        write(
            directory / "result.json",
            {
                "schema": batch.SCHEMA,
                "arm": asdict(arm),
                "environment_template": config.to_dict(),
                "total_timesteps": batch.TOTAL_STEPS,
                "registered_evaluation_eligible": False,
                "checkpoint_contract_digest": identity_policy[
                    "checkpoint_contract_digest"
                ],
                "policy_identity": identity_policy,
                "initial_policy": identity(directory / "initial-policy.pt"),
                "final_policy": identity(directory / "final-policy.zip"),
                "constraint_state": constraint_state,
                "allocator": {"fabricated": True},
            },
        )
    for seed in batch.DEV_SEEDS:
        for actor in audit.ACTORS:
            failed = actor == failed_training or (
                actor in batch.INITIAL_SOURCES
                and failed_training == batch.INITIAL_SOURCES[actor].name
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
            complete = actor in ("H", "initial-residual")
            decisions = 672 if complete else 0
            residual = audit.is_residual(actor)
            model_actor = (
                audit.residual_evaluation.ACTOR
                if residual
                else (actor if actor in batch.BASELINES else "PPO")
            )
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
                "schema": (
                    audit.residual_evaluation.SCHEMA
                    if residual
                    else audit.evaluation.SCHEMA
                ),
                "seed": seed,
                "fixture": False,
                "prefix_limit": None,
                "config": config.to_dict(),
                "evaluator_source_sha256": audit.sha(
                    Path(
                        (
                            audit.residual_evaluation if residual else audit.evaluation
                        ).__file__
                    ).read_bytes()
                ),
                "summary": summary,
                "manifest": {"fixture": True},
                "initial_observation": {"normalized": []},
                "final_observation": {"normalized": []},
                "history": [
                    {
                        "decision_index": i,
                        "observation": {},
                        "after_observation": {},
                        "command": {},
                        "controller_invalid": False,
                        "reward_eur": 0.0,
                        "terminated": i == decisions - 1,
                        "truncated": False,
                        "info": {},
                        "execution": {},
                        "raw_action_shape": [4],
                        "raw_action": [0.0] * 4,
                        "residual_evidence": {
                            "body": {"exact_zero_heuristic_command": True}
                        },
                    }
                    for i in range(decisions)
                ],
                "minute_truth_states": [{}] * (decisions * 15 + 1),
            }
            if residual:
                ident = audit.expected_identity(root, batch.ARMS[2])
                episode.update(
                    interface_contract=ident["interface_contract"],
                    interface_contract_digest=ident["interface_contract_digest"],
                    controller={
                        "id": actor,
                        "deterministic": True,
                        "interface": "residual-H-139",
                    },
                )
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
                        "schema": (
                            "plant-adaptive-residual-replay-v1"
                            if residual
                            else "plant-adaptive-research-replay-v1"
                        ),
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
                        **(
                            {
                                "saved_absolute_commands_executed": True,
                                "heuristic_reexecuted_for_mapping_check": True,
                                "learned_policy_reexecuted": False,
                            }
                            if residual
                            else {"controller_reexecuted": False}
                        ),
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
        batch.PlantCyclicMarketEnvV2Beta,
        "reset",
        lambda *args, **kwargs: pytest.fail("audit must not reset environments"),
    )
    monkeypatch.setattr(
        audit,
        "verify_loaded_checkpoint",
        lambda *args: {"verified": True, "fabricated_model_attestation": True},
    )
    monkeypatch.setattr(
        audit,
        "verify_allocator",
        lambda *args: {"verified": True, "fabricated_allocator_attestation": True},
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
    assert packet["integrity"]["training_slots"] == 4
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
    assert packet["initial_tensor_comparison"]["passed"] is None
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
    cell = root / "evaluations" / "9700100-H"
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
        path = root / "receipts" / "train-A-absolute-pi.json"
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
        value = audit.decode(path.read_bytes())
        value["rollout_index"] = 2
        write(path, value)
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
        ("._train-A-absolute-pi.json", True),
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


def rewrite_episode(root, actor, mutate):
    cell = root / "evaluations" / f"{batch.DEV_SEEDS[0]}-{actor}"
    path = cell / "episode.json.gz"
    value = audit.decode(gzip.decompress(path.read_bytes()))
    mutate(value)
    raw = audit.canonical_bytes(value)
    path.write_bytes(gzip.compress(raw, mtime=0))
    outer = audit.decode((cell / "summary.json").read_bytes())
    outer["archive"] = identity(path)
    outer["summary"] = value["summary"]
    write(cell / "summary.json", outer)
    replay = audit.decode((cell / "replay.json").read_bytes())
    replay["evidence_sha256"] = audit.sha(raw)
    write(cell / "replay.json", replay)


def test_initial_residual_equivalence_preserves_every_complete_pair(tmp_path):
    packet = audit.audit_closed_batch(fabricate(tmp_path / "batch"), fixture=True)
    eq = packet["initial_residual_heuristic_equivalence"]
    assert eq["all_eight_pairs_available"] and eq["all_eight_pairs_equal"]
    assert [slot["seed"] for slot in eq["slots"]] == list(batch.DEV_SEEDS)
    assert all(slot["equal"] and len(slot["comparisons"]) == 6 for slot in eq["slots"])
    assert all(
        len(a["rollouts"]) == len(a["diagnostics"]) == 16
        for a in packet["training"].values()
    )


def test_unequal_saved_physics_is_retained_without_a_false_equivalence_claim(tmp_path):
    root = fabricate(tmp_path / "batch")
    rewrite_episode(
        root,
        "initial-residual",
        lambda episode: episode["minute_truth_states"][0].update(
            {"changed_physics": True}
        ),
    )
    packet = audit.audit_closed_batch(root, fixture=True)
    eq = packet["initial_residual_heuristic_equivalence"]
    assert eq["all_eight_pairs_available"] and eq["all_eight_pairs_equal"] is False
    assert eq["slots"][0]["comparisons"]["minute_truth_states"] is False
    assert sum(s["equal"] for s in eq["slots"]) == 7


@pytest.mark.parametrize("arm", ["A-absolute-pi", "C-residual-pi"])
def test_failed_source_initial_is_unavailable_and_prefix_ledgers_survive(tmp_path, arm):
    root = fabricate(tmp_path / "batch", failed_training=arm, retain_failed_prefix=True)
    packet = audit.audit_closed_batch(root, fixture=True)
    entry = packet["training"][arm]
    assert entry["result"] is None and not entry["closed_successfully"]
    assert len(entry["diagnostics"]) == len(entry["rollouts"]) == 1
    assert entry["initial_policy"]["sha256"]
    initial = "initial-absolute" if arm.startswith("A") else "initial-residual"
    assert all(
        s["summary"] is None and s["failures"][0]["kind"] == "training_unavailable"
        for s in packet["actors"][initial]["slots"]
    )
    if initial == "initial-residual":
        eq = packet["initial_residual_heuristic_equivalence"]
        assert (
            not eq["all_eight_pairs_available"] and eq["all_eight_pairs_equal"] is None
        )
        assert not any(s["available"] for s in eq["slots"])


@pytest.mark.parametrize(
    "target",
    ["episode_seed", "policy_id", "interface", "zero_action", "residual_replay"],
)
def test_residual_cross_slot_interface_and_replay_tamper_fails_closed(tmp_path, target):
    root = fabricate(tmp_path / "batch")
    if target == "residual_replay":
        path = (
            root
            / "evaluations"
            / f"{batch.DEV_SEEDS[0]}-initial-residual"
            / "replay.json"
        )
        value = audit.decode(path.read_bytes())
        value["evidence_sha256"] = "f" * 64
        write(path, value)
    else:

        def mutate(episode):
            if target == "episode_seed":
                episode["seed"] += 1
            elif target == "policy_id":
                episode["summary"]["policy_id"] = "C-residual-pi"
            elif target == "interface":
                episode["interface_contract"]["observation_size"] = 135
            else:
                episode["history"][0]["raw_action"][0] = 0.01

        rewrite_episode(root, "initial-residual", mutate)
    with pytest.raises(audit.AuditError):
        audit.audit_closed_batch(root, fixture=True)


@pytest.mark.parametrize(
    "target",
    [
        "runtime",
        "gate_tree",
        "gate_clean",
        "training_interface",
        "training_settings",
        "training_space",
    ],
)
def test_frozen_source_gate_and_training_identity_mismatches_fail(tmp_path, target):
    root = fabricate(tmp_path / "batch")
    if target in ("runtime", "gate_tree", "gate_clean"):
        path = root / "plan.json"
        plan = audit.decode(path.read_bytes())
        if target == "runtime":
            plan["runtime"] = {"fixture": False}
        else:
            gate_path = Path(plan["gate"]["path"])
            gate = audit.decode(gate_path.read_bytes())
            gate[
                "source_tree" if target == "gate_tree" else "source_clean_and_unchanged"
            ] = ("wrong" if target == "gate_tree" else False)
            write(gate_path, gate)
            plan["gate"] = identity(gate_path)
        write(path, plan)
        report = audit.decode((root / "report.json").read_bytes())
        report["plan"] = identity(path)
        write(root / "report.json", report)
    else:
        path = root / "training" / batch.ARMS[2].name / "result.json"
        result = audit.decode(path.read_bytes())
        if target == "training_interface":
            result["policy_identity"]["observation_size"] = 135
        elif target == "training_settings":
            result["constraint_state"]["payload"]["settings"]["optimizer_seed"] += 1
        else:
            result["constraint_state"]["payload"]["ppo_configuration"][
                "observation_space"
            ]["shape"] = [135]
        write(path, result)
    with pytest.raises((audit.AuditError, ValueError)):
        audit.audit_closed_batch(root, fixture=True)


@pytest.mark.parametrize("mismatch", [None, "identity", "algorithm", "attestation"])
def test_loaded_model_attestation_binding_without_inference_or_resets(
    tmp_path, monkeypatch, mismatch
):
    from types import SimpleNamespace
    from plant_backend.rl.v2.training.constrained_ppo import ConstrainedPPO

    path = tmp_path / "fake-checkpoint.zip"
    path.write_bytes(b"fabricated closed bytes")
    reader = audit.Reader()
    reader.read(path)
    ident = {"fixture_interface": 139}
    state = {"fixture_state": True}
    result = {
        "policy_identity": ident,
        "constraint_state": state,
        "final_policy": reader.records[path],
    }
    attestation = SimpleNamespace(
        artifact_sha256=reader.records[path]["sha256"], constraint_state=state
    )
    model = SimpleNamespace(
        algorithm_id=batch.ARMS[2].algorithm,
        num_timesteps=batch.TOTAL_STEPS,
        research_identity=ident,
        loaded_constraint_state_attestation=lambda: attestation,
        policy=SimpleNamespace(state_dict=lambda: {"fake.weight": torch.zeros(2, 3)}),
    )
    if mismatch == "identity":
        model.research_identity = {"fixture_interface": 135}
    elif mismatch == "algorithm":
        model.algorithm_id = "wrong"
    elif mismatch == "attestation":
        attestation.artifact_sha256 = "0" * 64
    calls = []

    def load(stored_path, **kwargs):
        calls.append((stored_path, kwargs))
        return model

    monkeypatch.setattr(ConstrainedPPO, "load", load)
    monkeypatch.setattr(batch, "verify_model_spaces", lambda *args: None)
    if mismatch:
        with pytest.raises(audit.AuditError):
            REAL_VERIFY_LOADED(reader, path, result, object(), batch.ARMS[2])
    else:
        checked = REAL_VERIFY_LOADED(reader, path, result, object(), batch.ARMS[2])
        assert (
            checked["verified"] and not checked["policy_inference_or_learning_executed"]
        )
        assert checked["policy_tensor_shapes"] == {"fake.weight": [2, 3]}
    assert calls == [(path, {"device": "cpu"})]


@pytest.mark.parametrize("mismatch", [None, "pin", "seed", "count", "snapshot"])
def test_allocator_endpoint_pin_and_counts_without_any_reset(
    tmp_path, monkeypatch, mismatch
):
    from types import SimpleNamespace
    from plant_backend.rl.v2.training.episode_allocator import EpisodeSeedAllocatorV1
    from plant_backend.rl.v2.training.constrained_ppo import ExternalEpisodeStatePinV1

    snapshot = {"fabricated": "snapshot"}
    pin = ExternalEpisodeStatePinV1.build(snapshot).to_dict()
    result = {
        "allocator": snapshot,
        "constraint_state": {"payload": {"external_episode_state_pin": pin}},
        "final_policy": {"sha256": "a" * 64},
    }
    restored = SimpleNamespace(
        base_seed=batch.TRAIN_SEED_BASE,
        episode_decisions=672,
        decisions_consumed_by_environment=(32768,) * 4,
        ledger=[None] * 196,
        snapshot=lambda: SimpleNamespace(to_dict=lambda: snapshot),
    )
    if mismatch == "pin":
        result["allocator"] = {"different": "snapshot"}
    elif mismatch == "seed":
        restored.base_seed += 1
    elif mismatch == "count":
        restored.decisions_consumed_by_environment = (0,) * 4
    elif mismatch == "snapshot":
        restored.snapshot = lambda: SimpleNamespace(to_dict=lambda: {"changed": True})
    calls = []

    def restore(value, **kwargs):
        assert value["snapshot"] == snapshot
        assert kwargs["expected_parent_checkpoint_sha256"] == "a" * 64
        assert kwargs["expected_resume_sha256"] == value["sha256"]
        calls.append(value)
        return restored

    monkeypatch.setattr(EpisodeSeedAllocatorV1, "restore_fixture", restore)
    if mismatch:
        with pytest.raises(audit.AuditError):
            REAL_VERIFY_ALLOCATOR(result, batch.template(tmp_path))
    else:
        checked = REAL_VERIFY_ALLOCATOR(result, batch.template(tmp_path))
        assert (
            checked["verified"]
            and checked["resets_executed"] == 0
            and checked["ledger_entries"] == 196
        )
    assert len(calls) == (0 if mismatch == "pin" else 1)

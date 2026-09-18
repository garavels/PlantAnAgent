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
    "mpc_residual_closure_audit_v2",
    str(Path(__file__).with_name("plant-mpc-residual-closure-audit-v2.py")),
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


def fabricate_allocator_prefixes(directory, *, count):
    from plant_backend.rl.v2.training.episode_allocator import (
        EpisodeSeedAllocatorV1,
        ResetManifestLedgerEntryV1,
        EpisodeSeedSplitV1,
        BetaManifestIdentityV1,
        EpisodeCursorV1,
        EpisodeAllocatorSnapshotV1,
    )

    allocator_identity = EpisodeSeedAllocatorV1.fixture(
        base_seed=batch.TRAIN_SEED_BASE
    ).identity
    for i in range(1, count + 1):
        decisions = i * 2048
        ordinal, offset = divmod(decisions, 672)
        entries = tuple(
            ResetManifestLedgerEntryV1(
                ledger_index=n,
                allocator_identity_sha256=allocator_identity["sha256"],
                split=EpisodeSeedSplitV1.FIXTURE,
                environment_index=n % 4,
                local_episode_ordinal=n // 4,
                global_episode_ordinal=n,
                episode_seed=batch.TRAIN_SEED_BASE + n,
                ignored_framework_seed=None,  # ConstrainedPPO never forwards optimizer seeds to environments.
                manifest_identity=BetaManifestIdentityV1(
                    f"{n+1:064x}", f"{n+2:064x}", f"{n+3:064x}"
                ),
            )
            for n in range(4 * (ordinal + 1))
        )
        cursors = tuple(
            EpisodeCursorV1(j, ordinal, offset, ordinal + 1, False) for j in range(4)
        )
        snapshot = EpisodeAllocatorSnapshotV1(
            allocator_identity, cursors, entries
        ).to_dict()
        for entry in entries:
            write(
                directory / "allocator" / f"reset-{entry.ledger_index:03d}.json",
                entry.to_dict(),
            )
        write(
            directory / "allocator" / f"rollout-{i:03d}.json",
            {
                "schema": "plant-research-mpc-residual-allocator-prefix-v1",
                "allocator_identity": allocator_identity,
                "ledger": [e.to_dict() for e in entries],
                "cursors": [c.to_dict() for c in cursors],
                "decisions_consumed_by_environment": [decisions] * 4,
                "successful_endpoint": False,
            },
        )
    return snapshot


from functools import lru_cache


@lru_cache(maxsize=2048)
def physical_row(index, *, residual, final=False):
    """Fabricate complete hash descendants without calling plant or controller APIs."""
    from copy import deepcopy
    from plant_sim import PlantConfig
    from plant_backend.controllers.base import Constraints
    from plant_backend.operational_mpc import DECISION_SCHEMA, POLICY_ID
    from plant_backend.rl.v2.setpoint_command import issue_process_setpoint_command
    from plant_backend.rl.v2.setpoint_control import ProcessSetpointTargetsV1
    from plant_backend.rl.v2.mpc_schema import domain_sha256
    from plant_backend.rl.v2.training.research_mpc_residual import EVIDENCE_SCHEMA

    targets = ProcessSetpointTargetsV1(1800.0, 400.0, 80.0, 1000.0)
    operation = {
        "mode": "fallback",
        "solver_failure": {
            "stages": [
                {
                    "stage": "fixture",
                    "elapsed_ns": 29 if residual else 11,
                    "status": "fixture_rejection",
                    "message": "identical solver meaning",
                }
            ]
        },
    }
    payload = {
        "schema": DECISION_SCHEMA,
        "observation_sha256": "b" * 64,
        "operation": operation,
        "targets": targets.to_dict(),
        "plan_sha256": None,
    }
    source_sha = domain_sha256(DECISION_SCHEMA, payload)
    command = issue_process_setpoint_command(
        targets,
        decision_index=index,
        observation_sha256="a" * 64,
        hold_duration_s=900,
        issuer_id=POLICY_ID,
        source_schema=DECISION_SCHEMA,
        source_sha256=source_sha,
        constraints=Constraints(),
        config=PlantConfig(),
    ).to_dict()
    nominal = {
        "command": command,
        "invalid": False,
        "evidence": {
            "payload": payload,
            "sha256": source_sha,
            "source_notes": [],
            "plan": None,
        },
    }
    body = {
        "schema": EVIDENCE_SCHEMA,
        "mpc_decision": nominal,
        "exact_zero_mpc_command": True,
        "mapped_targets": targets.to_dict(),
        "raw_residual": [0.0] * 4,
        "bounded_residual": [0.0] * 4,
    }
    residual_evidence = {
        "status": "issued",
        "body": body,
        "sha256": domain_sha256(EVIDENCE_SCHEMA, body),
        "command": command,
        "controller_invalid": False,
    }

    def hashed(domain, value):
        return {
            **value,
            "schema": domain,
            "sha256": domain_sha256(domain, {**value, "schema": domain}),
        }

    allocation = {
        "schema": "fabricated-allocation",
        "routes_kw": [1.0, 2.0, 3.0, 4.0],
        "sha256": "c" * 64,
    }
    dispatch = hashed(
        "process-setpoint-minute-dispatch-v1",
        {
            "command_sha256": command["sha256"],
            "decision_index": index,
            "boundary": {"step": index * 15, "sha256": "d" * 64},
            "allocation": allocation,
        },
    )
    application = hashed(
        "process-setpoint-minute-application-v1",
        {
            "ordinal": 0,
            "dispatch": dispatch,
            "next_boundary": {"step": index * 15 + 1, "sha256": "e" * 64},
            "guard": {"action": {"electrolyser_kw": 1800}, "adjustments": []},
            "plant": {"applied_kw": {"electrolyser": 1800.0, "battery_support": 100.0}},
        },
    )
    execution = hashed(
        "process-setpoint-interval-execution-v1",
        {
            "command": command,
            "physics_step_s": 60.0,
            "application_count": 1,
            "first_boundary_sha256": "d" * 64,
            "final_boundary_sha256": "e" * 64,
            "applications": [application],
        },
    )
    observation = {
        "normalized": [0.0] * 135,
        "typed": {"fixture_public": index},
        "typed_sha256": "b" * 64,
    }
    return {
        "decision_index": index,
        "observation": observation,
        "after_observation": observation,
        "command": command,
        "controller_invalid": False,
        "reward_eur": 0.0,
        "terminated": final,
        "truncated": False,
        "info": {
            "requested": {**command, "requested_kw_basis": "fixture"},
            "process_setpoint_execution": execution,
            "cycle_constraint_transition": {"deficits": [1.0] * 4},
        },
        "execution": execution,
        "raw_action_shape": [4],
        "raw_action": [0.0] * 4,
        "residual_evidence": residual_evidence,
        "mpc_decision": nominal,
        "operation": operation,
        "controller_evidence": json.dumps(payload),
    }


def fabricate(
    root,
    *,
    failed_training=None,
    replay_timeout=False,
    retain_failed_prefix=False,
    complete_actors=None,
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
        (root / "logs" / f"{name}.stdout.log").write_bytes(b"")
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
        if False:
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
        for i in range(1, 2 if failed else 5):
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
        allocator_snapshot = fabricate_allocator_prefixes(
            directory, count=1 if failed else 4
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
                "ppo_epoch_update_count": 40,
                "external_episode_state_pin": __import__(
                    "plant_backend.rl.v2.training.constrained_ppo",
                    fromlist=["ExternalEpisodeStatePinV1"],
                )
                .ExternalEpisodeStatePinV1.build(allocator_snapshot)
                .to_dict(),
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
                "allocator": allocator_snapshot,
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
            timeout = (
                replay_timeout
                and actor == "OperatingMPC"
                and seed == batch.DEV_SEEDS[0]
            )
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
            write(
                directory / "replay-attempt.json",
                {"actor": actor, "seed": seed, "started_at": "fixture"},
            )
            complete = actor in (
                complete_actors
                if complete_actors is not None
                else ("OperatingMPC", "initial-mpc-residual")
            )
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
                "manifest": {
                    "fixture": True,
                    "delivery_accounting": {"schedule": [], "settlements": []},
                },
                "initial_observation": {"normalized": []},
                "final_observation": {"normalized": []},
                "history": [
                    physical_row(i, residual=residual, final=i == decisions - 1)
                    for i in range(decisions)
                ],
                "minute_truth_states": [{}] * (decisions * 15 + 1),
            }
            if residual:
                ident = audit.expected_identity(root, batch.ARMS[0])
                episode.update(
                    interface_contract=ident["interface_contract"],
                    interface_contract_digest=ident["interface_contract_digest"],
                    controller={
                        "id": actor,
                        "deterministic": True,
                        "interface": "residual-MPC-139",
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
                            "plant-adaptive-mpc-residual-replay-v1"
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
                                "mpc_reexecuted_for_mapping_check": True,
                                "solver_timing_semantics_verified": True,
                                "excluded_clock_paths": [
                                    "mpc_decision.evidence.plan.stages[*].elapsed_ns",
                                    "mpc_decision.evidence.payload.operation.solver_failure.stages[*].elapsed_ns",
                                ],
                                "derived_clock_hashes_recomputed": True,
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
    from copy import deepcopy

    original_identity = batch.policy_identity
    identity_cache = {}

    def cached_identity(config, arm):
        key = (audit.canonical_bytes(config.to_dict()), arm)
        if key not in identity_cache:
            identity_cache[key] = original_identity(config, arm)
        return deepcopy(identity_cache[key])

    monkeypatch.setattr(batch, "policy_identity", cached_identity)
    from plant_backend import operational_mpc

    monkeypatch.setattr(
        operational_mpc.OperationalMPCController,
        "compute_command",
        lambda *args, **kwargs: pytest.fail("audit must not execute a controller"),
    )
    monkeypatch.setattr(
        operational_mpc,
        "solve_economic_mpc_model",
        lambda *args, **kwargs: pytest.fail("audit must not solve a planner"),
    )
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
    assert packet["integrity"]["training_slots"] == 1
    assert packet["integrity"]["evaluation_slots"] == 48
    assert packet["integrity"]["replay_slots"] == 48
    assert packet["initial_tensor_comparison"]["passed"]
    assert all(len(a["slots"]) == 8 for a in packet["actors"].values())
    assert all(
        s["eligible_for_descriptive_comparison"]
        for s in packet["actors"]["OperatingMPC"]["slots"]
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
        tmp_path / "batch", failed_training=batch.ARMS[0].name, replay_timeout=True
    )
    packet = audit.audit_closed_batch(root, fixture=True)
    training = packet["training"][batch.ARMS[0].name]
    assert not training["closed_successfully"] and training["result"] is None
    assert packet["initial_tensor_comparison"]["passed"] is None
    slots = packet["actors"][batch.ARMS[0].name]["slots"]
    assert all(
        s["summary"] is None and not s["eligible_for_descriptive_comparison"]
        for s in slots
    )
    h = packet["actors"]["OperatingMPC"]["slots"][0]
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
    cell = root / "evaluations" / "9700200-H"
    training = root / "training" / batch.ARMS[0].name
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
        path = root / "receipts" / "train-A-mpc-residual-scalar.json"
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
        ("._train-A-mpc-residual-scalar.json", True),
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
    eq = packet["initial_residual_mpc_equivalence"]
    assert eq["all_eight_pairs_available"] and eq["all_eight_pairs_equal"]
    assert [slot["seed"] for slot in eq["slots"]] == list(batch.DEV_SEEDS)
    assert all(slot["equal"] and len(slot["comparisons"]) == 6 for slot in eq["slots"])
    assert all(
        len(a["rollouts"]) == len(a["diagnostics"]) == 4
        for a in packet["training"].values()
    )


def test_unequal_saved_physics_is_retained_without_a_false_equivalence_claim(tmp_path):
    root = fabricate(tmp_path / "batch")
    rewrite_episode(
        root,
        "initial-mpc-residual",
        lambda episode: episode["minute_truth_states"][0].update(
            {"changed_physics": True}
        ),
    )
    packet = audit.audit_closed_batch(root, fixture=True)
    eq = packet["initial_residual_mpc_equivalence"]
    assert eq["all_eight_pairs_available"] and eq["all_eight_pairs_equal"] is False
    assert eq["slots"][0]["comparisons"]["minute_truth_states"] is False
    assert sum(s["equal"] for s in eq["slots"]) == 7


@pytest.mark.parametrize("arm", ["A-mpc-residual-scalar"])
def test_failed_source_initial_is_unavailable_and_prefix_ledgers_survive(tmp_path, arm):
    root = fabricate(tmp_path / "batch", failed_training=arm, retain_failed_prefix=True)
    packet = audit.audit_closed_batch(root, fixture=True)
    entry = packet["training"][arm]
    assert entry["result"] is None and not entry["closed_successfully"]
    assert len(entry["diagnostics"]) == len(entry["rollouts"]) == 1
    assert entry["initial_policy"]["sha256"]
    initial = "initial-mpc-residual"
    assert all(
        s["summary"] is None and s["failures"][0]["kind"] == "training_unavailable"
        for s in packet["actors"][initial]["slots"]
    )
    if initial == "initial-mpc-residual":
        eq = packet["initial_residual_mpc_equivalence"]
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
            / f"{batch.DEV_SEEDS[0]}-initial-mpc-residual"
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
                episode["summary"]["policy_id"] = "A-mpc-residual-scalar"
            elif target == "interface":
                episode["interface_contract"]["observation_size"] = 135
            else:
                episode["history"][0]["raw_action"][0] = 0.01

        rewrite_episode(root, "initial-mpc-residual", mutate)
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
        path = root / "training" / batch.ARMS[0].name / "result.json"
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
        algorithm_id=batch.ARMS[0].algorithm,
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
            REAL_VERIFY_LOADED(reader, path, result, object(), batch.ARMS[0])
    else:
        checked = REAL_VERIFY_LOADED(reader, path, result, object(), batch.ARMS[0])
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
        decisions_consumed_by_environment=(8192,) * 4,
        ledger=[None] * 52,
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
            and checked["ledger_entries"] == 52
        )
    assert len(calls) == (0 if mismatch == "pin" else 1)


def projection_episode(*, residual):
    from copy import deepcopy

    return {
        "manifest": {"weather_truth_sha256": "1" * 64},
        "initial_observation": {"normalized": [0.0] * 135},
        "final_observation": {"normalized": [0.0] * 135},
        "minute_truth_states": [{"battery_soc": 0.5}, {"battery_soc": 0.49}],
        "history": [deepcopy(physical_row(0, residual=residual, final=True))],
        "summary": {
            "actor": "MPCResidualPPO" if residual else "OperatingMPC",
            "policy_id": "initial-mpc-residual" if residual else "OperatingMPC",
            "cycle_deficit_vectors": [[1.0] * 4],
            "value_eur": -100.0,
        },
    }


def rehash_execution(row):
    from copy import deepcopy
    from plant_backend.rl.v2.mpc_schema import domain_sha256

    def update(record):
        body = {k: v for k, v in record.items() if k != "sha256"}
        record["sha256"] = domain_sha256(record["schema"], body)

    for app in row["execution"]["applications"]:
        update(app["dispatch"])
        update(app)
    update(row["execution"])
    row["info"]["process_setpoint_execution"] = deepcopy(row["execution"])


def test_timing_only_pair_has_different_raw_hashes_and_equal_physical_projection():
    native = projection_episode(residual=False)
    initial = projection_episode(residual=True)
    assert (
        native["history"][0]["command"]["sha256"]
        != initial["history"][0]["command"]["sha256"]
    )
    assert (
        native["history"][0]["execution"]["sha256"]
        != initial["history"][0]["execution"]["sha256"]
    )
    assert audit.physical_projection(
        native, residual=False
    ) == audit.physical_projection(initial, residual=True)


@pytest.mark.parametrize(
    "field",
    [
        "target",
        "minute",
        "resource",
        "public",
        "allocation_hash",
        "boundary_hash",
        "other_time",
    ],
)
def test_one_physical_or_unrelated_provenance_change_cannot_be_timing_normalized(field):
    native = projection_episode(residual=False)
    initial = projection_episode(residual=True)
    row = initial["history"][0]
    if field == "target":
        row["command"]["targets"]["setpoints_kw"]["electrolyser"] += 1.0
    elif field == "minute":
        initial["minute_truth_states"][0]["battery_soc"] += 0.01
    elif field == "resource":
        initial["summary"]["cycle_deficit_vectors"][0][2] += 0.1
    elif field == "public":
        row["observation"]["normalized"][0] = 0.5
    elif field == "allocation_hash":
        row["execution"]["applications"][0]["dispatch"]["allocation"]["sha256"] = (
            "f" * 64
        )
        rehash_execution(row)
    elif field == "boundary_hash":
        row["execution"]["applications"][0]["dispatch"]["boundary"]["sha256"] = "f" * 64
        rehash_execution(row)
    else:
        row["info"]["physical_time_s"] = 123.0
    try:
        left = audit.physical_projection(native, residual=False)
        right = audit.physical_projection(initial, residual=True)
    except (audit.AuditError, ValueError):
        return
    assert left != right


def test_all_comparators_may_complete_without_being_declared_safe_or_feasible(tmp_path):
    root = fabricate(tmp_path / "all-complete", complete_actors=batch.ACTORS)
    packet = audit.audit_closed_batch(root, fixture=True)
    assert all(
        s["eligible_for_descriptive_comparison"]
        for actor in packet["actors"].values()
        for s in actor["slots"]
    )
    assert all(
        not s["summary"]["hard_safety_passed"]
        and not s["summary"]["all_cycles_feasible"]
        for actor in packet["actors"].values()
        for s in actor["slots"]
    )
    assert packet["performance_means_computed"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_reset",
        "changed_reset",
        "wrong_progress",
        "extra_file",
        "missing_rollout",
    ],
)
def test_allocator_extra_artifacts_are_endpoint_bound(tmp_path, mutation):
    root = fabricate(tmp_path / "batch")
    folder = root / "training" / batch.ARMS[0].name / "allocator"
    if mutation == "missing_reset":
        (folder / "reset-001.json").unlink()
    elif mutation == "missing_rollout":
        (folder / "rollout-004.json").unlink()
    elif mutation == "extra_file":
        write(folder / "unexpected.json", {"fixture": True})
    elif mutation == "changed_reset":
        path = folder / "reset-001.json"
        value = audit.decode(path.read_bytes())
        value["episode_seed"] += 1
        write(path, value)
    else:
        path = folder / "rollout-002.json"
        value = audit.decode(path.read_bytes())
        value["decisions_consumed_by_environment"][0] += 1
        write(path, value)
    with pytest.raises((audit.AuditError, ValueError)):
        audit.audit_closed_batch(root, fixture=True)


def test_timeout_leftover_checkpoint_remains_unadmitted(tmp_path):
    root = fabricate(
        tmp_path / "batch",
        failed_training=batch.ARMS[0].name,
        retain_failed_prefix=True,
    )
    directory = root / "training" / batch.ARMS[0].name
    (directory / "final-policy.zip").write_bytes(
        b"partial checkpoint bytes after timeout"
    )
    write(directory / "result.json", {"partial": "never admitted"})
    packet = audit.audit_closed_batch(root, fixture=True)
    training = packet["training"][batch.ARMS[0].name]
    assert len(training["unadmitted_artifacts"]) == 2 and training["result"] is None
    assert all(
        s["summary"] is None for s in packet["actors"]["initial-mpc-residual"]["slots"]
    )


def test_wrong_plain_zip_interface_is_rejected_before_model_reopen(
    tmp_path, monkeypatch
):
    root = fabricate(tmp_path / "batch")
    directory = root / "training" / batch.ARMS[0].name
    path = directory / "final-policy.zip"
    with zipfile.ZipFile(path) as original:
        data = json.loads(original.read("data"))
    data["research_identity"]["representation"] = "residual-h"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps(data))
    result = audit.decode((directory / "result.json").read_bytes())
    result["final_policy"] = identity(path)
    write(directory / "result.json", result)
    monkeypatch.setattr(
        audit,
        "verify_loaded_checkpoint",
        lambda *args: pytest.fail("reopened before plain identity check"),
    )
    with pytest.raises(ValueError, match="interface"):
        audit.audit_closed_batch(root, fixture=True)


@pytest.mark.parametrize("mutation", ["missing", "actor", "seed", "started_at"])
def test_attempt_slot_bindings_fail_closed_without_outcomes(tmp_path, mutation):
    path = tmp_path / "replay-attempt.json"
    value = {
        "actor": "OperatingMPC",
        "seed": batch.DEV_SEEDS[0],
        "started_at": "fixture",
    }
    if mutation != "missing":
        value[mutation] = None if mutation == "started_at" else "foreign-slot"
        write(path, value)
    with pytest.raises(audit.AuditError):
        audit.verify_attempt(
            audit.Reader(), path, "OperatingMPC", batch.DEV_SEEDS[0], required=True
        )


def test_failed_spawn_may_have_no_attempt_without_becoming_an_episode(tmp_path):
    assert (
        audit.verify_attempt(
            audit.Reader(),
            tmp_path / "absent.json",
            "H",
            batch.DEV_SEEDS[0],
            required=False,
        )
        is None
    )


def test_caught_learning_failure_retains_valid_allocator_and_optimizer_prefixes(
    tmp_path,
):
    root = fabricate(
        tmp_path / "batch",
        failed_training=batch.ARMS[0].name,
        retain_failed_prefix=True,
    )
    directory = root / "training" / batch.ARMS[0].name
    prefix = audit.decode((directory / "allocator" / "rollout-001.json").read_bytes())
    rollout = audit.decode(
        (directory / "diagnostics" / "rollout-001.json").read_bytes()
    )
    write(
        directory / "training-failure.json",
        {
            "schema": batch.SCHEMA,
            "failed_at": "fixture",
            "failure": {
                "type": "ValueError",
                "message": "controlled numerical failure",
            },
            "num_timesteps": 8192,
            "allocator_prefix": prefix,
            "rollout_ledgers": [rollout],
            "successful_endpoint": False,
            "registered_evaluation_eligible": False,
        },
    )
    packet = audit.audit_closed_batch(root, fixture=True)
    failure = packet["training"][batch.ARMS[0].name]["allocator_prefixes"][
        "training_failure"
    ]
    assert failure["num_timesteps"] == 8192 and len(failure["rollout_ledgers"]) == 1
    assert len(failure["allocator_prefix"]["ledger"]) == 16
    assert packet["training"][batch.ARMS[0].name]["result"] is None


def test_actual_producer_seeds_optimizer_and_action_space_but_never_environment(
    monkeypatch,
):
    """Exercise the real override with spies: no model, environment reset or RNG mutation."""
    from types import SimpleNamespace
    from plant_backend.rl.v2.training import constrained_ppo

    calls = []
    monkeypatch.setattr(
        constrained_ppo,
        "sb3_set_random_seed",
        lambda seed, **kwargs: calls.append(("optimizer", seed, kwargs)),
    )
    fake = SimpleNamespace(
        device=torch.device("cpu"),
        action_space=SimpleNamespace(
            seed=lambda seed: calls.append(("action_space", seed))
        ),
        env=SimpleNamespace(
            seed=lambda *args: pytest.fail("optimizer seed reached environment")
        ),
    )
    constrained_ppo.ConstrainedPPO.set_random_seed(fake, 101)
    assert calls == [("optimizer", 101, {"using_cuda": False}), ("action_space", 101)]
    calls.clear()
    constrained_ppo.ConstrainedPPO.set_random_seed(fake, None)
    assert calls == []


def test_fabricated_initial_framework_seeds_match_the_producer_contract(tmp_path):
    directory = tmp_path / "prefix-only"
    fabricate_allocator_prefixes(directory, count=4)
    raw = audit.decode((directory / "allocator" / "rollout-004.json").read_bytes())
    assert len(raw["ledger"]) == 52
    assert all(entry["ignored_framework_seed"] is None for entry in raw["ledger"])
    snapshot = audit._validate_allocator_prefix(raw, batch.template(tmp_path))
    assert [entry["episode_seed"] for entry in snapshot["ledger"]["entries"]] == list(
        range(9620000, 9620052)
    )
    assert [cursor["decision_offset"] for cursor in snapshot["cursors"]] == [128] * 4


@pytest.mark.parametrize(
    "mutation",
    [
        "initial_numeric",
        "mixed_initial",
        "later_numeric",
        "wrong_physical_seed",
        "duplicate_physical_seed",
        "wrong_progress",
    ],
)
def test_corrected_prefix_check_rejects_numeric_framework_seeds_and_physical_custody_changes(
    tmp_path, mutation
):
    from dataclasses import replace
    from plant_backend.rl.v2.training.episode_allocator import (
        ResetManifestLedgerEntryV1,
    )

    directory = tmp_path / "prefix-only"
    fabricate_allocator_prefixes(directory, count=1)
    raw = audit.decode((directory / "allocator" / "rollout-001.json").read_bytes())
    entries = [ResetManifestLedgerEntryV1.from_dict(value) for value in raw["ledger"]]
    if mutation == "initial_numeric":
        entries[:4] = [
            replace(entry, ignored_framework_seed=101 + entry.environment_index)
            for entry in entries[:4]
        ]
    elif mutation == "mixed_initial":
        entries[0] = replace(entries[0], ignored_framework_seed=101)
    elif mutation == "later_numeric":
        entries[4] = replace(entries[4], ignored_framework_seed=101)
    elif mutation == "wrong_physical_seed":
        entries[0] = replace(entries[0], episode_seed=9620100)
    elif mutation == "duplicate_physical_seed":
        entries[1] = replace(entries[1], episode_seed=entries[0].episode_seed)
    else:
        raw["decisions_consumed_by_environment"][0] += 1
    raw["ledger"] = [entry.to_dict() for entry in entries]
    with pytest.raises((audit.AuditError, ValueError)) as error:
        audit._validate_allocator_prefix(raw, batch.template(tmp_path))
    if mutation == "initial_numeric":
        assert "optimizer-only producer" in str(error.value)

"""Batch004 fixtures only; controlled solves and physical seeds 9_992_000..099."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import gzip
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from plant_backend.rl.v2.training import research_mpc_constraint_batch as batch
from plant_backend.rl.v2.training import ppo
from plant_backend.rl.v2.training.custody_io import publish_json
from plant_backend.rl.v2.training.custody_json import canonical_bytes


@pytest.fixture(autouse=True)
def no_production_solves(monkeypatch):
    from plant_backend import operational_mpc

    monkeypatch.setattr(
        operational_mpc.OperationalMPCController,
        "build_model",
        lambda *a, **kw: pytest.fail("uncontrolled MPC model"),
    )
    monkeypatch.setattr(
        operational_mpc,
        "solve_economic_mpc_model",
        lambda *a, **kw: pytest.fail("uncontrolled MPC solve"),
    )


@pytest.fixture
def cheap_prior(monkeypatch):
    from plant_backend import operational_mpc
    from plant_backend.rl.v2.economic_mpc_schema import EconomicMPCPlanError
    from plant_backend.rl.v2.mpc_schema import domain_sha256

    def model(controller, observation, **kwargs):
        return SimpleNamespace(
            sha256=domain_sha256("batch004-controlled-solver", observation.sha256),
            horizon_intervals=min(
                96, controller.settings.task_interval_count - controller._next_decision
            ),
        )

    def solve(*a, **kw):
        raise EconomicMPCPlanError("batch004 controlled fallback fixture")

    monkeypatch.setattr(operational_mpc.OperationalMPCController, "build_model", model)
    monkeypatch.setattr(operational_mpc, "solve_economic_mpc_model", solve)


def receipt(root, stage, name, extra, codes=()):
    value = {
        "stage": stage,
        "name": name,
        "codes": list(codes),
        "closed_at": "fixture-closed",
        "process": {
            "exit_code": 1 if codes else 0,
            "started": True,
            "command": [batch.sys.executable, "-m", batch.MODULE, stage, *extra],
        },
    }
    publish_json(root / "receipts" / f"{name}.json", value)
    return value


def test_prespecified_budget_and_unchanged_task():
    from plant_backend.rl.v2.training import research_parallel_supervision as helper
    from plant_backend.rl.v2.training.registered_contract import registered_config
    from plant_backend.rl.v2.training import research_mpc_residual_batch as old

    spec = batch.specification(batch.ROOT)
    config = batch.template(batch.ROOT)
    original = registered_config(batch.ROOT / "unused", batch.SCALAR, 101)
    assert ppo.policy_contract(config) == ppo.policy_contract(original)
    assert config.ppo == original.ppo
    assert batch.TOTAL_STEPS == 65_536
    assert batch.OPTIMIZER_SEED == 101
    assert batch.TRAIN_SEED_BASE == 9_630_000
    assert batch.TRAIN_SEED_LAST == 9_630_099
    assert batch.DEV_SEEDS == tuple(range(9_700_300, 9_700_308))
    assert len(batch.ARMS) == 2
    assert [a.algorithm for a in batch.ARMS] == [batch.SCALAR, batch.PI]
    assert all(
        (a.reward_scale, a.initial_std, a.residual) == (0.001, 0.2, True)
        for a in batch.ARMS
    )
    assert batch.INITIAL_SOURCES == {"initial-mpc-residual": batch.ARMS[0]}
    assert len(batch.ACTORS) == 7
    assert spec["evaluation_slots"] == spec["replay_slots"] == 56
    rollout_size = config.n_envs * config.ppo.n_steps
    assert rollout_size == spec["transitions_per_rollout"] == 8192
    assert config.total_timesteps // rollout_size == spec["training_rollouts"] == 8
    assert (
        config.total_timesteps * config.ppo.n_epochs // config.ppo.batch_size
        == spec["minibatch_optimizer_steps"]
        == 2560
    )
    decisions = config.episode.decision_steps(config.plant)
    assert config.total_timesteps // 4 // decisions + 1 == 25
    assert spec["training_resets_per_arm"] == 100
    assert spec["framework_environment_seed"] is None
    assert batch.TRAIN_TIMEOUT == 43200
    assert (
        spec["training_supervisor_startup_seconds"] == helper._STARTUP_SECONDS == 30.0
    )
    assert batch.EVAL_TIMEOUT == 600
    assert batch.LOG_LIMIT == 64 * 1024**2
    assert batch.ARCHIVE_LIMIT == 256 * 1024**2
    assert batch.MINIMUM_FREE == 10 * 1024**3
    assert spec["registered_evaluation_eligible"] is False
    assert old.TOTAL_STEPS == 32768 and len(old.ARMS) == 1


def test_repair_replication_is_new_root_with_explicit_reused_seed_provenance():
    from plant_backend.rl.v2.training import research_parallel_supervision as helper
    from plant_backend.rl.v2.operating_mpc_forecast import (
        OPERATING_FORECAST_PROJECTION_VERSION,
    )

    spec = batch.specification(batch.ROOT)
    assert batch.ROOT.name == "adaptive-rl-batch-005"
    assert batch.SCHEMA == "plant-adaptive-rl-mpc-constraint-batch-005"
    repair = spec["repair_replication"]
    assert repair["previous_batch"] == "adaptive-rl-batch-004"
    assert (
        repair["previous_source_commit"] == "4f908fcaeaeed417a793204778d87f62c29ec768"
    )
    assert repair["previous_report_sha256"] == (
        "f4b7c3a4796f99f45686db1ea6fe9d6250a260c79d4511094a7e88c7072a68f3"
    )
    assert "reused development" in repair["training_seed_status"]
    assert "never opened" in repair["development_seed_status"]
    assert "no resume" in repair["initialization"]
    assert repair["task_reward_learner_authority_changes"] is False
    assert repair["registered_projection_changes"] is False
    assert repair["operating_solar_projection"] == OPERATING_FORECAST_PROJECTION_VERSION
    assert spec["training_supervision_schema"] == helper.SCHEMA
    assert spec["training_log_monitor"] == helper.monitor_settings()
    spec["training_log_monitor"].clear()
    assert batch.specification(batch.ROOT)["training_log_monitor"]


def test_constructor_matches_exact_actor_reward_tensors_and_cost_independence(
    tmp_path, monkeypatch
):
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
        lambda *a, **kw: pytest.fail("constructor reset"),
    )
    config = batch.template(tmp_path)
    weights = []
    for arm in batch.ARMS:
        directory = tmp_path / "training" / arm.name
        directory.mkdir(parents=True)
        model, allocator, vector = batch.make_learner(config, arm, seed_base=9_992_000)
        try:
            assert not allocator.ledger
            assert allocator.decisions_consumed_by_environment == (0, 0, 0, 0)
            assert (model.observation_space.shape, model.action_space.shape) == (
                (139,),
                (4,),
            )
            for env in vector.envs:
                assert isinstance(env, batch.OptimizerRewardScale)
                assert isinstance(env.env, Monitor)
                assert isinstance(env.env.env, SeedCustodyWrapperV1)
                assert isinstance(env.env.env.env, ResearchMPCResidualWrapper)
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
            saved = {k: v.clone() for k, v in model.policy.state_dict().items()}
            weights.append(saved)
            buffer = io.BytesIO()
            torch.save(saved, buffer)
            batch.publish_bytes(directory / "initial-policy.pt", buffer.getvalue())
            publish_json(
                directory / "initial-policy-record.json",
                batch.initial_policy_record(directory, model, config, arm),
            )
        finally:
            vector.close()
    assert not any(k.startswith("cost_critic.") for k in weights[0])
    assert any(k.startswith("cost_critic.") for k in weights[1])
    assert all(torch.equal(v, weights[1][k]) for k, v in weights[0].items())
    matching = batch.verify_initial_matching(tmp_path, config)
    assert matching["matched"] is True
    assert len(matching["initial_policy_records"]) == 2
    assert matching["cost_critics_excluded"] is True
    directory = tmp_path / "training" / batch.ARMS[1].name
    record = json.loads((directory / "initial-policy-record.json").read_text())
    weights[1]["value_net.bias"].add_(1)
    buffer = io.BytesIO()
    torch.save(weights[1], buffer)
    (directory / "initial-policy.pt").write_bytes(buffer.getvalue())
    record["initial_policy"] = batch.file_record(directory / "initial-policy.pt")
    record["all_tensors"] = batch.tensor_records(weights[1], shared_only=False)
    record["actor_reward_tensors"] = batch.tensor_records(weights[1], shared_only=True)
    (directory / "initial-policy-record.json").write_text(json.dumps(record))
    with pytest.raises(batch.TrainingIntegrityError, match="tensors differ"):
        batch.verify_initial_matching(tmp_path, config)


@pytest.mark.parametrize("arm", batch.ARMS, ids=lambda a: a.name)
def test_real_eight_rollout_endpoint_dual_and_checkpoint(
    tmp_path, cheap_prior, arm, monkeypatch
):
    import torch

    torch.set_num_threads(1)
    original = batch.template(tmp_path)
    # Four environments each cross one full seven-day episode; eight rollouts.
    config = replace(
        original,
        total_timesteps=3072,
        ppo=replace(original.ppo, n_steps=96, batch_size=96, n_epochs=1),
    )
    result = batch.train_cell(tmp_path, config, arm, seed_base=9_992_010)
    assert result["total_timesteps"] == 3072
    for directory, pattern in (
        ("diagnostics", "diagnostic-*.json"),
        ("diagnostics", "rollout-*.json"),
        ("allocator", "rollout-*.json"),
        ("allocator", "reset-*.json"),
    ):
        assert len(list((tmp_path / directory).glob(pattern))) == 8
    ledger = result["constraint_state"]["payload"]["rollout_ledgers"]
    assert len(ledger) == 8
    assert sum(item["optimizer_steps"] for item in ledger) == 32
    resets = result["allocator"]["ledger"]
    # Snapshot serializes its immutable ledger as an envelope.
    if isinstance(resets, dict):
        resets = resets["entries"]
    assert [r["episode_seed"] for r in resets] == list(range(9_992_010, 9_992_018))
    assert all(r["ignored_framework_seed"] is None for r in resets)
    assert all(c["decision_offset"] == 96 for c in result["allocator"]["cursors"])
    if arm.algorithm == batch.PI:
        assert [l["multiplier_completed_update_count"] for l in ledger] == [
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
        ]
        assert ledger[6]["completed_episode_count"] == 4
        assert any(ledger[6]["mean_completed_episode_cost"])
        assert ledger[6]["multipliers"] != [1.0] * 4
    else:
        assert all(l["multiplier_completed_update_count"] == 0 for l in ledger)
        assert all(l["multipliers"] == [0.0] * 4 for l in ledger)
        assert all(l["cost_value_losses"] == [0.0] * 4 for l in ledger)
    with monkeypatch.context() as admission:
        admission.setattr(
            batch.PlantCyclicMarketEnvV2Beta,
            "reset",
            lambda *a, **kw: pytest.fail("checkpoint admission reset"),
        )
        model = batch.load_final(tmp_path, config, arm)
        initial = batch.load_initial(tmp_path, config, arm)
    assert (
        batch.tensor_records(model.policy.state_dict(), shared_only=False)
        == result["final_policy_tensors"]
    )
    action, _ = initial.predict(np.zeros(139, dtype=np.float32), deterministic=True)
    assert np.array_equal(action, np.zeros(4))
    with (tmp_path / "final-policy.zip").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        batch.load_final(tmp_path, config, arm)


def test_reward_units_preserve_constraint_transition(tmp_path, cheap_prior):
    config = batch.template(tmp_path)
    raw = batch.control_environment(config, batch.ARMS[0])
    scaled = batch.OptimizerRewardScale(
        batch.control_environment(config, batch.ARMS[1]), 0.001
    )
    try:
        a, ai = raw.reset(seed=9_992_030)
        b, bi = scaled.reset(seed=9_992_030)
        assert np.array_equal(a, b) and canonical_bytes(ai) == canonical_bytes(bi)
        for _ in range(96):
            action = np.array([-0.3, -0.3, -0.3, 1], dtype=np.float32)
            a, ar, at, ax, ai = raw.step(action)
            b, br, bt, bx, bi = scaled.step(action)
            assert np.array_equal(a, b) and br == ar * 0.001 and (at, ax) == (bt, bx)
            assert canonical_bytes(
                {k: v for k, v in ai.items() if k != "dispatcher_elapsed_ns"}
            ) == canonical_bytes(
                {k: v for k, v in bi.items() if k != "dispatcher_elapsed_ns"}
            )
        assert ai["cycle_constraint_transition"]["cycle_settled"] is True
    finally:
        raw.close()
        scaled.close()


def test_one_hundred_reset_stream_without_physics():
    from test_rl_v2_episode_allocator import _AuditableFixtureEnv
    from stable_baselines3.common.vec_env import DummyVecEnv

    allocator = batch.EpisodeSeedAllocatorV1.fixture(base_seed=9_992_000)
    vector = DummyVecEnv(
        [
            lambda i=i: allocator.wrap(
                _AuditableFixtureEnv(horizon=672), environment_index=i
            )
            for i in range(4)
        ]
    )
    try:
        vector.reset()
        for _ in range(16384):
            vector.step(np.zeros(4, dtype=int))
        assert len(allocator.ledger) == 100
        assert [e.episode_seed for e in allocator.ledger] == list(
            range(9_992_000, 9_992_100)
        )
        assert all(e.ignored_framework_seed is None for e in allocator.ledger)
        assert allocator.decisions_consumed_by_environment == (16384,) * 4
        assert [c.decision_offset for c in allocator.cursors] == [256] * 4
        assert [c.current_episode_ordinal for c in allocator.cursors] == [24] * 4
    finally:
        vector.close()


def fake_campaign(tmp_path, monkeypatch):
    root = tmp_path / "batch"
    monkeypatch.setattr(batch, "ROOT", root)
    monkeypatch.setattr(batch, "DEV_SEEDS", tuple(range(9_992_080, 9_992_088)))
    runtime = SimpleNamespace(git_head="a" * 40, to_dict=lambda: {"fixture": True})
    monkeypatch.setattr(batch, "capture", lambda **kw: runtime)
    monkeypatch.setattr(batch, "require_plan", lambda: {})
    monkeypatch.setattr(batch, "verify_initial_matching", lambda *a: {"fixture": True})
    gate = tmp_path / "gate.json"
    publish_json(gate, {"passed": True, "source_commit": runtime.git_head})
    return root, gate


@pytest.mark.parametrize(
    "failed_arm", [None, "A-mpc-residual-scalar", "B-mpc-residual-pi"]
)
def test_whole_fake_graph_retains_56_plus_56_slots(tmp_path, monkeypatch, failed_arm):
    root, gate = fake_campaign(tmp_path, monkeypatch)
    calls = []
    failed_actors = {failed_arm} if failed_arm else set()
    if failed_arm == batch.ARMS[0].name:
        failed_actors.add("initial-mpc-residual")

    def wave():
        return {
            "receipts": [
                receipt(
                    root,
                    "_train",
                    f"train-{a.name}",
                    ["--cell", a.name],
                    ["child_nonzero_exit"] if a.name == failed_arm else [],
                )
                for a in batch.ARMS
            ],
            "fatal": None,
            "supervisors": [{"fixture": a.name} for a in batch.ARMS],
        }

    def child(stage, name, timeout, extra):
        calls.append((stage, name))
        assert timeout == 600
        return receipt(
            root,
            stage,
            name,
            extra,
            ["child_nonzero_exit"] if extra[1] in failed_actors else [],
        )

    monkeypatch.setattr(batch, "training_wave", wave)
    monkeypatch.setattr(batch, "closed_child", child)
    report = batch.run(gate, authorized=True)
    assert report["status"] == "closed" and report["failure"] is None
    assert len(report["receipts"]) == 114
    assert (
        len(report["planned_slots"]["evaluation"])
        == len(report["planned_slots"]["replay"])
        == 56
    )
    assert [stage for stage, name in calls] == ["_evaluate", "_replay"] * 56
    assert sum(bool(r["codes"]) for r in report["receipts"]) == (
        1 + 16 * len(failed_actors) if failed_arm else 0
    )
    with pytest.raises(ValueError, match="no retry"):
        batch.run(gate, authorized=True)


@pytest.mark.parametrize(
    "fatal", ["root_custody_lost", "allocator", "source", "mpc", "checkpoint"]
)
def test_fatal_wave_stops_entire_graph_but_retains_all_planned_slots(
    tmp_path, monkeypatch, fatal
):
    root, gate = fake_campaign(tmp_path, monkeypatch)

    def wave():
        return {
            "receipts": [
                receipt(
                    root,
                    "_train",
                    f"train-{a.name}",
                    ["--cell", a.name],
                    ["child_nonzero_exit"],
                )
                for a in batch.ARMS
            ],
            "supervisors": [{"closed": True}, {"closed": True}],
            "fatal": {"code": fatal, "detail": "fixture"},
        }

    monkeypatch.setattr(batch, "training_wave", wave)
    monkeypatch.setattr(
        batch, "closed_child", lambda *a: pytest.fail("postfatal child")
    )
    report = batch.run(gate, authorized=True)
    assert report["status"] == "aborted"
    assert len(report["receipts"]) == 2
    assert not list((root / "evaluations").iterdir())
    assert len(report["planned_slots"]["replay"]) == 56
    assert (root / "training-wave.json").is_file()


@pytest.mark.parametrize(
    "actor,source",
    [
        ("initial-mpc-residual", 0),
        ("A-mpc-residual-scalar", 0),
        ("B-mpc-residual-pi", 1),
    ],
)
def test_failed_training_blocks_exact_source_and_never_substitutes(
    tmp_path, monkeypatch, actor, source
):
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    (tmp_path / "receipts").mkdir()
    for index, arm in enumerate(batch.ARMS):
        receipt(
            tmp_path,
            "_train",
            f"train-{arm.name}",
            ["--cell", arm.name],
            ["child_timeout"] if index == source else [],
        )
    directory = tmp_path / "evaluations" / f"9992080-{actor}"
    directory.mkdir(parents=True)
    monkeypatch.setattr(
        batch, "load_initial", lambda *a: pytest.fail("substituted initial")
    )
    monkeypatch.setattr(
        batch, "load_final", lambda *a: pytest.fail("substituted final")
    )
    with pytest.raises(batch.TrainingUnavailable, match="successfully closed training"):
        batch.evaluate_cell(actor, 9_992_080)
    assert (directory / "attempt.json").is_file()


def test_failed_evaluation_retains_ordinary_replay_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    (tmp_path / "receipts").mkdir()
    directory = tmp_path / "evaluations" / "9992080-H"
    directory.mkdir(parents=True)
    receipt(
        tmp_path,
        "_evaluate",
        "evaluate-9992080-H",
        ["--actor", "H", "--seed", "9992080"],
        ["child_timeout"],
    )
    monkeypatch.setattr(batch, "require_plan", lambda: {})
    with pytest.raises(batch.TrainingUnavailable, match="evaluation child"):
        batch.execute_child(
            lambda: batch.replay_cell("H", 9_992_080), artifact_phase=True
        )
    assert (directory / "replay-attempt.json").is_file()


@pytest.mark.parametrize(
    "kind", ["allocator", "mpc", "endpoint", "source", "artifact", "numerical"]
)
def test_reserved_integrity_exit_and_local_numerical_failure(monkeypatch, kind):
    monkeypatch.setattr(batch, "require_plan", lambda: {})
    errors = {
        "allocator": batch.EpisodeSeedCustodyError,
        "mpc": batch.MPCResidualIntegrityError,
        "endpoint": batch.TrainingIntegrityError,
        "source": ValueError,
        "artifact": ValueError,
        "numerical": ValueError,
    }

    def failure():
        raise errors[kind](f"fixture {kind}")

    if kind == "source":
        monkeypatch.setattr(batch, "require_plan", failure)
    if kind == "numerical":
        with pytest.raises(ValueError, match="numerical"):
            batch.execute_child(failure, artifact_phase=False)
    else:
        assert batch.execute_child(failure, artifact_phase=kind == "artifact") == 86


def test_archive_limit_and_slot_binding(tmp_path, monkeypatch):
    from plant_backend.rl.v2.training import research_evaluation

    monkeypatch.setattr(batch, "ROOT", tmp_path)
    (tmp_path / "receipts").mkdir()
    actor, seed = "H", 9_992_080
    directory = tmp_path / "evaluations" / f"{seed}-{actor}"
    directory.mkdir(parents=True)
    episode = {"seed": seed, "summary": {"actor": "H", "policy_id": "H"}}
    monkeypatch.setattr(
        research_evaluation, "evaluate_episode", lambda *a, **kw: episode
    )
    monkeypatch.setattr(batch, "ARCHIVE_LIMIT", 1)
    with pytest.raises(ValueError, match="stored-size cap"):
        batch.evaluate_cell(actor, seed)
    assert not (directory / "episode.json.gz").exists()
    monkeypatch.setattr(batch, "ARCHIVE_LIMIT", 256 * 1024**2)
    receipt(
        tmp_path,
        "_evaluate",
        f"evaluate-{seed}-{actor}",
        ["--actor", actor, "--seed", str(seed)],
    )
    archive = directory / "episode.json.gz"
    archive.write_bytes(gzip.compress(canonical_bytes(episode)))
    publish_json(
        directory / "summary.json",
        {
            "actor": actor,
            "seed": seed + 1,
            "summary": episode["summary"],
            "archive": batch.file_record(archive),
        },
    )
    monkeypatch.setattr(
        research_evaluation,
        "verify_episode_replay",
        lambda *a: pytest.fail("cross-slot replay"),
    )
    with pytest.raises(ValueError, match="slot identity"):
        batch.replay_cell(actor, seed)


def test_source_gate_and_resource_admission_precede_root_claim(tmp_path, monkeypatch):
    root = tmp_path / "absent"
    monkeypatch.setattr(batch, "ROOT", root)
    gate = tmp_path / "gate.json"
    publish_json(gate, {"passed": True, "source_commit": "a" * 40})
    with pytest.raises(ValueError, match="authorization"):
        batch.run(gate, authorized=False)
    monkeypatch.setattr(
        batch, "capture", lambda **kw: SimpleNamespace(git_head="b" * 40)
    )
    with pytest.raises(ValueError, match="exact clean source"):
        batch.run(gate, authorized=True)
    monkeypatch.setattr(
        batch, "capture", lambda **kw: SimpleNamespace(git_head="a" * 40)
    )
    monkeypatch.setattr(batch.shutil, "disk_usage", lambda *a: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="10 GiB"):
        batch.run(gate, authorized=True)
    assert not root.exists()


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
def test_failure_preserves_initial_policy_and_reached_prefix(
    tmp_path, monkeypatch, learning_fails
):
    import torch

    torch.set_num_threads(1)
    config = batch.template(tmp_path)
    arm = batch.ARMS[1]
    model, allocator, vector = batch.make_learner(config, arm, seed_base=9_992_070)
    monkeypatch.setattr(
        batch, "make_learner", lambda *a, **kw: (model, allocator, vector)
    )

    def learn(**kw):
        if learning_fails:
            raise ValueError("ordinary numerical failure")

    def bad_endpoint(*a):
        raise ValueError("wrong exact endpoint")

    monkeypatch.setattr(model, "learn", learn)
    monkeypatch.setattr(batch, "validate_training_endpoint", bad_endpoint)
    expected = ValueError if learning_fails else batch.TrainingIntegrityError
    with pytest.raises(expected):
        batch.train_cell(tmp_path, config, arm, seed_base=9_992_070)
    failure = json.loads((tmp_path / "training-failure.json").read_text())
    assert failure["num_timesteps"] == 0
    assert failure["allocator_prefix"]["ledger"] == []
    assert failure["rollout_ledgers"] == []
    assert failure["successful_endpoint"] is False
    assert (tmp_path / "initial-policy-record.json").is_file()
    assert not (tmp_path / "result.json").exists()


def test_absent_initial_record_requires_ordinary_failed_training(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    (tmp_path / "receipts").mkdir()
    for arm in batch.ARMS:
        (tmp_path / "training" / arm.name).mkdir(parents=True)
        receipt(
            tmp_path,
            "_train",
            f"train-{arm.name}",
            ["--cell", arm.name],
            ["child_timeout"],
        )
    result = batch.verify_initial_matching(tmp_path, batch.template(tmp_path))
    assert result["matched"] is False
    assert result["unavailable_arms"] == [a.name for a in batch.ARMS]
    first = tmp_path / "receipts" / f"train-{batch.ARMS[0].name}.json"
    value = json.loads(first.read_text())
    value["codes"] = []
    value["process"]["exit_code"] = 0
    first.write_text(json.dumps(value))
    with pytest.raises(batch.TrainingIntegrityError, match="omitted its initial"):
        batch.verify_initial_matching(tmp_path, batch.template(tmp_path))


def test_corrupt_reached_initial_is_fatal_even_if_training_failed(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    (tmp_path / "receipts").mkdir()
    for arm in batch.ARMS:
        directory = tmp_path / "training" / arm.name
        directory.mkdir(parents=True)
        receipt(
            tmp_path,
            "_train",
            f"train-{arm.name}",
            ["--cell", arm.name],
            ["child_timeout"],
        )
        (directory / "initial-policy-record.json").write_text("malformed")
    with pytest.raises(ValueError):
        batch.verify_initial_matching(tmp_path, batch.template(tmp_path))


def test_callback_publication_failure_is_reserved_integrity(tmp_path, monkeypatch):
    from stable_baselines3.common.callbacks import BaseCallback

    class BrokenRecorder(BaseCallback):
        def _on_step(self):
            return True

        def _on_rollout_end(self):
            raise OSError("injected immutable evidence failure")

    monkeypatch.setattr(batch, "training_callback", lambda *a: BrokenRecorder())
    callback = batch.cell_callback(
        tmp_path, batch.EpisodeSeedAllocatorV1.fixture(base_seed=9_992_070)
    )
    with pytest.raises(batch.TrainingIntegrityError, match="callback failed"):
        callback.on_rollout_end()


def test_parallel_helper_receives_exact_two_isolated_attempts(tmp_path, monkeypatch):
    from plant_backend.rl.v2.training import research_parallel_supervision as helper

    monkeypatch.setattr(batch, "ROOT", tmp_path)
    seen = []

    def fake(attempts, **kwargs):
        seen.extend(attempts)
        assert kwargs["supervisor_directory"] == tmp_path / "supervisors"
        assert kwargs["worker_module"] == batch.MODULE
        assert kwargs["worker_function"] == "closed_child"
        assert kwargs["check_integrity"] is batch.require_plan
        assert kwargs["minimum_free_bytes"] == 10 * 1024**3
        assert kwargs["supervisor_log_limit_bytes"] == 64 * 1024**2
        assert kwargs["child_log_limit_bytes"] == 64 * 1024**2
        assert kwargs["integrity_check_interval_seconds"] == 30.0
        return {"fixture": True}

    monkeypatch.setattr(helper, "run_training_wave", fake)
    assert batch.training_wave() == {"fixture": True}
    assert len(seen) == 2
    for arm, attempt in zip(batch.ARMS, seen):
        assert attempt.command == (
            batch.sys.executable,
            "-m",
            batch.MODULE,
            "_train",
            "--cell",
            arm.name,
        )
        assert attempt.timeout_seconds == 43200
        assert attempt.receipt == tmp_path / "receipts" / f"train-{arm.name}.json"


def test_supervisor_sets_threads_before_host_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    (tmp_path / "receipts").mkdir()
    events = []
    monkeypatch.setattr(batch, "configure_threads", lambda: events.append("threads"))
    monkeypatch.setattr(batch, "require_plan", lambda: events.append("admission"))
    monkeypatch.setattr(
        batch, "capture_process", lambda *a, **kw: ({"exit_code": 0}, set())
    )
    batch.closed_child("_train", "fixture", 10, [])
    assert events == ["threads", "admission", "admission"]


@pytest.mark.parametrize(
    "failure", ["exit86", "root_custody_lost", "unknown", "unclosed"]
)
def test_missing_initial_cannot_hide_integrity_loss(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    (tmp_path / "receipts").mkdir()
    for arm in batch.ARMS:
        (tmp_path / "training" / arm.name).mkdir(parents=True)
        receipt(
            tmp_path,
            "_train",
            f"train-{arm.name}",
            ["--cell", arm.name],
            ["child_timeout"],
        )
    first = tmp_path / "receipts" / f"train-{batch.ARMS[0].name}.json"
    value = json.loads(first.read_text())
    if failure == "exit86":
        value["process"]["exit_code"] = 86
    elif failure == "unclosed":
        value.pop("closed_at")
    else:
        value["codes"] = [failure]
    first.write_text(json.dumps(value))
    with pytest.raises((ValueError, batch.TrainingIntegrityError)):
        batch.verify_initial_matching(tmp_path, batch.template(tmp_path))


def test_initial_load_restores_caller_rng_without_reset(tmp_path, monkeypatch):
    import torch
    from plant_backend.rl.v2.training.constrained_ppo import LearnerRNGStateV1

    torch.set_num_threads(1)
    monkeypatch.setattr(
        batch.PlantCyclicMarketEnvV2Beta,
        "reset",
        lambda *a, **kw: pytest.fail("initial admission reset"),
    )
    config = batch.template(tmp_path)
    arm = batch.ARMS[0]
    model, allocator, vector = batch.make_learner(config, arm, seed_base=9_992_070)
    try:
        data = io.BytesIO()
        torch.save(model.policy.state_dict(), data)
        batch.publish_bytes(tmp_path / "initial-policy.pt", data.getvalue())
        publish_json(
            tmp_path / "initial-policy-record.json",
            batch.initial_policy_record(tmp_path, model, config, arm),
        )
    finally:
        vector.close()
    monkeypatch.setattr(
        batch,
        "read_training_result",
        lambda *a: {
            "initial_policy": batch.file_record(tmp_path / "initial-policy.pt")
        },
    )
    before = LearnerRNGStateV1.capture().to_dict()
    batch.load_initial(tmp_path, config, arm)
    assert LearnerRNGStateV1.capture().to_dict() == before
    assert allocator.ledger == ()

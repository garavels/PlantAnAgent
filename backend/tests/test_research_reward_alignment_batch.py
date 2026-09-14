"""Reward-alignment orchestration fixtures; no production resets or learning."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import plistlib
from types import SimpleNamespace

import pytest

from plant_backend.rl.v2.training import research_reward_alignment_batch as batch
from plant_backend.rl.v2.training.custody_io import publish_json
from plant_backend.rl.v2.training.custody_json import canonical_bytes


@pytest.fixture(autouse=True)
def prevent_production_learning(monkeypatch):
    for name in ("train_cell", "make_learner", "load_final", "read_initial_policy"):
        monkeypatch.setattr(
            batch.source, name, lambda *a, **kw: pytest.fail("unexpected model access")
        )
    monkeypatch.setattr(batch.source, "configure_threads", lambda: None)
    monkeypatch.setattr(
        batch.source.PlantCyclicMarketEnvV2Beta,
        "reset",
        lambda *a, **kw: pytest.fail("production reset"),
    )


@pytest.fixture
def clock(monkeypatch):
    value = SimpleNamespace(
        utc=datetime(2026, 9, 14, 22, tzinfo=timezone.utc), ns=2_000_000_000_000
    )
    monkeypatch.setattr(batch, "_utc", lambda: value.utc)
    monkeypatch.setattr(batch.time, "monotonic_ns", lambda: value.ns)
    return value


@pytest.fixture
def prepared(tmp_path, monkeypatch, clock):
    from plant_backend.rl.v2.training import capacity_benchmark

    root, reference, initial = (
        tmp_path / "batch009",
        tmp_path / "batch008",
        tmp_path / "batch007",
    )
    reference.mkdir()
    initial.mkdir()
    for name, value in (
        ("ACTIVE_BASE", tmp_path),
        ("ROOT", root),
        ("REFERENCE_ROOT", reference),
        ("INITIAL_ROOT", initial),
        ("_INTERNAL_MOUNT", tmp_path),
    ):
        monkeypatch.setattr(batch, name, value)
    runtime = SimpleNamespace(git_head="a" * 40, to_dict=lambda: {"fixture": True})
    monkeypatch.setattr(batch, "capture", lambda **kw: runtime)
    monkeypatch.setattr(batch, "_current_tree", lambda: "b" * 40)
    monkeypatch.setattr(batch, "verify_unchanged", lambda *a: None)
    monkeypatch.setattr(
        batch, "BetaTrainingRuntimeProvenanceV1", SimpleNamespace(from_dict=lambda x: x)
    )
    monkeypatch.setattr(
        capacity_benchmark, "_host_identity", lambda: {"fixture": "host"}
    )
    monkeypatch.setattr(
        batch.shutil, "disk_usage", lambda p: SimpleNamespace(free=20 * 1024**3)
    )

    def diskutil(command, **kwargs):
        assert command == ["/usr/sbin/diskutil", "info", "-plist", str(tmp_path)]
        return plistlib.dumps(
            {
                "VolumeUUID": batch._INTERNAL_VOLUME_UUID,
                "MountPoint": str(tmp_path),
                "FilesystemType": "apfs",
                "Internal": True,
                "Removable": False,
                "Ejectable": False,
            }
        )

    monkeypatch.setattr(batch.subprocess, "check_output", diskutil)
    monkeypatch.setattr(
        batch.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            stdout=b"Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t99%; finishing charge; 0:00 remaining present: true\n",
            stderr=b"",
            returncode=0,
        ),
    )
    records, bindings = [], {}
    for key, (name, _, _) in batch.INITIAL_BINDINGS.items():
        path = initial / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"opaque initializer {key}".encode())
        record = batch.file_record(path)
        bindings[key] = (name, record["bytes"], record["sha256"])
        records.append(record)
    monkeypatch.setattr(batch, "INITIAL_BINDINGS", bindings)
    origin = {"root": str(reference)}
    for key in ("analysis", "audit"):
        path = tmp_path / f"old-{key}.json"
        path.write_bytes(canonical_bytes({"fixture": key}))
        origin[key] = batch.file_record(path)
        records.append(origin[key])
        monkeypatch.setattr(
            batch, f"REFERENCE_{key.upper()}_SHA256", origin[key]["sha256"]
        )
    historical = reference / "evaluations" / "saved.gz"
    historical.parent.mkdir()
    historical.write_bytes(b"opaque historical episode")
    records.append(batch.file_record(historical))
    packet = {
        "schema": batch.PACKET_SCHEMA,
        "verified": True,
        "execution_source": {"git_head": "a" * 40, "git_tree": "b" * 40},
        "source_batch": origin,
        "development_seeds": list(batch.DEV_SEEDS),
        "actors": list(batch.REFERENCE_ACTORS),
        "scientific_compatibility": {"verified": True},
        "initial_reference": batch.required_initial_artifacts(),
        "artifact_records": records,
        "simulator_resets_executed": False,
        "policy_inference_or_learning_executed": False,
    }
    packet_path = tmp_path / "reference.json"
    packet_path.write_bytes(canonical_bytes(packet))
    gate = tmp_path / "gate.json"
    storage = batch.capture_storage_identity(gate)
    gate.write_bytes(
        canonical_bytes(
            {
                "passed": True,
                "source_commit": "a" * 40,
                "source_tree": "b" * 40,
                "source_clean_and_unchanged": True,
                "storage": storage,
                "power_policy": batch.power_policy(),
                "deadline_policy": batch.deadline_policy(),
                "reference_admission": batch.file_record(packet_path),
            }
        )
    )
    initialization = {
        "verified": True,
        "arms": {arm.name: {"all_tensors": {"fixture": True}} for arm in batch.ARMS},
    }
    return SimpleNamespace(
        root=root,
        initial=initial,
        reference=reference,
        packet=packet,
        packet_path=packet_path,
        gate=gate,
        clock=clock,
        initialization=initialization,
        historical=historical,
    )


def receipt(root, stage, name, extra, exit_code=0, timeout=None):
    timeout = timeout or (
        batch.TRAIN_TIMEOUT if stage == "_train" else batch.EVAL_TIMEOUT
    )
    value = {
        "stage": stage,
        "name": name,
        "codes": [] if exit_code == 0 else ["child_nonzero_exit"],
        "closed_at": "2026-09-14T22:00:01+00:00",
        "process": {
            "command": [batch.sys.executable, "-m", batch.MODULE, stage, *extra],
            "timeout_seconds": timeout,
            "started": True,
            "exit_code": exit_code,
            "termination_signal": None,
            "wall_elapsed_ns": 1_000_000,
            "child_user_cpu_ns": 1000,
            "child_system_cpu_ns": 1000,
        },
    }
    publish_json(root / "receipts" / f"{name}.json", value)
    return value


def fake_graph(prepared, monkeypatch):
    monkeypatch.setattr(batch, "verify_initializers", lambda: prepared.initialization)
    monkeypatch.setattr(batch, "verify_initial_matching", lambda *a: {"matched": True})

    def wave():
        return {
            "receipts": [
                receipt(
                    prepared.root, "_train", f"train-{arm.name}", ["--cell", arm.name]
                )
                for arm in batch.ARMS
            ],
            "fatal": None,
        }

    monkeypatch.setattr(batch, "training_wave", wave)
    monkeypatch.setattr(
        batch,
        "closed_child",
        lambda stage, name, timeout, extra: receipt(prepared.root, stage, name, extra),
    )


def test_only_reward_gamma_differs_between_frozen_cells():
    configs = [batch.cell_config(arm).to_dict() for arm in batch.ARMS]
    assert configs[0]["ppo"]["gamma"] == 0.995
    assert configs[1]["ppo"]["gamma"] == 1.0
    configs[0]["ppo"]["gamma"] = 1.0
    assert configs[0] == configs[1]
    spec = batch.specification()
    assert all(arm.algorithm == batch.SCALAR for arm in batch.ARMS)
    assert (
        spec["total_timesteps_per_arm"],
        spec["rollout_count_per_arm"],
        spec["optimizer_steps_per_arm"],
        spec["train_seed_count_per_arm"],
    ) == (16384, 2, 640, 28)
    assert (spec["training_slots"], spec["evaluation_slots"], spec["replay_slots"]) == (
        2,
        16,
        16,
    )
    assert spec["training_artifact_schema"] == batch.source.SCHEMA
    assert spec["reference_actors"] == list(batch.source.ACTORS)
    assert batch.TRAIN_SEED_BASE == 9630000
    assert batch.DEV_SEEDS == tuple(range(9700300, 9700308))
    spec["custody_policy"]["identity_fields"].clear()
    assert batch.CUSTODY_POLICY["identity_fields"]
    with pytest.raises(ValueError):
        batch.cell_config("unknown")


@pytest.mark.parametrize(
    "when",
    [
        "2026-09-14T21:01:27+00:00",
        "2026-09-14T23:15:00+00:00",
        "2026-09-15T07:01:28+00:00",
    ],
)
def test_deadline_admission_requires_exact_launch_window(clock, when):
    clock.utc = datetime.fromisoformat(when)
    with pytest.raises(batch.TrainingIntegrityError):
        batch.capture_deadline_admission()


def test_deadline_uses_earliest_utc_and_monotonic_remaining(clock):
    record = batch.capture_deadline_admission()
    initial = batch.require_deadline(record)
    clock.ns += 600 * 10**9
    assert batch.require_deadline(record) == initial - 600
    clock.utc += timedelta(seconds=900)
    assert batch.require_deadline(record) == initial - 900
    clock.ns = record["production_deadline_monotonic_ns"] - 90 * 10**9
    with pytest.raises(batch.TrainingIntegrityError, match="exhausted"):
        batch.require_deadline(record)


@pytest.mark.parametrize(
    "failure",
    [
        "absent",
        "missing",
        "policy",
        "utc",
        "anchor",
        "monotonic_type",
        "future_admission",
        "regressed",
    ],
)
def test_deadline_rejects_missing_or_altered_anchors(clock, failure):
    record = batch.capture_deadline_admission()
    if failure == "absent":
        record = None
    elif failure == "missing":
        record.pop("production_deadline_monotonic_ns")
    elif failure == "policy":
        record["policy"]["production_deadline_at"] = batch.FINAL_DEADLINE_AT
    elif failure == "utc":
        record["admitted_at"] = "2026-09-14T22:00:00"
    elif failure == "anchor":
        record["production_deadline_monotonic_ns"] += 1
    elif failure == "monotonic_type":
        record["admitted_monotonic_ns"] = True
    elif failure == "future_admission":
        record["admitted_at"] = "2026-09-15T00:00:00+00:00"
    else:
        clock.ns -= 1
    with pytest.raises((batch.TrainingIntegrityError, ValueError)):
        batch.require_deadline(record)


def test_deadline_stage_cap_and_sidecar_are_exact(clock):
    record = batch.capture_deadline_admission()
    clock.utc = datetime(2026, 9, 15, 6, 25, tzinfo=timezone.utc)
    assert batch.remaining_timeout(600, record) == 210
    check = batch.deadline_check(record, stage="_evaluate", name="fixture", cap=600)
    assert check["effective_timeout_seconds"] == 210
    assert check["remaining_ns"] == 300 * 10**9
    assert batch.validate_deadline_check(check, record) == check
    check["remaining_ns"] += 1
    with pytest.raises(batch.TrainingIntegrityError):
        batch.validate_deadline_check(check, record)


def test_registered_trainer_retains_wave_timeout_after_bounded_startup(clock):
    record = batch.capture_deadline_admission()
    clock.utc = datetime(2026, 9, 15, 6, 10, tzinfo=timezone.utc)
    wave = batch.deadline_check(
        record, stage="_train_wave", name="training-wave", cap=batch.TRAIN_TIMEOUT
    )
    assert wave["effective_timeout_seconds"] == 1110
    clock.utc += timedelta(seconds=2)
    current = batch.deadline_check(
        record,
        stage="_train",
        name="train-fixture",
        cap=batch.TRAIN_TIMEOUT,
        requested=1110,
    )
    assert current["effective_timeout_seconds"] == 1110
    assert batch.validate_deadline_check(current, record) == current
    clock.utc += timedelta(seconds=120)
    with pytest.raises(batch.TrainingIntegrityError):
        batch.deadline_check(
            record,
            stage="_train",
            name="train-fixture",
            cap=batch.TRAIN_TIMEOUT,
            requested=1110,
        )


def test_reference_packet_admits_complete_fixed_provenance(prepared):
    assert batch.read_reference_admission(prepared.packet_path) == prepared.packet
    assert not prepared.root.exists()


@pytest.mark.parametrize(
    "failure", ["bytes", "length", "missing", "symlink", "hardlink"]
)
def test_reference_rejects_changed_initial_content(prepared, failure):
    path = Path(prepared.packet["initial_reference"]["initial_policy"]["path"])
    if failure == "bytes":
        path.write_bytes(b"x" * path.stat().st_size)
    elif failure == "length":
        path.write_bytes(path.read_bytes() + b"x")
    elif failure == "missing":
        path.unlink()
    elif failure == "symlink":
        other = path.with_suffix(".preserved")
        path.rename(other)
        path.symlink_to(other)
    else:
        path.with_suffix(".linked").hardlink_to(path)
    with pytest.raises((ValueError, OSError, batch.TrainingIntegrityError)):
        batch.read_reference_admission(prepared.packet_path)


@pytest.mark.parametrize(
    "failure",
    [
        "verified",
        "source",
        "panel",
        "actors",
        "science",
        "reset",
        "inference",
        "analysis",
        "audit",
        "initial",
        "missing",
        "duplicate",
    ],
)
def test_reference_rejects_changed_admission_contract(prepared, failure):
    value = prepared.packet
    if failure == "verified":
        value["verified"] = False
    elif failure == "source":
        value["execution_source"]["git_head"] = "f" * 40
    elif failure == "panel":
        value["development_seeds"].pop()
    elif failure == "actors":
        value["actors"].pop()
    elif failure == "science":
        value["scientific_compatibility"]["verified"] = False
    elif failure == "reset":
        value["simulator_resets_executed"] = True
    elif failure == "inference":
        value["policy_inference_or_learning_executed"] = True
    elif failure in ("analysis", "audit"):
        value["source_batch"][failure]["sha256"] = "f" * 64
    elif failure == "initial":
        value["initial_reference"]["initial_policy"]["sha256"] = "f" * 64
    elif failure == "missing":
        value["artifact_records"].pop(0)
    else:
        value["artifact_records"].append(value["artifact_records"][0])
    prepared.packet_path.write_bytes(canonical_bytes(value))
    with pytest.raises(ValueError):
        batch.read_reference_admission(prepared.packet_path)


def test_reference_checks_saved_episodes_once_without_runtime_rescans(
    prepared, monkeypatch
):
    original = batch.file_record
    seen = []

    def record(path):
        seen.append(path)
        return original(path)

    monkeypatch.setattr(batch, "file_record", record)
    batch.read_reference_admission(prepared.packet_path)
    assert prepared.historical in seen
    seen.clear()
    batch.require_reference(prepared.packet_path, full=False)
    assert prepared.historical not in seen
    assert Path(prepared.packet["initial_reference"]["initial_policy"]["path"]) in seen


def test_reference_packet_digest_binds_additional_full_reference_fields(prepared):
    before = batch.file_record(prepared.packet_path)
    prepared.packet["reference_outcomes"] = {"fixture": True}
    prepared.packet_path.write_bytes(canonical_bytes(prepared.packet))
    with pytest.raises(ValueError, match="packet changed"):
        batch.require_reference(prepared.packet_path, before)


def test_whole_fake_graph_is_two_training_and_32_new_stages(prepared, monkeypatch):
    original_root, original_arms = batch.source.ROOT, batch.source.ARMS
    fake_graph(prepared, monkeypatch)
    report = batch.run(prepared.gate, authorized=True)
    assert report["status"] == "closed" and report["failure"] is None
    assert len(report["receipts"]) == 34
    assert [r["stage"] for r in report["receipts"]] == ["_train"] * 2 + [
        "_evaluate",
        "_replay",
    ] * 16
    assert report["planned_slots"]["training"] == list(batch.ACTORS)
    assert (
        len(report["planned_slots"]["evaluation"])
        == len(report["planned_slots"]["replay"])
        == 16
    )
    assert batch.require_plan()["reference_admission"] == batch.file_record(
        prepared.packet_path
    )
    assert batch.source.ROOT == original_root and batch.source.ARMS == original_arms
    with pytest.raises(ValueError, match="no retry"):
        batch.run(prepared.gate, authorized=True)


def test_abort_keeps_last_already_published_fatal_receipt(prepared, monkeypatch):
    fake_graph(prepared, monkeypatch)

    def child(stage, name, timeout, extra):
        value = receipt(
            prepared.root, stage, name, extra, 86 if stage == "_replay" else 0
        )
        if stage == "_replay":
            raise batch.TrainingIntegrityError("post-action deadline")
        return value

    monkeypatch.setattr(batch, "closed_child", child)
    report = batch.run(prepared.gate, authorized=True)
    assert report["status"] == "aborted" and len(report["receipts"]) == 4
    assert report["receipts"][-1]["process"]["exit_code"] == 86


@pytest.mark.parametrize(
    "failure",
    ["unauthorized", "source", "deadline", "packet", "power", "space", "existing_root"],
)
def test_admission_refuses_before_root_or_simulation(prepared, monkeypatch, failure):
    if failure == "source":
        value = json.loads(prepared.gate.read_bytes())
        value["source_tree"] = "f" * 40
        prepared.gate.write_bytes(canonical_bytes(value))
    elif failure == "deadline":
        prepared.clock.utc = datetime.fromisoformat(batch.LATEST_LAUNCH_AT)
    elif failure == "packet":
        prepared.packet_path.write_bytes(b"changed")
    elif failure == "space":
        monkeypatch.setattr(
            batch.shutil, "disk_usage", lambda p: SimpleNamespace(free=0)
        )
    elif failure == "existing_root":
        prepared.root.mkdir()
    elif failure == "power":
        monkeypatch.setattr(
            batch, "verify_initializers", lambda: prepared.initialization
        )
        monkeypatch.setattr(
            batch.subprocess,
            "run",
            lambda *a, **kw: SimpleNamespace(
                stdout=b"Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t80%; discharging; 1:00 remaining present: true\n",
                stderr=b"",
                returncode=0,
            ),
        )
    with pytest.raises((ValueError, OSError, batch.TrainingIntegrityError)):
        batch.run(prepared.gate, authorized=failure != "unauthorized")
    assert prepared.root.exists() is (failure == "existing_root")


@pytest.mark.parametrize(
    "failure",
    ["deadline", "missing_deadline", "gate", "reference", "initial", "space", "power"],
)
def test_periodic_integrity_rejects_changed_dependencies(
    prepared, monkeypatch, failure
):
    fake_graph(prepared, monkeypatch)
    batch.run(prepared.gate, authorized=True)
    if failure == "deadline":
        prepared.clock.utc = datetime.fromisoformat(batch.PRODUCTION_DEADLINE_AT)
    elif failure == "missing_deadline":
        path = prepared.root / "plan.json"
        value = json.loads(path.read_bytes())
        value.pop("deadline_admission")
        path.write_bytes(canonical_bytes(value))
    elif failure == "gate":
        prepared.gate.write_bytes(prepared.gate.read_bytes() + b" ")
    elif failure == "reference":
        prepared.packet_path.write_bytes(prepared.packet_path.read_bytes() + b" ")
    elif failure == "initial":
        Path(
            prepared.packet["initial_reference"]["initial_policy"]["path"]
        ).write_bytes(b"changed")
    elif failure == "space":
        monkeypatch.setattr(
            batch.shutil, "disk_usage", lambda p: SimpleNamespace(free=0)
        )
    else:
        monkeypatch.setattr(
            batch.subprocess,
            "run",
            lambda *a, **kw: SimpleNamespace(
                stdout=b"Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t19%; charging; 1:00 remaining present: true\n",
                stderr=b"",
                returncode=0,
            ),
        )
    with pytest.raises((ValueError, batch.TrainingIntegrityError)):
        batch.require_plan()
    if failure == "power":
        assert len(list((prepared.root / "power-failures").iterdir())) == 1
        monkeypatch.setattr(
            batch.source,
            "capture_power_sample",
            lambda: pytest.fail("resampled after refusal"),
        )
        with pytest.raises(batch.source.PowerAdmissionError, match="previous_refusal"):
            batch.require_plan()


@pytest.mark.parametrize(
    "args",
    [
        ["resume"],
        ["_train"],
        ["_train", "--cell", "A-mpc-residual-scalar"],
        ["_evaluate", "--actor", "H", "--seed", "9700300"],
        ["_evaluate", "--actor", batch.ACTORS[0], "--seed", "123"],
        ["run", "--cell", batch.ACTORS[0]],
    ],
)
def test_cli_rejects_unplanned_routes(args):
    with pytest.raises(SystemExit) as error:
        batch.main(args)
    assert error.value.code == 2


@pytest.fixture
def fake_models(prepared, monkeypatch):
    import torch

    head = torch.nn.Linear(3, 4)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    log_std = torch.full((4,), math.log(0.2))
    weights = {
        "action_net.weight": head.weight.detach().clone(),
        "action_net.bias": head.bias.detach().clone(),
        "log_std": log_std.clone(),
        "value_net.weight": torch.ones((1, 3)),
    }
    closed = []

    def model(arm):
        policy = SimpleNamespace(
            action_net=head,
            log_std=log_std,
            state_dict=lambda: {k: v.clone() for k, v in weights.items()},
        )
        return SimpleNamespace(
            algorithm_id=batch.SCALAR,
            gamma=batch.GAMMAS[arm.name],
            gae_lambda=0.95,
            num_timesteps=batch.TOTAL_STEPS,
            policy=policy,
            research_identity={"fixture": "identity"},
            embedded_constraint_state={
                "payload": {
                    "ppo_configuration": {
                        "gamma": batch.GAMMAS[arm.name],
                        "gae_lambda": 0.95,
                    },
                    "num_timesteps": 16384,
                    "pending_rollout": False,
                    "aborted_rollout": None,
                    "unledgered_transition_count": 0,
                    "resume_continuation_receipt": None,
                    "ppo_epoch_update_count": 20,
                    "rollout_ledgers": [
                        {"transition_count": 8192, "optimizer_steps": 320}
                    ]
                    * 2,
                }
            },
        )

    def make(config, arm, *, seed_base):
        assert config.to_dict() == batch.cell_config(arm, prepared.root).to_dict()
        assert seed_base == 9630000
        return model(arm), None, SimpleNamespace(close=lambda: closed.append(arm.name))

    monkeypatch.setattr(batch.source, "make_learner", make)
    monkeypatch.setattr(
        batch.source,
        "read_initial_policy",
        lambda *a: {k: v.clone() for k, v in weights.items()},
    )
    monkeypatch.setattr(batch.source, "verify_model_spaces", lambda *a: None)
    return SimpleNamespace(
        weights=weights, model=model, head=head, log_std=log_std, closed=closed
    )


def test_fresh_initializers_match_all_tensors_and_inherited_actor_without_reset(
    prepared, fake_models
):
    result = batch.verify_initializers()
    assert result["verified"] is True and result["matched"] is True
    assert result["simulator_resets_executed"] is False
    assert result["policy_inference_or_learning_executed"] is False
    assert (
        result["arms"][batch.ACTORS[0]]["all_tensors"]
        == result["arms"][batch.ACTORS[1]]["all_tensors"]
    )
    assert [result["arms"][arm]["gamma"] for arm in batch.ACTORS] == [0.995, 1.0]
    assert fake_models.closed == list(batch.ACTORS)


@pytest.mark.parametrize("failure", ["tensor", "head", "std", "gamma", "algorithm"])
def test_initializer_rejects_actual_mismatch(prepared, fake_models, failure):
    import torch

    model = fake_models.model(batch.ARMS[0])
    expected = {k: v.clone() for k, v in fake_models.weights.items()}
    if failure == "tensor":
        expected["value_net.weight"].add_(1)
    elif failure == "head":
        with torch.no_grad():
            model.policy.action_net.weight.fill_(1)
    elif failure == "std":
        model.policy.log_std = torch.full((4,), math.log(0.3))
    elif failure == "gamma":
        model.gamma = 1.0
    else:
        model.algorithm_id = "wrong"
    with pytest.raises(batch.TrainingIntegrityError):
        batch._assert_initial_policy(model, batch.ARMS[0], expected)


@pytest.mark.parametrize(
    "failure",
    [
        "gamma",
        "embedded_gamma",
        "lambda",
        "timesteps",
        "epochs",
        "rollout",
        "optimizer",
        "pending",
        "resume",
    ],
)
def test_endpoint_checks_actual_discount_and_complete_fixed_budget(
    prepared, fake_models, failure
):
    arm = batch.ARMS[1]
    model = fake_models.model(arm)
    batch.verify_endpoint_model(model, arm)
    payload = model.embedded_constraint_state["payload"]
    if failure == "gamma":
        model.gamma = 0.995
    elif failure == "embedded_gamma":
        payload["ppo_configuration"]["gamma"] = 0.995
    elif failure == "lambda":
        model.gae_lambda = 1.0
    elif failure == "timesteps":
        model.num_timesteps = 8192
    elif failure == "epochs":
        payload["ppo_epoch_update_count"] = 10
    elif failure == "rollout":
        payload["rollout_ledgers"].pop()
    elif failure == "optimizer":
        payload["rollout_ledgers"][0]["optimizer_steps"] = 319
    elif failure == "pending":
        payload["pending_rollout"] = True
    else:
        payload["resume_continuation_receipt"] = {"resumed": True}
    with pytest.raises(batch.TrainingIntegrityError):
        batch.verify_endpoint_model(model, arm)


def test_training_wrapper_reuses_unchanged_parameterized_producer(
    prepared, fake_models, monkeypatch
):
    initialization = batch.verify_initializers()
    plan = {
        "deadline_admission": batch.capture_deadline_admission(),
        "initialization_admission": initialization,
    }
    monkeypatch.setattr(batch, "require_plan", lambda: plan)
    seen = []

    def train(directory, config, arm, *, seed_base):
        assert directory == prepared.root / "training" / arm.name
        assert (
            config.total_timesteps == 16384
            and config.ppo.gamma == batch.GAMMAS[arm.name]
        )
        assert seed_base == 9630000
        seen.append(arm.name)
        return {"schema": batch.source.SCHEMA, "fixture": True}

    monkeypatch.setattr(batch.source, "train_cell", train)
    monkeypatch.setattr(
        batch.source, "load_final", lambda path, config, arm: fake_models.model(arm)
    )
    for arm in batch.ARMS:
        assert batch.train_cell(arm)["schema"] == batch.source.SCHEMA
    assert seen == list(batch.ACTORS)


def test_post_training_matching_rechecks_each_published_initializer(
    prepared, fake_models
):
    admission = batch.verify_initializers()
    for arm in batch.ARMS:
        directory = prepared.root / "training" / arm.name
        directory.mkdir(parents=True)
        publish_json(directory / "initial-policy-record.json", {"fixture": arm.name})
    result = batch.verify_initial_matching(prepared.root, admission)
    assert result["matched"] is True and result["expected_arms"] == list(batch.ACTORS)
    admission["arms"][batch.ACTORS[1]]["all_tensors"] = {}
    with pytest.raises(batch.TrainingIntegrityError):
        batch.verify_initial_matching(prepared.root, admission)


@pytest.mark.parametrize("arm", batch.ARMS)
def test_evaluation_and_replay_keep_their_own_gamma_and_fresh_paths(
    prepared, fake_models, monkeypatch, arm
):
    from plant_backend.rl.v2.training import (
        research_mpc_residual_evaluation as residual,
    )

    seed, actor = batch.DEV_SEEDS[0], arm.name
    directory = prepared.root / "evaluations" / f"{seed}-{actor}"
    directory.mkdir(parents=True)
    (prepared.root / "receipts").mkdir()
    receipt(prepared.root, "_train", f"train-{actor}", ["--cell", actor])

    def load(path, config, observed_arm):
        assert path == prepared.root / "training" / actor
        assert config.ppo.gamma == batch.GAMMAS[actor] and observed_arm == arm
        return fake_models.model(arm)

    monkeypatch.setattr(batch.source, "load_final", load)

    def evaluate(config, **kwargs):
        assert config.to_dict() == batch.cell_config(arm, prepared.root).to_dict()
        assert kwargs["seed"] == seed and kwargs["policy_id"] == actor
        return {
            "seed": seed,
            "config": config.to_dict(),
            "summary": {"policy_id": actor, "actor": residual.ACTOR},
        }

    monkeypatch.setattr(residual, "evaluate_residual_episode", evaluate)
    batch.evaluate_cell(actor, seed)
    receipt(
        prepared.root,
        "_evaluate",
        f"evaluate-{seed}-{actor}",
        ["--actor", actor, "--seed", str(seed)],
    )

    def replay(config, episode):
        assert (
            canonical_bytes(episode["config"])
            == canonical_bytes(config.to_dict())
            == canonical_bytes(batch.cell_config(arm, prepared.root).to_dict())
        )
        return {"verified": True}

    monkeypatch.setattr(residual, "verify_residual_episode_replay", replay)
    batch.replay_cell(actor, seed)
    assert json.loads((directory / "replay.json").read_bytes())["verified"] is True


def test_child_timeout_sidecar_precedes_capture_and_fatal_receipt_is_retained(
    prepared, monkeypatch
):
    for name in ("receipts", "deadlines", "logs"):
        (prepared.root / name).mkdir(parents=True, exist_ok=True)
    plan = {"deadline_admission": batch.capture_deadline_admission()}
    monkeypatch.setattr(batch, "require_plan", lambda: plan)
    prepared.clock.utc = datetime(2026, 9, 15, 6, 25, tzinfo=timezone.utc)
    name = "replay-fixture"

    def capture(command, **kwargs):
        timing = json.loads((prepared.root / "deadlines" / f"{name}.json").read_bytes())
        assert kwargs["timeout_seconds"] == timing["effective_timeout_seconds"] == 210
        assert kwargs["log_limit_bytes"] == 64 * 1024**2
        assert set(timing) == {
            "schema",
            "stage",
            "name",
            "deadline_admission",
            "checked_at",
            "checked_monotonic_ns",
            "remaining_ns",
            "stage_cap_seconds",
            "requested_timeout_seconds",
            "effective_timeout_seconds",
            "binding",
        }
        return {"command": command, "exit_code": 86, "timeout_seconds": 210}, {
            "child_nonzero_exit"
        }

    monkeypatch.setattr(batch, "capture_process", capture)
    with pytest.raises(batch.TrainingIntegrityError):
        batch.closed_child(
            "_replay",
            name,
            600,
            ["--actor", batch.ACTORS[0], "--seed", str(batch.DEV_SEEDS[0])],
        )
    saved = json.loads((prepared.root / "receipts" / f"{name}.json").read_bytes())
    assert saved["process"]["exit_code"] == 86
    assert set(saved) == {"stage", "name", "process", "codes", "closed_at"}


def test_wave_keeps_two_isolated_attempts_and_deadline_callback(prepared, monkeypatch):
    from plant_backend.rl.v2.training import (
        research_parallel_supervision as supervision,
    )

    (prepared.root / "deadlines").mkdir(parents=True)
    plan = {"deadline_admission": batch.capture_deadline_admission()}

    def check():
        batch.require_deadline(plan["deadline_admission"])
        return plan

    monkeypatch.setattr(batch, "require_plan", check)

    def wave(attempts, **kwargs):
        assert len(attempts) == 2
        assert {a.timeout_seconds for a in attempts} == {12600}
        assert [a.name for a in attempts] == [f"train-{name}" for name in batch.ACTORS]
        assert all(a.command[2] == batch.MODULE for a in attempts)
        assert kwargs["worker_module"] == batch.MODULE
        assert kwargs["check_integrity"] is check
        assert kwargs["integrity_check_interval_seconds"] == 30
        timing = json.loads(
            (prepared.root / "deadlines" / "training-wave.json").read_bytes()
        )
        assert timing["effective_timeout_seconds"] == 12600
        prepared.clock.utc = datetime.fromisoformat(batch.PRODUCTION_DEADLINE_AT)
        with pytest.raises(batch.TrainingIntegrityError):
            kwargs["check_integrity"]()
        return {"fatal": {"fixture": "deadline"}, "receipts": []}

    monkeypatch.setattr(supervision, "run_training_wave", wave)
    assert batch.training_wave()["fatal"] == {"fixture": "deadline"}


def test_training_capture_uses_exact_registered_wave_timeout(prepared, monkeypatch):
    for name in ("receipts", "deadlines", "logs"):
        (prepared.root / name).mkdir(parents=True, exist_ok=True)
    plan = {"deadline_admission": batch.capture_deadline_admission()}
    monkeypatch.setattr(batch, "require_plan", lambda: plan)
    prepared.clock.utc = datetime(2026, 9, 15, 6, 10, tzinfo=timezone.utc)
    timing = batch.deadline_check(
        plan["deadline_admission"],
        stage="_train_wave",
        name="training-wave",
        cap=batch.TRAIN_TIMEOUT,
    )
    publish_json(prepared.root / "deadlines" / "training-wave.json", timing)
    prepared.clock.utc += timedelta(seconds=2)

    def capture(command, **kwargs):
        assert kwargs["timeout_seconds"] == 1110
        return {"command": command, "exit_code": 0, "timeout_seconds": 1110}, set()

    monkeypatch.setattr(batch, "capture_process", capture)
    saved = batch.closed_child(
        "_train", "train-fixture", 1110, ["--cell", batch.ACTORS[0]]
    )
    assert saved["process"]["timeout_seconds"] == 1110
    with pytest.raises(batch.TrainingIntegrityError, match="registered wave"):
        batch.closed_child("_train", "train-other", 1109, ["--cell", batch.ACTORS[1]])


def test_child_checks_before_and_after_action_and_distinguishes_integrity(monkeypatch):
    calls = []
    monkeypatch.setattr(batch, "require_plan", lambda: calls.append("check"))
    assert batch.execute_child(lambda: calls.append("action"), artifact_phase=True) == 0
    assert calls == ["check", "action", "check"]

    def fail():
        raise ValueError("fixture producer failure")

    with pytest.raises(ValueError):
        batch.execute_child(fail, artifact_phase=False)
    assert batch.execute_child(fail, artifact_phase=True) == 86

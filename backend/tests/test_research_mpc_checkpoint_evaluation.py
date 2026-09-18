"""Evaluation-only custody fixtures; no production resets or real policy contents."""

from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import plistlib
from types import SimpleNamespace

import pytest

from plant_backend.rl.v2.training import research_mpc_checkpoint_evaluation as batch
from plant_backend.rl.v2.training.custody_io import publish_json
from plant_backend.rl.v2.training.custody_json import canonical_bytes


@pytest.fixture(autouse=True)
def no_learning_or_production_model_access(monkeypatch):
    for name in (
        "train_cell",
        "make_learner",
        "training_wave",
        "load_final",
        "load_initial",
        "verify_initial_matching",
    ):
        monkeypatch.setattr(
            batch.source,
            name,
            lambda *a, **kw: pytest.fail("unexpected training/model access"),
        )
    monkeypatch.setattr(batch.source, "configure_threads", lambda: None)
    monkeypatch.setattr(
        batch.source.PlantCyclicMarketEnvV2Beta,
        "reset",
        lambda *a, **kw: pytest.fail("production reset"),
    )


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from plant_backend.rl.v2.training import capacity_benchmark

    root, origin = tmp_path / "batch008", tmp_path / "batch007"
    origin.mkdir()
    monkeypatch.setattr(batch, "ACTIVE_BASE", tmp_path)
    monkeypatch.setattr(batch, "ROOT", root)
    monkeypatch.setattr(batch, "SOURCE_ROOT", origin)
    monkeypatch.setattr(batch, "_INTERNAL_MOUNT", tmp_path)
    monkeypatch.setattr(batch, "_current_tree", lambda: "b" * 40)
    runtime = SimpleNamespace(git_head="a" * 40, to_dict=lambda: {"fixture": True})
    monkeypatch.setattr(batch, "capture", lambda **kw: runtime)
    monkeypatch.setattr(batch, "verify_unchanged", lambda *a: None)
    monkeypatch.setattr(
        batch,
        "BetaTrainingRuntimeProvenanceV1",
        SimpleNamespace(from_dict=lambda value: value),
    )
    monkeypatch.setattr(
        capacity_benchmark, "_host_identity", lambda: {"fixture": "host"}
    )
    monkeypatch.setattr(
        batch.shutil, "disk_usage", lambda path: SimpleNamespace(free=20 * 1024**3)
    )

    def diskutil(command, **kwargs):
        assert command == ["/usr/sbin/diskutil", "info", "-plist", str(tmp_path)]
        assert kwargs == {"stderr": batch.subprocess.PIPE, "timeout": 10}
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

    def pmset(command, **kwargs):
        assert command == ["/usr/bin/pmset", "-g", "batt"]
        return SimpleNamespace(
            stdout=b"Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t80%; AC attached; not charging present: true\n",
            stderr=b"",
            returncode=0,
        )

    monkeypatch.setattr(batch.subprocess, "run", pmset)
    bindings = {}
    records = []
    for name in batch.SOURCE_BINDINGS:
        path = origin / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = f"opaque fixture {name}".encode()
        path.write_bytes(raw)
        record = batch.file_record(path)
        records.append(record)
        bindings[name] = (len(raw), record["sha256"])
    monkeypatch.setattr(batch, "SOURCE_BINDINGS", bindings)
    source_batch = {
        "root": str(origin),
        "git_head": batch.SOURCE_HEAD,
        "git_tree": batch.SOURCE_TREE,
        "status": "aborted",
        "normal_batch_admitted": False,
    }
    for key, path in {
        "plan": origin / "plan.json",
        "report": origin / "report.json",
        "training_wave": origin / "training-wave.json",
        "gate": tmp_path / "old-gate.json",
        "tools_manifest": tmp_path / "old-tools.json",
        "forensic_inventory": tmp_path / "old-forensic.json",
    }.items():
        path.write_bytes(canonical_bytes({"fixture": key}))
        source_batch[key] = batch.file_record(path)
        records.append(source_batch[key])
    required = batch.required_source_artifacts()
    endpoints = {}
    for arm in batch.ARMS:
        endpoints[arm.name] = {
            "verified": True,
            "source_root": str(origin),
            "total_timesteps": 65536,
            "rollout_count": 8,
            "optimizer_steps": 2560,
            "allocator_reset_count": 100,
            "model_attestation_verified": True,
        }
        for key, filename in (
            ("initial_policy", "initial-policy.pt"),
            ("initial_policy_record", "initial-policy-record.json"),
            ("final_policy", "final-policy.zip"),
            ("result", "result.json"),
        ):
            endpoints[arm.name][key] = required[f"training/{arm.name}/{filename}"]
    packet = {
        "schema": batch.PACKET_SCHEMA,
        "verified": True,
        "evaluation_source": {"git_head": runtime.git_head, "git_tree": "b" * 40},
        "source_batch": source_batch,
        "custody_policy": deepcopy(batch.CUSTODY_POLICY),
        "supervision": {"verified": True, "prospective_contract": True},
        "initial_tensor_comparison": {"performed": True, "passed": True},
        "training_endpoints": endpoints,
        "artifact_records": records,
        "normal_batch_007_admitted": False,
        "evaluation_artifacts_read": False,
        "policy_inference_or_learning_executed": False,
        "simulator_resets_executed": False,
    }
    packet_path, gate = tmp_path / "import.json", tmp_path / "gate.json"
    for key, filename in (
        ("import_tools_manifest", "import-tools-manifest.json"),
        ("verifier_source", "import-verifier.py"),
    ):
        path = tmp_path / filename
        path.write_bytes(f"fixture {key}".encode())
        packet[key] = batch.file_record(path)
    publish_json(packet_path, packet)
    publish_json(
        gate,
        {
            "passed": True,
            "source_commit": runtime.git_head,
            "storage": batch.capture_storage_identity(gate),
            "power_policy": batch.power_policy(),
            "checkpoint_import": batch.file_record(packet_path),
            "import_tools_manifest": packet["import_tools_manifest"],
        },
    )
    return SimpleNamespace(
        root=root,
        origin=origin,
        packet=packet,
        packet_path=packet_path,
        gate=gate,
        runtime=runtime,
    )


def write_packet(prepared):
    prepared.packet_path.write_bytes(canonical_bytes(prepared.packet))


def receipt(root, stage, name, extra, code=0):
    value = {
        "stage": stage,
        "name": name,
        "closed_at": "fixture-close",
        "codes": ["child_nonzero_exit"] if code else [],
        "process": {
            "command": [batch.sys.executable, "-m", batch.MODULE, stage, *extra],
            "exit_code": code,
            "started": True,
        },
    }
    publish_json(root / "receipts" / f"{name}.json", value)
    return value


def test_fixed_scientific_config_and_no_training_budget():
    spec = batch.specification(batch.ROOT)
    assert batch.ROOT.name == "adaptive-rl-batch-008"
    assert batch.SOURCE_ROOT.name == "adaptive-rl-batch-007"
    assert (
        batch.source_config().to_dict()
        == batch.source.template(batch.SOURCE_ROOT).to_dict()
    )
    assert spec["environment_template"]["output_dir"] == str(
        batch.SOURCE_ROOT / "artifacts"
    )
    assert spec["training_slots"] == spec["optimizer_updates"] == 0
    assert spec["evaluation_slots"] == spec["replay_slots"] == 56
    assert batch.DEV_SEEDS == tuple(range(9700300, 9700308))
    assert len(batch.ACTORS) == 7
    assert (
        batch.EVAL_TIMEOUT,
        batch.LOG_LIMIT,
        batch.ARCHIVE_LIMIT,
        batch.MINIMUM_FREE,
    ) == (600, 64 * 1024**2, 256 * 1024**2, 10 * 1024**3)
    assert batch.power_policy()["sampling"] == [
        "before_root_claim",
        "stage_entry_and_exit",
    ]
    assert "training_check_interval_seconds" not in batch.power_policy()
    spec["custody_policy"]["identity_fields"].clear()
    assert batch.CUSTODY_POLICY["identity_fields"]


@pytest.mark.parametrize(
    "args",
    [
        ["_train"],
        ["run", "--cell", "A-mpc-residual-scalar"],
        ["resume"],
        ["_preflight"],
    ],
)
def test_cli_rejects_any_training_or_resume_route(args):
    with pytest.raises(SystemExit) as error:
        batch.main(args)
    assert error.value.code == 2


def test_packet_rechecks_exact_bound_dependency_content(prepared):
    assert batch.read_import_admission(prepared.packet_path) == prepared.packet
    value = batch.file_record(prepared.packet_path)
    assert batch.require_packet(prepared.packet_path, value) == prepared.packet
    assert not prepared.root.exists()


@pytest.mark.parametrize(
    "failure", ["bytes", "length", "missing", "symlink", "hardlink"]
)
def test_packet_rejects_source_mutation_without_model_access(prepared, failure):
    path = prepared.origin / "training/A-mpc-residual-scalar/final-policy.zip"
    if failure == "bytes":
        path.write_bytes(b"z" * path.stat().st_size)
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
    with pytest.raises((ValueError, OSError)):
        batch.require_packet(prepared.packet_path)


@pytest.mark.parametrize(
    "failure",
    [
        "verified",
        "source",
        "tree",
        "custody",
        "inference",
        "reset",
        "old_admission",
        "initial_matching",
        "endpoint",
        "missing_dependency",
        "duplicate",
        "missing_source_record",
    ],
)
def test_packet_rejects_changed_admission_contract(prepared, failure):
    value = prepared.packet
    if failure == "verified":
        value["verified"] = False
    elif failure == "source":
        value["source_batch"]["git_head"] = "c" * 40
    elif failure == "tree":
        value["evaluation_source"]["git_tree"] = "c" * 40
    elif failure == "custody":
        value["custody_policy"]["identity_fields"].append("allocated_bytes")
    elif failure == "inference":
        value["policy_inference_or_learning_executed"] = True
    elif failure == "reset":
        value["simulator_resets_executed"] = True
    elif failure == "old_admission":
        value["normal_batch_007_admitted"] = True
    elif failure == "initial_matching":
        value["initial_tensor_comparison"]["passed"] = False
    elif failure == "endpoint":
        value["training_endpoints"][batch.ARMS[0].name]["total_timesteps"] = 24576
    elif failure == "missing_dependency":
        value["artifact_records"].pop(0)
    elif failure == "duplicate":
        value["artifact_records"].append(value["artifact_records"][0])
    else:
        value["source_batch"].pop("training_wave")
    write_packet(prepared)
    with pytest.raises(ValueError):
        batch.require_packet(prepared.packet_path)


def test_packet_never_reads_previous_evaluation_artifact(prepared, monkeypatch):
    path = prepared.origin / "evaluations" / "forbidden.gz"
    prepared.packet["artifact_records"].append(
        {"path": str(path), "bytes": 0, "sha256": "0" * 64}
    )
    write_packet(prepared)
    original = batch.file_record

    def guarded(item):
        assert item != path, "previous evaluation artifact read"
        return original(item)

    monkeypatch.setattr(batch, "file_record", guarded)
    with pytest.raises(ValueError, match="forbidden"):
        batch.require_packet(prepared.packet_path)


def test_packet_hash_join_rejects_edit_even_if_individual_flags_pass(prepared):
    original = batch.file_record(prepared.packet_path)
    prepared.packet["extra"] = "changed"
    write_packet(prepared)
    with pytest.raises(ValueError, match="packet changed"):
        batch.require_packet(prepared.packet_path, original)


def test_whole_fake_graph_is_112_new_slots_and_no_training(prepared, monkeypatch):
    original_root, original_module = batch.source.ROOT, batch.source.MODULE
    before = {
        path: path.read_bytes() for path in prepared.origin.rglob("*") if path.is_file()
    }
    monkeypatch.setattr(batch, "verify_imported_models", lambda: None)
    monkeypatch.setattr(
        batch,
        "closed_child",
        lambda stage, name, timeout, extra: receipt(prepared.root, stage, name, extra),
    )
    result = batch.run(prepared.gate, authorized=True)
    assert result["status"] == "closed" and result["failure"] is None
    assert len(result["receipts"]) == 112
    assert [item["stage"] for item in result["receipts"]] == [
        "_evaluate",
        "_replay",
    ] * 56
    assert result["planned_slots"]["training"] == []
    assert (
        not (prepared.root / "training").exists()
        and not (prepared.root / "supervisors").exists()
    )
    assert batch.source.ROOT == original_root and batch.source.MODULE == original_module
    assert before == {
        path: path.read_bytes() for path in prepared.origin.rglob("*") if path.is_file()
    }
    assert batch.require_plan()["checkpoint_import"] == batch.file_record(
        prepared.packet_path
    )
    with pytest.raises(ValueError, match="no retry"):
        batch.run(prepared.gate, authorized=True)


def test_last_published_failed_child_receipt_is_in_abort_report(prepared, monkeypatch):
    monkeypatch.setattr(batch, "verify_imported_models", lambda: None)
    calls = []

    def child(stage, name, timeout, extra):
        calls.append(stage)
        value = receipt(
            prepared.root, stage, name, extra, 86 if stage == "_replay" else 0
        )
        if stage == "_replay":
            raise batch.TrainingIntegrityError("fixture post-action refusal")
        return value

    monkeypatch.setattr(batch, "closed_child", child)
    result = batch.run(prepared.gate, authorized=True)
    assert result["status"] == "aborted" and calls == ["_evaluate", "_replay"]
    assert (
        len(result["receipts"]) == 2
        and result["receipts"][-1]["process"]["exit_code"] == 86
    )
    assert (
        len(result["planned_slots"]["evaluation"])
        == len(result["planned_slots"]["replay"])
        == 56
    )


@pytest.mark.parametrize(
    "failure", ["unauthorized", "source", "packet", "space", "power", "existing_root"]
)
def test_admission_failure_precedes_any_model_or_root_claim(
    prepared, monkeypatch, failure
):
    if failure == "source":
        gate = json.loads(prepared.gate.read_bytes())
        gate["source_commit"] = "c" * 40
        prepared.gate.write_bytes(canonical_bytes(gate))
    elif failure == "packet":
        prepared.packet_path.write_bytes(b"changed")
    elif failure == "space":
        monkeypatch.setattr(
            batch.shutil, "disk_usage", lambda p: SimpleNamespace(free=0)
        )
    elif failure == "power":
        monkeypatch.setattr(batch, "verify_imported_models", lambda: None)
        monkeypatch.setattr(
            batch.subprocess,
            "run",
            lambda *a, **kw: SimpleNamespace(
                stdout=b"Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t80%; discharging; 1:00 remaining present: true\n",
                stderr=b"",
                returncode=0,
            ),
        )
    elif failure == "existing_root":
        prepared.root.mkdir()
    with pytest.raises((ValueError, OSError)):
        batch.run(prepared.gate, authorized=failure != "unauthorized")
    assert prepared.root.exists() is (failure == "existing_root")
    if failure == "power":
        records = list(prepared.gate.parent.glob("batch008-power-admission-*.json"))
        assert len(records) == 1


@pytest.mark.parametrize(
    "actor",
    [
        "H",
        "OperatingMPC",
        "M24",
        "E-beta",
        "initial-mpc-residual",
        "A-mpc-residual-scalar",
        "B-mpc-residual-pi",
    ],
)
def test_evaluation_reuses_source_config_and_original_load_paths(
    prepared, monkeypatch, actor
):
    from plant_backend.rl.v2.training import (
        research_evaluation,
        research_mpc_residual_evaluation,
    )

    seed = batch.DEV_SEEDS[0]
    directory = prepared.root / "evaluations" / f"{seed}-{actor}"
    directory.mkdir(parents=True)
    seen = []

    def load(path, config, arm):
        assert path == prepared.origin / "training" / arm.name
        assert config.to_dict() == batch.source_config().to_dict()
        seen.append(arm.name)
        return "fixture-policy"

    monkeypatch.setattr(batch.source, "load_initial", load)
    monkeypatch.setattr(batch.source, "load_final", load)

    def evaluate(config, **kwargs):
        assert config.to_dict() == batch.source_config().to_dict()
        assert kwargs["policy_id"] == actor and kwargs["seed"] == seed
        assert kwargs["policy"] == (
            None if actor in batch.BASELINES else "fixture-policy"
        )
        return {
            "seed": seed,
            "summary": {
                "policy_id": actor,
                "actor": (
                    actor
                    if actor in batch.BASELINES
                    else research_mpc_residual_evaluation.ACTOR
                ),
            },
        }

    monkeypatch.setattr(research_evaluation, "evaluate_episode", evaluate)
    monkeypatch.setattr(
        research_mpc_residual_evaluation, "evaluate_residual_episode", evaluate
    )
    batch.evaluate_cell(actor, seed)
    assert len(seen) == (0 if actor in batch.BASELINES else 1)
    assert (directory / "summary.json").exists() and (
        directory / "episode.json.gz"
    ).exists()
    (prepared.root / "receipts").mkdir()
    receipt(
        prepared.root,
        "_evaluate",
        f"evaluate-{seed}-{actor}",
        ["--actor", actor, "--seed", str(seed)],
    )

    def replay(config, episode):
        assert config.to_dict() == batch.source_config().to_dict()
        assert episode["summary"]["policy_id"] == actor
        return {"fixture_replayed": True}

    monkeypatch.setattr(research_evaluation, "verify_episode_replay", replay)
    monkeypatch.setattr(
        research_mpc_residual_evaluation, "verify_residual_episode_replay", replay
    )
    batch.replay_cell(actor, seed)
    assert json.loads((directory / "replay.json").read_bytes()) == {
        "fixture_replayed": True
    }


def test_models_preflight_uses_exact_original_endpoints_without_inference(
    prepared, monkeypatch
):
    calls = []

    def load(path, config, arm):
        assert path == prepared.origin / "training" / arm.name
        assert config.to_dict() == batch.source_config().to_dict()
        calls.append(arm.name)
        return object()

    monkeypatch.setattr(batch.source, "load_initial", load)
    monkeypatch.setattr(batch.source, "load_final", load)
    monkeypatch.setattr(
        batch.source,
        "verify_initial_matching",
        lambda root, config: {"matched": root == prepared.origin},
    )
    batch.verify_imported_models()
    assert calls == [batch.ARMS[0].name, batch.ARMS[1].name, batch.ARMS[0].name]


def test_child_integrity_failure_retains_receipt_and_blocks_training(
    prepared, monkeypatch
):
    prepared.root.mkdir()
    (prepared.root / "receipts").mkdir()
    monkeypatch.setattr(batch, "require_plan", lambda: {})

    def capture(command, **kwargs):
        assert command == [
            batch.sys.executable,
            "-m",
            batch.MODULE,
            "_replay",
            "--actor",
            "H",
            "--seed",
            str(batch.DEV_SEEDS[0]),
        ]
        assert (
            kwargs["timeout_seconds"] == 600
            and kwargs["log_limit_bytes"] == 64 * 1024**2
        )
        return {"exit_code": 86, "command": command}, {"child_nonzero_exit"}

    monkeypatch.setattr(batch, "capture_process", capture)
    with pytest.raises(ValueError, match="evaluation-only"):
        batch.closed_child("_train", "forbidden", 600, [])
    with pytest.raises(batch.TrainingIntegrityError):
        batch.closed_child(
            "_replay",
            "fixture",
            600,
            ["--actor", "H", "--seed", str(batch.DEV_SEEDS[0])],
        )
    assert (
        json.loads((prepared.root / "receipts" / "fixture.json").read_bytes())[
            "process"
        ]["exit_code"]
        == 86
    )


def test_execute_child_checks_before_and_after_action(monkeypatch):
    calls = []
    monkeypatch.setattr(batch, "require_plan", lambda: calls.append("check"))
    assert batch.execute_child(lambda: calls.append("action")) == 0
    assert calls == ["check", "action", "check"]
    monkeypatch.setattr(
        batch, "require_plan", lambda: (_ for _ in ()).throw(ValueError("changed"))
    )
    assert (
        batch.execute_child(lambda: pytest.fail("action after failed admission")) == 86
    )


def test_packet_content_identity_excludes_changed_physical_allocation(
    prepared, monkeypatch
):
    original = Path.lstat
    target = prepared.origin / "training/A-mpc-residual-scalar/final-policy.zip"
    calls = []

    def lstat(path, *args, **kwargs):
        observed = original(path, *args, **kwargs)
        if path != target:
            return observed
        calls.append(path)
        fields = {
            name: getattr(observed, name)
            for name in dir(observed)
            if name.startswith("st_")
        }
        fields["st_blocks"] = observed.st_blocks + len(calls) * 16
        return SimpleNamespace(**fields)

    monkeypatch.setattr(Path, "lstat", lstat)
    assert batch.require_packet(prepared.packet_path) == prepared.packet
    assert len(calls) >= 2


@pytest.mark.parametrize(
    "failure", ["gate", "checkpoint", "packet", "runtime", "space", "power"]
)
def test_periodic_plan_rejects_changed_admission_dependencies(
    prepared, monkeypatch, failure
):
    monkeypatch.setattr(batch, "verify_imported_models", lambda: None)
    monkeypatch.setattr(
        batch,
        "closed_child",
        lambda stage, name, timeout, extra: receipt(prepared.root, stage, name, extra),
    )
    batch.run(prepared.gate, authorized=True)
    assert batch.require_plan()["checkpoint_import"] == batch.file_record(
        prepared.packet_path
    )
    if failure == "gate":
        prepared.gate.write_bytes(prepared.gate.read_bytes() + b" ")
    elif failure == "checkpoint":
        (
            prepared.origin / "training/A-mpc-residual-scalar/final-policy.zip"
        ).write_bytes(b"changed")
    elif failure == "packet":
        prepared.packet_path.write_bytes(prepared.packet_path.read_bytes() + b" ")
    elif failure == "runtime":
        monkeypatch.setattr(
            batch,
            "verify_unchanged",
            lambda *a: (_ for _ in ()).throw(ValueError("source changed")),
        )
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
    with pytest.raises(ValueError):
        batch.require_plan()
    if failure == "power":
        records = list((prepared.root / "power-failures").iterdir())
        assert (
            len(records) == 1
            and json.loads(records[0].read_bytes())["code"] == "power_reserve_low"
        )
        monkeypatch.setattr(
            batch.source,
            "capture_power_sample",
            lambda: pytest.fail("resampled after latched refusal"),
        )
        with pytest.raises(batch.source.PowerAdmissionError, match="previous_refusal"):
            batch.require_plan()


def test_periodic_accepted_charge_change_does_not_mutate_identity(
    prepared, monkeypatch
):
    monkeypatch.setattr(batch, "verify_imported_models", lambda: None)
    monkeypatch.setattr(
        batch,
        "closed_child",
        lambda stage, name, timeout, extra: receipt(prepared.root, stage, name, extra),
    )
    batch.run(prepared.gate, authorized=True)
    plan = batch.require_plan()
    for percent in (20, 100):
        raw = f"Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t{percent}%; charged; 0:00 remaining present: true\n".encode()
        monkeypatch.setattr(
            batch.subprocess,
            "run",
            lambda *a, **kw: SimpleNamespace(stdout=raw, stderr=b"", returncode=0),
        )
        assert batch.require_plan() == plan
    assert not (prepared.root / "power-failures").exists()


def test_source_path_mutation_during_read_fails_closed(prepared, monkeypatch):
    path = prepared.origin / "training/A-mpc-residual-scalar/final-policy.zip"
    original = Path.lstat
    count = 0

    def lstat(item, *args, **kwargs):
        nonlocal count
        value = original(item, *args, **kwargs)
        if item == path:
            count += 1
            if count > 1:
                fields = {
                    name: getattr(value, name)
                    for name in dir(value)
                    if name.startswith("st_")
                }
                fields["st_ino"] += 1
                return SimpleNamespace(**fields)
        return value

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="changed"):
        batch.require_packet(prepared.packet_path)

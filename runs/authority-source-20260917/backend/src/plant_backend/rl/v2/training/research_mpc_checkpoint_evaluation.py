"""Batch 008: evaluate the exact preserved batch-007 checkpoints, without training."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import stat
import subprocess
import sys
from typing import Any, Callable, cast

from . import research_mpc_constraint_batch as source
from .custody_io import mkdir_exclusive, publish_bytes, publish_json, read_regular
from .custody_io import require_directory
from .custody_json import canonical_bytes
from .custody_process import capture_process
from .research_batch import file_record, now
from .research_effort_batch import TrainingIntegrityError, TrainingUnavailable
from .runtime_provenance import (
    BetaTrainingRuntimeProvenanceV1,
    capture,
    verify_unchanged,
)

MODULE = "plant_backend.rl.v2.training.research_mpc_checkpoint_evaluation"
SCHEMA = "plant-adaptive-rl-mpc-checkpoint-evaluation-batch-008"
PACKET_SCHEMA = "plant-mpc-constraint-008-checkpoint-reuse-admission-v1"
ACTIVE_BASE = Path("/Users/filippogombac/Developer/PlantAnAgentResearch")
ROOT = ACTIVE_BASE / "adaptive-rl-batch-008"
SOURCE_ROOT = ACTIVE_BASE / "adaptive-rl-batch-007"
SOURCE_HEAD = "61b97851637a0f5a16e56c479e1a436cd30cf68d"
SOURCE_TREE = "311f8d3b5d048643e4a38b638c73ef95363b4d27"
STORAGE_SCHEMA = source.STORAGE_SCHEMA
_INTERNAL_VOLUME_UUID = source._INTERNAL_VOLUME_UUID
_INTERNAL_MOUNT = source._INTERNAL_MOUNT
ARMS = source.ARMS
ACTORS = source.ACTORS
BASELINES = source.BASELINES
INITIAL_SOURCES = source.INITIAL_SOURCES
DEV_SEEDS = source.DEV_SEEDS
EVAL_TIMEOUT = source.EVAL_TIMEOUT
LOG_LIMIT = source.LOG_LIMIT
ARCHIVE_LIMIT = source.ARCHIVE_LIMIT
MINIMUM_FREE = source.MINIMUM_FREE
INTEGRITY_EXIT = source.INTEGRITY_EXIT
CUSTODY_POLICY = {
    "schema": "plant-mpc-content-custody-v1",
    "identity_fields": ["path", "logical_bytes", "sha256"],
    "allocated_bytes": "observed-storage-not-content-identity",
}
SOURCE_BINDINGS = {
    "training/A-mpc-residual-scalar/final-policy.zip": (
        972651,
        "4031802ee0d8b7042de4b5193ab82797479dfa800d6943f62b134aa0e415e524",
    ),
    "training/B-mpc-residual-pi/final-policy.zip": (
        1639652,
        "881140cb3ddfa52d975017bb6cbbcbcecb8fbeb147a91cdd6a7db8430fd8e05d",
    ),
    "training/A-mpc-residual-scalar/initial-policy.pt": (
        111356,
        "ba9ab03b73f403c05664d03beb39909c0df53e8e459980768673c86819e7bf70",
    ),
    "training/B-mpc-residual-pi/initial-policy.pt": (
        330420,
        "263f21d2111f5382f8e93ef9b3d401a33e735c8d0dfa33cc56524f0b9a9a5029",
    ),
    "training/A-mpc-residual-scalar/initial-policy-record.json": (
        281127,
        "989cfea08a08d91a6076ba87318e391a170123ce136a8d6bd005a4b5b00b58b3",
    ),
    "training/B-mpc-residual-pi/initial-policy-record.json": (
        285966,
        "1b7fd963aa8cc153a906efed86014f841aad2a1f8c3dd4e4b766169bf05310f2",
    ),
    "training/A-mpc-residual-scalar/result.json": (
        572107,
        "87b351571f5d1ef17ad612f596f01ce1e77ba52d8f24d6b716d8accb9b26da8b",
    ),
    "training/B-mpc-residual-pi/result.json": (
        592451,
        "2512b9ead273bce18d8733d2ba22950d895633d6fdca79ff8bb497b0c962feb2",
    ),
    "initial-matching.json": (
        3596,
        "10558d1d585bc827d6c3dc484e7a90f4d88cb9fd46dfebc946dda7dd84918b24",
    ),
}


def source_config() -> Any:
    """Preserve the checkpoint's serialized configuration, including output_dir."""
    return source.template(SOURCE_ROOT)


def required_source_artifacts() -> dict[str, dict[str, Any]]:
    return {
        name: {"path": str(SOURCE_ROOT / name), "bytes": size, "sha256": digest}
        for name, (size, digest) in SOURCE_BINDINGS.items()
    }


def power_policy() -> dict[str, Any]:
    policy = source.power_policy()
    policy.pop("training_check_interval_seconds")
    policy["sampling"] = ["before_root_claim", "stage_entry_and_exit"]
    return policy


def specification(root: Path = ROOT) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "root": str(root),
        "source_root": str(SOURCE_ROOT),
        "source_commit": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "source_artifacts": required_source_artifacts(),
        "checkpoint_import_schema": PACKET_SCHEMA,
        "custody_policy": deepcopy(CUSTODY_POLICY),
        "environment_template": source_config().to_dict(),
        "actors": list(ACTORS),
        "development_seeds": list(DEV_SEEDS),
        "evaluation_slots": 56,
        "replay_slots": 56,
        "training_slots": 0,
        "optimizer_updates": 0,
        "checkpoint_selection": "exact preserved batch-007 endpoints only",
        "source_batch_status": "aborted; no normal batch-007 admission",
        "development_seed_status": "reused adaptive development; 26 prior evaluation and 26 replay attempts; none reused as batch-008 results",
        "execution": "sequential evaluation and independent replay; no training wave",
        "evaluation_timeout_seconds": EVAL_TIMEOUT,
        "stream_limit_bytes": LOG_LIMIT,
        "archive_limit_bytes": ARCHIVE_LIMIT,
        "minimum_free_bytes": MINIMUM_FREE,
        "torch_intraop_threads": 1,
        "torch_interop_threads": 1,
        "power_policy": power_policy(),
        "registered_evaluation_eligible": False,
    }


def _checked_record(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise ValueError("checkpoint import file record fields differ")
    if not isinstance(value["path"], str):
        raise ValueError("checkpoint import file record path differs")
    path = Path(value["path"])
    if (
        not path.is_absolute()
        or path != path.resolve()
        or type(value["bytes"]) is not int
        or value["bytes"] < 0
        or file_record(path) != value
    ):
        raise ValueError("checkpoint import artifact identity differs")
    return value


def _current_tree() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--verify", "HEAD^{tree}"],
        cwd=Path(__file__).resolve().parents[6],
        text=True,
        timeout=10,
    ).strip()


def require_packet(
    path: Path,
    expected_record: object | None = None,
    *,
    runtime: BetaTrainingRuntimeProvenanceV1 | None = None,
) -> dict[str, Any]:
    """Recheck admitted provenance and bytes without interpreting training outcomes."""
    if (
        not path.is_absolute()
        or path != path.resolve()
        or not path.is_relative_to(ACTIVE_BASE)
        or path.is_relative_to(ROOT)
        or path.is_relative_to(SOURCE_ROOT)
    ):
        raise ValueError("checkpoint import packet path differs")
    raw = read_regular(path)
    packet = json.loads(raw)
    record = file_record(path)
    if record["sha256"] != hashlib.sha256(raw).hexdigest() or (
        expected_record is not None and record != expected_record
    ):
        raise ValueError("checkpoint import packet changed")
    if not isinstance(packet, dict) or (
        packet.get("schema") != PACKET_SCHEMA
        or packet.get("verified") is not True
        or any(
            packet.get(key) is not False
            for key in (
                "normal_batch_007_admitted",
                "evaluation_artifacts_read",
                "policy_inference_or_learning_executed",
                "simulator_resets_executed",
            )
        )
        or canonical_bytes(packet.get("custody_policy"))
        != canonical_bytes(CUSTODY_POLICY)
    ):
        raise ValueError("checkpoint import admission or custody policy differs")
    for key in ("import_tools_manifest", "verifier_source"):
        artifact = packet.get(key)
        if (
            not isinstance(artifact, dict)
            or not isinstance(artifact.get("path"), str)
            or not Path(artifact["path"]).is_relative_to(ACTIVE_BASE)
            or Path(artifact["path"]).is_relative_to(ROOT)
            or Path(artifact["path"]).is_relative_to(SOURCE_ROOT)
        ):
            raise ValueError("checkpoint import verifier identity differs")
        _checked_record(artifact)
    current = runtime if runtime is not None else capture(require_clean=True)
    if packet.get("evaluation_source") != {
        "git_head": current.git_head,
        "git_tree": _current_tree(),
    }:
        raise ValueError("checkpoint import evaluation source differs")
    origin = packet.get("source_batch")
    if not isinstance(origin, dict) or any(
        type(origin.get(key)) is not type(value) or origin.get(key) != value
        for key, value in {
            "root": str(SOURCE_ROOT),
            "git_head": SOURCE_HEAD,
            "git_tree": SOURCE_TREE,
            "status": "aborted",
            "normal_batch_admitted": False,
        }.items()
    ):
        raise ValueError("checkpoint import training source differs")
    for key in (
        "plan",
        "report",
        "gate",
        "training_wave",
        "tools_manifest",
        "forensic_inventory",
    ):
        if not isinstance(origin.get(key), dict) or set(origin[key]) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise ValueError("checkpoint import source evidence is incomplete")
    supervision = packet.get("supervision")
    matching = packet.get("initial_tensor_comparison")
    if (
        not isinstance(supervision, dict)
        or not isinstance(matching, dict)
        or supervision.get("verified") is not True
        or supervision.get("prospective_contract") is not True
        or matching.get("performed") is not True
        or matching.get("passed") is not True
    ):
        raise ValueError("checkpoint import custody or initial matching is unverified")
    external = {
        value["path"]
        for value in origin.values()
        if isinstance(value, dict) and set(value) == {"path", "bytes", "sha256"}
    }
    records = packet.get("artifact_records")
    if not isinstance(records, list) or not records:
        raise ValueError("checkpoint import artifact inventory is absent")
    observed = {}
    for value in records:
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            raise ValueError("checkpoint import artifact path is absent")
        artifact = Path(value["path"])
        if artifact.is_relative_to(SOURCE_ROOT):
            relative = artifact.relative_to(SOURCE_ROOT)
            allowed = (
                relative.as_posix()
                in {
                    "plan.json",
                    "report.json",
                    "training-wave.json",
                    "initial-matching.json",
                }
                or relative.parts[0]
                in {"training", "logs", "supervisors", "power-failures"}
                or (
                    relative.parts[0] == "receipts"
                    and artifact.name.startswith("train-")
                )
            )
        else:
            allowed = str(artifact) in external and artifact.is_relative_to(ACTIVE_BASE)
        if not allowed or str(artifact) in observed:
            raise ValueError(
                "checkpoint import includes forbidden or duplicate artifacts"
            )
        observed[str(artifact)] = _checked_record(value)
    for value in origin.values():
        if isinstance(value, dict) and set(value) == {"path", "bytes", "sha256"}:
            if observed.get(value["path"]) != value:
                raise ValueError("checkpoint import source record is not inventoried")
    required = required_source_artifacts()
    if any(observed.get(value["path"]) != value for value in required.values()):
        raise ValueError("checkpoint import fixed endpoint binding differs")
    endpoints = packet.get("training_endpoints")
    if not isinstance(endpoints, dict) or set(endpoints) != {arm.name for arm in ARMS}:
        raise ValueError("checkpoint import endpoint set differs")
    for arm in ARMS:
        endpoint = endpoints[arm.name]
        if not isinstance(endpoint, dict) or any(
            type(endpoint.get(key)) is not type(value) or endpoint.get(key) != value
            for key, value in {
                "verified": True,
                "source_root": str(SOURCE_ROOT),
                "total_timesteps": 65536,
                "rollout_count": 8,
                "optimizer_steps": 2560,
                "allocator_reset_count": 100,
                "model_attestation_verified": True,
            }.items()
        ):
            raise ValueError("checkpoint import completed endpoint metadata differs")
        for key, filename in (
            ("initial_policy", "initial-policy.pt"),
            ("initial_policy_record", "initial-policy-record.json"),
            ("final_policy", "final-policy.zip"),
            ("result", "result.json"),
        ):
            if endpoint.get(key) != required[f"training/{arm.name}/{filename}"]:
                raise ValueError("checkpoint import endpoint artifact differs")
    return cast(dict[str, Any], packet)


def read_import_admission(path: Path) -> dict[str, Any]:
    return require_packet(path)


def verify_imported_models() -> None:
    """Verify the sealed fixed models before root claim, without prediction/reset."""
    config = source_config()
    for arm in ARMS:
        source.load_final(SOURCE_ROOT / "training" / arm.name, config, arm)
    source.load_initial(SOURCE_ROOT / "training" / ARMS[0].name, config, ARMS[0])
    if source.verify_initial_matching(SOURCE_ROOT, config).get("matched") is not True:
        raise TrainingIntegrityError("imported initial policy matching differs")


def require_power(
    *, failure_directory: Path, admission: bool = False
) -> dict[str, Any]:
    if not admission and os.path.lexists(failure_directory):
        raise source.PowerAdmissionError(
            f"power_previous_refusal; evidence={failure_directory}"
        )
    sample = source.capture_power_sample()
    if sample["status"] == "accepted":
        source.validate_power_admission(sample)
        return sample
    prefix = f"{ROOT.name}-power-admission" if admission else "power"
    path = (
        failure_directory
        / f"{prefix}-{sample['pid']}-{sample['closed_monotonic_ns']}.json"
    )
    try:
        if not admission:
            mkdir_exclusive(failure_directory)
        publish_json(path, sample)
    except Exception as exc:
        raise source.PowerAdmissionError(
            f"{sample['code']}; power evidence publication failed: {type(exc).__name__}"
        ) from exc
    raise source.PowerAdmissionError(f"{sample['code']}; evidence={path}")


def _storage_identity(gate: Path, *, require_gate: bool) -> dict[str, Any]:
    if (
        ACTIVE_BASE != ACTIVE_BASE.resolve()
        or ROOT.parent != ACTIVE_BASE
        or ROOT != ROOT.resolve()
        or not gate.is_absolute()
        or not gate.is_relative_to(ACTIVE_BASE)
        or gate.is_relative_to(ROOT)
        or gate != gate.resolve()
    ):
        raise ValueError("evaluation storage paths must remain canonical and internal")
    paths = [
        ACTIVE_BASE,
        _INTERNAL_MOUNT,
        Path(__file__).resolve().parents[6],
        gate.parent,
    ]
    for path in paths:
        require_directory(path)
    if os.path.lexists(ROOT):
        require_directory(ROOT)
        paths.append(ROOT)
    if os.path.lexists(gate):
        entry = gate.lstat()
        if not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1:
            raise ValueError("evaluation gate must be a singly linked regular file")
        paths.append(gate)
    elif require_gate:
        raise ValueError("evaluation storage gate is absent")
    device = ACTIVE_BASE.stat().st_dev
    if any(path.stat().st_dev != device for path in paths):
        raise ValueError(
            "evaluation storage device differs from the internal repository"
        )
    if shutil.disk_usage(ACTIVE_BASE).free < MINIMUM_FREE:
        raise ValueError("evaluation needs 10 GiB free on internal active storage")
    return {
        "schema": STORAGE_SCHEMA,
        "active_base": str(ACTIVE_BASE),
        "batch_root": str(ROOT),
        "gate_path": str(gate),
        "volume": {
            "uuid": _INTERNAL_VOLUME_UUID,
            "mount_point": str(_INTERNAL_MOUNT),
            "filesystem_type": "apfs",
            "internal": True,
            "removable": False,
            "ejectable": False,
            "st_dev": device,
        },
    }


def capture_storage_identity(gate: Path) -> dict[str, Any]:
    if os.path.lexists(ROOT):
        raise ValueError("evaluation root must be absent; no retry")
    identity = _storage_identity(gate, require_gate=False)
    try:
        volume = plistlib.loads(
            subprocess.check_output(
                ["/usr/sbin/diskutil", "info", "-plist", str(_INTERNAL_MOUNT)],
                stderr=subprocess.PIPE,
                timeout=10,
            )
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise ValueError("evaluation internal storage OS admission failed") from exc
    expected = {
        "VolumeUUID": _INTERNAL_VOLUME_UUID,
        "MountPoint": str(_INTERNAL_MOUNT),
        "FilesystemType": "apfs",
        "Internal": True,
        "Removable": False,
        "Ejectable": False,
    }
    if not isinstance(volume, dict) or any(
        type(volume.get(key)) is not type(value) or volume.get(key) != value
        for key, value in expected.items()
    ):
        raise ValueError("evaluation storage must use the frozen internal APFS volume")
    if os.path.lexists(ROOT) or _storage_identity(gate, require_gate=False) != identity:
        raise ValueError("evaluation storage changed during admission")
    return identity


def require_storage(gate: Path, expected: object) -> dict[str, Any]:
    identity = _storage_identity(gate, require_gate=True)
    if canonical_bytes(identity) != canonical_bytes(expected):
        raise ValueError("evaluation storage identity changed")
    return identity


def require_plan() -> dict[str, Any]:
    from .capacity_benchmark import _host_identity

    plan = json.loads(read_regular(ROOT / "plan.json"))
    gate_path = Path(plan["gate"]["path"])
    storage = require_storage(gate_path, plan["storage"])
    gate = json.loads(read_regular(gate_path))
    if (
        canonical_bytes(plan["specification"]) != canonical_bytes(specification(ROOT))
        or canonical_bytes(gate.get("storage")) != canonical_bytes(storage)
        or canonical_bytes(gate.get("power_policy")) != canonical_bytes(power_policy())
        or gate.get("checkpoint_import") != plan["checkpoint_import"]
        or gate.get("import_tools_manifest") != plan["import_tools_manifest"]
    ):
        raise ValueError("evaluation plan or gate contract differs")
    source.validate_power_admission(plan.get("power_admission"))
    runtime = capture(require_clean=True)
    verify_unchanged(
        BetaTrainingRuntimeProvenanceV1.from_dict(plan["runtime"]), runtime
    )
    if _host_identity() != plan["host"] or file_record(gate_path) != plan["gate"]:
        raise ValueError("evaluation host or gate changed")
    packet = require_packet(
        Path(plan["checkpoint_import"]["path"]),
        plan["checkpoint_import"],
        runtime=runtime,
    )
    if packet["import_tools_manifest"] != plan["import_tools_manifest"]:
        raise ValueError("evaluation import manifest differs")
    require_power(failure_directory=ROOT / "power-failures")
    return cast(dict[str, Any], plan)


def evaluate_cell(actor: str, seed: int) -> None:
    from .research_evaluation import evaluate_episode
    from .research_mpc_residual_evaluation import evaluate_residual_episode

    directory = ROOT / "evaluations" / f"{seed}-{actor}"
    publish_json(
        directory / "attempt.json", {"actor": actor, "seed": seed, "started_at": now()}
    )
    config = source_config()
    if actor in INITIAL_SOURCES:
        arm = INITIAL_SOURCES[actor]
        policy = source.load_initial(SOURCE_ROOT / "training" / arm.name, config, arm)
        episode = evaluate_residual_episode(
            config, policy=policy, seed=seed, policy_id=actor
        )
    elif actor not in BASELINES:
        arm = next(arm for arm in ARMS if arm.name == actor)
        policy = source.load_final(SOURCE_ROOT / "training" / actor, config, arm)
        episode = evaluate_residual_episode(
            config, policy=policy, seed=seed, policy_id=actor
        )
    else:
        episode = evaluate_episode(
            config, actor=actor, policy=None, seed=seed, policy_id=actor
        )
    encoded = gzip.compress(canonical_bytes(episode), compresslevel=6, mtime=0)
    if len(encoded) > ARCHIVE_LIMIT:
        raise ValueError("evaluation archive exceeded stored-size cap")
    publish_bytes(directory / "episode.json.gz", encoded)
    publish_json(
        directory / "summary.json",
        {
            "actor": actor,
            "seed": seed,
            "summary": episode["summary"],
            "archive": file_record(directory / "episode.json.gz"),
        },
    )


def validate_receipt_closure(receipt: dict[str, Any], extra: list[str]) -> None:
    if (
        receipt["process"].get("exit_code") == INTEGRITY_EXIT
        or "root_custody_lost" in receipt["codes"]
    ):
        raise TrainingIntegrityError(
            "evaluation receipt records evidence integrity loss"
        )
    if (
        any(code not in source.ORDINARY_CHILD_CODES for code in receipt["codes"])
        or not receipt.get("closed_at")
        or receipt["process"].get("command")
        != [sys.executable, "-m", MODULE, receipt["stage"], *extra]
    ):
        raise ValueError("evaluation receipt is not an ordinary closed attempt")


def require_successful_evaluation(actor: str, seed: int) -> None:
    name = f"evaluate-{seed}-{actor}"
    receipt = json.loads(read_regular(ROOT / "receipts" / f"{name}.json"))
    if (
        receipt.get("stage") != "_evaluate"
        or receipt.get("name") != name
        or not isinstance(receipt.get("codes"), list)
    ):
        raise ValueError("evaluation receipt identity is inconsistent")
    validate_receipt_closure(receipt, ["--actor", actor, "--seed", str(seed)])
    if receipt["codes"]:
        raise TrainingUnavailable("replay has no successfully closed evaluation child")
    if (
        receipt["process"].get("exit_code") != 0
        or receipt["process"].get("started") is not True
    ):
        raise ValueError("evaluation receipt omits a child failure")


def replay_cell(actor: str, seed: int) -> None:
    from .research_evaluation import verify_episode_replay
    from .research_mpc_residual_evaluation import ACTOR as residual_actor
    from .research_mpc_residual_evaluation import verify_residual_episode_replay

    directory = ROOT / "evaluations" / f"{seed}-{actor}"
    publish_json(
        directory / "replay-attempt.json",
        {"actor": actor, "seed": seed, "started_at": now()},
    )
    require_successful_evaluation(actor, seed)
    summary = json.loads(read_regular(directory / "summary.json"))
    archive = directory / "episode.json.gz"
    if file_record(archive) != summary["archive"]:
        raise ValueError("evaluation trajectory changed before replay")
    if summary["archive"]["bytes"] > ARCHIVE_LIMIT:
        raise ValueError("replay archive exceeded stored-size cap")
    episode = json.loads(gzip.decompress(read_regular(archive)))
    if (
        summary.get("actor") != actor
        or canonical_bytes(summary.get("seed")) != canonical_bytes(seed)
        or canonical_bytes(episode.get("seed")) != canonical_bytes(seed)
        or canonical_bytes(summary.get("summary"))
        != canonical_bytes(episode.get("summary"))
        or episode["summary"].get("policy_id") != actor
        or episode["summary"].get("actor")
        != (actor if actor in BASELINES else residual_actor)
    ):
        raise ValueError("replay slot identity or summary differs")
    result = (
        verify_episode_replay if actor in BASELINES else verify_residual_episode_replay
    )(source_config(), episode)
    publish_json(directory / "replay.json", result)


def closed_child(
    stage: str, name: str, timeout: int, extra: list[str]
) -> dict[str, Any]:
    if stage not in ("_evaluate", "_replay"):
        raise ValueError("evaluation-only runner rejects this child stage")
    source.configure_threads()
    require_plan()
    process, codes = capture_process(
        [sys.executable, "-m", MODULE, stage, *extra],
        stdout=ROOT / "logs" / f"{name}.stdout.log",
        stderr=ROOT / "logs" / f"{name}.stderr.log",
        timeout_seconds=timeout,
        log_limit_bytes=LOG_LIMIT,
    )
    receipt = {
        "stage": stage,
        "name": name,
        "process": process,
        "codes": sorted(codes),
        "closed_at": now(),
    }
    publish_json(ROOT / "receipts" / f"{name}.json", receipt)
    if "root_custody_lost" in codes or process["exit_code"] == INTEGRITY_EXIT:
        raise TrainingIntegrityError(
            "evaluation child integrity failure; batch aborted"
        )
    require_plan()
    return receipt


def _reached_receipts() -> list[dict[str, Any]]:
    """Include a fatal child's already-published receipt in forensic accounting."""
    records = []
    for seed in DEV_SEEDS:
        for actor in ACTORS:
            for label in ("evaluate", "replay"):
                path = ROOT / "receipts" / f"{label}-{seed}-{actor}.json"
                if os.path.lexists(path):
                    records.append(json.loads(read_regular(path)))
    return records


def run(gate: Path, *, authorized: bool) -> dict[str, Any]:
    from .capacity_benchmark import _host_identity

    if not authorized:
        raise ValueError("evaluation requires research authorization assertion")
    if os.path.lexists(ROOT) or ROOT != ROOT.resolve():
        raise ValueError("evaluation root must be absent and canonical; no retry")
    _storage_identity(gate, require_gate=True)
    runtime = capture(require_clean=True)
    raw = read_regular(gate)
    checked = json.loads(raw)
    if (
        checked.get("passed") is not True
        or checked.get("source_commit") != runtime.git_head
    ):
        raise ValueError("evaluation gate must pass at the exact clean source")
    if canonical_bytes(checked.get("power_policy")) != canonical_bytes(power_policy()):
        raise ValueError("evaluation gate power policy differs")
    storage = capture_storage_identity(gate)
    require_storage(gate, checked.get("storage"))
    imported = checked["checkpoint_import"]
    packet = require_packet(Path(imported["path"]), imported, runtime=runtime)
    if checked.get("import_tools_manifest") != packet["import_tools_manifest"]:
        raise ValueError("evaluation gate import manifest differs")
    verify_imported_models()
    gate_record = file_record(gate)
    if gate_record["sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("evaluation gate changed during admission")
    plan = {
        "specification": specification(ROOT),
        "runtime": runtime.to_dict(),
        "host": _host_identity(),
        "gate": gate_record,
        "storage": storage,
        "checkpoint_import": imported,
        "import_tools_manifest": packet["import_tools_manifest"],
        "started_at": now(),
        "power_admission": require_power(failure_directory=gate.parent, admission=True),
    }
    mkdir_exclusive(ROOT)
    publish_json(ROOT / "plan.json", plan)
    for name in ("logs", "receipts", "evaluations"):
        mkdir_exclusive(ROOT / name)
    receipts = []
    failure = None
    try:
        for seed in DEV_SEEDS:
            for actor in ACTORS:
                name = f"{seed}-{actor}"
                mkdir_exclusive(ROOT / "evaluations" / name)
                for stage, label in (("_evaluate", "evaluate"), ("_replay", "replay")):
                    receipts.append(
                        closed_child(
                            stage,
                            f"{label}-{name}",
                            EVAL_TIMEOUT,
                            ["--actor", actor, "--seed", str(seed)],
                        )
                    )
    except BaseException as reason:
        failure = {"type": type(reason).__name__, "message": str(reason)}
        try:
            receipts = _reached_receipts()
        except Exception as exc:
            failure["receipt_collection_failure"] = f"{type(exc).__name__}: {exc}"
    report = {
        "schema": SCHEMA,
        "status": "aborted" if failure else "closed",
        "closed_at": now(),
        "failure": failure,
        "plan": file_record(ROOT / "plan.json"),
        "checkpoint_import": imported,
        "receipts": receipts,
        "planned_slots": {
            "training": [],
            "evaluation": [f"{seed}-{actor}" for seed in DEV_SEEDS for actor in ACTORS],
            "replay": [f"{seed}-{actor}" for seed in DEV_SEEDS for actor in ACTORS],
        },
        "claim": "Adaptive checkpoint development evaluation; no selected policy or confirmation claim.",
        "registered_evaluation_eligible": False,
    }
    publish_json(ROOT / "report.json", report)
    return report


def execute_child(action: Callable[[], Any]) -> int:
    try:
        require_plan()
        action()
        require_plan()
    except TrainingUnavailable:
        raise
    except Exception as exc:
        print(
            json.dumps({"integrity_failure": type(exc).__name__, "message": str(exc)}),
            file=sys.stderr,
        )
        return INTEGRITY_EXIT
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "_evaluate", "_replay"))
    parser.add_argument("--assert-authorized", action="store_true")
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--actor", choices=ACTORS)
    parser.add_argument("--seed", type=int, choices=DEV_SEEDS)
    args = parser.parse_args(argv)
    source.configure_threads()
    if args.command == "run":
        if args.gate is None:
            parser.error("run requires --gate")
        result = run(args.gate, authorized=args.assert_authorized)
        print(
            json.dumps(
                {"status": result["status"], "report": str(ROOT / "report.json")}
            )
        )
        return 1 if result["failure"] else 0
    if args.actor is None or args.seed is None:
        parser.error("evaluation/replay requires --actor and --seed")
    return execute_child(
        lambda: (evaluate_cell if args.command == "_evaluate" else replay_cell)(
            args.actor, args.seed
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

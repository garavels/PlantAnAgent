"""Batch 009: paired fresh scalar reward discounts within a fixed wall deadline."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import plistlib
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Callable, cast

from . import research_mpc_constraint_batch as source
from .config import BETA_SCALAR_PPO_ALGORITHM_ID as SCALAR
from .config import ControllerNeutralTrainingConfigV1
from .episode_allocator import EpisodeSeedCustodyError
from .research_mpc_residual import MPCResidualIntegrityError
from .custody_io import mkdir_exclusive, publish_bytes, publish_json, read_regular
from .custody_io import require_directory
from .custody_json import canonical_bytes
from .custody_process import capture_process
from .research_batch import file_record, now
from .research_effort_batch import Arm, TrainingIntegrityError, TrainingUnavailable
from .runtime_provenance import (
    BetaTrainingRuntimeProvenanceV1,
    capture,
    verify_unchanged,
)

MODULE = "plant_backend.rl.v2.training.research_reward_alignment_batch"
SCHEMA = "plant-adaptive-rl-reward-alignment-batch-009"
PACKET_SCHEMA = "plant-mpc-constraint-009-reference-admission-v1"
ACTIVE_BASE = Path("/Users/filippogombac/Developer/PlantAnAgentResearch")
ROOT = ACTIVE_BASE / "adaptive-rl-batch-009"
REFERENCE_ROOT = ACTIVE_BASE / "adaptive-rl-batch-008"
INITIAL_ROOT = ACTIVE_BASE / "adaptive-rl-batch-007"
REFERENCE_ANALYSIS_SHA256 = (
    "9b4e86afe39148a9ad5f9877c8a2ddcdb5e135bbb104ba6043d51a76da56934c"
)
REFERENCE_AUDIT_SHA256 = (
    "22aff1c0aa3b7c4a3b9bf78d6ec9a7825c14b88534dbe77537021bd66f6d9536"
)
INITIAL_BINDINGS = {
    "initial_policy": (
        "training/A-mpc-residual-scalar/initial-policy.pt",
        111356,
        "ba9ab03b73f403c05664d03beb39909c0df53e8e459980768673c86819e7bf70",
    ),
    "initial_policy_record": (
        "training/A-mpc-residual-scalar/initial-policy-record.json",
        281127,
        "989cfea08a08d91a6076ba87318e391a170123ce136a8d6bd005a4b5b00b58b3",
    ),
}
ARMS = (
    Arm("A-scalar-gamma0995", SCALAR, 0.001, 0.2, True),
    Arm("B-scalar-gamma1000", SCALAR, 0.001, 0.2, True),
)
GAMMAS = {ARMS[0].name: 0.995, ARMS[1].name: 1.0}
ACTORS = tuple(arm.name for arm in ARMS)
REFERENCE_ACTORS = source.ACTORS
DEV_SEEDS = source.DEV_SEEDS
TRAIN_SEED_BASE = source.TRAIN_SEED_BASE
OPTIMIZER_SEED = source.OPTIMIZER_SEED
TOTAL_STEPS = 16384
TRAIN_TIMEOUT = 12600
EVAL_TIMEOUT = source.EVAL_TIMEOUT
LOG_LIMIT = source.LOG_LIMIT
ARCHIVE_LIMIT = source.ARCHIVE_LIMIT
MINIMUM_FREE = source.MINIMUM_FREE
INTEGRITY_EXIT = source.INTEGRITY_EXIT
INTEGRITY_CHECK_INTERVAL = source.INTEGRITY_CHECK_INTERVAL
STORAGE_SCHEMA = source.STORAGE_SCHEMA
_INTERNAL_VOLUME_UUID = source._INTERNAL_VOLUME_UUID
_INTERNAL_MOUNT = source._INTERNAL_MOUNT
WINDOW_STARTED_AT = "2026-09-14T21:01:28+00:00"
LATEST_LAUNCH_AT = "2026-09-14T23:15:00+00:00"
PRODUCTION_DEADLINE_AT = "2026-09-15T06:30:00+00:00"
FINAL_DEADLINE_AT = "2026-09-15T07:01:28+00:00"
CLOSURE_RESERVE_SECONDS = 90
CUSTODY_POLICY = {
    "schema": "plant-mpc-content-custody-v1",
    "identity_fields": ["path", "logical_bytes", "sha256"],
    "allocated_bytes": "observed-storage-not-content-identity",
}


def cell_config(arm: Arm | str, root: Path = ROOT) -> ControllerNeutralTrainingConfigV1:
    name = arm if isinstance(arm, str) else arm.name
    if name not in GAMMAS or (not isinstance(arm, str) and arm not in ARMS):
        raise ValueError("unknown reward-alignment arm")
    config = source.template(root)
    return replace(
        config,
        total_timesteps=TOTAL_STEPS,
        ppo=replace(config.ppo, gamma=GAMMAS[name]),
    ).validate()


def deadline_policy() -> dict[str, Any]:
    return {
        "schema": "plant-adaptive-deadline-policy-v1",
        "window_started_at": WINDOW_STARTED_AT,
        "latest_launch_at": LATEST_LAUNCH_AT,
        "production_deadline_at": PRODUCTION_DEADLINE_AT,
        "final_deadline_at": FINAL_DEADLINE_AT,
        "training_timeout_seconds": TRAIN_TIMEOUT,
        "evaluation_timeout_seconds": EVAL_TIMEOUT,
        "closure_reserve_seconds": CLOSURE_RESERVE_SECONDS,
        "integrity_check_interval_seconds": INTEGRITY_CHECK_INTERVAL,
        "clock_rule": "earliest of absolute UTC and admission-anchored monotonic deadline",
        "timeout_rule": "wave admission or serial capture: positive integer min(stage cap, floor(remaining production seconds minus closure reserve)); trainers retain registered wave timeout if still within remaining production time",
        "failure_action": "abort-no-retry-no-partial-endpoint-admission",
    }


def _utc() -> datetime:
    return datetime.now(timezone.utc)


def _utc_value(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("deadline timestamp is absent")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("deadline timestamp must be UTC")
    return parsed


def _interval_ns(later: datetime, earlier: datetime) -> int:
    delta = later - earlier
    return (delta.days * 86400 + delta.seconds) * 10**9 + delta.microseconds * 1000


def capture_deadline_admission() -> dict[str, Any]:
    admitted = _utc()
    monotonic = time.monotonic_ns()
    if not _utc_value(WINDOW_STARTED_AT) <= admitted < _utc_value(LATEST_LAUNCH_AT):
        raise TrainingIntegrityError("deadline admission window is closed")
    record = {
        "schema": "plant-adaptive-deadline-admission-v1",
        "policy": deadline_policy(),
        "admitted_at": admitted.isoformat(),
        "admitted_monotonic_ns": monotonic,
        "production_deadline_monotonic_ns": monotonic
        + _interval_ns(_utc_value(PRODUCTION_DEADLINE_AT), admitted),
    }
    require_deadline(record, admission=True)
    return record


def require_deadline(record: object, *, admission: bool = False) -> float:
    if (
        not isinstance(record, dict)
        or set(record)
        != {
            "schema",
            "policy",
            "admitted_at",
            "admitted_monotonic_ns",
            "production_deadline_monotonic_ns",
        }
        or record.get("schema") != "plant-adaptive-deadline-admission-v1"
        or canonical_bytes(record.get("policy")) != canonical_bytes(deadline_policy())
    ):
        raise TrainingIntegrityError("deadline admission contract differs")
    admitted = _utc_value(record["admitted_at"])
    anchor = record["admitted_monotonic_ns"]
    deadline = record["production_deadline_monotonic_ns"]
    if (
        type(anchor) is not int
        or anchor < 0
        or type(deadline) is not int
        or not _utc_value(WINDOW_STARTED_AT) <= admitted < _utc_value(LATEST_LAUNCH_AT)
        or deadline
        != anchor + _interval_ns(_utc_value(PRODUCTION_DEADLINE_AT), admitted)
    ):
        raise TrainingIntegrityError("deadline clock anchors differ")
    current, monotonic = _utc(), time.monotonic_ns()
    if current < admitted or monotonic < anchor:
        raise TrainingIntegrityError("deadline clock regressed")
    remaining = min(
        (_utc_value(PRODUCTION_DEADLINE_AT) - current).total_seconds(),
        (deadline - monotonic) / 10**9,
    )
    if remaining <= CLOSURE_RESERVE_SECONDS or (
        admission and current >= _utc_value(LATEST_LAUNCH_AT)
    ):
        raise TrainingIntegrityError("production deadline exhausted")
    return remaining


def remaining_timeout(cap: int, deadline: object) -> int:
    if type(cap) is not int or cap <= 0:
        raise ValueError("invalid child timeout cap")
    timeout = min(cap, math.floor(require_deadline(deadline) - CLOSURE_RESERVE_SECONDS))
    if timeout <= 0:
        raise TrainingIntegrityError("no child time remains before deadline")
    return timeout


def deadline_check(
    admission: dict[str, Any],
    *,
    stage: str,
    name: str,
    cap: int,
    requested: int | None = None,
) -> dict[str, Any]:
    require_deadline(admission)
    checked, monotonic = _utc(), time.monotonic_ns()
    remaining = min(
        _interval_ns(_utc_value(PRODUCTION_DEADLINE_AT), checked),
        admission["production_deadline_monotonic_ns"] - monotonic,
    )
    available = (remaining - CLOSURE_RESERVE_SECONDS * 10**9) // 10**9
    effective = min(cap, available) if requested is None else requested
    if (
        type(cap) is not int
        or cap <= 0
        or type(effective) is not int
        or not 0 < effective <= cap
        or remaining <= CLOSURE_RESERVE_SECONDS * 10**9
        or effective * 10**9 >= remaining
    ):
        raise TrainingIntegrityError("deadline check has no admissible child timeout")
    return {
        "schema": "plant-reward-alignment-deadline-check-v1",
        "stage": stage,
        "name": name,
        "deadline_admission": deepcopy(admission),
        "checked_at": checked.isoformat(),
        "checked_monotonic_ns": monotonic,
        "remaining_ns": remaining,
        "stage_cap_seconds": cap,
        "requested_timeout_seconds": requested,
        "effective_timeout_seconds": effective,
        "binding": (
            "stage_capture" if requested is None else "registered_parallel_attempt"
        ),
    }


def validate_deadline_check(value: object, admission: dict[str, Any]) -> dict[str, Any]:
    fields = {
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
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or (
            value["schema"] != "plant-reward-alignment-deadline-check-v1"
            or canonical_bytes(value["deadline_admission"])
            != canonical_bytes(admission)
            or value["stage"] not in ("_train_wave", "_train", "_evaluate", "_replay")
            or not isinstance(value["name"], str)
            or not value["name"]
        )
    ):
        raise TrainingIntegrityError("deadline sidecar fields differ")
    require_deadline(admission)
    checked = _utc_value(value["checked_at"])
    monotonic = value["checked_monotonic_ns"]
    for key in (
        "checked_monotonic_ns",
        "remaining_ns",
        "stage_cap_seconds",
        "effective_timeout_seconds",
    ):
        if type(value[key]) is not int:
            raise TrainingIntegrityError("deadline sidecar counters differ")
    if (
        not _utc_value(admission["admitted_at"]) <= checked <= _utc()
        or not admission["admitted_monotonic_ns"] <= monotonic <= time.monotonic_ns()
    ):
        raise TrainingIntegrityError("deadline sidecar clocks differ")
    remaining = min(
        _interval_ns(_utc_value(PRODUCTION_DEADLINE_AT), checked),
        admission["production_deadline_monotonic_ns"] - monotonic,
    )
    cap = TRAIN_TIMEOUT if value["stage"] in ("_train", "_train_wave") else EVAL_TIMEOUT
    requested = value["requested_timeout_seconds"]
    expected = min(cap, (remaining - CLOSURE_RESERVE_SECONDS * 10**9) // 10**9)
    if requested is not None:
        if type(requested) is not int or value["stage"] != "_train":
            raise TrainingIntegrityError("deadline registered timeout differs")
        expected = requested
    if (
        value["remaining_ns"] != remaining
        or value["stage_cap_seconds"] != cap
        or value["effective_timeout_seconds"] != expected
        or not 0 < expected <= cap
        or expected * 10**9 >= remaining
        or remaining <= CLOSURE_RESERVE_SECONDS * 10**9
        or value["binding"]
        != ("stage_capture" if requested is None else "registered_parallel_attempt")
    ):
        raise TrainingIntegrityError("deadline timeout binding differs")
    return value


def power_policy() -> dict[str, Any]:
    return source.power_policy()


def specification(root: Path = ROOT) -> dict[str, Any]:
    from .research_parallel_supervision import SCHEMA as supervision_schema
    from .research_parallel_supervision import monitor_settings

    return {
        "schema": SCHEMA,
        "root": str(root),
        "arms": [asdict(arm) for arm in ARMS],
        "cell_configs": {arm.name: cell_config(arm, root).to_dict() for arm in ARMS},
        "reward_gammas": dict(GAMMAS),
        "training_artifact_schema": source.SCHEMA,
        "training_producer_module": source.MODULE,
        "reference_admission_schema": PACKET_SCHEMA,
        "reference_root": str(REFERENCE_ROOT),
        "reference_analysis_sha256": REFERENCE_ANALYSIS_SHA256,
        "reference_audit_sha256": REFERENCE_AUDIT_SHA256,
        "reference_actors": list(REFERENCE_ACTORS),
        "actors": list(ACTORS),
        "development_seeds": list(DEV_SEEDS),
        "train_seed_base": TRAIN_SEED_BASE,
        "train_seed_count_per_arm": 28,
        "training_slots": 2,
        "total_timesteps_per_arm": TOTAL_STEPS,
        "rollout_count_per_arm": 2,
        "optimizer_steps_per_arm": 640,
        "evaluation_slots": 16,
        "replay_slots": 16,
        "checkpoint_selection": "fixed fresh final endpoints only",
        "development_seed_status": "reused training and evaluation development material",
        "training_supervision_schema": supervision_schema,
        "training_log_monitor": monitor_settings(),
        "training_integrity_check_interval_seconds": INTEGRITY_CHECK_INTERVAL,
        "train_timeout_seconds": TRAIN_TIMEOUT,
        "evaluation_timeout_seconds": EVAL_TIMEOUT,
        "stream_limit_bytes": LOG_LIMIT,
        "archive_limit_bytes": ARCHIVE_LIMIT,
        "minimum_free_bytes": MINIMUM_FREE,
        "torch_intraop_threads": 1,
        "torch_interop_threads": 1,
        "power_policy": power_policy(),
        "deadline_policy": deadline_policy(),
        "custody_policy": deepcopy(CUSTODY_POLICY),
        "registered_evaluation_eligible": False,
    }


def _current_tree() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--verify", "HEAD^{tree}"],
        cwd=Path(__file__).resolve().parents[6],
        text=True,
        timeout=10,
    ).strip()


def source_identity() -> dict[str, str]:
    return {
        "git_head": capture(require_clean=True).git_head,
        "git_tree": _current_tree(),
    }


def required_initial_artifacts() -> dict[str, dict[str, Any]]:
    return {
        key: {"path": str(INITIAL_ROOT / name), "bytes": size, "sha256": digest}
        for key, (name, size, digest) in INITIAL_BINDINGS.items()
    }


def _checked_record(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise ValueError("reference file record fields differ")
    path = Path(value["path"]) if isinstance(value["path"], str) else Path()
    if (
        not path.is_absolute()
        or path != path.resolve()
        or not path.is_relative_to(ACTIVE_BASE)
        or path.is_relative_to(ROOT)
        or type(value["bytes"]) is not int
        or value["bytes"] < 0
        or file_record(path) != value
    ):
        raise ValueError("reference artifact identity differs")
    return value


def require_reference(
    path: Path,
    expected_record: object | None = None,
    *,
    runtime: BetaTrainingRuntimeProvenanceV1 | None = None,
    full: bool = True,
) -> dict[str, Any]:
    if (
        not path.is_absolute()
        or path != path.resolve()
        or not path.is_relative_to(ACTIVE_BASE)
        or path.is_relative_to(ROOT)
        or path.is_relative_to(REFERENCE_ROOT)
    ):
        raise ValueError("reference admission must precede the new root")
    raw = read_regular(path)
    record = file_record(path)
    if record["sha256"] != hashlib.sha256(raw).hexdigest() or (
        expected_record is not None and record != expected_record
    ):
        raise ValueError("reference packet changed")
    packet = json.loads(raw)
    current = runtime if runtime is not None else capture(require_clean=True)
    if not isinstance(packet, dict) or (
        packet.get("schema") != PACKET_SCHEMA
        or packet.get("verified") is not True
        or packet.get("execution_source")
        != {"git_head": current.git_head, "git_tree": _current_tree()}
        or packet.get("development_seeds") != list(DEV_SEEDS)
        or packet.get("actors") != list(REFERENCE_ACTORS)
        or packet.get("scientific_compatibility", {}).get("verified") is not True
        or packet.get("simulator_resets_executed") is not False
        or packet.get("policy_inference_or_learning_executed") is not False
    ):
        raise ValueError("reference source, panel or admission differs")
    origin = packet.get("source_batch", {})
    if origin.get("root") != str(REFERENCE_ROOT):
        raise ValueError("reference batch root differs")
    required = required_initial_artifacts()
    initial = packet.get("initial_reference", {})
    for key, value in required.items():
        if initial.get(key) != value:
            raise ValueError("inherited initial policy binding differs")
        _checked_record(value)
    for key, digest in (
        ("analysis", REFERENCE_ANALYSIS_SHA256),
        ("audit", REFERENCE_AUDIT_SHA256),
    ):
        value = origin.get(key)
        if not isinstance(value, dict) or value.get("sha256") != digest:
            raise ValueError("reference analysis or audit pin differs")
        _checked_record(value)
    records = packet.get("artifact_records")
    if not isinstance(records, list) or not records:
        raise ValueError("reference artifact inventory absent")
    by_path = {}
    for value in records:
        if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
            raise ValueError("reference artifact fields differ")
        name = value["path"]
        if not isinstance(name, str) or name in by_path:
            raise ValueError("reference artifact path duplicated")
        if full:
            _checked_record(value)
        by_path[name] = value
    for value in [origin["analysis"], origin["audit"], *required.values()]:
        if by_path.get(value["path"]) != value:
            raise ValueError("required reference artifact not inventoried")
    return cast(dict[str, Any], packet)


def read_reference_admission(path: Path) -> dict[str, Any]:
    return require_reference(path)


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
    require_deadline(plan.get("deadline_admission"))
    gate_path = Path(plan["gate"]["path"])
    storage = require_storage(gate_path, plan["storage"])
    gate = json.loads(read_regular(gate_path))
    if (
        canonical_bytes(plan["specification"]) != canonical_bytes(specification(ROOT))
        or canonical_bytes(gate.get("storage")) != canonical_bytes(storage)
        or canonical_bytes(gate.get("power_policy")) != canonical_bytes(power_policy())
        or canonical_bytes(gate.get("deadline_policy"))
        != canonical_bytes(deadline_policy())
        or gate.get("reference_admission") != plan["reference_admission"]
        or plan.get("initialization_admission", {}).get("verified") is not True
    ):
        raise TrainingIntegrityError("reward-alignment plan or gate contract differs")
    source.validate_power_admission(plan.get("power_admission"))
    runtime = capture(require_clean=True)
    verify_unchanged(
        BetaTrainingRuntimeProvenanceV1.from_dict(plan["runtime"]), runtime
    )
    if _host_identity() != plan["host"] or file_record(gate_path) != plan["gate"]:
        raise TrainingIntegrityError("reward-alignment host or gate changed")
    require_reference(
        Path(plan["reference_admission"]["path"]),
        plan["reference_admission"],
        runtime=runtime,
        full=False,
    )
    require_power(failure_directory=ROOT / "power-failures")
    require_deadline(plan["deadline_admission"])
    return cast(dict[str, Any], plan)


def _assert_initial_policy(
    model: Any, arm: Arm, expected: dict[str, Any]
) -> dict[str, Any]:
    import torch

    config = cell_config(arm, ROOT)
    if model.algorithm_id != SCALAR or float(model.gamma) != GAMMAS[arm.name]:
        raise TrainingIntegrityError("fresh scalar algorithm or reward gamma differs")
    source.verify_model_spaces(model, config, arm)
    weights = model.policy.state_dict()
    observed = source.tensor_records(weights, shared_only=False)
    if observed != source.tensor_records(expected, shared_only=False) or any(
        key not in expected or not torch.equal(value, expected[key])
        for key, value in weights.items()
    ):
        raise TrainingIntegrityError(
            "fresh initial tensors differ from inherited scalar initializer"
        )
    head = model.policy.action_net
    if (
        not torch.equal(head.weight, torch.zeros_like(head.weight))
        or not torch.equal(head.bias, torch.zeros_like(head.bias))
        or not torch.equal(
            model.policy.log_std,
            torch.full_like(model.policy.log_std, math.log(arm.initial_std)),
        )
    ):
        raise TrainingIntegrityError(
            "fresh actor does not have exact MPC initialization"
        )
    return observed


def _reference_initial_weights() -> dict[str, Any]:
    for value in required_initial_artifacts().values():
        _checked_record(value)
    arm = source.ARMS[0]
    return source.read_initial_policy(
        INITIAL_ROOT / "training" / arm.name, source.template(INITIAL_ROOT), arm
    )


def verify_initializers() -> dict[str, Any]:
    """Reconstruct fresh models and compare tensors without reset or inference."""
    expected = _reference_initial_weights()
    values = {}
    for arm in ARMS:
        model, _, vector = source.make_learner(
            cell_config(arm, ROOT), arm, seed_base=TRAIN_SEED_BASE
        )
        try:
            values[arm.name] = {
                "gamma": GAMMAS[arm.name],
                "all_tensors": _assert_initial_policy(model, arm, expected),
                "policy_identity": model.research_identity,
            }
        finally:
            vector.close()
    if values[ARMS[0].name]["all_tensors"] != values[ARMS[1].name]["all_tensors"]:
        raise TrainingIntegrityError("paired fresh initializers differ")
    return {
        "schema": "plant-reward-alignment-initialization-admission-v1",
        "verified": True,
        "initial_reference": required_initial_artifacts(),
        "arms": values,
        "matched": True,
        "simulator_resets_executed": False,
        "policy_inference_or_learning_executed": False,
    }


def verify_initial_matching(root: Path, admission: dict[str, Any]) -> dict[str, Any]:
    import torch

    expected = _reference_initial_weights()
    reached, tensors = [], {}
    for arm in ARMS:
        directory = root / "training" / arm.name
        weights = source.read_initial_policy(directory, cell_config(arm, root), arm)
        values = source.tensor_records(weights, shared_only=False)
        if (
            values != admission["arms"][arm.name]["all_tensors"]
            or values != source.tensor_records(expected, shared_only=False)
            or any(
                key not in expected or not torch.equal(value, expected[key])
                for key, value in weights.items()
            )
        ):
            raise TrainingIntegrityError(
                "published initial policy differs from pre-simulation admission"
            )
        tensors[arm.name] = values
        reached.append(
            {
                "arm": arm.name,
                "record": file_record(directory / "initial-policy-record.json"),
            }
        )
    if tensors[ARMS[0].name] != tensors[ARMS[1].name]:
        raise TrainingIntegrityError("published paired initial tensors differ")
    return {
        "schema": "plant-reward-alignment-initial-matching-v1",
        "expected_arms": list(ACTORS),
        "initial_policy_records": reached,
        "matched": True,
        "all_tensors": tensors[ARMS[0].name],
        "inherited_initial_reference": required_initial_artifacts(),
        "new_initial_physical_evaluations": 0,
    }


def verify_endpoint_model(model: Any, arm: Arm) -> None:
    config = cell_config(arm, ROOT)
    state = model.embedded_constraint_state
    payload = state["payload"]
    if (
        model.algorithm_id != SCALAR
        or model.num_timesteps != TOTAL_STEPS
        or type(model.gamma) not in (int, float)
        or model.gamma != GAMMAS[arm.name]
        or model.gae_lambda != config.ppo.gae_lambda
        or payload["ppo_configuration"]["gamma"] != GAMMAS[arm.name]
        or payload["ppo_configuration"]["gae_lambda"] != config.ppo.gae_lambda
        or payload["num_timesteps"] != TOTAL_STEPS
        or payload["pending_rollout"] is not False
        or payload["aborted_rollout"] is not None
        or payload["unledgered_transition_count"] != 0
        or payload["resume_continuation_receipt"] is not None
        or payload["ppo_epoch_update_count"] != 20
        or len(payload["rollout_ledgers"]) != 2
        or any(
            row["transition_count"] != 8192 or row["optimizer_steps"] != 320
            for row in payload["rollout_ledgers"]
        )
    ):
        raise TrainingIntegrityError(
            "fixed endpoint gamma, optimizer or transition budget differs"
        )
    source.verify_model_spaces(model, config, arm)


def train_cell(arm: Arm) -> dict[str, Any]:
    plan = require_plan()
    # The unchanged producer independently reconstructs this same seed101 model.
    # Its published initial tensors are subsequently cross-bound to this admission.
    expected = _reference_initial_weights()
    model, _, vector = source.make_learner(
        cell_config(arm, ROOT), arm, seed_base=TRAIN_SEED_BASE
    )
    try:
        observed = _assert_initial_policy(model, arm, expected)
        if (
            observed
            != plan["initialization_admission"]["arms"][arm.name]["all_tensors"]
        ):
            raise TrainingIntegrityError(
                "trainer initializer differs from frozen admission"
            )
    finally:
        vector.close()
    require_deadline(plan["deadline_admission"])
    directory = ROOT / "training" / arm.name
    result = source.train_cell(
        directory, cell_config(arm, ROOT), arm, seed_base=TRAIN_SEED_BASE
    )
    verify_endpoint_model(
        source.load_final(directory, cell_config(arm, ROOT), arm), arm
    )
    return result


def validate_receipt_closure(receipt: dict[str, Any], extra: list[str]) -> None:
    process = receipt.get("process", {})
    codes = receipt.get("codes")
    if process.get("exit_code") == INTEGRITY_EXIT or (
        isinstance(codes, list) and "root_custody_lost" in codes
    ):
        raise TrainingIntegrityError("receipt records evidence integrity loss")
    cap = TRAIN_TIMEOUT if receipt.get("stage") == "_train" else EVAL_TIMEOUT
    if (
        not isinstance(codes, list)
        or any(code not in source.ORDINARY_CHILD_CODES for code in codes)
        or not receipt.get("closed_at")
        or process.get("command")
        != [sys.executable, "-m", MODULE, receipt.get("stage"), *extra]
        or type(process.get("timeout_seconds")) is not int
        or not 0 < process["timeout_seconds"] <= cap
    ):
        raise TrainingIntegrityError("receipt command, timeout or closure differs")


def require_successful_training(arm: Arm) -> None:
    name = f"train-{arm.name}"
    receipt = json.loads(read_regular(ROOT / "receipts" / f"{name}.json"))
    if receipt.get("stage") != "_train" or receipt.get("name") != name:
        raise TrainingIntegrityError("training receipt slot differs")
    validate_receipt_closure(receipt, ["--cell", arm.name])
    if receipt["codes"]:
        raise TrainingUnavailable("fixed endpoint has no successfully closed training")
    if (
        receipt["process"].get("started") is not True
        or receipt["process"].get("exit_code") != 0
    ):
        raise TrainingIntegrityError("training receipt hides an unsuccessful process")


def evaluate_cell(actor: str, seed: int) -> None:
    from .research_mpc_residual_evaluation import evaluate_residual_episode

    arm = next(arm for arm in ARMS if arm.name == actor)
    require_successful_training(arm)
    directory = ROOT / "evaluations" / f"{seed}-{actor}"
    publish_json(
        directory / "attempt.json", {"actor": actor, "seed": seed, "started_at": now()}
    )
    config = cell_config(arm, ROOT)
    policy = source.load_final(ROOT / "training" / actor, config, arm)
    verify_endpoint_model(policy, arm)
    episode = evaluate_residual_episode(
        config, policy=policy, seed=seed, policy_id=actor
    )
    encoded = gzip.compress(canonical_bytes(episode), compresslevel=6, mtime=0)
    if len(encoded) > ARCHIVE_LIMIT:
        raise TrainingIntegrityError("evaluation archive exceeds stored-size cap")
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


def require_successful_evaluation(actor: str, seed: int) -> None:
    name = f"evaluate-{seed}-{actor}"
    receipt = json.loads(read_regular(ROOT / "receipts" / f"{name}.json"))
    if receipt.get("stage") != "_evaluate" or receipt.get("name") != name:
        raise TrainingIntegrityError("evaluation receipt slot differs")
    validate_receipt_closure(receipt, ["--actor", actor, "--seed", str(seed)])
    if receipt["codes"]:
        raise TrainingUnavailable("replay has no successfully closed evaluator")
    if (
        receipt["process"].get("started") is not True
        or receipt["process"].get("exit_code") != 0
    ):
        raise TrainingIntegrityError("evaluation receipt hides an unsuccessful process")


def replay_cell(actor: str, seed: int) -> None:
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
    if (
        file_record(archive) != summary["archive"]
        or summary["archive"]["bytes"] > ARCHIVE_LIMIT
    ):
        raise TrainingIntegrityError("replay archive identity or size differs")
    episode = json.loads(gzip.decompress(read_regular(archive)))
    if (
        summary.get("actor") != actor
        or summary.get("seed") != seed
        or episode.get("seed") != seed
        or canonical_bytes(summary.get("summary"))
        != canonical_bytes(episode.get("summary"))
        or episode["summary"].get("policy_id") != actor
        or episode["summary"].get("actor") != residual_actor
    ):
        raise TrainingIntegrityError("replay slot identity differs")
    publish_json(
        directory / "replay.json",
        verify_residual_episode_replay(cell_config(actor, ROOT), episode),
    )


def closed_child(
    stage: str, name: str, timeout: int, extra: list[str]
) -> dict[str, Any]:
    if stage not in ("_train", "_evaluate", "_replay"):
        raise ValueError("unknown reward-alignment stage")
    source.configure_threads()
    plan = require_plan()
    cap = TRAIN_TIMEOUT if stage == "_train" else EVAL_TIMEOUT
    if type(timeout) is not int or not 0 < timeout <= cap:
        raise TrainingIntegrityError("child timeout exceeds frozen stage bound")
    if stage == "_train":
        wave = validate_deadline_check(
            json.loads(read_regular(ROOT / "deadlines" / "training-wave.json")),
            plan["deadline_admission"],
        )
        if (
            wave["stage"] != "_train_wave"
            or wave["name"] != "training-wave"
            or wave["effective_timeout_seconds"] != timeout
        ):
            raise TrainingIntegrityError(
                "training timeout differs from registered wave"
            )
    elif timeout != EVAL_TIMEOUT:
        raise TrainingIntegrityError("serial stage must request the frozen maximum")
    timing = deadline_check(
        plan["deadline_admission"],
        stage=stage,
        name=name,
        cap=cap,
        requested=timeout if stage == "_train" else None,
    )
    publish_json(ROOT / "deadlines" / f"{name}.json", timing)
    process, codes = capture_process(
        [sys.executable, "-m", MODULE, stage, *extra],
        stdout=ROOT / "logs" / f"{name}.stdout.log",
        stderr=ROOT / "logs" / f"{name}.stderr.log",
        timeout_seconds=timing["effective_timeout_seconds"],
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
        raise TrainingIntegrityError("child integrity failure; batch aborted")
    require_plan()
    return receipt


def training_wave() -> dict[str, Any]:
    from .research_parallel_supervision import ParallelAttempt, run_training_wave

    plan = require_plan()
    timing = deadline_check(
        plan["deadline_admission"],
        stage="_train_wave",
        name="training-wave",
        cap=TRAIN_TIMEOUT,
    )
    publish_json(ROOT / "deadlines" / "training-wave.json", timing)
    timeout = timing["effective_timeout_seconds"]
    attempts = []
    for arm in ARMS:
        name, extra = f"train-{arm.name}", ("--cell", arm.name)
        attempts.append(
            ParallelAttempt(
                name=name,
                stage="_train",
                timeout_seconds=timeout,
                extra=extra,
                command=(sys.executable, "-m", MODULE, "_train", *extra),
                stdout=ROOT / "logs" / f"{name}.stdout.log",
                stderr=ROOT / "logs" / f"{name}.stderr.log",
                receipt=ROOT / "receipts" / f"{name}.json",
            )
        )
    return run_training_wave(
        attempts,
        supervisor_directory=ROOT / "supervisors",
        worker_module=MODULE,
        worker_function="closed_child",
        check_integrity=require_plan,
        minimum_free_bytes=MINIMUM_FREE,
        supervisor_log_limit_bytes=LOG_LIMIT,
        child_log_limit_bytes=LOG_LIMIT,
        integrity_check_interval_seconds=INTEGRITY_CHECK_INTERVAL,
    )


def _reached_receipts() -> list[dict[str, Any]]:
    names = [f"train-{arm.name}" for arm in ARMS] + [
        f"{stage}-{seed}-{actor}"
        for seed in DEV_SEEDS
        for actor in ACTORS
        for stage in ("evaluate", "replay")
    ]
    return [
        json.loads(read_regular(ROOT / "receipts" / f"{name}.json"))
        for name in names
        if os.path.lexists(ROOT / "receipts" / f"{name}.json")
    ]


def run(gate: Path, *, authorized: bool) -> dict[str, Any]:
    from .capacity_benchmark import _host_identity

    if not authorized:
        raise ValueError("reward-alignment run requires research authorization")
    deadline = capture_deadline_admission()
    if os.path.lexists(ROOT) or ROOT != ROOT.resolve():
        raise ValueError("reward-alignment root must be absent and canonical; no retry")
    runtime = capture(require_clean=True)
    raw = read_regular(gate)
    checked = json.loads(raw)
    if (
        checked.get("passed") is not True
        or checked.get("source_commit") != runtime.git_head
        or checked.get("source_tree") != _current_tree()
        or checked.get("source_clean_and_unchanged") is not True
        or canonical_bytes(checked.get("power_policy"))
        != canonical_bytes(power_policy())
        or canonical_bytes(checked.get("deadline_policy"))
        != canonical_bytes(deadline_policy())
    ):
        raise ValueError("source gate or declared policies differ")
    storage = capture_storage_identity(gate)
    require_storage(gate, checked.get("storage"))
    reference = checked["reference_admission"]
    require_reference(Path(reference["path"]), reference, runtime=runtime)
    initialization = verify_initializers()
    gate_record = file_record(gate)
    if gate_record["sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("source gate changed during admission")
    plan = {
        "specification": specification(ROOT),
        "runtime": runtime.to_dict(),
        "host": _host_identity(),
        "gate": gate_record,
        "storage": storage,
        "reference_admission": reference,
        "deadline_admission": deadline,
        "initialization_admission": initialization,
        "started_at": now(),
        "power_admission": require_power(failure_directory=gate.parent, admission=True),
    }
    require_deadline(deadline, admission=True)
    mkdir_exclusive(ROOT)
    publish_json(ROOT / "plan.json", plan)
    receipts, failure = [], None
    try:
        for name in (
            "logs",
            "receipts",
            "training",
            "evaluations",
            "supervisors",
            "deadlines",
        ):
            mkdir_exclusive(ROOT / name)
        for arm in ARMS:
            mkdir_exclusive(ROOT / "training" / arm.name)
        wave = training_wave()
        publish_json(ROOT / "training-wave.json", wave)
        receipts.extend(wave["receipts"])
        if wave["fatal"] is not None:
            raise TrainingIntegrityError(f"training custody failed: {wave['fatal']}")
        for arm in ARMS:
            require_successful_training(arm)
        require_plan()
        publish_json(
            ROOT / "initial-matching.json",
            verify_initial_matching(ROOT, initialization),
        )
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
        require_plan()
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        try:
            receipts = _reached_receipts()
        except Exception as collection:
            failure["receipt_collection_failure"] = (
                f"{type(collection).__name__}: {collection}"
            )
    report = {
        "schema": SCHEMA,
        "status": "aborted" if failure else "closed",
        "closed_at": now(),
        "failure": failure,
        "plan": file_record(ROOT / "plan.json"),
        "reference_admission": reference,
        "receipts": receipts,
        "planned_slots": {
            "training": list(ACTORS),
            "evaluation": [f"{seed}-{actor}" for seed in DEV_SEEDS for actor in ACTORS],
            "replay": [f"{seed}-{actor}" for seed in DEV_SEEDS for actor in ACTORS],
        },
        "claim": "One-seed reward-discount development diagnostic; no selected policy or confirmation claim.",
        "registered_evaluation_eligible": False,
    }
    publish_json(ROOT / "report.json", report)
    return report


def execute_child(action: Callable[[], Any], *, artifact_phase: bool) -> int:
    def failed(exc: Exception) -> int:
        print(
            json.dumps({"integrity_failure": type(exc).__name__, "message": str(exc)}),
            file=sys.stderr,
        )
        return INTEGRITY_EXIT

    try:
        require_plan()
    except Exception as exc:
        return failed(exc)
    try:
        action()
    except (
        TrainingIntegrityError,
        EpisodeSeedCustodyError,
        MPCResidualIntegrityError,
    ) as exc:
        return failed(exc)
    except TrainingUnavailable:
        raise
    except Exception as exc:
        if not artifact_phase:
            raise
        return failed(exc)
    try:
        require_plan()
    except Exception as exc:
        return failed(exc)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "_train", "_evaluate", "_replay"))
    parser.add_argument("--assert-authorized", action="store_true")
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--cell", choices=ACTORS)
    parser.add_argument("--actor", choices=ACTORS)
    parser.add_argument("--seed", type=int, choices=DEV_SEEDS)
    args = parser.parse_args(argv)
    source.configure_threads()
    if args.command == "run":
        if (
            args.gate is None
            or args.cell is not None
            or args.actor is not None
            or args.seed is not None
        ):
            parser.error("run requires only --gate and authorization assertion")
        result = run(args.gate, authorized=args.assert_authorized)
        print(
            json.dumps(
                {"status": result["status"], "report": str(ROOT / "report.json")}
            )
        )
        return 1 if result["failure"] else 0
    if args.command == "_train":
        if args.cell is None or args.actor is not None or args.seed is not None:
            parser.error("training requires exactly --cell")
        arm = next(arm for arm in ARMS if arm.name == args.cell)
        return execute_child(lambda: train_cell(arm), artifact_phase=False)
    if args.actor is None or args.seed is None or args.cell is not None:
        parser.error("evaluation/replay requires exactly --actor and --seed")
    return execute_child(
        lambda: (evaluate_cell if args.command == "_evaluate" else replay_cell)(
            args.actor, args.seed
        ),
        artifact_phase=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())

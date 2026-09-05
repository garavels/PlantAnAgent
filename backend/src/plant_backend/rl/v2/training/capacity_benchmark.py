"""Fixture-only capacity measurement for the frozen constrained-RL campaign.

This module measures infrastructure, never controller quality.  It runs exactly
one registered-shaped fixture rollout for each frozen learner family in fresh,
sequential child processes.  Static reward, cost, tariff, and market parameters
remain bound as task identity; the report contains no *observed* reward, cost,
action, multiplier, loss, market, outage, inventory, or policy-output value.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path, PurePosixPath
import platform
import resource
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, NoReturn

from ... import checkpoints
from ...policies import load_policy
from ...training.config import PPOConfig
from .. import beta_spaces
from ..beta_config import CyclicMarketTaskConfigV1
from ..mpc_schema import canonical_json, domain_sha256
from ..policy_setpoint_controller import policy_contract_for_environment
from .beta_checkpoint import (
    parse_beta_checkpoint_evidence,
)
from .config import (
    BETA_FIXTURE_RUN_KIND,
    BETA_PI_LAGRANGIAN_ALGORITHM_ID,
    BETA_I_LAGRANGIAN_ALGORITHM_ID,
    BETA_REGISTERED_TOTAL_TIMESTEPS,
    BETA_SCALAR_PPO_ALGORITHM_ID,
    BetaRLTrainingRunV1,
    ControllerNeutralTrainingConfigV1,
)
from .ppo import BETA_CHECKPOINT_EXTRA_KEY, CONFIG_FILE, make_env
from .runtime_provenance import (
    BetaTrainingRuntimeProvenanceV1,
    capture as capture_runtime_provenance,
    verify_unchanged as verify_runtime_unchanged,
)


CAPACITY_PROTOCOL_ID = "plant-beta-constrained-rl-capacity-v1"
CAPACITY_PROTOCOL_PATH = "docs/research/CONSTRAINED_RL_CAPACITY_PROTOCOL.md"
CAPACITY_PROTOCOL_SHA256 = (
    "223947ccd8fb0b8c909ea89ee4019818081f4bf2f884301019450551f7b18b88"
)
CAPACITY_REPAIR_ID = "plant-beta-constrained-rl-capacity-repair-001"
CAPACITY_REPAIR_PATH = "docs/research/CONSTRAINED_RL_CAPACITY_REPAIR_001.md"
CAPACITY_REPAIR_SHA256 = (
    "55a14887379c48232abc75913950a02de83db3cd5e9144f9ecf323a0a26d5c53"
)
CAPACITY_REPAIR_CLARIFICATION_ID = (
    "plant-beta-constrained-rl-capacity-repair-clarification-001"
)
CAPACITY_REPAIR_CLARIFICATION_PATH = (
    "docs/research/CONSTRAINED_RL_CAPACITY_REPAIR_CLARIFICATION_001.md"
)
CAPACITY_REPAIR_CLARIFICATION_SHA256 = (
    "2175c56d85ceb2b363b85316a9e90d437f1249d00a869d86e99676525e22beda"
)
CAPACITY_REPAIR_CLARIFICATION_002_ID = (
    "plant-beta-constrained-rl-capacity-repair-clarification-002"
)
CAPACITY_REPAIR_CLARIFICATION_002_PATH = (
    "docs/research/CONSTRAINED_RL_CAPACITY_REPAIR_CLARIFICATION_002.md"
)
CAPACITY_REPAIR_CLARIFICATION_002_SHA256 = (
    "f4db393d54d48aa5deca69b5b1c220bccd94b5fd87bac4fc6b4b40416feca8ea"
)
CAPACITY_REPORT_SCHEMA = "plant-beta-constrained-rl-capacity-report-v2"
CAPACITY_REPORT_DOMAIN = CAPACITY_REPORT_SCHEMA
CAPACITY_DETERMINISTIC_DOMAIN = (
    "plant-beta-constrained-rl-capacity-deterministic-evidence-v2"
)
CAPACITY_SYNTHETIC_REPORT_SCHEMA = (
    "plant-beta-constrained-rl-capacity-synthetic-report-v1"
)
CAPACITY_SYNTHETIC_REPORT_DOMAIN = CAPACITY_SYNTHETIC_REPORT_SCHEMA
CAPACITY_SYNTHETIC_DETERMINISTIC_DOMAIN = (
    "plant-beta-constrained-rl-capacity-synthetic-deterministic-evidence-v1"
)
CAPACITY_CLAIM = "fixture-capacity-only; no controller-performance evidence"
CAPACITY_SYNTHETIC_CLAIM = (
    "synthetic-test-only; no capacity or controller-performance evidence"
)
CAPACITY_RESULT_CLAIM = (
    "On the recorded host and clean source revision, three fixture-domain, "
    "registered-shaped first rollouts imply the reported sequential capacity "
    "scenarios. No registered seed or controller-performance outcome was opened."
)

CAPACITY_ALGORITHM_ORDER = (
    BETA_PI_LAGRANGIAN_ALGORITHM_ID,
    BETA_I_LAGRANGIAN_ALGORITHM_ID,
    BETA_SCALAR_PPO_ALGORITHM_ID,
)
CAPACITY_ALGORITHM_SLUGS = {
    BETA_PI_LAGRANGIAN_ALGORITHM_ID: "pi-lagrangian",
    BETA_I_LAGRANGIAN_ALGORITHM_ID: "i-lagrangian",
    BETA_SCALAR_PPO_ALGORITHM_ID: "scalar-ppo",
}
CAPACITY_OPTIMIZER_SEED = 101
CAPACITY_FIXTURE_EPISODE_SEED_BASE = 9_880_000
CAPACITY_FIXTURE_LAST_EPISODE_SEED = 9_880_015
CAPACITY_TRANSITIONS = 8_192
CAPACITY_PHYSICAL_STEPS_PER_TRANSITION = 15
CAPACITY_PHYSICAL_MINUTE_STEPS = (
    CAPACITY_TRANSITIONS * CAPACITY_PHYSICAL_STEPS_PER_TRANSITION
)
CAPACITY_ROLLOUT_COUNT = 1
CAPACITY_OPTIMIZER_STEPS = 320
CAPACITY_ALLOCATOR_ENTRIES = 16
REGISTERED_ROLLOUTS_PER_POLICY = 128
REGISTERED_POLICIES_PER_FAMILY = 5
REGISTERED_POLICY_COUNT = 15
REGISTERED_MINIBATCH_STEPS_PER_POLICY = 40_960
REGISTERED_ALLOCATOR_ENTRIES_PER_POLICY = 1_564
REGISTERED_PHYSICAL_MINUTE_STEPS = (
    REGISTERED_POLICY_COUNT
    * BETA_REGISTERED_TOTAL_TIMESTEPS
    * CAPACITY_PHYSICAL_STEPS_PER_TRANSITION
)
REGISTERED_CAMPAIGN_ARITHMETIC = {
    "policies": REGISTERED_POLICY_COUNT,
    "controller_transitions": (
        REGISTERED_POLICY_COUNT * BETA_REGISTERED_TOTAL_TIMESTEPS
    ),
    "rollouts": REGISTERED_POLICY_COUNT * REGISTERED_ROLLOUTS_PER_POLICY,
    "minibatch_optimizer_steps": (
        REGISTERED_POLICY_COUNT * REGISTERED_MINIBATCH_STEPS_PER_POLICY
    ),
    "reset_manifests": (
        REGISTERED_POLICY_COUNT * REGISTERED_ALLOCATOR_ENTRIES_PER_POLICY
    ),
    "physical_minute_steps": REGISTERED_PHYSICAL_MINUTE_STEPS,
}
CAPACITY_CHILD_TIMEOUT_S = 3_600
CAPACITY_REPORT_FILENAME = "report.json"
CAPACITY_SYNTHETIC_REPORT_FILENAME = "synthetic-report.json"
CAPACITY_SYNTHETIC_EXECUTION_CLASS = "synthetic-test-only"
CAPACITY_CHECKPOINT_FILES = (
    checkpoints.POLICY_FILE,
    checkpoints.METADATA_FILE,
    CONFIG_FILE,
)
CAPACITY_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
CAPACITY_MAX_THREAD_COUNT = 2_147_483_647

_REPORT_FIELDS = frozenset(
    {
        "schema",
        "protocol",
        "claim_boundary",
        "result_claim",
        "status",
        "operator_assertions",
        "runtime_provenance",
        "host",
        "cell_contract",
        "cells",
        "campaign_arithmetic",
        "campaign_projection",
        "deterministic_evidence_sha256",
        "sha256",
    }
)
_SYNTHETIC_REPORT_FIELDS = _REPORT_FIELDS | frozenset({"execution_class"})
_CELL_FIELDS = frozenset(
    {
        "order",
        "algorithm",
        "optimizer_seed",
        "fixture_episode_seed_base",
        "config_path",
        "config_sha256",
        "output_path",
        "status",
        "exit_code",
        "wall_elapsed_ns",
        "child_user_cpu_ns",
        "child_system_cpu_ns",
        "stdout",
        "stderr",
        "checkpoint",
        "measurement",
        "projection",
        "failure",
    }
)
_LOG_FIELDS = frozenset({"path", "logical_bytes", "sha256"})
_CHECKPOINT_FIELDS = frozenset(
    {
        "path",
        "metadata_sha256",
        "checkpoint_evidence_sha256",
        "runtime_provenance_sha256",
        "environment_contract_sha256",
        "checkpoint_contract_sha256",
        "files",
        "logical_bytes",
        "allocated_bytes",
        "rollout_entry_count",
        "allocator_entry_count",
        "max_rollout_entry_bytes",
        "max_allocator_entry_bytes",
    }
)
_FILE_FIELDS = frozenset({"path", "logical_bytes", "allocated_bytes", "sha256"})
_MEASUREMENT_FIELDS = frozenset(
    {
        "controller_transitions",
        "physical_minute_steps",
        "rollout_elapsed_ns",
        "fixed_overhead_ns",
        "controller_transitions_per_second",
        "physical_minute_steps_per_second",
        "optimizer_steps",
    }
)
_PROJECTION_FIELDS = frozenset(
    {
        "registered_policy_wall_ns_by_factor",
        "five_policy_family_wall_ns_by_factor",
        "structural_growth_allowance_bytes",
        "projected_checkpoint_logical_bytes",
        "five_policy_family_logical_bytes",
    }
)
_HOST_FIELDS = frozenset(
    {
        "platform",
        "system",
        "machine",
        "processor",
        "cpu_brand",
        "physical_cpu_count",
        "logical_cpu_count",
        "physical_memory_bytes",
        "torch_intraop_threads",
        "torch_interop_threads",
        "thread_environment",
        "clock",
    }
)
_CLOCK_FIELDS = frozenset({"name", "monotonic", "adjustable", "resolution_s"})
_OPERATOR_ASSERTION_FIELDS = frozenset(
    {
        "cells_executed_sequentially",
        "no_other_repository_benchmark_intentionally_running",
        "attempts_not_retried_or_selected",
    }
)
_FAILURE_FIELDS = frozenset({"phase", "category", "message"})
_CAMPAIGN_PROJECTION_FIELDS = frozenset(
    {
        "sequential_wall_ns_by_factor",
        "projected_checkpoint_logical_bytes",
        "recommended_free_space_bytes",
        "duration_scenarios_are_not_confidence_intervals",
        "storage_is_a_structural_planning_allowance_not_zip_prediction",
    }
)

_FAILURE_CONTRACT = {
    "training_timeout": (
        "training_process",
        "training process timed out; retained output is identified only by digest",
    ),
    "training_nonzero_exit": (
        "training_process",
        "training process failed; retained output is identified only by digest",
    ),
    "training_launch_error": (
        "training_process",
        "training process could not be launched; no exception detail is reported",
    ),
    "checkpoint_admission_error": (
        "checkpoint_admission",
        "checkpoint admission failed; no controller outcome is reported",
    ),
}


class CapacityBenchmarkError(RuntimeError):
    """The capacity measurement or its report failed closed."""


def _repository_root() -> Path:
    repository = Path(__file__).resolve().parents[6]
    if not (repository / ".git").exists() or not (repository / "uv.lock").is_file():
        raise CapacityBenchmarkError(
            "capacity benchmark requires the repository checkout"
        )
    return repository


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed JSON object")
    return value


def _sequence(name: str, value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a JSON array")
    return value


def _exact_keys(name: str, value: Mapping[str, Any], expected: frozenset[str]) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise CapacityBenchmarkError(
            f"{name} fields differ; missing={sorted(missing)}, extra={sorted(unknown)}"
        )


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _signed_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _finite(name: str, value: object, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum:g}")
    return result


def _sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _relative_path(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.as_posix() != value
    ):
        raise ValueError(f"{name} must be a canonical relative path")
    return value


def _failure(category: str) -> dict[str, str]:
    try:
        phase, message = _FAILURE_CONTRACT[category]
    except KeyError as reason:
        raise ValueError("capacity failure category is not recognized") from reason
    return {
        "phase": phase,
        "category": category,
        "message": message,
    }


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number {value!r} is forbidden")
    return parsed


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular_bytes(path: Path) -> bytes:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CapacityBenchmarkError(
            f"artifact must be a regular nonsymlink file: {path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if _stat_identity(opened) != _stat_identity(before):
            raise CapacityBenchmarkError(f"artifact changed while opening: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if _stat_identity(after) != _stat_identity(before):
        raise CapacityBenchmarkError(f"artifact changed while reading: {path}")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise CapacityBenchmarkError(
            f"artifact byte count changed while reading: {path}"
        )
    return raw


def _load_canonical_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} is forbidden")
            result[key] = value
        return result

    raw = _read_regular_bytes(path)
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as reason:
        raise ValueError(f"cannot decode canonical JSON {path}: {reason}") from reason
    payload = dict(_mapping(str(path), decoded))
    expected = (canonical_json(payload) + "\n").encode("utf-8")
    if raw != expected:
        raise ValueError(f"JSON is not canonical with one trailing newline: {path}")
    return payload


def _publish_bytes_exclusive(target: Path, raw: bytes) -> str:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    if _read_regular_bytes(target) != raw:
        raise CapacityBenchmarkError(f"published bytes differ at {target}")
    return hashlib.sha256(raw).hexdigest()


def _publish_json_exclusive(target: Path, payload: Mapping[str, Any]) -> str:
    return _publish_bytes_exclusive(
        target,
        (canonical_json(dict(payload)) + "\n").encode("utf-8"),
    )


def _protocol_identity(repository: Path) -> dict[str, Any]:
    protocol = repository / CAPACITY_PROTOCOL_PATH
    digest = hashlib.sha256(_read_regular_bytes(protocol)).hexdigest()
    if digest != CAPACITY_PROTOCOL_SHA256:
        raise CapacityBenchmarkError(
            "capacity protocol content differs from frozen SHA-256"
        )
    repair = repository / CAPACITY_REPAIR_PATH
    repair_digest = hashlib.sha256(_read_regular_bytes(repair)).hexdigest()
    if repair_digest != CAPACITY_REPAIR_SHA256:
        raise CapacityBenchmarkError(
            "capacity repair protocol content differs from frozen SHA-256"
        )
    clarification = repository / CAPACITY_REPAIR_CLARIFICATION_PATH
    clarification_digest = hashlib.sha256(
        _read_regular_bytes(clarification)
    ).hexdigest()
    if clarification_digest != CAPACITY_REPAIR_CLARIFICATION_SHA256:
        raise CapacityBenchmarkError(
            "capacity repair clarification content differs from frozen SHA-256"
        )
    clarification_002 = repository / CAPACITY_REPAIR_CLARIFICATION_002_PATH
    clarification_002_digest = hashlib.sha256(
        _read_regular_bytes(clarification_002)
    ).hexdigest()
    if clarification_002_digest != CAPACITY_REPAIR_CLARIFICATION_002_SHA256:
        raise CapacityBenchmarkError(
            "capacity repair clarification 002 content differs from frozen SHA-256"
        )
    return {
        "id": CAPACITY_PROTOCOL_ID,
        "path": CAPACITY_PROTOCOL_PATH,
        "raw_file_sha256": digest,
        "repairs": [
            {
                "id": CAPACITY_REPAIR_ID,
                "path": CAPACITY_REPAIR_PATH,
                "raw_file_sha256": repair_digest,
                "clarifications": [
                    {
                        "id": CAPACITY_REPAIR_CLARIFICATION_ID,
                        "path": CAPACITY_REPAIR_CLARIFICATION_PATH,
                        "raw_file_sha256": clarification_digest,
                    },
                    {
                        "id": CAPACITY_REPAIR_CLARIFICATION_002_ID,
                        "path": CAPACITY_REPAIR_CLARIFICATION_002_PATH,
                        "raw_file_sha256": clarification_002_digest,
                    },
                ],
            }
        ],
    }


def _cell_config(output: Path, algorithm: str) -> ControllerNeutralTrainingConfigV1:
    if algorithm not in CAPACITY_ALGORITHM_ORDER:
        raise ValueError("capacity algorithm is outside the frozen three-cell order")
    return ControllerNeutralTrainingConfigV1(
        environment_id=beta_spaces.ENVIRONMENT_ID,
        seed=CAPACITY_OPTIMIZER_SEED,
        total_timesteps=CAPACITY_TRANSITIONS,
        n_envs=4,
        checkpoint_every=0,
        output_dir=str(output),
        resume_from=None,
        beta_task=CyclicMarketTaskConfigV1(),
        beta_rl=BetaRLTrainingRunV1(
            algorithm=algorithm,
            run_kind=BETA_FIXTURE_RUN_KIND,
            episode_seed_base=CAPACITY_FIXTURE_EPISODE_SEED_BASE,
        ),
        ppo=PPOConfig(),
        notes=(
            "fixture-only registered-shaped capacity measurement; no controller outcome"
        ),
    ).validate()


def _training_config_bytes(config: ControllerNeutralTrainingConfigV1) -> bytes:
    return (
        json.dumps(config.to_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _cell_layout(order: int, algorithm: str) -> dict[str, str]:
    normalized_order = _integer("capacity cell order", order, minimum=1)
    try:
        slug = CAPACITY_ALGORITHM_SLUGS[algorithm]
    except KeyError as reason:
        raise ValueError(
            "capacity algorithm is outside the frozen three-cell order"
        ) from reason
    prefix = f"{normalized_order:02d}-{slug}"
    output = f"artifacts/{slug}"
    return {
        "config": f"configs/{prefix}.json",
        "output": output,
        "checkpoint": f"{output}/checkpoint",
        "stdout": f"logs/{prefix}.stdout.log",
        "stderr": f"logs/{prefix}.stderr.log",
    }


def _cell_contract() -> dict[str, Any]:
    template = _cell_config(
        Path("/private/tmp/plantanagent-capacity-contract-placeholder"),
        CAPACITY_ALGORITHM_ORDER[0],
    )
    config = template.to_dict()
    config["output_dir"] = "<external-cell-output>"
    beta_run = dict(_mapping("beta run", config["beta_rl"]))
    beta_run["algorithm"] = "<frozen-cell-algorithm>"
    config["beta_rl"] = beta_run
    contract = {
        "algorithms": list(CAPACITY_ALGORITHM_ORDER),
        "optimizer_seed": CAPACITY_OPTIMIZER_SEED,
        "fixture_episode_seed_base": CAPACITY_FIXTURE_EPISODE_SEED_BASE,
        "fixture_last_episode_seed": CAPACITY_FIXTURE_LAST_EPISODE_SEED,
        "requested_and_collected_transitions": CAPACITY_TRANSITIONS,
        "physical_minute_steps": CAPACITY_PHYSICAL_MINUTE_STEPS,
        "rollouts": CAPACITY_ROLLOUT_COUNT,
        "optimizer_steps": CAPACITY_OPTIMIZER_STEPS,
        "allocator_entries": CAPACITY_ALLOCATOR_ENTRIES,
        "child_timeout_s": CAPACITY_CHILD_TIMEOUT_S,
        "registered_rollouts_per_policy": REGISTERED_ROLLOUTS_PER_POLICY,
        "registered_policies_per_family": REGISTERED_POLICIES_PER_FAMILY,
        "sensitivity_ratios": [
            {
                "label": _factor_label(numerator, denominator),
                "numerator": numerator,
                "denominator": denominator,
            }
            for numerator, denominator in _SENSITIVITY_RATIOS
        ],
        "duration_projection": {
            "registered_policy": (
                "fixed_overhead_ns + ceil(factor * 128 * rollout_elapsed_ns)"
            ),
            "family": "5 * registered_policy_wall_ns",
            "campaign": "sum(three five-policy family wall times)",
            "sensitivity_applies_to": "rollout term only",
        },
        "measurement_definitions": {
            "wall_elapsed_ns": (
                "parent perf_counter_ns around complete production training child"
            ),
            "rollout_elapsed_ns": (
                "checkpoint rollout ledger collection-plus-optimization elapsed_compute_s"
            ),
            "fixed_overhead_ns": "max(0, wall_elapsed_ns - rollout_elapsed_ns)",
            "throughput_denominator": "rollout_elapsed_ns",
        },
        "storage_projection": {
            "additional_rollout_entries": (
                REGISTERED_ROLLOUTS_PER_POLICY - CAPACITY_ROLLOUT_COUNT
            ),
            "rollout_entry_copies": 3,
            "additional_allocator_entries": (
                REGISTERED_ALLOCATOR_ENTRIES_PER_POLICY - CAPACITY_ALLOCATOR_ENTRIES
            ),
            "allocator_entry_copies": 4,
            "growth_safety_multiplier": 2,
            "recommended_free_space": (
                "max(2 * projected_total, projected_total + 1 GiB)"
            ),
        },
        "training_config_template": config,
        "training_config_template_sha256": domain_sha256(
            "plant-beta-capacity-training-template-v1", config
        ),
    }
    return dict(
        _mapping(
            "capacity cell contract",
            json.loads(canonical_json(contract)),
        )
    )


def _sysctl_value(name: str) -> str | None:
    if platform.system() != "Darwin":
        return None
    try:
        completed = subprocess.run(
            ("sysctl", "-n", name),
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _cpu_brand() -> str | None:
    system = platform.system()
    if system == "Darwin":
        return _sysctl_value("machdep.cpu.brand_string")
    if system == "Linux":
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        for line in cpuinfo.splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in {"model name", "Hardware"}:
                observed = value.strip()
                if observed:
                    return observed
        return None
    if system == "Windows":
        observed = os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
        return observed or None
    return None


def _optional_positive_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _canonical_thread_count(name: str, value: str | None) -> int | None:
    if value is None:
        return None
    if not value or not value.isascii() or not value.isdigit() or value.startswith("0"):
        raise CapacityBenchmarkError(
            f"host thread setting {name} must be a canonical positive integer"
        )
    parsed = int(value)
    if parsed < 1 or parsed > CAPACITY_MAX_THREAD_COUNT or str(parsed) != value:
        raise CapacityBenchmarkError(
            f"host thread setting {name} is outside the canonical range"
        )
    return parsed


def _physical_memory_bytes() -> int | None:
    darwin = _optional_positive_int(_sysctl_value("hw.memsize"))
    if darwin is not None:
        return darwin
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    total = page_size * pages
    return total if total > 0 else None


def _host_identity() -> dict[str, Any]:
    try:
        import torch
    except ImportError as reason:  # pragma: no cover - guarded by training extra
        raise CapacityBenchmarkError("capacity benchmark requires Torch") from reason
    logical_cpus = os.cpu_count()
    return {
        "platform": platform.platform(aliased=False, terse=False),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_brand": _cpu_brand(),
        "physical_cpu_count": _optional_positive_int(_sysctl_value("hw.physicalcpu")),
        "logical_cpu_count": (
            logical_cpus if logical_cpus and logical_cpus > 0 else None
        ),
        "physical_memory_bytes": _physical_memory_bytes(),
        "torch_intraop_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
        "thread_environment": {
            name: _canonical_thread_count(name, os.environ.get(name))
            for name in CAPACITY_THREAD_ENVIRONMENT
        },
        "clock": {
            "name": "time.perf_counter_ns",
            "monotonic": bool(time.get_clock_info("perf_counter").monotonic),
            "adjustable": bool(time.get_clock_info("perf_counter").adjustable),
            "resolution_s": float(time.get_clock_info("perf_counter").resolution),
        },
    }


def _allocated_bytes(value: os.stat_result) -> int:
    blocks = getattr(value, "st_blocks", None)
    if isinstance(blocks, int) and blocks >= 0:
        return blocks * 512
    block_size = max(1, int(getattr(value, "st_blksize", 4096)))
    return ((value.st_size + block_size - 1) // block_size) * block_size


def _relative_to_root(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as reason:
        raise CapacityBenchmarkError(
            f"artifact escapes capacity root: {path}"
        ) from reason
    return relative.as_posix()


def _resolve_report_path(root: Path, value: object, *, expect_file: bool) -> Path:
    relative = _relative_path("report artifact path", value)
    resolved_root = root.resolve(strict=True)
    candidate = resolved_root
    lexical: os.stat_result | None = None
    for part in PurePosixPath(relative).parts:
        candidate = candidate / part
        try:
            lexical = candidate.lstat()
        except OSError as reason:
            raise CapacityBenchmarkError(
                f"cannot inspect report artifact {candidate}: {reason}"
            ) from reason
        if stat.S_ISLNK(lexical.st_mode):
            raise CapacityBenchmarkError(
                f"report artifact path cannot contain a symlink: {candidate}"
            )
    if lexical is None:  # pragma: no cover - _relative_path rejects empty paths
        raise CapacityBenchmarkError("report artifact path is empty")
    if expect_file and not stat.S_ISREG(lexical.st_mode):
        raise CapacityBenchmarkError(
            f"report artifact must be a regular file: {candidate}"
        )
    if not expect_file and not stat.S_ISDIR(lexical.st_mode):
        raise CapacityBenchmarkError(
            f"report artifact must be a directory: {candidate}"
        )
    return candidate


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    before = path.lstat()
    raw = _read_regular_bytes(path)
    metadata = path.lstat()
    if _stat_identity(before) != _stat_identity(metadata):
        raise CapacityBenchmarkError(f"artifact changed while inventorying: {path}")
    return {
        "path": _relative_to_root(root, path),
        "logical_bytes": len(raw),
        "allocated_bytes": _allocated_bytes(metadata),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _checkpoint_file_inventory(root: Path, checkpoint: Path) -> list[dict[str, Any]]:
    lexical = checkpoint.lstat()
    if stat.S_ISLNK(lexical.st_mode) or not stat.S_ISDIR(lexical.st_mode):
        raise CapacityBenchmarkError("checkpoint must be a nonsymlink directory")
    children = sorted(checkpoint.iterdir(), key=lambda path: path.name)
    if [path.name for path in children] != sorted(CAPACITY_CHECKPOINT_FILES):
        raise CapacityBenchmarkError(
            "capacity checkpoint must contain exactly policy.zip, metadata.json, "
            "and training_config.json"
        )
    return [_file_record(root, path) for path in children]


def _admit_checkpoint(
    root: Path,
    checkpoint: Path,
    config: ControllerNeutralTrainingConfigV1,
    runtime: BetaTrainingRuntimeProvenanceV1,
) -> tuple[dict[str, Any], int, int]:
    checkpoint_config = checkpoint / CONFIG_FILE
    if _read_regular_bytes(checkpoint_config) != _training_config_bytes(config):
        raise CapacityBenchmarkError(
            "checkpoint training_config.json differs from the capacity cell"
        )
    metadata = checkpoints.load_metadata(checkpoint, check=False)
    if metadata.training_config != config.to_dict():
        raise CapacityBenchmarkError(
            "checkpoint training config differs from capacity cell"
        )
    if (
        metadata.algorithm != config.beta_rl.algorithm  # type: ignore[union-attr]
        or metadata.seed != CAPACITY_OPTIMIZER_SEED
        or metadata.total_timesteps != CAPACITY_TRANSITIONS
    ):
        raise CapacityBenchmarkError(
            "checkpoint campaign identity differs from capacity cell"
        )
    if set(metadata.extra) != {BETA_CHECKPOINT_EXTRA_KEY}:
        raise CapacityBenchmarkError(
            "checkpoint evidence fields differ from beta contract"
        )
    evidence = parse_beta_checkpoint_evidence(metadata.extra[BETA_CHECKPOINT_EXTRA_KEY])
    expected_algorithm = config.beta_rl.algorithm  # type: ignore[union-attr]
    if (
        evidence.algorithm_id != expected_algorithm
        or evidence.run_kind != BETA_FIXTURE_RUN_KIND
        or evidence.optimizer_seed != CAPACITY_OPTIMIZER_SEED
        or evidence.total_timesteps != CAPACITY_TRANSITIONS
        or evidence.resumed_from_parent
        or evidence.resume_parent is not None
        or evidence.exact_endpoint_admitted
        or evidence.eligible_for_registered_evaluation
        or evidence.runtime_provenance != runtime
    ):
        raise CapacityBenchmarkError(
            "checkpoint evidence is not the frozen fixture cell"
        )

    environment = make_env(config)
    try:
        loaded = load_policy(
            checkpoint,
            contract=policy_contract_for_environment(environment),
            environment=environment,
            expected_beta_runtime_provenance=runtime,
            require_registered_evaluation_eligibility=False,
        )
    finally:
        environment.close()
    if loaded.metadata != metadata:
        raise CapacityBenchmarkError(
            "checkpoint metadata changed during production admission"
        )

    constraint = _mapping("constraint state", evidence.constraint_state)
    payload = _mapping("constraint state payload", constraint.get("payload"))
    rollout_entries = _sequence("rollout ledgers", payload.get("rollout_ledgers"))
    if len(rollout_entries) != CAPACITY_ROLLOUT_COUNT:
        raise CapacityBenchmarkError(
            "capacity checkpoint must contain exactly one rollout"
        )
    rollout = _mapping("capacity rollout", rollout_entries[0])
    if (
        rollout.get("algorithm_id") != expected_algorithm
        or _integer(
            "rollout transition count", rollout.get("transition_count"), minimum=1
        )
        != CAPACITY_TRANSITIONS
        or _integer(
            "rollout optimizer steps", rollout.get("optimizer_steps"), minimum=1
        )
        != CAPACITY_OPTIMIZER_STEPS
    ):
        raise CapacityBenchmarkError(
            "capacity rollout shape differs from frozen contract"
        )
    rollout_elapsed_s = _finite(
        "rollout elapsed_compute_s", rollout.get("elapsed_compute_s"), minimum=0.0
    )
    if rollout_elapsed_s <= 0.0:
        raise CapacityBenchmarkError("capacity rollout elapsed time must be positive")
    rollout_elapsed_ns = max(1, int(round(rollout_elapsed_s * 1_000_000_000)))

    snapshot = _mapping("allocator snapshot", evidence.allocator_snapshot)
    identity = _mapping("allocator identity", snapshot.get("allocator_identity"))
    ledger = _mapping("allocator ledger", snapshot.get("ledger"))
    allocator_entries = _sequence("allocator entries", ledger.get("entries"))
    if (
        identity.get("mode") != BETA_FIXTURE_RUN_KIND
        or identity.get("split") != BETA_FIXTURE_RUN_KIND
        or _integer("allocator base seed", identity.get("base_seed"))
        != CAPACITY_FIXTURE_EPISODE_SEED_BASE
        or len(allocator_entries) != CAPACITY_ALLOCATOR_ENTRIES
    ):
        raise CapacityBenchmarkError("allocator evidence differs from capacity fixture")
    observed_seeds = sorted(
        _integer(
            f"allocator entry {index} seed",
            _mapping(f"allocator entry {index}", entry).get("episode_seed"),
        )
        for index, entry in enumerate(allocator_entries)
    )
    if observed_seeds != list(
        range(
            CAPACITY_FIXTURE_EPISODE_SEED_BASE,
            CAPACITY_FIXTURE_LAST_EPISODE_SEED + 1,
        )
    ):
        raise CapacityBenchmarkError(
            "allocator used a seed outside the frozen fixture wave"
        )

    files = _checkpoint_file_inventory(root, checkpoint)
    logical_bytes = sum(
        _integer("file logical bytes", item["logical_bytes"]) for item in files
    )
    allocated_bytes = sum(
        _integer("file allocated bytes", item["allocated_bytes"]) for item in files
    )
    max_rollout_bytes = max(
        len(canonical_json(dict(_mapping("rollout entry", entry))).encode("utf-8"))
        for entry in rollout_entries
    )
    max_allocator_bytes = max(
        len(canonical_json(dict(_mapping("allocator entry", entry))).encode("utf-8"))
        for entry in allocator_entries
    )
    record = {
        "path": _relative_to_root(root, checkpoint),
        "metadata_sha256": hashlib.sha256(
            _read_regular_bytes(checkpoints.metadata_path(checkpoint))
        ).hexdigest(),
        "checkpoint_evidence_sha256": evidence.sha256,
        "runtime_provenance_sha256": evidence.runtime_provenance.sha256,
        "environment_contract_sha256": metadata.environment_contract_digest,
        "checkpoint_contract_sha256": metadata.checkpoint_contract_digest,
        "files": files,
        "logical_bytes": logical_bytes,
        "allocated_bytes": allocated_bytes,
        "rollout_entry_count": len(rollout_entries),
        "allocator_entry_count": len(allocator_entries),
        "max_rollout_entry_bytes": max_rollout_bytes,
        "max_allocator_entry_bytes": max_allocator_bytes,
    }
    return record, rollout_elapsed_ns, CAPACITY_OPTIMIZER_STEPS


def _factor_label(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.2f}x"


_SENSITIVITY_RATIOS = ((1, 1), (5, 4), (3, 2))


def _scaled_ns(value: int, numerator: int, denominator: int) -> int:
    return (value * numerator + denominator - 1) // denominator


def _build_measurement(wall_ns: int, rollout_ns: int) -> dict[str, Any]:
    normalized_wall = _integer("wall elapsed ns", wall_ns, minimum=1)
    normalized_rollout = _integer("rollout elapsed ns", rollout_ns, minimum=1)
    fixed = max(0, normalized_wall - normalized_rollout)
    return {
        "controller_transitions": CAPACITY_TRANSITIONS,
        "physical_minute_steps": CAPACITY_PHYSICAL_MINUTE_STEPS,
        "rollout_elapsed_ns": normalized_rollout,
        "fixed_overhead_ns": fixed,
        "controller_transitions_per_second": (
            CAPACITY_TRANSITIONS * 1_000_000_000.0 / normalized_rollout
        ),
        "physical_minute_steps_per_second": (
            CAPACITY_PHYSICAL_MINUTE_STEPS * 1_000_000_000.0 / normalized_rollout
        ),
        "optimizer_steps": CAPACITY_OPTIMIZER_STEPS,
    }


def _build_projection(
    measurement: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    fixed_ns = _integer("fixed overhead ns", measurement["fixed_overhead_ns"])
    rollout_ns = _integer(
        "rollout elapsed ns", measurement["rollout_elapsed_ns"], minimum=1
    )
    per_policy: dict[str, int] = {}
    family: dict[str, int] = {}
    for numerator, denominator in _SENSITIVITY_RATIOS:
        label = _factor_label(numerator, denominator)
        projected = fixed_ns + _scaled_ns(
            REGISTERED_ROLLOUTS_PER_POLICY * rollout_ns,
            numerator,
            denominator,
        )
        per_policy[label] = projected
        family[label] = REGISTERED_POLICIES_PER_FAMILY * projected

    rollout_bytes = _integer(
        "maximum rollout entry bytes", checkpoint["max_rollout_entry_bytes"], minimum=1
    )
    allocator_bytes = _integer(
        "maximum allocator entry bytes",
        checkpoint["max_allocator_entry_bytes"],
        minimum=1,
    )
    growth = 2 * (
        (REGISTERED_ROLLOUTS_PER_POLICY - CAPACITY_ROLLOUT_COUNT) * 3 * rollout_bytes
        + (REGISTERED_ALLOCATOR_ENTRIES_PER_POLICY - CAPACITY_ALLOCATOR_ENTRIES)
        * 4
        * allocator_bytes
    )
    projected_checkpoint = (
        _integer("checkpoint logical bytes", checkpoint["logical_bytes"], minimum=1)
        + growth
    )
    return {
        "registered_policy_wall_ns_by_factor": per_policy,
        "five_policy_family_wall_ns_by_factor": family,
        "structural_growth_allowance_bytes": growth,
        "projected_checkpoint_logical_bytes": projected_checkpoint,
        "five_policy_family_logical_bytes": (
            REGISTERED_POLICIES_PER_FAMILY * projected_checkpoint
        ),
    }


def _campaign_projection(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if len(cells) != len(CAPACITY_ALGORITHM_ORDER) or any(
        cell.get("status") != "completed" for cell in cells
    ):
        return None
    wall: dict[str, int] = {}
    for numerator, denominator in _SENSITIVITY_RATIOS:
        label = _factor_label(numerator, denominator)
        wall[label] = sum(
            _integer(
                "family projected wall ns",
                _mapping("cell projection", cell["projection"])[
                    "five_policy_family_wall_ns_by_factor"
                ][label],
                minimum=1,
            )
            for cell in cells
        )
    projected_storage = sum(
        _integer(
            "family projected storage",
            _mapping("cell projection", cell["projection"])[
                "five_policy_family_logical_bytes"
            ],
            minimum=1,
        )
        for cell in cells
    )
    return {
        "sequential_wall_ns_by_factor": wall,
        "projected_checkpoint_logical_bytes": projected_storage,
        "recommended_free_space_bytes": max(
            2 * projected_storage,
            projected_storage + 1024**3,
        ),
        "duration_scenarios_are_not_confidence_intervals": True,
        "storage_is_a_structural_planning_allowance_not_zip_prediction": True,
    }


ProcessRunner = Callable[[tuple[str, ...], Path, int], tuple[int, bytes, bytes]]
CheckpointAdmitter = Callable[
    [Path, Path, ControllerNeutralTrainingConfigV1, BetaTrainingRuntimeProvenanceV1],
    tuple[dict[str, Any], int, int],
]
RuntimeCapture = Callable[[], BetaTrainingRuntimeProvenanceV1]


def _default_process_runner(
    command: tuple[str, ...], repository: Path, timeout_s: int
) -> tuple[int, bytes, bytes]:
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(
        command,
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        timeout=timeout_s,
    )
    return int(completed.returncode), completed.stdout, completed.stderr


def _clean_runtime_capture() -> BetaTrainingRuntimeProvenanceV1:
    return capture_runtime_provenance(require_clean=True)


def _rusage_children_ns() -> tuple[int, int]:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (
        max(0, int(round(float(usage.ru_utime) * 1_000_000_000))),
        max(0, int(round(float(usage.ru_stime) * 1_000_000_000))),
    )


def _new_external_root(output: Path, repository: Path) -> Path:
    if not output.is_absolute():
        raise ValueError("capacity output must be an absolute path")
    if output.exists() or output.is_symlink():
        raise ValueError("capacity output must be a new non-existing directory")
    try:
        parent = output.parent.resolve(strict=True)
    except OSError as reason:
        raise ValueError(f"capacity output parent is unavailable: {reason}") from reason
    target = parent / output.name
    if target.is_relative_to(repository.resolve(strict=True)):
        raise ValueError("capacity output must be external to the repository")
    target.mkdir(mode=0o750, exist_ok=False)
    for name in ("configs", "logs", "artifacts"):
        (target / name).mkdir(mode=0o750, exist_ok=False)
    return target


def _log_record(root: Path, path: Path, raw: bytes) -> dict[str, Any]:
    digest = _publish_bytes_exclusive(path, raw)
    return {
        "path": _relative_to_root(root, path),
        "logical_bytes": len(raw),
        "sha256": digest,
    }


def _execute_cell(
    *,
    root: Path,
    repository: Path,
    order: int,
    algorithm: str,
    runtime: BetaTrainingRuntimeProvenanceV1,
    process_runner: ProcessRunner,
    checkpoint_admitter: CheckpointAdmitter,
    clock_ns: Callable[[], int],
) -> dict[str, Any]:
    layout = _cell_layout(order, algorithm)
    output = root.joinpath(*PurePosixPath(layout["output"]).parts)
    config = _cell_config(output, algorithm)
    config_path = root.joinpath(*PurePosixPath(layout["config"]).parts)
    config_raw = _training_config_bytes(config)
    _publish_bytes_exclusive(config_path, config_raw)
    stdout_path = root.joinpath(*PurePosixPath(layout["stdout"]).parts)
    stderr_path = root.joinpath(*PurePosixPath(layout["stderr"]).parts)
    command = (
        sys.executable,
        "-m",
        "plant_backend.rl.v2.training",
        "--config",
        str(config_path),
    )
    before_user, before_system = _rusage_children_ns()
    started_ns = clock_ns()
    failure: dict[str, str] | None = None
    exit_code: int | None
    stdout = b""
    stderr = b""
    try:
        exit_code, stdout, stderr = process_runner(
            command, repository, CAPACITY_CHILD_TIMEOUT_S
        )
    except subprocess.TimeoutExpired as reason:
        exit_code = None
        failure = _failure("training_timeout")
        for supplied, destination in (
            (reason.stdout, "stdout"),
            (reason.stderr, "stderr"),
        ):
            if supplied is None:
                raw = b""
            elif isinstance(supplied, bytes):
                raw = supplied
            else:
                raw = supplied.encode("utf-8", errors="replace")
            if destination == "stdout":
                stdout = raw
            else:
                stderr = raw
    except Exception:
        exit_code = None
        failure = _failure("training_launch_error")
    finished_ns = clock_ns()
    wall_ns = finished_ns - started_ns
    if wall_ns <= 0:
        raise CapacityBenchmarkError("capacity child wall clock did not advance")
    after_user, after_system = _rusage_children_ns()
    stdout_record = _log_record(root, stdout_path, stdout)
    stderr_record = _log_record(root, stderr_path, stderr)

    checkpoint_record: dict[str, Any] | None = None
    measurement: dict[str, Any] | None = None
    projection: dict[str, Any] | None = None
    if failure is None and exit_code != 0:
        failure = _failure("training_nonzero_exit")
    if failure is None:
        try:
            checkpoint = output / "checkpoint"
            checkpoint_record, rollout_ns, optimizer_steps = checkpoint_admitter(
                root,
                checkpoint,
                config,
                runtime,
            )
            if optimizer_steps != CAPACITY_OPTIMIZER_STEPS:
                raise CapacityBenchmarkError("admitted optimizer-step count differs")
            measurement = _build_measurement(wall_ns, rollout_ns)
            projection = _build_projection(measurement, checkpoint_record)
        except Exception:
            failure = _failure("checkpoint_admission_error")
            checkpoint_record = None
            measurement = None
            projection = None

    return {
        "order": order,
        "algorithm": algorithm,
        "optimizer_seed": CAPACITY_OPTIMIZER_SEED,
        "fixture_episode_seed_base": CAPACITY_FIXTURE_EPISODE_SEED_BASE,
        "config_path": _relative_to_root(root, config_path),
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "output_path": _relative_to_root(root, output),
        "status": "completed" if failure is None else "failed",
        "exit_code": exit_code,
        "wall_elapsed_ns": wall_ns,
        "child_user_cpu_ns": max(0, after_user - before_user),
        "child_system_cpu_ns": max(0, after_system - before_system),
        "stdout": stdout_record,
        "stderr": stderr_record,
        "checkpoint": checkpoint_record,
        "measurement": measurement,
        "projection": projection,
        "failure": failure,
    }


def _deterministic_evidence(
    *,
    protocol: Mapping[str, Any],
    runtime: BetaTrainingRuntimeProvenanceV1,
    host: Mapping[str, Any],
    cell_contract: Mapping[str, Any],
    claim_boundary: str = CAPACITY_CLAIM,
) -> dict[str, Any]:
    """Return only the frozen pre-run measurement identity.

    Post-run cells are intentionally absent.  The full report digest owns all
    timing, logs, artifacts, outcomes, sizes, failures, and projections.
    """

    normalized_host = _verify_host(host)
    normalized_host.pop("clock")
    template_sha256 = _sha256(
        "capacity normalized training template sha256",
        cell_contract.get("training_config_template_sha256"),
    )
    return {
        "protocol": dict(protocol),
        "claim_boundary": claim_boundary,
        "runtime_provenance_sha256": runtime.sha256,
        "host_capacity": normalized_host,
        "operator_assertions": _operator_assertions(),
        "cell_contract": dict(cell_contract),
        "planned_cells": [
            {
                "order": order,
                "algorithm": algorithm,
                "optimizer_seed": CAPACITY_OPTIMIZER_SEED,
                "fixture_episode_seed_base": CAPACITY_FIXTURE_EPISODE_SEED_BASE,
                "normalized_training_config_template_sha256": template_sha256,
            }
            for order, algorithm in enumerate(CAPACITY_ALGORITHM_ORDER, start=1)
        ],
        "campaign_arithmetic": dict(REGISTERED_CAMPAIGN_ARITHMETIC),
    }


def _operator_assertions() -> dict[str, bool]:
    return {
        "cells_executed_sequentially": True,
        "no_other_repository_benchmark_intentionally_running": True,
        "attempts_not_retried_or_selected": True,
    }


@dataclass(frozen=True)
class _CapacityRunData:
    """Collected cell data with no report-issuance authority."""

    root: Path
    protocol: Mapping[str, Any]
    runtime: BetaTrainingRuntimeProvenanceV1
    host: Mapping[str, Any]
    cells: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class _SyntheticCapacityRunTrace:
    """A dependency-injected trace that is permanently synthetic."""

    data: _CapacityRunData


@dataclass(frozen=True)
class _ProductionCapacityRunTrace:
    """A trace created only through fixed production dependencies."""

    data: _CapacityRunData


def _collect_capacity_cells(
    output: Path | str,
    *,
    assert_exclusive: bool,
    assert_first_attempt: bool,
    process_runner: ProcessRunner,
    checkpoint_admitter: CheckpointAdmitter,
    runtime_capture: RuntimeCapture,
    host_capture: Callable[[], Mapping[str, Any]],
    clock_ns: Callable[[], int],
) -> _CapacityRunData:
    """Collect cells through explicit dependencies without issuing a report."""

    if assert_exclusive is not True:
        raise ValueError(
            "capacity execution requires --assert-exclusive after checking that no "
            "other repository benchmark is intentionally running"
        )
    if assert_first_attempt is not True:
        raise ValueError(
            "capacity execution requires --assert-first-attempt after checking that "
            "this source revision has no earlier capacity attempt"
        )
    repository = _repository_root().resolve(strict=True)
    protocol = _protocol_identity(repository)
    runtime = runtime_capture()
    if not runtime.worktree_clean:
        raise CapacityBenchmarkError("capacity benchmark requires a clean worktree")
    host = _verify_host(host_capture())
    root = _new_external_root(Path(output).expanduser(), repository)
    cells: list[dict[str, Any]] = []
    for order, algorithm in enumerate(CAPACITY_ALGORITHM_ORDER, start=1):
        cell = _execute_cell(
            root=root,
            repository=repository,
            order=order,
            algorithm=algorithm,
            runtime=runtime,
            process_runner=process_runner,
            checkpoint_admitter=checkpoint_admitter,
            clock_ns=clock_ns,
        )
        cells.append(cell)
        verify_runtime_unchanged(runtime, runtime_capture())
    return _CapacityRunData(
        root=root,
        protocol=protocol,
        runtime=runtime,
        host=host,
        cells=tuple(cells),
    )


def _run_synthetic_capacity_cells(
    output: Path | str,
    *,
    assert_exclusive: bool,
    assert_first_attempt: bool,
    process_runner: ProcessRunner,
    checkpoint_admitter: CheckpointAdmitter,
    runtime_capture: RuntimeCapture,
    host_capture: Callable[[], Mapping[str, Any]],
    clock_ns: Callable[[], int],
) -> _SyntheticCapacityRunTrace:
    return _SyntheticCapacityRunTrace(
        data=_collect_capacity_cells(
            output,
            assert_exclusive=assert_exclusive,
            assert_first_attempt=assert_first_attempt,
            process_runner=process_runner,
            checkpoint_admitter=checkpoint_admitter,
            runtime_capture=runtime_capture,
            host_capture=host_capture,
            clock_ns=clock_ns,
        )
    )


def _run_production_capacity_cells(
    output: Path | str,
    *,
    assert_exclusive: bool,
    assert_first_attempt: bool,
) -> _ProductionCapacityRunTrace:
    return _ProductionCapacityRunTrace(
        data=_collect_capacity_cells(
            output,
            assert_exclusive=assert_exclusive,
            assert_first_attempt=assert_first_attempt,
            process_runner=_default_process_runner,
            checkpoint_admitter=_admit_checkpoint,
            runtime_capture=_clean_runtime_capture,
            host_capture=_host_identity,
            clock_ns=time.perf_counter_ns,
        )
    )


def _build_synthetic_report(trace: _SyntheticCapacityRunTrace) -> dict[str, Any]:
    """Build explicitly ineligible evidence from a synthetic trace only."""

    data = trace.data
    contract = _cell_contract()
    status = "completed" if _campaign_projection(data.cells) is not None else "failed"
    body = {
        "schema": CAPACITY_SYNTHETIC_REPORT_SCHEMA,
        "protocol": dict(data.protocol),
        "claim_boundary": CAPACITY_SYNTHETIC_CLAIM,
        "status": status,
        "operator_assertions": _operator_assertions(),
        "runtime_provenance": data.runtime.to_dict(),
        "host": dict(data.host),
        "cell_contract": contract,
        "cells": [dict(cell) for cell in data.cells],
        "campaign_arithmetic": dict(REGISTERED_CAMPAIGN_ARITHMETIC),
        "deterministic_evidence_sha256": domain_sha256(
            CAPACITY_SYNTHETIC_DETERMINISTIC_DOMAIN,
            _deterministic_evidence(
                protocol=data.protocol,
                runtime=data.runtime,
                host=data.host,
                cell_contract=contract,
                claim_boundary=CAPACITY_SYNTHETIC_CLAIM,
            ),
        ),
        "execution_class": CAPACITY_SYNTHETIC_EXECUTION_CLASS,
        "result_claim": None,
        "campaign_projection": None,
    }
    body["sha256"] = domain_sha256(CAPACITY_SYNTHETIC_REPORT_DOMAIN, body)
    return body


def _execute_synthetic_capacity_benchmark(
    output: Path | str,
    *,
    assert_exclusive: bool,
    assert_first_attempt: bool,
    process_runner: ProcessRunner,
    checkpoint_admitter: CheckpointAdmitter,
    runtime_capture: RuntimeCapture,
    host_capture: Callable[[], Mapping[str, Any]],
    clock_ns: Callable[[], int],
) -> dict[str, Any]:
    """Serialize an explicitly synthetic, production-ineligible test report."""

    trace = _run_synthetic_capacity_cells(
        output,
        assert_exclusive=assert_exclusive,
        assert_first_attempt=assert_first_attempt,
        process_runner=process_runner,
        checkpoint_admitter=checkpoint_admitter,
        runtime_capture=runtime_capture,
        host_capture=host_capture,
        clock_ns=clock_ns,
    )
    report = _build_synthetic_report(trace)
    data = trace.data
    report_path = data.root / CAPACITY_SYNTHETIC_REPORT_FILENAME
    raw_file_sha256 = _publish_json_exclusive(report_path, report)
    _verify_synthetic_capacity_report(
        report_path,
        runtime_capture=runtime_capture,
        checkpoint_admitter=checkpoint_admitter,
        host_capture=host_capture,
    )
    verify_runtime_unchanged(data.runtime, runtime_capture())
    return {
        "report": str(report_path),
        "raw_file_sha256": raw_file_sha256,
        "report_sha256": report["sha256"],
        "status": report["status"],
    }


def execute_capacity_benchmark(
    output: Path | str,
    *,
    assert_exclusive: bool,
    assert_first_attempt: bool,
) -> dict[str, Any]:
    """Execute and issue production evidence through fixed concrete dependencies."""

    trace = _run_production_capacity_cells(
        output,
        assert_exclusive=assert_exclusive,
        assert_first_attempt=assert_first_attempt,
    )
    data = trace.data
    contract = _cell_contract()
    projection = _campaign_projection(data.cells)
    status = "completed" if projection is not None else "failed"
    report = {
        "schema": CAPACITY_REPORT_SCHEMA,
        "protocol": dict(data.protocol),
        "claim_boundary": CAPACITY_CLAIM,
        "status": status,
        "operator_assertions": _operator_assertions(),
        "runtime_provenance": data.runtime.to_dict(),
        "host": dict(data.host),
        "cell_contract": contract,
        "cells": [dict(cell) for cell in data.cells],
        "campaign_arithmetic": dict(REGISTERED_CAMPAIGN_ARITHMETIC),
        "deterministic_evidence_sha256": domain_sha256(
            CAPACITY_DETERMINISTIC_DOMAIN,
            _deterministic_evidence(
                protocol=data.protocol,
                runtime=data.runtime,
                host=data.host,
                cell_contract=contract,
            ),
        ),
        "result_claim": (CAPACITY_RESULT_CLAIM if projection is not None else None),
        "campaign_projection": projection,
    }
    report["sha256"] = domain_sha256(CAPACITY_REPORT_DOMAIN, report)
    report_path = data.root / CAPACITY_REPORT_FILENAME
    raw_file_sha256 = _publish_json_exclusive(report_path, report)
    _verify_capacity_report(report_path)
    verify_runtime_unchanged(data.runtime, _clean_runtime_capture())
    return {
        "report": str(report_path),
        "raw_file_sha256": raw_file_sha256,
        "report_sha256": report["sha256"],
        "status": report["status"],
    }


def _verify_host(value: object) -> dict[str, Any]:
    host = dict(_mapping("capacity host", value))
    _exact_keys("capacity host", host, _HOST_FIELDS)
    for name in ("platform", "system", "machine"):
        if not isinstance(host[name], str) or not host[name]:
            raise ValueError(f"host {name} must be a non-empty string")
    for name in ("processor", "cpu_brand"):
        if host[name] is not None and (
            not isinstance(host[name], str) or not host[name]
        ):
            raise ValueError(f"host {name} must be null or a non-empty string")
    for name in (
        "physical_cpu_count",
        "logical_cpu_count",
        "physical_memory_bytes",
    ):
        if host[name] is not None:
            _integer(f"host {name}", host[name], minimum=1)
    _integer("host torch_intraop_threads", host["torch_intraop_threads"], minimum=1)
    _integer("host torch_interop_threads", host["torch_interop_threads"], minimum=1)
    thread_environment = _mapping("host thread environment", host["thread_environment"])
    if set(thread_environment) != set(CAPACITY_THREAD_ENVIRONMENT):
        raise CapacityBenchmarkError("host thread-environment fields differ")
    for name, setting in thread_environment.items():
        if setting is not None:
            parsed = _integer(f"thread environment {name}", setting, minimum=1)
            if parsed > CAPACITY_MAX_THREAD_COUNT:
                raise ValueError(
                    f"thread environment {name} exceeds the canonical range"
                )
    clock = dict(_mapping("capacity clock", host["clock"]))
    _exact_keys("capacity clock", clock, _CLOCK_FIELDS)
    if clock["name"] != "time.perf_counter_ns":
        raise CapacityBenchmarkError("capacity clock differs")
    if clock["monotonic"] is not True or not isinstance(clock["adjustable"], bool):
        raise CapacityBenchmarkError("capacity clock properties differ")
    if _finite("capacity clock resolution", clock["resolution_s"]) <= 0.0:
        raise CapacityBenchmarkError("capacity clock resolution must be positive")
    return host


def _verify_log_record(root: Path, value: object) -> dict[str, Any]:
    record = dict(_mapping("capacity log record", value))
    _exact_keys("capacity log record", record, _LOG_FIELDS)
    path = _resolve_report_path(root, record["path"], expect_file=True)
    raw = _read_regular_bytes(path)
    if (
        _integer("log logical bytes", record["logical_bytes"]) != len(raw)
        or _sha256("log sha256", record["sha256"]) != hashlib.sha256(raw).hexdigest()
    ):
        raise CapacityBenchmarkError("capacity log identity differs")
    return record


def _verify_file_record(root: Path, value: object) -> dict[str, Any]:
    record = dict(_mapping("checkpoint file record", value))
    _exact_keys("checkpoint file record", record, _FILE_FIELDS)
    path = _resolve_report_path(root, record["path"], expect_file=True)
    raw = _read_regular_bytes(path)
    metadata = path.stat(follow_symlinks=False)
    if (
        _integer("checkpoint file logical bytes", record["logical_bytes"]) != len(raw)
        or _integer("checkpoint file allocated bytes", record["allocated_bytes"])
        != _allocated_bytes(metadata)
        or _sha256("checkpoint file sha256", record["sha256"])
        != hashlib.sha256(raw).hexdigest()
    ):
        raise CapacityBenchmarkError("checkpoint file identity differs")
    return record


def _verify_checkpoint_record(root: Path, value: object) -> dict[str, Any]:
    record = dict(_mapping("capacity checkpoint", value))
    _exact_keys("capacity checkpoint", record, _CHECKPOINT_FIELDS)
    checkpoint_relative = _relative_path("capacity checkpoint path", record["path"])
    _resolve_report_path(root, checkpoint_relative, expect_file=False)
    files = [
        _verify_file_record(root, item)
        for item in _sequence("checkpoint files", record["files"])
    ]
    if len(files) != len(CAPACITY_CHECKPOINT_FILES):
        raise CapacityBenchmarkError("capacity checkpoint file count differs")
    expected_file_paths = sorted(
        f"{checkpoint_relative}/{name}" for name in CAPACITY_CHECKPOINT_FILES
    )
    if [item["path"] for item in files] != expected_file_paths:
        raise CapacityBenchmarkError("capacity checkpoint file paths differ")
    for name in (
        "metadata_sha256",
        "checkpoint_evidence_sha256",
        "runtime_provenance_sha256",
        "environment_contract_sha256",
        "checkpoint_contract_sha256",
    ):
        _sha256(f"checkpoint {name}", record[name])
    if (
        _integer("checkpoint logical bytes", record["logical_bytes"], minimum=1)
        != sum(item["logical_bytes"] for item in files)
        or _integer("checkpoint allocated bytes", record["allocated_bytes"], minimum=1)
        != sum(item["allocated_bytes"] for item in files)
        or _integer("checkpoint rollout entries", record["rollout_entry_count"])
        != CAPACITY_ROLLOUT_COUNT
        or _integer("checkpoint allocator entries", record["allocator_entry_count"])
        != CAPACITY_ALLOCATOR_ENTRIES
    ):
        raise CapacityBenchmarkError("capacity checkpoint size/count summary differs")
    _integer(
        "checkpoint maximum rollout entry bytes",
        record["max_rollout_entry_bytes"],
        minimum=1,
    )
    _integer(
        "checkpoint maximum allocator entry bytes",
        record["max_allocator_entry_bytes"],
        minimum=1,
    )
    return record


def _verify_failure(value: object) -> dict[str, str]:
    failure = dict(_mapping("capacity failure", value))
    _exact_keys("capacity failure", failure, _FAILURE_FIELDS)
    category = failure["category"]
    if not isinstance(category, str) or category not in _FAILURE_CONTRACT:
        raise CapacityBenchmarkError("capacity failure category differs")
    expected = _failure(category)
    if failure != expected:
        raise CapacityBenchmarkError("capacity failure record is not canonical")
    return expected


def _verify_measurement(value: object) -> dict[str, Any]:
    measurement = dict(_mapping("capacity measurement", value))
    _exact_keys("capacity measurement", measurement, _MEASUREMENT_FIELDS)
    if (
        _integer("measurement transitions", measurement["controller_transitions"])
        != CAPACITY_TRANSITIONS
        or _integer("measurement physical steps", measurement["physical_minute_steps"])
        != CAPACITY_PHYSICAL_MINUTE_STEPS
        or _integer("measurement optimizer steps", measurement["optimizer_steps"])
        != CAPACITY_OPTIMIZER_STEPS
    ):
        raise CapacityBenchmarkError("capacity measurement work shape differs")
    rollout_ns = _integer(
        "measurement rollout elapsed ns", measurement["rollout_elapsed_ns"], minimum=1
    )
    _integer("measurement fixed overhead ns", measurement["fixed_overhead_ns"])
    expected_transition_rate = CAPACITY_TRANSITIONS * 1_000_000_000.0 / rollout_ns
    expected_physical_rate = (
        CAPACITY_PHYSICAL_MINUTE_STEPS * 1_000_000_000.0 / rollout_ns
    )
    if (
        _finite(
            "measurement controller transition rate",
            measurement["controller_transitions_per_second"],
        )
        != expected_transition_rate
        or _finite(
            "measurement physical step rate",
            measurement["physical_minute_steps_per_second"],
        )
        != expected_physical_rate
    ):
        raise CapacityBenchmarkError("capacity measurement rate differs")
    return measurement


def _verify_projection(value: object) -> dict[str, Any]:
    projection = dict(_mapping("capacity cell projection", value))
    _exact_keys("capacity cell projection", projection, _PROJECTION_FIELDS)
    expected_labels = {
        _factor_label(numerator, denominator)
        for numerator, denominator in _SENSITIVITY_RATIOS
    }
    for name in (
        "registered_policy_wall_ns_by_factor",
        "five_policy_family_wall_ns_by_factor",
    ):
        values = _mapping(f"projection {name}", projection[name])
        if set(values) != expected_labels:
            raise CapacityBenchmarkError(f"projection {name} factors differ")
        for label, item in values.items():
            _integer(f"projection {name}.{label}", item, minimum=1)
    for name in (
        "structural_growth_allowance_bytes",
        "projected_checkpoint_logical_bytes",
        "five_policy_family_logical_bytes",
    ):
        _integer(f"projection {name}", projection[name], minimum=1)
    return projection


def _verify_campaign_projection(value: object) -> dict[str, Any]:
    projection = dict(_mapping("capacity campaign projection", value))
    _exact_keys("capacity campaign projection", projection, _CAMPAIGN_PROJECTION_FIELDS)
    expected_labels = {
        _factor_label(numerator, denominator)
        for numerator, denominator in _SENSITIVITY_RATIOS
    }
    wall = _mapping(
        "campaign sequential wall scenarios", projection["sequential_wall_ns_by_factor"]
    )
    if set(wall) != expected_labels:
        raise CapacityBenchmarkError("campaign wall scenario factors differ")
    for label, item in wall.items():
        _integer(f"campaign wall scenario {label}", item, minimum=1)
    _integer(
        "campaign projected checkpoint bytes",
        projection["projected_checkpoint_logical_bytes"],
        minimum=1,
    )
    _integer(
        "campaign recommended free space",
        projection["recommended_free_space_bytes"],
        minimum=1,
    )
    if (
        projection["duration_scenarios_are_not_confidence_intervals"] is not True
        or projection["storage_is_a_structural_planning_allowance_not_zip_prediction"]
        is not True
    ):
        raise CapacityBenchmarkError("campaign projection caveats differ")
    return projection


@dataclass(frozen=True)
class _ReportEnvelope:
    path: Path
    root: Path
    report: dict[str, Any]
    supplied_sha256: str


@dataclass(frozen=True)
class _ReportContext:
    protocol: dict[str, Any]
    runtime: BetaTrainingRuntimeProvenanceV1
    host: dict[str, Any]
    contract: dict[str, Any]


CheckpointAdmission = tuple[dict[str, Any], int, int]


@dataclass(frozen=True)
class _CheckpointAdmissionInput:
    """A parsed completed-cell checkpoint reference with no admission authority."""

    order: int
    checkpoint: Path
    config: ControllerNeutralTrainingConfigV1


def _load_report_envelope(
    report_path: Path | str,
    *,
    expected_filename: str,
    expected_schema: str,
    expected_fields: frozenset[str],
    report_domain: str,
) -> _ReportEnvelope:
    """Load and authenticate an envelope before any external recapture/admission."""

    path = Path(report_path).expanduser()
    if path.name != expected_filename:
        raise ValueError(f"capacity report must be named {expected_filename}")
    root = path.parent.resolve(strict=True)
    path = _resolve_report_path(root, path.name, expect_file=True)
    report = _load_canonical_json(path)
    if report.get("schema") != expected_schema:
        raise CapacityBenchmarkError("capacity report schema differs")
    _exact_keys("capacity report", report, expected_fields)
    supplied_sha256 = _sha256("capacity report sha256", report["sha256"])
    body = dict(report)
    body.pop("sha256")
    if supplied_sha256 != domain_sha256(report_domain, body):
        raise CapacityBenchmarkError("capacity report digest differs")
    return _ReportEnvelope(
        path=path,
        root=root,
        report=report,
        supplied_sha256=supplied_sha256,
    )


def _verify_report_context(
    envelope: _ReportEnvelope,
    *,
    claim_boundary: str,
) -> _ReportContext:
    """Verify report-declared pre-run context without external hooks."""

    report = envelope.report
    repository = _repository_root().resolve(strict=True)
    if envelope.root.is_relative_to(repository):
        raise CapacityBenchmarkError("capacity report root must be external")
    protocol = _protocol_identity(repository)
    if dict(_mapping("capacity protocol", report["protocol"])) != protocol:
        raise CapacityBenchmarkError("capacity report protocol differs")
    if report["claim_boundary"] != claim_boundary:
        raise CapacityBenchmarkError("capacity report claim boundary differs")
    assertions = dict(_mapping("operator assertions", report["operator_assertions"]))
    _exact_keys("operator assertions", assertions, _OPERATOR_ASSERTION_FIELDS)
    if assertions != _operator_assertions():
        raise CapacityBenchmarkError("capacity operator assertions differ")
    runtime = BetaTrainingRuntimeProvenanceV1.from_dict(report["runtime_provenance"])
    if not runtime.worktree_clean:
        raise CapacityBenchmarkError("capacity report runtime must be clean")
    host = _verify_host(report["host"])
    contract = dict(_mapping("capacity cell contract", report["cell_contract"]))
    if contract != _cell_contract():
        raise CapacityBenchmarkError("capacity cell contract differs")
    if dict(_mapping("campaign arithmetic", report["campaign_arithmetic"])) != (
        REGISTERED_CAMPAIGN_ARITHMETIC
    ):
        raise CapacityBenchmarkError("registered campaign arithmetic differs")
    return _ReportContext(
        protocol=protocol,
        runtime=runtime,
        host=host,
        contract=contract,
    )


def _checkpoint_admission_inputs(
    envelope: _ReportEnvelope,
) -> tuple[_CheckpointAdmissionInput, ...]:
    """Parse completed checkpoint references without opening any checkpoint."""

    raw_cells = _sequence("capacity cells", envelope.report["cells"])
    if len(raw_cells) != len(CAPACITY_ALGORITHM_ORDER):
        raise CapacityBenchmarkError("capacity report must contain exactly three cells")
    inputs: list[_CheckpointAdmissionInput] = []
    for order, (algorithm, raw_cell) in enumerate(
        zip(CAPACITY_ALGORITHM_ORDER, raw_cells, strict=True), start=1
    ):
        cell = _mapping(f"capacity cell {order}", raw_cell)
        if cell.get("status") != "completed":
            continue
        layout = _cell_layout(order, algorithm)
        output = envelope.root.joinpath(*PurePosixPath(layout["output"]).parts)
        checkpoint = envelope.root.joinpath(*PurePosixPath(layout["checkpoint"]).parts)
        config = _cell_config(output, algorithm)
        inputs.append(
            _CheckpointAdmissionInput(
                order=order,
                checkpoint=checkpoint,
                config=config,
            )
        )
    return tuple(inputs)


def _collect_production_checkpoint_admissions(
    envelope: _ReportEnvelope,
    context: _ReportContext,
) -> dict[int, CheckpointAdmission]:
    """Admit only production-report checkpoints through the fixed real loader."""

    if envelope.report.get("schema") != CAPACITY_REPORT_SCHEMA:
        raise CapacityBenchmarkError(
            "production checkpoint admission requires the production report schema"
        )
    return {
        item.order: _admit_checkpoint(
            envelope.root,
            item.checkpoint,
            item.config,
            context.runtime,
        )
        for item in _checkpoint_admission_inputs(envelope)
    }


def _collect_synthetic_checkpoint_admissions(
    envelope: _ReportEnvelope,
    context: _ReportContext,
    checkpoint_admitter: CheckpointAdmitter,
) -> dict[int, CheckpointAdmission]:
    """Admit only explicitly synthetic checkpoints through a supplied test seam."""

    if (
        envelope.report.get("schema") != CAPACITY_SYNTHETIC_REPORT_SCHEMA
        or envelope.report.get("execution_class") != CAPACITY_SYNTHETIC_EXECUTION_CLASS
    ):
        raise CapacityBenchmarkError(
            "synthetic checkpoint admission requires synthetic report identity"
        )
    return {
        item.order: checkpoint_admitter(
            envelope.root,
            item.checkpoint,
            item.config,
            context.runtime,
        )
        for item in _checkpoint_admission_inputs(envelope)
    }


def _verify_report_cells(
    envelope: _ReportEnvelope,
    context: _ReportContext,
    admissions: Mapping[int, CheckpointAdmission],
) -> list[dict[str, Any]]:
    """Verify files and arithmetic against already observed checkpoint admissions."""

    report = envelope.report
    root = envelope.root
    raw_cells = _sequence("capacity cells", report["cells"])
    if len(raw_cells) != len(CAPACITY_ALGORITHM_ORDER):
        raise CapacityBenchmarkError("capacity report must contain exactly three cells")
    cells: list[dict[str, Any]] = []
    for expected_order, (expected_algorithm, raw_cell) in enumerate(
        zip(CAPACITY_ALGORITHM_ORDER, raw_cells, strict=True), start=1
    ):
        cell = dict(_mapping(f"capacity cell {expected_order}", raw_cell))
        _exact_keys(f"capacity cell {expected_order}", cell, _CELL_FIELDS)
        if (
            _integer("cell order", cell["order"], minimum=1) != expected_order
            or cell["algorithm"] != expected_algorithm
            or _integer("cell optimizer seed", cell["optimizer_seed"])
            != CAPACITY_OPTIMIZER_SEED
            or _integer("cell fixture seed base", cell["fixture_episode_seed_base"])
            != CAPACITY_FIXTURE_EPISODE_SEED_BASE
        ):
            raise CapacityBenchmarkError("capacity cell identity/order differs")
        layout = _cell_layout(expected_order, expected_algorithm)
        if (
            cell["config_path"] != layout["config"]
            or cell["output_path"] != layout["output"]
            or _mapping("capacity stdout", cell["stdout"]).get("path")
            != layout["stdout"]
            or _mapping("capacity stderr", cell["stderr"]).get("path")
            != layout["stderr"]
        ):
            raise CapacityBenchmarkError("capacity cell artifact layout differs")
        config_path = _resolve_report_path(root, cell["config_path"], expect_file=True)
        config_raw = _read_regular_bytes(config_path)
        if (
            _sha256("cell config sha256", cell["config_sha256"])
            != hashlib.sha256(config_raw).hexdigest()
        ):
            raise CapacityBenchmarkError("capacity cell config file digest differs")
        output_relative = _relative_path("cell output path", cell["output_path"])
        output = root.joinpath(*PurePosixPath(output_relative).parts)
        expected_config = _cell_config(output, expected_algorithm)
        if config_raw != _training_config_bytes(expected_config):
            raise CapacityBenchmarkError(
                "capacity cell config differs from frozen config"
            )
        _verify_log_record(root, cell["stdout"])
        _verify_log_record(root, cell["stderr"])
        wall_ns = _integer("cell wall elapsed ns", cell["wall_elapsed_ns"], minimum=1)
        _integer("cell child user CPU ns", cell["child_user_cpu_ns"])
        _integer("cell child system CPU ns", cell["child_system_cpu_ns"])
        if cell["status"] == "completed":
            if (
                _signed_integer("completed child exit code", cell["exit_code"]) != 0
                or cell["failure"] is not None
            ):
                raise CapacityBenchmarkError(
                    "completed capacity cell has failure state"
                )
            checkpoint_record = _verify_checkpoint_record(root, cell["checkpoint"])
            if checkpoint_record["path"] != layout["checkpoint"]:
                raise CapacityBenchmarkError("capacity checkpoint path differs")
            try:
                independently_admitted, rollout_ns, optimizer_steps = admissions[
                    expected_order
                ]
            except KeyError as reason:
                raise CapacityBenchmarkError(
                    "completed checkpoint was not independently admitted"
                ) from reason
            if independently_admitted != checkpoint_record:
                raise CapacityBenchmarkError("checkpoint changed after capacity report")
            if optimizer_steps != CAPACITY_OPTIMIZER_STEPS:
                raise CapacityBenchmarkError("capacity optimizer steps differ")
            measurement = _verify_measurement(cell["measurement"])
            if measurement != _build_measurement(wall_ns, rollout_ns):
                raise CapacityBenchmarkError("capacity measurement arithmetic differs")
            projection = _verify_projection(cell["projection"])
            if projection != _build_projection(measurement, checkpoint_record):
                raise CapacityBenchmarkError(
                    "capacity cell projection arithmetic differs"
                )
        elif cell["status"] == "failed":
            if expected_order in admissions:
                raise CapacityBenchmarkError(
                    "failed capacity cell has a checkpoint admission"
                )
            if (
                cell["checkpoint"] is not None
                or cell["measurement"] is not None
                or cell["projection"] is not None
                or not isinstance(cell["failure"], Mapping)
            ):
                raise CapacityBenchmarkError(
                    "failed capacity cell retained a projection"
                )
            failure = _verify_failure(cell["failure"])
            category = failure["category"]
            if category == "training_nonzero_exit":
                exit_code = _signed_integer("failed child exit code", cell["exit_code"])
                if exit_code == 0:
                    raise CapacityBenchmarkError(
                        "failed training process cannot have zero exit status"
                    )
            elif category in {"training_timeout", "training_launch_error"}:
                if cell["exit_code"] is not None:
                    raise CapacityBenchmarkError(
                        "failed launch or timeout must not report an exit status"
                    )
            elif category == "checkpoint_admission_error":
                if cell["exit_code"] != 0:
                    raise CapacityBenchmarkError(
                        "checkpoint admission requires child exit status zero"
                    )
            else:  # pragma: no cover - _verify_failure closes the vocabulary
                raise CapacityBenchmarkError(
                    "capacity failure category is not recognized"
                )
        else:
            raise CapacityBenchmarkError("capacity cell status is invalid")
        cells.append(cell)
    return cells


def _verify_production_result(
    report: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    expected_projection = _campaign_projection(cells)
    if expected_projection is None:
        if (
            report["status"] != "failed"
            or report["result_claim"] is not None
            or report["campaign_projection"] is not None
        ):
            raise CapacityBenchmarkError(
                "failed capacity report exposes a result claim"
            )
        return None
    if (
        report["status"] != "completed"
        or report["result_claim"] != CAPACITY_RESULT_CLAIM
    ):
        raise CapacityBenchmarkError("completed capacity report claim differs")
    projection = _verify_campaign_projection(report["campaign_projection"])
    if projection != expected_projection:
        raise CapacityBenchmarkError("campaign projection arithmetic differs")
    return projection


def _verify_synthetic_result(
    report: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
) -> None:
    expected_projection = _campaign_projection(cells)
    expected_status = "completed" if expected_projection is not None else "failed"
    if (
        report["status"] != expected_status
        or report["result_claim"] is not None
        or report["campaign_projection"] is not None
    ):
        raise CapacityBenchmarkError("synthetic report exposes a production result")


def _verify_deterministic_evidence(
    report: Mapping[str, Any],
    context: _ReportContext,
    *,
    claim_boundary: str,
    deterministic_domain: str,
) -> None:
    deterministic = _deterministic_evidence(
        protocol=context.protocol,
        runtime=context.runtime,
        host=context.host,
        cell_contract=context.contract,
        claim_boundary=claim_boundary,
    )
    if _sha256(
        "capacity deterministic evidence sha256",
        report["deterministic_evidence_sha256"],
    ) != domain_sha256(deterministic_domain, deterministic):
        raise CapacityBenchmarkError("capacity deterministic evidence digest differs")


def _verification_result(
    envelope: _ReportEnvelope,
) -> dict[str, Any]:
    return {
        "report": str(envelope.path),
        "raw_file_sha256": hashlib.sha256(
            _read_regular_bytes(envelope.path)
        ).hexdigest(),
        "report_sha256": envelope.supplied_sha256,
        "status": envelope.report["status"],
        "campaign_projection": envelope.report["campaign_projection"],
    }


def _verify_capacity_report(report_path: Path | str) -> dict[str, Any]:
    """Verify production evidence through fixed concrete dependencies."""

    envelope = _load_report_envelope(
        report_path,
        expected_filename=CAPACITY_REPORT_FILENAME,
        expected_schema=CAPACITY_REPORT_SCHEMA,
        expected_fields=_REPORT_FIELDS,
        report_domain=CAPACITY_REPORT_DOMAIN,
    )
    context = _verify_report_context(envelope, claim_boundary=CAPACITY_CLAIM)
    current_runtime = _clean_runtime_capture()
    verify_runtime_unchanged(context.runtime, current_runtime)
    current_host = _verify_host(_host_identity())
    if context.host != current_host:
        raise CapacityBenchmarkError("capacity report host differs from current host")
    admissions = _collect_production_checkpoint_admissions(envelope, context)
    cells = _verify_report_cells(envelope, context, admissions)
    _verify_production_result(envelope.report, cells)
    _verify_deterministic_evidence(
        envelope.report,
        context,
        claim_boundary=CAPACITY_CLAIM,
        deterministic_domain=CAPACITY_DETERMINISTIC_DOMAIN,
    )
    verify_runtime_unchanged(context.runtime, _clean_runtime_capture())
    return _verification_result(envelope)


def _verify_synthetic_capacity_report(
    report_path: Path | str,
    *,
    runtime_capture: RuntimeCapture,
    checkpoint_admitter: CheckpointAdmitter,
    host_capture: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify only explicitly synthetic, production-ineligible test evidence."""

    envelope = _load_report_envelope(
        report_path,
        expected_filename=CAPACITY_SYNTHETIC_REPORT_FILENAME,
        expected_schema=CAPACITY_SYNTHETIC_REPORT_SCHEMA,
        expected_fields=_SYNTHETIC_REPORT_FIELDS,
        report_domain=CAPACITY_SYNTHETIC_REPORT_DOMAIN,
    )
    if envelope.report["execution_class"] != CAPACITY_SYNTHETIC_EXECUTION_CLASS:
        raise CapacityBenchmarkError("synthetic execution class differs")
    context = _verify_report_context(
        envelope,
        claim_boundary=CAPACITY_SYNTHETIC_CLAIM,
    )
    current_runtime = runtime_capture()
    verify_runtime_unchanged(context.runtime, current_runtime)
    current_host = _verify_host(host_capture())
    if context.host != current_host:
        raise CapacityBenchmarkError("capacity report host differs from current host")
    admissions = _collect_synthetic_checkpoint_admissions(
        envelope,
        context,
        checkpoint_admitter,
    )
    cells = _verify_report_cells(envelope, context, admissions)
    _verify_synthetic_result(envelope.report, cells)
    _verify_deterministic_evidence(
        envelope.report,
        context,
        claim_boundary=CAPACITY_SYNTHETIC_CLAIM,
        deterministic_domain=CAPACITY_SYNTHETIC_DETERMINISTIC_DOMAIN,
    )
    verify_runtime_unchanged(context.runtime, runtime_capture())
    return _verification_result(envelope)


def verify_capacity_report(report_path: Path | str) -> dict[str, Any]:
    """Reopen a capacity report through the non-injectable production verifier."""

    return _verify_capacity_report(report_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m plant_backend.rl.v2.training.capacity_benchmark",
        description="Measure fixture-only constrained-RL campaign capacity.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run the three frozen fixture cells")
    run_parser.add_argument(
        "output", type=Path, help="new absolute external output root"
    )
    run_parser.add_argument(
        "--assert-exclusive",
        action="store_true",
        help="assert no other repository benchmark is intentionally running",
    )
    run_parser.add_argument(
        "--assert-first-attempt",
        action="store_true",
        help="assert this revision has no earlier capacity benchmark attempt",
    )
    verify_parser = subparsers.add_parser("verify", help="verify one capacity report")
    verify_parser.add_argument("report", type=Path, help="capacity report.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "run":
            result = execute_capacity_benchmark(
                arguments.output,
                assert_exclusive=arguments.assert_exclusive,
                assert_first_attempt=arguments.assert_first_attempt,
            )
        else:
            result = verify_capacity_report(arguments.report)
    except (
        checkpoints.CheckpointError,
        CapacityBenchmarkError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as reason:
        print(f"capacity benchmark refused: {reason}", file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CAPACITY_ALGORITHM_ORDER",
    "CAPACITY_FIXTURE_EPISODE_SEED_BASE",
    "CAPACITY_OPTIMIZER_SEED",
    "CAPACITY_PROTOCOL_ID",
    "CAPACITY_PROTOCOL_SHA256",
    "CAPACITY_REPAIR_CLARIFICATION_ID",
    "CAPACITY_REPAIR_CLARIFICATION_002_ID",
    "CAPACITY_REPAIR_CLARIFICATION_002_SHA256",
    "CAPACITY_REPAIR_CLARIFICATION_SHA256",
    "CAPACITY_REPAIR_ID",
    "CAPACITY_REPAIR_SHA256",
    "CAPACITY_REPORT_SCHEMA",
    "CapacityBenchmarkError",
    "build_parser",
    "execute_capacity_benchmark",
    "main",
    "verify_capacity_report",
]

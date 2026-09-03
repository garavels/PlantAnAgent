"""Provenance and publication boundary for feasible-planner evidence.

This module does not run the planner or solve its relaxation.  It captures the
two source/lock boundaries, records runtime identity, constructs the immutable
runner envelope, and performs the only permitted artifact write: an atomic,
external, reopen-and-revalidate publication.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import sys
import tempfile
from typing import Any, Literal

import gymnasium
import numpy
import scipy  # type: ignore[import-untyped]
from plant_sim import PlantConfig

from ...controllers.base import Constraints
from .config import EpisodeConfigV2
from .evaluation import collect_provenance
from .planner_schema import (
    SOURCE_BOUNDARY_CLAIM,
    CanonicalPayloadV1,
    FeasiblePlannerArtifactV1,
    FeasiblePlannerResultV1,
    PlannerRunnerFailureV1,
    canonical_json,
    planner_artifact_from_dict,
    runtime_payload,
    source_snapshot_payload,
)
from .reward import RewardWeightsV2
from .upper_bound import _linked_highs_version


RunnerPhase = Literal[
    "start_source",
    "child_execution",
    "end_source",
    "artifact_serialization",
]
RunnerCode = Literal[
    "source_unavailable",
    "wall_time_limit",
    "child_exit",
    "child_protocol_error",
    "serialization_error",
]
ProvenanceCollector = Callable[..., dict[str, Any]]


class PlannerEvidenceError(RuntimeError):
    """One runner-boundary failure with a schema-compatible phase and code."""

    def __init__(
        self,
        *,
        phase: RunnerPhase,
        code: RunnerCode,
        message: str,
        child_exit_code: int | None = None,
    ) -> None:
        if not isinstance(message, str) or not message:
            raise ValueError("planner evidence error message must be non-empty")
        allowed = {
            "start_source": {"source_unavailable"},
            "child_execution": {
                "wall_time_limit",
                "child_exit",
                "child_protocol_error",
            },
            "end_source": {"source_unavailable"},
            "artifact_serialization": {"serialization_error"},
        }
        if phase not in allowed:
            raise ValueError("unsupported planner evidence error phase")
        if code not in allowed[phase]:
            raise ValueError("planner evidence error phase/code are incompatible")
        if code == "child_exit" and child_exit_code is None:
            raise ValueError("child_exit errors require an exit code")
        if (
            code
            in {
                "source_unavailable",
                "wall_time_limit",
                "serialization_error",
            }
            and child_exit_code is not None
        ):
            raise ValueError("this planner evidence error cannot carry an exit code")
        super().__init__(message)
        self.phase = phase
        self.code = code
        self.child_exit_code = child_exit_code


@dataclass(frozen=True, slots=True)
class SourceBoundaryV1:
    """One complete source snapshot and raw dependency-lock identity."""

    source: CanonicalPayloadV1
    lock_sha256: str
    repository_root: Path

    def __post_init__(self) -> None:
        if type(self.source) is not CanonicalPayloadV1:
            raise TypeError("source must be an exact CanonicalPayloadV1")
        rebuilt_source = source_snapshot_payload(self.source.value())
        if rebuilt_source != self.source:
            raise ValueError("source snapshot does not match its canonical payload")
        if (
            not isinstance(self.lock_sha256, str)
            or len(self.lock_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.lock_sha256
            )
        ):
            raise ValueError("lock_sha256 must be lowercase SHA-256")
        root = Path(self.repository_root)
        if not root.is_absolute():
            raise ValueError("repository_root must be absolute")
        object.__setattr__(self, "repository_root", root.resolve())


def _exact_type(name: str, value: object, expected: type[object]) -> None:
    if type(value) is not expected:
        raise TypeError(f"{name} must have exact type {expected.__name__}")


def _source_value(raw: object) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise TypeError("provenance source must be an object")
    if raw.get("identity_complete") is not True:
        raise ValueError("source identity is incomplete")
    return {
        "schema_version": 1,
        "git_revision": raw.get("git_revision"),
        "git_dirty": raw.get("git_dirty"),
        "tracked_binary_diff_sha256": raw.get("tracked_binary_diff_sha256"),
        "tracked_binary_diff_bytes": raw.get("tracked_binary_diff_bytes"),
        "untracked_files": raw.get("untracked_files"),
        "worktree_identity_format": raw.get("worktree_identity_format"),
        "worktree_identity_sha256": raw.get("worktree_identity_sha256"),
        "identity_complete": True,
    }


def capture_source_boundary(
    *,
    episode: EpisodeConfigV2,
    plant: PlantConfig,
    constraints: Constraints,
    weights: RewardWeightsV2,
    root: Path | None = None,
    require_clean_source: bool,
    phase: Literal["start_source", "end_source"],
    collector: ProvenanceCollector = collect_provenance,
) -> SourceBoundaryV1:
    """Capture one complete source snapshot plus repository-root ``uv.lock``."""

    for name, value, expected in (
        ("episode", episode, EpisodeConfigV2),
        ("plant", plant, PlantConfig),
        ("constraints", constraints, Constraints),
        ("weights", weights, RewardWeightsV2),
    ):
        _exact_type(name, value, expected)
    if not isinstance(require_clean_source, bool):
        raise TypeError("require_clean_source must be a bool")
    if phase not in {"start_source", "end_source"}:
        raise ValueError("source boundary phase must be start_source or end_source")
    if root is not None:
        root = Path(root)
        if not root.is_absolute():
            raise ValueError("source root must be absolute")
        root = root.resolve()
    try:
        collected = collector(
            episode=episode,
            plant=plant,
            constraints=constraints,
            weights=weights,
            root=root,
        )
        if not isinstance(collected, Mapping):
            raise TypeError("provenance collector must return an object")
        raw_source = collected.get("source")
        value = _source_value(raw_source)
        source = source_snapshot_payload(value)
        parsed_source = source.value()
        assert isinstance(parsed_source, dict)
        if require_clean_source and parsed_source["git_dirty"] is not False:
            raise ValueError("registered planner execution requires a clean source")

        raw_lock = collected.get("dependency_lock")
        if not isinstance(raw_lock, Mapping):
            raise TypeError("provenance dependency_lock must be an object")
        repository_text = (
            raw_source.get("repository_root")
            if isinstance(raw_source, Mapping)
            else None
        )
        if not isinstance(repository_text, str) or not repository_text:
            raise ValueError("provenance repository root is unavailable")
        repository = Path(repository_text)
        if not repository.is_absolute():
            raise ValueError("provenance repository root must be absolute")
        repository = repository.resolve()
        if root is not None and repository != root:
            raise ValueError("provenance repository root differs from requested root")
        lock_path = repository / "uv.lock"
        lock_bytes = lock_path.read_bytes()
        lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
        if raw_lock.get("path") != "uv.lock" or raw_lock.get("sha256") != lock_sha256:
            raise ValueError(
                "dependency-lock identity disagrees with raw uv.lock bytes"
            )
        return SourceBoundaryV1(
            source=source,
            lock_sha256=lock_sha256,
            repository_root=repository,
        )
    except PlannerEvidenceError:
        raise
    except Exception as exc:
        raise PlannerEvidenceError(
            phase=phase,
            code="source_unavailable",
            message=str(exc) or type(exc).__name__,
        ) from exc


def capture_runtime_versions() -> CanonicalPayloadV1:
    """Capture the exact runtime fields required by the runner schema."""

    processor = platform.processor().strip()
    return runtime_payload(
        {
            "schema_version": 1,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_compiler": platform.python_compiler(),
            "numpy_version": numpy.__version__,
            "scipy_version": scipy.__version__,
            "highs_runtime_version": _linked_highs_version(),
            "gymnasium_version": gymnasium.__version__,
            "platform_system": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "platform_machine": platform.machine(),
            "platform_processor": processor or None,
            "pointer_bits": struct.calcsize("P") * 8,
            "byteorder": sys.byteorder,
            "clock_method": "time.perf_counter_ns",
            "rss_method": "resource.RUSAGE_SELF.ru_maxrss:darwin-bytes:other-kib",
        }
    )


def _runtime_matches_core(
    runtime_versions: CanonicalPayloadV1,
    core_result: FeasiblePlannerResultV1,
) -> None:
    runtime = runtime_versions.value()
    if not isinstance(runtime, dict):
        raise ValueError("runtime versions must contain an object")
    settings = core_result.settings
    if settings is not None and (
        runtime["clock_method"] != settings.clock_method
        or runtime["rss_method"] != settings.rss_method
    ):
        raise ValueError("runtime resource methods disagree with planner settings")
    relaxation = core_result.relaxation_result
    if relaxation is None:
        return
    result = relaxation.value()
    if not isinstance(result, dict):
        raise ValueError("relaxation result must contain an object")
    solve_settings = result.get("solve_settings")
    if not isinstance(solve_settings, dict):
        raise ValueError("relaxation result solve settings are missing")
    if solve_settings.get("highs_runtime_version") != runtime["highs_runtime_version"]:
        raise ValueError("parent and child HiGHS runtime identities disagree")


def _evidence_ready(
    *,
    core_result: FeasiblePlannerResultV1,
    start: SourceBoundaryV1,
    end: SourceBoundaryV1,
    runtime_versions: CanonicalPayloadV1,
) -> bool:
    comparison = core_result.comparison
    source = start.source.value()
    runtime = runtime_versions.value()
    settings = core_result.settings
    assert isinstance(source, dict)
    assert isinstance(runtime, dict)
    return bool(
        core_result.success
        and core_result.evidence_scope == "privileged_offline_development"
        and comparison is not None
        and comparison.bracket_status == "valid"
        and start.source.canonical_json == end.source.canonical_json
        and start.lock_sha256 == end.lock_sha256
        and source["git_dirty"] is False
        and source["untracked_files"] == []
        and settings is not None
        and runtime["clock_method"] == settings.clock_method
        and runtime["rss_method"] == settings.rss_method
    )


def build_completed_artifact(
    *,
    core_result: FeasiblePlannerResultV1,
    start: SourceBoundaryV1,
    end: SourceBoundaryV1,
    runtime_versions: CanonicalPayloadV1,
) -> FeasiblePlannerArtifactV1:
    """Wrap one validated child result with two complete provenance boundaries."""

    if type(core_result) is not FeasiblePlannerResultV1:
        raise TypeError("core_result must be an exact FeasiblePlannerResultV1")
    if type(start) is not SourceBoundaryV1 or type(end) is not SourceBoundaryV1:
        raise TypeError("source boundaries must be exact SourceBoundaryV1 records")
    if start.repository_root != end.repository_root:
        raise ValueError("source boundaries refer to different repositories")
    _runtime_matches_core(runtime_versions, core_result)
    snapshots_match = start.source.canonical_json == end.source.canonical_json
    return FeasiblePlannerArtifactV1(
        schema_version=1,
        success=core_result.success,
        evidence_ready=_evidence_ready(
            core_result=core_result,
            start=start,
            end=end,
            runtime_versions=runtime_versions,
        ),
        core_result=core_result,
        core_result_sha256=core_result.deterministic_result_sha256,
        runner_failure=None,
        start_source=start.source,
        end_source=end.source,
        start_lock_sha256=start.lock_sha256,
        end_lock_sha256=end.lock_sha256,
        runtime_versions=runtime_versions,
        source_boundary_claim=SOURCE_BOUNDARY_CLAIM if snapshots_match else None,
    )


def build_runner_failure_artifact(
    *,
    phase: RunnerPhase,
    code: RunnerCode,
    message: object,
    elapsed_ns: int,
    child_exit_code: int | None = None,
    core_result: FeasiblePlannerResultV1 | None = None,
    start: SourceBoundaryV1 | None = None,
    end: SourceBoundaryV1 | None = None,
    runtime_versions: CanonicalPayloadV1 | None = None,
) -> FeasiblePlannerArtifactV1:
    """Build the exact phase-dependent runner failure envelope."""

    failure = PlannerRunnerFailureV1.from_raw_message(
        phase=phase,
        code=code,
        message=message,
        elapsed_ns=elapsed_ns,
        child_exit_code=child_exit_code,
    )
    snapshots_match = (
        start is not None
        and end is not None
        and start.source.canonical_json == end.source.canonical_json
    )
    return FeasiblePlannerArtifactV1(
        schema_version=1,
        success=False,
        evidence_ready=False,
        core_result=core_result,
        core_result_sha256=(
            None if core_result is None else core_result.deterministic_result_sha256
        ),
        runner_failure=failure,
        start_source=None if start is None else start.source,
        end_source=None if end is None else end.source,
        start_lock_sha256=None if start is None else start.lock_sha256,
        end_lock_sha256=None if end is None else end.lock_sha256,
        runtime_versions=runtime_versions,
        source_boundary_claim=SOURCE_BOUNDARY_CLAIM if snapshots_match else None,
    )


def artifact_bytes(artifact: FeasiblePlannerArtifactV1) -> bytes:
    """Return canonical UTF-8 bytes after a full detached reconstruction."""

    if type(artifact) is not FeasiblePlannerArtifactV1:
        raise TypeError("artifact must be an exact FeasiblePlannerArtifactV1")
    rebuilt = planner_artifact_from_dict(artifact.to_dict())
    if rebuilt.artifact_sha256 != artifact.artifact_sha256:
        raise ValueError("rebuilt artifact identity differs")
    serialized = canonical_json(rebuilt.to_dict())
    if not isinstance(serialized, str):  # pragma: no cover - typed schema contract
        raise TypeError("canonical artifact serialization must be text")
    return serialized.encode("utf-8")


def _outside_repository(destination: Path, repository_root: Path) -> None:
    if not destination.is_absolute():
        raise ValueError("planner artifact destination must be absolute")
    root = repository_root.resolve()
    target = destination.resolve(strict=False)
    if target == root or target.is_relative_to(root):
        raise ValueError("planner artifacts must be written outside the repository")


def validate_artifact_destination(
    destination: Path,
    *,
    repository_root: Path,
) -> Path:
    """Validate publication location before a protected cell may execute."""

    target = Path(destination)
    repository = Path(repository_root)
    if not repository.is_absolute():
        raise ValueError("repository_root must be absolute")
    _outside_repository(target, repository)
    if not target.parent.is_dir():
        raise ValueError("planner artifact parent directory must already exist")
    return target.resolve(strict=False)


def write_planner_artifact(
    artifact: FeasiblePlannerArtifactV1,
    destination: Path,
    *,
    repository_root: Path,
) -> Path:
    """Atomically write externally, reopen, and fully revalidate one artifact."""

    repository = Path(repository_root)
    target = validate_artifact_destination(
        destination,
        repository_root=repository,
    )
    payload = artifact_bytes(artifact)
    parent = target.parent
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)

        persisted = target.read_bytes()
        if persisted != payload:
            raise ValueError("persisted planner artifact bytes differ from publication")
        decoded = persisted.decode("utf-8")
        parsed = json.loads(decoded)
        if decoded != canonical_json(parsed):
            raise ValueError("persisted planner artifact is not canonical JSON")
        rebuilt = planner_artifact_from_dict(parsed)
        if rebuilt.artifact_sha256 != artifact.artifact_sha256:
            raise ValueError("persisted planner artifact identity differs")
        return target
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


__all__ = [
    "PlannerEvidenceError",
    "SourceBoundaryV1",
    "artifact_bytes",
    "build_completed_artifact",
    "build_runner_failure_artifact",
    "capture_runtime_versions",
    "capture_source_boundary",
    "validate_artifact_destination",
    "write_planner_artifact",
]

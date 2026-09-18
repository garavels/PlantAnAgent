"""Fail-closed, outcome-free admission for the frozen M24 repair probe.

This module deliberately has no dependency on an environment, controller, or runner.  It
opens the already-public historical evidence, proves custody and identity, delegates the
semantic reconstruction to an injected deep verifier, and classifies the source delta before
returning any historical arm to an experiment runner.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, NoReturn, cast

from .economic_robustness import deterministic_arm_evidence
from .mpc_schema import (
    SERVICE_MPC_SETPOINT_CONTROLLER_ID,
    canonical_json,
    domain_sha256,
)
from .setpoint_controllers import LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID


HISTORICAL_EXECUTED_REVISION = "7b1b656964dcac4a25c6f5d4d89fed8abf4d4e0e"
HISTORICAL_CONTRACT_SHA256 = (
    "6f789629393188f406c929305721aa257e33e329fc3b5b85da9e5d62d77d7d38"
)
HISTORICAL_RAW_REPORT_SHA256 = (
    "26c9ab4a85693dab05d93f7a1b65e86564c74bfe667ee54813901a466d1e0755"
)
HISTORICAL_REPORT_DOMAIN_SHA256 = (
    "e0fe451861cda01f0a788adafed9bf2767f778d0141d7c76147b6eb0f3831f7f"
)
HISTORICAL_DETERMINISTIC_EVIDENCE_SHA256 = (
    "cff97bfc678f0b7ca47c516f3f7a80e39434aae83a6597576b1539eba5110b96"
)
HISTORICAL_WORKTREE_IDENTITY_SHA256 = (
    "eb34c2f8415f865ca6626a5e37e276441f3401e6d2c8cfdda3d4a13b82c5bbd6"
)
HISTORICAL_LOCK_SHA256 = (
    "080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd"
)
HISTORICAL_REPORT_STATUS = "complete_gate_failed"

REFERENCE_REPORT_FILENAME = "report.json"
REFERENCE_ARM_DETERMINISTIC_DOMAIN = "full-day-horizon-reference-arm-deterministic-v1"

PROTECTED_RUNTIME_PREFIXES = (
    "simulator/src/",
    "backend/src/plant_backend/controllers/",
    "backend/src/plant_backend/safety/",
    "backend/src/plant_backend/rl/v2/",
)
ALLOWED_NEW_M24_RUNTIME_PATHS = (
    "backend/src/plant_backend/rl/v2/full_day_horizon_codec.py",
    "backend/src/plant_backend/rl/v2/full_day_horizon_admission.py",
    "backend/src/plant_backend/rl/v2/full_day_horizon_repair.py",
    "backend/src/plant_backend/rl/v2/full_day_horizon_repair_experiment.py",
)
ALLOWED_NON_RUNTIME_PATHS = frozenset({"Makefile", "README.md"})
ALLOWED_NON_RUNTIME_PREFIXES = (
    "docs/",
    "backend/tests/",
    "simulator/tests/",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_DIFF_STATUS_RE = re.compile(r"(?:[AMDTUXB]|[RC][0-9]{1,3})")


def registered_reference_filenames() -> tuple[str, ...]:
    """Return the exact historical 20-cell filename sequence."""

    cell_ids = (
        f"h{horizon:g}-r{rate:g}-s{seed}"
        for horizon in (6.0, 24.0)
        for rate in (35.0, 42.0)
        for seed in (20, 21, 22, 23, 24)
    )
    return tuple(
        f"{index:02d}_{cell_id}.json" for index, cell_id in enumerate(cell_ids, start=1)
    )


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_revision(name: str, value: object) -> str:
    if not isinstance(value, str) or _REVISION_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 40-character Git revision")
    return value


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return cast(Mapping[str, Any], value)


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _validate_filename(name: str) -> str:
    if (
        not name
        or name in {".", ".."}
        or Path(name).name != name
        or PurePosixPath(name).name != name
        or "\\" in name
        or not name.endswith(".json")
    ):
        raise ValueError(f"reference filename is not a safe JSON basename: {name!r}")
    return name


@dataclass(frozen=True, slots=True)
class ReferenceIdentityV1:
    """Every identity frozen for the historical robustness directory."""

    executed_source_revision: str
    worktree_identity_sha256: str
    dependency_lock_sha256: str
    contract_sha256: str
    raw_report_sha256: str
    report_domain_sha256: str
    deterministic_evidence_sha256: str
    cell_filenames: tuple[str, ...]
    report_status: str = HISTORICAL_REPORT_STATUS

    def __post_init__(self) -> None:
        _require_revision(
            "historical executed source revision", self.executed_source_revision
        )
        for name, value in (
            ("historical worktree identity", self.worktree_identity_sha256),
            ("historical dependency lock", self.dependency_lock_sha256),
            ("historical contract", self.contract_sha256),
            ("historical raw report", self.raw_report_sha256),
            ("historical report-domain identity", self.report_domain_sha256),
            (
                "historical deterministic evidence",
                self.deterministic_evidence_sha256,
            ),
        ):
            _require_sha256(name, value)
        if not isinstance(self.cell_filenames, tuple) or any(
            not isinstance(filename, str) for filename in self.cell_filenames
        ):
            raise TypeError(
                "historical reference cell filenames must be a tuple of strings"
            )
        if len(self.cell_filenames) != 20:
            raise ValueError("historical reference must declare exactly 20 cell files")
        if len(set(self.cell_filenames)) != 20:
            raise ValueError("historical reference cell filenames must be unique")
        for filename in self.cell_filenames:
            _validate_filename(filename)
            if filename == REFERENCE_REPORT_FILENAME:
                raise ValueError(
                    "report.json cannot be declared as a historical cell file"
                )
        if not isinstance(self.report_status, str) or not self.report_status:
            raise ValueError("historical report status must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed_source_revision": self.executed_source_revision,
            "worktree_identity_sha256": self.worktree_identity_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "contract_sha256": self.contract_sha256,
            "raw_report_sha256": self.raw_report_sha256,
            "report_domain_sha256": self.report_domain_sha256,
            "deterministic_evidence_sha256": self.deterministic_evidence_sha256,
            "report_status": self.report_status,
            "cell_filenames": list(self.cell_filenames),
        }


REGISTERED_REFERENCE_IDENTITY = ReferenceIdentityV1(
    executed_source_revision=HISTORICAL_EXECUTED_REVISION,
    worktree_identity_sha256=HISTORICAL_WORKTREE_IDENTITY_SHA256,
    dependency_lock_sha256=HISTORICAL_LOCK_SHA256,
    contract_sha256=HISTORICAL_CONTRACT_SHA256,
    raw_report_sha256=HISTORICAL_RAW_REPORT_SHA256,
    report_domain_sha256=HISTORICAL_REPORT_DOMAIN_SHA256,
    deterministic_evidence_sha256=HISTORICAL_DETERMINISTIC_EVIDENCE_SHA256,
    cell_filenames=registered_reference_filenames(),
)


@dataclass(frozen=True, slots=True)
class SourceDiffEntryV1:
    """One normalized entry from Git's name/status comparison."""

    status: str
    path: str
    old_path: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.status, str)
            or _DIFF_STATUS_RE.fullmatch(self.status) is None
        ):
            raise ValueError(f"unsupported source-diff status: {self.status!r}")
        _validate_repository_path("source-diff path", self.path)
        is_pair = self.status.startswith(("R", "C"))
        if is_pair:
            if self.old_path is None:
                raise ValueError(f"{self.status} source-diff entry requires old_path")
            _validate_repository_path("source-diff old path", self.old_path)
            if self.old_path == self.path:
                raise ValueError("rename/copy source and destination paths must differ")
        elif self.old_path is not None:
            raise ValueError(f"{self.status} source-diff entry forbids old_path")

    @property
    def status_code(self) -> str:
        return self.status[0]

    def to_dict(self) -> dict[str, str]:
        body = {"status": self.status, "path": self.path}
        if self.old_path is not None:
            body["old_path"] = self.old_path
        return body

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> SourceDiffEntryV1:
        allowed = {"status", "path", "old_path"}
        if not {"status", "path"}.issubset(raw) or set(raw) - allowed:
            raise ValueError("source-diff entry fields differ from its schema")
        status = raw["status"]
        path = raw["path"]
        old_path = raw.get("old_path")
        if not isinstance(status, str) or not isinstance(path, str):
            raise TypeError("source-diff status and path must be strings")
        if old_path is not None and not isinstance(old_path, str):
            raise TypeError("source-diff old_path must be a string or null")
        return cls(
            status=status,
            path=path,
            old_path=old_path,
        )


@dataclass(frozen=True, slots=True)
class SourceDiffSnapshotV1:
    """Authenticated inputs supplied by the source comparison boundary."""

    base_revision: str
    target_revision: str
    worktree_clean: bool
    entries: tuple[SourceDiffEntryV1, ...]

    def __post_init__(self) -> None:
        _require_revision("source-diff base revision", self.base_revision)
        _require_revision("source-diff target revision", self.target_revision)
        if not isinstance(self.worktree_clean, bool):
            raise TypeError("source-diff worktree_clean must be bool")
        if not isinstance(self.entries, tuple) or any(
            not isinstance(item, SourceDiffEntryV1) for item in self.entries
        ):
            raise TypeError("source-diff entries must be SourceDiffEntryV1 values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_revision": self.base_revision,
            "target_revision": self.target_revision,
            "worktree_clean": self.worktree_clean,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> SourceDiffSnapshotV1:
        if set(raw) != {
            "base_revision",
            "target_revision",
            "worktree_clean",
            "entries",
        }:
            raise ValueError("source-diff snapshot fields differ from its schema")
        raw_entries = raw["entries"]
        if not isinstance(raw_entries, Sequence) or isinstance(
            raw_entries, (str, bytes, bytearray)
        ):
            raise TypeError("source-diff snapshot entries must be a sequence")
        entries = tuple(
            (
                item
                if isinstance(item, SourceDiffEntryV1)
                else SourceDiffEntryV1.from_mapping(_mapping("source-diff entry", item))
            )
            for item in raw_entries
        )
        clean = raw["worktree_clean"]
        if not isinstance(clean, bool):
            raise TypeError("source-diff worktree_clean must be bool")
        base = raw["base_revision"]
        target = raw["target_revision"]
        if not isinstance(base, str) or not isinstance(target, str):
            raise TypeError("source-diff revisions must be strings")
        return cls(
            base_revision=base,
            target_revision=target,
            worktree_clean=clean,
            entries=entries,
        )


@dataclass(frozen=True, slots=True)
class SourceDiffAdmissionV1:
    base_revision: str
    target_revision: str
    entries: tuple[SourceDiffEntryV1, ...]
    admitted_new_runtime_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_revision": self.base_revision,
            "target_revision": self.target_revision,
            "worktree_clean": True,
            "entries": [entry.to_dict() for entry in self.entries],
            "admitted_new_runtime_paths": list(self.admitted_new_runtime_paths),
        }


@dataclass(frozen=True, slots=True)
class HistoricalArmCitationV1:
    label: str
    controller_id: str
    deterministic_arm_sha256: str
    retained_evidence_sha256: str
    settings_sha256: str | None = None
    evidence_graph_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "controller_id": self.controller_id,
            "deterministic_arm_sha256": self.deterministic_arm_sha256,
            "retained_evidence_sha256": self.retained_evidence_sha256,
            "settings_sha256": self.settings_sha256,
            "evidence_graph_sha256": self.evidence_graph_sha256,
        }


@dataclass(frozen=True, slots=True)
class HistoricalCellCitationV1:
    index: int
    file_name: str
    cell_id: str
    horizon_h: float
    delivery_rate_kg_h: float
    seed: int
    raw_file_sha256: str
    cell_domain_sha256: str
    deterministic_evidence_sha256: str
    heuristic: HistoricalArmCitationV1
    m6: HistoricalArmCitationV1

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "file_name": self.file_name,
            "cell_id": self.cell_id,
            "horizon_h": self.horizon_h,
            "delivery_rate_kg_h": self.delivery_rate_kg_h,
            "seed": self.seed,
            "raw_file_sha256": self.raw_file_sha256,
            "cell_domain_sha256": self.cell_domain_sha256,
            "deterministic_evidence_sha256": self.deterministic_evidence_sha256,
            "arms": {
                "H": self.heuristic.to_dict(),
                "M6": self.m6.to_dict(),
            },
        }


@dataclass(frozen=True, slots=True)
class HistoricalReportCitationV1:
    file_name: str
    raw_file_sha256: str
    report_domain_sha256: str
    deterministic_evidence_sha256: str
    contract_sha256: str
    executed_source_revision: str
    worktree_identity_sha256: str
    dependency_lock_sha256: str
    status: str

    def to_dict(self) -> dict[str, str]:
        return {
            "file_name": self.file_name,
            "raw_file_sha256": self.raw_file_sha256,
            "report_domain_sha256": self.report_domain_sha256,
            "deterministic_evidence_sha256": self.deterministic_evidence_sha256,
            "contract_sha256": self.contract_sha256,
            "executed_source_revision": self.executed_source_revision,
            "worktree_identity_sha256": self.worktree_identity_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class AdmittedReferenceCellV1:
    """One selected full-day reference; bodies never enter serialized admission."""

    citation: HistoricalCellCitationV1
    reference_payload: Mapping[str, Any] = field(repr=False, compare=False)
    heuristic_arm: Mapping[str, Any] = field(repr=False, compare=False)
    m6_arm: Mapping[str, Any] = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return self.citation.to_dict()


@dataclass(frozen=True, slots=True)
class ReferenceAdmissionV1:
    """Complete pre-environment admission and in-memory historical runner inputs."""

    reference_directory: Path
    output_directory: Path
    report: HistoricalReportCitationV1
    cells: tuple[HistoricalCellCitationV1, ...]
    selected_cells: tuple[AdmittedReferenceCellV1, ...]
    source_diff: SourceDiffAdmissionV1
    _report_payload: Mapping[str, Any] = field(repr=False, compare=False)

    @property
    def report_payload(self) -> Mapping[str, Any]:
        return self._report_payload

    @property
    def selected_cells_by_id(self) -> dict[str, AdmittedReferenceCellV1]:
        return {cell.citation.cell_id: cell for cell in self.selected_cells}

    @property
    def runner_inputs_by_cell_id(
        self,
    ) -> dict[
        str,
        tuple[
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any],
            HistoricalCellCitationV1,
        ],
    ]:
        return {
            cell.citation.cell_id: (
                cell.reference_payload,
                cell.heuristic_arm,
                cell.m6_arm,
                cell.citation,
            )
            for cell in self.selected_cells
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the auditable compact form; never serialize historical arm bodies."""

        return {
            "reference_directory": str(self.reference_directory),
            "output_directory": str(self.output_directory),
            "report": self.report.to_dict(),
            "cells": [citation.to_dict() for citation in self.cells],
            "selected_cell_ids": [
                cell.citation.cell_id for cell in self.selected_cells
            ],
            "source_diff": self.source_diff.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    path: Path
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    raw_sha256: str


@dataclass(frozen=True, slots=True)
class _CanonicalDocument:
    payload: Mapping[str, Any]
    snapshot: _FileSnapshot


DeepReportVerifier = Callable[[Path], Mapping[str, Any]]
SourceDiffProvider = Callable[[Path, str], SourceDiffSnapshotV1 | Mapping[str, object]]


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number {value!r} is forbidden")
    return parsed


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r} is forbidden")
        result[key] = value
    return result


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )


def _read_regular_bytes(path: Path) -> tuple[bytes, _FileSnapshot]:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(f"symlink is forbidden in reference custody: {path}")
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"non-regular reference entry is forbidden: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"reference entry is no longer regular: {path}")
        if _stat_identity(before) != _stat_identity(opened):
            raise RuntimeError(f"reference entry changed while opening: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if _stat_identity(before) != _stat_identity(after):
        raise RuntimeError(f"reference entry changed while reading: {path}")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise RuntimeError(f"reference entry size changed while reading: {path}")
    digest = hashlib.sha256(raw).hexdigest()
    return raw, _FileSnapshot(
        path=path,
        device=before.st_dev,
        inode=before.st_ino,
        mode=before.st_mode,
        size=before.st_size,
        mtime_ns=before.st_mtime_ns,
        raw_sha256=digest,
    )


def _decode_canonical_mapping(path: Path, raw: bytes) -> Mapping[str, Any]:
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError(
            f"canonical JSON must have exactly one trailing newline: {path}"
        )
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as reason:
        raise ValueError(f"JSON is not strict UTF-8: {path}") from reason
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except (json.JSONDecodeError, OverflowError) as reason:
        raise ValueError(f"invalid strict JSON: {path}") from reason
    payload = _mapping(str(path), decoded)
    try:
        expected = (canonical_json(payload) + "\n").encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as reason:
        raise ValueError(f"JSON cannot be encoded canonically: {path}") from reason
    if raw != expected:
        raise ValueError(f"JSON bytes are not canonical: {path}")
    return payload


def read_canonical_json(path: Path | str) -> dict[str, Any]:
    """Read one regular file using the repository's exact canonical JSON encoding."""

    target = Path(path)
    raw, _ = _read_regular_bytes(target)
    return dict(_decode_canonical_mapping(target, raw))


def _read_canonical_document(path: Path) -> _CanonicalDocument:
    raw, snapshot = _read_regular_bytes(path)
    return _CanonicalDocument(
        payload=_decode_canonical_mapping(path, raw),
        snapshot=snapshot,
    )


def _validate_repository_path(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty POSIX repository path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must not be absolute or escape the repository")
    if str(path) != value:
        raise ValueError(f"{name} must use canonical POSIX spelling")
    return value


def _is_protected(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in PROTECTED_RUNTIME_PREFIXES)


def _is_allowed_non_runtime(path: str) -> bool:
    """Recognize the exact non-causal surfaces permitted by clarification 002."""

    return path in ALLOWED_NON_RUNTIME_PATHS or any(
        path.startswith(prefix) for prefix in ALLOWED_NON_RUNTIME_PREFIXES
    )


def validate_source_diff(
    snapshot: SourceDiffSnapshotV1,
    *,
    historical_revision: str = HISTORICAL_EXECUTED_REVISION,
    allowed_new_paths: Sequence[str] = ALLOWED_NEW_M24_RUNTIME_PATHS,
    require_all_allowed_new: bool = True,
) -> SourceDiffAdmissionV1:
    """Reject every causal-runtime change except the exact declared new M24 modules."""

    historical = _require_revision("historical revision", historical_revision)
    if snapshot.base_revision != historical:
        raise ValueError("source diff was not computed against the historical revision")
    if snapshot.worktree_clean is not True:
        raise ValueError("M24 admission requires a clean source worktree")
    allowed = tuple(
        _validate_repository_path("allowed new runtime path", path)
        for path in allowed_new_paths
    )
    if len(set(allowed)) != len(allowed):
        raise ValueError("allowed new runtime paths must be unique")
    if any(not _is_protected(path) for path in allowed):
        raise ValueError(
            "every allowed new runtime path must be under a protected surface"
        )

    seen_paths: set[str] = set()
    admitted: list[str] = []
    for entry in snapshot.entries:
        if entry.status_code in {"U", "X", "B"}:
            raise ValueError(
                f"unresolved source-diff status is forbidden: {entry.status} {entry.path}"
            )
        affected = {entry.path}
        if entry.old_path is not None:
            affected.add(entry.old_path)
        overlap = seen_paths.intersection(affected)
        if overlap:
            raise ValueError(
                f"source diff names a path more than once: {sorted(overlap)!r}"
            )
        seen_paths.update(affected)

        protected = [path for path in affected if _is_protected(path)]
        if not protected and all(_is_allowed_non_runtime(path) for path in affected):
            continue
        if not protected:
            raise ValueError(
                "undeclared source-diff path is forbidden by clarification 002: "
                f"{entry.status} {entry.path}"
            )
        if entry.status_code == "A" and entry.path in allowed:
            admitted.append(entry.path)
            continue
        if entry.status_code in {"M", "D"}:
            raise ValueError(
                f"historical protected runtime file was {entry.status_code}: "
                f"{entry.path}"
            )
        if entry.status_code == "R":
            raise ValueError(
                "renaming a historical protected runtime file is forbidden: "
                f"{entry.old_path} -> {entry.path}"
            )
        raise ValueError(
            "undeclared or non-additive protected runtime change is forbidden: "
            f"{entry.status} {entry.path}"
        )

    admitted_tuple = tuple(path for path in allowed if path in admitted)
    if require_all_allowed_new and admitted_tuple != allowed:
        missing = sorted(set(allowed) - set(admitted_tuple))
        raise ValueError(f"source diff omits declared M24 runtime modules: {missing!r}")
    return SourceDiffAdmissionV1(
        base_revision=snapshot.base_revision,
        target_revision=snapshot.target_revision,
        entries=snapshot.entries,
        admitted_new_runtime_paths=admitted_tuple,
    )


def validate_output_directory(
    output: Path | str,
    *,
    repository_root: Path | str,
    reference_directory: Path | str,
) -> Path:
    """Validate the exclusive external M24 destination without creating it."""

    raw = Path(output)
    if not raw.is_absolute():
        raise ValueError("M24 output directory must be absolute")
    if os.path.lexists(raw):
        raise FileExistsError(f"M24 output directory already exists: {raw}")
    if raw.parent.is_symlink():
        raise ValueError("M24 output parent directory must not be a symlink")
    if not raw.parent.is_dir():
        raise ValueError("M24 output parent directory must already exist")
    target = raw.resolve(strict=False)
    repository = Path(repository_root).resolve(strict=True)
    reference = Path(reference_directory).resolve(strict=True)
    for name, forbidden in (
        ("repository", repository),
        ("historical reference directory", reference),
    ):
        try:
            target.relative_to(forbidden)
        except ValueError:
            pass
        else:
            raise ValueError(f"M24 output directory must be outside the {name}")
    return target


def _scan_reference_directory(
    directory: Path,
    *,
    expected_names: frozenset[str],
) -> None:
    root_stat = directory.lstat()
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError(
            "historical reference directory must be a nonsymlink directory"
        )
    entries: dict[str, os.DirEntry[str]] = {}
    with os.scandir(directory) as iterator:
        for entry in iterator:
            if entry.name in entries:
                raise ValueError("historical reference directory repeats an entry name")
            entries[entry.name] = entry
    if len(entries) != 21 or set(entries) != expected_names:
        missing = sorted(expected_names - set(entries))
        extra = sorted(set(entries) - expected_names)
        raise ValueError(
            "historical reference directory must contain exactly the registered "
            f"21 entries; missing={missing!r}, extra={extra!r}"
        )
    for name, entry in entries.items():
        metadata = entry.stat(follow_symlinks=False)
        if entry.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(
                "historical reference entries must be regular nonsymlink files: "
                f"{name}"
            )


def _validate_reference_root(
    payload: Mapping[str, Any],
    *,
    raw_sha256: str,
    identity: ReferenceIdentityV1,
) -> HistoricalReportCitationV1:
    if raw_sha256 != identity.raw_report_sha256:
        raise ValueError("historical raw report identity differs from registration")
    if payload.get("sha256") != identity.report_domain_sha256:
        raise ValueError("historical report-domain identity differs from registration")
    if (
        payload.get("deterministic_evidence_sha256")
        != identity.deterministic_evidence_sha256
    ):
        raise ValueError(
            "historical deterministic evidence identity differs from registration"
        )
    if payload.get("contract_sha256") != identity.contract_sha256:
        raise ValueError("historical contract identity differs from registration")
    if payload.get("status") != identity.report_status:
        raise ValueError(
            "historical report status differs from registered retained evidence"
        )

    source_gate = _mapping("historical source gate", payload.get("source_gate"))
    if source_gate.get("status") != "passed":
        raise ValueError("historical source gate did not pass")
    for boundary_name in ("start", "end"):
        boundary = _mapping(
            f"historical {boundary_name} provenance", source_gate.get(boundary_name)
        )
        source = _mapping(f"historical {boundary_name} source", boundary.get("source"))
        lock = _mapping(
            f"historical {boundary_name} dependency lock",
            boundary.get("dependency_lock"),
        )
        if source.get("git_dirty") is not False:
            raise ValueError("historical report source was not clean")
        if source.get("identity_complete") is not True:
            raise ValueError("historical report source identity was incomplete")
        if source.get("git_revision") != identity.executed_source_revision:
            raise ValueError("historical executed source revision differs")
        if source.get("worktree_identity_sha256") != identity.worktree_identity_sha256:
            raise ValueError("historical worktree identity differs")
        if lock.get("sha256") != identity.dependency_lock_sha256:
            raise ValueError("historical dependency lock identity differs")

    return HistoricalReportCitationV1(
        file_name=REFERENCE_REPORT_FILENAME,
        raw_file_sha256=raw_sha256,
        report_domain_sha256=identity.report_domain_sha256,
        deterministic_evidence_sha256=identity.deterministic_evidence_sha256,
        contract_sha256=identity.contract_sha256,
        executed_source_revision=identity.executed_source_revision,
        worktree_identity_sha256=identity.worktree_identity_sha256,
        dependency_lock_sha256=identity.dependency_lock_sha256,
        status=identity.report_status,
    )


def _arm_citations(
    payload: Mapping[str, Any],
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    HistoricalArmCitationV1,
    HistoricalArmCitationV1,
]:
    arms = _mapping("historical cell arms", payload.get("arms"))
    if set(arms) != {"H", "M"}:
        raise ValueError("historical cell must contain exactly H and M arms")
    heuristic = _mapping("historical H arm", arms["H"])
    m6 = _mapping("historical M6 arm", arms["M"])
    if heuristic.get("arm") != "H" or m6.get("arm") != "M":
        raise ValueError("historical embedded arm labels differ")

    h_controller = _mapping("historical H controller", heuristic.get("controller"))
    m_controller = _mapping("historical M6 controller", m6.get("controller"))
    if h_controller.get("id") != LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID:
        raise ValueError("historical H controller identity differs")
    if m_controller.get("id") != SERVICE_MPC_SETPOINT_CONTROLLER_ID:
        raise ValueError("historical M6 controller identity differs")
    h_evidence = _mapping("historical H evidence", heuristic.get("evidence"))
    m_evidence = _mapping("historical M6 evidence", m6.get("evidence"))
    settings = _mapping("historical M6 settings", m_controller.get("settings"))
    h_evidence_sha = _require_sha256(
        "historical H retained evidence", h_evidence.get("sha256")
    )
    m_run_sha = _require_sha256("historical M6 run evidence", m_evidence.get("sha256"))
    m_graph_sha = _require_sha256(
        "historical M6 evidence graph", m_evidence.get("evidence_graph_sha256")
    )
    m_settings_sha = _require_sha256("historical M6 settings", settings.get("sha256"))
    if m_evidence.get("controller_id") != SERVICE_MPC_SETPOINT_CONTROLLER_ID:
        raise ValueError("historical M6 run controller identity differs")
    if m_evidence.get("settings_sha256") != m_settings_sha:
        raise ValueError("historical M6 run/settings identity differs")

    h_deterministic = domain_sha256(
        REFERENCE_ARM_DETERMINISTIC_DOMAIN,
        deterministic_arm_evidence(heuristic),
    )
    m_deterministic = domain_sha256(
        REFERENCE_ARM_DETERMINISTIC_DOMAIN,
        deterministic_arm_evidence(m6),
    )
    return (
        heuristic,
        m6,
        HistoricalArmCitationV1(
            label="H",
            controller_id=LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID,
            deterministic_arm_sha256=h_deterministic,
            retained_evidence_sha256=h_evidence_sha,
        ),
        HistoricalArmCitationV1(
            label="M6",
            controller_id=SERVICE_MPC_SETPOINT_CONTROLLER_ID,
            deterministic_arm_sha256=m_deterministic,
            retained_evidence_sha256=m_run_sha,
            settings_sha256=m_settings_sha,
            evidence_graph_sha256=m_graph_sha,
        ),
    )


def _cell_citation(
    *,
    position: int,
    filename: str,
    reference: Mapping[str, Any],
    document: _CanonicalDocument,
) -> tuple[HistoricalCellCitationV1, AdmittedReferenceCellV1]:
    if _integer("historical cell reference index", reference.get("index")) != position:
        raise ValueError("historical cell reference order differs")
    if reference.get("file_name") != filename:
        raise ValueError("historical cell reference filename differs")
    if reference.get("artifact_verified") is not True:
        raise ValueError("historical cell lacks a verified-artifact claim")
    if reference.get("verification_failure") is not None:
        raise ValueError("historical verified cell records a verification failure")

    payload = document.payload
    cell = _mapping("historical cell specification", payload.get("cell"))
    if reference.get("cell") != cell:
        raise ValueError("historical cell payload differs from its report reference")
    cell_id = cell.get("id")
    expected_id = filename.split("_", maxsplit=1)[1].removesuffix(".json")
    if not isinstance(cell_id, str) or cell_id != expected_id:
        raise ValueError("historical cell ID differs from its registered filename")
    if payload.get("status") != reference.get("status"):
        raise ValueError("historical cell status differs from its report reference")
    cell_sha = _require_sha256("historical cell-domain identity", payload.get("sha256"))
    deterministic_sha = _require_sha256(
        "historical cell deterministic evidence",
        payload.get("deterministic_evidence_sha256"),
    )
    if reference.get("raw_file_sha256") != document.snapshot.raw_sha256:
        raise ValueError(
            "historical cell raw identity differs from its report reference"
        )
    if reference.get("report_sha256") != cell_sha:
        raise ValueError(
            "historical cell-domain identity differs from its report reference"
        )
    if reference.get("deterministic_evidence_sha256") != deterministic_sha:
        raise ValueError(
            "historical cell deterministic identity differs from its report reference"
        )
    heuristic, m6, h_citation, m6_citation = _arm_citations(payload)

    horizon = cell.get("horizon_h")
    rate = cell.get("delivery_rate_kg_h")
    seed = cell.get("seed")
    if isinstance(horizon, bool) or not isinstance(horizon, (int, float)):
        raise TypeError("historical cell horizon must be numeric")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise TypeError("historical cell delivery rate must be numeric")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("historical cell seed must be an integer")
    if not math.isfinite(float(horizon)) or not math.isfinite(float(rate)):
        raise ValueError("historical cell axes must be finite")
    citation = HistoricalCellCitationV1(
        index=position,
        file_name=filename,
        cell_id=cell_id,
        horizon_h=float(horizon),
        delivery_rate_kg_h=float(rate),
        seed=seed,
        raw_file_sha256=document.snapshot.raw_sha256,
        cell_domain_sha256=cell_sha,
        deterministic_evidence_sha256=deterministic_sha,
        heuristic=h_citation,
        m6=m6_citation,
    )
    admitted = AdmittedReferenceCellV1(
        citation=citation,
        reference_payload=payload,
        heuristic_arm=heuristic,
        m6_arm=m6,
    )
    return citation, admitted


def _validate_deep_verification(
    result: object,
    *,
    report_path: Path,
    identity: ReferenceIdentityV1,
) -> None:
    verified = _mapping("deep historical report verification", result)
    expected = {
        "raw_file_sha256": identity.raw_report_sha256,
        "report_sha256": identity.report_domain_sha256,
        "deterministic_evidence_sha256": identity.deterministic_evidence_sha256,
        "status": identity.report_status,
        "cell_count": 20,
    }
    for key, value in expected.items():
        if verified.get(key) != value:
            raise ValueError(f"deep historical report verifier returned wrong {key}")
    if "path" in verified and Path(str(verified["path"])).resolve() != report_path:
        raise ValueError("deep historical report verifier returned a different path")


def _assert_snapshot_unchanged(snapshot: _FileSnapshot) -> None:
    raw, current = _read_regular_bytes(snapshot.path)
    del raw
    if current != snapshot:
        raise RuntimeError(f"admitted file changed during preflight: {snapshot.path}")


def _validate_repository_root(path: Path | str) -> Path:
    raw = Path(path)
    metadata = raw.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("repository root must be a nonsymlink directory")
    return raw.resolve(strict=True)


def _validate_current_lock(repository: Path, expected_sha256: str) -> _FileSnapshot:
    lock_path = repository / "uv.lock"
    _, snapshot = _read_regular_bytes(lock_path)
    if snapshot.raw_sha256 != expected_sha256:
        raise ValueError("current dependency lock differs from the frozen lock")
    return snapshot


def admit_full_day_horizon_probe(
    reference_report: Path | str,
    output: Path | str,
    repository_root: Path | str,
    deep_verifier: DeepReportVerifier,
    source_diff_provider: SourceDiffProvider,
    *,
    identity: ReferenceIdentityV1 = REGISTERED_REFERENCE_IDENTITY,
    allowed_new_paths: Sequence[str] = ALLOWED_NEW_M24_RUNTIME_PATHS,
) -> ReferenceAdmissionV1:
    """Admit the historical reference and new source before any M24 environment path.

    The injected verifier must deeply reconstruct the historical report.  The source provider
    must compare the requested historical revision to the clean candidate revision and return
    an authenticated :class:`SourceDiffSnapshotV1` (or its exact mapping form).
    """

    if not callable(deep_verifier) or not callable(source_diff_provider):
        raise TypeError("deep verifier and source-diff provider must be callable")
    repository = _validate_repository_root(repository_root)

    raw_report_path = Path(reference_report)
    if raw_report_path.name != REFERENCE_REPORT_FILENAME:
        raise ValueError("historical report must be named report.json")
    raw_reference_directory = raw_report_path.parent
    directory_metadata = raw_reference_directory.lstat()
    if stat.S_ISLNK(directory_metadata.st_mode) or not stat.S_ISDIR(
        directory_metadata.st_mode
    ):
        raise ValueError(
            "historical reference directory must be a nonsymlink directory"
        )
    reference_directory = raw_reference_directory.resolve(strict=True)
    report_path = reference_directory / REFERENCE_REPORT_FILENAME
    if raw_report_path.resolve(strict=True) != report_path:
        raise ValueError("historical report path escapes its reference directory")
    destination = validate_output_directory(
        output,
        repository_root=repository,
        reference_directory=reference_directory,
    )

    expected_names = frozenset((REFERENCE_REPORT_FILENAME, *identity.cell_filenames))
    _scan_reference_directory(
        reference_directory,
        expected_names=expected_names,
    )
    report_document = _read_canonical_document(report_path)
    report_citation = _validate_reference_root(
        report_document.payload,
        raw_sha256=report_document.snapshot.raw_sha256,
        identity=identity,
    )
    references = report_document.payload.get("cells")
    if not isinstance(references, list) or len(references) != 20:
        raise ValueError("historical report must cite exactly 20 cells")

    cell_citations: list[HistoricalCellCitationV1] = []
    selected: list[AdmittedReferenceCellV1] = []
    snapshots = [report_document.snapshot]
    for position, (filename, raw_reference) in enumerate(
        zip(identity.cell_filenames, references, strict=True), start=1
    ):
        reference = _mapping("historical cell reference", raw_reference)
        document = _read_canonical_document(reference_directory / filename)
        snapshots.append(document.snapshot)
        citation, admitted = _cell_citation(
            position=position,
            filename=filename,
            reference=reference,
            document=document,
        )
        cell_citations.append(citation)
        if position >= 11:
            if citation.horizon_h != 24.0:
                raise ValueError("selected historical positions must be full-day cells")
            selected.append(admitted)
    if len(selected) != 10 or len({cell.citation.cell_id for cell in selected}) != 10:
        raise ValueError("historical reference must yield ten unique full-day cells")

    _validate_deep_verification(
        deep_verifier(report_path),
        report_path=report_path,
        identity=identity,
    )
    lock_snapshot = _validate_current_lock(repository, identity.dependency_lock_sha256)
    raw_source_snapshot = source_diff_provider(
        repository, identity.executed_source_revision
    )
    source_snapshot = (
        raw_source_snapshot
        if isinstance(raw_source_snapshot, SourceDiffSnapshotV1)
        else SourceDiffSnapshotV1.from_mapping(
            _mapping("source-diff provider result", raw_source_snapshot)
        )
    )
    source_admission = validate_source_diff(
        source_snapshot,
        historical_revision=identity.executed_source_revision,
        allowed_new_paths=allowed_new_paths,
        require_all_allowed_new=True,
    )

    _scan_reference_directory(
        reference_directory,
        expected_names=expected_names,
    )
    for snapshot in snapshots:
        _assert_snapshot_unchanged(snapshot)
    _assert_snapshot_unchanged(lock_snapshot)

    return ReferenceAdmissionV1(
        reference_directory=reference_directory,
        output_directory=destination,
        report=report_citation,
        cells=tuple(cell_citations),
        selected_cells=tuple(selected),
        source_diff=source_admission,
        _report_payload=report_document.payload,
    )


__all__ = [
    "ALLOWED_NEW_M24_RUNTIME_PATHS",
    "AdmittedReferenceCellV1",
    "HISTORICAL_EXECUTED_REVISION",
    "HistoricalArmCitationV1",
    "HistoricalCellCitationV1",
    "HistoricalReportCitationV1",
    "PROTECTED_RUNTIME_PREFIXES",
    "REFERENCE_ARM_DETERMINISTIC_DOMAIN",
    "REGISTERED_REFERENCE_IDENTITY",
    "ReferenceAdmissionV1",
    "ReferenceIdentityV1",
    "SourceDiffAdmissionV1",
    "SourceDiffEntryV1",
    "SourceDiffSnapshotV1",
    "admit_full_day_horizon_probe",
    "read_canonical_json",
    "registered_reference_filenames",
    "validate_output_directory",
    "validate_source_diff",
]

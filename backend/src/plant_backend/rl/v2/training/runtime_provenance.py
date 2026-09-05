"""Strict runtime and source provenance for beta-policy training."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
from importlib import import_module
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import subprocess
from typing import Any, Mapping

import plant_sim


RUNTIME_PROVENANCE_SCHEMA = "beta-training-runtime-provenance-v1"
WORKTREE_IDENTITY_SCHEMA = "beta-training-worktree-content-v1"
DEPENDENCY_ENTRYPOINT_SCHEMA = "beta-training-dependency-entrypoints-v1"
GIT_TIMEOUT_SECONDS = 10.0
NOT_INSTALLED = "not-installed"
PLANT_SIM_MODULE_PATH = "simulator/src/plant_sim/__init__.py"

_STATUS_ARGUMENTS = (
    "status",
    "--porcelain=v1",
    "-z",
    "--untracked-files=all",
    "--ignore-submodules=none",
)

_TRACKED_DIFF_ARGUMENTS = (
    "diff",
    "--binary",
    "--full-index",
    "--no-color",
    "--no-ext-diff",
    "--no-textconv",
    "--no-renames",
    "--ignore-submodules=none",
    "--diff-algorithm=myers",
    "--src-prefix=a/",
    "--dst-prefix=b/",
    "HEAD",
    "--",
)
_UNTRACKED_ARGUMENTS = (
    "ls-files",
    "--others",
    "--exclude-standard",
    "-z",
)
_INDEX_FLAG_ARGUMENTS = ("ls-files", "-v", "-z")
_DEPENDENCIES = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("torch", "torch"),
    ("stable-baselines3", "stable_baselines3"),
    ("gymnasium", "gymnasium"),
)


class RuntimeProvenanceError(RuntimeError):
    """Runtime/source provenance could not be captured or admitted."""


class DirtyWorktreeError(RuntimeProvenanceError):
    """A clean-source capture found tracked or untracked changes."""


class RuntimeProvenanceChangedError(RuntimeProvenanceError):
    """Pre-run and post-run provenance are not identical."""


def _canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as reason:
        raise ValueError(
            f"runtime provenance is not canonical JSON: {reason}"
        ) from reason
    return hashlib.sha256(encoded).hexdigest()


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _sha256(name: str, value: object) -> str:
    digest = _string(name, value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return digest


def _git_head(value: object) -> str:
    revision = _string("git_head", value)
    if len(revision) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("git_head must be a full lowercase Git object id")
    return revision


def _relative_module_path(value: object) -> str:
    path = _string("plant_sim_module_path", value)
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != path:
        raise ValueError(
            "plant_sim_module_path must be canonical and repository-relative"
        )
    if parsed.suffix != ".py":
        raise ValueError("plant_sim_module_path must identify a Python module")
    if path != PLANT_SIM_MODULE_PATH:
        raise ValueError(f"plant_sim_module_path must be {PLANT_SIM_MODULE_PATH!r}")
    return path


@dataclass(frozen=True, slots=True)
class BetaTrainingRuntimeProvenanceV1:
    """Immutable source, dependency, interpreter, and numerical-runtime identity."""

    git_head: str
    worktree_clean: bool
    worktree_status_sha256: str
    worktree_content_sha256: str
    uv_lock_sha256: str
    dependency_entrypoints_sha256: str
    python_version: str
    python_implementation: str
    platform: str
    numpy_version: str
    scipy_version: str
    torch_version: str
    stable_baselines3_version: str
    gymnasium_version: str
    plant_sim_version: str
    plant_sim_module_path: str
    plant_sim_module_sha256: str

    def __post_init__(self) -> None:
        _git_head(self.git_head)
        if not isinstance(self.worktree_clean, bool):
            raise TypeError("worktree_clean must be a boolean")
        for name in (
            "worktree_status_sha256",
            "worktree_content_sha256",
            "uv_lock_sha256",
            "dependency_entrypoints_sha256",
            "plant_sim_module_sha256",
        ):
            _sha256(name, getattr(self, name))
        for name in (
            "python_version",
            "python_implementation",
            "platform",
            "numpy_version",
            "scipy_version",
            "torch_version",
            "stable_baselines3_version",
            "gymnasium_version",
            "plant_sim_version",
        ):
            version = _string(name, getattr(self, name))
            if name.endswith("_version") and version == NOT_INSTALLED:
                raise ValueError(f"{name} is required for beta training")
        _relative_module_path(self.plant_sim_module_path)
        status_is_empty = self.worktree_status_sha256 == hashlib.sha256(b"").hexdigest()
        content_is_empty = (
            self.worktree_content_sha256 == _empty_worktree_content_sha256()
        )
        if (
            self.worktree_clean != status_is_empty
            or self.worktree_clean != content_is_empty
        ):
            raise ValueError(
                "worktree_clean disagrees with the canonical status/content identity"
            )

    def body(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_PROVENANCE_SCHEMA,
            **{field.name: getattr(self, field.name) for field in fields(self)},
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: object) -> BetaTrainingRuntimeProvenanceV1:
        raw = _mapping("runtime provenance", value)
        field_names = {field.name for field in fields(cls)}
        expected = {"schema", "sha256", *field_names}
        missing = expected - set(raw)
        unknown = set(raw) - expected
        if missing:
            raise ValueError(
                "runtime provenance is missing fields: " + ", ".join(sorted(missing))
            )
        if unknown:
            raise ValueError(
                "runtime provenance has unknown fields: " + ", ".join(sorted(unknown))
            )
        if raw["schema"] != RUNTIME_PROVENANCE_SCHEMA:
            raise ValueError("runtime provenance schema differs")
        clean = raw["worktree_clean"]
        if not isinstance(clean, bool):
            raise TypeError("worktree_clean must be a boolean")
        result = cls(
            git_head=_git_head(raw["git_head"]),
            worktree_clean=clean,
            worktree_status_sha256=_sha256(
                "worktree_status_sha256", raw["worktree_status_sha256"]
            ),
            worktree_content_sha256=_sha256(
                "worktree_content_sha256", raw["worktree_content_sha256"]
            ),
            uv_lock_sha256=_sha256("uv_lock_sha256", raw["uv_lock_sha256"]),
            dependency_entrypoints_sha256=_sha256(
                "dependency_entrypoints_sha256",
                raw["dependency_entrypoints_sha256"],
            ),
            python_version=_string("python_version", raw["python_version"]),
            python_implementation=_string(
                "python_implementation", raw["python_implementation"]
            ),
            platform=_string("platform", raw["platform"]),
            numpy_version=_string("numpy_version", raw["numpy_version"]),
            scipy_version=_string("scipy_version", raw["scipy_version"]),
            torch_version=_string("torch_version", raw["torch_version"]),
            stable_baselines3_version=_string(
                "stable_baselines3_version", raw["stable_baselines3_version"]
            ),
            gymnasium_version=_string("gymnasium_version", raw["gymnasium_version"]),
            plant_sim_version=_string("plant_sim_version", raw["plant_sim_version"]),
            plant_sim_module_path=_relative_module_path(raw["plant_sim_module_path"]),
            plant_sim_module_sha256=_sha256(
                "plant_sim_module_sha256", raw["plant_sim_module_sha256"]
            ),
        )
        supplied = _sha256("runtime provenance sha256", raw["sha256"])
        if result.sha256 != supplied:
            raise ValueError("runtime provenance digest differs")
        return result


def _repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / ".git").exists()
            and (candidate / "pyproject.toml").is_file()
            and (candidate / "uv.lock").is_file()
        ):
            return candidate
    raise RuntimeProvenanceError(
        "beta training runtime provenance requires a repository checkout"
    )


def _run_git(repository: Path, *arguments: str) -> bytes:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LANG": "C",
            "LC_ALL": "C",
        }
    )
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repository,
            env=environment,
            check=True,
            capture_output=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as reason:
        raise RuntimeProvenanceError(
            f"cannot capture Git runtime provenance: {reason}"
        ) from reason
    return completed.stdout


def _regular_file_snapshot(path: Path, *, label: str) -> tuple[bytes, int]:
    """Read one stable regular file and retain its executable permission bits."""

    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        lexical = path.lstat()
        if (
            stat.S_ISLNK(lexical.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or (lexical.st_dev, lexical.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise RuntimeProvenanceError(f"{label} must be a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError as reason:
        raise RuntimeProvenanceError(
            f"cannot read {label} at {path}: {reason}"
        ) from reason
    finally:
        if descriptor is not None:
            os.close(descriptor)
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, name) != getattr(after, name) for name in identity_fields):
        raise RuntimeProvenanceError(
            f"{label} changed while it was being captured: {path}"
        )
    if len(content) != after.st_size:
        raise RuntimeProvenanceError(
            f"{label} byte count changed while it was being captured: {path}"
        )
    return content, stat.S_IMODE(after.st_mode)


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    return _regular_file_snapshot(path, label=label)[0]


def _digest_segment(digest: Any, value: bytes) -> None:
    """Append bytes with an unambiguous fixed-width length prefix."""

    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def _worktree_digest(*, status: bytes, tracked_diff: bytes) -> Any:
    digest = hashlib.sha256()
    _digest_segment(digest, WORKTREE_IDENTITY_SCHEMA.encode("ascii"))
    _digest_segment(digest, status)
    _digest_segment(digest, tracked_diff)
    return digest


def _empty_worktree_content_sha256() -> str:
    return _worktree_digest(status=b"", tracked_diff=b"").hexdigest()


def _untracked_paths(raw: bytes) -> tuple[bytes, ...]:
    if raw and not raw.endswith(b"\x00"):
        raise RuntimeProvenanceError("Git returned a truncated untracked-file listing")
    paths = tuple(sorted(item for item in raw.split(b"\x00") if item))
    if len(paths) != len(set(paths)):
        raise RuntimeProvenanceError("Git returned duplicate untracked-file paths")
    return paths


def _worktree_content_sha256(repository: Path, *, status: bytes) -> str:
    """Bind status, exact tracked diff, and every untracked regular-file byte."""

    tracked_diff = _run_git(repository, *_TRACKED_DIFF_ARGUMENTS)
    listed = _run_git(repository, *_UNTRACKED_ARGUMENTS)
    digest = _worktree_digest(status=status, tracked_diff=tracked_diff)
    for raw_path in _untracked_paths(listed):
        relative = os.fsdecode(raw_path)
        parsed = PurePosixPath(relative)
        if (
            parsed.is_absolute()
            or not parsed.parts
            or "." in parsed.parts
            or ".." in parsed.parts
            or parsed.as_posix() != relative
        ):
            raise RuntimeProvenanceError(
                "Git returned a non-canonical untracked-file path"
            )
        path = repository.joinpath(*parsed.parts)
        try:
            resolved = path.resolve(strict=True)
        except OSError as reason:
            raise RuntimeProvenanceError(
                f"cannot resolve untracked file at {path}: {reason}"
            ) from reason
        if not resolved.is_relative_to(repository):
            raise RuntimeProvenanceError(
                f"untracked file resolves outside the repository: {path}"
            )
        if resolved != path:
            raise RuntimeProvenanceError(
                f"untracked file path must not contain symlinks: {path}"
            )
        content, mode = _regular_file_snapshot(path, label="untracked source file")
        _digest_segment(digest, raw_path)
        _digest_segment(digest, f"{mode:04o}".encode("ascii"))
        _digest_segment(digest, content)
    return digest.hexdigest()


def _distribution_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return NOT_INSTALLED


def _dependency_versions_and_entrypoints() -> tuple[dict[str, str], str]:
    """Cross-check metadata against imported modules and bind their entrypoints."""

    versions: dict[str, str] = {}
    entrypoints: list[dict[str, Any]] = []
    for distribution_name, module_name in _DEPENDENCIES:
        try:
            distribution = importlib_metadata.distribution(distribution_name)
        except importlib_metadata.PackageNotFoundError as reason:
            raise RuntimeProvenanceError(
                f"required beta-training dependency is not installed: {distribution_name}"
            ) from reason
        distribution_version = distribution.version
        try:
            module = import_module(module_name)
        except (ImportError, OSError) as reason:
            raise RuntimeProvenanceError(
                f"cannot import beta-training dependency {module_name}: {reason}"
            ) from reason
        imported_version = getattr(module, "__version__", None)
        if not isinstance(imported_version, str) or not imported_version.strip():
            raise RuntimeProvenanceError(
                f"imported dependency {module_name} has no string __version__"
            )
        if imported_version != distribution_version:
            raise RuntimeProvenanceError(
                f"imported {module_name} version {imported_version!r} differs from "
                f"distribution {distribution_name} version {distribution_version!r}"
            )
        origin_value = getattr(module, "__file__", None)
        spec = getattr(module, "__spec__", None)
        spec_origin = getattr(spec, "origin", None)
        if not isinstance(origin_value, str) or not isinstance(spec_origin, str):
            raise RuntimeProvenanceError(
                f"imported dependency {module_name} has no filesystem origin"
            )
        origin = Path(os.path.abspath(origin_value))
        declared_origin = Path(os.path.abspath(spec_origin))
        if origin != declared_origin:
            raise RuntimeProvenanceError(
                f"imported dependency {module_name} origin disagrees with its spec"
            )
        try:
            resolved_origin = origin.resolve(strict=True)
            distribution_root = Path(str(distribution.locate_file(""))).resolve(
                strict=True
            )
        except OSError as reason:
            raise RuntimeProvenanceError(
                f"cannot resolve imported dependency {module_name}: {reason}"
            ) from reason
        if not resolved_origin.is_relative_to(distribution_root):
            raise RuntimeProvenanceError(
                f"imported dependency {module_name} is outside its distribution root"
            )
        content, mode = _regular_file_snapshot(
            origin,
            label=f"{module_name} module entrypoint",
        )
        versions[module_name] = distribution_version
        entrypoints.append(
            {
                "distribution": distribution_name,
                "module": module_name,
                "version": distribution_version,
                "origin": str(origin),
                "distribution_root": str(distribution_root),
                "mode": f"{mode:04o}",
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return versions, _canonical_sha256(
        {
            "schema": DEPENDENCY_ENTRYPOINT_SCHEMA,
            "entrypoints": entrypoints,
        }
    )


def _decode_git_head(raw: bytes) -> str:
    try:
        head = raw.strip().decode("ascii")
    except UnicodeDecodeError as reason:
        raise RuntimeProvenanceError("Git HEAD is not ASCII") from reason
    try:
        return _git_head(head)
    except (TypeError, ValueError) as reason:
        raise RuntimeProvenanceError(f"Git HEAD is malformed: {reason}") from reason


def _verify_git_repository(repository: Path) -> None:
    raw = _run_git(repository, "rev-parse", "--show-toplevel")
    try:
        reported = Path(os.fsdecode(raw.rstrip(b"\r\n"))).resolve(strict=True)
    except OSError as reason:
        raise RuntimeProvenanceError(
            f"cannot resolve Git repository root: {reason}"
        ) from reason
    if reported != repository:
        raise RuntimeProvenanceError(
            f"Git resolved {reported}, expected repository {repository}"
        )
    index = _run_git(repository, "ls-files", "--stage", "-z")
    if any(entry.startswith(b"160000 ") for entry in index.split(b"\x00") if entry):
        raise RuntimeProvenanceError(
            "beta-training source provenance does not admit Git submodules"
        )
    flagged_index = _run_git(repository, *_INDEX_FLAG_ARGUMENTS)
    for entry in (item for item in flagged_index.split(b"\x00") if item):
        marker = entry[:1]
        if marker == b"S" or (b"a" <= marker <= b"z"):
            raise RuntimeProvenanceError(
                "beta-training source provenance forbids assume-unchanged and "
                "skip-worktree index flags"
            )


def _plant_sim_path(repository: Path, module_value: object) -> Path:
    if not isinstance(module_value, str):
        raise RuntimeProvenanceError("plant_sim has no filesystem module path")
    module_path = Path(os.path.abspath(module_value))
    expected_module_path = repository / PLANT_SIM_MODULE_PATH
    if module_path != expected_module_path:
        raise RuntimeProvenanceError(
            f"plant_sim module path {module_path} differs from {expected_module_path}"
        )
    try:
        resolved_module_path = module_path.resolve(strict=True)
    except OSError as reason:
        raise RuntimeProvenanceError(
            f"cannot resolve plant_sim module path: {reason}"
        ) from reason
    if resolved_module_path != module_path:
        raise RuntimeProvenanceError("plant_sim module path must not contain symlinks")
    return module_path


def capture(*, require_clean: bool) -> BetaTrainingRuntimeProvenanceV1:
    """Capture the current beta-training runtime, optionally requiring clean Git."""

    if not isinstance(require_clean, bool):
        raise TypeError("require_clean must be a boolean")
    try:
        repository = _repository_root().resolve(strict=True)
    except OSError as reason:
        raise RuntimeProvenanceError(
            f"cannot resolve beta-training repository: {reason}"
        ) from reason
    _verify_git_repository(repository)
    head = _decode_git_head(_run_git(repository, "rev-parse", "--verify", "HEAD"))
    status_before = _run_git(repository, *_STATUS_ARGUMENTS)
    if require_clean and status_before:
        raise DirtyWorktreeError(
            "registered beta training requires a clean tracked and untracked worktree"
        )
    content_before = _worktree_content_sha256(repository, status=status_before)

    lock_bytes = _regular_file_bytes(repository / "uv.lock", label="uv.lock")
    module_path = _plant_sim_path(repository, getattr(plant_sim, "__file__", None))
    module_bytes = _regular_file_bytes(module_path, label="plant_sim module")
    plant_sim_version = getattr(plant_sim, "__version__", None)
    if not isinstance(plant_sim_version, str) or not plant_sim_version.strip():
        raise RuntimeProvenanceError("plant_sim has no string __version__")
    plant_sim_distribution_version = _distribution_version("plant-sim")
    if plant_sim_distribution_version == NOT_INSTALLED:
        raise RuntimeProvenanceError("plant-sim distribution metadata is not installed")
    if plant_sim_version != plant_sim_distribution_version:
        raise RuntimeProvenanceError(
            f"plant_sim source version {plant_sim_version!r} differs from installed "
            f"distribution version {plant_sim_distribution_version!r}"
        )
    dependency_versions, dependency_entrypoints_sha256 = (
        _dependency_versions_and_entrypoints()
    )
    python_version = platform.python_version()
    python_implementation = platform.python_implementation()
    platform_identity = platform.platform(aliased=False, terse=False)

    ending_head = _decode_git_head(
        _run_git(repository, "rev-parse", "--verify", "HEAD")
    )
    status = _run_git(repository, *_STATUS_ARGUMENTS)
    content_sha256 = _worktree_content_sha256(repository, status=status)
    _verify_git_repository(repository)
    if (
        head != ending_head
        or status_before != status
        or content_before != content_sha256
    ):
        raise RuntimeProvenanceError(
            "Git source changed while runtime provenance was being captured"
        )
    clean = not status

    return BetaTrainingRuntimeProvenanceV1(
        git_head=head,
        worktree_clean=clean,
        worktree_status_sha256=hashlib.sha256(status).hexdigest(),
        worktree_content_sha256=content_sha256,
        uv_lock_sha256=hashlib.sha256(lock_bytes).hexdigest(),
        dependency_entrypoints_sha256=dependency_entrypoints_sha256,
        python_version=python_version,
        python_implementation=python_implementation,
        platform=platform_identity,
        numpy_version=dependency_versions["numpy"],
        scipy_version=dependency_versions["scipy"],
        torch_version=dependency_versions["torch"],
        stable_baselines3_version=dependency_versions["stable_baselines3"],
        gymnasium_version=dependency_versions["gymnasium"],
        plant_sim_version=plant_sim_version,
        plant_sim_module_path=PLANT_SIM_MODULE_PATH,
        plant_sim_module_sha256=hashlib.sha256(module_bytes).hexdigest(),
    )


def verify_unchanged(
    before: BetaTrainingRuntimeProvenanceV1,
    after: BetaTrainingRuntimeProvenanceV1,
) -> None:
    """Fail unless pre-run and post-run provenance are exactly identical."""

    if not isinstance(before, BetaTrainingRuntimeProvenanceV1) or not isinstance(
        after, BetaTrainingRuntimeProvenanceV1
    ):
        raise TypeError("before and after must be runtime provenance records")
    if before == after:
        return
    changed = [
        field.name
        for field in fields(BetaTrainingRuntimeProvenanceV1)
        if getattr(before, field.name) != getattr(after, field.name)
    ]
    raise RuntimeProvenanceChangedError(
        "beta training runtime provenance changed: " + ", ".join(changed)
    )


__all__ = [
    "BetaTrainingRuntimeProvenanceV1",
    "DEPENDENCY_ENTRYPOINT_SCHEMA",
    "DirtyWorktreeError",
    "GIT_TIMEOUT_SECONDS",
    "NOT_INSTALLED",
    "PLANT_SIM_MODULE_PATH",
    "RUNTIME_PROVENANCE_SCHEMA",
    "WORKTREE_IDENTITY_SCHEMA",
    "RuntimeProvenanceChangedError",
    "RuntimeProvenanceError",
    "capture",
    "verify_unchanged",
]

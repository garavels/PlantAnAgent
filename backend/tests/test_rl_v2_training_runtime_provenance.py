"""Strict runtime-provenance tests for beta controller training."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import hashlib
from importlib import metadata as importlib_metadata
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

from plant_backend.rl.v2.training import runtime_provenance as provenance


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _record() -> provenance.BetaTrainingRuntimeProvenanceV1:
    return provenance.BetaTrainingRuntimeProvenanceV1(
        git_head="a" * 40,
        worktree_clean=True,
        worktree_status_sha256=_sha(""),
        worktree_content_sha256=provenance._empty_worktree_content_sha256(),
        uv_lock_sha256=_sha("lock"),
        dependency_entrypoints_sha256=_sha("dependency-entrypoints"),
        python_version="3.12.11",
        python_implementation="CPython",
        platform="macOS-15-arm64",
        numpy_version="2.2.6",
        scipy_version="1.15.3",
        torch_version="2.7.1",
        stable_baselines3_version="2.9.0",
        gymnasium_version="1.2.0",
        plant_sim_version="0.5.2",
        plant_sim_module_path="simulator/src/plant_sim/__init__.py",
        plant_sim_module_sha256=_sha("plant-sim-module"),
    )


def _repository() -> Path:
    return Path(provenance.__file__).resolve().parents[6]


def _fake_git(
    status: bytes,
    *,
    tracked_diff: bytes = b"",
    untracked: bytes = b"",
):
    def run(repository: Path, *arguments: str) -> bytes:
        assert repository == _repository()
        if arguments == ("rev-parse", "--show-toplevel"):
            return os.fsencode(repository) + b"\n"
        if arguments == ("ls-files", "--stage", "-z"):
            return b""
        if arguments == provenance._INDEX_FLAG_ARGUMENTS:
            return b"H tracked.py\x00"
        if arguments == ("rev-parse", "--verify", "HEAD"):
            return b"b" * 40 + b"\n"
        if arguments == provenance._STATUS_ARGUMENTS:
            return status
        if arguments == provenance._TRACKED_DIFF_ARGUMENTS:
            return tracked_diff
        if arguments == provenance._UNTRACKED_ARGUMENTS:
            return untracked
        raise AssertionError(f"unexpected Git arguments: {arguments!r}")

    return run


def test_canonical_round_trip_is_deterministic_and_immutable() -> None:
    original = _record()
    encoded = original.to_dict()
    reopened = provenance.BetaTrainingRuntimeProvenanceV1.from_dict(encoded)

    assert reopened == original
    assert reopened.to_dict() == encoded
    assert reopened.sha256 == original.sha256
    with pytest.raises(FrozenInstanceError):
        reopened.git_head = "c" * 40  # type: ignore[misc]


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "sha256",
        *[item.name for item in fields(provenance.BetaTrainingRuntimeProvenanceV1)],
    ],
)
def test_from_dict_rejects_every_missing_required_field(field: str) -> None:
    payload = _record().to_dict()
    payload.pop(field)

    with pytest.raises(ValueError, match="missing fields"):
        provenance.BetaTrainingRuntimeProvenanceV1.from_dict(payload)


def test_from_dict_rejects_unknown_field() -> None:
    payload = {**_record().to_dict(), "undeclared_runtime": "1.0"}

    with pytest.raises(ValueError, match="unknown fields"):
        provenance.BetaTrainingRuntimeProvenanceV1.from_dict(payload)


def test_from_dict_rejects_tampering_and_non_boolean_clean_flag() -> None:
    tampered = _record().to_dict()
    tampered["platform"] = "different-platform"
    with pytest.raises(ValueError, match="digest differs"):
        provenance.BetaTrainingRuntimeProvenanceV1.from_dict(tampered)

    wrong_type = _record().to_dict()
    wrong_type["worktree_clean"] = 1
    with pytest.raises(TypeError, match="worktree_clean"):
        provenance.BetaTrainingRuntimeProvenanceV1.from_dict(wrong_type)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"worktree_clean": False}, "worktree_clean disagrees"),
        (
            {"worktree_status_sha256": _sha("dirty status")},
            "worktree_clean disagrees",
        ),
        (
            {"worktree_content_sha256": _sha("dirty content")},
            "worktree_clean disagrees",
        ),
        ({"torch_version": provenance.NOT_INSTALLED}, "required for beta training"),
        (
            {"plant_sim_module_path": "shadow_plant_sim.py"},
            "plant_sim_module_path must be",
        ),
    ],
)
def test_record_rejects_semantically_impossible_identity(
    updates: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        replace(_record(), **updates)


def test_capture_records_a_clean_repository_and_complete_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance, "_repository_root", _repository)
    monkeypatch.setattr(provenance, "_run_git", _fake_git(b""))
    captured = provenance.capture(require_clean=True)

    assert captured.git_head == "b" * 40
    assert captured.worktree_clean is True
    assert captured.worktree_status_sha256 == hashlib.sha256(b"").hexdigest()
    assert (
        captured.worktree_content_sha256 == provenance._empty_worktree_content_sha256()
    )
    assert captured.plant_sim_module_path == provenance.PLANT_SIM_MODULE_PATH
    assert captured.plant_sim_version == importlib_metadata.version("plant-sim")
    assert (
        captured.plant_sim_module_sha256
        == hashlib.sha256(
            (_repository() / captured.plant_sim_module_path).read_bytes()
        ).hexdigest()
    )
    assert captured.numpy_version == importlib_metadata.version("numpy")
    assert captured.scipy_version == importlib_metadata.version("scipy")
    assert captured.torch_version == importlib_metadata.version("torch")
    assert captured.stable_baselines3_version == importlib_metadata.version(
        "stable-baselines3"
    )
    assert captured.gymnasium_version == importlib_metadata.version("gymnasium")
    assert captured.dependency_entrypoints_sha256


def test_registered_capture_rejects_dirty_and_fixture_capture_records_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = b" M tracked.py\x00?? untracked.py\x00"
    monkeypatch.setattr(provenance, "_repository_root", _repository)
    monkeypatch.setattr(provenance, "_run_git", _fake_git(status))

    with pytest.raises(provenance.DirtyWorktreeError, match="clean tracked"):
        provenance.capture(require_clean=True)

    captured = provenance.capture(require_clean=False)
    assert captured.worktree_clean is False
    assert captured.worktree_status_sha256 == hashlib.sha256(status).hexdigest()


def test_worktree_identity_binds_exact_untracked_bytes_and_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "same-path.py"
    source.write_bytes(b"first-content")
    status = b"?? same-path.py\x00"
    monkeypatch.setattr(
        provenance,
        "_run_git",
        lambda _repository, *arguments: (
            b"same-path.py\x00" if arguments == provenance._UNTRACKED_ARGUMENTS else b""
        ),
    )

    first = provenance._worktree_content_sha256(tmp_path, status=status)
    source.write_bytes(b"other-content")
    second = provenance._worktree_content_sha256(tmp_path, status=status)

    assert first != second


def test_worktree_identity_rejects_untracked_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.py"
    target.write_text("value = 1\n", encoding="utf-8")
    link = tmp_path / "link.py"
    link.symlink_to(target)
    monkeypatch.setattr(
        provenance,
        "_run_git",
        lambda _repository, *arguments: (
            b"link.py\x00" if arguments == provenance._UNTRACKED_ARGUMENTS else b""
        ),
    )

    with pytest.raises(
        provenance.RuntimeProvenanceError,
        match="cannot read|regular file|must not contain symlinks",
    ):
        provenance._worktree_content_sha256(
            tmp_path,
            status=b"?? link.py\x00",
        )


def test_worktree_identity_binds_untracked_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "executable.py"
    source.write_bytes(b"same-content")
    source.chmod(0o600)
    monkeypatch.setattr(
        provenance,
        "_run_git",
        lambda _repository, *arguments: (
            b"executable.py\x00"
            if arguments == provenance._UNTRACKED_ARGUMENTS
            else b""
        ),
    )
    first = provenance._worktree_content_sha256(
        tmp_path,
        status=b"?? executable.py\x00",
    )
    source.chmod(0o700)
    second = provenance._worktree_content_sha256(
        tmp_path,
        status=b"?? executable.py\x00",
    )
    assert first != second


def test_real_git_worktree_identity_distinguishes_clean_unstaged_staged_and_untracked(
    tmp_path: Path,
) -> None:
    def git(*arguments: str) -> bytes:
        return subprocess.run(
            ("git", *arguments),
            cwd=tmp_path,
            check=True,
            capture_output=True,
        ).stdout

    git("init", "--quiet")
    git("config", "user.name", "Runtime Provenance Test")
    git("config", "user.email", "runtime-provenance@example.invalid")
    tracked = tmp_path / "tracked.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    git("add", "tracked.py")
    git("commit", "--quiet", "-m", "initial")

    clean_status = provenance._run_git(tmp_path, *provenance._STATUS_ARGUMENTS)
    clean = provenance._worktree_content_sha256(tmp_path, status=clean_status)
    assert clean_status == b""
    assert clean == provenance._empty_worktree_content_sha256()

    tracked.write_text("value = 2\n", encoding="utf-8")
    unstaged_status = provenance._run_git(tmp_path, *provenance._STATUS_ARGUMENTS)
    unstaged = provenance._worktree_content_sha256(
        tmp_path,
        status=unstaged_status,
    )
    git("add", "tracked.py")
    staged_status = provenance._run_git(tmp_path, *provenance._STATUS_ARGUMENTS)
    staged = provenance._worktree_content_sha256(tmp_path, status=staged_status)

    untracked_path = tmp_path / "untracked file.py"
    untracked_path.write_text("value = 3\n", encoding="utf-8")
    untracked_status = provenance._run_git(tmp_path, *provenance._STATUS_ARGUMENTS)
    untracked = provenance._worktree_content_sha256(
        tmp_path,
        status=untracked_status,
    )

    assert len({clean, unstaged, staged, untracked}) == 4


def test_capture_rejects_non_boolean_clean_requirement() -> None:
    with pytest.raises(TypeError, match="require_clean"):
        provenance.capture(require_clean=1)  # type: ignore[arg-type]


def test_git_subprocess_has_a_bounded_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def run(*arguments: Any, **keywords: Any) -> subprocess.CompletedProcess[bytes]:
        observed.update(keywords)
        return subprocess.CompletedProcess(arguments, 0, stdout=b"a" * 40 + b"\n")

    monkeypatch.setattr(provenance.subprocess, "run", run)
    monkeypatch.setenv("GIT_DIR", "/tmp/alternate-repository")
    monkeypatch.setenv("GIT_WORK_TREE", "/tmp/alternate-worktree")
    output = provenance._run_git(_repository(), "rev-parse", "--verify", "HEAD")

    assert output == b"a" * 40 + b"\n"
    assert observed["timeout"] == provenance.GIT_TIMEOUT_SECONDS
    assert observed["check"] is True
    assert observed["capture_output"] is True
    assert "GIT_DIR" not in observed["env"]
    assert "GIT_WORK_TREE" not in observed["env"]
    assert observed["env"]["GIT_OPTIONAL_LOCKS"] == "0"


def test_git_timeout_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*arguments: Any, **keywords: Any) -> None:
        del arguments, keywords
        raise subprocess.TimeoutExpired("git", provenance.GIT_TIMEOUT_SECONDS)

    monkeypatch.setattr(provenance.subprocess, "run", timeout)
    with pytest.raises(provenance.RuntimeProvenanceError, match="cannot capture Git"):
        provenance._run_git(_repository(), "status")


def test_nonzero_git_exit_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*arguments: Any, **keywords: Any) -> None:
        del arguments, keywords
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(provenance.subprocess, "run", fail)
    with pytest.raises(provenance.RuntimeProvenanceError, match="cannot capture Git"):
        provenance._run_git(_repository(), "status")


def test_git_repository_identity_rejects_redirect_and_submodule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path.resolve()

    monkeypatch.setattr(
        provenance,
        "_run_git",
        lambda _repository, *arguments: (
            os.fsencode(_repository.parent) + b"\n"
            if arguments == ("rev-parse", "--show-toplevel")
            else b""
        ),
    )
    with pytest.raises(provenance.RuntimeProvenanceError, match="Git resolved"):
        provenance._verify_git_repository(repository)

    monkeypatch.setattr(
        provenance,
        "_run_git",
        lambda _repository, *arguments: (
            os.fsencode(_repository) + b"\n"
            if arguments == ("rev-parse", "--show-toplevel")
            else b"160000 deadbeef 0\tvendor/dependency\x00"
        ),
    )
    with pytest.raises(provenance.RuntimeProvenanceError, match="submodules"):
        provenance._verify_git_repository(repository)


def test_real_git_repository_rejects_hidden_index_flags(tmp_path: Path) -> None:
    def git(*arguments: str) -> None:
        subprocess.run(
            ("git", *arguments),
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init", "--quiet")
    tracked = tmp_path / "tracked.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    git("add", "tracked.py")

    git("update-index", "--assume-unchanged", "tracked.py")
    with pytest.raises(provenance.RuntimeProvenanceError, match="index flags"):
        provenance._verify_git_repository(tmp_path)
    git("update-index", "--no-assume-unchanged", "tracked.py")

    git("update-index", "--skip-worktree", "tracked.py")
    with pytest.raises(provenance.RuntimeProvenanceError, match="index flags"):
        provenance._verify_git_repository(tmp_path)


def test_plant_sim_path_rejects_symlinked_parent(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    module = external / "__init__.py"
    module.write_text("__version__ = '0.5.2'\n", encoding="utf-8")
    repository = tmp_path / "repository"
    (repository / "simulator" / "src").mkdir(parents=True)
    (repository / "simulator" / "src" / "plant_sim").symlink_to(external)
    lexical = repository / provenance.PLANT_SIM_MODULE_PATH

    with pytest.raises(
        provenance.RuntimeProvenanceError,
        match="must not contain symlinks",
    ):
        provenance._plant_sim_path(repository, str(lexical))


def test_capture_rejects_source_mutation_between_bracketing_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance, "_repository_root", _repository)
    monkeypatch.setattr(provenance, "_run_git", _fake_git(b""))
    snapshots = iter((_sha("before"), _sha("after")))
    monkeypatch.setattr(
        provenance,
        "_worktree_content_sha256",
        lambda _repository, *, status: next(snapshots),
    )

    with pytest.raises(provenance.RuntimeProvenanceError, match="source changed"):
        provenance.capture(require_clean=True)


def test_pre_post_verifier_accepts_equality_and_names_every_mismatch() -> None:
    before = _record()
    provenance.verify_unchanged(before, before)
    after = replace(
        before,
        uv_lock_sha256=_sha("changed-lock"),
        torch_version="2.8.0",
    )

    with pytest.raises(
        provenance.RuntimeProvenanceChangedError,
        match="uv_lock_sha256, torch_version",
    ):
        provenance.verify_unchanged(before, after)


def test_pre_post_verifier_rejects_unvalidated_mappings() -> None:
    with pytest.raises(TypeError, match="provenance records"):
        provenance.verify_unchanged(_record(), _record().to_dict())  # type: ignore[arg-type]

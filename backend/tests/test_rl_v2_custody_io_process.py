"""Filesystem and process fixtures only; no plant or learner is constructed."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from plant_backend.rl.v2.training import custody_io as io
from plant_backend.rl.v2.training.custody_json import (
    authenticate,
    canonical_bytes,
    require_envelope,
)
from plant_backend.rl.v2.training.custody_process import capture_process


def test_publication_barrier_and_non_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    original = os.fsync

    def fsync(fd: int) -> None:
        import stat

        events.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        original(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    target = tmp_path / "claimed.json"
    io.publish_json(target, {"claimed": True})
    assert events == ["file", "directory"]
    assert io.read_regular(target) == canonical_bytes({"claimed": True})
    with pytest.raises(FileExistsError):
        io.publish_json(target, {"claimed": False})


@pytest.mark.parametrize("failure_at", [1, 2])
def test_failed_barrier_consumes_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_at: int
) -> None:
    count = 0

    def fail(fd: int) -> None:
        nonlocal count
        count += 1
        if count == failure_at:
            raise OSError("fixture persistence failure")

    monkeypatch.setattr(os, "fsync", fail)
    target = tmp_path / "claimed.json"
    with pytest.raises(OSError):
        io.publish_json(target, {})
    assert target.exists()
    with pytest.raises(FileExistsError):
        io.publish_json(target, {})


def test_complete_inventory_retains_forbidden_and_partial_entries(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    absent = io.inventory(root)
    assert absent["exists"] is False and absent["entries"] == []
    io.mkdir_exclusive(root)
    io.mkdir_exclusive(root / "checkpoint")
    io.publish_bytes(root / "checkpoint" / "partial.zip", b"partial fixture")
    (root / "symlink").symlink_to(root / "checkpoint" / "partial.zip")
    os.link(root / "checkpoint" / "partial.zip", root / "hardlink")
    os.mkfifo(root / "pipe")
    io.publish_bytes(root / "unexpected.txt", b"retained")
    evidence = io.inventory(root)
    assert evidence["directory_count"] == 1
    assert evidence["regular_file_count"] == 1
    assert evidence["forbidden_count"] == 4
    assert [item["path"] for item in evidence["entries"]] == [
        "checkpoint",
        "checkpoint/partial.zip",
        "hardlink",
        "pipe",
        "symlink",
        "unexpected.txt",
    ]
    assert evidence == io.inventory(root)
    assert all(
        item["sha256"] is None
        for item in evidence["entries"]
        if item["node_type"] != "regular_file"
    )
    assert (root / "checkpoint" / "partial.zip").read_bytes() == b"partial fixture"


def test_file_and_parent_links_are_rejected(tmp_path: Path) -> None:
    target = tmp_path / "actual"
    io.publish_bytes(target, b"bytes")
    linked = tmp_path / "linked"
    linked.symlink_to(target)
    with pytest.raises(ValueError):
        io.read_regular(linked)
    os.link(target, tmp_path / "hard")
    with pytest.raises(ValueError):
        io.read_regular(target)
    directory = tmp_path / "alias"
    directory.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        io.publish_bytes(directory / "new", b"forbidden")
    assert not (tmp_path / "new").exists()


def test_new_root_and_storage_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaign"
    repository = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(
        os, "statvfs", lambda path: SimpleNamespace(f_bavail=10, f_frsize=10)
    )
    with pytest.raises(ValueError, match="storage"):
        io.require_new_root(root, repository, 101)
    assert not root.exists()
    io.require_new_root(root, repository, 100)
    io.mkdir_exclusive(root)
    with pytest.raises(ValueError, match="already exists"):
        io.require_new_root(root, repository, 100)
    assert io.budget_available(root, 100)
    assert not io.budget_available(root, 101)
    io.publish_bytes(root / "retained", b"retained")
    assert io.budget_available(root, 101)


def test_envelope_rejects_relabel_and_extra_fields() -> None:
    payload = authenticate({"schema": "fixture-v1", "value": 1})
    fields = {"schema", "value", "sha256"}
    assert (
        require_envelope(canonical_bytes(payload), schema="fixture-v1", fields=fields)
        == payload
    )
    for bad in (
        {**payload, "value": 2},
        authenticate({**payload, "schema": "other-v1"}),
        authenticate({**payload, "extra": True}),
    ):
        with pytest.raises(ValueError):
            require_envelope(canonical_bytes(bad), schema="fixture-v1", fields=fields)


@pytest.mark.parametrize(
    "scenario",
    [
        "success",
        "nonzero",
        "signal",
        "spawn",
        "timeout",
        "stdout_cap",
        "stderr_cap",
        "descendant",
    ],
)
def test_bounded_sealed_process_outcomes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], scenario: str
) -> None:
    commands = {
        "success": [
            sys.executable,
            "-c",
            "import os; os.write(1,b'fixture stdout'); os.write(2,b'fixture stderr')",
        ],
        "nonzero": [sys.executable, "-c", "raise SystemExit(3)"],
        "signal": [
            sys.executable,
            "-c",
            "import os,signal; os.kill(os.getpid(),signal.SIGTERM)",
        ],
        "spawn": [str(tmp_path / "missing")],
        "timeout": [sys.executable, "-c", "import time; time.sleep(30)"],
        "stdout_cap": [sys.executable, "-c", "import os; os.write(1,b'x'*8192)"],
        "stderr_cap": [sys.executable, "-c", "import os; os.write(2,b'x'*8192)"],
        "descendant": [
            sys.executable,
            "-c",
            "import os,sys,time; p=os.fork(); time.sleep(30) if p==0 else sys.exit(0)",
        ],
    }
    process, codes = capture_process(
        commands[scenario],
        stdout=tmp_path / "stdout",
        stderr=tmp_path / "stderr",
        timeout_seconds=1,
        log_limit_bytes=1024,
    )
    assert capsys.readouterr() == ("", "")
    assert process["started"] is (scenario != "spawn")
    expected = {
        "success": None,
        "nonzero": "child_nonzero_exit",
        "signal": "child_signaled",
        "spawn": "child_spawn_failed",
        "timeout": "child_timeout",
        "stdout_cap": "child_log_limit_exceeded",
        "stderr_cap": "child_log_limit_exceeded",
        "descendant": "child_timeout",
    }[scenario]
    assert not codes if expected is None else expected in codes
    for name in ("stdout", "stderr"):
        raw = io.read_regular(tmp_path / name)
        assert len(raw) <= 1024
        assert (
            io.file_record(tmp_path / name)["sha256"] == hashlib.sha256(raw).hexdigest()
        )
    if scenario.endswith("_cap"):
        assert (tmp_path / scenario.removesuffix("_cap")).read_bytes() == b"x" * 1024
    if scenario == "signal":
        assert process["termination_signal"] == signal.SIGTERM
    with pytest.raises(FileExistsError):
        capture_process(
            commands["success"],
            stdout=tmp_path / "stdout",
            stderr=tmp_path / "stderr",
            timeout_seconds=1,
            log_limit_bytes=1024,
        )


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_parent_interruption_closes_child_group(tmp_path: Path, signum: int) -> None:
    ready = tmp_path / "ready"
    code = (
        "from pathlib import Path; import os,time; "
        f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(30)"
    )
    wrapper = (
        "from pathlib import Path; import sys; "
        "from plant_backend.rl.v2.training.custody_process import capture_process; "
        f"process,codes=capture_process([sys.executable,'-c',{code!r}], "
        f"stdout=Path({str(tmp_path / 'stdout')!r}), stderr=Path({str(tmp_path / 'stderr')!r}), "
        "timeout_seconds=10,log_limit_bytes=1024); "
        "assert 'root_custody_lost' in codes"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", wrapper],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            assert parent.poll() is None
            time.sleep(0.01)
        assert ready.exists()
        child_pid = int(ready.read_text())
        parent.send_signal(signum)
        stdout, stderr = parent.communicate(timeout=10)
        assert parent.returncode == 0, stderr.decode()
        assert stdout == stderr == b""
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()

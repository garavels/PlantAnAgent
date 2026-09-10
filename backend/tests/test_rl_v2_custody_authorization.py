"""Synthetic authority records; never create the repository authorization file."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
from typing import Any

import pytest

from plant_backend.rl.v2.training import custody_authorization as auth
from plant_backend.rl.v2.training.custody_json import authenticate, canonical_bytes


HOST = {
    "platform": "fixture",
    "system": "Darwin",
    "machine": "arm64",
    "processor": None,
    "cpu_brand": "fixture",
    "physical_cpu_count": 1,
    "logical_cpu_count": 1,
    "physical_memory_bytes": 1024,
    "torch_intraop_threads": 1,
    "torch_interop_threads": 1,
    "thread_environment": {
        name: None for name in auth.capacity.CAPACITY_THREAD_ENVIRONMENT
    },
    "clock": {
        "name": "time.perf_counter_ns",
        "monotonic": True,
        "adjustable": False,
        "resolution_s": 1e-9,
    },
}


@pytest.fixture
def authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    report = authenticate({"schema": "synthetic-capacity-fixture", "host": HOST})
    raw = canonical_bytes(report)
    report_path = tmp_path / "synthetic-capacity.json"
    report_path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(auth, "CAPACITY_RAW_SHA256", digest)
    monkeypatch.setattr(auth, "CAPACITY_DOMAIN_SHA256", report["sha256"])
    return {
        "schema": auth.AUTHORIZATION_SCHEMA,
        "authorization_id": "f28e6e04-c99e-4705-adb5-01412b2f81e1",
        "authorized_at": "2026-09-11T12:00:00.000000Z",
        "protocols": {"fixture": True},
        "capacity_report": {
            "path": str(report_path),
            "raw_file_sha256": digest,
            "sha256": report["sha256"],
        },
        "approved_implementation": {
            "commit": "a" * 40,
            "tree": "b" * 40,
            "uv_lock_sha256": "c" * 64,
        },
        "campaign_root": str(tmp_path / "unopened-campaign"),
        "host_contract": {"host": HOST, "capacity_projection_applies": True},
        "campaign_contract": auth.EXECUTION,
        "registered_training_authorized": True,
        "development_evaluation_authorized": False,
        "held_out_evaluation_authorized": False,
        "non_retry_statement": auth.NON_RETRY_STATEMENT,
    }


def _validate(value: dict[str, Any], root: Path) -> dict[str, Any]:
    return auth.validate_authorization(
        canonical_bytes(authenticate(value)),
        root=root,
        protocols={"fixture": True},
        host=HOST,
    )


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("registered_training_authorized", False),
        ("registered_training_authorized", 1),
        ("development_evaluation_authorized", True),
        ("held_out_evaluation_authorized", True),
        ("campaign_root", "/private/tmp/wrong-root"),
        ("protocols", {}),
        ("non_retry_statement", "retry allowed"),
        ("campaign_contract", {"cell_count": 14}),
        ("authorization_id", "not-a-uuid"),
        ("authorized_at", "2026-09-11"),
    ],
)
def test_authorization_rejects_scope_before_output(
    authority: dict[str, Any], field: str, replacement: object
) -> None:
    root = Path(authority["campaign_root"])
    assert _validate(authority, root)["registered_training_authorized"] is True
    modified = deepcopy(authority)
    modified[field] = replacement
    with pytest.raises(ValueError):
        _validate(modified, root)
    assert not root.exists()


def test_host_projection_and_exact_shape(authority: dict[str, Any]) -> None:
    root = Path(authority["campaign_root"])
    for field, replacement in (("extra", True), ("torch_intraop_threads", 2)):
        modified = deepcopy(authority)
        modified["host_contract"]["host"][field] = replacement
        with pytest.raises(ValueError):
            _validate(modified, root)
    for replacement in (None, "yes", 1):
        modified = deepcopy(authority)
        modified["host_contract"]["capacity_projection_applies"] = replacement
        with pytest.raises(ValueError):
            _validate(modified, root)
    modified = deepcopy(authority)
    modified["host_contract"]["capacity_projection_applies"] = False
    assert (
        _validate(modified, root)["host_contract"]["capacity_projection_applies"]
        is False
    )


def test_authority_does_not_accept_untracked_or_missing_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Production resolves its fixed repository; this fixture substitutes only
    # that lookup to prove absent authority cannot create a campaign root.
    monkeypatch.setattr(auth, "repository_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError):
        auth.admit_authorization(tmp_path / "campaign", HOST)
    assert not (tmp_path / "campaign").exists()


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Custody Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            *args,
        ],
        check=True,
        capture_output=True,
    )
    return result.stdout.decode().strip()


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "dirty",
        "source",
        "wrong_parent",
        "wrong_tree",
        "wrong_lock",
        "untracked",
        "changed_bytes",
    ],
)
def test_source_commit_admission(tmp_path: Path, defect: str | None) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    (repository / "uv.lock").write_bytes(b"fixture lock\n")
    (repository / "README.md").write_text("fixture\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "fixture implementation")
    approved = {
        "commit": _git(repository, "rev-parse", "HEAD"),
        "tree": _git(repository, "rev-parse", "HEAD^{tree}"),
        "uv_lock_sha256": hashlib.sha256(
            (repository / "uv.lock").read_bytes()
        ).hexdigest(),
    }
    if defect in {"wrong_parent", "wrong_tree", "wrong_lock"}:
        approved[
            {
                "wrong_parent": "commit",
                "wrong_tree": "tree",
                "wrong_lock": "uv_lock_sha256",
            }[defect]
        ] = "0" * (64 if defect == "wrong_lock" else 40)
    value = {"approved_implementation": approved}
    raw = canonical_bytes(value)
    path = repository / auth.AUTHORIZATION_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    if defect == "source":
        (repository / "changed.py").write_text("# forbidden source change\n")
    if defect != "untracked":
        _git(repository, "add", ".")
        _git(repository, "commit", "-qm", "fixture authorization")
    if defect == "dirty":
        (repository / "README.md").write_text("dirty fixture\n")
    if defect == "changed_bytes":
        raw = canonical_bytes({"approved_implementation": approved, "extra": True})
    if defect is None:
        source = auth.capture_source(repository, value, raw)
        assert source["authorization_parent"] == approved["commit"]
        assert source["changed_paths"] == [auth.AUTHORIZATION_PATH]
    else:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            auth.capture_source(repository, value, raw)

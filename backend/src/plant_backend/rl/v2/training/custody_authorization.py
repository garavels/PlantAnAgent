"""Read-only admission of a tracked, explicitly approved training campaign."""

from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
import re
import subprocess
from typing import Any
from uuid import UUID

from . import capacity_benchmark as capacity
from .config import BETA_CONSTRAINED_RL_PROTOCOL_ID
from .custody_io import read_regular
from .custody_json import canonical_bytes, decode, require_envelope
from .registered_contract import (
    LOG_LIMIT_BYTES,
    MINIMUM_FREE_BYTES,
    PROTOCOL_ID,
    TIMEOUT_SECONDS,
    matrix,
)

AUTHORIZATION_PATH = (
    "docs/research/authorizations/v2_constrained_rl_registered_training_v1.json"
)
AUTHORIZATION_SCHEMA = "plant-beta-constrained-rl-registered-training-authorization-v1"
AUTHORIZATION_FIELDS = frozenset(
    {
        "schema",
        "authorization_id",
        "authorized_at",
        "protocols",
        "capacity_report",
        "approved_implementation",
        "campaign_root",
        "host_contract",
        "campaign_contract",
        "registered_training_authorized",
        "development_evaluation_authorized",
        "held_out_evaluation_authorized",
        "non_retry_statement",
        "sha256",
    }
)
ALLOWED_AUTHORIZATION_CHANGES = frozenset(
    {
        AUTHORIZATION_PATH,
        "README.md",
        "docs/research/README.md",
        "docs/research/ROADMAP.md",
        "docs/research/JOURNAL.md",
    }
)
NON_RETRY_STATEMENT = (
    "Failed, timed-out, interrupted, or rejected cells cannot be retried or "
    "replaced at the same source revision."
)
CAPACITY_RAW_SHA256 = "3b9074824c6e81733e18653fd87e0a470e51c478183ed1a0a656f3589c4851fa"
CAPACITY_DOMAIN_SHA256 = (
    "78b6c1f067d88aa43b672e42dd94c41d37ff9444e3963a2ebcb47f85933d9c3e"
)
EXECUTION = {
    "cell_count": 15,
    "sequential": True,
    "timeout_seconds": TIMEOUT_SECONDS,
    "log_limit_bytes_per_stream": LOG_LIMIT_BYTES,
    "minimum_free_bytes": MINIMUM_FREE_BYTES,
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[6]


def exact_object(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} fields differ")
    return value


def utc_timestamp(value: object) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z", value) is None
    ):
        raise ValueError("timestamp must be canonical UTC with six fractional digits")
    datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    return value


def protocol_identity(repository: Path) -> dict[str, Any]:
    protocols: dict[str, Any] = {}
    for label, identifier, path, expected in (
        (
            "constrained_rl",
            BETA_CONSTRAINED_RL_PROTOCOL_ID,
            "docs/research/CONSTRAINED_RL_BETA_PROTOCOL.md",
            "4d0269f37c5cf1fb398be655505ec3d63b081f55196eb81eec67db2a797f5a22",
        ),
        (
            "custody",
            PROTOCOL_ID,
            "docs/research/CONSTRAINED_RL_REGISTERED_TRAINING_CUSTODY_PROTOCOL.md",
            "af73687eefff32132409a8afcfc6d34528da68bf94c4788e697fdc62046a9801",
        ),
    ):
        digest = hashlib.sha256(read_regular(repository / path)).hexdigest()
        if digest != expected:
            raise ValueError("frozen protocol bytes differ")
        protocols[label] = {"id": identifier, "path": path, "raw_file_sha256": digest}
    protocols["capacity"] = capacity._protocol_identity(repository)
    return protocols


def _git(repository: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    return result.stdout


def capture_source(
    repository: Path, authorization: dict[str, Any], raw: bytes
) -> dict[str, Any]:
    """Prove the authority was added alone on top of the approved implementation."""
    if _git(repository, "status", "--porcelain=v1", "-z", "--untracked-files=all"):
        raise ValueError("registered training requires a clean source revision")
    approved = exact_object(
        authorization["approved_implementation"],
        {"commit", "tree", "uv_lock_sha256"},
        "approved implementation",
    )
    for key in ("commit", "tree", "uv_lock_sha256"):
        size = 64 if key == "uv_lock_sha256" else 40
        if (
            not isinstance(approved[key], str)
            or re.fullmatch(f"[0-9a-f]{{{size}}}", approved[key]) is None
        ):
            raise ValueError("approved implementation identity is malformed")
    lineage = (
        _git(repository, "rev-list", "--parents", "-n", "1", "HEAD")
        .decode("ascii")
        .strip()
        .split()
    )
    if len(lineage) != 2 or lineage[1] != approved["commit"]:
        raise ValueError(
            "authorization commit must have only the approved implementation as parent"
        )
    if (
        _git(repository, "rev-parse", f"{approved['commit']}^{{tree}}")
        .decode("ascii")
        .strip()
        != approved["tree"]
    ):
        raise ValueError("approved implementation tree differs")
    lock_sha256 = hashlib.sha256(read_regular(repository / "uv.lock")).hexdigest()
    if lock_sha256 != approved["uv_lock_sha256"]:
        raise ValueError("approved lockfile differs")
    if _git(repository, "show", f"HEAD:{AUTHORIZATION_PATH}") != raw:
        raise ValueError("authorization must be the exact tracked bytes")
    changes = (
        _git(
            repository,
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            lineage[1],
            lineage[0],
        )
        .decode("utf-8")
        .split("\0")[:-1]
    )
    if (
        AUTHORIZATION_PATH not in changes
        or not set(changes) <= ALLOWED_AUTHORIZATION_CHANGES
    ):
        raise ValueError("authorization commit changes unapproved source paths")
    # The approved revision must not already contain an authorization that can
    # be modified into a second attempt on the same implementation.
    added = (
        _git(
            repository,
            "diff",
            "--name-only",
            "--diff-filter=A",
            "--no-renames",
            "-z",
            lineage[1],
            lineage[0],
        )
        .decode("utf-8")
        .split("\0")[:-1]
    )
    if AUTHORIZATION_PATH not in added:
        raise ValueError(
            "authorization must be newly added after implementation approval"
        )
    return {
        "authorization_commit": lineage[0],
        "approved_implementation_commit": lineage[1],
        "approved_implementation_tree": approved["tree"],
        "uv_lock_sha256": lock_sha256,
        "authorization_parent": lineage[1],
        "changed_paths": sorted(changes),
    }


def validate_authorization(
    raw: bytes,
    *,
    root: Path,
    protocols: dict[str, Any],
    host: dict[str, Any],
) -> dict[str, Any]:
    """Validate scope before any campaign file or process can be created."""
    authorization = require_envelope(
        raw, schema=AUTHORIZATION_SCHEMA, fields=AUTHORIZATION_FIELDS
    )
    identifier = authorization["authorization_id"]
    if not isinstance(identifier, str) or str(UUID(identifier)) != identifier:
        raise ValueError("authorization identifier must be a canonical UUID")
    utc_timestamp(authorization["authorized_at"])
    for key, expected in (
        ("registered_training_authorized", True),
        ("development_evaluation_authorized", False),
        ("held_out_evaluation_authorized", False),
    ):
        if authorization[key] is not expected:
            raise ValueError(
                "authorization scope differs from registered training only"
            )
    if authorization["campaign_root"] != str(root):
        raise ValueError("authorization campaign root differs")
    matrix(root)
    if canonical_bytes(authorization["protocols"]) != canonical_bytes(protocols):
        raise ValueError("authorization protocol identities differ")
    if canonical_bytes(authorization["campaign_contract"]) != canonical_bytes(
        EXECUTION
    ):
        raise ValueError("authorization campaign contract differs")
    if authorization["non_retry_statement"] != NON_RETRY_STATEMENT:
        raise ValueError("authorization must accept the exact non-retry statement")
    contract = exact_object(
        authorization["host_contract"],
        {"host", "capacity_projection_applies"},
        "host contract",
    )
    try:
        capacity._verify_host(contract["host"])
    except (ValueError, TypeError, capacity.CapacityBenchmarkError) as reason:
        raise ValueError("authorized host schema differs") from reason
    if canonical_bytes(contract["host"]) != canonical_bytes(host):
        raise ValueError("authorized host differs from the current host")
    if type(contract["capacity_projection_applies"]) is not bool:
        raise ValueError("capacity projection applicability must be explicit")
    report = exact_object(
        authorization["capacity_report"],
        {"path", "raw_file_sha256", "sha256"},
        "capacity report",
    )
    if (
        report["raw_file_sha256"] != CAPACITY_RAW_SHA256
        or report["sha256"] != CAPACITY_DOMAIN_SHA256
    ):
        raise ValueError("capacity report identity differs from frozen evidence")
    if not isinstance(report["path"], str) or not Path(report["path"]).is_absolute():
        raise ValueError("capacity report path must be absolute")
    report_raw = read_regular(Path(report["path"]))
    if hashlib.sha256(report_raw).hexdigest() != CAPACITY_RAW_SHA256:
        raise ValueError("capacity report raw bytes differ")
    capacity_report = decode(report_raw)
    if capacity_report.get("sha256") != CAPACITY_DOMAIN_SHA256:
        raise ValueError("capacity report domain identity differs")
    if contract["capacity_projection_applies"] and canonical_bytes(
        capacity_report["host"]
    ) != canonical_bytes(host):
        raise ValueError("capacity projection does not apply to this host")
    return authorization


def admit_authorization(
    root: Path, host: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Production admission: no repository, reader, or source-verifier hooks."""
    repository = repository_root()
    raw = read_regular(repository / AUTHORIZATION_PATH)
    authorization = validate_authorization(
        raw, root=root, protocols=protocol_identity(repository), host=host
    )
    source = capture_source(repository, authorization, raw)
    return authorization, source

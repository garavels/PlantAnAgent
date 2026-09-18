"""Fixed production source/host recapture; no environment construction or reset."""

from __future__ import annotations

from typing import Any

from . import capacity_benchmark as capacity
from . import custody_authorization as authority
from .custody_state import IdentitySnapshot
from .runtime_provenance import capture


def capture_identity(authorization: dict[str, Any]) -> IdentitySnapshot:
    # Dirty evidence is retained as an integrity failure, never admitted as the
    # starting identity. The caller has already authenticated approved values.
    runtime = capture(require_clean=False)
    repository = authority.repository_root()
    lineage = (
        authority._git(repository, "rev-list", "--parents", "-n", "1", "HEAD")
        .decode("ascii")
        .strip()
        .split()
    )
    if len(lineage) != 2 or lineage[0] != runtime.git_head:
        raise ValueError("source identity cannot be captured consistently")
    approved = authorization["approved_implementation"]
    changed = (
        authority._git(
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
    source = {
        "authorization_commit": lineage[0],
        "approved_implementation_commit": approved["commit"],
        "approved_implementation_tree": authority._git(
            repository, "rev-parse", f"{approved['commit']}^{{tree}}"
        )
        .decode("ascii")
        .strip(),
        "uv_lock_sha256": runtime.uv_lock_sha256,
        "authorization_parent": lineage[1],
        "changed_paths": sorted(changed),
    }
    host = capacity._verify_host(capacity._host_identity())
    return IdentitySnapshot(source, runtime.to_dict(), host)

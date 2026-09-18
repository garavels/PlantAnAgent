"""Single-use parent-to-child launch channel for an already durable claim."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import select
import stat
from typing import Any

from . import capacity_benchmark as capacity
from . import custody_authorization as authority
from . import custody_documents as documents
from . import custody_io as io
from .config import ControllerNeutralTrainingConfigV1
from .custody_json import canonical_bytes, decode, require_envelope
from .custody_runtime import capture_identity
from .custody_validation import exact
from .registered_contract import matrix
from .runtime_provenance import BetaTrainingRuntimeProvenanceV1

LAUNCH_FD_ENV = "PLANT_BETA_REGISTERED_CUSTODY_FD"
LAUNCH_SCHEMA = "plant-beta-constrained-rl-registered-training-child-launch-v1"


def launch_message(cell: dict[str, Any]) -> bytes:
    claim = require_envelope(
        io.read_regular(Path(cell["claim_path"])),
        schema=documents.CLAIM_SCHEMA,
        fields=documents.CLAIM_FIELDS,
    )
    exact("launch claim cell", claim["cell"], cell)
    return canonical_bytes(
        {
            "schema": LAUNCH_SCHEMA,
            "parent_pid": os.getpid(),
            "claim_sha256": claim["sha256"],
            "config_sha256": cell["config_sha256"],
        }
    )


def _consume_permission() -> dict[str, Any]:
    descriptor_text = os.environ.pop(LAUNCH_FD_ENV, None)
    if (
        descriptor_text is None
        or not descriptor_text.isascii()
        or not descriptor_text.isdecimal()
    ):
        raise ValueError("registered training requires admitted campaign custody")
    descriptor = int(descriptor_text)
    if descriptor < 3 or str(descriptor) != descriptor_text:
        raise ValueError("registered launch descriptor is invalid")
    try:
        if not stat.S_ISFIFO(os.fstat(descriptor).st_mode):
            raise ValueError("registered launch channel must be an anonymous pipe")
        if not select.select([descriptor], [], [], 1.0)[0]:
            raise ValueError("registered launch channel has no permission")
        raw = os.read(descriptor, 1025)
    finally:
        os.close(descriptor)
    value = decode(raw)
    if (
        set(value) != {"schema", "parent_pid", "claim_sha256", "config_sha256"}
        or value["schema"] != LAUNCH_SCHEMA
    ):
        raise ValueError("registered launch permission fields differ")
    if type(value["parent_pid"]) is not int or value["parent_pid"] != os.getppid():
        raise ValueError("registered launch permission belongs to another parent")
    if len(raw) > 1024 or raw != canonical_bytes(value):
        raise ValueError("registered launch permission is not canonical")
    return value


def admit_registered_launch(
    config: ControllerNeutralTrainingConfigV1,
) -> BetaTrainingRuntimeProvenanceV1:
    """Authenticate the pipe, source, full plan, and open claim before any reset."""
    permission = _consume_permission()
    config.validate()
    output = Path(config.output_dir)
    root = output.parent.parent
    cells = matrix(root)
    matching = [cell for cell in cells if cell["output_path"] == str(output)]
    if len(matching) != 1:
        raise ValueError("registered output is not a declared campaign cell")
    cell = matching[0]
    exact(
        "launch config hash",
        permission["config_sha256"],
        hashlib.sha256(canonical_bytes(config.to_dict())).hexdigest(),
    )
    exact("declared config hash", permission["config_sha256"], cell["config_sha256"])
    host = capacity._verify_host(capacity._host_identity())
    authorization, source = authority.admit_authorization(root, host)
    identity = capture_identity(authorization)
    exact("launch source", identity.source, source)
    exact("launch host", identity.host, host)
    from .custody_verifier import _read_cells, _read_context

    context = _read_context(root, authorization, identity)
    if {path.name for path in root.iterdir()} != {
        "authorization.json",
        "plan.json",
        "configs",
        "logs",
        "attempts",
        "artifacts",
    }:
        raise ValueError("registered campaign has closed or undeclared root entries")
    if set((root / "configs").iterdir()) != {
        Path(item["config_path"]) for item in cells
    }:
        raise ValueError("registered config registry differs from the complete matrix")
    for declared in cells:
        if (
            hashlib.sha256(io.read_regular(Path(declared["config_path"]))).hexdigest()
            != declared["config_sha256"]
        ):
            raise ValueError("registered config bytes changed before launch")
    # The current claim is deliberately still open. Earlier complete receipts
    # are chained to the digest delivered by the live parent through the pipe.
    prior = _read_cells(context, stop_before=cell["order"])
    if any(item.terminal is None for item in prior):
        raise ValueError("registered launch would skip an earlier attempt")
    previous = prior[-1].terminal if prior else None
    claim = require_envelope(
        io.read_regular(Path(cell["claim_path"])),
        schema=documents.CLAIM_SCHEMA,
        fields=documents.CLAIM_FIELDS,
    )
    exact("parent launch claim", claim["sha256"], permission["claim_sha256"])
    exact(
        "open launch claim",
        claim,
        documents.make_claim(
            context.plan, cell, previous, timestamp=claim["claimed_at_utc"]
        ),
    )
    if any(
        os.path.lexists(cell[key])
        for key in ("completed_receipt_path", "failed_receipt_path", "output_path")
    ):
        raise ValueError("registered attempt is already terminal or has artifacts")
    for name in ("stdout", "stderr"):
        io.read_regular(Path(cell[f"{name}_path"]))
    if (root / "report.json").exists() or (root / "synthetic-report.json").exists():
        raise ValueError("registered campaign is already closed")
    for later in cells[cell["order"] :]:
        if any(
            os.path.lexists(later[key])
            for key in (
                "claim_path",
                "completed_receipt_path",
                "failed_receipt_path",
                "stdout_path",
                "stderr_path",
                "output_path",
            )
        ):
            raise ValueError("registered launch is not the sole next attempt")
    return BetaTrainingRuntimeProvenanceV1.from_dict(identity.runtime)

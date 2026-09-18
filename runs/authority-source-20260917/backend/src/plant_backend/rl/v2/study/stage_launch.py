"""One-use live-parent channel; closed files alone cannot start a study child."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import select
import stat
from typing import Any

from .. import beta_controller_benchmark as benchmark
from ..training import (
    custody_verifier as training_verifier,
    custody_documents as training_documents,
    custody_io as io,
)
from ..training.custody_checkpoint import admit_checkpoint
from ..training.custody_json import canonical_bytes, decode
from ..training.custody_launch import LAUNCH_FD_ENV
from ..training.custody_validation import exact
from ..training.config import ControllerNeutralTrainingConfigV1
from ..training.registered_contract import matrix, registered_config
from ..training.runtime_provenance import (
    BetaTrainingRuntimeProvenanceV1,
    capture,
    verify_unchanged,
)
from . import stage_documents as documents, registry as registry_module, stage_lock
from .execution_budget import ExecutionBudget, ROOT_CONTROL_LIMIT_BYTES
from .archive import _read_stored
from .arm_evidence import file_stamp
from .training_evidence import _tree_stamps
from .arm_execution import _execute_arm
from .latency_evidence import configure_latency_threads, capture_latency_host


def permission(
    registry: dict[str, Any], claim: dict[str, Any], authorization: dict[str, Any]
) -> bytes:
    return canonical_bytes(
        {
            "schema": documents.LAUNCH_SCHEMA,
            "parent_pid": os.getpid(),
            "registry_sha256": registry["sha256"],
            "claim_sha256": claim["sha256"],
            "authorization_sha256": authorization["sha256"],
        }
    )


def consume_permission() -> dict[str, Any]:
    text = os.environ.pop(LAUNCH_FD_ENV, None)
    if text is None or not text.isascii() or not text.isdecimal():
        raise ValueError("study child requires a live admitted parent")
    descriptor = int(text)
    if descriptor < 3 or str(descriptor) != text:
        raise ValueError("study launch descriptor is invalid")
    try:
        if (
            not stat.S_ISFIFO(os.fstat(descriptor).st_mode)
            or not select.select([descriptor], [], [], 1.0)[0]
        ):
            raise ValueError("study launch requires a readable anonymous pipe")
        raw = os.read(descriptor, 1025)
    finally:
        os.close(descriptor)
    value = decode(raw)
    if (
        set(value)
        != {
            "schema",
            "parent_pid",
            "registry_sha256",
            "claim_sha256",
            "authorization_sha256",
        }
        or value["schema"] != documents.LAUNCH_SCHEMA
    ):
        raise ValueError("study launch fields differ")
    if (
        type(value["parent_pid"]) is not int
        or value["parent_pid"] != os.getppid()
        or len(raw) > 1024
        or canonical_bytes(value) != raw
    ):
        raise ValueError("study launch does not belong to this live parent")
    return value


@dataclass(frozen=True)
class ChildBindings:
    """Parent-attested bindings, deliberately not a fresh TrainingAdmission object."""

    registry: dict[str, Any]
    authorization: dict[str, Any]
    budget: ExecutionBudget
    training_report: dict[str, Any]

    def config_for(self, cell: dict[str, Any]) -> ControllerNeutralTrainingConfigV1:
        item = matrix(Path(self.registry["training_root"]))[
            (cell["actor"]["training_order"] or 1) - 1
        ]
        return registered_config(
            Path(item["output_path"]), item["algorithm"], item["optimizer_seed"]
        )

    def checkpoint_for(self, cell: dict[str, Any]) -> dict[str, Any] | None:
        order = cell["actor"]["training_order"]
        return (
            None
            if order is None
            else self.training_report["cells"][order - 1]["checkpoint"]
        )

    def training_failure(self, cell: dict[str, Any]) -> str | None:
        order = cell["actor"]["training_order"]
        if order is None:
            return None
        record = self.training_report["cells"][order - 1]
        return None if record["state"] == "admitted" else record["failure"]["code"]


def run_child(training_root: Path | str, stage: str, order: int) -> None:
    permitted = consume_permission()
    registry = registry_module.stage_registry(training_root, stage)
    if type(order) is not int or not 1 <= order <= 576:
        raise ValueError("study child order is outside its registry")
    cell = registry["cells"][order - 1]
    authority_path = Path(registry["authorization_input_path"])
    authority_stamp = file_stamp(authority_path)
    authority = documents.read(
        authority_path,
        documents.AUTH_SCHEMA,
        documents.AUTH_FIELDS,
    )
    exact("parent stage registry", registry["sha256"], permitted["registry_sha256"])
    exact(
        "parent stage authorization",
        authority["sha256"],
        permitted["authorization_sha256"],
    )
    exact("authorized stage registry", authority["registry_sha256"], registry["sha256"])
    exact("stage scope", authority["stage"], stage)
    exact(
        "child operator assertions",
        authority["operator_assertions"],
        documents.ASSERTIONS,
    )
    if authority["stage_execution_authorized"] is not True:
        raise ValueError("child stage lacks authority")
    exact(
        "authority copy",
        documents.read(
            Path(registry["authorization_path"]),
            documents.AUTH_SCHEMA,
            documents.AUTH_FIELDS,
        ),
        authority,
    )
    exact(
        "child registry copy",
        _read_stored(
            Path(registry["registry_path"]), ROOT_CONTROL_LIMIT_BYTES
        ).decode(),
        canonical_bytes(registry).decode(),
    )
    root = Path(registry["stage_root"])
    io.require_directory(root / "cells")
    if {path.name for path in root.iterdir()} != {
        "registry.json",
        "authorization.json",
        "opened.json",
        "cells",
    }:
        raise ValueError("study stage is closed or has undeclared root entries")
    if {path.name for path in (root / "cells").iterdir()} != {
        Path(item["directory"]).name for item in registry["cells"]
    }:
        raise ValueError("child registry lacks its fixed cell directories")
    for item in registry["cells"]:
        io.require_directory(Path(item["directory"]))
    opened = documents.read(
        Path(registry["opening_claim_path"]),
        documents.OPEN_SCHEMA,
        documents.OPEN_FIELDS,
    )
    exact(
        "live parent opening",
        opened,
        documents.opening(
            authority,
            registry,
            parent_pid=permitted["parent_pid"],
            lock=opened["lock"],
            timestamp=opened["opened_at_utc"],
        ),
    )
    stage_lock.require_locked(opened["lock"])
    claim = documents.read(
        Path(cell["claim_path"]),
        documents.CLAIM_SCHEMA,
        documents.CLAIM_FIELDS,
        cell=True,
    )
    exact("parent child claim", claim["sha256"], permitted["claim_sha256"])
    previous = None
    if order > 1:
        previous_cell = registry["cells"][order - 2]
        previous = documents.read(
            Path(previous_cell["terminal_path"]),
            documents.TERMINAL_SCHEMA,
            documents.TERMINAL_FIELDS,
            cell=True,
        )
        exact("previous stage cell", previous["cell"], previous_cell)
        if previous["failure"] == "stage_integrity_lost":
            raise ValueError("stage already lost integrity")
    exact(
        "sole next stage claim",
        claim,
        documents.claim(
            registry, opened, cell, previous, timestamp=claim["claimed_at_utc"]
        ),
    )
    if {path.name for path in Path(cell["directory"]).iterdir()} != {
        "claimed.json",
        "stdout.log",
        "stderr.log",
    }:
        raise ValueError("stage attempt has already run or is malformed")
    for name in ("stdout_path", "stderr_path"):
        io.file_record(Path(cell[name]))
    if any(
        list(Path(later["directory"]).iterdir()) for later in registry["cells"][order:]
    ):
        raise ValueError("stage has a later consumed attempt")
    report_path = Path(registry["training_root"]) / "report.json"
    training_stamps = _tree_stamps(report_path.parent)
    # The parent owns the two live log files. Their content may grow during
    # admission; every other stage node must remain byte/identity stable.
    log_names = {
        Path(cell[key]).relative_to(root).as_posix()
        for key in ("stdout_path", "stderr_path")
    }
    stage_stamps = tuple(row for row in _tree_stamps(root) if row[0] not in log_names)
    report = training_verifier._read_report(report_path, synthetic=False)
    exact(
        "parent-admitted training report",
        training_documents.envelope_reference(report_path, report),
        authority["training_report"],
    )
    runtime = BetaTrainingRuntimeProvenanceV1.from_dict(report["runtime_provenance"])
    verify_unchanged(runtime, capture(require_clean=True))
    exact(
        "parent-admitted current source",
        benchmark._source_provenance(require_clean=True),
        authority["source_provenance"],
    )
    context = ChildBindings(
        registry, authority, ExecutionBudget.from_dict(authority["budget"]), report
    )
    if context.training_failure(cell) is not None:
        raise ValueError("unadmitted training cannot execute a study child")
    config = context.config_for(cell)
    expected_checkpoint = context.checkpoint_for(cell)
    if expected_checkpoint is not None:
        exact(
            "child checkpoint readmission",
            admit_checkpoint(config, runtime),
            expected_checkpoint,
        )

    def recheck_inputs() -> None:
        if (
            os.getppid() != permitted["parent_pid"]
            or file_stamp(authority_path) != authority_stamp
            or _tree_stamps(report_path.parent) != training_stamps
            or tuple(row for row in _tree_stamps(root) if row[0] not in log_names)
            != stage_stamps
        ):
            raise ValueError("study child inputs changed after launch admission")
        stage_lock.require_locked(opened["lock"])
        verify_unchanged(runtime, capture(require_clean=True))

    recheck_inputs()
    configure_latency_threads()
    exact(
        "child registered latency host",
        capture_latency_host(),
        authority["latency_host"],
    )
    recheck_inputs()
    executed = _execute_arm(
        config,
        arm=cell["actor"]["arm"],
        seed=cell["episode_seed"],
        checkpoint=(
            None if expected_checkpoint is None else Path(expected_checkpoint["path"])
        ),
        expected_training_runtime=runtime,
    )
    recheck_inputs()
    from .stage_arm import publish_arm

    publish_arm(context, cell, claim, executed)

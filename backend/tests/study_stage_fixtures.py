"""Explicit reset-free substitutes for testing production stage custody.

No fixture in this module admits a real model or produces physical evidence.
The training and arm boundaries are replaced only by individual test patches.
"""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from custody_fixtures import context as training_context, sha
from plant_backend.rl.v2.study import registry, stage_authority as authority
from plant_backend.rl.v2.study import stage_documents as documents
from plant_backend.rl.v2.study.execution_budget import EvidenceLimits, ExecutionBudget
from plant_backend.rl.v2.study.training_scorecards import TrainingSlot
from plant_backend.rl.v2.training.custody_json import authenticate, canonical_bytes
from test_rl_v2_study_training_evidence import ledgers

SOURCE = {"fixture": "reset-free stage test; no registered provenance"}
AUTHORIZED = "2026-09-01T00:00:00.000000Z"


@lru_cache
def rollout_rows(algorithm: str) -> tuple[dict[str, Any], ...]:
    return tuple(ledgers(algorithm))


def budget() -> ExecutionBudget:
    limits = EvidenceLimits(64 * 1024**2, 256 * 1024**2, 2 * 1024**2, 8 * 1024**2)
    return ExecutionBudget(
        60, 4096, (limits,) * 4, 64 * 1024**2, 256 * 1024**2, 1024**3
    )


def training(root: Path, *, failed: tuple[int, ...] = ()) -> SimpleNamespace:
    root.mkdir()
    base = training_context(root)
    base.identity.host.update(
        cpu_brand="Explicit fixture CPU",
        torch_intraop_threads=8,
        torch_interop_threads=12,
    )
    rows, slots = [], []
    for actor in registry.actors()[3:]:
        invalid = actor["training_order"] in failed
        failure = "checkpoint_admission_failed" if invalid else None
        slots.append(
            TrainingSlot(
                actor["algorithm"],
                actor["optimizer_seed"],
                True,
                not invalid,
                failure,
                None if invalid else rollout_rows(actor["algorithm"]),
            )
        )
        rows.append(
            {
                "state": "checkpoint_rejected" if invalid else "admitted",
                "failure": None if failure is None else {"code": failure},
                "checkpoint": (
                    None if invalid else {"fixture_order": actor["training_order"]}
                ),
            }
        )
    report = authenticate(
        {
            "schema": "explicit-test-training-input",
            "abort": None,
            "cells": rows,
            "runtime_provenance": base.identity.runtime,
        }
    )
    path = root / "report.json"
    path.write_bytes(canonical_bytes(report))
    calls: list[str] = []
    return SimpleNamespace(
        path=path,
        report=report,
        slots=tuple(slots),
        identity=base.identity,
        evidence=authenticate(
            {
                "schema": "fixture-training-inputs",
                "training_report": documents.reference(path, report),
            }
        ),
        recheck=lambda: calls.append("training_recheck"),
        calls=calls,
    )


def authorize(
    trained: SimpleNamespace,
    stage: str = "development",
    *,
    development: object = None,
    execution_budget: ExecutionBudget | None = None,
) -> dict[str, Any]:
    record = registry.stage_registry(trained.path.parent, stage)
    result = authenticate(
        {
            "schema": documents.AUTH_SCHEMA,
            "protocol": documents.protocol(),
            "authorization_id": "ba179814-ad2c-41f6-838a-8d01418e6e17",
            "authorized_at_utc": AUTHORIZED,
            "stage": stage,
            "registry_sha256": record["sha256"],
            "training_inputs_sha256": trained.evidence["sha256"],
            "training_report": trained.evidence["training_report"],
            "source_provenance": deepcopy(SOURCE),
            "latency_host": authority.expected_latency_host(trained),
            "budget": (
                budget() if execution_budget is None else execution_budget
            ).to_dict(),
            "development": development,
            "stage_execution_authorized": True,
            "operator_assertions": deepcopy(documents.ASSERTIONS),
        }
    )
    documents.publish(Path(record["authorization_input_path"]), result)
    return result


def patch_admission(monkeypatch: Any, trained: SimpleNamespace) -> None:
    def read(path: Path) -> SimpleNamespace:
        assert path == trained.path
        trained.calls.append("fresh_training_admission")
        return trained

    monkeypatch.setattr(authority, "admit_training", read)
    monkeypatch.setattr(
        authority.benchmark, "_source_provenance", lambda **_: deepcopy(SOURCE)
    )


def process_record(
    record: dict[str, Any], cell: dict[str, Any], *, code: int = 0
) -> dict[str, Any]:
    return {
        "command": documents.command(record, cell),
        "timeout_seconds": 60,
        "started": True,
        "exit_code": code,
        "termination_signal": -code if code < 0 else None,
        "wall_elapsed_ns": 1_000_000,
        "child_user_cpu_ns": 0,
        "child_system_cpu_ns": 0,
    }


def rehash_file(path: Path, change: Any) -> None:
    from plant_backend.rl.v2.training.custody_json import decode

    value = decode(path.read_bytes())
    change(value)
    path.write_bytes(canonical_bytes(authenticate(value)))


def opening_files(context: Any, lock: dict[str, Any]) -> dict[str, Any]:
    from plant_backend.rl.v2.training import custody_io as io

    io.mkdir_exclusive(context.root)
    documents.publish(
        Path(context.registry["authorization_path"]), context.authorization
    )
    documents.publish(Path(context.registry["registry_path"]), context.registry)
    io.mkdir_exclusive(context.root / "cells")
    for cell in context.registry["cells"]:
        io.mkdir_exclusive(Path(cell["directory"]))
    opened = documents.opening(
        context.authorization,
        context.registry,
        parent_pid=123,
        lock=lock,
        timestamp="2026-09-02T00:00:00.000000Z",
    )
    documents.publish(Path(context.registry["opening_claim_path"]), opened)
    return opened

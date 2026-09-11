"""Host-bound timing joined to independently verified synthetic physical arms."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
import platform
from typing import Any

from ..training import capacity_benchmark as capacity
from ..training import custody_io as io
from ..training.custody_json import (
    authenticate,
    canonical_bytes,
    decode,
    require_envelope,
)
from . import arm_evidence
from .contract import PREFIX
from .latency import (
    INFERENCE_LIMIT_NS,
    _integer,
    latency_contract,
    latency_summary,
    reconstruct_latency,
)

HOST_SCHEMA = PREFIX + "latency-host-v1"
SCHEMA = PREFIX + "synthetic-latency-evidence-v1"
FILENAME = "synthetic-latency.json"
PACKAGES = ("numpy", "scipy", "torch", "stable-baselines3", "gymnasium", "plant-sim")
CLAIM_BOUNDARY = "synthetic latency evidence only; no registered host, concurrency, stage or policy admission"
FIELDS = frozenset(
    {
        "schema",
        "latency_contract_sha256",
        "arm_file",
        "arm_envelope_sha256",
        "host_before",
        "host_after",
        "measurement",
        "result",
        "claim_boundary",
        "result_claim",
        "sha256",
    }
)


def configure_latency_threads() -> None:
    """Call only in a fresh study child, before any policy is loaded or warmed."""
    import torch

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("study latency requires Torch intra/inter-op threads of one")


def capture_latency_host() -> dict[str, Any]:
    """Read hardware/runtime only; this does not assert absence of other jobs."""
    return authenticate(
        {
            "schema": HOST_SCHEMA,
            "machine": capacity._host_identity(),
            "python": platform.python_version(),
            "dependencies": {name: version(name) for name in PACKAGES},
        }
    )


def validate_latency_host(raw: Mapping[str, Any]) -> dict[str, Any]:
    host = dict(raw)
    if (
        set(host) != {"schema", "machine", "python", "dependencies", "sha256"}
        or host["schema"] != HOST_SCHEMA
        or canonical_bytes(host) != canonical_bytes(authenticate(host))
    ):
        raise ValueError("latency host identity differs")
    machine = capacity._verify_host(host["machine"])
    if (
        not machine["cpu_brand"]
        or machine["logical_cpu_count"] is None
        or machine["clock"]["adjustable"] is not False
    ):
        raise ValueError("latency host omitted CPU/clock identity")
    dependencies = host["dependencies"]
    if not isinstance(dependencies, dict) or set(dependencies) != set(PACKAGES):
        raise ValueError("latency dependency fields differ")
    if any(
        not isinstance(value, str) or not value
        for value in (*dependencies.values(), host["python"])
    ):
        raise ValueError("latency versions must be nonempty strings")
    return host


def _solver_groups(run: Mapping[str, Any]) -> list[list[int]]:
    evidence = run["evidence"]
    if run["arm"] == "M24":
        events = evidence["run"]["events"]
        return [
            [
                _integer(stage["elapsed_ns"], "solver elapsed")
                for stage in event["stages"]
            ]
            for event in events
        ]
    if run["arm"] == "E-beta":
        groups = [
            [
                _integer(stage["elapsed_ns"], "solver elapsed")
                for stage in decision["evidence"]["plan"]["stages"]
            ]
            for decision in evidence["decisions"]
        ]
        failure = run.get("failure")
        if failure is not None and failure.get("solver_stages"):
            groups.append(
                [
                    _integer(stage["elapsed_ns"], "failed solver elapsed")
                    for stage in failure["solver_stages"]
                ]
            )
        return groups
    return []


def bind_latency_to_verified_run(
    measurement: Mapping[str, Any], run: Mapping[str, Any]
) -> dict[str, Any]:
    """Join admitted run witnesses to raw timings. Caller must admit the full run.

    This arithmetic seam deliberately grants no source admission. The synthetic
    file verifier below performs that separate complete-ledger check itself.
    """
    result = reconstruct_latency(measurement)
    if result["arm"] != run["arm"]:
        raise ValueError("latency actor differs from physical arm")
    rows = measurement["decisions"]
    evidence = run["evidence"]
    boundaries = evidence["boundary_ledger"]["decision_boundaries"]
    if len(rows) != len(boundaries):
        raise ValueError("timing ledger does not cover every attempted decision")
    completed = _integer(run["completed_decisions"], "completed decisions")
    if sum(row["interval_returned"] for row in rows) != completed:
        raise ValueError("timing interval count differs from physical execution")
    if any(not row["interval_returned"] for row in rows[:completed]):
        raise ValueError("completed physics has an incomplete timing interval")
    for row, boundary in zip(rows, boundaries, strict=True):
        if row["decision_index"] != boundary["decision_index"]:
            raise ValueError("timing index differs from public boundary")
        for inference in row["inferences"]:
            if inference["input_sha256"] != boundary["normalized_observation_sha256"]:
                raise ValueError("timed inference used a different public observation")
    if rows and not rows[-1]["interval_returned"]:
        failure = run.get("failure")
        if failure is None or rows[-1]["failure_type"] != failure["exception_type"]:
            raise ValueError("timing failure differs from arm failure")
    if run["status"] == "not_run_preflight_failed":
        dispatcher: list[list[int]] = []
        solver: list[list[int]] = []
    else:
        dispatcher = (
            evidence["run"]["dispatcher_elapsed_ns"]
            if run["arm"] == "M24"
            else run["dispatcher_elapsed_ns"]
        )
        solver = _solver_groups(run)
        if len(dispatcher) != completed:
            raise ValueError("dispatcher timing coverage differs")
        if run["arm"] != "M24":
            legacy = run["controller_elapsed_ns"]
            if len(legacy) != sum(row["controller_returned"] for row in rows):
                raise ValueError("controller timing coverage differs")
            for old, row in zip(legacy, rows):
                if (
                    _integer(old, "legacy controller time")
                    > row["controller_finished_ns"] - row["started_ns"]
                ):
                    raise ValueError("controller duration exceeds its study timing")
    if sum(len(group) for group in dispatcher) != run["physical_steps"]:
        raise ValueError("dispatcher samples do not cover every physical minute")
    for index, group in enumerate(dispatcher):
        values = [_integer(value, "dispatcher time") for value in group]
        if (
            sum(values)
            > rows[index]["finished_ns"] - rows[index]["controller_finished_ns"]
        ):
            raise ValueError("dispatcher duration exceeds simulated interval duration")
    if len(solver) > len(rows):
        raise ValueError("solver timing has an unrecorded decision")
    for index, group in enumerate(solver):
        if (
            sum(group)
            > rows[index]["controller_finished_ns"] - rows[index]["started_ns"]
        ):
            raise ValueError("solver duration exceeds its controller call")
    result["channels"]["dispatcher"] = latency_summary(
        [value for group in dispatcher for value in group]
    )
    result["channels"]["solver"] = latency_summary(
        [value for group in solver for value in group]
    )
    inference = result["channels"]["inference"]
    result["observed_inference_p99_within_limit"] = (
        None if inference["p99"] is None else inference["p99"] <= INFERENCE_LIMIT_NS
    )
    result["physical_decision_count"] = completed
    result["physical_minute_count"] = run["physical_steps"]
    return result


def synthetic_latency_evidence(
    arm_path: Path | str,
    measurement: Mapping[str, Any],
    host_before: Mapping[str, Any],
    host_after: Mapping[str, Any],
) -> dict[str, Any]:
    target = arm_evidence.require_synthetic_path(
        arm_path, filename=arm_evidence.FILENAME
    )
    stamp = arm_evidence.file_stamp(target)
    raw = io.read_regular(target)
    admitted = arm_evidence.verify_synthetic_arm_file(target)
    envelope = decode(raw)
    before, after = validate_latency_host(host_before), validate_latency_host(
        host_after
    )
    if canonical_bytes(before) != canonical_bytes(after):
        raise ValueError("latency host changed during the arm")
    source = admitted["source_provenance"]["runtime"]
    expected_versions = {
        "numpy": source["numpy"],
        "scipy": source["scipy"],
        "gymnasium": source["gymnasium"],
        "stable-baselines3": source["stable_baselines3"],
        "plant-sim": source["plant_sim"]["version"],
    }
    if before["python"] != source["python"] or any(
        before["dependencies"][key] != value for key, value in expected_versions.items()
    ):
        raise ValueError("latency and physical evidence runtime versions differ")
    result = bind_latency_to_verified_run(measurement, envelope["run"])
    result.update(
        source_evidence_verified=True,
        registered_concurrency_verified=False,
        torch_single_thread=before["machine"]["torch_intraop_threads"]
        == before["machine"]["torch_interop_threads"]
        == 1,
    )
    report = authenticate(
        {
            "schema": SCHEMA,
            "latency_contract_sha256": latency_contract()["sha256"],
            "arm_file": admitted["file"],
            "arm_envelope_sha256": admitted["envelope_sha256"],
            "host_before": before,
            "host_after": after,
            "measurement": dict(measurement),
            "result": result,
            "claim_boundary": CLAIM_BOUNDARY,
            "result_claim": None,
        }
    )
    if (
        io.file_record(target) != admitted["file"]
        or io.read_regular(target) != raw
        or arm_evidence.file_stamp(target) != stamp
    ):
        raise ValueError("physical arm changed during latency admission")
    return report


def verify_synthetic_latency_file(path: Path | str) -> dict[str, Any]:
    target = arm_evidence.require_synthetic_path(path, filename=FILENAME)
    before = arm_evidence.file_stamp(target)
    raw = io.read_regular(target)
    report = require_envelope(raw, schema=SCHEMA, fields=FIELDS)
    if (
        report["result_claim"] is not None
        or report["claim_boundary"] != CLAIM_BOUNDARY
        or report["latency_contract_sha256"] != latency_contract()["sha256"]
    ):
        raise ValueError("synthetic latency claim/contract differs")
    reference = report["arm_file"]
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        raise ValueError("latency arm reference is malformed")
    source = arm_evidence.require_synthetic_path(
        reference["path"], filename=arm_evidence.FILENAME
    )
    source_before = arm_evidence.file_stamp(source)
    rebuilt = synthetic_latency_evidence(
        source, report["measurement"], report["host_before"], report["host_after"]
    )
    if canonical_bytes(rebuilt) != raw:
        raise ValueError("latency report differs from reconstructed source evidence")
    record = io.file_record(target)
    if (
        io.read_regular(target) != raw
        or arm_evidence.file_stamp(target) != before
        or io.file_record(source) != rebuilt["arm_file"]
        or arm_evidence.file_stamp(source) != source_before
    ):
        raise ValueError("latency evidence graph changed during verification")
    return {
        **rebuilt["result"],
        "verification": "synthetic_latency_reconstructed",
        "file": record,
    }

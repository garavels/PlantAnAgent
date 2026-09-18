"""One-shot execution and verification for the frozen M24 repair probe.

This module is deliberately additive: the historical environment, controller,
dispatcher, guard, and simulator import no code from here.  The protected runner
admits the old reference directory and the candidate source before constructing an
environment, executes one fresh M24 trajectory per registered cell, publishes each
cell exactly once, and independently reopens every artifact before reporting.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import tempfile
import time
from typing import TYPE_CHECKING, Any, NoReturn

from plant_sim import PlantConfig

from ...controllers.base import Constraints
from .config import EpisodeConfigV2
from .economic_env import PlantEconomicSetpointEnvV2
from .economics import EconomicAssumptionsV1
from .mpc_execution import public_observation_sha256
from .mpc_schema import ServiceMPCSetpointSettingsV1, canonical_json, domain_sha256
from .mpc_setpoint_controller import ServiceMPCSetpointControllerV1
from .mpc_setpoint_decision import MPCSetpointDecisionEvidenceV1
from .mpc_setpoint_execution import MPCSetpointIntervalExecutionV1
from .mpc_setpoint_run import build_mpc_setpoint_run_evidence
from .reward import RewardWeightsV2

if TYPE_CHECKING:
    from .full_day_horizon_admission import (
        AdmittedReferenceCellV1,
        HistoricalCellCitationV1,
        HistoricalReportCitationV1,
        ReferenceAdmissionV1,
        SourceDiffSnapshotV1,
    )
    from .full_day_horizon_repair import FullDayHorizonRepairCellV1


M24_CELL_FORMAT = "full-day-horizon-repair-cell-v1"
M24_REPORT_FORMAT = "full-day-horizon-repair-report-v1"
M24_REPORT_FILENAME = "report.json"
M24_CELL_DOMAIN = "full-day-horizon-repair-cell-v1"
M24_CELL_EVIDENCE_DOMAIN = "full-day-horizon-repair-cell-evidence-v1"
M24_REPORT_DOMAIN = "full-day-horizon-repair-report-v1"
M24_REPORT_EVIDENCE_DOMAIN = "full-day-horizon-repair-report-evidence-v1"

_M24_ARM_FIELDS = frozenset(
    {
        "arm",
        "status",
        "controller",
        "manifest_identity",
        "completed_decisions",
        "physical_steps",
        "terminated",
        "truncated",
        "reward_trace_eur",
        "reward_trace_sum_eur",
        "interval_totals",
        "execution_count",
        "application_count",
        "physical_metrics",
        "economic_metrics",
        "first_boundary_audit",
        "interval_ledger",
        "run_evidence",
        "failure",
        "evidence_validation_failure",
        "elapsed_ns",
    }
)
_M24_CELL_FIELDS = frozenset(
    {
        "format",
        "contract",
        "contract_sha256",
        "status",
        "cell",
        "episode",
        "m24_settings",
        "manifest_identity",
        "historical_citations",
        "started_at_utc",
        "completed_at_utc",
        "elapsed_ns",
        "runtime",
        "m24",
        "analysis_inputs",
        "failure",
        "deterministic_evidence_sha256",
        "sha256",
    }
)
_M24_REPORT_FIELDS = frozenset(
    {
        "format",
        "contract",
        "contract_sha256",
        "status",
        "claim_boundary",
        "started_at_utc",
        "completed_at_utc",
        "elapsed_ns",
        "runtime",
        "source_gate",
        "reference_report_path",
        "reference",
        "cells",
        "custody",
        "aggregate",
        "deterministic_evidence_sha256",
        "sha256",
    }
)


def _require_exact_keys(
    name: str, value: Mapping[str, Any], expected: frozenset[str]
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} fields differ; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return value


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _failure(phase: str, reason: BaseException) -> dict[str, Any]:
    return {
        "phase": phase,
        "exception_type": type(reason).__name__,
        "message": (" ".join(str(reason).split()) or "(no exception message)")[:500],
    }


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number {value!r} is forbidden")
    return parsed


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )


def _read_regular_bytes(path: Path) -> bytes:
    """Read one stable regular file without following a substituted symlink."""

    before = path.lstat()
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(f"symlink artifact is forbidden: {path}")
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"artifact must be a regular file: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if _stat_identity(opened) != _stat_identity(before):
            raise RuntimeError(f"artifact changed while opening: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if _stat_identity(after) != _stat_identity(before):
        raise RuntimeError(f"artifact changed while reading: {path}")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise RuntimeError(f"artifact size changed while reading: {path}")
    return raw


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} is forbidden")
            result[key] = value
        return result

    raw = _read_regular_bytes(path)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as reason:
        raise ValueError(f"JSON artifact is not UTF-8: {path}") from reason
    decoded = json.loads(
        text,
        object_pairs_hook=reject_duplicates,
        parse_constant=_reject_json_constant,
        parse_float=_finite_json_float,
    )
    payload = dict(_mapping(str(path), decoded))
    expected = (canonical_json(payload) + "\n").encode("utf-8")
    if raw != expected:
        raise ValueError(
            f"JSON artifact is not canonical with one trailing newline: {path}"
        )
    return payload


def _publish_exclusive(target: Path, payload: Mapping[str, Any]) -> str:
    """Durably publish canonical JSON without replacing an existing artifact."""

    encoded = (canonical_json(dict(payload)) + "\n").encode("utf-8")
    raw_sha256 = hashlib.sha256(encoded).hexdigest()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    if _read_regular_bytes(target) != encoded:
        raise RuntimeError("published M24 artifact differs from encoded bytes")
    return raw_sha256


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, str]:
    assumptions = dict(
        _mapping("economic assumptions", manifest["economic_assumptions"])
    )
    checkpoint = _mapping("checkpoint contract", manifest["checkpoint_contract"])
    return {
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "physical_exogenous_episode_sha256": str(manifest["exogenous_episode_sha256"]),
        "economic_episode_sha256": str(manifest["economic_episode_sha256"]),
        "checkpoint_contract_sha256": str(checkpoint["checkpoint_contract_digest"]),
        "economic_assumptions_sha256": hashlib.sha256(
            canonical_json(assumptions).encode("utf-8")
        ).hexdigest(),
    }


def _prepared_environment_identity(
    environment: PlantEconomicSetpointEnvV2,
) -> dict[str, str]:
    if environment.decisions_taken or environment.physics_steps_taken:
        raise ValueError("prepared M24 environment has already moved")
    identity = _manifest_identity(environment.audit_manifest())
    if (
        environment.economic_assumptions.sha256
        != identity["economic_assumptions_sha256"]
    ):
        raise RuntimeError("economic assumptions identity drifted before M24 action")
    return identity


def _interval_row(
    *,
    event: MPCSetpointDecisionEvidenceV1,
    execution: MPCSetpointIntervalExecutionV1,
    reward_eur: float,
    info: Mapping[str, Any],
    physical_before: Mapping[str, Any],
    physical_after: Mapping[str, Any],
) -> dict[str, Any]:
    if not event.closed or event.residual is None or event.execution is None:
        raise RuntimeError("M24 interval event was not transition-reconciled")
    if event.execution.sha256 != execution.sha256:
        raise RuntimeError("M24 controller and environment executions differ")
    reward = dict(_mapping("economic reward breakdown", info["reward"]))
    delivery = dict(_mapping("delivery transition", info["delivery"]))
    economic_interval = dict(_mapping("economic interval", info["economic_interval"]))
    if not math.isclose(
        float(reward["total_eur"]), reward_eur, rel_tol=0.0, abs_tol=1e-9
    ):
        raise RuntimeError("M24 scalar reward differs from its component record")
    from .full_day_horizon_repair import build_interval_ledger_row

    first_application = execution.applications[0]
    last_application = execution.applications[-1]
    execution_energy = _mapping(
        "M24 interval execution energy", execution.to_dict()["energy_kwh"]
    )
    applied_energy = _mapping(
        "M24 interval applied energy", execution_energy["applied"]
    )
    energy = {
        "accepted_solar_kwh": float(economic_interval["accepted_solar_kwh"]),
        "accepted_process_energy_kwh": max(
            0.0,
            float(physical_after["accepted_process_energy_kwh"])
            - float(physical_before["accepted_process_energy_kwh"]),
        ),
        "curtailed_kwh": max(
            0.0,
            float(physical_after["curtailed_kwh"])
            - float(physical_before["curtailed_kwh"]),
        ),
        "battery_throughput_kwh": float(economic_interval["battery_throughput_kwh"]),
        "battery_support_kwh": float(applied_energy["battery_support"]),
        "electrolyser_equivalent_operating_h": float(
            economic_interval["electrolyser_equivalent_operating_h"]
        ),
    }
    return build_interval_ledger_row(
        decision_index=event.decision_index,
        bindings={
            "event_sha256": event.sha256,
            "observation_sha256": event.observation_sha256,
            "public_forecast_sha256": event.public_forecast_sha256,
            "projection_sha256": event.projection_sha256,
            "setpoints_sha256": event.setpoints.sha256,
            "execution_sha256": execution.sha256,
            "residual_sha256": event.residual.sha256,
            "boundary_sha256": first_application.boundary_sha256,
            "next_observation_sha256": event.residual.next_observation_sha256,
            "next_boundary_sha256": last_application.next_boundary_sha256,
        },
        reward_eur=float(reward_eur),
        delivery=delivery,
        economic_interval=economic_interval,
        reward_breakdown=reward,
        energy=energy,
    )


def execute_m24_arm(
    *,
    episode: EpisodeConfigV2,
    plant: PlantConfig,
    constraints: Constraints,
    weights: RewardWeightsV2,
    assumptions: EconomicAssumptionsV1,
    settings: ServiceMPCSetpointSettingsV1,
    seed: int,
    environment: PlantEconomicSetpointEnvV2,
) -> dict[str, Any]:
    """Execute one real M24 trajectory with complete interval and run evidence."""

    if not isinstance(environment, PlantEconomicSetpointEnvV2):
        raise TypeError("environment must be PlantEconomicSetpointEnvV2")
    if not isinstance(settings, ServiceMPCSetpointSettingsV1):
        raise TypeError("settings must be ServiceMPCSetpointSettingsV1")
    if settings.task_horizon_h != 24.0 or settings.planning_horizon_h != 24.0:
        raise ValueError("M24 execution requires a 24-hour task and planning horizon")
    if episode.horizon_h != 24.0 or episode.decision_interval_s != settings.interval_s:
        raise ValueError("M24 episode and controller horizons/cadences differ")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    # Imported lazily so codec/contract unit tests remain detached from execution.
    from .full_day_horizon_codec import verify_mpc_setpoint_run_semantics
    from .full_day_horizon_repair import (
        build_first_boundary_audit,
        build_interval_ledger,
    )

    manifest_identity = _prepared_environment_identity(environment)
    controller = ServiceMPCSetpointControllerV1(settings)
    controller.reset()
    reward_trace: list[float] = []
    interval_rows: list[dict[str, Any]] = []
    dispatcher_elapsed_ns: list[tuple[int, ...]] = []
    first_boundary_audit: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    terminated = truncated = False
    phase = "first_boundary_audit"
    started_ns = time.perf_counter_ns()
    try:
        first_boundary_audit = build_first_boundary_audit(
            environment.controller_observation,
            config=plant,
            interval_s=settings.interval_s,
        )
        while not (terminated or truncated):
            physical_before = environment.metrics.to_dict()
            phase = "controller_compute"
            decision = controller.compute_setpoints(
                environment.controller_observation,
                constraints=constraints,
                config=plant,
                commitment_profile=episode.electrolyser_commitment,
            )
            phase = "environment_setpoint_execution"
            _, reward, terminated, truncated, info = environment.step_setpoints(
                decision.setpoints
            )
            execution = environment.last_setpoint_execution
            if not isinstance(execution, MPCSetpointIntervalExecutionV1):
                raise RuntimeError("M24 environment omitted typed execution evidence")
            elapsed = info.get("dispatcher_elapsed_ns")
            if not isinstance(elapsed, list) or any(
                isinstance(item, bool) or not isinstance(item, Integral) or item < 0
                for item in elapsed
            ):
                raise RuntimeError("M24 dispatcher timing evidence is malformed")
            phase = "transition_reconciliation"
            controller.observe_transition(
                environment.controller_observation,
                execution=execution,
            )
            event = controller.decision_events[-1]
            if not isinstance(event, MPCSetpointDecisionEvidenceV1):
                raise RuntimeError(
                    "M24 controller omitted successful decision evidence"
                )
            reward_value = float(reward)
            interval_rows.append(
                _interval_row(
                    event=event,
                    execution=execution,
                    reward_eur=reward_value,
                    info=_mapping("M24 environment info", info),
                    physical_before=physical_before,
                    physical_after=environment.metrics.to_dict(),
                )
            )
            reward_trace.append(reward_value)
            dispatcher_elapsed_ns.append(tuple(int(item) for item in elapsed))
            if truncated and not terminated:
                raise RuntimeError("M24 episode truncated before its intrinsic horizon")
    except Exception as reason:
        failure = _failure(phase, reason)

    elapsed_ns = time.perf_counter_ns() - started_ns
    closed_decisions = sum(
        isinstance(event, MPCSetpointDecisionEvidenceV1) and event.closed
        for event in controller.decision_events
    )
    run_succeeded = bool(
        failure is None
        and terminated
        and not truncated
        and closed_decisions == settings.task_interval_count
        and not controller.has_pending_transition
    )
    if not run_succeeded and failure is None:
        failure = {
            "phase": "completion_validation",
            "exception_type": "IncompleteM24Episode",
            "message": "M24 episode did not close its declared horizon",
        }

    run_evidence: dict[str, Any] | None = None
    interval_ledger: dict[str, Any] | None = None
    evidence_failure: dict[str, Any] | None = None
    try:
        run_evidence = build_mpc_setpoint_run_evidence(
            controller.decision_events,
            tuple(dispatcher_elapsed_ns),
            settings=settings,
            completed_decisions=closed_decisions,
            run_succeeded=run_succeeded,
            pending_transition=controller.has_pending_transition,
            runner_failure_phase=None if failure is None else str(failure["phase"]),
            final_observation_sha256=public_observation_sha256(
                environment.controller_observation
            ),
        )
        verify_mpc_setpoint_run_semantics(run_evidence, settings=settings)
        physical_metrics = environment.metrics.to_dict()
        economic_metrics = environment.economic_metrics.to_dict()
        interval_ledger = build_interval_ledger(
            interval_rows,
            run_evidence=run_evidence,
            reward_trace_eur=reward_trace,
            physical_metrics=physical_metrics,
            economic_metrics=economic_metrics,
        )
    except Exception as reason:
        evidence_failure = _failure("evidence_validation", reason)
        if failure is None:
            failure = evidence_failure
        run_succeeded = False

    return {
        "arm": "M24",
        "status": "complete" if run_succeeded else "failed",
        "controller": controller.to_dict(),
        "manifest_identity": manifest_identity,
        "completed_decisions": environment.decisions_taken,
        "physical_steps": environment.physics_steps_taken,
        "terminated": terminated,
        "truncated": truncated,
        "reward_trace_eur": reward_trace,
        "reward_trace_sum_eur": float(math.fsum(reward_trace)),
        "interval_totals": (
            None
            if interval_ledger is None
            else dict(_mapping("M24 interval totals", interval_ledger["totals"]))
        ),
        "execution_count": len(interval_rows),
        "application_count": sum(
            len(event.execution.applications)
            for event in controller.decision_events
            if isinstance(event, MPCSetpointDecisionEvidenceV1)
            and event.execution is not None
        ),
        "physical_metrics": environment.metrics.to_dict(),
        "economic_metrics": environment.economic_metrics.to_dict(),
        "first_boundary_audit": first_boundary_audit,
        "interval_ledger": interval_ledger,
        "run_evidence": run_evidence,
        "failure": failure,
        "evidence_validation_failure": evidence_failure,
        "elapsed_ns": elapsed_ns,
    }


def _cell_filename(index: int, cell_id: str) -> str:
    return f"{index:02d}_{cell_id}.json"


def _run_git(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Git source admission failed: {detail or arguments[0]}")
    return completed.stdout


def _source_diff_snapshot(
    repository: Path, historical_revision: str
) -> SourceDiffSnapshotV1:
    """Return an authenticated clean HEAD/name-status snapshot for admission."""

    from .full_day_horizon_admission import (
        SourceDiffEntryV1,
        SourceDiffSnapshotV1,
    )

    top_level = Path(
        _run_git(repository, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    ).resolve(strict=True)
    if top_level != repository.resolve(strict=True):
        raise ValueError("M24 repository root is not the Git top level")
    target_revision = _run_git(repository, "rev-parse", "HEAD").decode().strip()
    dirty = _run_git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    raw = _run_git(
        repository,
        "diff",
        "--name-status",
        "-z",
        "--find-renames",
        historical_revision,
        target_revision,
        "--",
    )
    tokens = raw.decode("utf-8").split("\0")
    if tokens and tokens[-1] == "":
        tokens.pop()
    entries: list[SourceDiffEntryV1] = []
    cursor = 0
    while cursor < len(tokens):
        status_code = tokens[cursor]
        cursor += 1
        if status_code.startswith(("R", "C")):
            if cursor + 1 >= len(tokens):
                raise ValueError("Git emitted a truncated rename/copy source diff")
            old_path, new_path = tokens[cursor : cursor + 2]
            cursor += 2
            entries.append(
                SourceDiffEntryV1(
                    status=status_code,
                    path=new_path,
                    old_path=old_path,
                )
            )
        else:
            if cursor >= len(tokens):
                raise ValueError("Git emitted a truncated source diff")
            entries.append(SourceDiffEntryV1(status=status_code, path=tokens[cursor]))
            cursor += 1
    return SourceDiffSnapshotV1(
        base_revision=historical_revision,
        target_revision=target_revision,
        worktree_clean=not dirty,
        entries=tuple(entries),
    )


def _contract_file_identities(repository: Path) -> dict[str, str]:
    from .full_day_horizon_repair import (
        FULL_DAY_HORIZON_REPAIR_CLARIFICATION_001_PATH,
        FULL_DAY_HORIZON_REPAIR_CLARIFICATION_001_SHA256,
        FULL_DAY_HORIZON_REPAIR_CLARIFICATION_002_PATH,
        FULL_DAY_HORIZON_REPAIR_CLARIFICATION_002_SHA256,
        FULL_DAY_HORIZON_REPAIR_LOCK_PATH,
        FULL_DAY_HORIZON_REPAIR_LOCK_SHA256,
        FULL_DAY_HORIZON_REPAIR_PROTOCOL_PATH,
        FULL_DAY_HORIZON_REPAIR_PROTOCOL_SHA256,
    )

    expected = {
        "protocol_sha256": (
            FULL_DAY_HORIZON_REPAIR_PROTOCOL_PATH,
            FULL_DAY_HORIZON_REPAIR_PROTOCOL_SHA256,
        ),
        "clarification_001_sha256": (
            FULL_DAY_HORIZON_REPAIR_CLARIFICATION_001_PATH,
            FULL_DAY_HORIZON_REPAIR_CLARIFICATION_001_SHA256,
        ),
        "clarification_002_sha256": (
            FULL_DAY_HORIZON_REPAIR_CLARIFICATION_002_PATH,
            FULL_DAY_HORIZON_REPAIR_CLARIFICATION_002_SHA256,
        ),
        "dependency_lock_sha256": (
            FULL_DAY_HORIZON_REPAIR_LOCK_PATH,
            FULL_DAY_HORIZON_REPAIR_LOCK_SHA256,
        ),
    }
    observed: dict[str, str] = {}
    for label, (relative_path, required_sha256) in expected.items():
        path = repository / relative_path
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(
                f"M24 contract input must be a regular file: {relative_path}"
            )
        digest = hashlib.sha256(_read_regular_bytes(path)).hexdigest()
        if digest != required_sha256:
            raise ValueError(f"M24 contract input identity drifted: {relative_path}")
        observed[label] = digest
    return observed


def _validate_experiment_source_scope(source_admission: Any) -> None:
    """Limit the historical delta to docs/tests/Makefile and four new modules."""

    from .full_day_horizon_admission import ALLOWED_NEW_M24_RUNTIME_PATHS

    allowed_runtime = set(ALLOWED_NEW_M24_RUNTIME_PATHS)

    def allowed(path: str) -> bool:
        return bool(
            path in allowed_runtime
            or path == "Makefile"
            or ("/" not in path and path.endswith(".md"))
            or path.startswith("docs/")
            or path.startswith("tests/")
            or "/tests/" in path
            or path.startswith("backend/tests/")
            or path.startswith("simulator/tests/")
        )

    undeclared: list[str] = []
    for entry in source_admission.entries:
        paths = [entry.path]
        if entry.old_path is not None:
            paths.append(entry.old_path)
        undeclared.extend(path for path in paths if not allowed(path))
    if undeclared:
        raise ValueError(
            f"M24 source delta contains undeclared paths: {sorted(set(undeclared))}"
        )


def _deep_verify_historical_report(
    report_path: Path,
    *,
    repository: Path,
) -> Mapping[str, Any]:
    """Run the old deep verifier and the stronger typed codec over all 20 M6 runs."""

    from .economic_robustness_experiment import verify_robustness_report
    from .full_day_horizon_admission import registered_reference_filenames
    from .full_day_horizon_codec import verify_mpc_setpoint_run_semantics

    result = verify_robustness_report(
        report_path,
        require_registered=True,
        repository_root=repository,
    )
    for filename in registered_reference_filenames():
        payload = _load_json(report_path.parent / filename)
        cell = _mapping("historical cell", payload["cell"])
        horizon_h = float(cell["horizon_h"])
        settings = ServiceMPCSetpointSettingsV1(
            task_horizon_h=horizon_h,
            planning_horizon_h=6.0,
        )
        if settings.to_dict() != payload.get("mpc_settings"):
            raise ValueError("historical M6 settings differ from typed reconstruction")
        arms = _mapping("historical arms", payload["arms"])
        m6 = _mapping("historical M6 arm", arms["M"])
        evidence = _mapping("historical M6 evidence", m6["evidence"])
        verify_mpc_setpoint_run_semantics(evidence, settings=settings)
    return result


def _runtime_identity() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }


def _historical_runtime(report_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    source_gate = _mapping("historical source gate", report_payload["source_gate"])
    start = _mapping("historical source start", source_gate["start"])
    return _mapping("historical runtime", start["runtime"])


def _latency_summary(values: Sequence[object]) -> dict[str, int]:
    parsed = [_integer("latency", value) for value in values]
    if not parsed:
        return {"count": 0, "p99_ns": 0, "maximum_ns": 0}
    ordered = sorted(parsed)
    rank = max(1, math.ceil(0.99 * len(ordered)))
    return {
        "count": len(ordered),
        "p99_ns": ordered[rank - 1],
        "maximum_ns": ordered[-1],
    }


def _reported_latency(
    run: Mapping[str, Any], *, arm: str, source_arm_sha256: str
) -> dict[str, Any]:
    from .full_day_horizon_repair import measurement

    if arm == "H":
        evidence = _mapping("historical H evidence", run["evidence"])
        controller_values = evidence.get("controller_elapsed_ns", [])
        nested_dispatcher = evidence.get("dispatcher_elapsed_ns", [])
        if not isinstance(controller_values, list) or not isinstance(
            nested_dispatcher, list
        ):
            raise TypeError("historical H latency evidence is malformed")
        dispatcher_values: list[object] = []
        for group in nested_dispatcher:
            if not isinstance(group, list):
                raise TypeError("historical H dispatcher latency is malformed")
            dispatcher_values.extend(group)
        return {
            "arm_elapsed_ns": _integer("H arm elapsed", run["elapsed_ns"]),
            "controller": _latency_summary(controller_values),
            "solver_stage": measurement("not_applicable"),
            "dispatcher": _latency_summary(dispatcher_values),
        }
    evidence = _mapping(
        f"{arm} run evidence",
        run["run_evidence"] if arm == "M24" else run["evidence"],
    )
    decision = _mapping(f"{arm} decision latency", evidence["decision_latency"])
    dispatcher = _mapping(f"{arm} dispatcher latency", evidence["dispatcher_latency"])
    solver = _mapping(f"{arm} solver latency", evidence["solver_stage_latency"])
    return {
        "arm_elapsed_ns": _integer(f"{arm} arm elapsed", run["elapsed_ns"]),
        "controller": {
            "count": _integer("controller latency count", decision["count"]),
            "p99_ns": _integer("controller p99", decision["p99_ns"]),
            "maximum_ns": _integer("controller maximum", decision["maximum_ns"]),
        },
        "dispatcher": {
            "count": _integer("dispatcher latency count", dispatcher["count"]),
            "p99_ns": _integer("dispatcher p99", dispatcher["p99_ns"]),
            "maximum_ns": _integer("dispatcher maximum", dispatcher["maximum_ns"]),
        },
        "solver_stage": measurement(
            "available",
            {
                "count": _integer("solver latency count", solver["count"]),
                "p99_ns": _integer("solver p99", solver["p99_ns"]),
                "maximum_ns": _integer("solver maximum", solver["maximum_ns"]),
            },
            source_arm_sha256=source_arm_sha256,
        ),
    }


def _service_measurements(run: Mapping[str, Any]) -> dict[str, Any]:
    physical = _mapping("arm physical metrics", run["physical_metrics"])
    return {
        "requested_kg": float(physical["requested_kg"]),
        "delivered_kg": float(physical["delivered_kg"]),
        "unmet_kg": float(physical["unmet_kg"]),
        "delivery_fill": float(physical["delivery_fill_rate"]),
        "commitments_settled": _integer(
            "settled commitments", physical["commitments_settled"]
        ),
        "commitments_met": _integer("met commitments", physical["commitments_met"]),
        "worst_commitment_fill": float(physical["worst_commitment_fill_rate"]),
    }


def _physical_measurements(run: Mapping[str, Any], *, arm: str) -> dict[str, Any]:
    physical = dict(_mapping(f"{arm} physical metrics", run["physical_metrics"]))
    if arm == "H":
        evidence = _mapping("historical H evidence", run["evidence"])
        raw_executions = evidence.get("executions")
        if not isinstance(raw_executions, list):
            raise TypeError("historical H executions must be a list")
        support_kwh = 0.0
        plant_refusals = 0
        target_deficits = 0
        for raw_execution in raw_executions:
            execution = _mapping("historical H execution", raw_execution)
            physics_step_s = float(execution["physics_step_s"])
            applications = execution.get("applications")
            if not isinstance(applications, list):
                raise TypeError("historical H applications must be a list")
            for raw_application in applications:
                application = _mapping("historical H application", raw_application)
                plant = _mapping("historical H application plant", application["plant"])
                applied = _mapping("historical H applied power", plant["applied_kw"])
                dispatch = _mapping("historical H dispatch", application["dispatch"])
                allocation = _mapping("historical H allocation", dispatch["allocation"])
                targets = _mapping(
                    "historical H targets",
                    _mapping("historical H target record", allocation["targets"])[
                        "setpoints_kw"
                    ],
                )
                request = _mapping(
                    "historical H request power", allocation["request_kw"]
                )
                guard = _mapping("historical H guard", application["guard"])
                guarded_action = _mapping(
                    "historical H guarded action", guard["action"]
                )
                support_kwh += (
                    float(applied["battery_support"]) * physics_step_s / 3_600.0
                )
                target_kw = float(targets["electrolyser"])
                requested_kw = float(request["electrolyser_total"])
                guarded_kw = float(guarded_action["electrolyser_share"]) * float(
                    applied["solar_available"]
                ) + float(guarded_action["battery_support_kw"])
                if requested_kw > target_kw + 1e-9:
                    raise ValueError("historical H request exceeds its target")
                plant_refusals += int(
                    guarded_kw - float(applied["electrolyser"]) > 1e-9
                )
                target_deficits += int(
                    target_kw - float(applied["electrolyser"]) > 1e-9
                )
        physical["battery_support_kwh"] = support_kwh
        physical["plant_refusals"] = plant_refusals
        physical["target_deficits"] = target_deficits
    else:
        evidence = _mpc_evidence(run, arm=arm)
        counts = _mapping(f"{arm} run counts", evidence["counts"])
        applied = _mapping(
            f"{arm} applied energy",
            _mapping(f"{arm} run energy", evidence["energy_kwh"])["applied"],
        )
        physical["battery_support_kwh"] = float(applied["battery_support"])
        physical["plant_refusals"] = _integer(
            f"{arm} plant refusals", counts["plant_target_refusal_minutes"]
        )
        physical["target_deficits"] = _integer(
            f"{arm} target deficits", counts["target_deficit_minutes"]
        )
    return physical


def _mpc_evidence(run: Mapping[str, Any], *, arm: str) -> Mapping[str, Any]:
    return _mapping(
        f"{arm} run evidence",
        run["run_evidence"] if arm == "M24" else run["evidence"],
    )


def _mpc_events(run: Mapping[str, Any], *, arm: str) -> list[Mapping[str, Any]]:
    events = _mpc_evidence(run, arm=arm).get("events")
    if not isinstance(events, list):
        raise TypeError(f"{arm} events must be a list")
    return [_mapping(f"{arm} event", event) for event in events]


def _settlement_measurements(
    run: Mapping[str, Any],
    *,
    arm: str,
) -> list[dict[str, Any]]:
    events = _mpc_events(run, arm=arm)
    if arm == "M24":
        raw_targets = _mapping("M24 event zero", events[0]).get("deliveries")
        if not isinstance(raw_targets, list) or len(raw_targets) != 4:
            raise ValueError("M24 event zero must retain four delivery targets")
        targets = [_mapping("M24 delivery target", item) for item in raw_targets]
    else:
        targets = []
        for boundary in (24, 48, 72, 96):
            deliveries = _mapping("M6 settlement event", events[boundary - 1]).get(
                "deliveries"
            )
            if not isinstance(deliveries, list) or not deliveries:
                raise ValueError("M6 settlement event omits its announced target")
            targets.append(_mapping("M6 announced delivery target", deliveries[0]))
    ledger_rows: list[Mapping[str, Any]] | None = None
    if arm == "M24":
        ledger = _mapping("M24 interval ledger", run["interval_ledger"])
        raw_rows = ledger.get("rows")
        if not isinstance(raw_rows, list):
            raise TypeError("M24 interval-ledger rows must be a list")
        ledger_rows = [_mapping("M24 interval row", row) for row in raw_rows]
    settlements: list[dict[str, Any]] = []
    for ordinal, target in enumerate(targets, start=1):
        boundary = 24 * ordinal
        if (
            arm == "M24"
            and _integer("delivery due boundary", target["due_boundary"]) != boundary
        ):
            raise ValueError("M24 delivery targets have wrong due boundaries")
        event = _mapping(f"{arm} settlement event", events[boundary - 1])
        residual = _mapping(f"{arm} settlement residual", event["residual"])
        requested = float(target["requested_kg"])
        residual_delivered = float(residual["observed_delivered_kg"])
        if ledger_rows is None:
            delivered = residual_delivered
            unmet = max(0.0, requested - delivered)
        else:
            delivery = _mapping(
                "M24 settlement delivery", ledger_rows[boundary - 1]["delivery"]
            )
            if (
                _integer(
                    "M24 interval settlement count", delivery["commitments_settled"]
                )
                != 1
            ):
                raise ValueError("M24 due-boundary interval omitted its settlement")
            delivered = float(delivery["delivered_kg"])
            unmet = float(delivery["unmet_kg"])
            if not math.isclose(
                delivered, residual_delivered, rel_tol=0.0, abs_tol=1e-8
            ):
                raise ValueError("M24 interval and residual delivery differ")
        settlements.append(
            {
                "ordinal": ordinal,
                "requested_kg": requested,
                "delivered_kg": delivered,
                "unmet_kg": unmet,
                "fill": 1.0 if requested == 0.0 else delivered / requested,
            }
        )
    return settlements


def _boundary_record(event: Mapping[str, Any], *, first: bool) -> Mapping[str, Any]:
    execution = _mapping("event interval execution", event["execution"])
    raw_applications = execution.get("applications")
    if not isinstance(raw_applications, list) or not raw_applications:
        raise ValueError("event execution omits minute applications")
    application = _mapping(
        "event boundary application",
        raw_applications[0] if first else raw_applications[-1],
    )
    side = "boundary" if first else "next_boundary"
    return _mapping(
        "event physical boundary", _mapping(side, application[side])["record"]
    )


def _post_settlement_states(
    run: Mapping[str, Any], *, arm: str
) -> dict[str, dict[str, float]]:
    events = _mpc_events(run, arm=arm)
    result: dict[str, dict[str, float]] = {}
    for ordinal, boundary in ((2, 48), (3, 72)):
        record = _boundary_record(events[boundary - 1], first=False)
        result[str(ordinal)] = {
            "h2_kg": float(record["h2_level_kg"]),
            "co2_kg": float(record["co2_level_kg"]),
            "ch4_kg": float(record["ch4_level_kg"]),
            "battery_energy_kwh": float(record["battery_energy_kwh"]),
        }
    return result


def _first_decision_measurement(run: Mapping[str, Any], *, arm: str) -> dict[str, Any]:
    event = _mpc_events(run, arm=arm)[0]
    return {
        "decision_index": event["decision_index"],
        "horizon_intervals": event["horizon_intervals"],
        "observation_sha256": event["observation_sha256"],
        "public_forecast_sha256": event["public_forecast_sha256"],
        "projection_sha256": event["projection_sha256"],
        "deliveries": event["deliveries"],
    }


def _final_settlement_balance(
    run: Mapping[str, Any],
    *,
    arm: str,
    settlements: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    event = _mpc_events(run, arm=arm)[95]
    before = _boundary_record(event, first=True)
    after = _boundary_record(event, first=False)
    residual = _mapping("final-settlement residual", event["residual"])
    return {
        "ch4_before_final_interval_kg": float(before["ch4_level_kg"]),
        "ch4_production_delta_kg": max(
            0.0,
            float(after["ch4_produced_total_kg"])
            - float(before["ch4_produced_total_kg"]),
        ),
        "ch4_after_settlement_kg": float(after["ch4_level_kg"]),
        "residual_delivered_kg": float(residual["observed_delivered_kg"]),
        "interval_delivered_kg": float(settlements[3]["delivered_kg"]),
    }


def _compact_arm_measurements(
    run: Mapping[str, Any],
    *,
    arm: str,
    source_arm_sha256: str,
) -> dict[str, Any]:
    from .full_day_horizon_repair import measurement, verify_compact_arm_measurements

    if arm not in {"H", "M6", "M24"}:
        raise ValueError("compact arm label is unsupported")
    physical = _physical_measurements(run, arm=arm)
    economic = dict(_mapping(f"{arm} economic metrics", run["economic_metrics"]))
    if arm == "H":
        compact = {
            "service": _service_measurements(run),
            "settlements": measurement("not_retained"),
            "post_settlement_states": measurement("not_retained"),
            "first_decision": measurement("not_applicable"),
            "interval_economic_components": measurement("not_retained"),
            "final_settlement_balance": measurement("not_retained"),
            "physical": physical,
            "economic": economic,
            "latency": _reported_latency(
                run, arm=arm, source_arm_sha256=source_arm_sha256
            ),
        }
    else:
        settlements = _settlement_measurements(run, arm=arm)
        run_sha = source_arm_sha256
        if arm == "M24":
            ledger = _mapping("M24 interval ledger", run["interval_ledger"])
            rows = ledger.get("rows")
            if not isinstance(rows, list):
                raise TypeError("M24 interval rows must be a list")
            interval_components = measurement(
                "available",
                [
                    _mapping(
                        "M24 interval reward breakdown",
                        _mapping("M24 interval row", row)["reward_breakdown"],
                    )["components_eur"]
                    for row in rows
                ],
                source_arm_sha256=run_sha,
            )
        else:
            interval_components = measurement("not_retained")
        compact = {
            "service": _service_measurements(run),
            "settlements": measurement(
                "available", settlements, source_arm_sha256=run_sha
            ),
            "post_settlement_states": measurement(
                "available",
                _post_settlement_states(run, arm=arm),
                source_arm_sha256=run_sha,
            ),
            "first_decision": measurement(
                "available",
                _first_decision_measurement(run, arm=arm),
                source_arm_sha256=run_sha,
            ),
            "interval_economic_components": interval_components,
            "final_settlement_balance": measurement(
                "available",
                _final_settlement_balance(
                    run,
                    arm=arm,
                    settlements=settlements,
                ),
                source_arm_sha256=run_sha,
            ),
            "physical": physical,
            "economic": economic,
            "latency": _reported_latency(
                run, arm=arm, source_arm_sha256=source_arm_sha256
            ),
        }
    verify_compact_arm_measurements(arm, compact)
    return compact


def _reference_citations(
    report_citation: HistoricalReportCitationV1,
    cell_citation: HistoricalCellCitationV1,
) -> dict[str, Any]:
    report = report_citation.to_dict()
    cell = cell_citation.to_dict()
    arms = _mapping("historical arm citations", cell["arms"])
    heuristic = _mapping("historical H citation", arms["H"])
    m6 = _mapping("historical M6 citation", arms["M6"])
    return {
        "root_report": {
            "raw_file_sha256": report["raw_file_sha256"],
            "report_sha256": report["report_domain_sha256"],
            "deterministic_evidence_sha256": report["deterministic_evidence_sha256"],
            "contract_sha256": report["contract_sha256"],
        },
        "historical_cell": {
            "file_name": cell["file_name"],
            "raw_file_sha256": cell["raw_file_sha256"],
            "report_sha256": cell["cell_domain_sha256"],
            "deterministic_evidence_sha256": cell["deterministic_evidence_sha256"],
        },
        "H": {
            "controller_id": heuristic["controller_id"],
            "evidence_sha256": heuristic["retained_evidence_sha256"],
            "deterministic_arm_sha256": heuristic["deterministic_arm_sha256"],
        },
        "M6": {
            "controller_id": m6["controller_id"],
            "settings_sha256": m6["settings_sha256"],
            "run_evidence_sha256": m6["retained_evidence_sha256"],
            "evidence_graph_sha256": m6["evidence_graph_sha256"],
            "deterministic_arm_sha256": m6["deterministic_arm_sha256"],
        },
    }


def _analysis_citations(
    historical: Mapping[str, Any],
    m24: Mapping[str, Any],
) -> dict[str, Any]:
    run_evidence = _mapping("M24 run evidence", m24["run_evidence"])
    ledger = _mapping("M24 interval ledger", m24["interval_ledger"])
    return {
        **dict(historical),
        "M24": {
            "controller_id": _mapping("M24 controller", m24["controller"])["id"],
            "settings_sha256": run_evidence["settings_sha256"],
            "interval_evidence_sha256": ledger["sha256"],
            "run_evidence_sha256": run_evidence["sha256"],
            "evidence_graph_sha256": run_evidence["evidence_graph_sha256"],
        },
    }


def _reconstruct_m24_safety_counts(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Rebuild safety counts from typed minute and commitment evidence."""

    shutdown_steps = 0
    forced_stops = 0
    for event in events:
        execution = _mapping("M24 safety execution", event["execution"])
        applications = execution.get("applications")
        if not isinstance(applications, list):
            raise TypeError("M24 safety applications must be a list")
        for raw_application in applications:
            application = _mapping("M24 safety application", raw_application)
            next_boundary = _mapping(
                "M24 safety next boundary", application["next_boundary"]
            )
            next_record = _mapping(
                "M24 safety next-boundary record", next_boundary["record"]
            )
            shutdown_steps += int(
                next_record["plant_status"] in {"critical", "offline"}
            )
            commitment = _mapping("M24 safety commitment", application["commitment"])
            before = _mapping("M24 commitment before", commitment["before"])
            after = _mapping("M24 commitment after", commitment["after"])
            before_record = before.get("record")
            after_record = after.get("record")
            if before_record is None or after_record is None:
                if before_record is not None or after_record is not None:
                    raise ValueError("M24 commitment evidence changes availability")
                continue
            before_values = _mapping("M24 commitment-before record", before_record)
            after_values = _mapping("M24 commitment-after record", after_record)
            delta = _integer(
                "M24 commitment forced-stop counter", after_values["forced_stops"]
            ) - _integer(
                "M24 prior commitment forced-stop counter",
                before_values["forced_stops"],
            )
            if delta < 0:
                raise ValueError("M24 forced-stop counter moved backwards")
            forced_stops += delta
    # A decoded setpoint trajectory never traverses the clipped continuous-action route.
    return {
        "shutdown_steps": shutdown_steps,
        "out_of_range_decisions": 0,
        "electrolyser_forced_stops": forced_stops,
    }


def _m24_evidence_checks(
    m24: Mapping[str, Any],
    *,
    reference_identity: Mapping[str, Any],
    reference_m6: Mapping[str, Any],
    reference_m6_settings: Mapping[str, Any],
    m24_settings: Mapping[str, Any],
    historical_runtime: Mapping[str, Any],
    current_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    from .full_day_horizon_codec import verify_mpc_setpoint_run_semantics
    from .full_day_horizon_repair import (
        FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
        FULL_DAY_HORIZON_REPAIR_EXPECTED_DECISIONS_PER_CELL,
        FULL_DAY_HORIZON_REPAIR_EXPECTED_PHYSICAL_MINUTES_PER_CELL,
        FULL_DAY_HORIZON_REPAIR_EXPECTED_SETTLEMENTS_PER_CELL,
        FULL_DAY_HORIZON_REPAIR_EXPECTED_SOLVER_STAGES_PER_CELL,
        verify_first_boundary_audit,
        verify_horizon_only_settings,
        verify_interval_ledger,
    )

    settings = ServiceMPCSetpointSettingsV1(
        task_horizon_h=24.0,
        planning_horizon_h=24.0,
    )
    if settings.to_dict() != dict(m24_settings):
        raise ValueError("M24 serialized settings differ from typed settings")
    run = verify_mpc_setpoint_run_semantics(m24["run_evidence"], settings=settings)
    rows = _mapping("M24 interval ledger", m24["interval_ledger"])
    verify_interval_ledger(
        rows,
        run_evidence=run,
        reward_trace_eur=m24["reward_trace_eur"],
        physical_metrics=_mapping("M24 physical metrics", m24["physical_metrics"]),
        economic_metrics=_mapping("M24 economic metrics", m24["economic_metrics"]),
    )
    events = run.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("M24 run omits event evidence")
    m6_evidence = _mapping("historical M6 evidence", reference_m6["evidence"])
    m6_events = m6_evidence.get("events")
    if not isinstance(m6_events, list) or not m6_events:
        raise ValueError("historical M6 run omits event evidence")
    boundary = verify_first_boundary_audit(
        _mapping("M24 first boundary audit", m24["first_boundary_audit"]),
        event_zero=_mapping("M24 event zero", events[0]),
        reference_event_zero=_mapping("M6 event zero", m6_events[0]),
    )
    horizon = verify_horizon_only_settings(reference_m6_settings, m24_settings)
    counts = _mapping("M24 run counts", run["counts"])
    physical = _mapping("M24 physical metrics", m24["physical_metrics"])
    economic = _mapping("M24 economic metrics", m24["economic_metrics"])
    controller = _mapping("M24 controller descriptor", m24["controller"])
    safety_counts = _reconstruct_m24_safety_counts(
        [_mapping("M24 safety event", event) for event in events]
    )
    safety_counts_reconstructed = all(
        _integer(f"M24 physical {name}", physical[name]) == count
        for name, count in safety_counts.items()
    )
    if not safety_counts_reconstructed:
        raise ValueError("M24 reconstructed safety counts differ from physical metrics")
    if (
        controller.get("id") != run["controller_id"]
        or controller.get("settings") != settings.to_dict()
        or m24.get("status") != run["status"]
        or m24.get("manifest_identity") != reference_identity
        or _integer("M24 arm decisions", m24["completed_decisions"])
        != _integer("M24 run decisions", run["completed_decisions"])
        or _integer("M24 arm decisions", m24["completed_decisions"])
        != _integer("M24 physical decisions", physical["decisions"])
        or _integer("M24 arm physical steps", m24["physical_steps"])
        != _integer("M24 metric physical steps", physical["physical_steps"])
        or _integer("M24 arm executions", m24["execution_count"])
        != _integer("M24 run executions", counts["interval_execution_count"])
        or _integer("M24 arm applications", m24["application_count"])
        != _integer("M24 run applications", counts["physical_minute_count"])
        or m24.get("interval_totals") != rows.get("totals")
        or not math.isclose(
            float(m24["reward_trace_sum_eur"]),
            math.fsum(float(value) for value in m24["reward_trace_eur"]),
            rel_tol=0.0,
            abs_tol=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
        )
        or m24.get("evidence_validation_failure") is not None
    ):
        raise ValueError("M24 arm redundant evidence fields do not reconcile")
    components = _mapping("M24 economic components", economic["components_eur"])
    assumptions = EconomicAssumptionsV1()
    reward_closed = math.isclose(
        float(m24["reward_trace_sum_eur"]),
        float(economic["net_operating_value_eur"]),
        rel_tol=0.0,
        abs_tol=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
    )
    component_closed = math.isclose(
        math.fsum(float(value) for value in components.values()),
        float(economic["net_operating_value_eur"]),
        rel_tol=0.0,
        abs_tol=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
    )
    terminal_closed = math.isclose(
        float(components["methane_endowment_liability_change"]),
        -float(economic["terminal_ch4_equivalent_deficit_kg"])
        * assumptions.methane_endowment_value_eur_per_kg_ch4_equivalent,
        rel_tol=0.0,
        abs_tol=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
    ) and math.isclose(
        float(components["battery_endowment_liability_change"]),
        -float(economic["terminal_battery_deficit_kwh"])
        * assumptions.battery_endowment_value_eur_per_kwh,
        rel_tol=0.0,
        abs_tol=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
    )
    stage_versions = {
        (str(stage["scipy_version"]), str(stage["highs_version"]))
        for event in events
        for stage in event.get("stages", [])
        if isinstance(stage, Mapping)
    }
    scipy_version, highs_version = (
        next(iter(stage_versions)) if len(stage_versions) == 1 else ("mixed", "mixed")
    )
    expected_decisions = FULL_DAY_HORIZON_REPAIR_EXPECTED_DECISIONS_PER_CELL
    expected_minutes = FULL_DAY_HORIZON_REPAIR_EXPECTED_PHYSICAL_MINUTES_PER_CELL
    ledger_rows = rows.get("rows")
    evidence_cardinality = bool(
        isinstance(ledger_rows, list)
        and len(ledger_rows) == expected_decisions
        and len(events) == expected_decisions
        and len(run.get("event_sha256", [])) == expected_decisions
        and len(run.get("execution_sha256", [])) == expected_decisions
        and len(run.get("application_sha256", [])) == expected_minutes
    )
    reset_equal = dict(
        _mapping("M24 manifest identity", m24["manifest_identity"])
    ) == dict(reference_identity)
    m6_identity = _mapping(
        "historical M6 manifest identity", reference_m6["manifest_identity"]
    )
    return {
        "artifact_verified": True,
        "status": m24["status"],
        "terminated": m24["terminated"],
        "truncated": m24["truncated"],
        "failure": m24["failure"],
        "completed_decisions": _integer("M24 decisions", m24["completed_decisions"]),
        "physical_minutes": _integer("M24 physical minutes", m24["physical_steps"]),
        "settlements": _integer("M24 settlements", physical["commitments_settled"]),
        "successful_solver_stages": _integer(
            "M24 successful solver stages", counts["successful_solver_stage_count"]
        ),
        "closed_transitions": counts["closed_decision_count"] == expected_decisions,
        "evidence_cardinality_exact": evidence_cardinality,
        "fallback_count": _integer(
            "M24 fallback count", counts["fallback_action_count"]
        ),
        "reset_identity_equal_reference": reset_equal,
        "reward_accounting_closed": reward_closed,
        "component_accounting_closed": component_closed,
        "interval_accounting_closed": True,
        "terminal_liability_accounting_closed": terminal_closed,
        "safety_counts_reconstructed": safety_counts_reconstructed,
        "shutdown_count": _integer("M24 shutdown", physical["shutdown_steps"]),
        "out_of_range_count": _integer(
            "M24 out of range", physical["out_of_range_decisions"]
        ),
        "forced_stop_count": _integer(
            "M24 forced stops", physical["electrolyser_forced_stops"]
        ),
        "chain_violation_count": _integer(
            "M24 chain violations", counts["chain_violation_count"]
        ),
        "accounting_violation_count": _integer(
            "M24 accounting violations", counts["accounting_violation_count"]
        ),
        "requested_target_overshoot_count": _integer(
            "M24 requested overshoots", counts["requested_target_overshoot_minutes"]
        ),
        "reachable_tracking_violation_count": _integer(
            "M24 reachable violations", counts["reachable_tracking_violation_minutes"]
        ),
        "support_authority_violation_count": _integer(
            "M24 support violations",
            counts["applied_support_authority_violation_minutes"],
        ),
        "unattributed_target_deficit_count": _integer(
            "M24 unattributed deficits", counts["unattributed_target_deficit_minutes"]
        ),
        "settings_sha256": run["settings_sha256"],
        "settings_horizon_only_valid": horizon["passed"] is True,
        "first_boundary_audit_valid": boundary["passed"] is True,
        "environment_identity_equal_m6": reset_equal
        and dict(m6_identity) == dict(reference_identity),
        "scipy_version": scipy_version,
        "highs_version": highs_version,
        "python_version_equal_historical": current_runtime.get("python_version")
        == historical_runtime.get("python"),
        "platform_equal_historical": current_runtime.get("platform")
        == historical_runtime.get("platform"),
    }


def _deterministic_failure(value: object) -> dict[str, Any] | None:
    """Project failures without host-specific messages or paths."""

    if not isinstance(value, Mapping):
        return None
    return {
        "phase": value.get("phase"),
        "exception_type": value.get("exception_type"),
    }


def _deterministic_measurements(value: object) -> object:
    """Remove timing-bearing full-arm citations from deterministic identity."""

    if isinstance(value, Mapping):
        return {
            str(key): _deterministic_measurements(item)
            for key, item in value.items()
            if key != "source_arm_sha256"
        }
    if isinstance(value, list):
        return [_deterministic_measurements(item) for item in value]
    return value


def _deterministic_cell_body(payload: Mapping[str, Any]) -> dict[str, Any]:
    m24 = _mapping("M24 arm", payload["m24"])
    raw_run = m24.get("run_evidence")
    run = _mapping("M24 run evidence", raw_run) if raw_run is not None else {}
    raw_ledger = m24.get("interval_ledger")
    ledger = (
        _mapping("M24 interval ledger", raw_ledger) if raw_ledger is not None else {}
    )
    inputs = payload.get("analysis_inputs")
    deterministic_inputs: object = None
    if isinstance(inputs, Mapping):
        detached = json.loads(canonical_json(dict(inputs)))
        arms = _mapping("deterministic compact arms", detached["arms"])
        for arm_value in arms.values():
            if isinstance(arm_value, dict):
                arm_value.pop("latency", None)
        deterministic_inputs = _deterministic_measurements(detached)
    return {
        "format": payload["format"],
        "contract_sha256": payload["contract_sha256"],
        "cell": payload["cell"],
        "episode": payload["episode"],
        "m24_settings": payload["m24_settings"],
        "manifest_identity": payload["manifest_identity"],
        "historical_citations": payload["historical_citations"],
        "m24": {
            "arm": m24.get("arm"),
            "status": m24.get("status"),
            "manifest_identity": m24.get("manifest_identity"),
            "completed_decisions": m24.get("completed_decisions"),
            "physical_steps": m24.get("physical_steps"),
            "terminated": m24.get("terminated"),
            "truncated": m24.get("truncated"),
            "reward_trace_eur": m24.get("reward_trace_eur"),
            "reward_trace_sum_eur": m24.get("reward_trace_sum_eur"),
            "interval_totals": m24.get("interval_totals"),
            "execution_count": m24.get("execution_count"),
            "application_count": m24.get("application_count"),
            "physical_metrics": m24.get("physical_metrics"),
            "economic_metrics": m24.get("economic_metrics"),
            "failure": _deterministic_failure(m24.get("failure")),
            "run_summary": {
                key: run.get(key)
                for key in (
                    "format",
                    "controller_id",
                    "settings_sha256",
                    "status",
                    "expected_decisions",
                    "completed_decisions",
                    "runner_failure_phase",
                    "pending_transition",
                    "final_observation_sha256",
                    "counts",
                    "energy_kwh",
                    "target_tracking",
                    "target_deficit_attribution_kwh",
                )
                if key in run
            },
            "first_boundary_audit": m24.get("first_boundary_audit"),
            "interval_totals_evidence": ledger.get("totals"),
            "interval_evidence_sha256": ledger.get("sha256"),
            "evidence_graph_sha256": run.get("evidence_graph_sha256"),
        },
        "analysis_inputs": deterministic_inputs,
        "failure": _deterministic_failure(payload["failure"]),
    }


def _build_cell_payload(
    *,
    cell: FullDayHorizonRepairCellV1,
    episode: EpisodeConfigV2,
    settings: ServiceMPCSetpointSettingsV1,
    manifest_identity: Mapping[str, Any] | None,
    historical_citations: Mapping[str, Any],
    m24: Mapping[str, Any],
    reference_h: Mapping[str, Any],
    reference_m6: Mapping[str, Any],
    reference_m6_settings: Mapping[str, Any],
    historical_runtime: Mapping[str, Any],
    started_at: str,
    completed_at: str,
    elapsed_ns: int,
    failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    from .full_day_horizon_repair import (
        FULL_DAY_HORIZON_REPAIR_CELL_FORMAT,
        FULL_DAY_HORIZON_REPAIR_CONTRACT,
        FULL_DAY_HORIZON_REPAIR_CONTRACT_SHA256,
    )

    status = (
        "complete" if m24.get("status") == "complete" and failure is None else "failed"
    )
    current_runtime = _runtime_identity()
    analysis_inputs: dict[str, Any] | None = None
    if status == "complete":
        h_citation = _mapping("H citation", historical_citations["H"])
        m6_citation = _mapping("M6 citation", historical_citations["M6"])
        run = _mapping("M24 run evidence", m24["run_evidence"])
        arms = {
            "H": _compact_arm_measurements(
                reference_h,
                arm="H",
                source_arm_sha256=str(h_citation["deterministic_arm_sha256"]),
            ),
            "M6": _compact_arm_measurements(
                reference_m6,
                arm="M6",
                source_arm_sha256=str(m6_citation["deterministic_arm_sha256"]),
            ),
            "M24": _compact_arm_measurements(
                m24,
                arm="M24",
                source_arm_sha256=str(run["sha256"]),
            ),
        }
        evidence = _m24_evidence_checks(
            m24,
            reference_identity=_mapping(
                "reference manifest identity", manifest_identity
            ),
            reference_m6=reference_m6,
            reference_m6_settings=reference_m6_settings,
            m24_settings=settings.to_dict(),
            historical_runtime=historical_runtime,
            current_runtime=current_runtime,
        )
        analysis_inputs = {"arms": arms, "evidence": evidence}
    body: dict[str, Any] = {
        "format": FULL_DAY_HORIZON_REPAIR_CELL_FORMAT,
        "contract": FULL_DAY_HORIZON_REPAIR_CONTRACT,
        "contract_sha256": FULL_DAY_HORIZON_REPAIR_CONTRACT_SHA256,
        "status": status,
        "cell": cell.to_dict(),
        "episode": episode.to_dict(),
        "m24_settings": settings.to_dict(),
        "manifest_identity": (
            None if manifest_identity is None else dict(manifest_identity)
        ),
        "historical_citations": dict(historical_citations),
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "elapsed_ns": elapsed_ns,
        "runtime": current_runtime,
        "m24": dict(m24),
        "analysis_inputs": analysis_inputs,
        "failure": None if failure is None else dict(failure),
    }
    body["deterministic_evidence_sha256"] = domain_sha256(
        M24_CELL_EVIDENCE_DOMAIN,
        _deterministic_cell_body(body),
    )
    body["sha256"] = domain_sha256(M24_CELL_DOMAIN, body)
    return body


def _verify_failure_record(name: str, value: object) -> Mapping[str, Any]:
    record = _mapping(name, value)
    _require_exact_keys(name, record, frozenset({"phase", "exception_type", "message"}))
    for key in ("phase", "exception_type", "message"):
        if not isinstance(record[key], str) or not record[key]:
            raise ValueError(f"{name} {key} must be a non-empty string")
    return record


def _verify_m24_arm_shape(
    m24: Mapping[str, Any],
    *,
    settings: ServiceMPCSetpointSettingsV1,
    manifest_identity: object,
) -> None:
    """Verify complete and failed arm records before trusting cell status."""

    from .full_day_horizon_codec import verify_mpc_setpoint_run_semantics
    from .full_day_horizon_repair import verify_interval_ledger

    _require_exact_keys("M24 arm", m24, _M24_ARM_FIELDS)
    if m24["arm"] != "M24" or m24["status"] not in {"complete", "failed"}:
        raise ValueError("M24 arm label/status differs")
    if m24["controller"] != ServiceMPCSetpointControllerV1(settings).to_dict():
        raise ValueError("M24 controller descriptor differs from frozen settings")
    if m24["manifest_identity"] != manifest_identity:
        raise ValueError("M24 arm and cell manifest identities differ")
    completed = _integer("M24 arm completed decisions", m24["completed_decisions"])
    physical_steps = _integer("M24 arm physical steps", m24["physical_steps"])
    executions = _integer("M24 arm execution count", m24["execution_count"])
    applications = _integer("M24 arm application count", m24["application_count"])
    _integer("M24 arm elapsed", m24["elapsed_ns"])
    _boolean("M24 arm terminated", m24["terminated"])
    _boolean("M24 arm truncated", m24["truncated"])
    raw_trace = m24["reward_trace_eur"]
    if not isinstance(raw_trace, list):
        raise TypeError("M24 reward trace must be a list")
    trace: list[float] = []
    for value in raw_trace:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("M24 reward trace values must be real numbers")
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("M24 reward trace values must be finite")
        trace.append(parsed)
    if not math.isclose(
        float(m24["reward_trace_sum_eur"]),
        math.fsum(trace),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("M24 reward trace total differs")
    if (
        executions != len(trace)
        or executions > completed
        or applications > physical_steps
    ):
        raise ValueError("M24 retained execution cardinalities are inconsistent")
    physical = _mapping("M24 physical metrics", m24["physical_metrics"])
    economic = _mapping("M24 economic metrics", m24["economic_metrics"])
    if physical:
        if (
            _integer("M24 metric decisions", physical["decisions"]) != completed
            or _integer("M24 metric physical steps", physical["physical_steps"])
            != physical_steps
        ):
            raise ValueError("M24 arm and physical metric counts differ")
    elif completed or physical_steps:
        raise ValueError("M24 moved without retained physical metrics")
    if m24["first_boundary_audit"] is not None:
        _mapping("M24 first-boundary audit", m24["first_boundary_audit"])

    run_raw = m24["run_evidence"]
    run: Mapping[str, Any] | None = None
    if run_raw is not None:
        run = verify_mpc_setpoint_run_semantics(run_raw, settings=settings)
        counts = _mapping("M24 run counts", run["counts"])
        if (
            run["controller_id"] != _mapping("M24 controller", m24["controller"])["id"]
            or run["settings_sha256"] != settings.sha256
            or _integer("M24 run completed decisions", run["completed_decisions"])
            != completed
            or _integer("M24 run executions", counts["interval_execution_count"])
            < executions
            or _integer("M24 run applications", counts["physical_minute_count"])
            < applications
        ):
            raise ValueError("M24 run evidence and arm fields differ")
    elif executions or applications:
        raise ValueError("M24 executions lack retained run evidence")

    ledger_raw = m24["interval_ledger"]
    if ledger_raw is None:
        if m24["interval_totals"] is not None:
            raise ValueError("M24 interval totals lack a ledger")
    else:
        if run is None or not physical or not economic:
            raise ValueError("M24 interval ledger lacks its source evidence")
        ledger = _mapping("M24 interval ledger", ledger_raw)
        verify_interval_ledger(
            ledger,
            run_evidence=run,
            reward_trace_eur=trace,
            physical_metrics=physical,
            economic_metrics=economic,
        )
        if m24["interval_totals"] != ledger["totals"]:
            raise ValueError("M24 arm interval totals differ from its ledger")

    failure = m24["failure"]
    if m24["status"] == "complete":
        if failure is not None or m24["evidence_validation_failure"] is not None:
            raise ValueError("complete M24 arm records a failure")
    else:
        _verify_failure_record("M24 arm failure", failure)
        evidence_failure = m24["evidence_validation_failure"]
        if evidence_failure is not None:
            _verify_failure_record("M24 evidence-validation failure", evidence_failure)


def _verify_cell_payload(
    payload: Mapping[str, Any],
    *,
    expected_cell: FullDayHorizonRepairCellV1,
    expected_historical_citations: Mapping[str, Any],
    reference_h: Mapping[str, Any],
    reference_m6: Mapping[str, Any],
    reference_episode: Mapping[str, Any],
    reference_m6_settings: Mapping[str, Any],
    historical_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    from .full_day_horizon_repair import (
        FULL_DAY_HORIZON_REPAIR_CELL_FORMAT,
        FULL_DAY_HORIZON_REPAIR_CONTRACT,
        FULL_DAY_HORIZON_REPAIR_CONTRACT_SHA256,
        FullDayHorizonRepairCellV1,
    )

    _require_exact_keys("M24 cell artifact", payload, _M24_CELL_FIELDS)
    if payload.get("format") != FULL_DAY_HORIZON_REPAIR_CELL_FORMAT:
        raise ValueError("M24 cell format differs")
    if payload.get("contract") != FULL_DAY_HORIZON_REPAIR_CONTRACT:
        raise ValueError("M24 cell contract differs")
    if payload.get("contract_sha256") != FULL_DAY_HORIZON_REPAIR_CONTRACT_SHA256:
        raise ValueError("M24 cell contract digest differs")
    cell = FullDayHorizonRepairCellV1.from_dict(
        _mapping("M24 cell specification", payload["cell"])
    )
    if cell != expected_cell:
        raise ValueError("M24 cell differs from its registered position")
    episode = EpisodeConfigV2(
        horizon_h=cell.horizon_h,
        delivery_rate_kg_h=cell.delivery_rate_kg_h,
    )
    if canonical_json(payload.get("episode")) != canonical_json(
        episode.to_dict()
    ) or canonical_json(payload["episode"]) != canonical_json(reference_episode):
        raise ValueError("M24 episode differs from the exact reference task")
    settings = ServiceMPCSetpointSettingsV1(
        task_horizon_h=24.0,
        planning_horizon_h=24.0,
    )
    if payload.get("m24_settings") != settings.to_dict():
        raise ValueError("M24 settings differ from the registered intervention")
    if payload.get("historical_citations") != expected_historical_citations:
        raise ValueError("M24 historical citations differ from admission")
    for key in ("started_at_utc", "completed_at_utc"):
        if not isinstance(payload[key], str) or not payload[key]:
            raise ValueError(f"M24 cell {key} is missing")
    _integer("M24 cell elapsed", payload["elapsed_ns"])
    _mapping("M24 cell runtime", payload["runtime"])
    manifest = payload.get("manifest_identity")
    if manifest is not None:
        _mapping("M24 manifest identity", manifest)
    m24 = _mapping("M24 arm", payload["m24"])
    if payload.get("runtime") != _runtime_identity():
        raise ValueError("M24 cell runtime differs from the verifying runtime")
    _verify_m24_arm_shape(m24, settings=settings, manifest_identity=manifest)
    status = payload.get("status")
    if status == "complete":
        if payload.get("failure") is not None or m24.get("status") != "complete":
            raise ValueError("complete M24 cell records a failure")
        expected = _build_cell_payload(
            cell=cell,
            episode=episode,
            settings=settings,
            manifest_identity=_mapping("M24 manifest identity", manifest),
            historical_citations=expected_historical_citations,
            m24=m24,
            reference_h=reference_h,
            reference_m6=reference_m6,
            reference_m6_settings=reference_m6_settings,
            historical_runtime=historical_runtime,
            started_at=str(payload["started_at_utc"]),
            completed_at=str(payload["completed_at_utc"]),
            elapsed_ns=_integer("M24 cell elapsed", payload["elapsed_ns"]),
            failure=None,
        )
        if canonical_json(payload) != canonical_json(expected):
            raise ValueError("M24 cell differs from independent reconstruction")
        inputs = _mapping("M24 analysis inputs", expected["analysis_inputs"])
    elif status == "failed":
        if payload.get("analysis_inputs") is not None or payload.get("failure") is None:
            raise ValueError("failed M24 cell has inconsistent analysis/failure fields")
        _verify_failure_record("M24 cell failure", payload["failure"])
        expected = _build_cell_payload(
            cell=cell,
            episode=episode,
            settings=settings,
            manifest_identity=(
                None
                if manifest is None
                else _mapping("M24 manifest identity", manifest)
            ),
            historical_citations=expected_historical_citations,
            m24=m24,
            reference_h=reference_h,
            reference_m6=reference_m6,
            reference_m6_settings=reference_m6_settings,
            historical_runtime=historical_runtime,
            started_at=str(payload["started_at_utc"]),
            completed_at=str(payload["completed_at_utc"]),
            elapsed_ns=_integer("M24 cell elapsed", payload["elapsed_ns"]),
            failure=_mapping("M24 cell failure", payload["failure"]),
        )
        if canonical_json(payload) != canonical_json(expected):
            raise ValueError("failed M24 cell differs from independent reconstruction")
        inputs = {}
    else:
        raise ValueError("M24 cell status is unsupported")
    recorded = payload.get("deterministic_evidence_sha256")
    if recorded != domain_sha256(
        M24_CELL_EVIDENCE_DOMAIN, _deterministic_cell_body(payload)
    ):
        raise ValueError("M24 deterministic cell digest differs")
    body = dict(payload)
    supplied = body.pop("sha256", None)
    if supplied != domain_sha256(M24_CELL_DOMAIN, body):
        raise ValueError("M24 full cell digest differs")
    return dict(inputs)


def _failed_m24_arm(
    *,
    settings: ServiceMPCSetpointSettingsV1,
    manifest_identity: Mapping[str, Any] | None,
    failure: Mapping[str, Any],
) -> dict[str, Any]:
    controller = ServiceMPCSetpointControllerV1(settings)
    return {
        "arm": "M24",
        "status": "failed",
        "controller": controller.to_dict(),
        "manifest_identity": (
            None if manifest_identity is None else dict(manifest_identity)
        ),
        "completed_decisions": 0,
        "physical_steps": 0,
        "terminated": False,
        "truncated": False,
        "reward_trace_eur": [],
        "reward_trace_sum_eur": 0.0,
        "interval_totals": None,
        "execution_count": 0,
        "application_count": 0,
        "physical_metrics": {},
        "economic_metrics": {},
        "first_boundary_audit": None,
        "interval_ledger": None,
        "run_evidence": None,
        "failure": dict(failure),
        "evidence_validation_failure": None,
        "elapsed_ns": 0,
    }


def _new_environment(
    *,
    episode: EpisodeConfigV2,
    plant: PlantConfig,
    constraints: Constraints,
    weights: RewardWeightsV2,
    assumptions: EconomicAssumptionsV1,
) -> PlantEconomicSetpointEnvV2:
    return PlantEconomicSetpointEnvV2(
        config=plant,
        episode=episode,
        constraints=constraints,
        weights=weights,
        economic_assumptions=assumptions,
    )


def _execute_and_publish_cell(
    admitted_cell: AdmittedReferenceCellV1,
    *,
    index: int,
    destination: Path,
    report_citation: HistoricalReportCitationV1,
    historical_runtime: Mapping[str, Any],
    environment_factory: Callable[..., PlantEconomicSetpointEnvV2],
    arm_executor: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    from .full_day_horizon_repair import (
        FullDayHorizonRepairCellV1,
        evaluate_full_day_horizon_repair_cell,
        m24_settings_for_cell,
        verify_horizon_only_settings,
    )

    citation = admitted_cell.citation
    cell = FullDayHorizonRepairCellV1(
        citation.delivery_rate_kg_h,
        citation.seed,
    )
    if cell.id != citation.cell_id or cell.historical_file_name != citation.file_name:
        raise ValueError("admitted historical cell does not match the M24 registry")
    reference_payload = _mapping(
        "historical reference cell", admitted_cell.reference_payload
    )
    reference_h = _mapping("historical H arm", admitted_cell.heuristic_arm)
    reference_m6 = _mapping("historical M6 arm", admitted_cell.m6_arm)
    reference_episode = _mapping("historical episode", reference_payload["episode"])
    reference_settings = _mapping(
        "historical M6 settings", reference_payload["mpc_settings"]
    )
    episode = EpisodeConfigV2(
        horizon_h=cell.horizon_h,
        delivery_rate_kg_h=cell.delivery_rate_kg_h,
    )
    if canonical_json(episode.to_dict()) != canonical_json(reference_episode):
        raise ValueError("registered M24 task differs from its historical episode")
    settings = ServiceMPCSetpointSettingsV1(
        task_horizon_h=24.0,
        planning_horizon_h=24.0,
    )
    if settings.to_dict() != m24_settings_for_cell(cell):
        raise RuntimeError("registered M24 settings drifted")
    verify_horizon_only_settings(reference_settings, settings.to_dict())
    historical_citations = _reference_citations(report_citation, citation)
    plant = PlantConfig()
    constraints = Constraints()
    weights = RewardWeightsV2()
    assumptions = EconomicAssumptionsV1()
    started_at = _utc_now()
    started_ns = time.perf_counter_ns()
    manifest_identity: dict[str, Any] | None = None
    cell_failure: dict[str, Any] | None = None
    m24: Mapping[str, Any]
    try:
        environment = environment_factory(
            episode=episode,
            plant=plant,
            constraints=constraints,
            weights=weights,
            assumptions=assumptions,
        )
        environment.reset(seed=cell.seed)
        manifest_identity = _prepared_environment_identity(environment)
        reference_identity = dict(
            _mapping(
                "historical manifest identity", reference_payload["manifest_identity"]
            )
        )
        if manifest_identity != reference_identity:
            raise ValueError("M24 reset identity differs from the historical cell")
        if reference_h.get("manifest_identity") != reference_identity:
            raise ValueError("historical H arm identity differs from its paired cell")
        if reference_m6.get("manifest_identity") != reference_identity:
            raise ValueError("historical M6 arm identity differs from its paired cell")
        m24 = arm_executor(
            episode=episode,
            plant=plant,
            constraints=constraints,
            weights=weights,
            assumptions=assumptions,
            settings=settings,
            seed=cell.seed,
            environment=environment,
        )
        if m24.get("status") != "complete":
            cell_failure = dict(
                _mapping(
                    "M24 retained failure",
                    m24.get("failure")
                    or {
                        "phase": "arm_completion",
                        "exception_type": "IncompleteM24Arm",
                        "message": "M24 arm did not complete its declared horizon",
                    },
                )
            )
    except Exception as reason:
        cell_failure = _failure("cell_preflight_or_execution", reason)
        m24 = _failed_m24_arm(
            settings=settings,
            manifest_identity=manifest_identity,
            failure=cell_failure,
        )
    completed_at = _utc_now()
    elapsed_ns = time.perf_counter_ns() - started_ns
    try:
        body = _build_cell_payload(
            cell=cell,
            episode=episode,
            settings=settings,
            manifest_identity=manifest_identity,
            historical_citations=historical_citations,
            m24=m24,
            reference_h=reference_h,
            reference_m6=reference_m6,
            reference_m6_settings=reference_settings,
            historical_runtime=historical_runtime,
            started_at=started_at,
            completed_at=completed_at,
            elapsed_ns=elapsed_ns,
            failure=cell_failure,
        )
    except Exception as reason:
        cell_failure = _failure("cell_evidence_construction", reason)
        body = _build_cell_payload(
            cell=cell,
            episode=episode,
            settings=settings,
            manifest_identity=manifest_identity,
            historical_citations=historical_citations,
            m24=m24,
            reference_h=reference_h,
            reference_m6=reference_m6,
            reference_m6_settings=reference_settings,
            historical_runtime=historical_runtime,
            started_at=started_at,
            completed_at=completed_at,
            elapsed_ns=elapsed_ns,
            failure=cell_failure,
        )
    file_name = _cell_filename(index, cell.id)
    target = destination / file_name
    raw_sha256 = _publish_exclusive(target, body)
    artifact_verified = False
    verification_failure: dict[str, Any] | None = None
    analysis: dict[str, Any] | None = None
    try:
        published = _load_json(target)
        if hashlib.sha256(_read_regular_bytes(target)).hexdigest() != raw_sha256:
            raise ValueError("published M24 cell raw identity differs")
        inputs = _verify_cell_payload(
            published,
            expected_cell=cell,
            expected_historical_citations=historical_citations,
            reference_h=reference_h,
            reference_m6=reference_m6,
            reference_episode=reference_episode,
            reference_m6_settings=reference_settings,
            historical_runtime=historical_runtime,
        )
        artifact_verified = True
        if inputs:
            citations = _analysis_citations(
                historical_citations,
                _mapping("published M24 arm", published["m24"]),
            )
            analysis = evaluate_full_day_horizon_repair_cell(
                cell,
                arms=_mapping("published compact arms", inputs["arms"]),
                evidence=_mapping("published evidence checks", inputs["evidence"]),
                citations=citations,
            )
    except Exception as reason:
        verification_failure = _failure("cell_post_publish_verification", reason)
    m24_run = m24.get("run_evidence")
    run = m24_run if isinstance(m24_run, Mapping) else {}
    raw_ledger = m24.get("interval_ledger")
    ledger = raw_ledger if isinstance(raw_ledger, Mapping) else {}
    return {
        "index": index,
        "cell": cell.to_dict(),
        "file_name": file_name,
        "status": body["status"],
        "raw_file_sha256": raw_sha256,
        "report_sha256": body["sha256"],
        "deterministic_evidence_sha256": body["deterministic_evidence_sha256"],
        "contract_sha256": body["contract_sha256"],
        "interval_evidence_sha256": ledger.get("sha256"),
        "run_evidence_sha256": run.get("sha256"),
        "evidence_graph_sha256": run.get("evidence_graph_sha256"),
        "artifact_verified": artifact_verified,
        "verification_failure": verification_failure,
        "analysis": analysis,
    }


def _custody_payload(
    admission_start: ReferenceAdmissionV1,
    *,
    admission_end: ReferenceAdmissionV1 | None,
    contract_files: Mapping[str, str],
    destination_entries: Sequence[str],
    postflight_failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    from .full_day_horizon_repair import HISTORICAL_REFERENCE_FILE_NAMES

    start_source = admission_start.source_diff.to_dict()
    end_source = None if admission_end is None else admission_end.source_diff.to_dict()
    expected_destination = {
        _cell_filename(index, cell.citation.cell_id)
        for index, cell in enumerate(admission_start.selected_cells, start=1)
    }
    expected_destination.add(M24_REPORT_FILENAME)
    undeclared_destination = sorted(set(destination_entries) - expected_destination)
    return {
        "reference_directory_verified": admission_end is not None,
        "reference_entry_names": ["report.json", *HISTORICAL_REFERENCE_FILE_NAMES],
        "historical_executed_revision": admission_start.report.executed_source_revision,
        "historical_worktree_identity_sha256": (
            admission_start.report.worktree_identity_sha256
        ),
        "historical_contract_sha256": admission_start.report.contract_sha256,
        "historical_raw_report_sha256": admission_start.report.raw_file_sha256,
        "historical_report_domain_sha256": (
            admission_start.report.report_domain_sha256
        ),
        "historical_deterministic_evidence_sha256": (
            admission_start.report.deterministic_evidence_sha256
        ),
        **dict(contract_files),
        "source_clean_start": True,
        "source_clean_end": admission_end is not None,
        "source_identity_stable": end_source == start_source,
        "reference_cell_ids": [
            cell.citation.cell_id for cell in admission_start.selected_cells
        ],
        "historical_cell_citations": [
            {
                "file_name": citation.file_name,
                "raw_file_sha256": citation.raw_file_sha256,
            }
            for citation in admission_start.cells
        ],
        "undeclared_source_paths": (
            [] if postflight_failure is None else ["postflight_admission_failed"]
        ),
        "undeclared_destination_entries": undeclared_destination,
    }


def _deterministic_report_body(payload: Mapping[str, Any]) -> dict[str, Any]:
    from .full_day_horizon_repair import (
        deterministic_full_day_horizon_repair_aggregate,
    )

    aggregate = deterministic_full_day_horizon_repair_aggregate(
        _mapping("M24 aggregate", payload["aggregate"])
    )
    source = _mapping("M24 source gate", payload["source_gate"])
    start = _mapping("M24 source start", source["start"])
    return {
        "format": payload["format"],
        "contract_sha256": payload["contract_sha256"],
        "source": {
            "base_revision": start["base_revision"],
            "target_revision": start["target_revision"],
            "entries": start["entries"],
        },
        "reference": {
            "report": _mapping("historical reference", payload["reference"])["report"],
            "cells": _mapping("historical reference", payload["reference"])["cells"],
        },
        "cells": [
            {
                "index": reference["index"],
                "cell": reference["cell"],
                "status": reference["status"],
                "artifact_verified": reference["artifact_verified"],
                "deterministic_evidence_sha256": reference[
                    "deterministic_evidence_sha256"
                ],
            }
            for reference in payload["cells"]
        ],
        "aggregate_without_latency_gate": aggregate,
    }


def _postflight_admission_destination(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.postflight-admission")


def run_full_day_horizon_repair(
    reference_report: Path | str,
    output: Path | str,
    *,
    repository_root: Path | None = None,
    deep_verifier: Callable[[Path], Mapping[str, Any]] | None = None,
    source_diff_provider: (
        Callable[[Path, str], SourceDiffSnapshotV1 | Mapping[str, object]] | None
    ) = None,
    environment_factory: Callable[..., PlantEconomicSetpointEnvV2] = _new_environment,
    arm_executor: Callable[..., Mapping[str, Any]] = execute_m24_arm,
    cell_executor: Callable[..., dict[str, Any]] = _execute_and_publish_cell,
) -> dict[str, Any]:
    """Execute the exact ten-cell M24 matrix once and publish audited artifacts."""

    from .full_day_horizon_admission import admit_full_day_horizon_probe
    from .full_day_horizon_repair import (
        FULL_DAY_HORIZON_REPAIR_CLAIM_BOUNDARY,
        FULL_DAY_HORIZON_REPAIR_CONTRACT,
        FULL_DAY_HORIZON_REPAIR_CONTRACT_SHA256,
        FULL_DAY_HORIZON_REPAIR_FORMAT,
        REGISTERED_FULL_DAY_HORIZON_REPAIR_CELLS,
        aggregate_full_day_horizon_repair_cells,
    )
    from .mpc_experiment import find_repository_root

    repository = (
        find_repository_root()
        if repository_root is None
        else Path(repository_root).resolve(strict=True)
    )
    contract_files = _contract_file_identities(repository)
    source_provider = source_diff_provider or _source_diff_snapshot
    historical_verifier = deep_verifier or (
        lambda path: _deep_verify_historical_report(path, repository=repository)
    )
    admission_start = admit_full_day_horizon_probe(
        reference_report,
        output,
        repository,
        historical_verifier,
        source_provider,
    )
    _validate_experiment_source_scope(admission_start.source_diff)
    admitted_ids = [cell.citation.cell_id for cell in admission_start.selected_cells]
    if admitted_ids != [cell.id for cell in REGISTERED_FULL_DAY_HORIZON_REPAIR_CELLS]:
        raise ValueError(
            "admitted reference cells differ from the registered M24 matrix"
        )
    destination = admission_start.output_directory
    destination.mkdir(mode=0o700)
    started_at = _utc_now()
    started_ns = time.perf_counter_ns()
    references: list[dict[str, Any]] = []
    historical_runtime = _historical_runtime(admission_start.report_payload)
    for index, admitted_cell in enumerate(admission_start.selected_cells, start=1):
        references.append(
            cell_executor(
                admitted_cell,
                index=index,
                destination=destination,
                report_citation=admission_start.report,
                historical_runtime=historical_runtime,
                environment_factory=environment_factory,
                arm_executor=arm_executor,
            )
        )
    admission_end = None
    postflight_failure: dict[str, Any] | None = None
    postflight_output = _postflight_admission_destination(destination)
    try:
        admission_end = admit_full_day_horizon_probe(
            reference_report,
            postflight_output,
            repository,
            historical_verifier,
            source_provider,
        )
        _validate_experiment_source_scope(admission_end.source_diff)
        if admission_end.source_diff.to_dict() != admission_start.source_diff.to_dict():
            raise RuntimeError("M24 source identity changed during execution")
        if admission_end.report.to_dict() != admission_start.report.to_dict():
            raise RuntimeError("historical root citation changed during M24 execution")
        if [item.to_dict() for item in admission_end.cells] != [
            item.to_dict() for item in admission_start.cells
        ]:
            raise RuntimeError("historical cell citations changed during M24 execution")
    except Exception as reason:
        postflight_failure = _failure("postflight_admission", reason)
        admission_end = None
    try:
        contract_files_end = _contract_file_identities(repository)
        if contract_files_end != contract_files:
            raise RuntimeError("M24 contract files changed during execution")
    except Exception as reason:
        postflight_failure = _failure("postflight_contract_identity", reason)
        admission_end = None
    entries = [entry.name for entry in os.scandir(destination)]
    custody = _custody_payload(
        admission_start,
        admission_end=admission_end,
        contract_files=contract_files,
        destination_entries=entries,
        postflight_failure=postflight_failure,
    )
    aggregate = aggregate_full_day_horizon_repair_cells(
        [
            {"cell": reference["cell"], "analysis": reference["analysis"]}
            for reference in references
        ],
        custody=custody,
        claim_boundary=FULL_DAY_HORIZON_REPAIR_CLAIM_BOUNDARY,
    )
    body: dict[str, Any] = {
        "format": FULL_DAY_HORIZON_REPAIR_FORMAT,
        "contract": FULL_DAY_HORIZON_REPAIR_CONTRACT,
        "contract_sha256": FULL_DAY_HORIZON_REPAIR_CONTRACT_SHA256,
        "status": aggregate["status"],
        "claim_boundary": FULL_DAY_HORIZON_REPAIR_CLAIM_BOUNDARY,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "elapsed_ns": time.perf_counter_ns() - started_ns,
        "runtime": _runtime_identity(),
        "source_gate": {
            "status": "passed" if admission_end is not None else "failed",
            "start": admission_start.source_diff.to_dict(),
            "end": (
                None if admission_end is None else admission_end.source_diff.to_dict()
            ),
            "failure": postflight_failure,
        },
        "reference_report_path": str(
            admission_start.reference_directory / "report.json"
        ),
        "reference": {
            "report": admission_start.report.to_dict(),
            "cells": [cell.to_dict() for cell in admission_start.cells],
            "selected_cell_ids": admitted_ids,
        },
        "cells": references,
        "custody": custody,
        "aggregate": aggregate,
    }
    body["deterministic_evidence_sha256"] = domain_sha256(
        M24_REPORT_EVIDENCE_DOMAIN,
        _deterministic_report_body(body),
    )
    body["sha256"] = domain_sha256(M24_REPORT_DOMAIN, body)
    report_path = destination / M24_REPORT_FILENAME
    raw_sha256 = _publish_exclusive(report_path, body)
    verified = verify_full_day_horizon_report(
        report_path,
        repository_root=repository,
        deep_verifier=historical_verifier,
        source_diff_provider=source_provider,
    )
    if verified["raw_file_sha256"] != raw_sha256:
        raise RuntimeError("published M24 report raw identity differs")
    return verified


def verify_full_day_horizon_report(
    report: Path | str,
    *,
    repository_root: Path | None = None,
    deep_verifier: Callable[[Path], Mapping[str, Any]] | None = None,
    source_diff_provider: (
        Callable[[Path, str], SourceDiffSnapshotV1 | Mapping[str, object]] | None
    ) = None,
) -> dict[str, Any]:
    """Reopen and independently recompute the complete M24 artifact graph."""

    from .full_day_horizon_admission import admit_full_day_horizon_probe
    from .full_day_horizon_repair import (
        FULL_DAY_HORIZON_REPAIR_CLAIM_BOUNDARY,
        FULL_DAY_HORIZON_REPAIR_CONTRACT,
        FULL_DAY_HORIZON_REPAIR_CONTRACT_SHA256,
        FULL_DAY_HORIZON_REPAIR_FORMAT,
        REGISTERED_FULL_DAY_HORIZON_REPAIR_CELLS,
        aggregate_full_day_horizon_repair_cells,
        evaluate_full_day_horizon_repair_cell,
    )
    from .mpc_experiment import find_repository_root

    report_path = Path(report)
    if report_path.name != M24_REPORT_FILENAME:
        raise ValueError("M24 report must be named report.json")
    parent_metadata = report_path.parent.lstat()
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(
        parent_metadata.st_mode
    ):
        raise ValueError("M24 report directory must be a nonsymlink directory")
    report_path = report_path.resolve(strict=True)
    payload = _load_json(report_path)
    _require_exact_keys("M24 report artifact", payload, _M24_REPORT_FIELDS)
    if payload.get("format") != FULL_DAY_HORIZON_REPAIR_FORMAT:
        raise ValueError("M24 report format differs")
    if payload.get("contract") != FULL_DAY_HORIZON_REPAIR_CONTRACT:
        raise ValueError("M24 report contract differs")
    if payload.get("contract_sha256") != FULL_DAY_HORIZON_REPAIR_CONTRACT_SHA256:
        raise ValueError("M24 report contract digest differs")
    if payload.get("claim_boundary") != FULL_DAY_HORIZON_REPAIR_CLAIM_BOUNDARY:
        raise ValueError("M24 report claim boundary differs")
    if payload.get("runtime") != _runtime_identity():
        raise ValueError("M24 report runtime differs from the verifying runtime")
    repository = (
        find_repository_root()
        if repository_root is None
        else Path(repository_root).resolve(strict=True)
    )
    contract_files = _contract_file_identities(repository)
    source_provider = source_diff_provider or _source_diff_snapshot
    historical_verifier = deep_verifier or (
        lambda path: _deep_verify_historical_report(path, repository=repository)
    )
    raw_reference_path = payload.get("reference_report_path")
    if not isinstance(raw_reference_path, str):
        raise TypeError("M24 report omits its historical report path")
    admission = admit_full_day_horizon_probe(
        Path(raw_reference_path),
        _postflight_admission_destination(report_path.parent),
        repository,
        historical_verifier,
        source_provider,
    )
    _validate_experiment_source_scope(admission.source_diff)
    expected_reference = {
        "report": admission.report.to_dict(),
        "cells": [cell.to_dict() for cell in admission.cells],
        "selected_cell_ids": [
            cell.citation.cell_id for cell in admission.selected_cells
        ],
    }
    if payload.get("reference") != expected_reference:
        raise ValueError("M24 report historical citations differ from re-admission")
    source_gate = _mapping("M24 source gate", payload["source_gate"])
    current_source = admission.source_diff.to_dict()
    if (
        source_gate.get("status") != "passed"
        or source_gate.get("failure") is not None
        or source_gate.get("start") != current_source
        or source_gate.get("end") != current_source
    ):
        raise ValueError("M24 report source gate differs from current clean source")

    expected_names = {
        M24_REPORT_FILENAME,
        *(
            _cell_filename(index, cell.id)
            for index, cell in enumerate(
                REGISTERED_FULL_DAY_HORIZON_REPAIR_CELLS, start=1
            )
        ),
    }
    entries: dict[str, os.DirEntry[str]] = {}
    with os.scandir(report_path.parent) as iterator:
        for entry in iterator:
            if entry.name in entries:
                raise ValueError("M24 directory repeats an entry name")
            entries[entry.name] = entry
    if set(entries) != expected_names:
        missing = sorted(expected_names - set(entries))
        extra = sorted(set(entries) - expected_names)
        raise ValueError(
            f"M24 directory entries differ; missing={missing}, extra={extra}"
        )
    for name, entry in entries.items():
        metadata = entry.stat(follow_symlinks=False)
        if entry.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"M24 artifact must be a regular nonsymlink file: {name}")

    references = payload.get("cells")
    if not isinstance(references, list) or len(references) != 10:
        raise ValueError("M24 report must cite exactly ten cells")
    rebuilt_references: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    historical_runtime = _historical_runtime(admission.report_payload)
    for index, (raw_reference, cell, admitted_cell) in enumerate(
        zip(
            references,
            REGISTERED_FULL_DAY_HORIZON_REPAIR_CELLS,
            admission.selected_cells,
            strict=True,
        ),
        start=1,
    ):
        reference = _mapping("M24 cell reference", raw_reference)
        if _integer("M24 cell reference index", reference.get("index")) != index:
            raise ValueError("M24 cell references are reordered")
        file_name = _cell_filename(index, cell.id)
        if reference.get("file_name") != file_name:
            raise ValueError("M24 cell reference filename differs")
        cell_path = (report_path.parent / file_name).resolve(strict=True)
        if cell_path.parent != report_path.parent:
            raise ValueError("M24 cell path escapes the output directory")
        raw_sha256 = hashlib.sha256(_read_regular_bytes(cell_path)).hexdigest()
        if reference.get("raw_file_sha256") != raw_sha256:
            raise ValueError("M24 cell raw identity differs")
        cell_payload = _load_json(cell_path)
        historical_citations = _reference_citations(
            admission.report, admitted_cell.citation
        )
        inputs = _verify_cell_payload(
            cell_payload,
            expected_cell=cell,
            expected_historical_citations=historical_citations,
            reference_h=admitted_cell.heuristic_arm,
            reference_m6=admitted_cell.m6_arm,
            reference_episode=_mapping(
                "historical episode", admitted_cell.reference_payload["episode"]
            ),
            reference_m6_settings=_mapping(
                "historical M6 settings",
                admitted_cell.reference_payload["mpc_settings"],
            ),
            historical_runtime=historical_runtime,
        )
        analysis: dict[str, Any] | None = None
        if inputs:
            citations = _analysis_citations(
                historical_citations,
                _mapping("M24 cell arm", cell_payload["m24"]),
            )
            analysis = evaluate_full_day_horizon_repair_cell(
                cell,
                arms=_mapping("M24 compact arms", inputs["arms"]),
                evidence=_mapping("M24 evidence checks", inputs["evidence"]),
                citations=citations,
            )
        m24 = _mapping("M24 cell arm", cell_payload["m24"])
        raw_run = m24.get("run_evidence")
        run = raw_run if isinstance(raw_run, Mapping) else {}
        raw_ledger = m24.get("interval_ledger")
        ledger = raw_ledger if isinstance(raw_ledger, Mapping) else {}
        rebuilt = {
            "index": index,
            "cell": cell.to_dict(),
            "file_name": file_name,
            "status": cell_payload["status"],
            "raw_file_sha256": raw_sha256,
            "report_sha256": cell_payload["sha256"],
            "deterministic_evidence_sha256": cell_payload[
                "deterministic_evidence_sha256"
            ],
            "contract_sha256": cell_payload["contract_sha256"],
            "interval_evidence_sha256": ledger.get("sha256"),
            "run_evidence_sha256": run.get("sha256"),
            "evidence_graph_sha256": run.get("evidence_graph_sha256"),
            "artifact_verified": True,
            "verification_failure": None,
            "analysis": analysis,
        }
        if dict(reference) != rebuilt:
            raise ValueError("M24 compact cell reference differs from recomputation")
        rebuilt_references.append(rebuilt)
        summaries.append({"cell": cell.to_dict(), "analysis": analysis})

    custody = _custody_payload(
        admission,
        admission_end=admission,
        contract_files=contract_files,
        destination_entries=tuple(entries),
        postflight_failure=None,
    )
    if payload.get("custody") != custody:
        raise ValueError("M24 custody evidence differs from re-admission")
    aggregate = aggregate_full_day_horizon_repair_cells(
        summaries,
        custody=custody,
        claim_boundary=FULL_DAY_HORIZON_REPAIR_CLAIM_BOUNDARY,
    )
    if (
        payload.get("aggregate") != aggregate
        or payload.get("status") != aggregate["status"]
    ):
        raise ValueError("M24 aggregate/status differs from recomputation")
    if payload.get("cells") != rebuilt_references:
        raise ValueError("M24 report cell citations differ from recomputation")
    for key in ("started_at_utc", "completed_at_utc"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"M24 report {key} is missing")
    _integer("M24 report elapsed", payload.get("elapsed_ns"))
    _mapping("M24 report runtime", payload.get("runtime"))
    expected_deterministic = domain_sha256(
        M24_REPORT_EVIDENCE_DOMAIN,
        _deterministic_report_body(payload),
    )
    if payload.get("deterministic_evidence_sha256") != expected_deterministic:
        raise ValueError("M24 deterministic report identity differs")
    body = dict(payload)
    supplied_sha256 = body.pop("sha256", None)
    if supplied_sha256 != domain_sha256(M24_REPORT_DOMAIN, body):
        raise ValueError("M24 full report identity differs")
    return {
        "path": str(report_path),
        "raw_file_sha256": hashlib.sha256(_read_regular_bytes(report_path)).hexdigest(),
        "report_sha256": supplied_sha256,
        "deterministic_evidence_sha256": expected_deterministic,
        "status": aggregate["status"],
        "gates": [
            {"id": gate["id"], "status": gate["status"]} for gate in aggregate["gates"]
        ],
        "cell_count": len(references),
    }


def main(argv: list[str] | None = None) -> int:
    """Expose only the frozen run and detached verification operations."""

    parser = argparse.ArgumentParser(
        description="Run or verify the frozen full-day M24 horizon repair."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser(
        "run", help="execute the registered M24 matrix once"
    )
    run_parser.add_argument("reference_report", type=Path)
    run_parser.add_argument("output", type=Path)
    verify_parser = commands.add_parser("verify", help="deeply verify an M24 report")
    verify_parser.add_argument("report", type=Path)
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "run":
            result = run_full_day_horizon_repair(
                arguments.reference_report,
                arguments.output,
            )
        else:
            result = verify_full_day_horizon_report(arguments.report)
    except Exception as reason:
        print(f"M24 {arguments.command} failed: {reason}", file=sys.stderr)
        return 1
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if result["status"] == "complete_all_gates_passed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M24_CELL_FORMAT",
    "M24_REPORT_FILENAME",
    "M24_REPORT_FORMAT",
    "execute_m24_arm",
    "run_full_day_horizon_repair",
    "verify_full_day_horizon_report",
]

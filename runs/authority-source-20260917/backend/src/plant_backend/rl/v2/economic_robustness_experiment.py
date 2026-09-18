"""Exclusive runner and deep verifier for the frozen economic robustness matrix."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, NoReturn

from plant_sim import PlantConfig

from ...controllers.base import Constraints
from .config import EpisodeConfigV2
from .economic_env import PlantEconomicSetpointEnvV2
from .economic_robustness import (
    ECONOMIC_ROBUSTNESS_CELL_FORMAT,
    ECONOMIC_ROBUSTNESS_CONTRACT,
    ECONOMIC_ROBUSTNESS_CONTRACT_SHA256,
    ECONOMIC_ROBUSTNESS_FORMAT,
    ECONOMIC_ROBUSTNESS_LOCK_SHA256,
    ECONOMIC_ROBUSTNESS_PLANNING_HORIZON_H,
    ECONOMIC_ROBUSTNESS_PROTOCOL_PATH,
    ECONOMIC_ROBUSTNESS_PROTOCOL_SHA256,
    EconomicRobustnessCellV1,
    REGISTERED_ECONOMIC_ROBUSTNESS_CELLS,
    aggregate_robustness_cells,
    cell_deterministic_body,
    episode_for_cell,
    evaluate_robustness_cell,
    mpc_settings_for_cell,
    verify_robustness_cell_payload,
)
from .economic_smoke_experiment import execute_heuristic_arm, execute_mpc_arm
from .economics import EconomicAssumptionsV1
from .evaluation import collect_provenance
from .mpc_experiment import find_repository_root
from .mpc_schema import (
    SERVICE_MPC_SETPOINT_CONTROLLER_ID,
    ServiceMPCSetpointSettingsV1,
    canonical_json,
    domain_sha256,
)
from .reward import RewardWeightsV2
from .setpoint_controllers import LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID


ECONOMIC_ROBUSTNESS_REPORT_FILENAME = "report.json"
ECONOMIC_ROBUSTNESS_REPORT_DOMAIN = "economic-task-robustness-report-v1"
ECONOMIC_ROBUSTNESS_REPORT_EVIDENCE_DOMAIN = (
    "economic-task-robustness-report-evidence-v1"
)


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


def _failure(phase: str, reason: BaseException) -> dict[str, str]:
    return {
        "phase": phase,
        "exception_type": type(reason).__name__,
        "message": " ".join(str(reason).split())[:500],
    }


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _load_json(path: Path) -> dict[str, Any]:
    decoded = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
    )
    return dict(_mapping(str(path), decoded))


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, str]:
    assumptions = dict(
        _mapping("economic assumptions", manifest["economic_assumptions"])
    )
    return {
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "physical_exogenous_episode_sha256": str(manifest["exogenous_episode_sha256"]),
        "economic_episode_sha256": str(manifest["economic_episode_sha256"]),
        "checkpoint_contract_sha256": str(
            _mapping("checkpoint contract", manifest["checkpoint_contract"])[
                "checkpoint_contract_digest"
            ]
        ),
        "economic_assumptions_sha256": hashlib.sha256(
            canonical_json(assumptions).encode("utf-8")
        ).hexdigest(),
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


def _reset_environment(
    environment: PlantEconomicSetpointEnvV2,
    *,
    seed: int,
) -> dict[str, str]:
    environment.reset(seed=seed)
    if environment.decisions_taken or environment.physics_steps_taken:
        raise RuntimeError("reset robustness environment has already moved")
    identity = _manifest_identity(environment.audit_manifest())
    if (
        environment.economic_assumptions.sha256
        != identity["economic_assumptions_sha256"]
    ):
        raise RuntimeError("economic assumptions identity drifted during reset")
    return identity


def _validate_preflight_identities(
    heuristic: Mapping[str, str],
    mpc: Mapping[str, str],
) -> dict[str, str]:
    required = {
        "manifest_sha256",
        "physical_exogenous_episode_sha256",
        "economic_episode_sha256",
        "checkpoint_contract_sha256",
        "economic_assumptions_sha256",
    }
    if set(heuristic) != required or set(mpc) != required:
        raise RuntimeError("paired reset identity fields differ from the frozen schema")
    if dict(heuristic) != dict(mpc):
        raise RuntimeError("paired robustness reset identities differ")
    if any(not value for value in heuristic.values()):
        raise RuntimeError("paired robustness reset identity contains an empty value")
    return dict(heuristic)


def _episode(cell: EconomicRobustnessCellV1) -> EpisodeConfigV2:
    episode = EpisodeConfigV2(
        horizon_h=cell.horizon_h,
        delivery_rate_kg_h=cell.delivery_rate_kg_h,
    )
    if episode.to_dict() != episode_for_cell(cell):
        raise RuntimeError("frozen robustness episode identity drifted")
    return episode


def _mpc_settings(cell: EconomicRobustnessCellV1) -> ServiceMPCSetpointSettingsV1:
    settings = ServiceMPCSetpointSettingsV1(
        task_horizon_h=cell.horizon_h,
        planning_horizon_h=ECONOMIC_ROBUSTNESS_PLANNING_HORIZON_H,
    )
    if settings.to_dict() != mpc_settings_for_cell(cell):
        raise RuntimeError("frozen robustness MPC settings identity drifted")
    return settings


def _source_snapshot(
    *,
    repository: Path,
    episode: EpisodeConfigV2,
    plant: PlantConfig,
    constraints: Constraints,
    weights: RewardWeightsV2,
) -> dict[str, Any]:
    provenance = collect_provenance(
        episode=episode,
        plant=plant,
        constraints=constraints,
        weights=weights,
        root=repository,
    )
    return {
        "source": provenance["source"],
        "dependency_lock": provenance["dependency_lock"],
        "runtime": provenance["runtime"],
    }


def _protocol_valid(repository: Path) -> bool:
    protocol = repository / ECONOMIC_ROBUSTNESS_PROTOCOL_PATH
    return bool(
        protocol.is_file()
        and hashlib.sha256(protocol.read_bytes()).hexdigest()
        == ECONOMIC_ROBUSTNESS_PROTOCOL_SHA256
    )


def _validate_start_source(snapshot: Mapping[str, Any], repository: Path) -> None:
    source = _mapping("source", snapshot["source"])
    lock = _mapping("dependency lock", snapshot["dependency_lock"])
    if (
        source.get("git_dirty") is not False
        or source.get("identity_complete") is not True
    ):
        raise RuntimeError(
            "economic robustness tournament requires a clean, completely identified source"
        )
    if not isinstance(source.get("git_revision"), str) or not source["git_revision"]:
        raise RuntimeError("economic robustness source revision is missing")
    if lock.get("sha256") != ECONOMIC_ROBUSTNESS_LOCK_SHA256:
        raise RuntimeError("economic robustness dependency lock identity drifted")
    if not _protocol_valid(repository):
        raise RuntimeError("economic robustness protocol identity drifted")


def _source_unchanged(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    repository: Path,
) -> bool:
    first = _mapping("start source", before["source"])
    second = _mapping("end source", after["source"])
    first_lock = _mapping("start dependency lock", before["dependency_lock"])
    second_lock = _mapping("end dependency lock", after["dependency_lock"])
    return bool(
        first.get("git_dirty") is False
        and second.get("git_dirty") is False
        and first.get("identity_complete") is True
        and second.get("identity_complete") is True
        and first.get("git_revision") == second.get("git_revision")
        and first.get("worktree_identity_sha256")
        == second.get("worktree_identity_sha256")
        and first_lock.get("sha256") == ECONOMIC_ROBUSTNESS_LOCK_SHA256
        and second_lock.get("sha256") == ECONOMIC_ROBUSTNESS_LOCK_SHA256
        and _protocol_valid(repository)
    )


def validate_output_directory(
    output: Path | str,
    *,
    repository_root: Path,
) -> Path:
    raw = Path(output)
    if not raw.is_absolute():
        raise ValueError("robustness output directory must be absolute")
    target = raw.resolve(strict=False)
    repository = repository_root.resolve()
    try:
        target.relative_to(repository)
    except ValueError:
        pass
    else:
        raise ValueError("robustness output directory must be outside the repository")
    if target.exists():
        raise FileExistsError(f"robustness output directory already exists: {target}")
    if not target.parent.is_dir():
        raise ValueError("robustness output parent directory must already exist")
    return target


def _publish_exclusive(target: Path, payload: Mapping[str, Any]) -> str:
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
    finally:
        temporary.unlink(missing_ok=True)
    return raw_sha256


def _empty_heuristic_evidence() -> dict[str, Any]:
    body: dict[str, Any] = {
        "decision_count": 0,
        "execution_count": 0,
        "application_count": 0,
        "decision_sha256": [],
        "command_sha256": [],
        "execution_sha256": [],
        "decisions": [],
        "executions": [],
        "controller_elapsed_ns": [],
        "dispatcher_elapsed_ns": [],
    }
    body["sha256"] = domain_sha256("economic-smoke-heuristic-evidence-v1", body)
    return body


def _failed_arm(
    arm: str,
    *,
    failure: Mapping[str, Any],
    identity: Mapping[str, str] | None,
    settings: ServiceMPCSetpointSettingsV1,
) -> dict[str, Any]:
    controller: dict[str, Any]
    evidence: dict[str, Any]
    if arm == "H":
        controller = {"id": LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID}
        evidence = _empty_heuristic_evidence()
    elif arm == "M":
        controller = {
            "id": SERVICE_MPC_SETPOINT_CONTROLLER_ID,
            "settings": settings.to_dict(),
        }
        evidence = {"validation_failure": dict(failure)}
    else:
        raise ValueError("unknown robustness arm")
    return {
        "arm": arm,
        "status": "failed",
        "controller": controller,
        "manifest_identity": {} if identity is None else dict(identity),
        "completed_decisions": 0,
        "physical_steps": 0,
        "terminated": False,
        "truncated": False,
        "reward_trace_eur": [],
        "reward_trace_sum_eur": 0.0,
        "interval_totals": {},
        "execution_count": 0,
        "application_count": 0,
        "physical_metrics": {},
        "economic_metrics": {},
        "evidence": evidence,
        "failure": dict(failure),
        "elapsed_ns": 0,
    }


def _cell_filename(index: int, cell: EconomicRobustnessCellV1) -> str:
    return f"{index:02d}_{cell.id}.json"


def _execute_cell(
    cell: EconomicRobustnessCellV1,
    *,
    index: int,
    destination: Path,
    plant: PlantConfig,
    constraints: Constraints,
    weights: RewardWeightsV2,
    assumptions: EconomicAssumptionsV1,
    require_registered: bool,
) -> dict[str, Any]:
    episode = _episode(cell)
    settings = _mpc_settings(cell)
    started_at = _utc_now()
    started_ns = time.perf_counter_ns()
    identity: dict[str, str] | None = None
    cell_failure: dict[str, Any] | None = None

    try:
        heuristic_environment = _new_environment(
            episode=episode,
            plant=plant,
            constraints=constraints,
            weights=weights,
            assumptions=assumptions,
        )
        mpc_environment = _new_environment(
            episode=episode,
            plant=plant,
            constraints=constraints,
            weights=weights,
            assumptions=assumptions,
        )
        heuristic_identity = _reset_environment(
            heuristic_environment,
            seed=cell.seed,
        )
        mpc_identity = _reset_environment(mpc_environment, seed=cell.seed)
        identity = _validate_preflight_identities(heuristic_identity, mpc_identity)
    except Exception as reason:
        cell_failure = _failure("cell_preflight", reason)
        heuristic = _failed_arm(
            "H", failure=cell_failure, identity=identity, settings=settings
        )
        mpc = _failed_arm(
            "M", failure=cell_failure, identity=identity, settings=settings
        )
    else:
        try:
            heuristic = execute_heuristic_arm(
                episode=episode,
                plant=plant,
                constraints=constraints,
                weights=weights,
                assumptions=assumptions,
                seed=cell.seed,
                environment=heuristic_environment,
            )
        except Exception as reason:
            h_failure = _failure("heuristic_arm_invocation", reason)
            cell_failure = {
                "phase": "arm_invocation",
                "arms": {"H": h_failure},
            }
            heuristic = _failed_arm(
                "H", failure=h_failure, identity=identity, settings=settings
            )

        try:
            mpc = execute_mpc_arm(
                episode=episode,
                plant=plant,
                constraints=constraints,
                weights=weights,
                assumptions=assumptions,
                settings=settings,
                seed=cell.seed,
                environment=mpc_environment,
            )
        except Exception as reason:
            m_failure = _failure("mpc_arm_invocation", reason)
            if cell_failure is None:
                cell_failure = {"phase": "arm_invocation", "arms": {}}
            arms = dict(_mapping("arm invocation failures", cell_failure["arms"]))
            arms["M"] = m_failure
            cell_failure["arms"] = arms
            mpc = _failed_arm(
                "M", failure=m_failure, identity=identity, settings=settings
            )

    analysis: dict[str, Any] | None = None
    if cell_failure is None:
        try:
            analysis = evaluate_robustness_cell(
                cell,
                heuristic,
                mpc,
                assumptions=assumptions,
            )
        except Exception as reason:
            cell_failure = _failure("cell_analysis", reason)

    elapsed_ns = time.perf_counter_ns() - started_ns
    completed_at = _utc_now()
    status = (
        "complete"
        if analysis is not None
        and _mapping("cell execution", analysis["execution"])["passed"] is True
        else "failed"
    )
    body: dict[str, Any] = {
        "format": ECONOMIC_ROBUSTNESS_CELL_FORMAT,
        "contract": ECONOMIC_ROBUSTNESS_CONTRACT,
        "contract_sha256": ECONOMIC_ROBUSTNESS_CONTRACT_SHA256,
        "status": status,
        "cell": cell.to_dict(),
        "episode": episode.to_dict(),
        "mpc_settings": settings.to_dict(),
        "manifest_identity": identity,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "elapsed_ns": elapsed_ns,
        "arms": {"H": heuristic, "M": mpc},
        "analysis": analysis,
        "failure": cell_failure,
    }
    body["deterministic_evidence_sha256"] = domain_sha256(
        "economic-task-robustness-cell-evidence-v1",
        cell_deterministic_body(body),
    )
    body["sha256"] = domain_sha256(ECONOMIC_ROBUSTNESS_CELL_FORMAT, body)
    file_name = _cell_filename(index, cell)
    target = destination / file_name
    raw_sha256 = _publish_exclusive(target, body)
    artifact_verified = True
    verification_failure: dict[str, str] | None = None
    try:
        published = _load_json(target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != raw_sha256:
            raise ValueError("published cell raw digest differs")
        verify_robustness_cell_payload(
            published,
            require_registered=require_registered,
        )
    except Exception as reason:
        artifact_verified = False
        verification_failure = _failure("cell_post_publish_verification", reason)
    return {
        "index": index,
        "cell": cell.to_dict(),
        "file_name": file_name,
        "status": status,
        "raw_file_sha256": raw_sha256,
        "report_sha256": body["sha256"],
        "deterministic_evidence_sha256": body["deterministic_evidence_sha256"],
        "artifact_verified": artifact_verified,
        "verification_failure": verification_failure,
        "analysis": analysis,
    }


def _aggregate_cell_summary(reference: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cell": reference["cell"],
        "status": reference["status"],
        "artifact_verified": reference["artifact_verified"],
        "analysis": reference["analysis"],
    }


def _aggregate_without_latency(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    detached = json.loads(canonical_json(dict(aggregate)))
    strata = _mapping("deterministic strata", detached["strata"])
    for value in strata.values():
        mutable = _mapping("deterministic stratum", value)
        if isinstance(mutable, dict):
            mutable.pop("latency_by_seed", None)
            mutable.pop("latency_summaries", None)
    return dict(_mapping("deterministic aggregate", detached))


def _report_deterministic_body(payload: Mapping[str, Any]) -> dict[str, Any]:
    source_gate = _mapping("source gate", payload["source_gate"])
    start = _mapping("start provenance", source_gate["start"])
    start_source = _mapping("start source", start["source"])
    return {
        "format": payload["format"],
        "contract_sha256": payload["contract_sha256"],
        "source": {
            "git_revision": start_source["git_revision"],
            "worktree_identity_sha256": start_source["worktree_identity_sha256"],
            "dependency_lock_sha256": _mapping(
                "start dependency lock", start["dependency_lock"]
            )["sha256"],
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
        "aggregate": _aggregate_without_latency(
            _mapping("report aggregate", payload["aggregate"])
        ),
    }


def run_economic_robustness_tournament(
    output: Path | str,
    *,
    cells: Sequence[EconomicRobustnessCellV1] = (REGISTERED_ECONOMIC_ROBUSTNESS_CELLS),
    require_registered: bool = True,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Run cells sequentially and publish each before the compact matrix report."""

    selected = tuple(cells)
    if not selected:
        raise ValueError("robustness tournament requires at least one cell")
    if len(set(selected)) != len(selected):
        raise ValueError("robustness tournament cells must be unique")
    if require_registered and selected != REGISTERED_ECONOMIC_ROBUSTNESS_CELLS:
        raise ValueError(
            "registered robustness tournament requires the exact frozen matrix"
        )

    repository = (
        find_repository_root() if repository_root is None else repository_root.resolve()
    )
    destination = validate_output_directory(output, repository_root=repository)
    plant = PlantConfig()
    constraints = Constraints()
    weights = RewardWeightsV2()
    assumptions = EconomicAssumptionsV1()
    source_start = _source_snapshot(
        repository=repository,
        episode=_episode(selected[0]),
        plant=plant,
        constraints=constraints,
        weights=weights,
    )
    _validate_start_source(source_start, repository)
    destination.mkdir()

    started_at = _utc_now()
    started_ns = time.perf_counter_ns()
    references = [
        _execute_cell(
            cell,
            index=index,
            destination=destination,
            plant=plant,
            constraints=constraints,
            weights=weights,
            assumptions=assumptions,
            require_registered=require_registered,
        )
        for index, cell in enumerate(selected, start=1)
    ]
    source_end = _source_snapshot(
        repository=repository,
        episode=_episode(selected[0]),
        plant=plant,
        constraints=constraints,
        weights=weights,
    )
    source_valid = _source_unchanged(
        source_start,
        source_end,
        repository=repository,
    )
    aggregate = aggregate_robustness_cells(
        [_aggregate_cell_summary(reference) for reference in references],
        source_valid=source_valid,
    )
    completed_at = _utc_now()
    elapsed_ns = time.perf_counter_ns() - started_ns
    body: dict[str, Any] = {
        "format": ECONOMIC_ROBUSTNESS_FORMAT,
        "contract": ECONOMIC_ROBUSTNESS_CONTRACT,
        "contract_sha256": ECONOMIC_ROBUSTNESS_CONTRACT_SHA256,
        "status": aggregate["status"],
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "elapsed_ns": elapsed_ns,
        "source_gate": {
            "status": "passed" if source_valid else "failed",
            "start": source_start,
            "end": source_end,
        },
        "cells": references,
        "aggregate": aggregate,
    }
    body["deterministic_evidence_sha256"] = domain_sha256(
        ECONOMIC_ROBUSTNESS_REPORT_EVIDENCE_DOMAIN,
        _report_deterministic_body(body),
    )
    body["sha256"] = domain_sha256(ECONOMIC_ROBUSTNESS_REPORT_DOMAIN, body)
    report_path = destination / ECONOMIC_ROBUSTNESS_REPORT_FILENAME
    raw_sha256 = _publish_exclusive(report_path, body)
    verified = verify_robustness_report(
        report_path,
        require_registered=require_registered,
        repository_root=repository,
    )
    if verified["raw_file_sha256"] != raw_sha256:
        raise RuntimeError("published robustness report raw digest differs")
    return verified


def verify_robustness_report(
    report: Path | str,
    *,
    require_registered: bool = True,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Reopen every artifact and recompute all identities, analyses, and gates."""

    report_path = Path(report).resolve()
    if report_path.name != ECONOMIC_ROBUSTNESS_REPORT_FILENAME:
        raise ValueError("robustness report must be named report.json")
    payload = _load_json(report_path)
    if payload.get("format") != ECONOMIC_ROBUSTNESS_FORMAT:
        raise ValueError("robustness report format differs")
    if payload.get("contract") != ECONOMIC_ROBUSTNESS_CONTRACT:
        raise ValueError("robustness report contract differs")
    if payload.get("contract_sha256") != ECONOMIC_ROBUSTNESS_CONTRACT_SHA256:
        raise ValueError("robustness report contract digest differs")

    repository = (
        find_repository_root() if repository_root is None else repository_root.resolve()
    )
    source_gate = _mapping("source gate", payload["source_gate"])
    source_valid = _source_unchanged(
        _mapping("source start", source_gate["start"]),
        _mapping("source end", source_gate["end"]),
        repository=repository,
    )
    expected_source_status = "passed" if source_valid else "failed"
    if source_gate.get("status") != expected_source_status:
        raise ValueError("robustness source gate status differs from its evidence")

    references = payload.get("cells")
    if not isinstance(references, list):
        raise TypeError("robustness report cells must be a list")
    if require_registered and len(references) != len(
        REGISTERED_ECONOMIC_ROBUSTNESS_CELLS
    ):
        raise ValueError("robustness report omits registered cells")
    summaries: list[dict[str, Any]] = []
    seen: set[EconomicRobustnessCellV1] = set()
    expected_files = {ECONOMIC_ROBUSTNESS_REPORT_FILENAME}
    for position, raw_reference in enumerate(references, start=1):
        reference = _mapping("cell reference", raw_reference)
        if _integer("cell reference index", reference["index"]) != position:
            raise ValueError("robustness cell reference order differs")
        cell = EconomicRobustnessCellV1.from_dict(
            _mapping("cell reference specification", reference["cell"])
        )
        if cell in seen:
            raise ValueError("robustness report contains a duplicate cell")
        seen.add(cell)
        if require_registered:
            expected_cell = REGISTERED_ECONOMIC_ROBUSTNESS_CELLS[position - 1]
            if cell != expected_cell:
                raise ValueError(
                    "robustness report cell order differs from the protocol"
                )
        file_name = _cell_filename(position, cell)
        if reference.get("file_name") != file_name:
            raise ValueError("robustness cell filename differs")
        expected_files.add(file_name)
        cell_path = (report_path.parent / file_name).resolve()
        if cell_path.parent != report_path.parent:
            raise ValueError("robustness cell path escapes the output directory")
        raw_bytes = cell_path.read_bytes()
        raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        if reference.get("raw_file_sha256") != raw_sha256:
            raise ValueError("robustness cell raw digest differs")
        cell_payload = _load_json(cell_path)
        verify_robustness_cell_payload(
            cell_payload,
            require_registered=require_registered,
        )
        if cell_payload.get("cell") != cell.to_dict():
            raise ValueError(
                "robustness cell payload identity differs from its reference"
            )
        for key in (
            "status",
            "sha256",
            "deterministic_evidence_sha256",
        ):
            reference_key = "report_sha256" if key == "sha256" else key
            if reference.get(reference_key) != cell_payload.get(key):
                raise ValueError(f"robustness cell {key} differs from its reference")
        if reference.get("artifact_verified") is not True:
            raise ValueError(
                "robustness report does not claim cell artifact verification"
            )
        if reference.get("verification_failure") is not None:
            raise ValueError("verified robustness cell records a verification failure")
        if reference.get("analysis") != cell_payload.get("analysis"):
            raise ValueError("robustness compact cell analysis differs")
        summaries.append(
            {
                "cell": cell.to_dict(),
                "status": cell_payload["status"],
                "artifact_verified": True,
                "analysis": cell_payload["analysis"],
            }
        )

    actual_files = {
        path.name for path in report_path.parent.iterdir() if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("robustness output directory contains undeclared files")
    expected_aggregate = aggregate_robustness_cells(
        summaries,
        source_valid=source_valid,
    )
    if payload.get("aggregate") != expected_aggregate:
        raise ValueError("robustness aggregate differs from recomputation")
    if payload.get("status") != expected_aggregate["status"]:
        raise ValueError("robustness report status differs from its gates")
    expected_deterministic = domain_sha256(
        ECONOMIC_ROBUSTNESS_REPORT_EVIDENCE_DOMAIN,
        _report_deterministic_body(payload),
    )
    if payload.get("deterministic_evidence_sha256") != expected_deterministic:
        raise ValueError("robustness deterministic report digest differs")
    body = dict(payload)
    recorded_sha = body.pop("sha256", None)
    if domain_sha256(ECONOMIC_ROBUSTNESS_REPORT_DOMAIN, body) != recorded_sha:
        raise ValueError("robustness report digest differs")
    raw_report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
    return {
        "path": str(report_path),
        "raw_file_sha256": raw_report_sha256,
        "report_sha256": recorded_sha,
        "deterministic_evidence_sha256": expected_deterministic,
        "status": expected_aggregate["status"],
        "gates": [
            {"id": gate["id"], "status": gate["status"]}
            for gate in expected_aggregate["gates"]
        ],
        "cell_count": len(references),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run or verify the frozen economic robustness tournament."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="execute the frozen matrix once")
    run_parser.add_argument(
        "output",
        type=Path,
        help="new absolute directory outside this Git repository",
    )
    verify_parser = commands.add_parser("verify", help="deeply verify a report")
    verify_parser.add_argument("report", type=Path, help="path to report.json")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "run":
            result = run_economic_robustness_tournament(arguments.output)
        else:
            result = verify_robustness_report(arguments.report)
    except Exception as reason:
        print(
            f"economic robustness {arguments.command} failed: {reason}", file=sys.stderr
        )
        return 1
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if result["status"] == "complete_all_gates_passed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ECONOMIC_ROBUSTNESS_REPORT_FILENAME",
    "run_economic_robustness_tournament",
    "validate_output_directory",
    "verify_robustness_report",
]

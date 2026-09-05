"""Outcome-free runner/verifier tests; no protected environment is reset or acted on."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from plant_backend.rl.v2 import full_day_horizon_admission as admission
from plant_backend.rl.v2 import full_day_horizon_repair as repair
from plant_backend.rl.v2 import full_day_horizon_repair_experiment as experiment
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.mpc_schema import (
    ServiceMPCSetpointSettingsV1,
    canonical_json,
    domain_sha256,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _rehash_cell(payload: dict[str, Any]) -> None:
    payload["deterministic_evidence_sha256"] = domain_sha256(
        experiment.M24_CELL_EVIDENCE_DOMAIN,
        experiment._deterministic_cell_body(payload),
    )
    body = dict(payload)
    body.pop("sha256", None)
    payload["sha256"] = domain_sha256(experiment.M24_CELL_DOMAIN, body)


def _rehash_report(payload: dict[str, Any]) -> None:
    payload["deterministic_evidence_sha256"] = domain_sha256(
        experiment.M24_REPORT_EVIDENCE_DOMAIN,
        experiment._deterministic_report_body(payload),
    )
    body = dict(payload)
    body.pop("sha256", None)
    payload["sha256"] = domain_sha256(experiment.M24_REPORT_DOMAIN, body)


def _write_canonical(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes((canonical_json(payload) + "\n").encode("utf-8"))


def _synthetic_source_admission(
    target_revision: str,
) -> admission.SourceDiffAdmissionV1:
    entries = tuple(
        admission.SourceDiffEntryV1(status="A", path=path)
        for path in admission.ALLOWED_NEW_M24_RUNTIME_PATHS
    )
    return admission.SourceDiffAdmissionV1(
        base_revision=admission.HISTORICAL_EXECUTED_REVISION,
        target_revision=target_revision,
        entries=entries,
        admitted_new_runtime_paths=admission.ALLOWED_NEW_M24_RUNTIME_PATHS,
    )


def _synthetic_admission(
    root: Path,
    output: Path,
    *,
    target_revision: str,
) -> admission.ReferenceAdmissionV1:
    report = admission.HistoricalReportCitationV1(
        file_name="report.json",
        raw_file_sha256=admission.HISTORICAL_RAW_REPORT_SHA256,
        report_domain_sha256=admission.HISTORICAL_REPORT_DOMAIN_SHA256,
        deterministic_evidence_sha256=(
            admission.HISTORICAL_DETERMINISTIC_EVIDENCE_SHA256
        ),
        contract_sha256=admission.HISTORICAL_CONTRACT_SHA256,
        executed_source_revision=admission.HISTORICAL_EXECUTED_REVISION,
        worktree_identity_sha256=admission.HISTORICAL_WORKTREE_IDENTITY_SHA256,
        dependency_lock_sha256=admission.HISTORICAL_LOCK_SHA256,
        status=admission.HISTORICAL_REPORT_STATUS,
    )
    citations: list[admission.HistoricalCellCitationV1] = []
    selected: list[admission.AdmittedReferenceCellV1] = []
    filenames = admission.registered_reference_filenames()
    index = 0
    for horizon_h in (6.0, 24.0):
        for rate in repair.FULL_DAY_HORIZON_REPAIR_RATES_KG_H:
            for seed in repair.FULL_DAY_HORIZON_REPAIR_SEEDS:
                index += 1
                cell_id = f"h{horizon_h:g}-r{rate:g}-s{seed}"
                citation = admission.HistoricalCellCitationV1(
                    index=index,
                    file_name=filenames[index - 1],
                    cell_id=cell_id,
                    horizon_h=horizon_h,
                    delivery_rate_kg_h=rate,
                    seed=seed,
                    raw_file_sha256=_sha(f"historical-raw-{index}"),
                    cell_domain_sha256=_sha(f"historical-cell-{index}"),
                    deterministic_evidence_sha256=_sha(
                        f"historical-deterministic-{index}"
                    ),
                    heuristic=admission.HistoricalArmCitationV1(
                        label="H",
                        controller_id="legacy-heuristic-setpoint-v1",
                        deterministic_arm_sha256=_sha(f"H-det-{index}"),
                        retained_evidence_sha256=_sha(f"H-run-{index}"),
                    ),
                    m6=admission.HistoricalArmCitationV1(
                        label="M6",
                        controller_id="service-mpc-setpoint-v1",
                        deterministic_arm_sha256=_sha(f"M6-det-{index}"),
                        retained_evidence_sha256=_sha(f"M6-run-{index}"),
                        settings_sha256=_sha(f"M6-settings-{index}"),
                        evidence_graph_sha256=_sha(f"M6-graph-{index}"),
                    ),
                )
                citations.append(citation)
                if horizon_h == 24.0:
                    episode = EpisodeConfigV2(
                        horizon_h=24.0,
                        delivery_rate_kg_h=rate,
                    )
                    m6_settings = ServiceMPCSetpointSettingsV1(
                        task_horizon_h=24.0,
                        planning_horizon_h=6.0,
                    )
                    selected.append(
                        admission.AdmittedReferenceCellV1(
                            citation=citation,
                            reference_payload={
                                "episode": episode.to_dict(),
                                "mpc_settings": m6_settings.to_dict(),
                            },
                            heuristic_arm={},
                            m6_arm={},
                        )
                    )
    runtime = experiment._runtime_identity()
    return admission.ReferenceAdmissionV1(
        reference_directory=root / "reference",
        output_directory=output,
        report=report,
        cells=tuple(citations),
        selected_cells=tuple(selected),
        source_diff=_synthetic_source_admission(target_revision),
        _report_payload={
            "source_gate": {
                "start": {
                    "runtime": {
                        "python": runtime["python_version"],
                        "platform": runtime["platform"],
                    }
                }
            }
        },
    )


def _synthetic_contract_identities() -> dict[str, str]:
    return {
        "protocol_sha256": repair.FULL_DAY_HORIZON_REPAIR_PROTOCOL_SHA256,
        "clarification_001_sha256": (
            repair.FULL_DAY_HORIZON_REPAIR_CLARIFICATION_001_SHA256
        ),
        "clarification_002_sha256": (
            repair.FULL_DAY_HORIZON_REPAIR_CLARIFICATION_002_SHA256
        ),
        "dependency_lock_sha256": repair.FULL_DAY_HORIZON_REPAIR_LOCK_SHA256,
    }


def _synthetic_failed_cell_executor(
    admitted_cell: admission.AdmittedReferenceCellV1,
    **kwargs: Any,
) -> dict[str, Any]:
    index = int(kwargs["index"])
    destination = Path(kwargs["destination"])
    citation = admitted_cell.citation
    cell = repair.FullDayHorizonRepairCellV1(
        citation.delivery_rate_kg_h,
        citation.seed,
    )
    episode = EpisodeConfigV2(
        horizon_h=cell.horizon_h,
        delivery_rate_kg_h=cell.delivery_rate_kg_h,
    )
    settings = ServiceMPCSetpointSettingsV1(
        task_horizon_h=24.0,
        planning_horizon_h=24.0,
    )
    failure = {
        "phase": "synthetic_preflight",
        "exception_type": "SyntheticOutcomeFreeFailure",
        "message": "fixture intentionally stops before environment construction",
    }
    arm = experiment._failed_m24_arm(
        settings=settings,
        manifest_identity=None,
        failure=failure,
    )
    body = experiment._build_cell_payload(
        cell=cell,
        episode=episode,
        settings=settings,
        manifest_identity=None,
        historical_citations=experiment._reference_citations(
            kwargs["report_citation"], citation
        ),
        m24=arm,
        reference_h=admitted_cell.heuristic_arm,
        reference_m6=admitted_cell.m6_arm,
        reference_m6_settings=admitted_cell.reference_payload["mpc_settings"],
        historical_runtime=kwargs["historical_runtime"],
        started_at="2026-09-05T00:00:00.000000Z",
        completed_at="2026-09-05T00:00:00.000001Z",
        elapsed_ns=0,
        failure=failure,
    )
    file_name = experiment._cell_filename(index, cell.id)
    raw_sha256 = experiment._publish_exclusive(destination / file_name, body)
    return {
        "index": index,
        "cell": cell.to_dict(),
        "file_name": file_name,
        "status": "failed",
        "raw_file_sha256": raw_sha256,
        "report_sha256": body["sha256"],
        "deterministic_evidence_sha256": body["deterministic_evidence_sha256"],
        "contract_sha256": body["contract_sha256"],
        "interval_evidence_sha256": None,
        "run_evidence_sha256": None,
        "evidence_graph_sha256": None,
        "artifact_verified": True,
        "verification_failure": None,
        "analysis": None,
    }


def _install_synthetic_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    target_revisions: tuple[str, ...] = ("b" * 40,),
) -> list[str]:
    calls: list[str] = []

    def fake_admission(
        reference_report: Path | str,
        output: Path | str,
        *args: Any,
        **kwargs: Any,
    ) -> admission.ReferenceAdmissionV1:
        del reference_report, args, kwargs
        target = target_revisions[min(len(calls), len(target_revisions) - 1)]
        calls.append(target)
        return _synthetic_admission(root, Path(output), target_revision=target)

    monkeypatch.setattr(admission, "admit_full_day_horizon_probe", fake_admission)
    monkeypatch.setattr(
        experiment,
        "_contract_file_identities",
        lambda repository: _synthetic_contract_identities(),
    )
    return calls


def _run_synthetic_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, Any], list[str]]:
    calls = _install_synthetic_boundaries(monkeypatch, tmp_path)
    destination = tmp_path / "m24-output"
    result = experiment.run_full_day_horizon_repair(
        tmp_path / "reference" / "report.json",
        destination,
        repository_root=tmp_path,
        deep_verifier=lambda path: pytest.fail(
            f"mock admission unexpectedly invoked deep verifier for {path}"
        ),
        source_diff_provider=lambda repository, revision: pytest.fail(
            f"mock admission unexpectedly requested source diff for {repository}@{revision}"
        ),
        environment_factory=lambda **kwargs: pytest.fail(
            "synthetic report constructed an environment"
        ),
        cell_executor=_synthetic_failed_cell_executor,
    )
    return destination, result, calls


def test_source_scope_accepts_declared_additions_and_rejects_runtime_drift() -> None:
    allowed = tuple(
        admission.SourceDiffEntryV1(status="A", path=path)
        for path in admission.ALLOWED_NEW_M24_RUNTIME_PATHS
    ) + (
        admission.SourceDiffEntryV1(status="M", path="README.md"),
        admission.SourceDiffEntryV1(
            status="A",
            path="backend/tests/test_rl_v2_full_day_horizon_repair_experiment.py",
        ),
    )
    experiment._validate_experiment_source_scope(SimpleNamespace(entries=allowed))

    forbidden = (
        *allowed,
        admission.SourceDiffEntryV1(
            status="M", path="backend/src/plant_backend/rl/v2/env.py"
        ),
    )
    with pytest.raises(ValueError, match="undeclared paths"):
        experiment._validate_experiment_source_scope(SimpleNamespace(entries=forbidden))


def test_exclusive_publication_is_canonical_and_never_overwrites(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact.json"
    digest = experiment._publish_exclusive(target, {"z": 2, "a": 1})
    expected = (canonical_json({"a": 1, "z": 2}) + "\n").encode("utf-8")
    assert experiment._read_regular_bytes(target) == expected
    assert digest == hashlib.sha256(expected).hexdigest()
    assert experiment._load_json(target) == {"a": 1, "z": 2}

    with pytest.raises(FileExistsError):
        experiment._publish_exclusive(target, {"replacement": True})
    assert experiment._read_regular_bytes(target) == expected


def test_historical_heuristic_measurements_use_retained_allocation_shape() -> None:
    run = {
        "physical_metrics": {"accepted_process_energy_kwh": 1.0},
        "evidence": {
            "executions": [
                {
                    "physics_step_s": 60.0,
                    "applications": [
                        {
                            "dispatch": {
                                "allocation": {
                                    "targets": {
                                        "setpoints_kw": {"electrolyser": 450.0}
                                    },
                                    "request_kw": {"electrolyser_total": 450.0},
                                }
                            },
                            "guard": {
                                "action": {
                                    "electrolyser_share": 0.0,
                                    "battery_support_kw": 450.0,
                                }
                            },
                            "plant": {
                                "applied_kw": {
                                    "solar_available": 0.0,
                                    "electrolyser": 0.0,
                                    "battery_support": 0.0,
                                }
                            },
                        }
                    ],
                }
            ]
        },
    }

    measured = experiment._physical_measurements(run, arm="H")

    assert measured["battery_support_kwh"] == 0.0
    assert measured["plant_refusals"] == 1
    assert measured["target_deficits"] == 1


def test_deterministic_cell_and_report_exclude_all_timing_and_failure_paths() -> None:
    wrapper = {
        "schema": "full-day-horizon-measurement-v1",
        "availability": "available",
        "value": {"x": 1.0},
        "source_arm_sha256": _sha("timing-bearing-full-run"),
    }
    cell: dict[str, Any] = {
        "format": repair.FULL_DAY_HORIZON_REPAIR_CELL_FORMAT,
        "contract_sha256": repair.FULL_DAY_HORIZON_REPAIR_CONTRACT_SHA256,
        "cell": {"id": "h24-r35-s904"},
        "episode": {"horizon_h": 24.0},
        "m24_settings": {"planning_horizon_h": 24.0},
        "manifest_identity": {"manifest_sha256": _sha("manifest")},
        "historical_citations": {"root_report": {"raw_file_sha256": _sha("old")}},
        "m24": {
            "arm": "M24",
            "status": "failed",
            "manifest_identity": {"manifest_sha256": _sha("manifest")},
            "completed_decisions": 0,
            "physical_steps": 0,
            "terminated": False,
            "truncated": False,
            "reward_trace_eur": [],
            "reward_trace_sum_eur": 0.0,
            "interval_totals": {},
            "execution_count": 0,
            "application_count": 0,
            "physical_metrics": {"accepted_process_energy_kwh": 1.0},
            "economic_metrics": {},
            "failure": {
                "phase": "execution",
                "exception_type": "RuntimeError",
                "message": "/private/host-A/result.json failed",
            },
            "elapsed_ns": 10,
            "run_evidence": {
                "sha256": _sha("full-run-A"),
                "evidence_graph_sha256": _sha("graph"),
                "decision_latency": {"p99_ns": 10},
            },
            "interval_ledger": {"sha256": _sha("ledger"), "totals": {}},
            "first_boundary_audit": {},
        },
        "analysis_inputs": {
            "arms": {
                arm: {"latency": {"arm_elapsed_ns": 10}, "settlements": wrapper}
                for arm in ("H", "M6", "M24")
            },
            "evidence": {"artifact_verified": True},
        },
        "failure": {
            "phase": "cell",
            "exception_type": "RuntimeError",
            "message": "/private/host-A/output failed",
        },
    }
    baseline = domain_sha256(
        "test-cell-projection-v1", experiment._deterministic_cell_body(cell)
    )
    timing = deepcopy(cell)
    timing["m24"]["elapsed_ns"] = 99_999
    timing["m24"]["run_evidence"]["sha256"] = _sha("full-run-B")
    timing["m24"]["run_evidence"]["decision_latency"]["p99_ns"] = 99_999
    timing["analysis_inputs"]["arms"]["M24"]["latency"]["arm_elapsed_ns"] = 99_999
    timing["analysis_inputs"]["arms"]["M24"]["settlements"]["source_arm_sha256"] = _sha(
        "changed-full-run"
    )
    timing["m24"]["failure"]["message"] = "/private/host-B/result.json failed"
    timing["failure"]["message"] = "/private/host-B/output failed"
    projection = experiment._deterministic_cell_body(timing)
    assert domain_sha256("test-cell-projection-v1", projection) == baseline
    assert "/private/" not in canonical_json(projection)

    physical = deepcopy(cell)
    physical["m24"]["physical_metrics"]["accepted_process_energy_kwh"] = 2.0
    assert (
        domain_sha256(
            "test-cell-projection-v1", experiment._deterministic_cell_body(physical)
        )
        != baseline
    )

    aggregate: dict[str, Any] = {
        "status": "complete_all_gates_passed",
        "declared_matrix": True,
        "claim_boundary": repair.FULL_DAY_HORIZON_REPAIR_CLAIM_BOUNDARY,
        "custody": {},
        "per_rate": {
            "r35": {
                "registered_measurements": {
                    "M24": {
                        "latency": {"arm_elapsed_ns": {"mean": 10.0}},
                        "physical": {"accepted_process_energy_kwh": {"mean": 1.0}},
                    }
                }
            }
        },
        "gates": [{"id": "F8", "status": "passed", "observation": {"latency": 10}}],
    }
    report: dict[str, Any] = {
        "format": repair.FULL_DAY_HORIZON_REPAIR_FORMAT,
        "contract_sha256": repair.FULL_DAY_HORIZON_REPAIR_CONTRACT_SHA256,
        "source_gate": {
            "start": {"base_revision": "a", "target_revision": "b", "entries": []}
        },
        "reference": {"report": {}, "cells": []},
        "cells": [],
        "aggregate": aggregate,
    }
    report_baseline = domain_sha256(
        "test-report-projection-v1", experiment._deterministic_report_body(report)
    )
    changed = deepcopy(report)
    changed["aggregate"]["per_rate"]["r35"]["registered_measurements"]["M24"][
        "latency"
    ]["arm_elapsed_ns"]["mean"] = 90_000.0
    changed["aggregate"]["gates"][0]["status"] = "failed"
    assert (
        domain_sha256(
            "test-report-projection-v1", experiment._deterministic_report_body(changed)
        )
        == report_baseline
    )


def test_failed_cell_coherent_rehash_cannot_bypass_semantic_verifier() -> None:
    cell = repair.FullDayHorizonRepairCellV1(35.0, 904)
    episode = EpisodeConfigV2(horizon_h=24.0, delivery_rate_kg_h=35.0)
    settings = ServiceMPCSetpointSettingsV1(
        task_horizon_h=24.0,
        planning_horizon_h=24.0,
    )
    failure = {
        "phase": "synthetic_preflight",
        "exception_type": "RuntimeError",
        "message": "outcome-free fixture failure",
    }
    citations = {"fixture": _sha("citation")}
    arm = experiment._failed_m24_arm(
        settings=settings,
        manifest_identity=None,
        failure=failure,
    )
    payload = experiment._build_cell_payload(
        cell=cell,
        episode=episode,
        settings=settings,
        manifest_identity=None,
        historical_citations=citations,
        m24=arm,
        reference_h={},
        reference_m6={},
        reference_m6_settings={},
        historical_runtime={"python": "fixture", "platform": "fixture"},
        started_at="2026-09-05T00:00:00.000000Z",
        completed_at="2026-09-05T00:00:00.000001Z",
        elapsed_ns=1,
        failure=failure,
    )
    assert (
        experiment._verify_cell_payload(
            payload,
            expected_cell=cell,
            expected_historical_citations=citations,
            reference_h={},
            reference_m6={},
            reference_episode=episode.to_dict(),
            reference_m6_settings={},
            historical_runtime={"python": "fixture", "platform": "fixture"},
        )
        == {}
    )

    tampered = deepcopy(payload)
    tampered["m24"]["completed_decisions"] = 1
    _rehash_cell(tampered)
    with pytest.raises(ValueError, match="moved without retained physical metrics"):
        experiment._verify_cell_payload(
            tampered,
            expected_cell=cell,
            expected_historical_citations=citations,
            reference_h={},
            reference_m6={},
            reference_episode=episode.to_dict(),
            reference_m6_settings={},
            historical_runtime={"python": "fixture", "platform": "fixture"},
        )


def test_runner_retains_a_failed_cell_and_continues_without_environment_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "fresh-output"
    cells = repair.REGISTERED_FULL_DAY_HORIZON_REPAIR_CELLS
    selected = tuple(
        SimpleNamespace(citation=SimpleNamespace(cell_id=cell.id)) for cell in cells
    )
    source_dict = {
        "base_revision": "a" * 40,
        "target_revision": "b" * 40,
        "worktree_clean": True,
        "entries": [],
        "admitted_new_runtime_paths": list(admission.ALLOWED_NEW_M24_RUNTIME_PATHS),
    }
    source = SimpleNamespace(to_dict=lambda: source_dict, entries=())
    report_citation = SimpleNamespace(to_dict=lambda: {"id": "historical-report"})
    historical_cells = tuple(
        SimpleNamespace(to_dict=lambda cell=cell: {"cell_id": cell.id})
        for cell in cells
    )
    admitted = SimpleNamespace(
        output_directory=destination,
        reference_directory=tmp_path / "reference",
        report=report_citation,
        cells=historical_cells,
        selected_cells=selected,
        source_diff=source,
        report_payload={
            "source_gate": {
                "start": {
                    "runtime": {
                        "python": experiment._runtime_identity()["python_version"],
                        "platform": experiment._runtime_identity()["platform"],
                    }
                }
            }
        },
    )
    monkeypatch.setattr(
        admission, "admit_full_day_horizon_probe", lambda *args, **kwargs: admitted
    )
    monkeypatch.setattr(
        experiment,
        "_contract_file_identities",
        lambda repository: {"contract": _sha("contract")},
    )
    monkeypatch.setattr(
        experiment, "_validate_experiment_source_scope", lambda value: None
    )
    monkeypatch.setattr(experiment, "_custody_payload", lambda *args, **kwargs: {})

    aggregate = {
        "status": "complete_gate_failed",
        "declared_matrix": True,
        "claim_boundary": repair.FULL_DAY_HORIZON_REPAIR_CLAIM_BOUNDARY,
        "custody": {},
        "per_rate": {},
        "gates": [],
    }
    monkeypatch.setattr(
        repair,
        "aggregate_full_day_horizon_repair_cells",
        lambda *args, **kwargs: aggregate,
    )
    calls: list[str] = []

    def fake_cell_executor(admitted_cell: Any, **kwargs: Any) -> dict[str, Any]:
        index = int(kwargs["index"])
        cell = cells[index - 1]
        calls.append(admitted_cell.citation.cell_id)
        experiment._publish_exclusive(
            destination / experiment._cell_filename(index, cell.id),
            {"cell_id": cell.id, "status": "failed" if index == 1 else "complete"},
        )
        return {
            "index": index,
            "cell": cell.to_dict(),
            "file_name": experiment._cell_filename(index, cell.id),
            "status": "failed" if index == 1 else "complete",
            "raw_file_sha256": _sha(f"raw-{index}"),
            "report_sha256": _sha(f"report-{index}"),
            "deterministic_evidence_sha256": _sha(f"det-{index}"),
            "contract_sha256": repair.FULL_DAY_HORIZON_REPAIR_CONTRACT_SHA256,
            "interval_evidence_sha256": None,
            "run_evidence_sha256": None,
            "evidence_graph_sha256": None,
            "artifact_verified": True,
            "verification_failure": None,
            "analysis": None,
        }

    def fake_report_verifier(path: Path, **kwargs: Any) -> dict[str, Any]:
        return {
            "raw_file_sha256": hashlib.sha256(
                experiment._read_regular_bytes(path)
            ).hexdigest(),
            "status": "complete_gate_failed",
        }

    monkeypatch.setattr(
        experiment, "verify_full_day_horizon_report", fake_report_verifier
    )
    result = experiment.run_full_day_horizon_repair(
        tmp_path / "reference" / "report.json",
        destination,
        repository_root=tmp_path,
        deep_verifier=lambda path: {},
        source_diff_provider=lambda repository, revision: {},
        environment_factory=lambda **kwargs: pytest.fail(
            "runner constructed an environment in an outcome-free test"
        ),
        cell_executor=fake_cell_executor,
    )
    assert calls == [cell.id for cell in cells]
    assert result["status"] == "complete_gate_failed"
    assert (
        experiment._load_json(destination / experiment.M24_REPORT_FILENAME)["cells"][0][
            "status"
        ]
        == "failed"
    )


def test_run_and_real_verifier_accept_ten_outcome_free_failed_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination, result, admission_calls = _run_synthetic_report(tmp_path, monkeypatch)

    assert admission_calls == ["b" * 40, "b" * 40, "b" * 40]
    assert result["status"] == "complete_gate_failed"
    assert result["cell_count"] == 10
    assert result["gates"] == [
        {"id": "F1", "status": "passed"},
        *({"id": f"F{index}", "status": "failed"} for index in range(2, 9)),
    ]
    report = experiment._load_json(destination / experiment.M24_REPORT_FILENAME)
    assert [reference["status"] for reference in report["cells"]] == ["failed"] * 10
    assert report["source_gate"]["status"] == "passed"
    assert report["custody"]["source_identity_stable"] is True


def test_real_verifier_rejects_an_extra_destination_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination, _, _ = _run_synthetic_report(tmp_path, monkeypatch)
    (destination / "undeclared.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="directory entries differ"):
        experiment.verify_full_day_horizon_report(
            destination / experiment.M24_REPORT_FILENAME,
            repository_root=tmp_path,
        )


def test_real_verifier_rejects_coherently_rehashed_reordered_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination, _, _ = _run_synthetic_report(tmp_path, monkeypatch)
    report_path = destination / experiment.M24_REPORT_FILENAME
    payload = experiment._load_json(report_path)
    payload["cells"][0], payload["cells"][1] = (
        payload["cells"][1],
        payload["cells"][0],
    )
    _rehash_report(payload)
    _write_canonical(report_path, payload)

    with pytest.raises(ValueError, match="references are reordered"):
        experiment.verify_full_day_horizon_report(
            report_path,
            repository_root=tmp_path,
        )


def test_real_verifier_rejects_noncanonical_cell_even_when_raw_citation_is_rehashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination, _, _ = _run_synthetic_report(tmp_path, monkeypatch)
    report_path = destination / experiment.M24_REPORT_FILENAME
    report = experiment._load_json(report_path)
    cell_path = destination / report["cells"][0]["file_name"]
    noncanonical = b" " + experiment._read_regular_bytes(cell_path)
    cell_path.write_bytes(noncanonical)
    report["cells"][0]["raw_file_sha256"] = hashlib.sha256(noncanonical).hexdigest()
    _rehash_report(report)
    _write_canonical(report_path, report)

    with pytest.raises(ValueError, match="JSON artifact is not canonical"):
        experiment.verify_full_day_horizon_report(
            report_path,
            repository_root=tmp_path,
        )


def test_real_verifier_rejects_canonical_cell_body_tamper_with_updated_raw_citation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination, _, _ = _run_synthetic_report(tmp_path, monkeypatch)
    report_path = destination / experiment.M24_REPORT_FILENAME
    report = experiment._load_json(report_path)
    cell_path = destination / report["cells"][0]["file_name"]
    cell = experiment._load_json(cell_path)
    cell["elapsed_ns"] += 1
    _write_canonical(cell_path, cell)
    cell_raw = experiment._read_regular_bytes(cell_path)
    report["cells"][0]["raw_file_sha256"] = hashlib.sha256(cell_raw).hexdigest()
    _rehash_report(report)
    _write_canonical(report_path, report)

    with pytest.raises(ValueError, match="differs from independent reconstruction"):
        experiment.verify_full_day_horizon_report(
            report_path,
            repository_root=tmp_path,
        )


def test_run_detects_postflight_source_identity_drift_before_accepting_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_synthetic_boundaries(
        monkeypatch,
        tmp_path,
        target_revisions=("b" * 40, "c" * 40),
    )
    destination = tmp_path / "m24-output"

    with pytest.raises(ValueError, match="source gate differs"):
        experiment.run_full_day_horizon_repair(
            tmp_path / "reference" / "report.json",
            destination,
            repository_root=tmp_path,
            deep_verifier=lambda path: pytest.fail(
                f"mock admission unexpectedly invoked deep verifier for {path}"
            ),
            source_diff_provider=lambda repository, revision: pytest.fail(
                "mock admission unexpectedly requested a source diff"
            ),
            environment_factory=lambda **kwargs: pytest.fail(
                "source-drift test constructed an environment"
            ),
            cell_executor=_synthetic_failed_cell_executor,
        )

    assert calls == ["b" * 40, "c" * 40, "c" * 40]
    report = experiment._load_json(destination / experiment.M24_REPORT_FILENAME)
    assert report["source_gate"]["status"] == "failed"
    assert report["source_gate"]["failure"]["phase"] == "postflight_admission"
    assert report["custody"]["source_clean_end"] is False
    assert report["custody"]["source_identity_stable"] is False

"""Outcome-free contract tests for the frozen service-MPC matrix report."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest
from plant_sim import Action, PlantConfig

from plant_backend.controllers.base import Constraints
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.controllers import IssuedForecastControllerV2
from plant_backend.rl.v2.evaluation import (
    CAUSAL_COMPARATOR_DISCLAIMER,
    CAUSAL_INFORMATION_FAIRNESS,
    CAUSAL_RESULT_SCHEMA_VERSION,
    BaselineRun,
    CausalControllerEvaluationV2,
    ControllerFailureV2,
    LegacyHeuristicControllerV2,
    RequestedActionV2,
)
from plant_backend.rl.v2.metrics import EpisodeMetricsV2
from plant_backend.rl.v2.mpc_controller import ServiceMPCControllerV1
from plant_backend.rl.v2.mpc_execution import (
    SERVICE_MPC_ACTION_DOMAIN,
    SERVICE_MPC_ACTION_SCHEMA,
    SERVICE_MPC_APPLICATION_DOMAIN,
    SERVICE_MPC_DECISION_DOMAIN,
    SERVICE_MPC_FAILURE_DOMAIN,
    SERVICE_MPC_FAILURE_SCHEMA,
    SERVICE_MPC_GUARDED_ACTION_DOMAIN,
    SERVICE_MPC_INTERVAL_POWER_DOMAIN,
    SERVICE_MPC_REQUEST_DOMAIN,
    SERVICE_MPC_RESIDUAL_DOMAIN,
    SERVICE_MPC_RUN_EVIDENCE_DOMAIN,
    SERVICE_MPC_RUN_EVIDENCE_FORMAT,
)
from plant_backend.rl.v2.mpc_experiment import (
    REGISTERED_MPC_CELLS,
    REGISTERED_MPC_CELL_EXECUTION_FORMAT,
    REGISTERED_MPC_CONTRASTS,
    REGISTERED_MPC_DECISIONS,
    REGISTERED_MPC_MATRIX_CONTRACT_SHA256,
)
from plant_backend.rl.v2.mpc_report import (
    FROZEN_PLANNER_RAW_SHA256,
    FROZEN_PLANNER_RELATIVE_PATH,
    FROZEN_PLANNER_REPORT_SHA256,
    _CellAudit,
    _RunAudit,
    admit_registered_service_mpc_matrix,
    build_service_mpc_report,
    load_frozen_planner_cells,
    publish_registered_service_mpc_report,
)
import plant_backend.rl.v2.mpc_report as mpc_report
from plant_backend.rl.v2.mpc_schema import (
    SERVICE_MPC_CONTROLLER_ID,
    SERVICE_MPC_SOLVER,
    domain_sha256,
)
from plant_backend.rl.v2.reward import RewardWeightsV2


_SOURCE_REVISION = "d" * 40
_WORKTREE_SHA = "e" * 64
_EMPTY_SHA = hashlib.sha256(b"").hexdigest()


def _controllers() -> tuple[
    LegacyHeuristicControllerV2,
    IssuedForecastControllerV2,
    ServiceMPCControllerV1,
]:
    return (
        LegacyHeuristicControllerV2(),
        IssuedForecastControllerV2(),
        ServiceMPCControllerV1(),
    )


def _source(repository_root: Path) -> dict[str, Any]:
    return {
        "repository_root": str(repository_root),
        "git_revision": _SOURCE_REVISION,
        "git_dirty": False,
        "tracked_binary_diff_sha256": _EMPTY_SHA,
        "tracked_binary_diff_bytes": 0,
        "tracked_binary_diff_git_args": ["synthetic-pre-outcome-fixture"],
        "untracked_files": [],
        "worktree_identity_format": "git-worktree-content-v1",
        "worktree_identity_sha256": _WORKTREE_SHA,
        "identity_complete": True,
    }


def _runtime() -> dict[str, str]:
    return {
        "python": "fixture",
        "implementation": "fixture",
        "platform": "fixture",
    }


def _provenance(
    repository_root: Path,
    *,
    rate: float,
    seed: int,
    sequence_index: int,
) -> dict[str, Any]:
    source = _source(repository_root)
    lock_sha = hashlib.sha256((repository_root / "uv.lock").read_bytes()).hexdigest()
    plant = PlantConfig()
    episode = EpisodeConfigV2(
        horizon_h=72.0,
        delivery_rate_kg_h=rate,
    ).validate(plant)
    started = datetime(2026, 9, 4, tzinfo=timezone.utc) + timedelta(
        minutes=2 * sequence_index
    )
    completed = started + timedelta(minutes=1)
    return {
        "source": source,
        "dependency_lock": {"path": "uv.lock", "sha256": lock_sha},
        "runtime": _runtime(),
        "environment": {},
        "checkpoint_contract": {},
        "episode_config": episode.to_dict(),
        "plant_config": plant.to_dict(),
        "constraints": Constraints().to_dict(),
        "reward_weights": RewardWeightsV2().to_dict(),
        "execution_source_audit": {
            "status": "valid",
            "require_clean_source": True,
            "identity_complete_at_start": True,
            "identity_complete_at_end": True,
            "worktree_identity_unchanged": True,
            "clean_at_start": True,
            "clean_at_end": True,
            "start_worktree_identity_sha256": _WORKTREE_SHA,
            "end_worktree_identity_sha256": _WORKTREE_SHA,
        },
        "registered_service_mpc_cell": {
            "format": REGISTERED_MPC_CELL_EXECUTION_FORMAT,
            "matrix_contract_sha256": REGISTERED_MPC_MATRIX_CONTRACT_SHA256,
            "sequence_index": sequence_index,
            "cell": {"delivery_rate_kg_h": rate, "seed": seed},
            "started_at_utc": started.isoformat().replace("+00:00", "Z"),
            "evaluation_completed_at_utc": completed.isoformat().replace("+00:00", "Z"),
        },
    }


def _metrics_from_planner(
    planner: dict[str, Any],
    *,
    controller: str,
) -> EpisodeMetricsV2:
    planner_metrics = deepcopy(planner["a0"]["metrics"]["value"])
    planner_metrics["battery_throughput_kwh"] = planner["a0"]["battery_throughput_kwh"]
    constructor_names = {field.name for field in fields(EpisodeMetricsV2)}
    metrics = EpisodeMetricsV2(
        **{name: deepcopy(planner_metrics[name]) for name in constructor_names}
    )
    if controller == LegacyHeuristicControllerV2.id:
        return metrics
    if controller == IssuedForecastControllerV2.id:
        return metrics

    requested = metrics.requested_kg
    metrics.delivered_kg = float(planner["values"]["best_known_feasible_delivered_kg"])
    metrics.unmet_kg = requested - metrics.delivered_kg
    metrics.commitments_met = 12
    metrics.commitments_settled = 12
    metrics.worst_commitment_fill_rate = 1.0
    metrics.terminal_battery_soc = 0.30
    metrics.terminal_battery_energy_kwh = 2_400.0
    metrics.min_battery_soc = 0.25
    metrics.shutdown_steps = 0
    metrics.out_of_range_decisions = 0
    metrics.electrolyser_cold_starts = 1
    metrics.electrolyser_forced_stops = 0
    metrics.reward_components = deepcopy(metrics.reward_components)
    metrics.reward_components["reserve_breach"] = 0.0
    return metrics


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _stage(index: int, stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "success": True,
        "status": 0,
        "message": "synthetic outcome-free report fixture",
        "elapsed_ns": 100,
        "objective_value": 0.0,
        "mip_dual_bound": 0.0,
        "mip_gap": 0.0,
        "mip_node_count": 0,
        "solution_sha256": _sha(f"{index}:{stage}"),
        "solver": SERVICE_MPC_SOLVER,
        "scipy_version": "fixture",
        "highs_version": "fixture",
    }


def _digest(domain: str, body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "sha256": domain_sha256(domain, body)}


def _mpc_run_evidence() -> dict[str, Any]:
    settings_sha = ServiceMPCControllerV1().to_dict()["settings"]["sha256"]
    idle = Action.idle().to_dict()
    events: list[dict[str, Any]] = []
    applications: list[dict[str, Any]] = []
    for index in range(REGISTERED_MPC_DECISIONS):
        plan_sha = _sha(f"plan:{index}")
        stages = [
            _stage(index, "primary_delivery"),
            _stage(index, "secondary_tiebreak"),
        ]
        action = _digest(
            SERVICE_MPC_ACTION_DOMAIN,
            {
                "schema": SERVICE_MPC_ACTION_SCHEMA,
                "plan_sha256": plan_sha,
                "forecast_solar_kw": 0.0,
                "action": idle,
                "simplex_sum": 0.0,
                "realization": "synthetic outcome-free report fixture",
                "actuation_bridge": {
                    "planned_battery_support_kw": 0.0,
                    "battery_support_authority_kw": 0.0,
                    "battery_support_limit_kw": 2_500.0,
                    "electrolyser_target_kw": 0.0,
                    "authority_rule": (
                        "zero when electrolyser target is zero; otherwise "
                        "min(electrolyser target, active support limit)"
                    ),
                },
            },
        )
        residual = _digest(
            SERVICE_MPC_RESIDUAL_DOMAIN,
            {"decision_index": index},
        )
        event_body: dict[str, Any] = {
            "schema": "service-mpc-decision-evidence-v1",
            "controller_id": SERVICE_MPC_CONTROLLER_ID,
            "decision_index": index,
            "observation_sha256": _sha(f"observation:{index}"),
            "public_forecast_sha256": _sha(f"forecast:{index}"),
            "projection_sha256": _sha(f"projection:{index}"),
            "settings_sha256": settings_sha,
            "model": {
                "sha256": _sha(f"model:{index}"),
                "variables": 1,
                "binary_variables": 1,
                "rows": 1,
                "nonzeros": 1,
            },
            "horizon_intervals": min(96, REGISTERED_MPC_DECISIONS - index),
            "deliveries": [],
            "stages": stages,
            "plan": {
                "sha256": plan_sha,
                "primary_delivered_optimum_kg": 0.0,
                "selected_delivered_kg": 0.0,
                "minimum_commitment_fill": 0.0,
                "terminal_ch4_equivalent_kg": 0.0,
                "secondary_utility": 0.0,
            },
            "first_step": {
                "battery_support_kw": 0.0,
                "electrolyser_startup_kw": 0.0,
                "electrolyser_productive_kw": 0.0,
            },
            "action": action,
            "notes": ["synthetic outcome-free report fixture"],
            "decision_elapsed_ns": 300,
            "residual": residual,
        }
        event_identity = deepcopy(event_body)
        event_identity.pop("decision_elapsed_ns")
        identity_stages = event_identity.pop("stages")
        event_identity["stage_solution_sha256"] = [
            stage["solution_sha256"] for stage in identity_stages
        ]
        event = {
            **event_body,
            "sha256": domain_sha256(SERVICE_MPC_DECISION_DOMAIN, event_identity),
        }
        events.append(event)

        request: dict[str, Any] = {}
        guarded: dict[str, Any] = {}
        interval_power: dict[str, Any] = {}
        application_body = {
            "schema": "service-mpc-decision-application-v1",
            "controller_id": SERVICE_MPC_CONTROLLER_ID,
            "decision_index": index,
            "decision_evidence_sha256": event["sha256"],
            "action_sha256": action["sha256"],
            "residual_sha256": residual["sha256"],
            "request": request,
            "request_sha256": domain_sha256(SERVICE_MPC_REQUEST_DOMAIN, request),
            "guarded_action": guarded,
            "guarded_action_sha256": domain_sha256(
                SERVICE_MPC_GUARDED_ACTION_DOMAIN, guarded
            ),
            "interval_power": interval_power,
            "interval_power_sha256": domain_sha256(
                SERVICE_MPC_INTERVAL_POWER_DOMAIN, interval_power
            ),
            "guard_interventions": 0,
            "corrected_physics_steps": 0,
            "out_of_range": False,
            "guard_intervention_integral_l1_kwh": 0.0,
            "dispatch_gap_integral_l1_kwh": 0.0,
            "request_application_integral_l1_kwh": 0.0,
            "guard_changed_request": False,
            "plant_refused_or_rerouted_power": False,
            "adjustments": [],
        }
        applications.append(_digest(SERVICE_MPC_APPLICATION_DOMAIN, application_body))

    body = {
        "format": SERVICE_MPC_RUN_EVIDENCE_FORMAT,
        "controller_id": SERVICE_MPC_CONTROLLER_ID,
        "status": "complete",
        "completed_decisions": REGISTERED_MPC_DECISIONS,
        "runner_failure_phase": None,
        "pending_transition": False,
        "event_count": REGISTERED_MPC_DECISIONS,
        "successful_decision_count": REGISTERED_MPC_DECISIONS,
        "failed_decision_count": 0,
        "application_count": REGISTERED_MPC_DECISIONS,
        "fallback_action_count": 0,
        "all_successful_decisions_reconciled": True,
        "decision_latency": {
            "count": REGISTERED_MPC_DECISIONS,
            "mean_ns": 300.0,
            "minimum_ns": 300,
            "p99_ns": 300,
            "maximum_ns": 300,
            "p99_method": "empirical nearest rank",
        },
        "solver_stage_latency": {
            "count": REGISTERED_MPC_DECISIONS * 2,
            "mean_ns": 100.0,
            "minimum_ns": 100,
            "p99_ns": 100,
            "maximum_ns": 100,
            "p99_method": "empirical nearest rank",
        },
        "guard_intervention_count": 0,
        "corrected_physics_steps": 0,
        "out_of_range_decisions": 0,
        "guard_changed_decisions": 0,
        "plant_refusal_or_reroute_decisions": 0,
        "guard_intervention_integral_l1_kwh": 0.0,
        "dispatch_gap_integral_l1_kwh": 0.0,
        "request_application_integral_l1_kwh": 0.0,
        "events": events,
        "applications": applications,
    }
    return _digest(SERVICE_MPC_RUN_EVIDENCE_DOMAIN, body)


def _failed_mpc_run_evidence() -> dict[str, Any]:
    complete = _mpc_run_evidence()
    failure_body = {
        "schema": SERVICE_MPC_FAILURE_SCHEMA,
        "controller_id": SERVICE_MPC_CONTROLLER_ID,
        "decision_index": 1,
        "phase": "input_validation",
        "exception_type": "RuntimeError",
        "message": "synthetic outcome-free failure fixture",
        "elapsed_ns": 50,
        "observation_sha256": None,
        "public_forecast_sha256": None,
        "projection_sha256": None,
        "model_sha256": None,
        "plan_sha256": None,
        "stages": [],
    }
    failure_identity = deepcopy(failure_body)
    failure_identity.pop("elapsed_ns")
    failure = {
        **failure_body,
        "sha256": domain_sha256(SERVICE_MPC_FAILURE_DOMAIN, failure_identity),
    }
    body = {
        "format": SERVICE_MPC_RUN_EVIDENCE_FORMAT,
        "controller_id": SERVICE_MPC_CONTROLLER_ID,
        "status": "failed",
        "completed_decisions": 1,
        "runner_failure_phase": "controller_compute_action",
        "pending_transition": False,
        "event_count": 2,
        "successful_decision_count": 1,
        "failed_decision_count": 1,
        "application_count": 1,
        "fallback_action_count": 0,
        "all_successful_decisions_reconciled": True,
        "decision_latency": {
            "count": 1,
            "mean_ns": 300.0,
            "minimum_ns": 300,
            "p99_ns": 300,
            "maximum_ns": 300,
            "p99_method": "empirical nearest rank",
        },
        "solver_stage_latency": {
            "count": 2,
            "mean_ns": 100.0,
            "minimum_ns": 100,
            "p99_ns": 100,
            "maximum_ns": 100,
            "p99_method": "empirical nearest rank",
        },
        "guard_intervention_count": 0,
        "corrected_physics_steps": 0,
        "out_of_range_decisions": 0,
        "guard_changed_decisions": 0,
        "plant_refusal_or_reroute_decisions": 0,
        "guard_intervention_integral_l1_kwh": 0.0,
        "dispatch_gap_integral_l1_kwh": 0.0,
        "request_application_integral_l1_kwh": 0.0,
        "events": [deepcopy(complete["events"][0]), failure],
        "applications": [deepcopy(complete["applications"][0])],
    }
    return _digest(SERVICE_MPC_RUN_EVIDENCE_DOMAIN, body)


def _pending_failed_mpc_run_evidence() -> dict[str, Any]:
    failed = _failed_mpc_run_evidence()
    event = deepcopy(failed["events"][0])
    event["residual"] = None
    event_identity = deepcopy(event)
    event_identity.pop("sha256")
    event_identity.pop("decision_elapsed_ns")
    stages = event_identity.pop("stages")
    event_identity["stage_solution_sha256"] = [
        stage["solution_sha256"] for stage in stages
    ]
    event["sha256"] = domain_sha256(SERVICE_MPC_DECISION_DOMAIN, event_identity)

    body = {key: deepcopy(value) for key, value in failed.items() if key != "sha256"}
    body.update(
        {
            "runner_failure_phase": "controller_transition_observation",
            "pending_transition": True,
            "event_count": 1,
            "failed_decision_count": 0,
            "application_count": 0,
            "all_successful_decisions_reconciled": False,
            "events": [event],
            "applications": [],
        }
    )
    return _digest(SERVICE_MPC_RUN_EVIDENCE_DOMAIN, body)


def _cell_document(
    repository_root: Path,
    planner: dict[str, Any],
    *,
    rate: float,
    seed: int,
    sequence_index: int,
    fail_mpc: bool = False,
) -> dict[str, Any]:
    controllers = _controllers()
    traces = tuple(
        RequestedActionV2.from_action(index, Action.idle())
        for index in range(REGISTERED_MPC_DECISIONS)
    )
    manifest = planner["identities"]["manifest_sha256"]
    exogenous = planner["identities"]["exogenous_episode_sha256"]
    runs: list[BaselineRun] = []
    for controller in controllers:
        if fail_mpc and controller.id == SERVICE_MPC_CONTROLLER_ID:
            runs.append(
                BaselineRun(
                    controller=controller.id,
                    seed=seed,
                    manifest_sha256=manifest,
                    exogenous_episode_sha256=exogenous,
                    metrics=None,
                    failure=ControllerFailureV2(
                        phase="controller_compute_action",
                        exception_type="RuntimeError",
                        message="synthetic outcome-free failure fixture",
                        completed_decisions=1,
                        elapsed_hours=0.25,
                        partial_metrics={},
                    ),
                    requested_actions=traces[:1],
                    controller_evidence=_failed_mpc_run_evidence(),
                )
            )
            continue
        evidence = (
            _mpc_run_evidence() if controller.id == SERVICE_MPC_CONTROLLER_ID else None
        )
        runs.append(
            BaselineRun(
                controller=controller.id,
                seed=seed,
                manifest_sha256=manifest,
                exogenous_episode_sha256=exogenous,
                metrics=_metrics_from_planner(planner, controller=controller.id),
                requested_actions=traces,
                controller_evidence=evidence,
            )
        )
    episode = EpisodeConfigV2(
        horizon_h=72.0,
        delivery_rate_kg_h=rate,
    ).validate(PlantConfig())
    result = CausalControllerEvaluationV2(
        seeds=[seed],
        episode_config=episode,
        provenance=_provenance(
            repository_root,
            rate=rate,
            seed=seed,
            sequence_index=sequence_index,
        ),
        runs=runs,
        labels={controller.id: controller.label for controller in controllers},
        information_sets={
            controller.id: controller.information_set for controller in controllers
        },
        controller_specs={
            controller.id: controller.to_dict() for controller in controllers
        },
        study="environment_v2_commitment_alpha_causal_comparators",
        evidence_scope="development_comparator",
        disclaimer=CAUSAL_COMPARATOR_DISCLAIMER,
        title="Environment V2 commitment-alpha causal controller comparison",
        information_fairness=CAUSAL_INFORMATION_FAIRNESS,
        result_schema_version=CAUSAL_RESULT_SCHEMA_VERSION,
        contrasts=REGISTERED_MPC_CONTRASTS,
    )
    return cast(dict[str, Any], result.to_dict())


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _matrix_paths(
    directory: Path,
    *,
    fail_cell: tuple[float, int] | None = None,
) -> tuple[Path, ...]:
    repository = Path.cwd().resolve()
    planner = load_frozen_planner_cells(repository)
    paths: list[Path] = []
    for index, (rate, seed) in enumerate(REGISTERED_MPC_CELLS):
        path = directory / f"rate{rate:g}-seed{seed}.json"
        _write(
            path,
            _cell_document(
                repository,
                planner[(rate, seed)],
                rate=rate,
                seed=seed,
                sequence_index=index,
                fail_mpc=fail_cell == (rate, seed),
            ),
        )
        paths.append(path)
    return tuple(paths)


def _aggregation_provenance(repository: Path) -> dict[str, Any]:
    return {
        "source": _source(repository),
        "dependency_lock": {
            "path": "uv.lock",
            "sha256": hashlib.sha256((repository / "uv.lock").read_bytes()).hexdigest(),
        },
        "runtime": _runtime(),
    }


@pytest.fixture(scope="module")
def admitted_matrix(tmp_path_factory: pytest.TempPathFactory) -> tuple[_CellAudit, ...]:
    directory = tmp_path_factory.mktemp("synthetic-mpc-matrix")
    return cast(
        tuple[_CellAudit, ...],
        admit_registered_service_mpc_matrix(
            _matrix_paths(directory), repository_root=Path.cwd()
        ),
    )


def _report(cells: tuple[_CellAudit, ...]) -> dict[str, Any]:
    repository = Path.cwd().resolve()
    return cast(
        dict[str, Any],
        build_service_mpc_report(
            cells,
            repository_root=repository,
            aggregation_provenance=_aggregation_provenance(repository),
        ),
    )


def _gate(report: dict[str, Any], identifier: str) -> dict[str, Any]:
    return next(
        row for row in report["development_gate"]["gates"] if row["id"] == identifier
    )


def test_frozen_planner_reference_has_exact_file_and_matrix_identity() -> None:
    repository = Path.cwd().resolve()
    path = repository / FROZEN_PLANNER_RELATIVE_PATH
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FROZEN_PLANNER_RAW_SHA256
    assert (
        json.loads(path.read_text("utf-8"))["report_sha256"]
        == FROZEN_PLANNER_REPORT_SHA256
    )
    assert tuple(load_frozen_planner_cells(repository)) == REGISTERED_MPC_CELLS


def test_complete_synthetic_matrix_passes_every_frozen_gate(
    admitted_matrix: tuple[_CellAudit, ...],
) -> None:
    report = _report(admitted_matrix)

    assert report["status"] == "complete_engineering_replication_gate_passed"
    assert report["development_gate"]["all_predeclared_gates_passed"] is True
    assert [row["status"] for row in report["development_gate"]["gates"]] == [
        "pass"
    ] * 7
    assert report["aggregate"] is not None
    assert report["execution_integrity"]["no_surviving_subset_aggregate"] is True
    assert _gate(report, "G1")["observed"]["completed_mpc_decisions"] == 1_728
    assert _gate(report, "G1")["observed"]["successful_solver_stages"] == 3_456
    assert len(report["cells"]) == 6


def test_failed_mpc_arm_suppresses_every_subset_aggregate(
    tmp_path: Path,
) -> None:
    paths = _matrix_paths(tmp_path, fail_cell=(42.0, 1))
    cells = admit_registered_service_mpc_matrix(paths, repository_root=Path.cwd())
    report = _report(cells)

    assert report["status"] == "complete_engineering_replication_gate_failed"
    assert report["aggregate"] is None
    assert _gate(report, "G1")["status"] == "fail"
    assert [_gate(report, f"G{index}")["status"] for index in range(2, 8)] == [
        "not_evaluated"
    ] * 6
    failed_evidence = report["cells"][4]["runs"][-1]["controller_evidence"]
    assert failed_evidence["event_count"] == 2
    assert failed_evidence["application_count"] == 1
    assert "events" not in failed_evidence
    assert "applications" not in failed_evidence
    assert len(failed_evidence["event_ledger_sha256"]) == 64
    assert len(failed_evidence["application_ledger_sha256"]) == 64


def test_pending_transition_failure_is_validated_and_compacted() -> None:
    action = RequestedActionV2.from_action(0, Action.idle()).to_dict()
    encoded = json.dumps(
        [action],
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    trace = mpc_report._validate_trace(
        {
            "format": "controller-requested-action-v1",
            "count": 1,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "actions": [action],
        },
        name="pending failure trace",
    )

    compact = mpc_report._validate_failed_run_evidence(
        _pending_failed_mpc_run_evidence(),
        trace=trace,
        completed_decisions=1,
        runner_failure_phase="controller_transition_observation",
    )

    assert compact is not None
    assert compact["pending_transition"] is True
    assert compact["event_count"] == 1
    assert compact["application_count"] == 0
    assert "events" not in compact
    assert "applications" not in compact


def _mutate_gate(cells: tuple[_CellAudit, ...], gate_id: str) -> None:
    if gate_id == "G2":
        for cell in cells:
            if cell.rate != 42.0:
                continue
            mpc = cell.run(SERVICE_MPC_CONTROLLER_ID).metrics
            a0 = cell.run(LegacyHeuristicControllerV2.id).metrics
            assert mpc is not None and a0 is not None
            mpc["delivered_kg"] = a0["delivered_kg"]
        return
    if gate_id == "G3":
        for cell in cells:
            if cell.rate == 42.0:
                metrics = cell.run(SERVICE_MPC_CONTROLLER_ID).metrics
                assert metrics is not None
                metrics["commitments_met"] = 0
        return
    metrics = cells[0].run(SERVICE_MPC_CONTROLLER_ID).metrics
    assert metrics is not None
    if gate_id == "G4":
        for cell in cells:
            if cell.rate == 35.0:
                rate_metrics = cell.run(SERVICE_MPC_CONTROLLER_ID).metrics
                assert rate_metrics is not None
                rate_metrics["delivery_fill_rate"] = 0.95
    elif gate_id == "G5":
        metrics["electrolyser_forced_stops"] = 1
    elif gate_id == "G6":
        metrics["terminal_battery_soc"] = 0.24
    else:  # pragma: no cover - closed parameter table
        raise AssertionError(f"unsupported gate mutation {gate_id}")


@pytest.mark.parametrize("gate_id", ("G2", "G3", "G4", "G5", "G6"))
def test_each_physical_utility_threshold_fails_closed(
    admitted_matrix: tuple[_CellAudit, ...],
    gate_id: str,
) -> None:
    cells = deepcopy(admitted_matrix)
    _mutate_gate(cells, gate_id)
    report = _report(cells)
    assert _gate(report, gate_id)["status"] == "fail"
    assert report["development_gate"]["all_predeclared_gates_passed"] is False


def test_latency_gate_checks_each_rollout_and_the_pooled_distribution(
    admitted_matrix: tuple[_CellAudit, ...],
) -> None:
    cells = list(deepcopy(admitted_matrix))
    target = cells[0]
    runs = list(target.runs)
    index = next(
        i for i, run in enumerate(runs) if run.controller == SERVICE_MPC_CONTROLLER_ID
    )
    runs[index] = replace(
        runs[index],
        decision_latencies_ns=(2_000_000_001,) * REGISTERED_MPC_DECISIONS,
    )
    cells[0] = replace(target, runs=tuple(runs))

    report = _report(tuple(cells))
    gate = _gate(report, "G7")
    assert gate["status"] == "fail"
    assert gate["observed"]["by_cell"][0]["passed"] is False


def test_admission_rejects_reordering_and_cross_revision_mix(tmp_path: Path) -> None:
    paths = list(_matrix_paths(tmp_path))
    paths[0], paths[1] = paths[1], paths[0]
    with pytest.raises(ValueError, match="execution metadata|study scope|episode"):
        admit_registered_service_mpc_matrix(paths, repository_root=Path.cwd())

    fresh = tmp_path / "fresh"
    fresh.mkdir()
    paths = list(_matrix_paths(fresh))
    changed = json.loads(paths[-1].read_text("utf-8"))
    changed["provenance"]["source"]["git_revision"] = "f" * 40
    _write(paths[-1], changed)
    with pytest.raises(ValueError, match="different source revisions"):
        admit_registered_service_mpc_matrix(paths, repository_root=Path.cwd())


def test_admission_rejects_hidden_fallback_and_a0_drift(tmp_path: Path) -> None:
    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()
    paths = list(_matrix_paths(fallback_dir))
    changed = json.loads(paths[0].read_text("utf-8"))
    evidence = changed["runs"][-1]["controller_evidence"]
    evidence["fallback_action_count"] = 1
    body = dict(evidence)
    body.pop("sha256")
    evidence["sha256"] = domain_sha256(SERVICE_MPC_RUN_EVIDENCE_DOMAIN, body)
    _write(paths[0], changed)
    with pytest.raises(ValueError, match="fallback_action_count"):
        admit_registered_service_mpc_matrix(paths, repository_root=Path.cwd())

    drift_dir = tmp_path / "drift"
    drift_dir.mkdir()
    paths = list(_matrix_paths(drift_dir))
    changed = json.loads(paths[0].read_text("utf-8"))
    metrics = changed["runs"][0]["metrics"]
    metrics["curtailed_kwh"] += 1.0
    metrics["solar_utilisation"] = max(
        0.0, 1.0 - metrics["curtailed_kwh"] / metrics["solar_kwh"]
    )
    _write(paths[0], changed)
    with pytest.raises(ValueError, match="differs from the frozen planner A0"):
        admit_registered_service_mpc_matrix(paths, repository_root=Path.cwd())


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("event_digest", "event 0 digest"),
        ("application_count", "application_count"),
        ("application_binding", "application 0 binding"),
    ),
)
def test_failed_mpc_ledger_is_deeply_validated_before_compaction(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    paths = list(_matrix_paths(tmp_path, fail_cell=(42.0, 1)))
    changed = json.loads(paths[4].read_text("utf-8"))
    evidence = changed["runs"][-1]["controller_evidence"]
    if mutation == "event_digest":
        evidence["events"][0]["sha256"] = "f" * 64
    elif mutation == "application_count":
        evidence["application_count"] = 0
    elif mutation == "application_binding":
        application = evidence["applications"][0]
        application["decision_evidence_sha256"] = "f" * 64
        application_body = dict(application)
        application_body.pop("sha256")
        application["sha256"] = domain_sha256(
            SERVICE_MPC_APPLICATION_DOMAIN,
            application_body,
        )
    else:  # pragma: no cover - closed parameter table
        raise AssertionError(mutation)
    evidence_body = dict(evidence)
    evidence_body.pop("sha256")
    evidence["sha256"] = domain_sha256(
        SERVICE_MPC_RUN_EVIDENCE_DOMAIN,
        evidence_body,
    )
    _write(paths[4], changed)

    with pytest.raises(ValueError, match=match):
        admit_registered_service_mpc_matrix(paths, repository_root=Path.cwd())


def test_admission_rejects_planner_episode_identity_drift(tmp_path: Path) -> None:
    paths = list(_matrix_paths(tmp_path))
    changed = json.loads(paths[0].read_text("utf-8"))
    for run in changed["runs"]:
        run["manifest_sha256"] = "f" * 64
        run["exogenous_episode_sha256"] = "e" * 64
    _write(paths[0], changed)

    with pytest.raises(ValueError, match="do not match the frozen planner cell"):
        admit_registered_service_mpc_matrix(paths, repository_root=Path.cwd())


def test_admission_rejects_overlapping_execution_windows(tmp_path: Path) -> None:
    paths = list(_matrix_paths(tmp_path))
    changed = json.loads(paths[1].read_text("utf-8"))
    metadata = changed["provenance"]["registered_service_mpc_cell"]
    metadata["started_at_utc"] = "2026-09-04T00:00:30Z"
    metadata["evaluation_completed_at_utc"] = "2026-09-04T00:01:30Z"
    _write(paths[1], changed)
    with pytest.raises(ValueError, match="overlapped or ran out of order"):
        admit_registered_service_mpc_matrix(paths, repository_root=Path.cwd())


def test_admission_rejects_noncanonical_cell_bytes(tmp_path: Path) -> None:
    paths = list(_matrix_paths(tmp_path))
    paths[0].write_bytes(paths[0].read_bytes() + b" ")
    with pytest.raises(ValueError, match="canonical file form"):
        admit_registered_service_mpc_matrix(paths, repository_root=Path.cwd())


def test_existing_report_destination_blocks_before_any_cell_is_opened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "existing-report.json"
    target.write_text("preserve me\n", encoding="utf-8")
    called = False

    def forbidden_admission(*_: object, **__: object) -> tuple[_CellAudit, ...]:
        nonlocal called
        called = True
        raise AssertionError("cell admission must not run")

    monkeypatch.setattr(
        mpc_report,
        "admit_registered_service_mpc_matrix",
        forbidden_admission,
    )
    with pytest.raises(ValueError, match="already exists"):
        publish_registered_service_mpc_report(
            cell_paths=(),
            destination=target,
            repository_root=Path.cwd(),
        )
    assert called is False
    assert target.read_text("utf-8") == "preserve me\n"


def test_publisher_atomically_round_trips_complete_report(
    admitted_matrix: tuple[_CellAudit, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path.cwd().resolve()
    provenance = _aggregation_provenance(repository)

    def admitted(*_: object, **__: object) -> tuple[_CellAudit, ...]:
        return admitted_matrix

    def current(_: Path) -> dict[str, Any]:
        return deepcopy(provenance)

    monkeypatch.setattr(
        mpc_report,
        "admit_registered_service_mpc_matrix",
        admitted,
    )
    monkeypatch.setattr(mpc_report, "_current_report_provenance", current)
    target = tmp_path / "service-mpc-report.json"
    published = publish_registered_service_mpc_report(
        cell_paths=tuple(cell.path for cell in admitted_matrix),
        destination=target,
        repository_root=repository,
    )

    raw = target.read_bytes()
    reopened = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == published.file_sha256
    assert reopened == published.report
    body = dict(reopened)
    observed = body.pop("report_sha256")
    assert observed == domain_sha256(mpc_report.SERVICE_MPC_REPORT_DOMAIN, body)

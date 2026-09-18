"""Outcome-blind tests for the frozen economic robustness tournament."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
from typing import Any

import pytest
from plant_sim import PlantConfig

from plant_backend.controllers.base import Constraints
from plant_backend.rl.v2 import economic_robustness as robustness
from plant_backend.rl.v2 import economic_robustness_experiment as experiment
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.economic_smoke_experiment import execute_heuristic_arm
from plant_backend.rl.v2.economics import EconomicAssumptionsV1
from plant_backend.rl.v2.mpc_schema import domain_sha256
from plant_backend.rl.v2.reward import RewardWeightsV2


_REAL_VERIFY_HEURISTIC_ARM_EVIDENCE = robustness.verify_heuristic_arm_evidence


def _identity() -> dict[str, str]:
    return {
        "manifest_sha256": "1" * 64,
        "physical_exogenous_episode_sha256": "2" * 64,
        "economic_episode_sha256": "3" * 64,
        "checkpoint_contract_sha256": "4" * 64,
        "economic_assumptions_sha256": "5" * 64,
    }


def _clean_source() -> dict[str, Any]:
    return {
        "source": {
            "git_dirty": False,
            "identity_complete": True,
            "git_revision": "a" * 40,
            "worktree_identity_sha256": "b" * 64,
        },
        "dependency_lock": {"sha256": robustness.ECONOMIC_ROBUSTNESS_LOCK_SHA256},
        "runtime": {},
    }


def _latency(count: int) -> dict[str, int | float | str]:
    return {
        "count": count,
        "minimum_ns": 1,
        "maximum_ns": 1,
        "mean_ns": 1.0,
        "p99_ns": 1,
        "p99_method": "empirical nearest rank",
    }


def _components(*, arm: str) -> tuple[dict[str, float], float, float]:
    assumptions = EconomicAssumptionsV1()
    methane_deficit = 1.0 if arm == "H" else 0.25
    components = {
        "delivered_service_value": 1_050.0,
        "unmet_delivery_penalty": 0.0,
        "solar_opportunity_cost": -10.0 if arm == "H" else -5.0,
        "battery_throughput_reserve": -2.0 if arm == "H" else -3.0,
        "electrolyser_life_reserve": -1.0,
        "methane_endowment_liability_change": (
            -methane_deficit
            * assumptions.methane_endowment_value_eur_per_kg_ch4_equivalent
        ),
        "battery_endowment_liability_change": 0.0,
    }
    return components, math.fsum(components.values()), methane_deficit


def _heuristic_evidence(decisions: int, applications: int) -> dict[str, Any]:
    applications_per_decision = applications // decisions
    decision_rows = [
        {
            "evidence": {"sha256": f"{index + 1:064x}"},
            "command": {"sha256": f"{index + 101:064x}"},
        }
        for index in range(decisions)
    ]
    execution_rows = [
        {
            "sha256": f"{index + 201:064x}",
            "application_count": applications_per_decision,
            "physics_step_s": 60.0,
            "command": {"hold_duration_s": 900.0},
        }
        for index in range(decisions)
    ]
    body: dict[str, Any] = {
        "decision_count": decisions,
        "execution_count": decisions,
        "application_count": applications,
        "decision_sha256": [row["evidence"]["sha256"] for row in decision_rows],
        "command_sha256": [row["command"]["sha256"] for row in decision_rows],
        "execution_sha256": [row["sha256"] for row in execution_rows],
        "decisions": decision_rows,
        "executions": execution_rows,
        "controller_elapsed_ns": [1] * decisions,
        "dispatcher_elapsed_ns": [
            [1] * applications_per_decision for _ in range(decisions)
        ],
    }
    body["sha256"] = domain_sha256("economic-smoke-heuristic-evidence-v1", body)
    return body


def _arm(
    cell: robustness.EconomicRobustnessCellV1,
    arm: str,
) -> dict[str, Any]:
    decisions = cell.expected_decisions_per_arm
    applications = cell.expected_physical_steps_per_arm
    components, net_value, methane_deficit = _components(arm=arm)
    solar = 100.0 if arm == "H" else 105.0
    throughput = 20.0 if arm == "H" else 22.0
    exposure = 3.0 if arm == "H" else 3.02
    physical = {
        "accepted_process_energy_kwh": 80.0 if arm == "H" else 84.0,
        "curtailed_kwh": 50.0 if arm == "H" else 45.0,
        "electrolyser_cold_starts": 1,
        "electrolyser_hot_starts": 0,
        "electrolyser_aborted_starts": 0,
        "electrolyser_forced_stops": 0,
        "electrolyser_operating_time_s": 10_000.0,
        "terminal_ch4_kg": 100.0,
        "terminal_h2_kg": 10.0,
        "terminal_co2_kg": 20.0,
        "guard_interventions": 0,
        "corrected_physics_steps": 0,
        "shutdown_steps": 0,
        "out_of_range_decisions": 0,
        "commitments_settled": cell.expected_settlements_per_arm,
    }
    economic = {
        "net_operating_value_eur": net_value,
        "components_eur": components,
        "delivered_kg": cell.delivery_rate_kg_h * cell.horizon_h,
        "unmet_kg": 0.0,
        "accepted_solar_kwh": solar,
        "battery_throughput_kwh": throughput,
        "battery_equivalent_full_cycles": throughput / 2_000.0,
        "terminal_battery_energy_kwh": 500.0,
        "electrolyser_equivalent_operating_h": exposure,
        "terminal_ch4_equivalent_deficit_kg": methane_deficit,
        "terminal_battery_deficit_kwh": 0.0,
    }
    counts = {
        "closed_decision_count": decisions,
        "successful_decision_count": decisions,
        "failed_decision_count": 0,
        "fallback_action_count": 0,
        "successful_solver_stage_count": 2 * decisions,
        "attempted_solver_stage_count": 2 * decisions,
        "interval_execution_count": decisions,
        "physical_minute_count": applications,
        "accounting_violation_count": 0,
        "chain_violation_count": 0,
        "requested_target_overshoot_minutes": 0,
        "reachable_tracking_violation_minutes": 0,
        "unattributed_target_deficit_minutes": 0,
        "applied_support_authority_violation_minutes": 0,
    }
    settings = robustness.mpc_settings_for_cell(cell)
    if arm == "H":
        controller = {"id": robustness.LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID}
        evidence = _heuristic_evidence(decisions, applications)
    else:
        controller = {
            "id": robustness.SERVICE_MPC_SETPOINT_CONTROLLER_ID,
            "settings": settings,
        }
        evidence = {
            "format": "synthetic-service-mpc-evidence-v1",
            "status": "complete",
            "controller_id": robustness.SERVICE_MPC_SETPOINT_CONTROLLER_ID,
            "settings_sha256": settings["sha256"],
            "completed_decisions": decisions,
            "expected_decisions": decisions,
            "pending_transition": False,
            "runner_failure_phase": None,
            "final_observation_sha256": "6" * 64,
            "counts": counts,
            "target_tracking": {},
            "target_deficit_attribution_kwh": {},
            "energy_kwh": {},
            "event_sha256": [],
            "execution_sha256": [],
            "application_sha256": [],
            "evidence_graph_sha256": "7" * 64,
            "decision_latency": _latency(decisions),
            "solver_stage_latency": _latency(2 * decisions),
            "dispatcher_latency": _latency(applications),
        }
    return {
        "arm": arm,
        "status": "complete",
        "controller": controller,
        "manifest_identity": _identity(),
        "completed_decisions": decisions,
        "physical_steps": applications,
        "terminated": True,
        "truncated": False,
        "reward_trace_eur": [net_value],
        "reward_trace_sum_eur": net_value,
        "interval_totals": {
            "delivered_kg": economic["delivered_kg"],
            "unmet_kg": economic["unmet_kg"],
            "accepted_solar_kwh": economic["accepted_solar_kwh"],
            "battery_throughput_kwh": economic["battery_throughput_kwh"],
            "electrolyser_equivalent_operating_h": economic[
                "electrolyser_equivalent_operating_h"
            ],
        },
        "execution_count": decisions,
        "application_count": applications,
        "physical_metrics": physical,
        "economic_metrics": economic,
        "evidence": evidence,
        "failure": None,
        "elapsed_ns": applications,
    }


def _cell_payload(
    cell: robustness.EconomicRobustnessCellV1,
) -> dict[str, Any]:
    heuristic = _arm(cell, "H")
    mpc = _arm(cell, "M")
    analysis = robustness.evaluate_robustness_cell(
        cell,
        heuristic,
        mpc,
        assumptions=EconomicAssumptionsV1(),
    )
    body: dict[str, Any] = {
        "format": robustness.ECONOMIC_ROBUSTNESS_CELL_FORMAT,
        "contract": robustness.ECONOMIC_ROBUSTNESS_CONTRACT,
        "contract_sha256": robustness.ECONOMIC_ROBUSTNESS_CONTRACT_SHA256,
        "status": "complete",
        "cell": cell.to_dict(),
        "episode": robustness.episode_for_cell(cell),
        "mpc_settings": robustness.mpc_settings_for_cell(cell),
        "manifest_identity": _identity(),
        "started_at_utc": "2026-09-04T00:00:00.000000Z",
        "completed_at_utc": "2026-09-04T00:00:01.000000Z",
        "elapsed_ns": 1,
        "arms": {"H": heuristic, "M": mpc},
        "analysis": analysis,
        "failure": None,
    }
    body["deterministic_evidence_sha256"] = domain_sha256(
        "economic-task-robustness-cell-evidence-v1",
        robustness.cell_deterministic_body(body),
    )
    body["sha256"] = domain_sha256(
        robustness.ECONOMIC_ROBUSTNESS_CELL_FORMAT,
        body,
    )
    return body


@pytest.fixture(autouse=True)
def _admit_synthetic_mpc_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        robustness,
        "verify_mpc_setpoint_run_digests",
        lambda _: None,
    )
    monkeypatch.setattr(
        robustness,
        "verify_heuristic_arm_evidence",
        lambda _: None,
    )


def test_protocol_hash_matrix_order_and_arithmetic_are_frozen() -> None:
    repository = Path(__file__).resolve().parents[2]
    protocol = repository / robustness.ECONOMIC_ROBUSTNESS_PROTOCOL_PATH

    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == (
        robustness.ECONOMIC_ROBUSTNESS_PROTOCOL_SHA256
    )
    assert hashlib.sha256((repository / "uv.lock").read_bytes()).hexdigest() == (
        robustness.ECONOMIC_ROBUSTNESS_LOCK_SHA256
    )
    cells = robustness.REGISTERED_ECONOMIC_ROBUSTNESS_CELLS
    assert len(cells) == 20
    assert cells[0].id == "h6-r35-s20"
    assert cells[-1].id == "h24-r42-s24"
    assert sum(cell.expected_decisions_per_arm * 2 for cell in cells) == 2_400
    assert sum(cell.expected_physical_steps_per_arm * 2 for cell in cells) == 36_000
    assert [cell.seed for cell in cells[:5]] == [20, 21, 22, 23, 24]
    assert robustness.mpc_settings_for_cell(cells[0])["sha256"] == (
        robustness.ECONOMIC_ROBUSTNESS_MPC_SETTINGS_SHA256[6.0]
    )
    assert robustness.mpc_settings_for_cell(cells[-1])["sha256"] == (
        robustness.ECONOMIC_ROBUSTNESS_MPC_SETTINGS_SHA256[24.0]
    )


def test_real_tiny_heuristic_evidence_closes_the_deep_generic_command_graph() -> None:
    episode = EpisodeConfigV2(
        horizon_h=0.25,
        delivery_period_h=0.25,
        delivery_rate_kg_h=1.0,
        battery_soc=(0.5, 0.5),
        h2_fill=(0.0, 0.0),
        co2_fill=(0.0, 0.0),
        ch4_coverage_h=(0.0, 0.0),
    )
    run = execute_heuristic_arm(
        episode=episode,
        plant=PlantConfig(),
        constraints=Constraints(),
        weights=RewardWeightsV2(),
        assumptions=EconomicAssumptionsV1(),
        seed=903,
    )

    _REAL_VERIFY_HEURISTIC_ARM_EVIDENCE(run)
    assert run["status"] == "complete"
    assert len(run["evidence"]["controller_elapsed_ns"]) == 1
    assert len(run["evidence"]["dispatcher_elapsed_ns"][0]) == 15


def test_cell_schema_round_trip_is_strict() -> None:
    cell = robustness.EconomicRobustnessCellV1(6, 35, 901)

    assert robustness.EconomicRobustnessCellV1.from_dict(cell.to_dict()) == cell
    assert isinstance(cell.horizon_h, float)
    assert isinstance(cell.delivery_rate_kg_h, float)
    assert robustness.episode_for_cell(cell) == robustness.episode_for_cell(
        robustness.EconomicRobustnessCellV1.from_dict(cell.to_dict())
    )
    malformed = cell.to_dict()
    malformed["unregistered_note"] = True
    with pytest.raises(ValueError, match="canonical"):
        robustness.EconomicRobustnessCellV1.from_dict(malformed)


def test_sensitivities_are_additive_and_report_every_frozen_variant() -> None:
    cell = robustness.EconomicRobustnessCellV1(6.0, 35.0, 901)
    result = robustness.economic_sensitivities(_arm(cell, "H"), _arm(cell, "M"))
    components = result["component_deltas_eur"]
    counterfactuals = result["counterfactual_deltas_eur"]

    assert counterfactuals["default"] == pytest.approx(math.fsum(components.values()))
    assert counterfactuals["no_methane_endowment"] == pytest.approx(
        counterfactuals["default"] - components["methane_endowment_liability_change"]
    )
    assert set(result["one_factor_multiplier_deltas_eur"]) == {
        "methane_endowment",
        "declared_wear",
        "solar_opportunity",
    }
    assert all(
        set(values) == {"0", "0.5", "1", "2"}
        for values in result["one_factor_multiplier_deltas_eur"].values()
    )


def test_cell_analysis_closes_execution_accounting_safety_and_expression() -> None:
    cell = robustness.EconomicRobustnessCellV1(6.0, 35.0, 901)
    heuristic = _arm(cell, "H")
    mpc = _arm(cell, "M")

    analysis = robustness.evaluate_robustness_cell(
        cell,
        heuristic,
        mpc,
        assumptions=EconomicAssumptionsV1(),
    )

    assert analysis["execution"]["passed"] is True
    assert analysis["identity_accounting"]["passed"] is True
    assert analysis["safety_authority"]["passed"] is True
    assert analysis["service_comparability"]["passed"] is True
    assert analysis["task_expression"]["passed"] is True
    assert analysis["nonterminal_expression"]["passed"] is True
    assert analysis["settlements"]["passed"] is True
    assert analysis["latency"]["excluded_from_deterministic_evidence"] is True

    drifted = deepcopy(mpc)
    drifted["reward_trace_sum_eur"] += 0.01
    failed = robustness.evaluate_robustness_cell(
        cell,
        heuristic,
        drifted,
        assumptions=EconomicAssumptionsV1(),
    )
    assert failed["identity_accounting"]["passed"] is False


def test_complete_synthetic_matrix_reports_every_stratum_and_gate() -> None:
    summaries = []
    for cell in robustness.REGISTERED_ECONOMIC_ROBUSTNESS_CELLS:
        analysis = robustness.evaluate_robustness_cell(
            cell,
            _arm(cell, "H"),
            _arm(cell, "M"),
            assumptions=EconomicAssumptionsV1(),
        )
        summaries.append(
            {
                "cell": cell.to_dict(),
                "status": "complete",
                "artifact_verified": True,
                "analysis": analysis,
            }
        )

    result = robustness.aggregate_robustness_cells(summaries, source_valid=True)

    assert result["status"] == "complete_all_gates_passed"
    assert [(gate["id"], gate["status"]) for gate in result["gates"]] == [
        ("R1", "passed"),
        ("R2", "passed"),
        ("R3", "passed"),
        ("R4", "passed"),
        ("R5", "passed"),
        ("R6", "passed"),
        ("R7", "passed"),
        ("R8", "reported"),
    ]
    for stratum in result["strata"].values():
        assert stratum["observed_cells"] == 5
        assert stratum["missing_or_unanalyzable_cells"] == 0
        assert stratum["service_comparable_count"] == 5
        assert len(stratum["latency_by_seed"]) == 5
        assert (
            stratum["latency_summaries"]["H.controller_mean_ns"]["complete_pair_count"]
            == 5
        )
        assert (
            stratum["economic_sensitivity_summaries"]["counterfactual_deltas_eur"][
                "default"
            ]["missing_pair_count"]
            == 0
        )
    assert set(result["horizon_comparison_by_delivery_rate"]) == {"r35", "r42"}

    corrupted = deepcopy(summaries)
    corrupted[0]["artifact_verified"] = False
    failed = robustness.aggregate_robustness_cells(corrupted, source_valid=True)
    assert failed["gates"][0]["status"] == "passed"
    assert failed["gates"][1]["status"] == "failed"


def test_cell_verifier_recomputes_analysis_and_detects_tampering() -> None:
    cell = robustness.EconomicRobustnessCellV1(6.0, 35.0, 901)
    payload = _cell_payload(cell)

    robustness.verify_robustness_cell_payload(payload, require_registered=False)

    tampered = deepcopy(payload)
    tampered["arms"]["M"]["economic_metrics"]["accepted_solar_kwh"] += 1.0
    with pytest.raises(ValueError, match="analysis differs"):
        robustness.verify_robustness_cell_payload(
            tampered,
            require_registered=False,
        )


def test_cell_deterministic_digest_excludes_all_wall_clock_evidence() -> None:
    cell = robustness.EconomicRobustnessCellV1(6.0, 35.0, 901)
    payload = _cell_payload(cell)
    expected = robustness.cell_deterministic_body(payload)
    timed = deepcopy(payload)
    timed["elapsed_ns"] += 999
    timed["started_at_utc"] = "2099-01-01T00:00:00.000000Z"
    timed["completed_at_utc"] = "2099-01-01T00:00:01.000000Z"
    timed["arms"]["H"]["elapsed_ns"] += 999
    timed["arms"]["H"]["evidence"]["controller_elapsed_ns"][0] += 999
    timed["arms"]["H"]["evidence"]["dispatcher_elapsed_ns"][0][0] += 999
    timed["arms"]["M"]["elapsed_ns"] += 999
    timed["arms"]["M"]["evidence"]["decision_latency"]["mean_ns"] += 999
    timed["analysis"]["latency"]["H"]["arm_elapsed_ns"] += 999

    assert robustness.cell_deterministic_body(timed) == expected


def test_output_directory_must_be_absolute_new_and_outside_repository(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]

    with pytest.raises(ValueError, match="absolute"):
        experiment.validate_output_directory(
            Path("relative-output"), repository_root=repository
        )
    with pytest.raises(ValueError, match="outside"):
        experiment.validate_output_directory(
            repository / "inside-output", repository_root=repository
        )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        experiment.validate_output_directory(existing, repository_root=repository)
    assert (
        experiment.validate_output_directory(
            tmp_path / "new-output", repository_root=repository
        )
        == (tmp_path / "new-output").resolve()
    )


def test_noncanonical_runner_resets_both_arms_before_action_and_verifies_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    cell = robustness.EconomicRobustnessCellV1(6.0, 35.0, 901)
    events: list[str] = []

    class FakeEnvironment:
        def __init__(self, name: str) -> None:
            self.name = name

    environments: list[FakeEnvironment] = []

    def new_environment(**_: Any) -> FakeEnvironment:
        environment = FakeEnvironment(f"environment-{len(environments)}")
        environments.append(environment)
        events.append(f"construct-{environment.name}")
        return environment

    def reset_environment(environment: FakeEnvironment, *, seed: int) -> dict[str, str]:
        assert seed == 901
        events.append(f"reset-{environment.name}")
        return _identity()

    def heuristic_arm(**_: Any) -> dict[str, Any]:
        events.append("act-H")
        return _arm(cell, "H")

    def mpc_arm(**_: Any) -> dict[str, Any]:
        events.append("act-M")
        return _arm(cell, "M")

    monkeypatch.setattr(experiment, "_new_environment", new_environment)
    monkeypatch.setattr(experiment, "_reset_environment", reset_environment)
    monkeypatch.setattr(experiment, "execute_heuristic_arm", heuristic_arm)
    monkeypatch.setattr(experiment, "execute_mpc_arm", mpc_arm)
    monkeypatch.setattr(
        experiment, "_source_snapshot", lambda **_: deepcopy(_clean_source())
    )

    output = tmp_path / "robustness"
    result = experiment.run_economic_robustness_tournament(
        output,
        cells=[cell],
        require_registered=False,
        repository_root=repository,
    )

    assert events == [
        "construct-environment-0",
        "construct-environment-1",
        "reset-environment-0",
        "reset-environment-1",
        "act-H",
        "act-M",
    ]
    assert result["status"] == "complete_gate_failed"
    assert result["cell_count"] == 1
    assert sorted(path.name for path in output.iterdir()) == [
        "01_h6-r35-s901.json",
        "report.json",
    ]
    assert (
        experiment.verify_robustness_report(
            output / "report.json",
            require_registered=False,
            repository_root=repository,
        )["report_sha256"]
        == result["report_sha256"]
    )

    cell_path = output / "01_h6-r35-s901.json"
    cell_path.write_bytes(cell_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="raw digest"):
        experiment.verify_robustness_report(
            output / "report.json",
            require_registered=False,
            repository_root=repository,
        )


def test_runner_retains_a_preflight_failure_and_continues_to_the_next_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    failed_cell = robustness.EconomicRobustnessCellV1(6.0, 35.0, 901)
    next_cell = robustness.EconomicRobustnessCellV1(6.0, 35.0, 902)
    acted_seeds: list[int] = []

    monkeypatch.setattr(experiment, "_new_environment", lambda **_: object())

    def reset_environment(_: object, *, seed: int) -> dict[str, str]:
        if seed == failed_cell.seed:
            raise RuntimeError("synthetic reset failure")
        return _identity()

    def heuristic_arm(**values: Any) -> dict[str, Any]:
        seed = int(values["seed"])
        acted_seeds.append(seed)
        return _arm(next_cell, "H")

    def mpc_arm(**values: Any) -> dict[str, Any]:
        seed = int(values["seed"])
        acted_seeds.append(seed)
        return _arm(next_cell, "M")

    monkeypatch.setattr(experiment, "_reset_environment", reset_environment)
    monkeypatch.setattr(experiment, "execute_heuristic_arm", heuristic_arm)
    monkeypatch.setattr(experiment, "execute_mpc_arm", mpc_arm)
    monkeypatch.setattr(
        experiment, "_source_snapshot", lambda **_: deepcopy(_clean_source())
    )

    result = experiment.run_economic_robustness_tournament(
        tmp_path / "failure-retention",
        cells=[failed_cell, next_cell],
        require_registered=False,
        repository_root=repository,
    )

    assert acted_seeds == [902, 902]
    assert result["cell_count"] == 2
    failed_payload = experiment._load_json(
        tmp_path / "failure-retention" / "01_h6-r35-s901.json"
    )
    assert failed_payload["status"] == "failed"
    assert failed_payload["failure"]["phase"] == "cell_preflight"
    assert failed_payload["analysis"] is None
    assert failed_payload["arms"]["H"]["completed_decisions"] == 0
    assert failed_payload["arms"]["M"]["completed_decisions"] == 0


def test_registered_runner_refuses_seed_substitution_before_source_or_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_called = False

    def source_snapshot(**_: Any) -> dict[str, Any]:
        nonlocal source_called
        source_called = True
        return {}

    monkeypatch.setattr(experiment, "_source_snapshot", source_snapshot)
    substitute = robustness.EconomicRobustnessCellV1(6.0, 35.0, 901)

    with pytest.raises(ValueError, match="exact frozen matrix"):
        experiment.run_economic_robustness_tournament(
            tmp_path / "never-created",
            cells=[substitute],
            repository_root=Path(__file__).resolve().parents[2],
        )
    assert source_called is False
    assert not (tmp_path / "never-created").exists()

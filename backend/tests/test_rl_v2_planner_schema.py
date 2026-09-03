from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from types import SimpleNamespace

import pytest
from plant_sim import PlantState

from plant_backend.rl.v2 import planner_schema as schema
from plant_backend.rl.v2.metrics import EpisodeMetricsV2
from plant_backend.rl.v2.planner_schema import (
    A0_CONTROLLER,
    A0_CONTROLLER_SHA256,
    ACTION_LIBRARY,
    ACTION_LIBRARY_SHA256,
    ACTION_SOURCE_PHASE_ORDER,
    PROTOCOL_ANCHOR_COMMIT,
    PROTOCOL_BLOB_SHA256,
    PROTOCOL_PARENT_COMMIT,
    SOURCE_BOUNDARY_CLAIM,
    ActionRecordV1,
    CanonicalPayloadV1,
    FeasiblePlannerArtifactV1,
    FeasiblePlannerResultV1,
    FinalScoreV1,
    PlannerFailureV1,
    PlannerRunnerFailureV1,
    ReplayEvidenceV1,
    action_record_trace_payload,
    artifact_preimage,
    build_incumbent_comparison,
    build_planner_settings,
    build_search_diagnostics,
    canonical_json,
    canonical_payload,
    canonical_payload_preimage,
    compare_action_lineage,
    compare_final_scores,
    compare_service_fractions,
    core_result_preimage,
    outcome_payload,
    planner_artifact_from_dict,
    planner_result_from_dict,
    planner_settings_preimage,
    replay_binding_payload,
    request_trace_payload,
    runtime_payload,
    sanitize_failure_message,
    source_snapshot_payload,
    source_trace_payload,
)


SHA_A = "1" * 64
SHA_B = "2" * 64
SHA_C = "3" * 64
SHA_D = "4" * 64


def _score(**changes: object) -> FinalScoreV1:
    values: dict[str, object] = {
        "delivered_kg": 0.0,
        "service_numerator": 0,
        "service_denominator": 0,
        "worst_fill": 1.0,
        "terminal_ch4_equivalent_inventory_kg": 0.0,
        "terminal_battery_energy_kwh": 0.0,
        "forced_stops": 0,
        "aborted_starts": 0,
        "cold_starts": 0,
        "curtailed_kwh": 0.0,
        "battery_throughput_kwh": 0.0,
    }
    values.update(changes)
    return FinalScoreV1(**values)  # type: ignore[arg-type]


def _action(
    *,
    source_phase: str = "a0_capture",
    source_code: int = -1,
    binding: str = A0_CONTROLLER_SHA256,
    battery_share: float = 0.0,
) -> ActionRecordV1:
    return ActionRecordV1(
        decision_index=0,
        source_phase=source_phase,
        source_code=source_code,
        source_binding_sha256=binding,
        controller_invalid=False,
        battery_share=battery_share,
        electrolyser_share=0.0,
        dac_share=0.0,
        methanator_share=0.0,
        battery_support_kw=0.0,
    )


def _settings(*, result_sha: str, witness_sha: str, guidance_sha: str):
    return build_planner_settings(
        expected_decisions=1,
        expected_physics_steps=1,
        plant_config_sha256=SHA_A,
        constraints_sha256=SHA_A,
        reward_weights_sha256=SHA_A,
        episode_config_sha256=SHA_A,
        checkpoint_contract_digest=SHA_A,
        relaxation_model_sha256=SHA_B,
        relaxation_result_sha256=result_sha,
        relaxation_witness_sha256=witness_sha,
        relaxation_guidance_sha256=guidance_sha,
    )


def _replay(
    settings,
    result,
    witness,
    guidance,
    *,
    role: str = "a0_incumbent",
    record: ActionRecordV1 | None = None,
    score: FinalScoreV1 | None = None,
) -> ReplayEvidenceV1:
    records = (_action() if record is None else record,)
    final_score = _score() if score is None else score
    request = request_trace_payload(records)
    source = source_trace_payload(records)
    action_trace = action_record_trace_payload(records)
    metrics_value = EpisodeMetricsV2().to_dict()
    truth_value = PlantState().to_dict()
    metrics_value.update(
        {
            "delivered_kg": final_score.delivered_kg,
            "commitments_met": final_score.service_numerator,
            "commitments_settled": final_score.service_denominator,
            "worst_commitment_fill_rate": final_score.worst_fill,
            "terminal_battery_energy_kwh": final_score.terminal_battery_energy_kwh,
            "electrolyser_forced_stops": final_score.forced_stops,
            "electrolyser_aborted_starts": final_score.aborted_starts,
            "electrolyser_cold_starts": final_score.cold_starts,
            "curtailed_kwh": final_score.curtailed_kwh,
        }
    )
    truth_value["battery_energy_kwh"] = final_score.terminal_battery_energy_kwh
    metrics = canonical_payload("planner-metrics-v1", metrics_value)
    truth = canonical_payload("planner-terminal-truth-v1", truth_value)
    telemetry = canonical_payload("planner-terminal-telemetry-v1", truth_value)
    audit = canonical_payload(
        "planner-delivery-audit-v1",
        {
            "frozen": True,
            "schedule": [],
            "settlements": [],
            "current": {},
            "cumulative_requested_kg": 0.0,
            "cumulative_delivered_kg": final_score.delivered_kg,
            "cumulative_unmet_kg": 0.0,
            "pending_count": 0,
            "next_pending": None,
        },
    )
    outcome = outcome_payload(
        manifest_sha256=SHA_C,
        exogenous_episode_sha256=SHA_D,
        checkpoint_contract_digest=SHA_A,
        request_trace=request,
        metrics=metrics,
        terminal_truth=truth,
        terminal_telemetry=telemetry,
        delivery_audit=audit,
        battery_throughput_kwh=final_score.battery_throughput_kwh,
        decision_count=1,
        physics_steps=1,
        terminated=True,
        truncated=False,
    )
    binding = replay_binding_payload(
        role=role,
        settings_sha256=settings.settings_sha256,
        relaxation_guidance_sha256=guidance.sha256,
        manifest_sha256=SHA_C,
        exogenous_episode_sha256=SHA_D,
        checkpoint_contract_digest=SHA_A,
        request_trace=request,
        source_trace=source,
        action_record_trace=action_trace,
        outcome=outcome,
    )
    return ReplayEvidenceV1(
        role=role,
        manifest_sha256=SHA_C,
        exogenous_episode_sha256=SHA_D,
        checkpoint_contract_digest=SHA_A,
        action_records=records,
        request_trace=request,
        source_trace=source,
        action_record_trace=action_trace,
        decision_count=1,
        physics_steps=1,
        terminated=True,
        truncated=False,
        battery_throughput_kwh=final_score.battery_throughput_kwh,
        metrics=metrics,
        terminal_truth=truth,
        terminal_telemetry=telemetry,
        delivery_audit=audit,
        generation_outcome=outcome,
        replay_outcome=outcome,
        replay_binding=binding,
        final_score=final_score,
        replay_elapsed_ns=10,
    )


def _successful_core(
    *,
    finalist_score: FinalScoreV1 | None = None,
    selected_role_override: str | None = None,
) -> FeasiblePlannerResultV1:
    solve_settings = {
        "feasibility_tolerance": 1e-7,
        "physical_tolerance": 1e-6,
        "optimality_tolerance_kg": 1e-5,
        "marginal_sign_tolerance": 1e-9,
        "solver_options": {},
        "highs_runtime_version": None,
        "provenance_gaps": [],
    }
    witness_value = {"witness": []}
    relaxation_result = canonical_payload(
        "planner-relaxation-result-v1",
        {
            "schema_version": 1,
            "success": True,
            "status": 0,
            "failure_code": None,
            "message": "fixture",
            "solver": "highs",
            "solver_version": None,
            "solve_settings": solve_settings,
            "model_sha256": SHA_B,
            "checkpoint_contract_digest": SHA_A,
            "exogenous_episode_sha256": SHA_D,
            "canonical_episode_verified": False,
            "evidence_provenance_status": "noncanonical research fixture",
            "preserve_initial_endowment": False,
            "is_task_upper_bound": True,
            "requested_kg": 10.0,
            "numerical_primal_delivery_kg": 10.0,
            "dual_certified_delivery_upper_bound_kg": 10.0,
            "relaxed_fill_ceiling": 1.0,
            "primal_dual_gap_kg": 0.0,
            "numerical_objective": -10.0,
            "iterations": 1,
            "marginal_projection_within_tolerance": True,
            "certificate": {"fixture": True},
            "witness": witness_value,
            "privileged_information": "fixture",
            "certificate_scope": "fixture",
            "simulator_embedding_status": "fixture",
        },
    )
    witness = canonical_payload("planner-relaxation-witness-v1", witness_value)
    guidance = canonical_payload(
        "planner-relaxation-guidance-v1",
        {
            "schema_version": 1,
            "model_sha256": SHA_B,
            "result_sha256": relaxation_result.sha256,
            "witness_sha256": witness.sha256,
            "status": 0,
            "canonical_episode_verified": False,
            "preserve_initial_endowment": False,
            "is_task_upper_bound": True,
            "solver": "highs",
            "solver_version": None,
            "solve_settings": solve_settings,
            "checkpoint_contract_digest": SHA_A,
            "exogenous_episode_sha256": SHA_D,
            "solar_truth_sha256": SHA_C,
        },
    )
    settings = _settings(
        result_sha=relaxation_result.sha256,
        witness_sha=witness.sha256,
        guidance_sha=guidance.sha256,
    )
    replay = _replay(settings, relaxation_result, witness, guidance)
    finalist = None
    if finalist_score is not None:
        finalist = _replay(
            settings,
            relaxation_result,
            witness,
            guidance,
            role="searched_finalist",
            record=_action(
                source_phase="beam",
                source_code=2,
                binding=ACTION_LIBRARY_SHA256,
                battery_share=0.5,
            ),
            score=finalist_score,
        )
    selected_role = selected_role_override or (
        "searched_finalist"
        if finalist is not None
        and compare_final_scores(finalist.final_score, replay.final_score) > 0
        else "a0_incumbent"
    )
    selected = replay if selected_role == "a0_incumbent" else finalist
    assert selected is not None
    comparison = build_incumbent_comparison(
        selected_role=selected_role,
        a0_score=replay.final_score,
        selected_score=selected.final_score,
        a0_request_trace=replay.request_trace,
        selected_request_trace=selected.request_trace,
        upper_bound_kg=10.0,
        optimality_tolerance_kg=1e-5,
    )
    search = build_search_diagnostics(
        completed_macro_depths=1,
        attempted_edges=15,
        completed_edges=15,
        generated_candidates=15,
        a0_cursor_validations=1,
        a0_injections=0,
        exact_duplicates=0,
        frontier_retained_total=0,
        peak_frontier=1,
        coordinate_trials_attempted=45,
        coordinate_trials_completed=45,
        coordinate_trials_accepted=0,
        a0_generation_decisions=1,
        a0_evidence_replay_decisions=1,
        fresh_finalist_replay_decisions=1 if finalist is not None else 0,
        max_live_environments_observed=1,
        termination_reason="completed",
        elapsed_ns=20,
        peak_rss_bytes=100,
    )
    return FeasiblePlannerResultV1(
        schema_version=1,
        planner_kind="best-known-feasible-future-information-v1",
        evidence_scope="noncanonical_unit_fixture",
        privileged_information="complete private episode truth and commitments",
        optimality_claim="none",
        evidence_ready=False,
        provenance_gaps=("clean start/end source and dependency-lock audit required",),
        success=True,
        failure=None,
        seed=0,
        canonical_episode_verified=False,
        manifest_sha256=SHA_C,
        exogenous_episode_sha256=SHA_D,
        checkpoint_contract_digest=SHA_A,
        relaxation_model_sha256=SHA_B,
        relaxation_result_sha256=relaxation_result.sha256,
        relaxation_result=relaxation_result,
        relaxation_witness=witness,
        relaxation_guidance=guidance,
        settings=settings,
        search=search,
        a0_incumbent=replay,
        searched_finalist=finalist,
        selected_role=selected_role,
        comparison=comparison,
    )


def _source(*, dirty: bool = True):
    return source_snapshot_payload(
        {
            "schema_version": 1,
            "git_revision": "a" * 40,
            "git_dirty": dirty,
            "tracked_binary_diff_sha256": (
                SHA_A if dirty else hashlib.sha256(b"").hexdigest()
            ),
            "tracked_binary_diff_bytes": 0,
            "untracked_files": [],
            "worktree_identity_format": "git-worktree-content-v1",
            "worktree_identity_sha256": SHA_B,
            "identity_complete": True,
        }
    )


def _runtime():
    return runtime_payload(
        {
            "schema_version": 1,
            "python_version": "3.12",
            "python_implementation": "CPython",
            "python_compiler": "Clang",
            "numpy_version": "2",
            "scipy_version": "1.14",
            "highs_runtime_version": None,
            "gymnasium_version": "1",
            "platform_system": "Darwin",
            "platform_release": "x",
            "platform_version": "x",
            "platform_machine": "arm64",
            "platform_processor": None,
            "pointer_bits": 64,
            "byteorder": "little",
            "clock_method": "time.perf_counter_ns",
            "rss_method": "resource.RUSAGE_SELF.ru_maxrss:darwin-bytes:other-kib",
        }
    )


def _changed_search(search, **changes: object):
    values = search.to_dict()
    values.pop("deterministic_counters")
    values.update(changes)
    return build_search_diagnostics(**values)


def _registered_relaxation_graph():
    solve_settings = {
        "feasibility_tolerance": 1e-7,
        "physical_tolerance": 1e-6,
        "optimality_tolerance_kg": 1e-5,
        "marginal_sign_tolerance": 1e-9,
        "solver_options": {},
        "highs_runtime_version": None,
        "provenance_gaps": [
            "git source revision and dependency-lock digest must be attached by the evaluation runner",
            "linked HiGHS runtime version is not exposed by this SciPy build",
        ],
    }
    residuals = {
        "max_equality_normalized": 0.0,
        "max_inequality_violation_normalized": 0.0,
        "max_box_violation_normalized": 0.0,
        "max_solar_balance_kw": 0.0,
        "max_battery_balance_kwh": 0.0,
        "max_h2_balance_kg": 0.0,
        "max_co2_balance_kg": 0.0,
        "max_ch4_balance_kg": 0.0,
        "max_headroom_violation_kg": 0.0,
        "max_yield_violation_kg": 0.0,
        "max_delivery_bound_violation_kg": 0.0,
        "max_physical_box_violation": 0.0,
        "max_component_power_violation_kw": 0.0,
        "max_physical_violation": 0.0,
    }
    witness_value = {
        "normalized_variables": [0.0] * 29,
        "solar_to_battery_kw": [0.0],
        "solar_to_electrolyser_kw": [0.0],
        "solar_to_dac_kw": [0.0],
        "solar_to_methanator_kw": [0.0],
        "battery_to_electrolyser_kw": [0.0],
        "curtailed_solar_kw": [0.0],
        "h2_produced_kg": [0.0],
        "co2_captured_kg": [0.0],
        "ch4_produced_kg": [0.0],
        "battery_energy_kwh": [0.0, 0.0],
        "h2_inventory_kg": [0.0, 0.0],
        "co2_inventory_kg": [0.0, 0.0],
        "ch4_inventory_kg": [0.0, 0.0],
        "deliveries": [
            {
                "id": f"delivery-{index}",
                "due_step": 1,
                "requested_kg": 0.0,
                "delivered_kg": 0.0,
            }
            for index in range(12)
        ],
        "delivered_kg": 0.0,
        "residuals": residuals,
    }
    certificate = {
        "minimization_lower_bound_exact": "0/1",
        "uncapped_delivery_upper_bound_exact": "0/1",
        "requested_kg_exact": "0/1",
        "certified_delivery_upper_bound_exact": "0/1",
        "dual_certified_delivery_upper_bound_kg": 0.0,
        "positive_inequality_marginal_count": 0,
        "max_positive_inequality_marginal": 0.0,
        "arithmetic": "exact Fraction.from_float over stored IEEE-754 coefficients",
        "float_conversion": "directed toward positive infinity",
        "certificate_scope": "formal certificate for the encoded finite-box linear program only",
    }
    result_value = {
        "schema_version": 1,
        "success": True,
        "status": 0,
        "failure_code": None,
        "message": "fixture",
        "solver": "scipy.optimize.linprog(method='highs')",
        "solver_version": "1.14",
        "solve_settings": solve_settings,
        "model_sha256": SHA_B,
        "checkpoint_contract_digest": SHA_A,
        "exogenous_episode_sha256": SHA_D,
        "canonical_episode_verified": True,
        "evidence_provenance_status": "canonical sample_episode regeneration verified",
        "preserve_initial_endowment": False,
        "is_task_upper_bound": True,
        "requested_kg": 0.0,
        "numerical_primal_delivery_kg": 0.0,
        "dual_certified_delivery_upper_bound_kg": 0.0,
        "relaxed_fill_ceiling": 1.0,
        "primal_dual_gap_kg": 0.0,
        "numerical_objective": 0.0,
        "iterations": 0,
        "marginal_projection_within_tolerance": True,
        "certificate": certificate,
        "witness": witness_value,
        "privileged_information": "offline perfect-information research bound; not a deployable controller",
        "certificate_scope": "formal certificate for the encoded finite-box linear program only",
        "simulator_embedding_status": "relaxation embedding is numerically validated within declared tolerances; not bit-exact certified",
    }
    result = canonical_payload("planner-relaxation-result-v1", result_value)
    witness = canonical_payload("planner-relaxation-witness-v1", witness_value)
    guidance_value = {
        "schema_version": 1,
        "model_sha256": SHA_B,
        "result_sha256": result.sha256,
        "witness_sha256": witness.sha256,
        "status": 0,
        "canonical_episode_verified": True,
        "preserve_initial_endowment": False,
        "is_task_upper_bound": True,
        "solver": result_value["solver"],
        "solver_version": result_value["solver_version"],
        "solve_settings": solve_settings,
        "checkpoint_contract_digest": SHA_A,
        "exogenous_episode_sha256": SHA_D,
        "solar_truth_sha256": SHA_C,
    }
    guidance = canonical_payload("planner-relaxation-guidance-v1", guidance_value)
    settings = _settings(
        result_sha=result.sha256,
        witness_sha=witness.sha256,
        guidance_sha=guidance.sha256,
    )
    return settings, result, witness, guidance, result_value


def test_protocol_anchors_and_registered_payloads_are_exact() -> None:
    assert PROTOCOL_PARENT_COMMIT == "2a3c99a73d6e3f071d9721f72691291f307a6125"
    assert PROTOCOL_ANCHOR_COMMIT == "265d9ad4e430b3f1a68d7e3ce39ab0a60b26f000"
    assert (
        PROTOCOL_BLOB_SHA256
        == "50ce791a4c98e087c49dcb0eafe6210a081bd115cede1ba5f38610c8ff11f635"
    )
    assert ACTION_LIBRARY.sha256 == ACTION_LIBRARY_SHA256
    assert A0_CONTROLLER.sha256 == A0_CONTROLLER_SHA256
    assert len(json.loads(ACTION_LIBRARY.canonical_json)["templates"]) == 15


def test_canonical_payload_uses_exact_domain_preimage_and_is_detached() -> None:
    payload = canonical_payload("planner-state-v1", {"é": [1, 2.0]})
    assert payload.canonical_json == '{"é":[1,2.0]}'
    assert (
        payload.sha256
        == hashlib.sha256(
            canonical_payload_preimage(payload.domain, payload.canonical_json)
        ).hexdigest()
    )
    detached = payload.to_dict()
    detached["sha256"] = SHA_A
    assert payload.sha256 != SHA_A
    with pytest.raises(ValueError, match="canonical form"):
        CanonicalPayloadV1(payload.domain, '{"x": 1}', payload.sha256)
    with pytest.raises(ValueError, match="does not match"):
        CanonicalPayloadV1(payload.domain, payload.canonical_json, SHA_A)
    with pytest.raises(ValueError, match="unsupported"):
        canonical_payload("other", {})
    with pytest.raises(ValueError):
        canonical_json({"x": math.nan})


def test_action_records_are_closed_immutable_and_have_separate_trace_domains() -> None:
    record = ActionRecordV1.from_action(
        SimpleNamespace(
            battery_share=0.5,
            electrolyser_share=0.25,
            dac_share=0.125,
            methanator_share=0.0,
            battery_support_kw=3.0,
        ),
        decision_index=0,
        source_phase="beam",
        source_code=2,
        source_binding_sha256=ACTION_LIBRARY_SHA256,
    )
    assert (
        request_trace_payload((record,)).value()["records"][0] == record.request_dict()
    )
    assert source_trace_payload((record,)).value()["records"][0] == record.source_dict()
    assert (
        action_record_trace_payload((record,)).value()["records"][0] == record.to_dict()
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.battery_share = 0.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="sum"):
        dataclasses.replace(record, electrolyser_share=0.6)
    with pytest.raises(ValueError, match="together"):
        dataclasses.replace(record, source_phase="a0_capture")
    with pytest.raises(ValueError, match="controller-invalid"):
        dataclasses.replace(record, controller_invalid=True)
    with pytest.raises(ValueError, match="contiguous"):
        request_trace_payload((dataclasses.replace(record, decision_index=1),))


def test_final_score_comparator_uses_binary64_exact_service_and_all_directions() -> (
    None
):
    base = _score(delivered_kg=1.0, service_numerator=1, service_denominator=2)
    assert (
        compare_final_scores(
            dataclasses.replace(base, delivered_kg=math.nextafter(1.0, math.inf)), base
        )
        == 1
    )
    assert compare_service_fractions(0, 0, 1, 1) == 0
    assert (
        compare_final_scores(
            dataclasses.replace(base, service_numerator=2, service_denominator=3), base
        )
        == 1
    )
    assert (
        compare_final_scores(
            dataclasses.replace(base, terminal_battery_energy_kwh=1.0), base
        )
        == 1
    )
    assert compare_final_scores(dataclasses.replace(base, forced_stops=1), base) == -1
    assert (
        compare_final_scores(dataclasses.replace(base, curtailed_kwh=1.0), base) == -1
    )


@pytest.mark.parametrize(
    ("field", "higher_is_better"),
    [
        ("delivered_kg", True),
        ("worst_fill", True),
        ("terminal_ch4_equivalent_inventory_kg", True),
        ("terminal_battery_energy_kwh", True),
        ("forced_stops", False),
        ("aborted_starts", False),
        ("cold_starts", False),
        ("curtailed_kwh", False),
        ("battery_throughput_kwh", False),
    ],
)
def test_every_final_comparator_field_has_exact_direction_and_reverse(
    field: str, higher_is_better: bool
) -> None:
    base = _score(
        delivered_kg=5.0,
        service_numerator=1,
        service_denominator=2,
        worst_fill=0.5,
        terminal_ch4_equivalent_inventory_kg=5.0,
        terminal_battery_energy_kwh=5.0,
        forced_stops=1,
        aborted_starts=1,
        cold_starts=1,
        curtailed_kwh=5.0,
        battery_throughput_kwh=5.0,
    )
    value = getattr(base, field)
    if isinstance(value, int):
        changed_value = value + (1 if higher_is_better else -1)
    else:
        changed_value = math.nextafter(
            value, math.inf if higher_is_better else -math.inf
        )
    changed = dataclasses.replace(base, **{field: changed_value})
    assert compare_final_scores(changed, base) == 1
    assert compare_final_scores(base, changed) == -1
    assert compare_final_scores(base, dataclasses.replace(base)) == 0


def test_service_comparator_uses_exact_cross_products_and_reverse() -> None:
    base = _score(service_numerator=1, service_denominator=3)
    better = dataclasses.replace(base, service_numerator=2, service_denominator=5)
    assert compare_final_scores(better, base) == 1
    assert compare_final_scores(base, better) == -1
    assert compare_service_fractions(2, 4, 1, 2) == 0


def test_action_lineage_prefers_phase_then_code_then_request_bytes() -> None:
    beam_2 = (
        _action(source_phase="beam", source_code=2, binding=ACTION_LIBRARY_SHA256),
    )
    beam_3 = (
        _action(source_phase="beam", source_code=3, binding=ACTION_LIBRARY_SHA256),
    )
    assert ACTION_SOURCE_PHASE_ORDER["beam"] == 1
    assert compare_action_lineage(beam_2, beam_3) == 1
    beam_2_larger_request = (dataclasses.replace(beam_2[0], battery_share=0.5),)
    assert compare_action_lineage(beam_2, beam_2_larger_request) == 1


def test_settings_hash_preimage_reduces_nested_payloads_only() -> None:
    result = canonical_payload("planner-relaxation-result-v1", {})
    witness = canonical_payload("planner-relaxation-witness-v1", {})
    guidance = canonical_payload("planner-relaxation-guidance-v1", {})
    settings = _settings(
        result_sha=result.sha256,
        witness_sha=witness.sha256,
        guidance_sha=guidance.sha256,
    )
    preimage = planner_settings_preimage(settings)
    assert preimage["action_library"] == {
        "domain": ACTION_LIBRARY.domain,
        "sha256": ACTION_LIBRARY.sha256,
    }
    assert "canonical_json" not in preimage["a0_controller"]
    assert (
        settings.settings_sha256
        == canonical_payload("planner-settings-v1", preimage).sha256
    )
    with pytest.raises(ValueError, match="settings_sha256"):
        dataclasses.replace(settings, settings_sha256=SHA_A)


def test_failure_sanitizer_and_deterministic_counter_identity() -> None:
    assert sanitize_failure_message("  bad\nthing\té  ") == "bad thing ?"
    assert sanitize_failure_message(" \n ") == "unspecified failure"
    with pytest.raises(ValueError, match="sanitized"):
        PlannerFailureV1(
            phase="beam_search",
            code="clone_error",
            message="not\nclean",
            completed_macro_depths=0,
            attempted_edges=1,
            completed_edges=0,
            coordinate_trials_completed=0,
            elapsed_ns=1,
            observed_resource_value=None,
        )
    diagnostics = build_search_diagnostics(
        completed_macro_depths=0,
        attempted_edges=0,
        completed_edges=0,
        generated_candidates=0,
        a0_cursor_validations=0,
        a0_injections=0,
        exact_duplicates=0,
        frontier_retained_total=0,
        peak_frontier=0,
        coordinate_trials_attempted=0,
        coordinate_trials_completed=0,
        coordinate_trials_accepted=0,
        a0_generation_decisions=0,
        a0_evidence_replay_decisions=0,
        fresh_finalist_replay_decisions=0,
        max_live_environments_observed=0,
        termination_reason="invalid_input",
        elapsed_ns=99,
        peak_rss_bytes=100,
    )
    assert "elapsed_ns" not in diagnostics.deterministic_counters.value()
    assert "peak_rss_bytes" not in diagnostics.deterministic_counters.value()


def test_comparison_formulas_and_bridge_suppression() -> None:
    trace = request_trace_payload((_action(),))
    valid = build_incumbent_comparison(
        selected_role="searched_finalist",
        a0_score=_score(delivered_kg=2.0),
        selected_score=_score(delivered_kg=6.0),
        a0_request_trace=trace,
        selected_request_trace=trace,
        upper_bound_kg=10.0,
        optimality_tolerance_kg=1e-5,
    )
    assert valid.gain_kg == 4.0
    assert valid.residual_gap_kg == 4.0
    assert valid.fraction_closed == 0.5
    bridge = build_incumbent_comparison(
        selected_role="searched_finalist",
        a0_score=_score(delivered_kg=2.0),
        selected_score=_score(delivered_kg=11.0),
        a0_request_trace=trace,
        selected_request_trace=trace,
        upper_bound_kg=10.0,
        optimality_tolerance_kg=1e-5,
    )
    assert bridge.bracket_status == "bound_bridge_violation"
    assert bridge.residual_gap_kg is None and bridge.fraction_closed is None


@pytest.mark.parametrize(
    ("finalist_score", "expected_role"),
    [
        (_score(delivered_kg=1.0), "searched_finalist"),
        (_score(curtailed_kwh=1.0), "a0_incumbent"),
        (_score(), "a0_incumbent"),
    ],
)
def test_core_selects_searched_finalist_exactly_on_strict_score_improvement(
    finalist_score: FinalScoreV1, expected_role: str
) -> None:
    core = _successful_core(finalist_score=finalist_score)
    assert core.selected_role == expected_role
    wrong_role = (
        "searched_finalist" if expected_role == "a0_incumbent" else "a0_incumbent"
    )
    with pytest.raises(ValueError, match="exactly when"):
        _successful_core(
            finalist_score=finalist_score,
            selected_role_override=wrong_role,
        )


def test_replay_core_and_runner_records_validate_and_detach() -> None:
    core = _successful_core()
    assert (
        core.deterministic_result_sha256
        == canonical_payload("planner-result-v1", core_result_preimage(core)).sha256
    )
    detached = core.to_dict()
    detached["search"]["attempted_edges"] = 999
    assert core.search.attempted_edges == 15
    source = _source()
    artifact = FeasiblePlannerArtifactV1(
        schema_version=1,
        success=True,
        evidence_ready=False,
        core_result=core,
        core_result_sha256=core.deterministic_result_sha256,
        runner_failure=None,
        start_source=source,
        end_source=source,
        start_lock_sha256=SHA_A,
        end_lock_sha256=SHA_A,
        runtime_versions=_runtime(),
        source_boundary_claim=SOURCE_BOUNDARY_CLAIM,
    )
    assert (
        artifact.artifact_sha256
        == canonical_payload(
            "planner-runner-artifact-v1", artifact_preimage(artifact)
        ).sha256
    )
    assert planner_result_from_dict(core.to_dict()) == core
    assert planner_artifact_from_dict(artifact.to_dict()) == artifact
    altered = artifact.to_dict()
    altered["core_result"]["unexpected"] = True
    with pytest.raises(ValueError, match="incorrect field set"):
        planner_artifact_from_dict(altered)


def test_success_diagnostics_match_frozen_schedule_and_reject_counter_tampering() -> (
    None
):
    core = _successful_core()
    assert core.search.completed_macro_depths == 1
    assert core.search.attempted_edges == core.search.completed_edges == 15
    assert core.search.coordinate_trials_attempted == 45
    assert core.search.coordinate_trials_completed == 45
    assert core.search.fresh_finalist_replay_decisions == 0

    one_origin = _changed_search(
        core.search,
        coordinate_trials_attempted=30,
        coordinate_trials_completed=30,
    )
    rebuilt = dataclasses.replace(
        core,
        search=one_origin,
        deterministic_result_sha256="",
    )
    assert rebuilt.search.coordinate_trials_completed == 30
    overaccepted_one_origin = _changed_search(
        one_origin,
        coordinate_trials_accepted=3,
    )
    with pytest.raises(ValueError, match="executed frozen blocks"):
        dataclasses.replace(
            core,
            search=overaccepted_one_origin,
            deterministic_result_sha256="",
        )

    wrong_coordinate = _changed_search(
        core.search,
        coordinate_trials_attempted=44,
        coordinate_trials_completed=44,
    )
    with pytest.raises(ValueError, match="coordinate trials"):
        dataclasses.replace(
            core,
            search=wrong_coordinate,
            deterministic_result_sha256="",
        )

    over_edge_limit = _changed_search(
        core.search,
        attempted_edges=25_576,
        completed_edges=25_576,
        generated_candidates=25_576,
    )
    with pytest.raises(ValueError, match="edge ceiling"):
        dataclasses.replace(
            core,
            search=over_edge_limit,
            deterministic_result_sha256="",
        )

    over_frontier = _changed_search(core.search, peak_frontier=25)
    with pytest.raises(ValueError, match="beam width"):
        dataclasses.replace(
            core,
            search=over_frontier,
            deterministic_result_sha256="",
        )

    no_live_environment = _changed_search(
        core.search,
        max_live_environments_observed=0,
    )
    with pytest.raises(ValueError, match="at least one live environment"):
        dataclasses.replace(
            core,
            search=no_live_environment,
            deterministic_result_sha256="",
        )


def test_replay_and_core_reject_identity_action_binding_and_audit_tampering() -> None:
    core = _successful_core()
    replay = core.a0_incumbent
    assert replay is not None
    with pytest.raises(ValueError, match="outcome"):
        dataclasses.replace(replay, manifest_sha256=SHA_A)
    with pytest.raises(ValueError, match="outcome"):
        dataclasses.replace(replay, checkpoint_contract_digest=SHA_B)
    with pytest.raises(ValueError, match="request_trace"):
        dataclasses.replace(
            replay,
            action_records=(
                dataclasses.replace(replay.action_records[0], battery_share=0.5),
            ),
        )
    audit_value = replay.delivery_audit.value()
    assert isinstance(audit_value, dict)
    audit_value["cumulative_delivered_kg"] = 1.0
    with pytest.raises(ValueError, match="outcome"):
        dataclasses.replace(
            replay,
            delivery_audit=canonical_payload("planner-delivery-audit-v1", audit_value),
        )

    assert core.settings is not None
    assert core.relaxation_result is not None
    assert core.relaxation_witness is not None
    assert core.relaxation_guidance is not None
    bad_binding_replay = _replay(
        core.settings,
        core.relaxation_result,
        core.relaxation_witness,
        core.relaxation_guidance,
        record=_action(binding=SHA_A),
    )
    with pytest.raises(ValueError, match="source binding"):
        dataclasses.replace(
            core,
            a0_incumbent=bad_binding_replay,
            deterministic_result_sha256="",
        )


def test_registered_terminal_equivalent_recomputes_default_ratios_bit_exactly() -> None:
    core = _successful_core()
    replay = core.a0_incumbent
    assert replay is not None
    schema._validate_registered_terminal_scores((replay,))
    tampered_score = dataclasses.replace(
        replay.final_score,
        terminal_ch4_equivalent_inventory_kg=math.nextafter(0.0, math.inf),
    )
    tampered = dataclasses.replace(replay, final_score=tampered_score)
    with pytest.raises(ValueError, match="CH4-equivalent"):
        schema._validate_registered_terminal_scores((tampered,))


def test_registered_relaxation_requires_frozen_solver_and_certificate_structure() -> (
    None
):
    settings, result, witness, guidance, result_value = _registered_relaxation_graph()
    schema._validate_relaxation_graph(
        result=result,
        witness=witness,
        guidance=guidance,
        settings=settings,
        exogenous_episode_sha256=SHA_D,
        canonical_episode_verified=True,
    )

    wrong_solver = dict(result_value)
    wrong_solver["solver"] = "highs"
    with pytest.raises(ValueError, match="solver identity"):
        schema._validate_relaxation_graph(
            result=canonical_payload("planner-relaxation-result-v1", wrong_solver),
            witness=witness,
            guidance=guidance,
            settings=settings,
            exogenous_episode_sha256=SHA_D,
            canonical_episode_verified=True,
        )

    bad_certificate = dict(result_value)
    certificate = dict(bad_certificate["certificate"])
    certificate.pop("certificate_scope")
    bad_certificate["certificate"] = certificate
    with pytest.raises(ValueError, match="certificate.*field set"):
        schema._validate_relaxation_graph(
            result=canonical_payload("planner-relaxation-result-v1", bad_certificate),
            witness=witness,
            guidance=guidance,
            settings=settings,
            exogenous_episode_sha256=SHA_D,
            canonical_episode_verified=True,
        )

    wrong_settings = dict(result_value)
    solve_settings = dict(wrong_settings["solve_settings"])
    solve_settings["feasibility_tolerance"] = math.nextafter(1e-7, math.inf)
    wrong_settings["solve_settings"] = solve_settings
    with pytest.raises(ValueError, match="feasibility_tolerance"):
        schema._validate_relaxation_graph(
            result=canonical_payload("planner-relaxation-result-v1", wrong_settings),
            witness=witness,
            guidance=guidance,
            settings=settings,
            exogenous_episode_sha256=SHA_D,
            canonical_episode_verified=True,
        )


def test_registered_relaxation_rejects_rehashed_gap_and_objective_tampering() -> None:
    _, _, _, original_guidance, original_result = _registered_relaxation_graph()

    def rebound(result_value):
        result = canonical_payload("planner-relaxation-result-v1", result_value)
        witness = canonical_payload(
            "planner-relaxation-witness-v1", result_value["witness"]
        )
        guidance_value = original_guidance.value()
        assert isinstance(guidance_value, dict)
        guidance_value.update(
            {"result_sha256": result.sha256, "witness_sha256": witness.sha256}
        )
        guidance = canonical_payload("planner-relaxation-guidance-v1", guidance_value)
        settings = _settings(
            result_sha=result.sha256,
            witness_sha=witness.sha256,
            guidance_sha=guidance.sha256,
        )
        return settings, result, witness, guidance

    objective_tamper = json.loads(json.dumps(original_result))
    objective_tamper["numerical_objective"] = 123_456.0
    settings, result, witness, guidance = rebound(objective_tamper)
    with pytest.raises(ValueError, match="objective disagrees"):
        schema._validate_relaxation_graph(
            result=result,
            witness=witness,
            guidance=guidance,
            settings=settings,
            exogenous_episode_sha256=SHA_D,
            canonical_episode_verified=True,
        )

    gap_tamper = json.loads(json.dumps(original_result))
    gap_witness = gap_tamper["witness"]
    gap_witness["deliveries"][0]["requested_kg"] = 20.0
    gap_tamper.update(
        {
            "requested_kg": 20.0,
            "dual_certified_delivery_upper_bound_kg": 10.0,
            "relaxed_fill_ceiling": 0.5,
            "primal_dual_gap_kg": 10.0,
        }
    )
    gap_tamper["certificate"].update(
        {
            "minimization_lower_bound_exact": "-10/1",
            "uncapped_delivery_upper_bound_exact": "10/1",
            "requested_kg_exact": "20/1",
            "certified_delivery_upper_bound_exact": "10/1",
            "dual_certified_delivery_upper_bound_kg": 10.0,
        }
    )
    settings, result, witness, guidance = rebound(gap_tamper)
    with pytest.raises(ValueError, match="gap exceeds"):
        schema._validate_relaxation_graph(
            result=result,
            witness=witness,
            guidance=guidance,
            settings=settings,
            exogenous_episode_sha256=SHA_D,
            canonical_episode_verified=True,
        )


def test_noncanonical_relaxation_allows_minimal_synthetic_payloads_only() -> None:
    result = canonical_payload(
        "planner-relaxation-result-v1",
        {
            "dual_certified_delivery_upper_bound_kg": 1.0,
            "solve_settings": {"optimality_tolerance_kg": 1e-5},
        },
    )
    witness = canonical_payload("planner-relaxation-witness-v1", {})
    guidance = canonical_payload("planner-relaxation-guidance-v1", {})
    settings = _settings(
        result_sha=result.sha256,
        witness_sha=witness.sha256,
        guidance_sha=guidance.sha256,
    )
    schema._validate_relaxation_graph(
        result=result,
        witness=witness,
        guidance=guidance,
        settings=settings,
        exogenous_episode_sha256=SHA_D,
        canonical_episode_verified=False,
    )
    with pytest.raises(ValueError, match="field set"):
        schema._validate_relaxation_graph(
            result=result,
            witness=witness,
            guidance=guidance,
            settings=settings,
            exogenous_episode_sha256=SHA_D,
            canonical_episode_verified=True,
        )


def test_failed_core_cannot_retain_a0_without_settings_and_bound_identities() -> None:
    core = _successful_core()
    failed_search = _changed_search(core.search, termination_reason="invalid_input")
    failure = PlannerFailureV1(
        phase="input_validation",
        code="invalid_input",
        message="invalid",
        completed_macro_depths=failed_search.completed_macro_depths,
        attempted_edges=failed_search.attempted_edges,
        completed_edges=failed_search.completed_edges,
        coordinate_trials_completed=failed_search.coordinate_trials_completed,
        elapsed_ns=1,
        observed_resource_value=None,
    )
    with pytest.raises(ValueError, match="retained A0 evidence requires settings"):
        dataclasses.replace(
            core,
            success=False,
            failure=failure,
            settings=None,
            search=failed_search,
            searched_finalist=None,
            selected_role=None,
            comparison=None,
            deterministic_result_sha256="",
        )


def test_core_digest_excludes_only_declared_dynamic_failure_paths() -> None:
    def failed(message: str, elapsed: int, rss: int) -> FeasiblePlannerResultV1:
        failure = PlannerFailureV1(
            phase="input_validation",
            code="invalid_input",
            message=message,
            completed_macro_depths=0,
            attempted_edges=0,
            completed_edges=0,
            coordinate_trials_completed=0,
            elapsed_ns=elapsed,
            observed_resource_value=rss,
        )
        search = build_search_diagnostics(
            completed_macro_depths=0,
            attempted_edges=0,
            completed_edges=0,
            generated_candidates=0,
            a0_cursor_validations=0,
            a0_injections=0,
            exact_duplicates=0,
            frontier_retained_total=0,
            peak_frontier=0,
            coordinate_trials_attempted=0,
            coordinate_trials_completed=0,
            coordinate_trials_accepted=0,
            a0_generation_decisions=0,
            a0_evidence_replay_decisions=0,
            fresh_finalist_replay_decisions=0,
            max_live_environments_observed=0,
            termination_reason="invalid_input",
            elapsed_ns=elapsed,
            peak_rss_bytes=rss,
        )
        return FeasiblePlannerResultV1(
            schema_version=1,
            planner_kind="best-known-feasible-future-information-v1",
            evidence_scope="noncanonical_unit_fixture",
            privileged_information="complete private episode truth and commitments",
            optimality_claim="none",
            evidence_ready=False,
            provenance_gaps=(
                "clean start/end source and dependency-lock audit required",
            ),
            success=False,
            failure=failure,
            seed=None,
            canonical_episode_verified=False,
            manifest_sha256=None,
            exogenous_episode_sha256=None,
            checkpoint_contract_digest=None,
            relaxation_model_sha256=None,
            relaxation_result_sha256=None,
            relaxation_result=None,
            relaxation_witness=None,
            relaxation_guidance=None,
            settings=None,
            search=search,
            a0_incumbent=None,
            searched_finalist=None,
            selected_role=None,
            comparison=None,
        )

    left = failed("first", 1, 10)
    right = failed("second", 9, 99)
    assert left.deterministic_result_sha256 == right.deterministic_result_sha256
    assert left.to_dict() != right.to_dict()
    retained_change = dataclasses.replace(
        left,
        seed=1,
        deterministic_result_sha256="",
    )
    assert (
        retained_change.deterministic_result_sha256 != left.deterministic_result_sha256
    )


def test_runner_failure_is_sanitized_and_cannot_claim_success() -> None:
    failure = PlannerRunnerFailureV1.from_raw_message(
        phase="start_source",
        code="source_unavailable",
        message="no\nsource",
        elapsed_ns=1,
        child_exit_code=None,
    )
    artifact = FeasiblePlannerArtifactV1(
        schema_version=1,
        success=False,
        evidence_ready=False,
        core_result=None,
        core_result_sha256=None,
        runner_failure=failure,
        start_source=None,
        end_source=None,
        start_lock_sha256=None,
        end_lock_sha256=None,
        runtime_versions=None,
        source_boundary_claim=None,
    )
    assert artifact.runner_failure.message == "no source"
    with pytest.raises(ValueError, match="success"):
        dataclasses.replace(artifact, success=True)


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/result.json",
        r"C:\\temp\\result.json",
        "../outside",
        "a/../../outside",
        r"C:relative\\file",
        r"\\rooted\\file",
        ".",
    ],
)
def test_source_snapshot_rejects_absolute_untracked_paths(path: str) -> None:
    value = _source().value()
    assert isinstance(value, dict)
    value["untracked_files"] = [
        {"path": path, "kind": "file", "content_sha256": SHA_A, "bytes": 1}
    ]
    with pytest.raises(ValueError, match="repository-relative"):
        source_snapshot_payload(value)


def test_runner_failure_phase_code_and_progress_matrix() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        PlannerRunnerFailureV1.from_raw_message(
            phase="start_source",
            code="child_exit",
            message="bad",
            elapsed_ns=0,
            child_exit_code=1,
        )
    with pytest.raises(ValueError, match="requires child_exit_code"):
        PlannerRunnerFailureV1.from_raw_message(
            phase="child_execution",
            code="child_exit",
            message="bad",
            elapsed_ns=0,
            child_exit_code=None,
        )

    child_failure = PlannerRunnerFailureV1.from_raw_message(
        phase="child_execution",
        code="wall_time_limit",
        message="timeout",
        elapsed_ns=1,
        child_exit_code=None,
    )
    start = _source()
    runtime = _runtime()
    child_artifact = FeasiblePlannerArtifactV1(
        schema_version=1,
        success=False,
        evidence_ready=False,
        core_result=None,
        core_result_sha256=None,
        runner_failure=child_failure,
        start_source=start,
        end_source=None,
        start_lock_sha256=SHA_A,
        end_lock_sha256=None,
        runtime_versions=runtime,
        source_boundary_claim=None,
    )
    assert child_artifact.runner_failure is child_failure
    with pytest.raises(ValueError, match="cannot retain a core"):
        dataclasses.replace(
            child_artifact,
            core_result=_successful_core(),
            core_result_sha256=_successful_core().deterministic_result_sha256,
            artifact_sha256="",
        )

    core = _successful_core()
    end_failure = PlannerRunnerFailureV1.from_raw_message(
        phase="end_source",
        code="source_unavailable",
        message="end",
        elapsed_ns=1,
        child_exit_code=None,
    )
    end_artifact = FeasiblePlannerArtifactV1(
        schema_version=1,
        success=False,
        evidence_ready=False,
        core_result=core,
        core_result_sha256=core.deterministic_result_sha256,
        runner_failure=end_failure,
        start_source=start,
        end_source=None,
        start_lock_sha256=SHA_A,
        end_lock_sha256=None,
        runtime_versions=runtime,
        source_boundary_claim=None,
    )
    assert end_artifact.core_result is core

    serialization_failure = PlannerRunnerFailureV1.from_raw_message(
        phase="artifact_serialization",
        code="serialization_error",
        message="serialize",
        elapsed_ns=1,
        child_exit_code=None,
    )
    serialization_artifact = FeasiblePlannerArtifactV1(
        schema_version=1,
        success=False,
        evidence_ready=False,
        core_result=core,
        core_result_sha256=core.deterministic_result_sha256,
        runner_failure=serialization_failure,
        start_source=start,
        end_source=start,
        start_lock_sha256=SHA_A,
        end_lock_sha256=SHA_A,
        runtime_versions=runtime,
        source_boundary_claim=SOURCE_BOUNDARY_CLAIM,
    )
    assert serialization_artifact.runner_failure is serialization_failure

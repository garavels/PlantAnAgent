"""Public evidence assembly around the frozen feasible-planner operations.

The search and plant port deliberately return runtime records, not publishable
evidence.  This module is the single boundary that turns those records into the
frozen core-result graph, chooses the incumbent, validates the Part-I bracket,
and translates an operational failure into a stable diagnostic result.

It performs no source-control or filesystem I/O.  The external evidence runner
owns process isolation, source boundaries, dependency-lock identity, and atomic
artifact publication.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
import math
import platform
import resource
import time
from typing import Literal

from ._planner_actions import RelaxationBindingV1, require_attested_binding
from ._planner_port import (
    A0EvidenceV1,
    FreshReplayV1,
    PlannerPlantInputsV1,
    PlantPlannerSearchPortV1,
    capture_a0_evidence,
)
from ._planner_runtime import (
    PlannerRuntimeAdmissionV1,
    PlannerRuntimeError,
    TerminalSnapshotV1,
    _ResourceGuardV1,
)
from ._planner_search import (
    PlannerSearchCountersV1,
    PlannerSearchError,
    concrete_request_bytes,
    polish_traces,
    run_primary_beam,
)
from .planner_schema import (
    FAILURE_CODES,
    FAILURE_PHASES,
    OPTIMALITY_CLAIM,
    PLANNER_KIND,
    PRIVILEGED_INFORMATION,
    PROVENANCE_GAPS,
    FeasiblePlannerResultV1,
    PlannerFailureV1,
    ReplayEvidenceV1,
    SearchDiagnosticsV1,
    build_incumbent_comparison,
    build_search_diagnostics,
    canonical_json,
    compare_final_scores,
    replay_binding_payload,
)


ClockNs = Callable[[], int]
PeakRssRaw = Callable[[], int | float]
PlatformSystem = Callable[[], str]
FailurePhase = Literal[
    "input_validation",
    "relaxation_binding",
    "a0_generation",
    "a0_replay",
    "clone_preflight",
    "beam_search",
    "coarse_polish",
    "fine_polish",
    "finalist_replay",
    "bracket_validation",
    "result_serialization",
]


@dataclass(frozen=True, slots=True)
class _FailureDetailsV1:
    phase: str
    code: str
    message: str
    observed_resource_value: int | None


@dataclass(frozen=True, slots=True)
class _ValidatedContextV1:
    admission: PlannerRuntimeAdmissionV1
    binding: RelaxationBindingV1


def _production_peak_rss_raw() -> int | float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _finite(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}")
    return result


def _validated_context(inputs: PlannerPlantInputsV1) -> _ValidatedContextV1:
    if type(inputs) is not PlannerPlantInputsV1:
        raise TypeError("inputs must be an exact PlannerPlantInputsV1")
    inputs.validate()
    admission = inputs.admission
    admission.validate()
    binding = require_attested_binding(admission.binding)
    return _ValidatedContextV1(admission=admission, binding=binding)


def build_replay_evidence(
    *,
    role: Literal["a0_incumbent", "searched_finalist"],
    generation: TerminalSnapshotV1,
    replay: TerminalSnapshotV1,
    replay_elapsed_ns: int,
    inputs: PlannerPlantInputsV1,
) -> ReplayEvidenceV1:
    """Bind one evaluated terminal trace to its independent exact replay."""

    context = _validated_context(inputs)
    if type(generation) is not TerminalSnapshotV1:
        raise TypeError("generation must be an exact TerminalSnapshotV1")
    if type(replay) is not TerminalSnapshotV1:
        raise TypeError("replay must be an exact TerminalSnapshotV1")
    generation.validate_for_config(inputs.config)
    replay.validate_for_config(inputs.config)
    if generation.outcome.canonical_json != replay.outcome.canonical_json:
        raise ValueError("generation and replay outcome bytes differ")
    if generation.action_record_trace != replay.action_record_trace:
        raise ValueError("generation and replay action-record bytes differ")
    elapsed = _integer("replay_elapsed_ns", replay_elapsed_ns)
    settings = context.admission.settings
    replay_binding = replay_binding_payload(
        role=role,
        settings_sha256=settings.settings_sha256,
        relaxation_guidance_sha256=context.binding.guidance_sha256,
        manifest_sha256=replay.manifest_sha256,
        exogenous_episode_sha256=replay.exogenous_episode_sha256,
        checkpoint_contract_digest=replay.checkpoint_contract_digest,
        request_trace=replay.request_trace,
        source_trace=replay.source_trace,
        action_record_trace=replay.action_record_trace,
        outcome=replay.outcome,
    )
    return ReplayEvidenceV1(
        role=role,
        manifest_sha256=replay.manifest_sha256,
        exogenous_episode_sha256=replay.exogenous_episode_sha256,
        checkpoint_contract_digest=replay.checkpoint_contract_digest,
        action_records=replay.action_records,
        request_trace=replay.request_trace,
        source_trace=replay.source_trace,
        action_record_trace=replay.action_record_trace,
        decision_count=replay.decision_count,
        physics_steps=replay.physics_steps,
        terminated=replay.terminated,
        truncated=replay.truncated,
        battery_throughput_kwh=replay.battery_throughput_kwh,
        metrics=replay.metrics,
        terminal_truth=replay.terminal_truth,
        terminal_telemetry=replay.terminal_telemetry,
        delivery_audit=replay.delivery_audit,
        generation_outcome=generation.outcome,
        replay_outcome=replay.outcome,
        replay_binding=replay_binding,
        final_score=replay.final_score,
        replay_elapsed_ns=elapsed,
    )


def _relaxation_bracket(binding: RelaxationBindingV1) -> tuple[float, float]:
    value = binding.relaxation_result.value()
    if not isinstance(value, dict):
        raise ValueError("relaxation result must contain an object")
    solve_settings = value.get("solve_settings")
    if not isinstance(solve_settings, dict):
        raise ValueError("relaxation result has no solve-settings object")
    upper = _finite(
        "dual_certified_delivery_upper_bound_kg",
        value.get("dual_certified_delivery_upper_bound_kg"),
        minimum=0.0,
    )
    tolerance = _finite(
        "optimality_tolerance_kg",
        solve_settings.get("optimality_tolerance_kg"),
        minimum=0.0,
    )
    return upper, tolerance


def _counter_values(counters: PlannerSearchCountersV1) -> dict[str, int]:
    if type(counters) is not PlannerSearchCountersV1:
        raise TypeError("counters must be exact PlannerSearchCountersV1")
    return {
        item.name: _integer(
            f"counters.{item.name}",
            getattr(counters, item.name),
        )
        for item in fields(PlannerSearchCountersV1)
    }


def _diagnostics(
    counters: PlannerSearchCountersV1,
    *,
    termination_reason: str,
    elapsed_ns: int,
    peak_rss_bytes: int,
) -> SearchDiagnosticsV1:
    return build_search_diagnostics(
        **_counter_values(counters),
        termination_reason=termination_reason,
        elapsed_ns=_integer("elapsed_ns", elapsed_ns),
        peak_rss_bytes=_integer("peak_rss_bytes", peak_rss_bytes),
    )


def _result_identity(
    context: _ValidatedContextV1,
) -> dict[str, object]:
    admission = context.admission
    binding = context.binding
    return {
        "seed": admission.seed,
        "canonical_episode_verified": admission.canonical_episode_verified,
        "manifest_sha256": admission.manifest_sha256,
        "exogenous_episode_sha256": admission.exogenous_episode_sha256,
        "checkpoint_contract_digest": admission.checkpoint_contract_digest,
        "relaxation_model_sha256": binding.model_sha256,
        "relaxation_result_sha256": binding.result_sha256,
        "relaxation_result": binding.relaxation_result,
        "relaxation_witness": binding.relaxation_witness,
        "relaxation_guidance": binding.relaxation_guidance,
        "settings": admission.settings,
    }


def _success_result(
    *,
    context: _ValidatedContextV1,
    counters: PlannerSearchCountersV1,
    a0: ReplayEvidenceV1,
    searched: ReplayEvidenceV1 | None,
    selected_role: Literal["a0_incumbent", "searched_finalist"],
    elapsed_ns: int,
    peak_rss_bytes: int,
) -> FeasiblePlannerResultV1:
    selected = a0 if selected_role == "a0_incumbent" else searched
    if selected is None:
        raise ValueError("selected searched finalist is absent")
    upper_bound_kg, tolerance_kg = _relaxation_bracket(context.binding)
    comparison = build_incumbent_comparison(
        selected_role=selected_role,
        a0_score=a0.final_score,
        selected_score=selected.final_score,
        a0_request_trace=a0.request_trace,
        selected_request_trace=selected.request_trace,
        upper_bound_kg=upper_bound_kg,
        optimality_tolerance_kg=tolerance_kg,
    )
    return FeasiblePlannerResultV1(
        schema_version=1,
        planner_kind=PLANNER_KIND,
        evidence_scope=context.admission.evidence_scope,
        privileged_information=PRIVILEGED_INFORMATION,
        optimality_claim=OPTIMALITY_CLAIM,
        evidence_ready=False,
        provenance_gaps=PROVENANCE_GAPS,
        success=True,
        failure=None,
        **_result_identity(context),
        search=_diagnostics(
            counters,
            termination_reason="completed",
            elapsed_ns=elapsed_ns,
            peak_rss_bytes=peak_rss_bytes,
        ),
        a0_incumbent=a0,
        searched_finalist=searched,
        selected_role=selected_role,
        comparison=comparison,
    )


def _failure_result(
    *,
    evidence_scope: str,
    context: _ValidatedContextV1 | None,
    counters: PlannerSearchCountersV1,
    a0: ReplayEvidenceV1 | None,
    failure: _FailureDetailsV1,
    elapsed_ns: int,
    peak_rss_bytes: int,
) -> FeasiblePlannerResultV1:
    search = _diagnostics(
        counters,
        termination_reason=failure.code,
        elapsed_ns=elapsed_ns,
        peak_rss_bytes=peak_rss_bytes,
    )
    record = PlannerFailureV1.from_raw_message(
        phase=failure.phase,
        code=failure.code,
        message=failure.message,
        completed_macro_depths=search.completed_macro_depths,
        attempted_edges=search.attempted_edges,
        completed_edges=search.completed_edges,
        coordinate_trials_completed=search.coordinate_trials_completed,
        elapsed_ns=elapsed_ns,
        observed_resource_value=failure.observed_resource_value,
    )
    identities: dict[str, object]
    if context is None:
        identities = {
            "seed": None,
            "canonical_episode_verified": False,
            "manifest_sha256": None,
            "exogenous_episode_sha256": None,
            "checkpoint_contract_digest": None,
            "relaxation_model_sha256": None,
            "relaxation_result_sha256": None,
            "relaxation_result": None,
            "relaxation_witness": None,
            "relaxation_guidance": None,
            "settings": None,
        }
        a0 = None
    else:
        identities = _result_identity(context)
    return FeasiblePlannerResultV1(
        schema_version=1,
        planner_kind=PLANNER_KIND,
        evidence_scope=evidence_scope,
        privileged_information=PRIVILEGED_INFORMATION,
        optimality_claim=OPTIMALITY_CLAIM,
        evidence_ready=False,
        provenance_gaps=PROVENANCE_GAPS,
        success=False,
        failure=record,
        **identities,
        search=search,
        a0_incumbent=a0,
        searched_finalist=None,
        selected_role=None,
        comparison=None,
    )


_DEFAULT_FAILURE_CODE: dict[str, str] = {
    "input_validation": "invalid_input",
    "relaxation_binding": "relaxation_binding_mismatch",
    "a0_generation": "a0_generation_error",
    "a0_replay": "a0_replay_mismatch",
    "clone_preflight": "clone_error",
    "beam_search": "search_incomplete",
    "coarse_polish": "coordinate_error",
    "fine_polish": "coordinate_error",
    "finalist_replay": "replay_mismatch",
    "bracket_validation": "result_invariant_error",
    "result_serialization": "serialization_error",
}


def _failure_details(error: Exception, *, phase: FailurePhase) -> _FailureDetailsV1:
    if isinstance(error, PlannerRuntimeError):
        return _FailureDetailsV1(
            phase=error.phase,
            code=error.code,
            message=str(error),
            observed_resource_value=error.observed_resource_value,
        )
    error_phase = getattr(error, "phase", phase)
    if error_phase not in FAILURE_PHASES:
        error_phase = phase
    if isinstance(error, PlannerSearchError):
        error_code = error.code
    elif isinstance(error, MemoryError):
        error_code = "memory_error"
    else:
        error_code = _DEFAULT_FAILURE_CODE[error_phase]
    if error_code not in FAILURE_CODES:
        error_code = _DEFAULT_FAILURE_CODE[error_phase]
    return _FailureDetailsV1(
        phase=error_phase,
        code=error_code,
        message=str(error) or type(error).__name__,
        observed_resource_value=None,
    )


def _observations(guard: _ResourceGuardV1) -> tuple[int, int]:
    return (
        guard.elapsed_ns(phase="result_serialization"),
        guard.peak_rss_bytes,
    )


def build_unadmitted_failure_result(
    *,
    evidence_scope: Literal[
        "privileged_offline_development", "noncanonical_unit_fixture"
    ],
    phase: Literal["input_validation", "relaxation_binding"],
    code: str,
    message: object,
    elapsed_ns: int,
    peak_rss_bytes: int,
    observed_resource_value: int | None = None,
) -> FeasiblePlannerResultV1:
    """Represent a failure before a complete runtime admission existed."""

    if phase not in {"input_validation", "relaxation_binding"}:
        raise ValueError("unadmitted failures require an input/binding phase")
    if code not in FAILURE_CODES:
        raise ValueError("unadmitted failure code is unsupported")
    failure = _FailureDetailsV1(
        phase=phase,
        code=code,
        message=str(message) or type(message).__name__,
        observed_resource_value=observed_resource_value,
    )
    result = _failure_result(
        evidence_scope=evidence_scope,
        context=None,
        counters=PlannerSearchCountersV1(),
        a0=None,
        failure=failure,
        elapsed_ns=_integer("elapsed_ns", elapsed_ns),
        peak_rss_bytes=_integer("peak_rss_bytes", peak_rss_bytes),
    )
    canonical_json(result.to_dict())
    return result


def execute_feasible_planner(
    inputs: PlannerPlantInputsV1,
    *,
    start_ns: int,
    clock_ns: ClockNs = time.perf_counter_ns,
    peak_rss_raw: PeakRssRaw = _production_peak_rss_raw,
    platform_system: PlatformSystem = platform.system,
) -> FeasiblePlannerResultV1:
    """Execute one admitted cell and always return a frozen core result.

    ``start_ns`` is supplied by the worker because registered execution must take
    that sample as its first statement.  Source provenance and the external
    watchdog remain responsibilities of the parent evidence runner.
    """

    if type(inputs) is not PlannerPlantInputsV1:
        raise TypeError("inputs must be an exact PlannerPlantInputsV1")
    admission = inputs.admission
    evidence_scope = admission.evidence_scope
    if evidence_scope not in {
        "privileged_offline_development",
        "noncanonical_unit_fixture",
    }:
        raise ValueError("inputs carry an unsupported evidence scope")
    origin = _integer("start_ns", start_ns)
    counters = PlannerSearchCountersV1()
    context: _ValidatedContextV1 | None = None
    guard: _ResourceGuardV1 | None = None
    port: PlantPlannerSearchPortV1 | None = None
    a0_runtime: A0EvidenceV1 | None = None
    a0_evidence: ReplayEvidenceV1 | None = None
    searched_evidence: ReplayEvidenceV1 | None = None
    selected_role: Literal["a0_incumbent", "searched_finalist"] | None = None
    phase: FailurePhase = "input_validation"
    primary: Exception | None = None

    try:
        context = _validated_context(inputs)
        guard = _ResourceGuardV1(
            settings=context.admission.settings,
            counters=counters,
            start_ns=origin,
            clock_ns=clock_ns,
            peak_rss_raw=peak_rss_raw,
            platform_system=platform_system,
        )

        phase = "a0_generation"
        a0_runtime = capture_a0_evidence(
            inputs,
            guard=guard,
            counters=counters,
        )
        a0_evidence = build_replay_evidence(
            role="a0_incumbent",
            generation=a0_runtime.generation.terminal_snapshot,
            replay=a0_runtime.evidence_snapshot,
            replay_elapsed_ns=a0_runtime.replay_elapsed_ns,
            inputs=inputs,
        )

        phase = "clone_preflight"
        port = PlantPlannerSearchPortV1(
            inputs=inputs,
            a0=a0_runtime,
            guard=guard,
            counters=counters,
        )
        phase = "beam_search"
        beam_winner = run_primary_beam(
            port,
            a0_records=a0_runtime.generation.action_records,
            expected_decisions=context.admission.settings.expected_decisions,
            counters=counters,
        )

        phase = "coarse_polish"
        polished = polish_traces(
            port,
            a0=a0_runtime.search_trace(),
            beam_winner=beam_winner,
            expected_decisions=context.admission.settings.expected_decisions,
            counters=counters,
        )
        if concrete_request_bytes(polished.action_records) != concrete_request_bytes(
            a0_runtime.generation.action_records
        ):
            phase = "finalist_replay"
            replayed = port.replay_finalist(polished)
            if type(replayed) is not FreshReplayV1:
                raise TypeError("finalist replay returned the wrong record type")
            searched_evidence = build_replay_evidence(
                role="searched_finalist",
                generation=polished.payload,
                replay=replayed.snapshot,
                replay_elapsed_ns=replayed.elapsed_ns,
                inputs=inputs,
            )

        selected_role = (
            "searched_finalist"
            if searched_evidence is not None
            and compare_final_scores(
                searched_evidence.final_score,
                a0_evidence.final_score,
            )
            > 0
            else "a0_incumbent"
        )
    except Exception as exc:
        primary = exc
    finally:
        if port is not None:
            try:
                port.close()
            except Exception as cleanup_error:
                if primary is None:
                    primary = cleanup_error
                    phase = "result_serialization"

    result: FeasiblePlannerResultV1 | None = None
    if primary is None:
        assert context is not None
        assert guard is not None
        assert a0_evidence is not None
        assert selected_role is not None
        try:
            phase = "bracket_validation"
            elapsed, rss = _observations(guard)
            result = _success_result(
                context=context,
                counters=counters,
                a0=a0_evidence,
                searched=searched_evidence,
                selected_role=selected_role,
                elapsed_ns=elapsed,
                peak_rss_bytes=rss,
            )
            phase = "result_serialization"
            canonical_json(result.to_dict())
            elapsed, rss = guard.checkpoint(phase="result_serialization")
            result = _success_result(
                context=context,
                counters=counters,
                a0=a0_evidence,
                searched=searched_evidence,
                selected_role=selected_role,
                elapsed_ns=elapsed,
                peak_rss_bytes=rss,
            )
            canonical_json(result.to_dict())
            guard.checkpoint(phase="result_serialization")
        except Exception as exc:
            primary = exc

    if primary is None:
        assert guard is not None
        try:
            guard.close()
        except Exception as exc:
            primary = exc
            phase = "result_serialization"
        else:
            assert result is not None
            return result

    details = _failure_details(primary, phase=phase)
    elapsed_ns = 0
    peak_rss_bytes = 0
    if guard is not None:
        peak_rss_bytes = guard.peak_rss_bytes
        try:
            elapsed_ns = guard.elapsed_ns(phase=details.phase)
        except Exception:
            # The original failure remains authoritative. A broken clock cannot
            # create a second, partially ordered failure during cleanup.
            elapsed_ns = 0
    failed = _failure_result(
        evidence_scope=evidence_scope,
        context=context,
        counters=counters,
        a0=a0_evidence,
        failure=details,
        elapsed_ns=elapsed_ns,
        peak_rss_bytes=peak_rss_bytes,
    )
    canonical_json(failed.to_dict())
    if guard is not None:
        try:
            guard.checkpoint(phase="result_serialization")
        except Exception:
            pass
        try:
            guard.close()
        except Exception:
            pass
    return failed


__all__ = [
    "build_replay_evidence",
    "build_unadmitted_failure_result",
    "execute_feasible_planner",
]

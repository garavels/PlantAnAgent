from __future__ import annotations

from dataclasses import replace
import hashlib
import math

import pytest
from plant_sim import Action, PlantConfig, PlantState

from plant_backend.rl.v2 import _planner_runtime as runtime
from plant_backend.rl.v2._planner_search import PlannerSearchCountersV1
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.env import PlantEnvV2
from plant_backend.rl.v2.metrics import EpisodeMetricsV2
from plant_backend.rl.v2.planner_schema import (
    A0_CONTROLLER_SHA256,
    ActionRecordV1,
    FinalScoreV1,
    action_record_trace_payload,
    build_planner_settings,
    canonical_json,
    canonical_payload,
    outcome_payload,
    request_trace_payload,
    source_trace_payload,
)


SHA_A = "1" * 64
SHA_B = "2" * 64
DEFAULT_PLANT_CONFIG = PlantConfig()
DEFAULT_PLANT_CONFIG_SHA256 = hashlib.sha256(
    canonical_json(DEFAULT_PLANT_CONFIG.to_dict()).encode("utf-8")
).hexdigest()
NONCANONICAL_TWO_DECISIONS = EpisodeConfigV2(
    horizon_h=0.5,
    delivery_period_h=0.5,
)


def _settings():
    return build_planner_settings(
        expected_decisions=2,
        expected_physics_steps=30,
        plant_config_sha256=SHA_A,
        constraints_sha256=SHA_A,
        reward_weights_sha256=SHA_A,
        episode_config_sha256=SHA_A,
        checkpoint_contract_digest=SHA_A,
        relaxation_model_sha256=SHA_B,
        relaxation_result_sha256=SHA_B,
        relaxation_witness_sha256=SHA_B,
        relaxation_guidance_sha256=SHA_B,
    )


def _record(*, battery_share: float = 0.0) -> ActionRecordV1:
    return ActionRecordV1.from_action(
        Action(
            battery_share=battery_share,
            electrolyser_share=0.0,
            dac_share=0.0,
            methanator_share=0.0,
            battery_support_kw=0.0,
        ),
        decision_index=0,
        source_phase="a0_capture",
        source_code=-1,
        source_binding_sha256=A0_CONTROLLER_SHA256,
        controller_invalid=False,
    )


def _score() -> FinalScoreV1:
    return FinalScoreV1(
        delivered_kg=0.0,
        service_numerator=0,
        service_denominator=0,
        worst_fill=1.0,
        terminal_ch4_equivalent_inventory_kg=0.0,
        terminal_battery_energy_kwh=0.0,
        forced_stops=0,
        aborted_starts=0,
        cold_starts=0,
        curtailed_kwh=0.0,
        battery_throughput_kwh=0.0,
    )


def _terminal_snapshot(
    record: ActionRecordV1 | None = None,
) -> runtime.TerminalSnapshotV1:
    records = (_record() if record is None else record,)
    request = request_trace_payload(records)
    source = source_trace_payload(records)
    action_trace = action_record_trace_payload(records)
    metrics = canonical_payload("planner-metrics-v1", EpisodeMetricsV2().to_dict())
    state = PlantState().to_dict()
    truth = canonical_payload("planner-terminal-truth-v1", state)
    telemetry = canonical_payload("planner-terminal-telemetry-v1", state)
    audit = canonical_payload(
        "planner-delivery-audit-v1",
        {
            "frozen": True,
            "schedule": [],
            "settlements": [],
            "current": {},
            "cumulative_requested_kg": 0.0,
            "cumulative_delivered_kg": 0.0,
            "cumulative_unmet_kg": 0.0,
            "pending_count": 0,
            "next_pending": None,
        },
    )
    outcome = outcome_payload(
        manifest_sha256=SHA_A,
        exogenous_episode_sha256=SHA_B,
        checkpoint_contract_digest=SHA_A,
        request_trace=request,
        metrics=metrics,
        terminal_truth=truth,
        terminal_telemetry=telemetry,
        delivery_audit=audit,
        battery_throughput_kwh=0.0,
        decision_count=1,
        physics_steps=15,
        terminated=True,
        truncated=False,
    )
    return runtime.TerminalSnapshotV1(
        manifest_sha256=SHA_A,
        exogenous_episode_sha256=SHA_B,
        checkpoint_contract_digest=SHA_A,
        plant_config_sha256=DEFAULT_PLANT_CONFIG_SHA256,
        action_records=records,
        request_trace=request,
        source_trace=source,
        action_record_trace=action_trace,
        metrics=metrics,
        terminal_truth=truth,
        terminal_telemetry=telemetry,
        delivery_audit=audit,
        outcome=outcome,
        final_score=_score(),
        decision_count=1,
        physics_steps=15,
        terminated=True,
        truncated=False,
        battery_throughput_kwh=0.0,
        h2_kg_per_kg_ch4=DEFAULT_PLANT_CONFIG.methanator.h2_kg_per_kg_ch4,
        co2_kg_per_kg_ch4=DEFAULT_PLANT_CONFIG.methanator.co2_kg_per_kg_ch4,
    )


class _IntReader:
    def __init__(self, value: int) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value


def _guard(
    *,
    clock: _IntReader | None = None,
    rss: _IntReader | None = None,
    system: str = "Darwin",
) -> tuple[runtime._ResourceGuardV1, PlannerSearchCountersV1, _IntReader, _IntReader]:
    settings = _settings()
    clock_reader = _IntReader(100) if clock is None else clock
    rss_reader = _IntReader(0) if rss is None else rss
    counters = PlannerSearchCountersV1()
    guard = runtime._ResourceGuardV1(
        settings=settings,
        counters=counters,
        start_ns=100,
        clock_ns=clock_reader,
        peak_rss_raw=rss_reader,
        platform_system=lambda: system,
    )
    return guard, counters, clock_reader, rss_reader


def test_detached_terminal_and_a0_records_cross_validate_all_trace_payloads() -> None:
    record = _record()
    snapshot = _terminal_snapshot(record)
    generation = runtime.A0GenerationV1(
        action_records=(record,),
        terminal_snapshot=snapshot,
        prefix_state_bytes=(b"terminal-prefix",),
    )

    assert generation.action_records == (record,)
    assert generation.terminal_snapshot is snapshot
    assert generation.prefix_state_bytes == (b"terminal-prefix",)

    signed_zero = _record(battery_share=-0.0)
    with pytest.raises(ValueError, match="does not match"):
        runtime.A0GenerationV1(
            action_records=(signed_zero,),
            terminal_snapshot=snapshot,
            prefix_state_bytes=(b"terminal-prefix",),
        )
    with pytest.raises(ValueError, match="prefix count"):
        replace(generation, prefix_state_bytes=())


def test_terminal_snapshot_rejects_outcome_and_throughput_aliases() -> None:
    snapshot = _terminal_snapshot()
    with pytest.raises(ValueError, match="throughput"):
        replace(snapshot, battery_throughput_kwh=-0.0)
    with pytest.raises(ValueError, match="outcome payload"):
        replace(snapshot, outcome=canonical_payload("planner-outcome-v1", {}))


def test_terminal_snapshot_binds_conversion_ratios_to_exact_plant_config() -> None:
    snapshot = _terminal_snapshot()
    snapshot.validate_for_config(DEFAULT_PLANT_CONFIG)

    with pytest.raises(ValueError, match="conversion ratios"):
        replace(
            snapshot,
            h2_kg_per_kg_ch4=math.nextafter(
                snapshot.h2_kg_per_kg_ch4,
                math.inf,
            ),
        ).validate_for_config(DEFAULT_PLANT_CONFIG)

    changed_config = replace(
        DEFAULT_PLANT_CONFIG,
        methanator=replace(
            DEFAULT_PLANT_CONFIG.methanator,
            h2_kg_per_kg_ch4=math.nextafter(0.5, math.inf),
        ),
    )
    changed_sha256 = hashlib.sha256(
        canonical_json(changed_config.to_dict()).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="conversion ratios"):
        replace(snapshot, plant_config_sha256=changed_sha256).validate_for_config(
            changed_config
        )

    with pytest.raises(ValueError, match="configuration identity"):
        snapshot.validate_for_config(changed_config)


def test_terminal_snapshot_rejects_wrong_mutated_score_and_trace_projection() -> None:
    snapshot = _terminal_snapshot()
    with pytest.raises(ValueError, match="delivered_kg"):
        replace(snapshot, final_score=replace(_score(), delivered_kg=1.0))

    object.__setattr__(snapshot.final_score, "delivered_kg", -1.0)
    with pytest.raises(ValueError, match="delivered_kg"):
        snapshot.validate()

    clean = _terminal_snapshot()
    with pytest.raises(ValueError, match="trace payloads"):
        replace(clean, action_records=(_record(battery_share=-0.0),))


def test_a0_generation_rejects_wrong_controller_binding_even_when_self_consistent() -> (
    None
):
    wrong = replace(_record(), source_binding_sha256="f" * 64)
    snapshot = _terminal_snapshot(wrong)
    with pytest.raises(ValueError, match="exact A0 lineage"):
        runtime.A0GenerationV1(
            action_records=(wrong,),
            terminal_snapshot=snapshot,
            prefix_state_bytes=(b"terminal-prefix",),
        )

    corrupted_snapshot = _terminal_snapshot()
    object.__setattr__(corrupted_snapshot.final_score, "delivered_kg", -1.0)
    with pytest.raises(ValueError, match="delivered_kg"):
        runtime.A0GenerationV1(
            action_records=corrupted_snapshot.action_records,
            terminal_snapshot=corrupted_snapshot,
            prefix_state_bytes=(b"terminal-prefix",),
        )


def test_candidate_binds_exact_reset_environment_trace_depth_and_termination() -> None:
    env = PlantEnvV2(episode=NONCANONICAL_TWO_DECISIONS)
    env.reset(seed=7)
    candidate = runtime.PlantPlannerCandidateV1(
        env=env,
        action_records=(),
        battery_throughput_kwh=0.0,
        macro_depth=0,
        terminated=False,
        truncated=False,
    )

    assert candidate.env is env
    with pytest.raises(ValueError, match="macro depth"):
        replace(candidate, macro_depth=1)
    with pytest.raises(ValueError, match="truncated"):
        replace(candidate, truncated=True)
    candidate.battery_throughput_kwh = -0.0
    with pytest.raises(ValueError, match="creation attestation"):
        candidate.validate()

    releasable = runtime.PlantPlannerCandidateV1(
        env=env,
        action_records=(),
        battery_throughput_kwh=0.0,
        macro_depth=0,
        terminated=False,
        truncated=False,
    )
    assert releasable.mark_released() is env
    assert releasable.env is None and releasable._attestation is None
    with pytest.raises(ValueError, match="already released"):
        releasable.mark_released()


def test_candidate_cleanup_detaches_attested_env_after_validation_failure() -> None:
    env = PlantEnvV2(episode=NONCANONICAL_TWO_DECISIONS)
    env.reset(seed=7)
    candidate = runtime.PlantPlannerCandidateV1(
        env=env,
        action_records=(),
        battery_throughput_kwh=0.0,
        macro_depth=0,
        terminated=False,
        truncated=False,
    )
    candidate.battery_throughput_kwh = -0.0

    with pytest.raises(ValueError, match="creation attestation"):
        candidate.mark_released()
    assert candidate._dispose_for_cleanup() is env
    assert candidate.env is None and candidate._attestation is None
    with pytest.raises(ValueError, match="already released"):
        candidate._dispose_for_cleanup()


def test_candidate_rejects_unreset_and_counter_incoherent_environments() -> None:
    unreset = PlantEnvV2(episode=NONCANONICAL_TWO_DECISIONS)
    with pytest.raises(ValueError, match="must be reset"):
        runtime.PlantPlannerCandidateV1(
            env=unreset,
            action_records=(),
            battery_throughput_kwh=0.0,
            macro_depth=0,
            terminated=False,
            truncated=False,
        )

    reset = PlantEnvV2(episode=NONCANONICAL_TWO_DECISIONS)
    reset.reset(seed=7)
    reset.metrics.physical_steps = 1
    with pytest.raises(ValueError, match="physics counters"):
        runtime.PlantPlannerCandidateV1(
            env=reset,
            action_records=(),
            battery_throughput_kwh=0.0,
            macro_depth=0,
            terminated=False,
            truncated=False,
        )


def test_resource_limits_are_strict_and_report_exact_observed_value() -> None:
    settings = _settings()
    clock = _IntReader(100 + settings.wall_limit_ns)
    rss = _IntReader(settings.rss_limit_bytes)
    guard, _, _, _ = _guard(clock=clock, rss=rss)

    assert guard.checkpoint(phase="input_validation") == (
        settings.wall_limit_ns,
        settings.rss_limit_bytes,
    )
    clock.value += 1
    with pytest.raises(runtime.PlannerRuntimeError) as caught:
        guard.checkpoint(phase="beam_search")
    assert caught.value.phase == "beam_search"
    assert caught.value.code == "wall_time_limit"
    assert caught.value.observed_resource_value == settings.wall_limit_ns + 1

    regressed, _, regressed_clock, _ = _guard()
    regressed_clock.value = 99
    with pytest.raises(runtime.PlannerRuntimeError) as clock_error:
        regressed.checkpoint(phase="fine_polish")
    assert clock_error.value.phase == "fine_polish"
    assert clock_error.value.code == "invalid_input"


def test_resource_guard_converts_non_darwin_kib_and_rejects_unavailable_rss() -> None:
    settings = _settings()
    rss = _IntReader(settings.rss_limit_bytes // 1024)
    guard, _, _, _ = _guard(rss=rss, system="Linux")
    assert guard.checkpoint(phase="input_validation")[1] == settings.rss_limit_bytes
    rss.value += 1
    with pytest.raises(runtime.PlannerRuntimeError) as caught:
        guard.checkpoint(phase="beam_search")
    assert caught.value.code == "rss_limit"
    assert caught.value.observed_resource_value == settings.rss_limit_bytes + 1024

    def unavailable() -> int:
        raise OSError("synthetic unavailable RSS")

    broken = runtime._ResourceGuardV1(
        settings=settings,
        counters=PlannerSearchCountersV1(),
        start_ns=100,
        clock_ns=_IntReader(100),
        peak_rss_raw=unavailable,
        platform_system=lambda: "Darwin",
    )
    with pytest.raises(runtime.PlannerRuntimeError) as unavailable_error:
        broken.checkpoint(phase="input_validation")
    assert unavailable_error.value.code == "invalid_input"
    assert unavailable_error.value.observed_resource_value is None


def test_environment_capacity_is_reserved_before_construction() -> None:
    guard, counters, _, _ = _guard()
    tokens = [
        guard.reserve_environment(phase="beam_search")
        for _ in range(_settings().max_live_environments)
    ]
    assert guard.live_environments == 0
    assert guard.reserved_slots == _settings().max_live_environments
    assert counters.max_live_environments_observed == 0

    with pytest.raises(runtime.PlannerRuntimeError) as caught:
        guard.reserve_environment(phase="beam_search")
    assert caught.value.code == "live_environment_limit"
    assert caught.value.observed_resource_value == _settings().max_live_environments + 1

    for token in tokens:
        guard.cancel_reservation(token)
    guard.close()


def test_resource_guard_revalidates_and_detaches_frozen_settings() -> None:
    corrupted = _settings()
    object.__setattr__(corrupted, "max_live_environments", 999)
    with pytest.raises(ValueError, match="frozen protocol"):
        runtime._ResourceGuardV1(
            settings=corrupted,
            counters=PlannerSearchCountersV1(),
            start_ns=100,
            clock_ns=_IntReader(100),
            peak_rss_raw=_IntReader(0),
            platform_system=lambda: "Darwin",
        )

    original = _settings()
    guard = runtime._ResourceGuardV1(
        settings=original,
        counters=PlannerSearchCountersV1(),
        start_ns=100,
        clock_ns=_IntReader(100),
        peak_rss_raw=_IntReader(0),
        platform_system=lambda: "Darwin",
    )
    object.__setattr__(original, "max_live_environments", 999)
    tokens = [guard.reserve_environment(phase="beam_search") for _ in range(57)]
    with pytest.raises(runtime.PlannerRuntimeError, match="phase live limit"):
        guard.reserve_environment(phase="beam_search")
    for token in tokens:
        guard.cancel_reservation(token)
    guard.close()


def test_nonbeam_phases_enforce_the_two_environment_ceiling() -> None:
    guard, _, _, _ = _guard()
    first = guard.reserve_environment(phase="coarse_polish")
    second = guard.reserve_environment(phase="coarse_polish")

    with pytest.raises(runtime.PlannerRuntimeError) as caught:
        guard.reserve_environment(phase="coarse_polish")
    assert caught.value.phase == "coarse_polish"
    assert caught.value.code == "live_environment_limit"
    assert caught.value.observed_resource_value == 3
    with pytest.raises(runtime.PlannerRuntimeError) as mixed_phase:
        guard.reserve_environment(phase="beam_search")
    assert mixed_phase.value.code == "live_environment_limit"
    assert mixed_phase.value.observed_resource_value == 3

    guard.cancel_reservation(first)
    guard.cancel_reservation(second)
    guard.close()


def test_environment_identity_ownership_and_release_are_exact() -> None:
    guard, counters, _, _ = _guard()
    first = PlantEnvV2(episode=NONCANONICAL_TWO_DECISIONS)
    second = PlantEnvV2(episode=NONCANONICAL_TWO_DECISIONS)
    first_token = guard.reserve_environment(phase="a0_generation")
    second_token = guard.reserve_environment(phase="a0_generation")
    guard.bind_environment(first_token, first)
    assert guard.live_environments == 1
    assert counters.max_live_environments_observed == 1

    with pytest.raises(runtime.PlannerRuntimeError, match="already bound"):
        guard.bind_environment(first_token, second)
    with pytest.raises(runtime.PlannerRuntimeError, match="already live"):
        guard.bind_environment(second_token, first)

    guard.release_environment(first_token, first)
    guard.cancel_reservation(second_token)
    assert guard.live_environments == guard.reserved_slots == 0
    with pytest.raises(
        runtime.PlannerRuntimeError, match="unknown or already released"
    ):
        guard.release_environment(first_token, first)
    guard.close()


def test_foreign_tokens_and_cleanup_cannot_hide_a_success_path_leak() -> None:
    first, _, _, _ = _guard()
    second, _, _, _ = _guard()
    token = first.reserve_environment(phase="a0_generation")
    with pytest.raises(runtime.PlannerRuntimeError, match="another runtime"):
        second.cancel_reservation(token)

    with pytest.raises(runtime.PlannerRuntimeError, match="leaked 1") as caught:
        first.close()
    assert caught.value.observed_resource_value == 1
    first.close()
    with pytest.raises(runtime.PlannerRuntimeError, match="guard is closed"):
        first.reserve_environment(phase="beam_search")

    exceptional, _, _, _ = _guard()
    exceptional.reserve_environment(phase="beam_search")
    with pytest.raises(runtime.PlannerRuntimeError, match="leaked 1"):
        exceptional.close()
    assert exceptional.reserved_slots == 0


def test_failed_construction_reservation_can_be_cancelled_without_peak_inflation() -> (
    None
):
    guard, counters, _, _ = _guard()
    token = guard.reserve_environment(phase="beam_search")
    try:
        raise MemoryError("synthetic constructor failure")
    except MemoryError:
        guard.cancel_reservation(token)

    assert guard.live_environments == guard.reserved_slots == 0
    assert counters.max_live_environments_observed == 0
    guard.close()

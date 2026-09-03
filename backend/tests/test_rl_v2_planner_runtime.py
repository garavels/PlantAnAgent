from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import struct
from types import SimpleNamespace

import pytest
from plant_sim import Action, PlantConfig, PlantState, SolarArray

from plant_backend.controllers.base import Constraints
from plant_backend.rl.v2 import _planner_actions as actions
from plant_backend.rl.v2 import _planner_core as planner_core
from plant_backend.rl.v2 import _planner_port as planner_port
from plant_backend.rl.v2 import _planner_runtime as runtime
from plant_backend.rl.v2._planner_search import (
    PlannerSearchCountersV1,
    PlannerSearchError,
    polish_traces,
    run_primary_beam,
)
from plant_backend.rl.v2.config import EpisodeConfigV2, sample_episode
from plant_backend.rl.v2.contract import describe_checkpoint_contract
from plant_backend.rl.v2.env import PlantEnvV2
from plant_backend.rl.v2.metrics import EpisodeMetricsV2
from plant_backend.rl.v2.planner_schema import (
    A0_CONTROLLER,
    A0_CONTROLLER_SHA256,
    ActionRecordV1,
    CanonicalPayloadV1,
    FinalScoreV1,
    action_record_trace_payload,
    build_planner_settings,
    canonical_json,
    canonical_payload,
    outcome_payload,
    raw_canonical_sha256,
    request_trace_payload,
    source_trace_payload,
)
from plant_backend.rl.v2.reward import RewardWeightsV2


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


class _FixtureWitness:
    pass


def _runtime_fixture(
    *,
    seed: int = 7,
    episode_config: EpisodeConfigV2 = NONCANONICAL_TWO_DECISIONS,
) -> tuple[
    PlantConfig,
    EpisodeConfigV2,
    Constraints,
    RewardWeightsV2,
    runtime.PlannerRuntimeAdmissionV1,
]:
    config = PlantConfig()
    constraints = Constraints()
    weights = RewardWeightsV2()
    sampled = sample_episode(episode_config, config, seed=seed)
    solar = SolarArray(config.solar)
    start_s = sampled.scenario.start_hour * 3_600.0
    solar_available = tuple(
        solar.clear_sky_kw(start_s + step * config.dt_s)
        * sampled.weather.tape.for_interval(step)
        for step in range(sampled.physics_steps)
    )
    block_sizes = {
        "solar_to_battery_kw": sampled.physics_steps,
        "solar_to_electrolyser_kw": sampled.physics_steps,
        "solar_to_dac_kw": sampled.physics_steps,
        "solar_to_methanator_kw": sampled.physics_steps,
        "battery_to_electrolyser_kw": sampled.physics_steps,
        "curtailed_solar_kw": sampled.physics_steps,
        "h2_produced_kg": sampled.physics_steps,
        "co2_captured_kg": sampled.physics_steps,
        "ch4_produced_kg": sampled.physics_steps,
        "battery_energy_kwh": sampled.physics_steps + 1,
        "h2_inventory_kg": sampled.physics_steps + 1,
        "co2_inventory_kg": sampled.physics_steps + 1,
        "ch4_inventory_kg": sampled.physics_steps + 1,
        "delivered_ch4_kg": 1,
    }
    blocks = []
    start = 0
    for name, size in block_sizes.items():
        blocks.append(SimpleNamespace(name=name, start=start, size=size))
        start += size
    witness = _FixtureWitness()
    for name, size in block_sizes.items():
        if name != "delivered_ch4_kg":
            setattr(witness, name, (0.0,) * size)
    witness.normalized_variables = (0.0,) * start
    witness.deliveries = ()

    def witness_to_dict() -> dict[str, object]:
        return {
            "normalized_variables": list(witness.normalized_variables),
            **{
                name: list(getattr(witness, name))
                for name in block_sizes
                if name != "delivered_ch4_kg"
            },
            "deliveries": [],
            "delivered_kg": 0.0,
            "residuals": {"fixture": 0.0},
        }

    witness.to_dict = witness_to_dict
    solar_sha256 = hashlib.sha256()
    solar_sha256.update(b"plantanagent-v2-solar-kw-ieee754-v1\0")
    solar_sha256.update(len(solar_available).to_bytes(8, "big"))
    for value in solar_available:
        solar_sha256.update(struct.pack(">d", value))
    model = SimpleNamespace(
        schema_version=1,
        model_kind="v2-perfect-truth-physics-step-delivery-relaxation",
        model_sha256=SHA_A,
        preserve_initial_endowment=False,
        is_task_upper_bound=True,
        canonical_episode_verified=False,
        checkpoint_contract_digest=describe_checkpoint_contract(
            plant=config,
            episode=episode_config,
            constraints=constraints,
            weights=weights,
        )["checkpoint_contract_digest"],
        episode_manifest_json=canonical_json(sampled.audit_manifest()),
        exogenous_episode_sha256=sampled.exogenous_episode_sha256,
        physics_steps=sampled.physics_steps,
        dt_h=config.dt_h,
        solar_truth_sha256=solar_sha256.hexdigest(),
        solar_available_kw=solar_available,
        blocks=tuple(blocks),
        program=SimpleNamespace(variable_count=start),
        commitments=(),
    )
    certificate = SimpleNamespace(dual_certified_delivery_upper_bound_kg=0.0)
    solve_settings = SimpleNamespace(
        to_dict=lambda: {
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
    )
    result = SimpleNamespace(
        schema_version=1,
        success=True,
        status=0,
        failure_code=None,
        model_sha256=model.model_sha256,
        preserve_initial_endowment=False,
        is_task_upper_bound=True,
        canonical_episode_verified=False,
        checkpoint_contract_digest=model.checkpoint_contract_digest,
        exogenous_episode_sha256=model.exogenous_episode_sha256,
        witness=witness,
        certificate=certificate,
        dual_certified_delivery_upper_bound_kg=0.0,
        solver="scipy.optimize.linprog(method='highs')",
        solver_version="1.14.0-fixture",
        solve_settings=solve_settings,
    )

    def result_to_dict() -> dict[str, object]:
        return {
            "schema_version": result.schema_version,
            "success": result.success,
            "status": result.status,
            "failure_code": result.failure_code,
            "model_sha256": result.model_sha256,
            "checkpoint_contract_digest": result.checkpoint_contract_digest,
            "exogenous_episode_sha256": result.exogenous_episode_sha256,
            "canonical_episode_verified": result.canonical_episode_verified,
            "preserve_initial_endowment": result.preserve_initial_endowment,
            "is_task_upper_bound": result.is_task_upper_bound,
            "dual_certified_delivery_upper_bound_kg": 0.0,
            "solver": result.solver,
            "solver_version": result.solver_version,
            "solve_settings": result.solve_settings.to_dict(),
            "certificate": {"dual_certified_delivery_upper_bound_kg": 0.0},
            "witness": result.witness.to_dict(),
            "elapsed_s": 0.0,
        }

    result.to_dict = result_to_dict
    binding = actions.bind_relaxation_guidance(
        model,
        result,
        allow_noncanonical_fixture=True,
    )
    settings = build_planner_settings(
        expected_decisions=episode_config.decision_steps(config),
        expected_physics_steps=episode_config.physics_steps(config),
        plant_config_sha256=raw_canonical_sha256(config.to_dict()),
        constraints_sha256=raw_canonical_sha256(constraints.to_dict()),
        reward_weights_sha256=raw_canonical_sha256(weights.to_dict()),
        episode_config_sha256=raw_canonical_sha256(episode_config.to_dict()),
        checkpoint_contract_digest=model.checkpoint_contract_digest,
        relaxation_model_sha256=binding.model_sha256,
        relaxation_result_sha256=binding.result_sha256,
        relaxation_witness_sha256=binding.witness_sha256,
        relaxation_guidance_sha256=binding.guidance_sha256,
    )
    admission = runtime.admit_planner_runtime(
        config=config,
        episode_config=episode_config,
        constraints=constraints,
        weights=weights,
        settings=settings,
        binding=binding,
        seed=seed,
        evidence_scope="noncanonical_unit_fixture",
        canonical_episode_verified=False,
    )
    return config, episode_config, constraints, weights, admission


def _record(*, battery_share: float = 0.0, decision_index: int = 0) -> ActionRecordV1:
    return ActionRecordV1.from_action(
        Action(
            battery_share=battery_share,
            electrolyser_share=0.0,
            dac_share=0.0,
            methanator_share=0.0,
            battery_support_kw=0.0,
        ),
        decision_index=decision_index,
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


def test_runtime_admission_regenerates_identity_before_any_plant_step() -> None:
    config, episode, constraints, weights, admission = _runtime_fixture()
    env = PlantEnvV2(
        config=config,
        episode=episode,
        constraints=constraints,
        weights=weights,
    )
    env.reset(seed=admission.seed)
    runtime._validate_reset_environment(env, admission=admission)
    assert env.decisions_taken == env.physics_steps_taken == 0
    assert env.audit_manifest()["manifest_sha256"] == admission.manifest_sha256
    assert runtime._binary64_equal(True, 1.0) is False
    assert runtime._binary64_equal(1.0, 1.0) is True

    wrong_seed = PlantEnvV2(
        config=config,
        episode=episode,
        constraints=constraints,
        weights=weights,
    )
    wrong_seed.reset(seed=admission.seed + 1)
    with pytest.raises(ValueError, match="seed does not match"):
        runtime._validate_reset_environment(wrong_seed, admission=admission)

    stale_checkpoint = PlantEnvV2(
        config=config,
        episode=episode,
        constraints=constraints,
        weights=weights,
    )
    stale_checkpoint.reset(seed=admission.seed)
    stale_checkpoint._checkpoint_contract["objective"]["version"] = 999
    with pytest.raises(ValueError, match="checkpoint identity"):
        runtime._validate_reset_environment(stale_checkpoint, admission=admission)


def test_runtime_admission_rejects_rehashed_manifest_and_live_reset_mutation() -> None:
    config, episode, constraints, weights, admission = _runtime_fixture()
    changed_manifest = json.loads(admission.manifest_json)
    changed_manifest["objective_version"] = 999
    changed_json = canonical_json(changed_manifest)
    object.__setattr__(admission, "manifest_json", changed_json)
    object.__setattr__(
        admission,
        "manifest_sha256",
        hashlib.sha256(changed_json.encode("utf-8")).hexdigest(),
    )
    with pytest.raises(ValueError, match="not cross-bound"):
        admission.validate()

    _, _, _, _, clean_admission = _runtime_fixture()
    env = PlantEnvV2(
        config=config,
        episode=episode,
        constraints=constraints,
        weights=weights,
    )
    env.reset(seed=clean_admission.seed)
    env.metrics.initial_h2_kg = math.nextafter(
        env.metrics.initial_h2_kg,
        math.inf,
    )
    with pytest.raises(ValueError, match="metrics are not at their initial values"):
        runtime._validate_reset_environment(env, admission=clean_admission)


def test_exact_record_step_terminal_snapshot_and_rank_use_live_truth() -> None:
    config, episode, constraints, weights, admission = _runtime_fixture()
    env = PlantEnvV2(
        config=config,
        episode=episode,
        constraints=constraints,
        weights=weights,
    )
    env.reset(seed=admission.seed)
    runtime._validate_reset_environment(env, admission=admission)

    candidate = runtime.PlantPlannerCandidateV1(
        env=env,
        action_records=(),
        battery_throughput_kwh=0.0,
        macro_depth=0,
        terminated=False,
        truncated=False,
    )
    rank = runtime.extract_intermediate_rank(candidate)
    state = env.simulator.state
    pending = env.simulator.delivery_ledger.next_pending
    assert pending is not None
    feedstock = min(
        state.h2_level_kg / config.methanator.h2_kg_per_kg_ch4,
        state.co2_level_kg / config.methanator.co2_kg_per_kg_ch4,
    )
    conversion = (
        constraints.methanator_limit_kw(config)
        * (pending.due_step - env.simulator.step_index)
        * config.dt_h
        / config.methanator.specific_energy_kwh_per_kg
    )
    readiness = min(
        pending.requested_kg,
        state.ch4_level_kg + min(feedstock, conversion),
    )
    assert rank.delivered_plus_deadline_readiness_kg == readiness
    assert rank.feedstock_potential_kg == feedstock
    assert rank.commitment_mode == env.simulator.electrolyser_commitment.mode.value

    records = (_record(), _record(decision_index=1))
    throughput, terminated, truncated = runtime.step_action_record(
        env,
        records[0],
        battery_throughput_kwh=0.0,
    )
    assert throughput == 0.0
    assert terminated is truncated is False
    with pytest.raises(ValueError, match="decision index"):
        runtime.step_action_record(
            env,
            records[0],
            battery_throughput_kwh=throughput,
        )
    throughput, terminated, truncated = runtime.step_action_record(
        env,
        records[1],
        battery_throughput_kwh=throughput,
    )
    assert terminated is True and truncated is False
    snapshot = runtime.capture_terminal_snapshot(
        env,
        admission=admission,
        action_records=records,
        battery_throughput_kwh=throughput,
        terminated=terminated,
        truncated=truncated,
    )
    snapshot.validate_for_config(config)
    assert snapshot.action_records == records
    assert snapshot.decision_count == 2
    assert snapshot.physics_steps == 30
    assert snapshot.manifest_sha256 == admission.manifest_sha256
    with pytest.raises(ValueError, match="after planner termination"):
        runtime.step_action_record(
            env,
            ActionRecordV1.from_action(
                Action.idle(),
                decision_index=2,
                source_phase="a0_capture",
                source_code=-1,
                source_binding_sha256=A0_CONTROLLER_SHA256,
            ),
            battery_throughput_kwh=throughput,
        )


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


def test_runtime_records_and_payloads_require_exact_evidence_types() -> None:
    class ActionRecordAlias(ActionRecordV1):
        pass

    class PayloadAlias(CanonicalPayloadV1):
        def value(self) -> object:
            return {"forged": True}

    snapshot = _terminal_snapshot()
    alias_record = ActionRecordAlias(**snapshot.action_records[0].to_dict())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact ActionRecordV1"):
        replace(snapshot, action_records=(alias_record,))

    alias_payload = PayloadAlias(
        snapshot.metrics.domain,
        snapshot.metrics.canonical_json,
        snapshot.metrics.sha256,
    )
    with pytest.raises(TypeError, match="exact CanonicalPayloadV1"):
        replace(snapshot, metrics=alias_payload)


def test_port_rejects_semantically_wrong_source_binding_before_replay() -> None:
    _, _, _, _, admission = _runtime_fixture()
    wrong = ActionRecordV1.from_action(
        Action.idle(),
        decision_index=0,
        source_phase="beam",
        source_code=2,
        source_binding_sha256=admission.binding.guidance_sha256,
    )
    with pytest.raises(ValueError, match="wrong semantic source binding"):
        planner_port._validated_records((wrong,), admission=admission)


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


def test_external_candidate_registry_cleans_original_lease_after_corruption() -> None:
    env = PlantEnvV2(episode=NONCANONICAL_TWO_DECISIONS)
    env.reset(seed=7)
    guard, _, _, _ = _guard()
    registry = runtime.PlannerCandidateRegistryV1(guard)
    token = guard.reserve_environment(phase="beam_search")
    guard.bind_environment(token, env)
    candidate = runtime.PlantPlannerCandidateV1(
        env=env,
        action_records=(),
        battery_throughput_kwh=0.0,
        macro_depth=0,
        terminated=False,
        truncated=False,
    )
    registry.register(token, candidate)
    assert registry.validate_candidate(candidate) is env

    swapped = PlantEnvV2(episode=NONCANONICAL_TWO_DECISIONS)
    candidate.env = swapped
    candidate.battery_throughput_kwh = -0.0
    candidate._attestation = None
    with pytest.raises(ValueError):
        registry.validate_candidate(candidate)
    registry.release(candidate)

    assert candidate.env is None and candidate._attestation is None
    assert registry.live_candidates == 0
    assert guard.live_environments == guard.reserved_slots == 0
    guard.close()


def test_exact_a0_port_beam_polish_and_fresh_replay_integrate_on_tiny_plant() -> None:
    config, episode, constraints, weights, admission = _runtime_fixture()
    counters = PlannerSearchCountersV1()
    guard = runtime._ResourceGuardV1(
        settings=admission.settings,
        counters=counters,
        start_ns=100,
        clock_ns=lambda: 100,
        peak_rss_raw=lambda: 0,
        platform_system=lambda: "Darwin",
    )
    inputs = planner_port.PlannerPlantInputsV1(
        config=config,
        episode_config=episode,
        constraints=constraints,
        weights=weights,
        admission=admission,
    )
    a0 = planner_port.capture_a0_evidence(
        inputs,
        guard=guard,
        counters=counters,
    )
    a0.validate_for_config(config)
    assert counters.a0_generation_decisions == 2
    assert counters.a0_evidence_replay_decisions == 2
    assert a0.generation.prefix_state_bytes == a0.replay_prefix_state_bytes

    port = planner_port.PlantPlannerSearchPortV1(
        inputs=inputs,
        a0=a0,
        guard=guard,
        counters=counters,
    )
    try:
        beam = run_primary_beam(
            port,
            a0_records=a0.generation.action_records,
            expected_decisions=admission.settings.expected_decisions,
            counters=counters,
        )
        polished = polish_traces(
            port,
            a0=a0.search_trace(),
            beam_winner=beam,
            expected_decisions=admission.settings.expected_decisions,
            counters=counters,
        )
        fresh_replay = port.replay_finalist(polished)
        fresh_replay.validate_for_config(config)
        assert fresh_replay.snapshot.request_trace == polished.payload.request_trace
        assert fresh_replay.snapshot.metrics == polished.payload.metrics
        assert counters.attempted_edges == 15
        assert counters.completed_edges == counters.generated_candidates == 15
        assert counters.completed_macro_depths == 1
        assert counters.a0_cursor_validations == 1
        assert counters.a0_injections == 0
        assert counters.coordinate_trials_attempted in (30, 45)
        assert (
            counters.coordinate_trials_completed == counters.coordinate_trials_attempted
        )
        assert counters.fresh_finalist_replay_decisions == 2
    finally:
        port.close()
        guard.close()
    assert guard.live_environments == guard.reserved_slots == 0


def test_core_assembles_replay_evidence_selection_and_bracket() -> None:
    config, episode, constraints, weights, admission = _runtime_fixture()
    inputs = planner_port.PlannerPlantInputsV1(
        config=config,
        episode_config=episode,
        constraints=constraints,
        weights=weights,
        admission=admission,
    )

    result = planner_core.execute_feasible_planner(
        inputs,
        start_ns=100,
        clock_ns=lambda: 100,
        peak_rss_raw=lambda: 0,
        platform_system=lambda: "Darwin",
    )

    assert result.success is True and result.failure is None
    assert result.a0_incumbent is not None
    assert result.a0_incumbent.generation_outcome.canonical_json == (
        result.a0_incumbent.replay_outcome.canonical_json
    )
    assert result.selected_role in {"a0_incumbent", "searched_finalist"}
    assert result.comparison is not None
    assert result.comparison.selected_role == result.selected_role
    assert result.search.termination_reason == "completed"
    assert result.search.fresh_finalist_replay_decisions == (
        2 if result.searched_finalist is not None else 0
    )
    assert result.deterministic_result_sha256


def test_core_translates_search_failure_and_retains_only_verified_a0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, episode, constraints, weights, admission = _runtime_fixture()
    inputs = planner_port.PlannerPlantInputsV1(
        config=config,
        episode_config=episode,
        constraints=constraints,
        weights=weights,
        admission=admission,
    )

    def fail_beam(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise PlannerSearchError("search_incomplete", "injected beam failure")

    monkeypatch.setattr(planner_core, "run_primary_beam", fail_beam)
    result = planner_core.execute_feasible_planner(
        inputs,
        start_ns=100,
        clock_ns=lambda: 100,
        peak_rss_raw=lambda: 0,
        platform_system=lambda: "Darwin",
    )

    assert result.success is False
    assert result.failure is not None
    assert result.failure.phase == "beam_search"
    assert result.failure.code == "search_incomplete"
    assert result.failure.message == "injected beam failure"
    assert result.a0_incumbent is not None
    assert result.searched_finalist is None
    assert result.selected_role is None and result.comparison is None
    assert result.search.a0_generation_decisions == 2
    assert result.search.a0_evidence_replay_decisions == 2


def test_core_input_failure_publishes_no_unvalidated_identity() -> None:
    config, episode, constraints, weights, admission = _runtime_fixture()
    inputs = planner_port.PlannerPlantInputsV1(
        config=config,
        episode_config=episode,
        constraints=constraints,
        weights=weights,
        admission=admission,
    )
    object.__setattr__(inputs, "config", replace(config, dt_s=30.0))

    result = planner_core.execute_feasible_planner(
        inputs,
        start_ns=100,
        clock_ns=lambda: 100,
        peak_rss_raw=lambda: 0,
        platform_system=lambda: "Darwin",
    )

    assert result.success is False
    assert result.failure is not None
    assert (result.failure.phase, result.failure.code) == (
        "input_validation",
        "invalid_input",
    )
    assert result.settings is None
    assert result.manifest_sha256 is None
    assert result.a0_incumbent is None
    assert result.search.max_live_environments_observed == 0


def test_core_discards_partial_a0_on_generation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, episode, constraints, weights, admission = _runtime_fixture()
    inputs = planner_port.PlannerPlantInputsV1(
        config=config,
        episode_config=episode,
        constraints=constraints,
        weights=weights,
        admission=admission,
    )

    def fail_a0(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise runtime.PlannerRuntimeError(
            phase="a0_generation",
            code="a0_generation_error",
            message="injected A0 failure",
        )

    monkeypatch.setattr(planner_core, "capture_a0_evidence", fail_a0)
    result = planner_core.execute_feasible_planner(
        inputs,
        start_ns=100,
        clock_ns=lambda: 100,
        peak_rss_raw=lambda: 0,
        platform_system=lambda: "Darwin",
    )

    assert result.success is False
    assert result.failure is not None
    assert result.failure.phase == "a0_generation"
    assert result.failure.code == "a0_generation_error"
    assert result.settings == admission.settings
    assert result.a0_incumbent is None


def test_a0_cursor_injection_and_coordinate_fixed_regions_use_exact_states() -> None:
    five_decisions = EpisodeConfigV2(horizon_h=1.25, delivery_period_h=1.25)
    config, episode, constraints, weights, admission = _runtime_fixture(
        episode_config=five_decisions
    )
    counters = PlannerSearchCountersV1()
    guard = runtime._ResourceGuardV1(
        settings=admission.settings,
        counters=counters,
        start_ns=100,
        clock_ns=lambda: 100,
        peak_rss_raw=lambda: 0,
        platform_system=lambda: "Darwin",
    )
    inputs = planner_port.PlannerPlantInputsV1(
        config=config,
        episode_config=episode,
        constraints=constraints,
        weights=weights,
        admission=admission,
    )
    a0 = planner_port.capture_a0_evidence(
        inputs,
        guard=guard,
        counters=counters,
    )
    port = planner_port.PlantPlannerSearchPortV1(
        inputs=inputs,
        a0=a0,
        guard=guard,
        counters=counters,
    )
    try:
        root = port.initial_candidate()
        injected = port.advance_a0_cursor(
            macro_depth=1,
            start_decision=0,
            stop_decision=4,
            terminal=False,
        )
        assert injected is not None
        assert port.state_bytes(injected) == a0.replay_prefix_state_bytes[0]
        assert port.action_records(injected) == a0.generation.action_records[:4]
        port.release(injected)
        port.release(root)
        assert (
            port.advance_a0_cursor(
                macro_depth=2,
                start_decision=4,
                stop_decision=5,
                terminal=True,
            )
            is None
        )

        suffix_fixed = port.coordinate_trial(
            a0.search_trace(),
            start_decision=0,
            stop_decision=4,
            template_id=2,
            source_phase="fine_polish",
        )
        port.resource_checkpoint("completed_coordinate_trial")
        assert suffix_fixed.action_records[4:] == a0.generation.action_records[4:]
        assert all(
            record.source_phase == "fine_polish" and record.source_code == 2
            for record in suffix_fixed.action_records[:4]
        )
        prefix_fixed = port.coordinate_trial(
            a0.search_trace(),
            start_decision=4,
            stop_decision=5,
            template_id=2,
            source_phase="fine_polish",
        )
        port.resource_checkpoint("completed_coordinate_trial")
        assert prefix_fixed.action_records[:4] == a0.generation.action_records[:4]
        assert all(
            record.source_phase == "fine_polish" and record.source_code == 2
            for record in prefix_fixed.action_records[4:]
        )
    finally:
        port.close()
        guard.close()
    assert counters.max_live_environments_observed <= 3


def test_real_nonterminal_beam_retains_a0_and_releases_every_candidate() -> None:
    five_decisions = EpisodeConfigV2(horizon_h=1.25, delivery_period_h=1.25)
    config, episode, constraints, weights, admission = _runtime_fixture(
        episode_config=five_decisions
    )
    counters = PlannerSearchCountersV1()
    guard = runtime._ResourceGuardV1(
        settings=admission.settings,
        counters=counters,
        start_ns=100,
        clock_ns=lambda: 100,
        peak_rss_raw=lambda: 0,
        platform_system=lambda: "Darwin",
    )
    inputs = planner_port.PlannerPlantInputsV1(
        config=config,
        episode_config=episode,
        constraints=constraints,
        weights=weights,
        admission=admission,
    )
    a0 = planner_port.capture_a0_evidence(
        inputs,
        guard=guard,
        counters=counters,
    )
    port = planner_port.PlantPlannerSearchPortV1(
        inputs=inputs,
        a0=a0,
        guard=guard,
        counters=counters,
    )
    try:
        winner = run_primary_beam(
            port,
            a0_records=a0.generation.action_records,
            expected_decisions=admission.settings.expected_decisions,
            counters=counters,
        )
        winner.payload.validate_for_config(config)
        assert len(winner.action_records) == 5
        assert counters.completed_macro_depths == 2
        assert counters.a0_cursor_validations == 2
        assert counters.a0_injections == 1
        assert counters.attempted_edges > 15
        assert counters.completed_edges == counters.generated_candidates
        assert counters.max_live_environments_observed <= 57
    finally:
        port.close()
        guard.close()
    assert guard.live_environments == guard.reserved_slots == 0


def test_a0_descriptor_mutation_fails_before_step_and_releases_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, episode, constraints, weights, admission = _runtime_fixture()
    counters = PlannerSearchCountersV1()
    guard = runtime._ResourceGuardV1(
        settings=admission.settings,
        counters=counters,
        start_ns=100,
        clock_ns=lambda: 100,
        peak_rss_raw=lambda: 0,
        platform_system=lambda: "Darwin",
    )
    inputs = planner_port.PlannerPlantInputsV1(
        config=config,
        episode_config=episode,
        constraints=constraints,
        weights=weights,
        admission=admission,
    )

    class MutatingA0:
        def __init__(self) -> None:
            self.changed = False

        def reset(self) -> None:
            return None

        def to_dict(self) -> dict[str, object]:
            if self.changed:
                return {"id": "mutated-after-compute"}
            value = json.loads(A0_CONTROLLER.canonical_json)
            assert isinstance(value, dict)
            return value

        def compute_action(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            self.changed = True
            return object()

    monkeypatch.setattr(planner_port, "LegacyHeuristicControllerV2", MutatingA0)
    with pytest.raises(
        runtime.PlannerRuntimeError,
        match="descriptor disagrees",
    ) as captured:
        planner_port.capture_a0_evidence(
            inputs,
            guard=guard,
            counters=counters,
        )
    assert captured.value.phase == "a0_generation"
    assert captured.value.code == "a0_generation_error"
    assert counters.a0_generation_decisions == 0
    assert guard.live_environments == guard.reserved_slots == 0
    guard.close()


def test_a0_replay_timing_failure_keeps_the_replay_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, episode, constraints, weights, admission = _runtime_fixture()
    counters = PlannerSearchCountersV1()
    guard = runtime._ResourceGuardV1(
        settings=admission.settings,
        counters=counters,
        start_ns=100,
        clock_ns=lambda: 100,
        peak_rss_raw=lambda: 0,
        platform_system=lambda: "Darwin",
    )
    inputs = planner_port.PlannerPlantInputsV1(
        config=config,
        episode_config=episode,
        constraints=constraints,
        weights=weights,
        admission=admission,
    )

    def fail_elapsed(
        self: runtime._ResourceGuardV1,
        *,
        phase: str,
    ) -> int:
        del self
        raise runtime.PlannerRuntimeError(
            phase=phase,
            code="invalid_input",
            message="synthetic clock failure",
        )

    monkeypatch.setattr(runtime._ResourceGuardV1, "elapsed_ns", fail_elapsed)
    with pytest.raises(runtime.PlannerRuntimeError) as captured:
        planner_port.capture_a0_evidence(
            inputs,
            guard=guard,
            counters=counters,
        )

    assert captured.value.phase == "a0_replay"
    assert captured.value.code == "invalid_input"
    assert counters.a0_generation_decisions == 2
    assert counters.a0_evidence_replay_decisions == 0
    assert guard.live_environments == guard.reserved_slots == 0
    guard.close()


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

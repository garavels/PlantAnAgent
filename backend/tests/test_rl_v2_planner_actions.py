from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import math
from types import SimpleNamespace
import struct

import pytest
from plant_sim import Action, PlantConfig

from plant_backend.controllers.base import Constraints, ControlDecision
from plant_backend.rl.v2 import _planner_actions as actions
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.planner_schema import canonical_json, canonical_payload


def _solar_sha256(values: tuple[float, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b"plantanagent-v2-solar-kw-ieee754-v1\0")
    digest.update(struct.pack(">Q", len(values)))
    for value in values:
        digest.update(struct.pack(">d", value))
    return digest.hexdigest()


def _binding(
    *,
    solar: tuple[float, ...] = (5_000.0,) * 15,
    solar_to_electrolyser: tuple[float, ...] | None = None,
    battery_to_electrolyser: tuple[float, ...] | None = None,
    solar_to_dac: tuple[float, ...] | None = None,
    solar_to_methanator: tuple[float, ...] | None = None,
    dt_h: float = 1.0 / 60.0,
) -> actions.RelaxationBindingV1:
    model, result = _fake_relaxation_pair(
        solar=solar,
        solar_to_electrolyser=solar_to_electrolyser,
        battery_to_electrolyser=battery_to_electrolyser,
        solar_to_dac=solar_to_dac,
        solar_to_methanator=solar_to_methanator,
        dt_h=dt_h,
    )
    return actions.bind_relaxation_guidance(
        model, result, allow_noncanonical_fixture=True
    )


def _env(
    *,
    binding: actions.RelaxationBindingV1,
    battery_energy_kwh: float = 4_000.0,
    battery_capacity_kwh: float | None = None,
    constraints: Constraints | None = None,
    physics_steps_taken: int = 0,
    decisions_taken: int = 0,
    decision_stride: int = 15,
) -> SimpleNamespace:
    config = PlantConfig()
    return SimpleNamespace(
        config=config,
        constraints=Constraints() if constraints is None else constraints,
        episode_config=EpisodeConfigV2(),
        simulator=SimpleNamespace(
            state=SimpleNamespace(
                battery_energy_kwh=battery_energy_kwh,
                battery_capacity_kwh=(
                    config.battery.capacity_kwh
                    if battery_capacity_kwh is None
                    else battery_capacity_kwh
                ),
            )
        ),
        physics_steps_taken=physics_steps_taken,
        decisions_taken=decisions_taken,
        _episode=SimpleNamespace(decision_stride=decision_stride),
    )


def test_frozen_action_library_and_a0_payloads_are_exact() -> None:
    assert tuple(item.id for item in actions.ACTION_TEMPLATES) == tuple(range(15))
    assert [item.name for item in actions.ACTION_TEMPLATES] == [
        "legacy_live",
        "relaxation_projection",
        "idle",
        "charge",
        "warm_hold",
        "cold_start_hold",
        "h2_stable",
        "h2_full",
        "dac_full",
        "feedstocks_stable",
        "feedstocks_full",
        "methanate_full",
        "clipped_stoich_40",
        "clipped_stoich_70",
        "clipped_stoich_100",
    ]
    assert (
        actions.ACTION_LIBRARY_PAYLOAD.canonical_json
        == actions.ACTION_LIBRARY_CANONICAL_JSON
    )
    assert actions.ACTION_LIBRARY_PAYLOAD.sha256 == (
        "805bc4a6203026783a0d2d6514d25c83188d9ca4476b7b9c6c7763b6718456f2"
    )
    assert actions.A0_CONTROLLER_PAYLOAD.canonical_json == (
        actions.A0_CONTROLLER_CANONICAL_JSON
    )
    assert actions.A0_CONTROLLER_PAYLOAD.sha256 == (
        "d5405d5661b847b67a2999159391fd89fe0543f3459627e7a3a8ab25865a93d5"
    )


def test_projection_mean_uses_frozen_left_to_right_operation_order() -> None:
    binding = _binding(
        solar=(1.0, 2.0, 3.0),
        solar_to_electrolyser=(1.0, 2.0, 4.0),
        battery_to_electrolyser=(0.5, 0.25, 0.125),
    )
    electrolyser, dac, methanator = actions.projection_means(binding, 0, 3)
    assert electrolyser == 2.625
    assert electrolyser.hex() == "0x1.5000000000000p+1"
    assert dac == 0.0
    assert methanator == 0.0
    with pytest.raises(ValueError, match="non-empty"):
        actions.projection_means(binding, 2, 2)
    with pytest.raises(ValueError, match="in-range"):
        actions.projection_means(binding, 0, 4)


def test_proportional_closure_matches_frozen_one_ulp_reference() -> None:
    raw = (
        0.0,
        0.03491121386362534,
        0.7860655356893718,
        0.179023250447003,
    )
    raw_total = ((raw[0] + raw[1]) + raw[2]) + raw[3]
    assert raw_total.hex() == "0x1.0000000000001p+0"
    scale = float.fromhex("0x1.ffffffffffffdp-1")
    closed = actions.proportional_close(raw)
    assert closed == tuple(value * scale for value in raw)
    closed_total = ((closed[0] + closed[1]) + closed[2]) + closed[3]
    assert closed_total.hex() == "0x1.ffffffffffffep-1"
    with pytest.raises(ValueError, match="finite"):
        actions.proportional_close((1e308, 1e308, 0.0, 0.0))


def test_target_router_matches_frozen_binary64_reference_vector() -> None:
    binding = _binding()
    env = _env(binding=binding)
    targets = actions.target_powers(env=env, stack_fraction=0.40)
    routed = actions.future_truth_router(
        env=env,
        binding=binding,
        start=0,
        stop=15,
        electrolyser_kw=targets[0],
        dac_kw=targets[1],
        methanator_kw=targets[2],
    )
    assert tuple(
        value.hex()
        for value in (
            routed.battery_share,
            routed.electrolyser_share,
            routed.dac_share,
            routed.methanator_share,
            routed.battery_support_kw,
        )
    ) == (
        "0x1.e3ba3ba3ba3bap-1",
        "0x0.0p+0",
        "0x1.7357357357356p-5",
        "0x1.4414414414414p-7",
        "0x1.2c00000000000p+10",
    )
    assert actions.action_binary64(routed).hex() == (
        "3fee3ba3ba3ba3ba00000000000000003fa7357357357356"
        "3f844144144144144092c00000000000"
    )
    assert actions.actions_binary64_equal(routed, Action.sanitised(routed))


def test_packed_action_identity_distinguishes_one_ulp_and_signed_zero() -> None:
    positive_zero = Action.idle()
    negative_zero = Action(
        battery_share=-0.0,
        electrolyser_share=0.0,
        dac_share=0.0,
        methanator_share=0.0,
        battery_support_kw=0.0,
    )
    assert positive_zero == negative_zero
    assert not actions.actions_binary64_equal(positive_zero, negative_zero)
    assert actions.action_binary64(positive_zero)[0:8].hex() == "0000000000000000"
    assert actions.action_binary64(negative_zero)[0:8].hex() == "8000000000000000"

    base = Action(battery_share=0.25, electrolyser_share=0.25)
    one_ulp = Action(
        battery_share=math.nextafter(0.25, 1.0),
        electrolyser_share=0.25,
    )
    assert not actions.actions_binary64_equal(base, one_ulp)


def test_target_power_formula_uses_independently_resolved_active_caps() -> None:
    binding = _binding()
    env = _env(
        binding=binding,
        constraints=Constraints(
            electrolyser_max_kw=1_000.0,
            dac_max_kw=100.0,
            methanator_max_kw=10.0,
        ),
    )
    electrolyser, dac, methanator = actions.target_powers(env=env, stack_fraction=1.0)
    assert electrolyser == 1_000.0
    assert dac == 100.0
    assert methanator == 10.0


def test_router_scans_entire_half_open_hold_for_sunrise_maximum() -> None:
    binding = _binding(
        solar=(0.0, 250.0, 5_000.0, 1_000.0),
        solar_to_electrolyser=(1_000.0,) * 4,
    )
    env = _env(
        binding=binding,
        battery_energy_kwh=1_200.0,
        decision_stride=4,
    )
    action = actions.future_truth_router(
        env=env,
        binding=binding,
        start=0,
        stop=4,
        electrolyser_kw=1_000.0,
        dac_kw=0.0,
        methanator_kw=0.0,
    )
    assert action.electrolyser_share == 0.2
    assert action.battery_share == 0.8
    assert action.battery_support_kw == 0.0


def test_router_uses_degraded_boundary_capacity_and_nextafter_energy_limit() -> None:
    binding = _binding()
    env = _env(
        binding=binding,
        battery_energy_kwh=700.0,
        battery_capacity_kwh=4_000.0,
    )
    target = 1_200.0
    action = actions.future_truth_router(
        env=env,
        binding=binding,
        start=0,
        stop=15,
        electrolyser_kw=target,
        dac_kw=0.0,
        methanator_kw=0.0,
    )
    held_h = 15 * env.config.dt_h
    usable_kwh = 700.0 - env.constraints.reserve_soc(env.config) * 4_000.0
    expected = math.nextafter(
        usable_kwh * env.config.battery.discharge_efficiency / held_h,
        0.0,
    )
    assert action.battery_support_kw == expected
    assert action.battery_support_kw.hex() == expected.hex()
    assert tuple(
        value.hex()
        for value in (
            action.battery_share,
            action.electrolyser_share,
            action.dac_share,
            action.methanator_share,
            action.battery_support_kw,
        )
    ) == (
        "0x1.ac710cb295e9ep-1",
        "0x1.4e3bcd35a8588p-3",
        "0x0.0p+0",
        "0x0.0p+0",
        "0x1.7ffffffffffffp+8",
    )
    assert actions.action_binary64(action).hex() == (
        "3feac710cb295e9e3fc4e3bcd35a85880000000000000000"
        "00000000000000004077ffffffffffff"
    )
    assert action.battery_support_kw < target
    assert (
        action.battery_support_kw * held_h / env.config.battery.discharge_efficiency
        <= usable_kwh
    )


def test_relaxation_projection_is_clamped_to_each_active_cap() -> None:
    binding = _binding(
        solar=(5_000.0,) * 15,
        solar_to_electrolyser=(9_000.0,) * 15,
        solar_to_dac=(2_000.0,) * 15,
        solar_to_methanator=(500.0,) * 15,
    )
    env = _env(
        binding=binding,
        battery_energy_kwh=1_200.0,
        constraints=Constraints(
            electrolyser_max_kw=1_000.0,
            dac_max_kw=100.0,
            methanator_max_kw=10.0,
        ),
    )
    record = actions.generate_action_record(
        template_id=1,
        env=env,
        decision_index=0,
        source_phase="beam",
        binding=binding,
    )
    assert record.battery_support_kw == 0.0
    assert record.electrolyser_share == 1_000.0 / 5_000.0
    assert record.dac_share == 100.0 / 5_000.0
    assert record.methanator_share == 10.0 / 5_000.0


def test_all_templates_generate_records_with_frozen_source_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding()
    env = _env(binding=binding)

    class BlindObservation:
        forecast = object()
        blind_calls = 0

        def without_forecast(self) -> SimpleNamespace:
            self.blind_calls += 1
            return SimpleNamespace(forecast=None)

    observation = BlindObservation()
    env.controller_observation = observation
    created: list[FakeLegacy] = []

    class FakeLegacy:
        def __init__(self) -> None:
            self.reset_calls = 0
            self.compute_calls = 0
            self.descriptor_calls = 0
            created.append(self)

        def to_dict(self) -> dict[str, object]:
            self.descriptor_calls += 1
            value = json.loads(actions.A0_CONTROLLER_CANONICAL_JSON)
            assert isinstance(value, dict)
            return value

        def reset(self) -> None:
            self.reset_calls += 1

        def compute_action(self, seen: object, **kwargs: object) -> ControlDecision:
            self.compute_calls += 1
            assert getattr(seen, "forecast") is None
            assert kwargs == {
                "constraints": env.constraints,
                "config": env.config,
                "commitment_profile": env.episode_config.electrolyser_commitment,
            }
            return ControlDecision(Action(battery_share=0.25, electrolyser_share=0.75))

    monkeypatch.setattr(actions, "LegacyHeuristicControllerV2", FakeLegacy)
    records = [
        actions.generate_action_record(
            template_id=template_id,
            env=env,
            decision_index=0,
            source_phase="beam",
            binding=binding,
        )
        for template_id in range(15)
    ]
    second_legacy = actions.generate_action_record(
        template_id=0,
        env=env,
        decision_index=0,
        source_phase="coarse_polish",
        binding=binding,
    )
    assert [record.source_code for record in records] == list(range(15))
    assert records[0].source_binding_sha256 == actions.A0_CONTROLLER_SHA256
    assert records[1].source_binding_sha256 == binding.guidance_sha256
    assert all(
        record.source_binding_sha256 == actions.ACTION_LIBRARY_SHA256
        for record in records[2:]
    )
    assert all(record.controller_invalid is False for record in records)
    assert actions.action_record_to_action(records[2]) == Action.idle()
    assert actions.action_record_to_action(records[3]) == Action.all_to_battery()
    assert second_legacy.source_code == 0
    assert len(created) == 2
    assert all(item.reset_calls == 1 for item in created)
    assert all(item.compute_calls == 1 for item in created)
    assert all(item.descriptor_calls == 3 for item in created)
    assert observation.blind_calls == 2


def test_legacy_live_is_fresh_and_fails_if_descriptor_changes_after_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding()
    env = _env(binding=binding)
    env.controller_observation = SimpleNamespace(
        without_forecast=lambda: SimpleNamespace(forecast=None)
    )
    instances: list[ChangingLegacy] = []

    class ChangingLegacy:
        def __init__(self) -> None:
            self.changed = False
            self.compute_called = False
            instances.append(self)

        def to_dict(self) -> dict[str, object]:
            result = json.loads(actions.A0_CONTROLLER_CANONICAL_JSON)
            if self.changed:
                result["uses_forecast"] = True
            return result

        def reset(self) -> None:
            self.changed = True

        def compute_action(self, *_: object, **__: object) -> ControlDecision:
            self.compute_called = True
            return ControlDecision(Action.idle())

    monkeypatch.setattr(actions, "LegacyHeuristicControllerV2", ChangingLegacy)
    with pytest.raises(ValueError, match="frozen A0"):
        actions.generate_action_record(
            template_id=0,
            env=env,
            decision_index=0,
            source_phase="beam",
            binding=binding,
        )
    assert len(instances) == 1
    assert instances[0].compute_called is False


def test_legacy_live_fails_if_descriptor_changes_during_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding()
    env = _env(binding=binding)
    env.controller_observation = SimpleNamespace(
        without_forecast=lambda: SimpleNamespace(forecast=None)
    )

    class ChangingDuringCompute:
        def __init__(self) -> None:
            self.changed = False

        def to_dict(self) -> dict[str, object]:
            result = json.loads(actions.A0_CONTROLLER_CANONICAL_JSON)
            if self.changed:
                result["description"] = "changed during compute"
            return result

        def reset(self) -> None:
            pass

        def compute_action(self, *_: object, **__: object) -> ControlDecision:
            self.changed = True
            return ControlDecision(Action.idle())

    monkeypatch.setattr(actions, "LegacyHeuristicControllerV2", ChangingDuringCompute)
    with pytest.raises(ValueError, match="frozen A0"):
        actions.generate_action_record(
            template_id=0,
            env=env,
            decision_index=0,
            source_phase="fine_polish",
            binding=binding,
        )


def test_generation_uses_final_partial_half_open_hold_and_rejects_empty_hold() -> None:
    binding = _binding(
        solar=(10.0, 20.0, 30.0),
        solar_to_electrolyser=(100.0, 200.0, 300.0),
    )
    env = _env(
        binding=binding,
        physics_steps_taken=2,
        decisions_taken=4,
        decision_stride=15,
    )
    record = actions.generate_action_record(
        template_id=1,
        env=env,
        decision_index=4,
        source_phase="fine_polish",
        binding=binding,
    )
    assert record.electrolyser_share == 0.0
    assert record.battery_support_kw == 300.0
    env.physics_steps_taken = 3
    with pytest.raises(ValueError, match="empty"):
        actions.generate_action_record(
            template_id=2,
            env=env,
            decision_index=4,
            source_phase="beam",
            binding=binding,
        )


class _FakeWitness:
    pass


def _fake_relaxation_pair(
    *,
    solar: tuple[float, ...] = (12.0, 34.0),
    solar_to_electrolyser: tuple[float, ...] | None = (1.0, 2.0),
    battery_to_electrolyser: tuple[float, ...] | None = (0.5, 0.25),
    solar_to_dac: tuple[float, ...] | None = None,
    solar_to_methanator: tuple[float, ...] | None = None,
    dt_h: float = 1.0 / 60.0,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    physics_steps = len(solar)
    block_sizes = {
        "solar_to_battery_kw": physics_steps,
        "solar_to_electrolyser_kw": physics_steps,
        "solar_to_dac_kw": physics_steps,
        "solar_to_methanator_kw": physics_steps,
        "battery_to_electrolyser_kw": physics_steps,
        "curtailed_solar_kw": physics_steps,
        "h2_produced_kg": physics_steps,
        "co2_captured_kg": physics_steps,
        "ch4_produced_kg": physics_steps,
        "battery_energy_kwh": physics_steps + 1,
        "h2_inventory_kg": physics_steps + 1,
        "co2_inventory_kg": physics_steps + 1,
        "ch4_inventory_kg": physics_steps + 1,
        "delivered_ch4_kg": 1,
    }
    blocks = []
    start = 0
    for name, size in block_sizes.items():
        blocks.append(SimpleNamespace(name=name, start=start, size=size))
        start += size
    witness = _FakeWitness()
    for name, size in block_sizes.items():
        if name != "delivered_ch4_kg":
            setattr(witness, name, (0.0,) * size)
    witness.solar_to_electrolyser_kw = (
        (0.0,) * physics_steps
        if solar_to_electrolyser is None
        else solar_to_electrolyser
    )
    witness.battery_to_electrolyser_kw = (
        (0.0,) * physics_steps
        if battery_to_electrolyser is None
        else battery_to_electrolyser
    )
    witness.solar_to_dac_kw = (
        (0.0,) * physics_steps if solar_to_dac is None else solar_to_dac
    )
    witness.solar_to_methanator_kw = (
        (0.0,) * physics_steps if solar_to_methanator is None else solar_to_methanator
    )
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
    model = SimpleNamespace(
        schema_version=1,
        model_kind="v2-perfect-truth-physics-step-delivery-relaxation",
        model_sha256="a" * 64,
        preserve_initial_endowment=False,
        is_task_upper_bound=True,
        canonical_episode_verified=False,
        checkpoint_contract_digest="b" * 64,
        exogenous_episode_sha256="c" * 64,
        physics_steps=physics_steps,
        dt_h=dt_h,
        solar_truth_sha256=_solar_sha256(solar),
        solar_available_kw=solar,
        blocks=tuple(blocks),
        program=SimpleNamespace(variable_count=start),
        commitments=(),
    )
    certificate = SimpleNamespace(dual_certified_delivery_upper_bound_kg=123.0)
    solve_settings = SimpleNamespace(
        to_dict=lambda: {
            "feasibility_tolerance": 1e-7,
            "physical_tolerance": 1e-6,
            "optimality_tolerance_kg": 1e-5,
            "marginal_sign_tolerance": 1e-9,
            "solver_options": {},
            "highs_runtime_version": None,
            "provenance_gaps": [
                "git source revision and dependency-lock digest must be attached by "
                "the evaluation runner",
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
        dual_certified_delivery_upper_bound_kg=123.0,
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
            "dual_certified_delivery_upper_bound_kg": (
                result.dual_certified_delivery_upper_bound_kg
            ),
            "solver": result.solver,
            "solver_version": result.solver_version,
            "solve_settings": result.solve_settings.to_dict(),
            "certificate": {
                "dual_certified_delivery_upper_bound_kg": (
                    result.certificate.dual_certified_delivery_upper_bound_kg
                )
            },
            "witness": result.witness.to_dict(),
            "elapsed_s": 999.0,
        }

    result.to_dict = result_to_dict
    return model, result


def test_relaxation_binding_builds_exact_detached_payloads() -> None:
    model, result = _fake_relaxation_pair()
    binding = actions.bind_relaxation_guidance(
        model, result, allow_noncanonical_fixture=True
    )
    assert binding.solar_available_kw == (12.0, 34.0)
    assert binding.solar_to_electrolyser_kw == (1.0, 2.0)
    assert binding.battery_to_electrolyser_kw == (0.5, 0.25)
    result_value = binding.relaxation_result.value()
    assert "elapsed_s" not in result_value
    assert binding.relaxation_witness.value() == result.witness.to_dict()
    guidance = binding.relaxation_guidance.value()
    assert guidance == {
        "schema_version": 1,
        "model_sha256": model.model_sha256,
        "result_sha256": binding.result_sha256,
        "witness_sha256": binding.witness_sha256,
        "status": 0,
        "canonical_episode_verified": False,
        "preserve_initial_endowment": False,
        "is_task_upper_bound": True,
        "solver": "scipy.optimize.linprog(method='highs')",
        "solver_version": "1.14.0-fixture",
        "solve_settings": result.solve_settings.to_dict(),
        "checkpoint_contract_digest": model.checkpoint_contract_digest,
        "exogenous_episode_sha256": model.exogenous_episode_sha256,
        "solar_truth_sha256": model.solar_truth_sha256,
    }
    assert binding.relaxation_guidance.canonical_json == canonical_json(guidance)
    result.witness.solar_to_electrolyser_kw = (999.0, 999.0)
    assert binding.solar_to_electrolyser_kw == (1.0, 2.0)


def test_binding_factory_cross_checks_projection_arrays_against_witness_bytes() -> None:
    model, result = _fake_relaxation_pair()
    witness_payload = result.witness.to_dict()
    witness_payload["solar_to_electrolyser_kw"][0] = math.nextafter(1.0, math.inf)
    result.witness.to_dict = lambda: witness_payload
    with pytest.raises(ValueError, match="does not match its witness payload"):
        actions.bind_relaxation_guidance(model, result, allow_noncanonical_fixture=True)


def test_noncanonical_relaxation_binding_requires_explicit_fixture_admission() -> None:
    model, result = _fake_relaxation_pair()
    with pytest.raises(ValueError, match="fixture-only admission"):
        actions.bind_relaxation_guidance(model, result)
    binding = actions.bind_relaxation_guidance(
        model, result, allow_noncanonical_fixture=True
    )
    assert binding.binding_scope == actions.NONCANONICAL_BINDING_SCOPE

    model.canonical_episode_verified = True
    result.canonical_episode_verified = True
    with pytest.raises(TypeError, match="concrete upper-bound"):
        actions.bind_relaxation_guidance(model, result)
    with pytest.raises(ValueError, match="cannot use the fixture-only"):
        actions.bind_relaxation_guidance(model, result, allow_noncanonical_fixture=True)


def _change_solve_setting(result: SimpleNamespace, name: str, value: object) -> None:
    changed = result.solve_settings.to_dict()
    changed[name] = value
    result.solve_settings = SimpleNamespace(to_dict=lambda: changed)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda result: setattr(result, "solver", "scipy.optimize.linprog"),
            "frozen HiGHS method",
        ),
        (
            lambda result: _change_solve_setting(
                result, "feasibility_tolerance", math.nextafter(1e-7, math.inf)
            ),
            "feasibility_tolerance",
        ),
        (
            lambda result: _change_solve_setting(
                result, "physical_tolerance", math.nextafter(1e-6, math.inf)
            ),
            "physical_tolerance",
        ),
        (
            lambda result: _change_solve_setting(
                result, "optimality_tolerance_kg", math.nextafter(1e-5, math.inf)
            ),
            "optimality_tolerance_kg",
        ),
        (
            lambda result: _change_solve_setting(
                result, "marginal_sign_tolerance", math.nextafter(1e-9, math.inf)
            ),
            "marginal_sign_tolerance",
        ),
        (
            lambda result: _change_solve_setting(
                result, "solver_options", {"presolve": False}
            ),
            "solver_options",
        ),
        (
            lambda result: _change_solve_setting(result, "highs_runtime_version", ""),
            "highs_runtime_version",
        ),
    ],
)
def test_relaxation_binding_rejects_solve_contract_mutations(
    mutate: object, message: str
) -> None:
    model, result = _fake_relaxation_pair()
    mutate(result)  # type: ignore[operator]
    with pytest.raises((TypeError, ValueError), match=message):
        actions.bind_relaxation_guidance(model, result, allow_noncanonical_fixture=True)


def test_binding_clone_and_post_construction_mutation_fail_at_generation() -> None:
    binding = _binding()
    env = _env(binding=binding)

    direct_values = {
        name: getattr(binding, name)
        for name, field in binding.__dataclass_fields__.items()
        if field.init and name != "_factory_token"
    }
    with pytest.raises(TypeError, match="_factory_token"):
        actions.RelaxationBindingV1(**direct_values)

    cloned = copy.deepcopy(binding)
    assert cloned is not binding
    with pytest.raises(ValueError, match="validated factory"):
        actions.generate_action_record(
            template_id=2,
            env=env,
            decision_index=0,
            source_phase="beam",
            binding=cloned,
        )
    with pytest.raises(ValueError, match="InitVar"):
        replace(binding)

    object.__setattr__(binding, "solar_to_electrolyser_kw", (99.0,) * 15)
    with pytest.raises(ValueError, match="factory attestation"):
        actions.generate_action_record(
            template_id=2,
            env=env,
            decision_index=0,
            source_phase="beam",
            binding=binding,
        )


def test_binding_guidance_mutation_is_rechecked_by_router() -> None:
    binding = _binding()
    env = _env(binding=binding)
    guidance = binding.relaxation_guidance.value()
    guidance["model_sha256"] = "f" * 64
    object.__setattr__(
        binding,
        "relaxation_guidance",
        canonical_payload("planner-relaxation-guidance-v1", guidance),
    )
    with pytest.raises(ValueError, match="factory attestation"):
        actions.future_truth_router(
            env=env,
            binding=binding,
            start=0,
            stop=15,
            electrolyser_kw=0.0,
            dac_kw=0.0,
            methanator_kw=0.0,
        )


def test_binding_scalar_mutation_is_rechecked_by_projection() -> None:
    binding = _binding()
    object.__setattr__(binding, "checkpoint_contract_digest", "e" * 64)
    with pytest.raises(ValueError, match="factory attestation"):
        actions.projection_means(binding, 0, 15)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda model, result: setattr(result, "model_sha256", "d" * 64),
            "model identity",
        ),
        (lambda model, result: setattr(result, "status", 1), "status-0"),
        (
            lambda model, result: setattr(result, "preserve_initial_endowment", True),
            "terminal endowment",
        ),
        (
            lambda model, result: setattr(
                result, "checkpoint_contract_digest", "d" * 64
            ),
            "checkpoint",
        ),
        (
            lambda model, result: setattr(result, "exogenous_episode_sha256", "d" * 64),
            "episode identities",
        ),
        (
            lambda model, result: setattr(result, "canonical_episode_verified", True),
            "canonical-verification",
        ),
        (lambda model, result: setattr(result, "witness", None), "no witness"),
        (
            lambda model, result: setattr(result, "certificate", None),
            "no dual certificate",
        ),
        (
            lambda model, result: setattr(
                result.certificate, "dual_certified_delivery_upper_bound_kg", 124.0
            ),
            "dual ceilings",
        ),
        (
            lambda model, result: setattr(model, "solar_truth_sha256", "d" * 64),
            "solar truth digest",
        ),
        (
            lambda model, result: setattr(result.witness, "solar_to_dac_kw", (0.0,)),
            "length",
        ),
        (
            lambda model, result: setattr(
                result.witness, "solar_to_dac_kw", (0.0, float("nan"))
            ),
            "finite",
        ),
    ],
)
def test_relaxation_binding_rejects_mismatches(mutate: object, message: str) -> None:
    model, result = _fake_relaxation_pair()
    mutate(model, result)  # type: ignore[operator]
    with pytest.raises((TypeError, ValueError), match=message):
        actions.bind_relaxation_guidance(model, result, allow_noncanonical_fixture=True)


def test_generation_rejects_lineage_and_index_errors() -> None:
    binding = _binding()
    env = _env(binding=binding)
    with pytest.raises(ValueError, match="between 0 and 14"):
        actions.generate_action_record(
            template_id=15,
            env=env,
            decision_index=0,
            source_phase="beam",
            binding=binding,
        )
    with pytest.raises(ValueError, match="environment decision count"):
        actions.generate_action_record(
            template_id=2,
            env=env,
            decision_index=1,
            source_phase="beam",
            binding=binding,
        )
    with pytest.raises(ValueError, match="source_phase"):
        actions.generate_action_record(
            template_id=2,
            env=env,
            decision_index=0,
            source_phase="a0_capture",
            binding=binding,
        )

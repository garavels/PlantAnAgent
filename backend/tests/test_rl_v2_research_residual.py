"""Fixture-only causal cache, exact H baseline and restricted residual checks.

Seeds 9_990_700..9_990_799 are fixtures, never registered or development data.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from plant_sim import Action

from plant_backend.controllers.base import Constraints, ControlDecision
from plant_backend.rl.v2 import beta_spaces
from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.contract import contract_digest
from plant_backend.rl.v2.mpc_execution import public_observation_sha256
from plant_backend.rl.v2.setpoint_command import issue_process_setpoint_command
from plant_backend.rl.v2.setpoint_control import ProcessSetpointTargetsV1
from plant_backend.rl.v2.setpoint_controllers import (
    LegacyHeuristicSetpointControllerV1,
    LegacyHeuristicSetpointDecisionV1,
    LegacyHeuristicSetpointEvidenceV1,
)
from plant_backend.rl.v2.setpoint_env import PlantSetpointEnvV2
from plant_backend.rl.v2.training import research_residual as residual
from plant_backend.rl.v2.training.episode_allocator import (
    EpisodeSeedAllocatorV1,
    EpisodeSeedCustodyError,
)


def _base(**kwargs: Any) -> PlantCyclicMarketEnvV2Beta:
    return PlantCyclicMarketEnvV2Beta(**kwargs)


@pytest.fixture
def wrapper():
    instance = residual.ResearchResidualHWrapper(_base())
    instance.reset(seed=9_990_701)
    yield instance
    instance.close()


def _synthetic_h(
    base: PlantCyclicMarketEnvV2Beta,
    *,
    electrolyser: float = 1_800.0,
    dac: float = 400.0,
    methanator: float = 80.0,
    charge: float = 1_000.0,
    invalid: bool = False,
) -> LegacyHeuristicSetpointDecisionV1:
    """Typed synthetic H evidence for mapping only; no inferred plant outcomes."""
    targets = ProcessSetpointTargetsV1(electrolyser, dac, methanator, charge)
    observation = base.controller_observation
    source = Action(
        electrolyser_share=electrolyser / 10_000,
        dac_share=dac / 10_000,
        methanator_share=methanator / 10_000,
        battery_share=charge / 10_000,
    )
    evidence = LegacyHeuristicSetpointEvidenceV1(
        decision_index=base.decisions_taken,
        observation_sha256=public_observation_sha256(observation),
        boundary_solar_kw=10_000.0,
        source_action=source,
        source_notes=("mapping fixture; not a realized H trajectory",),
        source_invalid=invalid,
        requested_targets=targets,
        issued_targets=targets,
        battery_charge_limited=False,
    )
    command = issue_process_setpoint_command(
        targets,
        decision_index=evidence.decision_index,
        observation_sha256=evidence.observation_sha256,
        hold_duration_s=observation.decision_interval_s,
        issuer_id=LegacyHeuristicSetpointControllerV1.id,
        source_schema=evidence.schema,
        source_sha256=evidence.sha256,
        constraints=base.constraints,
        config=base.config,
    )
    return LegacyHeuristicSetpointDecisionV1(evidence, command)


def _prepare(wrapper, decision, action):
    return residual.prepare_residual_action(
        wrapper.base,
        decision,
        action,
        interface_contract_digest=wrapper.interface_contract_digest,
    )


def _without_clock(info):
    return {key: value for key, value in info.items() if key != "dispatcher_elapsed_ns"}


def test_zero_residual_is_exact_H_for_complete_168h_fixture(monkeypatch):
    direct = _base()
    wrapped_base = _base()
    wrapped = residual.ResearchResidualHWrapper(wrapped_base)
    h = LegacyHeuristicSetpointControllerV1()
    observe_calls = [0, 0]
    h_calls = 0
    for index, base in enumerate((direct, wrapped_base)):
        original = base._observe

        def observe(original=original, index=index):
            observe_calls[index] += 1
            return original()

        monkeypatch.setattr(base, "_observe", observe)
    original_h = wrapped._heuristic.compute_command

    def compute(*args, **kwargs):
        nonlocal h_calls
        h_calls += 1
        return original_h(*args, **kwargs)

    monkeypatch.setattr(wrapped._heuristic, "compute_command", compute)
    try:
        expected, direct_info = direct.reset(seed=9_990_700)
        actual, wrapped_info = wrapped.reset(seed=9_990_700)
        assert np.array_equal(actual[:135], expected)
        assert direct_info == wrapped_info
        assert observe_calls[0] == observe_calls[1]
        assert h_calls == 1
        for index in range(672):
            decision = h.compute_command(
                direct.controller_observation,
                constraints=direct.constraints,
                config=direct.config,
                commitment_profile=direct.episode_config.electrolyser_commitment,
            )
            expected, reward_h, done_h, trunc_h, info_h = direct.step_process_setpoints(
                decision.command, controller_invalid=decision.invalid
            )
            actual, reward_r, done_r, trunc_r, info_r = wrapped.step(
                np.zeros(4, dtype=np.float32)
            )
            assert np.array_equal(actual[:135], expected), index
            assert wrapped.observation_space.contains(actual)
            assert (reward_r, done_r, trunc_r) == (reward_h, done_h, trunc_h)
            # This includes every minute's execution evidence, public constraints,
            # delivery, economic EUR contributions and the daily return ledger.
            assert _without_clock(info_r) == _without_clock(info_h), index
            assert wrapped_base.simulator.state == direct.simulator.state
            assert wrapped_base.economic_metrics == direct.economic_metrics
            assert wrapped_base.cycle_settlements == direct.cycle_settlements
            assert (
                wrapped.last_residual_evidence["command"] == decision.command.to_dict()
            )
            assert (
                wrapped.last_residual_evidence["controller_invalid"] == decision.invalid
            )
            assert wrapped.last_residual_evidence["body"][
                "exact_zero_heuristic_command"
            ]
            assert observe_calls[0] == observe_calls[1]
            assert (
                wrapped_base.encoder._base._previous_solar_kw
                == direct.encoder._base._previous_solar_kw
            )
            assert h_calls == min(index + 2, 672)
        assert done_r or trunc_r
        assert wrapped_base.physics_steps_taken == direct.physics_steps_taken == 10_080
        assert wrapped_base.decisions_taken == direct.decisions_taken == 672
        assert len(wrapped_base.cycle_settlements) == 7
        assert wrapped.current_heuristic_decision is None
        assert np.array_equal(actual[-4:], np.zeros(4, dtype=np.float32))
        with pytest.raises(RuntimeError, match="reset"):
            wrapped.step(np.zeros(4, dtype=np.float32))
    finally:
        wrapped.close()
        direct.close()


def test_cache_and_helpers_use_actual_returned_vector_without_extra_observation(
    wrapper, monkeypatch
):
    base = wrapper.base
    original_h = wrapper._heuristic.compute_command
    original_step = base.step_process_setpoints
    h_calls = 0
    returned_info = None
    returned_vector = None

    def compute(*args, **kwargs):
        nonlocal h_calls
        h_calls += 1
        return original_h(*args, **kwargs)

    def step(*args, **kwargs):
        nonlocal returned_info, returned_vector
        observation, reward, terminated, truncated, returned_info = original_step(
            *args, **kwargs
        )
        # A sentinel on the returned vector detects reencoding even if encoder
        # history happened to reproduce the original vector at this boundary.
        returned_vector = observation.copy()
        returned_vector[1] = np.float32(0.1234567)
        return returned_vector, reward, terminated, truncated, returned_info

    monkeypatch.setattr(wrapper._heuristic, "compute_command", compute)
    monkeypatch.setattr(base, "step_process_setpoints", step)
    actual, _, _, _, info = wrapper.step([0, 0, 0, 0])
    assert h_calls == 1
    assert info is returned_info
    assert "research_residual" not in info
    assert wrapper.last_dispatcher_elapsed_ns == tuple(info["dispatcher_elapsed_ns"])
    assert np.array_equal(actual[:135], returned_vector)
    assert np.array_equal(wrapper.current_base_observation, returned_vector)
    previous_solar = base.encoder._base._previous_solar_kw
    state = base.simulator.state
    monkeypatch.setattr(base, "_observe", lambda: pytest.fail("extra observation"))
    monkeypatch.setattr(base, "reset", lambda **kwargs: pytest.fail("extra reset"))
    for _ in range(3):
        cached = wrapper.current_heuristic_decision
        combined = residual.extend_observation(
            wrapper.current_base_observation, base, cached
        )
        assert np.array_equal(combined, actual)
        _prepare(wrapper, cached, [0.4, -0.7, 0, 0])
        assert wrapper.current_decision_index == 1
        wrapper.current_observation[:] = 0
        wrapper.current_heuristic_features[:] = 0
    assert h_calls == 1
    assert base.simulator.state == state
    assert base.encoder._base._previous_solar_kw == previous_solar
    assert base.decisions_taken == 1 and base.physics_steps_taken == 15
    assert np.array_equal(wrapper.current_observation, actual)


@pytest.mark.parametrize(
    "h_e,r_e,expected",
    [
        (0.0, 1, 0),
        (450.0, -1, 450),
        (450.0, 1, 450),
        (1200.0, -1, 1200),
        (1200.0, 1, 1200),
        (1800.0, -1, 1500),
        (1800.0, 1, 2100),
        (3000.0, 1, 3000),
    ],
)
def test_fixed_floor_mapping_and_night_450_prior(wrapper, h_e, r_e, expected):
    command, invalid, evidence = _prepare(
        wrapper, _synthetic_h(wrapper.base, electrolyser=h_e), [r_e, -1, 1, 1]
    )
    assert command.targets.electrolyser_target_kw == expected
    assert command.targets.dac_target_kw == 200
    assert command.targets.methanator_target_kw == 120
    assert command.targets.battery_charge_ceiling_kw == 1500
    assert not invalid
    assert evidence["body"]["electrolyser_floor_kw"] == 1200


def test_nonzero_mapping_uses_active_equipment_and_physical_charge_limits(wrapper):
    wrapper.base.constraints = Constraints(
        electrolyser_max_kw=2000, dac_max_kw=450, methanator_max_kw=90
    )
    command, _, _ = _prepare(
        wrapper, _synthetic_h(wrapper.base, charge=2000), [1, 1, 1, 1]
    )
    assert command.targets == ProcessSetpointTargetsV1(2000, 450, 90, 2500)
    all_zero = _synthetic_h(wrapper.base, electrolyser=0, dac=0, methanator=0, charge=0)
    command, _, _ = _prepare(wrapper, all_zero, [-1, 1, 1, -1])
    assert command.targets == all_zero.command.targets


def test_exact_zero_keeps_original_float64_command_and_original_invalid(wrapper):
    target = 1800.123456789
    assert float(np.float32(target)) != target
    cached = _synthetic_h(wrapper.base, electrolyser=target, invalid=True)
    command, invalid, evidence = _prepare(
        wrapper, cached, np.zeros(4, dtype=np.float32)
    )
    assert command is cached.command
    assert command.targets.electrolyser_target_kw == target
    assert invalid is True
    assert evidence["command"] == cached.command.to_dict()
    assert evidence["body"]["exact_zero_heuristic_command"]
    assert not evidence["body"]["residual_clipped"]


def test_wrapper_passes_original_H_invalid_flag_to_common_execution(monkeypatch):
    wrapper = residual.ResearchResidualHWrapper(_base())
    source = ControlDecision(Action.idle(), invalid=True)
    monkeypatch.setattr(
        wrapper._heuristic._controller, "compute_action", lambda *args, **kwargs: source
    )
    try:
        wrapper.reset(seed=9_990_703)
        original = wrapper._cached.command
        captured = {}
        real_step = wrapper.base.step_process_setpoints

        def step(command, *, controller_invalid):
            captured.update(command=command, invalid=controller_invalid)
            return real_step(command, controller_invalid=controller_invalid)

        monkeypatch.setattr(wrapper.base, "step_process_setpoints", step)
        _, _, _, _, info = wrapper.step([0, 0, 0, 0])
        assert captured["command"] is original
        assert captured["invalid"] is True
        assert info["metrics"]["out_of_range_decisions"] == 1
    finally:
        wrapper.close()


def test_float32_and_numerically_equal_decoded_float64_have_identical_evidence(wrapper):
    raw32 = np.array([0.1, -0.12345679, 2.0, -0.9], dtype=np.float32)
    first = _prepare(wrapper, wrapper.current_heuristic_decision, raw32)
    second = _prepare(
        wrapper,
        wrapper.current_heuristic_decision,
        np.asarray(raw32.tolist(), dtype=np.float64),
    )
    assert first == second
    evidence = first[2]
    assert evidence["body"]["raw_residual"] == raw32.tolist()
    assert evidence["body"]["bounded_residual"][2] == 1.0
    assert evidence["controller_invalid"]
    assert evidence["body"]["residual_clipped"]
    _, _, _, _, info = wrapper.step(raw32)
    assert info["metrics"]["out_of_range_decisions"] == 1
    assert wrapper.last_residual_evidence == first[2]


@pytest.mark.parametrize(
    "action",
    [
        [0, 0, 0],
        [[0, 0, 0, 0]],
        [np.nan, 0, 0, 0],
        [0, np.inf, 0, 0],
        [True, 0, 0, 0],
        ["0", 0, 0, 0],
        [0j, 0, 0, 0],
        [[0], 0, 0, 0],
        None,
    ],
)
def test_malformed_actions_reject_before_movement_and_retain_failure_evidence(
    wrapper, action
):
    before = wrapper.base.simulator.state
    cached = wrapper.current_heuristic_decision
    with pytest.raises((TypeError, ValueError)):
        wrapper.step(action)
    evidence = wrapper.last_residual_evidence
    assert evidence["status"] == "rejected"
    assert evidence["decision_index"] == 0
    json.dumps(evidence, allow_nan=False)
    assert wrapper.current_heuristic_decision == cached
    assert wrapper.base.simulator.state == before
    assert wrapper.base.decisions_taken == wrapper.base.physics_steps_taken == 0
    assert wrapper._heuristic._decision_count == 1


def test_stale_boundary_and_changed_authority_reject_before_movement(wrapper):
    cached = wrapper.current_heuristic_decision
    wrapper.step([0, 0, 0, 0])
    with pytest.raises(ValueError, match="stale"):
        _prepare(wrapper, cached, [0, 0, 0, 0])
    wrapper.base.constraints = replace(wrapper.base.constraints, dac_max_kw=100)
    with pytest.raises(ValueError, match="authority is stale"):
        wrapper.step([0, 0, 0, 0])
    assert wrapper.base.decisions_taken == 1
    assert wrapper.last_residual_evidence["status"] == "rejected"


@pytest.mark.parametrize(
    "base_vector",
    [
        np.zeros(134, dtype=np.float32),
        np.zeros(135, dtype=np.float64),
        np.full(135, np.nan, dtype=np.float32),
        [0] * 135,
    ],
)
def test_observation_helper_rejects_wrong_base_vector(wrapper, base_vector):
    with pytest.raises(ValueError, match="135-field"):
        residual.extend_observation(
            base_vector, wrapper.base, wrapper.current_heuristic_decision
        )
    assert wrapper.base.physics_steps_taken == 0


def test_contract_binds_135_source_and_ordered_139_mapping_and_copies(wrapper):
    contract = wrapper.interface_contract
    assert contract["base_interface_contract"] == beta_spaces.contract_payload()
    assert (
        contract["base_checkpoint_contract_digest"]
        == wrapper.base.checkpoint_contract_digest
    )
    assert contract["observation_size"] == 139
    assert len(contract["observation_fields"]) == 139
    assert contract["observation_names"] == list(residual.OBSERVATION_NAMES)
    assert contract["observation_names"][:135] == list(beta_spaces.OBSERVATION_NAMES)
    assert contract["action_names"] == list(residual.ACTION_NAMES)
    assert (
        contract["residual_mapping_source_sha256"]
        == hashlib.sha256(Path(residual.__file__).read_bytes()).hexdigest()
    )
    assert len(contract["heuristic_source_sha256"]) == 4
    assert wrapper.interface_contract_digest == contract_digest(contract)
    assert wrapper.checkpoint_contract_digest != wrapper.base.checkpoint_contract_digest
    assert (
        wrapper.checkpoint_contract["interface_contract_digest"]
        == wrapper.interface_contract_digest
    )
    assert not wrapper.checkpoint_contract["registered_evaluation_eligible"]
    wrapper.require_interface_contract(deepcopy(contract))
    for key in (
        "observation_names",
        "mapping",
        "heuristic_source_sha256",
        "residual_mapping_source_sha256",
    ):
        changed = deepcopy(contract)
        changed[key] = None
        with pytest.raises(ValueError, match="contract mismatch"):
            wrapper.require_interface_contract(changed)
    with pytest.raises(ValueError, match="contract mismatch"):
        wrapper.require_interface_contract(beta_spaces.contract_payload())
    contract["observation_names"][0] = "mutated"
    assert wrapper.interface_contract["observation_names"][0] != "mutated"


@pytest.mark.parametrize(
    "wrong",
    ["alpha", "already_wrapped", "observation", "dtype", "action", "base_contract"],
)
def test_wrong_interface_is_rejected_without_reset(wrong, monkeypatch):
    base = PlantSetpointEnvV2() if wrong == "alpha" else _base()
    env = gym.Wrapper(base) if wrong == "already_wrapped" else base
    if wrong == "observation":
        base.observation_space = gym.spaces.Box(0, 1, (135,), np.float32)
    elif wrong == "dtype":
        base.observation_space = gym.spaces.Box(
            beta_spaces.observation_low().astype(np.float64),
            beta_spaces.observation_high().astype(np.float64),
            dtype=np.float64,
        )
    elif wrong == "action":
        base.action_space = gym.spaces.Box(-1, 1, (3,), np.float32)
    elif wrong == "base_contract":
        base._checkpoint_contract["interface_contract_digest"] = "0" * 64
    monkeypatch.setattr(base, "reset", lambda **kwargs: pytest.fail("unexpected reset"))
    try:
        with pytest.raises((ValueError, TypeError), match="beta"):
            residual.ResearchResidualHWrapper(env)
    finally:
        env.close()


def test_outer_custody_counts_each_reset_and_physical_step_once():
    # Four shortened fixture streams exercise actual beta physics plus allocator
    # nesting, including terminal cache behavior and the next reset wave.
    episode = EpisodeConfigV2(horizon_h=24, decision_interval_s=3600)
    allocator = EpisodeSeedAllocatorV1.fixture(
        base_seed=9_990_740, episode_decisions=24
    )
    residuals = [
        residual.ResearchResidualHWrapper(_base(episode=episode)) for _ in range(4)
    ]
    environments = [
        allocator.wrap(env, environment_index=i) for i, env in enumerate(residuals)
    ]
    try:
        for env in environments:
            assert env.reset()[0].shape == (139,)
        assert len(allocator.snapshot().ledger) == 4
        for offset in range(24):
            for env in environments:
                _, _, done, trunc, _ = env.step(np.zeros(4, dtype=np.float32))
                assert (done or trunc) == (offset == 23)
            assert [
                cursor.decision_offset for cursor in allocator.snapshot().cursors
            ] == [offset + 1] * 4
        assert all(env.base.physics_steps_taken == 1440 for env in residuals)
        assert all(env._heuristic._decision_count == 24 for env in residuals)
        assert len(allocator.snapshot().ledger) == 4
        for env in environments:
            env.reset()
        assert len(allocator.snapshot().ledger) == 8
        assert [cursor.decision_offset for cursor in allocator.snapshot().cursors] == [
            0
        ] * 4
        assert all(env._heuristic._decision_count == 1 for env in residuals)
    finally:
        for env in environments:
            env.close()


def test_outer_custody_fails_closed_on_rejected_residual():
    wrapper = residual.ResearchResidualHWrapper(_base())
    allocator = EpisodeSeedAllocatorV1.fixture(
        base_seed=9_990_760, episode_decisions=672
    )
    env = allocator.wrap(wrapper, environment_index=0)
    try:
        env.reset()
        with pytest.raises(ValueError):
            env.step([np.nan, 0, 0, 0])
        assert wrapper.base.physics_steps_taken == 0
        with pytest.raises(EpisodeSeedCustodyError):
            allocator.snapshot()
    finally:
        env.close()

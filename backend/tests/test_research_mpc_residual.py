"""Fixture-only causal cache, exact MPC baseline and restricted residual checks.

Seeds 9_990_900..9_990_979 are fixtures, never registered or development data.
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

from plant_backend.controllers.base import Constraints
from plant_backend import operational_mpc
from plant_backend.operational_mpc import OperatingEvidence, OperatingDecision
from plant_backend.rl.v2 import beta_spaces
from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.contract import contract_digest
from plant_backend.rl.v2.mpc_execution import public_observation_sha256
from plant_backend.rl.v2.setpoint_command import issue_process_setpoint_command
from plant_backend.rl.v2.setpoint_control import ProcessSetpointTargetsV1
from plant_backend.rl.v2.economic_mpc_schema import EconomicMPCPlanError
from plant_backend.rl.v2.mpc_schema import canonical_json, domain_sha256
from types import ModuleType, SimpleNamespace
from plant_backend.rl.v2.setpoint_env import PlantSetpointEnvV2
from plant_backend.rl.v2.training import research_mpc_residual as residual
from plant_backend.rl.v2.training.episode_allocator import (
    EpisodeSeedAllocatorV1,
    EpisodeSeedCustodyError,
)


def _base(**kwargs: Any) -> PlantCyclicMarketEnvV2Beta:
    return PlantCyclicMarketEnvV2Beta(**kwargs)


@pytest.fixture
def cheap_prior(monkeypatch):
    """Control only planner construction/solve; retain the native MPC fallback path."""

    def model(controller, observation, **kwargs):
        return SimpleNamespace(
            sha256=domain_sha256("controlled-mpc-model-fixture", observation.sha256),
            horizon_intervals=min(
                96, controller.settings.task_interval_count - controller._next_decision
            ),
        )

    def solve(*args, **kwargs):
        raise EconomicMPCPlanError("controlled solver fixture")

    monkeypatch.setattr(operational_mpc.OperationalMPCController, "build_model", model)
    monkeypatch.setattr(operational_mpc, "solve_economic_mpc_model", solve)


@pytest.fixture
def wrapper(cheap_prior):
    instance = residual.ResearchMPCResidualWrapper(_base())
    instance.reset(seed=9_990_901)
    yield instance
    instance.close()


def _synthetic_mpc(
    base: PlantCyclicMarketEnvV2Beta,
    *,
    electrolyser=1800.0,
    dac=400.0,
    methanator=80.0,
    charge=1000.0,
    invalid=False,
) -> OperatingDecision:
    """Typed fabricated public-bound decision for mapping, not operating evidence."""
    targets = ProcessSetpointTargetsV1(electrolyser, dac, methanator, charge)
    payload = {
        "schema": operational_mpc.DECISION_SCHEMA,
        "observation_sha256": base.cyclic_market_controller_observation.sha256,
        "operation": {"mode": "fallback", "solver_failure": {"reason": "fixture"}},
        "targets": targets.to_dict(),
        "plan_sha256": None,
    }
    evidence = OperatingEvidence(None, ("mapping fixture",), canonical_json(payload))
    command = issue_process_setpoint_command(
        targets,
        decision_index=base.decisions_taken,
        observation_sha256=public_observation_sha256(base.controller_observation),
        hold_duration_s=base.episode_config.decision_interval_s,
        issuer_id=operational_mpc.POLICY_ID,
        source_schema=operational_mpc.DECISION_SCHEMA,
        source_sha256=evidence.sha256,
        constraints=base.constraints,
        config=base.config,
    )
    return OperatingDecision(evidence, command, invalid)


def _prepare(wrapper, decision, action):
    return residual.prepare_residual_action(
        wrapper.base,
        decision,
        action,
        interface_contract_digest=wrapper.interface_contract_digest,
    )


def _without_clock(info):
    return {key: value for key, value in info.items() if key != "dispatcher_elapsed_ns"}


def test_zero_residual_is_exact_MPC_for_complete_168h_fixture(monkeypatch, cheap_prior):
    direct = _base()
    wrapped_base = _base()
    wrapped = residual.ResearchMPCResidualWrapper(wrapped_base)
    h = residual.make_mpc_controller(direct)
    observe_calls = [0, 0]
    h_calls = 0
    for index, base in enumerate((direct, wrapped_base)):
        original = base._observe

        def observe(original=original, index=index):
            observe_calls[index] += 1
            return original()

        monkeypatch.setattr(base, "_observe", observe)
    original_h = wrapped._mpc.compute_command

    def compute(*args, **kwargs):
        nonlocal h_calls
        h_calls += 1
        return original_h(*args, **kwargs)

    monkeypatch.setattr(wrapped._mpc, "compute_command", compute)
    try:
        expected, direct_info = direct.reset(seed=9_990_900)
        actual, wrapped_info = wrapped.reset(seed=9_990_900)
        assert np.array_equal(actual[:135], expected)
        assert direct_info == wrapped_info
        assert observe_calls[0] == observe_calls[1]
        assert h_calls == 1
        for index in range(672):
            decision = h.compute_command(
                direct.cyclic_market_controller_observation,
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
            assert wrapped.last_residual_evidence["body"]["exact_zero_mpc_command"]
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
        assert wrapped.current_mpc_decision is None
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
    original_h = wrapper._mpc.compute_command
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

    monkeypatch.setattr(wrapper._mpc, "compute_command", compute)
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
        cached = wrapper.current_mpc_decision
        combined = residual.extend_observation(
            wrapper.current_base_observation, base, cached
        )
        assert np.array_equal(combined, actual)
        _prepare(wrapper, cached, [0.4, -0.7, 0, 0])
        assert wrapper.current_decision_index == 1
        wrapper.current_observation[:] = 0
        wrapper.current_mpc_features[:] = 0
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
        (600.0, -1, 600),
        (600.0, 1, 600),
        (1200.0, -1, 1200),
        (1200.0, 1, 1200),
        (1800.0, -1, 1650),
        (1800.0, 1, 1950),
        (3000.0, 1, 3000),
    ],
)
def test_fixed_floor_mapping_and_night_450_prior(wrapper, h_e, r_e, expected):
    command, invalid, evidence = _prepare(
        wrapper, _synthetic_mpc(wrapper.base, electrolyser=h_e), [r_e, -1, 1, 1]
    )
    assert command.targets.electrolyser_target_kw == expected
    assert command.targets.dac_target_kw == 300
    assert command.targets.methanator_target_kw == 100
    assert command.targets.battery_charge_ceiling_kw == 1250
    assert not invalid
    assert evidence["body"]["electrolyser_floor_kw"] == 1200


def test_nonzero_mapping_uses_active_equipment_and_physical_charge_limits(wrapper):
    wrapper.base.constraints = Constraints(
        electrolyser_max_kw=2000, dac_max_kw=450, methanator_max_kw=90
    )
    command, _, _ = _prepare(
        wrapper, _synthetic_mpc(wrapper.base, charge=2000), [1, 1, 1, 1]
    )
    assert command.targets == ProcessSetpointTargetsV1(1950, 450, 90, 2500)
    all_zero = _synthetic_mpc(
        wrapper.base, electrolyser=0, dac=0, methanator=0, charge=0
    )
    command, _, _ = _prepare(wrapper, all_zero, [-1, 1, 1, -1])
    assert command.targets == all_zero.command.targets


def test_exact_zero_keeps_original_float64_command_and_original_invalid(wrapper):
    target = 1800.123456789
    assert float(np.float32(target)) != target
    cached = _synthetic_mpc(wrapper.base, electrolyser=target, invalid=True)
    command, invalid, evidence = _prepare(
        wrapper, cached, np.zeros(4, dtype=np.float32)
    )
    assert command is cached.command
    assert command.targets.electrolyser_target_kw == target
    assert invalid is True
    assert evidence["command"] == cached.command.to_dict()
    assert evidence["body"]["exact_zero_mpc_command"]
    assert not evidence["body"]["residual_clipped"]


def test_wrapper_passes_original_MPC_invalid_flag_to_common_execution(
    monkeypatch, cheap_prior
):
    wrapper = residual.ResearchMPCResidualWrapper(_base())
    try:
        wrapper.reset(seed=9_990_903)
        wrapper._cached = _synthetic_mpc(wrapper.base, invalid=True)
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
    first = _prepare(wrapper, wrapper.current_mpc_decision, raw32)
    second = _prepare(
        wrapper,
        wrapper.current_mpc_decision,
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
    cached = wrapper.current_mpc_decision
    with pytest.raises((TypeError, ValueError)):
        wrapper.step(action)
    evidence = wrapper.last_residual_evidence
    assert evidence["status"] == "rejected"
    assert evidence["decision_index"] == 0
    json.dumps(evidence, allow_nan=False)
    assert wrapper.current_mpc_decision == cached
    assert wrapper.base.simulator.state == before
    assert wrapper.base.decisions_taken == wrapper.base.physics_steps_taken == 0
    assert wrapper._mpc._next_decision == 1


def test_stale_boundary_and_changed_authority_reject_before_movement(wrapper):
    cached = wrapper.current_mpc_decision
    wrapper.step([0, 0, 0, 0])
    with pytest.raises(ValueError, match="stale"):
        _prepare(wrapper, cached, [0, 0, 0, 0])
    wrapper.base.constraints = replace(wrapper.base.constraints, dac_max_kw=100)
    with pytest.raises(residual.MPCResidualIntegrityError, match="authority is stale"):
        wrapper.step([0, 0, 0, 0])
    assert wrapper.base.decisions_taken == 1
    assert wrapper.last_residual_evidence["status"] == "rejected"
    assert wrapper.current_mpc_decision is None


@pytest.mark.parametrize("tamper", ["index", "public_binding"])
def test_cached_boundary_loss_is_fatal_before_physics(wrapper, tamper):
    original = wrapper.current_mpc_decision
    if tamper == "index":
        wrapper.step([0, 0, 0, 0])
        wrapper._cached = original
    else:
        wrapper._cached = replace(
            original,
            command=replace(original.command, observation_sha256="0" * 64),
        )
    before = wrapper.base.simulator.state
    count = wrapper.base.physics_steps_taken
    with pytest.raises(residual.MPCResidualIntegrityError) as caught:
        wrapper.step([0, 0, 0, 0])
    assert isinstance(caught.value.__cause__, ValueError)
    assert wrapper.current_mpc_decision is None
    with pytest.raises(RuntimeError, match="reset"):
        wrapper.current_observation
    assert wrapper.base.simulator.state == before
    assert wrapper.base.physics_steps_taken == count


@pytest.mark.parametrize("failure", ["vector", "raised"])
def test_reset_after_physical_reset_failure_is_fatal(monkeypatch, cheap_prior, failure):
    wrapper = residual.ResearchMPCResidualWrapper(_base())
    base_reset = wrapper.base.reset
    calls = []

    def reset(**kwargs):
        observation, info = base_reset(**kwargs)
        calls.append(kwargs["seed"])
        if failure == "raised":
            raise ValueError("physical reset fixture failed")
        return observation.astype(np.float64), info

    monkeypatch.setattr(wrapper.base, "reset", reset)
    try:
        with pytest.raises(residual.MPCResidualIntegrityError) as caught:
            wrapper.reset(seed=9_990_905)
        assert isinstance(caught.value.__cause__, ValueError)
        assert calls == [9_990_905]
        assert wrapper.current_mpc_decision is None
        assert wrapper._mpc._next_decision == 0
        with pytest.raises(RuntimeError, match="reset"):
            wrapper.current_observation
    finally:
        wrapper.close()


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
            base_vector, wrapper.base, wrapper.current_mpc_decision
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
    assert len(contract["mpc_dependencies"]["source_sha256"]) >= 30
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
        "mpc_dependencies",
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


def _fabricated_numerical_layout(monkeypatch, tmp_path, numpy_layout, scipy_layout):
    modules = {}

    def module(name, *, native=False):
        path = tmp_path / (
            name.replace(".", "_")
            + (residual.EXTENSION_SUFFIXES[0] if native else ".py")
        )
        path.write_bytes(f"fixture implementation: {name}".encode())
        result = ModuleType(name)
        result.__file__ = str(path)
        modules[name] = result
        return result

    numpy_name = f"numpy.{numpy_layout}._multiarray_umath"
    module(numpy_name, native=True)
    scipy = module("scipy")
    scipy.__version__ = "fabricated-layout"
    milp = module("scipy.optimize._milp")
    if scipy_layout == "cython":
        wrapper_name = "scipy.optimize._highs._highs_wrapper"
        core = module(wrapper_name, native=True)
        native_names = [wrapper_name]
    else:
        wrapper_name = "scipy.optimize._highspy._highs_wrapper"
        wrapper = module(wrapper_name)
        core = module("scipy.optimize._highspy._core", native=True)
        core.HIGHS_VERSION_MAJOR = 1
        core.HIGHS_VERSION_MINOR = 2
        core.HIGHS_VERSION_PATCH = 3
        wrapper._h = core
        wrapper.hopt = module("scipy.optimize._highspy._highs_options", native=True)
        native_names = [core.__name__, wrapper.hopt.__name__]
    milp._highs_wrapper = SimpleNamespace(__module__=wrapper_name)
    module("scipy.optimize._constraints")
    module("scipy.sparse._csc")

    def load(name):
        if name not in modules:
            raise ModuleNotFoundError(f"fixture missing {name}", name=name)
        return modules[name]

    monkeypatch.setattr(residual.importlib, "import_module", load)
    return modules, numpy_name, native_names, core


@pytest.mark.parametrize("numpy_layout", ["core", "_core"])
@pytest.mark.parametrize("scipy_layout", ["cython", "highspy"])
def test_numerical_layout_selection_binds_actual_native_implementation(
    monkeypatch, tmp_path, numpy_layout, scipy_layout
):
    modules, numpy_name, native_names, _ = _fabricated_numerical_layout(
        monkeypatch, tmp_path, numpy_layout, scipy_layout
    )
    identity = residual._numerical_sources()
    assert identity["native_implementation_modules"] == {
        "numpy": numpy_name,
        "scipy_highs": sorted(native_names),
    }
    assert identity["scipy_version"] == "fabricated-layout"
    assert identity["highs_version"] == (None if scipy_layout == "cython" else "1.2.3")
    for name in [numpy_name, *native_names]:
        assert (
            identity["source_sha256"][name]
            == hashlib.sha256(Path(modules[name].__file__).read_bytes()).hexdigest()
        )


@pytest.mark.parametrize("missing", ["numpy", "scipy"])
def test_numerical_identity_never_accepts_only_python_shim_or_options_extension(
    monkeypatch, tmp_path, missing
):
    modules, numpy_name, _, core = _fabricated_numerical_layout(
        monkeypatch, tmp_path, "_core", "highspy"
    )
    target = modules[numpy_name] if missing == "numpy" else core
    path = tmp_path / "python_shim.py"
    path.write_text("fixture shim")
    target.__file__ = str(path)
    with pytest.raises(RuntimeError, match="native implementation"):
        residual._numerical_sources()


def test_numpy_layout_probe_does_not_hide_missing_runtime_dependency(monkeypatch):
    def broken_import(name):
        raise ModuleNotFoundError("fixture missing runtime", name="required_runtime")

    monkeypatch.setattr(residual.importlib, "import_module", broken_import)
    with pytest.raises(ModuleNotFoundError, match="missing runtime"):
        residual._numpy_native_module()


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
            residual.ResearchMPCResidualWrapper(env)
    finally:
        env.close()


def test_outer_custody_counts_each_reset_and_physical_step_once(cheap_prior):
    # Four shortened fixture streams exercise actual beta physics plus allocator
    # nesting, including terminal cache behavior and the next reset wave.
    episode = EpisodeConfigV2(horizon_h=24, decision_interval_s=3600)
    allocator = EpisodeSeedAllocatorV1.fixture(
        base_seed=9_990_940, episode_decisions=24
    )
    residuals = [
        residual.ResearchMPCResidualWrapper(_base(episode=episode)) for _ in range(4)
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
        assert all(env._mpc._next_decision == 24 for env in residuals)
        assert len(allocator.snapshot().ledger) == 4
        for env in environments:
            env.reset()
        assert len(allocator.snapshot().ledger) == 8
        assert [cursor.decision_offset for cursor in allocator.snapshot().cursors] == [
            0
        ] * 4
        assert all(env._mpc._next_decision == 1 for env in residuals)
    finally:
        for env in environments:
            env.close()


def test_outer_custody_fails_closed_on_rejected_residual(cheap_prior):
    wrapper = residual.ResearchMPCResidualWrapper(_base())
    allocator = EpisodeSeedAllocatorV1.fixture(
        base_seed=9_990_960, episode_decisions=672
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


def test_constructor_never_resets_observes_or_solves(monkeypatch):
    base = _base()

    def forbidden(*args, **kwargs):
        pytest.fail("constructor invoked runtime control")

    monkeypatch.setattr(base, "reset", forbidden)
    monkeypatch.setattr(base, "_observe", forbidden)
    monkeypatch.setattr(
        operational_mpc.OperationalMPCController, "compute_command", forbidden
    )
    monkeypatch.setattr(operational_mpc, "solve_economic_mpc_model", forbidden)
    wrapper = residual.ResearchMPCResidualWrapper(base)
    try:
        assert wrapper.current_mpc_decision is None
        assert wrapper._mpc._next_decision == 0
        assert wrapper.base.decisions_taken == wrapper.base.physics_steps_taken == 0
        assert wrapper._mpc.decision_events == ()
    finally:
        wrapper.close()


def test_bounded_actual_mpc_prefix_preserves_zero_residual_physics_and_full_prior():
    """Four real LP/MILP decisions per path, on a disjoint disposable fixture."""
    direct = _base()
    wrapper = residual.ResearchMPCResidualWrapper(_base())
    controller = residual.make_mpc_controller(direct)
    try:
        direct.reset(seed=9_990_902)
        wrapper.reset(seed=9_990_902)
        for _ in range(4):
            decision = controller.compute_command(
                direct.cyclic_market_controller_observation,
                constraints=direct.constraints,
                config=direct.config,
                commitment_profile=direct.episode_config.electrolyser_commitment,
            )
            original = wrapper.current_mpc_decision
            assert original.command.targets == decision.command.targets
            assert original.invalid == decision.invalid
            expected, reward, done, trunc, info = direct.step_process_setpoints(
                decision.command,
                controller_invalid=decision.invalid,
            )
            actual, wrapped_reward, wrapped_done, wrapped_trunc, wrapped_info = (
                wrapper.step([0, 0, 0, 0])
            )
            assert np.array_equal(actual[:135], expected)
            assert (wrapped_reward, wrapped_done, wrapped_trunc) == (
                reward,
                done,
                trunc,
            )
            assert wrapper.base.simulator.state == direct.simulator.state
            assert (
                wrapper.base.last_minute_truth_states == direct.last_minute_truth_states
            )
            assert wrapper.base.economic_metrics == direct.economic_metrics
            assert wrapper.base.cycle_settlements == direct.cycle_settlements
            assert wrapped_info["metrics"] == info["metrics"]
            saved = wrapper.last_residual_evidence["body"]["mpc_decision"]
            assert saved == residual.mpc_decision_record(original)
            assert saved["evidence"]["source_notes"] == list(
                original.evidence.source_notes
            )
            assert saved["evidence"]["payload"] == json.loads(
                original.evidence.payload_json
            )
            if original.evidence.plan is not None:
                assert saved["evidence"]["plan"] == original.evidence.plan.to_dict()
        assert direct.physics_steps_taken == wrapper.base.physics_steps_taken == 60
    finally:
        wrapper.close()
        direct.close()


@pytest.mark.parametrize("after_step", [False, True])
def test_fatal_mpc_error_propagates_and_clears_cached_control(
    monkeypatch, cheap_prior, after_step
):
    wrapper = residual.ResearchMPCResidualWrapper(_base())
    if after_step:
        wrapper.reset(seed=9_990_904)

    def failed(*args, **kwargs):
        raise RuntimeError("fatal MPC fixture")

    monkeypatch.setattr(wrapper._mpc, "compute_command", failed)
    try:
        with pytest.raises(RuntimeError, match="fatal MPC fixture"):
            if after_step:
                wrapper.step([0, 0, 0, 0])
            else:
                wrapper.reset(seed=9_990_904)
        assert wrapper.base.physics_steps_taken == (15 if after_step else 0)
        assert wrapper.current_mpc_decision is None
        assert wrapper.last_mpc_failure == {
            "decision_index": int(after_step),
            "type": "RuntimeError",
            "message": "fatal MPC fixture",
        }
        with pytest.raises(RuntimeError, match="reset"):
            wrapper.current_observation
        with pytest.raises(RuntimeError, match="reset"):
            wrapper.step([0, 0, 0, 0])
        if after_step:
            assert wrapper.last_residual_evidence["status"] == "issued"
        failure = wrapper.last_mpc_failure
        failure["message"] = "mutated"
        assert wrapper.last_mpc_failure["message"] == "fatal MPC fixture"
    finally:
        wrapper.close()


def test_mpc139_cannot_alias_H139_or_absolute135_contract_or_loader(wrapper, tmp_path):
    import zipfile
    from plant_backend.rl.v2.training.research_residual import ResearchResidualHWrapper
    from plant_backend.rl.v2.training import research_effort_batch as old_batch
    from plant_backend.rl.checkpoints import verify_policy_model_spaces

    h = ResearchResidualHWrapper(_base())
    try:
        with pytest.raises(ValueError, match="contract mismatch"):
            wrapper.require_interface_contract(h.interface_contract)
        with pytest.raises(ValueError, match="contract mismatch"):
            h.require_interface_contract(wrapper.interface_contract)
        with pytest.raises(ValueError):
            verify_policy_model_spaces(
                SimpleNamespace(
                    observation_space=wrapper.observation_space,
                    action_space=wrapper.action_space,
                ),
                observation_space=wrapper.base.observation_space,
                action_space=wrapper.base.action_space,
            )
        config = old_batch.template(tmp_path)
        old_identity = old_batch.policy_identity(config, old_batch.ARMS[2])
        archive = tmp_path / "mpc139.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr(
                "data",
                json.dumps(
                    {
                        "research_identity": {
                            **old_identity,
                            "interface_contract": wrapper.interface_contract,
                            "interface_contract_digest": wrapper.interface_contract_digest,
                        }
                    }
                ),
            )
        with pytest.raises(ValueError, match="interface or initialization differs"):
            old_batch.preflight_checkpoint(archive, old_identity, {})
        changed = wrapper.interface_contract
        changed["registered_evaluation_eligible"] = 0
        with pytest.raises(ValueError, match="contract mismatch"):
            wrapper.require_interface_contract(changed)
    finally:
        h.close()

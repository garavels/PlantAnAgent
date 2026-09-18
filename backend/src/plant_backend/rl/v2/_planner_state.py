"""Privileged exact-state primitives for the frozen feasible planner.

This is the only planner module allowed to know about private ``PlantEnvV2``
implementation state.  Keeping that access here makes the exact signature and
clone contract reviewable as one boundary.
"""

from __future__ import annotations

import copy
import hashlib
import math
import struct
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from numbers import Integral, Real
from types import FunctionType, MethodType, ModuleType

import gymnasium as gym
import numpy as np
from plant_sim import (
    BatteryConfig,
    DACConfig,
    DeliveryCommitment,
    DeliverySettlement,
    DeliveryStepResult,
    ElectrolyserCommitmentProfile,
    ElectrolyserCommitmentState,
    ElectrolyserConfig,
    FaultSpec,
    InitialState,
    MethanatorConfig,
    PlantConfig,
    PlantState,
    PlantSimulator,
    Scenario,
    SolarConfig,
    SolarFactorTape,
    StorageConfig,
)
from plant_sim.components import Component
from plant_sim.delivery import DeliveryLedger
from plant_sim.events import EventLog
from plant_sim.faults import FaultRegistry

from ...controllers.base import Constraints
from ..actions import ActionAdapter
from ..observation import ObservationEncoder
from .boundary import ControlStateV2
from .config import EpisodeConfigV2
from .env import PlantEnvV2
from .metrics import EpisodeMetricsV2
from .observation import ObservationEncoderV2
from .planner_schema import (
    CLONE_ALLOWLIST_SHA256,
    CanonicalPayloadV1,
    FinalScoreV1,
    canonical_json,
    canonical_payload,
    require_sha256,
)
from .reward import RewardWeightsV2
from .weather import WeatherConfigV2, WeatherEpisodeV2


STATE_SIGNATURE_SCOPE = "planner-step-action-future-v1"
CLONE_CONTRACT_VERSION = "deepcopy-immutable-memo-v1"

_COMPONENT_NAMES = (
    "solar",
    "battery",
    "electrolyser",
    "h2_tank",
    "dac",
    "co2_tank",
    "methanator",
    "ch4_tank",
)

_ATTESTED_SHARED_IMMUTABLES: dict[int, object] = {}
_CLONE_SHARED_TYPES: dict[str, type[object]] = {
    "env.config": PlantConfig,
    "env.config.solar": SolarConfig,
    "env.config.battery": BatteryConfig,
    "env.config.electrolyser": ElectrolyserConfig,
    "env.config.h2_tank": StorageConfig,
    "env.config.dac": DACConfig,
    "env.config.co2_tank": StorageConfig,
    "env.config.methanator": MethanatorConfig,
    "env.config.ch4_tank": StorageConfig,
    "env.episode_config": EpisodeConfigV2,
    "env.episode_config.weather": WeatherConfigV2,
    "env.episode_config.electrolyser_commitment": ElectrolyserCommitmentProfile,
    "env.constraints": Constraints,
    "env.weights": RewardWeightsV2,
    "env._episode.scenario": Scenario,
    "env._episode.scenario.initial": InitialState,
    "env._episode.scenario.faults[*]": FaultSpec,
    "env._episode.commitments[*]": DeliveryCommitment,
    "env._episode.weather": WeatherEpisodeV2,
    "env.simulator._solar_factor_tape": SolarFactorTape,
    "env.simulator.delivery_ledger._commitments[*]": DeliveryCommitment,
    "env.simulator.delivery_ledger._settlements[*]": DeliverySettlement,
    "env.simulator.delivery_ledger._current": DeliveryStepResult,
    "env.simulator.electrolyser._commitment_state": ElectrolyserCommitmentState,
}


@dataclass(frozen=True, slots=True)
class PlannerStateSignature:
    """Canonical state bytes plus their domain-separated audit payload."""

    payload: CanonicalPayloadV1

    def __post_init__(self) -> None:
        if self.payload.domain != "planner-state-v1":
            raise ValueError("planner state signature has the wrong payload domain")

    @property
    def canonical_bytes(self) -> bytes:
        return str(self.payload.canonical_json).encode("utf-8")

    @property
    def sha256(self) -> str:
        return str(self.payload.sha256)


def serialize_exact(value: object) -> object:
    """Recursively encode the protocol's finite binary64 value grammar."""

    active: set[int] = set()

    def encode(item: object) -> object:
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, Enum):
            if not isinstance(item.value, str):
                raise TypeError("exact-state enum values must be strings")
            return item.value
        if isinstance(item, int):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("exact-state floats must be finite")
            return {"$float64": item.hex()}
        if isinstance(item, str):
            return item

        identity = id(item)
        if identity in active:
            raise ValueError("exact-state values must not contain cycles")
        active.add(identity)
        try:
            if is_dataclass(item) and not isinstance(item, type):
                return {
                    field.name: encode(getattr(item, field.name))
                    for field in fields(item)
                }
            if isinstance(item, (tuple, list)):
                return [encode(element) for element in item]
            if isinstance(item, Mapping):
                if any(not isinstance(key, str) for key in item):
                    raise TypeError("exact-state mappings must have string keys")
                return {key: encode(item[key]) for key in sorted(item)}
        finally:
            active.remove(identity)
        raise TypeError(f"unsupported exact-state value: {type(item).__qualname__}")

    return encode(value)


def build_state_signature(
    env: PlantEnvV2,
    *,
    manifest_sha256: str,
    exogenous_episode_sha256: str,
    settings_sha256: str,
    action_library_sha256: str,
    macro_depth: int,
    battery_throughput_kwh: float,
) -> PlannerStateSignature:
    """Capture the complete same-depth future-relevant planner state."""

    _require_reset_environment(env)
    manifest_digest = require_sha256("manifest_sha256", manifest_sha256)
    exogenous_digest = require_sha256(
        "exogenous_episode_sha256", exogenous_episode_sha256
    )
    settings_digest = require_sha256("settings_sha256", settings_sha256)
    library_digest = require_sha256("action_library_sha256", action_library_sha256)
    depth = _require_nonnegative_integer("macro_depth", macro_depth)
    throughput = _finite_real(
        "battery_throughput_kwh", battery_throughput_kwh, minimum=0.0
    )

    episode = env._episode
    assert episode is not None
    if episode.exogenous_episode_sha256 != exogenous_digest:
        raise ValueError("environment exogenous episode identity mismatch")

    manifest = env.audit_manifest()
    if manifest.get("manifest_sha256") != manifest_digest:
        raise ValueError("environment manifest identity mismatch")
    if manifest.get("exogenous_episode_sha256") != exogenous_digest:
        raise ValueError("environment manifest exogenous identity mismatch")

    simulator = env.simulator
    if simulator._solar_factor_tape is not episode.weather.tape:
        raise ValueError("environment future-truth tape identity mismatch")
    if not (
        env.physics_steps_taken == simulator.step_index == env.metrics.physical_steps
    ):
        raise ValueError("environment, simulator, and metrics physics steps disagree")
    if env.decisions_taken != env.metrics.decisions:
        raise ValueError("environment and metrics decision counts disagree")

    faults = tuple(simulator.faults._faults)
    fault_sequence = simulator.faults._sequence
    if faults or type(fault_sequence) is not int or fault_sequence != 0:
        raise ValueError("planner state requires the canonical fault-free scope")
    fault_registry = {
        "faults": faults,
        "sequence": fault_sequence,
    }

    ledger = simulator.delivery_ledger
    delivery_ledger = {
        "frozen": ledger._frozen,
        "schedule": ledger._commitments,
        "pending": tuple(
            (item.id, ledger._pending[item.id])
            for item in ledger._commitments
            if item.id in ledger._pending
        ),
        "by_step": tuple(
            (step, ledger._by_step[step]) for step in sorted(ledger._by_step)
        ),
        "order": tuple(sorted(ledger._order.items(), key=lambda pair: pair[1])),
        "settlements": tuple(ledger._settlements),
        "current": ledger._current,
        "cumulative_requested_kg": ledger._cumulative_requested_kg,
        "cumulative_delivered_kg": ledger._cumulative_delivered_kg,
        "cumulative_unmet_kg": ledger._cumulative_unmet_kg,
    }
    event_log = {
        "events": tuple(simulator.events._events),
        "conditions": tuple(sorted(simulator.events._conditions.items())),
        "sequence": simulator.events._sequence,
        "time_s": simulator.events.time_s,
        "clock": simulator.events.clock,
        "maxlen": simulator.events._events.maxlen,
    }
    simulator_payload = {
        "scenario_id": simulator.scenario,
        "status_memory": simulator._status,
        "step_index": simulator.step_index,
        "time_s": simulator.time_s,
        "energy_produced_kwh": simulator.energy_produced_kwh,
        "energy_curtailed_kwh": simulator.energy_curtailed_kwh,
        "h2_produced_total_kg": simulator.h2_produced_total_kg,
        "co2_captured_total_kg": simulator.co2_captured_total_kg,
        "ch4_produced_total_kg": simulator.ch4_produced_total_kg,
        "truth_state": simulator.state,
        "telemetry_state": env.telemetry,
        "component_snapshots": tuple(
            (name, getattr(simulator, name).snapshot()) for name in _COMPONENT_NAMES
        ),
        "commitment_snapshot": simulator.electrolyser_commitment,
        "fault_scope": "canonical_fault_free_v1",
        "fault_registry": fault_registry,
        "event_log": event_log,
        "delivery_ledger": delivery_ledger,
    }
    metrics_fields = {
        field.name: getattr(env.metrics, field.name)
        for field in fields(EpisodeMetricsV2)
    }
    signature = {
        "schema_version": 1,
        "signature_scope": STATE_SIGNATURE_SCOPE,
        "manifest_sha256": manifest_digest,
        "exogenous_episode_sha256": exogenous_digest,
        "checkpoint_contract_digest": env.checkpoint_contract_digest,
        "settings_sha256": settings_digest,
        "action_library_sha256": library_digest,
        "macro_depth": depth,
        "physics_steps_taken": env.physics_steps_taken,
        "decisions_taken": env.decisions_taken,
        "episode_physics_steps": env.episode_physics_steps,
        "episode_decisions": env.episode_decisions,
        "simulator": simulator_payload,
        "control_state": env.control_state,
        "metrics_fields": metrics_fields,
        "encoder_previous_solar_kw": env.encoder._base._previous_solar_kw,
        "battery_throughput_kwh": throughput,
    }
    return PlannerStateSignature(
        canonical_payload("planner-state-v1", serialize_exact(signature))
    )


def clone_environment(env: PlantEnvV2) -> PlantEnvV2:
    """Clone operational state while sharing only the frozen protocol allowlist."""

    _require_reset_environment(env)
    episode = env._episode
    assert episode is not None
    episode_copy = replace(
        episode,
        stream_spawn_keys=dict(episode.stream_spawn_keys),
    )
    _, checkpoint_memo = _copy_checkpoint_tree(env._checkpoint_contract)
    memo: dict[int, object] = {id(episode): episode_copy, **checkpoint_memo}
    missing = object()
    for item in _clone_shared_objects(env):
        if item is not None:
            existing = memo.get(id(item), missing)
            if existing is not missing and existing is not item:
                raise ValueError(
                    "clone contract violation: checkpoint container overlaps "
                    "the immutable sharing allowlist"
                )
            memo[id(item)] = item
    clone = copy.deepcopy(env, memo)
    if not isinstance(clone, PlantEnvV2):
        raise TypeError("deepcopy did not produce a PlantEnvV2")
    audit_clone_aliases(env, clone)
    return clone


def audit_clone_aliases(source: PlantEnvV2, clone: PlantEnvV2) -> None:
    """Reject an optimized clone with any undeclared operational alias."""

    _require_reset_environment(source)
    _require_reset_environment(clone)
    if source is clone:
        raise ValueError("clone must be a distinct environment")

    allowed_ids = {
        id(item) for item in _clone_shared_objects(source) if item is not None
    }
    clone_ids = {id(item) for item in _walk_graph(clone)}
    for item in _walk_graph(source):
        if (
            id(item) in clone_ids
            and id(item) not in allowed_ids
            and _identity_must_be_detached(item)
        ):
            raise ValueError(
                "clone contract violation: shared "
                f"{type(item).__module__}.{type(item).__qualname__}"
            )

    source_episode = source._episode
    clone_episode = clone._episode
    assert source_episode is not None and clone_episode is not None
    if source_episode is clone_episode:
        raise ValueError("clone contract violation: outer EpisodeV2 is shared")
    if source_episode.stream_spawn_keys is clone_episode.stream_spawn_keys:
        raise ValueError("clone contract violation: episode stream mapping is shared")

    _audit_checkpoint_tree(
        source._checkpoint_contract,
        clone._checkpoint_contract,
    )

    source_shared = tuple(_clone_shared_objects(source))
    clone_shared = tuple(_clone_shared_objects(clone))
    if len(source_shared) != len(clone_shared) or any(
        left is not right
        for left, right in zip(source_shared, clone_shared, strict=True)
    ):
        raise ValueError(
            "clone contract violation: immutable allowlist identity changed"
        )

    for name in _COMPONENT_NAMES:
        cloned_component = getattr(clone.simulator, name)
        if clone.simulator.components.get(name) is not cloned_component:
            raise ValueError(
                "clone contract violation: component registry alias mismatch "
                f"for {name}"
            )
        if cloned_component is getattr(source.simulator, name):
            raise ValueError(
                f"clone contract violation: operational component {name} is shared"
            )


def accumulate_battery_throughput(
    battery_throughput_kwh: float,
    step_result_or_info: object,
) -> float:
    """Apply the frozen left-to-right bus-side charge/discharge accumulation."""

    total = _finite_real("battery_throughput_kwh", battery_throughput_kwh, minimum=0.0)
    info: object = step_result_or_info
    if isinstance(step_result_or_info, tuple):
        if len(step_result_or_info) != 5:
            raise ValueError("step_action result must contain five values")
        info = step_result_or_info[4]
    if not isinstance(info, Mapping):
        raise TypeError("step_action info must be a mapping")
    interval_power = info.get("interval_power")
    if not isinstance(interval_power, Mapping):
        raise ValueError("step_action info is missing interval_power")
    applied = interval_power.get("applied_kwh")
    if not isinstance(applied, Mapping):
        raise ValueError("step_action interval_power is missing applied_kwh")
    battery = _finite_real("applied battery kWh", applied.get("battery"), minimum=0.0)
    support = _finite_real(
        "applied battery-support kWh",
        applied.get("battery_support"),
        minimum=0.0,
    )
    total += battery
    total += support
    if not math.isfinite(total):
        raise ValueError("battery throughput accumulation became non-finite")
    return total


def extract_final_score(
    env: PlantEnvV2,
    *,
    battery_throughput_kwh: float,
) -> FinalScoreV1:
    """Extract the ten authoritative terminal comparator fields."""

    _require_reset_environment(env)
    if (
        env.physics_steps_taken != env.episode_physics_steps
        or env.decisions_taken != env.episode_decisions
    ):
        raise ValueError("final score requires a complete environment horizon")
    if (
        env.simulator.step_index != env.physics_steps_taken
        or env.metrics.physical_steps != env.physics_steps_taken
        or env.metrics.decisions != env.decisions_taken
    ):
        raise ValueError("terminal environment, simulator, and metrics counts disagree")
    throughput = _finite_real(
        "battery_throughput_kwh", battery_throughput_kwh, minimum=0.0
    )
    metrics = env.metrics
    state = env.simulator.state
    paired_fields = (
        ("terminal_h2_kg", "h2_level_kg"),
        ("terminal_co2_kg", "co2_level_kg"),
        ("terminal_ch4_kg", "ch4_level_kg"),
        ("terminal_battery_soc", "battery_soc"),
        ("terminal_battery_energy_kwh", "battery_energy_kwh"),
    )
    for metric_name, state_name in paired_fields:
        if not _same_binary64(
            getattr(metrics, metric_name),
            getattr(state, state_name),
        ):
            raise ValueError(
                f"terminal metric/truth mismatch: {metric_name} != {state_name}"
            )
    ledger_delivered = env.simulator.delivery_ledger._cumulative_delivered_kg
    if not _same_binary64(metrics.delivered_kg, ledger_delivered):
        raise ValueError("delivered metric and delivery ledger disagree")

    methanator = env.config.methanator
    equivalent = state.ch4_level_kg + min(
        state.h2_level_kg / methanator.h2_kg_per_kg_ch4,
        state.co2_level_kg / methanator.co2_kg_per_kg_ch4,
    )
    return FinalScoreV1(
        delivered_kg=metrics.delivered_kg,
        service_numerator=metrics.commitments_met,
        service_denominator=metrics.commitments_settled,
        worst_fill=metrics.worst_commitment_fill_rate,
        terminal_ch4_equivalent_inventory_kg=equivalent,
        terminal_battery_energy_kwh=state.battery_energy_kwh,
        forced_stops=metrics.electrolyser_forced_stops,
        aborted_starts=metrics.electrolyser_aborted_starts,
        cold_starts=metrics.electrolyser_cold_starts,
        curtailed_kwh=metrics.curtailed_kwh,
        battery_throughput_kwh=throughput,
    )


def _clone_shared_entries(
    env: PlantEnvV2,
) -> tuple[tuple[str, tuple[object | None, ...]], ...]:
    """Bind each frozen canonical path to the exact objects it resolves to."""

    episode = env._episode
    assert episode is not None
    entries = (
        ("env.config", (env.config,)),
        ("env.config.solar", (env.config.solar,)),
        ("env.config.battery", (env.config.battery,)),
        ("env.config.electrolyser", (env.config.electrolyser,)),
        ("env.config.h2_tank", (env.config.h2_tank,)),
        ("env.config.dac", (env.config.dac,)),
        ("env.config.co2_tank", (env.config.co2_tank,)),
        ("env.config.methanator", (env.config.methanator,)),
        ("env.config.ch4_tank", (env.config.ch4_tank,)),
        ("env.episode_config", (env.episode_config,)),
        ("env.episode_config.weather", (env.episode_config.weather,)),
        (
            "env.episode_config.electrolyser_commitment",
            (env.episode_config.electrolyser_commitment,),
        ),
        ("env.constraints", (env.constraints,)),
        ("env.weights", (env.weights,)),
        ("env._episode.scenario", (episode.scenario,)),
        ("env._episode.scenario.initial", (episode.scenario.initial,)),
        ("env._episode.scenario.faults[*]", tuple(episode.scenario.faults)),
        ("env._episode.commitments[*]", tuple(episode.commitments)),
        ("env._episode.weather", (episode.weather,)),
        ("env.simulator._solar_factor_tape", (env.simulator._solar_factor_tape,)),
        (
            "env.simulator.delivery_ledger._commitments[*]",
            tuple(env.simulator.delivery_ledger._commitments),
        ),
        (
            "env.simulator.delivery_ledger._settlements[*]",
            tuple(env.simulator.delivery_ledger._settlements),
        ),
        (
            "env.simulator.delivery_ledger._current",
            (env.simulator.delivery_ledger._current,),
        ),
        (
            "env.simulator.electrolyser._commitment_state",
            (env.simulator.electrolyser._commitment_state,),
        ),
    )
    if {path for path, _ in entries} != set(_CLONE_SHARED_TYPES):
        raise RuntimeError("executable clone allowlist/type contract is incomplete")
    for path, values in entries:
        expected_type = _CLONE_SHARED_TYPES[path]
        for index, value in enumerate(values):
            if value is None:
                continue
            if type(value) is not expected_type:
                raise TypeError(
                    f"clone allowlist value {path}[{index}] must have exact type "
                    f"{expected_type.__module__}.{expected_type.__qualname__}"
                )
            _attest_transitively_immutable(value, path=f"{path}[{index}]")
    return entries


def _attest_transitively_immutable(value: object, *, path: str) -> None:
    """Admit one shareable object only after proving its complete immutable closure."""

    cached = _ATTESTED_SHARED_IMMUTABLES.get(id(value))
    if cached is value:
        return
    active: set[int] = set()

    def visit(item: object, *, item_path: str) -> None:
        if item is None or isinstance(item, (bool, int, str, bytes, Enum)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"clone allowlist float at {item_path} must be finite")
            return

        identity = id(item)
        if identity in active:
            raise ValueError(
                f"clone allowlist value at {item_path} contains an object cycle"
            )
        active.add(identity)
        try:
            if isinstance(item, tuple):
                for index, nested in enumerate(item):
                    visit(nested, item_path=f"{item_path}[{index}]")
                return
            if is_dataclass(item) and not isinstance(item, type):
                parameters = getattr(type(item), "__dataclass_params__", None)
                if parameters is None or not parameters.frozen:
                    raise TypeError(
                        f"clone allowlist dataclass at {item_path} must be frozen"
                    )
                for field in fields(item):
                    visit(
                        getattr(item, field.name),
                        item_path=f"{item_path}.{field.name}",
                    )
                return
        finally:
            active.remove(identity)
        raise TypeError(
            f"clone allowlist value at {item_path} is not transitively immutable: "
            f"{type(item).__module__}.{type(item).__qualname__}"
        )

    visit(value, item_path=path)
    _ATTESTED_SHARED_IMMUTABLES[id(value)] = value


def _clone_shared_objects(env: PlantEnvV2) -> Iterator[object | None]:
    """Yield canonical-path objects only after verifying the frozen path digest."""

    entries = _clone_shared_entries(env)
    path_payload = [path for path, _ in entries]
    actual_digest = hashlib.sha256(
        canonical_json(path_payload).encode("utf-8")
    ).hexdigest()
    if actual_digest != CLONE_ALLOWLIST_SHA256:
        raise RuntimeError(
            "executable clone allowlist does not match its frozen digest"
        )
    for _, values in entries:
        yield from values


def _walk_graph(root: object) -> Iterator[object]:
    """Walk Python-owned reachability without invoking arbitrary properties."""

    pending = [root]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        yield item

        if item is None or isinstance(
            item,
            (
                bool,
                int,
                float,
                complex,
                str,
                bytes,
                Enum,
                FunctionType,
                MethodType,
                ModuleType,
                type,
            ),
        ):
            continue
        if isinstance(item, Mapping):
            pending.extend(item.keys())
            pending.extend(item.values())
            continue
        if isinstance(item, (list, tuple, set, frozenset, deque)):
            pending.extend(item)
            continue
        if isinstance(item, np.random.Generator):
            pending.append(item.bit_generator)
            continue
        if isinstance(item, np.ndarray):
            continue
        if is_dataclass(item) and not isinstance(item, type):
            pending.extend(getattr(item, field.name) for field in fields(item))
            continue
        try:
            namespace = vars(item)
        except TypeError:
            namespace = None
        if namespace is not None:
            pending.extend(namespace.values())
        for owner in type(item).__mro__:
            slots = owner.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots:
                if slot in {"__dict__", "__weakref__"}:
                    continue
                try:
                    pending.append(getattr(item, slot))
                except AttributeError:
                    pass


def _identity_must_be_detached(value: object) -> bool:
    if value is None or isinstance(
        value,
        (
            bool,
            int,
            float,
            complex,
            str,
            bytes,
            Enum,
            FunctionType,
            MethodType,
            ModuleType,
            type,
            tuple,
            frozenset,
        ),
    ):
        return False
    mutable_containers = (dict, list, set, deque, np.ndarray)
    operational_types = (
        np.random.Generator,
        np.random.BitGenerator,
        gym.Space,
        PlantEnvV2,
        PlantSimulator,
        Component,
        FaultRegistry,
        DeliveryLedger,
        EventLog,
        ObservationEncoder,
        ObservationEncoderV2,
        ActionAdapter,
        EpisodeMetricsV2,
        PlantState,
        ControlStateV2,
    )
    if isinstance(value, mutable_containers + operational_types):
        return True
    module = type(value).__module__
    if module == "plant_backend" or module.startswith("plant_backend."):
        params = getattr(type(value), "__dataclass_params__", None)
        return params is None or not params.frozen
    if module == "plant_sim" or module.startswith("plant_sim."):
        params = getattr(type(value), "__dataclass_params__", None)
        return params is None or not params.frozen
    return False


def _copy_checkpoint_tree(root: object) -> tuple[object, dict[int, object]]:
    """Copy every checkpoint container, including otherwise-shareable tuples."""

    memo: dict[int, object] = {}
    active: set[int] = set()

    def clone(value: object, *, path: str) -> object:
        if _is_checkpoint_scalar(value):
            _validate_checkpoint_scalar(value, path=path)
            return value
        if type(value) not in (dict, list, tuple):
            raise TypeError(
                f"checkpoint value at {path} has unsupported type "
                f"{type(value).__qualname__}"
            )

        identity = id(value)
        if identity in active:
            raise ValueError(f"checkpoint value at {path} contains a cycle")
        if identity in memo:
            return memo[identity]

        active.add(identity)
        try:
            if isinstance(value, dict):
                if any(not isinstance(key, str) for key in value):
                    raise TypeError(
                        f"checkpoint mapping at {path} must have string keys"
                    )
                copied_mapping: dict[str, object] = {}
                memo[identity] = copied_mapping
                for key, item in value.items():
                    copied_mapping[key] = clone(item, path=f"{path}.{key}")
                return copied_mapping
            if isinstance(value, list):
                copied_list: list[object] = []
                memo[identity] = copied_list
                copied_list.extend(
                    clone(item, path=f"{path}[{index}]")
                    for index, item in enumerate(value)
                )
                return copied_list

            assert isinstance(value, tuple)
            if not value:
                raise ValueError(
                    f"empty checkpoint tuple at {path} cannot have a distinct "
                    "built-in tuple identity"
                )
            copied_tuple = tuple(
                clone(item, path=f"{path}[{index}]") for index, item in enumerate(value)
            )
            if copied_tuple is value:
                raise ValueError(
                    f"checkpoint tuple at {path} was not identity-detached"
                )
            memo[identity] = copied_tuple
            return copied_tuple
        finally:
            active.remove(identity)

    copied = clone(root, path="checkpoint")
    return copied, memo


def _audit_checkpoint_tree(source: object, clone: object) -> None:
    """Require exact values, matching alias topology, and detached containers."""

    source_to_clone: dict[int, int] = {}
    clone_to_source: dict[int, int] = {}
    active_source: set[int] = set()
    active_clone: set[int] = set()

    def audit(left: object, right: object, *, path: str) -> None:
        left_container = type(left) in (dict, list, tuple)
        right_container = type(right) in (dict, list, tuple)
        if not left_container or not right_container:
            if left_container != right_container:
                raise ValueError(
                    f"clone contract violation: checkpoint shape differs at {path}"
                )
            _validate_checkpoint_scalar(left, path=path)
            _validate_checkpoint_scalar(right, path=path)
            if type(left) is not type(right) or not _checkpoint_scalars_equal(
                left, right
            ):
                raise ValueError(
                    f"clone contract violation: checkpoint value differs at {path}"
                )
            return

        if type(left) is not type(right):
            raise ValueError(
                f"clone contract violation: checkpoint container type differs at {path}"
            )
        if left is right:
            raise ValueError(
                f"clone contract violation: checkpoint container is shared at {path}"
            )

        left_id = id(left)
        right_id = id(right)
        if left_id in active_source or right_id in active_clone:
            raise ValueError(
                f"clone contract violation: checkpoint cycle detected at {path}"
            )
        known_right = source_to_clone.get(left_id)
        known_left = clone_to_source.get(right_id)
        if known_right is not None or known_left is not None:
            if known_right != right_id or known_left != left_id:
                raise ValueError(
                    "clone contract violation: checkpoint container alias topology "
                    f"differs at {path}"
                )
            return

        source_to_clone[left_id] = right_id
        clone_to_source[right_id] = left_id
        active_source.add(left_id)
        active_clone.add(right_id)
        try:
            if isinstance(left, dict):
                assert isinstance(right, dict)
                if any(not isinstance(key, str) for key in left) or any(
                    not isinstance(key, str) for key in right
                ):
                    raise TypeError(
                        f"checkpoint mapping at {path} must have string keys"
                    )
                if set(left) != set(right):
                    raise ValueError(
                        f"clone contract violation: checkpoint keys differ at {path}"
                    )
                for key in sorted(left):
                    audit(left[key], right[key], path=f"{path}.{key}")
                return

            assert isinstance(left, (list, tuple))
            assert isinstance(right, (list, tuple))
            if len(left) != len(right):
                raise ValueError(
                    f"clone contract violation: checkpoint length differs at {path}"
                )
            for index, (left_item, right_item) in enumerate(
                zip(left, right, strict=True)
            ):
                audit(left_item, right_item, path=f"{path}[{index}]")
        finally:
            active_source.remove(left_id)
            active_clone.remove(right_id)

    audit(source, clone, path="checkpoint")


def _is_checkpoint_scalar(value: object) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _validate_checkpoint_scalar(value: object, *, path: str) -> None:
    if not _is_checkpoint_scalar(value):
        raise TypeError(
            f"checkpoint value at {path} has unsupported type "
            f"{type(value).__qualname__}"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"checkpoint float at {path} must be finite")


def _checkpoint_scalars_equal(left: object, right: object) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return struct.pack(">d", left) == struct.pack(">d", right)
    return left == right


def _require_reset_environment(env: object) -> None:
    if not isinstance(env, PlantEnvV2):
        raise TypeError("env must be a PlantEnvV2")
    if env._episode is None or env.control_state is None:
        raise RuntimeError("PlantEnvV2 must be reset before planner use")


def _require_nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _finite_real(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}")
    return result


def _same_binary64(left: object, right: object) -> bool:
    left_value = _finite_real("left binary64 value", left)
    right_value = _finite_real("right binary64 value", right)
    return struct.pack(">d", left_value) == struct.pack(">d", right_value)


__all__ = [
    "CLONE_ALLOWLIST_SHA256",
    "CLONE_CONTRACT_VERSION",
    "PlannerStateSignature",
    "STATE_SIGNATURE_SCOPE",
    "accumulate_battery_throughput",
    "audit_clone_aliases",
    "build_state_signature",
    "clone_environment",
    "extract_final_score",
    "serialize_exact",
]

"""NONCANONICAL two-decision fixtures for privileged planner state primitives."""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, fields, replace
from enum import Enum

import numpy as np
import pytest
from plant_sim import (
    Action,
    DeliverySettlement,
    DeliveryStepResult,
    Event,
    SolarFactorTape,
)

from plant_backend.rl.v2 import EpisodeConfigV2, EpisodeMetricsV2, PlantEnvV2
from plant_backend.rl.v2.boundary import IntervalSummaryV2
from plant_backend.rl.v2._planner_state import (
    accumulate_battery_throughput,
    audit_clone_aliases,
    build_state_signature,
    clone_environment,
    extract_final_score,
    serialize_exact,
)
from plant_backend.rl.v2.planner_schema import canonical_json, canonical_payload


# Explicitly noncanonical: exactly two 15-minute controller decisions.  These
# fixtures must never be treated as registered planner evidence.
NONCANONICAL_TWO_DECISIONS = EpisodeConfigV2(
    horizon_h=0.5,
    delivery_period_h=0.5,
)
_SETTINGS_SHA256 = "0" * 64
_ACTION_LIBRARY_SHA256 = "1" * 64


class _Mode(str, Enum):
    READY = "ready"


@dataclass(frozen=True)
class _Nested:
    number: float
    mode: _Mode


def _signature(
    env: PlantEnvV2,
    *,
    macro_depth: int,
    throughput: float,
):
    manifest = env.audit_manifest()
    return build_state_signature(
        env,
        manifest_sha256=manifest["manifest_sha256"],
        exogenous_episode_sha256=manifest["exogenous_episode_sha256"],
        settings_sha256=_SETTINGS_SHA256,
        action_library_sha256=_ACTION_LIBRARY_SHA256,
        macro_depth=macro_depth,
        battery_throughput_kwh=throughput,
    )


def _assert_checkpoint_containers_detached(source: object, clone: object) -> None:
    """Recursively prove identity separation for every checkpoint container."""

    assert type(source) is type(clone)
    if isinstance(source, dict):
        assert isinstance(clone, dict)
        assert source is not clone
        assert source.keys() == clone.keys()
        for key in source:
            _assert_checkpoint_containers_detached(source[key], clone[key])
        return
    if isinstance(source, (list, tuple)):
        assert isinstance(clone, (list, tuple))
        assert type(source) is type(clone)
        assert source is not clone
        assert len(source) == len(clone)
        for source_item, clone_item in zip(source, clone, strict=True):
            _assert_checkpoint_containers_detached(source_item, clone_item)
        return
    assert serialize_exact(source) == serialize_exact(clone)


def test_exact_serializer_preserves_binary64_and_rejects_open_grammar() -> None:
    encoded = serialize_exact(
        {
            "z": (_Nested(number=-0.0, mode=_Mode.READY), 3),
            "a": 0.1,
        }
    )
    assert encoded == {
        "a": {"$float64": "0x1.999999999999ap-4"},
        "z": [
            {
                "number": {"$float64": "-0x0.0p+0"},
                "mode": "ready",
            },
            3,
        ],
    }
    payload = canonical_payload("planner-state-v1", encoded)
    assert payload.sha256 == (
        "dfdcf08beacf67ef76fdf301888464e9758883ddd37c4d04a55fd5f06130c62c"
    )

    with pytest.raises(ValueError, match="finite"):
        serialize_exact(math.inf)
    with pytest.raises(TypeError, match="string keys"):
        serialize_exact({1: "not allowed"})
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cycles"):
        serialize_exact(cyclic)


def test_state_signature_is_exact_complete_and_sensitive_to_nested_state() -> None:
    env = PlantEnvV2(episode=NONCANONICAL_TWO_DECISIONS)
    env.reset(seed=7)

    initial = _signature(env, macro_depth=0, throughput=0.0)
    repeated = _signature(env, macro_depth=0, throughput=0.0)
    assert initial.canonical_bytes == repeated.canonical_bytes
    assert initial.sha256 == repeated.sha256
    assert initial.sha256 == (
        "813d00aa0681aaab954a8724ff87e49efcbf7148bbb3bfa05f14d32245a8227f"
    )

    changed_metric = clone_environment(env)
    changed_metric.metrics.reward_total = math.nextafter(
        changed_metric.metrics.reward_total, math.inf
    )
    assert (
        _signature(changed_metric, macro_depth=0, throughput=0.0).canonical_bytes
        != initial.canonical_bytes
    )

    changed_event = clone_environment(env)
    changed_event.simulator.events._conditions["unit.noncanonical"] = True
    assert _signature(changed_event, macro_depth=0, throughput=0.0).canonical_bytes != (
        initial.canonical_bytes
    )
    assert _signature(env, macro_depth=0, throughput=math.ulp(0.0)).canonical_bytes != (
        initial.canonical_bytes
    )

    for inventory_name in (
        "h2_level_kg",
        "co2_level_kg",
        "ch4_level_kg",
        "battery_soc",
        "battery_energy_kwh",
    ):
        changed = clone_environment(env)
        state = changed.simulator.state
        changed.simulator.state = replace(
            state,
            **{
                inventory_name: math.nextafter(
                    getattr(state, inventory_name),
                    math.inf,
                )
            },
        )
        assert _signature(changed, macro_depth=0, throughput=0.0).canonical_bytes != (
            initial.canonical_bytes
        )

    for commitment_name in (
        "mode_elapsed_s",
        "startup_remaining_s",
        "cooldown_remaining_s",
        "min_up_remaining_s",
        "min_down_remaining_s",
        "shortfall_elapsed_s",
        "previous_power_kw",
    ):
        changed = clone_environment(env)
        commitment = changed.simulator.electrolyser._commitment_state
        changed.simulator.electrolyser._commitment_state = replace(
            commitment,
            **{
                commitment_name: math.nextafter(
                    getattr(commitment, commitment_name),
                    math.inf,
                )
            },
        )
        assert _signature(changed, macro_depth=0, throughput=0.0).canonical_bytes != (
            initial.canonical_bytes
        )

    changed_ledger = clone_environment(env)
    changed_ledger.simulator.delivery_ledger._cumulative_unmet_kg = math.ulp(0.0)
    assert (
        _signature(changed_ledger, macro_depth=0, throughput=0.0).canonical_bytes
        != initial.canonical_bytes
    )

    changed_prior_interval = clone_environment(env)
    assert changed_prior_interval.control_state is not None
    changed_prior_interval.control_state = replace(
        changed_prior_interval.control_state,
        previous_interval=IntervalSummaryV2(
            physical_steps=1,
            hours=math.ulp(0.0),
        ),
    )
    assert (
        _signature(
            changed_prior_interval, macro_depth=0, throughput=0.0
        ).canonical_bytes
        != initial.canonical_bytes
    )

    changed_encoder = clone_environment(env)
    changed_encoder.encoder._base._previous_solar_kw = math.ulp(0.0)
    assert (
        _signature(changed_encoder, macro_depth=0, throughput=0.0).canonical_bytes
        != initial.canonical_bytes
    )


def test_state_signature_covers_metric_histories_and_every_prior_interval_field() -> (
    None
):
    env = PlantEnvV2(episode=NONCANONICAL_TWO_DECISIONS)
    env.reset(seed=7)
    initial = _signature(env, macro_depth=0, throughput=0.0)
    payload = initial.payload.value()
    assert isinstance(payload, dict)
    assert set(payload["metrics_fields"]) == {
        field.name for field in fields(EpisodeMetricsV2)
    }

    for history_name in ("reward_components", "electrolyser_mode_time_s"):
        changed = clone_environment(env)
        history = getattr(changed.metrics, history_name)
        history["unit.noncanonical"] = math.ulp(0.0)
        assert _signature(changed, macro_depth=0, throughput=0.0).canonical_bytes != (
            initial.canonical_bytes
        ), history_name

    for field in fields(IntervalSummaryV2):
        changed = clone_environment(env)
        assert changed.control_state is not None
        interval = changed.control_state.previous_interval
        current = getattr(interval, field.name)
        replacement = (
            current + 1
            if isinstance(current, int)
            else math.nextafter(current, math.inf)
        )
        changed.control_state = replace(
            changed.control_state,
            previous_interval=replace(interval, **{field.name: replacement}),
        )
        assert _signature(changed, macro_depth=0, throughput=0.0).canonical_bytes != (
            initial.canonical_bytes
        ), field.name


def test_state_signature_is_sensitive_to_each_delivery_ledger_field() -> None:
    env = PlantEnvV2(episode=NONCANONICAL_TWO_DECISIONS)
    env.reset(seed=7)
    initial = _signature(env, macro_depth=0, throughput=0.0).canonical_bytes

    for ledger_field in (
        "frozen",
        "schedule",
        "pending",
        "by_step",
        "order",
        "settlements",
        "current",
        "cumulative_requested_kg",
        "cumulative_delivered_kg",
        "cumulative_unmet_kg",
    ):
        changed = clone_environment(env)
        ledger = changed.simulator.delivery_ledger
        assert ledger._commitments
        commitment = ledger._commitments[0]
        if ledger_field == "frozen":
            ledger._frozen = not ledger._frozen
        elif ledger_field == "schedule":
            ledger._commitments = ()
        elif ledger_field == "pending":
            ledger._pending = {}
        elif ledger_field == "by_step":
            ledger._by_step = {}
        elif ledger_field == "order":
            ledger._order[commitment.id] += 1
        elif ledger_field == "settlements":
            ledger._settlements.append(
                DeliverySettlement(
                    id="unit.noncanonical",
                    due_step=1,
                    requested_kg=0.0,
                    delivered_kg=0.0,
                    unmet_kg=0.0,
                )
            )
        elif ledger_field == "current":
            ledger._current = replace(ledger._current, step=ledger._current.step + 1)
        elif ledger_field == "cumulative_requested_kg":
            ledger._cumulative_requested_kg = math.ulp(0.0)
        elif ledger_field == "cumulative_delivered_kg":
            ledger._cumulative_delivered_kg = math.ulp(0.0)
        else:
            assert ledger_field == "cumulative_unmet_kg"
            ledger._cumulative_unmet_kg = math.ulp(0.0)
        assert _signature(changed, macro_depth=0, throughput=0.0).canonical_bytes != (
            initial
        ), ledger_field


def test_state_signature_is_sensitive_to_each_event_log_and_event_field() -> None:
    env = PlantEnvV2(episode=NONCANONICAL_TWO_DECISIONS)
    env.reset(seed=7)
    initial = _signature(env, macro_depth=0, throughput=0.0).canonical_bytes

    for log_field in (
        "events",
        "conditions",
        "sequence",
        "time_s",
        "clock",
        "maxlen",
    ):
        changed = clone_environment(env)
        event_log = changed.simulator.events
        if log_field == "events":
            event_log._events.append(
                Event(
                    seq=0,
                    time_s=0.0,
                    clock="D1 00:00",
                    level="info",
                    code="unit.noncanonical",
                    message="noncanonical event fixture",
                )
            )
        elif log_field == "conditions":
            event_log._conditions["unit.noncanonical"] = True
        elif log_field == "sequence":
            event_log._sequence += 1
        elif log_field == "time_s":
            event_log.time_s = math.nextafter(event_log.time_s, math.inf)
        elif log_field == "clock":
            event_log.clock += " noncanonical"
        else:
            assert log_field == "maxlen"
            assert event_log._events.maxlen is not None
            event_log._events = deque(
                event_log._events,
                maxlen=event_log._events.maxlen + 1,
            )
        assert _signature(changed, macro_depth=0, throughput=0.0).canonical_bytes != (
            initial
        ), log_field

    event_env = clone_environment(env)
    event_env.simulator.events._events.append(
        Event(
            seq=1,
            time_s=0.0,
            clock="D1 00:00",
            level="info",
            code="unit.noncanonical",
            message="noncanonical event fixture",
        )
    )
    event_baseline = _signature(
        event_env,
        macro_depth=0,
        throughput=0.0,
    ).canonical_bytes
    for event_field in fields(Event):
        changed = clone_environment(event_env)
        event = changed.simulator.events._events[-1]
        value = getattr(event, event_field.name)
        replacement: object
        if isinstance(value, int):
            replacement = value + 1
        elif isinstance(value, float):
            replacement = math.nextafter(value, math.inf)
        elif event_field.name == "level":
            replacement = "warning"
        else:
            replacement = f"{value}.changed"
        changed.simulator.events._events[-1] = replace(
            event,
            **{event_field.name: replacement},
        )
        assert _signature(changed, macro_depth=0, throughput=0.0).canonical_bytes != (
            event_baseline
        ), event_field.name


@pytest.mark.parametrize("invalid_sequence", (False, 0.0, np.int64(0), -1, 1))
def test_state_signature_requires_exact_builtin_zero_fault_sequence(
    invalid_sequence: object,
) -> None:
    env = PlantEnvV2(episode=NONCANONICAL_TWO_DECISIONS)
    env.reset(seed=7)
    env.simulator.faults._sequence = invalid_sequence  # type: ignore[assignment]
    with pytest.raises(ValueError, match="canonical fault-free scope"):
        _signature(env, macro_depth=0, throughput=0.0)


def test_clone_allowlist_rejects_path_poisoning_and_mutable_closure() -> None:
    env = PlantEnvV2(episode=NONCANONICAL_TWO_DECISIONS)
    env.reset(seed=7)

    for poisoned in (
        {"apparently": "standalone"},
        env.metrics.reward_components,
        DeliveryStepResult(step=0, settlements=[]),  # type: ignore[arg-type]
        DeliveryStepResult(step=0, settlements=frozenset()),  # type: ignore[arg-type]
    ):
        env.simulator.delivery_ledger._current = poisoned  # type: ignore[assignment]
        with pytest.raises(
            TypeError,
            match="exact type|not transitively immutable",
        ):
            clone_environment(env)


def test_clone_allowlist_omits_null_compatibility_paths() -> None:
    compatibility_episode = replace(
        NONCANONICAL_TWO_DECISIONS,
        electrolyser_commitment=None,
    )
    env = PlantEnvV2(episode=compatibility_episode)
    env.reset(seed=7)

    cloned = clone_environment(env)
    audit_clone_aliases(env, cloned)
    assert cloned.episode_config.electrolyser_commitment is None
    assert (
        cloned.simulator.electrolyser._commitment_state
        is env.simulator.electrolyser._commitment_state
    )


def test_state_signature_requires_an_empty_fault_registry() -> None:
    env = PlantEnvV2(episode=NONCANONICAL_TWO_DECISIONS)
    env.reset(seed=7)
    env.simulator.faults._faults.append("unit.noncanonical")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="canonical fault-free scope"):
        _signature(env, macro_depth=0, throughput=0.0)


def test_state_signature_rejects_unbound_manifest_and_future_truth() -> None:
    env = PlantEnvV2(episode=NONCANONICAL_TWO_DECISIONS)
    env.reset(seed=7)
    manifest = env.audit_manifest()

    with pytest.raises(ValueError, match="manifest identity mismatch"):
        build_state_signature(
            env,
            manifest_sha256="f" * 64,
            exogenous_episode_sha256=manifest["exogenous_episode_sha256"],
            settings_sha256=_SETTINGS_SHA256,
            action_library_sha256=_ACTION_LIBRARY_SHA256,
            macro_depth=0,
            battery_throughput_kwh=0.0,
        )

    tape = env.simulator._solar_factor_tape
    assert tape is not None
    env.simulator._solar_factor_tape = SolarFactorTape(
        dt_s=tape.dt_s,
        boundary_factors=tape.boundary_factors,
    )
    with pytest.raises(ValueError, match="future-truth tape identity mismatch"):
        _signature(env, macro_depth=0, throughput=0.0)


def test_optimized_clone_matches_deepcopy_and_isolates_operational_state() -> None:
    parent = PlantEnvV2(episode=NONCANONICAL_TWO_DECISIONS)
    parent.reset(seed=11)
    parent_before = _signature(parent, macro_depth=0, throughput=0.0)
    parent_checkpoint_before = canonical_json(parent._checkpoint_contract)
    optimized = clone_environment(parent)
    sibling = clone_environment(parent)
    untouched_sibling = clone_environment(parent)
    untouched_before = _signature(
        untouched_sibling,
        macro_depth=0,
        throughput=0.0,
    )
    untouched_checkpoint_before = canonical_json(untouched_sibling._checkpoint_contract)
    ordinary = copy.deepcopy(parent)

    assert optimized.config is parent.config
    assert optimized._episode is not parent._episode
    assert optimized._episode is not None and parent._episode is not None
    assert sibling._episode is not None and untouched_sibling._episode is not None
    assert optimized._episode.stream_spawn_keys is not parent._episode.stream_spawn_keys
    assert optimized.metrics is not parent.metrics
    assert optimized.simulator.state is not parent.simulator.state
    assert optimized.control_state is not parent.control_state
    assert optimized._checkpoint_contract is not parent._checkpoint_contract
    _assert_checkpoint_containers_detached(
        parent._checkpoint_contract,
        optimized._checkpoint_contract,
    )
    _assert_checkpoint_containers_detached(
        parent._checkpoint_contract,
        sibling._checkpoint_contract,
    )
    assert optimized.simulator.components is not parent.simulator.components
    assert optimized.simulator.delivery_ledger is not parent.simulator.delivery_ledger
    assert optimized.simulator.delivery_ledger._pending is not (
        parent.simulator.delivery_ledger._pending
    )
    assert optimized.simulator.delivery_ledger._by_step is not (
        parent.simulator.delivery_ledger._by_step
    )
    assert optimized.simulator.delivery_ledger._order is not (
        parent.simulator.delivery_ledger._order
    )
    assert optimized.simulator.delivery_ledger._settlements is not (
        parent.simulator.delivery_ledger._settlements
    )
    assert optimized.simulator.events._events is not parent.simulator.events._events
    assert (
        optimized.simulator.events._conditions
        is not parent.simulator.events._conditions
    )
    assert optimized.simulator.faults._faults is not parent.simulator.faults._faults
    assert optimized.metrics.reward_components is not parent.metrics.reward_components
    assert optimized.metrics.electrolyser_mode_time_s is not (
        parent.metrics.electrolyser_mode_time_s
    )
    assert optimized.encoder._low is not parent.encoder._low
    assert optimized.encoder._high is not parent.encoder._high
    assert optimized.encoder._base._low is not parent.encoder._base._low
    assert optimized.encoder._base._high is not parent.encoder._base._high
    assert optimized.adapter._low is not parent.adapter._low
    assert optimized.adapter._high is not parent.adapter._high
    assert optimized.observation_space.low is not parent.observation_space.low
    assert optimized.observation_space.high is not parent.observation_space.high
    assert optimized.action_space.low is not parent.action_space.low
    assert optimized.action_space.high is not parent.action_space.high
    assert optimized.np_random is not parent.np_random
    assert optimized.np_random.bit_generator is not parent.np_random.bit_generator
    assert _signature(optimized, macro_depth=0, throughput=0.0).canonical_bytes == (
        _signature(ordinary, macro_depth=0, throughput=0.0).canonical_bytes
    )

    optimized.metrics.reward_components["noncanonical"] = 1.0
    assert "noncanonical" not in parent.metrics.reward_components
    assert "noncanonical" not in sibling.metrics.reward_components
    optimized.metrics.reward_components.pop("noncanonical")

    optimized._episode.stream_spawn_keys["unit.noncanonical"] = (999,)
    assert "unit.noncanonical" not in parent._episode.stream_spawn_keys
    assert "unit.noncanonical" not in sibling._episode.stream_spawn_keys
    optimized._episode.stream_spawn_keys.pop("unit.noncanonical")

    optimized.simulator.state.health["noncanonical"] = next(
        iter(optimized.simulator.state.health.values())
    )
    assert "noncanonical" not in parent.simulator.state.health
    assert "noncanonical" not in sibling.simulator.state.health
    optimized.simulator.state.health.pop("noncanonical")
    assert optimized.control_state is not None
    assert parent.control_state is not None
    assert sibling.control_state is not None
    optimized.control_state.boundary.health["noncanonical"] = next(
        iter(optimized.control_state.boundary.health.values())
    )
    assert "noncanonical" not in parent.control_state.boundary.health
    assert "noncanonical" not in sibling.control_state.boundary.health
    optimized.control_state.boundary.health.pop("noncanonical")
    optimized.metrics.electrolyser_mode_time_s["legacy"] = 123.0
    assert parent.metrics.electrolyser_mode_time_s["legacy"] == 0.0
    assert sibling.metrics.electrolyser_mode_time_s["legacy"] == 0.0
    optimized.metrics.electrolyser_mode_time_s["legacy"] = 0.0
    optimized.simulator.events._conditions["unit.nested"] = True
    assert "unit.nested" not in parent.simulator.events._conditions
    assert "unit.nested" not in sibling.simulator.events._conditions
    optimized.simulator.events._conditions.pop("unit.nested")
    optimized.simulator.events._events.append(
        Event(
            seq=0,
            time_s=0.0,
            clock="D1 00:00",
            level="info",
            code="unit.nested",
            message="noncanonical clone-isolation fixture",
        )
    )
    assert len(optimized.simulator.events._events) == (
        len(parent.simulator.events._events) + 1
    )
    assert len(sibling.simulator.events._events) == len(parent.simulator.events._events)
    optimized.simulator.events._events.pop()
    optimized.simulator.faults._faults.append("unit.nested")
    assert parent.simulator.faults._faults == []
    assert sibling.simulator.faults._faults == []
    optimized.simulator.faults._faults.pop()

    ledger = optimized.simulator.delivery_ledger
    parent_ledger = parent.simulator.delivery_ledger
    sibling_ledger = sibling.simulator.delivery_ledger
    assert ledger._commitments
    commitment = ledger._commitments[0]
    ledger._pending["unit.nested"] = commitment
    assert "unit.nested" not in parent_ledger._pending
    assert "unit.nested" not in sibling_ledger._pending
    ledger._pending.pop("unit.nested")
    ledger._by_step[-1] = (commitment,)
    assert -1 not in parent_ledger._by_step
    assert -1 not in sibling_ledger._by_step
    ledger._by_step.pop(-1)
    ledger._order["unit.nested"] = -1
    assert "unit.nested" not in parent_ledger._order
    assert "unit.nested" not in sibling_ledger._order
    ledger._order.pop("unit.nested")
    ledger._settlements.append(
        DeliverySettlement(
            id="unit.nested",
            due_step=1,
            requested_kg=0.0,
            delivered_kg=0.0,
            unmet_kg=0.0,
        )
    )
    assert parent_ledger._settlements == []
    assert sibling_ledger._settlements == []
    ledger._settlements.pop()

    optimized.simulator.components["unit.nested"] = optimized.simulator.solar
    assert "unit.nested" not in parent.simulator.components
    assert "unit.nested" not in sibling.simulator.components
    optimized.simulator.components.pop("unit.nested")
    optimized._checkpoint_contract["unit.nested"] = True
    assert "unit.nested" not in parent._checkpoint_contract
    assert "unit.nested" not in sibling._checkpoint_contract
    optimized._checkpoint_contract.pop("unit.nested")
    optimized._checkpoint_contract["interface_contract"]["observation_fields"][0][
        "unit.nested"
    ] = True
    assert (
        "unit.nested"
        not in parent._checkpoint_contract["interface_contract"]["observation_fields"][
            0
        ]
    )
    assert (
        "unit.nested"
        not in sibling._checkpoint_contract["interface_contract"]["observation_fields"][
            0
        ]
    )
    optimized._checkpoint_contract["interface_contract"]["observation_fields"][0].pop(
        "unit.nested"
    )
    anchor = optimized._checkpoint_contract["episode_distribution"][
        "electrolyser_commitment"
    ]["specific_energy_multiplier_anchors"][0]
    parent_anchor = parent._checkpoint_contract["episode_distribution"][
        "electrolyser_commitment"
    ]["specific_energy_multiplier_anchors"][0]
    original_anchor_value = anchor[0]
    anchor[0] = math.nextafter(original_anchor_value, math.inf)
    assert anchor[0] != parent_anchor[0]
    anchor[0] = original_anchor_value
    optimized.encoder._low[0] = 123.0
    assert parent.encoder._low[0] != 123.0
    optimized.encoder._low[0] = parent.encoder._low[0]
    optimized.adapter._low[0] = 123.0
    assert parent.adapter._low[0] != 123.0
    optimized.adapter._low[0] = parent.adapter._low[0]
    optimized.observation_space.low[0] = 123.0
    assert parent.observation_space.low[0] != 123.0
    optimized.observation_space.low[0] = parent.observation_space.low[0]
    optimized.action_space.low[0] = 123.0
    assert parent.action_space.low[0] != 123.0
    optimized.action_space.low[0] = parent.action_space.low[0]
    assert _signature(parent, macro_depth=0, throughput=0.0).canonical_bytes == (
        parent_before.canonical_bytes
    )

    optimized_total = sibling_total = ordinary_total = 0.0
    actions = (Action.all_to_battery(), Action.idle())
    for decision, action in enumerate(actions, start=1):
        optimized_result = optimized.step_action(action)
        sibling_result = sibling.step_action(action)
        ordinary_result = ordinary.step_action(action)
        optimized_total = accumulate_battery_throughput(
            optimized_total, optimized_result
        )
        sibling_total = accumulate_battery_throughput(sibling_total, sibling_result)
        ordinary_total = accumulate_battery_throughput(ordinary_total, ordinary_result)
        optimized_signature = _signature(
            optimized,
            macro_depth=decision,
            throughput=optimized_total,
        ).canonical_bytes
        assert (
            optimized_signature
            == _signature(
                sibling,
                macro_depth=decision,
                throughput=sibling_total,
            ).canonical_bytes
        )
        assert (
            optimized_signature
            == _signature(
                ordinary,
                macro_depth=decision,
                throughput=ordinary_total,
            ).canonical_bytes
        )
        assert _signature(parent, macro_depth=0, throughput=0.0).canonical_bytes == (
            parent_before.canonical_bytes
        )
        assert canonical_json(parent._checkpoint_contract) == parent_checkpoint_before
        assert (
            _signature(
                untouched_sibling,
                macro_depth=0,
                throughput=0.0,
            ).canonical_bytes
            == untouched_before.canonical_bytes
        )
        assert (
            canonical_json(untouched_sibling._checkpoint_contract)
            == untouched_checkpoint_before
        )

    assert parent.decisions_taken == 0
    assert untouched_sibling.decisions_taken == 0
    aliased = clone_environment(parent)
    aliased.metrics = parent.metrics
    with pytest.raises(ValueError, match="shared .*EpisodeMetricsV2"):
        audit_clone_aliases(parent, aliased)

    checkpoint_aliased = clone_environment(parent)
    checkpoint_aliased._checkpoint_contract["episode_distribution"]["battery_soc"] = (
        parent._checkpoint_contract["episode_distribution"]["battery_soc"]
    )
    with pytest.raises(ValueError, match="checkpoint container is shared"):
        audit_clone_aliases(parent, checkpoint_aliased)


def test_throughput_and_final_score_use_only_authoritative_sources() -> None:
    compatibility_off = replace(
        NONCANONICAL_TWO_DECISIONS,
        electrolyser_commitment=None,
    )
    env = PlantEnvV2(episode=compatibility_off)
    env.reset(seed=7)

    total = 0.0
    charge_result = env.step_action(Action.all_to_battery())
    total = accumulate_battery_throughput(total, charge_result)
    discharge_result = env.step_action(
        Action(
            battery_share=0.0,
            electrolyser_share=0.0,
            dac_share=0.0,
            methanator_share=0.0,
            battery_support_kw=4_000.0,
        )
    )
    total = accumulate_battery_throughput(total, discharge_result)

    charge = charge_result[4]["interval_power"]["applied_kwh"]
    discharge = discharge_result[4]["interval_power"]["applied_kwh"]
    expected = 0.0
    expected += charge["battery"]
    expected += charge["battery_support"]
    expected += discharge["battery"]
    expected += discharge["battery_support"]
    assert charge["battery"] > 0.0
    assert discharge["battery_support"] > 0.0
    assert total.hex() == expected.hex()

    score = extract_final_score(env, battery_throughput_kwh=total)
    assert score.delivered_kg == env.metrics.delivered_kg
    assert score.service_numerator == env.metrics.commitments_met
    assert score.service_denominator == env.metrics.commitments_settled
    assert score.worst_fill == env.metrics.worst_commitment_fill_rate
    expected_equivalent = env.simulator.state.ch4_level_kg + min(
        env.simulator.state.h2_level_kg / env.config.methanator.h2_kg_per_kg_ch4,
        env.simulator.state.co2_level_kg / env.config.methanator.co2_kg_per_kg_ch4,
    )
    assert score.terminal_ch4_equivalent_inventory_kg == expected_equivalent
    assert score.battery_throughput_kwh.hex() == total.hex()
    assert score.terminal_battery_energy_kwh == env.simulator.state.battery_energy_kwh
    assert score.forced_stops == env.metrics.electrolyser_forced_stops
    assert score.aborted_starts == env.metrics.electrolyser_aborted_starts
    assert score.cold_starts == env.metrics.electrolyser_cold_starts
    assert score.curtailed_kwh == env.metrics.curtailed_kwh

    env.metrics.terminal_battery_energy_kwh = math.nextafter(
        env.metrics.terminal_battery_energy_kwh,
        math.inf,
    )
    with pytest.raises(ValueError, match="terminal metric/truth mismatch"):
        extract_final_score(env, battery_throughput_kwh=total)


@pytest.mark.parametrize(
    ("metric_name", "state_name"),
    (
        ("terminal_h2_kg", "h2_level_kg"),
        ("terminal_co2_kg", "co2_level_kg"),
        ("terminal_ch4_kg", "ch4_level_kg"),
        ("terminal_battery_soc", "battery_soc"),
        ("terminal_battery_energy_kwh", "battery_energy_kwh"),
    ),
)
def test_final_score_rejects_every_metric_truth_mismatch(
    metric_name: str,
    state_name: str,
) -> None:
    env = PlantEnvV2(episode=NONCANONICAL_TWO_DECISIONS)
    env.reset(seed=7)
    env.step_action(Action.idle())
    env.step_action(Action.idle())
    setattr(
        env.metrics,
        metric_name,
        math.nextafter(getattr(env.simulator.state, state_name), math.inf),
    )
    with pytest.raises(ValueError, match="terminal metric/truth mismatch"):
        extract_final_score(env, battery_throughput_kwh=0.0)


def test_final_score_rejects_delivery_ledger_disagreement() -> None:
    env = PlantEnvV2(episode=NONCANONICAL_TWO_DECISIONS)
    env.reset(seed=7)
    env.step_action(Action.idle())
    env.step_action(Action.idle())
    env.simulator.delivery_ledger._cumulative_delivered_kg = math.nextafter(
        env.metrics.delivered_kg,
        math.inf,
    )
    with pytest.raises(ValueError, match="delivery ledger disagree"):
        extract_final_score(env, battery_throughput_kwh=0.0)


def test_throughput_rejects_missing_or_nonfinite_step_accounting() -> None:
    with pytest.raises(ValueError, match="interval_power"):
        accumulate_battery_throughput(0.0, {})
    with pytest.raises(ValueError, match="finite"):
        accumulate_battery_throughput(
            0.0,
            {
                "interval_power": {
                    "applied_kwh": {"battery": math.nan, "battery_support": 0.0}
                }
            },
        )
    with pytest.raises(ValueError, match="five values"):
        accumulate_battery_throughput(0.0, (None, None, None, None))
    with pytest.raises(TypeError, match="battery-support"):
        accumulate_battery_throughput(
            0.0,
            {"interval_power": {"applied_kwh": {"battery": 0.0}}},
        )
    with pytest.raises(ValueError, match=">= 0"):
        accumulate_battery_throughput(
            0.0,
            {
                "interval_power": {
                    "applied_kwh": {"battery": -1.0, "battery_support": 0.0}
                }
            },
        )
    with pytest.raises(TypeError, match="real number"):
        accumulate_battery_throughput(
            0.0,
            {
                "interval_power": {
                    "applied_kwh": {"battery": True, "battery_support": 0.0}
                }
            },
        )

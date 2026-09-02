"""Methane delivery kernel: boundary semantics, accounting, and V1 isolation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from plant_sim import (
    Action,
    DeliveryCommitment,
    InitialState,
    PlantConfig,
    PlantSimulator,
    Scenario,
    StorageConfig,
)
from plant_sim.config import SolarConfig

NOON = SolarConfig().sunrise_h + (SolarConfig().sunset_h - SolarConfig().sunrise_h) / 2


def _productive_simulator(*, initial_ch4_kg: float = 0.0) -> PlantSimulator:
    simulator = PlantSimulator(
        PlantConfig(
            ch4_tank=StorageConfig(capacity_kg=100.0, initial_kg=initial_ch4_kg),
        )
    )
    simulator.load_scenario(
        Scenario(
            id="delivery-test",
            name="Delivery test",
            description="Feedstocks and daylight isolate delivery boundary behaviour.",
            start_hour=NOON,
            initial=InitialState(h2_kg=100.0, co2_kg=500.0, ch4_kg=initial_ch4_kg),
        )
    )
    return simulator


def test_deadline_interval_production_is_available_before_settlement() -> None:
    simulator = _productive_simulator()
    simulator.set_delivery_schedule(
        [DeliveryCommitment(id="first", due_step=1, requested_kg=1.0)]
    )

    starting_ch4_kg = simulator.ch4_tank.level_kg
    state = simulator.step(Action(methanator_share=1.0))
    produced_kg = state.ch4_production_kg_h * simulator.config.dt_h
    delivery = simulator.delivery_step

    assert produced_kg > delivery.requested_kg
    assert delivery.delivered_kg == pytest.approx(1.0)
    assert delivery.unmet_kg == 0.0
    assert state.ch4_level_kg == pytest.approx(
        starting_ch4_kg + produced_kg - delivery.delivered_kg
    )


def test_delivery_shortfall_is_conserved_and_tank_never_goes_negative() -> None:
    simulator = PlantSimulator(
        PlantConfig(ch4_tank=StorageConfig(capacity_kg=10.0, initial_kg=1.25))
    )
    simulator.set_delivery_schedule(
        [DeliveryCommitment(id="short", due_step=1, requested_kg=3.0)]
    )

    starting_ch4_kg = simulator.ch4_tank.level_kg
    state = simulator.step(Action.idle())
    delivery = simulator.delivery_step
    produced_kg = state.ch4_production_kg_h * simulator.config.dt_h

    assert delivery.delivered_kg + delivery.unmet_kg == pytest.approx(
        delivery.requested_kg
    )
    assert state.ch4_level_kg == pytest.approx(
        starting_ch4_kg + produced_kg - delivery.delivered_kg
    )
    assert state.ch4_level_kg == 0.0
    assert simulator.ch4_tank.level_kg >= 0.0


def test_multiple_same_tick_commitments_settle_once_in_schedule_order() -> None:
    simulator = PlantSimulator(
        PlantConfig(ch4_tank=StorageConfig(capacity_kg=10.0, initial_kg=3.0))
    )
    simulator.set_delivery_schedule(
        [
            DeliveryCommitment(id="a", due_step=1, requested_kg=2.0),
            DeliveryCommitment(id="b", due_step=1, requested_kg=4.0),
        ]
    )

    simulator.step(Action.idle())
    due_result = simulator.delivery_step
    assert [item.id for item in due_result.settlements] == ["a", "b"]
    assert [item.delivered_kg for item in due_result.settlements] == pytest.approx([2.0, 1.0])
    assert [item.unmet_kg for item in due_result.settlements] == pytest.approx([0.0, 3.0])
    assert due_result.requested_kg == pytest.approx(6.0)
    assert due_result.delivered_kg == pytest.approx(3.0)
    assert due_result.unmet_kg == pytest.approx(3.0)

    simulator.step(Action.idle())
    no_op = simulator.delivery_step
    assert no_op.step == 2
    assert no_op.settlements == ()
    assert (no_op.requested_kg, no_op.delivered_kg, no_op.unmet_kg) == (0.0, 0.0, 0.0)
    assert no_op.cumulative_requested_kg == pytest.approx(6.0)
    assert no_op.cumulative_delivered_kg == pytest.approx(3.0)
    assert no_op.cumulative_unmet_kg == pytest.approx(3.0)
    assert len(simulator.delivery_audit()["settlements"]) == 2
    assert simulator.ch4_tank.level_kg == 0.0


def test_reset_and_scenario_load_clear_schedule_and_delivery_counters() -> None:
    simulator = PlantSimulator(
        PlantConfig(ch4_tank=StorageConfig(capacity_kg=10.0, initial_kg=2.0))
    )
    simulator.set_delivery_schedule(
        [DeliveryCommitment(id="reset-me", due_step=1, requested_kg=1.0)]
    )
    simulator.step(Action.idle())
    assert simulator.delivery_step.cumulative_delivered_kg == pytest.approx(1.0)

    simulator.reset()
    audit = simulator.delivery_audit()
    assert audit["schedule"] == []
    assert audit["settlements"] == []
    assert audit["pending_count"] == 0
    assert audit["cumulative_requested_kg"] == 0.0
    assert audit["cumulative_delivered_kg"] == 0.0
    assert audit["cumulative_unmet_kg"] == 0.0
    assert simulator.delivery_step.step == 0
    assert simulator.next_delivery is None

    simulator.set_delivery_schedule(
        [DeliveryCommitment(id="load-me", due_step=4, requested_kg=1.0)]
    )
    simulator.load_scenario("clear_day")
    assert simulator.delivery_audit()["schedule"] == []
    assert simulator.delivery_step == simulator.delivery_ledger.current
    assert simulator.next_delivery is None


@pytest.mark.parametrize("due_step", [0, -1, 1.0, True])
def test_commitment_requires_a_positive_integer_tick(due_step: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        DeliveryCommitment(id="bad-tick", due_step=due_step, requested_kg=1.0)


@pytest.mark.parametrize("requested_kg", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_commitment_requires_a_finite_nonnegative_amount(requested_kg: float) -> None:
    with pytest.raises(ValueError):
        DeliveryCommitment(id="bad-mass", due_step=1, requested_kg=requested_kg)


def test_schedule_ids_are_unique_and_failed_replacement_is_atomic() -> None:
    simulator = PlantSimulator()
    simulator.set_delivery_schedule(
        [DeliveryCommitment(id="kept", due_step=3, requested_kg=4.0)]
    )
    before = simulator.delivery_audit()

    with pytest.raises(ValueError, match="unique"):
        simulator.set_delivery_schedule(
            [
                DeliveryCommitment(id="duplicate", due_step=1, requested_kg=1.0),
                DeliveryCommitment(id="duplicate", due_step=2, requested_kg=2.0),
            ]
        )

    assert simulator.delivery_audit() == before


def test_schedule_deadlines_are_relative_to_episode_not_install_call() -> None:
    simulator = PlantSimulator()
    simulator.step(Action.idle())

    with pytest.raises(ValueError, match="current episode step is 1"):
        simulator.set_delivery_schedule(
            [DeliveryCommitment(id="already-due", due_step=1, requested_kg=1.0)]
        )

    simulator.set_delivery_schedule(
        [DeliveryCommitment(id="future", due_step=2, requested_kg=0.0)]
    )
    simulator.step(Action.idle())
    assert [item.id for item in simulator.delivery_step.settlements] == ["future"]


def test_frozen_schedule_cannot_erase_accounting_and_reset_unlocks_it() -> None:
    simulator = PlantSimulator(
        PlantConfig(ch4_tank=StorageConfig(capacity_kg=10.0, initial_kg=2.0))
    )
    simulator.set_delivery_schedule(
        [DeliveryCommitment(id="immutable", due_step=1, requested_kg=1.0)],
        freeze=True,
    )
    simulator.step(Action.idle())
    before = simulator.delivery_audit()

    assert before["frozen"] is True
    with pytest.raises(RuntimeError, match="frozen"):
        simulator.set_delivery_schedule(
            [DeliveryCommitment(id="replacement", due_step=2, requested_kg=1.0)]
        )
    assert simulator.delivery_audit() == before

    simulator.reset()
    assert simulator.delivery_audit()["frozen"] is False
    simulator.set_delivery_schedule(
        [DeliveryCommitment(id="new-episode", due_step=1, requested_kg=0.0)]
    )


def test_views_are_immutable_and_audit_serialization_is_detached() -> None:
    simulator = PlantSimulator()
    simulator.set_delivery_schedule(
        [
            {"id": "later", "due_step": 2, "requested_kg": 2},
            {"id": "next", "due_step": 1, "requested_kg": 1},
        ]
    )

    assert simulator.next_delivery == DeliveryCommitment("next", 1, 1.0)
    with pytest.raises(FrozenInstanceError):
        simulator.next_delivery.requested_kg = 99.0  # type: ignore[misc, union-attr]

    audit = simulator.delivery_audit()
    audit["schedule"].clear()
    audit["current"]["next_commitment"] = None
    assert len(simulator.delivery_audit()["schedule"]) == 2
    assert simulator.next_delivery is not None


def test_empty_or_absent_schedule_leaves_v1_rollouts_identical() -> None:
    untouched = PlantSimulator()
    explicit_empty = PlantSimulator()
    explicit_empty.set_delivery_schedule([])
    actions = [
        Action(battery_share=0.3, electrolyser_share=0.4, dac_share=0.2, methanator_share=0.1),
        Action(battery_share=0.1, electrolyser_share=0.6, dac_share=0.1, methanator_share=0.2),
        Action.idle(),
    ] * 30

    untouched_states = [untouched.step(action).to_dict() for action in actions]
    empty_states = [explicit_empty.step(action).to_dict() for action in actions]

    assert untouched_states == empty_states
    assert untouched.events.events == explicit_empty.events.events
    assert "delivery" not in untouched.state.to_dict()
    assert all("delivery" not in key for key in untouched.state.to_dict())
    assert untouched.delivery_step.settlements == ()
    assert explicit_empty.delivery_step.settlements == ()

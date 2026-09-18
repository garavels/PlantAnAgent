"""Focused checks for the beta constrained-RL transition-cost seam."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from plant_backend.rl.v2.beta_config import (
    CycleSettlementV1,
    CyclicMarketTaskConfigV1,
    canonical_sha256,
)
from plant_backend.rl.v2.beta_constraint_transition import (
    CYCLE_CONSTRAINT_RESOURCE_ORDER,
    CYCLE_CONSTRAINT_TRANSITION_SCHEMA,
    CycleConstraintTransitionV1,
)
from plant_backend.rl.v2.beta_disturbances import DisturbanceConfigV1
from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.market import MarketConfigV1


IDLE_ACTION = np.full(4, -1.0, dtype=np.float32)
FIXTURE_SEED = 9_000_001


def _settlement(*, boundary_decision: int = 96) -> CycleSettlementV1:
    return CycleSettlementV1(
        cycle_index=1,
        boundary_decision=boundary_decision,
        targets={"battery": 100.0, "h2": 10.0, "co2": 4.0, "ch4": 5.0},
        actual={"battery": 90.0, "h2": 8.0, "co2": 3.6, "ch4": 2.5},
        deficits={"battery": 10.0, "h2": 2.0, "co2": 0.4, "ch4": 2.5},
        tolerances={"battery": 1.0, "h2": 0.5, "co2": 0.5, "ch4": 0.25},
        feasible=False,
    )


def _canonical_body(payload: dict[str, object]) -> dict[str, object]:
    body = dict(payload)
    del body["sha256"]
    return body


def test_non_settlement_record_is_immutable_canonical_exact_zero() -> None:
    record = CycleConstraintTransitionV1.from_settlement(
        decision_index=0,
        settlement=None,
        episode_cycle_count=7,
    )
    payload = record.to_dict()

    assert payload == {
        "schema": CYCLE_CONSTRAINT_TRANSITION_SCHEMA,
        "decision_index": 0,
        "cycle_settled": False,
        "cycle_index": None,
        "episode_cycle_count": 7,
        "resource_order": ["battery", "h2", "co2", "ch4"],
        "raw_cost": [0.0, 0.0, 0.0, 0.0],
        "training_cost": [0.0, 0.0, 0.0, 0.0],
        "raw_cost_sum": 0.0,
        "training_cost_sum": 0.0,
        "sha256": record.sha256,
    }
    assert record.resource_order == CYCLE_CONSTRAINT_RESOURCE_ORDER
    assert record.sha256 == canonical_sha256(_canonical_body(payload))
    with pytest.raises(FrozenInstanceError):
        record.decision_index = 1  # type: ignore[misc]


def test_settlement_record_reconstructs_all_four_costs_and_scalar_checksum() -> None:
    settlement = _settlement()
    record = CycleConstraintTransitionV1.from_settlement(
        decision_index=settlement.boundary_decision,
        settlement=settlement,
        episode_cycle_count=7,
    )
    expected_raw = (0.09, 0.15, 0.0, 0.45)

    assert record.cycle_settled is True
    assert record.cycle_index == settlement.cycle_index
    assert record.resource_order == ("battery", "h2", "co2", "ch4")
    assert record.raw_cost == pytest.approx(expected_raw)
    assert record.training_cost == pytest.approx(
        tuple(value / 7.0 for value in expected_raw)
    )
    assert record.raw_cost_sum == settlement.total_constraint_cost
    assert record.raw_cost_sum == math.fsum(record.raw_cost)
    assert record.training_cost_sum == math.fsum(record.training_cost)
    assert record.sha256 == canonical_sha256(record.body())


def test_transition_record_rejects_noncanonical_or_inconsistent_values() -> None:
    with pytest.raises(ValueError, match="boundary_decision"):
        CycleConstraintTransitionV1.from_settlement(
            decision_index=95,
            settlement=_settlement(boundary_decision=96),
            episode_cycle_count=7,
        )
    with pytest.raises(ValueError, match="episode_cycle_count"):
        CycleConstraintTransitionV1.from_settlement(
            decision_index=0,
            settlement=None,
            episode_cycle_count=0,
        )
    with pytest.raises(ValueError, match="resource_order"):
        CycleConstraintTransitionV1(
            decision_index=1,
            cycle_settled=False,
            cycle_index=None,
            episode_cycle_count=7,
            resource_order=("h2", "battery", "co2", "ch4"),
            raw_cost=(0.0, 0.0, 0.0, 0.0),
            training_cost=(0.0, 0.0, 0.0, 0.0),
        )
    with pytest.raises(ValueError, match="training_cost"):
        CycleConstraintTransitionV1(
            decision_index=1,
            cycle_settled=False,
            cycle_index=None,
            episode_cycle_count=7,
            resource_order=CYCLE_CONSTRAINT_RESOURCE_ORDER,
            raw_cost=(0.0, 0.0, 0.0, 0.0),
            training_cost=(0.1, 0.0, 0.0, 0.0),
        )
    with pytest.raises(ValueError, match="non-settlement raw_cost"):
        CycleConstraintTransitionV1(
            decision_index=1,
            cycle_settled=False,
            cycle_index=None,
            episode_cycle_count=7,
            resource_order=CYCLE_CONSTRAINT_RESOURCE_ORDER,
            raw_cost=(7.0, 0.0, 0.0, 0.0),
            training_cost=(1.0, 0.0, 0.0, 0.0),
        )


def test_environment_emits_exact_record_on_reset_every_step_and_terminal() -> None:
    episode = EpisodeConfigV2(
        horizon_h=24.0,
        decision_interval_s=3_600.0,
        start_hour=6.0,
        delivery_period_h=24.0,
        delivery_rate_kg_h=1.0,
        battery_soc=(0.5, 0.5),
        h2_fill=(0.1, 0.1),
        co2_fill=(0.1, 0.1),
        ch4_coverage_h=(1.0, 1.0),
    )
    beta_task = CyclicMarketTaskConfigV1(
        market=MarketConfigV1(forecast_ensemble_members=10),
        disturbances=DisturbanceConfigV1(enabled=False),
    )
    environment = PlantCyclicMarketEnvV2Beta(
        episode=episode,
        beta_task=beta_task,
    )
    try:
        _, reset_info = environment.reset(seed=FIXTURE_SEED)
        reset_transition = reset_info["cycle_constraint_transition"]
        assert reset_transition["decision_index"] == 0
        assert reset_transition["episode_cycle_count"] == 1
        assert reset_transition["cycle_settled"] is False
        assert reset_transition["cycle_index"] is None
        assert reset_transition["raw_cost"] == [0.0, 0.0, 0.0, 0.0]
        assert reset_transition["training_cost"] == [0.0, 0.0, 0.0, 0.0]
        assert reset_transition["sha256"] == canonical_sha256(
            _canonical_body(reset_transition)
        )

        terminal_result = None
        for decision_index in range(1, 25):
            terminal_result = environment.step(IDLE_ACTION.copy())
            transition = terminal_result[4]["cycle_constraint_transition"]
            assert transition["decision_index"] == decision_index
            assert transition["resource_order"] == ["battery", "h2", "co2", "ch4"]
            assert transition["sha256"] == canonical_sha256(_canonical_body(transition))
            if decision_index < 24:
                assert transition["cycle_settled"] is False
                assert transition["cycle_index"] is None
                assert transition["raw_cost"] == [0.0, 0.0, 0.0, 0.0]
                assert transition["training_cost"] == [0.0, 0.0, 0.0, 0.0]

        assert terminal_result is not None
        _, _, terminated, truncated, terminal_info = terminal_result
        assert terminated is True
        assert truncated is False
        terminal_transition = terminal_info["cycle_constraint_transition"]
        assert terminal_transition["cycle_settled"] is True
        assert terminal_transition["cycle_index"] == 1
        assert any(value > 0.0 for value in terminal_transition["raw_cost"])
        settlement = environment.cycle_settlements[-1]
        assert (
            terminal_transition
            == CycleConstraintTransitionV1.from_settlement(
                decision_index=24,
                settlement=settlement,
                episode_cycle_count=1,
            ).to_dict()
        )
        assert terminal_info["cycle_constraint"]["settled_this_step"] == (
            settlement.to_dict()
        )
        assert terminal_transition["raw_cost_sum"] == (settlement.total_constraint_cost)
        assert (
            terminal_info["cycle_constraint"]["cumulative_constraint_cost"]
            == terminal_transition["raw_cost_sum"]
        )
    finally:
        environment.close()

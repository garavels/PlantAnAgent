"""Outcome-free tests for the controller-neutral absolute-setpoint boundary."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from plant_sim import BatteryConfig, PlantConfig

from plant_backend.controllers.base import Constraints
from plant_backend.rl.v2.setpoint_control import (
    PROCESS_SETPOINT_ACTION_NAMES,
    PROCESS_SETPOINT_ACTION_SIZE,
    DecodedProcessSetpointActionV1,
    ProcessSetpointActionAdapterV1,
    ProcessSetpointTargetsV1,
    allocate_process_setpoint_targets,
)
from plant_backend.safety.validation import BatterySupportAvailability


def _targets(
    *,
    electrolyser_kw: float = 600.0,
    dac_kw: float = 400.0,
    methanator_kw: float = 200.0,
    battery_charge_kw: float = 500.0,
) -> ProcessSetpointTargetsV1:
    return ProcessSetpointTargetsV1(
        electrolyser_target_kw=electrolyser_kw,
        dac_target_kw=dac_kw,
        methanator_target_kw=methanator_kw,
        battery_charge_ceiling_kw=battery_charge_kw,
    )


def _support(
    capacity_kw: float = 300.0,
    *,
    headroom_kw: float = 3_000.0,
    battery_available: bool = True,
    electrolyser_available: bool = True,
) -> BatterySupportAvailability:
    available_kw = (
        min(capacity_kw, headroom_kw)
        if battery_available and electrolyser_available
        else 0.0
    )
    return BatterySupportAvailability(
        available_kw=available_kw,
        support_limit_kw=capacity_kw,
        reserve_soc=0.15,
        usable_energy_kwh=1_000.0,
        energy_limited_kw=capacity_kw,
        electrolyser_headroom_kw=headroom_kw,
        battery_available=battery_available,
        electrolyser_available=electrolyser_available,
    )


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("electrolyser_target_kw", True, TypeError, "real number"),
        ("dac_target_kw", "1", TypeError, "real number"),
        ("methanator_target_kw", float("nan"), ValueError, "finite"),
        ("battery_charge_ceiling_kw", float("inf"), ValueError, "finite"),
        ("electrolyser_target_kw", -1.0, ValueError, "must be >= 0"),
    ],
)
def test_targets_reject_invalid_values(
    field: str,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        replace(_targets(), **{field: value})


@pytest.mark.parametrize(
    ("action", "error", "message"),
    [
        ([0.0, 0.0, 0.0], ValueError, "exactly four"),
        ([[0.0, 0.0], [0.0, 0.0]], ValueError, "exactly four"),
        ([0.0, True, 0.0, 0.0], TypeError, "real number"),
        ([0.0, "0", 0.0, 0.0], TypeError, "real number"),
        ([0.0, float("nan"), 0.0, 0.0], ValueError, "finite"),
        ([0.0, float("inf"), 0.0, 0.0], ValueError, "finite"),
    ],
)
def test_adapter_rejects_invalid_policy_vectors(
    action: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        ProcessSetpointActionAdapterV1().decode(
            action,  # type: ignore[arg-type]
            constraints=Constraints(),
            config=PlantConfig(),
        )


@pytest.mark.parametrize(
    ("coordinate", "expected"),
    [
        (-1.0, (0.0, 0.0, 0.0, 0.0)),
        (0.0, (1_500.0, 600.0, 90.0, 1_250.0)),
        (1.0, (3_000.0, 1_200.0, 180.0, 2_500.0)),
    ],
)
def test_adapter_maps_policy_endpoints_to_active_limits(
    coordinate: float,
    expected: tuple[float, float, float, float],
) -> None:
    adapter = ProcessSetpointActionAdapterV1()
    decoded = adapter.decode(
        np.full(PROCESS_SETPOINT_ACTION_SIZE, coordinate, dtype=np.float32),
        constraints=Constraints(),
        config=PlantConfig(),
    )

    assert adapter.names == PROCESS_SETPOINT_ACTION_NAMES
    assert adapter.low().dtype == np.float32
    assert adapter.high().dtype == np.float32
    assert decoded.raw_policy_vector == (coordinate,) * 4
    assert decoded.bounded_policy_vector == (coordinate,) * 4
    assert decoded.clipped is False
    assert (
        decoded.targets.electrolyser_target_kw,
        decoded.targets.dac_target_kw,
        decoded.targets.methanator_target_kw,
        decoded.targets.battery_charge_ceiling_kw,
    ) == expected


def test_adapter_retains_clipping_and_resolves_operator_and_equipment_limits() -> None:
    adapter = ProcessSetpointActionAdapterV1()
    config = replace(
        PlantConfig(),
        battery=BatteryConfig(max_charge_kw=700.0),
    )
    constraints = Constraints(
        electrolyser_max_kw=900.0,
        dac_max_kw=500.0,
        methanator_max_kw=100.0,
    )

    decoded = adapter.decode(
        [2.0, -2.0, 1.0, 1.0],
        constraints=constraints,
        config=config,
    )

    assert decoded == DecodedProcessSetpointActionV1(
        raw_policy_vector=(2.0, -2.0, 1.0, 1.0),
        bounded_policy_vector=(1.0, -1.0, 1.0, 1.0),
        targets=ProcessSetpointTargetsV1(
            electrolyser_target_kw=900.0,
            dac_target_kw=0.0,
            methanator_target_kw=100.0,
            battery_charge_ceiling_kw=700.0,
        ),
        clipped=True,
    )
    assert len(decoded.sha256) == 64


def test_surplus_allocation_tracks_targets_and_charges_from_the_remainder() -> None:
    allocation = allocate_process_setpoint_targets(
        _targets(),
        solar_available_kw=2_000.0,
        support_availability=_support(),
    )

    assert allocation.routes_kw == pytest.approx((500.0, 600.0, 400.0, 200.0))
    assert allocation.support_kw == 0.0
    assert allocation.unallocated_kw == pytest.approx(300.0)
    assert allocation.electrolyser_offered_kw == pytest.approx(600.0)
    assert allocation.target_reachable is True


def test_scarcity_preserves_unsupported_load_proportions_and_uses_support() -> None:
    allocation = allocate_process_setpoint_targets(
        _targets(),
        solar_available_kw=500.0,
        support_availability=_support(),
    )

    assert allocation.routes_kw == pytest.approx((0.0, 300.0, 400.0 / 3.0, 200.0 / 3.0))
    assert allocation.routes_kw[2] / allocation.routes_kw[3] == pytest.approx(2.0)
    assert allocation.support_kw == pytest.approx(300.0)
    assert allocation.unallocated_kw == pytest.approx(0.0)
    assert allocation.target_reachable is True
    assert allocation.action.battery_share == 0.0


def test_night_and_reserve_bound_allocations_report_reachability_exactly() -> None:
    reachable = allocate_process_setpoint_targets(
        _targets(
            electrolyser_kw=300.0,
            dac_kw=0.0,
            methanator_kw=0.0,
            battery_charge_kw=0.0,
        ),
        solar_available_kw=0.0,
        support_availability=_support(),
    )
    unreachable = allocate_process_setpoint_targets(
        _targets(
            electrolyser_kw=600.0,
            dac_kw=0.0,
            methanator_kw=0.0,
            battery_charge_kw=0.0,
        ),
        solar_available_kw=0.0,
        support_availability=_support(),
    )
    reserve_bound = allocate_process_setpoint_targets(
        _targets(
            electrolyser_kw=600.0,
            dac_kw=0.0,
            methanator_kw=0.0,
            battery_charge_kw=0.0,
        ),
        solar_available_kw=100.0,
        support_availability=_support(battery_available=False),
    )

    assert reachable.electrolyser_offered_kw == 300.0
    assert reachable.target_reachable is True
    assert unreachable.electrolyser_offered_kw == 300.0
    assert unreachable.target_reachable is False
    assert reserve_bound.electrolyser_offered_kw == 100.0
    assert reserve_bound.target_reachable is False


def test_allocation_is_deterministic_and_rejects_a_lowered_charge_ceiling() -> None:
    first = allocate_process_setpoint_targets(
        _targets(),
        solar_available_kw=2_000.0,
        support_availability=_support(),
    )
    second = allocate_process_setpoint_targets(
        _targets(),
        solar_available_kw=2_000.0,
        support_availability=_support(),
    )

    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.sha256 == second.sha256
    with pytest.raises(ValueError, match="charge ceiling"):
        replace(
            first,
            targets=replace(first.targets, battery_charge_ceiling_kw=0.0),
        )

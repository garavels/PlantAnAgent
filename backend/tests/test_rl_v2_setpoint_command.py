"""Outcome-free tests for common observation-bound setpoint commands."""

from __future__ import annotations

from dataclasses import replace

import pytest
from plant_sim import PlantConfig, PlantSimulator, PlantState

from plant_backend.controllers.base import Constraints
from plant_backend.rl.v2.setpoint_command import (
    ProcessSetpointCommandV1,
    ProcessSetpointEnvelopeV1,
    dispatch_process_setpoint_command,
    issue_process_setpoint_command,
    process_setpoint_boundary_sha256,
    resolve_process_setpoint_envelope,
)
from plant_backend.rl.v2.setpoint_control import ProcessSetpointTargetsV1


_OBSERVATION_SHA = "a" * 64
_SOURCE_SHA = "b" * 64


@pytest.fixture
def boundary() -> PlantState:
    initial = PlantSimulator().get_telemetry()
    return replace(
        initial,
        battery_soc=0.90,
        battery_energy_kwh=0.90 * initial.battery_capacity_kwh,
    )


def _targets(
    *,
    electrolyser_kw: float = 600.0,
    dac_kw: float = 200.0,
    methanator_kw: float = 100.0,
    battery_charge_kw: float = 300.0,
) -> ProcessSetpointTargetsV1:
    return ProcessSetpointTargetsV1(
        electrolyser_target_kw=electrolyser_kw,
        dac_target_kw=dac_kw,
        methanator_target_kw=methanator_kw,
        battery_charge_ceiling_kw=battery_charge_kw,
    )


def _command(
    *,
    targets: ProcessSetpointTargetsV1 | None = None,
    constraints: Constraints | None = None,
    config: PlantConfig | None = None,
) -> ProcessSetpointCommandV1:
    return issue_process_setpoint_command(
        targets or _targets(),
        decision_index=0,
        observation_sha256=_OBSERVATION_SHA,
        hold_duration_s=900.0,
        issuer_id="test-controller-v1",
        source_schema="test-controller-decision-v1",
        source_sha256=_SOURCE_SHA,
        constraints=constraints or Constraints(),
        config=config or PlantConfig(),
    )


def test_resolved_envelope_binds_equipment_operator_and_reserve_limits() -> None:
    plant = PlantConfig()
    constraints = Constraints(
        electrolyser_max_kw=900.0,
        dac_max_kw=500.0,
        methanator_max_kw=100.0,
        battery_support_max_kw=400.0,
        battery_min_soc=0.30,
    )

    envelope = resolve_process_setpoint_envelope(
        constraints=constraints,
        config=plant,
    )

    assert envelope == ProcessSetpointEnvelopeV1(
        electrolyser_limit_kw=900.0,
        dac_limit_kw=500.0,
        methanator_limit_kw=100.0,
        battery_charge_limit_kw=2_500.0,
        battery_support_limit_kw=400.0,
        battery_reserve_soc=0.30,
    )
    assert len(envelope.sha256) == 64


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"decision_index": -1}, ValueError, "non-negative integer"),
        ({"decision_index": True}, ValueError, "non-negative integer"),
        ({"observation_sha256": "bad"}, ValueError, "SHA-256"),
        ({"hold_duration_s": 0.0}, ValueError, "positive"),
        ({"hold_duration_s": float("nan")}, ValueError, "finite"),
        ({"issuer_id": ""}, ValueError, "non-empty"),
        ({"source_schema": "  "}, ValueError, "non-empty"),
        ({"source_sha256": "bad"}, ValueError, "SHA-256"),
    ],
)
def test_command_rejects_invalid_identity_or_timing(
    changes: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        replace(_command(), **changes)


@pytest.mark.parametrize(
    ("target", "value", "message"),
    [
        ("electrolyser_target_kw", 3_001.0, "electrolyser"),
        ("dac_target_kw", 1_201.0, "DAC"),
        ("methanator_target_kw", 181.0, "methanator"),
        ("battery_charge_ceiling_kw", 2_501.0, "battery charge"),
    ],
)
def test_command_rejects_targets_above_active_envelope(
    target: str,
    value: float,
    message: str,
) -> None:
    invalid = replace(_targets(), **{target: value})
    with pytest.raises(ValueError, match=message):
        _command(targets=invalid)


def test_command_is_deterministic_and_binds_complete_source_evidence() -> None:
    first = _command()
    second = _command()

    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.sha256 == second.sha256
    assert first.to_dict()["issuer"] == {
        "id": "test-controller-v1",
        "source_schema": "test-controller-decision-v1",
        "source_sha256": _SOURCE_SHA,
    }


def test_common_dispatch_tracks_surplus_and_deficit_through_one_kernel(
    boundary: PlantState,
) -> None:
    command = _command()
    surplus = dispatch_process_setpoint_command(
        command,
        boundary,
        solar_available_kw=2_000.0,
        constraints=Constraints(),
        config=PlantConfig(),
    )
    deficit = dispatch_process_setpoint_command(
        command,
        boundary,
        solar_available_kw=500.0,
        constraints=Constraints(),
        config=PlantConfig(),
    )

    assert surplus.command_sha256 == command.sha256
    assert surplus.routes_kw == pytest.approx((300.0, 600.0, 200.0, 100.0))
    assert surplus.support_kw == 0.0
    assert surplus.allocation.unallocated_kw == pytest.approx(800.0)
    assert deficit.routes_kw == pytest.approx((0.0, 200.0, 200.0, 100.0))
    assert deficit.support_kw == pytest.approx(400.0)
    assert deficit.action == deficit.allocation.action
    assert deficit.targets == command.targets
    assert deficit.target_reachable is True


def test_common_dispatch_is_boundary_bound_and_rejects_stale_authority(
    boundary: PlantState,
) -> None:
    command = _command()
    first = dispatch_process_setpoint_command(
        command,
        boundary,
        solar_available_kw=723.5,
        constraints=Constraints(),
        config=PlantConfig(),
    )
    second = dispatch_process_setpoint_command(
        command,
        replace(boundary),
        solar_available_kw=723.5,
        constraints=Constraints(),
        config=PlantConfig(),
    )

    assert first == second
    assert first.sha256 == second.sha256
    assert first.boundary_sha256 == process_setpoint_boundary_sha256(boundary)
    with pytest.raises(ValueError, match="envelope changed"):
        dispatch_process_setpoint_command(
            command,
            boundary,
            solar_available_kw=723.5,
            constraints=Constraints(battery_min_soc=0.30),
            config=PlantConfig(),
        )


def test_common_dispatch_rejects_invalid_boundary_or_solar(
    boundary: PlantState,
) -> None:
    with pytest.raises(TypeError, match="boundary"):
        dispatch_process_setpoint_command(
            _command(),
            object(),  # type: ignore[arg-type]
            solar_available_kw=1.0,
            constraints=Constraints(),
            config=PlantConfig(),
        )
    with pytest.raises(ValueError, match="must be >= 0"):
        dispatch_process_setpoint_command(
            _command(),
            boundary,
            solar_available_kw=-1.0,
            constraints=Constraints(),
            config=PlantConfig(),
        )

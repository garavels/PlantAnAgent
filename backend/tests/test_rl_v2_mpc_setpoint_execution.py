"""Opt-in environment execution and fail-closed evidence for MPC setpoints."""

from __future__ import annotations

from dataclasses import replace

import pytest
from plant_sim import (
    DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE,
    Action,
    PlantConfig,
    PlantSimulator,
    SolarFactorTape,
)

from plant_backend.controllers import Constraints
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.env import PlantEnvV2
from plant_backend.rl.v2.mpc_execution import public_observation_sha256
from plant_backend.rl.v2.mpc_forecast import public_forecast_issue_sha256
from plant_backend.rl.v2.mpc_schema import canonical_json
from plant_backend.rl.v2.mpc_setpoint_execution import (
    MPCMinuteApplicationV1,
    MPCSetpointIntervalExecutionV1,
    record_minute_application,
)
from plant_backend.rl.v2.mpc_setpoints import (
    MPCProcessSetpointsV1,
    dispatch_mpc_setpoints,
    resolve_setpoint_envelope,
)
from plant_backend.safety import validate


SHORT = EpisodeConfigV2(horizon_h=2.0, delivery_period_h=1.0)
_SHA = "a" * 64


def _setpoints(
    *,
    config: PlantConfig,
    constraints: Constraints,
    decision_index: int = 0,
    observation_sha256: str = _SHA,
    public_forecast_sha256: str = "b" * 64,
    hold_duration_s: float = 900.0,
    electrolyser_kw: float = 600.0,
    dac_kw: float = 200.0,
    methanator_kw: float = 100.0,
    battery_charge_kw: float = 300.0,
    expected_support_kw: float = 0.0,
    planned_curtailment_kw: float = 100.0,
) -> MPCProcessSetpointsV1:
    solar_electrolyser_kw = electrolyser_kw - expected_support_kw
    forecast_kw = (
        solar_electrolyser_kw
        + dac_kw
        + methanator_kw
        + battery_charge_kw
        + planned_curtailment_kw
    )
    return MPCProcessSetpointsV1(
        decision_index=decision_index,
        observation_sha256=observation_sha256,
        public_forecast_sha256=public_forecast_sha256,
        projection_sha256="c" * 64,
        first_interval_sha256="d" * 64,
        settings_sha256="e" * 64,
        model_sha256="f" * 64,
        plan_sha256="1" * 64,
        hold_duration_s=hold_duration_s,
        electrolyser_target_kw=electrolyser_kw,
        dac_target_kw=dac_kw,
        methanator_target_kw=methanator_kw,
        battery_charge_ceiling_kw=battery_charge_kw,
        expected_battery_support_kw=expected_support_kw,
        planned_forecast_solar_kw=forecast_kw,
        planned_solar_to_battery_kw=battery_charge_kw,
        planned_solar_to_electrolyser_kw=solar_electrolyser_kw,
        planned_solar_to_dac_kw=dac_kw,
        planned_solar_to_methanator_kw=methanator_kw,
        planned_solar_curtailed_kw=planned_curtailment_kw,
        envelope=resolve_setpoint_envelope(
            constraints=constraints,
            config=config,
        ),
    )


def _bound_setpoints(
    env: PlantEnvV2,
    **targets: float,
) -> MPCProcessSetpointsV1:
    observation = env.controller_observation
    assert observation.forecast is not None
    return _setpoints(
        config=env.config,
        constraints=env.constraints,
        decision_index=env.decisions_taken,
        observation_sha256=public_observation_sha256(observation),
        public_forecast_sha256=public_forecast_issue_sha256(observation.forecast),
        **targets,
    )


def _record_one(
    simulator: PlantSimulator,
    setpoints: MPCProcessSetpointsV1,
    *,
    ordinal: int,
    constraints: Constraints,
) -> MPCMinuteApplicationV1:
    before = simulator.get_telemetry()
    commitment_before = simulator.electrolyser_commitment
    solar_kw = simulator.next_dispatch_solar_kw()
    dispatch = dispatch_mpc_setpoints(
        setpoints,
        before,
        solar_available_kw=solar_kw,
        constraints=constraints,
        config=simulator.config,
    )
    guarded = validate(
        dispatch.action,
        before,
        constraints=constraints,
        config=simulator.config,
        dispatch_solar_available_kw=solar_kw,
    )
    after = simulator.step(guarded.action)
    return record_minute_application(
        ordinal=ordinal,
        before=before,
        dispatch=dispatch,
        validated=guarded,
        after=after,
        next_boundary=simulator.get_telemetry(),
        commitment_before=commitment_before,
        commitment_after=simulator.electrolyser_commitment,
    )


def test_historical_action_doors_do_not_emit_setpoint_fields() -> None:
    env = PlantEnvV2(episode=SHORT)
    env.reset(seed=3)

    _, _, _, _, info = env.step_action(Action.idle())

    assert "setpoint_execution" not in info
    assert "dispatcher_elapsed_ns" not in info
    assert info["requested"]["action"] == Action.idle().to_dict()
    assert env.last_setpoint_execution is None


def test_setpoint_door_emits_ordered_deterministic_minute_ledger() -> None:
    left = PlantEnvV2(episode=SHORT)
    right = PlantEnvV2(episode=SHORT)
    left.reset(seed=7)
    right.reset(seed=7)
    left_command = _bound_setpoints(left, electrolyser_kw=900.0)
    right_command = _bound_setpoints(right, electrolyser_kw=900.0)

    left_result = left.step_setpoints(left_command)
    right_result = right.step_setpoints(right_command)
    left_info = left_result[4]
    right_info = right_result[4]
    left_execution = left_info["setpoint_execution"]
    right_execution = right_info["setpoint_execution"]

    assert left_command == right_command
    assert left_execution == right_execution
    assert left.last_setpoint_execution is not None
    assert left.last_setpoint_execution.to_dict() == left_execution
    assert left_execution["physical_steps"] == 15
    assert [item["ordinal"] for item in left_execution["applications"]] == list(
        range(15)
    )
    assert all(
        right_item["dispatch"]["boundary_sha256"]
        == left_item["next_boundary"]["sha256"]
        for left_item, right_item in zip(
            left_execution["applications"][:-1],
            left_execution["applications"][1:],
            strict=True,
        )
    )
    assert len(left_execution["sha256"]) == 64
    assert len(left_info["dispatcher_elapsed_ns"]) == 15
    assert all(value >= 0 for value in left_info["dispatcher_elapsed_ns"])
    assert (
        left_execution["energy_kwh"]["electrolyser_target_error"]["requested_overshoot"]
        == 0.0
    )
    assert (
        left_execution["instantaneous_electrolyser_target_error_kw"][
            "requested_max_overshoot"
        ]
        == 0.0
    )
    assert all(
        not (
            item["dispatch"]["request_kw"]["solar_to_battery"] > 0.0
            and item["dispatch"]["request_kw"]["battery_support"] > 0.0
        )
        for item in left_execution["applications"]
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"decision_index": 1}, "decision index"),
        ({"observation_sha256": "0" * 64}, "public observation"),
        ({"public_forecast_sha256": "0" * 64}, "forecast issue"),
        ({"hold_duration_s": 60.0}, "hold duration"),
    ],
)
def test_setpoint_door_rejects_stale_identity_before_mutation(
    change: dict[str, object],
    message: str,
) -> None:
    env = PlantEnvV2(episode=SHORT)
    env.reset(seed=11)
    command = replace(_bound_setpoints(env), **change)  # type: ignore[arg-type]
    state_before = env.simulator.state
    metrics_before = env.metrics.to_dict()

    with pytest.raises(ValueError, match=message):
        env.step_setpoints(command)

    assert env.simulator.state == state_before
    assert env.physics_steps_taken == 0
    assert env.decisions_taken == 0
    assert env.metrics.to_dict() == metrics_before


def test_setpoint_door_rejects_stale_authority_envelope_before_mutation() -> None:
    env = PlantEnvV2(episode=SHORT)
    env.reset(seed=12)
    command = _bound_setpoints(env)
    stale_envelope = replace(
        command.envelope,
        battery_reserve_soc=command.envelope.battery_reserve_soc / 2.0,
    )

    with pytest.raises(ValueError, match="authority envelope"):
        env.step_setpoints(replace(command, envelope=stale_envelope))

    assert env.physics_steps_taken == 0
    assert env.decisions_taken == 0


def test_guard_and_component_refusal_remain_separate_causes() -> None:
    guard_env = PlantEnvV2(episode=SHORT)
    guard_env.reset(seed=5)
    guard_env.simulator.inject_fault({"type": "dac_outage"})
    guard_env.step_action(Action.idle())
    guarded_command = _bound_setpoints(
        guard_env,
        electrolyser_kw=0.0,
        dac_kw=200.0,
        methanator_kw=0.0,
        battery_charge_kw=0.0,
    )
    guarded_info = guard_env.step_setpoints(guarded_command)[4]["setpoint_execution"]

    assert guarded_info["counts"]["guard_changed_request"] > 0
    assert guarded_info["energy_kwh"]["applied"]["dac"] == 0.0
    assert any(
        "DAC is offline" in adjustment
        for item in guarded_info["applications"]
        for adjustment in item["guard"]["adjustments"]
    )

    refusal_env = PlantEnvV2(episode=SHORT)
    refusal_env.reset(seed=5)
    refusal_command = _bound_setpoints(
        refusal_env,
        electrolyser_kw=1_500.0,
        dac_kw=0.0,
        methanator_kw=0.0,
        battery_charge_kw=0.0,
    )
    refusal_info = refusal_env.step_setpoints(refusal_command)[4]["setpoint_execution"]

    assert refusal_info["counts"]["guard_changed_request"] == 0
    assert refusal_info["counts"]["plant_refused_or_rerouted_power"] > 0
    assert (
        refusal_info["energy_kwh"]["electrolyser_target_error"]["applied_undershoot"]
        > 0.0
    )
    assert (
        refusal_info["energy_kwh"]["electrolyser_target_error"]["applied_overshoot"]
        == 0.0
    )


def test_private_later_truth_cannot_change_command_or_same_minute_dispatch() -> None:
    left = PlantEnvV2(episode=SHORT)
    right = PlantEnvV2(episode=SHORT)
    left.reset(seed=19)
    right.reset(seed=19)
    left_command = _bound_setpoints(left, electrolyser_kw=1_200.0)
    right_command = _bound_setpoints(right, electrolyser_kw=1_200.0)
    assert right._episode is not None
    tape = right._episode.weather.tape
    altered = list(tape.boundary_factors)
    altered[2:16] = [1.0 - value for value in altered[2:16]]
    right.simulator._solar_factor_tape = SolarFactorTape(
        dt_s=tape.dt_s,
        boundary_factors=tuple(altered),
    )

    left_execution = left.step_setpoints(left_command)[4]["setpoint_execution"]
    right_execution = right.step_setpoints(right_command)[4]["setpoint_execution"]

    assert left_command.sha256 == right_command.sha256
    assert (
        left_execution["applications"][0]["dispatch"]
        == right_execution["applications"][0]["dispatch"]
    )
    assert left_execution != right_execution


def test_minute_and_interval_evidence_reject_substitution_and_chain_breaks() -> None:
    config = PlantConfig()
    constraints = Constraints()
    setpoints = _setpoints(
        config=config,
        constraints=constraints,
        hold_duration_s=120.0,
    )
    simulator = PlantSimulator(config)
    applications = tuple(
        _record_one(
            simulator,
            setpoints,
            ordinal=index,
            constraints=constraints,
        )
        for index in range(2)
    )
    execution = MPCSetpointIntervalExecutionV1(
        setpoints=setpoints,
        physics_step_s=config.dt_s,
        applications=applications,
    )

    assert execution.to_dict() == execution.to_dict()
    with pytest.raises(ValueError, match="boundary steps"):
        wrong_step = applications[0].next_boundary_record
        wrong_step["step"] = applications[0].next_boundary_step + 1
        replace(
            applications[0],
            next_boundary_json=canonical_json(wrong_step),
        )
    with pytest.raises(ValueError, match="does not balance"):
        replace(
            applications[0],
            applied_electrolyser_kw=setpoints.electrolyser_target_kw + 1.0,
        )
    with pytest.raises(ValueError, match="guard increased"):
        replace(
            applications[0],
            guarded_action=replace(
                applications[0].guarded_action,
                electrolyser_share=0.1,
            ),
        )
    with pytest.raises(ValueError, match="another setpoint"):
        MPCSetpointIntervalExecutionV1(
            setpoints=replace(setpoints, plan_sha256="2" * 64),
            physics_step_s=config.dt_s,
            applications=applications,
        )
    with pytest.raises(ValueError, match="chain is broken"):
        substituted_boundary = applications[0].next_boundary_record
        substituted_boundary["clock"] = "substituted"
        MPCSetpointIntervalExecutionV1(
            setpoints=setpoints,
            physics_step_s=config.dt_s,
            applications=(
                replace(
                    applications[0],
                    next_boundary_json=canonical_json(substituted_boundary),
                ),
                applications[1],
            ),
        )

    committed_simulator = PlantSimulator(
        config,
        electrolyser_commitment=DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE,
    )
    committed_applications = tuple(
        _record_one(
            committed_simulator,
            setpoints,
            ordinal=index,
            constraints=constraints,
        )
        for index in range(2)
    )
    assert (
        committed_applications[0].commitment_before
        != committed_applications[0].commitment_after
    )
    with pytest.raises(ValueError, match="chain is broken"):
        MPCSetpointIntervalExecutionV1(
            setpoints=setpoints,
            physics_step_s=config.dt_s,
            applications=(
                replace(
                    committed_applications[0],
                    commitment_after=committed_applications[0].commitment_before,
                ),
                committed_applications[1],
            ),
        )

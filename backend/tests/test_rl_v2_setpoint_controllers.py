"""Transparent heuristic and cross-controller common-authority tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from plant_backend.controllers.base import Constraints
from plant_backend.controllers.heuristic import HeuristicController
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.control import ControllerObservationV2
from plant_backend.rl.v2.env import PlantEnvV2
from plant_backend.rl.v2.mpc_execution import public_observation_sha256
from plant_backend.rl.v2.mpc_setpoints import (
    MPCProcessSetpointsV1,
    resolve_setpoint_envelope,
)
from plant_backend.rl.v2.setpoint_command import (
    adapt_mpc_process_setpoints,
    dispatch_process_setpoint_command,
    issue_process_setpoint_command,
)
from plant_backend.rl.v2.setpoint_controllers import (
    LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID,
    LegacyHeuristicSetpointControllerV1,
)
from plant_backend.rl.v2.setpoint_control import ProcessSetpointActionAdapterV1


TINY = EpisodeConfigV2(horizon_h=0.25, delivery_period_h=0.25)


def _full_tank_observation(
    env: PlantEnvV2,
    *,
    solar_kw: float,
) -> ControllerObservationV2:
    observation = env.controller_observation
    boundary = replace(
        observation.state.boundary,
        solar_available_kw=solar_kw,
        solar_potential_kw=solar_kw,
        solar_output_factor=1.0,
        battery_soc=0.90,
        h2_fill=0.99,
        co2_fill=0.99,
        ch4_fill=0.99,
    )
    return replace(
        observation,
        state=replace(observation.state, boundary=boundary),
    )


def test_heuristic_adapter_preserves_a0_decision_and_binds_its_source() -> None:
    env = PlantEnvV2(episode=TINY)
    env.reset(seed=23)
    observation = env.controller_observation
    original = HeuristicController().compute_action(
        observation.state.boundary,
        constraints=env.constraints,
        config=env.config,
    )
    controller = LegacyHeuristicSetpointControllerV1()

    decision = controller.compute_command(
        observation,
        constraints=env.constraints,
        config=env.config,
        commitment_profile=env.episode_config.electrolyser_commitment,
    )

    assert decision.evidence.source_action == original.action
    assert decision.evidence.source_notes == original.notes
    assert decision.evidence.observation_sha256 == public_observation_sha256(
        observation
    )
    assert decision.command.issuer_id == LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID
    assert decision.command.source_schema == decision.evidence.schema
    assert decision.command.source_sha256 == decision.evidence.sha256
    assert decision.command.targets == decision.evidence.issued_targets
    assert controller.to_dict()["behavioral_equivalence_claim"] is False

    _, _, _, _, info = env.step_process_setpoints(
        decision.command,
        controller_invalid=decision.invalid,
    )
    assert (
        info["process_setpoint_execution"]["command"]["sha256"]
        == decision.command.sha256
    )
    attributions = [
        item["plant"]["battery_charge_attribution"]
        for item in info["process_setpoint_execution"]["applications"]
    ]
    assert any(item["recovered_or_guard_redirect_kw"] > 0.0 for item in attributions)


def test_heuristic_adapter_caps_only_direct_battery_authority() -> None:
    env = PlantEnvV2(episode=TINY)
    env.reset(seed=29)
    observation = _full_tank_observation(env, solar_kw=5_000.0)

    decision = LegacyHeuristicSetpointControllerV1().compute_command(
        observation,
        constraints=env.constraints,
        config=env.config,
    )

    assert decision.evidence.source_action.battery_share == 1.0
    assert decision.evidence.requested_targets.battery_charge_ceiling_kw == 5_000.0
    assert (
        decision.evidence.issued_targets.battery_charge_ceiling_kw
        == env.config.battery.max_charge_kw
    )
    assert decision.evidence.battery_charge_limited is True
    assert decision.command.targets == decision.evidence.issued_targets


def test_heuristic_evidence_rejects_target_substitution() -> None:
    env = PlantEnvV2(episode=TINY)
    env.reset(seed=31)
    decision = LegacyHeuristicSetpointControllerV1().compute_command(
        env.controller_observation,
        constraints=env.constraints,
        config=env.config,
    )

    with pytest.raises(ValueError, match="reproduce the A0 action"):
        replace(
            decision.evidence,
            requested_targets=replace(
                decision.evidence.requested_targets,
                dac_target_kw=(decision.evidence.requested_targets.dac_target_kw + 1.0),
            ),
        )


def test_heuristic_reset_restores_deterministic_decision_identity() -> None:
    env = PlantEnvV2(episode=TINY)
    env.reset(seed=37)
    controller = LegacyHeuristicSetpointControllerV1()
    first = controller.compute_command(
        env.controller_observation,
        constraints=env.constraints,
        config=env.config,
    )
    controller.reset()
    repeated = controller.compute_command(
        env.controller_observation,
        constraints=env.constraints,
        config=env.config,
    )

    assert repeated == first
    assert repeated.to_dict() == first.to_dict()


def test_heuristic_normalized_and_mpc_sources_share_exact_physical_authority() -> None:
    env = PlantEnvV2(episode=TINY)
    env.reset(seed=41)
    observation = _full_tank_observation(env, solar_kw=2_500.0)
    observation_sha256 = public_observation_sha256(observation)
    constraints = Constraints()

    heuristic = (
        LegacyHeuristicSetpointControllerV1()
        .compute_command(
            observation,
            constraints=constraints,
            config=env.config,
        )
        .command
    )

    decoded = ProcessSetpointActionAdapterV1().decode(
        np.asarray([-1.0, -1.0, -1.0, 1.0], dtype=np.float32),
        constraints=constraints,
        config=env.config,
    )
    normalized = issue_process_setpoint_command(
        decoded.targets,
        decision_index=0,
        observation_sha256=observation_sha256,
        hold_duration_s=observation.decision_interval_s,
        issuer_id="normalized-test-policy-v1",
        source_schema=decoded.schema,
        source_sha256=decoded.sha256,
        constraints=constraints,
        config=env.config,
    )

    mpc_source = MPCProcessSetpointsV1(
        decision_index=0,
        observation_sha256=observation_sha256,
        public_forecast_sha256="1" * 64,
        projection_sha256="2" * 64,
        first_interval_sha256="3" * 64,
        settings_sha256="4" * 64,
        model_sha256="5" * 64,
        plan_sha256="6" * 64,
        hold_duration_s=observation.decision_interval_s,
        electrolyser_target_kw=0.0,
        dac_target_kw=0.0,
        methanator_target_kw=0.0,
        battery_charge_ceiling_kw=2_500.0,
        expected_battery_support_kw=0.0,
        planned_forecast_solar_kw=2_500.0,
        planned_solar_to_battery_kw=2_500.0,
        planned_solar_to_electrolyser_kw=0.0,
        planned_solar_to_dac_kw=0.0,
        planned_solar_to_methanator_kw=0.0,
        planned_solar_curtailed_kw=0.0,
        envelope=resolve_setpoint_envelope(
            constraints=constraints,
            config=env.config,
        ),
    )
    mpc = adapt_mpc_process_setpoints(mpc_source)

    assert heuristic.targets == normalized.targets == mpc.targets
    assert heuristic.envelope == normalized.envelope == mpc.envelope
    assert (
        len({heuristic.source_sha256, normalized.source_sha256, mpc.source_sha256}) == 3
    )

    dispatches = tuple(
        dispatch_process_setpoint_command(
            command,
            observation.state.boundary,
            solar_available_kw=2_500.0,
            constraints=constraints,
            config=env.config,
        )
        for command in (heuristic, normalized, mpc)
    )
    assert dispatches[0].allocation == dispatches[1].allocation
    assert dispatches[1].allocation == dispatches[2].allocation

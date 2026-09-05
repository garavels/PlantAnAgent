"""Common-command and fail-closed tests for tariff-aware economic MPC."""

from __future__ import annotations

from dataclasses import replace

import pytest

from plant_backend.rl.v2.beta_mpc_schema import BetaMPCSettingsV1
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.economic_env import PlantEconomicSetpointEnvV2
from plant_backend.rl.v2.economic_mpc_controller import (
    TariffAwareEconomicMPCControllerV1,
)
from plant_backend.rl.v2.economic_mpc_schema import (
    ECONOMIC_MPC_CONTROLLER_ID,
    EconomicMPCSettingsV1,
    EconomicMPCSolveError,
)
from plant_backend.rl.v2.setpoint_command import ProcessSetpointCommandV1
import plant_backend.rl.v2.economic_mpc_controller as controller_module


def _pair():  # type: ignore[no-untyped-def]
    env = PlantEconomicSetpointEnvV2(
        episode=EpisodeConfigV2(
            horizon_h=1.0,
            delivery_period_h=1.0,
            delivery_rate_kg_h=35.0,
        )
    )
    env.reset(seed=0)
    controller = TariffAwareEconomicMPCControllerV1(
        EconomicMPCSettingsV1(task_horizon_h=1.0, planning_horizon_h=1.0),
        env.economic_assumptions,
    )
    return env, controller


def test_alpha_controller_rejects_beta_settings_identity() -> None:
    with pytest.raises(TypeError, match="settings must be EconomicMPCSettingsV1"):
        TariffAwareEconomicMPCControllerV1(BetaMPCSettingsV1())


def test_controller_issues_an_executable_controller_neutral_command() -> None:
    pytest.importorskip("scipy")
    env, controller = _pair()

    decision = controller.compute_command(
        env.economic_controller_observation,
        constraints=env.constraints,
        config=env.config,
        commitment_profile=env.episode_config.electrolyser_commitment,
    )

    assert isinstance(decision.command, ProcessSetpointCommandV1)
    assert decision.command.issuer_id == ECONOMIC_MPC_CONTROLLER_ID
    assert decision.command.source_sha256 == decision.evidence.sha256
    assert decision.evidence.plan.predicted_economic_cost_eur >= 0.0
    _, _, terminated, truncated, info = env.step_process_setpoints(decision.command)
    assert terminated is False
    assert truncated is False
    assert info["reward"]["objective_version"] == 1


def test_descriptor_declares_scope_mismatch_and_no_fallback() -> None:
    _, controller = _pair()
    descriptor = controller.to_dict()

    assert descriptor["id"] == ECONOMIC_MPC_CONTROLLER_ID
    assert "no controller-internal fallback" in str(descriptor["failure_policy"])
    assert "low-load" in str(descriptor["model_mismatch_declaration"])
    assert descriptor["output_schema"] == "process-setpoint-command-v1"


def test_decision_evidence_binds_plan_inputs_and_first_command_targets() -> None:
    pytest.importorskip("scipy")
    env, controller = _pair()
    decision = controller.compute_command(
        env.economic_controller_observation,
        constraints=env.constraints,
        config=env.config,
        commitment_profile=env.episode_config.electrolyser_commitment,
    )

    with pytest.raises(ValueError, match="settings digest"):
        replace(decision.evidence, settings_sha256="0" * 64)
    with pytest.raises(ValueError, match="first economic plan step"):
        replace(
            decision.evidence,
            targets=replace(
                decision.evidence.targets,
                battery_charge_ceiling_kw=(
                    decision.evidence.targets.battery_charge_ceiling_kw + 1.0
                ),
            ),
        )


def test_tariff_summary_mismatch_fails_before_a_command() -> None:
    env, controller = _pair()
    observation = env.economic_controller_observation
    corrupted = replace(
        observation,
        tariff=replace(
            observation.tariff,
            current_eur_per_mwh=observation.tariff.current_eur_per_mwh + 1.0,
        ),
    )

    with pytest.raises(ValueError, match="full schedule"):
        controller.compute_command(
            corrupted,
            constraints=env.constraints,
            config=env.config,
            commitment_profile=env.episode_config.electrolyser_commitment,
        )
    assert controller.decision_events == ()


def test_solver_failure_is_not_replaced_by_another_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, controller = _pair()

    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise EconomicMPCSolveError("deliberate solver failure")

    monkeypatch.setattr(controller_module, "solve_economic_mpc_model", fail)
    with pytest.raises(EconomicMPCSolveError, match="deliberate"):
        controller.compute_command(
            env.economic_controller_observation,
            constraints=env.constraints,
            config=env.config,
            commitment_profile=env.episode_config.electrolyser_commitment,
        )
    assert controller.decision_events == ()


def test_controller_canonicalizes_highs_boundary_noise_before_issuing_targets() -> None:
    """Regression: an optimal 24 h vector contained -4.55e-13 kW at a bound."""

    pytest.importorskip("scipy")
    env = PlantEconomicSetpointEnvV2(
        episode=EpisodeConfigV2(
            horizon_h=24.0,
            delivery_period_h=6.0,
            delivery_rate_kg_h=100.0,
        )
    )
    env.reset(seed=905)
    controller = TariffAwareEconomicMPCControllerV1(
        EconomicMPCSettingsV1(
            task_horizon_h=24.0,
            planning_horizon_h=24.0,
        ),
        env.economic_assumptions,
    )

    decision = controller.compute_command(
        env.economic_controller_observation,
        constraints=env.constraints,
        config=env.config,
        commitment_profile=env.episode_config.electrolyser_commitment,
    )

    assert decision.evidence.plan.horizon_intervals == 96
    targets = decision.command.targets
    assert (
        min(
            targets.electrolyser_target_kw,
            targets.dac_target_kw,
            targets.methanator_target_kw,
            targets.battery_charge_ceiling_kw,
        )
        >= 0.0
    )
    assert decision.evidence.plan.primary_delivered_optimum_kg > 0.0

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from plant_sim import Action, PlantConfig

from plant_backend.rl.v2 import (
    DayAheadTariffV1,
    EconomicAssumptionsV1,
    EconomicIntervalOutcomeV1,
    PlantEconomicSetpointEnvV2,
    PlantEnvV2,
    PlantSetpointEnvV2,
    compute_economic_reward,
)
from plant_backend.rl.v2 import economic_spaces, setpoint_spaces
from plant_backend.rl.v2.boundary import CommitmentExposureV2
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.economics import MAX_ABS_TARIFF_EUR_PER_MWH
from plant_backend.rl.v2.mpc_schema import ServiceMPCSetpointSettingsV1
from plant_backend.rl.v2.mpc_setpoint_controller import ServiceMPCSetpointControllerV1


def _one_interval_episode(*, start_hour: float = 12.0) -> EpisodeConfigV2:
    return EpisodeConfigV2(
        horizon_h=0.25,
        decision_interval_s=900.0,
        start_hour=start_hour,
        delivery_period_h=0.25,
        delivery_rate_kg_h=1.0,
        battery_soc=(0.25, 0.25),
        h2_fill=(0.0, 0.0),
        co2_fill=(0.0, 0.0),
        ch4_coverage_h=(0.0, 0.0),
    )


def _outcome(
    *,
    price: float = 100.0,
    accepted_solar_kwh: float = 0.0,
    battery_throughput_kwh: float = 0.0,
    initial_ch4: float = 100.0,
    before_ch4: float = 100.0,
    after_ch4: float = 100.0,
    initial_battery: float = 1_000.0,
    before_battery: float = 1_000.0,
    after_battery: float = 1_000.0,
    commitment: CommitmentExposureV2 | None = None,
) -> EconomicIntervalOutcomeV1:
    return EconomicIntervalOutcomeV1(
        opportunity_price_eur_per_mwh=price,
        accepted_solar_kwh=accepted_solar_kwh,
        delivered_kg=0.0,
        unmet_kg=0.0,
        battery_throughput_kwh=battery_throughput_kwh,
        initial_ch4_equivalent_kg=initial_ch4,
        ch4_equivalent_before_kg=before_ch4,
        ch4_equivalent_after_kg=after_ch4,
        initial_battery_energy_kwh=initial_battery,
        battery_energy_before_kwh=before_battery,
        battery_energy_after_kwh=after_battery,
        commitment=commitment or CommitmentExposureV2(),
    )


def test_tariff_observation_integrates_exactly_across_midnight() -> None:
    prices = tuple(float(hour) for hour in range(24))
    tariff = DayAheadTariffV1(id="hour-index", hourly_eur_per_mwh=prices)

    observation = tariff.observe(23.5)

    assert observation.current_eur_per_mwh == 23.0
    assert observation.next_1h_mean_eur_per_mwh == pytest.approx(11.5)
    assert observation.next_6h_min_eur_per_mwh == 0.0
    assert observation.next_6h_max_eur_per_mwh == 23.0
    assert observation.next_6h_mean_eur_per_mwh == pytest.approx(4.0)


def test_tariff_and_assumption_validation_fail_closed() -> None:
    with pytest.raises(ValueError, match="exactly 24"):
        DayAheadTariffV1(hourly_eur_per_mwh=(1.0,)).validate()
    with pytest.raises(ValueError, match="at most"):
        DayAheadTariffV1(
            hourly_eur_per_mwh=(MAX_ABS_TARIFF_EUR_PER_MWH + 1.0,) * 24
        ).validate()
    with pytest.raises(ValueError, match="must exceed"):
        replace(
            EconomicAssumptionsV1(),
            unmet_delivery_penalty_eur_per_kg=4.0,
        ).validate()


def test_negative_price_is_visible_but_has_zero_opportunity_cost() -> None:
    result = compute_economic_reward(
        _outcome(price=-25.0, accepted_solar_kwh=100.0),
        config=PlantConfig(),
        assumptions=EconomicAssumptionsV1(),
    )

    assert result.signals["opportunity_price_eur_per_mwh"] == -25.0
    assert result.signals["effective_opportunity_price_eur_per_mwh"] == 0.0
    assert result.components_eur["solar_opportunity_cost"] == 0.0


def test_endowment_liability_is_dense_and_telescopes() -> None:
    config = PlantConfig()
    assumptions = EconomicAssumptionsV1()
    depletion = compute_economic_reward(
        _outcome(before_ch4=100.0, after_ch4=90.0),
        config=config,
        assumptions=assumptions,
    )
    restoration = compute_economic_reward(
        _outcome(before_ch4=90.0, after_ch4=100.0),
        config=config,
        assumptions=assumptions,
    )

    key = "methane_endowment_liability_change"
    assert depletion.components_eur[key] == -40.0
    assert restoration.components_eur[key] == 40.0
    assert depletion.components_eur[key] + restoration.components_eur[key] == 0.0


def test_battery_and_electrolyser_exposure_increase_cost() -> None:
    config = PlantConfig()
    assumptions = EconomicAssumptionsV1()
    exposure = CommitmentExposureV2(
        enabled=True,
        cold_starts=1,
        operating_time_s=3_600.0,
        low_load_time_s=3_600.0,
    )
    result = compute_economic_reward(
        _outcome(battery_throughput_kwh=1_000.0, commitment=exposure),
        config=config,
        assumptions=assumptions,
    )

    expected_battery = 1_000.0 * assumptions.battery_throughput_eur_per_kwh
    expected_equivalent_h = (
        1.0
        + assumptions.electrolyser_low_load_wear_multiplier
        - 1.0
        + assumptions.cold_start_equivalent_operating_h
    )
    expected_electrolyser = (
        expected_equivalent_h
        * config.electrolyser.rated_kw
        * assumptions.electrolyser_capital_eur_per_kw
        / assumptions.electrolyser_life_h
    )
    assert result.components_eur["battery_throughput_reserve"] == pytest.approx(
        -expected_battery
    )
    assert result.components_eur["electrolyser_life_reserve"] == pytest.approx(
        -expected_electrolyser
    )


def test_environment_has_distinct_interface_and_preserves_physical_episode() -> None:
    episode = _one_interval_episode()
    historical = PlantEnvV2(episode=episode)
    setpoint = PlantSetpointEnvV2(episode=episode)
    economic = PlantEconomicSetpointEnvV2(episode=episode)

    old_observation, old_info = historical.reset(seed=17)
    setpoint_observation, setpoint_info = setpoint.reset(seed=17)
    economic_observation, economic_info = economic.reset(seed=17)

    assert old_observation.shape == setpoint_observation.shape == (57,)
    assert economic_observation.shape == (62,)
    assert historical.action_space.shape == (5,)
    assert setpoint.action_space.shape == economic.action_space.shape == (4,)
    assert old_info["environment_id"] != setpoint_info["environment_id"]
    assert setpoint_info["environment_id"] == setpoint_spaces.ENVIRONMENT_ID
    assert economic_info["environment_id"] == economic_spaces.ENVIRONMENT_ID
    assert economic.checkpoint_contract["execution_doors"][
        "legacy_share_action"
    ].startswith("rejected")
    assert (
        len(
            {
                historical.checkpoint_contract_digest,
                setpoint.checkpoint_contract_digest,
                economic.checkpoint_contract_digest,
            }
        )
        == 3
    )
    limitations = economic_spaces.describe()["limitations"]
    assert all("five-field share action" not in item for item in limitations)
    assert all(
        "economic and wear pressure not added" not in item for item in limitations
    )
    manifests = (
        historical.audit_manifest(),
        setpoint.audit_manifest(),
        economic.audit_manifest(),
    )
    assert len({item["exogenous_episode_sha256"] for item in manifests}) == 1


def test_observation_appends_exact_public_tariff_features() -> None:
    environment = PlantEconomicSetpointEnvV2(episode=_one_interval_episode())
    observation, info = environment.reset(seed=3)
    expected = environment.economic_assumptions.tariff.observe(12.0)
    structured = environment.economic_controller_observation

    np.testing.assert_allclose(
        observation[-5:],
        np.asarray(expected.normalized(), dtype=np.float32),
    )
    assert structured.plant == environment.controller_observation
    assert structured.tariff == expected
    assert structured.tariff_id == environment.economic_assumptions.tariff.id
    assert len(structured.sha256) == 64
    assert info["economic_context"]["tariff"] == expected.to_dict()
    assert environment.observation_space.contains(observation)


def test_step_replaces_scalar_reward_but_retains_legacy_diagnostics() -> None:
    environment = PlantEconomicSetpointEnvV2(episode=_one_interval_episode())
    environment.reset(seed=5)
    action = np.asarray([-1.0, -1.0, -1.0, 1.0], dtype=np.float32)

    _, reward, terminated, truncated, info = environment.step(action)

    assert terminated is True
    assert truncated is False
    assert reward == info["reward"]["total_eur"]
    assert reward == pytest.approx(sum(info["reward"]["components_eur"].values()))
    assert info["legacy_reward"]["diagnostic_only"] is True
    assert info["legacy_reward"]["breakdown"]["objective_version"] == 3
    assert info["economic_metrics"]["decisions"] == 1
    assert info["economic_metrics"]["net_operating_value_eur"] == reward
    expected_energy_cost = (
        info["economic_interval"]["accepted_solar_kwh"]
        * max(0.0, info["economic_interval"]["opportunity_price_eur_per_mwh"])
        / 1_000.0
    )
    assert info["reward"]["components_eur"]["solar_opportunity_cost"] == pytest.approx(
        -expected_energy_cost
    )


def test_mpc_evidence_door_uses_economic_reward_and_remains_reconcilable() -> None:
    environment = PlantEconomicSetpointEnvV2(episode=_one_interval_episode())
    environment.reset(seed=7)
    controller = ServiceMPCSetpointControllerV1(
        ServiceMPCSetpointSettingsV1(
            task_horizon_h=0.25,
            planning_horizon_h=0.25,
        )
    )
    decision = controller.compute_setpoints(
        environment.controller_observation,
        constraints=environment.constraints,
        config=environment.config,
        commitment_profile=environment.episode_config.electrolyser_commitment,
    )

    _, reward, terminated, truncated, info = environment.step_setpoints(
        decision.setpoints
    )

    execution = environment.last_setpoint_execution
    assert execution is not None
    controller.observe_transition(
        environment.controller_observation,
        execution=execution,
    )
    assert controller.has_pending_transition is False
    assert terminated is True
    assert truncated is False
    assert reward == info["reward"]["total_eur"]
    assert info["legacy_reward"]["diagnostic_only"] is True


def test_legacy_share_door_cannot_bypass_the_economic_objective() -> None:
    environment = PlantEconomicSetpointEnvV2(episode=_one_interval_episode())
    environment.reset(seed=9)
    before = environment.telemetry

    with pytest.raises(RuntimeError, match="requires normalized"):
        environment.step_action(Action.idle())

    assert environment.telemetry == before
    assert environment.decisions_taken == 0


def test_tariff_changes_reward_and_identity_not_physics() -> None:
    episode = _one_interval_episode()
    low = EconomicAssumptionsV1(
        tariff=DayAheadTariffV1(id="zero", hourly_eur_per_mwh=(0.0,) * 24)
    )
    high = EconomicAssumptionsV1(
        tariff=DayAheadTariffV1(id="high", hourly_eur_per_mwh=(150.0,) * 24)
    )
    low_env = PlantEconomicSetpointEnvV2(episode=episode, economic_assumptions=low)
    high_env = PlantEconomicSetpointEnvV2(episode=episode, economic_assumptions=high)
    low_env.reset(seed=11)
    high_env.reset(seed=11)
    action = np.asarray([-1.0, -1.0, -1.0, 1.0], dtype=np.float32)

    _, low_reward, _, _, low_info = low_env.step(action)
    _, high_reward, _, _, high_info = high_env.step(action)

    assert low_info["economic_interval"]["accepted_solar_kwh"] > 0.0
    assert low_env.telemetry.to_dict() == high_env.telemetry.to_dict()
    assert high_reward < low_reward
    assert low_env.checkpoint_contract_digest != high_env.checkpoint_contract_digest
    low_manifest = low_env.audit_manifest()
    high_manifest = high_env.audit_manifest()
    assert (
        low_manifest["exogenous_episode_sha256"]
        == high_manifest["exogenous_episode_sha256"]
    )
    assert (
        low_manifest["economic_episode_sha256"]
        != high_manifest["economic_episode_sha256"]
    )


def test_environment_is_deterministic_and_passes_gym_contract() -> None:
    episode = _one_interval_episode()
    first = PlantEconomicSetpointEnvV2(episode=episode)
    second = PlantEconomicSetpointEnvV2(episode=episode)
    first_observation, first_info = first.reset(seed=29)
    second_observation, second_info = second.reset(seed=29)
    action = np.zeros(4, dtype=np.float32)

    first_step = first.step(action)
    second_step = second.step(action)

    np.testing.assert_array_equal(first_observation, second_observation)
    assert first_info == second_info
    np.testing.assert_array_equal(first_step[0], second_step[0])
    assert first_step[1:] == second_step[1:]
    check_env(
        PlantEconomicSetpointEnvV2(episode=episode),
        skip_render_check=True,
    )


def test_tariff_cadence_rejects_crossing_decisions() -> None:
    episode = EpisodeConfigV2(
        horizon_h=0.75,
        decision_interval_s=2_700.0,
        start_hour=6.0,
        delivery_period_h=0.75,
        delivery_rate_kg_h=1.0,
    )

    with pytest.raises(ValueError, match="divide one hour"):
        PlantEconomicSetpointEnvV2(episode=episode)

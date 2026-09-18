"""Delivery-only Environment V2 contract, isolation, accounting, and reward tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
import plant_sim
from plant_sim import (
    COMMITMENT_DYNAMICS_VERSION,
    DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE,
    Action,
    PlantConfig,
    PlantState,
    SOLAR_FORCING_VERSION,
)

from plant_backend.controllers import Constraints
from plant_backend.rl import PlantEnv as PlantEnvV1
from plant_backend.rl import spaces as v1_spaces
from plant_backend.rl.v2 import EpisodeConfigV2, PlantEnvV2, RewardWeightsV2
from plant_backend.rl.v2 import spaces as v2_spaces
from plant_backend.rl.v2.accounting import IntervalPowerAccounting
from plant_backend.rl.v2.boundary import CommitmentExposureV2
from plant_backend.rl.v2.config import sample_episode
from plant_backend.rl.v2.contract import (
    CheckpointContractError,
    require_compatible_checkpoint,
)
from plant_backend.rl.v2.metrics import EpisodeMetricsV2
from plant_backend.rl.v2.reward import IntervalOutcome, compute
from plant_backend.rl.v2.weather import WEATHER_GENERATOR_VERSION


SHORT = EpisodeConfigV2(horizon_h=2.0, delivery_period_h=1.0)

# Originally frozen at clean weather-alpha revision e0a0ada before equipment
# commitment dynamics were introduced. Advanced once after the registered
# service-MPC matrix exposed a methanator kW -> kg -> kW one-ULP energy creation.
# The audited old/new traces differ in nine telemetry scalars only, with maximum
# absolute delta 2.842170943040401e-14; the old digest is retained in the journal.
WEATHER_ALPHA_COMPATIBILITY_SHA256 = (
    "be16187347bf80825f191f392da595c5d7f6619ad617ceb830829e423135fcab"
)


def test_v1_and_v2_are_explicit_coexisting_contracts() -> None:
    v1 = PlantEnvV1()
    v2 = PlantEnvV2(episode=SHORT)
    v1_observation, _ = v1.reset(seed=3)
    v2_observation, _ = v2.reset(seed=3)

    assert v1_spaces.ENVIRONMENT_ID == "plant-env-v1"
    assert v2_spaces.ENVIRONMENT_ID == "plant-env-v2-commitment-alpha"
    assert v1_observation.shape == (30,)
    assert v2_observation.shape == (57,)
    assert (
        v1_spaces.ENVIRONMENT_CONTRACT_DIGEST != v2_spaces.ENVIRONMENT_CONTRACT_DIGEST
    )
    assert v1.observation_space.contains(v1_observation)
    assert v2.observation_space.contains(v2_observation)


def test_v2_passes_the_gymnasium_contract_checker() -> None:
    check_env(PlantEnvV2(episode=SHORT), skip_render_check=True)


def test_v2_reset_rejects_boolean_and_negative_seed_aliases() -> None:
    env = PlantEnvV2(episode=SHORT)
    with pytest.raises(ValueError, match="non-negative integer"):
        env.reset(seed=True)
    with pytest.raises(ValueError, match="non-negative integer"):
        env.reset(seed=-1)


def test_v2_controller_door_retains_controller_side_invalidity() -> None:
    env = PlantEnvV2(episode=SHORT)
    env.reset(seed=4)
    initial_soc = env.telemetry.battery_soc

    assert env.metrics.min_battery_soc == pytest.approx(initial_soc)
    _, _, _, _, info = env.step_action(
        Action.idle(),
        controller_invalid=True,
    )
    assert info["metrics"]["out_of_range_decisions"] == 1

    with pytest.raises(TypeError, match="controller_invalid"):
        env.step_action(Action.idle(), controller_invalid=1)  # type: ignore[arg-type]


def test_v2_descriptor_digest_is_complete_and_deterministic() -> None:
    described = v2_spaces.describe()
    assert described["environment_contract_digest"] == v2_spaces.contract_digest()
    assert len(v2_spaces.ENVIRONMENT_CONTRACT_DIGEST) == 64
    int(v2_spaces.ENVIRONMENT_CONTRACT_DIGEST, 16)
    observation_fields = described["observation_fields"]
    assert isinstance(observation_fields, list)
    assert len(observation_fields) == v2_spaces.OBSERVATION_SIZE
    assert len(set(v2_spaces.OBSERVATION_NAMES)) == v2_spaces.OBSERVATION_SIZE
    assert described["digest_scope"] == "observation/action interface only"


def test_full_checkpoint_contract_changes_with_timing_and_objective() -> None:
    base = PlantEnvV2(episode=SHORT)
    slower = PlantEnvV2(
        episode=EpisodeConfigV2(
            horizon_h=2.0,
            decision_interval_s=1_800.0,
            delivery_period_h=1.0,
        )
    )
    reweighted = PlantEnvV2(
        episode=SHORT,
        weights=RewardWeightsV2(shortfall=5.0),
    )
    changed_weather = PlantEnvV2(
        episode=replace(
            SHORT,
            weather=replace(
                SHORT.weather,
                autocorrelation_half_life_h=2.0,
            ),
        )
    )

    assert base.checkpoint_contract_digest != slower.checkpoint_contract_digest
    assert base.checkpoint_contract_digest != reweighted.checkpoint_contract_digest
    assert base.checkpoint_contract_digest != changed_weather.checkpoint_contract_digest
    assert (
        base.checkpoint_contract["interface_contract_digest"]
        == slower.checkpoint_contract["interface_contract_digest"]
    )
    assert base.checkpoint_contract["dynamics_versions"] == {
        "plant_sim": plant_sim.__version__,
        "solar_forcing": SOLAR_FORCING_VERSION,
        "clear_sky_envelope": "plant-sim-half-sine-v1",
        "weather_generator": WEATHER_GENERATOR_VERSION,
        "electrolyser_commitment": COMMITMENT_DYNAMICS_VERSION,
    }
    assert (
        "simulator-owned solar bus"
        in base.checkpoint_contract["timing"]["guard_semantics"]
    )
    assert (
        "terminated=true" in base.checkpoint_contract["timing"]["episode_end_semantics"]
    )


def test_commitment_profile_changes_full_identity_but_not_sampled_conditions() -> None:
    committed = PlantEnvV2(episode=SHORT)
    compatibility_off = PlantEnvV2(episode=replace(SHORT, electrolyser_commitment=None))
    slower_start = PlantEnvV2(
        episode=replace(
            SHORT,
            electrolyser_commitment=replace(
                DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE,
                cold_start_s=7_200.0,
            ),
        )
    )

    assert (
        committed.checkpoint_contract_digest
        != compatibility_off.checkpoint_contract_digest
    )
    assert (
        committed.checkpoint_contract_digest != slower_start.checkpoint_contract_digest
    )
    assert (
        committed.episode_config.to_dict()["electrolyser_commitment"]["profile_id"]
        == DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE.profile_id
    )
    assert compatibility_off.episode_config.to_dict()["electrolyser_commitment"] is None

    committed.reset(seed=41)
    compatibility_off.reset(seed=41)
    left = committed.audit_manifest()
    right = compatibility_off.audit_manifest()
    assert left["manifest_sha256"] != right["manifest_sha256"]
    assert left["exogenous_episode_sha256"] == right["exogenous_episode_sha256"]
    assert (
        left["episode"]["exogenous_episode_sha256"] == left["exogenous_episode_sha256"]
    )


def test_v2_checkpoint_compatibility_requires_the_full_exact_digest() -> None:
    env = PlantEnvV2(episode=SHORT)
    expected = env.checkpoint_contract
    require_compatible_checkpoint(
        {
            "environment_id": v2_spaces.ENVIRONMENT_ID,
            "checkpoint_contract_digest": env.checkpoint_contract_digest,
        },
        expected=expected,
    )

    with pytest.raises(CheckpointContractError, match="missing"):
        require_compatible_checkpoint(
            {"environment_id": v2_spaces.ENVIRONMENT_ID},
            expected=expected,
        )
    with pytest.raises(CheckpointContractError, match="mismatch"):
        require_compatible_checkpoint(
            {
                "environment_id": v2_spaces.ENVIRONMENT_ID,
                "checkpoint_contract_digest": "0" * 64,
            },
            expected=expected,
        )
    with pytest.raises(CheckpointContractError, match="incompatible"):
        require_compatible_checkpoint(
            {
                "environment_id": "plant-env-v2-weather-alpha",
                "checkpoint_contract_digest": env.checkpoint_contract_digest,
            },
            expected=expected,
        )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (EpisodeConfigV2(decision_interval_s=100.0), "physics timestep"),
        (EpisodeConfigV2(horizon_h=1.1), "controller decisions"),
        (
            EpisodeConfigV2(horizon_h=2.5, delivery_period_h=1.0),
            "whole number of delivery periods",
        ),
        (EpisodeConfigV2(delivery_period_h=1.1), "controller decisions"),
        (EpisodeConfigV2(horizon_h=1.0, delivery_period_h=2.0), "at least one"),
        (EpisodeConfigV2(delivery_rate_kg_h=0.0), "delivery_rate"),
        (EpisodeConfigV2(battery_soc=(0.2, 1.2)), "battery_soc"),
        (
            EpisodeConfigV2(
                horizon_h=1.0,
                delivery_period_h=1.0,
                delivery_rate_kg_h=4_000.0,
                ch4_coverage_h=(0.0, 0.0),
            ),
            "commitment exceeds",
        ),
        (
            EpisodeConfigV2(electrolyser_commitment=True),  # type: ignore[arg-type]
            "electrolyser_commitment",
        ),
    ],
)
def test_v2_episode_configuration_rejects_misaligned_or_invalid_values(
    config: EpisodeConfigV2,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        config.validate(PlantConfig())


def test_commitment_alpha_freezes_one_minute_physics_but_off_ablation_does_not() -> (
    None
):
    five_minute_plant = PlantConfig(dt_s=300.0)
    with pytest.raises(ValueError, match="requires a 60 s physics timestep"):
        SHORT.validate(five_minute_plant)

    replace(SHORT, electrolyser_commitment=None).validate(five_minute_plant)


def test_named_random_streams_keep_inventory_draws_stable() -> None:
    plant = PlantConfig()
    first = sample_episode(SHORT, plant, seed=42)
    changed_contract = sample_episode(
        EpisodeConfigV2(horizon_h=2.0, delivery_period_h=1.0, delivery_rate_kg_h=50.0),
        plant,
        seed=42,
    )
    assert tuple(first.stream_spawn_keys) == ("inventories", "weather", "contracts")
    assert (
        first.scenario.initial.battery_soc
        == changed_contract.scenario.initial.battery_soc
    )
    assert first.scenario.initial.h2_kg == changed_contract.scenario.initial.h2_kg
    assert first.scenario.initial.co2_kg == changed_contract.scenario.initial.co2_kg
    assert (
        first.commitments[0].requested_kg
        != changed_contract.commitments[0].requested_kg
    )


@pytest.mark.parametrize("value", [-0.01, float("nan"), float("inf")])
def test_v2_rejects_invalid_reward_weights(value: float) -> None:
    with pytest.raises(ValueError, match="shortfall.*finite and non-negative"):
        PlantEnvV2(episode=SHORT, weights=RewardWeightsV2(shortfall=value))


def test_public_info_contains_current_contract_context_but_no_future_tape() -> None:
    env = PlantEnvV2(episode=SHORT)
    _, info = env.reset(seed=17)
    serialized = json.dumps(info, sort_keys=True).lower()

    assert info["next_delivery"]["steps_until_due"] == 60
    assert "seed" not in serialized
    assert "scenario" not in serialized
    assert "fault" not in serialized
    assert "schedule" not in serialized
    assert "commitments" not in serialized
    assert "boundary_factors" not in serialized
    assert "issues" not in serialized
    assert "stream" not in serialized
    assert "truth_sha256" not in serialized
    assert "forecasts_sha256" not in serialized
    assert "episode_sha256" not in serialized
    assert info["forecast"]["issued_at_physics_boundary_step"] == 0
    assert len(info["forecast"]["windows"]) == 4
    assert not hasattr(env, "episode")
    assert env.current_forecast.to_dict() == info["forecast"]
    assert info["checkpoint_contract_digest"] == env.checkpoint_contract_digest
    assert info["electrolyser_commitment"]["mode"] == "off"
    assert "cold_start_s" not in info["electrolyser_commitment"]


def test_environment_consumes_private_weather_at_the_expected_boundaries() -> None:
    env = PlantEnvV2(episode=SHORT)
    initial_observation, initial_info = env.reset(seed=47)
    assert env._episode is not None  # test-only inspection of private truth
    tape = env._episode.weather.tape

    assert env.simulator.state.solar_output_factor == pytest.approx(tape.at_boundary(0))
    factor_index = v2_spaces.OBSERVATION_NAMES.index("solar_output_factor")
    assert env.telemetry.solar_output_factor == 0.0
    assert initial_observation[factor_index] == 0.0
    assert initial_info["forecast"]["issued_at_physics_boundary_step"] == 0
    assert initial_observation.shape == (57,)

    energy_before = env.simulator.state.energy_produced_kwh
    _, _, _, _, info = env.step_action(Action.idle())

    assert env.simulator.step_index == 15
    assert env.simulator.state.solar_output_factor == pytest.approx(
        tape.at_boundary(15)
    )
    assert info["forecast"]["issued_at_physics_boundary_step"] == 15
    assert env.control_state is not None
    interval = env.control_state.previous_interval
    assert interval.physical_steps == 15
    assert interval.hours == pytest.approx(0.25)
    assert interval.solar_kwh == pytest.approx(
        env.simulator.state.energy_produced_kwh - energy_before
    )


def test_a_decision_holds_one_action_for_fifteen_physics_steps() -> None:
    env = PlantEnvV2(episode=SHORT)
    env.reset(seed=2)
    _, _, terminated, truncated, info = env.step_action(Action.idle())

    assert not terminated and not truncated
    assert env.simulator.step_index == 15
    assert env.physics_steps_taken == 15
    assert env.decisions_taken == 1
    assert info["metrics"]["physical_steps"] == 15
    assert info["metrics"]["decisions"] == 1
    assert info["metrics"]["hours"] == pytest.approx(0.25)


def test_commitment_timers_and_exposures_advance_on_the_inner_physics_grid() -> None:
    env = PlantEnvV2(episode=replace(SHORT, battery_soc=(0.65, 0.65)))
    env.reset(seed=2)
    timer_at = v2_spaces.OBSERVATION_NAMES.index("electrolyser_startup_remaining")
    running_at = v2_spaces.OBSERVATION_NAMES.index("electrolyser_mode_running")
    cold_at = v2_spaces.OBSERVATION_NAMES.index("electrolyser_mode_cold_starting")

    observation, _, _, _, first_info = env.step_action(
        Action(battery_support_kw=3_000.0)
    )
    assert observation[cold_at] == 1.0
    assert observation[timer_at] == pytest.approx(0.75)
    assert first_info["electrolyser_commitment"]["startup_remaining_s"] == 2_700.0
    assert first_info["electrolyser_commitment_interval"]["cold_starts"] == 1
    assert first_info["electrolyser_commitment_interval"][
        "startup_energy_kwh"
    ] == pytest.approx(150.0)
    assert first_info["reward"]["components"]["nonproductive_energy"] < 0.0

    for _ in range(3):
        observation, _, _, _, info = env.step_action(Action(battery_support_kw=3_000.0))
    assert observation[running_at] == 1.0
    assert observation[timer_at] == 0.0
    assert info["electrolyser_commitment"]["mode"] == "running"
    assert info["electrolyser_commitment"]["min_up_remaining_s"] == 3_600.0
    assert info["metrics"]["electrolyser_cold_starts"] == 1
    assert info["metrics"]["electrolyser_startup_energy_kwh"] == pytest.approx(600.0)
    assert info["metrics"]["electrolyser_mode_time_s"][
        "cold_starting"
    ] == pytest.approx(3_600.0)


def test_near_endpoint_policy_vector_can_request_an_orderly_stop() -> None:
    profile = replace(
        DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE,
        cold_start_s=60.0,
        min_up_s=60.0,
    )
    env = PlantEnvV2(
        episode=EpisodeConfigV2(
            horizon_h=0.5,
            delivery_period_h=0.5,
            battery_soc=(0.65, 0.65),
            electrolyser_commitment=profile,
        )
    )
    env.reset(seed=2)
    start = np.full(v2_spaces.ACTION_SIZE, -1.0, dtype=np.float32)
    start[-1] = 1.0
    env.step(start)
    assert env.simulator.electrolyser_commitment is not None
    assert env.simulator.electrolyser_commitment.mode.value == "running"

    near_zero = np.full(v2_spaces.ACTION_SIZE, -1.0, dtype=np.float32)
    near_zero[1] = -0.999999
    near_zero[4] = -0.999999
    _, _, terminated, _, info = env.step(near_zero)

    assert terminated is True
    assert info["electrolyser_commitment_interval"]["controlled_stops"] == 1
    assert info["electrolyser_commitment_interval"]["forced_stops"] == 0


def test_vector_and_controller_doors_share_the_same_transition_path() -> None:
    vector_env = PlantEnvV2(episode=SHORT)
    controller_env = PlantEnvV2(episode=SHORT)
    vector_env.reset(seed=9)
    controller_env.reset(seed=9)
    action = Action(
        battery_share=0.25,
        electrolyser_share=0.5,
        dac_share=0.125,
        methanator_share=0.125,
        battery_support_kw=625.0,
    )
    vector = vector_env.adapter.encode(action, constraints=vector_env.constraints)

    vector_result = vector_env.step(vector)
    controller_result = controller_env.step_action(action)
    assert vector_result[0] == pytest.approx(controller_result[0], abs=1e-7)
    assert vector_result[1] == pytest.approx(controller_result[1], abs=1e-12)
    assert vector_result[2:4] == controller_result[2:4]
    assert vector_result[4]["reward"] == controller_result[4]["reward"]
    assert vector_result[4]["metrics"] == controller_result[4]["metrics"]


def test_raw_production_before_a_deadline_has_no_positive_reward_component() -> None:
    env = PlantEnvV2(episode=SHORT)
    env.reset(seed=5)
    vector = np.asarray([-1.0, -1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    _, reward, _, truncated, info = env.step(vector)

    assert not truncated
    assert info["reward"]["signals"]["raw_production_kg"] > 0.0
    assert info["reward"]["components"]["delivery"] == 0.0
    assert info["reward"]["components"]["terminal_inventory"] == 0.0
    assert reward <= 0.0
    assert sum(info["reward"]["components"].values()) == pytest.approx(reward)


def _interval_outcome(**changes: Any) -> IntervalOutcome:
    base = IntervalOutcome(
        hours=1.0,
        physical_steps=60,
        requested_kg=0.0,
        delivered_kg=0.0,
        unmet_kg=0.0,
        produced_kg=0.0,
        solar_kwh=2_000.0,
        curtailed_kwh=0.0,
        battery_discharged_kwh=0.0,
        min_battery_soc=0.50,
        shutdown_steps=0,
        corrections=0,
        guard_intervention_kwh=0.0,
        dispatch_gap_kwh=0.0,
        request_application_gap_kwh=0.0,
        out_of_range=False,
        terminal_ch4_kg=0.0,
        reference_delivery_kg=330.0,
        reference_period_h=6.0,
    )
    return replace(base, **changes)


def test_v2_energy_penalty_is_absolute_and_partition_invariant() -> None:
    config = PlantConfig()
    constraints = Constraints()
    whole = compute(
        _interval_outcome(
            hours=1.0,
            physical_steps=60,
            curtailed_kwh=600.0,
            battery_discharged_kwh=120.0,
            shutdown_steps=12,
            corrections=18,
            commitment=CommitmentExposureV2(
                enabled=True,
                startup_energy_kwh=300.0,
                forced_stops=2,
            ),
        ),
        config=config,
        constraints=constraints,
        terminal=False,
    )
    half = compute(
        _interval_outcome(
            hours=0.5,
            physical_steps=30,
            curtailed_kwh=300.0,
            battery_discharged_kwh=60.0,
            shutdown_steps=6,
            corrections=9,
            commitment=CommitmentExposureV2(
                enabled=True,
                startup_energy_kwh=150.0,
                forced_stops=1,
            ),
        ),
        config=config,
        constraints=constraints,
        terminal=False,
    )

    for component in (
        "curtailment",
        "battery_drain",
        "shutdown",
        "correction",
        "nonproductive_energy",
        "forced_stop",
    ):
        assert whole.components[component] == pytest.approx(
            2.0 * half.components[component]
        )

    small = compute(
        _interval_outcome(curtailed_kwh=1.0),
        config=config,
        constraints=constraints,
        terminal=False,
    )
    large = compute(
        _interval_outcome(curtailed_kwh=1_000.0),
        config=config,
        constraints=constraints,
        terminal=False,
    )
    assert large.components["curtailment"] == pytest.approx(
        1_000.0 * small.components["curtailment"]
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"curtailed_kwh": float("nan")},
        {"dispatch_gap_kwh": float("inf")},
        {"physical_steps": -1},
        {"min_battery_soc": 1.01},
    ),
)
def test_interval_outcome_rejects_corrupt_accounting(
    changes: dict[str, object]
) -> None:
    with pytest.raises((TypeError, ValueError), match="interval outcome"):
        _interval_outcome(**changes)


def test_v2_reports_requested_and_applied_energy_over_the_whole_decision() -> None:
    env = PlantEnvV2(episode=SHORT)
    env.reset(seed=4)
    _, _, _, _, info = env.step_action(Action(battery_support_kw=10_000.0))

    power = info["interval_power"]
    assert power["hours"] == pytest.approx(0.25)
    assert power["requested_kwh"]["battery_support"] == pytest.approx(2_500.0)
    assert power["applied_kwh"]["battery_support"] < 625.0
    assert power["guard_intervention_l1_kwh"] > 0.0
    assert power["request_application_l1_kwh"] >= power["guard_intervention_l1_kwh"]
    assert info["guard"]["intervention_l1_kwh"] == pytest.approx(
        power["guard_intervention_l1_kwh"]
    )
    assert info["metrics"]["guard_intervention_kwh"] == pytest.approx(
        power["guard_intervention_l1_kwh"]
    )
    assert "applied_action" not in info
    assert info["guarded_action"]["battery_support_kw"] < 10_000.0
    authority = info["control_authority"]
    assert authority["dispatch_gap_integral_l1_kwh"] > 0.0
    assert info["metrics"]["dispatch_gap_kwh"] == pytest.approx(
        authority["dispatch_gap_integral_l1_kwh"]
    )
    assert info["metrics"]["request_application_gap_kwh"] == pytest.approx(
        authority["request_application_integral_l1_kwh"]
    )
    accepted = sum(
        power["applied_kwh"][name] for name in ("electrolyser", "dac", "methanator")
    )
    assert info["metrics"]["accepted_process_energy_kwh"] == pytest.approx(accepted)
    assert info["metrics"]["electrolyser_applied_energy_kwh"] == pytest.approx(
        power["applied_kwh"]["electrolyser"]
    )
    assert info["metrics"]["battery_throughput_kwh"] == pytest.approx(
        power["applied_kwh"]["battery"] + power["applied_kwh"]["battery_support"]
    )


def test_process_energy_intensity_is_undefined_without_delivered_product() -> None:
    env = PlantEnvV2(episode=SHORT)
    env.reset(seed=4)
    _, _, _, _, info = env.step_action(Action.idle())

    assert info["metrics"]["delivered_kg"] == 0.0
    assert info["metrics"]["accepted_process_energy_per_delivered_kg"] is None


def test_process_energy_intensity_counts_each_accepted_conversion_channel_once() -> (
    None
):
    metrics = EpisodeMetricsV2(
        delivered_kg=10.0,
        electrolyser_applied_energy_kwh=520.0,
        dac_applied_energy_kwh=20.0,
        methanator_applied_energy_kwh=80.0,
    )

    assert metrics.accepted_process_energy_kwh == pytest.approx(620.0)
    assert metrics.accepted_process_energy_per_delivered_kg == pytest.approx(62.0)
    assert metrics.to_dict()[
        "accepted_process_energy_per_delivered_kg"
    ] == pytest.approx(62.0)


def test_episode_metrics_track_complete_initial_and_terminal_inventory_vector() -> None:
    env = PlantEnvV2(episode=SHORT)
    env.reset(seed=4)
    initial = env.telemetry
    initial_metrics = env.metrics.to_dict()

    assert initial_metrics["initial_h2_kg"] == pytest.approx(initial.h2_level_kg)
    assert initial_metrics["initial_co2_kg"] == pytest.approx(initial.co2_level_kg)
    assert initial_metrics["initial_ch4_kg"] == pytest.approx(initial.ch4_level_kg)
    assert initial_metrics["initial_battery_soc"] == pytest.approx(initial.battery_soc)
    assert initial_metrics["initial_battery_energy_kwh"] == pytest.approx(
        initial.battery_energy_kwh
    )

    _, _, _, _, info = env.step_action(Action.all_to_battery())
    terminal = env.telemetry
    metrics = info["metrics"]
    assert metrics["terminal_h2_kg"] == pytest.approx(terminal.h2_level_kg)
    assert metrics["terminal_co2_kg"] == pytest.approx(terminal.co2_level_kg)
    assert metrics["terminal_ch4_kg"] == pytest.approx(terminal.ch4_level_kg)
    assert metrics["terminal_battery_soc"] == pytest.approx(terminal.battery_soc)
    assert metrics["terminal_battery_energy_kwh"] == pytest.approx(
        terminal.battery_energy_kwh
    )
    assert metrics["h2_inventory_change_kg"] == pytest.approx(
        terminal.h2_level_kg - initial.h2_level_kg
    )
    assert metrics["co2_inventory_change_kg"] == pytest.approx(
        terminal.co2_level_kg - initial.co2_level_kg
    )
    assert metrics["ch4_inventory_change_kg"] == pytest.approx(
        terminal.ch4_level_kg - initial.ch4_level_kg
    )
    assert metrics["battery_energy_change_kwh"] == pytest.approx(
        terminal.battery_energy_kwh - initial.battery_energy_kwh
    )
    assert metrics["battery_throughput_kwh"] == pytest.approx(
        info["interval_power"]["applied_kwh"]["battery"]
        + info["interval_power"]["applied_kwh"]["battery_support"]
    )


def test_episode_metrics_reject_corrupt_terminal_inventory() -> None:
    metrics = EpisodeMetricsV2()
    outcome = _interval_outcome()
    breakdown = compute(
        outcome,
        config=PlantConfig(),
        constraints=Constraints(),
        terminal=False,
    )

    with pytest.raises(ValueError, match="terminal hydrogen inventory"):
        metrics.record(
            outcome,
            breakdown,
            (),
            corrected_physics_steps=0,
            battery_charge_applied_kwh=0.0,
            battery_support_applied_kwh=0.0,
            electrolyser_applied_energy_kwh=0.0,
            dac_applied_energy_kwh=0.0,
            methanator_applied_energy_kwh=0.0,
            terminal_state=PlantState(h2_level_kg=float("nan")),
        )


def test_share_accounting_uses_the_realized_interval_bus_not_prior_solar() -> None:
    accounting = IntervalPowerAccounting()
    action = Action.all_to_battery()
    accounting.record(
        action,
        action,
        after=PlantState(
            solar_available_kw=800.0,
            solar_to_battery_kw=800.0,
        ),
        dt_h=0.25,
    )

    result = accounting.to_dict()
    assert result["requested_kwh"]["battery"] == pytest.approx(200.0)
    assert result["applied_kwh"]["battery"] == pytest.approx(200.0)
    assert result["request_application_l1_kwh"] == 0.0
    assert result["share_command_basis"] == (
        "realized solar bus in each physical interval"
    )


def test_dispatch_gap_integrates_absolute_step_gaps_without_cancellation() -> None:
    accounting = IntervalPowerAccounting()
    guarded = Action.all_to_battery()
    accounting.record(
        guarded,
        guarded,
        after=PlantState(solar_available_kw=100.0, solar_to_battery_kw=0.0),
        dt_h=1.0,
    )
    accounting.record(
        Action.idle(),
        Action.idle(),
        after=PlantState(solar_available_kw=100.0, solar_to_battery_kw=100.0),
        dt_h=1.0,
    )

    assert accounting.guarded_kwh["battery"] == accounting.applied_kwh["battery"]
    assert accounting.dispatch_gap_l1_kwh == 0.0
    assert accounting.request_application_l1_kwh == 0.0
    assert accounting.dispatch_gap_integral_l1_kwh == pytest.approx(200.0)
    assert accounting.request_application_integral_l1_kwh == pytest.approx(200.0)


@pytest.mark.parametrize("dt_h", (0.0, -1.0, float("nan"), float("inf")))
def test_power_accounting_rejects_invalid_interval_duration(dt_h: float) -> None:
    accounting = IntervalPowerAccounting()
    with pytest.raises(ValueError, match="positive and finite"):
        accounting.record(Action.idle(), Action.idle(), after=PlantState(), dt_h=dt_h)
    assert accounting.hours == 0.0


def test_power_accounting_rejects_nonfinite_simulator_flow_without_mutation() -> None:
    accounting = IntervalPowerAccounting()
    with pytest.raises(ValueError, match="applied power channel electrolyser"):
        accounting.record(
            Action.idle(),
            Action.idle(),
            after=PlantState(electrolyser_power_kw=float("nan")),
            dt_h=0.25,
        )
    assert accounting.hours == 0.0
    assert all(value == 0.0 for value in accounting.applied_kwh.values())


def test_v2_enforces_absolute_power_caps_against_each_realized_bus() -> None:
    env = PlantEnvV2(
        episode=SHORT,
        constraints=Constraints(dac_max_kw=1.0),
    )
    env.reset(seed=4)
    _, _, _, _, info = env.step_action(Action(dac_share=1.0))

    interval = info["interval_power"]
    assert interval["applied_kwh"]["dac"] <= interval["hours"] + 1e-9
    assert env.simulator.state.dac_power_kw <= 1.0 + 1e-9
    assert info["guard"]["interventions"] > 0


def test_v2_retains_the_raw_policy_vector_for_aliasing_audits() -> None:
    env = PlantEnvV2(episode=SHORT)
    env.reset(seed=2)
    raw = np.asarray([2.0, -2.0, 0.25, float("nan"), float("inf")])
    _, _, _, _, info = env.step(raw)

    assert info["requested"]["raw_policy_vector"] == [
        2.0,
        -2.0,
        0.25,
        "nan",
        "+inf",
    ]
    assert info["guard"]["out_of_range"] is True
    json.dumps(info, allow_nan=False)


def test_delivery_shortfall_drives_service_metrics_and_terminal_ordering() -> None:
    episode = EpisodeConfigV2(
        horizon_h=1.0,
        delivery_period_h=1.0,
        ch4_coverage_h=(0.0, 0.0),
    )
    env = PlantEnvV2(episode=episode)
    env.reset(seed=1)
    while True:
        _, reward, terminated, truncated, info = env.step_action(Action.idle())
        assert not truncated
        if terminated:
            break

    metrics = info["metrics"]
    assert info["delivery"]["commitments_settled"] == 1
    assert metrics["commitments_settled"] == 1
    assert metrics["commitments_met"] == 0
    assert metrics["requested_kg"] == pytest.approx(40.0)
    assert metrics["delivered_kg"] == 0.0
    assert metrics["unmet_kg"] == pytest.approx(40.0)
    assert metrics["delivery_fill_rate"] == 0.0
    assert metrics["commitment_service_level"] == 0.0
    assert info["reward"]["components"]["shortfall"] == pytest.approx(-4.0)
    # Deadline settlement happens before terminal inventory valuation.
    assert info["reward"]["components"]["terminal_inventory"] == 0.0
    assert reward < 0.0


def test_contract_horizon_terminates_without_bootstrap_after_salvage_value() -> None:
    episode = EpisodeConfigV2(
        horizon_h=1.0,
        delivery_period_h=1.0,
        ch4_coverage_h=(2.0, 2.0),
    )
    env = PlantEnvV2(episode=episode)
    env.reset(seed=7)
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step_action(Action.idle())

    assert terminated is True
    assert truncated is False
    assert info["reward"]["components"]["terminal_inventory"] == pytest.approx(
        env.weights.terminal_inventory
    )
    with pytest.raises(RuntimeError, match="already terminated"):
        env.step_action(Action.idle())


def test_observations_remain_bounded_through_delivery_boundaries() -> None:
    env = PlantEnvV2(episode=SHORT)
    observation, _ = env.reset(seed=8)
    assert env.observation_space.contains(observation)
    finished = False
    while not finished:
        observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
        finished = terminated or truncated
        assert env.observation_space.contains(observation)
        assert np.all(np.isfinite(observation))


def test_same_seed_and_actions_replay_exactly() -> None:
    actions = [
        np.linspace(-1.0, 1.0, v2_spaces.ACTION_SIZE, dtype=np.float32),
        np.zeros(v2_spaces.ACTION_SIZE, dtype=np.float32),
    ] * 4

    def rollout() -> tuple[list[list[float]], list[float], dict[str, object], str]:
        env = PlantEnvV2(episode=SHORT)
        observation, _ = env.reset(seed=101)
        observations = [observation.tolist()]
        rewards: list[float] = []
        for action in actions:
            observation, reward, _, _, _ = env.step(action)
            observations.append(observation.tolist())
            rewards.append(reward)
        manifest = env.audit_manifest()
        return observations, rewards, env.metrics.to_dict(), manifest["manifest_sha256"]

    assert rollout() == rollout()


def test_weather_alpha_has_a_frozen_compatibility_trace() -> None:
    """Protect the last pre-commitment task behind an exact replay gate."""
    episode = EpisodeConfigV2(
        horizon_h=24.0,
        delivery_period_h=6.0,
        electrolyser_commitment=None,
    )
    env = PlantEnvV2(episode=episode)
    observation, _ = env.reset(seed=101)
    actions = (
        Action(
            battery_share=0.25,
            electrolyser_share=0.50,
            dac_share=0.15,
            methanator_share=0.10,
            battery_support_kw=400.0,
        ),
        Action.idle(),
        Action(
            battery_share=0.10,
            electrolyser_share=0.65,
            dac_share=0.10,
            methanator_share=0.15,
            battery_support_kw=1_200.0,
        ),
        Action(
            battery_share=0.70,
            electrolyser_share=0.15,
            dac_share=0.10,
            methanator_share=0.05,
        ),
    )
    trace: list[dict[str, object]] = [
        {
            "observation": observation[:42].tolist(),
            "state": env.simulator.state.to_dict(),
        }
    ]
    index = 0
    while True:
        observation, reward, terminated, truncated, info = env.step_action(
            actions[index % len(actions)]
        )
        trace.append(
            {
                "observation": observation[:42].tolist(),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "guard": info["guard"],
                "interval_power": info["interval_power"],
                "delivery": info["delivery"],
                "state": env.simulator.state.to_dict(),
            }
        )
        index += 1
        if terminated or truncated:
            break

    assert index == 96
    canonical = json.dumps(
        trace,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == WEATHER_ALPHA_COMPATIBILITY_SHA256


def test_private_audit_manifest_contains_replay_data_and_detached_accounting() -> None:
    env = PlantEnvV2(episode=SHORT)
    env.reset(seed=23)
    before = env.audit_manifest()
    assert before["manifest_schema_version"] == 3
    assert len(before["exogenous_episode_sha256"]) == 64
    assert (
        before["episode"]["exogenous_episode_sha256"]
        == before["exogenous_episode_sha256"]
    )
    assert before["episode"]["seed"] == 23
    assert len(before["episode"]["commitments"]) == 2
    assert before["delivery_accounting"]["pending_count"] == 2
    assert before["delivery_accounting"]["frozen"] is True
    digest = before["manifest_sha256"]
    assert (
        before["checkpoint_contract"]["checkpoint_contract_digest"]
        == env.checkpoint_contract_digest
    )
    assert before["episode"]["weather"]["replay"]["truth_origin_kind"] == (
        "synthetic_generator"
    )
    assert len(before["episode"]["weather"]["private_hashes"]["truth_sha256"]) == 64

    for _ in range(4):
        env.step_action(Action.idle())
    after = env.audit_manifest()
    assert after["manifest_sha256"] == digest
    assert after["delivery_accounting"]["pending_count"] < 2


def test_v2_delivery_schedule_cannot_be_replaced_mid_episode() -> None:
    env = PlantEnvV2(episode=SHORT)
    env.reset(seed=31)
    env.step_action(Action.idle())
    before = env.simulator.delivery_audit()

    with pytest.raises(RuntimeError, match="frozen"):
        env.simulator.set_delivery_schedule([])

    assert env.simulator.delivery_audit() == before

"""Fast integration checks for the cyclic stochastic-market beta task."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest

from plant_backend.rl import checkpoints
from plant_backend.rl.policies import load_policy
from plant_backend.rl.v2 import beta_env as beta_env_module
from plant_backend.rl.v2 import beta_spaces, economic_spaces, setpoint_spaces
from plant_backend.rl.v2.beta_config import CyclicMarketTaskConfigV1
from plant_backend.rl.v2.beta_disturbances import DisturbanceConfigV1
from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.economic_env import PlantEconomicSetpointEnvV2
from plant_backend.rl.v2.market import MarketConfigV1, generate_market_episode
from plant_backend.rl.v2.mpc_schema import domain_sha256
from plant_backend.rl.v2.policy_setpoint_controller import (
    NormalizedSetpointPolicyControllerV1,
    policy_contract_for_environment,
)
from plant_backend.rl.v2.policy_evaluation import (
    compare_policy_to_heuristic,
    execute_policy_episode,
)
from plant_backend.rl.v2.setpoint_env import PlantSetpointEnvV2
from plant_backend.rl.v2.training.config import ControllerNeutralTrainingConfigV1
from plant_backend.rl.v2.training.ppo import make_env, plan
from plant_backend.rl.v2.training.train import build_parser, resolve


FIXED_ACTION = np.asarray((-0.75, 0.25, 0.5, -0.5), dtype=np.float32)
IDLE_ACTION = np.full(4, -1.0, dtype=np.float32)


def _episode(*, horizon_h: float = 24.0) -> EpisodeConfigV2:
    """Smallest valid beta geometry, with only 24 controller calls per day."""

    return EpisodeConfigV2(
        horizon_h=horizon_h,
        decision_interval_s=3_600.0,
        start_hour=6.0,
        delivery_period_h=24.0,
        delivery_rate_kg_h=1.0,
        battery_soc=(0.5, 0.5),
        h2_fill=(0.1, 0.1),
        co2_fill=(0.1, 0.1),
        ch4_coverage_h=(1.0, 1.0),
    )


def _beta_task(*, disturbances: bool = False) -> CyclicMarketTaskConfigV1:
    return CyclicMarketTaskConfigV1(
        market=MarketConfigV1(forecast_ensemble_members=10),
        disturbances=DisturbanceConfigV1(
            enabled=disturbances,
            per_cycle_probability=1.0 if disturbances else 0.0,
            duration_h=(1.0, 1.0),
        ),
    )


def _beta_training_config() -> ControllerNeutralTrainingConfigV1:
    return ControllerNeutralTrainingConfigV1(
        environment_id=beta_spaces.ENVIRONMENT_ID,
        episode=_episode(),
        beta_task=_beta_task(),
        total_timesteps=4,
        n_envs=1,
        checkpoint_every=0,
        notes="beta integration fixture; no training",
    ).validate()


def _fixed_checkpoint(
    tmp_path: Path,
    environment: PlantCyclicMarketEnvV2Beta | PlantEconomicSetpointEnvV2,
    *,
    name: str,
) -> Path:
    return checkpoints.write_fixed_checkpoint(
        tmp_path / name,
        FIXED_ACTION,
        contract=policy_contract_for_environment(environment),
    )


def _mapping_keys(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _mapping_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _mapping_keys(item)


def _deterministic_step_info(info: dict[str, Any]) -> dict[str, Any]:
    """Discard only command provenance and non-deterministic wall-clock timing."""

    excluded = {
        "requested",
        "process_setpoint_execution",
        "source_decision",
        "dispatcher_elapsed_ns",
        "dispatcher_call_count",
    }
    return {key: value for key, value in info.items() if key not in excluded}


def test_beta_contract_is_exact_and_preserves_the_alpha_physical_draw() -> None:
    episode = _episode()
    beta = PlantCyclicMarketEnvV2Beta(
        episode=episode,
        beta_task=_beta_task(),
    )
    alpha = PlantSetpointEnvV2(episode=episode)
    try:
        beta_observation, beta_info = beta.reset(seed=907)
        alpha_observation, alpha_info = alpha.reset(seed=907)

        assert beta_spaces.OBSERVATION_SIZE == 135
        assert beta_spaces.ACTION_SIZE == 4
        assert beta_observation.shape == beta.observation_space.shape == (135,)
        assert beta.action_space.shape == (4,)
        assert beta.observation_space.contains(beta_observation)
        assert beta.action_space.contains(np.zeros(4, dtype=np.float32))
        assert beta_info["environment_id"] == beta_spaces.ENVIRONMENT_ID
        assert beta_info["environment_contract_digest"] == (
            beta_spaces.ENVIRONMENT_CONTRACT_DIGEST
        )
        assert beta_info["environment_id"] != alpha_info["environment_id"]
        assert beta.checkpoint_contract_digest != alpha.checkpoint_contract_digest
        assert beta_spaces.OBSERVATION_NAMES[: setpoint_spaces.OBSERVATION_SIZE] == (
            setpoint_spaces.OBSERVATION_NAMES
        )
        assert beta_spaces.ACTION_NAMES == setpoint_spaces.ACTION_NAMES

        np.testing.assert_array_equal(
            beta_observation[: setpoint_spaces.OBSERVATION_SIZE],
            alpha_observation,
        )
        assert beta.cyclic_market_controller_observation.plant.to_dict() == (
            alpha.controller_observation.to_dict()
        )
        beta_manifest = beta.audit_manifest()
        alpha_manifest = alpha.audit_manifest()
        assert beta_manifest["alpha_physical_exogenous_episode_sha256"] == (
            alpha_manifest["exogenous_episode_sha256"]
        )
        for key in (
            "seed",
            "random_streams",
            "start_hour",
            "initial",
            "commitments",
            "physics_steps",
            "decision_stride",
            "delivery_stride",
            "weather_episode_sha256",
        ):
            assert beta_manifest["episode"][key] == alpha_manifest["episode"][key]
    finally:
        beta.close()
        alpha.close()


def test_realized_market_price_drives_reward_and_issue_vintage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opening public price, not the baseline tariff or a later issue, is billed."""

    forced_price = [25.0]

    def forced_market(
        config: MarketConfigV1,
        *,
        seed: int | np.random.SeedSequence,
        decision_count: int,
        decision_interval_s: float,
        start_hour: float,
        baseline: Any,
    ) -> Any:
        return generate_market_episode(
            config,
            seed=seed,
            decision_count=decision_count,
            decision_interval_s=decision_interval_s,
            start_hour=start_hour,
            baseline=baseline,
            realized_prices_eur_per_mwh=[forced_price[0]] * decision_count,
        )

    monkeypatch.setattr(beta_env_module, "generate_market_episode", forced_market)
    low = PlantCyclicMarketEnvV2Beta(episode=_episode(), beta_task=_beta_task())
    high = PlantCyclicMarketEnvV2Beta(episode=_episode(), beta_task=_beta_task())
    try:
        low_observation, low_reset_info = low.reset(seed=908)
        low_issue = low.current_market_issue
        forced_price[0] = 225.0
        high_observation, high_reset_info = high.reset(seed=908)
        high_issue = high.current_market_issue

        assert low_issue.issued_at_decision == high_issue.issued_at_decision == 0
        assert low_issue.issued_at_offset_s == high_issue.issued_at_offset_s == 0.0
        assert low_reset_info["market"] == low_issue.to_dict()
        assert high_reset_info["market"] == high_issue.to_dict()
        assert low_issue.current_eur_per_mwh == 25.0
        assert high_issue.current_eur_per_mwh == 225.0
        np.testing.assert_array_equal(
            low_observation[: setpoint_spaces.OBSERVATION_SIZE],
            high_observation[: setpoint_spaces.OBSERVATION_SIZE],
        )

        low_result = low.step(np.zeros(4, dtype=np.float32))
        high_result = high.step(np.zeros(4, dtype=np.float32))
        low_interval = low_result[4]["economic_interval"]
        high_interval = high_result[4]["economic_interval"]
        accepted_kwh = low_interval["accepted_solar_kwh"]

        assert accepted_kwh > 0.0
        assert high_interval["accepted_solar_kwh"] == pytest.approx(accepted_kwh)
        assert low_interval["opportunity_price_eur_per_mwh"] == 25.0
        assert high_interval["opportunity_price_eur_per_mwh"] == 225.0
        assert low_interval["tariff"] == low_issue.to_dict()
        assert high_interval["tariff"] == high_issue.to_dict()
        assert low_result[4]["market"]["issued_at_decision"] == 1
        assert high_result[4]["market"]["issued_at_decision"] == 1
        assert low.current_market_issue.issued_at_decision == low.decisions_taken == 1
        assert high.current_market_issue.issued_at_decision == high.decisions_taken == 1
        expected_delta = -accepted_kwh * (225.0 - 25.0) / 1_000.0
        assert high_result[1] - low_result[1] == pytest.approx(expected_delta)
        assert high_result[4]["reward"]["components_eur"][
            "solar_opportunity_cost"
        ] - low_result[4]["reward"]["components_eur"][
            "solar_opportunity_cost"
        ] == pytest.approx(
            expected_delta
        )
    finally:
        low.close()
        high.close()


def test_public_info_excludes_private_market_and_disturbance_sources() -> None:
    environment = PlantCyclicMarketEnvV2Beta(
        episode=_episode(),
        beta_task=_beta_task(disturbances=True),
    )
    try:
        _, public_info = environment.reset(seed=909)
        public_keys = set(_mapping_keys(public_info))

        assert {
            "selected_regime",
            "selected_regime_parameters",
            "schedule",
            "random_streams",
            "truth_sha256",
            "prices_eur_per_mwh",
            "seed",
        }.isdisjoint(public_keys)
        assert set(public_info["market"]) == {
            "schema",
            "market_id",
            "issued_at_decision",
            "issued_at_offset_s",
            "issue_time_basis",
            "current_eur_per_mwh",
            "bins",
        }

        private = environment.audit_manifest()
        assert private["episode"]["seed"] == 909
        assert private["market"]["selected_regime"]
        assert private["market"]["random_streams"]
        assert private["market"]["random_streams"]["market.root"]["spawn_key"] == [3]
        assert private["market"]["private_hashes"]["truth_sha256"]
        assert len(private["disturbances"]["schedule"]) == 1
        assert private["disturbances"]["random_streams"]
        assert private["disturbances"]["random_streams"]["root"]["spawn_key"] == [4]
        assert public_info["market"] == environment.current_market_issue.to_dict()
    finally:
        environment.close()


def test_cycle_settles_exactly_every_24h_against_one_opening_reference() -> None:
    environment = PlantCyclicMarketEnvV2Beta(
        episode=_episode(horizon_h=48.0),
        beta_task=_beta_task(),
    )
    try:
        environment.reset(seed=910)
        opening_reference = environment.cycle_reference
        expected_targets = opening_reference.targets(environment.beta_task.cycle)
        first_boundary_state = None
        final_result = None

        for decision in range(48):
            final_result = environment.step(IDLE_ACTION.copy())
            cycle_info = final_result[4]["cycle_constraint"]
            if decision < 23 or 24 <= decision < 47:
                assert cycle_info["settled_this_step"] is None
            if decision == 23:
                first_boundary_state = environment.telemetry
                assert cycle_info["settled_this_step"]["boundary_decision"] == 24
                assert environment.cycle_observation.cycle_index == 2
                assert environment.cycle_observation.decision_in_cycle == 0
                assert environment.cycle_observation.progress == 0.0
                assert environment.cycle_reference == opening_reference
                # The due delivery consumes the opening methane. A new cycle does
                # not reset the plant or ratchet its target down to that new level.
                assert first_boundary_state.ch4_level_kg < opening_reference.ch4_kg
                assert environment.cycle_observation.ch4_margin_kg < 0.0

        assert final_result is not None
        assert first_boundary_state is not None
        assert final_result[2] is True
        assert final_result[3] is False
        assert [item.cycle_index for item in environment.cycle_settlements] == [1, 2]
        assert [item.boundary_decision for item in environment.cycle_settlements] == [
            24,
            48,
        ]
        assert environment.cycle_settlements[0].targets == expected_targets
        assert environment.cycle_settlements[1].targets == expected_targets
        assert environment.cycle_reference == opening_reference
        assert environment.physics_steps_taken == 48 * 60
        final_cycle = final_result[4]["cycle_constraint"]
        assert final_cycle["settled_cycle_count"] == 2
        assert final_cycle["settled_this_step"]["boundary_decision"] == 48
        assert final_cycle["scalar_reward_contains_constraint_penalty"] is False
        assert final_cycle["cumulative_constraint_cost"] > 0.0
        assert final_result[1] == pytest.approx(
            sum(final_result[4]["reward"]["components_eur"].values())
        )
    finally:
        environment.close()


def test_beta_fixed_policy_and_gym_action_share_one_physical_door(
    tmp_path: Path,
) -> None:
    config = _beta_training_config()
    direct = make_env(config)
    commanded = make_env(config)
    assert isinstance(direct, PlantCyclicMarketEnvV2Beta)
    assert isinstance(commanded, PlantCyclicMarketEnvV2Beta)
    try:
        direct_observation, direct_info = direct.reset(seed=911)
        commanded_observation, commanded_info = commanded.reset(seed=911)
        checkpoint = _fixed_checkpoint(
            tmp_path,
            commanded,
            name="fixed-beta-controller-neutral",
        )
        contract = policy_contract_for_environment(commanded)
        loaded = load_policy(checkpoint, contract=contract)
        controller = NormalizedSetpointPolicyControllerV1.from_environment(
            loaded,
            commanded,
        )
        controller.reset()
        source = commanded.cyclic_market_controller_observation
        decision = controller.compute_command(
            source,
            constraints=commanded.constraints,
            config=commanded.config,
            commitment_profile=commanded.episode_config.electrolyser_commitment,
        )

        assert loaded.metadata.environment_id == beta_spaces.ENVIRONMENT_ID
        assert loaded.metadata.observation_fields == beta_spaces.OBSERVATION_NAMES
        assert loaded.metadata.action_fields == beta_spaces.ACTION_NAMES
        assert loaded.metadata.checkpoint_contract_digest == (
            commanded.checkpoint_contract_digest
        )
        np.testing.assert_array_equal(direct_observation, commanded_observation)
        assert direct_info == commanded_info
        assert decision.evidence.policy_observation_sha256 == domain_sha256(
            "rl-policy-observation-vector-v1",
            [float(value) for value in commanded_observation],
        )
        assert decision.evidence.economic_observation_sha256 == source.sha256

        direct_result = direct.step(FIXED_ACTION.copy())
        commanded_result = commanded.step_process_setpoints(decision.command)

        np.testing.assert_array_equal(direct_result[0], commanded_result[0])
        assert direct_result[1:4] == commanded_result[1:4]
        assert _deterministic_step_info(direct_result[4]) == (
            _deterministic_step_info(commanded_result[4])
        )
        assert direct.telemetry == commanded.telemetry
        assert direct.control_state == commanded.control_state
        assert direct.metrics.to_dict() == commanded.metrics.to_dict()
        assert direct.economic_metrics.to_dict() == commanded.economic_metrics.to_dict()
        assert direct.cycle_settlements == commanded.cycle_settlements
        assert direct.audit_manifest() == commanded.audit_manifest()
    finally:
        direct.close()
        commanded.close()


def test_alpha_and_beta_fixed_checkpoints_cannot_cross_admission(
    tmp_path: Path,
) -> None:
    episode = _episode()
    beta = PlantCyclicMarketEnvV2Beta(episode=episode, beta_task=_beta_task())
    alpha = PlantEconomicSetpointEnvV2(episode=episode)
    try:
        beta_checkpoint = _fixed_checkpoint(
            tmp_path,
            beta,
            name="fixed-beta-admission",
        )
        alpha_checkpoint = _fixed_checkpoint(
            tmp_path,
            alpha,
            name="fixed-alpha-admission",
        )

        with pytest.raises(
            checkpoints.IncompatibleCheckpointError,
            match="observation fields|environment|task contract digest",
        ):
            load_policy(
                beta_checkpoint,
                contract=policy_contract_for_environment(alpha),
            )
        with pytest.raises(
            checkpoints.IncompatibleCheckpointError,
            match="observation fields|environment|task contract digest",
        ):
            load_policy(
                alpha_checkpoint,
                contract=policy_contract_for_environment(beta),
            )
    finally:
        beta.close()
        alpha.close()


def test_beta_training_config_round_trips_and_constructs_exact_environment(
    tmp_path: Path,
) -> None:
    config = _beta_training_config()
    loaded = ControllerNeutralTrainingConfigV1.from_file(
        config.write(tmp_path / "beta-training.json")
    )

    assert loaded == config
    assert loaded.to_dict() == config.to_dict()
    assert loaded.beta_task is not None
    environment = make_env(loaded)
    try:
        assert isinstance(environment, PlantCyclicMarketEnvV2Beta)
        assert environment.beta_task == loaded.beta_task
        assert environment.observation_space.shape == (beta_spaces.OBSERVATION_SIZE,)
        assert environment.action_space.shape == (beta_spaces.ACTION_SIZE,)
        assert policy_contract_for_environment(environment).environment_id == (
            beta_spaces.ENVIRONMENT_ID
        )
        assert environment.checkpoint_contract["environment_id"] == (
            beta_spaces.ENVIRONMENT_ID
        )
        assert environment.checkpoint_contract["interface_contract_digest"] == (
            beta_spaces.ENVIRONMENT_CONTRACT_DIGEST
        )
        assert beta_spaces.ENVIRONMENT_ID not in {
            economic_spaces.ENVIRONMENT_ID,
            setpoint_spaces.ENVIRONMENT_ID,
        }
        assert beta_spaces.ENVIRONMENT_CONTRACT_DIGEST not in {
            economic_spaces.ENVIRONMENT_CONTRACT_DIGEST,
            setpoint_spaces.ENVIRONMENT_CONTRACT_DIGEST,
        }
    finally:
        environment.close()


def test_beta_cli_override_constructs_a_complete_default_task() -> None:
    arguments = build_parser().parse_args(
        ["--environment", beta_spaces.ENVIRONMENT_ID, "--dry-run"]
    )
    config = resolve(arguments)

    assert config.environment_id == beta_spaces.ENVIRONMENT_ID
    assert config.beta_task == CyclicMarketTaskConfigV1()
    assert config.output_dir == "runs/v2-cyclic-market-beta-ppo"
    environment = make_env(config)
    try:
        observation, info = environment.reset(seed=912)
        assert observation.shape == (beta_spaces.OBSERVATION_SIZE,)
        assert info["environment_id"] == beta_spaces.ENVIRONMENT_ID
        dry_plan = plan(config)
        assert dry_plan["observation_shape"] == [beta_spaces.OBSERVATION_SIZE]
        assert dry_plan["action_shape"] == [beta_spaces.ACTION_SIZE]
        assert dry_plan["first_exogenous_episode_sha256"]
        assert dry_plan["first_physical_exogenous_episode_sha256"]
        assert dry_plan["first_exogenous_episode_sha256"] != (
            dry_plan["first_physical_exogenous_episode_sha256"]
        )
        assert dry_plan["training_library_imported"] is False
    finally:
        environment.close()


def test_beta_constructor_rejects_disturbance_geometry_before_reset() -> None:
    invalid_task = CyclicMarketTaskConfigV1(
        market=MarketConfigV1(forecast_ensemble_members=10),
        disturbances=DisturbanceConfigV1(
            duration_h=(1.01, 1.01),
        ),
    )

    with pytest.raises(ValueError, match="duration_h minimum must align"):
        PlantCyclicMarketEnvV2Beta(
            episode=_episode(),
            beta_task=invalid_task,
        )


def test_policy_evaluation_treats_cycle_violation_as_a_task_failure(
    tmp_path: Path,
) -> None:
    environment = PlantCyclicMarketEnvV2Beta(
        episode=_episode(),
        beta_task=_beta_task(),
    )
    try:
        checkpoint = checkpoints.write_fixed_checkpoint(
            tmp_path / "idle-beta-constraint-check",
            IDLE_ACTION,
            contract=policy_contract_for_environment(environment),
        )
        loaded = load_policy(
            checkpoint,
            contract=policy_contract_for_environment(environment),
        )
        result = execute_policy_episode(environment, loaded, seed=913)

        assert result["terminated"] is True
        assert result["truncated"] is False
        assert result["completed_decisions"] == 24
        assert result["status"] == "failed"
        assert result["cycle_constraint"]["all_settled_cycles_feasible"] is False
        assert result["cycle_constraint"]["cumulative_constraint_cost"] > 0.0
        assert result["failure"] == {
            "phase": "cycle_constraint",
            "exception_type": "CycleConstraintViolation",
            "message": "one or more 24 h resource-return constraints failed",
        }
    finally:
        environment.close()


def test_paired_beta_policy_evaluation_uses_identical_exogenous_episode(
    tmp_path: Path,
) -> None:
    config = _beta_training_config()
    environment = make_env(config)
    try:
        assert isinstance(environment, PlantCyclicMarketEnvV2Beta)
        checkpoint = _fixed_checkpoint(
            tmp_path,
            environment,
            name="fixed-beta-paired-evaluation",
        )
        loaded = load_policy(
            checkpoint,
            contract=policy_contract_for_environment(environment),
        )
    finally:
        environment.close()

    result = compare_policy_to_heuristic(loaded, config, seeds=(914,))

    assert result["seeds"] == [914]
    assert len(result["runs"]) == 2
    heuristic, policy = result["runs"]
    assert heuristic["manifest_identity"] == policy["manifest_identity"]
    assert heuristic["manifest_identity"]["beta_exogenous_episode_sha256"]
    assert heuristic["completed_decisions"] == policy["completed_decisions"] == 24
    assert heuristic["cycle_constraint"] is not None
    assert policy["cycle_constraint"] is not None

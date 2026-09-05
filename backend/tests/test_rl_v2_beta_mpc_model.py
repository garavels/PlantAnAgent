"""Structural audits for the causal hard-return beta MPC model."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from plant_sim import ComponentHealth

from plant_backend.rl.v2.beta_config import CyclicMarketTaskConfigV1
from plant_backend.rl.v2.beta_disturbances import DisturbanceConfigV1
from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta
from plant_backend.rl.v2.beta_control import CyclicMarketControllerObservationV1
from plant_backend.rl.v2.beta_mpc_model import (
    BetaMPCModelV1,
    build_beta_mpc_model,
)
from plant_backend.rl.v2.beta_mpc_schema import (
    BetaMPCInputError,
    BetaMPCPriceProjectionV1,
    BetaMPCSettingsV1,
    project_beta_mpc_prices,
)
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.economics import ch4_equivalent_inventory_kg
from plant_backend.rl.v2.market import MarketConfigV1
from plant_backend.rl.v2.mpc_forecast import project_issued_mean_solar


SETTINGS = BetaMPCSettingsV1(
    task_horizon_h=24.0,
    planning_horizon_h=24.0,
    interval_s=3_600.0,
)


def _episode() -> EpisodeConfigV2:
    return EpisodeConfigV2(
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


def _task() -> CyclicMarketTaskConfigV1:
    return CyclicMarketTaskConfigV1(
        market=MarketConfigV1(forecast_ensemble_members=10),
        disturbances=DisturbanceConfigV1(enabled=False, per_cycle_probability=0.0),
    )


def _environment() -> PlantCyclicMarketEnvV2Beta:
    environment = PlantCyclicMarketEnvV2Beta(episode=_episode(), beta_task=_task())
    environment.reset(seed=1701)
    return environment


def _build(
    environment: PlantCyclicMarketEnvV2Beta,
    *,
    observation: CyclicMarketControllerObservationV1 | None = None,
    price_projection: BetaMPCPriceProjectionV1 | None = None,
) -> BetaMPCModelV1:
    source = observation or environment.cyclic_market_controller_observation
    assert source.plant.forecast is not None
    horizon = source.cycle.cycle_decisions - source.cycle.decision_in_cycle
    weather = project_issued_mean_solar(
        source.plant.forecast,
        boundary_time_s=source.plant.state.boundary.time_s,
        config=environment.config,
        interval_s=SETTINGS.interval_s,
    )
    price = price_projection or project_beta_mpc_prices(
        source.market,
        interval_s=SETTINGS.interval_s,
        horizon_intervals=horizon,
    )
    boundary = source.plant.state.boundary
    return build_beta_mpc_model(
        source,
        weather,
        price,
        settings=SETTINGS,
        assumptions=environment.economic_assumptions,
        beta_task=environment.beta_task,
        constraints=environment.constraints,
        config=environment.config,
        commitment_profile=environment.episode_config.electrolyser_commitment,
        initial_ch4_equivalent_kg=ch4_equivalent_inventory_kg(
            boundary, environment.config
        ),
        initial_battery_energy_kwh=boundary.battery_energy_kwh,
    )


def test_model_adds_four_separate_hard_rows_at_exact_cycle_boundary() -> None:
    environment = _environment()
    try:
        model = _build(environment)

        assert model.horizon_intervals == 24
        assert model.cycle_return.next_boundary_decision == 24
        assert model.variable_count == model.economic_model.variable_count
        assert model.row_count == model.economic_model.row_count + 4
        assert model.nonzero_count == model.economic_model.nonzero_count + 4
        assert tuple(row.name for row in model.return_rows) == (
            "beta_cycle_return_battery",
            "beta_cycle_return_h2",
            "beta_cycle_return_co2",
            "beta_cycle_return_ch4",
        )
        groups = {
            "battery": "battery_energy_kwh",
            "h2": "h2_inventory_kg",
            "co2": "co2_inventory_kg",
            "ch4": "ch4_inventory_kg",
        }
        for row, requirement in zip(
            model.return_rows, model.cycle_return.resources, strict=True
        ):
            terminal = model.group(groups[requirement.resource])[-1]
            assert row.coefficients == ((terminal, 1.0),)
            assert row.lower == pytest.approx(
                max(0.0, requirement.target - requirement.tolerance)
            )
            assert row.upper is None
        assert not any("slack" in name for name in model.variable_names)
    finally:
        environment.close()


def test_horizon_shrinks_to_the_same_real_cycle_boundary_after_each_decision() -> None:
    environment = _environment()
    try:
        opening = _build(environment)
        environment.step(np.full(4, -1.0, dtype=np.float32))
        second = _build(environment)

        assert opening.horizon_intervals == 24
        assert second.horizon_intervals == 23
        assert opening.cycle_return.next_boundary_decision == 24
        assert second.cycle_return.next_boundary_decision == 24
        assert second.price_projection.prices_eur_per_mwh[0] == (
            environment.current_market_issue.current_eur_per_mwh
        )
        assert second.cycle_return.resource("battery").target == pytest.approx(
            opening.cycle_return.resource("battery").target
        )
    finally:
        environment.close()


def test_model_rejects_noncanonical_prices_and_stale_task_identity() -> None:
    environment = _environment()
    try:
        observation = environment.cyclic_market_controller_observation
        projection = project_beta_mpc_prices(
            observation.market,
            interval_s=SETTINGS.interval_s,
            horizon_intervals=24,
        )
        tampered = replace(
            projection,
            prices_eur_per_mwh=(
                projection.prices_eur_per_mwh[0] + 1.0,
                *projection.prices_eur_per_mwh[1:],
            ),
        )
        with pytest.raises(BetaMPCInputError, match="canonical path"):
            _build(environment, observation=observation, price_projection=tampered)

        stale = replace(observation, beta_task_sha256="0" * 64)
        with pytest.raises(BetaMPCInputError, match="task digest"):
            _build(environment, observation=stale)
    finally:
        environment.close()


def test_current_public_dac_outage_is_modeled_without_future_schedule() -> None:
    environment = _environment()
    try:
        observation = environment.cyclic_market_controller_observation
        health = dict(observation.plant.state.boundary.health)
        health["dac"] = ComponentHealth(status="offline", offline=True)
        boundary = replace(observation.plant.state.boundary, health=health)
        state = replace(observation.plant.state, boundary=boundary)
        public_outage = replace(
            observation,
            plant=replace(observation.plant, state=state),
        )
        model = _build(environment, observation=public_outage)
        physical_metadata = model.physical_model.metadata()

        assert model.variable_upper[model.group("co2_captured_kg")[0]] == 0.0
        assert (
            physical_metadata["causal_current_dac_health"]["dac_cap_kw_by_interval"][0]
            == 0.0
        )
        assert (
            physical_metadata["causal_current_dac_health"]["dac_cap_kw_by_interval"][1]
            > 0.0
        )
        information = model.metadata()["information_policy"]
        assert information["future_disturbance_schedule"] is False
        assert "schedule" not in model.to_dict()
    finally:
        environment.close()


def test_issue_quantiles_are_bound_even_when_objective_prices_do_not_change() -> None:
    environment = _environment()
    try:
        observation = environment.cyclic_market_controller_observation
        original = _build(environment, observation=observation)
        bins = list(observation.market.bins)
        bins[0] = replace(
            bins[0],
            p90_eur_per_mwh=min(300.0, bins[0].p90_eur_per_mwh + 0.5),
        )
        changed_issue = replace(observation.market, bins=tuple(bins))
        changed_observation = replace(observation, market=changed_issue)
        changed = _build(environment, observation=changed_observation)

        assert original.economic_model.interval_prices_eur_per_mwh == (
            changed.economic_model.interval_prices_eur_per_mwh
        )
        assert original.market_issue_sha256 != changed.market_issue_sha256
        assert original.price_projection.sha256 != changed.price_projection.sha256
        assert original.sha256 != changed.sha256
    finally:
        environment.close()

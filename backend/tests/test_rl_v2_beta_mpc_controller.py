"""Production-boundary tests for the cyclic-market economic MPC controller."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterator

import pytest

from plant_backend.rl.v2.beta_config import CyclicMarketTaskConfigV1
from plant_backend.rl.v2.beta_disturbances import DisturbanceConfigV1
from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta
from plant_backend.rl.v2.beta_mpc_controller import (
    CyclicMarketEconomicMPCControllerV1,
)
from plant_backend.rl.v2.beta_mpc_schema import (
    BETA_MPC_CONTROLLER_ID,
    BetaMPCDecisionV1,
    BetaMPCSettingsV1,
    BetaMPCSolveError,
)
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.market import MarketConfigV1
from plant_backend.rl.v2.setpoint_command import ProcessSetpointCommandV1
import plant_backend.rl.v2.beta_mpc_controller as controller_module


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
        disturbances=DisturbanceConfigV1(
            enabled=False,
            per_cycle_probability=0.0,
            duration_h=(1.0, 1.0),
        ),
    )


@pytest.fixture
def pair() -> (
    Iterator[tuple[PlantCyclicMarketEnvV2Beta, CyclicMarketEconomicMPCControllerV1]]
):
    task = _task()
    environment = PlantCyclicMarketEnvV2Beta(episode=_episode(), beta_task=task)
    environment.reset(seed=42)
    controller = CyclicMarketEconomicMPCControllerV1(
        BetaMPCSettingsV1(
            task_horizon_h=24.0,
            interval_s=3_600.0,
            solver_time_limit_s=10.0,
        ),
        environment.economic_assumptions,
        task,
    )
    try:
        yield environment, controller
    finally:
        environment.close()


def _compute(
    environment: PlantCyclicMarketEnvV2Beta,
    controller: CyclicMarketEconomicMPCControllerV1,
    observation: object | None = None,
) -> BetaMPCDecisionV1:
    source = (
        environment.cyclic_market_controller_observation
        if observation is None
        else observation
    )
    return controller.compute_command(  # type: ignore[arg-type]
        source,
        constraints=environment.constraints,
        config=environment.config,
        commitment_profile=environment.episode_config.electrolyser_commitment,
    )


def test_controller_issues_beta_evidence_and_exact_next_boundary_plan(
    pair: tuple[PlantCyclicMarketEnvV2Beta, CyclicMarketEconomicMPCControllerV1],
) -> None:
    pytest.importorskip("scipy")
    environment, controller = pair
    source = environment.cyclic_market_controller_observation

    decision = _compute(environment, controller, source)

    assert isinstance(decision, BetaMPCDecisionV1)
    assert isinstance(decision.command, ProcessSetpointCommandV1)
    assert decision.command.issuer_id == BETA_MPC_CONTROLLER_ID
    assert decision.to_dict()["command"]["issuer"]["id"] == BETA_MPC_CONTROLLER_ID
    assert BETA_MPC_CONTROLLER_ID == "cyclic-market-economic-mpc-v1"
    assert decision.command.source_sha256 == decision.evidence.sha256
    assert decision.evidence.beta_observation_sha256 == source.sha256
    assert decision.evidence.market_issue_sha256 == source.market.sha256
    assert decision.evidence.plan.horizon_intervals == 24
    assert decision.evidence.plan.cycle_return.remaining_intervals == 24
    assert decision.evidence.plan.cycle_return.next_boundary_decision == 24
    expected_prices = (
        source.market.current_eur_per_mwh,
        *(item.mean_eur_per_mwh for item in source.market.bins[1:]),
    )
    assert decision.evidence.plan.price_projection.prices_eur_per_mwh == (
        expected_prices
    )
    terminal = decision.evidence.plan.steps[-1]
    actual = {
        "battery": terminal.battery_energy_end_kwh,
        "h2": terminal.h2_end_kg,
        "co2": terminal.co2_end_kg,
        "ch4": terminal.ch4_end_kg,
    }
    for resource in decision.evidence.plan.cycle_return.resources:
        assert actual[resource.resource] >= resource.required_minimum - 1e-6

    _, _, terminated, truncated, _ = environment.step_process_setpoints(
        decision.command
    )
    assert terminated is False
    assert truncated is False


def test_descriptor_declares_beta_only_public_information_and_no_fallback(
    pair: tuple[PlantCyclicMarketEnvV2Beta, CyclicMarketEconomicMPCControllerV1],
) -> None:
    _, controller = pair
    descriptor = controller.to_dict()

    assert descriptor["id"] == "cyclic-market-economic-mpc-v1"
    assert descriptor["source_schema"] == (
        "cyclic-market-beta-mpc-decision-evidence-v1"
    )
    assert descriptor["beta_task"]["sha256"] == controller.beta_task.sha256  # type: ignore[index]
    assert "no controller-internal fallback" in str(descriptor["failure_policy"])
    assert "no market episode" in str(descriptor["information_set"])
    assert "future outage schedule" in str(descriptor["model_mismatch_declaration"])


def test_controller_accepts_only_exact_current_beta_contract(
    pair: tuple[PlantCyclicMarketEnvV2Beta, CyclicMarketEconomicMPCControllerV1],
) -> None:
    environment, template = pair
    source = environment.cyclic_market_controller_observation
    settings = template.settings
    assumptions = template.assumptions
    task = template.beta_task

    cases: tuple[tuple[object, str], ...] = (
        (source.plant, "CyclicMarketControllerObservationV1"),
        (
            replace(source, economic_assumptions_sha256="0" * 64),
            "assumptions digest",
        ),
        (replace(source, beta_task_sha256="0" * 64), "task digest"),
        (
            replace(
                source,
                market=replace(
                    source.market,
                    issued_at_decision=1,
                    issued_at_offset_s=3_600.0,
                ),
            ),
            "current decision",
        ),
        (
            replace(
                source,
                market=replace(source.market, issued_at_offset_s=1.0),
            ),
            "issue offset",
        ),
        (
            replace(
                source,
                cycle=replace(
                    source.cycle,
                    decision_in_cycle=1,
                    progress=1.0 / 24.0,
                ),
            ),
            "cycle counters",
        ),
        (
            replace(
                source,
                plant=replace(source.plant, decision_interval_s=900.0),
            ),
            "cadence",
        ),
        (
            replace(
                source,
                plant=replace(
                    source.plant,
                    forecast=replace(
                        source.plant.forecast,
                        issued_at_offset_s=3_600.0,
                    ),
                ),
            ),
            "solar forecast offset",
        ),
    )

    for corrupted, message in cases:
        controller = CyclicMarketEconomicMPCControllerV1(
            settings,
            assumptions,
            task,
        )
        with pytest.raises((TypeError, ValueError), match=message):
            _compute(environment, controller, corrupted)
        assert controller.decision_events == ()


def test_cycle_targets_are_immutable_and_horizon_closes_by_one_interval(
    pair: tuple[PlantCyclicMarketEnvV2Beta, CyclicMarketEconomicMPCControllerV1],
) -> None:
    pytest.importorskip("scipy")
    environment, controller = pair
    first = _compute(environment, controller)
    environment.step_process_setpoints(first.command)
    current = environment.cyclic_market_controller_observation
    corrupted = replace(
        current,
        cycle=replace(
            current.cycle,
            battery_margin_kwh=current.cycle.battery_margin_kwh + 0.01,
        ),
    )

    with pytest.raises(ValueError, match="cycle targets changed"):
        _compute(environment, controller, corrupted)
    assert len(controller.decision_events) == 1

    second = _compute(environment, controller, current)
    assert second.evidence.decision_index == 1
    assert second.evidence.plan.horizon_intervals == 23
    assert second.evidence.plan.cycle_return.next_boundary_decision == 24
    assert second.evidence.plan.price_projection.prices_eur_per_mwh[0] == (
        current.market.current_eur_per_mwh
    )


def test_solver_failure_is_not_replaced_by_another_controller(
    pair: tuple[PlantCyclicMarketEnvV2Beta, CyclicMarketEconomicMPCControllerV1],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment, controller = pair
    real_solver = controller_module.solve_beta_mpc_model

    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise BetaMPCSolveError("deliberate beta solver failure")

    monkeypatch.setattr(controller_module, "solve_beta_mpc_model", fail)
    with pytest.raises(BetaMPCSolveError, match="deliberate"):
        _compute(environment, controller)
    assert controller.decision_events == ()

    monkeypatch.setattr(controller_module, "solve_beta_mpc_model", real_solver)
    decision = _compute(environment, controller)
    assert decision.evidence.decision_index == 0
    assert len(controller.decision_events) == 1


def test_rejected_boundary_does_not_poison_controller_identity_state(
    pair: tuple[PlantCyclicMarketEnvV2Beta, CyclicMarketEconomicMPCControllerV1],
) -> None:
    pytest.importorskip("scipy")
    environment, controller = pair
    source = environment.cyclic_market_controller_observation
    assert source.plant.forecast is not None
    corrupted = replace(
        source,
        market=replace(source.market, market_id="unrelated-market"),
        plant=replace(
            source.plant,
            forecast=replace(source.plant.forecast, issued_at_offset_s=3_600.0),
        ),
    )

    with pytest.raises(ValueError, match="solar forecast offset"):
        _compute(environment, controller, corrupted)
    assert controller.decision_events == ()

    decision = _compute(environment, controller, source)
    assert decision.evidence.market_id == source.market.market_id
    assert len(controller.decision_events) == 1

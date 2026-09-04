"""Outcome-free admission tests for the registered economic mechanism smoke."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any

import pytest
from plant_sim import PlantConfig

from plant_backend.controllers.base import Constraints
from plant_backend.rl.v2 import economic_smoke_experiment as smoke
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.economic_env import PlantEconomicSetpointEnvV2
from plant_backend.rl.v2.economics import EconomicAssumptionsV1
from plant_backend.rl.v2.mpc_schema import (
    SERVICE_MPC_SETPOINT_CONTROLLER_ID,
    ServiceMPCSetpointSettingsV1,
)
from plant_backend.rl.v2.mpc_setpoint_run import verify_mpc_setpoint_run_digests
from plant_backend.rl.v2.reward import RewardWeightsV2
from plant_backend.rl.v2.setpoint_controllers import (
    LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID,
)


def _tiny_episode() -> EpisodeConfigV2:
    return EpisodeConfigV2(
        horizon_h=0.25,
        delivery_period_h=0.25,
        delivery_rate_kg_h=1.0,
        battery_soc=(0.5, 0.5),
        h2_fill=(0.0, 0.0),
        co2_fill=(0.0, 0.0),
        ch4_coverage_h=(0.0, 0.0),
    )


def _complete_gate_arm(
    arm: str, *, value_eur: float, solar_kwh: float
) -> dict[str, Any]:
    components = {
        "delivered_service_value": value_eur,
        "unmet_delivery_penalty": 0.0,
        "solar_opportunity_cost": 0.0,
        "battery_throughput_reserve": 0.0,
        "electrolyser_life_reserve": 0.0,
        "methane_endowment_liability_change": 0.0,
        "battery_endowment_liability_change": 0.0,
    }
    economic = {
        "net_operating_value_eur": value_eur,
        "components_eur": components,
        "delivered_kg": 210.0,
        "unmet_kg": 0.0,
        "accepted_solar_kwh": solar_kwh,
        "battery_throughput_kwh": 20.0,
        "electrolyser_equivalent_operating_h": 3.0,
        "terminal_ch4_equivalent_deficit_kg": 0.0,
        "terminal_battery_deficit_kwh": 0.0,
    }
    return {
        "arm": arm,
        "status": "complete",
        "manifest_identity": {
            "manifest_sha256": smoke.ECONOMIC_SMOKE_EXPECTED_MANIFEST_SHA256,
            "physical_exogenous_episode_sha256": (
                smoke.ECONOMIC_SMOKE_EXPECTED_PHYSICAL_EPISODE_SHA256
            ),
            "economic_episode_sha256": smoke.ECONOMIC_SMOKE_EXPECTED_EPISODE_SHA256,
            "checkpoint_contract_sha256": (
                smoke.ECONOMIC_SMOKE_EXPECTED_CHECKPOINT_SHA256
            ),
            "economic_assumptions_sha256": (
                smoke.ECONOMIC_SMOKE_EXPECTED_ASSUMPTIONS_SHA256
            ),
        },
        "completed_decisions": smoke.ECONOMIC_SMOKE_EXPECTED_DECISIONS,
        "physical_steps": smoke.ECONOMIC_SMOKE_EXPECTED_PHYSICAL_STEPS,
        "terminated": True,
        "truncated": False,
        "reward_trace_sum_eur": value_eur,
        "interval_totals": {
            "delivered_kg": economic["delivered_kg"],
            "unmet_kg": economic["unmet_kg"],
            "accepted_solar_kwh": economic["accepted_solar_kwh"],
            "battery_throughput_kwh": economic["battery_throughput_kwh"],
            "electrolyser_equivalent_operating_h": economic[
                "electrolyser_equivalent_operating_h"
            ],
        },
        "execution_count": smoke.ECONOMIC_SMOKE_EXPECTED_DECISIONS,
        "application_count": smoke.ECONOMIC_SMOKE_EXPECTED_PHYSICAL_STEPS,
        "physical_metrics": {
            "shutdown_steps": 0,
            "out_of_range_decisions": 0,
            "electrolyser_forced_stops": 0,
            "delivered_kg": economic["delivered_kg"],
            "unmet_kg": economic["unmet_kg"],
        },
        "economic_metrics": economic,
    }


def test_frozen_contract_uses_real_controller_and_source_identities() -> None:
    repository = Path(__file__).resolve().parents[2]
    protocol = repository / smoke.ECONOMIC_SMOKE_PROTOCOL_PATH

    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == (
        smoke.ECONOMIC_SMOKE_PROTOCOL_SHA256
    )
    assert hashlib.sha256((repository / "uv.lock").read_bytes()).hexdigest() == (
        smoke.ECONOMIC_SMOKE_LOCK_SHA256
    )
    assert smoke.ECONOMIC_SMOKE_CONTRACT["arms"]["H"]["controller_id"] == (
        LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID
    )
    assert smoke.ECONOMIC_SMOKE_CONTRACT["arms"]["M"]["controller_id"] == (
        SERVICE_MPC_SETPOINT_CONTROLLER_ID
    )


def test_frozen_reset_identities_match_before_either_arm_moves() -> None:
    plant = PlantConfig()
    episode = smoke.fixed_episode()
    constraints = Constraints()
    weights = RewardWeightsV2()
    assumptions = EconomicAssumptionsV1()
    environments = [
        PlantEconomicSetpointEnvV2(
            config=plant,
            episode=episode,
            constraints=constraints,
            weights=weights,
            economic_assumptions=assumptions,
        )
        for _ in range(2)
    ]
    identities = [
        smoke._reset_environment(environment, seed=smoke.ECONOMIC_SMOKE_SEED)
        for environment in environments
    ]

    smoke._validate_preflight_identities(*identities)
    assert identities[0] == identities[1]
    assert all(environment.decisions_taken == 0 for environment in environments)
    assert all(environment.physics_steps_taken == 0 for environment in environments)


def test_tiny_heuristic_and_mpc_runs_cross_the_economic_setpoint_doors() -> None:
    episode = _tiny_episode()
    plant = PlantConfig()
    constraints = Constraints()
    weights = RewardWeightsV2()
    assumptions = EconomicAssumptionsV1()
    heuristic = smoke.execute_heuristic_arm(
        episode=episode,
        plant=plant,
        constraints=constraints,
        weights=weights,
        assumptions=assumptions,
        seed=0,
    )
    settings = ServiceMPCSetpointSettingsV1(
        task_horizon_h=episode.horizon_h,
        planning_horizon_h=episode.horizon_h,
    )
    mpc = smoke.execute_mpc_arm(
        episode=episode,
        plant=plant,
        constraints=constraints,
        weights=weights,
        assumptions=assumptions,
        settings=settings,
        seed=0,
    )

    for run in (heuristic, mpc):
        assert run["status"] == "complete"
        assert run["completed_decisions"] == 1
        assert run["physical_steps"] == 15
        assert run["execution_count"] == 1
        assert run["application_count"] == 15
        assert run["economic_metrics"]["decisions"] == 1
        assert run["reward_trace_sum_eur"] == pytest.approx(
            run["economic_metrics"]["net_operating_value_eur"]
        )
    assert heuristic["controller"]["id"] == LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID
    assert mpc["controller"]["id"] == SERVICE_MPC_SETPOINT_CONTROLLER_ID
    verify_mpc_setpoint_run_digests(mpc["evidence"])


def test_prepared_arm_rejects_an_environment_that_already_moved() -> None:
    environment = PlantEconomicSetpointEnvV2(episode=_tiny_episode())
    observation, _ = environment.reset(seed=0)
    environment.step(environment.action_space.sample())
    assert observation.shape == environment.observation_space.shape

    with pytest.raises(ValueError, match="already moved"):
        smoke.execute_heuristic_arm(
            episode=environment.episode_config,
            plant=environment.config,
            constraints=environment.constraints,
            weights=environment.weights,
            assumptions=environment.economic_assumptions,
            seed=0,
            environment=environment,
        )


def test_gate_rule_passes_only_a_complete_paired_discriminative_result() -> None:
    heuristic = _complete_gate_arm("H", value_eur=10.0, solar_kwh=100.0)
    mpc = _complete_gate_arm("M", value_eur=12.0, solar_kwh=102.0)

    gates = smoke.evaluate_smoke_gates(
        heuristic,
        mpc,
        assumptions=EconomicAssumptionsV1(),
    )

    assert [(gate["id"], gate["status"]) for gate in gates] == [
        ("E1", "passed"),
        ("E2", "passed"),
        ("E3", "passed"),
        ("E4", "passed"),
        ("E5", "passed"),
        ("E6", "reported"),
    ]
    assert gates[-1]["observation"]["descriptive_higher_value_arm"] == "M"


def test_gate_rule_rejects_service_confounding_and_accounting_drift() -> None:
    heuristic = _complete_gate_arm("H", value_eur=10.0, solar_kwh=100.0)
    mpc = _complete_gate_arm("M", value_eur=12.0, solar_kwh=102.0)
    service_confounded = deepcopy(mpc)
    service_confounded["physical_metrics"]["delivered_kg"] += 1.0
    service_gates = smoke.evaluate_smoke_gates(
        heuristic,
        service_confounded,
        assumptions=EconomicAssumptionsV1(),
    )
    assert service_gates[2]["status"] == "failed"
    assert service_gates[3]["status"] == "failed"
    assert service_gates[5]["status"] == "not_evaluated"

    accounting_drift = deepcopy(mpc)
    accounting_drift["reward_trace_sum_eur"] += 0.01
    accounting_gates = smoke.evaluate_smoke_gates(
        heuristic,
        accounting_drift,
        assumptions=EconomicAssumptionsV1(),
    )
    assert accounting_gates[4]["status"] == "failed"
    assert accounting_gates[5]["status"] == "not_evaluated"


def test_gate_rule_retains_an_incomplete_arm_with_missing_accounting_components() -> (
    None
):
    heuristic = _complete_gate_arm("H", value_eur=10.0, solar_kwh=100.0)
    mpc = _complete_gate_arm("M", value_eur=0.0, solar_kwh=0.0)
    mpc["status"] = "failed"
    mpc["completed_decisions"] = 0
    mpc["physical_steps"] = 0
    mpc["terminated"] = False
    mpc["execution_count"] = 0
    mpc["application_count"] = 0
    mpc["economic_metrics"]["components_eur"] = {}

    gates = smoke.evaluate_smoke_gates(
        heuristic,
        mpc,
        assumptions=EconomicAssumptionsV1(),
    )

    assert gates[0]["status"] == "failed"
    assert gates[4]["status"] == "failed"
    assert gates[4]["observation"]["M"] == {"evidence_complete": False}
    assert gates[4]["observation"]["accounting_evidence_failure"]["M"]
    assert gates[5]["status"] == "not_evaluated"

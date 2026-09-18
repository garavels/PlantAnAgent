"""Task identity, causal electricity semantics and unchanged physics checks."""

from copy import deepcopy
import json

import numpy as np
import pytest

from docs.research.tools.feasible_benchmark_v1 import (
    BenchmarkEnv,
    ElectricityModelV1,
    ENVIRONMENT_ID,
    load_benchmark_contract,
)
from docs.research.tools.feasible_mission_controller import DailyCoverageController
from docs.research.tools.feasible_mission_replay import canonical_sha
from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta


def test_contract_checksum_and_disjoint_splits(tmp_path):
    contract = load_benchmark_contract()
    assert len(contract["splits"]["train"]) == 100
    assert len(contract["splits"]["development"]) == 8
    assert len(contract["splits"]["held_out"]) == 16
    modified = deepcopy(contract)
    modified["delivery_rate_kg_h"] = 40
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="checksum"):
        load_benchmark_contract(path)
    modified = deepcopy(contract)
    modified["splits"]["development"][0] = modified["splits"]["held_out"][0]
    modified["contract_sha256"] = canonical_sha(
        {k: v for k, v in modified.items() if k != "contract_sha256"}
    )
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="overlap"):
        load_benchmark_contract(path)


def test_new_task_samples_exact_original_inventory_and_forcing():
    env = BenchmarkEnv(split="development")
    cfg = env.source_config
    original = PlantCyclicMarketEnvV2Beta(
        config=cfg.plant,
        episode=cfg.episode,
        constraints=cfg.constraints,
        weights=cfg.diagnostic_reward,
        economic_assumptions=cfg.economic_assumptions,
        beta_task=cfg.beta_task,
    )
    try:
        obs, info = env.reset(seed=9920000)
        original.reset(seed=9920000)
        assert obs.dtype == np.float32 and obs.shape == (135,)
        assert env.simulator.state.to_dict() == original.simulator.state.to_dict()
        assert env._episode.weather == original._episode.weather
        assert env.market_episode == original.market_episode
        assert env.disturbance_episode == original.disturbance_episode
        assert [c.requested_kg for c in env._episode.commitments] == [150.0] * 28
        assert [c.requested_kg for c in original._episode.commitments] == [240.0] * 28
        assert env.cycle_reference == original.cycle_reference
        assert info["environment_id"] == ENVIRONMENT_ID
        assert env.checkpoint_contract_digest != original.checkpoint_contract_digest
        assert (
            env.checkpoint_contract["electricity_model"]["mode"]
            == "dedicated_pv_no_export"
        )
        assert all(x == 0 for x in env.current_market_issue.normalized())
        assert original.current_market_issue.current_eur_per_mwh != 0
        with pytest.raises(ValueError, match="outside benchmark split"):
            env.reset(seed=9930000)
    finally:
        env.close()
        original.close()


def test_offgrid_zero_charge_matches_actual_ledger_without_changing_physics():
    offgrid = BenchmarkEnv(split="development")
    export = BenchmarkEnv(
        split="development",
        electricity_model=ElectricityModelV1(mode="solar_export_opportunity"),
    )
    controller = DailyCoverageController()
    try:
        offgrid.reset(seed=9920000)
        export.reset(seed=9920000)
        assert offgrid.checkpoint_contract_digest != export.checkpoint_contract_digest
        assert (
            offgrid.audit_manifest()["manifest_sha256"]
            != export.audit_manifest()["manifest_sha256"]
        )
        for _ in range(24):
            decision = controller.compute_command(
                offgrid.cyclic_market_controller_observation,
                constraints=offgrid.constraints,
                config=offgrid.config,
                commitment_profile=offgrid.episode_config.electrolyser_commitment,
            )
            a = offgrid.step_process_setpoints(decision.command)
            b = export.step_process_setpoints(decision.command)
            assert offgrid.simulator.state.to_dict() == export.simulator.state.to_dict()
            assert a[4]["reward"]["components_eur"]["solar_opportunity_cost"] == 0
            assert a[1] == pytest.approx(sum(a[4]["reward"]["components_eur"].values()))
            assert (
                a[4]["cycle_constraint_transition"]
                == b[4]["cycle_constraint_transition"]
            )
            for key, value in a[4]["reward"]["components_eur"].items():
                if key != "solar_opportunity_cost":
                    assert value == b[4]["reward"]["components_eur"][key]
            assert a[4]["environment_id"] == ENVIRONMENT_ID
        e, o = export.economic_metrics.to_dict(), offgrid.economic_metrics.to_dict()
        charge = -e["components_eur"]["solar_opportunity_cost"]
        assert charge > 0
        assert o["net_operating_value_eur"] - e[
            "net_operating_value_eur"
        ] == pytest.approx(charge)
        assert offgrid.electricity_model.solar_capital_recovery_eur_per_week is None
        assert offgrid.electricity_model.solar_fixed_om_eur_per_week is None
    finally:
        offgrid.close()
        export.close()


@pytest.mark.parametrize("price", [None, True, -1, float("nan"), 301])
def test_electricity_prices_fail_before_environment_reset(price):
    with pytest.raises(ValueError, match="Marginal solar price"):
        ElectricityModelV1(marginal_solar_cost_eur_per_mwh=price)

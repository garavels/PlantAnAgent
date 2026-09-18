import numpy as np
import pytest

from docs.research.tools.authority_ppo_v2.benchmark import BenchmarkEnv, load_benchmark_contract, ENVIRONMENT_ID
from docs.research.tools.feasible_benchmark_v1 import BenchmarkEnv as OldEnv
from docs.research.tools.offgrid_coverage_v1 import OffgridCoverageController


def test_contract_split_and_budget():
    c = load_benchmark_contract()
    assert len(c['splits']['train']) == 392
    assert len(c['splits']['held_out']) == 32
    assert c['training']['total_timesteps_per_optimizer'] == 262144


def test_unopened_split_does_not_change_checkpoint_identity():
    a, b = BenchmarkEnv('train'), BenchmarkEnv('held_out')
    try:
        assert a.checkpoint_contract == b.checkpoint_contract
        assert a.checkpoint_contract['environment_id'] == ENVIRONMENT_ID
        with pytest.raises(ValueError):
            a.reset(seed=9960000)
        a.reset(seed=9940391)
        assert a.frozen_source.source_pin['path'].endswith('authority_ppo_v2.json')
        assert a.audit_manifest()['environment']['id'] == ENVIRONMENT_ID
    finally:
        a.close(); b.close()


def test_exposed_diagnostic_case_preserves_exact_physics_and_observations():
    a, b = BenchmarkEnv('diagnostic'), OldEnv('held_out')
    ca, cb = OffgridCoverageController(), OffgridCoverageController()
    try:
        va, _ = a.reset(seed=9930001)
        vb, _ = b.reset(seed=9930001)
        assert np.array_equal(va, vb)
        for _ in range(96):
            results = []
            for env, controller in ((a, ca), (b, cb)):
                command = controller.compute_command(
                    env.cyclic_market_controller_observation,
                    constraints=env.constraints, config=env.config,
                    commitment_profile=env.episode_config.electrolyser_commitment)
                results.append(env.step_process_setpoints(command.command, controller_invalid=command.invalid))
            assert a.simulator.state.to_dict() == b.simulator.state.to_dict()
            assert np.array_equal(results[0][0], results[1][0])
            assert results[0][1:4] == results[1][1:4]
        assert a.cycle_settlements == b.cycle_settlements
    finally:
        a.close(); b.close()

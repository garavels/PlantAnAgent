"""Recovery labels and clocks; no backend source or frozen policy is changed."""

from types import SimpleNamespace

import pytest

from docs.research.tools.feasible_benchmark_v1 import BenchmarkEnv
from docs.research.tools.mission_mpc_recovery_v1 import MissionMPCRecoveryControllerV1
from plant_backend.rl.v2.economic_mpc_schema import EconomicMPCSolveError


def test_recovery_labels_and_retry_clock(monkeypatch):
    env = BenchmarkEnv(split="development")
    env.reset(seed=9920003)
    controller = MissionMPCRecoveryControllerV1(
        horizon_h=168,
        interval_s=900,
        assumptions=env.economic_assumptions,
        beta_task=env.beta_task,
        electricity_model=env.electricity_model.to_dict(),
    )
    calls = []

    def reject(observation, **kwargs):
        calls.append(observation.market.issued_at_decision)
        raise EconomicMPCSolveError("test infeasible hard model", stages=())

    monkeypatch.setattr(controller.mpc, "compute_command", reject)
    arguments = dict(
        constraints=env.constraints,
        config=env.config,
        commitment_profile=env.episode_config.electrolyser_commitment,
    )
    try:
        for index in range(2):
            public = env.cyclic_market_controller_observation
            selected = controller.compute_command(public, **arguments)
            assert selected.command.decision_index == index
            assert selected.command.issuer_id == controller.id
            assert selected.evidence["decision_mode"] == "coverage_recovery"
            assert selected.evidence["audited_hard_return_plan"] is False
            assert selected.evidence["rejection"]["exception_type"] == "EconomicMPCSolveError"
            assert controller.mpc._next_decision == controller.coverage.decisions == index + 1
            env.step_process_setpoints(selected.command)
        assert calls == [0, 1]
        reference = controller.mpc._reference

        def accepted(public, **kwargs):
            # A schema-compatible accepted command fixture isolates wrapper
            # bookkeeping; the underlying hard solver has its own real tests.
            controller.coverage.decisions = 2
            selected = controller.coverage.compute_command(public, **kwargs)
            controller.mpc._next_decision += 1
            return SimpleNamespace(
                command=selected.command,
                invalid=selected.invalid,
                evidence=SimpleNamespace(to_dict=lambda: {"fixture": "accepted"}),
            )

        monkeypatch.setattr(controller.mpc, "compute_command", accepted)
        selected = controller.compute_command(env.cyclic_market_controller_observation, **arguments)
        assert selected.command.decision_index == 2
        assert selected.evidence["decision_mode"] == "hard_mpc_plan"
        assert selected.evidence["audited_hard_return_plan"] is True
        assert selected.evidence["rejection"] is None
        assert controller.mpc._reference == reference
        assert len(controller.decision_events) == 3
        with pytest.raises(ValueError, match="stale or mismatched"):
            controller.compute_command(env.cyclic_market_controller_observation, **arguments)
        controller.reset()
        assert controller.mpc._next_decision == controller.coverage.decisions == 0
        assert controller.decision_events == ()
    finally:
        env.close()

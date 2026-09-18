"""Focused adversarial checks for the separate cyclic-market beta benchmark."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from plant_backend.rl import checkpoints
from plant_backend.rl.policies import LoadedPolicy, load_policy
from plant_backend.rl.v2 import (
    beta_controller_benchmark as benchmark,
    beta_spaces,
    economic_spaces,
)
from plant_backend.rl.v2.beta_config import CyclicMarketTaskConfigV1
from plant_backend.rl.v2.beta_control import CyclicMarketControllerObservationV1
from plant_backend.rl.v2.beta_disturbances import DisturbanceConfigV1
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.policy_setpoint_controller import (
    NormalizedSetpointPolicyControllerV1,
    policy_contract_for_environment,
)
from plant_backend.rl.v2.mpc_setpoint_controller import (
    ServiceMPCSetpointControllerV1,
)
from plant_backend.rl.v2.setpoint_controllers import (
    LegacyHeuristicSetpointControllerV1,
)
from plant_backend.rl.v2.training.config import ControllerNeutralTrainingConfigV1
from plant_backend.rl.v2.training.ppo import make_env


def _config() -> ControllerNeutralTrainingConfigV1:
    return ControllerNeutralTrainingConfigV1(
        environment_id=beta_spaces.ENVIRONMENT_ID,
        episode=EpisodeConfigV2(
            horizon_h=24.0,
            decision_interval_s=3_600.0,
            start_hour=6.0,
            delivery_period_h=24.0,
            delivery_rate_kg_h=1.0,
            battery_soc=(0.5, 0.5),
            h2_fill=(0.1, 0.1),
            co2_fill=(0.1, 0.1),
            ch4_coverage_h=(1.0, 1.0),
        ),
        beta_task=CyclicMarketTaskConfigV1(),
        total_timesteps=4,
        n_envs=1,
        checkpoint_every=0,
        notes="beta benchmark integration fixture; not trained",
    ).validate()


def _loaded(
    tmp_path: Path,
    config: ControllerNeutralTrainingConfigV1,
    *,
    name: str = "beta-policy",
) -> LoadedPolicy:
    environment = make_env(config)
    try:
        contract = policy_contract_for_environment(environment)
        path = checkpoints.write_fixed_checkpoint(
            tmp_path / name,
            np.full(4, -1.0, dtype=np.float32),
            contract=contract,
        )
        metadata = checkpoints.load_metadata(path, check=False)
        checkpoints.save_metadata(
            path,
            replace(metadata, training_config=config.to_dict()),
        )
        return load_policy(path, contract=contract, environment=environment)
    finally:
        environment.close()


class _FailingBetaController:
    id = "test-failing-beta-controller"

    def reset(self) -> None:
        return None

    def compute_command(self, observation: object, **_: object) -> object:
        assert isinstance(observation, CyclicMarketControllerObservationV1)
        raise RuntimeError("deliberate beta economic controller failure")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "controller_fallback": None, "test_only": True}


class _TamperingController:
    id = "test-tampering-controller"

    def __init__(self, field: str) -> None:
        self._field = field
        self._heuristic = LegacyHeuristicSetpointControllerV1()

    def reset(self) -> None:
        self._heuristic.reset()

    def compute_command(self, observation: object, **kwargs: object) -> object:
        physical = (
            observation.plant
            if isinstance(observation, CyclicMarketControllerObservationV1)
            else observation
        )
        base = self._heuristic.compute_command(physical, **kwargs)  # type: ignore[arg-type]
        if self._field == "decision_index":
            command = replace(base.command, decision_index=99)
            evidence = base.evidence
        elif self._field == "observation_sha256":
            command = replace(base.command, observation_sha256="0" * 64)
            evidence = base.evidence
        elif self._field == "beta_observation_sha256":
            command = base.command
            evidence = SimpleNamespace(
                beta_observation_sha256="0" * 64,
                physical_observation_sha256=base.command.observation_sha256,
                market_issue_sha256=observation.market.sha256,  # type: ignore[union-attr]
                cycle_observation_sha256="0" * 64,
            )
        else:  # pragma: no cover - test constructor owns this surface
            raise AssertionError("unsupported test tamper")
        return SimpleNamespace(
            command=command,
            evidence=evidence,
            invalid=False,
            to_dict=lambda: {"test_tamper": self._field},
        )

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "test_tamper": self._field}


def test_production_beta_economic_factory_consumes_the_typed_beta_boundary() -> None:
    pytest.importorskip("scipy")
    config = _config()
    environment = benchmark._new_beta_environment(config)
    try:
        environment.reset(seed=30)
        controller = benchmark._real_beta_economic_controller(config)
        controller.reset()

        decision = controller.compute_command(
            environment.cyclic_market_controller_observation,
            constraints=environment.constraints,
            config=environment.config,
            commitment_profile=environment.episode_config.electrolyser_commitment,
        )
    finally:
        environment.close()

    assert controller.to_dict()["id"] == "cyclic-market-economic-mpc-v1"
    assert decision.to_dict()["command"]["issuer"]["id"] == (
        "cyclic-market-economic-mpc-v1"
    )


def test_beta_policy_arm_records_every_typed_boundary_and_terminal(
    tmp_path: Path,
) -> None:
    config = _config()
    loaded = _loaded(tmp_path, config)
    environment = benchmark._new_beta_environment(config)
    try:
        initial, _ = environment.reset(seed=31)
        identity = benchmark._beta_manifest_identity(environment)
        controller = NormalizedSetpointPolicyControllerV1.from_environment(
            loaded, environment
        )
        controller.reset()

        run = benchmark._generic_arm_result(
            arm=benchmark.PPO_ARM,
            environment=environment,
            controller=controller,
            manifest_identity=identity,
            initial_vector=initial,
        )
    finally:
        environment.close()

    assert run["execution_complete"] is True
    assert run["completed_decisions"] == 24
    assert run["physical_steps"] == 1_440
    assert run["failure"] is None
    assert len(run["cycle_settlements"]) == 1
    ledger = run["evidence"]["boundary_ledger"]
    assert ledger["format"] == benchmark.BETA_BOUNDARY_LEDGER_FORMAT
    assert len(ledger["decision_boundaries"]) == 24
    assert [item["decision_index"] for item in ledger["decision_boundaries"]] == (
        list(range(24))
    )
    assert all(
        item["typed_observation"]["schema"] == "cyclic-market-controller-observation-v1"
        and len(item["normalized_observation"]) == beta_spaces.OBSERVATION_SIZE
        and len(item["typed_observation_sha256"]) == 64
        and len(item["normalized_observation_sha256"]) == 64
        for item in ledger["decision_boundaries"]
    )
    assert ledger["final_boundary"]["terminal"] is True
    assert ledger["final_boundary"]["decision_index"] == 24
    truth_states = run["evidence"]["minute_truth_states"]
    assert len(truth_states) == run["physical_steps"] + 1
    assert [state["step"] for state in truth_states] == list(range(1_441))
    assert (
        min(state["battery_soc"] for state in truth_states)
        == run["physical_metrics"]["min_battery_soc"]
    )
    assert (
        sum(
            state["plant_status"] in {"critical", "offline"}
            for state in truth_states[1:]
        )
        == run["physical_metrics"]["shutdown_steps"]
    )
    assert run["information_profile"]["profile"] == ("full-public-beta-learned-v1")
    assert set(identity) == {
        "manifest_sha256",
        "alpha_physical_exogenous_episode_sha256",
        "beta_exogenous_episode_sha256",
        "market_episode_sha256",
        "disturbance_episode_sha256",
        "disturbance_schedule_sha256",
        "cycle_task_sha256",
        "checkpoint_contract_sha256",
        "economic_assumptions_sha256",
        "beta_task_sha256",
    }


def test_boundary_capture_rejects_stale_vectors_and_refreshes_final_state() -> None:
    config = _config()
    environment = benchmark._new_beta_environment(config)
    try:
        initial, _ = environment.reset(seed=39)
        stale = initial.copy()
        stale[0] = np.float32(stale[0] + 0.25)
        with pytest.raises(RuntimeError, match="returned beta vector differs"):
            benchmark._boundary_record(environment, stale, kind="decision")

        final, failure = benchmark._final_boundary(environment)
        expected = [float(value) for value in environment._observe()]
    finally:
        environment.close()

    assert failure is None
    assert final is not None
    assert final["normalized_observation"] == expected


def test_failed_beta_arm_is_retained_without_fallback_and_keeps_final_boundary(
    tmp_path: Path,
) -> None:
    config = _config()
    environment = benchmark._new_beta_environment(config)
    try:
        initial, _ = environment.reset(seed=32)
        controller = _FailingBetaController()
        controller.reset()
        run = benchmark._generic_arm_result(
            arm=benchmark.BETA_ECONOMIC_MPC_ARM,
            environment=environment,
            controller=controller,
            manifest_identity=benchmark._beta_manifest_identity(environment),
            initial_vector=initial,
        )
    finally:
        environment.close()

    assert run["status"] == "execution_failed"
    assert run["execution_complete"] is False
    assert run["cycle_feasible"] is False
    assert run["value_comparison_eligible"] is False
    assert run["completed_decisions"] == run["physical_steps"] == 0
    assert run["failure"]["exception_type"] == "RuntimeError"
    assert "deliberate beta economic" in run["failure"]["message"]
    ledger = run["evidence"]["boundary_ledger"]
    assert len(ledger["decision_boundaries"]) == 1
    assert ledger["final_boundary"]["terminal"] is False
    assert ledger["final_boundary"]["typed_observation_sha256"] == (
        ledger["decision_boundaries"][0]["typed_observation_sha256"]
    )
    assert ledger["final_boundary"]["normalized_observation"] == (
        ledger["decision_boundaries"][0]["normalized_observation"]
    )
    assert ledger["final_boundary"]["normalized_observation_sha256"] == (
        ledger["decision_boundaries"][0]["normalized_observation_sha256"]
    )
    assert run["controller"]["controller_fallback"] is None


def test_post_movement_generic_evidence_corruption_aborts_the_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    loaded = _loaded(tmp_path, config)
    environment = benchmark._new_beta_environment(config)
    try:
        initial, _ = environment.reset(seed=40)
        controller = NormalizedSetpointPolicyControllerV1.from_environment(
            loaded, environment
        )
        controller.reset()
        monkeypatch.setattr(
            benchmark,
            "_economic_interval",
            lambda _info: (_ for _ in ()).throw(RuntimeError("malformed interval")),
        )

        with pytest.raises(RuntimeError, match="complete verifiable interval bundle"):
            benchmark._generic_arm_result(
                arm=benchmark.PPO_ARM,
                environment=environment,
                controller=controller,
                manifest_identity=benchmark._beta_manifest_identity(environment),
                initial_vector=initial,
            )
        assert environment.decisions_taken == 1
        assert environment.physics_steps_taken == 60
    finally:
        environment.close()


def test_post_movement_m24_reconciliation_corruption_aborts_the_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("scipy")
    config = _config()
    settings = benchmark._service_settings(config, None)
    environment = benchmark._new_beta_environment(config)
    try:
        initial, _ = environment.reset(seed=41)
        controller = ServiceMPCSetpointControllerV1(settings)
        controller.reset()
        monkeypatch.setattr(
            benchmark,
            "_economic_interval",
            lambda _info: (_ for _ in ()).throw(RuntimeError("malformed interval")),
        )

        with pytest.raises(RuntimeError, match="complete reconciled interval bundle"):
            benchmark._service_mpc_result(
                environment=environment,
                controller=controller,
                settings=settings,
                manifest_identity=benchmark._beta_manifest_identity(environment),
                initial_vector=initial,
            )
        assert environment.decisions_taken == 1
        assert environment.physics_steps_taken == 60
    finally:
        environment.close()


def test_declared_dac_outage_is_not_misclassified_as_controller_safety_failure(
    tmp_path: Path,
) -> None:
    base = _config()
    config = replace(
        base,
        beta_task=replace(
            base.beta_task,
            disturbances=DisturbanceConfigV1(
                enabled=True,
                per_cycle_probability=1.0,
                duration_h=(1.0, 1.0),
            ),
        ),
    ).validate()
    loaded = _loaded(tmp_path, config)
    environment = benchmark._new_beta_environment(config)
    try:
        initial, _ = environment.reset(seed=42)
        controller = NormalizedSetpointPolicyControllerV1.from_environment(
            loaded, environment
        )
        controller.reset()
        run = benchmark._generic_arm_result(
            arm=benchmark.PPO_ARM,
            environment=environment,
            controller=controller,
            manifest_identity=benchmark._beta_manifest_identity(environment),
            initial_vector=initial,
        )
        simultaneous_failure = deepcopy(run["evidence"]["minute_truth_states"])
        outage_boundary = next(
            item
            for item in simultaneous_failure
            if item["plant_status"] in {"critical", "offline"}
        )
        outage_boundary["health"]["battery"]["offline"] = True
        simultaneous_passed, simultaneous_safety = (
            benchmark._hard_safety_classification(
                environment,
                run["physical_metrics"],
                simultaneous_failure,
            )
        )
    finally:
        environment.close()

    safety = run["hard_safety_metrics"]
    assert safety["observed_critical_or_offline_steps"] == 60
    assert safety["declared_dac_outage_steps"] == 60
    assert safety["unexpected_critical_or_offline_steps"] == 0
    assert safety["battery_reserve_respected"] is True
    assert run["hard_safety_passed"] is True
    assert simultaneous_passed is False
    assert simultaneous_safety["unexpected_critical_or_offline_steps"] == 1


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("decision_index", "decision index differs"),
        ("observation_sha256", "another physical observation"),
    ],
)
def test_tampered_common_command_is_rejected_before_plant_movement(
    field: str,
    message: str,
) -> None:
    config = _config()
    environment = benchmark._new_beta_environment(config)
    try:
        initial, _ = environment.reset(seed=35)
        controller = _TamperingController(field)
        controller.reset()
        run = benchmark._generic_arm_result(
            arm=benchmark.HEURISTIC_ARM,
            environment=environment,
            controller=controller,
            manifest_identity=benchmark._beta_manifest_identity(environment),
            initial_vector=initial,
        )
    finally:
        environment.close()

    assert run["execution_complete"] is False
    assert run["completed_decisions"] == run["physical_steps"] == 0
    assert message in run["failure"]["message"]


def test_tampered_e_beta_evidence_is_rejected_before_plant_movement() -> None:
    config = _config()
    environment = benchmark._new_beta_environment(config)
    try:
        initial, _ = environment.reset(seed=36)
        controller = _TamperingController("beta_observation_sha256")
        controller.reset()
        run = benchmark._generic_arm_result(
            arm=benchmark.BETA_ECONOMIC_MPC_ARM,
            environment=environment,
            controller=controller,
            manifest_identity=benchmark._beta_manifest_identity(environment),
            initial_vector=initial,
        )
    finally:
        environment.close()

    assert run["execution_complete"] is False
    assert run["completed_decisions"] == run["physical_steps"] == 0
    assert "beta_observation_sha256 differs" in run["failure"]["message"]


def test_manifest_mismatch_fails_all_four_arms_before_physical_movement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    loaded = _loaded(tmp_path, config)
    real_new_environment = benchmark._new_beta_environment
    real_identity = benchmark._beta_manifest_identity
    environments: list[Any] = []
    identity_calls = 0

    def tracked_environment(
        task: ControllerNeutralTrainingConfigV1,
    ) -> Any:
        environment = real_new_environment(task)
        environments.append(environment)
        return environment

    def mismatched_identity(environment: Any) -> dict[str, str]:
        nonlocal identity_calls
        identity_calls += 1
        identity = real_identity(environment)
        if identity_calls == 3:
            identity["market_episode_sha256"] = "0" * 64
        return identity

    def controller_must_not_be_constructed(
        _: ControllerNeutralTrainingConfigV1,
    ) -> Any:
        raise AssertionError("controller construction crossed failed preflight")

    monkeypatch.setattr(benchmark, "_new_beta_environment", tracked_environment)
    monkeypatch.setattr(benchmark, "_beta_manifest_identity", mismatched_identity)

    cell = benchmark.execute_beta_controller_benchmark_cell(
        loaded,
        config,
        seed=33,
        economic_controller_factory=controller_must_not_be_constructed,
    )

    assert len(environments) == 4
    assert len({id(environment) for environment in environments}) == 4
    assert all(
        environment.decisions_taken == 0 and environment.physics_steps_taken == 0
        for environment in environments
    )
    assert cell["status"] == "preflight_failed"
    assert cell["value_comparison_eligible"] is False
    assert set(cell["runs"]) == set(benchmark.BETA_CONTROLLER_ARM_ORDER)
    assert all(
        run["status"] == "not_run_preflight_failed"
        and run["completed_decisions"] == 0
        and run["physical_steps"] == 0
        for run in cell["runs"].values()
    )


def _synthetic_run(
    arm: str,
    *,
    value: float,
    reward: float,
    delivered: float = 24.0,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "status": "complete_feasible_safe",
        "execution_complete": True,
        "cycle_feasible": True,
        "hard_safety_passed": True,
        "reward_trace_sum_eur": reward,
        "physical_metrics": {
            "requested_kg": 24.0,
            "delivered_kg": delivered,
            "unmet_kg": 24.0 - delivered,
        },
        "economic_metrics": {"net_operating_value_eur": value},
    }


def test_value_aggregation_never_pools_an_ineligible_or_partial_cell() -> None:
    eligible_runs = {
        arm: _synthetic_run(arm, value=float(index + 1), reward=10.0 + index)
        for index, arm in enumerate(benchmark.BETA_CONTROLLER_ARM_ORDER)
    }
    excluded_runs = {
        arm: _synthetic_run(arm, value=1_000_000.0, reward=1_000_000.0)
        for arm in benchmark.BETA_CONTROLLER_ARM_ORDER
    }
    excluded_runs[benchmark.BETA_ECONOMIC_MPC_ARM].update(
        {
            "status": "execution_failed",
            "execution_complete": False,
            "cycle_feasible": False,
        }
    )
    cells = [
        {
            "seed": 1,
            "status": "value_comparison_eligible",
            "execution_complete": True,
            "cycle_feasible": True,
            "hard_safety_passed": True,
            "service_comparable": True,
            "value_comparison_eligible": True,
            "exclusion_reasons": [],
            "runs": eligible_runs,
        },
        {
            "seed": 2,
            "status": "ineligible",
            "execution_complete": False,
            "cycle_feasible": False,
            "hard_safety_passed": True,
            "service_comparable": False,
            "value_comparison_eligible": False,
            "exclusion_reasons": [
                f"arm_execution_incomplete:{benchmark.BETA_ECONOMIC_MPC_ARM}"
            ],
            "runs": excluded_runs,
        },
    ]

    aggregate = benchmark.aggregate_value_eligible_cells(cells)

    assert aggregate["registered_cell_count"] == 2
    assert aggregate["value_eligible_cell_count"] == 1
    assert aggregate["excluded_cell_count"] == 1
    assert aggregate["value_eligible_seeds"] == [1]
    assert (
        aggregate["arms"][benchmark.HEURISTIC_ARM]["net_operating_value_mean_eur"]
        == 1.0
    )
    assert (
        aggregate["arms"][benchmark.BETA_ECONOMIC_MPC_ARM][
            "net_operating_value_mean_eur"
        ]
        == 3.0
    )
    assert (
        aggregate["arms"][benchmark.BETA_ECONOMIC_MPC_ARM]["execution_complete_count"]
        == 1
    )


def test_service_comparability_is_a_separate_cell_gate() -> None:
    runs = {
        arm: _synthetic_run(arm, value=1.0, reward=1.0)
        for arm in benchmark.BETA_CONTROLLER_ARM_ORDER
    }
    runs[benchmark.PPO_ARM] = _synthetic_run(
        benchmark.PPO_ARM,
        value=1.0,
        reward=1.0,
        delivered=23.0,
    )

    classified = benchmark._cell_classification(runs)

    assert classified["execution_complete"] is True
    assert classified["cycle_feasible"] is True
    assert classified["hard_safety_passed"] is True
    assert classified["service_comparable"] is False
    assert classified["value_comparison_eligible"] is False
    assert classified["exclusion_reasons"] == ["service_not_comparable"]


def test_alpha_checkpoint_is_rejected_before_beta_evaluation(tmp_path: Path) -> None:
    beta_config = _config()
    alpha_config = replace(
        beta_config,
        environment_id=economic_spaces.ENVIRONMENT_ID,
        beta_task=None,
    ).validate()
    alpha_loaded = _loaded(tmp_path, alpha_config, name="alpha-policy")

    with pytest.raises(checkpoints.IncompatibleCheckpointError):
        benchmark.evaluate_beta_controller_benchmark(
            alpha_loaded,
            beta_config,
            seeds=[34],
            require_trained_policy=False,
        )


def test_programmatic_evaluation_cannot_relabel_checkpoint_training_config(
    tmp_path: Path,
) -> None:
    checkpoint_config = _config()
    loaded = _loaded(tmp_path, checkpoint_config)
    relabelled = replace(
        checkpoint_config,
        notes="different programmatic training claim",
    ).validate()

    with pytest.raises(
        checkpoints.IncompatibleCheckpointError,
        match="training task differs",
    ):
        benchmark.evaluate_beta_controller_benchmark(
            loaded,
            relabelled,
            seeds=[37],
            require_trained_policy=False,
        )


def test_policy_readmission_normalizes_relative_checkpoint_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    loaded = _loaded(tmp_path, config)
    monkeypatch.chdir(tmp_path)
    relative = replace(loaded, path=Path(loaded.path.name))

    readmitted, identity = benchmark._readmit_policy(
        relative,
        config,
        require_trained_policy=False,
    )

    assert readmitted.path == loaded.path.resolve()
    assert identity["path"] == str(loaded.path.resolve())
    assert readmitted.to_dict()["path"] == identity["path"]


def test_evaluation_rejects_source_change_between_start_and_finish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    loaded = _loaded(tmp_path, config)
    snapshots = iter(
        (
            {"git_revision": "1" * 40, "worktree_clean": True},
            {"git_revision": "2" * 40, "worktree_clean": True},
        )
    )
    monkeypatch.setattr(
        benchmark,
        "_source_provenance",
        lambda **_: next(snapshots),
    )
    monkeypatch.setattr(
        benchmark,
        "execute_beta_controller_benchmark_cell",
        lambda *_args, seed, **_kwargs: benchmark._preflight_failed_cell(
            seed,
            RuntimeError("bounded fixture"),
        ),
    )

    with pytest.raises(RuntimeError, match="changed during execution"):
        benchmark.evaluate_beta_controller_benchmark(
            loaded,
            config,
            seeds=[38],
            require_trained_policy=False,
        )

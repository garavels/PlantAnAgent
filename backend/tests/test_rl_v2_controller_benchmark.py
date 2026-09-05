"""Focused and adversarial checks for the unified four-arm benchmark."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
from typing import Any

import pytest

from plant_backend.rl import checkpoints
from plant_backend.rl.policies import LoadedPolicy, load_policy
from plant_backend.rl.v2 import controller_benchmark, economic_spaces
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.economic_mpc_schema import (
    EconomicMPCSolveError,
    EconomicMPCStageSolveV1,
)
from plant_backend.rl.v2.economics import EconomicControllerObservationV1
from plant_backend.rl.v2.mpc_schema import ServiceMPCSetpointSettingsV1
from plant_backend.rl.v2.policy_setpoint_controller import (
    policy_contract_for_environment,
)
from plant_backend.rl.v2.setpoint_controllers import (
    LegacyHeuristicSetpointControllerV1,
)
from plant_backend.rl.v2.training.config import ControllerNeutralTrainingConfigV1
from plant_backend.rl.v2.training.ppo import make_env
import plant_backend.rl.v2.economic_mpc_controller as economic_controller_module


def _config() -> ControllerNeutralTrainingConfigV1:
    return ControllerNeutralTrainingConfigV1(
        environment_id=economic_spaces.ENVIRONMENT_ID,
        total_timesteps=4,
        n_envs=1,
        checkpoint_every=0,
        episode=EpisodeConfigV2(
            horizon_h=0.25,
            delivery_period_h=0.25,
            delivery_rate_kg_h=1.0,
            battery_soc=(0.5, 0.5),
            h2_fill=(0.0, 0.0),
            co2_fill=(0.0, 0.0),
            ch4_coverage_h=(0.0, 0.0),
        ),
    ).validate()


def _checkpoint(tmp_path: Path, config: ControllerNeutralTrainingConfigV1) -> Path:
    environment = make_env(config)
    try:
        contract = policy_contract_for_environment(environment)
    finally:
        environment.close()
    path = checkpoints.write_fixed_checkpoint(
        tmp_path / "policy",
        [-1.0, -1.0, -1.0, -1.0],
        contract=contract,
    )
    metadata = checkpoints.load_metadata(path, check=False)
    checkpoints.save_metadata(path, replace(metadata, training_config=config.to_dict()))
    return path


def _loaded(tmp_path: Path, config: ControllerNeutralTrainingConfigV1) -> LoadedPolicy:
    environment = make_env(config)
    try:
        return load_policy(
            _checkpoint(tmp_path, config),
            contract=policy_contract_for_environment(environment),
            environment=environment,
        )
    finally:
        environment.close()


class _EconomicCommandController:
    """Test-only command source; production resolution never imports this class."""

    id = "test-economic-command-controller"

    def __init__(self, *, fail: bool = False) -> None:
        self._heuristic = LegacyHeuristicSetpointControllerV1()
        self._fail = fail

    def reset(self) -> None:
        self._heuristic.reset()

    def compute_command(
        self,
        observation: object,
        **kwargs: object,
    ) -> object:
        if self._fail:
            raise RuntimeError("economic controller failed deliberately")
        assert isinstance(observation, EconomicControllerObservationV1)
        return self._heuristic.compute_command(observation.plant, **kwargs)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "failure_policy": "raise; no fallback",
            "test_only": True,
        }


def _factory(_: ControllerNeutralTrainingConfigV1) -> _EconomicCommandController:
    return _EconomicCommandController()


def test_four_arms_share_manifest_and_retain_decision_and_minute_evidence(
    tmp_path: Path,
) -> None:
    pytest.importorskip("scipy")
    config = _config()
    loaded = _loaded(tmp_path, config)

    result = controller_benchmark.evaluate_controller_benchmark(
        loaded,
        config,
        seeds=[17],
        require_trained_policy=False,
    )

    assert result["policy_qualification"] == (
        "integration-fixture-not-a-learned-controller"
    )
    assert result["benchmark_qualification"] == (
        "unqualified-integration-fixture-or-custom-arm"
    )
    provenance = result["source_provenance"]
    assert len(provenance["git_revision"]) == 40
    assert len(provenance["dependency_lock_sha256"]) == 64
    assert isinstance(provenance["worktree_clean"], bool)
    assert provenance["runtime"]["python"]
    assert result["arm_order"] == ["H", "M24", "E", "PPO"]
    assert result["aggregate"]["complete_paired_cell_count"] == 1
    cell = result["cells"][0]
    assert cell["paired_complete"] is True
    assert all(run["status"] == "complete" for run in cell["runs"].values())
    assert {
        tuple(sorted(run["manifest_identity"].items())) for run in cell["runs"].values()
    } == {tuple(sorted(cell["manifest_identity"].items()))}
    for arm in ("H", "E", "PPO"):
        evidence = cell["runs"][arm]["evidence"]
        assert len(evidence["decisions"]) == 1
        assert len(evidence["executions"]) == 1
        assert evidence["physical_minute_count"] == 15
        assert len(evidence["executions"][0]["applications"]) == 15
    assert cell["runs"]["E"]["controller"]["id"] == ("tariff-aware-economic-mpc-v1")
    m24_run = cell["runs"]["M24"]["evidence"]["run"]
    assert m24_run["counts"]["physical_minute_count"] == 15
    assert len(m24_run["events"][0]["execution"]["applications"]) == 15
    assert result["deterministic_evidence_sha256"]


def test_cell_preflight_failure_moves_no_arm_and_is_not_hidden(tmp_path: Path) -> None:
    config = _config()
    loaded = _loaded(tmp_path, config)

    def unavailable(_: ControllerNeutralTrainingConfigV1) -> _EconomicCommandController:
        raise RuntimeError("real economic controller unavailable deliberately")

    cell = controller_benchmark.execute_controller_benchmark_cell(
        loaded,
        config,
        seed=18,
        economic_controller_factory=unavailable,
    )

    assert cell["status"] == "preflight_failed"
    assert cell["paired_complete"] is False
    assert set(cell["runs"]) == {"H", "M24", "E", "PPO"}
    assert all(
        run["status"] == "not_run_preflight_failed"
        and run["physical_steps"] == 0
        and run["completed_decisions"] == 0
        for run in cell["runs"].values()
    )
    aggregate = controller_benchmark.aggregate_complete_paired_cells([cell])
    assert aggregate["complete_paired_cell_count"] == 0
    assert aggregate["excluded_cell_count"] == 1
    assert set(aggregate["incomplete_cells"][0]["arm_failures"]) == {
        "H",
        "M24",
        "E",
        "PPO",
    }


def test_failed_arm_has_no_fallback_and_excludes_whole_pair(tmp_path: Path) -> None:
    config = _config()
    loaded = _loaded(tmp_path, config)

    cell = controller_benchmark.execute_controller_benchmark_cell(
        loaded,
        config,
        seed=19,
        economic_controller_factory=lambda _: _EconomicCommandController(fail=True),
    )

    assert cell["status"] == "arm_failed"
    assert cell["paired_complete"] is False
    assert cell["runs"]["E"]["status"] == "failed"
    assert cell["runs"]["E"]["completed_decisions"] == 0
    assert cell["runs"]["E"]["physical_steps"] == 0
    assert cell["runs"]["E"]["failure"]["exception_type"] == "RuntimeError"
    assert all(cell["runs"][arm]["status"] == "complete" for arm in ("H", "M24", "PPO"))
    aggregate = controller_benchmark.aggregate_complete_paired_cells([cell])
    assert aggregate["arms"]["H"]["paired_cell_count"] == 0
    assert aggregate["arms"]["E"]["reward_mean_eur"] is None
    assert aggregate["incomplete_cells"][0]["arm_status"]["E"] == "failed"


def test_typed_economic_solver_failure_retains_stage_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("scipy")
    config = _config()
    loaded = _loaded(tmp_path, config)
    failed_stage = EconomicMPCStageSolveV1(
        stage="service_maximization",
        status=2,
        message="deliberate infeasible model",
        elapsed_ns=123,
        objective_value=None,
        mip_gap=None,
        mip_node_count=None,
        solution_sha256=None,
        scipy_version="test-scipy",
        highs_version="test-highs",
        success=False,
    )

    def fail_solve(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise EconomicMPCSolveError(
            "deliberate typed solve failure",
            stages=(failed_stage,),
        )

    monkeypatch.setattr(
        economic_controller_module,
        "solve_economic_mpc_model",
        fail_solve,
    )
    cell = controller_benchmark.execute_controller_benchmark_cell(
        loaded,
        config,
        seed=23,
    )

    failure = cell["runs"]["E"]["failure"]
    assert failure["exception_type"] == "EconomicMPCSolveError"
    assert failure["solver_stages"] == [failed_stage.to_dict()]
    assert cell["paired_complete"] is False


def test_real_economic_arm_rolls_across_multiple_decisions(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    config = replace(
        _config(),
        episode=EpisodeConfigV2(
            horizon_h=1.0,
            delivery_period_h=1.0,
            delivery_rate_kg_h=1.0,
            battery_soc=(0.5, 0.5),
            h2_fill=(0.0, 0.0),
            co2_fill=(0.0, 0.0),
            ch4_coverage_h=(0.0, 0.0),
        ),
    ).validate()
    loaded = _loaded(tmp_path, config)

    result = controller_benchmark.evaluate_controller_benchmark(
        loaded,
        config,
        seeds=[24],
        require_trained_policy=False,
    )

    cell = result["cells"][0]
    economic = cell["runs"]["E"]
    assert cell["paired_complete"] is True
    assert economic["completed_decisions"] == 4
    assert economic["evidence"]["decision_count"] == 4
    assert economic["evidence"]["physical_minute_count"] == 60
    assert [
        decision["evidence"]["decision_index"]
        for decision in economic["evidence"]["decisions"]
    ] == [0, 1, 2, 3]


def test_deterministic_digest_excludes_all_wall_clock_measurements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    loaded = _loaded(tmp_path, config)
    ticks = iter(range(1, 1_000_000))
    monkeypatch.setattr(
        controller_benchmark.time, "perf_counter_ns", lambda: next(ticks)
    )
    first = controller_benchmark.evaluate_controller_benchmark(
        loaded,
        config,
        seeds=[20],
        economic_controller_factory=_factory,
        require_trained_policy=False,
    )
    ticks = iter(range(8_000_000, 9_000_000, 3))
    monkeypatch.setattr(
        controller_benchmark.time, "perf_counter_ns", lambda: next(ticks)
    )
    second = controller_benchmark.evaluate_controller_benchmark(
        loaded,
        config,
        seeds=[20],
        economic_controller_factory=_factory,
        require_trained_policy=False,
    )

    assert first["cells"][0]["runs"]["PPO"]["controller_elapsed_ns"] != (
        second["cells"][0]["runs"]["PPO"]["controller_elapsed_ns"]
    )
    assert first["deterministic_evidence_sha256"] == (
        second["deterministic_evidence_sha256"]
    )


def test_deterministic_digest_is_independent_of_checkpoint_location(
    tmp_path: Path,
) -> None:
    config = _config()
    first_loaded = _loaded(tmp_path / "first", config)
    relocated_path = tmp_path / "relocated" / "policy"
    relocated_path.parent.mkdir(parents=True)
    shutil.copytree(first_loaded.path, relocated_path)
    environment = make_env(config)
    try:
        relocated = load_policy(
            relocated_path,
            contract=policy_contract_for_environment(environment),
            environment=environment,
        )
    finally:
        environment.close()

    first = controller_benchmark.evaluate_controller_benchmark(
        first_loaded,
        config,
        seeds=[21],
        economic_controller_factory=_factory,
        require_trained_policy=False,
    )
    second = controller_benchmark.evaluate_controller_benchmark(
        relocated,
        config,
        seeds=[21],
        economic_controller_factory=_factory,
        require_trained_policy=False,
    )

    assert first["checkpoint"]["path"] != second["checkpoint"]["path"]
    assert first["deterministic_evidence_sha256"] == (
        second["deterministic_evidence_sha256"]
    )


def test_production_admission_rejects_fixed_policy_and_cli_preserves_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config()
    loaded = _loaded(tmp_path, config)
    with pytest.raises(
        checkpoints.IncompatibleCheckpointError,
        match="trained SB3 PPO",
    ):
        controller_benchmark.evaluate_controller_benchmark(
            loaded,
            config,
            seeds=[0],
        )

    with pytest.raises(ValueError, match="cannot replace"):
        controller_benchmark.evaluate_controller_benchmark(
            loaded,
            config,
            seeds=[0],
            economic_controller_factory=_factory,
        )
    custom_service = ServiceMPCSetpointSettingsV1(
        task_horizon_h=config.episode.horizon_h,
        planning_horizon_h=config.episode.horizon_h,
        interval_s=config.episode.decision_interval_s,
        cold_start_weight=0.2,
    )
    with pytest.raises(ValueError, match="canonical rolling M24"):
        controller_benchmark.evaluate_controller_benchmark(
            loaded,
            config,
            seeds=[0],
            service_settings=custom_service,
        )

    output = tmp_path / "owned.json"
    output.write_text(json.dumps({"owner": "operator"}) + "\n", encoding="utf-8")
    assert (
        controller_benchmark.main(
            [
                "--checkpoint",
                str(loaded.path),
                "--json",
                str(output),
            ]
        )
        == 2
    )
    assert json.loads(output.read_text(encoding="utf-8")) == {"owner": "operator"}
    assert "output already exists" in capsys.readouterr().err
    assert (
        controller_benchmark.main(
            [
                "--checkpoint",
                str(loaded.path),
                "--json",
                "relative-report.json",
            ]
        )
        == 2
    )
    assert "absolute new path" in capsys.readouterr().err


def test_seed_validation_is_bounded_and_unique(tmp_path: Path) -> None:
    config = _config()
    loaded = _loaded(tmp_path, config)
    for seeds, message in (([], "at least one"), ([1, 1], "unique")):
        with pytest.raises(ValueError, match=message):
            controller_benchmark.evaluate_controller_benchmark(
                loaded,
                config,
                seeds=seeds,
                economic_controller_factory=_factory,
                require_trained_policy=False,
            )
    with pytest.raises(ValueError, match="at most"):
        controller_benchmark.evaluate_controller_benchmark(
            loaded,
            config,
            seeds=list(
                range(controller_benchmark.CONTROLLER_NEUTRAL_BENCHMARK_MAX_SEEDS + 1)
            ),
            economic_controller_factory=_factory,
            require_trained_policy=False,
        )

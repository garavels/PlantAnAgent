"""Adversarial admission checks for detached beta benchmark evidence."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from plant_backend.rl import checkpoints
from plant_backend.rl.policies import LoadedPolicy, load_policy
from plant_backend.rl.v2 import beta_spaces
from plant_backend.rl.v2 import beta_controller_benchmark as benchmark
from plant_backend.rl.v2.beta_config import CyclicMarketTaskConfigV1, canonical_sha256
from plant_backend.rl.v2.beta_controller_benchmark_verifier import (
    _decode_beta_observation,
    verify_beta_controller_benchmark,
    verify_beta_controller_benchmark_file,
)
from plant_backend.rl.v2.beta_disturbances import generate_disturbance_episode
from plant_backend.rl.v2.beta_env import BETA_ROOT_STREAM_NAMES
from plant_backend.rl.v2.beta_mpc_schema import (
    BETA_MPC_CYCLE_RETURN_DOMAIN,
    BETA_MPC_DECISION_DOMAIN,
    BETA_MPC_PLAN_DOMAIN,
    BETA_MPC_PRICE_PROJECTION_DOMAIN,
    BETA_MPC_SOLUTION_DOMAIN,
    project_beta_mpc_prices,
)
from plant_backend.rl.v2.beta_mpc_model import build_beta_mpc_model
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.economic_mpc_model import economic_delivery_lock_row
from plant_backend.rl.v2.economics import ch4_equivalent_inventory_kg
from plant_backend.rl.v2.mpc_forecast import project_issued_mean_solar
from plant_backend.rl.v2.mpc_schema import domain_sha256
from plant_backend.rl.v2.policy_setpoint_controller import (
    NormalizedSetpointPolicyControllerV1,
    policy_contract_for_environment,
)
from plant_backend.rl.v2.setpoint_command import (
    PROCESS_SETPOINT_BOUNDARY_DOMAIN,
    PROCESS_SETPOINT_COMMAND_DOMAIN,
    PROCESS_SETPOINT_MINUTE_DISPATCH_DOMAIN,
)
from plant_backend.rl.v2.setpoint_execution import (
    PROCESS_SETPOINT_INTERVAL_EXECUTION_DOMAIN,
    PROCESS_SETPOINT_MINUTE_APPLICATION_DOMAIN,
)
from plant_backend.rl.v2.training.config import ControllerNeutralTrainingConfigV1
from plant_backend.rl.v2.training.ppo import make_env


_REPORT_DOMAIN = "cyclic-market-controller-benchmark-evidence-v1"


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
        notes="detached beta benchmark verifier fixture; not trained",
    ).validate()


def _loaded(
    directory: Path,
    config: ControllerNeutralTrainingConfigV1,
) -> LoadedPolicy:
    environment = make_env(config)
    try:
        contract = policy_contract_for_environment(environment)
        path = checkpoints.write_fixed_checkpoint(
            directory / "beta-verifier-policy",
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


@pytest.fixture(scope="module")
def benchmark_inputs(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[LoadedPolicy, ControllerNeutralTrainingConfigV1]:
    directory = tmp_path_factory.mktemp("beta-benchmark-verifier")
    config = _config()
    return _loaded(directory, config), config


@pytest.fixture(scope="module")
def real_report(
    benchmark_inputs: tuple[LoadedPolicy, ControllerNeutralTrainingConfigV1],
) -> dict[str, Any]:
    loaded, config = benchmark_inputs
    return benchmark.evaluate_beta_controller_benchmark(
        loaded,
        config,
        seeds=[1703],
        require_trained_policy=False,
    )


class _FailingCanonicalBetaController:
    def __init__(self, config: ControllerNeutralTrainingConfigV1) -> None:
        self._controller = benchmark._real_beta_economic_controller(config)

    def reset(self) -> None:
        self._controller.reset()

    def compute_command(self, *_args: object, **_kwargs: object) -> object:
        raise RuntimeError("deliberate pre-movement E-beta failure")

    def to_dict(self) -> dict[str, object]:
        return self._controller.to_dict()


@pytest.fixture(scope="module")
def failed_arm_report(
    benchmark_inputs: tuple[LoadedPolicy, ControllerNeutralTrainingConfigV1],
) -> dict[str, Any]:
    loaded, config = benchmark_inputs
    return benchmark.evaluate_beta_controller_benchmark(
        loaded,
        config,
        seeds=[1704],
        economic_controller_factory=_FailingCanonicalBetaController,
        require_trained_policy=False,
    )


@pytest.mark.parametrize(
    "defect",
    [
        "fallback",
        "missing_fallback",
        "label",
        "description",
        "uses_forecast",
        "information_set",
        "unknown_field",
        "loaded_unknown_field",
    ],
)
def test_policy_descriptor_cannot_relabel_fallback_or_information_contract(
    benchmark_inputs: tuple[LoadedPolicy, ControllerNeutralTrainingConfigV1],
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta
    from plant_backend.rl.v2.beta_controller_benchmark_verifier import (
        _verify_policy_controller,
    )

    def forbidden_reset(*args: object, **kwargs: object) -> None:
        pytest.fail("descriptor admission must not reset an environment")

    monkeypatch.setattr(PlantCyclicMarketEnvV2Beta, "reset", forbidden_reset)
    loaded, config = benchmark_inputs
    environment = make_env(config)
    try:
        descriptor = NormalizedSetpointPolicyControllerV1.from_environment(
            loaded, environment
        ).to_dict()
    finally:
        environment.close()
    checkpoint = benchmark._checkpoint_identity(loaded)
    _verify_policy_controller(descriptor, config=config, checkpoint=checkpoint)
    modified = deepcopy(descriptor)
    if defect == "fallback":
        modified["failure_policy"] = "switch to heuristic on failure"
    elif defect == "missing_fallback":
        del modified["failure_policy"]
    elif defect == "unknown_field":
        modified["fallback_enabled"] = True
    elif defect == "loaded_unknown_field":
        modified["checkpoint"]["fallback_enabled"] = True
    elif defect == "uses_forecast":
        modified[defect] = False
    else:
        modified[defect] = "coherently relabelled descriptor"
    # Descriptor validation is independent of the enclosing report hashes:
    # recomputing those hashes cannot make these declarations admissible.
    with pytest.raises(ValueError):
        _verify_policy_controller(modified, config=config, checkpoint=checkpoint)


def _report_body(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": report["format"],
        "claim_boundary": report["claim_boundary"],
        "policy_qualification": report["policy_qualification"],
        "benchmark_qualification": report["benchmark_qualification"],
        "training_task": report["training_task"],
        "checkpoint": benchmark._timing_free(report["checkpoint"]),
        "source_provenance": report["source_provenance"],
        "seeds": report["seeds"],
        "arm_order": report["arm_order"],
        "information_profiles": report["information_profiles"],
        "execution_doors": report["execution_doors"],
        "service_mpc_settings": report["service_mpc_settings"],
        "cell_evidence": [
            cell["deterministic_evidence_sha256"] for cell in report["cells"]
        ],
        "aggregate": report["aggregate"],
    }


def _rehash_report(report: dict[str, Any], *, aggregate: bool = True) -> None:
    for cell in report["cells"]:
        for run in cell["runs"].values():
            benchmark._attach_arm_digest(run)
        benchmark._attach_cell_digest(cell)
    if aggregate:
        report["aggregate"] = benchmark.aggregate_value_eligible_cells(report["cells"])
    report["deterministic_evidence_sha256"] = domain_sha256(
        _REPORT_DOMAIN,
        _report_body(report),
    )


def _rehash_disturbance_manifest(report: dict[str, Any]) -> None:
    cell = report["cells"][0]
    manifest = cell["audit_manifest"]
    disturbances = manifest["disturbances"]
    episode_identity = {
        "disturbance_generator_version": disturbances["disturbance_generator_version"],
        "config_sha256": disturbances["config_sha256"],
        "schedule_sha256": disturbances["schedule_sha256"],
        "cycle_h": disturbances["cycle_h"],
        "cycle_count": disturbances["cycle_count"],
        "physics_dt_s": disturbances["physics_dt_s"],
        "seed_stream": disturbances["random_streams"]["root"],
        "cycle_streams": disturbances["random_streams"]["cycles"],
    }
    episode_sha = canonical_sha256(episode_identity)
    disturbances["episode_sha256"] = episode_sha
    disturbances["private_hashes"]["episode_sha256"] = episode_sha
    manifest["beta_exogenous"]["disturbance_episode_sha256"] = episode_sha
    manifest["beta_exogenous_episode_sha256"] = canonical_sha256(
        manifest["beta_exogenous"]
    )
    immutable = {
        key: value
        for key, value in manifest.items()
        if key
        not in {
            "manifest_sha256",
            "alpha_physical_exogenous_episode_sha256",
            "beta_exogenous_episode_sha256",
            "delivery_accounting",
            "economic_accounting",
            "cycle_accounting",
        }
    }
    manifest["manifest_sha256"] = canonical_sha256(immutable)
    identity_updates = {
        "manifest_sha256": manifest["manifest_sha256"],
        "beta_exogenous_episode_sha256": manifest["beta_exogenous_episode_sha256"],
        "disturbance_episode_sha256": episode_sha,
        "disturbance_schedule_sha256": disturbances["schedule_sha256"],
    }
    cell["manifest_identity"].update(identity_updates)
    for run in cell["runs"].values():
        run["manifest_identity"].update(identity_updates)


def _first_beta_decision(report: dict[str, Any]) -> dict[str, Any]:
    return report["cells"][0]["runs"][benchmark.BETA_ECONOMIC_MPC_ARM]["evidence"][
        "decisions"
    ][0]


def _rehash_beta_decision(decision: dict[str, Any]) -> None:
    evidence = decision["evidence"]
    plan = evidence["plan"]
    plan_body = deepcopy(plan)
    del plan_body["sha256"]
    plan_body["stages"] = [
        {"stage": stage["stage"], "solution_sha256": stage["solution_sha256"]}
        for stage in plan["stages"]
    ]
    plan["sha256"] = domain_sha256(BETA_MPC_PLAN_DOMAIN, plan_body)
    evidence["plan_sha256"] = plan["sha256"]
    evidence_body = deepcopy(evidence)
    del evidence_body["sha256"]
    evidence_body["plan"] = {"sha256": evidence["plan_sha256"]}
    evidence["sha256"] = domain_sha256(BETA_MPC_DECISION_DOMAIN, evidence_body)
    command = decision["command"]
    command["issuer"]["source_sha256"] = evidence["sha256"]
    command_body = deepcopy(command)
    del command_body["sha256"]
    command["sha256"] = domain_sha256(PROCESS_SETPOINT_COMMAND_DOMAIN, command_body)


def _rehash_sha256(payload: dict[str, Any], *, domain: str) -> None:
    body = deepcopy(payload)
    del body["sha256"]
    payload["sha256"] = domain_sha256(domain, body)


def _propagate_first_beta_command(report: dict[str, Any]) -> None:
    decision = _first_beta_decision(report)
    command = decision["command"]
    execution = report["cells"][0]["runs"][benchmark.BETA_ECONOMIC_MPC_ARM]["evidence"][
        "executions"
    ][0]
    execution["command"] = deepcopy(command)
    for application in execution["applications"]:
        dispatch = application["dispatch"]
        dispatch["command_sha256"] = command["sha256"]
        _rehash_sha256(dispatch, domain=PROCESS_SETPOINT_MINUTE_DISPATCH_DOMAIN)
        _rehash_sha256(
            application,
            domain=PROCESS_SETPOINT_MINUTE_APPLICATION_DOMAIN,
        )
    _rehash_sha256(execution, domain=PROCESS_SETPOINT_INTERVAL_EXECUTION_DOMAIN)


def _selected_solution_digest(report: dict[str, Any], vector: list[float]) -> str:
    config = ControllerNeutralTrainingConfigV1.from_dict(report["training_task"])
    assert config.beta_task is not None
    run = report["cells"][0]["runs"][benchmark.BETA_ECONOMIC_MPC_ARM]
    boundary_payload = run["evidence"]["boundary_ledger"]["decision_boundaries"][0]
    boundary = _decode_beta_observation(
        boundary_payload["typed_observation"],
        name="test E-beta opening observation",
    )
    controller = benchmark._real_beta_economic_controller(config)
    settings = controller.settings
    forecast = boundary.plant.forecast
    assert forecast is not None
    weather_projection = project_issued_mean_solar(
        forecast,
        boundary_time_s=boundary.plant.state.boundary.time_s,
        config=config.plant,
        interval_s=settings.interval_s,
        energy_tolerance=settings.energy_reconstruction_tolerance,
    )
    price_projection = project_beta_mpc_prices(
        boundary.market,
        interval_s=settings.interval_s,
        horizon_intervals=(
            boundary.cycle.cycle_decisions - boundary.cycle.decision_in_cycle
        ),
    )
    initial_state = boundary.plant.state.boundary
    model = build_beta_mpc_model(
        boundary,
        weather_projection,
        price_projection,
        settings=settings,
        assumptions=config.economic_assumptions,
        beta_task=config.beta_task,
        constraints=config.constraints,
        config=config.plant,
        commitment_profile=config.episode.electrolyser_commitment,
        initial_ch4_equivalent_kg=ch4_equivalent_inventory_kg(
            initial_state, config.plant
        ),
        initial_battery_energy_kwh=initial_state.battery_energy_kwh,
    )
    plan = _first_beta_decision(report)["evidence"]["plan"]
    lock = economic_delivery_lock_row(
        model,
        delivered_optimum_kg=plan["primary_delivered_optimum_kg"],
        tolerance_kg=plan["delivery_lock_tolerance_kg"],
    )
    return domain_sha256(
        BETA_MPC_SOLUTION_DOMAIN,
        {
            "model_sha256": model.sha256,
            "stage": "economic_cost_minimization",
            "extra_rows": [lock.to_dict()],
            "values": vector,
        },
    )


def _rehash_internal_minute_boundary(
    execution: dict[str, Any],
    *,
    left_index: int,
    record: dict[str, Any],
) -> None:
    applications = execution["applications"]
    left = applications[left_index]
    right = applications[left_index + 1]
    boundary_sha256 = domain_sha256(PROCESS_SETPOINT_BOUNDARY_DOMAIN, record)
    left["next_boundary"]["sha256"] = boundary_sha256
    right["dispatch"]["boundary"]["sha256"] = boundary_sha256
    _rehash_sha256(
        right["dispatch"],
        domain=PROCESS_SETPOINT_MINUTE_DISPATCH_DOMAIN,
    )
    _rehash_sha256(left, domain=PROCESS_SETPOINT_MINUTE_APPLICATION_DOMAIN)
    _rehash_sha256(right, domain=PROCESS_SETPOINT_MINUTE_APPLICATION_DOMAIN)
    _rehash_sha256(execution, domain=PROCESS_SETPOINT_INTERVAL_EXECUTION_DOMAIN)


def _write_canonical(path: Path, report: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_verifies_real_unqualified_runner_report(real_report: dict[str, Any]) -> None:
    result = verify_beta_controller_benchmark(deepcopy(real_report))

    assert result["verification"] == "passed"
    assert (
        result["deterministic_evidence_sha256"]
        == real_report["deterministic_evidence_sha256"]
    )
    assert result["cell_count"] == 1


def test_verifies_retained_pre_movement_failure_boundary(
    failed_arm_report: dict[str, Any],
) -> None:
    run = failed_arm_report["cells"][0]["runs"][benchmark.BETA_ECONOMIC_MPC_ARM]
    ledger = run["evidence"]["boundary_ledger"]

    assert run["status"] == "execution_failed"
    assert run["completed_decisions"] == run["physical_steps"] == 0
    assert len(ledger["decision_boundaries"]) == 1
    assert (
        ledger["final_boundary"]["normalized_observation"]
        == ledger["decision_boundaries"][0]["normalized_observation"]
    )
    assert (
        verify_beta_controller_benchmark(deepcopy(failed_arm_report))["verification"]
        == "passed"
    )


def test_verifies_canonical_runner_file(
    real_report: dict[str, Any], tmp_path: Path
) -> None:
    path = tmp_path / "benchmark.json"
    _write_canonical(path, real_report)

    result = verify_beta_controller_benchmark_file(path)

    assert result["verification"] == "passed"
    assert result["path"] == str(path.resolve())
    assert len(result["raw_file_sha256"]) == 64


def test_timing_only_changes_preserve_semantic_admission(
    real_report: dict[str, Any],
) -> None:
    altered = deepcopy(real_report)
    beta_run = altered["cells"][0]["runs"][benchmark.BETA_ECONOMIC_MPC_ARM]
    beta_run["controller_elapsed_ns"][0] += 123_456
    beta_run["dispatcher_elapsed_ns"][0][0] += 654_321
    _first_beta_decision(altered)["evidence"]["plan"]["stages"][0][
        "elapsed_ns"
    ] += 999_999

    result = verify_beta_controller_benchmark(altered)

    assert (
        result["deterministic_evidence_sha256"]
        == real_report["deterministic_evidence_sha256"]
    )


def test_rejects_top_level_digest_tampering(real_report: dict[str, Any]) -> None:
    altered = deepcopy(real_report)
    altered["deterministic_evidence_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="deterministic evidence digest"):
        verify_beta_controller_benchmark(altered)


def test_rejects_coherently_rehashed_typed_boundary_tampering(
    real_report: dict[str, Any],
) -> None:
    altered = deepcopy(real_report)
    boundary = altered["cells"][0]["runs"][benchmark.HEURISTIC_ARM]["evidence"][
        "boundary_ledger"
    ]["decision_boundaries"][0]
    boundary["typed_observation"]["plant"]["state"]["boundary"][
        "solar_available_kw"
    ] += 1.0
    _rehash_report(altered)

    with pytest.raises(ValueError):
        verify_beta_controller_benchmark(altered)


def test_rejects_coherently_rehashed_normalization_tampering(
    real_report: dict[str, Any],
) -> None:
    altered = deepcopy(real_report)
    boundary = altered["cells"][0]["runs"][benchmark.PPO_ARM]["evidence"][
        "boundary_ledger"
    ]["decision_boundaries"][0]
    boundary["normalized_observation"][0] += 0.125
    _rehash_report(altered)

    with pytest.raises(ValueError, match="normalized"):
        verify_beta_controller_benchmark(altered)


def test_rejects_coherently_rehashed_cycle_observation_tampering(
    real_report: dict[str, Any],
) -> None:
    altered = deepcopy(real_report)
    boundary = altered["cells"][0]["runs"][benchmark.BETA_ECONOMIC_MPC_ARM]["evidence"][
        "boundary_ledger"
    ]["decision_boundaries"][0]
    boundary["typed_observation"]["cycle"]["battery_margin_kwh"] += 1.0
    _rehash_report(altered)

    with pytest.raises(ValueError):
        verify_beta_controller_benchmark(altered)


def test_rejects_coherently_rehashed_source_provenance_tampering(
    real_report: dict[str, Any],
) -> None:
    altered = deepcopy(real_report)
    altered["source_provenance"]["worktree_clean"] = True
    altered["source_provenance"]["worktree_status_sha256"] = "0" * 64
    _rehash_report(altered)

    with pytest.raises(ValueError, match="clean source provenance"):
        verify_beta_controller_benchmark(altered)


def test_rejects_coherently_rehashed_different_seed_disturbance_manifest(
    real_report: dict[str, Any],
) -> None:
    altered = deepcopy(real_report)
    config = ControllerNeutralTrainingConfigV1.from_dict(altered["training_task"])
    assert config.beta_task is not None
    manifest = altered["cells"][0]["audit_manifest"]
    original_schedule = manifest["disturbances"]["schedule"]
    cycle_decisions = config.beta_task.cycle.cycle_decisions(config.episode)
    cycle_count = config.episode.decision_steps(config.plant) // cycle_decisions
    alternate_manifest: dict[str, object] | None = None
    for alternate_seed in range(altered["seeds"][0] + 1, altered["seeds"][0] + 101):
        root_streams = np.random.SeedSequence(alternate_seed).spawn(
            len(BETA_ROOT_STREAM_NAMES)
        )
        candidate = generate_disturbance_episode(
            config.beta_task.disturbances,
            seed=root_streams[4],
            cycle_h=config.beta_task.cycle.cycle_h,
            cycle_count=cycle_count,
            physics_dt_s=config.plant.dt_s,
        ).private_audit_manifest()
        if candidate["schedule"] != original_schedule:
            alternate_manifest = candidate
            break
    assert alternate_manifest is not None
    manifest["disturbances"] = alternate_manifest
    _rehash_disturbance_manifest(altered)
    _rehash_report(altered)

    with pytest.raises(ValueError, match="regenerated disturbance episode"):
        verify_beta_controller_benchmark(altered)


@pytest.mark.parametrize("mutation", ["unknown", "missing", "reordered"])
def test_rejects_coherently_rehashed_arm_order_changes(
    real_report: dict[str, Any], mutation: str
) -> None:
    altered = deepcopy(real_report)
    if mutation == "unknown":
        altered["arm_order"].append("invented-arm")
    elif mutation == "missing":
        altered["arm_order"].pop()
    else:
        altered["arm_order"].reverse()
    _rehash_report(altered)

    with pytest.raises(ValueError, match="arm order"):
        verify_beta_controller_benchmark(altered)


def test_rejects_coherently_rehashed_eligibility_and_aggregate_tampering(
    real_report: dict[str, Any],
) -> None:
    altered = deepcopy(real_report)
    cell = altered["cells"][0]
    cell["value_comparison_eligible"] = not cell["value_comparison_eligible"]
    for run in cell["runs"].values():
        run["value_comparison_eligible"] = cell["value_comparison_eligible"]
    _rehash_report(altered)

    with pytest.raises(ValueError):
        verify_beta_controller_benchmark(altered)


def test_rejects_rehashed_report_owned_aggregate_tampering(
    real_report: dict[str, Any],
) -> None:
    altered = deepcopy(real_report)
    altered["aggregate"]["registered_cell_count"] += 1
    _rehash_report(altered, aggregate=False)

    with pytest.raises(ValueError, match="aggregate"):
        verify_beta_controller_benchmark(altered)


def test_rejects_coherently_rehashed_physical_metric_tampering(
    real_report: dict[str, Any],
) -> None:
    altered = deepcopy(real_report)
    metrics = altered["cells"][0]["runs"][benchmark.HEURISTIC_ARM]["physical_metrics"]
    metrics["delivered_kg"] += 1.0
    _rehash_report(altered)

    with pytest.raises(ValueError):
        verify_beta_controller_benchmark(altered)


@pytest.mark.parametrize(
    "field",
    ["declared_dac_outage_steps", "minimum_allowed_battery_soc"],
)
def test_rejects_rehashed_fault_adjusted_safety_tampering(
    real_report: dict[str, Any], field: str
) -> None:
    altered = deepcopy(real_report)
    safety = altered["cells"][0]["runs"][benchmark.HEURISTIC_ARM]["hard_safety_metrics"]
    safety[field] += 1
    _rehash_report(altered)

    with pytest.raises(ValueError):
        verify_beta_controller_benchmark(altered)


@pytest.mark.parametrize("field", ["min_battery_soc", "shutdown_steps"])
def test_rejects_coherently_rehashed_physical_safety_summary_tampering(
    real_report: dict[str, Any], field: str
) -> None:
    altered = deepcopy(real_report)
    run = altered["cells"][0]["runs"][benchmark.HEURISTIC_ARM]
    physical = run["physical_metrics"]
    safety = run["hard_safety_metrics"]
    if field == "min_battery_soc":
        physical[field] += 0.1
        safety["minimum_battery_soc"] = physical[field]
        safety["battery_reserve_respected"] = bool(
            physical[field] > safety["minimum_allowed_battery_soc"] + 1e-9
        )
    else:
        physical[field] += 1
        safety["observed_critical_or_offline_steps"] += 1
        safety["unexpected_critical_or_offline_steps"] += 1
    _rehash_report(altered)

    with pytest.raises(ValueError):
        verify_beta_controller_benchmark(altered)


def test_rejects_fully_rehashed_internal_minute_soc_tampering(
    real_report: dict[str, Any],
) -> None:
    altered = deepcopy(real_report)
    config = ControllerNeutralTrainingConfigV1.from_dict(altered["training_task"])
    run = altered["cells"][0]["runs"][benchmark.HEURISTIC_ARM]
    execution = run["evidence"]["executions"][0]
    record = run["evidence"]["minute_truth_states"][1]
    replacement_soc = max(
        config.plant.battery.min_soc + 0.05,
        record["battery_soc"] - 0.05,
    )
    assert replacement_soc != record["battery_soc"]
    record["battery_soc"] = replacement_soc
    run["physical_metrics"]["min_battery_soc"] = min(
        run["physical_metrics"]["min_battery_soc"], replacement_soc
    )
    run["hard_safety_metrics"]["minimum_battery_soc"] = run["physical_metrics"][
        "min_battery_soc"
    ]
    _rehash_internal_minute_boundary(
        execution,
        left_index=0,
        record=record,
    )
    _rehash_report(altered)

    with pytest.raises(ValueError):
        verify_beta_controller_benchmark(altered)


def test_rejects_fully_rehashed_internal_minute_status_tampering(
    real_report: dict[str, Any],
) -> None:
    altered = deepcopy(real_report)
    run = altered["cells"][0]["runs"][benchmark.HEURISTIC_ARM]
    chosen: tuple[dict[str, Any], int, dict[str, Any]] | None = None
    for execution in run["evidence"]["executions"]:
        for left_index, application in enumerate(execution["applications"][:-1]):
            next_step = application["next_boundary"]["step"]
            record = run["evidence"]["minute_truth_states"][next_step]
            if record["plant_status"] in {"critical", "offline"}:
                chosen = execution, left_index, record
                break
        if chosen is not None:
            break
    assert chosen is not None
    execution, left_index, record = chosen
    record["plant_status"] = (
        "critical" if record["plant_status"] == "offline" else "offline"
    )
    _rehash_internal_minute_boundary(
        execution,
        left_index=left_index,
        record=record,
    )
    _rehash_report(altered)

    with pytest.raises(ValueError):
        verify_beta_controller_benchmark(altered)


@pytest.mark.parametrize("field", ["price", "cycle", "model", "forecast"])
def test_rejects_rehashed_beta_plan_identity_tampering(
    real_report: dict[str, Any], field: str
) -> None:
    altered = deepcopy(real_report)
    decision = _first_beta_decision(altered)
    evidence = decision["evidence"]
    plan = evidence["plan"]
    if field == "price":
        projection = plan["price_projection"]
        projection["prices_eur_per_mwh"][0] += 1.0
        projection_body = deepcopy(projection)
        del projection_body["sha256"]
        projection["sha256"] = domain_sha256(
            BETA_MPC_PRICE_PROJECTION_DOMAIN, projection_body
        )
        evidence["price_projection_sha256"] = projection["sha256"]
    elif field == "cycle":
        cycle_return = plan["cycle_return"]
        battery = cycle_return["resources"]["battery"]
        battery["current"] += 1.0
        battery["signed_margin"] += 1.0
        cycle_body = deepcopy(cycle_return)
        del cycle_body["sha256"]
        cycle_return["sha256"] = domain_sha256(BETA_MPC_CYCLE_RETURN_DOMAIN, cycle_body)
        evidence["cycle_return_sha256"] = cycle_return["sha256"]
    elif field == "model":
        plan["model_sha256"] = "1" * 64
        evidence["model_sha256"] = plan["model_sha256"]
    else:
        plan["weather_projection_sha256"] = "2" * 64
        plan["forecast_sha256"] = plan["weather_projection_sha256"]
        evidence["weather_projection_sha256"] = plan["weather_projection_sha256"]
    _rehash_beta_decision(decision)
    _rehash_report(altered)

    with pytest.raises(ValueError):
        verify_beta_controller_benchmark(altered)


def test_rejects_rehashed_selected_solution_value_tampering(
    real_report: dict[str, Any],
) -> None:
    altered = deepcopy(real_report)
    decision = _first_beta_decision(altered)
    decision["evidence"]["plan"]["selected_solution_vector"][0] += 1.0
    _rehash_beta_decision(decision)
    _propagate_first_beta_command(altered)
    _rehash_report(altered)

    with pytest.raises(ValueError, match="selected solution"):
        verify_beta_controller_benchmark(altered)


def test_rejects_rehashed_selected_solution_cardinality_tampering(
    real_report: dict[str, Any],
) -> None:
    altered = deepcopy(real_report)
    decision = _first_beta_decision(altered)
    decision["evidence"]["plan"]["selected_solution_vector"].pop()
    _rehash_beta_decision(decision)
    _propagate_first_beta_command(altered)
    _rehash_report(altered)

    with pytest.raises(ValueError, match="cardinality"):
        verify_beta_controller_benchmark(altered)


def test_rejects_nonfinite_selected_solution_value(
    real_report: dict[str, Any],
) -> None:
    altered = deepcopy(real_report)
    _first_beta_decision(altered)["evidence"]["plan"]["selected_solution_vector"][0] = (
        float("nan")
    )

    with pytest.raises(ValueError, match="non-finite"):
        verify_beta_controller_benchmark(altered)


def test_rejects_coherently_rehashed_stage_two_solution_digest_tampering(
    real_report: dict[str, Any],
) -> None:
    altered = deepcopy(real_report)
    decision = _first_beta_decision(altered)
    decision["evidence"]["plan"]["stages"][1]["solution_sha256"] = "f" * 64
    _rehash_beta_decision(decision)
    _propagate_first_beta_command(altered)
    _rehash_report(altered)

    with pytest.raises(ValueError, match="solution digest"):
        verify_beta_controller_benchmark(altered)


def test_rejects_coherently_rehashed_solution_vector_and_digest_tampering(
    real_report: dict[str, Any],
) -> None:
    altered = deepcopy(real_report)
    decision = _first_beta_decision(altered)
    plan = decision["evidence"]["plan"]
    vector = plan["selected_solution_vector"]
    vector[0] += 1_000_000_000.0
    plan["stages"][1]["solution_sha256"] = _selected_solution_digest(altered, vector)
    _rehash_beta_decision(decision)
    _propagate_first_beta_command(altered)
    _rehash_report(altered)

    with pytest.raises(
        ValueError, match="(model audit|selected plan|selected solution)"
    ):
        verify_beta_controller_benchmark(altered)


def test_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"format":"first","format":"second"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        verify_beta_controller_benchmark_file(path)


def test_rejects_nonfinite_json_numbers(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite.json"
    path.write_text('{"value":NaN}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite JSON constant"):
        verify_beta_controller_benchmark_file(path)


def test_rejects_noncanonical_json(real_report: dict[str, Any], tmp_path: Path) -> None:
    path = tmp_path / "noncanonical.json"
    path.write_text(json.dumps(real_report, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="not canonical"):
        verify_beta_controller_benchmark_file(path)


def test_rejects_symlinked_json(real_report: dict[str, Any], tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    link = tmp_path / "link.json"
    _write_canonical(target, real_report)
    link.symlink_to(target)

    with pytest.raises(ValueError, match="nonsymlink"):
        verify_beta_controller_benchmark_file(link)

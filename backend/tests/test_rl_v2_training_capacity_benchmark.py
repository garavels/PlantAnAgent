"""Outcome-blind and adversarial checks for the beta capacity benchmark."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping

import pytest

from plant_backend.rl import checkpoints
from plant_backend.rl.v2.mpc_schema import canonical_json, domain_sha256
from plant_backend.rl.v2.training import capacity_benchmark as capacity
from plant_backend.rl.v2.training import runtime_provenance
from plant_backend.rl.v2.training.config import (
    BETA_FIXTURE_RUN_KIND,
    BETA_I_LAGRANGIAN_ALGORITHM_ID,
    BETA_PI_LAGRANGIAN_ALGORITHM_ID,
    BETA_SCALAR_PPO_ALGORITHM_ID,
    ControllerNeutralTrainingConfigV1,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _runtime(
    *, clean: bool = True
) -> runtime_provenance.BetaTrainingRuntimeProvenanceV1:
    return runtime_provenance.BetaTrainingRuntimeProvenanceV1(
        git_head="a" * 40,
        worktree_clean=clean,
        worktree_status_sha256=(
            hashlib.sha256(b"").hexdigest() if clean else _sha("dirty-status")
        ),
        worktree_content_sha256=(
            runtime_provenance._empty_worktree_content_sha256()
            if clean
            else _sha("dirty-content")
        ),
        uv_lock_sha256=_sha("lock"),
        dependency_entrypoints_sha256=_sha("entrypoints"),
        python_version="3.12.11",
        python_implementation="CPython",
        platform="test-platform",
        numpy_version="2.2.6",
        scipy_version="1.15.3",
        torch_version="2.7.1",
        stable_baselines3_version="2.9.0",
        gymnasium_version="1.2.0",
        plant_sim_version="0.5.2",
        plant_sim_module_path="simulator/src/plant_sim/__init__.py",
        plant_sim_module_sha256=_sha("simulator-source"),
    )


def _host() -> dict[str, Any]:
    return {
        "platform": "test-platform",
        "system": "TestOS",
        "machine": "test-machine",
        "processor": None,
        "cpu_brand": None,
        "physical_cpu_count": 4,
        "logical_cpu_count": 8,
        "physical_memory_bytes": 16 * 1024**3,
        "torch_intraop_threads": 4,
        "torch_interop_threads": 2,
        "thread_environment": {
            name: None for name in capacity.CAPACITY_THREAD_ENVIRONMENT
        },
        "clock": {
            "name": "time.perf_counter_ns",
            "monotonic": True,
            "adjustable": False,
            "resolution_s": 1e-9,
        },
    }


def _fake_checkpoint_admitter(
    root: Path,
    checkpoint: Path,
    config: ControllerNeutralTrainingConfigV1,
    runtime: runtime_provenance.BetaTrainingRuntimeProvenanceV1,
) -> tuple[dict[str, Any], int, int]:
    files = capacity._checkpoint_file_inventory(root, checkpoint)
    logical_bytes = sum(int(item["logical_bytes"]) for item in files)
    allocated_bytes = sum(int(item["allocated_bytes"]) for item in files)
    algorithm = config.beta_rl.algorithm if config.beta_rl is not None else "missing"
    return (
        {
            "path": capacity._relative_to_root(root, checkpoint),
            "metadata_sha256": hashlib.sha256(
                capacity._read_regular_bytes(checkpoint / checkpoints.METADATA_FILE)
            ).hexdigest(),
            "checkpoint_evidence_sha256": _sha(f"evidence:{algorithm}"),
            "runtime_provenance_sha256": runtime.sha256,
            "environment_contract_sha256": _sha("environment-contract"),
            "checkpoint_contract_sha256": _sha(f"checkpoint-contract:{algorithm}"),
            "files": files,
            "logical_bytes": logical_bytes,
            "allocated_bytes": allocated_bytes,
            "rollout_entry_count": capacity.CAPACITY_ROLLOUT_COUNT,
            "allocator_entry_count": capacity.CAPACITY_ALLOCATOR_ENTRIES,
            "max_rollout_entry_bytes": 701,
            "max_allocator_entry_bytes": 211,
        },
        1_000_000_001,
        capacity.CAPACITY_OPTIMIZER_STEPS,
    )


def _successful_process_runner(
    observed: list[tuple[str, ...]],
) -> capacity.ProcessRunner:
    def run(
        command: tuple[str, ...], repository: Path, timeout_s: int
    ) -> tuple[int, bytes, bytes]:
        del repository
        observed.append(command)
        assert timeout_s == capacity.CAPACITY_CHILD_TIMEOUT_S
        assert command[:4] == (
            sys.executable,
            "-m",
            "plant_backend.rl.v2.training",
            "--config",
        )
        config = ControllerNeutralTrainingConfigV1.from_file(Path(command[4]))
        checkpoint = Path(config.output_dir) / "checkpoint"
        checkpoint.mkdir(parents=True, exist_ok=False)
        algorithm = (
            config.beta_rl.algorithm if config.beta_rl is not None else "missing"
        )
        (checkpoint / checkpoints.POLICY_FILE).write_bytes(
            f"policy:{algorithm}".encode("utf-8")
        )
        (checkpoint / checkpoints.METADATA_FILE).write_bytes(
            (canonical_json({"algorithm": algorithm}) + "\n").encode("utf-8")
        )
        (checkpoint / capacity.CONFIG_FILE).write_bytes(
            capacity._training_config_bytes(config)
        )
        return 0, f"completed:{algorithm}\n".encode("utf-8"), b""

    return run


def _clock(*durations_ns: int) -> Callable[[], int]:
    moments: list[int] = []
    current = 10_000
    for duration in durations_ns:
        moments.extend((current, current + duration))
        current += duration + 10_000
    values = iter(moments)
    return lambda: next(values)


def _execute_successfully(
    tmp_path: Path,
) -> tuple[
    Path,
    runtime_provenance.BetaTrainingRuntimeProvenanceV1,
    dict[str, Any],
    list[tuple[str, ...]],
]:
    runtime = _runtime()
    host = _host()
    observed: list[tuple[str, ...]] = []
    result = capacity._execute_synthetic_capacity_benchmark(
        tmp_path / "capacity-run",
        assert_exclusive=True,
        assert_first_attempt=True,
        process_runner=_successful_process_runner(observed),
        checkpoint_admitter=_fake_checkpoint_admitter,
        runtime_capture=lambda: runtime,
        host_capture=lambda: deepcopy(host),
        clock_ns=_clock(2_000_000_000, 2_500_000_000, 3_000_000_000),
    )
    assert result["status"] == "completed"
    return Path(result["report"]), runtime, host, observed


def _read_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rehash_report(path: Path, report: dict[str, Any]) -> None:
    runtime = runtime_provenance.BetaTrainingRuntimeProvenanceV1.from_dict(
        report["runtime_provenance"]
    )
    synthetic = report["schema"] == capacity.CAPACITY_SYNTHETIC_REPORT_SCHEMA
    deterministic_domain = (
        capacity.CAPACITY_SYNTHETIC_DETERMINISTIC_DOMAIN
        if synthetic
        else capacity.CAPACITY_DETERMINISTIC_DOMAIN
    )
    report_domain = (
        capacity.CAPACITY_SYNTHETIC_REPORT_DOMAIN
        if synthetic
        else capacity.CAPACITY_REPORT_DOMAIN
    )
    report["deterministic_evidence_sha256"] = domain_sha256(
        deterministic_domain,
        capacity._deterministic_evidence(
            protocol=report["protocol"],
            runtime=runtime,
            host=report["host"],
            cell_contract=report["cell_contract"],
            claim_boundary=report["claim_boundary"],
        ),
    )
    body = dict(report)
    body.pop("sha256", None)
    report["sha256"] = domain_sha256(report_domain, body)
    path.write_bytes((canonical_json(report) + "\n").encode("utf-8"))


def _verify_synthetic(
    path: Path,
    runtime: runtime_provenance.BetaTrainingRuntimeProvenanceV1,
    host: Mapping[str, Any],
) -> dict[str, Any]:
    return capacity._verify_synthetic_capacity_report(
        path,
        runtime_capture=lambda: runtime,
        checkpoint_admitter=_fake_checkpoint_admitter,
        host_capture=lambda: deepcopy(dict(host)),
    )


def test_frozen_cell_configs_and_reported_formulas_are_exact(tmp_path: Path) -> None:
    assert capacity.CAPACITY_ALGORITHM_ORDER == (
        BETA_PI_LAGRANGIAN_ALGORITHM_ID,
        BETA_I_LAGRANGIAN_ALGORITHM_ID,
        BETA_SCALAR_PPO_ALGORITHM_ID,
    )
    for algorithm in capacity.CAPACITY_ALGORITHM_ORDER:
        output = (tmp_path / algorithm).resolve()
        config = capacity._cell_config(output, algorithm)
        assert config.environment_id == capacity.beta_spaces.ENVIRONMENT_ID
        assert config.seed == 101
        assert config.total_timesteps == 8_192
        assert config.n_envs == 4
        assert config.checkpoint_every == 0
        assert config.resume_from is None
        assert config.output_dir == str(output)
        assert config.ppo == capacity.PPOConfig()
        assert config.beta_rl is not None
        assert config.beta_rl.algorithm == algorithm
        assert config.beta_rl.run_kind == BETA_FIXTURE_RUN_KIND
        assert config.beta_rl.episode_seed_base == 9_880_000

    contract = capacity._cell_contract()
    assert contract["child_timeout_s"] == 3_600
    assert contract["registered_rollouts_per_policy"] == 128
    assert contract["registered_policies_per_family"] == 5
    assert contract["sensitivity_ratios"] == [
        {"label": "1.00x", "numerator": 1, "denominator": 1},
        {"label": "1.25x", "numerator": 5, "denominator": 4},
        {"label": "1.50x", "numerator": 3, "denominator": 2},
    ]
    assert contract["storage_projection"] == {
        "additional_rollout_entries": 127,
        "rollout_entry_copies": 3,
        "additional_allocator_entries": 1_548,
        "allocator_entry_copies": 4,
        "growth_safety_multiplier": 2,
        "recommended_free_space": ("max(2 * projected_total, projected_total + 1 GiB)"),
    }


def test_metadata_training_config_accepts_only_the_canonical_json_round_trip(
    tmp_path: Path,
) -> None:
    config = capacity._cell_config(
        (tmp_path / "pi").resolve(),
        BETA_PI_LAGRANGIAN_ALGORITHM_ID,
    )
    in_memory = config.to_dict()
    round_tripped = json.loads(capacity._training_config_bytes(config))

    assert in_memory != round_tripped
    assert isinstance(in_memory["episode"]["battery_soc"], tuple)
    assert isinstance(round_tripped["episode"]["battery_soc"], list)
    assert capacity._metadata_training_config_bytes(round_tripped) == (
        capacity._training_config_bytes(config)
    )
    capacity._verify_metadata_training_config(round_tripped, config)


def test_production_admission_accepts_the_json_representation_before_next_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    output = root / "artifacts" / "pi-lagrangian"
    checkpoint = output / "checkpoint"
    checkpoint.mkdir(parents=True)
    config = capacity._cell_config(output, BETA_PI_LAGRANGIAN_ALGORITHM_ID)
    (checkpoint / capacity.CONFIG_FILE).write_bytes(
        capacity._training_config_bytes(config)
    )
    metadata = checkpoints.CheckpointMetadata(
        format=checkpoints.FORMAT_SB3_PPO,
        algorithm="deliberately-next-gate",
        seed=capacity.CAPACITY_OPTIMIZER_SEED,
        total_timesteps=capacity.CAPACITY_TRANSITIONS,
        training_config=json.loads(capacity._training_config_bytes(config)),
    )
    monkeypatch.setattr(
        capacity.checkpoints,
        "load_metadata",
        lambda *_args, **_kwargs: metadata,
    )

    with pytest.raises(
        capacity.CapacityBenchmarkError,
        match="checkpoint campaign identity differs",
    ):
        capacity._admit_checkpoint(root, checkpoint, config, _runtime())


@pytest.mark.parametrize(
    "mutation",
    (
        "changed_scalar",
        "changed_nested_sequence",
        "missing_key",
        "extra_key",
        "wrong_sequence_order",
        "nonfinite",
    ),
)
def test_metadata_training_config_rejects_semantic_or_structural_changes(
    tmp_path: Path,
    mutation: str,
) -> None:
    config = capacity._cell_config(
        (tmp_path / mutation).resolve(),
        BETA_PI_LAGRANGIAN_ALGORITHM_ID,
    )
    metadata_config = json.loads(capacity._training_config_bytes(config))
    if mutation == "changed_scalar":
        metadata_config["episode"]["horizon_h"] += 1.0
    elif mutation == "changed_nested_sequence":
        metadata_config["beta_task"]["disturbances"]["duration_h"][1] += 0.5
    elif mutation == "missing_key":
        metadata_config["episode"].pop("start_hour")
    elif mutation == "extra_key":
        metadata_config["beta_task"]["unexpected"] = True
    elif mutation == "wrong_sequence_order":
        metadata_config["episode"]["weather"]["forecast_windows_h"].reverse()
    else:
        metadata_config["ppo"]["learning_rate"] = float("nan")

    message = "not canonical JSON" if mutation == "nonfinite" else "differs"
    with pytest.raises(capacity.CapacityBenchmarkError, match=message):
        capacity._verify_metadata_training_config(metadata_config, config)


def test_projection_arithmetic_rounds_once_after_128_rollouts() -> None:
    measurement = capacity._build_measurement(2_000_000_007, 1_000_000_001)
    checkpoint = {
        "logical_bytes": 10_000,
        "max_rollout_entry_bytes": 701,
        "max_allocator_entry_bytes": 211,
    }
    projection = capacity._build_projection(measurement, checkpoint)
    fixed = 1_000_000_006
    rollout_term = 128 * 1_000_000_001
    assert projection["registered_policy_wall_ns_by_factor"] == {
        "1.00x": fixed + rollout_term,
        "1.25x": fixed + (rollout_term * 5 + 3) // 4,
        "1.50x": fixed + (rollout_term * 3 + 1) // 2,
    }
    expected_growth = 2 * (127 * 3 * 701 + 1_548 * 4 * 211)
    assert projection["structural_growth_allowance_bytes"] == expected_growth
    assert projection["projected_checkpoint_logical_bytes"] == (
        10_000 + expected_growth
    )
    assert projection["five_policy_family_logical_bytes"] == 5 * (
        10_000 + expected_growth
    )


def test_successful_synthetic_run_is_sequential_and_self_verifying(
    tmp_path: Path,
) -> None:
    report_path, runtime, host, observed = _execute_successfully(tmp_path)
    assert len(observed) == 3
    algorithms = [
        ControllerNeutralTrainingConfigV1.from_file(Path(command[4])).beta_rl.algorithm
        for command in observed
        if ControllerNeutralTrainingConfigV1.from_file(Path(command[4])).beta_rl
        is not None
    ]
    assert algorithms == list(capacity.CAPACITY_ALGORITHM_ORDER)

    verified = _verify_synthetic(report_path, runtime, host)
    report = _read_report(report_path)
    assert verified["status"] == "completed"
    assert report_path.name == capacity.CAPACITY_SYNTHETIC_REPORT_FILENAME
    assert report["schema"] == capacity.CAPACITY_SYNTHETIC_REPORT_SCHEMA
    assert report["execution_class"] == capacity.CAPACITY_SYNTHETIC_EXECUTION_CLASS
    assert report["claim_boundary"] == capacity.CAPACITY_SYNTHETIC_CLAIM
    assert report["result_claim"] is None
    assert report["campaign_projection"] is None
    assert not (report_path.parent / capacity.CAPACITY_REPORT_FILENAME).exists()
    assert report["operator_assertions"] == {
        "cells_executed_sequentially": True,
        "no_other_repository_benchmark_intentionally_running": True,
        "attempts_not_retried_or_selected": True,
    }
    assert report["campaign_arithmetic"] == capacity.REGISTERED_CAMPAIGN_ARITHMETIC
    assert [cell["status"] for cell in report["cells"]] == ["completed"] * 3
    assert capacity._campaign_projection(report["cells"]) is not None


def test_deterministic_evidence_has_exact_pre_run_allowlist() -> None:
    protocol = capacity._protocol_identity(capacity._repository_root())
    runtime = _runtime()
    host = _host()
    contract = capacity._cell_contract()
    evidence = capacity._deterministic_evidence(
        protocol=protocol,
        runtime=runtime,
        host=host,
        cell_contract=contract,
    )

    assert set(evidence) == {
        "protocol",
        "claim_boundary",
        "runtime_provenance_sha256",
        "host_capacity",
        "operator_assertions",
        "cell_contract",
        "planned_cells",
        "campaign_arithmetic",
    }
    assert set(evidence["host_capacity"]) == set(capacity._HOST_FIELDS) - {"clock"}
    assert "clock" not in evidence["host_capacity"]
    assert evidence["operator_assertions"] == capacity._operator_assertions()
    assert evidence["planned_cells"] == [
        {
            "order": order,
            "algorithm": algorithm,
            "optimizer_seed": capacity.CAPACITY_OPTIMIZER_SEED,
            "fixture_episode_seed_base": (capacity.CAPACITY_FIXTURE_EPISODE_SEED_BASE),
            "normalized_training_config_template_sha256": contract[
                "training_config_template_sha256"
            ],
        }
        for order, algorithm in enumerate(capacity.CAPACITY_ALGORITHM_ORDER, start=1)
    ]
    assert "cells" not in evidence
    assert "status" not in evidence
    assert "campaign_projection" not in evidence


def test_direct_and_transitive_timing_change_only_full_report_identity(
    tmp_path: Path,
) -> None:
    report_path, _, _, _ = _execute_successfully(tmp_path)
    report = _read_report(report_path)
    runtime = runtime_provenance.BetaTrainingRuntimeProvenanceV1.from_dict(
        report["runtime_provenance"]
    )

    def deterministic(value: Mapping[str, Any]) -> str:
        return domain_sha256(
            capacity.CAPACITY_SYNTHETIC_DETERMINISTIC_DOMAIN,
            capacity._deterministic_evidence(
                protocol=value["protocol"],
                runtime=runtime,
                host=value["host"],
                cell_contract=value["cell_contract"],
                claim_boundary=value["claim_boundary"],
            ),
        )

    def full(value: Mapping[str, Any]) -> str:
        body = dict(value)
        body.pop("sha256", None)
        return domain_sha256(capacity.CAPACITY_SYNTHETIC_REPORT_DOMAIN, body)

    base_deterministic = deterministic(report)
    base_full = full(report)
    assert base_deterministic == report["deterministic_evidence_sha256"]
    assert base_full == report["sha256"]

    def mutate_clock(value: dict[str, Any]) -> None:
        value["host"]["clock"]["resolution_s"] = 2e-9

    def mutate_wall(value: dict[str, Any]) -> None:
        value["cells"][0]["wall_elapsed_ns"] += 123

    def mutate_cpu(value: dict[str, Any]) -> None:
        value["cells"][0]["child_user_cpu_ns"] += 456

    def mutate_system_cpu(value: dict[str, Any]) -> None:
        value["cells"][0]["child_system_cpu_ns"] += 789

    def mutate_stdout(value: dict[str, Any]) -> None:
        value["cells"][0]["stdout"]["logical_bytes"] += 1
        value["cells"][0]["stdout"]["sha256"] = _sha("timed-stdout")

    def mutate_stderr(value: dict[str, Any]) -> None:
        value["cells"][0]["stderr"]["logical_bytes"] += 1
        value["cells"][0]["stderr"]["sha256"] = _sha("timed-stderr")

    def mutate_config_instance(value: dict[str, Any]) -> None:
        value["cells"][0]["config_sha256"] = _sha("different-output-root")

    def mutate_status(value: dict[str, Any]) -> None:
        value["cells"][0]["status"] = "failed"

    def mutate_exit(value: dict[str, Any]) -> None:
        value["cells"][0]["exit_code"] = None

    def mutate_failure(value: dict[str, Any]) -> None:
        value["cells"][0]["failure"] = capacity._failure("training_timeout")

    def mutate_metadata_digest(value: dict[str, Any]) -> None:
        checkpoint = value["cells"][0]["checkpoint"]
        checkpoint["metadata_sha256"] = _sha("different-created-at")

    def mutate_evidence_digest(value: dict[str, Any]) -> None:
        checkpoint = value["cells"][0]["checkpoint"]
        checkpoint["checkpoint_evidence_sha256"] = _sha("different-elapsed")

    def mutate_checkpoint_file(value: dict[str, Any]) -> None:
        checkpoint = value["cells"][0]["checkpoint"]
        checkpoint["files"][0]["sha256"] = _sha("different-zip-timestamp")
        checkpoint["files"][0]["logical_bytes"] += 1

    def mutate_checkpoint_size(value: dict[str, Any]) -> None:
        checkpoint = value["cells"][0]["checkpoint"]
        checkpoint["logical_bytes"] += 1
        checkpoint["allocated_bytes"] += 4_096

    def mutate_max_rollout_entry(value: dict[str, Any]) -> None:
        checkpoint = value["cells"][0]["checkpoint"]
        checkpoint["max_rollout_entry_bytes"] += 1

    def mutate_max_allocator_entry(value: dict[str, Any]) -> None:
        checkpoint = value["cells"][0]["checkpoint"]
        checkpoint["max_allocator_entry_bytes"] += 1

    def remove_checkpoint(value: dict[str, Any]) -> None:
        value["cells"][0]["checkpoint"] = None

    def mutate_measurement(value: dict[str, Any]) -> None:
        value["cells"][0]["measurement"]["rollout_elapsed_ns"] += 1

    def mutate_projection(value: dict[str, Any]) -> None:
        value["cells"][0]["projection"]["registered_policy_wall_ns_by_factor"][
            "1.00x"
        ] += 1

    for mutation in (
        mutate_clock,
        mutate_wall,
        mutate_cpu,
        mutate_system_cpu,
        mutate_stdout,
        mutate_stderr,
        mutate_config_instance,
        mutate_status,
        mutate_exit,
        mutate_failure,
        mutate_metadata_digest,
        mutate_evidence_digest,
        mutate_checkpoint_file,
        mutate_checkpoint_size,
        mutate_max_rollout_entry,
        mutate_max_allocator_entry,
        remove_checkpoint,
        mutate_measurement,
        mutate_projection,
    ):
        changed = deepcopy(report)
        mutation(changed)
        assert deterministic(changed) == base_deterministic
        assert full(changed) != base_full


def test_deterministic_evidence_changes_for_bound_semantic_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = capacity._protocol_identity(capacity._repository_root())
    runtime = _runtime()
    host = _host()
    contract = capacity._cell_contract()

    def digest(
        *,
        selected_protocol: Mapping[str, Any] = protocol,
        selected_runtime: runtime_provenance.BetaTrainingRuntimeProvenanceV1 = runtime,
        selected_host: Mapping[str, Any] = host,
        selected_contract: Mapping[str, Any] = contract,
    ) -> str:
        return domain_sha256(
            capacity.CAPACITY_DETERMINISTIC_DOMAIN,
            capacity._deterministic_evidence(
                protocol=selected_protocol,
                runtime=selected_runtime,
                host=selected_host,
                cell_contract=selected_contract,
            ),
        )

    baseline = digest()
    changed_protocol = deepcopy(protocol)
    changed_protocol["id"] = "different-protocol"
    assert digest(selected_protocol=changed_protocol) != baseline

    changed_runtime = _runtime(clean=False)
    assert digest(selected_runtime=changed_runtime) != baseline

    changed_host = deepcopy(host)
    changed_host["logical_cpu_count"] += 1
    assert digest(selected_host=changed_host) != baseline

    changed_contract = deepcopy(contract)
    changed_contract["requested_and_collected_transitions"] += 1
    assert digest(selected_contract=changed_contract) != baseline

    with monkeypatch.context() as scoped:
        scoped.setattr(
            capacity,
            "CAPACITY_OPTIMIZER_SEED",
            capacity.CAPACITY_OPTIMIZER_SEED + 1,
        )
        assert digest() != baseline
    with monkeypatch.context() as scoped:
        scoped.setattr(
            capacity,
            "CAPACITY_FIXTURE_EPISODE_SEED_BASE",
            capacity.CAPACITY_FIXTURE_EPISODE_SEED_BASE + 1,
        )
        assert digest() != baseline
    with monkeypatch.context() as scoped:
        scoped.setattr(
            capacity,
            "CAPACITY_ALGORITHM_ORDER",
            tuple(reversed(capacity.CAPACITY_ALGORITHM_ORDER)),
        )
        assert digest() != baseline
    with monkeypatch.context() as scoped:
        altered_assertions = capacity._operator_assertions()
        altered_assertions["attempts_not_retried_or_selected"] = False
        scoped.setattr(capacity, "_operator_assertions", lambda: altered_assertions)
        assert digest() != baseline
    with monkeypatch.context() as scoped:
        altered_arithmetic = dict(capacity.REGISTERED_CAMPAIGN_ARITHMETIC)
        altered_arithmetic["policies"] += 1
        scoped.setattr(capacity, "REGISTERED_CAMPAIGN_ARITHMETIC", altered_arithmetic)
        assert digest() != baseline


def test_external_output_roots_do_not_change_deterministic_identity(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_path, _, _, _ = _execute_successfully(first_root)
    second_path, _, _, _ = _execute_successfully(second_root)
    first = _read_report(first_path)
    second = _read_report(second_path)

    assert first["cells"][0]["config_sha256"] != second["cells"][0]["config_sha256"]
    assert first["sha256"] != second["sha256"]
    assert (
        first["deterministic_evidence_sha256"]
        == second["deterministic_evidence_sha256"]
    )


def test_public_execution_and_verification_surfaces_have_no_test_hooks() -> None:
    execute_parameters = inspect.signature(
        capacity.execute_capacity_benchmark
    ).parameters
    verify_parameters = inspect.signature(capacity.verify_capacity_report).parameters
    assert set(execute_parameters) == {
        "output",
        "assert_exclusive",
        "assert_first_attempt",
    }
    assert set(verify_parameters) == {"report_path"}
    assert set(inspect.signature(capacity._verify_capacity_report).parameters) == {
        "report_path"
    }
    assert set(
        inspect.signature(capacity._run_production_capacity_cells).parameters
    ) == {"output", "assert_exclusive", "assert_first_attempt"}
    assert set(
        inspect.signature(capacity._collect_production_checkpoint_admissions).parameters
    ) == {"envelope", "context"}
    assert not hasattr(capacity, "_build_report")
    assert not hasattr(capacity, "_verify_report")
    assert not hasattr(capacity, "_collect_report_checkpoint_admissions")
    for _, function in inspect.getmembers(capacity, inspect.isfunction):
        assert "production_eligible" not in inspect.signature(function).parameters


def test_synthetic_issuer_has_no_production_builder_or_verifier_call_path() -> None:
    issuer_source = inspect.getsource(capacity._execute_synthetic_capacity_benchmark)
    builder_source = inspect.getsource(capacity._build_synthetic_report)
    for forbidden in (
        "CAPACITY_REPORT_FILENAME",
        "CAPACITY_REPORT_SCHEMA",
        "CAPACITY_REPORT_DOMAIN",
        "CAPACITY_RESULT_CLAIM",
        "_verify_capacity_report",
    ):
        assert forbidden not in issuer_source
        assert forbidden not in builder_source


def test_hookable_cell_runner_cannot_publish_any_report(tmp_path: Path) -> None:
    runtime = _runtime()
    host = _host()
    observed: list[tuple[str, ...]] = []
    trace = capacity._run_synthetic_capacity_cells(
        tmp_path / "trace-only",
        assert_exclusive=True,
        assert_first_attempt=True,
        process_runner=_successful_process_runner(observed),
        checkpoint_admitter=_fake_checkpoint_admitter,
        runtime_capture=lambda: runtime,
        host_capture=lambda: deepcopy(host),
        clock_ns=_clock(2_000_000_000, 2_500_000_000, 3_000_000_000),
    )

    assert len(observed) == 3
    assert len(trace.data.cells) == 3
    assert not hasattr(trace, "report")
    assert not (trace.data.root / capacity.CAPACITY_REPORT_FILENAME).exists()
    assert not (trace.data.root / capacity.CAPACITY_SYNTHETIC_REPORT_FILENAME).exists()


@pytest.mark.parametrize(
    "rejected_schema",
    (
        "plant-beta-constrained-rl-capacity-report-v1",
        "plant-beta-constrained-rl-capacity-report-v2",
        capacity.CAPACITY_SYNTHETIC_REPORT_SCHEMA,
    ),
)
def test_production_verifier_rejects_old_or_synthetic_schema_before_external_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rejected_schema: str
) -> None:
    synthetic_path, _, _, _ = _execute_successfully(tmp_path)
    production_name = synthetic_path.parent / capacity.CAPACITY_REPORT_FILENAME
    report = _read_report(synthetic_path)
    report["schema"] = rejected_schema
    production_name.write_bytes((canonical_json(report) + "\n").encode("utf-8"))
    observed: list[str] = []

    def unexpected_runtime() -> runtime_provenance.BetaTrainingRuntimeProvenanceV1:
        observed.append("runtime")
        raise AssertionError("runtime capture must not run")

    def unexpected_host() -> Mapping[str, Any]:
        observed.append("host")
        raise AssertionError("host capture must not run")

    def unexpected_admission(*_args: object) -> tuple[dict[str, Any], int, int]:
        observed.append("checkpoint")
        raise AssertionError("checkpoint admission must not run")

    monkeypatch.setattr(capacity, "_clean_runtime_capture", unexpected_runtime)
    monkeypatch.setattr(capacity, "_host_identity", unexpected_host)
    monkeypatch.setattr(capacity, "_admit_checkpoint", unexpected_admission)

    with pytest.raises(capacity.CapacityBenchmarkError, match="schema differs"):
        capacity.verify_capacity_report(production_name)
    assert observed == []


@pytest.mark.parametrize(
    "rejected_schema",
    (
        "plant-beta-constrained-rl-capacity-synthetic-report-v1",
        capacity.CAPACITY_REPORT_SCHEMA,
    ),
)
def test_synthetic_verifier_rejects_old_or_production_schema_before_external_access(
    tmp_path: Path,
    rejected_schema: str,
) -> None:
    synthetic_path, _, _, _ = _execute_successfully(tmp_path)
    report = _read_report(synthetic_path)
    report["schema"] = rejected_schema
    synthetic_path.write_bytes((canonical_json(report) + "\n").encode("utf-8"))
    observed: list[str] = []

    def unexpected_runtime() -> runtime_provenance.BetaTrainingRuntimeProvenanceV1:
        observed.append("runtime")
        raise AssertionError("runtime capture must not run")

    def unexpected_host() -> Mapping[str, Any]:
        observed.append("host")
        raise AssertionError("host capture must not run")

    def unexpected_admission(*_args: object) -> tuple[dict[str, Any], int, int]:
        observed.append("checkpoint")
        raise AssertionError("checkpoint admission must not run")

    with pytest.raises(capacity.CapacityBenchmarkError, match="schema differs"):
        capacity._verify_synthetic_capacity_report(
            synthetic_path,
            runtime_capture=unexpected_runtime,
            checkpoint_admitter=unexpected_admission,
            host_capture=unexpected_host,
        )
    assert observed == []


def test_checkpoint_admission_seams_reject_the_opposite_evidence_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synthetic_path, runtime, host, _ = _execute_successfully(tmp_path)
    synthetic_report = _read_report(synthetic_path)
    context = capacity._ReportContext(
        protocol=dict(synthetic_report["protocol"]),
        runtime=runtime,
        host=host,
        contract=dict(synthetic_report["cell_contract"]),
    )
    synthetic_envelope = capacity._ReportEnvelope(
        path=synthetic_path,
        root=synthetic_path.parent,
        report=synthetic_report,
        supplied_sha256=synthetic_report["sha256"],
    )
    observed: list[str] = []

    def unexpected_production_admission(
        *_args: object,
    ) -> tuple[dict[str, Any], int, int]:
        observed.append("production")
        raise AssertionError("production admission must not run")

    monkeypatch.setattr(
        capacity,
        "_admit_checkpoint",
        unexpected_production_admission,
    )
    with pytest.raises(
        capacity.CapacityBenchmarkError,
        match="production checkpoint admission requires",
    ):
        capacity._collect_production_checkpoint_admissions(
            synthetic_envelope,
            context,
        )

    production_report = deepcopy(synthetic_report)
    production_report["schema"] = capacity.CAPACITY_REPORT_SCHEMA
    production_report.pop("execution_class")
    production_envelope = capacity._ReportEnvelope(
        path=synthetic_path.parent / capacity.CAPACITY_REPORT_FILENAME,
        root=synthetic_path.parent,
        report=production_report,
        supplied_sha256="0" * 64,
    )

    def unexpected_synthetic_admission(
        *_args: object,
    ) -> tuple[dict[str, Any], int, int]:
        observed.append("synthetic")
        raise AssertionError("synthetic admission must not run")

    with pytest.raises(
        capacity.CapacityBenchmarkError,
        match="synthetic checkpoint admission requires",
    ):
        capacity._collect_synthetic_checkpoint_admissions(
            production_envelope,
            context,
            unexpected_synthetic_admission,
        )
    assert observed == []


@pytest.mark.parametrize(
    ("exclusive", "first_attempt", "message"),
    [
        (False, True, "assert-exclusive"),
        (True, False, "assert-first-attempt"),
    ],
)
def test_execution_requires_both_operator_assertions_before_output_creation(
    tmp_path: Path,
    exclusive: bool,
    first_attempt: bool,
    message: str,
) -> None:
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match=message):
        capacity._run_synthetic_capacity_cells(
            output,
            assert_exclusive=exclusive,
            assert_first_attempt=first_attempt,
            process_runner=lambda *_args: (0, b"", b""),
            checkpoint_admitter=_fake_checkpoint_admitter,
            runtime_capture=_runtime,
            host_capture=_host,
            clock_ns=_clock(1, 1, 1),
        )
    assert not output.exists()


def test_execution_rejects_dirty_runtime_before_output_creation(tmp_path: Path) -> None:
    output = tmp_path / "not-created"
    dirty = _runtime(clean=False)
    with pytest.raises(capacity.CapacityBenchmarkError, match="clean worktree"):
        capacity._run_synthetic_capacity_cells(
            output,
            assert_exclusive=True,
            assert_first_attempt=True,
            process_runner=lambda *_args: (0, b"", b""),
            checkpoint_admitter=_fake_checkpoint_admitter,
            runtime_capture=lambda: dirty,
            host_capture=_host,
            clock_ns=_clock(1, 1, 1),
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, None), ("1", 1), ("32", 32), ("2147483647", 2_147_483_647)],
)
def test_thread_environment_parser_retains_only_canonical_positive_counts(
    raw: str | None, expected: int | None
) -> None:
    assert capacity._canonical_thread_count("OMP_NUM_THREADS", raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "0",
        "-1",
        "+1",
        "01",
        " 1",
        "1 ",
        "1,2",
        "1; observed_reward=123",
        "timestamp=2026-09-05T12:00:00Z",
        "１２",
        "2147483648",
    ],
)
def test_thread_environment_parser_rejects_free_text_and_noncanonical_counts(
    raw: str,
) -> None:
    with pytest.raises(capacity.CapacityBenchmarkError, match="thread setting"):
        capacity._canonical_thread_count("OMP_NUM_THREADS", raw)


def test_noncanonical_thread_environment_fails_before_output_creation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "not-created"
    host = _host()
    host["thread_environment"]["OMP_NUM_THREADS"] = "observed_reward=123"
    with pytest.raises(TypeError, match="must be an integer"):
        capacity._run_synthetic_capacity_cells(
            output,
            assert_exclusive=True,
            assert_first_attempt=True,
            process_runner=lambda *_args: (0, b"", b""),
            checkpoint_admitter=_fake_checkpoint_admitter,
            runtime_capture=_runtime,
            host_capture=lambda: host,
            clock_ns=_clock(1, 1, 1),
        )
    assert not output.exists()


def test_external_output_contract_rejects_relative_existing_and_repository_paths(
    tmp_path: Path,
) -> None:
    repository = capacity._repository_root().resolve()
    with pytest.raises(ValueError, match="absolute"):
        capacity._new_external_root(Path("relative"), repository)
    with pytest.raises(ValueError, match="new non-existing"):
        capacity._new_external_root(tmp_path, repository)
    repository_output = repository / "backend" / "capacity-must-not-be-created"
    with pytest.raises(ValueError, match="external"):
        capacity._new_external_root(repository_output, repository)
    assert not repository_output.exists()


def test_checkpoint_admission_rejects_changed_internal_training_config_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    checkpoint = root / "checkpoint"
    checkpoint.mkdir(parents=True)
    config = capacity._cell_config(
        tmp_path / "output", capacity.CAPACITY_ALGORITHM_ORDER[0]
    )
    metadata = checkpoints.CheckpointMetadata(
        format=checkpoints.FORMAT_SB3_PPO,
        algorithm=capacity.CAPACITY_ALGORITHM_ORDER[0],
        seed=capacity.CAPACITY_OPTIMIZER_SEED,
        total_timesteps=capacity.CAPACITY_TRANSITIONS,
        training_config=json.loads(capacity._training_config_bytes(config)),
    )
    metadata_loads = 0

    def load_metadata(*_args: object, **_kwargs: object) -> object:
        nonlocal metadata_loads
        metadata_loads += 1
        return metadata

    monkeypatch.setattr(capacity.checkpoints, "load_metadata", load_metadata)
    (checkpoint / capacity.CONFIG_FILE).write_bytes(b"{}\n")
    with pytest.raises(capacity.CapacityBenchmarkError, match="training_config"):
        capacity._admit_checkpoint(root, checkpoint, config, _runtime())
    assert metadata_loads == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("cell_order", "identity/order"),
        ("cell_projection", "capacity cell projection arithmetic"),
        ("checkpoint_digest", "checkpoint changed"),
        ("operator_assertion", "operator assertions"),
        ("unexpected_outcome", "fields differ"),
    ],
)
def test_rehashed_semantic_report_tampering_is_rejected(
    tmp_path: Path, mutation: str, message: str
) -> None:
    report_path, runtime, host, _ = _execute_successfully(tmp_path)
    report = _read_report(report_path)
    if mutation == "cell_order":
        report["cells"][0]["order"] = 2
    elif mutation == "cell_projection":
        report["cells"][0]["projection"]["registered_policy_wall_ns_by_factor"][
            "1.25x"
        ] += 1
    elif mutation == "checkpoint_digest":
        report["cells"][0]["checkpoint"]["checkpoint_evidence_sha256"] = _sha(
            "substituted-evidence"
        )
    elif mutation == "operator_assertion":
        report["operator_assertions"]["attempts_not_retried_or_selected"] = False
    elif mutation == "unexpected_outcome":
        report["observed_reward_returns"] = [123.0]
    else:  # pragma: no cover - parameter list is closed
        raise AssertionError(mutation)
    _rehash_report(report_path, report)

    with pytest.raises(capacity.CapacityBenchmarkError, match=message):
        _verify_synthetic(report_path, runtime, host)


def test_stale_outer_report_digest_fails_before_artifact_admission(
    tmp_path: Path,
) -> None:
    report_path, runtime, host, _ = _execute_successfully(tmp_path)
    report = _read_report(report_path)
    report["cells"][0]["wall_elapsed_ns"] += 1
    report_path.write_bytes((canonical_json(report) + "\n").encode("utf-8"))

    calls = 0

    def admitter(*_args: object) -> tuple[dict[str, Any], int, int]:
        nonlocal calls
        calls += 1
        raise AssertionError("artifact admission must not run for a stale report hash")

    with pytest.raises(capacity.CapacityBenchmarkError, match="report digest"):
        capacity._verify_synthetic_capacity_report(
            report_path,
            runtime_capture=lambda: runtime,
            checkpoint_admitter=admitter,
            host_capture=lambda: host,
        )
    assert calls == 0


def test_external_config_tampering_fails_even_when_all_digests_are_recomputed(
    tmp_path: Path,
) -> None:
    report_path, runtime, host, _ = _execute_successfully(tmp_path)
    report = _read_report(report_path)
    config_path = report_path.parent / report["cells"][0]["config_path"]
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config_payload["notes"] = "post-run substitution"
    config_raw = (
        json.dumps(config_payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    config_path.write_bytes(config_raw)
    report["cells"][0]["config_sha256"] = hashlib.sha256(config_raw).hexdigest()
    _rehash_report(report_path, report)

    with pytest.raises(capacity.CapacityBenchmarkError, match="frozen config"):
        _verify_synthetic(report_path, runtime, host)


def test_immediate_verifier_recaptures_and_matches_host_identity(
    tmp_path: Path,
) -> None:
    report_path, runtime, host, _ = _execute_successfully(tmp_path)
    different = deepcopy(host)
    different["logical_cpu_count"] = int(different["logical_cpu_count"]) + 1

    with pytest.raises(capacity.CapacityBenchmarkError, match="host differs"):
        capacity._verify_synthetic_capacity_report(
            report_path,
            runtime_capture=lambda: runtime,
            checkpoint_admitter=_fake_checkpoint_admitter,
            host_capture=lambda: different,
        )


def test_report_verifier_rejects_intermediate_directory_symlink(tmp_path: Path) -> None:
    report_path, runtime, host, _ = _execute_successfully(tmp_path)
    logs = report_path.parent / "logs"
    retained = report_path.parent / "retained-logs"
    logs.rename(retained)
    logs.symlink_to(retained, target_is_directory=True)

    with pytest.raises(capacity.CapacityBenchmarkError, match="contain a symlink"):
        _verify_synthetic(report_path, runtime, host)


def test_regular_reader_rejects_final_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"{}\n")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(capacity.CapacityBenchmarkError, match="nonsymlink"):
        capacity._read_regular_bytes(link)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b'{"a":1,"a":2}\n', "duplicate JSON key"),
        (b'{"a":NaN}\n', "non-finite JSON constant"),
        (b'{"a": 1}\n', "not canonical"),
        (b'{"a":1}', "not canonical"),
    ],
)
def test_canonical_report_loader_rejects_ambiguous_or_noncanonical_json(
    tmp_path: Path, raw: bytes, message: str
) -> None:
    path = tmp_path / "report.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match=message):
        capacity._load_canonical_json(path)


def _execute_failed(
    output: Path,
    runner: capacity.ProcessRunner,
    *,
    admitter: capacity.CheckpointAdmitter = _fake_checkpoint_admitter,
) -> tuple[Path, dict[str, Any]]:
    runtime = _runtime()
    host = _host()
    result = capacity._execute_synthetic_capacity_benchmark(
        output,
        assert_exclusive=True,
        assert_first_attempt=True,
        process_runner=runner,
        checkpoint_admitter=admitter,
        runtime_capture=lambda: runtime,
        host_capture=lambda: deepcopy(host),
        clock_ns=_clock(1_000_000_000, 1_000_000_000, 1_000_000_000),
    )
    assert result["status"] == "failed"
    report_path = Path(result["report"])
    return report_path, _read_report(report_path)


def test_signal_terminated_children_are_retained_without_retry_or_projection(
    tmp_path: Path,
) -> None:
    attempts = 0

    def signal_failure(
        _command: tuple[str, ...], _repository: Path, _timeout_s: int
    ) -> tuple[int, bytes, bytes]:
        nonlocal attempts
        attempts += 1
        return -9, b"partial standard output", b"partial standard error"

    report_path, report = _execute_failed(tmp_path / "signal", signal_failure)
    assert attempts == 3
    assert report["result_claim"] is None
    assert report["campaign_projection"] is None
    assert report["status"] == "failed"
    for cell in report["cells"]:
        assert cell["status"] == "failed"
        assert cell["exit_code"] == -9
        assert cell["checkpoint"] is None
        assert cell["measurement"] is None
        assert cell["projection"] is None
        assert cell["failure"] == {
            "phase": "training_process",
            "category": "training_nonzero_exit",
            "message": (
                "training process failed; retained output is identified only by digest"
            ),
        }
        assert (
            report_path.parent / cell["stdout"]["path"]
        ).read_bytes() == b"partial standard output"
        assert (
            report_path.parent / cell["stderr"]["path"]
        ).read_bytes() == b"partial standard error"


def test_timeout_retains_partial_streams_and_is_not_retried(tmp_path: Path) -> None:
    attempts = 0

    def timeout(
        command: tuple[str, ...], _repository: Path, timeout_s: int
    ) -> tuple[int, bytes, bytes]:
        nonlocal attempts
        attempts += 1
        raise subprocess.TimeoutExpired(
            command,
            timeout_s,
            output=b"timeout partial stdout",
            stderr=b"timeout partial stderr",
        )

    report_path, report = _execute_failed(tmp_path / "timeout", timeout)
    assert attempts == 3
    for cell in report["cells"]:
        assert cell["exit_code"] is None
        assert cell["failure"] == {
            "phase": "training_process",
            "category": "training_timeout",
            "message": (
                "training process timed out; retained output is identified only by digest"
            ),
        }
        assert (
            report_path.parent / cell["stdout"]["path"]
        ).read_bytes() == b"timeout partial stdout"
        assert (
            report_path.parent / cell["stderr"]["path"]
        ).read_bytes() == b"timeout partial stderr"


def test_training_launch_failure_uses_closed_vocabulary_without_exception_text(
    tmp_path: Path,
) -> None:
    class Reward123SecretOutcomeError(RuntimeError):
        pass

    def refuse_launch(*_args: object) -> tuple[int, bytes, bytes]:
        raise Reward123SecretOutcomeError("reward=123.4 secret-outcome")

    _, report = _execute_failed(tmp_path / "launch-failure", refuse_launch)
    encoded = canonical_json(report)
    assert "Reward123SecretOutcomeError" not in encoded
    assert "123.4" not in encoded
    assert "secret-outcome" not in encoded
    for cell in report["cells"]:
        assert cell["exit_code"] is None
        assert cell["failure"] == {
            "phase": "training_process",
            "category": "training_launch_error",
            "message": (
                "training process could not be launched; no exception detail is reported"
            ),
        }


def test_checkpoint_admission_failure_is_retained_without_outcome_text(
    tmp_path: Path,
) -> None:
    class Cost999SecretAdmissionError(RuntimeError):
        pass

    observed: list[tuple[str, ...]] = []
    admissions = 0

    def refuse(*_args: object) -> tuple[dict[str, Any], int, int]:
        nonlocal admissions
        admissions += 1
        raise Cost999SecretAdmissionError("reward=123.4 secret-outcome")

    _, report = _execute_failed(
        tmp_path / "admission",
        _successful_process_runner(observed),
        admitter=refuse,
    )
    assert len(observed) == 3
    assert admissions == 3
    encoded = canonical_json(report)
    assert "Cost999SecretAdmissionError" not in encoded
    assert "123.4" not in encoded
    assert "secret-outcome" not in encoded
    for cell in report["cells"]:
        assert cell["exit_code"] == 0
        assert cell["failure"] == {
            "phase": "checkpoint_admission",
            "category": "checkpoint_admission_error",
            "message": "checkpoint admission failed; no controller outcome is reported",
        }


def test_failed_report_rejects_unsanitized_failure_even_when_rehashed(
    tmp_path: Path,
) -> None:
    report_path, report = _execute_failed(
        tmp_path / "failed",
        lambda *_args: (7, b"", b""),
    )
    runtime = _runtime()
    host = _host()
    report["cells"][0]["failure"]["message"] = "observed reward was 10"
    _rehash_report(report_path, report)
    with pytest.raises(capacity.CapacityBenchmarkError, match="not canonical"):
        _verify_synthetic(report_path, runtime, host)


@pytest.mark.parametrize("mutation", ["extra_exception_type", "unknown_category"])
def test_failed_report_rejects_open_failure_vocabulary_even_when_rehashed(
    tmp_path: Path,
    mutation: str,
) -> None:
    report_path, report = _execute_failed(
        tmp_path / mutation,
        lambda *_args: (7, b"", b""),
    )
    failure = report["cells"][0]["failure"]
    if mutation == "extra_exception_type":
        failure["exception_type"] = "ObservedReward123"
        message = "fields differ"
    else:
        failure["category"] = "observed_reward_123"
        message = "category differs"
    _rehash_report(report_path, report)
    with pytest.raises(capacity.CapacityBenchmarkError, match=message):
        _verify_synthetic(report_path, _runtime(), _host())


def test_report_has_no_controller_outcome_fields(tmp_path: Path) -> None:
    report_path, _, _, _ = _execute_successfully(tmp_path)
    report = _read_report(report_path)
    observed_keys: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                observed_keys.add(str(key))
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(report)
    assert observed_keys.isdisjoint(
        {
            "reward_returns",
            "cost_returns",
            "actions",
            "multipliers",
            "losses",
            "policy_outputs",
            "market_regimes",
            "outages",
            "inventories",
        }
    )


def test_protocol_digest_constant_matches_current_frozen_bytes() -> None:
    repository = capacity._repository_root()
    protocol = repository / capacity.CAPACITY_PROTOCOL_PATH
    repair = repository / capacity.CAPACITY_REPAIR_PATH
    clarification = repository / capacity.CAPACITY_REPAIR_CLARIFICATION_PATH
    clarification_002 = repository / capacity.CAPACITY_REPAIR_CLARIFICATION_002_PATH
    repair_002 = repository / capacity.CAPACITY_REPAIR_002_PATH
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == (
        capacity.CAPACITY_PROTOCOL_SHA256
    )
    assert hashlib.sha256(repair.read_bytes()).hexdigest() == (
        capacity.CAPACITY_REPAIR_SHA256
    )
    assert hashlib.sha256(clarification.read_bytes()).hexdigest() == (
        capacity.CAPACITY_REPAIR_CLARIFICATION_SHA256
    )
    assert hashlib.sha256(clarification_002.read_bytes()).hexdigest() == (
        capacity.CAPACITY_REPAIR_CLARIFICATION_002_SHA256
    )
    assert hashlib.sha256(repair_002.read_bytes()).hexdigest() == (
        capacity.CAPACITY_REPAIR_002_SHA256
    )
    assert capacity.CAPACITY_REPORT_SCHEMA == (
        "plant-beta-constrained-rl-capacity-report-v3"
    )
    assert capacity.CAPACITY_REPORT_DOMAIN == (
        "plant-beta-constrained-rl-capacity-report-v3"
    )
    assert capacity.CAPACITY_SYNTHETIC_REPORT_SCHEMA == (
        "plant-beta-constrained-rl-capacity-synthetic-report-v2"
    )
    assert capacity.CAPACITY_SYNTHETIC_REPORT_DOMAIN == (
        "plant-beta-constrained-rl-capacity-synthetic-report-v2"
    )
    assert capacity.CAPACITY_DETERMINISTIC_DOMAIN == (
        "plant-beta-constrained-rl-capacity-deterministic-evidence-v3"
    )
    assert capacity.CAPACITY_SYNTHETIC_DETERMINISTIC_DOMAIN == (
        "plant-beta-constrained-rl-capacity-synthetic-deterministic-evidence-v2"
    )
    assert capacity._protocol_identity(repository)["repairs"] == [
        {
            "id": capacity.CAPACITY_REPAIR_ID,
            "path": capacity.CAPACITY_REPAIR_PATH,
            "raw_file_sha256": capacity.CAPACITY_REPAIR_SHA256,
            "clarifications": [
                {
                    "id": capacity.CAPACITY_REPAIR_CLARIFICATION_ID,
                    "path": capacity.CAPACITY_REPAIR_CLARIFICATION_PATH,
                    "raw_file_sha256": (capacity.CAPACITY_REPAIR_CLARIFICATION_SHA256),
                },
                {
                    "id": capacity.CAPACITY_REPAIR_CLARIFICATION_002_ID,
                    "path": capacity.CAPACITY_REPAIR_CLARIFICATION_002_PATH,
                    "raw_file_sha256": (
                        capacity.CAPACITY_REPAIR_CLARIFICATION_002_SHA256
                    ),
                },
            ],
        },
        {
            "id": capacity.CAPACITY_REPAIR_002_ID,
            "path": capacity.CAPACITY_REPAIR_002_PATH,
            "raw_file_sha256": capacity.CAPACITY_REPAIR_002_SHA256,
        },
    ]

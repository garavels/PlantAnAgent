"""Instrumented four-arm fixtures and detached latency/physics cross-binding."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from plant_backend.rl.v2 import beta_controller_benchmark as benchmark
from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta
from plant_backend.rl.v2.policy_setpoint_controller import (
    NormalizedSetpointPolicyControllerV1,
)
from plant_backend.rl.v2.study import arm_evidence as arms
from plant_backend.rl.v2.study import latency, latency_evidence as timing
from plant_backend.rl.v2.study.contract import (
    PROTOCOL_ID,
    PROTOCOL_SHA256,
    analysis_contract,
)
from plant_backend.rl.v2.training import custody_io as io
from plant_backend.rl.v2.training.custody_json import authenticate, canonical_bytes
from test_rl_v2_beta_controller_benchmark_verifier import _config, _loaded


@pytest.fixture(scope="module")
def measured(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("study-instrumented-latency").resolve()
    config = replace(
        _config(), notes="synthetic study timing fixture; not trained"
    ).validate()
    loaded = _loaded(root, config)
    before = timing.capture_latency_host()
    source = benchmark._source_provenance(require_clean=False)
    recorders = {
        arm: latency.LatencyRecorder(arm) for arm in benchmark.BETA_CONTROLLER_ARM_ORDER
    }
    warmed = recorders["PPO"].warm_policy(loaded)
    settings = arms._settings(config)
    prepared = benchmark._prepare_cell(
        warmed,
        config,
        seed=9_000_091,
        service_settings=settings,
        economic_controller_factory=benchmark._real_beta_economic_controller,
    )
    runs = {}
    try:
        for arm in benchmark.BETA_CONTROLLER_ARM_ORDER:
            if arm == "M24":
                run = benchmark._service_mpc_result(
                    environment=prepared.environments[arm],
                    controller=prepared.controllers[arm],
                    settings=settings,
                    manifest_identity=prepared.manifest_identity,
                    initial_vector=prepared.policy_observations[arm],
                    study_timing=recorders[arm],
                )
            else:
                run = benchmark._generic_arm_result(
                    arm=arm,
                    environment=prepared.environments[arm],
                    controller=prepared.controllers[arm],
                    manifest_identity=prepared.manifest_identity,
                    initial_vector=prepared.policy_observations[arm],
                    study_timing=recorders[arm],
                )
            runs[arm] = run
        # Same permitted fixture episode, original unwrapped policy, to verify
        # that recording has no effect on deterministic physical/economic output.
        environment = prepared.environments["PPO"]
        vector, _ = environment.reset(seed=9_000_091)
        controller = NormalizedSetpointPolicyControllerV1.from_environment(
            loaded, environment
        )
        controller.reset()
        untimed = benchmark._generic_arm_result(
            arm="PPO",
            environment=environment,
            controller=controller,
            manifest_identity=prepared.manifest_identity,
            initial_vector=vector,
        )
        assert (
            untimed["deterministic_evidence_sha256"]
            == runs["PPO"]["deterministic_evidence_sha256"]
        )
        envelopes = {
            arm: authenticate(
                {
                    "schema": arms.SCHEMA,
                    "protocol": {"id": PROTOCOL_ID, "raw_sha256": PROTOCOL_SHA256},
                    "analysis_contract_sha256": analysis_contract()["sha256"],
                    "stage": "fixture",
                    "seed": 9_000_091,
                    "actor": arms._actor(arm, config),
                    "training_task": config.to_dict(),
                    "checkpoint": (
                        benchmark._checkpoint_identity(loaded) if arm == "PPO" else None
                    ),
                    "source_provenance": source,
                    "service_mpc_settings": settings.to_dict(),
                    "manifest_identity": prepared.manifest_identity,
                    "audit_manifest": prepared.audit_manifest,
                    "run": arms._standalone_result(run),
                    "claim_boundary": arms.CLAIM_BOUNDARY,
                    "result_claim": None,
                }
            )
            for arm, run in runs.items()
        }
    finally:
        prepared.close()
    after = timing.capture_latency_host()
    assert source == benchmark._source_provenance(require_clean=False)
    reports, paths = {}, {}
    for arm, envelope in envelopes.items():
        parent = root / arm
        parent.mkdir()
        arm_path = parent / arms.FILENAME
        io.publish_json(arm_path, envelope)
        report = timing.synthetic_latency_evidence(
            arm_path, recorders[arm].to_dict(), before, after
        )
        path = parent / timing.FILENAME
        io.publish_json(path, report)
        paths[arm], reports[arm] = path, report
    return {"reports": reports, "paths": paths, "envelopes": envelopes}


@pytest.fixture(autouse=True)
def no_verification_execution(
    measured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("latency verification attempted plant execution or fresh timing")

    monkeypatch.setattr(PlantCyclicMarketEnvV2Beta, "__init__", forbidden)
    monkeypatch.setattr(PlantCyclicMarketEnvV2Beta, "reset", forbidden)
    monkeypatch.setattr(latency.LatencyRecorder, "warm_policy", forbidden)
    monkeypatch.setattr(timing, "capture_latency_host", forbidden)


@pytest.mark.parametrize("arm", benchmark.BETA_CONTROLLER_ARM_ORDER)
def test_all_arms_reconstruct_timing_and_physics(
    measured: dict[str, Any], arm: str
) -> None:
    result = timing.verify_synthetic_latency_file(measured["paths"][arm])
    run = measured["envelopes"][arm]["run"]
    assert result["verification"] == "synthetic_latency_reconstructed"
    assert result["source_evidence_verified"] is True
    assert (
        result["registered_hardware_verified"]
        is result["registered_concurrency_verified"]
        is False
    )
    assert result["result_claim"] is None
    assert result["channels"]["dispatcher"]["sample_count"] == run["physical_steps"]
    assert set(result["channels"]) == {
        "controller",
        "inference",
        "end_to_end",
        "dispatcher",
        "solver",
    }
    if arm == "PPO":
        assert result["warmup_attempted_calls"] == 1000
        assert (
            result["channels"]["inference"]["sample_count"]
            == run["completed_decisions"]
        )
    else:
        assert result["channels"]["inference"]["p99"] is None


@pytest.mark.parametrize(
    "defect",
    ["input", "missing_decision", "truncated_duration", "arm", "missing_warmup"],
)
def test_timing_cannot_be_substituted_for_another_physical_call(
    measured: dict[str, Any], defect: str
) -> None:
    report = measured["reports"]["PPO"]
    record = deepcopy(report["measurement"])
    row = record["decisions"][0]
    if defect == "input":
        row["inferences"][0]["input_sha256"] = "0" * 64
    elif defect == "missing_decision":
        record["decisions"].pop()
    elif defect == "truncated_duration":
        row["finished_ns"] = row["controller_finished_ns"]
    elif defect == "arm":
        record["arm"] = "H"
    elif defect == "missing_warmup":
        record["warmup"]["attempted_calls"] = 0
    with pytest.raises(ValueError):
        timing.bind_latency_to_verified_run(
            authenticate(record), measured["envelopes"]["PPO"]["run"]
        )


@pytest.mark.parametrize(
    "defect",
    [
        "result",
        "runtime",
        "host_change",
        "arm_reference",
        "schema",
        "claim",
        "contract",
    ],
)
def test_detached_rejects_rehashed_report_mutations(
    measured: dict[str, Any], tmp_path: Path, defect: str
) -> None:
    report = deepcopy(measured["reports"]["PPO"])
    if defect == "result":
        report["result"]["channels"]["inference"]["p99"] = 0
    elif defect == "runtime":
        for key in ("host_before", "host_after"):
            report[key]["dependencies"]["numpy"] = "wrong"
            report[key] = authenticate(report[key])
    elif defect == "host_change":
        report["host_after"]["machine"]["cpu_brand"] = "different CPU"
        report["host_after"] = authenticate(report["host_after"])
    elif defect == "arm_reference":
        report["arm_file"]["path"] = str(measured["paths"]["H"].parent / arms.FILENAME)
    elif defect == "schema":
        report["schema"] = "production-latency"
    elif defect == "claim":
        report["result_claim"] = "registered-pass"
    elif defect == "contract":
        report["latency_contract_sha256"] = "0" * 64
    path = tmp_path.resolve() / timing.FILENAME
    io.publish_json(path, authenticate(report))
    with pytest.raises(ValueError):
        timing.verify_synthetic_latency_file(path)


@pytest.mark.parametrize("channel", ["dispatcher", "solver"])
def test_retained_component_timing_cannot_exceed_outer_call(
    measured: dict[str, Any], channel: str
) -> None:
    arm = "E-beta" if channel == "solver" else "PPO"
    run = deepcopy(measured["envelopes"][arm]["run"])
    record = measured["reports"][arm]["measurement"]
    if channel == "dispatcher":
        run["dispatcher_elapsed_ns"][0][0] = 10**30
    else:
        run["evidence"]["decisions"][0]["evidence"]["plan"]["stages"][0][
            "elapsed_ns"
        ] = (10**30)
    with pytest.raises(ValueError, match="duration exceeds"):
        timing.bind_latency_to_verified_run(record, run)


def test_identical_source_replacement_at_final_graph_check_is_rejected(
    measured: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = deepcopy(measured["reports"]["PPO"])
    source = tmp_path.resolve() / arms.FILENAME
    io.publish_json(source, measured["envelopes"]["PPO"])
    report = timing.synthetic_latency_evidence(
        source, report["measurement"], report["host_before"], report["host_after"]
    )
    target = tmp_path.resolve() / timing.FILENAME
    io.publish_json(target, report)
    original = io.read_regular
    reads = 0

    def swapped(path: Path) -> bytes:
        nonlocal reads
        raw = original(path)
        if path == target:
            reads += 1
            if reads == 3:
                replacement = tmp_path / "replacement.json"
                replacement.write_bytes(canonical_bytes(measured["envelopes"]["PPO"]))
                replacement.replace(source)
        return raw

    monkeypatch.setattr(io, "read_regular", swapped)
    with pytest.raises(ValueError, match="graph changed"):
        timing.verify_synthetic_latency_file(target)


@pytest.mark.parametrize(
    "duration_ns, passes", [(100_000_000, True), (100_000_001, False)]
)
def test_inference_gate_uses_inclusive_exact_integer_threshold(
    measured: dict[str, Any],
    duration_ns: int,
    passes: bool,
) -> None:
    # Explicit synthetic-clock arithmetic on an admitted run; it never becomes
    # registered hardware evidence and cannot be used as a production gate.
    run = measured["envelopes"]["PPO"]["run"]
    record = deepcopy(measured["reports"]["PPO"]["measurement"])
    for index, row in enumerate(record["decisions"]):
        start = index * 10**15
        inference = row["inferences"][0]
        row["started_ns"] = start
        inference["started_ns"] = start + 1
        inference["finished_ns"] = start + 1 + duration_ns
        row["controller_finished_ns"] = start + max(
            duration_ns + 2, run["controller_elapsed_ns"][index] + 2
        )
        row["finished_ns"] = (
            row["controller_finished_ns"] + sum(run["dispatcher_elapsed_ns"][index]) + 1
        )
    result = timing.bind_latency_to_verified_run(authenticate(record), run)
    assert result["observed_inference_p99_within_limit"] is passes
    assert result["channels"]["inference"]["p99"] == duration_ns
    assert result["registered_hardware_verified"] is False


def _failed_arm(root: Path, arm: str) -> dict[str, Any]:
    config = replace(
        _config(), notes="synthetic timed failure fixture; not trained"
    ).validate()
    loaded = _loaded(root, config)
    recorder = latency.LatencyRecorder(arm)
    source = benchmark._source_provenance(require_clean=False)
    host = timing.capture_latency_host()
    if arm == "PPO":

        class FailingPolicy:
            calls = 0

            def act(self, vector: Any) -> Any:
                self.calls += 1
                if self.calls > 1000:
                    raise RuntimeError("synthetic inference failure")
                return np.full(4, -1.0, dtype=np.float32)

        loaded = recorder.warm_policy(replace(loaded, policy=FailingPolicy()))
    settings = arms._settings(config)
    prepared = benchmark._prepare_cell(
        loaded,
        config,
        seed=9_000_092,
        service_settings=settings,
        economic_controller_factory=benchmark._real_beta_economic_controller,
    )
    try:
        controller = prepared.controllers[arm]
        if arm == "H":

            def failed(*_args: object, **_kwargs: object) -> object:
                raise RuntimeError("synthetic heuristic failure")

        with pytest.MonkeyPatch.context() as patch:
            if arm == "H":
                patch.setattr(controller, "compute_command", failed)
            run = benchmark._generic_arm_result(
                arm=arm,
                environment=prepared.environments[arm],
                controller=controller,
                manifest_identity=prepared.manifest_identity,
                initial_vector=prepared.policy_observations[arm],
                study_timing=recorder,
            )
        envelope = authenticate(
            {
                "schema": arms.SCHEMA,
                "protocol": {"id": PROTOCOL_ID, "raw_sha256": PROTOCOL_SHA256},
                "analysis_contract_sha256": analysis_contract()["sha256"],
                "stage": "fixture",
                "seed": 9_000_092,
                "actor": arms._actor(arm, config),
                "training_task": config.to_dict(),
                "checkpoint": (
                    benchmark._checkpoint_identity(loaded) if arm == "PPO" else None
                ),
                "source_provenance": source,
                "service_mpc_settings": settings.to_dict(),
                "manifest_identity": prepared.manifest_identity,
                "audit_manifest": prepared.audit_manifest,
                "run": arms._standalone_result(run),
                "claim_boundary": arms.CLAIM_BOUNDARY,
                "result_claim": None,
            }
        )
    finally:
        prepared.close()
    target = root / arms.FILENAME
    io.publish_json(target, envelope)
    return {
        "path": target,
        "measurement": recorder.to_dict(),
        "host_before": host,
        "host_after": timing.capture_latency_host(),
        "run": run,
    }


@pytest.fixture(scope="module")
def failed_measured(
    measured: dict[str, Any], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("study-timed-failures").resolve()
    result = {}
    for arm in ("H", "PPO"):
        parent = root / arm
        parent.mkdir()
        result[arm] = _failed_arm(parent, arm)
    return result


@pytest.mark.parametrize("arm", ["H", "PPO"])
def test_actual_controller_failure_retains_attempt_and_inference(
    failed_measured: dict[str, Any], arm: str
) -> None:
    record = failed_measured[arm]
    result = timing.synthetic_latency_evidence(
        record["path"],
        record["measurement"],
        record["host_before"],
        record["host_after"],
    )["result"]
    run = record["run"]
    assert run["completed_decisions"] == run["physical_steps"] == 0
    assert result["controller_failures"] == result["interval_failures"] == 1
    assert result["channels"]["controller"]["sample_count"] == 1
    assert result["channels"]["end_to_end"]["sample_count"] == 1
    assert result["channels"]["dispatcher"]["p99"] is None
    assert result["inference_failures"] == int(arm == "PPO")

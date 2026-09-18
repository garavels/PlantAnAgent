"""Independent timing arithmetic and one-use warm-up/call custody fixtures."""

from __future__ import annotations

from copy import deepcopy
import itertools
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from plant_backend.rl import checkpoints
from plant_backend.rl.policies import FixedActionPolicy, LoadedPolicy
from plant_backend.rl.v2.study import latency
from plant_backend.rl.v2.study import latency_evidence as evidence
from plant_backend.rl.v2.training.custody_json import authenticate, decode


def _loaded(policy: Any = None) -> LoadedPolicy:
    return LoadedPolicy(
        policy=policy if policy is not None else FixedActionPolicy([-1.0] * 4),
        metadata=checkpoints.CheckpointMetadata(
            format=checkpoints.FORMAT_FIXED, algorithm="timing-fixture"
        ),
        path=Path("/private/tmp/unused-latency-fixture"),
    )


@pytest.fixture
def completed(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    recorder = latency.LatencyRecorder("PPO")
    original = _loaded()
    timed = recorder.warm_policy(original)
    assert timed.to_dict() == original.to_dict()
    monkeypatch.setattr(
        latency.time, "perf_counter_ns", itertools.count(10, 10).__next__
    )
    recorder.begin(0)
    timed.policy.act(np.zeros(135, dtype=np.float32))
    recorder.controller_finished()
    recorder.finish()
    return recorder.to_dict()


def test_actual_inference_is_nested_and_warmup_excluded(
    completed: dict[str, Any]
) -> None:
    result = latency.reconstruct_latency(completed)
    assert (
        result["warmup_attempted_calls"]
        == result["warmup_returned_calls"]
        == result["warmup_output_count"]
        == 1000
    )
    assert result["channels"]["inference"] == {
        "unit": "ns",
        "sample_count": 1,
        "minimum": 10,
        "maximum": 10,
        "total": 10,
        "p50": 10,
        "p95": 10,
        "p99": 10,
    }
    assert result["channels"]["controller"]["p99"] == 30
    assert result["channels"]["end_to_end"]["p99"] == 40
    assert (
        result["source_evidence_verified"]
        is result["registered_hardware_verified"]
        is False
    )
    assert result["result_claim"] is None


@pytest.mark.parametrize("size", [1, 2, 3, 99, 100, 101, 201, 1001])
def test_nearest_rank_uses_integer_indices_and_all_samples(size: int) -> None:
    values = list(reversed(range(1, size + 1)))
    result = latency.latency_summary(values)
    assert result["sample_count"] == size
    for p in (50, 95, 99):
        # Count integers at or below each candidate until the empirical CDF
        # reaches p%; no NumPy quantile or implementation index helper.
        expected = next(
            value for value in range(1, size + 1) if 100 * value >= p * size
        )
        assert result[f"p{p}"] == expected
    assert result["total"] == size * (size + 1) // 2


def test_empty_channel_is_null_and_large_integer_clocks_remain_exact() -> None:
    empty = latency.latency_summary([])
    assert empty["sample_count"] == 0 and empty["p99"] is None
    assert latency.latency_summary([2**60, 2**60 + 1])["p99"] == 2**60 + 1


@pytest.mark.parametrize(
    "bad", [[True], [-1], [1.0], ["1"], [float("nan")], "123", {"a": 1}]
)
def test_noninteger_or_unordered_timing_rejected(bad: Any) -> None:
    with pytest.raises(ValueError):
        latency.latency_summary(bad)


@pytest.mark.parametrize("kind", ["raised", "nondeterministic", "nonfinite", "shape"])
def test_failed_warmup_keeps_attempt_and_forbids_retry(kind: str) -> None:
    class Policy:
        calls = 0

        def act(self, vector: Any) -> Any:
            assert (
                vector.dtype == np.float32
                and vector.shape == (135,)
                and not vector.any()
            )
            self.calls += 1
            if kind == "raised":
                raise RuntimeError("fixture warm-up failure")
            if kind == "shape":
                return np.zeros(3, dtype=np.float32)
            if kind == "nonfinite":
                return np.full(4, np.nan, dtype=np.float32)
            return np.full(4, self.calls, dtype=np.float32)

    policy = Policy()
    recorder = latency.LatencyRecorder("PPO")
    with pytest.raises((ValueError, RuntimeError)):
        recorder.warm_policy(_loaded(policy))
    before = policy.calls
    with pytest.raises(ValueError, match="exclusive"):
        recorder.warm_policy(_loaded(policy))
    with pytest.raises(ValueError, match="completed warm-up"):
        recorder.begin(0)
    assert policy.calls == before
    result = latency.reconstruct_latency(recorder.to_dict())
    assert result["warmup_status"] == "failed"
    assert result["warmup_attempted_calls"] == (2 if kind == "nondeterministic" else 1)
    assert result["channels"]["inference"]["sample_count"] == 0


def test_failed_inference_is_timed_and_cannot_be_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Policy:
        calls = 0

        def act(self, vector: Any) -> Any:
            self.calls += 1
            if self.calls > 1000:
                raise RuntimeError("fixture inference failure")
            return np.zeros(4, dtype=np.float32)

    recorder = latency.LatencyRecorder("PPO")
    policy = Policy()
    timed = recorder.warm_policy(_loaded(policy))
    monkeypatch.setattr(
        latency.time, "perf_counter_ns", itertools.count(10, 10).__next__
    )
    recorder.begin(0)
    try:
        timed.policy.act(np.zeros(135, dtype=np.float32))
    except RuntimeError as error:
        recorder.abort(error)
    with pytest.raises(ValueError):
        recorder.begin(1)
    result = latency.reconstruct_latency(recorder.to_dict())
    assert policy.calls == 1001
    assert (
        result["inference_failures"]
        == result["controller_failures"]
        == result["interval_failures"]
        == 1
    )
    assert result["channels"]["inference"]["p99"] == 10
    assert result["channels"]["end_to_end"]["sample_count"] == 1


@pytest.mark.parametrize(
    "defect",
    [
        "schema",
        "contract",
        "warm_count",
        "warm_returned",
        "warm_outputs",
        "warm_input",
        "warm_status",
        "warm_output",
        "index",
        "start",
        "controller_end",
        "end",
        "missing_inference",
        "extra_inference",
        "inference_start",
        "inference_end",
        "inference_input",
        "failure",
        "return_flag",
        "infer_failure",
        "baseline",
        "extra_field",
    ],
)
def test_rehashed_timing_mutations_are_rejected(
    completed: dict[str, Any], defect: str
) -> None:
    raw = deepcopy(completed)
    row = raw["decisions"][0]
    warm = raw["warmup"]
    inference = row["inferences"][0]
    if defect == "schema":
        raw["schema"] = "other"
    elif defect == "contract":
        raw["latency_contract_sha256"] = "0" * 64
    elif defect == "warm_count":
        warm["attempted_calls"] = 999
    elif defect == "warm_returned":
        warm["returned_calls"] = 999
    elif defect == "warm_outputs":
        warm["action_sha256"].pop()
    elif defect == "warm_input":
        warm["input_sha256"] = "0" * 64
    elif defect == "warm_status":
        warm["status"] = "not_started"
    elif defect == "warm_output":
        warm["action_sha256"][-1] = "0" * 64
    elif defect == "index":
        row["decision_index"] = 1
    elif defect == "start":
        row["started_ns"] = True
    elif defect == "controller_end":
        row["controller_finished_ns"] = 9
    elif defect == "end":
        row["finished_ns"] = 39
    elif defect == "missing_inference":
        row["inferences"] = []
    elif defect == "extra_inference":
        row["inferences"].append(dict(inference))
    elif defect == "inference_start":
        inference["started_ns"] = 9
    elif defect == "inference_end":
        inference["finished_ns"] = 41
    elif defect == "inference_input":
        inference["input_sha256"] = "not-a-digest"
    elif defect == "failure":
        row["failure_type"] = "RuntimeError"
    elif defect == "return_flag":
        row["controller_returned"] = 1
    elif defect == "infer_failure":
        inference.update(returned=False, failure_type="RuntimeError")
    elif defect == "baseline":
        raw["arm"] = "H"
    elif defect == "extra_field":
        row["filtered_outliers"] = []
    with pytest.raises(ValueError):
        latency.reconstruct_latency(authenticate(raw))


def test_open_timing_cannot_be_published_or_double_closed() -> None:
    recorder = latency.LatencyRecorder("H")
    recorder.begin(0)
    with pytest.raises(ValueError):
        recorder.to_dict()
    with pytest.raises(ValueError):
        recorder.finish()
    recorder.controller_finished()
    with pytest.raises(ValueError):
        recorder.controller_finished()
    recorder.finish()
    with pytest.raises(ValueError):
        recorder.finish()


def test_fresh_child_registers_single_thread_host_without_plant_or_training() -> None:
    script = """
import json
from plant_backend.rl.v2.study.latency_evidence import configure_latency_threads, capture_latency_host, validate_latency_host
configure_latency_threads()
host = capture_latency_host()
validate_latency_host(host)
print(json.dumps(host))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    host = decode(completed.stdout)
    assert (
        host["machine"]["torch_intraop_threads"]
        == host["machine"]["torch_interop_threads"]
        == 1
    )
    assert host["machine"]["cpu_brand"]


@pytest.mark.parametrize(
    "defect", ["cpu", "clock", "dependencies", "version", "thread_type", "schema"]
)
def test_rehashed_host_contract_mutations_rejected(defect: str) -> None:
    host = evidence.capture_latency_host()
    if defect == "cpu":
        host["machine"]["cpu_brand"] = None
    elif defect == "clock":
        host["machine"]["clock"]["adjustable"] = True
    elif defect == "dependencies":
        del host["dependencies"]["torch"]
    elif defect == "version":
        host["python"] = ""
    elif defect == "thread_type":
        host["machine"]["torch_intraop_threads"] = True
    elif defect == "schema":
        host["schema"] = "other"
    with pytest.raises((ValueError, TypeError)):
        evidence.validate_latency_host(authenticate(host))


@pytest.mark.parametrize("arm", ["M24", "E-beta"])
def test_failed_solver_stages_remain_in_latency_samples(
    arm: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        latency.time, "perf_counter_ns", itertools.count(10, 10).__next__
    )
    recorder = latency.LatencyRecorder(arm)
    recorder.begin(0)
    recorder.abort(RuntimeError("synthetic solver failure"))
    stage = {"elapsed_ns": 10}
    run = {
        "arm": arm,
        "status": "execution_failed",
        "completed_decisions": 0,
        "physical_steps": 0,
        "failure": {"exception_type": "RuntimeError", "solver_stages": [stage]},
        "evidence": {
            "boundary_ledger": {"decision_boundaries": [{"decision_index": 0}]},
            "decisions": [],
            "run": {"events": [{"stages": [stage]}], "dispatcher_elapsed_ns": []},
        },
        "controller_elapsed_ns": [],
        "dispatcher_elapsed_ns": [],
    }
    # Explicit arithmetic fixture; source evidence is not admitted by this seam.
    result = evidence.bind_latency_to_verified_run(recorder.to_dict(), run)
    assert result["channels"]["solver"]["sample_count"] == 1
    assert result["channels"]["solver"]["p99"] == 10
    assert result["controller_failures"] == 1
    assert result["source_evidence_verified"] is False

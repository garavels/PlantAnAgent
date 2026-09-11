"""Complete call timing for the fixed study runners; no execution authority."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...policies import LoadedPolicy, Policy
from ..mpc_schema import domain_sha256
from ..training.custody_json import authenticate, canonical_bytes
from .contract import PREFIX, PROTOCOL_ID, PROTOCOL_SHA256

CONTRACT_SCHEMA = PREFIX + "latency-contract-v1"
RECORD_SCHEMA = PREFIX + "latency-record-v1"
WARMUP_CALLS = 1000
INFERENCE_LIMIT_NS = 100_000_000
ARMS = ("H", "M24", "E-beta", "PPO")


def latency_contract() -> dict[str, Any]:
    return authenticate(
        {
            "schema": CONTRACT_SCHEMA,
            "protocol": {"id": PROTOCOL_ID, "raw_sha256": PROTOCOL_SHA256},
            "clock": "time.perf_counter_ns",
            "warmup": {
                "calls": WARMUP_CALLS,
                "input": "135 float32 zeros",
                "deterministic": True,
                "scope": "once per fresh PPO arm process",
                "retry": False,
            },
            "inference": "actual policy.act call; retain failed calls and exact normalized input digest",
            "controller": "complete command-producing call; retain failed attempts",
            "end_to_end": "controller start through simulated interval return and M24 transition reconciliation; excludes publication and detached verification",
            "dispatcher": "each executed minute; original physical evidence timing",
            "solver": "each retained solver stage, including failed stages",
            "percentiles": {
                "p50": 50,
                "p95": 95,
                "p99": 99,
                "method": "nearest rank ceil(p*n/100)",
                "empty": None,
            },
            "inference_p99_limit_ns": INFERENCE_LIMIT_NS,
            "host": {
                "torch_intraop_threads": 1,
                "torch_interop_threads": 1,
                "concurrent_benchmark_jobs": 0,
                "identity": "CPU, OS, dependencies and clock bound before and after",
            },
            "selection_role": "none",
            "claim_boundary": "timing evidence only; stage and physical evidence admission required",
        }
    )


def _integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _digest(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("latency vector digest differs")
    return value


def vector_digest(vector: NDArray[np.generic], size: int) -> str:
    raw = np.asarray(vector)
    if (
        raw.shape != (size,)
        or not np.issubdtype(raw.dtype, np.floating)
        or not np.isfinite(raw).all()
    ):
        raise ValueError("timing vector must have the finite admitted interface")
    return domain_sha256(
        (
            "rl-policy-observation-vector-v1"
            if size == 135
            else PREFIX + "warmup-action-v1"
        ),
        [float(value) for value in raw],
    )


def latency_summary(raw: Sequence[int]) -> dict[str, Any]:
    if isinstance(raw, (str, bytes, Mapping)) or not isinstance(raw, Sequence):
        raise ValueError("latency samples must be an ordered sequence")
    values = sorted(_integer(value, "latency sample") for value in raw)
    count = len(values)
    return {
        "unit": "ns",
        "sample_count": count,
        "minimum": values[0] if count else None,
        "maximum": values[-1] if count else None,
        "total": sum(values),
        **{
            f"p{p}": values[(p * count + 99) // 100 - 1] if count else None
            for p in (50, 95, 99)
        },
    }


class LatencyRecorder:
    """One arm's ordered timing ledger. Callbacks cannot grant custody admission."""

    def __init__(self, arm: str) -> None:
        if arm not in ARMS:
            raise ValueError("unknown latency arm")
        self.arm = arm
        self._warmup: dict[str, Any] = {
            "status": "not_started" if arm == "PPO" else "not_applicable",
            "input_sha256": (
                vector_digest(np.zeros(135, dtype=np.float32), 135)
                if arm == "PPO"
                else None
            ),
            "action_sha256": [],
            "attempted_calls": 0,
            "returned_calls": 0,
            "failure_type": None,
        }
        self._warmup_consumed = False
        self._rows: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None
        self._inference_active = False

    def warm_policy(self, loaded: LoadedPolicy) -> LoadedPolicy:
        if (
            self.arm != "PPO"
            or self._warmup_consumed
            or self._rows
            or self._current is not None
        ):
            raise ValueError("policy warm-up is exclusive and precedes all timed calls")
        self._warmup_consumed = True
        vector = np.zeros(135, dtype=np.float32)
        try:
            for _ in range(WARMUP_CALLS):
                self._warmup["attempted_calls"] += 1
                action = loaded.policy.act(vector.copy())
                self._warmup["returned_calls"] += 1
                digest = vector_digest(action, 4)
                self._warmup["action_sha256"].append(digest)
                if digest != self._warmup["action_sha256"][0]:
                    raise ValueError("warm-up policy output is not deterministic")
        except BaseException as error:
            self._warmup["status"] = "failed"
            self._warmup["failure_type"] = type(error).__name__
            raise
        self._warmup["status"] = "complete"
        return replace(loaded, policy=_TimedPolicy(loaded.policy, self))

    def begin(self, decision_index: int) -> None:
        if (
            self._current is not None
            or _integer(decision_index, "decision index") != len(self._rows)
            or (self._rows and not self._rows[-1]["interval_returned"])
        ):
            raise ValueError("timing decisions must be exclusive and contiguous")
        if self.arm == "PPO" and self._warmup["status"] != "complete":
            raise ValueError("PPO timing requires completed warm-up")
        self._current = {
            "decision_index": decision_index,
            "started_ns": time.perf_counter_ns(),
            "controller_finished_ns": None,
            "finished_ns": None,
            "controller_returned": False,
            "interval_returned": False,
            "failure_type": None,
            "inferences": [],
        }

    def controller_finished(self) -> None:
        row = self._require_current()
        if row["controller_finished_ns"] is not None or self._inference_active:
            raise ValueError(
                "controller timing is already closed or inference is active"
            )
        row["controller_finished_ns"] = time.perf_counter_ns()
        row["controller_returned"] = True

    def finish(self) -> None:
        row = self._require_current()
        if row["controller_finished_ns"] is None or self._inference_active:
            raise ValueError("interval timing cannot precede controller completion")
        row["finished_ns"] = time.perf_counter_ns()
        row["interval_returned"] = True
        self._rows.append(row)
        self._current = None

    def abort(self, error: BaseException) -> None:
        if self._current is None:
            return
        if self._inference_active:
            raise ValueError(
                "inference must retain its failure before arm timing closes"
            )
        row = self._current
        now = time.perf_counter_ns()
        if row["controller_finished_ns"] is None:
            row["controller_finished_ns"] = now
        row["finished_ns"] = now
        row["failure_type"] = type(error).__name__
        self._rows.append(row)
        self._current = None

    def _require_current(self) -> dict[str, Any]:
        if self._current is None:
            raise ValueError("no active timing decision")
        return self._current

    def _infer(
        self, policy: Policy, vector: NDArray[np.generic]
    ) -> NDArray[np.float32]:
        row = self._require_current()
        if (
            self.arm != "PPO"
            or row["controller_finished_ns"] is not None
            or row["inferences"]
            or self._inference_active
        ):
            raise ValueError("one inference must be nested in each PPO controller call")
        observation_digest = vector_digest(vector, 135)
        self._inference_active = True
        started = time.perf_counter_ns()
        failure = None
        try:
            return policy.act(vector)
        except BaseException as error:
            failure = type(error).__name__
            raise
        finally:
            finished = time.perf_counter_ns()
            self._inference_active = False
            row["inferences"].append(
                {
                    "input_sha256": observation_digest,
                    "started_ns": started,
                    "finished_ns": finished,
                    "returned": failure is None,
                    "failure_type": failure,
                }
            )

    def to_dict(self) -> dict[str, Any]:
        if self._current is not None or self._inference_active:
            raise ValueError("cannot publish an open timing decision")
        result = authenticate(
            {
                "schema": RECORD_SCHEMA,
                "latency_contract_sha256": latency_contract()["sha256"],
                "arm": self.arm,
                "warmup": deepcopy(self._warmup),
                "decisions": deepcopy(self._rows),
            }
        )
        reconstruct_latency(result)
        return result


class _TimedPolicy:
    def __init__(self, policy: Policy, recorder: LatencyRecorder) -> None:
        self._policy = policy
        self._recorder = recorder

    def act(self, observation: NDArray[np.generic]) -> NDArray[np.float32]:
        return self._recorder._infer(self._policy, observation)


def _exact_fields(value: object, names: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError(f"{label} fields differ")
    return value


def _failure(value: object) -> str | None:
    if value is not None and (
        not isinstance(value, str) or not value or not value.isidentifier()
    ):
        raise ValueError("failure type must be a nonempty identifier or null")
    return value


def reconstruct_latency(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Check raw call order/nesting and compute all-call timing; no host admission."""
    record = _exact_fields(
        dict(raw),
        {"schema", "latency_contract_sha256", "arm", "warmup", "decisions", "sha256"},
        "latency record",
    )
    if (
        record["schema"] != RECORD_SCHEMA
        or record["latency_contract_sha256"] != latency_contract()["sha256"]
        or canonical_bytes(record) != canonical_bytes(authenticate(record))
        or record["arm"] not in ARMS
    ):
        raise ValueError("latency record identity differs")
    warm = _exact_fields(
        record["warmup"],
        {
            "status",
            "input_sha256",
            "action_sha256",
            "attempted_calls",
            "returned_calls",
            "failure_type",
        },
        "warm-up",
    )
    actions = warm["action_sha256"]
    if not isinstance(actions, list) or len(actions) > WARMUP_CALLS:
        raise ValueError("warm-up action count differs")
    for digest in actions:
        _digest(digest)
    attempted = _integer(warm["attempted_calls"], "warm-up attempted calls")
    returned = _integer(warm["returned_calls"], "warm-up returned calls")
    if (
        not 0 <= len(actions) <= returned <= attempted <= WARMUP_CALLS
        or attempted - returned > 1
        or returned - len(actions) > 1
    ):
        raise ValueError("warm-up call/output counts differ")
    failure = _failure(warm["failure_type"])
    if record["arm"] != "PPO":
        if warm != {
            "status": "not_applicable",
            "input_sha256": None,
            "action_sha256": [],
            "attempted_calls": 0,
            "returned_calls": 0,
            "failure_type": None,
        }:
            raise ValueError("baseline cannot declare policy warm-up")
    else:
        if warm["input_sha256"] != vector_digest(np.zeros(135, dtype=np.float32), 135):
            raise ValueError("warm-up input differs from the fixed vector")
        if warm["status"] == "complete":
            if (
                failure is not None
                or attempted != returned
                or len(actions) != WARMUP_CALLS
                or len(set(actions)) != 1
            ):
                raise ValueError("policy warm-up is incomplete or nondeterministic")
        elif warm["status"] == "not_started":
            if failure is not None or actions or attempted:
                raise ValueError("unstarted policy warm-up retains outcomes")
        elif warm["status"] != "failed" or failure is None or attempted == 0:
            raise ValueError("policy warm-up status differs")
    rows = record["decisions"]
    if not isinstance(rows, list):
        raise ValueError("timing decisions must be an array")
    if record["arm"] == "PPO" and rows and warm["status"] != "complete":
        raise ValueError("timed policy calls precede warm-up completion")
    channels: dict[str, list[int]] = {
        key: [] for key in ("inference", "controller", "end_to_end")
    }
    counts = {"controller_failures": 0, "interval_failures": 0, "inference_failures": 0}
    previous_end = 0
    failed = False
    for index, raw_row in enumerate(rows):
        row = _exact_fields(
            raw_row,
            {
                "decision_index",
                "started_ns",
                "controller_finished_ns",
                "finished_ns",
                "controller_returned",
                "interval_returned",
                "failure_type",
                "inferences",
            },
            "timing decision",
        )
        if _integer(row["decision_index"], "timing index") != index or failed:
            raise ValueError(
                "timing decision sequence is discontinuous or retried after failure"
            )
        start, controller_end, end = (
            _integer(row[key], key)
            for key in ("started_ns", "controller_finished_ns", "finished_ns")
        )
        if not previous_end <= start <= controller_end <= end:
            raise ValueError("timing timestamps are not ordered")
        previous_end = end
        for key in ("controller_returned", "interval_returned"):
            if type(row[key]) is not bool:
                raise ValueError("timing outcomes must be booleans")
        failure = _failure(row["failure_type"])
        if row["interval_returned"] and (
            not row["controller_returned"] or failure is not None
        ):
            raise ValueError("successful interval has failed controller timing")
        if not row["interval_returned"] and failure is None:
            raise ValueError("failed interval omits its failure")
        failed = not row["interval_returned"]
        counts["controller_failures"] += int(not row["controller_returned"])
        counts["interval_failures"] += int(failed)
        channels["controller"].append(controller_end - start)
        channels["end_to_end"].append(end - start)
        inferences = row["inferences"]
        if not isinstance(inferences, list) or len(inferences) > 1:
            raise ValueError("timing decision has an invalid inference count")
        if record["arm"] != "PPO" and inferences:
            raise ValueError("baseline timing contains policy inference")
        if (
            record["arm"] == "PPO"
            and row["controller_returned"]
            and len(inferences) != 1
        ):
            raise ValueError("successful PPO command omitted its inference")
        for raw_inference in inferences:
            inference = _exact_fields(
                raw_inference,
                {
                    "input_sha256",
                    "started_ns",
                    "finished_ns",
                    "returned",
                    "failure_type",
                },
                "inference timing",
            )
            _digest(inference["input_sha256"])
            call_start, call_end = (
                _integer(inference[key], key) for key in ("started_ns", "finished_ns")
            )
            if not start <= call_start <= call_end <= controller_end:
                raise ValueError("inference timing lies outside its controller call")
            call_failure = _failure(inference["failure_type"])
            if type(inference["returned"]) is not bool or inference["returned"] != (
                call_failure is None
            ):
                raise ValueError("inference failure declaration differs")
            if call_failure is not None and (
                row["controller_returned"] or failure != call_failure
            ):
                raise ValueError("inference failure was hidden by the controller")
            counts["inference_failures"] += int(call_failure is not None)
            channels["inference"].append(call_end - call_start)
    return {
        "verification": "latency_reconstructed",
        "source_evidence_verified": False,
        "registered_hardware_verified": False,
        "result_claim": None,
        "raw_sha256": record["sha256"],
        "arm": record["arm"],
        "warmup_status": warm["status"],
        "warmup_attempted_calls": attempted,
        "warmup_returned_calls": returned,
        "warmup_output_count": len(actions),
        "decision_count": len(rows),
        **counts,
        "channels": {key: latency_summary(values) for key, values in channels.items()},
    }

"""Plant-specific runtime primitives for the frozen feasible planner.

This privileged module is intentionally separate from causal controllers.  It owns
exact-environment lifetime and resource accounting, while the generic search module
owns ordering and pruning.  Full rollout behavior is layered onto these primitives in
subsequent audited slices.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields
import hashlib
import math
from numbers import Integral, Real
import platform
import resource
import struct
import time

from plant_sim import PlantConfig, PlantState

from ._planner_search import PlannerSearchCountersV1
from .env import PlantEnvV2
from .metrics import EpisodeMetricsV2
from .planner_schema import (
    A0_CONTROLLER_SHA256,
    FAILURE_CODES,
    FAILURE_PHASES,
    ActionRecordV1,
    CanonicalPayloadV1,
    FinalScoreV1,
    PlannerSettingsV1,
    action_record_trace_payload,
    canonical_json,
    outcome_payload,
    request_trace_payload,
    require_sha256,
    source_trace_payload,
)


ClockNs = Callable[[], int]
PeakRssRaw = Callable[[], int | float]
PlatformSystem = Callable[[], str]


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _finite(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}")
    return result


def _bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _payload(name: str, value: object, *, domain: str) -> CanonicalPayloadV1:
    if not isinstance(value, CanonicalPayloadV1):
        raise TypeError(f"{name} must be a CanonicalPayloadV1")
    checked = CanonicalPayloadV1(value.domain, value.canonical_json, value.sha256)
    if checked.domain != domain:
        raise ValueError(f"{name} must use domain {domain!r}")
    return checked


def _binary64_equal(left: object, right: object) -> bool:
    try:
        return struct.pack(">d", float(left)) == struct.pack(  # type: ignore[arg-type]
            ">d", float(right)  # type: ignore[arg-type]
        )
    except (TypeError, ValueError, OverflowError):
        return False


class PlannerRuntimeError(RuntimeError):
    """Stable plant-runtime failure translated by the future core orchestrator."""

    def __init__(
        self,
        *,
        phase: str,
        code: str,
        message: str,
        observed_resource_value: int | None = None,
    ) -> None:
        if phase not in FAILURE_PHASES:
            raise ValueError(f"unsupported planner failure phase: {phase}")
        if code not in FAILURE_CODES:
            raise ValueError(f"unsupported planner failure code: {code}")
        if not isinstance(message, str) or not message:
            raise ValueError("planner runtime failure message must be non-empty")
        if observed_resource_value is not None:
            observed_resource_value = _integer(
                "observed_resource_value", observed_resource_value
            )
        super().__init__(message)
        self.phase = phase
        self.code = code
        self.observed_resource_value = observed_resource_value


@dataclass(frozen=True, slots=True)
class TerminalSnapshotV1:
    """Detached terminal outcome; it never retains a live environment object."""

    manifest_sha256: str
    exogenous_episode_sha256: str
    checkpoint_contract_digest: str
    plant_config_sha256: str
    action_records: tuple[ActionRecordV1, ...]
    request_trace: CanonicalPayloadV1
    source_trace: CanonicalPayloadV1
    action_record_trace: CanonicalPayloadV1
    metrics: CanonicalPayloadV1
    terminal_truth: CanonicalPayloadV1
    terminal_telemetry: CanonicalPayloadV1
    delivery_audit: CanonicalPayloadV1
    outcome: CanonicalPayloadV1
    final_score: FinalScoreV1
    decision_count: int
    physics_steps: int
    terminated: bool
    truncated: bool
    battery_throughput_kwh: float
    h2_kg_per_kg_ch4: float
    co2_kg_per_kg_ch4: float

    def __post_init__(self) -> None:
        records = tuple(self.action_records)
        object.__setattr__(self, "action_records", records)
        object.__setattr__(
            self, "decision_count", _integer("decision_count", self.decision_count)
        )
        object.__setattr__(
            self, "physics_steps", _integer("physics_steps", self.physics_steps)
        )
        object.__setattr__(
            self,
            "battery_throughput_kwh",
            _finite(
                "battery_throughput_kwh",
                self.battery_throughput_kwh,
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "h2_kg_per_kg_ch4",
            _finite("h2_kg_per_kg_ch4", self.h2_kg_per_kg_ch4, minimum=0.0),
        )
        object.__setattr__(
            self,
            "co2_kg_per_kg_ch4",
            _finite("co2_kg_per_kg_ch4", self.co2_kg_per_kg_ch4, minimum=0.0),
        )
        self._validate()

    def validate(self) -> None:
        """Re-run every invariant so illicit mutation cannot survive later use."""

        self._validate()

    def _validate(self) -> None:
        for name in (
            "manifest_sha256",
            "exogenous_episode_sha256",
            "checkpoint_contract_digest",
            "plant_config_sha256",
        ):
            require_sha256(name, getattr(self, name))
        domains = (
            ("request_trace", "planner-request-trace-v1"),
            ("source_trace", "planner-source-trace-v1"),
            ("action_record_trace", "planner-action-record-trace-v1"),
            ("metrics", "planner-metrics-v1"),
            ("terminal_truth", "planner-terminal-truth-v1"),
            ("terminal_telemetry", "planner-terminal-telemetry-v1"),
            ("delivery_audit", "planner-delivery-audit-v1"),
            ("outcome", "planner-outcome-v1"),
        )
        for name, domain in domains:
            _payload(name, getattr(self, name), domain=domain)
        if not isinstance(self.final_score, FinalScoreV1):
            raise TypeError("final_score must be a FinalScoreV1")
        checked_score = FinalScoreV1(**self.final_score.to_dict())
        if canonical_json(self.final_score.to_dict()) != canonical_json(
            checked_score.to_dict()
        ):
            raise ValueError("final_score is not canonically normalized")
        decision_count = _integer("decision_count", self.decision_count)
        physics_steps = _integer("physics_steps", self.physics_steps)
        terminated = _bool("terminated", self.terminated)
        truncated = _bool("truncated", self.truncated)
        throughput = _finite(
            "battery_throughput_kwh",
            self.battery_throughput_kwh,
            minimum=0.0,
        )
        h2_ratio = _finite("h2_kg_per_kg_ch4", self.h2_kg_per_kg_ch4, minimum=0.0)
        co2_ratio = _finite("co2_kg_per_kg_ch4", self.co2_kg_per_kg_ch4, minimum=0.0)
        if h2_ratio == 0.0 or co2_ratio == 0.0:
            raise ValueError("terminal methane conversion ratios must be positive")
        if not terminated or truncated:
            raise ValueError("terminal snapshot must terminate without truncation")
        records = tuple(self.action_records)
        expected_request = request_trace_payload(records)
        expected_source = source_trace_payload(records)
        expected_action_trace = action_record_trace_payload(records)
        if (
            self.request_trace != expected_request
            or self.source_trace != expected_source
            or self.action_record_trace != expected_action_trace
        ):
            raise ValueError("terminal trace payloads disagree with action_records")
        if decision_count != len(records):
            raise ValueError("terminal decision count disagrees with action_records")
        if not _binary64_equal(self.final_score.battery_throughput_kwh, throughput):
            raise ValueError("terminal score throughput disagrees with snapshot")
        self._validate_score_sources(
            h2_ratio=h2_ratio,
            co2_ratio=co2_ratio,
            throughput=throughput,
        )
        expected_outcome = outcome_payload(
            manifest_sha256=self.manifest_sha256,
            exogenous_episode_sha256=self.exogenous_episode_sha256,
            checkpoint_contract_digest=self.checkpoint_contract_digest,
            request_trace=self.request_trace,
            metrics=self.metrics,
            terminal_truth=self.terminal_truth,
            terminal_telemetry=self.terminal_telemetry,
            delivery_audit=self.delivery_audit,
            battery_throughput_kwh=throughput,
            decision_count=decision_count,
            physics_steps=physics_steps,
            terminated=terminated,
            truncated=truncated,
        )
        if self.outcome != expected_outcome:
            raise ValueError("terminal outcome payload disagrees with its snapshot")

    def validate_for_config(self, config: PlantConfig) -> None:
        """Bind score conversion ratios to one exact plant configuration."""

        self.validate()
        if type(config) is not PlantConfig:
            raise TypeError("terminal snapshot config must be an exact PlantConfig")
        config_sha256 = hashlib.sha256(
            canonical_json(config.to_dict()).encode("utf-8")
        ).hexdigest()
        if config_sha256 != self.plant_config_sha256:
            raise ValueError("terminal snapshot plant configuration identity disagrees")
        methanator = config.methanator
        if not _binary64_equal(
            self.h2_kg_per_kg_ch4,
            methanator.h2_kg_per_kg_ch4,
        ) or not _binary64_equal(
            self.co2_kg_per_kg_ch4,
            methanator.co2_kg_per_kg_ch4,
        ):
            raise ValueError(
                "terminal methane conversion ratios disagree with plant configuration"
            )

    def _validate_score_sources(
        self,
        *,
        h2_ratio: float,
        co2_ratio: float,
        throughput: float,
    ) -> None:
        metrics = self.metrics.value()
        truth = self.terminal_truth.value()
        telemetry = self.terminal_telemetry.value()
        audit = self.delivery_audit.value()
        if not all(
            isinstance(value, dict) for value in (metrics, truth, telemetry, audit)
        ):
            raise ValueError("terminal score sources must all contain objects")
        assert isinstance(metrics, dict)
        assert isinstance(truth, dict)
        assert isinstance(telemetry, dict)
        assert isinstance(audit, dict)
        if set(metrics) != set(EpisodeMetricsV2().to_dict()):
            raise ValueError("terminal metrics payload has an incomplete field set")
        plant_fields = set(PlantState().to_dict())
        if set(truth) != plant_fields or set(telemetry) != plant_fields:
            raise ValueError("terminal plant payload has an incomplete field set")
        expected_audit_fields = {
            "frozen",
            "schedule",
            "settlements",
            "current",
            "cumulative_requested_kg",
            "cumulative_delivered_kg",
            "cumulative_unmet_kg",
            "pending_count",
            "next_pending",
        }
        if set(audit) != expected_audit_fields:
            raise ValueError("terminal delivery audit has an incomplete field set")

        score = self.final_score
        float_metrics = (
            (score.delivered_kg, metrics["delivered_kg"], "delivered_kg"),
            (score.worst_fill, metrics["worst_commitment_fill_rate"], "worst_fill"),
            (
                score.terminal_battery_energy_kwh,
                metrics["terminal_battery_energy_kwh"],
                "terminal_battery_energy_kwh",
            ),
            (score.curtailed_kwh, metrics["curtailed_kwh"], "curtailed_kwh"),
        )
        for score_value, source_value, name in float_metrics:
            if not _binary64_equal(score_value, source_value):
                raise ValueError(f"terminal score {name} disagrees with metrics")
        integer_metrics = (
            (score.service_numerator, "commitments_met"),
            (score.service_denominator, "commitments_settled"),
            (score.forced_stops, "electrolyser_forced_stops"),
            (score.aborted_starts, "electrolyser_aborted_starts"),
            (score.cold_starts, "electrolyser_cold_starts"),
        )
        for score_value, name in integer_metrics:
            if _integer(f"metrics.{name}", metrics[name]) != score_value:
                raise ValueError(f"terminal score {name} disagrees with metrics")
        truth_metrics = (
            ("h2_level_kg", "terminal_h2_kg"),
            ("co2_level_kg", "terminal_co2_kg"),
            ("ch4_level_kg", "terminal_ch4_kg"),
            ("battery_soc", "terminal_battery_soc"),
            ("battery_energy_kwh", "terminal_battery_energy_kwh"),
        )
        for truth_name, metric_name in truth_metrics:
            if not _binary64_equal(truth[truth_name], metrics[metric_name]):
                raise ValueError(f"terminal truth {truth_name} disagrees with metrics")
        equivalent = _finite("terminal truth ch4_level_kg", truth["ch4_level_kg"])
        equivalent += min(
            _finite("terminal truth h2_level_kg", truth["h2_level_kg"]) / h2_ratio,
            _finite("terminal truth co2_level_kg", truth["co2_level_kg"]) / co2_ratio,
        )
        if not _binary64_equal(score.terminal_ch4_equivalent_inventory_kg, equivalent):
            raise ValueError("terminal equivalent inventory disagrees with truth")
        if not _binary64_equal(audit["cumulative_delivered_kg"], score.delivered_kg):
            raise ValueError("terminal delivered score disagrees with delivery audit")
        if not _binary64_equal(score.battery_throughput_kwh, throughput):
            raise ValueError("terminal throughput score disagrees with accumulator")


@dataclass(frozen=True, slots=True)
class A0GenerationV1:
    """Detached A0 generation trace and every completed macro prefix identity."""

    action_records: tuple[ActionRecordV1, ...]
    terminal_snapshot: TerminalSnapshotV1
    prefix_state_bytes: tuple[bytes, ...]

    def __post_init__(self) -> None:
        records = tuple(self.action_records)
        request_trace = request_trace_payload(records)
        source_trace = source_trace_payload(records)
        action_trace = action_record_trace_payload(records)
        if any(
            record.source_phase != "a0_capture"
            or record.source_code != -1
            or record.source_binding_sha256 != A0_CONTROLLER_SHA256
            for record in records
        ):
            raise ValueError("A0 generation records must retain exact A0 lineage")
        if not isinstance(self.terminal_snapshot, TerminalSnapshotV1):
            raise TypeError("terminal_snapshot must be a TerminalSnapshotV1")
        snapshot = self.terminal_snapshot
        snapshot.validate()
        if (
            snapshot.request_trace != request_trace
            or snapshot.source_trace != source_trace
            or snapshot.action_record_trace != action_trace
        ):
            raise ValueError("A0 terminal snapshot does not match its action records")
        if snapshot.decision_count != len(records):
            raise ValueError("A0 decision count does not match its action trace")
        prefixes = tuple(self.prefix_state_bytes)
        expected_prefixes = (len(records) + 3) // 4
        if len(prefixes) != expected_prefixes:
            raise ValueError("A0 prefix count does not match four-decision macros")
        for index, prefix in enumerate(prefixes):
            if not isinstance(prefix, bytes) or not prefix:
                raise TypeError(f"prefix_state_bytes[{index}] must be non-empty bytes")
        object.__setattr__(self, "action_records", records)
        object.__setattr__(self, "prefix_state_bytes", prefixes)


@dataclass(eq=False, slots=True)
class PlantPlannerCandidateV1:
    """One explicitly owned mutable environment plus immutable search metadata."""

    env: PlantEnvV2 | None
    action_records: tuple[ActionRecordV1, ...]
    battery_throughput_kwh: float
    macro_depth: int
    terminated: bool
    truncated: bool
    _attestation: tuple[PlantEnvV2, str, bytes, int, bool, bool] | None = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        records, throughput, depth, terminated, truncated = self._validate_contents()
        object.__setattr__(self, "action_records", records)
        object.__setattr__(self, "battery_throughput_kwh", throughput)
        object.__setattr__(self, "macro_depth", depth)
        assert self.env is not None
        object.__setattr__(
            self,
            "_attestation",
            (
                self.env,
                action_record_trace_payload(records).canonical_json,
                struct.pack(">d", throughput),
                depth,
                terminated,
                truncated,
            ),
        )

    def validate(self) -> None:
        """Re-attest immutable candidate metadata before every port operation."""

        expected = self._attestation
        if expected is None:
            raise ValueError("candidate was already released")
        records, throughput, depth, terminated, truncated = self._validate_contents()
        env = self.env
        assert env is not None
        if (
            expected[0] is not env
            or expected[1] != action_record_trace_payload(records).canonical_json
            or expected[2] != struct.pack(">d", throughput)
            or expected[3:] != (depth, terminated, truncated)
        ):
            raise ValueError("candidate no longer matches its creation attestation")

    def mark_released(self) -> PlantEnvV2:
        """Invalidate this ownership handle and return its exact environment once."""

        self.validate()
        return self._dispose_for_cleanup()

    def _dispose_for_cleanup(self) -> PlantEnvV2:
        """Detach the originally attested environment without endorsing metadata.

        The plant runtime uses this only while unwinding an integrity failure.  It must
        still release the returned environment's independent resource-guard token and
        propagate the original validation failure.
        """

        expected = self._attestation
        if expected is None:
            raise ValueError("candidate was already released")
        env = expected[0]
        self.env = None
        self._attestation = None
        return env

    def _validate_contents(
        self,
    ) -> tuple[tuple[ActionRecordV1, ...], float, int, bool, bool]:
        if type(self.env) is not PlantEnvV2:
            raise TypeError("candidate env must be an exact PlantEnvV2")
        env = self.env
        if env._episode is None or env.control_state is None:
            raise ValueError("candidate environment must be reset")
        records = tuple(self.action_records)
        request_trace_payload(records)
        throughput = _finite(
            "battery_throughput_kwh",
            self.battery_throughput_kwh,
            minimum=0.0,
        )
        depth = _integer("macro_depth", self.macro_depth)
        terminated = _bool("terminated", self.terminated)
        truncated = _bool("truncated", self.truncated)
        if truncated:
            raise ValueError("planner candidates cannot retain truncated environments")
        if env.decisions_taken != len(records):
            raise ValueError("candidate trace length disagrees with environment")
        if not (
            env.physics_steps_taken
            == env.simulator.step_index
            == env.metrics.physical_steps
        ):
            raise ValueError("candidate physics counters disagree")
        if env.decisions_taken != env.metrics.decisions:
            raise ValueError("candidate decision counters disagree")
        if depth != (len(records) + 3) // 4:
            raise ValueError("candidate macro depth disagrees with its trace")
        if terminated != (env.physics_steps_taken >= env.episode_physics_steps):
            raise ValueError("candidate termination flag disagrees with environment")
        return records, throughput, depth, terminated, truncated


@dataclass(frozen=True, slots=True, eq=False)
class _EnvironmentReservationV1:
    owner: object
    serial: int
    phase: str


def _production_peak_rss_raw() -> int | float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


class _ResourceGuardV1:
    """Own exact resource sampling and every logical environment reference."""

    def __init__(
        self,
        *,
        settings: PlannerSettingsV1,
        counters: PlannerSearchCountersV1,
        start_ns: int,
        clock_ns: ClockNs = time.perf_counter_ns,
        peak_rss_raw: PeakRssRaw = _production_peak_rss_raw,
        platform_system: PlatformSystem = platform.system,
    ) -> None:
        if not isinstance(settings, PlannerSettingsV1):
            raise TypeError("settings must be a PlannerSettingsV1")
        if not isinstance(counters, PlannerSearchCountersV1):
            raise TypeError("counters must be PlannerSearchCountersV1")
        if (
            not callable(clock_ns)
            or not callable(peak_rss_raw)
            or not callable(platform_system)
        ):
            raise TypeError("resource readers must be callable")
        settings_values = {
            item.name: getattr(settings, item.name) for item in fields(settings)
        }
        checked_settings = PlannerSettingsV1(**settings_values)
        self._counters = counters
        self._start_ns = _integer("start_ns", start_ns)
        self._wall_limit_ns = checked_settings.wall_limit_ns
        self._rss_limit_bytes = checked_settings.rss_limit_bytes
        self._max_live_environments = checked_settings.max_live_environments
        self._clock_ns = clock_ns
        self._peak_rss_raw = peak_rss_raw
        self._platform_system = platform_system
        self._owner = object()
        self._next_serial = 0
        self._reservations: dict[
            int, tuple[_EnvironmentReservationV1, PlantEnvV2 | None]
        ] = {}
        self._environment_ids: dict[int, PlantEnvV2] = {}
        self._peak_rss_bytes = 0
        self._closed = False

    @property
    def live_environments(self) -> int:
        return len(self._environment_ids)

    @property
    def reserved_slots(self) -> int:
        return len(self._reservations)

    @property
    def peak_rss_bytes(self) -> int:
        return self._peak_rss_bytes

    def elapsed_ns(self) -> int:
        self._require_open()
        return self._elapsed_ns(phase="input_validation")

    def _elapsed_ns(self, *, phase: str) -> int:
        now = _integer("clock reading", self._clock_ns())
        if now < self._start_ns:
            raise PlannerRuntimeError(
                phase=phase,
                code="invalid_input",
                message="monotonic clock precedes the supplied worker start",
            )
        return now - self._start_ns

    def reserve_environment(self, *, phase: str) -> _EnvironmentReservationV1:
        self._require_open()
        self._validate_phase(phase)
        proposed = len(self._reservations) + 1
        beam_only = phase == "beam_search" and all(
            item[0].phase == "beam_search" for item in self._reservations.values()
        )
        phase_limit = self._max_live_environments if beam_only else 2
        limit = min(self._max_live_environments, phase_limit)
        if proposed > limit:
            raise PlannerRuntimeError(
                phase=phase,
                code="live_environment_limit",
                message="another environment would exceed the phase live limit",
                observed_resource_value=proposed,
            )
        token = _EnvironmentReservationV1(self._owner, self._next_serial, phase)
        self._next_serial += 1
        self._reservations[id(token)] = (token, None)
        return token

    def bind_environment(
        self, token: _EnvironmentReservationV1, env: PlantEnvV2
    ) -> None:
        self._require_open()
        current = self._lookup_token(token)
        if current is not None:
            raise PlannerRuntimeError(
                phase="input_validation",
                code="result_invariant_error",
                message="environment reservation is already bound",
            )
        if type(env) is not PlantEnvV2:
            raise TypeError("reserved environment must be an exact PlantEnvV2")
        existing = self._environment_ids.get(id(env))
        if existing is not None:
            raise PlannerRuntimeError(
                phase="input_validation",
                code="result_invariant_error",
                message="environment identity is already live",
            )
        try:
            self._reservations[id(token)] = (token, env)
            self._environment_ids[id(env)] = env
        except BaseException:
            registered = self._environment_ids.get(id(env))
            if registered is env:
                del self._environment_ids[id(env)]
            entry = self._reservations.get(id(token))
            if entry is not None and entry[0] is token:
                self._reservations[id(token)] = (token, None)
            raise
        self._counters.max_live_environments_observed = max(
            self._counters.max_live_environments_observed,
            len(self._environment_ids),
        )

    def cancel_reservation(self, token: _EnvironmentReservationV1) -> None:
        self._require_open()
        current = self._lookup_token(token)
        if current is not None:
            raise PlannerRuntimeError(
                phase="input_validation",
                code="result_invariant_error",
                message="a bound environment reservation cannot be cancelled",
            )
        del self._reservations[id(token)]

    def release_environment(
        self, token: _EnvironmentReservationV1, env: PlantEnvV2
    ) -> None:
        self._require_open()
        current = self._lookup_token(token)
        if current is None or current is not env:
            raise PlannerRuntimeError(
                phase="result_serialization",
                code="result_invariant_error",
                message="environment release does not match its reservation",
            )
        registered = self._environment_ids.get(id(env))
        if registered is not env:
            raise PlannerRuntimeError(
                phase="result_serialization",
                code="result_invariant_error",
                message="environment identity is not live",
            )
        del self._environment_ids[id(env)]
        del self._reservations[id(token)]

    def checkpoint(self, *, phase: str) -> tuple[int, int]:
        self._require_open()
        self._validate_phase(phase)
        elapsed = self._elapsed_ns(phase=phase)
        rss = self._read_peak_rss_bytes(phase=phase)
        self._peak_rss_bytes = max(self._peak_rss_bytes, rss)
        if elapsed > self._wall_limit_ns:
            raise PlannerRuntimeError(
                phase=phase,
                code="wall_time_limit",
                message="planner wall-time limit exceeded",
                observed_resource_value=elapsed,
            )
        if rss > self._rss_limit_bytes:
            raise PlannerRuntimeError(
                phase=phase,
                code="rss_limit",
                message="planner peak-RSS limit exceeded",
                observed_resource_value=rss,
            )
        return elapsed, rss

    def close(self) -> None:
        if self._closed:
            return
        leaked = len(self._reservations)
        self._environment_ids.clear()
        self._reservations.clear()
        self._closed = True
        if leaked:
            raise PlannerRuntimeError(
                phase="result_serialization",
                code="result_invariant_error",
                message=f"planner runtime leaked {leaked} environment reservations",
                observed_resource_value=leaked,
            )

    def _lookup_token(self, token: object) -> PlantEnvV2 | None:
        if not isinstance(token, _EnvironmentReservationV1):
            raise TypeError("environment reservation token has the wrong type")
        if token.owner is not self._owner:
            raise PlannerRuntimeError(
                phase="input_validation",
                code="result_invariant_error",
                message="environment reservation belongs to another runtime",
            )
        entry = self._reservations.get(id(token))
        if entry is None or entry[0] is not token:
            raise PlannerRuntimeError(
                phase="result_serialization",
                code="result_invariant_error",
                message="environment reservation is unknown or already released",
            )
        return entry[1]

    def _read_peak_rss_bytes(self, *, phase: str) -> int:
        try:
            raw_value = self._peak_rss_raw()
            system = self._platform_system()
            raw = _finite("peak RSS reading", raw_value, minimum=0.0)
            integer_raw = int(raw)
            if raw != float(integer_raw):
                raise ValueError("peak RSS reading must be integral")
            if not isinstance(system, str) or not system:
                raise ValueError("platform system must be a non-empty string")
            return integer_raw if system == "Darwin" else integer_raw * 1024
        except PlannerRuntimeError:
            raise
        except (OSError, TypeError, ValueError, OverflowError) as exc:
            raise PlannerRuntimeError(
                phase=phase,
                code="invalid_input",
                message="peak RSS is unavailable on this execution platform",
            ) from exc

    @staticmethod
    def _validate_phase(phase: str) -> None:
        if phase not in FAILURE_PHASES:
            raise ValueError(f"unsupported planner resource phase: {phase}")

    def _require_open(self) -> None:
        if self._closed:
            raise PlannerRuntimeError(
                phase="result_serialization",
                code="result_invariant_error",
                message="planner resource guard is closed",
            )


__all__ = [
    "A0GenerationV1",
    "PlantPlannerCandidateV1",
    "PlannerRuntimeError",
    "TerminalSnapshotV1",
]

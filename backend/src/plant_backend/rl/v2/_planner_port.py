"""Exact-plant bridge for the frozen feasible-planner search.

The generic search layer owns deterministic ordering and bounded retention.  This
module supplies its only privileged operations: fresh canonical resets, captured A0,
exact replay, structural-sharing clones, live action generation, and environment
ownership.  It deliberately remains private; the later public orchestrator translates
all exceptions into the frozen result schema and owns evidence publication.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import struct
from typing import Literal

from plant_sim import Action, PlantConfig

from ._planner_actions import generate_action_record, require_attested_binding
from ._planner_runtime import (
    A0GenerationV1,
    PlannerCandidateRegistryV1,
    PlannerRuntimeAdmissionV1,
    PlantPlannerCandidateV1,
    TerminalSnapshotV1,
    _EnvironmentReservationV1,
    _ResourceGuardV1,
    _validate_reset_environment,
    admit_planner_runtime,
    capture_terminal_snapshot,
    extract_intermediate_rank,
    step_action_record,
)
from ._planner_search import (
    IntermediateRankV1,
    PlannerSearchCountersV1,
    PlannerSearchPortV1,
    SearchTraceV1,
)
from ._planner_state import build_state_signature, clone_environment
from .config import EpisodeConfigV2
from .contract import describe_checkpoint_contract
from .env import PlantEnvV2
from .evaluation import LegacyHeuristicControllerV2
from .planner_schema import (
    ActionRecordV1,
    FinalScoreV1,
    action_record_trace_payload,
    canonical_json,
    raw_canonical_sha256,
    request_trace_payload,
)
from .reward import RewardWeightsV2

from ...controllers.base import Constraints, ControlDecision


CoordinatePhase = Literal["coarse_polish", "fine_polish"]


def _counter_values(counters: PlannerSearchCountersV1) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in fields(PlannerSearchCountersV1):
        value = getattr(counters, item.name)
        result[item.name] = _integer(f"counters.{item.name}", value)
    return result


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _same_binary64(left: float, right: float) -> bool:
    return struct.pack(">d", left) == struct.pack(">d", right)


@dataclass(frozen=True, slots=True)
class PlannerPlantInputsV1:
    """Exact static plant inputs cross-bound to one runtime admission."""

    config: PlantConfig
    episode_config: EpisodeConfigV2
    constraints: Constraints
    weights: RewardWeightsV2
    admission: PlannerRuntimeAdmissionV1

    def __post_init__(self) -> None:
        self._validate_types()
        if type(self.admission) is not PlannerRuntimeAdmissionV1:
            raise TypeError("admission must be an exact PlannerRuntimeAdmissionV1")
        self.admission.validate()
        rebuilt = admit_planner_runtime(
            config=self.config,
            episode_config=self.episode_config,
            constraints=self.constraints,
            weights=self.weights,
            settings=self.admission.settings,
            binding=self.admission.binding,
            seed=self.admission.seed,
            evidence_scope=self.admission.evidence_scope,
            canonical_episode_verified=self.admission.canonical_episode_verified,
        )
        if not _same_admission(self.admission, rebuilt):
            raise ValueError("planner plant inputs disagree with runtime admission")
        object.__setattr__(self, "admission", rebuilt)

    def validate(self) -> None:
        """Recheck current static references without sampling an episode."""

        self._validate_types()
        self.admission.validate()
        settings = self.admission.settings
        static = (
            (self.config.to_dict(), settings.plant_config_sha256, "plant config"),
            (
                self.episode_config.to_dict(),
                settings.episode_config_sha256,
                "episode config",
            ),
            (self.constraints.to_dict(), settings.constraints_sha256, "constraints"),
            (self.weights.to_dict(), settings.reward_weights_sha256, "reward weights"),
        )
        for value, expected, name in static:
            if raw_canonical_sha256(value) != expected:
                raise ValueError(f"planner {name} changed after admission")
        checkpoint = describe_checkpoint_contract(
            plant=self.config,
            episode=self.episode_config,
            constraints=self.constraints,
            weights=self.weights,
        )
        if canonical_json(checkpoint) != self.admission.checkpoint_contract_json:
            raise ValueError("planner checkpoint changed after admission")
        require_attested_binding(self.admission.binding)

    def _validate_types(self) -> None:
        exact = (
            ("config", self.config, PlantConfig),
            ("episode_config", self.episode_config, EpisodeConfigV2),
            ("constraints", self.constraints, Constraints),
            ("weights", self.weights, RewardWeightsV2),
        )
        for name, value, expected in exact:
            if type(value) is not expected:
                raise TypeError(
                    f"planner {name} must have exact type {expected.__name__}"
                )


def _same_admission(
    left: PlannerRuntimeAdmissionV1,
    right: PlannerRuntimeAdmissionV1,
) -> bool:
    if left.binding is not right.binding:
        return False
    if canonical_json(left.settings.to_dict()) != canonical_json(
        right.settings.to_dict()
    ):
        return False
    names = (
        "seed",
        "evidence_scope",
        "canonical_episode_verified",
        "episode_manifest_json",
        "episode_manifest_sha256",
        "exogenous_episode_sha256",
        "checkpoint_contract_json",
        "checkpoint_contract_digest",
        "manifest_json",
        "manifest_sha256",
    )
    return all(getattr(left, name) == getattr(right, name) for name in names)


def _state_bytes(
    env: PlantEnvV2,
    *,
    admission: PlannerRuntimeAdmissionV1,
    macro_depth: int,
    battery_throughput_kwh: float,
) -> bytes:
    signature = build_state_signature(
        env,
        manifest_sha256=admission.manifest_sha256,
        exogenous_episode_sha256=admission.exogenous_episode_sha256,
        settings_sha256=admission.settings.settings_sha256,
        action_library_sha256=admission.settings.action_library.sha256,
        macro_depth=macro_depth,
        battery_throughput_kwh=battery_throughput_kwh,
    )
    result: bytes = signature.canonical_bytes
    if not result:
        raise ValueError("planner state signature bytes must not be empty")
    return result


def _validated_records(
    records: tuple[ActionRecordV1, ...],
    *,
    admission: PlannerRuntimeAdmissionV1,
) -> tuple[ActionRecordV1, ...]:
    checked: list[ActionRecordV1] = []
    for index, record in enumerate(records):
        if type(record) is not ActionRecordV1:
            raise TypeError(f"action_records[{index}] must be an exact ActionRecordV1")
        rebuilt = ActionRecordV1(**record.to_dict())
        if canonical_json(rebuilt.to_dict()) != canonical_json(record.to_dict()):
            raise ValueError(f"action_records[{index}] is not canonically normalized")
        if rebuilt.source_code in (-1, 0):
            expected_binding = admission.settings.a0_controller.sha256
        elif rebuilt.source_code == 1:
            expected_binding = admission.binding.guidance_sha256
        else:
            expected_binding = admission.settings.action_library.sha256
        if rebuilt.source_binding_sha256 != expected_binding:
            raise ValueError(
                f"action_records[{index}] has the wrong semantic source binding"
            )
        checked.append(rebuilt)
    result = tuple(checked)
    request_trace_payload(result)
    return result


def _cleanup_reservation(
    guard: _ResourceGuardV1,
    token: _EnvironmentReservationV1,
    env: PlantEnvV2 | None,
    *,
    bound: bool,
    primary: BaseException,
) -> None:
    try:
        if bound:
            if env is None:  # pragma: no cover - local construction invariant
                raise RuntimeError("bound reservation lost its environment")
            guard.release_environment(token, env)
        else:
            guard.cancel_reservation(token)
    except BaseException as cleanup_error:
        primary.add_note(
            "planner cleanup also failed: "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )


def _fresh_reset(
    inputs: PlannerPlantInputsV1,
    guard: _ResourceGuardV1,
    *,
    phase: str,
) -> tuple[_EnvironmentReservationV1, PlantEnvV2]:
    inputs.validate()
    guard.checkpoint(phase=phase)
    token = guard.reserve_environment(phase=phase)
    env: PlantEnvV2 | None = None
    bound = False
    try:
        env = PlantEnvV2(
            config=inputs.config,
            episode=inputs.episode_config,
            constraints=inputs.constraints,
            weights=inputs.weights,
        )
        guard.bind_environment(token, env)
        bound = True
        env.reset(seed=inputs.admission.seed)
        _validate_reset_environment(env, admission=inputs.admission)
        guard.checkpoint(phase=phase)
        return token, env
    except BaseException as exc:
        _cleanup_reservation(guard, token, env, bound=bound, primary=exc)
        raise


def _release_environment(
    guard: _ResourceGuardV1,
    token: _EnvironmentReservationV1,
    env: PlantEnvV2,
    *,
    primary: BaseException | None,
) -> None:
    try:
        guard.release_environment(token, env)
    except BaseException as cleanup_error:
        if primary is None:
            raise
        primary.add_note(
            "planner cleanup also failed: "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )


def _require_a0_descriptor(
    controller: LegacyHeuristicControllerV2,
    *,
    expected_canonical_json: str,
) -> None:
    if type(controller) is not LegacyHeuristicControllerV2:
        raise TypeError("A0 must be the exact LegacyHeuristicControllerV2")
    if canonical_json(controller.to_dict()) != expected_canonical_json:
        raise ValueError("executed A0 descriptor disagrees with the frozen protocol")


def _capture_a0_generation(
    inputs: PlannerPlantInputsV1,
    guard: _ResourceGuardV1,
    counters: PlannerSearchCountersV1,
) -> A0GenerationV1:
    controller = LegacyHeuristicControllerV2()
    descriptor_json = inputs.admission.settings.a0_controller.canonical_json
    _require_a0_descriptor(controller, expected_canonical_json=descriptor_json)
    token, env = _fresh_reset(inputs, guard, phase="a0_generation")
    primary: BaseException | None = None
    try:
        controller.reset()
        _require_a0_descriptor(controller, expected_canonical_json=descriptor_json)
        records: list[ActionRecordV1] = []
        prefixes: list[bytes] = []
        throughput = 0.0
        terminated = False
        truncated = False
        decisions = inputs.admission.settings.expected_decisions
        for decision_index in range(decisions):
            observation = env.controller_observation.without_forecast()
            if observation.forecast is not None:
                raise ValueError("A0 generation observation must be forecast-blind")
            decision = controller.compute_action(
                observation,
                constraints=inputs.constraints,
                config=inputs.config,
                commitment_profile=inputs.episode_config.electrolyser_commitment,
            )
            _require_a0_descriptor(controller, expected_canonical_json=descriptor_json)
            if type(decision) is not ControlDecision:
                raise TypeError("A0 must return an exact ControlDecision")
            if type(decision.action) is not Action:
                raise TypeError("A0 ControlDecision must contain an exact Action")
            invalid = _bool("A0 controller invalid flag", decision.invalid)
            record = ActionRecordV1.from_action(
                decision.action,
                decision_index=decision_index,
                source_phase="a0_capture",
                source_code=-1,
                source_binding_sha256=inputs.admission.settings.a0_controller.sha256,
                controller_invalid=invalid,
            )
            throughput, terminated, truncated = step_action_record(
                env,
                record,
                battery_throughput_kwh=throughput,
            )
            counters.a0_generation_decisions += 1
            records.append(record)
            expected_terminal = decision_index + 1 == decisions
            if terminated != expected_terminal or truncated:
                raise ValueError("A0 generation terminated at the wrong decision")
            if (decision_index + 1) % 4 == 0 or terminated:
                depth = (decision_index + 4) // 4
                prefixes.append(
                    _state_bytes(
                        env,
                        admission=inputs.admission,
                        macro_depth=depth,
                        battery_throughput_kwh=throughput,
                    )
                )
                guard.checkpoint(phase="a0_generation")
        snapshot = capture_terminal_snapshot(
            env,
            admission=inputs.admission,
            action_records=tuple(records),
            battery_throughput_kwh=throughput,
            terminated=terminated,
            truncated=truncated,
        )
        return A0GenerationV1(
            action_records=tuple(records),
            terminal_snapshot=snapshot,
            prefix_state_bytes=tuple(prefixes),
        )
    except BaseException as exc:
        primary = exc
        raise
    finally:
        _release_environment(guard, token, env, primary=primary)


def _replay_records(
    inputs: PlannerPlantInputsV1,
    guard: _ResourceGuardV1,
    counters: PlannerSearchCountersV1,
    *,
    records: tuple[ActionRecordV1, ...],
    phase: Literal["a0_replay", "finalist_replay"],
) -> tuple[TerminalSnapshotV1, tuple[bytes, ...]]:
    records = _validated_records(records, admission=inputs.admission)
    if len(records) != inputs.admission.settings.expected_decisions:
        raise ValueError("fresh replay trace does not cover the admitted horizon")
    token, env = _fresh_reset(inputs, guard, phase=phase)
    primary: BaseException | None = None
    try:
        throughput = 0.0
        terminated = False
        truncated = False
        prefixes: list[bytes] = []
        for decision_index, record in enumerate(records):
            throughput, terminated, truncated = step_action_record(
                env,
                record,
                battery_throughput_kwh=throughput,
            )
            if phase == "a0_replay":
                counters.a0_evidence_replay_decisions += 1
            else:
                counters.fresh_finalist_replay_decisions += 1
            expected_terminal = decision_index + 1 == len(records)
            if terminated != expected_terminal or truncated:
                raise ValueError("fresh replay terminated at the wrong decision")
            guard.checkpoint(phase=phase)
            if (decision_index + 1) % 4 == 0 or terminated:
                depth = (decision_index + 4) // 4
                prefixes.append(
                    _state_bytes(
                        env,
                        admission=inputs.admission,
                        macro_depth=depth,
                        battery_throughput_kwh=throughput,
                    )
                )
        snapshot = capture_terminal_snapshot(
            env,
            admission=inputs.admission,
            action_records=records,
            battery_throughput_kwh=throughput,
            terminated=terminated,
            truncated=truncated,
        )
        return snapshot, tuple(prefixes)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        _release_environment(guard, token, env, primary=primary)


def _validate_replay_match(
    expected: TerminalSnapshotV1,
    replayed: TerminalSnapshotV1,
    *,
    config: PlantConfig | None = None,
) -> None:
    if (
        type(expected) is not TerminalSnapshotV1
        or type(replayed) is not TerminalSnapshotV1
    ):
        raise TypeError("replay comparison requires exact TerminalSnapshotV1 values")
    expected.validate()
    replayed.validate()
    if config is not None:
        expected.validate_for_config(config)
        replayed.validate_for_config(config)
    scalar_names = (
        "manifest_sha256",
        "exogenous_episode_sha256",
        "checkpoint_contract_digest",
        "plant_config_sha256",
        "decision_count",
        "physics_steps",
        "terminated",
        "truncated",
    )
    for name in scalar_names:
        if getattr(expected, name) != getattr(replayed, name):
            raise ValueError(f"fresh replay {name} differs from evaluated trajectory")
    payload_names = (
        "request_trace",
        "metrics",
        "terminal_truth",
        "terminal_telemetry",
        "delivery_audit",
        "outcome",
    )
    for name in payload_names:
        if getattr(expected, name) != getattr(replayed, name):
            raise ValueError(f"fresh replay {name} differs from evaluated trajectory")
    if canonical_json(expected.final_score.to_dict()) != canonical_json(
        replayed.final_score.to_dict()
    ):
        raise ValueError("fresh replay final score differs from evaluated trajectory")
    if not _same_binary64(
        expected.battery_throughput_kwh,
        replayed.battery_throughput_kwh,
    ):
        raise ValueError("fresh replay battery throughput differs from evaluation")


@dataclass(frozen=True, slots=True)
class A0EvidenceV1:
    """A0 generation plus its independent evidentiary replay and macro states."""

    generation: A0GenerationV1
    evidence_snapshot: TerminalSnapshotV1
    replay_prefix_state_bytes: tuple[bytes, ...]

    def __post_init__(self) -> None:
        self._validate()
        object.__setattr__(
            self,
            "replay_prefix_state_bytes",
            tuple(self.replay_prefix_state_bytes),
        )

    def validate_for_config(self, config: PlantConfig) -> None:
        self._validate()
        self.generation.terminal_snapshot.validate_for_config(config)
        self.evidence_snapshot.validate_for_config(config)

    def search_trace(self) -> SearchTraceV1[TerminalSnapshotV1]:
        self._validate()
        return SearchTraceV1(
            action_records=self.evidence_snapshot.action_records,
            final_score=self.evidence_snapshot.final_score,
            payload=self.evidence_snapshot,
        )

    def _validate(self) -> None:
        if type(self.generation) is not A0GenerationV1:
            raise TypeError("generation must be an exact A0GenerationV1")
        if type(self.evidence_snapshot) is not TerminalSnapshotV1:
            raise TypeError("evidence_snapshot must be an exact TerminalSnapshotV1")
        self.generation.validate()
        self.evidence_snapshot.validate()
        _validate_replay_match(
            self.generation.terminal_snapshot,
            self.evidence_snapshot,
        )
        if (
            action_record_trace_payload(self.generation.action_records)
            != self.evidence_snapshot.action_record_trace
        ):
            raise ValueError("A0 replay action-record lineage differs from capture")
        prefixes = tuple(self.replay_prefix_state_bytes)
        for index, prefix in enumerate(prefixes):
            if not isinstance(prefix, bytes) or not prefix:
                raise TypeError(
                    f"replay_prefix_state_bytes[{index}] must be non-empty bytes"
                )
        if prefixes != self.generation.prefix_state_bytes:
            raise ValueError("A0 generation and replay macro states differ")


def capture_a0_evidence(
    inputs: PlannerPlantInputsV1,
    *,
    guard: _ResourceGuardV1,
    counters: PlannerSearchCountersV1,
) -> A0EvidenceV1:
    """Generate exact A0 once and independently replay it as evidence."""

    if type(inputs) is not PlannerPlantInputsV1:
        raise TypeError("inputs must be an exact PlannerPlantInputsV1")
    if type(guard) is not _ResourceGuardV1:
        raise TypeError("guard must be an exact _ResourceGuardV1")
    if type(counters) is not PlannerSearchCountersV1:
        raise TypeError("counters must be exact PlannerSearchCountersV1")
    if guard._counters is not counters:
        raise ValueError(
            "A0 counters must be the resource guard's exact counter object"
        )
    if any(_counter_values(counters).values()):
        raise ValueError("A0 capture requires a fresh zeroed counter record")
    inputs.validate()
    require_attested_binding(inputs.admission.binding)
    generation = _capture_a0_generation(inputs, guard, counters)
    replay, prefixes = _replay_records(
        inputs,
        guard,
        counters,
        records=generation.action_records,
        phase="a0_replay",
    )
    evidence = A0EvidenceV1(
        generation=generation,
        evidence_snapshot=replay,
        replay_prefix_state_bytes=prefixes,
    )
    evidence.validate_for_config(inputs.config)
    return evidence


class PlantPlannerSearchPortV1(
    PlannerSearchPortV1[PlantPlannerCandidateV1, TerminalSnapshotV1]
):
    """Concrete exact-simulator implementation of `PlannerSearchPortV1`."""

    def __init__(
        self,
        *,
        inputs: PlannerPlantInputsV1,
        a0: A0EvidenceV1,
        guard: _ResourceGuardV1,
        counters: PlannerSearchCountersV1,
    ) -> None:
        if type(inputs) is not PlannerPlantInputsV1:
            raise TypeError("inputs must be an exact PlannerPlantInputsV1")
        if type(a0) is not A0EvidenceV1:
            raise TypeError("a0 must be an exact A0EvidenceV1")
        if type(guard) is not _ResourceGuardV1:
            raise TypeError("guard must be an exact _ResourceGuardV1")
        if type(counters) is not PlannerSearchCountersV1:
            raise TypeError("counters must be exact PlannerSearchCountersV1")
        if guard._counters is not counters:
            raise ValueError(
                "search counters must be the resource guard's exact counter object"
            )
        counter_values = _counter_values(counters)
        expected_decisions = inputs.admission.settings.expected_decisions
        if (
            counter_values.pop("a0_generation_decisions") != expected_decisions
            or counter_values.pop("a0_evidence_replay_decisions") != expected_decisions
            or counter_values.pop("max_live_environments_observed") != 1
            or any(counter_values.values())
        ):
            raise ValueError(
                "search port requires exactly one completed A0 evidence run"
            )
        inputs.validate()
        a0.validate_for_config(inputs.config)
        self._inputs = inputs
        self._a0 = a0
        self._guard = guard
        self._counters = counters
        self._registry = PlannerCandidateRegistryV1(guard)
        self._root_issued = False
        self._cursor_token: _EnvironmentReservationV1 | None = None
        self._cursor_env: PlantEnvV2 | None = None
        self._cursor_records: tuple[ActionRecordV1, ...] = ()
        self._cursor_throughput = 0.0
        self._cursor_depth = 0
        self._pending_coordinate_phase: CoordinatePhase | None = None
        self._closed = False

    def initial_candidate(self) -> PlantPlannerCandidateV1:
        self._require_open()
        if self._root_issued:
            raise ValueError("primary-search root was already issued")
        self._root_issued = True
        token, env = _fresh_reset(self._inputs, self._guard, phase="beam_search")
        try:
            return self._register_bound_candidate(
                token,
                env,
                action_records=(),
                battery_throughput_kwh=0.0,
                macro_depth=0,
                terminated=False,
                truncated=False,
            )
        except BaseException as exc:
            _cleanup_reservation(
                self._guard,
                token,
                env,
                bound=True,
                primary=exc,
            )
            raise

    def advance_a0_cursor(
        self,
        *,
        macro_depth: int,
        start_decision: int,
        stop_decision: int,
        terminal: bool,
    ) -> PlantPlannerCandidateV1 | None:
        self._require_open()
        if not self._root_issued:
            raise ValueError("A0 cursor cannot advance before the beam root exists")
        depth = _integer("macro_depth", macro_depth, minimum=1)
        start = _integer("start_decision", start_decision)
        stop = _integer("stop_decision", stop_decision, minimum=1)
        ended = _bool("terminal", terminal)
        decisions = self._inputs.admission.settings.expected_decisions
        if (
            depth != self._cursor_depth + 1
            or start != (depth - 1) * 4
            or start != len(self._cursor_records)
            or stop != min(start + 4, decisions)
            or ended != (stop == decisions)
        ):
            raise ValueError("A0 cursor request is not the next frozen macro")
        if self._cursor_env is None:
            token, env = _fresh_reset(
                self._inputs,
                self._guard,
                phase="beam_search",
            )
            self._cursor_token = token
            self._cursor_env = env
        env = self._cursor_env
        assert env is not None
        throughput = self._cursor_throughput
        records = list(self._cursor_records)
        terminated = False
        truncated = False
        a0_records = self._a0.generation.action_records
        for record in a0_records[start:stop]:
            throughput, terminated, truncated = step_action_record(
                env,
                record,
                battery_throughput_kwh=throughput,
            )
            records.append(record)
        if terminated != ended or truncated:
            raise ValueError("A0 cursor termination disagrees with macro boundary")
        actual_state = _state_bytes(
            env,
            admission=self._inputs.admission,
            macro_depth=depth,
            battery_throughput_kwh=throughput,
        )
        expected_state = self._a0.replay_prefix_state_bytes[depth - 1]
        if actual_state != expected_state:
            raise ValueError("rolling A0 cursor differs from evidentiary replay")
        self._guard.checkpoint(phase="beam_search")
        self._cursor_records = tuple(records)
        self._cursor_throughput = throughput
        self._cursor_depth = depth
        if ended:
            self._release_cursor()
            return None

        token = self._guard.reserve_environment(phase="beam_search")
        clone: PlantEnvV2 | None = None
        bound = False
        try:
            clone = clone_environment(env)
            self._guard.bind_environment(token, clone)
            bound = True
            return self._register_bound_candidate(
                token,
                clone,
                action_records=self._cursor_records,
                battery_throughput_kwh=throughput,
                macro_depth=depth,
                terminated=False,
                truncated=False,
            )
        except BaseException as exc:
            _cleanup_reservation(self._guard, token, clone, bound=bound, primary=exc)
            raise

    def expand_macro(
        self,
        parent: PlantPlannerCandidateV1,
        *,
        template_id: int,
        macro_depth: int,
        start_decision: int,
        stop_decision: int,
    ) -> PlantPlannerCandidateV1:
        self._require_open()
        env = self._registry.validate_candidate(parent)
        template = _integer("template_id", template_id)
        depth = _integer("macro_depth", macro_depth, minimum=1)
        start = _integer("start_decision", start_decision)
        stop = _integer("stop_decision", stop_decision, minimum=1)
        decisions = self._inputs.admission.settings.expected_decisions
        if (
            template > 14
            or start != (depth - 1) * 4
            or start != len(parent.action_records)
            or depth != parent.macro_depth + 1
            or stop != min(start + 4, decisions)
        ):
            raise ValueError("macro expansion does not match the frozen schedule")
        token = self._guard.reserve_environment(phase="beam_search")
        child_env: PlantEnvV2 | None = None
        bound = False
        try:
            child_env = clone_environment(env)
            self._guard.bind_environment(token, child_env)
            bound = True
            records = list(parent.action_records)
            throughput = parent.battery_throughput_kwh
            terminated = False
            truncated = False
            for decision_index in range(start, stop):
                record = generate_action_record(
                    template_id=template,
                    env=child_env,
                    decision_index=decision_index,
                    source_phase="beam",
                    binding=self._inputs.admission.binding,
                )
                throughput, terminated, truncated = step_action_record(
                    child_env,
                    record,
                    battery_throughput_kwh=throughput,
                )
                records.append(record)
                if terminated != (decision_index + 1 == decisions) or truncated:
                    raise ValueError("beam macro terminated at the wrong decision")
            return self._register_bound_candidate(
                token,
                child_env,
                action_records=tuple(records),
                battery_throughput_kwh=throughput,
                macro_depth=depth,
                terminated=terminated,
                truncated=truncated,
            )
        except BaseException as exc:
            _cleanup_reservation(
                self._guard,
                token,
                child_env,
                bound=bound,
                primary=exc,
            )
            raise

    def state_bytes(self, candidate: PlantPlannerCandidateV1) -> bytes:
        env = self._validated_candidate(candidate)
        return _state_bytes(
            env,
            admission=self._inputs.admission,
            macro_depth=candidate.macro_depth,
            battery_throughput_kwh=candidate.battery_throughput_kwh,
        )

    def intermediate_rank(
        self,
        candidate: PlantPlannerCandidateV1,
    ) -> IntermediateRankV1:
        self._validated_candidate(candidate)
        return extract_intermediate_rank(candidate)

    def action_records(
        self,
        candidate: PlantPlannerCandidateV1,
    ) -> tuple[ActionRecordV1, ...]:
        self._validated_candidate(candidate)
        return tuple(candidate.action_records)

    def terminal_trace(
        self,
        candidate: PlantPlannerCandidateV1,
    ) -> SearchTraceV1[TerminalSnapshotV1]:
        env = self._validated_candidate(candidate)
        if not candidate.terminated or candidate.truncated:
            raise ValueError("terminal trace requires a complete candidate")
        snapshot = capture_terminal_snapshot(
            env,
            admission=self._inputs.admission,
            action_records=candidate.action_records,
            battery_throughput_kwh=candidate.battery_throughput_kwh,
            terminated=candidate.terminated,
            truncated=candidate.truncated,
        )
        snapshot.validate_for_config(self._inputs.config)
        return SearchTraceV1(
            action_records=snapshot.action_records,
            final_score=snapshot.final_score,
            payload=snapshot,
        )

    def coordinate_trial(
        self,
        base: SearchTraceV1[TerminalSnapshotV1],
        *,
        start_decision: int,
        stop_decision: int,
        template_id: int,
        source_phase: CoordinatePhase,
    ) -> SearchTraceV1[TerminalSnapshotV1]:
        self._require_open()
        if self._pending_coordinate_phase is not None:
            raise ValueError("prior coordinate trial lacks its resource checkpoint")
        if self._registry.live_candidates or self._cursor_env is not None:
            raise ValueError("coordinate trials require the beam phase to be released")
        self._validate_trace(base)
        start = _integer("start_decision", start_decision)
        stop = _integer("stop_decision", stop_decision, minimum=1)
        template = _integer("template_id", template_id)
        decisions = self._inputs.admission.settings.expected_decisions
        block_size = 24 if source_phase == "coarse_polish" else 4
        if (
            source_phase not in ("coarse_polish", "fine_polish")
            or template > 14
            or not 0 <= start < stop <= decisions
            or start % block_size != 0
            or stop != min(start + block_size, decisions)
        ):
            raise ValueError("coordinate trial request is outside the frozen schedule")
        token, env = _fresh_reset(self._inputs, self._guard, phase=source_phase)
        primary: BaseException | None = None
        try:
            records: list[ActionRecordV1] = []
            throughput = 0.0
            terminated = False
            truncated = False
            for decision_index in range(decisions):
                if start <= decision_index < stop:
                    record = generate_action_record(
                        template_id=template,
                        env=env,
                        decision_index=decision_index,
                        source_phase=source_phase,
                        binding=self._inputs.admission.binding,
                    )
                else:
                    record = base.action_records[decision_index]
                throughput, terminated, truncated = step_action_record(
                    env,
                    record,
                    battery_throughput_kwh=throughput,
                )
                records.append(record)
                if terminated != (decision_index + 1 == decisions) or truncated:
                    raise ValueError(
                        "coordinate trial terminated at the wrong decision"
                    )
            snapshot = capture_terminal_snapshot(
                env,
                admission=self._inputs.admission,
                action_records=tuple(records),
                battery_throughput_kwh=throughput,
                terminated=terminated,
                truncated=truncated,
            )
            snapshot.validate_for_config(self._inputs.config)
            result = SearchTraceV1(
                action_records=snapshot.action_records,
                final_score=snapshot.final_score,
                payload=snapshot,
            )
            self._pending_coordinate_phase = source_phase
            return result
        except BaseException as exc:
            primary = exc
            raise
        finally:
            _release_environment(self._guard, token, env, primary=primary)

    def replay_finalist(
        self,
        trace: SearchTraceV1[TerminalSnapshotV1],
    ) -> TerminalSnapshotV1:
        self._require_open()
        if self._pending_coordinate_phase is not None:
            raise ValueError("finalist replay requires the coordinate checkpoint")
        if self._registry.live_candidates or self._cursor_env is not None:
            raise ValueError(
                "finalist replay requires all search environments released"
            )
        self._validate_trace(trace)
        replayed, _ = _replay_records(
            self._inputs,
            self._guard,
            self._counters,
            records=trace.action_records,
            phase="finalist_replay",
        )
        _validate_replay_match(
            trace.payload,
            replayed,
            config=self._inputs.config,
        )
        return replayed

    def release(self, candidate: PlantPlannerCandidateV1) -> None:
        self._require_open()
        self._registry.release(candidate)

    def resource_checkpoint(
        self,
        checkpoint: Literal["completed_edge", "completed_coordinate_trial"],
    ) -> None:
        self._require_open()
        if checkpoint == "completed_edge":
            if self._pending_coordinate_phase is not None:
                raise ValueError("coordinate resource checkpoint is still pending")
            self._guard.checkpoint(phase="beam_search")
            return
        if checkpoint != "completed_coordinate_trial":
            raise ValueError("unknown planner search resource checkpoint")
        phase = self._pending_coordinate_phase
        if phase is None:
            raise ValueError("coordinate resource checkpoint has no completed trial")
        self._pending_coordinate_phase = None
        self._guard.checkpoint(phase=phase)

    def close(self) -> None:
        if self._closed:
            return
        first_error: BaseException | None = None
        try:
            self._registry.release_all()
        except BaseException as exc:
            first_error = exc
        try:
            self._release_cursor()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        if first_error is not None:
            raise first_error
        self._pending_coordinate_phase = None
        self._closed = True

    def _validated_candidate(
        self,
        candidate: PlantPlannerCandidateV1,
    ) -> PlantEnvV2:
        self._require_open()
        return self._registry.validate_candidate(candidate)

    def _validate_trace(self, trace: SearchTraceV1[TerminalSnapshotV1]) -> None:
        if type(trace) is not SearchTraceV1:
            raise TypeError("planner trace must be an exact SearchTraceV1")
        if type(trace.final_score) is not FinalScoreV1:
            raise TypeError("planner trace score must be an exact FinalScoreV1")
        if type(trace.payload) is not TerminalSnapshotV1:
            raise TypeError("planner trace payload must be an exact TerminalSnapshotV1")
        trace.payload.validate_for_config(self._inputs.config)
        checked_records = _validated_records(
            tuple(trace.action_records),
            admission=self._inputs.admission,
        )
        if action_record_trace_payload(
            checked_records
        ) != trace.payload.action_record_trace or canonical_json(
            trace.final_score.to_dict()
        ) != canonical_json(
            trace.payload.final_score.to_dict()
        ):
            raise ValueError("planner trace disagrees with its terminal snapshot")

    def _register_bound_candidate(
        self,
        token: _EnvironmentReservationV1,
        env: PlantEnvV2,
        *,
        action_records: tuple[ActionRecordV1, ...],
        battery_throughput_kwh: float,
        macro_depth: int,
        terminated: bool,
        truncated: bool,
    ) -> PlantPlannerCandidateV1:
        candidate = PlantPlannerCandidateV1(
            env=env,
            action_records=action_records,
            battery_throughput_kwh=battery_throughput_kwh,
            macro_depth=macro_depth,
            terminated=terminated,
            truncated=truncated,
        )
        self._registry.register(token, candidate)
        return candidate

    def _release_cursor(self) -> None:
        token = self._cursor_token
        env = self._cursor_env
        if token is None and env is None:
            return
        if token is None or env is None:
            raise ValueError("A0 cursor ownership is internally inconsistent")
        self._guard.release_environment(token, env)
        self._cursor_token = None
        self._cursor_env = None

    def _require_open(self) -> None:
        if self._closed:
            raise ValueError("planner search port is closed")


__all__ = [
    "A0EvidenceV1",
    "PlannerPlantInputsV1",
    "PlantPlannerSearchPortV1",
    "capture_a0_evidence",
]

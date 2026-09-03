"""Generic deterministic search for the frozen feasible-planner protocol.

This module deliberately has no dependency on :class:`PlantEnvV2`, the simulator,
or the Part-I relaxation.  It owns only ordering, exact same-depth deduplication,
the bounded streaming frontier, terminal selection, and block-coordinate polishing.
The plant-specific runtime implements :class:`PlannerSearchPortV1`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key
import math
from numbers import Integral, Real
from typing import Generic, Literal, Protocol, TypeVar, cast

from .planner_schema import (
    MODE_PROTECTION_ORDER,
    ActionRecordV1,
    FinalScoreV1,
    canonical_json,
    compare_action_lineage,
    compare_final_scores,
    compare_service_fractions,
    request_trace_payload,
)


BEAM_WIDTH = 24
MACRO_DECISIONS = 4
MAX_ATTEMPTED_EDGES = 25_575
TEMPLATE_IDS = tuple(range(15))
COARSE_BLOCK_DECISIONS = 24
FINE_BLOCK_DECISIONS = 4

MODE_READINESS_ORDER = (
    "running",
    "hot_standby",
    "hot_starting",
    "cold_starting",
    "off",
    "cooldown",
)
_MODE_READINESS = {
    mode: len(MODE_READINESS_ORDER) - index
    for index, mode in enumerate(MODE_READINESS_ORDER)
}

ResourceCheckpoint = Literal["completed_edge", "completed_coordinate_trial"]

CandidateT = TypeVar("CandidateT")
PayloadT = TypeVar("PayloadT")


class PlannerSearchError(RuntimeError):
    """Stable algorithm-layer failure for translation by the public planner."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


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


@dataclass(frozen=True, slots=True)
class IntermediateRankV1:
    """The twelve state-derived fields preceding the deterministic path tie."""

    delivered_plus_deadline_readiness_kg: float
    delivered_kg: float
    service_numerator: int
    service_denominator: int
    worst_fill: float
    ch4_inventory_kg: float
    feedstock_potential_kg: float
    battery_energy_kwh: float
    commitment_mode: str
    startup_remaining_s: float
    cooldown_remaining_s: float
    shortfall_elapsed_s: float
    forced_stops: int
    aborted_starts: int
    cold_starts: int
    curtailed_kwh: float
    battery_throughput_kwh: float

    def __post_init__(self) -> None:
        for name in (
            "delivered_plus_deadline_readiness_kg",
            "delivered_kg",
            "worst_fill",
            "ch4_inventory_kg",
            "feedstock_potential_kg",
            "battery_energy_kwh",
            "startup_remaining_s",
            "cooldown_remaining_s",
            "shortfall_elapsed_s",
            "curtailed_kwh",
            "battery_throughput_kwh",
        ):
            object.__setattr__(
                self, name, _finite(name, getattr(self, name), minimum=0.0)
            )
        if self.worst_fill > 1.0:
            raise ValueError("worst_fill must not exceed one")
        numerator = _integer("service_numerator", self.service_numerator)
        denominator = _integer("service_denominator", self.service_denominator)
        # Reuse the schema's exact service validation, including the 0/0 rule.
        compare_service_fractions(numerator, denominator, numerator, denominator)
        object.__setattr__(self, "service_numerator", numerator)
        object.__setattr__(self, "service_denominator", denominator)
        for name in ("forced_stops", "aborted_starts", "cold_starts"):
            object.__setattr__(
                self, name, _integer(name, getattr(self, name), minimum=0)
            )
        if self.commitment_mode not in _MODE_READINESS:
            raise ValueError(f"unsupported commitment_mode: {self.commitment_mode!r}")


@dataclass(frozen=True, slots=True)
class SearchTraceV1(Generic[PayloadT]):
    """A complete trace plus a detached runtime-owned terminal payload."""

    action_records: tuple[ActionRecordV1, ...]
    final_score: FinalScoreV1
    payload: PayloadT

    def __post_init__(self) -> None:
        records = tuple(self.action_records)
        # This validates record types and contiguous zero-based indices.
        request_trace_payload(records)
        object.__setattr__(self, "action_records", records)
        if not isinstance(self.final_score, FinalScoreV1):
            raise TypeError("final_score must be a FinalScoreV1")


@dataclass(slots=True)
class PlannerSearchCountersV1:
    """Mutable deterministic counters shared with the plant-specific runtime."""

    completed_macro_depths: int = 0
    attempted_edges: int = 0
    completed_edges: int = 0
    generated_candidates: int = 0
    a0_cursor_validations: int = 0
    a0_injections: int = 0
    exact_duplicates: int = 0
    frontier_retained_total: int = 0
    peak_frontier: int = 0
    coordinate_trials_attempted: int = 0
    coordinate_trials_completed: int = 0
    coordinate_trials_accepted: int = 0
    a0_generation_decisions: int = 0
    a0_evidence_replay_decisions: int = 0
    fresh_finalist_replay_decisions: int = 0
    max_live_environments_observed: int = 0


class PlannerSearchPortV1(Protocol[CandidateT, PayloadT]):
    """Plant-independent operations required by the deterministic search."""

    def initial_candidate(self) -> CandidateT:
        """Return one fresh-reset root with an empty action trace."""

    def advance_a0_cursor(
        self,
        *,
        macro_depth: int,
        start_decision: int,
        stop_decision: int,
        terminal: bool,
    ) -> CandidateT | None:
        """Validate one stored A0 prefix and clone it only when nonterminal."""

    def expand_macro(
        self,
        parent: CandidateT,
        *,
        template_id: int,
        macro_depth: int,
        start_decision: int,
        stop_decision: int,
    ) -> CandidateT:
        """Clone and complete one ordered semantic-template macro."""

    def state_bytes(self, candidate: CandidateT) -> bytes:
        """Return the complete canonical same-depth signature bytes."""

    def intermediate_rank(self, candidate: CandidateT) -> IntermediateRankV1:
        """Return authoritative intermediate fields for one candidate."""

    def action_records(self, candidate: CandidateT) -> tuple[ActionRecordV1, ...]:
        """Return the candidate's immutable, contiguous action records."""

    def terminal_trace(self, candidate: CandidateT) -> SearchTraceV1[PayloadT]:
        """Detach a complete terminal evaluation from a terminal candidate."""

    def coordinate_trial(
        self,
        base: SearchTraceV1[PayloadT],
        *,
        start_decision: int,
        stop_decision: int,
        template_id: int,
        source_phase: Literal["coarse_polish", "fine_polish"],
    ) -> SearchTraceV1[PayloadT]:
        """Evaluate one full-horizon immutable-block replacement."""

    def release(self, candidate: CandidateT) -> None:
        """Release exactly one live candidate environment."""

    def resource_checkpoint(self, checkpoint: ResourceCheckpoint) -> None:
        """Apply the frozen wall/RSS check after a completed unit of work."""


def compare_intermediate_ranks(
    left: IntermediateRankV1, right: IntermediateRankV1
) -> int:
    """Return +1 iff ``left`` wins the state-derived intermediate rank."""

    if not isinstance(left, IntermediateRankV1) or not isinstance(
        right, IntermediateRankV1
    ):
        raise TypeError("intermediate operands must be IntermediateRankV1")

    higher_fields = (
        "delivered_plus_deadline_readiness_kg",
        "delivered_kg",
    )
    for name in higher_fields:
        result = (getattr(left, name) > getattr(right, name)) - (
            getattr(left, name) < getattr(right, name)
        )
        if result:
            return int(result)

    result = compare_service_fractions(
        left.service_numerator,
        left.service_denominator,
        right.service_numerator,
        right.service_denominator,
    )
    if result:
        return int(result)

    for name in (
        "worst_fill",
        "ch4_inventory_kg",
        "feedstock_potential_kg",
        "battery_energy_kwh",
    ):
        result = (getattr(left, name) > getattr(right, name)) - (
            getattr(left, name) < getattr(right, name)
        )
        if result:
            return int(result)

    result = (
        _MODE_READINESS[left.commitment_mode] > _MODE_READINESS[right.commitment_mode]
    ) - (_MODE_READINESS[left.commitment_mode] < _MODE_READINESS[right.commitment_mode])
    if result:
        return int(result)

    for name in (
        "startup_remaining_s",
        "cooldown_remaining_s",
        "shortfall_elapsed_s",
        "forced_stops",
        "aborted_starts",
        "cold_starts",
        "curtailed_kwh",
        "battery_throughput_kwh",
    ):
        result = (getattr(right, name) > getattr(left, name)) - (
            getattr(right, name) < getattr(left, name)
        )
        if result:
            return int(result)
    return 0


def concrete_request_bytes(records: tuple[ActionRecordV1, ...]) -> bytes:
    """Return the protocol's canonical concrete-request trace bytes."""

    serialized = cast(str, request_trace_payload(records).canonical_json)
    return serialized.encode("utf-8")


def canonical_action_record_bytes(
    records: tuple[ActionRecordV1, ...],
) -> bytes:
    """Serialize any contiguous action-record region with binary64 exactness.

    Full zero-based traces have exactly the canonical JSON of the protocol's
    ``planner-action-record-trace-v1`` payload.  Allowing a non-zero first index
    lets the same representation compare fixed prefixes and suffixes without
    renumbering them and accidentally hiding an index change.
    """

    items = tuple(records)
    if items:
        if not isinstance(items[0], ActionRecordV1):
            raise TypeError("records[0] must be an ActionRecordV1")
        first_index = items[0].decision_index
        for offset, record in enumerate(items):
            if not isinstance(record, ActionRecordV1):
                raise TypeError(f"records[{offset}] must be an ActionRecordV1")
            if record.decision_index != first_index + offset:
                raise ValueError("action-record region indices must be contiguous")
            # Reconstruct to reject illicit mutation of a frozen record.
            ActionRecordV1(
                decision_index=record.decision_index,
                source_phase=record.source_phase,
                source_code=record.source_code,
                source_binding_sha256=record.source_binding_sha256,
                controller_invalid=record.controller_invalid,
                battery_share=record.battery_share,
                electrolyser_share=record.electrolyser_share,
                dac_share=record.dac_share,
                methanator_share=record.methanator_share,
                battery_support_kw=record.battery_support_kw,
            )
    serialized = cast(
        str,
        canonical_json(
            {"schema_version": 1, "records": [record.to_dict() for record in items]}
        ),
    )
    return serialized.encode("utf-8")


def _same_action_records(
    left: tuple[ActionRecordV1, ...], right: tuple[ActionRecordV1, ...]
) -> bool:
    return canonical_action_record_bytes(left) == canonical_action_record_bytes(right)


def compare_searched_traces(
    left: SearchTraceV1[PayloadT], right: SearchTraceV1[PayloadT]
) -> int:
    """Compare complete searched traces by task score and frozen lineage."""

    result = compare_final_scores(left.final_score, right.final_score)
    if result:
        return int(result)
    return int(compare_action_lineage(left.action_records, right.action_records))


def is_unchanged_a0_trace(
    trace: SearchTraceV1[PayloadT],
    a0: SearchTraceV1[PayloadT],
) -> bool:
    """Apply the complete-request plus all-``-1`` incumbent-priority rule."""

    return all(
        record.source_phase == "a0_capture" and record.source_code == -1
        for record in trace.action_records
    ) and concrete_request_bytes(trace.action_records) == concrete_request_bytes(
        a0.action_records
    )


def compare_traces_with_a0_priority(
    left: SearchTraceV1[PayloadT],
    right: SearchTraceV1[PayloadT],
    *,
    a0: SearchTraceV1[PayloadT],
) -> int:
    """Compare final traces, granting only unchanged A0 an equal-score priority."""

    result = compare_final_scores(left.final_score, right.final_score)
    if result:
        return int(result)
    left_a0 = is_unchanged_a0_trace(left, a0)
    right_a0 = is_unchanged_a0_trace(right, a0)
    if left_a0 != right_a0:
        return 1 if left_a0 else -1
    return int(compare_action_lineage(left.action_records, right.action_records))


@dataclass(frozen=True, slots=True)
class _CandidateSummary:
    signature: bytes
    rank: IntermediateRankV1
    records: tuple[ActionRecordV1, ...]
    exact_a0_prefix: bool


@dataclass(eq=False, slots=True)
class _LiveEntry(Generic[CandidateT]):
    summary: _CandidateSummary
    candidate: CandidateT | None


@dataclass(slots=True)
class _SeenState(Generic[CandidateT]):
    summary: _CandidateSummary
    live: _LiveEntry[CandidateT] | None


def _compare_summaries(left: _CandidateSummary, right: _CandidateSummary) -> int:
    result = compare_intermediate_ranks(left.rank, right.rank)
    if result:
        return result
    if left.exact_a0_prefix != right.exact_a0_prefix:
        return 1 if left.exact_a0_prefix else -1
    return int(compare_action_lineage(left.records, right.records))


def _validated_state_bytes(
    port: PlannerSearchPortV1[CandidateT, PayloadT],
    candidate: CandidateT,
    *,
    terminal: bool,
) -> bytes:
    signature = port.state_bytes(candidate)
    if not isinstance(signature, bytes) or not signature:
        label = "terminal state" if terminal else "candidate state"
        raise PlannerSearchError(
            "state_signature_error", f"{label} signature must be non-empty bytes"
        )
    return signature


class _FrontierAccumulator(Generic[CandidateT, PayloadT]):
    """Depth-local exact dedup plus bounded live candidate reservoirs."""

    def __init__(
        self,
        port: PlannerSearchPortV1[CandidateT, PayloadT],
        counters: PlannerSearchCountersV1,
    ) -> None:
        self._port = port
        self._counters = counters
        self._seen: dict[bytes, _SeenState[CandidateT]] = {}
        self._top: list[_LiveEntry[CandidateT]] = []
        self._mode: dict[str, _LiveEntry[CandidateT]] = {}
        self._a0: _LiveEntry[CandidateT] | None = None

    def offer(self, candidate: CandidateT, *, exact_a0_prefix: bool) -> None:
        try:
            summary = self.prepare(candidate, exact_a0_prefix=exact_a0_prefix)
        except Exception:
            self._port.release(candidate)
            raise

        self.offer_prepared(candidate, summary=summary)

    def prepare(
        self,
        candidate: CandidateT,
        *,
        exact_a0_prefix: bool,
        signature: bytes | None = None,
        records: tuple[ActionRecordV1, ...] | None = None,
    ) -> _CandidateSummary:
        """Build and validate the signature/rank material for one completed edge."""

        return self._summary(
            candidate,
            exact_a0_prefix=exact_a0_prefix,
            signature=signature,
            records=records,
        )

    def offer_prepared(
        self, candidate: CandidateT, *, summary: _CandidateSummary
    ) -> None:
        """Take ownership of a candidate whose completed-edge checks already ran."""

        entry: _LiveEntry[CandidateT] | None = None
        try:
            prior = self._seen.get(summary.signature)
            if prior is not None:
                self._counters.exact_duplicates += 1
                if compare_intermediate_ranks(summary.rank, prior.summary.rank) != 0:
                    raise PlannerSearchError(
                        "state_signature_error",
                        "equal state bytes produced different intermediate ranks",
                    )
                if _compare_summaries(summary, prior.summary) <= 0:
                    self._port.release(candidate)
                    return
                if prior.live is not None:
                    self._remove_all_references(prior.live)
                    self._release_and_unlink(prior.live)

            entry = _LiveEntry(summary=summary, candidate=candidate)
            self._seen[summary.signature] = _SeenState(summary=summary, live=entry)

            if summary.exact_a0_prefix:
                self._a0 = entry
            self._consider_top(entry)
            self._consider_mode(entry)
            if not self._is_referenced(entry):
                self._release_and_unlink(entry)
        except BaseException:
            if entry is None:
                self._port.release(candidate)
            else:
                self._remove_all_references(entry)
                self._release_and_unlink(entry)
            raise

    def take_frontier(self) -> list[CandidateT]:
        if self._a0 is None or self._a0.candidate is None:
            raise PlannerSearchError(
                "search_incomplete", "the exact A0 prefix did not survive the cut"
            )
        selected: list[_LiveEntry[CandidateT]] = []
        signatures: set[bytes] = set()

        def insert(entry: _LiveEntry[CandidateT] | None) -> None:
            if (
                entry is None
                or entry.candidate is None
                or entry.summary.signature in signatures
                or len(selected) >= BEAM_WIDTH
            ):
                return
            selected.append(entry)
            signatures.add(entry.summary.signature)

        insert(self._a0)
        for mode in MODE_PROTECTION_ORDER:
            insert(self._mode.get(mode))
        for entry in self._top:
            insert(entry)

        selected_ids = {id(entry) for entry in selected}
        for entry in self._all_live_entries():
            if id(entry) not in selected_ids:
                self._release_entry(entry)

        frontier: list[CandidateT] = []
        for entry in selected:
            candidate = entry.candidate
            if candidate is None:  # pragma: no cover - guarded by insert
                raise RuntimeError("selected frontier entry was released")
            frontier.append(candidate)
            entry.candidate = None
        self._top.clear()
        self._mode.clear()
        self._a0 = None
        self._seen.clear()
        return frontier

    def close(self) -> None:
        for entry in self._all_live_entries():
            self._release_entry(entry)
        self._top.clear()
        self._mode.clear()
        self._a0 = None
        self._seen.clear()

    def _summary(
        self,
        candidate: CandidateT,
        *,
        exact_a0_prefix: bool,
        signature: bytes | None,
        records: tuple[ActionRecordV1, ...] | None,
    ) -> _CandidateSummary:
        if signature is None:
            signature = _validated_state_bytes(self._port, candidate, terminal=False)
        elif not isinstance(signature, bytes) or not signature:
            raise PlannerSearchError(
                "state_signature_error",
                "candidate state signature must be non-empty bytes",
            )
        rank = self._port.intermediate_rank(candidate)
        if not isinstance(rank, IntermediateRankV1):
            raise PlannerSearchError(
                "state_signature_error",
                "candidate rank must be IntermediateRankV1",
            )
        if records is None:
            records = tuple(self._port.action_records(candidate))
            request_trace_payload(records)
        if exact_a0_prefix and any(
            record.source_phase != "a0_capture" or record.source_code != -1
            for record in records
        ):
            raise PlannerSearchError(
                "search_incomplete", "injected A0 candidate lost its source lineage"
            )
        return _CandidateSummary(
            signature=signature,
            rank=rank,
            records=records,
            exact_a0_prefix=exact_a0_prefix,
        )

    def _consider_top(self, entry: _LiveEntry[CandidateT]) -> None:
        self._top.append(entry)

        def compare_live_entries(
            left: _LiveEntry[CandidateT], right: _LiveEntry[CandidateT]
        ) -> int:
            return -_compare_summaries(left.summary, right.summary)

        self._top.sort(key=cmp_to_key(compare_live_entries))
        if len(self._top) > BEAM_WIDTH:
            removed = self._top.pop()
            if removed is not entry and not self._is_referenced(removed):
                self._release_and_unlink(removed)

    def _consider_mode(self, entry: _LiveEntry[CandidateT]) -> None:
        mode = entry.summary.rank.commitment_mode
        prior = self._mode.get(mode)
        if prior is None or _compare_summaries(entry.summary, prior.summary) > 0:
            self._mode[mode] = entry
            if prior is not None and not self._is_referenced(prior):
                self._release_and_unlink(prior)

    def _remove_all_references(self, entry: _LiveEntry[CandidateT]) -> None:
        self._top = [item for item in self._top if item is not entry]
        for mode, item in tuple(self._mode.items()):
            if item is entry:
                del self._mode[mode]
        if self._a0 is entry:
            self._a0 = None

    def _is_referenced(self, entry: _LiveEntry[CandidateT]) -> bool:
        return (
            self._a0 is entry
            or any(item is entry for item in self._top)
            or any(item is entry for item in self._mode.values())
        )

    def _release_and_unlink(self, entry: _LiveEntry[CandidateT]) -> None:
        self._release_entry(entry)
        seen = self._seen.get(entry.summary.signature)
        if seen is not None and seen.live is entry:
            seen.live = None

    def _release_entry(self, entry: _LiveEntry[CandidateT]) -> None:
        if entry.candidate is not None:
            candidate = entry.candidate
            entry.candidate = None
            self._port.release(candidate)

    def _all_live_entries(self) -> list[_LiveEntry[CandidateT]]:
        result: list[_LiveEntry[CandidateT]] = []
        seen_ids: set[int] = set()
        for entry in (*self._top, *self._mode.values(), self._a0):
            if entry is None or entry.candidate is None or id(entry) in seen_ids:
                continue
            seen_ids.add(id(entry))
            result.append(entry)
        return result


@dataclass(slots=True)
class _TerminalSeen(Generic[PayloadT]):
    score: FinalScoreV1
    records: tuple[ActionRecordV1, ...]


@dataclass(frozen=True, slots=True)
class _TerminalCandidate(Generic[PayloadT]):
    signature: bytes
    trace: SearchTraceV1[PayloadT]


class _TerminalAccumulator(Generic[CandidateT, PayloadT]):
    def __init__(
        self,
        port: PlannerSearchPortV1[CandidateT, PayloadT],
        counters: PlannerSearchCountersV1,
    ) -> None:
        self._port = port
        self._counters = counters
        self._seen: dict[bytes, _TerminalSeen[PayloadT]] = {}
        self.best: SearchTraceV1[PayloadT] | None = None

    def offer(self, candidate: CandidateT) -> None:
        try:
            prepared = self.prepare(candidate)
        except Exception:
            self._port.release(candidate)
            raise
        self.offer_prepared(candidate, prepared=prepared)

    def prepare(
        self, candidate: CandidateT, *, signature: bytes | None = None
    ) -> _TerminalCandidate[PayloadT]:
        """Build the exact state signature and detached terminal evaluation."""

        if signature is None:
            signature = _validated_state_bytes(self._port, candidate, terminal=True)
        elif not isinstance(signature, bytes) or not signature:
            raise PlannerSearchError(
                "state_signature_error",
                "terminal state signature must be non-empty bytes",
            )
        trace = self._port.terminal_trace(candidate)
        if not isinstance(trace, SearchTraceV1):
            raise PlannerSearchError(
                "search_incomplete", "terminal_trace returned the wrong type"
            )
        candidate_records = tuple(self._port.action_records(candidate))
        request_trace_payload(candidate_records)
        if not _same_action_records(trace.action_records, candidate_records):
            raise PlannerSearchError(
                "search_incomplete",
                "terminal trace differs from its candidate lineage",
            )
        return _TerminalCandidate(signature=signature, trace=trace)

    def offer_prepared(
        self,
        candidate: CandidateT,
        *,
        prepared: _TerminalCandidate[PayloadT],
    ) -> None:
        """Stream one prepared terminal and release its environment immediately."""

        try:
            signature = prepared.signature
            trace = prepared.trace
            prior = self._seen.get(signature)
            if prior is not None:
                self._counters.exact_duplicates += 1
                if compare_final_scores(trace.final_score, prior.score) != 0:
                    raise PlannerSearchError(
                        "state_signature_error",
                        "equal terminal state bytes produced different final scores",
                    )
                if compare_action_lineage(trace.action_records, prior.records) <= 0:
                    return
            self._seen[signature] = _TerminalSeen(
                score=trace.final_score,
                records=trace.action_records,
            )
            if self.best is None or compare_searched_traces(trace, self.best) > 0:
                self.best = trace
        finally:
            self._port.release(candidate)


def _records_for_candidate(
    port: PlannerSearchPortV1[CandidateT, PayloadT], candidate: CandidateT
) -> tuple[ActionRecordV1, ...]:
    records = tuple(port.action_records(candidate))
    request_trace_payload(records)
    return records


def _release_many(
    port: PlannerSearchPortV1[CandidateT, PayloadT], candidates: list[CandidateT]
) -> None:
    released: set[int] = set()
    for candidate in candidates:
        if id(candidate) in released:
            continue
        released.add(id(candidate))
        port.release(candidate)


def run_primary_beam(
    port: PlannerSearchPortV1[CandidateT, PayloadT],
    *,
    a0_records: tuple[ActionRecordV1, ...],
    expected_decisions: int,
    counters: PlannerSearchCountersV1,
) -> SearchTraceV1[PayloadT]:
    """Run the frozen macro beam and return its best searched terminal trace."""

    decisions = _integer("expected_decisions", expected_decisions, minimum=1)
    incumbent = tuple(a0_records)
    request_trace_payload(incumbent)
    if len(incumbent) != decisions or any(
        record.source_phase != "a0_capture" or record.source_code != -1
        for record in incumbent
    ):
        raise PlannerSearchError(
            "search_incomplete", "A0 records are incomplete or have invalid lineage"
        )

    macro_count = (decisions + MACRO_DECISIONS - 1) // MACRO_DECISIONS
    frontier = [port.initial_candidate()]
    accumulator: _FrontierAccumulator[CandidateT, PayloadT] | None = None
    try:
        root_records = _records_for_candidate(port, frontier[0])
        if root_records:
            raise PlannerSearchError(
                "search_incomplete", "the primary-search root must have an empty trace"
            )

        terminal_best: SearchTraceV1[PayloadT] | None = None
        for macro_depth in range(1, macro_count + 1):
            start = (macro_depth - 1) * MACRO_DECISIONS
            stop = min(start + MACRO_DECISIONS, decisions)
            terminal = macro_depth == macro_count

            injected = port.advance_a0_cursor(
                macro_depth=macro_depth,
                start_decision=start,
                stop_decision=stop,
                terminal=terminal,
            )
            counters.a0_cursor_validations += 1
            if terminal:
                if injected is not None:
                    port.release(injected)
                    raise PlannerSearchError(
                        "search_incomplete",
                        "terminal A0 cursor validation returned an injected clone",
                    )
                terminal_accumulator = _TerminalAccumulator(port, counters)
            else:
                if injected is None:
                    raise PlannerSearchError(
                        "search_incomplete", "nonterminal A0 injection is missing"
                    )
                try:
                    injected_records = _records_for_candidate(port, injected)
                except Exception:
                    port.release(injected)
                    raise
                if not _same_action_records(injected_records, incumbent[:stop]):
                    port.release(injected)
                    raise PlannerSearchError(
                        "search_incomplete", "injected A0 prefix differs from capture"
                    )
                accumulator = _FrontierAccumulator(port, counters)
                accumulator.offer(injected, exact_a0_prefix=True)
                counters.a0_injections += 1

            for parent in frontier:
                parent_records = _records_for_candidate(port, parent)
                if len(parent_records) != start:
                    raise PlannerSearchError(
                        "search_incomplete",
                        "frontier parent has the wrong decision depth",
                    )
                for template_id in TEMPLATE_IDS:
                    if counters.attempted_edges >= MAX_ATTEMPTED_EDGES:
                        raise PlannerSearchError(
                            "attempted_edge_limit",
                            "another macro edge would exceed the frozen ceiling",
                        )
                    counters.attempted_edges += 1
                    child = port.expand_macro(
                        parent,
                        template_id=template_id,
                        macro_depth=macro_depth,
                        start_decision=start,
                        stop_decision=stop,
                    )
                    try:
                        child_records = _records_for_candidate(port, child)
                        if len(child_records) != stop:
                            raise PlannerSearchError(
                                "search_incomplete",
                                "completed macro has the wrong decision depth",
                            )
                        if not _same_action_records(
                            child_records[:start], parent_records
                        ):
                            raise PlannerSearchError(
                                "search_incomplete",
                                "macro expansion changed its parent prefix",
                            )
                        for record in child_records[start:stop]:
                            if (
                                record.source_phase != "beam"
                                or record.source_code != template_id
                            ):
                                raise PlannerSearchError(
                                    "search_incomplete",
                                    "macro expansion has incorrect beam lineage",
                                )
                    except Exception:
                        port.release(child)
                        raise

                    try:
                        signature = _validated_state_bytes(
                            port, child, terminal=terminal
                        )
                    except BaseException:
                        port.release(child)
                        raise

                    counters.completed_edges += 1
                    counters.generated_candidates += 1
                    try:
                        port.resource_checkpoint("completed_edge")
                    except BaseException:
                        port.release(child)
                        raise

                    try:
                        if terminal:
                            prepared_terminal = terminal_accumulator.prepare(
                                child, signature=signature
                            )
                        else:
                            assert accumulator is not None
                            prepared_frontier = accumulator.prepare(
                                child,
                                exact_a0_prefix=False,
                                signature=signature,
                                records=child_records,
                            )
                    except BaseException:
                        port.release(child)
                        raise
                    if terminal:
                        terminal_accumulator.offer_prepared(
                            child, prepared=prepared_terminal
                        )
                    else:
                        assert accumulator is not None
                        accumulator.offer_prepared(child, summary=prepared_frontier)

            counters.completed_macro_depths += 1
            if terminal:
                terminal_best = terminal_accumulator.best
            else:
                assert accumulator is not None
                next_frontier = accumulator.take_frontier()
                accumulator = None
                if not next_frontier:
                    raise PlannerSearchError(
                        "search_incomplete", "a nonterminal beam cut was empty"
                    )
                counters.frontier_retained_total += len(next_frontier)
                counters.peak_frontier = max(counters.peak_frontier, len(next_frontier))
                _release_many(port, frontier)
                frontier = next_frontier

        if terminal_best is None:
            raise PlannerSearchError(
                "search_incomplete", "no searched terminal trace survived"
            )
        return terminal_best
    finally:
        if accumulator is not None:
            accumulator.close()
        _release_many(port, frontier)


def _validate_complete_trial(
    trial: SearchTraceV1[PayloadT], *, expected_decisions: int
) -> None:
    """Validate that a coordinate call returned one complete terminal trace."""

    if not isinstance(trial, SearchTraceV1):
        raise PlannerSearchError(
            "coordinate_error", "coordinate_trial returned the wrong type"
        )
    if len(trial.action_records) != expected_decisions:
        raise PlannerSearchError(
            "coordinate_error", "coordinate trial has an incomplete trace"
        )


def _validate_trial_lineage(
    trial: SearchTraceV1[PayloadT],
    *,
    base: SearchTraceV1[PayloadT],
    start: int,
    stop: int,
    template_id: int,
    source_phase: Literal["coarse_polish", "fine_polish"],
) -> None:
    if not _same_action_records(
        trial.action_records[:start], base.action_records[:start]
    ):
        raise PlannerSearchError(
            "coordinate_error", "coordinate trial changed its fixed prefix"
        )
    if not _same_action_records(
        trial.action_records[stop:], base.action_records[stop:]
    ):
        raise PlannerSearchError(
            "coordinate_error", "coordinate trial changed its fixed suffix"
        )
    for record in trial.action_records[start:stop]:
        if record.source_phase != source_phase or record.source_code != template_id:
            raise PlannerSearchError(
                "coordinate_error", "coordinate replacement lineage is invalid"
            )


def _coordinate_sweep(
    port: PlannerSearchPortV1[CandidateT, PayloadT],
    base: SearchTraceV1[PayloadT],
    *,
    expected_decisions: int,
    block_decisions: int,
    source_phase: Literal["coarse_polish", "fine_polish"],
    counters: PlannerSearchCountersV1,
) -> SearchTraceV1[PayloadT]:
    current = base
    for start in range(0, expected_decisions, block_decisions):
        stop = min(start + block_decisions, expected_decisions)
        block_start = current
        best_trial: SearchTraceV1[PayloadT] | None = None
        for template_id in TEMPLATE_IDS:
            counters.coordinate_trials_attempted += 1
            trial = port.coordinate_trial(
                block_start,
                start_decision=start,
                stop_decision=stop,
                template_id=template_id,
                source_phase=source_phase,
            )
            _validate_complete_trial(trial, expected_decisions=expected_decisions)
            counters.coordinate_trials_completed += 1
            port.resource_checkpoint("completed_coordinate_trial")
            _validate_trial_lineage(
                trial,
                base=block_start,
                start=start,
                stop=stop,
                template_id=template_id,
                source_phase=source_phase,
            )
            if best_trial is None or compare_searched_traces(trial, best_trial) > 0:
                best_trial = trial
        if best_trial is None:  # pragma: no cover - frozen library is non-empty
            raise PlannerSearchError(
                "coordinate_error", "coordinate block produced no trials"
            )
        if compare_final_scores(best_trial.final_score, block_start.final_score) > 0:
            current = best_trial
            counters.coordinate_trials_accepted += 1
    return current


def polish_traces(
    port: PlannerSearchPortV1[CandidateT, PayloadT],
    *,
    a0: SearchTraceV1[PayloadT],
    beam_winner: SearchTraceV1[PayloadT] | None,
    expected_decisions: int,
    counters: PlannerSearchCountersV1,
) -> SearchTraceV1[PayloadT]:
    """Run the frozen one-pass coarse and fine coordinate polishing stages."""

    decisions = _integer("expected_decisions", expected_decisions, minimum=1)
    if len(a0.action_records) != decisions or not is_unchanged_a0_trace(a0, a0):
        raise PlannerSearchError(
            "coordinate_error", "A0 polishing origin is incomplete or malformed"
        )
    origins = [a0]
    if beam_winner is not None:
        if len(beam_winner.action_records) != decisions:
            raise PlannerSearchError(
                "coordinate_error", "beam polishing origin is incomplete"
            )
        if concrete_request_bytes(beam_winner.action_records) == concrete_request_bytes(
            a0.action_records
        ):
            if compare_final_scores(beam_winner.final_score, a0.final_score) != 0:
                raise PlannerSearchError(
                    "coordinate_error",
                    "byte-identical A0 and beam traces have different scores",
                )
        else:
            origins.append(beam_winner)

    coarse = [
        _coordinate_sweep(
            port,
            origin,
            expected_decisions=decisions,
            block_decisions=COARSE_BLOCK_DECISIONS,
            source_phase="coarse_polish",
            counters=counters,
        )
        for origin in origins
    ]
    selected = coarse[0]
    for candidate in coarse[1:]:
        if compare_traces_with_a0_priority(candidate, selected, a0=a0) > 0:
            selected = candidate

    return _coordinate_sweep(
        port,
        selected,
        expected_decisions=decisions,
        block_decisions=FINE_BLOCK_DECISIONS,
        source_phase="fine_polish",
        counters=counters,
    )


__all__ = [
    "BEAM_WIDTH",
    "COARSE_BLOCK_DECISIONS",
    "FINE_BLOCK_DECISIONS",
    "MACRO_DECISIONS",
    "MAX_ATTEMPTED_EDGES",
    "MODE_READINESS_ORDER",
    "PlannerSearchCountersV1",
    "PlannerSearchError",
    "PlannerSearchPortV1",
    "SearchTraceV1",
    "TEMPLATE_IDS",
    "IntermediateRankV1",
    "canonical_action_record_bytes",
    "compare_intermediate_ranks",
    "compare_searched_traces",
    "compare_traces_with_a0_priority",
    "concrete_request_bytes",
    "is_unchanged_a0_trace",
    "polish_traces",
    "run_primary_beam",
]

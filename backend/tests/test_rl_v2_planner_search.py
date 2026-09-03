from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Literal

import pytest

import plant_backend.rl.v2._planner_search as planner_search
from plant_backend.rl.v2._planner_search import (
    MAX_ATTEMPTED_EDGES,
    MODE_READINESS_ORDER,
    IntermediateRankV1,
    PlannerSearchCountersV1,
    PlannerSearchError,
    SearchTraceV1,
    _FrontierAccumulator,
    canonical_action_record_bytes,
    compare_intermediate_ranks,
    compare_searched_traces,
    compare_traces_with_a0_priority,
    concrete_request_bytes,
    is_unchanged_a0_trace,
    polish_traces,
    run_primary_beam,
)
from plant_backend.rl.v2.planner_schema import (
    A0_CONTROLLER_SHA256,
    ACTION_LIBRARY_SHA256,
    MODE_PROTECTION_ORDER,
    ActionRecordV1,
    FinalScoreV1,
)


GUIDANCE_SHA256 = "b" * 64


def _record(
    decision_index: int,
    *,
    phase: str,
    code: int,
    request_value: float | None = None,
) -> ActionRecordV1:
    if code in (-1, 0):
        binding = A0_CONTROLLER_SHA256
    elif code == 1:
        binding = GUIDANCE_SHA256
    else:
        binding = ACTION_LIBRARY_SHA256
    value = (
        (0.0 if code == -1 else (code + 1) / 100.0)
        if request_value is None
        else request_value
    )
    return ActionRecordV1(
        decision_index=decision_index,
        source_phase=phase,
        source_code=code,
        source_binding_sha256=binding,
        controller_invalid=False,
        battery_share=value,
        electrolyser_share=0.0,
        dac_share=0.0,
        methanator_share=0.0,
        battery_support_kw=0.0,
    )


def _a0_records(count: int) -> tuple[ActionRecordV1, ...]:
    return tuple(_record(index, phase="a0_capture", code=-1) for index in range(count))


def _score(delivered: float = 0.0, **changes: object) -> FinalScoreV1:
    values: dict[str, object] = {
        "delivered_kg": delivered,
        "service_numerator": 0,
        "service_denominator": 0,
        "worst_fill": 1.0,
        "terminal_ch4_equivalent_inventory_kg": 0.0,
        "terminal_battery_energy_kwh": 0.0,
        "forced_stops": 0,
        "aborted_starts": 0,
        "cold_starts": 0,
        "curtailed_kwh": 0.0,
        "battery_throughput_kwh": 0.0,
    }
    values.update(changes)
    return FinalScoreV1(**values)


def _rank(
    value: float = 0.0,
    *,
    mode: str = "running",
    **changes: object,
) -> IntermediateRankV1:
    values: dict[str, object] = {
        "delivered_plus_deadline_readiness_kg": value,
        "delivered_kg": value,
        "service_numerator": 0,
        "service_denominator": 0,
        "worst_fill": 1.0,
        "ch4_inventory_kg": 0.0,
        "feedstock_potential_kg": 0.0,
        "battery_energy_kwh": 0.0,
        "commitment_mode": mode,
        "startup_remaining_s": 0.0,
        "cooldown_remaining_s": 0.0,
        "shortfall_elapsed_s": 0.0,
        "forced_stops": 0,
        "aborted_starts": 0,
        "cold_starts": 0,
        "curtailed_kwh": 0.0,
        "battery_throughput_kwh": 0.0,
    }
    values.update(changes)
    return IntermediateRankV1(**values)  # type: ignore[arg-type]


@dataclass(slots=True)
class _Candidate:
    records: tuple[ActionRecordV1, ...]
    signature: bytes
    rank: IntermediateRankV1
    score: FinalScoreV1
    released: bool = False


ExpandRule = Callable[
    ["_FakePort", _Candidate, int, int, int, int],
    tuple[bytes, IntermediateRankV1, FinalScoreV1],
]


class _FakePort:
    """Outcome-free deterministic port; it never imports or models a plant."""

    def __init__(
        self,
        *,
        a0_records: tuple[ActionRecordV1, ...] = (),
        expand_rule: ExpandRule | None = None,
        coordinate_deltas: dict[tuple[str, int, int], float] | None = None,
        fail_resource_at: int | None = None,
        fail_state_at: int | None = None,
        fail_rank_at: int | None = None,
        fail_terminal_at: int | None = None,
    ) -> None:
        self.a0_records = a0_records
        self.expand_rule = expand_rule
        self.coordinate_deltas = coordinate_deltas or {}
        self.fail_resource_at = fail_resource_at
        self.fail_state_at = fail_state_at
        self.fail_rank_at = fail_rank_at
        self.fail_terminal_at = fail_terminal_at
        self.expand_calls: list[
            tuple[int, bytes, int, int, int, tuple[ActionRecordV1, ...]]
        ] = []
        self.coordinate_calls: list[tuple[str, int, int, int, bytes]] = []
        self.resource_calls: list[str] = []
        self.state_calls = 0
        self.rank_calls = 0
        self.terminal_calls = 0
        self._active: set[int] = set()
        self.peak_active = 0

    @property
    def active(self) -> int:
        return len(self._active)

    def _own(self, candidate: _Candidate) -> _Candidate:
        identity = id(candidate)
        assert identity not in self._active
        self._active.add(identity)
        self.peak_active = max(self.peak_active, len(self._active))
        return candidate

    def make_candidate(
        self,
        *,
        records: tuple[ActionRecordV1, ...],
        signature: bytes,
        rank: IntermediateRankV1,
        score: FinalScoreV1 | None = None,
    ) -> _Candidate:
        return self._own(
            _Candidate(
                records=records,
                signature=signature,
                rank=rank,
                score=score or _score(rank.delivered_kg),
            )
        )

    def initial_candidate(self) -> _Candidate:
        return self.make_candidate(
            records=(), signature=b"root", rank=_rank(), score=_score()
        )

    def advance_a0_cursor(
        self,
        *,
        macro_depth: int,
        start_decision: int,
        stop_decision: int,
        terminal: bool,
    ) -> _Candidate | None:
        del start_decision
        if terminal:
            return None
        records = self.a0_records[:stop_decision]
        return self.make_candidate(
            records=records,
            signature=f"a0-{macro_depth}".encode(),
            rank=_rank(0.0, mode="off"),
            score=_score(),
        )

    def expand_macro(
        self,
        parent: _Candidate,
        *,
        template_id: int,
        macro_depth: int,
        start_decision: int,
        stop_decision: int,
    ) -> _Candidate:
        self.expand_calls.append(
            (
                macro_depth,
                parent.signature,
                template_id,
                start_decision,
                stop_decision,
                parent.records,
            )
        )
        records = parent.records + tuple(
            _record(index, phase="beam", code=template_id)
            for index in range(start_decision, stop_decision)
        )
        if self.expand_rule is None:
            signature = (
                parent.signature
                + b"/"
                + str(macro_depth).encode()
                + b":"
                + str(template_id).encode()
            )
            value = parent.rank.delivered_kg + float(template_id + 1)
            mode = MODE_PROTECTION_ORDER[template_id % len(MODE_PROTECTION_ORDER)]
            rank = _rank(value, mode=mode)
            score = _score(value)
        else:
            signature, rank, score = self.expand_rule(
                self,
                parent,
                template_id,
                macro_depth,
                start_decision,
                stop_decision,
            )
        return self.make_candidate(
            records=records,
            signature=signature,
            rank=rank,
            score=score,
        )

    def state_bytes(self, candidate: _Candidate) -> bytes:
        self.state_calls += 1
        if self.fail_state_at is not None and self.state_calls == self.fail_state_at:
            raise PlannerSearchError("state_signature_error", "synthetic signature")
        return candidate.signature

    def intermediate_rank(self, candidate: _Candidate) -> IntermediateRankV1:
        self.rank_calls += 1
        if self.fail_rank_at is not None and self.rank_calls == self.fail_rank_at:
            raise PlannerSearchError("state_signature_error", "synthetic rank")
        return candidate.rank

    def action_records(self, candidate: _Candidate) -> tuple[ActionRecordV1, ...]:
        return candidate.records

    def terminal_trace(self, candidate: _Candidate) -> SearchTraceV1[bytes]:
        self.terminal_calls += 1
        if (
            self.fail_terminal_at is not None
            and self.terminal_calls == self.fail_terminal_at
        ):
            raise PlannerSearchError("search_incomplete", "synthetic terminal")
        return SearchTraceV1(
            action_records=candidate.records,
            final_score=candidate.score,
            payload=candidate.signature,
        )

    def coordinate_trial(
        self,
        base: SearchTraceV1[bytes],
        *,
        start_decision: int,
        stop_decision: int,
        template_id: int,
        source_phase: Literal["coarse_polish", "fine_polish"],
    ) -> SearchTraceV1[bytes]:
        self.coordinate_calls.append(
            (
                source_phase,
                start_decision,
                stop_decision,
                template_id,
                concrete_request_bytes(base.action_records),
            )
        )
        replacement = tuple(
            _record(index, phase=source_phase, code=template_id)
            for index in range(start_decision, stop_decision)
        )
        records = (
            base.action_records[:start_decision]
            + replacement
            + base.action_records[stop_decision:]
        )
        delta = self.coordinate_deltas.get(
            (source_phase, start_decision, template_id), 0.0
        )
        score = replace(
            base.final_score,
            delivered_kg=base.final_score.delivered_kg + delta,
        )
        return SearchTraceV1(
            action_records=records,
            final_score=score,
            payload=(source_phase + f":{start_decision}:{template_id}").encode(),
        )

    def release(self, candidate: _Candidate) -> None:
        assert not candidate.released, "candidate released twice"
        candidate.released = True
        self._active.remove(id(candidate))

    def resource_checkpoint(
        self,
        checkpoint: Literal["completed_edge", "completed_coordinate_trial"],
    ) -> None:
        self.resource_calls.append(checkpoint)
        if (
            self.fail_resource_at is not None
            and len(self.resource_calls) == self.fail_resource_at
        ):
            raise PlannerSearchError("wall_time_limit", "synthetic resource limit")


def _trace(
    records: tuple[ActionRecordV1, ...],
    *,
    delivered: float = 0.0,
) -> SearchTraceV1[bytes]:
    return SearchTraceV1(
        action_records=records,
        final_score=_score(delivered),
        payload=b"terminal",
    )


def test_intermediate_rank_and_complete_trace_comparators_are_exact() -> None:
    base = _rank(1.0, service_numerator=1, service_denominator=2)
    assert (
        compare_intermediate_ranks(
            replace(
                base,
                delivered_plus_deadline_readiness_kg=math.nextafter(1.0, math.inf),
            ),
            base,
        )
        == 1
    )
    assert (
        compare_intermediate_ranks(
            replace(base, service_numerator=2, service_denominator=3), base
        )
        == 1
    )
    assert compare_intermediate_ranks(replace(base, worst_fill=0.9), base) == -1
    assert (
        compare_intermediate_ranks(
            replace(base, commitment_mode="hot_standby"),
            replace(base, commitment_mode="cold_starting"),
        )
        == 1
    )
    for lower_rank in (
        replace(base, startup_remaining_s=1),
        replace(base, cooldown_remaining_s=1),
        replace(base, shortfall_elapsed_s=1),
        replace(base, forced_stops=1),
        replace(base, aborted_starts=1),
        replace(base, cold_starts=1),
        replace(base, curtailed_kwh=1),
        replace(base, battery_throughput_kwh=1),
    ):
        assert compare_intermediate_ranks(lower_rank, base) == -1

    lower_code = _trace((_record(0, phase="beam", code=2),))
    higher_code = _trace((_record(0, phase="beam", code=3),))
    assert compare_searched_traces(lower_code, higher_code) == 1
    a0 = _trace(_a0_records(1))
    assert compare_traces_with_a0_priority(a0, lower_code, a0=a0) == 1
    assert MODE_READINESS_ORDER == (
        "running",
        "hot_standby",
        "hot_starting",
        "cold_starting",
        "off",
        "cooldown",
    )


def test_primary_terminal_stream_matches_exhaustive_one_macro_choice() -> None:
    records = _a0_records(4)
    port = _FakePort(a0_records=records)
    counters = PlannerSearchCountersV1()

    winner = run_primary_beam(
        port, a0_records=records, expected_decisions=4, counters=counters
    )

    assert {record.source_code for record in winner.action_records} == {14}
    assert counters.completed_macro_depths == 1
    assert counters.attempted_edges == counters.completed_edges == 15
    assert counters.generated_candidates == 15
    assert counters.a0_cursor_validations == 1
    assert counters.a0_injections == 0
    assert counters.frontier_retained_total == 0
    assert port.resource_calls == ["completed_edge"] * 15
    assert port.active == 0


def test_shortened_final_macro_and_counter_formula_are_exact() -> None:
    records = _a0_records(5)
    port = _FakePort(a0_records=records)
    counters = PlannerSearchCountersV1()

    winner = run_primary_beam(
        port, a0_records=records, expected_decisions=5, counters=counters
    )

    assert len(winner.action_records) == 5
    assert tuple(record.source_code for record in winner.action_records) == (
        14,
        14,
        14,
        14,
        14,
    )
    final_calls = [call for call in port.expand_calls if call[0] == 2]
    assert final_calls
    assert {(call[3], call[4]) for call in final_calls} == {(4, 5)}
    assert counters.completed_macro_depths == 2
    assert counters.a0_cursor_validations == 2
    assert counters.a0_injections == 1
    assert counters.frontier_retained_total == 16
    assert counters.attempted_edges == 15 * (1 + counters.frontier_retained_total)
    assert counters.completed_edges == counters.attempted_edges
    assert port.active == 0


def test_two_macro_search_matches_an_independent_exhaustive_enumerator() -> None:
    def non_greedy_tree(
        port: _FakePort,
        parent: _Candidate,
        template_id: int,
        macro_depth: int,
        start: int,
        stop: int,
    ) -> tuple[bytes, IntermediateRankV1, FinalScoreV1]:
        del port, start, stop
        signature = parent.signature + f"/{macro_depth}:{template_id}".encode()
        if macro_depth == 1:
            # Deliberately prefer low template IDs in the pruning rank.
            return signature, _rank(float(14 - template_id)), _score()
        first_code = parent.records[0].source_code
        delivered = 100.0 - 3.0 * abs(first_code - 4) - abs(template_id - 11)
        return signature, _rank(), _score(delivered)

    records = _a0_records(5)
    port = _FakePort(a0_records=records, expand_rule=non_greedy_tree)
    counters = PlannerSearchCountersV1()
    winner = run_primary_beam(
        port, a0_records=records, expected_decisions=5, counters=counters
    )

    exhaustive = [
        (100.0 - 3.0 * abs(first - 4) - abs(second - 11), first, second)
        for first in (-1, *range(15))
        for second in range(15)
    ]
    expected_score, expected_first, expected_second = max(exhaustive)
    assert winner.final_score.delivered_kg == expected_score
    assert {record.source_code for record in winner.action_records[:4]} == {
        expected_first
    }
    assert winner.action_records[4].source_code == expected_second
    assert counters.frontier_retained_total == 16
    assert port.active == 0


def test_a0_and_each_commitment_mode_survive_a_pressured_cut() -> None:
    records = _a0_records(9)
    port = _FakePort(a0_records=records)
    counters = PlannerSearchCountersV1()

    run_primary_beam(port, a0_records=records, expected_decisions=9, counters=counters)

    depth_three_parents = [
        call for call in port.expand_calls if call[0] == 3 and call[2] == 0
    ]
    assert sum(call[0] == 3 for call in port.expand_calls) == 24 * 15
    assert len(depth_three_parents) == 24
    assert canonical_action_record_bytes(
        depth_three_parents[0][5]
    ) == canonical_action_record_bytes(records[:8])
    selected_modes = set()
    for call in depth_three_parents:
        parent_signature = call[1]
        if parent_signature.startswith(b"a0-"):
            selected_modes.add("off")
            continue
        final_template = int(parent_signature.rsplit(b":", 1)[1])
        selected_modes.add(
            MODE_PROTECTION_ORDER[final_template % len(MODE_PROTECTION_ORDER)]
        )
    assert selected_modes == set(MODE_PROTECTION_ORDER)
    assert counters.a0_injections == 2
    assert counters.peak_frontier == 24
    assert port.active == 0


def test_frontier_order_is_a0_then_modes_then_global_rank_fill() -> None:
    port = _FakePort()
    counters = PlannerSearchCountersV1()
    accumulator = _FrontierAccumulator(port, counters)
    a0 = port.make_candidate(
        records=_a0_records(1), signature=b"a0", rank=_rank(mode="off")
    )
    accumulator.offer(a0, exact_a0_prefix=True)

    protected: dict[str, _Candidate] = {}
    protected_values = {
        "off": 100.0,
        "cold_starting": 99.0,
        "hot_standby": 98.0,
        "hot_starting": 97.0,
        "cooldown": 96.0,
    }
    for index, mode in enumerate(reversed(tuple(protected_values))):
        candidate = port.make_candidate(
            records=(_record(0, phase="beam", code=index + 2),),
            signature=f"mode-{mode}".encode(),
            rank=_rank(protected_values[mode], mode=mode),
        )
        protected[mode] = candidate
        accumulator.offer(candidate, exact_a0_prefix=False)

    globals_by_rank: list[_Candidate] = []
    for index in reversed(range(30)):
        candidate = port.make_candidate(
            records=(_record(0, phase="beam", code=2),),
            signature=f"global-{index}".encode(),
            rank=_rank(1000.0 - index, mode="running"),
        )
        globals_by_rank.append(candidate)
        accumulator.offer(candidate, exact_a0_prefix=False)
    globals_by_rank.reverse()

    frontier = accumulator.take_frontier()
    expected = [
        a0,
        protected["off"],
        protected["cold_starting"],
        globals_by_rank[0],
        protected["hot_standby"],
        protected["hot_starting"],
        protected["cooldown"],
        *globals_by_rank[1:18],
    ]
    assert len(frontier) == 24
    assert all(actual is wanted for actual, wanted in zip(frontier, expected))
    for candidate in frontier:
        port.release(candidate)
    assert port.active == 0


def test_streaming_frontier_reservoir_stays_within_live_reference_proxy_bound() -> None:
    port = _FakePort()
    counters = PlannerSearchCountersV1()
    parents = [
        port.make_candidate(
            records=(), signature=f"parent-{index}".encode(), rank=_rank()
        )
        for index in range(24)
    ]
    accumulator = _FrontierAccumulator(port, counters)
    a0 = port.make_candidate(
        records=_a0_records(1), signature=b"a0", rank=_rank(mode="off")
    )
    accumulator.offer(a0, exact_a0_prefix=True)

    for index in range(500):
        candidate = port.make_candidate(
            records=(_record(0, phase="beam", code=index % 15),),
            signature=f"candidate-{index}".encode(),
            rank=_rank(
                float(index + 1),
                mode=MODE_PROTECTION_ORDER[index % len(MODE_PROTECTION_ORDER)],
            ),
        )
        accumulator.offer(candidate, exact_a0_prefix=False)
        assert len(accumulator._all_live_entries()) <= 31

    # This synthetic port models 24 current parents, up to 31 next-reservoir
    # references, and the transient child. The separate rolling cursor is absent.
    assert port.peak_active <= 24 + 31 + 1
    frontier = accumulator.take_frontier()
    assert len(frontier) == 24
    for candidate in (*parents, *frontier):
        port.release(candidate)
    assert port.active == 0


def test_same_state_dedup_is_insertion_independent_and_releases_losers() -> None:
    def selected_code(order: tuple[int, int]) -> tuple[int, int]:
        port = _FakePort()
        counters = PlannerSearchCountersV1()
        accumulator = _FrontierAccumulator(port, counters)
        a0 = port.make_candidate(
            records=_a0_records(1),
            signature=b"a0",
            rank=_rank(mode="off"),
        )
        accumulator.offer(a0, exact_a0_prefix=True)
        for code in order:
            candidate = port.make_candidate(
                records=(_record(0, phase="beam", code=code),),
                signature=b"duplicate",
                rank=_rank(1.0),
            )
            accumulator.offer(candidate, exact_a0_prefix=False)
        frontier = accumulator.take_frontier()
        searched = next(
            candidate for candidate in frontier if candidate.signature == b"duplicate"
        )
        result = searched.records[0].source_code, counters.exact_duplicates
        for candidate in frontier:
            port.release(candidate)
        assert port.active == 0
        return result

    assert selected_code((3, 2)) == (2, 1)
    assert selected_code((2, 3)) == (2, 1)


def test_equal_state_bytes_with_different_rank_fail_closed() -> None:
    port = _FakePort()
    counters = PlannerSearchCountersV1()
    accumulator = _FrontierAccumulator(port, counters)
    a0 = port.make_candidate(
        records=_a0_records(1), signature=b"a0", rank=_rank(mode="off")
    )
    accumulator.offer(a0, exact_a0_prefix=True)
    first = port.make_candidate(
        records=(_record(0, phase="beam", code=2),),
        signature=b"same",
        rank=_rank(1.0),
    )
    second = port.make_candidate(
        records=(_record(0, phase="beam", code=3),),
        signature=b"same",
        rank=_rank(2.0),
    )
    accumulator.offer(first, exact_a0_prefix=False)
    with pytest.raises(PlannerSearchError, match="different intermediate") as caught:
        accumulator.offer(second, exact_a0_prefix=False)
    assert caught.value.code == "state_signature_error"
    accumulator.close()
    assert port.active == 0


def test_exact_a0_prefix_wins_an_equal_rank_state_duplicate() -> None:
    port = _FakePort()
    counters = PlannerSearchCountersV1()
    accumulator = _FrontierAccumulator(port, counters)
    a0 = port.make_candidate(
        records=_a0_records(1), signature=b"same", rank=_rank(mode="off")
    )
    searched = port.make_candidate(
        records=(_record(0, phase="beam", code=0, request_value=0.0),),
        signature=b"same",
        rank=_rank(mode="off"),
    )

    accumulator.offer(a0, exact_a0_prefix=True)
    accumulator.offer(searched, exact_a0_prefix=False)
    frontier = accumulator.take_frontier()

    assert len(frontier) == 1
    assert frontier[0] is a0
    assert counters.exact_duplicates == 1
    assert searched.released
    port.release(a0)
    assert port.active == 0


def test_frontier_offer_cleans_candidate_and_seen_reference_on_memory_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port = _FakePort()
    counters = PlannerSearchCountersV1()
    accumulator = _FrontierAccumulator(port, counters)
    a0 = port.make_candidate(
        records=_a0_records(1), signature=b"a0", rank=_rank(mode="off")
    )
    accumulator.offer(a0, exact_a0_prefix=True)
    transient = port.make_candidate(
        records=(_record(0, phase="beam", code=2),),
        signature=b"transient",
        rank=_rank(1.0),
    )

    def injected_memory_error(left: object, right: object) -> int:
        del left, right
        raise MemoryError("synthetic reservoir failure")

    monkeypatch.setattr(planner_search, "_compare_summaries", injected_memory_error)
    with pytest.raises(MemoryError, match="synthetic reservoir failure"):
        accumulator.offer(transient, exact_a0_prefix=False)

    assert transient.released
    assert accumulator._seen[b"transient"].live is None
    assert port.active == 1
    accumulator.close()
    assert port.active == 0


def test_terminal_dedup_streams_by_lineage_without_retaining_candidates() -> None:
    def duplicate_terminal(
        port: _FakePort,
        parent: _Candidate,
        template_id: int,
        macro_depth: int,
        start: int,
        stop: int,
    ) -> tuple[bytes, IntermediateRankV1, FinalScoreV1]:
        del port, parent, template_id, macro_depth, start, stop
        return b"one-terminal-state", _rank(1.0), _score(1.0)

    records = _a0_records(1)
    port = _FakePort(a0_records=records, expand_rule=duplicate_terminal)
    counters = PlannerSearchCountersV1()
    winner = run_primary_beam(
        port, a0_records=records, expected_decisions=1, counters=counters
    )

    assert winner.action_records[0].source_code == 0
    assert counters.exact_duplicates == 14
    assert port.peak_active <= 2  # root plus the one transient terminal child
    assert port.active == 0


def test_terminal_selection_uses_final_score_not_intermediate_rank() -> None:
    def opposed_ranks(
        port: _FakePort,
        parent: _Candidate,
        template_id: int,
        macro_depth: int,
        start: int,
        stop: int,
    ) -> tuple[bytes, IntermediateRankV1, FinalScoreV1]:
        del port, parent, macro_depth, start, stop
        return (
            f"terminal-{template_id}".encode(),
            _rank(float(15 - template_id)),
            _score(float(template_id)),
        )

    records = _a0_records(1)
    port = _FakePort(a0_records=records, expand_rule=opposed_ranks)
    winner = run_primary_beam(
        port,
        a0_records=records,
        expected_decisions=1,
        counters=PlannerSearchCountersV1(),
    )

    assert winner.action_records[0].source_code == 14
    assert winner.final_score.delivered_kg == 14.0
    assert port.rank_calls == 0
    assert port.active == 0


def test_terminal_equal_state_with_different_score_fails_closed_and_releases() -> None:
    def inconsistent_terminal(
        port: _FakePort,
        parent: _Candidate,
        template_id: int,
        macro_depth: int,
        start: int,
        stop: int,
    ) -> tuple[bytes, IntermediateRankV1, FinalScoreV1]:
        del port, parent, macro_depth, start, stop
        return b"one-terminal-state", _rank(), _score(float(template_id))

    records = _a0_records(1)
    port = _FakePort(a0_records=records, expand_rule=inconsistent_terminal)
    counters = PlannerSearchCountersV1()

    with pytest.raises(PlannerSearchError, match="different final scores") as caught:
        run_primary_beam(
            port, a0_records=records, expected_decisions=1, counters=counters
        )

    assert caught.value.code == "state_signature_error"
    assert counters.attempted_edges == counters.completed_edges == 2
    assert counters.exact_duplicates == 1
    assert port.active == 0


def test_resource_failure_releases_transient_child_and_frontier() -> None:
    records = _a0_records(1)
    port = _FakePort(a0_records=records, fail_resource_at=1)
    counters = PlannerSearchCountersV1()

    with pytest.raises(PlannerSearchError, match="resource limit") as caught:
        run_primary_beam(
            port, a0_records=records, expected_decisions=1, counters=counters
        )
    assert caught.value.code == "wall_time_limit"
    assert counters.attempted_edges == 1
    assert counters.completed_edges == counters.generated_candidates == 1
    assert counters.completed_macro_depths == 0
    assert port.active == 0


def test_edge_is_not_completed_until_its_signature_is_built() -> None:
    records = _a0_records(1)
    port = _FakePort(a0_records=records, fail_state_at=1)
    counters = PlannerSearchCountersV1()

    with pytest.raises(PlannerSearchError, match="synthetic signature"):
        run_primary_beam(
            port, a0_records=records, expected_decisions=1, counters=counters
        )

    assert counters.attempted_edges == 1
    assert counters.completed_edges == counters.generated_candidates == 0
    assert port.resource_calls == []
    assert port.active == 0


def test_post_signature_rank_failure_keeps_completed_edge_and_checkpoint() -> None:
    records = _a0_records(5)
    # Rank call one belongs to the injected A0 prefix; call two is the first edge.
    port = _FakePort(a0_records=records, fail_rank_at=2)
    counters = PlannerSearchCountersV1()

    with pytest.raises(PlannerSearchError, match="synthetic rank"):
        run_primary_beam(
            port, a0_records=records, expected_decisions=5, counters=counters
        )

    assert counters.attempted_edges == 1
    assert counters.completed_edges == counters.generated_candidates == 1
    assert port.resource_calls == ["completed_edge"]
    assert counters.a0_injections == 1
    assert port.active == 0


def test_post_signature_terminal_failure_keeps_completed_edge_and_checkpoint() -> None:
    records = _a0_records(1)
    port = _FakePort(a0_records=records, fail_terminal_at=1)
    counters = PlannerSearchCountersV1()

    with pytest.raises(PlannerSearchError, match="synthetic terminal"):
        run_primary_beam(
            port, a0_records=records, expected_decisions=1, counters=counters
        )

    assert counters.attempted_edges == 1
    assert counters.completed_edges == counters.generated_candidates == 1
    assert port.resource_calls == ["completed_edge"]
    assert port.active == 0


def test_a0_injection_counts_only_after_it_enters_the_candidate_pool() -> None:
    records = _a0_records(5)
    port = _FakePort(a0_records=records, fail_state_at=1)
    counters = PlannerSearchCountersV1()

    with pytest.raises(PlannerSearchError, match="synthetic signature"):
        run_primary_beam(
            port, a0_records=records, expected_decisions=5, counters=counters
        )

    assert counters.a0_cursor_validations == 1
    assert counters.a0_injections == 0
    assert counters.attempted_edges == 0
    assert port.active == 0


def test_malformed_a0_injection_is_released_before_failure() -> None:
    class BrokenA0Port(_FakePort):
        def action_records(self, candidate: _Candidate) -> tuple[ActionRecordV1, ...]:
            if candidate.signature.startswith(b"a0-"):
                raise PlannerSearchError("search_incomplete", "synthetic A0 lineage")
            return super().action_records(candidate)

    records = _a0_records(5)
    port = BrokenA0Port(a0_records=records)
    counters = PlannerSearchCountersV1()

    with pytest.raises(PlannerSearchError, match="synthetic A0 lineage"):
        run_primary_beam(
            port, a0_records=records, expected_decisions=5, counters=counters
        )

    assert counters.a0_cursor_validations == 1
    assert counters.a0_injections == 0
    assert port.active == 0


def test_a0_injection_comparison_distinguishes_signed_zero() -> None:
    class PlusZeroInjectionPort(_FakePort):
        def advance_a0_cursor(
            self,
            *,
            macro_depth: int,
            start_decision: int,
            stop_decision: int,
            terminal: bool,
        ) -> _Candidate | None:
            candidate = super().advance_a0_cursor(
                macro_depth=macro_depth,
                start_decision=start_decision,
                stop_decision=stop_decision,
                terminal=terminal,
            )
            if candidate is not None:
                candidate.records = (
                    replace(candidate.records[0], battery_share=0.0),
                    *candidate.records[1:],
                )
            return candidate

    records = tuple(
        _record(index, phase="a0_capture", code=-1, request_value=-0.0)
        for index in range(5)
    )
    port = PlusZeroInjectionPort(a0_records=records)
    counters = PlannerSearchCountersV1()

    with pytest.raises(PlannerSearchError, match="differs from capture"):
        run_primary_beam(
            port, a0_records=records, expected_decisions=5, counters=counters
        )

    assert counters.a0_cursor_validations == 1
    assert counters.a0_injections == 0
    assert counters.attempted_edges == 0
    assert port.active == 0


def test_parent_prefix_comparison_distinguishes_signed_zero() -> None:
    class ChangedPrefixPort(_FakePort):
        def expand_macro(
            self,
            parent: _Candidate,
            *,
            template_id: int,
            macro_depth: int,
            start_decision: int,
            stop_decision: int,
        ) -> _Candidate:
            child = super().expand_macro(
                parent,
                template_id=template_id,
                macro_depth=macro_depth,
                start_decision=start_decision,
                stop_decision=stop_decision,
            )
            if macro_depth == 2:
                child.records = (
                    replace(child.records[0], battery_share=0.0),
                    *child.records[1:],
                )
            return child

    records = tuple(
        _record(index, phase="a0_capture", code=-1, request_value=-0.0)
        for index in range(5)
    )
    port = ChangedPrefixPort(a0_records=records)
    counters = PlannerSearchCountersV1()

    with pytest.raises(PlannerSearchError, match="changed its parent prefix"):
        run_primary_beam(
            port, a0_records=records, expected_decisions=5, counters=counters
        )

    assert counters.attempted_edges == 16
    assert counters.completed_edges == counters.generated_candidates == 15
    assert port.resource_calls == ["completed_edge"] * 15
    assert port.active == 0


def test_terminal_trace_comparison_distinguishes_signed_zero() -> None:
    class ChangedTerminalTracePort(_FakePort):
        def expand_macro(
            self,
            parent: _Candidate,
            *,
            template_id: int,
            macro_depth: int,
            start_decision: int,
            stop_decision: int,
        ) -> _Candidate:
            child = super().expand_macro(
                parent,
                template_id=template_id,
                macro_depth=macro_depth,
                start_decision=start_decision,
                stop_decision=stop_decision,
            )
            child.records = (
                replace(child.records[0], battery_share=0.0),
                *child.records[1:],
            )
            return child

        def terminal_trace(self, candidate: _Candidate) -> SearchTraceV1[bytes]:
            records = (
                replace(candidate.records[0], battery_share=-0.0),
                *candidate.records[1:],
            )
            return SearchTraceV1(
                action_records=records,
                final_score=candidate.score,
                payload=candidate.signature,
            )

    records = _a0_records(1)
    port = ChangedTerminalTracePort(a0_records=records)
    counters = PlannerSearchCountersV1()

    with pytest.raises(PlannerSearchError, match="differs from its candidate"):
        run_primary_beam(
            port, a0_records=records, expected_decisions=1, counters=counters
        )

    assert counters.attempted_edges == counters.completed_edges == 1
    assert port.resource_calls == ["completed_edge"]
    assert port.active == 0


def test_attempted_edge_ceiling_fails_before_clone_and_releases_root() -> None:
    records = _a0_records(1)
    port = _FakePort(a0_records=records)
    counters = PlannerSearchCountersV1(attempted_edges=MAX_ATTEMPTED_EDGES)

    with pytest.raises(PlannerSearchError, match="edge") as caught:
        run_primary_beam(
            port, a0_records=records, expected_decisions=1, counters=counters
        )
    assert caught.value.code == "attempted_edge_limit"
    assert counters.attempted_edges == MAX_ATTEMPTED_EDGES
    assert not port.expand_calls
    assert port.active == 0


def test_coordinate_polish_uses_two_origins_fixed_blocks_and_strict_acceptance() -> (
    None
):
    decisions = 25
    a0 = _trace(_a0_records(decisions), delivered=100.0)
    beam = _trace(
        tuple(_record(index, phase="beam", code=2) for index in range(decisions)),
        delivered=100.0,
    )
    deltas = {
        ("coarse_polish", 0, 5): 1.0,
        ("fine_polish", 0, 7): 1.0,
    }
    port = _FakePort(coordinate_deltas=deltas)
    counters = PlannerSearchCountersV1()

    finalist = polish_traces(
        port,
        a0=a0,
        beam_winner=beam,
        expected_decisions=decisions,
        counters=counters,
    )

    assert finalist.final_score.delivered_kg == 102.0
    assert {record.source_code for record in finalist.action_records[:4]} == {7}
    assert {record.source_code for record in finalist.action_records[4:24]} == {5}
    assert canonical_action_record_bytes(
        finalist.action_records[24:]
    ) == canonical_action_record_bytes(a0.action_records[24:])
    coarse_blocks = {
        (start, stop)
        for phase, start, stop, _, _ in port.coordinate_calls
        if phase == "coarse_polish"
    }
    fine_blocks = {
        (start, stop)
        for phase, start, stop, _, _ in port.coordinate_calls
        if phase == "fine_polish"
    }
    assert coarse_blocks == {(0, 24), (24, 25)}
    assert fine_blocks == {
        (0, 4),
        (4, 8),
        (8, 12),
        (12, 16),
        (16, 20),
        (20, 24),
        (24, 25),
    }
    assert counters.coordinate_trials_attempted == 15 * (2 * 2 + 7)
    assert counters.coordinate_trials_completed == counters.coordinate_trials_attempted
    assert counters.coordinate_trials_accepted == 3
    assert len(port.resource_calls) == counters.coordinate_trials_completed


def _assert_coordinate_fixed_region_distinguishes_signed_zero(
    target_start: int,
    changed_index: int,
    expected_trials: int,
    message: str,
) -> None:
    class ChangedFixedRegionPort(_FakePort):
        def coordinate_trial(
            self,
            base: SearchTraceV1[bytes],
            *,
            start_decision: int,
            stop_decision: int,
            template_id: int,
            source_phase: Literal["coarse_polish", "fine_polish"],
        ) -> SearchTraceV1[bytes]:
            trial = super().coordinate_trial(
                base,
                start_decision=start_decision,
                stop_decision=stop_decision,
                template_id=template_id,
                source_phase=source_phase,
            )
            if (
                source_phase == "coarse_polish"
                and start_decision == target_start
                and template_id == 0
            ):
                records = list(trial.action_records)
                records[changed_index] = replace(
                    records[changed_index], battery_share=0.0
                )
                return SearchTraceV1(
                    action_records=tuple(records),
                    final_score=trial.final_score,
                    payload=trial.payload,
                )
            return trial

    decisions = 25
    records = tuple(
        _record(index, phase="a0_capture", code=-1, request_value=-0.0)
        for index in range(decisions)
    )
    port = ChangedFixedRegionPort()
    counters = PlannerSearchCountersV1()

    with pytest.raises(PlannerSearchError, match=message):
        polish_traces(
            port,
            a0=_trace(records),
            beam_winner=None,
            expected_decisions=decisions,
            counters=counters,
        )

    assert counters.coordinate_trials_attempted == expected_trials
    assert counters.coordinate_trials_completed == expected_trials
    assert port.resource_calls == ["completed_coordinate_trial"] * expected_trials


def test_coordinate_fixed_regions_distinguish_signed_zero_after_checkpoint() -> None:
    _assert_coordinate_fixed_region_distinguishes_signed_zero(
        target_start=0,
        changed_index=24,
        expected_trials=1,
        message="fixed suffix",
    )
    _assert_coordinate_fixed_region_distinguishes_signed_zero(
        target_start=24,
        changed_index=0,
        expected_trials=16,
        message="fixed prefix",
    )


def test_byte_identical_beam_origin_is_elided_and_equal_trials_retain_a0() -> None:
    decisions = 24
    a0 = _trace(_a0_records(decisions), delivered=10.0)
    beam_records = tuple(
        _record(index, phase="beam", code=0, request_value=0.0)
        for index in range(decisions)
    )
    beam = _trace(beam_records, delivered=10.0)
    assert concrete_request_bytes(beam.action_records) == concrete_request_bytes(
        a0.action_records
    )
    port = _FakePort()
    counters = PlannerSearchCountersV1()

    finalist = polish_traces(
        port,
        a0=a0,
        beam_winner=beam,
        expected_decisions=decisions,
        counters=counters,
    )

    assert is_unchanged_a0_trace(finalist, a0)
    assert canonical_action_record_bytes(
        finalist.action_records
    ) == canonical_action_record_bytes(a0.action_records)
    assert counters.coordinate_trials_attempted == 15 * (1 + 6)
    assert counters.coordinate_trials_completed == counters.coordinate_trials_attempted
    assert counters.coordinate_trials_accepted == 0


def test_byte_identical_origins_with_different_scores_fail_closed() -> None:
    a0 = _trace(_a0_records(4), delivered=1.0)
    beam = _trace(
        tuple(
            _record(index, phase="beam", code=0, request_value=0.0)
            for index in range(4)
        ),
        delivered=2.0,
    )
    with pytest.raises(PlannerSearchError, match="different scores") as caught:
        polish_traces(
            _FakePort(),
            a0=a0,
            beam_winner=beam,
            expected_decisions=4,
            counters=PlannerSearchCountersV1(),
        )
    assert caught.value.code == "coordinate_error"

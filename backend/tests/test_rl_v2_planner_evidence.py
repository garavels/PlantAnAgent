from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import time
from typing import Any

import pytest
from plant_sim import PlantConfig

from plant_backend.controllers.base import Constraints
from plant_backend.rl.v2._planner_evidence import (
    PlannerEvidenceError,
    SourceBoundaryV1,
    build_completed_artifact,
    build_runner_failure_artifact,
    capture_runtime_versions,
    capture_source_boundary,
    write_planner_artifact,
)
from plant_backend.rl.v2 import _planner_worker as planner_worker
from plant_backend.rl.v2 import planner
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.planner_schema import (
    FeasiblePlannerResultV1,
    PlannerFailureV1,
    build_search_diagnostics,
    canonical_json,
    planner_artifact_from_dict,
)
from plant_backend.rl.v2.reward import RewardWeightsV2


SHORT = EpisodeConfigV2(horizon_h=0.5, delivery_period_h=0.5)


def _collector(
    root: Path, *, lock_digest: str | None = None
) -> Callable[..., dict[str, Any]]:
    def collect(**values: object) -> dict[str, Any]:
        assert values["root"] == root
        digest = (
            lock_digest or hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
        )
        return {
            "source": {
                "repository_root": str(root),
                "git_revision": "a" * 40,
                "git_dirty": False,
                "tracked_binary_diff_sha256": hashlib.sha256(b"").hexdigest(),
                "tracked_binary_diff_bytes": 0,
                "untracked_files": [],
                "worktree_identity_format": "git-worktree-content-v1",
                "worktree_identity_sha256": "b" * 64,
                "identity_complete": True,
            },
            "dependency_lock": {"path": "uv.lock", "sha256": digest},
        }

    return collect


def _capture(root: Path, **changes: object) -> SourceBoundaryV1:
    values: dict[str, object] = {
        "episode": SHORT,
        "plant": PlantConfig(),
        "constraints": Constraints(),
        "weights": RewardWeightsV2(),
        "root": root,
        "require_clean_source": True,
        "phase": "start_source",
        "collector": _collector(root),
    }
    values.update(changes)
    return capture_source_boundary(**values)  # type: ignore[arg-type]


def _failed_core(
    *,
    evidence_scope: str = "noncanonical_unit_fixture",
) -> FeasiblePlannerResultV1:
    search = build_search_diagnostics(
        completed_macro_depths=0,
        attempted_edges=0,
        completed_edges=0,
        generated_candidates=0,
        a0_cursor_validations=0,
        a0_injections=0,
        exact_duplicates=0,
        frontier_retained_total=0,
        peak_frontier=0,
        coordinate_trials_attempted=0,
        coordinate_trials_completed=0,
        coordinate_trials_accepted=0,
        a0_generation_decisions=0,
        a0_evidence_replay_decisions=0,
        fresh_finalist_replay_decisions=0,
        max_live_environments_observed=0,
        termination_reason="invalid_input",
        elapsed_ns=1,
        peak_rss_bytes=0,
    )
    failure = PlannerFailureV1(
        phase="input_validation",
        code="invalid_input",
        message="fixture",
        completed_macro_depths=0,
        attempted_edges=0,
        completed_edges=0,
        coordinate_trials_completed=0,
        elapsed_ns=1,
        observed_resource_value=None,
    )
    return FeasiblePlannerResultV1(
        schema_version=1,
        planner_kind="best-known-feasible-future-information-v1",
        evidence_scope=evidence_scope,
        privileged_information="complete private episode truth and commitments",
        optimality_claim="none",
        evidence_ready=False,
        provenance_gaps=("clean start/end source and dependency-lock audit required",),
        success=False,
        failure=failure,
        seed=None,
        canonical_episode_verified=False,
        manifest_sha256=None,
        exogenous_episode_sha256=None,
        checkpoint_contract_digest=None,
        relaxation_model_sha256=None,
        relaxation_result_sha256=None,
        relaxation_result=None,
        relaxation_witness=None,
        relaxation_guidance=None,
        settings=None,
        search=search,
        a0_incumbent=None,
        searched_finalist=None,
        selected_role=None,
        comparison=None,
    )


def test_source_boundary_excludes_paths_and_rechecks_raw_lock(tmp_path: Path) -> None:
    repository = (tmp_path / "repo").resolve()
    repository.mkdir()
    lock = repository / "uv.lock"
    lock.write_bytes(b"frozen lock bytes\n")

    boundary = _capture(repository)
    source = boundary.source.value()

    assert isinstance(source, dict)
    assert set(source) == {
        "schema_version",
        "git_revision",
        "git_dirty",
        "tracked_binary_diff_sha256",
        "tracked_binary_diff_bytes",
        "untracked_files",
        "worktree_identity_format",
        "worktree_identity_sha256",
        "identity_complete",
    }
    assert "repository_root" not in source
    assert boundary.lock_sha256 == hashlib.sha256(lock.read_bytes()).hexdigest()
    assert boundary.repository_root == repository


def test_source_boundary_failure_retains_requested_end_phase(tmp_path: Path) -> None:
    repository = (tmp_path / "repo").resolve()
    repository.mkdir()
    (repository / "uv.lock").write_bytes(b"lock")

    with pytest.raises(PlannerEvidenceError) as captured:
        _capture(
            repository,
            phase="end_source",
            collector=_collector(repository, lock_digest="c" * 64),
        )

    assert captured.value.phase == "end_source"
    assert captured.value.code == "source_unavailable"


def test_runtime_versions_fill_the_exact_schema() -> None:
    runtime = capture_runtime_versions()
    value = runtime.value()

    assert isinstance(value, dict)
    assert runtime.domain == "planner-runtime-v1"
    assert value["clock_method"] == "time.perf_counter_ns"
    assert value["rss_method"] == (
        "resource.RUSAGE_SELF.ru_maxrss:darwin-bytes:other-kib"
    )
    assert value["pointer_bits"] in {32, 64}
    assert value["byteorder"] in {"little", "big"}


def test_completed_boundaries_wrap_an_ordinary_core_failure(tmp_path: Path) -> None:
    repository = (tmp_path / "repo").resolve()
    repository.mkdir()
    (repository / "uv.lock").write_bytes(b"lock")
    start = _capture(repository)
    end = _capture(repository, phase="end_source")

    artifact = build_completed_artifact(
        core_result=_failed_core(),
        start=start,
        end=end,
        runtime_versions=capture_runtime_versions(),
    )

    assert artifact.success is False
    assert artifact.runner_failure is None
    assert artifact.core_result is not None
    assert artifact.source_boundary_claim is not None
    assert artifact.evidence_ready is False


def test_atomic_writer_reopens_and_revalidates_external_artifact(
    tmp_path: Path,
) -> None:
    repository = (tmp_path / "repo").resolve()
    repository.mkdir()
    external = (tmp_path / "evidence").resolve()
    external.mkdir()
    target = external / "failure.json"
    artifact = build_runner_failure_artifact(
        phase="start_source",
        code="source_unavailable",
        message="synthetic source failure",
        elapsed_ns=7,
    )

    written = write_planner_artifact(
        artifact,
        target,
        repository_root=repository,
    )
    rebuilt = planner_artifact_from_dict(json.loads(written.read_text("utf-8")))

    assert written == target
    assert rebuilt.artifact_sha256 == artifact.artifact_sha256
    assert written.read_text("utf-8") == json.dumps(
        json.loads(written.read_text("utf-8")),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert not list(external.glob("*.tmp"))


def test_writer_rejects_repository_and_relative_destinations(tmp_path: Path) -> None:
    repository = (tmp_path / "repo").resolve()
    repository.mkdir()
    artifact = build_runner_failure_artifact(
        phase="start_source",
        code="source_unavailable",
        message="synthetic source failure",
        elapsed_ns=0,
    )

    with pytest.raises(ValueError, match="outside"):
        write_planner_artifact(
            artifact,
            repository / "artifact.json",
            repository_root=repository,
        )
    with pytest.raises(ValueError, match="absolute"):
        write_planner_artifact(
            artifact,
            Path("artifact.json"),
            repository_root=repository,
        )


class _FakeProcess:
    def __init__(self, stdout: bytes, stderr: bytes = b"", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.timeouts: list[float | None] = []
        self.killed = False

    def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
        self.timeouts.append(timeout)
        return self.stdout, self.stderr

    def kill(self) -> None:
        self.killed = True


def test_parent_runner_validates_child_and_publishes_completed_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = (tmp_path / "repo").resolve()
    repository.mkdir()
    (repository / ".git").mkdir()
    (repository / "uv.lock").write_bytes(b"lock")
    external = (tmp_path / "evidence").resolve()
    external.mkdir()
    core = _failed_core(evidence_scope="privileged_offline_development")
    process = _FakeProcess(canonical_json(core.to_dict()).encode("utf-8"))
    spawn_calls: list[tuple[tuple[str, ...], Path]] = []

    def spawn(command: tuple[str, ...], **values: object) -> _FakeProcess:
        spawn_calls.append((command, values["cwd"]))  # type: ignore[arg-type]
        return process

    def boundary(**values: object) -> SourceBoundaryV1:
        phase = values["phase"]
        return _capture(repository, phase=phase)

    monkeypatch.setattr(planner, "capture_source_boundary", boundary)
    published = planner.run_registered_planner_cell(
        delivery_rate_kg_h=35.0,
        seed=0,
        destination=external / "cell.json",
        repository_root=repository,
        popen_factory=spawn,
        clock_ns=lambda: 100,
    )

    assert published.artifact.success is False
    assert published.artifact.runner_failure is None
    assert published.artifact.core_result is not None
    assert published.artifact.source_boundary_claim is not None
    assert published.path.is_file()
    assert process.timeouts == [planner.EXTERNAL_WATCHDOG_SECONDS]
    assert spawn_calls[0][1] == repository
    assert spawn_calls[0][0][-4:] == (
        "--delivery-rate",
        "35.0",
        "--seed",
        "0",
    )


def test_external_watchdog_kills_child_and_reports_no_core(tmp_path: Path) -> None:
    class TimedOutProcess(_FakeProcess):
        def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
            self.timeouts.append(timeout)
            if len(self.timeouts) == 1:
                raise subprocess.TimeoutExpired("worker", timeout or 0.0)
            return b"discarded", b"solver still running"

    process = TimedOutProcess(b"")
    readings = iter((100, 100, 200))

    with pytest.raises(PlannerEvidenceError) as captured:
        planner._execute_child(
            rate=35.0,
            seed=0,
            repository_root=tmp_path,
            popen_factory=lambda *args, **kwargs: process,
            clock_ns=lambda: next(readings),
            timeout_s=1.0,
        )

    assert process.killed is True
    assert captured.value.phase == "child_execution"
    assert captured.value.code == "wall_time_limit"
    assert captured.value.child_exit_code is None


def test_external_watchdog_rejects_invalid_limit_before_spawn(tmp_path: Path) -> None:
    for timeout in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive and finite"):
            planner._execute_child(
                rate=35.0,
                seed=0,
                repository_root=tmp_path,
                popen_factory=lambda *args, **kwargs: pytest.fail("child was spawned"),
                timeout_s=timeout,
            )


def test_external_watchdog_rejects_late_return_after_deducting_spawn(
    tmp_path: Path,
) -> None:
    process = _FakeProcess(b"ignored")
    readings = iter((0, 200_000_000, 1_100_000_001))

    with pytest.raises(PlannerEvidenceError) as captured:
        planner._execute_child(
            rate=35.0,
            seed=0,
            repository_root=tmp_path,
            popen_factory=lambda *args, **kwargs: process,
            clock_ns=lambda: next(readings),
            timeout_s=1.0,
        )

    assert process.timeouts == [pytest.approx(0.8)]
    assert captured.value.code == "wall_time_limit"


def test_external_watchdog_kills_child_if_clock_moves_backwards(
    tmp_path: Path,
) -> None:
    process = _FakeProcess(b"ignored")
    readings = iter((200, 100))

    with pytest.raises(PlannerEvidenceError) as captured:
        planner._execute_child(
            rate=35.0,
            seed=0,
            repository_root=tmp_path,
            popen_factory=lambda *args, **kwargs: process,
            clock_ns=lambda: next(readings),
            timeout_s=1.0,
        )

    assert process.killed is True
    assert captured.value.code == "child_protocol_error"
    assert "backwards" in str(captured.value)


def test_parent_publishes_protocol_failure_for_malformed_child_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = (tmp_path / "repo").resolve()
    repository.mkdir()
    (repository / ".git").mkdir()
    (repository / "uv.lock").write_bytes(b"lock")
    external = (tmp_path / "evidence").resolve()
    external.mkdir()
    phases: list[object] = []

    def boundary(**values: object) -> SourceBoundaryV1:
        phases.append(values["phase"])
        return _capture(repository, phase=values["phase"])

    monkeypatch.setattr(planner, "capture_source_boundary", boundary)
    published = planner.run_registered_planner_cell(
        delivery_rate_kg_h=35.0,
        seed=0,
        destination=external / "malformed.json",
        repository_root=repository,
        popen_factory=lambda *args, **kwargs: _FakeProcess(b"{}\n"),
        clock_ns=lambda: 100,
    )

    failure = published.artifact.runner_failure
    assert failure is not None
    assert failure.phase == "child_execution"
    assert failure.code == "child_protocol_error"
    assert published.artifact.core_result is None
    assert published.artifact.start_source is not None
    assert published.artifact.end_source is None
    assert phases == ["start_source"]


def test_parent_retains_validated_core_when_end_boundary_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = (tmp_path / "repo").resolve()
    repository.mkdir()
    (repository / ".git").mkdir()
    (repository / "uv.lock").write_bytes(b"lock")
    external = (tmp_path / "evidence").resolve()
    external.mkdir()
    core = _failed_core(evidence_scope="privileged_offline_development")

    def boundary(**values: object) -> SourceBoundaryV1:
        if values["phase"] == "end_source":
            raise PlannerEvidenceError(
                phase="end_source",
                code="source_unavailable",
                message="synthetic end failure",
            )
        return _capture(repository)

    monkeypatch.setattr(planner, "capture_source_boundary", boundary)
    published = planner.run_registered_planner_cell(
        delivery_rate_kg_h=35.0,
        seed=0,
        destination=external / "end-failure.json",
        repository_root=repository,
        popen_factory=lambda *args, **kwargs: _FakeProcess(
            canonical_json(core.to_dict()).encode("utf-8")
        ),
        clock_ns=lambda: 100,
    )

    failure = published.artifact.runner_failure
    assert failure is not None
    assert failure.phase == "end_source"
    assert failure.code == "source_unavailable"
    assert published.artifact.core_result == core
    assert published.artifact.start_source is not None
    assert published.artifact.end_source is None
    assert published.path.is_file()


def test_cli_succeeds_only_for_evidence_ready_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = (tmp_path / "cell.json").resolve()
    cases = (
        (True, True, 0, "evidence-ready"),
        (True, False, 1, "completed-not-evidence-ready"),
        (False, False, 1, "failed"),
    )
    for success, evidence_ready, exit_code, label in cases:
        published = SimpleNamespace(
            path=target,
            artifact=SimpleNamespace(
                success=success,
                evidence_ready=evidence_ready,
                artifact_sha256="a" * 64,
            ),
        )
        monkeypatch.setattr(
            planner,
            "run_registered_planner_cell",
            lambda **kwargs: published,
        )

        assert (
            planner.main(
                [
                    "--delivery-rate",
                    "35",
                    "--seed",
                    "0",
                    "--output",
                    str(target),
                ]
            )
            == exit_code
        )
        assert f"planner cell {label}" in capsys.readouterr().out


def test_worker_rejects_unregistered_cell_without_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        planner_worker,
        "sample_episode",
        lambda *args, **kwargs: pytest.fail("unregistered cell sampled an episode"),
    )

    result = planner_worker.execute_registered_cell(
        delivery_rate_kg_h=40.0,
        seed=0,
        start_ns=time.perf_counter_ns(),
    )

    assert result.success is False
    assert result.failure is not None
    assert result.failure.phase == "input_validation"
    assert result.failure.code == "invalid_input"


def test_worker_invokes_the_relaxation_solver_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object()
    relaxation = SimpleNamespace(success=True)
    binding = SimpleNamespace(
        checkpoint_contract_digest="a" * 64,
        model_sha256="b" * 64,
        result_sha256="c" * 64,
        witness_sha256="d" * 64,
        guidance_sha256="e" * 64,
    )
    expected = _failed_core(evidence_scope="privileged_offline_development")
    solver_calls: list[dict[str, object]] = []
    sample_calls: list[tuple[object, object, int]] = []
    start_ns = time.perf_counter_ns()
    received_start_ns: list[int] = []

    def sample(episode: object, config: object, *, seed: int) -> object:
        sample_calls.append((episode, config, seed))
        return object()

    monkeypatch.setattr(planner_worker, "sample_episode", sample)
    monkeypatch.setattr(
        planner_worker,
        "build_delivery_relaxation",
        lambda **kwargs: model,
    )

    def solve(supplied: object, **kwargs: object) -> object:
        assert supplied is model
        solver_calls.append(kwargs)
        return relaxation

    monkeypatch.setattr(planner_worker, "solve_delivery_relaxation", solve)
    monkeypatch.setattr(
        planner_worker,
        "bind_relaxation_guidance",
        lambda supplied_model, supplied_result: binding,
    )
    monkeypatch.setattr(
        planner_worker, "build_planner_settings", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        planner_worker, "admit_planner_runtime", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        planner_worker, "PlannerPlantInputsV1", lambda **kwargs: object()
    )

    def execute(inputs: object, *, start_ns: int) -> FeasiblePlannerResultV1:
        received_start_ns.append(start_ns)
        return expected

    monkeypatch.setattr(
        planner_worker,
        "execute_feasible_planner",
        execute,
    )

    result = planner_worker.execute_registered_cell(
        delivery_rate_kg_h=35.0,
        seed=0,
        start_ns=start_ns,
    )

    assert result is expected
    assert len(sample_calls) == 1
    assert sample_calls[0][2] == 0
    assert len(solver_calls) == 1
    assert solver_calls[0]["solver_options"] == {}
    assert received_start_ns == [start_ns]

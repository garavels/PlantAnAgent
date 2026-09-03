from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from plant_sim import PlantConfig

from plant_backend.controllers.base import Constraints
from plant_backend.rl.v2._planner_evidence import (
    PlannerEvidenceError,
    build_completed_artifact,
    build_runner_failure_artifact,
    capture_runtime_versions,
    capture_source_boundary,
    write_planner_artifact,
)
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.planner_schema import (
    FeasiblePlannerResultV1,
    PlannerFailureV1,
    build_search_diagnostics,
    planner_artifact_from_dict,
)
from plant_backend.rl.v2.reward import RewardWeightsV2


SHORT = EpisodeConfigV2(horizon_h=0.5, delivery_period_h=0.5)


def _collector(root: Path, *, lock_digest: str | None = None):
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


def _capture(root: Path, **changes: object):
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


def _failed_core() -> FeasiblePlannerResultV1:
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
        evidence_scope="noncanonical_unit_fixture",
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

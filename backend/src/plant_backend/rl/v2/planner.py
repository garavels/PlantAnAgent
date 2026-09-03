"""Isolated evidence runner for one preregistered feasible-planner cell."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from plant_sim import PlantConfig

from ...controllers.base import Constraints
from ._planner_evidence import (
    PlannerEvidenceError,
    SourceBoundaryV1,
    build_completed_artifact,
    build_runner_failure_artifact,
    capture_runtime_versions,
    capture_source_boundary,
    validate_artifact_destination,
    write_planner_artifact,
)
from .config import EpisodeConfigV2
from .planner_schema import (
    EPISODE_CONFIG_SHA256_BY_RATE,
    CanonicalPayloadV1,
    FeasiblePlannerArtifactV1,
    FeasiblePlannerResultV1,
    canonical_json,
    planner_result_from_dict,
)
from .reward import RewardWeightsV2


EXTERNAL_WATCHDOG_SECONDS = 5_400.0
_REGISTERED_RATES = (35.0, 42.0)
_REGISTERED_SEEDS = (0, 1, 2)
PopenFactory = Callable[..., Any]
ClockNs = Callable[[], int]


@dataclass(frozen=True, slots=True)
class PublishedPlannerCellV1:
    artifact: FeasiblePlannerArtifactV1
    path: Path

    def __post_init__(self) -> None:
        if type(self.artifact) is not FeasiblePlannerArtifactV1:
            raise TypeError("artifact must be an exact FeasiblePlannerArtifactV1")
        path = Path(self.path)
        if not path.is_absolute():
            raise ValueError("published planner path must be absolute")
        object.__setattr__(self, "path", path.resolve(strict=False))


def _registered_cell(delivery_rate_kg_h: object, seed: object) -> tuple[float, int]:
    if isinstance(delivery_rate_kg_h, bool) or not isinstance(
        delivery_rate_kg_h, (int, float)
    ):
        raise TypeError("delivery_rate_kg_h must be numeric")
    rate = float(delivery_rate_kg_h)
    if rate not in _REGISTERED_RATES:
        raise ValueError("delivery_rate_kg_h must be 35.0 or 42.0")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed not in _REGISTERED_SEEDS:
        raise ValueError("seed must be 0, 1, or 2")
    return rate, seed


def find_repository_root(start: Path | None = None) -> Path:
    """Find the nearest repository carrying both ``.git`` and ``uv.lock``."""

    candidate = (start or Path(__file__)).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for parent in (candidate, *candidate.parents):
        if (parent / ".git").exists() and (parent / "uv.lock").is_file():
            return parent
    raise ValueError("unable to locate the planner Git repository")


def _worker_command(*, rate: float, seed: int) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "plant_backend.rl.v2._planner_worker",
        "--delivery-rate",
        repr(rate),
        "--seed",
        str(seed),
    )


def _message(stderr: bytes, *, fallback: str) -> str:
    try:
        decoded = stderr.decode("utf-8", errors="replace")
    except Exception:
        return fallback
    cleaned = " ".join(decoded.split())
    return cleaned or fallback


def _watchdog_reading(clock_ns: ClockNs, *, previous: int | None = None) -> int:
    value = clock_ns()
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PlannerEvidenceError(
            phase="child_execution",
            code="child_protocol_error",
            message="external watchdog clock did not return a non-negative integer",
        )
    if previous is not None and value < previous:
        raise PlannerEvidenceError(
            phase="child_execution",
            code="child_protocol_error",
            message="external watchdog clock moved backwards",
        )
    return value


def _terminate_child(process: Any) -> bytes:
    """Best-effort child cleanup that never replaces the primary runner failure."""

    try:
        process.kill()
    except Exception:
        pass
    try:
        _, stderr = process.communicate()
    except Exception:
        return b""
    return stderr if isinstance(stderr, bytes) else b""


def _execute_child(
    *,
    rate: float,
    seed: int,
    repository_root: Path,
    popen_factory: PopenFactory = subprocess.Popen,
    clock_ns: ClockNs = time.perf_counter_ns,
    timeout_s: float = EXTERNAL_WATCHDOG_SECONDS,
) -> bytes:
    """Return stdout from one bounded fresh worker or raise a typed runner error."""

    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise TypeError("timeout_s must be numeric")
    timeout = float(timeout_s)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("timeout_s must be positive and finite")
    watchdog_start = _watchdog_reading(clock_ns)
    try:
        process = popen_factory(
            _worker_command(rate=rate, seed=seed),
            cwd=repository_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as exc:
        raise PlannerEvidenceError(
            phase="child_execution",
            code="child_protocol_error",
            message=str(exc) or type(exc).__name__,
        ) from exc
    try:
        after_spawn = _watchdog_reading(clock_ns, previous=watchdog_start)
    except PlannerEvidenceError:
        _terminate_child(process)
        raise
    remaining_s = timeout - (after_spawn - watchdog_start) / 1_000_000_000
    if remaining_s <= 0.0:
        stderr = _terminate_child(process)
        raise PlannerEvidenceError(
            phase="child_execution",
            code="wall_time_limit",
            message=(
                "planner child exhausted the external limit during process startup; "
                f"stderr={_message(stderr, fallback='empty')}"
            ),
        )
    try:
        stdout, stderr = process.communicate(timeout=remaining_s)
    except subprocess.TimeoutExpired as exc:
        stderr = _terminate_child(process)
        try:
            elapsed = _watchdog_reading(clock_ns, previous=watchdog_start)
            elapsed -= watchdog_start
        except PlannerEvidenceError:
            elapsed = int(timeout * 1_000_000_000)
        raise PlannerEvidenceError(
            phase="child_execution",
            code="wall_time_limit",
            message=(
                f"planner child exceeded {timeout:g} seconds; "
                f"elapsed_ns={elapsed}; "
                f"stderr={_message(stderr, fallback='empty')}"
            ),
        ) from exc
    except Exception as exc:
        stderr = _terminate_child(process)
        raise PlannerEvidenceError(
            phase="child_execution",
            code="child_protocol_error",
            message=(
                f"child communication failed: {str(exc) or type(exc).__name__}; "
                f"stderr={_message(stderr, fallback='empty')}"
            ),
        ) from exc
    completed_at = _watchdog_reading(clock_ns, previous=after_spawn)
    elapsed_s = (completed_at - watchdog_start) / 1_000_000_000
    if elapsed_s > timeout:
        raise PlannerEvidenceError(
            phase="child_execution",
            code="wall_time_limit",
            message=(
                f"planner child exceeded {timeout:g} seconds; "
                f"elapsed_ns={completed_at - watchdog_start}; "
                f"stderr={_message(stderr, fallback='empty')}"
            ),
        )
    return_code = process.returncode
    if isinstance(return_code, bool) or not isinstance(return_code, int):
        raise PlannerEvidenceError(
            phase="child_execution",
            code="child_protocol_error",
            message="planner child returned no integral exit status",
        )
    if return_code != 0:
        raise PlannerEvidenceError(
            phase="child_execution",
            code="child_exit",
            message=_message(stderr, fallback="planner child exited without stderr"),
            child_exit_code=return_code,
        )
    if not isinstance(stdout, bytes) or not isinstance(stderr, bytes):
        raise PlannerEvidenceError(
            phase="child_execution",
            code="child_protocol_error",
            message="planner child streams were not bytes",
            child_exit_code=return_code,
        )
    return stdout


def _decode_core(stdout: bytes, *, rate: float, seed: int) -> FeasiblePlannerResultV1:
    try:
        decoded = stdout.decode("utf-8")
        parsed: object = json.loads(decoded)
        if decoded != canonical_json(parsed):
            raise ValueError("worker stdout is not one canonical JSON object")
        result = planner_result_from_dict(parsed)
        if result.evidence_scope != "privileged_offline_development":
            raise ValueError("worker core result has the wrong evidence scope")
        if result.seed is not None and result.seed != seed:
            raise ValueError("worker core result uses a different seed")
        expected_episode_sha = dict(EPISODE_CONFIG_SHA256_BY_RATE)[rate]
        if result.settings is not None and (
            result.settings.episode_config_sha256 != expected_episode_sha
        ):
            raise ValueError("worker core result uses a different registered rate")
        return result
    except PlannerEvidenceError:
        raise
    except Exception as exc:
        raise PlannerEvidenceError(
            phase="child_execution",
            code="child_protocol_error",
            message=str(exc) or type(exc).__name__,
            child_exit_code=0,
        ) from exc


def _elapsed_ns(start_ns: int, clock_ns: ClockNs) -> int:
    now = clock_ns()
    if isinstance(now, bool) or not isinstance(now, int) or now < start_ns:
        return 0
    return now - start_ns


def _publish(
    artifact: FeasiblePlannerArtifactV1,
    *,
    destination: Path,
    repository_root: Path,
) -> PublishedPlannerCellV1:
    try:
        path = write_planner_artifact(
            artifact,
            destination,
            repository_root=repository_root,
        )
    except Exception as exc:
        raise PlannerEvidenceError(
            phase="artifact_serialization",
            code="serialization_error",
            message=str(exc) or type(exc).__name__,
        ) from exc
    return PublishedPlannerCellV1(artifact=artifact, path=path)


def _runner_failure(
    error: PlannerEvidenceError,
    *,
    run_start_ns: int,
    clock_ns: ClockNs,
    destination: Path,
    repository_root: Path,
    core_result: FeasiblePlannerResultV1 | None = None,
    start: SourceBoundaryV1 | None = None,
    end: SourceBoundaryV1 | None = None,
    runtime_versions: CanonicalPayloadV1 | None = None,
) -> PublishedPlannerCellV1:
    artifact = build_runner_failure_artifact(
        phase=error.phase,
        code=error.code,
        message=error,
        elapsed_ns=_elapsed_ns(run_start_ns, clock_ns),
        child_exit_code=error.child_exit_code,
        core_result=core_result,
        start=start,
        end=end,
        runtime_versions=runtime_versions,
    )
    return _publish(
        artifact,
        destination=destination,
        repository_root=repository_root,
    )


def run_registered_planner_cell(
    *,
    delivery_rate_kg_h: float,
    seed: int,
    destination: Path,
    repository_root: Path | None = None,
    popen_factory: PopenFactory = subprocess.Popen,
    clock_ns: ClockNs = time.perf_counter_ns,
) -> PublishedPlannerCellV1:
    """Execute, wrap, and externally publish one preregistered cell."""

    run_start_ns = clock_ns()
    rate, checked_seed = _registered_cell(delivery_rate_kg_h, seed)
    root = find_repository_root(repository_root)
    target = validate_artifact_destination(destination, repository_root=root)
    plant = PlantConfig()
    episode = EpisodeConfigV2(horizon_h=72.0, delivery_rate_kg_h=rate)
    constraints = Constraints()
    weights = RewardWeightsV2()

    try:
        start = capture_source_boundary(
            episode=episode,
            plant=plant,
            constraints=constraints,
            weights=weights,
            root=root,
            require_clean_source=True,
            phase="start_source",
        )
        runtime_versions = capture_runtime_versions()
    except PlannerEvidenceError as exc:
        return _runner_failure(
            exc,
            run_start_ns=run_start_ns,
            clock_ns=clock_ns,
            destination=target,
            repository_root=root,
        )
    except Exception as exc:
        return _runner_failure(
            PlannerEvidenceError(
                phase="start_source",
                code="source_unavailable",
                message=str(exc) or type(exc).__name__,
            ),
            run_start_ns=run_start_ns,
            clock_ns=clock_ns,
            destination=target,
            repository_root=root,
        )

    try:
        stdout = _execute_child(
            rate=rate,
            seed=checked_seed,
            repository_root=root,
            popen_factory=popen_factory,
            clock_ns=clock_ns,
        )
        core_result = _decode_core(stdout, rate=rate, seed=checked_seed)
    except PlannerEvidenceError as exc:
        return _runner_failure(
            exc,
            run_start_ns=run_start_ns,
            clock_ns=clock_ns,
            destination=target,
            repository_root=root,
            start=start,
            runtime_versions=runtime_versions,
        )
    except Exception as exc:
        return _runner_failure(
            PlannerEvidenceError(
                phase="child_execution",
                code="child_protocol_error",
                message=str(exc) or type(exc).__name__,
            ),
            run_start_ns=run_start_ns,
            clock_ns=clock_ns,
            destination=target,
            repository_root=root,
            start=start,
            runtime_versions=runtime_versions,
        )

    try:
        end = capture_source_boundary(
            episode=episode,
            plant=plant,
            constraints=constraints,
            weights=weights,
            root=root,
            require_clean_source=False,
            phase="end_source",
        )
    except PlannerEvidenceError as exc:
        return _runner_failure(
            exc,
            run_start_ns=run_start_ns,
            clock_ns=clock_ns,
            destination=target,
            repository_root=root,
            core_result=core_result,
            start=start,
            runtime_versions=runtime_versions,
        )

    try:
        artifact = build_completed_artifact(
            core_result=core_result,
            start=start,
            end=end,
            runtime_versions=runtime_versions,
        )
        return _publish(
            artifact,
            destination=target,
            repository_root=root,
        )
    except Exception as exc:
        failure = PlannerEvidenceError(
            phase="artifact_serialization",
            code="serialization_error",
            message=str(exc) or type(exc).__name__,
        )
        return _runner_failure(
            failure,
            run_start_ns=run_start_ns,
            clock_ns=clock_ns,
            destination=target,
            repository_root=root,
            core_result=core_result,
            start=start,
            end=end,
            runtime_versions=runtime_versions,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m plant_backend.rl.v2.planner",
        description="Run one preregistered future-information planner cell.",
    )
    parser.add_argument(
        "--delivery-rate",
        type=float,
        required=True,
        choices=_REGISTERED_RATES,
        help="registered methane delivery rate in kg/h",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        choices=_REGISTERED_SEEDS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="absolute JSON destination outside this Git repository",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        published = run_registered_planner_cell(
            delivery_rate_kg_h=arguments.delivery_rate,
            seed=arguments.seed,
            destination=arguments.output,
        )
    except (OSError, TypeError, ValueError, PlannerEvidenceError) as exc:
        print(f"planner runner failed: {exc}", file=sys.stderr)
        return 2
    artifact = published.artifact
    status = (
        "evidence-ready"
        if artifact.evidence_ready
        else "completed-not-evidence-ready" if artifact.success else "failed"
    )
    print(
        f"planner cell {status}; artifact={published.path}; "
        f"sha256={artifact.artifact_sha256}"
    )
    return 0 if artifact.evidence_ready else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "EXTERNAL_WATCHDOG_SECONDS",
    "PublishedPlannerCellV1",
    "build_parser",
    "find_repository_root",
    "main",
    "run_registered_planner_cell",
]

"""Registered one-cell runner for the causal service-MPC development matrix.

This boundary deliberately exposes fewer choices than the generic comparator CLI.  One
fresh process may evaluate one frozen rate/seed cell, in the declared A0/A2/MPC order,
and publish one non-overwriting JSON artifact outside the repository.  Controller failure
is an experimental result and is therefore written before the command returns failure.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from numbers import Integral, Real
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from plant_sim import PlantConfig

from ...controllers.base import Constraints
from .config import EpisodeConfigV2
from .controllers import IssuedForecastControllerV2
from .evaluation import (
    CAUSAL_RESULT_SCHEMA_VERSION,
    CausalControllerEvaluationV2,
    LegacyHeuristicControllerV2,
    PairedContrastV2,
    collect_provenance,
    evaluate_controllers,
    write,
)
from .mpc_controller import ServiceMPCControllerV1
from .mpc_execution import SERVICE_MPC_RUN_EVIDENCE_FORMAT
from .mpc_schema import SERVICE_MPC_CONTROLLER_ID, canonical_json, domain_sha256
from .reward import RewardWeightsV2


REGISTERED_MPC_RATES = (35.0, 42.0)
REGISTERED_MPC_SEEDS = (0, 1, 2)
REGISTERED_MPC_HORIZON_H = 72.0
REGISTERED_MPC_DECISIONS = 288
REGISTERED_MPC_CELLS = tuple(
    (rate, seed) for rate in REGISTERED_MPC_RATES for seed in REGISTERED_MPC_SEEDS
)
REGISTERED_MPC_CONTROLLER_IDS = (
    LegacyHeuristicControllerV2.id,
    IssuedForecastControllerV2.id,
    SERVICE_MPC_CONTROLLER_ID,
)
REGISTERED_MPC_CONTRASTS = (
    PairedContrastV2(
        treatment=IssuedForecastControllerV2.id,
        control=LegacyHeuristicControllerV2.id,
    ),
    PairedContrastV2(
        treatment=SERVICE_MPC_CONTROLLER_ID,
        control=IssuedForecastControllerV2.id,
    ),
)
REGISTERED_MPC_CELL_EXECUTION_FORMAT = "registered-service-mpc-cell-execution-v1"
REGISTERED_MPC_MATRIX_CONTRACT = {
    "format": "causal-service-mpc-development-matrix-v1",
    "order": [
        {"delivery_rate_kg_h": rate, "seed": seed}
        for rate, seed in REGISTERED_MPC_CELLS
    ],
    "horizon_h": REGISTERED_MPC_HORIZON_H,
    "decisions_per_arm": REGISTERED_MPC_DECISIONS,
    "controller_order": list(REGISTERED_MPC_CONTROLLER_IDS),
    "contrasts": [item.to_dict() for item in REGISTERED_MPC_CONTRASTS],
    "execution_unit": "one fresh process per cell",
}
REGISTERED_MPC_MATRIX_CONTRACT_SHA256 = domain_sha256(
    "causal-service-mpc-development-matrix-contract-v1",
    REGISTERED_MPC_MATRIX_CONTRACT,
)

Evaluator = Callable[..., CausalControllerEvaluationV2]


@dataclass(frozen=True, slots=True)
class PublishedServiceMPCCellV1:
    """One validated external cell artifact and its raw-byte identity."""

    result: CausalControllerEvaluationV2
    path: Path
    file_sha256: str
    elapsed_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.result, CausalControllerEvaluationV2):
            raise TypeError("result must be CausalControllerEvaluationV2")
        path = Path(self.path)
        if not path.is_absolute():
            raise ValueError("published service-MPC path must be absolute")
        object.__setattr__(self, "path", path.resolve(strict=False))
        _require_sha256("file_sha256", self.file_sha256)
        if (
            isinstance(self.elapsed_ns, bool)
            or not isinstance(self.elapsed_ns, Integral)
            or self.elapsed_ns < 0
        ):
            raise ValueError("elapsed_ns must be a non-negative integer")


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 digest") from exc
    return value


def _registered_cell(delivery_rate_kg_h: object, seed: object) -> tuple[float, int]:
    if isinstance(delivery_rate_kg_h, bool) or not isinstance(delivery_rate_kg_h, Real):
        raise TypeError("delivery_rate_kg_h must be a real number")
    rate = float(delivery_rate_kg_h)
    if rate not in REGISTERED_MPC_RATES:
        raise ValueError("delivery_rate_kg_h must be 35.0 or 42.0")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    chosen_seed = int(seed)
    if chosen_seed not in REGISTERED_MPC_SEEDS:
        raise ValueError("seed must be 0, 1, or 2")
    return rate, chosen_seed


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _utc_instant(name: str, value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be an RFC 3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} must be an RFC 3339 UTC timestamp") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"{name} must be an RFC 3339 UTC timestamp")
    return parsed


def _execution_metadata(
    *,
    delivery_rate_kg_h: float,
    seed: int,
    started_at_utc: str,
    evaluation_completed_at_utc: str,
) -> dict[str, Any]:
    cell = (delivery_rate_kg_h, seed)
    try:
        sequence_index = REGISTERED_MPC_CELLS.index(cell)
    except ValueError as exc:  # pragma: no cover - guarded by _registered_cell
        raise ValueError("registered cell is absent from the matrix") from exc
    started = _utc_instant("started_at_utc", started_at_utc)
    completed = _utc_instant("evaluation_completed_at_utc", evaluation_completed_at_utc)
    if completed < started:
        raise ValueError("registered cell completion precedes its start")
    return {
        "format": REGISTERED_MPC_CELL_EXECUTION_FORMAT,
        "matrix_contract_sha256": REGISTERED_MPC_MATRIX_CONTRACT_SHA256,
        "sequence_index": sequence_index,
        "cell": {"delivery_rate_kg_h": delivery_rate_kg_h, "seed": seed},
        "started_at_utc": started_at_utc,
        "evaluation_completed_at_utc": evaluation_completed_at_utc,
    }


def find_repository_root(start: Path | None = None) -> Path:
    """Find the nearest Git worktree containing the dependency lock."""

    candidate = (start or Path(__file__)).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for parent in (candidate, *candidate.parents):
        if (parent / ".git").exists() and (parent / "uv.lock").is_file():
            return parent
    raise ValueError("unable to locate the service-MPC Git repository")


def validate_artifact_destination(path: Path | str, *, repository_root: Path) -> Path:
    """Resolve a new external JSON destination before any outcome is opened."""

    requested = Path(path)
    if not requested.is_absolute():
        raise ValueError("registered service-MPC output must be an absolute path")
    root = Path(repository_root).resolve(strict=True)
    target = requested.resolve(strict=False)
    parent = target.parent
    if not parent.exists() or not parent.is_dir():
        raise ValueError("registered service-MPC output parent must already exist")
    if target.suffix.lower() != ".json":
        raise ValueError("registered service-MPC output must use a .json suffix")
    if target.exists() or target.is_symlink():
        raise ValueError("registered service-MPC output already exists")
    try:
        target.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("registered service-MPC output must be outside the repository")
    return target


def _controllers() -> tuple[
    LegacyHeuristicControllerV2,
    IssuedForecastControllerV2,
    ServiceMPCControllerV1,
]:
    return (
        LegacyHeuristicControllerV2(),
        IssuedForecastControllerV2(),
        ServiceMPCControllerV1(),
    )


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def validate_registered_result(
    result: CausalControllerEvaluationV2,
    *,
    delivery_rate_kg_h: float,
    seed: int,
    repository_root: Path,
) -> None:
    """Reject a result that escaped any frozen registered-cell boundary."""

    rate, chosen_seed = _registered_cell(delivery_rate_kg_h, seed)
    if not isinstance(result, CausalControllerEvaluationV2):
        raise TypeError("registered result must be CausalControllerEvaluationV2")
    expected_episode = EpisodeConfigV2(
        horizon_h=REGISTERED_MPC_HORIZON_H,
        delivery_rate_kg_h=rate,
    ).validate(PlantConfig())
    if result.seeds != [chosen_seed]:
        raise ValueError("registered result must contain exactly its declared seed")
    if result.episode_config.to_dict() != expected_episode.to_dict():
        raise ValueError("registered result episode differs from the frozen cell")
    if result.result_schema_version != CAUSAL_RESULT_SCHEMA_VERSION:
        raise ValueError("registered result uses the wrong causal schema")
    if tuple(result.controller_specs) != REGISTERED_MPC_CONTROLLER_IDS:
        raise ValueError("registered result controller order is invalid")
    expected_specs = tuple(controller.to_dict() for controller in _controllers())
    if tuple(result.controller_specs.values()) != expected_specs:
        raise ValueError("registered result controller descriptors are invalid")
    if result.contrasts != REGISTERED_MPC_CONTRASTS:
        raise ValueError("registered result contrasts are invalid")
    if len(result.runs) != len(REGISTERED_MPC_CONTROLLER_IDS):
        raise ValueError("registered result must contain exactly three controller arms")
    if tuple(run.controller for run in result.runs) != REGISTERED_MPC_CONTROLLER_IDS:
        raise ValueError("registered result run order is invalid")
    if any(run.seed != chosen_seed for run in result.runs):
        raise ValueError("registered result contains a run for another seed")
    if any(run.requested_actions is None for run in result.runs):
        raise ValueError("registered result must retain every requested-action trace")
    for run in result.runs:
        if run.successful:
            assert run.metrics is not None
            if run.metrics.decisions != REGISTERED_MPC_DECISIONS:
                raise ValueError(
                    "successful registered run has the wrong decision count"
                )

    pairing = result.pairing()
    for name in (
        "common_random_numbers",
        "exogenous_episode_match_by_seed",
        "full_manifest_match_by_seed",
        "all_arms_present_by_seed",
        "exactly_one_run_per_arm_by_seed",
        "only_declared_seed_controller_runs",
    ):
        if pairing.get(name) is not True:
            raise ValueError(f"registered result pairing failed: {name}")

    provenance = _mapping("registered provenance", result.provenance)
    execution_metadata = _mapping(
        "registered cell execution metadata",
        provenance.get("registered_service_mpc_cell"),
    )
    started_at_utc = execution_metadata.get("started_at_utc")
    evaluation_completed_at_utc = execution_metadata.get("evaluation_completed_at_utc")
    if not isinstance(started_at_utc, str) or not isinstance(
        evaluation_completed_at_utc, str
    ):
        raise ValueError("registered cell execution timestamps are invalid")
    expected_metadata = _execution_metadata(
        delivery_rate_kg_h=rate,
        seed=chosen_seed,
        started_at_utc=started_at_utc,
        evaluation_completed_at_utc=evaluation_completed_at_utc,
    )
    if dict(execution_metadata) != expected_metadata:
        raise ValueError("registered cell execution metadata is invalid")
    source = _mapping("registered source", provenance.get("source"))
    source_audit = _mapping(
        "registered execution source audit",
        provenance.get("execution_source_audit"),
    )
    if (
        source.get("identity_complete") is not True
        or source.get("git_dirty") is not False
    ):
        raise ValueError("registered result did not start from complete clean source")
    for name in (
        "identity_complete_at_start",
        "identity_complete_at_end",
        "worktree_identity_unchanged",
        "clean_at_start",
        "clean_at_end",
        "require_clean_source",
    ):
        if source_audit.get(name) is not True:
            raise ValueError(f"registered source audit failed: {name}")
    if source_audit.get("status") != "valid":
        raise ValueError("registered source audit status is invalid")
    start_identity = _require_sha256(
        "start_worktree_identity_sha256",
        source_audit.get("start_worktree_identity_sha256"),
    )
    end_identity = _require_sha256(
        "end_worktree_identity_sha256",
        source_audit.get("end_worktree_identity_sha256"),
    )
    if (
        start_identity != end_identity
        or source.get("worktree_identity_sha256") != start_identity
    ):
        raise ValueError("registered source identities disagree")

    root = Path(repository_root).resolve(strict=True)
    reported_root = source.get("repository_root")
    if not isinstance(reported_root, str) or Path(reported_root).resolve() != root:
        raise ValueError("registered result reports a different repository")
    dependency_lock = _mapping(
        "registered dependency lock",
        provenance.get("dependency_lock"),
    )
    if dependency_lock.get("path") != "uv.lock":
        raise ValueError("registered result did not bind repository uv.lock")
    observed_lock_sha = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    if dependency_lock.get("sha256") != observed_lock_sha:
        raise ValueError("registered result dependency lock identity is stale")

    mpc_run = result.runs[-1]
    if mpc_run.successful:
        evidence = _mapping(
            "registered service-MPC evidence", mpc_run.controller_evidence
        )
        if evidence.get("format") != SERVICE_MPC_RUN_EVIDENCE_FORMAT:
            raise ValueError("registered service-MPC evidence format is invalid")
        if evidence.get("status") != "complete":
            raise ValueError("successful registered service-MPC evidence is incomplete")
        for name in (
            "completed_decisions",
            "event_count",
            "successful_decision_count",
            "application_count",
        ):
            if evidence.get(name) != REGISTERED_MPC_DECISIONS:
                raise ValueError(f"registered service-MPC {name} is invalid")
        if evidence.get("failed_decision_count") != 0:
            raise ValueError("successful registered service-MPC has failure evidence")
        if evidence.get("fallback_action_count") != 0:
            raise ValueError("registered service-MPC used a hidden fallback")
        if evidence.get("all_successful_decisions_reconciled") is not True:
            raise ValueError("registered service-MPC decisions are not reconciled")
        events = evidence.get("events")
        applications = evidence.get("applications")
        if not isinstance(events, list) or len(events) != REGISTERED_MPC_DECISIONS:
            raise ValueError("registered service-MPC event ledger is incomplete")
        if (
            not isinstance(applications, list)
            or len(applications) != REGISTERED_MPC_DECISIONS
        ):
            raise ValueError("registered service-MPC application ledger is incomplete")

    canonical_json(result.to_dict())


def _assert_current_source_matches(
    result: CausalControllerEvaluationV2,
    *,
    repository_root: Path,
    episode: EpisodeConfigV2,
    plant: PlantConfig,
    constraints: Constraints,
    weights: RewardWeightsV2,
) -> None:
    """Extend the evaluator's source boundary through artifact publication."""

    current = collect_provenance(
        episode=episode,
        plant=plant,
        constraints=constraints,
        weights=weights,
        root=repository_root,
    )
    current_source = _mapping("current source", current.get("source"))
    recorded_source = _mapping("recorded source", result.provenance.get("source"))
    recorded_audit = _mapping(
        "recorded source audit",
        result.provenance.get("execution_source_audit"),
    )
    if current_source.get("identity_complete") is not True:
        raise RuntimeError("current source identity is incomplete")
    if current_source.get("git_dirty") is not False:
        raise RuntimeError("current source is dirty")
    if current_source.get("git_revision") != recorded_source.get("git_revision"):
        raise RuntimeError("source revision changed after controller evaluation")
    if current_source.get("worktree_identity_sha256") != recorded_audit.get(
        "end_worktree_identity_sha256"
    ):
        raise RuntimeError("source content changed after controller evaluation")
    if current.get("dependency_lock") != result.provenance.get("dependency_lock"):
        raise RuntimeError("dependency lock changed after controller evaluation")


def _atomic_write_result(
    result: CausalControllerEvaluationV2,
    target: Path,
) -> str:
    """Validate, serialize, fsync, link without overwrite, and reopen one result."""

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        write(result, temporary)
        expected = result.to_dict()
        raw = temporary.read_bytes()
        expected_raw = (json.dumps(expected, indent=2, allow_nan=False) + "\n").encode(
            "utf-8"
        )
        if raw != expected_raw:
            raise ValueError("serialized service-MPC result bytes are not canonical")
        decoded = json.loads(raw.decode("utf-8"))
        if canonical_json(decoded) != canonical_json(expected):
            raise ValueError("serialized service-MPC result changed during reopening")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        published = target.read_bytes()
        if published != raw:
            raise ValueError("published service-MPC artifact bytes changed")
        return hashlib.sha256(published).hexdigest()
    finally:
        temporary.unlink(missing_ok=True)


def run_registered_service_mpc_cell(
    *,
    delivery_rate_kg_h: object,
    seed: object,
    destination: Path | str,
    repository_root: Path | None = None,
    evaluator: Evaluator = evaluate_controllers,
) -> PublishedServiceMPCCellV1:
    """Evaluate and publish exactly one frozen development cell."""

    started_ns = time.perf_counter_ns()
    rate, chosen_seed = _registered_cell(delivery_rate_kg_h, seed)
    root = find_repository_root(repository_root)
    target = validate_artifact_destination(destination, repository_root=root)
    plant = PlantConfig()
    episode = EpisodeConfigV2(
        horizon_h=REGISTERED_MPC_HORIZON_H,
        delivery_rate_kg_h=rate,
    ).validate(plant)
    constraints = Constraints()
    weights = RewardWeightsV2()
    controllers = _controllers()
    started_at_utc = _utc_now()
    result = evaluator(
        seeds=[chosen_seed],
        controllers=controllers,
        episode=episode,
        plant=plant,
        constraints=constraints,
        weights=weights,
        require_clean_source=True,
    )
    result.provenance["registered_service_mpc_cell"] = _execution_metadata(
        delivery_rate_kg_h=rate,
        seed=chosen_seed,
        started_at_utc=started_at_utc,
        evaluation_completed_at_utc=_utc_now(),
    )
    validate_registered_result(
        result,
        delivery_rate_kg_h=rate,
        seed=chosen_seed,
        repository_root=root,
    )
    _assert_current_source_matches(
        result,
        repository_root=root,
        episode=episode,
        plant=plant,
        constraints=constraints,
        weights=weights,
    )
    file_sha256 = _atomic_write_result(result, target)
    _assert_current_source_matches(
        result,
        repository_root=root,
        episode=episode,
        plant=plant,
        constraints=constraints,
        weights=weights,
    )
    return PublishedServiceMPCCellV1(
        result=result,
        path=target,
        file_sha256=file_sha256,
        elapsed_ns=time.perf_counter_ns() - started_ns,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m plant_backend.rl.v2.mpc_experiment",
        description=(
            "Run one preregistered, development-contaminated causal service-MPC cell."
        ),
    )
    parser.add_argument(
        "--delivery-rate",
        type=float,
        choices=REGISTERED_MPC_RATES,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=REGISTERED_MPC_SEEDS,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        published = run_registered_service_mpc_cell(
            delivery_rate_kg_h=arguments.delivery_rate,
            seed=arguments.seed,
            destination=arguments.output,
        )
    except (TypeError, ValueError) as exc:
        message = " ".join(str(exc).split()) or type(exc).__name__
        print(f"invalid registered service-MPC cell: {message[:500]}", file=sys.stderr)
        return 2
    except Exception as exc:
        message = " ".join(str(exc).split()) or type(exc).__name__
        print(f"registered service-MPC cell failed: {message[:500]}", file=sys.stderr)
        return 1

    print(published.result.report())
    print(f"\nwritten to {published.path}")
    print(f"artifact sha256: {published.file_sha256}")
    print(f"elapsed seconds: {published.elapsed_ns / 1_000_000_000:.6f}")
    if not published.result.execution_summary()["all_planned_arms_succeeded"]:
        print(
            "registered service-MPC cell is ineligible: at least one arm failed",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "PublishedServiceMPCCellV1",
    "REGISTERED_MPC_CELLS",
    "REGISTERED_MPC_CELL_EXECUTION_FORMAT",
    "REGISTERED_MPC_CONTROLLER_IDS",
    "REGISTERED_MPC_CONTRASTS",
    "REGISTERED_MPC_DECISIONS",
    "REGISTERED_MPC_HORIZON_H",
    "REGISTERED_MPC_MATRIX_CONTRACT",
    "REGISTERED_MPC_MATRIX_CONTRACT_SHA256",
    "REGISTERED_MPC_RATES",
    "REGISTERED_MPC_SEEDS",
    "build_parser",
    "find_repository_root",
    "main",
    "run_registered_service_mpc_cell",
    "validate_artifact_destination",
    "validate_registered_result",
]

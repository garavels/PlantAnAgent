"""Outcome-free admission and publication tests for registered service MPC."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from plant_sim import Action, PlantConfig

from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.evaluation import (
    CAUSAL_COMPARATOR_DISCLAIMER,
    CAUSAL_INFORMATION_FAIRNESS,
    CAUSAL_RESULT_SCHEMA_VERSION,
    BaselineRun,
    CausalControllerEvaluationV2,
    ControllerFailureV2,
    LegacyHeuristicControllerV2,
    RequestedActionV2,
)
from plant_backend.rl.v2.metrics import EpisodeMetricsV2
from plant_backend.rl.v2.mpc_controller import ServiceMPCControllerV1
from plant_backend.rl.v2.mpc_execution import SERVICE_MPC_RUN_EVIDENCE_FORMAT
from plant_backend.rl.v2.mpc_experiment import (
    REGISTERED_MPC_CONTROLLER_IDS,
    REGISTERED_MPC_CONTRASTS,
    REGISTERED_MPC_DECISIONS,
    run_registered_service_mpc_cell,
    validate_artifact_destination,
    validate_registered_result,
)
from plant_backend.rl.v2.controllers import IssuedForecastControllerV2
import plant_backend.rl.v2.mpc_experiment as mpc_experiment


_MANIFEST_SHA = "a" * 64
_EXOGENOUS_SHA = "b" * 64
_SOURCE_SHA = "c" * 64


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


def _provenance(repository_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    lock_sha = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    source = {
        "repository_root": str(root),
        "git_revision": "fixture-revision",
        "git_dirty": False,
        "worktree_identity_sha256": _SOURCE_SHA,
        "identity_complete": True,
    }
    return {
        "source": source,
        "dependency_lock": {"path": "uv.lock", "sha256": lock_sha},
        "execution_source_audit": {
            "status": "valid",
            "require_clean_source": True,
            "identity_complete_at_start": True,
            "identity_complete_at_end": True,
            "worktree_identity_unchanged": True,
            "clean_at_start": True,
            "clean_at_end": True,
            "start_worktree_identity_sha256": _SOURCE_SHA,
            "end_worktree_identity_sha256": _SOURCE_SHA,
        },
    }


def _registered_result(
    repository_root: Path,
    *,
    rate: float = 35.0,
    seed: int = 0,
    successful: bool,
) -> CausalControllerEvaluationV2:
    controllers = _controllers()
    traces = tuple(
        RequestedActionV2.from_action(index, Action.idle())
        for index in range(REGISTERED_MPC_DECISIONS)
    )
    runs: list[BaselineRun] = []
    for controller in controllers:
        if successful:
            metrics = EpisodeMetricsV2(
                decisions=REGISTERED_MPC_DECISIONS,
                physical_steps=4_320,
                hours=72.0,
                electrolyser_commitment_enabled=True,
            )
            evidence = None
            if controller.id == ServiceMPCControllerV1.id:
                evidence = {
                    "format": SERVICE_MPC_RUN_EVIDENCE_FORMAT,
                    "status": "complete",
                    "completed_decisions": REGISTERED_MPC_DECISIONS,
                    "event_count": REGISTERED_MPC_DECISIONS,
                    "successful_decision_count": REGISTERED_MPC_DECISIONS,
                    "failed_decision_count": 0,
                    "application_count": REGISTERED_MPC_DECISIONS,
                    "fallback_action_count": 0,
                    "all_successful_decisions_reconciled": True,
                    "events": [{} for _ in range(REGISTERED_MPC_DECISIONS)],
                    "applications": [{} for _ in range(REGISTERED_MPC_DECISIONS)],
                }
            runs.append(
                BaselineRun(
                    controller=controller.id,
                    seed=seed,
                    manifest_sha256=_MANIFEST_SHA,
                    exogenous_episode_sha256=_EXOGENOUS_SHA,
                    metrics=metrics,
                    requested_actions=traces,
                    controller_evidence=evidence,
                )
            )
        else:
            failure = ControllerFailureV2(
                phase="controller_compute_action",
                exception_type="RuntimeError",
                message="synthetic outcome-free runner fixture",
                completed_decisions=0,
                elapsed_hours=0.0,
                partial_metrics={},
            )
            evidence = None
            if controller.id == ServiceMPCControllerV1.id:
                evidence = {
                    "format": SERVICE_MPC_RUN_EVIDENCE_FORMAT,
                    "status": "failed",
                }
            runs.append(
                BaselineRun(
                    controller=controller.id,
                    seed=seed,
                    manifest_sha256=_MANIFEST_SHA,
                    exogenous_episode_sha256=_EXOGENOUS_SHA,
                    metrics=None,
                    failure=failure,
                    requested_actions=(),
                    controller_evidence=evidence,
                )
            )
    return CausalControllerEvaluationV2(
        seeds=[seed],
        episode_config=EpisodeConfigV2(
            horizon_h=72.0,
            delivery_rate_kg_h=rate,
        ).validate(PlantConfig()),
        provenance=_provenance(repository_root),
        runs=runs,
        labels={controller.id: controller.label for controller in controllers},
        information_sets={
            controller.id: controller.information_set for controller in controllers
        },
        controller_specs={
            controller.id: controller.to_dict() for controller in controllers
        },
        study="environment_v2_commitment_alpha_causal_comparators",
        evidence_scope="development_comparator",
        disclaimer=CAUSAL_COMPARATOR_DISCLAIMER,
        title="Environment V2 commitment-alpha causal controller comparison",
        information_fairness=CAUSAL_INFORMATION_FAIRNESS,
        result_schema_version=CAUSAL_RESULT_SCHEMA_VERSION,
        contrasts=REGISTERED_MPC_CONTRASTS,
    )


@pytest.mark.parametrize(
    ("rate", "seed", "message"),
    (
        (True, 0, "real number"),
        (40.0, 0, "35.0 or 42.0"),
        (35.0, True, "integer"),
        (35.0, 3, "0, 1, or 2"),
    ),
)
def test_registered_scope_rejects_any_other_rate_or_seed_before_evaluation(
    tmp_path: Path,
    rate: object,
    seed: object,
    message: str,
) -> None:
    called = False

    def forbidden_evaluator(**_: object) -> CausalControllerEvaluationV2:
        nonlocal called
        called = True
        raise AssertionError("evaluator must not be called")

    with pytest.raises((TypeError, ValueError), match=message):
        run_registered_service_mpc_cell(
            delivery_rate_kg_h=rate,
            seed=seed,
            destination=tmp_path / "cell.json",
            repository_root=Path.cwd(),
            evaluator=forbidden_evaluator,
        )
    assert called is False


def test_destination_is_external_absolute_new_and_json(tmp_path: Path) -> None:
    repository = Path.cwd().resolve()
    target = tmp_path / "cell.json"
    assert validate_artifact_destination(target, repository_root=repository) == target

    with pytest.raises(ValueError, match="absolute"):
        validate_artifact_destination("relative.json", repository_root=repository)
    with pytest.raises(ValueError, match="outside"):
        validate_artifact_destination(
            repository / "forbidden-mpc-cell.json",
            repository_root=repository,
        )
    with pytest.raises(ValueError, match=".json"):
        validate_artifact_destination(
            tmp_path / "cell.txt",
            repository_root=repository,
        )
    target.write_text("preserve me\n", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        validate_artifact_destination(target, repository_root=repository)
    assert target.read_text(encoding="utf-8") == "preserve me\n"


def test_registered_result_validator_accepts_failure_and_complete_evidence() -> None:
    repository = Path.cwd().resolve()
    for successful in (False, True):
        result = _registered_result(repository, successful=successful)
        validate_registered_result(
            result,
            delivery_rate_kg_h=35.0,
            seed=0,
            repository_root=repository,
        )


def test_registered_result_validator_rejects_scope_evidence_and_source_mutation() -> (
    None
):
    repository = Path.cwd().resolve()
    result = _registered_result(repository, successful=True)

    bad_order = deepcopy(result)
    bad_order.runs.reverse()
    with pytest.raises(ValueError, match="run order"):
        validate_registered_result(
            bad_order,
            delivery_rate_kg_h=35.0,
            seed=0,
            repository_root=repository,
        )

    hidden_fallback = deepcopy(result)
    evidence = hidden_fallback.runs[-1].controller_evidence
    assert evidence is not None
    evidence["fallback_action_count"] = 1
    with pytest.raises(ValueError, match="hidden fallback"):
        validate_registered_result(
            hidden_fallback,
            delivery_rate_kg_h=35.0,
            seed=0,
            repository_root=repository,
        )

    dirty = deepcopy(result)
    dirty.provenance["source"]["git_dirty"] = True
    with pytest.raises(ValueError, match="complete clean source"):
        validate_registered_result(
            dirty,
            delivery_rate_kg_h=35.0,
            seed=0,
            repository_root=repository,
        )


def test_registered_runner_uses_exact_inputs_and_atomically_publishes_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path.cwd().resolve()
    result = _registered_result(repository, rate=42.0, seed=2, successful=False)
    captured: dict[str, Any] = {}

    def fake_evaluator(**kwargs: Any) -> CausalControllerEvaluationV2:
        captured.update(kwargs)
        return result

    def current_provenance(**_: Any) -> dict[str, Any]:
        return {
            "source": deepcopy(result.provenance["source"]),
            "dependency_lock": deepcopy(result.provenance["dependency_lock"]),
        }

    monkeypatch.setattr(mpc_experiment, "collect_provenance", current_provenance)
    target = tmp_path / "rate42-seed2.json"
    published = run_registered_service_mpc_cell(
        delivery_rate_kg_h=42.0,
        seed=2,
        destination=target,
        repository_root=repository,
        evaluator=fake_evaluator,
    )

    assert captured["seeds"] == [2]
    assert captured["require_clean_source"] is True
    episode = captured["episode"]
    assert isinstance(episode, EpisodeConfigV2)
    assert episode.horizon_h == 72.0
    assert episode.delivery_rate_kg_h == 42.0
    controllers = captured["controllers"]
    assert tuple(controller.id for controller in controllers) == (
        REGISTERED_MPC_CONTROLLER_IDS
    )
    assert published.path == target.resolve()
    assert published.result is result
    assert published.elapsed_ns >= 0
    raw = target.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == published.file_sha256
    assert mpc_experiment.canonical_json(json.loads(raw)) == (
        mpc_experiment.canonical_json(result.to_dict())
    )
    assert result.execution_summary()["all_planned_arms_succeeded"] is False


def test_existing_destination_blocks_valid_cell_before_evaluator(
    tmp_path: Path,
) -> None:
    target = tmp_path / "existing.json"
    target.write_text("do not replace\n", encoding="utf-8")
    called = False

    def forbidden_evaluator(**_: object) -> CausalControllerEvaluationV2:
        nonlocal called
        called = True
        raise AssertionError("evaluator must not be called")

    with pytest.raises(ValueError, match="already exists"):
        run_registered_service_mpc_cell(
            delivery_rate_kg_h=35.0,
            seed=0,
            destination=target,
            repository_root=Path.cwd(),
            evaluator=forbidden_evaluator,
        )
    assert called is False
    assert target.read_text(encoding="utf-8") == "do not replace\n"

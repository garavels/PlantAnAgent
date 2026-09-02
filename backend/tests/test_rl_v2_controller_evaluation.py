"""Paired evaluation through the typed Environment V2 controller seam."""

from __future__ import annotations

from typing import Any

import pytest
from plant_sim import Action, ElectrolyserCommitmentProfile, PlantConfig

from plant_backend.controllers import Constraints, ControlDecision
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.control import ControllerObservationV2
from plant_backend.rl.v2.env import PlantEnvV2
from plant_backend.rl.v2.evaluation import (
    CAUSAL_COMPARATOR_DISCLAIMER,
    BASELINES,
    LegacyHeuristicControllerV2,
    evaluate_controllers,
    run_baseline,
    run_controller,
)
from plant_backend.rl.v2.weather import PublicForecastIssueV2


SHORT = EpisodeConfigV2(horizon_h=0.5, delivery_period_h=0.5)


class SpyControllerV2:
    id = "spy"
    label = "Spy"
    description = "Contract-test controller"
    information_set = "typed causal V2 boundary"

    def __init__(self, *, uses_forecast: bool = False) -> None:
        self.uses_forecast = uses_forecast
        self.reset_calls = 0
        self.observations: list[ControllerObservationV2] = []

    def reset(self) -> None:
        self.reset_calls += 1
        self.observations.clear()

    def compute_action(
        self,
        observation: ControllerObservationV2,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: ElectrolyserCommitmentProfile | None,
    ) -> ControlDecision:
        assert isinstance(constraints, Constraints)
        assert isinstance(config, PlantConfig)
        assert commitment_profile == SHORT.electrolyser_commitment
        self.observations.append(observation)
        return ControlDecision(Action.idle())

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "uses_forecast": self.uses_forecast,
            "information_set": self.information_set,
        }


class InvalidDecisionControllerV2(SpyControllerV2):
    id = "invalid-spy"
    label = "Invalid spy"

    def compute_action(
        self,
        observation: ControllerObservationV2,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: ElectrolyserCommitmentProfile | None,
    ) -> ControlDecision:
        del observation, constraints, config, commitment_profile
        return ControlDecision(Action.idle(), invalid=True)


def test_run_controller_redacts_forecast_for_blind_policy() -> None:
    controller = SpyControllerV2(uses_forecast=False)
    run = run_controller(PlantEnvV2(episode=SHORT), controller, seed=5)

    assert run.controller == "spy"
    assert controller.reset_calls == 1
    assert len(controller.observations) == 2
    assert all(item.forecast is None for item in controller.observations)
    assert all(not hasattr(item, "environment") for item in controller.observations)


def test_run_controller_retains_controller_invalidity_in_metrics_and_trace() -> None:
    run = run_controller(
        PlantEnvV2(episode=SHORT),
        InvalidDecisionControllerV2(),
        seed=5,
    )

    assert run.metrics is not None
    assert run.metrics.out_of_range_decisions == run.metrics.decisions == 2
    assert run.requested_actions is not None
    assert all(item.controller_invalid for item in run.requested_actions)
    assert all(
        item["controller_invalid"]
        for item in run.to_dict()["requested_action_trace"]["actions"]
    )


def test_run_controller_passes_only_public_current_issue_to_forecast_policy() -> None:
    controller = SpyControllerV2(uses_forecast=True)
    run_controller(PlantEnvV2(episode=SHORT), controller, seed=7)

    issues = [item.forecast for item in controller.observations]
    assert all(isinstance(item, PublicForecastIssueV2) for item in issues)
    assert [item.issued_at_physics_boundary_step for item in issues if item] == [0, 15]
    assert all(not hasattr(item, "_issue_stream_sha256") for item in issues)


def test_paired_controller_evaluation_records_declared_information_and_settings() -> None:
    blind = SpyControllerV2(uses_forecast=False)
    forecast = SpyControllerV2(uses_forecast=True)
    forecast.id = "spy_forecast"
    forecast.label = "Spy forecast"
    result = evaluate_controllers(
        seeds=[3, 11],
        controllers=(blind, forecast),
        episode=SHORT,
    )
    payload = result.to_dict()

    assert len(result.runs) == 4
    assert result.pairing()["exogenous_episode_match_by_seed"] is True
    assert result.pairing()["full_manifest_match_by_seed"] is True
    assert result.pairing()["all_arms_present_by_seed"] is True
    assert payload["evidence_scope"] == "development_comparator"
    assert payload["schema_version"] == 5
    assert payload["disclaimer"] == CAUSAL_COMPARATOR_DISCLAIMER
    assert payload["controller_specs"]["spy"]["uses_forecast"] is False
    assert payload["controller_specs"]["spy_forecast"]["uses_forecast"] is True
    assert all(run["requested_action_trace"]["count"] == 2 for run in payload["runs"])
    source_audit = payload["provenance"]["execution_source_audit"]
    assert source_audit["status"] == "valid"
    assert source_audit["worktree_identity_unchanged"] is True
    assert "richer than" in result.report()


def test_legacy_v2_adapter_preserves_the_frozen_baseline_behavior() -> None:
    baseline = next(item for item in BASELINES if item.id == "heuristic")
    direct = run_baseline(PlantEnvV2(episode=SHORT), baseline, seed=13)
    adapted = run_controller(
        PlantEnvV2(episode=SHORT),
        LegacyHeuristicControllerV2(),
        seed=13,
    )

    assert adapted.metrics.to_dict() == direct.metrics.to_dict()
    assert adapted.manifest_sha256 == direct.manifest_sha256
    assert adapted.exogenous_episode_sha256 == direct.exogenous_episode_sha256


def test_controller_evaluation_rejects_duplicate_ids_and_bad_descriptors() -> None:
    with pytest.raises(ValueError, match="unique"):
        evaluate_controllers(
            seeds=[1],
            controllers=(SpyControllerV2(), SpyControllerV2()),
            episode=SHORT,
        )

    bad_id = SpyControllerV2()
    bad_id.to_dict = lambda: {"id": "wrong"}  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="descriptor id"):
        evaluate_controllers(seeds=[1], controllers=(bad_id,), episode=SHORT)

    bad_json = SpyControllerV2()
    bad_json.to_dict = lambda: {"id": "spy", "bad": float("nan")}  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="strict JSON"):
        evaluate_controllers(seeds=[1], controllers=(bad_json,), episode=SHORT)


@pytest.mark.parametrize("bad_seed", [True, -1, 1.5])
def test_controller_evaluation_rejects_invalid_seeds(bad_seed: Any) -> None:
    with pytest.raises(ValueError, match="non-negative integers"):
        evaluate_controllers(
            seeds=[bad_seed],
            controllers=(SpyControllerV2(),),
            episode=SHORT,
        )

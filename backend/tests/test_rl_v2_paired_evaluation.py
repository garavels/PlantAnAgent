"""Failure-safe paired effects and source identity for V2 evaluation."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from plant_sim import Action, ElectrolyserCommitmentProfile, PlantConfig

from plant_backend.controllers import Constraints, ControlDecision
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.control import ControllerObservationV2
from plant_backend.rl.v2.evaluation import (
    BaselineRun,
    ControllerFailureV2,
    ControllerEvaluationV2,
    PairedContrastV2,
    RequestedActionV2,
    collect_provenance,
    evaluate_controllers,
    write,
)
from plant_backend.rl.v2.metrics import EpisodeMetricsV2
from plant_backend.rl.v2.reward import RewardWeightsV2


SHORT = EpisodeConfigV2(horizon_h=0.5, delivery_period_h=0.5)
HASH = "a" * 64


class IdleController:
    label = "Idle test controller"
    description = "Deterministic test-only idle controller"
    uses_forecast = False
    information_set = "typed causal V2 boundary without forecast"

    def __init__(self, controller_id: str = "idle-control") -> None:
        self.id = controller_id

    def reset(self) -> None:
        pass

    def compute_action(
        self,
        observation: ControllerObservationV2,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: ElectrolyserCommitmentProfile | None,
    ) -> ControlDecision:
        del observation, constraints, config, commitment_profile
        return ControlDecision(Action.idle())

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "uses_forecast": self.uses_forecast,
            "information_set": self.information_set,
        }


class FailingController(IdleController):
    label = "Failing test controller"

    def compute_action(
        self,
        observation: ControllerObservationV2,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: ElectrolyserCommitmentProfile | None,
    ) -> ControlDecision:
        del observation, constraints, config, commitment_profile
        raise RuntimeError("intentional\ncontroller failure")


class ResetMutatesDescriptorController(IdleController):
    label = "Descriptor-mutating test controller"

    def reset(self) -> None:
        self.uses_forecast = True


class UncopyableController(IdleController):
    label = "Uncopyable test controller"

    def __deepcopy__(self, memo: dict[int, object]) -> "UncopyableController":
        del memo
        raise RuntimeError("copy refused")


class SelfCopyingController(IdleController):
    label = "Self-copying test controller"

    def __deepcopy__(self, memo: dict[int, object]) -> "SelfCopyingController":
        del memo
        return self


class DecisionMutatesDescriptorController(IdleController):
    label = "Decision-mutating test controller"

    def compute_action(
        self,
        observation: ControllerObservationV2,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: ElectrolyserCommitmentProfile | None,
    ) -> ControlDecision:
        del observation, constraints, config, commitment_profile
        self.label = "Descriptor changed during execution"
        return ControlDecision(Action.idle())


def _metrics(*, delivered: float, requested: float) -> EpisodeMetricsV2:
    unmet = requested - delivered
    return EpisodeMetricsV2(
        commitments_settled=1,
        commitments_met=int(unmet <= 1e-9),
        requested_kg=requested,
        delivered_kg=delivered,
        unmet_kg=unmet,
        worst_commitment_fill_rate=delivered / requested,
    )


def _run(controller: str, seed: int, delivered: float) -> BaselineRun:
    return BaselineRun(
        controller=controller,
        seed=seed,
        manifest_sha256=HASH,
        exogenous_episode_sha256=HASH,
        metrics=_metrics(delivered=delivered, requested=100.0),
    )


def _traced_run(
    controller: str,
    seed: int,
    delivered: float,
    actions: tuple[Action, ...],
) -> BaselineRun:
    metrics = _metrics(delivered=delivered, requested=100.0)
    metrics.decisions = len(actions)
    return BaselineRun(
        controller=controller,
        seed=seed,
        manifest_sha256=HASH,
        exogenous_episode_sha256=HASH,
        metrics=metrics,
        requested_actions=tuple(
            RequestedActionV2.from_action(index, action)
            for index, action in enumerate(actions)
        ),
    )


def test_predeclared_contrast_preserves_raw_and_benefit_oriented_deltas() -> None:
    result = ControllerEvaluationV2(
        seeds=[1, 2],
        episode_config=SHORT,
        provenance={},
        labels={"legacy": "Legacy", "delivery": "Delivery"},
        information_sets={"legacy": "blind", "delivery": "blind plus contract"},
        runs=[
            _run("legacy", 1, 80.0),
            _run("delivery", 1, 90.0),
            _run("legacy", 2, 60.0),
            _run("delivery", 2, 50.0),
        ],
        contrasts=(PairedContrastV2("delivery", "legacy"),),
    )

    contrast = result.to_dict()["contrasts"][0]
    fill = contrast["metrics"]["delivery_fill_rate"]
    unmet = contrast["metrics"]["unmet_kg"]

    assert contrast["treatment"] == "delivery"
    assert contrast["control"] == "legacy"
    assert [item["seed"] for item in fill["by_seed"]] == [1, 2]
    assert [
        item["raw_treatment_minus_control"] for item in fill["by_seed"]
    ] == pytest.approx([0.1, -0.1])
    assert [item["raw_treatment_minus_control"] for item in unmet["by_seed"]] == [
        -10.0,
        10.0,
    ]
    assert [item["benefit_delta"] for item in unmet["by_seed"]] == [10.0, -10.0]
    assert fill["benefit_delta_summary"] == pytest.approx(
        {
            "count": 2,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.1,
            "min": -0.1,
            "max": 0.1,
            "worst": -0.1,
            "worst_seeds": [2],
            "best": 0.1,
            "lower_tail_fraction": 0.2,
            "lower_tail_count": 1,
            "lower_tail_mean": -0.1,
            "lower_tail_seeds": [2],
        }
    )
    report = result.report()
    assert "benefit signs only where direction is declared" in report
    assert "lower-tail mean" in report
    assert "routing, equipment-refusal, and overlap-cancellation" in report


def test_inventory_diagnostics_never_receive_a_fabricated_benefit_sign() -> None:
    control = _run("control", 1, 80.0)
    treatment = _run("treatment", 1, 90.0)
    assert control.metrics is not None
    assert treatment.metrics is not None
    control.metrics.terminal_h2_kg = 12.0
    treatment.metrics.terminal_h2_kg = 7.0
    result = ControllerEvaluationV2(
        seeds=[1],
        episode_config=SHORT,
        provenance={},
        labels={"control": "Control", "treatment": "Treatment"},
        information_sets={"control": "same", "treatment": "same"},
        runs=[control, treatment],
        contrasts=(PairedContrastV2("treatment", "control"),),
    )

    metrics = result.to_dict()["contrasts"][0]["metrics"]
    diagnostic = metrics["terminal_h2_kg"]
    assert diagnostic["direction"] == "diagnostic"
    assert diagnostic["raw_treatment_minus_control_summary"]["mean"] == -5.0
    assert diagnostic["benefit_delta_summary"] is None
    assert "benefit_delta" not in diagnostic["by_seed"][0]
    for name in (
        "battery_discharged_kwh",
        "dispatch_gap_kwh",
        "request_application_gap_kwh",
    ):
        assert metrics[name]["direction"] == "diagnostic"
        assert metrics[name]["benefit_delta_summary"] is None
        assert "benefit_delta" not in metrics[name]["by_seed"][0]
    assert "diagnostic, no benefit sign" in result.report()


def test_requested_action_trace_proves_whether_treatment_was_expressed() -> None:
    result = ControllerEvaluationV2(
        seeds=[1],
        episode_config=SHORT,
        provenance={},
        labels={"control": "Control", "treatment": "Treatment"},
        information_sets={"control": "same", "treatment": "same"},
        runs=[
            _traced_run("control", 1, 80.0, (Action.idle(), Action.idle())),
            _traced_run(
                "treatment",
                1,
                90.0,
                (
                    Action.idle(),
                    Action(battery_share=0.0, electrolyser_share=0.25),
                ),
            ),
        ],
        contrasts=(PairedContrastV2("treatment", "control"),),
    )

    payload = result.to_dict()
    trace = payload["runs"][0]["requested_action_trace"]
    assert trace["count"] == 2
    assert len(trace["sha256"]) == 64
    mechanism = payload["contrasts"][0]["requested_action_mechanism"]
    assert mechanism["status"] == "available"
    assert mechanism["treatment_expression_status"] == "expressed"
    assert mechanism["decisions_compared"] == 2
    assert mechanism["unequal_requested_decisions"] == 1
    assert mechanism["share_l1_sum"] == pytest.approx(0.25)
    assert mechanism["battery_support_abs_delta_kw_sum"] == 0.0


def test_no_contrast_keeps_successful_baseline_shape_free_of_new_sections() -> None:
    run = _run("legacy", 1, 80.0)
    result = ControllerEvaluationV2(
        seeds=[1],
        episode_config=SHORT,
        provenance={},
        labels={"legacy": "Legacy"},
        information_sets={"legacy": "blind"},
        runs=[run],
    )

    payload = result.to_dict()
    assert "contrasts" not in payload
    assert "execution" not in payload
    assert run.to_dict() == {
        "controller": "legacy",
        "seed": 1,
        "manifest_sha256": HASH,
        "exogenous_episode_sha256": HASH,
        "metrics": run.metrics.to_dict(),  # type: ignore[union-attr]
    }


def test_optional_energy_intensity_is_never_imputed_as_zero() -> None:
    no_delivery = _metrics(delivered=0.0, requested=100.0)
    delivered = _metrics(delivered=50.0, requested=100.0)
    delivered.electrolyser_applied_energy_kwh = 2_500.0
    result = ControllerEvaluationV2(
        seeds=[1, 2],
        episode_config=SHORT,
        provenance={},
        labels={"control": "Control", "treatment": "Treatment"},
        information_sets={"control": "same", "treatment": "same"},
        runs=[
            BaselineRun("control", 1, HASH, HASH, no_delivery),
            BaselineRun("treatment", 1, HASH, HASH, delivered),
            BaselineRun("control", 2, HASH, HASH, delivered),
            BaselineRun("treatment", 2, HASH, HASH, delivered),
        ],
        contrasts=(PairedContrastV2("treatment", "control"),),
    )

    summary = result.summary()["control"]["metrics"]
    assert summary["accepted_process_energy_per_delivered_kg"]["count"] == 1
    energy = result.to_dict()["contrasts"][0]["metrics"][
        "accepted_process_energy_per_delivered_kg"
    ]
    assert energy["available_pairs"] == 1
    assert energy["unavailable_pairs"] == 1
    assert energy["all_metric_pairs_defined"] is False
    assert energy["raw_treatment_minus_control_summary"] is None
    assert energy["benefit_delta_summary"] is None
    assert energy["by_seed"][0] == {
        "seed": 1,
        "status": "unavailable",
        "reasons": ["control_metric_unavailable"],
    }


def test_controller_order_predeclares_only_adjacent_contrasts() -> None:
    result = evaluate_controllers(
        seeds=[8],
        controllers=(
            IdleController("legacy"),
            IdleController("delivery"),
            IdleController("forecast"),
        ),
        episode=SHORT,
    )

    assert [
        (contrast.treatment, contrast.control) for contrast in result.contrasts
    ] == [("delivery", "legacy"), ("forecast", "delivery")]
    assert [item["id"] for item in result.to_dict()["contrasts"]] == [
        "delivery_minus_legacy",
        "forecast_minus_delivery",
    ]


def test_controller_failure_is_retained_and_pair_is_explicitly_unavailable() -> None:
    result = evaluate_controllers(
        seeds=[3, 4],
        controllers=(IdleController(), FailingController("treatment")),
        episode=SHORT,
    )
    payload = result.to_dict()

    assert payload["schema_version"] == 6
    assert len(payload["runs"]) == 4
    assert payload["execution"]["by_controller"]["treatment"] == {
        "planned_runs": 2,
        "attempted_runs": 2,
        "successful_runs": 0,
        "failed_runs": 2,
    }
    failures = payload["execution"]["failed_runs"]
    assert [item["seed"] for item in failures] == [3, 4]
    assert all(item["status"] == "failed" for item in failures)
    assert all(item["metrics"] is None for item in failures)
    assert all(
        item["failure"]["phase"] == "controller_compute_action"
        for item in failures
    )
    assert all(
        item["failure"]["message"] == "intentional controller failure"
        for item in failures
    )
    contrast = payload["contrasts"][0]
    assert all(
        item == {
            "seed": seed,
            "status": "unavailable",
            "reasons": ["treatment_failed"],
            "treatment_failure_phase": "controller_compute_action",
        }
        for seed, item in zip([3, 4], contrast["paired_seed_execution"], strict=True)
    )
    fill = contrast["metrics"]["delivery_fill_rate"]
    assert fill["available_pairs"] == 0
    assert fill["unavailable_pairs"] == 2
    assert fill["benefit_delta_summary"] is None
    assert fill["raw_treatment_minus_control_summary"] is None
    assert contrast["effect_summary_status"] == (
        "invalid_incomplete_or_mismatched_execution"
    )
    assert result.pairing()["all_arms_present_by_seed"] is True
    assert result.pairing()["all_arms_successful_by_seed"] is False
    assert "Failed controller runs (retained, never imputed)" in result.report()


def test_failure_in_any_planned_arm_invalidates_every_aggregate_contrast() -> None:
    failure = ControllerFailureV2(
        phase="controller_compute_action",
        exception_type="RuntimeError",
        message="failed third arm",
        completed_decisions=0,
        elapsed_hours=0.0,
        partial_metrics=EpisodeMetricsV2().to_dict(),
    )
    result = ControllerEvaluationV2(
        seeds=[1],
        episode_config=SHORT,
        provenance={},
        labels={"control": "Control", "treatment": "Treatment", "third": "Third"},
        information_sets={"control": "same", "treatment": "same", "third": "same"},
        runs=[
            _traced_run("control", 1, 80.0, (Action.idle(),)),
            _traced_run("treatment", 1, 90.0, (Action(electrolyser_share=0.2),)),
            BaselineRun(
                controller="third",
                seed=1,
                manifest_sha256=HASH,
                exogenous_episode_sha256=HASH,
                metrics=None,
                failure=failure,
                requested_actions=(),
            ),
        ],
        contrasts=(PairedContrastV2("treatment", "control"),),
    )

    contrast = result.to_dict()["contrasts"][0]
    assert contrast["local_pair_execution_complete"] is True
    assert contrast["whole_study_execution_complete"] is False
    assert contrast["effect_summary_status"] == (
        "invalid_incomplete_or_mismatched_execution"
    )
    assert contrast["metrics"]["delivery_fill_rate"][
        "raw_treatment_minus_control_summary"
    ] is None
    assert result.execution_summary()["all_planned_arms_succeeded"] is False
    assert "Aggregate effects are ineligible" in result.report()


def test_pair_hash_mismatch_is_not_claimed_as_common_random_numbers() -> None:
    control = _traced_run("control", 1, 80.0, (Action.idle(),))
    treatment = replace(
        _traced_run("treatment", 1, 90.0, (Action.idle(),)),
        exogenous_episode_sha256="b" * 64,
        manifest_sha256="c" * 64,
    )
    result = ControllerEvaluationV2(
        seeds=[1],
        episode_config=SHORT,
        provenance={},
        labels={"control": "Control", "treatment": "Treatment"},
        information_sets={"control": "same", "treatment": "same"},
        runs=[control, treatment],
        contrasts=(PairedContrastV2("treatment", "control"),),
    )

    assert result.pairing()["common_random_numbers"] is False
    execution = result.to_dict()["contrasts"][0]["paired_seed_execution"][0]
    assert execution["status"] == "unavailable"
    assert execution["reasons"] == [
        "exogenous_episode_mismatch",
        "full_manifest_mismatch",
    ]


def test_write_revalidates_mutated_causal_pair_identity(tmp_path: Path) -> None:
    result = evaluate_controllers(
        seeds=[1],
        controllers=(IdleController("control"), IdleController("treatment")),
        episode=SHORT,
    )
    result.runs[1] = replace(result.runs[1], manifest_sha256="b" * 64)

    with pytest.raises(ValueError, match="pairing is invalid"):
        write(result, tmp_path / "invalid.json")
    assert not (tmp_path / "invalid.json").exists()


def test_write_rejects_mutated_duplicate_causal_seeds(tmp_path: Path) -> None:
    result = evaluate_controllers(
        seeds=[1],
        controllers=(IdleController("control"), IdleController("treatment")),
        episode=SHORT,
    )
    result.seeds.append(1)

    with pytest.raises(ValueError, match="unique"):
        write(result, tmp_path / "duplicate-seeds.json")
    assert not (tmp_path / "duplicate-seeds.json").exists()


def test_write_rejects_removed_or_post_hoc_causal_contrasts(tmp_path: Path) -> None:
    result = evaluate_controllers(
        seeds=[1],
        controllers=(
            IdleController("control"),
            IdleController("middle"),
            IdleController("treatment"),
        ),
        episode=SHORT,
    )
    result.contrasts = (PairedContrastV2("treatment", "control"),)

    with pytest.raises(ValueError, match="contrasts were mutated"):
        write(result, tmp_path / "post-hoc-contrast.json")
    assert not (tmp_path / "post-hoc-contrast.json").exists()


def test_causal_result_cannot_mutate_into_baseline_shape_to_bypass_write(
    tmp_path: Path,
) -> None:
    result = evaluate_controllers(
        seeds=[1],
        controllers=(IdleController("control"), IdleController("treatment")),
        episode=SHORT,
    )
    result.controller_specs.clear()
    result.contrasts = ()

    with pytest.raises(ValueError, match="declared controller specs"):
        write(result, tmp_path / "causal-as-baseline.json")
    assert not (tmp_path / "causal-as-baseline.json").exists()


@pytest.mark.parametrize(
    ("controller", "phase"),
    [
        (
            ResetMutatesDescriptorController("mutating"),
            "descriptor_integrity_after_reset",
        ),
        (UncopyableController("uncopyable"), "controller_copy"),
        (SelfCopyingController("self-copying"), "controller_copy"),
        (
            DecisionMutatesDescriptorController("decision-mutating"),
            "descriptor_integrity_after_decision",
        ),
    ],
)
def test_executed_descriptor_or_copy_mismatch_fails_closed(
    controller: IdleController,
    phase: str,
) -> None:
    result = evaluate_controllers(
        seeds=[9],
        controllers=(controller,),
        episode=SHORT,
    )

    assert len(result.runs) == 1
    assert result.runs[0].successful is False
    assert result.runs[0].failure is not None
    assert result.runs[0].failure.phase == phase


def _run_git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
    ).stdout


def test_dirty_provenance_binds_tracked_patch_and_untracked_contents(
    tmp_path: Path,
) -> None:
    _run_git(tmp_path, "init", "-q")
    (tmp_path / "uv.lock").write_text("lock-v1\n", encoding="utf-8")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("committed\n", encoding="utf-8")
    _run_git(tmp_path, "add", "uv.lock", "tracked.txt")
    _run_git(
        tmp_path,
        "-c",
        "user.name=Evaluator Test",
        "-c",
        "user.email=evaluator@example.invalid",
        "commit",
        "-q",
        "-m",
        "fixture",
    )
    tracked.write_text("modified\n", encoding="utf-8")
    untracked = tmp_path / "untracked.txt"
    untracked.write_bytes(b"untracked payload\n")

    plant = PlantConfig()
    provenance = collect_provenance(
        episode=SHORT.validate(plant),
        plant=plant,
        constraints=Constraints(),
        weights=RewardWeightsV2(),
        root=tmp_path,
    )
    source = provenance["source"]
    expected_patch = _run_git(
        tmp_path,
        "diff",
        "--binary",
        "--full-index",
        "--no-color",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--diff-algorithm=myers",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        "HEAD",
        "--",
    )

    assert source["git_dirty"] is True
    assert source["tracked_binary_diff_sha256"] == hashlib.sha256(
        expected_patch
    ).hexdigest()
    assert source["tracked_binary_diff_bytes"] == len(expected_patch)
    assert source["untracked_files"] == [
        {
            "path": "untracked.txt",
            "kind": "file",
            "content_sha256": hashlib.sha256(untracked.read_bytes()).hexdigest(),
            "bytes": len(untracked.read_bytes()),
        }
    ]
    assert len(source["worktree_identity_sha256"]) == 64
    assert source["identity_complete"] is True
    assert "tracked_binary_diff" not in source

    untracked.write_bytes(b"changed untracked payload\n")
    changed = collect_provenance(
        episode=SHORT.validate(plant),
        plant=plant,
        constraints=Constraints(),
        weights=RewardWeightsV2(),
        root=tmp_path,
    )
    assert (
        changed["source"]["worktree_identity_sha256"]
        != source["worktree_identity_sha256"]
    )

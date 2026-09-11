"""Real fixture ledgers, adversarial admission, and two-arm eligibility."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from plant_backend.rl import checkpoints, policies
from plant_backend.rl.v2 import beta_controller_benchmark as benchmark
from plant_backend.rl.v2 import beta_controller_benchmark_verifier as verifier
from plant_backend.rl.v2 import beta_spaces
from plant_backend.rl.v2.beta_config import (
    CyclicMarketTaskConfigV1,
    CyclicReturnConfigV1,
)
from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.mpc_schema import domain_sha256
from plant_backend.rl.v2.policy_setpoint_controller import (
    policy_contract_for_environment,
)
from plant_backend.rl.v2.study import arm_evidence as arms
from plant_backend.rl.v2.study import pairwise
from plant_backend.rl.v2.training import custody_io as io
from plant_backend.rl.v2.training.config import ControllerNeutralTrainingConfigV1
from plant_backend.rl.v2.training.custody_json import authenticate, canonical_bytes
from plant_backend.rl.v2.training.ppo import make_env


class _FailingEconomic:
    def __init__(self, config: ControllerNeutralTrainingConfigV1) -> None:
        self.original = benchmark._real_beta_economic_controller(config)

    def reset(self) -> None:
        self.original.reset()

    def compute_command(self, *_args: object, **_kwargs: object) -> object:
        raise RuntimeError("fixture: E-beta fails before movement")

    def to_dict(self) -> dict[str, object]:
        return self.original.to_dict()


@pytest.fixture(scope="module")
def evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("study-arm-evidence").resolve()
    # Disposable positive-pair geometry only; the registered task is unchanged.
    config = ControllerNeutralTrainingConfigV1(
        environment_id=beta_spaces.ENVIRONMENT_ID,
        episode=EpisodeConfigV2(
            horizon_h=24.0,
            decision_interval_s=3600.0,
            start_hour=6.0,
            delivery_period_h=24.0,
            delivery_rate_kg_h=1.0,
            battery_soc=(0.9, 0.9),
            h2_fill=(0.1, 0.1),
            co2_fill=(0.1, 0.1),
            ch4_coverage_h=(24.0, 24.0),
        ),
        beta_task=CyclicMarketTaskConfigV1(
            cycle=CyclicReturnConfigV1(
                battery_return_fraction=0.0,
                h2_return_fraction=0.0,
                co2_return_fraction=0.0,
                ch4_return_fraction=0.0,
            )
        ),
        total_timesteps=4,
        n_envs=1,
        checkpoint_every=0,
        notes="synthetic study evidence fixture; no trained or registered policy",
    ).validate()
    environment = make_env(config)
    try:
        contract = policy_contract_for_environment(environment)
        checkpoint = checkpoints.write_fixed_checkpoint(
            root / "fixed", np.full(4, -1.0, dtype=np.float32), contract=contract
        )
        metadata = checkpoints.load_metadata(checkpoint, check=False)
        checkpoints.save_metadata(
            checkpoint, replace(metadata, training_config=config.to_dict())
        )
        loaded = policies.load_policy(
            checkpoint, contract=contract, environment=environment
        )
    finally:
        environment.close()

    def fail_context(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("fixture: context heuristic fails before movement")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            benchmark.LegacyHeuristicSetpointControllerV1,
            "compute_command",
            fail_context,
        )
        report = benchmark.evaluate_beta_controller_benchmark(
            loaded,
            config,
            seeds=[9_000_061],
            require_trained_policy=False,
        )
        failed = benchmark.evaluate_beta_controller_benchmark(
            loaded,
            config,
            seeds=[9_000_062],
            require_trained_policy=False,
            economic_controller_factory=_FailingEconomic,
        )
    envelopes = {
        value["actor"]["arm"]: value
        for value in arms.synthetic_arms_from_benchmark(report)
    }
    failed_envelopes = {
        value["actor"]["arm"]: value
        for value in arms.synthetic_arms_from_benchmark(failed)
    }
    paths = {}
    for arm, value in envelopes.items():
        target = root / arm / arms.FILENAME
        target.parent.mkdir()
        io.publish_json(target, value)
        paths[arm] = target
    return {
        "report": report,
        "config": config,
        "envelopes": envelopes,
        "failed": failed_envelopes,
        "paths": paths,
    }


@pytest.fixture(autouse=True)
def no_execution(evidence: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail(
            "study verification must not construct/reset a plant or load a policy"
        )

    monkeypatch.setattr(PlantCyclicMarketEnvV2Beta, "__init__", forbidden)
    monkeypatch.setattr(PlantCyclicMarketEnvV2Beta, "reset", forbidden)
    monkeypatch.setattr(policies, "load_policy", forbidden)


def _publish(
    tmp_path: Path, value: dict[str, Any], filename: str = arms.FILENAME
) -> Path:
    path = tmp_path.resolve() / filename
    io.publish_json(path, value)
    return path


def _rehash(value: dict[str, Any]) -> dict[str, Any]:
    body = benchmark._timing_free(value["run"])
    body.pop("deterministic_evidence_sha256", None)
    value["run"]["deterministic_evidence_sha256"] = domain_sha256(
        verifier.STUDY_ARM_RESULT_FORMAT, body
    )
    return authenticate(value)


@pytest.mark.parametrize("arm", benchmark.BETA_CONTROLLER_ARM_ORDER)
def test_each_standalone_arm_reconstructs_full_evidence(
    evidence: dict[str, Any], arm: str
) -> None:
    result = arms.verify_synthetic_arm_file(evidence["paths"][arm])
    assert result["source_evidence_verified"] is True
    assert result["registered_checkpoint_verified"] is False
    assert result["result_claim"] is None
    assert "value_comparison_eligible" not in result["classification"]
    assert "paired_cell_ineligible" not in result["classification"]["exclusion_reasons"]
    if arm == "H":
        assert result["classification"]["execution_complete"] is False
        assert result["economic_metrics"]["net_operating_value_eur"] == 0.0


def test_primary_pair_survives_context_failure_and_detached_reopening(
    evidence: dict[str, Any], tmp_path: Path
) -> None:
    assert evidence["report"]["cells"][0]["value_comparison_eligible"] is False
    pair = pairwise.analyze_synthetic_pair(
        evidence["paths"]["PPO"], evidence["paths"]["E-beta"]
    )
    assert pair["value_comparison_eligible"] is True
    assert pair["exclusion_reasons"] == []
    runs = evidence["report"]["cells"][0]["runs"]
    expected = (
        runs["PPO"]["economic_metrics"]["net_operating_value_eur"]
        - runs["E-beta"]["economic_metrics"]["net_operating_value_eur"]
    )
    assert pair["delta_eur"] == expected
    assert set(pair["arms"]) == {"policy", "e_beta"}
    result = pairwise.verify_synthetic_pair_file(
        _publish(tmp_path, pair, pairwise.FILENAME)
    )
    assert result["delta_eur"] == expected
    assert result["source_evidence_verified"] is True
    assert result["result_claim"] is None


@pytest.mark.parametrize("missing", ["policy", "e_beta", "both"])
def test_explicit_missing_arm_has_no_value_or_sensitivity(
    evidence: dict[str, Any], missing: str
) -> None:
    pair = pairwise.analyze_synthetic_pair(
        None if missing in {"policy", "both"} else evidence["paths"]["PPO"],
        None if missing in {"e_beta", "both"} else evidence["paths"]["E-beta"],
    )
    assert pair["value_comparison_eligible"] is False
    assert pair["delta_eur"] is pair["sensitivity"] is None


@pytest.mark.parametrize(
    "defect",
    [
        "schema",
        "stage",
        "protected_seed",
        "boolean_seed",
        "registered",
        "claim",
        "protocol",
        "contract",
        "actor",
        "settings",
        "baseline_checkpoint",
    ],
)
def test_early_scope_rejection_before_manifest_regeneration(
    evidence: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    value = deepcopy(evidence["envelopes"]["E-beta"])
    if defect == "schema":
        value["schema"] = "production-arm"
    elif defect == "stage":
        value["stage"] = "held_out"
    elif defect == "protected_seed":
        value["seed"] = 3_000_000
    elif defect == "boolean_seed":
        value["seed"] = True
    elif defect == "registered":
        value["training_task"]["beta_rl"] = {"run_kind": "registered"}
    elif defect == "claim":
        value["result_claim"] = "production_eligible"
    elif defect == "protocol":
        value["protocol"]["raw_sha256"] = "0" * 64
    elif defect == "contract":
        value["analysis_contract_sha256"] = "0" * 64
    elif defect == "actor":
        value["actor"]["optimizer_seed"] = 101
    elif defect == "settings":
        value["service_mpc_settings"]["task_horizon_h"] = 168.0
    elif defect == "baseline_checkpoint":
        value["checkpoint"] = evidence["envelopes"]["PPO"]["checkpoint"]

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("invalid synthetic scope reached manifest regeneration")

    monkeypatch.setattr(verifier, "_verify_audit_manifest", forbidden)
    with pytest.raises(ValueError):
        arms.verify_synthetic_arm_file(_publish(tmp_path, authenticate(value)))


@pytest.mark.parametrize(
    "defect",
    [
        "minute",
        "cycle",
        "completed",
        "safety",
        "forced",
        "invalid",
        "service",
        "profit",
        "forecast",
        "descriptor",
        "legacy_flag",
    ],
)
def test_rehashed_physical_or_controller_forgery_is_rejected(
    evidence: dict[str, Any], tmp_path: Path, defect: str
) -> None:
    value = deepcopy(evidence["envelopes"]["PPO"])
    run = value["run"]
    if defect == "minute":
        run["evidence"]["minute_truth_states"].pop()
    elif defect == "cycle":
        run["cycle_settlements"].clear()
    elif defect == "completed":
        run["completed_decisions"] -= 1
    elif defect == "safety":
        run["hard_safety_passed"] = False
    elif defect == "forced":
        run["physical_metrics"]["electrolyser_forced_stops"] += 1
    elif defect == "invalid":
        run["physical_metrics"]["out_of_range_decisions"] += 1
    elif defect == "service":
        run["physical_metrics"]["delivered_kg"] += 1.0
    elif defect == "profit":
        run["economic_metrics"]["net_operating_value_eur"] += 1.0
    elif defect == "forecast":
        run["evidence"]["boundary_ledger"]["decision_boundaries"][0][
            "typed_observation"
        ]["market"]["current_eur_per_mwh"] = 5000.0
    elif defect == "descriptor":
        run["controller"]["failure_policy"] = "fallback to heuristic"
    elif defect == "legacy_flag":
        run["value_comparison_eligible"] = True
    with pytest.raises(ValueError):
        arms.verify_synthetic_arm_file(_publish(tmp_path, _rehash(value)))


@pytest.mark.parametrize(
    "defect", ["value", "component", "delivered", "inventory", "wear"]
)
def test_failed_zero_interval_metrics_cannot_fabricate_value_or_exposure(
    evidence: dict[str, Any], tmp_path: Path, defect: str
) -> None:
    value = deepcopy(evidence["envelopes"]["H"])
    run = value["run"]
    assert run["completed_decisions"] == 0
    if defect == "value":
        run["economic_metrics"]["net_operating_value_eur"] = 1.0
    elif defect == "component":
        run["economic_metrics"]["components_eur"] = {"delivered_service_value": 1.0}
    elif defect == "delivered":
        run["physical_metrics"]["delivered_kg"] = 1.0
    elif defect == "inventory":
        run["economic_metrics"]["terminal_ch4_equivalent_kg"] += 1.0
    elif defect == "wear":
        run["physical_metrics"]["electrolyser_operating_time_s"] = 1.0
    with pytest.raises(ValueError, match="zero-interval"):
        arms.verify_synthetic_arm_file(_publish(tmp_path, _rehash(value)))


def test_primary_failed_arm_retained_without_value(
    evidence: dict[str, Any], tmp_path: Path
) -> None:
    paths = {}
    for arm in ("PPO", "E-beta"):
        root = tmp_path / arm
        root.mkdir()
        paths[arm] = _publish(root, evidence["failed"][arm])
    result = pairwise.analyze_synthetic_pair(paths["PPO"], paths["E-beta"])
    assert result["value_comparison_eligible"] is False
    assert result["delta_eur"] is result["sensitivity"] is None
    assert "e_beta_execution_complete" in result["exclusion_reasons"]
    assert result["arms"]["e_beta"]["classification"]["failure"] is not None


@pytest.mark.parametrize(
    "defect", ["delta", "eligibility", "sensitivity", "missing", "source", "reversed"]
)
def test_detached_pair_rejects_rehashed_substitution(
    evidence: dict[str, Any], tmp_path: Path, defect: str
) -> None:
    pair = pairwise.analyze_synthetic_pair(
        evidence["paths"]["PPO"], evidence["paths"]["E-beta"]
    )
    if defect == "delta":
        pair["delta_eur"] += 10.0
    elif defect == "eligibility":
        pair["value_comparison_eligible"] = False
    elif defect == "sensitivity":
        pair["sensitivity"]["settings"][0]["delta_eur"] += 10.0
    elif defect == "missing":
        pair["inputs"]["policy"] = None
    elif defect == "source":
        pair["arms"]["policy"]["source_provenance"]["git_revision"] = "1" * 40
    elif defect == "reversed":
        pair["inputs"] = {
            "policy": str(evidence["paths"]["E-beta"]),
            "e_beta": str(evidence["paths"]["PPO"]),
        }
    with pytest.raises(ValueError):
        pairwise.verify_synthetic_pair_file(
            _publish(tmp_path, authenticate(pair), pairwise.FILENAME)
        )


@pytest.mark.parametrize(
    "defect", ["filename", "symlink", "hardlink", "noncanonical", "duplicate", "digest"]
)
def test_file_custody_rejects_aliases_and_noncanonical_documents(
    evidence: dict[str, Any], tmp_path: Path, defect: str
) -> None:
    value = evidence["envelopes"]["PPO"]
    target = tmp_path.resolve() / arms.FILENAME
    if defect == "filename":
        target = _publish(tmp_path, value, "other.json")
    elif defect == "symlink":
        target.symlink_to(evidence["paths"]["PPO"])
    elif defect == "hardlink":
        target.hardlink_to(evidence["paths"]["PPO"])
    elif defect == "noncanonical":
        target.write_bytes(canonical_bytes(value) + b"\n")
    elif defect == "duplicate":
        target.write_bytes(
            canonical_bytes(value).replace(b"{\n", b'{\n  "schema": "duplicate",\n', 1)
        )
    elif defect == "digest":
        target.write_bytes(canonical_bytes({**value, "sha256": "0" * 64}))
    try:
        with pytest.raises(ValueError):
            arms.verify_synthetic_arm_file(target)
    finally:
        if defect == "hardlink":
            target.unlink()


@pytest.mark.parametrize("arm", benchmark.BETA_CONTROLLER_ARM_ORDER)
def test_preflight_failure_is_retained_without_public_state(
    evidence: dict[str, Any],
    tmp_path: Path,
    arm: str,
) -> None:
    value = deepcopy(evidence["envelopes"][arm])
    cell = benchmark._preflight_failed_cell(
        9_000_063, RuntimeError("synthetic preflight rejection")
    )
    value.update(
        seed=9_000_063,
        manifest_identity=None,
        audit_manifest=None,
        run=arms._standalone_result(cell["runs"][arm]),
    )
    result = arms.verify_synthetic_arm_file(_publish(tmp_path, authenticate(value)))
    assert result["classification"]["status"] == "not_run_preflight_failed"
    assert result["initial_public_state_sha256"] is None
    assert result["public_issue_sha256"] == []
    assert result["economic_metrics"] is result["physical_metrics"] is None


@pytest.mark.parametrize("defect", ["failure", "manifest", "movement", "unknown"])
def test_preflight_envelope_cannot_retain_movement_or_hide_reason(
    evidence: dict[str, Any],
    tmp_path: Path,
    defect: str,
) -> None:
    value = deepcopy(evidence["envelopes"]["PPO"])
    cell = benchmark._preflight_failed_cell(
        9_000_063, RuntimeError("synthetic preflight rejection")
    )
    value.update(
        seed=9_000_063,
        manifest_identity=None,
        audit_manifest=None,
        run=arms._standalone_result(cell["runs"]["PPO"]),
    )
    if defect == "failure":
        value["run"]["failure"] = None
    elif defect == "manifest":
        value["audit_manifest"] = evidence["envelopes"]["PPO"]["audit_manifest"]
    elif defect == "movement":
        value["run"]["evidence"]["minute_truth_states"] = [{"step": 1}]
    elif defect == "unknown":
        value["run"]["evidence"]["hidden_executions"] = []
    with pytest.raises(ValueError):
        arms.verify_synthetic_arm_file(_publish(tmp_path, _rehash(value)))


@pytest.fixture(scope="module")
def admitted_pair(evidence: dict[str, Any]) -> dict[str, Any]:
    return pairwise.analyze_synthetic_pair(
        evidence["paths"]["PPO"], evidence["paths"]["E-beta"]
    )


@pytest.mark.parametrize(
    "gate",
    [
        "policy_execution_complete",
        "e_beta_execution_complete",
        "policy_hard_safety_passed",
        "e_beta_hard_safety_passed",
        "policy_cycle_feasible",
        "e_beta_cycle_feasible",
        "same_stage_and_seed",
        "same_source",
        "same_manifest",
        "same_initial_public_state",
        "same_common_public_issues",
        "consistent_solver_versions",
        "service_comparable",
    ],
)
def test_every_pair_gate_independently_excludes_value(
    admitted_pair: dict[str, Any], gate: str
) -> None:
    # Exercise gate arithmetic on already admitted projections. The public API
    # never accepts these in-memory projections as source evidence.
    left, right = deepcopy(admitted_pair["arms"]["policy"]), deepcopy(
        admitted_pair["arms"]["e_beta"]
    )
    for actor, target in (("policy", left), ("e_beta", right)):
        prefix = actor + "_"
        if gate.startswith(prefix):
            target["classification"][gate[len(prefix) :]] = False
    if gate == "same_stage_and_seed":
        left["seed"] += 1
    elif gate == "same_source":
        left["source_provenance"]["git_revision"] = "1" * 40
    elif gate == "same_manifest":
        left["manifest_identity"]["manifest_sha256"] = "1" * 64
    elif gate == "same_initial_public_state":
        left["initial_public_state_sha256"] = "1" * 64
    elif gate == "same_common_public_issues":
        left["public_issue_sha256"][-1] = "1" * 64
    elif gate == "consistent_solver_versions":
        left["highs_versions"] = ["different"]
    elif gate == "service_comparable":
        left["physical_metrics"]["delivered_kg"] += 2e-6
    result = pairwise._comparison(left, right)
    assert result["exclusion_reasons"] == [gate]
    assert result["delta_eur"] is result["sensitivity"] is None


@pytest.mark.parametrize("quantity", ["requested_kg", "delivered_kg", "unmet_kg"])
@pytest.mark.parametrize("difference, eligible", [(1e-6, True), (1.000001e-6, False)])
def test_service_gate_uses_each_quantity_and_absolute_inclusive_tolerance(
    admitted_pair: dict[str, Any],
    quantity: str,
    difference: float,
    eligible: bool,
) -> None:
    left, right = deepcopy(admitted_pair["arms"]["policy"]), deepcopy(
        admitted_pair["arms"]["e_beta"]
    )
    left["physical_metrics"][quantity] = difference
    right["physical_metrics"][quantity] = 0.0
    result = pairwise._comparison(left, right)
    assert result["checks"]["service_comparable"] is eligible


def test_coherent_source_relabelling_cannot_make_a_paired_comparison(
    evidence: dict[str, Any],
    tmp_path: Path,
) -> None:
    value = deepcopy(evidence["envelopes"]["PPO"])
    value["source_provenance"]["git_revision"] = "1" * 40
    changed = _publish(tmp_path, authenticate(value))
    result = pairwise.analyze_synthetic_pair(changed, evidence["paths"]["E-beta"])
    assert result["exclusion_reasons"] == ["same_source"]
    assert result["delta_eur"] is None


@pytest.mark.parametrize("phase", ["admission", "detached_end"])
def test_graph_reopening_rejects_even_identical_file_replacement(
    evidence: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    policy = _publish(tmp_path, evidence["envelopes"]["PPO"])
    pair = pairwise.analyze_synthetic_pair(policy, evidence["paths"]["E-beta"])
    target = _publish(tmp_path, pair, pairwise.FILENAME)

    def replace_policy() -> None:
        replacement = tmp_path / "replacement.json"
        replacement.write_bytes(policy.read_bytes())
        replacement.replace(policy)

    if phase == "admission":
        original = arms._verify_fixture_envelope

        def change_during_other_arm(envelope: dict[str, Any]) -> dict[str, Any]:
            result = original(envelope)
            if envelope["actor"]["arm"] == "E-beta":
                replace_policy()
            return result

        monkeypatch.setattr(arms, "_verify_fixture_envelope", change_during_other_arm)
    else:
        original_read = io.read_regular
        pair_reads = 0

        def change_during_final_pair_read(path: Path) -> bytes:
            nonlocal pair_reads
            raw = original_read(path)
            if path == target:
                pair_reads += 1
                if pair_reads == 2:
                    replace_policy()
            return raw

        monkeypatch.setattr(io, "read_regular", change_during_final_pair_read)
    with pytest.raises(ValueError, match="source changed"):
        pairwise.verify_synthetic_pair_file(target)


def test_converter_rejects_protected_seed_before_legacy_verifier(
    evidence: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = deepcopy(evidence["report"])
    report["seeds"] = [2_000_000]

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("protected seed reached legacy verifier")

    monkeypatch.setattr(verifier, "verify_beta_controller_benchmark", forbidden)
    with pytest.raises(ValueError, match="9,000,000"):
        arms.synthetic_arms_from_benchmark(report)

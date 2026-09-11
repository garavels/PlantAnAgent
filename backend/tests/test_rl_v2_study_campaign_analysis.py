"""All-slot arithmetic fixtures, never registered policy or episode evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any

import pytest

from plant_backend.rl.v2.study import campaign_analysis as analysis
from plant_backend.rl.v2.study import registry, pairwise
from plant_backend.rl.v2.study.contract import (
    BETA_CORE_ALGORITHM_IDS,
    BETA_OPTIMIZER_SEEDS,
    COMPONENT_NAMES,
)
from plant_backend.rl.v2.study.latency import latency_summary
from plant_backend.rl.v2.training.custody_json import authenticate, canonical_bytes
from plant_backend.rl.v2.study.training_scorecards import TrainingSlot
from test_rl_v2_study_operational import arithmetic_episode, project, _rollouts

SEEDS = list(range(9_000_200, 9_000_232))
HOST = "c" * 64
FACTS = analysis.CustodyFacts(True, True, True, True, True)


@pytest.fixture(autouse=True)
def forbid_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta
    from plant_backend.rl import policies

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail(
            "campaign arithmetic cannot construct/reset a plant or load a policy"
        )

    monkeypatch.setattr(PlantCyclicMarketEnvV2Beta, "__init__", forbidden)
    monkeypatch.setattr(PlantCyclicMarketEnvV2Beta, "reset", forbidden)
    monkeypatch.setattr(policies, "load_policy", forbidden)


def _timing(arm: str, digest: str, samples: list[int] | None = None) -> dict[str, Any]:
    channels = {name: [] for name in analysis.CHANNELS}
    channels["inference"] = [100_000_000] if samples is None else samples
    if arm != "PPO":
        channels["inference"] = []
    return {
        "arm": arm,
        "arm_envelope_sha256": digest,
        "raw_sha256": hashlib.sha256(digest.encode()).hexdigest(),
        "host_sha256": HOST,
        "torch_single_thread": True,
        "warmup_status": "complete" if arm == "PPO" else "not_applicable",
        "channel_samples_ns": channels,
        "channels": {key: latency_summary(value) for key, value in channels.items()},
        "inference_failures": 0,
        "controller_failures": 0,
        "interval_failures": 0,
    }


@pytest.fixture(scope="module")
def inputs() -> dict[str, Any]:
    # Reuse the reset-free arithmetic template, not a physical verification fixture.
    template = project(arithmetic_episode.__wrapped__())
    cells = []
    for identity in registry.cells_for_seeds(SEEDS):
        actor = identity["actor"]
        seed = identity["episode_seed"]
        index = identity["episode_index"]
        value = 168.0
        if actor["arm"] == "PPO":
            value += (
                (10, 5, 2)[BETA_CORE_ALGORITHM_IDS.index(actor["algorithm"])]
                + BETA_OPTIMIZER_SEEDS.index(actor["optimizer_seed"])
                + index / 100
            )
        op = deepcopy(template)
        op.update(
            seed=seed,
            actor=registry.actor_identity(actor),
            market_regime=("calm", "volatile", "scarcity")[index % 3],
        )
        op["economic_metrics"]["net_operating_value_eur"] = value
        op["economic_metrics"]["components_eur"] = {
            key: value if i == 0 else 0.0 for i, key in enumerate(COMPONENT_NAMES)
        }
        for cycle in op["cycles"]:
            cycle["net_operating_value_eur"] = value / 7
            cycle["components_eur"]["delivered_service_value"] = value / 7
        # Context failure must remain visible without removing valid primary pairs.
        if actor["arm"] in {"H", "M24"}:
            op.update(hard_safety_passed=False)
        op = authenticate(op)
        digest = hashlib.sha256(identity["stem"].encode()).hexdigest()
        arm = {
            "stage": "fixture",
            "seed": seed,
            "actor": registry.actor_identity(actor),
            "source_provenance": {"fixture": "arithmetic only"},
            "manifest_identity": {"fixture_seed": seed},
            "initial_public_state_sha256": str(seed),
            "public_issue_sha256": [str(seed)],
            "highs_versions": ["fixture"],
            "envelope_sha256": digest,
            "classification": {
                "execution_complete": True,
                "hard_safety_passed": op["hard_safety_passed"],
                "cycle_feasible": True,
            },
            "physical_metrics": op["physical_metrics"],
            "economic_metrics": op["economic_metrics"],
            "operational": op,
        }
        cells.append(
            analysis.CellInput(
                identity, "verified", None, arm, _timing(actor["arm"], digest)
            )
        )
    ledgers = {family: tuple(_rollouts(family)) for family in BETA_CORE_ALGORITHM_IDS}
    slots = [
        TrainingSlot(family, seed, True, True, None, ledgers[family])
        for family in BETA_CORE_ALGORITHM_IDS
        for seed in BETA_OPTIMIZER_SEEDS
    ]
    return {
        "cells": cells,
        "training_slots": slots,
        "facts": FACTS,
        "episode_seeds": SEEDS,
        "expected_host_sha256": HOST,
        "expected_training_rollouts": 2,
    }


@pytest.fixture(scope="module")
def development(inputs: dict[str, Any]) -> dict[str, Any]:
    return analysis.analyze_campaign(stage="development", **inputs)


@pytest.fixture(scope="module")
def heldout(inputs: dict[str, Any]) -> dict[str, Any]:
    return analysis.analyze_campaign(
        stage="heldout",
        frozen_champions={family: 101 for family in BETA_CORE_ALGORITHM_IDS},
        **inputs,
    )


def test_registry_exact_order_and_destinations_without_side_effects(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve() / "training"
    for stage, start in (("development", 2_000_000), ("heldout", 3_000_000)):
        record = registry.stage_registry(root, stage)
        assert len(record["cells"]) == 576
        assert record["episode_seeds"] == list(range(start, start + 32))
        assert [row["actor"]["arm"] for row in record["cells"][:3]] == [
            "H",
            "M24",
            "E-beta",
        ]
        assert [
            row["actor"]["training_order"] for row in record["cells"][3:18]
        ] == list(range(1, 16))
        assert [row["episode_index"] for row in record["cells"][:18]] == [0] * 18
        assert record["cells"][-1]["actor"]["optimizer_seed"] == 503
        assert Path(record["stage_root"]).name == f"training.{stage}-v1"
        assert len({row["arm_path"] for row in record["cells"]}) == 576
        assert (
            record["archive_schema"]
            == "plant-beta-constrained-rl-study-canonical-json-gzip-v1"
        )
        assert all(
            Path(row["arm_path"]).name == "arm.json.gz" for row in record["cells"]
        )
        assert all(
            Path(row["latency_path"]).name == "latency.json.gz"
            for row in record["cells"]
        )
        assert registry.verify_registry(record) == record
    assert not root.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "mutation",
    [
        "stage",
        "seed",
        "order",
        "missing",
        "actor",
        "path",
        "contract",
        "archive",
        "uncompressed_path",
    ],
)
def test_rehashed_registry_substitution_rejected(tmp_path: Path, mutation: str) -> None:
    raw = registry.stage_registry(tmp_path.resolve() / "training", "development")
    if mutation == "stage":
        raw["stage"] = "heldout"
    elif mutation == "seed":
        raw["episode_seeds"][0] += 32
    elif mutation == "order":
        raw["cells"].reverse()
    elif mutation == "missing":
        raw["cells"].pop()
    elif mutation == "actor":
        raw["cells"][3]["actor"]["optimizer_seed"] = 211
    elif mutation == "path":
        raw["cells"][0]["arm_path"] = str(tmp_path / "replacement.json")
    elif mutation == "archive":
        raw["archive_schema"] = "unregistered archive"
    elif mutation == "uncompressed_path":
        raw["cells"][0]["latency_path"] = raw["cells"][0]["latency_path"][:-3]
    else:
        raw["latency_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        registry.verify_registry(authenticate(raw))


@pytest.mark.parametrize("stage", ["fixture", "training", "held_out", "", None])
def test_unknown_production_stage_rejected(tmp_path: Path, stage: object) -> None:
    with pytest.raises(ValueError):
        registry.stage_registry(tmp_path.resolve() / "training", stage)


def test_registry_rejects_repository_relative_and_symlink_roots(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        registry.stage_registry(Path("training"), "development")
    with pytest.raises(ValueError):
        registry.stage_registry(registry.repository_root() / "artifacts", "development")
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError):
        registry.stage_registry(alias, "development")


def test_complete_composition_keeps_context_failures_and_all_families(
    development: dict[str, Any]
) -> None:
    assert development["cell_count"] == 576
    assert development["state_counts"]["verified"] == 576
    assert len(development["actors"]) == 18
    assert (
        development["actors"]["h"]["operational"]["hard_safety_complete_episode"][
            "successes"
        ]
        == 0
    )
    for family in BETA_CORE_ALGORITHM_IDS:
        assert development["frozen_champions"][family] == 503
        assert (
            development["development_selections"][family]["development_admission"][
                "successes"
            ]
            == 5
        )
        assert len(development["families"][family]["sensitivity"]["settings"]) == 16
        assert (
            development["families"][family]["primary_family_summary"][
                "policy_admission"
            ]["total"]
            == 5
        )
    assert (
        development["development_opening_requirements"][
            "all_calculated_requirements_met"
        ]
        is True
    )
    assert (
        development["development_opening_requirements"]["execution_authorized"] is False
    )
    assert development["source_evidence_verified"] is False
    assert development["result_claim"] is None


def test_secondary_statistics_are_actual_matched_policy_differences(
    development: dict[str, Any]
) -> None:
    secondary = development["families"][BETA_CORE_ALGORITHM_IDS[0]][
        "secondary_vs_scalar"
    ]
    assert [row["optimizer_seed"] for row in secondary["policies"]] == list(
        BETA_OPTIMIZER_SEEDS
    )
    assert all(
        row["summary"]["statistics"]["mean"] == pytest.approx(8)
        for row in secondary["policies"]
    )
    assert secondary["family_summary"]["statistics"]["mean"] == pytest.approx(8)
    assert all(
        "scalar_ppo_execution_complete" in pair["checks"]
        for row in secondary["policies"]
        for pair in row["pairs"]
    )
    assert all(
        pair["sensitivity"] is None
        for row in secondary["policies"]
        for pair in row["pairs"]
    )


def test_secondary_pair_survives_absent_e_beta(inputs: dict[str, Any]) -> None:
    policy = inputs["cells"][3].arm
    scalar = inputs["cells"][13].arm
    assert pairwise._comparison(policy, None)["value_comparison_eligible"] is False
    direct = pairwise._comparison(policy, scalar, comparator="scalar_ppo")
    assert direct["value_comparison_eligible"] is True
    assert direct["delta_eur"] == pytest.approx(8)
    assert direct["sensitivity"] is None
    with pytest.raises(ValueError):
        pairwise._comparison(policy, scalar, comparator="replacement")


def test_heldout_uses_frozen_choice_without_reselection(
    heldout: dict[str, Any]
) -> None:
    assert heldout["development_selections"] is None
    assert all(value == 101 for value in heldout["frozen_champions"].values())
    assert (
        heldout["heldout_success_requirements"]["all_calculated_requirements_met"]
        is True
    )
    assert len(heldout["heldout_success_requirements"]["checks"]) == 10
    assert heldout["heldout_success_requirements"]["result_claim"] is None
    assert heldout["source_evidence_verified"] is False


@pytest.mark.parametrize("clause", range(1, 11))
def test_each_heldout_clause_can_independently_prevent_success(
    heldout: dict[str, Any], clause: int
) -> None:
    policies = deepcopy(heldout["families"][BETA_CORE_ALGORITHM_IDS[0]]["policies"])
    champion = policies[0]
    facts = FACTS
    if clause == 1:
        champion["primary_pairs"][0]["value_comparison_eligible"] = False
    elif clause == 2:
        champion["physical_arms"][0]["classification"]["hard_safety_passed"] = False
    elif clause == 3:
        champion["physical_arms"][0]["classification"]["cycle_feasible"] = False
    elif clause == 4:
        champion["primary_pairs"][0]["checks"]["service_comparable"] = False
    elif clause == 5:
        champion["primary_summary"]["bootstrap_95"]["iqm"][0] = 0.0
    elif clause == 6:
        champion["primary_summary"]["sign_counts"]["positive"] = 23
    elif clause == 7:
        champion["primary_summary"]["statistics"]["cvar20"] = -1e-9
    elif clause == 8:
        champion["latency"]["channels"]["inference"]["p99"] = 100_000_001
    elif clause == 9:
        policies[3]["primary_summary"] = None
        policies[4]["primary_summary"] = None
    else:
        facts = replace(facts, published_fields_verified=False)
    result = analysis.heldout_clauses(
        champion=champion, pi_policies=policies, facts=facts
    )
    assert result["all_calculated_requirements_met"] is False
    assert (
        next(
            value
            for key, value in result["checks"].items()
            if key.startswith(f"{clause}_")
        )
        is False
    )


@pytest.mark.parametrize("change", ["hardware", "missing", "host", "warmup"])
def test_latency_claim_needs_full_registered_measurement(
    heldout: dict[str, Any], change: str
) -> None:
    policies = deepcopy(heldout["families"][BETA_CORE_ALGORITHM_IDS[0]]["policies"])
    champion = policies[0]
    facts = FACTS
    if change == "hardware":
        facts = replace(facts, registered_hardware_verified=False)
    else:
        champion["latency"][
            {
                "missing": "all_32_measured",
                "host": "host_matches_registered_identity",
                "warmup": "warmups_complete",
            }[change]
        ] = False
    result = analysis.heldout_clauses(
        champion=champion, pi_policies=policies, facts=facts
    )
    assert result["checks"]["8_registered_inference_p99_at_most_100_ms"] is False


def test_latency_pool_uses_all_calls_including_failures_not_episode_percentiles() -> (
    None
):
    records = [_timing("PPO", "a" * 64, [100_000_001] * 100)] + [
        _timing("PPO", "b" * 64, [1])
    ] * 31
    records[0]["inference_failures"] = 1
    pooled = analysis.pooled_latency(
        SEEDS, records, arm="PPO", expected_host_sha256=HOST
    )
    assert pooled["channels"]["inference"]["sample_count"] == 131
    assert pooled["channels"]["inference"]["p99"] == 100_000_001
    assert (
        sum(row["channels"]["inference"]["p99"] for row in records) / 32 < 100_000_000
    )
    assert pooled["inference_failures"] == 1
    assert pooled["channel_samples_ns"]["inference"] == [100_000_001] * 100 + [1] * 31
    records[0]["channels"]["inference"]["p99"] = 1
    with pytest.raises(ValueError):
        analysis.pooled_latency(SEEDS, records, arm="PPO", expected_host_sha256=HOST)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "order",
        "identity",
        "failure",
        "actor",
        "source",
        "timing",
        "stage",
        "training",
        "choice",
    ],
)
def test_bad_campaign_inputs_rejected_before_aggregation(
    inputs: dict[str, Any], mutation: str
) -> None:
    args = {
        **inputs,
        "cells": inputs["cells"].copy(),
        "training_slots": inputs["training_slots"].copy(),
    }
    cells = args["cells"]
    if mutation == "missing":
        cells.pop()
    elif mutation == "order":
        cells[0], cells[1] = cells[1], cells[0]
    elif mutation == "identity":
        cells[0] = replace(cells[0], identity={**cells[0].identity, "order": 3})
    elif mutation == "failure":
        cells[0] = replace(cells[0], failure="hidden")
    elif mutation == "timing":
        cells[0] = replace(
            cells[0], latency={**cells[0].latency, "arm_envelope_sha256": "0" * 64}
        )
    elif mutation == "training":
        args["training_slots"][0] = replace(
            args["training_slots"][0],
            checkpoint_admitted=False,
            failure="checkpoint_rejected",
            rollouts=None,
        )
    elif mutation == "choice":
        args["frozen_champions"] = {family: 101 for family in BETA_CORE_ALGORITHM_IDS}
    else:
        arm = deepcopy(cells[1].arm)
        if mutation == "actor":
            arm["actor"]["arm"] = "H"
        elif mutation == "source":
            arm["source_provenance"] = {"different": True}
        else:
            arm["stage"] = "heldout"
        cells[1] = replace(cells[1], arm=arm)
    with pytest.raises(ValueError):
        analysis.analyze_campaign(stage="development", **args)


def test_missing_pair_forbids_vector_and_scalar_order_is_exact() -> None:
    pairs = [{"value_comparison_eligible": True, "delta_eur": 1.0}] * 32
    assert analysis._vector(pairs, admitted=True) == [1.0] * 32
    assert analysis._vector(pairs, admitted=False) is None
    pairs[0] = {"value_comparison_eligible": False, "delta_eur": None}
    assert analysis._vector(pairs, admitted=True) is None
    with pytest.raises(ValueError):
        analysis._vector(pairs[:-1], admitted=True)


def test_custody_flags_do_not_coerce() -> None:
    with pytest.raises(ValueError):
        analysis.CustodyFacts(training_verified=1)


def test_all_failed_training_retains_480_skipped_and_96_missing_baseline_slots() -> (
    None
):
    slots = [
        TrainingSlot(family, seed, False, False, "training_failed", None)
        for family in BETA_CORE_ALGORITHM_IDS
        for seed in BETA_OPTIMIZER_SEEDS
    ]
    cells = [
        analysis.CellInput(
            identity,
            (
                "training_not_admitted"
                if identity["actor"]["arm"] == "PPO"
                else "not_started"
            ),
            (
                "training_failed"
                if identity["actor"]["arm"] == "PPO"
                else "stage_not_opened"
            ),
            None,
            None,
        )
        for identity in registry.cells_for_seeds(SEEDS)
    ]
    result = analysis.analyze_campaign(
        stage="development",
        episode_seeds=SEEDS,
        cells=cells,
        training_slots=slots,
        facts=analysis.CustodyFacts(),
    )
    assert result["state_counts"]["training_not_admitted"] == 480
    assert result["state_counts"]["not_started"] == 96
    assert all(choice is None for choice in result["frozen_champions"].values())
    for family in BETA_CORE_ALGORITHM_IDS:
        assert result["families"][family]["primary_family_summary"][
            "missing_optimizer_seeds"
        ] == list(BETA_OPTIMIZER_SEEDS)
        assert all(
            policy["primary_vector_eur"] is None
            for policy in result["families"][family]["policies"]
        )
    assert (
        result["development_opening_requirements"]["all_calculated_requirements_met"]
        is False
    )
    cells[3] = replace(cells[3], failure="different_failure")
    with pytest.raises(ValueError, match="training custody reason"):
        analysis.analyze_campaign(
            stage="development",
            episode_seeds=SEEDS,
            cells=cells,
            training_slots=slots,
            facts=analysis.CustodyFacts(),
        )


@pytest.mark.parametrize(
    "choice",
    [
        None,
        {},
        {family: 1 for family in BETA_CORE_ALGORITHM_IDS},
        {family: True for family in BETA_CORE_ALGORITHM_IDS},
    ],
)
def test_heldout_requires_complete_frozen_choices(
    inputs: dict[str, Any], choice: object
) -> None:
    with pytest.raises(ValueError):
        analysis.analyze_campaign(stage="heldout", frozen_champions=choice, **inputs)


def test_unselected_or_unadmitted_champion_never_gets_success(
    heldout: dict[str, Any]
) -> None:
    policies = heldout["families"][BETA_CORE_ALGORITHM_IDS[0]]["policies"]
    result = analysis.heldout_clauses(champion=None, pi_policies=policies, facts=FACTS)
    assert result["all_calculated_requirements_met"] is False
    assert result["checks"]["1_all_32_primary_pairs"] is False
    assert result["checks"]["8_registered_inference_p99_at_most_100_ms"] is False


def test_heldout_requires_frozen_identities_and_distinct_pi_policies(
    heldout: dict[str, Any]
) -> None:
    policies = heldout["families"][BETA_CORE_ALGORITHM_IDS[0]]["policies"]
    result = analysis.heldout_clauses(
        champion=policies[0],
        pi_policies=policies,
        facts=replace(FACTS, identities_frozen=False),
    )
    assert result["all_calculated_requirements_met"] is False
    with pytest.raises(ValueError, match="independent PI identities"):
        analysis.heldout_clauses(
            champion=policies[0], pi_policies=[policies[0]] * 5, facts=FACTS
        )
    substitute = deepcopy(policies[0])
    substitute["primary_summary"]["statistics"]["iqm"] += 1
    with pytest.raises(ValueError, match="unchanged frozen family member"):
        analysis.heldout_clauses(champion=substitute, pi_policies=policies, facts=FACTS)

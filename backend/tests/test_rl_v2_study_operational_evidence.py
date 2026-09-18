"""Detached all-slot operational evidence with real fixture minute ledgers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from plant_backend.rl.v2.study import arm_evidence as arms
from plant_backend.rl.v2.study import operational_evidence as reports
from plant_backend.rl.v2.training import custody_io as io
from plant_backend.rl.v2.training.custody_json import authenticate, canonical_bytes
from test_rl_v2_study_arm_evidence import evidence, no_execution, _rehash

SEEDS = list(range(9_000_061, 9_000_093))


@pytest.fixture(scope="module")
def scorecard(
    evidence: dict[str, Any], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    report = reports.synthetic_operational_evidence(
        SEEDS, [evidence["paths"]["PPO"]] + [None] * 31, cycle_count=1
    )
    target = (
        tmp_path_factory.mktemp("study-operational-evidence").resolve()
        / reports.FILENAME
    )
    io.publish_json(target, report)
    return {"report": report, "path": target}


@pytest.mark.parametrize("arm", ["H", "M24", "E-beta", "PPO"])
def test_all_arms_operational_projection_is_physically_reconstructed(
    evidence: dict[str, Any], arm: str
) -> None:
    admitted = arms.verify_synthetic_arm_file(evidence["paths"][arm])
    operational = admitted["operational"]
    original = evidence["envelopes"][arm]["run"]
    assert operational["physical_metrics"] == original["physical_metrics"]
    assert operational["economic_metrics"] == original["economic_metrics"]
    assert operational["cycles"][0]["settlement"] == (
        original["cycle_settlements"][0] if original["cycle_settlements"] else None
    )
    if arm == "H":
        assert operational["execution_complete"] is False
        assert operational["physical_steps"] == 0
        assert operational["attempted_decisions"] == 1
        assert operational["exposure_rates"]["guard_correction"]["rate"] is None
        assert operational["exposure_rates"]["invalid_command"]["exposure"] == 1
        assert operational["cycles"][0]["resources"] is None
        assert operational["cycles"][0]["net_operating_value_eur"] is None
        assert all(
            value is None
            for value in operational["cycles"][0]["components_eur"].values()
        )
    else:
        assert operational["execution_complete"] is True
        assert sum(
            row["net_operating_value_eur"] for row in operational["cycles"]
        ) == pytest.approx(original["economic_metrics"]["net_operating_value_eur"])


def test_detached_all_slot_summary_reopens_physics_and_keeps_missingness(
    scorecard: dict[str, Any]
) -> None:
    admitted = reports.verify_synthetic_operational_file(scorecard["path"])
    assert admitted["source_evidence_verified"] is True
    assert admitted["registered_registry_verified"] is False
    assert admitted["registered_checkpoint_verified"] is False
    assert admitted["result_claim"] is None
    result = admitted["result"]
    assert result["arm_evidence_available"]["successes"] == 1
    assert result["missing_episode_seeds"] == SEEDS[1:]
    assert (
        result["resources"]["battery"]["per_cycle"][0]["feasible_over_observed"][
            "total"
        ]
        == 1
    )
    assert (
        result["resources"]["battery"]["per_cycle"][0][
            "verified_feasible_over_registered"
        ]["total"]
        == 32
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "denominator",
        "energy",
        "deficit",
        "bootstrap",
        "missingness",
        "economic",
        "failure",
        "contract",
        "claim",
        "reference",
        "schema",
    ],
)
def test_rehashed_scorecard_substitutions_rejected(
    scorecard: dict[str, Any], tmp_path: Path, mutation: str
) -> None:
    report = deepcopy(scorecard["report"])
    result = report["result"]
    if mutation == "denominator":
        result["execution_complete"]["total"] = 1
    elif mutation == "energy":
        result["episode_records"][0]["physical_metrics"]["solar_kwh"] += 1
    elif mutation == "deficit":
        result["resources"]["battery"]["observed_deficits"]["maximum"] = 123
    elif mutation == "bootstrap":
        result["strata"]["outage"]["no_outage"]["metrics"]["net_operating_value_eur"][
            "bootstrap_defined_replicates"
        ] = 0
    elif mutation == "missingness":
        result["missing_episode_seeds"] = []
    elif mutation == "economic":
        result["episode_records"][0]["economic_metrics"]["net_operating_value_eur"] += 1
    elif mutation == "failure":
        result["episode_records"][0]["failure"] = {"exception_type": "hidden"}
    elif mutation == "contract":
        report["operational_contract_sha256"] = "a" * 64
    elif mutation == "claim":
        report["result_claim"] = "registered"
    elif mutation == "reference":
        report["arms"][0]["envelope_sha256"] = "b" * 64
    else:
        report["schema"] = "registered-operational-evidence-v1"
    report["result"] = authenticate(result)
    target = tmp_path / reports.FILENAME
    io.publish_json(target, authenticate(report))
    with pytest.raises(ValueError):
        reports.verify_synthetic_operational_file(target)


def test_rehashed_physical_source_tamper_rejected(
    evidence: dict[str, Any], tmp_path: Path
) -> None:
    envelope = deepcopy(evidence["envelopes"]["PPO"])
    envelope["run"]["physical_metrics"]["solar_kwh"] += 1
    path = tmp_path / arms.FILENAME
    io.publish_json(path, _rehash(envelope))
    with pytest.raises(ValueError):
        reports.synthetic_operational_evidence(
            SEEDS, [path] + [None] * 31, cycle_count=1
        )


@pytest.mark.parametrize(
    "seed", [1_000_000, 2_000_000, 3_000_000, 8_999_999, 10_000_000, True]
)
def test_scope_rejected_before_any_physical_manifest_open(
    seed: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("protected seed reached physical admission")

    monkeypatch.setattr(arms, "verify_synthetic_arm_file", forbidden)
    seeds = SEEDS.copy()
    seeds[0] = seed
    with pytest.raises(ValueError):
        reports.synthetic_operational_evidence(seeds, [None] * 32, cycle_count=1)


def test_same_bytes_source_replacement_detected(
    evidence: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / arms.FILENAME
    raw = canonical_bytes(evidence["envelopes"]["PPO"])
    path.write_bytes(raw)
    original = reports.policy_scorecard

    def replacing(*args: object, **kwargs: object) -> dict[str, Any]:
        replacement = path.with_name("replacement.json")
        replacement.write_bytes(raw)
        replacement.replace(path)
        return original(*args, **kwargs)

    monkeypatch.setattr(reports, "policy_scorecard", replacing)
    with pytest.raises(ValueError, match="graph changed"):
        reports.synthetic_operational_evidence(
            SEEDS, [path] + [None] * 31, cycle_count=1
        )


def test_duplicate_reference_or_wrong_slot_rejected(evidence: dict[str, Any]) -> None:
    path = evidence["paths"]["PPO"]
    with pytest.raises(ValueError, match="unique"):
        reports.synthetic_operational_evidence(
            SEEDS, [path, path] + [None] * 30, cycle_count=1
        )
    with pytest.raises(ValueError, match="seed differs"):
        reports.synthetic_operational_evidence(
            SEEDS, [None, path] + [None] * 30, cycle_count=1
        )


def test_different_actor_cannot_enter_same_policy_scorecard(
    evidence: dict[str, Any], tmp_path: Path
) -> None:
    path = tmp_path / arms.FILENAME
    io.publish_json(path, evidence["failed"]["E-beta"])
    with pytest.raises(ValueError, match="mixes actor"):
        reports.synthetic_operational_evidence(
            SEEDS, [evidence["paths"]["PPO"], path] + [None] * 30, cycle_count=1
        )

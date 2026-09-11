"""Reset-free projection of complete synthetic analysis, including negative pairs."""

from copy import deepcopy
from types import SimpleNamespace
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from plant_backend import ppo_evaluation as public, ppo_evidence, publish_ppo
from plant_backend.rl.v2.study import stage_verifier
from test_rl_v2_study_campaign_analysis import (
    inputs,
    development,
    heldout,
    forbid_execution,
)  # noqa: F401


def source_report():
    return {"source": {"authorization_commit": "a" * 40}, "sha256": "b" * 64}


def stage_report(analysis, stage="development"):
    return {
        "analysis": analysis,
        "stage": stage,
        "campaign_status": "completed",
        "result_claim": None,
        "sha256": "c" * 64,
    }


def test_all_18_actors_and_computed_gates_project_without_new_execution(
    development, heldout
):
    for stage, analysis in (("development", development), ("heldout", heldout)):
        result = public._projection(stage_report(analysis, stage), source_report())
        assert len(result["actors"]) == 18
        assert sum(sum(row["states"].values()) for row in result["actors"]) == 576
        assert all(row["eligible_pairs"] == 32 for row in result["actors"][3:])
        assert sum(row["frozen_champion"] for row in result["actors"]) == 3
        assert result["actors"][0]["hard_safety_passed"] == 0
        assert result["actors"][0]["execution_complete"] == 32
        assert "episode_seeds" not in json.dumps(result)
        assert result["gate_passed"] == all(result["gate_checks"].values())


def test_ineligible_pairs_have_no_invented_statistics(development):
    analysis = deepcopy(development)
    row = analysis["actors"]["pi-lagrangian-seed-101"]
    row["primary_summary"] = None
    row["primary_pairs"][0]["value_comparison_eligible"] = False
    row["cell_failures"][0] = "evidence_rejected"
    row["cell_states"][0] = "evidence_rejected"
    result = public._projection(stage_report(analysis), source_report())
    slot = result["actors"][3]
    assert slot["eligible_pairs"] == 31
    assert slot["iqm_delta_eur"] is slot["cvar20_delta_eur"] is None
    assert slot["failures"] == {"evidence_rejected": 1}


def test_stage_export_must_reconstruct_and_recheck_before_publication(
    development, tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(ppo_evidence, "PUBLIC_ROOT", tmp_path / "public")

    def verify(root, stage):
        calls.append((root, stage))
        return SimpleNamespace(
            report=stage_report(development),
            context=SimpleNamespace(training=SimpleNamespace(report=source_report())),
            recheck=lambda: calls.append("recheck"),
        )

    monkeypatch.setattr(stage_verifier, "verify_stage", verify)
    result = public.export_stage(tmp_path / "training", "development")
    assert calls == [(tmp_path / "training", "development"), "recheck"]
    assert (
        public.validate(
            json.loads((ppo_evidence.PUBLIC_ROOT / "development.json").read_text())
        )
        == result
    )
    with pytest.raises(FileExistsError):
        public.export_stage(tmp_path / "training", "development")

    def reject(*args):
        raise ValueError("detached reconstruction rejected")

    monkeypatch.setattr(stage_verifier, "verify_stage", reject)
    with pytest.raises(ValueError, match="rejected"):
        public.export_stage(tmp_path / "training", "heldout")
    assert not (ppo_evidence.PUBLIC_ROOT / "heldout.json").exists()


def test_http_requires_matching_training_and_development_before_heldout(
    development, heldout, tmp_path, monkeypatch
):
    monkeypatch.setattr(ppo_evidence, "PUBLIC_ROOT", tmp_path)
    app = FastAPI()
    app.include_router(public.router)
    with TestClient(app) as client:
        data = public._projection(stage_report(development), source_report())
        (tmp_path / "development.json").write_text(json.dumps(data))
        assert client.get("/api/v2/ppo/evaluation").status_code == 503
        monkeypatch.setattr(
            ppo_evidence,
            "read_export",
            lambda: {"source_commit": "a" * 40, "training_report_sha256": "b" * 64},
        )
        assert client.get("/api/v2/ppo/evaluation").json()["development"] == data
        held = public._projection(stage_report(heldout, "heldout"), source_report())
        (tmp_path / "heldout.json").write_text(json.dumps(held))
        assert client.get("/api/v2/ppo/evaluation").status_code == 200
        (tmp_path / "development.json").unlink()
        assert client.get("/api/v2/ppo/evaluation").status_code == 503


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["actors"].pop(),
        lambda p: p["actors"][0]["states"].update(verified=31),
        lambda p: p["actors"][3].update(eligible_pairs=31),
        lambda p: p.update(gate_passed=False),
        lambda p: p.update(result_claim="winner"),
    ],
)
def test_rehashed_incomplete_or_unearned_results_rejected(development, mutate):
    data = public._projection(stage_report(development), source_report())
    mutate(data)
    data["sha256"] = ppo_evidence.digest(
        {k: v for k, v in data.items() if k != "sha256"}
    )
    with pytest.raises(ValueError):
        public.validate(data)

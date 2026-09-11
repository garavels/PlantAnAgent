"""Public exports retain closed-custody failures; no real training or reset."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from plant_backend import ppo_evidence as public, ppo_evaluation, publish_ppo
from plant_backend.rl.v2.study import training_evidence
from test_rl_v2_study_training_evidence import campaign, no_execution  # noqa: F401


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(public, "PUBLIC_ROOT", tmp_path)
    app = FastAPI()
    app.include_router(public.router)
    app.include_router(ppo_evaluation.router)
    with TestClient(app) as connection:
        yield connection


def test_pending_and_invalid_requests_do_not_admit_or_reset(client, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("HTTP attempted training admission")

    monkeypatch.setattr(training_evidence, "admit_training", forbidden)
    assert client.get("/api/v2/ppo/training").json() == {
        "status": "pending",
        "evidence": None,
    }
    assert client.get("/api/v2/ppo/evaluation").json() == {
        "development": None,
        "heldout": None,
    }
    for route in ("training", "evaluation"):
        assert client.get(f"/api/v2/ppo/{route}?path=/private").status_code == 422
        assert client.post(f"/api/v2/ppo/{route}").status_code == 405


@pytest.mark.parametrize("defect", [None, "checkpoint", "process", "storage"])
def test_export_reopens_custody_retains_every_slot_and_is_exclusive(
    campaign, client, tmp_path, defect
):
    fixture = campaign(defect=defect)
    exported = publish_ppo.export_training(fixture.path)
    assert len(exported["slots"]) == 15
    assert client.get("/api/v2/ppo/training").json()["evidence"] == exported
    assert len(fixture.parsed) == sum(
        s["checkpoint_admitted"] for s in exported["slots"]
    )
    for slot in exported["slots"]:
        if slot["checkpoint_admitted"]:
            assert len(slot["rollouts"]) == 128
            assert slot["rollouts"][-1]["transitions"] == 1_048_576
        else:
            assert slot["failure"] and slot["rollouts"] == []
    assert exported["evaluation_performed"] is False
    text = json.dumps(exported)
    for private in (
        "allocator",
        "episode_seed",
        "audit_manifest",
        "rollout_ledgers",
        str(fixture.ctx.root),
    ):
        assert private not in text
    original = (tmp_path / "training.json").read_bytes()
    with pytest.raises(FileExistsError):
        publish_ppo.export_training(fixture.path)
    assert (tmp_path / "training.json").read_bytes() == original


def test_export_rejects_unclosed_synthetic_custody_without_publication(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(public, "PUBLIC_ROOT", tmp_path / "not-created")
    report = tmp_path / "report.json"
    report.write_text('{"schema":"synthetic-training"}')
    with pytest.raises(ValueError):
        publish_ppo.export_training(report)
    assert not public.PUBLIC_ROOT.exists()


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["slots"].pop(),
        lambda p: p["slots"][0].update(optimizer_seed=999),
        lambda p: p["slots"][0]["rollouts"].pop(),
        lambda p: p["slots"][0]["rollouts"][0].update(objective_sum=float("nan")),
        lambda p: p["slots"][0]["rollouts"][0].update(transitions=123),
        lambda p: p.update(private_manifest={}),
    ],
)
def test_even_rehashed_incomplete_or_malformed_exports_are_rejected(campaign, change):
    fixture = campaign()
    data = publish_ppo._projection(training_evidence.admit_training(fixture.path))
    change(data)
    with pytest.raises(ValueError):
        data["sha256"] = public.digest({k: v for k, v in data.items() if k != "sha256"})
        public.validate(data)


def test_tampered_oversized_and_linked_exports_fail_closed(
    campaign, client, tmp_path, monkeypatch
):
    fixture = campaign()
    data = publish_ppo.export_training(fixture.path)
    path = tmp_path / "training.json"
    changed = deepcopy(data)
    changed["slots"][0]["rollouts"][0]["objective_sum"] = 1234.0
    path.write_text(json.dumps(changed))
    assert client.get("/api/v2/ppo/training").status_code == 503
    path.write_text(json.dumps(data))
    raw = path.read_text()
    path.write_text(
        raw.replace(
            '"schema_version":', '"schema_version": "duplicate", "schema_version":', 1
        )
    )
    assert client.get("/api/v2/ppo/training").status_code == 503
    path.write_text(raw)
    monkeypatch.setattr(public, "MAX_BYTES", 10)
    assert client.get("/api/v2/ppo/training").status_code == 503
    path.unlink()
    path.symlink_to(tmp_path / "missing")
    assert client.get("/api/v2/ppo/training").status_code == 503


def test_rehearsal_requires_explicit_flag_before_opening_report(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unauthorized rehearsal opened a report")

    monkeypatch.setattr(publish_ppo, "record_rehearsal", forbidden)
    with pytest.raises(SystemExit) as result:
        publish_ppo.main(["rehearsal", "/nonexistent/report.json"])
    assert result.value.code == 2


def test_failed_fixed_policy_has_no_substitute(campaign):
    fixture = campaign(defect="process")
    admission = training_evidence.admit_training(fixture.path)
    admission = replace(
        admission,
        slots=(
            replace(
                admission.slots[0],
                checkpoint_admitted=False,
                rollouts=None,
                failure="child_nonzero_exit",
            ),
            *admission.slots[1:],
        ),
    )
    with pytest.raises(ValueError, match="no replacement"):
        publish_ppo._presentation_policy(admission)

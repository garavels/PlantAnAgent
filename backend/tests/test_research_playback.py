"""Separate research admission, complete playback evidence and catalog priority."""

from copy import deepcopy
import json

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from plant_backend import demo_recordings, research_playback


def _candidate():
    """Synthetic projection only; test fixture admission is patched explicitly."""
    initial = {
        "solar_available_kw": 0.0,
        "solar_potential_kw": 0.0,
        "battery_energy_kwh": 5000.0,
        "battery_soc": 0.625,
        "h2_level_kg": 200.0,
        "co2_level_kg": 1000.0,
        "ch4_level_kg": 250.0,
    }
    stocks = {"battery": 5000.0, "h2": 200.0, "co2": 1000.0, "ch4": 250.0}
    cycles = [
        {
            "cycle_index": index,
            "feasible": True,
            "targets": dict(stocks),
            "actual": dict(stocks),
            "deficits": dict.fromkeys(stocks, 0.0),
            "tolerances": dict.fromkeys(stocks, 1e-6),
        }
        for index in range(1, 8)
    ]
    frames, history = [], []
    for index in range(673):
        amount = index // 24 * 150
        net_value = amount * 4
        frames.append(
            {
                "state": dict(initial),
                "clock": {
                    "decision_index": index,
                    "decision_count": 672,
                    "cycle_count": 7,
                    "elapsed_hours": index / 4,
                },
                "delivery": {
                    "delivered_kg": amount,
                    "requested_kg": amount,
                    "unmet_kg": 0.0,
                    "commitments": [
                        {
                            "due_hours": ordinal * 6,
                            "requested_kg": 150.0,
                            "delivered_kg": 150.0 if ordinal <= index // 24 else None,
                            "unmet_kg": 0.0 if ordinal <= index // 24 else None,
                            "status": "met" if ordinal <= index // 24 else "pending",
                        }
                        for ordinal in range(1, 29)
                    ],
                },
                "economics": {
                    "net_value_eur": net_value,
                    "entries": [
                        {"id": "delivered_service_value", "value_eur": net_value},
                        {"id": "solar_opportunity_cost", "value_eur": 0.0},
                    ],
                },
                "forecasts": {"price_current_eur_mwh": 0.0, "prices": []},
                "cycle_settlements": deepcopy(cycles[: index // 96]),
            }
        )
        history.append(
            {
                "elapsed_hours": index / 4,
                "solar_kw": 0.0,
                "clear_sky_kw": 0.0,
                "h2_kg": 200.0,
                "co2_kg": 1000.0,
                "battery_soc": 0.625,
                "ch4_kg": 250.0,
                "delivered_kg": amount,
                "net_value_eur": net_value,
            }
        )
    snapshot = {
        "schema": "plant-v2-demo-v1",
        "run_id": "ffffffff-ffff-ffff-ffff-ffffffffffff",
        "mode": "recorded_research",
        "status": "complete",
        "error": None,
        "controller": {"id": "PPO"},
        "environment": {
            "id": research_playback.ENVIRONMENT_ID,
            "fixture_seed": 9_930_000,
            "horizon_hours": 168,
            "decision_minutes": 15,
            "physics_minutes": 1,
        },
        "frame": frames[-1],
        "history": history,
    }
    provenance = {
        "source_recording_sha256": research_playback.SOURCE_SHA256,
        "benchmark_contract_sha256": research_playback.BENCHMARK_SHA256,
        "checkpoint_sha256": research_playback.CHECKPOINT_SHA256,
        "scenario_seed": 9_930_000,
        "optimizer_seed": 101,
        "mission": {
            "delivery_rate_kg_h": 25,
            "horizon_h": 168,
            "deliveries_met": 28,
            "deliveries_total": 28,
            "daily_returns_met": 7,
            "daily_returns_total": 7,
            "mission_passed": True,
            "hard_safety_passed": True,
        },
        "evaluation": {"split": "held_out", "passed": 14, "total": 16},
        "electricity": {
            "mode": "dedicated_pv_no_export",
            "marginal_eur_per_mwh": 0,
            "fixed_costs_included": False,
        },
    }
    return demo_recordings.seal_research_recording(snapshot, frames, provenance)


@pytest.fixture
def admitted_candidate(monkeypatch):
    record = _candidate()
    monkeypatch.setattr(
        research_playback, "APPROVED_RECORDING_ID", record["snapshot"]["run_id"]
    )
    monkeypatch.setattr(
        research_playback, "APPROVED_PRESENTATION_SHA256", record["sha256"]
    )
    return record


def _reseal(record):
    record["sha256"] = demo_recordings.digest(
        {key: value for key, value in record.items() if key != "sha256"}
    )


def test_research_is_separate_from_fixture_admission(admitted_candidate):
    assert demo_recordings.validate(admitted_candidate) == admitted_candidate
    summary = demo_recordings.summarize(admitted_candidate, "saved")
    assert summary["featured"] is True
    assert summary["provenance"]["evaluation"]["passed"] == 14
    assert summary["failed_cycles"] == 0
    public_envelope = demo_recordings.seal_recording(
        admitted_candidate["snapshot"], admitted_candidate["frames"]
    )
    with pytest.raises(ValueError, match="only disposable"):
        demo_recordings.validate(public_envelope)


@pytest.mark.parametrize(
    "tamper,reason",
    [
        ("source", "provenance"),
        ("checkpoint", "provenance"),
        ("evaluation", "evaluation"),
        ("boundary", "boundaries"),
        ("history", "history"),
        ("delivery", "delivery boundary"),
        ("daily", "daily stock"),
        ("electricity", "economic ledger"),
        ("new_payload", "unapproved research presentation"),
    ],
)
def test_resealed_tampering_is_rejected(admitted_candidate, tamper, reason):
    record = admitted_candidate
    if tamper in ("source", "checkpoint"):
        key = "source_recording_sha256" if tamper == "source" else "checkpoint_sha256"
        record["provenance"][key] = "0" * 64
    elif tamper == "evaluation":
        record["provenance"]["evaluation"]["passed"] = 16
    elif tamper == "boundary":
        record["frames"].pop(1)
    elif tamper == "history":
        record["snapshot"]["history"][1]["solar_kw"] = 100.0
    elif tamper == "delivery":
        record["frames"][24]["delivery"]["commitments"][0]["delivered_kg"] = 149.0
    elif tamper == "daily":
        record["frames"][-1]["cycle_settlements"][0]["actual"]["ch4"] = 249.0
        record["snapshot"]["frame"] = deepcopy(record["frames"][-1])
    elif tamper == "electricity":
        record["frames"][1]["economics"]["entries"][1]["value_eur"] = -10.0
    elif tamper == "new_payload":
        record["frames"][1]["unreviewed_claim"] = "fabricated"
    _reseal(record)
    with pytest.raises(ValueError, match=reason):
        demo_recordings.validate(record)


def test_unresealed_tampering_fails_checksum(admitted_candidate):
    admitted_candidate["provenance"]["scenario_seed"] += 1
    with pytest.raises(ValueError, match="checksum"):
        demo_recordings.validate(admitted_candidate)


def test_featured_research_is_visible_before_eight_legacy_files(
    admitted_candidate, monkeypatch, tmp_path
):
    legacy = tmp_path / "legacy"
    research = tmp_path / "research"
    legacy.mkdir()
    research.mkdir()
    monkeypatch.setattr(demo_recordings, "RECORDING_ROOT", legacy)
    monkeypatch.setattr(demo_recordings, "RESEARCH_RECORDING_ROOT", research)
    identifier = admitted_candidate["snapshot"]["run_id"]
    (research / f"{identifier}.json").write_text(json.dumps(admitted_candidate))
    # Eight invalid legacy files still consume their historical bounded slots.
    for index in range(8):
        (legacy / f"00000000-0000-0000-0000-{index:012d}.json").write_text("{}")
    rows = demo_recordings.catalog()
    assert len(rows) == 9
    assert rows[0]["id"] == identifier
    assert rows[0]["available"] is True
    assert rows[0]["featured"] is True
    assert all(row["available"] is False for row in rows[1:])
    assert demo_recordings.read_recording(identifier) == admitted_candidate


def test_checked_in_approved_playback_is_admitted():
    """The shipped projection must agree with its reviewed presentation pin."""
    record = demo_recordings.read_recording(research_playback.APPROVED_RECORDING_ID)
    assert record["snapshot"]["frame"]["delivery"]["delivered_kg"] == 4200
    assert len(record["frames"]) == 673
    assert record["provenance"]["mission"]["hard_safety_passed"] is True


def test_existing_http_routes_serve_research_without_resetting_a_plant(monkeypatch):
    from plant_backend import v2_demo

    def forbidden(*args, **kwargs):
        pytest.fail("read-only research playback attempted a plant reset")

    monkeypatch.setattr(v2_demo.PlantCyclicMarketEnvV2Beta, "reset", forbidden)
    app = FastAPI()
    app.include_router(v2_demo.router)
    with TestClient(app) as client:
        catalog = client.get("/api/v2/demo/recordings")
        assert catalog.status_code == 200
        assert catalog.json()[0]["featured"] is True
        response = client.get(
            f"/api/v2/demo/recordings/{research_playback.APPROVED_RECORDING_ID}"
        )
    assert response.status_code == 200
    assert response.json()["sha256"] == research_playback.APPROVED_PRESENTATION_SHA256


def test_research_path_cannot_be_downgraded_to_a_fixture(monkeypatch, tmp_path):
    identifier = research_playback.APPROVED_RECORDING_ID
    frame = {"clock": {"decision_index": 0, "elapsed_hours": 0}}
    snapshot = {
        "run_id": identifier,
        "environment": {"fixture_seed": 9_000_034},
        "mode": "interactive_fixture",
        "frame": frame,
        "history": [{}],
    }
    public = demo_recordings.seal_recording(snapshot, [frame])
    # The old envelope remains valid under its original fixture gate.
    assert demo_recordings.validate(public) == public
    monkeypatch.setattr(demo_recordings, "RESEARCH_RECORDING_ROOT", tmp_path)
    (tmp_path / f"{identifier}.json").write_text(json.dumps(public))
    with pytest.raises(HTTPException) as failure:
        demo_recordings.read_recording(identifier)
    assert failure.value.status_code == 404

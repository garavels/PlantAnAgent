"""A public research view cannot open custody or turn incomplete results into approval."""

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from plant_backend import adaptive_learning as public


def artifact(name: str) -> dict[str, Any]:
    return {"id": name, "sha256": "1" * 64, "bytes": 100}


def signed(payload: dict[str, Any]) -> dict[str, Any]:
    payload["sha256"] = public.digest(
        {k: v for k, v in payload.items() if k != "sha256"}
    )
    return payload


@pytest.fixture
def payload() -> dict[str, Any]:
    actors = []
    for index, identifier in enumerate(public.ACTOR_IDS):
        stopped = identifier in ("M24", "E-beta")
        cases = []
        for seed in public.SEEDS:
            deliveries = [
                {
                    "ordinal": ordinal,
                    "requested_kg": 240.0,
                    "delivered_kg": 230.0 - index,
                }
                for ordinal in range(1, 29 if not stopped else 3)
            ]
            cases.append(
                {
                    "seed": seed,
                    "status": "stopped" if stopped else "completed",
                    "decisions": 50 if stopped else 672,
                    "failures": ["planner stopped"] if stopped else [],
                    "net_operating_value_eur": None if stopped else -6000.0 + index,
                    "delivered_kg": (
                        None if stopped else sum(r["delivered_kg"] for r in deliveries)
                    ),
                    "battery_throughput_kwh": None if stopped else 9000.0 - index,
                    "hard_safety_passed": None if stopped else True,
                    "prefix_hard_safety_passed": True,
                    "all_daily_returns_passed": None if stopped else False,
                    "mpc_modes": (
                        None
                        if identifier in ("H", "M24", "E-beta")
                        else {
                            "scheduled": 0,
                            "recovery": 665,
                            "fallback": 7,
                        }
                    ),
                    "deliveries": deliveries,
                    "daily_resources": [
                        {
                            "cycle": cycle,
                            "deficits": {
                                "battery": float(index + 1),
                                "h2": 1.0,
                                "co2": 0.0,
                                "ch4": 0.0,
                            },
                        }
                        for cycle in range(1, 8 if not stopped else 1)
                    ],
                }
            )
        actors.append(
            {
                "id": identifier,
                "label": identifier,
                "kind": (
                    "ppo"
                    if identifier in public.POLICY_IDS
                    else (
                        "initial"
                        if identifier == "initial-mpc-residual"
                        else "heuristic" if identifier == "H" else "mpc"
                    )
                ),
                "evidence_origin": (
                    "fresh_batch009"
                    if identifier in public.POLICY_IDS[2:]
                    else "reused_batch008"
                ),
                "cases": cases,
            }
        )
    actor_index = {row["id"]: row for row in actors}
    policies = []
    for identifier in public.POLICY_IDS:
        older = identifier in public.POLICY_IDS[:2]
        rollouts = 8 if older else 2
        policies.append(
            {
                "id": identifier,
                "label": identifier,
                "campaign_id": "007" if older else "009",
                "algorithm": "pi" if identifier == "B-mpc-residual-pi" else "scalar",
                "optimizer_seed": 101,
                "training_status": "completed",
                "evaluation_status": "evaluated",
                "operational_status": "experimental",
                "transitions": rollouts * 8192,
                "minibatch_updates": rollouts * 320,
                "reward_gamma": 1.0 if identifier == "B-scalar-gamma1000" else 0.995,
                "gae_lambda": 0.95,
                "reward_scale": 0.001,
                "checkpoint": artifact(identifier + "-checkpoint"),
                "training_source": artifact(identifier + "-training"),
                "rollouts": [
                    {
                        "index": index,
                        "transitions": index * 8192,
                        "reward_sum": -50.0,
                        "completed_episodes": 4,
                        "complete_cycles": 28,
                        "training_mean_reward": None,
                        "explained_variance": None,
                        "value_loss": 0.5,
                        "approx_kl": 0.01,
                        "clip_fraction": 0.2,
                        "mean_episode_cost": [1.0, 0.0, 0.0, 0.0],
                        "multipliers": (
                            [0.1, 0.0, 0.0, 0.0]
                            if identifier == "B-mpc-residual-pi"
                            else None
                        ),
                    }
                    for index in range(1, rollouts + 1)
                ],
                "limitations": ["Development evaluation only"],
            }
        )
    comparisons = []
    for identifier in public.POLICY_IDS:
        for other in (
            actor
            for actor in public.ACTOR_IDS
            if (identifier, actor) in public.COMPARISON_IDS
        ):
            cases = []
            for a, b in zip(
                actor_index[identifier]["cases"],
                actor_index[other]["cases"],
                strict=True,
            ):
                available = a["status"] == b["status"] == "completed"
                cases.append(
                    {
                        "seed": a["seed"],
                        "available": available,
                        "candidate_failures": a["failures"],
                        "comparator_failures": b["failures"],
                        **{
                            f"delta_{key}": a[key] - b[key] if available else None
                            for key in (
                                "net_operating_value_eur",
                                "delivered_kg",
                                "battery_throughput_kwh",
                            )
                        },
                        "delivery_regressions": [
                            {
                                "ordinal": x["ordinal"],
                                "difference_kg": x["delivered_kg"] - y["delivered_kg"],
                            }
                            for x, y in zip(a["deliveries"], b["deliveries"])
                            if available and x["delivered_kg"] < y["delivered_kg"]
                        ],
                        "resource_regressions": [
                            {
                                "cycle": x["cycle"],
                                "resource": r,
                                "difference": x["deficits"][r] - y["deficits"][r],
                            }
                            for x, y in zip(a["daily_resources"], b["daily_resources"])
                            for r in public.RESOURCES
                            if available and x["deficits"][r] > y["deficits"][r]
                        ],
                    }
                )
            comparisons.append(
                {
                    "candidate": identifier,
                    "comparator": other,
                    "evidence_batch": (
                        "008" if identifier in public.POLICY_IDS[:2] else "009"
                    ),
                    "cases": cases,
                }
            )
    return signed(
        {
            "schema_version": public.SCHEMA,
            "source": {"git_head": "a" * 40, "git_tree": "b" * 40},
            "provenance": {
                "exported_from_closed_evidence": True,
                "independent_verification_passed": True,
                "development_only": True,
                "artifacts": [artifact("analysis")],
            },
            "task": {
                "hours": 168,
                "decisions": 672,
                "deliveries": 28,
                "requested_methane_kg": 6720,
                "development_seeds": public.SEEDS,
            },
            "policies": policies,
            "actors": actors,
            "comparisons": comparisons,
            "history": [
                {
                    "id": f"{i:03}",
                    "label": f"Batch {i}",
                    "status": "aborted" if i == 7 else "completed",
                    "summary": "Retained research campaign",
                    "policy_ids": [],
                    "source": artifact(f"history-{i}"),
                    "limitations": [],
                }
                for i in range(1, 10)
            ],
            "registered_study": {
                "status": "pending",
                "families": 3,
                "optimizer_seeds": 5,
                "policies": 15,
                "protected_seeds_opened": False,
            },
            "limitations": ["No held-out evaluation or deployment"],
        }
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(public, "PUBLIC_PATH", tmp_path / "adaptive_learning.json")
    app = FastAPI()
    app.include_router(public.router)
    with TestClient(app) as connection:
        yield connection


def test_valid_export_round_trip_is_read_only(payload, client, monkeypatch):
    public.PUBLIC_PATH.write_text(json.dumps(payload))
    before = public.PUBLIC_PATH.read_bytes()
    original_open = public.os.open
    opened = []

    def read_only_open(path, flags, *args, **kwargs):
        assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        opened.append(str(path))
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(public.os, "open", read_only_open)
    response = client.get("/api/v2/ppo/research")
    assert response.status_code == 200
    assert response.json() == {"status": "available", "evidence": payload}
    assert opened == [str(public.PUBLIC_PATH.parent), public.PUBLIC_PATH.name]
    assert public.PUBLIC_PATH.read_bytes() == before


def test_pending_no_parameters_no_body_no_mutating_methods(client):
    assert client.get("/api/v2/ppo/research").json() == {
        "status": "pending",
        "evidence": None,
    }
    assert client.get("/api/v2/ppo/research?path=/private").status_code == 422
    assert (
        client.request("GET", "/api/v2/ppo/research", content=b"{}").status_code == 422
    )
    for method in ("POST", "PUT", "DELETE", "PATCH"):
        assert client.request(method, "/api/v2/ppo/research").status_code == 405


def test_missing_public_directory_is_pending(client, monkeypatch, tmp_path):
    monkeypatch.setattr(public, "PUBLIC_PATH", tmp_path / "absent" / "data.json")
    assert client.get("/api/v2/ppo/research").json()["status"] == "pending"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["policies"][0].update(operational_status="approved"),
        lambda p: p["policies"][0].update(training_status="failed"),
        lambda p: p["policies"][0].update(transitions=16384),
        lambda p: p["policies"][0]["rollouts"].pop(),
        lambda p: p["policies"][0]["rollouts"][0].update(multipliers=[1.0] * 4),
        lambda p: p["policies"][0]["rollouts"][0].update(mean_episode_cost=[1.0]),
        lambda p: p["policies"][0]["rollouts"][0].update(index=2),
        lambda p: p["policies"][0].update(optimizer_seed=102),
        lambda p: p["policies"][0].update(campaign_id="001"),
        lambda p: p["history"][6].update(status="completed"),
        lambda p: p["history"][3].update(id="pilot"),
        lambda p: p["actors"].pop(),
        lambda p: p["actors"][0]["cases"].pop(),
        lambda p: p["actors"][0]["cases"][0].update(seed=9700301),
        lambda p: p["actors"][0]["cases"][0].update(all_daily_returns_passed=True),
        lambda p: p["actors"][0]["cases"][0]["deliveries"].pop(),
        lambda p: p["actors"][0]["cases"][0].update(delivered_kg=1.0),
        lambda p: p["actors"][1]["cases"][0]["mpc_modes"].update(fallback=0),
        lambda p: p["actors"][2]["cases"][0].update(net_operating_value_eur=100.0),
        lambda p: p["actors"][2]["cases"][0].update(failures=[]),
        lambda p: p["comparisons"][0]["cases"][0]["delivery_regressions"].pop(),
        lambda p: p["comparisons"][0]["cases"][0]["resource_regressions"].pop(),
        lambda p: p["comparisons"][0]["cases"][0].update(
            delta_net_operating_value_eur=999.0
        ),
        lambda p: p["comparisons"][2]["cases"][0].update(available=True),
        lambda p: p["comparisons"][2]["cases"][0].update(comparator_failures=[]),
        lambda p: p["comparisons"][2]["cases"][0].update(
            delta_net_operating_value_eur=1.0
        ),
        lambda p: p["comparisons"][0].update(evidence_batch="009"),
        lambda p: p["comparisons"].pop(),
        lambda p: p["history"][0].update(policy_ids=["unknown-policy"]),
        lambda p: p["history"][0].update(id=p["history"][1]["id"]),
        lambda p: p["registered_study"].update(protected_seeds_opened=True),
        lambda p: p["provenance"].update(development_only=False),
        lambda p: p["provenance"]["artifacts"][0].update(id="../private/checkpoint.pt"),
        lambda p: p.update(arbitrary_approval=True),
    ],
)
def test_rehashed_semantic_corruption_rejected(payload, mutate):
    mutate(payload)
    with pytest.raises(ValueError):
        public.validate(signed(payload))


def test_digest_tampering_nonfinite_and_duplicate_keys_fail_closed(payload, client):
    changed = deepcopy(payload)
    changed["policies"][0]["label"] = "Tampered"
    public.PUBLIC_PATH.write_text(json.dumps(changed))
    assert client.get("/api/v2/ppo/research").status_code == 503
    for value in (float("nan"), float("inf"), float("-inf")):
        changed = deepcopy(payload)
        changed["policies"][0]["rollouts"][0]["reward_sum"] = value
        public.PUBLIC_PATH.write_text(json.dumps(changed))
        assert client.get("/api/v2/ppo/research").status_code == 503
    raw = json.dumps(payload)
    public.PUBLIC_PATH.write_text(
        raw.replace(
            '"schema_version":', '"schema_version": "duplicate", "schema_version":', 1
        )
    )
    assert client.get("/api/v2/ppo/research").status_code == 503


@pytest.mark.parametrize(
    "kind",
    [
        "directory",
        "symlink",
        "dangling",
        "fifo",
        "empty",
        "oversized",
        "parent-symlink",
        "nonobject",
    ],
)
def test_unsafe_or_unbounded_files_fail_closed(client, tmp_path, monkeypatch, kind):
    path = public.PUBLIC_PATH
    if kind == "directory":
        path.mkdir()
    elif kind in ("symlink", "dangling"):
        target = tmp_path / "target"
        if kind == "symlink":
            target.write_text("{}")
        path.symlink_to(target)
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "parent-symlink":
        folder = tmp_path / "real"
        folder.mkdir()
        (tmp_path / "alias").symlink_to(folder, target_is_directory=True)
        monkeypatch.setattr(public, "PUBLIC_PATH", tmp_path / "alias" / "data.json")
    elif kind == "oversized":
        monkeypatch.setattr(public, "MAX_BYTES", 10)
        path.write_text("x" * 11)
    elif kind == "nonobject":
        path.write_text("[]")
    else:
        path.write_text("")
    assert client.get("/api/v2/ppo/research").status_code == 503


def test_replacement_during_read_is_rejected(payload, client, monkeypatch):
    public.PUBLIC_PATH.write_text(json.dumps(payload))
    original = public.os.read
    replaced = False

    def swap(handle, size):
        nonlocal replaced
        result = original(handle, size)
        if not replaced:
            replacement = public.PUBLIC_PATH.with_suffix(".replacement")
            replacement.write_text(json.dumps(payload))
            replacement.replace(public.PUBLIC_PATH)
            replaced = True
        return result

    monkeypatch.setattr(public.os, "read", swap)
    assert client.get("/api/v2/ppo/research").status_code == 503


def test_router_has_no_runtime_or_learning_dependencies():
    source = Path(public.__file__).read_text()
    assert "from .rl" not in source
    assert "import torch" not in source
    assert "subprocess" not in source
    assert [(r.path, r.methods) for r in public.router.routes] == [
        ("/api/v2/ppo/research", {"GET"})
    ]


def test_invalid_unicode_and_excessive_nesting_fail_closed(payload, client):
    payload["limitations"] = ["\ud800"]
    public.PUBLIC_PATH.write_text(json.dumps(signed(payload)))
    assert client.get("/api/v2/ppo/research").status_code == 503
    public.PUBLIC_PATH.write_text("[" * 2000 + "]" * 2000)
    assert client.get("/api/v2/ppo/research").status_code == 503


def test_committed_research_export_matches_read_only_route(monkeypatch):
    # This portable checked-in projection is the actual UI input, not a raw run.
    path = Path(public.__file__).parent / "data" / "adaptive_learning.json"
    monkeypatch.setattr(public, "PUBLIC_PATH", path)
    app = FastAPI()
    app.include_router(public.router)
    with TestClient(app) as client:
        response = client.get("/api/v2/ppo/research")
    assert response.status_code == 200
    evidence = response.json()["evidence"]
    assert evidence["sha256"] == public.digest(
        {k: v for k, v in evidence.items() if k != "sha256"}
    )
    assert len(evidence["policies"]) == 4
    assert len(evidence["comparisons"]) == 26
    actors = {row["id"]: row for row in evidence["actors"]}
    for identifier in ("M24", "E-beta"):
        assert all(
            row["status"] == "stopped" and row["hard_safety_passed"] is None
            for row in actors[identifier]["cases"]
        )
    assert (
        sum(row["mpc_modes"]["recovery"] for row in actors["OperatingMPC"]["cases"])
        == 5327
    )
    assert (
        sum(row["mpc_modes"]["fallback"] for row in actors["OperatingMPC"]["cases"])
        == 49
    )
    assert all(p["operational_status"] == "experimental" for p in evidence["policies"])

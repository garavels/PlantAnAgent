"""Inert plans and receipts using synthetic identities, never production reports."""

from copy import deepcopy
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest

from plant_backend.rl.v2.training import custody_documents as docs
from plant_backend.rl.v2.training.custody_json import (
    authenticate,
    canonical_bytes,
    domain_digest,
)


def _plan(root: Path) -> dict[str, Any]:
    return docs.make_plan(
        root,
        authorization=authenticate(
            {"schema": "synthetic-authorization", "protocols": {"fixture": True}}
        ),
        source={"fixture": True},
        runtime={"sha256": "a" * 64},
        host={"fixture": True},
    )


def _terminal(plan: dict[str, Any], index: int) -> dict[str, Any]:
    # A self-hashed inert envelope is not production admission. The production
    # verifier must reconstruct its referenced files and full nested shapes.
    return authenticate(
        {
            **{key: None for key in docs.TERMINAL_FIELDS},
            "schema": docs.TERMINAL_SCHEMA,
            "status": "admitted",
            "cell": plan["cells"][index],
            "plan_sha256": plan["sha256"],
        }
    )


@pytest.mark.parametrize("code", list(docs.FAILURES))
def test_each_failure_has_closed_categorical_evidence(code: str) -> None:
    status, failure = docs.choose_failure({code: "post_child"})
    assert status == docs.FAILURES[code][0]
    assert failure == {
        "code": code,
        "stage": "post_child",
        "message": docs.FAILURES[code][1],
    }
    assert docs.choose_failure({}) == ("admitted", None)


@pytest.mark.parametrize("earlier,later", list(combinations(docs.FAILURES, 2)))
def test_all_failure_precedence_collisions(earlier: str, later: str) -> None:
    # Reverse insertion order so observation arrival cannot choose the result.
    status, result = docs.choose_failure({later: "child", earlier: "post_child"})
    assert status == docs.FAILURES[earlier][0]
    assert result is not None and result["code"] == earlier


def test_failure_payload_cannot_be_an_exception_or_observed_value() -> None:
    with pytest.raises(ValueError):
        docs.choose_failure({"arbitrary exception text": "child"})
    with pytest.raises(ValueError):
        docs.failure("child_timeout", "arbitrary stage")


def test_plan_digest_binds_only_complete_pre_run_identity(tmp_path: Path) -> None:
    root = tmp_path / "unopened"
    plan = _plan(root)
    assert set(plan) == docs.PLAN_FIELDS
    assert len(plan["cells"]) == 15
    assert not root.exists()
    identity = {key: plan[key] for key in docs.PLAN_IDENTITY_FIELDS}
    assert plan["deterministic_evidence_sha256"] == domain_digest(
        docs.DETERMINISTIC_DOMAIN, identity
    )
    assert _plan(root) == plan
    assert (
        _plan(tmp_path / "different")["deterministic_evidence_sha256"]
        != plan["deterministic_evidence_sha256"]
    )
    for field in docs.PLAN_IDENTITY_FIELDS:
        modified = deepcopy(identity)
        modified[field] = {"mutation": True}
        assert (
            domain_digest(docs.DETERMINISTIC_DOMAIN, modified)
            != plan["deterministic_evidence_sha256"]
        )


def test_every_claim_binds_the_immediately_preceding_terminal(tmp_path: Path) -> None:
    plan = _plan(tmp_path / "unopened")
    previous = None
    for index, cell in enumerate(plan["cells"]):
        claim = docs.make_claim(
            plan, cell, previous, timestamp="2026-09-11T12:00:00.000000Z"
        )
        assert set(claim) == docs.CLAIM_FIELDS
        assert claim["previous_terminal_sha256"] == (
            None if previous is None else previous["sha256"]
        )
        assert claim["cell"] == cell
        previous = _terminal(plan, index)
    assert not (tmp_path / "unopened").exists()


@pytest.mark.parametrize(
    "defect", ["missing", "wrong_cell", "aborted", "digest", "timestamp"]
)
def test_claim_cannot_skip_or_relabel_previous_attempt(
    tmp_path: Path, defect: str
) -> None:
    plan = _plan(tmp_path / "unopened")
    previous: dict[str, Any] | None = _terminal(plan, 0)
    timestamp = "2026-09-11T12:00:00.000000Z"
    if defect == "missing":
        previous = None
    elif defect == "wrong_cell":
        previous = _terminal(plan, 2)
    elif defect == "aborted":
        previous = authenticate({**previous, "status": "campaign_integrity_failed"})
    elif defect == "digest":
        previous = {**previous, "sha256": "0" * 64}
    else:
        timestamp = "2026-09-11"
    with pytest.raises(ValueError):
        docs.make_claim(plan, plan["cells"][1], previous, timestamp=timestamp)


def test_production_and_synthetic_schemas_remain_distinct() -> None:
    assert docs.REPORT_SCHEMA != docs.SYNTHETIC_REPORT_SCHEMA
    assert (
        docs.SUCCESS_CLAIM
        not in canonical_bytes(_plan(Path("/private/tmp/fixture-unopened"))).decode()
    )

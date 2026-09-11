"""Two-arm study eligibility reconstructed from independently reopened fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import beta_controller_benchmark as benchmark
from ..training import custody_io as io
from ..training.custody_json import authenticate, canonical_bytes, require_envelope
from . import arm_evidence
from .contract import PREFIX, analysis_contract
from .sensitivity import rescore_pair
from .statistics import finite_real

SCHEMA = PREFIX + "synthetic-pair-evidence-v1"
FILENAME = "synthetic-pair.json"
CLAIM_BOUNDARY = "synthetic pairwise physical evidence only; no production eligibility, selection or stage authority"
FIELDS = frozenset(
    {
        "schema",
        "analysis_contract_sha256",
        "stage",
        "inputs",
        "arms",
        "checks",
        "exclusion_reasons",
        "value_comparison_eligible",
        "delta_eur",
        "sensitivity",
        "service_comparison",
        "claim_boundary",
        "result_claim",
        "sha256",
    }
)


def _comparison(
    policy: dict[str, Any] | None, baseline: dict[str, Any] | None
) -> dict[str, Any]:
    arms = {"policy": policy, "e_beta": baseline}
    checks = {}
    for name, evidence in arms.items():
        checks[f"{name}_evidence_present"] = evidence is not None
        for gate in ("execution_complete", "hard_safety_passed", "cycle_feasible"):
            checks[f"{name}_{gate}"] = (
                evidence is not None and evidence["classification"][gate] is True
            )
    checks.update(
        {
            "same_stage_and_seed": False,
            "same_source": False,
            "same_manifest": False,
            "same_initial_public_state": False,
            "same_common_public_issues": False,
            "consistent_solver_versions": False,
            "service_comparable": False,
        }
    )
    service: dict[str, Any] | None = None
    if policy is not None and baseline is not None:
        checks["same_stage_and_seed"] = (policy["stage"], policy["seed"]) == (
            baseline["stage"],
            baseline["seed"],
        )
        checks["same_source"] = canonical_bytes(
            policy["source_provenance"]
        ) == canonical_bytes(baseline["source_provenance"])
        checks["same_manifest"] = (
            policy["manifest_identity"] is not None
            and policy["manifest_identity"] == baseline["manifest_identity"]
        )
        checks["same_initial_public_state"] = (
            policy["initial_public_state_sha256"] is not None
            and policy["initial_public_state_sha256"]
            == baseline["initial_public_state_sha256"]
        )
        left, right = policy["public_issue_sha256"], baseline["public_issue_sha256"]
        common = min(len(left), len(right))
        checks["same_common_public_issues"] = (
            common > 0 and left[:common] == right[:common]
        )
        checks["consistent_solver_versions"] = (
            len(set(policy["highs_versions"]) | set(baseline["highs_versions"])) <= 1
        )
        if (
            policy["physical_metrics"] is not None
            and baseline["physical_metrics"] is not None
        ):
            service = {
                "tolerance_kg": benchmark.BETA_SERVICE_COMPARISON_TOLERANCE_KG,
                "absolute_differences_kg": {
                    key: abs(
                        finite_real(policy["physical_metrics"][key])
                        - finite_real(baseline["physical_metrics"][key])
                    )
                    for key in ("requested_kg", "delivered_kg", "unmet_kg")
                },
            }
            checks["service_comparable"] = all(
                value <= benchmark.BETA_SERVICE_COMPARISON_TOLERANCE_KG
                for value in service["absolute_differences_kg"].values()
            )
    eligible = all(checks.values())
    delta = None
    sensitivity = None
    if eligible:
        if policy is None or baseline is None:  # pragma: no cover - gate invariant
            raise RuntimeError("eligible pair has missing evidence")
        delta = finite_real(
            policy["economic_metrics"]["net_operating_value_eur"]
            - baseline["economic_metrics"]["net_operating_value_eur"]
        )
        sensitivity = rescore_pair(
            policy["economic_metrics"]["components_eur"],
            baseline["economic_metrics"]["components_eur"],
        )
    return {
        "checks": checks,
        "exclusion_reasons": [name for name, passed in checks.items() if not passed],
        "value_comparison_eligible": eligible,
        "service_comparison": service,
        "delta_eur": delta,
        "sensitivity": sensitivity,
    }


def analyze_synthetic_pair(
    policy_path: Path | str | None, e_beta_path: Path | str | None
) -> dict[str, Any]:
    """Admit each source afresh; explicit missing arms get no value or sensitivity.

    Invalid or missing files raise rather than being silently converted into an
    absent arm. Production registries will bind which attempts are absent.
    """
    paths = {
        name: (
            None
            if path is None
            else arm_evidence.require_synthetic_path(
                path, filename=arm_evidence.FILENAME
            )
        )
        for name, path in (("policy", policy_path), ("e_beta", e_beta_path))
    }
    before = {
        name: arm_evidence.file_stamp(path)
        for name, path in paths.items()
        if path is not None
    }
    admitted: dict[str, dict[str, Any] | None] = {}
    for name, path in paths.items():
        evidence = (
            None if path is None else arm_evidence.verify_synthetic_arm_file(path)
        )
        expected_arm = (
            benchmark.PPO_ARM if name == "policy" else benchmark.BETA_ECONOMIC_MPC_ARM
        )
        if evidence is not None and evidence["actor"]["arm"] != expected_arm:
            raise ValueError("pair reference has the wrong actor")
        admitted[name] = evidence
    result = authenticate(
        {
            "schema": SCHEMA,
            "analysis_contract_sha256": analysis_contract()["sha256"],
            "stage": "fixture",
            "inputs": {
                name: None if path is None else str(path)
                for name, path in paths.items()
            },
            "arms": admitted,
            **_comparison(admitted["policy"], admitted["e_beta"]),
            "claim_boundary": CLAIM_BOUNDARY,
            "result_claim": None,
        }
    )
    for name, path in paths.items():
        if path is not None:
            evidence = admitted[name]
            if (
                evidence is None
                or io.file_record(path) != evidence["file"]
                or arm_evidence.file_stamp(path) != before[name]
            ):
                raise ValueError("pair source changed during evidence admission")
    return result


def verify_synthetic_pair_file(path: Path | str) -> dict[str, Any]:
    """Reopen both complete arm ledgers; report arithmetic is never admission."""
    target = arm_evidence.require_synthetic_path(path, filename=FILENAME)
    before = arm_evidence.file_stamp(target)
    raw = io.read_regular(target)
    report = require_envelope(raw, schema=SCHEMA, fields=FIELDS)
    if (
        report["stage"] != "fixture"
        or report["result_claim"] is not None
        or report["claim_boundary"] != CLAIM_BOUNDARY
        or report["analysis_contract_sha256"] != analysis_contract()["sha256"]
    ):
        raise ValueError("synthetic pair stage/claim differs")
    inputs = report["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != {"policy", "e_beta"}:
        raise ValueError("pair input references differ")
    if any(
        value is not None and not isinstance(value, str) for value in inputs.values()
    ):
        raise ValueError("pair input references must be paths or explicit nulls")
    source_stamps = {
        name: arm_evidence.file_stamp(
            arm_evidence.require_synthetic_path(value, filename=arm_evidence.FILENAME)
        )
        for name, value in inputs.items()
        if value is not None
    }
    expected = analyze_synthetic_pair(inputs["policy"], inputs["e_beta"])
    if canonical_bytes(expected) != raw:
        raise ValueError("pair evidence differs from its independently admitted arms")
    if io.read_regular(target) != raw or arm_evidence.file_stamp(target) != before:
        raise ValueError("synthetic pair changed during verification")
    record = io.file_record(target)
    for name, stamp in source_stamps.items():
        source = Path(inputs[name])
        if (
            io.file_record(source) != expected["arms"][name]["file"]
            or arm_evidence.file_stamp(source) != stamp
        ):
            raise ValueError("pair source changed during detached verification")
    if arm_evidence.file_stamp(target) != before:
        raise ValueError("synthetic pair changed during final source checks")
    return {
        "verification": "synthetic_pair_reconstructed",
        "source_evidence_verified": True,
        "registered_checkpoint_verified": False,
        "result_claim": None,
        "sha256": expected["sha256"],
        "value_comparison_eligible": expected["value_comparison_eligible"],
        "exclusion_reasons": expected["exclusion_reasons"],
        "delta_eur": expected["delta_eur"],
        "file": record,
    }

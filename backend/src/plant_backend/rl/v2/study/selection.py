"""Frozen development ranking over complete, independently admitted pair inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .contract import (
    BETA_CORE_ALGORITHM_IDS,
    BETA_OPTIMIZER_SEEDS,
    EPISODE_COUNT,
    PREFIX,
    analysis_contract,
)
from .statistics import binomial, finite_real, point_statistics


@dataclass(frozen=True)
class DevelopmentPolicy:
    """Analysis input, not a verification certificate or stage authorization.

    The campaign's detached layer must establish the admission flags and supply
    null for each pair that fails its complete execution/safety/cycle/service gate.
    """

    training_admitted: bool
    development_report_verified: bool
    paired_deltas_eur: tuple[float | None, ...]

    def __post_init__(self) -> None:
        if (
            type(self.training_admitted) is not bool
            or type(self.development_report_verified) is not bool
        ):
            raise ValueError("development admission flags must be exact booleans")
        if (
            type(self.paired_deltas_eur) is not tuple
            or len(self.paired_deltas_eur) != EPISODE_COUNT
        ):
            raise ValueError(
                "development input must retain exactly 32 ordered pair slots"
            )
        for value in self.paired_deltas_eur:
            if value is not None:
                finite_real(value)


def select_family(
    family: str, policies: Mapping[int, DevelopmentPolicy]
) -> dict[str, Any]:
    if family not in BETA_CORE_ALGORITHM_IDS:
        raise ValueError("development selection requires a frozen core family")
    if any(type(seed) is not int for seed in policies) or set(policies) != set(
        BETA_OPTIMIZER_SEEDS
    ):
        raise ValueError("selection must retain all five exact optimizer identifiers")
    rows: list[dict[str, Any]] = []
    for seed in BETA_OPTIMIZER_SEEDS:
        policy = policies[seed]
        if type(policy) is not DevelopmentPolicy:
            raise ValueError("selection requires explicit development evidence inputs")
        policy.__post_init__()
        missing = [
            index
            for index, value in enumerate(policy.paired_deltas_eur)
            if value is None
        ]
        reasons = []
        if not policy.training_admitted:
            reasons.append("training_not_admitted")
        if not policy.development_report_verified:
            reasons.append("development_report_not_verified")
        if missing:
            reasons.append("incomplete_eligible_pair_vector")
        statistics = None if reasons else point_statistics(policy.paired_deltas_eur)
        key = (
            None
            if statistics is None
            else [
                statistics["iqm"],
                statistics["cvar20"],
                statistics["median"],
                statistics["positive_fraction"],
                -seed,
            ]
        )
        rows.append(
            {
                "optimizer_seed": seed,
                "training_admitted": policy.training_admitted,
                "development_report_verified": policy.development_report_verified,
                "paired_deltas_eur": [
                    None if value is None else float(value)
                    for value in policy.paired_deltas_eur
                ],
                "available_pair_count": EPISODE_COUNT - len(missing),
                "missing_pair_indices": missing,
                "development_admissible": not reasons,
                "exclusion_reasons": reasons,
                "selection_statistics": statistics,
                "selection_key": key,
                "selected": False,
            }
        )
    eligible = [row for row in rows if row["development_admissible"]]
    champion = (
        max(eligible, key=lambda row: tuple(row["selection_key"])) if eligible else None
    )
    if champion is not None:
        champion["selected"] = True
    return {
        "schema": PREFIX + "development-selection-v1",
        "analysis_contract_sha256": analysis_contract()["sha256"],
        "family": family,
        "orientation": "policy minus E-beta",
        "policies": rows,
        "development_admission": binomial(len(eligible), 5),
        "champion_optimizer_seed": (
            None if champion is None else champion["optimizer_seed"]
        ),
        "claim_boundary": "selection calculation from supplied evidence; no stage authorization",
    }


def verify_selection_result(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct arithmetic/ranking; independent arm verification is separate."""
    from ..training.custody_json import canonical_bytes

    rows = raw.get("policies")
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("selection must retain five policy rows")
    policies: dict[int, DevelopmentPolicy] = {}
    for expected_seed, row in zip(BETA_OPTIMIZER_SEEDS, rows, strict=True):
        if (
            not isinstance(row, dict)
            or type(row.get("optimizer_seed")) is not int
            or row["optimizer_seed"] != expected_seed
        ):
            raise ValueError("selection optimizer order differs")
        values = row.get("paired_deltas_eur")
        if not isinstance(values, list):
            raise ValueError("selection must retain its pair inputs")
        training = row.get("training_admitted")
        development = row.get("development_report_verified")
        if type(training) is not bool or type(development) is not bool:
            raise ValueError("selection admission flags must be exact booleans")
        policies[expected_seed] = DevelopmentPolicy(
            training, development, tuple(values)
        )
    family = raw.get("family")
    if not isinstance(family, str):
        raise ValueError("selection family must be a frozen identifier")
    expected = select_family(family, policies)
    if canonical_bytes(raw) != canonical_bytes(expected):
        raise ValueError("selection calculation differs from its inputs")
    return {
        "verification": "selection_reconstructed",
        "source_evidence_verified": False,
        "result_claim": None,
    }

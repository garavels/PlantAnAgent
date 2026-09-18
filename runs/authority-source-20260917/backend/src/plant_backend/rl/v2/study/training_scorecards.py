"""All-family training diagnostics, downstream of checkpoint/custody admission."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any

from ..beta_config import RESOURCE_NAMES
from ..training.custody_json import authenticate
from .contract import BETA_CORE_ALGORITHM_IDS, BETA_OPTIMIZER_SEEDS, PREFIX
from .operational import _count, distribution, operational_contract
from .statistics import binomial


def rollout_scorecard(
    algorithm: str,
    records: Sequence[Mapping[str, Any]],
    *,
    expected_rollouts: int = 128,
) -> dict[str, Any]:
    """Validate the ledger chain, not the trained artifact or its state attestation."""
    from ..training.constrained_ppo import RolloutLedgerV1

    if (
        algorithm not in BETA_CORE_ALGORITHM_IDS
        or _count(expected_rollouts, "expected rollouts") == 0
    ):
        raise ValueError("training scorecard algorithm/endpoint differs")
    if len(records) != expected_rollouts:
        raise ValueError("training scorecard requires the complete admitted endpoint")
    previous = None
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(records, 1):
        ledger = RolloutLedgerV1.from_dict(raw)
        ledger.verify(expected_previous_sha256=previous)
        if ledger.algorithm_id != algorithm or ledger.rollout_index != index:
            raise ValueError("training rollout identity differs")
        previous = ledger.sha256
        rows.append(ledger.to_dict())
    resources = {}
    for index, resource in enumerate(RESOURCE_NAMES):
        resources[resource] = {
            "raw_cost_total": math.fsum(row["raw_cost"][index] for row in rows),
            "training_cost_total": math.fsum(
                row["training_cost"][index] for row in rows
            ),
            "completed_episode_cost_mean_by_rollout": [
                (
                    None
                    if row["mean_completed_episode_cost"] is None
                    else row["mean_completed_episode_cost"][index]
                )
                for row in rows
            ],
            "multipliers": [row["multipliers"][index] for row in rows],
            "proportional_state": [row["proportional_state"][index] for row in rows],
            "integral_state": [row["integral_state"][index] for row in rows],
            "cost_value_losses": [row["cost_value_losses"][index] for row in rows],
            "saturated": [row["multiplier_saturated"][index] for row in rows],
            "saturated_rollouts": sum(
                row["multiplier_saturated"][index] for row in rows
            ),
            "observed_rollouts": len(rows),
            "saturation_fraction": sum(
                row["multiplier_saturated"][index] for row in rows
            )
            / len(rows),
        }
    return authenticate(
        {
            "schema": PREFIX + "training-rollout-scorecard-v1",
            "operational_contract_sha256": operational_contract()["sha256"],
            "algorithm": algorithm,
            "expected_rollouts": expected_rollouts,
            "rollouts": rows,
            "resources": resources,
            "transitions": sum(row["transition_count"] for row in rows),
            "optimizer_steps": sum(row["optimizer_steps"] for row in rows),
            "complete_cycles": sum(row["complete_cycle_count"] for row in rows),
            "completed_episodes": sum(row["completed_episode_count"] for row in rows),
            "reward_return": distribution([row["reward_return"] for row in rows]),
            "elapsed_compute_s": math.fsum(row["elapsed_compute_s"] for row in rows),
            "source_evidence_verified": False,
            "registered_checkpoint_verified": False,
            "result_claim": None,
        }
    )


@dataclass(frozen=True)
class TrainingSlot:
    algorithm: str
    optimizer_seed: int
    training_completed: bool
    checkpoint_admitted: bool
    failure: str | None
    rollouts: tuple[Mapping[str, Any], ...] | None


def training_scorecard(
    slots: Sequence[TrainingSlot], *, expected_rollouts: int = 128
) -> dict[str, Any]:
    """Pure all-15 accounting; stage verifier must supply independently admitted inputs."""
    if _count(expected_rollouts, "expected rollouts") == 0:
        raise ValueError("expected rollouts must be positive")
    identities = [
        (family, seed)
        for family in BETA_CORE_ALGORITHM_IDS
        for seed in BETA_OPTIMIZER_SEEDS
    ]
    if [(row.algorithm, row.optimizer_seed) for row in slots] != identities:
        raise ValueError(
            "training scorecard requires exact family-first 3 x 5 registry"
        )
    rows: list[dict[str, Any]] = []
    for slot in slots:
        if (
            type(slot.optimizer_seed) is not int
            or type(slot.training_completed) is not bool
            or type(slot.checkpoint_admitted) is not bool
        ):
            raise ValueError("training slot declarations must have exact types")
        if slot.checkpoint_admitted and not slot.training_completed:
            raise ValueError("checkpoint cannot be admitted before training completes")
        if slot.checkpoint_admitted != (slot.rollouts is not None):
            raise ValueError(
                "training diagnostics must cover every admitted checkpoint only"
            )
        if slot.checkpoint_admitted != (slot.failure is None) or (
            slot.failure is not None
            and (not isinstance(slot.failure, str) or not slot.failure)
        ):
            raise ValueError("training slot must retain its failure or admission")
        rows.append(
            {
                "algorithm": slot.algorithm,
                "optimizer_seed": slot.optimizer_seed,
                "training_completed": slot.training_completed,
                "checkpoint_admitted": slot.checkpoint_admitted,
                "failure": slot.failure,
                "diagnostics": (
                    None
                    if slot.rollouts is None
                    else rollout_scorecard(
                        slot.algorithm,
                        slot.rollouts,
                        expected_rollouts=expected_rollouts,
                    )
                ),
            }
        )
    families = {}
    for family in BETA_CORE_ALGORITHM_IDS:
        subset = [row for row in rows if row["algorithm"] == family]
        families[family] = {
            "training_completed": binomial(
                sum(row["training_completed"] for row in subset), 5
            ),
            "checkpoint_admitted": binomial(
                sum(row["checkpoint_admitted"] for row in subset), 5
            ),
            "missing_optimizer_seeds": [
                row["optimizer_seed"]
                for row in subset
                if not row["checkpoint_admitted"]
            ],
            "slots": deepcopy(subset),
        }
    return authenticate(
        {
            "schema": PREFIX + "training-scorecard-v1",
            "operational_contract_sha256": operational_contract()["sha256"],
            "families": families,
            "training_completed_count": sum(row["training_completed"] for row in rows),
            "checkpoint_admitted_count": sum(
                row["checkpoint_admitted"] for row in rows
            ),
            "registered_slot_count": 15,
            "source_evidence_verified": False,
            "registered_checkpoint_verified": False,
            "result_claim": None,
        }
    )

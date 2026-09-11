"""Fixed stage destinations and all-576 registry; no authorization or resets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..training.custody_authorization import repository_root
from ..training.custody_json import authenticate, canonical_bytes
from ..training.episode_allocator import DEVELOPMENT_SEED_MIN, HELDOUT_SEED_MIN
from ..training.registered_contract import FAMILY_SLUGS
from .contract import (
    BETA_CORE_ALGORITHM_IDS,
    BETA_OPTIMIZER_SEEDS,
    EPISODE_COUNT,
    PREFIX,
    analysis_contract,
)
from .latency import latency_contract
from .operational import operational_contract

SCHEMA = PREFIX + "registry-v1"
STAGES = ("development", "heldout")
BASELINES = ("H", "M24", "E-beta")


def actors() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = [
        {
            "id": arm.lower(),
            "arm": arm,
            "algorithm": None,
            "optimizer_seed": None,
            "training_order": None,
        }
        for arm in BASELINES
    ]
    for algorithm, slug in zip(BETA_CORE_ALGORITHM_IDS, FAMILY_SLUGS, strict=True):
        for seed in BETA_OPTIMIZER_SEEDS:
            result.append(
                {
                    "id": f"{slug}-seed-{seed}",
                    "arm": "PPO",
                    "algorithm": algorithm,
                    "optimizer_seed": seed,
                    "training_order": len(result) - 2,
                }
            )
    return result


def actor_identity(actor: dict[str, Any]) -> dict[str, Any]:
    return {key: actor[key] for key in ("arm", "algorithm", "optimizer_seed")}


def episode_seeds(stage: str) -> list[int]:
    if stage not in STAGES:
        raise ValueError("study stage must be development or heldout")
    start = DEVELOPMENT_SEED_MIN if stage == "development" else HELDOUT_SEED_MIN
    return list(range(start, start + EPISODE_COUNT))


def cells_for_seeds(seeds: list[int]) -> list[dict[str, Any]]:
    """Identity-only arithmetic used by fixtures and the exact production registry."""
    if (
        len(seeds) != EPISODE_COUNT
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != EPISODE_COUNT
    ):
        raise ValueError("study registry requires 32 unique exact episode seeds")
    result: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        for actor in actors():
            order = len(result) + 1
            result.append(
                {
                    "order": order,
                    "episode_index": index,
                    "episode_seed": seed,
                    "actor": actor,
                    "stem": f"{order:04d}-episode-{seed}-{actor['id']}",
                }
            )
    return result


def stage_registry(training_root: Path | str, stage: str) -> dict[str, Any]:
    seeds = episode_seeds(stage)
    root = Path(training_root)
    if not root.is_absolute() or root != root.resolve() or root == root.parent:
        raise ValueError("training root must be an absolute canonical named directory")
    stage_root = root.with_name(root.name + f".{stage}-v1")
    if root.is_relative_to(repository_root()) or stage_root.is_relative_to(
        repository_root()
    ):
        raise ValueError("study evidence roots must remain outside Git")
    cells = []
    for identity in cells_for_seeds(seeds):
        directory = stage_root / "cells" / identity["stem"]
        cells.append(
            {
                **identity,
                "directory": str(directory),
                "claim_path": str(directory / "claimed.json"),
                "terminal_path": str(directory / "terminal.json"),
                "arm_path": str(directory / "arm.json"),
                "latency_path": str(directory / "latency.json"),
                "stdout_path": str(directory / "stdout.log"),
                "stderr_path": str(directory / "stderr.log"),
            }
        )
    return authenticate(
        {
            "schema": SCHEMA,
            "stage": stage,
            "analysis_contract_sha256": analysis_contract()["sha256"],
            "operational_contract_sha256": operational_contract()["sha256"],
            "latency_contract_sha256": latency_contract()["sha256"],
            "training_root": str(root),
            "stage_root": str(stage_root),
            "episode_seeds": seeds,
            "actors": actors(),
            "cells": cells,
            "registry_path": str(stage_root / "registry.json"),
            "authorization_path": str(stage_root / "authorization.json"),
            "opening_claim_path": str(stage_root / "opened.json"),
            "report_path": str(stage_root / "report.json"),
            "verification_path": str(stage_root / "verification.json"),
            "claim_boundary": "fixed identities only; no reset, execution, stage or checkpoint authorization",
        }
    )


def verify_registry(raw: dict[str, Any]) -> dict[str, Any]:
    rebuilt = stage_registry(raw["training_root"], raw["stage"])
    if canonical_bytes(raw) != canonical_bytes(rebuilt):
        raise ValueError(
            "study registry differs from its exact stage/order/destinations"
        )
    return rebuilt

"""Frozen configuration and ordered matrix; no process or environment execution."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .. import beta_spaces
from ..beta_config import CyclicMarketTaskConfigV1
from .config import (
    BETA_CORE_ALGORITHM_IDS,
    BETA_OPTIMIZER_SEEDS,
    BETA_REGISTERED_RUN_KIND,
    BETA_REGISTERED_TOTAL_TIMESTEPS,
    BetaRLTrainingRunV1,
    ControllerNeutralTrainingConfigV1,
)
from .custody_json import canonical_bytes, domain_digest

PROTOCOL_ID = "plant-beta-constrained-rl-registered-training-custody-v1"
TEMPLATE_DOMAIN = "plant-beta-constrained-rl-registered-task-template-v1"
TEMPLATE_SHA256 = "89abbe6288d512c254d8186f56c6d269b9f273fb49bbb13d24d7b02425bfa77b"
FAMILY_SLUGS = ("pi-lagrangian", "i-lagrangian", "scalar-ppo")
TIMEOUT_SECONDS = 36_000
LOG_LIMIT_BYTES = 67_108_864
MINIMUM_FREE_BYTES = 5_368_709_120


def is_registered_config_path(path: Path) -> bool:
    """Recognize reserved cell paths even if the enclosing plan was removed."""
    if path.parent.name != "configs":
        return False
    if (path.parent.parent / "plan.json").exists():
        return True
    names = {
        f"{family * 5 + offset:02d}-{slug}-seed-{seed}.json"
        for family, slug in enumerate(FAMILY_SLUGS)
        for offset, seed in enumerate(BETA_OPTIMIZER_SEEDS, start=1)
    }
    return path.name in names


def task_template_digest(config: ControllerNeutralTrainingConfigV1) -> str:
    value = config.to_dict()
    del value["seed"], value["output_dir"]
    if not isinstance(value["beta_rl"], dict):
        raise ValueError("registered task template requires beta_rl")
    del value["beta_rl"]["algorithm"]
    return domain_digest(TEMPLATE_DOMAIN, value)


def require_task_template(config: ControllerNeutralTrainingConfigV1) -> None:
    if task_template_digest(config) != TEMPLATE_SHA256:
        raise ValueError("registered task template differs from frozen defaults")


def registered_config(
    output: Path, algorithm: str, seed: int
) -> ControllerNeutralTrainingConfigV1:
    return ControllerNeutralTrainingConfigV1(
        environment_id=beta_spaces.ENVIRONMENT_ID,
        seed=seed,
        total_timesteps=BETA_REGISTERED_TOTAL_TIMESTEPS,
        checkpoint_every=0,
        output_dir=str(output),
        beta_task=CyclicMarketTaskConfigV1(),
        beta_rl=BetaRLTrainingRunV1(
            algorithm=algorithm,
            run_kind=BETA_REGISTERED_RUN_KIND,
            episode_seed_base=1_000_000,
        ),
    ).validate()


def matrix(root: Path) -> list[dict[str, Any]]:
    """Return all 15 immutable identities; never create paths or reset tasks."""
    if not root.is_absolute() or root != root.resolve():
        raise ValueError(
            "campaign root must be absolute and canonical without symlinks"
        )
    repository = Path(__file__).resolve().parents[6]
    if root.is_relative_to(repository):
        raise ValueError("campaign root must be external to the repository")
    cells: list[dict[str, Any]] = []
    for algorithm, slug in zip(BETA_CORE_ALGORITHM_IDS, FAMILY_SLUGS, strict=True):
        for seed in BETA_OPTIMIZER_SEEDS:
            order = len(cells) + 1
            stem = f"{order:02d}-{slug}-seed-{seed}"
            output = root / "artifacts" / stem
            config = registered_config(output, algorithm, seed)
            cells.append(
                {
                    "order": order,
                    "algorithm": algorithm,
                    "family_slug": slug,
                    "optimizer_seed": seed,
                    "config_path": str(root / "configs" / f"{stem}.json"),
                    "config_sha256": hashlib.sha256(
                        canonical_bytes(config.to_dict())
                    ).hexdigest(),
                    "output_path": str(output),
                    "stdout_path": str(root / "logs" / f"{stem}.stdout.log"),
                    "stderr_path": str(root / "logs" / f"{stem}.stderr.log"),
                    "claim_path": str(root / "attempts" / stem / "claimed.json"),
                    "completed_receipt_path": str(
                        root / "attempts" / stem / "completed.json"
                    ),
                    "failed_receipt_path": str(
                        root / "attempts" / stem / "failed.json"
                    ),
                }
            )
    return cells


def require_canonical_config(
    raw: bytes, config: ControllerNeutralTrainingConfigV1
) -> None:
    if config.beta_rl is None or config.beta_rl.run_kind != BETA_REGISTERED_RUN_KIND:
        raise ValueError("registered config path requires a registered declaration")
    require_task_template(config)
    if raw != canonical_bytes(config.to_dict()):
        raise ValueError("registered configuration must contain exact canonical bytes")

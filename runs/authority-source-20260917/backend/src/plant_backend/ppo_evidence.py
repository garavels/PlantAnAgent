"""Bounded public training exports; the HTTP runtime never opens training custody."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .rl.v2.training.config import BETA_CORE_ALGORITHM_IDS, BETA_OPTIMIZER_SEEDS

PUBLIC_ROOT = Path(__file__).resolve().parents[3] / "runs" / "commission-rl"
SCHEMA = "plant-public-ppo-training-v1"
MAX_BYTES = 4 * 1024 * 1024
router = APIRouter(prefix="/api/v2/ppo", tags=["ppo-evidence"])
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Vector = Annotated[list[float], Field(min_length=4, max_length=4)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Rollout(StrictModel):
    index: int = Field(ge=1, le=128)
    transitions: int = Field(ge=8192, le=1_048_576)
    objective_sum: float
    completed_episodes: int = Field(ge=0)
    complete_cycles: int = Field(ge=0)
    raw_cost: Vector
    training_cost: Vector
    mean_episode_cost: Vector | None
    multipliers: Vector
    cost_value_losses: Vector


class Slot(StrictModel):
    algorithm: str
    optimizer_seed: int
    training_completed: bool
    checkpoint_admitted: bool
    failure: str | None
    checkpoint_evidence_sha256: Digest | None
    rollouts: list[Rollout] = Field(max_length=128)

    @model_validator(mode="after")
    def endpoint(self) -> Slot:
        if self.checkpoint_admitted:
            if (
                not self.training_completed
                or self.failure is not None
                or self.checkpoint_evidence_sha256 is None
                or len(self.rollouts) != 128
            ):
                raise ValueError("admitted slot requires its complete endpoint")
        elif not self.failure or self.checkpoint_evidence_sha256 or self.rollouts:
            raise ValueError("failed slot must retain failure without admitted curves")
        for index, row in enumerate(self.rollouts, 1):
            if row.index != index or row.transitions != index * 8192:
                raise ValueError(
                    "rollout sequence differs from the registered endpoint"
                )
        return self


class TrainingExport(StrictModel):
    schema_version: Literal["plant-public-ppo-training-v1"]
    source_commit: Annotated[str, Field(pattern=r"^[a-f0-9]{40}$")]
    training_report_sha256: Digest
    runtime_provenance_sha256: Digest
    campaign_status: Literal["completed", "failed", "aborted"]
    custody_admitted_at_export: Literal[True]
    evaluation_performed: Literal[False]
    slots: list[Slot] = Field(min_length=15, max_length=15)
    sha256: Digest

    @model_validator(mode="after")
    def registry(self) -> TrainingExport:
        identities = [(s.algorithm, s.optimizer_seed) for s in self.slots]
        if identities != [
            (family, seed)
            for family in BETA_CORE_ALGORITHM_IDS
            for seed in BETA_OPTIMIZER_SEEDS
        ]:
            raise ValueError("training export requires every registered slot in order")
        return self


def digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def validate(payload: dict[str, Any]) -> dict[str, Any]:
    TrainingExport.model_validate(payload)
    if digest({k: v for k, v in payload.items() if k != "sha256"}) != payload["sha256"]:
        raise ValueError("training export checksum differs")
    return payload


def read_export() -> dict[str, Any] | None:
    path = PUBLIC_ROOT / "training.json"
    if not path.exists() and not path.is_symlink():
        return None
    # Same bounded, no-follow regular-file reader used by custody; importing it
    # performs no host probe, model load, graph admission or environment reset.
    from .rl.v2.study.archive import _read_stored
    from .rl.v2.training.custody_io import require_directory
    from .rl.v2.training.custody_json import decode

    require_directory(path.parent)
    raw = _read_stored(path, MAX_BYTES)
    return validate(decode(raw))


@router.get("/training")
def training(request: Request) -> dict[str, Any]:
    if request.query_params:
        raise HTTPException(
            status_code=422, detail="Training export takes no parameters"
        )
    try:
        evidence = read_export()
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise HTTPException(
            status_code=503, detail="Training export failed integrity checks"
        ) from exc
    return {
        "status": "pending" if evidence is None else "available",
        "evidence": evidence,
    }

"""Public projections of independently reconstructed closed study stages."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import Field, model_validator

from . import ppo_evidence as public
from .rl.v2.study.registry import actors

SCHEMA = "plant-public-ppo-evaluation-v1"
router = APIRouter(prefix="/api/v2/ppo", tags=["ppo-evidence"])


class Actor(public.StrictModel):
    id: str
    arm: Literal["H", "M24", "E-beta", "PPO"]
    algorithm: str | None
    optimizer_seed: int | None
    states: dict[str, Annotated[int, Field(ge=0, le=32)]]
    failures: dict[str, Annotated[int, Field(ge=0, le=32)]]
    execution_complete: int = Field(ge=0, le=32)
    hard_safety_passed: int = Field(ge=0, le=32)
    cycle_feasible: int = Field(ge=0, le=32)
    eligible_pairs: int | None = Field(ge=0, le=32)
    iqm_delta_eur: float | None
    cvar20_delta_eur: float | None
    frozen_champion: bool


class EvaluationExport(public.StrictModel):
    schema_version: Literal["plant-public-ppo-evaluation-v1"]
    source_commit: Annotated[str, Field(pattern=r"^[a-f0-9]{40}$")]
    training_report_sha256: public.Digest
    stage_report_sha256: public.Digest
    stage: Literal["development", "heldout"]
    campaign_status: Literal["completed", "aborted"]
    reconstructed_at_export: Literal[True]
    actors: list[Actor] = Field(min_length=18, max_length=18)
    gate_checks: dict[str, bool]
    gate_passed: bool
    result_claim: str | None
    sha256: public.Digest

    @model_validator(mode="after")
    def registry(self) -> EvaluationExport:
        for row, expected in zip(self.actors, actors(), strict=True):
            if any(
                getattr(row, k) != expected[k]
                for k in ("id", "arm", "algorithm", "optimizer_seed")
            ):
                raise ValueError("evaluation actor registry differs")
            if sum(row.states.values()) != 32:
                raise ValueError("evaluation must retain all 32 cells per actor")
            if row.iqm_delta_eur is not None or row.cvar20_delta_eur is not None:
                if (
                    row.arm != "PPO"
                    or row.eligible_pairs != 32
                    or self.campaign_status != "completed"
                ):
                    raise ValueError("aggregate value requires every eligible pair")
        if not self.gate_checks or self.gate_passed != all(self.gate_checks.values()):
            raise ValueError("evaluation gate differs from its checks")
        if self.result_claim is not None and (
            self.stage != "heldout"
            or not self.gate_passed
            or self.campaign_status != "completed"
        ):
            raise ValueError("result claim lacks held-out success")
        return self


def validate(payload: dict[str, Any]) -> dict[str, Any]:
    EvaluationExport.model_validate(payload)
    if (
        public.digest({k: v for k, v in payload.items() if k != "sha256"})
        != payload["sha256"]
    ):
        raise ValueError("evaluation export checksum differs")
    return payload


def _projection(
    report: dict[str, Any], training_report: dict[str, Any]
) -> dict[str, Any]:
    analysis = report["analysis"]
    rows = []
    for actor in actors():
        row = analysis["actors"][actor["id"]]
        summary = row.get("primary_summary")
        pairs = row.get("primary_pairs")
        rows.append(
            {
                **{
                    key: actor[key]
                    for key in ("id", "arm", "algorithm", "optimizer_seed")
                },
                "states": dict(Counter(row["cell_states"])),
                "failures": dict(
                    Counter(reason for reason in row["cell_failures"] if reason)
                ),
                **{
                    key: sum(
                        arm is not None and arm["classification"][key] is True
                        for arm in row["physical_arms"]
                    )
                    for key in (
                        "execution_complete",
                        "hard_safety_passed",
                        "cycle_feasible",
                    )
                },
                "eligible_pairs": (
                    None
                    if pairs is None
                    else sum(p["value_comparison_eligible"] for p in pairs)
                ),
                "iqm_delta_eur": (
                    None if summary is None else summary["statistics"]["iqm"]
                ),
                "cvar20_delta_eur": (
                    None if summary is None else summary["statistics"]["cvar20"]
                ),
                "frozen_champion": actor["arm"] == "PPO"
                and analysis["frozen_champions"][actor["algorithm"]]
                == actor["optimizer_seed"],
            }
        )
    requirements = analysis[
        (
            "development_opening_requirements"
            if report["stage"] == "development"
            else "heldout_success_requirements"
        )
    ]
    payload = {
        "schema_version": SCHEMA,
        "source_commit": training_report["source"]["authorization_commit"],
        "training_report_sha256": training_report["sha256"],
        "stage_report_sha256": report["sha256"],
        "stage": report["stage"],
        "campaign_status": report["campaign_status"],
        "reconstructed_at_export": True,
        "actors": rows,
        "gate_checks": requirements["checks"],
        "gate_passed": requirements["all_calculated_requirements_met"],
        "result_claim": report["result_claim"],
    }
    return validate({**payload, "sha256": public.digest(payload)})


def export_stage(training_root: Path, stage: str) -> dict[str, Any]:
    from .rl.v2.study.stage_verifier import verify_stage
    from .rl.v2.training.custody_io import publish_bytes

    verified = verify_stage(training_root, stage)
    payload = _projection(verified.report, verified.context.training.report)
    verified.recheck()
    raw = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > public.MAX_BYTES:
        raise ValueError("evaluation export exceeds its public byte cap")
    public.PUBLIC_ROOT.mkdir(parents=True, exist_ok=True)
    publish_bytes(public.PUBLIC_ROOT / f"{stage}.json", raw)
    return payload


@router.get("/evaluation")
def evaluation(request: Request) -> dict[str, Any]:
    from .rl.v2.study.archive import _read_stored
    from .rl.v2.training.custody_io import require_directory
    from .rl.v2.training.custody_json import decode

    if request.query_params:
        raise HTTPException(
            status_code=422, detail="Evaluation export takes no parameters"
        )
    try:
        training = public.read_export()
        result: dict[str, Any] = {}
        for stage in ("development", "heldout"):
            path = public.PUBLIC_ROOT / f"{stage}.json"
            if not path.exists() and not path.is_symlink():
                result[stage] = None
                continue
            require_directory(path.parent)
            payload = validate(decode(_read_stored(path, public.MAX_BYTES)))
            if (
                training is None
                or payload["stage"] != stage
                or any(
                    payload[key] != training[key]
                    for key in ("source_commit", "training_report_sha256")
                )
            ):
                raise ValueError("evaluation and training export identities differ")
            result[stage] = payload
        if result["heldout"] and (
            not result["development"] or not result["development"]["gate_passed"]
        ):
            raise ValueError("held-out export lacks admitted development gate")
        return result
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise HTTPException(
            status_code=503, detail="Evaluation export failed integrity checks"
        ) from exc

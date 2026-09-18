"""Read-only, bounded public projection of the completed adaptive PPO research."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

PUBLIC_PATH = Path(__file__).parent / "data" / "adaptive_learning.json"
MAX_BYTES = 4 * 1024 * 1024
SCHEMA = "plant-public-adaptive-learning-v1"
SEEDS = list(range(9700300, 9700308))
POLICY_IDS = (
    "A-mpc-residual-scalar",
    "B-mpc-residual-pi",
    "A-scalar-gamma0995",
    "B-scalar-gamma1000",
)
ACTOR_IDS = ("H", "OperatingMPC", "M24", "E-beta", "initial-mpc-residual", *POLICY_IDS)
COMPARISON_IDS = {
    *(
        (policy, actor)
        for policy in POLICY_IDS[:2]
        for actor in ("H", "OperatingMPC", "M24", "E-beta", "initial-mpc-residual")
    ),
    (POLICY_IDS[1], POLICY_IDS[0]),
    *((policy, actor) for policy in POLICY_IDS[2:] for actor in ACTOR_IDS[:7]),
    (POLICY_IDS[3], POLICY_IDS[2]),
}
RESOURCES = ("battery", "h2", "co2", "ch4")
router = APIRouter(prefix="/api/v2/ppo", tags=["ppo-evidence"])
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
GitHash = Annotated[str, Field(pattern=r"^[a-f0-9]{40}$")]
Text = Annotated[str, Field(min_length=1, max_length=2000)]
Identifier = Annotated[
    str, Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
]
Vector = Annotated[list[float], Field(min_length=4, max_length=4)]
Resource = Literal["battery", "h2", "co2", "ch4"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Artifact(StrictModel):
    id: Identifier
    sha256: Digest
    bytes: int = Field(gt=0, le=1_000_000_000)

    @model_validator(mode="after")
    def relative_id(self) -> Artifact:
        if ".." in self.id.split("/"):
            raise ValueError("artifact identifier cannot traverse directories")
        return self


class Rollout(StrictModel):
    index: int = Field(ge=1, le=8)
    transitions: int = Field(ge=8192, le=65536)
    reward_sum: float
    completed_episodes: int = Field(ge=0, le=1000)
    complete_cycles: int = Field(ge=0, le=1000)
    training_mean_reward: float | None
    explained_variance: float | None
    value_loss: float | None
    approx_kl: float | None
    clip_fraction: float | None
    mean_episode_cost: Vector | None
    multipliers: Vector | None


class Policy(StrictModel):
    id: Identifier
    label: Text
    campaign_id: Identifier
    algorithm: Identifier
    optimizer_seed: Literal[101]
    training_status: Literal["completed"]
    evaluation_status: Literal["evaluated"]
    operational_status: Literal["experimental"]
    transitions: int = Field(ge=16384, le=65536)
    minibatch_updates: int = Field(ge=640, le=2560)
    reward_gamma: float = Field(gt=0, le=1)
    gae_lambda: float
    reward_scale: float
    checkpoint: Artifact
    training_source: Artifact
    rollouts: list[Rollout] = Field(min_length=2, max_length=8)
    limitations: list[Text] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def endpoint(self) -> Policy:
        if self.id not in POLICY_IDS:
            raise ValueError("unknown evaluated policy")
        older = self.id in POLICY_IDS[:2]
        budget = 65536 if older else 16384
        if self.campaign_id != ("007" if older else "009"):
            raise ValueError("policy must retain its original training campaign")
        gamma = 1.0 if self.id == "B-scalar-gamma1000" else 0.995
        if (
            self.transitions != budget
            or len(self.rollouts) * 8192 != budget
            or self.minibatch_updates != budget // 8192 * 320
            or self.reward_gamma != gamma
            or self.gae_lambda != 0.95
            or self.reward_scale != 0.001
        ):
            raise ValueError("policy training identity differs from its fixed endpoint")
        for index, row in enumerate(self.rollouts, 1):
            if row.index != index or row.transitions != index * 8192:
                raise ValueError(
                    "rollout history must preserve every recorded endpoint"
                )
            if self.id != "B-mpc-residual-pi" and row.multipliers is not None:
                raise ValueError(
                    "scalar policy cannot display active constraint multipliers"
                )
        return self


class ResourceValues(StrictModel):
    battery: float = Field(ge=0)
    h2: float = Field(ge=0)
    co2: float = Field(ge=0)
    ch4: float = Field(ge=0)


class Delivery(StrictModel):
    ordinal: int = Field(ge=1, le=28)
    delivered_kg: float = Field(ge=0)
    requested_kg: float = Field(gt=0)


class DailyResources(StrictModel):
    cycle: int = Field(ge=1, le=7)
    deficits: ResourceValues


class MpcModes(StrictModel):
    scheduled: int = Field(ge=0, le=672)
    recovery: int = Field(ge=0, le=672)
    fallback: int = Field(ge=0, le=672)


class Case(StrictModel):
    seed: int
    status: Literal["completed", "stopped"]
    decisions: int = Field(ge=0, le=672)
    failures: list[Text] = Field(max_length=32)
    net_operating_value_eur: float | None
    delivered_kg: float | None = Field(ge=0)
    battery_throughput_kwh: float | None = Field(ge=0)
    hard_safety_passed: bool | None
    prefix_hard_safety_passed: bool
    all_daily_returns_passed: bool | None
    mpc_modes: MpcModes | None
    deliveries: list[Delivery] = Field(max_length=28)
    daily_resources: list[DailyResources] = Field(max_length=7)

    @model_validator(mode="after")
    def completeness(self) -> Case:
        for rows, key in (
            (self.deliveries, "ordinal"),
            (self.daily_resources, "cycle"),
        ):
            if [getattr(row, key) for row in rows] != list(range(1, len(rows) + 1)):
                raise ValueError("observations must retain a contiguous ordered prefix")
        metrics = (
            self.net_operating_value_eur,
            self.delivered_kg,
            self.battery_throughput_kwh,
            self.hard_safety_passed,
            self.all_daily_returns_passed,
        )
        if self.status == "completed":
            if (
                self.decisions != 672
                or self.failures
                or any(v is None for v in metrics)
                or len(self.deliveries) != 28
                or len(self.daily_resources) != 7
            ):
                raise ValueError("complete case lacks a full reconstructed week")
            if self.hard_safety_passed != self.prefix_hard_safety_passed:
                raise ValueError(
                    "complete-case safety cannot differ from its recorded prefix"
                )
            assert self.delivered_kg is not None
            _close(
                self.delivered_kg,
                sum(row.delivered_kg for row in self.deliveries),
            )
            _close(sum(row.requested_kg for row in self.deliveries), 6720.0)
            if self.all_daily_returns_passed and any(
                getattr(row.deficits, resource) > 0
                for row in self.daily_resources
                for resource in RESOURCES
            ):
                raise ValueError(
                    "resource deficits cannot be labelled as passed returns"
                )
        elif not self.failures or any(v is not None for v in metrics):
            raise ValueError(
                "stopped prefix must retain failures and no full-week metrics"
            )
        if (
            self.mpc_modes
            and sum(self.mpc_modes.model_dump().values()) != self.decisions
        ):
            raise ValueError("MPC modes differ from the number of decisions")
        return self


class Actor(StrictModel):
    id: Identifier
    label: Text
    kind: Literal["heuristic", "mpc", "ppo", "initial"]
    evidence_origin: Literal["reused_batch008", "fresh_batch009"]
    cases: list[Case] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def identities(self) -> Actor:
        _seeds(self.cases)
        expected_kind = (
            "ppo"
            if self.id in POLICY_IDS
            else (
                "initial"
                if self.id == "initial-mpc-residual"
                else "heuristic" if self.id == "H" else "mpc"
            )
        )
        if self.id not in ACTOR_IDS or self.kind != expected_kind:
            raise ValueError("actor identity differs")
        expected_origin = (
            "fresh_batch009" if self.id in POLICY_IDS[2:] else "reused_batch008"
        )
        if self.evidence_origin != expected_origin:
            raise ValueError("historical references must remain explicitly reused")
        uses_prior = (
            self.id == "OperatingMPC"
            or self.id == "initial-mpc-residual"
            or self.id in POLICY_IDS
        )
        if any((row.mpc_modes is not None) != uses_prior for row in self.cases):
            raise ValueError(
                "causal MPC-prior mode records are missing or misattributed"
            )
        if self.id in POLICY_IDS and any(
            row.status != "completed" for row in self.cases
        ):
            raise ValueError("evaluated policy lacks all eight completed cases")
        return self


class DeliveryRegression(StrictModel):
    ordinal: int = Field(ge=1, le=28)
    difference_kg: float = Field(lt=0)


class ResourceRegression(StrictModel):
    cycle: int = Field(ge=1, le=7)
    resource: Resource
    difference: float = Field(gt=0)


class PairCase(StrictModel):
    seed: int
    available: bool
    candidate_failures: list[Text] = Field(max_length=32)
    comparator_failures: list[Text] = Field(max_length=32)
    delta_net_operating_value_eur: float | None
    delta_delivered_kg: float | None
    delta_battery_throughput_kwh: float | None
    delivery_regressions: list[DeliveryRegression] = Field(max_length=28)
    resource_regressions: list[ResourceRegression] = Field(max_length=28)


class Comparison(StrictModel):
    candidate: Identifier
    comparator: Identifier
    evidence_batch: Literal["008", "009"]
    cases: list[PairCase] = Field(min_length=8, max_length=8)


class History(StrictModel):
    id: Identifier
    label: Text
    status: Literal["completed", "aborted", "failed", "incomplete"]
    summary: Text
    policy_ids: list[Identifier] = Field(max_length=4)
    source: Artifact
    limitations: list[Text] = Field(max_length=32)


class Source(StrictModel):
    git_head: GitHash
    git_tree: GitHash


class Provenance(StrictModel):
    exported_from_closed_evidence: Literal[True]
    independent_verification_passed: Literal[True]
    development_only: Literal[True]
    artifacts: list[Artifact] = Field(min_length=1, max_length=128)


class Task(StrictModel):
    hours: Literal[168]
    decisions: Literal[672]
    deliveries: Literal[28]
    requested_methane_kg: Literal[6720]
    development_seeds: list[int] = Field(min_length=8, max_length=8)


class RegisteredStudy(StrictModel):
    status: Literal["pending"]
    families: Literal[3]
    optimizer_seeds: Literal[5]
    policies: Literal[15]
    protected_seeds_opened: Literal[False]


def _close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError("projected value differs from its case records")


def _seeds(rows: list[Any]) -> None:
    if [row.seed for row in rows] != SEEDS:
        raise ValueError("all eight development seeds must be retained in order")


def _check_pair(pair: PairCase, candidate: Case, comparator: Case) -> None:
    available = candidate.status == comparator.status == "completed"
    if (
        pair.available != available
        or pair.candidate_failures != candidate.failures
        or pair.comparator_failures != comparator.failures
    ):
        raise ValueError("pair availability or failures differ from its actors")
    for name in ("net_operating_value_eur", "delivered_kg", "battery_throughput_kwh"):
        delta = getattr(pair, f"delta_{name}")
        if available:
            if delta is None:
                raise ValueError("complete pair is missing its value difference")
            _close(delta, getattr(candidate, name) - getattr(comparator, name))
        elif delta is not None:
            raise ValueError(
                "stopped comparison cannot show a full-week value difference"
            )
    deliveries = (
        {
            a.ordinal: a.delivered_kg - b.delivered_kg
            for a, b in zip(
                candidate.deliveries, comparator.deliveries, strict=available
            )
            if a.delivered_kg < b.delivered_kg
        }
        if available
        else {}
    )
    resources = (
        {
            (a.cycle, resource): getattr(a.deficits, resource)
            - getattr(b.deficits, resource)
            for a, b in zip(
                candidate.daily_resources, comparator.daily_resources, strict=available
            )
            for resource in RESOURCES
            if getattr(a.deficits, resource) > getattr(b.deficits, resource)
        }
        if available
        else {}
    )
    if len(pair.delivery_regressions) != len(deliveries) or {
        r.ordinal for r in pair.delivery_regressions
    } != set(deliveries):
        raise ValueError("individual delivery regressions were omitted or duplicated")
    if len(pair.resource_regressions) != len(resources) or {
        (r.cycle, r.resource) for r in pair.resource_regressions
    } != set(resources):
        raise ValueError("daily resource regressions were omitted or duplicated")
    for row in pair.delivery_regressions:
        _close(row.difference_kg, deliveries[row.ordinal])
    for resource_row in pair.resource_regressions:
        _close(
            resource_row.difference,
            resources[(resource_row.cycle, resource_row.resource)],
        )


class ResearchEvidence(StrictModel):
    schema_version: Literal["plant-public-adaptive-learning-v1"]
    source: Source
    provenance: Provenance
    task: Task
    policies: list[Policy] = Field(min_length=4, max_length=4)
    actors: list[Actor] = Field(min_length=9, max_length=9)
    comparisons: list[Comparison] = Field(min_length=26, max_length=26)
    history: list[History] = Field(min_length=9, max_length=10)
    registered_study: RegisteredStudy
    limitations: list[Text] = Field(min_length=1, max_length=32)
    sha256: Digest

    @model_validator(mode="after")
    def joins(self) -> ResearchEvidence:
        if {p.id for p in self.policies} != set(POLICY_IDS) or {
            a.id for a in self.actors
        } != set(ACTOR_IDS):
            raise ValueError("the complete policy and actor registry is required")
        if self.task.development_seeds != SEEDS:
            raise ValueError("task development seeds differ")
        actors = {a.id: a for a in self.actors}
        seen: set[tuple[str, str]] = set()
        for comparison in self.comparisons:
            identity = comparison.candidate, comparison.comparator
            if (
                identity in seen
                or comparison.candidate not in POLICY_IDS
                or comparison.comparator not in actors
                or comparison.candidate == comparison.comparator
            ):
                raise ValueError(
                    "comparison identity is missing, duplicated, or invalid"
                )
            seen.add(identity)
            expected_batch = "009" if comparison.candidate in POLICY_IDS[2:] else "008"
            if comparison.evidence_batch != expected_batch:
                raise ValueError("comparison source batch differs")
            _seeds(comparison.cases)
            for pair, a, b in zip(
                comparison.cases,
                actors[comparison.candidate].cases,
                actors[comparison.comparator].cases,
                strict=True,
            ):
                _check_pair(pair, a, b)
        if seen != COMPARISON_IDS:
            raise ValueError("all published policy comparisons must be retained")
        if len({r.id for r in self.history}) != len(self.history):
            raise ValueError("campaign history identities are duplicated")
        history_by_id = {row.id: row for row in self.history}
        required_history = {f"{index:03}" for index in range(1, 10)}
        if not required_history.issubset(history_by_id) or set(
            history_by_id
        ) - required_history - {"pilot"}:
            raise ValueError("all nine campaign records must be retained")
        for identifier, status in (
            ("007", "aborted"),
            ("008", "completed"),
            ("009", "completed"),
        ):
            if history_by_id[identifier].status != status:
                raise ValueError("historical campaign completion status differs")
        for history in self.history:
            if len(set(history.policy_ids)) != len(history.policy_ids) or not set(
                history.policy_ids
            ).issubset(POLICY_IDS):
                raise ValueError("campaign references an unknown or duplicate policy")
        if any(
            policy.campaign_id not in {row.id for row in self.history}
            for policy in self.policies
        ):
            raise ValueError("policy is missing its campaign history")
        ids = [a.id for a in self.provenance.artifacts]
        if len(ids) != len(set(ids)):
            raise ValueError("provenance artifact identities are duplicated")
        return self


def digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def validate(payload: dict[str, Any]) -> dict[str, Any]:
    # Reject lone surrogates before a response serializer can fail on UTF-8.
    json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    ResearchEvidence.model_validate(payload)
    if (
        digest({key: value for key, value in payload.items() if key != "sha256"})
        != payload["sha256"]
    ):
        raise ValueError("adaptive learning export checksum differs")
    return payload


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def read_export() -> dict[str, Any] | None:
    path = PUBLIC_PATH
    try:
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    try:
        try:
            handle = os.open(
                path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
            )
        except FileNotFoundError:
            return None
        try:
            before = os.fstat(handle)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_size <= 0
                or before.st_size > MAX_BYTES
            ):
                raise ValueError(
                    "public export must be a bounded nonempty regular file"
                )
            chunks = bytearray()
            while len(chunks) <= MAX_BYTES:
                chunk = os.read(handle, min(65536, MAX_BYTES + 1 - len(chunks)))
                if not chunk:
                    break
                chunks.extend(chunk)
            after = os.fstat(handle)
            named = os.stat(path.name, dir_fd=directory, follow_symlinks=False)

            def identity(s: os.stat_result) -> tuple[int, int, int, int, int]:
                return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

            if (
                len(chunks) != before.st_size
                or identity(before) != identity(after)
                or identity(after) != identity(named)
                or not stat.S_ISREG(named.st_mode)
            ):
                raise ValueError("public export changed while being read")
        finally:
            os.close(handle)
    finally:
        os.close(directory)
    payload = json.loads(chunks, object_pairs_hook=_object, parse_constant=_constant)
    if not isinstance(payload, dict):
        raise ValueError("public export must be an object")
    return validate(payload)


@router.get("/research")
async def research(request: Request) -> dict[str, Any]:
    if request.query_params:
        raise HTTPException(
            status_code=422, detail="Research export takes no parameters"
        )
    async for chunk in request.stream():
        if chunk:
            raise HTTPException(status_code=422, detail="Research export takes no body")
    try:
        evidence = read_export()
    except (ValueError, TypeError, KeyError, OSError, RecursionError) as exc:
        raise HTTPException(
            status_code=503, detail="Research export failed integrity checks"
        ) from exc
    return {
        "status": "pending" if evidence is None else "available",
        "evidence": evidence,
    }

"""Explicit stage limits and mechanical reservations, independent of outcomes.

Actual limits are registered in external stage authority before development,
then reused unchanged for held-out evaluation. This arithmetic neither measures
capacity nor creates authorization, directories, or a running process.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from ..training.custody_json import authenticate, canonical_bytes
from .archive import SCHEMA as ARCHIVE_SCHEMA
from .contract import PREFIX

ARMS = ("H", "M24", "E-beta", "PPO")
ROOT_CONTROL_LIMIT_BYTES = 8 * 1024**2
CELL_CONTROL_LIMIT_BYTES = 64 * 1024
ROOT_CONTROL_DOCUMENT_COUNT = 5  # registry, authority input/copy, opening, verification
CELL_CONTROL_DOCUMENT_COUNT = 2  # claim and terminal
SCHEMA = PREFIX + "execution-budget-v1"


def _positive(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive exact integer")
    return value


@dataclass(frozen=True)
class EvidenceLimits:
    arm_stored_bytes: int
    arm_decoded_bytes: int
    timing_stored_bytes: int
    timing_decoded_bytes: int

    def __post_init__(self) -> None:
        for label, value in asdict(self).items():
            _positive(value, label)

    def to_dict(self) -> dict[str, int]:
        self.__post_init__()
        return asdict(self)

    def arm(self) -> dict[str, int]:
        self.__post_init__()
        return {
            "stored_limit": self.arm_stored_bytes,
            "uncompressed_limit": self.arm_decoded_bytes,
        }

    def timing(self) -> dict[str, int]:
        self.__post_init__()
        return {
            "stored_limit": self.timing_stored_bytes,
            "uncompressed_limit": self.timing_decoded_bytes,
        }


def execution_contract() -> dict[str, Any]:
    return authenticate(
        {
            "schema": PREFIX + "execution-contract-v1",
            "archive_schema": ARCHIVE_SCHEMA,
            "episode_count": 32,
            "actor_count": 18,
            "cell_count": 576,
            "arm_order": list(ARMS),
            "arm_cell_counts": {"H": 32, "M24": 32, "E-beta": 32, "PPO": 480},
            "root_control_limit_bytes": ROOT_CONTROL_LIMIT_BYTES,
            "cell_control_limit_bytes": CELL_CONTROL_LIMIT_BYTES,
            "root_control_document_count": ROOT_CONTROL_DOCUMENT_COUNT,
            "cell_control_document_count": CELL_CONTROL_DOCUMENT_COUNT,
            "concurrent_children": 1,
            "attempts_per_cell": 1,
            "resets_per_executed_cell": 1,
            "same_limits_for_all_ppo_families_and_seeds": True,
            "limits_frozen_before_development_and_unchanged_in_heldout": True,
            "development_reserves_stage_count": 2,
            "heldout_reserves_stage_count": 1,
            "disk_reservation_includes": [
                "compressed_arm",
                "compressed_timing",
                "both_bounded_logs",
                "control_documents",
                "compressed_report",
                "filesystem_reserve",
            ],
            "timeout_sum_excludes": [
                "parent_admission",
                "parent_verification",
                "parent_publication",
            ],
            "capacity_measured": False,
            "execution_authorized": False,
        }
    )


@dataclass(frozen=True)
class ExecutionBudget:
    child_timeout_seconds: int
    log_limit_bytes: int
    arm_limits: tuple[EvidenceLimits, EvidenceLimits, EvidenceLimits, EvidenceLimits]
    report_stored_bytes: int
    report_decoded_bytes: int
    filesystem_reserve_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "child_timeout_seconds",
            "log_limit_bytes",
            "report_stored_bytes",
            "report_decoded_bytes",
            "filesystem_reserve_bytes",
        ):
            _positive(getattr(self, name), name)
        if type(self.arm_limits) is not tuple or len(self.arm_limits) != 4:
            raise ValueError("execution budget requires exact H/M24/E-beta/PPO limits")
        for limits in self.arm_limits:
            if type(limits) is not EvidenceLimits:
                raise ValueError("arm limits must be exact EvidenceLimits")
            limits.__post_init__()

    def for_arm(self, arm: str) -> EvidenceLimits:
        self.__post_init__()
        if arm not in ARMS:
            raise ValueError("unknown arm in execution budget")
        return self.arm_limits[ARMS.index(arm)]

    @property
    def stage_allocation_bound_bytes(self) -> int:
        self.__post_init__()
        counts = (32, 32, 32, 480)
        return (
            sum(
                count
                * (
                    limits.arm_stored_bytes
                    + limits.timing_stored_bytes
                    + 2 * self.log_limit_bytes
                )
                for count, limits in zip(counts, self.arm_limits, strict=True)
            )
            + ROOT_CONTROL_DOCUMENT_COUNT * ROOT_CONTROL_LIMIT_BYTES
            + 576 * CELL_CONTROL_DOCUMENT_COUNT * CELL_CONTROL_LIMIT_BYTES
            + self.report_stored_bytes
        )

    def opening_free_bytes(self, stage: str) -> int:
        if stage not in {"development", "heldout"}:
            raise ValueError("budget stage must be development or heldout")
        return (
            self.stage_allocation_bound_bytes * (2 if stage == "development" else 1)
            + self.filesystem_reserve_bytes
        )

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return authenticate(
            {
                "schema": SCHEMA,
                "execution_contract_sha256": execution_contract()["sha256"],
                "child_timeout_seconds": self.child_timeout_seconds,
                "log_limit_bytes": self.log_limit_bytes,
                "arm_limits": {
                    arm: limit.to_dict()
                    for arm, limit in zip(ARMS, self.arm_limits, strict=True)
                },
                "report_stored_bytes": self.report_stored_bytes,
                "report_decoded_bytes": self.report_decoded_bytes,
                "filesystem_reserve_bytes": self.filesystem_reserve_bytes,
                "stage_allocation_bound_bytes": self.stage_allocation_bound_bytes,
                "development_opening_free_bytes": self.opening_free_bytes(
                    "development"
                ),
                "heldout_opening_free_bytes": self.opening_free_bytes("heldout"),
                "all_child_timeouts_seconds": 576 * self.child_timeout_seconds,
                "capacity_measured": False,
                "execution_authorized": False,
            }
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExecutionBudget:
        profiles = raw.get("arm_limits")
        if not isinstance(profiles, dict) or set(profiles) != set(ARMS):
            raise ValueError("execution budget arm profiles differ")
        parsed = []
        for arm in ARMS:
            profile = profiles[arm]
            if not isinstance(profile, dict) or set(profile) != {
                "arm_stored_bytes",
                "arm_decoded_bytes",
                "timing_stored_bytes",
                "timing_decoded_bytes",
            }:
                raise ValueError("execution budget evidence-limit fields differ")
            parsed.append(EvidenceLimits(**profile))
        result = cls(
            child_timeout_seconds=_positive(
                raw.get("child_timeout_seconds"), "child timeout"
            ),
            log_limit_bytes=_positive(raw.get("log_limit_bytes"), "log limit"),
            arm_limits=(parsed[0], parsed[1], parsed[2], parsed[3]),
            report_stored_bytes=_positive(
                raw.get("report_stored_bytes"), "stored report limit"
            ),
            report_decoded_bytes=_positive(
                raw.get("report_decoded_bytes"), "decoded report limit"
            ),
            filesystem_reserve_bytes=_positive(
                raw.get("filesystem_reserve_bytes"), "filesystem reserve"
            ),
        )
        if canonical_bytes(result.to_dict()) != canonical_bytes(dict(raw)):
            raise ValueError(
                "execution budget differs from its complete mechanical reconstruction"
            )
        return result


def control_bytes(value: object, *, cell: bool) -> bytes:
    """Encode and enforce the control-document share used in the reservation."""
    if type(cell) is not bool:
        raise ValueError("control-document class must be exact boolean")
    raw = canonical_bytes(value)
    if len(raw) > (CELL_CONTROL_LIMIT_BYTES if cell else ROOT_CONTROL_LIMIT_BYTES):
        raise ValueError("study control document exceeds its registered byte limit")
    return raw

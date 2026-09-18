"""Transparent controller adapters for the common absolute-setpoint door."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, ClassVar

from plant_sim import Action, ElectrolyserCommitmentProfile, PlantConfig

from ...controllers.base import Constraints
from ...controllers.heuristic import HeuristicController
from .control import ControllerObservationV2
from .mpc_execution import public_observation_sha256
from .mpc_schema import domain_sha256, require_sha256
from .setpoint_command import (
    ProcessSetpointCommandV1,
    issue_process_setpoint_command,
)
from .setpoint_control import (
    PROCESS_SETPOINT_TOLERANCE_KW,
    ProcessSetpointTargetsV1,
)


LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID = "legacy-heuristic-setpoint-authority-v1"
LEGACY_HEURISTIC_SETPOINT_EVIDENCE_SCHEMA = (
    "legacy-heuristic-setpoint-decision-evidence-v1"
)
LEGACY_HEURISTIC_SETPOINT_EVIDENCE_DOMAIN = LEGACY_HEURISTIC_SETPOINT_EVIDENCE_SCHEMA


def _index(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _finite(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}")
    return number


def _targets_close(
    left: ProcessSetpointTargetsV1,
    right: ProcessSetpointTargetsV1,
) -> bool:
    return all(
        math.isclose(
            getattr(left, field),
            getattr(right, field),
            rel_tol=0.0,
            abs_tol=PROCESS_SETPOINT_TOLERANCE_KW,
        )
        for field in (
            "electrolyser_target_kw",
            "dac_target_kw",
            "methanator_target_kw",
            "battery_charge_ceiling_kw",
        )
    )


@dataclass(frozen=True, slots=True)
class LegacyHeuristicSetpointEvidenceV1:
    """Exact A0 boundary decision and its deterministic target conversion."""

    decision_index: int
    observation_sha256: str
    boundary_solar_kw: float
    source_action: Action
    source_notes: tuple[str, ...]
    source_invalid: bool
    requested_targets: ProcessSetpointTargetsV1
    issued_targets: ProcessSetpointTargetsV1
    battery_charge_limited: bool

    schema: ClassVar[str] = LEGACY_HEURISTIC_SETPOINT_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        _index("decision_index", self.decision_index)
        require_sha256("observation_sha256", self.observation_sha256)
        solar_kw = _finite(
            "boundary_solar_kw",
            self.boundary_solar_kw,
            minimum=0.0,
        )
        if (
            not isinstance(self.source_action, Action)
            or Action.sanitised(self.source_action) != self.source_action
        ):
            raise ValueError("source_action must be a sanitized Action")
        if not isinstance(self.source_notes, tuple) or any(
            not isinstance(note, str) or not note for note in self.source_notes
        ):
            raise ValueError("source_notes must be an immutable text tuple")
        if not isinstance(self.source_invalid, bool):
            raise TypeError("source_invalid must be a bool")
        if not isinstance(self.requested_targets, ProcessSetpointTargetsV1):
            raise TypeError("requested_targets must be ProcessSetpointTargetsV1")
        if not isinstance(self.issued_targets, ProcessSetpointTargetsV1):
            raise TypeError("issued_targets must be ProcessSetpointTargetsV1")
        if not isinstance(self.battery_charge_limited, bool):
            raise TypeError("battery_charge_limited must be a bool")

        action = self.source_action
        expected = ProcessSetpointTargetsV1(
            electrolyser_target_kw=(
                action.electrolyser_share * solar_kw + action.battery_support_kw
            ),
            dac_target_kw=action.dac_share * solar_kw,
            methanator_target_kw=action.methanator_share * solar_kw,
            battery_charge_ceiling_kw=action.battery_share * solar_kw,
        )
        if not _targets_close(self.requested_targets, expected):
            raise ValueError("requested targets do not reproduce the A0 action")
        if not all(
            math.isclose(
                getattr(self.issued_targets, field),
                getattr(self.requested_targets, field),
                rel_tol=0.0,
                abs_tol=PROCESS_SETPOINT_TOLERANCE_KW,
            )
            for field in (
                "electrolyser_target_kw",
                "dac_target_kw",
                "methanator_target_kw",
            )
        ):
            raise ValueError("adapter changed an A0 process target")
        requested_charge = self.requested_targets.battery_charge_ceiling_kw
        issued_charge = self.issued_targets.battery_charge_ceiling_kw
        if issued_charge > requested_charge + PROCESS_SETPOINT_TOLERANCE_KW:
            raise ValueError("adapter increased A0 battery-charge authority")
        was_limited = issued_charge < requested_charge - PROCESS_SETPOINT_TOLERANCE_KW
        if self.battery_charge_limited != was_limited:
            raise ValueError("battery_charge_limited disagrees with target conversion")

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "controller_id": LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID,
            "decision_index": int(self.decision_index),
            "observation_sha256": self.observation_sha256,
            "boundary_solar_kw": float(self.boundary_solar_kw),
            "source_decision": {
                "action": self.source_action.to_dict(),
                "notes": list(self.source_notes),
                "invalid": self.source_invalid,
            },
            "requested_targets": self.requested_targets.to_dict(),
            "issued_targets": self.issued_targets.to_dict(),
            "battery_charge_limited": self.battery_charge_limited,
            "conversion": (
                "process target = boundary solar * sanitized A0 share; "
                "electrolyser additionally includes requested support; battery "
                "charge is capped only by the active physical charge rating"
            ),
        }

    @property
    def sha256(self) -> str:
        return str(
            domain_sha256(
                LEGACY_HEURISTIC_SETPOINT_EVIDENCE_DOMAIN,
                self.body(),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class LegacyHeuristicSetpointDecisionV1:
    """A0 conversion evidence paired with the common command it issued."""

    evidence: LegacyHeuristicSetpointEvidenceV1
    command: ProcessSetpointCommandV1

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, LegacyHeuristicSetpointEvidenceV1):
            raise TypeError("evidence must be LegacyHeuristicSetpointEvidenceV1")
        if not isinstance(self.command, ProcessSetpointCommandV1):
            raise TypeError("command must be ProcessSetpointCommandV1")
        if self.command.decision_index != self.evidence.decision_index:
            raise ValueError("command decision index differs from A0 evidence")
        if self.command.observation_sha256 != self.evidence.observation_sha256:
            raise ValueError("command observation differs from A0 evidence")
        if self.command.issuer_id != LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID:
            raise ValueError("command issuer is not the A0 setpoint adapter")
        if self.command.source_schema != self.evidence.schema:
            raise ValueError("command source schema differs from A0 evidence")
        if self.command.source_sha256 != self.evidence.sha256:
            raise ValueError("command source digest differs from A0 evidence")
        if self.command.targets != self.evidence.issued_targets:
            raise ValueError("command targets differ from A0 evidence")

    @property
    def invalid(self) -> bool:
        return self.evidence.source_invalid

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": self.evidence.to_dict(),
            "command": self.command.to_dict(),
            "invalid": self.invalid,
        }


class LegacyHeuristicSetpointControllerV1:
    """A0 logic with boundary shares transparently converted to held targets."""

    id: ClassVar[str] = LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID
    label: ClassVar[str] = "Legacy heuristic (absolute-setpoint authority)"
    description: ClassVar[str] = (
        "The frozen forecast-blind heuristic evaluated once at the public boundary, "
        "then transparently converted from solar shares to absolute held targets."
    )
    uses_forecast: ClassVar[bool] = False
    information_set: ClassVar[str] = (
        "latest one-minute V2 boundary telemetry plus active safety constraints; "
        "no preceding-interval aggregate, delivery context, issued forecast, "
        "private weather truth, or future schedule"
    )

    def __init__(self) -> None:
        self._controller = HeuristicController()
        self._decision_count = 0

    def reset(self) -> None:
        self._controller.reset()
        self._decision_count = 0

    def compute_command(
        self,
        observation: ControllerObservationV2,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: ElectrolyserCommitmentProfile | None = None,
    ) -> LegacyHeuristicSetpointDecisionV1:
        """Issue one common command without adding information or policy logic."""

        del commitment_profile
        if not isinstance(observation, ControllerObservationV2):
            raise TypeError("observation must be ControllerObservationV2")
        if not isinstance(constraints, Constraints):
            raise TypeError("constraints must be Constraints")
        if not isinstance(config, PlantConfig):
            raise TypeError("config must be PlantConfig")

        source = self._controller.compute_action(
            observation.state.boundary,
            constraints=constraints,
            config=config,
        )
        action = Action.sanitised(source.action)
        solar_kw = max(0.0, observation.state.boundary.solar_available_kw)
        requested = ProcessSetpointTargetsV1(
            electrolyser_target_kw=(
                action.electrolyser_share * solar_kw + action.battery_support_kw
            ),
            dac_target_kw=action.dac_share * solar_kw,
            methanator_target_kw=action.methanator_share * solar_kw,
            battery_charge_ceiling_kw=action.battery_share * solar_kw,
        )
        charge_limit_kw = config.battery.max_charge_kw
        issued = ProcessSetpointTargetsV1(
            electrolyser_target_kw=requested.electrolyser_target_kw,
            dac_target_kw=requested.dac_target_kw,
            methanator_target_kw=requested.methanator_target_kw,
            battery_charge_ceiling_kw=min(
                requested.battery_charge_ceiling_kw,
                charge_limit_kw,
            ),
        )
        evidence = LegacyHeuristicSetpointEvidenceV1(
            decision_index=self._decision_count,
            observation_sha256=public_observation_sha256(observation),
            boundary_solar_kw=solar_kw,
            source_action=action,
            source_notes=tuple(source.notes),
            source_invalid=source.invalid or action != source.action,
            requested_targets=requested,
            issued_targets=issued,
            battery_charge_limited=(
                requested.battery_charge_ceiling_kw
                > charge_limit_kw + PROCESS_SETPOINT_TOLERANCE_KW
            ),
        )
        command = issue_process_setpoint_command(
            evidence.issued_targets,
            decision_index=evidence.decision_index,
            observation_sha256=evidence.observation_sha256,
            hold_duration_s=observation.decision_interval_s,
            issuer_id=self.id,
            source_schema=evidence.schema,
            source_sha256=evidence.sha256,
            constraints=constraints,
            config=config,
        )
        result = LegacyHeuristicSetpointDecisionV1(
            evidence=evidence,
            command=command,
        )
        self._decision_count += 1
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "uses_forecast": self.uses_forecast,
            "information_set": self.information_set,
            "source_controller": self._controller.to_dict(),
            "adapter": "boundary-share-to-held-absolute-target-v1",
            "behavioral_equivalence_claim": False,
        }


__all__ = [
    "LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID",
    "LEGACY_HEURISTIC_SETPOINT_EVIDENCE_DOMAIN",
    "LEGACY_HEURISTIC_SETPOINT_EVIDENCE_SCHEMA",
    "LegacyHeuristicSetpointControllerV1",
    "LegacyHeuristicSetpointDecisionV1",
    "LegacyHeuristicSetpointEvidenceV1",
]

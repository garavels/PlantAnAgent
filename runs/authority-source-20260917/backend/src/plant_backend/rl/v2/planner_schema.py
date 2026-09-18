"""Immutable evidence schema for the frozen feasible-planner protocol.

This module is intentionally pure.  It owns canonical serialization, domain-separated
identities, immutable detached records, and the exact task comparators.  It does not
step an environment, generate an action, search, solve, or perform runner I/O.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass, fields
from fractions import Fraction
from numbers import Integral, Real
from pathlib import PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, ClassVar, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
PLANNER_KIND = "best-known-feasible-future-information-v1"
PRIVILEGED_INFORMATION = "complete private episode truth and commitments"
OPTIMALITY_CLAIM = "none"
PROVENANCE_GAPS = ("clean start/end source and dependency-lock audit required",)

PROTOCOL_PATH = "docs/research/FEASIBLE_PLANNER_PROTOCOL.md"
PROTOCOL_PARENT_COMMIT = "2a3c99a73d6e3f071d9721f72691291f307a6125"
PROTOCOL_ANCHOR_COMMIT = "265d9ad4e430b3f1a68d7e3ce39ab0a60b26f000"
PROTOCOL_BLOB_SHA256 = (
    "50ce791a4c98e087c49dcb0eafe6210a081bd115cede1ba5f38610c8ff11f635"
)

ACTION_LIBRARY_SHA256 = (
    "805bc4a6203026783a0d2d6514d25c83188d9ca4476b7b9c6c7763b6718456f2"
)
A0_CONTROLLER_SHA256 = (
    "d5405d5661b847b67a2999159391fd89fe0543f3459627e7a3a8ab25865a93d5"
)
CLONE_ALLOWLIST_SHA256 = (
    "694c1b8259270bc6be719fc65e93af006fb01c2f10f6a6907734e309f49fcbba"
)
COMMITMENT_PROFILE_SHA256 = (
    "c6123ad4c30ef29445fa68e37d7f3d55bdeec05129d8535ab9dca2bc4715c4f9"
)
PLANT_CONFIG_SHA256 = "a0b374d24368c15402b2cfca2d1ce1ecc50e9c931fc2fef9fbd407c3ba95140a"
CONSTRAINTS_SHA256 = "ed16cc2355f037b870030e42f743c6aed7b9221da6b5b43ebdcfffc086e59161"
REWARD_WEIGHTS_SHA256 = (
    "1fee0632e10730014cdf5f4e7d883d8e01d12fd6c8a527b74b30c922cd256ee3"
)
EPISODE_CONFIG_SHA256_BY_RATE = (
    (35.0, "115612216fc1ad749e6505cc0ea567265222695243078690b19095b3d750b7d7"),
    (42.0, "82838f8f36af09ab09d705292b1ecbe203220fbd7d6ebe0790214912d6d07649"),
)
CHECKPOINT_CONTRACT_SHA256_BY_RATE = (
    (35.0, "4ef8f7c92c26d0ea5cd0b6a1b4dc7a0384c9776fe265838b081887f760b91a0a"),
    (42.0, "5605c2bf5e99bb1bbec584e29f23773d5233b5eadcdcd11f269bdf03d12adfb0"),
)

CANONICAL_PAYLOAD_DOMAINS = frozenset(
    {
        "planner-action-library-v1",
        "planner-settings-v1",
        "planner-a0-controller-v1",
        "planner-relaxation-result-v1",
        "planner-relaxation-witness-v1",
        "planner-relaxation-guidance-v1",
        "planner-request-trace-v1",
        "planner-source-trace-v1",
        "planner-action-record-trace-v1",
        "planner-state-v1",
        "planner-metrics-v1",
        "planner-terminal-truth-v1",
        "planner-terminal-telemetry-v1",
        "planner-delivery-audit-v1",
        "planner-outcome-v1",
        "planner-replay-binding-v1",
        "planner-deterministic-counters-v1",
        "planner-result-v1",
        "planner-source-snapshot-v1",
        "planner-runtime-v1",
        "planner-runner-artifact-v1",
    }
)

ACTION_SOURCE_PHASES = (
    "a0_capture",
    "beam",
    "coarse_polish",
    "fine_polish",
)
ACTION_SOURCE_PHASE_ORDER = MappingProxyType(
    {phase: index for index, phase in enumerate(ACTION_SOURCE_PHASES)}
)
MODE_PROTECTION_ORDER = (
    "off",
    "cold_starting",
    "running",
    "hot_standby",
    "hot_starting",
    "cooldown",
)
EVIDENCE_SCOPES = frozenset(
    {"privileged_offline_development", "noncanonical_unit_fixture"}
)
REPLAY_ROLES = frozenset({"a0_incumbent", "searched_finalist"})
FAILURE_PHASES = frozenset(
    {
        "input_validation",
        "relaxation_binding",
        "a0_generation",
        "a0_replay",
        "clone_preflight",
        "beam_search",
        "coarse_polish",
        "fine_polish",
        "finalist_replay",
        "bracket_validation",
        "result_serialization",
    }
)
FAILURE_CODES = frozenset(
    {
        "invalid_input",
        "noncanonical_episode",
        "fault_scope_violation",
        "relaxation_unavailable",
        "relaxation_binding_mismatch",
        "nonfinite_value",
        "manifest_mismatch",
        "checkpoint_mismatch",
        "a0_generation_error",
        "a0_replay_mismatch",
        "clone_error",
        "clone_contract_violation",
        "action_generation_error",
        "action_not_closed",
        "simulator_step_error",
        "premature_termination",
        "unexpected_truncation",
        "incomplete_horizon",
        "state_signature_error",
        "attempted_edge_limit",
        "live_environment_limit",
        "wall_time_limit",
        "rss_limit",
        "memory_error",
        "search_incomplete",
        "coordinate_error",
        "replay_mismatch",
        "incumbent_regression",
        "result_invariant_error",
        "serialization_error",
    }
)
RUNNER_FAILURE_PHASES = frozenset(
    {"start_source", "child_execution", "end_source", "artifact_serialization"}
)
RUNNER_FAILURE_CODES = frozenset(
    {
        "source_unavailable",
        "wall_time_limit",
        "child_exit",
        "child_protocol_error",
        "serialization_error",
    }
)
SOURCE_BOUNDARY_CLAIM = (
    "matching complete content identities were observed immediately before "
    "relaxation construction and immediately after core serialization; no claim is "
    "made about undetected change-and-restoration within the interval"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _require_string(name: str, value: object, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if nonempty and not value:
        raise ValueError(f"{name} must not be empty")
    return value


def require_sha256(name: str, value: object) -> str:
    """Return a validated lowercase hexadecimal SHA-256 string."""
    text = _require_string(name, value)
    if _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return text


def _require_int(name: str, value: object, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _require_float(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def canonical_json(value: object) -> str:
    """Serialize with the protocol's exact strict canonical-JSON rule."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def raw_canonical_sha256(value: object) -> str:
    """Hash strict canonical JSON without a domain prefix.

    Static V2 configuration and checkpoint contracts predate the planner's
    domain-separated evidence payloads.  Naming their raw hashing rule prevents
    callers from accidentally substituting :func:`canonical_payload` and
    producing a different identity.
    """

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_payload_preimage(domain: str, serialized: str) -> bytes:
    """Return the exact bytes hashed by every domain-separated payload."""
    _require_string("domain", domain)
    _require_string("canonical_json", serialized, nonempty=False)
    return (
        b"plantanagent:" + domain.encode("utf-8") + b"\0" + serialized.encode("utf-8")
    )


def _domain_digest(domain: str, serialized: str) -> str:
    return hashlib.sha256(canonical_payload_preimage(domain, serialized)).hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalPayloadV1:
    domain: str
    canonical_json: str
    sha256: str

    def __post_init__(self) -> None:
        domain = _require_string("domain", self.domain)
        if domain not in CANONICAL_PAYLOAD_DOMAINS:
            raise ValueError(f"unsupported canonical payload domain: {domain}")
        serialized = _require_string(
            "canonical_json", self.canonical_json, nonempty=False
        )
        try:
            value = json.loads(serialized)
            reproduced = canonical_json(value)
        except (UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                "canonical_json is not valid strict canonical JSON"
            ) from exc
        if reproduced != serialized:
            raise ValueError("canonical_json is not in canonical form")
        digest = require_sha256("sha256", self.sha256)
        if digest != _domain_digest(domain, serialized):
            raise ValueError("canonical payload SHA-256 does not match its preimage")

    def to_dict(self) -> dict[str, str]:
        return {
            "domain": self.domain,
            "canonical_json": self.canonical_json,
            "sha256": self.sha256,
        }

    def value(self) -> object:
        """Return a newly parsed, detached JSON value."""
        return json.loads(self.canonical_json)


def canonical_payload(domain: str, value: object) -> CanonicalPayloadV1:
    """Construct a validated domain-separated canonical payload."""
    serialized = canonical_json(value)
    return CanonicalPayloadV1(
        domain=domain,
        canonical_json=serialized,
        sha256=_domain_digest(domain, serialized),
    )


_ACTION_LIBRARY_VALUE = {
    "formula_version": "future-truth-router-binary64-v1",
    "schema_version": 1,
    "templates": [
        {
            "generator": "legacy-live-v1",
            "id": 0,
            "name": "legacy_live",
            "stack_fraction": None,
        },
        {
            "generator": "relaxation-projection-v1",
            "id": 1,
            "name": "relaxation_projection",
            "stack_fraction": None,
        },
        {
            "generator": "constant-idle-v1",
            "id": 2,
            "name": "idle",
            "stack_fraction": None,
        },
        {
            "generator": "constant-charge-v1",
            "id": 3,
            "name": "charge",
            "stack_fraction": None,
        },
        {
            "generator": "stack-only-router-v1",
            "id": 4,
            "name": "warm_hold",
            "stack_fraction": 0.02,
        },
        {
            "generator": "stack-only-router-v1",
            "id": 5,
            "name": "cold_start_hold",
            "stack_fraction": 0.2,
        },
        {
            "generator": "stack-only-router-v1",
            "id": 6,
            "name": "h2_stable",
            "stack_fraction": 0.4,
        },
        {
            "generator": "stack-only-router-v1",
            "id": 7,
            "name": "h2_full",
            "stack_fraction": 1.0,
        },
        {
            "generator": "dac-only-active-cap-router-v1",
            "id": 8,
            "name": "dac_full",
            "stack_fraction": None,
        },
        {
            "generator": "feedstocks-router-v1",
            "id": 9,
            "name": "feedstocks_stable",
            "stack_fraction": 0.4,
        },
        {
            "generator": "feedstocks-router-v1",
            "id": 10,
            "name": "feedstocks_full",
            "stack_fraction": 1.0,
        },
        {
            "generator": "methanator-only-active-cap-router-v1",
            "id": 11,
            "name": "methanate_full",
            "stack_fraction": None,
        },
        {
            "generator": "clipped-stoich-router-v1",
            "id": 12,
            "name": "clipped_stoich_40",
            "stack_fraction": 0.4,
        },
        {
            "generator": "clipped-stoich-router-v1",
            "id": 13,
            "name": "clipped_stoich_70",
            "stack_fraction": 0.7,
        },
        {
            "generator": "clipped-stoich-router-v1",
            "id": 14,
            "name": "clipped_stoich_100",
            "stack_fraction": 1.0,
        },
    ],
}

ACTION_LIBRARY = canonical_payload("planner-action-library-v1", _ACTION_LIBRARY_VALUE)
if ACTION_LIBRARY.sha256 != ACTION_LIBRARY_SHA256:  # pragma: no cover - import guard
    raise RuntimeError("frozen action-library payload identity is inconsistent")

_A0_CONTROLLER_VALUE = {
    "adapter": "legacy_v1_boundary_telemetry",
    "description": (
        "V1 operator heuristic adapted without behavior changes; it receives only the "
        "latest one-minute boundary telemetry and active safety constraints."
    ),
    "id": "legacy_heuristic",
    "information_set": (
        "latest one-minute V2 telemetry plus safety constraints; no preceding-interval "
        "aggregate, delivery context, or issued forecast"
    ),
    "label": "Legacy forecast-blind heuristic",
    "uses_forecast": False,
}
A0_CONTROLLER = canonical_payload("planner-a0-controller-v1", _A0_CONTROLLER_VALUE)
if A0_CONTROLLER.sha256 != A0_CONTROLLER_SHA256:  # pragma: no cover - import guard
    raise RuntimeError("frozen A0-controller payload identity is inconsistent")


def _require_payload(
    name: str, value: object, domain: str | Sequence[str]
) -> CanonicalPayloadV1:
    if not isinstance(value, CanonicalPayloadV1):
        raise TypeError(f"{name} must be a CanonicalPayloadV1")
    allowed = (domain,) if isinstance(domain, str) else tuple(domain)
    if value.domain not in allowed:
        raise ValueError(f"{name} must use domain {allowed!r}")
    # Reconstruct to protect against illicit object.__setattr__ mutation.
    CanonicalPayloadV1(value.domain, value.canonical_json, value.sha256)
    return value


def _compare(left: object, right: object) -> int:
    return int(left > right) - int(left < right)  # type: ignore[operator]


def _same_binary64(left: object, right: object) -> bool:
    if (
        isinstance(left, bool)
        or isinstance(right, bool)
        or not isinstance(left, (int, float))
        or not isinstance(right, (int, float))
    ):
        return False
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        return False
    return struct.pack(">d", left_value) == struct.pack(">d", right_value)


def _shares_are_sanitiser_fixed_point(shares: Sequence[float]) -> bool:
    """Reproduce ``Action.sanitised`` without coupling the pure schema to the plant."""

    sanitised = tuple(shares)
    total = sum(sanitised)
    if total > 1.0:
        sanitised = tuple(share / total for share in sanitised)
    return all(
        _same_binary64(original, checked)
        for original, checked in zip(shares, sanitised, strict=True)
    )


@dataclass(frozen=True, slots=True)
class ActionRecordV1:
    decision_index: int
    source_phase: str
    source_code: int
    source_binding_sha256: str
    controller_invalid: bool
    battery_share: float
    electrolyser_share: float
    dac_share: float
    methanator_share: float
    battery_support_kw: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decision_index",
            _require_int("decision_index", self.decision_index, minimum=0),
        )
        phase = _require_string("source_phase", self.source_phase)
        if phase not in ACTION_SOURCE_PHASE_ORDER:
            raise ValueError(f"unsupported source_phase: {phase}")
        code = _require_int("source_code", self.source_code, minimum=-1)
        if code > 14:
            raise ValueError("source_code must be -1 or an action-template ID 0..14")
        if (phase == "a0_capture") != (code == -1):
            raise ValueError("a0_capture and source_code=-1 must occur together")
        object.__setattr__(self, "source_code", code)
        require_sha256("source_binding_sha256", self.source_binding_sha256)
        _require_bool("controller_invalid", self.controller_invalid)
        if code >= 0 and self.controller_invalid:
            raise ValueError("planner template actions cannot be controller-invalid")
        shares = []
        for name in (
            "battery_share",
            "electrolyser_share",
            "dac_share",
            "methanator_share",
        ):
            value = _require_float(name, getattr(self, name), minimum=0.0, maximum=1.0)
            object.__setattr__(self, name, value)
            shares.append(value)
        ordered_total = ((shares[0] + shares[1]) + shares[2]) + shares[3]
        legacy_fixed_point = (
            code in (-1, 0)
            and self.source_binding_sha256 == A0_CONTROLLER_SHA256
            and _shares_are_sanitiser_fixed_point(shares)
        )
        if ordered_total > 1.0 and not legacy_fixed_point:
            raise ValueError(
                "action shares must have a binary64 sum no greater than one"
            )
        object.__setattr__(
            self,
            "battery_support_kw",
            _require_float("battery_support_kw", self.battery_support_kw, minimum=0.0),
        )

    @classmethod
    def from_action(
        cls,
        action: object,
        *,
        decision_index: int,
        source_phase: str,
        source_code: int,
        source_binding_sha256: str,
        controller_invalid: bool = False,
    ) -> ActionRecordV1:
        """Detach an Action-shaped object into a validated evidence record."""
        if isinstance(action, Mapping):
            getter = action.__getitem__
        else:
            getter = lambda name: getattr(action, name)
        try:
            values = {
                name: getter(name)
                for name in (
                    "battery_share",
                    "electrolyser_share",
                    "dac_share",
                    "methanator_share",
                    "battery_support_kw",
                )
            }
        except (AttributeError, KeyError, TypeError) as exc:
            raise TypeError(
                "action must expose exactly the five action values"
            ) from exc
        return cls(
            decision_index=decision_index,
            source_phase=source_phase,
            source_code=source_code,
            source_binding_sha256=source_binding_sha256,
            controller_invalid=controller_invalid,
            **values,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_index": self.decision_index,
            "source_phase": self.source_phase,
            "source_code": self.source_code,
            "source_binding_sha256": self.source_binding_sha256,
            "controller_invalid": self.controller_invalid,
            "battery_share": self.battery_share,
            "electrolyser_share": self.electrolyser_share,
            "dac_share": self.dac_share,
            "methanator_share": self.methanator_share,
            "battery_support_kw": self.battery_support_kw,
        }

    def request_dict(self) -> dict[str, object]:
        return {
            "decision_index": self.decision_index,
            "controller_invalid": self.controller_invalid,
            "battery_share": self.battery_share,
            "electrolyser_share": self.electrolyser_share,
            "dac_share": self.dac_share,
            "methanator_share": self.methanator_share,
            "battery_support_kw": self.battery_support_kw,
        }

    def source_dict(self) -> dict[str, object]:
        return {
            "decision_index": self.decision_index,
            "source_phase": self.source_phase,
            "source_code": self.source_code,
            "source_binding_sha256": self.source_binding_sha256,
        }

    @property
    def source_token(self) -> tuple[int, int]:
        return ACTION_SOURCE_PHASE_ORDER[self.source_phase], self.source_code


def _validated_action_records(
    values: Iterable[ActionRecordV1],
) -> tuple[ActionRecordV1, ...]:
    try:
        result = tuple(values)
    except TypeError as exc:
        raise TypeError("action_records must be iterable") from exc
    for index, record in enumerate(result):
        if not isinstance(record, ActionRecordV1):
            raise TypeError(f"action_records[{index}] must be an ActionRecordV1")
        if record.decision_index != index:
            raise ValueError("action record indices must be contiguous from zero")
        # Re-run validation in case frozen fields were illicitly mutated.
        ActionRecordV1(
            decision_index=record.decision_index,
            source_phase=record.source_phase,
            source_code=record.source_code,
            source_binding_sha256=record.source_binding_sha256,
            controller_invalid=record.controller_invalid,
            battery_share=record.battery_share,
            electrolyser_share=record.electrolyser_share,
            dac_share=record.dac_share,
            methanator_share=record.methanator_share,
            battery_support_kw=record.battery_support_kw,
        )
    return result


def request_trace_payload(records: Iterable[ActionRecordV1]) -> CanonicalPayloadV1:
    records_tuple = _validated_action_records(records)
    return canonical_payload(
        "planner-request-trace-v1",
        {
            "schema_version": 1,
            "records": [item.request_dict() for item in records_tuple],
        },
    )


def source_trace_payload(records: Iterable[ActionRecordV1]) -> CanonicalPayloadV1:
    records_tuple = _validated_action_records(records)
    return canonical_payload(
        "planner-source-trace-v1",
        {
            "schema_version": 1,
            "records": [item.source_dict() for item in records_tuple],
        },
    )


def action_record_trace_payload(
    records: Iterable[ActionRecordV1],
) -> CanonicalPayloadV1:
    records_tuple = _validated_action_records(records)
    return canonical_payload(
        "planner-action-record-trace-v1",
        {"schema_version": 1, "records": [item.to_dict() for item in records_tuple]},
    )


def compare_action_lineage(
    left: Sequence[ActionRecordV1], right: Sequence[ActionRecordV1]
) -> int:
    """Return +1 when *left* wins the searched-candidate lineage tie-break."""
    left_records = _validated_action_records(left)
    right_records = _validated_action_records(right)
    left_source = tuple(item.source_token for item in left_records)
    right_source = tuple(item.source_token for item in right_records)
    if left_source != right_source:
        return 1 if left_source < right_source else -1
    left_requests = request_trace_payload(left_records).canonical_json.encode("utf-8")
    right_requests = request_trace_payload(right_records).canonical_json.encode("utf-8")
    return (
        1
        if left_requests < right_requests
        else -1 if left_requests > right_requests else 0
    )


@dataclass(frozen=True, slots=True)
class FinalScoreV1:
    delivered_kg: float
    service_numerator: int
    service_denominator: int
    worst_fill: float
    terminal_ch4_equivalent_inventory_kg: float
    terminal_battery_energy_kwh: float
    forced_stops: int
    aborted_starts: int
    cold_starts: int
    curtailed_kwh: float
    battery_throughput_kwh: float

    def __post_init__(self) -> None:
        for name in (
            "delivered_kg",
            "terminal_ch4_equivalent_inventory_kg",
            "terminal_battery_energy_kwh",
            "curtailed_kwh",
            "battery_throughput_kwh",
        ):
            object.__setattr__(
                self,
                name,
                _require_float(name, getattr(self, name), minimum=0.0),
            )
        object.__setattr__(
            self,
            "worst_fill",
            _require_float("worst_fill", self.worst_fill, minimum=0.0, maximum=1.0),
        )
        numerator = _require_int("service_numerator", self.service_numerator, minimum=0)
        denominator = _require_int(
            "service_denominator", self.service_denominator, minimum=0
        )
        if denominator == 0 and numerator != 0:
            raise ValueError("zero settlements require zero commitments met")
        if denominator > 0 and numerator > denominator:
            raise ValueError("service numerator cannot exceed its denominator")
        object.__setattr__(self, "service_numerator", numerator)
        object.__setattr__(self, "service_denominator", denominator)
        for name in ("forced_stops", "aborted_starts", "cold_starts"):
            object.__setattr__(
                self, name, _require_int(name, getattr(self, name), minimum=0)
            )

    def to_dict(self) -> dict[str, int | float]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def _normalized_service(numerator: int, denominator: int) -> tuple[int, int]:
    if denominator == 0:
        return 1, 1
    return numerator, denominator


def compare_service_fractions(
    left_numerator: int,
    left_denominator: int,
    right_numerator: int,
    right_denominator: int,
) -> int:
    """Compare service fractions by exact integer cross-products."""
    ln = _require_int("left_numerator", left_numerator, minimum=0)
    ld = _require_int("left_denominator", left_denominator, minimum=0)
    rn = _require_int("right_numerator", right_numerator, minimum=0)
    rd = _require_int("right_denominator", right_denominator, minimum=0)
    if (ld == 0 and ln != 0) or (rd == 0 and rn != 0):
        raise ValueError("zero service denominators require zero numerators")
    if (ld > 0 and ln > ld) or (rd > 0 and rn > rd):
        raise ValueError("service numerator cannot exceed denominator")
    ln, ld = _normalized_service(ln, ld)
    rn, rd = _normalized_service(rn, rd)
    return _compare(ln * rd, rn * ld)


def compare_final_scores(left: FinalScoreV1, right: FinalScoreV1) -> int:
    """Return +1 iff *left* ranks higher, -1 iff lower, and 0 iff tied."""
    if not isinstance(left, FinalScoreV1) or not isinstance(right, FinalScoreV1):
        raise TypeError("final score operands must be FinalScoreV1 records")
    for name in (
        "delivered_kg",
        "worst_fill",
        "terminal_ch4_equivalent_inventory_kg",
        "terminal_battery_energy_kwh",
    ):
        result = _compare(getattr(left, name), getattr(right, name))
        if result:
            return result
        if name == "delivered_kg":
            result = compare_service_fractions(
                left.service_numerator,
                left.service_denominator,
                right.service_numerator,
                right.service_denominator,
            )
            if result:
                return result
    for name in (
        "forced_stops",
        "aborted_starts",
        "cold_starts",
        "curtailed_kwh",
        "battery_throughput_kwh",
    ):
        result = _compare(getattr(right, name), getattr(left, name))
        if result:
            return result
    return 0


@dataclass(frozen=True, slots=True)
class PlannerSettingsV1:
    schema_version: int
    planner_kind: str
    action_library: CanonicalPayloadV1
    beam_width: int
    macro_decisions: int
    max_attempted_edges: int
    mode_protection_order: tuple[str, ...]
    intermediate_rank_version: str
    final_score_version: str
    coarse_block_decisions: int
    coarse_sweeps: int
    coarse_origins: tuple[str, ...]
    fine_block_decisions: int
    fine_sweeps: int
    random_policy: str
    clone_contract_version: str
    clone_allowlist_sha256: str
    state_signature_version: str
    wall_limit_ns: int
    rss_limit_bytes: int
    clock_method: str
    rss_method: str
    max_live_environments: int
    expected_decisions: int
    expected_physics_steps: int
    protocol_path: str
    protocol_parent_commit: str
    protocol_anchor_commit: str
    protocol_blob_sha256: str
    registered_rates: tuple[float, ...]
    registered_seeds: tuple[int, ...]
    plant_config_sha256: str
    constraints_sha256: str
    reward_weights_sha256: str
    episode_config_sha256: str
    checkpoint_contract_digest: str
    episode_config_sha256_by_rate: tuple[tuple[float, str], ...]
    checkpoint_contract_sha256_by_rate: tuple[tuple[float, str], ...]
    commitment_profile_sha256: str
    a0_controller: CanonicalPayloadV1
    relaxation_model_sha256: str
    relaxation_result_sha256: str
    relaxation_witness_sha256: str
    relaxation_guidance_sha256: str
    settings_sha256: str = ""

    _FROZEN_VALUES: ClassVar[dict[str, object]] = {
        "schema_version": 1,
        "planner_kind": PLANNER_KIND,
        "beam_width": 24,
        "macro_decisions": 4,
        "max_attempted_edges": 25_575,
        "mode_protection_order": MODE_PROTECTION_ORDER,
        "intermediate_rank_version": "deadline-readiness-lexicographic-v1",
        "final_score_version": "delivery-service-endowment-lexicographic-v1",
        "coarse_block_decisions": 24,
        "coarse_sweeps": 1,
        "coarse_origins": ("a0_incumbent", "beam_winner"),
        "fine_block_decisions": 4,
        "fine_sweeps": 1,
        "random_policy": "none",
        "clone_contract_version": "deepcopy-immutable-memo-v1",
        "clone_allowlist_sha256": CLONE_ALLOWLIST_SHA256,
        "state_signature_version": "planner-step-action-future-v1",
        "wall_limit_ns": 5_400_000_000_000,
        "rss_limit_bytes": 4_294_967_296,
        "clock_method": "time.perf_counter_ns",
        "rss_method": "resource.RUSAGE_SELF.ru_maxrss:darwin-bytes:other-kib",
        "max_live_environments": 57,
        "protocol_path": PROTOCOL_PATH,
        "protocol_parent_commit": PROTOCOL_PARENT_COMMIT,
        "protocol_anchor_commit": PROTOCOL_ANCHOR_COMMIT,
        "protocol_blob_sha256": PROTOCOL_BLOB_SHA256,
        "registered_rates": (35.0, 42.0),
        "registered_seeds": (0, 1, 2),
        "episode_config_sha256_by_rate": EPISODE_CONFIG_SHA256_BY_RATE,
        "checkpoint_contract_sha256_by_rate": CHECKPOINT_CONTRACT_SHA256_BY_RATE,
        "commitment_profile_sha256": COMMITMENT_PROFILE_SHA256,
    }

    def __post_init__(self) -> None:
        for name in ("mode_protection_order", "coarse_origins"):
            try:
                value = tuple(getattr(self, name))
            except TypeError as exc:
                raise TypeError(f"{name} must be iterable") from exc
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "registered_rates",
            tuple(
                _require_float("registered rate", item)
                for item in self.registered_rates
            ),
        )
        object.__setattr__(
            self,
            "registered_seeds",
            tuple(
                _require_int("registered seed", item, minimum=0)
                for item in self.registered_seeds
            ),
        )
        for name in (
            "episode_config_sha256_by_rate",
            "checkpoint_contract_sha256_by_rate",
        ):
            try:
                raw_pairs = tuple(getattr(self, name))
            except TypeError as exc:
                raise TypeError(f"{name} must be iterable") from exc
            pairs: list[tuple[float, str]] = []
            for index, pair in enumerate(raw_pairs):
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise TypeError(f"{name}[{index}] must be a rate/digest pair")
                pairs.append(
                    (
                        _require_float(f"{name}[{index}].rate", pair[0]),
                        require_sha256(f"{name}[{index}].sha256", pair[1]),
                    )
                )
            object.__setattr__(self, name, tuple(pairs))
        for name, expected in self._FROZEN_VALUES.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} does not match the frozen protocol")
        for name in (
            "schema_version",
            "coarse_block_decisions",
            "coarse_sweeps",
            "fine_block_decisions",
            "fine_sweeps",
            "expected_decisions",
            "expected_physics_steps",
            "beam_width",
            "macro_decisions",
            "max_attempted_edges",
            "wall_limit_ns",
            "rss_limit_bytes",
            "max_live_environments",
        ):
            object.__setattr__(
                self, name, _require_int(name, getattr(self, name), minimum=1)
            )
        _require_payload(
            "action_library", self.action_library, "planner-action-library-v1"
        )
        if self.action_library != ACTION_LIBRARY:
            raise ValueError("action_library is not the frozen registered library")
        _require_payload(
            "a0_controller", self.a0_controller, "planner-a0-controller-v1"
        )
        if self.a0_controller != A0_CONTROLLER:
            raise ValueError("a0_controller is not the frozen registered descriptor")
        for name in (
            "plant_config_sha256",
            "constraints_sha256",
            "reward_weights_sha256",
            "episode_config_sha256",
            "checkpoint_contract_digest",
            "relaxation_model_sha256",
            "relaxation_result_sha256",
            "relaxation_witness_sha256",
            "relaxation_guidance_sha256",
        ):
            require_sha256(name, getattr(self, name))
        expected = _settings_digest(self)
        if self.settings_sha256 == "":
            object.__setattr__(self, "settings_sha256", expected)
        elif require_sha256("settings_sha256", self.settings_sha256) != expected:
            raise ValueError("settings_sha256 does not match the frozen preimage")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, CanonicalPayloadV1):
                result[field.name] = value.to_dict()
            elif field.name in {
                "mode_protection_order",
                "coarse_origins",
                "registered_rates",
                "registered_seeds",
            }:
                result[field.name] = list(value)
            elif field.name in {
                "episode_config_sha256_by_rate",
                "checkpoint_contract_sha256_by_rate",
            }:
                result[field.name] = [list(pair) for pair in value]
            else:
                result[field.name] = value
        return result


def revalidate_planner_settings(settings: PlannerSettingsV1) -> PlannerSettingsV1:
    """Return a fully reconstructed, detached planner-settings record.

    ``PlannerSettingsV1`` is frozen for ordinary callers, but Python permits
    low-level mutation through ``object.__setattr__``. Runtime admission therefore
    reconstructs the complete record, including new nested canonical-payload
    objects, before trusting any limit or identity.
    """

    if type(settings) is not PlannerSettingsV1:
        raise TypeError("settings must be an exact PlannerSettingsV1")
    values = {item.name: getattr(settings, item.name) for item in fields(settings)}
    for name in ("action_library", "a0_controller"):
        payload = values[name]
        if type(payload) is not CanonicalPayloadV1:
            raise TypeError(f"settings.{name} must be an exact CanonicalPayloadV1")
        values[name] = CanonicalPayloadV1(
            domain=payload.domain,
            canonical_json=payload.canonical_json,
            sha256=payload.sha256,
        )
    return PlannerSettingsV1(**values)


def require_planner_admission(
    settings: PlannerSettingsV1,
    *,
    seed: int,
    evidence_scope: str,
    canonical_episode_verified: bool,
) -> PlannerSettingsV1:
    """Fail closed before a planner may construct or step an environment.

    Registered 72-hour development runs and bounded noncanonical unit fixtures are
    intentionally disjoint. This early gate mirrors the result-schema checks so a
    long or unregistered rollout cannot be observed first and rejected only during
    result serialization.
    """

    checked = revalidate_planner_settings(settings)
    checked_seed = _require_int("seed", seed, minimum=0)
    scope = _require_string("evidence_scope", evidence_scope)
    canonical = _require_bool("canonical_episode_verified", canonical_episode_verified)
    if scope not in EVIDENCE_SCOPES:
        raise ValueError(f"unsupported evidence_scope: {scope}")
    if scope == "noncanonical_unit_fixture":
        if canonical:
            raise ValueError(
                "noncanonical unit fixtures cannot claim canonical verification"
            )
        if checked.expected_decisions > 8:
            raise ValueError(
                "noncanonical unit fixtures are limited to eight decisions"
            )
        return checked

    if not canonical:
        raise ValueError("privileged development requires canonical verification")
    if checked_seed not in checked.registered_seeds:
        raise ValueError("privileged development seed is not registered")
    for name, expected in (
        ("plant_config_sha256", PLANT_CONFIG_SHA256),
        ("constraints_sha256", CONSTRAINTS_SHA256),
        ("reward_weights_sha256", REWARD_WEIGHTS_SHA256),
    ):
        if getattr(checked, name) != expected:
            raise ValueError(f"privileged development {name} is noncanonical")
    if not any(
        checked.episode_config_sha256 == episode_pair[1]
        and checked.checkpoint_contract_digest == checkpoint_pair[1]
        and _same_binary64(episode_pair[0], checkpoint_pair[0])
        for episode_pair, checkpoint_pair in zip(
            checked.episode_config_sha256_by_rate,
            checked.checkpoint_contract_sha256_by_rate,
            strict=True,
        )
    ):
        raise ValueError(
            "privileged development episode/checkpoint pair is unregistered"
        )
    if checked.expected_decisions != 288 or checked.expected_physics_steps != 4320:
        raise ValueError("privileged development requires registered horizon/cadence")
    return checked


def planner_settings_preimage(settings: PlannerSettingsV1) -> dict[str, object]:
    """Return the exact detached object hashed as ``planner-settings-v1``."""
    if not isinstance(settings, PlannerSettingsV1):
        raise TypeError("settings must be PlannerSettingsV1")
    result = settings.to_dict()
    result.pop("settings_sha256")
    for name in ("action_library", "a0_controller"):
        payload = getattr(settings, name)
        result[name] = {"domain": payload.domain, "sha256": payload.sha256}
    return result


def _settings_digest(settings: PlannerSettingsV1) -> str:
    value = planner_settings_preimage(settings)
    return canonical_payload("planner-settings-v1", value).sha256


def build_planner_settings(
    *,
    expected_decisions: int,
    expected_physics_steps: int,
    plant_config_sha256: str,
    constraints_sha256: str,
    reward_weights_sha256: str,
    episode_config_sha256: str,
    checkpoint_contract_digest: str,
    relaxation_model_sha256: str,
    relaxation_result_sha256: str,
    relaxation_witness_sha256: str,
    relaxation_guidance_sha256: str,
) -> PlannerSettingsV1:
    """Build settings with every structural protocol value filled exactly."""
    return PlannerSettingsV1(
        schema_version=1,
        planner_kind=PLANNER_KIND,
        action_library=ACTION_LIBRARY,
        beam_width=24,
        macro_decisions=4,
        max_attempted_edges=25_575,
        mode_protection_order=MODE_PROTECTION_ORDER,
        intermediate_rank_version="deadline-readiness-lexicographic-v1",
        final_score_version="delivery-service-endowment-lexicographic-v1",
        coarse_block_decisions=24,
        coarse_sweeps=1,
        coarse_origins=("a0_incumbent", "beam_winner"),
        fine_block_decisions=4,
        fine_sweeps=1,
        random_policy="none",
        clone_contract_version="deepcopy-immutable-memo-v1",
        clone_allowlist_sha256=CLONE_ALLOWLIST_SHA256,
        state_signature_version="planner-step-action-future-v1",
        wall_limit_ns=5_400_000_000_000,
        rss_limit_bytes=4_294_967_296,
        clock_method="time.perf_counter_ns",
        rss_method="resource.RUSAGE_SELF.ru_maxrss:darwin-bytes:other-kib",
        max_live_environments=57,
        expected_decisions=expected_decisions,
        expected_physics_steps=expected_physics_steps,
        protocol_path=PROTOCOL_PATH,
        protocol_parent_commit=PROTOCOL_PARENT_COMMIT,
        protocol_anchor_commit=PROTOCOL_ANCHOR_COMMIT,
        protocol_blob_sha256=PROTOCOL_BLOB_SHA256,
        registered_rates=(35.0, 42.0),
        registered_seeds=(0, 1, 2),
        plant_config_sha256=plant_config_sha256,
        constraints_sha256=constraints_sha256,
        reward_weights_sha256=reward_weights_sha256,
        episode_config_sha256=episode_config_sha256,
        checkpoint_contract_digest=checkpoint_contract_digest,
        episode_config_sha256_by_rate=EPISODE_CONFIG_SHA256_BY_RATE,
        checkpoint_contract_sha256_by_rate=CHECKPOINT_CONTRACT_SHA256_BY_RATE,
        commitment_profile_sha256=COMMITMENT_PROFILE_SHA256,
        a0_controller=A0_CONTROLLER,
        relaxation_model_sha256=relaxation_model_sha256,
        relaxation_result_sha256=relaxation_result_sha256,
        relaxation_witness_sha256=relaxation_witness_sha256,
        relaxation_guidance_sha256=relaxation_guidance_sha256,
    )


def sanitize_failure_message(value: object) -> str:
    """Apply the shared frozen core/runner failure-message sanitizer."""
    message = " ".join(str(value).split())
    message = "".join(
        character if 32 <= ord(character) <= 126 else "?" for character in message
    )
    message = message[:240]
    return message or "unspecified failure"


@dataclass(frozen=True, slots=True)
class PlannerFailureV1:
    phase: str
    code: str
    message: str
    completed_macro_depths: int
    attempted_edges: int
    completed_edges: int
    coordinate_trials_completed: int
    elapsed_ns: int
    observed_resource_value: int | None

    def __post_init__(self) -> None:
        if self.phase not in FAILURE_PHASES:
            raise ValueError(f"unsupported planner failure phase: {self.phase}")
        if self.code not in FAILURE_CODES:
            raise ValueError(f"unsupported planner failure code: {self.code}")
        message = _require_string("message", self.message)
        if message != sanitize_failure_message(message):
            raise ValueError("message must already be in frozen sanitized form")
        for name in (
            "completed_macro_depths",
            "attempted_edges",
            "completed_edges",
            "coordinate_trials_completed",
            "elapsed_ns",
        ):
            object.__setattr__(
                self, name, _require_int(name, getattr(self, name), minimum=0)
            )
        if self.completed_edges > self.attempted_edges:
            raise ValueError("completed_edges cannot exceed attempted_edges")
        if self.observed_resource_value is not None:
            object.__setattr__(
                self,
                "observed_resource_value",
                _require_int(
                    "observed_resource_value",
                    self.observed_resource_value,
                    minimum=0,
                ),
            )

    @classmethod
    def from_raw_message(cls, *, message: object, **values: object) -> PlannerFailureV1:
        return cls(message=sanitize_failure_message(message), **values)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class PlannerRunnerFailureV1:
    phase: str
    code: str
    message: str
    elapsed_ns: int
    child_exit_code: int | None

    def __post_init__(self) -> None:
        if self.phase not in RUNNER_FAILURE_PHASES:
            raise ValueError(f"unsupported runner failure phase: {self.phase}")
        if self.code not in RUNNER_FAILURE_CODES:
            raise ValueError(f"unsupported runner failure code: {self.code}")
        message = _require_string("message", self.message)
        if message != sanitize_failure_message(message):
            raise ValueError("message must already be in frozen sanitized form")
        object.__setattr__(
            self, "elapsed_ns", _require_int("elapsed_ns", self.elapsed_ns, minimum=0)
        )
        if self.child_exit_code is not None:
            object.__setattr__(
                self,
                "child_exit_code",
                _require_int("child_exit_code", self.child_exit_code),
            )
        allowed_codes = {
            "start_source": {"source_unavailable"},
            "child_execution": {
                "wall_time_limit",
                "child_exit",
                "child_protocol_error",
            },
            "end_source": {"source_unavailable"},
            "artifact_serialization": {"serialization_error"},
        }
        if self.code not in allowed_codes[self.phase]:
            raise ValueError("runner failure code is incompatible with its phase")
        if self.code == "child_exit" and self.child_exit_code is None:
            raise ValueError("child_exit failure requires child_exit_code")
        if (
            self.code
            in {
                "source_unavailable",
                "wall_time_limit",
                "serialization_error",
            }
            and self.child_exit_code is not None
        ):
            raise ValueError("runner failure code cannot carry child_exit_code")

    @classmethod
    def from_raw_message(
        cls, *, message: object, **values: object
    ) -> PlannerRunnerFailureV1:
        return cls(message=sanitize_failure_message(message), **values)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


DETERMINISTIC_COUNTER_FIELDS = (
    "completed_macro_depths",
    "attempted_edges",
    "completed_edges",
    "generated_candidates",
    "a0_cursor_validations",
    "a0_injections",
    "exact_duplicates",
    "frontier_retained_total",
    "peak_frontier",
    "coordinate_trials_attempted",
    "coordinate_trials_completed",
    "coordinate_trials_accepted",
    "a0_generation_decisions",
    "a0_evidence_replay_decisions",
    "fresh_finalist_replay_decisions",
    "max_live_environments_observed",
    "termination_reason",
)


@dataclass(frozen=True, slots=True)
class SearchDiagnosticsV1:
    completed_macro_depths: int
    attempted_edges: int
    completed_edges: int
    generated_candidates: int
    a0_cursor_validations: int
    a0_injections: int
    exact_duplicates: int
    frontier_retained_total: int
    peak_frontier: int
    coordinate_trials_attempted: int
    coordinate_trials_completed: int
    coordinate_trials_accepted: int
    a0_generation_decisions: int
    a0_evidence_replay_decisions: int
    fresh_finalist_replay_decisions: int
    max_live_environments_observed: int
    termination_reason: str
    deterministic_counters: CanonicalPayloadV1
    elapsed_ns: int
    peak_rss_bytes: int

    def __post_init__(self) -> None:
        for name in DETERMINISTIC_COUNTER_FIELDS[:-1] + (
            "elapsed_ns",
            "peak_rss_bytes",
        ):
            object.__setattr__(
                self, name, _require_int(name, getattr(self, name), minimum=0)
            )
        reason = _require_string("termination_reason", self.termination_reason)
        if reason != "completed" and reason not in FAILURE_CODES:
            raise ValueError(
                "termination_reason must be completed or a planner failure code"
            )
        if self.completed_edges > self.attempted_edges:
            raise ValueError("completed_edges cannot exceed attempted_edges")
        if self.generated_candidates != self.completed_edges:
            raise ValueError("generated_candidates must equal completed_edges")
        if self.coordinate_trials_completed > self.coordinate_trials_attempted:
            raise ValueError(
                "completed coordinate trials cannot exceed attempted trials"
            )
        if self.coordinate_trials_accepted > self.coordinate_trials_completed:
            raise ValueError(
                "accepted coordinate trials cannot exceed completed trials"
            )
        payload = _require_payload(
            "deterministic_counters",
            self.deterministic_counters,
            "planner-deterministic-counters-v1",
        )
        if payload != deterministic_counters_payload(self):
            raise ValueError("deterministic_counters does not match diagnostics")

    def to_dict(self) -> dict[str, object]:
        result = {field.name: getattr(self, field.name) for field in fields(self)}
        result["deterministic_counters"] = self.deterministic_counters.to_dict()
        return result


def deterministic_counters_payload(
    diagnostics: SearchDiagnosticsV1 | Mapping[str, object],
) -> CanonicalPayloadV1:
    if isinstance(diagnostics, SearchDiagnosticsV1):
        source: Mapping[str, object] = diagnostics.to_dict()
    elif isinstance(diagnostics, Mapping):
        source = diagnostics
    else:
        raise TypeError("diagnostics must be SearchDiagnosticsV1 or a mapping")
    value: dict[str, object] = {"schema_version": 1}
    try:
        for name in DETERMINISTIC_COUNTER_FIELDS:
            value[name] = source[name]
    except KeyError as exc:
        raise ValueError(f"missing deterministic counter: {exc.args[0]}") from exc
    return canonical_payload("planner-deterministic-counters-v1", value)


def build_search_diagnostics(**values: object) -> SearchDiagnosticsV1:
    """Build diagnostics and its redundant deterministic counter payload."""
    payload = deterministic_counters_payload(values)
    return SearchDiagnosticsV1(deterministic_counters=payload, **values)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ReplayEvidenceV1:
    role: str
    manifest_sha256: str
    exogenous_episode_sha256: str
    checkpoint_contract_digest: str
    action_records: tuple[ActionRecordV1, ...]
    request_trace: CanonicalPayloadV1
    source_trace: CanonicalPayloadV1
    action_record_trace: CanonicalPayloadV1
    decision_count: int
    physics_steps: int
    terminated: bool
    truncated: bool
    battery_throughput_kwh: float
    metrics: CanonicalPayloadV1
    terminal_truth: CanonicalPayloadV1
    terminal_telemetry: CanonicalPayloadV1
    delivery_audit: CanonicalPayloadV1
    generation_outcome: CanonicalPayloadV1
    replay_outcome: CanonicalPayloadV1
    replay_binding: CanonicalPayloadV1
    final_score: FinalScoreV1
    replay_elapsed_ns: int

    def __post_init__(self) -> None:
        if self.role not in REPLAY_ROLES:
            raise ValueError(f"unsupported replay role: {self.role}")
        for name in (
            "manifest_sha256",
            "exogenous_episode_sha256",
            "checkpoint_contract_digest",
        ):
            require_sha256(name, getattr(self, name))
        records = _validated_action_records(self.action_records)
        object.__setattr__(self, "action_records", records)
        if self.role == "a0_incumbent" and any(
            item.source_code != -1 for item in records
        ):
            raise ValueError("a0 incumbent records must all retain A0 lineage")
        expected_traces = (
            (
                "request_trace",
                "planner-request-trace-v1",
                request_trace_payload(records),
            ),
            ("source_trace", "planner-source-trace-v1", source_trace_payload(records)),
            (
                "action_record_trace",
                "planner-action-record-trace-v1",
                action_record_trace_payload(records),
            ),
        )
        for name, domain, expected in expected_traces:
            payload = _require_payload(name, getattr(self, name), domain)
            if payload != expected:
                raise ValueError(f"{name} does not match action_records")
        object.__setattr__(
            self,
            "decision_count",
            _require_int("decision_count", self.decision_count, minimum=0),
        )
        object.__setattr__(
            self,
            "physics_steps",
            _require_int("physics_steps", self.physics_steps, minimum=0),
        )
        if self.decision_count != len(records):
            raise ValueError("decision_count must equal the action-record length")
        _require_bool("terminated", self.terminated)
        _require_bool("truncated", self.truncated)
        if not self.terminated or self.truncated:
            raise ValueError(
                "complete replay evidence must terminate without truncation"
            )
        object.__setattr__(
            self,
            "battery_throughput_kwh",
            _require_float(
                "battery_throughput_kwh", self.battery_throughput_kwh, minimum=0.0
            ),
        )
        domain_fields = (
            ("metrics", "planner-metrics-v1"),
            ("terminal_truth", "planner-terminal-truth-v1"),
            ("terminal_telemetry", "planner-terminal-telemetry-v1"),
            ("delivery_audit", "planner-delivery-audit-v1"),
            ("generation_outcome", "planner-outcome-v1"),
            ("replay_outcome", "planner-outcome-v1"),
            ("replay_binding", "planner-replay-binding-v1"),
        )
        for name, domain in domain_fields:
            _require_payload(name, getattr(self, name), domain)
        if self.generation_outcome.canonical_json != self.replay_outcome.canonical_json:
            raise ValueError("generation and replay outcome bytes must match exactly")
        if not isinstance(self.final_score, FinalScoreV1):
            raise TypeError("final_score must be FinalScoreV1")
        object.__setattr__(
            self,
            "replay_elapsed_ns",
            _require_int("replay_elapsed_ns", self.replay_elapsed_ns, minimum=0),
        )
        self._validate_outcome()
        self._validate_binding()
        self._validate_score_sources()

    def _validate_outcome(self) -> None:
        expected = {
            "schema_version": 1,
            "manifest_sha256": self.manifest_sha256,
            "exogenous_episode_sha256": self.exogenous_episode_sha256,
            "checkpoint_contract_digest": self.checkpoint_contract_digest,
            "request_trace_sha256": self.request_trace.sha256,
            "metrics_sha256": self.metrics.sha256,
            "terminal_truth_sha256": self.terminal_truth.sha256,
            "terminal_telemetry_sha256": self.terminal_telemetry.sha256,
            "delivery_audit_sha256": self.delivery_audit.sha256,
            "battery_throughput_kwh": self.battery_throughput_kwh,
            "decision_count": self.decision_count,
            "physics_steps": self.physics_steps,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }
        if self.replay_outcome != canonical_payload("planner-outcome-v1", expected):
            raise ValueError("outcome payload does not match replay evidence")

    def _validate_binding(self) -> None:
        value = self.replay_binding.value()
        if not isinstance(value, dict):
            raise ValueError("replay_binding must contain an object")
        required = {
            "schema_version",
            "role",
            "settings_sha256",
            "relaxation_guidance_sha256",
            "manifest_sha256",
            "exogenous_episode_sha256",
            "checkpoint_contract_digest",
            "request_trace_sha256",
            "source_trace_sha256",
            "action_record_trace_sha256",
            "outcome_sha256",
        }
        if set(value) != required:
            raise ValueError("replay_binding has an incorrect field set")
        settings_sha256 = require_sha256(
            "replay_binding.settings_sha256", value["settings_sha256"]
        )
        relaxation_guidance_sha256 = require_sha256(
            "replay_binding.relaxation_guidance_sha256",
            value["relaxation_guidance_sha256"],
        )
        expected = {
            "schema_version": 1,
            "role": self.role,
            "settings_sha256": settings_sha256,
            "relaxation_guidance_sha256": relaxation_guidance_sha256,
            "manifest_sha256": self.manifest_sha256,
            "exogenous_episode_sha256": self.exogenous_episode_sha256,
            "checkpoint_contract_digest": self.checkpoint_contract_digest,
            "request_trace_sha256": self.request_trace.sha256,
            "source_trace_sha256": self.source_trace.sha256,
            "action_record_trace_sha256": self.action_record_trace.sha256,
            "outcome_sha256": self.replay_outcome.sha256,
        }
        if self.replay_binding.canonical_json != canonical_json(expected):
            raise ValueError("replay_binding canonical bytes do not match evidence")

    def _validate_score_sources(self) -> None:
        metrics = self.metrics.value()
        truth = self.terminal_truth.value()
        telemetry = self.terminal_telemetry.value()
        audit = self.delivery_audit.value()
        if (
            not isinstance(metrics, dict)
            or not isinstance(truth, dict)
            or not isinstance(telemetry, dict)
            or not isinstance(audit, dict)
        ):
            raise ValueError(
                "metrics, terminal truth, and delivery audit must be objects"
            )
        # Lazy imports keep this schema module pure while binding the two repository
        # serializers whose complete field sets the protocol names explicitly.
        from plant_sim import PlantState

        from .metrics import EpisodeMetricsV2

        if set(metrics) != set(EpisodeMetricsV2().to_dict()):
            raise ValueError(
                "metrics payload is not a complete EpisodeMetricsV2.to_dict()"
            )
        if set(truth) != set(PlantState().to_dict()):
            raise ValueError(
                "terminal truth payload is not a complete PlantState.to_dict()"
            )
        if set(telemetry) != set(PlantState().to_dict()):
            raise ValueError(
                "terminal telemetry payload is not a complete PlantState.to_dict()"
            )
        if set(audit) != {
            "frozen",
            "schedule",
            "settlements",
            "current",
            "cumulative_requested_kg",
            "cumulative_delivered_kg",
            "cumulative_unmet_kg",
            "pending_count",
            "next_pending",
        }:
            raise ValueError("delivery audit payload has an incorrect field set")
        float_score_fields = {
            "delivered_kg": "delivered_kg",
            "worst_fill": "worst_commitment_fill_rate",
            "terminal_battery_energy_kwh": "terminal_battery_energy_kwh",
            "curtailed_kwh": "curtailed_kwh",
        }
        for score_name, metric_name in float_score_fields.items():
            if metric_name not in metrics or not _same_binary64(
                metrics[metric_name], getattr(self.final_score, score_name)
            ):
                raise ValueError(f"final_score.{score_name} does not match metrics")
        integer_score_fields = {
            "service_numerator": "commitments_met",
            "service_denominator": "commitments_settled",
            "forced_stops": "electrolyser_forced_stops",
            "aborted_starts": "electrolyser_aborted_starts",
            "cold_starts": "electrolyser_cold_starts",
        }
        for score_name, metric_name in integer_score_fields.items():
            metric_value = _require_int(
                f"metrics.{metric_name}", metrics.get(metric_name), minimum=0
            )
            if metric_value != getattr(self.final_score, score_name):
                raise ValueError(f"final_score.{score_name} does not match metrics")
        if not _same_binary64(
            self.final_score.battery_throughput_kwh,
            self.battery_throughput_kwh,
        ):
            raise ValueError("final-score throughput does not match replay throughput")
        truth_metrics = {
            "h2_level_kg": "terminal_h2_kg",
            "co2_level_kg": "terminal_co2_kg",
            "ch4_level_kg": "terminal_ch4_kg",
            "battery_soc": "terminal_battery_soc",
            "battery_energy_kwh": "terminal_battery_energy_kwh",
        }
        for truth_name, metric_name in truth_metrics.items():
            if (
                truth_name not in truth
                or metric_name not in metrics
                or not _same_binary64(truth[truth_name], metrics[metric_name])
            ):
                raise ValueError(f"terminal truth {truth_name} does not match metrics")
        if not _same_binary64(
            audit.get("cumulative_delivered_kg"),
            self.final_score.delivered_kg,
        ):
            raise ValueError("delivery audit does not match delivered score")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "action_records":
                result[field.name] = [item.to_dict() for item in value]
            elif isinstance(value, CanonicalPayloadV1):
                result[field.name] = value.to_dict()
            elif isinstance(value, FinalScoreV1):
                result[field.name] = value.to_dict()
            else:
                result[field.name] = value
        return result


def outcome_payload(
    *,
    manifest_sha256: str,
    exogenous_episode_sha256: str,
    checkpoint_contract_digest: str,
    request_trace: CanonicalPayloadV1,
    metrics: CanonicalPayloadV1,
    terminal_truth: CanonicalPayloadV1,
    terminal_telemetry: CanonicalPayloadV1,
    delivery_audit: CanonicalPayloadV1,
    battery_throughput_kwh: float,
    decision_count: int,
    physics_steps: int,
    terminated: bool,
    truncated: bool,
) -> CanonicalPayloadV1:
    """Build the exact deterministic terminal-outcome fingerprint payload."""
    for name, value in (
        ("manifest_sha256", manifest_sha256),
        ("exogenous_episode_sha256", exogenous_episode_sha256),
        ("checkpoint_contract_digest", checkpoint_contract_digest),
    ):
        require_sha256(name, value)
    for name, payload_item, domain in (
        ("request_trace", request_trace, "planner-request-trace-v1"),
        ("metrics", metrics, "planner-metrics-v1"),
        ("terminal_truth", terminal_truth, "planner-terminal-truth-v1"),
        ("terminal_telemetry", terminal_telemetry, "planner-terminal-telemetry-v1"),
        ("delivery_audit", delivery_audit, "planner-delivery-audit-v1"),
    ):
        _require_payload(name, payload_item, domain)
    outcome_value = {
        "schema_version": 1,
        "manifest_sha256": manifest_sha256,
        "exogenous_episode_sha256": exogenous_episode_sha256,
        "checkpoint_contract_digest": checkpoint_contract_digest,
        "request_trace_sha256": request_trace.sha256,
        "metrics_sha256": metrics.sha256,
        "terminal_truth_sha256": terminal_truth.sha256,
        "terminal_telemetry_sha256": terminal_telemetry.sha256,
        "delivery_audit_sha256": delivery_audit.sha256,
        "battery_throughput_kwh": _require_float(
            "battery_throughput_kwh", battery_throughput_kwh, minimum=0.0
        ),
        "decision_count": _require_int("decision_count", decision_count, minimum=0),
        "physics_steps": _require_int("physics_steps", physics_steps, minimum=0),
        "terminated": _require_bool("terminated", terminated),
        "truncated": _require_bool("truncated", truncated),
    }
    return canonical_payload("planner-outcome-v1", outcome_value)


def replay_binding_payload(
    *,
    role: str,
    settings_sha256: str,
    relaxation_guidance_sha256: str,
    manifest_sha256: str,
    exogenous_episode_sha256: str,
    checkpoint_contract_digest: str,
    request_trace: CanonicalPayloadV1,
    source_trace: CanonicalPayloadV1,
    action_record_trace: CanonicalPayloadV1,
    outcome: CanonicalPayloadV1,
) -> CanonicalPayloadV1:
    """Build the exact replay lineage/binding payload."""
    if role not in REPLAY_ROLES:
        raise ValueError(f"unsupported replay role: {role}")
    for name, digest in (
        ("settings_sha256", settings_sha256),
        ("relaxation_guidance_sha256", relaxation_guidance_sha256),
        ("manifest_sha256", manifest_sha256),
        ("exogenous_episode_sha256", exogenous_episode_sha256),
        ("checkpoint_contract_digest", checkpoint_contract_digest),
    ):
        require_sha256(name, digest)
    for name, payload, domain in (
        ("request_trace", request_trace, "planner-request-trace-v1"),
        ("source_trace", source_trace, "planner-source-trace-v1"),
        (
            "action_record_trace",
            action_record_trace,
            "planner-action-record-trace-v1",
        ),
        ("outcome", outcome, "planner-outcome-v1"),
    ):
        _require_payload(name, payload, domain)
    return canonical_payload(
        "planner-replay-binding-v1",
        {
            "schema_version": 1,
            "role": role,
            "settings_sha256": settings_sha256,
            "relaxation_guidance_sha256": relaxation_guidance_sha256,
            "manifest_sha256": manifest_sha256,
            "exogenous_episode_sha256": exogenous_episode_sha256,
            "checkpoint_contract_digest": checkpoint_contract_digest,
            "request_trace_sha256": request_trace.sha256,
            "source_trace_sha256": source_trace.sha256,
            "action_record_trace_sha256": action_record_trace.sha256,
            "outcome_sha256": outcome.sha256,
        },
    )


@dataclass(frozen=True, slots=True)
class IncumbentComparisonV1:
    selected_role: str
    scores_equal: bool
    actions_equal_to_a0: bool
    a0_delivered_kg: float
    selected_delivered_kg: float
    gain_kg: float
    upper_bound_kg: float
    bracket_status: str
    residual_gap_kg: float | None
    fraction_closed: float | None

    def __post_init__(self) -> None:
        if self.selected_role not in REPLAY_ROLES:
            raise ValueError(f"unsupported selected_role: {self.selected_role}")
        _require_bool("scores_equal", self.scores_equal)
        _require_bool("actions_equal_to_a0", self.actions_equal_to_a0)
        for name in (
            "a0_delivered_kg",
            "selected_delivered_kg",
            "upper_bound_kg",
        ):
            object.__setattr__(
                self, name, _require_float(name, getattr(self, name), minimum=0.0)
            )
        object.__setattr__(self, "gain_kg", _require_float("gain_kg", self.gain_kg))
        expected_gain = self.selected_delivered_kg - self.a0_delivered_kg
        if not _same_binary64(self.gain_kg, expected_gain):
            raise ValueError("gain_kg does not match delivered values")
        if self.bracket_status not in {"valid", "bound_bridge_violation"}:
            raise ValueError("unsupported bracket_status")
        bridge = (
            self.a0_delivered_kg > self.upper_bound_kg
            or self.selected_delivered_kg > self.upper_bound_kg
        )
        if bridge != (self.bracket_status == "bound_bridge_violation"):
            raise ValueError(
                "bracket_status does not match the direct bound comparison"
            )
        if bridge:
            if self.residual_gap_kg is not None or self.fraction_closed is not None:
                raise ValueError(
                    "bridge violations must suppress residual and fraction"
                )
        else:
            expected_gap = self.upper_bound_kg - self.selected_delivered_kg
            if self.residual_gap_kg is None:
                raise ValueError("a valid bracket requires residual_gap_kg")
            gap = _require_float("residual_gap_kg", self.residual_gap_kg, minimum=0.0)
            object.__setattr__(self, "residual_gap_kg", gap)
            if not _same_binary64(gap, expected_gap):
                raise ValueError("residual_gap_kg does not match the bracket")
            if self.fraction_closed is not None:
                object.__setattr__(
                    self,
                    "fraction_closed",
                    _require_float("fraction_closed", self.fraction_closed),
                )
                denominator = self.upper_bound_kg - self.a0_delivered_kg
                if denominator <= 0.0 or not _same_binary64(
                    self.fraction_closed, self.gain_kg / denominator
                ):
                    raise ValueError("fraction_closed does not match the valid bracket")
        if self.selected_role == "a0_incumbent" and not (
            self.scores_equal
            and self.actions_equal_to_a0
            and _same_binary64(self.a0_delivered_kg, self.selected_delivered_kg)
        ):
            raise ValueError("selection of A0 must compare A0 with itself")

    def to_dict(self) -> dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def build_incumbent_comparison(
    *,
    selected_role: str,
    a0_score: FinalScoreV1,
    selected_score: FinalScoreV1,
    a0_request_trace: CanonicalPayloadV1,
    selected_request_trace: CanonicalPayloadV1,
    upper_bound_kg: float,
    optimality_tolerance_kg: float,
) -> IncumbentComparisonV1:
    """Evaluate the exact score/action equality and Part-I bracket formulas."""
    if selected_role not in REPLAY_ROLES:
        raise ValueError(f"unsupported selected_role: {selected_role}")
    if not isinstance(a0_score, FinalScoreV1) or not isinstance(
        selected_score, FinalScoreV1
    ):
        raise TypeError("scores must be FinalScoreV1")
    _require_payload("a0_request_trace", a0_request_trace, "planner-request-trace-v1")
    _require_payload(
        "selected_request_trace", selected_request_trace, "planner-request-trace-v1"
    )
    upper = _require_float("upper_bound_kg", upper_bound_kg, minimum=0.0)
    tolerance = _require_float(
        "optimality_tolerance_kg", optimality_tolerance_kg, minimum=0.0
    )
    a0_delivered = a0_score.delivered_kg
    selected_delivered = selected_score.delivered_kg
    gain = selected_delivered - a0_delivered
    bridge = a0_delivered > upper or selected_delivered > upper
    residual: float | None
    fraction: float | None
    if bridge:
        residual = None
        fraction = None
    else:
        residual = upper - selected_delivered
        denominator = upper - a0_delivered
        fraction = None if denominator <= tolerance else gain / denominator
    return IncumbentComparisonV1(
        selected_role=selected_role,
        scores_equal=compare_final_scores(a0_score, selected_score) == 0,
        actions_equal_to_a0=(
            a0_request_trace.canonical_json == selected_request_trace.canonical_json
        ),
        a0_delivered_kg=a0_delivered,
        selected_delivered_kg=selected_delivered,
        gain_kg=gain,
        upper_bound_kg=upper,
        bracket_status="bound_bridge_violation" if bridge else "valid",
        residual_gap_kg=residual,
        fraction_closed=fraction,
    )


def _relaxation_upper_bound_and_tolerance(
    payload: CanonicalPayloadV1,
) -> tuple[float, float]:
    value = payload.value()
    if not isinstance(value, dict):
        raise ValueError("relaxation_result must contain an object")
    if value.get("dual_certified_delivery_upper_bound_kg") is None:
        raise ValueError(
            "relaxation_result must contain a non-null dual-certified bound"
        )
    solve_settings = value.get("solve_settings")
    if (
        not isinstance(solve_settings, dict)
        or solve_settings.get("optimality_tolerance_kg") is None
    ):
        raise ValueError("relaxation_result must contain an optimality tolerance")
    return (
        _require_float(
            "dual_certified_delivery_upper_bound_kg",
            value["dual_certified_delivery_upper_bound_kg"],
            minimum=0.0,
        ),
        _require_float(
            "optimality_tolerance_kg",
            solve_settings["optimality_tolerance_kg"],
            minimum=0.0,
        ),
    )


def _validate_action_bindings(
    replay: ReplayEvidenceV1,
    settings: PlannerSettingsV1,
) -> None:
    for record in replay.action_records:
        if record.source_code in {-1, 0}:
            expected = settings.a0_controller.sha256
        elif record.source_code == 1:
            expected = settings.relaxation_guidance_sha256
        else:
            expected = settings.action_library.sha256
        if record.source_binding_sha256 != expected:
            raise ValueError("action source binding does not match planner settings")


_RELAXATION_RESULT_FIELDS = {
    "schema_version",
    "success",
    "status",
    "failure_code",
    "message",
    "solver",
    "solver_version",
    "solve_settings",
    "model_sha256",
    "checkpoint_contract_digest",
    "exogenous_episode_sha256",
    "canonical_episode_verified",
    "evidence_provenance_status",
    "preserve_initial_endowment",
    "is_task_upper_bound",
    "requested_kg",
    "numerical_primal_delivery_kg",
    "dual_certified_delivery_upper_bound_kg",
    "relaxed_fill_ceiling",
    "primal_dual_gap_kg",
    "numerical_objective",
    "iterations",
    "marginal_projection_within_tolerance",
    "certificate",
    "witness",
    "privileged_information",
    "certificate_scope",
    "simulator_embedding_status",
}
_RELAXATION_GUIDANCE_FIELDS = {
    "schema_version",
    "model_sha256",
    "result_sha256",
    "witness_sha256",
    "status",
    "canonical_episode_verified",
    "preserve_initial_endowment",
    "is_task_upper_bound",
    "solver",
    "solver_version",
    "solve_settings",
    "checkpoint_contract_digest",
    "exogenous_episode_sha256",
    "solar_truth_sha256",
}
_RELAXATION_SOLVER = "scipy.optimize.linprog(method='highs')"
_SOLVE_SETTINGS_FIELDS = {
    "feasibility_tolerance",
    "physical_tolerance",
    "optimality_tolerance_kg",
    "marginal_sign_tolerance",
    "solver_options",
    "highs_runtime_version",
    "provenance_gaps",
}
_WITNESS_FIELDS = {
    "normalized_variables",
    "solar_to_battery_kw",
    "solar_to_electrolyser_kw",
    "solar_to_dac_kw",
    "solar_to_methanator_kw",
    "battery_to_electrolyser_kw",
    "curtailed_solar_kw",
    "h2_produced_kg",
    "co2_captured_kg",
    "ch4_produced_kg",
    "battery_energy_kwh",
    "h2_inventory_kg",
    "co2_inventory_kg",
    "ch4_inventory_kg",
    "deliveries",
    "delivered_kg",
    "residuals",
}
_RESIDUAL_FIELDS = {
    "max_equality_normalized",
    "max_inequality_violation_normalized",
    "max_box_violation_normalized",
    "max_solar_balance_kw",
    "max_battery_balance_kwh",
    "max_h2_balance_kg",
    "max_co2_balance_kg",
    "max_ch4_balance_kg",
    "max_headroom_violation_kg",
    "max_yield_violation_kg",
    "max_delivery_bound_violation_kg",
    "max_physical_box_violation",
    "max_component_power_violation_kw",
    "max_physical_violation",
}
_CERTIFICATE_FIELDS = {
    "minimization_lower_bound_exact",
    "uncapped_delivery_upper_bound_exact",
    "requested_kg_exact",
    "certified_delivery_upper_bound_exact",
    "dual_certified_delivery_upper_bound_kg",
    "positive_inequality_marginal_count",
    "max_positive_inequality_marginal",
    "arithmetic",
    "float_conversion",
    "certificate_scope",
}
_UPPER_BOUND_PRIVILEGED_INFORMATION = (
    "offline perfect-information research bound; not a deployable controller"
)
_CERTIFICATE_SCOPE = "formal certificate for the encoded finite-box linear program only"
_SIMULATOR_EMBEDDING_STATUS = (
    "relaxation embedding is numerically validated within declared tolerances; "
    "not bit-exact certified"
)
_LOCK_PROVENANCE_GAP = (
    "git source revision and dependency-lock digest must be attached by the "
    "evaluation runner"
)
_HIGHS_PROVENANCE_GAP = (
    "linked HiGHS runtime version is not exposed by this SciPy build"
)


def _fraction(name: str, value: object) -> Fraction:
    text = _require_string(name, value)
    if re.fullmatch(r"-?[0-9]+/[1-9][0-9]*", text) is None:
        raise ValueError(f"{name} must be a canonical integer ratio")
    numerator_text, denominator_text = text.split("/", 1)
    result = Fraction(int(numerator_text), int(denominator_text))
    if f"{result.numerator}/{result.denominator}" != text:
        raise ValueError(f"{name} must be reduced with a positive denominator")
    return result


def _next_float_up(value: Fraction) -> float:
    rounded = float(value)
    if Fraction.from_float(rounded) < value:
        rounded = math.nextafter(rounded, math.inf)
    return rounded


def _finite_json_vector(
    name: str,
    value: object,
    *,
    size: int,
    minimum: float | None = None,
    maximum: float | None = None,
) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} must be an array of length {size}")
    return [
        _require_float(f"{name}[{index}]", item, minimum=minimum, maximum=maximum)
        for index, item in enumerate(value)
    ]


def _validate_registered_relaxation_witness(
    witness: dict[str, object],
    *,
    physics_steps: int,
    physical_tolerance: float,
    feasibility_tolerance: float,
) -> tuple[float, float]:
    if set(witness) != _WITNESS_FIELDS:
        raise ValueError("registered relaxation witness has an incorrect field set")
    physics_vectors = (
        "solar_to_battery_kw",
        "solar_to_electrolyser_kw",
        "solar_to_dac_kw",
        "solar_to_methanator_kw",
        "battery_to_electrolyser_kw",
        "curtailed_solar_kw",
        "h2_produced_kg",
        "co2_captured_kg",
        "ch4_produced_kg",
    )
    for name in physics_vectors:
        _finite_json_vector(name, witness[name], size=physics_steps)
    for name in (
        "battery_energy_kwh",
        "h2_inventory_kg",
        "co2_inventory_kg",
        "ch4_inventory_kg",
    ):
        _finite_json_vector(name, witness[name], size=physics_steps + 1)
    deliveries = witness["deliveries"]
    if not isinstance(deliveries, list) or len(deliveries) != 12:
        raise ValueError("registered witness must contain twelve delivery rows")
    delivered_values: list[float] = []
    requested_values: list[float] = []
    seen_ids: set[str] = set()
    previous_due = 0
    for index, delivery in enumerate(deliveries):
        if not isinstance(delivery, dict) or set(delivery) != {
            "id",
            "due_step",
            "requested_kg",
            "delivered_kg",
        }:
            raise ValueError(f"witness delivery {index} has an incorrect field set")
        delivery_id = _require_string(f"witness delivery {index} id", delivery["id"])
        if delivery_id in seen_ids:
            raise ValueError("witness delivery IDs must be unique")
        seen_ids.add(delivery_id)
        due_step = _require_int(
            f"witness delivery {index} due_step", delivery["due_step"], minimum=1
        )
        if due_step < previous_due or due_step > physics_steps:
            raise ValueError("witness delivery deadlines must be ordered in-horizon")
        previous_due = due_step
        requested_value = _require_float(
            f"witness delivery {index} requested_kg",
            delivery["requested_kg"],
            minimum=0.0,
        )
        delivered_value = _require_float(
            f"witness delivery {index} delivered_kg",
            delivery["delivered_kg"],
            minimum=0.0,
        )
        if delivered_value > requested_value:
            raise ValueError("witness delivery cannot exceed its request")
        requested_values.append(requested_value)
        delivered_values.append(delivered_value)
    delivered = math.fsum(delivered_values)
    if not _same_binary64(delivered, witness["delivered_kg"]):
        raise ValueError("witness delivered_kg does not match its delivery rows")
    expected_normalized_size = 13 * physics_steps + 16
    _finite_json_vector(
        "normalized_variables",
        witness["normalized_variables"],
        size=expected_normalized_size,
    )
    residuals = witness["residuals"]
    if not isinstance(residuals, dict) or set(residuals) != _RESIDUAL_FIELDS:
        raise ValueError("registered witness residuals have an incorrect field set")
    residual_values = {
        name: _require_float(f"witness residual {name}", value, minimum=0.0)
        for name, value in residuals.items()
    }
    for name in (
        "max_equality_normalized",
        "max_inequality_violation_normalized",
        "max_box_violation_normalized",
    ):
        if residual_values[name] > feasibility_tolerance:
            raise ValueError("registered normalized residual exceeds tolerance")
    physical_names = _RESIDUAL_FIELDS - {
        "max_equality_normalized",
        "max_inequality_violation_normalized",
        "max_box_violation_normalized",
        "max_physical_violation",
    }
    maximum_physical = max(residual_values[name] for name in physical_names)
    if not _same_binary64(maximum_physical, residual_values["max_physical_violation"]):
        raise ValueError("max_physical_violation does not match residual fields")
    if maximum_physical > physical_tolerance:
        raise ValueError("registered physical residual exceeds tolerance")
    requested_exact = sum(
        (Fraction.from_float(value) for value in requested_values),
        start=Fraction(0),
    )
    return delivered, _next_float_up(requested_exact)


def _validate_registered_certificate(
    certificate: object,
    *,
    result_upper: float,
    requested_kg: float,
    primal_kg: float,
    marginal_tolerance: float,
) -> None:
    if not isinstance(certificate, dict) or set(certificate) != _CERTIFICATE_FIELDS:
        raise ValueError("registered dual certificate has an incorrect field set")
    lower = _fraction(
        "certificate minimization_lower_bound_exact",
        certificate["minimization_lower_bound_exact"],
    )
    uncapped = _fraction(
        "certificate uncapped_delivery_upper_bound_exact",
        certificate["uncapped_delivery_upper_bound_exact"],
    )
    requested = _fraction(
        "certificate requested_kg_exact", certificate["requested_kg_exact"]
    )
    certified = _fraction(
        "certificate certified_delivery_upper_bound_exact",
        certificate["certified_delivery_upper_bound_exact"],
    )
    if uncapped != -lower or certified != min(requested, uncapped) or certified < 0:
        raise ValueError("dual certificate exact rationals are internally inconsistent")
    if not _same_binary64(_next_float_up(requested), requested_kg):
        raise ValueError("dual certificate request does not match result request")
    if certified < Fraction.from_float(primal_kg):
        raise ValueError("dual certificate lies below the numerical primal")
    certificate_upper = _require_float(
        "certificate dual upper",
        certificate["dual_certified_delivery_upper_bound_kg"],
        minimum=0.0,
    )
    if not _same_binary64(certificate_upper, result_upper) or not _same_binary64(
        certificate_upper, _next_float_up(certified)
    ):
        raise ValueError("dual certificate float is not its directed exact bound")
    positive_count = _require_int(
        "certificate positive marginal count",
        certificate["positive_inequality_marginal_count"],
        minimum=0,
    )
    max_positive = _require_float(
        "certificate maximum positive marginal",
        certificate["max_positive_inequality_marginal"],
        minimum=0.0,
    )
    if (positive_count == 0) != (max_positive == 0.0):
        raise ValueError("dual certificate positive-marginal summary is inconsistent")
    if max_positive > marginal_tolerance:
        raise ValueError("dual certificate marginal projection exceeds tolerance")
    if (
        certificate["arithmetic"]
        != ("exact Fraction.from_float over stored IEEE-754 coefficients")
        or certificate["float_conversion"] != "directed toward positive infinity"
    ):
        raise ValueError("dual certificate arithmetic descriptor is invalid")
    if certificate["certificate_scope"] != _CERTIFICATE_SCOPE:
        raise ValueError("dual certificate scope is invalid")


def _validate_relaxation_graph(
    *,
    result: CanonicalPayloadV1,
    witness: CanonicalPayloadV1,
    guidance: CanonicalPayloadV1,
    settings: PlannerSettingsV1,
    exogenous_episode_sha256: str,
    canonical_episode_verified: bool,
) -> None:
    result_value = result.value()
    witness_value = witness.value()
    guidance_value = guidance.value()
    if not isinstance(result_value, dict):
        raise ValueError("relaxation result payload must contain an object")
    if not isinstance(witness_value, dict):
        raise ValueError("relaxation witness payload must contain an object")
    if not isinstance(guidance_value, dict):
        raise ValueError("relaxation guidance payload must contain an object")
    if not canonical_episode_verified:
        # Synthetic unit fixtures intentionally need no fabricated SciPy certificate.
        # The result's bound/tolerance are still consumed and validated by comparison.
        _relaxation_upper_bound_and_tolerance(result)
        return
    if set(result_value) != _RELAXATION_RESULT_FIELDS:
        raise ValueError("relaxation result payload has an incorrect field set")
    if set(guidance_value) != _RELAXATION_GUIDANCE_FIELDS:
        raise ValueError("relaxation guidance payload has an incorrect field set")
    required_result_values = {
        "schema_version": 1,
        "success": True,
        "status": 0,
        "failure_code": None,
        "model_sha256": settings.relaxation_model_sha256,
        "checkpoint_contract_digest": settings.checkpoint_contract_digest,
        "exogenous_episode_sha256": exogenous_episode_sha256,
        "canonical_episode_verified": canonical_episode_verified,
        "preserve_initial_endowment": False,
        "is_task_upper_bound": True,
    }
    for name, expected in required_result_values.items():
        if result_value[name] != expected:
            raise ValueError(f"relaxation result {name} does not match the planner")
    _require_int(
        "relaxation result schema_version", result_value["schema_version"], minimum=1
    )
    _require_int("relaxation result status", result_value["status"], minimum=0)
    for name in (
        "success",
        "canonical_episode_verified",
        "preserve_initial_endowment",
        "is_task_upper_bound",
    ):
        _require_bool(f"relaxation result {name}", result_value[name])
    if result_value["certificate"] is None or result_value["witness"] is None:
        raise ValueError("relaxation result requires certificate and witness")
    if canonical_json(result_value["witness"]) != witness.canonical_json:
        raise ValueError("relaxation result witness is not byte-exactly cross-bound")
    result_upper = _require_float(
        "dual_certified_delivery_upper_bound_kg",
        result_value["dual_certified_delivery_upper_bound_kg"],
        minimum=0.0,
    )
    solve_settings = result_value["solve_settings"]
    if (
        not isinstance(solve_settings, dict)
        or set(solve_settings) != _SOLVE_SETTINGS_FIELDS
    ):
        raise ValueError("relaxation solve_settings has an incorrect field set")
    expected_numeric_settings = {
        "feasibility_tolerance": 1e-7,
        "physical_tolerance": 1e-6,
        "optimality_tolerance_kg": 1e-5,
        "marginal_sign_tolerance": 1e-9,
    }
    for name, expected_float in expected_numeric_settings.items():
        if not _same_binary64(solve_settings[name], expected_float):
            raise ValueError(f"registered relaxation {name} is not frozen")
    if solve_settings["solver_options"] != {}:
        raise ValueError("registered relaxation solver_options must be empty")
    highs_version = solve_settings["highs_runtime_version"]
    if highs_version is not None:
        _require_string("registered HiGHS runtime version", highs_version)
    expected_gaps = [_LOCK_PROVENANCE_GAP]
    if highs_version is None:
        expected_gaps.append(_HIGHS_PROVENANCE_GAP)
    if solve_settings["provenance_gaps"] != expected_gaps:
        raise ValueError("registered relaxation provenance gaps are inconsistent")
    for name in ("solver", "message"):
        _require_string(f"relaxation result {name}", result_value[name])
    if result_value["solver"] != _RELAXATION_SOLVER:
        raise ValueError("registered relaxation solver identity is not frozen")
    _require_string("relaxation result solver_version", result_value["solver_version"])
    if result_value["evidence_provenance_status"] != (
        "canonical sample_episode regeneration verified"
    ):
        raise ValueError("registered relaxation provenance status is invalid")
    if result_value["privileged_information"] != _UPPER_BOUND_PRIVILEGED_INFORMATION:
        raise ValueError(
            "registered relaxation privileged-information label is invalid"
        )
    if result_value["certificate_scope"] != _CERTIFICATE_SCOPE:
        raise ValueError("registered relaxation certificate scope is invalid")
    if result_value["simulator_embedding_status"] != _SIMULATOR_EMBEDDING_STATUS:
        raise ValueError("registered relaxation simulator-embedding status is invalid")
    requested = _require_float(
        "registered relaxation requested_kg", result_value["requested_kg"], minimum=0.0
    )
    primal = _require_float(
        "registered relaxation numerical primal",
        result_value["numerical_primal_delivery_kg"],
        minimum=0.0,
    )
    fill = _require_float(
        "registered relaxation fill ceiling",
        result_value["relaxed_fill_ceiling"],
        minimum=0.0,
    )
    gap = _require_float(
        "registered relaxation primal-dual gap",
        result_value["primal_dual_gap_kg"],
        minimum=0.0,
    )
    objective = _require_float(
        "registered relaxation objective", result_value["numerical_objective"]
    )
    _require_int(
        "registered relaxation iterations", result_value["iterations"], minimum=0
    )
    if result_value["marginal_projection_within_tolerance"] is not True:
        raise ValueError("registered relaxation marginal projection must pass")
    if not _same_binary64(gap, result_upper - primal):
        raise ValueError("registered relaxation primal-dual gap is inconsistent")
    if gap > expected_numeric_settings["optimality_tolerance_kg"]:
        raise ValueError(
            "registered relaxation primal-dual gap exceeds the frozen tolerance"
        )
    objective_tolerance = max(
        expected_numeric_settings["physical_tolerance"],
        expected_numeric_settings["feasibility_tolerance"] * max(1.0, requested),
    )
    if abs(objective + primal) > objective_tolerance:
        raise ValueError(
            "registered relaxation numerical objective disagrees with the primal"
        )
    expected_fill = 1.0 if requested == 0.0 else result_upper / requested
    if not _same_binary64(fill, expected_fill):
        raise ValueError("registered relaxation fill ceiling is inconsistent")
    witness_delivered, witness_requested = _validate_registered_relaxation_witness(
        witness_value,
        physics_steps=settings.expected_physics_steps,
        physical_tolerance=1e-6,
        feasibility_tolerance=1e-7,
    )
    if not _same_binary64(primal, witness_delivered):
        raise ValueError("registered relaxation primal does not match witness")
    if not _same_binary64(requested, witness_requested):
        raise ValueError("registered relaxation request does not match witness")
    _validate_registered_certificate(
        result_value["certificate"],
        result_upper=result_upper,
        requested_kg=requested,
        primal_kg=primal,
        marginal_tolerance=1e-9,
    )
    expected_guidance = {
        "schema_version": 1,
        "model_sha256": settings.relaxation_model_sha256,
        "result_sha256": result.sha256,
        "witness_sha256": witness.sha256,
        "status": 0,
        "canonical_episode_verified": canonical_episode_verified,
        "preserve_initial_endowment": False,
        "is_task_upper_bound": True,
        "solver": result_value["solver"],
        "solver_version": result_value["solver_version"],
        "solve_settings": solve_settings,
        "checkpoint_contract_digest": settings.checkpoint_contract_digest,
        "exogenous_episode_sha256": exogenous_episode_sha256,
        "solar_truth_sha256": require_sha256(
            "relaxation guidance solar_truth_sha256",
            guidance_value["solar_truth_sha256"],
        ),
    }
    if guidance.canonical_json != canonical_json(expected_guidance):
        raise ValueError(
            "relaxation guidance canonical bytes do not match result/settings"
        )


def _validate_search_diagnostics(
    diagnostics: SearchDiagnosticsV1,
    *,
    settings: PlannerSettingsV1,
    success: bool,
    has_a0: bool,
    has_finalist: bool,
) -> None:
    decisions = settings.expected_decisions
    macro_depths = (
        decisions + settings.macro_decisions - 1
    ) // settings.macro_decisions
    nonterminal_cuts = max(macro_depths - 1, 0)
    coarse_blocks = (
        decisions + settings.coarse_block_decisions - 1
    ) // settings.coarse_block_decisions
    fine_blocks = (
        decisions + settings.fine_block_decisions - 1
    ) // settings.fine_block_decisions
    maximum_coordinate_trials = 15 * (
        len(settings.coarse_origins) * coarse_blocks + fine_blocks
    )
    maximum_coordinate_accepts = (
        len(settings.coarse_origins) * coarse_blocks + fine_blocks
    )

    if diagnostics.completed_macro_depths > macro_depths:
        raise ValueError("completed_macro_depths exceeds the frozen horizon")
    if diagnostics.attempted_edges > settings.max_attempted_edges:
        raise ValueError("attempted_edges exceeds the frozen edge ceiling")
    if diagnostics.peak_frontier > settings.beam_width:
        raise ValueError("peak_frontier exceeds the frozen beam width")
    if diagnostics.max_live_environments_observed > settings.max_live_environments:
        raise ValueError("observed live environments exceed the frozen ceiling")
    if diagnostics.a0_cursor_validations > macro_depths:
        raise ValueError("A0 cursor validations exceed macro depth")
    if diagnostics.a0_injections > nonterminal_cuts:
        raise ValueError("A0 injections exceed nonterminal cuts")
    if diagnostics.a0_injections > diagnostics.a0_cursor_validations:
        raise ValueError("A0 injections cannot exceed cursor validations")
    if diagnostics.a0_generation_decisions > decisions:
        raise ValueError("A0 generation decisions exceed the horizon")
    if diagnostics.a0_evidence_replay_decisions > decisions:
        raise ValueError("A0 replay decisions exceed the horizon")
    if diagnostics.fresh_finalist_replay_decisions > decisions:
        raise ValueError("finalist replay decisions exceed the horizon")
    if diagnostics.coordinate_trials_attempted > maximum_coordinate_trials:
        raise ValueError("coordinate attempts exceed the frozen schedule")
    if diagnostics.coordinate_trials_accepted > maximum_coordinate_accepts:
        raise ValueError("coordinate accepts exceed the number of frozen blocks")
    retained_cut_count = min(diagnostics.completed_macro_depths, nonterminal_cuts)
    if diagnostics.frontier_retained_total > retained_cut_count * settings.beam_width:
        raise ValueError("frontier retention exceeds completed beam cuts")
    if diagnostics.exact_duplicates > (
        diagnostics.generated_candidates + diagnostics.a0_injections
    ):
        raise ValueError("duplicate count exceeds materialized candidates")
    if has_a0 and (
        diagnostics.a0_generation_decisions != decisions
        or diagnostics.a0_evidence_replay_decisions != decisions
    ):
        raise ValueError("retained A0 evidence requires two complete A0 traces")

    if not success:
        return
    legal_coordinate_trials = {
        15 * (coarse_origin_count * coarse_blocks + fine_blocks): coarse_origin_count
        for coarse_origin_count in (1, len(settings.coarse_origins))
    }
    if diagnostics.completed_macro_depths != macro_depths:
        raise ValueError("successful search must complete every macro depth")
    if diagnostics.max_live_environments_observed < 1:
        raise ValueError("successful search must observe at least one live environment")
    if diagnostics.a0_cursor_validations != macro_depths:
        raise ValueError("successful search must validate every A0 cursor prefix")
    if diagnostics.a0_injections != nonterminal_cuts:
        raise ValueError("successful search must inject A0 at every nonterminal cut")
    if diagnostics.completed_edges != diagnostics.attempted_edges:
        raise ValueError("successful search must complete every attempted edge")
    if diagnostics.attempted_edges != 15 * (1 + diagnostics.frontier_retained_total):
        raise ValueError("edge count does not match retained-parent expansion")
    if not nonterminal_cuts:
        if diagnostics.frontier_retained_total != 0:
            raise ValueError("single-macro search cannot retain a nonterminal frontier")
    elif not (
        nonterminal_cuts
        <= diagnostics.frontier_retained_total
        <= nonterminal_cuts * settings.beam_width
    ):
        raise ValueError("successful retained frontier total is outside beam bounds")
    if diagnostics.frontier_retained_total and not (
        1 <= diagnostics.peak_frontier <= settings.beam_width
        and diagnostics.peak_frontier <= diagnostics.frontier_retained_total
    ):
        raise ValueError("peak_frontier is inconsistent with retained frontiers")
    if (
        diagnostics.a0_generation_decisions != decisions
        or diagnostics.a0_evidence_replay_decisions != decisions
    ):
        raise ValueError("successful search requires complete A0 generation/replay")
    expected_finalist_replay = decisions if has_finalist else 0
    if diagnostics.fresh_finalist_replay_decisions != expected_finalist_replay:
        raise ValueError("finalist replay count does not match finalist presence")
    if (
        diagnostics.coordinate_trials_attempted not in legal_coordinate_trials
        or diagnostics.coordinate_trials_completed
        != diagnostics.coordinate_trials_attempted
    ):
        raise ValueError(
            "successful coordinate trials do not match the frozen schedule"
        )
    executed_coarse_origins = legal_coordinate_trials[
        diagnostics.coordinate_trials_attempted
    ]
    executed_coordinate_blocks = executed_coarse_origins * coarse_blocks + fine_blocks
    if diagnostics.coordinate_trials_accepted > executed_coordinate_blocks:
        raise ValueError(
            "coordinate accepts exceed the number of executed frozen blocks"
        )


def _validate_registered_terminal_scores(
    replays: Iterable[ReplayEvidenceV1 | None],
) -> None:
    from plant_sim import PlantConfig

    config = PlantConfig()
    config_sha256 = hashlib.sha256(
        canonical_json(config.to_dict()).encode("utf-8")
    ).hexdigest()
    if config_sha256 != PLANT_CONFIG_SHA256:
        raise ValueError(
            "current default PlantConfig no longer matches the frozen hash"
        )
    h2_per_ch4 = _require_float(
        "default methanator h2 ratio",
        config.methanator.h2_kg_per_kg_ch4,
        minimum=0.0,
    )
    co2_per_ch4 = _require_float(
        "default methanator co2 ratio",
        config.methanator.co2_kg_per_kg_ch4,
        minimum=0.0,
    )
    if h2_per_ch4 == 0.0 or co2_per_ch4 == 0.0:
        raise ValueError("default methanator ratios must be positive")
    for replay in replays:
        if replay is None:
            continue
        truth = replay.terminal_truth.value()
        if not isinstance(truth, dict):  # Replay construction already enforces this.
            raise ValueError("terminal truth must be an object")
        h2 = _require_float(
            "terminal h2_level_kg", truth.get("h2_level_kg"), minimum=0.0
        )
        co2 = _require_float(
            "terminal co2_level_kg", truth.get("co2_level_kg"), minimum=0.0
        )
        ch4 = _require_float(
            "terminal ch4_level_kg", truth.get("ch4_level_kg"), minimum=0.0
        )
        equivalent = ch4 + min(h2 / h2_per_ch4, co2 / co2_per_ch4)
        if not _same_binary64(
            equivalent,
            replay.final_score.terminal_ch4_equivalent_inventory_kg,
        ):
            raise ValueError(
                "registered terminal CH4-equivalent inventory does not match truth"
            )


@dataclass(frozen=True, slots=True)
class FeasiblePlannerResultV1:
    schema_version: int
    planner_kind: str
    evidence_scope: str
    privileged_information: str
    optimality_claim: str
    evidence_ready: bool
    provenance_gaps: tuple[str, ...]
    success: bool
    failure: PlannerFailureV1 | None
    seed: int | None
    canonical_episode_verified: bool
    manifest_sha256: str | None
    exogenous_episode_sha256: str | None
    checkpoint_contract_digest: str | None
    relaxation_model_sha256: str | None
    relaxation_result_sha256: str | None
    relaxation_result: CanonicalPayloadV1 | None
    relaxation_witness: CanonicalPayloadV1 | None
    relaxation_guidance: CanonicalPayloadV1 | None
    settings: PlannerSettingsV1 | None
    search: SearchDiagnosticsV1
    a0_incumbent: ReplayEvidenceV1 | None
    searched_finalist: ReplayEvidenceV1 | None
    selected_role: str | None
    comparison: IncumbentComparisonV1 | None
    deterministic_result_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _require_int("schema_version", self.schema_version, minimum=1),
        )
        if self.schema_version != 1 or self.planner_kind != PLANNER_KIND:
            raise ValueError("core result schema/planner kind is not version 1")
        if self.evidence_scope not in EVIDENCE_SCOPES:
            raise ValueError(f"unsupported evidence_scope: {self.evidence_scope}")
        if self.privileged_information != PRIVILEGED_INFORMATION:
            raise ValueError("privileged_information does not match the protocol")
        if self.optimality_claim != OPTIMALITY_CLAIM:
            raise ValueError("optimality_claim must be none")
        if self.evidence_ready is not False:
            raise ValueError("a core result can never be evidence-ready")
        object.__setattr__(self, "provenance_gaps", tuple(self.provenance_gaps))
        if self.provenance_gaps != PROVENANCE_GAPS:
            raise ValueError("provenance_gaps does not match the frozen core record")
        _require_bool("success", self.success)
        _require_bool("canonical_episode_verified", self.canonical_episode_verified)
        if self.seed is not None:
            object.__setattr__(self, "seed", _require_int("seed", self.seed, minimum=0))
        for name in (
            "manifest_sha256",
            "exogenous_episode_sha256",
            "checkpoint_contract_digest",
            "relaxation_model_sha256",
            "relaxation_result_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                require_sha256(name, value)
        for name, domain in (
            ("relaxation_result", "planner-relaxation-result-v1"),
            ("relaxation_witness", "planner-relaxation-witness-v1"),
            ("relaxation_guidance", "planner-relaxation-guidance-v1"),
        ):
            value = getattr(self, name)
            if value is not None:
                _require_payload(name, value, domain)
        if not isinstance(self.search, SearchDiagnosticsV1):
            raise TypeError("search must be SearchDiagnosticsV1")
        if self.failure is not None and not isinstance(self.failure, PlannerFailureV1):
            raise TypeError("failure must be PlannerFailureV1 or null")
        for name, role in (
            ("a0_incumbent", "a0_incumbent"),
            ("searched_finalist", "searched_finalist"),
        ):
            replay = getattr(self, name)
            if replay is not None:
                if not isinstance(replay, ReplayEvidenceV1) or replay.role != role:
                    raise ValueError(f"{name} has the wrong replay role")
        self._validate_status()
        self._validate_identities()
        if self.settings is not None:
            _validate_search_diagnostics(
                self.search,
                settings=self.settings,
                success=self.success,
                has_a0=self.a0_incumbent is not None,
                has_finalist=self.searched_finalist is not None,
            )
        expected_digest = _core_result_digest(self)
        if self.deterministic_result_sha256 == "":
            object.__setattr__(self, "deterministic_result_sha256", expected_digest)
        elif (
            require_sha256(
                "deterministic_result_sha256", self.deterministic_result_sha256
            )
            != expected_digest
        ):
            raise ValueError("deterministic_result_sha256 does not match its preimage")

    def _validate_status(self) -> None:
        if self.success:
            if self.failure is not None:
                raise ValueError("successful core result cannot contain a failure")
            if self.search.termination_reason != "completed":
                raise ValueError(
                    "successful search termination_reason must be completed"
                )
            if (
                self.a0_incumbent is None
                or self.selected_role is None
                or self.comparison is None
                or self.settings is None
                or self.relaxation_result is None
                or self.relaxation_witness is None
                or self.relaxation_guidance is None
            ):
                raise ValueError("successful core result is missing required evidence")
            if self.selected_role not in REPLAY_ROLES:
                raise ValueError("successful selected_role is invalid")
            selected = (
                self.a0_incumbent
                if self.selected_role == "a0_incumbent"
                else self.searched_finalist
            )
            if selected is None:
                raise ValueError("selected_role does not resolve to replay evidence")
            if self.searched_finalist is not None and (
                self.searched_finalist.request_trace.canonical_json
                == self.a0_incumbent.request_trace.canonical_json
            ):
                raise ValueError("a byte-identical searched finalist must be omitted")
            searched_is_better = (
                self.searched_finalist is not None
                and compare_final_scores(
                    self.searched_finalist.final_score,
                    self.a0_incumbent.final_score,
                )
                > 0
            )
            expected_role = (
                "searched_finalist" if searched_is_better else "a0_incumbent"
            )
            if self.selected_role != expected_role:
                raise ValueError(
                    "selection must choose the searched finalist exactly when its "
                    "complete score exceeds A0"
                )
            score_order = compare_final_scores(
                selected.final_score, self.a0_incumbent.final_score
            )
            if score_order < 0:
                raise ValueError("selected score regresses below A0")
            if score_order == 0 and self.selected_role != "a0_incumbent":
                raise ValueError("A0 must win an equal final score")
            if self.comparison.selected_role != self.selected_role:
                raise ValueError("comparison.selected_role does not match selection")
            upper, tolerance = _relaxation_upper_bound_and_tolerance(
                self.relaxation_result
            )
            expected = build_incumbent_comparison(
                selected_role=self.selected_role,
                a0_score=self.a0_incumbent.final_score,
                selected_score=selected.final_score,
                a0_request_trace=self.a0_incumbent.request_trace,
                selected_request_trace=selected.request_trace,
                upper_bound_kg=upper,
                optimality_tolerance_kg=tolerance,
            )
            if canonical_json(self.comparison.to_dict()) != canonical_json(
                expected.to_dict()
            ):
                raise ValueError("comparison does not match the exact finalist bracket")
        else:
            if self.failure is None:
                raise ValueError("failed core result requires PlannerFailureV1")
            if any(
                value is not None
                for value in (
                    self.searched_finalist,
                    self.selected_role,
                    self.comparison,
                )
            ):
                raise ValueError("failed core result cannot retain finalist/selection")
            if self.search.termination_reason != self.failure.code:
                raise ValueError("failure code must equal search termination_reason")
            if (
                self.failure.completed_macro_depths
                != self.search.completed_macro_depths
                or self.failure.attempted_edges != self.search.attempted_edges
                or self.failure.completed_edges != self.search.completed_edges
                or self.failure.coordinate_trials_completed
                != self.search.coordinate_trials_completed
            ):
                raise ValueError("failure counters must agree with diagnostics")

    def _validate_identities(self) -> None:
        if self.relaxation_result is not None:
            if self.relaxation_result_sha256 != self.relaxation_result.sha256:
                raise ValueError("relaxation result identity mismatch")
        if self.settings is None:
            if self.success or self.a0_incumbent is not None:
                raise ValueError("success or retained A0 evidence requires settings")
            return
        if not isinstance(self.settings, PlannerSettingsV1):
            raise TypeError("settings must be PlannerSettingsV1")
        for name in (
            "manifest_sha256",
            "exogenous_episode_sha256",
            "checkpoint_contract_digest",
            "relaxation_model_sha256",
            "relaxation_result_sha256",
        ):
            if getattr(self, name) is None:
                raise ValueError(f"settings-bearing result requires {name}")
        if any(
            value is None
            for value in (
                self.relaxation_result,
                self.relaxation_witness,
                self.relaxation_guidance,
            )
        ):
            raise ValueError(
                "settings-bearing result requires the complete relaxation payload graph"
            )
        identity_pairs = (
            (self.manifest_sha256, None, "manifest_sha256"),
            (
                self.exogenous_episode_sha256,
                None,
                "exogenous_episode_sha256",
            ),
            (
                self.checkpoint_contract_digest,
                self.settings.checkpoint_contract_digest,
                "checkpoint_contract_digest",
            ),
            (
                self.relaxation_model_sha256,
                self.settings.relaxation_model_sha256,
                "relaxation_model_sha256",
            ),
            (
                self.relaxation_result_sha256,
                self.settings.relaxation_result_sha256,
                "relaxation_result_sha256",
            ),
        )
        for actual, expected, name in identity_pairs:
            if self.success and actual is None:
                raise ValueError(f"successful result requires {name}")
            if expected is not None and actual is not None and actual != expected:
                raise ValueError(f"{name} disagrees with settings")
        if self.relaxation_witness is not None and (
            self.relaxation_witness.sha256 != self.settings.relaxation_witness_sha256
        ):
            raise ValueError("relaxation witness identity mismatch")
        if self.relaxation_guidance is not None and (
            self.relaxation_guidance.sha256 != self.settings.relaxation_guidance_sha256
        ):
            raise ValueError("relaxation guidance identity mismatch")
        if (
            self.relaxation_result is not None
            and self.relaxation_witness is not None
            and self.relaxation_guidance is not None
            and self.exogenous_episode_sha256 is not None
        ):
            _validate_relaxation_graph(
                result=self.relaxation_result,
                witness=self.relaxation_witness,
                guidance=self.relaxation_guidance,
                settings=self.settings,
                exogenous_episode_sha256=self.exogenous_episode_sha256,
                canonical_episode_verified=self.canonical_episode_verified,
            )
        for replay in (self.a0_incumbent, self.searched_finalist):
            if replay is None:
                continue
            for name in (
                "manifest_sha256",
                "exogenous_episode_sha256",
                "checkpoint_contract_digest",
            ):
                if getattr(replay, name) != getattr(self, name):
                    raise ValueError(f"replay {name} disagrees with core result")
            if replay.decision_count != self.settings.expected_decisions:
                raise ValueError("replay decision count disagrees with settings")
            if replay.physics_steps != self.settings.expected_physics_steps:
                raise ValueError("replay physics count disagrees with settings")
            binding = replay.replay_binding.value()
            assert isinstance(binding, dict)
            if binding["settings_sha256"] != self.settings.settings_sha256:
                raise ValueError("replay binding uses the wrong settings")
            if (
                binding["relaxation_guidance_sha256"]
                != self.settings.relaxation_guidance_sha256
            ):
                raise ValueError("replay binding uses the wrong guidance")
            _validate_action_bindings(replay, self.settings)
        if self.searched_finalist is not None and self.a0_incumbent is not None:
            for searched_record, a0_record in zip(
                self.searched_finalist.action_records,
                self.a0_incumbent.action_records,
                strict=True,
            ):
                if searched_record.source_code == -1 and (
                    canonical_json(searched_record.request_dict())
                    != canonical_json(a0_record.request_dict())
                    or searched_record.source_binding_sha256
                    != a0_record.source_binding_sha256
                ):
                    raise ValueError(
                        "retained A0 lineage does not match the A0 request"
                    )
        if self.evidence_scope == "privileged_offline_development":
            if self.seed is None:
                raise ValueError("privileged development requires a seed")
            require_planner_admission(
                self.settings,
                seed=self.seed,
                evidence_scope=self.evidence_scope,
                canonical_episode_verified=self.canonical_episode_verified,
            )
            if self.success and (
                self.search.a0_cursor_validations != 72
                or self.search.a0_injections != 71
            ):
                raise ValueError(
                    "registered success requires exact A0 protection counts"
                )
            _validate_registered_terminal_scores(
                (self.a0_incumbent, self.searched_finalist)
            )
        else:
            if self.canonical_episode_verified:
                raise ValueError(
                    "noncanonical unit fixtures cannot claim canonical verification"
                )
            if self.settings.expected_decisions > 8:
                raise ValueError(
                    "noncanonical unit fixtures are limited to eight decisions"
                )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, tuple):
                result[field.name] = list(value)
            elif hasattr(value, "to_dict"):
                result[field.name] = value.to_dict()
            else:
                result[field.name] = value
        return result


def core_result_preimage(result: FeasiblePlannerResultV1) -> dict[str, object]:
    """Return the exact planner-result-v1 object after path-specific omissions."""
    if not isinstance(result, FeasiblePlannerResultV1):
        raise TypeError("result must be FeasiblePlannerResultV1")
    value = result.to_dict()
    value.pop("deterministic_result_sha256")
    search = value["search"]
    assert isinstance(search, dict)
    search.pop("elapsed_ns")
    search.pop("peak_rss_bytes")
    failure = value.get("failure")
    if isinstance(failure, dict):
        failure.pop("message")
        failure.pop("elapsed_ns")
        failure.pop("observed_resource_value")
    for name in ("a0_incumbent", "searched_finalist"):
        replay = value.get(name)
        if isinstance(replay, dict):
            replay.pop("replay_elapsed_ns")
    return value


def _core_result_digest(result: FeasiblePlannerResultV1) -> str:
    return canonical_payload("planner-result-v1", core_result_preimage(result)).sha256


_SOURCE_SNAPSHOT_FIELDS = {
    "schema_version",
    "git_revision",
    "git_dirty",
    "tracked_binary_diff_sha256",
    "tracked_binary_diff_bytes",
    "untracked_files",
    "worktree_identity_format",
    "worktree_identity_sha256",
    "identity_complete",
}
_RUNTIME_FIELDS = {
    "schema_version",
    "python_version",
    "python_implementation",
    "python_compiler",
    "numpy_version",
    "scipy_version",
    "highs_runtime_version",
    "gymnasium_version",
    "platform_system",
    "platform_release",
    "platform_version",
    "platform_machine",
    "platform_processor",
    "pointer_bits",
    "byteorder",
    "clock_method",
    "rss_method",
}


def _validate_source_snapshot(payload: CanonicalPayloadV1) -> dict[str, object]:
    _require_payload("source snapshot", payload, "planner-source-snapshot-v1")
    value = payload.value()
    if not isinstance(value, dict) or set(value) != _SOURCE_SNAPSHOT_FIELDS:
        raise ValueError("source snapshot has an incorrect field set")
    if value["schema_version"] != 1:
        raise ValueError("source snapshot schema_version must be 1")
    revision = _require_string("git_revision", value["git_revision"])
    if _GIT_COMMIT_RE.fullmatch(revision) is None:
        raise ValueError("git_revision must be a complete lowercase 40-hex commit ID")
    _require_bool("git_dirty", value["git_dirty"])
    require_sha256("tracked_binary_diff_sha256", value["tracked_binary_diff_sha256"])
    _require_int(
        "tracked_binary_diff_bytes", value["tracked_binary_diff_bytes"], minimum=0
    )
    if value["worktree_identity_format"] != "git-worktree-content-v1":
        raise ValueError("unsupported worktree identity format")
    require_sha256("worktree_identity_sha256", value["worktree_identity_sha256"])
    if value["identity_complete"] is not True:
        raise ValueError("source snapshots must have complete identity")
    untracked = value["untracked_files"]
    if not isinstance(untracked, list):
        raise TypeError("untracked_files must be an array")
    paths: list[str] = []
    for index, entry in enumerate(untracked):
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "kind",
            "content_sha256",
            "bytes",
        }:
            raise ValueError(f"untracked_files[{index}] has an incorrect field set")
        path = _require_string(f"untracked_files[{index}].path", entry["path"])
        posix_path = PurePosixPath(path)
        windows_path = PureWindowsPath(path)
        if (
            posix_path.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or windows_path.root
            or not posix_path.parts
            or ".." in posix_path.parts
            or ".." in windows_path.parts
        ):
            raise ValueError(
                f"untracked_files[{index}].path must be repository-relative"
            )
        if entry["kind"] not in {"file", "symlink"}:
            raise ValueError(f"untracked_files[{index}].kind is invalid")
        require_sha256(
            f"untracked_files[{index}].content_sha256", entry["content_sha256"]
        )
        _require_int(f"untracked_files[{index}].bytes", entry["bytes"], minimum=0)
        paths.append(path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("untracked_files must be uniquely path-sorted")
    if value["git_dirty"] is False and (
        value["tracked_binary_diff_bytes"] != 0
        or value["tracked_binary_diff_sha256"] != _EMPTY_SHA256
        or untracked
    ):
        raise ValueError(
            "a clean source snapshot must have an empty diff and no untracked files"
        )
    return value


def source_snapshot_payload(value: Mapping[str, object]) -> CanonicalPayloadV1:
    payload = canonical_payload("planner-source-snapshot-v1", dict(value))
    _validate_source_snapshot(payload)
    return payload


def _validate_runtime(payload: CanonicalPayloadV1) -> dict[str, object]:
    _require_payload("runtime_versions", payload, "planner-runtime-v1")
    value = payload.value()
    if not isinstance(value, dict) or set(value) != _RUNTIME_FIELDS:
        raise ValueError("runtime payload has an incorrect field set")
    if value["schema_version"] != 1:
        raise ValueError("runtime schema_version must be 1")
    nullable = {"highs_runtime_version", "platform_processor"}
    for name in _RUNTIME_FIELDS - {"schema_version", "pointer_bits", "byteorder"}:
        if value[name] is None and name in nullable:
            continue
        _require_string(name, value[name])
    _require_int("pointer_bits", value["pointer_bits"], minimum=1)
    if value["byteorder"] not in {"little", "big"}:
        raise ValueError("byteorder must be little or big")
    return value


def runtime_payload(value: Mapping[str, object]) -> CanonicalPayloadV1:
    payload = canonical_payload("planner-runtime-v1", dict(value))
    _validate_runtime(payload)
    return payload


@dataclass(frozen=True, slots=True)
class FeasiblePlannerArtifactV1:
    schema_version: int
    success: bool
    evidence_ready: bool
    core_result: FeasiblePlannerResultV1 | None
    core_result_sha256: str | None
    runner_failure: PlannerRunnerFailureV1 | None
    start_source: CanonicalPayloadV1 | None
    end_source: CanonicalPayloadV1 | None
    start_lock_sha256: str | None
    end_lock_sha256: str | None
    runtime_versions: CanonicalPayloadV1 | None
    source_boundary_claim: str | None
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _require_int("schema_version", self.schema_version, minimum=1),
        )
        if self.schema_version != 1:
            raise ValueError("artifact schema_version must be 1")
        _require_bool("success", self.success)
        _require_bool("evidence_ready", self.evidence_ready)
        if self.core_result is not None:
            if not isinstance(self.core_result, FeasiblePlannerResultV1):
                raise TypeError("core_result must be FeasiblePlannerResultV1 or null")
            if self.core_result_sha256 != self.core_result.deterministic_result_sha256:
                raise ValueError("core_result_sha256 does not match embedded core")
        elif self.core_result_sha256 is not None:
            raise ValueError("core result and digest must be present or null together")
        if self.core_result_sha256 is not None:
            require_sha256("core_result_sha256", self.core_result_sha256)
        if self.runner_failure is not None and not isinstance(
            self.runner_failure, PlannerRunnerFailureV1
        ):
            raise TypeError("runner_failure must be PlannerRunnerFailureV1 or null")
        for name in ("start_source", "end_source"):
            payload = getattr(self, name)
            if payload is not None:
                _validate_source_snapshot(payload)
        for name in ("start_lock_sha256", "end_lock_sha256"):
            value = getattr(self, name)
            if value is not None:
                require_sha256(name, value)
        if self.runtime_versions is not None:
            _validate_runtime(self.runtime_versions)
        expected_success = self.runner_failure is None and (
            self.core_result is not None and self.core_result.success
        )
        if self.success != expected_success:
            raise ValueError("artifact success does not match runner/core status")
        if self.runner_failure is None:
            if self.core_result is None:
                raise ValueError("runner completion requires a validated core result")
            if any(
                value is None
                for value in (
                    self.start_source,
                    self.end_source,
                    self.start_lock_sha256,
                    self.end_lock_sha256,
                    self.runtime_versions,
                )
            ):
                raise ValueError(
                    "runner completion requires both boundaries and runtime"
                )
        else:
            phase = self.runner_failure.phase
            if phase == "start_source":
                if any(
                    value is not None
                    for value in (
                        self.core_result,
                        self.start_source,
                        self.end_source,
                        self.start_lock_sha256,
                        self.end_lock_sha256,
                        self.runtime_versions,
                    )
                ):
                    raise ValueError(
                        "start_source failure cannot retain unobserved provenance/core"
                    )
            elif phase == "child_execution":
                if (
                    self.core_result is not None
                    or self.end_source is not None
                    or self.end_lock_sha256 is not None
                ):
                    raise ValueError(
                        "child_execution failure cannot retain a core or end boundary"
                    )
                if any(
                    value is None
                    for value in (
                        self.start_source,
                        self.start_lock_sha256,
                        self.runtime_versions,
                    )
                ):
                    raise ValueError(
                        "child_execution failure requires complete start provenance"
                    )
            elif phase == "end_source":
                if self.core_result is None:
                    raise ValueError("end_source failure requires a validated core")
                if self.end_source is not None or self.end_lock_sha256 is not None:
                    raise ValueError("end_source failure cannot claim an end boundary")
                if any(
                    value is None
                    for value in (
                        self.start_source,
                        self.start_lock_sha256,
                        self.runtime_versions,
                    )
                ):
                    raise ValueError("end_source failure requires start provenance")
            else:
                if self.core_result is None or any(
                    value is None
                    for value in (
                        self.start_source,
                        self.end_source,
                        self.start_lock_sha256,
                        self.end_lock_sha256,
                        self.runtime_versions,
                    )
                ):
                    raise ValueError(
                        "artifact_serialization failure requires the complete graph"
                    )
        snapshots_match = (
            self.start_source is not None
            and self.end_source is not None
            and self.start_source.canonical_json == self.end_source.canonical_json
        )
        expected_claim = SOURCE_BOUNDARY_CLAIM if snapshots_match else None
        if self.source_boundary_claim != expected_claim:
            raise ValueError("source_boundary_claim does not match boundary snapshots")
        eligible = self._is_evidence_ready()
        if self.evidence_ready != eligible:
            raise ValueError("evidence_ready does not match complete eligibility")
        expected_digest = _artifact_digest(self)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", expected_digest)
        elif require_sha256("artifact_sha256", self.artifact_sha256) != expected_digest:
            raise ValueError("artifact_sha256 does not match its preimage")

    def _is_evidence_ready(self) -> bool:
        if (
            not self.success
            or self.runner_failure is not None
            or self.core_result is None
            or self.core_result.evidence_scope != "privileged_offline_development"
            or self.core_result.comparison is None
            or self.core_result.comparison.bracket_status != "valid"
            or self.start_source is None
            or self.end_source is None
            or self.runtime_versions is None
            or self.start_lock_sha256 is None
            or self.end_lock_sha256 is None
            or self.start_source.canonical_json != self.end_source.canonical_json
            or self.start_lock_sha256 != self.end_lock_sha256
        ):
            return False
        source = _validate_source_snapshot(self.start_source)
        runtime = _validate_runtime(self.runtime_versions)
        settings = self.core_result.settings
        if settings is None:
            return False
        return (
            source["git_dirty"] is False
            and source["untracked_files"] == []
            and runtime["clock_method"] == settings.clock_method
            and runtime["rss_method"] == settings.rss_method
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if hasattr(value, "to_dict"):
                result[field.name] = value.to_dict()
            else:
                result[field.name] = value
        return result


def artifact_preimage(artifact: FeasiblePlannerArtifactV1) -> dict[str, object]:
    """Return the exact planner-runner-artifact-v1 hash preimage object."""
    if not isinstance(artifact, FeasiblePlannerArtifactV1):
        raise TypeError("artifact must be FeasiblePlannerArtifactV1")
    value = artifact.to_dict()
    value.pop("artifact_sha256")
    return value


def _artifact_digest(artifact: FeasiblePlannerArtifactV1) -> str:
    return canonical_payload(
        "planner-runner-artifact-v1", artifact_preimage(artifact)
    ).sha256


def _strict_record_dict(
    name: str, value: object, record_type: Any
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    expected = {field.name for field in fields(record_type)}
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{name} has an incorrect field set; missing={missing!r}, extra={extra!r}"
        )
    return dict(value)


def canonical_payload_from_dict(value: object) -> CanonicalPayloadV1:
    """Strictly reconstruct a canonical payload from its exact three fields."""
    item = _strict_record_dict("canonical payload", value, CanonicalPayloadV1)
    return CanonicalPayloadV1(**item)  # type: ignore[arg-type]


def _action_record_from_dict(value: object) -> ActionRecordV1:
    item = _strict_record_dict("action record", value, ActionRecordV1)
    return ActionRecordV1(**item)  # type: ignore[arg-type]


def _final_score_from_dict(value: object) -> FinalScoreV1:
    item = _strict_record_dict("final score", value, FinalScoreV1)
    return FinalScoreV1(**item)  # type: ignore[arg-type]


def _settings_from_dict(value: object) -> PlannerSettingsV1:
    item = _strict_record_dict("planner settings", value, PlannerSettingsV1)
    item["action_library"] = canonical_payload_from_dict(item["action_library"])
    item["a0_controller"] = canonical_payload_from_dict(item["a0_controller"])
    for name in (
        "mode_protection_order",
        "coarse_origins",
        "registered_rates",
        "registered_seeds",
        "episode_config_sha256_by_rate",
        "checkpoint_contract_sha256_by_rate",
    ):
        if not isinstance(item[name], list):
            raise TypeError(f"planner settings {name} must be an array")
    return PlannerSettingsV1(**item)  # type: ignore[arg-type]


def _failure_from_dict(value: object) -> PlannerFailureV1:
    item = _strict_record_dict("planner failure", value, PlannerFailureV1)
    return PlannerFailureV1(**item)  # type: ignore[arg-type]


def _runner_failure_from_dict(value: object) -> PlannerRunnerFailureV1:
    item = _strict_record_dict("planner runner failure", value, PlannerRunnerFailureV1)
    return PlannerRunnerFailureV1(**item)  # type: ignore[arg-type]


def _search_from_dict(value: object) -> SearchDiagnosticsV1:
    item = _strict_record_dict("search diagnostics", value, SearchDiagnosticsV1)
    item["deterministic_counters"] = canonical_payload_from_dict(
        item["deterministic_counters"]
    )
    return SearchDiagnosticsV1(**item)  # type: ignore[arg-type]


def _replay_from_dict(value: object) -> ReplayEvidenceV1:
    item = _strict_record_dict("replay evidence", value, ReplayEvidenceV1)
    records = item["action_records"]
    if not isinstance(records, list):
        raise TypeError("replay evidence action_records must be an array")
    item["action_records"] = tuple(_action_record_from_dict(row) for row in records)
    for name in (
        "request_trace",
        "source_trace",
        "action_record_trace",
        "metrics",
        "terminal_truth",
        "terminal_telemetry",
        "delivery_audit",
        "generation_outcome",
        "replay_outcome",
        "replay_binding",
    ):
        item[name] = canonical_payload_from_dict(item[name])
    item["final_score"] = _final_score_from_dict(item["final_score"])
    return ReplayEvidenceV1(**item)  # type: ignore[arg-type]


def _comparison_from_dict(value: object) -> IncumbentComparisonV1:
    item = _strict_record_dict("incumbent comparison", value, IncumbentComparisonV1)
    return IncumbentComparisonV1(**item)  # type: ignore[arg-type]


def planner_result_from_dict(value: object) -> FeasiblePlannerResultV1:
    """Strictly rebuild and fully validate one detached core-result object graph."""
    item = _strict_record_dict(
        "feasible planner result", value, FeasiblePlannerResultV1
    )
    provenance_gaps = item["provenance_gaps"]
    if not isinstance(provenance_gaps, list):
        raise TypeError("feasible planner result provenance_gaps must be an array")
    item["provenance_gaps"] = tuple(provenance_gaps)
    if item["failure"] is not None:
        item["failure"] = _failure_from_dict(item["failure"])
    for name in (
        "relaxation_result",
        "relaxation_witness",
        "relaxation_guidance",
    ):
        if item[name] is not None:
            item[name] = canonical_payload_from_dict(item[name])
    if item["settings"] is not None:
        item["settings"] = _settings_from_dict(item["settings"])
    item["search"] = _search_from_dict(item["search"])
    for name in ("a0_incumbent", "searched_finalist"):
        if item[name] is not None:
            item[name] = _replay_from_dict(item[name])
    if item["comparison"] is not None:
        item["comparison"] = _comparison_from_dict(item["comparison"])
    return FeasiblePlannerResultV1(**item)  # type: ignore[arg-type]


def planner_artifact_from_dict(value: object) -> FeasiblePlannerArtifactV1:
    """Strictly rebuild and fully validate one detached runner-artifact graph."""
    item = _strict_record_dict(
        "feasible planner artifact", value, FeasiblePlannerArtifactV1
    )
    if item["core_result"] is not None:
        item["core_result"] = planner_result_from_dict(item["core_result"])
    if item["runner_failure"] is not None:
        item["runner_failure"] = _runner_failure_from_dict(item["runner_failure"])
    for name in ("start_source", "end_source", "runtime_versions"):
        if item[name] is not None:
            item[name] = canonical_payload_from_dict(item[name])
    return FeasiblePlannerArtifactV1(**item)  # type: ignore[arg-type]


__all__ = [
    "A0_CONTROLLER",
    "A0_CONTROLLER_SHA256",
    "ACTION_LIBRARY",
    "ACTION_LIBRARY_SHA256",
    "ACTION_SOURCE_PHASE_ORDER",
    "ACTION_SOURCE_PHASES",
    "CANONICAL_PAYLOAD_DOMAINS",
    "CHECKPOINT_CONTRACT_SHA256_BY_RATE",
    "CLONE_ALLOWLIST_SHA256",
    "COMMITMENT_PROFILE_SHA256",
    "CONSTRAINTS_SHA256",
    "DETERMINISTIC_COUNTER_FIELDS",
    "EPISODE_CONFIG_SHA256_BY_RATE",
    "FAILURE_CODES",
    "FAILURE_PHASES",
    "MODE_PROTECTION_ORDER",
    "PLANNER_KIND",
    "PLANT_CONFIG_SHA256",
    "PRIVILEGED_INFORMATION",
    "PROTOCOL_ANCHOR_COMMIT",
    "PROTOCOL_BLOB_SHA256",
    "PROTOCOL_PARENT_COMMIT",
    "PROTOCOL_PATH",
    "PROVENANCE_GAPS",
    "REWARD_WEIGHTS_SHA256",
    "RUNNER_FAILURE_CODES",
    "RUNNER_FAILURE_PHASES",
    "SCHEMA_VERSION",
    "SOURCE_BOUNDARY_CLAIM",
    "ActionRecordV1",
    "CanonicalPayloadV1",
    "FeasiblePlannerArtifactV1",
    "FeasiblePlannerResultV1",
    "FinalScoreV1",
    "IncumbentComparisonV1",
    "PlannerFailureV1",
    "PlannerRunnerFailureV1",
    "PlannerSettingsV1",
    "ReplayEvidenceV1",
    "SearchDiagnosticsV1",
    "action_record_trace_payload",
    "artifact_preimage",
    "build_incumbent_comparison",
    "build_planner_settings",
    "build_search_diagnostics",
    "canonical_json",
    "canonical_payload",
    "canonical_payload_from_dict",
    "canonical_payload_preimage",
    "compare_action_lineage",
    "compare_final_scores",
    "compare_service_fractions",
    "core_result_preimage",
    "deterministic_counters_payload",
    "outcome_payload",
    "planner_settings_preimage",
    "planner_artifact_from_dict",
    "planner_result_from_dict",
    "raw_canonical_sha256",
    "require_planner_admission",
    "revalidate_planner_settings",
    "replay_binding_payload",
    "request_trace_payload",
    "require_sha256",
    "runtime_payload",
    "sanitize_failure_message",
    "source_snapshot_payload",
    "source_trace_payload",
]

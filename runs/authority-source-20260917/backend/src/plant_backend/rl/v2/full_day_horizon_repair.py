"""Outcome-independent contract and analysis for the M24 horizon repair.

This module deliberately performs no I/O and imports no experiment runner.  It freezes the
scientific contract, validates the two evidence ledgers that are new for M24, and evaluates
the preregistered F1--F8 rules from detached values.  Environment construction, reset, action
selection, artifact custody, and typed event decoding belong to separate admission/runner
modules so importing or testing this file cannot open a protected outcome.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from numbers import Integral, Real
import statistics
from typing import Any, ClassVar, Literal, cast

from plant_sim import PlantConfig

from .control import ControllerObservationV2
from .economics import ECONOMIC_OBJECTIVE_VERSION, EconomicAssumptionsV1
from .economic_robustness import ECONOMIC_COMPONENT_NAMES
from .mpc_execution import SERVICE_MPC_OBSERVATION_DOMAIN, public_observation_sha256
from .mpc_forecast import (
    FORECAST_TRAJECTORY_DOMAIN,
    PUBLIC_FORECAST_ISSUE_DOMAIN,
    project_issued_mean_solar,
    public_forecast_issue_sha256,
)
from .mpc_schema import (
    SERVICE_MPC_SETPOINT_CONTROLLER_ID,
    ServiceMPCSetpointSettingsV1,
    canonical_json,
    domain_sha256,
    require_sha256,
)
from .weather import ForecastWindowV2, PublicForecastIssueV2


FULL_DAY_HORIZON_REPAIR_FORMAT = "full-day-horizon-repair-v1"
FULL_DAY_HORIZON_REPAIR_CELL_FORMAT = "full-day-horizon-repair-cell-v1"
FULL_DAY_HORIZON_REPAIR_PROTOCOL_PATH = (
    "docs/research/FULL_DAY_HORIZON_REPAIR_PROTOCOL.md"
)
FULL_DAY_HORIZON_REPAIR_PROTOCOL_SHA256 = (
    "e203d1bb12246bc43278b686779afdfde310f8b56e487dac7f9d05947079e0e1"
)
FULL_DAY_HORIZON_REPAIR_CLARIFICATION_001_PATH = (
    "docs/research/FULL_DAY_HORIZON_REPAIR_CLARIFICATION_001.md"
)
FULL_DAY_HORIZON_REPAIR_CLARIFICATION_001_SHA256 = (
    "8dc341b142a41214b7845b0f5debab29b7f23751ff8f7cd71ab68ac30f6b7ce3"
)
FULL_DAY_HORIZON_REPAIR_CLARIFICATION_002_PATH = (
    "docs/research/FULL_DAY_HORIZON_REPAIR_CLARIFICATION_002.md"
)
FULL_DAY_HORIZON_REPAIR_CLARIFICATION_002_SHA256 = (
    "e2ed43908242a228669859fb7111e4b4f24513d23f467daf60b8c9b1f7692966"
)
FULL_DAY_HORIZON_REPAIR_LOCK_PATH = "uv.lock"
FULL_DAY_HORIZON_REPAIR_LOCK_SHA256 = (
    "080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd"
)

HISTORICAL_EXECUTED_REVISION = "7b1b656964dcac4a25c6f5d4d89fed8abf4d4e0e"
HISTORICAL_WORKTREE_IDENTITY_SHA256 = (
    "eb34c2f8415f865ca6626a5e37e276441f3401e6d2c8cfdda3d4a13b82c5bbd6"
)
HISTORICAL_CONTRACT_SHA256 = (
    "6f789629393188f406c929305721aa257e33e329fc3b5b85da9e5d62d77d7d38"
)
HISTORICAL_RAW_REPORT_SHA256 = (
    "26c9ab4a85693dab05d93f7a1b65e86564c74bfe667ee54813901a466d1e0755"
)
HISTORICAL_REPORT_DOMAIN_SHA256 = (
    "e0fe451861cda01f0a788adafed9bf2767f778d0141d7c76147b6eb0f3831f7f"
)
HISTORICAL_DETERMINISTIC_EVIDENCE_SHA256 = (
    "cff97bfc678f0b7ca47c516f3f7a80e39434aae83a6597576b1539eba5110b96"
)

FULL_DAY_HORIZON_REPAIR_TASK_HORIZON_H = 24.0
FULL_DAY_HORIZON_REPAIR_M6_PLANNING_HORIZON_H = 6.0
FULL_DAY_HORIZON_REPAIR_M24_PLANNING_HORIZON_H = 24.0
FULL_DAY_HORIZON_REPAIR_INTERVAL_S = 900.0
FULL_DAY_HORIZON_REPAIR_PHYSICS_STEP_S = 60.0
FULL_DAY_HORIZON_REPAIR_RATES_KG_H = (35.0, 42.0)
FULL_DAY_HORIZON_REPAIR_SEEDS = (20, 21, 22, 23, 24)
FULL_DAY_HORIZON_REPAIR_M6_SETTINGS_SHA256 = (
    "28dd0dbbd732f4defcc700e3fcb91e59b2d14d6df7bac728963090e37ba48b0f"
)
FULL_DAY_HORIZON_REPAIR_M24_SETTINGS_SHA256 = (
    "67877c94438fc1bf1bd3ad564f17d860f77cea10b82f07bba0d52b6a6f6b1778"
)

FULL_DAY_HORIZON_REPAIR_EXPECTED_CELLS = 10
FULL_DAY_HORIZON_REPAIR_EXPECTED_DECISIONS_PER_CELL = 96
FULL_DAY_HORIZON_REPAIR_EXPECTED_PHYSICAL_MINUTES_PER_CELL = 1_440
FULL_DAY_HORIZON_REPAIR_EXPECTED_SETTLEMENTS_PER_CELL = 4
FULL_DAY_HORIZON_REPAIR_EXPECTED_SOLVER_STAGES_PER_CELL = 192
FULL_DAY_HORIZON_REPAIR_EXPECTED_TOTAL_DECISIONS = 960
FULL_DAY_HORIZON_REPAIR_EXPECTED_TOTAL_SOLVER_STAGES = 1_920
FULL_DAY_HORIZON_REPAIR_EXPECTED_TOTAL_PHYSICAL_MINUTES = 14_400

FULL_DAY_HORIZON_REPAIR_SERVICE_TOLERANCE_KG = 1e-6
FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE = 1e-8
FULL_DAY_HORIZON_REPAIR_VALUE_SEPARATION_EUR = 1.0
FULL_DAY_HORIZON_REPAIR_REQUIRED_CELLS_PER_RATE = 4
FULL_DAY_HORIZON_REPAIR_MULTIPLIERS = (0.0, 0.5, 1.0, 2.0)
FULL_DAY_HORIZON_REPAIR_CONTROLLER_P99_LIMIT_NS = 2_000_000_000
FULL_DAY_HORIZON_REPAIR_CONTROLLER_MAX_LIMIT_NS = 5_000_000_000
FULL_DAY_HORIZON_REPAIR_DISPATCHER_P99_LIMIT_NS = 5_000_000
FULL_DAY_HORIZON_REPAIR_DISPATCHER_MAX_LIMIT_NS = 25_000_000
FULL_DAY_HORIZON_REPAIR_SCIPY_VERSION = "1.18.1"
FULL_DAY_HORIZON_REPAIR_HIGHS_VERSION = "1.12.0"

FULL_DAY_HORIZON_REPAIR_CLAIM_BOUNDARY = (
    "Contaminated development mechanism only; no RL, controller-superiority, "
    "calibrated-economic, held-out, site, commercial, savings, commission, or "
    "generalization claim."
)

RAW_DRIVER_MATERIALITY: dict[str, float] = {
    "accepted_solar_kwh": 1.0,
    "accepted_process_energy_kwh": 1.0,
    "battery_throughput_kwh": 1.0,
    "electrolyser_equivalent_operating_h": 0.01,
    "terminal_ch4_equivalent_deficit_kg": 0.001,
    "terminal_battery_deficit_kwh": 0.1,
}
NONTERMINAL_DRIVER_NAMES = (
    "accepted_solar_kwh",
    "battery_throughput_kwh",
    "electrolyser_equivalent_operating_h",
)

AVAILABLE = "available"
NOT_RETAINED = "not_retained"
NOT_APPLICABLE = "not_applicable"
Availability = Literal["available", "not_retained", "not_applicable"]

FIRST_BOUNDARY_AUDIT_SCHEMA = "full-day-horizon-first-boundary-audit-v1"
FIRST_BOUNDARY_AUDIT_DOMAIN = FIRST_BOUNDARY_AUDIT_SCHEMA
INTERVAL_LEDGER_SCHEMA = "full-day-horizon-interval-ledger-v1"
INTERVAL_LEDGER_DOMAIN = INTERVAL_LEDGER_SCHEMA
INTERVAL_LEDGER_ROW_SCHEMA = "full-day-horizon-interval-row-v1"
INTERVAL_LEDGER_ROW_DOMAIN = INTERVAL_LEDGER_ROW_SCHEMA
MEASUREMENT_SCHEMA = "full-day-horizon-measurement-v1"


def _finite(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}")
    return number


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number


def _boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return cast(Mapping[str, Any], value)


def _sequence(name: str, value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return cast(Sequence[object], value)


def _strict_copy(name: str, value: object) -> Any:
    """Detach a JSON value and reject non-finite/non-serializable content."""

    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as reason:
        raise ValueError(f"{name} must be strict finite JSON") from reason


def _close(left: object, right: object, *, tolerance: float) -> bool:
    difference = abs(_finite("left value", left) - _finite("right value", right))
    return difference <= tolerance or math.isclose(
        difference,
        tolerance,
        rel_tol=0.0,
        abs_tol=max(1e-15, 8.0 * math.ulp(tolerance)),
    )


def _at_least_with_tolerance(
    candidate: object, reference: object, *, tolerance: float
) -> bool:
    candidate_value = _finite("candidate", candidate)
    reference_value = _finite("reference", reference)
    return candidate_value >= reference_value or _close(
        candidate_value, reference_value, tolerance=tolerance
    )


def _at_most_with_tolerance(
    candidate: object, reference: object, *, tolerance: float
) -> bool:
    candidate_value = _finite("candidate", candidate)
    reference_value = _finite("reference", reference)
    return candidate_value <= reference_value or _close(
        candidate_value, reference_value, tolerance=tolerance
    )


def _require_exact_keys(
    name: str, value: Mapping[str, Any], expected: set[str]
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{name} fields differ; missing={missing}, extra={extra}")


@dataclass(frozen=True, slots=True, order=True)
class FullDayHorizonRepairCellV1:
    """One full-day rate/seed cell; canonical order is rate then seed."""

    delivery_rate_kg_h: float
    seed: int

    schema: ClassVar[str] = "full-day-horizon-repair-cell-spec-v1"

    def __post_init__(self) -> None:
        rate = _finite("cell delivery_rate_kg_h", self.delivery_rate_kg_h, minimum=0.0)
        if rate <= 0.0:
            raise ValueError("cell delivery rate must be positive")
        seed = _integer("cell seed", self.seed)
        object.__setattr__(self, "delivery_rate_kg_h", rate)
        object.__setattr__(self, "seed", seed)

    @property
    def horizon_h(self) -> float:
        return FULL_DAY_HORIZON_REPAIR_TASK_HORIZON_H

    @property
    def id(self) -> str:
        return f"h24-r{self.delivery_rate_kg_h:g}-s{self.seed}"

    @property
    def historical_position(self) -> int | None:
        try:
            rate_index = FULL_DAY_HORIZON_REPAIR_RATES_KG_H.index(
                self.delivery_rate_kg_h
            )
            seed_index = FULL_DAY_HORIZON_REPAIR_SEEDS.index(self.seed)
        except ValueError:
            return None
        return 11 + rate_index * len(FULL_DAY_HORIZON_REPAIR_SEEDS) + seed_index

    @property
    def historical_file_name(self) -> str | None:
        position = self.historical_position
        return None if position is None else f"{position:02d}_{self.id}.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "id": self.id,
            "horizon_h": self.horizon_h,
            "delivery_rate_kg_h": self.delivery_rate_kg_h,
            "seed": self.seed,
            "expected_decisions": FULL_DAY_HORIZON_REPAIR_EXPECTED_DECISIONS_PER_CELL,
            "expected_physical_minutes": (
                FULL_DAY_HORIZON_REPAIR_EXPECTED_PHYSICAL_MINUTES_PER_CELL
            ),
            "expected_settlements": (
                FULL_DAY_HORIZON_REPAIR_EXPECTED_SETTLEMENTS_PER_CELL
            ),
            "expected_solver_stages": (
                FULL_DAY_HORIZON_REPAIR_EXPECTED_SOLVER_STAGES_PER_CELL
            ),
            "historical_position": self.historical_position,
            "historical_file_name": self.historical_file_name,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FullDayHorizonRepairCellV1":
        cell = cls(
            delivery_rate_kg_h=_finite(
                "cell delivery_rate_kg_h", raw.get("delivery_rate_kg_h"), minimum=0.0
            ),
            seed=_integer("cell seed", raw.get("seed")),
        )
        if dict(raw) != cell.to_dict():
            raise ValueError(
                "serialized horizon-repair cell differs from canonical form"
            )
        return cell


def registered_cells() -> tuple[FullDayHorizonRepairCellV1, ...]:
    return tuple(
        FullDayHorizonRepairCellV1(rate, seed)
        for rate in FULL_DAY_HORIZON_REPAIR_RATES_KG_H
        for seed in FULL_DAY_HORIZON_REPAIR_SEEDS
    )


REGISTERED_FULL_DAY_HORIZON_REPAIR_CELLS = registered_cells()


def historical_reference_file_names() -> tuple[str, ...]:
    """Return the complete 20-cell robustness directory order, not just M24 inputs."""

    names: list[str] = []
    position = 1
    for horizon in (6.0, 24.0):
        for rate in FULL_DAY_HORIZON_REPAIR_RATES_KG_H:
            for seed in FULL_DAY_HORIZON_REPAIR_SEEDS:
                names.append(f"{position:02d}_h{horizon:g}-r{rate:g}-s{seed}.json")
                position += 1
    return tuple(names)


HISTORICAL_REFERENCE_FILE_NAMES = historical_reference_file_names()
HISTORICAL_ANALYSIS_REFERENCE_FILE_NAMES = HISTORICAL_REFERENCE_FILE_NAMES[10:]


def m6_settings_for_cell(
    cell: FullDayHorizonRepairCellV1,
) -> dict[str, Any]:
    del cell  # rate and seed cannot alter controller settings
    settings = ServiceMPCSetpointSettingsV1(
        task_horizon_h=FULL_DAY_HORIZON_REPAIR_TASK_HORIZON_H,
        planning_horizon_h=FULL_DAY_HORIZON_REPAIR_M6_PLANNING_HORIZON_H,
        interval_s=FULL_DAY_HORIZON_REPAIR_INTERVAL_S,
    )
    if settings.sha256 != FULL_DAY_HORIZON_REPAIR_M6_SETTINGS_SHA256:
        raise RuntimeError("frozen M6 settings identity drifted")
    return settings.to_dict()


def m24_settings_for_cell(
    cell: FullDayHorizonRepairCellV1,
) -> dict[str, Any]:
    del cell  # rate and seed cannot alter controller settings
    settings = ServiceMPCSetpointSettingsV1(
        task_horizon_h=FULL_DAY_HORIZON_REPAIR_TASK_HORIZON_H,
        planning_horizon_h=FULL_DAY_HORIZON_REPAIR_M24_PLANNING_HORIZON_H,
        interval_s=FULL_DAY_HORIZON_REPAIR_INTERVAL_S,
    )
    if settings.sha256 != FULL_DAY_HORIZON_REPAIR_M24_SETTINGS_SHA256:
        raise RuntimeError("frozen M24 settings identity drifted")
    return settings.to_dict()


def verify_horizon_only_settings(
    m6: Mapping[str, Any], m24: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove the sole configurable intervention and enumerate derived changes."""

    fixture = FullDayHorizonRepairCellV1(35.0, 904)
    expected_m6 = m6_settings_for_cell(fixture)
    expected_m24 = m24_settings_for_cell(fixture)
    if dict(m6) != expected_m6:
        raise ValueError("M6 settings differ from the frozen full-day reference")
    if dict(m24) != expected_m24:
        raise ValueError("M24 settings differ from the frozen horizon repair")

    changed = tuple(
        key
        for key in expected_m6
        if canonical_json(expected_m6[key]) != canonical_json(expected_m24[key])
    )
    if changed != ("planning_horizon_h", "maximum_planning_intervals", "sha256"):
        raise RuntimeError("unexpected M6/M24 serialized settings difference")
    configurable = tuple(key for key in changed if key == "planning_horizon_h")
    derived = tuple(key for key in changed if key != "planning_horizon_h")
    return {
        "passed": configurable == ("planning_horizon_h",),
        "configurable_differences": list(configurable),
        "derived_differences": list(derived),
        "M6_planning_horizon_h": expected_m6["planning_horizon_h"],
        "M24_planning_horizon_h": expected_m24["planning_horizon_h"],
        "M6_settings_sha256": expected_m6["sha256"],
        "M24_settings_sha256": expected_m24["sha256"],
    }


REFERENCE_IDENTITY_AVAILABILITY: dict[str, dict[str, str]] = {
    "old_root_report": {
        "raw_file": AVAILABLE,
        "full_domain": AVAILABLE,
        "deterministic": AVAILABLE,
        "evidence_graph": NOT_APPLICABLE,
    },
    "old_cell": {
        "raw_file": AVAILABLE,
        "full_domain": AVAILABLE,
        "deterministic": AVAILABLE,
        "evidence_graph": NOT_APPLICABLE,
    },
    "H": {
        "raw_file": NOT_APPLICABLE,
        "full_domain": AVAILABLE,
        "deterministic": AVAILABLE,
        "evidence_graph": NOT_RETAINED,
    },
    "M6": {
        "raw_file": NOT_APPLICABLE,
        "full_domain": AVAILABLE,
        "deterministic": AVAILABLE,
        "evidence_graph": AVAILABLE,
    },
    "M24": {
        "raw_file": AVAILABLE,
        "full_domain": AVAILABLE,
        "deterministic": AVAILABLE,
        "evidence_graph": AVAILABLE,
    },
    "new_root_report": {
        "raw_file": AVAILABLE,
        "full_domain": AVAILABLE,
        "deterministic": AVAILABLE,
        "evidence_graph": NOT_APPLICABLE,
    },
}

LEGACY_MEASUREMENT_AVAILABILITY: dict[str, dict[str, str]] = {
    "H": {
        "aggregate_service": AVAILABLE,
        "aggregate_physical": AVAILABLE,
        "aggregate_economic": AVAILABLE,
        "settlement_deliveries": NOT_RETAINED,
        "post_settlement_states": NOT_RETAINED,
        "mpc_first_boundary": NOT_APPLICABLE,
        "mpc_solve_and_residual": NOT_APPLICABLE,
        "interval_economic_components": NOT_RETAINED,
    },
    "M6": {
        "aggregate_service": AVAILABLE,
        "aggregate_physical": AVAILABLE,
        "aggregate_economic": AVAILABLE,
        "settlement_deliveries": AVAILABLE,
        "post_settlement_states": AVAILABLE,
        "mpc_first_boundary": AVAILABLE,
        "mpc_solve_and_residual": AVAILABLE,
        "interval_economic_components": NOT_RETAINED,
    },
    "M24": {
        "aggregate_service": AVAILABLE,
        "aggregate_physical": AVAILABLE,
        "aggregate_economic": AVAILABLE,
        "settlement_deliveries": AVAILABLE,
        "post_settlement_states": AVAILABLE,
        "mpc_first_boundary": AVAILABLE,
        "mpc_solve_and_residual": AVAILABLE,
        "interval_economic_components": AVAILABLE,
    },
}


def measurement(
    availability: Availability,
    value: object = None,
    *,
    source_arm_sha256: str | None = None,
) -> dict[str, Any]:
    """Create one explicit available/missing compact measurement."""

    if availability not in {AVAILABLE, NOT_RETAINED, NOT_APPLICABLE}:
        raise ValueError("measurement availability is invalid")
    if availability == AVAILABLE:
        if value is None:
            raise ValueError("an available measurement requires a value")
        detached = _strict_copy("measurement value", value)
        if source_arm_sha256 is None:
            raise ValueError("an available measurement requires its source arm digest")
        digest = require_sha256("source_arm_sha256", source_arm_sha256)
    else:
        if value is not None or source_arm_sha256 is not None:
            raise ValueError(
                "an unavailable measurement cannot invent value or citation"
            )
        detached = None
        digest = None
    return {
        "schema": MEASUREMENT_SCHEMA,
        "availability": availability,
        "value": detached,
        "source_arm_sha256": digest,
    }


def verify_measurement(value: Mapping[str, Any]) -> None:
    item = _mapping("measurement", value)
    _require_exact_keys(
        "measurement",
        item,
        {"schema", "availability", "value", "source_arm_sha256"},
    )
    expected = measurement(
        cast(Availability, item["availability"]),
        item["value"],
        source_arm_sha256=item["source_arm_sha256"],
    )
    if dict(item) != expected:
        raise ValueError("measurement differs from canonical form")


def _validate_public_issue(issue: Mapping[str, Any]) -> None:
    _require_exact_keys(
        "public forecast issue",
        issue,
        {
            "issued_at_physics_boundary_step",
            "issued_at_offset_s",
            "issue_time_basis",
            "windows",
        },
    )
    if (
        _integer("forecast issue boundary", issue["issued_at_physics_boundary_step"])
        != 0
    ):
        raise ValueError("first public forecast issue must be issued at boundary zero")
    if (
        _finite("forecast issue offset", issue["issued_at_offset_s"], minimum=0.0)
        != 0.0
    ):
        raise ValueError("first public forecast issue must be issued at offset zero")
    if issue["issue_time_basis"] != "seconds since episode start":
        raise ValueError("public forecast issue time basis differs")
    windows = _sequence("public forecast windows", issue["windows"])
    expected_windows = (
        ("0-1h", 0.0, 3_600.0),
        ("1-6h", 3_600.0, 21_600.0),
        ("6-12h", 21_600.0, 43_200.0),
        ("12-24h", 43_200.0, 86_400.0),
    )
    if len(windows) != len(expected_windows):
        raise ValueError("public forecast issue must contain four windows")
    for ordinal, (raw, expected) in enumerate(
        zip(windows, expected_windows, strict=True)
    ):
        window = _mapping(f"forecast window {ordinal}", raw)
        _require_exact_keys(
            f"forecast window {ordinal}",
            window,
            {
                "name",
                "start_offset_s_from_issue",
                "end_offset_s_from_issue",
                "offset_interval",
                "mean_peak_power_fraction",
                "p10_peak_power_fraction",
            },
        )
        name, start, end = expected
        if (
            window["name"] != name
            or _finite("window start", window["start_offset_s_from_issue"]) != start
            or _finite("window end", window["end_offset_s_from_issue"]) != end
            or window["offset_interval"] != "half-open [start, end)"
        ):
            raise ValueError("public forecast window order or geometry differs")
        mean = _finite("forecast mean", window["mean_peak_power_fraction"])
        p10 = _finite("forecast p10", window["p10_peak_power_fraction"])
        if not 0.0 <= p10 <= mean <= 1.0:
            raise ValueError("public forecast fractions are invalid")


def _decode_public_issue(issue: Mapping[str, Any]) -> PublicForecastIssueV2:
    """Reconstruct the typed issue from its complete retained public body."""

    _validate_public_issue(issue)
    windows = tuple(
        ForecastWindowV2(
            name=str(window["name"]),
            start_offset_s_from_issue=_finite(
                "forecast window start", window["start_offset_s_from_issue"]
            ),
            end_offset_s_from_issue=_finite(
                "forecast window end", window["end_offset_s_from_issue"]
            ),
            mean_peak_power_fraction=_finite(
                "forecast window mean", window["mean_peak_power_fraction"]
            ),
            p10_peak_power_fraction=_finite(
                "forecast window p10", window["p10_peak_power_fraction"]
            ),
        )
        for window in (
            _mapping("public forecast window", raw)
            for raw in _sequence("public forecast windows", issue["windows"])
        )
    )
    decoded = PublicForecastIssueV2(
        issued_at_physics_boundary_step=_integer(
            "forecast issue boundary", issue["issued_at_physics_boundary_step"]
        ),
        issued_at_offset_s=_finite(
            "forecast issue offset", issue["issued_at_offset_s"], minimum=0.0
        ),
        windows=windows,
    )
    if decoded.to_dict() != dict(issue):
        raise ValueError("public forecast issue differs from typed reconstruction")
    return decoded


def _validate_projection(
    projection: Mapping[str, Any],
    *,
    issue_sha256: str,
    expected_boundary_time_s: float,
) -> str:
    expected_keys = {
        "projection_version",
        "source_issue_sha256",
        "issued_at_physics_boundary_step",
        "issued_at_offset_s",
        "boundary_time_s",
        "interval_s",
        "peak_kw",
        "clear_sky_mean_peak_fractions",
        "available_peak_fractions",
        "available_kw",
        "interval_band_names",
        "bands",
        "sha256",
    }
    _require_exact_keys("forecast projection", projection, expected_keys)
    recorded = require_sha256("forecast projection sha256", projection["sha256"])
    body = dict(projection)
    del body["sha256"]
    if domain_sha256(FORECAST_TRAJECTORY_DOMAIN, body) != recorded:
        raise ValueError("forecast projection digest differs")
    if projection["source_issue_sha256"] != issue_sha256:
        raise ValueError("forecast projection belongs to another public issue")
    if (
        _integer(
            "projection issue boundary", projection["issued_at_physics_boundary_step"]
        )
        != 0
    ):
        raise ValueError("first forecast projection must begin at boundary zero")
    if _finite("projection issue offset", projection["issued_at_offset_s"]) != 0.0:
        raise ValueError("first forecast projection issue offset differs")
    if _finite("projection boundary time", projection["boundary_time_s"]) != _finite(
        "expected projection boundary time", expected_boundary_time_s, minimum=0.0
    ):
        raise ValueError("first forecast projection boundary time differs")
    if _finite("projection interval", projection["interval_s"], minimum=0.0) != 900.0:
        raise ValueError("forecast projection interval must be 900 seconds")
    if _finite("projection peak power", projection["peak_kw"], minimum=0.0) <= 0.0:
        raise ValueError("forecast projection peak power must be positive")
    interval_fields = (
        "clear_sky_mean_peak_fractions",
        "available_peak_fractions",
        "available_kw",
        "interval_band_names",
    )
    arrays = [
        _sequence(f"projection {key}", projection[key]) for key in interval_fields
    ]
    if any(len(array) != 96 for array in arrays):
        raise ValueError("forecast projection must contain exactly 96 intervals")
    if 96 * _finite("projection interval", projection["interval_s"]) != 86_400.0:
        raise ValueError("forecast projection must span exactly 24 hours")
    return str(recorded)


def _event_zero_binding(
    name: str,
    event: Mapping[str, Any],
    *,
    require_delivery_contract: bool,
) -> dict[str, Any]:
    if _integer(f"{name} decision index", event.get("decision_index")) != 0:
        raise ValueError(f"{name} must be decision zero")
    binding: dict[str, Any] = {
        "decision_index": 0,
        "event_sha256": require_sha256(f"{name} event sha256", event.get("sha256")),
        "observation_sha256": require_sha256(
            f"{name} observation sha256", event.get("observation_sha256")
        ),
        "public_forecast_sha256": require_sha256(
            f"{name} public forecast sha256", event.get("public_forecast_sha256")
        ),
        "projection_sha256": require_sha256(
            f"{name} projection sha256", event.get("projection_sha256")
        ),
    }
    if require_delivery_contract:
        if _integer(f"{name} horizon intervals", event.get("horizon_intervals")) != 96:
            raise ValueError("M24 event zero must plan all 96 intervals")
        deliveries = _sequence(f"{name} deliveries", event.get("deliveries"))
        if len(deliveries) != 4:
            raise ValueError(
                "M24 event zero must contain exactly four delivery targets"
            )
        normalized: list[dict[str, Any]] = []
        expected_sources = (
            "announced",
            "inferred_recurring",
            "inferred_recurring",
            "inferred_recurring",
        )
        for ordinal, raw in enumerate(deliveries):
            target = _mapping(f"M24 delivery target {ordinal}", raw)
            if (
                _integer("delivery ordinal", target.get("ordinal")) != ordinal
                or _integer("delivery due boundary", target.get("due_boundary"))
                != 24 * (ordinal + 1)
                or _finite("delivery due offset", target.get("due_offset_s"))
                != 21_600.0 * (ordinal + 1)
                or _finite("delivery request", target.get("requested_kg"), minimum=0.0)
                <= 0.0
                or target.get("source") != expected_sources[ordinal]
            ):
                raise ValueError("M24 event-zero delivery target contract differs")
            normalized.append(_strict_copy("delivery target", target))
        binding["horizon_intervals"] = 96
        binding["deliveries"] = normalized
    return binding


def build_first_boundary_audit(
    observation: ControllerObservationV2,
    *,
    config: PlantConfig,
    interval_s: float,
) -> dict[str, Any]:
    """Capture complete causal input/projection bodies before the first M24 action."""

    if not isinstance(observation, ControllerObservationV2):
        raise TypeError("first-boundary observation must be ControllerObservationV2")
    if not isinstance(config, PlantConfig):
        raise TypeError("first-boundary plant config must be PlantConfig")
    source_observation = observation
    interval = _finite("first-boundary interval_s", interval_s, minimum=0.0)
    if interval != FULL_DAY_HORIZON_REPAIR_INTERVAL_S:
        raise ValueError("first-boundary interval must be exactly 900 seconds")
    forecast = source_observation.forecast
    if forecast is None:
        raise ValueError("M24 first-boundary observation requires a public forecast")
    projected = project_issued_mean_solar(
        forecast,
        boundary_time_s=observation.state.boundary.time_s,
        config=config,
        interval_s=interval,
        energy_tolerance=1e-12,
    )
    observation_body = _mapping(
        "public observation",
        _strict_copy("public observation", source_observation.to_dict()),
    )
    issue = _mapping(
        "public forecast issue",
        _strict_copy("public forecast issue", observation_body["forecast"]),
    )
    projection = _mapping(
        "forecast projection", _strict_copy("forecast projection", projected.to_dict())
    )
    _require_exact_keys(
        "public observation",
        observation_body,
        {"state", "delivery", "forecast", "progress", "decision_interval_s"},
    )
    if observation_body["forecast"] != issue:
        raise ValueError("complete public observation embeds another forecast issue")
    if _finite("first observation progress", observation_body["progress"]) != 0.0:
        raise ValueError("first public observation progress must be zero")
    if _finite("decision interval", observation_body["decision_interval_s"]) != 900.0:
        raise ValueError("first public observation decision interval differs")
    _mapping("public observation state", observation_body["state"])
    _mapping("public observation delivery", observation_body["delivery"])
    _validate_public_issue(issue)
    observation_sha = domain_sha256(SERVICE_MPC_OBSERVATION_DOMAIN, observation_body)
    issue_sha = domain_sha256(PUBLIC_FORECAST_ISSUE_DOMAIN, issue)
    if observation_sha != public_observation_sha256(source_observation):
        raise RuntimeError("public observation hashing implementation drifted")
    if issue_sha != public_forecast_issue_sha256(forecast):
        raise RuntimeError("public forecast hashing implementation drifted")
    projection_sha = _validate_projection(
        projection,
        issue_sha256=issue_sha,
        expected_boundary_time_s=observation.state.boundary.time_s,
    )
    body = {
        "schema": FIRST_BOUNDARY_AUDIT_SCHEMA,
        "public_observation": dict(observation_body),
        "public_forecast_issue": dict(issue),
        "forecast_projection": dict(projection),
        "identities": {
            "public_observation_sha256": observation_sha,
            "public_forecast_issue_sha256": issue_sha,
            "forecast_projection_sha256": projection_sha,
        },
    }
    return {**body, "sha256": domain_sha256(FIRST_BOUNDARY_AUDIT_DOMAIN, body)}


def verify_first_boundary_audit(
    payload: Mapping[str, Any],
    *,
    event_zero: Mapping[str, Any],
    reference_event_zero: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute complete-domain hashes and all F4 event-zero bindings."""

    item = _mapping("first boundary audit", payload)
    _require_exact_keys(
        "first boundary audit",
        item,
        {
            "schema",
            "public_observation",
            "public_forecast_issue",
            "forecast_projection",
            "identities",
            "sha256",
        },
    )
    if item["schema"] != FIRST_BOUNDARY_AUDIT_SCHEMA:
        raise ValueError("first-boundary audit schema differs")
    observation = _mapping("public observation", item["public_observation"])
    issue = _mapping("public issue", item["public_forecast_issue"])
    projection = _mapping("projection", item["forecast_projection"])
    _require_exact_keys(
        "public observation",
        observation,
        {"state", "delivery", "forecast", "progress", "decision_interval_s"},
    )
    if observation["forecast"] != issue:
        raise ValueError("complete public observation embeds another forecast issue")
    if _finite("first observation progress", observation["progress"]) != 0.0:
        raise ValueError("first public observation progress must be zero")
    if _finite("decision interval", observation["decision_interval_s"]) != 900.0:
        raise ValueError("first public observation decision interval differs")
    decoded_issue = _decode_public_issue(issue)
    state = _mapping("public observation state", observation["state"])
    boundary = _mapping("public observation boundary", state.get("boundary"))
    boundary_time_s = _finite(
        "public observation boundary time", boundary.get("time_s"), minimum=0.0
    )
    reconstructed_projection = project_issued_mean_solar(
        decoded_issue,
        boundary_time_s=boundary_time_s,
        config=PlantConfig(),
        interval_s=FULL_DAY_HORIZON_REPAIR_INTERVAL_S,
        energy_tolerance=1e-12,
    ).to_dict()
    if reconstructed_projection != dict(projection):
        raise ValueError(
            "forecast projection differs from the retained public-issue reconstruction"
        )
    observation_sha = domain_sha256(SERVICE_MPC_OBSERVATION_DOMAIN, observation)
    issue_sha = domain_sha256(PUBLIC_FORECAST_ISSUE_DOMAIN, issue)
    projection_sha = _validate_projection(
        projection,
        issue_sha256=issue_sha,
        expected_boundary_time_s=boundary_time_s,
    )
    identities = {
        "public_observation_sha256": observation_sha,
        "public_forecast_issue_sha256": issue_sha,
        "forecast_projection_sha256": projection_sha,
    }
    if item["identities"] != identities:
        raise ValueError("first-boundary audit identities differ")
    body = dict(item)
    recorded = require_sha256("first-boundary audit sha256", body.pop("sha256"))
    if domain_sha256(FIRST_BOUNDARY_AUDIT_DOMAIN, body) != recorded:
        raise ValueError("first-boundary audit digest differs")
    m24 = _event_zero_binding(
        "M24 event zero", event_zero, require_delivery_contract=True
    )
    m6 = _event_zero_binding(
        "M6 event zero", reference_event_zero, require_delivery_contract=False
    )
    expected = (observation_sha, issue_sha, projection_sha)
    for label, binding in (("M24", m24), ("M6", m6)):
        observed = (
            binding["observation_sha256"],
            binding["public_forecast_sha256"],
            binding["projection_sha256"],
        )
        if observed != expected:
            raise ValueError(f"{label} event zero is not bound to the complete audit")
    return {
        "passed": True,
        "identities": identities,
        "M6_event_zero": m6,
        "M24_event_zero": m24,
        "first_boundary_audit_sha256": recorded,
    }


_DELIVERY_ROW_FIELDS = {
    "commitments_settled",
    "requested_kg",
    "delivered_kg",
    "unmet_kg",
}
_ENERGY_ROW_FIELDS = {
    "accepted_solar_kwh",
    "accepted_process_energy_kwh",
    "curtailed_kwh",
    "battery_throughput_kwh",
    "battery_support_kwh",
    "electrolyser_equivalent_operating_h",
}
_LEDGER_BINDING_FIELDS = {
    "event_sha256",
    "observation_sha256",
    "public_forecast_sha256",
    "projection_sha256",
    "setpoints_sha256",
    "execution_sha256",
    "residual_sha256",
    "boundary_sha256",
    "next_observation_sha256",
    "next_boundary_sha256",
}
_ECONOMIC_SIGNAL_FIELDS = {
    "opportunity_price_eur_per_mwh",
    "effective_opportunity_price_eur_per_mwh",
    "accepted_solar_kwh",
    "delivered_kg",
    "unmet_kg",
    "battery_throughput_kwh",
    "battery_equivalent_full_cycles",
    "electrolyser_equivalent_operating_h",
    "methane_endowment_deficit_before_kg",
    "methane_endowment_deficit_after_kg",
    "battery_endowment_deficit_before_kwh",
    "battery_endowment_deficit_after_kwh",
    "solar_opportunity_cost_eur",
    "battery_throughput_reserve_eur",
    "electrolyser_life_reserve_eur",
}


def _verify_interval_economics(
    *,
    decision_index: int,
    delivery: Mapping[str, Any],
    economic_interval: Mapping[str, Any],
    reward_breakdown: Mapping[str, Any],
    energy: Mapping[str, Any],
) -> None:
    """Recompute the frozen objective from retained primitive interval signals."""

    assumptions = EconomicAssumptionsV1().validate()
    config = PlantConfig()
    if reward_breakdown["objective_version"] != ECONOMIC_OBJECTIVE_VERSION:
        raise ValueError("interval economic objective version differs")
    signals = _mapping("reward signals", reward_breakdown["signals"])
    _require_exact_keys("reward signals", signals, _ECONOMIC_SIGNAL_FIELDS)
    tariff = assumptions.tariff.observe(
        6.0 + decision_index * FULL_DAY_HORIZON_REPAIR_INTERVAL_S / 3_600.0
    ).to_dict()
    expected_economic_interval = {**dict(signals), "tariff": tariff}
    if dict(economic_interval) != expected_economic_interval:
        raise ValueError("economic interval differs from reward signals and tariff")

    opportunity_price = _finite(
        "opportunity price", signals["opportunity_price_eur_per_mwh"]
    )
    if not _close(
        opportunity_price,
        tariff["current_eur_per_mwh"],
        tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
    ):
        raise ValueError("reward opportunity price differs from frozen tariff")
    nonnegative_signal_names = _ECONOMIC_SIGNAL_FIELDS - {
        "opportunity_price_eur_per_mwh"
    }
    normalized_signals = {
        name: _finite(f"reward signal {name}", signals[name], minimum=0.0)
        for name in nonnegative_signal_names
    }
    for name in ("delivered_kg", "unmet_kg"):
        if not _close(
            normalized_signals[name],
            delivery[name],
            tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
        ):
            raise ValueError(f"reward signal {name} differs from interval delivery")
    for name in (
        "accepted_solar_kwh",
        "battery_throughput_kwh",
        "electrolyser_equivalent_operating_h",
    ):
        if not _close(
            normalized_signals[name],
            energy[name],
            tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
        ):
            raise ValueError(f"reward signal {name} differs from interval energy")

    effective_price = max(0.0, opportunity_price)
    expected_derived = {
        "effective_opportunity_price_eur_per_mwh": effective_price,
        "battery_equivalent_full_cycles": (
            normalized_signals["battery_throughput_kwh"]
            / (2.0 * config.battery.capacity_kwh)
        ),
        "solar_opportunity_cost_eur": (
            normalized_signals["accepted_solar_kwh"] * effective_price / 1_000.0
        ),
        "battery_throughput_reserve_eur": (
            normalized_signals["battery_throughput_kwh"]
            * assumptions.battery_throughput_eur_per_kwh
        ),
        "electrolyser_life_reserve_eur": (
            normalized_signals["electrolyser_equivalent_operating_h"]
            * config.electrolyser.rated_kw
            * assumptions.electrolyser_capital_eur_per_kw
            / assumptions.electrolyser_life_h
        ),
    }
    for name, expected in expected_derived.items():
        if not _close(
            normalized_signals[name],
            expected,
            tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
        ):
            raise ValueError(f"derived reward signal {name} differs")

    expected_components = {
        "delivered_service_value": (
            normalized_signals["delivered_kg"]
            * assumptions.delivered_service_value_eur_per_kg
        ),
        "unmet_delivery_penalty": -(
            normalized_signals["unmet_kg"]
            * assumptions.unmet_delivery_penalty_eur_per_kg
        ),
        "solar_opportunity_cost": -expected_derived["solar_opportunity_cost_eur"],
        "battery_throughput_reserve": -expected_derived[
            "battery_throughput_reserve_eur"
        ],
        "electrolyser_life_reserve": -expected_derived["electrolyser_life_reserve_eur"],
        "methane_endowment_liability_change": -(
            normalized_signals["methane_endowment_deficit_after_kg"]
            - normalized_signals["methane_endowment_deficit_before_kg"]
        )
        * assumptions.methane_endowment_value_eur_per_kg_ch4_equivalent,
        "battery_endowment_liability_change": -(
            normalized_signals["battery_endowment_deficit_after_kwh"]
            - normalized_signals["battery_endowment_deficit_before_kwh"]
        )
        * assumptions.battery_endowment_value_eur_per_kwh,
    }
    components = _mapping("reward components", reward_breakdown["components_eur"])
    for name, expected in expected_components.items():
        if not _close(
            components[name],
            expected,
            tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
        ):
            raise ValueError(f"economic component {name} differs from formula")
    if not _close(
        reward_breakdown["total_eur"],
        math.fsum(expected_components.values()),
        tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
    ):
        raise ValueError("economic reward total differs from formula")


def build_interval_ledger_row(
    *,
    decision_index: int,
    bindings: Mapping[str, Any],
    reward_eur: float,
    delivery: Mapping[str, Any],
    economic_interval: Mapping[str, Any],
    reward_breakdown: Mapping[str, Any],
    energy: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one hash-bound interval row without duplicating run event bodies."""

    index = _integer("interval decision index", decision_index)
    bound = _mapping("interval bindings", _strict_copy("interval bindings", bindings))
    _require_exact_keys("interval bindings", bound, _LEDGER_BINDING_FIELDS)
    for key in _LEDGER_BINDING_FIELDS:
        require_sha256(f"interval {key}", bound[key])
    delivery_row = _mapping(
        "interval delivery", _strict_copy("interval delivery", delivery)
    )
    _require_exact_keys("interval delivery", delivery_row, _DELIVERY_ROW_FIELDS)
    _integer("settlements in interval", delivery_row["commitments_settled"])
    for key in _DELIVERY_ROW_FIELDS - {"commitments_settled"}:
        _finite(f"interval delivery {key}", delivery_row[key], minimum=0.0)
    if not _close(
        _finite("interval delivered", delivery_row["delivered_kg"])
        + _finite("interval unmet", delivery_row["unmet_kg"]),
        delivery_row["requested_kg"],
        tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
    ):
        raise ValueError("interval delivery does not reconcile requested product")

    economic = _mapping(
        "economic interval", _strict_copy("economic interval", economic_interval)
    )
    breakdown = _mapping(
        "reward breakdown", _strict_copy("reward breakdown", reward_breakdown)
    )
    _require_exact_keys(
        "reward breakdown",
        breakdown,
        {"objective_version", "unit", "total_eur", "components_eur", "signals"},
    )
    if breakdown["unit"] != "EUR per decision interval":
        raise ValueError("reward-breakdown unit differs")
    total = _finite("reward breakdown total", breakdown["total_eur"])
    reward = _finite("interval reward", reward_eur)
    if not _close(
        reward,
        total,
        tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
    ):
        raise ValueError("interval scalar reward differs from its breakdown")
    components = _mapping("reward components", breakdown["components_eur"])
    if set(components) != set(ECONOMIC_COMPONENT_NAMES):
        raise ValueError("interval economic component set differs")
    component_sum = math.fsum(
        _finite(f"economic component {name}", components[name])
        for name in ECONOMIC_COMPONENT_NAMES
    )
    if not _close(
        component_sum,
        total,
        tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
    ):
        raise ValueError("interval economic components do not sum to reward")
    signals = _mapping("reward signals", breakdown["signals"])
    for key in ("delivered_kg", "unmet_kg"):
        if not _close(
            signals.get(key),
            delivery_row[key],
            tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
        ):
            raise ValueError(f"interval {key} differs between delivery and reward")
        if key in economic and not _close(
            economic[key],
            delivery_row[key],
            tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
        ):
            raise ValueError(f"interval {key} differs from economic record")

    energy_row = _mapping("interval energy", _strict_copy("interval energy", energy))
    _require_exact_keys("interval energy", energy_row, _ENERGY_ROW_FIELDS)
    for key in _ENERGY_ROW_FIELDS:
        value = _finite(f"interval energy {key}", energy_row[key], minimum=0.0)
        if key in economic and not _close(
            value,
            economic[key],
            tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
        ):
            raise ValueError(f"interval energy {key} differs from economic record")
        if key in signals and not _close(
            value,
            signals[key],
            tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
        ):
            raise ValueError(f"interval energy {key} differs from reward signals")
    _verify_interval_economics(
        decision_index=index,
        delivery=delivery_row,
        economic_interval=economic,
        reward_breakdown=breakdown,
        energy=energy_row,
    )
    body = {
        "schema": INTERVAL_LEDGER_ROW_SCHEMA,
        "decision_index": index,
        "bindings": dict(bound),
        "reward_eur": reward,
        "delivery": dict(delivery_row),
        "economic_interval": dict(economic),
        "reward_breakdown": dict(breakdown),
        "energy": dict(energy_row),
    }
    return {**body, "sha256": domain_sha256(INTERVAL_LEDGER_ROW_DOMAIN, body)}


def _ledger_totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    components = {
        name: math.fsum(
            _finite(
                f"row component {name}",
                _mapping(
                    "row reward components",
                    _mapping("row reward", row["reward_breakdown"])["components_eur"],
                )[name],
            )
            for row in rows
        )
        for name in ECONOMIC_COMPONENT_NAMES
    }
    result: dict[str, Any] = {
        "reward_eur": math.fsum(
            _finite("row reward", row["reward_eur"]) for row in rows
        ),
        "components_eur": components,
    }
    for key in _DELIVERY_ROW_FIELDS:
        if key == "commitments_settled":
            result[key] = sum(
                _integer(
                    "row commitments settled",
                    _mapping("row delivery", row["delivery"])[key],
                )
                for row in rows
            )
        else:
            result[key] = math.fsum(
                _finite(
                    f"row delivery {key}",
                    _mapping("row delivery", row["delivery"])[key],
                )
                for row in rows
            )
    for key in _ENERGY_ROW_FIELDS:
        result[key] = math.fsum(
            _finite(f"row energy {key}", _mapping("row energy", row["energy"])[key])
            for row in rows
        )
    return result


def _verify_ledger_totals_against_arm(
    totals: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    reward_trace_eur: Sequence[object],
    physical_metrics: Mapping[str, Any],
    economic_metrics: Mapping[str, Any],
    run_evidence: Mapping[str, Any],
) -> None:
    trace = [_finite("reward trace value", value) for value in reward_trace_eur]
    row_rewards = [_finite("row reward", row["reward_eur"]) for row in rows]
    if len(trace) != len(row_rewards) or any(
        not _close(
            left,
            right,
            tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
        )
        for left, right in zip(trace, row_rewards, strict=True)
    ):
        raise ValueError("interval ledger differs from reward trace")
    economic = _mapping("arm economic metrics", economic_metrics)
    physical = _mapping("arm physical metrics", physical_metrics)
    comparisons: dict[str, object] = {
        "reward_eur": economic["net_operating_value_eur"],
        "delivered_kg": economic["delivered_kg"],
        "unmet_kg": economic["unmet_kg"],
        "accepted_solar_kwh": economic["accepted_solar_kwh"],
        "battery_throughput_kwh": economic["battery_throughput_kwh"],
        "electrolyser_equivalent_operating_h": economic[
            "electrolyser_equivalent_operating_h"
        ],
        "requested_kg": physical["requested_kg"],
        "commitments_settled": physical["commitments_settled"],
        "accepted_process_energy_kwh": physical["accepted_process_energy_kwh"],
        "curtailed_kwh": physical["curtailed_kwh"],
        "battery_support_kwh": _mapping(
            "run applied energy",
            _mapping("run energy", run_evidence["energy_kwh"])["applied"],
        )["battery_support"],
    }
    for key, expected in comparisons.items():
        if key == "commitments_settled":
            if _integer(f"ledger {key}", totals[key]) != _integer(
                f"arm {key}", expected
            ):
                raise ValueError(f"arm {key} differs from interval ledger")
        elif not _close(
            totals[key],
            expected,
            tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
        ):
            raise ValueError(f"arm {key} differs from interval ledger")
    arm_components = _mapping("arm economic components", economic["components_eur"])
    for name in ECONOMIC_COMPONENT_NAMES:
        if not _close(
            _mapping("ledger components", totals["components_eur"])[name],
            arm_components[name],
            tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
        ):
            raise ValueError(f"arm component {name} differs from interval ledger")
    if not _close(
        math.fsum(trace),
        totals["reward_eur"],
        tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
    ):
        raise ValueError("reward trace sum differs from interval ledger")


def _ch4_equivalent_from_boundary(
    boundary: Mapping[str, Any], *, config: PlantConfig
) -> float:
    h2_ratio = _finite(
        "methanator H2 ratio",
        config.methanator.h2_kg_per_kg_ch4,
        minimum=1e-12,
    )
    co2_ratio = _finite(
        "methanator CO2 ratio",
        config.methanator.co2_kg_per_kg_ch4,
        minimum=1e-12,
    )
    ch4 = _finite("boundary CH4 inventory", boundary.get("ch4_level_kg"), minimum=0.0)
    h2 = _finite("boundary H2 inventory", boundary.get("h2_level_kg"), minimum=0.0)
    co2 = _finite("boundary CO2 inventory", boundary.get("co2_level_kg"), minimum=0.0)
    return ch4 + min(h2 / h2_ratio, co2 / co2_ratio)


def _commitment_equivalent_operating_h(
    applications: Sequence[object], *, assumptions: EconomicAssumptionsV1
) -> float:
    first = _mapping("first minute application", applications[0])
    last = _mapping("last minute application", applications[-1])
    before = _mapping("first commitment link", first.get("commitment")).get("before")
    after = _mapping("last commitment link", last.get("commitment")).get("after")
    before_link = _mapping("commitment before", before)
    after_link = _mapping("commitment after", after)
    before_record = before_link.get("record")
    after_record = after_link.get("record")
    if before_record is None or after_record is None:
        if before_record is not None or after_record is not None:
            raise ValueError("commitment enablement changed inside an interval")
        return 0.0
    start = _mapping("commitment before record", before_record)
    end = _mapping("commitment after record", after_record)
    for name in (
        "profile_id",
        "technology",
        "calibration_basis",
        "dynamics_version",
        "profile_sha256",
    ):
        if start.get(name) != end.get(name):
            raise ValueError("commitment profile changed inside an interval")

    def delta(name: str, *, integral: bool = False) -> float:
        if integral:
            difference = float(
                _integer(f"commitment after {name}", end.get(name))
                - _integer(f"commitment before {name}", start.get(name))
            )
        else:
            difference = _finite(f"commitment after {name}", end.get(name)) - _finite(
                f"commitment before {name}", start.get(name)
            )
        if difference < -FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE:
            raise ValueError(f"commitment counter {name} moved backwards")
        return max(0.0, float(difference))

    operating_h = delta("operating_time_s") / 3_600.0
    low_load_h = delta("low_load_time_s") / 3_600.0
    return (
        operating_h
        + low_load_h * (assumptions.electrolyser_low_load_wear_multiplier - 1.0)
        + delta("cold_starts", integral=True)
        * assumptions.cold_start_equivalent_operating_h
        + delta("hot_starts", integral=True)
        * assumptions.hot_start_equivalent_operating_h
        + delta("aborted_starts", integral=True)
        * assumptions.aborted_start_equivalent_operating_h
        + delta("forced_stops", integral=True)
        * assumptions.forced_stop_equivalent_operating_h
    )


def _verify_rows_against_run(
    rows: Sequence[Mapping[str, Any]], run_evidence: Mapping[str, Any]
) -> None:
    events = _sequence("run events", run_evidence.get("events"))
    if len(events) != len(rows):
        raise ValueError("run event cardinality differs from interval ledger")
    recorded_event_sha = _sequence(
        "run event digest ledger", run_evidence.get("event_sha256")
    )
    if list(recorded_event_sha) != [
        _mapping("row bindings", row["bindings"])["event_sha256"] for row in rows
    ]:
        raise ValueError("run event digest ledger differs from interval ledger")
    first_event = _mapping("first run event", events[0])
    first_execution = _mapping("first run execution", first_event.get("execution"))
    first_applications = _sequence(
        "first run applications", first_execution.get("applications")
    )
    if not first_applications:
        raise ValueError("first run execution has no applications")
    initial_application = _mapping("initial minute application", first_applications[0])
    initial_boundary = _mapping(
        "initial boundary",
        _mapping("initial boundary link", initial_application.get("boundary")).get(
            "record"
        ),
    )
    config = PlantConfig()
    assumptions = EconomicAssumptionsV1().validate()
    initial_ch4_equivalent = _ch4_equivalent_from_boundary(
        initial_boundary, config=config
    )
    initial_battery_energy = _finite(
        "initial battery energy",
        initial_boundary.get("battery_energy_kwh"),
        minimum=0.0,
    )
    for index, (row, raw_event) in enumerate(zip(rows, events, strict=True)):
        binding = _mapping("row bindings", row["bindings"])
        event = _mapping("run event", raw_event)
        if (
            _integer("run event decision", event.get("decision_index")) != index
            or event.get("sha256") != binding["event_sha256"]
            or event.get("observation_sha256") != binding["observation_sha256"]
            or event.get("public_forecast_sha256") != binding["public_forecast_sha256"]
            or event.get("projection_sha256") != binding["projection_sha256"]
        ):
            raise ValueError("interval ledger event binding differs from run evidence")
        setpoints = _mapping("run event setpoints", event.get("setpoints"))
        execution = _mapping("run event execution", event.get("execution"))
        residual = _mapping("run event residual", event.get("residual"))
        applications = _sequence(
            "run event applications", execution.get("applications")
        )
        if not applications:
            raise ValueError("run event execution has no applications")
        first_application = _mapping("first minute application", applications[0])
        first_boundary = _mapping(
            "first minute boundary", first_application.get("boundary")
        )
        first_boundary_record = _mapping(
            "first minute boundary record", first_boundary.get("record")
        )
        last_application = _mapping("last minute application", applications[-1])
        next_boundary = _mapping(
            "last minute next boundary", last_application.get("next_boundary")
        )
        next_boundary_record = _mapping(
            "last minute next boundary record", next_boundary.get("record")
        )
        terminal = _mapping("execution terminal", execution.get("terminal"))
        if (
            setpoints.get("sha256") != binding["setpoints_sha256"]
            or execution.get("sha256") != binding["execution_sha256"]
            or first_boundary.get("sha256") != binding["boundary_sha256"]
            or terminal.get("boundary_sha256") != binding["next_boundary_sha256"]
            or residual.get("sha256") != binding["residual_sha256"]
            or residual.get("next_observation_sha256")
            != binding["next_observation_sha256"]
        ):
            raise ValueError(
                "interval ledger transition binding differs from run evidence"
            )
        delivery = _mapping("row delivery", row["delivery"])
        expected_settlements = 1 if index in {23, 47, 71, 95} else 0
        if _integer("row commitments settled", delivery["commitments_settled"]) != (
            expected_settlements
        ):
            raise ValueError("settlement cadence differs from boundaries 24/48/72/96")
        due_targets = [
            _mapping("due delivery target", target)
            for target in _sequence("run event deliveries", event.get("deliveries"))
            if _integer(
                "delivery due boundary",
                _mapping("run delivery target", target).get("due_boundary"),
                minimum=1,
            )
            == 1
        ]
        if len(due_targets) != expected_settlements:
            raise ValueError(
                "event due-target cadence differs from interval settlement"
            )
        expected_requested = math.fsum(
            _finite("due delivery request", target.get("requested_kg"), minimum=0.0)
            for target in due_targets
        )
        if not _close(
            delivery["requested_kg"],
            expected_requested,
            tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
        ):
            raise ValueError(
                "interval requested delivery differs from due event target"
            )
        if not _close(
            delivery["delivered_kg"],
            residual.get("observed_delivered_kg"),
            tolerance=FULL_DAY_HORIZON_REPAIR_SERVICE_TOLERANCE_KG,
        ):
            raise ValueError("interval delivery differs from public residual")
        if not _close(
            delivery["unmet_kg"],
            max(
                0.0,
                expected_requested
                - _finite("interval delivered", delivery["delivered_kg"]),
            ),
            tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
        ):
            raise ValueError("interval unmet delivery differs from due target")
        applied = _mapping(
            "execution applied energy",
            _mapping("execution energy", execution.get("energy_kwh"))["applied"],
        )
        expected_energy = {
            "accepted_solar_kwh": _finite(
                "applied solar available", applied["solar_available"]
            )
            - _finite("applied solar curtailed", applied["solar_curtailed"]),
            "accepted_process_energy_kwh": math.fsum(
                _finite(f"applied process {name}", applied[name])
                for name in ("electrolyser_total", "dac", "methanator")
            ),
            "curtailed_kwh": _finite(
                "applied solar curtailed", applied["solar_curtailed"]
            ),
            "battery_throughput_kwh": _finite(
                "applied battery charge", applied["solar_to_battery"]
            )
            + _finite("applied battery support", applied["battery_support"]),
            "battery_support_kwh": _finite(
                "applied battery support", applied["battery_support"]
            ),
        }
        row_energy = _mapping("row energy", row["energy"])
        for energy_name, expected_value in expected_energy.items():
            if (
                expected_value < -FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE
                or not _close(
                    row_energy[energy_name],
                    max(0.0, expected_value),
                    tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
                )
            ):
                raise ValueError(
                    f"interval {energy_name} differs from execution energy"
                )
        signals = _mapping(
            "row reward signals",
            _mapping("row reward breakdown", row["reward_breakdown"])["signals"],
        )
        expected_exposure = _commitment_equivalent_operating_h(
            applications, assumptions=assumptions
        )
        if not _close(
            signals["electrolyser_equivalent_operating_h"],
            expected_exposure,
            tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
        ):
            raise ValueError("interval electrolyser exposure differs from commitment")
        before_ch4_equivalent = _ch4_equivalent_from_boundary(
            first_boundary_record, config=config
        )
        after_ch4_equivalent = _ch4_equivalent_from_boundary(
            next_boundary_record, config=config
        )
        expected_deficits = {
            "methane_endowment_deficit_before_kg": max(
                0.0, initial_ch4_equivalent - before_ch4_equivalent
            ),
            "methane_endowment_deficit_after_kg": max(
                0.0, initial_ch4_equivalent - after_ch4_equivalent
            ),
            "battery_endowment_deficit_before_kwh": max(
                0.0,
                initial_battery_energy
                - _finite(
                    "boundary battery energy",
                    first_boundary_record.get("battery_energy_kwh"),
                    minimum=0.0,
                ),
            ),
            "battery_endowment_deficit_after_kwh": max(
                0.0,
                initial_battery_energy
                - _finite(
                    "next boundary battery energy",
                    next_boundary_record.get("battery_energy_kwh"),
                    minimum=0.0,
                ),
            ),
        }
        for signal_name, expected_value in expected_deficits.items():
            if not _close(
                signals[signal_name],
                expected_value,
                tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
            ):
                raise ValueError(
                    f"interval {signal_name} differs from physical boundaries"
                )
    if (
        run_evidence.get("final_observation_sha256")
        != _mapping("last row bindings", rows[-1]["bindings"])[
            "next_observation_sha256"
        ]
    ):
        raise ValueError("run final observation differs from interval ledger")


def build_interval_ledger(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_evidence: Mapping[str, Any],
    reward_trace_eur: Sequence[object],
    physical_metrics: Mapping[str, Any],
    economic_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the ordered M24 interval ledger and reconcile every additive total."""

    expected = FULL_DAY_HORIZON_REPAIR_EXPECTED_DECISIONS_PER_CELL
    if len(rows) != expected:
        raise ValueError("interval ledger decision cardinality differs")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        row = _mapping("interval row", raw)
        rebuilt = build_interval_ledger_row(
            decision_index=_integer("row decision index", row.get("decision_index")),
            bindings=_mapping("row bindings", row.get("bindings")),
            reward_eur=_finite("row reward", row.get("reward_eur")),
            delivery=_mapping("row delivery", row.get("delivery")),
            economic_interval=_mapping(
                "row economic interval", row.get("economic_interval")
            ),
            reward_breakdown=_mapping(
                "row reward breakdown", row.get("reward_breakdown")
            ),
            energy=_mapping("row energy", row.get("energy")),
        )
        if dict(row) != rebuilt:
            raise ValueError("interval row differs from canonical reconstruction")
        if rebuilt["decision_index"] != index:
            raise ValueError("interval ledger rows are reordered or non-contiguous")
        normalized.append(rebuilt)
    for current, following in zip(normalized, normalized[1:]):
        left = _mapping("current row bindings", current["bindings"])
        right = _mapping("next row bindings", following["bindings"])
        if (
            left["next_observation_sha256"] != right["observation_sha256"]
            or left["next_boundary_sha256"] != right["boundary_sha256"]
        ):
            raise ValueError("interval ledger public/physical boundary chain is broken")
    totals = _ledger_totals(normalized)
    _verify_ledger_totals_against_arm(
        totals,
        normalized,
        reward_trace_eur=reward_trace_eur,
        physical_metrics=physical_metrics,
        economic_metrics=economic_metrics,
        run_evidence=run_evidence,
    )
    _verify_rows_against_run(normalized, run_evidence)
    body = {
        "schema": INTERVAL_LEDGER_SCHEMA,
        "expected_decisions": expected,
        "row_count": len(normalized),
        "first_observation_sha256": _mapping(
            "first row bindings", normalized[0]["bindings"]
        )["observation_sha256"],
        "final_observation_sha256": _mapping(
            "last row bindings", normalized[-1]["bindings"]
        )["next_observation_sha256"],
        "row_sha256": [row["sha256"] for row in normalized],
        "totals": totals,
        "rows": normalized,
    }
    return {**body, "sha256": domain_sha256(INTERVAL_LEDGER_DOMAIN, body)}


def verify_interval_ledger(
    payload: Mapping[str, Any],
    *,
    run_evidence: Mapping[str, Any],
    reward_trace_eur: Sequence[object],
    physical_metrics: Mapping[str, Any],
    economic_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    item = _mapping("interval ledger", payload)
    _require_exact_keys(
        "interval ledger",
        item,
        {
            "schema",
            "expected_decisions",
            "row_count",
            "first_observation_sha256",
            "final_observation_sha256",
            "row_sha256",
            "totals",
            "rows",
            "sha256",
        },
    )
    if item["schema"] != INTERVAL_LEDGER_SCHEMA:
        raise ValueError("interval-ledger schema differs")
    recorded_expected = _integer(
        "ledger expected decisions", item["expected_decisions"], minimum=1
    )
    if recorded_expected != FULL_DAY_HORIZON_REPAIR_EXPECTED_DECISIONS_PER_CELL:
        raise ValueError("interval-ledger expected decision count differs")
    rows = [
        _mapping("interval row", row) for row in _sequence("ledger rows", item["rows"])
    ]
    rebuilt = build_interval_ledger(
        rows,
        run_evidence=run_evidence,
        reward_trace_eur=reward_trace_eur,
        physical_metrics=physical_metrics,
        economic_metrics=economic_metrics,
    )
    if dict(item) != rebuilt:
        raise ValueError("interval ledger differs from canonical reconstruction")
    return dict(_mapping("interval ledger totals", rebuilt["totals"]))


_SERVICE_FIELDS = {
    "requested_kg",
    "delivered_kg",
    "unmet_kg",
    "delivery_fill",
    "commitments_settled",
    "commitments_met",
    "worst_commitment_fill",
}
_REQUIRED_PHYSICAL_FIELDS = {
    "accepted_process_energy_kwh",
    "curtailed_kwh",
    "battery_support_kwh",
    "terminal_battery_energy_kwh",
    "electrolyser_cold_starts",
    "electrolyser_hot_starts",
    "electrolyser_aborted_starts",
    "electrolyser_controlled_stops",
    "electrolyser_forced_stops",
    "electrolyser_operating_time_s",
    "electrolyser_nonproductive_energy_kwh",
    "corrected_physics_steps",
    "plant_refusals",
    "target_deficits",
    "guard_interventions",
    "shutdown_steps",
    "out_of_range_decisions",
}
_REQUIRED_ECONOMIC_FIELDS = {
    "net_operating_value_eur",
    "components_eur",
    "accepted_solar_kwh",
    "battery_throughput_kwh",
    "electrolyser_equivalent_operating_h",
    "terminal_ch4_equivalent_deficit_kg",
    "terminal_battery_deficit_kwh",
}
_COMPACT_ARM_FIELDS = {
    "service",
    "settlements",
    "post_settlement_states",
    "first_decision",
    "interval_economic_components",
    "final_settlement_balance",
    "physical",
    "economic",
    "latency",
}


def _validate_service(name: str, value: object) -> dict[str, Any]:
    service = _mapping(f"{name} service", value)
    _require_exact_keys(f"{name} service", service, _SERVICE_FIELDS)
    result: dict[str, float | int] = {
        key: _finite(f"{name} {key}", service[key], minimum=0.0)
        for key in ("requested_kg", "delivered_kg", "unmet_kg")
    }
    result["delivery_fill"] = _finite(
        f"{name} delivery fill", service["delivery_fill"], minimum=0.0
    )
    result.update(
        {
            "commitments_settled": _integer(
                f"{name} commitments settled", service["commitments_settled"]
            ),
            "commitments_met": _integer(
                f"{name} commitments met", service["commitments_met"]
            ),
            "worst_commitment_fill": _finite(
                f"{name} worst commitment fill",
                service["worst_commitment_fill"],
                minimum=0.0,
            ),
        }
    )
    if result["commitments_met"] > result["commitments_settled"]:
        raise ValueError(f"{name} met commitments exceed settled commitments")
    if not _close(
        result["delivered_kg"] + result["unmet_kg"],
        result["requested_kg"],
        tolerance=FULL_DAY_HORIZON_REPAIR_SERVICE_TOLERANCE_KG,
    ):
        raise ValueError(f"{name} aggregate delivery accounting does not close")
    expected_fill = (
        1.0
        if result["requested_kg"] == 0.0
        else result["delivered_kg"] / result["requested_kg"]
    )
    if not _close(result["delivery_fill"], expected_fill, tolerance=1e-12):
        raise ValueError(f"{name} aggregate delivery fill differs")
    return result


def _validate_settlements(name: str, record: Mapping[str, Any]) -> None:
    verify_measurement(record)
    if record["availability"] != AVAILABLE:
        return
    rows = _sequence(f"{name} settlement rows", record["value"])
    if len(rows) != 4:
        raise ValueError(f"{name} must report exactly four settlements")
    for index, raw in enumerate(rows, start=1):
        item = _mapping(f"{name} settlement {index}", raw)
        _require_exact_keys(
            f"{name} settlement {index}",
            item,
            {"ordinal", "requested_kg", "delivered_kg", "unmet_kg", "fill"},
        )
        if _integer("settlement ordinal", item["ordinal"], minimum=1) != index:
            raise ValueError(f"{name} settlement order differs")
        requested = _finite("settlement requested", item["requested_kg"], minimum=0.0)
        delivered = _finite("settlement delivered", item["delivered_kg"], minimum=0.0)
        unmet = _finite("settlement unmet", item["unmet_kg"], minimum=0.0)
        fill = _finite("settlement fill", item["fill"], minimum=0.0)
        if not _close(
            delivered + unmet,
            requested,
            tolerance=FULL_DAY_HORIZON_REPAIR_SERVICE_TOLERANCE_KG,
        ):
            raise ValueError(f"{name} settlement delivery does not close")
        expected_fill = 1.0 if requested == 0.0 else delivered / requested
        if not _close(fill, expected_fill, tolerance=1e-12):
            raise ValueError(f"{name} settlement fill differs")


def _validate_post_settlement_states(name: str, record: Mapping[str, Any]) -> None:
    verify_measurement(record)
    if record["availability"] != AVAILABLE:
        return
    states = _mapping(f"{name} post-settlement states", record["value"])
    if set(states) != {"2", "3"}:
        raise ValueError(f"{name} must report post-settlement states two and three")
    for settlement, raw in states.items():
        state = _mapping(f"{name} post-settlement state {settlement}", raw)
        _require_exact_keys(
            f"{name} post-settlement state {settlement}",
            state,
            {"h2_kg", "co2_kg", "ch4_kg", "battery_energy_kwh"},
        )
        for key, value in state.items():
            _finite(f"{name} {settlement} {key}", value, minimum=0.0)


def _validate_final_balance(name: str, record: Mapping[str, Any]) -> None:
    verify_measurement(record)
    if record["availability"] != AVAILABLE:
        return
    balance = _mapping(f"{name} fourth balance", record["value"])
    _require_exact_keys(
        f"{name} fourth balance",
        balance,
        {
            "ch4_before_final_interval_kg",
            "ch4_production_delta_kg",
            "ch4_after_settlement_kg",
            "residual_delivered_kg",
            "interval_delivered_kg",
        },
    )
    for key, value in balance.items():
        _finite(f"{name} fourth balance {key}", value, minimum=0.0)


def _validate_latency(name: str, value: object) -> dict[str, Any]:
    latency = _mapping(f"{name} latency", value)
    _require_exact_keys(
        f"{name} latency",
        latency,
        {"arm_elapsed_ns", "controller", "solver_stage", "dispatcher"},
    )
    result: dict[str, Any] = {
        "arm_elapsed_ns": _integer(
            f"{name} arm elapsed latency", latency["arm_elapsed_ns"]
        )
    }
    for channel in ("controller", "dispatcher"):
        summary = _mapping(f"{name} {channel} latency", latency[channel])
        _require_exact_keys(
            f"{name} {channel} latency", summary, {"count", "p99_ns", "maximum_ns"}
        )
        result[channel] = {
            "count": _integer(f"{name} {channel} latency count", summary["count"]),
            "p99_ns": _integer(f"{name} {channel} latency p99", summary["p99_ns"]),
            "maximum_ns": _integer(
                f"{name} {channel} latency maximum", summary["maximum_ns"]
            ),
        }
    solver = _mapping(f"{name} solver-stage latency", latency["solver_stage"])
    verify_measurement(solver)
    expected_solver_availability = NOT_APPLICABLE if name == "H" else AVAILABLE
    if solver["availability"] != expected_solver_availability:
        raise ValueError(f"{name} solver-stage latency availability differs")
    if solver["availability"] == AVAILABLE:
        summary = _mapping(f"{name} solver-stage latency value", solver["value"])
        _require_exact_keys(
            f"{name} solver-stage latency value",
            summary,
            {"count", "p99_ns", "maximum_ns"},
        )
        for key in ("count", "p99_ns", "maximum_ns"):
            _integer(f"{name} solver-stage latency {key}", summary[key])
    result["solver_stage"] = dict(solver)
    return result


def _expected_compact_availability(arm: str) -> dict[str, str]:
    return {
        "settlements": (NOT_RETAINED if arm == "H" else AVAILABLE),
        "post_settlement_states": (NOT_RETAINED if arm == "H" else AVAILABLE),
        "first_decision": NOT_APPLICABLE if arm == "H" else AVAILABLE,
        "interval_economic_components": (AVAILABLE if arm == "M24" else NOT_RETAINED),
        "final_settlement_balance": NOT_RETAINED if arm == "H" else AVAILABLE,
    }


def verify_compact_arm_measurements(arm: str, value: Mapping[str, Any]) -> None:
    if arm not in {"H", "M6", "M24"}:
        raise ValueError("compact measurement arm is invalid")
    item = _mapping(f"{arm} compact measurements", value)
    _require_exact_keys(f"{arm} compact measurements", item, _COMPACT_ARM_FIELDS)
    service = _validate_service(arm, item["service"])
    expected_availability = _expected_compact_availability(arm)
    for field, expected in expected_availability.items():
        record = _mapping(f"{arm} {field}", item[field])
        verify_measurement(record)
        if record["availability"] != expected:
            raise ValueError(f"{arm} {field} availability differs")
    _validate_settlements(arm, _mapping(f"{arm} settlements", item["settlements"]))
    _validate_post_settlement_states(
        arm, _mapping(f"{arm} post-settlement states", item["post_settlement_states"])
    )
    _validate_final_balance(
        arm, _mapping(f"{arm} final balance", item["final_settlement_balance"])
    )
    physical = _mapping(f"{arm} physical measurements", item["physical"])
    economic = _mapping(f"{arm} economic measurements", item["economic"])
    if not _REQUIRED_PHYSICAL_FIELDS <= set(physical):
        raise ValueError(f"{arm} physical measurements are incomplete")
    if not _REQUIRED_ECONOMIC_FIELDS <= set(economic):
        raise ValueError(f"{arm} economic measurements are incomplete")
    for key in _REQUIRED_PHYSICAL_FIELDS:
        _finite(f"{arm} physical {key}", physical[key], minimum=0.0)
    for key in _REQUIRED_ECONOMIC_FIELDS - {"components_eur"}:
        minimum = None if key == "net_operating_value_eur" else 0.0
        _finite(f"{arm} economic {key}", economic[key], minimum=minimum)
    components = _mapping(f"{arm} economic components", economic["components_eur"])
    if set(components) != set(ECONOMIC_COMPONENT_NAMES):
        raise ValueError(f"{arm} economic component set differs")
    for key, component in components.items():
        _finite(f"{arm} component {key}", component)
    if not _close(
        math.fsum(_finite("component", component) for component in components.values()),
        economic["net_operating_value_eur"],
        tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
    ):
        raise ValueError(f"{arm} economic components do not sum to net value")
    _validate_latency(arm, item["latency"])
    if arm != "H":
        settlements = _sequence(
            f"{arm} settlements",
            _mapping(f"{arm} settlement measurement", item["settlements"])["value"],
        )
        if not _close(
            math.fsum(
                _finite(
                    "settlement requested", _mapping("settlement", row)["requested_kg"]
                )
                for row in settlements
            ),
            service["requested_kg"],
            tolerance=FULL_DAY_HORIZON_REPAIR_SERVICE_TOLERANCE_KG,
        ) or not _close(
            math.fsum(
                _finite(
                    "settlement delivered", _mapping("settlement", row)["delivered_kg"]
                )
                for row in settlements
            ),
            service["delivered_kg"],
            tolerance=FULL_DAY_HORIZON_REPAIR_SERVICE_TOLERANCE_KG,
        ):
            raise ValueError(f"{arm} settlement totals differ from aggregate service")


_CITATION_REQUIRED_FIELDS: dict[str, set[str]] = {
    "root_report": {
        "raw_file_sha256",
        "report_sha256",
        "deterministic_evidence_sha256",
        "contract_sha256",
    },
    "historical_cell": {
        "file_name",
        "raw_file_sha256",
        "report_sha256",
        "deterministic_evidence_sha256",
    },
    "H": {"controller_id", "evidence_sha256", "deterministic_arm_sha256"},
    "M6": {
        "controller_id",
        "settings_sha256",
        "run_evidence_sha256",
        "evidence_graph_sha256",
        "deterministic_arm_sha256",
    },
    "M24": {
        "controller_id",
        "settings_sha256",
        "interval_evidence_sha256",
        "run_evidence_sha256",
        "evidence_graph_sha256",
    },
}


def verify_cell_citations(
    cell: FullDayHorizonRepairCellV1, citations: Mapping[str, Any]
) -> None:
    values = _mapping("cell citations", citations)
    if set(values) != set(_CITATION_REQUIRED_FIELDS):
        raise ValueError("cell citation objects differ")
    for object_name, required in _CITATION_REQUIRED_FIELDS.items():
        citation = _mapping(f"{object_name} citation", values[object_name])
        if set(citation) != required:
            raise ValueError(f"{object_name} citation fields differ")
        for key, raw in citation.items():
            if key.endswith("sha256"):
                require_sha256(f"{object_name} {key}", raw)
    historical = _mapping("historical cell citation", values["historical_cell"])
    if cell.historical_file_name is not None and (
        historical["file_name"] != cell.historical_file_name
    ):
        raise ValueError("historical cell citation filename differs")
    m6 = _mapping("M6 citation", values["M6"])
    if (
        m6["controller_id"] != SERVICE_MPC_SETPOINT_CONTROLLER_ID
        or m6["settings_sha256"] != FULL_DAY_HORIZON_REPAIR_M6_SETTINGS_SHA256
    ):
        raise ValueError("M6 citation controller/settings differ")
    m24 = _mapping("M24 citation", values["M24"])
    if (
        m24["controller_id"] != SERVICE_MPC_SETPOINT_CONTROLLER_ID
        or m24["settings_sha256"] != FULL_DAY_HORIZON_REPAIR_M24_SETTINGS_SHA256
    ):
        raise ValueError("M24 citation controller/settings differ")


_CELL_EVIDENCE_FIELDS = {
    "artifact_verified",
    "status",
    "terminated",
    "truncated",
    "failure",
    "completed_decisions",
    "physical_minutes",
    "settlements",
    "successful_solver_stages",
    "closed_transitions",
    "evidence_cardinality_exact",
    "fallback_count",
    "reset_identity_equal_reference",
    "reward_accounting_closed",
    "component_accounting_closed",
    "interval_accounting_closed",
    "terminal_liability_accounting_closed",
    "safety_counts_reconstructed",
    "shutdown_count",
    "out_of_range_count",
    "forced_stop_count",
    "chain_violation_count",
    "accounting_violation_count",
    "requested_target_overshoot_count",
    "reachable_tracking_violation_count",
    "support_authority_violation_count",
    "unattributed_target_deficit_count",
    "settings_sha256",
    "settings_horizon_only_valid",
    "first_boundary_audit_valid",
    "environment_identity_equal_m6",
    "scipy_version",
    "highs_version",
    "python_version_equal_historical",
    "platform_equal_historical",
}


def _validate_cell_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    item = _mapping("M24 cell evidence checks", value)
    _require_exact_keys("M24 cell evidence checks", item, _CELL_EVIDENCE_FIELDS)
    booleans = {
        key: _boolean(f"M24 evidence {key}", item[key])
        for key in (
            "artifact_verified",
            "terminated",
            "truncated",
            "closed_transitions",
            "evidence_cardinality_exact",
            "reset_identity_equal_reference",
            "reward_accounting_closed",
            "component_accounting_closed",
            "interval_accounting_closed",
            "terminal_liability_accounting_closed",
            "safety_counts_reconstructed",
            "settings_horizon_only_valid",
            "first_boundary_audit_valid",
            "environment_identity_equal_m6",
            "python_version_equal_historical",
            "platform_equal_historical",
        )
    }
    counts = {
        key: _integer(f"M24 evidence {key}", item[key])
        for key in (
            "completed_decisions",
            "physical_minutes",
            "settlements",
            "successful_solver_stages",
            "fallback_count",
            "shutdown_count",
            "out_of_range_count",
            "forced_stop_count",
            "chain_violation_count",
            "accounting_violation_count",
            "requested_target_overshoot_count",
            "reachable_tracking_violation_count",
            "support_authority_violation_count",
            "unattributed_target_deficit_count",
        )
    }
    return {**dict(item), **booleans, **counts}


def _available_value(name: str, record: object) -> Any:
    item = _mapping(name, record)
    verify_measurement(item)
    if item["availability"] != AVAILABLE:
        raise ValueError(f"{name} is unavailable")
    return item["value"]


def _economic_deltas(h: Mapping[str, Any], m24: Mapping[str, Any]) -> dict[str, Any]:
    h_economic = _mapping("H economic measurements", h["economic"])
    m_economic = _mapping("M24 economic measurements", m24["economic"])
    h_components = _mapping("H components", h_economic["components_eur"])
    m_components = _mapping("M24 components", m_economic["components_eur"])
    component_deltas = {
        name: _finite(f"M24 component {name}", m_components[name])
        - _finite(f"H component {name}", h_components[name])
        for name in ECONOMIC_COMPONENT_NAMES
    }
    default = math.fsum(component_deltas.values())
    direct = _finite("M24 net value", m_economic["net_operating_value_eur"]) - _finite(
        "H net value", h_economic["net_operating_value_eur"]
    )
    if not _close(
        default,
        direct,
        tolerance=FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
    ):
        raise ValueError(
            "M24-H economic components do not reproduce net-value contrast"
        )
    methane = component_deltas["methane_endowment_liability_change"]
    wear = (
        component_deltas["battery_throughput_reserve"]
        + component_deltas["electrolyser_life_reserve"]
    )
    solar = component_deltas["solar_opportunity_cost"]
    groups = {
        "methane_endowment": methane,
        "declared_wear": wear,
        "solar_opportunity": solar,
    }
    multiplier_deltas = {
        name: {
            f"{multiplier:g}": default + (multiplier - 1.0) * group
            for multiplier in FULL_DAY_HORIZON_REPAIR_MULTIPLIERS
        }
        for name, group in groups.items()
    }
    counterfactuals = {
        "default": default,
        "no_methane_endowment": default - methane,
        "no_declared_wear": default - wear,
        "service_plus_solar_only": math.fsum(
            component_deltas[name]
            for name in (
                "delivered_service_value",
                "unmet_delivery_penalty",
                "solar_opportunity_cost",
            )
        ),
    }
    default_sign = contrast_sign(default)
    flips = [
        f"counterfactuals.{name}"
        for name, value in counterfactuals.items()
        if name != "default" and contrast_sign(value) != default_sign
    ]
    flips.extend(
        f"{group}.{multiplier}"
        for group, values in multiplier_deltas.items()
        for multiplier, value in values.items()
        if multiplier != "1" and contrast_sign(value) != default_sign
    )
    return {
        "unit": "EUR; M24 minus H",
        "component_deltas_eur": component_deltas,
        "counterfactual_deltas_eur": counterfactuals,
        "one_factor_multiplier_deltas_eur": multiplier_deltas,
        "ranking_flips_from_default": flips,
    }


def contrast_sign(value: object) -> str:
    number = _finite("contrast", value)
    if number > FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE:
        return "positive"
    if number < -FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE:
        return "negative"
    return "tie"


def evaluate_full_day_horizon_repair_cell(
    cell: FullDayHorizonRepairCellV1,
    *,
    arms: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Any],
    citations: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute all intrinsic per-cell observations used by frozen F2--F8."""

    if not isinstance(cell, FullDayHorizonRepairCellV1):
        raise TypeError("cell must be FullDayHorizonRepairCellV1")
    if set(arms) != {"H", "M6", "M24"}:
        raise ValueError("cell measurements require exactly H, M6, and M24")
    for arm in ("H", "M6", "M24"):
        verify_compact_arm_measurements(arm, arms[arm])
    verify_cell_citations(cell, citations)
    checks = _validate_cell_evidence(evidence)
    h = _mapping("H measurements", arms["H"])
    m6 = _mapping("M6 measurements", arms["M6"])
    m24 = _mapping("M24 measurements", arms["M24"])
    h_service = _validate_service("H", h["service"])
    m6_service = _validate_service("M6", m6["service"])
    m24_service = _validate_service("M24", m24["service"])

    f2 = (
        checks["artifact_verified"]
        and checks["status"] == "complete"
        and checks["terminated"]
        and not checks["truncated"]
        and checks["failure"] is None
        and checks["completed_decisions"]
        == FULL_DAY_HORIZON_REPAIR_EXPECTED_DECISIONS_PER_CELL
        and checks["physical_minutes"]
        == FULL_DAY_HORIZON_REPAIR_EXPECTED_PHYSICAL_MINUTES_PER_CELL
        and checks["settlements"]
        == FULL_DAY_HORIZON_REPAIR_EXPECTED_SETTLEMENTS_PER_CELL
        and checks["successful_solver_stages"]
        == FULL_DAY_HORIZON_REPAIR_EXPECTED_SOLVER_STAGES_PER_CELL
        and checks["closed_transitions"]
        and checks["evidence_cardinality_exact"]
        and checks["fallback_count"] == 0
    )
    f3_zero_fields = (
        "shutdown_count",
        "out_of_range_count",
        "forced_stop_count",
        "chain_violation_count",
        "accounting_violation_count",
        "requested_target_overshoot_count",
        "reachable_tracking_violation_count",
        "support_authority_violation_count",
        "unattributed_target_deficit_count",
    )
    f3 = (
        checks["reset_identity_equal_reference"]
        and checks["reward_accounting_closed"]
        and checks["component_accounting_closed"]
        and checks["interval_accounting_closed"]
        and checks["terminal_liability_accounting_closed"]
        and checks["safety_counts_reconstructed"]
        and all(checks[name] == 0 for name in f3_zero_fields)
    )
    f4 = (
        checks["settings_sha256"] == FULL_DAY_HORIZON_REPAIR_M24_SETTINGS_SHA256
        and checks["settings_horizon_only_valid"]
        and checks["first_boundary_audit_valid"]
        and checks["environment_identity_equal_m6"]
        and checks["scipy_version"] == FULL_DAY_HORIZON_REPAIR_SCIPY_VERSION
        and checks["highs_version"] == FULL_DAY_HORIZON_REPAIR_HIGHS_VERSION
    )

    tolerance = FULL_DAY_HORIZON_REPAIR_SERVICE_TOLERANCE_KG
    no_regression = _at_least_with_tolerance(
        m24_service["delivered_kg"],
        m6_service["delivered_kg"],
        tolerance=tolerance,
    ) and _at_most_with_tolerance(
        m24_service["unmet_kg"],
        m6_service["unmet_kg"],
        tolerance=tolerance,
    )
    m24_settlements = _sequence(
        "M24 settlements", _available_value("M24 settlements", m24["settlements"])
    )
    fourth = _mapping("M24 fourth settlement", m24_settlements[3])
    positive_fourth_delivery = (
        _finite("M24 fourth delivery", fourth["delivered_kg"]) > tolerance
    )
    full_request = (
        _close(
            m24_service["delivered_kg"],
            m24_service["requested_kg"],
            tolerance=tolerance,
        )
        and _at_most_with_tolerance(m24_service["unmet_kg"], 0.0, tolerance=tolerance)
        and m24_service["commitments_met"] == 4
    )
    service_comparable_h = _close(
        m24_service["delivered_kg"],
        h_service["delivered_kg"],
        tolerance=tolerance,
    ) and _close(
        m24_service["unmet_kg"],
        h_service["unmet_kg"],
        tolerance=tolerance,
    )

    post_states = _mapping(
        "M24 post-settlement states",
        _available_value("M24 post-settlement states", m24["post_settlement_states"]),
    )
    state_three = _mapping("M24 post-settlement-three state", post_states["3"])
    fourth_request = _finite("fourth requested", fourth["requested_kg"], minimum=0.0)
    post_third_adequate = _at_least_with_tolerance(
        _finite("post-third methane", state_three["ch4_kg"], minimum=0.0),
        fourth_request,
        tolerance=tolerance,
    )
    balance = _mapping(
        "M24 fourth-settlement balance",
        _available_value(
            "M24 fourth-settlement balance", m24["final_settlement_balance"]
        ),
    )
    balance_derived = (
        _finite("CH4 before final interval", balance["ch4_before_final_interval_kg"])
        + _finite("CH4 final-interval production", balance["ch4_production_delta_kg"])
        - _finite("CH4 after fourth settlement", balance["ch4_after_settlement_kg"])
    )
    balance_agrees = all(
        _close(balance_derived, balance[key], tolerance=tolerance)
        for label, key in (
            ("residual fourth delivery", "residual_delivered_kg"),
            ("interval fourth delivery", "interval_delivered_kg"),
        )
    ) and _close(
        balance_derived,
        fourth["delivered_kg"],
        tolerance=tolerance,
    )

    economics = _economic_deltas(h, m24)
    default_delta = _finite(
        "default M24-H value delta",
        _mapping("economic counterfactuals", economics["counterfactual_deltas_eur"])[
            "default"
        ],
    )
    no_methane_delta = _finite(
        "no-methane M24-H value delta",
        _mapping("economic counterfactuals", economics["counterfactual_deltas_eur"])[
            "no_methane_endowment"
        ],
    )
    h_physical = _mapping("H physical", h["physical"])
    m_physical = _mapping("M24 physical", m24["physical"])
    h_economic = _mapping("H economic", h["economic"])
    m_economic = _mapping("M24 economic", m24["economic"])
    driver_deltas = {
        "accepted_solar_kwh": _finite(
            "M24 accepted solar", m_economic["accepted_solar_kwh"]
        )
        - _finite("H accepted solar", h_economic["accepted_solar_kwh"]),
        "accepted_process_energy_kwh": _finite(
            "M24 accepted process energy", m_physical["accepted_process_energy_kwh"]
        )
        - _finite(
            "H accepted process energy", h_physical["accepted_process_energy_kwh"]
        ),
        "battery_throughput_kwh": _finite(
            "M24 battery throughput", m_economic["battery_throughput_kwh"]
        )
        - _finite("H battery throughput", h_economic["battery_throughput_kwh"]),
        "electrolyser_equivalent_operating_h": _finite(
            "M24 electrolyser exposure",
            m_economic["electrolyser_equivalent_operating_h"],
        )
        - _finite(
            "H electrolyser exposure", h_economic["electrolyser_equivalent_operating_h"]
        ),
        "terminal_ch4_equivalent_deficit_kg": _finite(
            "M24 methane-equivalent deficit",
            m_economic["terminal_ch4_equivalent_deficit_kg"],
        )
        - _finite(
            "H methane-equivalent deficit",
            h_economic["terminal_ch4_equivalent_deficit_kg"],
        ),
        "terminal_battery_deficit_kwh": _finite(
            "M24 battery deficit", m_economic["terminal_battery_deficit_kwh"]
        )
        - _finite("H battery deficit", h_economic["terminal_battery_deficit_kwh"]),
    }
    material = {
        name: abs(driver_deltas[name]) > threshold
        for name, threshold in RAW_DRIVER_MATERIALITY.items()
    }
    task_expression = (
        service_comparable_h
        and abs(default_delta) > FULL_DAY_HORIZON_REPAIR_VALUE_SEPARATION_EUR
        and any(material.values())
    )
    nonterminal_expression = (
        service_comparable_h
        and abs(no_methane_delta) > FULL_DAY_HORIZON_REPAIR_VALUE_SEPARATION_EUR
        and any(material[name] for name in NONTERMINAL_DRIVER_NAMES)
    )
    latency = _validate_latency("M24", m24["latency"])
    controller_latency = _mapping("M24 controller latency", latency["controller"])
    dispatcher_latency = _mapping("M24 dispatcher latency", latency["dispatcher"])
    latency_pass = (
        controller_latency["p99_ns"] <= FULL_DAY_HORIZON_REPAIR_CONTROLLER_P99_LIMIT_NS
        and controller_latency["maximum_ns"]
        <= FULL_DAY_HORIZON_REPAIR_CONTROLLER_MAX_LIMIT_NS
        and dispatcher_latency["p99_ns"]
        <= FULL_DAY_HORIZON_REPAIR_DISPATCHER_P99_LIMIT_NS
        and dispatcher_latency["maximum_ns"]
        <= FULL_DAY_HORIZON_REPAIR_DISPATCHER_MAX_LIMIT_NS
    )
    return {
        "cell": cell.to_dict(),
        "measurements": _strict_copy("compact measurements", arms),
        "citations": _strict_copy("cell citations", citations),
        "intrinsic": {
            "F2_complete_execution_evidence": f2,
            "F3_identity_accounting_safety_authority": f3,
            "F4_horizon_intervention_fidelity": f4,
            "F5_no_regression_M6": no_regression,
            "F5_positive_fourth_delivery": positive_fourth_delivery,
            "F5_full_request": full_request,
            "F5_service_comparable_H": service_comparable_h,
            "F6_post_third_adequacy": post_third_adequate,
            "F6_fourth_material_balance": balance_agrees,
            "F7_task_expression": task_expression,
            "F7_nonterminal_expression": nonterminal_expression,
            "F7_both_expression_rules": task_expression and nonterminal_expression,
            "F7_economic_comparison_eligible": service_comparable_h,
            "F8_latency": latency_pass,
        },
        "service": {
            "H": h_service,
            "M6": m6_service,
            "M24": m24_service,
            "M24_minus_H_delivered_kg": (
                m24_service["delivered_kg"] - h_service["delivered_kg"]
            ),
            "M24_minus_H_unmet_kg": m24_service["unmet_kg"] - h_service["unmet_kg"],
            "M24_minus_M6_delivered_kg": (
                m24_service["delivered_kg"] - m6_service["delivered_kg"]
            ),
            "M24_minus_M6_unmet_kg": (m24_service["unmet_kg"] - m6_service["unmet_kg"]),
            "fourth_requested_kg": fourth_request,
            "fourth_delivered_kg": _finite("fourth delivery", fourth["delivered_kg"]),
        },
        "mechanism": {
            "post_third_ch4_kg": _finite("post-third methane", state_three["ch4_kg"]),
            "balance_derived_fourth_delivery_kg": balance_derived,
            "residual_fourth_delivery_kg": _finite(
                "residual fourth delivery", balance["residual_delivered_kg"]
            ),
            "interval_fourth_delivery_kg": _finite(
                "interval fourth delivery", balance["interval_delivered_kg"]
            ),
        },
        "economics": {
            **economics,
            "raw_driver_deltas": driver_deltas,
            "material_driver_flags": material,
        },
        "latency": latency,
        "evidence": dict(checks),
        "runtime_identity_limitations": [
            name
            for name, equal in (
                (
                    "python_version_differs_from_historical",
                    checks["python_version_equal_historical"],
                ),
                (
                    "platform_differs_from_historical",
                    checks["platform_equal_historical"],
                ),
            )
            if not equal
        ],
    }


_CUSTODY_FIELDS = {
    "reference_directory_verified",
    "reference_entry_names",
    "historical_executed_revision",
    "historical_worktree_identity_sha256",
    "historical_contract_sha256",
    "historical_raw_report_sha256",
    "historical_report_domain_sha256",
    "historical_deterministic_evidence_sha256",
    "protocol_sha256",
    "clarification_001_sha256",
    "clarification_002_sha256",
    "dependency_lock_sha256",
    "source_clean_start",
    "source_clean_end",
    "source_identity_stable",
    "reference_cell_ids",
    "historical_cell_citations",
    "undeclared_source_paths",
    "undeclared_destination_entries",
}


def evaluate_source_reference_custody(custody: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute F1 from detached admission/citation facts."""

    item = _mapping("source/reference custody", custody)
    _require_exact_keys("source/reference custody", item, _CUSTODY_FIELDS)
    entries = list(
        _sequence("reference directory entries", item["reference_entry_names"])
    )
    expected_entries = ["report.json", *HISTORICAL_REFERENCE_FILE_NAMES]
    cell_ids = list(_sequence("reference cell ids", item["reference_cell_ids"]))
    expected_cell_ids = [cell.id for cell in REGISTERED_FULL_DAY_HORIZON_REPAIR_CELLS]
    citations = list(
        _sequence("historical cell citations", item["historical_cell_citations"])
    )
    citation_names: list[str] = []
    citations_valid = len(citations) == 20
    for raw in citations:
        citation = _mapping("historical cell citation", raw)
        if set(citation) != {"file_name", "raw_file_sha256"}:
            citations_valid = False
            continue
        citation_names.append(str(citation["file_name"]))
        try:
            require_sha256("historical cell raw file", citation["raw_file_sha256"])
        except ValueError:
            citations_valid = False
    no_undeclared = (
        list(_sequence("undeclared source paths", item["undeclared_source_paths"]))
        == []
        and list(
            _sequence(
                "undeclared destination entries", item["undeclared_destination_entries"]
            )
        )
        == []
    )
    checks = {
        "reference_directory_verified": _boolean(
            "reference directory verified", item["reference_directory_verified"]
        ),
        "exact_21_entry_directory": entries == expected_entries,
        "historical_executed_revision": (
            item["historical_executed_revision"] == HISTORICAL_EXECUTED_REVISION
        ),
        "historical_worktree_identity": (
            item["historical_worktree_identity_sha256"]
            == HISTORICAL_WORKTREE_IDENTITY_SHA256
        ),
        "historical_contract": item["historical_contract_sha256"]
        == HISTORICAL_CONTRACT_SHA256,
        "historical_raw_report": item["historical_raw_report_sha256"]
        == HISTORICAL_RAW_REPORT_SHA256,
        "historical_report_domain": item["historical_report_domain_sha256"]
        == HISTORICAL_REPORT_DOMAIN_SHA256,
        "historical_deterministic_evidence": (
            item["historical_deterministic_evidence_sha256"]
            == HISTORICAL_DETERMINISTIC_EVIDENCE_SHA256
        ),
        "protocol": item["protocol_sha256"] == FULL_DAY_HORIZON_REPAIR_PROTOCOL_SHA256,
        "clarification_001": item["clarification_001_sha256"]
        == FULL_DAY_HORIZON_REPAIR_CLARIFICATION_001_SHA256,
        "clarification_002": item["clarification_002_sha256"]
        == FULL_DAY_HORIZON_REPAIR_CLARIFICATION_002_SHA256,
        "dependency_lock": item["dependency_lock_sha256"]
        == FULL_DAY_HORIZON_REPAIR_LOCK_SHA256,
        "source_clean_start": _boolean(
            "source clean start", item["source_clean_start"]
        ),
        "source_clean_end": _boolean("source clean end", item["source_clean_end"]),
        "source_identity_stable": _boolean(
            "source identity stable", item["source_identity_stable"]
        ),
        "ten_reference_cells_once_in_order": cell_ids == expected_cell_ids,
        "all_20_historical_cells_cited": citations_valid
        and citation_names == list(HISTORICAL_REFERENCE_FILE_NAMES),
        "no_undeclared_source_or_destination": no_undeclared,
    }
    return {"passed": all(checks.values()), "checks": checks}


def summarize_rate_values(
    values_by_seed: Sequence[tuple[int, float]],
    *,
    contrast: bool,
    expected_seed_ids: Sequence[int] = FULL_DAY_HORIZON_REPAIR_SEEDS,
) -> dict[str, Any]:
    """Summarize one rate only, retaining missingness and every tied extremum."""

    expected = tuple(_integer("expected seed", seed) for seed in expected_seed_ids)
    if len(set(expected)) != len(expected):
        raise ValueError("expected seed ids must be unique")
    parsed: dict[int, float] = {}
    for raw_seed, raw_value in values_by_seed:
        seed = _integer("series seed", raw_seed)
        if seed not in expected:
            raise ValueError("series contains an undeclared seed")
        if seed in parsed:
            raise ValueError("series contains a duplicate seed")
        parsed[seed] = _finite("series value", raw_value)
    ordered = [(seed, parsed[seed]) for seed in expected if seed in parsed]
    missing = [seed for seed in expected if seed not in parsed]
    values = [value for _, value in ordered]
    minimum = min(values) if values else None
    maximum = max(values) if values else None
    result: dict[str, Any] = {
        "values_by_seed": [{"seed": seed, "value": value} for seed, value in ordered],
        "expected_count": len(expected),
        "observed_count": len(ordered),
        "missing_count": len(missing),
        "missing_seed_ids": missing,
        "mean": None if not values else statistics.fmean(values),
        "population_std": None if not values else statistics.pstdev(values),
        "median": None if not values else statistics.median(values),
        "minimum": minimum,
        "maximum": maximum,
        "minimum_seed_ids": (
            []
            if minimum is None
            else [seed for seed, value in ordered if value == minimum]
        ),
        "maximum_seed_ids": (
            []
            if maximum is None
            else [seed for seed, value in ordered if value == maximum]
        ),
    }
    if contrast:
        signs = [contrast_sign(value) for value in values]
        result.update(
            {
                "positive_count": signs.count("positive"),
                "negative_count": signs.count("negative"),
                "tie_count": signs.count("tie"),
            }
        )
    else:
        result.update(
            {
                "positive_count": NOT_APPLICABLE,
                "negative_count": NOT_APPLICABLE,
                "tie_count": NOT_APPLICABLE,
            }
        )
    return result


def _analysis_cell(analysis: Mapping[str, Any]) -> FullDayHorizonRepairCellV1:
    return FullDayHorizonRepairCellV1.from_dict(
        _mapping("analysis cell", analysis["cell"])
    )


def _analysis_value(analysis: Mapping[str, Any], path: Sequence[str]) -> float:
    current: object = analysis
    for key in path:
        current = _mapping(f"analysis path {'.'.join(path)}", current)[key]
    return _finite(f"analysis value {'.'.join(path)}", current)


def _availability_summary(
    analyses: Sequence[Mapping[str, Any]], arm: str, field: str
) -> dict[str, Any]:
    states: list[dict[str, Any]] = []
    seen: set[int] = set()
    for analysis in analyses:
        seed = _analysis_cell(analysis).seed
        record = _mapping(
            f"{arm} {field} measurement",
            _mapping(
                f"{arm} compact measurements",
                _mapping("compact measurements", analysis["measurements"])[arm],
            )[field],
        )
        verify_measurement(record)
        states.append({"seed": seed, "availability": record["availability"]})
        seen.add(seed)
    return {
        "expected_count": len(FULL_DAY_HORIZON_REPAIR_SEEDS),
        "observed_count": len(states),
        "missing_count": len(FULL_DAY_HORIZON_REPAIR_SEEDS) - len(states),
        "missing_seed_ids": [
            seed for seed in FULL_DAY_HORIZON_REPAIR_SEEDS if seed not in seen
        ],
        "available_count": sum(item["availability"] == AVAILABLE for item in states),
        "not_retained_count": sum(
            item["availability"] == NOT_RETAINED for item in states
        ),
        "not_applicable_count": sum(
            item["availability"] == NOT_APPLICABLE for item in states
        ),
        "availability_by_seed": states,
    }


def _solver_stage_availability_summary(
    analyses: Sequence[Mapping[str, Any]], arm: str
) -> dict[str, Any]:
    states: list[dict[str, Any]] = []
    seen: set[int] = set()
    for analysis in analyses:
        seed = _analysis_cell(analysis).seed
        compact = _mapping(
            f"{arm} compact measurements",
            _mapping("compact measurements", analysis["measurements"])[arm],
        )
        latency = _mapping(f"{arm} latency", compact["latency"])
        record = _mapping(f"{arm} solver-stage latency", latency["solver_stage"])
        verify_measurement(record)
        states.append({"seed": seed, "availability": record["availability"]})
        seen.add(seed)
    return {
        "expected_count": len(FULL_DAY_HORIZON_REPAIR_SEEDS),
        "observed_count": len(states),
        "missing_count": len(FULL_DAY_HORIZON_REPAIR_SEEDS) - len(states),
        "missing_seed_ids": [
            seed for seed in FULL_DAY_HORIZON_REPAIR_SEEDS if seed not in seen
        ],
        "available_count": sum(item["availability"] == AVAILABLE for item in states),
        "not_retained_count": sum(
            item["availability"] == NOT_RETAINED for item in states
        ),
        "not_applicable_count": sum(
            item["availability"] == NOT_APPLICABLE for item in states
        ),
        "availability_by_seed": states,
    }


def _retained_measurements_by_seed(
    analyses: Sequence[Mapping[str, Any]], arm: str, field: str
) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    for analysis in analyses:
        compact = _mapping(
            f"{arm} compact measurements",
            _mapping("compact measurements", analysis["measurements"])[arm],
        )
        record = _mapping(f"{arm} {field}", compact[field])
        verify_measurement(record)
        retained.append(
            {
                "seed": _analysis_cell(analysis).seed,
                "availability": record["availability"],
                "value": _strict_copy(f"{arm} {field} value", record["value"]),
                "source_arm_sha256": record["source_arm_sha256"],
            }
        )
    return retained


def _registered_measurement_summaries(
    analyses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ("H", "M6", "M24"):
        arm_result: dict[str, Any] = {
            "availability": {
                field: _availability_summary(analyses, arm, field)
                for field in (
                    "settlements",
                    "post_settlement_states",
                    "first_decision",
                    "interval_economic_components",
                    "final_settlement_balance",
                )
            },
            "retained_values_by_seed": {
                field: _retained_measurements_by_seed(analyses, arm, field)
                for field in ("first_decision", "interval_economic_components")
            },
            "service": {},
            "physical": {},
            "economic": {},
            "settlements": {},
            "post_settlement_states": {},
            "final_settlement_balance": {},
            "latency": {
                "solver_stage_availability": _solver_stage_availability_summary(
                    analyses, arm
                )
            },
        }

        def compact(analysis: Mapping[str, Any]) -> Mapping[str, Any]:
            return _mapping(
                f"{arm} compact measurements",
                _mapping("compact measurements", analysis["measurements"])[arm],
            )

        for field in _SERVICE_FIELDS:
            arm_result["service"][field] = summarize_rate_values(
                [
                    (
                        _analysis_cell(item).seed,
                        _finite(
                            f"{arm} service {field}",
                            _mapping(f"{arm} service", compact(item)["service"])[field],
                        ),
                    )
                    for item in analyses
                ],
                contrast=False,
            )
        for field in sorted(_REQUIRED_PHYSICAL_FIELDS):
            arm_result["physical"][field] = summarize_rate_values(
                [
                    (
                        _analysis_cell(item).seed,
                        _finite(
                            f"{arm} physical {field}",
                            _mapping(f"{arm} physical", compact(item)["physical"])[
                                field
                            ],
                        ),
                    )
                    for item in analyses
                ],
                contrast=False,
            )
        for field in sorted(_REQUIRED_ECONOMIC_FIELDS - {"components_eur"}):
            arm_result["economic"][field] = summarize_rate_values(
                [
                    (
                        _analysis_cell(item).seed,
                        _finite(
                            f"{arm} economic {field}",
                            _mapping(f"{arm} economic", compact(item)["economic"])[
                                field
                            ],
                        ),
                    )
                    for item in analyses
                ],
                contrast=False,
            )
        arm_result["economic"]["components_eur"] = {
            component: summarize_rate_values(
                [
                    (
                        _analysis_cell(item).seed,
                        _finite(
                            f"{arm} component {component}",
                            _mapping(
                                f"{arm} components",
                                _mapping(f"{arm} economic", compact(item)["economic"])[
                                    "components_eur"
                                ],
                            )[component],
                        ),
                    )
                    for item in analyses
                ],
                contrast=False,
            )
            for component in ECONOMIC_COMPONENT_NAMES
        }
        for ordinal in range(4):
            key = str(ordinal + 1)
            arm_result["settlements"][key] = {}
            for field in ("requested_kg", "delivered_kg", "unmet_kg", "fill"):
                values: list[tuple[int, float]] = []
                for item in analyses:
                    record = _mapping(
                        f"{arm} settlements", compact(item)["settlements"]
                    )
                    if record["availability"] == AVAILABLE:
                        rows = _sequence(f"{arm} settlement rows", record["value"])
                        values.append(
                            (
                                _analysis_cell(item).seed,
                                _finite(
                                    f"{arm} settlement {key} {field}",
                                    _mapping("settlement row", rows[ordinal])[field],
                                ),
                            )
                        )
                arm_result["settlements"][key][field] = summarize_rate_values(
                    values, contrast=False
                )
        for settlement in ("2", "3"):
            arm_result["post_settlement_states"][settlement] = {}
            for field in ("h2_kg", "co2_kg", "ch4_kg", "battery_energy_kwh"):
                values = []
                for item in analyses:
                    record = _mapping(
                        f"{arm} post states", compact(item)["post_settlement_states"]
                    )
                    if record["availability"] == AVAILABLE:
                        states = _mapping(f"{arm} post-state values", record["value"])
                        values.append(
                            (
                                _analysis_cell(item).seed,
                                _finite(
                                    f"{arm} state {settlement} {field}",
                                    _mapping(
                                        "post-settlement state", states[settlement]
                                    )[field],
                                ),
                            )
                        )
                arm_result["post_settlement_states"][settlement][field] = (
                    summarize_rate_values(values, contrast=False)
                )
        for field in (
            "ch4_before_final_interval_kg",
            "ch4_production_delta_kg",
            "ch4_after_settlement_kg",
            "residual_delivered_kg",
            "interval_delivered_kg",
        ):
            values = []
            for item in analyses:
                record = _mapping(
                    f"{arm} final balance", compact(item)["final_settlement_balance"]
                )
                if record["availability"] == AVAILABLE:
                    balance = _mapping(f"{arm} final balance value", record["value"])
                    values.append(
                        (
                            _analysis_cell(item).seed,
                            _finite(f"{arm} final balance {field}", balance[field]),
                        )
                    )
            arm_result["final_settlement_balance"][field] = summarize_rate_values(
                values, contrast=False
            )
        arm_result["latency"]["arm_elapsed_ns"] = summarize_rate_values(
            [
                (
                    _analysis_cell(item).seed,
                    _finite(
                        f"{arm} arm elapsed latency",
                        _mapping(f"{arm} latency", compact(item)["latency"])[
                            "arm_elapsed_ns"
                        ],
                    ),
                )
                for item in analyses
            ],
            contrast=False,
        )
        for channel in ("controller", "dispatcher"):
            for field in ("count", "p99_ns", "maximum_ns"):
                arm_result["latency"][f"{channel}_{field}"] = summarize_rate_values(
                    [
                        (
                            _analysis_cell(item).seed,
                            _finite(
                                f"{arm} {channel} {field}",
                                _mapping(
                                    f"{arm} {channel} latency",
                                    _mapping(
                                        f"{arm} latency", compact(item)["latency"]
                                    )[channel],
                                )[field],
                            ),
                        )
                        for item in analyses
                    ],
                    contrast=False,
                )
        for field in ("count", "p99_ns", "maximum_ns"):
            values = []
            for item in analyses:
                latency = _mapping(f"{arm} latency", compact(item)["latency"])
                record = _mapping(
                    f"{arm} solver-stage latency", latency["solver_stage"]
                )
                if record["availability"] == AVAILABLE:
                    summary = _mapping(
                        f"{arm} solver-stage latency value", record["value"]
                    )
                    values.append(
                        (
                            _analysis_cell(item).seed,
                            _finite(
                                f"{arm} solver-stage latency {field}", summary[field]
                            ),
                        )
                    )
            arm_result["latency"][f"solver_stage_{field}"] = summarize_rate_values(
                values, contrast=False
            )
        result[arm] = arm_result
    return result


def _rate_summaries(
    rate: float, analyses: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    eligible_analyses = [
        analysis
        for analysis in analyses
        if _mapping("cell intrinsic observations", analysis["intrinsic"])[
            "F7_economic_comparison_eligible"
        ]
        is True
    ]
    paths: dict[str, tuple[bool, tuple[str, ...]]] = {
        "M6_unmet_kg": (False, ("service", "M6", "unmet_kg")),
        "M24_unmet_kg": (False, ("service", "M24", "unmet_kg")),
        "M24_fourth_delivered_kg": (False, ("service", "fourth_delivered_kg")),
        "M24_post_third_ch4_kg": (False, ("mechanism", "post_third_ch4_kg")),
        "M24_minus_H_delivered_kg": (
            True,
            ("service", "M24_minus_H_delivered_kg"),
        ),
        "M24_minus_H_unmet_kg": (True, ("service", "M24_minus_H_unmet_kg")),
        "M24_minus_M6_delivered_kg": (
            True,
            ("service", "M24_minus_M6_delivered_kg"),
        ),
        "M24_minus_M6_unmet_kg": (
            True,
            ("service", "M24_minus_M6_unmet_kg"),
        ),
        "M24_minus_H_default_value_eur": (
            True,
            ("economics", "counterfactual_deltas_eur", "default"),
        ),
        "M24_minus_H_no_methane_value_eur": (
            True,
            ("economics", "counterfactual_deltas_eur", "no_methane_endowment"),
        ),
    }
    summaries: dict[str, Any] = {}
    for name, (contrast, path) in paths.items():
        selected = (
            eligible_analyses
            if name
            in {
                "M24_minus_H_default_value_eur",
                "M24_minus_H_no_methane_value_eur",
            }
            else analyses
        )
        values = [
            (_analysis_cell(item).seed, _analysis_value(item, path))
            for item in selected
        ]
        summaries[name] = summarize_rate_values(values, contrast=contrast)
    component_summaries = {
        name: summarize_rate_values(
            [
                (
                    _analysis_cell(item).seed,
                    _analysis_value(item, ("economics", "component_deltas_eur", name)),
                )
                for item in eligible_analyses
            ],
            contrast=True,
        )
        for name in ECONOMIC_COMPONENT_NAMES
    }
    driver_summaries = {
        name: summarize_rate_values(
            [
                (
                    _analysis_cell(item).seed,
                    _analysis_value(item, ("economics", "raw_driver_deltas", name)),
                )
                for item in eligible_analyses
            ],
            contrast=True,
        )
        for name in RAW_DRIVER_MATERIALITY
    }
    sensitivity: dict[str, Any] = {"counterfactuals": {}, "multipliers": {}}
    for counterfactual in (
        "default",
        "no_methane_endowment",
        "no_declared_wear",
        "service_plus_solar_only",
    ):
        sensitivity["counterfactuals"][counterfactual] = summarize_rate_values(
            [
                (
                    _analysis_cell(item).seed,
                    _analysis_value(
                        item,
                        ("economics", "counterfactual_deltas_eur", counterfactual),
                    ),
                )
                for item in eligible_analyses
            ],
            contrast=True,
        )
    for group in ("methane_endowment", "declared_wear", "solar_opportunity"):
        sensitivity["multipliers"][group] = {
            f"{multiplier:g}": summarize_rate_values(
                [
                    (
                        _analysis_cell(item).seed,
                        _analysis_value(
                            item,
                            (
                                "economics",
                                "one_factor_multiplier_deltas_eur",
                                group,
                                f"{multiplier:g}",
                            ),
                        ),
                    )
                    for item in eligible_analyses
                ],
                contrast=True,
            )
            for multiplier in FULL_DAY_HORIZON_REPAIR_MULTIPLIERS
        }
    intrinsic = [
        _mapping("cell intrinsic observations", analysis["intrinsic"])
        for analysis in analyses
    ]
    m6_unmet = summaries["M6_unmet_kg"]
    m24_unmet = summaries["M24_unmet_kg"]
    m6_mean = m6_unmet["mean"]
    m24_mean = m24_unmet["mean"]
    complete = len(analyses) == len(FULL_DAY_HORIZON_REPAIR_SEEDS)
    reduction = (
        complete
        and m6_mean is not None
        and m24_mean is not None
        and _finite("M6 mean unmet", m6_mean)
        > FULL_DAY_HORIZON_REPAIR_SERVICE_TOLERANCE_KG
        and _at_most_with_tolerance(
            _finite("M24 mean unmet", m24_mean),
            0.10 * _finite("M6 mean unmet", m6_mean),
            tolerance=FULL_DAY_HORIZON_REPAIR_SERVICE_TOLERANCE_KG,
        )
    )
    return {
        "delivery_rate_kg_h": rate,
        "expected_cells": len(FULL_DAY_HORIZON_REPAIR_SEEDS),
        "observed_analyses": len(analyses),
        "missing_seed_ids": [
            seed
            for seed in FULL_DAY_HORIZON_REPAIR_SEEDS
            if seed not in {_analysis_cell(item).seed for item in analyses}
        ],
        "numeric_summaries": summaries,
        "economic_component_contrasts": component_summaries,
        "raw_driver_contrasts": driver_summaries,
        "economic_sensitivities": sensitivity,
        "F7_economic_eligibility": {
            "eligible_count": len(eligible_analyses),
            "ineligible_count": len(analyses) - len(eligible_analyses),
            "ineligible_seed_ids": [
                _analysis_cell(item).seed
                for item in analyses
                if item not in eligible_analyses
            ],
            "eligibility_by_seed": [
                {
                    "seed": _analysis_cell(item).seed,
                    "eligible": item in eligible_analyses,
                    "reason": (
                        None
                        if item in eligible_analyses
                        else "M24 service is not comparable with H within 1e-6 kg"
                    ),
                }
                for item in analyses
            ],
        },
        "registered_measurements": _registered_measurement_summaries(analyses),
        "F5_full_request_count": sum(
            item["F5_full_request"] is True for item in intrinsic
        ),
        "F5_service_comparable_H_count": sum(
            item["F5_service_comparable_H"] is True for item in intrinsic
        ),
        "F6_mean_unmet_reduction_pass": reduction,
        "F6_post_third_adequacy_count": sum(
            item["F6_post_third_adequacy"] is True for item in intrinsic
        ),
        "F6_fourth_material_balance_all": complete
        and all(item["F6_fourth_material_balance"] is True for item in intrinsic),
        "F7_both_expression_rules_count": sum(
            _mapping("cell intrinsic observations", item["intrinsic"])[
                "F7_both_expression_rules"
            ]
            is True
            for item in eligible_analyses
        ),
    }


def aggregate_full_day_horizon_repair_cells(
    cells: Sequence[Mapping[str, Any]],
    *,
    custody: Mapping[str, Any],
    claim_boundary: str,
) -> dict[str, Any]:
    """Build exact chained F1--F8 gates from ten retained cell summaries."""

    expected_ids = [cell.id for cell in REGISTERED_FULL_DAY_HORIZON_REPAIR_CELLS]
    observed_ids: list[str] = []
    analyses: list[Mapping[str, Any]] = []
    for raw in cells:
        summary = _mapping("repair cell summary", raw)
        cell = FullDayHorizonRepairCellV1.from_dict(
            _mapping("repair cell summary specification", summary["cell"])
        )
        observed_ids.append(cell.id)
        analysis = summary.get("analysis")
        if analysis is not None:
            parsed = _mapping("repair cell analysis", analysis)
            if _analysis_cell(parsed) != cell:
                raise ValueError("repair analysis belongs to another cell")
            analyses.append(parsed)
    matrix_exact = observed_ids == expected_ids and len(cells) == 10
    custody_result = evaluate_source_reference_custody(custody)
    by_rate: dict[str, Any] = {}
    selected_by_rate: dict[float, list[Mapping[str, Any]]] = {}
    for rate in FULL_DAY_HORIZON_REPAIR_RATES_KG_H:
        selected = [
            analysis
            for analysis in analyses
            if _analysis_cell(analysis).delivery_rate_kg_h == rate
        ]
        selected_by_rate[rate] = selected
        by_rate[f"r{rate:g}"] = _rate_summaries(rate, selected)

    complete_analyses = len(analyses) == 10
    intrinsic = [
        _mapping("cell intrinsic observations", analysis["intrinsic"])
        for analysis in analyses
    ]
    f1_intrinsic = custody_result["passed"] is True and matrix_exact
    f2_intrinsic = complete_analyses and all(
        item["F2_complete_execution_evidence"] is True for item in intrinsic
    )
    f3_intrinsic = complete_analyses and all(
        item["F3_identity_accounting_safety_authority"] is True for item in intrinsic
    )
    f4_intrinsic = complete_analyses and all(
        item["F4_horizon_intervention_fidelity"] is True for item in intrinsic
    )
    f5_intrinsic = (
        complete_analyses
        and all(item["F5_no_regression_M6"] is True for item in intrinsic)
        and all(item["F5_positive_fourth_delivery"] is True for item in intrinsic)
        and all(
            by_rate[f"r{rate:g}"]["F5_full_request_count"]
            >= FULL_DAY_HORIZON_REPAIR_REQUIRED_CELLS_PER_RATE
            and by_rate[f"r{rate:g}"]["F5_service_comparable_H_count"]
            >= FULL_DAY_HORIZON_REPAIR_REQUIRED_CELLS_PER_RATE
            for rate in FULL_DAY_HORIZON_REPAIR_RATES_KG_H
        )
    )
    f6_intrinsic = complete_analyses and all(
        by_rate[f"r{rate:g}"]["F6_mean_unmet_reduction_pass"] is True
        and by_rate[f"r{rate:g}"]["F6_post_third_adequacy_count"]
        >= FULL_DAY_HORIZON_REPAIR_REQUIRED_CELLS_PER_RATE
        and by_rate[f"r{rate:g}"]["F6_fourth_material_balance_all"] is True
        for rate in FULL_DAY_HORIZON_REPAIR_RATES_KG_H
    )
    f7_intrinsic = complete_analyses and all(
        by_rate[f"r{rate:g}"]["F7_both_expression_rules_count"]
        >= FULL_DAY_HORIZON_REPAIR_REQUIRED_CELLS_PER_RATE
        for rate in FULL_DAY_HORIZON_REPAIR_RATES_KG_H
    )
    claim_valid = claim_boundary == FULL_DAY_HORIZON_REPAIR_CLAIM_BOUNDARY
    f8_intrinsic = (
        complete_analyses
        and all(item["F8_latency"] is True for item in intrinsic)
        and claim_valid
    )
    intrinsic_passes = (
        f1_intrinsic,
        f2_intrinsic,
        f3_intrinsic,
        f4_intrinsic,
        f5_intrinsic,
        f6_intrinsic,
        f7_intrinsic,
    )
    chained: list[bool] = []
    prior = True
    for intrinsic_pass in intrinsic_passes:
        prior = prior and intrinsic_pass
        chained.append(prior)

    observations = (
        {"custody": custody_result, "matrix_exact": matrix_exact},
        {
            "analyzable_cells": len(analyses),
            "all_complete_execution_evidence": f2_intrinsic,
        },
        {"all_identity_accounting_safety_authority": f3_intrinsic},
        {"all_horizon_intervention_fidelity": f4_intrinsic},
        {
            key: {
                "full_request_count": value["F5_full_request_count"],
                "service_comparable_H_count": value["F5_service_comparable_H_count"],
            }
            for key, value in by_rate.items()
        },
        {
            key: {
                "mean_unmet_reduction_pass": value["F6_mean_unmet_reduction_pass"],
                "post_third_adequacy_count": value["F6_post_third_adequacy_count"],
                "fourth_material_balance_all": value["F6_fourth_material_balance_all"],
            }
            for key, value in by_rate.items()
        },
        {
            key: value["F7_both_expression_rules_count"]
            for key, value in by_rate.items()
        },
    )
    gates = [
        {
            "id": f"F{index}",
            "status": "passed" if chained[index - 1] else "failed",
            "intrinsic_passed": intrinsic_passes[index - 1],
            "observation": observations[index - 1],
        }
        for index in range(1, 8)
    ]
    gates.append(
        {
            "id": "F8",
            "status": "passed" if f8_intrinsic else "failed",
            "intrinsic_passed": f8_intrinsic,
            "observation": {
                "all_M24_latency_within_limits": complete_analyses
                and all(item["F8_latency"] is True for item in intrinsic),
                "claim_boundary_exact": claim_valid,
                "claim_boundary": claim_boundary,
            },
        }
    )
    all_pass = all(gate["status"] == "passed" for gate in gates)
    return {
        "status": ("complete_all_gates_passed" if all_pass else "complete_gate_failed"),
        "declared_matrix": matrix_exact,
        "claim_boundary": claim_boundary,
        "custody": custody_result,
        "per_rate": by_rate,
        "gates": gates,
    }


def deterministic_full_day_horizon_repair_aggregate(
    aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove every wall-clock-derived aggregate field from identity material."""

    detached = _mapping(
        "deterministic aggregate", _strict_copy("deterministic aggregate", aggregate)
    )
    result = dict(detached)
    result.pop("status", None)
    gates = _sequence("aggregate gates", result.get("gates"))
    result["gates"] = [
        dict(_mapping("aggregate gate", gate))
        for gate in gates
        if _mapping("aggregate gate", gate).get("id") != "F8"
    ]
    per_rate = _mapping("aggregate per-rate summaries", result.get("per_rate"))
    for rate_summary_value in per_rate.values():
        rate_summary = _mapping("per-rate summary", rate_summary_value)
        registered = _mapping(
            "registered measurement summaries",
            rate_summary.get("registered_measurements"),
        )
        for arm_summary_value in registered.values():
            arm_summary = _mapping("registered arm summary", arm_summary_value)
            if isinstance(arm_summary, dict):
                arm_summary.pop("latency", None)
                retained = arm_summary.get("retained_values_by_seed")
                if isinstance(retained, dict):
                    for records in retained.values():
                        if isinstance(records, list):
                            for record in records:
                                if isinstance(record, dict):
                                    record.pop("source_arm_sha256", None)
    return result


FULL_DAY_HORIZON_REPAIR_CONTRACT: dict[str, Any] = {
    "format": FULL_DAY_HORIZON_REPAIR_FORMAT,
    "cell_format": FULL_DAY_HORIZON_REPAIR_CELL_FORMAT,
    "protocol_inputs": [
        {
            "path": FULL_DAY_HORIZON_REPAIR_PROTOCOL_PATH,
            "sha256": FULL_DAY_HORIZON_REPAIR_PROTOCOL_SHA256,
        },
        {
            "path": FULL_DAY_HORIZON_REPAIR_CLARIFICATION_001_PATH,
            "sha256": FULL_DAY_HORIZON_REPAIR_CLARIFICATION_001_SHA256,
        },
        {
            "path": FULL_DAY_HORIZON_REPAIR_CLARIFICATION_002_PATH,
            "sha256": FULL_DAY_HORIZON_REPAIR_CLARIFICATION_002_SHA256,
        },
    ],
    "dependency_lock": {
        "path": FULL_DAY_HORIZON_REPAIR_LOCK_PATH,
        "sha256": FULL_DAY_HORIZON_REPAIR_LOCK_SHA256,
    },
    "historical_reference": {
        "executed_revision": HISTORICAL_EXECUTED_REVISION,
        "worktree_identity_sha256": HISTORICAL_WORKTREE_IDENTITY_SHA256,
        "contract_sha256": HISTORICAL_CONTRACT_SHA256,
        "raw_report_sha256": HISTORICAL_RAW_REPORT_SHA256,
        "report_domain_sha256": HISTORICAL_REPORT_DOMAIN_SHA256,
        "deterministic_evidence_sha256": (HISTORICAL_DETERMINISTIC_EVIDENCE_SHA256),
        "directory_entry_count": 21,
        "cell_file_names": list(HISTORICAL_REFERENCE_FILE_NAMES),
        "analysis_positions": list(range(11, 21)),
    },
    "matrix": {
        "horizon_h": FULL_DAY_HORIZON_REPAIR_TASK_HORIZON_H,
        "delivery_rates_kg_h": list(FULL_DAY_HORIZON_REPAIR_RATES_KG_H),
        "seeds": list(FULL_DAY_HORIZON_REPAIR_SEEDS),
        "cell_ids": [cell.id for cell in REGISTERED_FULL_DAY_HORIZON_REPAIR_CELLS],
        "cell_count": FULL_DAY_HORIZON_REPAIR_EXPECTED_CELLS,
        "cell_order": "delivery rate then seed; ascending frozen tuple order",
        "expected_controller_decisions": (
            FULL_DAY_HORIZON_REPAIR_EXPECTED_TOTAL_DECISIONS
        ),
        "expected_solver_stages": FULL_DAY_HORIZON_REPAIR_EXPECTED_TOTAL_SOLVER_STAGES,
        "expected_physical_minutes": (
            FULL_DAY_HORIZON_REPAIR_EXPECTED_TOTAL_PHYSICAL_MINUTES
        ),
    },
    "intervention": {
        "controller_id": SERVICE_MPC_SETPOINT_CONTROLLER_ID,
        "task_horizon_h": FULL_DAY_HORIZON_REPAIR_TASK_HORIZON_H,
        "interval_s": FULL_DAY_HORIZON_REPAIR_INTERVAL_S,
        "M6_planning_horizon_h": FULL_DAY_HORIZON_REPAIR_M6_PLANNING_HORIZON_H,
        "M24_planning_horizon_h": FULL_DAY_HORIZON_REPAIR_M24_PLANNING_HORIZON_H,
        "M6_settings_sha256": FULL_DAY_HORIZON_REPAIR_M6_SETTINGS_SHA256,
        "M24_settings_sha256": FULL_DAY_HORIZON_REPAIR_M24_SETTINGS_SHA256,
        "sole_configurable_difference": "planning_horizon_h",
        "derived_serialized_differences": ["maximum_planning_intervals", "sha256"],
    },
    "thresholds": {
        "service_tolerance_kg": FULL_DAY_HORIZON_REPAIR_SERVICE_TOLERANCE_KG,
        "accounting_tolerance": FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE,
        "value_separation_eur_strict": FULL_DAY_HORIZON_REPAIR_VALUE_SEPARATION_EUR,
        "required_cells_per_rate": FULL_DAY_HORIZON_REPAIR_REQUIRED_CELLS_PER_RATE,
        "raw_driver_materiality_strict": RAW_DRIVER_MATERIALITY,
        "controller_p99_limit_ns": FULL_DAY_HORIZON_REPAIR_CONTROLLER_P99_LIMIT_NS,
        "controller_max_limit_ns": FULL_DAY_HORIZON_REPAIR_CONTROLLER_MAX_LIMIT_NS,
        "dispatcher_p99_limit_ns": FULL_DAY_HORIZON_REPAIR_DISPATCHER_P99_LIMIT_NS,
        "dispatcher_max_limit_ns": FULL_DAY_HORIZON_REPAIR_DISPATCHER_MAX_LIMIT_NS,
    },
    "availability": LEGACY_MEASUREMENT_AVAILABILITY,
    "identity_availability": REFERENCE_IDENTITY_AVAILABILITY,
    "gate_chain": ["F1", "F2", "F3", "F4", "F5", "F6", "F7"],
    "claim_boundary": FULL_DAY_HORIZON_REPAIR_CLAIM_BOUNDARY,
}
FULL_DAY_HORIZON_REPAIR_CONTRACT_SHA256 = domain_sha256(
    "full-day-horizon-repair-contract-v1", FULL_DAY_HORIZON_REPAIR_CONTRACT
)


__all__ = [
    "AVAILABLE",
    "FIRST_BOUNDARY_AUDIT_DOMAIN",
    "FIRST_BOUNDARY_AUDIT_SCHEMA",
    "FULL_DAY_HORIZON_REPAIR_ACCOUNTING_TOLERANCE",
    "FULL_DAY_HORIZON_REPAIR_CELL_FORMAT",
    "FULL_DAY_HORIZON_REPAIR_CLAIM_BOUNDARY",
    "FULL_DAY_HORIZON_REPAIR_CLARIFICATION_001_PATH",
    "FULL_DAY_HORIZON_REPAIR_CLARIFICATION_001_SHA256",
    "FULL_DAY_HORIZON_REPAIR_CLARIFICATION_002_PATH",
    "FULL_DAY_HORIZON_REPAIR_CLARIFICATION_002_SHA256",
    "FULL_DAY_HORIZON_REPAIR_CONTRACT",
    "FULL_DAY_HORIZON_REPAIR_CONTRACT_SHA256",
    "FULL_DAY_HORIZON_REPAIR_EXPECTED_CELLS",
    "FULL_DAY_HORIZON_REPAIR_EXPECTED_DECISIONS_PER_CELL",
    "FULL_DAY_HORIZON_REPAIR_EXPECTED_PHYSICAL_MINUTES_PER_CELL",
    "FULL_DAY_HORIZON_REPAIR_EXPECTED_SETTLEMENTS_PER_CELL",
    "FULL_DAY_HORIZON_REPAIR_EXPECTED_SOLVER_STAGES_PER_CELL",
    "FULL_DAY_HORIZON_REPAIR_FORMAT",
    "FULL_DAY_HORIZON_REPAIR_HIGHS_VERSION",
    "FULL_DAY_HORIZON_REPAIR_INTERVAL_S",
    "FULL_DAY_HORIZON_REPAIR_LOCK_PATH",
    "FULL_DAY_HORIZON_REPAIR_LOCK_SHA256",
    "FULL_DAY_HORIZON_REPAIR_M24_SETTINGS_SHA256",
    "FULL_DAY_HORIZON_REPAIR_M6_SETTINGS_SHA256",
    "FULL_DAY_HORIZON_REPAIR_PROTOCOL_PATH",
    "FULL_DAY_HORIZON_REPAIR_PROTOCOL_SHA256",
    "FULL_DAY_HORIZON_REPAIR_RATES_KG_H",
    "FULL_DAY_HORIZON_REPAIR_SCIPY_VERSION",
    "FULL_DAY_HORIZON_REPAIR_SEEDS",
    "FullDayHorizonRepairCellV1",
    "HISTORICAL_ANALYSIS_REFERENCE_FILE_NAMES",
    "HISTORICAL_CONTRACT_SHA256",
    "HISTORICAL_DETERMINISTIC_EVIDENCE_SHA256",
    "HISTORICAL_EXECUTED_REVISION",
    "HISTORICAL_RAW_REPORT_SHA256",
    "HISTORICAL_REFERENCE_FILE_NAMES",
    "HISTORICAL_REPORT_DOMAIN_SHA256",
    "HISTORICAL_WORKTREE_IDENTITY_SHA256",
    "INTERVAL_LEDGER_DOMAIN",
    "INTERVAL_LEDGER_ROW_DOMAIN",
    "INTERVAL_LEDGER_ROW_SCHEMA",
    "INTERVAL_LEDGER_SCHEMA",
    "LEGACY_MEASUREMENT_AVAILABILITY",
    "NOT_APPLICABLE",
    "NOT_RETAINED",
    "RAW_DRIVER_MATERIALITY",
    "REFERENCE_IDENTITY_AVAILABILITY",
    "REGISTERED_FULL_DAY_HORIZON_REPAIR_CELLS",
    "aggregate_full_day_horizon_repair_cells",
    "build_first_boundary_audit",
    "build_interval_ledger",
    "build_interval_ledger_row",
    "contrast_sign",
    "deterministic_full_day_horizon_repair_aggregate",
    "evaluate_full_day_horizon_repair_cell",
    "evaluate_source_reference_custody",
    "historical_reference_file_names",
    "m24_settings_for_cell",
    "m6_settings_for_cell",
    "measurement",
    "registered_cells",
    "summarize_rate_values",
    "verify_cell_citations",
    "verify_compact_arm_measurements",
    "verify_first_boundary_audit",
    "verify_horizon_only_settings",
    "verify_interval_ledger",
    "verify_measurement",
]

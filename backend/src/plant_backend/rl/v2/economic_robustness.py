"""Outcome-blind analysis contract for the economic robustness tournament."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from numbers import Integral, Real
import statistics
from typing import Any, ClassVar, cast

from .economics import EconomicAssumptionsV1
from .mpc_schema import (
    SERVICE_MPC_SETPOINT_CONTROLLER_ID,
    canonical_json,
    domain_sha256,
)
from .mpc_setpoint_run import verify_mpc_setpoint_run_digests
from .setpoint_command import (
    PROCESS_SETPOINT_COMMAND_SCHEMA,
    PROCESS_SETPOINT_ENVELOPE_SCHEMA,
    PROCESS_SETPOINT_MINUTE_DISPATCH_SCHEMA,
)
from .setpoint_control import (
    PROCESS_SETPOINT_ALLOCATION_SCHEMA,
    PROCESS_SETPOINT_TARGET_SCHEMA,
)
from .setpoint_controllers import (
    LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID,
    LEGACY_HEURISTIC_SETPOINT_EVIDENCE_SCHEMA,
)
from .setpoint_execution import (
    PROCESS_SETPOINT_INTERVAL_EXECUTION_SCHEMA,
    PROCESS_SETPOINT_MINUTE_APPLICATION_SCHEMA,
)


ECONOMIC_ROBUSTNESS_FORMAT = "economic-task-robustness-tournament-v1"
ECONOMIC_ROBUSTNESS_CELL_FORMAT = "economic-task-robustness-cell-v1"
ECONOMIC_ROBUSTNESS_PROTOCOL_PATH = (
    "docs/research/ECONOMIC_ROBUSTNESS_TOURNAMENT_PROTOCOL.md"
)
ECONOMIC_ROBUSTNESS_PROTOCOL_SHA256 = (
    "6fb5eb11933ba5ee601849210b4e61278e4cec2ddb97c8e31b05817ee7141412"
)
ECONOMIC_ROBUSTNESS_LOCK_SHA256 = (
    "080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd"
)
ECONOMIC_ROBUSTNESS_HORIZONS_H = (6.0, 24.0)
ECONOMIC_ROBUSTNESS_RATES_KG_H = (35.0, 42.0)
ECONOMIC_ROBUSTNESS_SEEDS = (20, 21, 22, 23, 24)
ECONOMIC_ROBUSTNESS_PLANNING_HORIZON_H = 6.0
ECONOMIC_ROBUSTNESS_MPC_SETTINGS_SHA256 = {
    6.0: "7f89b35d23ee9313f9a8723050a2dcbc94f9303e18e96cab81e86fa9f2a0261e",
    24.0: "28dd0dbbd732f4defcc700e3fcb91e59b2d14d6df7bac728963090e37ba48b0f",
}
ECONOMIC_ROBUSTNESS_MULTIPLIERS = (0.0, 0.5, 1.0, 2.0)
ECONOMIC_ROBUSTNESS_SERVICE_TOLERANCE_KG = 1e-6
ECONOMIC_ROBUSTNESS_ACCOUNTING_TOLERANCE_EUR = 1e-8
ECONOMIC_ROBUSTNESS_VALUE_SEPARATION_EUR = 1.0
ECONOMIC_ROBUSTNESS_REQUIRED_SEEDS_PER_STRATUM = 4

ECONOMIC_COMPONENT_NAMES = (
    "delivered_service_value",
    "unmet_delivery_penalty",
    "solar_opportunity_cost",
    "battery_throughput_reserve",
    "electrolyser_life_reserve",
    "methane_endowment_liability_change",
    "battery_endowment_liability_change",
)

# Every value is reported as M minus H. Raw minimum is the frozen "worst seed"
# convention; benefit direction is deliberately not inferred for diagnostic metrics.
PAIRED_METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "net_operating_value_eur": ("economic_metrics", "net_operating_value_eur"),
    **{
        f"component.{name}": ("economic_metrics", "components_eur", name)
        for name in ECONOMIC_COMPONENT_NAMES
    },
    "delivered_kg": ("economic_metrics", "delivered_kg"),
    "unmet_kg": ("economic_metrics", "unmet_kg"),
    "accepted_solar_kwh": ("economic_metrics", "accepted_solar_kwh"),
    "accepted_process_energy_kwh": (
        "physical_metrics",
        "accepted_process_energy_kwh",
    ),
    "curtailed_kwh": ("physical_metrics", "curtailed_kwh"),
    "battery_throughput_kwh": ("economic_metrics", "battery_throughput_kwh"),
    "battery_equivalent_full_cycles": (
        "economic_metrics",
        "battery_equivalent_full_cycles",
    ),
    "terminal_battery_energy_kwh": (
        "economic_metrics",
        "terminal_battery_energy_kwh",
    ),
    "electrolyser_equivalent_operating_h": (
        "economic_metrics",
        "electrolyser_equivalent_operating_h",
    ),
    "electrolyser_cold_starts": ("physical_metrics", "electrolyser_cold_starts"),
    "electrolyser_hot_starts": ("physical_metrics", "electrolyser_hot_starts"),
    "electrolyser_aborted_starts": (
        "physical_metrics",
        "electrolyser_aborted_starts",
    ),
    "electrolyser_forced_stops": (
        "physical_metrics",
        "electrolyser_forced_stops",
    ),
    "electrolyser_operating_time_s": (
        "physical_metrics",
        "electrolyser_operating_time_s",
    ),
    "terminal_ch4_kg": ("physical_metrics", "terminal_ch4_kg"),
    "terminal_h2_kg": ("physical_metrics", "terminal_h2_kg"),
    "terminal_co2_kg": ("physical_metrics", "terminal_co2_kg"),
    "terminal_ch4_equivalent_deficit_kg": (
        "economic_metrics",
        "terminal_ch4_equivalent_deficit_kg",
    ),
    "terminal_battery_deficit_kwh": (
        "economic_metrics",
        "terminal_battery_deficit_kwh",
    ),
    "guard_interventions": ("physical_metrics", "guard_interventions"),
    "corrected_physics_steps": ("physical_metrics", "corrected_physics_steps"),
    "shutdown_steps": ("physical_metrics", "shutdown_steps"),
    "out_of_range_decisions": ("physical_metrics", "out_of_range_decisions"),
}

ECONOMIC_ROBUSTNESS_CONTRACT: dict[str, Any] = {
    "format": ECONOMIC_ROBUSTNESS_FORMAT,
    "cell_format": ECONOMIC_ROBUSTNESS_CELL_FORMAT,
    "protocol": {
        "path": ECONOMIC_ROBUSTNESS_PROTOCOL_PATH,
        "raw_file_sha256": ECONOMIC_ROBUSTNESS_PROTOCOL_SHA256,
    },
    "dependency_lock": {
        "path": "uv.lock",
        "sha256": ECONOMIC_ROBUSTNESS_LOCK_SHA256,
    },
    "matrix": {
        "horizons_h": list(ECONOMIC_ROBUSTNESS_HORIZONS_H),
        "delivery_rates_kg_h": list(ECONOMIC_ROBUSTNESS_RATES_KG_H),
        "seeds": list(ECONOMIC_ROBUSTNESS_SEEDS),
        "controller_order": ["H", "M"],
        "cell_order": "horizon, delivery rate, seed; ascending frozen tuple order",
        "cell_count": 20,
        "arm_count": 40,
        "expected_controller_decisions": 2_400,
        "expected_minute_applications": 36_000,
    },
    "controllers": {
        "H": {
            "id": LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID,
            "uses_tariff": False,
        },
        "M": {
            "id": SERVICE_MPC_SETPOINT_CONTROLLER_ID,
            "planning_horizon_h": ECONOMIC_ROBUSTNESS_PLANNING_HORIZON_H,
            "settings_sha256_by_task_horizon": {
                f"{horizon:g}": digest
                for horizon, digest in ECONOMIC_ROBUSTNESS_MPC_SETTINGS_SHA256.items()
            },
            "uses_tariff": False,
            "economic_mpc": False,
        },
    },
    "sensitivities": {
        "multipliers": list(ECONOMIC_ROBUSTNESS_MULTIPLIERS),
        "one_factor_at_a_time": [
            "methane_endowment_liability_change",
            "battery_throughput_reserve + electrolyser_life_reserve",
            "solar_opportunity_cost",
        ],
        "leave_out": [
            "methane endowment liability",
            "battery and electrolyser wear reserve",
            "all except service and solar opportunity cost",
        ],
    },
    "thresholds": {
        "service_tolerance_kg": ECONOMIC_ROBUSTNESS_SERVICE_TOLERANCE_KG,
        "accounting_tolerance_eur": ECONOMIC_ROBUSTNESS_ACCOUNTING_TOLERANCE_EUR,
        "value_separation_eur": ECONOMIC_ROBUSTNESS_VALUE_SEPARATION_EUR,
        "required_cells_per_five_seed_stratum": (
            ECONOMIC_ROBUSTNESS_REQUIRED_SEEDS_PER_STRATUM
        ),
        "raw_driver_materiality": {
            "accepted_solar_kwh": 1.0,
            "accepted_process_energy_kwh": 1.0,
            "battery_throughput_kwh": 1.0,
            "electrolyser_equivalent_operating_h": 0.01,
            "terminal_ch4_equivalent_deficit_kg": 1e-3,
            "terminal_battery_deficit_kwh": 0.1,
        },
    },
    "execution": {
        "sequential": True,
        "within_cell_order": ["H", "M"],
        "failure_policy": "retain and continue; no surviving-subset aggregate",
        "retry_policy": "no same-revision controller-cell retry",
        "publication": "exclusive external per-cell JSON plus compact report JSON",
    },
    "claim_boundary": (
        "development task robustness only; no controller, RL, commercial, "
        "commission, held-out, or generalization claim"
    ),
}
ECONOMIC_ROBUSTNESS_CONTRACT_SHA256 = domain_sha256(
    "economic-task-robustness-contract-v1",
    ECONOMIC_ROBUSTNESS_CONTRACT,
)


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return value


def _boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def _close(left: float, right: float, *, tolerance: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _sign(value: float, *, tolerance: float = 1e-8) -> str:
    if value > tolerance:
        return "M"
    if value < -tolerance:
        return "H"
    return "tie"


@dataclass(frozen=True, slots=True, order=True)
class EconomicRobustnessCellV1:
    """One frozen task cell; order is horizon, rate, then seed."""

    horizon_h: float
    delivery_rate_kg_h: float
    seed: int

    schema: ClassVar[str] = "economic-task-robustness-cell-spec-v1"

    def __post_init__(self) -> None:
        horizon = _finite("cell horizon_h", self.horizon_h)
        rate = _finite("cell delivery_rate_kg_h", self.delivery_rate_kg_h)
        if horizon <= 0.0 or rate <= 0.0:
            raise ValueError("cell horizon and delivery rate must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("cell seed must be an integer")
        if self.seed < 0:
            raise ValueError("cell seed must be non-negative")
        object.__setattr__(self, "horizon_h", horizon)
        object.__setattr__(self, "delivery_rate_kg_h", rate)
        object.__setattr__(self, "seed", int(self.seed))
        decisions = horizon * 3_600.0 / 900.0
        physics_steps = horizon * 3_600.0 / 60.0
        if not decisions.is_integer() or not physics_steps.is_integer():
            raise ValueError(
                "cell horizon must align with both controller and physics cadence"
            )
        settlements = horizon / 6.0
        if not settlements.is_integer():
            raise ValueError("cell horizon must contain complete six-hour settlements")

    @property
    def id(self) -> str:
        return f"h{self.horizon_h:g}-r{self.delivery_rate_kg_h:g}-s{int(self.seed)}"

    @property
    def expected_decisions_per_arm(self) -> int:
        return int(round(self.horizon_h * 3_600.0 / 900.0))

    @property
    def expected_physical_steps_per_arm(self) -> int:
        return int(round(self.horizon_h * 3_600.0 / 60.0))

    @property
    def expected_settlements_per_arm(self) -> int:
        return int(round(self.horizon_h / 6.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "id": self.id,
            "horizon_h": float(self.horizon_h),
            "delivery_rate_kg_h": float(self.delivery_rate_kg_h),
            "seed": int(self.seed),
            "expected_decisions_per_arm": self.expected_decisions_per_arm,
            "expected_physical_steps_per_arm": self.expected_physical_steps_per_arm,
            "expected_settlements_per_arm": self.expected_settlements_per_arm,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EconomicRobustnessCellV1":
        result = cls(
            horizon_h=_finite("cell horizon_h", raw.get("horizon_h")),
            delivery_rate_kg_h=_finite(
                "cell delivery_rate_kg_h", raw.get("delivery_rate_kg_h")
            ),
            seed=_integer("cell seed", raw.get("seed")),
        )
        if dict(raw) != result.to_dict():
            raise ValueError(
                "serialized robustness cell differs from its canonical form"
            )
        return result


def registered_cells() -> tuple[EconomicRobustnessCellV1, ...]:
    return tuple(
        EconomicRobustnessCellV1(horizon, rate, seed)
        for horizon in ECONOMIC_ROBUSTNESS_HORIZONS_H
        for rate in ECONOMIC_ROBUSTNESS_RATES_KG_H
        for seed in ECONOMIC_ROBUSTNESS_SEEDS
    )


REGISTERED_ECONOMIC_ROBUSTNESS_CELLS = registered_cells()


def episode_for_cell(cell: EconomicRobustnessCellV1) -> dict[str, Any]:
    """Return the canonical task configuration represented by one matrix cell."""

    # Imported lazily so the pure aggregation helpers remain inexpensive to import.
    from .config import EpisodeConfigV2

    return cast(
        dict[str, Any],
        EpisodeConfigV2(
            horizon_h=cell.horizon_h,
            delivery_rate_kg_h=cell.delivery_rate_kg_h,
        ).to_dict(),
    )


def _expected_mpc_settings_sha256(cell: EconomicRobustnessCellV1) -> str:
    try:
        return ECONOMIC_ROBUSTNESS_MPC_SETTINGS_SHA256[cell.horizon_h]
    except KeyError as reason:
        raise ValueError("cell horizon has no frozen MPC settings identity") from reason


def mpc_settings_for_cell(cell: EconomicRobustnessCellV1) -> dict[str, Any]:
    """Return and identity-check the frozen service-MPC settings for one cell."""

    from .mpc_schema import ServiceMPCSetpointSettingsV1

    settings = ServiceMPCSetpointSettingsV1(
        task_horizon_h=cell.horizon_h,
        planning_horizon_h=ECONOMIC_ROBUSTNESS_PLANNING_HORIZON_H,
    )
    if settings.sha256 != _expected_mpc_settings_sha256(cell):
        raise RuntimeError("frozen robustness MPC settings identity drifted")
    return cast(dict[str, Any], settings.to_dict())


def _verify_schema_digest(
    name: str,
    value: object,
    *,
    expected_schema: str,
) -> Mapping[str, Any]:
    item = dict(_mapping(name, value))
    recorded = item.pop("sha256", None)
    if item.get("schema") != expected_schema:
        raise ValueError(f"{name} schema differs")
    if (
        not isinstance(recorded, str)
        or domain_sha256(expected_schema, item) != recorded
    ):
        raise ValueError(f"{name} digest differs")
    return _mapping(name, value)


def _verify_targets(name: str, value: object) -> Mapping[str, Any]:
    target = _verify_schema_digest(
        name,
        value,
        expected_schema=PROCESS_SETPOINT_TARGET_SCHEMA,
    )
    setpoints = _mapping(f"{name} setpoints", target["setpoints_kw"])
    if set(setpoints) != {
        "electrolyser",
        "dac",
        "methanator",
        "battery_charge_ceiling",
    }:
        raise ValueError(f"{name} setpoint fields differ")
    for key, value in setpoints.items():
        if _finite(f"{name} {key}", value) < 0.0:
            raise ValueError(f"{name} {key} must be non-negative")
    return target


def _verify_heuristic_decision(
    decision: object,
    *,
    expected_index: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    item = _mapping("H decision", decision)
    evidence = _verify_schema_digest(
        "H decision evidence",
        item["evidence"],
        expected_schema=LEGACY_HEURISTIC_SETPOINT_EVIDENCE_SCHEMA,
    )
    command = _verify_schema_digest(
        "H command",
        item["command"],
        expected_schema=PROCESS_SETPOINT_COMMAND_SCHEMA,
    )
    requested = _verify_targets("H requested targets", evidence["requested_targets"])
    issued = _verify_targets("H issued targets", evidence["issued_targets"])
    command_targets = _verify_targets("H command targets", command["targets"])
    envelope = _verify_schema_digest(
        "H command envelope",
        command["envelope"],
        expected_schema=PROCESS_SETPOINT_ENVELOPE_SCHEMA,
    )
    if (
        _integer("H evidence decision index", evidence["decision_index"])
        != expected_index
    ):
        raise ValueError("H evidence decision order differs")
    if (
        _integer("H command decision index", command["decision_index"])
        != expected_index
    ):
        raise ValueError("H command decision order differs")
    if evidence["observation_sha256"] != command["observation_sha256"]:
        raise ValueError("H command observation differs from decision evidence")
    issuer = _mapping("H command issuer", command["issuer"])
    if (
        issuer.get("id") != LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID
        or issuer.get("source_schema") != LEGACY_HEURISTIC_SETPOINT_EVIDENCE_SCHEMA
        or issuer.get("source_sha256") != evidence["sha256"]
    ):
        raise ValueError("H command issuer binding differs")
    if command_targets != issued:
        raise ValueError("H command targets differ from issued targets")
    source = _mapping("H source decision", evidence["source_decision"])
    action = _mapping("H source action", source["action"])
    if (
        not isinstance(source.get("invalid"), bool)
        or item.get("invalid") is not source["invalid"]
    ):
        raise ValueError("H invalid flag differs from source evidence")
    solar = _finite("H boundary solar", evidence["boundary_solar_kw"])
    requested_kw = _mapping("H requested setpoints", requested["setpoints_kw"])
    expected_requested = {
        "electrolyser": _finite(
            "H source electrolyser share", action["electrolyser_share"]
        )
        * solar
        + _finite("H source battery support", action["battery_support_kw"]),
        "dac": _finite("H source DAC share", action["dac_share"]) * solar,
        "methanator": _finite("H source methanator share", action["methanator_share"])
        * solar,
        "battery_charge_ceiling": _finite(
            "H source battery share", action["battery_share"]
        )
        * solar,
    }
    if any(
        not _close(
            _finite(f"H requested {name}", requested_kw[name]),
            expected,
            tolerance=1e-7,
        )
        for name, expected in expected_requested.items()
    ):
        raise ValueError("H requested targets do not reproduce the source action")
    issued_kw = _mapping("H issued setpoints", issued["setpoints_kw"])
    if any(
        not _close(
            _finite(f"H issued {name}", issued_kw[name]),
            _finite(f"H requested {name}", requested_kw[name]),
            tolerance=1e-7,
        )
        for name in ("electrolyser", "dac", "methanator")
    ):
        raise ValueError("H adapter changed a process target")
    requested_charge = _finite(
        "H requested battery charge", requested_kw["battery_charge_ceiling"]
    )
    issued_charge = _finite(
        "H issued battery charge", issued_kw["battery_charge_ceiling"]
    )
    if issued_charge > requested_charge + 1e-7:
        raise ValueError("H adapter increased battery-charge authority")
    limited = issued_charge < requested_charge - 1e-7
    if evidence.get("battery_charge_limited") is not limited:
        raise ValueError("H battery-charge limitation flag differs")
    limits = _mapping("H command limits", envelope["limits_kw"])
    for target_name, limit_name in (
        ("electrolyser", "electrolyser"),
        ("dac", "dac"),
        ("methanator", "methanator"),
        ("battery_charge_ceiling", "battery_charge"),
    ):
        if _finite(f"H command {target_name}", issued_kw[target_name]) > (
            _finite(f"H envelope {limit_name}", limits[limit_name]) + 1e-7
        ):
            raise ValueError("H command target exceeds its envelope")
    return evidence, command


def _verify_heuristic_execution(
    execution: object,
    *,
    command: Mapping[str, Any],
    previous_final: tuple[int, str] | None,
) -> tuple[int, str, int]:
    item = _verify_schema_digest(
        "H interval execution",
        execution,
        expected_schema=PROCESS_SETPOINT_INTERVAL_EXECUTION_SCHEMA,
    )
    nested_command = _verify_schema_digest(
        "H execution command",
        item["command"],
        expected_schema=PROCESS_SETPOINT_COMMAND_SCHEMA,
    )
    if nested_command != command:
        raise ValueError("H interval execution belongs to another command")
    applications = item.get("applications")
    if not isinstance(applications, list) or not applications:
        raise ValueError("H interval execution has no minute applications")
    if _integer("H interval application count", item["application_count"]) != len(
        applications
    ):
        raise ValueError("H interval application count differs")
    hold_duration = _finite("H command hold duration", command["hold_duration_s"])
    physics_step = _finite("H execution physics step", item["physics_step_s"])
    if physics_step <= 0.0 or not _close(
        hold_duration,
        physics_step * len(applications),
        tolerance=1e-9,
    ):
        raise ValueError("H minute evidence does not cover the command hold")

    prior = previous_final
    for ordinal, application_value in enumerate(applications):
        application = _verify_schema_digest(
            "H minute application",
            application_value,
            expected_schema=PROCESS_SETPOINT_MINUTE_APPLICATION_SCHEMA,
        )
        if _integer("H application ordinal", application["ordinal"]) != ordinal:
            raise ValueError("H application ordinal differs")
        dispatch = _verify_schema_digest(
            "H minute dispatch",
            application["dispatch"],
            expected_schema=PROCESS_SETPOINT_MINUTE_DISPATCH_SCHEMA,
        )
        allocation = _verify_schema_digest(
            "H setpoint allocation",
            dispatch["allocation"],
            expected_schema=PROCESS_SETPOINT_ALLOCATION_SCHEMA,
        )
        allocation_targets = _verify_targets(
            "H allocation targets", allocation["targets"]
        )
        command_targets = _mapping("H command targets", command["targets"])
        if allocation_targets != command_targets:
            raise ValueError("H allocation belongs to another command")
        if dispatch.get("command_sha256") != command["sha256"]:
            raise ValueError("H dispatch command digest differs")
        if dispatch.get("decision_index") != command["decision_index"]:
            raise ValueError("H dispatch decision index differs")
        boundary = _mapping("H dispatch boundary", dispatch["boundary"])
        next_boundary = _mapping("H next boundary", application["next_boundary"])
        boundary_step = _integer("H boundary step", boundary["step"])
        next_step = _integer("H next boundary step", next_boundary["step"])
        if next_step != boundary_step + 1:
            raise ValueError("H application boundary steps are not contiguous")
        if prior is not None and prior != (boundary_step, str(boundary["sha256"])):
            raise ValueError("H physical boundary chain is broken")
        guard = _mapping("H application guard", application["guard"])
        plant = _mapping("H application plant", application["plant"])
        if guard.get("action") != plant.get("action"):
            raise ValueError("H applied action differs from the guarded action")
        adjustments = guard.get("adjustments")
        if not isinstance(adjustments, list):
            raise TypeError("H guard adjustments must be a list")
        changed = guard.get("action") != allocation.get("action") or bool(adjustments)
        if guard.get("changed_request") is not changed:
            raise ValueError("H guard changed-request flag differs")
        requested_action = _mapping("H requested minute action", allocation["action"])
        guarded_action = _mapping("H guarded minute action", guard["action"])
        for name in (
            "electrolyser_share",
            "dac_share",
            "methanator_share",
            "battery_share",
        ):
            requested_share = _finite(f"H requested {name}", requested_action[name])
            guarded_share = _finite(f"H guarded {name}", guarded_action[name])
            if not 0.0 <= requested_share <= 1.0 or not 0.0 <= guarded_share <= 1.0:
                raise ValueError("H minute action share is outside [0, 1]")
        for name in ("electrolyser_share", "dac_share", "methanator_share"):
            if _finite(f"H guarded {name}", guarded_action[name]) > (
                _finite(f"H requested {name}", requested_action[name]) + 1e-7
            ):
                raise ValueError("H guard increased common process authority")
        requested_support = _finite(
            "H requested support", requested_action["battery_support_kw"]
        )
        guarded_support = _finite(
            "H guarded support", guarded_action["battery_support_kw"]
        )
        if (
            requested_support < 0.0
            or not 0.0 <= guarded_support <= requested_support + 1e-7
        ):
            raise ValueError("H guard increased common support authority")

        solar = _finite("H dispatched solar", allocation["solar_available_kw"])
        if solar < 0.0:
            raise ValueError("H dispatched solar must be non-negative")
        applied = _mapping("H applied power", plant["applied_kw"])
        applied_values = {
            name: _finite(f"H applied {name}", applied[name])
            for name in (
                "solar_available",
                "solar_to_battery",
                "solar_to_electrolyser",
                "solar_to_dac",
                "solar_to_methanator",
                "battery_support",
                "electrolyser",
                "dac",
                "methanator",
                "solar_curtailed",
            )
        }
        if any(value < 0.0 for value in applied_values.values()):
            raise ValueError("H applied power must be non-negative")
        if not _close(applied_values["solar_available"], solar, tolerance=1e-7):
            raise ValueError("H applied solar differs from dispatch")
        solar_total = math.fsum(
            applied_values[name]
            for name in (
                "solar_to_battery",
                "solar_to_electrolyser",
                "solar_to_dac",
                "solar_to_methanator",
                "solar_curtailed",
            )
        )
        if not _close(solar_total, solar, tolerance=1e-7):
            raise ValueError("H applied solar bus does not balance")
        if not _close(
            applied_values["electrolyser"],
            applied_values["solar_to_electrolyser"] + applied_values["battery_support"],
            tolerance=1e-7,
        ):
            raise ValueError("H applied electrolyser bus does not balance")
        target_kw = _mapping("H command target values", command["targets"])[
            "setpoints_kw"
        ]
        targets = _mapping("H command setpoint values", target_kw)
        guarded_electrolyser = (
            _finite(
                "H guarded electrolyser share", guarded_action["electrolyser_share"]
            )
            * solar
            + guarded_support
        )
        if (
            applied_values["battery_support"] > guarded_support + 1e-7
            or applied_values["electrolyser"] > guarded_electrolyser + 1e-7
            or applied_values["electrolyser"]
            > _finite("H electrolyser target", targets["electrolyser"]) + 1e-7
            or applied_values["dac"]
            > _finite("H guarded DAC share", guarded_action["dac_share"]) * solar + 1e-7
            or applied_values["dac"] > _finite("H DAC target", targets["dac"]) + 1e-7
            or applied_values["methanator"]
            > _finite("H guarded methanator share", guarded_action["methanator_share"])
            * solar
            + 1e-7
            or applied_values["methanator"]
            > _finite("H methanator target", targets["methanator"]) + 1e-7
        ):
            raise ValueError("H plant draw exceeds guarded target authority")
        prior = (next_step, str(next_boundary["sha256"]))

    first_boundary = _mapping(
        "H first application dispatch",
        _mapping("H first application", applications[0])["dispatch"],
    )["boundary"]
    first = _mapping("H first boundary", first_boundary)
    if item.get("first_boundary_sha256") != first.get("sha256"):
        raise ValueError("H interval first boundary digest differs")
    if prior is None or item.get("final_boundary_sha256") != prior[1]:
        raise ValueError("H interval final boundary digest differs")
    return prior[0], prior[1], len(applications)


def verify_heuristic_arm_evidence(run: Mapping[str, Any]) -> None:
    """Verify the generic-command evidence graph retained for arm H."""

    evidence = dict(_mapping("H evidence", run["evidence"]))
    recorded = evidence.pop("sha256", None)
    if not isinstance(recorded, str):
        raise ValueError("H evidence omits its digest")
    if domain_sha256("economic-smoke-heuristic-evidence-v1", evidence) != recorded:
        raise ValueError("H evidence digest differs")

    decisions = evidence.get("decisions")
    executions = evidence.get("executions")
    if not isinstance(decisions, list) or not isinstance(executions, list):
        raise TypeError("H decisions and executions must be lists")
    if _integer("H decision count", evidence.get("decision_count")) != len(decisions):
        raise ValueError("H decision count differs from retained decisions")
    if _integer("H execution count", evidence.get("execution_count")) != len(
        executions
    ):
        raise ValueError("H execution count differs from retained executions")
    decision_sha = evidence.get("decision_sha256")
    command_sha = evidence.get("command_sha256")
    execution_sha = evidence.get("execution_sha256")
    if not all(
        isinstance(values, list)
        for values in (decision_sha, command_sha, execution_sha)
    ):
        raise TypeError("H digest ledgers must be lists")
    expected_decisions = [
        str(_mapping("H decision evidence", item["evidence"])["sha256"])
        for item in decisions
        if isinstance(item, Mapping)
    ]
    expected_commands = [
        str(_mapping("H decision command", item["command"])["sha256"])
        for item in decisions
        if isinstance(item, Mapping)
    ]
    expected_executions = [
        str(_mapping("H execution", item)["sha256"])
        for item in executions
        if isinstance(item, Mapping)
    ]
    if len(expected_decisions) != len(decisions) or decision_sha != expected_decisions:
        raise ValueError("H decision digest ledger differs")
    if command_sha != expected_commands:
        raise ValueError("H command digest ledger differs")
    if (
        len(expected_executions) != len(executions)
        or execution_sha != expected_executions
    ):
        raise ValueError("H execution digest ledger differs")
    application_count = sum(
        _integer(
            "H interval application count",
            _mapping("H execution", item)["application_count"],
        )
        for item in executions
    )
    if _integer("H application count", evidence.get("application_count")) != (
        application_count
    ):
        raise ValueError("H application count differs from interval evidence")

    previous_final: tuple[int, str] | None = None
    deeply_verified_applications = 0
    for index, (decision, execution) in enumerate(
        zip(decisions, executions, strict=True)
    ):
        _, command = _verify_heuristic_decision(decision, expected_index=index)
        final_step, final_sha, count = _verify_heuristic_execution(
            execution,
            command=command,
            previous_final=previous_final,
        )
        previous_final = (final_step, final_sha)
        deeply_verified_applications += count
    if deeply_verified_applications != application_count:
        raise ValueError("H deeply verified minute count differs")


def _latency_series(values: Sequence[object]) -> dict[str, int | float | str | None]:
    parsed = [_integer("latency", value) for value in values]
    if not parsed:
        return {
            "count": 0,
            "minimum_ns": None,
            "maximum_ns": None,
            "mean_ns": None,
            "p99_ns": None,
            "p99_method": "empirical nearest rank",
        }
    ordered = sorted(parsed)
    rank = max(1, math.ceil(0.99 * len(ordered)))
    return {
        "count": len(ordered),
        "minimum_ns": ordered[0],
        "maximum_ns": ordered[-1],
        "mean_ns": statistics.fmean(ordered),
        "p99_ns": ordered[rank - 1],
        "p99_method": "empirical nearest rank",
    }


def arm_latency_evidence(
    heuristic: Mapping[str, Any],
    mpc: Mapping[str, Any],
) -> dict[str, Any]:
    """Report wall-clock evidence separately from deterministic performance data."""

    h_evidence = _mapping("H evidence", heuristic["evidence"])
    h_dispatcher_nested = h_evidence.get("dispatcher_elapsed_ns", [])
    if not isinstance(h_dispatcher_nested, list) or any(
        not isinstance(interval, list) for interval in h_dispatcher_nested
    ):
        raise TypeError("H dispatcher timing must be a list")
    h_dispatcher = [value for interval in h_dispatcher_nested for value in interval]
    h_controller = h_evidence.get("controller_elapsed_ns", [])
    if not isinstance(h_controller, list):
        raise TypeError("H controller timing must be a list")

    m_evidence = _mapping("M evidence", mpc["evidence"])
    return {
        "excluded_from_deterministic_evidence": True,
        "H": {
            "arm_elapsed_ns": _integer("H arm elapsed_ns", heuristic["elapsed_ns"]),
            "controller": _latency_series(h_controller),
            "dispatcher": _latency_series(h_dispatcher),
        },
        "M": {
            "arm_elapsed_ns": _integer("M arm elapsed_ns", mpc["elapsed_ns"]),
            "decision": dict(
                _mapping("M decision latency", m_evidence["decision_latency"])
            ),
            "solver_stage": dict(
                _mapping("M solver-stage latency", m_evidence["solver_stage_latency"])
            ),
            "dispatcher": dict(
                _mapping("M dispatcher latency", m_evidence["dispatcher_latency"])
            ),
        },
    }


def _nested_number(run: Mapping[str, Any], path: Sequence[str]) -> float:
    current: object = run
    for part in path:
        current = _mapping(f"metric path {'.'.join(path)}", current)[part]
    return _finite(f"metric {'.'.join(path)}", current)


def paired_metric_deltas(
    heuristic: Mapping[str, Any],
    mpc: Mapping[str, Any],
) -> dict[str, float]:
    return {
        name: _nested_number(mpc, path) - _nested_number(heuristic, path)
        for name, path in PAIRED_METRIC_PATHS.items()
    }


def _component_deltas(
    heuristic: Mapping[str, Any],
    mpc: Mapping[str, Any],
) -> dict[str, float]:
    h_components = _mapping(
        "H economic components",
        _mapping("H economic metrics", heuristic["economic_metrics"])["components_eur"],
    )
    m_components = _mapping(
        "M economic components",
        _mapping("M economic metrics", mpc["economic_metrics"])["components_eur"],
    )
    if set(h_components) != set(ECONOMIC_COMPONENT_NAMES) or set(m_components) != set(
        ECONOMIC_COMPONENT_NAMES
    ):
        raise ValueError("economic component set differs from the frozen ledger")
    return {
        name: _finite(f"M component {name}", m_components[name])
        - _finite(f"H component {name}", h_components[name])
        for name in ECONOMIC_COMPONENT_NAMES
    }


def economic_sensitivities(
    heuristic: Mapping[str, Any],
    mpc: Mapping[str, Any],
) -> dict[str, Any]:
    """Algebraically rescore one retained pair without rerunning either controller."""

    components = _component_deltas(heuristic, mpc)
    default = math.fsum(components.values())
    methane = components["methane_endowment_liability_change"]
    wear = (
        components["battery_throughput_reserve"]
        + components["electrolyser_life_reserve"]
    )
    solar = components["solar_opportunity_cost"]

    def scaled(group: float, multiplier: float) -> float:
        return default + (multiplier - 1.0) * group

    multiplier_results = {
        "methane_endowment": {
            f"{value:g}": scaled(methane, value)
            for value in ECONOMIC_ROBUSTNESS_MULTIPLIERS
        },
        "declared_wear": {
            f"{value:g}": scaled(wear, value)
            for value in ECONOMIC_ROBUSTNESS_MULTIPLIERS
        },
        "solar_opportunity": {
            f"{value:g}": scaled(solar, value)
            for value in ECONOMIC_ROBUSTNESS_MULTIPLIERS
        },
    }
    counterfactuals = {
        "default": default,
        "no_methane_endowment": default - methane,
        "no_declared_wear": default - wear,
        "service_plus_solar_only": math.fsum(
            components[name]
            for name in (
                "delivered_service_value",
                "unmet_delivery_penalty",
                "solar_opportunity_cost",
            )
        ),
    }
    default_sign = _sign(default)
    sign_results = {
        group: {key: _sign(value) for key, value in values.items()}
        for group, values in multiplier_results.items()
    }
    sign_results["counterfactuals"] = {
        key: _sign(value) for key, value in counterfactuals.items()
    }
    flips = [
        f"{group}.{key}"
        for group, values in sign_results.items()
        for key, value in values.items()
        if key != "default" and value != default_sign
    ]
    return {
        "unit": "EUR; M minus H",
        "component_deltas_eur": components,
        "counterfactual_deltas_eur": counterfactuals,
        "one_factor_multiplier_deltas_eur": multiplier_results,
        "higher_arm": sign_results,
        "ranking_flips_from_default": flips,
    }


def _arm_accounting_checks(
    run: Mapping[str, Any],
    *,
    assumptions: EconomicAssumptionsV1,
) -> dict[str, bool]:
    economic = _mapping("economic metrics", run["economic_metrics"])
    components = _mapping("economic components", economic["components_eur"])
    totals = _mapping("interval totals", run["interval_totals"])
    if set(components) != set(ECONOMIC_COMPONENT_NAMES):
        raise ValueError("economic component set differs from the frozen ledger")
    checks = {
        "reward_trace": _close(
            _finite("reward trace sum", run["reward_trace_sum_eur"]),
            _finite("net operating value", economic["net_operating_value_eur"]),
            tolerance=ECONOMIC_ROBUSTNESS_ACCOUNTING_TOLERANCE_EUR,
        ),
        "component_sum": _close(
            math.fsum(
                _finite(f"component {key}", value) for key, value in components.items()
            ),
            _finite("net operating value", economic["net_operating_value_eur"]),
            tolerance=ECONOMIC_ROBUSTNESS_ACCOUNTING_TOLERANCE_EUR,
        ),
    }
    for name in (
        "delivered_kg",
        "unmet_kg",
        "accepted_solar_kwh",
        "battery_throughput_kwh",
        "electrolyser_equivalent_operating_h",
    ):
        checks[f"interval_{name}"] = _close(
            _finite(f"interval {name}", totals[name]),
            _finite(f"episode {name}", economic[name]),
            tolerance=ECONOMIC_ROBUSTNESS_ACCOUNTING_TOLERANCE_EUR,
        )
    checks["methane_liability"] = _close(
        _finite(
            "methane liability component",
            components["methane_endowment_liability_change"],
        ),
        -_finite(
            "terminal methane deficit",
            economic["terminal_ch4_equivalent_deficit_kg"],
        )
        * assumptions.methane_endowment_value_eur_per_kg_ch4_equivalent,
        tolerance=ECONOMIC_ROBUSTNESS_ACCOUNTING_TOLERANCE_EUR,
    )
    checks["battery_liability"] = _close(
        _finite(
            "battery liability component",
            components["battery_endowment_liability_change"],
        ),
        -_finite("terminal battery deficit", economic["terminal_battery_deficit_kwh"])
        * assumptions.battery_endowment_value_eur_per_kwh,
        tolerance=ECONOMIC_ROBUSTNESS_ACCOUNTING_TOLERANCE_EUR,
    )
    return checks


def _material_driver_flags(deltas: Mapping[str, float]) -> dict[str, bool]:
    thresholds = _mapping(
        "materiality thresholds",
        _mapping("robustness thresholds", ECONOMIC_ROBUSTNESS_CONTRACT["thresholds"])[
            "raw_driver_materiality"
        ],
    )
    return {
        name: abs(_finite(f"driver delta {name}", deltas[name]))
        > _finite(f"driver threshold {name}", threshold)
        for name, threshold in thresholds.items()
    }


def evaluate_robustness_cell(
    cell: EconomicRobustnessCellV1,
    heuristic: Mapping[str, Any],
    mpc: Mapping[str, Any],
    *,
    assumptions: EconomicAssumptionsV1,
) -> dict[str, Any]:
    """Apply the frozen per-cell portions of R1--R8."""

    if heuristic.get("arm") != "H" or mpc.get("arm") != "M":
        raise ValueError("robustness pair must be ordered H then M")
    h_controller = _mapping("H controller", heuristic["controller"])
    m_controller = _mapping("M controller", mpc["controller"])
    if h_controller.get("id") != LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID:
        raise ValueError("H uses an undeclared controller")
    if m_controller.get("id") != SERVICE_MPC_SETPOINT_CONTROLLER_ID:
        raise ValueError("M uses an undeclared controller")

    expected_settings_sha256 = _expected_mpc_settings_sha256(cell)
    m_settings = _mapping("M controller settings", m_controller["settings"])
    if m_settings.get("sha256") != expected_settings_sha256:
        raise ValueError("M controller settings differ from the frozen cell")
    verify_heuristic_arm_evidence(heuristic)
    verify_mpc_setpoint_run_digests(dict(_mapping("M evidence", mpc["evidence"])))

    expected_decisions = cell.expected_decisions_per_arm
    expected_steps = cell.expected_physical_steps_per_arm
    runs = (heuristic, mpc)
    complete = all(
        run.get("status") == "complete"
        and _integer("completed decisions", run["completed_decisions"])
        == expected_decisions
        and _integer("physical steps", run["physical_steps"]) == expected_steps
        and _boolean("terminated", run.get("terminated")) is True
        and _boolean("truncated", run.get("truncated")) is False
        and run.get("failure") is None
        and _integer("interval execution count", run["execution_count"])
        == expected_decisions
        and _integer("minute application count", run["application_count"])
        == expected_steps
        for run in runs
    )
    h_evidence = _mapping("H evidence", heuristic["evidence"])
    h_executions = h_evidence["executions"]
    if not isinstance(h_executions, list):
        raise TypeError("H interval executions must be a list")
    h_interval_shape = all(
        _close(
            _finite(
                "H command hold duration",
                _mapping(
                    "H execution command",
                    _mapping("H execution", execution)["command"],
                )["hold_duration_s"],
            ),
            900.0,
            tolerance=1e-9,
        )
        and _close(
            _finite(
                "H execution physics step",
                _mapping("H execution", execution)["physics_step_s"],
            ),
            60.0,
            tolerance=1e-9,
        )
        and _integer(
            "H execution application count",
            _mapping("H execution", execution)["application_count"],
        )
        == 15
        for execution in h_executions
    )
    h_complete = (
        _integer("H evidence decision count", h_evidence["decision_count"])
        == expected_decisions
        and _integer("H evidence execution count", h_evidence["execution_count"])
        == expected_decisions
        and _integer("H evidence application count", h_evidence["application_count"])
        == expected_steps
        and h_interval_shape
    )
    m_evidence = _mapping("M evidence", mpc["evidence"])
    m_counts = _mapping("M evidence counts", m_evidence["counts"])
    m_complete = (
        m_evidence.get("status") == "complete"
        and m_evidence.get("controller_id") == SERVICE_MPC_SETPOINT_CONTROLLER_ID
        and m_evidence.get("settings_sha256") == expected_settings_sha256
        and _integer("M evidence expected decisions", m_evidence["expected_decisions"])
        == expected_decisions
        and _integer(
            "M evidence completed decisions", m_evidence["completed_decisions"]
        )
        == expected_decisions
        and _boolean("M pending transition", m_evidence["pending_transition"]) is False
        and m_evidence.get("runner_failure_phase") is None
        and _integer("M closed decision count", m_counts["closed_decision_count"])
        == expected_decisions
        and _integer(
            "M successful decision count", m_counts["successful_decision_count"]
        )
        == expected_decisions
        and _integer("M failed decision count", m_counts["failed_decision_count"]) == 0
        and _integer("M fallback action count", m_counts["fallback_action_count"]) == 0
        and _integer(
            "M successful solver stages", m_counts["successful_solver_stage_count"]
        )
        == 2 * expected_decisions
        and _integer(
            "M attempted solver stages", m_counts["attempted_solver_stage_count"]
        )
        == 2 * expected_decisions
        and _integer("M interval execution count", m_counts["interval_execution_count"])
        == expected_decisions
        and _integer("M physical minute count", m_counts["physical_minute_count"])
        == expected_steps
    )
    identity_fields = {
        "manifest_sha256",
        "physical_exogenous_episode_sha256",
        "economic_episode_sha256",
        "checkpoint_contract_sha256",
        "economic_assumptions_sha256",
    }
    h_identity = _mapping("H manifest identity", heuristic["manifest_identity"])
    m_identity = _mapping("M manifest identity", mpc["manifest_identity"])
    if set(h_identity) != identity_fields or set(m_identity) != identity_fields:
        raise ValueError(
            "paired manifest identity fields differ from the frozen schema"
        )
    if any(not isinstance(value, str) or not value for value in h_identity.values()):
        raise ValueError("H manifest identity contains an empty value")
    if any(not isinstance(value, str) or not value for value in m_identity.values()):
        raise ValueError("M manifest identity contains an empty value")
    identity_equal = dict(h_identity) == dict(m_identity)
    h_accounting = _arm_accounting_checks(heuristic, assumptions=assumptions)
    m_accounting = _arm_accounting_checks(mpc, assumptions=assumptions)
    accounting = all(h_accounting.values()) and all(m_accounting.values())

    safety_by_arm: dict[str, dict[str, int]] = {}
    safety = True
    for run in runs:
        arm = str(run["arm"])
        physical = _mapping(f"{arm} physical metrics", run["physical_metrics"])
        values = {
            "shutdown_steps": _integer("shutdown steps", physical["shutdown_steps"]),
            "out_of_range_decisions": _integer(
                "out-of-range decisions", physical["out_of_range_decisions"]
            ),
            "forced_stops": _integer(
                "forced stops", physical["electrolyser_forced_stops"]
            ),
        }
        safety_by_arm[arm] = values
        safety = safety and all(value == 0 for value in values.values())
    m_authority_names = (
        "accounting_violation_count",
        "chain_violation_count",
        "requested_target_overshoot_minutes",
        "reachable_tracking_violation_minutes",
        "unattributed_target_deficit_minutes",
        "applied_support_authority_violation_minutes",
    )
    m_authority = {
        name: _integer(f"M {name}", m_counts[name]) for name in m_authority_names
    }
    safety = safety and all(value == 0 for value in m_authority.values())

    deltas = paired_metric_deltas(heuristic, mpc)
    sensitivities = economic_sensitivities(heuristic, mpc)
    delivered_delta = deltas["delivered_kg"]
    unmet_delta = deltas["unmet_kg"]
    service_comparable = (
        abs(delivered_delta) <= ECONOMIC_ROBUSTNESS_SERVICE_TOLERANCE_KG
        and abs(unmet_delta) <= ECONOMIC_ROBUSTNESS_SERVICE_TOLERANCE_KG
    )
    material = _material_driver_flags(deltas)
    task_expression = (
        service_comparable
        and abs(deltas["net_operating_value_eur"])
        > ECONOMIC_ROBUSTNESS_VALUE_SEPARATION_EUR
        and any(material.values())
    )
    no_methane_delta = _finite(
        "no-methane value delta",
        _mapping("counterfactual deltas", sensitivities["counterfactual_deltas_eur"])[
            "no_methane_endowment"
        ],
    )
    nonterminal_material = any(
        material[name]
        for name in (
            "accepted_solar_kwh",
            "battery_throughput_kwh",
            "electrolyser_equivalent_operating_h",
        )
    )
    nonterminal_expression = (
        service_comparable
        and abs(no_methane_delta) > ECONOMIC_ROBUSTNESS_VALUE_SEPARATION_EUR
        and nonterminal_material
    )
    settlements = {
        arm: _integer(
            f"{arm} settled commitments",
            _mapping(f"{arm} physical metrics", run["physical_metrics"])[
                "commitments_settled"
            ],
        )
        for arm, run in (("H", heuristic), ("M", mpc))
    }
    settlements_complete = all(
        value == cell.expected_settlements_per_arm for value in settlements.values()
    )
    return {
        "cell": cell.to_dict(),
        "expected": {
            "decisions_per_arm": expected_decisions,
            "physical_steps_per_arm": expected_steps,
            "solver_stages_M": 2 * expected_decisions,
            "settlements_per_arm": cell.expected_settlements_per_arm,
        },
        "execution": {
            "passed": complete and h_complete and m_complete,
            "arm_completion": {
                str(run["arm"]): {
                    "status": run["status"],
                    "decisions": run["completed_decisions"],
                    "physical_steps": run["physical_steps"],
                    "executions": run["execution_count"],
                    "applications": run["application_count"],
                    "failure": run["failure"],
                }
                for run in runs
            },
            "M_counts": dict(m_counts),
            "H_evidence_complete": h_complete,
            "M_evidence_complete": m_complete,
        },
        "identity_accounting": {
            "passed": identity_equal and accounting,
            "paired_identity": identity_equal,
            "H": h_accounting,
            "M": m_accounting,
        },
        "safety_authority": {
            "passed": safety,
            "arms": safety_by_arm,
            "M_authority": m_authority,
        },
        "service_comparability": {
            "passed": service_comparable,
            "M_minus_H_delivered_kg": delivered_delta,
            "M_minus_H_unmet_kg": unmet_delta,
        },
        "task_expression": {
            "passed": task_expression,
            "M_minus_H_net_operating_value_eur": deltas["net_operating_value_eur"],
            "material_driver_flags": material,
        },
        "nonterminal_expression": {
            "passed": nonterminal_expression,
            "M_minus_H_no_methane_endowment_eur": no_methane_delta,
            "material_nonterminal_driver": nonterminal_material,
        },
        "settlements": {
            "passed": settlements_complete,
            "by_arm": settlements,
        },
        "paired_metric_deltas": deltas,
        "economic_sensitivities": sensitivities,
        "latency": arm_latency_evidence(heuristic, mpc),
    }


def _series_summary(
    values_by_seed: Sequence[tuple[int, float]],
    *,
    expected_count: int | None = None,
) -> dict[str, Any]:
    expected = (
        len(values_by_seed)
        if expected_count is None
        else _integer("expected series count", expected_count)
    )
    if len(values_by_seed) > expected:
        raise ValueError("series contains more values than expected")
    if not values_by_seed:
        return {
            "count": 0,
            "complete_pair_count": 0,
            "missing_pair_count": expected,
            "values_by_seed": [],
            "mean": None,
            "population_std": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "minimum_seed": None,
            "maximum_seed": None,
            "worst_seed_by_raw_M_minus_H": None,
            "positive_count": 0,
            "negative_count": 0,
            "tie_count": 0,
        }
    pairs = [
        (int(seed), _finite("series value", value)) for seed, value in values_by_seed
    ]
    if len({seed for seed, _ in pairs}) != len(pairs):
        raise ValueError("series seeds must be unique")
    values = [value for _, value in pairs]
    minimum_seed, minimum = min(pairs, key=lambda item: (item[1], item[0]))
    maximum_seed, maximum = max(pairs, key=lambda item: (item[1], -item[0]))
    signs = [_sign(value) for value in values]
    return {
        "count": len(values),
        "complete_pair_count": len(values),
        "missing_pair_count": expected - len(values),
        "values_by_seed": [{"seed": seed, "M_minus_H": value} for seed, value in pairs],
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
        "median": statistics.median(values),
        "minimum": minimum,
        "maximum": maximum,
        "minimum_seed": minimum_seed,
        "maximum_seed": maximum_seed,
        "worst_seed_by_raw_M_minus_H": minimum_seed,
        "positive_count": signs.count("M"),
        "negative_count": signs.count("H"),
        "tie_count": signs.count("tie"),
    }


def _sensitivity_summaries(
    analyses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    counterfactual_names = (
        "default",
        "no_methane_endowment",
        "no_declared_wear",
        "service_plus_solar_only",
    )
    multiplier_groups = (
        "methane_endowment",
        "declared_wear",
        "solar_opportunity",
    )

    def seed_of(analysis: Mapping[str, Any]) -> int:
        return _integer(
            "sensitivity seed", _mapping("analysis cell", analysis["cell"])["seed"]
        )

    counterfactuals = {
        name: _series_summary(
            [
                (
                    seed_of(analysis),
                    _finite(
                        f"counterfactual {name}",
                        _mapping(
                            "counterfactual deltas",
                            _mapping(
                                "economic sensitivities",
                                analysis["economic_sensitivities"],
                            )["counterfactual_deltas_eur"],
                        )[name],
                    ),
                )
                for analysis in analyses
            ],
            expected_count=len(ECONOMIC_ROBUSTNESS_SEEDS),
        )
        for name in counterfactual_names
    }
    multipliers = {
        group: {
            f"{multiplier:g}": _series_summary(
                [
                    (
                        seed_of(analysis),
                        _finite(
                            f"{group} multiplier {multiplier:g}",
                            _mapping(
                                f"{group} multiplier deltas",
                                _mapping(
                                    "one-factor multiplier deltas",
                                    _mapping(
                                        "economic sensitivities",
                                        analysis["economic_sensitivities"],
                                    )["one_factor_multiplier_deltas_eur"],
                                )[group],
                            )[f"{multiplier:g}"],
                        ),
                    )
                    for analysis in analyses
                ],
                expected_count=len(ECONOMIC_ROBUSTNESS_SEEDS),
            )
            for multiplier in ECONOMIC_ROBUSTNESS_MULTIPLIERS
        }
        for group in multiplier_groups
    }
    direction_and_flips = []
    for analysis in analyses:
        sensitivity = _mapping(
            "economic sensitivities", analysis["economic_sensitivities"]
        )
        higher = _mapping("higher arm", sensitivity["higher_arm"])
        direction_and_flips.append(
            {
                "seed": seed_of(analysis),
                "default_higher_arm": _mapping(
                    "counterfactual higher arm", higher["counterfactuals"]
                )["default"],
                "higher_arm": dict(higher),
                "ranking_flips_from_default": list(
                    sensitivity["ranking_flips_from_default"]
                ),
            }
        )
    return {
        "counterfactual_deltas_eur": counterfactuals,
        "one_factor_multiplier_deltas_eur": multipliers,
        "direction_and_flips_by_seed": direction_and_flips,
    }


def _latency_stratum_summaries(
    analyses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    paths = {
        "H.arm_elapsed_ns": ("H", "arm_elapsed_ns"),
        "H.controller_mean_ns": ("H", "controller", "mean_ns"),
        "H.dispatcher_mean_ns": ("H", "dispatcher", "mean_ns"),
        "M.arm_elapsed_ns": ("M", "arm_elapsed_ns"),
        "M.decision_mean_ns": ("M", "decision", "mean_ns"),
        "M.solver_stage_mean_ns": ("M", "solver_stage", "mean_ns"),
        "M.dispatcher_mean_ns": ("M", "dispatcher", "mean_ns"),
    }
    result: dict[str, Any] = {}
    for name, path in paths.items():
        values: list[tuple[int, float]] = []
        for analysis in analyses:
            seed = _integer(
                "latency seed", _mapping("analysis cell", analysis["cell"])["seed"]
            )
            current: object = analysis["latency"]
            for part in path:
                current = _mapping(f"latency path {name}", current)[part]
            if current is not None:
                values.append((seed, _finite(f"latency {name}", current)))
        summary = _series_summary(
            values,
            expected_count=len(ECONOMIC_ROBUSTNESS_SEEDS),
        )
        summary["values_by_seed"] = [
            {"seed": item["seed"], "nanoseconds": item["M_minus_H"]}
            for item in summary["values_by_seed"]
        ]
        summary["unit"] = "ns"
        summary.pop("positive_count")
        summary.pop("negative_count")
        summary.pop("tie_count")
        summary.pop("worst_seed_by_raw_M_minus_H")
        result[name] = summary
    return result


def aggregate_robustness_cells(
    cells: Sequence[Mapping[str, Any]],
    *,
    source_valid: bool,
) -> dict[str, Any]:
    """Build frozen R1--R8 from compact verified cell summaries."""

    expected_ids = [cell.id for cell in REGISTERED_ECONOMIC_ROBUSTNESS_CELLS]
    observed_ids = [
        str(_mapping("cell summary cell", item["cell"])["id"]) for item in cells
    ]
    declared_matrix = observed_ids == expected_ids
    strata: dict[str, Any] = {}
    for horizon in ECONOMIC_ROBUSTNESS_HORIZONS_H:
        for rate in ECONOMIC_ROBUSTNESS_RATES_KG_H:
            key = f"h{horizon:g}-r{rate:g}"
            selected = [
                item
                for item in cells
                if _mapping("cell summary cell", item["cell"])["horizon_h"] == horizon
                and _mapping("cell summary cell", item["cell"])["delivery_rate_kg_h"]
                == rate
            ]
            analyses = [
                _mapping("cell analysis", item["analysis"])
                for item in selected
                if isinstance(item.get("analysis"), Mapping)
            ]
            metrics: dict[str, Any] = {}
            for metric in PAIRED_METRIC_PATHS:
                values = [
                    (
                        _integer(
                            "summary seed",
                            _mapping("analysis cell", analysis["cell"])["seed"],
                        ),
                        _finite(
                            f"paired metric {metric}",
                            _mapping(
                                "paired metric deltas",
                                analysis["paired_metric_deltas"],
                            )[metric],
                        ),
                    )
                    for analysis in analyses
                ]
                metrics[metric] = _series_summary(
                    values,
                    expected_count=len(ECONOMIC_ROBUSTNESS_SEEDS),
                )
            no_methane_values = [
                (
                    _integer(
                        "summary seed",
                        _mapping("analysis cell", analysis["cell"])["seed"],
                    ),
                    _finite(
                        "no-methane delta",
                        _mapping(
                            "nonterminal expression", analysis["nonterminal_expression"]
                        )["M_minus_H_no_methane_endowment_eur"],
                    ),
                )
                for analysis in analyses
            ]
            comparable = sum(
                _mapping("service comparability", analysis["service_comparability"])[
                    "passed"
                ]
                is True
                for analysis in analyses
            )
            expressing = sum(
                _mapping("task expression", analysis["task_expression"])["passed"]
                is True
                for analysis in analyses
            )
            nonterminal = sum(
                _mapping("nonterminal expression", analysis["nonterminal_expression"])[
                    "passed"
                ]
                is True
                for analysis in analyses
            )
            strata[key] = {
                "horizon_h": horizon,
                "delivery_rate_kg_h": rate,
                "expected_cells": len(ECONOMIC_ROBUSTNESS_SEEDS),
                "observed_cells": len(selected),
                "analyzable_cells": len(analyses),
                "missing_or_unanalyzable_cells": (
                    len(ECONOMIC_ROBUSTNESS_SEEDS) - len(analyses)
                ),
                "artifact_integrity_all": len(selected)
                == len(ECONOMIC_ROBUSTNESS_SEEDS)
                and all(item.get("artifact_verified") is True for item in selected),
                "execution_all": len(analyses) == len(ECONOMIC_ROBUSTNESS_SEEDS)
                and all(
                    _mapping("execution", analysis["execution"])["passed"] is True
                    for analysis in analyses
                ),
                "identity_accounting_all": len(analyses)
                == len(ECONOMIC_ROBUSTNESS_SEEDS)
                and all(
                    _mapping("identity accounting", analysis["identity_accounting"])[
                        "passed"
                    ]
                    is True
                    for analysis in analyses
                ),
                "safety_authority_all": len(analyses) == len(ECONOMIC_ROBUSTNESS_SEEDS)
                and all(
                    _mapping("safety authority", analysis["safety_authority"])["passed"]
                    is True
                    for analysis in analyses
                ),
                "settlements_all": len(analyses) == len(ECONOMIC_ROBUSTNESS_SEEDS)
                and all(
                    _mapping("settlements", analysis["settlements"])["passed"] is True
                    for analysis in analyses
                ),
                "service_comparable_count": comparable,
                "task_expression_count": expressing,
                "nonterminal_expression_count": nonterminal,
                "service_comparability_pass": comparable
                >= ECONOMIC_ROBUSTNESS_REQUIRED_SEEDS_PER_STRATUM,
                "task_expression_pass": expressing
                >= ECONOMIC_ROBUSTNESS_REQUIRED_SEEDS_PER_STRATUM,
                "nonterminal_expression_pass": nonterminal
                >= ECONOMIC_ROBUSTNESS_REQUIRED_SEEDS_PER_STRATUM,
                "paired_metric_summaries": metrics,
                "no_methane_value_summary": _series_summary(
                    no_methane_values,
                    expected_count=len(ECONOMIC_ROBUSTNESS_SEEDS),
                ),
                "economic_sensitivity_summaries": _sensitivity_summaries(analyses),
                "latency_by_seed": [
                    {
                        "seed": _integer(
                            "latency seed",
                            _mapping("analysis cell", analysis["cell"])["seed"],
                        ),
                        "latency": analysis["latency"],
                    }
                    for analysis in analyses
                ],
                "latency_summaries": _latency_stratum_summaries(analyses),
            }

    all_strata = list(strata.values())
    r1_pass = (
        source_valid
        and declared_matrix
        and len(cells) == len(REGISTERED_ECONOMIC_ROBUSTNESS_CELLS)
        and all(item["execution_all"] for item in all_strata)
    )
    r2_pass = r1_pass and all(
        item["artifact_integrity_all"] and item["identity_accounting_all"]
        for item in all_strata
    )
    r3_pass = r2_pass and all(item["safety_authority_all"] for item in all_strata)
    r4_pass = r3_pass and all(item["service_comparability_pass"] for item in all_strata)
    r5_pass = r4_pass and all(item["task_expression_pass"] for item in all_strata)
    r6_pass = r5_pass and all(
        item["nonterminal_expression_pass"] for item in all_strata
    )
    day_strata = [item for item in all_strata if item["horizon_h"] == 24.0]
    r7_pass = (
        r6_pass
        and len(day_strata) == len(ECONOMIC_ROBUSTNESS_RATES_KG_H)
        and all(
            item["execution_all"]
            and item["identity_accounting_all"]
            and item["safety_authority_all"]
            and item["service_comparability_pass"]
            and item["task_expression_pass"]
            and item["nonterminal_expression_pass"]
            and item["settlements_all"]
            for item in day_strata
        )
    )
    flips = {
        str(_mapping("cell summary cell", item["cell"])["id"]): list(
            _mapping(
                "economic sensitivities",
                _mapping("cell analysis", item["analysis"])["economic_sensitivities"],
            )["ranking_flips_from_default"]
        )
        for item in cells
        if isinstance(item.get("analysis"), Mapping)
    }
    horizon_comparison: dict[str, Any] = {}
    for rate in ECONOMIC_ROBUSTNESS_RATES_KG_H:
        short = strata[f"h6-r{rate:g}"]
        day = strata[f"h24-r{rate:g}"]
        short_default = _mapping(
            "six-hour default summary",
            _mapping("six-hour metric summaries", short["paired_metric_summaries"])[
                "net_operating_value_eur"
            ],
        )
        day_default = _mapping(
            "24-hour default summary",
            _mapping("24-hour metric summaries", day["paired_metric_summaries"])[
                "net_operating_value_eur"
            ],
        )
        short_no_methane = _mapping(
            "six-hour no-methane summary", short["no_methane_value_summary"]
        )
        day_no_methane = _mapping(
            "24-hour no-methane summary", day["no_methane_value_summary"]
        )

        def mean_change(
            short_summary: Mapping[str, Any], day_summary: Mapping[str, Any]
        ) -> float | None:
            short_mean = short_summary.get("mean")
            day_mean = day_summary.get("mean")
            if short_mean is None or day_mean is None:
                return None
            return _finite("24-hour mean", day_mean) - _finite(
                "six-hour mean", short_mean
            )

        horizon_comparison[f"r{rate:g}"] = {
            "delivery_rate_kg_h": rate,
            "default_net_value_M_minus_H_eur": {
                "six_hour": dict(short_default),
                "twenty_four_hour": dict(day_default),
                "change_in_mean_24h_minus_6h": mean_change(short_default, day_default),
            },
            "no_methane_M_minus_H_eur": {
                "six_hour": dict(short_no_methane),
                "twenty_four_hour": dict(day_no_methane),
                "change_in_mean_24h_minus_6h": mean_change(
                    short_no_methane, day_no_methane
                ),
            },
            "direction_threshold": None,
        }

    def gate(
        identifier: str, passed: bool, observation: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "id": identifier,
            "status": "passed" if passed else "failed",
            "observation": dict(observation),
        }

    gates = [
        gate(
            "R1",
            r1_pass,
            {
                "source_valid": source_valid,
                "declared_matrix": declared_matrix,
                "observed_cells": len(cells),
                "expected_cells": len(REGISTERED_ECONOMIC_ROBUSTNESS_CELLS),
            },
        ),
        gate(
            "R2",
            r2_pass,
            {
                key: {
                    "artifact_integrity_all": value["artifact_integrity_all"],
                    "identity_accounting_all": value["identity_accounting_all"],
                }
                for key, value in strata.items()
            },
        ),
        gate(
            "R3",
            r3_pass,
            {key: value["safety_authority_all"] for key, value in strata.items()},
        ),
        gate(
            "R4",
            r4_pass,
            {key: value["service_comparable_count"] for key, value in strata.items()},
        ),
        gate(
            "R5",
            r5_pass,
            {key: value["task_expression_count"] for key, value in strata.items()},
        ),
        gate(
            "R6",
            r6_pass,
            {
                key: value["nonterminal_expression_count"]
                for key, value in strata.items()
            },
        ),
        gate(
            "R7",
            r7_pass,
            {
                key: {
                    "settlements_all": value["settlements_all"],
                    "service_comparability_pass": value["service_comparability_pass"],
                    "task_expression_pass": value["task_expression_pass"],
                    "nonterminal_expression_pass": value["nonterminal_expression_pass"],
                }
                for key, value in strata.items()
                if value["horizon_h"] == 24.0
            },
        ),
        {
            "id": "R8",
            "status": "reported",
            "observation": {
                "ranking_flips_from_default_by_cell": flips,
                "sensitivity_summaries_by_stratum": {
                    key: value["economic_sensitivity_summaries"]
                    for key, value in strata.items()
                },
                "claim_boundary": ECONOMIC_ROBUSTNESS_CONTRACT["claim_boundary"],
            },
        },
    ]
    return {
        "status": (
            "complete_all_gates_passed"
            if all(gate["status"] == "passed" for gate in gates[:-1])
            else "complete_gate_failed"
        ),
        "source_valid": source_valid,
        "declared_matrix": declared_matrix,
        "strata": strata,
        "horizon_comparison_by_delivery_rate": horizon_comparison,
        "gates": gates,
    }


def deterministic_arm_evidence(run: Mapping[str, Any]) -> dict[str, Any]:
    """Retain outcomes/evidence identities while excluding every elapsed time."""

    arm = str(run["arm"])
    evidence = _mapping(f"{arm} evidence", run["evidence"])
    if arm == "H":
        selected_evidence = {
            key: evidence[key]
            for key in (
                "decision_count",
                "execution_count",
                "application_count",
                "decision_sha256",
                "command_sha256",
                "execution_sha256",
                "decisions",
                "executions",
            )
            if key in evidence
        }
    elif arm == "M":
        selected_evidence = {
            key: evidence[key]
            for key in (
                "format",
                "status",
                "controller_id",
                "settings_sha256",
                "completed_decisions",
                "expected_decisions",
                "pending_transition",
                "runner_failure_phase",
                "final_observation_sha256",
                "counts",
                "target_tracking",
                "target_deficit_attribution_kwh",
                "energy_kwh",
                "event_sha256",
                "execution_sha256",
                "application_sha256",
                "evidence_graph_sha256",
                "validation_failure",
            )
            if key in evidence
        }
    else:
        raise ValueError("unknown robustness arm")
    return {
        "arm": arm,
        "status": run["status"],
        "controller_id": _mapping(f"{arm} controller", run["controller"])["id"],
        "manifest_identity": run["manifest_identity"],
        "completed_decisions": run["completed_decisions"],
        "physical_steps": run["physical_steps"],
        "terminated": run["terminated"],
        "truncated": run["truncated"],
        "reward_trace_eur": run["reward_trace_eur"],
        "reward_trace_sum_eur": run["reward_trace_sum_eur"],
        "interval_totals": run["interval_totals"],
        "execution_count": run["execution_count"],
        "application_count": run["application_count"],
        "physical_metrics": run["physical_metrics"],
        "economic_metrics": run["economic_metrics"],
        "failure": run["failure"],
        "evidence": selected_evidence,
    }


def cell_deterministic_body(payload: Mapping[str, Any]) -> dict[str, Any]:
    arms = _mapping("cell arms", payload["arms"])
    analysis = payload["analysis"]
    if analysis is not None:
        analysis = json.loads(canonical_json(dict(_mapping("cell analysis", analysis))))
        if isinstance(analysis, dict):
            analysis.pop("latency", None)
    return {
        "format": payload["format"],
        "contract_sha256": payload["contract_sha256"],
        "cell": payload["cell"],
        "episode": payload["episode"],
        "mpc_settings": payload["mpc_settings"],
        "manifest_identity": payload["manifest_identity"],
        "arms": {
            "H": deterministic_arm_evidence(_mapping("H arm", arms["H"])),
            "M": deterministic_arm_evidence(_mapping("M arm", arms["M"])),
        },
        "analysis": analysis,
        "failure": payload["failure"],
    }


def verify_robustness_cell_payload(
    payload: Mapping[str, Any],
    *,
    require_registered: bool = True,
) -> None:
    """Deeply recompute one published cell and its controller evidence."""

    if payload.get("format") != ECONOMIC_ROBUSTNESS_CELL_FORMAT:
        raise ValueError("robustness cell format differs")
    if payload.get("contract") != ECONOMIC_ROBUSTNESS_CONTRACT:
        raise ValueError("robustness cell contract differs")
    if payload.get("contract_sha256") != ECONOMIC_ROBUSTNESS_CONTRACT_SHA256:
        raise ValueError("robustness cell contract digest differs")
    cell = EconomicRobustnessCellV1.from_dict(
        _mapping("serialized robustness cell", payload["cell"])
    )
    if require_registered and cell not in REGISTERED_ECONOMIC_ROBUSTNESS_CELLS:
        raise ValueError("artifact contains an undeclared robustness cell")
    if canonical_json(payload.get("episode")) != canonical_json(episode_for_cell(cell)):
        raise ValueError("robustness cell episode differs from its canonical task")
    if canonical_json(payload.get("mpc_settings")) != canonical_json(
        mpc_settings_for_cell(cell)
    ):
        raise ValueError("robustness cell MPC settings differ")
    arms = _mapping("cell arms", payload["arms"])
    heuristic = _mapping("H arm", arms["H"])
    mpc = _mapping("M arm", arms["M"])

    failure = payload.get("failure")
    if failure is not None:
        _mapping("cell failure", failure)
    verify_heuristic_arm_evidence(heuristic)
    m_evidence = _mapping("M evidence", mpc["evidence"])
    if m_evidence.get("format") is not None:
        verify_mpc_setpoint_run_digests(dict(m_evidence))
    elif failure is None:
        raise ValueError("non-failed robustness cell omits M evidence")

    paired_identity = payload.get("manifest_identity")
    if paired_identity is not None:
        identity = dict(_mapping("paired manifest identity", paired_identity))
        if heuristic.get("manifest_identity") != identity:
            raise ValueError("H identity differs from the paired cell identity")
        if mpc.get("manifest_identity") != identity:
            raise ValueError("M identity differs from the paired cell identity")
    elif failure is None:
        raise ValueError("non-failed robustness cell omits paired identity")

    analysis = payload.get("analysis")
    if analysis is None:
        if failure is None:
            raise ValueError("non-failed robustness cell omits analysis")
        expected_status = "failed"
    else:
        if failure is not None:
            raise ValueError(
                "analyzed robustness cell also records orchestration failure"
            )
        analysis_mapping = _mapping("cell analysis", analysis)
        expected_analysis = evaluate_robustness_cell(
            cell,
            heuristic,
            mpc,
            assumptions=EconomicAssumptionsV1(),
        )
        if analysis != expected_analysis:
            raise ValueError("robustness cell analysis differs from recomputation")
        expected_status = (
            "complete"
            if _mapping("cell execution", analysis_mapping["execution"])["passed"]
            is True
            else "failed"
        )
    if payload.get("status") != expected_status:
        raise ValueError("robustness cell status differs from its evidence")
    _integer("robustness cell elapsed_ns", payload.get("elapsed_ns"))
    for key in ("started_at_utc", "completed_at_utc"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"robustness cell {key} is missing")
    expected_deterministic = domain_sha256(
        "economic-task-robustness-cell-evidence-v1",
        cell_deterministic_body(payload),
    )
    if payload.get("deterministic_evidence_sha256") != expected_deterministic:
        raise ValueError("robustness cell deterministic evidence digest differs")
    body = dict(payload)
    recorded_sha = str(body.pop("sha256"))
    if domain_sha256(ECONOMIC_ROBUSTNESS_CELL_FORMAT, body) != recorded_sha:
        raise ValueError("robustness cell report digest differs")


__all__ = [
    "ECONOMIC_COMPONENT_NAMES",
    "ECONOMIC_ROBUSTNESS_CELL_FORMAT",
    "ECONOMIC_ROBUSTNESS_CONTRACT",
    "ECONOMIC_ROBUSTNESS_CONTRACT_SHA256",
    "ECONOMIC_ROBUSTNESS_FORMAT",
    "ECONOMIC_ROBUSTNESS_HORIZONS_H",
    "ECONOMIC_ROBUSTNESS_LOCK_SHA256",
    "ECONOMIC_ROBUSTNESS_MPC_SETTINGS_SHA256",
    "ECONOMIC_ROBUSTNESS_MULTIPLIERS",
    "ECONOMIC_ROBUSTNESS_PLANNING_HORIZON_H",
    "ECONOMIC_ROBUSTNESS_PROTOCOL_PATH",
    "ECONOMIC_ROBUSTNESS_PROTOCOL_SHA256",
    "ECONOMIC_ROBUSTNESS_RATES_KG_H",
    "ECONOMIC_ROBUSTNESS_SEEDS",
    "EconomicRobustnessCellV1",
    "PAIRED_METRIC_PATHS",
    "REGISTERED_ECONOMIC_ROBUSTNESS_CELLS",
    "aggregate_robustness_cells",
    "arm_latency_evidence",
    "cell_deterministic_body",
    "deterministic_arm_evidence",
    "economic_sensitivities",
    "episode_for_cell",
    "evaluate_robustness_cell",
    "mpc_settings_for_cell",
    "paired_metric_deltas",
    "registered_cells",
    "verify_heuristic_arm_evidence",
    "verify_robustness_cell_payload",
]

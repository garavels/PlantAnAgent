"""Admission of one approved research playback, without executing research code.

The source and endpoint identities describe the evidence. The separately pinned
presentation digest binds the actual projected frames to the reviewed export;
a self-signed checksum alone must not grant research authority.
"""

from __future__ import annotations

from math import isclose
from typing import Any

SCHEMA = "plant-research-playback-v1"
EVIDENCE_LEVEL = (
    "verified simulated research playback; selected successful held-out week, "
    "not deployment qualification"
)
SOURCE_SHA256 = "68229f0f6ad24de5a541409bad9e671b66ec998c57ca38bf31a5e0d2f647a548"
BENCHMARK_SHA256 = "deae125d30f30cf63860aeaec76db566dffee18e6523de2fa508b26f57dafe7e"
CHECKPOINT_SHA256 = "6c36cc24692a7d46528436f6f5fabcff1e1bed2862153a49eb5e1bd8cc49ba27"
# These bind the reviewed deterministic export. Exporters may seal a candidate,
# but cannot replace this admission pin through a recording or API request.
APPROVED_RECORDING_ID = "cd1ff361-33ac-5f84-9c94-8363ffdbe12f"
APPROVED_PRESENTATION_SHA256 = (
    "dce9429fe21d39774f5eb30155bab726ffafc3a41a6a74fbb98ad77255ce95ae"
)
ENVIRONMENT_ID = "plant-env-feasible25-offgrid-v1"

_STOCK_FIELDS = {
    "battery": "battery_energy_kwh",
    "h2": "h2_level_kg",
    "co2": "co2_level_kg",
    "ch4": "ch4_level_kg",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _close(a: float, b: float) -> bool:
    return isclose(a, b, rel_tol=0.0, abs_tol=1e-8)


def _provenance(value: dict[str, Any]) -> None:
    expected = {
        "source_recording_sha256": SOURCE_SHA256,
        "benchmark_contract_sha256": BENCHMARK_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "scenario_seed": 9_930_000,
        "optimizer_seed": 101,
    }
    _require(
        all(value.get(key) == item for key, item in expected.items()),
        "unapproved research provenance",
    )
    _require(
        value["mission"]
        == {
            "delivery_rate_kg_h": 25,
            "horizon_h": 168,
            "deliveries_met": 28,
            "deliveries_total": 28,
            "daily_returns_met": 7,
            "daily_returns_total": 7,
            "mission_passed": True,
            "hard_safety_passed": True,
        }
        and value["mission"]["mission_passed"] is True
        and value["mission"]["hard_safety_passed"] is True,
        "research mission evidence mismatch",
    )
    _require(
        value["evaluation"] == {"split": "held_out", "passed": 14, "total": 16},
        "research evaluation evidence mismatch",
    )
    _require(
        value["electricity"]
        == {
            "mode": "dedicated_pv_no_export",
            "marginal_eur_per_mwh": 0,
            "fixed_costs_included": False,
        }
        and value["electricity"]["fixed_costs_included"] is False,
        "research electricity evidence mismatch",
    )


def _delivery(frame: dict[str, Any], index: int) -> None:
    delivery = frame["delivery"]
    settled_count = index // 24
    _require(
        delivery["delivered_kg"] == settled_count * 150
        and delivery["requested_kg"] == settled_count * 150
        and delivery["unmet_kg"] == 0
        and len(delivery["commitments"]) == 28,
        "research delivery totals mismatch",
    )
    for ordinal, item in enumerate(delivery["commitments"], start=1):
        settled = ordinal <= settled_count
        _require(
            item["due_hours"] == ordinal * 6
            and item["requested_kg"] == 150
            and item["status"] == ("met" if settled else "pending")
            and item["delivered_kg"] == (150 if settled else None)
            and item["unmet_kg"] == (0 if settled else None),
            "research delivery boundary evidence mismatch",
        )


def _history(frame: dict[str, Any], point: dict[str, Any], index: int) -> None:
    state = frame["state"]
    expected = {
        "elapsed_hours": index / 4,
        "solar_kw": state["solar_available_kw"],
        "clear_sky_kw": state["solar_potential_kw"],
        "h2_kg": state["h2_level_kg"],
        "co2_kg": state["co2_level_kg"],
        "battery_soc": state["battery_soc"],
        "ch4_kg": state["ch4_level_kg"],
        "delivered_kg": frame["delivery"]["delivered_kg"],
        "net_value_eur": frame["economics"]["net_value_eur"],
    }
    _require(point == expected, "research history differs from boundary evidence")


def _daily_returns(frames: list[dict[str, Any]]) -> None:
    cycles = frames[-1]["cycle_settlements"]
    _require(len(cycles) == 7, "research daily return count mismatch")
    initial = frames[0]["state"]
    for ordinal, cycle in enumerate(cycles, start=1):
        _require(
            cycle["cycle_index"] == ordinal
            and cycle["feasible"] is True
            and set(cycle["targets"]) == set(_STOCK_FIELDS)
            and set(cycle["actual"]) == set(_STOCK_FIELDS)
            and set(cycle["deficits"]) == set(_STOCK_FIELDS)
            and set(cycle["tolerances"]) == set(_STOCK_FIELDS),
            "research daily return evidence mismatch",
        )
        state = frames[ordinal * 96]["state"]
        for key, field in _STOCK_FIELDS.items():
            target, actual = cycle["targets"][key], cycle["actual"][key]
            tolerance, deficit = cycle["tolerances"][key], cycle["deficits"][key]
            _require(
                _close(target, initial[field])
                and _close(actual, state[field])
                and 0 <= tolerance <= 1e-6
                and _close(deficit, max(0.0, target - actual))
                and deficit <= tolerance,
                "research daily stock return mismatch",
            )
    for index, frame in enumerate(frames):
        _require(
            frame["cycle_settlements"] == cycles[: index // 96],
            "research daily return boundary mismatch",
        )


def validate_research(record: dict[str, Any]) -> dict[str, Any]:
    """Validate a checksummed envelope; caller checks its canonical checksum."""
    _provenance(record["provenance"])
    snap, frames = record["snapshot"], record["frames"]
    env = snap["environment"]
    _require(
        record["evidence_level"] == EVIDENCE_LEVEL
        and snap["schema"] == "plant-v2-demo-v1"
        and snap["mode"] == "recorded_research"
        and snap["status"] == "complete"
        and snap["error"] is None
        and snap["controller"]["id"] == "PPO"
        and env["id"] == ENVIRONMENT_ID
        and env["fixture_seed"] == 9_930_000
        and env["horizon_hours"] == 168
        and env["decision_minutes"] == 15
        and env["physics_minutes"] == 1,
        "research snapshot identity mismatch",
    )
    _require(
        len(frames) == 673
        and len(snap["history"]) == 673
        and frames[-1] == snap["frame"],
        "incomplete research recording boundaries",
    )
    for index, frame in enumerate(frames):
        _require(
            frame["clock"]["decision_index"] == index
            and frame["clock"]["elapsed_hours"] == index / 4
            and frame["clock"]["decision_count"] == 672
            and frame["clock"]["cycle_count"] == 7
            and frame.get("operating_policy") is None,
            "research recording decision sequence mismatch",
        )
        _history(frame, snap["history"][index], index)
        _delivery(frame, index)
        entries = frame["economics"]["entries"]
        components = {item["id"]: item["value_eur"] for item in entries}
        _require(
            len(entries) == len(components)
            and components.get("solar_opportunity_cost") == 0
            and _close(sum(components.values()), frame["economics"]["net_value_eur"]),
            "research economic ledger mismatch",
        )
        forecast = frame["forecasts"]
        _require(
            forecast["price_current_eur_mwh"] == 0
            and all(
                point[key] == 0
                for point in forecast["prices"]
                for key in ("mean", "p10", "p90")
            ),
            "research off-grid price boundary mismatch",
        )
    _daily_returns(frames)
    _require(
        bool(APPROVED_RECORDING_ID)
        and bool(APPROVED_PRESENTATION_SHA256)
        and snap["run_id"] == APPROVED_RECORDING_ID
        and record["sha256"] == APPROVED_PRESENTATION_SHA256,
        "unapproved research presentation",
    )
    return record

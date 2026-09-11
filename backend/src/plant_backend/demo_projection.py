"""Public presentation projections. No forecast/truth generator access."""

from __future__ import annotations

from typing import Any


def weather_view(state: dict[str, Any], elapsed: float) -> dict[str, Any]:
    potential = float(state["solar_potential_kw"])
    available = float(state["solar_available_kw"])
    health = state.get("health", {}).get("solar", {})
    healthy = health.get("status") == "nominal"
    observed = potential > 1.0 and healthy
    factor = max(0.0, min(1.0, available / potential)) if observed else None
    return {
        "sample_end_hours": elapsed,
        "status": (
            "observed" if observed else "night" if potential <= 1.0 else "unattributed"
        ),
        "clear_sky_kw": potential,
        "available_kw": available,
        "resource_factor": factor,
        "attenuation": None if factor is None else 1.0 - factor,
        "curtailed_kw": float(state["solar_curtailed_kw"]),
        "spatial_basis": "synthetic illustration; no geographic cloud observations",
    }


def forecast_view(observation: Any, peak_kw: float) -> dict[str, Any]:
    forecast = observation.plant.forecast
    assert forecast is not None
    return {
        "issued_at_hours": observation.market.issued_at_offset_s / 3600.0,
        "price_current_eur_mwh": observation.market.current_eur_per_mwh,
        "prices": [
            {
                "offset_hours": p.start_offset_h,
                "mean": p.mean_eur_per_mwh,
                "p10": p.p10_eur_per_mwh,
                "p90": p.p90_eur_per_mwh,
            }
            for p in observation.market.bins
        ],
        "solar": [
            {
                "offset_hours": p.start_offset_s_from_issue / 3600.0,
                "end_offset_hours": p.end_offset_s_from_issue / 3600.0,
                "mean_kw": p.mean_peak_power_fraction * peak_kw,
                "p10_kw": p.p10_peak_power_fraction * peak_kw,
            }
            for p in forecast.windows
        ],
    }


def boundary_events(
    previous: dict[str, Any] | None, current: dict[str, Any]
) -> list[dict[str, Any]]:
    """Observed boundary changes only; never inferred fault schedules or future events."""
    index = current["clock"]["decision_index"]
    elapsed = current["clock"]["elapsed_hours"]
    events: list[dict[str, Any]] = []

    def add(kind: str, title: str, detail: str, unit: str | None = None) -> None:
        events.append(
            {
                "id": f"{index}-{len(events)}",
                "decision_index": index,
                "at_hours": elapsed,
                "kind": kind,
                "title": title,
                "detail": detail,
                "unit": unit,
            }
        )

    if previous is None:
        add("run", "Mission started", "Seven days; 240 kg methane every six hours.")
        return events
    old_mode = (previous.get("commitment") or {}).get("mode")
    new_mode = (current.get("commitment") or {}).get("mode")
    if old_mode != new_mode:
        add(
            "equipment",
            f"Electrolyser: {str(new_mode).replace('_', ' ')}",
            f"Previously {str(old_mode).replace('_', ' ')}. Observed at this decision boundary.",
            "electrolyser",
        )
    for unit in ("solar", "battery", "electrolyser", "dac", "methanator"):
        before = previous["state"]["health"].get(unit, {}).get("status")
        after = current["state"]["health"].get(unit, {}).get("status")
        if before != after:
            add(
                "recovery" if after == "nominal" else "warning",
                f"{unit.upper() if unit == 'dac' else unit.capitalize()}: {after}",
                f"Observed status changed from {before}. No recovery time was known ahead.",
                unit,
            )
    for item in current["delivery"]["commitments"]:
        if (
            item["status"] != "pending"
            and previous["clock"]["elapsed_hours"] < item["due_hours"] <= elapsed
        ):
            add(
                "delivery",
                "Delivery met" if item["status"] == "met" else "Delivery shortfall",
                f"{item['delivered_kg']:.1f} / {item['requested_kg']:.1f} kg delivered; {item['unmet_kg']:.1f} kg unmet.",
                "ch4Tank",
            )
    for cycle in current["cycle_settlements"][len(previous["cycle_settlements"]) :]:
        operating = current.get("operating_policy") == "inventory-targets-v1"
        add(
            "cycle",
            f"Day {cycle['cycle_index']} "
            + (
                f"inventory {'restored' if cycle['feasible'] else 'deficit'}"
                if operating
                else f"resource returns {'met' if cycle['feasible'] else 'failed'}"
            ),
            "Battery, H2, CO2 and CH4 checked against the fixed opening reference.",
        )
    decision = current.get("decision")
    operation = (decision or {}).get("operation")
    old_operation = (previous.get("decision") or {}).get("operation")
    if operation and (
        operation["mode"] == "fallback"
        or not old_operation
        or old_operation["mode"] != operation["mode"]
    ):
        failure = operation.get("solver_failure")
        add(
            "warning" if operation["mode"] == "fallback" else "recovery",
            f"Control mode: {operation['mode'].capitalize()}",
            (
                f"Guarded heuristic applied: {failure['reason']}."
                if failure
                else f"Issued {operation['horizon_hours']:g}-hour plan; predicted delivery shortfall {operation['projected_shortfall_kg']:.1f} kg."
            ),
        )
    if decision and decision["guard_adjustments"]:
        add(
            "guard",
            f"{decision['guard_adjustments']} guard adjustments",
            "Inspect the recorded minute execution for applied targets and reasons.",
        )
    before_factor = previous["weather"]["resource_factor"]
    after_factor = current["weather"]["resource_factor"]
    if (
        before_factor is not None
        and after_factor is not None
        and abs(after_factor - before_factor) >= 0.12
    ):
        add(
            "weather",
            "Solar conditions changed",
            f"Available fraction of clear-sky power: {before_factor:.0%} → {after_factor:.0%}.",
            "solar",
        )
    return events

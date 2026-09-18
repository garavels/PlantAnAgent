from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any

import pytest
from plant_sim import PlantConfig, PlantState

from plant_backend.rl.v2.boundary import ControlStateV2
from plant_backend.rl.v2.control import ControllerObservationV2, DeliveryObservation
from plant_backend.rl.v2.economics import EconomicAssumptionsV1
from plant_backend.rl.v2 import full_day_horizon_repair as repair
from plant_backend.rl.v2.weather import ForecastWindowV2, PublicForecastIssueV2


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _forecast() -> PublicForecastIssueV2:
    spans = (("0-1h", 0, 1), ("1-6h", 1, 6), ("6-12h", 6, 12), ("12-24h", 12, 24))
    return PublicForecastIssueV2(
        issued_at_physics_boundary_step=0,
        issued_at_offset_s=0.0,
        windows=tuple(
            ForecastWindowV2(
                name=name,
                start_offset_s_from_issue=start * 3_600.0,
                end_offset_s_from_issue=end * 3_600.0,
                mean_peak_power_fraction=0.0,
                p10_peak_power_fraction=0.0,
            )
            for name, start, end in spans
        ),
    )


def _observation() -> ControllerObservationV2:
    return ControllerObservationV2(
        # The registered economic episode starts at 06:00, so simulator time is
        # deliberately non-zero at decision zero.
        state=ControlStateV2(boundary=PlantState(time_s=21_600.0)),
        delivery=DeliveryObservation(
            steps_to_next=360,
            delivery_period_steps=360,
            next_requested_kg=210.0,
            cumulative_requested_kg=0.0,
            cumulative_delivered_kg=0.0,
        ),
        forecast=_forecast(),
        progress=0.0,
        decision_interval_s=900.0,
    )


def _event_zero(audit: dict[str, Any], *, m24: bool) -> dict[str, Any]:
    identities = audit["identities"]
    result: dict[str, Any] = {
        "decision_index": 0,
        "sha256": _sha("M24-event" if m24 else "M6-event"),
        "observation_sha256": identities["public_observation_sha256"],
        "public_forecast_sha256": identities["public_forecast_issue_sha256"],
        "projection_sha256": identities["forecast_projection_sha256"],
    }
    if m24:
        result.update(
            {
                "horizon_intervals": 96,
                "deliveries": [
                    {
                        "ordinal": index,
                        "due_boundary": 24 * (index + 1),
                        "due_offset_s": 21_600.0 * (index + 1),
                        "requested_kg": 210.0,
                        "source": "announced" if index == 0 else "inferred_recurring",
                    }
                    for index in range(4)
                ],
            }
        )
    return result


def test_frozen_bytes_matrix_and_horizon_only_settings() -> None:
    repository = Path(__file__).resolve().parents[2]
    for path, digest in (
        (
            repair.FULL_DAY_HORIZON_REPAIR_PROTOCOL_PATH,
            repair.FULL_DAY_HORIZON_REPAIR_PROTOCOL_SHA256,
        ),
        (
            repair.FULL_DAY_HORIZON_REPAIR_CLARIFICATION_001_PATH,
            repair.FULL_DAY_HORIZON_REPAIR_CLARIFICATION_001_SHA256,
        ),
        (
            repair.FULL_DAY_HORIZON_REPAIR_CLARIFICATION_002_PATH,
            repair.FULL_DAY_HORIZON_REPAIR_CLARIFICATION_002_SHA256,
        ),
        (
            repair.FULL_DAY_HORIZON_REPAIR_LOCK_PATH,
            repair.FULL_DAY_HORIZON_REPAIR_LOCK_SHA256,
        ),
    ):
        assert hashlib.sha256((repository / path).read_bytes()).hexdigest() == digest
    cells = repair.REGISTERED_FULL_DAY_HORIZON_REPAIR_CELLS
    assert len(cells) == 10
    assert [cell.id for cell in cells[:5]] == [
        f"h24-r35-s{seed}" for seed in range(20, 25)
    ]
    assert [cell.historical_position for cell in cells] == list(range(11, 21))
    assert len(repair.HISTORICAL_REFERENCE_FILE_NAMES) == 20
    fixture = repair.FullDayHorizonRepairCellV1(35, 904)
    assert repair.FullDayHorizonRepairCellV1.from_dict(fixture.to_dict()) == fixture
    settings = repair.verify_horizon_only_settings(
        repair.m6_settings_for_cell(fixture), repair.m24_settings_for_cell(fixture)
    )
    assert settings["configurable_differences"] == ["planning_horizon_h"]
    assert settings["derived_differences"] == ["maximum_planning_intervals", "sha256"]


def test_first_boundary_audit_binds_complete_projection_and_both_events() -> None:
    audit = repair.build_first_boundary_audit(
        _observation(), config=PlantConfig(), interval_s=900.0
    )
    result = repair.verify_first_boundary_audit(
        audit,
        event_zero=_event_zero(audit, m24=True),
        reference_event_zero=_event_zero(audit, m24=False),
    )
    assert result["passed"] is True
    assert len(audit["forecast_projection"]["available_kw"]) == 96

    wrong = _event_zero(audit, m24=True)
    wrong["deliveries"][1]["source"] = "inferred"
    with pytest.raises(ValueError, match="delivery target contract"):
        repair.verify_first_boundary_audit(
            audit,
            event_zero=wrong,
            reference_event_zero=_event_zero(audit, m24=False),
        )

    rehashed = deepcopy(audit)
    rehashed["forecast_projection"]["available_kw"][0] += 1.0
    projection_body = dict(rehashed["forecast_projection"])
    projection_body.pop("sha256")
    projection_sha = repair.domain_sha256(
        repair.FORECAST_TRAJECTORY_DOMAIN, projection_body
    )
    rehashed["forecast_projection"]["sha256"] = projection_sha
    rehashed["identities"]["forecast_projection_sha256"] = projection_sha
    audit_body = dict(rehashed)
    audit_body.pop("sha256")
    rehashed["sha256"] = repair.domain_sha256(
        repair.FIRST_BOUNDARY_AUDIT_DOMAIN, audit_body
    )
    with pytest.raises(ValueError, match="public-issue reconstruction"):
        repair.verify_first_boundary_audit(
            rehashed,
            event_zero=_event_zero(rehashed, m24=True),
            reference_event_zero=_event_zero(rehashed, m24=False),
        )


def _economic_record(
    index: int, delivered: float, energy: dict[str, float]
) -> tuple[dict[str, Any], dict[str, Any]]:
    assumptions = EconomicAssumptionsV1()
    config = PlantConfig()
    tariff = assumptions.tariff.observe(6.0 + index * 0.25).to_dict()
    opportunity_price = tariff["current_eur_per_mwh"]
    effective_price = max(0.0, opportunity_price)
    solar_cost = energy["accepted_solar_kwh"] * effective_price / 1_000.0
    battery_reserve = (
        energy["battery_throughput_kwh"] * assumptions.battery_throughput_eur_per_kwh
    )
    electrolyser_reserve = (
        energy["electrolyser_equivalent_operating_h"]
        * config.electrolyser.rated_kw
        * assumptions.electrolyser_capital_eur_per_kw
        / assumptions.electrolyser_life_h
    )
    signals = {
        "opportunity_price_eur_per_mwh": opportunity_price,
        "effective_opportunity_price_eur_per_mwh": effective_price,
        "accepted_solar_kwh": energy["accepted_solar_kwh"],
        "delivered_kg": delivered,
        "unmet_kg": 0.0,
        "battery_throughput_kwh": energy["battery_throughput_kwh"],
        "battery_equivalent_full_cycles": energy["battery_throughput_kwh"]
        / (2.0 * config.battery.capacity_kwh),
        "electrolyser_equivalent_operating_h": energy[
            "electrolyser_equivalent_operating_h"
        ],
        "methane_endowment_deficit_before_kg": 0.0,
        "methane_endowment_deficit_after_kg": 0.0,
        "battery_endowment_deficit_before_kwh": 0.0,
        "battery_endowment_deficit_after_kwh": 0.0,
        "solar_opportunity_cost_eur": solar_cost,
        "battery_throughput_reserve_eur": battery_reserve,
        "electrolyser_life_reserve_eur": electrolyser_reserve,
    }
    components = {
        "delivered_service_value": 4.0 * delivered,
        "unmet_delivery_penalty": 0.0,
        "solar_opportunity_cost": -solar_cost,
        "battery_throughput_reserve": -battery_reserve,
        "electrolyser_life_reserve": -electrolyser_reserve,
        "methane_endowment_liability_change": 0.0,
        "battery_endowment_liability_change": 0.0,
    }
    breakdown = {
        "objective_version": 1,
        "unit": "EUR per decision interval",
        "total_eur": sum(components.values()),
        "components_eur": components,
        "signals": signals,
    }
    return {**signals, "tariff": tariff}, breakdown


def _ledger_fixture() -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[float],
    dict[str, Any],
    dict[str, Any],
]:
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    trace: list[float] = []
    for index in range(96):
        delivered = 1.0 if index in {23, 47, 71, 95} else 0.0
        settled = int(delivered > 0.0)
        observation = _sha(f"observation-{index}")
        next_observation = _sha(f"observation-{index + 1}")
        boundary = _sha(f"boundary-{index}")
        next_boundary = _sha(f"boundary-{index + 1}")
        boundary_record = {
            "h2_level_kg": 0.0,
            "co2_level_kg": 0.0,
            "ch4_level_kg": 0.0,
            "battery_energy_kwh": 0.0,
        }
        execution = {
            "sha256": _sha(f"execution-{index}"),
            "applications": [
                {
                    "boundary": {"sha256": boundary, "record": boundary_record},
                    "next_boundary": {
                        "sha256": next_boundary,
                        "record": boundary_record,
                    },
                    "commitment": {
                        "before": {"record": None},
                        "after": {"record": None},
                    },
                }
            ],
            "terminal": {"boundary_sha256": next_boundary},
            "energy_kwh": {
                "applied": {
                    "solar_available": 1.0,
                    "solar_curtailed": 0.1,
                    "electrolyser_total": 0.2,
                    "dac": 0.1,
                    "methanator": 0.1,
                    "solar_to_battery": 0.1,
                    "battery_support": 0.05,
                }
            },
        }
        residual = {
            "sha256": _sha(f"residual-{index}"),
            "next_observation_sha256": next_observation,
            "observed_delivered_kg": delivered,
        }
        event = {
            "decision_index": index,
            "sha256": _sha(f"event-{index}"),
            "observation_sha256": observation,
            "public_forecast_sha256": _sha(f"forecast-{index}"),
            "projection_sha256": _sha(f"projection-{index}"),
            "deliveries": (
                [
                    {
                        "ordinal": 0,
                        "due_boundary": 1,
                        "due_offset_s": 900.0,
                        "requested_kg": 1.0,
                        "source": "announced",
                    }
                ]
                if settled
                else []
            ),
            "setpoints": {"sha256": _sha(f"setpoint-{index}")},
            "execution": execution,
            "residual": residual,
        }
        energy = {
            "accepted_solar_kwh": 0.9,
            "accepted_process_energy_kwh": 0.4,
            "curtailed_kwh": 0.1,
            "battery_throughput_kwh": 0.15,
            "battery_support_kwh": 0.05,
            "electrolyser_equivalent_operating_h": 0.0,
        }
        delivery = {
            "commitments_settled": settled,
            "requested_kg": delivered,
            "delivered_kg": delivered,
            "unmet_kg": 0.0,
        }
        economic_interval, breakdown = _economic_record(index, delivered, energy)
        rows.append(
            repair.build_interval_ledger_row(
                decision_index=index,
                bindings={
                    "event_sha256": event["sha256"],
                    "observation_sha256": observation,
                    "public_forecast_sha256": event["public_forecast_sha256"],
                    "projection_sha256": event["projection_sha256"],
                    "setpoints_sha256": event["setpoints"]["sha256"],
                    "execution_sha256": execution["sha256"],
                    "residual_sha256": residual["sha256"],
                    "boundary_sha256": boundary,
                    "next_observation_sha256": next_observation,
                    "next_boundary_sha256": next_boundary,
                },
                reward_eur=breakdown["total_eur"],
                delivery=delivery,
                economic_interval=economic_interval,
                reward_breakdown=breakdown,
                energy=energy,
            )
        )
        events.append(event)
        trace.append(breakdown["total_eur"])
    run = {
        "events": events,
        "event_sha256": [event["sha256"] for event in events],
        "final_observation_sha256": _sha("observation-96"),
        "energy_kwh": {"applied": {"battery_support": 4.8}},
    }
    physical = {
        "requested_kg": 4.0,
        "commitments_settled": 4,
        "accepted_process_energy_kwh": 38.4,
        "curtailed_kwh": 9.6,
    }
    components = {
        name: sum(row["reward_breakdown"]["components_eur"][name] for row in rows)
        for name in repair.ECONOMIC_COMPONENT_NAMES
    }
    economic = {
        "net_operating_value_eur": sum(trace),
        "components_eur": components,
        "delivered_kg": 4.0,
        "unmet_kg": 0.0,
        "accepted_solar_kwh": 86.4,
        "battery_throughput_kwh": 14.4,
        "electrolyser_equivalent_operating_h": 0.0,
    }
    return rows, run, trace, physical, economic


def test_interval_ledger_recomputes_rows_totals_cadence_and_boundary_chain() -> None:
    rows, run, trace, physical, economic = _ledger_fixture()
    ledger = repair.build_interval_ledger(
        rows,
        run_evidence=run,
        reward_trace_eur=trace,
        physical_metrics=physical,
        economic_metrics=economic,
    )
    totals = repair.verify_interval_ledger(
        ledger,
        run_evidence=run,
        reward_trace_eur=trace,
        physical_metrics=physical,
        economic_metrics=economic,
    )
    assert totals["delivered_kg"] == 4.0
    assert totals["commitments_settled"] == 4

    redistributed = deepcopy(rows)
    changed_economic, changed_breakdown = _economic_record(
        23, 0.0, redistributed[23]["energy"]
    )
    redistributed[23] = repair.build_interval_ledger_row(
        decision_index=23,
        bindings=redistributed[23]["bindings"],
        reward_eur=changed_breakdown["total_eur"],
        delivery={
            "commitments_settled": 1,
            "requested_kg": 0.0,
            "delivered_kg": 0.0,
            "unmet_kg": 0.0,
        },
        economic_interval=changed_economic,
        reward_breakdown=changed_breakdown,
        energy=redistributed[23]["energy"],
    )
    changed_trace = [
        changed_breakdown["total_eur"] if i == 23 else value
        for i, value in enumerate(trace)
    ]
    changed_components = {
        name: sum(
            row["reward_breakdown"]["components_eur"][name] for row in redistributed
        )
        for name in repair.ECONOMIC_COMPONENT_NAMES
    }
    with pytest.raises(ValueError, match="due event target"):
        repair.build_interval_ledger(
            redistributed,
            run_evidence=run,
            reward_trace_eur=changed_trace,
            physical_metrics={**physical, "requested_kg": 3.0},
            economic_metrics={
                **economic,
                "net_operating_value_eur": sum(changed_trace),
                "components_eur": changed_components,
                "delivered_kg": 3.0,
            },
        )

    physical_attack = deepcopy(rows)
    attacked_energy = dict(physical_attack[0]["energy"])
    attacked_energy["accepted_solar_kwh"] += 1.0
    attacked_economic, attacked_breakdown = _economic_record(0, 0.0, attacked_energy)
    physical_attack[0] = repair.build_interval_ledger_row(
        decision_index=0,
        bindings=physical_attack[0]["bindings"],
        reward_eur=attacked_breakdown["total_eur"],
        delivery=physical_attack[0]["delivery"],
        economic_interval=attacked_economic,
        reward_breakdown=attacked_breakdown,
        energy=attacked_energy,
    )
    attacked_trace = [attacked_breakdown["total_eur"], *trace[1:]]
    attacked_components = {
        name: sum(
            row["reward_breakdown"]["components_eur"][name] for row in physical_attack
        )
        for name in repair.ECONOMIC_COMPONENT_NAMES
    }
    with pytest.raises(ValueError, match="execution energy"):
        repair.build_interval_ledger(
            physical_attack,
            run_evidence=run,
            reward_trace_eur=attacked_trace,
            physical_metrics=physical,
            economic_metrics={
                **economic,
                "net_operating_value_eur": sum(attacked_trace),
                "components_eur": attacked_components,
                "accepted_solar_kwh": economic["accepted_solar_kwh"] + 1.0,
            },
        )


def test_series_summary_reports_missingness_all_ties_and_no_absolute_signs() -> None:
    summary = repair.summarize_rate_values(
        [(904, 2.0), (905, 1.0), (906, 1.0), (907, 2.0)],
        contrast=False,
        expected_seed_ids=(904, 905, 906, 907, 908),
    )
    assert summary["missing_seed_ids"] == [908]
    assert summary["minimum_seed_ids"] == [905, 906]
    assert summary["maximum_seed_ids"] == [904, 907]
    assert summary["positive_count"] == repair.NOT_APPLICABLE

    contrast = repair.summarize_rate_values(
        [(904, -1.0), (905, 0.0), (906, 2.0)],
        contrast=True,
        expected_seed_ids=(904, 905, 906),
    )
    assert (
        contrast["negative_count"],
        contrast["tie_count"],
        contrast["positive_count"],
    ) == (1, 1, 1)


@pytest.mark.parametrize(
    ("value", "passes"),
    [
        (repair.FULL_DAY_HORIZON_REPAIR_SERVICE_TOLERANCE_KG, False),
        (
            pytest.param(
                1.0000001 * repair.FULL_DAY_HORIZON_REPAIR_SERVICE_TOLERANCE_KG,
                True,
                id="strictly-above-positive-delivery-threshold",
            )
        ),
    ],
)
def test_positive_fourth_delivery_threshold_is_strict(
    value: float, passes: bool
) -> None:
    arms = {arm: _compact_arm(arm) for arm in ("H", "M6", "M24")}
    _set_m24_total_delivery(arms, 4.0 * value)
    result = _evaluate(repair.FullDayHorizonRepairCellV1(35.0, 905), arms)
    assert result["intrinsic"]["F5_positive_fourth_delivery"] is passes


def _service(requested: float, delivered: float, unmet: float) -> dict[str, Any]:
    return {
        "requested_kg": requested,
        "delivered_kg": delivered,
        "unmet_kg": unmet,
        "delivery_fill": 1.0 if requested == 0.0 else delivered / requested,
        "commitments_settled": 4,
        "commitments_met": 4,
        "worst_commitment_fill": 1.0 if requested == 0.0 else delivered / requested,
    }


def _settlements(delivered_total: float) -> list[dict[str, Any]]:
    delivered = delivered_total / 4.0
    unmet = 1.0 - delivered
    return [
        {
            "ordinal": ordinal,
            "requested_kg": 1.0,
            "delivered_kg": delivered,
            "unmet_kg": unmet,
            "fill": delivered,
        }
        for ordinal in range(1, 5)
    ]


def _compact_arm(arm: str) -> dict[str, Any]:
    source = _sha(f"{arm}-source-arm")

    def retained(value: object) -> dict[str, Any]:
        return repair.measurement(repair.AVAILABLE, value, source_arm_sha256=source)

    unavailable = {
        "settlements": repair.NOT_RETAINED if arm == "H" else repair.AVAILABLE,
        "post_settlement_states": (
            repair.NOT_RETAINED if arm == "H" else repair.AVAILABLE
        ),
        "first_decision": repair.NOT_APPLICABLE if arm == "H" else repair.AVAILABLE,
        "interval_economic_components": (
            repair.AVAILABLE if arm == "M24" else repair.NOT_RETAINED
        ),
        "final_settlement_balance": (
            repair.NOT_RETAINED if arm == "H" else repair.AVAILABLE
        ),
    }
    delivered = {"H": 4.0, "M6": 3.0, "M24": 4.0}[arm]
    unmet = 4.0 - delivered
    components = {name: 0.0 for name in repair.ECONOMIC_COMPONENT_NAMES}
    components["delivered_service_value"] = 4.0 * delivered
    components["unmet_delivery_penalty"] = -20.0 * unmet
    accepted_solar = 0.0
    if arm == "M24":
        accepted_solar = 2.0
        components["solar_opportunity_cost"] = -2.0
    physical = {name: 0.0 for name in repair._REQUIRED_PHYSICAL_FIELDS}
    economic = {
        "net_operating_value_eur": sum(components.values()),
        "components_eur": components,
        "accepted_solar_kwh": accepted_solar,
        "battery_throughput_kwh": 0.0,
        "electrolyser_equivalent_operating_h": 0.0,
        "terminal_ch4_equivalent_deficit_kg": 0.0,
        "terminal_battery_deficit_kwh": 0.0,
    }
    settlement_value = _settlements(delivered)
    state_value = {
        "2": {"h2_kg": 0.0, "co2_kg": 0.0, "ch4_kg": 1.0, "battery_energy_kwh": 0.0},
        "3": {"h2_kg": 0.0, "co2_kg": 0.0, "ch4_kg": 1.0, "battery_energy_kwh": 0.0},
    }
    fourth_delivered = settlement_value[3]["delivered_kg"]
    balance_value = {
        "ch4_before_final_interval_kg": fourth_delivered,
        "ch4_production_delta_kg": 0.0,
        "ch4_after_settlement_kg": 0.0,
        "residual_delivered_kg": fourth_delivered,
        "interval_delivered_kg": fourth_delivered,
    }

    def field_measurement(field: str) -> dict[str, Any]:
        availability = unavailable[field]
        if availability != repair.AVAILABLE:
            return repair.measurement(availability)
        values: dict[str, object] = {
            "settlements": settlement_value,
            "post_settlement_states": state_value,
            "first_decision": {"decision_index": 0},
            "interval_economic_components": {"row_count": 96},
            "final_settlement_balance": balance_value,
        }
        return retained(values[field])

    solver_stage = (
        repair.measurement(repair.NOT_APPLICABLE)
        if arm == "H"
        else retained({"count": 192, "p99_ns": 1, "maximum_ns": 1})
    )
    return {
        "service": _service(4.0, delivered, unmet),
        "settlements": field_measurement("settlements"),
        "post_settlement_states": field_measurement("post_settlement_states"),
        "first_decision": field_measurement("first_decision"),
        "interval_economic_components": field_measurement(
            "interval_economic_components"
        ),
        "final_settlement_balance": field_measurement("final_settlement_balance"),
        "physical": physical,
        "economic": economic,
        "latency": {
            "arm_elapsed_ns": 1,
            "controller": {"count": 96, "p99_ns": 1, "maximum_ns": 1},
            "solver_stage": solver_stage,
            "dispatcher": {"count": 1_440, "p99_ns": 1, "maximum_ns": 1},
        },
    }


def _citations(cell: repair.FullDayHorizonRepairCellV1) -> dict[str, Any]:
    return {
        "root_report": {
            "raw_file_sha256": _sha("root-raw"),
            "report_sha256": _sha("root-report"),
            "deterministic_evidence_sha256": _sha("root-deterministic"),
            "contract_sha256": _sha("root-contract"),
        },
        "historical_cell": {
            "file_name": cell.historical_file_name or "fixture.json",
            "raw_file_sha256": _sha(f"{cell.id}-raw"),
            "report_sha256": _sha(f"{cell.id}-report"),
            "deterministic_evidence_sha256": _sha(f"{cell.id}-deterministic"),
        },
        "H": {
            "controller_id": "heuristic-v2",
            "evidence_sha256": _sha(f"{cell.id}-H-evidence"),
            "deterministic_arm_sha256": _sha(f"{cell.id}-H-deterministic"),
        },
        "M6": {
            "controller_id": repair.SERVICE_MPC_SETPOINT_CONTROLLER_ID,
            "settings_sha256": repair.FULL_DAY_HORIZON_REPAIR_M6_SETTINGS_SHA256,
            "run_evidence_sha256": _sha(f"{cell.id}-M6-run"),
            "evidence_graph_sha256": _sha(f"{cell.id}-M6-graph"),
            "deterministic_arm_sha256": _sha(f"{cell.id}-M6-deterministic"),
        },
        "M24": {
            "controller_id": repair.SERVICE_MPC_SETPOINT_CONTROLLER_ID,
            "settings_sha256": repair.FULL_DAY_HORIZON_REPAIR_M24_SETTINGS_SHA256,
            "interval_evidence_sha256": _sha(f"{cell.id}-M24-interval"),
            "run_evidence_sha256": _sha(f"{cell.id}-M24-run"),
            "evidence_graph_sha256": _sha(f"{cell.id}-M24-graph"),
        },
    }


def _cell_evidence() -> dict[str, Any]:
    return {
        "artifact_verified": True,
        "status": "complete",
        "terminated": True,
        "truncated": False,
        "failure": None,
        "completed_decisions": 96,
        "physical_minutes": 1_440,
        "settlements": 4,
        "successful_solver_stages": 192,
        "closed_transitions": True,
        "evidence_cardinality_exact": True,
        "fallback_count": 0,
        "reset_identity_equal_reference": True,
        "reward_accounting_closed": True,
        "component_accounting_closed": True,
        "interval_accounting_closed": True,
        "terminal_liability_accounting_closed": True,
        "safety_counts_reconstructed": True,
        "shutdown_count": 0,
        "out_of_range_count": 0,
        "forced_stop_count": 0,
        "chain_violation_count": 0,
        "accounting_violation_count": 0,
        "requested_target_overshoot_count": 0,
        "reachable_tracking_violation_count": 0,
        "support_authority_violation_count": 0,
        "unattributed_target_deficit_count": 0,
        "settings_sha256": repair.FULL_DAY_HORIZON_REPAIR_M24_SETTINGS_SHA256,
        "settings_horizon_only_valid": True,
        "first_boundary_audit_valid": True,
        "environment_identity_equal_m6": True,
        "scipy_version": repair.FULL_DAY_HORIZON_REPAIR_SCIPY_VERSION,
        "highs_version": repair.FULL_DAY_HORIZON_REPAIR_HIGHS_VERSION,
        "python_version_equal_historical": False,
        "platform_equal_historical": False,
    }


def _evaluate(
    cell: repair.FullDayHorizonRepairCellV1,
    arms: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return repair.evaluate_full_day_horizon_repair_cell(
        cell,
        arms=arms or {arm: _compact_arm(arm) for arm in ("H", "M6", "M24")},
        evidence=_cell_evidence(),
        citations=_citations(cell),
    )


def _custody() -> dict[str, Any]:
    return {
        "reference_directory_verified": True,
        "reference_entry_names": [
            "report.json",
            *repair.HISTORICAL_REFERENCE_FILE_NAMES,
        ],
        "historical_executed_revision": repair.HISTORICAL_EXECUTED_REVISION,
        "historical_worktree_identity_sha256": repair.HISTORICAL_WORKTREE_IDENTITY_SHA256,
        "historical_contract_sha256": repair.HISTORICAL_CONTRACT_SHA256,
        "historical_raw_report_sha256": repair.HISTORICAL_RAW_REPORT_SHA256,
        "historical_report_domain_sha256": repair.HISTORICAL_REPORT_DOMAIN_SHA256,
        "historical_deterministic_evidence_sha256": repair.HISTORICAL_DETERMINISTIC_EVIDENCE_SHA256,
        "protocol_sha256": repair.FULL_DAY_HORIZON_REPAIR_PROTOCOL_SHA256,
        "clarification_001_sha256": repair.FULL_DAY_HORIZON_REPAIR_CLARIFICATION_001_SHA256,
        "clarification_002_sha256": repair.FULL_DAY_HORIZON_REPAIR_CLARIFICATION_002_SHA256,
        "dependency_lock_sha256": repair.FULL_DAY_HORIZON_REPAIR_LOCK_SHA256,
        "source_clean_start": True,
        "source_clean_end": True,
        "source_identity_stable": True,
        "reference_cell_ids": [
            cell.id for cell in repair.REGISTERED_FULL_DAY_HORIZON_REPAIR_CELLS
        ],
        "historical_cell_citations": [
            {"file_name": name, "raw_file_sha256": _sha(f"raw-{name}")}
            for name in repair.HISTORICAL_REFERENCE_FILE_NAMES
        ],
        "undeclared_source_paths": [],
        "undeclared_destination_entries": [],
    }


def _set_m24_total_delivery(arms: dict[str, dict[str, Any]], delivered: float) -> None:
    unmet = 4.0 - delivered
    arms["M24"]["service"] = _service(4.0, delivered, unmet)
    rows = _settlements(delivered)
    arms["M24"]["settlements"]["value"] = rows
    fourth = rows[3]["delivered_kg"]
    arms["M24"]["final_settlement_balance"]["value"] = {
        "ch4_before_final_interval_kg": fourth,
        "ch4_production_delta_kg": 0.0,
        "ch4_after_settlement_kg": 0.0,
        "residual_delivered_kg": fourth,
        "interval_delivered_kg": fourth,
    }


def test_cell_evaluator_thresholds_and_citation_self_reference_exclusion() -> None:
    cell = repair.FullDayHorizonRepairCellV1(35.0, 904)
    baseline = _evaluate(cell)
    assert all(baseline["intrinsic"].values())
    assert baseline["runtime_identity_limitations"] == [
        "python_version_differs_from_historical",
        "platform_differs_from_historical",
    ]

    citations = _citations(cell)
    citations["M24"]["raw_file_sha256"] = _sha("circular-self-reference")
    with pytest.raises(ValueError, match="M24 citation fields"):
        repair.verify_cell_citations(cell, citations)

    tolerance = repair.FULL_DAY_HORIZON_REPAIR_SERVICE_TOLERANCE_KG
    for delta, expected in ((0.0, True), (1e-12, False)):
        arms = {arm: _compact_arm(arm) for arm in ("H", "M6", "M24")}
        _set_m24_total_delivery(arms, 3.0 - tolerance - delta)
        result = _evaluate(cell, arms)
        assert result["intrinsic"]["F5_no_regression_M6"] is expected

    for delta, expected in ((0.0, True), (1e-12, False)):
        arms = {arm: _compact_arm(arm) for arm in ("H", "M6", "M24")}
        _set_m24_total_delivery(arms, 4.0 - tolerance - delta)
        result = _evaluate(cell, arms)
        assert result["intrinsic"]["F5_full_request"] is expected
        assert result["intrinsic"]["F5_service_comparable_H"] is expected

    for delta, expected in ((0.0, True), (-1e-12, False)):
        arms = {arm: _compact_arm(arm) for arm in ("H", "M6", "M24")}
        arms["M24"]["post_settlement_states"]["value"]["3"]["ch4_kg"] = (
            1.0 - tolerance + delta
        )
        result = _evaluate(cell, arms)
        assert result["intrinsic"]["F6_post_third_adequacy"] is expected

    for delta, expected in ((0.0, True), (1e-12, False)):
        arms = {arm: _compact_arm(arm) for arm in ("H", "M6", "M24")}
        arms["M24"]["final_settlement_balance"]["value"]["residual_delivered_kg"] = (
            1.0 + tolerance + delta
        )
        result = _evaluate(cell, arms)
        assert result["intrinsic"]["F6_fourth_material_balance"] is expected

    for value, expected in ((1.0, False), (1.0 + 1e-12, True)):
        arms = {arm: _compact_arm(arm) for arm in ("H", "M6", "M24")}
        components = arms["M24"]["economic"]["components_eur"]
        components["solar_opportunity_cost"] = -value
        arms["M24"]["economic"]["net_operating_value_eur"] = sum(components.values())
        result = _evaluate(cell, arms)
        assert result["intrinsic"]["F7_task_expression"] is expected

    for value, expected in ((1.0, False), (1.0 + 1e-12, True)):
        arms = {arm: _compact_arm(arm) for arm in ("H", "M6", "M24")}
        arms["M24"]["economic"]["accepted_solar_kwh"] = value
        result = _evaluate(cell, arms)
        assert result["intrinsic"]["F7_both_expression_rules"] is expected

    for increment, expected in ((0, True), (1, False)):
        arms = {arm: _compact_arm(arm) for arm in ("H", "M6", "M24")}
        arms["M24"]["latency"]["controller"] = {
            "count": 96,
            "p99_ns": repair.FULL_DAY_HORIZON_REPAIR_CONTROLLER_P99_LIMIT_NS
            + increment,
            "maximum_ns": repair.FULL_DAY_HORIZON_REPAIR_CONTROLLER_MAX_LIMIT_NS,
        }
        result = _evaluate(cell, arms)
        assert result["intrinsic"]["F8_latency"] is expected


def _all_cell_summaries() -> list[dict[str, Any]]:
    return [
        {"cell": cell.to_dict(), "analysis": _evaluate(cell)}
        for cell in repair.REGISTERED_FULL_DAY_HORIZON_REPAIR_CELLS
    ]


@pytest.mark.parametrize(
    ("intrinsic_field", "gate_id", "count_field"),
    [
        ("F5_full_request", "F5", "F5_full_request_count"),
        ("F6_post_third_adequacy", "F6", "F6_post_third_adequacy_count"),
        ("F7_both_expression_rules", "F7", "F7_both_expression_rules_count"),
    ],
)
def test_aggregate_per_rate_denominators_are_four_of_five(
    intrinsic_field: str, gate_id: str, count_field: str
) -> None:
    summaries = _all_cell_summaries()
    for offset in (0, 5):
        summaries[offset]["analysis"]["intrinsic"][intrinsic_field] = False
    report = repair.aggregate_full_day_horizon_repair_cells(
        summaries,
        custody=_custody(),
        claim_boundary=repair.FULL_DAY_HORIZON_REPAIR_CLAIM_BOUNDARY,
    )
    assert report["per_rate"]["r35"][count_field] == 4
    assert report["per_rate"]["r42"][count_field] == 4
    gate = next(item for item in report["gates"] if item["id"] == gate_id)
    assert gate["intrinsic_passed"] is True

    summaries[1]["analysis"]["intrinsic"][intrinsic_field] = False
    failed = repair.aggregate_full_day_horizon_repair_cells(
        summaries,
        custody=_custody(),
        claim_boundary=repair.FULL_DAY_HORIZON_REPAIR_CLAIM_BOUNDARY,
    )
    assert failed["per_rate"]["r35"][count_field] == 3
    gate = next(item for item in failed["gates"] if item["id"] == gate_id)
    assert gate["intrinsic_passed"] is False


def test_aggregate_gate_chain_f8_independence_and_registered_missingness() -> None:
    summaries = _all_cell_summaries()
    passed = repair.aggregate_full_day_horizon_repair_cells(
        summaries,
        custody=_custody(),
        claim_boundary=repair.FULL_DAY_HORIZON_REPAIR_CLAIM_BOUNDARY,
    )
    assert passed["status"] == "complete_all_gates_passed"
    retained = passed["per_rate"]["r35"]["registered_measurements"]["M24"][
        "retained_values_by_seed"
    ]
    assert [item["seed"] for item in retained["first_decision"]] == [20, 21, 22, 23, 24]
    assert len(retained["interval_economic_components"]) == 5

    bad_custody = _custody()
    bad_custody["source_identity_stable"] = False
    chained = repair.aggregate_full_day_horizon_repair_cells(
        summaries,
        custody=bad_custody,
        claim_boundary=repair.FULL_DAY_HORIZON_REPAIR_CLAIM_BOUNDARY,
    )
    assert [gate["status"] for gate in chained["gates"][:7]] == ["failed"] * 7
    assert chained["gates"][1]["intrinsic_passed"] is True
    assert chained["gates"][7]["status"] == "passed"

    bad_claim = repair.aggregate_full_day_horizon_repair_cells(
        summaries,
        custody=_custody(),
        claim_boundary=repair.FULL_DAY_HORIZON_REPAIR_CLAIM_BOUNDARY + " expanded",
    )
    assert bad_claim["gates"][7]["status"] == "failed"

    missing = deepcopy(summaries)
    missing[0]["analysis"] = None
    incomplete = repair.aggregate_full_day_horizon_repair_cells(
        missing,
        custody=_custody(),
        claim_boundary=repair.FULL_DAY_HORIZON_REPAIR_CLAIM_BOUNDARY,
    )
    r35 = incomplete["per_rate"]["r35"]
    assert r35["missing_seed_ids"] == [20]
    availability = r35["registered_measurements"]["H"]["availability"]
    assert availability["settlements"]["missing_seed_ids"] == [20]

    ineligible = deepcopy(summaries)
    ineligible[0]["analysis"]["intrinsic"]["F7_economic_comparison_eligible"] = False
    ineligible[0]["analysis"]["intrinsic"]["F7_both_expression_rules"] = False
    filtered = repair.aggregate_full_day_horizon_repair_cells(
        ineligible,
        custody=_custody(),
        claim_boundary=repair.FULL_DAY_HORIZON_REPAIR_CLAIM_BOUNDARY,
    )
    eligibility = filtered["per_rate"]["r35"]["F7_economic_eligibility"]
    assert eligibility["ineligible_seed_ids"] == [20]
    economics = filtered["per_rate"]["r35"]["numeric_summaries"]
    assert economics["M24_minus_H_default_value_eur"]["missing_seed_ids"] == [20]


@pytest.mark.parametrize(("excess", "expected"), [(0.0, True), (1e-12, False)])
def test_mean_unmet_reduction_inclusive_boundary(excess: float, expected: bool) -> None:
    summaries = _all_cell_summaries()
    boundary = 0.10 * 1.0 + repair.FULL_DAY_HORIZON_REPAIR_SERVICE_TOLERANCE_KG
    for summary in summaries:
        summary["analysis"]["service"]["M24"]["unmet_kg"] = boundary + excess
    report = repair.aggregate_full_day_horizon_repair_cells(
        summaries,
        custody=_custody(),
        claim_boundary=repair.FULL_DAY_HORIZON_REPAIR_CLAIM_BOUNDARY,
    )
    assert report["per_rate"]["r35"]["F6_mean_unmet_reduction_pass"] is expected
    assert report["per_rate"]["r42"]["F6_mean_unmet_reduction_pass"] is expected


def test_deterministic_aggregate_excludes_timing_but_retains_physics() -> None:
    aggregate = repair.aggregate_full_day_horizon_repair_cells(
        _all_cell_summaries(),
        custody=_custody(),
        claim_boundary=repair.FULL_DAY_HORIZON_REPAIR_CLAIM_BOUNDARY,
    )

    def identity(value: dict[str, Any]) -> str:
        deterministic = repair.deterministic_full_day_horizon_repair_aggregate(value)
        return repair.domain_sha256("fixture-deterministic-aggregate-v1", deterministic)

    baseline = identity(aggregate)
    timing = deepcopy(aggregate)
    timing["per_rate"]["r35"]["registered_measurements"]["M24"]["latency"][
        "arm_elapsed_ns"
    ]["mean"] += 10_000
    timing["per_rate"]["r35"]["registered_measurements"]["M24"][
        "retained_values_by_seed"
    ]["first_decision"][0]["source_arm_sha256"] = _sha("timing-mutated-arm")
    timing["gates"][7]["status"] = "failed"
    timing["gates"][7]["intrinsic_passed"] = False
    assert identity(timing) == baseline

    physical = deepcopy(aggregate)
    physical["per_rate"]["r35"]["registered_measurements"]["M24"]["physical"][
        "accepted_process_energy_kwh"
    ]["mean"] += 1.0
    assert identity(physical) != baseline

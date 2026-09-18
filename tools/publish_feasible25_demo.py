"""Project one closed, hash-pinned PPO evaluation into read-only demo playback.

No model is loaded, no environment is reset, and no physics is recomputed.
The original experiment and checkpoints are never modified. Presentation data
are drawn only from retained observations, states, commands and settlements.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend/src"))

from plant_backend.demo_projection import boundary_events, weather_view

SOURCE = (
    ROOT
    / "docs/research/results/feasible25_offgrid_20260915/ppo-101-held-out/9930000.json.gz"
)
SOURCE_SHA = "68229f0f6ad24de5a541409bad9e671b66ec998c57ca38bf31a5e0d2f647a548"
CHECKPOINT_SHA = "6c36cc24692a7d46528436f6f5fabcff1e1bed2862153a49eb5e1bd8cc49ba27"
BENCHMARK_SHA = "deae125d30f30cf63860aeaec76db566dffee18e6523de2fa508b26f57dafe7e"
RUN_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, f"plant-research-playback-v1:{SOURCE_SHA}"))
EVIDENCE = "verified simulated research playback; selected successful held-out week, not deployment qualification"


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def close(actual, expected, label, tolerance=1e-7):
    if not math.isfinite(actual) or not math.isclose(
        actual, expected, rel_tol=1e-12, abs_tol=tolerance
    ):
        raise ValueError(f"{label} differs: {actual!r} vs {expected!r}")


def read_source(path=SOURCE):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("Demo source differs from the approved evaluation")
    source = json.loads(gzip.decompress(raw))
    if (source["seed"], source["controller"], source["benchmark_contract_sha256"]) != (
        9930000,
        "coverage-ppo-101",
        BENCHMARK_SHA,
    ):
        raise ValueError("Wrong policy, scenario or mission")
    if not all(
        source[key]
        for key in (
            "execution_complete",
            "mission_passed",
            "hard_safety_passed",
            "every_delivery_passed",
            "all_daily_returns_passed",
        )
    ):
        raise ValueError("The approved demonstration must complete its full mission")
    if (
        len(source["commands"]) != 672
        or len(source["decision_frames"]) != 673
        or len(source["minute_truth_states"]) != 10081
    ):
        raise ValueError("Incomplete recorded trajectory")
    if digest(source["minute_truth_states"]) != source["minute_truth_sha256"]:
        raise ValueError("Minute truth digest differs")
    return source


def forecasts(source, index, config):
    # At the terminal boundary there is no next policy call. Show the last
    # retained issue with its original issue time, rather than inventing one.
    issue = min(index, 671)
    identity = source["controller_configuration"]["policy_identity"]
    fields = identity["base_checkpoint_contract"]["interface_contract"]["base_layout"][
        "observation_fields"
    ]
    vector = source["commands"][issue]["evidence"]["policy_observation"]
    observed = {field["name"]: vector[i] for i, field in enumerate(fields)}
    peak = config["solar"]["peak_kw"]
    prices = []
    for hour in range(24):
        values = {
            q: observed[f"market_price_h{hour:02d}_{q}"] * 300
            for q in ("mean", "p10", "p90")
        }
        if any(values.values()):
            raise ValueError(
                "Nonzero effective price in an off-grid policy observation"
            )
        prices.append({"offset_hours": hour, **values})
    return {
        "issued_at_hours": issue / 4,
        "price_current_eur_mwh": observed["market_price_current"] * 300,
        "prices": prices,
        "solar": [
            {
                "offset_hours": start,
                "end_offset_hours": end,
                "mean_kw": observed[f"solar_forecast_{start}_{end}h_mean"] * peak,
                "p10_kw": observed[f"solar_forecast_{start}_{end}h_p10"] * peak,
            }
            for start, end in ((0, 1), (1, 6), (6, 12), (12, 24))
        ],
        "source": "Decoded from the retained float32 policy input; final boundary retains the last issued forecast.",
    }


def project(source):
    manifest = source["manifest"]
    config = deepcopy(manifest["plant_config"])
    config.update(
        dt_h=config["dt_s"] / 3600,
        steps_per_day=1440,
        electrolyser_min_load_kw=config["electrolyser"]["rated_kw"]
        * config["electrolyser"]["min_load_frac"],
    )
    assumptions = deepcopy(manifest["economic_assumptions"])
    # Publish the effective tariff, never the inactive private synthetic tape.
    assumptions["tariff"] = {
        "id": "dedicated-pv-no-export",
        "hourly_eur_per_mwh": [0.0] * 24,
        "negative_price_semantics": "No electricity import/export settlement.",
    }
    assumptions["electricity_model"] = deepcopy(source["electricity_model"])
    minutes = source["minute_truth_states"]
    opening = minutes[0]
    config["battery"]["initial_soc"] = opening["battery_soc"]
    for tank, key in (
        ("h2_tank", "h2_level_kg"),
        ("co2_tank", "co2_level_kg"),
        ("ch4_tank", "ch4_level_kg"),
    ):
        config[tank]["initial_kg"] = opening[key]
    resource_defs = (
        ("battery", "Battery", "kWh", "battery_energy_kwh"),
        ("h2", "Hydrogen", "kg", "h2_level_kg"),
        ("co2", "Carbon dioxide", "kg", "co2_level_kg"),
        ("ch4", "Methane", "kg", "ch4_level_kg"),
    )
    settlements = source["delivery"]["settlements"]
    eq = lambda s: s["ch4_level_kg"] + min(
        s["h2_level_kg"] / config["methanator"]["h2_kg_per_kg_ch4"],
        s["co2_level_kg"] / config["methanator"]["co2_kg_per_kg_ch4"],
    )
    frames, history, events = [], [], []
    throughput = discharged = process_energy = electrolyser_energy = dac_energy = (
        methane_energy
    ) = 0.0
    reward = 0.0
    previous_net = 0.0
    for index, retained in enumerate(source["decision_frames"]):
        state = deepcopy(retained["state"])
        if index != retained["decision_index"] or state != minutes[index * 15]:
            raise ValueError("Boundary state differs from exact minute truth")
        elapsed = index / 4
        commitment = deepcopy(retained["commitment"])
        interval = (
            minutes[max(0, (index - 1) * 15) + 1 : index * 15 + 1] if index else []
        )
        for minute in interval:
            throughput += (
                minute["solar_to_battery_kw"] + minute["battery_to_electrolyser_kw"]
            ) / 60
            discharged += max(0.0, -minute["battery_power_kw"]) / 60
            electrolyser_energy += minute["electrolyser_power_kw"] / 60
            dac_energy += minute["dac_power_kw"] / 60
            methane_energy += minute["methanator_power_kw"] / 60
        process_energy = electrolyser_energy + dac_energy + methane_energy
        settled = [item for item in settlements if item["due_step"] <= index * 15]
        delivered = sum(item["delivered_kg"] for item in settled)
        unmet = sum(item["unmet_kg"] for item in settled)
        equivalent_h = commitment["operating_time_s"] / 3600 + commitment[
            "low_load_time_s"
        ] / 3600 * (assumptions["electrolyser_low_load_wear_multiplier"] - 1)
        for event in ("cold_start", "hot_start", "aborted_start", "forced_stop"):
            equivalent_h += (
                commitment[event + "s"] * assumptions[event + "_equivalent_operating_h"]
            )
        entries = {
            "delivered_service_value": delivered
            * assumptions["delivered_service_value_eur_per_kg"],
            "unmet_delivery_penalty": -unmet
            * assumptions["unmet_delivery_penalty_eur_per_kg"],
            "solar_opportunity_cost": 0.0,
            "battery_throughput_reserve": -throughput
            * assumptions["battery_throughput_eur_per_kwh"],
            "electrolyser_life_reserve": -equivalent_h
            * config["electrolyser"]["rated_kw"]
            * assumptions["electrolyser_capital_eur_per_kw"]
            / assumptions["electrolyser_life_h"],
            "methane_endowment_liability_change": -max(0, eq(opening) - eq(state))
            * assumptions["methane_endowment_value_eur_per_kg_ch4_equivalent"],
            "battery_endowment_liability_change": -max(
                0, opening["battery_energy_kwh"] - state["battery_energy_kwh"]
            )
            * assumptions["battery_endowment_value_eur_per_kwh"],
        }
        net = sum(entries.values())
        decision = None
        if index:
            command_record = source["commands"][index - 1]
            reward += command_record["reward_eur"]
            close(
                net - previous_net,
                command_record["reward_eur"],
                f"interval {index} economic reward",
            )
            close(net, reward, f"boundary {index} cumulative economic reward")
            command = command_record["command"]
            targets = command["targets"]["setpoints_kw"]
            decision = {
                "at_hours": (index - 1) / 4,
                "controller_id": "PPO",
                "operation": None,
                "computation_ms": None,
                "execution_digest": None,
                "command_digest": command["sha256"],
                "guard_adjustments": None,
                "notes": [
                    "Recorded PPO residual adjustments around the coverage controller.",
                    "Targets held for 15 minutes; applied flows are the retained one-minute plant states.",
                    "Detailed guard reasons and computation timing were not retained.",
                    "Forecast bands are decoded from the recorded policy input.",
                    "Battery charging includes recovered spill and can exceed the direct-charge ceiling.",
                ],
                "input_state": deepcopy(source["decision_frames"][index - 1]["state"]),
                "input_forecasts": forecasts(source, index - 1, config),
                "targets": [
                    {
                        "id": identifier,
                        "label": label,
                        "requested_kw": targets[target],
                        "applied_mean_kw": sum(m[field] for m in interval) / 15,
                    }
                    for identifier, label, target, field in (
                        (
                            "electrolyser",
                            "Electrolyser",
                            "electrolyser",
                            "electrolyser_power_kw",
                        ),
                        ("dac", "Direct air capture", "dac", "dac_power_kw"),
                        (
                            "methanator",
                            "Methanator",
                            "methanator",
                            "methanator_power_kw",
                        ),
                        (
                            "battery",
                            "Battery charge ceiling",
                            "battery_charge_ceiling",
                            "solar_to_battery_kw",
                        ),
                    )
                ],
                "minutes": [
                    {
                        "offset_minutes": ordinal,
                        "solar_kw": m["solar_available_kw"],
                        "electrolyser_kw": m["electrolyser_power_kw"],
                        "dac_kw": m["dac_power_kw"],
                        "methanator_kw": m["methanator_power_kw"],
                        "battery_charge_kw": m["solar_to_battery_kw"],
                        "battery_support_kw": m["battery_to_electrolyser_kw"],
                        "adjustments": None,
                    }
                    for ordinal, m in enumerate(interval)
                ],
            }
        previous_net = net
        metrics = {
            "decisions": index,
            "physical_steps": index * 15,
            "hours": elapsed,
            "solar_kwh": state["energy_produced_kwh"],
            "curtailed_kwh": state["energy_curtailed_kwh"],
            "battery_throughput_kwh": throughput,
            "battery_discharged_kwh": discharged,
            "accepted_process_energy_kwh": process_energy,
            "electrolyser_applied_energy_kwh": electrolyser_energy,
            "dac_applied_energy_kwh": dac_energy,
            "methanator_applied_energy_kwh": methane_energy,
            "ch4_produced_kg": state["ch4_produced_total_kg"],
            "delivered_kg": delivered,
            "unmet_kg": unmet,
            "electrolyser_nonproductive_energy_kwh": commitment["startup_energy_kwh"]
            + commitment["standby_energy_kwh"],
        }
        for key in (
            "operating_time_s",
            "low_load_time_s",
            "cold_starts",
            "hot_starts",
            "forced_stops",
            "aborted_starts",
            "controlled_stops",
            "startup_energy_kwh",
            "standby_energy_kwh",
        ):
            metrics["electrolyser_" + key] = commitment[key]
        next_due = next(
            (i["due_step"] / 60 for i in settlements if i["due_step"] > index * 15),
            None,
        )
        commitments = [
            {
                "due_hours": item["due_step"] / 60,
                "requested_kg": item["requested_kg"],
                "delivered_kg": item["delivered_kg"] if item in settled else None,
                "unmet_kg": item["unmet_kg"] if item in settled else None,
                "status": "met" if item in settled else "pending",
            }
            for item in settlements
        ]
        frame = {
            "state": state,
            "weather": weather_view(state, elapsed),
            "commitment": commitment,
            "commitment_profile": deepcopy(
                manifest["episode_config"]["electrolyser_commitment"]
            ),
            "metrics": metrics,
            "assumptions": deepcopy(assumptions),
            "clock": {
                "elapsed_hours": elapsed,
                "decision_index": index,
                "decision_count": 672,
                "cycle_index": min(index // 96 + 1, 7),
                "cycle_count": 7,
                "hours_to_cycle_end": (96 - index % 96) / 4 if index < 672 else 0.0,
            },
            "delivery": {
                "next_due_hours": next_due,
                "next_requested_kg": 150.0 if next_due is not None else 0.0,
                "delivered_kg": delivered,
                "unmet_kg": unmet,
                "requested_kg": sum(i["requested_kg"] for i in settled),
                "commitments": commitments,
            },
            "forecasts": forecasts(source, index, config),
            "resources": [
                {
                    "id": rid,
                    "label": label,
                    "unit": unit,
                    "current": state[key],
                    "required": opening[key],
                    "margin": state[key] - opening[key],
                }
                for rid, label, unit, key in resource_defs
            ],
            "economics": {
                "net_value_eur": net,
                "entries": [
                    {
                        "id": key,
                        "label": key.replace("_", " ").capitalize(),
                        "value_eur": value,
                    }
                    for key, value in entries.items()
                ],
                "assumption_note": "Dedicated off-grid PV: no electricity purchase/export charge. Benchmark wear and inventory reserves retained; solar capital and fixed O&M unestimated and excluded. Modeled contribution, not business profit.",
            },
            "decision": decision,
            "plan": None,
            "cycle_settlements": deepcopy(
                [
                    c
                    for c in source["cycle_settlements"]
                    if c["boundary_decision"] <= index
                ]
            ),
        }
        if index == 0:
            events.append(
                {
                    "id": "0-0",
                    "decision_index": 0,
                    "at_hours": 0.0,
                    "kind": "run",
                    "title": "Recorded mission started",
                    "detail": "Seven days; 150 kg methane every six hours. Trained PPO 101; scenario 9930000.",
                    "unit": None,
                }
            )
        else:
            events.extend(boundary_events(frames[-1], frame))
        frames.append(frame)
        history.append(
            {
                "elapsed_hours": elapsed,
                "solar_kw": state["solar_available_kw"],
                "clear_sky_kw": state["solar_potential_kw"],
                "h2_kg": state["h2_level_kg"],
                "co2_kg": state["co2_level_kg"],
                "battery_soc": state["battery_soc"],
                "ch4_kg": state["ch4_level_kg"],
                "delivered_kg": delivered,
                "net_value_eur": net,
            }
        )
    for key, value in source["economic_metrics"]["components_eur"].items():
        close(entries[key], value, f"final economic component {key}")
    for key, value in metrics.items():
        if key in source["physical_metrics"]:
            close(
                value, source["physical_metrics"][key], f"final physical metric {key}"
            )
    identity = source["controller_configuration"]["policy_identity"]
    snapshot = {
        "schema": "plant-v2-demo-v1",
        "run_id": RUN_ID,
        "revision": 672,
        "mode": "recorded_research",
        "status": "complete",
        "error": None,
        "controller": {
            "id": "PPO",
            "label": "PPO 101 · successful off-grid week",
            "uses_forecast": True,
            "policy_id": "coverage-residual-full-mission-ppo-v1",
            "checkpoint": {
                "optimizer_seed": 101,
                "training_transitions": 65536,
                "sha256": CHECKPOINT_SHA,
            },
        },
        "available_controllers": [],
        "environment": {
            "id": "plant-env-feasible25-offgrid-v1",
            "observation_count": 139,
            "action_count": 4,
            "decision_minutes": 15,
            "physics_minutes": 1,
            "horizon_hours": 168,
            "fixture_seed": 9930000,
            "interface_digest": digest(identity),
            "task_digest": BENCHMARK_SHA,
        },
        "config": config,
        "frame": frames[-1],
        "history": history,
        "events": events,
    }
    provenance = {
        "source_recording_sha256": SOURCE_SHA,
        "benchmark_contract_sha256": BENCHMARK_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "scenario_seed": 9930000,
        "optimizer_seed": 101,
        "mission": {
            "delivery_rate_kg_h": 25,
            "horizon_h": 168,
            "deliveries_met": 28,
            "deliveries_total": 28,
            "daily_returns_met": 7,
            "daily_returns_total": 7,
            "mission_passed": True,
            "hard_safety_passed": True,
        },
        "evaluation": {"split": "held_out", "passed": 14, "total": 16},
        "electricity": {
            "mode": "dedicated_pv_no_export",
            "marginal_eur_per_mwh": 0,
            "fixed_costs_included": False,
        },
    }
    payload = {
        "schema": "plant-research-playback-v1",
        "snapshot": snapshot,
        "frames": frames,
        "evidence_level": EVIDENCE,
        "provenance": provenance,
    }
    return {**payload, "sha256": digest(payload)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    target = args.output or ROOT / "demo/recordings" / f"{RUN_ID}.json"
    record = project(read_source())
    raw = canonical(record)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != raw:
            raise FileExistsError("Refusing to replace a different published demo")
    else:
        with target.open("xb") as stream:
            stream.write(raw)
    print(
        json.dumps(
            {
                "run_id": RUN_ID,
                "path": str(target),
                "sha256": record["sha256"],
                "bytes": len(raw),
                "frames": len(record["frames"]),
            }
        )
    )


if __name__ == "__main__":
    main()

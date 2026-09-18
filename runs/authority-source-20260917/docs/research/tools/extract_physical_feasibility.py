"""Read-only extraction from existing saved episodes. No simulator invocation.

Research minute row n is a state after interval [n-1,n), with interval power
fields and post-delivery inventories. Output opening state is separate.
Demo rows contain exact interval-integrated energy from retained 15min frames;
no minute forcing or fine-grained DAC fault schedule is inferred for the demo.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO.parents[1] / "PlantAnAgentResearch/adaptive-rl-batch-008/evaluations"


def pin(path):
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


MINUTE_COLUMNS = [
    "step_end",
    "solar_available_kw",
    "solar_potential_kw",
    "dac_offline",
    "dac_degradation",
    "electrolyser_offline",
    "electrolyser_degradation",
    "methanator_offline",
    "methanator_degradation",
    "solar_to_battery_kw",
    "solar_to_electrolyser_kw",
    "solar_to_dac_kw",
    "solar_to_methanator_kw",
    "solar_curtailed_kw",
    "battery_to_electrolyser_kw",
    "battery_power_kw",
    "battery_energy_kwh",
    "h2_level_kg",
    "co2_level_kg",
    "ch4_level_kg",
    "electrolyser_power_kw",
    "dac_power_kw",
    "methanator_power_kw",
    "h2_production_kg_h",
    "co2_capture_kg_h",
    "ch4_production_kg_h",
]


def minute_row(s):
    special = {"step_end": s["step"]}
    for component in ["dac", "electrolyser", "methanator"]:
        for name in ["offline", "degradation"]:
            special[f"{component}_{name}"] = s["health"][component][name]
    return [special[k] if k in special else s[k] for k in MINUTE_COLUMNS]


def common_initial(s):
    return {
        "battery_kwh": s["battery_energy_kwh"],
        "battery_soc": s["battery_soc"],
        "h2_kg": s["h2_level_kg"],
        "co2_kg": s["co2_level_kg"],
        "ch4_kg": s["ch4_level_kg"],
    }


def read_research(seed, root=ROOT):
    path = root / f"{seed}-OperatingMPC" / "episode.json.gz"
    a = json.loads(gzip.decompress(path.read_bytes()))
    s = a["minute_truth_states"]
    m = a["manifest"]
    dt_h = a["config"]["plant"]["dt_s"] / 3600
    assert len(s) == 10081 and all(x["step"] == n for n, x in enumerate(s))
    solar = math.fsum(x["solar_available_kw"] * dt_h for x in s[1:])
    assert abs(solar - s[-1]["energy_produced_kwh"]) < 1e-6
    rows = [minute_row(x) for x in s[1:]]
    schedule = m["disturbances"]["schedule"]
    for n, state in enumerate(s[1:]):
        fault = any(
            f["start_s"] <= n * 60 < f["start_s"] + f["duration_s"] for f in schedule
        )
        assert state["health"]["dac"]["offline"] == fault
        assert state["health"]["dac"]["degradation"] == 0
        assert all(
            v["degradation"] == 0 and not v["offline"]
            for k, v in state["health"].items()
            if k != "dac"
        )
    day = []
    for d in range(1, 8):
        slice_s = s[(d - 1) * 1440 + 1 : d * 1440 + 1]
        day.append(
            {
                "cycle_index": d,
                "solar_available_kwh": math.fsum(
                    x["solar_available_kw"] * dt_h for x in slice_s
                ),
                "dac_outage_h": sum(x["health"]["dac"]["offline"] for x in slice_s)
                * dt_h,
                "settlement": a["summary"]["cycle_settlements"][d - 1],
            }
        )
    decisions = []
    for i in range(672):
        mins = s[i * 15 + 1 : (i + 1) * 15 + 1]
        decisions.append(
            {
                "decision_end": i + 1,
                "elapsed_end_h": (i + 1) * 0.25,
                "solar_available_kwh": math.fsum(
                    x["solar_available_kw"] * dt_h for x in mins
                ),
                "dac_online_h": sum(not x["health"]["dac"]["offline"] for x in mins)
                * dt_h,
            }
        )
    return {
        "seed": seed,
        "kind": "research_saved_full_minute_episode",
        "source": pin(path),
        "source_schema": a["schema"],
        "source_controller": a["controller"]["id"],
        "alpha_physical_exogenous_episode_sha256": m[
            "alpha_physical_exogenous_episode_sha256"
        ],
        "beta_exogenous_episode_sha256": m["beta_exogenous_episode_sha256"],
        "plant_config": a["config"]["plant"],
        "episode_config": a["config"]["episode"],
        "operator_constraints": a["config"]["constraints"],
        "cycle_config": a["config"]["beta_task"]["cycle"],
        "cycle_reference": m["cycle_reference"],
        "cycle_accounting": m["cycle_accounting"],
        "delivery_accounting": m["delivery_accounting"],
        "initial": common_initial(s[0]),
        "initial_commitment": a["initial_observation"]["typed"]["plant"]["state"][
            "electrolyser_commitment"
        ],
        "initial_state": s[0],
        "start_hour": 6.0,
        "physics_dt_s": 60,
        "physics_steps": 10080,
        "commitments": m["episode"]["commitments"],
        "disturbances": m["disturbances"],
        "weekly_solar_available_kwh": solar,
        "recorded_solar_cumulative_kwh": s[-1]["energy_produced_kwh"],
        "solar_integration_residual_kwh": solar - s[-1]["energy_produced_kwh"],
        "minute_columns": MINUTE_COLUMNS,
        "minutes": rows,
        "minute_rows_sha256": canonical_sha(rows),
        "decision_intervals": decisions,
        "days": day,
        "physical_metrics": a["summary"]["physical_metrics"],
        "hard_safety_passed": a["summary"]["hard_safety_passed"],
        "daily_returns_all_passed": a["summary"]["all_cycles_feasible"],
        "final": common_initial(s[-1]),
    }


def read_demo():
    path = REPO / "runs/commission-demo/5a25ba80-6227-4c07-8790-adc38f02a3a1.json"
    a = json.loads(path.read_bytes())
    snap = a["snapshot"]
    frames = a["frames"]
    payload = {k: v for k, v in a.items() if k != "sha256"}
    assert canonical_sha(payload) == a["sha256"]
    assert len(frames) == 673
    s = frames[0]["state"]
    final = frames[-1]
    intervals = []
    for i in range(672):
        before, after = frames[i]["state"], frames[i + 1]["state"]
        intervals.append(
            {
                "decision_end": i + 1,
                "elapsed_end_h": (i + 1) * 0.25,
                "solar_available_kwh": after["energy_produced_kwh"]
                - before["energy_produced_kwh"],
                "endpoint_solar_available_kw": after["solar_available_kw"],
                "endpoint_dac_offline": after["health"]["dac"]["offline"],
                "endpoint_dac_degradation": after["health"]["dac"]["degradation"],
            }
        )
    days = []
    for d in range(1, 8):
        days.append(
            {
                "cycle_index": d,
                "solar_available_kwh": frames[d * 96]["state"]["energy_produced_kwh"]
                - frames[(d - 1) * 96]["state"]["energy_produced_kwh"],
                "settlement": final["cycle_settlements"][d - 1],
            }
        )
    return {
        "seed": 9000034,
        "kind": "public_demo_saved_decision_frames",
        "source": pin(path),
        "source_schema": a["schema"],
        "recording_payload_sha256": a["sha256"],
        "source_controller": snap["controller"]["policy_id"],
        "evidence_level": a["evidence_level"],
        "plant_config": snap["config"],
        "initial": common_initial(s),
        "initial_state": s,
        "initial_commitment": frames[0]["commitment"],
        "commitment_profile": frames[0]["commitment_profile"],
        "start_hour": 6.0,
        "physics_dt_s": 60,
        "physics_steps": 10080,
        "minute_columns": None,
        "minutes": None,
        "limits": [
            "No retained minute truth tape or exact fine-grained DAC schedule in public recording.",
            "Decision interval solar energy is exact cumulative-counter difference, not 15min endpoint quadrature.",
        ],
        "weekly_solar_available_kwh": final["state"]["energy_produced_kwh"],
        "decision_intervals": intervals,
        "days": days,
        "commitments": final["delivery"]["commitments"],
        "physical_metrics": final["metrics"],
        "daily_returns_all_passed": all(
            x["feasible"] for x in final["cycle_settlements"]
        ),
        "final": common_initial(final["state"]),
    }


def build_output(root=ROOT):
    out = {
        "schema": "saved-plant-physical-feasibility-scenarios-v1",
        "description": "Read-only extraction from eight complete previously saved OperatingMPC research episodes and one complete public MPC recording; no new controller or simulator run.",
        "source_files": [
            pin(REPO / "simulator/src/plant_sim/simulator.py"),
            pin(REPO / "backend/src/plant_backend/rl/v2/env.py"),
            pin(REPO / "backend/src/plant_backend/rl/v2/beta_env.py"),
            pin(Path(__file__)),
        ],
        "minute_semantics": "Opening state is separate. Research minute row at step_end=n contains powers applied over physics interval [n-1,n), followed by inventories after production and any delivery due at n. Availability solar and scheduled DAC health are exogenous and controller independent. Other flows and inventories describe the saved MPC realization only.",
        "health_checks": "All research rows exactly match manifest half-open DAC outage schedules using interval start seconds since episode opening. All other health remains nominal with zero degradation.",
        "scenarios": [read_research(seed, root) for seed in range(9700300, 9700308)]
        + [read_demo()],
    }
    for scenario in out["scenarios"]:
        cumulative_solar = 0.0
        opening = scenario["initial"]
        expected_targets = {
            "battery": opening["battery_kwh"],
            "h2": opening["h2_kg"],
            "co2": opening["co2_kg"],
            "ch4": opening["ch4_kg"],
        }
        for day in scenario["days"]:
            cumulative_solar += day["solar_available_kwh"]
            day["cumulative_solar_available_kwh"] = cumulative_solar
            day["cumulative_requested_kg"] = 960.0 * day["cycle_index"]
            assert day["settlement"]["targets"] == expected_targets
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--research-evaluations", type=Path, default=ROOT)
    args = parser.parse_args()
    out = build_output(args.research_evaluations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, separators=(",", ":"), allow_nan=False))
    print(
        json.dumps(
            {"output": pin(args.output), "scenario_count": len(out["scenarios"])},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Run a separately versioned, delivery-only physical feasibility experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from docs.research.tools.feasible_mission_replay import (
    ALLOWED_SEEDS,
    canonical_sha,
    file_pin,
    make_environment,
    run_controller,
)
from docs.research.tools.feasible_mission_controller import DailyCoverageController
from plant_backend.rl.v2.setpoint_control import ProcessSetpointTargetsV1


class ReplayTargets:
    """Playback verification of a recorded schedule, not an online controller."""

    id = "feasible-mission-target-replay-v1"

    def __init__(self, commands):
        self.commands = commands
        self.index = 0

    def reset(self):
        self.index = 0

    def compute_command(self, observation, **kwargs):
        saved = self.commands[self.index]["command"]["targets"]["setpoints_kw"]
        self.index += 1
        return ProcessSetpointTargetsV1(
            saved["electrolyser"],
            saved["dac"],
            saved["methanator"],
            saved["battery_charge_ceiling"],
        )


def compact(result, path):
    omitted = {"commands", "decision_frames", "minute_truth_states", "manifest"}
    return {
        **{k: v for k, v in result.items() if k not in omitted},
        "recording": file_pin(path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=25.0)
    parser.add_argument(
        "--controller", choices=("coverage", "mpc", "ppo"), default="coverage"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ALLOWED_SEEDS))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/research/results/feasible_mission_20260915",
    )
    parser.add_argument("--verify-replay", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.controller == "ppo":
        from docs.research.tools.feasible_mission_comparators import (
            load_gamma_one_policy,
        )

        loaded = load_gamma_one_policy()
    report = {
        "schema": "delivery-only-feasible-mission-study-v1",
        "mission_id": f"delivery-{args.rate:g}kg-h-physical-baseline-v1",
        "controller_kind": args.controller,
        "delivery_rate_kg_h": args.rate,
        "requested_weekly_kg": args.rate * 168,
        "interpretation": "Development feasibility/mission-transfer experiment; no held-out qualification or maximum feasible rate claim.",
        "change": "Only delivery amounts changed; source equipment, initial inventory, weather, forecasts, market, faults, operator limits and full daily stock-return requirements preserved.",
        "source_files": [
            file_pin(Path(__file__)),
            file_pin(Path(__file__).with_name("feasible_mission_controller.py")),
            file_pin(Path(__file__).with_name("feasible_mission_replay.py")),
        ],
        "runs": [],
    }
    summary = args.output / f"{args.controller}-{args.rate:g}-summary.json"
    for seed in args.seeds:
        start = time.monotonic()
        env = make_environment(seed, args.rate)
        if args.controller == "coverage":
            controller = DailyCoverageController()
        elif args.controller == "mpc":
            from docs.research.tools.feasible_mission_comparators import (
                make_operating_mpc,
            )

            controller = make_operating_mpc(env)
        else:
            from docs.research.tools.feasible_mission_comparators import (
                MPCResidualTransferController,
            )

            controller = MPCResidualTransferController(env, loaded)
        path = args.output / f"{args.controller}-{args.rate:g}-{seed}.json.gz"
        result = run_controller(env, controller, output_path=path)
        entry = compact(result, path)
        entry["controller_descriptor"] = (
            controller.to_dict()
            if hasattr(controller, "to_dict")
            else {"id": result["controller"]}
        )
        if args.verify_replay and result["execution_complete"]:
            replay_env = make_environment(seed, args.rate)
            playback = run_controller(
                replay_env, ReplayTargets(result["commands"]), retain_minutes=False
            )
            entry["independent_target_replay"] = {
                "minute_truth_sha256": playback["minute_truth_sha256"],
                "exact_minute_state_match": playback["minute_truth_sha256"]
                == result["minute_truth_sha256"],
                "mission_passed": playback["mission_passed"],
                "execution_complete": playback["execution_complete"],
                "failure": playback["failure"],
            }
            replay_env.close()
            if not entry["independent_target_replay"]["exact_minute_state_match"]:
                raise RuntimeError(
                    f"Recorded targets did not reproduce minute states for seed {seed}"
                )
        env.close()
        entry["wall_time_s"] = time.monotonic() - start
        report["runs"].append(entry)
        report["all_requested_runs_completed"] = len(report["runs"]) == len(args.seeds)
        report["mission_pass_count"] = sum(x["mission_passed"] for x in report["runs"])
        summary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "controller": args.controller,
                    "seed": seed,
                    "rate": args.rate,
                    "mission_passed": result["mission_passed"],
                    "deliveries": result["physical_metrics"]["commitments_met"],
                    "cycles": sum(x["feasible"] for x in result["cycle_settlements"]),
                    "safety": result["hard_safety_passed"],
                    "failure": result["failure"],
                    "seconds": round(entry["wall_time_s"], 1),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

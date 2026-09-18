"""Exact-minute benchmark runner for explicitly labelled MPC/coverage recovery."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
for folder in ("", "backend/src", "simulator/src"):
    if str(ROOT / folder) not in sys.path:
        sys.path.insert(0, str(ROOT / folder))

from docs.research.tools.feasible_benchmark_v1 import BenchmarkEnv
from docs.research.tools.feasible_mission_replay import file_pin, run_controller
from docs.research.tools.mission_mpc_recovery_v1 import MissionMPCRecoveryControllerV1


def run(seed, *, split, output, max_decisions=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    paths = [
        ROOT / name
        for name in (
            "backend/src/plant_backend/mission_mpc.py",
            "docs/research/tools/mission_mpc_recovery_v1.py",
            "docs/research/tools/run_mission_mpc_recovery_v1.py",
            "docs/research/tools/feasible_benchmark_v1.py",
            "docs/research/tools/offgrid_coverage_v1.py",
            "docs/research/tools/feasible_mission_controller.py",
        )
    ]
    pins = [file_pin(path) for path in paths]
    env = BenchmarkEnv(split=split)
    started = time.monotonic()
    try:
        env.reset(seed=seed)
        controller = MissionMPCRecoveryControllerV1(
            horizon_h=env.episode_config.horizon_h,
            interval_s=env.episode_config.decision_interval_s,
            assumptions=env.economic_assumptions,
            beta_task=env.beta_task,
            electricity_model=env.electricity_model.to_dict(),
        )

        class Logged:
            id = controller.id

            def compute_command(self, observation, **kwargs):
                selected = controller.compute_command(observation, **kwargs)
                if controller.mpc._next_decision % 96 == 0:
                    print(
                        json.dumps(
                            {
                                "seed": seed,
                                "decisions": controller.mpc._next_decision,
                                "wall_s": round(time.monotonic() - started, 2),
                                "modes": dict(Counter(x["decision_mode"] for x in controller.decision_events)),
                            }
                        ),
                        flush=True,
                    )
                return selected

        result = run_controller(env, Logged(), max_decisions=max_decisions)
        result.update(
            schema="mission-mpc-coverage-recovery-recording-v1",
            split=split,
            benchmark_contract_sha256=env.benchmark_contract["contract_sha256"],
            electricity_model=env.electricity_model.to_dict(),
            change="25 kg/h mission, original physical scenarios and daily returns, explicit effective electricity model, and labelled coverage recovery after hard-MPC rejection.",
            controller_configuration=controller.to_dict(),
            decision_modes=dict(Counter(x["decision_mode"] for x in controller.decision_events)),
            additional_source_code=pins,
            source_files_unchanged_during_run=(pins == [file_pin(path) for path in paths]),
            wall_s=time.monotonic() - started,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(result, separators=(",", ":"), allow_nan=False).encode()
        with output.open("xb") as stream:
            stream.write(gzip.compress(raw, mtime=0) if output.suffix == ".gz" else raw)
        print(
            json.dumps({key: result[key] for key in ("seed", "failure", "execution_complete", "mission_passed", "hard_safety_passed", "every_delivery_passed", "all_daily_returns_passed", "decision_modes", "source_files_unchanged_during_run", "wall_s")}),
            flush=True,
        )
        return result
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", default="development", choices=("development", "held_out"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-decisions", type=int)
    options = parser.parse_args()
    run(options.seed, split=options.split, output=options.output, max_decisions=options.max_decisions)

"""Run the versioned hard-return MPC on an explicitly selected benchmark split."""

from __future__ import annotations

import argparse
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
from plant_backend.mission_mpc import MissionMPCControllerV1
from plant_backend.rl.v2.economic_mpc_schema import (
    EconomicMPCPlanError,
    EconomicMPCSolveError,
)


def run(seed: int, *, split: str, output: Path, max_decisions: int | None = None):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite recording {output}")
    env = BenchmarkEnv(split=split)
    env.reset(seed=seed)
    controller = MissionMPCControllerV1(
        horizon_h=env.episode_config.horizon_h,
        interval_s=env.episode_config.decision_interval_s,
        assumptions=env.economic_assumptions,
        beta_task=env.beta_task,
        electricity_model=env.electricity_model.to_dict(),
    )
    source_paths = (
        ROOT / "backend/src/plant_backend/mission_mpc.py",
        ROOT / "docs/research/tools/feasible_benchmark_v1.py",
        Path(__file__).resolve(),
    )
    source_pins = [file_pin(path) for path in source_paths]
    started = time.monotonic()
    rejected_plans = []

    class LoggedController:
        id = controller.id

        def to_dict(self):
            return controller.to_dict()

        def compute_command(self, observation, **kwargs):
            try:
                decision = controller.compute_command(observation, **kwargs)
            except (EconomicMPCSolveError, EconomicMPCPlanError) as exc:
                rejected_plans.append(
                    {
                        "decision": observation.market.issued_at_decision,
                        "observation_sha256": observation.sha256,
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "stages": [
                            stage.to_dict() for stage in getattr(exc, "stages", ())
                        ],
                        "audited_hard_return_plan": False,
                        "fallback": None,
                    }
                )
                raise
            if controller._next_decision % 24 == 0:
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "decisions": controller._next_decision,
                            "wall_s": round(time.monotonic() - started, 3),
                        }
                    ),
                    flush=True,
                )
            return decision

    try:
        result = run_controller(env, LoggedController(), max_decisions=max_decisions)
        result.update(
            schema="mission-mpc-development-recording-v1",
            split=split,
            benchmark_contract_sha256=env.benchmark_contract["contract_sha256"],
            electricity_model=env.electricity_model.to_dict(),
            change=(
                "25 kg/h delivery mission with original sampled physical "
                "inventories, weather and faults; explicit effective electricity "
                "prices; original daily returns and safety constraints."
            ),
            wall_s=time.monotonic() - started,
            additional_source_code=source_pins,
            controller_configuration=controller.to_dict(),
            rejected_plans=rejected_plans,
            source_files_unchanged_during_run=(
                source_pins == [file_pin(path) for path in source_paths]
            ),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(result, separators=(",", ":"), allow_nan=False).encode()
        with output.open("xb") as destination:
            destination.write(
                gzip.compress(raw, mtime=0) if output.suffix == ".gz" else raw
            )
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "failure",
                        "decisions",
                        "execution_complete",
                        "hard_safety_passed",
                        "every_delivery_passed",
                        "all_daily_returns_passed",
                        "mission_passed",
                        "wall_s",
                    )
                }
            ),
            flush=True,
        )
        return result
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--split",
        choices=("development", "legacy_development", "held_out"),
        default="development",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-decisions", type=int)
    args = parser.parse_args()
    run(args.seed, split=args.split, output=args.output, max_decisions=args.max_decisions)

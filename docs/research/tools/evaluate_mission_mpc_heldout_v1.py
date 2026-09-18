"""Run the unchanged strict MPC only after the joint comparison freeze passes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
for folder in ("", "backend/src", "simulator/src"):
    if str(ROOT / folder) not in sys.path:
        sys.path.insert(0, str(ROOT / folder))

from docs.research.tools.evaluate_feasible25_v1 import require_evaluation_freeze
from docs.research.tools.feasible_mission_replay import file_pin


def verified_joint_freeze():
    """No environment construction or reset occurs before this gate returns."""
    freeze = require_evaluation_freeze()
    if freeze["schema"] != "feasible25-comparison-freeze-v1":
        raise ValueError("Unexpected joint evaluation freeze schema")
    own_pin = file_pin(Path(__file__).resolve())
    if own_pin not in freeze["sources"]:
        raise ValueError("Joint evaluation freeze does not bind this launcher")
    return freeze


def _invoke_strict_run(seed, output):
    # The imported runner constructs its environment inside run(), after the
    # parent and child gates. Its existing controller, limits and source checks
    # are unchanged.
    from docs.research.tools.run_mission_mpc_v1 import run

    return run(seed, split="held_out", output=output)


def run_case(seed, expected_freeze_sha256, output):
    freeze = verified_joint_freeze()
    if freeze["sha256"] != expected_freeze_sha256:
        raise ValueError("Joint freeze changed before child evaluation")
    if seed not in freeze["held_out_seeds"]:
        raise ValueError("Seed is outside the frozen held-out set")
    output = Path(output)
    recording = output / f"{seed}.json.gz"
    with (output / f"{seed}.log").open("x") as log, redirect_stdout(log):
        result = _invoke_strict_run(seed, recording)
    after = verified_joint_freeze()
    if after["sha256"] != expected_freeze_sha256:
        raise ValueError("Joint freeze changed during child evaluation")
    row = {
        "seed": seed,
        "evaluation_freeze_sha256": expected_freeze_sha256,
        "recording": file_pin(recording),
        "failure": result["failure"],
        "execution_complete": result["execution_complete"],
        "mission_passed": result["mission_passed"],
        "hard_safety_passed": result["hard_safety_passed"],
        "every_delivery_passed": result["every_delivery_passed"],
        "all_daily_returns_passed": result["all_daily_returns_passed"],
        "decisions": result["decisions"],
        "deliveries_met": result["physical_metrics"]["commitments_met"],
        "delivered_kg": result["physical_metrics"]["delivered_kg"],
        "unmet_kg": result["physical_metrics"]["unmet_kg"],
        "daily_returns_met": sum(x["feasible"] for x in result["cycle_settlements"]),
        "daily_returns_observed": len(result["cycle_settlements"]),
        "operating_contribution_eur": result["economic_metrics"]["net_operating_value_eur"],
        "rejected_plans": result["rejected_plans"],
        "source_files_unchanged_during_run": result["source_files_unchanged_during_run"],
        "wall_s": result["wall_s"],
    }
    with (output / f"{seed}-provenance.json").open("x") as stream:
        json.dump(row, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return row


def run_panel(output, *, workers=4):
    freeze = verified_joint_freeze()
    seeds = freeze["held_out_seeds"]
    if len(seeds) != 16 or len(set(seeds)) != 16:
        raise ValueError("The final strict-MPC panel requires all 16 frozen cases")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError("Worker count must be an integer from 1 through 8")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to reuse held-out output directory {output}")
    output.mkdir(parents=True)
    started = time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(run_case, seed, freeze["sha256"], output): seed
            for seed in seeds
        }
        for future in as_completed(pending):
            seed = pending[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "seed": seed,
                    "evaluation_freeze_sha256": freeze["sha256"],
                    "runner_failure": {"type": type(exc).__name__, "message": str(exc)},
                    "execution_complete": False,
                    "mission_passed": False,
                }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "event": "held_out_case_complete",
                        "seed": seed,
                        "execution_complete": row["execution_complete"],
                        "mission_passed": row["mission_passed"],
                        "decisions": row.get("decisions"),
                        "failure": row.get("failure", row.get("runner_failure")),
                    }
                ),
                flush=True,
            )
    after = verified_joint_freeze()
    if after["sha256"] != freeze["sha256"]:
        raise ValueError("Joint freeze changed during the final panel")
    summary = {
        "schema": "strict-mission-mpc-held-out-panel-v1",
        "evaluation_freeze_sha256": freeze["sha256"],
        "launcher": file_pin(Path(__file__).resolve()),
        "split": "held_out",
        "seeds": seeds,
        "workers": workers,
        "runtime_context": (
            f"{workers} independent MPC processes on the shared local host; "
            "five-second solver budget per stage, unchanged; no retries or tuning"
        ),
        "wall_s": time.monotonic() - started,
        "runs": sorted(rows, key=lambda row: row["seed"]),
    }
    with (output / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "event": "held_out_panel_complete",
                "cases": len(rows),
                "full_mission_passes": sum(row["mission_passed"] for row in rows),
                "wall_s": summary["wall_s"],
            }
        ),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    run_panel(arguments.output, workers=arguments.workers)
